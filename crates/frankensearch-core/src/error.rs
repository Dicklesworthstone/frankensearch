use std::path::PathBuf;

/// Unified error type covering all failure modes across the frankensearch search pipeline.
///
/// Every variant includes an actionable error message guiding the consumer toward resolution.
/// The `TwoTierSearcher` catches transient errors and degrades gracefully: fast embedding
/// failures can still yield lexical-only initial results when lexical retrieval is available,
/// `RerankFailed` skips reranking, and `SearchTimeout` yields initial results.
/// Index/configuration failures and unverifiable embedding-space identity can prevent a
/// semantic search from starting. Callers must only degrade when the selected public API
/// explicitly preserves a verified lexical result path.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    // === Embedding errors ===
    /// An embedding model is not available (not compiled in, or model files missing).
    #[error(
        "Embedder unavailable: {model} — {reason}. Set FRANKENSEARCH_MODEL_DIR or enable the corresponding feature flag."
    )]
    EmbedderUnavailable {
        /// Identifier of the unavailable model.
        model: String,
        /// Why it is unavailable.
        reason: String,
    },

    /// Embedding inference failed for a given model.
    #[error(
        "Embedding failed for {model}: {source}. Transient error; retry or use lexical fallback when configured."
    )]
    EmbeddingFailed {
        /// Which model failed.
        model: String,
        /// The underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Model files were not found at any searched path.
    #[error("Model {name} not found. Run download or set FRANKENSEARCH_MODEL_DIR.")]
    ModelNotFound {
        /// Model identifier.
        name: String,
    },

    /// Model files exist but failed to load (corrupted, incompatible version, etc.).
    #[error("Failed to load model from {path}: {source}")]
    ModelLoadFailed {
        /// Path that was attempted.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    // === Index errors ===
    /// The vector index file is corrupted (bad magic, CRC mismatch, truncated).
    #[error(
        "Vector index corrupted at {path}: {detail}. Delete and rebuild with index_documents()."
    )]
    IndexCorrupted {
        /// Path to the corrupted file.
        path: PathBuf,
        /// Nature of the corruption.
        detail: String,
    },

    /// The FSVI file version does not match what this build expects.
    #[error(
        "Index version mismatch at index: expected v{expected}, found v{found}. Rebuild the index."
    )]
    IndexVersionMismatch {
        /// The version this library expects.
        expected: u16,
        /// The version found in the file.
        found: u16,
    },

    /// Query vector dimension does not match the index dimension.
    #[error(
        "Dimension mismatch: index has {expected}-dim vectors, query has {found}-dim. Use matching embedder."
    )]
    DimensionMismatch {
        /// Dimension the index was built with.
        expected: usize,
        /// Dimension of the query vector.
        found: usize,
    },

    /// No vector index file exists at the expected path.
    #[error(
        "Vector index not found at {path}. Run index_documents() first, or check FRANKENSEARCH_DATA_DIR."
    )]
    IndexNotFound {
        /// Expected path.
        path: PathBuf,
    },

    /// No vector index file exists at any of the candidate paths.
    #[error(
        "Vector index not found at any candidate path: {paths:?}. Run index_documents() first, or check FRANKENSEARCH_DATA_DIR."
    )]
    IndexCandidatesNotFound {
        /// Candidate paths, in lookup order.
        paths: Vec<PathBuf>,
    },

    // === Search errors ===
    /// The query string could not be parsed.
    #[error("Query parse error for \"{query}\": {detail}")]
    QueryParseError {
        /// The problematic query.
        query: String,
        /// What went wrong.
        detail: String,
    },

    /// A search phase exceeded its time budget.
    #[error(
        "Search timed out after {elapsed_ms}ms (budget: {budget_ms}ms). Increase timeout in TwoTierConfig."
    )]
    SearchTimeout {
        /// How long the operation ran.
        elapsed_ms: u64,
        /// The configured budget.
        budget_ms: u64,
    },

    /// Federated search did not receive enough successful shard responses.
    #[error(
        "Federated search required at least {required} index responses, but only {received} succeeded."
    )]
    FederatedInsufficientResponses {
        /// Minimum responses required by config.
        required: usize,
        /// Number of successful shard responses observed.
        received: usize,
    },

    // === Reranker errors ===
    /// The reranking model is not available.
    #[error(
        "Reranker unavailable: {model}. Results are valid without reranking; enable 'rerank' feature."
    )]
    RerankerUnavailable {
        /// Model identifier.
        model: String,
    },

    /// Reranking inference failed.
    #[error(
        "Reranking failed for {model}: {source}. Results still valid with original RRF scores."
    )]
    RerankFailed {
        /// Which reranker model failed.
        model: String,
        /// The underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    // === I/O errors ===
    /// Wraps `std::io::Error` for file operations.
    #[error("I/O error: {0}. Check file permissions and disk space.")]
    Io(#[from] std::io::Error),

    // === Configuration errors ===
    /// A configuration value is invalid.
    #[error("Invalid config: {field} = \"{value}\" — {reason}")]
    InvalidConfig {
        /// Which config field.
        field: String,
        /// The invalid value.
        value: String,
        /// Why it is invalid.
        reason: String,
    },

    // === Hash verification ===
    /// Downloaded or loaded file does not match expected hash.
    #[error("Hash mismatch for {path}: expected {expected}, got {actual}. File may be corrupted.")]
    HashMismatch {
        /// Path to the file.
        path: PathBuf,
        /// Expected hash (hex string).
        expected: String,
        /// Actual computed hash.
        actual: String,
    },

    /// A producer cannot prove the canonical identity of its embedding space.
    #[error(
        "Embedding space identity is unverifiable. Refuse semantic comparison; require a producer-authenticated immutable identity, or use a verified local embedder or lexical fallback."
    )]
    UnverifiableRemoteSpace {
        /// Component or remote producer that supplied the identity claim.
        ///
        /// This field is intentionally omitted from [`Display`](std::fmt::Display). Public
        /// serializers must validate and bound it before exposure.
        producer: String,
        /// Why the supplied identity cannot be verified.
        ///
        /// This field is intentionally omitted from [`Display`](std::fmt::Display). It may be
        /// retained for bounded, redacted structured diagnostics.
        reason: String,
    },

    // === Cancellation ===
    /// Operation was cancelled via the asupersync structured concurrency protocol.
    #[error("Operation cancelled during {phase}: {reason}")]
    Cancelled {
        /// Which phase was active when cancelled.
        phase: String,
        /// Cancellation reason.
        reason: String,
    },

    // === Queue errors ===
    /// The embedding job queue is full.
    #[error(
        "Embedding queue full ({pending}/{capacity} pending). Apply backpressure or increase capacity."
    )]
    QueueFull {
        /// Number of pending items.
        pending: usize,
        /// Queue capacity.
        capacity: usize,
    },

    // === Subsystem errors ===
    /// Wraps errors from optional subsystems (storage, durability, FTS5, etc.).
    ///
    /// Always present in the enum regardless of feature flags, avoiding
    /// match-arm breakage across feature combinations.
    #[error("{subsystem} error: {source}")]
    SubsystemError {
        /// Which subsystem produced the error (e.g., "storage", "durability", "fts5").
        subsystem: &'static str,
        /// The underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A durability/repair feature was requested but is not compiled in.
    #[error(
        "Durability feature is not enabled. Enable the 'durability' Cargo feature for self-healing indices."
    )]
    DurabilityDisabled,
}

impl SearchError {
    /// A stable, bounded diagnostic code for logs, metrics, and public telemetry.
    ///
    /// Error fields and source chains can contain queries, document content,
    /// filesystem paths, provider credentials, or arbitrary control characters.
    /// This projection inspects only the variant and never formats those values.
    /// Keep the original error for callers that need its structured details;
    /// use this code when publishing a diagnostic outside that call boundary.
    #[must_use]
    pub const fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::EmbedderUnavailable { .. } => "embedder_unavailable",
            Self::EmbeddingFailed { .. } => "embedding_failed",
            Self::ModelNotFound { .. } => "model_not_found",
            Self::ModelLoadFailed { .. } => "model_load_failed",
            Self::IndexCorrupted { .. } => "index_corrupted",
            Self::IndexVersionMismatch { .. } => "index_version_mismatch",
            Self::DimensionMismatch { .. } => "dimension_mismatch",
            Self::IndexNotFound { .. } => "index_not_found",
            Self::IndexCandidatesNotFound { .. } => "index_candidates_not_found",
            Self::QueryParseError { .. } => "query_parse_error",
            Self::SearchTimeout { .. } => "search_timeout",
            Self::FederatedInsufficientResponses { .. } => "federated_insufficient_responses",
            Self::RerankerUnavailable { .. } => "reranker_unavailable",
            Self::RerankFailed { .. } => "rerank_failed",
            Self::Io(_) => "io_error",
            Self::InvalidConfig { .. } => "invalid_config",
            Self::HashMismatch { .. } => "hash_mismatch",
            Self::UnverifiableRemoteSpace { .. } => "unverifiable_remote_space",
            Self::Cancelled { .. } => "cancelled",
            Self::QueueFull { .. } => "queue_full",
            Self::SubsystemError { .. } => "subsystem_error",
            Self::DurabilityDisabled => "durability_disabled",
        }
    }
}

/// Convenience alias used throughout the frankensearch crate hierarchy.
pub type SearchResult<T> = Result<T, SearchError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SearchError>();
    }

    #[test]
    fn diagnostics_never_format_untrusted_fields_or_sources() {
        #[derive(Debug)]
        struct UntrustedSource;

        impl std::fmt::Display for UntrustedSource {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("public diagnostics must not format a provider error");
            }
        }

        impl std::error::Error for UntrustedSource {}

        const PRIVATE: &str = "private query\nhttps://user:secret@provider/private/path\x1b[31m";
        let errors = [
            SearchError::EmbedderUnavailable {
                model: PRIVATE.into(),
                reason: PRIVATE.into(),
            },
            SearchError::EmbeddingFailed {
                model: PRIVATE.into(),
                source: Box::new(UntrustedSource),
            },
            SearchError::ModelNotFound {
                name: PRIVATE.into(),
            },
            SearchError::ModelLoadFailed {
                path: PRIVATE.into(),
                source: Box::new(UntrustedSource),
            },
            SearchError::IndexCorrupted {
                path: PRIVATE.into(),
                detail: PRIVATE.into(),
            },
            SearchError::IndexVersionMismatch {
                expected: u16::MAX,
                found: u16::MAX,
            },
            SearchError::DimensionMismatch {
                expected: usize::MAX,
                found: usize::MAX,
            },
            SearchError::IndexNotFound {
                path: PRIVATE.into(),
            },
            SearchError::IndexCandidatesNotFound {
                paths: vec![PRIVATE.into()],
            },
            SearchError::QueryParseError {
                query: PRIVATE.into(),
                detail: PRIVATE.into(),
            },
            SearchError::SearchTimeout {
                elapsed_ms: u64::MAX,
                budget_ms: u64::MAX,
            },
            SearchError::FederatedInsufficientResponses {
                required: usize::MAX,
                received: usize::MAX,
            },
            SearchError::RerankerUnavailable {
                model: PRIVATE.into(),
            },
            SearchError::RerankFailed {
                model: PRIVATE.into(),
                source: Box::new(UntrustedSource),
            },
            SearchError::Io(std::io::Error::other(UntrustedSource)),
            SearchError::InvalidConfig {
                field: PRIVATE.into(),
                value: PRIVATE.into(),
                reason: PRIVATE.into(),
            },
            SearchError::HashMismatch {
                path: PRIVATE.into(),
                expected: PRIVATE.into(),
                actual: PRIVATE.into(),
            },
            SearchError::UnverifiableRemoteSpace {
                producer: PRIVATE.into(),
                reason: PRIVATE.into(),
            },
            SearchError::Cancelled {
                phase: PRIVATE.into(),
                reason: PRIVATE.into(),
            },
            SearchError::QueueFull {
                pending: usize::MAX,
                capacity: usize::MAX,
            },
            SearchError::SubsystemError {
                subsystem: PRIVATE,
                source: Box::new(UntrustedSource),
            },
            SearchError::DurabilityDisabled,
        ];
        let mut codes = std::collections::BTreeSet::new();
        for error in &errors {
            let code = error.diagnostic_code();
            assert!(!code.is_empty() && code.len() <= 40);
            assert!(
                code.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            );
            assert!(codes.insert(code), "distinct errors need distinct codes");
            assert!(!code.contains("private"));
            assert!(!code.contains("secret"));
        }
    }

    #[test]
    fn io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let search_err: SearchError = io_err.into();
        assert!(matches!(search_err, SearchError::Io(_)));
        assert!(search_err.to_string().contains("gone"));
    }

    #[test]
    fn display_messages_are_actionable() {
        let err = SearchError::IndexNotFound {
            path: PathBuf::from("/tmp/missing.fsvi"),
        };
        let msg = err.to_string();
        assert!(msg.contains("index_documents()"), "should suggest recovery");

        let err = SearchError::IndexCandidatesNotFound {
            paths: vec![
                PathBuf::from("/tmp/vector.fast.idx"),
                PathBuf::from("/tmp/vector.idx"),
            ],
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/vector.fast.idx"));
        assert!(msg.contains("/tmp/vector.idx"));

        let err = SearchError::UnverifiableRemoteSpace {
            producer: "https://user:secret@embedding.example/v1".to_owned(),
            reason: "response contained private query text".to_owned(),
        };
        let msg = err.to_string();
        assert!(!msg.contains("secret"));
        assert!(!msg.contains("private query text"));
        assert!(msg.contains("Refuse semantic comparison"));
        assert!(msg.contains("producer-authenticated immutable identity"));

        let err = SearchError::DimensionMismatch {
            expected: 256,
            found: 384,
        };
        let msg = err.to_string();
        assert!(msg.contains("256"));
        assert!(msg.contains("384"));
    }

    #[test]
    fn federated_insufficient_responses_message_has_counts() {
        let err = SearchError::FederatedInsufficientResponses {
            required: 2,
            received: 1,
        };
        let msg = err.to_string();
        assert!(msg.contains('2'));
        assert!(msg.contains('1'));
    }

    #[test]
    fn subsystem_error_wraps_arbitrary_errors() {
        let inner = std::io::Error::other("db locked");
        let err = SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(inner),
        };
        assert!(err.to_string().contains("storage"));
        assert!(err.to_string().contains("db locked"));
    }

    #[test]
    fn search_result_alias_works() {
        // Verify the type alias compiles and works with both Ok and Err variants.
        let ok: SearchResult<u32> = Ok(42);
        assert!(ok.is_ok());

        let err: SearchResult<u32> = Err(SearchError::DurabilityDisabled);
        assert!(err.is_err());
    }

    #[test]
    fn embedding_failed_preserves_source() {
        let inner = std::io::Error::other("onnx crash");
        let err = SearchError::EmbeddingFailed {
            model: "MiniLM".into(),
            source: Box::new(inner),
        };
        assert!(err.to_string().contains("MiniLM"));
        assert!(err.to_string().contains("onnx crash"));
    }

    #[test]
    fn cancelled_variant() {
        let err = SearchError::Cancelled {
            phase: "quality_embed".into(),
            reason: "parent scope dropped".into(),
        };
        assert!(err.to_string().contains("quality_embed"));
        assert!(err.to_string().contains("parent scope dropped"));
    }

    #[test]
    fn embedder_unavailable_display() {
        let err = SearchError::EmbedderUnavailable {
            model: "MiniLM".into(),
            reason: "feature not enabled".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("MiniLM"));
        assert!(msg.contains("feature not enabled"));
    }

    #[test]
    fn model_not_found_display() {
        let err = SearchError::ModelNotFound {
            name: "all-MiniLM-L6-v2".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("all-MiniLM-L6-v2"));
        assert!(msg.contains("FRANKENSEARCH_MODEL_DIR"));
    }

    #[test]
    fn model_load_failed_preserves_source() {
        let inner = std::io::Error::other("mmap failed");
        let err = SearchError::ModelLoadFailed {
            path: PathBuf::from("/models/broken.onnx"),
            source: Box::new(inner),
        };
        let msg = err.to_string();
        assert!(msg.contains("/models/broken.onnx"));
        assert!(msg.contains("mmap failed"));
        // Verify source chain via std::error::Error trait
        assert!(err.source().is_some());
    }

    #[test]
    fn index_corrupted_display() {
        let err = SearchError::IndexCorrupted {
            path: PathBuf::from("/data/index.fsvi"),
            detail: "CRC mismatch in header".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/data/index.fsvi"));
        assert!(msg.contains("CRC mismatch"));
        assert!(msg.contains("rebuild"));
    }

    #[test]
    fn index_version_mismatch_display() {
        let err = SearchError::IndexVersionMismatch {
            expected: 3,
            found: 1,
        };
        let msg = err.to_string();
        assert!(msg.contains("v3"));
        assert!(msg.contains("v1"));
        assert!(msg.contains("Rebuild"));
    }

    #[test]
    fn query_parse_error_display() {
        let err = SearchError::QueryParseError {
            query: "foo AND OR bar".into(),
            detail: "unexpected OR after AND".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("foo AND OR bar"));
        assert!(msg.contains("unexpected OR after AND"));
    }

    #[test]
    fn search_timeout_display() {
        let err = SearchError::SearchTimeout {
            elapsed_ms: 750,
            budget_ms: 500,
        };
        let msg = err.to_string();
        assert!(msg.contains("750"));
        assert!(msg.contains("500"));
    }

    #[test]
    fn reranker_unavailable_display() {
        let err = SearchError::RerankerUnavailable {
            model: "cross-encoder".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("cross-encoder"));
        assert!(msg.contains("rerank"));
    }

    #[test]
    fn rerank_failed_preserves_source() {
        let inner = std::io::Error::other("inference oom");
        let err = SearchError::RerankFailed {
            model: "cross-encoder".into(),
            source: Box::new(inner),
        };
        let msg = err.to_string();
        assert!(msg.contains("cross-encoder"));
        assert!(msg.contains("inference oom"));
        assert!(err.source().is_some());
    }

    #[test]
    fn invalid_config_display() {
        let err = SearchError::InvalidConfig {
            field: "quality_weight".into(),
            value: "-1.0".into(),
            reason: "must be between 0.0 and 1.0".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("quality_weight"));
        assert!(msg.contains("-1.0"));
        assert!(msg.contains("must be between"));
    }

    #[test]
    fn hash_mismatch_display() {
        let err = SearchError::HashMismatch {
            path: PathBuf::from("/tmp/model.bin"),
            expected: "abc123".into(),
            actual: "def456".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("/tmp/model.bin"));
        assert!(msg.contains("abc123"));
        assert!(msg.contains("def456"));
    }

    #[test]
    fn queue_full_display() {
        let err = SearchError::QueueFull {
            pending: 100,
            capacity: 100,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("backpressure"));
    }

    #[test]
    fn durability_disabled_display() {
        let err = SearchError::DurabilityDisabled;
        let msg = err.to_string();
        assert!(msg.contains("durability"));
    }

    #[test]
    fn error_debug_format() {
        let err = SearchError::DimensionMismatch {
            expected: 128,
            found: 256,
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("DimensionMismatch"));
        assert!(debug.contains("128"));
        assert!(debug.contains("256"));
    }
}
