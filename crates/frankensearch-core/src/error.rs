use std::path::PathBuf;

/// Unified error type covering all failure modes across the frankensearch search pipeline.
///
/// Every variant includes an actionable error message guiding the consumer toward resolution.
/// The `TwoTierSearcher` catches transient errors and degrades gracefully: `EmbeddingFailed`
/// falls back to hash embedding, `RerankFailed` skips reranking, `SearchTimeout` yields
/// initial results. Only `IndexNotFound` and `InvalidConfig` prevent search from starting.
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
        "Embedding failed for {model}: {source}. Transient error; retry or fall back to hash embedder."
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
    #[error("Reranking failed: {source}. Results still valid with original RRF scores.")]
    RerankFailed {
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

/// Convenience alias used throughout the frankensearch crate hierarchy.
pub type SearchResult<T> = Result<T, SearchError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SearchError>();
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

        let err = SearchError::DimensionMismatch {
            expected: 256,
            found: 384,
        };
        let msg = err.to_string();
        assert!(msg.contains("256"));
        assert!(msg.contains("384"));
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
}
