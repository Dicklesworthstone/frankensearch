//! Optional tracing subscriber setup for frankensearch.
//!
//! This module provides a convenience function for consumers who want structured
//! logging without configuring `tracing-subscriber` themselves. It is entirely
//! optional: consumers may bring their own subscriber.
//!
//! # Usage
//!
//! ```ignore
//! use frankensearch_core::tracing_config::init_tracing;
//! use tracing::Level;
//!
//! let _guard = init_tracing(Level::INFO);
//! // All frankensearch spans and events are now captured.
//! ```

use tracing::Level;

/// Target prefix used by all frankensearch tracing spans and events.
///
/// Consumers can use this to filter frankensearch logs:
/// ```text
/// RUST_LOG=frankensearch=debug
/// ```
pub const TARGET_PREFIX: &str = "frankensearch";

/// Standard tracing span names used across the pipeline.
///
/// These constants ensure consistent span naming so that consumers can
/// match on them in subscribers, dashboards, and tests.
pub mod span_names {
    /// Root span for a search query.
    pub const SEARCH: &str = "frankensearch::search";
    /// Fast-tier embedding.
    pub const FAST_EMBED: &str = "frankensearch::fast_embed";
    /// Fast-tier vector search.
    pub const FAST_SEARCH: &str = "frankensearch::fast_search";
    /// Lexical (BM25) search.
    pub const LEXICAL_SEARCH: &str = "frankensearch::lexical_search";
    /// RRF fusion step.
    pub const RRF_FUSE: &str = "frankensearch::rrf_fuse";
    /// Quality-tier embedding.
    pub const QUALITY_EMBED: &str = "frankensearch::quality_embed";
    /// Fast + quality score blending.
    pub const BLEND: &str = "frankensearch::blend";
    /// Cross-encoder reranking.
    pub const RERANK: &str = "frankensearch::rerank";
    /// Index rebuild.
    pub const INDEX_REBUILD: &str = "frankensearch::index_rebuild";
    /// Refresh worker cycle.
    pub const REFRESH_CYCLE: &str = "frankensearch::refresh_cycle";
    /// Embedding batch processing.
    pub const EMBED_BATCH: &str = "frankensearch::embed_batch";
}

/// Standard structured field names used in tracing events.
///
/// Using consistent field names enables structured log queries across
/// the entire pipeline.
pub mod field_names {
    pub const QUERY_LEN: &str = "query_len";
    pub const QUERY_CLASS: &str = "query_class";
    pub const PHASE: &str = "phase";
    pub const RESULT_COUNT: &str = "result_count";
    pub const DOC_COUNT: &str = "doc_count";
    pub const DURATION_US: &str = "duration_us";
    pub const MODEL_ID: &str = "model_id";
    pub const DIMENSION: &str = "dimension";
    pub const BLEND_FACTOR: &str = "blend_factor";
    pub const K: &str = "k";
    pub const LEXICAL_COUNT: &str = "lexical_count";
    pub const SEMANTIC_COUNT: &str = "semantic_count";
    pub const FUSED_COUNT: &str = "fused_count";
    pub const OVERLAP_COUNT: &str = "overlap_count";
}

/// Parse a log level string (case-insensitive).
///
/// Recognized values: `trace`, `debug`, `info`, `warn`, `error`.
/// Returns `None` for unrecognized strings.
#[must_use]
pub fn parse_level(s: &str) -> Option<Level> {
    match s.to_lowercase().as_str() {
        "trace" => Some(Level::TRACE),
        "debug" => Some(Level::DEBUG),
        "info" => Some(Level::INFO),
        "warn" => Some(Level::WARN),
        "error" => Some(Level::ERROR),
        _ => None,
    }
}

/// Returns the recommended `tracing::Level` for the given environment.
///
/// Checks `FRANKENSEARCH_LOG_LEVEL` first, then falls back to the provided
/// default. Recognized values: `trace`, `debug`, `info`, `warn`, `error`.
#[must_use]
pub fn level_from_env(default: Level) -> Level {
    std::env::var("FRANKENSEARCH_LOG_LEVEL")
        .ok()
        .and_then(|s| parse_level(&s))
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_prefix_is_frankensearch() {
        assert_eq!(TARGET_PREFIX, "frankensearch");
    }

    #[test]
    fn span_names_are_consistent() {
        assert!(span_names::SEARCH.starts_with("frankensearch::"));
        assert!(span_names::RRF_FUSE.starts_with("frankensearch::"));
        assert!(span_names::BLEND.starts_with("frankensearch::"));
        assert!(span_names::RERANK.starts_with("frankensearch::"));
    }

    #[test]
    fn parse_level_recognizes_valid_levels() {
        assert_eq!(parse_level("trace"), Some(Level::TRACE));
        assert_eq!(parse_level("debug"), Some(Level::DEBUG));
        assert_eq!(parse_level("info"), Some(Level::INFO));
        assert_eq!(parse_level("warn"), Some(Level::WARN));
        assert_eq!(parse_level("error"), Some(Level::ERROR));
    }

    #[test]
    fn parse_level_case_insensitive() {
        assert_eq!(parse_level("TRACE"), Some(Level::TRACE));
        assert_eq!(parse_level("Debug"), Some(Level::DEBUG));
        assert_eq!(parse_level("INFO"), Some(Level::INFO));
        assert_eq!(parse_level("Error"), Some(Level::ERROR));
    }

    #[test]
    fn parse_level_returns_none_for_invalid() {
        assert_eq!(parse_level("nonsense"), None);
        assert_eq!(parse_level(""), None);
        assert_eq!(parse_level("verbose"), None);
    }
}
