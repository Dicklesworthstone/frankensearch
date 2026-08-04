//! Quill tracing convention.
//!
//! Every engine span uses target [`TARGET`] and a stable
//! `frankensearch::quill::<subsystem>::<phase>` name. Later implementations
//! attach only applicable fields from this common vocabulary: `phase`,
//! `schema_id`, `generation`, `segment_id`, `shard_id`, `doc_count`,
//! `query_len`, `result_count`, `duration_us`, privacy-safe query-shape counts
//! and topology hash, low-cardinality `query_form` (`raw` or `preparsed`) and
//! `cache_lookup` (`not_checked`, `disabled`, `miss`, or `hit`), `plan`,
//! `segments_touched`, `pruning_windows`, `blocks_skipped`, and
//! `candidate_docs`. High-cardinality document IDs, terms, query text, cache
//! keys or fingerprints, source content, and score cutoffs are never span
//! fields.

use std::time::Instant;

/// Records a span's wall-clock duration even when the instrumented stage exits
/// through an error path.
pub(crate) struct StageTimer {
    span: tracing::Span,
    started: Instant,
}

impl StageTimer {
    /// Start timing one span that declares an empty `duration_us` field.
    pub(crate) fn new(span: &tracing::Span) -> Self {
        Self {
            span: span.clone(),
            started: Instant::now(),
        }
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        let duration_us = u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.span.record("duration_us", duration_us);
    }
}

/// Stable tracing target for every Quill event and span.
pub const TARGET: &str = "frankensearch.quill";
/// Scribe batch ingest.
pub const SCRIBE_INGEST: &str = "frankensearch::quill::scribe::ingest";
/// Scalar tokenizer execution within one accumulated document.
pub const SCRIBE_TOKENIZE: &str = "frankensearch::quill::scribe::tokenize";
/// Columnar accumulation and arena growth.
pub const SCRIBE_ACCUMULATE: &str = "frankensearch::quill::scribe::accumulate";
/// Accumulator-to-FSLX radix flush.
pub const SCRIBE_FLUSH: &str = "frankensearch::quill::scribe::flush";
/// Grimoire dictionary lookup or scan.
pub const GRIMOIRE_LOOKUP: &str = "frankensearch::quill::grimoire::lookup";
/// Quiver posting/position decode.
pub const QUIVER_DECODE: &str = "frankensearch::quill::quiver::decode";
/// Argus parse, plan, and query execution.
pub const ARGUS_QUERY: &str = "frankensearch::quill::argus::query";
/// Default-query parse and lenient diagnostics.
pub const ARGUS_PARSE: &str = "frankensearch::quill::argus::parse";
/// BM25 scorer construction and exhaustive or rank-safe pruned traversal.
pub const ARGUS_SCORE: &str = "frankensearch::quill::argus::score";
/// Global top-doc/count collection and materialization.
pub const ARGUS_COLLECT: &str = "frankensearch::quill::argus::collect";
/// Keeper index open/recovery.
pub const KEEPER_OPEN: &str = "frankensearch::quill::keeper::open";
/// Unified Keeper seal, publish, and policy-merge lifecycle family.
pub const KEEPER_LIFECYCLE: &str = "frankensearch::quill::keeper::lifecycle";
/// Compatibility alias for the unified Keeper lifecycle family.
pub const KEEPER_SEAL: &str = KEEPER_LIFECYCLE;
/// Compatibility alias for the unified Keeper lifecycle family.
pub const KEEPER_COMMIT: &str = KEEPER_LIFECYCLE;
/// Keeper tombstone compaction.
pub const KEEPER_COMPACT: &str = "frankensearch::quill::keeper::compact";

/// All stable span names, used by policy tests and logging adapters.
pub const ALL_SPAN_NAMES: &[&str] = &[
    SCRIBE_INGEST,
    SCRIBE_TOKENIZE,
    SCRIBE_ACCUMULATE,
    SCRIBE_FLUSH,
    GRIMOIRE_LOOKUP,
    QUIVER_DECODE,
    ARGUS_QUERY,
    ARGUS_PARSE,
    ARGUS_SCORE,
    ARGUS_COLLECT,
    KEEPER_OPEN,
    KEEPER_LIFECYCLE,
    KEEPER_COMPACT,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_names_use_the_stable_namespace() {
        assert_eq!(TARGET, "frankensearch.quill");
        assert!(
            ALL_SPAN_NAMES
                .iter()
                .all(|name| name.starts_with("frankensearch::quill::"))
        );
    }
}
