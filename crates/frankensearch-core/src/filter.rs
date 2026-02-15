//! Search-time filter predicates for narrowing results without modifying the index.
//!
//! Filters are applied **during** the top-k scan (before computing similarity
//! scores) in vector search, enabling early-exit when the filter is selective.
//! This avoids computing dot products for non-matching records.
//!
//! When `metadata` is `None` (e.g., during vector search where metadata is not
//! stored alongside vectors), filters that require metadata return `true`
//! (permissive default) to avoid incorrectly filtering out documents whose
//! metadata is simply unavailable.

use std::collections::HashSet;
use std::fmt;

/// Trait for search-time filter predicates applied during scoring.
///
/// Implementations must be cheap (`O(1)` ideally). Expensive predicates should
/// pre-compute a [`BitsetFilter`] or [`PredicateFilter`] with a captured set.
pub trait SearchFilter: Send + Sync {
    /// Returns `true` if this document should be **included** in results.
    ///
    /// * `doc_id` — The document identifier (always available).
    /// * `metadata` — Document metadata, if available at this pipeline stage.
    ///   `None` during vector search; `Some` after fusion or from stored fields.
    fn matches(&self, doc_id: &str, metadata: Option<&serde_json::Value>) -> bool;

    /// Optional fast path for hash-addressable filters.
    ///
    /// Returns:
    /// - `Some(result)` when this filter can decide from `doc_id_hash`
    /// - `None` when a full `doc_id` string is required
    fn matches_doc_id_hash(
        &self,
        _doc_id_hash: u64,
        _metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        None
    }

    /// A short, descriptive name for diagnostics and tracing.
    fn name(&self) -> &str;
}

/// How multiple filters in a [`FilterChain`] are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// All filters must match (logical AND).
    All,
    /// At least one filter must match (logical OR).
    Any,
}

/// Chains multiple [`SearchFilter`] implementations with configurable semantics.
pub struct FilterChain {
    filters: Vec<Box<dyn SearchFilter>>,
    mode: FilterMode,
}

impl FilterChain {
    /// Create a new filter chain with the given combination mode.
    #[must_use]
    pub fn new(mode: FilterMode) -> Self {
        Self {
            filters: Vec::new(),
            mode,
        }
    }

    /// Add a filter to the chain (mutating).
    pub fn add(&mut self, filter: Box<dyn SearchFilter>) -> &mut Self {
        self.filters.push(filter);
        self
    }

    /// Add a filter to the chain (builder pattern).
    #[must_use]
    pub fn with(mut self, filter: Box<dyn SearchFilter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// Number of filters in this chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Whether the chain contains no filters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl SearchFilter for FilterChain {
    fn matches(&self, doc_id: &str, metadata: Option<&serde_json::Value>) -> bool {
        if self.filters.is_empty() {
            return true;
        }
        match self.mode {
            FilterMode::All => self.filters.iter().all(|f| f.matches(doc_id, metadata)),
            FilterMode::Any => self.filters.iter().any(|f| f.matches(doc_id, metadata)),
        }
    }

    fn matches_doc_id_hash(
        &self,
        doc_id_hash: u64,
        metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        if self.filters.is_empty() {
            return Some(true);
        }
        match self.mode {
            FilterMode::All => {
                let mut has_unknown = false;
                for filter in &self.filters {
                    match filter.matches_doc_id_hash(doc_id_hash, metadata) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => has_unknown = true,
                    }
                }
                if has_unknown { None } else { Some(true) }
            }
            FilterMode::Any => {
                let mut has_unknown = false;
                for filter in &self.filters {
                    match filter.matches_doc_id_hash(doc_id_hash, metadata) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => has_unknown = true,
                    }
                }
                if has_unknown { None } else { Some(false) }
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        match self.mode {
            FilterMode::All => "filter_chain(all)",
            FilterMode::Any => "filter_chain(any)",
        }
    }
}

impl fmt::Debug for FilterChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterChain")
            .field("mode", &self.mode)
            .field(
                "filters",
                &self.filters.iter().map(|f| f.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

// ─── Built-in Filters ────────────────────────────────────────────────────────

/// Matches documents whose `doc_type` metadata field is in a given set.
///
/// Returns `true` (permissive) when metadata is `None`, since `doc_type`
/// cannot be determined without metadata.
#[derive(Debug, Clone)]
pub struct DocTypeFilter {
    allowed_types: HashSet<String>,
}

impl DocTypeFilter {
    /// Create a filter that accepts documents with any of the given `doc_type` values.
    #[must_use]
    pub fn new(types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_types: types.into_iter().map(Into::into).collect(),
        }
    }
}

impl SearchFilter for DocTypeFilter {
    fn matches(&self, _doc_id: &str, metadata: Option<&serde_json::Value>) -> bool {
        let Some(meta) = metadata else {
            return true;
        };
        let Some(doc_type) = meta.get("doc_type").and_then(serde_json::Value::as_str) else {
            return false;
        };
        self.allowed_types.contains(doc_type)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "doc_type_filter"
    }
}

/// Matches documents whose `created_at` metadata field (unix timestamp, seconds)
/// falls within a given range.
///
/// Either bound may be `None` for an open-ended range.
/// Returns `true` (permissive) when metadata is `None`.
#[derive(Debug, Clone, Copy)]
pub struct DateRangeFilter {
    min_timestamp: Option<i64>,
    max_timestamp: Option<i64>,
}

impl DateRangeFilter {
    /// Create a filter with optional lower and upper bounds (inclusive).
    #[must_use]
    pub const fn new(min: Option<i64>, max: Option<i64>) -> Self {
        Self {
            min_timestamp: min,
            max_timestamp: max,
        }
    }

    /// Filter for documents created at or after a timestamp.
    #[must_use]
    pub const fn after(timestamp: i64) -> Self {
        Self::new(Some(timestamp), None)
    }

    /// Filter for documents created at or before a timestamp.
    #[must_use]
    pub const fn before(timestamp: i64) -> Self {
        Self::new(None, Some(timestamp))
    }

    /// Filter for documents within a timestamp range (inclusive on both ends).
    #[must_use]
    pub const fn between(min: i64, max: i64) -> Self {
        Self::new(Some(min), Some(max))
    }
}

impl SearchFilter for DateRangeFilter {
    fn matches(&self, _doc_id: &str, metadata: Option<&serde_json::Value>) -> bool {
        let Some(meta) = metadata else {
            return true;
        };
        let Some(created_at) = meta.get("created_at").and_then(serde_json::Value::as_i64) else {
            return false;
        };
        if let Some(min) = self.min_timestamp
            && created_at < min
        {
            return false;
        }
        if let Some(max) = self.max_timestamp
            && created_at > max
        {
            return false;
        }
        true
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "date_range_filter"
    }
}

/// Matches documents whose `doc_id` hash is in a pre-computed set.
///
/// Uses FNV-1a hashing for `O(1)` membership checks. Useful when the set of
/// allowed documents is known ahead of time and potentially large.
#[derive(Debug, Clone)]
pub struct BitsetFilter {
    hashes: HashSet<u64>,
}

impl BitsetFilter {
    /// Create a filter from pre-computed FNV-1a hashes of `doc_id` values.
    #[must_use]
    pub const fn from_hashes(hashes: HashSet<u64>) -> Self {
        Self { hashes }
    }

    /// Create a filter by computing FNV-1a hashes from `doc_id` strings.
    #[must_use]
    pub fn from_doc_ids(doc_ids: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        Self {
            hashes: doc_ids
                .into_iter()
                .map(|id| fnv1a_hash(id.as_ref().as_bytes()))
                .collect(),
        }
    }
}

impl SearchFilter for BitsetFilter {
    fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
        self.matches_doc_id_hash(fnv1a_hash(doc_id.as_bytes()), None)
            .unwrap_or(false)
    }

    fn matches_doc_id_hash(
        &self,
        doc_id_hash: u64,
        _metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        Some(self.hashes.contains(&doc_id_hash))
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "bitset_filter"
    }
}

/// Matches documents using an arbitrary predicate on `doc_id`.
///
/// Useful for ad-hoc filtering when the built-in filter types are insufficient.
pub struct PredicateFilter {
    filter_name: String,
    predicate: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl PredicateFilter {
    /// Create a named predicate filter.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        predicate: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            filter_name: name.into(),
            predicate: Box::new(predicate),
        }
    }
}

impl SearchFilter for PredicateFilter {
    fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
        (self.predicate)(doc_id)
    }

    fn name(&self) -> &str {
        &self.filter_name
    }
}

impl fmt::Debug for PredicateFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredicateFilter")
            .field("name", &self.filter_name)
            .finish_non_exhaustive()
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// FNV-1a hash for byte sequences.
///
/// This is the same FNV-1a variant used by the FSVI index for `doc_id` hashing.
/// Exposed here so consumers can pre-compute hashes for [`BitsetFilter`].
#[must_use]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- DocTypeFilter ---

    #[test]
    fn doc_type_filter_matches_allowed_type() {
        let filter = DocTypeFilter::new(["tweet", "reply"]);
        let meta = json!({"doc_type": "tweet"});
        assert!(filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn doc_type_filter_rejects_disallowed_type() {
        let filter = DocTypeFilter::new(["tweet", "reply"]);
        let meta = json!({"doc_type": "retweet"});
        assert!(!filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn doc_type_filter_rejects_missing_doc_type_field() {
        let filter = DocTypeFilter::new(["tweet"]);
        let meta = json!({"source": "api"});
        assert!(!filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn doc_type_filter_permissive_when_no_metadata() {
        let filter = DocTypeFilter::new(["tweet"]);
        assert!(filter.matches("doc-1", None));
    }

    // --- DateRangeFilter ---

    #[test]
    fn date_range_filter_within_range() {
        let filter = DateRangeFilter::between(1000, 2000);
        let meta = json!({"created_at": 1500});
        assert!(filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn date_range_filter_at_boundaries() {
        let filter = DateRangeFilter::between(1000, 2000);
        let meta_min = json!({"created_at": 1000});
        let meta_max = json!({"created_at": 2000});
        assert!(filter.matches("doc-1", Some(&meta_min)));
        assert!(filter.matches("doc-1", Some(&meta_max)));
    }

    #[test]
    fn date_range_filter_outside_range() {
        let filter = DateRangeFilter::between(1000, 2000);
        let meta_low = json!({"created_at": 999});
        let meta_high = json!({"created_at": 2001});
        assert!(!filter.matches("doc-1", Some(&meta_low)));
        assert!(!filter.matches("doc-1", Some(&meta_high)));
    }

    #[test]
    fn date_range_filter_open_ended_after() {
        let filter = DateRangeFilter::after(1000);
        let meta = json!({"created_at": 999});
        assert!(!filter.matches("doc-1", Some(&meta)));
        let meta = json!({"created_at": 1000});
        assert!(filter.matches("doc-1", Some(&meta)));
        let meta = json!({"created_at": 5000});
        assert!(filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn date_range_filter_open_ended_before() {
        let filter = DateRangeFilter::before(2000);
        let meta = json!({"created_at": 2001});
        assert!(!filter.matches("doc-1", Some(&meta)));
        let meta = json!({"created_at": 2000});
        assert!(filter.matches("doc-1", Some(&meta)));
        let meta = json!({"created_at": 0});
        assert!(filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn date_range_filter_permissive_when_no_metadata() {
        let filter = DateRangeFilter::between(1000, 2000);
        assert!(filter.matches("doc-1", None));
    }

    #[test]
    fn date_range_filter_rejects_missing_created_at_field() {
        let filter = DateRangeFilter::between(1000, 2000);
        let meta = json!({"source": "api"});
        assert!(!filter.matches("doc-1", Some(&meta)));
    }

    // --- BitsetFilter ---

    #[test]
    fn bitset_filter_matches_known_doc_ids() {
        let filter = BitsetFilter::from_doc_ids(["doc-a", "doc-b"]);
        assert!(filter.matches("doc-a", None));
        assert!(filter.matches("doc-b", None));
        assert!(!filter.matches("doc-c", None));
    }

    #[test]
    fn bitset_filter_from_hashes() {
        let hash = fnv1a_hash(b"doc-x");
        let filter = BitsetFilter::from_hashes(HashSet::from([hash]));
        assert!(filter.matches("doc-x", None));
        assert!(!filter.matches("doc-y", None));
    }

    #[test]
    fn bitset_filter_hash_fast_path_matches() {
        let hash = fnv1a_hash(b"doc-x");
        let filter = BitsetFilter::from_hashes(HashSet::from([hash]));
        assert_eq!(filter.matches_doc_id_hash(hash, None), Some(true));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"doc-y"), None),
            Some(false)
        );
    }

    // --- PredicateFilter ---

    #[test]
    fn predicate_filter_with_closure() {
        let filter = PredicateFilter::new("starts_with_test", |doc_id| doc_id.starts_with("test-"));
        assert!(filter.matches("test-123", None));
        assert!(!filter.matches("prod-123", None));
    }

    #[test]
    fn predicate_filter_name() {
        let filter = PredicateFilter::new("my-filter", |_| true);
        assert_eq!(filter.name(), "my-filter");
    }

    // --- FilterChain ---

    #[test]
    fn empty_filter_chain_matches_everything() {
        let chain = FilterChain::new(FilterMode::All);
        assert!(chain.matches("anything", None));
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn filter_chain_all_semantics() {
        let chain = FilterChain::new(FilterMode::All)
            .with(Box::new(PredicateFilter::new("has-prefix", |id| {
                id.starts_with("doc-")
            })))
            .with(Box::new(PredicateFilter::new("not-doc-c", |id| {
                id != "doc-c"
            })));

        assert!(chain.matches("doc-a", None));
        assert!(chain.matches("doc-b", None));
        assert!(!chain.matches("doc-c", None)); // Rejected by second filter
        assert!(!chain.matches("other", None)); // Rejected by first filter
    }

    #[test]
    fn filter_chain_any_semantics() {
        let chain = FilterChain::new(FilterMode::Any)
            .with(Box::new(PredicateFilter::new("is-a", |id| id == "doc-a")))
            .with(Box::new(PredicateFilter::new("is-b", |id| id == "doc-b")));

        assert!(chain.matches("doc-a", None));
        assert!(chain.matches("doc-b", None));
        assert!(!chain.matches("doc-c", None));
    }

    #[test]
    fn filter_chain_hash_fast_path_all_with_unknown_filter() {
        let chain = FilterChain::new(FilterMode::All)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])))
            .with(Box::new(PredicateFilter::new("starts-with-doc", |id| {
                id.starts_with("doc-")
            })));

        assert_eq!(chain.matches_doc_id_hash(fnv1a_hash(b"doc-a"), None), None);
    }

    #[test]
    fn filter_chain_hash_fast_path_any_short_circuits_true() {
        let chain = FilterChain::new(FilterMode::Any)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])))
            .with(Box::new(PredicateFilter::new("is-b", |id| id == "doc-b")));

        assert_eq!(
            chain.matches_doc_id_hash(fnv1a_hash(b"doc-a"), None),
            Some(true)
        );
    }

    #[test]
    fn filter_chain_debug() {
        let chain = FilterChain::new(FilterMode::All)
            .with(Box::new(DocTypeFilter::new(["tweet"])))
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])));
        let debug = format!("{chain:?}");
        assert!(debug.contains("All"));
        assert!(debug.contains("doc_type_filter"));
        assert!(debug.contains("bitset_filter"));
    }

    #[test]
    fn filter_chain_len_tracks_additions() {
        let mut chain = FilterChain::new(FilterMode::All);
        assert_eq!(chain.len(), 0);
        chain.add(Box::new(PredicateFilter::new("f1", |_| true)));
        assert_eq!(chain.len(), 1);
        chain.add(Box::new(PredicateFilter::new("f2", |_| true)));
        assert_eq!(chain.len(), 2);
    }

    // --- DocTypeFilter: empty set ---

    #[test]
    fn doc_type_filter_empty_set_rejects_everything() {
        let filter = DocTypeFilter::new(Vec::<String>::new());
        let meta = json!({"doc_type": "tweet"});
        assert!(!filter.matches("doc-1", Some(&meta)));
        let meta = json!({"doc_type": "dm"});
        assert!(!filter.matches("doc-2", Some(&meta)));
    }

    #[test]
    fn doc_type_filter_empty_set_still_permissive_without_metadata() {
        let filter = DocTypeFilter::new(Vec::<String>::new());
        // Without metadata, DocTypeFilter is always permissive.
        assert!(filter.matches("doc-1", None));
    }

    // --- DateRangeFilter: unbounded ---

    #[test]
    fn date_range_filter_both_none_matches_everything() {
        let filter = DateRangeFilter::new(None, None);
        let meta = json!({"created_at": 0});
        assert!(filter.matches("doc-1", Some(&meta)));
        let meta = json!({"created_at": i64::MAX});
        assert!(filter.matches("doc-1", Some(&meta)));
    }

    #[test]
    fn date_range_filter_rejects_non_integer_timestamp() {
        let filter = DateRangeFilter::between(1000, 2000);
        let meta = json!({"created_at": "2024-01-01"});
        assert!(!filter.matches("doc-1", Some(&meta)));
    }

    // --- BitsetFilter: empty ---

    #[test]
    fn bitset_filter_empty_rejects_everything() {
        let filter = BitsetFilter::from_doc_ids(Vec::<String>::new());
        assert!(!filter.matches("doc-a", None));
        assert!(!filter.matches("", None));
    }

    // --- Filter names ---

    #[test]
    fn all_filter_names_are_descriptive() {
        assert_eq!(DocTypeFilter::new(["tweet"]).name(), "doc_type_filter");
        assert_eq!(DateRangeFilter::between(0, 100).name(), "date_range_filter");
        assert_eq!(
            BitsetFilter::from_doc_ids(["doc-a"]).name(),
            "bitset_filter"
        );
        assert_eq!(PredicateFilter::new("custom", |_| true).name(), "custom");
        assert_eq!(
            FilterChain::new(FilterMode::All).name(),
            "filter_chain(all)"
        );
        assert_eq!(
            FilterChain::new(FilterMode::Any).name(),
            "filter_chain(any)"
        );
    }

    // --- fnv1a_hash ---

    #[test]
    fn fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv1a_hash_different_inputs_differ() {
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    #[test]
    fn fnv1a_hash_empty_input() {
        let hash = fnv1a_hash(b"");
        assert_eq!(hash, 0xcbf2_9ce4_8422_2325); // FNV offset basis
    }

    // --- SearchFilter is object-safe ---

    #[test]
    fn search_filter_is_object_safe() {
        fn accept_filter(_f: &dyn SearchFilter) {}
        let filter = DocTypeFilter::new(["tweet"]);
        accept_filter(&filter);
    }

    // --- Mixed filter with metadata ---

    #[test]
    fn filter_chain_mixed_doc_type_and_predicate() {
        let chain = FilterChain::new(FilterMode::All)
            .with(Box::new(DocTypeFilter::new(["tweet"])))
            .with(Box::new(PredicateFilter::new("has-prefix", |id| {
                id.starts_with("tw-")
            })));

        // With metadata: both must pass
        let meta = json!({"doc_type": "tweet"});
        assert!(chain.matches("tw-123", Some(&meta)));
        assert!(!chain.matches("msg-123", Some(&meta))); // Predicate fails

        let meta = json!({"doc_type": "reply"});
        assert!(!chain.matches("tw-123", Some(&meta))); // DocType fails

        // Without metadata: DocTypeFilter is permissive, only predicate checked
        assert!(chain.matches("tw-123", None));
        assert!(!chain.matches("msg-123", None));
    }

    // --- PredicateFilter Debug ---

    #[test]
    fn predicate_filter_debug() {
        let filter = PredicateFilter::new("test-filter", |_| true);
        let debug = format!("{filter:?}");
        assert!(debug.contains("test-filter"));
    }
}
