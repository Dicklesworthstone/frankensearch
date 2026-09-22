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

    /// Optional allow-set for selective hash-addressable filters.
    ///
    /// Returns `Some(set)` when this filter is an explicit FNV-1a allow-list (a
    /// document passes **iff** its `doc_id_hash` is in the set), letting a scan
    /// that knows the corpus iterate the (small) allow-set and gather only those
    /// positions — `O(|allow-set|)` instead of one membership probe per corpus
    /// document. Returns `None` for predicate/metadata/composite filters that
    /// cannot be enumerated. The result of a gather must equal the per-document
    /// scan, so this is only sound for filters whose `matches_doc_id_hash` is
    /// exactly `Some(set.contains(hash))`.
    fn candidate_hashes(&self) -> Option<&DocIdHashSet> {
        None
    }

    /// An exact allow-set that can be safely compiled into a parent chain.
    ///
    /// This has the same membership contract as [`Self::candidate_hashes`],
    /// plus a stability requirement: neither the set nor its membership
    /// semantics may change through shared references or depend on metadata.
    /// Changes through `&mut self` are allowed. This opt-in is separate so an
    /// existing filter with interior-mutably selected candidate sets is never
    /// silently frozen when added to a [`FilterChain`].
    fn immutable_candidate_hashes(&self) -> Option<&DocIdHashSet> {
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
///
/// Immutable hash allow-lists are intersected (`All`) or unioned (`Any`) when
/// filters are added, preserving selective vector gathering through nested
/// chains. Predicate and metadata filters retain the ordinary per-document
/// path unless an empty AND operand proves that no document can match.
pub struct FilterChain {
    filters: Vec<Box<dyn SearchFilter>>,
    mode: FilterMode,
    compiled_hashes: Option<DocIdHashSet>,
}

impl FilterChain {
    /// Create a new filter chain with the given combination mode.
    #[must_use]
    pub fn new(mode: FilterMode) -> Self {
        Self {
            filters: Vec::new(),
            mode,
            compiled_hashes: None,
        }
    }

    /// Add a filter to the chain (mutating).
    pub fn add(&mut self, filter: Box<dyn SearchFilter>) -> &mut Self {
        // A single-child chain delegates without copying its allow-list. Only
        // the second and later operands need an owned, compiled result.
        if !self.filters.is_empty() {
            let current = if self.filters.len() == 1 {
                self.filters[0].immutable_candidate_hashes().cloned()
            } else {
                self.compiled_hashes.take()
            };
            self.compiled_hashes = match (self.mode, current, filter.immutable_candidate_hashes()) {
                (FilterMode::All, Some(mut current), Some(next)) => {
                    current.retain(|hash| next.contains(hash));
                    Some(current)
                }
                (FilterMode::All, Some(current), None) if current.is_empty() => Some(current),
                (FilterMode::All, None, Some(next)) if next.is_empty() => Some(next.clone()),
                (FilterMode::Any, Some(mut current), Some(next)) => {
                    current.extend(next.iter().copied());
                    Some(current)
                }
                _ => None,
            };
        }
        self.filters.push(filter);
        self
    }

    /// Add a filter to the chain (builder pattern).
    #[must_use]
    pub fn with(mut self, filter: Box<dyn SearchFilter>) -> Self {
        self.add(filter);
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
        if let Some(hashes) = &self.compiled_hashes {
            return Some(hashes.contains(&doc_id_hash));
        }
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

    fn candidate_hashes(&self) -> Option<&DocIdHashSet> {
        match self.filters.as_slice() {
            [filter] => filter.candidate_hashes(),
            _ => self.compiled_hashes.as_ref(),
        }
    }

    fn immutable_candidate_hashes(&self) -> Option<&DocIdHashSet> {
        match self.filters.as_slice() {
            [filter] => filter.immutable_candidate_hashes(),
            _ => self.compiled_hashes.as_ref(),
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

    fn matches_doc_id_hash(
        &self,
        _doc_id_hash: u64,
        metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        // The decision depends only on metadata, never on the document ID.
        Some(self.matches("", metadata))
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

    fn matches_doc_id_hash(
        &self,
        _doc_id_hash: u64,
        metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        // The decision depends only on metadata, never on the document ID.
        Some(self.matches("", metadata))
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "date_range_filter"
    }
}

/// Matches documents whose `doc_id` hash is in a pre-computed set.
///
/// Identity `Hasher` for `u64` keys that are ALREADY well-distributed hashes (the
/// FNV-1a `doc_id` hashes the filter stores). A default `HashSet<u64>` re-runs
/// `SipHash` over each key (~25 cyc); since the key is already a uniform hash,
/// passing it straight through is correct (FNV-1a's high bits feed the `SwissTable`
/// control byte) and ~10× cheaper per probe — which dominates the *filtered* vector
/// scan, where one membership probe runs per candidate before the (cheap, SIMD) dot.
#[derive(Default, Clone, Copy)]
pub struct IdentityHasherU64(u64);

impl std::hash::Hasher for IdentityHasherU64 {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    // Defensive fold: only `write_u64` is exercised for `u64` keys; bytes never
    // arrive in practice, but folding (rather than panicking) keeps the hasher
    // total if a non-`u64` key is ever inserted.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) | u64::from(b);
        }
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

/// `BuildHasher` for [`IdentityHasherU64`].
#[derive(Default, Clone, Copy)]
pub struct BuildIdentityHasherU64;

impl std::hash::BuildHasher for BuildIdentityHasherU64 {
    type Hasher = IdentityHasherU64;
    #[inline]
    fn build_hasher(&self) -> IdentityHasherU64 {
        IdentityHasherU64(0)
    }
}

/// Identity-hashed `u64` set used as a [`BitsetFilter`]'s allow-list.
///
/// Exposed via [`SearchFilter::candidate_hashes`] so a selective filtered scan can
/// iterate the allow-set directly instead of probing it per corpus document.
pub type DocIdHashSet = HashSet<u64, BuildIdentityHasherU64>;

/// Uses FNV-1a hashing for `O(1)` membership checks. Useful when the set of
/// allowed documents is known ahead of time and potentially large.
#[derive(Debug, Clone)]
pub struct BitsetFilter {
    hashes: DocIdHashSet,
}

impl BitsetFilter {
    /// Create a filter from pre-computed FNV-1a hashes of `doc_id` values.
    ///
    /// The keys are re-bucketed into an identity-hashed set ([`IdentityHasherU64`])
    /// so per-probe membership skips `SipHash` — paid once at construction, amortized
    /// over every candidate of a filtered scan.
    #[must_use]
    pub fn from_hashes(hashes: HashSet<u64>) -> Self {
        Self {
            hashes: hashes.into_iter().collect(),
        }
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

    fn candidate_hashes(&self) -> Option<&DocIdHashSet> {
        Some(&self.hashes)
    }

    fn immutable_candidate_hashes(&self) -> Option<&DocIdHashSet> {
        Some(&self.hashes)
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

    fn assert_candidates(filter: &dyn SearchFilter, ids: &[&str]) {
        let expected = BitsetFilter::from_doc_ids(ids.iter().copied());
        assert_eq!(filter.candidate_hashes(), Some(&expected.hashes));
        assert_eq!(filter.immutable_candidate_hashes(), Some(&expected.hashes));
    }

    #[test]
    fn compound_allow_lists_preserve_exhaustive_boolean_membership() {
        let ids = ["doc-a", "doc-b", "doc-c", "doc-d", "outside"];
        let metadata = json!({"doc_type": "tweet", "created_at": 1500});
        for mode in [FilterMode::All, FilterMode::Any] {
            for left in 0_u8..16 {
                for right in 0_u8..16 {
                    let subset = |mask: u8| {
                        (0..4)
                            .filter(move |&i| mask & (1_u8 << i) != 0)
                            .map(|i| ids[i])
                    };
                    let chain = FilterChain::new(mode)
                        .with(Box::new(BitsetFilter::from_doc_ids(subset(left))))
                        .with(Box::new(BitsetFilter::from_doc_ids(subset(right))));
                    let allowed = chain.candidate_hashes().expect("compiled allow-list");
                    for (i, id) in ids.iter().enumerate() {
                        let in_left = left & (1_u8 << i) != 0;
                        let in_right = right & (1_u8 << i) != 0;
                        let expected = match mode {
                            FilterMode::All => in_left && in_right,
                            FilterMode::Any => in_left || in_right,
                        };
                        let hash = fnv1a_hash(id.as_bytes());
                        assert_eq!(allowed.contains(&hash), expected);
                        for meta in [None, Some(&metadata)] {
                            assert_eq!(chain.matches(id, meta), expected);
                            assert_eq!(chain.matches_doc_id_hash(hash, meta), Some(expected));
                        }
                    }
                    assert_eq!(chain.immutable_candidate_hashes(), Some(allowed));
                }
            }
        }
    }

    #[test]
    fn nested_chains_compile_union_then_intersection() {
        let left = FilterChain::new(FilterMode::Any)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-a", "doc-b"])))
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-b", "doc-c"])));
        let right = FilterChain::new(FilterMode::All)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-b", "doc-c", "doc-d"])))
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-c", "doc-d"])));
        let chain = FilterChain::new(FilterMode::All)
            .with(Box::new(left))
            .with(Box::new(right));
        assert_candidates(&chain, &["doc-c"]);
        assert!(chain.matches("doc-c", None));
        assert!(!chain.matches("doc-b", None));
        assert!(!chain.matches("doc-d", None));
    }

    #[test]
    fn single_child_delegates_without_copying_candidates() {
        let filter = Box::new(BitsetFilter::from_doc_ids(["doc-a", "doc-b"]));
        let original = std::ptr::from_ref(filter.candidate_hashes().expect("allow-list"));
        let chain = FilterChain::new(FilterMode::All).with(filter);
        let delegated = std::ptr::from_ref(chain.candidate_hashes().expect("allow-list"));
        assert!(std::ptr::eq(original, delegated));
        assert_candidates(&chain, &["doc-a", "doc-b"]);
    }

    #[test]
    fn adding_filters_rebuilds_or_invalidates_compiled_candidates() {
        let mut all = FilterChain::new(FilterMode::All)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-a", "doc-b"])));
        assert_candidates(&all, &["doc-a", "doc-b"]);
        all.add(Box::new(BitsetFilter::from_doc_ids(["doc-b", "doc-c"])));
        assert_candidates(&all, &["doc-b"]);
        all.add(Box::new(BitsetFilter::from_doc_ids(["doc-c"])));
        assert_candidates(&all, &[]);

        let mut any = FilterChain::new(FilterMode::Any)
            .with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])));
        any.add(Box::new(BitsetFilter::from_doc_ids(["doc-b"])));
        assert_candidates(&any, &["doc-a", "doc-b"]);
        any.add(Box::new(PredicateFilter::new("outside", |id| id == "outside")));
        assert!(any.candidate_hashes().is_none());
        assert!(any.immutable_candidate_hashes().is_none());
        assert!(any.matches("outside", None));
        any.add(Box::new(BitsetFilter::from_doc_ids(["doc-c"])));
        assert!(any.candidate_hashes().is_none());
        assert!(any.matches("outside", None));
    }

    #[test]
    fn empty_chain_is_not_an_empty_allow_list() {
        for mode in [FilterMode::All, FilterMode::Any] {
            let chain = FilterChain::new(mode);
            assert!(chain.candidate_hashes().is_none());
            assert!(chain.immutable_candidate_hashes().is_none());
            assert!(chain.matches("outside", None));
            assert_eq!(chain.matches_doc_id_hash(0, None), Some(true));
        }
    }

    #[test]
    fn empty_and_operand_annihilates_unknown_filters_in_either_order() {
        for empty_first in [false, true] {
            let empty = Box::new(BitsetFilter::from_doc_ids(std::iter::empty::<&str>()));
            let unknown = Box::new(PredicateFilter::new("unknown", |_| true));
            let mut chain = FilterChain::new(FilterMode::All);
            if empty_first {
                chain.add(empty).add(unknown);
            } else {
                chain.add(unknown).add(empty);
            }
            assert_candidates(&chain, &[]);
            chain.add(Box::new(DocTypeFilter::new(["tweet"])));
            assert_candidates(&chain, &[]);
            assert!(!chain.matches("doc-a", None));
            assert_eq!(
                chain.matches_doc_id_hash(fnv1a_hash(b"doc-a"), None),
                Some(false)
            );
        }
    }

    #[test]
    fn metadata_and_predicates_do_not_masquerade_as_exact_allow_lists() {
        let accepted = json!({"doc_type": "tweet", "created_at": 1500});
        let rejected = json!({"doc_type": "reply", "created_at": 3000});
        for mode in [FilterMode::All, FilterMode::Any] {
            let chain = FilterChain::new(mode)
                .with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])))
                .with(Box::new(DocTypeFilter::new(["tweet"])))
                .with(Box::new(DateRangeFilter::between(1000, 2000)));
            assert!(chain.candidate_hashes().is_none());
            assert!(chain.immutable_candidate_hashes().is_none());
            for meta in [None, Some(&accepted), Some(&rejected)] {
                for id in ["doc-a", "outside"] {
                    assert_eq!(
                        chain.matches_doc_id_hash(fnv1a_hash(id.as_bytes()), meta),
                        Some(chain.matches(id, meta))
                    );
                }
            }
            let with_predicate = FilterChain::new(mode)
                .with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])))
                .with(Box::new(PredicateFilter::new("outside", |id| id == "outside")));
            assert!(with_predicate.candidate_hashes().is_none());
        }
    }

    #[test]
    fn metadata_hash_decisions_equal_document_decisions() {
        let filters: Vec<Box<dyn SearchFilter>> = vec![
            Box::new(DocTypeFilter::new(["tweet"])),
            Box::new(DocTypeFilter::new(std::iter::empty::<&str>())),
            Box::new(DateRangeFilter::between(1000, 2000)),
            Box::new(DateRangeFilter::new(None, None)),
            Box::new(DateRangeFilter::between(i64::MAX, i64::MIN)),
        ];
        let metadata = [
            json!({"doc_type": "tweet", "created_at": 1500}),
            json!({"doc_type": "reply", "created_at": 1000}),
            json!({"doc_type": "tweet", "created_at": 2000}),
            json!({"doc_type": 1, "created_at": "1500"}),
            json!({"created_at": i64::MIN}),
            json!({"created_at": i64::MAX}),
            json!({}),
            json!(null),
            json!([]),
        ];
        for filter in filters {
            for meta in std::iter::once(None).chain(metadata.iter().map(Some)) {
                assert_eq!(
                    filter.matches_doc_id_hash(u64::MAX, meta),
                    Some(filter.matches("any-document", meta))
                );
            }
        }
    }

    struct SwitchingCandidates {
        switched: std::sync::Arc<std::sync::atomic::AtomicBool>,
        first: BitsetFilter,
        second: BitsetFilter,
    }

    impl SwitchingCandidates {
        fn current(&self) -> &BitsetFilter {
            if self.switched.load(std::sync::atomic::Ordering::Relaxed) {
                &self.second
            } else {
                &self.first
            }
        }
    }

    impl SearchFilter for SwitchingCandidates {
        fn matches(&self, id: &str, metadata: Option<&serde_json::Value>) -> bool {
            self.current().matches(id, metadata)
        }

        fn matches_doc_id_hash(
            &self,
            hash: u64,
            metadata: Option<&serde_json::Value>,
        ) -> Option<bool> {
            self.current().matches_doc_id_hash(hash, metadata)
        }

        fn candidate_hashes(&self) -> Option<&DocIdHashSet> {
            self.current().candidate_hashes()
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "switching-candidates"
        }
    }

    #[test]
    fn dynamic_candidate_sets_are_not_frozen_by_compilation() {
        let switched = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dynamic = SwitchingCandidates {
            switched: std::sync::Arc::clone(&switched),
            first: BitsetFilter::from_doc_ids(["doc-a"]),
            second: BitsetFilter::from_doc_ids(["doc-b"]),
        };
        let mut chain = FilterChain::new(FilterMode::All).with(Box::new(dynamic));
        assert!(
            chain
                .candidate_hashes()
                .expect("delegated")
                .contains(&fnv1a_hash(b"doc-a"))
        );
        assert!(chain.immutable_candidate_hashes().is_none());
        chain.add(Box::new(BitsetFilter::from_doc_ids(["doc-a", "doc-b"])));
        assert!(chain.candidate_hashes().is_none());
        assert!(chain.matches("doc-a", None));
        assert!(!chain.matches("doc-b", None));
        switched.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(!chain.matches("doc-a", None));
        assert!(chain.matches("doc-b", None));
        assert_eq!(
            chain.matches_doc_id_hash(fnv1a_hash(b"doc-a"), None),
            Some(false)
        );
        assert_eq!(
            chain.matches_doc_id_hash(fnv1a_hash(b"doc-b"), None),
            Some(true)
        );
    }
}
