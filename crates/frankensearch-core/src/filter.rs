//! Search-time filter predicates for narrowing results without modifying the index.
//!
//! Filters are applied **during** the top-k scan (before computing similarity
//! scores) in vector search, enabling early-exit when the filter is selective.
//! This avoids computing dot products for non-matching records.
//!
//! When `metadata` is `None` (e.g., during vector search where metadata is not
//! stored alongside vectors), built-in metadata filters are permissive by
//! default. Select [`MissingMetadataPolicy::Reject`] for strict filtering:
//! unavailable metadata cannot satisfy that predicate. Searchers can use
//! [`SearchFilter::requires_metadata`] to resolve metadata before selecting
//! vector candidates, so eligible documents are not lost at the top-k cutoff.

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
    ///   `None` in metadata-less vector scans; `Some` after hydration from stored
    ///   fields. Implementations decide how missing metadata affects matching.
    fn matches(&self, doc_id: &str, metadata: Option<&serde_json::Value>) -> bool;

    /// Whether metadata should be resolved before selecting candidate documents.
    ///
    /// Strict metadata predicates return `true`, allowing searchers to hydrate
    /// metadata before the vector top-k scan rather than rejecting eligible
    /// documents because their metadata was not available yet. If metadata
    /// cannot be resolved, scanners still call [`Self::matches`] (or the hash
    /// fast path) with `None`; strict predicates must reject it themselves.
    ///
    /// This is a conservative planning hint, not a blanket rejection rule:
    /// an OR chain containing a strict metadata predicate and an ID predicate
    /// may still accept an ID without metadata. Composite filters propagate
    /// this hint whenever any child requires metadata, preserving their normal
    /// Boolean matching semantics.
    fn requires_metadata(&self) -> bool {
        false
    }

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

    /// Optional exact document IDs bounding every possible match.
    ///
    /// A selective scanner can look up these IDs instead of walking the corpus.
    /// Unlike [`Self::candidate_hashes`], this set may be a **superset** of the
    /// matches: scanners must still call [`Self::matches`] for each gathered ID.
    /// This permits an allow-list to retain its efficient lookup through an
    /// intersection or exact-ID exclusion, without treating hash collisions as
    /// document identity. Every matching document must appear in the set,
    /// independently of metadata, for the duration of a search.
    fn candidate_doc_ids(&self) -> Option<&HashSet<String>> {
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

/// A borrowed filter that excludes exact document IDs from another filter.
///
/// Hashes only accelerate negative checks. A possible hash collision always
/// falls back to the full document ID, so excluding one ID cannot hide another.
/// Nonempty exclusions deliberately do not expose `candidate_hashes`: the
/// gather contract cannot express exact string exclusions with only hashes.
/// The inner filter and the excluded strings must outlive this view.
pub struct ExcludeDocIdsFilter<'a> {
    inner: Option<&'a dyn SearchFilter>,
    excluded: HashSet<&'a str>,
    excluded_hashes: DocIdHashSet,
}

impl<'a> ExcludeDocIdsFilter<'a> {
    /// Exclude the supplied exact IDs, optionally retaining an existing filter.
    #[must_use]
    pub fn new(
        inner: Option<&'a dyn SearchFilter>,
        excluded: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let excluded: HashSet<_> = excluded.into_iter().collect();
        let excluded_hashes = excluded
            .iter()
            .map(|id| fnv1a_hash(id.as_bytes()))
            .collect();
        Self {
            inner,
            excluded,
            excluded_hashes,
        }
    }
}

impl fmt::Debug for ExcludeDocIdsFilter<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExcludeDocIdsFilter")
            .field("excluded_count", &self.excluded.len())
            .field("has_inner", &self.inner.is_some())
            .finish_non_exhaustive()
    }
}

impl SearchFilter for ExcludeDocIdsFilter<'_> {
    fn matches(&self, doc_id: &str, metadata: Option<&serde_json::Value>) -> bool {
        !self.excluded.contains(doc_id)
            && self
                .inner
                .is_none_or(|inner| inner.matches(doc_id, metadata))
    }

    fn requires_metadata(&self) -> bool {
        self.inner.is_some_and(SearchFilter::requires_metadata)
    }

    fn matches_doc_id_hash(
        &self,
        doc_id_hash: u64,
        metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        let inner = self.inner.map_or(Some(true), |inner| {
            inner.matches_doc_id_hash(doc_id_hash, metadata)
        });
        if inner == Some(false) {
            return Some(false);
        }
        if self.excluded_hashes.contains(&doc_id_hash) {
            // Even an exact allow-list cannot turn an exclusion hash into an
            // exclusion identity. Let the scanner perform the string check.
            None
        } else {
            inner
        }
    }

    fn candidate_hashes(&self) -> Option<&DocIdHashSet> {
        if self.excluded.is_empty() {
            self.inner.and_then(SearchFilter::candidate_hashes)
        } else {
            None
        }
    }

    fn candidate_doc_ids(&self) -> Option<&HashSet<String>> {
        // Exclusions can only narrow the inner candidate set. Gatherers still
        // apply this view's exact predicate before admitting a row.
        self.inner.and_then(SearchFilter::candidate_doc_ids)
    }

    fn immutable_candidate_hashes(&self) -> Option<&DocIdHashSet> {
        if self.excluded.is_empty() {
            self.inner
                .and_then(SearchFilter::immutable_candidate_hashes)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "exclude_doc_ids"
    }
}

/// An exact document-ID allow-list with collision-safe hash pre-screening.
///
/// Hash misses are conclusive; hash hits always require a full string match.
/// Selective vector searches can enumerate the IDs and gather their stored
/// rows without scanning unrelated documents. Debug output includes only the
/// count, never the document identifiers.
#[derive(Clone)]
pub struct IncludeDocIdsFilter {
    included: HashSet<String>,
    included_hashes: DocIdHashSet,
}

impl IncludeDocIdsFilter {
    /// Include exactly the supplied IDs. Repeated IDs are deduplicated.
    #[must_use]
    pub fn new(ids: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let included: HashSet<String> = ids.into_iter().map(|id| id.as_ref().to_owned()).collect();
        let included_hashes = included
            .iter()
            .map(|id| fnv1a_hash(id.as_bytes()))
            .collect();
        Self {
            included,
            included_hashes,
        }
    }
}

impl fmt::Debug for IncludeDocIdsFilter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IncludeDocIdsFilter")
            .field("included_count", &self.included.len())
            .finish_non_exhaustive()
    }
}

impl SearchFilter for IncludeDocIdsFilter {
    fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
        self.included.contains(doc_id)
    }

    fn matches_doc_id_hash(
        &self,
        doc_id_hash: u64,
        _metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        (!self.included_hashes.contains(&doc_id_hash)).then_some(false)
    }

    fn candidate_doc_ids(&self) -> Option<&HashSet<String>> {
        Some(&self.included)
    }

    fn name(&self) -> &'static str {
        "include_doc_ids"
    }
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

    fn requires_metadata(&self) -> bool {
        self.filters.iter().any(|filter| filter.requires_metadata())
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

    fn candidate_doc_ids(&self) -> Option<&HashSet<String>> {
        match self.filters.as_slice() {
            [filter] => filter.candidate_doc_ids(),
            _ if self.mode == FilterMode::All => self
                .filters
                .iter()
                .filter_map(|filter| filter.candidate_doc_ids())
                .min_by_key(|ids| ids.len()),
            // An OR can match outside any individual child's allow-list.
            _ => None,
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
            .field("compiled_hashes", &self.compiled_hashes)
            .field(
                "filters",
                &self.filters.iter().map(|f| f.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

// ─── Built-in Filters ────────────────────────────────────────────────────────

/// How a metadata predicate handles unavailable document metadata (`None`).
///
/// Present metadata with a missing or malformed required field is rejected
/// under either policy. This policy controls only unavailable metadata.
///
/// ```
/// use frankensearch_core::{DocTypeFilter, MissingMetadataPolicy, SearchFilter};
/// use serde_json::json;
///
/// let filter = DocTypeFilter::new(["active"]).on_missing_metadata(MissingMetadataPolicy::Reject);
/// assert!(!filter.matches("unknown", None));
/// assert!(filter.matches("known", Some(&json!({"doc_type": "active"}))));
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MissingMetadataPolicy {
    /// Include the document when metadata is unavailable (the default).
    ///
    /// This permissive mode is appropriate only when a later stage hydrates
    /// metadata and rechecks the filter, or when best-effort filtering is enough.
    #[default]
    Allow,
    /// Reject documents whose metadata is unavailable.
    ///
    /// Metadata-less vector calls return no matches for this predicate. Hybrid
    /// searchers can resolve metadata before scoring to retain eligible hits.
    Reject,
}

/// Matches documents whose `doc_type` metadata field is in a given set.
///
/// Permits unavailable metadata by default. Use [`Self::new_strict`] or
/// [`Self::on_missing_metadata`] with [`MissingMetadataPolicy::Reject`] when
/// documents without metadata must not match.
#[derive(Debug, Clone)]
pub struct DocTypeFilter {
    allowed_types: HashSet<String>,
    missing_metadata: MissingMetadataPolicy,
}

impl DocTypeFilter {
    /// Create a filter that accepts documents with any of the given `doc_type` values.
    #[must_use]
    pub fn new(types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed_types: types.into_iter().map(Into::into).collect(),
            missing_metadata: MissingMetadataPolicy::Allow,
        }
    }

    /// Create a filter that requires a matching `doc_type` in available metadata.
    #[must_use]
    pub fn new_strict(types: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::new(types).on_missing_metadata(MissingMetadataPolicy::Reject)
    }

    /// Choose how unavailable metadata affects matching.
    ///
    /// Missing or non-string `doc_type` fields in present metadata are always
    /// rejected, regardless of this policy.
    #[must_use]
    pub const fn on_missing_metadata(mut self, policy: MissingMetadataPolicy) -> Self {
        self.missing_metadata = policy;
        self
    }
}

impl SearchFilter for DocTypeFilter {
    fn matches(&self, _doc_id: &str, metadata: Option<&serde_json::Value>) -> bool {
        let Some(meta) = metadata else {
            return self.missing_metadata == MissingMetadataPolicy::Allow;
        };
        let Some(doc_type) = meta.get("doc_type").and_then(serde_json::Value::as_str) else {
            return false;
        };
        self.allowed_types.contains(doc_type)
    }

    fn requires_metadata(&self) -> bool {
        self.missing_metadata == MissingMetadataPolicy::Reject
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
/// Permits unavailable metadata by default. Use [`Self::new_strict`] or
/// [`Self::on_missing_metadata`] with [`MissingMetadataPolicy::Reject`] when
/// documents without metadata must not match, including for unbounded ranges.
#[derive(Debug, Clone, Copy)]
pub struct DateRangeFilter {
    min_timestamp: Option<i64>,
    max_timestamp: Option<i64>,
    missing_metadata: MissingMetadataPolicy,
}

impl DateRangeFilter {
    /// Create a filter with optional lower and upper bounds (inclusive).
    #[must_use]
    pub const fn new(min: Option<i64>, max: Option<i64>) -> Self {
        Self {
            min_timestamp: min,
            max_timestamp: max,
            missing_metadata: MissingMetadataPolicy::Allow,
        }
    }

    /// Create an inclusive range filter that rejects unavailable metadata.
    #[must_use]
    pub const fn new_strict(min: Option<i64>, max: Option<i64>) -> Self {
        Self::new(min, max).on_missing_metadata(MissingMetadataPolicy::Reject)
    }

    /// Choose how unavailable metadata affects matching.
    ///
    /// Missing or non-integer `created_at` fields in present metadata are always
    /// rejected, regardless of this policy.
    #[must_use]
    pub const fn on_missing_metadata(mut self, policy: MissingMetadataPolicy) -> Self {
        self.missing_metadata = policy;
        self
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
            return self.missing_metadata == MissingMetadataPolicy::Allow;
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

    fn requires_metadata(&self) -> bool {
        self.missing_metadata == MissingMetadataPolicy::Reject
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

    #[test]
    fn exact_inclusions_require_identity_after_a_hash_match() {
        let mut filter = IncludeDocIdsFilter::new(["private-canary", "private-canary", "café"]);
        assert_eq!(filter.candidate_doc_ids().unwrap().len(), 2);
        assert!(filter.matches("private-canary", None));
        assert!(filter.matches("café", None));
        assert!(!filter.matches("cafe", None));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"cafe"), None),
            Some(false)
        );
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"private-canary"), None),
            None
        );
        // Inject a hash pre-screen collision. It must never authorize the ID.
        filter.included_hashes.insert(fnv1a_hash(b"different"));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"different"), None),
            None
        );
        assert!(!filter.matches("different", None));
        assert!(filter.candidate_hashes().is_none());
        assert!(filter.immutable_candidate_hashes().is_none());
        assert!(!format!("{filter:?}").contains("private-canary"));

        let empty = IncludeDocIdsFilter::new(std::iter::empty::<&str>());
        assert!(empty.candidate_doc_ids().unwrap().is_empty());
        assert!(!empty.matches("anything", None));
    }

    #[test]
    fn exact_candidate_bounds_survive_intersection_and_exclusion() {
        let included = IncludeDocIdsFilter::new(["keep.rs", "drop.rs", "keep.txt"]);
        let excluded = ExcludeDocIdsFilter::new(Some(&included), ["drop.rs"]);
        assert_eq!(excluded.candidate_doc_ids(), included.candidate_doc_ids());
        assert!(!excluded.matches("drop.rs", None));
        assert!(excluded.matches("keep.rs", None));

        let chain = FilterChain::new(FilterMode::All)
            .with(Box::new(included))
            .with(Box::new(PredicateFilter::new("keep-rust", |id| {
                id == "keep.rs"
            })))
            .with(Box::new(IncludeDocIdsFilter::new(["keep.rs", "keep.txt"])));
        assert_eq!(chain.candidate_doc_ids().unwrap().len(), 2);
        assert!(chain.matches("keep.rs", None));
        assert!(!chain.matches("keep.txt", None));
        assert!(!chain.matches("drop.rs", None));

        let disjunction = FilterChain::new(FilterMode::Any)
            .with(Box::new(chain))
            .with(Box::new(PredicateFilter::new("anything", |_| true)));
        assert!(disjunction.candidate_doc_ids().is_none());
        assert!(disjunction.matches("outside", None));
    }

    struct ExclusionMetadataFilter;

    impl SearchFilter for ExclusionMetadataFilter {
        fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
            metadata
                .and_then(|value| value.get("allowed"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        }

        fn name(&self) -> &'static str {
            "exclusion-metadata-test"
        }
    }

    #[test]
    fn exact_exclusions_compose_with_hash_and_predicate_filters() {
        let allowed = BitsetFilter::from_doc_ids(["replace", "keep"]);
        let filter = ExcludeDocIdsFilter::new(Some(&allowed), ["replace", "replace"]);
        assert!(!filter.matches("replace", None));
        assert!(filter.matches("keep", None));
        assert!(!filter.matches("outside", None));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"replace"), None),
            None
        );
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"keep"), None),
            Some(true)
        );
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"outside"), None),
            Some(false)
        );
        assert!(filter.candidate_hashes().is_none());
        assert!(filter.immutable_candidate_hashes().is_none());
        let predicate = PredicateFilter::new("suffix", |id| {
            std::path::Path::new(id)
                .extension()
                .is_some_and(|ext| ext == "rs")
        });
        let filter = ExcludeDocIdsFilter::new(Some(&predicate), ["old.rs"]);
        assert!(!filter.matches("old.rs", None));
        assert!(filter.matches("new.rs", None));
        assert!(!filter.matches("new.txt", None));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"new.rs"), None),
            None
        );
    }

    #[test]
    fn simulated_hash_collision_requires_exact_id_comparison() {
        let mut filter = ExcludeDocIdsFilter::new(None, ["excluded"]);
        // Simulate a collision in the hash pre-screen, not in the exact set.
        // This proves the fallback; it is not a claim to have found FNV collisions.
        filter.excluded_hashes.insert(fnv1a_hash(b"different"));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"different"), None),
            None
        );
        assert!(filter.matches("different", None));
        assert!(!filter.matches("excluded", None));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"unrelated"), None),
            Some(true)
        );
    }

    #[test]
    fn empty_exclusions_preserve_enumeration_and_metadata_is_forwarded() {
        let allowed = BitsetFilter::from_doc_ids(["keep"]);
        let filter = ExcludeDocIdsFilter::new(Some(&allowed), []);
        assert_eq!(filter.candidate_hashes(), allowed.candidate_hashes());
        assert_eq!(
            filter.immutable_candidate_hashes(),
            allowed.immutable_candidate_hashes()
        );
        assert!(filter.matches("keep", None));
        assert!(!filter.matches("outside", None));

        let metadata = serde_json::json!({"allowed": true});
        let inner = ExclusionMetadataFilter;
        let filter = ExcludeDocIdsFilter::new(Some(&inner), ["excluded"]);
        assert!(filter.matches("keep", Some(&metadata)));
        assert!(!filter.matches("excluded", Some(&metadata)));
        assert!(!filter.matches("keep", None));
    }

    #[test]
    fn debug_output_does_not_disclose_excluded_ids() {
        let filter = ExcludeDocIdsFilter::new(None, ["private-document-canary"]);
        let debug = format!("{filter:?}");
        assert!(debug.contains("excluded_count"));
        assert!(!debug.contains("private-document-canary"));
    }

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

    #[test]
    fn strict_doc_type_rejects_missing_and_malformed_metadata() {
        let filter = DocTypeFilter::new_strict(["active"]);
        assert!(filter.requires_metadata());
        assert!(!filter.matches("active-looking-id", None));
        assert_eq!(filter.matches_doc_id_hash(0, None), Some(false));
        for metadata in [
            json!(null),
            json!([]),
            json!({}),
            json!({"doc_type": null}),
            json!({"doc_type": ["active"]}),
            json!({"doc_type": true}),
            json!({"doc_type": 1}),
            json!({"doc_type": "archived"}),
        ] {
            assert!(!filter.matches("doc", Some(&metadata)), "{metadata}");
            assert_eq!(filter.matches_doc_id_hash(0, Some(&metadata)), Some(false));
        }
        let accepted = json!({"doc_type": "active"});
        assert!(filter.matches("doc", Some(&accepted)));
        assert_eq!(filter.matches_doc_id_hash(0, Some(&accepted)), Some(true));

        let empty = DocTypeFilter::new_strict(std::iter::empty::<&str>());
        assert!(!empty.matches("doc", None));
        assert!(!empty.matches("doc", Some(&accepted)));
    }

    #[test]
    fn metadata_policy_changes_only_unavailable_metadata() {
        assert_eq!(
            MissingMetadataPolicy::default(),
            MissingMetadataPolicy::Allow
        );
        let valid = json!({"doc_type": "active", "created_at": 1500});
        let missing_fields = json!({});
        for policy in [MissingMetadataPolicy::Allow, MissingMetadataPolicy::Reject] {
            let filters: [Box<dyn SearchFilter>; 2] = [
                Box::new(DocTypeFilter::new_strict(["active"]).on_missing_metadata(policy)),
                Box::new(
                    DateRangeFilter::new_strict(Some(1000), Some(2000)).on_missing_metadata(policy),
                ),
            ];
            for filter in filters {
                assert_eq!(
                    filter.matches("doc", None),
                    policy == MissingMetadataPolicy::Allow
                );
                assert_eq!(
                    filter.requires_metadata(),
                    policy == MissingMetadataPolicy::Reject
                );
                assert!(filter.matches("doc", Some(&valid)));
                assert!(!filter.matches("doc", Some(&missing_fields)));
            }
        }
        assert!(!DocTypeFilter::new(["active"]).requires_metadata());
        assert!(!DateRangeFilter::between(1000, 2000).requires_metadata());
        assert!(
            DocTypeFilter::new(["active"])
                .on_missing_metadata(MissingMetadataPolicy::Reject)
                .requires_metadata()
        );
    }

    // --- DateRangeFilter ---

    #[test]
    fn strict_filters_require_known_matching_fields_on_both_scan_paths() {
        let types =
            DocTypeFilter::new(["active"]).on_missing_metadata(MissingMetadataPolicy::Reject);
        let dates =
            DateRangeFilter::between(10, 20).on_missing_metadata(MissingMetadataPolicy::Reject);
        for (metadata, type_match, date_match) in [
            (None, false, false),
            (Some(json!(null)), false, false),
            (Some(json!({})), false, false),
            (
                Some(json!({"doc_type": 1, "created_at": "10"})),
                false,
                false,
            ),
            (
                Some(json!({"doc_type": "archived", "created_at": 20})),
                false,
                true,
            ),
            (
                Some(json!({"doc_type": "active", "created_at": 21})),
                true,
                false,
            ),
            (
                Some(json!({"doc_type": "active", "created_at": 10})),
                true,
                true,
            ),
        ] {
            let metadata = metadata.as_ref();
            assert_eq!(types.matches("doc", metadata), type_match);
            assert_eq!(dates.matches("doc", metadata), date_match);
            assert_eq!(types.matches_doc_id_hash(42, metadata), Some(type_match));
            assert_eq!(dates.matches_doc_id_hash(42, metadata), Some(date_match));
        }
    }

    #[test]
    fn strict_date_ranges_keep_inclusive_bounds_and_require_an_integer() {
        const STRICT: DateRangeFilter = DateRangeFilter::new_strict(Some(1000), Some(2000));
        assert!(STRICT.requires_metadata());
        assert!(!STRICT.matches("doc", None));
        for (timestamp, accepted) in [
            (i64::MIN, false),
            (999, false),
            (1000, true),
            (1500, true),
            (2000, true),
            (2001, false),
            (i64::MAX, false),
        ] {
            let metadata = json!({"created_at": timestamp});
            assert_eq!(STRICT.matches("doc", Some(&metadata)), accepted);
            assert_eq!(
                STRICT.matches_doc_id_hash(0, Some(&metadata)),
                Some(accepted)
            );
        }
        for metadata in [
            json!(null),
            json!([]),
            json!({}),
            json!({"created_at": null}),
            json!({"created_at": "1500"}),
            json!({"created_at": 1500.5}),
            json!({"created_at": true}),
            json!({"created_at": u64::MAX}),
        ] {
            assert!(!STRICT.matches("doc", Some(&metadata)), "{metadata}");
            assert_eq!(STRICT.matches_doc_id_hash(0, Some(&metadata)), Some(false));
        }
        for filter in [
            DateRangeFilter::new_strict(None, None),
            DateRangeFilter::after(i64::MIN).on_missing_metadata(MissingMetadataPolicy::Reject),
            DateRangeFilter::before(i64::MAX).on_missing_metadata(MissingMetadataPolicy::Reject),
        ] {
            assert!(filter.requires_metadata());
            assert!(!filter.matches("doc", None));
            assert!(!filter.matches("doc", Some(&json!({}))));
            for timestamp in [i64::MIN, 0, i64::MAX] {
                assert!(filter.matches("doc", Some(&json!({"created_at": timestamp}))));
            }
        }
        let inverted = DateRangeFilter::new_strict(Some(2000), Some(1000));
        assert!(!inverted.matches("doc", Some(&json!({"created_at": 1500}))));
    }

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
    fn strict_metadata_hint_preserves_nested_boolean_semantics() {
        let metadata = json!({"doc_type": "active"});
        for mode in [FilterMode::All, FilterMode::Any] {
            let inner = FilterChain::new(mode)
                .with(Box::new(DocTypeFilter::new_strict(["active"])))
                .with(Box::new(BitsetFilter::from_doc_ids(["included"])));
            let chain = FilterChain::new(FilterMode::All).with(Box::new(inner));
            assert!(chain.requires_metadata());
            assert!(chain.candidate_hashes().is_none());
            assert!(chain.immutable_candidate_hashes().is_none());
            for (id, meta, all_matches, any_matches) in [
                ("included", None, false, true),
                ("outside", None, false, false),
                ("included", Some(&metadata), true, true),
                ("outside", Some(&metadata), false, true),
            ] {
                let expected = match mode {
                    FilterMode::All => all_matches,
                    FilterMode::Any => any_matches,
                };
                assert_eq!(chain.matches(id, meta), expected);
                assert_eq!(
                    chain.matches_doc_id_hash(fnv1a_hash(id.as_bytes()), meta),
                    Some(expected)
                );
            }
        }
        for mode in [FilterMode::All, FilterMode::Any] {
            assert!(!FilterChain::new(mode).requires_metadata());
            let permissive = FilterChain::new(mode)
                .with(Box::new(DocTypeFilter::new(["active"])))
                .with(Box::new(PredicateFilter::new("id", |id| id == "included")));
            assert!(!permissive.requires_metadata());
        }
    }

    #[test]
    fn strict_metadata_hint_and_exact_candidates_survive_exclusion() {
        let strict = FilterChain::new(FilterMode::All)
            .with(Box::new(IncludeDocIdsFilter::new(["keep", "excluded"])))
            .with(Box::new(DocTypeFilter::new_strict(["active"])));
        let filter = ExcludeDocIdsFilter::new(Some(&strict), ["excluded"]);
        assert!(filter.requires_metadata());
        assert_eq!(filter.candidate_doc_ids(), strict.candidate_doc_ids());
        assert!(!filter.matches("keep", None));
        assert_eq!(
            filter.matches_doc_id_hash(fnv1a_hash(b"keep"), None),
            Some(false)
        );
        let metadata = json!({"doc_type": "active"});
        assert!(filter.matches("keep", Some(&metadata)));
        assert!(!filter.matches("excluded", Some(&metadata)));
        assert!(!filter.matches("outside", Some(&metadata)));
        assert!(!ExcludeDocIdsFilter::new(None, ["excluded"]).requires_metadata());
        assert!(!BitsetFilter::from_doc_ids(["keep"]).requires_metadata());
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
            .with(Box::new(BitsetFilter::from_doc_ids([
                "doc-b", "doc-c", "doc-d",
            ])))
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

        let mut any =
            FilterChain::new(FilterMode::Any).with(Box::new(BitsetFilter::from_doc_ids(["doc-a"])));
        any.add(Box::new(BitsetFilter::from_doc_ids(["doc-b"])));
        assert_candidates(&any, &["doc-a", "doc-b"]);
        any.add(Box::new(PredicateFilter::new("outside", |id| {
            id == "outside"
        })));
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
                .with(Box::new(PredicateFilter::new("outside", |id| {
                    id == "outside"
                })));
            assert!(with_predicate.candidate_hashes().is_none());
        }
    }

    #[test]
    fn metadata_hash_decisions_equal_document_decisions() {
        let filters: Vec<Box<dyn SearchFilter>> = vec![
            Box::new(DocTypeFilter::new(["tweet"])),
            Box::new(DocTypeFilter::new(std::iter::empty::<&str>())),
            Box::new(DocTypeFilter::new_strict(["tweet"])),
            Box::new(DocTypeFilter::new_strict(std::iter::empty::<&str>())),
            Box::new(DateRangeFilter::between(1000, 2000)),
            Box::new(DateRangeFilter::new(None, None)),
            Box::new(DateRangeFilter::between(i64::MAX, i64::MIN)),
            Box::new(DateRangeFilter::new_strict(Some(1000), Some(2000))),
            Box::new(DateRangeFilter::new_strict(None, None)),
            Box::new(DateRangeFilter::new_strict(Some(i64::MAX), Some(i64::MIN))),
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
