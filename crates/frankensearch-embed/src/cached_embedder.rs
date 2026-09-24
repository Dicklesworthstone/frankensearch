//! Caching wrapper for any [`Embedder`] implementation.
//!
//! `CachedEmbedder` sits between the search pipeline and an inner embedder,
//! caching repeated raw and identity-bound query embeddings separately. Bound
//! entries retain the provider's actual complete response and are keyed by its
//! independently admitted identity and the query. Raw entries can never supply
//! bound requests. Bound batches bypass the cache and are never deduplicated.
//!
//! The cache uses FIFO eviction with a bounded capacity (default 128 entries).
//! Cache hits return a cloned `Vec<f32>`, which is cheap (~1.5 KiB for 384-dim).
//!
//! # Thread Safety
//!
//! The cache is protected by a `std::sync::Mutex`, keeping the wrapper `Send + Sync`.
//! The lock is held only for the brief `HashMap` lookup/insert — never across an
//! async `.await` boundary. Clearing replaces the admission epoch, so inference
//! started before a clear cannot refill or evict entries in the new cache.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{
    Embedder, IdentityBoundEmbedding, ModelCategory, ModelTier, SearchFuture,
};
use frankensearch_core::{QuantizationFormat, SearchError, SearchResult};

/// Default maximum number of cached query embeddings.
const DEFAULT_CAPACITY: usize = 128;

fn cache_checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "embedding.cache".to_owned(),
        reason: "embedding request cancelled".to_owned(),
    })
}

/// Statistics snapshot from a [`CachedEmbedder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache hits since creation (or last clear).
    pub hits: u64,
    /// Number of cache misses since creation (or last clear).
    pub misses: u64,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Maximum capacity before FIFO eviction kicks in.
    pub capacity: usize,
}

struct CacheEntry<T> {
    value: T,
    /// Access frequency, saturating at [`FREQ_CAP`]. Drives S3-FIFO promotion
    /// (Small→Main) and the Main second-chance.
    freq: u8,
}

const FREQ_CAP: u8 = 3;

#[derive(Clone, PartialEq, Eq, Hash)]
struct BoundCacheKey {
    identity: Arc<EmbeddingIdentityBundleV1>,
    text: String,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum CacheKey {
    Raw(String),
    Bound(BoundCacheKey),
}

/// S3-FIFO query-embedding cache (Yang et al., SOSP 2023), entry-count form.
///
/// Three queues over both entry maps: **Small** (new/one-hit-wonder admissions,
/// ~10% of capacity), **Main** (proven-reused, ~90%), and **Ghost** (keys recently
/// evicted from Small, metadata-only). Unlike the previous plain FIFO, a key
/// re-requested while resident is promoted to Main and survives the scan churn that
/// evicts cold one-hit-wonders from Small — measurably fewer embed misses on skewed
/// / scan-heavy query streams (see the `cache_replay` bench + `PERF_LEDGER` 2026-06-29).
/// Raw lookups borrow `&str` (no per-get allocation). Both namespaces share one
/// capacity, eviction policy, statistics and clear epoch; storing a bound response
/// never gives raw callers an entry or allocates a second independent budget.
struct CacheState {
    entries: HashMap<String, CacheEntry<Vec<f32>>>,
    bound_entries: HashMap<BoundCacheKey, CacheEntry<IdentityBoundEmbedding>>,
    small: VecDeque<CacheKey>,
    main: VecDeque<CacheKey>,
    ghost: VecDeque<CacheKey>,
    ghost_set: HashSet<CacheKey>,
    capacity: usize,
    small_cap: usize,
    ghost_cap: usize,
    hits: u64,
    misses: u64,
    // Retained by each miss until admission. Pointer identity cannot wrap or
    // be reused while an older in-flight operation still owns its token.
    epoch: Arc<()>,
}

impl CacheState {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            bound_entries: HashMap::new(),
            small: VecDeque::new(),
            main: VecDeque::new(),
            ghost: VecDeque::new(),
            ghost_set: HashSet::new(),
            capacity,
            // Small queue ~10% of capacity (≥1 when caching is enabled).
            small_cap: (capacity / 10).max(1),
            ghost_cap: capacity,
            hits: 0,
            misses: 0,
            epoch: Arc::new(()),
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(entry) = self.entries.get_mut(key) {
            self.hits += 1;
            entry.freq = entry.freq.saturating_add(1).min(FREQ_CAP);
            Some(entry.value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Record a hit that was served without a fresh lookup — used for an in-batch
    /// duplicate query that folds onto another slot's just-computed embedding, so
    /// batch stats match the old per-text loop (dup = hit, not a second miss).
    fn record_hit(&mut self) {
        self.hits += 1;
    }

    fn lookup_bound(&mut self, key: &BoundCacheKey) -> Option<IdentityBoundEmbedding> {
        if let Some(entry) = self.bound_entries.get(key) {
            // A bound hit is counted only after producer/cancellation revalidation.
            Some(entry.value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn record_bound_hit(&mut self, key: &BoundCacheKey) {
        self.hits += 1;
        if let Some(entry) = self.bound_entries.get_mut(key) {
            entry.freq = entry.freq.saturating_add(1).min(FREQ_CAP);
        }
    }

    fn insert(&mut self, key: String, value: Vec<f32>) {
        // capacity == 0 means caching is disabled.
        if self.capacity == 0 || self.entries.contains_key(&key) {
            return;
        }
        self.entries
            .insert(key.clone(), CacheEntry { value, freq: 0 });
        self.admit(CacheKey::Raw(key));
    }

    fn insert_bound(&mut self, key: BoundCacheKey, value: IdentityBoundEmbedding) {
        if self.capacity == 0 || self.bound_entries.contains_key(&key) {
            return;
        }
        self.bound_entries
            .insert(key.clone(), CacheEntry { value, freq: 0 });
        self.admit(CacheKey::Bound(key));
    }

    fn admit(&mut self, key: CacheKey) {
        // A key seen recently (in Ghost) re-enters straight into Main; a fresh key
        // starts in Small so a scan of one-hit-wonders can't displace the hot set.
        if self.ghost_set.remove(&key) {
            self.main.push_back(key);
        } else {
            self.small.push_back(key);
        }
        while self.entries.len() + self.bound_entries.len() > self.capacity {
            self.evict_one();
        }
    }

    fn frequency_mut(&mut self, key: &CacheKey) -> Option<&mut u8> {
        match key {
            CacheKey::Raw(key) => self.entries.get_mut(key).map(|entry| &mut entry.freq),
            CacheKey::Bound(key) => self.bound_entries.get_mut(key).map(|entry| &mut entry.freq),
        }
    }

    fn remove(&mut self, key: &CacheKey) {
        match key {
            CacheKey::Raw(key) => {
                self.entries.remove(key);
            }
            CacheKey::Bound(key) => {
                self.bound_entries.remove(key);
            }
        }
    }

    /// Free exactly one slot: evict from Small when it's over its target (promoting
    /// reused keys to Main, demoting cold ones to Ghost), otherwise give Main keys a
    /// frequency-decremented second chance until one with `freq == 0` is dropped.
    fn evict_one(&mut self) {
        loop {
            // Evict from Small whenever it is at-or-over its target (`>=`): a fresh
            // key on probation survives only if re-accessed (freq>0 → promoted to
            // Main) before the next eviction; cold one-hit-wonders are dropped. Using
            // `>=` (not `>`) keeps this correct at the degenerate `small_cap ==
            // capacity` size (e.g. capacity 1), where `>` would evict a just-promoted
            // Main entry instead of the cold Small one.
            if !self.small.is_empty() && self.small.len() >= self.small_cap {
                let Some(k) = self.small.pop_front() else {
                    continue;
                };
                if let Some(freq) = self.frequency_mut(&k).filter(|freq| **freq > 0) {
                    *freq = 0;
                    self.main.push_back(k); // promote — no slot freed, keep going
                } else {
                    self.remove(&k);
                    self.push_ghost(k);
                    return;
                }
            } else if let Some(k) = self.main.pop_front() {
                if let Some(freq) = self.frequency_mut(&k).filter(|freq| **freq > 0) {
                    *freq -= 1;
                    self.main.push_back(k); // second chance — keep going
                } else {
                    self.remove(&k);
                    return;
                }
            } else {
                return; // defensive: both queues drained
            }
        }
    }

    fn push_ghost(&mut self, key: CacheKey) {
        if self.ghost_cap == 0 {
            return;
        }
        self.ghost.push_back(key.clone());
        self.ghost_set.insert(key);
        while self.ghost.len() > self.ghost_cap {
            if let Some(old) = self.ghost.pop_front() {
                self.ghost_set.remove(&old);
            }
        }
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len() + self.bound_entries.len(),
            capacity: self.capacity,
        }
    }

    fn clear(&mut self) {
        self.epoch = Arc::new(());
        self.entries.clear();
        self.bound_entries.clear();
        self.small.clear();
        self.main.clear();
        self.ghost.clear();
        self.ghost_set.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Caching wrapper around any [`Embedder`].
///
/// Caches raw `embed()` and `embed_batch()` results for previously seen queries,
/// and `embed_bound()` responses under their complete producer/input identity.
/// Raw and bound entries are separate, sharing one total capacity and statistics.
/// Every bound lookup admits the current identity and native output contract;
/// hits and fills require the producer to retain that identity. A response's
/// identity must exactly match, and its actual values and identity are retained.
/// Bound batches retain the original input order and duplicate slots and invoke
/// the provider's bound batch operation once, even for empty input. Responses
/// are validated all-or-error without relabeling; admission against a stored
/// generation remains the caller's responsibility. Bound batches do not affect
/// cache statistics or contents, whether they succeed or fail.
///
/// Raw responses are validated before insertion: batches must contain exactly
/// one finite, correctly sized vector per distinct miss. An invalid batch never
/// inserts a valid-looking prefix into the cache. Cancellation after inference
/// takes precedence over either a successful response or a provider error.
///
/// # Construction
///
/// ```ignore
/// use frankensearch_embed::CachedEmbedder;
///
/// let inner: Arc<dyn Embedder> = /* ... */;
/// let cached = CachedEmbedder::new(inner, 128);
/// ```
pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    state: Mutex<CacheState>,
}

impl std::fmt::Debug for CachedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.cache_stats();
        f.debug_struct("CachedEmbedder")
            .field("inner_id", &self.inner.id())
            .field("hits", &stats.hits)
            .field("misses", &stats.misses)
            .field("entries", &stats.entries)
            .field("capacity", &stats.capacity)
            .finish_non_exhaustive()
    }
}

impl CachedEmbedder {
    /// Wrap an embedder with a bounded query cache.
    ///
    /// `capacity` controls the maximum number of cached embeddings before
    /// FIFO eviction begins.
    #[must_use]
    pub fn new(inner: Arc<dyn Embedder>, capacity: usize) -> Self {
        Self {
            inner,
            state: Mutex::new(CacheState::new(capacity)),
        }
    }

    /// Wrap an embedder with the default capacity (128 entries).
    #[must_use]
    pub fn with_default_capacity(inner: Arc<dyn Embedder>) -> Self {
        Self::new(inner, DEFAULT_CAPACITY)
    }

    fn state_lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn invalid_response(&self, detail: impl Into<String>) -> SearchError {
        SearchError::EmbeddingFailed {
            model: self.inner.id().to_owned(),
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                detail.into(),
            )),
        }
    }

    fn validate_bound_response(response: &IdentityBoundEmbedding) -> SearchResult<()> {
        response.validate().map_err(|_| SearchError::InvalidConfig {
            field: "embedding.cache.bound_response".to_owned(),
            value: String::new(),
            reason: "provider returned an invalid identity, representation, dimension or non-finite vector"
                .to_owned(),
        })
    }

    fn bound_identity_error() -> SearchError {
        SearchError::UnverifiableRemoteSpace {
            producer: "embedding.cache.bound".to_owned(),
            reason:
                "producer or response does not retain the admitted complete native-f32 identity"
                    .to_owned(),
        }
    }

    fn capture_bound_identity(&self) -> SearchResult<EmbeddingIdentityBundleV1> {
        let identity = self
            .inner
            .identity()
            .map_err(|_| Self::bound_identity_error())?
            .clone();
        if identity.validate().is_err()
            || usize::try_from(identity.space.dimension).ok() != Some(self.inner.dimension())
            || identity.storage.quantization != QuantizationFormat::F32
            || !identity.storage.format.starts_with("in-memory-")
            || !matches!(
                identity.storage.endianness.as_str(),
                "native-f32-values" | "native-test-only"
            )
        {
            return Err(Self::bound_identity_error());
        }
        Ok(identity)
    }

    fn validate_bound_producer(&self, expected: &EmbeddingIdentityBundleV1) -> SearchResult<()> {
        if self
            .inner
            .identity()
            .is_ok_and(|identity| identity == expected)
            && usize::try_from(expected.space.dimension).ok() == Some(self.inner.dimension())
        {
            Ok(())
        } else {
            Err(Self::bound_identity_error())
        }
    }

    fn validate_vector(&self, values: &[f32]) -> SearchResult<()> {
        let expected = self.inner.dimension();
        if values.len() != expected {
            return Err(SearchError::DimensionMismatch {
                expected,
                found: values.len(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            // Do not echo query text, vector values or a provider's raw payload.
            return Err(self.invalid_response("embedding response contains non-finite values"));
        }
        Ok(())
    }

    fn validate_batch(&self, embedded: &[Vec<f32>], expected: usize) -> SearchResult<()> {
        // Validate the entire response before indexing it or mutating the cache.
        // Short and oversized batches used to panic after caching a partial prefix.
        if embedded.len() != expected {
            return Err(self.invalid_response(format!(
                "embedding batch returned {} vectors for {expected} distinct inputs",
                embedded.len(),
            )));
        }
        for values in embedded {
            self.validate_vector(values)?;
        }
        Ok(())
    }

    fn fill_batch_misses(
        out: &mut [Option<Vec<f32>>],
        slot_miss: &[Option<usize>],
        embedded: Vec<Vec<f32>>,
    ) {
        // Cardinality was checked before admission. Move each result into its
        // last output slot, cloning only for same-batch duplicate queries.
        let mut miss_use_counts = vec![0_usize; embedded.len()];
        for maybe_idx in slot_miss {
            if let Some(idx) = *maybe_idx {
                miss_use_counts[idx] += 1;
            }
        }
        let mut embedded_slots: Vec<Option<Vec<f32>>> = embedded.into_iter().map(Some).collect();
        for (slot, maybe_idx) in out.iter_mut().zip(slot_miss) {
            if let Some(idx) = *maybe_idx {
                let use_count = &mut miss_use_counts[idx];
                *use_count = use_count.saturating_sub(1);
                let vec = if *use_count == 0 {
                    embedded_slots[idx].take().expect("embedding slot filled")
                } else {
                    embedded_slots[idx]
                        .as_ref()
                        .expect("embedding slot filled")
                        .clone()
                };
                *slot = Some(vec);
            }
        }
    }

    /// Return a snapshot of cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.state_lock().stats()
    }

    /// Clear all cached embeddings and reset statistics.
    ///
    /// This also fences admission by already-running requests. They may return
    /// their validated results to their original callers, but cannot refill the
    /// cleared cache, change its reset statistics or evict newer entries.
    pub fn clear_cache(&self) {
        self.state_lock().clear();
    }

    /// Reference to the inner embedder.
    #[must_use]
    pub fn inner(&self) -> &dyn Embedder {
        &*self.inner
    }
}

impl Embedder for CachedEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            // Resolve lazily: constructing an unpolled future must not capture
            // a stale cache hit, update statistics or begin provider work.
            cache_checkpoint(cx)?;
            let epoch = {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                if let Some(vec) = cache.get(text) {
                    return Ok(vec);
                }
                Arc::clone(&cache.epoch)
            };
            let outcome = self.inner.embed(cx, text).await;
            cache_checkpoint(cx)?;
            let vec = outcome?;
            self.validate_vector(&vec)?;
            {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                if Arc::ptr_eq(&epoch, &cache.epoch) {
                    cache.insert(text.to_owned(), vec.clone());
                }
            }
            Ok(vec)
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            cache_checkpoint(cx)?;
            let identity = self.capture_bound_identity();
            cache_checkpoint(cx)?;
            let key = BoundCacheKey {
                identity: Arc::new(identity?),
                text: text.to_owned(),
            };
            let (cached, epoch) = {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                (cache.lookup_bound(&key), Arc::clone(&cache.epoch))
            };
            if let Some(response) = cached {
                // Provider callbacks run outside the mutex: even identity() may
                // reenter clear_cache(). Its epoch fences any deferred hit count.
                let producer = self.validate_bound_producer(&key.identity);
                cache_checkpoint(cx)?;
                producer?;
                {
                    let mut cache = self.state_lock();
                    cache_checkpoint(cx)?;
                    if Arc::ptr_eq(&epoch, &cache.epoch) {
                        cache.record_bound_hit(&key);
                    }
                }
                cache_checkpoint(cx)?;
                return Ok(response);
            }
            let outcome = self.inner.embed_bound(cx, text).await;
            cache_checkpoint(cx)?;
            let producer = self.validate_bound_producer(&key.identity);
            cache_checkpoint(cx)?;
            producer?;
            let response = outcome?;
            // Compare before diagnostics inspect provider-controlled fields.
            // Equal names, dimensions, or mathematical spaces are insufficient.
            if response.identity != *key.identity {
                return Err(Self::bound_identity_error());
            }
            Self::validate_bound_response(&response)?;
            {
                let mut cache = self.state_lock();
                cache_checkpoint(cx)?;
                if Arc::ptr_eq(&epoch, &cache.epoch) {
                    cache.insert_bound(key, response.clone());
                }
            }
            cache_checkpoint(cx)?;
            Ok(response)
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            cache_checkpoint(cx)?;
            // Never route this through the raw cache or deduplicate its inputs.
            // The batch provider owns response identity and per-input ordering.
            let outcome = self.inner.embed_batch_bound(cx, texts).await;
            cache_checkpoint(cx)?;
            let responses = outcome?;
            if responses.len() != texts.len() {
                return Err(SearchError::InvalidConfig {
                    field: "embedder.batch_length".to_owned(),
                    value: responses.len().to_string(),
                    reason: format!(
                        "expected exactly {} identity-bound output vectors, one per input",
                        texts.len(),
                    ),
                });
            }
            if let Some(first) = responses.first() {
                for response in &responses {
                    cache_checkpoint(cx)?;
                    Self::validate_bound_response(response)?;
                    if response.identity != first.identity {
                        return Err(SearchError::UnverifiableRemoteSpace {
                            producer: "embedding.cache.bound_batch".to_owned(),
                            reason: "a bound embedding batch must carry one producing identity"
                                .to_owned(),
                        });
                    }
                }
            }
            cache_checkpoint(cx)?;
            Ok(responses)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            cache_checkpoint(cx)?;
            // Pass 1 (single lock scope, released BEFORE the await so the batched
            // inner call holds no lock): resolve cache hits and collect the *distinct*
            // misses. A miss text repeated within the batch folds onto one inner
            // embedding and counts as a hit — the in-batch dedup the old per-text loop
            // got for free — so a batch of repeated queries embeds each text at most once.
            let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
            let mut miss_texts: Vec<&str> = Vec::new();
            // Output slot -> index into `miss_texts` (None once the slot is resolved).
            let mut slot_miss: Vec<Option<usize>> = Vec::with_capacity(texts.len());
            // Dedup map: distinct miss text -> its index in `miss_texts`.
            let mut miss_index: HashMap<&str, usize> = HashMap::new();
            let epoch = {
                let mut cache = self.state_lock();
                for &text in texts {
                    cache_checkpoint(cx)?;
                    if let Some(&idx) = miss_index.get(text) {
                        // Repeat of a text already queued this batch: fold onto the
                        // same inner result, no second inner call, record a hit.
                        cache.record_hit();
                        out.push(None);
                        slot_miss.push(Some(idx));
                        continue;
                    }
                    match cache.get(text) {
                        Some(vec) => {
                            out.push(Some(vec));
                            slot_miss.push(None);
                        }
                        None => {
                            let idx = miss_texts.len();
                            miss_texts.push(text);
                            miss_index.insert(text, idx);
                            out.push(None);
                            slot_miss.push(Some(idx));
                        }
                    }
                }
                Arc::clone(&cache.epoch)
            };
            // ONE batched inner call for all distinct misses — vs the old per-text loop
            // that called `inner.embed` N times, defeating a batching inner (e.g.
            // fastembed embeds the whole batch in a single model invocation). N → 1.
            if !miss_texts.is_empty() {
                cache_checkpoint(cx)?;
                let all_slots_are_distinct_misses = miss_texts.len() == texts.len();
                let outcome = self.inner.embed_batch(cx, &miss_texts).await;
                cache_checkpoint(cx)?;
                let embedded = outcome?;
                self.validate_batch(&embedded, miss_texts.len())?;
                {
                    let mut cache = self.state_lock();
                    cache_checkpoint(cx)?;
                    if Arc::ptr_eq(&epoch, &cache.epoch) {
                        for (idx, vec) in embedded.iter().enumerate() {
                            cache.insert(miss_texts[idx].to_owned(), vec.clone());
                        }
                    }
                }
                if all_slots_are_distinct_misses {
                    return Ok(embedded);
                }
                Self::fill_batch_misses(&mut out, &slot_miss, embedded);
            }
            cache_checkpoint(cx)?;
            Ok(out
                .into_iter()
                .map(|v| v.expect("every slot filled"))
                .collect())
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.inner.identity()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        self.inner.supports_mrl()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::traits::l2_normalize;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Test double: counts how many times `embed()` is called.
    struct CountingEmbedder {
        dim: usize,
        calls: AtomicUsize,
        identity: EmbeddingIdentityBundleV1,
        bound_calls: AtomicUsize,
        bound_response_identity: Option<EmbeddingIdentityBundleV1>,
    }

    impl CountingEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
                identity: EmbeddingIdentityBundleV1::explicit_test_model(
                    "counting-test",
                    u32::try_from(dim).unwrap_or(u32::MAX),
                ),
                bound_calls: AtomicUsize::new(0),
                bound_response_identity: None,
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl Embedder for CountingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let mut vec = vec![0.0_f32; self.dim];
            // Deterministic: use text length to seed a simple pattern
            for (i, b) in text.bytes().enumerate() {
                vec[i % self.dim] += f32::from(b);
            }
            let normalized = l2_normalize(&vec);
            Box::pin(async move { Ok(normalized) })
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            self.bound_calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                let bound = IdentityBoundEmbedding {
                    values: self.embed(cx, text).await?,
                    identity: self
                        .bound_response_identity
                        .as_ref()
                        .unwrap_or(&self.identity)
                        .clone(),
                };
                bound.validate()?;
                Ok(bound)
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn id(&self) -> &'static str {
            "counting-test"
        }

        fn model_name(&self) -> &'static str {
            "Counting Test Embedder"
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    /// Counts `embed` vs `embed_batch` inner calls separately, so a test can prove
    /// `CachedEmbedder::embed_batch` funnels misses through ONE `inner.embed_batch`
    /// (not N `inner.embed`), which is the whole point when the inner batches.
    struct BatchCountingEmbedder {
        dim: usize,
        embed_calls: AtomicUsize,
        batch_calls: AtomicUsize,
        identity: EmbeddingIdentityBundleV1,
    }

    impl BatchCountingEmbedder {
        fn deterministic(dim: usize, text: &str) -> Vec<f32> {
            let mut vec = vec![0.0_f32; dim];
            for (i, b) in text.bytes().enumerate() {
                vec[i % dim] += f32::from(b);
            }
            l2_normalize(&vec)
        }
    }

    impl Embedder for BatchCountingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.embed_calls.fetch_add(1, Ordering::Relaxed);
            let v = Self::deterministic(self.dim, text);
            Box::pin(async move { Ok(v) })
        }

        fn embed_batch<'a>(
            &'a self,
            _cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<Vec<f32>>> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            let out: Vec<Vec<f32>> = texts
                .iter()
                .map(|t| Self::deterministic(self.dim, t))
                .collect();
            Box::pin(async move { Ok(out) })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn dimension(&self) -> usize {
            self.dim
        }
        fn id(&self) -> &'static str {
            "batch-counting-test"
        }
        fn model_name(&self) -> &'static str {
            "Batch Counting Test Embedder"
        }
        fn is_semantic(&self) -> bool {
            false
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    #[test]
    fn embed_batch_funnels_misses_through_one_inner_embed_batch() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(BatchCountingEmbedder {
                dim: 64,
                embed_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
                identity: EmbeddingIdentityBundleV1::explicit_test_model("batch-counting-test", 64),
            });
            let cached = CachedEmbedder::new(inner.clone(), 128);
            let texts = ["a", "bb", "ccc", "dddd", "eeeee"];
            let refs: Vec<&str> = texts.to_vec();

            let out = cached.embed_batch(&cx, &refs).await.expect("embed_batch");

            // The win: all misses go through ONE inner.embed_batch, zero inner.embed.
            assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
            assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 0);
            // Correctness: same vectors the direct (uncached) embed_batch produces.
            for (i, t) in texts.iter().enumerate() {
                assert_eq!(out[i], BatchCountingEmbedder::deterministic(64, t));
            }

            // A second call is fully cache-served: no further inner work.
            let out2 = cached.embed_batch(&cx, &refs).await.expect("embed_batch 2");
            assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
            assert_eq!(inner.embed_calls.load(Ordering::Relaxed), 0);
            assert_eq!(out, out2);
        });
    }

    fn make_cached(capacity: usize) -> (CachedEmbedder, Arc<CountingEmbedder>) {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner.clone(), capacity);
        (cached, inner)
    }

    #[test]
    fn cache_wrapper_forwards_complete_identity_exactly() {
        let (cached, inner) = make_cached(16);
        assert_eq!(cached.identity().unwrap(), inner.identity().unwrap());
    }

    #[test]
    fn bound_embedding_bypasses_warm_raw_cache_and_rejects_foreign_response_identity() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let response_identity =
                EmbeddingIdentityBundleV1::explicit_test_model("foreign-response", 64);
            let mut inner = CountingEmbedder::new(64);
            inner.bound_response_identity = Some(response_identity.clone());
            let inner = Arc::new(inner);
            let cached = CachedEmbedder::new(inner.clone(), 16);
            assert_ne!(cached.identity().unwrap(), &response_identity);

            let raw = cached.embed(&cx, "query").await.unwrap();
            assert_eq!(cached.embed(&cx, "query").await.unwrap(), raw);
            assert_eq!(inner.call_count(), 1);
            let warmed_stats = cached.cache_stats();

            for _ in 0..2 {
                assert!(matches!(
                    cached.embed_bound(&cx, "query").await,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
            }
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            assert_eq!(inner.call_count(), 3);
            assert_eq!(cached.cache_stats().entries, warmed_stats.entries);
            assert_eq!(cached.cache_stats().hits, warmed_stats.hits);
            assert_eq!(cached.cache_stats().misses, warmed_stats.misses + 2);
            assert!(cached.state_lock().bound_entries.is_empty());

            assert_eq!(cached.embed(&cx, "query").await.unwrap(), raw);
            assert_eq!(inner.call_count(), 3);
            assert_eq!(cached.cache_stats().hits, warmed_stats.hits + 1);
        });
    }

    #[test]
    fn cache_hit_avoids_inner_call() {
        let (cached, inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let v1 = cached.embed(&cx, "hello world").await.unwrap();
            let v2 = cached.embed(&cx, "hello world").await.unwrap();
            assert_eq!(v1, v2);
            assert_eq!(inner.call_count(), 1);
        });
    }

    #[test]
    fn cache_miss_calls_inner() {
        let (cached, inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "query a").await.unwrap();
            cached.embed(&cx, "query b").await.unwrap();
            assert_eq!(inner.call_count(), 2);
        });
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let (cached, _inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "alpha").await.unwrap();
            cached.embed(&cx, "alpha").await.unwrap();
            cached.embed(&cx, "beta").await.unwrap();
            let stats = cached.cache_stats();
            assert_eq!(stats.misses, 2);
            assert_eq!(stats.hits, 1);
            assert_eq!(stats.entries, 2);
        });
    }

    #[test]
    fn eviction_at_capacity() {
        let (cached, inner) = make_cached(2);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "first").await.unwrap();
            cached.embed(&cx, "second").await.unwrap();
            // Cache full (2 entries), all freq 0 → S3-FIFO evicts the oldest Small
            // entry ("first") on the third insert.
            cached.embed(&cx, "third").await.unwrap();
            assert_eq!(inner.call_count(), 3);

            // "first" was evicted → miss; re-inserting it evicts "second".
            cached.embed(&cx, "first").await.unwrap();
            assert_eq!(inner.call_count(), 4);

            // "third" is still cached → hit (no inner call).
            cached.embed(&cx, "third").await.unwrap();
            assert_eq!(inner.call_count(), 4);
        });
    }

    #[test]
    fn s3fifo_keeps_reused_key_through_scan() {
        // The S3-FIFO win over plain FIFO: a key re-requested while resident is
        // promoted to Main and survives a scan of cold one-hit-wonders that overflows
        // the cache — a FIFO would have evicted it by insertion order.
        let (cached, inner) = make_cached(4);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "hot").await.unwrap(); // miss → Small
            cached.embed(&cx, "hot").await.unwrap(); // hit → freq++ (promotable)
            assert_eq!(inner.call_count(), 1);
            let after_hot = inner.call_count();

            // Scan: 6 unique cold keys through a capacity-4 cache.
            for i in 0..6 {
                cached.embed(&cx, &format!("cold-{i}")).await.unwrap();
            }
            assert_eq!(inner.call_count(), after_hot + 6); // all cold = misses

            // "hot" survived the scan (promoted to Main) → still a hit.
            cached.embed(&cx, "hot").await.unwrap();
            assert_eq!(
                inner.call_count(),
                after_hot + 6,
                "S3-FIFO must keep the reused 'hot' key through the cold scan"
            );
        });
    }

    #[test]
    fn clear_resets_stats_and_entries() {
        let (cached, _inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "test").await.unwrap();
            assert_eq!(cached.cache_stats().entries, 1);

            cached.clear_cache();
            let stats = cached.cache_stats();
            assert_eq!(stats.entries, 0);
            assert_eq!(stats.hits, 0);
            assert_eq!(stats.misses, 0);
        });
    }

    #[test]
    fn delegates_trait_methods_to_inner() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner, 16);

        assert_eq!(cached.dimension(), 64);
        assert_eq!(cached.id(), "counting-test");
        assert_eq!(cached.model_name(), "Counting Test Embedder");
        assert!(!cached.is_semantic());
        assert_eq!(cached.category(), ModelCategory::HashEmbedder);
    }

    #[test]
    fn with_default_capacity_uses_128() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::with_default_capacity(inner);
        assert_eq!(cached.cache_stats().capacity, 128);
    }

    #[test]
    fn debug_format_includes_stats() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner, 16);
        let dbg = format!("{cached:?}");
        assert!(dbg.contains("CachedEmbedder"));
        assert!(dbg.contains("counting-test"));
    }

    #[test]
    fn embed_batch_uses_per_item_cache() {
        let (cached, inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // Pre-warm "alpha" into cache
            cached.embed(&cx, "alpha").await.unwrap();
            assert_eq!(inner.call_count(), 1);

            // Batch with "alpha" (cached) and "beta" (miss)
            let batch = cached.embed_batch(&cx, &["alpha", "beta"]).await.unwrap();
            assert_eq!(batch.len(), 2);
            // Only "beta" should have triggered an inner call
            assert_eq!(inner.call_count(), 2);
        });
    }

    #[test]
    fn duplicate_insert_is_idempotent() {
        let (cached, inner) = make_cached(4);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "same").await.unwrap();
            assert_eq!(inner.call_count(), 1);
            assert_eq!(cached.cache_stats().entries, 1);
            // Re-embed same query — should be a cache hit, not a duplicate insert
            cached.embed(&cx, "same").await.unwrap();
            assert_eq!(inner.call_count(), 1);
            assert_eq!(cached.cache_stats().entries, 1);
        });
    }

    // ─── bd-1ocg tests begin ───

    #[test]
    fn cache_stats_debug_clone_copy_eq() {
        let stats = CacheStats {
            hits: 5,
            misses: 3,
            entries: 8,
            capacity: 128,
        };
        let copied = stats; // Copy
        let cloned = { stats }; // Clone trait is available (Copy implies Clone)
        assert_eq!(stats, copied);
        assert_eq!(stats, cloned);

        let different = CacheStats {
            hits: 0,
            misses: 0,
            entries: 0,
            capacity: 128,
        };
        assert_ne!(stats, different);

        let dbg = format!("{stats:?}");
        assert!(dbg.contains("CacheStats"));
        assert!(dbg.contains("hits: 5"));
    }

    #[test]
    fn capacity_one_evicts_immediately() {
        let (cached, inner) = make_cached(1);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "first").await.unwrap();
            assert_eq!(cached.cache_stats().entries, 1);

            // Second insert evicts "first"
            cached.embed(&cx, "second").await.unwrap();
            assert_eq!(inner.call_count(), 2);
            assert_eq!(cached.cache_stats().entries, 1);

            // "first" is evicted, so it's a miss
            cached.embed(&cx, "first").await.unwrap();
            assert_eq!(inner.call_count(), 3);

            // "second" was evicted by "first" re-insert
            cached.embed(&cx, "second").await.unwrap();
            assert_eq!(inner.call_count(), 4);
        });
    }

    #[test]
    fn inner_accessor_returns_same_embedder() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner, 16);
        assert_eq!(cached.inner().id(), "counting-test");
        assert_eq!(cached.inner().dimension(), 64);
        assert_eq!(cached.inner().model_name(), "Counting Test Embedder");
    }

    #[test]
    fn is_ready_delegates() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner, 16);
        // CountingEmbedder uses default is_ready() which returns true
        assert!(cached.is_ready());
    }

    #[test]
    fn tier_delegates() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner, 16);
        // CountingEmbedder uses default tier() which returns ModelTier::Fast
        assert_eq!(cached.tier(), ModelTier::Fast);
    }

    #[test]
    fn supports_mrl_delegates() {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner, 16);
        // CountingEmbedder uses default supports_mrl() which returns false
        assert!(!cached.supports_mrl());
    }

    #[test]
    fn clear_then_reuse_resets_everything() {
        let (cached, inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "alpha").await.unwrap();
            cached.embed(&cx, "alpha").await.unwrap(); // hit
            assert_eq!(cached.cache_stats().hits, 1);
            assert_eq!(cached.cache_stats().misses, 1);

            cached.clear_cache();
            assert_eq!(cached.cache_stats().hits, 0);
            assert_eq!(cached.cache_stats().misses, 0);
            assert_eq!(cached.cache_stats().entries, 0);

            // After clear, "alpha" is a miss again
            cached.embed(&cx, "alpha").await.unwrap();
            assert_eq!(inner.call_count(), 2); // called again
            assert_eq!(cached.cache_stats().misses, 1);
            assert_eq!(cached.cache_stats().entries, 1);
        });
    }

    #[test]
    fn sequential_evictions_maintain_fifo_order() {
        let (cached, inner) = make_cached(3);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // Fill cache: a, b, c
            cached.embed(&cx, "a").await.unwrap();
            cached.embed(&cx, "b").await.unwrap();
            cached.embed(&cx, "c").await.unwrap();
            assert_eq!(inner.call_count(), 3);
            assert_eq!(cached.cache_stats().entries, 3);

            // Insert d -> evicts a (FIFO)
            cached.embed(&cx, "d").await.unwrap();
            assert_eq!(inner.call_count(), 4);

            // a is evicted (miss), b is still cached (hit)
            cached.embed(&cx, "a").await.unwrap();
            assert_eq!(inner.call_count(), 5); // miss
            cached.embed(&cx, "b").await.unwrap();
            // b was evicted when d was added (b was 2nd oldest after a was evicted,
            // then a was re-added evicting b)
            // Actually let's check: after d inserted, cache = [b, c, d]
            // Then a inserted -> evicts b, cache = [c, d, a]
            // So b should be a miss
            assert_eq!(inner.call_count(), 6); // b is a miss
        });
    }

    #[test]
    fn empty_string_embedding() {
        let (cached, inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let v1 = cached.embed(&cx, "").await.unwrap();
            let v2 = cached.embed(&cx, "").await.unwrap();
            assert_eq!(v1, v2);
            assert_eq!(inner.call_count(), 1); // second is cache hit
        });
    }

    #[test]
    fn stats_entries_accurate_after_evictions() {
        let (cached, _inner) = make_cached(2);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "x").await.unwrap();
            cached.embed(&cx, "y").await.unwrap();
            assert_eq!(cached.cache_stats().entries, 2);

            // Evict x, add z
            cached.embed(&cx, "z").await.unwrap();
            assert_eq!(cached.cache_stats().entries, 2); // stays at capacity

            // Evict y, add w
            cached.embed(&cx, "w").await.unwrap();
            assert_eq!(cached.cache_stats().entries, 2);
        });
    }

    #[test]
    fn debug_format_after_operations() {
        let (cached, _inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "test").await.unwrap();
            cached.embed(&cx, "test").await.unwrap(); // hit
            let dbg = format!("{cached:?}");
            assert!(dbg.contains("hits"));
            assert!(dbg.contains("misses"));
            assert!(dbg.contains("entries"));
            assert!(dbg.contains("capacity"));
        });
    }

    #[test]
    fn embed_batch_empty_input() {
        let (cached, _inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let empty: &[&str] = &[];
            let result = cached.embed_batch(&cx, empty).await.unwrap();
            assert!(result.is_empty());
            assert_eq!(cached.cache_stats().entries, 0);
        });
    }

    #[test]
    fn embed_batch_deduplicates_within_batch() {
        let (cached, inner) = make_cached(16);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // Batch with duplicate items: "hello" appears twice, "world" once
            let batch = cached
                .embed_batch(&cx, &["hello", "hello", "world"])
                .await
                .unwrap();
            assert_eq!(batch.len(), 3);
            // Only 2 unique texts → 2 inner calls (second "hello" hits cache)
            assert_eq!(inner.call_count(), 2);
            // Both "hello" embeddings should be identical
            assert_eq!(batch[0], batch[1]);
            // "world" should differ
            assert_ne!(batch[0], batch[2]);
            // Stats: 1 hit (second "hello"), 2 misses (first "hello" + "world")
            assert_eq!(cached.cache_stats().hits, 1);
            assert_eq!(cached.cache_stats().misses, 2);
        });
    }

    // ─── bd-1ocg tests end ───

    struct ResponseEmbedder {
        response: Mutex<Vec<Vec<f32>>>,
        calls: AtomicUsize,
        identity: EmbeddingIdentityBundleV1,
        yield_response: bool,
        cancel_response: AtomicBool,
    }

    impl ResponseEmbedder {
        fn new() -> Self {
            Self {
                response: Mutex::new(vec![vec![1.0, 0.0]]),
                calls: AtomicUsize::new(0),
                identity: EmbeddingIdentityBundleV1::explicit_test_model("cache-response-test", 2),
                yield_response: false,
                cancel_response: AtomicBool::new(false),
            }
        }

        async fn before_response(&self, cx: &Cx) {
            if self.yield_response {
                let mut yielded = false;
                std::future::poll_fn(|task| {
                    if yielded {
                        std::task::Poll::Ready(())
                    } else {
                        yielded = true;
                        task.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
            if self.cancel_response.load(Ordering::Relaxed) {
                cx.set_cancel_requested(true);
            }
        }
    }

    impl Embedder for ResponseEmbedder {
        fn embed<'a>(&'a self, cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let response = self.response.lock().unwrap()[0].clone();
            Box::pin(async move {
                self.before_response(cx).await;
                Ok(response)
            })
        }

        fn embed_batch<'a>(
            &'a self,
            cx: &'a Cx,
            _texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let response = self.response.lock().unwrap().clone();
            Box::pin(async move {
                self.before_response(cx).await;
                Ok(response)
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn dimension(&self) -> usize {
            2
        }

        fn id(&self) -> &'static str {
            "cache-response-test"
        }

        fn model_name(&self) -> &'static str {
            "Cache Response Test Embedder"
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    #[test]
    fn malformed_batch_cardinality_is_an_error_without_partial_cache_admission() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for returned in [0, 1, 3] {
                let inner = Arc::new(ResponseEmbedder::new());
                let cached = CachedEmbedder::new(inner.clone(), 2);
                let warm = cached.embed(&cx, "warm").await.unwrap();
                *inner.response.lock().unwrap() = vec![vec![0.0, 1.0]; returned];
                let error = cached
                    .embed_batch(&cx, &["warm", "first", "second", "first"])
                    .await
                    .unwrap_err();
                assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
                assert_eq!(cached.cache_stats().entries, 1);
                assert!(!cached.state_lock().entries.contains_key("first"));
                assert_eq!(cached.embed(&cx, "warm").await.unwrap(), warm);
                assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
            }
        });
    }

    #[test]
    fn invalid_batch_vector_does_not_cache_the_valid_prefix() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for invalid in [
                vec![],
                vec![1.0],
                vec![1.0, 0.0, 0.0],
                vec![f32::NAN, 0.0],
                vec![0.0, f32::INFINITY],
                vec![f32::NEG_INFINITY, 0.0],
            ] {
                let inner = Arc::new(ResponseEmbedder::new());
                let cached = CachedEmbedder::new(inner.clone(), 2);
                let warm = cached.embed(&cx, "warm").await.unwrap();
                *inner.response.lock().unwrap() = vec![vec![0.0, 1.0], invalid];
                assert!(matches!(
                    cached.embed_batch(&cx, &["first", "second"]).await,
                    Err(SearchError::DimensionMismatch { .. } | SearchError::EmbeddingFailed { .. })
                ));
                assert_eq!(cached.cache_stats().entries, 1);
                assert!(!cached.state_lock().entries.contains_key("first"));
                assert_eq!(cached.embed(&cx, "warm").await.unwrap(), warm);
            }
        });
    }

    #[test]
    fn invalid_single_vectors_are_not_returned_or_cached_and_retry_can_recover() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for invalid in [vec![1.0], vec![f32::NAN, 0.0], vec![0.0, f32::INFINITY]] {
                let inner = Arc::new(ResponseEmbedder::new());
                let cached = CachedEmbedder::new(inner.clone(), 4);
                *inner.response.lock().unwrap() = vec![invalid];
                for _ in 0..2 {
                    assert!(cached.embed(&cx, "query").await.is_err());
                    assert_eq!(cached.cache_stats().entries, 0);
                }
                *inner.response.lock().unwrap() = vec![vec![1.0, 0.0]];
                assert_eq!(cached.embed(&cx, "query").await.unwrap(), vec![1.0, 0.0]);
                assert_eq!(cached.embed(&cx, "query").await.unwrap(), vec![1.0, 0.0]);
                assert_eq!(inner.calls.load(Ordering::Relaxed), 3);
            }
        });
    }

    #[test]
    fn disabling_storage_does_not_disable_provider_response_validation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(ResponseEmbedder::new());
            let cached = CachedEmbedder::new(inner.clone(), 0);
            assert!(matches!(
                cached.embed_batch(&cx, &["first", "second"]).await,
                Err(SearchError::EmbeddingFailed { .. })
            ));
            *inner.response.lock().unwrap() = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
            assert_eq!(
                cached.embed_batch(&cx, &["first", "second"]).await.unwrap(),
                vec![vec![0.0, 0.0], vec![1.0, 0.0]]
            );
            assert_eq!(cached.cache_stats().entries, 0);
        });
    }

    #[test]
    fn unpolled_futures_do_not_touch_cache_or_capture_pre_clear_hits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (cached, inner) = make_cached(4);
            cached.embed(&cx, "warm").await.unwrap();
            let before = cached.cache_stats();
            let hit = cached.embed(&cx, "warm");
            let miss = cached.embed(&cx, "new");
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.call_count(), 1);
            drop(miss);
            cached.clear_cache();
            hit.await.unwrap();
            assert_eq!(inner.call_count(), 2);
            assert_eq!(cached.cache_stats().misses, 1);
            assert_eq!(cached.cache_stats().hits, 0);
        });
    }

    #[test]
    fn clear_fences_in_flight_single_response_and_preserves_new_entries() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for refill in [false, true] {
                let mut provider = ResponseEmbedder::new();
                provider.yield_response = true;
                let inner = Arc::new(provider);
                let cached = CachedEmbedder::new(inner.clone(), 1);
                let mut flight = cached.embed(&cx, "old");
                {
                    let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                    assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
                }
                assert_eq!(inner.calls.load(Ordering::Relaxed), 1);
                cached.clear_cache();
                cached.clear_cache();
                if refill {
                    *inner.response.lock().unwrap() = vec![vec![0.0, 1.0]];
                    cached.embed(&cx, "new").await.unwrap();
                }
                let after_clear = cached.cache_stats();
                assert_eq!(flight.await.unwrap(), vec![1.0, 0.0]);
                assert_eq!(cached.cache_stats(), after_clear);
                assert!(!cached.state_lock().entries.contains_key("old"));
                if refill {
                    assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![0.0, 1.0]);
                    assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
                }
            }
        });
    }

    #[test]
    fn clear_fences_in_flight_batch_without_losing_order_or_duplicates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut provider = ResponseEmbedder::new();
            provider.yield_response = true;
            let inner = Arc::new(provider);
            *inner.response.lock().unwrap() = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
            let cached = CachedEmbedder::new(inner.clone(), 1);
            let texts = ["first", "second", "first"];
            let mut flight = cached.embed_batch(&cx, &texts);
            {
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
            }
            cached.clear_cache();
            *inner.response.lock().unwrap() = vec![vec![0.0, 0.0]];
            cached.embed(&cx, "new").await.unwrap();
            let after_clear = cached.cache_stats();
            assert_eq!(
                flight.await.unwrap(),
                vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]]
            );
            assert_eq!(cached.cache_stats(), after_clear);
            assert!(cached.state_lock().entries.contains_key("new"));
            assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![0.0, 0.0]);
            assert_eq!(inner.calls.load(Ordering::Relaxed), 2);
        });
    }

    #[test]
    fn cancellation_precedes_hits_misses_empty_batches_and_bound_calls() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (cached, inner) = make_cached(4);
            cached.embed(&cx, "warm").await.unwrap();
            let before = cached.cache_stats();
            let constructed_hit = cached.embed(&cx, "warm");
            cx.set_cancel_requested(true);
            assert!(matches!(
                constructed_hit.await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                cached.embed(&cx, "new").await,
                Err(SearchError::Cancelled { .. })
            ));
            for texts in [vec![], vec!["warm"], vec!["warm", "new"]] {
                assert!(matches!(
                    cached.embed_batch(&cx, &texts).await,
                    Err(SearchError::Cancelled { .. })
                ));
            }
            assert!(matches!(
                cached.embed_bound(&cx, "warm").await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.call_count(), 1);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn cancellation_after_provider_success_does_not_admit_single_or_batch_misses() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(ResponseEmbedder::new());
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed(&cx, "warm").await.unwrap();
            inner.cancel_response.store(true, Ordering::Relaxed);
            assert!(matches!(
                cached.embed(&cx, "new").await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(cached.cache_stats().entries, 1);
            cx.set_cancel_requested(false);
            assert!(matches!(
                cached.embed_batch(&cx, &["warm", "new"]).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(cached.cache_stats().entries, 1);
            assert!(!cached.state_lock().entries.contains_key("new"));
            cx.set_cancel_requested(false);
            inner.cancel_response.store(false, Ordering::Relaxed);
            assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![1.0, 0.0]);
            assert_eq!(cached.cache_stats().entries, 2);
        });
    }

    type IdentityHook = (usize, Box<dyn FnOnce() + Send>);

    /// A real suspending bound operation with values independent of the raw API.
    /// Immutable bundles are selected atomically to model a reload at any boundary.
    struct BoundCacheProvider {
        identities: [EmbeddingIdentityBundleV1; 2],
        selected: AtomicUsize,
        dimension: AtomicUsize,
        unavailable: AtomicBool,
        raw_calls: AtomicUsize,
        bound_calls: AtomicUsize,
        identity_calls: AtomicUsize,
        identity_hook: Mutex<Option<IdentityHook>>,
        values: Mutex<Vec<f32>>,
        response_identity: Mutex<Option<EmbeddingIdentityBundleV1>>,
        // 0: success, 1: error, 2: cancelled success, 3: cancelled error.
        action: AtomicUsize,
    }

    impl BoundCacheProvider {
        fn new() -> Self {
            let mut identity =
                EmbeddingIdentityBundleV1::explicit_test_model("bound-query-test", 2);
            // Synthetic semantic authority permits an explicit weights-revision
            // mutation; this fixture makes no claim of real model verification.
            identity.space.kind = EmbeddingSpaceKindV1::Semantic;
            identity.space.hash_control = None;
            identity.space.artifacts.push(EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "c".repeat(64),
                size: 1,
            });
            identity.producer.space_fingerprint = identity.space.fingerprint();
            identity.validate().unwrap();
            let mut replacement = identity.clone();
            replacement
                .producer
                .implementation_revision
                .push_str("-new");
            replacement.validate().unwrap();
            Self {
                identities: [identity, replacement],
                selected: AtomicUsize::new(0),
                dimension: AtomicUsize::new(2),
                unavailable: AtomicBool::new(false),
                raw_calls: AtomicUsize::new(0),
                bound_calls: AtomicUsize::new(0),
                identity_calls: AtomicUsize::new(0),
                identity_hook: Mutex::new(None),
                values: Mutex::new(vec![-0.0, f32::from_bits(1)]),
                response_identity: Mutex::new(None),
                action: AtomicUsize::new(0),
            }
        }

        fn at_identity_read(&self, read: usize, action: impl FnOnce() + Send + 'static) {
            *self.identity_hook.lock().unwrap() = Some((read, Box::new(action)));
        }
    }

    impl Embedder for BoundCacheProvider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.raw_calls.fetch_add(1, Ordering::Relaxed);
                Ok(vec![99.0, 0.0])
            })
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.bound_calls.fetch_add(1, Ordering::Relaxed);
                let response = IdentityBoundEmbedding {
                    values: self.values.lock().unwrap().clone(),
                    identity: self
                        .response_identity
                        .lock()
                        .unwrap()
                        .clone()
                        .unwrap_or_else(|| {
                            self.identities[self.selected.load(Ordering::Relaxed)].clone()
                        }),
                };
                let action = self.action.load(Ordering::Relaxed);
                let mut yielded = false;
                std::future::poll_fn(|task| {
                    if yielded {
                        std::task::Poll::Ready(())
                    } else {
                        yielded = true;
                        task.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
                if action >= 2 {
                    cx.set_cancel_requested(true);
                }
                if action % 2 == 1 {
                    return Err(SearchError::EmbeddingFailed {
                        model: "bound-query-test".to_owned(),
                        source: "bound provider failed".into(),
                    });
                }
                Ok(response)
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            let read = self.identity_calls.fetch_add(1, Ordering::Relaxed) + 1;
            let hook = {
                let mut hook = self.identity_hook.lock().unwrap();
                if hook.as_ref().is_some_and(|(at, _)| *at == read) {
                    hook.take()
                } else {
                    None
                }
            };
            if let Some((_, hook)) = hook {
                hook();
            }
            if self.unavailable.load(Ordering::Relaxed) {
                return Err(SearchError::InvalidConfig {
                    field: "private-canary".to_owned(),
                    value: "private-canary".to_owned(),
                    reason: "private-canary".to_owned(),
                });
            }
            Ok(&self.identities[self.selected.load(Ordering::Relaxed)])
        }

        fn dimension(&self) -> usize {
            self.dimension.load(Ordering::Relaxed)
        }

        fn id(&self) -> &'static str {
            "bound-query-test"
        }

        fn model_name(&self) -> &'static str {
            "Bound Query Test Provider"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    #[test]
    fn bound_query_cache_retains_actual_bits_and_cannot_alias_raw_queries() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = CachedEmbedder::new(inner.clone(), 4);
            // NULs and identity-looking text are still ordinary raw queries.
            let query = format!("\0bound:{}:query", inner.identities[0].fingerprint());
            assert_eq!(cached.embed(&cx, &query).await.unwrap(), vec![99.0, 0.0]);
            let response = cached.embed_bound(&cx, &query).await.unwrap();
            *inner.values.lock().unwrap() = vec![1.0, 0.0];
            let hit = cached.embed_bound(&cx, &query).await.unwrap();
            assert_eq!(hit.identity, response.identity);
            assert_eq!(hit.identity, inner.identities[0]);
            assert_eq!(hit.values[0].to_bits(), (-0.0_f32).to_bits());
            assert_eq!(hit.values[1].to_bits(), 1);
            assert_eq!(cached.embed(&cx, &query).await.unwrap(), vec![99.0, 0.0]);
            assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
            assert_eq!(cached.cache_stats().entries, 2);
            assert_eq!(cached.cache_stats().hits, 2);
            assert_eq!(cached.cache_stats().misses, 2);
        });
    }

    #[test]
    fn bound_query_keys_include_every_contract_for_same_name_and_dimension() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for component in 0..6 {
                let mut provider = BoundCacheProvider::new();
                provider.identities[1] = provider.identities[0].clone();
                let changed = &mut provider.identities[1];
                match component {
                    0 => changed.space.immutable_revision.push_str("-new"),
                    1 => changed.space.artifacts[0].sha256 = "a".repeat(64),
                    2 => changed.space.tokenizer_fingerprint = "b".repeat(64),
                    3 => changed.input.canonicalization.push_str("-new"),
                    4 => changed.producer.implementation_revision.push_str("-new"),
                    _ => changed.storage.format.push_str("-new"),
                }
                changed.space.input_contract_fingerprint = changed.input.fingerprint();
                changed.producer.space_fingerprint = changed.space.fingerprint();
                changed.validate().unwrap();
                assert_eq!(
                    provider.identities[1].space.logical_model_id,
                    provider.identities[0].space.logical_model_id
                );
                let inner = Arc::new(provider);
                let cached = CachedEmbedder::new(inner.clone(), 4);
                let original = cached.embed_bound(&cx, "query").await.unwrap();
                inner.selected.store(1, Ordering::Relaxed);
                *inner.values.lock().unwrap() = vec![0.0, 1.0];
                let replacement = cached.embed_bound(&cx, "query").await.unwrap();
                assert_eq!(replacement.values, vec![0.0, 1.0]);
                assert_eq!(replacement.identity, inner.identities[1]);
                assert_ne!(replacement.identity, original.identity);
                assert_eq!(cached.embed_bound(&cx, "query").await.unwrap(), replacement);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
                assert_eq!(cached.cache_stats().entries, 2);
            }
        });
    }

    #[test]
    fn bound_invalid_producer_contract_never_reads_a_warm_cache_or_dispatches() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for invalid in 0..6 {
                let mut provider = BoundCacheProvider::new();
                provider.identities[1] = provider.identities[0].clone();
                match invalid {
                    0 => provider.identities[1].space.dimension = 0,
                    1 => {
                        "fsvi-v2".clone_into(&mut provider.identities[1].storage.format);
                        "little-endian".clone_into(&mut provider.identities[1].storage.endianness);
                    }
                    2 => provider.identities[1].storage.quantization = QuantizationFormat::F16,
                    3 => {
                        "private-canary".clone_into(&mut provider.identities[1].storage.endianness)
                    }
                    _ => {}
                }
                let inner = Arc::new(provider);
                let cached = CachedEmbedder::new(inner.clone(), 4);
                let original = cached.embed_bound(&cx, "query").await.unwrap();
                let before = cached.cache_stats();
                inner.selected.store(1, Ordering::Relaxed);
                if invalid == 4 {
                    inner.dimension.store(3, Ordering::Relaxed);
                } else if invalid == 5 {
                    inner.unavailable.store(true, Ordering::Relaxed);
                }
                let error = cached.embed_bound(&cx, "query").await.unwrap_err();
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                assert!(!error.to_string().contains("private-canary"));
                assert_eq!(cached.cache_stats(), before);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
                inner.selected.store(0, Ordering::Relaxed);
                inner.dimension.store(2, Ordering::Relaxed);
                inner.unavailable.store(false, Ordering::Relaxed);
                assert_eq!(cached.embed_bound(&cx, "query").await.unwrap(), original);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
            }
        });
    }

    #[test]
    fn bound_foreign_and_malformed_responses_never_pollute_cache_and_retry_recovers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for invalid in 0..5 {
                let inner = Arc::new(BoundCacheProvider::new());
                match invalid {
                    0 => {
                        *inner.response_identity.lock().unwrap() = Some(inner.identities[1].clone())
                    }
                    1 => {
                        let mut malformed = inner.identities[0].clone();
                        "private-canary".clone_into(&mut malformed.storage.format);
                        *inner.response_identity.lock().unwrap() = Some(malformed);
                    }
                    2 => *inner.values.lock().unwrap() = vec![1.0],
                    3 => *inner.values.lock().unwrap() = vec![f32::NAN, 0.0],
                    _ => *inner.values.lock().unwrap() = vec![0.0, f32::INFINITY],
                }
                let cached = CachedEmbedder::new(inner.clone(), 4);
                cached.embed(&cx, "warm").await.unwrap();
                for _ in 0..2 {
                    let error = cached.embed_bound(&cx, "warm").await.unwrap_err();
                    if invalid < 2 {
                        assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                    } else {
                        assert!(matches!(error, SearchError::InvalidConfig { .. }));
                    }
                    assert!(!error.to_string().contains("private-canary"));
                    assert_eq!(cached.cache_stats().entries, 1);
                }
                *inner.response_identity.lock().unwrap() = None;
                *inner.values.lock().unwrap() = vec![0.0, 1.0];
                let response = cached.embed_bound(&cx, "warm").await.unwrap();
                assert_eq!(response.values, vec![0.0, 1.0]);
                assert_eq!(cached.embed_bound(&cx, "warm").await.unwrap(), response);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 3);
                assert_eq!(cached.cache_stats().entries, 2);
            }
        });
    }

    #[test]
    fn bound_inflight_producer_drift_precedes_late_success_and_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for action in [0, 1] {
                let inner = Arc::new(BoundCacheProvider::new());
                inner.action.store(action, Ordering::Relaxed);
                let cached = CachedEmbedder::new(inner.clone(), 4);
                let mut flight = cached.embed_bound(&cx, "query");
                {
                    let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                    assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
                }
                inner.selected.store(1, Ordering::Relaxed);
                assert!(matches!(
                    flight.await,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
                assert_eq!(cached.cache_stats().entries, 0);
                inner.action.store(0, Ordering::Relaxed);
                let response = cached.embed_bound(&cx, "query").await.unwrap();
                assert_eq!(response.identity, inner.identities[1]);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            }
        });
    }

    #[test]
    fn bound_inflight_cancellation_precedes_success_and_provider_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for action in [2, 3] {
                let inner = Arc::new(BoundCacheProvider::new());
                inner.action.store(action, Ordering::Relaxed);
                let cached = CachedEmbedder::new(inner.clone(), 4);
                cached.embed(&cx, "warm").await.unwrap();
                assert!(matches!(
                    cached.embed_bound(&cx, "warm").await,
                    Err(SearchError::Cancelled { .. })
                ));
                assert_eq!(cached.cache_stats().entries, 1);
                assert!(cached.state_lock().bound_entries.is_empty());
                cx.set_cancel_requested(false);
                inner.action.store(0, Ordering::Relaxed);
                let response = cached.embed_bound(&cx, "warm").await.unwrap();
                assert_eq!(cached.embed_bound(&cx, "warm").await.unwrap(), response);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            }
        });
    }

    #[test]
    fn bound_clear_fences_suspended_fill_and_preserves_new_raw_or_bound_entry() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for bound_refill in [false, true] {
                let inner = Arc::new(BoundCacheProvider::new());
                let cached = CachedEmbedder::new(inner.clone(), 1);
                let mut flight = cached.embed_bound(&cx, "old");
                {
                    let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                    assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
                }
                cached.clear_cache();
                cached.clear_cache();
                *inner.values.lock().unwrap() = vec![0.0, 1.0];
                if bound_refill {
                    cached.embed_bound(&cx, "new").await.unwrap();
                } else {
                    cached.embed(&cx, "new").await.unwrap();
                }
                let after_clear = cached.cache_stats();
                let old = flight.await.unwrap();
                assert_eq!(old.values[0].to_bits(), (-0.0_f32).to_bits());
                assert_eq!(old.values[1].to_bits(), 1);
                assert_eq!(cached.cache_stats(), after_clear);
                if bound_refill {
                    assert_eq!(
                        cached.embed_bound(&cx, "new").await.unwrap().values,
                        vec![0.0, 1.0]
                    );
                    assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
                } else {
                    assert_eq!(cached.embed(&cx, "new").await.unwrap(), vec![99.0, 0.0]);
                    assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
                }
                cached.embed_bound(&cx, "old").await.unwrap();
                assert_eq!(
                    inner.bound_calls.load(Ordering::Relaxed),
                    if bound_refill { 3 } else { 2 }
                );
                assert_eq!(cached.cache_stats().entries, 1);
            }
        });
    }

    #[test]
    fn bound_hit_rechecks_producer_before_serving_and_counts_only_valid_hits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed_bound(&cx, "warm").await.unwrap();
            let before = cached.cache_stats();
            let provider = Arc::downgrade(&inner);
            inner.at_identity_read(4, move || {
                provider
                    .upgrade()
                    .unwrap()
                    .selected
                    .store(1, Ordering::Relaxed);
            });
            assert!(matches!(
                cached.embed_bound(&cx, "warm").await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
            assert_eq!(
                cached.embed_bound(&cx, "warm").await.unwrap().identity,
                inner.identities[1]
            );
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
        });
    }

    #[test]
    fn bound_identity_callback_can_clear_cache_without_deadlock_or_reset_stat_mutation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for during_hit in [false, true] {
                let inner = Arc::new(BoundCacheProvider::new());
                let cached = Arc::new(CachedEmbedder::new(inner.clone(), 4));
                if during_hit {
                    cached.embed_bound(&cx, "warm").await.unwrap();
                }
                let cache = Arc::downgrade(&cached);
                inner.at_identity_read(if during_hit { 4 } else { 2 }, move || {
                    cache.upgrade().unwrap().clear_cache();
                });
                cached.embed_bound(&cx, "warm").await.unwrap();
                assert_eq!(
                    cached.cache_stats(),
                    CacheStats {
                        hits: 0,
                        misses: 0,
                        entries: 0,
                        capacity: 4
                    }
                );
                cached.embed_bound(&cx, "warm").await.unwrap();
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
                assert_eq!(cached.cache_stats().entries, 1);
            }
        });
    }

    #[test]
    fn bound_cancellation_in_identity_callback_dominates_drift_or_identity_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for boundary in [1, 2, 4] {
                for unavailable in [false, true] {
                    let inner = Arc::new(BoundCacheProvider::new());
                    let cached = CachedEmbedder::new(inner.clone(), 4);
                    if boundary == 4 {
                        cached.embed_bound(&cx, "warm").await.unwrap();
                    }
                    let before = cached.cache_stats();
                    let provider = Arc::downgrade(&inner);
                    let cancel_cx = cx.clone();
                    inner.at_identity_read(boundary, move || {
                        let provider = provider.upgrade().unwrap();
                        provider.selected.store(1, Ordering::Relaxed);
                        provider.unavailable.store(unavailable, Ordering::Relaxed);
                        cancel_cx.set_cancel_requested(true);
                    });
                    assert!(matches!(
                        cached.embed_bound(&cx, "warm").await,
                        Err(SearchError::Cancelled { .. })
                    ));
                    assert_eq!(cached.cache_stats().entries, before.entries);
                    assert_eq!(cached.cache_stats().hits, before.hits);
                    assert_eq!(
                        inner.bound_calls.load(Ordering::Relaxed),
                        usize::from(boundary != 1)
                    );
                    cx.set_cancel_requested(false);
                }
            }
        });
    }

    #[test]
    fn bound_unpolled_future_is_lazy_and_resolves_again_after_clear() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed_bound(&cx, "warm").await.unwrap();
            let before = cached.cache_stats();
            let reads = inner.identity_calls.load(Ordering::Relaxed);
            let future = cached.embed_bound(&cx, "warm");
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.identity_calls.load(Ordering::Relaxed), reads);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
            cached.clear_cache();
            *inner.values.lock().unwrap() = vec![1.0, 0.0];
            assert_eq!(future.await.unwrap().values, vec![1.0, 0.0]);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 2);
            assert_eq!(cached.cache_stats().hits, 0);
            assert_eq!(cached.cache_stats().misses, 1);
        });
    }

    #[test]
    fn bound_and_raw_entries_share_capacity_and_zero_capacity_still_validates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for capacity in [0, 1] {
                let inner = Arc::new(BoundCacheProvider::new());
                let cached = CachedEmbedder::new(inner.clone(), capacity);
                for index in 0..3 {
                    let query = format!("query-{index}");
                    cached.embed(&cx, &query).await.unwrap();
                    assert_eq!(cached.cache_stats().entries, capacity);
                    cached.embed_bound(&cx, &query).await.unwrap();
                    assert_eq!(cached.cache_stats().entries, capacity);
                }
                assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 3);
                assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 3);
                let state = cached.state_lock();
                assert!(state.ghost.len() <= capacity);
                assert!(state.ghost_set.len() <= capacity);
                drop(state);
                cached.clear_cache();
                *inner.values.lock().unwrap() = vec![f32::NAN, 0.0];
                assert!(matches!(
                    cached.embed_bound(&cx, "query").await,
                    Err(SearchError::InvalidConfig { .. })
                ));
                assert_eq!(cached.cache_stats().entries, 0);
            }
        });
    }

    #[test]
    fn bound_hot_entries_survive_cold_raw_scans_under_shared_s3_fifo() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let inner = Arc::new(BoundCacheProvider::new());
            let cached = CachedEmbedder::new(inner.clone(), 4);
            let hot = cached.embed_bound(&cx, "hot").await.unwrap();
            cached.embed_bound(&cx, "hot").await.unwrap();
            for index in 0..12 {
                cached.embed(&cx, &format!("cold-{index}")).await.unwrap();
                assert!(cached.cache_stats().entries <= 4);
            }
            assert_eq!(cached.embed_bound(&cx, "hot").await.unwrap(), hot);
            assert_eq!(inner.bound_calls.load(Ordering::Relaxed), 1);
            assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 12);
        });
    }
}
