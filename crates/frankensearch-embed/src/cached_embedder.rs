//! Caching wrapper for any [`Embedder`] implementation.
//!
//! `CachedEmbedder` sits between the search pipeline and an inner embedder,
//! caching query embeddings so that repeated queries skip inference entirely.
//!
//! The cache uses FIFO eviction with a bounded capacity (default 128 entries).
//! Cache hits return a cloned `Vec<f32>`, which is cheap (~1.5 KiB for 384-dim).
//!
//! # Thread Safety
//!
//! The cache is protected by a `std::sync::Mutex`, keeping the wrapper `Send + Sync`.
//! The lock is held only for the brief `HashMap` lookup/insert — never across an
//! async `.await` boundary.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use frankensearch_core::SearchResult;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{Embedder, ModelCategory, ModelTier, SearchFuture};

/// Default maximum number of cached query embeddings.
const DEFAULT_CAPACITY: usize = 128;

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

struct CacheEntry {
    value: Vec<f32>,
    /// Access frequency, saturating at [`FREQ_CAP`]. Drives S3-FIFO promotion
    /// (Small→Main) and the Main second-chance.
    freq: u8,
}

const FREQ_CAP: u8 = 3;

/// S3-FIFO query-embedding cache (Yang et al., SOSP 2023), entry-count form.
///
/// Three queues over a single entry map: **Small** (new/one-hit-wonder admissions,
/// ~10% of capacity), **Main** (proven-reused, ~90%), and **Ghost** (keys recently
/// evicted from Small, metadata-only). Unlike the previous plain FIFO, a key
/// re-requested while resident is promoted to Main and survives the scan churn that
/// evicts cold one-hit-wonders from Small — measurably fewer embed misses on skewed
/// / scan-heavy query streams (see the `cache_replay` bench + `PERF_LEDGER` 2026-06-29).
/// Lookups borrow `&str` (no per-get allocation); the external `CacheState` API
/// (`get`/`insert`/`stats`/`clear`, entry-count `capacity`, hit/miss counters) is
/// unchanged, so `CachedEmbedder` and `CacheStats` are untouched.
struct CacheState {
    entries: HashMap<String, CacheEntry>,
    small: VecDeque<String>,
    main: VecDeque<String>,
    ghost: VecDeque<String>,
    ghost_set: HashSet<String>,
    capacity: usize,
    small_cap: usize,
    ghost_cap: usize,
    hits: u64,
    misses: u64,
}

impl CacheState {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
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

    fn insert(&mut self, key: String, value: Vec<f32>) {
        // capacity == 0 means caching is disabled.
        if self.capacity == 0 || self.entries.contains_key(&key) {
            return;
        }
        // A key seen recently (in Ghost) re-enters straight into Main; a fresh key
        // starts in Small so a scan of one-hit-wonders can't displace the hot set.
        if self.ghost_set.remove(&key) {
            self.main.push_back(key.clone());
        } else {
            self.small.push_back(key.clone());
        }
        self.entries.insert(key, CacheEntry { value, freq: 0 });
        while self.entries.len() > self.capacity {
            self.evict_one();
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
                if self.entries.get(&k).is_some_and(|e| e.freq > 0) {
                    if let Some(e) = self.entries.get_mut(&k) {
                        e.freq = 0;
                    }
                    self.main.push_back(k); // promote — no slot freed, keep going
                } else {
                    self.entries.remove(&k);
                    self.push_ghost(k);
                    return;
                }
            } else if let Some(k) = self.main.pop_front() {
                if self.entries.get(&k).is_some_and(|e| e.freq > 0) {
                    if let Some(e) = self.entries.get_mut(&k) {
                        e.freq -= 1;
                    }
                    self.main.push_back(k); // second chance — keep going
                } else {
                    self.entries.remove(&k);
                    return;
                }
            } else {
                return; // defensive: both queues drained
            }
        }
    }

    fn push_ghost(&mut self, key: String) {
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
            entries: self.entries.len(),
            capacity: self.capacity,
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
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
/// Intercepts `embed()` calls and returns cached vectors for previously-seen
/// query strings. All other trait methods delegate directly to the inner embedder.
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

    /// Return a snapshot of cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.state_lock().stats()
    }

    /// Clear all cached embeddings and reset statistics.
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
        // Check cache before acquiring any async resources.
        // Lock scope is tiny: just a HashMap lookup.
        let cached = self.state_lock().get(text);
        if let Some(vec) = cached {
            return Box::pin(async move { Ok(vec) });
        }

        let key = text.to_owned();
        Box::pin(async move {
            let vec = self.inner.embed(cx, text).await?;
            // Insert into cache (lock scope: HashMap insert + possible eviction).
            self.state_lock().insert(key, vec.clone());
            Ok(vec)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
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
            {
                let mut cache = self.state_lock();
                for &text in texts {
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
            }
            // ONE batched inner call for all distinct misses — vs the old per-text loop
            // that called `inner.embed` N times, defeating a batching inner (e.g.
            // fastembed embeds the whole batch in a single model invocation). N → 1.
            if !miss_texts.is_empty() {
                let all_slots_are_distinct_misses = miss_texts.len() == texts.len();
                let embedded = self.inner.embed_batch(cx, &miss_texts).await?;
                {
                    let mut cache = self.state_lock();
                    for (idx, vec) in embedded.iter().enumerate() {
                        cache.insert(miss_texts[idx].to_owned(), vec.clone());
                    }
                }
                if all_slots_are_distinct_misses && embedded.len() == miss_texts.len() {
                    return Ok(embedded);
                }
                // Fan the distinct embeddings back out to every slot that needed them.
                // Move the owned inner result into its last output slot; clone only
                // same-batch duplicates that need the vector more than once.
                let mut miss_use_counts = vec![0_usize; embedded.len()];
                for maybe_idx in &slot_miss {
                    if let Some(idx) = *maybe_idx {
                        miss_use_counts[idx] += 1;
                    }
                }
                let mut embedded_slots: Vec<Option<Vec<f32>>> =
                    embedded.into_iter().map(Some).collect();
                for (slot, maybe_idx) in slot_miss.iter().enumerate() {
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
                        out[slot] = Some(vec);
                    }
                }
            }
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
    use frankensearch_core::traits::l2_normalize;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test double: counts how many times `embed()` is called.
    struct CountingEmbedder {
        dim: usize,
        calls: AtomicUsize,
        identity: EmbeddingIdentityBundleV1,
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
}
