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

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use asupersync::Cx;
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

struct CacheState {
    map: HashMap<String, Vec<f32>>,
    order: VecDeque<String>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl CacheState {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(vec) = self.map.get(key) {
            self.hits += 1;
            Some(vec.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, key: String, value: Vec<f32>) {
        if self.map.contains_key(&key) {
            return;
        }
        if self.order.len() >= self.capacity
            && let Some(evicted) = self.order.pop_front()
        {
            self.map.remove(&evicted);
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.map.len(),
            capacity: self.capacity,
        }
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
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
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                out.push(self.embed(cx, text).await?);
            }
            Ok(out)
        })
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
    }

    impl CountingEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                calls: AtomicUsize::new(0),
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

    fn make_cached(capacity: usize) -> (CachedEmbedder, Arc<CountingEmbedder>) {
        let inner = Arc::new(CountingEmbedder::new(64));
        let cached = CachedEmbedder::new(inner.clone(), capacity);
        (cached, inner)
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
    fn fifo_eviction_at_capacity() {
        let (cached, inner) = make_cached(2);
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cached.embed(&cx, "first").await.unwrap();
            cached.embed(&cx, "second").await.unwrap();
            // Cache is full (2 entries). Inserting a third evicts "first".
            cached.embed(&cx, "third").await.unwrap();
            assert_eq!(inner.call_count(), 3);

            // "first" was evicted, so this is a miss.
            cached.embed(&cx, "first").await.unwrap();
            assert_eq!(inner.call_count(), 4);

            // "second" was evicted when "first" was re-inserted.
            // "third" should still be cached.
            cached.embed(&cx, "third").await.unwrap();
            assert_eq!(inner.call_count(), 4);
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

    // ─── bd-1ocg tests end ───
}
