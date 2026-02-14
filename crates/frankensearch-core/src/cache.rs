//! S3-FIFO cache eviction for embedding vectors and search results.
//!
//! Implements the S3-FIFO algorithm (Yang et al., SOSP 2023) using three FIFO
//! queues (Small, Main, Ghost). Unlike LRU, S3-FIFO requires no per-access
//! list manipulation — only a frequency counter increment — making it fast
//! under contention.
//!
//! The cache is thread-safe via `std::sync::Mutex` (lock held only for O(1)
//! hash-table operations, never across await points).

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::sync::Mutex;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Eviction-policy abstraction for frankensearch caching layers.
///
/// Implementations must be `Send + Sync` for use across async tasks and Rayon
/// threads.
pub trait CachePolicy<K, V>: Send + Sync
where
    K: Hash + Eq,
{
    /// Look up a cached value. Returns `None` on miss.
    fn get(&self, key: &K) -> Option<V>;

    /// Insert a value with its estimated byte size.
    ///
    /// If the cache is at capacity, one or more entries are evicted first.
    fn insert(&self, key: K, value: V, size_bytes: usize);

    /// Exponential moving average hit rate in `[0.0, 1.0]`.
    fn hit_rate(&self) -> f64;

    /// Approximate memory currently consumed by cached entries (bytes).
    fn memory_used(&self) -> usize;
}

// ─── NoCache ─────────────────────────────────────────────────────────────────

/// Zero-cost passthrough that never caches anything.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCache;

impl<K: Hash + Eq + Send + Sync, V: Send + Sync> CachePolicy<K, V> for NoCache {
    fn get(&self, _key: &K) -> Option<V> {
        None
    }

    fn insert(&self, _key: K, _value: V, _size_bytes: usize) {}

    fn hit_rate(&self) -> f64 {
        0.0
    }

    fn memory_used(&self) -> usize {
        0
    }
}

// ─── S3-FIFO Cache ──────────────────────────────────────────────────────────

/// Configuration for [`S3FifoCache`].
#[derive(Debug, Clone)]
pub struct S3FifoConfig {
    /// Total memory budget in bytes (default: 256 MB).
    pub max_bytes: usize,
    /// Fraction of `max_bytes` allocated to the Small queue (default: 0.10).
    pub small_ratio: f64,
    /// Access-count threshold for promotion from Small to Main (default: 1).
    pub freq_threshold: u8,
    /// EMA smoothing factor for hit rate (default: 0.01).
    pub hit_rate_alpha: f64,
}

impl Default for S3FifoConfig {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            small_ratio: 0.10,
            freq_threshold: 1,
            hit_rate_alpha: 0.01,
        }
    }
}

impl S3FifoConfig {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn max_small_bytes(&self) -> usize {
        (self.max_bytes as f64 * self.small_ratio) as usize
    }

    fn max_main_bytes(&self) -> usize {
        self.max_bytes.saturating_sub(self.max_small_bytes())
    }
}

/// S3-FIFO three-queue cache (Yang et al., SOSP 2023).
///
/// - **Small** (10% capacity): new entries land here.
/// - **Main** (90% capacity): promoted entries with high access frequency.
/// - **Ghost** (metadata-only): tracks recently evicted keys for quick re-admission.
pub struct S3FifoCache<K, V> {
    state: Mutex<CacheState<K, V>>,
    config: S3FifoConfig,
}

impl<K, V> std::fmt::Debug for S3FifoCache<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3FifoCache")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

struct CacheState<K, V> {
    /// Key → entry mapping for O(1) lookup.
    entries: HashMap<K, CacheEntry<V>>,
    /// Small FIFO eviction order (keys, front = oldest).
    small_order: VecDeque<K>,
    /// Main FIFO eviction order (keys, front = oldest).
    main_order: VecDeque<K>,
    /// Ghost set for O(1) membership check of exact keys.
    ghost_set: HashSet<K>,
    /// Ghost FIFO order for bounded eviction.
    ghost_order: VecDeque<K>,

    small_bytes: usize,
    main_bytes: usize,

    /// EMA hit rate.
    hit_rate_ema: f64,
}

struct CacheEntry<V> {
    value: V,
    size_bytes: usize,
    freq: u8,
    location: EntryLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryLocation {
    Small,
    Main,
}

impl<K, V> S3FifoCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    /// Create a new S3-FIFO cache with the given configuration.
    #[must_use]
    pub fn new(config: S3FifoConfig) -> Self {
        Self {
            state: Mutex::new(CacheState {
                entries: HashMap::new(),
                small_order: VecDeque::new(),
                main_order: VecDeque::new(),
                ghost_set: HashSet::new(),
                ghost_order: VecDeque::new(),
                small_bytes: 0,
                main_bytes: 0,
                hit_rate_ema: 0.0,
            }),
            config,
        }
    }

    /// Create a cache with default configuration (256 MB budget).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(S3FifoConfig::default())
    }

    /// Number of entries currently cached (Small + Main).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        let state = self.state.lock().expect("cache lock poisoned");
        state.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K, V> CachePolicy<K, V> for S3FifoCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    fn get(&self, key: &K) -> Option<V> {
        let mut state = self.state.lock().expect("cache lock poisoned");
        if let Some(entry) = state.entries.get_mut(key) {
            entry.freq = entry.freq.saturating_add(1);
            let value = entry.value.clone();
            let decay = 1.0 - self.config.hit_rate_alpha;
            state.hit_rate_ema = decay.mul_add(state.hit_rate_ema, self.config.hit_rate_alpha);
            Some(value)
        } else {
            state.hit_rate_ema *= 1.0 - self.config.hit_rate_alpha;
            None
        }
    }

    fn insert(&self, key: K, value: V, size_bytes: usize) {
        if size_bytes > self.config.max_bytes {
            return; // Single entry exceeds entire budget — skip.
        }

        let mut state = self.state.lock().expect("cache lock poisoned");

        // Update existing entry in-place.
        if let Some(entry) = state.entries.get_mut(&key) {
            let old_size = entry.size_bytes;
            entry.value = value;
            entry.size_bytes = size_bytes;
            match entry.location {
                EntryLocation::Small => {
                    state.small_bytes = state.small_bytes.wrapping_sub(old_size) + size_bytes;
                }
                EntryLocation::Main => {
                    state.main_bytes = state.main_bytes.wrapping_sub(old_size) + size_bytes;
                }
            }
            return;
        }

        // Check ghost: re-accessed evicted key goes directly to Main.
        if state.ghost_set.remove(&key) {
            // Evict from Main if needed to make room.
            evict_main(&mut state, size_bytes, &self.config);
            state.main_order.push_back(key.clone());
            state.entries.insert(
                key,
                CacheEntry {
                    value,
                    size_bytes,
                    freq: 0,
                    location: EntryLocation::Main,
                },
            );
            state.main_bytes += size_bytes;
        } else {
            // New entry → Small queue.
            evict_small(&mut state, size_bytes, &self.config);
            state.small_order.push_back(key.clone());
            state.entries.insert(
                key,
                CacheEntry {
                    value,
                    size_bytes,
                    freq: 0,
                    location: EntryLocation::Small,
                },
            );
            state.small_bytes += size_bytes;
        }
    }

    fn hit_rate(&self) -> f64 {
        let state = self.state.lock().expect("cache lock poisoned");
        state.hit_rate_ema
    }

    fn memory_used(&self) -> usize {
        let state = self.state.lock().expect("cache lock poisoned");
        state.small_bytes + state.main_bytes
    }
}

// ─── Eviction helpers ────────────────────────────────────────────────────────

/// Evict entries from Small until `needed_bytes` fits within the Small budget.
fn evict_small<K, V>(state: &mut CacheState<K, V>, needed_bytes: usize, config: &S3FifoConfig)
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    let max_small = config.max_small_bytes();
    while state.small_bytes + needed_bytes > max_small {
        let Some(evict_key) = state.small_order.pop_front() else {
            break;
        };
        let Some(entry) = state.entries.remove(&evict_key) else {
            continue; // Key was already removed (e.g., updated to Main).
        };
        state.small_bytes = state.small_bytes.saturating_sub(entry.size_bytes);

        if entry.freq >= config.freq_threshold {
            // Promote to Main.
            evict_main(state, entry.size_bytes, config);
            state.main_order.push_back(evict_key.clone());
            state.entries.insert(
                evict_key,
                CacheEntry {
                    value: entry.value,
                    size_bytes: entry.size_bytes,
                    freq: 0,
                    location: EntryLocation::Main,
                },
            );
            state.main_bytes += entry.size_bytes;
        } else {
            // Evict to ghost.
            add_to_ghost(state, evict_key, config);
        }
    }
}

/// Evict entries from Main until `needed_bytes` fits within the Main budget.
fn evict_main<K, V>(state: &mut CacheState<K, V>, needed_bytes: usize, config: &S3FifoConfig)
where
    K: Hash + Eq + Clone,
{
    let max_main = config.max_main_bytes();
    while state.main_bytes + needed_bytes > max_main {
        let Some(evict_key) = state.main_order.pop_front() else {
            break;
        };
        let Some(entry) = state.entries.remove(&evict_key) else {
            continue;
        };
        state.main_bytes = state.main_bytes.saturating_sub(entry.size_bytes);
    }
}

/// Add a key hash to the ghost set, evicting the oldest ghost if at capacity.
fn add_to_ghost<K, V>(state: &mut CacheState<K, V>, key: K, config: &S3FifoConfig)
where
    K: Clone + Eq + Hash,
{
    // Ghost capacity: 2x the estimated number of Main entries.
    // Rough estimate: max_main_bytes / average_entry_size. We use entry count as proxy.
    let max_ghost = (state.main_order.len() + state.small_order.len())
        .saturating_mul(2)
        .max(1024);
    let _ = config; // config.ghost_ratio could be used for refinement.

    while state.ghost_order.len() >= max_ghost {
        if let Some(old_key) = state.ghost_order.pop_front() {
            state.ghost_set.remove(&old_key);
        }
    }
    state.ghost_order.push_back(key.clone());
    state.ghost_set.insert(key);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    fn small_config(max_bytes: usize) -> S3FifoConfig {
        S3FifoConfig {
            max_bytes,
            small_ratio: 0.10,
            freq_threshold: 1,
            hit_rate_alpha: 0.1, // Faster convergence for tests.
        }
    }

    // --- NoCache ---

    #[test]
    fn no_cache_always_misses() {
        let cache = NoCache;
        CachePolicy::<&str, &str>::insert(&cache, "key", "value", 5);
        assert!(CachePolicy::<&str, &str>::get(&cache, &"key").is_none());
        assert!(CachePolicy::<&str, &str>::hit_rate(&cache).abs() < f64::EPSILON);
        assert_eq!(CachePolicy::<&str, &str>::memory_used(&cache), 0);
    }

    // --- S3FifoCache basics ---

    #[test]
    fn insert_and_get() {
        let cache = S3FifoCache::new(small_config(1024));
        cache.insert("key1", "value1", 100);
        assert_eq!(cache.get(&"key1"), Some("value1"));
        assert!(cache.get(&"missing").is_none());
    }

    #[test]
    fn memory_used_tracks_inserts() {
        // Budget must be large enough that Small (10%) fits both entries.
        let cache = S3FifoCache::new(small_config(2048));
        assert_eq!(cache.memory_used(), 0);
        cache.insert("a", "v", 50);
        assert_eq!(cache.memory_used(), 50);
        cache.insert("b", "v", 70);
        assert_eq!(cache.memory_used(), 120);
    }

    #[test]
    fn len_and_is_empty() {
        let cache = S3FifoCache::new(small_config(1024));
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.insert("a", 1, 10);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn update_existing_key() {
        let cache = S3FifoCache::new(small_config(1024));
        cache.insert("key", "old", 50);
        cache.insert("key", "new", 60);
        assert_eq!(cache.get(&"key"), Some("new"));
        assert_eq!(cache.memory_used(), 60); // Updated, not doubled.
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn oversized_entry_skipped() {
        let cache = S3FifoCache::new(small_config(100));
        cache.insert("big", "value", 200);
        assert!(cache.get(&"big").is_none());
        assert_eq!(cache.memory_used(), 0);
    }

    // --- Eviction ---

    #[test]
    fn evicts_from_small_when_budget_exceeded() {
        // max_bytes=200, small=20, main=180
        let cache = S3FifoCache::new(small_config(200));
        // Fill Small (20 bytes budget).
        cache.insert("a", 1, 10);
        cache.insert("b", 2, 10);
        // This should trigger eviction of "a" (no accesses, freq=0 < threshold=1).
        cache.insert("c", 3, 10);

        // "a" should be evicted (freq=0, not promoted).
        assert!(cache.get(&"a").is_none());
        // "b" and "c" should still be present.
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.get(&"c"), Some(3));
    }

    #[test]
    fn promotion_from_small_to_main() {
        // max_bytes=200, small=20, main=180
        let cache = S3FifoCache::new(small_config(200));
        cache.insert("a", 1, 10);

        // Access "a" once — freq becomes 1 (>= threshold).
        let _ = cache.get(&"a");

        // Fill small to force eviction of "a".
        cache.insert("b", 2, 10);
        cache.insert("c", 3, 10);

        // "a" should have been promoted to Main (not evicted).
        assert_eq!(cache.get(&"a"), Some(1));
    }

    #[test]
    fn ghost_readmission_to_main() {
        // max_bytes=200, small=20, main=180
        let cache = S3FifoCache::new(small_config(200));

        // Insert and evict "a" (no accesses, goes to ghost).
        cache.insert("a", 1, 10);
        cache.insert("b", 2, 10);
        cache.insert("c", 3, 10); // "a" evicted from Small → ghost

        assert!(cache.get(&"a").is_none()); // Confirmed evicted.

        // Re-insert "a" — should go directly to Main (ghost hit).
        cache.insert("a", 1, 10);
        assert_eq!(cache.get(&"a"), Some(1));

        // Verify it's in Main: fill+evict Small without affecting "a".
        cache.insert("d", 4, 10);
        cache.insert("e", 5, 10);
        cache.insert("f", 6, 10);
        // "a" should still be in Main.
        assert_eq!(cache.get(&"a"), Some(1));
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CollidingKey(&'static str);

    impl Hash for CollidingKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            // Deliberately force collisions to test ghost identity handling.
            0_u8.hash(state);
        }
    }

    #[test]
    fn ghost_readmission_requires_exact_key_not_hash_collision() {
        // max_bytes=200, small=20, main=180
        let cache = S3FifoCache::new(small_config(200));
        let key_a = CollidingKey("a");
        let key_b = CollidingKey("b");

        // Evict key_a from Small into Ghost.
        cache.insert(key_a.clone(), 1, 10);
        cache.insert(CollidingKey("x"), 2, 10);
        cache.insert(CollidingKey("y"), 3, 10);
        assert!(cache.get(&key_a).is_none());

        // key_b collides in hash-space with key_a, but was never in Ghost.
        // It must be treated as a fresh Small insertion, not a ghost hit.
        cache.insert(key_b.clone(), 10, 10);
        cache.insert(CollidingKey("c"), 11, 10);
        cache.insert(CollidingKey("d"), 12, 10);

        // If key_b were falsely treated as a ghost hit, it would live in Main.
        // Correct behavior keeps it in Small and allows eviction during churn.
        assert!(cache.get(&key_b).is_none());
    }

    // --- Hit rate ---

    #[test]
    fn hit_rate_starts_at_zero() {
        let cache = S3FifoCache::<String, i32>::new(small_config(1024));
        assert!(cache.hit_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_increases_on_hits() {
        let cache = S3FifoCache::new(small_config(1024));
        cache.insert("a", 1, 10);

        // Several hits should increase hit rate.
        for _ in 0..20 {
            let _ = cache.get(&"a");
        }
        assert!(cache.hit_rate() > 0.5);
    }

    #[test]
    fn hit_rate_decreases_on_misses() {
        let cache = S3FifoCache::new(small_config(1024));
        cache.insert("a", 1, 10);

        // One hit.
        let _ = cache.get(&"a");
        let rate_after_hit = cache.hit_rate();

        // Many misses.
        for _ in 0..20 {
            let _ = cache.get(&"missing");
        }
        assert!(cache.hit_rate() < rate_after_hit);
    }

    // --- Thread safety ---

    #[test]
    fn concurrent_get_insert() {
        use std::sync::Arc;

        let cache = Arc::new(S3FifoCache::new(small_config(65536)));
        let mut handles = Vec::new();

        // Spawn writers.
        for t in 0..4 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("t{t}-k{i}");
                    cache.insert(key, i, 16);
                }
            }));
        }

        // Spawn readers.
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for i in 0..100 {
                    let _ = cache.get(&format!("t0-k{i}"));
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        // No panic, no data corruption — cache should contain entries.
        assert!(!cache.is_empty());
    }

    // --- Main eviction ---

    #[test]
    fn main_eviction_when_full() {
        // max_bytes=100, small=10, main=90
        let cache = S3FifoCache::new(small_config(100));

        // Insert and promote 9 entries to Main (each 10 bytes = 90 bytes total).
        for i in 0..9 {
            cache.insert(i, i * 10, 10);
            let _ = cache.get(&i); // Freq → 1, will be promoted.
        }
        // Force eviction of all Small entries into Main.
        for i in 9..18 {
            cache.insert(i, i * 10, 10);
        }

        // Main should have evicted oldest promoted entries to fit newer ones.
        // We don't check exact state — just verify no panic and budget respected.
        assert!(cache.memory_used() <= 100);
    }

    // --- Debug ---

    #[test]
    fn debug_output() {
        let cache = S3FifoCache::<String, i32>::new(S3FifoConfig::default());
        let debug = format!("{cache:?}");
        assert!(debug.contains("S3FifoCache"));
    }

    // --- Edge cases ---

    #[test]
    fn zero_size_entries() {
        let cache = S3FifoCache::new(small_config(100));
        cache.insert("a", 1, 0);
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.memory_used(), 0);
    }

    #[test]
    fn single_byte_budget() {
        let cache = S3FifoCache::new(small_config(1));
        cache.insert("a", 1, 1);
        // Small budget is 0 (10% of 1 = 0), so entry goes to ghost path or is evicted.
        // The exact behavior depends on budget math, but it shouldn't panic.
        let _ = cache.get(&"a");
    }
}
