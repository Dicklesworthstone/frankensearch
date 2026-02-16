//! Index warm-up and adaptive page prefaulting for memory-mapped FSVI indices.
//!
//! Cold-start latency for mmap'd indices can be 10-100x higher than warm due to
//! page faults. This module provides controlled prefaulting to eliminate cold-start
//! variance.
//!
//! # Strategies
//!
//! - **None**: No prefaulting (default, current behavior).
//! - **Full**: Touch every page sequentially — simple but may waste I/O budget.
//! - **Header**: Prefault only the header and record table (smallest footprint).
//! - **Adaptive**: Heat-map based intelligent prefaulting that learns which pages
//!   are actually accessed during typical queries.
//!
//! # Example
//!
//! ```
//! use frankensearch_index::warmup::{WarmUpConfig, WarmUpStrategy, HeatMap};
//!
//! let config = WarmUpConfig::default();
//! assert!(matches!(config.strategy, WarmUpStrategy::None));
//!
//! let heat_map = HeatMap::new(1_000_000);
//! assert_eq!(heat_map.page_count(), 245); // ceil(1_000_000 / 4096)
//! ```

use std::sync::atomic::{AtomicU8, Ordering};

use serde::{Deserialize, Serialize};
use tracing::debug;

/// OS page size used for heat map granularity.
const PAGE_SIZE: usize = 4096;

/// Maximum heat value per page (u8 max).
const MAX_HEAT: u8 = 255;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for index warm-up behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmUpConfig {
    /// Which prefaulting strategy to use.
    pub strategy: WarmUpStrategy,
    /// Maximum bytes to prefault across all indices (default: 256 MB).
    pub max_bytes: usize,
    /// Number of concurrent prefault threads (default: 2).
    pub parallel_readers: usize,
}

impl Default for WarmUpConfig {
    fn default() -> Self {
        Self {
            strategy: WarmUpStrategy::None,
            max_bytes: 256 * 1024 * 1024,
            parallel_readers: 2,
        }
    }
}

impl WarmUpConfig {
    /// Create a config with the adaptive strategy using default parameters.
    #[must_use]
    pub fn adaptive() -> Self {
        Self {
            strategy: WarmUpStrategy::Adaptive(AdaptiveConfig::default()),
            ..Self::default()
        }
    }

    /// Create a config that prefaults all pages.
    #[must_use]
    pub fn full() -> Self {
        Self {
            strategy: WarmUpStrategy::Full,
            ..Self::default()
        }
    }

    /// Create a config that prefaults only the header region.
    #[must_use]
    pub fn header_only() -> Self {
        Self {
            strategy: WarmUpStrategy::Header,
            ..Self::default()
        }
    }
}

/// Prefaulting strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmUpStrategy {
    /// No prefaulting (current behavior).
    None,
    /// Touch every page sequentially.
    Full,
    /// Prefault header + record table only (smallest footprint).
    Header,
    /// Heat-map based intelligent prefaulting.
    Adaptive(AdaptiveConfig),
}

/// Configuration for the adaptive prefaulting strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Exponential decay factor applied to heat after each search cycle.
    /// Must be in [0.0, 1.0]. Default: 0.95.
    pub heat_decay: f64,
    /// Minimum normalized heat (0.0-1.0) to prefault a page.
    /// Pages below this threshold are not prefaulted. Default: 0.1.
    pub min_heat: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            heat_decay: 0.95,
            min_heat: 0.1,
        }
    }
}

impl AdaptiveConfig {
    /// Clamp `heat_decay` to `[0.0, 1.0]`.
    ///
    /// Non-finite values (NaN/Inf from deserialization) fall back to 0.95.
    #[must_use]
    pub const fn clamped_heat_decay(&self) -> f64 {
        if self.heat_decay.is_finite() {
            self.heat_decay.clamp(0.0, 1.0)
        } else {
            0.95
        }
    }

    /// Clamp `min_heat` to `[0.0, 1.0]`.
    ///
    /// Non-finite values (NaN/Inf from deserialization) fall back to 0.1.
    #[must_use]
    pub const fn clamped_min_heat(&self) -> f64 {
        if self.min_heat.is_finite() {
            self.min_heat.clamp(0.0, 1.0)
        } else {
            0.1
        }
    }
}

// ─── Heat Map ───────────────────────────────────────────────────────────────

/// Per-page heat tracker for adaptive prefaulting.
///
/// Each 4 KB page gets an `AtomicU8` heat counter (0-255). Heat is incremented
/// on every page access detected via [`record_access`](Self::record_access),
/// and decayed exponentially per search cycle via [`decay`](Self::decay).
///
/// Memory overhead: ~1 byte per 4 KB page = ~256 KB for a 1 GB index.
pub struct HeatMap {
    /// Per-page heat values. Index = page number.
    pages: Vec<AtomicU8>,
    /// Total data size this heat map covers.
    total_bytes: usize,
}

impl HeatMap {
    /// Create a new heat map covering `total_bytes` of data.
    ///
    /// The heat map allocates one `AtomicU8` per 4 KB page.
    #[must_use]
    pub fn new(total_bytes: usize) -> Self {
        let page_count = pages_for_bytes(total_bytes);
        let pages = (0..page_count).map(|_| AtomicU8::new(0)).collect();
        Self { pages, total_bytes }
    }

    /// Number of tracked pages.
    #[must_use]
    pub const fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Total bytes this heat map covers.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Record an access to the byte range `[byte_offset, byte_offset + len)`.
    ///
    /// Increments heat for all pages overlapping the accessed range.
    /// Heat saturates at 255 (no overflow).
    pub fn record_access(&self, byte_offset: usize, len: usize) {
        if len == 0 || self.pages.is_empty() {
            return;
        }
        let end = byte_offset.saturating_add(len).min(self.total_bytes);
        let start_page = byte_offset / PAGE_SIZE;
        let end_page = end.saturating_sub(1) / PAGE_SIZE;

        for page in start_page..=end_page.min(self.pages.len() - 1) {
            // Saturating increment: load, add, store. Race conditions are
            // acceptable (heat is approximate).
            let current = self.pages[page].load(Ordering::Relaxed);
            if current < MAX_HEAT {
                self.pages[page].store(current.saturating_add(1), Ordering::Relaxed);
            }
        }
    }

    /// Apply exponential decay to all page heats.
    ///
    /// Called once per search cycle: `heat = (heat * decay_factor) as u8`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn decay(&self, decay_factor: f64) {
        let factor = if decay_factor.is_finite() {
            decay_factor.clamp(0.0, 1.0)
        } else {
            0.0 // Non-finite decay → zero out all heat (safe reset)
        };

        for page in &self.pages {
            let current = page.load(Ordering::Relaxed);
            if current > 0 {
                let decayed = (f64::from(current) * factor) as u8;
                page.store(decayed, Ordering::Relaxed);
            }
        }
    }

    /// Get the heat value for a specific page.
    ///
    /// Returns 0 if the page index is out of bounds.
    #[must_use]
    pub fn heat_at(&self, page_index: usize) -> u8 {
        self.pages
            .get(page_index)
            .map_or(0, |p| p.load(Ordering::Relaxed))
    }

    /// Get normalized heat (0.0-1.0) for a specific page.
    #[must_use]
    pub fn normalized_heat_at(&self, page_index: usize) -> f64 {
        f64::from(self.heat_at(page_index)) / f64::from(MAX_HEAT)
    }

    /// Return page indices with heat above `min_heat` (normalized 0.0-1.0),
    /// sorted by heat descending (hottest first), capped at `max_bytes` budget.
    #[must_use]
    pub fn hot_pages(&self, min_heat: f64, max_bytes: usize) -> Vec<usize> {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let min_raw = (min_heat * f64::from(MAX_HEAT)) as u8;
        let max_pages = max_bytes / PAGE_SIZE;

        let mut hot: Vec<(usize, u8)> = self
            .pages
            .iter()
            .enumerate()
            .filter_map(|(idx, page)| {
                let heat = page.load(Ordering::Relaxed);
                if heat >= min_raw {
                    Some((idx, heat))
                } else {
                    None
                }
            })
            .collect();

        // Sort by heat descending (hottest first).
        hot.sort_unstable_by_key(|&(_, heat)| std::cmp::Reverse(heat));

        // Cap at budget.
        hot.truncate(max_pages);

        hot.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Reset all heat values to zero.
    pub fn reset(&self) {
        for page in &self.pages {
            page.store(0, Ordering::Relaxed);
        }
    }

    /// Count of pages with non-zero heat.
    #[must_use]
    pub fn warm_page_count(&self) -> usize {
        self.pages
            .iter()
            .filter(|p| p.load(Ordering::Relaxed) > 0)
            .count()
    }
}

impl std::fmt::Debug for HeatMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeatMap")
            .field("pages", &format_args!("[AtomicU8; {}]", self.pages.len()))
            .field("total_bytes", &self.total_bytes)
            .field("warm_pages", &self.warm_page_count())
            .finish()
    }
}

// ─── Warm-Up Execution ─────────────────────────────────────────────────────

/// Result of a warm-up operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmUpResult {
    /// Number of pages touched.
    pub pages_touched: usize,
    /// Total bytes prefaulted.
    pub bytes_touched: usize,
    /// Strategy that was applied.
    pub strategy_name: String,
    /// Whether the budget was fully consumed.
    pub budget_exhausted: bool,
}

/// Warm up a byte slice by reading through targeted pages.
///
/// For heap-allocated data (e.g., `Vec<u8>` from `fs::read`), this ensures
/// pages are resident in the process's address space. For mmap'd data, this
/// triggers page faults that load pages from disk into the OS page cache.
///
/// The function reads one byte per targeted page, which is sufficient to
/// trigger a page fault and make the page resident.
#[must_use]
pub fn warm_up_bytes(
    data: &[u8],
    header_end: usize,
    config: &WarmUpConfig,
    heat_map: Option<&HeatMap>,
) -> WarmUpResult {
    if data.is_empty() {
        return empty_result(&config.strategy);
    }

    match &config.strategy {
        WarmUpStrategy::None => empty_result(&WarmUpStrategy::None),
        WarmUpStrategy::Full => warm_up_bytes_full(data, config),
        WarmUpStrategy::Header => warm_up_bytes_header(data, header_end, config),
        WarmUpStrategy::Adaptive(adaptive_config) => {
            warm_up_bytes_adaptive(data, header_end, config, adaptive_config, heat_map)
        }
    }
}

fn warm_up_bytes_full(data: &[u8], config: &WarmUpConfig) -> WarmUpResult {
    let max_pages = config.max_bytes / PAGE_SIZE;
    let total_pages = pages_for_bytes(data.len());
    let pages_to_touch = total_pages.min(max_pages);
    let touched = touch_pages(data, 0..pages_to_touch);
    let budget_exhausted = total_pages > max_pages;

    debug!(
        target: "frankensearch.warmup",
        pages_touched = touched,
        total_pages,
        budget_exhausted,
        "full warm-up complete"
    );

    WarmUpResult {
        pages_touched: touched,
        bytes_touched: touched * PAGE_SIZE,
        strategy_name: "full".into(),
        budget_exhausted,
    }
}

fn warm_up_bytes_header(data: &[u8], header_end: usize, config: &WarmUpConfig) -> WarmUpResult {
    let header_bytes = header_end.min(data.len());
    let header_pages = pages_for_bytes(header_bytes);
    let max_pages = config.max_bytes / PAGE_SIZE;
    let pages_to_touch = header_pages.min(max_pages);
    let touched = touch_pages(data, 0..pages_to_touch);

    debug!(
        target: "frankensearch.warmup",
        pages_touched = touched,
        header_bytes,
        "header warm-up complete"
    );

    WarmUpResult {
        pages_touched: touched,
        bytes_touched: touched * PAGE_SIZE,
        strategy_name: "header".into(),
        budget_exhausted: header_pages > max_pages,
    }
}

fn warm_up_bytes_adaptive(
    data: &[u8],
    header_end: usize,
    config: &WarmUpConfig,
    adaptive_config: &AdaptiveConfig,
    heat_map: Option<&HeatMap>,
) -> WarmUpResult {
    let header_fallback = || {
        warm_up_bytes(
            data,
            header_end,
            &WarmUpConfig {
                strategy: WarmUpStrategy::Header,
                ..*config
            },
            None,
        )
    };

    let Some(heat_map) = heat_map else {
        debug!(target: "frankensearch.warmup", "adaptive: no heat map, falling back to header");
        return header_fallback();
    };

    let min_heat = adaptive_config.clamped_min_heat();
    let hot_pages = heat_map.hot_pages(min_heat, config.max_bytes);

    if hot_pages.is_empty() {
        debug!(target: "frankensearch.warmup", "adaptive: no hot pages, falling back to header");
        return header_fallback();
    }

    let mut touched = 0;
    for &page in &hot_pages {
        let offset = page * PAGE_SIZE;
        if offset < data.len() {
            std::hint::black_box(data[offset]);
            touched += 1;
        }
    }

    let budget_pages = config.max_bytes / PAGE_SIZE;
    let budget_exhausted = hot_pages.len() >= budget_pages;

    debug!(
        target: "frankensearch.warmup",
        pages_touched = touched,
        hot_page_count = hot_pages.len(),
        min_heat,
        budget_exhausted,
        "adaptive warm-up complete"
    );

    WarmUpResult {
        pages_touched: touched,
        bytes_touched: touched * PAGE_SIZE,
        strategy_name: "adaptive".into(),
        budget_exhausted,
    }
}

/// Touch one byte per page in the given range to trigger page faults.
fn touch_pages(data: &[u8], page_range: std::ops::Range<usize>) -> usize {
    let mut touched = 0;
    for page in page_range {
        let offset = page * PAGE_SIZE;
        if offset < data.len() {
            std::hint::black_box(data[offset]);
            touched += 1;
        }
    }
    touched
}

/// Warm up a memory-mapped file using `madvise(MADV_WILLNEED)`.
///
/// This requests the OS kernel to asynchronously read targeted pages into the
/// page cache, avoiding page faults on subsequent access.
///
/// # Errors
///
/// Returns `Err` if the madvise call fails (unlikely in practice).
pub fn warm_up_mmap(
    mmap: &memmap2::Mmap,
    header_end: usize,
    config: &WarmUpConfig,
    heat_map: Option<&HeatMap>,
) -> Result<WarmUpResult, std::io::Error> {
    if mmap.is_empty() {
        return Ok(empty_result(&config.strategy));
    }

    match &config.strategy {
        WarmUpStrategy::None => Ok(empty_result(&WarmUpStrategy::None)),
        WarmUpStrategy::Full => mmap_warm_up_full(mmap, config),
        WarmUpStrategy::Header => mmap_warm_up_header(mmap, header_end, config),
        WarmUpStrategy::Adaptive(ac) => {
            mmap_warm_up_adaptive(mmap, header_end, config, ac, heat_map)
        }
    }
}

fn mmap_warm_up_full(
    mmap: &memmap2::Mmap,
    config: &WarmUpConfig,
) -> Result<WarmUpResult, std::io::Error> {
    let total_pages = pages_for_bytes(mmap.len());
    let max_pages = config.max_bytes / PAGE_SIZE;
    let budget_exhausted = total_pages > max_pages;
    let actual_bytes = (total_pages.min(max_pages) * PAGE_SIZE).min(mmap.len());

    if actual_bytes > 0 {
        mmap.advise_range(memmap2::Advice::WillNeed, 0, actual_bytes)?;
    }

    let pages_touched = pages_for_bytes(actual_bytes);
    debug!(target: "frankensearch.warmup", pages_touched, total_pages, budget_exhausted, "mmap full warm-up");

    Ok(WarmUpResult {
        pages_touched,
        bytes_touched: actual_bytes,
        strategy_name: "full".into(),
        budget_exhausted,
    })
}

fn mmap_warm_up_header(
    mmap: &memmap2::Mmap,
    header_end: usize,
    config: &WarmUpConfig,
) -> Result<WarmUpResult, std::io::Error> {
    let header_bytes = header_end.min(mmap.len());
    let max_bytes = config.max_bytes.min(mmap.len());
    let actual = header_bytes.min(max_bytes);

    if actual > 0 {
        mmap.advise_range(memmap2::Advice::WillNeed, 0, actual)?;
    }

    let pages_touched = pages_for_bytes(actual);
    debug!(target: "frankensearch.warmup", pages_touched, header_bytes, "mmap header warm-up");

    Ok(WarmUpResult {
        pages_touched,
        bytes_touched: actual,
        strategy_name: "header".into(),
        budget_exhausted: header_bytes > max_bytes,
    })
}

fn mmap_warm_up_adaptive(
    mmap: &memmap2::Mmap,
    header_end: usize,
    config: &WarmUpConfig,
    adaptive_config: &AdaptiveConfig,
    heat_map: Option<&HeatMap>,
) -> Result<WarmUpResult, std::io::Error> {
    let header_fallback = || {
        warm_up_mmap(
            mmap,
            header_end,
            &WarmUpConfig {
                strategy: WarmUpStrategy::Header,
                ..*config
            },
            None,
        )
    };

    let Some(heat_map) = heat_map else {
        return header_fallback();
    };

    let min_heat = adaptive_config.clamped_min_heat();
    let hot_pages = heat_map.hot_pages(min_heat, config.max_bytes);

    if hot_pages.is_empty() {
        return header_fallback();
    }

    let mut touched = 0;
    for &page in &hot_pages {
        let offset = page * PAGE_SIZE;
        let len = PAGE_SIZE.min(mmap.len().saturating_sub(offset));
        if len > 0 {
            mmap.advise_range(memmap2::Advice::WillNeed, offset, len)?;
            touched += 1;
        }
    }

    let budget_pages = config.max_bytes / PAGE_SIZE;
    let budget_exhausted = hot_pages.len() >= budget_pages;

    debug!(
        target: "frankensearch.warmup",
        pages_touched = touched,
        hot_page_count = hot_pages.len(),
        budget_exhausted,
        "mmap adaptive warm-up"
    );

    Ok(WarmUpResult {
        pages_touched: touched,
        bytes_touched: touched * PAGE_SIZE,
        strategy_name: "adaptive".into(),
        budget_exhausted,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Number of pages needed to cover `bytes`.
#[must_use]
const fn pages_for_bytes(bytes: usize) -> usize {
    bytes.div_ceil(PAGE_SIZE)
}

/// Create a zero-work result for empty data or no-op strategies.
fn empty_result(strategy: &WarmUpStrategy) -> WarmUpResult {
    WarmUpResult {
        pages_touched: 0,
        bytes_touched: 0,
        strategy_name: strategy_name(strategy),
        budget_exhausted: false,
    }
}

/// Human-readable strategy name.
fn strategy_name(strategy: &WarmUpStrategy) -> String {
    match strategy {
        WarmUpStrategy::None => "none".into(),
        WarmUpStrategy::Full => "full".into(),
        WarmUpStrategy::Header => "header".into(),
        WarmUpStrategy::Adaptive(_) => "adaptive".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── HeatMap basics ────────────────────────────────────────────────

    #[test]
    fn heat_map_page_count() {
        assert_eq!(HeatMap::new(0).page_count(), 0);
        assert_eq!(HeatMap::new(1).page_count(), 1);
        assert_eq!(HeatMap::new(4096).page_count(), 1);
        assert_eq!(HeatMap::new(4097).page_count(), 2);
        assert_eq!(HeatMap::new(1_000_000).page_count(), 245);
    }

    #[test]
    fn heat_map_record_and_read() {
        let hm = HeatMap::new(100_000);
        assert_eq!(hm.heat_at(0), 0);

        // Access first page.
        hm.record_access(0, 100);
        assert_eq!(hm.heat_at(0), 1);

        // Access again — heat increments.
        hm.record_access(500, 200);
        assert_eq!(hm.heat_at(0), 2);

        // Access second page.
        hm.record_access(4096, 10);
        assert_eq!(hm.heat_at(1), 1);

        // First page unchanged.
        assert_eq!(hm.heat_at(0), 2);
    }

    #[test]
    fn heat_map_spanning_access() {
        let hm = HeatMap::new(20_000);
        // Access spanning pages 0, 1, and 2.
        hm.record_access(3000, 6000); // 3000..9000 → pages 0, 1, 2
        assert_eq!(hm.heat_at(0), 1);
        assert_eq!(hm.heat_at(1), 1);
        assert_eq!(hm.heat_at(2), 1);
        assert_eq!(hm.heat_at(3), 0);
    }

    #[test]
    fn heat_map_saturates_at_max() {
        let hm = HeatMap::new(4096);
        for _ in 0..300 {
            hm.record_access(0, 1);
        }
        assert_eq!(hm.heat_at(0), MAX_HEAT);
    }

    #[test]
    fn heat_map_decay() {
        let hm = HeatMap::new(4096);
        for _ in 0..100 {
            hm.record_access(0, 1);
        }
        assert_eq!(hm.heat_at(0), 100);

        hm.decay(0.5);
        assert_eq!(hm.heat_at(0), 50);

        hm.decay(0.5);
        assert_eq!(hm.heat_at(0), 25);

        // Decay to zero.
        for _ in 0..20 {
            hm.decay(0.5);
        }
        assert_eq!(hm.heat_at(0), 0);
    }

    #[test]
    fn heat_map_decay_zero_factor() {
        let hm = HeatMap::new(4096);
        hm.record_access(0, 1);
        assert_eq!(hm.heat_at(0), 1);

        hm.decay(0.0);
        assert_eq!(hm.heat_at(0), 0);
    }

    #[test]
    fn heat_map_decay_one_factor() {
        let hm = HeatMap::new(4096);
        for _ in 0..50 {
            hm.record_access(0, 1);
        }
        assert_eq!(hm.heat_at(0), 50);

        hm.decay(1.0);
        assert_eq!(hm.heat_at(0), 50);
    }

    #[test]
    fn heat_map_hot_pages_sorted_by_heat() {
        let hm = HeatMap::new(20_000); // 5 pages

        // Page 0: 10 accesses.
        for _ in 0..10 {
            hm.record_access(0, 1);
        }
        // Page 2: 50 accesses.
        for _ in 0..50 {
            hm.record_access(8192, 1);
        }
        // Page 4: 30 accesses.
        for _ in 0..30 {
            hm.record_access(16384, 1);
        }

        let hot = hm.hot_pages(0.01, usize::MAX);
        assert_eq!(hot.len(), 3);
        // Sorted by heat descending: page 2 (50), page 4 (30), page 0 (10).
        assert_eq!(hot[0], 2);
        assert_eq!(hot[1], 4);
        assert_eq!(hot[2], 0);
    }

    #[test]
    fn heat_map_hot_pages_respects_min_heat() {
        let hm = HeatMap::new(8192); // 2 pages
        for _ in 0..100 {
            hm.record_access(0, 1);
        }
        hm.record_access(4096, 1);

        // min_heat = 0.2 → min_raw = 51. Page 0 (100) passes, page 1 (1) does not.
        let hot = hm.hot_pages(0.2, usize::MAX);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0], 0);
    }

    #[test]
    fn heat_map_hot_pages_respects_budget() {
        let hm = HeatMap::new(20_000);
        for page in 0..5 {
            for _ in 0..10 {
                hm.record_access(page * PAGE_SIZE, 1);
            }
        }

        // Budget for 2 pages only.
        let hot = hm.hot_pages(0.01, 2 * PAGE_SIZE);
        assert_eq!(hot.len(), 2);
    }

    #[test]
    fn heat_map_reset() {
        let hm = HeatMap::new(20_000);
        for page in 0..5 {
            hm.record_access(page * PAGE_SIZE, 1);
        }
        assert_eq!(hm.warm_page_count(), 5);

        hm.reset();
        assert_eq!(hm.warm_page_count(), 0);
    }

    #[test]
    fn heat_map_normalized_heat() {
        let hm = HeatMap::new(4096);
        assert!((hm.normalized_heat_at(0) - 0.0).abs() < f64::EPSILON);

        for _ in 0..255 {
            hm.record_access(0, 1);
        }
        assert!((hm.normalized_heat_at(0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn heat_map_empty_index() {
        let hm = HeatMap::new(0);
        assert_eq!(hm.page_count(), 0);
        assert_eq!(hm.warm_page_count(), 0);
        hm.record_access(0, 100); // Should not panic.
        hm.decay(0.5); // Should not panic.
        assert!(hm.hot_pages(0.0, usize::MAX).is_empty());
    }

    #[test]
    fn heat_map_access_beyond_bounds() {
        let hm = HeatMap::new(4096);
        // Access beyond the tracked region — should be clamped, not panic.
        hm.record_access(10_000, 100);
        assert_eq!(hm.heat_at(0), 0);
    }

    #[test]
    fn heat_map_zero_length_access() {
        let hm = HeatMap::new(4096);
        hm.record_access(0, 0);
        assert_eq!(hm.heat_at(0), 0);
    }

    #[test]
    fn heat_map_index_smaller_than_page() {
        // Index smaller than one page (< 4 KB).
        let hm = HeatMap::new(100);
        assert_eq!(hm.page_count(), 1);
        hm.record_access(0, 50);
        assert_eq!(hm.heat_at(0), 1);
    }

    // ─── Warm-up execution ─────────────────────────────────────────────

    #[test]
    fn warm_up_none_is_noop() {
        let data = vec![0u8; 10_000];
        let result = warm_up_bytes(&data, 100, &WarmUpConfig::default(), None);
        assert_eq!(result.pages_touched, 0);
        assert_eq!(result.bytes_touched, 0);
        assert_eq!(result.strategy_name, "none");
        assert!(!result.budget_exhausted);
    }

    #[test]
    fn warm_up_full_touches_all_pages() {
        let data = vec![42u8; 20_000]; // 5 pages
        let config = WarmUpConfig::full();
        let result = warm_up_bytes(&data, 100, &config, None);
        assert_eq!(result.pages_touched, 5);
        assert_eq!(result.strategy_name, "full");
        assert!(!result.budget_exhausted);
    }

    #[test]
    fn warm_up_full_respects_budget() {
        let data = vec![42u8; 20_000]; // 5 pages
        let config = WarmUpConfig {
            strategy: WarmUpStrategy::Full,
            max_bytes: 2 * PAGE_SIZE, // Budget for 2 pages.
            parallel_readers: 1,
        };
        let result = warm_up_bytes(&data, 100, &config, None);
        assert_eq!(result.pages_touched, 2);
        assert!(result.budget_exhausted);
    }

    #[test]
    fn warm_up_header_only_touches_header() {
        let data = vec![42u8; 100_000]; // Many pages
        let header_end = 5000; // Header spans ~2 pages.
        let config = WarmUpConfig::header_only();
        let result = warm_up_bytes(&data, header_end, &config, None);
        assert_eq!(result.pages_touched, 2); // ceil(5000 / 4096) = 2
        assert_eq!(result.strategy_name, "header");
    }

    #[test]
    fn warm_up_adaptive_uses_heat_map() {
        let data = vec![42u8; 100_000]; // ~25 pages
        let hm = HeatMap::new(data.len());

        // Heat up pages 3 and 7.
        for _ in 0..50 {
            hm.record_access(3 * PAGE_SIZE, 100);
            hm.record_access(7 * PAGE_SIZE, 100);
        }

        let config = WarmUpConfig::adaptive();
        let result = warm_up_bytes(&data, 100, &config, Some(&hm));
        assert_eq!(result.pages_touched, 2);
        assert_eq!(result.strategy_name, "adaptive");
    }

    #[test]
    fn warm_up_adaptive_falls_back_without_heat_map() {
        let data = vec![42u8; 20_000];
        let config = WarmUpConfig::adaptive();
        let result = warm_up_bytes(&data, 5000, &config, None);
        // Falls back to header strategy.
        assert_eq!(result.strategy_name, "header");
        assert_eq!(result.pages_touched, 2);
    }

    #[test]
    fn warm_up_adaptive_falls_back_with_empty_heat_map() {
        let data = vec![42u8; 20_000];
        let hm = HeatMap::new(data.len());
        // No accesses recorded → empty heat map → falls back to header.
        let config = WarmUpConfig::adaptive();
        let result = warm_up_bytes(&data, 5000, &config, Some(&hm));
        assert_eq!(result.strategy_name, "header");
    }

    #[test]
    fn warm_up_empty_data() {
        let data: Vec<u8> = vec![];
        let result = warm_up_bytes(&data, 0, &WarmUpConfig::full(), None);
        assert_eq!(result.pages_touched, 0);
    }

    // ─── Config ────────────────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let config = WarmUpConfig::default();
        assert!(matches!(config.strategy, WarmUpStrategy::None));
        assert_eq!(config.max_bytes, 256 * 1024 * 1024);
        assert_eq!(config.parallel_readers, 2);
    }

    #[test]
    fn adaptive_config_clamping() {
        let ac = AdaptiveConfig {
            heat_decay: -0.5,
            min_heat: 2.0,
        };
        assert!((ac.clamped_heat_decay() - 0.0).abs() < f64::EPSILON);
        assert!((ac.clamped_min_heat() - 1.0).abs() < f64::EPSILON);

        let ac2 = AdaptiveConfig {
            heat_decay: 1.5,
            min_heat: -0.3,
        };
        assert!((ac2.clamped_heat_decay() - 1.0).abs() < f64::EPSILON);
        assert!((ac2.clamped_min_heat() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = WarmUpConfig::adaptive();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WarmUpConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized.strategy, WarmUpStrategy::Adaptive(_)));
        assert_eq!(deserialized.max_bytes, config.max_bytes);
    }

    // ─── HeatMap concurrent access ─────────────────────────────────────

    #[test]
    fn heat_map_concurrent_access_no_panic() {
        use std::sync::Arc;

        let hm = Arc::new(HeatMap::new(100_000));
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let hm = Arc::clone(&hm);
                std::thread::spawn(move || {
                    for i in 0..1000 {
                        hm.record_access((t * 10_000 + i * 10) % 100_000, 100);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Some pages should have heat.
        assert!(hm.warm_page_count() > 0);
    }

    // ─── Pages-for-bytes helper ────────────────────────────────────────

    #[test]
    fn pages_for_bytes_calculation() {
        assert_eq!(pages_for_bytes(0), 0);
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(4095), 1);
        assert_eq!(pages_for_bytes(4096), 1);
        assert_eq!(pages_for_bytes(4097), 2);
        assert_eq!(pages_for_bytes(8192), 2);
        assert_eq!(pages_for_bytes(1_073_741_824), 262_144); // 1 GB
    }

    // ─── HeatMap debug formatting ──────────────────────────────────────

    #[test]
    fn heat_map_debug() {
        let hm = HeatMap::new(20_000);
        hm.record_access(0, 100);
        let debug = format!("{hm:?}");
        assert!(debug.contains("HeatMap"));
        assert!(debug.contains("AtomicU8; 5"));
        assert!(debug.contains("warm_pages: 1"));
    }

    // ─── Warm-up result ────────────────────────────────────────────────

    #[test]
    fn warm_up_result_serde() {
        let result = WarmUpResult {
            pages_touched: 10,
            bytes_touched: 40960,
            strategy_name: "adaptive".into(),
            budget_exhausted: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: WarmUpResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.pages_touched, 10);
        assert_eq!(deserialized.strategy_name, "adaptive");
    }

    #[test]
    fn adaptive_config_nan_heat_decay_uses_default() {
        let config = AdaptiveConfig {
            heat_decay: f64::NAN,
            min_heat: 0.1,
        };
        let decay = config.clamped_heat_decay();
        assert!(decay.is_finite(), "NaN heat_decay must fallback to default");
        assert!((decay - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn adaptive_config_nan_min_heat_uses_default() {
        let config = AdaptiveConfig {
            heat_decay: 0.85,
            min_heat: f64::NAN,
        };
        let min_heat = config.clamped_min_heat();
        assert!(
            min_heat.is_finite(),
            "NaN min_heat must fallback to default"
        );
        assert!((min_heat - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn heat_map_decay_nan_factor_zeroes_heat() {
        let map = HeatMap::new(PAGE_SIZE * 4);
        map.record_access(0, PAGE_SIZE);
        assert!(map.heat_at(0) > 0, "heat should be recorded");

        map.decay(f64::NAN);
        // NaN decay factor → falls back to 0.0 → heat zeroed
        assert_eq!(map.heat_at(0), 0, "NaN decay factor should zero out heat");
    }
}
