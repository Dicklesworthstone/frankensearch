//! Frame budget enforcement and jank detection.
//!
//! Provides a [`FrameBudget`] that tracks render timing and detects
//! when frames exceed their time budget (jank). Product crates can
//! register a [`JankCallback`] to be notified of slow frames for
//! diagnostics and telemetry.

use std::time::{Duration, Instant};

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use tracing::warn;

// ─── Jank Callback ──────────────────────────────────────────────────────────

/// Callback invoked when a frame exceeds its time budget.
///
/// Receives the frame metrics for the slow frame.
pub type JankCallback = Box<dyn Fn(&FrameMetrics) + Send>;

// ─── Frame Metrics ──────────────────────────────────────────────────────────

/// Timing metrics for a single rendered frame.
#[derive(Debug, Clone)]
pub struct FrameMetrics {
    /// Sequence number (monotonically increasing).
    pub frame_number: u64,
    /// Time spent rendering this frame.
    pub render_duration: Duration,
    /// The target frame budget.
    pub budget: Duration,
    /// Whether this frame exceeded the budget (jank).
    pub is_jank: bool,
    /// When the frame started rendering.
    pub timestamp: Instant,
}

impl FrameMetrics {
    /// How much the frame exceeded the budget, if at all.
    #[must_use]
    pub const fn overshoot(&self) -> Duration {
        self.render_duration.saturating_sub(self.budget)
    }

    /// Ratio of render time to budget (1.0 = exactly on budget).
    #[must_use]
    pub fn budget_ratio(&self) -> f64 {
        if self.budget.is_zero() {
            return f64::INFINITY;
        }
        self.render_duration.as_secs_f64() / self.budget.as_secs_f64()
    }
}

// ─── Frame Budget ────────────────────────────────────────────────────────────

/// Tracks frame timing and detects jank.
///
/// # Usage
///
/// ```ignore
/// let mut budget = FrameBudget::new(Duration::from_millis(16));
/// budget.begin_frame();
/// // ... render ...
/// let metrics = budget.end_frame();
/// if metrics.is_jank {
///     tracing::warn!("Jank detected: {:?}", metrics.render_duration);
/// }
/// ```
pub struct FrameBudget {
    /// Target frame duration.
    target: Duration,
    /// Current frame number.
    frame_number: u64,
    /// When the current frame started.
    frame_start: Option<Instant>,
    /// Optional jank callback.
    jank_callback: Option<JankCallback>,
    /// Running count of janky frames.
    jank_count: u64,
    /// Total frames rendered.
    total_frames: u64,
}

impl FrameBudget {
    /// Create a new frame budget with the given target duration.
    #[must_use]
    pub fn new(target: Duration) -> Self {
        Self {
            target,
            frame_number: 0,
            frame_start: None,
            jank_callback: None,
            jank_count: 0,
            total_frames: 0,
        }
    }

    /// Create a frame budget targeting 60 FPS (~16.67ms per frame).
    #[must_use]
    pub fn at_60fps() -> Self {
        Self::new(Duration::from_micros(16_667))
    }

    /// Create a frame budget targeting 30 FPS (~33.33ms per frame).
    #[must_use]
    pub fn at_30fps() -> Self {
        Self::new(Duration::from_micros(33_333))
    }

    /// Set the jank callback.
    pub fn on_jank(&mut self, callback: JankCallback) {
        self.jank_callback = Some(callback);
    }

    /// Mark the start of a frame.
    pub fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
    }

    /// Mark the end of a frame. Returns metrics for the completed frame.
    pub fn end_frame(&mut self) -> FrameMetrics {
        let Some(start) = self.frame_start.take() else {
            warn!(
                target: "frankensearch.tui.frame",
                "end_frame called without begin_frame; returning empty metrics"
            );
            return FrameMetrics {
                frame_number: self.frame_number,
                render_duration: Duration::ZERO,
                budget: self.target,
                is_jank: false,
                timestamp: Instant::now(),
            };
        };
        let render_duration = start.elapsed();
        let is_jank = render_duration > self.target;

        self.frame_number += 1;
        self.total_frames += 1;
        if is_jank {
            self.jank_count += 1;
        }

        let metrics = FrameMetrics {
            frame_number: self.frame_number,
            render_duration,
            budget: self.target,
            is_jank,
            timestamp: start,
        };

        if is_jank {
            if let Some(cb) = &self.jank_callback {
                cb(&metrics);
            }
        }

        metrics
    }

    /// Get the target frame budget.
    #[must_use]
    pub const fn target(&self) -> Duration {
        self.target
    }

    /// Total frames rendered.
    #[must_use]
    pub const fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Number of janky frames.
    #[must_use]
    pub const fn jank_count(&self) -> u64 {
        self.jank_count
    }

    /// Jank rate (0.0 = no jank, 1.0 = all frames janky).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn jank_rate(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.jank_count as f64 / self.total_frames as f64
    }
}

// ─── Frame Pipeline Timer ──────────────────────────────────────────────────

/// Per-phase timing metrics for a single frame's pipeline.
///
/// Decomposes total frame time into three phases matching the TUI latency
/// boundary model:
/// 1. **`InputToIntent`** — event receipt to action resolution
/// 2. **`IntentToState`** — action dispatch to state mutation complete
/// 3. **`StateToFrame`** — state read to rendered frame
#[derive(Debug, Clone)]
pub struct FramePipelineMetrics {
    /// Time spent resolving input to an intent/action.
    pub input_to_intent: Duration,
    /// Time spent applying the intent to application state.
    pub intent_to_state: Duration,
    /// Time spent rendering state to the frame buffer.
    pub state_to_frame: Duration,
    /// Sum of all three phases.
    pub total: Duration,
    /// Per-phase budget conformance (true = within budget).
    pub input_to_intent_ok: bool,
    /// Per-phase budget conformance for intent-to-state.
    pub intent_to_state_ok: bool,
    /// Per-phase budget conformance for state-to-frame.
    pub state_to_frame_ok: bool,
}

impl FramePipelineMetrics {
    /// Whether all three phases met their budgets.
    #[must_use]
    pub const fn all_within_budget(&self) -> bool {
        self.input_to_intent_ok && self.intent_to_state_ok && self.state_to_frame_ok
    }
}

/// Three-phase pipeline timer for decomposing frame latency.
///
/// Call `begin_input()`, `end_input_begin_state()`, `end_state_begin_render()`,
/// and `end_render()` at the corresponding pipeline boundaries. The timer
/// collects per-phase durations and checks them against configurable budgets.
///
/// # Example
///
/// ```ignore
/// let mut timer = FramePipelineTimer::new(
///     Duration::from_millis(8),   // input budget
///     Duration::from_millis(12),  // state budget
///     Duration::from_millis(16),  // render budget
/// );
/// timer.begin_input();
/// // ... resolve key event ...
/// timer.end_input_begin_state();
/// // ... update app state ...
/// timer.end_state_begin_render();
/// // ... render frame ...
/// let metrics = timer.end_render();
/// ```
pub struct FramePipelineTimer {
    input_budget: Duration,
    state_budget: Duration,
    render_budget: Duration,
    input_start: Option<Instant>,
    state_start: Option<Instant>,
    render_start: Option<Instant>,
    input_duration: Duration,
    state_duration: Duration,
}

impl FramePipelineTimer {
    /// Create a pipeline timer with per-phase budgets.
    #[must_use]
    pub const fn new(
        input_budget: Duration,
        state_budget: Duration,
        render_budget: Duration,
    ) -> Self {
        Self {
            input_budget,
            state_budget,
            render_budget,
            input_start: None,
            state_start: None,
            render_start: None,
            input_duration: Duration::ZERO,
            state_duration: Duration::ZERO,
        }
    }

    /// Create a pipeline timer using the default fsfs budgets (8/12/16 ms).
    #[must_use]
    pub const fn with_default_budgets() -> Self {
        Self::new(
            Duration::from_millis(8),
            Duration::from_millis(12),
            Duration::from_millis(16),
        )
    }

    /// Mark the start of the input-to-intent phase.
    pub fn begin_input(&mut self) {
        self.input_start = Some(Instant::now());
    }

    /// End input phase, begin intent-to-state phase.
    pub fn end_input_begin_state(&mut self) {
        if let Some(start) = self.input_start.take() {
            self.input_duration = start.elapsed();
        }
        self.state_start = Some(Instant::now());
    }

    /// End state phase, begin state-to-frame (render) phase.
    pub fn end_state_begin_render(&mut self) {
        if let Some(start) = self.state_start.take() {
            self.state_duration = start.elapsed();
        }
        self.render_start = Some(Instant::now());
    }

    /// End render phase and produce pipeline metrics.
    pub fn end_render(&mut self) -> FramePipelineMetrics {
        let render_duration = self
            .render_start
            .take()
            .map_or(Duration::ZERO, |s| s.elapsed());

        let total = self.input_duration + self.state_duration + render_duration;

        FramePipelineMetrics {
            input_to_intent: self.input_duration,
            intent_to_state: self.state_duration,
            state_to_frame: render_duration,
            total,
            input_to_intent_ok: self.input_duration <= self.input_budget,
            intent_to_state_ok: self.state_duration <= self.state_budget,
            state_to_frame_ok: render_duration <= self.render_budget,
        }
    }

    /// Skip the input phase (e.g., for render-only frames triggered by
    /// timers or data updates, not user input).
    pub fn skip_input_begin_state(&mut self) {
        self.input_duration = Duration::ZERO;
        self.state_start = Some(Instant::now());
    }
}

// ─── Cached Layout ─────────────────────────────────────────────────────────

/// Cached layout split result to avoid recomputing on every frame.
///
/// The layout only needs recomputing when the terminal area changes or the
/// shell chrome configuration changes (breadcrumbs/status bar visibility,
/// screen count).
#[derive(Debug, Clone)]
pub struct CachedLayout {
    /// The area that was used to compute the cached layout.
    area: Rect,
    /// Whether breadcrumbs were shown.
    show_breadcrumbs: bool,
    /// Whether status bar was shown.
    show_status_bar: bool,
    /// Number of screens (breadcrumbs shown when > 1).
    screen_count: usize,
    /// The cached layout chunks.
    chunks: Vec<Rect>,
}

impl CachedLayout {
    /// Create a new cached layout (initially invalid — will be recomputed
    /// on first `get_or_compute` call).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            area: Rect::new(0, 0, 0, 0),
            show_breadcrumbs: false,
            show_status_bar: false,
            screen_count: 0,
            chunks: Vec::new(),
        }
    }

    /// Get the cached layout if valid, or recompute and cache.
    ///
    /// Returns the layout chunks and the starting chunk index for the
    /// breadcrumbs (0 if breadcrumbs are shown, otherwise the content
    /// starts at index 0).
    pub fn get_or_compute(
        &mut self,
        area: Rect,
        show_breadcrumbs: bool,
        show_status_bar: bool,
        screen_count: usize,
    ) -> &[Rect] {
        let breadcrumbs_visible = show_breadcrumbs && screen_count > 1;

        if self.area == area
            && self.show_breadcrumbs == breadcrumbs_visible
            && self.show_status_bar == show_status_bar
            && self.screen_count == screen_count
        {
            return &self.chunks;
        }

        // Recompute layout.
        let mut constraints = Vec::with_capacity(3);
        if breadcrumbs_visible {
            constraints.push(Constraint::Length(1));
        }
        constraints.push(Constraint::Min(1));
        if show_status_bar {
            constraints.push(Constraint::Length(1));
        }

        let layout_result = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);

        self.chunks = layout_result.to_vec();
        self.area = area;
        self.show_breadcrumbs = breadcrumbs_visible;
        self.show_status_bar = show_status_bar;
        self.screen_count = screen_count;

        &self.chunks
    }

    /// Force invalidation (e.g., after config change).
    pub const fn invalidate(&mut self) {
        self.area = Rect::new(0, 0, 0, 0);
    }
}

impl Default for CachedLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Cached Tab State ──────────────────────────────────────────────────────

/// Cached tab bar titles and selected index to avoid rebuilding every frame.
///
/// Invalidated when screens are registered/unregistered or the active screen
/// changes.
#[derive(Debug, Clone)]
pub struct CachedTabState {
    /// Cached tab titles in order.
    pub titles: Vec<String>,
    /// Index of the selected (active) tab.
    pub selected: usize,
    /// Screen IDs at the time of caching (for invalidation check).
    screen_ids_hash: u64,
    /// Screen IDs + titles signature to catch dynamic title updates.
    title_signature: u64,
    /// Active screen ID at the time of caching.
    active_screen: Option<String>,
}

impl CachedTabState {
    /// Create a new empty cached tab state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            titles: Vec::new(),
            selected: 0,
            screen_ids_hash: 0,
            title_signature: 0,
            active_screen: None,
        }
    }

    /// Check if the cache is valid for the current state.
    #[must_use]
    pub fn is_valid(
        &self,
        screen_ids_hash: u64,
        title_signature: u64,
        active_screen: Option<&str>,
    ) -> bool {
        self.screen_ids_hash == screen_ids_hash
            && self.title_signature == title_signature
            && self.active_screen.as_deref() == active_screen
    }

    /// Update the cache with new values.
    pub fn update(
        &mut self,
        titles: Vec<String>,
        selected: usize,
        screen_ids_hash: u64,
        title_signature: u64,
        active_screen: Option<&str>,
    ) {
        self.titles = titles;
        self.selected = selected;
        self.screen_ids_hash = screen_ids_hash;
        self.title_signature = title_signature;
        self.active_screen = active_screen.map(String::from);
    }

    /// Force invalidation.
    pub fn invalidate(&mut self) {
        self.screen_ids_hash = 0;
        self.title_signature = 0;
        self.active_screen = None;
    }
}

impl Default for CachedTabState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn frame_budget_creation() {
        let budget = FrameBudget::new(Duration::from_millis(16));
        assert_eq!(budget.target(), Duration::from_millis(16));
        assert_eq!(budget.total_frames(), 0);
        assert_eq!(budget.jank_count(), 0);
    }

    #[test]
    fn frame_budget_60fps() {
        let budget = FrameBudget::at_60fps();
        assert_eq!(budget.target(), Duration::from_micros(16_667));
    }

    #[test]
    fn frame_budget_30fps() {
        let budget = FrameBudget::at_30fps();
        assert_eq!(budget.target(), Duration::from_micros(33_333));
    }

    #[test]
    fn frame_budget_begin_end() {
        let mut budget = FrameBudget::new(Duration::from_secs(10)); // Very generous.
        budget.begin_frame();
        let metrics = budget.end_frame();

        assert_eq!(metrics.frame_number, 1);
        assert!(!metrics.is_jank);
        assert_eq!(budget.total_frames(), 1);
        assert_eq!(budget.jank_count(), 0);
    }

    #[test]
    fn end_frame_without_begin_is_safe_noop() {
        let mut budget = FrameBudget::new(Duration::from_millis(16));
        let metrics = budget.end_frame();

        assert_eq!(metrics.render_duration, Duration::ZERO);
        assert!(!metrics.is_jank);
        assert_eq!(metrics.frame_number, 0);
        assert_eq!(budget.total_frames(), 0);
        assert_eq!(budget.jank_count(), 0);
    }

    #[test]
    fn frame_metrics_overshoot() {
        let metrics = FrameMetrics {
            frame_number: 1,
            render_duration: Duration::from_millis(20),
            budget: Duration::from_millis(16),
            is_jank: true,
            timestamp: Instant::now(),
        };
        assert_eq!(metrics.overshoot(), Duration::from_millis(4));
    }

    #[test]
    fn frame_metrics_budget_ratio() {
        let metrics = FrameMetrics {
            frame_number: 1,
            render_duration: Duration::from_millis(32),
            budget: Duration::from_millis(16),
            is_jank: true,
            timestamp: Instant::now(),
        };
        let ratio = metrics.budget_ratio();
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn frame_metrics_no_overshoot() {
        let metrics = FrameMetrics {
            frame_number: 1,
            render_duration: Duration::from_millis(10),
            budget: Duration::from_millis(16),
            is_jank: false,
            timestamp: Instant::now(),
        };
        assert_eq!(metrics.overshoot(), Duration::ZERO);
    }

    #[test]
    fn jank_rate_zero_frames() {
        let budget = FrameBudget::new(Duration::from_millis(16));
        assert!(budget.jank_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn jank_callback_fires() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);

        let mut budget = FrameBudget::new(Duration::ZERO); // Everything is jank.
        budget.on_jank(Box::new(move |_metrics| {
            fired_clone.store(true, Ordering::Relaxed);
        }));

        budget.begin_frame();
        std::thread::sleep(Duration::from_millis(1));
        let _metrics = budget.end_frame();

        assert!(fired.load(Ordering::Relaxed));
    }

    // ─── FramePipelineTimer Tests ──────────────────────────────────────

    #[test]
    fn pipeline_timer_default_budgets() {
        let timer = FramePipelineTimer::with_default_budgets();
        assert_eq!(timer.input_budget, Duration::from_millis(8));
        assert_eq!(timer.state_budget, Duration::from_millis(12));
        assert_eq!(timer.render_budget, Duration::from_millis(16));
    }

    #[test]
    fn pipeline_timer_full_cycle() {
        let mut timer = FramePipelineTimer::new(
            Duration::from_secs(10),
            Duration::from_secs(10),
            Duration::from_secs(10),
        );
        timer.begin_input();
        timer.end_input_begin_state();
        timer.end_state_begin_render();
        let metrics = timer.end_render();

        assert!(metrics.all_within_budget());
        assert!(metrics.input_to_intent < Duration::from_millis(100));
        assert!(metrics.intent_to_state < Duration::from_millis(100));
        assert!(metrics.state_to_frame < Duration::from_millis(100));
    }

    #[test]
    fn pipeline_timer_skip_input() {
        let mut timer = FramePipelineTimer::with_default_budgets();
        timer.skip_input_begin_state();
        timer.end_state_begin_render();
        let metrics = timer.end_render();

        assert_eq!(metrics.input_to_intent, Duration::ZERO);
        assert!(metrics.input_to_intent_ok);
    }

    #[test]
    fn pipeline_metrics_budget_violation() {
        let metrics = FramePipelineMetrics {
            input_to_intent: Duration::from_millis(10),
            intent_to_state: Duration::from_millis(5),
            state_to_frame: Duration::from_millis(5),
            total: Duration::from_millis(20),
            input_to_intent_ok: false,
            intent_to_state_ok: true,
            state_to_frame_ok: true,
        };
        assert!(!metrics.all_within_budget());
    }

    // ─── CachedLayout Tests ───────────────────────────────────────────

    #[test]
    fn cached_layout_computes_on_first_call() {
        let mut cache = CachedLayout::new();
        let area = Rect::new(0, 0, 100, 40);
        let chunks = cache.get_or_compute(area, true, true, 3);
        // 3 chunks: breadcrumbs + content + status bar
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn cached_layout_returns_cached_on_same_input() {
        let mut cache = CachedLayout::new();
        let area = Rect::new(0, 0, 100, 40);

        let first = cache.get_or_compute(area, true, true, 3).to_vec();
        let second = cache.get_or_compute(area, true, true, 3).to_vec();

        assert_eq!(first, second);
    }

    #[test]
    fn cached_layout_recomputes_on_area_change() {
        let mut cache = CachedLayout::new();

        let chunks1 = cache
            .get_or_compute(Rect::new(0, 0, 100, 40), false, true, 1)
            .to_vec();
        let chunks2 = cache
            .get_or_compute(Rect::new(0, 0, 120, 50), false, true, 1)
            .to_vec();

        // Content area should differ in height.
        assert_ne!(chunks1[0].height, chunks2[0].height);
    }

    #[test]
    fn cached_layout_no_breadcrumbs_single_screen() {
        let mut cache = CachedLayout::new();
        let chunks = cache.get_or_compute(Rect::new(0, 0, 80, 24), true, true, 1);
        // breadcrumbs hidden when screen_count <= 1
        assert_eq!(chunks.len(), 2); // content + status bar only
    }

    #[test]
    fn cached_layout_invalidate() {
        let mut cache = CachedLayout::new();
        let area = Rect::new(0, 0, 100, 40);
        let _ = cache.get_or_compute(area, false, true, 1);

        cache.invalidate();

        // After invalidation, area won't match so it recomputes.
        assert_eq!(cache.area, Rect::new(0, 0, 0, 0));
    }

    // ─── CachedTabState Tests ─────────────────────────────────────────

    #[test]
    fn cached_tab_state_initially_invalid() {
        let cache = CachedTabState::new();
        assert!(!cache.is_valid(12345, 777, Some("test")));
    }

    #[test]
    fn cached_tab_state_valid_after_update() {
        let mut cache = CachedTabState::new();
        cache.update(vec!["A".into(), "B".into()], 1, 999, 4242, Some("B"));
        assert!(cache.is_valid(999, 4242, Some("B")));
        assert_eq!(cache.selected, 1);
        assert_eq!(cache.titles, vec!["A", "B"]);
    }

    #[test]
    fn cached_tab_state_invalidate() {
        let mut cache = CachedTabState::new();
        cache.update(vec!["X".into()], 0, 42, 17, Some("X"));
        assert!(cache.is_valid(42, 17, Some("X")));

        cache.invalidate();
        assert!(!cache.is_valid(42, 17, Some("X")));
    }

    #[test]
    fn cached_tab_state_detects_screen_change() {
        let mut cache = CachedTabState::new();
        cache.update(vec!["A".into()], 0, 100, 200, Some("A"));

        // Different hash = different screen set.
        assert!(!cache.is_valid(101, 200, Some("A")));
        // Different title signature.
        assert!(!cache.is_valid(100, 201, Some("A")));
        // Different active screen.
        assert!(!cache.is_valid(100, 200, Some("B")));
    }
}
