//! Frame budget enforcement and jank detection.
//!
//! Provides a [`FrameBudget`] that tracks render timing and detects
//! when frames exceed their time budget (jank). Product crates can
//! register a [`JankCallback`] to be notified of slow frames for
//! diagnostics and telemetry.

use std::time::{Duration, Instant};
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
}
