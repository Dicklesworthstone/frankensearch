//! Accessibility constraints, keyboard parity audit, and frame-time quality budgets.
//!
//! This module defines the formal quality constraints that all TUI screens must
//! satisfy. It covers three domains:
//!
//! - **Keyboard-only parity**: every action reachable via mouse must also be
//!   reachable via keyboard alone
//! - **Frame-time quality**: render budgets and violation detection
//! - **Reduced motion**: animation timing constants gated on `MotionPreference`

use serde::{Deserialize, Serialize};

use crate::preferences::{DisplayPreferences, MotionPreference};

// ─── Schema Version ─────────────────────────────────────────────────────────

/// Accessibility constraint schema version.
pub const ACCESSIBILITY_SCHEMA_VERSION: u32 = 1;

// ─── Frame-Time Quality Budget ──────────────────────────────────────────────

/// Target frame rate (60 fps = 16.67ms per frame).
pub const TARGET_FPS: u32 = 60;

/// Maximum frame render time in milliseconds (at 60 fps).
pub const FRAME_BUDGET_MS: u16 = 16;

/// Hard deadline: frames exceeding this are considered dropped.
pub const FRAME_DROP_THRESHOLD_MS: u16 = 33;

/// Maximum acceptable input-to-visual-feedback latency in milliseconds.
/// Exceeding this feels sluggish to the user (100ms perceptual threshold).
pub const INPUT_FEEDBACK_BUDGET_MS: u16 = 100;

/// Frame-time quality constraint for a specific render phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameTimeBudget {
    /// Phase name (e.g., `state_to_frame`, `input_to_intent`).
    pub phase: FramePhase,
    /// Budget in milliseconds.
    pub budget_ms: u16,
    /// Whether exceeding this budget triggers a tracing warning.
    pub warn_on_exceed: bool,
}

/// Named phases of the frame pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FramePhase {
    /// User input event → semantic action dispatch.
    InputToIntent,
    /// Action dispatch → application state mutation.
    IntentToState,
    /// State update → screen rendering complete.
    StateToFrame,
    /// Full pipeline: input → pixels.
    InputToPixels,
}

impl FramePhase {
    /// Human label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::InputToIntent => "Input → Intent",
            Self::IntentToState => "Intent → State",
            Self::StateToFrame => "State → Frame",
            Self::InputToPixels => "Input → Pixels",
        }
    }
}

/// Canonical frame-time budget set for the ops TUI.
pub const FRAME_BUDGETS: &[FrameTimeBudget] = &[
    FrameTimeBudget {
        phase: FramePhase::InputToIntent,
        budget_ms: 8,
        warn_on_exceed: true,
    },
    FrameTimeBudget {
        phase: FramePhase::IntentToState,
        budget_ms: 12,
        warn_on_exceed: true,
    },
    FrameTimeBudget {
        phase: FramePhase::StateToFrame,
        budget_ms: 16,
        warn_on_exceed: true,
    },
    FrameTimeBudget {
        phase: FramePhase::InputToPixels,
        budget_ms: 33,
        warn_on_exceed: true,
    },
];

/// Outcome of a single frame render against quality constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameQualityVerdict {
    /// Frame completed within budget.
    OnBudget,
    /// Frame exceeded soft budget but not drop threshold.
    OverBudget,
    /// Frame exceeded drop threshold — user-visible jank.
    Dropped,
}

impl FrameQualityVerdict {
    /// Classify a frame duration against the standard budgets.
    #[must_use]
    pub const fn classify(duration_ms: u16) -> Self {
        if duration_ms <= FRAME_BUDGET_MS {
            Self::OnBudget
        } else if duration_ms <= FRAME_DROP_THRESHOLD_MS {
            Self::OverBudget
        } else {
            Self::Dropped
        }
    }
}

/// Rolling frame quality tracker.
///
/// Tracks frame quality over a sliding window for diagnostic display
/// and SLO violation detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameQualityTracker {
    /// Window size (number of frames to track).
    window: usize,
    /// Ring buffer of frame durations in ms.
    samples: Vec<u16>,
    /// Write cursor into the ring buffer.
    cursor: usize,
    /// Total frames recorded.
    total_frames: u64,
    /// Count of on-budget frames.
    on_budget_count: u64,
    /// Count of dropped frames.
    dropped_count: u64,
}

impl FrameQualityTracker {
    /// Create a tracker with the given window size.
    #[must_use]
    pub fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            samples: vec![0; window],
            cursor: 0,
            total_frames: 0,
            on_budget_count: 0,
            dropped_count: 0,
        }
    }

    /// Record a frame duration.
    pub fn record(&mut self, duration_ms: u16) {
        self.samples[self.cursor] = duration_ms;
        self.cursor = (self.cursor + 1) % self.window;
        self.total_frames += 1;
        match FrameQualityVerdict::classify(duration_ms) {
            FrameQualityVerdict::OnBudget => self.on_budget_count += 1,
            FrameQualityVerdict::OverBudget => {}
            FrameQualityVerdict::Dropped => self.dropped_count += 1,
        }
    }

    /// Total frames recorded.
    #[must_use]
    pub const fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Fraction of frames that were on-budget (0.0..=1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn on_budget_ratio(&self) -> f64 {
        if self.total_frames == 0 {
            return 1.0;
        }
        self.on_budget_count as f64 / self.total_frames as f64
    }

    /// Number of dropped frames.
    #[must_use]
    pub const fn dropped_count(&self) -> u64 {
        self.dropped_count
    }

    /// Whether the frame quality SLO is met (>= 95% on-budget).
    #[must_use]
    pub fn slo_met(&self) -> bool {
        self.on_budget_ratio() >= 0.95
    }

    /// P95 frame time from the current window.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn p95_frame_time_ms(&self) -> u16 {
        let active = self.total_frames.min(self.window as u64) as usize;
        if active == 0 {
            return 0;
        }
        let mut sorted: Vec<u16> = self.samples[..active].to_vec();
        sorted.sort_unstable();
        let idx = ((active as f64) * 0.95).ceil() as usize;
        sorted[idx.min(active - 1)]
    }
}

// ─── Reduced Motion Timing ──────────────────────────────────────────────────

/// Animation timing constants.
///
/// When `MotionPreference::Reduced` is active, all animation durations
/// collapse to zero and transition timings are instant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationTiming {
    /// Duration of a cursor/highlight slide animation (ms).
    pub cursor_slide_ms: u16,
    /// Duration of a panel expand/collapse animation (ms).
    pub panel_transition_ms: u16,
    /// Duration of a sparkline/chart redraw animation (ms).
    pub chart_redraw_ms: u16,
    /// Duration of a status badge fade animation (ms).
    pub badge_fade_ms: u16,
    /// Spinner frame interval (ms). 0 = no spinner, show static indicator.
    pub spinner_interval_ms: u16,
}

impl AnimationTiming {
    /// Full-motion timing (default).
    pub const FULL: Self = Self {
        cursor_slide_ms: 80,
        panel_transition_ms: 150,
        chart_redraw_ms: 200,
        badge_fade_ms: 120,
        spinner_interval_ms: 100,
    };

    /// Reduced-motion timing (instant transitions, no spinners).
    pub const REDUCED: Self = Self {
        cursor_slide_ms: 0,
        panel_transition_ms: 0,
        chart_redraw_ms: 0,
        badge_fade_ms: 0,
        spinner_interval_ms: 0,
    };

    /// Resolve timing from motion preference.
    #[must_use]
    pub const fn from_preference(motion: MotionPreference) -> Self {
        match motion {
            MotionPreference::Full => Self::FULL,
            MotionPreference::Reduced => Self::REDUCED,
        }
    }

    /// Resolve from display preferences.
    #[must_use]
    pub const fn from_display_preferences(prefs: &DisplayPreferences) -> Self {
        Self::from_preference(prefs.motion)
    }

    /// Whether all animations are disabled.
    #[must_use]
    pub const fn is_instant(&self) -> bool {
        self.cursor_slide_ms == 0
            && self.panel_transition_ms == 0
            && self.chart_redraw_ms == 0
            && self.badge_fade_ms == 0
            && self.spinner_interval_ms == 0
    }
}

// ─── Keyboard Parity Audit ──────────────────────────────────────────────────

/// A keyboard binding entry for audit purposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyboardBinding {
    /// The action this binding triggers (e.g., `nav.fleet`, `submit_query`).
    pub action_id: String,
    /// Human-readable key description (e.g., "1", "Ctrl+P", "Enter").
    pub key_label: String,
    /// The screen scope where this binding is active (None = global).
    pub scope: Option<String>,
}

/// Result of a keyboard parity audit.
///
/// Lists all actions and whether they have keyboard bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyboardParityAudit {
    /// Actions that have keyboard bindings.
    pub covered: Vec<KeyboardBinding>,
    /// Action IDs that lack keyboard bindings (parity violations).
    pub uncovered: Vec<String>,
}

impl KeyboardParityAudit {
    /// Whether full keyboard parity is achieved (no uncovered actions).
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.uncovered.is_empty()
    }

    /// Coverage ratio (0.0..=1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage_ratio(&self) -> f64 {
        let total = self.covered.len() + self.uncovered.len();
        if total == 0 {
            return 1.0;
        }
        self.covered.len() as f64 / total as f64
    }
}

// ─── Flicker Quality ────────────────────────────────────────────────────────

/// Maximum consecutive dropped frames before triggering a quality alert.
pub const MAX_CONSECUTIVE_DROPS: u32 = 3;

/// Minimum effective frame rate before triggering degraded rendering mode.
pub const MIN_EFFECTIVE_FPS: u32 = 30;

/// Screen-level quality constraint aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityConstraints {
    /// Target frame budget in ms.
    pub frame_budget_ms: u16,
    /// Maximum acceptable P95 frame time in ms.
    pub max_p95_frame_ms: u16,
    /// Minimum on-budget frame ratio (e.g., 0.95 = 95%).
    pub min_on_budget_ratio_permille: u16,
    /// Maximum consecutive dropped frames.
    pub max_consecutive_drops: u32,
    /// Whether to enforce these constraints (vs advisory-only).
    pub enforce: bool,
}

impl Default for QualityConstraints {
    fn default() -> Self {
        Self {
            frame_budget_ms: FRAME_BUDGET_MS,
            max_p95_frame_ms: FRAME_DROP_THRESHOLD_MS,
            min_on_budget_ratio_permille: 950,
            max_consecutive_drops: MAX_CONSECUTIVE_DROPS,
            enforce: true,
        }
    }
}

impl QualityConstraints {
    /// Relaxed constraints for low-powered devices.
    #[must_use]
    pub const fn relaxed() -> Self {
        Self {
            frame_budget_ms: 33,
            max_p95_frame_ms: 50,
            min_on_budget_ratio_permille: 900,
            max_consecutive_drops: 5,
            enforce: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_quality_classify() {
        assert_eq!(
            FrameQualityVerdict::classify(10),
            FrameQualityVerdict::OnBudget
        );
        assert_eq!(
            FrameQualityVerdict::classify(16),
            FrameQualityVerdict::OnBudget
        );
        assert_eq!(
            FrameQualityVerdict::classify(17),
            FrameQualityVerdict::OverBudget
        );
        assert_eq!(
            FrameQualityVerdict::classify(33),
            FrameQualityVerdict::OverBudget
        );
        assert_eq!(
            FrameQualityVerdict::classify(34),
            FrameQualityVerdict::Dropped
        );
    }

    #[test]
    fn frame_quality_tracker_empty() {
        let tracker = FrameQualityTracker::new(100);
        assert_eq!(tracker.total_frames(), 0);
        assert!((tracker.on_budget_ratio() - 1.0).abs() < f64::EPSILON);
        assert!(tracker.slo_met());
    }

    #[test]
    fn frame_quality_tracker_all_on_budget() {
        let mut tracker = FrameQualityTracker::new(100);
        for _ in 0..50 {
            tracker.record(10);
        }
        assert_eq!(tracker.total_frames(), 50);
        assert!((tracker.on_budget_ratio() - 1.0).abs() < f64::EPSILON);
        assert_eq!(tracker.dropped_count(), 0);
        assert!(tracker.slo_met());
    }

    #[test]
    fn frame_quality_tracker_dropped_frames() {
        let mut tracker = FrameQualityTracker::new(100);
        for _ in 0..90 {
            tracker.record(10);
        }
        for _ in 0..10 {
            tracker.record(50);
        }
        assert_eq!(tracker.total_frames(), 100);
        assert_eq!(tracker.dropped_count(), 10);
        assert!((tracker.on_budget_ratio() - 0.9).abs() < f64::EPSILON);
        assert!(!tracker.slo_met());
    }

    #[test]
    fn frame_quality_tracker_p95() {
        let mut tracker = FrameQualityTracker::new(100);
        for i in 1_u16..=100 {
            tracker.record(i);
        }
        let p95 = tracker.p95_frame_time_ms();
        assert!((95..=100).contains(&p95));
    }

    #[test]
    fn animation_timing_full_has_nonzero_durations() {
        let t = AnimationTiming::FULL;
        assert!(!t.is_instant());
        assert!(t.cursor_slide_ms > 0);
        assert!(t.panel_transition_ms > 0);
    }

    #[test]
    fn animation_timing_reduced_is_instant() {
        let t = AnimationTiming::REDUCED;
        assert!(t.is_instant());
        assert_eq!(t.cursor_slide_ms, 0);
        assert_eq!(t.panel_transition_ms, 0);
        assert_eq!(t.spinner_interval_ms, 0);
    }

    #[test]
    fn animation_timing_from_preference() {
        assert_eq!(
            AnimationTiming::from_preference(MotionPreference::Full),
            AnimationTiming::FULL
        );
        assert_eq!(
            AnimationTiming::from_preference(MotionPreference::Reduced),
            AnimationTiming::REDUCED
        );
    }

    #[test]
    fn animation_timing_from_display_preferences() {
        let mut prefs = DisplayPreferences::new();
        assert_eq!(
            AnimationTiming::from_display_preferences(&prefs),
            AnimationTiming::FULL
        );
        prefs.toggle_motion();
        assert_eq!(
            AnimationTiming::from_display_preferences(&prefs),
            AnimationTiming::REDUCED
        );
    }

    #[test]
    fn keyboard_parity_audit_empty_is_complete() {
        let audit = KeyboardParityAudit {
            covered: vec![],
            uncovered: vec![],
        };
        assert!(audit.is_complete());
        assert!((audit.coverage_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn keyboard_parity_audit_with_gaps() {
        let audit = KeyboardParityAudit {
            covered: vec![
                KeyboardBinding {
                    action_id: "nav.fleet".to_string(),
                    key_label: "1".to_string(),
                    scope: None,
                },
                KeyboardBinding {
                    action_id: "nav.project".to_string(),
                    key_label: "4".to_string(),
                    scope: None,
                },
            ],
            uncovered: vec!["some.action".to_string()],
        };
        assert!(!audit.is_complete());
        let ratio = audit.coverage_ratio();
        assert!((ratio - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn quality_constraints_default() {
        let qc = QualityConstraints::default();
        assert_eq!(qc.frame_budget_ms, 16);
        assert_eq!(qc.max_p95_frame_ms, 33);
        assert_eq!(qc.min_on_budget_ratio_permille, 950);
        assert!(qc.enforce);
    }

    #[test]
    fn quality_constraints_relaxed() {
        let qc = QualityConstraints::relaxed();
        assert_eq!(qc.frame_budget_ms, 33);
        assert!(!qc.enforce);
    }

    #[test]
    fn frame_phase_labels() {
        assert_eq!(FramePhase::InputToIntent.label(), "Input → Intent");
        assert_eq!(FramePhase::StateToFrame.label(), "State → Frame");
    }

    #[test]
    fn frame_budgets_cover_all_phases() {
        assert!(
            FRAME_BUDGETS
                .iter()
                .any(|b| b.phase == FramePhase::InputToIntent)
        );
        assert!(
            FRAME_BUDGETS
                .iter()
                .any(|b| b.phase == FramePhase::IntentToState)
        );
        assert!(
            FRAME_BUDGETS
                .iter()
                .any(|b| b.phase == FramePhase::StateToFrame)
        );
        assert!(
            FRAME_BUDGETS
                .iter()
                .any(|b| b.phase == FramePhase::InputToPixels)
        );
    }

    #[test]
    fn frame_budgets_are_monotonically_increasing() {
        let budgets: Vec<u16> = FRAME_BUDGETS.iter().map(|b| b.budget_ms).collect();
        for window in budgets.windows(2) {
            assert!(window[0] <= window[1], "budgets should be non-decreasing");
        }
    }

    #[test]
    fn frame_phase_serde_roundtrip() {
        for phase in [
            FramePhase::InputToIntent,
            FramePhase::IntentToState,
            FramePhase::StateToFrame,
            FramePhase::InputToPixels,
        ] {
            let json = serde_json::to_string(&phase).unwrap();
            let decoded: FramePhase = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, phase);
        }
    }

    #[test]
    fn quality_constraints_serde_roundtrip() {
        let qc = QualityConstraints::default();
        let json = serde_json::to_string(&qc).unwrap();
        let decoded: QualityConstraints = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, qc);
    }
}
