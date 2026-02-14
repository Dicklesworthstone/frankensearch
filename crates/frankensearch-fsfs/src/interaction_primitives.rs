//! Fsfs deluxe-TUI interaction primitives ported from ftui-demo-showcase.
//!
//! This module defines the canonical contracts that all downstream fsfs screen
//! implementations (Search, Indexing, Pressure, Explainability, Configuration,
//! Timeline) must inherit. It bridges the shared [`frankensearch_tui`] framework
//! with fsfs-specific concerns:
//!
//! - **Card/layout grammar** for consistent panel organization
//! - **Cross-screen action semantics** for intent routing
//! - **Deterministic state serialization** for replay/snapshot tests
//! - **Interaction latency budget hooks** at component boundaries

use std::fmt;
use std::time::Duration;

use crate::adapters::tui::FsfsScreen;
use crate::orchestration::{BackpressureMode, WatcherThrottle};
use crate::pressure::{
    DegradationOverride as PressureDegradationOverride, DegradationStage, PressureState,
};
use crate::query_execution::DegradedRetrievalMode;

// ─── Schema Version ──────────────────────────────────────────────────────────

/// Schema version for interaction primitive contracts.
pub const INTERACTION_PRIMITIVES_SCHEMA_VERSION: u32 = 1;

// ─── Card / Layout Grammar ──────────────────────────────────────────────────

/// Canonical panel role within a screen layout.
///
/// Every fsfs screen decomposes into panels with one of these semantic roles.
/// This enables consistent keyboard focus cycling, accessibility labeling, and
/// deterministic snapshot ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelRole {
    /// Primary content area (search results list, index job list, etc.).
    Primary,
    /// Secondary detail area (score breakdown, job details, etc.).
    Detail,
    /// Filter/query input area at top of screen.
    QueryInput,
    /// Metrics/sparkline sidebar.
    Metrics,
    /// Status footer (progress, latency, degradation tier).
    StatusFooter,
    /// Evidence/explanation panel for score provenance.
    Evidence,
}

impl PanelRole {
    /// Semantic accessibility role string.
    #[must_use]
    pub const fn semantic_role(self) -> &'static str {
        match self {
            Self::Primary => "list",
            Self::Detail => "complementary",
            Self::QueryInput => "search",
            Self::Metrics => "status",
            Self::StatusFooter => "contentinfo",
            Self::Evidence => "log",
        }
    }
}

impl fmt::Display for PanelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Primary => "primary",
            Self::Detail => "detail",
            Self::QueryInput => "query_input",
            Self::Metrics => "metrics",
            Self::StatusFooter => "status_footer",
            Self::Evidence => "evidence",
        })
    }
}

/// Layout constraint for a panel within a screen.
///
/// Maps to ratatui `Constraint` semantics but expressed as a portable
/// contract that doesn't depend on the rendering backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayoutConstraint {
    /// Fixed number of terminal rows/columns.
    Fixed(u16),
    /// Percentage of available space (0.0..=100.0).
    Percentage(f32),
    /// Minimum size — takes at least this many rows/columns.
    Min(u16),
    /// Fill remaining space after fixed/percentage panels.
    Fill,
}

/// A panel descriptor within a screen layout.
#[derive(Debug, Clone, PartialEq)]
pub struct PanelDescriptor {
    /// Semantic role of this panel.
    pub role: PanelRole,
    /// Layout constraint (height for vertical layouts, width for horizontal).
    pub constraint: LayoutConstraint,
    /// Whether this panel can receive keyboard focus.
    pub focusable: bool,
    /// Tab-order index within the screen (lower = earlier in cycle).
    pub focus_order: u8,
}

/// Canonical screen layout template.
///
/// Each fsfs screen declares its layout as a sequence of panel descriptors.
/// The layout direction and panel list are fixed at build time; only the
/// data flowing into each panel changes at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenLayout {
    /// Screen this layout applies to.
    pub screen: FsfsScreen,
    /// Whether panels are arranged vertically (rows) or horizontally (columns).
    pub direction: LayoutDirection,
    /// Ordered list of panels in layout order.
    pub panels: Vec<PanelDescriptor>,
}

/// Layout direction for panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutDirection {
    /// Panels stacked top-to-bottom (most common for full-screen views).
    Vertical,
    /// Panels arranged left-to-right (used for split-pane views).
    Horizontal,
}

impl ScreenLayout {
    /// Focusable panels in tab-cycle order.
    #[must_use]
    pub fn focusable_panels(&self) -> Vec<&PanelDescriptor> {
        let mut focusable: Vec<&PanelDescriptor> =
            self.panels.iter().filter(|p| p.focusable).collect();
        focusable.sort_by_key(|p| p.focus_order);
        focusable
    }

    /// Validate layout invariants.
    ///
    /// # Errors
    ///
    /// Returns a description of the first violated invariant.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.panels.is_empty() {
            return Err("layout must contain at least one panel");
        }
        let fill_count = self
            .panels
            .iter()
            .filter(|p| matches!(p.constraint, LayoutConstraint::Fill))
            .count();
        if fill_count > 1 {
            return Err("layout may contain at most one Fill panel");
        }
        let focusable = self.focusable_panels();
        let mut orders: Vec<u8> = focusable.iter().map(|p| p.focus_order).collect();
        orders.sort_unstable();
        orders.dedup();
        if orders.len() != focusable.len() {
            return Err("focusable panels must have unique focus_order values");
        }
        Ok(())
    }
}

/// Build the canonical layout for each fsfs screen.
///
/// These layouts define the contract that screen implementations must follow.
/// Downstream beads (bd-2hz.7.2 through bd-2hz.7.6) implement actual rendering
/// against these descriptors.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn canonical_layout(screen: FsfsScreen) -> ScreenLayout {
    match screen {
        FsfsScreen::Search => ScreenLayout {
            screen,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::QueryInput,
                    constraint: LayoutConstraint::Fixed(3),
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 1,
                },
                PanelDescriptor {
                    role: PanelRole::StatusFooter,
                    constraint: LayoutConstraint::Fixed(1),
                    focusable: false,
                    focus_order: u8::MAX,
                },
            ],
        },
        FsfsScreen::Indexing => ScreenLayout {
            screen,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Detail,
                    constraint: LayoutConstraint::Percentage(30.0),
                    focusable: true,
                    focus_order: 1,
                },
                PanelDescriptor {
                    role: PanelRole::StatusFooter,
                    constraint: LayoutConstraint::Fixed(1),
                    focusable: false,
                    focus_order: u8::MAX,
                },
            ],
        },
        FsfsScreen::Pressure => ScreenLayout {
            screen,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::Metrics,
                    constraint: LayoutConstraint::Percentage(40.0),
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 1,
                },
                PanelDescriptor {
                    role: PanelRole::StatusFooter,
                    constraint: LayoutConstraint::Fixed(1),
                    focusable: false,
                    focus_order: u8::MAX,
                },
            ],
        },
        FsfsScreen::Explainability => ScreenLayout {
            screen,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Percentage(50.0),
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Evidence,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 1,
                },
                PanelDescriptor {
                    role: PanelRole::StatusFooter,
                    constraint: LayoutConstraint::Fixed(1),
                    focusable: false,
                    focus_order: u8::MAX,
                },
            ],
        },
        FsfsScreen::Configuration => ScreenLayout {
            screen,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::StatusFooter,
                    constraint: LayoutConstraint::Fixed(1),
                    focusable: false,
                    focus_order: u8::MAX,
                },
            ],
        },
        FsfsScreen::OpsTimeline => ScreenLayout {
            screen,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::QueryInput,
                    constraint: LayoutConstraint::Fixed(3),
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 1,
                },
                PanelDescriptor {
                    role: PanelRole::Detail,
                    constraint: LayoutConstraint::Percentage(25.0),
                    focusable: true,
                    focus_order: 2,
                },
                PanelDescriptor {
                    role: PanelRole::StatusFooter,
                    constraint: LayoutConstraint::Fixed(1),
                    focusable: false,
                    focus_order: u8::MAX,
                },
            ],
        },
    }
}

// ─── Cross-Screen Action Semantics ──────────────────────────────────────────

/// Semantic action intent routed from palette or keyboard to a screen.
///
/// These actions are the canonical vocabulary for cross-screen communication.
/// Each screen handles the subset relevant to it and ignores the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScreenAction {
    // -- Navigation --
    /// Navigate to a specific screen.
    NavigateTo(FsfsScreen),
    /// Focus the next panel within the current screen.
    FocusNextPanel,
    /// Focus the previous panel within the current screen.
    FocusPrevPanel,
    /// Focus a specific panel by role.
    FocusPanel(PanelRole),

    // -- List navigation --
    /// Move selection up by one row.
    SelectUp,
    /// Move selection down by one row.
    SelectDown,
    /// Jump to the first item.
    SelectFirst,
    /// Jump to the last item.
    SelectLast,
    /// Page up (viewport height).
    PageUp,
    /// Page down (viewport height).
    PageDown,

    // -- Search --
    /// Focus the query input field.
    FocusQuery,
    /// Submit the current query for execution.
    SubmitQuery,
    /// Clear the query input.
    ClearQuery,
    /// Repeat the most recent query.
    RepeatLastQuery,

    // -- Filtering --
    /// Cycle a named filter axis to the next value.
    CycleFilter(String),
    /// Clear all active filters.
    ClearAllFilters,

    // -- Details/Evidence --
    /// Toggle the detail/evidence panel visibility.
    ToggleDetailPanel,
    /// Expand the selected item's details.
    ExpandSelected,
    /// Collapse the selected item's details.
    CollapseSelected,
    /// Open the currently selected search result.
    OpenSelectedResult,
    /// Jump to the selected result's source location.
    JumpToSelectedSource,

    // -- Timeline-specific --
    /// Toggle auto-follow mode (scroll to newest events).
    ToggleFollow,

    // -- Indexing --
    /// Pause background indexing.
    PauseIndexing,
    /// Resume background indexing.
    ResumeIndexing,
    /// Apply constrained throttle to indexing/watcher cadence.
    ThrottleIndexing,
    /// Request recovery toward normal indexing/query operation.
    RecoverIndexing,
    /// Clear any manual degradation override and return to automatic policy.
    SetOverrideAuto,
    /// Force fully-enabled degradation stage.
    ForceOverrideFull,
    /// Force embed-deferred degradation stage.
    ForceOverrideEmbedDeferred,
    /// Force lexical-only degradation stage.
    ForceOverrideLexicalOnly,
    /// Force metadata-only degradation stage.
    ForceOverrideMetadataOnly,
    /// Force paused degradation stage.
    ForceOverridePaused,

    // -- Configuration --
    /// Reload configuration from disk.
    ReloadConfig,

    // -- Diagnostics --
    /// Replay the last failing trace.
    ReplayTrace,
    /// Reset collected metrics.
    ResetMetrics,

    // -- Generic --
    /// Dismiss the topmost overlay or clear the current focus.
    Dismiss,
}

impl ScreenAction {
    /// Resolve a palette action ID to a `ScreenAction`.
    ///
    /// Returns `None` for unrecognized action IDs, which the caller should
    /// handle gracefully (log and ignore).
    #[must_use]
    pub fn from_palette_action_id(action_id: &str) -> Option<Self> {
        match action_id {
            "search.focus_query" => Some(Self::FocusQuery),
            "search.submit_query" => Some(Self::SubmitQuery),
            "search.clear_query" => Some(Self::ClearQuery),
            "search.repeat_last" => Some(Self::RepeatLastQuery),
            "search.select_up" => Some(Self::SelectUp),
            "search.select_down" => Some(Self::SelectDown),
            "search.select_first" => Some(Self::SelectFirst),
            "search.select_last" => Some(Self::SelectLast),
            "search.page_up" => Some(Self::PageUp),
            "search.page_down" => Some(Self::PageDown),
            "search.toggle_explain" | "explain.toggle_panel" => Some(Self::ToggleDetailPanel),
            "search.expand_selected" => Some(Self::ExpandSelected),
            "search.collapse_selected" => Some(Self::CollapseSelected),
            "search.open_selected" => Some(Self::OpenSelectedResult),
            "search.jump_to_source" => Some(Self::JumpToSelectedSource),
            "index.pause" => Some(Self::PauseIndexing),
            "index.resume" => Some(Self::ResumeIndexing),
            "index.throttle" => Some(Self::ThrottleIndexing),
            "index.recover" => Some(Self::RecoverIndexing),
            "index.override.auto" => Some(Self::SetOverrideAuto),
            "index.override.full" => Some(Self::ForceOverrideFull),
            "index.override.embed_deferred" => Some(Self::ForceOverrideEmbedDeferred),
            "index.override.lexical_only" => Some(Self::ForceOverrideLexicalOnly),
            "index.override.metadata_only" => Some(Self::ForceOverrideMetadataOnly),
            "index.override.paused" => Some(Self::ForceOverridePaused),
            "config.reload" => Some(Self::ReloadConfig),
            "ops.open_timeline" => Some(Self::NavigateTo(FsfsScreen::OpsTimeline)),
            "diag.replay_trace" => Some(Self::ReplayTrace),
            id if id.starts_with("nav.") => {
                for screen in FsfsScreen::all() {
                    if id == format!("nav.{}", screen.id()) {
                        return Some(Self::NavigateTo(screen));
                    }
                }
                None
            }
            _ => None,
        }
    }
}

// ─── Focus Model ────────────────────────────────────────────────────────────

/// Tracks which panel within a screen currently has keyboard focus.
///
/// The focus model enforces the tab-cycle order defined in [`ScreenLayout`]
/// and provides deterministic state for snapshot/replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelFocusState {
    /// Ordered list of focusable panel roles (from layout).
    cycle: Vec<PanelRole>,
    /// Index into `cycle` of the currently focused panel.
    current: usize,
}

impl PanelFocusState {
    /// Create a focus state from a screen layout.
    ///
    /// Focuses the first focusable panel. Returns `None` if the layout has
    /// no focusable panels.
    #[must_use]
    pub fn from_layout(layout: &ScreenLayout) -> Option<Self> {
        let focusable = layout.focusable_panels();
        if focusable.is_empty() {
            return None;
        }
        let cycle: Vec<PanelRole> = focusable.iter().map(|p| p.role).collect();
        Some(Self { cycle, current: 0 })
    }

    /// Currently focused panel role.
    #[must_use]
    pub fn focused(&self) -> PanelRole {
        self.cycle[self.current]
    }

    /// Advance focus to the next panel in tab order (wraps).
    #[allow(clippy::missing_const_for_fn)]
    pub fn focus_next(&mut self) {
        self.current = (self.current + 1) % self.cycle.len();
    }

    /// Move focus to the previous panel in tab order (wraps).
    #[allow(clippy::missing_const_for_fn)]
    pub fn focus_prev(&mut self) {
        self.current = if self.current == 0 {
            self.cycle.len() - 1
        } else {
            self.current - 1
        };
    }

    /// Focus a specific panel by role. No-op if the role isn't in the cycle.
    pub fn focus_role(&mut self, role: PanelRole) {
        if let Some(idx) = self.cycle.iter().position(|r| *r == role) {
            self.current = idx;
        }
    }

    /// Number of focusable panels.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn len(&self) -> usize {
        self.cycle.len()
    }

    /// Whether there are no focusable panels.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn is_empty(&self) -> bool {
        self.cycle.is_empty()
    }
}

// ─── Deterministic State Serialization ──────────────────────────────────────

/// FNV-1a 64-bit hash for deterministic state checksums.
///
/// Used to verify that snapshot state matches expected values during
/// replay without comparing entire serialized payloads.
#[must_use]
pub const fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    let mut i = 0;
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// A deterministic snapshot of a screen's interaction state.
///
/// Screen implementations build these snapshots each frame (or on state change)
/// so that replay infrastructure can verify state convergence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionSnapshot {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Screen this snapshot belongs to.
    pub screen: FsfsScreen,
    /// Tick number from the deterministic clock (0 in live mode).
    pub tick: u64,
    /// Currently focused panel.
    pub focused_panel: PanelRole,
    /// Selected item index within the primary list (if applicable).
    pub selected_index: Option<usize>,
    /// Scroll offset for virtualized lists.
    pub scroll_offset: Option<usize>,
    /// Number of items in the filtered/visible list.
    pub visible_count: Option<usize>,
    /// Active query text (if any).
    pub query_text: Option<String>,
    /// Active filter descriptions.
    pub active_filters: Vec<String>,
    /// Whether auto-follow mode is enabled (timeline screens).
    pub follow_mode: Option<bool>,
    /// Current degradation mode.
    pub degradation_mode: DegradedRetrievalMode,
    /// FNV-1a checksum over the serialized state fields.
    pub checksum: u64,
}

impl InteractionSnapshot {
    /// Compute and set the checksum from current field values.
    ///
    /// The checksum covers: screen ID, tick, `focused_panel`, `selected_index`,
    /// `scroll_offset`, `visible_count`, `query_text`, filters, `follow_mode`,
    /// and `degradation_mode`. It does NOT include `seq` or `checksum` itself.
    #[must_use]
    pub fn with_checksum(mut self) -> Self {
        self.checksum = self.compute_checksum();
        self
    }

    /// Compute FNV-1a checksum over snapshot state fields.
    #[must_use]
    fn compute_checksum(&self) -> u64 {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(self.screen.id().as_bytes());
        buf.extend_from_slice(&self.tick.to_le_bytes());
        buf.extend_from_slice(self.focused_panel.to_string().as_bytes());
        if let Some(idx) = self.selected_index {
            buf.extend_from_slice(&idx.to_le_bytes());
        }
        if let Some(off) = self.scroll_offset {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        if let Some(cnt) = self.visible_count {
            buf.extend_from_slice(&cnt.to_le_bytes());
        }
        if let Some(ref q) = self.query_text {
            buf.extend_from_slice(q.as_bytes());
        }
        for filter in &self.active_filters {
            buf.extend_from_slice(filter.as_bytes());
        }
        if let Some(follow) = self.follow_mode {
            buf.push(u8::from(follow));
        }
        buf.extend_from_slice(&(self.degradation_mode as u8).to_le_bytes());
        fnv1a_64(&buf)
    }

    /// Verify the checksum matches the current state.
    #[must_use]
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

// ─── Interaction Latency Budget Hooks ───────────────────────────────────────

/// Latency budget phase within a single interaction cycle.
///
/// Each cycle has three measurable phases that map to the input → update →
/// render pipeline ported from the ftui-demo-showcase performance HUD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyPhase {
    /// Input handling (event dispatch, key resolution).
    Input,
    /// State update (filtering, sorting, data fetching).
    Update,
    /// Rendering (layout, paint, present).
    Render,
}

impl fmt::Display for LatencyPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Input => "input",
            Self::Update => "update",
            Self::Render => "render",
        })
    }
}

/// Per-phase latency measurement for one interaction cycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseTiming {
    pub phase: LatencyPhase,
    pub duration: Duration,
    pub budget: Duration,
}

impl PhaseTiming {
    /// Whether this phase exceeded its budget.
    #[must_use]
    pub const fn is_over_budget(&self) -> bool {
        self.duration.as_nanos() > self.budget.as_nanos()
    }

    /// Overshoot amount (zero if within budget).
    #[must_use]
    pub const fn overshoot(&self) -> Duration {
        self.duration.saturating_sub(self.budget)
    }
}

/// Budget allocation for a complete input → update → render cycle.
///
/// Default budgets target 60 FPS with balanced phase allocation:
/// - Input: 1ms (key dispatch should be near-instant)
/// - Update: 5ms (filtering/sorting for typical result sets)
/// - Render: 10ms (layout + paint within remaining frame budget)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionBudget {
    pub input_budget: Duration,
    pub update_budget: Duration,
    pub render_budget: Duration,
}

impl Default for InteractionBudget {
    fn default() -> Self {
        Self::at_60fps()
    }
}

impl InteractionBudget {
    /// 60 FPS budget (1ms input + 5ms update + 10ms render = 16ms total).
    #[must_use]
    pub const fn at_60fps() -> Self {
        Self {
            input_budget: Duration::from_millis(1),
            update_budget: Duration::from_millis(5),
            render_budget: Duration::from_millis(10),
        }
    }

    /// 30 FPS budget (2ms input + 10ms update + 20ms render = 32ms total).
    #[must_use]
    pub const fn at_30fps() -> Self {
        Self {
            input_budget: Duration::from_millis(2),
            update_budget: Duration::from_millis(10),
            render_budget: Duration::from_millis(20),
        }
    }

    /// Total cycle budget across all phases.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub const fn total(&self) -> Duration {
        Duration::from_nanos(
            self.input_budget.as_nanos() as u64
                + self.update_budget.as_nanos() as u64
                + self.render_budget.as_nanos() as u64,
        )
    }

    /// Budget for a specific phase.
    #[must_use]
    pub const fn for_phase(&self, phase: LatencyPhase) -> Duration {
        match phase {
            LatencyPhase::Input => self.input_budget,
            LatencyPhase::Update => self.update_budget,
            LatencyPhase::Render => self.render_budget,
        }
    }

    /// Degraded budget: widens update + render budgets to accommodate
    /// pressure-driven slowdowns without marking every frame as jank.
    #[must_use]
    pub const fn degraded(mode: DegradedRetrievalMode) -> Self {
        match mode {
            DegradedRetrievalMode::Normal => Self::at_60fps(),
            DegradedRetrievalMode::EmbedDeferred => Self {
                input_budget: Duration::from_millis(1),
                update_budget: Duration::from_millis(8),
                render_budget: Duration::from_millis(12),
            },
            DegradedRetrievalMode::LexicalOnly => Self::at_30fps(),
            DegradedRetrievalMode::MetadataOnly | DegradedRetrievalMode::Paused => Self {
                input_budget: Duration::from_millis(2),
                update_budget: Duration::from_millis(15),
                render_budget: Duration::from_millis(30),
            },
        }
    }
}

/// Collected timing for a complete interaction cycle.
///
/// Consumers build this by measuring each phase and then check
/// budget compliance against an [`InteractionBudget`].
#[derive(Debug, Clone)]
pub struct InteractionCycleTiming {
    pub input: PhaseTiming,
    pub update: PhaseTiming,
    pub render: PhaseTiming,
    pub frame_seq: u64,
}

impl InteractionCycleTiming {
    /// Total duration of the cycle across all phases.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.input.duration + self.update.duration + self.render.duration
    }

    /// Whether any phase exceeded its individual budget.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn has_phase_overrun(&self) -> bool {
        self.input.is_over_budget() || self.update.is_over_budget() || self.render.is_over_budget()
    }

    /// List of phases that exceeded their budgets.
    #[must_use]
    pub fn overrun_phases(&self) -> Vec<LatencyPhase> {
        let mut overruns = Vec::new();
        if self.input.is_over_budget() {
            overruns.push(LatencyPhase::Input);
        }
        if self.update.is_over_budget() {
            overruns.push(LatencyPhase::Update);
        }
        if self.render.is_over_budget() {
            overruns.push(LatencyPhase::Render);
        }
        overruns
    }

    /// Coarse latency bucket for artifact-friendly telemetry output.
    #[must_use]
    pub fn latency_bucket(&self, budget: &InteractionBudget) -> LatencyBucket {
        LatencyBucket::from_totals(self.total_duration(), budget.total())
    }
}

/// Coarse interaction latency category used by replay/diagnostic artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyBucket {
    /// Uses <= 80% of frame budget.
    UnderBudget,
    /// Uses > 80% and <= 100% of frame budget.
    NearBudget,
    /// Exceeds frame budget.
    OverBudget,
}

impl LatencyBucket {
    /// Derive a bucket from observed total duration and frame budget.
    #[must_use]
    pub const fn from_totals(total: Duration, budget: Duration) -> Self {
        let total_ns = total.as_nanos();
        let budget_ns = budget.as_nanos();
        if budget_ns == 0 {
            return Self::OverBudget;
        }
        if total_ns <= (budget_ns * 4) / 5 {
            Self::UnderBudget
        } else if total_ns <= budget_ns {
            Self::NearBudget
        } else {
            Self::OverBudget
        }
    }
}

impl fmt::Display for LatencyBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnderBudget => "under_budget",
            Self::NearBudget => "near_budget",
            Self::OverBudget => "over_budget",
        })
    }
}

/// Deterministic search interaction telemetry payload for logs/replay artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchInteractionTelemetry {
    pub interaction_id: u64,
    pub frame_seq: u64,
    pub frame_budget_ms: u16,
    pub total_latency_ms: u16,
    pub latency_bucket: LatencyBucket,
    pub visible_window: (usize, usize),
    pub visible_count: usize,
    pub selected_index: usize,
    pub detail_panel_visible: bool,
    pub query_len: usize,
}

// ─── Degradation Tier (presentation layer) ──────────────────────────────────

/// Presentation-layer degradation tier derived from frame timing.
///
/// Mirrors the ftui-demo-showcase's four-tier model and drives rendering
/// complexity decisions (animations, sparklines, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RenderTier {
    /// Full fidelity: animations, sparklines, color gradients.
    Full,
    /// Reduced: disable animations, simplify sparklines.
    Reduced,
    /// Minimal: text-only, no charts.
    Minimal,
    /// Safety: bare minimum rendering to keep the shell responsive.
    Safety,
}

impl RenderTier {
    /// Determine render tier from observed FPS.
    #[must_use]
    pub const fn from_fps(fps: u32) -> Self {
        if fps >= 50 {
            Self::Full
        } else if fps >= 20 {
            Self::Reduced
        } else if fps >= 5 {
            Self::Minimal
        } else {
            Self::Safety
        }
    }

    /// Whether animations should be rendered at this tier.
    #[must_use]
    pub const fn animations_enabled(self) -> bool {
        matches!(self, Self::Full)
    }

    /// Whether chart/sparkline widgets should be rendered.
    #[must_use]
    pub const fn charts_enabled(self) -> bool {
        matches!(self, Self::Full | Self::Reduced)
    }
}

impl fmt::Display for RenderTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Full => "full",
            Self::Reduced => "reduced",
            Self::Minimal => "minimal",
            Self::Safety => "safety",
        })
    }
}

// ─── Virtualized List Contract ──────────────────────────────────────────────

/// State for a virtualized scrollable list.
///
/// Ported from the ftui-demo-showcase's `VirtualizedSearchScreen` pattern.
/// Provides viewport-aware navigation with `ensure_visible` semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualizedListState {
    /// Total number of items (after filtering).
    pub total_items: usize,
    /// Currently selected item index.
    pub selected: usize,
    /// First visible item index (scroll position).
    pub scroll_offset: usize,
    /// Number of visible rows in the viewport.
    pub viewport_height: usize,
}

impl VirtualizedListState {
    /// Create a new list state with zero items.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            total_items: 0,
            selected: 0,
            scroll_offset: 0,
            viewport_height: 0,
        }
    }

    /// Ensure the selected item is visible by adjusting `scroll_offset`.
    #[allow(clippy::missing_const_for_fn)]
    pub fn ensure_visible(&mut self) {
        if self.viewport_height == 0 {
            return;
        }
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + self.viewport_height {
            self.scroll_offset = self.selected + 1 - self.viewport_height;
        }
    }

    /// Move selection down by one, clamping to bounds.
    pub fn select_next(&mut self) {
        if self.total_items > 0 && self.selected < self.total_items - 1 {
            self.selected += 1;
            self.ensure_visible();
        }
    }

    /// Move selection up by one, clamping to bounds.
    pub fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
            self.ensure_visible();
        }
    }

    /// Jump to first item.
    pub fn select_first(&mut self) {
        self.selected = 0;
        self.ensure_visible();
    }

    /// Jump to last item.
    pub fn select_last(&mut self) {
        if self.total_items > 0 {
            self.selected = self.total_items - 1;
            self.ensure_visible();
        }
    }

    /// Page down (move by viewport height).
    pub fn page_down(&mut self) {
        if self.total_items == 0 {
            return;
        }
        let jump = self.viewport_height.max(1);
        self.selected = (self.selected + jump).min(self.total_items - 1);
        self.ensure_visible();
    }

    /// Page up (move by viewport height).
    pub fn page_up(&mut self) {
        let jump = self.viewport_height.max(1);
        self.selected = self.selected.saturating_sub(jump);
        self.ensure_visible();
    }

    /// Update total items (e.g., after re-filtering) and clamp selection.
    pub fn set_total_items(&mut self, count: usize) {
        self.total_items = count;
        if count == 0 {
            self.selected = 0;
            self.scroll_offset = 0;
        } else if self.selected >= count {
            self.selected = count - 1;
        }
        self.ensure_visible();
    }

    /// Update viewport height (e.g., after terminal resize).
    pub fn set_viewport_height(&mut self, height: usize) {
        self.viewport_height = height;
        self.ensure_visible();
    }
}

/// Minimal result payload tracked by the interactive search state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResultEntry {
    pub doc_id: String,
    pub source_path: String,
    pub snippet: String,
}

impl SearchResultEntry {
    #[must_use]
    pub fn new(
        doc_id: impl Into<String>,
        source_path: impl Into<String>,
        snippet: impl Into<String>,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            source_path: source_path.into(),
            snippet: snippet.into(),
        }
    }
}

/// Event emitted when search interaction state triggers an external operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchInteractionEvent {
    QuerySubmitted(String),
    OpenSelected { doc_id: String, source_path: String },
    JumpToSource { doc_id: String, source_path: String },
}

/// Stateful contract for the fsfs interactive search screen.
///
/// This model handles incremental query updates, virtualized result navigation,
/// inline explainability toggles, and explicit open/jump actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchInteractionState {
    pub focus: PanelFocusState,
    pub list: VirtualizedListState,
    pub query_input: String,
    pub pending_incremental_query: Option<String>,
    pub last_submitted_query: Option<String>,
    pub detail_panel_visible: bool,
    pub results: Vec<SearchResultEntry>,
}

impl SearchInteractionState {
    /// Build the default search interaction state with fixed viewport height.
    #[must_use]
    pub fn new(viewport_height: usize) -> Self {
        let layout = canonical_layout(FsfsScreen::Search);
        let focus = PanelFocusState::from_layout(&layout).unwrap_or_else(|| PanelFocusState {
            cycle: vec![PanelRole::QueryInput],
            current: 0,
        });
        let mut list = VirtualizedListState::empty();
        list.set_viewport_height(viewport_height);
        Self {
            focus,
            list,
            query_input: String::new(),
            pending_incremental_query: None,
            last_submitted_query: None,
            detail_panel_visible: false,
            results: Vec::new(),
        }
    }

    /// Apply an incremental query edit and queue it for submit.
    pub fn apply_incremental_query(&mut self, query: impl Into<String>) {
        let query = query.into();
        self.query_input.clone_from(&query);
        let trimmed = query.trim();
        self.pending_incremental_query = (!trimmed.is_empty()).then(|| trimmed.to_owned());
    }

    /// Replace result rows and synchronize virtualization bounds.
    pub fn set_results(&mut self, results: Vec<SearchResultEntry>) {
        self.results = results;
        self.list.set_total_items(self.results.len());
    }

    /// Current visible window as `[start, end)` over `results`.
    #[must_use]
    pub fn visible_window(&self) -> (usize, usize) {
        let start = self.list.scroll_offset.min(self.results.len());
        let end = (start + self.list.viewport_height.max(1)).min(self.results.len());
        (start, end)
    }

    /// Borrow the currently visible result rows.
    #[must_use]
    pub fn visible_results(&self) -> &[SearchResultEntry] {
        let (start, end) = self.visible_window();
        &self.results[start..end]
    }

    /// Selected result, if any.
    #[must_use]
    pub fn selected_result(&self) -> Option<&SearchResultEntry> {
        self.results.get(self.list.selected)
    }

    /// Build deterministic telemetry artifact fields for one interaction cycle.
    ///
    /// Includes required evidence fields: `interaction_id`, `visible_window`,
    /// `frame_budget_ms`, and `latency_bucket`.
    #[must_use]
    pub fn telemetry_sample(
        &self,
        cycle: &InteractionCycleTiming,
        budget: &InteractionBudget,
    ) -> SearchInteractionTelemetry {
        let visible_window = self.visible_window();
        let total_latency = cycle.total_duration();
        SearchInteractionTelemetry {
            interaction_id: self.compute_interaction_id(cycle.frame_seq, visible_window),
            frame_seq: cycle.frame_seq,
            frame_budget_ms: saturating_duration_ms_u16(budget.total()),
            total_latency_ms: saturating_duration_ms_u16(total_latency),
            latency_bucket: cycle.latency_bucket(budget),
            visible_window,
            visible_count: visible_window.1.saturating_sub(visible_window.0),
            selected_index: self.list.selected,
            detail_panel_visible: self.detail_panel_visible,
            query_len: self.query_input.trim().chars().count(),
        }
    }

    /// Apply one semantic action and return any external side-effect event.
    pub fn apply_action(&mut self, action: &ScreenAction) -> Option<SearchInteractionEvent> {
        match action {
            ScreenAction::FocusNextPanel => self.focus.focus_next(),
            ScreenAction::FocusPrevPanel => self.focus.focus_prev(),
            ScreenAction::FocusPanel(role) => self.focus.focus_role(*role),
            ScreenAction::SelectUp => self.list.select_prev(),
            ScreenAction::SelectDown => self.list.select_next(),
            ScreenAction::SelectFirst => self.list.select_first(),
            ScreenAction::SelectLast => self.list.select_last(),
            ScreenAction::PageUp => self.list.page_up(),
            ScreenAction::PageDown => self.list.page_down(),
            ScreenAction::FocusQuery => self.focus.focus_role(PanelRole::QueryInput),
            ScreenAction::SubmitQuery => return self.submit_current_query(),
            ScreenAction::ClearQuery => {
                self.query_input.clear();
                self.pending_incremental_query = None;
            }
            ScreenAction::RepeatLastQuery => return self.repeat_last_query(),
            ScreenAction::ToggleDetailPanel => {
                self.detail_panel_visible = !self.detail_panel_visible;
            }
            ScreenAction::ExpandSelected => {
                self.detail_panel_visible = true;
            }
            ScreenAction::CollapseSelected => {
                self.detail_panel_visible = false;
            }
            ScreenAction::OpenSelectedResult => {
                return self
                    .selected_result()
                    .map(|entry| SearchInteractionEvent::OpenSelected {
                        doc_id: entry.doc_id.clone(),
                        source_path: entry.source_path.clone(),
                    });
            }
            ScreenAction::JumpToSelectedSource => {
                return self
                    .selected_result()
                    .map(|entry| SearchInteractionEvent::JumpToSource {
                        doc_id: entry.doc_id.clone(),
                        source_path: entry.source_path.clone(),
                    });
            }
            _ => {}
        }
        None
    }

    fn submit_current_query(&mut self) -> Option<SearchInteractionEvent> {
        if let Some(query) = self.pending_incremental_query.take() {
            self.last_submitted_query = Some(query.clone());
            return Some(SearchInteractionEvent::QuerySubmitted(query));
        }

        let trimmed = self.query_input.trim();
        if trimmed.is_empty() {
            return None;
        }

        let query = trimmed.to_owned();
        self.last_submitted_query = Some(query.clone());
        Some(SearchInteractionEvent::QuerySubmitted(query))
    }

    fn repeat_last_query(&mut self) -> Option<SearchInteractionEvent> {
        let last = self.last_submitted_query.clone()?;
        self.query_input.clone_from(&last);
        self.pending_incremental_query = Some(last);
        self.submit_current_query()
    }

    fn compute_interaction_id(&self, frame_seq: u64, visible_window: (usize, usize)) -> u64 {
        let mut buf = Vec::new();
        buf.extend_from_slice(&frame_seq.to_le_bytes());
        append_usize_as_u64_le(&mut buf, visible_window.0);
        append_usize_as_u64_le(&mut buf, visible_window.1);
        append_usize_as_u64_le(&mut buf, self.list.selected);
        buf.push(u8::from(self.detail_panel_visible));
        buf.extend_from_slice(self.query_input.trim().as_bytes());
        if let Some(last) = &self.last_submitted_query {
            buf.extend_from_slice(last.as_bytes());
        }
        fnv1a_64(&buf)
    }
}

fn saturating_duration_ms_u16(duration: Duration) -> u16 {
    u16::try_from(duration.as_millis()).unwrap_or(u16::MAX)
}

fn append_usize_as_u64_le(buf: &mut Vec<u8>, value: usize) {
    let as_u64 = u64::try_from(value).unwrap_or(u64::MAX);
    buf.extend_from_slice(&as_u64.to_le_bytes());
}

// ─── Filter Cycling ─────────────────────────────────────────────────────────

/// A cycleable filter axis with a finite set of values.
///
/// Ported from the ftui-demo-showcase timeline's cyclic filter pattern.
/// Cycles through: None → Value(0) → Value(1) → ... → None.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclicFilter {
    /// Human label for this filter axis (e.g., "Severity").
    pub label: String,
    /// Available filter values.
    pub values: Vec<String>,
    /// Current selection: `None` means "show all".
    pub selected: Option<usize>,
}

impl CyclicFilter {
    /// Create a new filter with no active selection.
    #[must_use]
    pub fn new(label: impl Into<String>, values: Vec<String>) -> Self {
        Self {
            label: label.into(),
            values,
            selected: None,
        }
    }

    /// Advance to the next value in the cycle (wraps through None).
    #[allow(clippy::missing_const_for_fn)]
    pub fn cycle_next(&mut self) {
        self.selected = match self.selected {
            None if self.values.is_empty() => None,
            None => Some(0),
            Some(idx) => match idx.checked_add(1) {
                Some(next) if next < self.values.len() => Some(next),
                _ => None,
            },
        };
    }

    /// Active filter value, or `None` for "show all".
    #[must_use]
    pub fn active_value(&self) -> Option<&str> {
        self.selected
            .and_then(|idx| self.values.get(idx).map(String::as_str))
    }

    /// Clear the filter (back to "show all").
    #[allow(clippy::missing_const_for_fn)]
    pub fn clear(&mut self) {
        self.selected = None;
    }

    /// Display string for the current state.
    #[must_use]
    pub fn display(&self) -> String {
        self.active_value().map_or_else(
            || format!("{}: all", self.label),
            |value| format!("{}: {value}", self.label),
        )
    }
}

// ─── Indexing Cockpit Contracts ────────────────────────────────────────────

/// Directional trend used by backlog/throughput visualizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Rising,
    Stable,
    Falling,
}

impl TrendDirection {
    #[must_use]
    pub fn from_delta(delta_per_min: f64) -> Self {
        const EPSILON: f64 = 0.05;
        if delta_per_min > EPSILON {
            Self::Rising
        } else if delta_per_min < -EPSILON {
            Self::Falling
        } else {
            Self::Stable
        }
    }
}

/// One throughput datapoint for cockpit charting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThroughputSample {
    pub timestamp_ms: u64,
    pub docs_per_min: f64,
    pub embeds_per_min: f64,
}

impl ThroughputSample {
    #[must_use]
    pub const fn new(timestamp_ms: u64, docs_per_min: f64, embeds_per_min: f64) -> Self {
        Self {
            timestamp_ms,
            docs_per_min,
            embeds_per_min,
        }
    }
}

/// Backlog visualization state for indexing/job cockpit screens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacklogVisualization {
    pub current_depth: usize,
    pub high_watermark: usize,
    pub hard_limit: usize,
    pub pending_replay_items: usize,
    pub trend: TrendDirection,
}

impl BacklogVisualization {
    /// Construct normalized backlog visualization state.
    #[must_use]
    pub fn new(
        current_depth: usize,
        high_watermark: usize,
        hard_limit: usize,
        pending_replay_items: usize,
        delta_per_min: f64,
    ) -> Self {
        let normalized_high = high_watermark.max(1);
        let normalized_hard = hard_limit.max(normalized_high);
        Self {
            current_depth,
            high_watermark: normalized_high,
            hard_limit: normalized_hard,
            pending_replay_items,
            trend: TrendDirection::from_delta(delta_per_min),
        }
    }

    /// Queue utilization against hard limit in the range `0.0..=1.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization_ratio(&self) -> f64 {
        if self.hard_limit == 0 {
            return 0.0;
        }
        let depth = self.current_depth.min(self.hard_limit) as f64;
        let hard = self.hard_limit as f64;
        (depth / hard).clamp(0.0, 1.0)
    }

    /// Whether queue pressure warrants operator attention.
    #[must_use]
    pub const fn is_hot(&self) -> bool {
        self.current_depth >= self.high_watermark
    }
}

/// Throughput visualization contract for indexing cockpit screens.
#[derive(Debug, Clone, PartialEq)]
pub struct ThroughputVisualization {
    pub recent_samples: Vec<ThroughputSample>,
    pub rolling_docs_per_min: f64,
    pub rolling_embeds_per_min: f64,
}

impl ThroughputVisualization {
    /// Build a rolling throughput view from recent samples.
    #[must_use]
    pub fn from_samples(samples: Vec<ThroughputSample>) -> Self {
        let mut docs_avg = 0.0_f64;
        let mut embeds_avg = 0.0_f64;
        let mut seen = 0.0_f64;

        for sample in &samples {
            seen += 1.0;
            docs_avg += (sample.docs_per_min - docs_avg) / seen;
            embeds_avg += (sample.embeds_per_min - embeds_avg) / seen;
        }

        Self {
            recent_samples: samples,
            rolling_docs_per_min: docs_avg.max(0.0),
            rolling_embeds_per_min: embeds_avg.max(0.0),
        }
    }

    /// True when both document and embedding throughput are effectively stalled.
    #[must_use]
    pub fn is_stalled(&self) -> bool {
        self.rolling_docs_per_min < 1.0 && self.rolling_embeds_per_min < 1.0
    }
}

/// Pressure/degradation indicator payload rendered in cockpit screens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePressureIndicator {
    pub pressure_state: PressureState,
    pub degradation_stage: DegradationStage,
    pub backpressure_mode: BackpressureMode,
    pub throttle: WatcherThrottle,
    pub user_banner: &'static str,
    pub transition_reason_code: &'static str,
    pub override_mode: PressureDegradationOverride,
    pub override_allowed: bool,
    pub reason_code: &'static str,
}

impl ResourcePressureIndicator {
    /// Whether pressure/degradation state currently needs operator attention.
    #[must_use]
    pub const fn requires_attention(&self) -> bool {
        !matches!(self.pressure_state, PressureState::Normal)
            || !matches!(self.degradation_stage, DegradationStage::Full)
            || !matches!(self.backpressure_mode, BackpressureMode::Normal)
    }

    /// Whether indexing is effectively paused/suspended.
    #[must_use]
    pub const fn indexing_paused(&self) -> bool {
        matches!(self.degradation_stage, DegradationStage::Paused) || self.throttle.suspended
    }

    /// Whether manual override is currently active.
    #[must_use]
    pub const fn manual_override_active(&self) -> bool {
        !matches!(self.override_mode, PressureDegradationOverride::Auto)
    }

    /// Build a user-facing banner projection for deterministic UI rendering.
    #[must_use]
    pub const fn banner(&self) -> DegradationBanner {
        DegradationBanner {
            stage: self.degradation_stage,
            text: self.user_banner,
            transition_reason_code: self.transition_reason_code,
            override_mode: self.override_mode,
            override_allowed: self.override_allowed,
        }
    }
}

/// User-facing degradation banner with transition context and override state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DegradationBanner {
    pub stage: DegradationStage,
    pub text: &'static str,
    pub transition_reason_code: &'static str,
    pub override_mode: PressureDegradationOverride,
    pub override_allowed: bool,
}

/// Operator control taxonomy for indexing/resource cockpit screens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CockpitControlKind {
    PauseIndexing,
    ResumeIndexing,
    ThrottleIndexing,
    RecoverIndexing,
    SetOverrideAuto,
    ForceOverrideFull,
    ForceOverrideEmbedDeferred,
    ForceOverrideLexicalOnly,
    ForceOverrideMetadataOnly,
    ForceOverridePaused,
}

/// One actionable cockpit control with deterministic enablement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CockpitControl {
    pub kind: CockpitControlKind,
    pub action: ScreenAction,
    pub label: &'static str,
    pub enabled: bool,
    pub reason_code: &'static str,
}

impl CockpitControl {
    #[must_use]
    pub const fn new(
        kind: CockpitControlKind,
        action: ScreenAction,
        label: &'static str,
        enabled: bool,
        reason_code: &'static str,
    ) -> Self {
        Self {
            kind,
            action,
            label,
            enabled,
            reason_code,
        }
    }
}

/// Real-time indexing/jobs/resource cockpit snapshot contract.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexingCockpitSnapshot {
    pub timestamp_ms: u64,
    pub backlog: BacklogVisualization,
    pub throughput: ThroughputVisualization,
    pub pressure: ResourcePressureIndicator,
    pub controls: Vec<CockpitControl>,
}

impl IndexingCockpitSnapshot {
    /// Construct snapshot and derive default operator controls.
    #[must_use]
    pub fn new(
        timestamp_ms: u64,
        backlog: BacklogVisualization,
        throughput: ThroughputVisualization,
        pressure: ResourcePressureIndicator,
    ) -> Self {
        let controls = Self::derive_controls(&pressure);
        Self {
            timestamp_ms,
            backlog,
            throughput,
            pressure,
            controls,
        }
    }

    fn derive_controls(pressure: &ResourcePressureIndicator) -> Vec<CockpitControl> {
        let mut controls = Self::derive_indexing_controls(pressure);
        controls.extend(Self::derive_override_controls(pressure));
        controls
    }

    fn derive_indexing_controls(pressure: &ResourcePressureIndicator) -> Vec<CockpitControl> {
        let pause_enabled = !pressure.indexing_paused();
        let resume_enabled = pressure.indexing_paused();
        let throttle_enabled = pressure.requires_attention()
            && !matches!(pressure.backpressure_mode, BackpressureMode::Saturated);
        let recovery_needed = !matches!(pressure.degradation_stage, DegradationStage::Full)
            || pressure.throttle.suspended;
        let recover_enabled = matches!(pressure.pressure_state, PressureState::Normal)
            && matches!(pressure.backpressure_mode, BackpressureMode::Normal)
            && recovery_needed;

        vec![
            CockpitControl::new(
                CockpitControlKind::PauseIndexing,
                ScreenAction::PauseIndexing,
                "Pause indexing",
                pause_enabled,
                "cockpit.control.pause",
            ),
            CockpitControl::new(
                CockpitControlKind::ResumeIndexing,
                ScreenAction::ResumeIndexing,
                "Resume indexing",
                resume_enabled,
                "cockpit.control.resume",
            ),
            CockpitControl::new(
                CockpitControlKind::ThrottleIndexing,
                ScreenAction::ThrottleIndexing,
                "Throttle indexing",
                throttle_enabled,
                "cockpit.control.throttle",
            ),
            CockpitControl::new(
                CockpitControlKind::RecoverIndexing,
                ScreenAction::RecoverIndexing,
                "Recover normal mode",
                recover_enabled,
                "cockpit.control.recover",
            ),
        ]
    }

    fn derive_override_controls(pressure: &ResourcePressureIndicator) -> Vec<CockpitControl> {
        let override_auto_enabled = pressure.override_allowed && pressure.manual_override_active();
        let force_full_enabled = pressure.override_allowed
            && matches!(pressure.pressure_state, PressureState::Normal)
            && matches!(pressure.backpressure_mode, BackpressureMode::Normal)
            && !matches!(
                pressure.override_mode,
                PressureDegradationOverride::ForceFull
            );
        let force_embed_deferred_enabled = pressure.override_allowed
            && !matches!(pressure.degradation_stage, DegradationStage::Paused)
            && !matches!(
                pressure.override_mode,
                PressureDegradationOverride::ForceEmbedDeferred
            );
        let force_lexical_only_enabled = pressure.override_allowed
            && pressure.requires_attention()
            && !matches!(
                pressure.override_mode,
                PressureDegradationOverride::ForceLexicalOnly
            );
        let force_metadata_only_enabled = pressure.override_allowed
            && pressure.requires_attention()
            && !matches!(
                pressure.override_mode,
                PressureDegradationOverride::ForceMetadataOnly
            );
        let force_paused_enabled = pressure.override_allowed
            && pressure.requires_attention()
            && !matches!(
                pressure.override_mode,
                PressureDegradationOverride::ForcePaused
            );

        vec![
            CockpitControl::new(
                CockpitControlKind::SetOverrideAuto,
                ScreenAction::SetOverrideAuto,
                "Override: auto policy",
                override_auto_enabled,
                "cockpit.control.override.auto",
            ),
            CockpitControl::new(
                CockpitControlKind::ForceOverrideFull,
                ScreenAction::ForceOverrideFull,
                "Override: force full",
                force_full_enabled,
                "cockpit.control.override.force_full",
            ),
            CockpitControl::new(
                CockpitControlKind::ForceOverrideEmbedDeferred,
                ScreenAction::ForceOverrideEmbedDeferred,
                "Override: force embed-deferred",
                force_embed_deferred_enabled,
                "cockpit.control.override.force_embed_deferred",
            ),
            CockpitControl::new(
                CockpitControlKind::ForceOverrideLexicalOnly,
                ScreenAction::ForceOverrideLexicalOnly,
                "Override: force lexical-only",
                force_lexical_only_enabled,
                "cockpit.control.override.force_lexical_only",
            ),
            CockpitControl::new(
                CockpitControlKind::ForceOverrideMetadataOnly,
                ScreenAction::ForceOverrideMetadataOnly,
                "Override: force metadata-only",
                force_metadata_only_enabled,
                "cockpit.control.override.force_metadata_only",
            ),
            CockpitControl::new(
                CockpitControlKind::ForceOverridePaused,
                ScreenAction::ForceOverridePaused,
                "Override: force pause",
                force_paused_enabled,
                "cockpit.control.override.force_paused",
            ),
        ]
    }

    /// Enabled actions that should be surfaced as quick controls.
    #[must_use]
    pub fn enabled_actions(&self) -> Vec<ScreenAction> {
        self.controls
            .iter()
            .filter(|control| control.enabled)
            .map(|control| control.action.clone())
            .collect()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- PanelRole --

    #[test]
    fn panel_role_semantic_roles_are_non_empty() {
        for role in [
            PanelRole::Primary,
            PanelRole::Detail,
            PanelRole::QueryInput,
            PanelRole::Metrics,
            PanelRole::StatusFooter,
            PanelRole::Evidence,
        ] {
            assert!(!role.semantic_role().is_empty());
            assert!(!role.to_string().is_empty());
        }
    }

    // -- ScreenLayout --

    #[test]
    fn all_canonical_layouts_are_valid() {
        for screen in FsfsScreen::all() {
            let layout = canonical_layout(screen);
            layout
                .validate()
                .unwrap_or_else(|e| panic!("layout for {} invalid: {e}", screen.id()));
            assert!(
                !layout.panels.is_empty(),
                "layout for {} has no panels",
                screen.id()
            );
        }
    }

    #[test]
    fn search_layout_has_query_input_and_primary() {
        let layout = canonical_layout(FsfsScreen::Search);
        let roles: Vec<PanelRole> = layout.panels.iter().map(|p| p.role).collect();
        assert!(roles.contains(&PanelRole::QueryInput));
        assert!(roles.contains(&PanelRole::Primary));
    }

    #[test]
    fn focusable_panels_in_order() {
        let layout = canonical_layout(FsfsScreen::OpsTimeline);
        let focusable = layout.focusable_panels();
        assert!(focusable.len() >= 2);
        for w in focusable.windows(2) {
            assert!(w[0].focus_order < w[1].focus_order);
        }
    }

    #[test]
    fn validate_rejects_empty_layout() {
        let layout = ScreenLayout {
            screen: FsfsScreen::Search,
            direction: LayoutDirection::Vertical,
            panels: vec![],
        };
        assert!(layout.validate().is_err());
    }

    #[test]
    fn validate_rejects_multiple_fill() {
        let layout = ScreenLayout {
            screen: FsfsScreen::Search,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Detail,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 1,
                },
            ],
        };
        assert!(layout.validate().is_err());
    }

    #[test]
    fn validate_rejects_duplicate_focus_order() {
        let layout = ScreenLayout {
            screen: FsfsScreen::Search,
            direction: LayoutDirection::Vertical,
            panels: vec![
                PanelDescriptor {
                    role: PanelRole::Primary,
                    constraint: LayoutConstraint::Fill,
                    focusable: true,
                    focus_order: 0,
                },
                PanelDescriptor {
                    role: PanelRole::Detail,
                    constraint: LayoutConstraint::Fixed(10),
                    focusable: true,
                    focus_order: 0,
                },
            ],
        };
        assert!(layout.validate().is_err());
    }

    // -- ScreenAction --

    #[test]
    fn palette_action_resolution() {
        assert_eq!(
            ScreenAction::from_palette_action_id("search.focus_query"),
            Some(ScreenAction::FocusQuery)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("search.submit_query"),
            Some(ScreenAction::SubmitQuery)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("search.clear_query"),
            Some(ScreenAction::ClearQuery)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("search.toggle_explain"),
            Some(ScreenAction::ToggleDetailPanel)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("search.open_selected"),
            Some(ScreenAction::OpenSelectedResult)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("search.jump_to_source"),
            Some(ScreenAction::JumpToSelectedSource)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("index.pause"),
            Some(ScreenAction::PauseIndexing)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("nav.fsfs.search"),
            Some(ScreenAction::NavigateTo(FsfsScreen::Search))
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("nav.fsfs.timeline"),
            Some(ScreenAction::NavigateTo(FsfsScreen::OpsTimeline))
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("index.throttle"),
            Some(ScreenAction::ThrottleIndexing)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("index.recover"),
            Some(ScreenAction::RecoverIndexing)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("index.override.auto"),
            Some(ScreenAction::SetOverrideAuto)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("index.override.lexical_only"),
            Some(ScreenAction::ForceOverrideLexicalOnly)
        );
        assert_eq!(
            ScreenAction::from_palette_action_id("index.override.paused"),
            Some(ScreenAction::ForceOverridePaused)
        );
        assert_eq!(ScreenAction::from_palette_action_id("unknown.action"), None);
    }

    // -- PanelFocusState --

    #[test]
    fn focus_state_cycles_through_panels() {
        let layout = canonical_layout(FsfsScreen::Search);
        let mut focus = PanelFocusState::from_layout(&layout).unwrap();
        assert_eq!(focus.focused(), PanelRole::QueryInput);

        focus.focus_next();
        assert_eq!(focus.focused(), PanelRole::Primary);

        focus.focus_next();
        // Wraps back to first.
        assert_eq!(focus.focused(), PanelRole::QueryInput);

        focus.focus_prev();
        assert_eq!(focus.focused(), PanelRole::Primary);
    }

    #[test]
    fn focus_state_focus_by_role() {
        let layout = canonical_layout(FsfsScreen::OpsTimeline);
        let mut focus = PanelFocusState::from_layout(&layout).unwrap();
        focus.focus_role(PanelRole::Detail);
        assert_eq!(focus.focused(), PanelRole::Detail);
    }

    #[test]
    fn focus_state_ignores_unknown_role() {
        let layout = canonical_layout(FsfsScreen::Configuration);
        let mut focus = PanelFocusState::from_layout(&layout).unwrap();
        let before = focus.focused();
        focus.focus_role(PanelRole::Evidence); // Not in config layout.
        assert_eq!(focus.focused(), before);
    }

    // -- FNV-1a --

    #[test]
    fn fnv1a_deterministic() {
        let hash1 = fnv1a_64(b"hello");
        let hash2 = fnv1a_64(b"hello");
        assert_eq!(hash1, hash2);

        let hash3 = fnv1a_64(b"world");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn fnv1a_empty_is_offset_basis() {
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    }

    // -- InteractionSnapshot --

    #[test]
    fn snapshot_checksum_is_deterministic() {
        let snap = InteractionSnapshot {
            seq: 1,
            screen: FsfsScreen::Search,
            tick: 42,
            focused_panel: PanelRole::Primary,
            selected_index: Some(5),
            scroll_offset: Some(0),
            visible_count: Some(100),
            query_text: Some("test query".to_string()),
            active_filters: vec!["severity: warn".to_string()],
            follow_mode: None,
            degradation_mode: DegradedRetrievalMode::Normal,
            checksum: 0,
        }
        .with_checksum();

        let snap2 = InteractionSnapshot {
            seq: 2, // Different seq shouldn't affect checksum.
            ..snap.clone()
        }
        .with_checksum();

        assert_eq!(snap.checksum, snap2.checksum);
        assert!(snap.verify_checksum());
    }

    #[test]
    fn snapshot_checksum_changes_on_state_change() {
        let snap1 = InteractionSnapshot {
            seq: 1,
            screen: FsfsScreen::Search,
            tick: 0,
            focused_panel: PanelRole::Primary,
            selected_index: Some(0),
            scroll_offset: None,
            visible_count: None,
            query_text: None,
            active_filters: vec![],
            follow_mode: None,
            degradation_mode: DegradedRetrievalMode::Normal,
            checksum: 0,
        }
        .with_checksum();

        let snap2 = InteractionSnapshot {
            selected_index: Some(1), // Changed field.
            ..snap1.clone()
        }
        .with_checksum();

        assert_ne!(snap1.checksum, snap2.checksum);
    }

    // -- InteractionBudget --

    #[test]
    fn default_budget_is_60fps() {
        let budget = InteractionBudget::default();
        assert_eq!(budget, InteractionBudget::at_60fps());
        assert_eq!(budget.total(), Duration::from_millis(16));
    }

    #[test]
    fn budget_30fps() {
        let budget = InteractionBudget::at_30fps();
        assert_eq!(budget.total(), Duration::from_millis(32));
    }

    #[test]
    fn budget_for_phase() {
        let budget = InteractionBudget::at_60fps();
        assert_eq!(
            budget.for_phase(LatencyPhase::Input),
            Duration::from_millis(1)
        );
        assert_eq!(
            budget.for_phase(LatencyPhase::Update),
            Duration::from_millis(5)
        );
        assert_eq!(
            budget.for_phase(LatencyPhase::Render),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn degraded_budgets_widen_with_severity() {
        let normal = InteractionBudget::degraded(DegradedRetrievalMode::Normal);
        let deferred = InteractionBudget::degraded(DegradedRetrievalMode::EmbedDeferred);
        let lexical = InteractionBudget::degraded(DegradedRetrievalMode::LexicalOnly);
        let paused = InteractionBudget::degraded(DegradedRetrievalMode::Paused);

        assert!(normal.total() <= deferred.total());
        assert!(deferred.total() <= lexical.total());
        assert!(lexical.total() <= paused.total());
    }

    // -- PhaseTiming --

    #[test]
    fn phase_timing_over_budget() {
        let timing = PhaseTiming {
            phase: LatencyPhase::Render,
            duration: Duration::from_millis(15),
            budget: Duration::from_millis(10),
        };
        assert!(timing.is_over_budget());
        assert_eq!(timing.overshoot(), Duration::from_millis(5));
    }

    #[test]
    fn phase_timing_within_budget() {
        let timing = PhaseTiming {
            phase: LatencyPhase::Input,
            duration: Duration::from_micros(500),
            budget: Duration::from_millis(1),
        };
        assert!(!timing.is_over_budget());
        assert_eq!(timing.overshoot(), Duration::ZERO);
    }

    // -- InteractionCycleTiming --

    #[test]
    fn cycle_timing_total_and_overruns() {
        let cycle = InteractionCycleTiming {
            input: PhaseTiming {
                phase: LatencyPhase::Input,
                duration: Duration::from_micros(800),
                budget: Duration::from_millis(1),
            },
            update: PhaseTiming {
                phase: LatencyPhase::Update,
                duration: Duration::from_millis(7),
                budget: Duration::from_millis(5),
            },
            render: PhaseTiming {
                phase: LatencyPhase::Render,
                duration: Duration::from_millis(9),
                budget: Duration::from_millis(10),
            },
            frame_seq: 1,
        };

        assert_eq!(
            cycle.total_duration(),
            Duration::from_micros(800) + Duration::from_millis(7) + Duration::from_millis(9)
        );
        assert!(cycle.has_phase_overrun());
        assert_eq!(cycle.overrun_phases(), vec![LatencyPhase::Update]);
    }

    #[test]
    fn cycle_timing_latency_bucket_classification() {
        let budget = InteractionBudget::at_60fps();
        let under = InteractionCycleTiming {
            input: PhaseTiming {
                phase: LatencyPhase::Input,
                duration: Duration::from_millis(1),
                budget: budget.input_budget,
            },
            update: PhaseTiming {
                phase: LatencyPhase::Update,
                duration: Duration::from_millis(3),
                budget: budget.update_budget,
            },
            render: PhaseTiming {
                phase: LatencyPhase::Render,
                duration: Duration::from_millis(4),
                budget: budget.render_budget,
            },
            frame_seq: 1,
        };
        assert_eq!(under.latency_bucket(&budget), LatencyBucket::UnderBudget);

        let near = InteractionCycleTiming {
            input: PhaseTiming {
                phase: LatencyPhase::Input,
                duration: Duration::from_millis(1),
                budget: budget.input_budget,
            },
            update: PhaseTiming {
                phase: LatencyPhase::Update,
                duration: Duration::from_millis(5),
                budget: budget.update_budget,
            },
            render: PhaseTiming {
                phase: LatencyPhase::Render,
                duration: Duration::from_millis(10),
                budget: budget.render_budget,
            },
            frame_seq: 2,
        };
        assert_eq!(near.latency_bucket(&budget), LatencyBucket::NearBudget);

        let over = InteractionCycleTiming {
            input: PhaseTiming {
                phase: LatencyPhase::Input,
                duration: Duration::from_millis(2),
                budget: budget.input_budget,
            },
            update: PhaseTiming {
                phase: LatencyPhase::Update,
                duration: Duration::from_millis(8),
                budget: budget.update_budget,
            },
            render: PhaseTiming {
                phase: LatencyPhase::Render,
                duration: Duration::from_millis(12),
                budget: budget.render_budget,
            },
            frame_seq: 3,
        };
        assert_eq!(over.latency_bucket(&budget), LatencyBucket::OverBudget);
    }

    // -- RenderTier --

    #[test]
    fn render_tier_from_fps() {
        assert_eq!(RenderTier::from_fps(60), RenderTier::Full);
        assert_eq!(RenderTier::from_fps(50), RenderTier::Full);
        assert_eq!(RenderTier::from_fps(49), RenderTier::Reduced);
        assert_eq!(RenderTier::from_fps(20), RenderTier::Reduced);
        assert_eq!(RenderTier::from_fps(19), RenderTier::Minimal);
        assert_eq!(RenderTier::from_fps(5), RenderTier::Minimal);
        assert_eq!(RenderTier::from_fps(4), RenderTier::Safety);
        assert_eq!(RenderTier::from_fps(0), RenderTier::Safety);
    }

    #[test]
    fn render_tier_feature_gates() {
        assert!(RenderTier::Full.animations_enabled());
        assert!(!RenderTier::Reduced.animations_enabled());

        assert!(RenderTier::Full.charts_enabled());
        assert!(RenderTier::Reduced.charts_enabled());
        assert!(!RenderTier::Minimal.charts_enabled());
        assert!(!RenderTier::Safety.charts_enabled());
    }

    #[test]
    fn render_tier_display() {
        assert_eq!(RenderTier::Full.to_string(), "full");
        assert_eq!(RenderTier::Safety.to_string(), "safety");
    }

    // -- VirtualizedListState --

    #[test]
    fn virtualized_list_navigation() {
        let mut list = VirtualizedListState {
            total_items: 100,
            selected: 0,
            scroll_offset: 0,
            viewport_height: 10,
        };

        list.select_next();
        assert_eq!(list.selected, 1);

        list.select_last();
        assert_eq!(list.selected, 99);
        assert!(list.scroll_offset > 0);

        list.select_first();
        assert_eq!(list.selected, 0);
        assert_eq!(list.scroll_offset, 0);
    }

    #[test]
    fn virtualized_list_page_navigation() {
        let mut list = VirtualizedListState {
            total_items: 50,
            selected: 0,
            scroll_offset: 0,
            viewport_height: 10,
        };

        list.page_down();
        assert_eq!(list.selected, 10);

        list.page_down();
        assert_eq!(list.selected, 20);

        list.page_up();
        assert_eq!(list.selected, 10);
    }

    #[test]
    fn virtualized_list_clamps_on_resize() {
        let mut list = VirtualizedListState {
            total_items: 10,
            selected: 9,
            scroll_offset: 5,
            viewport_height: 5,
        };

        list.set_total_items(5);
        assert_eq!(list.selected, 4);

        list.set_total_items(0);
        assert_eq!(list.selected, 0);
        assert_eq!(list.scroll_offset, 0);
    }

    #[test]
    fn virtualized_list_ensure_visible_scrolls_down() {
        let mut list = VirtualizedListState {
            total_items: 100,
            selected: 15,
            scroll_offset: 0,
            viewport_height: 10,
        };
        list.ensure_visible();
        assert_eq!(list.scroll_offset, 6); // 15 + 1 - 10 = 6
    }

    #[test]
    fn virtualized_list_ensure_visible_scrolls_up() {
        let mut list = VirtualizedListState {
            total_items: 100,
            selected: 2,
            scroll_offset: 10,
            viewport_height: 10,
        };
        list.ensure_visible();
        assert_eq!(list.scroll_offset, 2);
    }

    #[test]
    fn virtualized_list_empty_is_safe() {
        let mut list = VirtualizedListState::empty();
        list.select_next(); // No-op.
        list.select_prev(); // No-op.
        list.page_down(); // No-op.
        list.page_up(); // No-op.
        assert_eq!(list.selected, 0);
    }

    // -- SearchInteractionState --

    #[test]
    fn search_interaction_submit_and_repeat_last_query() {
        let mut state = SearchInteractionState::new(4);
        state.apply_incremental_query("  fn parse_query  ");

        let first = state.apply_action(&ScreenAction::SubmitQuery);
        assert_eq!(
            first,
            Some(SearchInteractionEvent::QuerySubmitted(
                "fn parse_query".to_owned(),
            ))
        );
        assert_eq!(
            state.last_submitted_query.as_deref(),
            Some("fn parse_query")
        );

        state.query_input.clear();
        let second = state.apply_action(&ScreenAction::RepeatLastQuery);
        assert_eq!(
            second,
            Some(SearchInteractionEvent::QuerySubmitted(
                "fn parse_query".to_owned(),
            ))
        );
    }

    #[test]
    fn search_interaction_open_and_jump_use_selected_row() {
        let mut state = SearchInteractionState::new(3);
        state.set_results(vec![
            SearchResultEntry::new("doc-1", "src/a.rs", "alpha"),
            SearchResultEntry::new("doc-2", "src/b.rs", "beta"),
        ]);
        state.apply_action(&ScreenAction::SelectDown);

        let open = state.apply_action(&ScreenAction::OpenSelectedResult);
        assert_eq!(
            open,
            Some(SearchInteractionEvent::OpenSelected {
                doc_id: "doc-2".to_owned(),
                source_path: "src/b.rs".to_owned(),
            })
        );

        let jump = state.apply_action(&ScreenAction::JumpToSelectedSource);
        assert_eq!(
            jump,
            Some(SearchInteractionEvent::JumpToSource {
                doc_id: "doc-2".to_owned(),
                source_path: "src/b.rs".to_owned(),
            })
        );
    }

    #[test]
    fn search_interaction_detail_toggle_and_clear_query_are_deterministic() {
        let mut state = SearchInteractionState::new(2);
        state.apply_incremental_query("query");
        assert!(state.pending_incremental_query.is_some());
        assert!(!state.detail_panel_visible);

        state.apply_action(&ScreenAction::ToggleDetailPanel);
        assert!(state.detail_panel_visible);
        state.apply_action(&ScreenAction::CollapseSelected);
        assert!(!state.detail_panel_visible);
        state.apply_action(&ScreenAction::ExpandSelected);
        assert!(state.detail_panel_visible);

        state.apply_action(&ScreenAction::ClearQuery);
        assert!(state.query_input.is_empty());
        assert!(state.pending_incremental_query.is_none());
    }

    #[test]
    fn search_interaction_visible_window_tracks_virtualized_selection() {
        let mut state = SearchInteractionState::new(2);
        state.set_results(vec![
            SearchResultEntry::new("doc-1", "src/one.rs", "one"),
            SearchResultEntry::new("doc-2", "src/two.rs", "two"),
            SearchResultEntry::new("doc-3", "src/three.rs", "three"),
            SearchResultEntry::new("doc-4", "src/four.rs", "four"),
        ]);

        state.apply_action(&ScreenAction::PageDown);
        assert_eq!(state.list.selected, 2);
        assert_eq!(state.visible_window(), (1, 3));
        assert_eq!(state.visible_results()[0].doc_id, "doc-2");
        assert_eq!(
            state.selected_result().map(|row| row.doc_id.as_str()),
            Some("doc-3")
        );
    }

    #[test]
    fn search_interaction_telemetry_contains_required_fields() {
        let mut state = SearchInteractionState::new(2);
        state.apply_incremental_query("auth middleware");
        state.set_results(vec![
            SearchResultEntry::new("doc-1", "src/a.rs", "a"),
            SearchResultEntry::new("doc-2", "src/b.rs", "b"),
            SearchResultEntry::new("doc-3", "src/c.rs", "c"),
        ]);
        state.apply_action(&ScreenAction::SelectDown);

        let budget = InteractionBudget::at_60fps();
        let cycle = InteractionCycleTiming {
            input: PhaseTiming {
                phase: LatencyPhase::Input,
                duration: Duration::from_millis(1),
                budget: budget.input_budget,
            },
            update: PhaseTiming {
                phase: LatencyPhase::Update,
                duration: Duration::from_millis(4),
                budget: budget.update_budget,
            },
            render: PhaseTiming {
                phase: LatencyPhase::Render,
                duration: Duration::from_millis(7),
                budget: budget.render_budget,
            },
            frame_seq: 77,
        };

        let telemetry = state.telemetry_sample(&cycle, &budget);
        assert_eq!(telemetry.frame_seq, 77);
        assert_eq!(telemetry.frame_budget_ms, 16);
        assert_eq!(telemetry.total_latency_ms, 12);
        assert_eq!(telemetry.latency_bucket, LatencyBucket::UnderBudget);
        assert_eq!(telemetry.visible_window, (0, 2));
        assert_eq!(telemetry.visible_count, 2);
        assert_eq!(telemetry.selected_index, 1);
        assert_eq!(telemetry.query_len, "auth middleware".len());
    }

    #[test]
    fn search_interaction_telemetry_interaction_id_is_deterministic() {
        let mut state = SearchInteractionState::new(2);
        state.apply_incremental_query("query");
        state.set_results(vec![
            SearchResultEntry::new("doc-1", "src/a.rs", "a"),
            SearchResultEntry::new("doc-2", "src/b.rs", "b"),
        ]);

        let budget = InteractionBudget::at_60fps();
        let cycle = InteractionCycleTiming {
            input: PhaseTiming {
                phase: LatencyPhase::Input,
                duration: Duration::from_millis(1),
                budget: budget.input_budget,
            },
            update: PhaseTiming {
                phase: LatencyPhase::Update,
                duration: Duration::from_millis(2),
                budget: budget.update_budget,
            },
            render: PhaseTiming {
                phase: LatencyPhase::Render,
                duration: Duration::from_millis(3),
                budget: budget.render_budget,
            },
            frame_seq: 11,
        };

        let first = state.telemetry_sample(&cycle, &budget).interaction_id;
        let second = state.telemetry_sample(&cycle, &budget).interaction_id;
        assert_eq!(first, second);

        state.apply_action(&ScreenAction::SelectDown);
        let third = state.telemetry_sample(&cycle, &budget).interaction_id;
        assert_ne!(first, third);
    }

    // -- CyclicFilter --

    #[test]
    fn cyclic_filter_cycles_through_values() {
        let mut filter = CyclicFilter::new(
            "Severity",
            vec!["Info".to_string(), "Warn".to_string(), "Error".to_string()],
        );
        assert!(filter.active_value().is_none());

        filter.cycle_next();
        assert_eq!(filter.active_value(), Some("Info"));

        filter.cycle_next();
        assert_eq!(filter.active_value(), Some("Warn"));

        filter.cycle_next();
        assert_eq!(filter.active_value(), Some("Error"));

        filter.cycle_next();
        assert!(filter.active_value().is_none()); // Wraps to None.
    }

    #[test]
    fn cyclic_filter_clear() {
        let mut filter = CyclicFilter::new("Kind", vec!["A".to_string(), "B".to_string()]);
        filter.cycle_next();
        assert!(filter.active_value().is_some());

        filter.clear();
        assert!(filter.active_value().is_none());
    }

    #[test]
    fn cyclic_filter_display() {
        let mut filter = CyclicFilter::new("Type", vec!["X".to_string()]);
        assert_eq!(filter.display(), "Type: all");

        filter.cycle_next();
        assert_eq!(filter.display(), "Type: X");
    }

    #[test]
    fn cyclic_filter_empty_values() {
        let mut filter = CyclicFilter::new("Empty", vec![]);
        filter.cycle_next();
        assert!(filter.active_value().is_none()); // Stays None.
    }

    #[test]
    fn cyclic_filter_out_of_range_selection_is_safe() {
        let mut filter = CyclicFilter::new("Kind", vec!["A".to_string(), "B".to_string()]);
        filter.selected = Some(usize::MAX);
        assert!(filter.active_value().is_none());
        assert_eq!(filter.display(), "Kind: all");

        filter.cycle_next();
        assert!(filter.active_value().is_none());
    }

    // -- IndexingCockpitSnapshot --

    #[test]
    fn backlog_visualization_computes_trend_and_utilization() {
        let backlog = BacklogVisualization::new(80, 50, 100, 12, 8.0);
        assert_eq!(backlog.trend, TrendDirection::Rising);
        assert!(backlog.is_hot());
        assert!((backlog.utilization_ratio() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn throughput_visualization_detects_stall() {
        let throughput = ThroughputVisualization::from_samples(vec![
            ThroughputSample::new(1, 0.2, 0.3),
            ThroughputSample::new(2, 0.1, 0.0),
        ]);
        assert!(throughput.is_stalled());
    }

    #[test]
    fn cockpit_controls_enable_throttle_under_pressure() {
        let backlog = BacklogVisualization::new(120, 64, 256, 0, 5.0);
        let throughput =
            ThroughputVisualization::from_samples(vec![ThroughputSample::new(1, 22.0, 18.0)]);
        let pressure = ResourcePressureIndicator {
            pressure_state: PressureState::Constrained,
            degradation_stage: DegradationStage::EmbedDeferred,
            backpressure_mode: BackpressureMode::HighWatermark,
            throttle: WatcherThrottle {
                debounce_ms: 2_000,
                batch_size: 25,
                suspended: false,
                reason_code: "watcher.throttle.constrained",
            },
            user_banner: "Constrained mode active",
            transition_reason_code: "degrade.transition.embed_deferred",
            override_mode: PressureDegradationOverride::Auto,
            override_allowed: true,
            reason_code: "pressure.transition.constrained",
        };

        let snapshot = IndexingCockpitSnapshot::new(42, backlog, throughput, pressure);
        let enabled = snapshot.enabled_actions();
        assert!(enabled.contains(&ScreenAction::PauseIndexing));
        assert!(enabled.contains(&ScreenAction::ThrottleIndexing));
        assert!(!enabled.contains(&ScreenAction::RecoverIndexing));
    }

    #[test]
    fn cockpit_controls_enable_recover_when_pressure_is_healthy() {
        let backlog = BacklogVisualization::new(20, 64, 256, 0, -5.0);
        let throughput =
            ThroughputVisualization::from_samples(vec![ThroughputSample::new(1, 120.0, 95.0)]);
        let pressure = ResourcePressureIndicator {
            pressure_state: PressureState::Normal,
            degradation_stage: DegradationStage::LexicalOnly,
            backpressure_mode: BackpressureMode::Normal,
            throttle: WatcherThrottle {
                debounce_ms: 500,
                batch_size: 64,
                suspended: false,
                reason_code: "watcher.throttle.normal",
            },
            user_banner: "Recovered from degradation",
            transition_reason_code: "degrade.transition.recovered",
            override_mode: PressureDegradationOverride::Auto,
            override_allowed: true,
            reason_code: "degrade.transition.recovered",
        };

        let snapshot = IndexingCockpitSnapshot::new(99, backlog, throughput, pressure);
        let enabled = snapshot.enabled_actions();
        assert!(enabled.contains(&ScreenAction::RecoverIndexing));
        assert!(enabled.contains(&ScreenAction::PauseIndexing));
    }

    #[test]
    fn cockpit_controls_enable_recover_when_watcher_is_suspended() {
        let backlog = BacklogVisualization::new(12, 64, 256, 0, -1.0);
        let throughput =
            ThroughputVisualization::from_samples(vec![ThroughputSample::new(1, 90.0, 72.0)]);
        let pressure = ResourcePressureIndicator {
            pressure_state: PressureState::Normal,
            degradation_stage: DegradationStage::Full,
            backpressure_mode: BackpressureMode::Normal,
            throttle: WatcherThrottle {
                debounce_ms: 2_000,
                batch_size: 16,
                suspended: true,
                reason_code: "watcher.suspended",
            },
            user_banner: "Normal operation",
            transition_reason_code: "pressure.transition.normalized",
            override_mode: PressureDegradationOverride::Auto,
            override_allowed: true,
            reason_code: "pressure.transition.normalized",
        };

        let snapshot = IndexingCockpitSnapshot::new(100, backlog, throughput, pressure);
        let enabled = snapshot.enabled_actions();
        assert!(enabled.contains(&ScreenAction::RecoverIndexing));
    }

    #[test]
    fn cockpit_controls_disable_recover_when_backpressure_not_normal() {
        let backlog = BacklogVisualization::new(22, 64, 256, 0, -2.0);
        let throughput =
            ThroughputVisualization::from_samples(vec![ThroughputSample::new(1, 110.0, 90.0)]);
        let pressure = ResourcePressureIndicator {
            pressure_state: PressureState::Normal,
            degradation_stage: DegradationStage::LexicalOnly,
            backpressure_mode: BackpressureMode::HighWatermark,
            throttle: WatcherThrottle {
                debounce_ms: 500,
                batch_size: 64,
                suspended: false,
                reason_code: "watcher.throttle.high-watermark",
            },
            user_banner: "Recovering from high watermark",
            transition_reason_code: "pressure.transition.normalizing",
            override_mode: PressureDegradationOverride::Auto,
            override_allowed: true,
            reason_code: "pressure.transition.normalizing",
        };

        let snapshot = IndexingCockpitSnapshot::new(101, backlog, throughput, pressure);
        let enabled = snapshot.enabled_actions();
        assert!(!enabled.contains(&ScreenAction::RecoverIndexing));
    }

    #[test]
    fn cockpit_override_controls_are_guarded_and_auditable() {
        let backlog = BacklogVisualization::new(80, 64, 256, 0, 4.0);
        let throughput =
            ThroughputVisualization::from_samples(vec![ThroughputSample::new(1, 20.0, 14.0)]);
        let pressure = ResourcePressureIndicator {
            pressure_state: PressureState::Degraded,
            degradation_stage: DegradationStage::LexicalOnly,
            backpressure_mode: BackpressureMode::HighWatermark,
            throttle: WatcherThrottle {
                debounce_ms: 1_500,
                batch_size: 24,
                suspended: false,
                reason_code: "watcher.throttle.degraded",
            },
            user_banner: "Degraded mode active",
            transition_reason_code: "degrade.transition.override",
            override_mode: PressureDegradationOverride::ForceLexicalOnly,
            override_allowed: true,
            reason_code: "pressure.transition.degraded",
        };

        let snapshot = IndexingCockpitSnapshot::new(102, backlog, throughput, pressure.clone());
        let enabled = snapshot.enabled_actions();
        assert!(enabled.contains(&ScreenAction::SetOverrideAuto));
        assert!(enabled.contains(&ScreenAction::ForceOverrideMetadataOnly));
        assert!(enabled.contains(&ScreenAction::ForceOverridePaused));
        assert!(!enabled.contains(&ScreenAction::ForceOverrideLexicalOnly));
        assert!(
            snapshot
                .controls
                .iter()
                .any(|control| control.reason_code == "cockpit.control.override.force_paused")
        );

        let banner = pressure.banner();
        assert_eq!(banner.text, "Degraded mode active");
        assert_eq!(banner.transition_reason_code, "degrade.transition.override");
        assert_eq!(
            banner.override_mode,
            PressureDegradationOverride::ForceLexicalOnly
        );
        assert!(banner.override_allowed);
    }

    #[test]
    fn cockpit_override_controls_stay_disabled_without_override_permission() {
        let backlog = BacklogVisualization::new(40, 64, 256, 0, 1.0);
        let throughput =
            ThroughputVisualization::from_samples(vec![ThroughputSample::new(1, 60.0, 48.0)]);
        let pressure = ResourcePressureIndicator {
            pressure_state: PressureState::Constrained,
            degradation_stage: DegradationStage::EmbedDeferred,
            backpressure_mode: BackpressureMode::Normal,
            throttle: WatcherThrottle {
                debounce_ms: 800,
                batch_size: 32,
                suspended: false,
                reason_code: "watcher.throttle.constrained",
            },
            user_banner: "Constrained mode active",
            transition_reason_code: "degrade.transition.embed_deferred",
            override_mode: PressureDegradationOverride::Auto,
            override_allowed: false,
            reason_code: "pressure.transition.constrained",
        };

        let snapshot = IndexingCockpitSnapshot::new(103, backlog, throughput, pressure);
        let enabled = snapshot.enabled_actions();
        assert!(!enabled.contains(&ScreenAction::SetOverrideAuto));
        assert!(!enabled.contains(&ScreenAction::ForceOverrideFull));
        assert!(!enabled.contains(&ScreenAction::ForceOverrideEmbedDeferred));
        assert!(!enabled.contains(&ScreenAction::ForceOverrideLexicalOnly));
        assert!(!enabled.contains(&ScreenAction::ForceOverrideMetadataOnly));
        assert!(!enabled.contains(&ScreenAction::ForceOverridePaused));
    }

    // -- LatencyPhase --

    #[test]
    fn latency_phase_display() {
        assert_eq!(LatencyPhase::Input.to_string(), "input");
        assert_eq!(LatencyPhase::Update.to_string(), "update");
        assert_eq!(LatencyPhase::Render.to_string(), "render");
    }

    // -- Layout direction --

    #[test]
    fn layout_directions_distinguishable() {
        assert_ne!(LayoutDirection::Vertical, LayoutDirection::Horizontal);
    }
}
