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

    // -- Timeline-specific --
    /// Toggle auto-follow mode (scroll to newest events).
    ToggleFollow,

    // -- Indexing --
    /// Pause background indexing.
    PauseIndexing,
    /// Resume background indexing.
    ResumeIndexing,

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
            "search.repeat_last" => Some(Self::RepeatLastQuery),
            "index.pause" => Some(Self::PauseIndexing),
            "index.resume" => Some(Self::ResumeIndexing),
            "explain.toggle_panel" => Some(Self::ToggleDetailPanel),
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
    pub fn focus_next(&mut self) {
        self.current = (self.current + 1) % self.cycle.len();
    }

    /// Move focus to the previous panel in tab order (wraps).
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
    pub fn len(&self) -> usize {
        self.cycle.len()
    }

    /// Whether there are no focusable panels.
    #[must_use]
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
    /// The checksum covers: screen ID, tick, focused_panel, selected_index,
    /// scroll_offset, visible_count, query_text, filters, follow_mode,
    /// and degradation_mode. It does NOT include seq or checksum itself.
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
/// Consumers build this by measuring each phase and then call
/// [`InteractionCycleTiming::evaluate`] to check budget compliance.
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
    pub fn has_phase_overrun(&self) -> bool {
        self.input.is_over_budget()
            || self.update.is_over_budget()
            || self.render.is_over_budget()
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

    /// Ensure the selected item is visible by adjusting scroll_offset.
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
    pub fn cycle_next(&mut self) {
        self.selected = match self.selected {
            None if self.values.is_empty() => None,
            None => Some(0),
            Some(idx) if idx + 1 >= self.values.len() => None,
            Some(idx) => Some(idx + 1),
        };
    }

    /// Active filter value, or `None` for "show all".
    #[must_use]
    pub fn active_value(&self) -> Option<&str> {
        self.selected.map(|idx| self.values[idx].as_str())
    }

    /// Clear the filter (back to "show all").
    pub fn clear(&mut self) {
        self.selected = None;
    }

    /// Display string for the current state.
    #[must_use]
    pub fn display(&self) -> String {
        match self.active_value() {
            Some(val) => format!("{}: {val}", self.label),
            None => format!("{}: all", self.label),
        }
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
            ScreenAction::from_palette_action_id("unknown.action"),
            None
        );
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
        assert_eq!(budget.for_phase(LatencyPhase::Input), Duration::from_millis(1));
        assert_eq!(budget.for_phase(LatencyPhase::Update), Duration::from_millis(5));
        assert_eq!(budget.for_phase(LatencyPhase::Render), Duration::from_millis(10));
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
