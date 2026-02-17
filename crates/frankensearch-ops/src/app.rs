//! Ops TUI application entry point.
//!
//! [`OpsApp`] assembles the app shell, registers screens, wires up the
//! data source, and drives the event loop. Product binaries create an
//! `OpsApp` and call `run()`.

use frankensearch_tui::Screen;
use frankensearch_tui::overlay::{OverlayKind, OverlayRequest};
use frankensearch_tui::palette::{Action, ActionCategory};
use frankensearch_tui::screen::ScreenId;
use frankensearch_tui::shell::{AppShell, ShellConfig};
use frankensearch_tui::theme::Theme;
use ratatui::Frame;

use crate::data_source::DataSource;
use crate::preferences::DisplayPreferences;
use crate::presets::{ViewPreset, ViewState};
use crate::screens::{
    ActionTimelineScreen, AlertsSloScreen, FleetOverviewScreen, HistoricalAnalyticsScreen,
    IndexResourceScreen, LiveSearchStreamScreen, ProjectDetailScreen,
};
use crate::state::{AppState, ControlPlaneHealth};

// ─── Ops App ─────────────────────────────────────────────────────────────────

/// Ops control-plane TUI application.
///
/// Wires the shared TUI framework with ops-specific screens and data sources.
pub struct OpsApp {
    /// The app shell (owns registry, keymap, overlays).
    pub shell: AppShell,
    /// Shared application state.
    pub state: AppState,
    /// Data source for populating state.
    data_source: Box<dyn DataSource>,
    /// Registered fleet screen id.
    fleet_screen_id: ScreenId,
    /// Registered project detail screen id.
    project_screen_id: ScreenId,
    /// Registered live stream screen id.
    live_stream_screen_id: ScreenId,
    /// Registered timeline screen id.
    timeline_screen_id: ScreenId,
    /// Registered index/resource monitoring screen id.
    index_screen_id: ScreenId,
    /// Registered alerts/SLO/capacity screen id.
    alerts_screen_id: ScreenId,
    /// Registered historical analytics/explainability screen id.
    analytics_screen_id: ScreenId,
    /// Accessibility and display preferences.
    pub preferences: DisplayPreferences,
    /// Current view state (preset + density + filters).
    pub view: ViewState,
    /// Most severe control-plane state we already surfaced as an alert.
    last_alerted_health: Option<ControlPlaneHealth>,
}

impl OpsApp {
    /// Create a new ops app with the given data source.
    #[must_use]
    pub fn new(data_source: Box<dyn DataSource>) -> Self {
        let config = ShellConfig {
            title: "frankensearch ops".to_string(),
            theme: Theme::dark(),
            show_status_bar: true,
            show_breadcrumbs: true,
        };

        let mut shell = AppShell::new(config);

        // Register palette actions.
        for action in Self::palette_actions() {
            shell.palette.register(action);
        }

        // Create and register screens.
        let mut project_screen = ProjectDetailScreen::new();
        let project_id = project_screen.id().clone();

        let mut fleet_screen = FleetOverviewScreen::new();
        fleet_screen.set_project_screen_id(project_id.clone());
        let fleet_id = fleet_screen.id().clone();

        let live_stream_screen = LiveSearchStreamScreen::new();
        let live_stream_id = live_stream_screen.id().clone();

        let timeline_screen = ActionTimelineScreen::new();
        let timeline_id = timeline_screen.id().clone();

        let index_screen = IndexResourceScreen::new();
        let index_id = index_screen.id().clone();

        let alerts_screen = AlertsSloScreen::new();
        let alerts_id = alerts_screen.id().clone();

        let analytics_screen = HistoricalAnalyticsScreen::new();
        let analytics_id = analytics_screen.id().clone();

        fleet_screen.set_live_stream_screen_id(live_stream_id.clone());
        fleet_screen.set_timeline_screen_id(timeline_id.clone());
        fleet_screen.set_analytics_screen_id(analytics_id.clone());

        project_screen.set_fleet_screen_id(fleet_id.clone());
        project_screen.set_live_stream_screen_id(live_stream_id.clone());
        project_screen.set_timeline_screen_id(timeline_id.clone());
        project_screen.set_analytics_screen_id(analytics_id.clone());

        shell.registry.register(Box::new(fleet_screen));
        shell.registry.register(Box::new(project_screen));
        shell.registry.register(Box::new(live_stream_screen));
        shell.registry.register(Box::new(timeline_screen));
        shell.registry.register(Box::new(index_screen));
        shell.registry.register(Box::new(alerts_screen));
        shell.registry.register(Box::new(analytics_screen));

        // Set initial active screen.
        shell.navigate_to(&fleet_id);

        Self {
            shell,
            state: AppState::new(),
            data_source,
            fleet_screen_id: fleet_id,
            project_screen_id: project_id,
            live_stream_screen_id: live_stream_id,
            timeline_screen_id: timeline_id,
            index_screen_id: index_id,
            alerts_screen_id: alerts_id,
            analytics_screen_id: analytics_id,
            preferences: DisplayPreferences::new(),
            view: ViewState::default(),
            last_alerted_health: None,
        }
    }

    /// Refresh state from the data source.
    pub fn refresh_data(&mut self) {
        let previous_health = self.state.control_plane_health();
        let snapshot = self.data_source.fleet_snapshot();
        let control_plane = self.data_source.control_plane_metrics();
        self.state.update_fleet(snapshot);
        self.state.update_control_plane(control_plane);

        // Update status line with connection info.
        self.shell.status_line = self
            .shell
            .status_line
            .clone()
            .with_center(self.state.connection_status().to_string())
            .with_right(self.state.control_plane_health().badge());

        let current_health = self.state.control_plane_health();
        self.maybe_emit_control_plane_alert(previous_health, current_health);
        self.sync_screen_states();
    }

    /// Process an input event. Returns `true` if the app should quit.
    pub fn handle_input(&mut self, event: &frankensearch_tui::InputEvent) -> bool {
        let previous_screen = self.shell.active_screen.clone();
        let quit = self.shell.handle_input(event);
        self.sync_project_filter_from_screen_transition(previous_screen.as_ref(), event);

        // Handle palette action confirmations.
        if let Some(action_id) = self.shell.last_palette_action().map(str::to_string) {
            self.dispatch_palette_action(&action_id);
        }

        quit
    }

    /// Render the shell plus ops-specific overlays.
    ///
    /// Consumers should render via this method (instead of calling
    /// `self.shell.render(...)` directly) so palette/help/alert overlays
    /// are painted on top of the active screen.
    pub fn render(&mut self, frame: &mut Frame<'_>) {
        self.shell.render(frame);
        let area = frame.area();
        if let Some(request) = self.shell.overlays.top() {
            crate::overlays::render_overlay(frame, area, request);
        }
        crate::overlays::render_palette_overlay(frame, area, &self.shell.palette);
    }

    /// Dispatch a confirmed palette action by ID.
    #[allow(clippy::too_many_lines)]
    fn dispatch_palette_action(&mut self, action_id: &str) {
        match action_id {
            "nav.fleet" => {
                if self.view.preset == ViewPreset::ProjectDeepDive {
                    self.view.apply_preset(ViewPreset::FleetTriage);
                } else {
                    self.view.clear_project_filter();
                }
                let id = self.fleet_screen_id.clone();
                self.shell.navigate_to(&id);
                self.sync_screen_states();
            }
            "nav.project" | "view.project_deep_dive" => {
                self.open_project_detail_for_selected_project();
            }
            "nav.search" => {
                let id = self.live_stream_screen_id.clone();
                self.shell.navigate_to(&id);
            }
            "nav.index" => {
                let id = self.index_screen_id.clone();
                self.shell.navigate_to(&id);
            }
            "nav.alerts" => {
                let id = self.alerts_screen_id.clone();
                self.shell.navigate_to(&id);
            }
            "nav.analytics" => {
                let id = self.analytics_screen_id.clone();
                self.shell.navigate_to(&id);
            }
            "analytics.export_snapshot" => {
                let id = self.analytics_screen_id.clone();
                self.shell.navigate_to(&id);

                let toggle_export_mode = frankensearch_tui::InputEvent::Key(
                    crossterm::event::KeyCode::Char('e'),
                    crossterm::event::KeyModifiers::NONE,
                );
                let _ = self.shell.handle_input(&toggle_export_mode);

                let selected_project = self
                    .selected_project_from_analytics()
                    .unwrap_or_else(|| "none selected".to_owned());
                let body = format!(
                    "Snapshot mode toggled for Historical Analytics.\nSelected project: {selected_project}\nUse `g` to open project detail and copy replay handles from the evidence table."
                );
                self.shell.overlays.push(
                    OverlayRequest::new(OverlayKind::Alert, "Analytics Snapshot Export")
                        .with_body(body),
                );
            }
            "debug.refresh" => {
                self.refresh_data();
            }
            "debug.self_check" => {
                let metrics = self.state.control_plane_metrics();
                tracing::info!(
                    target: "frankensearch.ops",
                    health = %self.state.control_plane_health(),
                    ingestion_lag_events = metrics.ingestion_lag_events,
                    dead_letter_events = metrics.dead_letter_events,
                    discovery_latency_ms = metrics.discovery_latency_ms,
                    "control-plane self-check requested"
                );
                self.shell.overlays.push(
                    OverlayRequest::new(OverlayKind::Alert, "Control Plane Self-Check")
                        .with_body(self.state.self_check_report()),
                );
            }
            "settings.theme" => {
                self.shell.config.theme = Theme::from_preset(self.shell.config.theme.preset.next());
            }
            "settings.contrast" => {
                self.preferences.toggle_contrast();
            }
            "settings.motion" => {
                self.preferences.toggle_motion();
            }
            "settings.focus" => {
                self.preferences.toggle_focus_visibility();
            }
            "settings.hints" => {
                self.preferences.toggle_shortcut_hints();
            }
            "view.density" => {
                self.view.cycle_density();
                self.sync_screen_states();
            }
            "view.fleet_triage" => {
                self.view.apply_preset(ViewPreset::FleetTriage);
                self.view.clear_project_filter();
                let id = self.fleet_screen_id.clone();
                self.shell.navigate_to(&id);
                self.sync_screen_states();
            }
            "view.incident_mode" => {
                self.view.apply_preset(ViewPreset::IncidentMode);
                if self.view.preset.prefer_high_contrast() {
                    self.preferences.contrast = crate::preferences::ContrastMode::High;
                }
                self.sync_screen_states();
            }
            "view.low_noise" => {
                self.view.apply_preset(ViewPreset::LowNoise);
                self.sync_screen_states();
            }
            "view.hide_healthy" => {
                self.view.toggle_hide_healthy();
                self.sync_screen_states();
            }
            "view.unhealthy_first" => {
                self.view.toggle_unhealthy_first();
                self.sync_screen_states();
            }
            _ => {
                tracing::warn!(
                    target: "frankensearch.ops",
                    action_id,
                    "unhandled palette action"
                );
            }
        }
    }

    /// Whether the app should quit.
    #[must_use]
    pub const fn should_quit(&self) -> bool {
        self.shell.should_quit
    }

    /// Get a reference to the fleet screen (for testing/inspection).
    #[must_use]
    pub fn fleet_screen(&self) -> Option<&FleetOverviewScreen> {
        self.shell
            .registry
            .get(&self.fleet_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<FleetOverviewScreen>())
    }

    /// Get a reference to the project detail screen (for testing/inspection).
    #[must_use]
    pub fn project_screen(&self) -> Option<&ProjectDetailScreen> {
        self.shell
            .registry
            .get(&self.project_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<ProjectDetailScreen>())
    }

    /// Get a reference to the index/resource screen (for testing/inspection).
    #[must_use]
    pub fn index_screen(&self) -> Option<&IndexResourceScreen> {
        self.shell
            .registry
            .get(&self.index_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<IndexResourceScreen>())
    }

    /// Get a reference to the alerts/SLO screen (for testing/inspection).
    #[must_use]
    pub fn alerts_screen(&self) -> Option<&AlertsSloScreen> {
        self.shell
            .registry
            .get(&self.alerts_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<AlertsSloScreen>())
    }

    /// Get a reference to the historical analytics screen (for testing/inspection).
    #[must_use]
    pub fn analytics_screen(&self) -> Option<&HistoricalAnalyticsScreen> {
        self.shell
            .registry
            .get(&self.analytics_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<HistoricalAnalyticsScreen>())
    }

    /// Get a reference to the live stream screen (for testing/inspection).
    #[must_use]
    pub fn live_stream_screen(&self) -> Option<&LiveSearchStreamScreen> {
        self.shell
            .registry
            .get(&self.live_stream_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<LiveSearchStreamScreen>())
    }

    /// Get a list of all registered palette actions (for help screen).
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn palette_actions() -> Vec<Action> {
        vec![
            Action::new(
                "nav.fleet",
                "Go to Fleet Overview",
                ActionCategory::Navigation,
            )
            .with_shortcut("1"),
            Action::new(
                "nav.project",
                "Go to Project Detail",
                ActionCategory::Navigation,
            )
            .with_shortcut("4"),
            Action::new(
                "nav.search",
                "Go to Search Stream",
                ActionCategory::Navigation,
            )
            .with_shortcut("2"),
            Action::new(
                "nav.index",
                "Go to Index Status",
                ActionCategory::Navigation,
            )
            .with_shortcut("3"),
            Action::new("nav.alerts", "Go to Alerts/SLO", ActionCategory::Navigation)
                .with_shortcut("5"),
            Action::new(
                "nav.analytics",
                "Go to Historical Analytics",
                ActionCategory::Navigation,
            )
            .with_shortcut("6"),
            Action::new(
                "analytics.export_snapshot",
                "Export Analytics Snapshot",
                ActionCategory::Custom("Analytics".to_string()),
            )
            .with_description(
                "Open analytics, toggle snapshot mode, and show replay/export guidance",
            ),
            Action::new("debug.refresh", "Force Refresh Data", ActionCategory::Debug)
                .with_shortcut("F5"),
            Action::new(
                "debug.self_check",
                "Run Control-Plane Self-Check",
                ActionCategory::Debug,
            )
            .with_description("Show internal health metrics and degradation status"),
            Action::new("settings.theme", "Cycle Theme", ActionCategory::Settings)
                .with_shortcut("Ctrl+T")
                .with_description("Cycle through all 6 theme presets"),
            Action::new(
                "settings.contrast",
                "Toggle High Contrast",
                ActionCategory::Settings,
            )
            .with_description("Switch between normal and high contrast mode"),
            Action::new(
                "settings.motion",
                "Toggle Reduced Motion",
                ActionCategory::Settings,
            )
            .with_description("Enable or disable non-essential animations"),
            Action::new(
                "settings.focus",
                "Toggle Focus Visibility",
                ActionCategory::Settings,
            )
            .with_description("Switch between normal and enhanced focus indicators"),
            Action::new(
                "settings.hints",
                "Toggle Shortcut Hints",
                ActionCategory::Settings,
            )
            .with_description("Show or hide inline keyboard shortcut hints"),
            // ── View Presets ────────────────────────────────────────
            Action::new(
                "view.density",
                "Cycle Density Mode",
                ActionCategory::Custom("View".to_string()),
            )
            .with_shortcut("D")
            .with_description("Cycle between compact, normal, and expanded"),
            Action::new(
                "view.fleet_triage",
                "Fleet Triage View",
                ActionCategory::Custom("View".to_string()),
            )
            .with_description("Compact fleet overview, health-first sorting"),
            Action::new(
                "view.project_deep_dive",
                "Project Deep Dive View",
                ActionCategory::Custom("View".to_string()),
            )
            .with_description("Expanded single-project view with full metrics"),
            Action::new(
                "view.incident_mode",
                "Incident Mode View",
                ActionCategory::Custom("View".to_string()),
            )
            .with_description("High-contrast, alerts prominent, unhealthy first"),
            Action::new(
                "view.low_noise",
                "Low Noise View",
                ActionCategory::Custom("View".to_string()),
            )
            .with_description("Hide healthy instances, show only actionable items"),
            Action::new(
                "view.hide_healthy",
                "Toggle Hide Healthy",
                ActionCategory::Custom("View".to_string()),
            )
            .with_description("Show or hide healthy instances"),
            Action::new(
                "view.unhealthy_first",
                "Toggle Unhealthy First",
                ActionCategory::Custom("View".to_string()),
            )
            .with_description("Sort unhealthy instances to the top"),
        ]
    }

    /// Get the current display preferences.
    #[must_use]
    pub const fn preferences(&self) -> &DisplayPreferences {
        &self.preferences
    }

    /// Get the current view state.
    #[must_use]
    pub const fn view(&self) -> &ViewState {
        &self.view
    }

    fn fleet_screen_mut(&mut self) -> Option<&mut FleetOverviewScreen> {
        self.shell
            .registry
            .get_mut(&self.fleet_screen_id)
            .and_then(|screen| screen.as_any_mut().downcast_mut::<FleetOverviewScreen>())
    }

    fn project_screen_mut(&mut self) -> Option<&mut ProjectDetailScreen> {
        self.shell
            .registry
            .get_mut(&self.project_screen_id)
            .and_then(|screen| screen.as_any_mut().downcast_mut::<ProjectDetailScreen>())
    }

    fn live_stream_screen_mut(&mut self) -> Option<&mut LiveSearchStreamScreen> {
        self.shell
            .registry
            .get_mut(&self.live_stream_screen_id)
            .and_then(|screen| screen.as_any_mut().downcast_mut::<LiveSearchStreamScreen>())
    }

    fn timeline_screen_mut(&mut self) -> Option<&mut ActionTimelineScreen> {
        self.shell
            .registry
            .get_mut(&self.timeline_screen_id)
            .and_then(|screen| screen.as_any_mut().downcast_mut::<ActionTimelineScreen>())
    }

    fn index_screen_mut(&mut self) -> Option<&mut IndexResourceScreen> {
        self.shell
            .registry
            .get_mut(&self.index_screen_id)
            .and_then(|screen| screen.as_any_mut().downcast_mut::<IndexResourceScreen>())
    }

    fn alerts_screen_mut(&mut self) -> Option<&mut AlertsSloScreen> {
        self.shell
            .registry
            .get_mut(&self.alerts_screen_id)
            .and_then(|screen| screen.as_any_mut().downcast_mut::<AlertsSloScreen>())
    }

    fn analytics_screen_mut(&mut self) -> Option<&mut HistoricalAnalyticsScreen> {
        self.shell
            .registry
            .get_mut(&self.analytics_screen_id)
            .and_then(|screen| {
                screen
                    .as_any_mut()
                    .downcast_mut::<HistoricalAnalyticsScreen>()
            })
    }

    fn selected_project_from_fleet(&self) -> Option<String> {
        self.fleet_screen()
            .and_then(FleetOverviewScreen::selected_project)
            .map(ToOwned::to_owned)
    }

    fn selected_project_from_alerts(&self) -> Option<String> {
        self.alerts_screen()
            .and_then(AlertsSloScreen::selected_project)
    }

    fn selected_project_from_analytics(&self) -> Option<String> {
        self.analytics_screen()
            .and_then(HistoricalAnalyticsScreen::selected_project)
    }

    fn selected_project_from_timeline(&self) -> Option<String> {
        self.shell
            .registry
            .get(&self.timeline_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<ActionTimelineScreen>())
            .and_then(ActionTimelineScreen::selected_project)
    }

    fn selected_reason_from_timeline(&self) -> Option<String> {
        self.shell
            .registry
            .get(&self.timeline_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<ActionTimelineScreen>())
            .and_then(ActionTimelineScreen::selected_reason_code)
    }

    fn selected_host_from_timeline(&self) -> Option<String> {
        self.shell
            .registry
            .get(&self.timeline_screen_id)
            .and_then(|screen| screen.as_any().downcast_ref::<ActionTimelineScreen>())
            .and_then(ActionTimelineScreen::selected_host)
    }

    fn open_project_detail_for_selected_project(&mut self) {
        self.view.apply_preset(ViewPreset::ProjectDeepDive);
        if let Some(project) = self.selected_project_from_fleet() {
            self.view.set_project_filter(project);
        }
        let id = self.project_screen_id.clone();
        self.shell.navigate_to(&id);
        self.sync_screen_states();
    }

    const fn is_enter_key(event: &frankensearch_tui::InputEvent) -> bool {
        matches!(
            event,
            frankensearch_tui::InputEvent::Key(crossterm::event::KeyCode::Enter, _)
        )
    }

    const fn is_enter_or_g_key(event: &frankensearch_tui::InputEvent) -> bool {
        matches!(
            event,
            frankensearch_tui::InputEvent::Key(
                crossterm::event::KeyCode::Enter | crossterm::event::KeyCode::Char('g'),
                _
            )
        )
    }

    const fn is_char_key(event: &frankensearch_tui::InputEvent, expected: char) -> bool {
        matches!(
            event,
            frankensearch_tui::InputEvent::Key(crossterm::event::KeyCode::Char(actual), _)
            if *actual == expected
        )
    }

    fn sync_fleet_to_project_transition(&mut self) {
        self.view.apply_preset(ViewPreset::ProjectDeepDive);
        if let Some(project) = self.selected_project_from_fleet() {
            self.view.set_project_filter(project);
        }
        self.sync_screen_states();
    }

    fn sync_alerts_to_project_transition(&mut self) {
        self.view.apply_preset(ViewPreset::ProjectDeepDive);
        if let Some(project) = self.selected_project_from_alerts() {
            self.view.set_project_filter(project);
        }
        self.sync_screen_states();
    }

    fn sync_analytics_to_project_transition(&mut self) {
        self.view.apply_preset(ViewPreset::ProjectDeepDive);
        if let Some(project) = self.selected_project_from_analytics() {
            self.view.set_project_filter(project);
        }
        self.sync_screen_states();
    }

    fn sync_timeline_to_project_transition(&mut self) {
        self.view.apply_preset(ViewPreset::ProjectDeepDive);
        if let Some(project) = self.selected_project_from_timeline() {
            self.view.set_project_filter(project);
        }
        self.sync_screen_states();
    }

    fn sync_timeline_to_analytics_transition(&mut self) {
        let project = self.selected_project_from_timeline();
        let reason = self.selected_reason_from_timeline();
        let host = self.selected_host_from_timeline();
        if let Some(screen) = self.analytics_screen_mut() {
            screen.set_project_filter(project.as_deref().unwrap_or("all"));
            screen.set_reason_filter(reason.as_deref().unwrap_or("all"));
            screen.set_host_filter(host.as_deref().unwrap_or("all"));
        }
        self.sync_screen_states();
    }

    fn sync_timeline_to_live_stream_transition(&mut self) {
        let project = self.selected_project_from_timeline();
        if let Some(screen) = self.live_stream_screen_mut() {
            screen.set_project_filter(project.as_deref().unwrap_or("all"));
        }
        self.sync_screen_states();
    }

    fn sync_project_to_fleet_transition(&mut self) {
        if self.view.preset == ViewPreset::ProjectDeepDive {
            self.view.apply_preset(ViewPreset::FleetTriage);
        } else {
            self.view.clear_project_filter();
        }
        self.sync_screen_states();
    }

    fn sync_project_filter_from_screen_transition(
        &mut self,
        previous_screen: Option<&ScreenId>,
        event: &frankensearch_tui::InputEvent,
    ) {
        let current_screen = self.shell.active_screen.clone();
        let moved_fleet_to_project = previous_screen == Some(&self.fleet_screen_id)
            && current_screen.as_ref() == Some(&self.project_screen_id);
        if moved_fleet_to_project && Self::is_enter_key(event) {
            self.sync_fleet_to_project_transition();
        }

        let moved_alerts_to_project = previous_screen == Some(&self.alerts_screen_id)
            && current_screen.as_ref() == Some(&self.project_screen_id);
        if moved_alerts_to_project && Self::is_enter_or_g_key(event) {
            self.sync_alerts_to_project_transition();
        }

        let moved_analytics_to_project = previous_screen == Some(&self.analytics_screen_id)
            && current_screen.as_ref() == Some(&self.project_screen_id);
        if moved_analytics_to_project && Self::is_enter_or_g_key(event) {
            self.sync_analytics_to_project_transition();
        }

        let moved_timeline_to_project = previous_screen == Some(&self.timeline_screen_id)
            && current_screen.as_ref() == Some(&self.project_screen_id);
        if moved_timeline_to_project && Self::is_enter_or_g_key(event) {
            self.sync_timeline_to_project_transition();
        }

        let moved_timeline_to_analytics = previous_screen == Some(&self.timeline_screen_id)
            && current_screen.as_ref() == Some(&self.analytics_screen_id);
        if moved_timeline_to_analytics && Self::is_char_key(event, 'a') {
            self.sync_timeline_to_analytics_transition();
        }

        let moved_timeline_to_live_stream = previous_screen == Some(&self.timeline_screen_id)
            && current_screen.as_ref() == Some(&self.live_stream_screen_id);
        if moved_timeline_to_live_stream && Self::is_char_key(event, 'l') {
            self.sync_timeline_to_live_stream_transition();
        }

        let moved_project_to_fleet = previous_screen == Some(&self.project_screen_id)
            && current_screen.as_ref() == Some(&self.fleet_screen_id);
        if moved_project_to_fleet {
            self.sync_project_to_fleet_transition();
        }
    }

    fn sync_screen_states(&mut self) {
        let state_snapshot = self.state.clone();
        let view_snapshot = self.view.clone();
        if let Some(screen) = self.fleet_screen_mut() {
            screen.update_state(&state_snapshot, &view_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.fleet_screen_id,
                "fleet screen missing or wrong type; skipping screen state refresh"
            );
        }
        if let Some(screen) = self.project_screen_mut() {
            screen.update_state(&state_snapshot, &view_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.project_screen_id,
                "project detail screen missing or wrong type; skipping screen state refresh"
            );
        }
        if let Some(screen) = self.live_stream_screen_mut() {
            screen.update_state(&state_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.live_stream_screen_id,
                "live stream screen missing or wrong type; skipping screen state refresh"
            );
        }
        if let Some(screen) = self.timeline_screen_mut() {
            screen.update_state(&state_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.timeline_screen_id,
                "timeline screen missing or wrong type; skipping screen state refresh"
            );
        }
        if let Some(screen) = self.index_screen_mut() {
            screen.update_state(&state_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.index_screen_id,
                "index/resource screen missing or wrong type; skipping screen state refresh"
            );
        }
        if let Some(screen) = self.alerts_screen_mut() {
            screen.update_state(&state_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.alerts_screen_id,
                "alerts/slo screen missing or wrong type; skipping screen state refresh"
            );
        }
        if let Some(screen) = self.analytics_screen_mut() {
            screen.update_state(&state_snapshot);
        } else {
            tracing::warn!(
                target: "frankensearch.ops",
                screen_id = %self.analytics_screen_id,
                "historical analytics screen missing or wrong type; skipping screen state refresh"
            );
        }
    }

    #[must_use]
    const fn health_rank(health: ControlPlaneHealth) -> u8 {
        match health {
            ControlPlaneHealth::Healthy => 0,
            ControlPlaneHealth::Degraded => 1,
            ControlPlaneHealth::Critical => 2,
        }
    }

    fn maybe_emit_control_plane_alert(
        &mut self,
        previous_health: ControlPlaneHealth,
        current_health: ControlPlaneHealth,
    ) {
        if Self::health_rank(current_health) < Self::health_rank(previous_health) {
            self.last_alerted_health = None;
            return;
        }
        if Self::health_rank(current_health) <= Self::health_rank(previous_health) {
            return;
        }
        if self.last_alerted_health == Some(current_health) {
            return;
        }

        let title = match current_health {
            ControlPlaneHealth::Healthy => return,
            ControlPlaneHealth::Degraded => "Control Plane Degraded",
            ControlPlaneHealth::Critical => "Control Plane Critical",
        };
        tracing::warn!(
            target: "frankensearch.ops",
            previous = %previous_health,
            current = %current_health,
            "control-plane health degraded"
        );
        let body = format!(
            "health transition: {previous_health} -> {current_health}\n\n{}",
            self.state.self_check_report()
        );
        self.shell
            .overlays
            .push(OverlayRequest::new(OverlayKind::Alert, title).with_body(body));
        self.last_alerted_health = Some(current_health);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use frankensearch_tui::palette::PaletteState;
    use frankensearch_tui::screen::ScreenId;

    use super::*;
    use crate::data_source::{DataSource, MockDataSource, TimeWindow};
    use crate::preferences::ContrastMode;
    use crate::state::{
        ControlPlaneMetrics, FleetSnapshot, InstanceAttribution, InstanceLifecycle,
        ResourceMetrics, SearchMetrics,
    };
    use crate::{DiscoveredInstance, DiscoverySignalKind, DiscoveryStatus};

    struct SequencedControlPlaneSource {
        metrics: Vec<ControlPlaneMetrics>,
        index: AtomicUsize,
    }

    impl SequencedControlPlaneSource {
        fn new(metrics: Vec<ControlPlaneMetrics>) -> Self {
            assert!(!metrics.is_empty());
            Self {
                metrics,
                index: AtomicUsize::new(0),
            }
        }
    }

    impl DataSource for SequencedControlPlaneSource {
        fn fleet_snapshot(&self) -> FleetSnapshot {
            FleetSnapshot::default()
        }

        fn search_metrics(&self, _instance_id: &str, _window: TimeWindow) -> Option<SearchMetrics> {
            None
        }

        fn resource_metrics(&self, _instance_id: &str) -> Option<ResourceMetrics> {
            None
        }

        fn control_plane_metrics(&self) -> ControlPlaneMetrics {
            let idx = self.index.fetch_add(1, Ordering::Relaxed);
            let bounded = idx.min(self.metrics.len() - 1);
            self.metrics[bounded].clone()
        }

        fn attribution(&self, _instance_id: &str) -> Option<InstanceAttribution> {
            None
        }

        fn lifecycle(&self, _instance_id: &str) -> Option<InstanceLifecycle> {
            None
        }
    }

    fn healthy_metrics() -> ControlPlaneMetrics {
        ControlPlaneMetrics::default()
    }

    fn degraded_metrics() -> ControlPlaneMetrics {
        ControlPlaneMetrics {
            ingestion_lag_events: 2_500,
            event_throughput_eps: 10.0,
            ..ControlPlaneMetrics::default()
        }
    }

    fn critical_metrics() -> ControlPlaneMetrics {
        ControlPlaneMetrics {
            ingestion_lag_events: 12_000,
            event_throughput_eps: 10.0,
            ..ControlPlaneMetrics::default()
        }
    }

    #[test]
    fn ops_app_creation() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.should_quit());
        assert!(app.shell.active_screen.is_some());
    }

    #[test]
    fn ops_app_refresh_data() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        assert!(app.state.has_data());
        assert_eq!(app.state.fleet().instance_count(), 3);
    }

    #[test]
    fn ops_app_empty_data_source() {
        let mut app = OpsApp::new(Box::new(MockDataSource::empty()));
        app.refresh_data();
        assert!(app.state.has_data());
        assert_eq!(app.state.fleet().instance_count(), 0);
    }

    #[test]
    fn ops_app_quit() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        let event = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&event);
        assert!(quit);
        assert!(app.should_quit());
    }

    #[test]
    fn palette_actions_nonempty() {
        let actions = OpsApp::palette_actions();
        assert!(!actions.is_empty());
        assert!(actions.iter().any(|a| a.id == "nav.fleet"));
        assert!(actions.iter().any(|a| a.id == "nav.project"));
        assert!(actions.iter().any(|a| a.id == "nav.alerts"));
        assert!(actions.iter().any(|a| a.id == "nav.analytics"));
        assert!(actions.iter().any(|a| a.id == "analytics.export_snapshot"));
    }

    #[test]
    fn palette_actions_include_accessibility() {
        let actions = OpsApp::palette_actions();
        assert!(actions.iter().any(|a| a.id == "settings.contrast"));
        assert!(actions.iter().any(|a| a.id == "settings.motion"));
        assert!(actions.iter().any(|a| a.id == "settings.focus"));
        assert!(actions.iter().any(|a| a.id == "settings.hints"));
        assert!(actions.iter().any(|a| a.id == "debug.self_check"));
    }

    #[test]
    fn fleet_screen_accessible() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let fleet = app.fleet_screen().expect("fleet screen should exist");
        assert_eq!(fleet.id(), &ScreenId::new("ops.fleet"));
    }

    #[test]
    fn project_screen_accessible() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let project = app.project_screen().expect("project screen should exist");
        assert_eq!(project.id(), &ScreenId::new("ops.project"));
    }

    #[test]
    fn index_screen_accessible() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let index = app.index_screen().expect("index screen should exist");
        assert_eq!(index.id(), &ScreenId::new("ops.index"));
    }

    #[test]
    fn alerts_screen_accessible() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let alerts = app.alerts_screen().expect("alerts screen should exist");
        assert_eq!(alerts.id(), &ScreenId::new("ops.alerts"));
    }

    #[test]
    fn analytics_screen_accessible() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let analytics = app
            .analytics_screen()
            .expect("analytics screen should exist");
        assert_eq!(analytics.id(), &ScreenId::new("ops.analytics"));
    }

    #[test]
    fn refresh_updates_registered_fleet_screen_state() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        let fleet = app.fleet_screen().expect("fleet screen should exist");
        assert_eq!(fleet.instance_count(), 3);
    }

    #[test]
    fn palette_registered_on_creation() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.shell.palette.is_empty());
        assert_eq!(app.shell.palette.len(), OpsApp::palette_actions().len());
    }

    #[test]
    fn palette_toggle_via_ctrl_p() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.shell.palette.state(), &PaletteState::Closed);

        let event = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        app.handle_input(&event);
        assert_eq!(app.shell.palette.state(), &PaletteState::Open);

        // Esc closes.
        let esc = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        );
        app.handle_input(&esc);
        assert_eq!(app.shell.palette.state(), &PaletteState::Closed);
    }

    #[test]
    fn palette_text_input() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));

        // Open palette.
        let open = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char(':'),
            crossterm::event::KeyModifiers::NONE,
        );
        app.handle_input(&open);
        assert_eq!(app.shell.palette.state(), &PaletteState::Open);

        // Type "fleet".
        for ch in "fleet".chars() {
            let event = frankensearch_tui::InputEvent::Key(
                crossterm::event::KeyCode::Char(ch),
                crossterm::event::KeyModifiers::NONE,
            );
            app.handle_input(&event);
        }
        assert_eq!(app.shell.palette.query(), "fleet");

        // Filtered results should include fleet action.
        let filtered = app.shell.palette.filtered();
        assert!(filtered.iter().any(|a| a.id == "nav.fleet"));
    }

    #[test]
    fn enter_on_fleet_opens_project_detail_with_project_filter() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.shell.overlays.dismiss();

        let enter = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&enter);
        assert!(!quit);
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.project")));
        assert_eq!(app.view.project_filter.as_deref(), Some("cass"));
        assert_eq!(app.view.preset, crate::presets::ViewPreset::ProjectDeepDive);
    }

    #[test]
    fn esc_from_project_detail_returns_to_fleet_and_clears_project_filter() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.shell.overlays.dismiss();
        app.handle_input(&frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.project")));
        assert_eq!(app.view.project_filter.as_deref(), Some("cass"));

        let esc = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&esc);
        assert!(!quit);
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.fleet")));
        assert!(app.view.project_filter.is_none());
        assert_eq!(app.view.preset, crate::presets::ViewPreset::FleetTriage);
    }

    #[test]
    fn nav_project_action_opens_selected_project_detail() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.dispatch_palette_action("nav.project");
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.project")));
        assert_eq!(app.view.project_filter.as_deref(), Some("cass"));
    }

    #[test]
    fn nav_alerts_action_opens_alerts_screen() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.dispatch_palette_action("nav.alerts");
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.alerts")));
    }

    #[test]
    fn nav_analytics_action_opens_analytics_screen() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.dispatch_palette_action("nav.analytics");
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.analytics"))
        );
    }

    #[test]
    fn analytics_export_snapshot_action_opens_overlay_and_analytics_screen() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.dispatch_palette_action("analytics.export_snapshot");
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.analytics"))
        );
        let overlay = app.shell.overlays.top().expect("overlay should be visible");
        assert_eq!(overlay.kind, OverlayKind::Alert);
        assert_eq!(overlay.title, "Analytics Snapshot Export");
        let body = overlay
            .body
            .as_deref()
            .expect("analytics export overlay should include body");
        assert!(body.contains("Snapshot mode toggled"));
        assert!(body.contains("Selected project"));
    }

    #[test]
    fn g_from_alerts_opens_project_detail_with_project_context() {
        let discovered = vec![DiscoveredInstance {
            instance_id: "host-a:cass-001".to_string(),
            project_key_hint: Some("cass".to_string()),
            host_name: Some("cass-host".to_string()),
            pid: Some(4242),
            version: Some("0.1.0".to_string()),
            first_seen_ms: 1_000,
            last_seen_ms: 2_000,
            status: DiscoveryStatus::Active,
            sources: vec![DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["instance:host-a:cass-001".to_string()],
        }];
        let mut app = OpsApp::new(Box::new(MockDataSource::from_discovery(&discovered)));
        app.refresh_data();
        app.dispatch_palette_action("nav.alerts");
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.alerts")));

        let goto_project = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('g'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_project);
        assert!(!quit);
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.project")));
        assert!(app.view.project_filter.is_some());
        assert_eq!(app.view.preset, crate::presets::ViewPreset::ProjectDeepDive);
    }

    #[test]
    fn g_from_analytics_opens_project_detail_with_project_context() {
        let discovered = vec![DiscoveredInstance {
            instance_id: "host-a:cass-002".to_string(),
            project_key_hint: Some("cass".to_string()),
            host_name: Some("cass-host".to_string()),
            pid: Some(4343),
            version: Some("0.1.0".to_string()),
            first_seen_ms: 1_000,
            last_seen_ms: 2_000,
            status: DiscoveryStatus::Active,
            sources: vec![DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["instance:host-a:cass-002".to_string()],
        }];
        let mut app = OpsApp::new(Box::new(MockDataSource::from_discovery(&discovered)));
        app.refresh_data();
        app.dispatch_palette_action("nav.analytics");
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.analytics"))
        );

        let goto_project = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('g'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_project);
        assert!(!quit);
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.project")));
        assert!(app.view.project_filter.is_some());
        assert_eq!(app.view.preset, crate::presets::ViewPreset::ProjectDeepDive);
    }

    #[test]
    fn g_from_timeline_opens_project_detail_with_project_context() {
        let discovered = vec![DiscoveredInstance {
            instance_id: "host-a:cass-003".to_string(),
            project_key_hint: Some("cass".to_string()),
            host_name: Some("cass-host".to_string()),
            pid: Some(4444),
            version: Some("0.1.0".to_string()),
            first_seen_ms: 1_000,
            last_seen_ms: 2_000,
            status: DiscoveryStatus::Active,
            sources: vec![DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["instance:host-a:cass-003".to_string()],
        }];
        let mut app = OpsApp::new(Box::new(MockDataSource::from_discovery(&discovered)));
        app.refresh_data();
        app.shell.navigate_to(&ScreenId::new("ops.timeline"));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.timeline")));

        let goto_project = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('g'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_project);
        assert!(!quit);
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.project")));
        assert!(app.view.project_filter.is_some());
        assert_eq!(app.view.preset, crate::presets::ViewPreset::ProjectDeepDive);
    }

    #[test]
    fn a_from_timeline_opens_analytics_with_project_context() {
        let discovered = vec![DiscoveredInstance {
            instance_id: "host-a:cass-004".to_string(),
            project_key_hint: Some("cass".to_string()),
            host_name: Some("cass-host".to_string()),
            pid: Some(4545),
            version: Some("0.1.0".to_string()),
            first_seen_ms: 1_000,
            last_seen_ms: 2_000,
            status: DiscoveryStatus::Active,
            sources: vec![DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["instance:host-a:cass-004".to_string()],
        }];
        let mut app = OpsApp::new(Box::new(MockDataSource::from_discovery(&discovered)));
        app.refresh_data();
        app.shell.navigate_to(&ScreenId::new("ops.timeline"));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.timeline")));
        let expected_project = app.selected_project_from_timeline();

        let goto_analytics = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_analytics);
        assert!(!quit);
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.analytics"))
        );
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_project_filter)
                .as_deref(),
            expected_project.as_deref()
        );
    }

    #[test]
    fn a_from_timeline_overrides_stale_analytics_reason_and_host_filters() {
        let mut app = OpsApp::new(Box::new(MockDataSource::empty()));
        app.state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "host-a:cass-ctx-1".to_owned(),
                    project: "cass".to_owned(),
                    pid: Some(1),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "host-b:xf-ctx-1".to_owned(),
                    project: "xf".to_owned(),
                    pid: Some(2),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 0,
                },
            ],
            lifecycle_events: vec![
                crate::state::LifecycleEvent {
                    instance_id: "host-a:cass-ctx-1".to_owned(),
                    from: crate::state::LifecycleState::Started,
                    to: crate::state::LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.ready".to_owned(),
                    at_ms: 2_000,
                    attribution_confidence_score: 95,
                    attribution_collision: false,
                },
                crate::state::LifecycleEvent {
                    instance_id: "host-b:xf-ctx-1".to_owned(),
                    from: crate::state::LifecycleState::Healthy,
                    to: crate::state::LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 1_000,
                    attribution_confidence_score: 65,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        });
        app.sync_screen_states();

        if let Some(screen) = app.analytics_screen_mut() {
            screen.set_reason_filter("lifecycle.heartbeat_gap");
            screen.set_host_filter("host-b");
        }
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_reason_filter)
                .as_deref(),
            Some("lifecycle.heartbeat_gap")
        );
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_host_filter)
                .as_deref(),
            Some("host-b")
        );

        app.shell.navigate_to(&ScreenId::new("ops.timeline"));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.timeline")));
        let expected_project = app.selected_project_from_timeline();
        let expected_reason = app.selected_reason_from_timeline();
        let expected_host = app.selected_host_from_timeline();

        let goto_analytics = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_analytics);
        assert!(!quit);
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.analytics"))
        );
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_project_filter)
                .as_deref(),
            expected_project.as_deref()
        );
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_reason_filter)
                .as_deref(),
            expected_reason.as_deref()
        );
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_host_filter)
                .as_deref(),
            expected_host.as_deref()
        );
    }

    #[test]
    fn l_from_timeline_opens_live_stream_with_project_context() {
        let discovered = vec![DiscoveredInstance {
            instance_id: "host-a:cass-005".to_string(),
            project_key_hint: Some("cass".to_string()),
            host_name: Some("cass-host".to_string()),
            pid: Some(4646),
            version: Some("0.1.0".to_string()),
            first_seen_ms: 1_000,
            last_seen_ms: 2_000,
            status: DiscoveryStatus::Active,
            sources: vec![DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["instance:host-a:cass-005".to_string()],
        }];
        let mut app = OpsApp::new(Box::new(MockDataSource::from_discovery(&discovered)));
        app.refresh_data();
        app.shell.navigate_to(&ScreenId::new("ops.timeline"));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.timeline")));
        let expected_project = app.selected_project_from_timeline();

        let goto_stream = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('l'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_stream);
        assert!(!quit);
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.live_stream"))
        );
        assert_eq!(
            app.live_stream_screen()
                .and_then(LiveSearchStreamScreen::active_project_filter)
                .as_deref(),
            expected_project.as_deref()
        );
    }

    #[test]
    fn a_from_timeline_with_unattributed_row_resets_analytics_project_filter_to_all() {
        let mut app = OpsApp::new(Box::new(MockDataSource::empty()));
        app.state.update_fleet(FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "known-001".to_owned(),
                project: "known".to_owned(),
                pid: Some(1),
                healthy: true,
                doc_count: 10,
                pending_jobs: 0,
            }],
            lifecycle_events: vec![
                crate::state::LifecycleEvent {
                    instance_id: "orphan-001".to_owned(),
                    from: crate::state::LifecycleState::Started,
                    to: crate::state::LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.orphan".to_owned(),
                    at_ms: 2_000,
                    attribution_confidence_score: 50,
                    attribution_collision: false,
                },
                crate::state::LifecycleEvent {
                    instance_id: "known-001".to_owned(),
                    from: crate::state::LifecycleState::Started,
                    to: crate::state::LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.known".to_owned(),
                    at_ms: 1_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        });
        app.sync_screen_states();

        if let Some(screen) = app.analytics_screen_mut() {
            screen.set_project_filter("known");
        }
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_project_filter)
                .as_deref(),
            Some("known")
        );

        app.shell.navigate_to(&ScreenId::new("ops.timeline"));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.timeline")));
        assert!(
            app.selected_project_from_timeline().is_none(),
            "top timeline row should be unattributed in this test"
        );

        let goto_analytics = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_analytics);
        assert!(!quit);
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.analytics"))
        );
        assert_eq!(
            app.analytics_screen()
                .and_then(HistoricalAnalyticsScreen::active_project_filter),
            None
        );
    }

    #[test]
    fn l_from_timeline_with_unattributed_row_resets_live_stream_project_filter_to_all() {
        let mut app = OpsApp::new(Box::new(MockDataSource::empty()));
        app.state.update_fleet(FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "known-001".to_owned(),
                project: "known".to_owned(),
                pid: Some(1),
                healthy: true,
                doc_count: 10,
                pending_jobs: 0,
            }],
            lifecycle_events: vec![
                crate::state::LifecycleEvent {
                    instance_id: "orphan-001".to_owned(),
                    from: crate::state::LifecycleState::Started,
                    to: crate::state::LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.orphan".to_owned(),
                    at_ms: 2_000,
                    attribution_confidence_score: 50,
                    attribution_collision: false,
                },
                crate::state::LifecycleEvent {
                    instance_id: "known-001".to_owned(),
                    from: crate::state::LifecycleState::Started,
                    to: crate::state::LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.known".to_owned(),
                    at_ms: 1_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        });
        app.sync_screen_states();

        if let Some(screen) = app.live_stream_screen_mut() {
            screen.set_project_filter("known");
        }
        assert_eq!(
            app.live_stream_screen()
                .and_then(LiveSearchStreamScreen::active_project_filter)
                .as_deref(),
            Some("known")
        );

        app.shell.navigate_to(&ScreenId::new("ops.timeline"));
        assert_eq!(app.shell.active_screen, Some(ScreenId::new("ops.timeline")));
        assert!(
            app.selected_project_from_timeline().is_none(),
            "top timeline row should be unattributed in this test"
        );

        let goto_live_stream = frankensearch_tui::InputEvent::Key(
            crossterm::event::KeyCode::Char('l'),
            crossterm::event::KeyModifiers::NONE,
        );
        let quit = app.handle_input(&goto_live_stream);
        assert!(!quit);
        assert_eq!(
            app.shell.active_screen,
            Some(ScreenId::new("ops.live_stream"))
        );
        assert_eq!(
            app.live_stream_screen()
                .and_then(LiveSearchStreamScreen::active_project_filter),
            None
        );
    }

    #[test]
    fn dispatch_contrast_toggle() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.preferences.contrast, ContrastMode::Normal);
        app.dispatch_palette_action("settings.contrast");
        assert_eq!(app.preferences.contrast, ContrastMode::High);
    }

    #[test]
    fn dispatch_refresh() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.state.has_data());
        app.dispatch_palette_action("debug.refresh");
        assert!(app.state.has_data());
        assert_eq!(
            app.shell.status_line.right,
            app.state.control_plane_health().badge()
        );
    }

    #[test]
    fn dispatch_self_check_opens_overlay() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        app.refresh_data();
        app.dispatch_palette_action("debug.self_check");
        let overlay = app.shell.overlays.top().expect("overlay should be visible");
        assert_eq!(overlay.kind, OverlayKind::Alert);
        assert_eq!(overlay.title, "Control Plane Self-Check");
        assert!(
            overlay
                .body
                .as_deref()
                .is_some_and(|body| body.contains("ingestion_lag_events"))
        );
    }

    #[test]
    fn refresh_emits_alert_when_health_degrades() {
        let source = SequencedControlPlaneSource::new(vec![healthy_metrics(), degraded_metrics()]);
        let mut app = OpsApp::new(Box::new(source));

        app.refresh_data();
        assert_eq!(app.shell.overlays.depth(), 0);

        app.refresh_data();
        assert_eq!(
            app.state.control_plane_health(),
            ControlPlaneHealth::Degraded
        );
        assert_eq!(app.shell.overlays.depth(), 1);
        let overlay = app
            .shell
            .overlays
            .top()
            .expect("degradation alert should exist");
        assert_eq!(overlay.kind, OverlayKind::Alert);
        assert_eq!(overlay.title, "Control Plane Degraded");
        assert!(
            overlay
                .body
                .as_deref()
                .is_some_and(|body| body.contains("health transition: healthy -> degraded"))
        );
    }

    #[test]
    fn refresh_does_not_duplicate_alert_when_state_stays_degraded() {
        let source = SequencedControlPlaneSource::new(vec![
            healthy_metrics(),
            degraded_metrics(),
            degraded_metrics(),
        ]);
        let mut app = OpsApp::new(Box::new(source));

        app.refresh_data();
        app.refresh_data();
        assert_eq!(app.shell.overlays.depth(), 1);

        app.refresh_data();
        assert_eq!(
            app.state.control_plane_health(),
            ControlPlaneHealth::Degraded
        );
        assert_eq!(app.shell.overlays.depth(), 1);
    }

    #[test]
    fn refresh_emits_critical_alert_when_health_worsens_again() {
        let source = SequencedControlPlaneSource::new(vec![
            healthy_metrics(),
            degraded_metrics(),
            critical_metrics(),
        ]);
        let mut app = OpsApp::new(Box::new(source));

        app.refresh_data();
        app.refresh_data();
        assert_eq!(app.shell.overlays.depth(), 1);

        app.refresh_data();
        assert_eq!(
            app.state.control_plane_health(),
            ControlPlaneHealth::Critical
        );
        assert_eq!(app.shell.overlays.depth(), 2);
        let overlay = app
            .shell
            .overlays
            .top()
            .expect("critical alert should exist");
        assert_eq!(overlay.title, "Control Plane Critical");
        assert!(
            overlay
                .body
                .as_deref()
                .is_some_and(|body| body.contains("health transition: degraded -> critical"))
        );
    }

    #[test]
    fn preferences_default() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        let prefs = app.preferences();
        assert_eq!(prefs.contrast, ContrastMode::Normal);
        assert!(prefs.show_shortcut_hints);
    }

    #[test]
    fn view_state_default() {
        let app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.view().preset, crate::presets::ViewPreset::FleetTriage);
        assert_eq!(app.view().density, crate::presets::Density::Compact);
    }

    #[test]
    fn dispatch_cycle_density() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.view.density, crate::presets::Density::Compact);
        app.dispatch_palette_action("view.density");
        assert_eq!(app.view.density, crate::presets::Density::Normal);
        app.dispatch_palette_action("view.density");
        assert_eq!(app.view.density, crate::presets::Density::Expanded);
    }

    #[test]
    fn dispatch_incident_mode_sets_high_contrast() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert_eq!(app.preferences.contrast, ContrastMode::Normal);
        app.dispatch_palette_action("view.incident_mode");
        assert_eq!(app.view.preset, crate::presets::ViewPreset::IncidentMode);
        assert_eq!(app.preferences.contrast, ContrastMode::High);
    }

    #[test]
    fn dispatch_low_noise_hides_healthy() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.view.hide_healthy);
        app.dispatch_palette_action("view.low_noise");
        assert!(app.view.hide_healthy);
    }

    #[test]
    fn dispatch_toggle_hide_healthy() {
        let mut app = OpsApp::new(Box::new(MockDataSource::sample()));
        assert!(!app.view.hide_healthy);
        app.dispatch_palette_action("view.hide_healthy");
        assert!(app.view.hide_healthy);
        app.dispatch_palette_action("view.hide_healthy");
        assert!(!app.view.hide_healthy);
    }

    #[test]
    fn palette_actions_include_view_presets() {
        let actions = OpsApp::palette_actions();
        assert!(actions.iter().any(|a| a.id == "view.density"));
        assert!(actions.iter().any(|a| a.id == "view.fleet_triage"));
        assert!(actions.iter().any(|a| a.id == "view.incident_mode"));
        assert!(actions.iter().any(|a| a.id == "view.low_noise"));
    }
}
