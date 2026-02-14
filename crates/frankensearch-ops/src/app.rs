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
use crate::screens::FleetOverviewScreen;
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

        // Create and register the fleet screen.
        let fleet_screen = FleetOverviewScreen::new();
        let fleet_id = fleet_screen.id().clone();
        shell.registry.register(Box::new(fleet_screen));

        // Set initial active screen.
        shell.navigate_to(&fleet_id);

        Self {
            shell,
            state: AppState::new(),
            data_source,
            fleet_screen_id: fleet_id,
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
        self.sync_fleet_screen_state();
    }

    /// Process an input event. Returns `true` if the app should quit.
    pub fn handle_input(&mut self, event: &frankensearch_tui::InputEvent) -> bool {
        let quit = self.shell.handle_input(event);

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
    pub fn render(&self, frame: &mut Frame<'_>) {
        self.shell.render(frame);
        let area = frame.area();
        if let Some(request) = self.shell.overlays.top() {
            crate::overlays::render_overlay(frame, area, request);
        }
        crate::overlays::render_palette_overlay(frame, area, &self.shell.palette);
    }

    /// Dispatch a confirmed palette action by ID.
    fn dispatch_palette_action(&mut self, action_id: &str) {
        match action_id {
            "nav.fleet" => {
                let id = frankensearch_tui::screen::ScreenId::new("ops.fleet");
                self.shell.navigate_to(&id);
            }
            "nav.search" | "nav.index" => {
                tracing::warn!(
                    target: "frankensearch.ops",
                    action_id,
                    "palette navigation action is not implemented yet"
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
                self.shell.config.theme = if self.shell.config.theme == Theme::dark() {
                    Theme::light()
                } else {
                    Theme::dark()
                };
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
                self.sync_fleet_screen_state();
            }
            "view.fleet_triage" => {
                self.view.apply_preset(ViewPreset::FleetTriage);
                self.sync_fleet_screen_state();
            }
            "view.project_deep_dive" => {
                self.view.apply_preset(ViewPreset::ProjectDeepDive);
                self.sync_fleet_screen_state();
            }
            "view.incident_mode" => {
                self.view.apply_preset(ViewPreset::IncidentMode);
                if self.view.preset.prefer_high_contrast() {
                    self.preferences.contrast = crate::preferences::ContrastMode::High;
                }
                self.sync_fleet_screen_state();
            }
            "view.low_noise" => {
                self.view.apply_preset(ViewPreset::LowNoise);
                self.sync_fleet_screen_state();
            }
            "view.hide_healthy" => {
                self.view.toggle_hide_healthy();
                self.sync_fleet_screen_state();
            }
            "view.unhealthy_first" => {
                self.view.toggle_unhealthy_first();
                self.sync_fleet_screen_state();
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

    /// Get a list of all registered palette actions (for help screen).
    #[must_use]
    pub fn palette_actions() -> Vec<Action> {
        vec![
            Action::new(
                "nav.fleet",
                "Go to Fleet Overview",
                ActionCategory::Navigation,
            )
            .with_shortcut("1"),
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
            Action::new("debug.refresh", "Force Refresh Data", ActionCategory::Debug)
                .with_shortcut("F5"),
            Action::new(
                "debug.self_check",
                "Run Control-Plane Self-Check",
                ActionCategory::Debug,
            )
            .with_description("Show internal health metrics and degradation status"),
            Action::new("settings.theme", "Toggle Theme", ActionCategory::Settings)
                .with_shortcut("T")
                .with_description("Switch between dark and light theme"),
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

    fn sync_fleet_screen_state(&mut self) {
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
