//! Project detail dashboard screen.
//!
//! Provides a project-scoped operational deep-dive with rollup summary cards,
//! per-instance inventory, and fast drilldowns into stream/timeline surfaces.

use std::any::Any;

use frankensearch_core::LifecycleState;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::presets::ViewState;
use crate::state::AppState;

/// Project-scoped dashboard screen.
pub struct ProjectDetailScreen {
    id: ScreenId,
    fleet_screen_id: ScreenId,
    live_stream_screen_id: ScreenId,
    timeline_screen_id: ScreenId,
    analytics_screen_id: ScreenId,
    state: AppState,
    view: ViewState,
    selected_row: usize,
}

impl ProjectDetailScreen {
    /// Create a new project detail screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.project"),
            fleet_screen_id: ScreenId::new("ops.fleet"),
            live_stream_screen_id: ScreenId::new("ops.live_stream"),
            timeline_screen_id: ScreenId::new("ops.timeline"),
            analytics_screen_id: ScreenId::new("ops.timeline"),
            state: AppState::new(),
            view: ViewState::default(),
            selected_row: 0,
        }
    }

    /// Override the fleet drilldown destination used for back navigation.
    pub fn set_fleet_screen_id(&mut self, id: ScreenId) {
        self.fleet_screen_id = id;
    }

    /// Override the live stream drilldown destination used for `s`.
    pub fn set_live_stream_screen_id(&mut self, id: ScreenId) {
        self.live_stream_screen_id = id;
    }

    /// Override the timeline drilldown destination used for `t`.
    pub fn set_timeline_screen_id(&mut self, id: ScreenId) {
        self.timeline_screen_id = id;
    }

    /// Override the analytics drilldown destination used for `a`.
    pub fn set_analytics_screen_id(&mut self, id: ScreenId) {
        self.analytics_screen_id = id;
    }

    /// Update screen state from shared app state.
    pub fn update_state(&mut self, state: &AppState, view: &ViewState) {
        self.state = state.clone();
        self.view = view.clone();
        let count = self.project_instances().len();
        if count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= count {
            self.selected_row = count.saturating_sub(1);
        }
    }

    /// Selected project filter currently in effect.
    #[must_use]
    pub fn selected_project(&self) -> Option<&str> {
        self.view.project_filter.as_deref()
    }

    fn project_instances(&self) -> Vec<&crate::state::InstanceInfo> {
        let Some(project) = self.selected_project() else {
            return Vec::new();
        };
        self.state
            .fleet()
            .instances
            .iter()
            .filter(|instance| instance.project.eq_ignore_ascii_case(project))
            .collect()
    }

    fn render_bar(value: u64, max_value: u64, width: usize) -> String {
        if width == 0 {
            return String::new();
        }

        let safe_max = max_value.max(1);
        let width_u64 = u64::try_from(width).unwrap_or(u64::MAX);
        let filled_u64 = value
            .saturating_mul(width_u64)
            .saturating_add(safe_max / 2)
            .saturating_div(safe_max)
            .min(width_u64);
        let filled = usize::try_from(filled_u64).unwrap_or(width);
        let mut bar = String::with_capacity(width);
        bar.push_str(&"=".repeat(filled));
        bar.push_str(&"-".repeat(width.saturating_sub(filled)));
        bar
    }

    fn phase_latency_lines(
        instances: &[&crate::state::InstanceInfo],
        fleet: &crate::state::FleetSnapshot,
    ) -> Vec<Line<'static>> {
        let mut weighted_count = 0_u64;
        let mut weighted_avg_sum = 0_u128;
        let mut weighted_p95_sum = 0_u128;
        let mut refined_total = 0_u64;

        for instance in instances {
            let Some(metrics) = fleet.search_metrics.get(&instance.id) else {
                continue;
            };
            let weight = metrics.total_searches.max(1);
            weighted_count = weighted_count.saturating_add(weight);
            weighted_avg_sum = weighted_avg_sum
                .saturating_add(u128::from(metrics.avg_latency_us) * u128::from(weight));
            weighted_p95_sum = weighted_p95_sum
                .saturating_add(u128::from(metrics.p95_latency_us) * u128::from(weight));
            refined_total = refined_total.saturating_add(metrics.refined_count.min(weight));
        }

        let initial_avg_us = if weighted_count == 0 {
            0
        } else {
            let value = weighted_avg_sum / u128::from(weighted_count);
            u64::try_from(value).unwrap_or(u64::MAX)
        };
        let refined_p95_us = if weighted_count == 0 {
            0
        } else {
            let value = weighted_p95_sum / u128::from(weighted_count);
            u64::try_from(value).unwrap_or(u64::MAX)
        };
        let refined_share_pct = if weighted_count == 0 {
            0
        } else {
            let value = refined_total
                .saturating_mul(100)
                .saturating_add(weighted_count / 2)
                .saturating_div(weighted_count)
                .min(100);
            u8::try_from(value).unwrap_or(100)
        };

        let bar_max = initial_avg_us.max(refined_p95_us).max(1);
        let initial_bar = Self::render_bar(initial_avg_us, bar_max, 16);
        let refined_bar = Self::render_bar(refined_p95_us, bar_max, 16);
        vec![
            Line::from("Phase latency bars (weighted):"),
            Line::from(format!("  initial [{initial_bar}] {initial_avg_us:>6}us")),
            Line::from(format!(
                "  refined [{refined_bar}] {refined_p95_us:>6}us | refined_share={refined_share_pct}%"
            )),
        ]
    }

    fn anomaly_lines(
        instances: &[&crate::state::InstanceInfo],
        fleet: &crate::state::FleetSnapshot,
    ) -> Vec<Line<'static>> {
        let mut cards: Vec<(u32, String, String)> = Vec::new();
        for instance in instances {
            let mut score = 0_u32;
            let mut signals: Vec<String> = Vec::new();

            if !instance.healthy {
                score = score.saturating_add(45);
                signals.push("unhealthy".to_owned());
            }

            if instance.pending_jobs >= 1_000 {
                score = score.saturating_add(30);
                signals.push(format!("pending={}", instance.pending_jobs));
            } else if instance.pending_jobs >= 100 {
                score = score.saturating_add(15);
                signals.push(format!("pending={}", instance.pending_jobs));
            }

            if let Some(metrics) = fleet.search_metrics.get(&instance.id) {
                if metrics.p95_latency_us >= 5_000 {
                    score = score.saturating_add(30);
                    signals.push(format!("p95={}us", metrics.p95_latency_us));
                } else if metrics.p95_latency_us >= 2_000 {
                    score = score.saturating_add(15);
                    signals.push(format!("p95={}us", metrics.p95_latency_us));
                }
            }

            if let Some(attribution) = fleet.attribution.get(&instance.id) {
                if attribution.collision {
                    score = score.saturating_add(20);
                    signals.push("attribution_collision".to_owned());
                }
                if attribution.confidence_score < 70 {
                    score = score.saturating_add(10);
                    signals.push(format!("attr={}%", attribution.confidence_score));
                }
            }

            if let Some(lifecycle) = fleet.lifecycle_for(&instance.id)
                && matches!(
                    lifecycle.state,
                    LifecycleState::Degraded
                        | LifecycleState::Recovering
                        | LifecycleState::Stale
                        | LifecycleState::Stopped
                )
            {
                score = score.saturating_add(15);
                signals.push(format!("lifecycle={:?}", lifecycle.state));
            }

            if score > 0 && !signals.is_empty() {
                let severity = if score >= 80 { "CRIT" } else { "WARN" };
                cards.push((
                    score,
                    instance.id.clone(),
                    format!("{severity} {}: {}", instance.id, signals.join(", ")),
                ));
            }
        }

        cards.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));

        let mut lines = vec![Line::from("Top anomaly cards:")];
        if cards.is_empty() {
            lines.push(Line::from("  none"));
            return lines;
        }
        lines.extend(
            cards
                .into_iter()
                .take(3)
                .map(|(_, _, card)| Line::from(format!("  {card}"))),
        );
        lines
    }

    fn summary_lines(&self) -> Vec<Line<'static>> {
        let Some(project) = self.selected_project() else {
            return vec![Line::from(
                "No project selected. Press Enter on a fleet row to open Project Detail.",
            )];
        };

        let fleet = self.state.fleet();
        let instances = self.project_instances();
        if instances.is_empty() {
            return vec![Line::from(format!(
                "No visible instances for project `{project}` in current filters.",
            ))];
        }

        let total_instances = instances.len();
        let unhealthy_count = instances
            .iter()
            .filter(|instance| !instance.healthy)
            .count();
        let docs_total: u64 = instances.iter().map(|instance| instance.doc_count).sum();
        let pending_total: u64 = instances.iter().map(|instance| instance.pending_jobs).sum();
        let search_total: u64 = instances
            .iter()
            .map(|instance| {
                fleet
                    .search_metrics
                    .get(&instance.id)
                    .map_or(0, |metrics| metrics.total_searches)
            })
            .sum();

        let cpu_samples: Vec<f64> = instances
            .iter()
            .filter_map(|instance| {
                fleet
                    .resources
                    .get(&instance.id)
                    .map(|metrics| metrics.cpu_percent)
            })
            .collect();
        let cpu_avg = if cpu_samples.is_empty() {
            0.0
        } else {
            let total: f64 = cpu_samples.iter().sum();
            let denom = u32::try_from(cpu_samples.len()).unwrap_or(u32::MAX);
            total / f64::from(denom)
        };

        let attribution_samples: Vec<u64> = instances
            .iter()
            .filter_map(|instance| {
                fleet
                    .attribution
                    .get(&instance.id)
                    .map(|attribution| u64::from(attribution.confidence_score))
            })
            .collect();
        let attribution_avg = if attribution_samples.is_empty() {
            0
        } else {
            let sum: u64 = attribution_samples.iter().sum();
            let denom = u64::try_from(attribution_samples.len()).unwrap_or(u64::MAX);
            u8::try_from(sum.saturating_add(denom / 2).saturating_div(denom).min(100))
                .unwrap_or(100)
        };

        let health_badge = if unhealthy_count == 0 {
            "GREEN"
        } else if unhealthy_count < total_instances {
            "YELLOW"
        } else {
            "RED"
        };

        let mut lines = vec![
            Line::from(format!(
                "{health_badge} project={project} inst={total_instances} unhealthy={unhealthy_count}"
            )),
            Line::from(format!(
                "Index docs={docs_total} | Embedding pending={pending_total} | Search total={search_total}"
            )),
            Line::from(format!(
                "Resources avg_cpu={cpu_avg:.1}% | Attribution avg={attribution_avg}%"
            )),
            Line::from(format!(
                "Control-plane={} ({})",
                self.state.control_plane_health().badge(),
                self.state.control_plane_health()
            )),
        ];
        lines.extend(Self::phase_latency_lines(&instances, fleet));
        lines.extend(Self::anomaly_lines(&instances, fleet));
        lines
    }

    fn build_rows(&self) -> Vec<Row<'static>> {
        let fleet = self.state.fleet();
        self.project_instances()
            .into_iter()
            .enumerate()
            .map(|(index, instance)| {
                let health = if instance.healthy { "OK" } else { "WARN" };
                let search_total = fleet
                    .search_metrics
                    .get(&instance.id)
                    .map_or(0, |metrics| metrics.total_searches);
                let cpu_percent = fleet.resources.get(&instance.id).map_or_else(
                    || "-".to_owned(),
                    |metrics| format!("{:.1}%", metrics.cpu_percent),
                );
                let attribution = fleet.attribution.get(&instance.id).map_or_else(
                    || "n/a".to_owned(),
                    |value| format!("{}%", value.confidence_score),
                );
                let lifecycle = fleet.lifecycle_for(&instance.id).map_or_else(
                    || "-".to_owned(),
                    |lifecycle| format!("{:?}", lifecycle.state),
                );
                let style = if index == self.selected_row {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else if !instance.healthy {
                    Style::default().fg(ratatui::style::Color::Red)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    health.to_owned(),
                    instance.id.clone(),
                    instance.doc_count.to_string(),
                    instance.pending_jobs.to_string(),
                    search_total.to_string(),
                    cpu_percent,
                    attribution,
                    lifecycle,
                ])
                .style(style)
            })
            .collect()
    }

    fn row_count(&self) -> usize {
        self.project_instances().len()
    }
}

impl Default for ProjectDetailScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for ProjectDetailScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Project Detail"
    }

    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let area = frame.area();
        let summary_lines = self.summary_lines();
        let summary_height = u16::try_from(summary_lines.len().saturating_add(2))
            .unwrap_or(14)
            .clamp(6, 14);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(summary_height),
                Constraint::Min(5),
            ])
            .split(area);

        let project = self.selected_project().unwrap_or("<none>");
        let header = Paragraph::new(Line::from(vec![
            Span::styled("Project: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(project.to_owned()),
            Span::raw(" | Esc back | s stream | t timeline | a analytics"),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Project Detail "),
        );
        frame.render_widget(header, chunks[0]);

        let summary = Paragraph::new(summary_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Summary Cards "),
        );
        frame.render_widget(summary, chunks[1]);

        let rows = self.build_rows();
        if rows.is_empty() {
            let empty = Paragraph::new(
                "No instance rows available. Select a project from Fleet Overview to populate this screen.",
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Instances "),
            );
            frame.render_widget(empty, chunks[2]);
            return;
        }

        let table = Table::new(
            rows,
            [
                Constraint::Length(6),
                Constraint::Length(16),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(12),
            ],
        )
        .header(
            Row::new(vec![
                "Health",
                "Instance",
                "Docs",
                "Pending",
                "Searches",
                "CPU",
                "Attr",
                "Lifecycle",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(Block::default().borders(Borders::ALL).title(" Instances "));
        frame.render_widget(table, chunks[2]);
    }

    fn handle_input(&mut self, event: &InputEvent, _ctx: &ScreenContext) -> ScreenAction {
        if let InputEvent::Key(key, _mods) = event {
            match key {
                crossterm::event::KeyCode::Up | crossterm::event::KeyCode::Char('k') => {
                    if self.selected_row > 0 {
                        self.selected_row -= 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Down | crossterm::event::KeyCode::Char('j') => {
                    let count = self.row_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Esc | crossterm::event::KeyCode::Char('q') => {
                    return ScreenAction::Navigate(self.fleet_screen_id.clone());
                }
                crossterm::event::KeyCode::Char('s') => {
                    return ScreenAction::Navigate(self.live_stream_screen_id.clone());
                }
                crossterm::event::KeyCode::Char('t') => {
                    return ScreenAction::Navigate(self.timeline_screen_id.clone());
                }
                crossterm::event::KeyCode::Char('a') => {
                    return ScreenAction::Navigate(self.analytics_screen_id.clone());
                }
                _ => {}
            }
        }
        ScreenAction::Ignored
    }

    fn semantic_role(&self) -> &'static str {
        "region"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        FleetSnapshot, InstanceInfo, InstanceLifecycle, LifecycleSignal,
        ProjectAttributionResolver, ResourceMetrics, SearchMetrics,
    };

    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let resolver = ProjectAttributionResolver;
        let mut fleet = FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "cass-1".to_string(),
                    project: "cass".to_string(),
                    pid: Some(101),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 2,
                },
                InstanceInfo {
                    id: "cass-2".to_string(),
                    project: "cass".to_string(),
                    pid: Some(102),
                    healthy: false,
                    doc_count: 20,
                    pending_jobs: 6,
                },
                InstanceInfo {
                    id: "xf-1".to_string(),
                    project: "xf".to_string(),
                    pid: Some(201),
                    healthy: true,
                    doc_count: 5,
                    pending_jobs: 0,
                },
            ],
            ..FleetSnapshot::default()
        };
        fleet.resources.insert(
            "cass-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 10.0,
                memory_bytes: 64 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.resources.insert(
            "cass-2".to_owned(),
            ResourceMetrics {
                cpu_percent: 30.0,
                memory_bytes: 64 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.search_metrics.insert(
            "cass-1".to_owned(),
            SearchMetrics {
                total_searches: 11,
                avg_latency_us: 900,
                p95_latency_us: 1500,
                refined_count: 2,
            },
        );
        fleet.search_metrics.insert(
            "cass-2".to_owned(),
            SearchMetrics {
                total_searches: 22,
                avg_latency_us: 1800,
                p95_latency_us: 2500,
                refined_count: 3,
            },
        );
        fleet.attribution.insert(
            "cass-1".to_owned(),
            resolver.resolve(Some("cass"), Some("cass-host"), Some("cass")),
        );
        fleet.attribution.insert(
            "cass-2".to_owned(),
            resolver.resolve(Some("cass"), Some("cass-host"), Some("cass")),
        );
        let mut lifecycle_1 = InstanceLifecycle::new(1_000);
        lifecycle_1.apply_signal(LifecycleSignal::Heartbeat, 1_500, None);
        fleet.lifecycle.insert("cass-1".to_owned(), lifecycle_1);

        state.update_fleet(fleet);
        state
    }

    #[test]
    fn project_screen_defaults() {
        let screen = ProjectDetailScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.project"));
        assert_eq!(screen.title(), "Project Detail");
        assert_eq!(screen.semantic_role(), "region");
    }

    #[test]
    fn summary_requires_project_selection() {
        let mut screen = ProjectDetailScreen::new();
        screen.update_state(&sample_state(), &ViewState::default());
        let text = screen
            .summary_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("No project selected"));
    }

    #[test]
    fn summary_rolls_up_project_metrics() {
        let mut screen = ProjectDetailScreen::new();
        let mut view = ViewState::default();
        view.set_project_filter("cass");
        screen.update_state(&sample_state(), &view);

        let summary = screen
            .summary_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(summary.contains("project=cass"));
        assert!(summary.contains("inst=2"));
        assert!(summary.contains("docs=30"));
        assert!(summary.contains("pending=8"));
        assert!(summary.contains("Search total=33"));
        assert!(summary.contains("Phase latency bars"));
        assert!(summary.contains("Top anomaly cards"));

        let rows = screen.build_rows();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn summary_includes_phase_bars_and_refinement_share() {
        let mut screen = ProjectDetailScreen::new();
        let mut view = ViewState::default();
        view.set_project_filter("cass");
        screen.update_state(&sample_state(), &view);

        let summary = screen
            .summary_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(summary.contains("initial ["));
        assert!(summary.contains("refined ["));
        assert!(summary.contains("refined_share="));
    }

    #[test]
    fn summary_anomaly_cards_surface_high_risk_instances() {
        let mut screen = ProjectDetailScreen::new();
        let mut view = ViewState::default();
        view.set_project_filter("cass");
        screen.update_state(&sample_state(), &view);

        let summary = screen
            .summary_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(summary.contains("WARN cass-2"));
        assert!(summary.contains("unhealthy"));
        assert!(summary.contains("p95=2500us"));
    }

    #[test]
    fn navigation_shortcuts_route_to_expected_screens() {
        let mut screen = ProjectDetailScreen::new();
        let mut view = ViewState::default();
        view.set_project_filter("cass");
        screen.update_state(&sample_state(), &view);
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.project"),
            terminal_width: 100,
            terminal_height: 40,
            focused: true,
        };

        let esc = InputEvent::Key(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&esc, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.fleet"))
        );

        let stream = InputEvent::Key(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&stream, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.live_stream"))
        );

        let timeline = InputEvent::Key(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&timeline, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.timeline"))
        );

        let analytics = InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&analytics, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.timeline"))
        );
    }

    #[test]
    fn analytics_shortcut_uses_configured_destination() {
        let mut screen = ProjectDetailScreen::new();
        screen.set_analytics_screen_id(ScreenId::new("ops.analytics"));
        screen.update_state(&sample_state(), &ViewState::default());
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.project"),
            terminal_width: 100,
            terminal_height: 40,
            focused: true,
        };

        let analytics = InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&analytics, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.analytics"))
        );
    }

    #[test]
    fn configured_shortcuts_use_custom_destinations() {
        let mut screen = ProjectDetailScreen::new();
        screen.set_fleet_screen_id(ScreenId::new("ops.custom_fleet"));
        screen.set_live_stream_screen_id(ScreenId::new("ops.custom_stream"));
        screen.set_timeline_screen_id(ScreenId::new("ops.custom_timeline"));
        screen.update_state(&sample_state(), &ViewState::default());
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.project"),
            terminal_width: 100,
            terminal_height: 40,
            focused: true,
        };

        let esc = InputEvent::Key(
            crossterm::event::KeyCode::Esc,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&esc, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.custom_fleet"))
        );

        let stream = InputEvent::Key(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&stream, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.custom_stream"))
        );

        let timeline = InputEvent::Key(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&timeline, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.custom_timeline"))
        );
    }
}
