//! Live Search Stream screen scaffold.
//!
//! Presents high-signal per-instance search activity and stream health
//! indicators from the latest fleet snapshot.

use std::any::Any;
use std::collections::BTreeSet;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::data_source::TimeWindow;
use crate::state::{AppState, ControlPlaneHealth};

#[derive(Clone)]
struct StreamRowData {
    instance_id: String,
    project: String,
    searches: u64,
    avg_latency_us: u64,
    p95_latency_us: u64,
    refined_count: u64,
    memory_bytes: u64,
    degraded: bool,
}

/// Live stream screen with recent activity and stream-health status.
pub struct LiveSearchStreamScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
    project_filter_index: usize,
    degraded_only: bool,
}

impl LiveSearchStreamScreen {
    /// Create a new live stream screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.live_stream"),
            state: AppState::new(),
            selected_row: 0,
            project_filter_index: 0,
            degraded_only: false,
        }
    }

    /// Update screen data from shared application state.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        self.clamp_filter_index();
        self.clamp_selected_row();
    }

    fn row_data(&self) -> Vec<StreamRowData> {
        let fleet = self.state.fleet();
        let project_filters = self.project_filters();
        let project_filter = project_filters
            .get(self.project_filter_index)
            .map(String::as_str);
        let project_filter = project_filter.filter(|value| *value != "all");

        let mut rows: Vec<_> = fleet
            .instances
            .iter()
            .filter(|instance| {
                project_filter.is_none_or(|project| instance.project.eq_ignore_ascii_case(project))
            })
            .map(|instance| {
                let metrics = fleet.search_metrics.get(&instance.id);
                let resources = fleet.resources.get(&instance.id);
                let degraded = !instance.healthy
                    || metrics.is_none_or(|value| {
                        value.p95_latency_us >= 5_000 || value.avg_latency_us >= 2_000
                    })
                    || resources.is_some_and(|value| value.memory_bytes >= 512 * 1024 * 1024);
                StreamRowData {
                    instance_id: instance.id.clone(),
                    project: instance.project.clone(),
                    searches: metrics.map_or(0, |value| value.total_searches),
                    avg_latency_us: metrics.map_or(0, |value| value.avg_latency_us),
                    p95_latency_us: metrics.map_or(0, |value| value.p95_latency_us),
                    refined_count: metrics.map_or(0, |value| value.refined_count),
                    memory_bytes: resources.map_or(0, |value| value.memory_bytes),
                    degraded,
                }
            })
            .filter(|row| !self.degraded_only || row.degraded)
            .collect();

        rows.sort_by(|left, right| {
            right
                .degraded
                .cmp(&left.degraded)
                .then_with(|| right.searches.cmp(&left.searches))
                .then_with(|| right.p95_latency_us.cmp(&left.p95_latency_us))
                .then_with(|| right.memory_bytes.cmp(&left.memory_bytes))
                .then_with(|| left.project.cmp(&right.project))
                .then_with(|| left.instance_id.cmp(&right.instance_id))
        });
        rows
    }

    fn project_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let projects: BTreeSet<_> = self
            .state
            .fleet()
            .instances
            .iter()
            .map(|instance| instance.project.clone())
            .collect();
        values.extend(projects);
        values
    }

    fn clamp_filter_index(&mut self) {
        let max_index = self.project_filters().len().saturating_sub(1);
        if self.project_filter_index > max_index {
            self.project_filter_index = max_index;
        }
    }

    fn clamp_selected_row(&mut self) {
        let count = self.row_data().len();
        if count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= count {
            self.selected_row = count.saturating_sub(1);
        }
    }

    fn cycle_project_filter(&mut self) {
        let len = self.project_filters().len();
        if len == 0 {
            self.project_filter_index = 0;
            return;
        }
        self.project_filter_index = (self.project_filter_index + 1) % len;
        self.clamp_selected_row();
    }

    fn build_rows(&self) -> Vec<Row<'static>> {
        self.row_data()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let refined_rate = if row.searches == 0 {
                    "0.0%".to_owned()
                } else {
                    let scaled = row
                        .refined_count
                        .saturating_mul(1000)
                        .saturating_div(row.searches);
                    let whole = scaled / 10;
                    let frac = scaled % 10;
                    format!("{whole}.{frac}%")
                };
                let mem_mib = row.memory_bytes / (1024 * 1024);
                let mut style = if index == self.selected_row {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                if row.degraded && index != self.selected_row {
                    style = style.fg(Color::Yellow);
                }
                let state = if row.degraded { "WARN" } else { "OK" };
                Row::new(vec![
                    row.instance_id,
                    row.project,
                    row.searches.to_string(),
                    row.avg_latency_us.to_string(),
                    row.p95_latency_us.to_string(),
                    mem_mib.to_string(),
                    refined_rate,
                    state.to_owned(),
                ])
                .style(style)
            })
            .collect()
    }

    fn stream_health_summary(&self) -> String {
        let metrics = self.state.control_plane_metrics();
        let health = self.state.control_plane_health();
        let reconnect_state = match (self.state.has_data(), health) {
            (false, _) => "reconnecting",
            (true, ControlPlaneHealth::Critical) => "backoff",
            (true, ControlPlaneHealth::Degraded) => "degraded",
            (true, ControlPlaneHealth::Healthy) => "steady",
        };
        format!(
            "health={health} | reconnect={reconnect_state} | lag={} | drops={} | throughput={:.2} eps",
            metrics.ingestion_lag_events, metrics.dead_letter_events, metrics.event_throughput_eps
        )
    }

    fn rolling_counter_summary(&self) -> String {
        let throughput = self
            .state
            .control_plane_metrics()
            .event_throughput_eps
            .max(0.0);
        let estimates = TimeWindow::ALL
            .iter()
            .map(|window| {
                let seconds = u32::try_from(window.seconds()).unwrap_or(u32::MAX);
                let count = throughput * f64::from(seconds);
                format!("{}={count:.0}", window.label())
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("events: {estimates}")
    }

    fn filter_summary(&self) -> String {
        let project = self
            .project_filters()
            .get(self.project_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned());
        format!(
            "filters: project={project}, degraded_only={} | keys: p=project d=degraded x=clear",
            self.degraded_only
        )
    }

    fn instance_count(&self) -> usize {
        self.row_data().len()
    }

    #[cfg(test)]
    fn selected_project_filter(&self) -> String {
        self.project_filters()
            .get(self.project_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned())
    }
}

impl Default for LiveSearchStreamScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for LiveSearchStreamScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Live Search Stream"
    }

    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(5)])
            .split(area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Stream: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(self.stream_health_summary()),
            ]),
            Line::from(self.rolling_counter_summary()),
            Line::from(self.filter_summary()),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Live Search Stream "),
        );
        frame.render_widget(header, chunks[0]);

        let table = Table::new(
            self.build_rows(),
            [
                Constraint::Length(16),
                Constraint::Length(18),
                Constraint::Length(9),
                Constraint::Length(11),
                Constraint::Length(11),
                Constraint::Length(10),
                Constraint::Length(12),
                Constraint::Length(7),
            ],
        )
        .header(
            Row::new(vec![
                "Instance", "Project", "Searches", "Avg(us)", "P95(us)", "Mem(MiB)", "Refined",
                "State",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(Block::default().borders(Borders::ALL).title(" Activity "));
        frame.render_widget(table, chunks[1]);
    }

    fn handle_input(&mut self, event: &InputEvent, _ctx: &ScreenContext) -> ScreenAction {
        if let InputEvent::Key(code, _mods) = event {
            match code {
                crossterm::event::KeyCode::Up | crossterm::event::KeyCode::Char('k') => {
                    if self.selected_row > 0 {
                        self.selected_row -= 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Down | crossterm::event::KeyCode::Char('j') => {
                    let count = self.instance_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('p') => {
                    self.cycle_project_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('d') => {
                    self.degraded_only = !self.degraded_only;
                    self.clamp_selected_row();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('x') => {
                    self.project_filter_index = 0;
                    self.degraded_only = false;
                    self.clamp_selected_row();
                    return ScreenAction::Consumed;
                }
                _ => {}
            }
        }
        ScreenAction::Ignored
    }

    fn semantic_role(&self) -> &'static str {
        "log"
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
        ControlPlaneMetrics, FleetSnapshot, InstanceInfo, ResourceMetrics, SearchMetrics,
    };

    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "inst-a".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(10),
                    healthy: true,
                    doc_count: 42,
                    pending_jobs: 0,
                },
                InstanceInfo {
                    id: "inst-b".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(11),
                    healthy: false,
                    doc_count: 64,
                    pending_jobs: 3,
                },
            ],
            ..FleetSnapshot::default()
        };
        fleet.search_metrics.insert(
            "inst-a".to_owned(),
            SearchMetrics {
                total_searches: 10,
                avg_latency_us: 1000,
                p95_latency_us: 2000,
                refined_count: 4,
            },
        );
        fleet.search_metrics.insert(
            "inst-b".to_owned(),
            SearchMetrics {
                total_searches: 20,
                avg_latency_us: 5000,
                p95_latency_us: 9000,
                refined_count: 10,
            },
        );
        fleet.resources.insert(
            "inst-a".to_owned(),
            ResourceMetrics {
                cpu_percent: 10.0,
                memory_bytes: 128 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.resources.insert(
            "inst-b".to_owned(),
            ResourceMetrics {
                cpu_percent: 90.0,
                memory_bytes: 768 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        state.update_fleet(fleet);
        state.update_control_plane(ControlPlaneMetrics {
            event_throughput_eps: 2.5,
            ..ControlPlaneMetrics::default()
        });
        state
    }

    fn screen_context() -> ScreenContext {
        ScreenContext {
            active_screen: ScreenId::new("ops.live_stream"),
            terminal_width: 120,
            terminal_height: 40,
            focused: true,
        }
    }

    #[test]
    fn live_stream_screen_defaults() {
        let screen = LiveSearchStreamScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.live_stream"));
        assert_eq!(screen.title(), "Live Search Stream");
        assert_eq!(screen.semantic_role(), "log");
    }

    #[test]
    fn live_stream_builds_rows_sorted_by_search_volume() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let data = screen.row_data();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].instance_id, "inst-b");
        assert_eq!(data[1].instance_id, "inst-a");
    }

    #[test]
    fn live_stream_navigation_is_bounded() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let down = InputEvent::Key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);

        let up = InputEvent::Key(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&up, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 0);
    }

    #[test]
    fn live_stream_filters_cycle_and_reset() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        assert_eq!(screen.row_data().len(), 2);
        assert_eq!(screen.selected_project_filter(), "all");

        let project = InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&project, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_project_filter(), "proj-a");
        assert_eq!(screen.row_data().len(), 1);

        let reset = InputEvent::Key(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&reset, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_project_filter(), "all");
        assert_eq!(screen.row_data().len(), 2);
    }

    #[test]
    fn live_stream_degraded_filter_keeps_only_warn_rows() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let degraded = InputEvent::Key(
            crossterm::event::KeyCode::Char('d'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&degraded, &ctx), ScreenAction::Consumed);
        let rows = screen.row_data();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].instance_id, "inst-b");
        assert!(rows[0].degraded);
    }

    #[test]
    fn live_stream_rolling_summary_contains_all_windows() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let summary = screen.rolling_counter_summary();
        assert!(summary.contains("1m="));
        assert!(summary.contains("15m="));
        assert!(summary.contains("1h="));
        assert!(summary.contains("6h="));
        assert!(summary.contains("24h="));
        assert!(summary.contains("3d="));
        assert!(summary.contains("1w="));
    }
}
