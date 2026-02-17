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
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::data_source::TimeWindow;
use crate::state::{AppState, ControlPlaneHealth};

#[derive(Clone)]
struct StreamRowData {
    instance_id: String,
    correlation_id: String,
    project: String,
    host: String,
    searches: u64,
    avg_latency_us: u64,
    p95_latency_us: u64,
    refined_count: u64,
    memory_bytes: u64,
    severity: StreamSeverity,
    degradation_marker: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamSeverity {
    Info,
    Warn,
    Critical,
}

impl StreamSeverity {
    const fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Critical => "critical",
        }
    }

    const fn color(self) -> Color {
        match self {
            Self::Info => Color::Gray,
            Self::Warn => Color::Yellow,
            Self::Critical => Color::Red,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamSeverityFilter {
    All,
    Info,
    Warn,
    Critical,
}

impl StreamSeverityFilter {
    const fn next(self) -> Self {
        match self {
            Self::All => Self::Info,
            Self::Info => Self::Warn,
            Self::Warn => Self::Critical,
            Self::Critical => Self::All,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Critical => "critical",
        }
    }

    const fn allows(self, severity: StreamSeverity) -> bool {
        match self {
            Self::All => true,
            Self::Info => matches!(severity, StreamSeverity::Info),
            Self::Warn => matches!(severity, StreamSeverity::Warn),
            Self::Critical => matches!(severity, StreamSeverity::Critical),
        }
    }
}

const fn severity_rank(severity: StreamSeverity) -> u8 {
    match severity {
        StreamSeverity::Info => 0,
        StreamSeverity::Warn => 1,
        StreamSeverity::Critical => 2,
    }
}

/// Live stream screen with recent activity and stream-health status.
pub struct LiveSearchStreamScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
    project_filter_index: usize,
    host_filter_index: usize,
    severity_filter: StreamSeverityFilter,
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
            host_filter_index: 0,
            severity_filter: StreamSeverityFilter::All,
            degraded_only: false,
        }
    }

    /// Update screen data from shared application state.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        self.clamp_filter_index();
        self.clamp_selected_row();
    }

    /// Apply a project filter by value, defaulting to `all` when absent.
    pub fn set_project_filter(&mut self, project: &str) {
        self.project_filter_index = self
            .project_filters()
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(project))
            .unwrap_or(0);
        self.clamp_filter_index();
        self.clamp_selected_row();
    }

    /// Active project filter label, if not `all`.
    #[must_use]
    pub fn active_project_filter(&self) -> Option<String> {
        self.project_filters()
            .get(self.project_filter_index)
            .filter(|value| value.as_str() != "all")
            .cloned()
    }

    fn row_data(&self) -> Vec<StreamRowData> {
        let fleet = self.state.fleet();
        let project_filters = self.project_filters();
        let host_filters = self.host_filters();
        let project_filter = project_filters
            .get(self.project_filter_index)
            .map(String::as_str);
        let project_filter = project_filter.filter(|value| *value != "all");
        let host_filter = host_filters
            .get(self.host_filter_index)
            .map(String::as_str)
            .filter(|value| *value != "all");

        let mut rows: Vec<_> = fleet
            .instances
            .iter()
            .filter(|instance| {
                project_filter.is_none_or(|project| instance.project.eq_ignore_ascii_case(project))
            })
            .map(|instance| {
                let metrics = fleet.search_metrics.get(&instance.id);
                let resources = fleet.resources.get(&instance.id);
                let searches = metrics.map_or(0, |value| value.total_searches);
                let avg_latency_us = metrics.map_or(0, |value| value.avg_latency_us);
                let p95_latency_us = metrics.map_or(0, |value| value.p95_latency_us);
                let refined_count = metrics.map_or(0, |value| value.refined_count);
                let memory_bytes = resources.map_or(0, |value| value.memory_bytes);
                let host = Self::host_bucket(&instance.id);
                let mut markers: Vec<&'static str> = Vec::new();
                if !instance.healthy {
                    markers.push("health");
                }
                if p95_latency_us >= 5_000 {
                    markers.push("latency_p95");
                }
                if avg_latency_us >= 2_000 {
                    markers.push("latency_avg");
                }
                if memory_bytes >= 512 * 1024 * 1024 {
                    markers.push("memory");
                }
                let severity = if !instance.healthy
                    || p95_latency_us >= 10_000
                    || memory_bytes >= 1024 * 1024 * 1024
                {
                    StreamSeverity::Critical
                } else if !markers.is_empty() {
                    StreamSeverity::Warn
                } else {
                    StreamSeverity::Info
                };
                if markers.is_empty() {
                    markers.push("nominal");
                }
                StreamRowData {
                    instance_id: instance.id.clone(),
                    correlation_id: Self::correlation_id(
                        &instance.id,
                        searches,
                        avg_latency_us,
                        p95_latency_us,
                        memory_bytes,
                    ),
                    project: instance.project.clone(),
                    host,
                    searches,
                    avg_latency_us,
                    p95_latency_us,
                    refined_count,
                    memory_bytes,
                    severity,
                    degradation_marker: markers.join("+"),
                }
            })
            .filter(|row| host_filter.is_none_or(|host| row.host.eq_ignore_ascii_case(host)))
            .filter(|row| self.severity_filter.allows(row.severity))
            .filter(|row| !self.degraded_only || !matches!(row.severity, StreamSeverity::Info))
            .collect();

        rows.sort_by(|left, right| {
            severity_rank(right.severity)
                .cmp(&severity_rank(left.severity))
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

    fn host_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let hosts: BTreeSet<_> = self
            .state
            .fleet()
            .instances
            .iter()
            .map(|instance| Self::host_bucket(&instance.id))
            .collect();
        values.extend(hosts);
        values
    }

    fn clamp_filter_index(&mut self) {
        let project_max = self.project_filters().len().saturating_sub(1);
        let host_max = self.host_filters().len().saturating_sub(1);
        if self.project_filter_index > project_max {
            self.project_filter_index = project_max;
        }
        if self.host_filter_index > host_max {
            self.host_filter_index = host_max;
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

    fn cycle_host_filter(&mut self) {
        let len = self.host_filters().len();
        if len == 0 {
            self.host_filter_index = 0;
            return;
        }
        self.host_filter_index = (self.host_filter_index + 1) % len;
        self.clamp_selected_row();
    }

    fn cycle_severity_filter(&mut self) {
        self.severity_filter = self.severity_filter.next();
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
                if index != self.selected_row {
                    style = style.fg(row.severity.color());
                }
                Row::new(vec![
                    row.instance_id,
                    row.correlation_id,
                    row.project,
                    row.host,
                    row.searches.to_string(),
                    row.avg_latency_us.to_string(),
                    row.p95_latency_us.to_string(),
                    mem_mib.to_string(),
                    refined_rate,
                    row.severity.label().to_owned(),
                    row.degradation_marker,
                ])
                .style(style)
            })
            .collect()
    }

    fn stream_health_summary(&self) -> String {
        let metrics = self.state.control_plane_metrics();
        let health = self.state.control_plane_health();
        let lag_state = if metrics.ingestion_lag_events >= 10_000 {
            "crit"
        } else if metrics.ingestion_lag_events >= 1_000 {
            "warn"
        } else {
            "ok"
        };
        let drop_state = if metrics.dead_letter_events >= 20 {
            "crit"
        } else if metrics.dead_letter_events > 0 {
            "warn"
        } else {
            "ok"
        };
        let reconnect_state = match (self.state.has_data(), health) {
            (false, _) => "reconnecting",
            (true, ControlPlaneHealth::Critical) => "backoff",
            (true, ControlPlaneHealth::Degraded) => "degraded",
            (true, ControlPlaneHealth::Healthy) => "steady",
        };
        format!(
            "health={health} | reconnect={reconnect_state} | lag={}({lag_state}) | drops={}({drop_state}) | throughput={:.2} eps",
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
        let host = self
            .host_filters()
            .get(self.host_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned());
        format!(
            "filters: project={project}, host={host}, severity={}, degraded_only={} | keys: p/h/s/d/x",
            self.severity_filter.label(),
            self.degraded_only
        )
    }

    fn instance_count(&self) -> usize {
        self.row_data().len()
    }

    fn selected_context_summary(&self) -> String {
        if let Some(selected) = self.row_data().get(self.selected_row) {
            let mem_mib = selected.memory_bytes / (1024 * 1024);
            return format!(
                "focus: {} corr={} host={} sev={} marker={} p95={}us mem={}MiB",
                selected.instance_id,
                selected.correlation_id,
                selected.host,
                selected.severity.label(),
                selected.degradation_marker,
                selected.p95_latency_us,
                mem_mib
            );
        }
        "focus: none".to_owned()
    }

    fn host_bucket(instance_id: &str) -> String {
        if let Some((host, _)) = instance_id.split_once(':') {
            return host.to_owned();
        }
        if let Some((host, _)) = instance_id.split_once('-') {
            return host.to_owned();
        }
        instance_id.to_owned()
    }

    fn correlation_id(
        instance_id: &str,
        searches: u64,
        avg_latency_us: u64,
        p95_latency_us: u64,
        memory_bytes: u64,
    ) -> String {
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut digest: u64 = 1_469_598_103_934_665_603;
        for byte in instance_id.bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(FNV_PRIME);
        }
        digest ^= searches;
        digest = digest.wrapping_mul(FNV_PRIME);
        digest ^= avg_latency_us;
        digest = digest.wrapping_mul(FNV_PRIME);
        digest ^= p95_latency_us;
        digest = digest.wrapping_mul(FNV_PRIME);
        digest ^= memory_bytes;
        format!("{digest:016x}")
    }

    #[cfg(test)]
    fn selected_project_filter(&self) -> String {
        self.project_filters()
            .get(self.project_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned())
    }

    #[cfg(test)]
    fn selected_host_filter(&self) -> String {
        self.host_filters()
            .get(self.host_filter_index)
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
            .constraints([Constraint::Length(6), Constraint::Min(5)])
            .split(area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Stream: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(self.stream_health_summary()),
            ]),
            Line::from(self.rolling_counter_summary()),
            Line::from(self.filter_summary()),
            Line::from(self.selected_context_summary()),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Live Search Stream "),
        );
        frame.render_widget(header, chunks[0]);

        let table = Table::new(
            self.build_rows(),
            [
                Constraint::Length(14),
                Constraint::Length(16),
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(12),
                Constraint::Length(9),
                Constraint::Min(10),
            ],
        )
        .header(
            Row::new(vec![
                "Instance", "Corr ID", "Project", "Host", "Searches", "Avg(us)", "P95(us)",
                "Mem(MiB)", "Refined", "Severity", "Marker",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Activity "),
        );
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
                crossterm::event::KeyCode::Char('h') => {
                    self.cycle_host_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('s') => {
                    self.cycle_severity_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('d') => {
                    self.degraded_only = !self.degraded_only;
                    self.clamp_selected_row();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('x') => {
                    self.project_filter_index = 0;
                    self.host_filter_index = 0;
                    self.severity_filter = StreamSeverityFilter::All;
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
        assert_eq!(screen.selected_host_filter(), "all");

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
        assert_eq!(screen.selected_host_filter(), "all");
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
        assert!(!matches!(rows[0].severity, StreamSeverity::Info));
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

    #[test]
    fn live_stream_host_and_severity_filters_apply() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let host = InputEvent::Key(
            crossterm::event::KeyCode::Char('h'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&host, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_host_filter(), "inst");
        assert_eq!(screen.row_data().len(), 2);

        let severity = InputEvent::Key(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&severity, &ctx), ScreenAction::Consumed);
        let info_rows = screen.row_data();
        assert_eq!(info_rows.len(), 1);
        assert_eq!(info_rows[0].instance_id, "inst-a");
        assert_eq!(info_rows[0].severity.label(), "info");
    }

    #[test]
    fn live_stream_context_summary_contains_correlation_marker() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let summary = screen.selected_context_summary();
        assert!(summary.contains("focus: "));
        assert!(summary.contains("corr="));
        assert!(summary.contains("marker="));
    }

    // --- StreamSeverity ---

    #[test]
    fn severity_label_covers_all_variants() {
        assert_eq!(StreamSeverity::Info.label(), "info");
        assert_eq!(StreamSeverity::Warn.label(), "warn");
        assert_eq!(StreamSeverity::Critical.label(), "critical");
    }

    #[test]
    fn severity_color_distinct_per_variant() {
        let info_color = StreamSeverity::Info.color();
        let warn_color = StreamSeverity::Warn.color();
        let crit_color = StreamSeverity::Critical.color();
        assert_ne!(info_color, warn_color);
        assert_ne!(warn_color, crit_color);
        assert_ne!(info_color, crit_color);
    }

    // --- StreamSeverityFilter ---

    #[test]
    fn severity_filter_cycles_through_all_states() {
        let start = StreamSeverityFilter::All;
        let step1 = start.next();
        let step2 = step1.next();
        let step3 = step2.next();
        let step4 = step3.next();
        assert_eq!(step1, StreamSeverityFilter::Info);
        assert_eq!(step2, StreamSeverityFilter::Warn);
        assert_eq!(step3, StreamSeverityFilter::Critical);
        assert_eq!(step4, StreamSeverityFilter::All);
    }

    #[test]
    fn severity_filter_labels_cover_all_variants() {
        assert_eq!(StreamSeverityFilter::All.label(), "all");
        assert_eq!(StreamSeverityFilter::Info.label(), "info");
        assert_eq!(StreamSeverityFilter::Warn.label(), "warn");
        assert_eq!(StreamSeverityFilter::Critical.label(), "critical");
    }

    #[test]
    fn severity_filter_all_allows_every_severity() {
        assert!(StreamSeverityFilter::All.allows(StreamSeverity::Info));
        assert!(StreamSeverityFilter::All.allows(StreamSeverity::Warn));
        assert!(StreamSeverityFilter::All.allows(StreamSeverity::Critical));
    }

    #[test]
    fn severity_filter_info_allows_only_info() {
        assert!(StreamSeverityFilter::Info.allows(StreamSeverity::Info));
        assert!(!StreamSeverityFilter::Info.allows(StreamSeverity::Warn));
        assert!(!StreamSeverityFilter::Info.allows(StreamSeverity::Critical));
    }

    #[test]
    fn severity_filter_warn_allows_only_warn() {
        assert!(!StreamSeverityFilter::Warn.allows(StreamSeverity::Info));
        assert!(StreamSeverityFilter::Warn.allows(StreamSeverity::Warn));
        assert!(!StreamSeverityFilter::Warn.allows(StreamSeverity::Critical));
    }

    #[test]
    fn severity_filter_critical_allows_only_critical() {
        assert!(!StreamSeverityFilter::Critical.allows(StreamSeverity::Info));
        assert!(!StreamSeverityFilter::Critical.allows(StreamSeverity::Warn));
        assert!(StreamSeverityFilter::Critical.allows(StreamSeverity::Critical));
    }

    // --- severity_rank ---

    #[test]
    fn severity_rank_ordering() {
        assert!(severity_rank(StreamSeverity::Info) < severity_rank(StreamSeverity::Warn));
        assert!(severity_rank(StreamSeverity::Warn) < severity_rank(StreamSeverity::Critical));
    }

    // --- host_bucket ---

    #[test]
    fn host_bucket_splits_on_colon() {
        assert_eq!(LiveSearchStreamScreen::host_bucket("host:instance"), "host");
    }

    #[test]
    fn host_bucket_splits_on_dash_when_no_colon() {
        assert_eq!(LiveSearchStreamScreen::host_bucket("host-instance"), "host");
    }

    #[test]
    fn host_bucket_returns_full_id_when_no_separator() {
        assert_eq!(
            LiveSearchStreamScreen::host_bucket("singleword"),
            "singleword"
        );
    }

    #[test]
    fn host_bucket_prefers_colon_over_dash() {
        assert_eq!(
            LiveSearchStreamScreen::host_bucket("host:part-suffix"),
            "host"
        );
    }

    #[test]
    fn host_bucket_empty_string() {
        assert_eq!(LiveSearchStreamScreen::host_bucket(""), "");
    }

    // --- correlation_id ---

    #[test]
    fn correlation_id_is_deterministic() {
        let id1 = LiveSearchStreamScreen::correlation_id("inst-a", 10, 1000, 2000, 4096);
        let id2 = LiveSearchStreamScreen::correlation_id("inst-a", 10, 1000, 2000, 4096);
        assert_eq!(id1, id2);
    }

    #[test]
    fn correlation_id_differs_for_different_inputs() {
        let id1 = LiveSearchStreamScreen::correlation_id("inst-a", 10, 1000, 2000, 4096);
        let id2 = LiveSearchStreamScreen::correlation_id("inst-b", 10, 1000, 2000, 4096);
        assert_ne!(id1, id2);
    }

    #[test]
    fn correlation_id_is_hex_formatted() {
        let id = LiveSearchStreamScreen::correlation_id("test", 0, 0, 0, 0);
        assert_eq!(id.len(), 16, "correlation id should be 16 hex chars");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "should be valid hex: {id}"
        );
    }

    // --- set_project_filter / active_project_filter ---

    #[test]
    fn set_project_filter_by_name() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        screen.set_project_filter("proj-a");
        assert_eq!(screen.active_project_filter(), Some("proj-a".to_owned()));
    }

    #[test]
    fn set_project_filter_case_insensitive() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        screen.set_project_filter("PROJ-A");
        assert_eq!(screen.active_project_filter(), Some("proj-a".to_owned()));
    }

    #[test]
    fn set_project_filter_unknown_defaults_to_all() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        screen.set_project_filter("nonexistent");
        assert_eq!(screen.active_project_filter(), None);
    }

    #[test]
    fn active_project_filter_none_when_all() {
        let screen = LiveSearchStreamScreen::new();
        assert_eq!(screen.active_project_filter(), None);
    }

    // --- empty state ---

    #[test]
    fn empty_state_has_no_rows() {
        let screen = LiveSearchStreamScreen::new();
        assert_eq!(screen.row_data().len(), 0);
    }

    #[test]
    fn empty_state_context_summary_says_none() {
        let screen = LiveSearchStreamScreen::new();
        assert_eq!(screen.selected_context_summary(), "focus: none");
    }

    // --- stream_health_summary ---

    #[test]
    fn stream_health_summary_contains_throughput() {
        let mut screen = LiveSearchStreamScreen::new();
        screen.update_state(&sample_state());
        let summary = screen.stream_health_summary();
        assert!(summary.contains("throughput="));
        assert!(summary.contains("eps"));
    }

    // --- Default trait ---

    #[test]
    fn default_matches_new() {
        let screen = LiveSearchStreamScreen::default();
        assert_eq!(screen.id(), &ScreenId::new("ops.live_stream"));
        assert_eq!(screen.selected_row, 0);
    }
}
