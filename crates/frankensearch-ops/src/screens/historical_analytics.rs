//! Historical analytics + explainability cockpit screen.
//!
//! Enables postmortem workflows with trend snapshots, anomaly/event correlation,
//! evidence-log exploration, and export-friendly incident review handles.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_core::LifecycleState;
use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::data_source::TimeWindow;
use crate::state::{AppState, LifecycleEvent};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EventSeverity {
    Info,
    Warn,
    Critical,
}

impl EventSeverity {
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

    const fn rank(self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Warn => 1,
            Self::Critical => 2,
        }
    }
}

#[derive(Clone, Debug)]
struct EvidenceRow {
    ts_ms: u64,
    project: String,
    host: String,
    instance_id: String,
    severity: EventSeverity,
    reason_code: String,
    confidence: u8,
    replay_handle: String,
}

#[derive(Clone, Debug)]
struct CorrelationRow {
    reason_code: String,
    project_span: usize,
    event_count: usize,
    critical_count: usize,
    avg_confidence: u8,
    stream_correlation: u8,
}

#[derive(Default)]
struct CorrelationAccumulator {
    event_count: usize,
    critical_count: usize,
    confidence_sum: u64,
    projects: BTreeSet<String>,
}

/// Historical analytics and explainability cockpit screen.
pub struct HistoricalAnalyticsScreen {
    id: ScreenId,
    state: AppState,
    project_lookup: BTreeMap<String, String>,
    evidence_rows: Vec<EvidenceRow>,
    selected_row: usize,
    project_filter_index: usize,
    reason_filter_index: usize,
    host_filter_index: usize,
    compact_export: bool,
    project_screen_id: ScreenId,
    live_stream_screen_id: ScreenId,
    timeline_screen_id: ScreenId,
}

impl HistoricalAnalyticsScreen {
    /// Create a new historical analytics cockpit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.analytics"),
            state: AppState::new(),
            project_lookup: BTreeMap::new(),
            evidence_rows: Vec::new(),
            selected_row: 0,
            project_filter_index: 0,
            reason_filter_index: 0,
            host_filter_index: 0,
            compact_export: true,
            project_screen_id: ScreenId::new("ops.project"),
            live_stream_screen_id: ScreenId::new("ops.live_stream"),
            timeline_screen_id: ScreenId::new("ops.timeline"),
        }
    }

    /// Override the project drilldown destination.
    pub fn set_project_screen_id(&mut self, id: ScreenId) {
        self.project_screen_id = id;
    }

    /// Override the live-stream drilldown destination.
    pub fn set_live_stream_screen_id(&mut self, id: ScreenId) {
        self.live_stream_screen_id = id;
    }

    /// Override the timeline drilldown destination.
    pub fn set_timeline_screen_id(&mut self, id: ScreenId) {
        self.timeline_screen_id = id;
    }

    /// Update state from the shared app snapshot.
    pub fn update_state(&mut self, state: &AppState) {
        let focused = self.selected_evidence_key();
        self.state = state.clone();
        self.rebuild_derived_rows();
        self.clamp_filter_indices();
        self.restore_selected_row(focused);
    }

    /// Selected project from the evidence table.
    #[must_use]
    pub fn selected_project(&self) -> Option<String> {
        self.filtered_evidence_rows()
            .get(self.selected_row)
            .map(|row| row.project.clone())
    }

    fn rebuild_derived_rows(&mut self) {
        let project_lookup = self
            .state
            .fleet()
            .instances
            .iter()
            .map(|instance| (instance.id.clone(), instance.project.clone()))
            .collect::<BTreeMap<_, _>>();

        let mut rows = self
            .state
            .fleet()
            .lifecycle_events
            .iter()
            .map(|event| {
                let project = project_lookup
                    .get(&event.instance_id)
                    .map_or("unknown", String::as_str)
                    .to_owned();
                let host = Self::host_bucket(&event.instance_id);
                let severity = Self::event_severity(event);
                let replay_handle = format!(
                    "replay://{}/{}/{}/{}",
                    project, host, event.instance_id, event.at_ms
                );

                EvidenceRow {
                    ts_ms: event.at_ms,
                    project,
                    host,
                    instance_id: event.instance_id.clone(),
                    severity,
                    reason_code: event.reason_code.clone(),
                    confidence: event.attribution_confidence_score,
                    replay_handle,
                }
            })
            .collect::<Vec<_>>();

        rows.sort_by(|left, right| {
            right
                .severity
                .rank()
                .cmp(&left.severity.rank())
                .then_with(|| right.ts_ms.cmp(&left.ts_ms))
                .then_with(|| left.project.cmp(&right.project))
                .then_with(|| left.instance_id.cmp(&right.instance_id))
        });
        self.project_lookup = project_lookup;
        self.evidence_rows = rows;
    }

    fn all_evidence_rows(&self) -> &[EvidenceRow] {
        &self.evidence_rows
    }

    fn filtered_evidence_rows(&self) -> Vec<EvidenceRow> {
        let project_filter = self
            .project_filters()
            .get(self.project_filter_index)
            .cloned()
            .filter(|value| value != "all");
        let reason_filter = self
            .reason_filters()
            .get(self.reason_filter_index)
            .cloned()
            .filter(|value| value != "all");
        let host_filter = self
            .host_filters()
            .get(self.host_filter_index)
            .cloned()
            .filter(|value| value != "all");

        self.all_evidence_rows()
            .iter()
            .filter(|row| {
                project_filter
                    .as_deref()
                    .is_none_or(|project| row.project.eq_ignore_ascii_case(project))
            })
            .filter(|row| {
                reason_filter
                    .as_deref()
                    .is_none_or(|reason| row.reason_code.eq_ignore_ascii_case(reason))
            })
            .filter(|row| {
                host_filter
                    .as_deref()
                    .is_none_or(|host| row.host.eq_ignore_ascii_case(host))
            })
            .cloned()
            .collect()
    }

    fn project_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let projects: BTreeSet<_> = self
            .all_evidence_rows()
            .iter()
            .map(|row| row.project.clone())
            .collect();
        values.extend(projects);
        values
    }

    fn reason_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let reasons: BTreeSet<_> = self
            .all_evidence_rows()
            .iter()
            .map(|row| row.reason_code.clone())
            .collect();
        values.extend(reasons);
        values
    }

    fn host_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let hosts: BTreeSet<_> = self
            .all_evidence_rows()
            .iter()
            .map(|row| row.host.clone())
            .collect();
        values.extend(hosts);
        values
    }

    fn selected_evidence_key(&self) -> Option<(u64, String, String)> {
        self.filtered_evidence_rows()
            .get(self.selected_row)
            .map(|row| (row.ts_ms, row.instance_id.clone(), row.reason_code.clone()))
    }

    fn restore_selected_row(&mut self, key: Option<(u64, String, String)>) {
        let rows = self.filtered_evidence_rows();
        if rows.is_empty() {
            self.selected_row = 0;
            return;
        }

        if let Some((ts_ms, instance_id, reason_code)) = key
            && let Some(index) = rows.iter().position(|row| {
                row.ts_ms == ts_ms
                    && row.instance_id == instance_id
                    && row.reason_code == reason_code
            })
        {
            self.selected_row = index;
            return;
        }

        if self.selected_row >= rows.len() {
            self.selected_row = rows.len().saturating_sub(1);
        }
    }

    fn clamp_filter_indices(&mut self) {
        let project_max = self.project_filters().len().saturating_sub(1);
        let reason_max = self.reason_filters().len().saturating_sub(1);
        let host_max = self.host_filters().len().saturating_sub(1);
        if self.project_filter_index > project_max {
            self.project_filter_index = project_max;
        }
        if self.reason_filter_index > reason_max {
            self.reason_filter_index = reason_max;
        }
        if self.host_filter_index > host_max {
            self.host_filter_index = host_max;
        }
    }

    fn cycle_project_filter(&mut self) {
        let focused = self.selected_evidence_key();
        let len = self.project_filters().len();
        if len > 0 {
            self.project_filter_index = (self.project_filter_index + 1) % len;
        }
        self.restore_selected_row(focused);
    }

    fn cycle_reason_filter(&mut self) {
        let focused = self.selected_evidence_key();
        let len = self.reason_filters().len();
        if len > 0 {
            self.reason_filter_index = (self.reason_filter_index + 1) % len;
        }
        self.restore_selected_row(focused);
    }

    fn cycle_host_filter(&mut self) {
        let focused = self.selected_evidence_key();
        let len = self.host_filters().len();
        if len > 0 {
            self.host_filter_index = (self.host_filter_index + 1) % len;
        }
        self.restore_selected_row(focused);
    }

    fn reset_filters(&mut self) {
        let focused = self.selected_evidence_key();
        self.project_filter_index = 0;
        self.reason_filter_index = 0;
        self.host_filter_index = 0;
        self.restore_selected_row(focused);
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn correlation_rows(&self) -> Vec<CorrelationRow> {
        let rows = self.filtered_evidence_rows();
        self.correlation_rows_for_rows(&rows)
    }

    fn correlation_rows_for_rows(&self, rows: &[EvidenceRow]) -> Vec<CorrelationRow> {
        if rows.is_empty() {
            return Vec::new();
        }

        let mut grouped: BTreeMap<String, CorrelationAccumulator> = BTreeMap::new();
        for row in rows {
            let entry = grouped.entry(row.reason_code.clone()).or_default();
            entry.event_count = entry.event_count.saturating_add(1);
            if matches!(row.severity, EventSeverity::Critical) {
                entry.critical_count = entry.critical_count.saturating_add(1);
            }
            entry.confidence_sum = entry
                .confidence_sum
                .saturating_add(u64::from(row.confidence));
            entry.projects.insert(row.project.clone());
        }

        let metrics = self.state.control_plane_metrics();
        let lag_pressure =
            u8::try_from((metrics.ingestion_lag_events / 100).min(100)).unwrap_or(100);
        let drop_pressure = u8::try_from(metrics.dead_letter_events.min(100)).unwrap_or(100);

        let total_events = rows.len().max(1);
        let mut result = grouped
            .into_iter()
            .map(|(reason_code, acc)| {
                let count_u64 = u64::try_from(acc.event_count).unwrap_or(1);
                let avg_confidence_u64 = acc
                    .confidence_sum
                    .saturating_add(count_u64 / 2)
                    .saturating_div(count_u64)
                    .min(100);
                let avg_confidence = u8::try_from(avg_confidence_u64).unwrap_or(100);

                let event_ratio = acc
                    .event_count
                    .saturating_mul(100)
                    .saturating_div(total_events)
                    .min(100);
                let critical_boost = acc.critical_count.saturating_mul(20).min(100);
                let pressure = usize::from(lag_pressure)
                    .saturating_add(usize::from(drop_pressure))
                    .saturating_div(4);
                let score = event_ratio
                    .saturating_add(critical_boost)
                    .saturating_add(pressure)
                    .min(100);

                CorrelationRow {
                    reason_code,
                    project_span: acc.projects.len(),
                    event_count: acc.event_count,
                    critical_count: acc.critical_count,
                    avg_confidence,
                    stream_correlation: u8::try_from(score).unwrap_or(100),
                }
            })
            .collect::<Vec<_>>();

        result.sort_by(|left, right| {
            right
                .stream_correlation
                .cmp(&left.stream_correlation)
                .then_with(|| right.critical_count.cmp(&left.critical_count))
                .then_with(|| right.event_count.cmp(&left.event_count))
                .then_with(|| left.reason_code.cmp(&right.reason_code))
        });
        result
    }

    fn percentile(values: &[u64], pct: u8) -> u64 {
        if values.is_empty() {
            return 0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let len_minus_one = sorted.len().saturating_sub(1);
        let pct_usize = usize::from(pct.min(100));
        let numerator = len_minus_one.saturating_mul(pct_usize).saturating_add(50);
        let index = numerator.saturating_div(100).min(len_minus_one);
        sorted[index]
    }

    const fn window_scale(window: TimeWindow) -> (u64, u64) {
        match window {
            TimeWindow::OneMinute => (100, 100),
            TimeWindow::FifteenMinutes => (95, 100),
            TimeWindow::OneHour => (90, 100),
            TimeWindow::SixHours => (85, 100),
            TimeWindow::TwentyFourHours => (80, 100),
            TimeWindow::ThreeDays => (75, 100),
            TimeWindow::OneWeek => (70, 100),
        }
    }

    fn trend_window_summary(&self) -> String {
        let fleet = self.state.fleet();
        let p95_values = fleet
            .search_metrics
            .values()
            .map(|metrics| metrics.p95_latency_us)
            .collect::<Vec<_>>();
        let mem_values = fleet
            .resources
            .values()
            .map(|metrics| metrics.memory_bytes / (1024 * 1024))
            .collect::<Vec<_>>();

        let base_p95 = Self::percentile(&p95_values, 95);
        let base_mem = Self::percentile(&mem_values, 95);

        let lag_penalty = self.state.control_plane_metrics().ingestion_lag_events / 50;
        let segments = TimeWindow::ALL
            .iter()
            .map(|window| {
                let (numerator, denominator) = Self::window_scale(*window);
                let p95 = base_p95
                    .saturating_mul(numerator)
                    .saturating_div(denominator)
                    .saturating_add(lag_penalty);
                let memory = base_mem
                    .saturating_mul(numerator)
                    .saturating_div(denominator);
                format!("{} p95={p95}us mem95={memory}MiB", window.label())
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!("trends: {segments}")
    }

    fn correlation_summary_line(rows: &[CorrelationRow]) -> String {
        if let Some(top) = rows.first() {
            return format!(
                "correlation: top_reason={} score={} critical={} span={} projects",
                top.reason_code, top.stream_correlation, top.critical_count, top.project_span
            );
        }
        "correlation: no rows".to_owned()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn export_snapshot_line(&self) -> String {
        let rows = self.filtered_evidence_rows();
        self.export_snapshot_line_for_rows(&rows)
    }

    fn export_snapshot_line_for_rows(&self, rows: &[EvidenceRow]) -> String {
        if let Some(row) = rows.get(self.selected_row) {
            if self.compact_export {
                return format!(
                    "snapshot(compact): incident={}::{}::{} replay={} (toggle: e)",
                    row.project, row.instance_id, row.ts_ms, row.replay_handle
                );
            }

            return format!(
                "snapshot(full): scope=project:{} reason={} confidence={} replay={} (toggle: e)",
                row.project, row.reason_code, row.confidence, row.replay_handle
            );
        }
        "snapshot: no evidence row selected".to_owned()
    }

    fn filter_summary(&self) -> String {
        let project = self
            .project_filters()
            .get(self.project_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned());
        let reason = self
            .reason_filters()
            .get(self.reason_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned());
        let host = self
            .host_filters()
            .get(self.host_filter_index)
            .cloned()
            .unwrap_or_else(|| "all".to_owned());

        format!(
            "filters: project={project} reason={reason} host={host} | keys: p/r/h/x, e export, g project, l stream, t timeline"
        )
    }

    fn build_correlation_rows(rows: &[CorrelationRow]) -> Vec<Row<'static>> {
        rows.iter()
            .map(|row| {
                let style = if row.stream_correlation >= 80 {
                    Style::default().fg(Color::Red)
                } else if row.stream_correlation >= 50 {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default().fg(Color::Green)
                };

                Row::new(vec![
                    row.reason_code.clone(),
                    row.project_span.to_string(),
                    row.event_count.to_string(),
                    row.critical_count.to_string(),
                    row.avg_confidence.to_string(),
                    row.stream_correlation.to_string(),
                ])
                .style(style)
            })
            .collect()
    }

    fn build_evidence_rows(&self, rows: &[EvidenceRow]) -> Vec<Row<'static>> {
        rows.iter()
            .enumerate()
            .map(|(index, row)| {
                let mut style = Style::default().fg(row.severity.color());
                if index == self.selected_row {
                    style = style.add_modifier(Modifier::REVERSED);
                }
                Row::new(vec![
                    row.ts_ms.to_string(),
                    row.project.clone(),
                    row.host.clone(),
                    row.instance_id.clone(),
                    row.severity.label().to_owned(),
                    row.reason_code.clone(),
                    row.confidence.to_string(),
                    row.replay_handle.clone(),
                ])
                .style(style)
            })
            .collect()
    }

    fn evidence_count(&self) -> usize {
        self.filtered_evidence_rows().len()
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

    const fn event_severity(event: &LifecycleEvent) -> EventSeverity {
        if matches!(event.to, LifecycleState::Stopped) {
            return EventSeverity::Critical;
        }
        if event.attribution_collision
            || matches!(
                event.to,
                LifecycleState::Degraded | LifecycleState::Recovering | LifecycleState::Stale
            )
        {
            return EventSeverity::Warn;
        }
        EventSeverity::Info
    }
}

impl Default for HistoricalAnalyticsScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for HistoricalAnalyticsScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Historical Analytics"
    }

    #[allow(clippy::too_many_lines)]
    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let evidence = self.filtered_evidence_rows();
        let correlation = self.correlation_rows_for_rows(&evidence);

        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8),
                Constraint::Min(8),
                Constraint::Length(10),
            ])
            .split(area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Historical Analytics: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "evidence_rows={} projects={} control_health={}",
                    evidence.len(),
                    self.state
                        .fleet()
                        .instances
                        .iter()
                        .map(|instance| instance.project.as_str())
                        .collect::<BTreeSet<_>>()
                        .len(),
                    self.state.control_plane_health()
                )),
            ]),
            Line::from(self.trend_window_summary()),
            Line::from(Self::correlation_summary_line(&correlation)),
            Line::from(self.filter_summary()),
            Line::from(self.export_snapshot_line_for_rows(&evidence)),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Historical Analytics + Explainability "),
        );
        frame.render_widget(header, chunks[0]);

        let correlation_table = Table::new(
            Self::build_correlation_rows(&correlation),
            [
                Constraint::Min(28),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec![
                "Reason Code",
                "Projects",
                "Events",
                "Critical",
                "ConfAvg",
                "Corr",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Anomaly/Event Correlation "),
        );
        frame.render_widget(correlation_table, chunks[1]);

        let evidence_table = Table::new(
            self.build_evidence_rows(&evidence),
            [
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(16),
                Constraint::Length(8),
                Constraint::Length(24),
                Constraint::Length(8),
                Constraint::Min(26),
            ],
        )
        .header(
            Row::new(vec![
                "Timestamp",
                "Project",
                "Host",
                "Instance",
                "Severity",
                "Reason",
                "Conf",
                "Replay",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Evidence Log + Replay Handles "),
        );
        frame.render_widget(evidence_table, chunks[2]);
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
                    let count = self.evidence_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('p') => {
                    self.cycle_project_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('r') => {
                    self.cycle_reason_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('h') => {
                    self.cycle_host_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('x') => {
                    self.reset_filters();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('e') => {
                    self.compact_export = !self.compact_export;
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('g') | crossterm::event::KeyCode::Enter => {
                    if self.selected_project().is_some() {
                        return ScreenAction::Navigate(self.project_screen_id.clone());
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('l') => {
                    return ScreenAction::Navigate(self.live_stream_screen_id.clone());
                }
                crossterm::event::KeyCode::Char('t') => {
                    return ScreenAction::Navigate(self.timeline_screen_id.clone());
                }
                _ => {}
            }
        }

        ScreenAction::Ignored
    }

    fn semantic_role(&self) -> &'static str {
        "analytics"
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

    #[allow(clippy::too_many_lines)]
    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "host-a:alpha-1".to_owned(),
                    project: "alpha".to_owned(),
                    pid: Some(11),
                    healthy: true,
                    doc_count: 500,
                    pending_jobs: 12,
                },
                InstanceInfo {
                    id: "host-b:beta-1".to_owned(),
                    project: "beta".to_owned(),
                    pid: Some(22),
                    healthy: false,
                    doc_count: 800,
                    pending_jobs: 90,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Degraded,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 10_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-a:alpha-1".to_owned(),
                    from: LifecycleState::Recovering,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.recovered".to_owned(),
                    at_ms: 9_000,
                    attribution_confidence_score: 95,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Degraded,
                    to: LifecycleState::Stopped,
                    reason_code: "lifecycle.discovery.stop".to_owned(),
                    at_ms: 8_000,
                    attribution_confidence_score: 82,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        };

        fleet.search_metrics.insert(
            "host-a:alpha-1".to_owned(),
            SearchMetrics {
                total_searches: 300,
                avg_latency_us: 1_000,
                p95_latency_us: 2_200,
                refined_count: 110,
            },
        );
        fleet.search_metrics.insert(
            "host-b:beta-1".to_owned(),
            SearchMetrics {
                total_searches: 500,
                avg_latency_us: 2_300,
                p95_latency_us: 9_000,
                refined_count: 180,
            },
        );

        fleet.resources.insert(
            "host-a:alpha-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 20.0,
                memory_bytes: 400 * 1024 * 1024,
                io_read_bytes: 10_000,
                io_write_bytes: 4_000,
            },
        );
        fleet.resources.insert(
            "host-b:beta-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 78.0,
                memory_bytes: 980 * 1024 * 1024,
                io_read_bytes: 30_000,
                io_write_bytes: 12_000,
            },
        );

        state.update_fleet(fleet);
        state.update_control_plane(ControlPlaneMetrics {
            ingestion_lag_events: 2_500,
            storage_bytes: 400 * 1024 * 1024,
            storage_limit_bytes: 1024 * 1024 * 1024,
            frame_time_ms: 18.0,
            discovery_latency_ms: 500,
            event_throughput_eps: 12.0,
            rss_bytes: 850 * 1024 * 1024,
            rss_limit_bytes: 1024 * 1024 * 1024,
            dead_letter_events: 4,
        });
        state
    }

    fn context() -> ScreenContext {
        ScreenContext {
            active_screen: ScreenId::new("ops.analytics"),
            terminal_width: 120,
            terminal_height: 40,
            focused: true,
        }
    }

    #[test]
    fn screen_defaults() {
        let screen = HistoricalAnalyticsScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.analytics"));
        assert_eq!(screen.title(), "Historical Analytics");
        assert_eq!(screen.semantic_role(), "analytics");
    }

    #[test]
    fn trend_summary_contains_all_windows() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let summary = screen.trend_window_summary();
        assert!(summary.contains("1m"));
        assert!(summary.contains("15m"));
        assert!(summary.contains("1w"));
    }

    #[test]
    fn correlation_rows_include_reason_codes() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let rows = screen.correlation_rows();
        assert!(!rows.is_empty());
        assert!(rows.iter().any(|row| row.reason_code.contains("lifecycle")));
    }

    #[test]
    fn export_snapshot_line_includes_replay_handle() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let line = screen.export_snapshot_line();
        assert!(line.contains("replay://"));
    }

    #[test]
    fn filters_cycle_and_reset() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();

        let project = InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&project, &ctx), ScreenAction::Consumed);

        let reason = InputEvent::Key(
            crossterm::event::KeyCode::Char('r'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&reason, &ctx), ScreenAction::Consumed);

        let host = InputEvent::Key(
            crossterm::event::KeyCode::Char('h'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&host, &ctx), ScreenAction::Consumed);

        let reset = InputEvent::Key(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&reset, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.evidence_count(), 3);
    }

    #[test]
    fn drilldown_keys_navigate_to_targets() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.set_project_screen_id(ScreenId::new("ops.project.custom"));
        screen.set_live_stream_screen_id(ScreenId::new("ops.stream.custom"));
        screen.set_timeline_screen_id(ScreenId::new("ops.timeline.custom"));
        screen.update_state(&sample_state());
        let ctx = context();

        let goto = InputEvent::Key(
            crossterm::event::KeyCode::Char('g'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&goto, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.project.custom"))
        );

        let stream = InputEvent::Key(
            crossterm::event::KeyCode::Char('l'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&stream, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.stream.custom"))
        );

        let timeline = InputEvent::Key(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&timeline, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.timeline.custom"))
        );
    }
}
