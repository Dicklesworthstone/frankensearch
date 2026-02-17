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
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Row, Table};

use frankensearch_core::LifecycleState;
use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::data_source::TimeWindow;
use crate::state::{AppState, LifecycleEvent};
use crate::theme::SemanticPalette;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnapshotExportMode {
    Compact,
    Full,
}

impl SnapshotExportMode {
    const fn from_compact(compact: bool) -> Self {
        if compact { Self::Compact } else { Self::Full }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SnapshotExportPayload {
    mode: SnapshotExportMode,
    project: String,
    host: String,
    instance_id: String,
    ts_ms: u64,
    reason_code: String,
    confidence: u8,
    replay_handle: String,
}

impl SnapshotExportPayload {
    fn from_row(row: &EvidenceRow, compact_export: bool) -> Self {
        Self {
            mode: SnapshotExportMode::from_compact(compact_export),
            project: row.project.clone(),
            host: row.host.clone(),
            instance_id: row.instance_id.clone(),
            ts_ms: row.ts_ms,
            reason_code: row.reason_code.clone(),
            confidence: row.confidence,
            replay_handle: HistoricalAnalyticsScreen::normalize_replay_handle(row),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReplayTarget {
    project: String,
    instance_id: String,
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
    project_filter_values: Vec<String>,
    reason_filter_values: Vec<String>,
    host_filter_values: Vec<String>,
    selected_row: usize,
    project_filter_index: usize,
    reason_filter_index: usize,
    host_filter_index: usize,
    compact_export: bool,
    project_screen_id: ScreenId,
    live_stream_screen_id: ScreenId,
    timeline_screen_id: ScreenId,
    palette: SemanticPalette,
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
            project_filter_values: vec!["all".to_owned()],
            reason_filter_values: vec!["all".to_owned()],
            host_filter_values: vec!["all".to_owned()],
            selected_row: 0,
            project_filter_index: 0,
            reason_filter_index: 0,
            host_filter_index: 0,
            compact_export: true,
            project_screen_id: ScreenId::new("ops.project"),
            live_stream_screen_id: ScreenId::new("ops.live_stream"),
            timeline_screen_id: ScreenId::new("ops.timeline"),
            palette: SemanticPalette::dark(),
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
        let (project_filter, reason_filter, host_filter) = self.selected_filter_values();
        self.state = state.clone();
        self.rebuild_derived_rows();
        self.restore_filter_indices(&project_filter, &reason_filter, &host_filter);
        self.clamp_filter_indices();
        self.restore_selected_row(focused);
    }

    pub fn set_palette(&mut self, palette: SemanticPalette) {
        self.palette = palette;
    }

    /// Selected project from the evidence table.
    #[must_use]
    pub fn selected_project(&self) -> Option<String> {
        let rows = self.filtered_evidence_rows();
        let row = rows.get(self.selected_row)?;
        self.project_lookup.get(&row.instance_id).cloned()
    }

    /// Apply a project filter by value, defaulting to `all` when absent.
    pub fn set_project_filter(&mut self, project: &str) {
        let focused = self.selected_evidence_key();
        self.project_filter_index = self
            .project_filters()
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(project))
            .unwrap_or(0);
        self.clamp_filter_indices();
        self.restore_selected_row(focused);
    }

    /// Apply a reason filter by value, defaulting to `all` when absent.
    pub fn set_reason_filter(&mut self, reason: &str) {
        let focused = self.selected_evidence_key();
        self.reason_filter_index = self
            .reason_filters()
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(reason))
            .unwrap_or(0);
        self.clamp_filter_indices();
        self.restore_selected_row(focused);
    }

    /// Apply a host filter by value, defaulting to `all` when absent.
    pub fn set_host_filter(&mut self, host: &str) {
        let focused = self.selected_evidence_key();
        self.host_filter_index = self
            .host_filters()
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(host))
            .unwrap_or(0);
        self.clamp_filter_indices();
        self.restore_selected_row(focused);
    }

    /// Active project filter label, if not `all`.
    #[must_use]
    pub fn active_project_filter(&self) -> Option<String> {
        self.project_filters()
            .get(self.project_filter_index)
            .filter(|value| value.as_str() != "all")
            .cloned()
    }

    /// Active reason filter label, if not `all`.
    #[must_use]
    pub fn active_reason_filter(&self) -> Option<String> {
        self.reason_filters()
            .get(self.reason_filter_index)
            .filter(|value| value.as_str() != "all")
            .cloned()
    }

    /// Active host filter label, if not `all`.
    #[must_use]
    pub fn active_host_filter(&self) -> Option<String> {
        self.host_filters()
            .get(self.host_filter_index)
            .filter(|value| value.as_str() != "all")
            .cloned()
    }

    fn selected_replay_target_for_rows(&self, rows: &[EvidenceRow]) -> Option<ReplayTarget> {
        self.selected_snapshot_payload_for_rows(rows)
            .map(|payload| ReplayTarget {
                project: payload.project,
                instance_id: payload.instance_id,
                replay_handle: payload.replay_handle,
            })
    }

    #[cfg(test)]
    #[must_use]
    fn selected_replay_target(&self) -> Option<ReplayTarget> {
        let rows = self.filtered_evidence_rows();
        self.selected_replay_target_for_rows(&rows)
    }

    #[cfg(test)]
    #[must_use]
    fn selected_snapshot_payload(&self) -> Option<SnapshotExportPayload> {
        let rows = self.filtered_evidence_rows();
        self.selected_snapshot_payload_for_rows(&rows)
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
        self.rebuild_filter_values();
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

    fn rebuild_filter_values(&mut self) {
        let projects: BTreeSet<_> = self
            .all_evidence_rows()
            .iter()
            .map(|row| row.project.clone())
            .collect();
        let reasons: BTreeSet<_> = self
            .all_evidence_rows()
            .iter()
            .map(|row| row.reason_code.clone())
            .collect();
        let hosts: BTreeSet<_> = self
            .all_evidence_rows()
            .iter()
            .map(|row| row.host.clone())
            .collect();

        self.project_filter_values.clear();
        self.project_filter_values.push("all".to_owned());
        self.project_filter_values.extend(projects);

        self.reason_filter_values.clear();
        self.reason_filter_values.push("all".to_owned());
        self.reason_filter_values.extend(reasons);

        self.host_filter_values.clear();
        self.host_filter_values.push("all".to_owned());
        self.host_filter_values.extend(hosts);
    }

    fn project_filters(&self) -> &[String] {
        &self.project_filter_values
    }

    fn reason_filters(&self) -> &[String] {
        &self.reason_filter_values
    }

    fn host_filters(&self) -> &[String] {
        &self.host_filter_values
    }

    fn selected_filter_values(&self) -> (String, String, String) {
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
        (project, reason, host)
    }

    fn restore_filter_indices(&mut self, project: &str, reason: &str, host: &str) {
        let project_filters = self.project_filters();
        self.project_filter_index = project_filters
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(project))
            .unwrap_or(0);

        let reason_filters = self.reason_filters();
        self.reason_filter_index = reason_filters
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(reason))
            .unwrap_or(0);

        let host_filters = self.host_filters();
        self.host_filter_index = host_filters
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(host))
            .unwrap_or(0);
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
        if let Some(payload) = self.selected_snapshot_payload_for_rows(rows) {
            if matches!(payload.mode, SnapshotExportMode::Compact) {
                return format!(
                    "snapshot(compact): incident={}::{}::{} replay={} (toggle: e)",
                    payload.project, payload.instance_id, payload.ts_ms, payload.replay_handle
                );
            }

            return format!(
                "snapshot(full): scope=project:{} host:{} reason={} confidence={} replay={} (toggle: e)",
                payload.project,
                payload.host,
                payload.reason_code,
                payload.confidence,
                payload.replay_handle
            );
        }
        "snapshot: no evidence row selected".to_owned()
    }

    fn selected_snapshot_payload_for_rows(
        &self,
        rows: &[EvidenceRow],
    ) -> Option<SnapshotExportPayload> {
        let row = rows.get(self.selected_row)?;
        Some(SnapshotExportPayload::from_row(row, self.compact_export))
    }

    fn normalize_replay_handle(row: &EvidenceRow) -> String {
        let trimmed = row.replay_handle.trim();
        if trimmed.is_empty() {
            return Self::fallback_replay_handle(row);
        }
        if trimmed.starts_with("replay:") || trimmed.starts_with("file:") || trimmed.contains("://")
        {
            return trimmed.to_owned();
        }

        format!(
            "replay://legacy/{}/{}/{}/{}/{}",
            row.project, row.host, row.instance_id, row.ts_ms, trimmed
        )
    }

    fn fallback_replay_handle(row: &EvidenceRow) -> String {
        format!(
            "replay://{}/{}/{}/{}",
            row.project, row.host, row.instance_id, row.ts_ms
        )
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
        let replay_summary = self.selected_replay_target_for_rows(&evidence).map_or_else(
            || "selected_replay: none".to_owned(),
            |target| {
                format!(
                    "selected_replay: {}::{} ({})",
                    target.project, target.instance_id, target.replay_handle
                )
            },
        );

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
            Line::from(replay_summary),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
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
                .border_type(BorderType::Rounded)
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
                .border_type(BorderType::Rounded)
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

    #[test]
    fn export_toggle_switches_snapshot_mode_line() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();

        let compact_line = screen.export_snapshot_line();
        assert!(compact_line.contains("snapshot(compact)"));

        let toggle = InputEvent::Key(
            crossterm::event::KeyCode::Char('e'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&toggle, &ctx), ScreenAction::Consumed);

        let full_line = screen.export_snapshot_line();
        assert!(full_line.contains("snapshot(full)"));
        assert!(full_line.contains("replay://"));
    }

    #[test]
    fn selected_snapshot_payload_and_replay_target_follow_selection() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();

        let first_payload = screen
            .selected_snapshot_payload()
            .expect("payload should exist for initial selection");
        let first_target = screen
            .selected_replay_target()
            .expect("replay target should exist for initial selection");
        assert_eq!(first_payload.project, first_target.project);
        assert_eq!(first_payload.instance_id, first_target.instance_id);
        assert_eq!(first_payload.replay_handle, first_target.replay_handle);

        let down = InputEvent::Key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);

        let second_payload = screen
            .selected_snapshot_payload()
            .expect("payload should still exist after moving selection");
        assert_ne!(
            (first_payload.ts_ms, first_payload.reason_code),
            (second_payload.ts_ms, second_payload.reason_code)
        );
    }

    #[test]
    fn selected_snapshot_payload_handles_empty_rows() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&AppState::new());
        assert!(screen.selected_snapshot_payload().is_none());
        assert!(screen.selected_replay_target().is_none());
        assert_eq!(
            screen.export_snapshot_line(),
            "snapshot: no evidence row selected"
        );
    }

    #[test]
    fn selected_snapshot_payload_handles_filtered_zero_rows() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());

        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "alpha")
            .expect("alpha project filter should exist");
        screen.reason_filter_index = screen
            .reason_filters()
            .iter()
            .position(|value| value == "lifecycle.heartbeat_gap")
            .expect("heartbeat-gap reason filter should exist");
        screen.clamp_filter_indices();

        assert_eq!(screen.evidence_count(), 0);
        assert!(screen.selected_snapshot_payload().is_none());
        assert!(screen.selected_replay_target().is_none());
        assert_eq!(
            screen.export_snapshot_line(),
            "snapshot: no evidence row selected"
        );
    }

    #[test]
    fn selected_snapshot_payload_normalizes_missing_replay_handle() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.evidence_rows[0].replay_handle.clear();

        let payload = screen
            .selected_snapshot_payload()
            .expect("payload should still exist with fallback replay handle");
        assert!(payload.replay_handle.starts_with("replay://"));
        assert!(payload.replay_handle.contains(&payload.project));
        assert!(payload.replay_handle.contains(&payload.instance_id));
    }

    #[test]
    fn selected_snapshot_payload_normalizes_legacy_replay_handle() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.evidence_rows[0].replay_handle = "legacy-handle".to_owned();

        let payload = screen
            .selected_snapshot_payload()
            .expect("payload should exist with normalized legacy replay handle");
        assert!(payload.replay_handle.starts_with("replay://legacy/"));
        assert!(payload.replay_handle.contains(&payload.host));
        assert!(payload.replay_handle.ends_with("/legacy-handle"));
    }

    #[test]
    fn selected_snapshot_payload_preserves_existing_uri_scheme() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.evidence_rows[0].replay_handle = "artifact://bundle/segment/42".to_owned();

        let payload = screen
            .selected_snapshot_payload()
            .expect("payload should preserve existing uri-schemed replay handle");
        assert_eq!(payload.replay_handle, "artifact://bundle/segment/42");
    }

    #[test]
    fn selected_project_ignores_unknown_sentinel_rows() {
        let mut state = sample_state();
        let mut fleet = state.fleet().clone();
        fleet.lifecycle_events.push(LifecycleEvent {
            instance_id: "orphan-host:missing-inst".to_owned(),
            from: LifecycleState::Started,
            to: LifecycleState::Stale,
            reason_code: "lifecycle.instance.orphan".to_owned(),
            at_ms: 12_500,
            attribution_confidence_score: 40,
            attribution_collision: false,
        });
        state.update_fleet(fleet);

        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&state);
        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "unknown")
            .expect("unknown project filter should exist");

        assert_eq!(
            screen.selected_project(),
            None,
            "unknown sentinel should not be forwarded as a project drilldown target"
        );
    }

    #[test]
    fn selected_project_allows_real_unknown_project_name() {
        let mut state = sample_state();
        let mut fleet = state.fleet().clone();
        fleet.instances.push(InstanceInfo {
            id: "host-u:unknown-1".to_owned(),
            project: "unknown".to_owned(),
            pid: Some(99),
            healthy: true,
            doc_count: 12,
            pending_jobs: 0,
        });
        fleet.lifecycle_events.insert(
            0,
            LifecycleEvent {
                instance_id: "host-u:unknown-1".to_owned(),
                from: LifecycleState::Started,
                to: LifecycleState::Healthy,
                reason_code: "lifecycle.discovery.ready".to_owned(),
                at_ms: 13_000,
                attribution_confidence_score: 96,
                attribution_collision: false,
            },
        );
        state.update_fleet(fleet);

        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&state);
        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "unknown")
            .expect("unknown project filter should exist");

        assert_eq!(
            screen.selected_project().as_deref(),
            Some("unknown"),
            "real project names should not be dropped as sentinel values"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn historical_update_state_preserves_filter_value_when_new_option_is_inserted() {
        let mut screen = HistoricalAnalyticsScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "host-a:alpha-1".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(11),
                    healthy: true,
                    doc_count: 140,
                    pending_jobs: 1,
                },
                InstanceInfo {
                    id: "host-b:beta-1".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 90,
                    pending_jobs: 4,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:alpha-1".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.ready".to_owned(),
                    at_ms: 12_000,
                    attribution_confidence_score: 95,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 10_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
            ],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state);

        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "proj-b")
            .expect("project filter should exist");

        state.update_fleet(FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "host-a:alpha-1".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(11),
                    healthy: true,
                    doc_count: 140,
                    pending_jobs: 1,
                },
                InstanceInfo {
                    id: "host-b:beta-1".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 90,
                    pending_jobs: 4,
                },
                InstanceInfo {
                    id: "host-z:zeta-1".to_owned(),
                    project: "proj-aa".to_owned(),
                    pid: Some(13),
                    healthy: true,
                    doc_count: 20,
                    pending_jobs: 0,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:alpha-1".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.ready".to_owned(),
                    at_ms: 12_000,
                    attribution_confidence_score: 95,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 10_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-z:zeta-1".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.ready".to_owned(),
                    at_ms: 9_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        });

        screen.update_state(&state);

        let selected_project = screen
            .project_filters()
            .get(screen.project_filter_index)
            .cloned();
        assert_eq!(selected_project.as_deref(), Some("proj-b"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn historical_update_state_preserves_reason_and_host_filter_values_when_new_options_are_inserted()
     {
        let mut screen = HistoricalAnalyticsScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "host-a:alpha-1".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(11),
                    healthy: true,
                    doc_count: 140,
                    pending_jobs: 1,
                },
                InstanceInfo {
                    id: "host-b:beta-1".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 90,
                    pending_jobs: 4,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:alpha-1".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.ready".to_owned(),
                    at_ms: 12_000,
                    attribution_confidence_score: 95,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 10_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
            ],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state);

        screen.reason_filter_index = screen
            .reason_filters()
            .iter()
            .position(|value| value == "lifecycle.heartbeat_gap")
            .expect("reason filter should exist");
        screen.host_filter_index = screen
            .host_filters()
            .iter()
            .position(|value| value == "host-b")
            .expect("host filter should exist");

        state.update_fleet(FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "host-a:alpha-1".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(11),
                    healthy: true,
                    doc_count: 140,
                    pending_jobs: 1,
                },
                InstanceInfo {
                    id: "host-b:beta-1".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 90,
                    pending_jobs: 4,
                },
                InstanceInfo {
                    id: "host-aa:gamma-1".to_owned(),
                    project: "proj-c".to_owned(),
                    pid: Some(13),
                    healthy: true,
                    doc_count: 20,
                    pending_jobs: 0,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:alpha-1".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.ready".to_owned(),
                    at_ms: 12_000,
                    attribution_confidence_score: 95,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 10_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-aa:gamma-1".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.anomaly.alpha".to_owned(),
                    at_ms: 9_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        });

        screen.update_state(&state);

        let selected_reason = screen
            .reason_filters()
            .get(screen.reason_filter_index)
            .cloned();
        let selected_host = screen.host_filters().get(screen.host_filter_index).cloned();
        assert_eq!(selected_reason.as_deref(), Some("lifecycle.heartbeat_gap"));
        assert_eq!(selected_host.as_deref(), Some("host-b"));
    }

    // ─── bd-3q2p tests begin ───

    // ── EventSeverity ──

    #[test]
    fn event_severity_label_all_variants() {
        assert_eq!(EventSeverity::Info.label(), "info");
        assert_eq!(EventSeverity::Warn.label(), "warn");
        assert_eq!(EventSeverity::Critical.label(), "critical");
    }

    #[test]
    fn event_severity_color_all_variants() {
        assert_eq!(EventSeverity::Info.color(), Color::Gray);
        assert_eq!(EventSeverity::Warn.color(), Color::Yellow);
        assert_eq!(EventSeverity::Critical.color(), Color::Red);
    }

    #[test]
    fn event_severity_rank_ordering() {
        assert_eq!(EventSeverity::Info.rank(), 0);
        assert_eq!(EventSeverity::Warn.rank(), 1);
        assert_eq!(EventSeverity::Critical.rank(), 2);
        assert!(EventSeverity::Critical.rank() > EventSeverity::Warn.rank());
        assert!(EventSeverity::Warn.rank() > EventSeverity::Info.rank());
    }

    // ── SnapshotExportMode ──

    #[test]
    fn snapshot_export_mode_from_compact() {
        assert_eq!(
            SnapshotExportMode::from_compact(true),
            SnapshotExportMode::Compact
        );
        assert_eq!(
            SnapshotExportMode::from_compact(false),
            SnapshotExportMode::Full
        );
    }

    // ── host_bucket ──

    #[test]
    fn host_bucket_colon_separator() {
        assert_eq!(
            HistoricalAnalyticsScreen::host_bucket("host-a:alpha-1"),
            "host-a"
        );
    }

    #[test]
    fn host_bucket_dash_separator() {
        assert_eq!(
            HistoricalAnalyticsScreen::host_bucket("hostonly-inst42"),
            "hostonly"
        );
    }

    #[test]
    fn host_bucket_no_separator() {
        assert_eq!(
            HistoricalAnalyticsScreen::host_bucket("singlesegment"),
            "singlesegment"
        );
    }

    #[test]
    fn host_bucket_colon_takes_priority_over_dash() {
        assert_eq!(HistoricalAnalyticsScreen::host_bucket("a-b:c-d"), "a-b");
    }

    // ── event_severity classification ──

    #[test]
    fn event_severity_stopped_is_critical() {
        let event = LifecycleEvent {
            instance_id: "x:y".into(),
            from: LifecycleState::Healthy,
            to: LifecycleState::Stopped,
            reason_code: "test".into(),
            at_ms: 1,
            attribution_confidence_score: 80,
            attribution_collision: false,
        };
        assert_eq!(
            HistoricalAnalyticsScreen::event_severity(&event),
            EventSeverity::Critical
        );
    }

    #[test]
    fn event_severity_degraded_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".into(),
            from: LifecycleState::Healthy,
            to: LifecycleState::Degraded,
            reason_code: "test".into(),
            at_ms: 1,
            attribution_confidence_score: 60,
            attribution_collision: false,
        };
        assert_eq!(
            HistoricalAnalyticsScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_recovering_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".into(),
            from: LifecycleState::Degraded,
            to: LifecycleState::Recovering,
            reason_code: "test".into(),
            at_ms: 1,
            attribution_confidence_score: 50,
            attribution_collision: false,
        };
        assert_eq!(
            HistoricalAnalyticsScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_stale_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".into(),
            from: LifecycleState::Healthy,
            to: LifecycleState::Stale,
            reason_code: "test".into(),
            at_ms: 1,
            attribution_confidence_score: 40,
            attribution_collision: false,
        };
        assert_eq!(
            HistoricalAnalyticsScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_collision_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".into(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "test".into(),
            at_ms: 1,
            attribution_confidence_score: 90,
            attribution_collision: true,
        };
        assert_eq!(
            HistoricalAnalyticsScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_healthy_no_collision_is_info() {
        let event = LifecycleEvent {
            instance_id: "x:y".into(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "test".into(),
            at_ms: 1,
            attribution_confidence_score: 95,
            attribution_collision: false,
        };
        assert_eq!(
            HistoricalAnalyticsScreen::event_severity(&event),
            EventSeverity::Info
        );
    }

    // ── percentile ──

    #[test]
    fn percentile_empty_returns_zero() {
        assert_eq!(HistoricalAnalyticsScreen::percentile(&[], 50), 0);
    }

    #[test]
    fn percentile_single_element() {
        assert_eq!(HistoricalAnalyticsScreen::percentile(&[42], 0), 42);
        assert_eq!(HistoricalAnalyticsScreen::percentile(&[42], 50), 42);
        assert_eq!(HistoricalAnalyticsScreen::percentile(&[42], 100), 42);
    }

    #[test]
    fn percentile_sorted_input() {
        let values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let p0 = HistoricalAnalyticsScreen::percentile(&values, 0);
        let p50 = HistoricalAnalyticsScreen::percentile(&values, 50);
        let p95 = HistoricalAnalyticsScreen::percentile(&values, 95);
        let p100 = HistoricalAnalyticsScreen::percentile(&values, 100);
        assert_eq!(p0, 10);
        assert!((50..=60).contains(&p50));
        assert!(p95 >= 90);
        assert_eq!(p100, 100);
    }

    #[test]
    fn percentile_unsorted_input() {
        let values = [50, 10, 90, 30, 70];
        let p50 = HistoricalAnalyticsScreen::percentile(&values, 50);
        assert_eq!(p50, 50);
    }

    #[test]
    fn percentile_pct_clamped_above_100() {
        let values = [1, 2, 3];
        let p200 = HistoricalAnalyticsScreen::percentile(&values, 200);
        assert_eq!(p200, 3);
    }

    // ── window_scale ──

    #[test]
    fn window_scale_all_variants() {
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::OneMinute),
            (100, 100)
        );
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::FifteenMinutes),
            (95, 100)
        );
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::OneHour),
            (90, 100)
        );
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::SixHours),
            (85, 100)
        );
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::TwentyFourHours),
            (80, 100)
        );
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::ThreeDays),
            (75, 100)
        );
        assert_eq!(
            HistoricalAnalyticsScreen::window_scale(TimeWindow::OneWeek),
            (70, 100)
        );
    }

    #[test]
    fn window_scale_monotonically_decreases() {
        let windows = TimeWindow::ALL;
        for pair in windows.windows(2) {
            let (a_num, _) = HistoricalAnalyticsScreen::window_scale(pair[0]);
            let (b_num, _) = HistoricalAnalyticsScreen::window_scale(pair[1]);
            assert!(
                a_num >= b_num,
                "window_scale should decrease: {:?}={a_num} >= {:?}={b_num}",
                pair[0],
                pair[1]
            );
        }
    }

    // ── normalize_replay_handle ──

    #[test]
    fn normalize_replay_handle_empty_produces_fallback() {
        let row = EvidenceRow {
            ts_ms: 100,
            project: "proj".into(),
            host: "h1".into(),
            instance_id: "h1:inst".into(),
            severity: EventSeverity::Info,
            reason_code: "test".into(),
            confidence: 90,
            replay_handle: String::new(),
        };
        let normalized = HistoricalAnalyticsScreen::normalize_replay_handle(&row);
        assert!(normalized.starts_with("replay://"));
        assert!(normalized.contains("proj"));
        assert!(normalized.contains("h1:inst"));
    }

    #[test]
    fn normalize_replay_handle_preserves_replay_prefix() {
        let row = EvidenceRow {
            ts_ms: 100,
            project: "proj".into(),
            host: "h1".into(),
            instance_id: "h1:inst".into(),
            severity: EventSeverity::Info,
            reason_code: "test".into(),
            confidence: 90,
            replay_handle: "replay://proj/h1/inst/100".into(),
        };
        assert_eq!(
            HistoricalAnalyticsScreen::normalize_replay_handle(&row),
            "replay://proj/h1/inst/100"
        );
    }

    #[test]
    fn normalize_replay_handle_preserves_file_prefix() {
        let row = EvidenceRow {
            ts_ms: 100,
            project: "proj".into(),
            host: "h1".into(),
            instance_id: "h1:inst".into(),
            severity: EventSeverity::Info,
            reason_code: "test".into(),
            confidence: 90,
            replay_handle: "file:///tmp/replay.json".into(),
        };
        assert_eq!(
            HistoricalAnalyticsScreen::normalize_replay_handle(&row),
            "file:///tmp/replay.json"
        );
    }

    #[test]
    fn normalize_replay_handle_preserves_arbitrary_uri() {
        let row = EvidenceRow {
            ts_ms: 100,
            project: "proj".into(),
            host: "h1".into(),
            instance_id: "h1:inst".into(),
            severity: EventSeverity::Info,
            reason_code: "test".into(),
            confidence: 90,
            replay_handle: "s3://bucket/key".into(),
        };
        assert_eq!(
            HistoricalAnalyticsScreen::normalize_replay_handle(&row),
            "s3://bucket/key"
        );
    }

    #[test]
    fn normalize_replay_handle_wraps_legacy() {
        let row = EvidenceRow {
            ts_ms: 100,
            project: "proj".into(),
            host: "h1".into(),
            instance_id: "h1:inst".into(),
            severity: EventSeverity::Info,
            reason_code: "test".into(),
            confidence: 90,
            replay_handle: "legacy-token".into(),
        };
        let normalized = HistoricalAnalyticsScreen::normalize_replay_handle(&row);
        assert!(normalized.starts_with("replay://legacy/"));
        assert!(normalized.ends_with("/legacy-token"));
    }

    // ── fallback_replay_handle ──

    #[test]
    fn fallback_replay_handle_format() {
        let row = EvidenceRow {
            ts_ms: 42,
            project: "myproj".into(),
            host: "h1".into(),
            instance_id: "h1:abc".into(),
            severity: EventSeverity::Info,
            reason_code: "test".into(),
            confidence: 50,
            replay_handle: String::new(),
        };
        let fallback = HistoricalAnalyticsScreen::fallback_replay_handle(&row);
        assert_eq!(fallback, "replay://myproj/h1/h1:abc/42");
    }

    // ── correlation_summary_line ──

    #[test]
    fn correlation_summary_line_empty() {
        let line = HistoricalAnalyticsScreen::correlation_summary_line(&[]);
        assert_eq!(line, "correlation: no rows");
    }

    #[test]
    fn correlation_summary_line_with_data() {
        let rows = vec![CorrelationRow {
            reason_code: "lifecycle.crash".into(),
            project_span: 3,
            event_count: 10,
            critical_count: 4,
            avg_confidence: 80,
            stream_correlation: 92,
        }];
        let line = HistoricalAnalyticsScreen::correlation_summary_line(&rows);
        assert!(line.contains("lifecycle.crash"));
        assert!(line.contains("score=92"));
        assert!(line.contains("critical=4"));
        assert!(line.contains("span=3"));
    }

    // ── Default impl ──

    #[test]
    fn default_impl_matches_new() {
        let from_new = HistoricalAnalyticsScreen::new();
        let from_default = HistoricalAnalyticsScreen::default();
        assert_eq!(from_new.id(), from_default.id());
        assert_eq!(from_new.title(), from_default.title());
        assert_eq!(from_new.semantic_role(), from_default.semantic_role());
        assert_eq!(from_new.evidence_count(), from_default.evidence_count());
    }

    // ── active_*_filter ──

    #[test]
    fn active_project_filter_none_when_all() {
        let screen = HistoricalAnalyticsScreen::new();
        assert_eq!(screen.active_project_filter(), None);
    }

    #[test]
    fn active_reason_filter_none_when_all() {
        let screen = HistoricalAnalyticsScreen::new();
        assert_eq!(screen.active_reason_filter(), None);
    }

    #[test]
    fn active_host_filter_none_when_all() {
        let screen = HistoricalAnalyticsScreen::new();
        assert_eq!(screen.active_host_filter(), None);
    }

    #[test]
    fn active_project_filter_some_when_filtered() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_project_filter("alpha");
        assert_eq!(screen.active_project_filter().as_deref(), Some("alpha"));
    }

    #[test]
    fn active_reason_filter_some_when_filtered() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_reason_filter("lifecycle.heartbeat_gap");
        assert_eq!(
            screen.active_reason_filter().as_deref(),
            Some("lifecycle.heartbeat_gap")
        );
    }

    #[test]
    fn active_host_filter_some_when_filtered() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_host_filter("host-b");
        assert_eq!(screen.active_host_filter().as_deref(), Some("host-b"));
    }

    // ── set_*_filter falls back to all when unknown ──

    #[test]
    fn set_project_filter_unknown_falls_back_to_all() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_project_filter("nonexistent");
        assert_eq!(screen.active_project_filter(), None);
    }

    #[test]
    fn set_reason_filter_unknown_falls_back_to_all() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_reason_filter("nonexistent");
        assert_eq!(screen.active_reason_filter(), None);
    }

    #[test]
    fn set_host_filter_unknown_falls_back_to_all() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_host_filter("nonexistent");
        assert_eq!(screen.active_host_filter(), None);
    }

    // ── Selection navigation ──

    #[test]
    fn nav_up_at_zero_stays_at_zero() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();
        let up = InputEvent::Key(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&up, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 0);
    }

    #[test]
    fn nav_down_increments_selection() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();
        let down = InputEvent::Key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
    }

    #[test]
    fn nav_k_moves_up() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.selected_row = 2;
        let ctx = context();
        let k = InputEvent::Key(
            crossterm::event::KeyCode::Char('k'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&k, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
    }

    #[test]
    fn nav_j_moves_down() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();
        let j = InputEvent::Key(
            crossterm::event::KeyCode::Char('j'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&j, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
    }

    #[test]
    fn nav_down_clamps_at_last_row() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();
        let count = screen.evidence_count();
        screen.selected_row = count.saturating_sub(1);
        let down = InputEvent::Key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, count.saturating_sub(1));
    }

    #[test]
    fn unrecognized_key_returns_ignored() {
        let mut screen = HistoricalAnalyticsScreen::new();
        let ctx = context();
        let f12 = InputEvent::Key(
            crossterm::event::KeyCode::F(12),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&f12, &ctx), ScreenAction::Ignored);
    }

    // ── Evidence sorting ──

    #[test]
    fn evidence_rows_sorted_severity_then_timestamp() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let rows = screen.filtered_evidence_rows();
        assert!(rows.len() >= 3);
        // Critical should come first, then Warn, then Info
        assert_eq!(rows[0].severity, EventSeverity::Critical);
        assert_eq!(rows[1].severity, EventSeverity::Warn);
        assert_eq!(rows[2].severity, EventSeverity::Info);
    }

    // ── Correlation rows ──

    #[test]
    fn correlation_rows_aggregate_correctly() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let rows = screen.correlation_rows();
        // We have 3 events with 3 distinct reason codes
        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert!(row.event_count >= 1);
            assert!(row.avg_confidence > 0);
            assert!(row.stream_correlation <= 100);
        }
    }

    #[test]
    fn correlation_rows_empty_state() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&AppState::new());
        let rows = screen.correlation_rows();
        assert!(rows.is_empty());
    }

    // ── filter_summary ──

    #[test]
    fn filter_summary_contains_default_labels() {
        let screen = HistoricalAnalyticsScreen::new();
        let summary = screen.filter_summary();
        assert!(summary.contains("project=all"));
        assert!(summary.contains("reason=all"));
        assert!(summary.contains("host=all"));
    }

    #[test]
    fn filter_summary_reflects_active_filter() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        screen.set_project_filter("alpha");
        let summary = screen.filter_summary();
        assert!(summary.contains("project=alpha"));
    }

    // ── evidence_count ──

    #[test]
    fn evidence_count_empty() {
        let screen = HistoricalAnalyticsScreen::new();
        assert_eq!(screen.evidence_count(), 0);
    }

    #[test]
    fn evidence_count_with_data() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        assert_eq!(screen.evidence_count(), 3);
    }

    // ── SnapshotExportPayload from_row ──

    #[test]
    fn snapshot_export_payload_from_row_compact() {
        let row = EvidenceRow {
            ts_ms: 100,
            project: "proj".into(),
            host: "h1".into(),
            instance_id: "h1:inst".into(),
            severity: EventSeverity::Info,
            reason_code: "test.reason".into(),
            confidence: 80,
            replay_handle: "replay://proj/h1/h1:inst/100".into(),
        };
        let payload = SnapshotExportPayload::from_row(&row, true);
        assert_eq!(payload.mode, SnapshotExportMode::Compact);
        assert_eq!(payload.project, "proj");
        assert_eq!(payload.instance_id, "h1:inst");
        assert_eq!(payload.ts_ms, 100);
        assert_eq!(payload.confidence, 80);
    }

    #[test]
    fn snapshot_export_payload_from_row_full() {
        let row = EvidenceRow {
            ts_ms: 200,
            project: "proj".into(),
            host: "h2".into(),
            instance_id: "h2:inst".into(),
            severity: EventSeverity::Warn,
            reason_code: "test.warn".into(),
            confidence: 60,
            replay_handle: "replay://proj/h2/h2:inst/200".into(),
        };
        let payload = SnapshotExportPayload::from_row(&row, false);
        assert_eq!(payload.mode, SnapshotExportMode::Full);
        assert_eq!(payload.reason_code, "test.warn");
        assert_eq!(payload.host, "h2");
    }

    // ── Enter key navigates to project drilldown ──

    #[test]
    fn enter_key_navigates_to_project() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();
        let enter = InputEvent::Key(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        );
        let action = screen.handle_input(&enter, &ctx);
        assert!(matches!(action, ScreenAction::Navigate(_)));
    }

    // ── selected_project with valid data ──

    #[test]
    fn selected_project_returns_project_name() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let project = screen.selected_project();
        assert!(project.is_some());
    }

    #[test]
    fn selected_project_empty_state_is_none() {
        let screen = HistoricalAnalyticsScreen::new();
        assert!(screen.selected_project().is_none());
    }

    // ── set_*_screen_id ──

    #[test]
    fn set_screen_ids_affect_navigation() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.set_project_screen_id(ScreenId::new("custom.project"));
        screen.set_live_stream_screen_id(ScreenId::new("custom.stream"));
        screen.set_timeline_screen_id(ScreenId::new("custom.timeline"));
        screen.update_state(&sample_state());
        let ctx = context();

        let l = InputEvent::Key(
            crossterm::event::KeyCode::Char('l'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&l, &ctx),
            ScreenAction::Navigate(ScreenId::new("custom.stream"))
        );

        let t = InputEvent::Key(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&t, &ctx),
            ScreenAction::Navigate(ScreenId::new("custom.timeline"))
        );
    }

    // ── trend_window_summary ──

    #[test]
    fn trend_window_summary_contains_all_labels() {
        let mut screen = HistoricalAnalyticsScreen::new();
        screen.update_state(&sample_state());
        let summary = screen.trend_window_summary();
        assert!(summary.starts_with("trends:"));
        assert!(summary.contains("p95="));
        assert!(summary.contains("mem95="));
    }

    // ── as_any ──

    #[test]
    fn as_any_downcast_works() {
        let screen = HistoricalAnalyticsScreen::new();
        let any_ref: &dyn Any = screen.as_any();
        assert!(
            any_ref
                .downcast_ref::<HistoricalAnalyticsScreen>()
                .is_some()
        );
    }

    // ─── bd-3q2p tests end ───
}
