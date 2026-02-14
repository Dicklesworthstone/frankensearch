//! Alerts + SLO health + capacity forecast screen.
//!
//! Provides a triage-focused view that combines active alerts,
//! error-budget signals, and capacity risk indicators.

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

use crate::state::{AppState, ControlPlaneHealth, LifecycleEvent};

const TARGET_P95_US: u64 = 3_000;
const BURN_WARN: f64 = 0.8;
const BURN_CRITICAL: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AlertSeverity {
    Info,
    Warn,
    Critical,
}

impl AlertSeverity {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeverityFilter {
    All,
    Warn,
    Critical,
}

impl SeverityFilter {
    const fn next(self) -> Self {
        match self {
            Self::All => Self::Warn,
            Self::Warn => Self::Critical,
            Self::Critical => Self::All,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Warn => "warn+",
            Self::Critical => "critical",
        }
    }

    const fn allows(self, severity: AlertSeverity) -> bool {
        match self {
            Self::All => true,
            Self::Warn => matches!(severity, AlertSeverity::Warn | AlertSeverity::Critical),
            Self::Critical => matches!(severity, AlertSeverity::Critical),
        }
    }
}

#[derive(Clone, Debug)]
struct AlertRow {
    ts_ms: u64,
    project: String,
    host: String,
    instance_id: String,
    severity: AlertSeverity,
    confidence: u8,
    suppression_state: &'static str,
    reason_code: String,
}

#[derive(Clone, Debug)]
struct SloProjectRow {
    project: String,
    instance_count: usize,
    unhealthy_count: usize,
    pending_jobs: u64,
    p95_latency_us: u64,
    burn_ratio: f64,
    remaining_ratio: f64,
    backlog_eta_s: u64,
    saturation_risk: &'static str,
    status: ControlPlaneHealth,
}

/// Alerts + SLO + capacity screen for operator triage.
pub struct AlertsSloScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
    project_filter_index: usize,
    reason_filter_index: usize,
    host_filter_index: usize,
    severity_filter: SeverityFilter,
    project_screen_id: ScreenId,
    live_stream_screen_id: ScreenId,
    timeline_screen_id: ScreenId,
}

impl AlertsSloScreen {
    /// Create a new alerts/SLO screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.alerts"),
            state: AppState::new(),
            selected_row: 0,
            project_filter_index: 0,
            reason_filter_index: 0,
            host_filter_index: 0,
            severity_filter: SeverityFilter::All,
            project_screen_id: ScreenId::new("ops.project"),
            live_stream_screen_id: ScreenId::new("ops.live_stream"),
            timeline_screen_id: ScreenId::new("ops.timeline"),
        }
    }

    /// Update the project-detail drilldown destination.
    pub fn set_project_screen_id(&mut self, id: ScreenId) {
        self.project_screen_id = id;
    }

    /// Update the live-stream drilldown destination.
    pub fn set_live_stream_screen_id(&mut self, id: ScreenId) {
        self.live_stream_screen_id = id;
    }

    /// Update the timeline drilldown destination.
    pub fn set_timeline_screen_id(&mut self, id: ScreenId) {
        self.timeline_screen_id = id;
    }

    /// Update state from shared app snapshot.
    pub fn update_state(&mut self, state: &AppState) {
        let focused = self.selected_alert_key();
        self.state = state.clone();
        self.clamp_filter_indices();
        self.restore_selected_alert(focused);
    }

    /// Selected project in the filtered alert table.
    #[must_use]
    pub fn selected_project(&self) -> Option<String> {
        let filtered = self.filtered_alerts();
        let row = filtered.get(self.selected_row)?;
        self.project_for_instance(&row.instance_id)
            .map(std::borrow::ToOwned::to_owned)
    }

    fn all_alerts(&self) -> Vec<AlertRow> {
        let mut rows = self
            .state
            .fleet()
            .lifecycle_events
            .iter()
            .map(|event| AlertRow {
                ts_ms: event.at_ms,
                project: self
                    .project_for_instance(&event.instance_id)
                    .unwrap_or("unknown")
                    .to_owned(),
                host: Self::host_bucket(&event.instance_id),
                instance_id: event.instance_id.clone(),
                severity: Self::event_severity(event),
                confidence: event.attribution_confidence_score,
                suppression_state: Self::suppression_state(event),
                reason_code: event.reason_code.clone(),
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
        rows
    }

    fn filtered_alerts(&self) -> Vec<AlertRow> {
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

        self.all_alerts()
            .into_iter()
            .filter(|row| {
                project_filter
                    .as_deref()
                    .is_none_or(|project| row.project.eq_ignore_ascii_case(project))
            })
            .filter(|row| self.severity_filter.allows(row.severity))
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
            .collect()
    }

    fn project_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let projects: BTreeSet<_> = self
            .all_alerts()
            .into_iter()
            .map(|row| row.project)
            .collect();
        values.extend(projects);
        values
    }

    fn reason_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let reasons: BTreeSet<_> = self
            .all_alerts()
            .into_iter()
            .map(|row| row.reason_code)
            .collect();
        values.extend(reasons);
        values
    }

    fn host_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let hosts: BTreeSet<_> = self.all_alerts().into_iter().map(|row| row.host).collect();
        values.extend(hosts);
        values
    }

    fn project_for_instance(&self, instance_id: &str) -> Option<&str> {
        self.state
            .fleet()
            .instances
            .iter()
            .find(|instance| instance.id == instance_id)
            .map(|instance| instance.project.as_str())
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

    const fn event_severity(event: &LifecycleEvent) -> AlertSeverity {
        if matches!(event.to, LifecycleState::Stopped) {
            return AlertSeverity::Critical;
        }
        if event.attribution_collision
            || matches!(
                event.to,
                LifecycleState::Degraded | LifecycleState::Recovering | LifecycleState::Stale
            )
        {
            return AlertSeverity::Warn;
        }
        AlertSeverity::Info
    }

    const fn suppression_state(event: &LifecycleEvent) -> &'static str {
        if matches!(event.to, LifecycleState::Healthy) {
            return "suppressed";
        }
        if event.attribution_collision {
            return "investigating";
        }
        "active"
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

    fn selected_alert_key(&self) -> Option<(u64, String, String)> {
        self.filtered_alerts()
            .get(self.selected_row)
            .map(|row| (row.ts_ms, row.instance_id.clone(), row.reason_code.clone()))
    }

    fn restore_selected_alert(&mut self, key: Option<(u64, String, String)>) {
        let alerts = self.filtered_alerts();
        if alerts.is_empty() {
            self.selected_row = 0;
            return;
        }

        if let Some((ts_ms, instance_id, reason_code)) = key
            && let Some(index) = alerts.iter().position(|row| {
                row.ts_ms == ts_ms
                    && row.instance_id == instance_id
                    && row.reason_code == reason_code
            })
        {
            self.selected_row = index;
            return;
        }

        if self.selected_row >= alerts.len() {
            self.selected_row = alerts.len().saturating_sub(1);
        }
    }

    fn cycle_project_filter(&mut self) {
        let focused = self.selected_alert_key();
        let len = self.project_filters().len();
        if len > 0 {
            self.project_filter_index = (self.project_filter_index + 1) % len;
        }
        self.restore_selected_alert(focused);
    }

    fn cycle_reason_filter(&mut self) {
        let focused = self.selected_alert_key();
        let len = self.reason_filters().len();
        if len > 0 {
            self.reason_filter_index = (self.reason_filter_index + 1) % len;
        }
        self.restore_selected_alert(focused);
    }

    fn cycle_host_filter(&mut self) {
        let focused = self.selected_alert_key();
        let len = self.host_filters().len();
        if len > 0 {
            self.host_filter_index = (self.host_filter_index + 1) % len;
        }
        self.restore_selected_alert(focused);
    }

    fn cycle_severity_filter(&mut self) {
        let focused = self.selected_alert_key();
        self.severity_filter = self.severity_filter.next();
        self.restore_selected_alert(focused);
    }

    fn reset_filters(&mut self) {
        let focused = self.selected_alert_key();
        self.project_filter_index = 0;
        self.reason_filter_index = 0;
        self.host_filter_index = 0;
        self.severity_filter = SeverityFilter::All;
        self.restore_selected_alert(focused);
    }

    fn alert_count(&self) -> usize {
        self.filtered_alerts().len()
    }

    fn average_u64(values: &[u64]) -> u64 {
        if values.is_empty() {
            return 0;
        }
        let total: u64 = values.iter().sum();
        let count_u64 = u64::try_from(values.len()).unwrap_or(1);
        total
            .saturating_add(count_u64 / 2)
            .saturating_div(count_u64)
    }

    fn eta_seconds(backlog: u64, throughput_eps: f64) -> u64 {
        if !throughput_eps.is_finite() || throughput_eps <= 0.0 {
            return u64::MAX;
        }
        let backlog_f64 = f64::from(u32::try_from(backlog).unwrap_or(u32::MAX));
        let eta = (backlog_f64 / throughput_eps).round();
        if !eta.is_finite() || eta < 0.0 {
            return u64::MAX;
        }
        let max_u64_f64 = f64::from(u32::MAX);
        if eta > max_u64_f64 {
            return u64::MAX;
        }
        format!("{eta:.0}").parse::<u64>().unwrap_or(u64::MAX)
    }

    fn project_slo_rows(&self) -> Vec<SloProjectRow> {
        let fleet = self.state.fleet();
        if fleet.instances.is_empty() {
            return Vec::new();
        }

        let mut grouped: BTreeMap<String, Vec<&crate::state::InstanceInfo>> = BTreeMap::new();
        for instance in &fleet.instances {
            grouped
                .entry(instance.project.clone())
                .or_default()
                .push(instance);
        }

        let total_instances = fleet.instances.len();
        let throughput_eps = self
            .state
            .control_plane_metrics()
            .event_throughput_eps
            .max(0.1);

        let mut rows = grouped
            .into_iter()
            .map(|(project, instances)| {
                let instance_count = instances.len();
                let unhealthy_count = instances
                    .iter()
                    .filter(|instance| !instance.healthy)
                    .count();
                let pending_jobs: u64 =
                    instances.iter().map(|instance| instance.pending_jobs).sum();

                let p95_values = instances
                    .iter()
                    .filter_map(|instance| {
                        fleet
                            .search_metrics
                            .get(&instance.id)
                            .map(|metrics| metrics.p95_latency_us)
                    })
                    .collect::<Vec<_>>();
                let p95_latency_us = Self::average_u64(&p95_values);

                let instance_count_f64 = f64::from(u32::try_from(instance_count).unwrap_or(1));
                let total_instance_weight = f64::from(u32::try_from(total_instances).unwrap_or(1));
                let throughput_share =
                    (throughput_eps * (instance_count_f64 / total_instance_weight)).max(0.1);
                let backlog_eta_s = Self::eta_seconds(pending_jobs, throughput_share);

                let unhealthy_ratio = if instance_count == 0 {
                    0.0
                } else {
                    let unhealthy_f64 = f64::from(u32::try_from(unhealthy_count).unwrap_or(0));
                    unhealthy_f64 / instance_count_f64
                };
                let latency_ratio = if p95_latency_us == 0 {
                    0.0
                } else {
                    f64::from(u32::try_from(p95_latency_us).unwrap_or(u32::MAX))
                        / f64::from(u32::try_from(TARGET_P95_US).unwrap_or(1))
                };
                let latency_burn = (latency_ratio - 1.0).max(0.0);
                let backlog_burn =
                    (f64::from(u32::try_from(pending_jobs).unwrap_or(u32::MAX)) / 2_000.0).min(2.0);
                let burn_ratio = latency_burn + (unhealthy_ratio * 1.5) + (backlog_burn * 0.5);
                let remaining_ratio = (1.0 - burn_ratio).clamp(0.0, 1.0);

                let status = if burn_ratio >= BURN_CRITICAL {
                    ControlPlaneHealth::Critical
                } else if burn_ratio >= BURN_WARN {
                    ControlPlaneHealth::Degraded
                } else {
                    ControlPlaneHealth::Healthy
                };

                let saturation_risk = if backlog_eta_s >= 3_600 || burn_ratio >= BURN_CRITICAL {
                    "high"
                } else if backlog_eta_s >= 900 || burn_ratio >= BURN_WARN {
                    "elevated"
                } else {
                    "low"
                };

                SloProjectRow {
                    project,
                    instance_count,
                    unhealthy_count,
                    pending_jobs,
                    p95_latency_us,
                    burn_ratio,
                    remaining_ratio,
                    backlog_eta_s,
                    saturation_risk,
                    status,
                }
            })
            .collect::<Vec<_>>();

        rows.sort_by(|left, right| {
            right
                .burn_ratio
                .total_cmp(&left.burn_ratio)
                .then_with(|| right.pending_jobs.cmp(&left.pending_jobs))
                .then_with(|| left.project.cmp(&right.project))
        });

        rows
    }

    fn fleet_rollup_row(project_rows: &[SloProjectRow]) -> Option<SloProjectRow> {
        if project_rows.is_empty() {
            return None;
        }

        let total_instances: usize = project_rows.iter().map(|row| row.instance_count).sum();
        let total_unhealthy: usize = project_rows.iter().map(|row| row.unhealthy_count).sum();
        let total_pending: u64 = project_rows.iter().map(|row| row.pending_jobs).sum();

        let total_instances_u64 = u64::try_from(total_instances).unwrap_or(1);
        let weighted_p95 = project_rows
            .iter()
            .map(|row| {
                let weight = u64::try_from(row.instance_count).unwrap_or(0);
                row.p95_latency_us.saturating_mul(weight)
            })
            .sum::<u64>()
            .saturating_add(total_instances_u64 / 2)
            .saturating_div(total_instances_u64);

        let total_instance_weight = f64::from(u32::try_from(total_instances).unwrap_or(1));
        let weighted_burn = project_rows
            .iter()
            .map(|row| {
                let weight = f64::from(u32::try_from(row.instance_count).unwrap_or(0));
                row.burn_ratio * weight
            })
            .sum::<f64>()
            / total_instance_weight;

        let remaining_ratio = (1.0 - weighted_burn).clamp(0.0, 1.0);
        let backlog_eta_s = project_rows
            .iter()
            .map(|row| row.backlog_eta_s)
            .max()
            .unwrap_or(0);

        let saturation_risk = if project_rows
            .iter()
            .any(|row| matches!(row.saturation_risk, "high"))
        {
            "high"
        } else if project_rows
            .iter()
            .any(|row| matches!(row.saturation_risk, "elevated"))
        {
            "elevated"
        } else {
            "low"
        };

        let status = if weighted_burn >= BURN_CRITICAL {
            ControlPlaneHealth::Critical
        } else if weighted_burn >= BURN_WARN {
            ControlPlaneHealth::Degraded
        } else {
            ControlPlaneHealth::Healthy
        };

        Some(SloProjectRow {
            project: "fleet".to_owned(),
            instance_count: total_instances,
            unhealthy_count: total_unhealthy,
            pending_jobs: total_pending,
            p95_latency_us: weighted_p95,
            burn_ratio: weighted_burn,
            remaining_ratio,
            backlog_eta_s,
            saturation_risk,
            status,
        })
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
            "filters: project={project}, severity={}, reason={reason}, host={host} | keys: p/s/r/h/x, g project, l stream, t timeline",
            self.severity_filter.label()
        )
    }

    fn alerts_summary_line(alerts: &[AlertRow]) -> String {
        if alerts.is_empty() {
            return "alerts: no lifecycle alerts for current filter".to_owned();
        }

        let critical = alerts
            .iter()
            .filter(|row| matches!(row.severity, AlertSeverity::Critical))
            .count();
        let warn = alerts
            .iter()
            .filter(|row| matches!(row.severity, AlertSeverity::Warn))
            .count();
        let suppressed = alerts
            .iter()
            .filter(|row| matches!(row.suppression_state, "suppressed"))
            .count();
        let active = alerts.len().saturating_sub(suppressed);

        format!(
            "alerts: total={} active={active} critical={critical} warn={warn} suppressed={suppressed}",
            alerts.len()
        )
    }

    fn slo_summary_line(project_rows: &[SloProjectRow]) -> String {
        let Some(fleet_row) = Self::fleet_rollup_row(project_rows) else {
            return "slo fleet: unavailable (no instances)".to_owned();
        };

        let remaining_pct = (fleet_row.remaining_ratio * 100.0).round();
        format!(
            "slo fleet: burn={:.2} remaining={remaining_pct:.0}% p95={}us/{}us status={}",
            fleet_row.burn_ratio, fleet_row.p95_latency_us, TARGET_P95_US, fleet_row.status
        )
    }

    fn capacity_summary_line(&self, project_rows: &[SloProjectRow]) -> String {
        let metrics = self.state.control_plane_metrics();
        let pending_total = self.state.fleet().total_pending_jobs();
        let throughput = metrics.event_throughput_eps.max(0.1);
        let clear_eta_s = Self::eta_seconds(pending_total, throughput);
        let rss_pct = metrics.rss_utilization() * 100.0;
        let storage_pct = metrics.storage_utilization() * 100.0;

        let saturation_risk = if project_rows
            .iter()
            .any(|row| matches!(row.saturation_risk, "high"))
            || clear_eta_s >= 3_600
            || rss_pct >= 90.0
            || metrics.ingestion_lag_events >= 10_000
        {
            "high"
        } else if project_rows
            .iter()
            .any(|row| matches!(row.saturation_risk, "elevated"))
            || clear_eta_s >= 900
            || rss_pct >= 75.0
            || metrics.ingestion_lag_events >= 1_000
        {
            "elevated"
        } else {
            "low"
        };

        format!(
            "capacity: backlog={pending_total} jobs clear_eta={clear_eta_s}s throughput={throughput:.2}eps rss={rss_pct:.1}% storage={storage_pct:.1}% lag={} risk={saturation_risk}",
            metrics.ingestion_lag_events
        )
    }

    fn selected_context_line(&self, alerts: &[AlertRow]) -> String {
        if let Some(alert) = alerts.get(self.selected_row) {
            return format!(
                "focus: ts={} project={} host={} instance={} severity={} reason={} suppression={}",
                alert.ts_ms,
                alert.project,
                alert.host,
                alert.instance_id,
                alert.severity.label(),
                alert.reason_code,
                alert.suppression_state
            );
        }
        "focus: none".to_owned()
    }

    fn build_alert_rows(&self, alerts: &[AlertRow]) -> Vec<Row<'static>> {
        alerts
            .iter()
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
                    row.confidence.to_string(),
                    row.suppression_state.to_owned(),
                    row.reason_code.clone(),
                ])
                .style(style)
            })
            .collect()
    }

    fn build_slo_rows(project_rows: &[SloProjectRow]) -> Vec<Row<'static>> {
        let mut rows = project_rows.to_vec();
        if let Some(fleet) = Self::fleet_rollup_row(project_rows) {
            rows.insert(0, fleet);
        }

        rows.into_iter()
            .map(|row| {
                let mut style = Style::default();
                style = match row.status {
                    ControlPlaneHealth::Healthy => style.fg(Color::Green),
                    ControlPlaneHealth::Degraded => style.fg(Color::Yellow),
                    ControlPlaneHealth::Critical => style.fg(Color::Red),
                };
                Row::new(vec![
                    row.project,
                    row.instance_count.to_string(),
                    row.unhealthy_count.to_string(),
                    row.pending_jobs.to_string(),
                    row.p95_latency_us.to_string(),
                    format!("{:.2}", row.burn_ratio),
                    format!("{:.0}%", row.remaining_ratio * 100.0),
                    row.backlog_eta_s.to_string(),
                    row.saturation_risk.to_owned(),
                    row.status.to_string(),
                ])
                .style(style)
            })
            .collect()
    }
}

impl Default for AlertsSloScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for AlertsSloScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Alerts + SLO + Capacity"
    }

    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let alerts = self.filtered_alerts();
        let project_rows = self.project_slo_rows();

        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Min(8),
                Constraint::Length(9),
            ])
            .split(area);

        let summary = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    "Alerts/SLO: ",
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(Self::alerts_summary_line(&alerts)),
            ]),
            Line::from(Self::slo_summary_line(&project_rows)),
            Line::from(self.capacity_summary_line(&project_rows)),
            Line::from(self.filter_summary()),
            Line::from(self.selected_context_line(&alerts)),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Alerts + SLO + Capacity "),
        );
        frame.render_widget(summary, chunks[0]);

        let alerts_table = Table::new(
            self.build_alert_rows(&alerts),
            [
                Constraint::Length(12),
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Length(18),
                Constraint::Length(8),
                Constraint::Length(6),
                Constraint::Length(13),
                Constraint::Min(24),
            ],
        )
        .header(
            Row::new(vec![
                "Timestamp",
                "Project",
                "Host",
                "Instance",
                "Severity",
                "Conf",
                "Suppression",
                "Reason",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Active Alerts "),
        );
        frame.render_widget(alerts_table, chunks[1]);

        let slo_table = Table::new(
            Self::build_slo_rows(&project_rows),
            [
                Constraint::Length(12),
                Constraint::Length(6),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec![
                "Scope",
                "Inst",
                "Unhealthy",
                "Pending",
                "P95us",
                "Burn",
                "Remain",
                "ETA(s)",
                "Risk",
                "Status",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" SLO + Capacity by Scope "),
        );
        frame.render_widget(slo_table, chunks[2]);
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
                    let count = self.alert_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('p') => {
                    self.cycle_project_filter();
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('s') => {
                    self.cycle_severity_filter();
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
        "alert"
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
        ControlPlaneMetrics, FleetSnapshot, InstanceInfo, LifecycleEvent, ResourceMetrics,
        SearchMetrics,
    };

    #[allow(clippy::too_many_lines)]
    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "host-a:alpha-1".to_owned(),
                    project: "alpha".to_owned(),
                    pid: Some(101),
                    healthy: true,
                    doc_count: 1_000,
                    pending_jobs: 20,
                },
                InstanceInfo {
                    id: "host-a:alpha-2".to_owned(),
                    project: "alpha".to_owned(),
                    pid: Some(102),
                    healthy: false,
                    doc_count: 1_200,
                    pending_jobs: 240,
                },
                InstanceInfo {
                    id: "host-b:beta-1".to_owned(),
                    project: "beta".to_owned(),
                    pid: Some(201),
                    healthy: true,
                    doc_count: 900,
                    pending_jobs: 8,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:alpha-2".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Degraded,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 4_000,
                    attribution_confidence_score: 72,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-b:beta-1".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stopped,
                    reason_code: "lifecycle.discovery.stop".to_owned(),
                    at_ms: 3_500,
                    attribution_confidence_score: 80,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-a:alpha-1".to_owned(),
                    from: LifecycleState::Recovering,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.recovered".to_owned(),
                    at_ms: 3_000,
                    attribution_confidence_score: 96,
                    attribution_collision: false,
                },
            ],
            ..FleetSnapshot::default()
        };

        fleet.search_metrics.insert(
            "host-a:alpha-1".to_owned(),
            SearchMetrics {
                total_searches: 600,
                avg_latency_us: 900,
                p95_latency_us: 2_300,
                refined_count: 210,
            },
        );
        fleet.search_metrics.insert(
            "host-a:alpha-2".to_owned(),
            SearchMetrics {
                total_searches: 800,
                avg_latency_us: 2_100,
                p95_latency_us: 7_800,
                refined_count: 190,
            },
        );
        fleet.search_metrics.insert(
            "host-b:beta-1".to_owned(),
            SearchMetrics {
                total_searches: 500,
                avg_latency_us: 1_100,
                p95_latency_us: 1_900,
                refined_count: 155,
            },
        );

        fleet.resources.insert(
            "host-a:alpha-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 20.0,
                memory_bytes: 500 * 1024 * 1024,
                io_read_bytes: 10_000,
                io_write_bytes: 6_000,
            },
        );
        fleet.resources.insert(
            "host-a:alpha-2".to_owned(),
            ResourceMetrics {
                cpu_percent: 76.0,
                memory_bytes: 950 * 1024 * 1024,
                io_read_bytes: 20_000,
                io_write_bytes: 10_000,
            },
        );
        fleet.resources.insert(
            "host-b:beta-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 15.0,
                memory_bytes: 300 * 1024 * 1024,
                io_read_bytes: 6_000,
                io_write_bytes: 3_000,
            },
        );

        state.update_fleet(fleet);
        state.update_control_plane(ControlPlaneMetrics {
            ingestion_lag_events: 2_200,
            storage_bytes: 700 * 1024 * 1024,
            storage_limit_bytes: 1024 * 1024 * 1024,
            frame_time_ms: 16.0,
            discovery_latency_ms: 420,
            event_throughput_eps: 18.5,
            rss_bytes: 820 * 1024 * 1024,
            rss_limit_bytes: 1024 * 1024 * 1024,
            dead_letter_events: 3,
        });
        state
    }

    fn screen_context() -> ScreenContext {
        ScreenContext {
            active_screen: ScreenId::new("ops.alerts"),
            terminal_width: 120,
            terminal_height: 45,
            focused: true,
        }
    }

    #[test]
    fn alerts_screen_defaults() {
        let screen = AlertsSloScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.alerts"));
        assert_eq!(screen.title(), "Alerts + SLO + Capacity");
        assert_eq!(screen.semantic_role(), "alert");
    }

    #[test]
    fn selected_project_tracks_alert_focus() {
        let mut screen = AlertsSloScreen::new();
        screen.update_state(&sample_state());
        assert_eq!(screen.selected_project().as_deref(), Some("beta"));
    }

    #[test]
    fn selected_project_ignores_unknown_sentinel_rows() {
        let mut state = sample_state();
        let mut fleet = state.fleet().clone();
        fleet.lifecycle_events.insert(
            0,
            LifecycleEvent {
                instance_id: "orphan-host:missing-inst".to_owned(),
                from: LifecycleState::Started,
                to: LifecycleState::Stopped,
                reason_code: "lifecycle.instance.orphan".to_owned(),
                at_ms: 12_500,
                attribution_confidence_score: 40,
                attribution_collision: false,
            },
        );
        state.update_fleet(fleet);

        let mut screen = AlertsSloScreen::new();
        screen.update_state(&state);

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
                to: LifecycleState::Stopped,
                reason_code: "lifecycle.discovery.ready".to_owned(),
                at_ms: 13_000,
                attribution_confidence_score: 96,
                attribution_collision: false,
            },
        );
        state.update_fleet(fleet);

        let mut screen = AlertsSloScreen::new();
        screen.update_state(&state);

        assert_eq!(
            screen.selected_project().as_deref(),
            Some("unknown"),
            "real project names should not be dropped as sentinel values"
        );
    }

    #[test]
    fn alert_filters_cycle_and_reset() {
        let mut screen = AlertsSloScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let severity = InputEvent::Key(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&severity, &ctx), ScreenAction::Consumed);

        let project = InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&project, &ctx), ScreenAction::Consumed);

        let reset = InputEvent::Key(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&reset, &ctx), ScreenAction::Consumed);
        assert!(screen.alert_count() >= 3);
    }

    #[test]
    fn summaries_include_slo_and_capacity_signals() {
        let mut screen = AlertsSloScreen::new();
        screen.update_state(&sample_state());

        let alerts = screen.filtered_alerts();
        let project_rows = screen.project_slo_rows();

        let alerts_summary = AlertsSloScreen::alerts_summary_line(&alerts);
        let slo_summary = AlertsSloScreen::slo_summary_line(&project_rows);
        let capacity_summary = screen.capacity_summary_line(&project_rows);

        assert!(alerts_summary.contains("critical="));
        assert!(slo_summary.contains("slo fleet:"));
        assert!(slo_summary.contains("burn="));
        assert!(capacity_summary.contains("capacity:"));
        assert!(capacity_summary.contains("clear_eta="));
    }

    #[test]
    fn drilldown_keys_navigate_to_targets() {
        let mut screen = AlertsSloScreen::new();
        screen.set_project_screen_id(ScreenId::new("ops.project.custom"));
        screen.set_live_stream_screen_id(ScreenId::new("ops.stream.custom"));
        screen.set_timeline_screen_id(ScreenId::new("ops.timeline.custom"));
        screen.update_state(&sample_state());

        let ctx = screen_context();
        let project = InputEvent::Key(
            crossterm::event::KeyCode::Char('g'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&project, &ctx),
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
    fn filter_summary_contains_drilldown_hints() {
        let mut screen = AlertsSloScreen::new();
        screen.update_state(&sample_state());
        let summary = screen.filter_summary();
        assert!(summary.contains("g project"));
        assert!(summary.contains("l stream"));
        assert!(summary.contains("t timeline"));
    }
}
