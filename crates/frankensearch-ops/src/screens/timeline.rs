//! Action Timeline screen scaffold.
//!
//! Visualizes lifecycle transition events to support rapid triage.

use std::any::Any;
use std::collections::BTreeSet;

use ftui_core::geometry::Rect;
use ftui_layout::{Constraint, Flex};
use ftui_render::frame::Frame;
use ftui_style::Style;
use ftui_text::{Line, Span, Text};
use ftui_widgets::{
    Widget,
    block::Block,
    borders::{BorderType, Borders},
    paragraph::Paragraph,
    table::{Row, Table},
};

use frankensearch_core::LifecycleState;
use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{KeybindingHint, ScreenAction, ScreenContext, ScreenId};

use crate::data_source::TimeWindow;
use crate::state::{AppState, LifecycleEvent};
use crate::theme::SemanticPalette;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SeverityFilter {
    All,
    Info,
    Warn,
    Critical,
}

impl SeverityFilter {
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

    fn allows(self, severity: EventSeverity) -> bool {
        match self {
            Self::All => true,
            Self::Info => severity == EventSeverity::Info,
            Self::Warn => severity == EventSeverity::Warn,
            Self::Critical => severity == EventSeverity::Critical,
        }
    }
}

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
}

/// Timeline screen for lifecycle and operational transition events.
pub struct ActionTimelineScreen {
    id: ScreenId,
    state: AppState,
    project_filter_values: Vec<String>,
    reason_filter_values: Vec<String>,
    host_filter_values: Vec<String>,
    selected_row: usize,
    project_filter_index: usize,
    reason_filter_index: usize,
    host_filter_index: usize,
    severity_filter: SeverityFilter,
    project_screen_id: ScreenId,
    live_stream_screen_id: ScreenId,
    analytics_screen_id: ScreenId,
    palette: SemanticPalette,
}

const TIMELINE_KEYBINDINGS: &[KeybindingHint] = &[
    KeybindingHint {
        key: "j / Down",
        description: "Move selection down",
    },
    KeybindingHint {
        key: "k / Up",
        description: "Move selection up",
    },
    KeybindingHint {
        key: "p",
        description: "Cycle project filter",
    },
    KeybindingHint {
        key: "s",
        description: "Cycle severity filter",
    },
    KeybindingHint {
        key: "r",
        description: "Cycle reason filter",
    },
    KeybindingHint {
        key: "h",
        description: "Cycle host filter",
    },
    KeybindingHint {
        key: "x",
        description: "Reset filters",
    },
    KeybindingHint {
        key: "g / Enter",
        description: "Open project detail",
    },
    KeybindingHint {
        key: "l",
        description: "Open live stream",
    },
    KeybindingHint {
        key: "a",
        description: "Open analytics",
    },
];

impl ActionTimelineScreen {
    /// Create a new timeline screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.timeline"),
            state: AppState::new(),
            project_filter_values: vec!["all".to_owned()],
            reason_filter_values: vec!["all".to_owned()],
            host_filter_values: vec!["all".to_owned()],
            selected_row: 0,
            project_filter_index: 0,
            reason_filter_index: 0,
            host_filter_index: 0,
            severity_filter: SeverityFilter::All,
            project_screen_id: ScreenId::new("ops.project"),
            live_stream_screen_id: ScreenId::new("ops.live_stream"),
            analytics_screen_id: ScreenId::new("ops.analytics"),
            palette: SemanticPalette::dark(),
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

    /// Update the historical-analytics drilldown destination.
    pub fn set_analytics_screen_id(&mut self, id: ScreenId) {
        self.analytics_screen_id = id;
    }

    /// Update timeline data from shared app state.
    pub fn update_state(&mut self, state: &AppState) {
        let focused = self.selected_event_key();
        let (project_filter, reason_filter, host_filter) = self.selected_filter_values();
        self.state = state.clone();
        self.rebuild_filter_values();
        self.restore_filter_indices(&project_filter, &reason_filter, &host_filter);
        self.clamp_filter_indices();
        self.restore_selected_event(focused);
    }

    pub const fn set_palette(&mut self, palette: SemanticPalette) {
        self.palette = palette;
    }

    fn all_events(&self) -> Vec<LifecycleEvent> {
        let mut events = self.state.fleet().lifecycle_events.clone();
        events.sort_by(|left, right| {
            right
                .at_ms
                .cmp(&left.at_ms)
                .then_with(|| left.instance_id.cmp(&right.instance_id))
        });
        events
    }

    fn filtered_events(&self) -> Vec<LifecycleEvent> {
        let project_filters = self.project_filters();
        let reason_filters = self.reason_filters();
        let host_filters = self.host_filters();

        let project_filter = project_filters
            .get(self.project_filter_index)
            .map(String::as_str)
            .filter(|value| *value != "all");
        let reason_filter = reason_filters
            .get(self.reason_filter_index)
            .map(String::as_str)
            .filter(|value| *value != "all");
        let host_filter = host_filters
            .get(self.host_filter_index)
            .map(String::as_str)
            .filter(|value| *value != "all");

        self.all_events()
            .into_iter()
            .filter(|event| {
                project_filter.is_none_or(|project| {
                    if project.eq_ignore_ascii_case("unknown") {
                        self.project_for_instance(&event.instance_id).is_none_or(
                            |instance_project| instance_project.eq_ignore_ascii_case("unknown"),
                        )
                    } else {
                        self.project_for_instance(&event.instance_id).is_some_and(
                            |instance_project| instance_project.eq_ignore_ascii_case(project),
                        )
                    }
                })
            })
            .filter(|event| self.severity_filter.allows(Self::event_severity(event)))
            .filter(|event| reason_filter.is_none_or(|reason| event.reason_code == reason))
            .filter(|event| {
                host_filter.is_none_or(|host| {
                    Self::host_bucket(&event.instance_id).eq_ignore_ascii_case(host)
                })
            })
            .collect()
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

    fn rebuild_filter_values(&mut self) {
        let mut values = vec!["all".to_owned()];
        let mut projects: BTreeSet<_> = self
            .state
            .fleet()
            .instances
            .iter()
            .map(|instance| instance.project.clone())
            .collect();
        projects.extend(self.state.fleet().lifecycle_events.iter().map(|event| {
            self.project_for_instance(&event.instance_id)
                .unwrap_or("unknown")
                .to_owned()
        }));
        values.extend(projects);
        self.project_filter_values = values;

        let mut values = vec!["all".to_owned()];
        let reasons: BTreeSet<_> = self
            .state
            .fleet()
            .lifecycle_events
            .iter()
            .map(|event| event.reason_code.clone())
            .collect();
        values.extend(reasons);
        self.reason_filter_values = values;

        let mut values = vec!["all".to_owned()];
        let hosts: BTreeSet<_> = self
            .state
            .fleet()
            .lifecycle_events
            .iter()
            .map(|event| Self::host_bucket(&event.instance_id))
            .collect();
        values.extend(hosts);
        self.host_filter_values = values;
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

    fn selected_event_key(&self) -> Option<(u64, String, String)> {
        self.filtered_events().get(self.selected_row).map(|event| {
            (
                event.at_ms,
                event.instance_id.clone(),
                event.reason_code.clone(),
            )
        })
    }

    fn restore_selected_event(&mut self, key: Option<(u64, String, String)>) {
        let events = self.filtered_events();
        if events.is_empty() {
            self.selected_row = 0;
            return;
        }

        if let Some((at_ms, instance_id, reason_code)) = key
            && let Some(index) = events.iter().position(|event| {
                event.at_ms == at_ms
                    && event.instance_id == instance_id
                    && event.reason_code == reason_code
            })
        {
            self.selected_row = index;
            return;
        }

        if self.selected_row >= events.len() {
            self.selected_row = events.len().saturating_sub(1);
        }
    }

    fn cycle_project_filter(&mut self) {
        let focused = self.selected_event_key();
        let len = self.project_filters().len();
        if len > 0 {
            self.project_filter_index = (self.project_filter_index + 1) % len;
        }
        self.restore_selected_event(focused);
    }

    fn cycle_reason_filter(&mut self) {
        let focused = self.selected_event_key();
        let len = self.reason_filters().len();
        if len > 0 {
            self.reason_filter_index = (self.reason_filter_index + 1) % len;
        }
        self.restore_selected_event(focused);
    }

    fn cycle_host_filter(&mut self) {
        let focused = self.selected_event_key();
        let len = self.host_filters().len();
        if len > 0 {
            self.host_filter_index = (self.host_filter_index + 1) % len;
        }
        self.restore_selected_event(focused);
    }

    fn cycle_severity_filter(&mut self) {
        let focused = self.selected_event_key();
        self.severity_filter = self.severity_filter.next();
        self.restore_selected_event(focused);
    }

    fn reset_filters(&mut self) {
        let focused = self.selected_event_key();
        self.project_filter_index = 0;
        self.reason_filter_index = 0;
        self.host_filter_index = 0;
        self.severity_filter = SeverityFilter::All;
        self.restore_selected_event(focused);
    }

    fn ratio_percent_u64(numer: u64, denom: u64) -> u8 {
        if denom == 0 {
            return 0;
        }
        let rounded = numer
            .saturating_mul(100)
            .saturating_add(denom / 2)
            .saturating_div(denom)
            .min(100);
        u8::try_from(rounded).unwrap_or(100)
    }

    fn spark_char(percent: u8) -> char {
        const BINS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let idx = (u16::from(percent).saturating_mul(7).saturating_add(50)) / 100;
        BINS[usize::from(idx.min(7))]
    }

    fn sparkline(values: &[u8]) -> String {
        values
            .iter()
            .map(|value| Self::spark_char(*value))
            .collect()
    }

    fn event_pulse(event: &LifecycleEvent, newest_ts: u64, span_ms: u64) -> String {
        let severity_pressure = match Self::event_severity(event) {
            EventSeverity::Info => 25,
            EventSeverity::Warn => 65,
            EventSeverity::Critical => 100,
        };
        let collision_pressure = if event.attribution_collision { 100 } else { 20 };
        let confidence_pressure = event.attribution_confidence_score;
        let event_age = newest_ts.saturating_sub(event.at_ms);
        let recency_pressure =
            100u8.saturating_sub(Self::ratio_percent_u64(event_age, span_ms.max(1)));
        Self::sparkline(&[
            severity_pressure,
            collision_pressure,
            confidence_pressure,
            recency_pressure,
        ])
    }

    fn severity_mix_line(events: &[LifecycleEvent]) -> String {
        let mut info = 0usize;
        let mut warn = 0usize;
        let mut critical = 0usize;
        for event in events {
            match Self::event_severity(event) {
                EventSeverity::Info => info = info.saturating_add(1),
                EventSeverity::Warn => warn = warn.saturating_add(1),
                EventSeverity::Critical => critical = critical.saturating_add(1),
            }
        }
        format!(
            "severity mix: critical={critical} warn={warn} info={info} | rows={}",
            events.len()
        )
    }

    fn timeline_density_line(events: &[LifecycleEvent]) -> String {
        if events.is_empty() {
            return "density: (no events)".to_owned();
        }
        let newest = events.iter().map(|event| event.at_ms).max().unwrap_or(0);
        let oldest = events
            .iter()
            .map(|event| event.at_ms)
            .min()
            .unwrap_or(newest);
        let span = newest.saturating_sub(oldest);
        if span == 0 {
            return format!("density: {}", Self::sparkline(&[100]));
        }
        let mut buckets = [0u64; 12];
        for event in events {
            let from_oldest = event.at_ms.saturating_sub(oldest);
            let idx_u64 = from_oldest.saturating_mul(11).saturating_div(span).min(11);
            let idx = usize::try_from(idx_u64).unwrap_or(11);
            buckets[idx] = buckets[idx].saturating_add(1);
        }
        let max_bucket = buckets.iter().copied().max().unwrap_or(0).max(1);
        let normalized: Vec<u8> = buckets
            .iter()
            .map(|count| Self::ratio_percent_u64(*count, max_bucket))
            .collect();
        format!("density: {}", Self::sparkline(&normalized))
    }

    fn incident_badge_line(&self, events: &[LifecycleEvent]) -> Line {
        let status = if events
            .iter()
            .any(|event| matches!(Self::event_severity(event), EventSeverity::Critical))
        {
            ("incident=critical", self.palette.style_error().bold())
        } else if events
            .iter()
            .any(|event| matches!(Self::event_severity(event), EventSeverity::Warn))
        {
            ("incident=degraded", self.palette.style_warning().bold())
        } else if events.is_empty() {
            ("incident=idle", self.palette.style_muted())
        } else {
            ("incident=stable", self.palette.style_success().bold())
        };
        Line::from_spans(vec![
            Span::styled("status: ", self.palette.style_muted()),
            Span::styled(status.0, status.1),
            Span::raw(" "),
            Span::styled(
                format!("severity_filter={}", self.severity_filter.label()),
                self.palette.style_muted(),
            ),
        ])
    }

    fn severity_pills_line(&self) -> Line {
        let mut spans = vec![
            Span::styled("filters:", self.palette.style_muted()),
            Span::raw(" "),
        ];
        let filters = [
            ("all", SeverityFilter::All),
            ("info", SeverityFilter::Info),
            ("warn", SeverityFilter::Warn),
            ("critical", SeverityFilter::Critical),
        ];
        for (index, (label, filter)) in filters.iter().enumerate() {
            if index > 0 {
                spans.push(Span::raw(" "));
            }
            let style = if *filter == self.severity_filter {
                self.palette.style_highlight().bold()
            } else {
                self.palette.style_muted()
            };
            spans.push(Span::styled(format!("[{label}]"), style));
        }
        Line::from_spans(spans)
    }

    fn render_compact(&self, frame: &mut Frame, area: Rect) {
        let events = self.filtered_events();
        let mut lines = vec![
            Line::from_spans(vec![
                Span::styled("Timeline: ", Style::new().bold()),
                Span::raw(Self::timeline_summary(&events)),
            ]),
            self.incident_badge_line(&events),
            Line::from(Self::severity_mix_line(&events)),
            Line::from(Self::timeline_density_line(&events)),
            self.severity_pills_line(),
            Line::from(self.selected_context_summary()),
        ];
        let visible_lines = usize::from(area.height.saturating_sub(2));
        if visible_lines > 0 {
            lines.truncate(visible_lines);
        } else {
            lines.truncate(1);
        }
        Paragraph::new(Text::from_lines(lines))
            .block(
                Block::new()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(self.palette.style_border())
                    .title(" Action Timeline "),
            )
            .render(area, frame);
    }

    fn build_rows(&self) -> Vec<Row> {
        let events = self.filtered_events();
        let newest_ts = events.first().map_or(0, |event| event.at_ms);
        let oldest_ts = events.last().map_or(newest_ts, |event| event.at_ms);
        let span_ms = newest_ts.saturating_sub(oldest_ts).max(1);
        events
            .into_iter()
            .enumerate()
            .map(|(index, event)| {
                let severity = Self::event_severity(&event);
                let severity_badge = match severity {
                    EventSeverity::Info => "[I] info",
                    EventSeverity::Warn => "[W] warn",
                    EventSeverity::Critical => "[C] critical",
                };
                let transition = format!("{:?}->{:?}", event.from, event.to);
                let confidence = if event.attribution_collision {
                    format!("{}!", event.attribution_confidence_score)
                } else {
                    event.attribution_confidence_score.to_string()
                };
                let pulse = Self::event_pulse(&event, newest_ts, span_ms);
                let project = self
                    .project_for_instance(&event.instance_id)
                    .unwrap_or("unknown")
                    .to_owned();
                let host = Self::host_bucket(&event.instance_id);
                let style = if index == self.selected_row {
                    self.palette.style_highlight().bold()
                } else {
                    match severity {
                        EventSeverity::Info => self.palette.style_row_muted(index),
                        EventSeverity::Warn => self.palette.style_row_warning(index),
                        EventSeverity::Critical => self.palette.style_row_error(index),
                    }
                };
                Row::new(vec![
                    event.at_ms.to_string(),
                    project,
                    host,
                    event.instance_id,
                    severity_badge.to_owned(),
                    transition,
                    event.reason_code,
                    confidence,
                    pulse,
                ])
                .style(style)
            })
            .collect()
    }

    fn rolling_counter_summary(events: &[LifecycleEvent]) -> String {
        let now_ms = events.first().map_or(0, |event| event.at_ms);
        let counts = TimeWindow::ALL
            .iter()
            .map(|window| {
                let window_ms = window.seconds().saturating_mul(1_000);
                let window_start = now_ms.saturating_sub(window_ms);
                let count = events
                    .iter()
                    .filter(|event| event.at_ms >= window_start)
                    .count();
                format!("{}={count}", window.label())
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("windows: {counts}")
    }

    fn timeline_summary(events: &[LifecycleEvent]) -> String {
        if events.is_empty() {
            return "no lifecycle events".to_owned();
        }
        let collisions = events
            .iter()
            .filter(|event| event.attribution_collision)
            .count();
        format!(
            "{} events | {} attribution collisions",
            events.len(),
            collisions
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
            "filters: project={project}, severity={}, reason={reason}, host={host} | keys: p/s/r/h/x g/l/a",
            self.severity_filter.label()
        )
    }

    fn event_count(&self) -> usize {
        self.filtered_events().len()
    }

    fn stream_health_summary(&self) -> String {
        let metrics = self.state.control_plane_metrics();
        let health = self.state.control_plane_health();
        let reconnect_state = match (self.state.has_data(), health) {
            (false, _) => "reconnecting",
            (true, crate::state::ControlPlaneHealth::Critical) => "backoff",
            (true, crate::state::ControlPlaneHealth::Degraded) => "degraded",
            (true, crate::state::ControlPlaneHealth::Healthy) => "steady",
        };
        format!(
            "stream: health={health} reconnect={reconnect_state} lag={} drops={} throughput={:.2} eps",
            metrics.ingestion_lag_events, metrics.dead_letter_events, metrics.event_throughput_eps
        )
    }

    fn selected_context_summary(&self) -> String {
        if let Some(event) = self.filtered_events().get(self.selected_row) {
            let severity = Self::event_severity(event);
            let project = self
                .project_for_instance(&event.instance_id)
                .unwrap_or("unknown")
                .to_owned();
            let host = Self::host_bucket(&event.instance_id);
            return format!(
                "focus: ts={} project={} host={} instance={} severity={} reason={}",
                event.at_ms,
                project,
                host,
                event.instance_id,
                severity.label(),
                event.reason_code
            );
        }
        "focus: none".to_owned()
    }

    pub fn selected_project(&self) -> Option<String> {
        self.filtered_events()
            .get(self.selected_row)
            .and_then(|event| self.project_for_instance(&event.instance_id))
            .map(std::borrow::ToOwned::to_owned)
    }

    /// Selected reason code from the focused timeline row.
    #[must_use]
    pub fn selected_reason_code(&self) -> Option<String> {
        self.filtered_events()
            .get(self.selected_row)
            .map(|event| event.reason_code.clone())
    }

    /// Selected host bucket from the focused timeline row.
    #[must_use]
    pub fn selected_host(&self) -> Option<String> {
        self.filtered_events()
            .get(self.selected_row)
            .map(|event| Self::host_bucket(&event.instance_id))
    }

    #[cfg(test)]
    fn selected_instance_id(&self) -> Option<String> {
        self.filtered_events()
            .get(self.selected_row)
            .map(|event| event.instance_id.clone())
    }
}

impl Default for ActionTimelineScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for ActionTimelineScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Action Timeline"
    }

    fn render(&self, frame: &mut Frame, _ctx: &ScreenContext) {
        let p = &self.palette;
        let border_style = p.style_border();

        let area = frame.bounds();
        if area.width < 104 || area.height < 12 {
            self.render_compact(frame, area);
            return;
        }

        let events = self.filtered_events();
        let chunks = Flex::vertical()
            .constraints([Constraint::Fixed(10), Constraint::Min(6)])
            .split(area);

        let header = Paragraph::new(Text::from_lines(vec![
            Line::from_spans(vec![
                Span::styled("Timeline: ", Style::new().bold()),
                Span::raw(Self::timeline_summary(&events)),
            ]),
            self.incident_badge_line(&events),
            self.severity_pills_line(),
            Line::from(Self::severity_mix_line(&events)),
            Line::from(Self::timeline_density_line(&events)),
            Line::from(Self::rolling_counter_summary(&events)),
            Line::from(self.stream_health_summary()),
            Line::from(self.filter_summary()),
            Line::from(self.selected_context_summary()),
        ]))
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(border_style)
                .title(" Action Timeline "),
        );
        header.render(chunks[0], frame);

        let table = Table::new(
            self.build_rows(),
            [
                Constraint::Fixed(14),
                Constraint::Fixed(16),
                Constraint::Fixed(12),
                Constraint::Fixed(20),
                Constraint::Fixed(12),
                Constraint::Fixed(22),
                Constraint::Min(22),
                Constraint::Fixed(6),
                Constraint::Fixed(6),
            ],
        )
        .header(
            Row::new(vec![
                "Timestamp",
                "Project",
                "Host",
                "Instance",
                "Severity",
                "Transition",
                "Reason",
                "Attr",
                "Pulse",
            ])
            .style(Style::new().fg(p.accent).bold()),
        )
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(border_style)
                .title(" Events "),
        );
        table.render(chunks[1], frame);
    }

    fn handle_input(&mut self, event: &InputEvent, _ctx: &ScreenContext) -> ScreenAction {
        if let InputEvent::Key(code, _mods) = event {
            match code {
                ftui_core::event::KeyCode::Up | ftui_core::event::KeyCode::Char('k') => {
                    if self.selected_row > 0 {
                        self.selected_row -= 1;
                    }
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Down | ftui_core::event::KeyCode::Char('j') => {
                    let count = self.event_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('p') => {
                    self.cycle_project_filter();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('s') => {
                    self.cycle_severity_filter();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('r') => {
                    self.cycle_reason_filter();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('h') => {
                    self.cycle_host_filter();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('x') => {
                    self.reset_filters();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('g') | ftui_core::event::KeyCode::Enter => {
                    if self.selected_project().is_some() {
                        return ScreenAction::Navigate(self.project_screen_id.clone());
                    }
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('l') => {
                    if self.event_count() > 0 {
                        return ScreenAction::Navigate(self.live_stream_screen_id.clone());
                    }
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('a') => {
                    if self.event_count() > 0 {
                        return ScreenAction::Navigate(self.analytics_screen_id.clone());
                    }
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

    fn keybindings(&self) -> &'static [KeybindingHint] {
        TIMELINE_KEYBINDINGS
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

    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let fleet = crate::state::FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "host-a:inst-a".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(10),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "host-b:inst-b".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(11),
                    healthy: false,
                    doc_count: 20,
                    pending_jobs: 2,
                },
                crate::state::InstanceInfo {
                    id: "host-a:inst-c".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 30,
                    pending_jobs: 4,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:inst-a".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.heartbeat".to_owned(),
                    at_ms: 1_000_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:inst-b".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 970_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-a:inst-c".to_owned(),
                    from: LifecycleState::Recovering,
                    to: LifecycleState::Stopped,
                    reason_code: "lifecycle.discovery.stop".to_owned(),
                    at_ms: 100_000,
                    attribution_confidence_score: 60,
                    attribution_collision: false,
                },
            ],
            ..crate::state::FleetSnapshot::default()
        };
        state.update_fleet(fleet);
        state
    }

    fn screen_context() -> ScreenContext {
        ScreenContext {
            active_screen: ScreenId::new("ops.timeline"),
            terminal_width: 100,
            terminal_height: 40,
            focused: true,
        }
    }

    #[test]
    fn timeline_screen_defaults() {
        let screen = ActionTimelineScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.timeline"));
        assert_eq!(screen.title(), "Action Timeline");
        assert_eq!(screen.semantic_role(), "log");
    }

    #[test]
    fn timeline_rows_are_sorted_descending_by_timestamp() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let events = screen.all_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].instance_id, "host-a:inst-a");
        assert_eq!(events[1].instance_id, "host-b:inst-b");
        assert_eq!(events[2].instance_id, "host-a:inst-c");
    }

    #[test]
    fn timeline_navigation_is_bounded() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let down = InputEvent::Key(
            ftui_core::event::KeyCode::Down,
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 2);
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 2);

        let up = InputEvent::Key(
            ftui_core::event::KeyCode::Up,
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&up, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
    }

    #[test]
    fn timeline_filters_project_reason_and_host() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());

        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "proj-a")
            .expect("project filter should exist");
        let project_events = screen.filtered_events();
        assert_eq!(project_events.len(), 2);

        screen.project_filter_index = 0;
        screen.reason_filter_index = screen
            .reason_filters()
            .iter()
            .position(|value| value == "lifecycle.heartbeat_gap")
            .expect("reason filter should exist");
        let reason_events = screen.filtered_events();
        assert_eq!(reason_events.len(), 1);
        assert_eq!(reason_events[0].instance_id, "host-b:inst-b");

        screen.reason_filter_index = 0;
        screen.host_filter_index = screen
            .host_filters()
            .iter()
            .position(|value| value == "host-a")
            .expect("host filter should exist");
        let host_events = screen.filtered_events();
        assert_eq!(host_events.len(), 2);
    }

    #[test]
    fn timeline_severity_filter_is_applied() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());

        screen.severity_filter = SeverityFilter::Critical;
        let critical = screen.filtered_events();
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].instance_id, "host-a:inst-c");

        screen.severity_filter = SeverityFilter::Warn;
        let warn = screen.filtered_events();
        assert_eq!(warn.len(), 1);
        assert_eq!(warn[0].instance_id, "host-b:inst-b");
    }

    #[test]
    fn timeline_rolling_summary_counts_windows() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let events = screen.filtered_events();
        let summary = ActionTimelineScreen::rolling_counter_summary(&events);

        assert!(summary.contains("1m=2"));
        assert!(summary.contains("15m=3"));
        assert!(summary.contains("1w=3"));
    }

    #[test]
    fn timeline_filter_cycles_preserve_selected_context_when_possible() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        screen.selected_row = 2;
        assert_eq!(
            screen.selected_instance_id().as_deref(),
            Some("host-a:inst-c")
        );

        screen.cycle_project_filter(); // all -> proj-a (selected event still visible)
        assert_eq!(
            screen.selected_instance_id().as_deref(),
            Some("host-a:inst-c")
        );
    }

    #[test]
    fn timeline_context_summary_reports_selected_event_and_stream_health() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let focus = screen.selected_context_summary();
        let stream = screen.stream_health_summary();

        assert!(focus.contains("focus: ts="));
        assert!(focus.contains("reason="));
        assert!(stream.contains("stream: health="));
        assert!(stream.contains("lag="));
    }

    #[test]
    fn timeline_drilldown_keys_navigate_to_targets() {
        let mut screen = ActionTimelineScreen::new();
        screen.set_project_screen_id(ScreenId::new("ops.project.custom"));
        screen.set_live_stream_screen_id(ScreenId::new("ops.stream.custom"));
        screen.set_analytics_screen_id(ScreenId::new("ops.analytics.custom"));
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let project = InputEvent::Key(
            ftui_core::event::KeyCode::Char('g'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&project, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.project.custom"))
        );

        let project_enter = InputEvent::Key(
            ftui_core::event::KeyCode::Enter,
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&project_enter, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.project.custom"))
        );

        let stream = InputEvent::Key(
            ftui_core::event::KeyCode::Char('l'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&stream, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.stream.custom"))
        );

        let analytics = InputEvent::Key(
            ftui_core::event::KeyCode::Char('a'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&analytics, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.analytics.custom"))
        );
    }

    #[test]
    fn timeline_drilldown_keys_are_consumed_when_no_rows_exist() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&AppState::new());
        let ctx = screen_context();

        let project = InputEvent::Key(
            ftui_core::event::KeyCode::Char('g'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&project, &ctx), ScreenAction::Consumed);

        let project_enter = InputEvent::Key(
            ftui_core::event::KeyCode::Enter,
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&project_enter, &ctx),
            ScreenAction::Consumed
        );

        let stream = InputEvent::Key(
            ftui_core::event::KeyCode::Char('l'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&stream, &ctx), ScreenAction::Consumed);

        let analytics = InputEvent::Key(
            ftui_core::event::KeyCode::Char('a'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&analytics, &ctx),
            ScreenAction::Consumed
        );
    }

    #[test]
    fn timeline_update_state_preserves_filter_value_when_new_option_is_inserted() {
        let mut screen = ActionTimelineScreen::new();
        let mut state = sample_state();
        screen.update_state(&state);

        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "proj-b")
            .expect("project filter should exist");

        state.update_fleet(crate::state::FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "host-a:inst-a".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(10),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "host-b:inst-b".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(11),
                    healthy: false,
                    doc_count: 20,
                    pending_jobs: 2,
                },
                crate::state::InstanceInfo {
                    id: "host-a:inst-c".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 30,
                    pending_jobs: 4,
                },
                crate::state::InstanceInfo {
                    id: "host-z:inst-z".to_owned(),
                    project: "proj-aa".to_owned(),
                    pid: Some(13),
                    healthy: true,
                    doc_count: 8,
                    pending_jobs: 0,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:inst-a".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.heartbeat".to_owned(),
                    at_ms: 1_000_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:inst-b".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 970_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-a:inst-c".to_owned(),
                    from: LifecycleState::Recovering,
                    to: LifecycleState::Stopped,
                    reason_code: "lifecycle.discovery.stop".to_owned(),
                    at_ms: 100_000,
                    attribution_confidence_score: 60,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-z:inst-z".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.heartbeat".to_owned(),
                    at_ms: 950_000,
                    attribution_confidence_score: 88,
                    attribution_collision: false,
                },
            ],
            ..crate::state::FleetSnapshot::default()
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
    fn timeline_update_state_preserves_reason_and_host_filter_values_when_new_options_are_inserted()
    {
        let mut screen = ActionTimelineScreen::new();
        let mut state = AppState::new();
        state.update_fleet(crate::state::FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "host-a:inst-a".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(10),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 1,
                },
                crate::state::InstanceInfo {
                    id: "host-b:inst-b".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(11),
                    healthy: false,
                    doc_count: 20,
                    pending_jobs: 2,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:inst-a".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.heartbeat".to_owned(),
                    at_ms: 1_000_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:inst-b".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 970_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
            ],
            ..crate::state::FleetSnapshot::default()
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

        state.update_fleet(crate::state::FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "host-a:inst-a".to_owned(),
                    project: "proj-a".to_owned(),
                    pid: Some(10),
                    healthy: true,
                    doc_count: 10,
                    pending_jobs: 1,
                },
                crate::state::InstanceInfo {
                    id: "host-b:inst-b".to_owned(),
                    project: "proj-b".to_owned(),
                    pid: Some(11),
                    healthy: false,
                    doc_count: 20,
                    pending_jobs: 2,
                },
                crate::state::InstanceInfo {
                    id: "host-aa:inst-c".to_owned(),
                    project: "proj-c".to_owned(),
                    pid: Some(12),
                    healthy: true,
                    doc_count: 8,
                    pending_jobs: 0,
                },
            ],
            lifecycle_events: vec![
                LifecycleEvent {
                    instance_id: "host-a:inst-a".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.discovery.heartbeat".to_owned(),
                    at_ms: 1_000_000,
                    attribution_confidence_score: 90,
                    attribution_collision: false,
                },
                LifecycleEvent {
                    instance_id: "host-b:inst-b".to_owned(),
                    from: LifecycleState::Healthy,
                    to: LifecycleState::Stale,
                    reason_code: "lifecycle.heartbeat_gap".to_owned(),
                    at_ms: 970_000,
                    attribution_confidence_score: 70,
                    attribution_collision: true,
                },
                LifecycleEvent {
                    instance_id: "host-aa:inst-c".to_owned(),
                    from: LifecycleState::Started,
                    to: LifecycleState::Healthy,
                    reason_code: "lifecycle.anomaly.alpha".to_owned(),
                    at_ms: 950_000,
                    attribution_confidence_score: 88,
                    attribution_collision: false,
                },
            ],
            ..crate::state::FleetSnapshot::default()
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

    #[test]
    fn timeline_project_filter_supports_unknown_orphan_events() {
        let mut state = sample_state();
        let mut fleet = state.fleet().clone();
        fleet.lifecycle_events.push(LifecycleEvent {
            instance_id: "orphan-host:missing-inst".to_owned(),
            from: LifecycleState::Started,
            to: LifecycleState::Stale,
            reason_code: "lifecycle.instance.orphan".to_owned(),
            at_ms: 1_200_000,
            attribution_confidence_score: 40,
            attribution_collision: false,
        });
        state.update_fleet(fleet);

        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&state);
        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "unknown")
            .expect("unknown project filter should exist");

        let orphan_events = screen.filtered_events();
        assert_eq!(orphan_events.len(), 1);
        assert_eq!(orphan_events[0].instance_id, "orphan-host:missing-inst");
    }

    #[test]
    fn timeline_unknown_filter_includes_real_unknown_project_instances() {
        let mut state = sample_state();
        let mut fleet = state.fleet().clone();
        fleet.instances.push(crate::state::InstanceInfo {
            id: "host-u:unknown-1".to_owned(),
            project: "unknown".to_owned(),
            pid: Some(77),
            healthy: true,
            doc_count: 11,
            pending_jobs: 0,
        });
        fleet.lifecycle_events.push(LifecycleEvent {
            instance_id: "host-u:unknown-1".to_owned(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "lifecycle.discovery.heartbeat".to_owned(),
            at_ms: 1_300_000,
            attribution_confidence_score: 95,
            attribution_collision: false,
        });
        state.update_fleet(fleet);

        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&state);
        screen.project_filter_index = screen
            .project_filters()
            .iter()
            .position(|value| value == "unknown")
            .expect("unknown project filter should exist");

        let events = screen.filtered_events();
        assert!(
            events
                .iter()
                .any(|event| event.instance_id == "host-u:unknown-1"),
            "unknown filter should include events for real projects named unknown"
        );
    }

    // ─── bd-u944 tests begin ───

    // --- SeverityFilter ---

    #[test]
    fn severity_filter_next_cycles() {
        assert_eq!(SeverityFilter::All.next(), SeverityFilter::Info);
        assert_eq!(SeverityFilter::Info.next(), SeverityFilter::Warn);
        assert_eq!(SeverityFilter::Warn.next(), SeverityFilter::Critical);
        assert_eq!(SeverityFilter::Critical.next(), SeverityFilter::All);
    }

    #[test]
    fn severity_filter_label_all_variants() {
        assert_eq!(SeverityFilter::All.label(), "all");
        assert_eq!(SeverityFilter::Info.label(), "info");
        assert_eq!(SeverityFilter::Warn.label(), "warn");
        assert_eq!(SeverityFilter::Critical.label(), "critical");
    }

    #[test]
    fn severity_filter_allows_all_passes_everything() {
        assert!(SeverityFilter::All.allows(EventSeverity::Info));
        assert!(SeverityFilter::All.allows(EventSeverity::Warn));
        assert!(SeverityFilter::All.allows(EventSeverity::Critical));
    }

    #[test]
    fn severity_filter_info_only_passes_info() {
        assert!(SeverityFilter::Info.allows(EventSeverity::Info));
        assert!(!SeverityFilter::Info.allows(EventSeverity::Warn));
        assert!(!SeverityFilter::Info.allows(EventSeverity::Critical));
    }

    #[test]
    fn severity_filter_warn_only_passes_warn() {
        assert!(!SeverityFilter::Warn.allows(EventSeverity::Info));
        assert!(SeverityFilter::Warn.allows(EventSeverity::Warn));
        assert!(!SeverityFilter::Warn.allows(EventSeverity::Critical));
    }

    #[test]
    fn severity_filter_critical_only_passes_critical() {
        assert!(!SeverityFilter::Critical.allows(EventSeverity::Info));
        assert!(!SeverityFilter::Critical.allows(EventSeverity::Warn));
        assert!(SeverityFilter::Critical.allows(EventSeverity::Critical));
    }

    // --- EventSeverity ---

    #[test]
    fn event_severity_labels() {
        assert_eq!(EventSeverity::Info.label(), "info");
        assert_eq!(EventSeverity::Warn.label(), "warn");
        assert_eq!(EventSeverity::Critical.label(), "critical");
    }

    // --- host_bucket ---

    #[test]
    fn host_bucket_colon_separator() {
        assert_eq!(ActionTimelineScreen::host_bucket("host-a:inst-1"), "host-a");
    }

    #[test]
    fn host_bucket_dash_separator() {
        assert_eq!(ActionTimelineScreen::host_bucket("server-123"), "server");
    }

    #[test]
    fn host_bucket_no_separator() {
        assert_eq!(
            ActionTimelineScreen::host_bucket("standalone"),
            "standalone"
        );
    }

    #[test]
    fn host_bucket_colon_takes_priority_over_dash() {
        assert_eq!(
            ActionTimelineScreen::host_bucket("my-host:my-inst"),
            "my-host"
        );
    }

    // --- event_severity ---

    #[test]
    fn event_severity_stopped_is_critical() {
        let event = LifecycleEvent {
            instance_id: "x:y".to_owned(),
            from: LifecycleState::Healthy,
            to: LifecycleState::Stopped,
            reason_code: "test".to_owned(),
            at_ms: 0,
            attribution_confidence_score: 100,
            attribution_collision: false,
        };
        assert_eq!(
            ActionTimelineScreen::event_severity(&event),
            EventSeverity::Critical
        );
    }

    #[test]
    fn event_severity_stale_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".to_owned(),
            from: LifecycleState::Healthy,
            to: LifecycleState::Stale,
            reason_code: "test".to_owned(),
            at_ms: 0,
            attribution_confidence_score: 100,
            attribution_collision: false,
        };
        assert_eq!(
            ActionTimelineScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_degraded_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".to_owned(),
            from: LifecycleState::Healthy,
            to: LifecycleState::Degraded,
            reason_code: "test".to_owned(),
            at_ms: 0,
            attribution_confidence_score: 100,
            attribution_collision: false,
        };
        assert_eq!(
            ActionTimelineScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_recovering_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".to_owned(),
            from: LifecycleState::Degraded,
            to: LifecycleState::Recovering,
            reason_code: "test".to_owned(),
            at_ms: 0,
            attribution_confidence_score: 100,
            attribution_collision: false,
        };
        assert_eq!(
            ActionTimelineScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_collision_is_warn() {
        let event = LifecycleEvent {
            instance_id: "x:y".to_owned(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "test".to_owned(),
            at_ms: 0,
            attribution_confidence_score: 50,
            attribution_collision: true,
        };
        assert_eq!(
            ActionTimelineScreen::event_severity(&event),
            EventSeverity::Warn
        );
    }

    #[test]
    fn event_severity_healthy_no_collision_is_info() {
        let event = LifecycleEvent {
            instance_id: "x:y".to_owned(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "test".to_owned(),
            at_ms: 0,
            attribution_confidence_score: 100,
            attribution_collision: false,
        };
        assert_eq!(
            ActionTimelineScreen::event_severity(&event),
            EventSeverity::Info
        );
    }

    // --- timeline_summary ---

    #[test]
    fn timeline_summary_empty() {
        assert_eq!(
            ActionTimelineScreen::timeline_summary(&[]),
            "no lifecycle events"
        );
    }

    #[test]
    fn timeline_summary_with_data() {
        let events = vec![
            LifecycleEvent {
                instance_id: "a:1".to_owned(),
                from: LifecycleState::Started,
                to: LifecycleState::Healthy,
                reason_code: "test".to_owned(),
                at_ms: 1000,
                attribution_confidence_score: 90,
                attribution_collision: false,
            },
            LifecycleEvent {
                instance_id: "b:2".to_owned(),
                from: LifecycleState::Healthy,
                to: LifecycleState::Stale,
                reason_code: "test".to_owned(),
                at_ms: 900,
                attribution_confidence_score: 70,
                attribution_collision: true,
            },
        ];
        let summary = ActionTimelineScreen::timeline_summary(&events);
        assert!(summary.contains("2 events"));
        assert!(summary.contains("1 attribution collision"));
    }

    // --- rolling_counter_summary ---

    #[test]
    fn rolling_counter_summary_empty() {
        let summary = ActionTimelineScreen::rolling_counter_summary(&[]);
        assert!(summary.contains("windows:"));
    }

    // --- filter_summary ---

    #[test]
    fn filter_summary_default_all() {
        let screen = ActionTimelineScreen::new();
        let summary = screen.filter_summary();
        assert!(summary.contains("project=all"));
        assert!(summary.contains("severity=all"));
        assert!(summary.contains("reason=all"));
        assert!(summary.contains("host=all"));
    }

    // --- selected accessors ---

    #[test]
    fn selected_project_with_data() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        // First event (row 0, sorted desc by timestamp) is host-a:inst-a → proj-a
        assert_eq!(screen.selected_project().as_deref(), Some("proj-a"));
    }

    #[test]
    fn selected_project_empty_state() {
        let screen = ActionTimelineScreen::new();
        assert!(screen.selected_project().is_none());
    }

    #[test]
    fn selected_reason_code_with_data() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        assert_eq!(
            screen.selected_reason_code().as_deref(),
            Some("lifecycle.discovery.heartbeat")
        );
    }

    #[test]
    fn selected_host_with_data() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        assert_eq!(screen.selected_host().as_deref(), Some("host-a"));
    }

    // --- event_count ---

    #[test]
    fn event_count_with_data() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        assert_eq!(screen.event_count(), 3);
    }

    #[test]
    fn event_count_empty() {
        let screen = ActionTimelineScreen::new();
        assert_eq!(screen.event_count(), 0);
    }

    // --- selected_context_summary ---

    #[test]
    fn selected_context_summary_none_when_empty() {
        let screen = ActionTimelineScreen::new();
        assert_eq!(screen.selected_context_summary(), "focus: none");
    }

    // --- Default impl ---

    #[test]
    fn default_matches_new() {
        let default_screen = ActionTimelineScreen::default();
        let new_screen = ActionTimelineScreen::new();
        assert_eq!(default_screen.id(), new_screen.id());
        assert_eq!(default_screen.selected_row, new_screen.selected_row);
    }

    // --- as_any downcast ---

    #[test]
    fn as_any_downcast() {
        let screen = ActionTimelineScreen::new();
        let any_ref: &dyn Any = screen.as_any();
        assert!(any_ref.downcast_ref::<ActionTimelineScreen>().is_some());
    }

    // --- reset_filters ---

    #[test]
    fn reset_filters_restores_all() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        screen.project_filter_index = 1;
        screen.reason_filter_index = 1;
        screen.host_filter_index = 1;
        screen.severity_filter = SeverityFilter::Critical;

        screen.reset_filters();

        assert_eq!(screen.project_filter_index, 0);
        assert_eq!(screen.reason_filter_index, 0);
        assert_eq!(screen.host_filter_index, 0);
        assert_eq!(screen.severity_filter, SeverityFilter::All);
    }

    // --- k/j navigation ---

    #[test]
    fn k_key_navigates_up() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        screen.selected_row = 1;
        let ctx = screen_context();

        let k = InputEvent::Key(
            ftui_core::event::KeyCode::Char('k'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&k, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 0);
    }

    #[test]
    fn j_key_navigates_down() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let j = InputEvent::Key(
            ftui_core::event::KeyCode::Char('j'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&j, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
    }

    // --- unrecognized key ---

    #[test]
    fn unrecognized_key_returns_ignored() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let z = InputEvent::Key(
            ftui_core::event::KeyCode::Char('z'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&z, &ctx), ScreenAction::Ignored);
    }

    // --- severity cycle key ---

    #[test]
    fn s_key_cycles_severity_filter() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let s = InputEvent::Key(
            ftui_core::event::KeyCode::Char('s'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&s, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.severity_filter, SeverityFilter::Info);
    }

    // --- reason cycle key ---

    #[test]
    fn r_key_cycles_reason_filter() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let r = InputEvent::Key(
            ftui_core::event::KeyCode::Char('r'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&r, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.reason_filter_index, 1);
    }

    // --- host cycle key ---

    #[test]
    fn h_key_cycles_host_filter() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        let h = InputEvent::Key(
            ftui_core::event::KeyCode::Char('h'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&h, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.host_filter_index, 1);
    }

    // --- x key resets ---

    #[test]
    fn x_key_resets_filters() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = screen_context();

        screen.project_filter_index = 1;
        screen.severity_filter = SeverityFilter::Warn;

        let x = InputEvent::Key(
            ftui_core::event::KeyCode::Char('x'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&x, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.project_filter_index, 0);
        assert_eq!(screen.severity_filter, SeverityFilter::All);
    }

    // ─── bd-u944 tests end ───
}
