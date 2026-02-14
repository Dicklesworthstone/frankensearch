//! Action Timeline screen scaffold.
//!
//! Visualizes lifecycle transition events to support rapid triage.

use std::any::Any;
use std::collections::BTreeSet;

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

    const fn color(self) -> Color {
        match self {
            Self::Info => Color::Gray,
            Self::Warn => Color::Yellow,
            Self::Critical => Color::Red,
        }
    }
}

/// Timeline screen for lifecycle and operational transition events.
pub struct ActionTimelineScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
    project_filter_index: usize,
    reason_filter_index: usize,
    host_filter_index: usize,
    severity_filter: SeverityFilter,
}

impl ActionTimelineScreen {
    /// Create a new timeline screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.timeline"),
            state: AppState::new(),
            selected_row: 0,
            project_filter_index: 0,
            reason_filter_index: 0,
            host_filter_index: 0,
            severity_filter: SeverityFilter::All,
        }
    }

    /// Update timeline data from shared app state.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        self.clamp_filter_indices();
        self.clamp_selected_row();
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
                    self.project_for_instance(&event.instance_id)
                        .is_some_and(|instance_project| {
                            instance_project.eq_ignore_ascii_case(project)
                        })
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

    fn reason_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let reasons: BTreeSet<_> = self
            .state
            .fleet()
            .lifecycle_events
            .iter()
            .map(|event| event.reason_code.clone())
            .collect();
        values.extend(reasons);
        values
    }

    fn host_filters(&self) -> Vec<String> {
        let mut values = vec!["all".to_owned()];
        let hosts: BTreeSet<_> = self
            .state
            .fleet()
            .lifecycle_events
            .iter()
            .map(|event| Self::host_bucket(&event.instance_id))
            .collect();
        values.extend(hosts);
        values
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

    fn clamp_selected_row(&mut self) {
        let count = self.filtered_events().len();
        if count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= count {
            self.selected_row = count.saturating_sub(1);
        }
    }

    fn cycle_project_filter(&mut self) {
        let len = self.project_filters().len();
        if len > 0 {
            self.project_filter_index = (self.project_filter_index + 1) % len;
        }
        self.clamp_selected_row();
    }

    fn cycle_reason_filter(&mut self) {
        let len = self.reason_filters().len();
        if len > 0 {
            self.reason_filter_index = (self.reason_filter_index + 1) % len;
        }
        self.clamp_selected_row();
    }

    fn cycle_host_filter(&mut self) {
        let len = self.host_filters().len();
        if len > 0 {
            self.host_filter_index = (self.host_filter_index + 1) % len;
        }
        self.clamp_selected_row();
    }

    fn cycle_severity_filter(&mut self) {
        self.severity_filter = self.severity_filter.next();
        self.clamp_selected_row();
    }

    fn reset_filters(&mut self) {
        self.project_filter_index = 0;
        self.reason_filter_index = 0;
        self.host_filter_index = 0;
        self.severity_filter = SeverityFilter::All;
        self.clamp_selected_row();
    }

    fn build_rows(&self) -> Vec<Row<'static>> {
        self.filtered_events()
            .into_iter()
            .enumerate()
            .map(|(index, event)| {
                let severity = Self::event_severity(&event);
                let transition = format!("{:?}->{:?}", event.from, event.to);
                let confidence = if event.attribution_collision {
                    format!("{}!", event.attribution_confidence_score)
                } else {
                    event.attribution_confidence_score.to_string()
                };
                let project = self
                    .project_for_instance(&event.instance_id)
                    .unwrap_or("unknown")
                    .to_owned();
                let host = Self::host_bucket(&event.instance_id);
                let mut style = Style::default().fg(severity.color());
                if index == self.selected_row {
                    style = style.add_modifier(Modifier::REVERSED);
                }
                Row::new(vec![
                    event.at_ms.to_string(),
                    project,
                    host,
                    event.instance_id,
                    severity.label().to_owned(),
                    transition,
                    event.reason_code,
                    confidence,
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
            "filters: project={project}, severity={}, reason={reason}, host={host} | keys: p/s/r/h/x",
            self.severity_filter.label()
        )
    }

    fn event_count(&self) -> usize {
        self.filtered_events().len()
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

    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let events = self.filtered_events();
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(5)])
            .split(area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Timeline: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(Self::timeline_summary(&events)),
            ]),
            Line::from(Self::rolling_counter_summary(&events)),
            Line::from(self.filter_summary()),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Action Timeline "),
        );
        frame.render_widget(header, chunks[0]);

        let table = Table::new(
            self.build_rows(),
            [
                Constraint::Length(14),
                Constraint::Length(16),
                Constraint::Length(12),
                Constraint::Length(20),
                Constraint::Length(9),
                Constraint::Length(22),
                Constraint::Min(22),
                Constraint::Length(6),
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
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(Block::default().borders(Borders::ALL).title(" Events "));
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
                    let count = self.event_count();
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
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 2);
        assert_eq!(screen.handle_input(&down, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 2);

        let up = InputEvent::Key(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
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
}
