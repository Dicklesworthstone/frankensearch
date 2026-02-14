//! Action Timeline screen scaffold.
//!
//! Visualizes lifecycle transition events to support rapid triage.

use std::any::Any;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::state::{AppState, LifecycleEvent};

/// Timeline screen for lifecycle and operational transition events.
pub struct ActionTimelineScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
}

impl ActionTimelineScreen {
    /// Create a new timeline screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.timeline"),
            state: AppState::new(),
            selected_row: 0,
        }
    }

    /// Update timeline data from shared app state.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        let count = self.events().len();
        if count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= count {
            self.selected_row = count.saturating_sub(1);
        }
    }

    fn events(&self) -> Vec<LifecycleEvent> {
        let mut events = self.state.fleet().lifecycle_events.clone();
        events.sort_by(|left, right| {
            right
                .at_ms
                .cmp(&left.at_ms)
                .then_with(|| left.instance_id.cmp(&right.instance_id))
        });
        events
    }

    fn build_rows(&self) -> Vec<Row<'static>> {
        self.events()
            .into_iter()
            .enumerate()
            .map(|(index, event)| {
                let style = if index == self.selected_row {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                let transition = format!("{:?}->{:?}", event.from, event.to);
                let confidence = if event.attribution_collision {
                    format!("{}!", event.attribution_confidence_score)
                } else {
                    event.attribution_confidence_score.to_string()
                };
                Row::new(vec![
                    event.at_ms.to_string(),
                    event.instance_id,
                    transition,
                    event.reason_code,
                    confidence,
                ])
                .style(style)
            })
            .collect()
    }

    fn timeline_summary(&self) -> String {
        let events = self.events();
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

    fn event_count(&self) -> usize {
        self.events().len()
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
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5)])
            .split(area);

        let header = Paragraph::new(Line::from(vec![
            Span::styled("Timeline: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(self.timeline_summary()),
        ]))
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
                Constraint::Length(20),
                Constraint::Length(24),
                Constraint::Min(24),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec![
                "Timestamp",
                "Instance",
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
    use frankensearch_core::LifecycleState;

    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let mut fleet = crate::state::FleetSnapshot::default();
        fleet.lifecycle_events = vec![
            LifecycleEvent {
                instance_id: "inst-a".to_owned(),
                from: LifecycleState::Started,
                to: LifecycleState::Healthy,
                reason_code: "lifecycle.discovery.heartbeat".to_owned(),
                at_ms: 100,
                attribution_confidence_score: 90,
                attribution_collision: false,
            },
            LifecycleEvent {
                instance_id: "inst-b".to_owned(),
                from: LifecycleState::Healthy,
                to: LifecycleState::Stale,
                reason_code: "lifecycle.heartbeat_gap".to_owned(),
                at_ms: 200,
                attribution_confidence_score: 70,
                attribution_collision: true,
            },
        ];
        state.update_fleet(fleet);
        state
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
        let events = screen.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].instance_id, "inst-b");
        assert_eq!(events[1].instance_id, "inst-a");
    }

    #[test]
    fn timeline_navigation_is_bounded() {
        let mut screen = ActionTimelineScreen::new();
        screen.update_state(&sample_state());
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.timeline"),
            terminal_width: 100,
            terminal_height: 40,
            focused: true,
        };

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
}
