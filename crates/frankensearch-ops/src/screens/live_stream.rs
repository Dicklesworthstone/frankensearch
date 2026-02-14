//! Live Search Stream screen scaffold.
//!
//! Presents high-signal per-instance search activity and stream health
//! indicators from the latest fleet snapshot.

use std::any::Any;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::state::AppState;

#[derive(Clone)]
struct StreamRowData {
    instance_id: String,
    project: String,
    searches: u64,
    avg_latency_us: u64,
    p95_latency_us: u64,
    refined_count: u64,
}

/// Live stream screen with recent activity and stream-health status.
pub struct LiveSearchStreamScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
}

impl LiveSearchStreamScreen {
    /// Create a new live stream screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.live_stream"),
            state: AppState::new(),
            selected_row: 0,
        }
    }

    /// Update screen data from shared application state.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        let count = self.row_data().len();
        if count == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= count {
            self.selected_row = count.saturating_sub(1);
        }
    }

    fn row_data(&self) -> Vec<StreamRowData> {
        let fleet = self.state.fleet();
        let mut rows: Vec<_> = fleet
            .instances
            .iter()
            .map(|instance| {
                let metrics = fleet.search_metrics.get(&instance.id);
                StreamRowData {
                    instance_id: instance.id.clone(),
                    project: instance.project.clone(),
                    searches: metrics.map_or(0, |value| value.total_searches),
                    avg_latency_us: metrics.map_or(0, |value| value.avg_latency_us),
                    p95_latency_us: metrics.map_or(0, |value| value.p95_latency_us),
                    refined_count: metrics.map_or(0, |value| value.refined_count),
                }
            })
            .collect();

        rows.sort_by(|left, right| {
            right
                .searches
                .cmp(&left.searches)
                .then_with(|| left.instance_id.cmp(&right.instance_id))
        });
        rows
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
                let style = if index == self.selected_row {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                Row::new(vec![
                    row.instance_id,
                    row.project,
                    row.searches.to_string(),
                    row.avg_latency_us.to_string(),
                    row.p95_latency_us.to_string(),
                    refined_rate,
                ])
                .style(style)
            })
            .collect()
    }

    fn stream_health_summary(&self) -> String {
        let metrics = self.state.control_plane_metrics();
        let stream_state = if self.state.has_data() {
            "connected"
        } else {
            "disconnected"
        };
        format!(
            "state={stream_state} | lag={} | dead_letter={} | throughput={:.2} eps",
            metrics.ingestion_lag_events, metrics.dead_letter_events, metrics.event_throughput_eps
        )
    }

    fn instance_count(&self) -> usize {
        self.row_data().len()
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
            .constraints([Constraint::Length(3), Constraint::Min(5)])
            .split(area);

        let header = Paragraph::new(Line::from(vec![
            Span::styled("Stream: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(self.stream_health_summary()),
        ]))
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
                Constraint::Length(10),
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(12),
            ],
        )
        .header(
            Row::new(vec![
                "Instance",
                "Project",
                "Searches",
                "Avg Lat(us)",
                "P95 Lat(us)",
                "Refined Rate",
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
    use crate::state::{FleetSnapshot, InstanceInfo, SearchMetrics};

    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot::default();
        fleet.instances = vec![
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
                healthy: true,
                doc_count: 64,
                pending_jobs: 3,
            },
        ];
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
                avg_latency_us: 900,
                p95_latency_us: 1800,
                refined_count: 10,
            },
        );
        state.update_fleet(fleet);
        state
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
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.live_stream"),
            terminal_width: 120,
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
