//! Index + embedding + resource monitoring screen.
//!
//! Focuses on fleet-vs-project comparisons for index inventory, embedding
//! backlog, and host resource pressure.

use std::any::Any;
use std::collections::BTreeSet;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::state::AppState;

#[derive(Debug, Clone, PartialEq)]
struct MonitorRow {
    project: String,
    instance_id: String,
    healthy: bool,
    docs: u64,
    pending_jobs: u64,
    cpu_percent: f64,
    memory_mib: u64,
    io_kib: u64,
    p95_latency_us: u64,
    docs_percentile: u8,
    p95_percentile: u8,
    cpu_percentile: u8,
}

/// Index/resource monitoring screen.
pub struct IndexResourceScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
    project_filter_index: usize,
}

impl IndexResourceScreen {
    /// Create a new index/resource monitoring screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.index"),
            state: AppState::new(),
            selected_row: 0,
            project_filter_index: 0,
        }
    }

    /// Update state from shared app snapshot.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        self.clamp_filter_index();
        self.clamp_selected_row();
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

    fn selected_project_filter(&self) -> Option<String> {
        self.project_filters()
            .get(self.project_filter_index)
            .cloned()
            .filter(|value| value != "all")
    }

    fn filtered_instances(&self) -> Vec<&crate::state::InstanceInfo> {
        let project_filter = self.selected_project_filter();
        self.state
            .fleet()
            .instances
            .iter()
            .filter(|instance| {
                project_filter
                    .as_deref()
                    .is_none_or(|project| instance.project.eq_ignore_ascii_case(project))
            })
            .collect()
    }

    fn clamp_filter_index(&mut self) {
        let max = self.project_filters().len().saturating_sub(1);
        if self.project_filter_index > max {
            self.project_filter_index = max;
        }
    }

    fn clamp_selected_row(&mut self) {
        let count = self.filtered_instances().len();
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

    fn reset_filters(&mut self) {
        self.project_filter_index = 0;
        self.clamp_selected_row();
    }

    fn percentile_rank_u64(values: &[u64], value: u64) -> u8 {
        if values.is_empty() {
            return 0;
        }
        let le_count = values
            .iter()
            .filter(|candidate| **candidate <= value)
            .count();
        let value_count_u64 = u64::try_from(values.len()).unwrap_or(u64::MAX);
        let less_equal_count_u64 = u64::try_from(le_count).unwrap_or(value_count_u64);
        let pct = less_equal_count_u64
            .saturating_mul(100)
            .saturating_add(value_count_u64 / 2)
            .saturating_div(value_count_u64)
            .min(100);
        u8::try_from(pct).unwrap_or(100)
    }

    fn percentile_rank_f64(values: &[f64], value: f64) -> u8 {
        if values.is_empty() {
            return 0;
        }
        let le_count = values
            .iter()
            .filter(|candidate| **candidate <= value + f64::EPSILON)
            .count();
        let value_count_u64 = u64::try_from(values.len()).unwrap_or(u64::MAX);
        let less_equal_count_u64 = u64::try_from(le_count).unwrap_or(value_count_u64);
        let pct = less_equal_count_u64
            .saturating_mul(100)
            .saturating_add(value_count_u64 / 2)
            .saturating_div(value_count_u64)
            .min(100);
        u8::try_from(pct).unwrap_or(100)
    }

    fn row_models(&self) -> Vec<MonitorRow> {
        let fleet = self.state.fleet();
        let docs_values: Vec<u64> = fleet
            .instances
            .iter()
            .map(|instance| instance.doc_count)
            .collect();
        let p95_values: Vec<u64> = fleet
            .instances
            .iter()
            .filter_map(|instance| {
                fleet
                    .search_metrics
                    .get(&instance.id)
                    .map(|metrics| metrics.p95_latency_us)
            })
            .collect();
        let cpu_values: Vec<f64> = fleet
            .instances
            .iter()
            .filter_map(|instance| {
                fleet
                    .resources
                    .get(&instance.id)
                    .map(|metrics| metrics.cpu_percent)
            })
            .collect();

        let mut rows = self
            .filtered_instances()
            .into_iter()
            .map(|instance| {
                let search = fleet.search_metrics.get(&instance.id);
                let resource = fleet.resources.get(&instance.id);
                let p95_latency_us = search.map_or(0, |metrics| metrics.p95_latency_us);
                let cpu_percent = resource.map_or(0.0, |metrics| metrics.cpu_percent);
                let memory_mib = resource.map_or(0, |metrics| metrics.memory_bytes / (1024 * 1024));
                let io_kib = resource.map_or(0, |metrics| {
                    metrics
                        .io_read_bytes
                        .saturating_add(metrics.io_write_bytes)
                        .saturating_div(1024)
                });
                MonitorRow {
                    project: instance.project.clone(),
                    instance_id: instance.id.clone(),
                    healthy: instance.healthy,
                    docs: instance.doc_count,
                    pending_jobs: instance.pending_jobs,
                    cpu_percent,
                    memory_mib,
                    io_kib,
                    p95_latency_us,
                    docs_percentile: Self::percentile_rank_u64(&docs_values, instance.doc_count),
                    p95_percentile: Self::percentile_rank_u64(&p95_values, p95_latency_us),
                    cpu_percentile: Self::percentile_rank_f64(&cpu_values, cpu_percent),
                }
            })
            .collect::<Vec<_>>();

        rows.sort_by(|left, right| {
            right
                .p95_latency_us
                .cmp(&left.p95_latency_us)
                .then_with(|| right.pending_jobs.cmp(&left.pending_jobs))
                .then_with(|| right.docs.cmp(&left.docs))
                .then_with(|| left.instance_id.cmp(&right.instance_id))
        });
        rows
    }

    fn summary_lines(&self, rows: &[MonitorRow]) -> Vec<Line<'static>> {
        if rows.is_empty() {
            return vec![
                Line::from("No rows for current project filter."),
                Line::from("Press `p` to cycle project filter or `x` to reset."),
            ];
        }

        let docs_total: u64 = rows.iter().map(|row| row.docs).sum();
        let pending_total: u64 = rows.iter().map(|row| row.pending_jobs).sum();
        let io_total: u64 = rows.iter().map(|row| row.io_kib).sum();
        let memory_total: u64 = rows.iter().map(|row| row.memory_mib).sum();
        let p95_total: u64 = rows.iter().map(|row| row.p95_latency_us).sum();
        let cpu_total: f64 = rows.iter().map(|row| row.cpu_percent).sum();

        let count_u64 = u64::try_from(rows.len()).unwrap_or(1);
        let count_u32 = u32::try_from(rows.len()).unwrap_or(1);
        let cpu_avg = cpu_total / f64::from(count_u32);
        let memory_avg = memory_total.saturating_add(count_u64 / 2) / count_u64;
        let p95_avg = p95_total.saturating_add(count_u64 / 2) / count_u64;

        let fleet = self.state.fleet();
        let filtered_ids: BTreeSet<_> = rows.iter().map(|row| row.instance_id.as_str()).collect();
        let mut total_searches = 0_u64;
        let mut refined_total = 0_u64;
        for (instance_id, metrics) in &fleet.search_metrics {
            if filtered_ids.contains(instance_id.as_str()) {
                total_searches = total_searches.saturating_add(metrics.total_searches);
                refined_total = refined_total.saturating_add(metrics.refined_count);
            }
        }
        let refined_share = if total_searches == 0 {
            0
        } else {
            refined_total
                .saturating_mul(100)
                .saturating_add(total_searches / 2)
                .saturating_div(total_searches)
                .min(100)
        };

        let project = self
            .selected_project_filter()
            .unwrap_or_else(|| "all".to_owned());
        vec![
            Line::from(format!(
                "project_filter={project} | visible_instances={}",
                rows.len()
            )),
            Line::from(format!(
                "Index docs={docs_total} | Embedding pending={pending_total} | refined_share={refined_share}%"
            )),
            Line::from(format!(
                "Resources cpu_avg={cpu_avg:.1}% | mem_avg={memory_avg}MiB | io_total={io_total}KiB"
            )),
            Line::from(format!(
                "Search p95_avg={p95_avg}us | keys: p cycle project, x reset filters"
            )),
        ]
    }

    fn build_rows(&self) -> Vec<Row<'static>> {
        self.row_models()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let mut style = if row.healthy {
                    Style::default()
                } else {
                    Style::default().fg(Color::Red)
                };
                if index == self.selected_row {
                    style = style.add_modifier(Modifier::REVERSED);
                }

                Row::new(vec![
                    row.project,
                    row.instance_id,
                    row.docs.to_string(),
                    row.pending_jobs.to_string(),
                    format!("{:.1}%", row.cpu_percent),
                    format!("{}MiB", row.memory_mib),
                    format!("{}KiB", row.io_kib),
                    row.p95_latency_us.to_string(),
                    format!("{}%", row.docs_percentile),
                    format!("{}%", row.p95_percentile),
                    format!("{}%", row.cpu_percentile),
                ])
                .style(style)
            })
            .collect()
    }

    fn row_count(&self) -> usize {
        self.row_models().len()
    }
}

impl Default for IndexResourceScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for IndexResourceScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Index + Resources"
    }

    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(6), Constraint::Min(6)])
            .split(area);

        let rows = self.row_models();
        let summary = Paragraph::new(self.summary_lines(&rows)).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Index + Embedding + Resources "),
        );
        frame.render_widget(summary, chunks[0]);

        if rows.is_empty() {
            let empty = Paragraph::new("No index/resource rows available for this filter.")
                .block(Block::default().borders(Borders::ALL).title(" Inventory "));
            frame.render_widget(empty, chunks[1]);
            return;
        }

        let table = Table::new(
            self.build_rows(),
            [
                Constraint::Length(12),
                Constraint::Length(18),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(8),
            ],
        )
        .header(
            Row::new(vec![
                "Project", "Instance", "Docs", "Pending", "CPU", "Mem", "IO", "P95us", "Docs%",
                "P95%", "CPU%",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(Block::default().borders(Borders::ALL).title(" Inventory "));
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
                    let count = self.row_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('p') => {
                    self.cycle_project_filter();
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
        "table"
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
    use crate::state::{FleetSnapshot, InstanceInfo, ResourceMetrics, SearchMetrics};

    fn sample_state() -> AppState {
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "alpha-1".to_owned(),
                    project: "alpha".to_owned(),
                    pid: Some(11),
                    healthy: true,
                    doc_count: 100,
                    pending_jobs: 5,
                },
                InstanceInfo {
                    id: "alpha-2".to_owned(),
                    project: "alpha".to_owned(),
                    pid: Some(12),
                    healthy: false,
                    doc_count: 40,
                    pending_jobs: 20,
                },
                InstanceInfo {
                    id: "beta-1".to_owned(),
                    project: "beta".to_owned(),
                    pid: Some(21),
                    healthy: true,
                    doc_count: 200,
                    pending_jobs: 2,
                },
            ],
            ..FleetSnapshot::default()
        };

        fleet.search_metrics.insert(
            "alpha-1".to_owned(),
            SearchMetrics {
                total_searches: 50,
                avg_latency_us: 900,
                p95_latency_us: 1_500,
                refined_count: 10,
            },
        );
        fleet.search_metrics.insert(
            "alpha-2".to_owned(),
            SearchMetrics {
                total_searches: 20,
                avg_latency_us: 2_500,
                p95_latency_us: 7_000,
                refined_count: 2,
            },
        );
        fleet.search_metrics.insert(
            "beta-1".to_owned(),
            SearchMetrics {
                total_searches: 80,
                avg_latency_us: 800,
                p95_latency_us: 1_000,
                refined_count: 24,
            },
        );

        fleet.resources.insert(
            "alpha-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 20.0,
                memory_bytes: 512 * 1024 * 1024,
                io_read_bytes: 100 * 1024,
                io_write_bytes: 50 * 1024,
            },
        );
        fleet.resources.insert(
            "alpha-2".to_owned(),
            ResourceMetrics {
                cpu_percent: 70.0,
                memory_bytes: 1024 * 1024 * 1024,
                io_read_bytes: 500 * 1024,
                io_write_bytes: 300 * 1024,
            },
        );
        fleet.resources.insert(
            "beta-1".to_owned(),
            ResourceMetrics {
                cpu_percent: 10.0,
                memory_bytes: 256 * 1024 * 1024,
                io_read_bytes: 60 * 1024,
                io_write_bytes: 40 * 1024,
            },
        );

        state.update_fleet(fleet);
        state
    }

    fn context() -> ScreenContext {
        ScreenContext {
            active_screen: ScreenId::new("ops.index"),
            terminal_width: 120,
            terminal_height: 40,
            focused: true,
        }
    }

    #[test]
    fn screen_defaults() {
        let screen = IndexResourceScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.index"));
        assert_eq!(screen.title(), "Index + Resources");
        assert_eq!(screen.semantic_role(), "table");
    }

    #[test]
    fn row_models_are_sorted_by_p95_descending() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        let rows = screen.row_models();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].instance_id, "alpha-2");
        assert_eq!(rows[0].p95_latency_us, 7_000);
    }

    #[test]
    fn row_models_include_percentile_context() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        let rows = screen.row_models();
        let alpha_2 = rows
            .iter()
            .find(|row| row.instance_id == "alpha-2")
            .expect("alpha-2 row exists");
        assert_eq!(alpha_2.p95_percentile, 100);
        assert!(alpha_2.cpu_percentile >= 67);
    }

    #[test]
    fn project_filter_cycles_and_resets() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();
        assert_eq!(screen.row_models().len(), 3);

        let cycle = InputEvent::Key(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&cycle, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.row_models().len(), 2);

        assert_eq!(screen.handle_input(&cycle, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.row_models().len(), 1);

        let reset = InputEvent::Key(
            crossterm::event::KeyCode::Char('x'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(screen.handle_input(&reset, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.row_models().len(), 3);
    }

    #[test]
    fn navigation_bounds_selection() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();

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
    }

    #[test]
    fn summary_contains_index_embedding_and_resource_signals() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        let text = screen
            .summary_lines(&screen.row_models())
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Index docs="));
        assert!(text.contains("Embedding pending="));
        assert!(text.contains("Resources cpu_avg="));
        assert!(text.contains("refined_share="));
    }
}
