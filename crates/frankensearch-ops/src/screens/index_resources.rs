//! Index + embedding + resource monitoring screen.
//!
//! Focuses on fleet-vs-project comparisons for index inventory, embedding
//! backlog, and host resource pressure.

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

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{KeybindingHint, ScreenAction, ScreenContext, ScreenId};

use crate::state::AppState;
use crate::theme::SemanticPalette;

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonWindow {
    CurrentVsPreviousHourEstimate,
    CurrentVsSameHourYesterdayEstimate,
}

impl ComparisonWindow {
    const fn label(self) -> &'static str {
        match self {
            Self::CurrentVsPreviousHourEstimate => "current_hour vs previous_hour_estimate",
            Self::CurrentVsSameHourYesterdayEstimate => {
                "current_hour vs same_hour_yesterday_estimate"
            }
        }
    }

    const fn baseline_scale(self) -> (u16, u16) {
        match self {
            Self::CurrentVsPreviousHourEstimate => (92, 100),
            Self::CurrentVsSameHourYesterdayEstimate => (85, 100),
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::CurrentVsPreviousHourEstimate => Self::CurrentVsSameHourYesterdayEstimate,
            Self::CurrentVsSameHourYesterdayEstimate => Self::CurrentVsPreviousHourEstimate,
        }
    }
}

/// Index/resource monitoring screen.
pub struct IndexResourceScreen {
    id: ScreenId,
    state: AppState,
    selected_row: usize,
    project_filter_index: usize,
    comparison_window: ComparisonWindow,
    palette: SemanticPalette,
}

const INDEX_RESOURCES_KEYBINDINGS: &[KeybindingHint] = &[
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
        key: "o",
        description: "Cycle comparison window",
    },
    KeybindingHint {
        key: "x",
        description: "Reset filters",
    },
];

impl IndexResourceScreen {
    /// Create a new index/resource monitoring screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.index"),
            state: AppState::new(),
            selected_row: 0,
            project_filter_index: 0,
            comparison_window: ComparisonWindow::CurrentVsPreviousHourEstimate,
            palette: SemanticPalette::dark(),
        }
    }

    /// Update state from shared app snapshot.
    pub fn update_state(&mut self, state: &AppState) {
        self.state = state.clone();
        self.clamp_filter_index();
        self.clamp_selected_row();
    }

    pub const fn set_palette(&mut self, palette: SemanticPalette) {
        self.palette = palette;
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

    const fn cycle_comparison_window(&mut self) {
        self.comparison_window = self.comparison_window.next();
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

    #[allow(clippy::cast_precision_loss)]
    fn summary_lines(&self, rows: &[MonitorRow]) -> Vec<Line> {
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
        let (baseline_numerator, baseline_denominator) = self.comparison_window.baseline_scale();
        let baseline_numerator_u64 = u64::from(baseline_numerator);
        let baseline_denominator_u64 = u64::from(baseline_denominator);
        let baseline_cpu_avg =
            cpu_avg * f64::from(baseline_numerator) / f64::from(baseline_denominator);
        let baseline_p95_avg = p95_avg
            .saturating_mul(baseline_numerator_u64)
            .saturating_div(baseline_denominator_u64);
        let baseline_pending_total = pending_total
            .saturating_mul(baseline_numerator_u64)
            .saturating_div(baseline_denominator_u64);
        let delta_cpu_avg = cpu_avg - baseline_cpu_avg;
        let delta_p95_avg = i128::from(p95_avg) - i128::from(baseline_p95_avg);
        let delta_pending_total = i128::from(pending_total) - i128::from(baseline_pending_total);

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
                "comparison[{}]: p95 Δ{delta_p95_avg:+}us | cpu Δ{delta_cpu_avg:+.1}% | pending Δ{delta_pending_total:+}",
                self.comparison_window.label(),
            )),
            Line::from(
                "legend: unhealthy rows render in alert colors; percentiles are fleet-relative",
            ),
            Line::from(format!(
                "Search p95_avg={p95_avg}us | keys: p cycle project, o cycle comparison, x reset filters"
            )),
        ]
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

    fn row_pulse(row: &MonitorRow) -> String {
        let pending_pressure = Self::ratio_percent_u64(row.pending_jobs, row.docs.max(1));
        Self::sparkline(&[
            row.docs_percentile,
            row.p95_percentile,
            row.cpu_percentile,
            pending_pressure,
        ])
    }

    fn percentile_heatstrip_line(rows: &[MonitorRow]) -> String {
        if rows.is_empty() {
            return "percentile strip: (no rows)".to_owned();
        }
        let values: Vec<u8> = rows.iter().take(24).map(|row| row.p95_percentile).collect();
        format!("percentile strip: {}", Self::sparkline(&values))
    }

    fn hotspot_line(rows: &[MonitorRow]) -> String {
        let Some(top) = rows.first() else {
            return "hotspot: none".to_owned();
        };
        format!(
            "hotspot: {} p95={}us cpu={:.1}% pending={} docs={} pulse={}",
            top.instance_id,
            top.p95_latency_us,
            top.cpu_percent,
            top.pending_jobs,
            top.docs,
            Self::row_pulse(top)
        )
    }

    fn comparison_pills_line(&self) -> Line {
        let mut spans = vec![
            Span::styled("window:", self.palette.style_muted()),
            Span::raw(" "),
        ];
        let current_hour_style = if matches!(
            self.comparison_window,
            ComparisonWindow::CurrentVsPreviousHourEstimate
        ) {
            self.palette.style_highlight().bold()
        } else {
            self.palette.style_muted()
        };
        spans.push(Span::styled("[hour]", current_hour_style));
        spans.push(Span::raw(" "));
        let same_hour_style = if matches!(
            self.comparison_window,
            ComparisonWindow::CurrentVsSameHourYesterdayEstimate
        ) {
            self.palette.style_highlight().bold()
        } else {
            self.palette.style_muted()
        };
        spans.push(Span::styled("[yesterday]", same_hour_style));
        spans.push(Span::raw(" "));
        let filter_label = self
            .selected_project_filter()
            .unwrap_or_else(|| "all".to_owned());
        spans.push(Span::styled(
            format!("project={filter_label}"),
            self.palette.style_muted(),
        ));
        Line::from_spans(spans)
    }

    fn selected_context_line(&self, rows: &[MonitorRow]) -> String {
        if let Some(row) = rows.get(self.selected_row) {
            return format!(
                "focus: project={} instance={} p95={}us cpu={:.1}% pending={} pulse={}",
                row.project,
                row.instance_id,
                row.p95_latency_us,
                row.cpu_percent,
                row.pending_jobs,
                Self::row_pulse(row)
            );
        }
        "focus: none".to_owned()
    }

    fn render_compact(&self, frame: &mut Frame, area: Rect) {
        let rows = self.row_models();
        let mut lines = vec![
            Line::from("Index + Resources"),
            self.comparison_pills_line(),
            Line::from(Self::percentile_heatstrip_line(&rows)),
            Line::from(Self::hotspot_line(&rows)),
            Line::from(self.selected_context_line(&rows)),
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
                    .title(" Index + Resources "),
            )
            .render(area, frame);
    }

    fn build_rows(&self) -> Vec<Row> {
        self.row_models()
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let style = if index == self.selected_row {
                    self.palette.style_highlight().bold()
                } else if row.healthy {
                    self.palette.style_row_base(index)
                } else {
                    self.palette.style_row_error(index)
                };
                let pulse = Self::row_pulse(&row);

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
                    pulse,
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

    fn render(&self, frame: &mut Frame, _ctx: &ScreenContext) {
        let p = &self.palette;
        let border_style = p.style_border();

        let area = frame.bounds();
        if area.width < 110 || area.height < 14 {
            self.render_compact(frame, area);
            return;
        }

        let chunks = Flex::vertical()
            .constraints([Constraint::Fixed(10), Constraint::Min(6)])
            .split(area);

        let rows = self.row_models();
        let mut summary_lines = self.summary_lines(&rows);
        summary_lines.insert(1, self.comparison_pills_line());
        summary_lines.insert(2, Line::from(Self::percentile_heatstrip_line(&rows)));
        summary_lines.insert(3, Line::from(Self::hotspot_line(&rows)));
        summary_lines.push(Line::from(self.selected_context_line(&rows)));
        let summary = Paragraph::new(Text::from_lines(summary_lines)).block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(border_style)
                .title(" Index + Embedding + Resources "),
        );
        summary.render(chunks[0], frame);

        if rows.is_empty() {
            let empty = Paragraph::new("No index/resource rows available for this filter.").block(
                Block::new()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(border_style)
                    .title(" Inventory "),
            );
            empty.render(chunks[1], frame);
            return;
        }

        let table = Table::new(
            self.build_rows(),
            [
                Constraint::Fixed(12),
                Constraint::Fixed(18),
                Constraint::Fixed(8),
                Constraint::Fixed(8),
                Constraint::Fixed(8),
                Constraint::Fixed(10),
                Constraint::Fixed(10),
                Constraint::Fixed(8),
                Constraint::Fixed(8),
                Constraint::Fixed(8),
                Constraint::Fixed(8),
                Constraint::Fixed(6),
            ],
        )
        .header(
            Row::new(vec![
                "Project", "Instance", "Docs", "Pending", "CPU", "Mem", "IO", "P95us", "Docs%",
                "P95%", "CPU%", "Pulse",
            ])
            .style(Style::new().fg(self.palette.accent).bold()),
        )
        .block(
            Block::new()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(border_style)
                .title(" Inventory "),
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
                    let count = self.row_count();
                    if count > 0 && self.selected_row < count.saturating_sub(1) {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('p') => {
                    self.cycle_project_filter();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('x') => {
                    self.reset_filters();
                    return ScreenAction::Consumed;
                }
                ftui_core::event::KeyCode::Char('o') => {
                    self.cycle_comparison_window();
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

    fn keybindings(&self) -> &'static [KeybindingHint] {
        INDEX_RESOURCES_KEYBINDINGS
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
            ftui_core::event::KeyCode::Char('p'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&cycle, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.row_models().len(), 2);

        assert_eq!(screen.handle_input(&cycle, &ctx), ScreenAction::Consumed);
        assert_eq!(screen.row_models().len(), 1);

        let reset = InputEvent::Key(
            ftui_core::event::KeyCode::Char('x'),
            ftui_core::event::Modifiers::NONE,
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
            ftui_core::event::KeyCode::Down,
            ftui_core::event::Modifiers::NONE,
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
            .map(Line::to_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Index docs="));
        assert!(text.contains("Embedding pending="));
        assert!(text.contains("Resources cpu_avg="));
        assert!(text.contains("refined_share="));
        assert!(text.contains("comparison["));
    }

    #[test]
    fn comparison_window_cycles_via_keyboard() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        let ctx = context();

        let initial = screen
            .summary_lines(&screen.row_models())
            .iter()
            .map(Line::to_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(initial.contains("current_hour vs previous_hour_estimate"));

        let cycle = InputEvent::Key(
            ftui_core::event::KeyCode::Char('o'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&cycle, &ctx), ScreenAction::Consumed);
        let after_first = screen
            .summary_lines(&screen.row_models())
            .iter()
            .map(Line::to_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(after_first.contains("current_hour vs same_hour_yesterday_estimate"));

        assert_eq!(screen.handle_input(&cycle, &ctx), ScreenAction::Consumed);
        let after_second = screen
            .summary_lines(&screen.row_models())
            .iter()
            .map(Line::to_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(after_second.contains("current_hour vs previous_hour_estimate"));
    }

    // ── ComparisonWindow tests ───────────────────────────────────────

    #[test]
    fn comparison_window_labels_are_nonempty() {
        assert!(
            !ComparisonWindow::CurrentVsPreviousHourEstimate
                .label()
                .is_empty()
        );
        assert!(
            !ComparisonWindow::CurrentVsSameHourYesterdayEstimate
                .label()
                .is_empty()
        );
    }

    #[test]
    fn comparison_window_baseline_scales_are_valid_fractions() {
        for window in [
            ComparisonWindow::CurrentVsPreviousHourEstimate,
            ComparisonWindow::CurrentVsSameHourYesterdayEstimate,
        ] {
            let (numer, denom) = window.baseline_scale();
            assert!(denom > 0, "denominator must be positive");
            assert!(numer <= denom, "scale must be <= 1.0");
        }
    }

    #[test]
    fn comparison_window_cycles_back_to_start() {
        let start = ComparisonWindow::CurrentVsPreviousHourEstimate;
        let second = start.next();
        assert_eq!(second, ComparisonWindow::CurrentVsSameHourYesterdayEstimate);
        let third = second.next();
        assert_eq!(third, start);
    }

    // ── percentile_rank_u64 tests ────────────────────────────────────

    #[test]
    fn percentile_rank_u64_empty_returns_zero() {
        assert_eq!(IndexResourceScreen::percentile_rank_u64(&[], 42), 0);
    }

    #[test]
    fn percentile_rank_u64_single_at_target() {
        assert_eq!(IndexResourceScreen::percentile_rank_u64(&[10], 10), 100);
    }

    #[test]
    fn percentile_rank_u64_below_all() {
        assert_eq!(
            IndexResourceScreen::percentile_rank_u64(&[10, 20, 30], 5),
            0
        );
    }

    #[test]
    fn percentile_rank_u64_above_all() {
        assert_eq!(
            IndexResourceScreen::percentile_rank_u64(&[10, 20, 30], 30),
            100
        );
    }

    // ── percentile_rank_f64 tests ────────────────────────────────────

    #[test]
    fn percentile_rank_f64_empty_returns_zero() {
        assert_eq!(IndexResourceScreen::percentile_rank_f64(&[], 1.0), 0);
    }

    #[test]
    fn percentile_rank_f64_single_at_target() {
        assert_eq!(IndexResourceScreen::percentile_rank_f64(&[5.0], 5.0), 100);
    }

    // ── summary_lines tests ──────────────────────────────────────────

    #[test]
    fn summary_lines_empty_rows() {
        let screen = IndexResourceScreen::new();
        let lines = screen.summary_lines(&[]);
        let text = lines
            .iter()
            .map(Line::to_plain_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("No rows"));
        assert!(text.contains("cycle project filter"));
    }

    // ── Default impl ─────────────────────────────────────────────────

    #[test]
    fn default_matches_new() {
        let new_screen = IndexResourceScreen::new();
        let default_screen = IndexResourceScreen::default();
        assert_eq!(new_screen.id(), default_screen.id());
        assert_eq!(new_screen.selected_row, default_screen.selected_row);
    }

    // ── Navigation bounds ────────────────────────────────────────────

    #[test]
    fn up_at_zero_stays_at_zero() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        screen.selected_row = 0;
        let ctx = context();
        let up = InputEvent::Key(
            ftui_core::event::KeyCode::Up,
            ftui_core::event::Modifiers::NONE,
        );
        screen.handle_input(&up, &ctx);
        assert_eq!(screen.selected_row, 0);
    }

    // ── Unhandled key ────────────────────────────────────────────────

    #[test]
    fn unhandled_key_returns_ignored() {
        let mut screen = IndexResourceScreen::new();
        let ctx = context();
        let event = InputEvent::Key(
            ftui_core::event::KeyCode::Char('z'),
            ftui_core::event::Modifiers::NONE,
        );
        assert_eq!(screen.handle_input(&event, &ctx), ScreenAction::Ignored);
    }

    // ── clamp_selected_row ───────────────────────────────────────────

    #[test]
    fn clamp_selected_row_resets_on_empty() {
        let mut screen = IndexResourceScreen::new();
        screen.selected_row = 5;
        screen.clamp_selected_row();
        assert_eq!(screen.selected_row, 0);
    }

    #[test]
    fn clamp_selected_row_clamps_to_last() {
        let mut screen = IndexResourceScreen::new();
        screen.update_state(&sample_state());
        screen.selected_row = 100;
        screen.clamp_selected_row();
        assert_eq!(screen.selected_row, 2); // 3 instances, max index = 2
    }

    // ── row_models with missing metrics ──────────────────────────────

    #[test]
    fn row_models_handle_missing_metrics() {
        let mut screen = IndexResourceScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![InstanceInfo {
                id: "bare".to_owned(),
                project: "test".to_owned(),
                pid: None,
                healthy: true,
                doc_count: 10,
                pending_jobs: 0,
            }],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state);
        let rows = screen.row_models();
        assert_eq!(rows.len(), 1);
        assert!((rows[0].cpu_percent - 0.0).abs() < f64::EPSILON);
        assert_eq!(rows[0].memory_mib, 0);
        assert_eq!(rows[0].io_kib, 0);
        assert_eq!(rows[0].p95_latency_us, 0);
    }
}
