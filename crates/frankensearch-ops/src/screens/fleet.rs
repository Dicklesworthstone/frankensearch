//! Fleet overview screen — primary dashboard showing all discovered instances.
//!
//! Displays instance list with health status, document counts, pending jobs,
//! and resource utilization at a glance.

use std::any::Any;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use frankensearch_tui::Screen;
use frankensearch_tui::input::InputEvent;
use frankensearch_tui::screen::{ScreenAction, ScreenContext, ScreenId};

use crate::presets::ViewState;
use crate::state::AppState;

/// Fleet overview screen showing all discovered instances.
pub struct FleetOverviewScreen {
    id: ScreenId,
    project_screen_id: ScreenId,
    live_stream_screen_id: ScreenId,
    timeline_screen_id: ScreenId,
    analytics_screen_id: ScreenId,
    state: AppState,
    view: ViewState,
    selected_row: usize,
}

impl FleetOverviewScreen {
    /// Create a new fleet overview screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id: ScreenId::new("ops.fleet"),
            project_screen_id: ScreenId::new("ops.project"),
            live_stream_screen_id: ScreenId::new("ops.live_stream"),
            timeline_screen_id: ScreenId::new("ops.timeline"),
            analytics_screen_id: ScreenId::new("ops.timeline"),
            state: AppState::new(),
            view: ViewState::default(),
            selected_row: 0,
        }
    }

    /// Override the drilldown destination used for Enter navigation.
    pub fn set_project_screen_id(&mut self, id: ScreenId) {
        self.project_screen_id = id;
    }

    /// Override the live stream drilldown destination used for `s`.
    pub fn set_live_stream_screen_id(&mut self, id: ScreenId) {
        self.live_stream_screen_id = id;
    }

    /// Override the timeline drilldown destination used for `t`.
    pub fn set_timeline_screen_id(&mut self, id: ScreenId) {
        self.timeline_screen_id = id;
    }

    /// Override the analytics drilldown destination used for `a`.
    pub fn set_analytics_screen_id(&mut self, id: ScreenId) {
        self.analytics_screen_id = id;
    }

    /// Update the screen's data from shared state.
    pub fn update_state(&mut self, state: &AppState, view: &ViewState) {
        self.state = state.clone();
        self.view = view.clone();
        let visible = self.visible_instances().len();
        if visible == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= visible {
            self.selected_row = visible - 1;
        }
    }

    fn visible_instances(&self) -> Vec<&crate::state::InstanceInfo> {
        let fleet = self.state.fleet();
        let mut visible: Vec<_> = fleet
            .instances
            .iter()
            .filter(|inst| !self.view.hide_healthy || !inst.healthy)
            .filter(|inst| {
                self.view
                    .project_filter
                    .as_deref()
                    .is_none_or(|project| inst.project.eq_ignore_ascii_case(project))
            })
            .collect();

        if self.view.unhealthy_first {
            visible.sort_by(|left, right| {
                (left.healthy, left.project.as_str(), left.id.as_str()).cmp(&(
                    right.healthy,
                    right.project.as_str(),
                    right.id.as_str(),
                ))
            });
        }

        visible
    }

    fn selected_instance(&self) -> Option<&crate::state::InstanceInfo> {
        let visible = self.visible_instances();
        visible.get(self.selected_row).copied()
    }

    fn percentile_rank_u64(values: &[u64], target: u64) -> u8 {
        if values.is_empty() {
            return 0;
        }
        let less_or_equal = values.iter().filter(|value| **value <= target).count();
        let less_or_equal = u64::try_from(less_or_equal).unwrap_or(u64::MAX);
        let total = u64::try_from(values.len()).unwrap_or(u64::MAX);
        if total == 0 {
            return 0;
        }
        let percentile = less_or_equal
            .saturating_mul(100)
            .saturating_add(total.saturating_sub(1))
            .saturating_div(total)
            .min(100);
        u8::try_from(percentile).unwrap_or(100)
    }

    fn percentile_rank_f64(values: &[f64], target: f64) -> u8 {
        if values.is_empty() {
            return 0;
        }
        let less_or_equal = values
            .iter()
            .filter(|value| value.total_cmp(&target).is_le())
            .count();
        let less_or_equal = u64::try_from(less_or_equal).unwrap_or(u64::MAX);
        let total = u64::try_from(values.len()).unwrap_or(u64::MAX);
        if total == 0 {
            return 0;
        }
        let percentile = less_or_equal
            .saturating_mul(100)
            .saturating_add(total.saturating_sub(1))
            .saturating_div(total)
            .min(100);
        u8::try_from(percentile).unwrap_or(100)
    }

    #[allow(clippy::too_many_lines)]
    fn selected_monitor_lines(&self) -> Vec<Line<'static>> {
        let fleet = self.state.fleet();
        let Some(instance) = self.selected_instance() else {
            return vec![Line::from("No instance selected")];
        };

        let resource = fleet.resources.get(&instance.id);
        let search = fleet.search_metrics.get(&instance.id);
        let attribution = fleet.attribution_for(&instance.id);

        let cpu_values: Vec<_> = fleet
            .resources
            .values()
            .map(|metric| metric.cpu_percent)
            .collect();
        let memory_values: Vec<_> = fleet
            .resources
            .values()
            .map(|metric| metric.memory_bytes)
            .collect();
        let p95_values: Vec<_> = fleet
            .search_metrics
            .values()
            .map(|metric| metric.p95_latency_us)
            .collect();
        let pending_values: Vec<_> = fleet
            .instances
            .iter()
            .map(|item| item.pending_jobs)
            .collect();
        let docs_values: Vec<_> = fleet.instances.iter().map(|item| item.doc_count).collect();

        let project_docs_values: Vec<_> = fleet
            .instances
            .iter()
            .filter(|item| item.project == instance.project)
            .map(|item| item.doc_count)
            .collect();
        let project_pending_values: Vec<_> = fleet
            .instances
            .iter()
            .filter(|item| item.project == instance.project)
            .map(|item| item.pending_jobs)
            .collect();
        let project_cpu_values: Vec<_> = fleet
            .instances
            .iter()
            .filter(|item| item.project == instance.project)
            .filter_map(|item| {
                fleet
                    .resources
                    .get(&item.id)
                    .map(|metric| metric.cpu_percent)
            })
            .collect();
        let project_memory_values: Vec<_> = fleet
            .instances
            .iter()
            .filter(|item| item.project == instance.project)
            .filter_map(|item| {
                fleet
                    .resources
                    .get(&item.id)
                    .map(|metric| metric.memory_bytes)
            })
            .collect();
        let project_p95_values: Vec<_> = fleet
            .instances
            .iter()
            .filter(|item| item.project == instance.project)
            .filter_map(|item| {
                fleet
                    .search_metrics
                    .get(&item.id)
                    .map(|metric| metric.p95_latency_us)
            })
            .collect();

        let docs_line = format!(
            "Docs: {} (project p{} | fleet p{})",
            instance.doc_count,
            Self::percentile_rank_u64(&project_docs_values, instance.doc_count),
            Self::percentile_rank_u64(&docs_values, instance.doc_count)
        );
        let pending_line = format!(
            "Pending: {} (project p{} | fleet p{})",
            instance.pending_jobs,
            Self::percentile_rank_u64(&project_pending_values, instance.pending_jobs),
            Self::percentile_rank_u64(&pending_values, instance.pending_jobs)
        );

        let cpu_line = resource.map_or_else(
            || "CPU: n/a".to_owned(),
            |metric| {
                format!(
                    "CPU: {:.1}% (project p{} | fleet p{})",
                    metric.cpu_percent,
                    Self::percentile_rank_f64(&project_cpu_values, metric.cpu_percent),
                    Self::percentile_rank_f64(&cpu_values, metric.cpu_percent)
                )
            },
        );
        let memory_line = resource.map_or_else(
            || "Memory: n/a".to_owned(),
            |metric| {
                format!(
                    "Memory: {} MiB (project p{} | fleet p{})",
                    metric.memory_bytes / (1024 * 1024),
                    Self::percentile_rank_u64(&project_memory_values, metric.memory_bytes),
                    Self::percentile_rank_u64(&memory_values, metric.memory_bytes)
                )
            },
        );
        let search_line = search.map_or_else(
            || "Search: n/a".to_owned(),
            |metric| {
                format!(
                    "Search: total={} avg={}us p95={}us (project p{} | fleet p{})",
                    metric.total_searches,
                    metric.avg_latency_us,
                    metric.p95_latency_us,
                    Self::percentile_rank_u64(&project_p95_values, metric.p95_latency_us),
                    Self::percentile_rank_u64(&p95_values, metric.p95_latency_us)
                )
            },
        );

        vec![
            Line::from(vec![
                Span::styled("Instance: ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(instance.id.clone()),
            ]),
            Line::from(format!("Project: {}", instance.project)),
            Line::from(if instance.healthy {
                "Health: OK".to_owned()
            } else {
                "Health: WARN".to_owned()
            }),
            Line::from(attribution.map_or_else(
                || "Attribution: n/a".to_owned(),
                |attribution| {
                    format!(
                        "Attribution: {} ({}%) reason={}",
                        attribution.resolved_project,
                        attribution.confidence_score,
                        attribution.reason_code
                    )
                },
            )),
            Line::from(String::new()),
            Line::from(docs_line),
            Line::from(pending_line),
            Line::from(cpu_line),
            Line::from(memory_line),
            Line::from(search_line),
        ]
    }

    #[cfg(test)]
    fn selected_monitor_text(&self) -> String {
        self.selected_monitor_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
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

    fn ratio_percent_usize(numer: usize, denom: usize) -> u8 {
        let numer = u64::try_from(numer).unwrap_or(u64::MAX);
        let denom = u64::try_from(denom).unwrap_or(u64::MAX);
        Self::ratio_percent_u64(numer, denom)
    }

    fn clamp_percent(value: f64) -> u8 {
        if !value.is_finite() {
            return 0;
        }
        let bounded = value.clamp(0.0, 100.0).round();
        let mut percent = 0u8;
        while percent < 100 && f64::from(percent) < bounded {
            percent = percent.saturating_add(1);
        }
        percent
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

    fn kpi_tile_lines(&self) -> Vec<Line<'static>> {
        let fleet = self.state.fleet();
        let visible = self.visible_instances();
        let visible_count = visible.len();
        let total_count = fleet.instance_count();
        let healthy_count = visible.iter().filter(|instance| instance.healthy).count();
        let unhealthy_count = visible_count.saturating_sub(healthy_count);
        let health_pct = Self::ratio_percent_usize(healthy_count, visible_count);

        let docs: u64 = visible.iter().map(|instance| instance.doc_count).sum();
        let pending: u64 = visible.iter().map(|instance| instance.pending_jobs).sum();
        let pending_pct = Self::ratio_percent_u64(pending, docs.saturating_add(pending));

        let search_total: u64 = visible
            .iter()
            .map(|instance| {
                fleet
                    .search_metrics
                    .get(&instance.id)
                    .map_or(0, |metrics| metrics.total_searches)
            })
            .sum();
        let per_instance_searches = if visible_count == 0 {
            0
        } else {
            search_total / u64::try_from(visible_count).unwrap_or(u64::MAX)
        };

        let attribution_scores: Vec<u8> = visible
            .iter()
            .filter_map(|instance| fleet.attribution_for(&instance.id))
            .map(|attribution| attribution.confidence_score)
            .collect();
        let attribution_avg = if attribution_scores.is_empty() {
            0
        } else {
            let score_sum: u64 = attribution_scores
                .iter()
                .map(|score| u64::from(*score))
                .sum();
            let sample_count = u64::try_from(attribution_scores.len()).unwrap_or(u64::MAX);
            u8::try_from(
                score_sum
                    .saturating_add(sample_count / 2)
                    .saturating_div(sample_count)
                    .min(100),
            )
            .unwrap_or(100)
        };
        let collisions = visible
            .iter()
            .filter_map(|instance| fleet.attribution_for(&instance.id))
            .filter(|attribution| attribution.collision)
            .count();

        vec![
            Line::from(format!(
                "[Instances] vis={visible_count}/{total_count} | [Healthy] {healthy_count}/{visible_count} ({health_pct}%) | [Unhealthy] {unhealthy_count}"
            )),
            Line::from(format!(
                "[Corpus] docs={docs} | pending={pending} ({pending_pct}%) | [Search] total={search_total} avg/inst={per_instance_searches}"
            )),
            Line::from(format!(
                "[Attribution] avg={attribution_avg}% collisions={collisions} | [Control] {}",
                self.state.control_plane_health().badge()
            )),
        ]
    }

    fn status_sparkline_lines(&self) -> Vec<Line<'static>> {
        let fleet = self.state.fleet();
        let metrics = self.state.control_plane_metrics();
        let visible = self.visible_instances();

        let visible_count = visible.len();
        let healthy_count = visible.iter().filter(|instance| instance.healthy).count();
        let health_pct = Self::ratio_percent_usize(healthy_count, visible_count);

        let docs: u64 = visible.iter().map(|instance| instance.doc_count).sum();
        let pending: u64 = visible.iter().map(|instance| instance.pending_jobs).sum();
        let pending_pressure = Self::ratio_percent_u64(pending, docs.saturating_add(pending));
        let queue_headroom = 100u8.saturating_sub(pending_pressure);

        let throughput_strength =
            Self::clamp_percent((metrics.event_throughput_eps / 20.0) * 100.0);
        let lag_pressure =
            Self::ratio_percent_u64(metrics.ingestion_lag_events.min(20_000), 20_000);
        let lag_headroom = 100u8.saturating_sub(lag_pressure);
        let storage_headroom =
            100u8.saturating_sub(Self::clamp_percent(metrics.storage_utilization() * 100.0));
        let rss_headroom =
            100u8.saturating_sub(Self::clamp_percent(metrics.rss_utilization() * 100.0));
        let dead_letter_pressure =
            Self::ratio_percent_u64(metrics.dead_letter_events.min(200), 200);
        let dead_letter_headroom = 100u8.saturating_sub(dead_letter_pressure);
        let discovery_pressure =
            Self::ratio_percent_u64(metrics.discovery_latency_ms.min(5_000), 5_000);
        let discovery_headroom = 100u8.saturating_sub(discovery_pressure);
        let search_total: u64 = visible
            .iter()
            .map(|instance| {
                fleet
                    .search_metrics
                    .get(&instance.id)
                    .map_or(0, |search| search.total_searches)
            })
            .sum();
        let search_strength = Self::ratio_percent_u64(search_total.min(5_000), 5_000);

        let strip_values = [
            health_pct,
            queue_headroom,
            throughput_strength,
            lag_headroom,
            storage_headroom,
            rss_headroom,
            dead_letter_headroom,
            discovery_headroom.max(search_strength),
        ];
        let strip = Self::sparkline(&strip_values);

        vec![
            Line::from(format!("H Q T L S R D X | {strip}")),
            Line::from(format!(
                "H={health_pct}% Q={queue_headroom}% T={throughput_strength}% L={lag_headroom}% S={storage_headroom}% R={rss_headroom}% D={dead_letter_headroom}% X={}%",
                discovery_headroom.max(search_strength)
            )),
        ]
    }

    fn project_summary_lines(&self) -> Vec<Line<'static>> {
        #[derive(Default)]
        struct ProjectAccumulator {
            instance_count: usize,
            unhealthy_count: usize,
            docs: u64,
            pending: u64,
            searches: u64,
            cpu_sum: f64,
            cpu_samples: usize,
            attribution_sum: u64,
            attribution_samples: usize,
        }

        let fleet = self.state.fleet();
        let mut accumulators: std::collections::BTreeMap<String, ProjectAccumulator> =
            std::collections::BTreeMap::new();
        for instance in self.visible_instances() {
            let accumulator = accumulators.entry(instance.project.clone()).or_default();
            accumulator.instance_count = accumulator.instance_count.saturating_add(1);
            if !instance.healthy {
                accumulator.unhealthy_count = accumulator.unhealthy_count.saturating_add(1);
            }
            accumulator.docs = accumulator.docs.saturating_add(instance.doc_count);
            accumulator.pending = accumulator.pending.saturating_add(instance.pending_jobs);
            accumulator.searches = accumulator.searches.saturating_add(
                fleet
                    .search_metrics
                    .get(&instance.id)
                    .map_or(0, |metrics| metrics.total_searches),
            );
            if let Some(resource) = fleet.resources.get(&instance.id) {
                accumulator.cpu_sum += resource.cpu_percent;
                accumulator.cpu_samples = accumulator.cpu_samples.saturating_add(1);
            }
            if let Some(attribution) = fleet.attribution_for(&instance.id) {
                accumulator.attribution_sum = accumulator
                    .attribution_sum
                    .saturating_add(u64::from(attribution.confidence_score));
                accumulator.attribution_samples = accumulator.attribution_samples.saturating_add(1);
            }
        }

        if accumulators.is_empty() {
            return vec![Line::from("No projects in active view")];
        }

        let mut cards: Vec<(String, ProjectAccumulator)> = accumulators.into_iter().collect();
        cards.sort_by(|(left_project, left), (right_project, right)| {
            right
                .unhealthy_count
                .cmp(&left.unhealthy_count)
                .then_with(|| left_project.cmp(right_project))
        });

        cards
            .into_iter()
            .map(|(project, card)| {
                let badge = if card.unhealthy_count == 0 {
                    "GREEN"
                } else if card.unhealthy_count < card.instance_count {
                    "YELLOW"
                } else {
                    "RED"
                };
                let avg_cpu = if card.cpu_samples == 0 {
                    0.0
                } else {
                    let samples = u32::try_from(card.cpu_samples).unwrap_or(u32::MAX);
                    card.cpu_sum / f64::from(samples)
                };
                let avg_attribution = if card.attribution_samples == 0 {
                    0
                } else {
                    let samples = u64::try_from(card.attribution_samples).unwrap_or(u64::MAX);
                    u8::try_from(
                        card.attribution_sum
                            .saturating_add(samples / 2)
                            .saturating_div(samples)
                            .min(100),
                    )
                    .unwrap_or(100)
                };

                Line::from(format!(
                    "{badge} {project}: inst={} docs={} pending={} search={} cpu={avg_cpu:.1}% attr={}%",
                    card.instance_count,
                    card.docs,
                    card.pending,
                    card.searches,
                    avg_attribution
                ))
            })
            .collect()
    }

    fn pipeline_health_lines(&self) -> Vec<Line<'static>> {
        let metrics = self.state.control_plane_metrics();
        let health = self.state.control_plane_health();
        let (badge, phase, error_hint, recovery) = match health {
            crate::state::ControlPlaneHealth::Healthy => (
                "GREEN",
                "Refined (quality enabled)",
                "none",
                "No action required.",
            ),
            crate::state::ControlPlaneHealth::Degraded => {
                let error_hint =
                    if metrics.ingestion_lag_events > 0 && metrics.event_throughput_eps < 1.0 {
                        "SearchError::Cancelled{phase=\"quality_refine\",reason=\"backpressure\"}"
                    } else {
                        "SearchError::EmbeddingFailed{model=\"quality\"}"
                    };
                (
                    "YELLOW",
                    "RefinementFailed (fast-only fallback)",
                    error_hint,
                    "Scale ingestion/embedding workers; verify quality model availability.",
                )
            }
            crate::state::ControlPlaneHealth::Critical => {
                let error_hint = if metrics.dead_letter_events >= 20 {
                    "SearchError::IndexCorrupted{path=\"<fleet index>\"}"
                } else {
                    "SearchError::EmbedderUnavailable{model=\"quality\"}"
                };
                (
                    "RED",
                    "RefinementFailed (error)",
                    error_hint,
                    "Run index validation + durability repair; restart failed embedders before enabling refinement.",
                )
            }
        };

        let quality_mode = if matches!(health, crate::state::ControlPlaneHealth::Healthy) {
            "active"
        } else {
            "skipped"
        };
        let index_status = if matches!(health, crate::state::ControlPlaneHealth::Critical) {
            "error"
        } else if matches!(health, crate::state::ControlPlaneHealth::Degraded) {
            "degraded"
        } else {
            "healthy"
        };
        let durability_status = if metrics.storage_utilization() >= 0.95 {
            "at-risk"
        } else {
            "enabled"
        };

        vec![
            Line::from(format!("{badge} phase={phase}")),
            Line::from(format!(
                "Embedders: fast=loaded quality={quality_mode} | Index={index_status} | Durability={durability_status}"
            )),
            Line::from(format!("Error hint: {error_hint}")),
            Line::from(format!("Recovery: {recovery}")),
        ]
    }

    #[allow(clippy::missing_const_for_fn)]
    fn empty_state_message(&self) -> &'static str {
        if !self.state.has_data() {
            "Scanning for instances..."
        } else if matches!(
            self.state.control_plane_health(),
            crate::state::ControlPlaneHealth::Critical
        ) {
            "Discovery failed. Inspect control-plane diagnostics and lifecycle events."
        } else {
            "No frankensearch instances found. Start a frankensearch-enabled application to begin monitoring."
        }
    }

    fn dashboard_signal_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        lines.extend(self.selected_monitor_lines());
        lines.push(Line::from(String::new()));
        lines.push(Line::from("Project Summary Cards"));
        lines.extend(self.project_summary_lines());
        lines.push(Line::from(String::new()));
        lines.push(Line::from("Pipeline Health"));
        lines.extend(self.pipeline_health_lines());
        lines.push(Line::from(String::new()));
        lines.push(Line::from(
            "Drilldown: Enter project | s live stream | t timeline | a analytics",
        ));
        lines
    }

    /// Build the instance table rows.
    fn build_rows(&self) -> Vec<Row<'_>> {
        let fleet = self.state.fleet();
        self.visible_instances()
            .into_iter()
            .enumerate()
            .map(|(i, inst)| {
                let health = if inst.healthy { "OK" } else { "WARN" };
                let resources = fleet
                    .resources
                    .get(&inst.id)
                    .map_or_else(|| "-".to_string(), |r| format!("{:.1}%", r.cpu_percent));
                let attribution = fleet.attribution_for(&inst.id).map_or_else(
                    || "n/a".to_owned(),
                    |value| format!("{}%", value.confidence_score),
                );

                let style = if i == self.selected_row {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else if !inst.healthy {
                    Style::default().fg(ratatui::style::Color::Red)
                } else {
                    Style::default()
                };

                let cells = if self.view.density.show_inline_metrics() {
                    vec![
                        health.to_string(),
                        inst.project.clone(),
                        inst.id.clone(),
                        format!("{}", inst.doc_count),
                        format!("{}", inst.pending_jobs),
                        resources,
                        attribution,
                    ]
                } else {
                    vec![
                        health.to_string(),
                        inst.project.clone(),
                        inst.id.clone(),
                        format!("{}", inst.doc_count),
                        format!("{}", inst.pending_jobs),
                        attribution,
                    ]
                };

                Row::new(cells)
                    .style(style)
                    .height(self.view.density.row_height())
            })
            .collect()
    }

    /// Number of instances currently visible to this screen.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.visible_instances().len()
    }

    /// Selected project name from the visible instance cursor.
    #[must_use]
    pub fn selected_project(&self) -> Option<&str> {
        self.selected_instance()
            .map(|instance| instance.project.as_str())
    }
}

impl Default for FleetOverviewScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for FleetOverviewScreen {
    fn id(&self) -> &ScreenId {
        &self.id
    }

    fn title(&self) -> &'static str {
        "Fleet Overview"
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_lines)]
    fn render(&self, frame: &mut Frame<'_>, _ctx: &ScreenContext) {
        let area = frame.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5)])
            .split(area);

        // Header with summary stats.
        let fleet = self.state.fleet();
        let visible = self.visible_instances();
        let visible_count = visible.len();
        let visible_healthy = visible.iter().filter(|inst| inst.healthy).count();
        let visible_docs: u64 = visible.iter().map(|inst| inst.doc_count).sum();
        let visible_pending: u64 = visible.iter().map(|inst| inst.pending_jobs).sum();
        let health_badge = self.state.control_plane_health().badge();
        let summary = format!(
            " {visible_count}/{} instances | {visible_healthy} healthy | {visible_docs} docs | {visible_pending} pending | {} density | {health_badge}",
            fleet.instance_count(),
            self.view.density,
        );
        let header = Paragraph::new(Line::from(vec![
            Span::styled("Fleet: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(summary),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Fleet Overview "),
        );
        frame.render_widget(header, chunks[0]);

        let body_chunks = if chunks[1].width >= 120 {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
                .split(chunks[1])
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
                .split(chunks[1])
        };

        let primary_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Length(4),
                Constraint::Min(5),
            ])
            .split(body_chunks[0]);

        let kpi_tiles = Paragraph::new(self.kpi_tile_lines()).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" KPI Tile Grid "),
        );
        frame.render_widget(kpi_tiles, primary_chunks[0]);

        let sparkline = Paragraph::new(self.status_sparkline_lines()).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Status Sparkline Strip "),
        );
        frame.render_widget(sparkline, primary_chunks[1]);

        // Instance table.
        let show_metrics = self.view.density.show_inline_metrics();
        let header_row = if show_metrics {
            Row::new(vec![
                "Health", "Project", "Instance", "Docs", "Pending", "CPU", "Attr",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD))
        } else {
            Row::new(vec![
                "Health", "Project", "Instance", "Docs", "Pending", "Attr",
            ])
            .style(Style::default().add_modifier(Modifier::BOLD))
        };

        let rows = self.build_rows();
        if rows.is_empty() {
            let empty = Paragraph::new(self.empty_state_message())
                .block(Block::default().borders(Borders::ALL).title(" Instances "));
            frame.render_widget(empty, primary_chunks[2]);
        } else {
            let table = if show_metrics {
                Table::new(
                    rows,
                    [
                        Constraint::Length(6),
                        Constraint::Length(12),
                        Constraint::Length(15),
                        Constraint::Length(10),
                        Constraint::Length(10),
                        Constraint::Length(8),
                        Constraint::Length(8),
                    ],
                )
                .header(header_row)
                .block(Block::default().borders(Borders::ALL).title(" Instances "))
            } else {
                Table::new(
                    rows,
                    [
                        Constraint::Length(6),
                        Constraint::Length(14),
                        Constraint::Length(16),
                        Constraint::Length(10),
                        Constraint::Length(10),
                        Constraint::Length(8),
                    ],
                )
                .header(header_row)
                .block(Block::default().borders(Borders::ALL).title(" Instances "))
            };

            frame.render_widget(table, primary_chunks[2]);
        }

        let details = Paragraph::new(self.dashboard_signal_lines()).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Fleet Dashboard Signals "),
        );
        frame.render_widget(details, body_chunks[1]);
    }

    fn handle_input(&mut self, event: &InputEvent, _ctx: &ScreenContext) -> ScreenAction {
        if let InputEvent::Key(key, _mods) = event {
            match key {
                crossterm::event::KeyCode::Up | crossterm::event::KeyCode::Char('k') => {
                    if self.selected_row > 0 {
                        self.selected_row -= 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Down | crossterm::event::KeyCode::Char('j') => {
                    let count = self.instance_count();
                    if count > 0 && self.selected_row < count - 1 {
                        self.selected_row += 1;
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Enter => {
                    if self.instance_count() > 0 {
                        return ScreenAction::Navigate(self.project_screen_id.clone());
                    }
                    return ScreenAction::Consumed;
                }
                crossterm::event::KeyCode::Char('s') => {
                    return ScreenAction::Navigate(self.live_stream_screen_id.clone());
                }
                crossterm::event::KeyCode::Char('t') => {
                    return ScreenAction::Navigate(self.timeline_screen_id.clone());
                }
                crossterm::event::KeyCode::Char('a') => {
                    return ScreenAction::Navigate(self.analytics_screen_id.clone());
                }
                _ => {}
            }
        }
        ScreenAction::Ignored
    }

    fn semantic_role(&self) -> &'static str {
        "grid"
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
    use crate::presets::ViewState;
    use crate::state::{FleetSnapshot, ResourceMetrics, SearchMetrics};

    #[test]
    fn fleet_screen_default() {
        let screen = FleetOverviewScreen::new();
        assert_eq!(screen.id(), &ScreenId::new("ops.fleet"));
        assert_eq!(screen.title(), "Fleet Overview");
        assert_eq!(screen.semantic_role(), "grid");
    }

    #[test]
    fn fleet_screen_empty_state() {
        let screen = FleetOverviewScreen::new();
        let rows = screen.build_rows();
        assert!(rows.is_empty());
    }

    #[test]
    fn fleet_screen_with_data() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "a".to_string(),
                    project: "test".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 100,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "b".to_string(),
                    project: "test2".to_string(),
                    pid: None,
                    healthy: false,
                    doc_count: 200,
                    pending_jobs: 50,
                },
            ],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());
        let rows = screen.build_rows();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn fleet_screen_navigation() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "a".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "b".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 2,
                    pending_jobs: 0,
                },
            ],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());

        assert_eq!(screen.selected_row, 0);

        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.fleet"),
            terminal_width: 80,
            terminal_height: 24,
            focused: true,
        };

        // Move down.
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        );
        let result = screen.handle_input(&event, &ctx);
        assert_eq!(result, ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);

        // Don't go past end.
        let result = screen.handle_input(&event, &ctx);
        assert_eq!(result, ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 1);

        // Move up.
        let event = InputEvent::Key(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
        );
        let result = screen.handle_input(&event, &ctx);
        assert_eq!(result, ScreenAction::Consumed);
        assert_eq!(screen.selected_row, 0);
    }

    #[test]
    fn enter_navigates_to_project_screen() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "a".to_string(),
                project: "cass".to_string(),
                pid: None,
                healthy: true,
                doc_count: 1,
                pending_jobs: 0,
            }],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());

        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.fleet"),
            terminal_width: 80,
            terminal_height: 24,
            focused: true,
        };

        let event = InputEvent::Key(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&event, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.project"))
        );
    }

    #[test]
    fn drilldown_shortcuts_navigate_to_defaults() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "a".to_string(),
                project: "cass".to_string(),
                pid: None,
                healthy: true,
                doc_count: 1,
                pending_jobs: 0,
            }],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.fleet"),
            terminal_width: 80,
            terminal_height: 24,
            focused: true,
        };

        let stream = InputEvent::Key(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&stream, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.live_stream"))
        );

        let timeline = InputEvent::Key(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&timeline, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.timeline"))
        );

        let analytics = InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&analytics, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.timeline"))
        );
    }

    #[test]
    fn configured_shortcuts_use_custom_destinations() {
        let mut screen = FleetOverviewScreen::new();
        screen.set_live_stream_screen_id(ScreenId::new("ops.custom_stream"));
        screen.set_timeline_screen_id(ScreenId::new("ops.custom_timeline"));
        screen.set_analytics_screen_id(ScreenId::new("ops.custom_analytics"));
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "a".to_string(),
                project: "cass".to_string(),
                pid: None,
                healthy: true,
                doc_count: 1,
                pending_jobs: 0,
            }],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());
        let ctx = ScreenContext {
            active_screen: ScreenId::new("ops.fleet"),
            terminal_width: 80,
            terminal_height: 24,
            focused: true,
        };

        let stream = InputEvent::Key(
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&stream, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.custom_stream"))
        );

        let timeline = InputEvent::Key(
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&timeline, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.custom_timeline"))
        );

        let analytics = InputEvent::Key(
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyModifiers::NONE,
        );
        assert_eq!(
            screen.handle_input(&analytics, &ctx),
            ScreenAction::Navigate(ScreenId::new("ops.custom_analytics"))
        );
    }

    #[test]
    fn selected_project_tracks_cursor() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "a".to_string(),
                    project: "cass".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "b".to_string(),
                    project: "xf".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1,
                    pending_jobs: 0,
                },
            ],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());
        assert_eq!(screen.selected_project(), Some("cass"));

        screen.selected_row = 1;
        assert_eq!(screen.selected_project(), Some("xf"));
    }

    #[test]
    fn hide_healthy_filter_applies_to_rows() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "healthy".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1,
                    pending_jobs: 0,
                },
                crate::state::InstanceInfo {
                    id: "warn".to_string(),
                    project: "p".to_string(),
                    pid: None,
                    healthy: false,
                    doc_count: 2,
                    pending_jobs: 1,
                },
            ],
            ..FleetSnapshot::default()
        });
        let view = ViewState {
            hide_healthy: true,
            ..ViewState::default()
        };
        screen.update_state(&state, &view);

        let rows = screen.build_rows();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn selected_monitor_includes_fleet_percentiles() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "alpha".to_string(),
                    project: "a".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 100,
                    pending_jobs: 2,
                },
                crate::state::InstanceInfo {
                    id: "beta".to_string(),
                    project: "b".to_string(),
                    pid: None,
                    healthy: false,
                    doc_count: 800,
                    pending_jobs: 30,
                },
            ],
            ..FleetSnapshot::default()
        };
        fleet.resources.insert(
            "alpha".to_owned(),
            ResourceMetrics {
                cpu_percent: 15.0,
                memory_bytes: 100 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.resources.insert(
            "beta".to_owned(),
            ResourceMetrics {
                cpu_percent: 70.0,
                memory_bytes: 400 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.search_metrics.insert(
            "alpha".to_owned(),
            SearchMetrics {
                total_searches: 10,
                avg_latency_us: 100,
                p95_latency_us: 300,
                refined_count: 2,
            },
        );
        fleet.search_metrics.insert(
            "beta".to_owned(),
            SearchMetrics {
                total_searches: 5,
                avg_latency_us: 200,
                p95_latency_us: 700,
                refined_count: 1,
            },
        );

        state.update_fleet(fleet);
        screen.update_state(&state, &ViewState::default());
        screen.selected_row = 1;
        let details = screen.selected_monitor_text();

        assert!(details.contains("Docs: 800 (project p100 | fleet p100)"));
        assert!(details.contains("Pending: 30 (project p100 | fleet p100)"));
        assert!(details.contains("CPU: 70.0% (project p100 | fleet p100)"));
        assert!(details.contains("Memory: 400 MiB (project p100 | fleet p100)"));
        assert!(
            details.contains("Search: total=5 avg=200us p95=700us (project p100 | fleet p100)")
        );
    }

    #[test]
    fn selected_monitor_handles_missing_metrics() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        state.update_fleet(FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "solo".to_string(),
                project: "demo".to_string(),
                pid: None,
                healthy: true,
                doc_count: 5,
                pending_jobs: 0,
            }],
            ..FleetSnapshot::default()
        });
        screen.update_state(&state, &ViewState::default());

        let details = screen.selected_monitor_text();
        assert!(details.contains("CPU: n/a"));
        assert!(details.contains("Memory: n/a"));
        assert!(details.contains("Search: n/a"));
    }

    #[test]
    fn selected_monitor_reports_project_vs_fleet_percentiles() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "alpha".to_string(),
                    project: "a".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 100,
                    pending_jobs: 1,
                },
                crate::state::InstanceInfo {
                    id: "beta".to_string(),
                    project: "a".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 300,
                    pending_jobs: 5,
                },
                crate::state::InstanceInfo {
                    id: "gamma".to_string(),
                    project: "b".to_string(),
                    pid: None,
                    healthy: true,
                    doc_count: 1000,
                    pending_jobs: 20,
                },
            ],
            ..FleetSnapshot::default()
        };
        fleet.resources.insert(
            "alpha".to_owned(),
            ResourceMetrics {
                cpu_percent: 20.0,
                memory_bytes: 100 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.resources.insert(
            "beta".to_owned(),
            ResourceMetrics {
                cpu_percent: 40.0,
                memory_bytes: 300 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.resources.insert(
            "gamma".to_owned(),
            ResourceMetrics {
                cpu_percent: 95.0,
                memory_bytes: 1000 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
        );
        fleet.search_metrics.insert(
            "alpha".to_owned(),
            SearchMetrics {
                total_searches: 10,
                avg_latency_us: 100,
                p95_latency_us: 150,
                refined_count: 2,
            },
        );
        fleet.search_metrics.insert(
            "beta".to_owned(),
            SearchMetrics {
                total_searches: 10,
                avg_latency_us: 200,
                p95_latency_us: 300,
                refined_count: 2,
            },
        );
        fleet.search_metrics.insert(
            "gamma".to_owned(),
            SearchMetrics {
                total_searches: 10,
                avg_latency_us: 500,
                p95_latency_us: 1200,
                refined_count: 2,
            },
        );

        state.update_fleet(fleet);
        screen.update_state(&state, &ViewState::default());
        screen.selected_row = 1;
        let details = screen.selected_monitor_text();

        assert!(details.contains("Docs: 300 (project p100 | fleet p67)"));
        assert!(details.contains("Pending: 5 (project p100 | fleet p67)"));
        assert!(details.contains("CPU: 40.0% (project p100 | fleet p67)"));
        assert!(details.contains("Memory: 300 MiB (project p100 | fleet p67)"));
        assert!(
            details.contains("Search: total=10 avg=200us p95=300us (project p100 | fleet p67)")
        );
    }

    #[test]
    fn dashboard_signals_include_project_cards_and_pipeline() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "cass-1".to_string(),
                project: "cass".to_string(),
                pid: None,
                healthy: true,
                doc_count: 10,
                pending_jobs: 2,
            }],
            ..FleetSnapshot::default()
        };
        fleet.search_metrics.insert(
            "cass-1".to_owned(),
            SearchMetrics {
                total_searches: 44,
                avg_latency_us: 900,
                p95_latency_us: 1800,
                refined_count: 12,
            },
        );
        state.update_fleet(fleet);
        screen.update_state(&state, &ViewState::default());

        let text = screen
            .dashboard_signal_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Project Summary Cards"));
        assert!(text.contains("Pipeline Health"));
        assert!(text.contains("GREEN cass: inst=1"));
    }

    #[test]
    fn kpi_tiles_include_fleet_and_attribution_metrics() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![
                crate::state::InstanceInfo {
                    id: "cass-a".to_owned(),
                    project: "cass".to_owned(),
                    pid: None,
                    healthy: true,
                    doc_count: 100,
                    pending_jobs: 20,
                },
                crate::state::InstanceInfo {
                    id: "xf-a".to_owned(),
                    project: "xf".to_owned(),
                    pid: None,
                    healthy: false,
                    doc_count: 50,
                    pending_jobs: 10,
                },
            ],
            ..FleetSnapshot::default()
        };
        fleet.search_metrics.insert(
            "cass-a".to_owned(),
            SearchMetrics {
                total_searches: 120,
                avg_latency_us: 100,
                p95_latency_us: 150,
                refined_count: 90,
            },
        );
        fleet.search_metrics.insert(
            "xf-a".to_owned(),
            SearchMetrics {
                total_searches: 80,
                avg_latency_us: 200,
                p95_latency_us: 320,
                refined_count: 40,
            },
        );
        fleet.attribution.insert(
            "cass-a".to_owned(),
            crate::state::InstanceAttribution {
                project_key_hint: Some("cass".to_owned()),
                host_name_hint: Some("cass-host".to_owned()),
                resolved_project: "cass".to_owned(),
                confidence_score: 95,
                reason_code: "attribution.adapter_identity".to_owned(),
                collision: false,
                evidence_trace: vec!["adapter".to_owned()],
            },
        );
        fleet.attribution.insert(
            "xf-a".to_owned(),
            crate::state::InstanceAttribution {
                project_key_hint: Some("xf".to_owned()),
                host_name_hint: Some("xf-host".to_owned()),
                resolved_project: "xf".to_owned(),
                confidence_score: 75,
                reason_code: "attribution.project_key_hint".to_owned(),
                collision: true,
                evidence_trace: vec!["project_key_hint".to_owned()],
            },
        );
        state.update_fleet(fleet);
        screen.update_state(&state, &ViewState::default());

        let text = screen
            .kpi_tile_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("vis=2/2"));
        assert!(text.contains("[Healthy] 1/2 (50%)"));
        assert!(text.contains("[Corpus] docs=150 | pending=30 (17%)"));
        assert!(text.contains("[Search] total=200 avg/inst=100"));
        assert!(text.contains("[Attribution] avg=85% collisions=1"));
        assert!(text.contains("[Control] CP:OK"));
    }

    #[test]
    fn status_sparkline_strip_is_deterministic() {
        let mut screen = FleetOverviewScreen::new();
        let mut state = AppState::new();
        let mut fleet = FleetSnapshot {
            instances: vec![crate::state::InstanceInfo {
                id: "cass-1".to_owned(),
                project: "cass".to_owned(),
                pid: None,
                healthy: true,
                doc_count: 200,
                pending_jobs: 50,
            }],
            ..FleetSnapshot::default()
        };
        fleet.search_metrics.insert(
            "cass-1".to_owned(),
            SearchMetrics {
                total_searches: 400,
                avg_latency_us: 500,
                p95_latency_us: 900,
                refined_count: 220,
            },
        );
        state.update_fleet(fleet);
        state.update_control_plane(crate::state::ControlPlaneMetrics {
            ingestion_lag_events: 2_000,
            storage_bytes: 600,
            storage_limit_bytes: 1_000,
            frame_time_ms: 25.0,
            discovery_latency_ms: 1_500,
            event_throughput_eps: 10.0,
            rss_bytes: 500,
            rss_limit_bytes: 1_000,
            dead_letter_events: 10,
        });
        screen.update_state(&state, &ViewState::default());

        let text = screen
            .status_sparkline_lines()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        let strip_line = text
            .lines()
            .find(|line| line.starts_with("H Q T L S R D X | "))
            .expect("sparkline strip line should exist");
        let strip = strip_line
            .split_once("| ")
            .map(|(_, rhs)| rhs)
            .expect("sparkline strip should include separator");
        assert_eq!(strip.chars().count(), 8);
        assert!(strip.chars().all(|glyph| "▁▂▃▄▅▆▇█".contains(glyph)));
        assert!(text.contains("H=100%"));
        assert!(text.contains("Q=80%"));
        assert!(text.contains("T=50%"));
        assert!(text.contains("L=90%"));
        assert!(text.contains("S=40%"));
        assert!(text.contains("R=50%"));
        assert!(text.contains("D=95%"));
        assert!(text.contains("X=70%"));
    }
}
