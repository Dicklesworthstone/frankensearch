//! Data source trait and mock implementation.
//!
//! The [`DataSource`] trait decouples the TUI from the concrete data backend.
//! Product screens read from a `DataSource`; the real implementation queries
//! FrankenSQLite (wired in via bd-2yu.4.3), while [`MockDataSource`] provides
//! test data for development and testing.

use crate::state::{FleetSnapshot, InstanceInfo, ResourceMetrics, SearchMetrics};

// ─── Time Window ─────────────────────────────────────────────────────────────

/// Time window for metric queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeWindow {
    /// Last 1 minute.
    OneMinute,
    /// Last 15 minutes.
    FifteenMinutes,
    /// Last 1 hour.
    OneHour,
    /// Last 6 hours.
    SixHours,
    /// Last 24 hours.
    TwentyFourHours,
    /// Last 3 days.
    ThreeDays,
    /// Last 1 week.
    OneWeek,
}

impl TimeWindow {
    /// All windows in ascending order.
    pub const ALL: &'static [Self] = &[
        Self::OneMinute,
        Self::FifteenMinutes,
        Self::OneHour,
        Self::SixHours,
        Self::TwentyFourHours,
        Self::ThreeDays,
        Self::OneWeek,
    ];

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OneMinute => "1m",
            Self::FifteenMinutes => "15m",
            Self::OneHour => "1h",
            Self::SixHours => "6h",
            Self::TwentyFourHours => "24h",
            Self::ThreeDays => "3d",
            Self::OneWeek => "1w",
        }
    }

    /// Duration in seconds.
    #[must_use]
    pub const fn seconds(self) -> u64 {
        match self {
            Self::OneMinute => 60,
            Self::FifteenMinutes => 15 * 60,
            Self::OneHour => 3600,
            Self::SixHours => 6 * 3600,
            Self::TwentyFourHours => 24 * 3600,
            Self::ThreeDays => 3 * 24 * 3600,
            Self::OneWeek => 7 * 24 * 3600,
        }
    }
}

impl std::fmt::Display for TimeWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── Data Source Trait ────────────────────────────────────────────────────────

/// Trait for data backends that feed the ops TUI.
///
/// Synchronous interface for the render loop. Background async tasks
/// populate the backing store; the `DataSource` provides read access.
///
/// The real implementation queries FrankenSQLite (bd-2yu.4.3).
/// [`MockDataSource`] provides synthetic data for development.
pub trait DataSource: Send {
    /// Get the current fleet snapshot.
    fn fleet_snapshot(&self) -> FleetSnapshot;

    /// Get search metrics for a given time window.
    fn search_metrics(&self, instance_id: &str, window: TimeWindow) -> Option<SearchMetrics>;

    /// Get resource metrics for a given instance.
    fn resource_metrics(&self, instance_id: &str) -> Option<ResourceMetrics>;
}

// ─── Mock Data Source ────────────────────────────────────────────────────────

/// Mock data source for development and testing.
///
/// Provides synthetic fleet data so the TUI can be developed and tested
/// independently of the real FrankenSQLite backend.
pub struct MockDataSource {
    snapshot: FleetSnapshot,
}

impl MockDataSource {
    /// Create a mock with sample data.
    #[must_use]
    pub fn sample() -> Self {
        let mut snapshot = FleetSnapshot::default();

        snapshot.instances.push(InstanceInfo {
            id: "cass-001".to_string(),
            project: "cass".to_string(),
            pid: Some(12345),
            healthy: true,
            doc_count: 48_532,
            pending_jobs: 7,
        });
        snapshot.instances.push(InstanceInfo {
            id: "xf-001".to_string(),
            project: "xf".to_string(),
            pid: Some(23456),
            healthy: true,
            doc_count: 12_801,
            pending_jobs: 0,
        });
        snapshot.instances.push(InstanceInfo {
            id: "amail-001".to_string(),
            project: "agent-mail".to_string(),
            pid: Some(34567),
            healthy: false,
            doc_count: 91_204,
            pending_jobs: 1_542,
        });

        snapshot.resources.insert(
            "cass-001".to_string(),
            ResourceMetrics {
                cpu_percent: 12.5,
                memory_bytes: 256 * 1024 * 1024,
                io_read_bytes: 1024 * 1024,
                io_write_bytes: 512 * 1024,
            },
        );
        snapshot.resources.insert(
            "xf-001".to_string(),
            ResourceMetrics {
                cpu_percent: 3.2,
                memory_bytes: 128 * 1024 * 1024,
                io_read_bytes: 256 * 1024,
                io_write_bytes: 64 * 1024,
            },
        );
        snapshot.resources.insert(
            "amail-001".to_string(),
            ResourceMetrics {
                cpu_percent: 87.3,
                memory_bytes: 512 * 1024 * 1024,
                io_read_bytes: 4 * 1024 * 1024,
                io_write_bytes: 2 * 1024 * 1024,
            },
        );

        snapshot.search_metrics.insert(
            "cass-001".to_string(),
            SearchMetrics {
                total_searches: 1_247,
                avg_latency_us: 850,
                p95_latency_us: 2_100,
                refined_count: 312,
            },
        );
        snapshot.search_metrics.insert(
            "xf-001".to_string(),
            SearchMetrics {
                total_searches: 89,
                avg_latency_us: 1_200,
                p95_latency_us: 3_500,
                refined_count: 15,
            },
        );

        Self { snapshot }
    }

    /// Create an empty mock (no instances).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            snapshot: FleetSnapshot::default(),
        }
    }
}

impl DataSource for MockDataSource {
    fn fleet_snapshot(&self) -> FleetSnapshot {
        self.snapshot.clone()
    }

    fn search_metrics(&self, instance_id: &str, _window: TimeWindow) -> Option<SearchMetrics> {
        self.snapshot.search_metrics.get(instance_id).cloned()
    }

    fn resource_metrics(&self, instance_id: &str) -> Option<ResourceMetrics> {
        self.snapshot.resources.get(instance_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_sample_has_instances() {
        let mock = MockDataSource::sample();
        let snap = mock.fleet_snapshot();
        assert_eq!(snap.instance_count(), 3);
        assert_eq!(snap.healthy_count(), 2);
    }

    #[test]
    fn mock_empty_has_no_instances() {
        let mock = MockDataSource::empty();
        let snap = mock.fleet_snapshot();
        assert_eq!(snap.instance_count(), 0);
    }

    #[test]
    fn mock_search_metrics() {
        let mock = MockDataSource::sample();
        let metrics = mock.search_metrics("cass-001", TimeWindow::OneHour);
        assert!(metrics.is_some());
        assert!(metrics.unwrap().total_searches > 0);
    }

    #[test]
    fn mock_resource_metrics() {
        let mock = MockDataSource::sample();
        let metrics = mock.resource_metrics("xf-001");
        assert!(metrics.is_some());
    }

    #[test]
    fn mock_unknown_instance() {
        let mock = MockDataSource::sample();
        assert!(mock.search_metrics("unknown", TimeWindow::OneMinute).is_none());
        assert!(mock.resource_metrics("unknown").is_none());
    }

    #[test]
    fn time_window_all() {
        assert_eq!(TimeWindow::ALL.len(), 7);
    }

    #[test]
    fn time_window_labels() {
        assert_eq!(TimeWindow::OneMinute.label(), "1m");
        assert_eq!(TimeWindow::OneWeek.label(), "1w");
    }

    #[test]
    fn time_window_seconds() {
        assert_eq!(TimeWindow::OneHour.seconds(), 3600);
        assert_eq!(TimeWindow::OneWeek.seconds(), 7 * 24 * 3600);
    }

    #[test]
    fn time_window_display() {
        assert_eq!(TimeWindow::FifteenMinutes.to_string(), "15m");
    }
}
