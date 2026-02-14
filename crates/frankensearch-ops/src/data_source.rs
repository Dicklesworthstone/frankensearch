//! Data source trait and mock implementation.
//!
//! The [`DataSource`] trait decouples the TUI from the concrete data backend.
//! Product screens read from a `DataSource`; the real implementation queries
//! `FrankenSQLite` (wired in via bd-2yu.4.3), while [`MockDataSource`] provides
//! test data for development and testing.

use crate::state::{
    ControlPlaneMetrics, FleetSnapshot, InstanceAttribution, InstanceInfo, InstanceLifecycle,
    LifecycleSignal, LifecycleTrackerConfig, ProjectAttributionResolver, ProjectLifecycleTracker,
    ResourceMetrics, SearchMetrics,
};
use crate::{DiscoveredInstance, DiscoveryStatus};

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
/// The real implementation queries `FrankenSQLite` (bd-2yu.4.3).
/// [`MockDataSource`] provides synthetic data for development.
pub trait DataSource: Send {
    /// Get the current fleet snapshot.
    fn fleet_snapshot(&self) -> FleetSnapshot;

    /// Get search metrics for a given time window.
    fn search_metrics(&self, instance_id: &str, window: TimeWindow) -> Option<SearchMetrics>;

    /// Get resource metrics for a given instance.
    fn resource_metrics(&self, instance_id: &str) -> Option<ResourceMetrics>;

    /// Get control-plane self-monitoring metrics.
    fn control_plane_metrics(&self) -> ControlPlaneMetrics;

    /// Attribution metadata for an instance.
    fn attribution(&self, instance_id: &str) -> Option<InstanceAttribution>;

    /// Lifecycle snapshot for an instance.
    fn lifecycle(&self, instance_id: &str) -> Option<InstanceLifecycle>;
}

// ─── Mock Data Source ────────────────────────────────────────────────────────

/// Mock data source for development and testing.
///
/// Provides synthetic fleet data so the TUI can be developed and tested
/// independently of the real `FrankenSQLite` backend.
pub struct MockDataSource {
    snapshot: FleetSnapshot,
    control_plane: ControlPlaneMetrics,
}

impl MockDataSource {
    /// Create a mock with sample data.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn sample() -> Self {
        let mut snapshot = FleetSnapshot::default();
        let resolver = ProjectAttributionResolver;

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

        snapshot.attribution.insert(
            "cass-001".to_string(),
            resolver.resolve(
                Some("cass"),
                Some("cass-devbox"),
                Some("coding-agent-session-search"),
            ),
        );
        snapshot.attribution.insert(
            "xf-001".to_string(),
            resolver.resolve(Some("xf"), Some("xf-worker"), Some("xf")),
        );
        snapshot.attribution.insert(
            "amail-001".to_string(),
            resolver.resolve(
                Some("agent-mail"),
                Some("mail-runner-42"),
                Some("mcp_agent_mail_rust"),
            ),
        );

        let mut cass_lifecycle = InstanceLifecycle::new(1_000);
        cass_lifecycle.apply_signal(LifecycleSignal::Heartbeat, 1_250, None);
        snapshot
            .lifecycle
            .insert("cass-001".to_string(), cass_lifecycle);

        let mut xf_lifecycle = InstanceLifecycle::new(1_000);
        xf_lifecycle.apply_signal(LifecycleSignal::Heartbeat, 1_300, None);
        snapshot
            .lifecycle
            .insert("xf-001".to_string(), xf_lifecycle);

        let mut amail_lifecycle = InstanceLifecycle::new(1_000);
        amail_lifecycle.apply_signal(LifecycleSignal::Heartbeat, 1_100, None);
        amail_lifecycle.mark_stale_if_heartbeat_gap(9_500, 5_000);
        snapshot
            .lifecycle
            .insert("amail-001".to_string(), amail_lifecycle);

        let control_plane = ControlPlaneMetrics {
            ingestion_lag_events: 1_549,
            storage_bytes: 384 * 1024 * 1024,
            storage_limit_bytes: 512 * 1024 * 1024,
            frame_time_ms: 19.8,
            discovery_latency_ms: 320,
            event_throughput_eps: 145.2,
            rss_bytes: 620 * 1024 * 1024,
            rss_limit_bytes: 1024 * 1024 * 1024,
            dead_letter_events: 2,
        };

        Self {
            snapshot,
            control_plane,
        }
    }

    /// Create an empty mock (no instances).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            snapshot: FleetSnapshot::default(),
            control_plane: ControlPlaneMetrics::default(),
        }
    }

    /// Build a mock snapshot from reconciled discovery output.
    ///
    /// This keeps view-level tests deterministic while exercising the same
    /// instance identity model used by the discovery engine.
    #[must_use]
    pub fn from_discovery(instances: &[DiscoveredInstance]) -> Self {
        let mut snapshot = FleetSnapshot::default();
        let now_ms = instances
            .iter()
            .map(|instance| instance.last_seen_ms)
            .max()
            .unwrap_or(0);
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig::default());
        let lifecycle_events = tracker.ingest_discovery(now_ms, instances);

        for instance in instances {
            let instance_id = instance.instance_id.clone();
            let attribution = tracker
                .attribution_for(&instance_id)
                .cloned()
                .unwrap_or_else(|| {
                    InstanceAttribution::unknown(
                        instance.project_key_hint.as_deref(),
                        instance.host_name.as_deref(),
                        "attribution.discovery.unresolved",
                    )
                });

            snapshot.instances.push(InstanceInfo {
                id: instance_id.clone(),
                project: attribution.resolved_project.clone(),
                pid: instance.pid,
                healthy: instance.status == DiscoveryStatus::Active,
                doc_count: 0,
                pending_jobs: 0,
            });
        }
        snapshot.attribution = tracker.attribution_snapshot();
        snapshot.lifecycle = tracker.lifecycle_snapshot();
        snapshot.lifecycle_events = lifecycle_events;
        snapshot
            .instances
            .sort_by(|left, right| left.id.cmp(&right.id));
        Self {
            snapshot,
            control_plane: ControlPlaneMetrics::default(),
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

    fn control_plane_metrics(&self) -> ControlPlaneMetrics {
        self.control_plane.clone()
    }

    fn attribution(&self, instance_id: &str) -> Option<InstanceAttribution> {
        self.snapshot.attribution.get(instance_id).cloned()
    }

    fn lifecycle(&self, instance_id: &str) -> Option<InstanceLifecycle> {
        self.snapshot.lifecycle.get(instance_id).cloned()
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
        assert!(
            mock.search_metrics("unknown", TimeWindow::OneMinute)
                .is_none()
        );
        assert!(mock.resource_metrics("unknown").is_none());
        assert!(mock.attribution("unknown").is_none());
        assert!(mock.lifecycle("unknown").is_none());
    }

    #[test]
    fn mock_control_plane_metrics_present() {
        let mock = MockDataSource::sample();
        let metrics = mock.control_plane_metrics();
        assert!(metrics.ingestion_lag_events > 0);
        assert!(metrics.storage_limit_bytes > metrics.storage_bytes / 2);
    }

    #[test]
    fn mock_attribution_metadata_present() {
        let mock = MockDataSource::sample();
        let attribution = mock
            .attribution("cass-001")
            .expect("cass attribution exists");
        assert_eq!(attribution.resolved_project, "coding_agent_session_search");
        assert!(attribution.confidence_score >= 80);
    }

    #[test]
    fn mock_lifecycle_contains_stale_instance() {
        let mock = MockDataSource::sample();
        let lifecycle = mock.lifecycle("amail-001").expect("amail lifecycle exists");
        assert_eq!(lifecycle.state, frankensearch_core::LifecycleState::Stale);
    }

    #[test]
    fn mock_from_discovery_maps_instance_health() {
        let instances = vec![
            DiscoveredInstance {
                instance_id: "inst-a".to_owned(),
                project_key_hint: Some("cass".to_owned()),
                host_name: Some("host-a".to_owned()),
                pid: Some(11),
                version: Some("0.1.0".to_owned()),
                first_seen_ms: 10,
                last_seen_ms: 20,
                status: DiscoveryStatus::Active,
                sources: vec![crate::DiscoverySignalKind::Process],
                identity_keys: vec!["hostpid:host-a:11".to_owned()],
            },
            DiscoveredInstance {
                instance_id: "inst-b".to_owned(),
                project_key_hint: None,
                host_name: Some("host-b".to_owned()),
                pid: Some(22),
                version: Some("0.1.0".to_owned()),
                first_seen_ms: 10,
                last_seen_ms: 20,
                status: DiscoveryStatus::Stale,
                sources: vec![crate::DiscoverySignalKind::Heartbeat],
                identity_keys: vec!["heartbeat:/tmp/inst-b".to_owned()],
            },
        ];

        let mock = MockDataSource::from_discovery(&instances);
        let fleet = mock.fleet_snapshot();
        assert_eq!(fleet.instance_count(), 2);
        assert_eq!(fleet.healthy_count(), 1);
        assert_eq!(fleet.instances[0].project, "coding_agent_session_search");
        assert_eq!(fleet.instances[1].project, "unknown");

        let inst_a_attr = mock
            .attribution("inst-a")
            .expect("inst-a attribution should be present");
        assert_eq!(inst_a_attr.resolved_project, "coding_agent_session_search");
        assert!(!inst_a_attr.collision);
        assert_eq!(inst_a_attr.reason_code, "attribution.telemetry_project_key");

        let inst_b_lifecycle = mock
            .lifecycle("inst-b")
            .expect("inst-b lifecycle should be present");
        assert_eq!(
            inst_b_lifecycle.state,
            frankensearch_core::LifecycleState::Stale
        );
        assert_eq!(inst_b_lifecycle.reason_code, "lifecycle.heartbeat_gap");
        assert!(
            !fleet.lifecycle_events().is_empty(),
            "lifecycle transitions should be emitted for timeline/alerts"
        );
    }

    #[test]
    fn mock_from_discovery_surfaces_attribution_collisions() {
        let instances = vec![DiscoveredInstance {
            instance_id: "inst-collision".to_owned(),
            project_key_hint: Some("xf".to_owned()),
            host_name: Some("cass-devbox".to_owned()),
            pid: Some(333),
            version: Some("0.3.0".to_owned()),
            first_seen_ms: 100,
            last_seen_ms: 120,
            status: DiscoveryStatus::Active,
            sources: vec![crate::DiscoverySignalKind::Heartbeat],
            identity_keys: vec!["hostpid:cass-devbox:333".to_owned()],
        }];

        let mock = MockDataSource::from_discovery(&instances);
        let attribution = mock
            .attribution("inst-collision")
            .expect("collision attribution should exist");
        assert_eq!(attribution.resolved_project, "xf");
        assert!(attribution.collision);
        assert_eq!(attribution.reason_code, "attribution.collision");

        let lifecycle = mock
            .lifecycle("inst-collision")
            .expect("collision lifecycle should exist");
        assert_eq!(lifecycle.state, frankensearch_core::LifecycleState::Healthy);

        let events = mock.fleet_snapshot().lifecycle_events;
        assert!(
            events
                .iter()
                .any(|event| event.instance_id == "inst-collision"),
            "collision instance should emit lifecycle events"
        );
        assert!(
            events.iter().all(|event| event.attribution_collision),
            "event metadata should carry attribution collision flag"
        );
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
