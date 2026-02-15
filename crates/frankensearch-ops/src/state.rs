//! Shared application state for async/sync bridge.
//!
//! The [`AppState`] holds fleet status, metrics, and connection info.
//! Background async tasks write updates; the synchronous render loop reads.
//! Thread safety is provided by the consumer's runtime (asupersync `RwLock`
//! when integrated; `std::sync::RwLock` for standalone testing).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

use crate::discovery::{DiscoveredInstance, DiscoveryStatus};
use frankensearch_core::host_adapter::resolve_host_project_attribution;
pub use frankensearch_core::{LifecycleSeverity, LifecycleState};
use serde::{Deserialize, Serialize};

// ─── Instance Info ───────────────────────────────────────────────────────────

/// Discovered frankensearch instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    /// Unique instance identifier.
    pub id: String,
    /// Host project name (e.g., "cass", "xf", "agent-mail").
    pub project: String,
    /// Process ID on the host machine.
    pub pid: Option<u32>,
    /// Whether the instance is currently healthy.
    pub healthy: bool,
    /// Number of indexed documents.
    pub doc_count: u64,
    /// Number of pending embedding jobs.
    pub pending_jobs: u64,
}

/// Attribution metadata attached to a discovered instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceAttribution {
    /// Raw project key hint from telemetry/discovery.
    pub project_key_hint: Option<String>,
    /// Raw host name hint from telemetry/discovery.
    pub host_name_hint: Option<String>,
    /// Canonical resolved project key, or `unknown`.
    pub resolved_project: String,
    /// Confidence score in `[0, 100]`.
    pub confidence_score: u8,
    /// Machine-stable reason code.
    pub reason_code: String,
    /// Whether competing project candidates were observed.
    pub collision: bool,
    /// Explainable attribution evidence trace (ordered strongest -> weakest).
    pub evidence_trace: Vec<String>,
}

impl InstanceAttribution {
    #[must_use]
    pub fn unknown(
        project_key_hint: Option<&str>,
        host_name_hint: Option<&str>,
        reason_code: impl Into<String>,
    ) -> Self {
        let project_key_hint = normalize_hint(project_key_hint);
        let host_name_hint = normalize_hint(host_name_hint);
        let reason_code = reason_code.into();
        let mut evidence_trace = Vec::new();
        if let Some(project_key_hint) = project_key_hint.as_deref() {
            evidence_trace.push(format!("project_key_hint={project_key_hint}"));
        }
        if let Some(host_name_hint) = host_name_hint.as_deref() {
            evidence_trace.push(format!("host_name_hint={host_name_hint}"));
        }
        evidence_trace.push("resolved_project=unknown".to_owned());
        evidence_trace.push(format!("reason={reason_code}"));

        Self {
            project_key_hint,
            host_name_hint,
            resolved_project: "unknown".to_owned(),
            confidence_score: 20,
            reason_code,
            collision: false,
            evidence_trace,
        }
    }
}

/// Deterministic project attribution resolver used by dashboards and alerts.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectAttributionResolver;

impl ProjectAttributionResolver {
    /// Resolve instance attribution from available project/host hints.
    #[must_use]
    pub fn resolve(
        self,
        project_key_hint: Option<&str>,
        host_name_hint: Option<&str>,
        adapter_identity_hint: Option<&str>,
    ) -> InstanceAttribution {
        let project_key_hint = normalize_hint(project_key_hint);
        let host_name_hint = normalize_hint(host_name_hint);
        let adapter_identity_hint = normalize_hint(adapter_identity_hint);

        let attribution = resolve_host_project_attribution(
            adapter_identity_hint.as_deref(),
            project_key_hint.as_deref(),
            host_name_hint.as_deref(),
        );

        let mut evidence_trace = Vec::new();
        if let Some(adapter_identity_hint) = adapter_identity_hint.as_deref() {
            evidence_trace.push(format!("adapter_identity_hint={adapter_identity_hint}"));
        }
        if let Some(project_key_hint) = project_key_hint.as_deref() {
            evidence_trace.push(format!("project_key_hint={project_key_hint}"));
        }
        if let Some(host_name_hint) = host_name_hint.as_deref() {
            evidence_trace.push(format!("host_name_hint={host_name_hint}"));
        }
        evidence_trace.push(format!(
            "resolved_project={}",
            attribution.resolved_project_key
        ));
        evidence_trace.push(format!("reason={}", attribution.reason_code));
        evidence_trace.push(format!("confidence_score={}", attribution.confidence_score));
        if attribution.collision {
            evidence_trace.push("collision=true".to_owned());
        }

        InstanceAttribution {
            project_key_hint,
            host_name_hint,
            resolved_project: attribution.resolved_project_key,
            confidence_score: attribution.confidence_score,
            reason_code: attribution.reason_code,
            collision: attribution.collision,
            evidence_trace,
        }
    }
}

fn normalize_hint(value: Option<&str>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        }
    })
}

/// Discrete lifecycle signals ingested by the tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleSignal {
    Start,
    Heartbeat,
    Degraded,
    Recovering,
    Stop,
}

/// One lifecycle transition decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleTransition {
    /// Prior lifecycle state.
    pub from: LifecycleState,
    /// Resulting lifecycle state.
    pub to: LifecycleState,
    /// Stable reason code explaining the transition.
    pub reason_code: String,
    /// Whether the transition changed state.
    pub changed: bool,
}

/// Current lifecycle snapshot for an instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceLifecycle {
    /// Current lifecycle state.
    pub state: LifecycleState,
    /// Current lifecycle severity.
    pub severity: LifecycleSeverity,
    /// Last transition reason code.
    pub reason_code: String,
    /// Last state transition timestamp (unix ms).
    pub last_transition_ms: u64,
    /// Last heartbeat timestamp (unix ms).
    pub last_heartbeat_ms: u64,
    /// Number of restart classifications observed.
    pub restart_count: u32,
}

impl Default for InstanceLifecycle {
    fn default() -> Self {
        Self::new(0)
    }
}

impl InstanceLifecycle {
    /// Create a new lifecycle snapshot in `started` state.
    #[must_use]
    pub fn new(ts_ms: u64) -> Self {
        Self {
            state: LifecycleState::Started,
            severity: LifecycleSeverity::Info,
            reason_code: "lifecycle.started".to_owned(),
            last_transition_ms: ts_ms,
            last_heartbeat_ms: ts_ms,
            restart_count: 0,
        }
    }

    /// Apply a deterministic lifecycle signal transition.
    pub fn apply_signal(
        &mut self,
        signal: LifecycleSignal,
        ts_ms: u64,
        reason_code: Option<String>,
    ) -> LifecycleTransition {
        let from = self.state;
        let to;
        let mut reason = reason_code.unwrap_or_else(|| match signal {
            LifecycleSignal::Start => "lifecycle.started".to_owned(),
            LifecycleSignal::Heartbeat => "lifecycle.heartbeat".to_owned(),
            LifecycleSignal::Degraded => "lifecycle.degraded".to_owned(),
            LifecycleSignal::Recovering => "lifecycle.recovering".to_owned(),
            LifecycleSignal::Stop => "lifecycle.stopped".to_owned(),
        });

        match signal {
            LifecycleSignal::Start => {
                let restarting = matches!(from, LifecycleState::Stopped | LifecycleState::Stale);
                if restarting {
                    self.restart_count = self.restart_count.saturating_add(1);
                    to = LifecycleState::Recovering;
                    self.severity = LifecycleSeverity::Warn;
                    reason.clear();
                    reason.push_str("lifecycle.restart");
                } else {
                    to = LifecycleState::Started;
                    self.severity = LifecycleSeverity::Info;
                }
                self.last_heartbeat_ms = ts_ms;
            }
            LifecycleSignal::Heartbeat => {
                self.last_heartbeat_ms = ts_ms;
                to = LifecycleState::Healthy;
                self.severity = LifecycleSeverity::Info;
            }
            LifecycleSignal::Degraded => {
                to = LifecycleState::Degraded;
                self.severity = LifecycleSeverity::Warn;
            }
            LifecycleSignal::Recovering => {
                to = LifecycleState::Recovering;
                self.severity = LifecycleSeverity::Warn;
            }
            LifecycleSignal::Stop => {
                to = LifecycleState::Stopped;
                self.severity = LifecycleSeverity::Info;
            }
        }

        let changed = to != from;
        self.state = to;
        self.reason_code.clone_from(&reason);
        if changed {
            self.last_transition_ms = ts_ms;
        }

        LifecycleTransition {
            from,
            to,
            reason_code: reason,
            changed,
        }
    }

    /// Mark the instance stale if heartbeat gap exceeds the timeout.
    pub fn mark_stale_if_heartbeat_gap(
        &mut self,
        now_ms: u64,
        heartbeat_timeout_ms: u64,
    ) -> Option<LifecycleTransition> {
        if heartbeat_timeout_ms == 0
            || matches!(self.state, LifecycleState::Stopped | LifecycleState::Stale)
        {
            return None;
        }

        let deadline = self.last_heartbeat_ms.saturating_add(heartbeat_timeout_ms);
        if now_ms < deadline {
            return None;
        }

        let from = self.state;
        self.state = LifecycleState::Stale;
        self.severity = LifecycleSeverity::Warn;
        self.reason_code.clear();
        self.reason_code.push_str("lifecycle.heartbeat_gap");
        self.last_transition_ms = now_ms;

        Some(LifecycleTransition {
            from,
            to: LifecycleState::Stale,
            reason_code: self.reason_code.clone(),
            changed: true,
        })
    }
}

/// Configuration for deterministic lifecycle tracking over discovery snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleTrackerConfig {
    /// Heartbeat gap threshold that transitions healthy/recovering instances to stale.
    pub stale_after_ms: u64,
    /// Absence threshold that transitions stale/healthy instances to stopped.
    pub stop_after_ms: u64,
    /// Maximum number of retained lifecycle events for timeline/alerts.
    pub max_retained_events: usize,
}

impl Default for LifecycleTrackerConfig {
    fn default() -> Self {
        Self {
            stale_after_ms: 30_000,
            stop_after_ms: 120_000,
            max_retained_events: 4_096,
        }
    }
}

impl LifecycleTrackerConfig {
    #[must_use]
    pub const fn normalized(self) -> Self {
        let stop_after_ms = if self.stop_after_ms < self.stale_after_ms {
            self.stale_after_ms
        } else {
            self.stop_after_ms
        };
        let max_retained_events = if self.max_retained_events == 0 {
            1
        } else {
            self.max_retained_events
        };
        Self {
            stale_after_ms: self.stale_after_ms,
            stop_after_ms,
            max_retained_events,
        }
    }
}

/// Lifecycle transition event suitable for timeline and alert surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleEvent {
    /// Instance identifier this event belongs to.
    pub instance_id: String,
    /// Previous lifecycle state.
    pub from: LifecycleState,
    /// Resulting lifecycle state.
    pub to: LifecycleState,
    /// Reason code associated with the transition.
    pub reason_code: String,
    /// Transition timestamp (unix ms).
    pub at_ms: u64,
    /// Attribution confidence attached to the instance at transition time.
    pub attribution_confidence_score: u8,
    /// Attribution collision status attached at transition time.
    pub attribution_collision: bool,
}

/// Deterministic attribution + lifecycle tracker for discovery snapshots.
#[derive(Debug, Clone)]
pub struct ProjectLifecycleTracker {
    config: LifecycleTrackerConfig,
    resolver: ProjectAttributionResolver,
    attributions: HashMap<String, InstanceAttribution>,
    lifecycles: HashMap<String, InstanceLifecycle>,
    event_log: Vec<LifecycleEvent>,
}

impl Default for ProjectLifecycleTracker {
    fn default() -> Self {
        Self::new(LifecycleTrackerConfig::default())
    }
}

impl ProjectLifecycleTracker {
    #[must_use]
    pub fn new(config: LifecycleTrackerConfig) -> Self {
        Self {
            config: config.normalized(),
            resolver: ProjectAttributionResolver,
            attributions: HashMap::new(),
            lifecycles: HashMap::new(),
            event_log: Vec::new(),
        }
    }

    /// Ingest one discovery snapshot and return transitions emitted in this update.
    #[allow(clippy::too_many_lines)]
    pub fn ingest_discovery(
        &mut self,
        now_ms: u64,
        instances: &[DiscoveredInstance],
    ) -> Vec<LifecycleEvent> {
        let mut events: Vec<LifecycleEvent> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for instance in instances {
            let instance_id = instance.instance_id.clone();
            seen.insert(instance_id.clone());

            let attribution = self.resolver.resolve(
                instance.project_key_hint.as_deref(),
                instance.host_name.as_deref(),
                None,
            );
            self.attributions
                .insert(instance_id.clone(), attribution.clone());

            let lifecycle = self
                .lifecycles
                .entry(instance_id.clone())
                .or_insert_with(|| InstanceLifecycle::new(instance.first_seen_ms));

            if lifecycle.last_transition_ms == instance.first_seen_ms
                && lifecycle.state == LifecycleState::Started
            {
                events.push(Self::event_from_transition(
                    &instance_id,
                    LifecycleTransition {
                        from: LifecycleState::Stopped,
                        to: LifecycleState::Started,
                        reason_code: "lifecycle.discovery.start".to_owned(),
                        changed: true,
                    },
                    instance.first_seen_ms,
                    &attribution,
                ));
            }

            if instance.status == DiscoveryStatus::Active
                && matches!(
                    lifecycle.state,
                    LifecycleState::Stopped | LifecycleState::Stale
                )
            {
                let restart = lifecycle.apply_signal(
                    LifecycleSignal::Start,
                    instance.last_seen_ms,
                    Some("lifecycle.discovery.start".to_owned()),
                );
                if restart.changed {
                    events.push(Self::event_from_transition(
                        &instance_id,
                        restart,
                        instance.last_seen_ms,
                        &attribution,
                    ));
                }
            }

            let heartbeat = lifecycle.apply_signal(
                LifecycleSignal::Heartbeat,
                instance.last_seen_ms,
                Some("lifecycle.discovery.heartbeat".to_owned()),
            );
            if heartbeat.changed {
                events.push(Self::event_from_transition(
                    &instance_id,
                    heartbeat,
                    instance.last_seen_ms,
                    &attribution,
                ));
            }

            if instance.status == DiscoveryStatus::Stale {
                let stale_at_ms = instance
                    .last_seen_ms
                    .saturating_add(self.config.stale_after_ms);
                if let Some(stale) =
                    lifecycle.mark_stale_if_heartbeat_gap(stale_at_ms, self.config.stale_after_ms)
                {
                    events.push(Self::event_from_transition(
                        &instance_id,
                        stale,
                        stale_at_ms,
                        &attribution,
                    ));
                }
            }
        }

        for (instance_id, lifecycle) in &mut self.lifecycles {
            if seen.contains(instance_id) {
                continue;
            }

            let attribution = self
                .attributions
                .get(instance_id)
                .cloned()
                .unwrap_or_else(|| {
                    InstanceAttribution::unknown(None, None, "attribution.lifecycle_missing")
                });

            let stop_deadline = lifecycle
                .last_heartbeat_ms
                .saturating_add(self.config.stop_after_ms);
            if now_ms >= stop_deadline {
                let stop = lifecycle.apply_signal(
                    LifecycleSignal::Stop,
                    now_ms,
                    Some("lifecycle.discovery.stop".to_owned()),
                );
                if stop.changed {
                    events.push(Self::event_from_transition(
                        instance_id,
                        stop,
                        now_ms,
                        &attribution,
                    ));
                }
                continue;
            }

            if let Some(stale) =
                lifecycle.mark_stale_if_heartbeat_gap(now_ms, self.config.stale_after_ms)
            {
                events.push(Self::event_from_transition(
                    instance_id,
                    stale,
                    now_ms,
                    &attribution,
                ));
            }
        }

        self.event_log.extend(events.iter().cloned());
        if self.event_log.len() > self.config.max_retained_events {
            let excess = self
                .event_log
                .len()
                .saturating_sub(self.config.max_retained_events);
            self.event_log.drain(0..excess);
        }

        events
    }

    #[must_use]
    pub fn attribution_snapshot(&self) -> HashMap<String, InstanceAttribution> {
        self.attributions.clone()
    }

    #[must_use]
    pub fn lifecycle_snapshot(&self) -> HashMap<String, InstanceLifecycle> {
        self.lifecycles.clone()
    }

    #[must_use]
    pub fn event_log(&self) -> &[LifecycleEvent] {
        &self.event_log
    }

    #[must_use]
    pub fn attribution_for(&self, instance_id: &str) -> Option<&InstanceAttribution> {
        self.attributions.get(instance_id)
    }

    #[must_use]
    pub fn lifecycle_for(&self, instance_id: &str) -> Option<&InstanceLifecycle> {
        self.lifecycles.get(instance_id)
    }

    fn event_from_transition(
        instance_id: &str,
        transition: LifecycleTransition,
        at_ms: u64,
        attribution: &InstanceAttribution,
    ) -> LifecycleEvent {
        LifecycleEvent {
            instance_id: instance_id.to_owned(),
            from: transition.from,
            to: transition.to,
            reason_code: transition.reason_code,
            at_ms,
            attribution_confidence_score: attribution.confidence_score,
            attribution_collision: attribution.collision,
        }
    }
}

// ─── Metrics Snapshot ────────────────────────────────────────────────────────

/// Resource metrics snapshot for an instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU usage percentage (0.0 - 100.0).
    pub cpu_percent: f64,
    /// Memory usage in bytes.
    pub memory_bytes: u64,
    /// Disk I/O bytes read since last snapshot.
    pub io_read_bytes: u64,
    /// Disk I/O bytes written since last snapshot.
    pub io_write_bytes: u64,
}

/// Search performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Total searches in the current window.
    pub total_searches: u64,
    /// Average search latency in microseconds.
    pub avg_latency_us: u64,
    /// P95 search latency in microseconds.
    pub p95_latency_us: u64,
    /// Number of searches that used refinement.
    pub refined_count: u64,
}

// ─── Control-Plane Health ───────────────────────────────────────────────────

/// Severity level for control-plane health.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlPlaneHealth {
    /// All monitored signals are within healthy thresholds.
    Healthy,
    /// One or more signals are degraded but not critical.
    Degraded,
    /// One or more signals are at a critical threshold.
    Critical,
}

impl ControlPlaneHealth {
    /// Short status badge for status bar chrome.
    #[must_use]
    pub const fn badge(self) -> &'static str {
        match self {
            Self::Healthy => "CP:OK",
            Self::Degraded => "CP:WARN",
            Self::Critical => "CP:CRIT",
        }
    }
}

impl fmt::Display for ControlPlaneHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Internal self-monitoring metrics for the ops control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlaneMetrics {
    /// Events currently backlogged in ingestion.
    pub ingestion_lag_events: u64,
    /// Approximate persisted bytes used by control-plane storage.
    pub storage_bytes: u64,
    /// Storage soft limit for warnings.
    pub storage_limit_bytes: u64,
    /// Average frame time budget in milliseconds.
    pub frame_time_ms: f64,
    /// Discovery scan latency in milliseconds.
    pub discovery_latency_ms: u64,
    /// Event processing throughput (events/sec).
    pub event_throughput_eps: f64,
    /// Resident set size of control-plane process.
    pub rss_bytes: u64,
    /// RSS soft limit for warnings.
    pub rss_limit_bytes: u64,
    /// Dead-letter events pending triage.
    pub dead_letter_events: u64,
}

impl Default for ControlPlaneMetrics {
    fn default() -> Self {
        Self {
            ingestion_lag_events: 0,
            storage_bytes: 0,
            storage_limit_bytes: 1,
            frame_time_ms: 16.0,
            discovery_latency_ms: 0,
            event_throughput_eps: 0.0,
            rss_bytes: 0,
            rss_limit_bytes: 1,
            dead_letter_events: 0,
        }
    }
}

impl ControlPlaneMetrics {
    const LAG_WARN_EVENTS: u64 = 1_000;
    const LAG_CRIT_EVENTS: u64 = 10_000;
    const DISCOVERY_WARN_MS: u64 = 2_000;
    const DISCOVERY_CRIT_MS: u64 = 5_000;
    const DEAD_LETTER_WARN: u64 = 1;
    const DEAD_LETTER_CRIT: u64 = 20;
    const STORAGE_WARN_RATIO: f64 = 0.80;
    const STORAGE_CRIT_RATIO: f64 = 0.95;
    const RSS_WARN_RATIO: f64 = 0.80;
    const RSS_CRIT_RATIO: f64 = 0.95;
    const FPS_WARN: f64 = 30.0;
    const FPS_CRIT: f64 = 15.0;

    fn ratio_as_f64(numer: u64, denom: u64) -> f64 {
        if denom == 0 {
            return 0.0;
        }
        let scaled = numer.saturating_mul(10_000).saturating_div(denom);
        let scaled_u32 = u32::try_from(scaled).unwrap_or(u32::MAX);
        f64::from(scaled_u32) / 10_000.0
    }

    /// Storage utilization ratio in `[0.0, +inf)`.
    #[must_use]
    pub fn storage_utilization(&self) -> f64 {
        Self::ratio_as_f64(self.storage_bytes, self.storage_limit_bytes)
    }

    /// RSS utilization ratio in `[0.0, +inf)`.
    #[must_use]
    pub fn rss_utilization(&self) -> f64 {
        Self::ratio_as_f64(self.rss_bytes, self.rss_limit_bytes)
    }

    /// Approximate renderer frame rate.
    #[must_use]
    pub fn estimated_fps(&self) -> f64 {
        if !self.frame_time_ms.is_finite() || self.frame_time_ms <= 0.0 {
            0.0
        } else {
            1000.0 / self.frame_time_ms
        }
    }

    /// Compute aggregate control-plane health.
    #[must_use]
    pub fn health(&self) -> ControlPlaneHealth {
        let storage_ratio = self.storage_utilization();
        let rss_ratio = self.rss_utilization();
        let fps = self.estimated_fps();
        let lag = self.ingestion_lag_events;
        let dead = self.dead_letter_events;
        let discovery = self.discovery_latency_ms;
        // Non-finite throughput is treated as zero (worst-case) so that
        // lag-gated rules fire correctly when telemetry produces NaN/Inf.
        let throughput = if self.event_throughput_eps.is_finite() {
            self.event_throughput_eps
        } else {
            0.0
        };

        if lag >= Self::LAG_CRIT_EVENTS
            || storage_ratio >= Self::STORAGE_CRIT_RATIO
            || rss_ratio >= Self::RSS_CRIT_RATIO
            || fps <= Self::FPS_CRIT
            || discovery >= Self::DISCOVERY_CRIT_MS
            || dead >= Self::DEAD_LETTER_CRIT
            || (lag >= 5_000 && throughput < 0.5)
        {
            return ControlPlaneHealth::Critical;
        }

        if lag >= Self::LAG_WARN_EVENTS
            || storage_ratio >= Self::STORAGE_WARN_RATIO
            || rss_ratio >= Self::RSS_WARN_RATIO
            || fps <= Self::FPS_WARN
            || discovery >= Self::DISCOVERY_WARN_MS
            || dead >= Self::DEAD_LETTER_WARN
            || (lag > 0 && throughput < 1.0)
        {
            return ControlPlaneHealth::Degraded;
        }

        ControlPlaneHealth::Healthy
    }

    /// Deterministic multi-line report for diagnostics and operator overlays.
    #[must_use]
    pub fn self_check_report(&self) -> String {
        format!(
            "health: {}\ningestion_lag_events: {}\nstorage_utilization: {:.1}% ({}/{})\nframe_rate_fps: {:.1}\ndiscovery_latency_ms: {}\nevent_throughput_eps: {:.2}\nrss_utilization: {:.1}% ({}/{})\ndead_letter_events: {}",
            self.health(),
            self.ingestion_lag_events,
            self.storage_utilization() * 100.0,
            self.storage_bytes,
            self.storage_limit_bytes,
            self.estimated_fps(),
            self.discovery_latency_ms,
            self.event_throughput_eps,
            self.rss_utilization() * 100.0,
            self.rss_bytes,
            self.rss_limit_bytes,
            self.dead_letter_events
        )
    }
}

// ─── Fleet Snapshot ──────────────────────────────────────────────────────────

/// Complete fleet snapshot for rendering.
#[derive(Debug, Clone, Default)]
pub struct FleetSnapshot {
    /// All discovered instances.
    pub instances: Vec<InstanceInfo>,
    /// Per-instance resource metrics (keyed by instance ID).
    pub resources: HashMap<String, ResourceMetrics>,
    /// Per-instance search metrics (keyed by instance ID).
    pub search_metrics: HashMap<String, SearchMetrics>,
    /// Per-instance project attribution metadata (keyed by instance ID).
    pub attribution: HashMap<String, InstanceAttribution>,
    /// Per-instance lifecycle state snapshots (keyed by instance ID).
    pub lifecycle: HashMap<String, InstanceLifecycle>,
    /// Recent lifecycle transition events for timeline/alerts.
    pub lifecycle_events: Vec<LifecycleEvent>,
}

impl FleetSnapshot {
    /// Number of discovered instances.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Number of healthy instances.
    #[must_use]
    pub fn healthy_count(&self) -> usize {
        self.instances.iter().filter(|i| i.healthy).count()
    }

    /// Total documents across all instances.
    #[must_use]
    pub fn total_docs(&self) -> u64 {
        self.instances.iter().map(|i| i.doc_count).sum()
    }

    /// Total pending jobs across all instances.
    #[must_use]
    pub fn total_pending_jobs(&self) -> u64 {
        self.instances.iter().map(|i| i.pending_jobs).sum()
    }

    /// Number of instances currently marked stale.
    #[must_use]
    pub fn stale_count(&self) -> usize {
        self.lifecycle
            .values()
            .filter(|lifecycle| lifecycle.state == LifecycleState::Stale)
            .count()
    }

    /// Attribution metadata for an instance id.
    #[must_use]
    pub fn attribution_for(&self, instance_id: &str) -> Option<&InstanceAttribution> {
        self.attribution.get(instance_id)
    }

    /// Lifecycle snapshot for an instance id.
    #[must_use]
    pub fn lifecycle_for(&self, instance_id: &str) -> Option<&InstanceLifecycle> {
        self.lifecycle.get(instance_id)
    }

    /// Recent lifecycle transition events.
    #[must_use]
    pub fn lifecycle_events(&self) -> &[LifecycleEvent] {
        &self.lifecycle_events
    }
}

// ─── App State ───────────────────────────────────────────────────────────────

/// Shared application state read by the render loop.
///
/// Background async tasks update this via `update_fleet()`.
/// The render loop reads it via `fleet()`.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Latest fleet snapshot.
    fleet: FleetSnapshot,
    /// When the fleet was last updated.
    last_update: Option<Instant>,
    /// Connection status message.
    connection_status: String,
    /// Internal self-monitoring metrics for this control plane.
    control_plane: ControlPlaneMetrics,
    /// Aggregate health derived from `control_plane`.
    control_plane_health: ControlPlaneHealth,
}

impl AppState {
    /// Create a new empty app state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fleet: FleetSnapshot::default(),
            last_update: None,
            connection_status: "Discovering instances...".to_string(),
            control_plane: ControlPlaneMetrics::default(),
            control_plane_health: ControlPlaneHealth::Healthy,
        }
    }

    /// Update the fleet snapshot.
    pub fn update_fleet(&mut self, snapshot: FleetSnapshot) {
        self.fleet = snapshot;
        self.last_update = Some(Instant::now());
        self.refresh_connection_status();
    }

    /// Update internal control-plane metrics.
    pub fn update_control_plane(&mut self, metrics: ControlPlaneMetrics) {
        self.control_plane_health = metrics.health();
        self.control_plane = metrics;
    }

    fn refresh_connection_status(&mut self) {
        let count = self.fleet.instance_count();
        let healthy = self.fleet.healthy_count();
        let stale = self.fleet.stale_count();
        self.connection_status = format!("{count} instances, {healthy} healthy, {stale} stale");
    }

    /// Get the current fleet snapshot.
    #[must_use]
    pub const fn fleet(&self) -> &FleetSnapshot {
        &self.fleet
    }

    /// Get the connection status string.
    #[must_use]
    pub fn connection_status(&self) -> &str {
        &self.connection_status
    }

    /// Get control-plane self-monitoring metrics.
    #[must_use]
    pub const fn control_plane_metrics(&self) -> &ControlPlaneMetrics {
        &self.control_plane
    }

    /// Get current aggregate control-plane health.
    #[must_use]
    pub const fn control_plane_health(&self) -> ControlPlaneHealth {
        self.control_plane_health
    }

    /// Produce a deterministic self-check report for overlays/logging.
    #[must_use]
    pub fn self_check_report(&self) -> String {
        self.control_plane.self_check_report()
    }

    /// When the fleet was last updated.
    #[must_use]
    pub const fn last_update(&self) -> Option<Instant> {
        self.last_update
    }

    /// Whether we have received at least one fleet update.
    #[must_use]
    pub const fn has_data(&self) -> bool {
        self.last_update.is_some()
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> FleetSnapshot {
        let resolver = ProjectAttributionResolver;
        let mut attribution = HashMap::new();
        attribution.insert(
            "inst-1".to_string(),
            resolver.resolve(
                Some("cass"),
                Some("cass-devbox"),
                Some("coding_agent_session_search"),
            ),
        );
        attribution.insert(
            "inst-2".to_string(),
            resolver.resolve(Some("xf"), Some("xf-node-02"), Some("xf")),
        );

        let mut lifecycle = HashMap::new();
        let mut lifecycle_1 = InstanceLifecycle::new(1_000);
        lifecycle_1.apply_signal(LifecycleSignal::Heartbeat, 1_250, None);
        lifecycle.insert("inst-1".to_string(), lifecycle_1);

        let mut lifecycle_2 = InstanceLifecycle::new(1_000);
        lifecycle_2.apply_signal(
            LifecycleSignal::Degraded,
            1_400,
            Some("health.timeout".to_string()),
        );
        lifecycle.insert("inst-2".to_string(), lifecycle_2);

        FleetSnapshot {
            instances: vec![
                InstanceInfo {
                    id: "inst-1".to_string(),
                    project: "cass".to_string(),
                    pid: Some(1234),
                    healthy: true,
                    doc_count: 5000,
                    pending_jobs: 10,
                },
                InstanceInfo {
                    id: "inst-2".to_string(),
                    project: "xf".to_string(),
                    pid: Some(5678),
                    healthy: false,
                    doc_count: 3000,
                    pending_jobs: 200,
                },
            ],
            resources: HashMap::new(),
            search_metrics: HashMap::new(),
            attribution,
            lifecycle,
            lifecycle_events: Vec::new(),
        }
    }

    #[test]
    fn app_state_initial() {
        let state = AppState::new();
        assert!(!state.has_data());
        assert_eq!(state.fleet().instance_count(), 0);
    }

    #[test]
    fn app_state_update_fleet() {
        let mut state = AppState::new();
        state.update_fleet(sample_snapshot());
        assert!(state.has_data());
        assert_eq!(state.fleet().instance_count(), 2);
        assert_eq!(state.fleet().healthy_count(), 1);
    }

    #[test]
    fn fleet_snapshot_aggregates() {
        let snap = sample_snapshot();
        assert_eq!(snap.total_docs(), 8000);
        assert_eq!(snap.total_pending_jobs(), 210);
        assert_eq!(snap.stale_count(), 0);
    }

    #[test]
    fn connection_status_updates() {
        let mut state = AppState::new();
        assert!(state.connection_status().contains("Discovering"));
        state.update_fleet(sample_snapshot());
        assert!(state.connection_status().contains("2 instances"));
        assert!(state.connection_status().contains("1 healthy"));
    }

    #[test]
    fn control_plane_health_transitions() {
        let mut state = AppState::new();
        let healthy = ControlPlaneMetrics {
            event_throughput_eps: 10.0,
            ..ControlPlaneMetrics::default()
        };
        state.update_control_plane(healthy.clone());
        assert_eq!(state.control_plane_health(), ControlPlaneHealth::Healthy);

        let degraded = ControlPlaneMetrics {
            ingestion_lag_events: 2_500,
            ..healthy
        };
        state.update_control_plane(degraded.clone());
        assert_eq!(state.control_plane_health(), ControlPlaneHealth::Degraded);

        let critical = ControlPlaneMetrics {
            ingestion_lag_events: 12_000,
            ..degraded
        };
        state.update_control_plane(critical);
        assert_eq!(state.control_plane_health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn self_check_report_contains_core_fields() {
        let mut state = AppState::new();
        state.update_control_plane(ControlPlaneMetrics {
            ingestion_lag_events: 42,
            storage_bytes: 800,
            storage_limit_bytes: 1000,
            frame_time_ms: 20.0,
            discovery_latency_ms: 25,
            event_throughput_eps: 12.5,
            rss_bytes: 256,
            rss_limit_bytes: 1024,
            dead_letter_events: 0,
        });

        let report = state.self_check_report();
        assert!(report.contains("ingestion_lag_events: 42"));
        assert!(report.contains("storage_utilization"));
        assert!(report.contains("frame_rate_fps"));
        assert!(report.contains("dead_letter_events: 0"));
    }

    #[test]
    fn instance_info_serde_roundtrip() {
        let info = InstanceInfo {
            id: "test".to_string(),
            project: "proj".to_string(),
            pid: Some(42),
            healthy: true,
            doc_count: 100,
            pending_jobs: 5,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: InstanceInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, info.id);
        assert_eq!(decoded.doc_count, info.doc_count);
    }

    #[test]
    fn attribution_resolver_maps_known_and_unknown_projects() {
        let resolver = ProjectAttributionResolver;
        let known = resolver.resolve(Some("agent-mail"), Some("mail-host"), None);
        assert_eq!(known.resolved_project, "mcp_agent_mail_rust");
        assert!(known.confidence_score >= 80);
        assert!(!known.collision);
        assert!(
            known
                .evidence_trace
                .iter()
                .any(|entry| entry.starts_with("reason=attribution."))
        );

        let unknown = resolver.resolve(Some("custom-app"), Some("mystery-box"), None);
        assert_eq!(unknown.resolved_project, "unknown");
        assert_eq!(unknown.confidence_score, 20);
        assert_eq!(unknown.reason_code, "attribution.unknown");
        assert!(
            unknown
                .evidence_trace
                .iter()
                .any(|entry| entry == "resolved_project=unknown")
        );
    }

    #[test]
    fn attribution_resolver_marks_conflicting_hints() {
        let resolver = ProjectAttributionResolver;
        let result = resolver.resolve(
            Some("xf"),
            Some("cass-host"),
            Some("coding-agent-session-search"),
        );
        assert!(result.collision);
        assert_eq!(result.reason_code, "attribution.collision");
        assert!(
            result
                .evidence_trace
                .iter()
                .any(|entry| entry == "collision=true")
        );
    }

    #[test]
    fn unknown_attribution_records_hint_trace() {
        let unknown =
            InstanceAttribution::unknown(Some(" custom-app "), Some(" mystery-host "), "manual");
        assert_eq!(unknown.project_key_hint.as_deref(), Some("custom-app"));
        assert_eq!(unknown.host_name_hint.as_deref(), Some("mystery-host"));
        assert!(
            unknown
                .evidence_trace
                .iter()
                .any(|entry| entry == "project_key_hint=custom-app")
        );
        assert!(
            unknown
                .evidence_trace
                .iter()
                .any(|entry| entry == "host_name_hint=mystery-host")
        );
        assert!(
            unknown
                .evidence_trace
                .iter()
                .any(|entry| entry == "reason=manual")
        );
    }

    #[test]
    fn lifecycle_transitions_are_deterministic() {
        let mut lifecycle = InstanceLifecycle::new(10);

        let transition = lifecycle.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        assert_eq!(transition.from, LifecycleState::Started);
        assert_eq!(transition.to, LifecycleState::Healthy);

        lifecycle.apply_signal(LifecycleSignal::Stop, 30, None);
        let restart = lifecycle.apply_signal(LifecycleSignal::Start, 40, None);
        assert_eq!(restart.to, LifecycleState::Recovering);
        assert_eq!(lifecycle.restart_count, 1);

        let stale = lifecycle.mark_stale_if_heartbeat_gap(10_000, 5_000);
        assert!(stale.is_some());
        assert_eq!(lifecycle.state, LifecycleState::Stale);
    }

    #[test]
    fn lifecycle_heartbeat_is_idempotent_when_already_healthy() {
        let mut lifecycle = InstanceLifecycle::new(10);
        lifecycle.apply_signal(LifecycleSignal::Heartbeat, 20, None);

        let transition = lifecycle.apply_signal(LifecycleSignal::Heartbeat, 25, None);
        assert_eq!(transition.from, LifecycleState::Healthy);
        assert_eq!(transition.to, LifecycleState::Healthy);
        assert!(!transition.changed);
        assert_eq!(lifecycle.last_transition_ms, 20);
        assert_eq!(lifecycle.last_heartbeat_ms, 25);
    }

    #[test]
    fn lifecycle_stale_gap_respects_stopped_and_zero_timeout_guards() {
        let mut stopped = InstanceLifecycle::new(10);
        stopped.apply_signal(LifecycleSignal::Stop, 15, None);
        assert!(stopped.mark_stale_if_heartbeat_gap(1_000, 5_000).is_none());
        assert_eq!(stopped.state, LifecycleState::Stopped);

        let mut healthy = InstanceLifecycle::new(10);
        healthy.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        assert!(healthy.mark_stale_if_heartbeat_gap(1_000, 0).is_none());
        assert_eq!(healthy.state, LifecycleState::Healthy);
    }

    #[test]
    fn lifecycle_tracker_config_normalization_enforces_bounds() {
        let normalized = LifecycleTrackerConfig {
            stale_after_ms: 50,
            stop_after_ms: 10,
            max_retained_events: 0,
        }
        .normalized();
        assert_eq!(normalized.stale_after_ms, 50);
        assert_eq!(normalized.stop_after_ms, 50);
        assert_eq!(normalized.max_retained_events, 1);
    }

    fn discovery_instance(
        id: &str,
        project_key_hint: Option<&str>,
        host_name: Option<&str>,
        last_seen_ms: u64,
        status: DiscoveryStatus,
    ) -> DiscoveredInstance {
        DiscoveredInstance {
            instance_id: id.to_owned(),
            project_key_hint: project_key_hint.map(str::to_owned),
            host_name: host_name.map(str::to_owned),
            pid: Some(111),
            version: Some("0.1.0".to_owned()),
            first_seen_ms: last_seen_ms.saturating_sub(10),
            last_seen_ms,
            status,
            sources: vec![crate::discovery::DiscoverySignalKind::Heartbeat],
            identity_keys: vec![format!(
                "hostpid:{}:111",
                host_name.unwrap_or("unknown-host")
            )],
        }
    }

    #[test]
    fn lifecycle_tracker_emits_start_stale_stop_and_restart_transitions() {
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig {
            stale_after_ms: 10,
            stop_after_ms: 20,
            max_retained_events: 128,
        });

        let first = vec![discovery_instance(
            "inst-a",
            Some("cass"),
            Some("cass-host"),
            100,
            DiscoveryStatus::Active,
        )];
        let events_first = tracker.ingest_discovery(100, &first);
        assert!(
            events_first
                .iter()
                .any(|event| event.to == LifecycleState::Started)
        );
        assert!(
            events_first
                .iter()
                .any(|event| event.to == LifecycleState::Healthy)
        );
        assert_eq!(
            tracker
                .lifecycle_for("inst-a")
                .expect("lifecycle present")
                .state,
            LifecycleState::Healthy
        );

        let events_stale = tracker.ingest_discovery(112, &[]);
        assert!(
            events_stale
                .iter()
                .any(|event| event.to == LifecycleState::Stale)
        );
        assert_eq!(
            tracker
                .lifecycle_for("inst-a")
                .expect("lifecycle present")
                .state,
            LifecycleState::Stale
        );

        let events_stop = tracker.ingest_discovery(125, &[]);
        assert!(
            events_stop
                .iter()
                .any(|event| event.to == LifecycleState::Stopped)
        );
        assert_eq!(
            tracker
                .lifecycle_for("inst-a")
                .expect("lifecycle present")
                .state,
            LifecycleState::Stopped
        );

        let restart = vec![discovery_instance(
            "inst-a",
            Some("cass"),
            Some("cass-host"),
            130,
            DiscoveryStatus::Active,
        )];
        let events_restart = tracker.ingest_discovery(130, &restart);
        assert!(
            events_restart
                .iter()
                .any(|event| event.to == LifecycleState::Recovering)
        );
        assert!(
            events_restart
                .iter()
                .any(|event| event.to == LifecycleState::Healthy)
        );
        assert_eq!(
            tracker
                .lifecycle_for("inst-a")
                .expect("lifecycle present")
                .restart_count,
            1
        );
        assert!(
            events_restart
                .iter()
                .all(|event| event.reason_code.starts_with("lifecycle."))
        );
    }

    #[test]
    fn lifecycle_tracker_surfaces_attribution_collision_metadata() {
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig::default());
        let collision = vec![discovery_instance(
            "inst-collision",
            Some("xf"),
            Some("cass-devbox"),
            50,
            DiscoveryStatus::Active,
        )];

        let events = tracker.ingest_discovery(50, &collision);
        let attribution = tracker
            .attribution_for("inst-collision")
            .expect("attribution should exist");
        assert!(attribution.collision);
        assert_eq!(attribution.reason_code, "attribution.collision");
        assert!(
            events
                .iter()
                .all(|event| event.attribution_confidence_score > 0)
        );
        assert!(events.iter().all(|event| event.attribution_collision));
    }

    #[test]
    fn lifecycle_tracker_respects_event_retention_limit() {
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig {
            stale_after_ms: 1,
            stop_after_ms: 2,
            max_retained_events: 2,
        });
        let first = vec![discovery_instance(
            "inst-retain",
            Some("cass"),
            Some("cass-host"),
            10,
            DiscoveryStatus::Active,
        )];
        let _ = tracker.ingest_discovery(10, &first);
        let _ = tracker.ingest_discovery(12, &[]);

        assert_eq!(tracker.event_log().len(), 2);
    }

    #[test]
    fn lifecycle_tracker_uses_unknown_attribution_when_cache_missing() {
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig {
            stale_after_ms: 5,
            stop_after_ms: 10,
            max_retained_events: 16,
        });
        let first = vec![discovery_instance(
            "inst-missing-attr",
            Some("cass"),
            Some("cass-host"),
            10,
            DiscoveryStatus::Active,
        )];
        let _ = tracker.ingest_discovery(10, &first);

        tracker.attributions.clear();
        let events = tracker.ingest_discovery(25, &[]);
        let stop = events
            .iter()
            .find(|event| event.to == LifecycleState::Stopped)
            .expect("stop transition should be emitted");

        assert_eq!(stop.reason_code, "lifecycle.discovery.stop");
        assert_eq!(stop.attribution_confidence_score, 20);
        assert!(!stop.attribution_collision);
    }

    // ─── bd-244k tests begin ───

    #[test]
    fn normalize_hint_none() {
        assert_eq!(normalize_hint(None), None);
    }

    #[test]
    fn normalize_hint_empty() {
        assert_eq!(normalize_hint(Some("")), None);
    }

    #[test]
    fn normalize_hint_whitespace_only() {
        assert_eq!(normalize_hint(Some("   ")), None);
    }

    #[test]
    fn normalize_hint_trims() {
        assert_eq!(normalize_hint(Some("  value  ")), Some("value".to_owned()));
    }

    #[test]
    fn normalize_hint_preserves_clean() {
        assert_eq!(normalize_hint(Some("val")), Some("val".to_owned()));
    }

    #[test]
    fn control_plane_health_badge_healthy() {
        assert_eq!(ControlPlaneHealth::Healthy.badge(), "CP:OK");
    }

    #[test]
    fn control_plane_health_badge_degraded() {
        assert_eq!(ControlPlaneHealth::Degraded.badge(), "CP:WARN");
    }

    #[test]
    fn control_plane_health_badge_critical() {
        assert_eq!(ControlPlaneHealth::Critical.badge(), "CP:CRIT");
    }

    #[test]
    fn control_plane_health_display_healthy() {
        assert_eq!(format!("{}", ControlPlaneHealth::Healthy), "healthy");
    }

    #[test]
    fn control_plane_health_display_degraded() {
        assert_eq!(format!("{}", ControlPlaneHealth::Degraded), "degraded");
    }

    #[test]
    fn control_plane_health_display_critical() {
        assert_eq!(format!("{}", ControlPlaneHealth::Critical), "critical");
    }

    #[test]
    fn control_plane_health_serde_roundtrip() {
        for variant in [
            ControlPlaneHealth::Healthy,
            ControlPlaneHealth::Degraded,
            ControlPlaneHealth::Critical,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: ControlPlaneHealth = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, variant);
        }
        // snake_case rename_all
        let json = serde_json::to_string(&ControlPlaneHealth::Healthy).unwrap();
        assert_eq!(json, "\"healthy\"");
    }

    #[test]
    fn control_plane_metrics_default_field_values() {
        let m = ControlPlaneMetrics::default();
        assert_eq!(m.ingestion_lag_events, 0);
        assert_eq!(m.storage_bytes, 0);
        assert_eq!(m.storage_limit_bytes, 1);
        assert!((m.frame_time_ms - 16.0).abs() < f64::EPSILON);
        assert_eq!(m.discovery_latency_ms, 0);
        assert!((m.event_throughput_eps - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.rss_bytes, 0);
        assert_eq!(m.rss_limit_bytes, 1);
        assert_eq!(m.dead_letter_events, 0);
    }

    #[test]
    fn ratio_as_f64_zero_denom() {
        assert!((ControlPlaneMetrics::ratio_as_f64(100, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ratio_as_f64_normal() {
        let ratio = ControlPlaneMetrics::ratio_as_f64(500, 1000);
        assert!((ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn ratio_as_f64_full() {
        let ratio = ControlPlaneMetrics::ratio_as_f64(1000, 1000);
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn ratio_as_f64_over_100_percent() {
        let ratio = ControlPlaneMetrics::ratio_as_f64(2000, 1000);
        assert!((ratio - 2.0).abs() < 0.001);
    }

    #[test]
    fn storage_utilization_direct() {
        let m = ControlPlaneMetrics {
            storage_bytes: 800,
            storage_limit_bytes: 1000,
            ..ControlPlaneMetrics::default()
        };
        assert!((m.storage_utilization() - 0.8).abs() < 0.001);
    }

    #[test]
    fn rss_utilization_direct() {
        let m = ControlPlaneMetrics {
            rss_bytes: 950,
            rss_limit_bytes: 1000,
            ..ControlPlaneMetrics::default()
        };
        assert!((m.rss_utilization() - 0.95).abs() < 0.001);
    }

    #[test]
    fn estimated_fps_normal() {
        let m = ControlPlaneMetrics {
            frame_time_ms: 16.0,
            ..ControlPlaneMetrics::default()
        };
        assert!((m.estimated_fps() - 62.5).abs() < 0.1);
    }

    #[test]
    fn estimated_fps_zero_frame_time() {
        let m = ControlPlaneMetrics {
            frame_time_ms: 0.0,
            ..ControlPlaneMetrics::default()
        };
        assert!((m.estimated_fps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimated_fps_negative_frame_time() {
        let m = ControlPlaneMetrics {
            frame_time_ms: -5.0,
            ..ControlPlaneMetrics::default()
        };
        assert!((m.estimated_fps() - 0.0).abs() < f64::EPSILON);
    }

    fn healthy_metrics() -> ControlPlaneMetrics {
        ControlPlaneMetrics {
            ingestion_lag_events: 0,
            storage_bytes: 0,
            storage_limit_bytes: 1000,
            frame_time_ms: 10.0, // 100 fps
            discovery_latency_ms: 0,
            event_throughput_eps: 100.0,
            rss_bytes: 0,
            rss_limit_bytes: 1000,
            dead_letter_events: 0,
        }
    }

    #[test]
    fn health_all_healthy() {
        assert_eq!(healthy_metrics().health(), ControlPlaneHealth::Healthy);
    }

    #[test]
    fn health_lag_warn_boundary() {
        let mut m = healthy_metrics();
        m.ingestion_lag_events = 999;
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
        m.ingestion_lag_events = 1_000;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_lag_crit_boundary() {
        let mut m = healthy_metrics();
        m.ingestion_lag_events = 9_999;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
        m.ingestion_lag_events = 10_000;
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_storage_warn_boundary() {
        let mut m = healthy_metrics();
        m.storage_bytes = 799;
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
        m.storage_bytes = 800; // 0.80 ratio
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_storage_crit_boundary() {
        let mut m = healthy_metrics();
        m.storage_bytes = 949;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
        m.storage_bytes = 950; // 0.95 ratio
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_rss_warn_boundary() {
        let mut m = healthy_metrics();
        m.rss_bytes = 799;
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
        m.rss_bytes = 800;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_rss_crit_boundary() {
        let mut m = healthy_metrics();
        m.rss_bytes = 949;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
        m.rss_bytes = 950;
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_fps_warn_boundary() {
        let mut m = healthy_metrics();
        // fps = 1000 / frame_time_ms
        // fps=30.0 → frame_time=33.33 → Degraded boundary
        m.frame_time_ms = 33.0; // fps ~30.3 > 30 → Healthy
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
        m.frame_time_ms = 34.0; // fps ~29.4 <= 30 → Degraded
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_fps_crit_boundary() {
        let mut m = healthy_metrics();
        m.frame_time_ms = 66.0; // fps ~15.15 > 15 → Degraded
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
        m.frame_time_ms = 67.0; // fps ~14.9 <= 15 → Critical
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_discovery_warn_boundary() {
        let mut m = healthy_metrics();
        m.discovery_latency_ms = 1_999;
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
        m.discovery_latency_ms = 2_000;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_discovery_crit_boundary() {
        let mut m = healthy_metrics();
        m.discovery_latency_ms = 4_999;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
        m.discovery_latency_ms = 5_000;
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_dead_letter_warn_boundary() {
        let mut m = healthy_metrics();
        m.dead_letter_events = 0;
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
        m.dead_letter_events = 1;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_dead_letter_crit_boundary() {
        let mut m = healthy_metrics();
        m.dead_letter_events = 19;
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
        m.dead_letter_events = 20;
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_combined_lag_throughput_crit() {
        let mut m = healthy_metrics();
        m.ingestion_lag_events = 5_000;
        m.event_throughput_eps = 0.4; // lag>=5000 && eps<0.5 → Critical
        assert_eq!(m.health(), ControlPlaneHealth::Critical);
    }

    #[test]
    fn health_combined_lag_throughput_degraded() {
        let mut m = healthy_metrics();
        m.ingestion_lag_events = 1; // lag>0
        m.event_throughput_eps = 0.9; // eps<1.0 → Degraded
        assert_eq!(m.health(), ControlPlaneHealth::Degraded);
    }

    #[test]
    fn health_combined_lag_throughput_not_degraded_when_throughput_ok() {
        let mut m = healthy_metrics();
        m.ingestion_lag_events = 1;
        m.event_throughput_eps = 1.0; // not < 1.0 → stays Healthy
        assert_eq!(m.health(), ControlPlaneHealth::Healthy);
    }

    #[test]
    fn self_check_report_format_verification() {
        let m = ControlPlaneMetrics {
            ingestion_lag_events: 0,
            storage_bytes: 500,
            storage_limit_bytes: 1000,
            frame_time_ms: 16.0,
            discovery_latency_ms: 100,
            event_throughput_eps: 50.0,
            rss_bytes: 200,
            rss_limit_bytes: 1000,
            dead_letter_events: 3,
        };
        let report = m.self_check_report();
        assert!(report.starts_with("health: "));
        assert!(report.contains("ingestion_lag_events: 0"));
        assert!(report.contains("500/1000"));
        assert!(report.contains("200/1000"));
        assert!(report.contains("dead_letter_events: 3"));
        assert!(report.contains("discovery_latency_ms: 100"));
        assert!(report.contains("event_throughput_eps: 50.00"));
    }

    #[test]
    fn instance_lifecycle_default() {
        let lc = InstanceLifecycle::default();
        assert_eq!(lc.state, LifecycleState::Started);
        assert_eq!(lc.severity, LifecycleSeverity::Info);
        assert_eq!(lc.reason_code, "lifecycle.started");
        assert_eq!(lc.last_transition_ms, 0);
        assert_eq!(lc.last_heartbeat_ms, 0);
        assert_eq!(lc.restart_count, 0);
    }

    #[test]
    fn instance_lifecycle_new_preserves_timestamp() {
        let lc = InstanceLifecycle::new(42);
        assert_eq!(lc.last_transition_ms, 42);
        assert_eq!(lc.last_heartbeat_ms, 42);
    }

    #[test]
    fn apply_signal_start_from_healthy_stays_started() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        assert_eq!(lc.state, LifecycleState::Healthy);

        // Start from Healthy → stays Started (not restarting)
        let t = lc.apply_signal(LifecycleSignal::Start, 30, None);
        assert_eq!(t.from, LifecycleState::Healthy);
        assert_eq!(t.to, LifecycleState::Started);
        assert!(t.changed);
        assert_eq!(lc.severity, LifecycleSeverity::Info);
        assert_eq!(lc.restart_count, 0);
    }

    #[test]
    fn apply_signal_start_from_degraded_stays_started() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Degraded, 20, None);
        assert_eq!(lc.state, LifecycleState::Degraded);

        let t = lc.apply_signal(LifecycleSignal::Start, 30, None);
        assert_eq!(t.from, LifecycleState::Degraded);
        assert_eq!(t.to, LifecycleState::Started);
        assert!(t.changed);
        assert_eq!(lc.severity, LifecycleSeverity::Info);
        assert_eq!(lc.restart_count, 0);
    }

    #[test]
    fn apply_signal_start_from_stale_is_restart() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        lc.mark_stale_if_heartbeat_gap(10_000, 5_000);
        assert_eq!(lc.state, LifecycleState::Stale);

        let t = lc.apply_signal(LifecycleSignal::Start, 10_001, None);
        assert_eq!(t.to, LifecycleState::Recovering);
        assert_eq!(lc.restart_count, 1);
        assert_eq!(lc.severity, LifecycleSeverity::Warn);
        assert_eq!(lc.reason_code, "lifecycle.restart");
    }

    #[test]
    fn apply_signal_custom_reason_code() {
        let mut lc = InstanceLifecycle::new(10);
        let t = lc.apply_signal(
            LifecycleSignal::Degraded,
            20,
            Some("custom.reason".to_owned()),
        );
        assert_eq!(t.reason_code, "custom.reason");
        assert_eq!(lc.reason_code, "custom.reason");
    }

    #[test]
    fn apply_signal_degraded_severity() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Degraded, 20, None);
        assert_eq!(lc.severity, LifecycleSeverity::Warn);
        assert_eq!(lc.state, LifecycleState::Degraded);
    }

    #[test]
    fn apply_signal_recovering_severity() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Recovering, 20, None);
        assert_eq!(lc.severity, LifecycleSeverity::Warn);
        assert_eq!(lc.state, LifecycleState::Recovering);
    }

    #[test]
    fn apply_signal_stop_severity() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        lc.apply_signal(LifecycleSignal::Stop, 30, None);
        assert_eq!(lc.severity, LifecycleSeverity::Info);
        assert_eq!(lc.state, LifecycleState::Stopped);
    }

    #[test]
    fn apply_signal_unchanged_does_not_update_transition_ts() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        assert_eq!(lc.last_transition_ms, 20);
        // second heartbeat: Healthy→Healthy, changed=false
        let t = lc.apply_signal(LifecycleSignal::Heartbeat, 30, None);
        assert!(!t.changed);
        assert_eq!(lc.last_transition_ms, 20); // unchanged
        assert_eq!(lc.last_heartbeat_ms, 30); // updated
    }

    #[test]
    fn mark_stale_already_stale_returns_none() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        lc.mark_stale_if_heartbeat_gap(10_000, 5_000);
        assert_eq!(lc.state, LifecycleState::Stale);
        // already stale → None
        assert!(lc.mark_stale_if_heartbeat_gap(20_000, 5_000).is_none());
    }

    #[test]
    fn mark_stale_within_deadline_returns_none() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 100, None);
        // deadline = 100 + 5000 = 5100; now_ms=5099 < 5100 → None
        assert!(lc.mark_stale_if_heartbeat_gap(5099, 5000).is_none());
        assert_eq!(lc.state, LifecycleState::Healthy);
    }

    #[test]
    fn mark_stale_at_deadline_returns_some() {
        let mut lc = InstanceLifecycle::new(10);
        lc.apply_signal(LifecycleSignal::Heartbeat, 100, None);
        // deadline = 100 + 5000 = 5100; now_ms=5100 >= 5100 → Some
        let t = lc.mark_stale_if_heartbeat_gap(5100, 5000);
        assert!(t.is_some());
        let t = t.unwrap();
        assert_eq!(t.from, LifecycleState::Healthy);
        assert_eq!(t.to, LifecycleState::Stale);
        assert!(t.changed);
    }

    #[test]
    fn lifecycle_signal_serde_roundtrip() {
        for signal in [
            LifecycleSignal::Start,
            LifecycleSignal::Heartbeat,
            LifecycleSignal::Degraded,
            LifecycleSignal::Recovering,
            LifecycleSignal::Stop,
        ] {
            let json = serde_json::to_string(&signal).unwrap();
            let decoded: LifecycleSignal = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, signal);
        }
        // snake_case
        assert_eq!(
            serde_json::to_string(&LifecycleSignal::Heartbeat).unwrap(),
            "\"heartbeat\""
        );
    }

    #[test]
    fn lifecycle_tracker_config_default_values() {
        let c = LifecycleTrackerConfig::default();
        assert_eq!(c.stale_after_ms, 30_000);
        assert_eq!(c.stop_after_ms, 120_000);
        assert_eq!(c.max_retained_events, 4_096);
    }

    #[test]
    fn lifecycle_tracker_config_normalized_no_clamping() {
        let c = LifecycleTrackerConfig {
            stale_after_ms: 100,
            stop_after_ms: 200,
            max_retained_events: 50,
        }
        .normalized();
        assert_eq!(c.stale_after_ms, 100);
        assert_eq!(c.stop_after_ms, 200);
        assert_eq!(c.max_retained_events, 50);
    }

    #[test]
    fn fleet_snapshot_default_empty() {
        let snap = FleetSnapshot::default();
        assert_eq!(snap.instance_count(), 0);
        assert_eq!(snap.healthy_count(), 0);
        assert_eq!(snap.total_docs(), 0);
        assert_eq!(snap.total_pending_jobs(), 0);
        assert_eq!(snap.stale_count(), 0);
        assert!(snap.lifecycle_events().is_empty());
    }

    #[test]
    fn fleet_snapshot_stale_count_with_stale() {
        let mut lifecycle = HashMap::new();
        let mut stale = InstanceLifecycle::new(10);
        stale.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        stale.mark_stale_if_heartbeat_gap(10_000, 5_000);
        lifecycle.insert("stale-1".to_string(), stale);

        let healthy = InstanceLifecycle::new(10);
        lifecycle.insert("healthy-1".to_string(), healthy);

        let snap = FleetSnapshot {
            lifecycle,
            ..FleetSnapshot::default()
        };
        assert_eq!(snap.stale_count(), 1);
    }

    #[test]
    fn fleet_snapshot_attribution_for_hit_miss() {
        let mut attribution = HashMap::new();
        attribution.insert(
            "inst-1".to_string(),
            InstanceAttribution::unknown(Some("proj"), None, "test"),
        );
        let snap = FleetSnapshot {
            attribution,
            ..FleetSnapshot::default()
        };
        assert!(snap.attribution_for("inst-1").is_some());
        assert!(snap.attribution_for("nonexistent").is_none());
    }

    #[test]
    fn fleet_snapshot_lifecycle_for_hit_miss() {
        let mut lifecycle = HashMap::new();
        lifecycle.insert("inst-1".to_string(), InstanceLifecycle::new(0));
        let snap = FleetSnapshot {
            lifecycle,
            ..FleetSnapshot::default()
        };
        assert!(snap.lifecycle_for("inst-1").is_some());
        assert!(snap.lifecycle_for("nonexistent").is_none());
    }

    #[test]
    fn fleet_snapshot_lifecycle_events_accessor() {
        let events = vec![LifecycleEvent {
            instance_id: "i1".to_string(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "test".to_string(),
            at_ms: 100,
            attribution_confidence_score: 80,
            attribution_collision: false,
        }];
        let snap = FleetSnapshot {
            lifecycle_events: events,
            ..FleetSnapshot::default()
        };
        assert_eq!(snap.lifecycle_events().len(), 1);
        assert_eq!(snap.lifecycle_events()[0].instance_id, "i1");
    }

    #[test]
    fn app_state_default_matches_new() {
        let d = AppState::default();
        let n = AppState::new();
        assert!(!d.has_data());
        assert!(!n.has_data());
        assert_eq!(d.fleet().instance_count(), n.fleet().instance_count());
        assert_eq!(d.connection_status(), n.connection_status());
        assert_eq!(d.control_plane_health(), n.control_plane_health());
    }

    #[test]
    fn app_state_last_update_none_then_some() {
        let mut state = AppState::new();
        assert!(state.last_update().is_none());
        state.update_fleet(FleetSnapshot::default());
        assert!(state.last_update().is_some());
    }

    #[test]
    fn app_state_control_plane_metrics_accessor() {
        let mut state = AppState::new();
        let m = ControlPlaneMetrics {
            ingestion_lag_events: 42,
            ..ControlPlaneMetrics::default()
        };
        state.update_control_plane(m);
        assert_eq!(state.control_plane_metrics().ingestion_lag_events, 42);
    }

    #[test]
    fn instance_attribution_unknown_with_none_hints() {
        let attr = InstanceAttribution::unknown(None, None, "no_hints");
        assert!(attr.project_key_hint.is_none());
        assert!(attr.host_name_hint.is_none());
        assert_eq!(attr.resolved_project, "unknown");
        assert_eq!(attr.confidence_score, 20);
        assert!(!attr.collision);
        // evidence trace should NOT contain project_key_hint or host_name_hint entries
        assert!(
            !attr
                .evidence_trace
                .iter()
                .any(|e| e.starts_with("project_key_hint="))
        );
        assert!(
            !attr
                .evidence_trace
                .iter()
                .any(|e| e.starts_with("host_name_hint="))
        );
        assert!(attr.evidence_trace.iter().any(|e| e == "reason=no_hints"));
        assert!(
            attr.evidence_trace
                .iter()
                .any(|e| e == "resolved_project=unknown")
        );
    }

    #[test]
    fn instance_attribution_unknown_empty_hints_normalized() {
        let attr = InstanceAttribution::unknown(Some("  "), Some(""), "whitespace");
        assert!(attr.project_key_hint.is_none());
        assert!(attr.host_name_hint.is_none());
    }

    #[test]
    fn instance_attribution_serde_roundtrip() {
        let attr = InstanceAttribution::unknown(Some("proj"), Some("host"), "serde_test");
        let json = serde_json::to_string(&attr).unwrap();
        let decoded: InstanceAttribution = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, attr);
    }

    #[test]
    fn lifecycle_event_serde_roundtrip() {
        let event = LifecycleEvent {
            instance_id: "i-42".to_string(),
            from: LifecycleState::Started,
            to: LifecycleState::Healthy,
            reason_code: "test.heartbeat".to_string(),
            at_ms: 1000,
            attribution_confidence_score: 90,
            attribution_collision: false,
        };
        let json = serde_json::to_string(&event).unwrap();
        let decoded: LifecycleEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn project_lifecycle_tracker_default() {
        let tracker = ProjectLifecycleTracker::default();
        assert!(tracker.event_log().is_empty());
        assert!(tracker.attribution_snapshot().is_empty());
        assert!(tracker.lifecycle_snapshot().is_empty());
    }

    #[test]
    fn project_lifecycle_tracker_accessors_after_ingest() {
        let mut tracker = ProjectLifecycleTracker::new(LifecycleTrackerConfig::default());
        let instances = vec![discovery_instance(
            "inst-acc",
            Some("xf"),
            Some("xf-host"),
            100,
            DiscoveryStatus::Active,
        )];
        tracker.ingest_discovery(100, &instances);
        assert_eq!(tracker.attribution_snapshot().len(), 1);
        assert!(tracker.attribution_for("inst-acc").is_some());
        assert!(tracker.lifecycle_for("inst-acc").is_some());
        assert_eq!(tracker.lifecycle_snapshot().len(), 1);
    }

    #[test]
    fn lifecycle_restart_count_saturates() {
        let mut lc = InstanceLifecycle::new(10);
        lc.restart_count = u32::MAX;
        lc.apply_signal(LifecycleSignal::Stop, 20, None);
        let t = lc.apply_signal(LifecycleSignal::Start, 30, None);
        assert_eq!(t.to, LifecycleState::Recovering);
        assert_eq!(lc.restart_count, u32::MAX); // saturating_add
    }

    #[test]
    fn lifecycle_transition_changed_flag_same_state() {
        let mut lc = InstanceLifecycle::new(10);
        // Started → Started (Start from non-stopped/stale)
        let t = lc.apply_signal(LifecycleSignal::Start, 20, None);
        assert_eq!(t.from, LifecycleState::Started);
        assert_eq!(t.to, LifecycleState::Started);
        assert!(!t.changed);
    }

    #[test]
    fn app_state_connection_status_reflects_stale() {
        let mut state = AppState::new();
        let mut lifecycle = HashMap::new();
        let mut stale_lc = InstanceLifecycle::new(10);
        stale_lc.apply_signal(LifecycleSignal::Heartbeat, 20, None);
        stale_lc.mark_stale_if_heartbeat_gap(10_000, 5_000);
        lifecycle.insert("inst-s".to_string(), stale_lc);

        let snap = FleetSnapshot {
            instances: vec![InstanceInfo {
                id: "inst-s".to_string(),
                project: "p".to_string(),
                pid: None,
                healthy: false,
                doc_count: 0,
                pending_jobs: 0,
            }],
            lifecycle,
            ..FleetSnapshot::default()
        };
        state.update_fleet(snap);
        assert!(state.connection_status().contains("1 stale"));
    }

    // ─── bd-244k tests end ───
}
