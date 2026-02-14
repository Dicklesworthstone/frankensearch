//! Multi-source instance discovery and deterministic reconciliation.
//!
//! The discovery engine merges weak signals (process scans, sockets, control
//! endpoints, heartbeat files) into stable instance identities. This keeps
//! fleet views deterministic even when different probes report the same
//! runtime from different surfaces.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use serde::{Deserialize, Serialize};

/// Discovery source category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoverySignalKind {
    /// Process inspection signal.
    Process,
    /// Domain socket or named pipe signal.
    Socket,
    /// HTTP control endpoint signal.
    ControlEndpoint,
    /// Heartbeat file or registration signal.
    Heartbeat,
}

impl DiscoverySignalKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Process => "process",
            Self::Socket => "socket",
            Self::ControlEndpoint => "control_endpoint",
            Self::Heartbeat => "heartbeat",
        }
    }
}

/// One raw sighting from a discovery source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceSighting {
    /// Source that emitted this sighting.
    pub source: DiscoverySignalKind,
    /// Observation time (unix epoch milliseconds).
    pub observed_at_ms: u64,
    /// Optional project hint (for attribution).
    pub project_key_hint: Option<String>,
    /// Optional host hint.
    pub host_name: Option<String>,
    /// Optional process id.
    pub pid: Option<u32>,
    /// Optional stable identity hint from the source.
    pub instance_key_hint: Option<String>,
    /// Optional control endpoint URL/address.
    pub control_endpoint: Option<String>,
    /// Optional socket path/address.
    pub socket_path: Option<String>,
    /// Optional heartbeat/registration file path.
    pub heartbeat_path: Option<String>,
    /// Optional runtime version hint.
    pub version: Option<String>,
}

impl InstanceSighting {
    /// Build identity keys used for reconciliation.
    ///
    /// Keys are ordered strongest -> weakest, and all keys participate in
    /// alias-bridging merges.
    #[must_use]
    pub fn identity_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();

        if let Some(instance_key) = normalized(self.instance_key_hint.as_deref()) {
            keys.push(format!("instance:{instance_key}"));
        }

        if let (Some(host), Some(pid)) = (normalized(self.host_name.as_deref()), self.pid) {
            keys.push(format!("hostpid:{host}:{pid}"));
        }

        if let Some(endpoint) = normalized(self.control_endpoint.as_deref()) {
            keys.push(format!("endpoint:{endpoint}"));
        }

        if let Some(socket_path) = normalized(self.socket_path.as_deref()) {
            keys.push(format!("socket:{socket_path}"));
        }

        if let Some(heartbeat_path) = normalized(self.heartbeat_path.as_deref()) {
            keys.push(format!("heartbeat:{heartbeat_path}"));
        }

        if keys.is_empty() {
            keys.push(format!(
                "fallback:{}:{}",
                self.source.as_str(),
                self.observed_at_ms
            ));
        }

        keys
    }
}

/// Discovery lifecycle for a reconciled instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryStatus {
    /// Seen recently (inside stale window).
    Active,
    /// Not seen recently but retained for visibility.
    Stale,
}

/// Reconciled instance synthesized from one or more sightings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredInstance {
    /// Stable synthetic instance identifier.
    pub instance_id: String,
    /// Canonical project hint when available.
    pub project_key_hint: Option<String>,
    /// Canonical host hint when available.
    pub host_name: Option<String>,
    /// Canonical process id when available.
    pub pid: Option<u32>,
    /// Runtime version hint when available.
    pub version: Option<String>,
    /// First observed timestamp (unix ms).
    pub first_seen_ms: u64,
    /// Most recent observed timestamp (unix ms).
    pub last_seen_ms: u64,
    /// Current discovery lifecycle status.
    pub status: DiscoveryStatus,
    /// Source kinds that observed this instance.
    pub sources: Vec<DiscoverySignalKind>,
    /// Identity keys currently bound to this instance.
    pub identity_keys: Vec<String>,
}

impl DiscoveredInstance {
    #[must_use]
    pub fn healthy(&self) -> bool {
        self.status == DiscoveryStatus::Active
    }
}

/// Engine tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Age threshold (ms) after which unseen instances become stale.
    pub stale_after_ms: u64,
    /// Age threshold (ms) after which unseen instances are pruned.
    pub prune_after_ms: u64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            stale_after_ms: 30_000,
            prune_after_ms: 300_000,
        }
    }
}

impl DiscoveryConfig {
    #[must_use]
    pub const fn normalized(self) -> Self {
        if self.prune_after_ms < self.stale_after_ms {
            Self {
                stale_after_ms: self.stale_after_ms,
                prune_after_ms: self.stale_after_ms,
            }
        } else {
            self
        }
    }
}

/// Poll-cycle telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryStats {
    /// Number of sources polled.
    pub sources_polled: usize,
    /// Number of sightings read this poll.
    pub sightings_observed: usize,
    /// Number of merge events caused by alias-key collisions.
    pub duplicates_merged: usize,
    /// Number of stale instances after this poll.
    pub stale_instances: usize,
    /// Number of pruned instances in this poll.
    pub pruned_instances: usize,
}

/// Source adapter for discovery polling.
pub trait DiscoverySource: Send {
    /// Collect sightings from this source.
    fn collect(&mut self, now_ms: u64) -> Vec<InstanceSighting>;
}

/// Static source helper for deterministic tests.
#[derive(Debug, Clone)]
pub struct StaticDiscoverySource {
    sightings: Vec<InstanceSighting>,
}

impl StaticDiscoverySource {
    #[must_use]
    pub const fn new(sightings: Vec<InstanceSighting>) -> Self {
        Self { sightings }
    }
}

impl DiscoverySource for StaticDiscoverySource {
    fn collect(&mut self, _now_ms: u64) -> Vec<InstanceSighting> {
        self.sightings.clone()
    }
}

/// Deterministic multi-source discovery reconciler.
#[derive(Debug, Clone)]
pub struct DiscoveryEngine {
    config: DiscoveryConfig,
    instances: HashMap<String, DiscoveredInstance>,
    key_to_instance: HashMap<String, String>,
}

impl DiscoveryEngine {
    #[must_use]
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config: config.normalized(),
            instances: HashMap::new(),
            key_to_instance: HashMap::new(),
        }
    }

    /// Poll all configured sources and reconcile their sightings.
    #[allow(clippy::too_many_lines)]
    pub fn poll(
        &mut self,
        now_ms: u64,
        sources: &mut [&mut dyn DiscoverySource],
    ) -> DiscoveryStats {
        let mut stats = DiscoveryStats {
            sources_polled: sources.len(),
            ..DiscoveryStats::default()
        };

        for source in sources {
            let sightings = source.collect(now_ms);
            stats.sightings_observed = stats.sightings_observed.saturating_add(sightings.len());

            for mut sighting in sightings {
                if sighting.observed_at_ms == 0 {
                    sighting.observed_at_ms = now_ms;
                }

                let keys = sighting.identity_keys();
                let mut existing_ids: BTreeSet<String> = BTreeSet::new();
                for key in &keys {
                    if let Some(instance_id) = self.key_to_instance.get(key) {
                        existing_ids.insert(instance_id.clone());
                    }
                }

                let canonical_id = existing_ids
                    .iter()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| stable_instance_id(&keys[0]));

                if existing_ids.len() > 1 {
                    for stale_id in existing_ids.iter().skip(1) {
                        if self.merge_instances(stale_id, &canonical_id) {
                            stats.duplicates_merged = stats.duplicates_merged.saturating_add(1);
                        }
                    }
                }

                let instance = self
                    .instances
                    .entry(canonical_id.clone())
                    .or_insert_with(|| DiscoveredInstance {
                        instance_id: canonical_id.clone(),
                        project_key_hint: sighting.project_key_hint.clone(),
                        host_name: sighting.host_name.clone(),
                        pid: sighting.pid,
                        version: sighting.version.clone(),
                        first_seen_ms: sighting.observed_at_ms,
                        last_seen_ms: sighting.observed_at_ms,
                        status: DiscoveryStatus::Active,
                        sources: vec![sighting.source],
                        identity_keys: Vec::new(),
                    });

                let previous_last_seen = instance.last_seen_ms;
                instance.first_seen_ms = instance.first_seen_ms.min(sighting.observed_at_ms);
                instance.last_seen_ms = instance.last_seen_ms.max(sighting.observed_at_ms);
                instance.status = DiscoveryStatus::Active;

                if !instance.sources.contains(&sighting.source) {
                    instance.sources.push(sighting.source);
                    instance.sources.sort_unstable();
                }

                merge_hint(
                    &mut instance.project_key_hint,
                    sighting.project_key_hint.as_deref(),
                );
                merge_hint(&mut instance.host_name, sighting.host_name.as_deref());
                refresh_pid_hint(
                    &mut instance.pid,
                    sighting.pid,
                    sighting.observed_at_ms,
                    previous_last_seen,
                );
                refresh_version_hint(
                    &mut instance.version,
                    sighting.version.as_deref(),
                    sighting.observed_at_ms,
                    previous_last_seen,
                );

                for key in keys {
                    if !instance.identity_keys.contains(&key) {
                        instance.identity_keys.push(key.clone());
                    }
                    self.key_to_instance.insert(key, canonical_id.clone());
                }
                instance.identity_keys.sort();
            }
        }

        let mut pruned: Vec<String> = Vec::new();
        for (id, instance) in &mut self.instances {
            let age_ms = now_ms.saturating_sub(instance.last_seen_ms);
            if age_ms >= self.config.prune_after_ms {
                pruned.push(id.clone());
                continue;
            }
            instance.status = if age_ms >= self.config.stale_after_ms {
                DiscoveryStatus::Stale
            } else {
                DiscoveryStatus::Active
            };
        }

        for id in pruned {
            self.instances.remove(&id);
            self.key_to_instance.retain(|_, mapped| mapped != &id);
            stats.pruned_instances = stats.pruned_instances.saturating_add(1);
        }

        stats.stale_instances = self
            .instances
            .values()
            .filter(|instance| instance.status == DiscoveryStatus::Stale)
            .count();
        stats
    }

    /// Snapshot of discovered instances sorted by `instance_id`.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DiscoveredInstance> {
        let mut out: Vec<DiscoveredInstance> = self.instances.values().cloned().collect();
        out.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
        out
    }

    fn merge_instances(&mut self, from_id: &str, into_id: &str) -> bool {
        if from_id == into_id {
            return false;
        }

        let Some(mut from) = self.instances.remove(from_id) else {
            return false;
        };
        let Some(into) = self.instances.get_mut(into_id) else {
            self.instances.insert(from_id.to_owned(), from);
            return false;
        };

        let into_last_seen_before_merge = into.last_seen_ms;
        let from_is_newer_or_equal = from.last_seen_ms >= into_last_seen_before_merge;
        into.first_seen_ms = into.first_seen_ms.min(from.first_seen_ms);
        into.last_seen_ms = into.last_seen_ms.max(from.last_seen_ms);
        into.status = DiscoveryStatus::Active;

        merge_hint(&mut into.project_key_hint, from.project_key_hint.as_deref());
        merge_hint(&mut into.host_name, from.host_name.as_deref());
        if from_is_newer_or_equal {
            if from.pid.is_some() {
                into.pid = from.pid;
            }
            if from.version.is_some() {
                into.version.clone_from(&from.version);
            }
        } else {
            if into.pid.is_none() {
                into.pid = from.pid;
            }
            if into.version.is_none() {
                into.version.clone_from(&from.version);
            }
        }

        let mut sources: HashSet<DiscoverySignalKind> = into.sources.iter().copied().collect();
        for source in from.sources.drain(..) {
            sources.insert(source);
        }
        into.sources = sources.into_iter().collect();
        into.sources.sort_unstable();

        for key in from.identity_keys {
            if !into.identity_keys.contains(&key) {
                into.identity_keys.push(key.clone());
            }
            self.key_to_instance.insert(key, into_id.to_owned());
        }
        into.identity_keys.sort();
        true
    }
}

fn merge_hint(current: &mut Option<String>, candidate: Option<&str>) {
    if current.is_some() {
        return;
    }
    if let Some(candidate) = normalized(candidate) {
        *current = Some(candidate);
    }
}

const fn refresh_pid_hint(
    current: &mut Option<u32>,
    candidate: Option<u32>,
    observed_at_ms: u64,
    previous_last_seen: u64,
) {
    let Some(pid) = candidate else {
        return;
    };
    if current.is_none() || observed_at_ms >= previous_last_seen {
        *current = Some(pid);
    }
}

fn refresh_version_hint(
    current: &mut Option<String>,
    candidate: Option<&str>,
    observed_at_ms: u64,
    previous_last_seen: u64,
) {
    let Some(version) = normalized(candidate) else {
        return;
    };
    if current.is_none() || observed_at_ms >= previous_last_seen {
        *current = Some(version);
    }
}

fn normalized(raw: Option<&str>) -> Option<String> {
    raw.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_ascii_lowercase())
        }
    })
}

fn stable_instance_id(seed: &str) -> String {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    format!("inst-{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process_sighting(
        observed_at_ms: u64,
        project: &str,
        host: &str,
        pid: u32,
    ) -> InstanceSighting {
        InstanceSighting {
            source: DiscoverySignalKind::Process,
            observed_at_ms,
            project_key_hint: Some(project.to_owned()),
            host_name: Some(host.to_owned()),
            pid: Some(pid),
            instance_key_hint: None,
            control_endpoint: None,
            socket_path: None,
            heartbeat_path: None,
            version: Some("0.1.0".to_owned()),
        }
    }

    #[test]
    fn reconciles_cross_source_duplicates_via_host_pid() {
        let mut process =
            StaticDiscoverySource::new(vec![process_sighting(10, "cass", "host-a", 42)]);
        let mut endpoint = StaticDiscoverySource::new(vec![InstanceSighting {
            source: DiscoverySignalKind::ControlEndpoint,
            observed_at_ms: 11,
            project_key_hint: Some("cass".to_owned()),
            host_name: Some("host-a".to_owned()),
            pid: Some(42),
            instance_key_hint: None,
            control_endpoint: Some("http://127.0.0.1:9898/control".to_owned()),
            socket_path: None,
            heartbeat_path: None,
            version: None,
        }]);

        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let stats = engine.poll(20, &mut [&mut process, &mut endpoint]);

        let snapshot = engine.snapshot();
        assert_eq!(stats.sightings_observed, 2);
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].host_name.as_deref(), Some("host-a"));
        assert_eq!(snapshot[0].pid, Some(42));
        assert!(snapshot[0].sources.contains(&DiscoverySignalKind::Process));
        assert!(
            snapshot[0]
                .sources
                .contains(&DiscoverySignalKind::ControlEndpoint)
        );
        assert!(snapshot[0].healthy());
    }

    #[test]
    fn keeps_distinct_instances_when_identity_keys_do_not_overlap() {
        let mut process = StaticDiscoverySource::new(vec![
            process_sighting(10, "cass", "host-a", 42),
            process_sighting(11, "xf", "host-b", 84),
        ]);

        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let stats = engine.poll(20, &mut [&mut process]);
        let snapshot = engine.snapshot();

        assert_eq!(stats.sightings_observed, 2);
        assert_eq!(snapshot.len(), 2);
        assert_ne!(snapshot[0].instance_id, snapshot[1].instance_id);
    }

    #[test]
    fn alias_bridge_merges_preexisting_instances() {
        let mut first_pass_a = StaticDiscoverySource::new(vec![InstanceSighting {
            source: DiscoverySignalKind::ControlEndpoint,
            observed_at_ms: 10,
            project_key_hint: Some("agent-mail".to_owned()),
            host_name: Some("host-z".to_owned()),
            pid: None,
            instance_key_hint: None,
            control_endpoint: Some("http://host-z:7777/control".to_owned()),
            socket_path: None,
            heartbeat_path: None,
            version: None,
        }]);
        let mut first_pass_b =
            StaticDiscoverySource::new(vec![process_sighting(10, "agent-mail", "host-z", 7777)]);

        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let _ = engine.poll(20, &mut [&mut first_pass_a, &mut first_pass_b]);
        assert_eq!(engine.snapshot().len(), 2);

        let mut bridge = StaticDiscoverySource::new(vec![InstanceSighting {
            source: DiscoverySignalKind::Heartbeat,
            observed_at_ms: 30,
            project_key_hint: Some("agent-mail".to_owned()),
            host_name: Some("host-z".to_owned()),
            pid: Some(7777),
            instance_key_hint: None,
            control_endpoint: Some("http://host-z:7777/control".to_owned()),
            socket_path: None,
            heartbeat_path: Some("/tmp/agent-mail.heartbeat".to_owned()),
            version: None,
        }]);

        let stats = engine.poll(40, &mut [&mut bridge]);
        assert!(stats.duplicates_merged >= 1);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].pid, Some(7777));
        assert_eq!(snapshot[0].version.as_deref(), Some("0.1.0"));
    }

    #[test]
    fn refreshes_pid_and_version_on_restart_like_transition() {
        let mut initial = StaticDiscoverySource::new(vec![InstanceSighting {
            source: DiscoverySignalKind::ControlEndpoint,
            observed_at_ms: 10,
            project_key_hint: Some("cass".to_owned()),
            host_name: Some("host-a".to_owned()),
            pid: Some(111),
            instance_key_hint: None,
            control_endpoint: Some("http://host-a:8787/control".to_owned()),
            socket_path: None,
            heartbeat_path: None,
            version: Some("0.1.0".to_owned()),
        }]);
        let mut restarted = StaticDiscoverySource::new(vec![InstanceSighting {
            source: DiscoverySignalKind::ControlEndpoint,
            observed_at_ms: 40,
            project_key_hint: Some("cass".to_owned()),
            host_name: Some("host-a".to_owned()),
            pid: Some(222),
            instance_key_hint: None,
            control_endpoint: Some("http://host-a:8787/control".to_owned()),
            socket_path: None,
            heartbeat_path: None,
            version: Some("0.2.0".to_owned()),
        }]);

        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let _ = engine.poll(20, &mut [&mut initial]);
        assert_eq!(engine.snapshot().len(), 1);
        assert_eq!(engine.snapshot()[0].pid, Some(111));
        assert_eq!(engine.snapshot()[0].version.as_deref(), Some("0.1.0"));

        let _ = engine.poll(50, &mut [&mut restarted]);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].pid, Some(222));
        assert_eq!(snapshot[0].version.as_deref(), Some("0.2.0"));
    }

    #[test]
    fn older_sighting_does_not_regress_pid_and_version() {
        let mut mixed = StaticDiscoverySource::new(vec![
            InstanceSighting {
                source: DiscoverySignalKind::ControlEndpoint,
                observed_at_ms: 50,
                project_key_hint: Some("xf".to_owned()),
                host_name: Some("host-b".to_owned()),
                pid: Some(9100),
                instance_key_hint: None,
                control_endpoint: Some("http://host-b:9100/control".to_owned()),
                socket_path: None,
                heartbeat_path: None,
                version: Some("2.0.0".to_owned()),
            },
            InstanceSighting {
                source: DiscoverySignalKind::ControlEndpoint,
                observed_at_ms: 20,
                project_key_hint: Some("xf".to_owned()),
                host_name: Some("host-b".to_owned()),
                pid: Some(9001),
                instance_key_hint: None,
                control_endpoint: Some("http://host-b:9100/control".to_owned()),
                socket_path: None,
                heartbeat_path: None,
                version: Some("1.9.0".to_owned()),
            },
        ]);

        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let _ = engine.poll(60, &mut [&mut mixed]);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].pid, Some(9100));
        assert_eq!(snapshot[0].version.as_deref(), Some("2.0.0"));
    }

    #[test]
    fn alias_bridge_merge_prefers_newer_pid_and_version_hints() {
        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());

        engine.instances.insert(
            "inst-into".to_owned(),
            DiscoveredInstance {
                instance_id: "inst-into".to_owned(),
                project_key_hint: Some("agent-mail".to_owned()),
                host_name: Some("host-z".to_owned()),
                pid: Some(1111),
                version: Some("0.1.0".to_owned()),
                first_seen_ms: 10,
                last_seen_ms: 20,
                status: DiscoveryStatus::Active,
                sources: vec![DiscoverySignalKind::ControlEndpoint],
                identity_keys: vec!["endpoint:http://host-z:7777/control".to_owned()],
            },
        );
        engine.instances.insert(
            "inst-from".to_owned(),
            DiscoveredInstance {
                instance_id: "inst-from".to_owned(),
                project_key_hint: Some("agent-mail".to_owned()),
                host_name: Some("host-z".to_owned()),
                pid: Some(2222),
                version: Some("0.2.0".to_owned()),
                first_seen_ms: 15,
                last_seen_ms: 40,
                status: DiscoveryStatus::Active,
                sources: vec![DiscoverySignalKind::Process],
                identity_keys: vec!["hostpid:host-z:2222".to_owned()],
            },
        );

        assert!(engine.merge_instances("inst-from", "inst-into"));
        let merged = engine
            .instances
            .get("inst-into")
            .expect("merged instance should exist");

        assert_eq!(merged.pid, Some(2222));
        assert_eq!(merged.version.as_deref(), Some("0.2.0"));
        assert_eq!(merged.last_seen_ms, 40);
        assert!(!engine.instances.contains_key("inst-from"));
    }

    #[test]
    fn marks_stale_and_then_prunes() {
        let config = DiscoveryConfig {
            stale_after_ms: 100,
            prune_after_ms: 200,
        };
        let mut engine = DiscoveryEngine::new(config);
        let mut process =
            StaticDiscoverySource::new(vec![process_sighting(1, "cass", "host-a", 42)]);
        let _ = engine.poll(1, &mut [&mut process]);
        assert_eq!(engine.snapshot().len(), 1);
        assert_eq!(engine.snapshot()[0].status, DiscoveryStatus::Active);

        let mut none = StaticDiscoverySource::new(Vec::new());
        let stale = engine.poll(160, &mut [&mut none]);
        assert_eq!(stale.stale_instances, 1);
        assert_eq!(engine.snapshot().len(), 1);
        assert_eq!(engine.snapshot()[0].status, DiscoveryStatus::Stale);

        let pruned = engine.poll(260, &mut [&mut none]);
        assert_eq!(pruned.pruned_instances, 1);
        assert_eq!(engine.snapshot().len(), 0);
    }

    #[test]
    fn fallback_identity_is_stable_for_identical_sighting() {
        let sighting = InstanceSighting {
            source: DiscoverySignalKind::Socket,
            observed_at_ms: 123,
            project_key_hint: None,
            host_name: None,
            pid: None,
            instance_key_hint: None,
            control_endpoint: None,
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let keys_a = sighting.identity_keys();
        let keys_b = sighting.identity_keys();
        assert_eq!(keys_a, keys_b);
        assert_eq!(keys_a.len(), 1);
        assert!(keys_a[0].starts_with("fallback:"));
    }
}
