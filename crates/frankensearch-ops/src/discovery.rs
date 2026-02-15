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

        if let Some(instance_key) = normalized_folded(self.instance_key_hint.as_deref()) {
            keys.push(format!("instance:{instance_key}"));
        }

        if let (Some(host), Some(pid)) = (normalized_folded(self.host_name.as_deref()), self.pid) {
            keys.push(format!("hostpid:{host}:{pid}"));
        }

        if let Some(endpoint) = normalized_endpoint(self.control_endpoint.as_deref()) {
            keys.push(format!("endpoint:{endpoint}"));
        }

        if let Some(socket_path) = normalized_exact(self.socket_path.as_deref()) {
            keys.push(format!("socket:{socket_path}"));
        }

        if let Some(heartbeat_path) = normalized_exact(self.heartbeat_path.as_deref()) {
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
        // Enforce minimum 1ms to prevent zero-thresholds from causing
        // immediate stale/prune of all instances on every poll cycle.
        let stale = if self.stale_after_ms == 0 {
            1
        } else {
            self.stale_after_ms
        };
        let prune = if self.prune_after_ms < stale {
            stale
        } else if self.prune_after_ms == 0 {
            1
        } else {
            self.prune_after_ms
        };
        Self {
            stale_after_ms: stale,
            prune_after_ms: prune,
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

                refresh_lowercase_hint(
                    &mut instance.project_key_hint,
                    sighting.project_key_hint.as_deref(),
                    sighting.observed_at_ms >= previous_last_seen,
                );
                refresh_lowercase_hint(
                    &mut instance.host_name,
                    sighting.host_name.as_deref(),
                    sighting.observed_at_ms >= previous_last_seen,
                );
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

        refresh_lowercase_hint(
            &mut into.project_key_hint,
            from.project_key_hint.as_deref(),
            from_is_newer_or_equal,
        );
        refresh_lowercase_hint(
            &mut into.host_name,
            from.host_name.as_deref(),
            from_is_newer_or_equal,
        );
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

fn refresh_lowercase_hint(current: &mut Option<String>, candidate: Option<&str>, prefer: bool) {
    let Some(candidate) = normalized_folded(candidate) else {
        return;
    };
    if current.is_none() || prefer {
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
    let Some(version) = normalized_folded(candidate) else {
        return;
    };
    if current.is_none() || observed_at_ms >= previous_last_seen {
        *current = Some(version);
    }
}

fn normalized_folded(raw: Option<&str>) -> Option<String> {
    raw.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_ascii_lowercase())
        }
    })
}

fn normalized_exact(raw: Option<&str>) -> Option<String> {
    raw.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        }
    })
}

fn normalized_endpoint(raw: Option<&str>) -> Option<String> {
    let raw = normalized_exact(raw)?;
    let Some((scheme, rest)) = raw.split_once("://") else {
        return Some(raw);
    };

    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, suffix) = rest.split_at(authority_end);
    if authority.is_empty() {
        return Some(format!("{}://{suffix}", scheme.to_ascii_lowercase()));
    }

    let normalized_authority = normalize_endpoint_authority(authority);
    Some(format!(
        "{}://{}{}",
        scheme.to_ascii_lowercase(),
        normalized_authority,
        suffix
    ))
}

fn normalize_endpoint_authority(authority: &str) -> String {
    let (userinfo, host_port) = authority
        .rsplit_once('@')
        .map_or(("", authority), |(userinfo, host_port)| {
            (userinfo, host_port)
        });
    let normalized_host_port = normalize_host_port(host_port);
    if userinfo.is_empty() {
        normalized_host_port
    } else {
        format!("{userinfo}@{normalized_host_port}")
    }
}

fn normalize_host_port(host_port: &str) -> String {
    if let Some(rest) = host_port.strip_prefix('[')
        && let Some(closing) = rest.find(']')
    {
        let host = rest[..closing].to_ascii_lowercase();
        let suffix = &rest[closing + 1..];
        return format!("[{host}]{suffix}");
    }

    if let Some((host, port)) = host_port.rsplit_once(':')
        && !port.is_empty()
        && port.chars().all(|ch| ch.is_ascii_digit())
    {
        return format!("{}:{port}", host.to_ascii_lowercase());
    }

    host_port.to_ascii_lowercase()
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

    #[test]
    fn identity_keys_preserve_case_for_path_hints() {
        let upper = InstanceSighting {
            source: DiscoverySignalKind::Socket,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: None,
            pid: None,
            instance_key_hint: None,
            control_endpoint: None,
            socket_path: Some("/tmp/Agent.sock".to_owned()),
            heartbeat_path: Some("/tmp/Heartbeat.pid".to_owned()),
            version: None,
        };
        let lower = InstanceSighting {
            socket_path: Some("/tmp/agent.sock".to_owned()),
            heartbeat_path: Some("/tmp/heartbeat.pid".to_owned()),
            ..upper.clone()
        };

        let upper_keys = upper.identity_keys();
        let lower_keys = lower.identity_keys();

        assert!(upper_keys.contains(&"socket:/tmp/Agent.sock".to_owned()));
        assert!(upper_keys.contains(&"heartbeat:/tmp/Heartbeat.pid".to_owned()));
        assert!(lower_keys.contains(&"socket:/tmp/agent.sock".to_owned()));
        assert!(lower_keys.contains(&"heartbeat:/tmp/heartbeat.pid".to_owned()));
    }

    #[test]
    fn identity_keys_fold_control_endpoint_scheme_and_host_case() {
        let upper = InstanceSighting {
            source: DiscoverySignalKind::ControlEndpoint,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: None,
            pid: None,
            instance_key_hint: None,
            control_endpoint: Some("HTTP://LOCALHOST:9000/control".to_owned()),
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let lower = InstanceSighting {
            control_endpoint: Some("http://localhost:9000/control".to_owned()),
            ..upper.clone()
        };

        assert_eq!(upper.identity_keys(), lower.identity_keys());
    }

    #[test]
    fn identity_keys_preserve_control_endpoint_path_case() {
        let upper = InstanceSighting {
            source: DiscoverySignalKind::ControlEndpoint,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: None,
            pid: None,
            instance_key_hint: None,
            control_endpoint: Some("http://localhost:9000/Control".to_owned()),
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let lower = InstanceSighting {
            control_endpoint: Some("http://localhost:9000/control".to_owned()),
            ..upper.clone()
        };

        let upper_keys = upper.identity_keys();
        let lower_keys = lower.identity_keys();
        assert!(upper_keys.contains(&"endpoint:http://localhost:9000/Control".to_owned()));
        assert!(lower_keys.contains(&"endpoint:http://localhost:9000/control".to_owned()));
        assert_ne!(upper_keys, lower_keys);
    }

    #[test]
    fn newer_sighting_refreshes_project_and_host_hints() {
        let mut source = StaticDiscoverySource::new(vec![
            InstanceSighting {
                source: DiscoverySignalKind::Process,
                observed_at_ms: 10,
                project_key_hint: Some("Legacy-Project".to_owned()),
                host_name: Some("Legacy-Host".to_owned()),
                pid: None,
                instance_key_hint: Some("shared-instance".to_owned()),
                control_endpoint: None,
                socket_path: None,
                heartbeat_path: None,
                version: None,
            },
            InstanceSighting {
                source: DiscoverySignalKind::ControlEndpoint,
                observed_at_ms: 20,
                project_key_hint: Some("Fresh-Project".to_owned()),
                host_name: Some("Fresh-Host".to_owned()),
                pid: None,
                instance_key_hint: Some("shared-instance".to_owned()),
                control_endpoint: Some("http://localhost:9000/control".to_owned()),
                socket_path: None,
                heartbeat_path: None,
                version: None,
            },
        ]);

        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let _ = engine.poll(25, &mut [&mut source]);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(
            snapshot[0].project_key_hint.as_deref(),
            Some("fresh-project")
        );
        assert_eq!(snapshot[0].host_name.as_deref(), Some("fresh-host"));
    }

    #[test]
    fn zero_observed_timestamp_uses_poll_time_and_prunes_at_normalized_threshold() {
        let config = DiscoveryConfig {
            stale_after_ms: 50,
            prune_after_ms: 10,
        };
        let mut engine = DiscoveryEngine::new(config);
        let mut source = StaticDiscoverySource::new(vec![process_sighting(0, "cass", "host-a", 7)]);

        let first = engine.poll(100, &mut [&mut source]);
        assert_eq!(first.pruned_instances, 0);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].first_seen_ms, 100);
        assert_eq!(snapshot[0].last_seen_ms, 100);
        assert_eq!(snapshot[0].status, DiscoveryStatus::Active);

        let mut empty = StaticDiscoverySource::new(Vec::new());
        let prune = engine.poll(150, &mut [&mut empty]);
        assert_eq!(prune.pruned_instances, 1);
        assert_eq!(prune.stale_instances, 0);
        assert!(engine.snapshot().is_empty());
    }

    // ─── bd-3l9h tests begin ───

    #[test]
    fn signal_kind_as_str_all_variants() {
        assert_eq!(DiscoverySignalKind::Process.as_str(), "process");
        assert_eq!(DiscoverySignalKind::Socket.as_str(), "socket");
        assert_eq!(
            DiscoverySignalKind::ControlEndpoint.as_str(),
            "control_endpoint"
        );
        assert_eq!(DiscoverySignalKind::Heartbeat.as_str(), "heartbeat");
    }

    #[test]
    fn signal_kind_serde_roundtrip() {
        for kind in [
            DiscoverySignalKind::Process,
            DiscoverySignalKind::Socket,
            DiscoverySignalKind::ControlEndpoint,
            DiscoverySignalKind::Heartbeat,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let decoded: DiscoverySignalKind = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, kind);
        }
        // snake_case
        assert_eq!(
            serde_json::to_string(&DiscoverySignalKind::ControlEndpoint).unwrap(),
            "\"control_endpoint\""
        );
    }

    #[test]
    fn signal_kind_ord() {
        let mut kinds = [
            DiscoverySignalKind::Heartbeat,
            DiscoverySignalKind::Process,
            DiscoverySignalKind::Socket,
        ];
        kinds.sort();
        // PartialOrd/Ord derive follows declaration order
        assert_eq!(kinds[0], DiscoverySignalKind::Process);
    }

    #[test]
    fn discovery_status_serde_roundtrip() {
        for status in [DiscoveryStatus::Active, DiscoveryStatus::Stale] {
            let json = serde_json::to_string(&status).unwrap();
            let decoded: DiscoveryStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, status);
        }
        assert_eq!(
            serde_json::to_string(&DiscoveryStatus::Active).unwrap(),
            "\"active\""
        );
    }

    #[test]
    fn discovered_instance_healthy_active() {
        let inst = DiscoveredInstance {
            instance_id: "i1".to_owned(),
            project_key_hint: None,
            host_name: None,
            pid: None,
            version: None,
            first_seen_ms: 0,
            last_seen_ms: 0,
            status: DiscoveryStatus::Active,
            sources: vec![],
            identity_keys: vec![],
        };
        assert!(inst.healthy());
    }

    #[test]
    fn discovered_instance_healthy_stale() {
        let inst = DiscoveredInstance {
            instance_id: "i2".to_owned(),
            project_key_hint: None,
            host_name: None,
            pid: None,
            version: None,
            first_seen_ms: 0,
            last_seen_ms: 0,
            status: DiscoveryStatus::Stale,
            sources: vec![],
            identity_keys: vec![],
        };
        assert!(!inst.healthy());
    }

    #[test]
    fn discovery_config_default_values() {
        let c = DiscoveryConfig::default();
        assert_eq!(c.stale_after_ms, 30_000);
        assert_eq!(c.prune_after_ms, 300_000);
    }

    #[test]
    fn discovery_config_normalized_valid() {
        let c = DiscoveryConfig {
            stale_after_ms: 100,
            prune_after_ms: 200,
        }
        .normalized();
        assert_eq!(c.stale_after_ms, 100);
        assert_eq!(c.prune_after_ms, 200);
    }

    #[test]
    fn discovery_config_normalized_clamps() {
        let c = DiscoveryConfig {
            stale_after_ms: 500,
            prune_after_ms: 100,
        }
        .normalized();
        assert_eq!(c.stale_after_ms, 500);
        assert_eq!(c.prune_after_ms, 500);
    }

    #[test]
    fn discovery_config_serde_roundtrip() {
        let c = DiscoveryConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let decoded: DiscoveryConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, c);
    }

    #[test]
    fn discovery_stats_default_all_zeros() {
        let s = DiscoveryStats::default();
        assert_eq!(s.sources_polled, 0);
        assert_eq!(s.sightings_observed, 0);
        assert_eq!(s.duplicates_merged, 0);
        assert_eq!(s.stale_instances, 0);
        assert_eq!(s.pruned_instances, 0);
    }

    #[test]
    fn discovery_stats_serde_roundtrip() {
        let s = DiscoveryStats {
            sources_polled: 3,
            sightings_observed: 10,
            duplicates_merged: 1,
            stale_instances: 2,
            pruned_instances: 0,
        };
        let json = serde_json::to_string(&s).unwrap();
        let decoded: DiscoveryStats = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, s);
    }

    #[test]
    fn normalized_folded_none() {
        assert_eq!(normalized_folded(None), None);
    }

    #[test]
    fn normalized_folded_empty() {
        assert_eq!(normalized_folded(Some("")), None);
    }

    #[test]
    fn normalized_folded_whitespace() {
        assert_eq!(normalized_folded(Some("   ")), None);
    }

    #[test]
    fn normalized_folded_trims_and_lowercases() {
        assert_eq!(
            normalized_folded(Some("  VALUE  ")),
            Some("value".to_owned())
        );
    }

    #[test]
    fn normalized_exact_none() {
        assert_eq!(normalized_exact(None), None);
    }

    #[test]
    fn normalized_exact_empty() {
        assert_eq!(normalized_exact(Some("")), None);
    }

    #[test]
    fn normalized_exact_whitespace() {
        assert_eq!(normalized_exact(Some("   ")), None);
    }

    #[test]
    fn normalized_exact_trims_preserves_case() {
        assert_eq!(
            normalized_exact(Some("  VALUE  ")),
            Some("VALUE".to_owned())
        );
    }

    #[test]
    fn normalized_endpoint_none() {
        assert_eq!(normalized_endpoint(None), None);
    }

    #[test]
    fn normalized_endpoint_no_scheme() {
        assert_eq!(
            normalized_endpoint(Some("localhost:9000")),
            Some("localhost:9000".to_owned())
        );
    }

    #[test]
    fn normalized_endpoint_lowercases_scheme_and_host() {
        assert_eq!(
            normalized_endpoint(Some("HTTP://MYHOST:8080/path")),
            Some("http://myhost:8080/path".to_owned())
        );
    }

    #[test]
    fn normalized_endpoint_preserves_path_case() {
        assert_eq!(
            normalized_endpoint(Some("http://host/MyPath")),
            Some("http://host/MyPath".to_owned())
        );
    }

    #[test]
    fn normalized_endpoint_empty_authority() {
        assert_eq!(
            normalized_endpoint(Some("HTTP:///path")),
            Some("http:///path".to_owned())
        );
    }

    #[test]
    fn normalize_host_port_ipv6() {
        assert_eq!(normalize_host_port("[::1]:8080"), "[::1]:8080".to_owned());
    }

    #[test]
    fn normalize_host_port_ipv6_uppercase() {
        assert_eq!(
            normalize_host_port("[FE80::1]:8080"),
            "[fe80::1]:8080".to_owned()
        );
    }

    #[test]
    fn normalize_host_port_with_port() {
        assert_eq!(normalize_host_port("MyHost:9090"), "myhost:9090".to_owned());
    }

    #[test]
    fn normalize_host_port_bare() {
        assert_eq!(normalize_host_port("MyHost"), "myhost".to_owned());
    }

    #[test]
    fn normalize_endpoint_authority_with_userinfo() {
        assert_eq!(
            normalize_endpoint_authority("user@MyHost:8080"),
            "user@myhost:8080".to_owned()
        );
    }

    #[test]
    fn normalize_endpoint_authority_no_userinfo() {
        assert_eq!(
            normalize_endpoint_authority("MyHost:8080"),
            "myhost:8080".to_owned()
        );
    }

    #[test]
    fn stable_instance_id_deterministic() {
        let id1 = stable_instance_id("seed-abc");
        let id2 = stable_instance_id("seed-abc");
        assert_eq!(id1, id2);
        assert!(id1.starts_with("inst-"));
    }

    #[test]
    fn stable_instance_id_different_seeds() {
        let id1 = stable_instance_id("seed-1");
        let id2 = stable_instance_id("seed-2");
        assert_ne!(id1, id2);
    }

    #[test]
    fn identity_keys_with_instance_key_hint() {
        let sighting = InstanceSighting {
            source: DiscoverySignalKind::Process,
            observed_at_ms: 100,
            project_key_hint: None,
            host_name: None,
            pid: None,
            instance_key_hint: Some("my-instance".to_owned()),
            control_endpoint: None,
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let keys = sighting.identity_keys();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], "instance:my-instance");
    }

    #[test]
    fn identity_keys_instance_key_hint_lowercased() {
        let sighting = InstanceSighting {
            source: DiscoverySignalKind::Process,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: None,
            pid: None,
            instance_key_hint: Some("MY-INSTANCE".to_owned()),
            control_endpoint: None,
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let keys = sighting.identity_keys();
        assert!(keys.contains(&"instance:my-instance".to_owned()));
    }

    #[test]
    fn identity_keys_host_pid_lowercases_host() {
        let sighting = InstanceSighting {
            source: DiscoverySignalKind::Process,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: Some("MY-HOST".to_owned()),
            pid: Some(42),
            instance_key_hint: None,
            control_endpoint: None,
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let keys = sighting.identity_keys();
        assert!(keys.contains(&"hostpid:my-host:42".to_owned()));
    }

    #[test]
    fn identity_keys_host_without_pid_no_hostpid_key() {
        let sighting = InstanceSighting {
            source: DiscoverySignalKind::Process,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: Some("host-a".to_owned()),
            pid: None,
            instance_key_hint: None,
            control_endpoint: None,
            socket_path: None,
            heartbeat_path: None,
            version: None,
        };
        let keys = sighting.identity_keys();
        assert!(!keys.iter().any(|k| k.starts_with("hostpid:")));
        // Falls back
        assert!(keys.iter().any(|k| k.starts_with("fallback:")));
    }

    #[test]
    fn identity_keys_multiple_signals() {
        let sighting = InstanceSighting {
            source: DiscoverySignalKind::Process,
            observed_at_ms: 1,
            project_key_hint: None,
            host_name: Some("host".to_owned()),
            pid: Some(99),
            instance_key_hint: Some("inst-x".to_owned()),
            control_endpoint: Some("http://host:8080/ctrl".to_owned()),
            socket_path: Some("/tmp/test.sock".to_owned()),
            heartbeat_path: Some("/tmp/hb".to_owned()),
            version: None,
        };
        let keys = sighting.identity_keys();
        assert!(keys.iter().any(|k| k.starts_with("instance:")));
        assert!(keys.iter().any(|k| k.starts_with("hostpid:")));
        assert!(keys.iter().any(|k| k.starts_with("endpoint:")));
        assert!(keys.iter().any(|k| k.starts_with("socket:")));
        assert!(keys.iter().any(|k| k.starts_with("heartbeat:")));
        assert!(!keys.iter().any(|k| k.starts_with("fallback:")));
    }

    #[test]
    fn engine_empty_snapshot() {
        let engine = DiscoveryEngine::new(DiscoveryConfig::default());
        assert!(engine.snapshot().is_empty());
    }

    #[test]
    fn engine_poll_no_sources() {
        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let stats = engine.poll(100, &mut []);
        assert_eq!(stats.sources_polled, 0);
        assert_eq!(stats.sightings_observed, 0);
        assert!(engine.snapshot().is_empty());
    }

    #[test]
    fn engine_merge_self_noop() {
        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        assert!(!engine.merge_instances("a", "a"));
    }

    #[test]
    fn engine_merge_nonexistent_from() {
        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        assert!(!engine.merge_instances("nonexistent", "also-nonexistent"));
    }

    #[test]
    fn static_source_returns_sightings() {
        let sightings = vec![process_sighting(10, "cass", "host-a", 42)];
        let mut source = StaticDiscoverySource::new(sightings);
        let collected = source.collect(100);
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].source, DiscoverySignalKind::Process);
        // Collect returns same sightings each time
        let again = source.collect(200);
        assert_eq!(again.len(), 1);
    }

    #[test]
    fn refresh_lowercase_hint_none_candidate() {
        let mut current = Some("existing".to_owned());
        refresh_lowercase_hint(&mut current, None, true);
        assert_eq!(current.as_deref(), Some("existing"));
    }

    #[test]
    fn refresh_lowercase_hint_prefer_true_updates() {
        let mut current = Some("old".to_owned());
        refresh_lowercase_hint(&mut current, Some("NEW"), true);
        assert_eq!(current.as_deref(), Some("new"));
    }

    #[test]
    fn refresh_lowercase_hint_prefer_false_keeps() {
        let mut current = Some("old".to_owned());
        refresh_lowercase_hint(&mut current, Some("new"), false);
        assert_eq!(current.as_deref(), Some("old"));
    }

    #[test]
    fn refresh_lowercase_hint_none_current_always_updates() {
        let mut current = None;
        refresh_lowercase_hint(&mut current, Some("VALUE"), false);
        assert_eq!(current.as_deref(), Some("value"));
    }

    #[test]
    fn refresh_pid_hint_none_candidate() {
        let mut current = Some(42);
        refresh_pid_hint(&mut current, None, 100, 50);
        assert_eq!(current, Some(42));
    }

    #[test]
    fn refresh_pid_hint_newer_updates() {
        let mut current = Some(42);
        refresh_pid_hint(&mut current, Some(99), 100, 50);
        assert_eq!(current, Some(99));
    }

    #[test]
    fn refresh_pid_hint_older_keeps() {
        let mut current = Some(42);
        refresh_pid_hint(&mut current, Some(99), 30, 50);
        assert_eq!(current, Some(42));
    }

    #[test]
    fn refresh_pid_hint_none_current_updates() {
        let mut current = None;
        refresh_pid_hint(&mut current, Some(99), 30, 50);
        assert_eq!(current, Some(99));
    }

    #[test]
    fn refresh_version_hint_none_candidate() {
        let mut current = Some("1.0".to_owned());
        refresh_version_hint(&mut current, None, 100, 50);
        assert_eq!(current.as_deref(), Some("1.0"));
    }

    #[test]
    fn refresh_version_hint_newer_updates() {
        let mut current = Some("1.0".to_owned());
        refresh_version_hint(&mut current, Some("2.0"), 100, 50);
        assert_eq!(current.as_deref(), Some("2.0"));
    }

    #[test]
    fn refresh_version_hint_older_keeps() {
        let mut current = Some("2.0".to_owned());
        refresh_version_hint(&mut current, Some("1.0"), 30, 50);
        assert_eq!(current.as_deref(), Some("2.0"));
    }

    #[test]
    fn instance_sighting_serde_roundtrip() {
        let sighting = process_sighting(42, "proj", "host", 99);
        let json = serde_json::to_string(&sighting).unwrap();
        let decoded: InstanceSighting = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, sighting);
    }

    #[test]
    fn discovered_instance_serde_roundtrip() {
        let inst = DiscoveredInstance {
            instance_id: "i1".to_owned(),
            project_key_hint: Some("proj".to_owned()),
            host_name: Some("host".to_owned()),
            pid: Some(42),
            version: Some("1.0".to_owned()),
            first_seen_ms: 10,
            last_seen_ms: 20,
            status: DiscoveryStatus::Active,
            sources: vec![DiscoverySignalKind::Process],
            identity_keys: vec!["hostpid:host:42".to_owned()],
        };
        let json = serde_json::to_string(&inst).unwrap();
        let decoded: DiscoveredInstance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, inst);
    }

    #[test]
    fn engine_snapshot_sorted_by_instance_id() {
        let mut engine = DiscoveryEngine::new(DiscoveryConfig::default());
        let mut source = StaticDiscoverySource::new(vec![
            process_sighting(10, "b", "host-b", 2),
            process_sighting(10, "a", "host-a", 1),
        ]);
        engine.poll(10, &mut [&mut source]);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot[0].instance_id < snapshot[1].instance_id);
    }

    // ─── bd-3l9h tests end ───
}
