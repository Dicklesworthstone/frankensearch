//! Host adapter SDK and conformance harness for telemetry integrations.
//!
//! This module provides:
//! - A stable adapter interface for host identity handshake, telemetry emission,
//!   and lifecycle hooks.
//! - Shared conformance validators for schema-version and redaction compliance.
//! - A harness that runs fixture-based contract checks with actionable diagnostics.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::collectors::{
    TELEMETRY_SCHEMA_VERSION, TelemetryCorrelation, TelemetryEnvelope, TelemetryEvent,
    TelemetryInstance,
};
use crate::error::{SearchError, SearchResult};

/// Default forbidden substrings used by redaction conformance checks.
///
/// These patterns intentionally target high-signal secret formats to minimize false positives
/// on normal telemetry payloads.
pub const DEFAULT_REDACTION_FORBIDDEN_PATTERNS: &[&str] = &[
    "BEGIN PRIVATE KEY",
    "AWS_SECRET_ACCESS_KEY=",
    "ghp_",
    "xoxb-",
];

/// Adapter identity handshake payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterIdentity {
    /// Stable adapter identifier (e.g. `xf-host-adapter`).
    pub adapter_id: String,
    /// Adapter implementation version.
    pub adapter_version: String,
    /// Host project identifier.
    pub host_project: String,
    /// Telemetry schema version emitted by this adapter.
    pub telemetry_schema_version: u8,
    /// Redaction policy version enforced by this adapter.
    pub redaction_policy_version: String,
}

/// Canonical host-project attribution result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProjectAttribution {
    /// Canonical project key or `unknown` when unresolved.
    pub resolved_project_key: String,
    /// Confidence score in `[0, 100]`.
    pub confidence_score: u8,
    /// Machine-stable reason code for diagnostics.
    pub reason_code: String,
    /// Whether multiple conflicting project candidates were observed.
    pub collision: bool,
}

impl HostProjectAttribution {
    #[must_use]
    pub fn unknown(reason_code: impl Into<String>) -> Self {
        Self {
            resolved_project_key: "unknown".to_owned(),
            confidence_score: 20,
            reason_code: reason_code.into(),
            collision: false,
        }
    }
}

/// Resolve canonical host-project attribution from adapter identity + telemetry hints.
///
/// The algorithm is deterministic with source-precedence tie breaks:
/// adapter identity > telemetry project key > host name hint.
#[must_use]
pub fn resolve_host_project_attribution(
    identity_host_project: Option<&str>,
    telemetry_project_key: Option<&str>,
    host_name_hint: Option<&str>,
) -> HostProjectAttribution {
    #[derive(Debug, Clone, Copy)]
    struct Candidate {
        project: &'static str,
        weight: u8,
        reason: &'static str,
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    if let Some(hint) = identity_host_project {
        for project in canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 4,
                reason: "adapter_identity",
            });
        }
    }

    if let Some(hint) = telemetry_project_key {
        for project in canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 3,
                reason: "telemetry_project_key",
            });
        }
    }

    if let Some(hint) = host_name_hint {
        for project in canonical_projects_from_hint(hint) {
            candidates.push(Candidate {
                project,
                weight: 1,
                reason: "host_name",
            });
        }
    }

    if candidates.is_empty() {
        return HostProjectAttribution::unknown("attribution.unknown");
    }

    let unique_projects: BTreeSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.project)
        .collect();
    let collision = unique_projects.len() > 1;

    let Some(winner) = candidates.into_iter().max_by(|left, right| {
        left.weight
            .cmp(&right.weight)
            .then_with(|| right.project.cmp(left.project))
            .then_with(|| right.reason.cmp(left.reason))
    }) else {
        return HostProjectAttribution::unknown("attribution.unknown");
    };

    let mut confidence_score: u8 = match winner.weight {
        4 => 95,
        3 => 85,
        1 => 60,
        _ => 50,
    };
    if collision {
        confidence_score = confidence_score.saturating_sub(25);
    }

    let reason_code = if collision {
        "attribution.collision".to_owned()
    } else {
        format!("attribution.{}", winner.reason)
    };

    HostProjectAttribution {
        resolved_project_key: winner.project.to_owned(),
        confidence_score,
        reason_code,
        collision,
    }
}

/// Lifecycle hooks exposed to host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AdapterLifecycleEvent {
    /// Adapter session startup.
    SessionStart { ts: String },
    /// Adapter session shutdown.
    SessionStop { ts: String },
    /// Periodic health tick.
    HealthTick { ts: String },
}

/// Host adapter interface used by integrations.
pub trait HostAdapter: fmt::Debug + Send + Sync {
    /// Identity handshake called during initialization and conformance checks.
    fn identity(&self) -> AdapterIdentity;

    /// Emit one canonical telemetry envelope.
    ///
    /// Implementations must be non-blocking and should preserve envelope fidelity.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when emission fails due to adapter transport,
    /// serialization, or sink-level validation problems.
    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()>;

    /// Lifecycle hook invoked by orchestration and conformance harnesses.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when lifecycle processing fails.
    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()>;
}

/// One machine-readable conformance violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceViolation {
    /// Stable violation code.
    pub code: String,
    /// Field/path associated with the violation.
    pub field: String,
    /// Human-readable diagnostic.
    pub message: String,
}

/// Summary result for an adapter conformance run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceReport {
    /// Whether all checks passed.
    pub passed: bool,
    /// Number of fixture events validated.
    pub fixtures_checked: usize,
    /// Number of successful adapter `emit_telemetry` calls.
    pub emitted_events: usize,
    /// Number of successful lifecycle hook invocations.
    pub lifecycle_hooks_checked: usize,
    /// Validation failures.
    pub violations: Vec<ConformanceViolation>,
}

impl ConformanceReport {
    const fn with_violations(
        fixtures_checked: usize,
        emitted_events: usize,
        lifecycle_hooks_checked: usize,
        violations: Vec<ConformanceViolation>,
    ) -> Self {
        Self {
            passed: violations.is_empty(),
            fixtures_checked,
            emitted_events,
            lifecycle_hooks_checked,
            violations,
        }
    }
}

/// Configuration for adapter conformance checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceConfig {
    /// Required telemetry schema version.
    pub expected_schema_version: u8,
    /// Required redaction policy version label.
    pub required_redaction_policy_version: String,
    /// Forbidden substrings that must not appear in serialized telemetry payloads.
    pub forbidden_substrings: Vec<String>,
}

impl Default for ConformanceConfig {
    fn default() -> Self {
        Self {
            expected_schema_version: TELEMETRY_SCHEMA_VERSION,
            required_redaction_policy_version: "v1".to_owned(),
            forbidden_substrings: DEFAULT_REDACTION_FORBIDDEN_PATTERNS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }
}

/// Conformance harness for host adapters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceHarness {
    config: ConformanceConfig,
}

impl Default for ConformanceHarness {
    fn default() -> Self {
        Self::new(ConformanceConfig::default())
    }
}

impl ConformanceHarness {
    /// Construct a harness with explicit configuration.
    #[must_use]
    pub const fn new(config: ConformanceConfig) -> Self {
        Self { config }
    }

    /// Access harness configuration.
    #[must_use]
    pub const fn config(&self) -> &ConformanceConfig {
        &self.config
    }

    /// Validate adapter identity handshake.
    #[must_use]
    pub fn validate_identity(&self, identity: &AdapterIdentity) -> Vec<ConformanceViolation> {
        let mut violations = Vec::new();

        if identity.adapter_id.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_adapter_id",
                "identity.adapter_id",
                "adapter_id must be non-empty",
            ));
        }
        if identity.adapter_version.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_adapter_version",
                "identity.adapter_version",
                "adapter_version must be non-empty",
            ));
        }
        if identity.host_project.trim().is_empty() {
            violations.push(violation(
                "adapter.identity.missing_host_project",
                "identity.host_project",
                "host_project must be non-empty",
            ));
        }

        if identity.telemetry_schema_version != self.config.expected_schema_version {
            violations.push(ConformanceViolation {
                code: "adapter.identity.schema_version_mismatch".to_owned(),
                field: "identity.telemetry_schema_version".to_owned(),
                message: format!(
                    "expected schema v{}, got v{}",
                    self.config.expected_schema_version, identity.telemetry_schema_version
                ),
            });
        }

        if identity.redaction_policy_version != self.config.required_redaction_policy_version {
            violations.push(ConformanceViolation {
                code: "adapter.identity.redaction_policy_mismatch".to_owned(),
                field: "identity.redaction_policy_version".to_owned(),
                message: format!(
                    "expected redaction policy {}, got {}",
                    self.config.required_redaction_policy_version,
                    identity.redaction_policy_version
                ),
            });
        }

        violations
    }

    /// Validate one telemetry envelope for schema/version and redaction compliance.
    #[must_use]
    pub fn validate_envelope(&self, envelope: &TelemetryEnvelope) -> Vec<ConformanceViolation> {
        let mut violations = Vec::new();

        if envelope.v != self.config.expected_schema_version {
            violations.push(ConformanceViolation {
                code: "adapter.envelope.schema_version_mismatch".to_owned(),
                field: "envelope.v".to_owned(),
                message: format!(
                    "expected schema v{}, got v{}",
                    self.config.expected_schema_version, envelope.v
                ),
            });
        }

        if envelope.ts.trim().is_empty() {
            violations.push(violation(
                "adapter.envelope.missing_timestamp",
                "envelope.ts",
                "timestamp must be non-empty",
            ));
        }

        match &envelope.event {
            TelemetryEvent::Search {
                instance,
                correlation,
                query,
                ..
            } => validate_search_event(instance, correlation, query.text.as_str(), &mut violations),
            TelemetryEvent::Embedding {
                instance,
                correlation,
                duration_ms,
                ..
            } => validate_duration_event(
                "embedding",
                instance,
                correlation,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Index {
                instance,
                correlation,
                duration_ms,
                ..
            } => validate_duration_event(
                "index",
                instance,
                correlation,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Resource {
                instance,
                correlation,
                sample,
            } => {
                validate_resource_event(instance, correlation, sample.interval_ms, &mut violations);
            }
            TelemetryEvent::Lifecycle {
                instance,
                correlation,
                ..
            } => validate_lifecycle_event(instance, correlation, &mut violations),
        }

        violations.extend(validate_redaction_forbidden_substrings(
            envelope,
            &self.config.forbidden_substrings,
        ));

        violations
    }

    /// Execute a full conformance run over identity, lifecycle hooks, and fixture envelopes.
    ///
    /// Lifecycle order:
    /// 1. `SessionStart`
    /// 2. per-envelope `HealthTick`
    /// 3. `SessionStop`
    #[must_use]
    pub fn run(
        &self,
        adapter: &dyn HostAdapter,
        fixtures: &[TelemetryEnvelope],
    ) -> ConformanceReport {
        let identity = adapter.identity();
        let mut violations = self.validate_identity(&identity);
        let mut lifecycle_hooks_checked = 0usize;
        let mut emitted_events = 0usize;

        let start_event = AdapterLifecycleEvent::SessionStart {
            ts: "conformance-start".to_owned(),
        };
        match adapter.on_lifecycle_event(&start_event) {
            Ok(()) => lifecycle_hooks_checked += 1,
            Err(err) => violations.push(adapter_error_violation("lifecycle_start", &err)),
        }

        for (idx, envelope) in fixtures.iter().enumerate() {
            let event_violations = self.validate_envelope(envelope);
            if !event_violations.is_empty() {
                for violation in event_violations {
                    violations.push(ConformanceViolation {
                        code: violation.code,
                        field: format!("fixtures[{idx}].{}", violation.field),
                        message: violation.message,
                    });
                }
            }

            match adapter.emit_telemetry(envelope) {
                Ok(()) => emitted_events += 1,
                Err(err) => violations.push(adapter_error_violation("emit_telemetry", &err)),
            }

            let tick_event = AdapterLifecycleEvent::HealthTick {
                ts: format!("conformance-tick-{idx}"),
            };
            match adapter.on_lifecycle_event(&tick_event) {
                Ok(()) => lifecycle_hooks_checked += 1,
                Err(err) => violations.push(adapter_error_violation("lifecycle_tick", &err)),
            }
        }

        let stop_event = AdapterLifecycleEvent::SessionStop {
            ts: "conformance-stop".to_owned(),
        };
        match adapter.on_lifecycle_event(&stop_event) {
            Ok(()) => lifecycle_hooks_checked += 1,
            Err(err) => violations.push(adapter_error_violation("lifecycle_stop", &err)),
        }

        ConformanceReport::with_violations(
            fixtures.len(),
            emitted_events,
            lifecycle_hooks_checked,
            violations,
        )
    }
}

fn validate_instance(
    prefix: &str,
    instance: &TelemetryInstance,
    violations: &mut Vec<ConformanceViolation>,
) {
    if !is_valid_ulid(&instance.instance_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.invalid_instance_id".to_owned(),
            field: format!("{prefix}.instance_id"),
            message: "instance_id must be a 26-char ULID".to_owned(),
        });
    }
    if instance.project_key.trim().is_empty() {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.missing_project_key".to_owned(),
            field: format!("{prefix}.project_key"),
            message: "project_key must be non-empty".to_owned(),
        });
    }
    if instance.host_name.trim().is_empty() {
        violations.push(ConformanceViolation {
            code: "adapter.event.instance.missing_host_name".to_owned(),
            field: format!("{prefix}.host_name"),
            message: "host_name must be non-empty".to_owned(),
        });
    }
}

fn validate_search_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    query_text: &str,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("search.instance", instance, violations);
    validate_correlation("search.correlation", correlation, violations);
    if query_text.trim().is_empty() {
        violations.push(violation(
            "adapter.event.search.empty_query",
            "search.query.text",
            "query text must be non-empty",
        ));
    }
    if query_text.len() > 500 {
        violations.push(ConformanceViolation {
            code: "adapter.event.search.query_too_long".to_owned(),
            field: "search.query.text".to_owned(),
            message: format!("query length {} exceeds 500", query_text.len()),
        });
    }
}

fn validate_duration_event(
    prefix: &str,
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance(&format!("{prefix}.instance"), instance, violations);
    validate_correlation(&format!("{prefix}.correlation"), correlation, violations);
    if duration_ms == 0 {
        violations.push(ConformanceViolation {
            code: format!("adapter.event.{prefix}.zero_duration"),
            field: format!("{prefix}.duration_ms"),
            message: "duration_ms should be > 0 for completed telemetry".to_owned(),
        });
    }
}

fn validate_resource_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    interval_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("resource.instance", instance, violations);
    validate_correlation("resource.correlation", correlation, violations);
    if interval_ms == 0 {
        violations.push(violation(
            "adapter.event.resource.zero_interval",
            "resource.sample.interval_ms",
            "interval_ms must be greater than zero",
        ));
    }
}

fn validate_lifecycle_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("lifecycle.instance", instance, violations);
    validate_correlation("lifecycle.correlation", correlation, violations);
}

fn validate_correlation(
    prefix: &str,
    correlation: &TelemetryCorrelation,
    violations: &mut Vec<ConformanceViolation>,
) {
    if !is_valid_ulid(&correlation.event_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_event_id".to_owned(),
            field: format!("{prefix}.event_id"),
            message: "event_id must be a 26-char ULID".to_owned(),
        });
    }
    if !is_valid_ulid(&correlation.root_request_id) {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_root_request_id".to_owned(),
            field: format!("{prefix}.root_request_id"),
            message: "root_request_id must be a 26-char ULID".to_owned(),
        });
    }
    if let Some(parent_event_id) = &correlation.parent_event_id
        && !is_valid_ulid(parent_event_id)
    {
        violations.push(ConformanceViolation {
            code: "adapter.event.correlation.invalid_parent_event_id".to_owned(),
            field: format!("{prefix}.parent_event_id"),
            message: "parent_event_id must be a 26-char ULID when provided".to_owned(),
        });
    }
}

fn validate_redaction_forbidden_substrings(
    envelope: &TelemetryEnvelope,
    forbidden_substrings: &[String],
) -> Vec<ConformanceViolation> {
    let mut violations = Vec::new();
    let payload = match serde_json::to_string(envelope) {
        Ok(payload) => payload,
        Err(source) => {
            violations.push(ConformanceViolation {
                code: "adapter.redaction.serialization_failed".to_owned(),
                field: "envelope".to_owned(),
                message: format!("failed to serialize envelope for redaction check: {source}"),
            });
            return violations;
        }
    };

    let payload_lower = payload.to_ascii_lowercase();
    for pattern in forbidden_substrings {
        if pattern.is_empty() {
            continue;
        }
        let pattern_lower = pattern.to_ascii_lowercase();
        if payload_lower.contains(&pattern_lower) {
            violations.push(ConformanceViolation {
                code: "adapter.redaction.forbidden_pattern".to_owned(),
                field: "envelope".to_owned(),
                message: format!("payload contains forbidden pattern '{pattern}'"),
            });
        }
    }

    violations
}

fn is_valid_ulid(candidate: &str) -> bool {
    if candidate.len() != 26 {
        return false;
    }
    candidate.bytes().all(|byte| {
        matches!(
            byte,
            b'0'..=b'9'
                | b'A'..=b'H'
                | b'J'..=b'K'
                | b'M'..=b'N'
                | b'P'..=b'T'
                | b'V'..=b'Z'
                | b'a'..=b'h'
                | b'j'..=b'k'
                | b'm'..=b'n'
                | b'p'..=b't'
                | b'v'..=b'z'
        )
    })
}

fn violation(code: &str, field: &str, message: &str) -> ConformanceViolation {
    ConformanceViolation {
        code: code.to_owned(),
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

fn adapter_error_violation(context: &str, error: &SearchError) -> ConformanceViolation {
    ConformanceViolation {
        code: "adapter.hook.error".to_owned(),
        field: context.to_owned(),
        message: error.to_string(),
    }
}

fn canonical_projects_from_hint(hint: &str) -> Vec<&'static str> {
    const CANONICAL_ALIASES: &[(&str, &[&str])] = &[
        (
            "coding_agent_session_search",
            &[
                "coding_agent_session_search",
                "codingagentsessionsearch",
                "cass",
            ],
        ),
        ("xf", &["xf"]),
        (
            "mcp_agent_mail_rust",
            &[
                "mcp_agent_mail_rust",
                "mcpagentmailrust",
                "mcpagentmail",
                "agent_mail",
                "agentmail",
                "amail",
            ],
        ),
        ("frankenterm", &["frankenterm"]),
    ];

    let normalized = normalize_project_hint(hint);
    if normalized.is_empty() {
        return Vec::new();
    }
    let tokens: BTreeSet<&str> = normalized
        .split('_')
        .filter(|token| !token.is_empty())
        .collect();

    let mut matches = Vec::new();
    for (canonical, aliases) in CANONICAL_ALIASES {
        if aliases
            .iter()
            .any(|alias| normalized == *alias || tokens.contains(alias))
        {
            matches.push(*canonical);
        }
    }

    matches.sort_unstable();
    matches.dedup();
    matches
}

fn normalize_project_hint(hint: &str) -> String {
    let mut normalized = String::with_capacity(hint.len());
    let mut pending_separator = false;

    for ch in hint.chars() {
        if ch.is_ascii_alphanumeric() {
            if pending_separator && !normalized.is_empty() {
                normalized.push('_');
            }
            normalized.push(ch.to_ascii_lowercase());
            pending_separator = false;
        } else {
            pending_separator = true;
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Debug)]
    struct MemoryHostAdapter {
        identity: AdapterIdentity,
        emitted: Arc<Mutex<Vec<TelemetryEnvelope>>>,
        lifecycle: Arc<Mutex<Vec<AdapterLifecycleEvent>>>,
        fail_emit: bool,
        fail_lifecycle: bool,
    }

    impl MemoryHostAdapter {
        fn new(identity: AdapterIdentity) -> Self {
            Self {
                identity,
                emitted: Arc::new(Mutex::new(Vec::new())),
                lifecycle: Arc::new(Mutex::new(Vec::new())),
                fail_emit: false,
                fail_lifecycle: false,
            }
        }
    }

    impl HostAdapter for MemoryHostAdapter {
        fn identity(&self) -> AdapterIdentity {
            self.identity.clone()
        }

        fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
            if self.fail_emit {
                return Err(SearchError::InvalidConfig {
                    field: "adapter.emit".to_owned(),
                    value: "forced_failure".to_owned(),
                    reason: "emit failure requested in test".to_owned(),
                });
            }
            self.emitted.lock().unwrap().push(envelope.clone());
            Ok(())
        }

        fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
            if self.fail_lifecycle {
                return Err(SearchError::InvalidConfig {
                    field: "adapter.lifecycle".to_owned(),
                    value: "forced_failure".to_owned(),
                    reason: "lifecycle failure requested in test".to_owned(),
                });
            }
            self.lifecycle.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn default_identity() -> AdapterIdentity {
        AdapterIdentity {
            adapter_id: "sample-host-adapter".to_owned(),
            adapter_version: "0.1.0".to_owned(),
            host_project: "sample-host".to_owned(),
            telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
            redaction_policy_version: "v1".to_owned(),
        }
    }

    fn load_fixture(name: &str) -> TelemetryEnvelope {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../schemas/fixtures")
            .join(name);
        let raw = fs::read_to_string(path).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    #[test]
    fn sample_host_adapter_passes_conformance_with_golden_fixtures() {
        let harness = ConformanceHarness::default();
        let adapter = MemoryHostAdapter::new(default_identity());
        let fixtures = vec![
            load_fixture("telemetry-search-v1.json"),
            load_fixture("telemetry-embedding-v1.json"),
            load_fixture("telemetry-index-v1.json"),
            load_fixture("telemetry-resource-v1.json"),
            load_fixture("telemetry-lifecycle-v1.json"),
        ];

        let report = harness.run(&adapter, &fixtures);
        assert!(report.passed, "{:?}", report.violations);
        assert_eq!(report.fixtures_checked, fixtures.len());
        assert_eq!(report.emitted_events, fixtures.len());
        assert!(report.lifecycle_hooks_checked >= 2);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn identity_schema_version_mismatch_is_reported() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.telemetry_schema_version = 9;

        let violations = harness.validate_identity(&identity);
        assert!(violations.iter().any(|violation| {
            violation.code == "adapter.identity.schema_version_mismatch"
                && violation.field == "identity.telemetry_schema_version"
        }));
    }

    #[test]
    fn redaction_forbidden_pattern_is_detected() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
            query.text = "BEGIN PRIVATE KEY".to_owned();
        } else {
            panic!("fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        assert!(
            violations
                .iter()
                .any(|violation| violation.code == "adapter.redaction.forbidden_pattern")
        );
    }

    #[test]
    fn harness_reports_adapter_hook_errors_with_diagnostics() {
        let harness = ConformanceHarness::default();
        let mut adapter = MemoryHostAdapter::new(default_identity());
        adapter.fail_emit = true;
        adapter.fail_lifecycle = true;

        let fixtures = vec![load_fixture("telemetry-search-v1.json")];
        let report = harness.run(&adapter, &fixtures);

        assert!(!report.passed);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.code == "adapter.hook.error")
        );
    }

    #[test]
    fn host_project_attribution_prefers_adapter_identity_hint() {
        let attribution = resolve_host_project_attribution(
            Some("mcp_agent_mail_rust"),
            Some("agent-mail"),
            Some("mail-host-01"),
        );
        assert_eq!(attribution.resolved_project_key, "mcp_agent_mail_rust");
        assert_eq!(attribution.reason_code, "attribution.adapter_identity");
        assert!(!attribution.collision);
        assert!(attribution.confidence_score >= 90);
    }

    #[test]
    fn host_project_attribution_falls_back_to_unknown_bucket() {
        let attribution =
            resolve_host_project_attribution(None, Some("custom-app"), Some("odd-host-name"));
        assert_eq!(attribution.resolved_project_key, "unknown");
        assert_eq!(attribution.reason_code, "attribution.unknown");
        assert_eq!(attribution.confidence_score, 20);
        assert!(!attribution.collision);
    }

    #[test]
    fn host_project_attribution_flags_collisions() {
        let attribution =
            resolve_host_project_attribution(Some("cass"), Some("xf"), Some("mixed-host"));
        assert_eq!(
            attribution.resolved_project_key,
            "coding_agent_session_search"
        );
        assert_eq!(attribution.reason_code, "attribution.collision");
        assert!(attribution.collision);
        assert!(attribution.confidence_score < 95);
    }

    #[test]
    fn host_project_attribution_uses_host_name_hint_when_needed() {
        let attribution =
            resolve_host_project_attribution(None, None, Some("frankenterm-prod-runner"));
        assert_eq!(attribution.resolved_project_key, "frankenterm");
        assert_eq!(attribution.reason_code, "attribution.host_name");
        assert!(!attribution.collision);
    }
}
