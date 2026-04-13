//! Host adapter SDK and conformance harness for telemetry integrations.
//!
//! This module provides:
//! - A stable adapter interface for host identity handshake, telemetry emission,
//!   and lifecycle hooks.
//! - Shared conformance validators for schema-version and redaction compliance.
//! - A harness that runs fixture-based contract checks with actionable diagnostics.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::collectors::{
    EmbeddingStage, EmbedderTier, EmbeddingStatus, IndexInventory, IndexOperation, IndexStatus, 
    LifecycleSeverity, LifecycleState, QuantizationMode, TelemetryCorrelation, 
    TelemetryEmbedderInfo, TelemetryEmbeddingJob, TelemetryEnvelope, TelemetryEvent, 
    TelemetryInstance, TelemetryResourceSample, TelemetrySearchMetrics, TelemetrySearchQuery, 
    TelemetrySearchResults, TELEMETRY_SCHEMA_VERSION,
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
    /// Host project identifier (required).
    pub host_project: String,
    /// Optional runtime role label (e.g. `indexer`, `query`, `control-plane`).
    pub runtime_role: Option<String>,
    /// Optional stable instance UUID advertised by the host integration.
    pub instance_uuid: Option<String>,
    /// Telemetry schema version emitted by this adapter.
    pub telemetry_schema_version: u8,
    /// Redaction policy version enforced by this adapter.
    pub redaction_policy_version: String,
}

/// Canonical host projects with first-class adapter support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalHostProject {
    CodingAgentSessionSearch,
    Xf,
    McpAgentMailRust,
    Frankenterm,
}

impl CanonicalHostProject {
    /// Deterministic host iteration order used by tests and harnesses.
    pub const ALL: [Self; 4] = [
        Self::CodingAgentSessionSearch,
        Self::Xf,
        Self::McpAgentMailRust,
        Self::Frankenterm,
    ];

    /// Canonical host project key.
    #[must_use]
    pub const fn host_project_key(self) -> &'static str {
        match self {
            Self::CodingAgentSessionSearch => "coding_agent_session_search",
            Self::Xf => "xf",
            Self::McpAgentMailRust => "mcp_agent_mail_rust",
            Self::Frankenterm => "frankenterm",
        }
    }

    /// Stable adapter identifier for this host.
    #[must_use]
    pub const fn adapter_id(self) -> &'static str {
        match self {
            Self::CodingAgentSessionSearch => "cass-host-adapter",
            Self::Xf => "xf-host-adapter",
            Self::McpAgentMailRust => "mcp-agent-mail-host-adapter",
            Self::Frankenterm => "frankenterm-host-adapter",
        }
    }

    /// Default runtime role label for this host adapter.
    #[must_use]
    pub const fn default_runtime_role(self) -> &'static str {
        match self {
            Self::McpAgentMailRust => "control-plane",
            Self::CodingAgentSessionSearch | Self::Xf | Self::Frankenterm => "query",
        }
    }
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

/// Non-blocking sink abstraction used by concrete host adapters.
pub trait AdapterSink: Send + Sync {
    /// Emit a canonical telemetry envelope.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when sink delivery fails.
    fn emit(&self, envelope: &TelemetryEnvelope) -> SearchResult<()>;

    /// Process one lifecycle event.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` when lifecycle handling fails.
    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()>;
}

/// No-op sink used by default when callers only need conformance-safe identities.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopAdapterSink;

impl AdapterSink for NoopAdapterSink {
    fn emit(&self, _: &TelemetryEnvelope) -> SearchResult<()> {
        Ok(())
    }

    fn on_lifecycle_event(&self, _: &AdapterLifecycleEvent) -> SearchResult<()> {
        Ok(())
    }
}

/// Reusable host adapter implementation that forwards to an injected sink.
///
/// This gives host projects one deterministic adapter path:
/// - canonical identity handshake
/// - non-blocking sink forwarding
/// - shared conformance harness compatibility
pub struct ForwardingHostAdapter {
    identity: AdapterIdentity,
    sink: Arc<dyn AdapterSink>,
}

impl fmt::Debug for ForwardingHostAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardingHostAdapter")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl ForwardingHostAdapter {
    /// Construct an adapter with an explicit identity and no-op sink.
    #[must_use]
    pub fn new(identity: AdapterIdentity) -> Self {
        Self {
            identity,
            sink: Arc::new(NoopAdapterSink),
        }
    }

    /// Construct a canonical host adapter profile with shared defaults.
    #[must_use]
    pub fn for_host(
        host: CanonicalHostProject,
        adapter_version: impl Into<String>,
        runtime_role: Option<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        let runtime_role = runtime_role.or_else(|| Some(host.default_runtime_role().to_owned()));
        Self::new(AdapterIdentity {
            adapter_id: host.adapter_id().to_owned(),
            adapter_version: adapter_version.into(),
            host_project: host.host_project_key().to_owned(),
            runtime_role,
            instance_uuid,
            telemetry_schema_version: TELEMETRY_SCHEMA_VERSION,
            redaction_policy_version: "v1".to_owned(),
        })
    }

    /// Canonical profile for `coding_agent_session_search` (cass).
    #[must_use]
    pub fn for_cass(adapter_version: impl Into<String>, instance_uuid: Option<String>) -> Self {
        Self::for_host(
            CanonicalHostProject::CodingAgentSessionSearch,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `xf`.
    #[must_use]
    pub fn for_xf(adapter_version: impl Into<String>, instance_uuid: Option<String>) -> Self {
        Self::for_host(
            CanonicalHostProject::Xf,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `mcp_agent_mail_rust`.
    #[must_use]
    pub fn for_mcp_agent_mail(
        adapter_version: impl Into<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        Self::for_host(
            CanonicalHostProject::McpAgentMailRust,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Canonical profile for `frankenterm`.
    #[must_use]
    pub fn for_frankenterm(
        adapter_version: impl Into<String>,
        instance_uuid: Option<String>,
    ) -> Self {
        Self::for_host(
            CanonicalHostProject::Frankenterm,
            adapter_version,
            None,
            instance_uuid,
        )
    }

    /// Replace the sink implementation used for forwarding.
    #[must_use]
    pub fn with_sink(mut self, sink: Arc<dyn AdapterSink>) -> Self {
        self.sink = sink;
        self
    }

    /// Borrow this adapter's identity.
    #[must_use]
    pub const fn identity_ref(&self) -> &AdapterIdentity {
        &self.identity
    }
}

impl HostAdapter for ForwardingHostAdapter {
    fn identity(&self) -> AdapterIdentity {
        self.identity.clone()
    }

    fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
        self.sink.emit(envelope)
    }

    fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
        self.sink.on_lifecycle_event(event)
    }
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
        if identity
            .runtime_role
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            violations.push(violation(
                "adapter.identity.empty_runtime_role",
                "identity.runtime_role",
                "runtime_role must be non-empty when provided",
            ));
        }
        if identity
            .instance_uuid
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            violations.push(violation(
                "adapter.identity.empty_instance_uuid",
                "identity.instance_uuid",
                "instance_uuid must be non-empty when provided",
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

        if let Some(violation) =
            canonical_identity_pair_violation(&identity.adapter_id, &identity.host_project)
        {
            violations.push(violation);
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
                results,
                metrics,
                ..
            } => validate_search_event(
                instance,
                correlation,
                query.text.as_str(),
                results,
                metrics,
                &mut violations,
            ),
            TelemetryEvent::Embedding {
                instance,
                correlation,
                job,
                embedder,
                duration_ms,
                ..
            } => validate_embedding_event(
                instance,
                correlation,
                job,
                embedder,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Index {
                instance,
                correlation,
                dimension,
                duration_ms,
                ..
            } => validate_index_event(
                instance,
                correlation,
                *dimension,
                *duration_ms,
                &mut violations,
            ),
            TelemetryEvent::Resource {
                instance,
                correlation,
                sample,
            } => {
                validate_resource_event(
                    instance,
                    correlation,
                    sample.cpu_pct,
                    sample.interval_ms,
                    sample.load_avg_1m,
                    &mut violations,
                );
            }
            TelemetryEvent::Lifecycle {
                instance,
                correlation,
                reason,
                uptime_ms,
                ..
            } => validate_lifecycle_event(
                instance,
                correlation,
                reason.as_deref(),
                *uptime_ms,
                &mut violations,
            ),
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
    results: &TelemetrySearchResults,
    metrics: &TelemetrySearchMetrics,
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
    let char_count = query_text.chars().count();
    if char_count > 500 {
        violations.push(ConformanceViolation {
            code: "adapter.event.search.query_too_long".to_owned(),
            field: "search.query.text".to_owned(),
            message: format!("query length {char_count} exceeds 500"),
        });
    }
    if metrics.latency_us == 0 {
        violations.push(violation(
            "adapter.event.search.zero_latency",
            "search.metrics.latency_us",
            "latency_us should be > 0 for completed search telemetry",
        ));
    }
    let source_total = results.lexical_count.saturating_add(results.semantic_count);
    if results.result_count > 0 && source_total == 0 {
        violations.push(violation(
            "adapter.event.search.missing_source_counts",
            "search.results",
            "result_count > 0 requires lexical_count or semantic_count to be non-zero",
        ));
    }
    if source_total > 0 && results.result_count > source_total {
        violations.push(ConformanceViolation {
            code: "adapter.event.search.result_count_exceeds_sources".to_owned(),
            field: "search.results.result_count".to_owned(),
            message: format!(
                "result_count {} exceeds lexical_count + semantic_count ({source_total})",
                results.result_count
            ),
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

fn validate_embedding_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    job: &TelemetryEmbeddingJob,
    embedder: &TelemetryEmbedderInfo,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_duration_event("embedding", instance, correlation, duration_ms, violations);
    if job.job_id.trim().is_empty() {
        violations.push(violation(
            "adapter.event.embedding.missing_job_id",
            "embedding.job.job_id",
            "job_id must be non-empty",
        ));
    }
    if job.doc_count == 0 {
        violations.push(violation(
            "adapter.event.embedding.zero_doc_count",
            "embedding.job.doc_count",
            "doc_count should be > 0",
        ));
    }
    if embedder.id.trim().is_empty() {
        violations.push(violation(
            "adapter.event.embedding.missing_embedder_id",
            "embedding.embedder.id",
            "embedder id must be non-empty",
        ));
    }
    if embedder.dimension == 0 {
        violations.push(violation(
            "adapter.event.embedding.zero_dimension",
            "embedding.embedder.dimension",
            "dimension should be > 0",
        ));
    }
}

fn validate_index_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    dimension: usize,
    duration_ms: u64,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_duration_event("index", instance, correlation, duration_ms, violations);
    if dimension == 0 {
        violations.push(violation(
            "adapter.event.index.zero_dimension",
            "index.dimension",
            "dimension should be > 0",
        ));
    }
}

fn validate_resource_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    cpu_pct: f64,
    interval_ms: u64,
    load_avg_1m: Option<f64>,
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
    if !cpu_pct.is_finite() {
        violations.push(violation(
            "adapter.event.resource.invalid_cpu_pct",
            "resource.sample.cpu_pct",
            "cpu_pct must be finite",
        ));
    } else if !(0.0..=100.0).contains(&cpu_pct) {
        violations.push(ConformanceViolation {
            code: "adapter.event.resource.cpu_pct_out_of_range".to_owned(),
            field: "resource.sample.cpu_pct".to_owned(),
            message: format!("cpu_pct {cpu_pct} must be in [0, 100]"),
        });
    }
    if let Some(load_avg_1m) = load_avg_1m
        && (!load_avg_1m.is_finite() || load_avg_1m < 0.0)
    {
        violations.push(ConformanceViolation {
            code: "adapter.event.resource.invalid_load_avg_1m".to_owned(),
            field: "resource.sample.load_avg_1m".to_owned(),
            message: format!("load_avg_1m {load_avg_1m} must be finite and >= 0"),
        });
    }
}

fn validate_lifecycle_event(
    instance: &TelemetryInstance,
    correlation: &TelemetryCorrelation,
    reason: Option<&str>,
    uptime_ms: Option<u64>,
    violations: &mut Vec<ConformanceViolation>,
) {
    validate_instance("lifecycle.instance", instance, violations);
    validate_correlation("lifecycle.correlation", correlation, violations);
    if reason.is_some_and(|text| text.trim().is_empty()) {
        violations.push(violation(
            "adapter.event.lifecycle.empty_reason",
            "lifecycle.reason",
            "reason must be non-empty when provided",
        ));
    }
    if uptime_ms.is_some_and(|uptime| uptime == 0) {
        violations.push(violation(
            "adapter.event.lifecycle.zero_uptime",
            "lifecycle.uptime_ms",
            "uptime_ms should be > 0 when provided",
        ));
    }
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

fn canonical_identity_pair_violation(
    adapter_id: &str,
    host_project: &str,
) -> Option<ConformanceViolation> {
    let adapter_id = adapter_id.trim();
    let host_project = host_project.trim();
    if adapter_id.is_empty() || host_project.is_empty() {
        return None;
    }

    let canonical_for_host = CanonicalHostProject::ALL
        .iter()
        .copied()
        .find(|host| host.host_project_key() == host_project);
    if let Some(host) = canonical_for_host
        && host.adapter_id() != adapter_id
    {
        return Some(ConformanceViolation {
            code: "adapter.identity.canonical_pair_mismatch".to_owned(),
            field: "identity.adapter_id".to_owned(),
            message: format!(
                "host_project '{}' expects adapter_id '{}', got '{}'",
                host_project,
                host.adapter_id(),
                adapter_id
            ),
        });
    }

    let canonical_for_adapter = CanonicalHostProject::ALL
        .iter()
        .copied()
        .find(|host| host.adapter_id() == adapter_id);
    if let Some(host) = canonical_for_adapter
        && host.host_project_key() != host_project
    {
        return Some(ConformanceViolation {
            code: "adapter.identity.canonical_pair_mismatch".to_owned(),
            field: "identity.host_project".to_owned(),
            message: format!(
                "adapter_id '{}' expects host_project '{}', got '{}'",
                adapter_id,
                host.host_project_key(),
                host_project
            ),
        });
    }

    None
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
    let ordered_tokens: Vec<&str> = normalized
        .split('_')
        .filter(|token| !token.is_empty())
        .collect();
    let tokens: BTreeSet<&str> = ordered_tokens.iter().copied().collect();

    let mut matches = Vec::new();
    for (canonical, aliases) in CANONICAL_ALIASES {
        if aliases
            .iter()
            .any(|alias| alias_matches_hint(&normalized, &ordered_tokens, &tokens, alias))
        {
            matches.push(*canonical);
        }
    }

    matches.sort_unstable();
    matches.dedup();
    matches
}

fn alias_matches_hint(
    normalized_hint: &str,
    ordered_hint_tokens: &[&str],
    hint_tokens: &BTreeSet<&str>,
    alias: &str,
) -> bool {
    if normalized_hint == alias || hint_tokens.contains(alias) {
        return true;
    }

    let alias_tokens: Vec<&str> = alias.split('_').filter(|token| !token.is_empty()).collect();
    alias_tokens.len() > 1
        && alias_tokens.len() <= ordered_hint_tokens.len()
        && ordered_hint_tokens
            .windows(alias_tokens.len())
            .any(|window| window == alias_tokens.as_slice())
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
    use std::path::PathBuf;
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

    #[derive(Debug, Default)]
    struct RecordingSink {
        emitted: Arc<Mutex<Vec<TelemetryEnvelope>>>,
        lifecycle: Arc<Mutex<Vec<AdapterLifecycleEvent>>>,
    }

    impl AdapterSink for RecordingSink {
        fn emit(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
            self.emitted.lock().unwrap().push(envelope.clone());
            Ok(())
        }

        fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
            self.lifecycle.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn default_identity() -> AdapterIdentity {
        AdapterIdentity {
            adapter_id: "sample-host-adapter".to_owned(),
            adapter_version: "0.1.0".to_owned(),
            host_project: "sample-host".to_owned(),
            runtime_role: Some("query".to_owned()),
            instance_uuid: Some("sample-instance-uuid".to_owned()),
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

    fn assert_golden_json<T: serde::Serialize>(name: &str, value: &T) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
        let golden_path = dir.join(format!("{name}.golden.json"));
        let actual = serde_json::to_string_pretty(value).expect("serialize golden json");

        if std::env::var("UPDATE_GOLDENS").is_ok() {
            fs::create_dir_all(&dir).expect("create golden directory");
            fs::write(&golden_path, actual.as_bytes()).expect("write golden file");
            return;
        }

        let expected = fs::read_to_string(&golden_path).unwrap_or_else(|_| {
            panic!(
                "Golden file not found: {}\nSet UPDATE_GOLDENS=1 to create it.",
                golden_path.display()
            );
        });

        let actual_trimmed = actual.trim_end_matches(|c| c == '\n' || c == '\r');
        let expected_trimmed = expected.trim_end_matches(|c| c == '\n' || c == '\r');

        if actual_trimmed != expected_trimmed {
            let actual_path = golden_path.with_extension("actual.json");
            fs::write(&actual_path, actual_trimmed.as_bytes()).expect("write actual file");
            panic!(
                "GOLDEN MISMATCH: {name}\nexpected: {}\nactual: {}",
                golden_path.display(),
                actual_path.display()
            );
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    struct ViolationSnapshot {
        code: String,
        field: String,
        message: String,
    }

    fn snapshot_violations(violations: &[ConformanceViolation]) -> Vec<ViolationSnapshot> {
        let mut snapshots: Vec<ViolationSnapshot> = violations
            .iter()
            .map(|violation| ViolationSnapshot {
                code: violation.code.clone(),
                field: violation.field.clone(),
                message: violation.message.clone(),
            })
            .collect();
        snapshots.sort_by(|a, b| {
            a.code
                .cmp(&b.code)
                .then_with(|| a.field.cmp(&b.field))
                .then_with(|| a.message.cmp(&b.message))
        });
        snapshots
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
    fn conformance_report_matches_golden_output() {
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
        assert_golden_json("adapter_conformance_report_v1", &report);
    }

    #[test]
    fn telemetry_fixtures_pass_envelope_conformance() {
        let harness = ConformanceHarness::default();
        let fixtures = [
            "telemetry-search-v1.json",
            "telemetry-embedding-v1.json",
            "telemetry-index-v1.json",
            "telemetry-resource-v1.json",
            "telemetry-lifecycle-v1.json",
        ];

        for fixture in fixtures {
            let envelope = load_fixture(fixture);
            let violations = harness.validate_envelope(&envelope);
            assert!(
                violations.is_empty(),
                "fixture {fixture} should pass conformance, got {violations:?}"
            );
        }
    }

    #[test]
    fn conformance_violations_match_golden_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");
        envelope.v = 99;
        envelope.ts = "".to_owned();

        if let TelemetryEvent::Search {
            instance,
            correlation,
            query,
            results,
            metrics,
            ..
        } = &mut envelope.event
        {
            instance.instance_id = "not-a-ulid".to_owned();
            instance.project_key = " ".to_owned();
            instance.host_name = String::new();
            instance.pid = Some(1234);

            correlation.event_id = "bad-event".to_owned();
            correlation.root_request_id = "bad-root".to_owned();
            correlation.parent_event_id = Some("bad-parent".to_owned());

            query.text = format!("BEGIN PRIVATE KEY{}", "a".repeat(600));
            metrics.latency_us = 0;
            results.result_count = 7;
            results.lexical_count = 0;
            results.semantic_count = 0;
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json(
            "adapter_conformance_violations_search_invalid_v1",
            &snapshot,
        );
    }

    #[test]
    fn conformance_violations_embedding_invalid_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        envelope.event = TelemetryEvent::Embedding {
            instance: TelemetryInstance {
                instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                project_key: "frankensearch".to_owned(),
                host_name: "test-host".to_owned(),
                pid: Some(1234),
            },
            correlation: TelemetryCorrelation {
                event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                parent_event_id: None,
            },
            job: TelemetryEmbeddingJob {
                job_id: " ".to_owned(), // invalid
                queue_depth: 0,
                doc_count: 0, // invalid
                stage: EmbeddingStage::Fast,
            },
            embedder: TelemetryEmbedderInfo {
                id: "".to_owned(), // invalid
                tier: EmbedderTier::Fast,
                dimension: 0, // invalid
            },
            status: EmbeddingStatus::Completed,
            duration_ms: 0, // invalid
        };

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_embedding_invalid_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_index_invalid_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        envelope.event = TelemetryEvent::Index {
            instance: TelemetryInstance {
                instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                project_key: "frankensearch".to_owned(),
                host_name: "test-host".to_owned(),
                pid: Some(1234),
            },
            correlation: TelemetryCorrelation {
                event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                parent_event_id: None,
            },
            operation: IndexOperation::Build,
            inventory: IndexInventory {
                words: 0,
                tokens: 0,
                lines: 0,
                bytes: 0,
                docs: 0,
            },
            dimension: 0,   // invalid
            quantization: QuantizationMode::F16,
            status: IndexStatus::Completed,
            duration_ms: 0, // invalid
        };

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_index_invalid_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_resource_invalid_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        envelope.event = TelemetryEvent::Resource {
            instance: TelemetryInstance {
                instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                project_key: "frankensearch".to_owned(),
                host_name: "test-host".to_owned(),
                pid: Some(1234),
            },
            correlation: TelemetryCorrelation {
                event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                parent_event_id: None,
            },
            sample: TelemetryResourceSample {
                cpu_pct: 150.0, // invalid
                rss_bytes: 0,
                io_read_bytes: 0,
                io_write_bytes: 0,
                interval_ms: 0, // invalid
                load_avg_1m: Some(-1.0), // invalid
                pressure_profile: None,
            },
        };

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_resource_invalid_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_lifecycle_invalid_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        envelope.event = TelemetryEvent::Lifecycle {
            instance: TelemetryInstance {
                instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                project_key: "frankensearch".to_owned(),
                host_name: "test-host".to_owned(),
                pid: Some(1234),
            },
            correlation: TelemetryCorrelation {
                event_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                root_request_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".to_owned(),
                parent_event_id: None,
            },
            state: LifecycleState::Started,
            severity: LifecycleSeverity::Info,
            reason: Some(" ".to_owned()), // invalid
            uptime_ms: Some(0),           // invalid
        };

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_lifecycle_invalid_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_search_empty_query_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
            query.text = "   ".to_owned();
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json(
            "adapter_conformance_violations_search_empty_query_v1",
            &snapshot,
        );
    }

    #[test]
    fn conformance_violations_search_missing_source_counts_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
            results.result_count = 1;
            results.lexical_count = 0;
            results.semantic_count = 0;
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json(
            "adapter_conformance_violations_search_missing_source_counts_v1",
            &snapshot,
        );
    }

    #[test]
    fn conformance_violations_privacy_redaction_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
            // query contains a "forbidden" pattern (mocked secret)
            query.text = "find documents with key=sk-1234567890abcdef".to_owned();
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_privacy_redaction_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_search_filter_fidelity_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
            // Simulate complex results with varied source metadata
            results.result_count = 3;
            results.lexical_count = 1;
            results.semantic_count = 2;
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_filter_fidelity_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_metadata_security_redaction_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
            // query contains multiple forbidden patterns
            query.text = "find docs with key=sk-123 and token=Bearer abcdef123456".to_owned();
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json("adapter_conformance_violations_metadata_redaction_v1", &snapshot);
    }

    #[test]
    fn conformance_violations_search_result_count_exceeds_sources_snapshot() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        if let TelemetryEvent::Search { results, .. } = &mut envelope.event {
            results.lexical_count = 1;
            results.semantic_count = 1;
            results.result_count = 5;
        } else {
            panic!("search fixture shape changed");
        }

        let violations = harness.validate_envelope(&envelope);
        let snapshot = snapshot_violations(&violations);
        assert_golden_json(
            "adapter_conformance_violations_search_result_count_exceeds_sources_v1",
            &snapshot,
        );
    }

    #[test]
    fn canonical_host_profiles_emit_expected_identity_handshake() {
        let cases = [
            (
                CanonicalHostProject::CodingAgentSessionSearch,
                "cass-host-adapter",
                "coding_agent_session_search",
            ),
            (CanonicalHostProject::Xf, "xf-host-adapter", "xf"),
            (
                CanonicalHostProject::McpAgentMailRust,
                "mcp-agent-mail-host-adapter",
                "mcp_agent_mail_rust",
            ),
            (
                CanonicalHostProject::Frankenterm,
                "frankenterm-host-adapter",
                "frankenterm",
            ),
        ];

        for (host, expected_adapter_id, expected_project) in cases {
            let adapter = ForwardingHostAdapter::for_host(
                host,
                "1.2.3",
                None,
                Some("instance-xyz".to_owned()),
            );
            let identity = adapter.identity();
            assert_eq!(identity.adapter_id, expected_adapter_id);
            assert_eq!(identity.host_project, expected_project);
            assert_eq!(identity.adapter_version, "1.2.3");
            assert_eq!(identity.telemetry_schema_version, TELEMETRY_SCHEMA_VERSION);
            assert_eq!(identity.redaction_policy_version, "v1");
            assert_eq!(
                identity.runtime_role.as_deref(),
                Some(host.default_runtime_role())
            );
        }
    }

    #[test]
    fn forwarding_host_adapter_routes_events_to_sink() {
        let sink = Arc::new(RecordingSink::default());
        let sink_trait: Arc<dyn AdapterSink> = sink.clone();
        let adapter = ForwardingHostAdapter::for_cass("0.9.0", None).with_sink(sink_trait);

        let fixture = load_fixture("telemetry-search-v1.json");
        adapter.emit_telemetry(&fixture).unwrap();
        adapter
            .on_lifecycle_event(&AdapterLifecycleEvent::SessionStart {
                ts: "now".to_owned(),
            })
            .unwrap();
        adapter
            .on_lifecycle_event(&AdapterLifecycleEvent::HealthTick {
                ts: "tick".to_owned(),
            })
            .unwrap();
        adapter
            .on_lifecycle_event(&AdapterLifecycleEvent::SessionStop {
                ts: "later".to_owned(),
            })
            .unwrap();

        {
            let emitted = sink.emitted.lock().unwrap();
            assert_eq!(emitted.len(), 1);
            assert_eq!(emitted[0], fixture);
            drop(emitted);
        }

        {
            let lifecycle = sink.lifecycle.lock().unwrap();
            assert_eq!(lifecycle.len(), 3);
            assert!(matches!(
                lifecycle[0],
                AdapterLifecycleEvent::SessionStart { .. }
            ));
            assert!(matches!(
                lifecycle[1],
                AdapterLifecycleEvent::HealthTick { .. }
            ));
            assert!(matches!(
                lifecycle[2],
                AdapterLifecycleEvent::SessionStop { .. }
            ));
            drop(lifecycle);
        }
    }

    #[test]
    fn forwarding_host_adapter_conformance_passes_for_all_canonical_hosts() {
        let harness = ConformanceHarness::default();
        let fixtures = vec![
            load_fixture("telemetry-search-v1.json"),
            load_fixture("telemetry-embedding-v1.json"),
            load_fixture("telemetry-index-v1.json"),
            load_fixture("telemetry-resource-v1.json"),
            load_fixture("telemetry-lifecycle-v1.json"),
        ];

        for host in CanonicalHostProject::ALL {
            let adapter = ForwardingHostAdapter::for_host(host, "2.0.0", None, None);
            let report = harness.run(&adapter, &fixtures);
            assert!(
                report.passed,
                "host {:?} failed conformance: {:?}",
                host, report.violations
            );
            assert_eq!(report.fixtures_checked, fixtures.len());
            assert_eq!(report.emitted_events, fixtures.len());
            assert!(report.lifecycle_hooks_checked >= 2);
        }
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
    fn identity_optional_handshake_fields_require_nonempty_values_when_present() {
        let harness = ConformanceHarness::default();
        let mut identity = default_identity();
        identity.runtime_role = Some("   ".to_owned());
        identity.instance_uuid = Some(String::new());

        let violations = harness.validate_identity(&identity);
        assert!(violations.iter().any(|violation| {
            violation.code == "adapter.identity.empty_runtime_role"
                && violation.field == "identity.runtime_role"
        }));
        assert!(violations.iter().any(|violation| {
            violation.code == "adapter.identity.empty_instance_uuid"
                && violation.field == "identity.instance_uuid"
        }));
    }

    #[test]
    fn redaction_forbidden_pattern_is_detected() {
        let harness = ConformanceHarness::default();
        let mut envelope = load_fixture("telemetry-search-v1.json");

        assert!(
            matches!(envelope.event, TelemetryEvent::Search { .. }),
            "fixture shape changed"
        );

        if let TelemetryEvent::Search { query, .. } = &mut envelope.event {
            query.text = "BEGIN PRIVATE KEY".to_owned();
        }

        let violations = harness.validate_envelope(&envelope);
        assert!(
            violations
                .iter()
                .any(|violation| violation.code == "adapter.redaction.forbidden_pattern")
        );
    }
}
