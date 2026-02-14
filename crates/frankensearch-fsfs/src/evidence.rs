//! Evidence-ledger taxonomy and trace-link model for fsfs runtime decisions.
//!
//! This module defines the canonical evidence events and linkage IDs emitted by
//! fsfs during discovery, indexing, search, pressure management, and privacy
//! enforcement. It extends the core [`EvidenceRecord`](frankensearch_core::EvidenceRecord)
//! and [`ReasonCode`](frankensearch_core::ReasonCode) types with fsfs-specific
//! event families.
//!
//! # Trace-Link Model
//!
//! Every fsfs operation produces a chain of evidence events linked by correlation
//! IDs. The [`TraceLink`] struct captures the causal relationships:
//!
//! ```text
//! trace_id ─────────────────────────────────────────────────┐
//!   ├── event_id: ULID (unique per event)                   │
//!   ├── parent_event_id: Option<ULID> (direct parent)       │
//!   ├── claim_id: Option<String> (worker claim on a job)    │
//!   └── policy_id: Option<String> (policy rule that fired)  │
//!                                                           │
//! trace_id is the root_request_id from the telemetry        │
//! contract — it ties all events from a single user          │
//! action (query, index rebuild, config reload) together.    │
//! ───────────────────────────────────────────────────────────┘
//! ```
//!
//! # Event Families
//!
//! | Family      | Namespace prefix   | Emitted by                       |
//! |-------------|-------------------|----------------------------------|
//! | Discovery   | `discovery.*`     | File scanner, scope resolver     |
//! | Ingest      | `ingest.*`        | Document pipeline, dedup, queue  |
//! | Query       | `query.*`         | Search orchestrator, fusion      |
//! | Degrade     | `degrade.*`       | Pressure controller, circuit     |
//! | Override    | `override.*`      | Operator config change, hot reload|
//! | Privacy     | `privacy.*`       | Redaction engine, scope policy   |
//! | Durability  | `durability.*`    | `RaptorQ` repair, integrity check  |
//! | Lifecycle   | `lifecycle.*`     | Startup, shutdown, health checks |
//!
//! # Validation
//!
//! All reason codes conform to the `^[a-z0-9]+\.[a-z0-9_]+\.[a-z0-9_]+$`
//! pattern enforced by [`ReasonCode::is_valid`](frankensearch_core::ReasonCode::is_valid).
//! The [`FsfsEvidenceEvent`] enum serializes to the evidence JSONL envelope
//! defined in `schemas/evidence-jsonl-v1.schema.json`.

use std::fmt;

use frankensearch_core::{EvidenceRecord, ReasonCode};
use serde::{Deserialize, Serialize};

// ─── Trace-Link Model ───────────────────────────────────────────────────────

/// Causal linkage IDs for evidence event chains.
///
/// Every fsfs evidence event carries a `TraceLink` that connects it to its
/// causal context. This enables offline replay, dependency tracking, and
/// postmortem analysis.
///
/// # ID Semantics
///
/// - `trace_id`: Correlates all events from a single user action. Matches
///   the `root_request_id` in the telemetry contract. Format: ULID.
/// - `event_id`: Unique per event. Format: ULID.
/// - `parent_event_id`: Points to the direct causal predecessor. `None` for
///   root events (e.g., the initial search request or startup).
/// - `claim_id`: Present when an event relates to a worker claim on a job
///   (embedding queue visibility timeout pattern). Format: opaque string.
/// - `policy_id`: Present when a privacy/scope/pressure policy rule fired.
///   Format: `{policy_kind}:{rule_name}` (e.g., `scope:hard_deny_ssh`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLink {
    /// Root trace ID (ULID) tying all events from one user action.
    pub trace_id: String,
    /// Unique event ID (ULID).
    pub event_id: String,
    /// Direct causal parent, if any.
    pub parent_event_id: Option<String>,
    /// Worker claim ID for embedding queue events.
    pub claim_id: Option<String>,
    /// Policy rule that triggered this event.
    pub policy_id: Option<String>,
}

impl TraceLink {
    /// Create a root trace link (no parent, no claim, no policy).
    #[must_use]
    pub fn root(trace_id: impl Into<String>, event_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            event_id: event_id.into(),
            parent_event_id: None,
            claim_id: None,
            policy_id: None,
        }
    }

    /// Create a child trace link under an existing parent.
    #[must_use]
    pub fn child(
        trace_id: impl Into<String>,
        event_id: impl Into<String>,
        parent_event_id: impl Into<String>,
    ) -> Self {
        Self {
            trace_id: trace_id.into(),
            event_id: event_id.into(),
            parent_event_id: Some(parent_event_id.into()),
            claim_id: None,
            policy_id: None,
        }
    }

    /// Attach a worker claim ID.
    #[must_use]
    pub fn with_claim(mut self, claim_id: impl Into<String>) -> Self {
        self.claim_id = Some(claim_id.into());
        self
    }

    /// Attach a policy rule ID.
    #[must_use]
    pub fn with_policy(mut self, policy_id: impl Into<String>) -> Self {
        self.policy_id = Some(policy_id.into());
        self
    }
}

// ─── fsfs Event Families ────────────────────────────────────────────────────

/// Top-level event family classification for fsfs evidence events.
///
/// This extends the core `EvidenceEventType` with fsfs-specific operational
/// categories that map to distinct runtime subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsfsEventFamily {
    /// File discovery and scope resolution.
    Discovery,
    /// Document ingestion, canonicalization, dedup, embedding queue.
    Ingest,
    /// Search orchestration, fusion, result delivery.
    Query,
    /// Pressure management, resource constraints, degradation.
    Degrade,
    /// Operator-initiated config changes, hot reloads.
    Override,
    /// Privacy enforcement, redaction, scope denial.
    Privacy,
    /// `RaptorQ` integrity checks, repair, sidecar management.
    Durability,
    /// Process lifecycle: startup, shutdown, health transitions.
    Lifecycle,
}

impl fmt::Display for FsfsEventFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Discovery => write!(f, "discovery"),
            Self::Ingest => write!(f, "ingest"),
            Self::Query => write!(f, "query"),
            Self::Degrade => write!(f, "degrade"),
            Self::Override => write!(f, "override"),
            Self::Privacy => write!(f, "privacy"),
            Self::Durability => write!(f, "durability"),
            Self::Lifecycle => write!(f, "lifecycle"),
        }
    }
}

// ─── fsfs Reason Codes ──────────────────────────────────────────────────────

/// fsfs-specific reason codes organized by event family.
///
/// These extend the core `ReasonCode` constants with fsfs runtime decision
/// codes. All codes follow the `namespace.subject.detail` pattern.
pub struct FsfsReasonCode;

impl FsfsReasonCode {
    // ── Discovery ───────────────────────────────────────────────────────

    /// A root directory was resolved and accepted for scanning.
    pub const DISCOVERY_ROOT_ACCEPTED: &str = "discovery.root.accepted";
    /// A root directory was rejected (does not exist or not accessible).
    pub const DISCOVERY_ROOT_REJECTED: &str = "discovery.root.rejected";
    /// A file was included in the discovery scope.
    pub const DISCOVERY_FILE_INCLUDED: &str = "discovery.file.included";
    /// A file was excluded by pattern match.
    pub const DISCOVERY_FILE_EXCLUDED: &str = "discovery.file.excluded_pattern";
    /// A file was excluded because it exceeds the size limit.
    pub const DISCOVERY_FILE_TOO_LARGE: &str = "discovery.file.too_large";
    /// A file was excluded because it matches the binary blocklist.
    pub const DISCOVERY_FILE_BINARY: &str = "discovery.file.binary_blocked";
    /// A symlink was skipped because `follow_symlinks` is disabled.
    pub const DISCOVERY_SYMLINK_SKIPPED: &str = "discovery.symlink.skipped";
    /// Discovery scan completed for a root.
    pub const DISCOVERY_SCAN_COMPLETE: &str = "discovery.scan.complete";
    /// Discovery scan encountered an I/O error on a path.
    pub const DISCOVERY_SCAN_ERROR: &str = "discovery.scan.io_error";
    /// Watch mode detected a filesystem change.
    pub const DISCOVERY_WATCH_CHANGE: &str = "discovery.watch.change_detected";

    // ── Ingest ──────────────────────────────────────────────────────────

    /// A document was enqueued for embedding.
    pub const INGEST_DOC_ENQUEUED: &str = "ingest.doc.enqueued";
    /// A document was deduplicated (same content hash already pending).
    pub const INGEST_DOC_DEDUPLICATED: &str = "ingest.doc.deduplicated";
    /// A document was skipped (low-signal content after canonicalization).
    pub const INGEST_DOC_LOW_SIGNAL: &str = "ingest.doc.low_signal";
    /// An embedding job was claimed by a worker.
    pub const INGEST_JOB_CLAIMED: &str = "ingest.job.claimed";
    /// An embedding job completed successfully.
    pub const INGEST_JOB_COMPLETED: &str = "ingest.job.completed";
    /// An embedding job failed and will be retried.
    pub const INGEST_JOB_FAILED_RETRY: &str = "ingest.job.failed_retry";
    /// An embedding job failed permanently (max retries exceeded).
    pub const INGEST_JOB_FAILED_TERMINAL: &str = "ingest.job.failed_terminal";
    /// A stale job was reclaimed after visibility timeout.
    pub const INGEST_JOB_RECLAIMED: &str = "ingest.job.reclaimed";
    /// Backpressure threshold was reached on the embedding queue.
    pub const INGEST_BACKPRESSURE_HIT: &str = "ingest.backpressure.threshold_hit";
    /// Backpressure cleared (queue depth fell below threshold).
    pub const INGEST_BACKPRESSURE_CLEARED: &str = "ingest.backpressure.cleared";
    /// A batch of embeddings was committed to the vector index.
    pub const INGEST_BATCH_COMMITTED: &str = "ingest.batch.committed";
    /// Index rebuild was triggered by staleness detection.
    pub const INGEST_REBUILD_TRIGGERED: &str = "ingest.rebuild.triggered";

    // ── Query ───────────────────────────────────────────────────────────

    /// Search query received and canonicalized.
    pub const QUERY_RECEIVED: &str = "query.search.received";
    /// Query was classified (`empty`/`identifier`/`short_keyword`/`natural_language`).
    pub const QUERY_CLASSIFIED: &str = "query.search.classified";
    /// Lexical (BM25) search phase completed.
    pub const QUERY_LEXICAL_DONE: &str = "query.lexical.completed";
    /// Semantic (vector) search phase completed.
    pub const QUERY_SEMANTIC_DONE: &str = "query.semantic.completed";
    /// RRF fusion produced merged results.
    pub const QUERY_FUSION_DONE: &str = "query.fusion.completed";
    /// Quality refinement (reranking) completed.
    pub const QUERY_RERANK_DONE: &str = "query.rerank.completed";
    /// Quality refinement was skipped (fast-only mode or circuit open).
    pub const QUERY_RERANK_SKIPPED: &str = "query.rerank.skipped";
    /// Quality refinement timed out.
    pub const QUERY_RERANK_TIMEOUT: &str = "query.rerank.timeout";
    /// Explain payload was attached to search results.
    pub const QUERY_EXPLAIN_ATTACHED: &str = "query.explain.attached";
    /// Search results were delivered to the caller.
    pub const QUERY_RESULTS_DELIVERED: &str = "query.results.delivered";

    // ── Degrade ─────────────────────────────────────────────────────────

    /// Pressure profile transitioned (e.g., performance → degraded).
    pub const DEGRADE_PROFILE_CHANGED: &str = "degrade.profile.changed";
    /// CPU ceiling was breached.
    pub const DEGRADE_CPU_CEILING_HIT: &str = "degrade.cpu.ceiling_hit";
    /// Memory ceiling was breached.
    pub const DEGRADE_MEMORY_CEILING_HIT: &str = "degrade.memory.ceiling_hit";
    /// I/O ceiling was breached.
    pub const DEGRADE_IO_CEILING_HIT: &str = "degrade.io.ceiling_hit";
    /// Load average ceiling was breached.
    pub const DEGRADE_LOAD_CEILING_HIT: &str = "degrade.load.ceiling_hit";
    /// Quality tier was disabled due to resource pressure.
    pub const DEGRADE_QUALITY_DISABLED: &str = "degrade.quality.disabled";
    /// Quality tier was re-enabled after pressure subsided.
    pub const DEGRADE_QUALITY_RESTORED: &str = "degrade.quality.restored";
    /// Embedding batch size was reduced due to pressure.
    pub const DEGRADE_BATCH_REDUCED: &str = "degrade.batch.reduced";
    /// Embedding batch size was restored to normal.
    pub const DEGRADE_BATCH_RESTORED: &str = "degrade.batch.restored";
    /// State transition was held by hysteresis/anti-flap guard.
    pub const DEGRADE_TRANSITION_HELD: &str = "degrade.transition.hysteresis_hold";
    /// Controller recovered to a lower-pressure state.
    pub const DEGRADE_TRANSITION_RECOVERED: &str = "degrade.transition.recovered";
    /// Index footprint is healthy relative to configured disk budget.
    pub const DEGRADE_DISK_WITHIN_BUDGET: &str = "degrade.disk.within_budget";
    /// Index footprint approached disk budget threshold.
    pub const DEGRADE_DISK_NEAR_BUDGET: &str = "degrade.disk.near_budget";
    /// Index footprint exceeded configured disk budget.
    pub const DEGRADE_DISK_OVER_BUDGET: &str = "degrade.disk.over_budget";
    /// Index footprint entered critical disk-budget region.
    pub const DEGRADE_DISK_CRITICAL: &str = "degrade.disk.critical";

    // ── Override ────────────────────────────────────────────────────────

    /// Configuration was reloaded from file.
    pub const OVERRIDE_CONFIG_RELOADED: &str = "override.config.reloaded";
    /// A config field was changed at runtime.
    pub const OVERRIDE_FIELD_CHANGED: &str = "override.field.changed";
    /// A config validation warning was emitted.
    pub const OVERRIDE_VALIDATION_WARN: &str = "override.validation.warning";
    /// A config reload was rejected (validation failure).
    pub const OVERRIDE_RELOAD_REJECTED: &str = "override.reload.rejected";
    /// Search `fast_only` was toggled at runtime.
    pub const OVERRIDE_FAST_ONLY_TOGGLED: &str = "override.fast_only.toggled";
    /// Pressure profile was changed by operator.
    pub const OVERRIDE_PROFILE_SET: &str = "override.profile.set";
    /// Privacy redaction settings were changed.
    pub const OVERRIDE_PRIVACY_CHANGED: &str = "override.privacy.changed";

    // ── Privacy ─────────────────────────────────────────────────────────

    /// A path was denied by the hard-deny scope rule.
    pub const PRIVACY_SCOPE_HARD_DENY: &str = "privacy.scope.hard_deny";
    /// A path was denied by explicit opt-out.
    pub const PRIVACY_SCOPE_OPT_OUT: &str = "privacy.scope.opt_out";
    /// A path required opt-in and was excluded (no opt-in).
    pub const PRIVACY_SCOPE_REQUIRE_OPT_IN: &str = "privacy.scope.require_opt_in";
    /// File content was redacted in log/evidence output.
    pub const PRIVACY_REDACT_CONTENT: &str = "privacy.redact.content";
    /// File path was tokenized in telemetry output.
    pub const PRIVACY_REDACT_PATH: &str = "privacy.redact.path_tokenized";
    /// Query text was hashed/truncated in evidence output.
    pub const PRIVACY_REDACT_QUERY: &str = "privacy.redact.query";
    /// Redaction profile was applied to an evidence payload.
    pub const PRIVACY_REDACT_APPLIED: &str = "privacy.redact.applied";

    // ── Durability ──────────────────────────────────────────────────────

    /// Integrity verification passed (xxh3 fast path).
    pub const DURABILITY_VERIFY_INTACT: &str = "durability.verify.intact";
    /// Integrity verification detected corruption.
    pub const DURABILITY_VERIFY_CORRUPT: &str = "durability.verify.corruption_detected";
    /// Repair succeeded via `RaptorQ` decode.
    pub const DURABILITY_REPAIR_SUCCESS: &str = "durability.repair.success";
    /// Repair failed (insufficient symbols).
    pub const DURABILITY_REPAIR_FAILED: &str = "durability.repair.failed";
    /// FEC sidecar was generated for a file.
    pub const DURABILITY_PROTECT_DONE: &str = "durability.protect.completed";
    /// FEC sidecar was missing for a protected file.
    pub const DURABILITY_SIDECAR_MISSING: &str = "durability.sidecar.missing";

    // ── Lifecycle ───────────────────────────────────────────────────────

    /// fsfs process started.
    pub const LIFECYCLE_STARTED: &str = "lifecycle.process.started";
    /// fsfs process shutting down gracefully.
    pub const LIFECYCLE_SHUTDOWN: &str = "lifecycle.process.shutdown";
    /// Health check passed.
    pub const LIFECYCLE_HEALTH_OK: &str = "lifecycle.health.ok";
    /// Health check detected a degraded component.
    pub const LIFECYCLE_HEALTH_DEGRADED: &str = "lifecycle.health.degraded";
    /// Model download started.
    pub const LIFECYCLE_MODEL_DOWNLOAD: &str = "lifecycle.model.download_started";
    /// Model download completed.
    pub const LIFECYCLE_MODEL_READY: &str = "lifecycle.model.ready";
    /// Model download failed.
    pub const LIFECYCLE_MODEL_FAILED: &str = "lifecycle.model.download_failed";
}

// ─── Scope Decision ─────────────────────────────────────────────────────────

/// Outcome of a scope/privacy decision for a single path.
///
/// Every scope evaluation emits one of these, enabling full audit of what
/// fsfs included, excluded, and why. This matches the required decision
/// fields from `docs/fsfs-scope-privacy-contract.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeDecision {
    /// The path being evaluated (project-relative if redaction is on).
    pub path: String,
    /// The decision outcome.
    pub decision: ScopeDecisionKind,
    /// Machine-stable reason code explaining the decision.
    pub reason_code: String,
    /// Sensitive data classes detected in this path (if any).
    pub sensitive_classes: Vec<String>,
    /// Whether the content may be persisted to storage.
    pub persist_allowed: bool,
    /// Whether the content may be emitted in telemetry/evidence.
    pub emit_allowed: bool,
    /// Whether the content may be displayed in the TUI.
    pub display_allowed: bool,
    /// Redaction profile applied.
    pub redaction_profile: String,
}

/// Possible scope decision outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeDecisionKind {
    /// Path is included in the discovery scope.
    Include,
    /// Path is excluded from the discovery scope.
    Exclude,
    /// Path requires explicit opt-in to be included.
    RequireOptIn,
}

impl fmt::Display for ScopeDecisionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Include => write!(f, "include"),
            Self::Exclude => write!(f, "exclude"),
            Self::RequireOptIn => write!(f, "require_opt_in"),
        }
    }
}

// ─── Fsfs Evidence Event ────────────────────────────────────────────────────

/// Complete fsfs evidence event combining trace link, family, and payload.
///
/// This wraps a core `EvidenceRecord` with fsfs-specific context (trace link,
/// event family, optional scope decision). It serializes to the evidence JSONL
/// envelope and supports deterministic replay via the trace link chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsfsEvidenceEvent {
    /// Causal linkage IDs.
    pub trace: TraceLink,
    /// Event family classification.
    pub family: FsfsEventFamily,
    /// Core evidence record (reason code, severity, pipeline state, etc.).
    pub record: EvidenceRecord,
    /// Scope decision details (present for discovery/privacy events).
    pub scope_decision: Option<ScopeDecision>,
    /// RFC 3339 UTC timestamp.
    pub timestamp: String,
}

impl FsfsEvidenceEvent {
    /// Create an fsfs evidence event.
    #[must_use]
    pub fn new(
        trace: TraceLink,
        family: FsfsEventFamily,
        record: EvidenceRecord,
        timestamp: impl Into<String>,
    ) -> Self {
        Self {
            trace,
            family,
            record,
            scope_decision: None,
            timestamp: timestamp.into(),
        }
    }

    /// Attach a scope decision to this event.
    #[must_use]
    pub fn with_scope_decision(mut self, decision: ScopeDecision) -> Self {
        self.scope_decision = Some(decision);
        self
    }
}

// ─── Validation ─────────────────────────────────────────────────────────────

/// Validate that an fsfs reason code follows the canonical pattern.
///
/// Returns `true` if the code matches `^[a-z0-9]+\.[a-z0-9_]+\.[a-z0-9_]+$`
/// and belongs to a recognized fsfs namespace.
#[must_use]
pub fn is_valid_fsfs_reason_code(code: &str) -> bool {
    let rc = ReasonCode::new(code);
    if !rc.is_valid() {
        return false;
    }

    let namespace = code.split('.').next().unwrap_or("");
    matches!(
        namespace,
        "discovery"
            | "ingest"
            | "query"
            | "degrade"
            | "override"
            | "privacy"
            | "durability"
            | "lifecycle"
    )
}

/// All fsfs reason code constants for validation and enumeration.
pub const ALL_FSFS_REASON_CODES: &[&str] = &[
    // Discovery
    FsfsReasonCode::DISCOVERY_ROOT_ACCEPTED,
    FsfsReasonCode::DISCOVERY_ROOT_REJECTED,
    FsfsReasonCode::DISCOVERY_FILE_INCLUDED,
    FsfsReasonCode::DISCOVERY_FILE_EXCLUDED,
    FsfsReasonCode::DISCOVERY_FILE_TOO_LARGE,
    FsfsReasonCode::DISCOVERY_FILE_BINARY,
    FsfsReasonCode::DISCOVERY_SYMLINK_SKIPPED,
    FsfsReasonCode::DISCOVERY_SCAN_COMPLETE,
    FsfsReasonCode::DISCOVERY_SCAN_ERROR,
    FsfsReasonCode::DISCOVERY_WATCH_CHANGE,
    // Ingest
    FsfsReasonCode::INGEST_DOC_ENQUEUED,
    FsfsReasonCode::INGEST_DOC_DEDUPLICATED,
    FsfsReasonCode::INGEST_DOC_LOW_SIGNAL,
    FsfsReasonCode::INGEST_JOB_CLAIMED,
    FsfsReasonCode::INGEST_JOB_COMPLETED,
    FsfsReasonCode::INGEST_JOB_FAILED_RETRY,
    FsfsReasonCode::INGEST_JOB_FAILED_TERMINAL,
    FsfsReasonCode::INGEST_JOB_RECLAIMED,
    FsfsReasonCode::INGEST_BACKPRESSURE_HIT,
    FsfsReasonCode::INGEST_BACKPRESSURE_CLEARED,
    FsfsReasonCode::INGEST_BATCH_COMMITTED,
    FsfsReasonCode::INGEST_REBUILD_TRIGGERED,
    // Query
    FsfsReasonCode::QUERY_RECEIVED,
    FsfsReasonCode::QUERY_CLASSIFIED,
    FsfsReasonCode::QUERY_LEXICAL_DONE,
    FsfsReasonCode::QUERY_SEMANTIC_DONE,
    FsfsReasonCode::QUERY_FUSION_DONE,
    FsfsReasonCode::QUERY_RERANK_DONE,
    FsfsReasonCode::QUERY_RERANK_SKIPPED,
    FsfsReasonCode::QUERY_RERANK_TIMEOUT,
    FsfsReasonCode::QUERY_EXPLAIN_ATTACHED,
    FsfsReasonCode::QUERY_RESULTS_DELIVERED,
    // Degrade
    FsfsReasonCode::DEGRADE_PROFILE_CHANGED,
    FsfsReasonCode::DEGRADE_CPU_CEILING_HIT,
    FsfsReasonCode::DEGRADE_MEMORY_CEILING_HIT,
    FsfsReasonCode::DEGRADE_IO_CEILING_HIT,
    FsfsReasonCode::DEGRADE_LOAD_CEILING_HIT,
    FsfsReasonCode::DEGRADE_QUALITY_DISABLED,
    FsfsReasonCode::DEGRADE_QUALITY_RESTORED,
    FsfsReasonCode::DEGRADE_BATCH_REDUCED,
    FsfsReasonCode::DEGRADE_BATCH_RESTORED,
    FsfsReasonCode::DEGRADE_TRANSITION_HELD,
    FsfsReasonCode::DEGRADE_TRANSITION_RECOVERED,
    FsfsReasonCode::DEGRADE_DISK_WITHIN_BUDGET,
    FsfsReasonCode::DEGRADE_DISK_NEAR_BUDGET,
    FsfsReasonCode::DEGRADE_DISK_OVER_BUDGET,
    FsfsReasonCode::DEGRADE_DISK_CRITICAL,
    // Override
    FsfsReasonCode::OVERRIDE_CONFIG_RELOADED,
    FsfsReasonCode::OVERRIDE_FIELD_CHANGED,
    FsfsReasonCode::OVERRIDE_VALIDATION_WARN,
    FsfsReasonCode::OVERRIDE_RELOAD_REJECTED,
    FsfsReasonCode::OVERRIDE_FAST_ONLY_TOGGLED,
    FsfsReasonCode::OVERRIDE_PROFILE_SET,
    FsfsReasonCode::OVERRIDE_PRIVACY_CHANGED,
    // Privacy
    FsfsReasonCode::PRIVACY_SCOPE_HARD_DENY,
    FsfsReasonCode::PRIVACY_SCOPE_OPT_OUT,
    FsfsReasonCode::PRIVACY_SCOPE_REQUIRE_OPT_IN,
    FsfsReasonCode::PRIVACY_REDACT_CONTENT,
    FsfsReasonCode::PRIVACY_REDACT_PATH,
    FsfsReasonCode::PRIVACY_REDACT_QUERY,
    FsfsReasonCode::PRIVACY_REDACT_APPLIED,
    // Durability
    FsfsReasonCode::DURABILITY_VERIFY_INTACT,
    FsfsReasonCode::DURABILITY_VERIFY_CORRUPT,
    FsfsReasonCode::DURABILITY_REPAIR_SUCCESS,
    FsfsReasonCode::DURABILITY_REPAIR_FAILED,
    FsfsReasonCode::DURABILITY_PROTECT_DONE,
    FsfsReasonCode::DURABILITY_SIDECAR_MISSING,
    // Lifecycle
    FsfsReasonCode::LIFECYCLE_STARTED,
    FsfsReasonCode::LIFECYCLE_SHUTDOWN,
    FsfsReasonCode::LIFECYCLE_HEALTH_OK,
    FsfsReasonCode::LIFECYCLE_HEALTH_DEGRADED,
    FsfsReasonCode::LIFECYCLE_MODEL_DOWNLOAD,
    FsfsReasonCode::LIFECYCLE_MODEL_READY,
    FsfsReasonCode::LIFECYCLE_MODEL_FAILED,
];

// ─── Schema Validation ─────────────────────────────────────────────────────

/// Validation result for a producer/consumer conformance check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the event passed all validation rules.
    pub valid: bool,
    /// List of validation violations (empty if valid).
    pub violations: Vec<ValidationViolation>,
}

/// A single schema validation violation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// The field that failed validation.
    pub field: String,
    /// Machine-stable violation code.
    pub code: String,
    /// Human-readable description.
    pub message: String,
}

/// Validate an `FsfsEvidenceEvent` against the schema contract.
///
/// Checks:
/// 1. Reason code follows `namespace.subject.detail` pattern.
/// 2. Reason code namespace matches the event family.
/// 3. Trace link has non-empty `trace_id` and `event_id`.
/// 4. Timestamp is non-empty.
/// 5. Scope decision, if present, has a valid reason code.
#[must_use]
pub fn validate_event(event: &FsfsEvidenceEvent) -> ValidationResult {
    let mut violations = Vec::new();

    // 1. Reason code format
    if !event.record.reason_code.is_valid() {
        violations.push(ValidationViolation {
            field: "record.reason_code".into(),
            code: "evidence.reason_code.invalid_format".into(),
            message: format!(
                "reason code '{}' does not match namespace.subject.detail pattern",
                event.record.reason_code
            ),
        });
    }

    // 2. Namespace matches family
    let expected_namespace = event.family.to_string();
    let actual_namespace = event
        .record
        .reason_code
        .as_str()
        .split('.')
        .next()
        .unwrap_or("");
    if actual_namespace != expected_namespace {
        violations.push(ValidationViolation {
            field: "record.reason_code".into(),
            code: "evidence.reason_code.namespace_mismatch".into(),
            message: format!(
                "reason code namespace '{actual_namespace}' does not match event family '{expected_namespace}'"
            ),
        });
    }

    // 3. Trace link IDs
    if event.trace.trace_id.is_empty() {
        violations.push(ValidationViolation {
            field: "trace.trace_id".into(),
            code: "evidence.trace.missing_trace_id".into(),
            message: "trace_id must be non-empty".into(),
        });
    }
    if event.trace.event_id.is_empty() {
        violations.push(ValidationViolation {
            field: "trace.event_id".into(),
            code: "evidence.trace.missing_event_id".into(),
            message: "event_id must be non-empty".into(),
        });
    }

    // 4. Timestamp
    if event.timestamp.is_empty() {
        violations.push(ValidationViolation {
            field: "timestamp".into(),
            code: "evidence.timestamp.empty".into(),
            message: "timestamp must be non-empty RFC 3339".into(),
        });
    }

    // 5. Scope decision reason code
    if let Some(scope) = &event.scope_decision {
        let scope_rc = ReasonCode::new(&scope.reason_code);
        if !scope_rc.is_valid() {
            violations.push(ValidationViolation {
                field: "scope_decision.reason_code".into(),
                code: "evidence.scope.invalid_reason_code".into(),
                message: format!(
                    "scope decision reason code '{}' is not valid",
                    scope.reason_code
                ),
            });
        }
    }

    ValidationResult {
        valid: violations.is_empty(),
        violations,
    }
}

#[cfg(test)]
mod tests {
    use frankensearch_core::{EvidenceEventType, PipelineState, Severity};

    use super::*;

    // ─── Reason Code Validation ─────────────────────────────────────────

    #[test]
    fn all_fsfs_reason_codes_are_valid() {
        for &code in ALL_FSFS_REASON_CODES {
            let rc = ReasonCode::new(code);
            assert!(rc.is_valid(), "invalid fsfs reason code: {code}");
        }
    }

    #[test]
    fn all_fsfs_reason_codes_belong_to_recognized_namespace() {
        for &code in ALL_FSFS_REASON_CODES {
            assert!(
                is_valid_fsfs_reason_code(code),
                "unrecognized namespace in: {code}"
            );
        }
    }

    #[test]
    fn fsfs_reason_codes_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for &code in ALL_FSFS_REASON_CODES {
            assert!(seen.insert(code), "duplicate fsfs reason code: {code}");
        }
    }

    #[test]
    fn core_reason_codes_are_not_fsfs_namespace() {
        assert!(!is_valid_fsfs_reason_code(
            ReasonCode::DECISION_REFINE_NOMINAL
        ));
        assert!(!is_valid_fsfs_reason_code(
            ReasonCode::CIRCUIT_OPEN_FAILURES
        ));
        assert!(!is_valid_fsfs_reason_code(
            ReasonCode::FUSION_BLEND_ADJUSTED
        ));
    }

    #[test]
    fn invalid_reason_code_rejected() {
        assert!(!is_valid_fsfs_reason_code("bad"));
        assert!(!is_valid_fsfs_reason_code("a.b"));
        assert!(!is_valid_fsfs_reason_code("a.b.c.d"));
        assert!(!is_valid_fsfs_reason_code(""));
    }

    // ─── TraceLink ──────────────────────────────────────────────────────

    #[test]
    fn trace_link_root() {
        let link = TraceLink::root("trace-001", "event-001");
        assert_eq!(link.trace_id, "trace-001");
        assert_eq!(link.event_id, "event-001");
        assert!(link.parent_event_id.is_none());
        assert!(link.claim_id.is_none());
        assert!(link.policy_id.is_none());
    }

    #[test]
    fn trace_link_child() {
        let link = TraceLink::child("trace-001", "event-002", "event-001");
        assert_eq!(link.parent_event_id, Some("event-001".into()));
    }

    #[test]
    fn trace_link_with_claim_and_policy() {
        let link = TraceLink::root("t1", "e1")
            .with_claim("worker-42-batch-7")
            .with_policy("scope:hard_deny_ssh");
        assert_eq!(link.claim_id, Some("worker-42-batch-7".into()));
        assert_eq!(link.policy_id, Some("scope:hard_deny_ssh".into()));
    }

    #[test]
    fn trace_link_serde_roundtrip() {
        let link = TraceLink::child("t1", "e2", "e1")
            .with_claim("c1")
            .with_policy("p1");
        let json = serde_json::to_string(&link).unwrap();
        let decoded: TraceLink = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, link);
    }

    // ─── FsfsEventFamily ────────────────────────────────────────────────

    #[test]
    fn event_family_display() {
        assert_eq!(FsfsEventFamily::Discovery.to_string(), "discovery");
        assert_eq!(FsfsEventFamily::Ingest.to_string(), "ingest");
        assert_eq!(FsfsEventFamily::Query.to_string(), "query");
        assert_eq!(FsfsEventFamily::Degrade.to_string(), "degrade");
        assert_eq!(FsfsEventFamily::Override.to_string(), "override");
        assert_eq!(FsfsEventFamily::Privacy.to_string(), "privacy");
        assert_eq!(FsfsEventFamily::Durability.to_string(), "durability");
        assert_eq!(FsfsEventFamily::Lifecycle.to_string(), "lifecycle");
    }

    #[test]
    fn event_family_serde_roundtrip() {
        for family in [
            FsfsEventFamily::Discovery,
            FsfsEventFamily::Ingest,
            FsfsEventFamily::Query,
            FsfsEventFamily::Degrade,
            FsfsEventFamily::Override,
            FsfsEventFamily::Privacy,
            FsfsEventFamily::Durability,
            FsfsEventFamily::Lifecycle,
        ] {
            let json = serde_json::to_string(&family).unwrap();
            let decoded: FsfsEventFamily = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, family);
        }
    }

    // ─── ScopeDecision ──────────────────────────────────────────────────

    #[test]
    fn scope_decision_serde_roundtrip() {
        let decision = ScopeDecision {
            path: "src/main.rs".into(),
            decision: ScopeDecisionKind::Include,
            reason_code: FsfsReasonCode::DISCOVERY_FILE_INCLUDED.into(),
            sensitive_classes: vec![],
            persist_allowed: true,
            emit_allowed: true,
            display_allowed: true,
            redaction_profile: "v1".into(),
        };
        let json = serde_json::to_string(&decision).unwrap();
        let decoded: ScopeDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, decision);
    }

    #[test]
    fn scope_decision_with_sensitive_classes() {
        let decision = ScopeDecision {
            path: "~/.ssh/id_rsa".into(),
            decision: ScopeDecisionKind::Exclude,
            reason_code: FsfsReasonCode::PRIVACY_SCOPE_HARD_DENY.into(),
            sensitive_classes: vec!["ssh_key".into(), "credential".into()],
            persist_allowed: false,
            emit_allowed: false,
            display_allowed: false,
            redaction_profile: "v1".into(),
        };
        assert_eq!(decision.sensitive_classes.len(), 2);
        assert!(!decision.persist_allowed);
    }

    // ─── ScopeDecisionKind ──────────────────────────────────────────────

    #[test]
    fn scope_decision_kind_display() {
        assert_eq!(ScopeDecisionKind::Include.to_string(), "include");
        assert_eq!(ScopeDecisionKind::Exclude.to_string(), "exclude");
        assert_eq!(
            ScopeDecisionKind::RequireOptIn.to_string(),
            "require_opt_in"
        );
    }

    // ─── FsfsEvidenceEvent ──────────────────────────────────────────────

    fn sample_event(family: FsfsEventFamily, reason_code: &str) -> FsfsEvidenceEvent {
        FsfsEvidenceEvent::new(
            TraceLink::root("01JAH9A2W8F8Q6GQ4C7M3N2P1R", "01JAH9A2X1K2M3N4P5Q6R7S8T9"),
            family,
            EvidenceRecord::new(
                EvidenceEventType::Decision,
                reason_code,
                "test event",
                Severity::Info,
                PipelineState::Nominal,
                "test_component",
            ),
            "2026-02-14T00:00:00Z",
        )
    }

    #[test]
    fn evidence_event_construction() {
        let event = sample_event(
            FsfsEventFamily::Discovery,
            FsfsReasonCode::DISCOVERY_FILE_INCLUDED,
        );
        assert_eq!(event.family, FsfsEventFamily::Discovery);
        assert_eq!(event.record.reason_code.as_str(), "discovery.file.included");
        assert!(event.scope_decision.is_none());
    }

    #[test]
    fn evidence_event_with_scope_decision() {
        let event = sample_event(
            FsfsEventFamily::Privacy,
            FsfsReasonCode::PRIVACY_SCOPE_HARD_DENY,
        )
        .with_scope_decision(ScopeDecision {
            path: "~/.ssh/id_rsa".into(),
            decision: ScopeDecisionKind::Exclude,
            reason_code: FsfsReasonCode::PRIVACY_SCOPE_HARD_DENY.into(),
            sensitive_classes: vec!["ssh_key".into()],
            persist_allowed: false,
            emit_allowed: false,
            display_allowed: false,
            redaction_profile: "v1".into(),
        });
        assert!(event.scope_decision.is_some());
    }

    #[test]
    fn evidence_event_serde_roundtrip() {
        let event = sample_event(FsfsEventFamily::Ingest, FsfsReasonCode::INGEST_DOC_ENQUEUED);
        let json = serde_json::to_string(&event).unwrap();
        let decoded: FsfsEvidenceEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.family, FsfsEventFamily::Ingest);
        assert_eq!(decoded.record.reason_code.as_str(), "ingest.doc.enqueued");
        assert_eq!(decoded.timestamp, "2026-02-14T00:00:00Z");
    }

    // ─── Validation ─────────────────────────────────────────────────────

    #[test]
    fn valid_event_passes_validation() {
        let event = sample_event(FsfsEventFamily::Query, FsfsReasonCode::QUERY_RECEIVED);
        let result = validate_event(&event);
        assert!(result.valid, "violations: {:?}", result.violations);
    }

    #[test]
    fn invalid_reason_code_format_fails_validation() {
        let event = sample_event(FsfsEventFamily::Query, "bad-code");
        let result = validate_event(&event);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "evidence.reason_code.invalid_format")
        );
    }

    #[test]
    fn namespace_mismatch_fails_validation() {
        let event = sample_event(
            FsfsEventFamily::Query,
            FsfsReasonCode::DISCOVERY_FILE_INCLUDED,
        );
        let result = validate_event(&event);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "evidence.reason_code.namespace_mismatch")
        );
    }

    #[test]
    fn empty_trace_id_fails_validation() {
        let mut event = sample_event(
            FsfsEventFamily::Lifecycle,
            FsfsReasonCode::LIFECYCLE_STARTED,
        );
        event.trace.trace_id = String::new();
        let result = validate_event(&event);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "evidence.trace.missing_trace_id")
        );
    }

    #[test]
    fn empty_event_id_fails_validation() {
        let mut event = sample_event(
            FsfsEventFamily::Lifecycle,
            FsfsReasonCode::LIFECYCLE_STARTED,
        );
        event.trace.event_id = String::new();
        let result = validate_event(&event);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "evidence.trace.missing_event_id")
        );
    }

    #[test]
    fn empty_timestamp_fails_validation() {
        let mut event = sample_event(
            FsfsEventFamily::Lifecycle,
            FsfsReasonCode::LIFECYCLE_STARTED,
        );
        event.timestamp = String::new();
        let result = validate_event(&event);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "evidence.timestamp.empty")
        );
    }

    #[test]
    fn invalid_scope_decision_reason_code_fails() {
        let event = sample_event(
            FsfsEventFamily::Privacy,
            FsfsReasonCode::PRIVACY_SCOPE_HARD_DENY,
        )
        .with_scope_decision(ScopeDecision {
            path: "test".into(),
            decision: ScopeDecisionKind::Exclude,
            reason_code: "not-valid".into(),
            sensitive_classes: vec![],
            persist_allowed: false,
            emit_allowed: false,
            display_allowed: false,
            redaction_profile: "v1".into(),
        });
        let result = validate_event(&event);
        assert!(!result.valid);
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.code == "evidence.scope.invalid_reason_code")
        );
    }

    #[test]
    fn validation_result_serde_roundtrip() {
        let result = ValidationResult {
            valid: false,
            violations: vec![ValidationViolation {
                field: "test".into(),
                code: "test.code.example".into(),
                message: "test message".into(),
            }],
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: ValidationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, result);
    }

    // ─── Event family coverage ──────────────────────────────────────────

    #[test]
    fn every_family_has_at_least_one_reason_code() {
        let families = [
            "discovery",
            "ingest",
            "query",
            "degrade",
            "override",
            "privacy",
            "durability",
            "lifecycle",
        ];
        for family in families {
            let count = ALL_FSFS_REASON_CODES
                .iter()
                .filter(|code| code.starts_with(family))
                .count();
            assert!(count > 0, "no reason codes found for family '{family}'");
        }
    }

    #[test]
    fn reason_code_count_matches_expectations() {
        // 10 discovery + 12 ingest + 10 query + 15 degrade + 7 override
        // + 7 privacy + 6 durability + 7 lifecycle = 74
        assert_eq!(ALL_FSFS_REASON_CODES.len(), 74);
    }
}
