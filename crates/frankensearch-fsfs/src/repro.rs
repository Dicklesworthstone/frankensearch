//! Reproducibility artifact pack: capture, schema, and retention.
//!
//! A "repro pack" is a self-contained archive of everything needed to
//! deterministically replay an fsfs session or incident. It ties together
//! evidence JSONL logs, configuration snapshots, model manifests, and
//! index checksums into a single, trace-linked artifact.
//!
//! # Artifact Set
//!
//! Each repro pack contains:
//!
//! | File                     | Description                                    |
//! |--------------------------|------------------------------------------------|
//! | `manifest.json`          | Pack metadata, schema version, capture context |
//! | `evidence.jsonl`         | Evidence events (redacted) for the trace        |
//! | `config.toml`            | Resolved fsfs configuration at capture time    |
//! | `env.json`               | Relevant environment variables (redacted)      |
//! | `model-manifest.json`    | Embedder model IDs, revisions, dimensions      |
//! | `index-checksums.json`   | xxh3 hashes of vector/lexical indices           |
//! | `provenance-attestation.json` | Build/runtime provenance + optional signature |
//! | `replay-meta.json`       | Seed, `tick_ms`, `frame_seq` range for replay   |
//!
//! # Capture Points
//!
//! Repro packs are captured at:
//! 1. **On demand**: Operator requests via CLI `fsfs repro capture`.
//! 2. **On incident**: When a critical SLO violation or unrecoverable error occurs.
//! 3. **On test failure**: E2E test harness captures repro for failing scenarios.
//!
//! # Retention Policy
//!
//! Artifacts follow a tiered retention schedule:
//! - **Hot** (0-7 days): Full pack stored locally, ready for replay.
//! - **Warm** (7-90 days): Compressed pack, evidence trimmed to decisions/alerts.
//! - **Cold** (90+ days): Manifest + checksums + provenance attestation.
//!
//! The retention tier is configurable via `storage.evidence_retention_days` and
//! `storage.summary_retention_days` in the fsfs config.

use serde::{Deserialize, Serialize};

/// Schema version for the repro pack manifest.
pub const REPRO_SCHEMA_VERSION: u8 = 1;

/// File names within a repro pack.
pub const MANIFEST_FILENAME: &str = "manifest.json";
pub const EVIDENCE_FILENAME: &str = "evidence.jsonl";
pub const CONFIG_FILENAME: &str = "config.toml";
pub const ENV_FILENAME: &str = "env.json";
pub const MODEL_MANIFEST_FILENAME: &str = "model-manifest.json";
pub const INDEX_CHECKSUMS_FILENAME: &str = "index-checksums.json";
pub const PROVENANCE_ATTESTATION_FILENAME: &str = "provenance-attestation.json";
pub const REPLAY_META_FILENAME: &str = "replay-meta.json";
pub const PROVENANCE_SCHEMA_VERSION: u8 = 1;

// ─── Capture Context ────────────────────────────────────────────────────────

/// What triggered the repro pack capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureReason {
    /// Operator manually requested capture.
    Manual,
    /// Critical SLO violation detected.
    SloViolation,
    /// Unrecoverable error (index corruption, model load failure).
    Error,
    /// E2E test failure.
    TestFailure,
    /// Benchmark baseline capture.
    Benchmark,
    /// Periodic scheduled capture.
    Scheduled,
}

impl std::fmt::Display for CaptureReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Manual => write!(f, "manual"),
            Self::SloViolation => write!(f, "slo_violation"),
            Self::Error => write!(f, "error"),
            Self::TestFailure => write!(f, "test_failure"),
            Self::Benchmark => write!(f, "benchmark"),
            Self::Scheduled => write!(f, "scheduled"),
        }
    }
}

// ─── Pack Manifest ──────────────────────────────────────────────────────────

/// Top-level manifest for a repro pack.
///
/// Stored as `manifest.json` in the pack root. Contains all metadata
/// needed to identify, correlate, and replay the captured session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproManifest {
    /// Schema version (currently 1).
    pub schema_version: u8,
    /// Unique pack ID (ULID).
    pub pack_id: String,
    /// Trace ID this pack correlates to (matches `TraceLink.trace_id`).
    pub trace_id: String,
    /// What triggered the capture.
    pub capture_reason: CaptureReason,
    /// RFC 3339 UTC timestamp of capture.
    pub captured_at: String,
    /// Instance identity at capture time.
    pub instance: ReproInstance,
    /// List of files in this pack with sizes and checksums.
    pub artifacts: Vec<ArtifactEntry>,
    /// Current retention tier.
    pub retention_tier: RetentionTier,
    /// When this pack expires from the current tier.
    pub expires_at: Option<String>,
    /// Redaction policy version applied to evidence.
    pub redaction_policy_version: String,
}

/// Instance identity snapshot for the repro manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproInstance {
    /// Instance ULID.
    pub instance_id: String,
    /// Project key (project root path, redacted if configured).
    pub project_key: String,
    /// Hostname.
    pub host_name: String,
    /// Process ID.
    pub pid: Option<u32>,
    /// fsfs version string.
    pub version: String,
    /// Rust toolchain version.
    pub rust_version: Option<String>,
}

/// An entry in the manifest's artifact list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEntry {
    /// Filename relative to the pack root.
    pub filename: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// `xxh3_64` hash of the file contents (hex-encoded).
    pub checksum_xxh3: String,
    /// MIME type hint.
    pub content_type: String,
    /// Whether this file is present in the current retention tier.
    pub present: bool,
}

// ─── Retention ──────────────────────────────────────────────────────────────

/// Retention tier for repro artifacts.
///
/// Packs transition through tiers based on age:
/// - Hot (0-7d): All files present, ready for immediate replay.
/// - Warm (7-90d): Evidence trimmed to decision/alert events only.
/// - Cold (90+d): Only manifest and checksums retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionTier {
    /// Full pack, ready for replay.
    Hot,
    /// Compressed, evidence trimmed.
    Warm,
    /// Manifest + checksums only.
    Cold,
}

impl std::fmt::Display for RetentionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot => write!(f, "hot"),
            Self::Warm => write!(f, "warm"),
            Self::Cold => write!(f, "cold"),
        }
    }
}

/// Retention policy configuration.
///
/// Derived from `StorageConfig.evidence_retention_days` and
/// `StorageConfig.summary_retention_days`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Days before transitioning from Hot to Warm.
    pub hot_days: u16,
    /// Days before transitioning from Warm to Cold.
    pub warm_days: u16,
    /// Days before deleting Cold packs entirely (0 = keep forever).
    pub cold_max_days: u16,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            hot_days: 7,
            warm_days: 90,
            cold_max_days: 0,
        }
    }
}

impl RetentionPolicy {
    /// Determine which tier a pack belongs to based on age in days.
    #[must_use]
    pub const fn tier_for_age(&self, age_days: u16) -> Option<RetentionTier> {
        if age_days <= self.hot_days {
            Some(RetentionTier::Hot)
        } else if age_days <= self.warm_days {
            Some(RetentionTier::Warm)
        } else if self.cold_max_days == 0 || age_days <= self.cold_max_days {
            Some(RetentionTier::Cold)
        } else {
            None // expired
        }
    }
}

// ─── Model Manifest ─────────────────────────────────────────────────────────

/// Embedder model snapshot for the repro pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSnapshot {
    /// Model identifier (e.g., "potion-multilingual-128M").
    pub model_id: String,
    /// Model commit/revision SHA.
    pub revision: String,
    /// Embedding dimension.
    pub dimension: usize,
    /// Which tier this model serves.
    pub tier: String,
    /// Where the model files are stored.
    pub model_dir: String,
}

/// Complete model manifest for the repro pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Models active at capture time.
    pub models: Vec<ModelSnapshot>,
}

// ─── Index Checksums ────────────────────────────────────────────────────────

/// Index checksum entry for the repro pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexChecksum {
    /// Index name (e.g., "vector.fast", "vector.quality", "lexical").
    pub index_name: String,
    /// Index type ("fsvi", "tantivy", "hnsw").
    pub index_type: String,
    /// `xxh3_64` hash of the index file (hex-encoded).
    pub checksum_xxh3: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Record count at capture time.
    pub record_count: u64,
    /// FEC sidecar checksum (if durability enabled).
    pub fec_checksum_xxh3: Option<String>,
}

/// Complete index checksums for the repro pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexChecksums {
    /// All index checksums at capture time.
    pub indices: Vec<IndexChecksum>,
}

// ─── Provenance Attestation ────────────────────────────────────────────────

/// Build/runtime provenance envelope used for startup trust checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceAttestation {
    /// Schema version for attestation payload.
    pub schema_version: u8,
    /// Stable attestation identifier (typically ULID).
    pub attestation_id: String,
    /// RFC 3339 UTC timestamp when attestation was generated.
    pub generated_at: String,
    /// Build-time provenance snapshot.
    pub build: BuildProvenance,
    /// Runtime provenance hashes consumed by startup verification.
    pub runtime: RuntimeProvenance,
    /// Optional per-artifact SHA-256 hashes.
    pub artifact_hashes: Vec<ArtifactHash>,
    /// Optional signature block for signed attestations.
    pub signature: Option<AttestationSignature>,
}

/// Build metadata captured in a provenance attestation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildProvenance {
    /// Source commit used for build.
    pub source_commit: String,
    /// Build profile (`debug`/`release`).
    pub build_profile: String,
    /// Rust compiler version string.
    pub rustc_version: String,
    /// Compilation target triple.
    pub target_triple: String,
}

/// Runtime hash anchors for startup verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeProvenance {
    /// Hash of fsfs runtime binary (`sha256:<hex>`).
    pub binary_hash_sha256: String,
    /// Hash of resolved runtime config (`sha256:<hex>`).
    pub config_hash_sha256: String,
    /// Hash of index manifest used for replay/bootstrap (`sha256:<hex>`).
    pub index_manifest_hash_sha256: String,
}

/// Optional per-artifact digest entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactHash {
    /// Artifact path relative to the repro pack root.
    pub path: String,
    /// SHA-256 digest (`sha256:<hex>`).
    pub sha256: String,
}

/// Signature metadata for signed attestations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationSignature {
    /// Signature algorithm identifier.
    pub algorithm: String,
    /// Signing key identifier.
    pub key_id: String,
    /// Base64-encoded detached signature payload.
    pub signature_b64: String,
}

/// Startup verification terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupVerificationStatus {
    /// All checks passed and startup may continue normally.
    Verified,
    /// Startup can continue, but with constrained behavior.
    Degraded,
    /// Startup must not continue.
    Failed,
}

/// Startup fallback action when provenance validation fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartupVerificationAction {
    Continue,
    ContinueWithAlert,
    EnterReadOnly,
    EnterSafeMode,
    AbortStartup,
}

impl StartupVerificationAction {
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Continue => 0,
            Self::ContinueWithAlert => 1,
            Self::EnterReadOnly => 2,
            Self::EnterSafeMode => 3,
            Self::AbortStartup => 4,
        }
    }
}

impl std::fmt::Display for StartupVerificationAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Continue => write!(f, "continue"),
            Self::ContinueWithAlert => write!(f, "continue_with_alert"),
            Self::EnterReadOnly => write!(f, "enter_read_only"),
            Self::EnterSafeMode => write!(f, "enter_safe_mode"),
            Self::AbortStartup => write!(f, "abort_startup"),
        }
    }
}

/// Alert severity emitted by startup verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationSeverity {
    Info,
    Warn,
    Error,
}

/// Structured startup verification alert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupVerificationAlert {
    pub reason_code: String,
    pub severity: VerificationSeverity,
    pub message: String,
}

/// Policy describing required attestation checks and fallback behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupVerificationPolicy {
    pub require_attestation: bool,
    pub require_signature: bool,
    pub on_attestation_missing: StartupVerificationAction,
    pub on_signature_missing: StartupVerificationAction,
    pub on_signature_invalid: StartupVerificationAction,
    pub on_hash_mismatch: StartupVerificationAction,
}

impl Default for StartupVerificationPolicy {
    fn default() -> Self {
        Self {
            require_attestation: true,
            require_signature: false,
            on_attestation_missing: StartupVerificationAction::EnterSafeMode,
            on_signature_missing: StartupVerificationAction::EnterSafeMode,
            on_signature_invalid: StartupVerificationAction::AbortStartup,
            on_hash_mismatch: StartupVerificationAction::AbortStartup,
        }
    }
}

/// Raw startup verification check results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct StartupVerificationReport {
    pub attestation_present: bool,
    pub attestation_parsed: bool,
    pub signature_present: bool,
    pub signature_valid: bool,
    pub binary_hash_match: bool,
    pub config_hash_match: bool,
    pub index_manifest_hash_match: bool,
}

impl StartupVerificationReport {
    #[must_use]
    pub const fn has_hash_mismatch(self) -> bool {
        !self.binary_hash_match || !self.config_hash_match || !self.index_manifest_hash_match
    }
}

/// Final startup verification decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartupVerificationOutcome {
    pub status: StartupVerificationStatus,
    pub action: StartupVerificationAction,
    pub alerts: Vec<StartupVerificationAlert>,
}

impl StartupVerificationOutcome {
    #[must_use]
    pub const fn permits_startup(&self) -> bool {
        !matches!(self.action, StartupVerificationAction::AbortStartup)
    }
}

pub const REASON_ATTESTATION_MISSING: &str = "provenance.startup.attestation_missing";
pub const REASON_SIGNATURE_MISSING: &str = "provenance.startup.signature_missing";
pub const REASON_SIGNATURE_INVALID: &str = "provenance.startup.signature_invalid";
pub const REASON_HASH_MISMATCH: &str = "provenance.startup.hash_mismatch";

const fn select_stricter_action(
    current: StartupVerificationAction,
    candidate: StartupVerificationAction,
) -> StartupVerificationAction {
    if candidate.rank() > current.rank() {
        candidate
    } else {
        current
    }
}

const fn severity_for_action(action: StartupVerificationAction) -> VerificationSeverity {
    match action {
        StartupVerificationAction::Continue => VerificationSeverity::Info,
        StartupVerificationAction::ContinueWithAlert => VerificationSeverity::Warn,
        StartupVerificationAction::EnterReadOnly
        | StartupVerificationAction::EnterSafeMode
        | StartupVerificationAction::AbortStartup => VerificationSeverity::Error,
    }
}

fn push_alert(
    alerts: &mut Vec<StartupVerificationAlert>,
    reason_code: &str,
    action: StartupVerificationAction,
    message: &str,
) {
    alerts.push(StartupVerificationAlert {
        reason_code: reason_code.to_owned(),
        severity: severity_for_action(action),
        message: message.to_owned(),
    });
}

/// Evaluate startup provenance verification checks against policy.
///
/// This computes:
/// 1) explicit mismatch reason codes for diagnostics
/// 2) deterministic fallback action
/// 3) startup status (`verified`, `degraded`, or `failed`)
#[must_use]
pub fn evaluate_startup_verification(
    report: StartupVerificationReport,
    policy: StartupVerificationPolicy,
) -> StartupVerificationOutcome {
    let mut action = StartupVerificationAction::Continue;
    let mut alerts = Vec::new();

    if policy.require_attestation && !report.attestation_present {
        action = select_stricter_action(action, policy.on_attestation_missing);
        push_alert(
            &mut alerts,
            REASON_ATTESTATION_MISSING,
            policy.on_attestation_missing,
            "startup attestation missing",
        );
    }

    if report.attestation_present {
        if policy.require_signature && !report.signature_present {
            action = select_stricter_action(action, policy.on_signature_missing);
            push_alert(
                &mut alerts,
                REASON_SIGNATURE_MISSING,
                policy.on_signature_missing,
                "signed attestation required but signature missing",
            );
        }

        if report.signature_present && !report.signature_valid {
            action = select_stricter_action(action, policy.on_signature_invalid);
            push_alert(
                &mut alerts,
                REASON_SIGNATURE_INVALID,
                policy.on_signature_invalid,
                "attestation signature validation failed",
            );
        }

        if report.attestation_parsed && report.has_hash_mismatch() {
            action = select_stricter_action(action, policy.on_hash_mismatch);
            push_alert(
                &mut alerts,
                REASON_HASH_MISMATCH,
                policy.on_hash_mismatch,
                "runtime provenance hash mismatch",
            );
        }
    }

    let status = if alerts.is_empty() {
        StartupVerificationStatus::Verified
    } else if matches!(action, StartupVerificationAction::AbortStartup) {
        StartupVerificationStatus::Failed
    } else {
        StartupVerificationStatus::Degraded
    };

    StartupVerificationOutcome {
        status,
        action,
        alerts,
    }
}

// ─── Replay Metadata ────────────────────────────────────────────────────────

/// Replay metadata for deterministic replay of captured sessions.
///
/// This matches the `replay` block in the evidence JSONL schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayMeta {
    /// Replay mode used during capture.
    pub mode: ReplayMode,
    /// Random seed for deterministic replay.
    pub seed: Option<u64>,
    /// Tick interval in milliseconds for deterministic replay.
    pub tick_ms: Option<u64>,
    /// Range of `frame_seq` values in the evidence log.
    pub frame_seq_range: Option<FrameSeqRange>,
    /// Total evidence events in the trace.
    pub event_count: u64,
    /// Time span of the captured trace.
    pub trace_duration_ms: u64,
}

/// Replay mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayMode {
    /// Captured from live operation.
    Live,
    /// Captured from a deterministic replay environment.
    Deterministic,
}

/// Range of frame sequence numbers in a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSeqRange {
    /// First `frame_seq` in the evidence log.
    pub first: u64,
    /// Last `frame_seq` in the evidence log.
    pub last: u64,
}

// ─── Trace Query + Replay Tooling Contract ────────────────────────────────

/// Canonical evidence event types used by trace query filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceEventType {
    /// Decision event.
    Decision,
    /// Alert event.
    Alert,
    /// Degradation event.
    Degradation,
    /// State transition event.
    Transition,
    /// Replay marker event.
    ReplayMarker,
}

/// Sort order for trace query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TraceSortOrder {
    /// Earliest frame sequence first.
    OldestFirst,
    /// Latest frame sequence first.
    #[default]
    NewestFirst,
}

/// Query/filter model for evidence trace lookups.
///
/// This contract is shared by CLI and TUI flows:
/// - CLI: `fsfs trace query ...`
/// - TUI: trace/evidence views and drilldown panels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceQueryFilter {
    /// Exact trace ID match.
    pub trace_id: Option<String>,
    /// Root request correlation ID.
    pub root_request_id: Option<String>,
    /// Restrict to one project key.
    pub project_key: Option<String>,
    /// Restrict to one instance ID.
    pub instance_id: Option<String>,
    /// Prefix match on canonical reason codes.
    pub reason_code_prefix: Option<String>,
    /// Event-type whitelist; empty means all types.
    pub event_types: Vec<TraceEventType>,
    /// Inclusive lower bound on frame sequence.
    pub since_frame_seq: Option<u64>,
    /// Inclusive upper bound on frame sequence.
    pub until_frame_seq: Option<u64>,
    /// Max records to return.
    pub limit: u16,
    /// Ordering mode.
    pub sort: TraceSortOrder,
}

impl Default for TraceQueryFilter {
    fn default() -> Self {
        Self {
            trace_id: None,
            root_request_id: None,
            project_key: None,
            instance_id: None,
            reason_code_prefix: None,
            event_types: Vec::new(),
            since_frame_seq: None,
            until_frame_seq: None,
            limit: 200,
            sort: TraceSortOrder::NewestFirst,
        }
    }
}

impl TraceQueryFilter {
    /// Validate filter invariants and bounded query semantics.
    ///
    /// # Errors
    ///
    /// Returns a static reason code when any filter invariant is violated.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.limit == 0 {
            return Err("trace.query.limit.zero");
        }
        if self.limit > 1_000 {
            return Err("trace.query.limit.too_large");
        }

        for (field, value) in [
            ("trace_id", self.trace_id.as_deref()),
            ("root_request_id", self.root_request_id.as_deref()),
            ("project_key", self.project_key.as_deref()),
            ("instance_id", self.instance_id.as_deref()),
            ("reason_code_prefix", self.reason_code_prefix.as_deref()),
        ] {
            if value.is_some_and(|text| text.trim().is_empty()) {
                return Err(match field {
                    "trace_id" => "trace.query.trace_id.empty",
                    "root_request_id" => "trace.query.root_request_id.empty",
                    "project_key" => "trace.query.project_key.empty",
                    "instance_id" => "trace.query.instance_id.empty",
                    _ => "trace.query.reason_code_prefix.empty",
                });
            }
        }

        if let (Some(start), Some(end)) = (self.since_frame_seq, self.until_frame_seq)
            && start > end
        {
            return Err("trace.query.frame_seq_range.invalid");
        }

        Ok(())
    }

    /// Whether an event type passes this filter.
    #[must_use]
    pub fn includes_event_type(&self, event_type: TraceEventType) -> bool {
        self.event_types.is_empty() || self.event_types.contains(&event_type)
    }
}

/// Client surface for replay entrypoint invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayClientSurface {
    /// Agent/automation friendly command invocation.
    Cli,
    /// Interactive debug flow launched from TUI.
    Tui,
}

/// Replay entrypoint semantics for deterministic trace replays by trace ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayEntrypoint {
    /// Trace ID to replay.
    pub trace_id: String,
    /// Surface initiating the replay flow.
    pub client_surface: ReplayClientSurface,
    /// Optional absolute/relative path to `manifest.json`.
    pub manifest_path: Option<String>,
    /// Optional artifact-pack root that can resolve `manifest.json`.
    pub artifact_root: Option<String>,
    /// Optional start frame sequence for partial replay.
    pub start_frame_seq: Option<u64>,
    /// Optional end frame sequence for partial replay.
    pub end_frame_seq: Option<u64>,
    /// Whether to fail replay on unknown reason codes.
    pub strict_reason_codes: bool,
}

impl ReplayEntrypoint {
    /// Validate replay entrypoint invariants.
    ///
    /// # Errors
    ///
    /// Returns a static reason code when replay parameters are invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.trace_id.trim().is_empty() {
            return Err("trace.replay.trace_id.empty");
        }
        if self
            .manifest_path
            .as_deref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err("trace.replay.manifest_path.empty");
        }
        if self
            .artifact_root
            .as_deref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err("trace.replay.artifact_root.empty");
        }
        if self.manifest_path.is_none() && self.artifact_root.is_none() {
            return Err("trace.replay.source.missing");
        }
        if let (Some(start), Some(end)) = (self.start_frame_seq, self.end_frame_seq)
            && start > end
        {
            return Err("trace.replay.frame_seq_range.invalid");
        }
        Ok(())
    }

    /// Build deterministic CLI arguments for replay invocation.
    #[must_use]
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "repro".to_owned(),
            "replay".to_owned(),
            "--trace-id".to_owned(),
            self.trace_id.clone(),
        ];

        if let Some(manifest_path) = &self.manifest_path {
            args.push("--manifest".to_owned());
            args.push(manifest_path.clone());
        }
        if let Some(artifact_root) = &self.artifact_root {
            args.push("--artifact-root".to_owned());
            args.push(artifact_root.clone());
        }
        if let Some(start) = self.start_frame_seq {
            args.push("--from-frame".to_owned());
            args.push(start.to_string());
        }
        if let Some(end) = self.end_frame_seq {
            args.push("--to-frame".to_owned());
            args.push(end.to_string());
        }
        if self.strict_reason_codes {
            args.push("--strict-reason-codes".to_owned());
        }
        args
    }

    /// Canonical TUI action ID for replay requests.
    #[must_use]
    pub const fn tui_action_id(&self) -> &'static str {
        "diag.replay_trace"
    }
}

// ─── Environment Snapshot ───────────────────────────────────────────────────

/// Captured environment variables (redacted per privacy policy).
///
/// Only captures FSFS_* and selected system variables. Sensitive values
/// (tokens, keys) are replaced with `<redacted>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvSnapshot {
    /// Environment variable entries.
    pub variables: Vec<EnvEntry>,
    /// Redaction note.
    pub redaction_note: String,
}

/// A single environment variable entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvEntry {
    /// Variable name.
    pub key: String,
    /// Variable value (may be redacted).
    pub value: String,
    /// Whether the value was redacted.
    pub redacted: bool,
}

/// Environment variable prefixes that are safe to capture.
pub const SAFE_ENV_PREFIXES: &[&str] = &[
    "FSFS_",
    "FRANKENSEARCH_",
    "RUST_LOG",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
];

/// Environment variable substrings that trigger redaction.
pub const REDACT_PATTERNS: &[&str] = &["TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL", "AUTH"];

/// Check if an environment variable value should be redacted.
#[must_use]
pub fn should_redact_env(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    REDACT_PATTERNS.iter().any(|pat| upper.contains(pat))
}

/// Check if an environment variable should be captured at all.
#[must_use]
pub fn should_capture_env(key: &str) -> bool {
    SAFE_ENV_PREFIXES
        .iter()
        .any(|prefix| key.starts_with(prefix))
}

/// Build an environment snapshot from an iterator of key/value pairs.
///
/// Captures only allowlisted keys, redacts sensitive values, and sorts entries
/// by key for deterministic pack generation.
#[must_use]
pub fn capture_env_snapshot<I>(vars: I) -> EnvSnapshot
where
    I: IntoIterator<Item = (String, String)>,
{
    let mut variables: Vec<EnvEntry> = vars
        .into_iter()
        .filter_map(|(key, value)| {
            if !should_capture_env(&key) {
                return None;
            }
            let redacted = should_redact_env(&key);
            Some(EnvEntry {
                key,
                value: if redacted { "<redacted>".into() } else { value },
                redacted,
            })
        })
        .collect();

    variables.sort_by(|left, right| left.key.cmp(&right.key));

    EnvSnapshot {
        variables,
        redaction_note: "Sensitive values redacted by key policy".into(),
    }
}

/// Capture a snapshot of the current process environment.
#[must_use]
pub fn capture_current_env_snapshot() -> EnvSnapshot {
    capture_env_snapshot(std::env::vars())
}

// ─── Pack File Listing ──────────────────────────────────────────────────────

/// All files that belong to a complete repro pack.
pub const PACK_FILES: &[&str] = &[
    MANIFEST_FILENAME,
    EVIDENCE_FILENAME,
    CONFIG_FILENAME,
    ENV_FILENAME,
    MODEL_MANIFEST_FILENAME,
    INDEX_CHECKSUMS_FILENAME,
    PROVENANCE_ATTESTATION_FILENAME,
    REPLAY_META_FILENAME,
];

/// Files retained in each tier.
#[must_use]
pub const fn files_for_tier(tier: RetentionTier) -> &'static [&'static str] {
    match tier {
        RetentionTier::Hot => PACK_FILES,
        RetentionTier::Warm => &[
            MANIFEST_FILENAME,
            EVIDENCE_FILENAME,
            CONFIG_FILENAME,
            MODEL_MANIFEST_FILENAME,
            INDEX_CHECKSUMS_FILENAME,
            PROVENANCE_ATTESTATION_FILENAME,
            REPLAY_META_FILENAME,
        ],
        RetentionTier::Cold => &[
            MANIFEST_FILENAME,
            INDEX_CHECKSUMS_FILENAME,
            PROVENANCE_ATTESTATION_FILENAME,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_primitives::ScreenAction;

    // ─── CaptureReason ──────────────────────────────────────────────────

    #[test]
    fn capture_reason_display() {
        assert_eq!(CaptureReason::Manual.to_string(), "manual");
        assert_eq!(CaptureReason::SloViolation.to_string(), "slo_violation");
        assert_eq!(CaptureReason::Error.to_string(), "error");
        assert_eq!(CaptureReason::TestFailure.to_string(), "test_failure");
        assert_eq!(CaptureReason::Benchmark.to_string(), "benchmark");
        assert_eq!(CaptureReason::Scheduled.to_string(), "scheduled");
    }

    #[test]
    fn capture_reason_serde_roundtrip() {
        for reason in [
            CaptureReason::Manual,
            CaptureReason::SloViolation,
            CaptureReason::Error,
            CaptureReason::TestFailure,
            CaptureReason::Benchmark,
            CaptureReason::Scheduled,
        ] {
            let json = serde_json::to_string(&reason).unwrap();
            let decoded: CaptureReason = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, reason);
        }
    }

    // ─── RetentionTier ──────────────────────────────────────────────────

    #[test]
    fn retention_tier_display() {
        assert_eq!(RetentionTier::Hot.to_string(), "hot");
        assert_eq!(RetentionTier::Warm.to_string(), "warm");
        assert_eq!(RetentionTier::Cold.to_string(), "cold");
    }

    #[test]
    fn retention_tier_serde_roundtrip() {
        for tier in [RetentionTier::Hot, RetentionTier::Warm, RetentionTier::Cold] {
            let json = serde_json::to_string(&tier).unwrap();
            let decoded: RetentionTier = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, tier);
        }
    }

    // ─── RetentionPolicy ────────────────────────────────────────────────

    #[test]
    fn retention_policy_default() {
        let policy = RetentionPolicy::default();
        assert_eq!(policy.hot_days, 7);
        assert_eq!(policy.warm_days, 90);
        assert_eq!(policy.cold_max_days, 0);
    }

    #[test]
    fn retention_tier_for_age() {
        let policy = RetentionPolicy::default();

        assert_eq!(policy.tier_for_age(0), Some(RetentionTier::Hot));
        assert_eq!(policy.tier_for_age(7), Some(RetentionTier::Hot));
        assert_eq!(policy.tier_for_age(8), Some(RetentionTier::Warm));
        assert_eq!(policy.tier_for_age(90), Some(RetentionTier::Warm));
        assert_eq!(policy.tier_for_age(91), Some(RetentionTier::Cold));
        // cold_max_days=0 means keep forever
        assert_eq!(policy.tier_for_age(365), Some(RetentionTier::Cold));
    }

    #[test]
    fn retention_with_cold_expiry() {
        let policy = RetentionPolicy {
            hot_days: 7,
            warm_days: 30,
            cold_max_days: 180,
        };

        assert_eq!(policy.tier_for_age(180), Some(RetentionTier::Cold));
        assert_eq!(policy.tier_for_age(181), None); // expired
    }

    #[test]
    fn retention_policy_serde_roundtrip() {
        let policy = RetentionPolicy {
            hot_days: 14,
            warm_days: 60,
            cold_max_days: 365,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let decoded: RetentionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, policy);
    }

    // ─── ReproManifest ──────────────────────────────────────────────────

    #[test]
    fn manifest_serde_roundtrip() {
        let manifest = ReproManifest {
            schema_version: REPRO_SCHEMA_VERSION,
            pack_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".into(),
            trace_id: "01JAH9A2WZZZZZZZZZZZZZZZZZ".into(),
            capture_reason: CaptureReason::Manual,
            captured_at: "2026-02-14T00:00:00Z".into(),
            instance: ReproInstance {
                instance_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".into(),
                project_key: "/data/projects/test".into(),
                host_name: "atlas".into(),
                pid: Some(4242),
                version: "0.1.0".into(),
                rust_version: Some("1.85.0-nightly".into()),
            },
            artifacts: vec![ArtifactEntry {
                filename: MANIFEST_FILENAME.into(),
                size_bytes: 1024,
                checksum_xxh3: "abcdef0123456789".into(),
                content_type: "application/json".into(),
                present: true,
            }],
            retention_tier: RetentionTier::Hot,
            expires_at: Some("2026-02-21T00:00:00Z".into()),
            redaction_policy_version: "v1".into(),
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let decoded: ReproManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.schema_version, REPRO_SCHEMA_VERSION);
        assert_eq!(decoded.capture_reason, CaptureReason::Manual);
        assert_eq!(decoded.retention_tier, RetentionTier::Hot);
        assert_eq!(decoded.artifacts.len(), 1);
    }

    // ─── ModelSnapshot ──────────────────────────────────────────────────

    #[test]
    fn model_manifest_serde_roundtrip() {
        let manifest = ModelManifest {
            models: vec![
                ModelSnapshot {
                    model_id: "potion-multilingual-128M".into(),
                    revision: "abc123def456".into(),
                    dimension: 256,
                    tier: "fast".into(),
                    model_dir: "/home/user/.cache/frankensearch/models".into(),
                },
                ModelSnapshot {
                    model_id: "all-MiniLM-L6-v2".into(),
                    revision: "fed654cba321".into(),
                    dimension: 384,
                    tier: "quality".into(),
                    model_dir: "/home/user/.cache/frankensearch/models".into(),
                },
            ],
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let decoded: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.models.len(), 2);
        assert_eq!(decoded.models[0].dimension, 256);
        assert_eq!(decoded.models[1].dimension, 384);
    }

    // ─── Provenance Attestation ────────────────────────────────────────

    #[test]
    fn provenance_attestation_serde_roundtrip() {
        let attestation = ProvenanceAttestation {
            schema_version: PROVENANCE_SCHEMA_VERSION,
            attestation_id: "01JAH9A2W8F8Q6GQ4C7M3N2P1R".into(),
            generated_at: "2026-02-14T00:00:00Z".into(),
            build: BuildProvenance {
                source_commit: "0123456789abcdef0123456789abcdef01234567".into(),
                build_profile: "release".into(),
                rustc_version: "1.85.0-nightly".into(),
                target_triple: "x86_64-unknown-linux-gnu".into(),
            },
            runtime: RuntimeProvenance {
                binary_hash_sha256:
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                config_hash_sha256:
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
                index_manifest_hash_sha256:
                    "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            },
            artifact_hashes: vec![ArtifactHash {
                path: "manifest.json".into(),
                sha256: "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                    .into(),
            }],
            signature: Some(AttestationSignature {
                algorithm: "ed25519".into(),
                key_id: "build-key-1".into(),
                signature_b64: "c2lnbmF0dXJl".into(),
            }),
        };

        let json = serde_json::to_string(&attestation).unwrap();
        let decoded: ProvenanceAttestation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, attestation);
        assert_eq!(decoded.schema_version, PROVENANCE_SCHEMA_VERSION);
    }

    // ─── Startup Verification ──────────────────────────────────────────

    #[test]
    fn startup_verification_verified_when_all_checks_pass() {
        let outcome = evaluate_startup_verification(
            StartupVerificationReport {
                attestation_present: true,
                attestation_parsed: true,
                signature_present: true,
                signature_valid: true,
                binary_hash_match: true,
                config_hash_match: true,
                index_manifest_hash_match: true,
            },
            StartupVerificationPolicy::default(),
        );

        assert_eq!(outcome.status, StartupVerificationStatus::Verified);
        assert_eq!(outcome.action, StartupVerificationAction::Continue);
        assert!(outcome.alerts.is_empty());
        assert!(outcome.permits_startup());
    }

    #[test]
    fn startup_verification_attestation_missing_uses_policy_fallback() {
        let outcome = evaluate_startup_verification(
            StartupVerificationReport {
                attestation_present: false,
                attestation_parsed: false,
                signature_present: false,
                signature_valid: false,
                binary_hash_match: true,
                config_hash_match: true,
                index_manifest_hash_match: true,
            },
            StartupVerificationPolicy {
                on_attestation_missing: StartupVerificationAction::EnterReadOnly,
                ..StartupVerificationPolicy::default()
            },
        );

        assert_eq!(outcome.status, StartupVerificationStatus::Degraded);
        assert_eq!(outcome.action, StartupVerificationAction::EnterReadOnly);
        assert_eq!(outcome.alerts.len(), 1);
        assert_eq!(outcome.alerts[0].reason_code, REASON_ATTESTATION_MISSING);
    }

    #[test]
    fn startup_verification_signature_invalid_can_abort_startup() {
        let outcome = evaluate_startup_verification(
            StartupVerificationReport {
                attestation_present: true,
                attestation_parsed: true,
                signature_present: true,
                signature_valid: false,
                binary_hash_match: true,
                config_hash_match: true,
                index_manifest_hash_match: true,
            },
            StartupVerificationPolicy {
                require_signature: true,
                ..StartupVerificationPolicy::default()
            },
        );

        assert_eq!(outcome.status, StartupVerificationStatus::Failed);
        assert_eq!(outcome.action, StartupVerificationAction::AbortStartup);
        assert_eq!(outcome.alerts.len(), 1);
        assert_eq!(outcome.alerts[0].reason_code, REASON_SIGNATURE_INVALID);
        assert!(!outcome.permits_startup());
    }

    #[test]
    fn startup_verification_hash_mismatch_emits_alert_and_safe_mode() {
        let outcome = evaluate_startup_verification(
            StartupVerificationReport {
                attestation_present: true,
                attestation_parsed: true,
                signature_present: false,
                signature_valid: false,
                binary_hash_match: false,
                config_hash_match: true,
                index_manifest_hash_match: true,
            },
            StartupVerificationPolicy {
                on_hash_mismatch: StartupVerificationAction::EnterSafeMode,
                ..StartupVerificationPolicy::default()
            },
        );

        assert_eq!(outcome.status, StartupVerificationStatus::Degraded);
        assert_eq!(outcome.action, StartupVerificationAction::EnterSafeMode);
        assert_eq!(outcome.alerts.len(), 1);
        assert_eq!(outcome.alerts[0].reason_code, REASON_HASH_MISMATCH);
    }

    #[test]
    fn startup_verification_uses_strictest_action_across_mismatches() {
        let outcome = evaluate_startup_verification(
            StartupVerificationReport {
                attestation_present: true,
                attestation_parsed: true,
                signature_present: false,
                signature_valid: false,
                binary_hash_match: false,
                config_hash_match: false,
                index_manifest_hash_match: true,
            },
            StartupVerificationPolicy {
                require_signature: true,
                on_signature_missing: StartupVerificationAction::EnterReadOnly,
                on_hash_mismatch: StartupVerificationAction::ContinueWithAlert,
                ..StartupVerificationPolicy::default()
            },
        );

        assert_eq!(outcome.status, StartupVerificationStatus::Degraded);
        assert_eq!(outcome.action, StartupVerificationAction::EnterReadOnly);
        assert_eq!(outcome.alerts.len(), 2);
    }

    // ─── IndexChecksums ─────────────────────────────────────────────────

    #[test]
    fn index_checksums_serde_roundtrip() {
        let checksums = IndexChecksums {
            indices: vec![
                IndexChecksum {
                    index_name: "vector.fast".into(),
                    index_type: "fsvi".into(),
                    checksum_xxh3: "1234567890abcdef".into(),
                    size_bytes: 102_400,
                    record_count: 5000,
                    fec_checksum_xxh3: Some("fedcba0987654321".into()),
                },
                IndexChecksum {
                    index_name: "lexical".into(),
                    index_type: "tantivy".into(),
                    checksum_xxh3: "abcdef1234567890".into(),
                    size_bytes: 204_800,
                    record_count: 5000,
                    fec_checksum_xxh3: None,
                },
            ],
        };

        let json = serde_json::to_string(&checksums).unwrap();
        let decoded: IndexChecksums = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.indices.len(), 2);
        assert!(decoded.indices[0].fec_checksum_xxh3.is_some());
        assert!(decoded.indices[1].fec_checksum_xxh3.is_none());
    }

    // ─── ReplayMeta ─────────────────────────────────────────────────────

    #[test]
    fn replay_meta_live_mode() {
        let meta = ReplayMeta {
            mode: ReplayMode::Live,
            seed: None,
            tick_ms: None,
            frame_seq_range: None,
            event_count: 42,
            trace_duration_ms: 1500,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let decoded: ReplayMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.mode, ReplayMode::Live);
        assert!(decoded.seed.is_none());
    }

    #[test]
    fn replay_meta_deterministic_mode() {
        let meta = ReplayMeta {
            mode: ReplayMode::Deterministic,
            seed: Some(12_345),
            tick_ms: Some(10),
            frame_seq_range: Some(FrameSeqRange { first: 0, last: 99 }),
            event_count: 100,
            trace_duration_ms: 1000,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let decoded: ReplayMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.mode, ReplayMode::Deterministic);
        assert_eq!(decoded.seed, Some(12_345));
        assert_eq!(decoded.tick_ms, Some(10));
        let range = decoded.frame_seq_range.unwrap();
        assert_eq!(range.first, 0);
        assert_eq!(range.last, 99);
    }

    // ─── Trace Query + Replay Tooling Contract ────────────────────────

    #[test]
    fn trace_query_filter_default_is_valid_and_bounded() {
        let filter = TraceQueryFilter::default();
        assert_eq!(filter.limit, 200);
        assert_eq!(filter.sort, TraceSortOrder::NewestFirst);
        assert!(filter.validate().is_ok());
        assert!(filter.includes_event_type(TraceEventType::Decision));
    }

    #[test]
    fn trace_query_filter_validation_rejects_invalid_requests() {
        let invalid_limit = TraceQueryFilter {
            limit: 0,
            ..TraceQueryFilter::default()
        };
        assert_eq!(invalid_limit.validate(), Err("trace.query.limit.zero"));

        let invalid_bounds = TraceQueryFilter {
            since_frame_seq: Some(20),
            until_frame_seq: Some(10),
            ..TraceQueryFilter::default()
        };
        assert_eq!(
            invalid_bounds.validate(),
            Err("trace.query.frame_seq_range.invalid")
        );

        let invalid_trace_id = TraceQueryFilter {
            trace_id: Some("  ".to_owned()),
            ..TraceQueryFilter::default()
        };
        assert_eq!(
            invalid_trace_id.validate(),
            Err("trace.query.trace_id.empty")
        );
    }

    #[test]
    fn trace_query_filter_event_whitelist_is_respected() {
        let filter = TraceQueryFilter {
            event_types: vec![TraceEventType::Alert, TraceEventType::ReplayMarker],
            ..TraceQueryFilter::default()
        };
        assert!(filter.includes_event_type(TraceEventType::Alert));
        assert!(filter.includes_event_type(TraceEventType::ReplayMarker));
        assert!(!filter.includes_event_type(TraceEventType::Decision));
    }

    #[test]
    fn replay_entrypoint_validation_rejects_missing_sources_and_bad_ranges() {
        let missing_source = ReplayEntrypoint {
            trace_id: "trace-1".to_owned(),
            client_surface: ReplayClientSurface::Cli,
            manifest_path: None,
            artifact_root: None,
            start_frame_seq: None,
            end_frame_seq: None,
            strict_reason_codes: false,
        };
        assert_eq!(
            missing_source.validate(),
            Err("trace.replay.source.missing")
        );

        let invalid_bounds = ReplayEntrypoint {
            trace_id: "trace-1".to_owned(),
            client_surface: ReplayClientSurface::Cli,
            manifest_path: Some("manifest.json".to_owned()),
            artifact_root: None,
            start_frame_seq: Some(200),
            end_frame_seq: Some(100),
            strict_reason_codes: false,
        };
        assert_eq!(
            invalid_bounds.validate(),
            Err("trace.replay.frame_seq_range.invalid")
        );
    }

    #[test]
    fn replay_entrypoint_cli_args_are_deterministic() {
        let entrypoint = ReplayEntrypoint {
            trace_id: "trace-abc".to_owned(),
            client_surface: ReplayClientSurface::Cli,
            manifest_path: Some("/tmp/repro/manifest.json".to_owned()),
            artifact_root: Some("/tmp/repro".to_owned()),
            start_frame_seq: Some(10),
            end_frame_seq: Some(99),
            strict_reason_codes: true,
        };
        assert!(entrypoint.validate().is_ok());
        assert_eq!(
            entrypoint.to_cli_args(),
            vec![
                "repro",
                "replay",
                "--trace-id",
                "trace-abc",
                "--manifest",
                "/tmp/repro/manifest.json",
                "--artifact-root",
                "/tmp/repro",
                "--from-frame",
                "10",
                "--to-frame",
                "99",
                "--strict-reason-codes",
            ]
        );
    }

    #[test]
    fn replay_entrypoint_tui_action_id_is_stable() {
        let entrypoint = ReplayEntrypoint {
            trace_id: "trace-tui".to_owned(),
            client_surface: ReplayClientSurface::Tui,
            manifest_path: Some("manifest.json".to_owned()),
            artifact_root: None,
            start_frame_seq: None,
            end_frame_seq: None,
            strict_reason_codes: false,
        };
        assert_eq!(entrypoint.tui_action_id(), "diag.replay_trace");
        assert_eq!(
            ScreenAction::from_palette_action_id(entrypoint.tui_action_id()),
            Some(ScreenAction::ReplayTrace)
        );
    }

    // ─── Environment Snapshot ───────────────────────────────────────────

    #[test]
    fn env_snapshot_serde_roundtrip() {
        let snapshot = EnvSnapshot {
            variables: vec![
                EnvEntry {
                    key: "FSFS_SEARCH_FAST_ONLY".into(),
                    value: "true".into(),
                    redacted: false,
                },
                EnvEntry {
                    key: "FSFS_SECRET_TOKEN".into(),
                    value: "<redacted>".into(),
                    redacted: true,
                },
            ],
            redaction_note:
                "Sensitive values matching TOKEN/SECRET/KEY/PASSWORD/CREDENTIAL/AUTH are redacted"
                    .into(),
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let decoded: EnvSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.variables.len(), 2);
        assert!(!decoded.variables[0].redacted);
        assert!(decoded.variables[1].redacted);
    }

    #[test]
    fn should_redact_env_detects_sensitive_keys() {
        assert!(should_redact_env("FSFS_SECRET_TOKEN"));
        assert!(should_redact_env("API_KEY"));
        assert!(should_redact_env("DATABASE_PASSWORD"));
        assert!(should_redact_env("OAUTH_SECRET"));
        assert!(should_redact_env("AWS_CREDENTIAL_FILE"));
        assert!(should_redact_env("GITHUB_AUTH_TOKEN"));

        assert!(!should_redact_env("FSFS_SEARCH_FAST_ONLY"));
        assert!(!should_redact_env("HOME"));
        assert!(!should_redact_env("RUST_LOG"));
    }

    #[test]
    fn should_capture_env_filters_by_prefix() {
        assert!(should_capture_env("FSFS_DISCOVERY_ROOTS"));
        assert!(should_capture_env("FRANKENSEARCH_OPS_SLO_SEARCH_P99_MS"));
        assert!(should_capture_env("RUST_LOG"));
        assert!(should_capture_env("HOME"));
        assert!(should_capture_env("XDG_CONFIG_HOME"));

        assert!(!should_capture_env("PATH"));
        assert!(!should_capture_env("LD_LIBRARY_PATH"));
        assert!(!should_capture_env("AWS_ACCESS_KEY_ID"));
    }

    #[test]
    fn capture_env_snapshot_filters_redacts_and_sorts() {
        let snapshot = capture_env_snapshot(vec![
            ("PATH".into(), "/usr/bin".into()),
            ("FSFS_SECRET_TOKEN".into(), "abc123".into()),
            ("HOME".into(), "/home/tester".into()),
            ("FRANKENSEARCH_OPS_MODE".into(), "dev".into()),
        ]);

        let keys: Vec<&str> = snapshot
            .variables
            .iter()
            .map(|entry| entry.key.as_str())
            .collect();
        assert_eq!(
            keys,
            vec!["FRANKENSEARCH_OPS_MODE", "FSFS_SECRET_TOKEN", "HOME"]
        );
        let secret = snapshot
            .variables
            .iter()
            .find(|entry| entry.key == "FSFS_SECRET_TOKEN")
            .expect("secret present");
        assert!(secret.redacted);
        assert_eq!(secret.value, "<redacted>");
    }

    // ─── Pack Files ─────────────────────────────────────────────────────

    #[test]
    fn pack_files_count() {
        assert_eq!(PACK_FILES.len(), 8);
    }

    #[test]
    fn hot_tier_has_all_files() {
        let files = files_for_tier(RetentionTier::Hot);
        assert_eq!(files.len(), PACK_FILES.len());
    }

    #[test]
    fn warm_tier_drops_env() {
        let files = files_for_tier(RetentionTier::Warm);
        assert!(!files.contains(&ENV_FILENAME));
        assert!(files.contains(&EVIDENCE_FILENAME));
        assert!(files.contains(&MANIFEST_FILENAME));
        assert!(files.contains(&PROVENANCE_ATTESTATION_FILENAME));
    }

    #[test]
    fn cold_tier_keeps_only_manifest_and_checksums() {
        let files = files_for_tier(RetentionTier::Cold);
        assert_eq!(files.len(), 3);
        assert!(files.contains(&MANIFEST_FILENAME));
        assert!(files.contains(&INDEX_CHECKSUMS_FILENAME));
        assert!(files.contains(&PROVENANCE_ATTESTATION_FILENAME));
    }

    // ─── ReplayMode ─────────────────────────────────────────────────────

    #[test]
    fn replay_mode_serde_roundtrip() {
        for mode in [ReplayMode::Live, ReplayMode::Deterministic] {
            let json = serde_json::to_string(&mode).unwrap();
            let decoded: ReplayMode = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, mode);
        }
    }

    // ─── ArtifactEntry ──────────────────────────────────────────────────

    #[test]
    fn artifact_entry_serde_roundtrip() {
        let entry = ArtifactEntry {
            filename: "evidence.jsonl".into(),
            size_bytes: 4096,
            checksum_xxh3: "0123456789abcdef".into(),
            content_type: "application/x-ndjson".into(),
            present: true,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let decoded: ArtifactEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, entry);
    }

    #[test]
    fn artifact_entry_not_present_in_cold_tier() {
        let entry = ArtifactEntry {
            filename: EVIDENCE_FILENAME.into(),
            size_bytes: 0,
            checksum_xxh3: String::new(),
            content_type: "application/x-ndjson".into(),
            present: false,
        };
        assert!(!entry.present);
    }
}
