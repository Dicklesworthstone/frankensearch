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
//! 1. **On demand**: Operator requests via the TUI diagnostics action
//!    (`diag.replay_trace`); a dedicated `fsfs repro` CLI subcommand is a
//!    planned surface and does not exist today.
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

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

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
/// Schema version for canonical fsfs replay bundle contracts.
pub const REPLAY_BUNDLE_SCHEMA_VERSION: u8 = 1;
/// Kind tag for replay bundle contract-definition fixtures.
pub const REPLAY_BUNDLE_CONTRACT_KIND: &str = "fsfs_replay_bundle_contract_definition";
/// Kind tag for replay bundle manifest fixtures.
pub const REPLAY_BUNDLE_MANIFEST_KIND: &str = "fsfs_replay_bundle_manifest";
/// Schema version for deterministic degraded-mode incident suites.
pub const DEGRADED_INCIDENT_SUITE_SCHEMA_VERSION: u8 = 1;
/// Kind tag for degraded-mode incident suite contract fixtures.
pub const DEGRADED_INCIDENT_SUITE_CONTRACT_KIND: &str =
    "fsfs_degraded_incident_suite_contract_definition";
/// Kind tag for concrete degraded-mode incident suite fixtures.
pub const DEGRADED_INCIDENT_SUITE_KIND: &str = "fsfs_degraded_incident_suite";
/// Schema discriminator for deterministic operator workflow playback fixtures.
pub const OPERATOR_WORKFLOW_PLAYBACK_SCHEMA_VERSION: &str = "fsfs-operator-workflow-playback-v1";
/// Exact replay command for the deterministic operator workflow playback suite.
pub const OPERATOR_WORKFLOW_PLAYBACK_REPLAY_COMMAND: &str = "RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- cargo test -p frankensearch-fsfs --lib operator_workflow_playback -- --nocapture";

// ─── Capture Context ────────────────────────────────────────────────────────

/// What triggered the repro pack capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
    ///
    /// If `hot_days > warm_days` (which can happen via `Deserialize`
    /// bypassing the constructor), `warm_days` is treated as at least
    /// `hot_days` so the Warm tier is never silently unreachable.
    /// Similarly, when `cold_max_days > 0` and `cold_max_days < warm_days`,
    /// `cold_max_days` is clamped up to the effective warm boundary.
    #[must_use]
    pub const fn tier_for_age(&self, age_days: u16) -> Option<RetentionTier> {
        // Clamp thresholds so tiers are always reachable even if
        // deserialized values violate hot_days <= warm_days <= cold_max_days.
        let effective_warm = if self.warm_days >= self.hot_days {
            self.warm_days
        } else {
            self.hot_days
        };
        let effective_cold = if self.cold_max_days == 0 {
            0 // keep forever
        } else if self.cold_max_days >= effective_warm {
            self.cold_max_days
        } else {
            effective_warm
        };

        if age_days <= self.hot_days {
            Some(RetentionTier::Hot)
        } else if age_days <= effective_warm {
            Some(RetentionTier::Warm)
        } else if effective_cold == 0 || age_days <= effective_cold {
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

/// Structured startup verification artifact emitted for provenance checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceStartupCheck {
    /// Optional schema discriminator for external contracts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Schema version (currently 1).
    pub schema_version: u8,
    /// Trace ID correlating this check to a session.
    pub trace_id: String,
    /// Attestation identifier if available.
    pub attestation_id: String,
    /// Terminal verification status.
    pub status: StartupVerificationStatus,
    /// Selected fallback action.
    pub action: StartupVerificationAction,
    /// Raw verification checks.
    pub checks: StartupVerificationReport,
    /// Alert entries emitted for operators.
    pub alerts: Vec<StartupVerificationAlert>,
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
/// This contract is shared by the TUI trace/evidence views and drilldown
/// panels today; a dedicated `fsfs trace` CLI subcommand is a planned
/// surface and does not exist today.
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

// ─── Canonical Replay Bundle Contract ───────────────────────────────────────

/// Replay bundle schema version marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayBundleSchemaVersion1;

impl Serialize for ReplayBundleSchemaVersion1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(REPLAY_BUNDLE_SCHEMA_VERSION)
    }
}

impl<'de> Deserialize<'de> for ReplayBundleSchemaVersion1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        if value == REPLAY_BUNDLE_SCHEMA_VERSION {
            Ok(Self)
        } else {
            Err(de::Error::invalid_value(
                de::Unexpected::Unsigned(u64::from(value)),
                &"replay bundle schema version 1",
            ))
        }
    }
}

/// Kind marker for replay bundle contract definitions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplayBundleContractKind {
    /// Current replay bundle contract-definition kind.
    #[serde(rename = "fsfs_replay_bundle_contract_definition")]
    Current,
}

/// Kind marker for concrete replay bundle manifests.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReplayBundleManifestKind {
    /// Current replay bundle manifest kind.
    #[serde(rename = "fsfs_replay_bundle_manifest")]
    Current,
}

/// Scenario families that the canonical replay bundle can describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBundleScenarioKind {
    /// Search/query replay scenario.
    Search,
    /// Index construction or refresh replay scenario.
    Index,
    /// Doctor/repair diagnostic scenario.
    Doctor,
    /// Audit/evidence review scenario.
    Audit,
    /// Scenario that intentionally exercises degraded mode.
    DegradedMode,
}

/// Phase labels used in expected replay outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBundlePhase {
    /// Fast initial search phase.
    Initial,
    /// Quality/refined search phase.
    Refined,
    /// Index build or refresh phase.
    IndexBuild,
    /// Doctor diagnostic phase.
    Doctor,
    /// Audit/reporting phase.
    Audit,
    /// Degraded-mode phase.
    DegradedMode,
}

/// Expected status for one replay bundle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBundleOutcomeStatus {
    /// Phase must complete successfully.
    Succeeded,
    /// Phase is expected to complete with degraded behavior.
    Degraded,
    /// Phase must be skipped by policy.
    Skipped,
    /// Phase is expected to fail with the declared reason code.
    Failed,
}

/// Machine-readable contract definition for canonical replay bundles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleContractDefinition {
    /// Contract definition kind tag.
    pub kind: ReplayBundleContractKind,
    /// Schema version.
    pub v: ReplayBundleSchemaVersion1,
    /// Scenario kinds covered by the replay bundle schema.
    pub supported_scenarios: Vec<ReplayBundleScenarioKind>,
    /// Required top-level manifest fields.
    pub required_top_level_fields: Vec<String>,
    /// Required command fields.
    pub required_command_fields: Vec<String>,
    /// Required environment fields.
    pub required_environment_fields: Vec<String>,
    /// Required expected-outcome fields.
    pub required_expected_outcome_fields: Vec<String>,
    /// Required artifact manifest fields.
    pub required_artifact_fields: Vec<String>,
    /// Validation lanes that must accept/reject bundle fixtures.
    pub validation_modes: Vec<String>,
}

impl Default for ReplayBundleContractDefinition {
    fn default() -> Self {
        Self {
            kind: ReplayBundleContractKind::Current,
            v: ReplayBundleSchemaVersion1,
            supported_scenarios: vec![
                ReplayBundleScenarioKind::Search,
                ReplayBundleScenarioKind::Index,
                ReplayBundleScenarioKind::Doctor,
                ReplayBundleScenarioKind::Audit,
                ReplayBundleScenarioKind::DegradedMode,
            ],
            required_top_level_fields: vec![
                "kind".to_owned(),
                "v".to_owned(),
                "bundle_id".to_owned(),
                "scenario_id".to_owned(),
                "scenario_kind".to_owned(),
                "created_at".to_owned(),
                "command".to_owned(),
                "environment".to_owned(),
                "fixture_refs".to_owned(),
                "expected_phase_outcomes".to_owned(),
                "artifact_manifest".to_owned(),
            ],
            required_command_fields: vec![
                "client_surface".to_owned(),
                "argv".to_owned(),
                "working_dir".to_owned(),
            ],
            required_environment_fields: vec![
                "seed".to_owned(),
                "config_hash".to_owned(),
                "snapshot".to_owned(),
            ],
            required_expected_outcome_fields: vec![
                "phase".to_owned(),
                "status".to_owned(),
                "artifact_refs".to_owned(),
            ],
            required_artifact_fields: vec![
                "artifact_id".to_owned(),
                "path".to_owned(),
                "content_type".to_owned(),
                "checksum_sha256".to_owned(),
                "required".to_owned(),
            ],
            validation_modes: vec![
                "json_schema_valid_fixture".to_owned(),
                "json_schema_invalid_fixture".to_owned(),
                "rust_deserialize_roundtrip".to_owned(),
                "script_all_mode".to_owned(),
            ],
        }
    }
}

/// Command invocation captured by a replay bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleCommand {
    /// Client surface expected to run the command.
    pub client_surface: ReplayClientSurface,
    /// Exact argv vector, including executable name.
    pub argv: Vec<String>,
    /// Working directory for replay.
    pub working_dir: String,
}

/// Deterministic environment identity for replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleEnvironment {
    /// Required deterministic seed.
    pub seed: u64,
    /// Hash of resolved fsfs configuration.
    pub config_hash: String,
    /// Redacted environment snapshot.
    pub snapshot: EnvSnapshot,
}

/// Reference to an input fixture consumed by replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleFixtureRef {
    /// Stable fixture identifier used by phase expectations.
    pub fixture_id: String,
    /// Path relative to the repository or bundle root.
    pub path: String,
    /// SHA-256 digest for the fixture content.
    pub checksum_sha256: String,
}

/// One artifact expected or emitted by replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleArtifactRef {
    /// Stable artifact identifier used by phase expectations.
    pub artifact_id: String,
    /// Path relative to the bundle root.
    pub path: String,
    /// MIME type hint.
    pub content_type: String,
    /// SHA-256 digest for the artifact content.
    pub checksum_sha256: String,
    /// Whether the replay is invalid without this artifact.
    pub required: bool,
}

/// Artifact manifest for a replay bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleArtifactManifest {
    /// Artifacts expected or emitted by the replay.
    pub artifacts: Vec<ReplayBundleArtifactRef>,
}

/// Expected outcome for one replay phase.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleExpectedPhaseOutcome {
    /// Phase being asserted.
    pub phase: ReplayBundlePhase,
    /// Expected status for the phase.
    pub status: ReplayBundleOutcomeStatus,
    /// Required reason code for degraded/failed/skipped phases.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    /// Artifact identifiers that must exist for this phase.
    pub artifact_refs: Vec<String>,
}

/// Concrete manifest for a canonical fsfs replay bundle.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReplayBundleManifest {
    /// Manifest kind tag.
    pub kind: ReplayBundleManifestKind,
    /// Schema version.
    pub v: ReplayBundleSchemaVersion1,
    /// Stable bundle identifier.
    pub bundle_id: String,
    /// Scenario identifier.
    pub scenario_id: String,
    /// Replay scenario family.
    pub scenario_kind: ReplayBundleScenarioKind,
    /// RFC 3339 UTC timestamp for bundle creation.
    pub created_at: String,
    /// Canonical command invocation.
    pub command: ReplayBundleCommand,
    /// Deterministic environment identity.
    pub environment: ReplayBundleEnvironment,
    /// Input fixtures referenced by the replay.
    pub fixture_refs: Vec<ReplayBundleFixtureRef>,
    /// Expected phase-level replay outcomes.
    pub expected_phase_outcomes: Vec<ReplayBundleExpectedPhaseOutcome>,
    /// Artifact manifest for emitted or required replay outputs.
    pub artifact_manifest: ReplayBundleArtifactManifest,
}

impl ReplayBundleManifest {
    /// Validate replay bundle invariants beyond raw JSON shape.
    ///
    /// # Errors
    ///
    /// Returns a stable reason code when the bundle contract is violated.
    pub fn validate(&self) -> Result<(), &'static str> {
        require_non_empty("bundle_id", &self.bundle_id)?;
        require_non_empty("scenario_id", &self.scenario_id)?;
        require_non_empty("created_at", &self.created_at)?;
        require_non_empty("command.working_dir", &self.command.working_dir)?;
        require_non_empty("environment.config_hash", &self.environment.config_hash)?;
        if !is_sha256_digest(&self.environment.config_hash) {
            return Err("replay.bundle.environment.config_hash.invalid");
        }
        if self.command.argv.is_empty() {
            return Err("replay.bundle.command.argv.empty");
        }
        if self.command.argv.iter().any(|arg| arg.trim().is_empty()) {
            return Err("replay.bundle.command.argv.blank");
        }
        if self.fixture_refs.is_empty() {
            return Err("replay.bundle.fixture_refs.empty");
        }
        if self.expected_phase_outcomes.is_empty() {
            return Err("replay.bundle.expected_phase_outcomes.empty");
        }
        if self.artifact_manifest.artifacts.is_empty() {
            return Err("replay.bundle.artifact_manifest.empty");
        }
        if self.environment.snapshot.redaction_note.trim().is_empty() {
            return Err("replay.bundle.environment.redaction_note.empty");
        }

        let mut fixture_ids = std::collections::BTreeSet::new();
        for fixture in &self.fixture_refs {
            require_non_empty("fixture.fixture_id", &fixture.fixture_id)?;
            require_non_empty("fixture.path", &fixture.path)?;
            if !is_sha256_digest(&fixture.checksum_sha256) {
                return Err("replay.bundle.fixture.checksum.invalid");
            }
            if !fixture_ids.insert(fixture.fixture_id.as_str()) {
                return Err("replay.bundle.fixture_refs.duplicate_id");
            }
        }

        let mut artifact_ids = std::collections::BTreeSet::new();
        for artifact in &self.artifact_manifest.artifacts {
            require_non_empty("artifact.artifact_id", &artifact.artifact_id)?;
            require_non_empty("artifact.path", &artifact.path)?;
            require_non_empty("artifact.content_type", &artifact.content_type)?;
            if !is_sha256_digest(&artifact.checksum_sha256) {
                return Err("replay.bundle.artifact.checksum.invalid");
            }
            if !artifact_ids.insert(artifact.artifact_id.as_str()) {
                return Err("replay.bundle.artifact_manifest.duplicate_id");
            }
        }

        for outcome in &self.expected_phase_outcomes {
            if matches!(
                outcome.status,
                ReplayBundleOutcomeStatus::Degraded
                    | ReplayBundleOutcomeStatus::Failed
                    | ReplayBundleOutcomeStatus::Skipped
            ) && outcome
                .reason_code
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
            {
                return Err("replay.bundle.expected_phase.reason_code.missing");
            }
            for artifact_ref in &outcome.artifact_refs {
                require_non_empty("expected_phase.artifact_ref", artifact_ref)?;
                if !artifact_ids.contains(artifact_ref.as_str()) {
                    return Err("replay.bundle.expected_phase.artifact_ref.unknown");
                }
            }
        }

        for entry in &self.environment.snapshot.variables {
            require_non_empty("environment.snapshot.key", &entry.key)?;
        }

        Ok(())
    }
}

impl<'de> Deserialize<'de> for ReplayBundleManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawReplayBundleManifest {
            kind: ReplayBundleManifestKind,
            v: ReplayBundleSchemaVersion1,
            bundle_id: String,
            scenario_id: String,
            scenario_kind: ReplayBundleScenarioKind,
            created_at: String,
            command: ReplayBundleCommand,
            environment: ReplayBundleEnvironment,
            fixture_refs: Vec<ReplayBundleFixtureRef>,
            expected_phase_outcomes: Vec<ReplayBundleExpectedPhaseOutcome>,
            artifact_manifest: ReplayBundleArtifactManifest,
        }

        let raw = RawReplayBundleManifest::deserialize(deserializer)?;
        let manifest = Self {
            kind: raw.kind,
            v: raw.v,
            bundle_id: raw.bundle_id,
            scenario_id: raw.scenario_id,
            scenario_kind: raw.scenario_kind,
            created_at: raw.created_at,
            command: raw.command,
            environment: raw.environment,
            fixture_refs: raw.fixture_refs,
            expected_phase_outcomes: raw.expected_phase_outcomes,
            artifact_manifest: raw.artifact_manifest,
        };
        manifest.validate().map_err(de::Error::custom)?;
        Ok(manifest)
    }
}

// ─── Operator Workflow Playback Fixtures ───────────────────────────────────

/// Operator workflow families covered by deterministic playback fixtures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorWorkflowKind {
    /// Progressive search workflow.
    Search,
    /// Explainability drilldown workflow.
    Explain,
    /// Doctor/status diagnostics workflow.
    Doctor,
    /// Audit/report review workflow.
    Audit,
    /// Degraded-mode recovery workflow.
    DegradedMode,
    /// Profile and benchmark review workflow.
    ProfileReview,
    /// Ops TUI fleet triage workflow.
    OpsTui,
}

/// One replayable input step for an operator workflow fixture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorWorkflowReplayInput {
    /// Monotonic input sequence number.
    pub seq: u32,
    /// Offset from workflow start.
    pub offset_ms: u64,
    /// Stable input encoding such as `key:/`, `cmd:search`, or `resize:132x40`.
    pub input: String,
    /// Screen expected after applying the input.
    pub expected_screen: String,
    /// Stable reason code emitted by this replay step.
    pub reason_code: String,
}

/// Deterministic snapshot captured during workflow playback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorWorkflowSnapshot {
    /// Human-readable stable label.
    pub label: String,
    /// Screen or output surface identifier.
    pub screen_id: String,
    /// Stable checksum for the redacted rendered state.
    pub state_checksum: String,
    /// Reason code expected in the snapshot metadata.
    pub reason_code: String,
    /// Artifact identifiers required by this snapshot.
    pub artifact_refs: Vec<String>,
}

/// One deterministic workflow playback fixture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorWorkflowPlaybackFixture {
    /// Stable fixture identifier.
    pub fixture_id: String,
    /// Workflow family.
    pub workflow: OperatorWorkflowKind,
    /// Surface used to replay the fixture.
    pub client_surface: ReplayClientSurface,
    /// Exact command vector for the replay surface.
    pub command: ReplayBundleCommand,
    /// Stable replay input sequence.
    pub replay_inputs: Vec<OperatorWorkflowReplayInput>,
    /// Expected state snapshots after key navigation/output points.
    pub snapshots: Vec<OperatorWorkflowSnapshot>,
    /// Exact command that replays or validates this fixture.
    pub replay_command: String,
}

/// Deterministic fixture pack for fsfs and ops operator workflows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorWorkflowPlaybackSuite {
    /// Contract schema discriminator.
    pub schema_version: String,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Deterministic playback seed.
    pub seed: u64,
    /// Hash of the resolved workflow fixture configuration.
    pub config_hash: String,
    /// RFC 3339 UTC timestamp for fixture creation.
    pub generated_at: String,
    /// Redacted deterministic environment identity.
    pub environment: ReplayBundleEnvironment,
    /// Shared artifact manifest for all workflow fixtures.
    pub artifact_manifest: ReplayBundleArtifactManifest,
    /// Workflow fixtures in deterministic order.
    pub workflows: Vec<OperatorWorkflowPlaybackFixture>,
    /// Exact command for validating the entire suite.
    pub replay_command: String,
}

impl OperatorWorkflowPlaybackSuite {
    /// Validate suite invariants beyond raw JSON shape.
    ///
    /// # Errors
    ///
    /// Returns a stable reason code when playback fixture contracts are broken.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != OPERATOR_WORKFLOW_PLAYBACK_SCHEMA_VERSION {
            return Err("operator.workflow.schema_version.invalid");
        }
        require_operator_non_empty("suite_id", &self.suite_id)?;
        require_operator_non_empty("generated_at", &self.generated_at)?;
        require_operator_non_empty("config_hash", &self.config_hash)?;
        require_operator_non_empty("replay_command", &self.replay_command)?;
        if !is_sha256_digest(&self.config_hash) {
            return Err("operator.workflow.config_hash.invalid");
        }
        if self.environment.seed != self.seed {
            return Err("operator.workflow.environment.seed_mismatch");
        }
        if self.environment.config_hash != self.config_hash {
            return Err("operator.workflow.environment.config_hash_mismatch");
        }
        if self.environment.snapshot.redaction_note.trim().is_empty() {
            return Err("operator.workflow.environment.redaction_note.empty");
        }
        if contains_forbidden_incident_command(&self.replay_command) {
            return Err("operator.workflow.replay_command.forbidden");
        }
        if self.workflows.is_empty() {
            return Err("operator.workflow.workflows.empty");
        }
        if self.artifact_manifest.artifacts.is_empty() {
            return Err("operator.workflow.artifact_manifest.empty");
        }

        let mut artifact_ids = std::collections::BTreeSet::new();
        for artifact in &self.artifact_manifest.artifacts {
            require_operator_non_empty("artifact.artifact_id", &artifact.artifact_id)?;
            require_operator_non_empty("artifact.path", &artifact.path)?;
            require_operator_non_empty("artifact.content_type", &artifact.content_type)?;
            if !is_sha256_digest(&artifact.checksum_sha256) {
                return Err("operator.workflow.artifact.checksum.invalid");
            }
            if !artifact_ids.insert(artifact.artifact_id.as_str()) {
                return Err("operator.workflow.artifact.duplicate_id");
            }
        }

        let mut fixture_ids = std::collections::BTreeSet::new();
        let mut covered = std::collections::BTreeSet::new();
        for fixture in &self.workflows {
            validate_operator_workflow_fixture(fixture, &artifact_ids)?;
            if !fixture_ids.insert(fixture.fixture_id.as_str()) {
                return Err("operator.workflow.fixture.duplicate_id");
            }
            covered.insert(fixture.workflow);
        }

        for required in required_operator_workflows() {
            if !covered.contains(&required) {
                return Err("operator.workflow.coverage.missing");
            }
        }

        Ok(())
    }
}

impl<'de> Deserialize<'de> for OperatorWorkflowPlaybackSuite {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawOperatorWorkflowPlaybackSuite {
            schema_version: String,
            suite_id: String,
            seed: u64,
            config_hash: String,
            generated_at: String,
            environment: ReplayBundleEnvironment,
            artifact_manifest: ReplayBundleArtifactManifest,
            workflows: Vec<OperatorWorkflowPlaybackFixture>,
            replay_command: String,
        }

        let raw = RawOperatorWorkflowPlaybackSuite::deserialize(deserializer)?;
        let suite = Self {
            schema_version: raw.schema_version,
            suite_id: raw.suite_id,
            seed: raw.seed,
            config_hash: raw.config_hash,
            generated_at: raw.generated_at,
            environment: raw.environment,
            artifact_manifest: raw.artifact_manifest,
            workflows: raw.workflows,
            replay_command: raw.replay_command,
        };
        suite.validate().map_err(de::Error::custom)?;
        Ok(suite)
    }
}

/// Build the deterministic operator workflow playback fixture pack.
#[must_use]
pub fn operator_workflow_playback_suite() -> OperatorWorkflowPlaybackSuite {
    let seed = 91_337;
    let config_hash = "sha256:7777777777777777777777777777777777777777777777777777777777777777";
    let workflows = vec![
        operator_search_workflow(),
        operator_explain_workflow(),
        operator_doctor_workflow(),
        operator_audit_workflow(),
        operator_degraded_workflow(),
        operator_profile_review_workflow(),
        operator_ops_tui_workflow(),
    ];

    OperatorWorkflowPlaybackSuite {
        schema_version: OPERATOR_WORKFLOW_PLAYBACK_SCHEMA_VERSION.to_owned(),
        suite_id: "fsfs-operator-workflow-playback-v1".to_owned(),
        seed,
        config_hash: config_hash.to_owned(),
        generated_at: "2026-05-08T16:00:00Z".to_owned(),
        environment: ReplayBundleEnvironment {
            seed,
            config_hash: config_hash.to_owned(),
            snapshot: EnvSnapshot {
                variables: vec![
                    EnvEntry {
                        key: "FSFS_OPERATOR_REPLAY_SEED".to_owned(),
                        value: seed.to_string(),
                        redacted: false,
                    },
                    EnvEntry {
                        key: "FSFS_OPERATOR_REPLAY_MODE".to_owned(),
                        value: "fixture".to_owned(),
                        redacted: false,
                    },
                    EnvEntry {
                        key: "FSFS_MODEL_TOKEN".to_owned(),
                        value: "<redacted>".to_owned(),
                        redacted: true,
                    },
                ],
                redaction_note: "Operator playback fixtures contain no raw corpus text or secrets"
                    .to_owned(),
            },
        },
        artifact_manifest: ReplayBundleArtifactManifest {
            artifacts: operator_workflow_artifacts(&workflows),
        },
        workflows,
        replay_command: OPERATOR_WORKFLOW_PLAYBACK_REPLAY_COMMAND.to_owned(),
    }
}

#[must_use]
fn required_operator_workflows() -> Vec<OperatorWorkflowKind> {
    vec![
        OperatorWorkflowKind::Search,
        OperatorWorkflowKind::Explain,
        OperatorWorkflowKind::Doctor,
        OperatorWorkflowKind::Audit,
        OperatorWorkflowKind::DegradedMode,
        OperatorWorkflowKind::ProfileReview,
        OperatorWorkflowKind::OpsTui,
    ]
}

fn operator_search_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "fsfs-search-playback",
        OperatorWorkflowKind::Search,
        ReplayClientSurface::Tui,
        &["fsfs", "tui", "--screen", "search"],
        &[
            ("resize:132x40", "fsfs.search", "operator.search.resize"),
            ("key:/", "fsfs.search.query", "operator.search.focus_query"),
            (
                "text:rust async cancellation",
                "fsfs.search.query",
                "operator.search.type_query",
            ),
            (
                "key:enter",
                "fsfs.search.results",
                "operator.search.initial_results",
            ),
            (
                "key:e",
                "fsfs.search.explain",
                "operator.search.toggle_explain",
            ),
        ],
        &[
            ("query_submitted", "fsfs.search.results", "a111111111111111"),
            ("explain_open", "fsfs.search.explain", "a222222222222222"),
        ],
    )
}

fn operator_explain_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "fsfs-explain-playback",
        OperatorWorkflowKind::Explain,
        ReplayClientSurface::Cli,
        &["fsfs", "explain", "R0", "--format", "toon"],
        &[
            (
                "cmd:explain R0",
                "fsfs.explain.lookup",
                "operator.explain.lookup",
            ),
            (
                "cmd:render toon",
                "fsfs.explain.output",
                "operator.explain.render",
            ),
        ],
        &[
            (
                "policy_card",
                "fsfs.explain.policy_card",
                "b111111111111111",
            ),
            (
                "rank_movement",
                "fsfs.explain.rank_movement",
                "b222222222222222",
            ),
        ],
    )
}

fn operator_doctor_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "fsfs-doctor-playback",
        OperatorWorkflowKind::Doctor,
        ReplayClientSurface::Cli,
        &["fsfs", "doctor", "--format", "json"],
        &[
            (
                "cmd:doctor --format json",
                "fsfs.doctor.checks",
                "operator.doctor.start",
            ),
            (
                "cmd:doctor model-cache",
                "fsfs.doctor.model_cache",
                "operator.doctor.model_cache",
            ),
        ],
        &[
            ("doctor_summary", "fsfs.doctor.summary", "c111111111111111"),
            ("doctor_advice", "fsfs.doctor.advice", "c222222222222222"),
        ],
    )
}

fn operator_audit_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "fsfs-audit-playback",
        OperatorWorkflowKind::Audit,
        ReplayClientSurface::Cli,
        &["fsfs", "audit", "--format", "jsonl"],
        &[
            (
                "cmd:audit --format jsonl",
                "fsfs.audit.scan",
                "operator.audit.start",
            ),
            (
                "cmd:audit classify-staleness",
                "fsfs.audit.drift",
                "operator.audit.classify",
            ),
        ],
        &[
            ("audit_jsonl", "fsfs.audit.events", "d111111111111111"),
            ("repair_plan", "fsfs.audit.repair_plan", "d222222222222222"),
        ],
    )
}

fn operator_degraded_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "fsfs-degraded-mode-playback",
        OperatorWorkflowKind::DegradedMode,
        ReplayClientSurface::Tui,
        &["fsfs", "tui", "--profile", "degraded"],
        &[
            (
                "resize:120x32",
                "fsfs.degraded.banner",
                "operator.degraded.resize",
            ),
            (
                "key:enter",
                "fsfs.degraded.initial",
                "operator.degraded.initial_results",
            ),
            (
                "key:r",
                "fsfs.degraded.recovery",
                "operator.degraded.recovery",
            ),
        ],
        &[
            (
                "degraded_banner",
                "fsfs.degraded.banner",
                "e111111111111111",
            ),
            (
                "recovery_advice",
                "fsfs.degraded.recovery",
                "e222222222222222",
            ),
        ],
    )
}

fn operator_profile_review_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "fsfs-profile-review-playback",
        OperatorWorkflowKind::ProfileReview,
        ReplayClientSurface::Cli,
        &["fsfs", "profile", "review", "--dataset", "small"],
        &[
            (
                "cmd:profile review --dataset small",
                "fsfs.profile.summary",
                "operator.profile.start",
            ),
            (
                "cmd:profile compare-baseline",
                "fsfs.profile.drift",
                "operator.profile.compare",
            ),
        ],
        &[
            (
                "profile_summary",
                "fsfs.profile.summary",
                "f111111111111111",
            ),
            (
                "profile_drift",
                "fsfs.profile.drift_dashboard",
                "f222222222222222",
            ),
        ],
    )
}

fn operator_ops_tui_workflow() -> OperatorWorkflowPlaybackFixture {
    operator_workflow_fixture(
        "ops-tui-triage-playback",
        OperatorWorkflowKind::OpsTui,
        ReplayClientSurface::Tui,
        &["frankensearch-ops", "--deterministic", "--screen", "fleet"],
        &[
            ("resize:132x40", "ops.fleet", "operator.ops.resize"),
            ("key:t", "ops.timeline", "operator.ops.timeline"),
            ("key:g", "ops.project", "operator.ops.drilldown"),
            ("key:esc", "ops.fleet", "operator.ops.recovery"),
        ],
        &[
            ("fleet_overview", "ops.fleet", "0111111111111111"),
            ("timeline_triage", "ops.timeline", "0222222222222222"),
            ("project_drilldown", "ops.project", "0333333333333333"),
        ],
    )
}

fn operator_workflow_fixture(
    fixture_id: &str,
    workflow: OperatorWorkflowKind,
    client_surface: ReplayClientSurface,
    argv: &[&str],
    inputs: &[(&str, &str, &str)],
    snapshots: &[(&str, &str, &str)],
) -> OperatorWorkflowPlaybackFixture {
    let command = ReplayBundleCommand {
        client_surface,
        argv: argv.iter().map(|arg| (*arg).to_owned()).collect(),
        working_dir: "/data/projects/frankensearch".to_owned(),
    };
    let replay_inputs = inputs
        .iter()
        .enumerate()
        .map(
            |(index, (input, screen, reason))| OperatorWorkflowReplayInput {
                seq: u32::try_from(index).expect("operator fixture input index fits u32"),
                offset_ms: u64::try_from(index)
                    .expect("operator fixture input index fits u64")
                    .saturating_mul(25),
                input: (*input).to_owned(),
                expected_screen: (*screen).to_owned(),
                reason_code: (*reason).to_owned(),
            },
        )
        .collect();
    let snapshots = snapshots
        .iter()
        .map(|(label, screen, checksum)| OperatorWorkflowSnapshot {
            label: (*label).to_owned(),
            screen_id: (*screen).to_owned(),
            state_checksum: (*checksum).to_owned(),
            reason_code: format!("operator.snapshot.{label}"),
            artifact_refs: vec![format!("{fixture_id}-{label}")],
        })
        .collect();

    OperatorWorkflowPlaybackFixture {
        fixture_id: fixture_id.to_owned(),
        workflow,
        client_surface,
        command,
        replay_inputs,
        snapshots,
        replay_command: OPERATOR_WORKFLOW_PLAYBACK_REPLAY_COMMAND.to_owned(),
    }
}

fn operator_workflow_artifacts(
    workflows: &[OperatorWorkflowPlaybackFixture],
) -> Vec<ReplayBundleArtifactRef> {
    let mut artifacts = Vec::new();
    for (fixture_index, workflow) in workflows.iter().enumerate() {
        for (snapshot_index, snapshot) in workflow.snapshots.iter().enumerate() {
            let digit = char::from_digit(
                u32::try_from((fixture_index + snapshot_index) % 10)
                    .expect("operator fixture index fits u32"),
                10,
            )
            .expect("digit exists");
            let digest = std::iter::repeat_n(digit, 64).collect::<String>();
            let artifact_id = snapshot
                .artifact_refs
                .first()
                .expect("operator workflow snapshot has artifact ref")
                .clone();
            artifacts.push(ReplayBundleArtifactRef {
                artifact_id,
                path: format!(
                    "artifacts/operator-workflows/{}/{}.json",
                    workflow.fixture_id, snapshot.label
                ),
                content_type: "application/json".to_owned(),
                checksum_sha256: format!("sha256:{digest}"),
                required: true,
            });
        }
    }
    artifacts
}

fn validate_operator_workflow_fixture(
    fixture: &OperatorWorkflowPlaybackFixture,
    artifact_ids: &std::collections::BTreeSet<&str>,
) -> Result<(), &'static str> {
    require_operator_non_empty("fixture.fixture_id", &fixture.fixture_id)?;
    require_operator_non_empty("fixture.replay_command", &fixture.replay_command)?;
    require_operator_non_empty("command.working_dir", &fixture.command.working_dir)?;
    if fixture.command.client_surface != fixture.client_surface {
        return Err("operator.workflow.fixture.client_surface_mismatch");
    }
    if fixture.command.argv.is_empty() {
        return Err("operator.workflow.fixture.command.argv.empty");
    }
    if fixture.command.argv.iter().any(|arg| arg.trim().is_empty()) {
        return Err("operator.workflow.fixture.command.argv.blank");
    }
    if contains_forbidden_incident_command(&fixture.command.argv.join(" "))
        || contains_forbidden_incident_command(&fixture.replay_command)
    {
        return Err("operator.workflow.fixture.command.forbidden");
    }
    if fixture.replay_inputs.is_empty() {
        return Err("operator.workflow.fixture.replay_inputs.empty");
    }
    if fixture.snapshots.is_empty() {
        return Err("operator.workflow.fixture.snapshots.empty");
    }

    for (expected_seq, input) in fixture.replay_inputs.iter().enumerate() {
        if input.seq != u32::try_from(expected_seq).expect("operator fixture input index fits u32")
        {
            return Err("operator.workflow.fixture.replay_input.seq.non_monotonic");
        }
        require_operator_non_empty("replay_input.input", &input.input)?;
        require_operator_non_empty("replay_input.expected_screen", &input.expected_screen)?;
        require_operator_non_empty("replay_input.reason_code", &input.reason_code)?;
        if !input.reason_code.starts_with("operator.") {
            return Err("operator.workflow.fixture.replay_input.reason_code.invalid");
        }
    }

    for snapshot in &fixture.snapshots {
        require_operator_non_empty("snapshot.label", &snapshot.label)?;
        require_operator_non_empty("snapshot.screen_id", &snapshot.screen_id)?;
        require_operator_non_empty("snapshot.state_checksum", &snapshot.state_checksum)?;
        require_operator_non_empty("snapshot.reason_code", &snapshot.reason_code)?;
        if !is_fixed_hex(&snapshot.state_checksum, 16) {
            return Err("operator.workflow.fixture.snapshot.checksum.invalid");
        }
        if !snapshot.reason_code.starts_with("operator.snapshot.") {
            return Err("operator.workflow.fixture.snapshot.reason_code.invalid");
        }
        if snapshot.artifact_refs.is_empty() {
            return Err("operator.workflow.fixture.snapshot.artifact_refs.empty");
        }
        for artifact_ref in &snapshot.artifact_refs {
            require_operator_non_empty("snapshot.artifact_ref", artifact_ref)?;
            if !artifact_ids.contains(artifact_ref.as_str()) {
                return Err("operator.workflow.fixture.snapshot.artifact_ref.unknown");
            }
        }
    }

    if fixture.client_surface == ReplayClientSurface::Tui
        && !fixture
            .replay_inputs
            .iter()
            .any(|input| input.input.starts_with("resize:"))
    {
        return Err("operator.workflow.fixture.tui.resize_missing");
    }

    Ok(())
}

fn require_operator_non_empty(_field: &'static str, value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        return Err("operator.workflow.field.empty");
    }
    Ok(())
}

fn is_fixed_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

// ─── Degraded-Mode Synthetic Incident Suite ────────────────────────────────

/// Kind marker for degraded-mode incident suite contract definitions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DegradedIncidentSuiteContractKind {
    /// Current degraded incident suite contract-definition kind.
    #[serde(rename = "fsfs_degraded_incident_suite_contract_definition")]
    Current,
}

/// Kind marker for concrete degraded-mode incident suites.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DegradedIncidentSuiteKindMarker {
    /// Current degraded incident suite kind.
    #[serde(rename = "fsfs_degraded_incident_suite")]
    Current,
}

/// Execution lane for a deterministic incident suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedIncidentSuiteMode {
    /// Short lane with representative incidents for fast smoke validation.
    Smoke,
    /// Full lane covering every declared incident kind.
    Full,
}

/// Incident families covered by the degraded-mode synthetic suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedIncidentKind {
    /// Quality embedder exceeds the configured latency budget.
    QualityEmbedderTimeout,
    /// Quality model cache or revision is unavailable.
    ModelUnavailable,
    /// Vector artifact fails checksum/format validation.
    CorruptVectorArtifact,
    /// Lexical backend cannot produce the fast-tier candidate set.
    LexicalBackendFailure,
    /// Storage/catalog lock pressure prevents normal metadata access.
    StorageLockPressure,
    /// Watcher backlog exceeds the deterministic catch-up budget.
    WatcherBacklog,
}

/// Expected externally visible output for a synthetic incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedIncidentExpectedOutput {
    /// Incident must still emit a fast initial search phase.
    SearchPhaseInitial,
    /// Incident must emit a refinement-failed search phase.
    SearchPhaseRefinementFailed,
    /// Incident is asserted through a doctor reason code.
    DoctorReasonCode,
    /// Incident is asserted through an audit reason code.
    AuditReasonCode,
    /// Incident is asserted through a watcher reason code.
    WatcherReasonCode,
}

/// Network posture for deterministic incident replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradedIncidentNetworkPolicy {
    /// Suite must not require real network access.
    OfflineOnly,
}

/// Deterministic trigger details for one synthetic incident.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegradedIncidentInjection {
    /// Stable trigger reason code.
    pub reason_code: String,
    /// Human-readable trigger summary.
    pub trigger: String,
    /// Deterministic payload or fixture selector used by replay.
    pub deterministic_payload: String,
}

/// Expected outcome for one degraded incident.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegradedIncidentExpectedOutcome {
    /// Output surface where the reason code must appear.
    pub output: DegradedIncidentExpectedOutput,
    /// Expected terminal status for the incident.
    pub status: ReplayBundleOutcomeStatus,
    /// Canonical reason code expected in search/doctor/audit output.
    pub reason_code: String,
    /// Whether initial results are still safe to display.
    pub preserves_initial_results: bool,
    /// Degradation stage expected after this incident is observed.
    pub degradation_stage: crate::pressure::DegradationStage,
    /// Artifact identifiers that must be produced for this scenario.
    pub artifact_refs: Vec<String>,
}

/// Structured-log requirements for incident replay evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegradedIncidentLogSpec {
    /// Event schema name used by emitted JSONL evidence.
    pub event_schema: String,
    /// Required event fields. Must include seed, config hash, scenario ID, and reason code.
    pub required_fields: Vec<String>,
    /// Redaction note for emitted evidence.
    pub redaction_note: String,
}

/// One deterministic degraded-mode incident scenario.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegradedIncidentScenario {
    /// Stable scenario identifier.
    pub scenario_id: String,
    /// Incident family under test.
    pub incident: DegradedIncidentKind,
    /// Deterministic failure injection.
    pub injection: DegradedIncidentInjection,
    /// Expected externally visible outcome.
    pub expected: DegradedIncidentExpectedOutcome,
    /// Replay command for this scenario.
    pub replay_command: String,
}

/// Machine-readable degraded-mode incident suite contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DegradedIncidentSuiteContractDefinition {
    /// Contract definition kind tag.
    pub kind: DegradedIncidentSuiteContractKind,
    /// Schema version.
    pub v: u8,
    /// Incident kinds that the full suite must cover.
    pub required_incidents: Vec<DegradedIncidentKind>,
    /// Execution modes supported by the validator.
    pub supported_modes: Vec<DegradedIncidentSuiteMode>,
    /// Required top-level suite fields.
    pub required_suite_fields: Vec<String>,
    /// Required scenario fields.
    pub required_scenario_fields: Vec<String>,
    /// Required structured-log fields.
    pub required_log_fields: Vec<String>,
    /// Validation lanes that must accept/reject suite fixtures.
    pub validation_modes: Vec<String>,
}

impl Default for DegradedIncidentSuiteContractDefinition {
    fn default() -> Self {
        Self {
            kind: DegradedIncidentSuiteContractKind::Current,
            v: DEGRADED_INCIDENT_SUITE_SCHEMA_VERSION,
            required_incidents: all_degraded_incident_kinds(),
            supported_modes: vec![
                DegradedIncidentSuiteMode::Smoke,
                DegradedIncidentSuiteMode::Full,
            ],
            required_suite_fields: vec![
                "kind".to_owned(),
                "v".to_owned(),
                "suite_id".to_owned(),
                "mode".to_owned(),
                "seed".to_owned(),
                "config_hash".to_owned(),
                "generated_at".to_owned(),
                "command".to_owned(),
                "environment".to_owned(),
                "network_policy".to_owned(),
                "destructive_actions_allowed".to_owned(),
                "structured_log".to_owned(),
                "artifact_manifest".to_owned(),
                "scenarios".to_owned(),
                "replay_command".to_owned(),
            ],
            required_scenario_fields: vec![
                "scenario_id".to_owned(),
                "incident".to_owned(),
                "injection".to_owned(),
                "expected".to_owned(),
                "replay_command".to_owned(),
            ],
            required_log_fields: vec![
                "seed".to_owned(),
                "config_hash".to_owned(),
                "scenario_id".to_owned(),
                "reason_code".to_owned(),
            ],
            validation_modes: vec![
                "json_schema_contract".to_owned(),
                "json_schema_smoke".to_owned(),
                "json_schema_full".to_owned(),
                "json_schema_invalid".to_owned(),
                "rust_deserialize_roundtrip".to_owned(),
                "script_smoke_full_modes".to_owned(),
            ],
        }
    }
}

/// Concrete deterministic degraded-mode incident suite.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DegradedIncidentSuite {
    /// Suite kind tag.
    pub kind: DegradedIncidentSuiteKindMarker,
    /// Schema version.
    pub v: u8,
    /// Stable suite identifier.
    pub suite_id: String,
    /// Execution mode.
    pub mode: DegradedIncidentSuiteMode,
    /// Deterministic seed for every scenario.
    pub seed: u64,
    /// Hash of the resolved fsfs configuration used by the suite.
    pub config_hash: String,
    /// RFC 3339 UTC timestamp for fixture creation.
    pub generated_at: String,
    /// Canonical command invocation for the suite.
    pub command: ReplayBundleCommand,
    /// Deterministic environment identity.
    pub environment: ReplayBundleEnvironment,
    /// Network policy; must be offline-only.
    pub network_policy: DegradedIncidentNetworkPolicy,
    /// Must remain false. Suites are validators, not cleanup tools.
    pub destructive_actions_allowed: bool,
    /// Structured log contract for emitted JSONL evidence.
    pub structured_log: DegradedIncidentLogSpec,
    /// Artifact manifest for scenario evidence.
    pub artifact_manifest: ReplayBundleArtifactManifest,
    /// Deterministic incident scenarios.
    pub scenarios: Vec<DegradedIncidentScenario>,
    /// Replay command for the entire suite.
    pub replay_command: String,
}

impl DegradedIncidentSuite {
    /// Validate degraded incident suite invariants beyond raw JSON shape.
    ///
    /// # Errors
    ///
    /// Returns a stable reason code when the suite contract is violated.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.v != DEGRADED_INCIDENT_SUITE_SCHEMA_VERSION {
            return Err("degraded.incident_suite.version.invalid");
        }
        require_incident_non_empty("suite_id", &self.suite_id)?;
        require_incident_non_empty("generated_at", &self.generated_at)?;
        require_incident_non_empty("config_hash", &self.config_hash)?;
        if !is_sha256_digest(&self.config_hash) {
            return Err("degraded.incident_suite.config_hash.invalid");
        }
        if self.environment.seed != self.seed {
            return Err("degraded.incident_suite.environment.seed_mismatch");
        }
        if self.environment.config_hash != self.config_hash {
            return Err("degraded.incident_suite.environment.config_hash_mismatch");
        }
        if self.environment.snapshot.redaction_note.trim().is_empty() {
            return Err("degraded.incident_suite.environment.redaction_note.empty");
        }
        if self.destructive_actions_allowed {
            return Err("degraded.incident_suite.destructive_actions.enabled");
        }
        if self.command.argv.is_empty() {
            return Err("degraded.incident_suite.command.argv.empty");
        }
        if self.command.argv.iter().any(|arg| arg.trim().is_empty()) {
            return Err("degraded.incident_suite.command.argv.blank");
        }
        require_incident_non_empty("command.working_dir", &self.command.working_dir)?;
        require_incident_non_empty(
            "structured_log.event_schema",
            &self.structured_log.event_schema,
        )?;
        require_incident_non_empty(
            "structured_log.redaction_note",
            &self.structured_log.redaction_note,
        )?;
        require_incident_non_empty("replay_command", &self.replay_command)?;
        if contains_forbidden_incident_command(&self.command.argv.join(" "))
            || contains_forbidden_incident_command(&self.replay_command)
        {
            return Err("degraded.incident_suite.command.forbidden");
        }
        for field in ["seed", "config_hash", "scenario_id", "reason_code"] {
            if !self
                .structured_log
                .required_fields
                .iter()
                .any(|required| required == field)
            {
                return Err("degraded.incident_suite.structured_log.required_field.missing");
            }
        }
        if self.artifact_manifest.artifacts.is_empty() {
            return Err("degraded.incident_suite.artifact_manifest.empty");
        }
        if self.scenarios.is_empty() {
            return Err("degraded.incident_suite.scenarios.empty");
        }

        let mut artifact_ids = std::collections::BTreeSet::new();
        for artifact in &self.artifact_manifest.artifacts {
            require_incident_non_empty("artifact.artifact_id", &artifact.artifact_id)?;
            require_incident_non_empty("artifact.path", &artifact.path)?;
            require_incident_non_empty("artifact.content_type", &artifact.content_type)?;
            if !is_sha256_digest(&artifact.checksum_sha256) {
                return Err("degraded.incident_suite.artifact.checksum.invalid");
            }
            if !artifact_ids.insert(artifact.artifact_id.as_str()) {
                return Err("degraded.incident_suite.artifact_manifest.duplicate_id");
            }
        }

        let mut scenario_ids = std::collections::BTreeSet::new();
        let mut incidents = std::collections::BTreeSet::new();
        for scenario in &self.scenarios {
            validate_degraded_incident_scenario(scenario, &artifact_ids)?;
            if !scenario_ids.insert(scenario.scenario_id.as_str()) {
                return Err("degraded.incident_suite.scenario.duplicate_id");
            }
            incidents.insert(scenario.incident);
        }

        if matches!(self.mode, DegradedIncidentSuiteMode::Full) {
            for incident in all_degraded_incident_kinds() {
                if !incidents.contains(&incident) {
                    return Err("degraded.incident_suite.full.coverage.missing");
                }
            }
        }

        Ok(())
    }
}

impl<'de> Deserialize<'de> for DegradedIncidentSuite {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawDegradedIncidentSuite {
            kind: DegradedIncidentSuiteKindMarker,
            v: u8,
            suite_id: String,
            mode: DegradedIncidentSuiteMode,
            seed: u64,
            config_hash: String,
            generated_at: String,
            command: ReplayBundleCommand,
            environment: ReplayBundleEnvironment,
            network_policy: DegradedIncidentNetworkPolicy,
            destructive_actions_allowed: bool,
            structured_log: DegradedIncidentLogSpec,
            artifact_manifest: ReplayBundleArtifactManifest,
            scenarios: Vec<DegradedIncidentScenario>,
            replay_command: String,
        }

        let raw = RawDegradedIncidentSuite::deserialize(deserializer)?;
        let suite = Self {
            kind: raw.kind,
            v: raw.v,
            suite_id: raw.suite_id,
            mode: raw.mode,
            seed: raw.seed,
            config_hash: raw.config_hash,
            generated_at: raw.generated_at,
            command: raw.command,
            environment: raw.environment,
            network_policy: raw.network_policy,
            destructive_actions_allowed: raw.destructive_actions_allowed,
            structured_log: raw.structured_log,
            artifact_manifest: raw.artifact_manifest,
            scenarios: raw.scenarios,
            replay_command: raw.replay_command,
        };
        suite.validate().map_err(de::Error::custom)?;
        Ok(suite)
    }
}

/// Build the canonical degraded incident suite contract definition.
#[must_use]
pub fn degraded_incident_suite_contract_definition() -> DegradedIncidentSuiteContractDefinition {
    DegradedIncidentSuiteContractDefinition::default()
}

/// Build the deterministic smoke-mode degraded incident suite fixture.
#[must_use]
pub fn degraded_incident_smoke_suite() -> DegradedIncidentSuite {
    degraded_incident_suite(
        DegradedIncidentSuiteMode::Smoke,
        "fsfs-degraded-incident-smoke-v1",
        vec![
            incident_quality_embedder_timeout(),
            incident_corrupt_vector_artifact(),
        ],
    )
}

/// Build the deterministic full-mode degraded incident suite fixture.
#[must_use]
pub fn degraded_incident_full_suite() -> DegradedIncidentSuite {
    degraded_incident_suite(
        DegradedIncidentSuiteMode::Full,
        "fsfs-degraded-incident-full-v1",
        vec![
            incident_quality_embedder_timeout(),
            incident_model_unavailable(),
            incident_corrupt_vector_artifact(),
            incident_lexical_backend_failure(),
            incident_storage_lock_pressure(),
            incident_watcher_backlog(),
        ],
    )
}

#[must_use]
fn all_degraded_incident_kinds() -> Vec<DegradedIncidentKind> {
    vec![
        DegradedIncidentKind::QualityEmbedderTimeout,
        DegradedIncidentKind::ModelUnavailable,
        DegradedIncidentKind::CorruptVectorArtifact,
        DegradedIncidentKind::LexicalBackendFailure,
        DegradedIncidentKind::StorageLockPressure,
        DegradedIncidentKind::WatcherBacklog,
    ]
}

fn degraded_incident_suite(
    mode: DegradedIncidentSuiteMode,
    suite_id: &str,
    scenarios: Vec<DegradedIncidentScenario>,
) -> DegradedIncidentSuite {
    let seed = 424_242;
    let config_hash = "sha256:1212121212121212121212121212121212121212121212121212121212121212";
    DegradedIncidentSuite {
        kind: DegradedIncidentSuiteKindMarker::Current,
        v: DEGRADED_INCIDENT_SUITE_SCHEMA_VERSION,
        suite_id: suite_id.to_owned(),
        mode,
        seed,
        config_hash: config_hash.to_owned(),
        generated_at: "2026-05-08T12:00:00Z".to_owned(),
        command: ReplayBundleCommand {
            client_surface: ReplayClientSurface::Cli,
            argv: vec![
                "fsfs".to_owned(),
                "incident-suite".to_owned(),
                "--mode".to_owned(),
                mode_arg(mode).to_owned(),
                "--seed".to_owned(),
                seed.to_string(),
                "--format".to_owned(),
                "jsonl".to_owned(),
            ],
            working_dir: "/data/projects/frankensearch".to_owned(),
        },
        environment: ReplayBundleEnvironment {
            seed,
            config_hash: config_hash.to_owned(),
            snapshot: EnvSnapshot {
                variables: vec![
                    EnvEntry {
                        key: "FSFS_INCIDENT_SUITE_MODE".to_owned(),
                        value: mode_arg(mode).to_owned(),
                        redacted: false,
                    },
                    EnvEntry {
                        key: "FSFS_INCIDENT_SUITE_SEED".to_owned(),
                        value: seed.to_string(),
                        redacted: false,
                    },
                    EnvEntry {
                        key: "RUST_LOG".to_owned(),
                        value: "info".to_owned(),
                        redacted: false,
                    },
                ],
                redaction_note: "Synthetic suite uses deterministic offline fixtures only"
                    .to_owned(),
            },
        },
        network_policy: DegradedIncidentNetworkPolicy::OfflineOnly,
        destructive_actions_allowed: false,
        structured_log: DegradedIncidentLogSpec {
            event_schema: "fsfs.degraded_incident.event.v1".to_owned(),
            required_fields: vec![
                "seed".to_owned(),
                "config_hash".to_owned(),
                "suite_id".to_owned(),
                "scenario_id".to_owned(),
                "incident".to_owned(),
                "reason_code".to_owned(),
                "expected_output".to_owned(),
            ],
            redaction_note: "No raw corpus text or host secrets are emitted".to_owned(),
        },
        artifact_manifest: ReplayBundleArtifactManifest {
            artifacts: scenario_artifacts(&scenarios),
        },
        scenarios,
        replay_command: format!(
            "fsfs incident-suite --mode {} --seed {seed} --format jsonl",
            mode_arg(mode)
        ),
    }
}

#[must_use]
const fn mode_arg(mode: DegradedIncidentSuiteMode) -> &'static str {
    match mode {
        DegradedIncidentSuiteMode::Smoke => "smoke",
        DegradedIncidentSuiteMode::Full => "full",
    }
}

fn incident_quality_embedder_timeout() -> DegradedIncidentScenario {
    scenario(
        "incident-quality-embedder-timeout",
        DegradedIncidentKind::QualityEmbedderTimeout,
        "incident.inject.quality_timeout",
        "quality embedder exceeds 150 ms budget",
        "quality_timeout_ms=151,budget_ms=150",
        DegradedIncidentExpectedOutput::SearchPhaseRefinementFailed,
        ReplayBundleOutcomeStatus::Degraded,
        "degrade.advice.timeout",
        true,
        crate::pressure::DegradationStage::EmbedDeferred,
    )
}

fn incident_model_unavailable() -> DegradedIncidentScenario {
    scenario(
        "incident-model-unavailable",
        DegradedIncidentKind::ModelUnavailable,
        "incident.inject.model_unavailable",
        "quality model cache lookup returns unavailable",
        "model_revision=fixture-missing,download=disabled",
        DegradedIncidentExpectedOutput::SearchPhaseRefinementFailed,
        ReplayBundleOutcomeStatus::Degraded,
        "degrade.advice.quality_model_missing",
        true,
        crate::pressure::DegradationStage::EmbedDeferred,
    )
}

fn incident_corrupt_vector_artifact() -> DegradedIncidentScenario {
    scenario(
        "incident-corrupt-vector-artifact",
        DegradedIncidentKind::CorruptVectorArtifact,
        "incident.inject.corrupt_vector_artifact",
        "vector artifact header checksum mismatch",
        "artifact=quality.fsvi,checksum=bad_magic",
        DegradedIncidentExpectedOutput::DoctorReasonCode,
        ReplayBundleOutcomeStatus::Failed,
        "degrade.advice.index_corrupt",
        false,
        crate::pressure::DegradationStage::LexicalOnly,
    )
}

fn incident_lexical_backend_failure() -> DegradedIncidentScenario {
    scenario(
        "incident-lexical-backend-failure",
        DegradedIncidentKind::LexicalBackendFailure,
        "incident.inject.lexical_backend_failure",
        "lexical backend returns deterministic query parser failure",
        "tantivy_query=unterminated_phrase",
        DegradedIncidentExpectedOutput::SearchPhaseRefinementFailed,
        ReplayBundleOutcomeStatus::Failed,
        "fsfs.incident.lexical_backend_failure",
        false,
        crate::pressure::DegradationStage::MetadataOnly,
    )
}

fn incident_storage_lock_pressure() -> DegradedIncidentScenario {
    scenario(
        "incident-storage-lock-pressure",
        DegradedIncidentKind::StorageLockPressure,
        "incident.inject.storage_lock_pressure",
        "catalog write lock exceeds deterministic wait budget",
        "lock_wait_ms=250,budget_ms=50",
        DegradedIncidentExpectedOutput::AuditReasonCode,
        ReplayBundleOutcomeStatus::Degraded,
        "fsfs.incident.storage_lock_pressure",
        true,
        crate::pressure::DegradationStage::LexicalOnly,
    )
}

fn incident_watcher_backlog() -> DegradedIncidentScenario {
    scenario(
        "incident-watcher-backlog",
        DegradedIncidentKind::WatcherBacklog,
        "incident.inject.watcher_backlog",
        "watcher backlog crosses catch-up watermark",
        "pending_events=4096,watermark=1024",
        DegradedIncidentExpectedOutput::WatcherReasonCode,
        ReplayBundleOutcomeStatus::Degraded,
        "fsfs.incident.watcher_backlog",
        true,
        crate::pressure::DegradationStage::EmbedDeferred,
    )
}

#[allow(clippy::too_many_arguments)]
fn scenario(
    scenario_id: &str,
    incident: DegradedIncidentKind,
    injection_reason_code: &str,
    trigger: &str,
    deterministic_payload: &str,
    output: DegradedIncidentExpectedOutput,
    status: ReplayBundleOutcomeStatus,
    expected_reason_code: &str,
    preserves_initial_results: bool,
    degradation_stage: crate::pressure::DegradationStage,
) -> DegradedIncidentScenario {
    DegradedIncidentScenario {
        scenario_id: scenario_id.to_owned(),
        incident,
        injection: DegradedIncidentInjection {
            reason_code: injection_reason_code.to_owned(),
            trigger: trigger.to_owned(),
            deterministic_payload: deterministic_payload.to_owned(),
        },
        expected: DegradedIncidentExpectedOutcome {
            output,
            status,
            reason_code: expected_reason_code.to_owned(),
            preserves_initial_results,
            degradation_stage,
            artifact_refs: vec![format!("{scenario_id}-events")],
        },
        replay_command: format!(
            "fsfs incident-suite replay --scenario {scenario_id} --format jsonl"
        ),
    }
}

fn scenario_artifacts(scenarios: &[DegradedIncidentScenario]) -> Vec<ReplayBundleArtifactRef> {
    scenarios
        .iter()
        .enumerate()
        .map(|(index, scenario)| ReplayBundleArtifactRef {
            artifact_id: format!("{}-events", scenario.scenario_id),
            path: format!("artifacts/incidents/{}/events.jsonl", scenario.scenario_id),
            content_type: "application/x-ndjson".to_owned(),
            checksum_sha256: fixture_digest(index),
            required: true,
        })
        .collect()
}

fn fixture_digest(index: usize) -> String {
    let digit = match index {
        0 => '1',
        1 => '2',
        2 => '3',
        3 => '4',
        4 => '5',
        _ => '6',
    };
    let digest = std::iter::repeat_n(digit, 64).collect::<String>();
    format!("sha256:{digest}")
}

fn validate_degraded_incident_scenario(
    scenario: &DegradedIncidentScenario,
    artifact_ids: &std::collections::BTreeSet<&str>,
) -> Result<(), &'static str> {
    require_incident_non_empty("scenario.scenario_id", &scenario.scenario_id)?;
    require_incident_non_empty(
        "scenario.injection.reason_code",
        &scenario.injection.reason_code,
    )?;
    require_incident_non_empty("scenario.injection.trigger", &scenario.injection.trigger)?;
    require_incident_non_empty(
        "scenario.injection.deterministic_payload",
        &scenario.injection.deterministic_payload,
    )?;
    require_incident_non_empty(
        "scenario.expected.reason_code",
        &scenario.expected.reason_code,
    )?;
    require_incident_non_empty("scenario.replay_command", &scenario.replay_command)?;
    if contains_forbidden_incident_command(&scenario.replay_command) {
        return Err("degraded.incident_suite.scenario.replay_command.forbidden");
    }
    if scenario.expected.artifact_refs.is_empty() {
        return Err("degraded.incident_suite.scenario.artifact_refs.empty");
    }
    for artifact_ref in &scenario.expected.artifact_refs {
        require_incident_non_empty("scenario.expected.artifact_ref", artifact_ref)?;
        if !artifact_ids.contains(artifact_ref.as_str()) {
            return Err("degraded.incident_suite.scenario.artifact_ref.unknown");
        }
    }
    if matches!(
        scenario.expected.status,
        ReplayBundleOutcomeStatus::Degraded
            | ReplayBundleOutcomeStatus::Failed
            | ReplayBundleOutcomeStatus::Skipped
    ) && scenario.expected.reason_code.trim().is_empty()
    {
        return Err("degraded.incident_suite.scenario.reason_code.missing");
    }
    Ok(())
}

fn require_incident_non_empty(field: &'static str, value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        match field {
            "suite_id" => Err("degraded.incident_suite.suite_id.empty"),
            "generated_at" => Err("degraded.incident_suite.generated_at.empty"),
            "config_hash" => Err("degraded.incident_suite.config_hash.empty"),
            "command.working_dir" => Err("degraded.incident_suite.command.working_dir.empty"),
            "structured_log.event_schema" => {
                Err("degraded.incident_suite.structured_log.event_schema.empty")
            }
            "structured_log.redaction_note" => {
                Err("degraded.incident_suite.structured_log.redaction_note.empty")
            }
            "replay_command" => Err("degraded.incident_suite.replay_command.empty"),
            "artifact.artifact_id" => Err("degraded.incident_suite.artifact.artifact_id.empty"),
            "artifact.path" => Err("degraded.incident_suite.artifact.path.empty"),
            "artifact.content_type" => Err("degraded.incident_suite.artifact.content_type.empty"),
            "scenario.scenario_id" => Err("degraded.incident_suite.scenario.scenario_id.empty"),
            "scenario.injection.reason_code" => {
                Err("degraded.incident_suite.scenario.injection.reason_code.empty")
            }
            "scenario.injection.trigger" => {
                Err("degraded.incident_suite.scenario.injection.trigger.empty")
            }
            "scenario.injection.deterministic_payload" => {
                Err("degraded.incident_suite.scenario.injection.payload.empty")
            }
            "scenario.expected.reason_code" => {
                Err("degraded.incident_suite.scenario.reason_code.missing")
            }
            "scenario.replay_command" => {
                Err("degraded.incident_suite.scenario.replay_command.empty")
            }
            "scenario.expected.artifact_ref" => {
                Err("degraded.incident_suite.scenario.artifact_ref.empty")
            }
            _ => Err("degraded.incident_suite.field.empty"),
        }
    } else {
        Ok(())
    }
}

#[must_use]
fn contains_forbidden_incident_command(command: &str) -> bool {
    let command = command.to_ascii_lowercase();
    [
        "rm -rf",
        "git clean -fd",
        "git reset --hard",
        "curl ",
        "wget ",
        "http://",
        "https://",
    ]
    .iter()
    .any(|token| command.contains(token))
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), &'static str> {
    if value.trim().is_empty() {
        match field {
            "bundle_id" => Err("replay.bundle.bundle_id.empty"),
            "scenario_id" => Err("replay.bundle.scenario_id.empty"),
            "created_at" => Err("replay.bundle.created_at.empty"),
            "command.working_dir" => Err("replay.bundle.command.working_dir.empty"),
            "environment.config_hash" => Err("replay.bundle.environment.config_hash.empty"),
            "fixture.fixture_id" => Err("replay.bundle.fixture.fixture_id.empty"),
            "fixture.path" => Err("replay.bundle.fixture.path.empty"),
            "artifact.artifact_id" => Err("replay.bundle.artifact.artifact_id.empty"),
            "artifact.path" => Err("replay.bundle.artifact.path.empty"),
            "artifact.content_type" => Err("replay.bundle.artifact.content_type.empty"),
            "expected_phase.artifact_ref" => Err("replay.bundle.expected_phase.artifact_ref.empty"),
            "environment.snapshot.key" => Err("replay.bundle.environment.snapshot.key.empty"),
            _ => Err("replay.bundle.field.empty"),
        }
    } else {
        Ok(())
    }
}

#[must_use]
fn is_sha256_digest(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|digest| digest.len() == 64 && digest.chars().all(is_lower_hex_digit))
}

#[must_use]
const fn is_lower_hex_digit(ch: char) -> bool {
    matches!(ch, '0'..='9' | 'a'..='f')
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
    capture_env_snapshot(
        std::env::vars_os()
            .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?))),
    )
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
    fn retention_inverted_hot_warm_still_reachable() {
        // Deserialize can set hot_days > warm_days, which would make Warm
        // unreachable without the effective_warm clamp.
        let policy = RetentionPolicy {
            hot_days: 100,
            warm_days: 50,
            cold_max_days: 0,
        };
        // Below hot_days → Hot.
        assert_eq!(policy.tier_for_age(50), Some(RetentionTier::Hot));
        assert_eq!(policy.tier_for_age(100), Some(RetentionTier::Hot));
        // Above hot_days → effective_warm is clamped to hot_days, so Warm
        // tier has zero width; jump straight to Cold (keep-forever).
        assert_eq!(policy.tier_for_age(101), Some(RetentionTier::Cold));
    }

    #[test]
    fn retention_inverted_cold_less_than_warm() {
        // cold_max_days < warm_days is contradictory; the Cold tier gets
        // clamped to zero-width (effective_cold = effective_warm), so
        // anything beyond warm_days is immediately expired.
        let policy = RetentionPolicy {
            hot_days: 7,
            warm_days: 90,
            cold_max_days: 30, // less than warm_days
        };
        assert_eq!(policy.tier_for_age(7), Some(RetentionTier::Hot));
        assert_eq!(policy.tier_for_age(50), Some(RetentionTier::Warm));
        assert_eq!(policy.tier_for_age(90), Some(RetentionTier::Warm));
        // effective_cold = effective_warm = 90, so 91 → expired.
        assert_eq!(policy.tier_for_age(91), None);
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

    // ─── Canonical Replay Bundle Contract ───────────────────────────────

    fn valid_replay_bundle_manifest() -> ReplayBundleManifest {
        ReplayBundleManifest {
            kind: ReplayBundleManifestKind::Current,
            v: ReplayBundleSchemaVersion1,
            bundle_id: "bundle-search-0001".to_owned(),
            scenario_id: "search-basic-refine".to_owned(),
            scenario_kind: ReplayBundleScenarioKind::Search,
            created_at: "2026-05-08T11:00:00Z".to_owned(),
            command: ReplayBundleCommand {
                client_surface: ReplayClientSurface::Cli,
                argv: vec![
                    "fsfs".to_owned(),
                    "search".to_owned(),
                    "--query".to_owned(),
                    "hybrid search".to_owned(),
                    "--replay-bundle".to_owned(),
                    "artifacts/replay/bundle-search-0001".to_owned(),
                ],
                working_dir: "/data/projects/frankensearch".to_owned(),
            },
            environment: ReplayBundleEnvironment {
                seed: 42,
                config_hash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_owned(),
                snapshot: EnvSnapshot {
                    variables: vec![EnvEntry {
                        key: "FSFS_REPLAY_SEED".to_owned(),
                        value: "42".to_owned(),
                        redacted: false,
                    }],
                    redaction_note: "Sensitive values redacted by key policy".to_owned(),
                },
            },
            fixture_refs: vec![ReplayBundleFixtureRef {
                fixture_id: "query-fixture".to_owned(),
                path: "schemas/fixtures/search-query-basic.json".to_owned(),
                checksum_sha256:
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                        .to_owned(),
            }],
            expected_phase_outcomes: vec![
                ReplayBundleExpectedPhaseOutcome {
                    phase: ReplayBundlePhase::Initial,
                    status: ReplayBundleOutcomeStatus::Succeeded,
                    reason_code: None,
                    artifact_refs: vec!["initial-results".to_owned()],
                },
                ReplayBundleExpectedPhaseOutcome {
                    phase: ReplayBundlePhase::Refined,
                    status: ReplayBundleOutcomeStatus::Degraded,
                    reason_code: Some("fsfs.replay.quality_model_unavailable".to_owned()),
                    artifact_refs: vec!["refined-results".to_owned()],
                },
            ],
            artifact_manifest: ReplayBundleArtifactManifest {
                artifacts: vec![
                    ReplayBundleArtifactRef {
                        artifact_id: "initial-results".to_owned(),
                        path: "artifacts/replay/bundle-search-0001/initial.json".to_owned(),
                        content_type: "application/json".to_owned(),
                        checksum_sha256:
                            "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                                .to_owned(),
                        required: true,
                    },
                    ReplayBundleArtifactRef {
                        artifact_id: "refined-results".to_owned(),
                        path: "artifacts/replay/bundle-search-0001/refined.json".to_owned(),
                        content_type: "application/json".to_owned(),
                        checksum_sha256:
                            "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                                .to_owned(),
                        required: true,
                    },
                ],
            },
        }
    }

    #[test]
    fn replay_bundle_contract_default_covers_required_surfaces() {
        let contract = ReplayBundleContractDefinition::default();

        assert_eq!(contract.kind, ReplayBundleContractKind::Current);
        assert!(
            contract
                .supported_scenarios
                .contains(&ReplayBundleScenarioKind::DegradedMode)
        );
        assert!(
            contract
                .required_top_level_fields
                .contains(&"command".to_owned())
        );
        assert!(
            contract
                .required_environment_fields
                .contains(&"seed".to_owned())
        );
        assert!(
            contract
                .required_artifact_fields
                .contains(&"checksum_sha256".to_owned())
        );
    }

    #[test]
    fn replay_bundle_manifest_roundtrips_and_validates() {
        let manifest = valid_replay_bundle_manifest();

        manifest.validate().expect("valid replay bundle manifest");
        let json = serde_json::to_string(&manifest).expect("serialize replay bundle manifest");
        let decoded: ReplayBundleManifest =
            serde_json::from_str(&json).expect("deserialize replay bundle manifest");

        assert_eq!(decoded, manifest);
    }

    #[test]
    fn replay_bundle_manifest_rejects_missing_reason_for_degraded_phase() {
        let mut manifest = valid_replay_bundle_manifest();
        for outcome in &mut manifest.expected_phase_outcomes {
            if matches!(outcome.phase, ReplayBundlePhase::Refined) {
                outcome.reason_code = None;
            }
        }
        let json = serde_json::to_value(&manifest).expect("serialize manifest");
        let error = serde_json::from_value::<ReplayBundleManifest>(json)
            .expect_err("reject degraded phase without reason");

        assert!(error.to_string().contains("reason_code"));
    }

    #[test]
    fn replay_bundle_manifest_rejects_unknown_artifact_refs() {
        let mut manifest = valid_replay_bundle_manifest();
        for outcome in &mut manifest.expected_phase_outcomes {
            if matches!(outcome.phase, ReplayBundlePhase::Initial) {
                outcome.artifact_refs.push("missing-artifact".to_owned());
            }
        }
        let json = serde_json::to_value(&manifest).expect("serialize manifest");
        let error = serde_json::from_value::<ReplayBundleManifest>(json)
            .expect_err("reject unknown artifact ref");

        assert!(error.to_string().contains("artifact_ref.unknown"));
    }

    // ─── Operator Workflow Playback Fixtures ──────────────────────────

    #[test]
    fn operator_workflow_playback_suite_covers_required_workflows() {
        let suite = operator_workflow_playback_suite();

        suite
            .validate()
            .expect("operator workflow playback suite should validate");
        assert_eq!(
            suite.schema_version,
            OPERATOR_WORKFLOW_PLAYBACK_SCHEMA_VERSION
        );
        assert_eq!(
            suite.replay_command,
            OPERATOR_WORKFLOW_PLAYBACK_REPLAY_COMMAND
        );

        let covered: std::collections::BTreeSet<_> =
            suite.workflows.iter().map(|item| item.workflow).collect();
        assert_eq!(covered, required_operator_workflows().into_iter().collect());
        assert!(
            suite
                .workflows
                .iter()
                .any(|workflow| workflow.client_surface == ReplayClientSurface::Tui)
        );
        assert!(
            suite
                .workflows
                .iter()
                .any(|workflow| workflow.client_surface == ReplayClientSurface::Cli)
        );
    }

    #[test]
    fn operator_workflow_playback_navigation_and_outputs_are_replay_safe() {
        let suite = operator_workflow_playback_suite();
        let artifact_ids: std::collections::BTreeSet<_> = suite
            .artifact_manifest
            .artifacts
            .iter()
            .map(|artifact| artifact.artifact_id.as_str())
            .collect();

        for workflow in &suite.workflows {
            let mut expected_seq = 0_u32;
            let mut previous_offset = 0_u64;
            for input in &workflow.replay_inputs {
                assert_eq!(input.seq, expected_seq);
                assert!(input.offset_ms >= previous_offset);
                assert!(input.reason_code.starts_with("operator."));
                assert!(!input.expected_screen.trim().is_empty());
                expected_seq = expected_seq.saturating_add(1);
                previous_offset = input.offset_ms;
            }

            for snapshot in &workflow.snapshots {
                assert!(is_fixed_hex(&snapshot.state_checksum, 16));
                assert!(snapshot.reason_code.starts_with("operator.snapshot."));
                assert!(
                    snapshot
                        .artifact_refs
                        .iter()
                        .all(|artifact_ref| artifact_ids.contains(artifact_ref.as_str()))
                );
            }

            if workflow.client_surface == ReplayClientSurface::Tui {
                assert!(
                    workflow
                        .replay_inputs
                        .iter()
                        .any(|input| input.input.starts_with("resize:")),
                    "TUI workflow {} must pin viewport size",
                    workflow.fixture_id
                );
            }
        }
    }

    #[test]
    fn operator_workflow_playback_suite_roundtrips_through_json() {
        let suite = operator_workflow_playback_suite();
        let json = serde_json::to_string(&suite).expect("serialize operator workflow suite");
        let decoded: OperatorWorkflowPlaybackSuite =
            serde_json::from_str(&json).expect("deserialize operator workflow suite");

        assert_eq!(decoded, suite);
    }

    #[test]
    fn operator_workflow_playback_suite_rejects_unknown_snapshot_artifact_refs() {
        let mut suite = operator_workflow_playback_suite();
        suite
            .workflows
            .first_mut()
            .expect("suite has workflows")
            .snapshots
            .first_mut()
            .expect("workflow has snapshots")
            .artifact_refs
            .push("missing-artifact".to_owned());

        let json = serde_json::to_value(&suite).expect("serialize operator workflow suite");
        let error = serde_json::from_value::<OperatorWorkflowPlaybackSuite>(json)
            .expect_err("unknown snapshot artifact ref should fail validation");

        assert!(error.to_string().contains("artifact_ref.unknown"));
    }

    // ─── Degraded-Mode Synthetic Incident Suite ────────────────────────

    #[test]
    fn degraded_incident_contract_default_covers_required_scenarios() {
        let contract = degraded_incident_suite_contract_definition();

        assert_eq!(contract.kind, DegradedIncidentSuiteContractKind::Current);
        assert_eq!(contract.v, DEGRADED_INCIDENT_SUITE_SCHEMA_VERSION);
        assert_eq!(contract.required_incidents, all_degraded_incident_kinds());
        assert!(
            contract
                .supported_modes
                .contains(&DegradedIncidentSuiteMode::Smoke)
        );
        assert!(
            contract
                .supported_modes
                .contains(&DegradedIncidentSuiteMode::Full)
        );
        for required in ["seed", "config_hash", "scenario_id", "reason_code"] {
            assert!(
                contract
                    .required_log_fields
                    .iter()
                    .any(|field| field == required),
                "missing required log field {required}"
            );
        }
    }

    #[test]
    fn degraded_incident_smoke_suite_roundtrips_and_validates() {
        let suite = degraded_incident_smoke_suite();

        assert_eq!(suite.mode, DegradedIncidentSuiteMode::Smoke);
        assert_eq!(suite.scenarios.len(), 2);
        assert!(!suite.destructive_actions_allowed);
        assert_eq!(
            suite.network_policy,
            DegradedIncidentNetworkPolicy::OfflineOnly
        );
        suite.validate().expect("valid smoke incident suite");

        let json = serde_json::to_string(&suite).expect("serialize smoke incident suite");
        let decoded: DegradedIncidentSuite =
            serde_json::from_str(&json).expect("deserialize smoke incident suite");

        assert_eq!(decoded, suite);
    }

    #[test]
    fn degraded_incident_full_suite_covers_all_incidents() {
        let suite = degraded_incident_full_suite();
        let incidents = suite
            .scenarios
            .iter()
            .map(|scenario| scenario.incident)
            .collect::<std::collections::BTreeSet<_>>();
        let expected_outputs = suite
            .scenarios
            .iter()
            .map(|scenario| scenario.expected.output)
            .collect::<std::collections::BTreeSet<_>>();

        suite.validate().expect("valid full incident suite");
        for incident in all_degraded_incident_kinds() {
            assert!(
                incidents.contains(&incident),
                "missing incident {incident:?}"
            );
        }
        assert!(
            expected_outputs.contains(&DegradedIncidentExpectedOutput::SearchPhaseRefinementFailed)
        );
        assert!(expected_outputs.contains(&DegradedIncidentExpectedOutput::DoctorReasonCode));
        assert!(expected_outputs.contains(&DegradedIncidentExpectedOutput::AuditReasonCode));
        assert!(expected_outputs.contains(&DegradedIncidentExpectedOutput::WatcherReasonCode));
    }

    #[test]
    fn degraded_incident_suite_rejects_missing_reason_for_failure() {
        let mut suite = degraded_incident_smoke_suite();
        for scenario in &mut suite.scenarios {
            if matches!(
                scenario.incident,
                DegradedIncidentKind::CorruptVectorArtifact
            ) {
                scenario.expected.reason_code.clear();
            }
        }
        let json = serde_json::to_value(&suite).expect("serialize incident suite");
        let error = serde_json::from_value::<DegradedIncidentSuite>(json)
            .expect_err("reject missing reason code");

        assert!(error.to_string().contains("reason_code"));
    }

    #[test]
    fn degraded_incident_suite_rejects_destructive_or_incomplete_full_suite() {
        let mut destructive = degraded_incident_smoke_suite();
        destructive.destructive_actions_allowed = true;
        let error = serde_json::from_value::<DegradedIncidentSuite>(
            serde_json::to_value(&destructive).expect("serialize destructive suite"),
        )
        .expect_err("reject destructive incident suite");
        assert!(error.to_string().contains("destructive_actions"));

        let mut incomplete = degraded_incident_full_suite();
        let _removed = incomplete.scenarios.pop();
        let error = serde_json::from_value::<DegradedIncidentSuite>(
            serde_json::to_value(&incomplete).expect("serialize incomplete suite"),
        )
        .expect_err("reject incomplete full suite");
        assert!(error.to_string().contains("full.coverage"));
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
