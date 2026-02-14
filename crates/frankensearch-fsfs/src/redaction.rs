//! Redaction and retention policy engine for fsfs logs, artifacts, and evidence.
//!
//! This module implements the privacy contract from
//! `docs/fsfs-scope-privacy-contract.md` as a deterministic, testable policy
//! engine. It defines:
//!
//! - **Data class taxonomy**: Classification of sensitive content types
//!   (credentials, private keys, personal data, etc.).
//! - **Transformation rules**: Per-class deterministic redaction operations
//!   (mask, hash, truncate, drop).
//! - **Retention windows**: Per-artifact-type retention schedules tied to
//!   [`RetentionTier`] from the repro module.
//! - **Policy engine**: Applies rules to produce deterministic, replay-safe
//!   artifacts.
//!
//! # Deterministic Masking
//!
//! All masking operations are deterministic given the same input: the same
//! secret always produces the same masked output. This preserves identity
//! across events for correlation during replay without leaking the original
//! value. The masking uses a keyed truncated hash (first 8 hex chars of a
//! seeded hash) so that two different secrets produce different masks.
//!
//! # Policy Versioning
//!
//! Every redacted artifact carries a `redaction_policy_version` string
//! (e.g., `"v1"`) that identifies which rules were applied. This enables
//! forward-compatible replay: a newer engine can detect that an artifact
//! was redacted under an older policy and adjust interpretation accordingly.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::repro::RetentionTier;

/// Current redaction policy version.
pub const REDACTION_POLICY_VERSION: &str = "v1";

// ─── Data Class Taxonomy ────────────────────────────────────────────────────

/// Classification of sensitive data types found in fsfs artifacts.
///
/// Each class maps to a set of transformation rules that determine how
/// content of that class is handled in logs, evidence, explain payloads,
/// and replay bundles.
///
/// The taxonomy is ordered by severity — higher-severity classes take
/// precedence when multiple classes apply to the same content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DataClass {
    /// Credentials and tokens (API keys, bearer tokens, passwords).
    Credential,
    /// Private keys (SSH, GPG, TLS, signing keys).
    PrivateKey,
    /// Cloud provider secrets (AWS, GCP, Azure credentials).
    CloudSecret,
    /// Browser/session artifacts (cookies, session tokens).
    SessionArtifact,
    /// Personal data (email addresses, phone numbers in file content).
    PersonalData,
    /// Financial data markers.
    FinancialData,
    /// Health data markers.
    HealthData,
    /// File content from sensitive paths (e.g., dotfiles with secrets).
    SensitiveFileContent,
    /// File paths that reveal user/project structure.
    UserPath,
    /// Query text that may contain sensitive terms.
    QueryText,
    /// Non-sensitive operational data (metrics, counts, durations).
    Operational,
}

impl fmt::Display for DataClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Credential => write!(f, "credential"),
            Self::PrivateKey => write!(f, "private_key"),
            Self::CloudSecret => write!(f, "cloud_secret"),
            Self::SessionArtifact => write!(f, "session_artifact"),
            Self::PersonalData => write!(f, "personal_data"),
            Self::FinancialData => write!(f, "financial_data"),
            Self::HealthData => write!(f, "health_data"),
            Self::SensitiveFileContent => write!(f, "sensitive_file_content"),
            Self::UserPath => write!(f, "user_path"),
            Self::QueryText => write!(f, "query_text"),
            Self::Operational => write!(f, "operational"),
        }
    }
}

impl DataClass {
    /// All data classes in severity order (highest first).
    pub const ALL: &[Self] = &[
        Self::PrivateKey,
        Self::Credential,
        Self::CloudSecret,
        Self::SessionArtifact,
        Self::HealthData,
        Self::FinancialData,
        Self::PersonalData,
        Self::SensitiveFileContent,
        Self::UserPath,
        Self::QueryText,
        Self::Operational,
    ];
}

// ─── Transformation Rules ───────────────────────────────────────────────────

/// How sensitive content is transformed for a given output surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionTransform {
    /// Content is dropped entirely — not emitted at all.
    Drop,
    /// Content is replaced with a deterministic masked token.
    /// The mask preserves identity (same input → same mask) for correlation.
    Mask,
    /// Content is replaced with a one-way hash (hex-encoded truncated hash).
    Hash,
    /// Content is truncated to a safe prefix length.
    Truncate,
    /// Content is passed through unmodified.
    Passthrough,
}

impl fmt::Display for RedactionTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Drop => write!(f, "drop"),
            Self::Mask => write!(f, "mask"),
            Self::Hash => write!(f, "hash"),
            Self::Truncate => write!(f, "truncate"),
            Self::Passthrough => write!(f, "passthrough"),
        }
    }
}

/// Output surface where redacted content may appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputSurface {
    /// Structured log output (tracing spans/events).
    Log,
    /// Evidence JSONL for replay.
    Evidence,
    /// Explain payloads attached to search results.
    Explain,
    /// TUI display surface.
    Display,
    /// Repro pack artifacts.
    ReproPack,
}

impl fmt::Display for OutputSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Log => write!(f, "log"),
            Self::Evidence => write!(f, "evidence"),
            Self::Explain => write!(f, "explain"),
            Self::Display => write!(f, "display"),
            Self::ReproPack => write!(f, "repro_pack"),
        }
    }
}

/// A single transformation rule: which data class gets which transform
/// on which output surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformRule {
    /// The data class this rule applies to.
    pub data_class: DataClass,
    /// The output surface this rule applies to.
    pub surface: OutputSurface,
    /// The transformation to apply.
    pub transform: RedactionTransform,
}

// ─── Default Rule Matrix ────────────────────────────────────────────────────

/// Build the default v1 rule matrix.
///
/// Rules follow the principle of least exposure:
/// - Highest severity classes (keys, credentials) are dropped or masked everywhere.
/// - Moderate classes (paths, query text) are hashed/truncated in logs but may
///   appear in explain/display surfaces.
/// - Operational data passes through everywhere.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn default_rule_matrix() -> Vec<TransformRule> {
    use DataClass::{
        CloudSecret, Credential, FinancialData, HealthData, Operational, PersonalData, PrivateKey,
        QueryText, SensitiveFileContent, SessionArtifact, UserPath,
    };
    use OutputSurface::{Display, Evidence, Explain, Log, ReproPack};
    use RedactionTransform::{Drop, Hash, Mask, Passthrough, Truncate};

    vec![
        // ── Credential: mask in evidence (for correlation), drop elsewhere ──
        TransformRule {
            data_class: Credential,
            surface: Log,
            transform: Drop,
        },
        TransformRule {
            data_class: Credential,
            surface: Evidence,
            transform: Mask,
        },
        TransformRule {
            data_class: Credential,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: Credential,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: Credential,
            surface: ReproPack,
            transform: Mask,
        },
        // ── PrivateKey: drop everywhere (never emit) ──
        TransformRule {
            data_class: PrivateKey,
            surface: Log,
            transform: Drop,
        },
        TransformRule {
            data_class: PrivateKey,
            surface: Evidence,
            transform: Drop,
        },
        TransformRule {
            data_class: PrivateKey,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: PrivateKey,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: PrivateKey,
            surface: ReproPack,
            transform: Drop,
        },
        // ── CloudSecret: same as Credential ──
        TransformRule {
            data_class: CloudSecret,
            surface: Log,
            transform: Drop,
        },
        TransformRule {
            data_class: CloudSecret,
            surface: Evidence,
            transform: Mask,
        },
        TransformRule {
            data_class: CloudSecret,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: CloudSecret,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: CloudSecret,
            surface: ReproPack,
            transform: Mask,
        },
        // ── SessionArtifact: drop everywhere ──
        TransformRule {
            data_class: SessionArtifact,
            surface: Log,
            transform: Drop,
        },
        TransformRule {
            data_class: SessionArtifact,
            surface: Evidence,
            transform: Drop,
        },
        TransformRule {
            data_class: SessionArtifact,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: SessionArtifact,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: SessionArtifact,
            surface: ReproPack,
            transform: Drop,
        },
        // ── PersonalData: hash in logs/evidence, truncate in explain ──
        TransformRule {
            data_class: PersonalData,
            surface: Log,
            transform: Hash,
        },
        TransformRule {
            data_class: PersonalData,
            surface: Evidence,
            transform: Hash,
        },
        TransformRule {
            data_class: PersonalData,
            surface: Explain,
            transform: Truncate,
        },
        TransformRule {
            data_class: PersonalData,
            surface: Display,
            transform: Truncate,
        },
        TransformRule {
            data_class: PersonalData,
            surface: ReproPack,
            transform: Hash,
        },
        // ── FinancialData: hash everywhere ──
        TransformRule {
            data_class: FinancialData,
            surface: Log,
            transform: Hash,
        },
        TransformRule {
            data_class: FinancialData,
            surface: Evidence,
            transform: Hash,
        },
        TransformRule {
            data_class: FinancialData,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: FinancialData,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: FinancialData,
            surface: ReproPack,
            transform: Hash,
        },
        // ── HealthData: hash everywhere ──
        TransformRule {
            data_class: HealthData,
            surface: Log,
            transform: Hash,
        },
        TransformRule {
            data_class: HealthData,
            surface: Evidence,
            transform: Hash,
        },
        TransformRule {
            data_class: HealthData,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: HealthData,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: HealthData,
            surface: ReproPack,
            transform: Hash,
        },
        // ── SensitiveFileContent: mask in evidence, drop in logs ──
        TransformRule {
            data_class: SensitiveFileContent,
            surface: Log,
            transform: Drop,
        },
        TransformRule {
            data_class: SensitiveFileContent,
            surface: Evidence,
            transform: Mask,
        },
        TransformRule {
            data_class: SensitiveFileContent,
            surface: Explain,
            transform: Drop,
        },
        TransformRule {
            data_class: SensitiveFileContent,
            surface: Display,
            transform: Drop,
        },
        TransformRule {
            data_class: SensitiveFileContent,
            surface: ReproPack,
            transform: Mask,
        },
        // ── UserPath: hash in logs/evidence, truncate in display ──
        TransformRule {
            data_class: UserPath,
            surface: Log,
            transform: Hash,
        },
        TransformRule {
            data_class: UserPath,
            surface: Evidence,
            transform: Hash,
        },
        TransformRule {
            data_class: UserPath,
            surface: Explain,
            transform: Truncate,
        },
        TransformRule {
            data_class: UserPath,
            surface: Display,
            transform: Passthrough,
        },
        TransformRule {
            data_class: UserPath,
            surface: ReproPack,
            transform: Hash,
        },
        // ── QueryText: truncate in logs, passthrough in display/explain ──
        TransformRule {
            data_class: QueryText,
            surface: Log,
            transform: Truncate,
        },
        TransformRule {
            data_class: QueryText,
            surface: Evidence,
            transform: Truncate,
        },
        TransformRule {
            data_class: QueryText,
            surface: Explain,
            transform: Passthrough,
        },
        TransformRule {
            data_class: QueryText,
            surface: Display,
            transform: Passthrough,
        },
        TransformRule {
            data_class: QueryText,
            surface: ReproPack,
            transform: Truncate,
        },
        // ── Operational: passthrough everywhere ──
        TransformRule {
            data_class: Operational,
            surface: Log,
            transform: Passthrough,
        },
        TransformRule {
            data_class: Operational,
            surface: Evidence,
            transform: Passthrough,
        },
        TransformRule {
            data_class: Operational,
            surface: Explain,
            transform: Passthrough,
        },
        TransformRule {
            data_class: Operational,
            surface: Display,
            transform: Passthrough,
        },
        TransformRule {
            data_class: Operational,
            surface: ReproPack,
            transform: Passthrough,
        },
    ]
}

// ─── Artifact Retention Schedule ────────────────────────────────────────────

/// Artifact type for retention scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    /// Evidence JSONL logs.
    EvidenceLog,
    /// Structured tracing spans (log output).
    TracingSpan,
    /// Search explain payloads.
    ExplainPayload,
    /// Repro pack manifest.
    ReproManifest,
    /// Repro pack evidence bundle.
    ReproEvidence,
    /// Repro pack configuration snapshot.
    ReproConfig,
    /// Repro pack environment snapshot.
    ReproEnv,
    /// Repro pack model manifest.
    ReproModel,
    /// Repro pack index checksums.
    ReproChecksums,
    /// Repro pack replay metadata.
    ReproReplay,
    /// Telemetry counters/gauges.
    TelemetryMetrics,
    /// Anomaly/SLO alert records.
    AnomalyAlert,
}

impl fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvidenceLog => write!(f, "evidence_log"),
            Self::TracingSpan => write!(f, "tracing_span"),
            Self::ExplainPayload => write!(f, "explain_payload"),
            Self::ReproManifest => write!(f, "repro_manifest"),
            Self::ReproEvidence => write!(f, "repro_evidence"),
            Self::ReproConfig => write!(f, "repro_config"),
            Self::ReproEnv => write!(f, "repro_env"),
            Self::ReproModel => write!(f, "repro_model"),
            Self::ReproChecksums => write!(f, "repro_checksums"),
            Self::ReproReplay => write!(f, "repro_replay"),
            Self::TelemetryMetrics => write!(f, "telemetry_metrics"),
            Self::AnomalyAlert => write!(f, "anomaly_alert"),
        }
    }
}

/// Retention schedule entry for a single artifact type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRetention {
    /// The artifact type.
    pub artifact_type: ArtifactType,
    /// Maximum age in days before deletion (0 = keep forever).
    pub max_age_days: u16,
    /// Whether this artifact is retained in Hot tier.
    pub hot: bool,
    /// Whether this artifact is retained in Warm tier.
    pub warm: bool,
    /// Whether this artifact is retained in Cold tier.
    pub cold: bool,
}

/// Build the default artifact retention schedule.
///
/// Schedules are aligned with the repro tier model:
/// - Hot (0-7d): everything retained.
/// - Warm (7-90d): logs trimmed, repro compressed, explain dropped.
/// - Cold (90+d): only manifests, checksums, and anomaly alerts.
#[must_use]
pub fn default_artifact_retention() -> Vec<ArtifactRetention> {
    use ArtifactType::{
        AnomalyAlert, EvidenceLog, ExplainPayload, ReproChecksums, ReproConfig, ReproEnv,
        ReproEvidence, ReproManifest, ReproModel, ReproReplay, TelemetryMetrics, TracingSpan,
    };

    vec![
        ArtifactRetention {
            artifact_type: EvidenceLog,
            max_age_days: 90,
            hot: true,
            warm: true,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: TracingSpan,
            max_age_days: 7,
            hot: true,
            warm: false,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: ExplainPayload,
            max_age_days: 7,
            hot: true,
            warm: false,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: ReproManifest,
            max_age_days: 0,
            hot: true,
            warm: true,
            cold: true,
        },
        ArtifactRetention {
            artifact_type: ReproEvidence,
            max_age_days: 90,
            hot: true,
            warm: true,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: ReproConfig,
            max_age_days: 90,
            hot: true,
            warm: true,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: ReproEnv,
            max_age_days: 7,
            hot: true,
            warm: false,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: ReproModel,
            max_age_days: 90,
            hot: true,
            warm: true,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: ReproChecksums,
            max_age_days: 0,
            hot: true,
            warm: true,
            cold: true,
        },
        ArtifactRetention {
            artifact_type: ReproReplay,
            max_age_days: 90,
            hot: true,
            warm: true,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: TelemetryMetrics,
            max_age_days: 90,
            hot: true,
            warm: true,
            cold: false,
        },
        ArtifactRetention {
            artifact_type: AnomalyAlert,
            max_age_days: 0,
            hot: true,
            warm: true,
            cold: true,
        },
    ]
}

// ─── Deterministic Masking ──────────────────────────────────────────────────

/// Seed for deterministic masking operations.
///
/// The same seed + input always produces the same masked output, enabling
/// correlation across evidence events during replay. The seed should be
/// set per-instance at startup and persisted in the repro manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaskSeed(pub u64);

impl Default for MaskSeed {
    fn default() -> Self {
        Self(0xf5f5_cafe_babe_d00d)
    }
}

/// Apply deterministic masking to a value.
///
/// Produces a fixed-length hex string that is:
/// - Deterministic (same seed + input → same output).
/// - Non-reversible (truncated hash).
/// - Distinguishable (different inputs → different outputs with high probability).
///
/// Format: `<MASKED:xxxxxxxx>` where `xxxxxxxx` is 8 hex chars.
#[must_use]
pub fn deterministic_mask(seed: MaskSeed, value: &str) -> String {
    // FNV-1a 64-bit seeded hash for fast, deterministic, non-crypto masking.
    let mut hash: u64 = seed.0;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    #[allow(clippy::cast_possible_truncation)]
    let low32 = hash as u32;
    format!("<MASKED:{low32:08x}>")
}

/// Apply deterministic hashing to a value.
///
/// Produces a hex-encoded truncated hash for correlation without exposure.
/// Format: `<HASH:xxxxxxxxxxxxxxxx>` where the hex string is 16 chars.
#[must_use]
pub fn deterministic_hash(seed: MaskSeed, value: &str) -> String {
    let mut hash: u64 = seed.0;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("<HASH:{hash:016x}>")
}

/// Apply deterministic truncation to a value.
///
/// Keeps the first `max_len` characters and appends `...` if truncated.
#[must_use]
pub fn deterministic_truncate(value: &str, max_len: usize) -> String {
    if value.len() <= max_len {
        return value.to_string();
    }
    let truncated: String = value.chars().take(max_len).collect();
    format!("{truncated}...")
}

// ─── Policy Engine ──────────────────────────────────────────────────────────

/// Redaction policy engine that applies transformation rules to content.
///
/// Constructed from a rule matrix and mask seed, the engine provides
/// deterministic redaction for any (`DataClass`, `OutputSurface`) pair.
#[derive(Debug, Clone)]
pub struct RedactionPolicy {
    /// Policy version string.
    pub version: String,
    /// Mask seed for deterministic operations.
    pub seed: MaskSeed,
    /// Default truncation length for `Truncate` transforms.
    pub truncate_max_len: usize,
    /// Rule lookup: (`DataClass`, `OutputSurface`) → `RedactionTransform`.
    rules: HashMap<(DataClass, OutputSurface), RedactionTransform>,
    /// Artifact retention schedule.
    artifact_retention: Vec<ArtifactRetention>,
}

/// Serialization-friendly representation of the policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RedactionPolicySerde {
    version: String,
    seed: MaskSeed,
    truncate_max_len: usize,
    rules: Vec<TransformRule>,
    artifact_retention: Vec<ArtifactRetention>,
}

impl Serialize for RedactionPolicy {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let rules: Vec<TransformRule> = self
            .rules
            .iter()
            .map(|(&(data_class, surface), &transform)| TransformRule {
                data_class,
                surface,
                transform,
            })
            .collect();
        RedactionPolicySerde {
            version: self.version.clone(),
            seed: self.seed,
            truncate_max_len: self.truncate_max_len,
            rules,
            artifact_retention: self.artifact_retention.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RedactionPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let serde = RedactionPolicySerde::deserialize(deserializer)?;
        let rules = serde
            .rules
            .into_iter()
            .map(|rule| ((rule.data_class, rule.surface), rule.transform))
            .collect();
        Ok(Self {
            version: serde.version,
            seed: serde.seed,
            truncate_max_len: serde.truncate_max_len,
            rules,
            artifact_retention: serde.artifact_retention,
        })
    }
}

impl Default for RedactionPolicy {
    fn default() -> Self {
        Self::new(MaskSeed::default())
    }
}

impl RedactionPolicy {
    /// Create a new policy with the default v1 rule matrix.
    #[must_use]
    pub fn new(seed: MaskSeed) -> Self {
        let rules = default_rule_matrix()
            .into_iter()
            .map(|rule| ((rule.data_class, rule.surface), rule.transform))
            .collect();

        Self {
            version: REDACTION_POLICY_VERSION.to_string(),
            seed,
            truncate_max_len: 64,
            rules,
            artifact_retention: default_artifact_retention(),
        }
    }

    /// Look up the transformation for a given (`data_class`, `surface`) pair.
    ///
    /// Returns `Drop` if no explicit rule exists (fail-closed).
    #[must_use]
    pub fn transform_for(
        &self,
        data_class: DataClass,
        surface: OutputSurface,
    ) -> RedactionTransform {
        self.rules
            .get(&(data_class, surface))
            .copied()
            .unwrap_or(RedactionTransform::Drop)
    }

    /// Apply the policy to a value, returning the redacted output.
    ///
    /// Returns `None` if the transform is `Drop` (content should not be emitted).
    #[must_use]
    pub fn apply(
        &self,
        data_class: DataClass,
        surface: OutputSurface,
        value: &str,
    ) -> Option<String> {
        match self.transform_for(data_class, surface) {
            RedactionTransform::Drop => None,
            RedactionTransform::Mask => Some(deterministic_mask(self.seed, value)),
            RedactionTransform::Hash => Some(deterministic_hash(self.seed, value)),
            RedactionTransform::Truncate => {
                Some(deterministic_truncate(value, self.truncate_max_len))
            }
            RedactionTransform::Passthrough => Some(value.to_string()),
        }
    }

    /// Check whether an artifact type is retained in a given tier.
    #[must_use]
    pub fn is_retained(&self, artifact_type: ArtifactType, tier: RetentionTier) -> bool {
        self.artifact_retention
            .iter()
            .find(|entry| entry.artifact_type == artifact_type)
            .is_some_and(|entry| match tier {
                RetentionTier::Hot => entry.hot,
                RetentionTier::Warm => entry.warm,
                RetentionTier::Cold => entry.cold,
            })
    }

    /// Get the maximum age in days for an artifact type (0 = forever).
    #[must_use]
    pub fn max_age_days(&self, artifact_type: ArtifactType) -> u16 {
        self.artifact_retention
            .iter()
            .find(|entry| entry.artifact_type == artifact_type)
            .map_or(0, |entry| entry.max_age_days)
    }

    /// Determine which artifacts should be deleted at a given age.
    #[must_use]
    pub fn expired_artifacts(&self, age_days: u16) -> Vec<ArtifactType> {
        self.artifact_retention
            .iter()
            .filter(|entry| entry.max_age_days > 0 && age_days > entry.max_age_days)
            .map(|entry| entry.artifact_type)
            .collect()
    }

    /// Get the artifact retention schedule.
    #[must_use]
    pub fn artifact_retention(&self) -> &[ArtifactRetention] {
        &self.artifact_retention
    }

    /// Override a specific rule in the matrix.
    pub fn set_rule(
        &mut self,
        data_class: DataClass,
        surface: OutputSurface,
        transform: RedactionTransform,
    ) {
        self.rules.insert((data_class, surface), transform);
    }
}

// ─── Path Classification ────────────────────────────────────────────────────

/// Sensitive path patterns that trigger hard deny.
///
/// These are glob-style suffixes checked against file paths. If a path
/// matches any pattern, its content MUST NOT be persisted, emitted, or
/// displayed (per the scope-privacy contract).
pub const HARD_DENY_PATH_PATTERNS: &[&str] = &[
    ".ssh/",
    ".gnupg/",
    ".aws/credentials",
    ".config/gcloud/",
    ".azure/",
    ".kube/config",
    ".docker/config.json",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
];

/// Check whether a path matches a hard-deny pattern.
#[must_use]
pub fn is_hard_deny_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    HARD_DENY_PATH_PATTERNS
        .iter()
        .any(|pattern| normalized.contains(pattern))
}

/// Classify the data classes present in a given path.
///
/// Returns the set of data classes detected, ordered by severity.
#[must_use]
pub fn classify_path(path: &str) -> Vec<DataClass> {
    let mut classes = Vec::new();
    let normalized = path.replace('\\', "/");
    let lower = normalized.to_ascii_lowercase();

    if lower.contains(".ssh/")
        || lower.contains("id_rsa")
        || lower.contains("id_ed25519")
        || lower.contains("id_ecdsa")
        || lower.contains(".gnupg/")
    {
        classes.push(DataClass::PrivateKey);
    }

    if lower.contains(".aws/credentials")
        || lower.contains(".config/gcloud/")
        || lower.contains(".azure/")
    {
        classes.push(DataClass::CloudSecret);
    }

    if lower.contains(".env")
        || lower.contains(".npmrc")
        || lower.contains(".pypirc")
        || lower.contains(".netrc")
        || lower.contains(".docker/config.json")
    {
        classes.push(DataClass::Credential);
    }

    if lower.contains("cookie") || lower.contains("session") {
        classes.push(DataClass::SessionArtifact);
    }

    if classes.is_empty() && normalized.contains('/') {
        classes.push(DataClass::UserPath);
    }

    classes.sort();
    classes.dedup();
    classes
}

/// Result of applying the redaction policy to a single field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionResult {
    /// The redacted value (None if dropped).
    pub value: Option<String>,
    /// The transformation applied.
    pub transform: RedactionTransform,
    /// The data class that triggered the transformation.
    pub data_class: DataClass,
    /// Policy version used.
    pub policy_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Data Class ─────────────────────────────────────────────────────

    #[test]
    fn data_class_display() {
        assert_eq!(DataClass::Credential.to_string(), "credential");
        assert_eq!(DataClass::PrivateKey.to_string(), "private_key");
        assert_eq!(DataClass::CloudSecret.to_string(), "cloud_secret");
        assert_eq!(DataClass::Operational.to_string(), "operational");
    }

    #[test]
    fn data_class_serde_roundtrip() {
        for &class in DataClass::ALL {
            let json = serde_json::to_string(&class).unwrap();
            let decoded: DataClass = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, class);
        }
    }

    #[test]
    fn data_class_all_has_every_variant() {
        assert_eq!(DataClass::ALL.len(), 11);
    }

    // ─── Transform ──────────────────────────────────────────────────────

    #[test]
    fn transform_display() {
        assert_eq!(RedactionTransform::Drop.to_string(), "drop");
        assert_eq!(RedactionTransform::Mask.to_string(), "mask");
        assert_eq!(RedactionTransform::Hash.to_string(), "hash");
        assert_eq!(RedactionTransform::Truncate.to_string(), "truncate");
        assert_eq!(RedactionTransform::Passthrough.to_string(), "passthrough");
    }

    #[test]
    fn transform_serde_roundtrip() {
        for transform in [
            RedactionTransform::Drop,
            RedactionTransform::Mask,
            RedactionTransform::Hash,
            RedactionTransform::Truncate,
            RedactionTransform::Passthrough,
        ] {
            let json = serde_json::to_string(&transform).unwrap();
            let decoded: RedactionTransform = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, transform);
        }
    }

    // ─── Output Surface ─────────────────────────────────────────────────

    #[test]
    fn output_surface_display() {
        assert_eq!(OutputSurface::Log.to_string(), "log");
        assert_eq!(OutputSurface::Evidence.to_string(), "evidence");
        assert_eq!(OutputSurface::Explain.to_string(), "explain");
        assert_eq!(OutputSurface::Display.to_string(), "display");
        assert_eq!(OutputSurface::ReproPack.to_string(), "repro_pack");
    }

    // ─── Default Rule Matrix ────────────────────────────────────────────

    #[test]
    fn default_rule_matrix_covers_all_classes_and_surfaces() {
        let rules = default_rule_matrix();
        // 11 data classes × 5 surfaces = 55 rules
        assert_eq!(rules.len(), 55);

        let surfaces = [
            OutputSurface::Log,
            OutputSurface::Evidence,
            OutputSurface::Explain,
            OutputSurface::Display,
            OutputSurface::ReproPack,
        ];

        for &class in DataClass::ALL {
            for &surface in &surfaces {
                assert!(
                    rules
                        .iter()
                        .any(|r| r.data_class == class && r.surface == surface),
                    "missing rule for ({class}, {surface})"
                );
            }
        }
    }

    #[test]
    fn private_key_is_dropped_everywhere() {
        let rules = default_rule_matrix();
        for rule in &rules {
            if rule.data_class == DataClass::PrivateKey {
                assert_eq!(
                    rule.transform,
                    RedactionTransform::Drop,
                    "PrivateKey should be dropped on {:?}",
                    rule.surface
                );
            }
        }
    }

    #[test]
    fn operational_is_passthrough_everywhere() {
        let rules = default_rule_matrix();
        for rule in &rules {
            if rule.data_class == DataClass::Operational {
                assert_eq!(
                    rule.transform,
                    RedactionTransform::Passthrough,
                    "Operational should be passthrough on {:?}",
                    rule.surface
                );
            }
        }
    }

    // ─── Deterministic Masking ──────────────────────────────────────────

    #[test]
    fn mask_is_deterministic() {
        let seed = MaskSeed::default();
        let a = deterministic_mask(seed, "my-secret-token");
        let b = deterministic_mask(seed, "my-secret-token");
        assert_eq!(a, b);
    }

    #[test]
    fn mask_differs_for_different_inputs() {
        let seed = MaskSeed::default();
        let a = deterministic_mask(seed, "secret-a");
        let b = deterministic_mask(seed, "secret-b");
        assert_ne!(a, b);
    }

    #[test]
    fn mask_differs_for_different_seeds() {
        let a = deterministic_mask(MaskSeed(1), "same-input");
        let b = deterministic_mask(MaskSeed(2), "same-input");
        assert_ne!(a, b);
    }

    #[test]
    fn mask_format() {
        let result = deterministic_mask(MaskSeed::default(), "test");
        assert!(result.starts_with("<MASKED:"));
        assert!(result.ends_with('>'));
        // <MASKED:xxxxxxxx> = 17 chars
        assert_eq!(result.len(), 17);
    }

    #[test]
    fn hash_is_deterministic() {
        let seed = MaskSeed::default();
        let a = deterministic_hash(seed, "my-path/file.txt");
        let b = deterministic_hash(seed, "my-path/file.txt");
        assert_eq!(a, b);
    }

    #[test]
    fn hash_differs_for_different_inputs() {
        let seed = MaskSeed::default();
        let a = deterministic_hash(seed, "path-a");
        let b = deterministic_hash(seed, "path-b");
        assert_ne!(a, b);
    }

    #[test]
    fn hash_format() {
        let result = deterministic_hash(MaskSeed::default(), "test");
        assert!(result.starts_with("<HASH:"));
        assert!(result.ends_with('>'));
        // <HASH:xxxxxxxxxxxxxxxx> = 23 chars
        assert_eq!(result.len(), 23);
    }

    #[test]
    fn truncate_short_values_unchanged() {
        let result = deterministic_truncate("short", 10);
        assert_eq!(result, "short");
    }

    #[test]
    fn truncate_long_values() {
        let result = deterministic_truncate("this is a long string that should be truncated", 10);
        assert_eq!(result, "this is a ...");
    }

    #[test]
    fn truncate_exact_length() {
        let result = deterministic_truncate("exactly10!", 10);
        assert_eq!(result, "exactly10!");
    }

    // ─── Policy Engine ──────────────────────────────────────────────────

    #[test]
    fn policy_default_version() {
        let policy = RedactionPolicy::default();
        assert_eq!(policy.version, "v1");
    }

    #[test]
    fn policy_drops_private_keys() {
        let policy = RedactionPolicy::default();
        let result = policy.apply(
            DataClass::PrivateKey,
            OutputSurface::Log,
            "-----BEGIN RSA PRIVATE KEY-----",
        );
        assert!(result.is_none());
    }

    #[test]
    fn policy_masks_credentials_in_evidence() {
        let policy = RedactionPolicy::default();
        let result = policy.apply(
            DataClass::Credential,
            OutputSurface::Evidence,
            "sk-1234567890abcdef",
        );
        assert!(result.is_some());
        let masked = result.unwrap();
        assert!(masked.starts_with("<MASKED:"));
        assert!(!masked.contains("sk-1234567890abcdef"));
    }

    #[test]
    fn policy_passes_operational_data() {
        let policy = RedactionPolicy::default();
        let result = policy.apply(DataClass::Operational, OutputSurface::Log, "latency_ms=42");
        assert_eq!(result, Some("latency_ms=42".to_string()));
    }

    #[test]
    fn policy_hashes_user_paths_in_logs() {
        let policy = RedactionPolicy::default();
        let result = policy.apply(
            DataClass::UserPath,
            OutputSurface::Log,
            "/home/user/projects/secret-project/main.rs",
        );
        assert!(result.is_some());
        let hashed = result.unwrap();
        assert!(hashed.starts_with("<HASH:"));
    }

    #[test]
    fn policy_truncates_query_text_in_logs() {
        let policy = RedactionPolicy::new(MaskSeed::default());
        let long_query = "a".repeat(200);
        let result = policy.apply(DataClass::QueryText, OutputSurface::Log, &long_query);
        assert!(result.is_some());
        let truncated = result.unwrap();
        assert!(truncated.ends_with("..."));
        assert!(truncated.len() < 200);
    }

    #[test]
    fn policy_unknown_pair_defaults_to_drop() {
        let mut policy = RedactionPolicy::default();
        // Remove a rule to test fallback
        policy
            .rules
            .remove(&(DataClass::Credential, OutputSurface::Log));
        let result = policy.apply(DataClass::Credential, OutputSurface::Log, "secret");
        assert!(result.is_none()); // fail-closed
    }

    #[test]
    fn policy_set_rule_overrides() {
        let mut policy = RedactionPolicy::default();
        // Override: allow credentials in display (not recommended, but testable)
        policy.set_rule(
            DataClass::Credential,
            OutputSurface::Display,
            RedactionTransform::Passthrough,
        );
        let result = policy.apply(DataClass::Credential, OutputSurface::Display, "my-token");
        assert_eq!(result, Some("my-token".to_string()));
    }

    // ─── Artifact Retention ─────────────────────────────────────────────

    #[test]
    fn default_retention_covers_all_artifact_types() {
        let schedule = default_artifact_retention();
        assert_eq!(schedule.len(), 12);
    }

    #[test]
    fn repro_manifest_retained_in_all_tiers() {
        let policy = RedactionPolicy::default();
        assert!(policy.is_retained(ArtifactType::ReproManifest, RetentionTier::Hot));
        assert!(policy.is_retained(ArtifactType::ReproManifest, RetentionTier::Warm));
        assert!(policy.is_retained(ArtifactType::ReproManifest, RetentionTier::Cold));
    }

    #[test]
    fn tracing_spans_only_in_hot() {
        let policy = RedactionPolicy::default();
        assert!(policy.is_retained(ArtifactType::TracingSpan, RetentionTier::Hot));
        assert!(!policy.is_retained(ArtifactType::TracingSpan, RetentionTier::Warm));
        assert!(!policy.is_retained(ArtifactType::TracingSpan, RetentionTier::Cold));
    }

    #[test]
    fn evidence_log_not_in_cold() {
        let policy = RedactionPolicy::default();
        assert!(policy.is_retained(ArtifactType::EvidenceLog, RetentionTier::Hot));
        assert!(policy.is_retained(ArtifactType::EvidenceLog, RetentionTier::Warm));
        assert!(!policy.is_retained(ArtifactType::EvidenceLog, RetentionTier::Cold));
    }

    #[test]
    fn anomaly_alerts_kept_forever() {
        let policy = RedactionPolicy::default();
        assert_eq!(policy.max_age_days(ArtifactType::AnomalyAlert), 0);
        assert!(policy.is_retained(ArtifactType::AnomalyAlert, RetentionTier::Cold));
    }

    #[test]
    fn expired_artifacts_at_various_ages() {
        let policy = RedactionPolicy::default();

        // At day 5: nothing expired
        let expired = policy.expired_artifacts(5);
        assert!(expired.is_empty());

        // At day 8: tracing spans (7d) and explain payloads (7d) expired
        let expired = policy.expired_artifacts(8);
        assert!(expired.contains(&ArtifactType::TracingSpan));
        assert!(expired.contains(&ArtifactType::ExplainPayload));
        assert!(expired.contains(&ArtifactType::ReproEnv));

        // At day 91: most things expired except manifests, checksums, alerts (max_age_days=0)
        let expired = policy.expired_artifacts(91);
        assert!(expired.contains(&ArtifactType::EvidenceLog));
        assert!(!expired.contains(&ArtifactType::ReproManifest)); // 0 = forever
        assert!(!expired.contains(&ArtifactType::AnomalyAlert)); // 0 = forever
    }

    // ─── Path Classification ────────────────────────────────────────────

    #[test]
    fn hard_deny_detects_ssh() {
        assert!(is_hard_deny_path("/home/user/.ssh/id_rsa"));
        assert!(is_hard_deny_path("~/.ssh/authorized_keys"));
    }

    #[test]
    fn hard_deny_detects_cloud_creds() {
        assert!(is_hard_deny_path("/home/user/.aws/credentials"));
        assert!(is_hard_deny_path(
            "/home/user/.config/gcloud/application_default_credentials.json"
        ));
    }

    #[test]
    fn hard_deny_detects_dotenv() {
        assert!(is_hard_deny_path("/project/.env"));
        assert!(is_hard_deny_path("/project/.env.local")); // contains .env
    }

    #[test]
    fn hard_deny_allows_safe_paths() {
        assert!(!is_hard_deny_path("/home/user/projects/main.rs"));
        assert!(!is_hard_deny_path("/home/user/docs/notes.md"));
    }

    #[test]
    fn classify_ssh_key() {
        let classes = classify_path("/home/user/.ssh/id_rsa");
        assert!(classes.contains(&DataClass::PrivateKey));
    }

    #[test]
    fn classify_windows_ssh_key() {
        let classes = classify_path(r"C:\Users\alice\.ssh\id_rsa");
        assert!(classes.contains(&DataClass::PrivateKey));
    }

    #[test]
    fn classify_aws_creds() {
        let classes = classify_path("/home/user/.aws/credentials");
        assert!(classes.contains(&DataClass::CloudSecret));
    }

    #[test]
    fn classify_windows_cloud_creds() {
        let classes =
            classify_path(r"C:\Users\alice\.config\gcloud\application_default_credentials.json");
        assert!(classes.contains(&DataClass::CloudSecret));
    }

    #[test]
    fn classify_normal_source_file() {
        let classes = classify_path("/home/user/project/src/main.rs");
        assert_eq!(classes, vec![DataClass::UserPath]);
    }

    #[test]
    fn classify_dotenv() {
        let classes = classify_path("/project/.env");
        assert!(classes.contains(&DataClass::Credential));
    }

    // ─── Redaction Result ───────────────────────────────────────────────

    #[test]
    fn redaction_result_serde_roundtrip() {
        let result = RedactionResult {
            value: Some("<MASKED:abcdef01>".to_string()),
            transform: RedactionTransform::Mask,
            data_class: DataClass::Credential,
            policy_version: "v1".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: RedactionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, result);
    }

    #[test]
    fn redaction_result_dropped() {
        let result = RedactionResult {
            value: None,
            transform: RedactionTransform::Drop,
            data_class: DataClass::PrivateKey,
            policy_version: "v1".to_string(),
        };
        assert!(result.value.is_none());
    }

    // ─── Policy Serde ───────────────────────────────────────────────────

    #[test]
    fn policy_serde_roundtrip() {
        let policy = RedactionPolicy::default();
        let json = serde_json::to_string(&policy).unwrap();
        let decoded: RedactionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.version, policy.version);
        assert_eq!(decoded.seed, policy.seed);
        assert_eq!(decoded.truncate_max_len, policy.truncate_max_len);
        // Verify a sample rule survived
        assert_eq!(
            decoded.transform_for(DataClass::PrivateKey, OutputSurface::Log),
            RedactionTransform::Drop
        );
    }

    // ─── Integration: Policy + Path Classification ──────────────────────

    #[test]
    fn ssh_key_path_fully_redacted_in_all_surfaces() {
        let policy = RedactionPolicy::default();
        let classes = classify_path("/home/user/.ssh/id_rsa");

        // PrivateKey should be in the class list
        assert!(classes.contains(&DataClass::PrivateKey));

        // PrivateKey is dropped on every surface
        let surfaces = [
            OutputSurface::Log,
            OutputSurface::Evidence,
            OutputSurface::Explain,
            OutputSurface::Display,
            OutputSurface::ReproPack,
        ];
        for surface in surfaces {
            let result = policy.apply(DataClass::PrivateKey, surface, "secret-key-content");
            assert!(
                result.is_none(),
                "PrivateKey should be dropped on {surface}"
            );
        }
    }

    #[test]
    fn user_path_hashed_in_evidence_but_visible_in_display() {
        let policy = RedactionPolicy::default();

        let evidence = policy.apply(
            DataClass::UserPath,
            OutputSurface::Evidence,
            "/home/user/project/src/main.rs",
        );
        assert!(evidence.unwrap().starts_with("<HASH:"));

        let display = policy.apply(
            DataClass::UserPath,
            OutputSurface::Display,
            "/home/user/project/src/main.rs",
        );
        assert_eq!(display.unwrap(), "/home/user/project/src/main.rs");
    }
}
