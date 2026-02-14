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
//! - **Cold** (90+ days): Manifest + checksums only (evidence deleted).
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
pub const REPLAY_META_FILENAME: &str = "replay-meta.json";

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
            REPLAY_META_FILENAME,
        ],
        RetentionTier::Cold => &[MANIFEST_FILENAME, INDEX_CHECKSUMS_FILENAME],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(PACK_FILES.len(), 7);
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
    }

    #[test]
    fn cold_tier_keeps_only_manifest_and_checksums() {
        let files = files_for_tier(RetentionTier::Cold);
        assert_eq!(files.len(), 2);
        assert!(files.contains(&MANIFEST_FILENAME));
        assert!(files.contains(&INDEX_CHECKSUMS_FILENAME));
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
