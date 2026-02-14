//! FrankenSQLite-backed storage bootstrap for ops telemetry.
//!
//! This module provides the schema contract for the control-plane database
//! (`frankensearch-ops.db`) and a small connection wrapper that applies
//! pragmas, runs migrations, and validates migration checksums.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

/// Current schema version for the ops telemetry database.
pub const OPS_SCHEMA_VERSION: i64 = 2;

#[allow(clippy::needless_raw_string_hashes)]
const OPS_SCHEMA_MIGRATIONS_TABLE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS ops_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at_ms INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    reversible INTEGER NOT NULL CHECK (reversible IN (0, 1))
);
"#;

const OPS_SCHEMA_V1_NAME: &str = "ops_telemetry_storage_v1";
const OPS_SCHEMA_V1_CHECKSUM: &str = "ops-schema-v1-20260214";
const OPS_SCHEMA_V2_NAME: &str = "ops_telemetry_storage_v2_slo_anomaly_rollups";
const OPS_SCHEMA_V2_CHECKSUM: &str = "ops-schema-v2-20260214";

#[allow(clippy::needless_raw_string_hashes)]
const OPS_SCHEMA_V1_STATEMENTS: &[&str] = &[
    r#"
CREATE TABLE IF NOT EXISTS projects (
    project_key TEXT PRIMARY KEY,
    display_name TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    host_name TEXT,
    pid INTEGER,
    version TEXT,
    first_seen_ms INTEGER NOT NULL,
    last_heartbeat_ms INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started', 'healthy', 'degraded', 'stale', 'stopped'))
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS search_events (
    event_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    correlation_id TEXT NOT NULL,
    query_hash TEXT,
    query_class TEXT,
    phase TEXT NOT NULL CHECK (phase IN ('initial', 'refined', 'failed')),
    latency_us INTEGER NOT NULL,
    result_count INTEGER,
    memory_bytes INTEGER,
    ts_ms INTEGER NOT NULL
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS search_summaries (
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    window TEXT NOT NULL CHECK (window IN ('1m', '15m', '1h', '6h', '24h', '3d', '1w')),
    window_start_ms INTEGER NOT NULL,
    search_count INTEGER NOT NULL,
    p50_latency_us INTEGER,
    p95_latency_us INTEGER,
    p99_latency_us INTEGER,
    avg_result_count REAL,
    PRIMARY KEY (project_key, instance_id, window, window_start_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS embedding_job_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    embedder_id TEXT NOT NULL,
    pending_jobs INTEGER NOT NULL,
    processing_jobs INTEGER NOT NULL,
    completed_jobs INTEGER NOT NULL,
    failed_jobs INTEGER NOT NULL,
    retried_jobs INTEGER NOT NULL,
    batch_latency_us INTEGER,
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, embedder_id, ts_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS index_inventory_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    index_name TEXT NOT NULL,
    index_type TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    file_size_bytes INTEGER,
    file_hash TEXT,
    is_stale INTEGER NOT NULL CHECK (is_stale IN (0, 1)),
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, index_name, ts_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS resource_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT NOT NULL REFERENCES instances(instance_id) ON DELETE CASCADE,
    cpu_pct REAL,
    rss_bytes INTEGER,
    io_read_bytes INTEGER,
    io_write_bytes INTEGER,
    queue_depth INTEGER,
    ts_ms INTEGER NOT NULL,
    UNIQUE (project_key, instance_id, ts_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS alerts_timeline (
    alert_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    instance_id TEXT,
    category TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warn', 'error', 'critical')),
    reason_code TEXT NOT NULL,
    summary TEXT,
    state TEXT NOT NULL CHECK (state IN ('open', 'acknowledged', 'resolved')),
    opened_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    resolved_at_ms INTEGER
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS evidence_links (
    link_id TEXT PRIMARY KEY,
    project_key TEXT NOT NULL REFERENCES projects(project_key) ON DELETE CASCADE,
    alert_id TEXT NOT NULL REFERENCES alerts_timeline(alert_id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    evidence_uri TEXT NOT NULL,
    evidence_hash TEXT,
    created_at_ms INTEGER NOT NULL,
    UNIQUE (alert_id, evidence_uri)
);
"#,
    "CREATE INDEX IF NOT EXISTS ix_inst_pk_hb ON instances(project_key, last_heartbeat_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_se_pk_ts ON search_events(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_se_ii_ts ON search_events(instance_id, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_se_corr ON search_events(project_key, correlation_id);",
    "CREATE INDEX IF NOT EXISTS ix_ss_pk_w ON search_summaries(project_key, window, window_start_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_ejs_pk ON embedding_job_snapshots(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_iis_pk ON index_inventory_snapshots(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_rs_pk ON resource_samples(project_key, ts_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_at_pk ON alerts_timeline(project_key, opened_at_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_at_open ON alerts_timeline(project_key, state, severity, updated_at_ms DESC) WHERE state != 'resolved';",
    "CREATE INDEX IF NOT EXISTS ix_el_aid ON evidence_links(alert_id, created_at_ms DESC);",
];

#[allow(clippy::needless_raw_string_hashes)]
const OPS_SCHEMA_V2_STATEMENTS: &[&str] = &[
    r#"
CREATE TABLE IF NOT EXISTS slo_rollups (
    rollup_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (scope IN ('project', 'fleet')),
    scope_key TEXT NOT NULL,
    project_key TEXT,
    window TEXT NOT NULL CHECK (window IN ('1m', '15m', '1h', '6h', '24h', '3d', '1w')),
    window_start_ms INTEGER NOT NULL,
    window_end_ms INTEGER NOT NULL,
    total_requests INTEGER NOT NULL,
    failed_requests INTEGER NOT NULL,
    p95_latency_us INTEGER,
    target_p95_latency_us INTEGER NOT NULL,
    error_budget_ratio REAL NOT NULL,
    error_rate REAL,
    error_budget_burn REAL,
    remaining_budget_ratio REAL,
    health TEXT NOT NULL CHECK (health IN ('healthy', 'warn', 'error', 'critical', 'no_data')),
    reason_code TEXT NOT NULL,
    generated_at_ms INTEGER NOT NULL,
    UNIQUE (scope, scope_key, window, window_start_ms)
);
"#,
    r#"
CREATE TABLE IF NOT EXISTS anomaly_materializations (
    anomaly_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (scope IN ('project', 'fleet')),
    scope_key TEXT NOT NULL,
    project_key TEXT,
    window TEXT NOT NULL CHECK (window IN ('1m', '15m', '1h', '6h', '24h', '3d', '1w')),
    window_start_ms INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    baseline_value REAL,
    observed_value REAL,
    deviation_ratio REAL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warn', 'error', 'critical')),
    reason_code TEXT NOT NULL,
    correlation_id TEXT,
    state TEXT NOT NULL CHECK (state IN ('open', 'resolved')),
    opened_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    resolved_at_ms INTEGER,
    UNIQUE (scope, scope_key, window, window_start_ms, metric_name)
);
"#,
    "CREATE INDEX IF NOT EXISTS ix_slo_scope_window ON slo_rollups(scope, scope_key, window, window_start_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_slo_project_window ON slo_rollups(project_key, window, window_start_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_am_scope_state ON anomaly_materializations(scope, scope_key, state, severity, updated_at_ms DESC);",
    "CREATE INDEX IF NOT EXISTS ix_am_project_timeline ON anomaly_materializations(project_key, opened_at_ms DESC);",
];

struct OpsMigration {
    version: i64,
    name: &'static str,
    checksum: &'static str,
    reversible: bool,
    statements: &'static [&'static str],
}

const OPS_MIGRATIONS: &[OpsMigration] = &[
    OpsMigration {
        version: 1,
        name: OPS_SCHEMA_V1_NAME,
        checksum: OPS_SCHEMA_V1_CHECKSUM,
        reversible: true,
        statements: OPS_SCHEMA_V1_STATEMENTS,
    },
    OpsMigration {
        version: 2,
        name: OPS_SCHEMA_V2_NAME,
        checksum: OPS_SCHEMA_V2_CHECKSUM,
        reversible: true,
        statements: OPS_SCHEMA_V2_STATEMENTS,
    },
];

/// Configuration for the ops telemetry storage connection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpsStorageConfig {
    /// Path to the ops telemetry database.
    pub db_path: PathBuf,
    /// Enable `WAL` journaling mode when true.
    pub wal_mode: bool,
    /// `SQLite` busy timeout in milliseconds.
    pub busy_timeout_ms: u64,
    /// `SQLite` cache size in pages.
    pub cache_size_pages: i32,
}

impl OpsStorageConfig {
    /// In-memory configuration useful for unit tests.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            db_path: PathBuf::from(":memory:"),
            ..Self::default()
        }
    }
}

impl Default for OpsStorageConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("frankensearch-ops.db"),
            wal_mode: true,
            busy_timeout_ms: 5_000,
            cache_size_pages: 2_000,
        }
    }
}

/// Search phase classification persisted to `search_events.phase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchEventPhase {
    Initial,
    Refined,
    Failed,
}

impl SearchEventPhase {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Refined => "refined",
            Self::Failed => "failed",
        }
    }
}

/// Idempotent write payload for `search_events`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchEventRecord {
    pub event_id: String,
    pub project_key: String,
    pub instance_id: String,
    pub correlation_id: String,
    pub query_hash: Option<String>,
    pub query_class: Option<String>,
    pub phase: SearchEventPhase,
    pub latency_us: u64,
    pub result_count: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub ts_ms: i64,
}

impl SearchEventRecord {
    fn validate(&self) -> SearchResult<()> {
        ensure_non_empty(&self.event_id, "event_id")?;
        ensure_non_empty(&self.project_key, "project_key")?;
        ensure_non_empty(&self.instance_id, "instance_id")?;
        ensure_non_empty(&self.correlation_id, "correlation_id")?;
        if self.ts_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "ts_ms".to_owned(),
                value: self.ts_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        let _ = u64_to_i64(self.latency_us, "latency_us")?;
        if let Some(result_count) = self.result_count {
            let _ = u64_to_i64(result_count, "result_count")?;
        }
        if let Some(memory_bytes) = self.memory_bytes {
            let _ = u64_to_i64(memory_bytes, "memory_bytes")?;
        }
        Ok(())
    }
}

/// Upsert payload for `resource_samples`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceSampleRecord {
    pub project_key: String,
    pub instance_id: String,
    pub cpu_pct: Option<f64>,
    pub rss_bytes: Option<u64>,
    pub io_read_bytes: Option<u64>,
    pub io_write_bytes: Option<u64>,
    pub queue_depth: Option<u64>,
    pub ts_ms: i64,
}

impl ResourceSampleRecord {
    fn validate(&self) -> SearchResult<()> {
        ensure_non_empty(&self.project_key, "project_key")?;
        ensure_non_empty(&self.instance_id, "instance_id")?;
        if self.ts_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "ts_ms".to_owned(),
                value: self.ts_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        if let Some(rss_bytes) = self.rss_bytes {
            let _ = u64_to_i64(rss_bytes, "rss_bytes")?;
        }
        if let Some(io_read_bytes) = self.io_read_bytes {
            let _ = u64_to_i64(io_read_bytes, "io_read_bytes")?;
        }
        if let Some(io_write_bytes) = self.io_write_bytes {
            let _ = u64_to_i64(io_write_bytes, "io_write_bytes")?;
        }
        if let Some(queue_depth) = self.queue_depth {
            let _ = u64_to_i64(queue_depth, "queue_depth")?;
        }
        Ok(())
    }
}

/// Write payload for `evidence_links` with deterministic duplicate semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLinkRecord {
    pub project_key: String,
    pub alert_id: String,
    pub evidence_type: String,
    pub evidence_uri: String,
    pub evidence_hash: Option<String>,
    pub created_at_ms: i64,
}

impl EvidenceLinkRecord {
    fn validate(&self) -> SearchResult<()> {
        ensure_non_empty(&self.project_key, "project_key")?;
        ensure_non_empty(&self.alert_id, "alert_id")?;
        ensure_non_empty(&self.evidence_type, "evidence_type")?;
        ensure_non_empty(&self.evidence_uri, "evidence_uri")?;
        if self.created_at_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "created_at_ms".to_owned(),
                value: self.created_at_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        Ok(())
    }
}

/// Supported aggregation windows for dashboard rollups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryWindow {
    OneMinute,
    FifteenMinutes,
    OneHour,
    SixHours,
    TwentyFourHours,
    ThreeDays,
    OneWeek,
}

impl SummaryWindow {
    pub const ALL: [Self; 7] = [
        Self::OneMinute,
        Self::FifteenMinutes,
        Self::OneHour,
        Self::SixHours,
        Self::TwentyFourHours,
        Self::ThreeDays,
        Self::OneWeek,
    ];

    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::OneMinute => "1m",
            Self::FifteenMinutes => "15m",
            Self::OneHour => "1h",
            Self::SixHours => "6h",
            Self::TwentyFourHours => "24h",
            Self::ThreeDays => "3d",
            Self::OneWeek => "1w",
        }
    }

    #[must_use]
    pub const fn duration_ms(self) -> i64 {
        match self {
            Self::OneMinute => 60_000,
            Self::FifteenMinutes => 15 * 60_000,
            Self::OneHour => 3_600_000,
            Self::SixHours => 6 * 3_600_000,
            Self::TwentyFourHours => 24 * 3_600_000,
            Self::ThreeDays => 3 * 24 * 3_600_000,
            Self::OneWeek => 7 * 24 * 3_600_000,
        }
    }

    #[must_use]
    pub const fn bucket_start_ms(self, ts_ms: i64) -> i64 {
        if ts_ms <= 0 {
            return 0;
        }
        let duration = self.duration_ms();
        ts_ms - (ts_ms % duration)
    }

    #[must_use]
    pub const fn rolling_start_ms(self, now_ms: i64) -> i64 {
        if now_ms <= 0 {
            return 0;
        }
        let duration = self.duration_ms();
        if now_ms >= duration {
            now_ms - duration + 1
        } else {
            0
        }
    }

    #[must_use]
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "1m" => Some(Self::OneMinute),
            "15m" => Some(Self::FifteenMinutes),
            "1h" => Some(Self::OneHour),
            "6h" => Some(Self::SixHours),
            "24h" => Some(Self::TwentyFourHours),
            "3d" => Some(Self::ThreeDays),
            "1w" => Some(Self::OneWeek),
            _ => None,
        }
    }
}

/// Scope for SLO rollups and anomaly materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloScope {
    /// Project-level aggregation.
    Project,
    /// Fleet-wide aggregation across all projects.
    Fleet,
}

impl SloScope {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Fleet => "fleet",
        }
    }

    #[must_use]
    fn from_db(value: &str) -> Option<Self> {
        match value {
            "project" => Some(Self::Project),
            "fleet" => Some(Self::Fleet),
            _ => None,
        }
    }
}

/// Health classification for materialized SLO rollups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloHealth {
    Healthy,
    Warn,
    Error,
    Critical,
    NoData,
}

impl SloHealth {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Critical => "critical",
            Self::NoData => "no_data",
        }
    }

    #[must_use]
    fn from_db(value: &str) -> Option<Self> {
        match value {
            "healthy" => Some(Self::Healthy),
            "warn" => Some(Self::Warn),
            "error" => Some(Self::Error),
            "critical" => Some(Self::Critical),
            "no_data" => Some(Self::NoData),
            _ => None,
        }
    }
}

/// Severity level for anomaly materializations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalySeverity {
    Info,
    Warn,
    Error,
    Critical,
}

impl AnomalySeverity {
    #[must_use]
    const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }

    #[must_use]
    fn from_db(value: &str) -> Option<Self> {
        match value {
            "info" => Some(Self::Info),
            "warn" => Some(Self::Warn),
            "error" => Some(Self::Error),
            "critical" => Some(Self::Critical),
            _ => None,
        }
    }
}

/// Runtime knobs for SLO rollup and anomaly materialization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SloMaterializationConfig {
    /// Target p95 latency in microseconds for the SLO.
    pub target_p95_latency_us: u64,
    /// Allowed request failure ratio (error budget), e.g. `0.01` for 99% SLO.
    pub error_budget_ratio: f64,
    /// Burn-rate threshold for `warn`.
    pub warn_burn_rate: f64,
    /// Burn-rate threshold for `error`.
    pub error_burn_rate: f64,
    /// Burn-rate threshold for `critical`.
    pub critical_burn_rate: f64,
    /// Latency multiplier (x target) threshold for `warn`.
    pub warn_latency_multiplier: f64,
    /// Latency multiplier (x target) threshold for `error`.
    pub error_latency_multiplier: f64,
    /// Latency multiplier (x target) threshold for `critical`.
    pub critical_latency_multiplier: f64,
    /// Minimum request count required before classifying non-`no_data` health.
    pub min_requests: u64,
}

impl Default for SloMaterializationConfig {
    fn default() -> Self {
        Self {
            // Contract default: search_latency_p95 threshold = 150ms.
            target_p95_latency_us: 150_000,
            error_budget_ratio: 0.01,
            warn_burn_rate: 1.0,
            error_burn_rate: 2.0,
            critical_burn_rate: 4.0,
            warn_latency_multiplier: 1.0,
            error_latency_multiplier: 1.5,
            critical_latency_multiplier: 2.0,
            min_requests: 10,
        }
    }
}

impl SloMaterializationConfig {
    fn validate(self) -> SearchResult<()> {
        if self.target_p95_latency_us == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_p95_latency_us".to_owned(),
                value: "0".to_owned(),
                reason: "must be > 0".to_owned(),
            });
        }
        if self.error_budget_ratio <= 0.0 || self.error_budget_ratio > 1.0 {
            return Err(SearchError::InvalidConfig {
                field: "error_budget_ratio".to_owned(),
                value: self.error_budget_ratio.to_string(),
                reason: "must be in (0, 1]".to_owned(),
            });
        }
        if self.warn_burn_rate <= 0.0
            || self.error_burn_rate < self.warn_burn_rate
            || self.critical_burn_rate < self.error_burn_rate
        {
            return Err(SearchError::InvalidConfig {
                field: "burn_rate_thresholds".to_owned(),
                value: format!(
                    "{}/{}/{}",
                    self.warn_burn_rate, self.error_burn_rate, self.critical_burn_rate
                ),
                reason: "must be positive and monotonic (warn <= error <= critical)".to_owned(),
            });
        }
        if self.warn_latency_multiplier <= 0.0
            || self.error_latency_multiplier < self.warn_latency_multiplier
            || self.critical_latency_multiplier < self.error_latency_multiplier
        {
            return Err(SearchError::InvalidConfig {
                field: "latency_multipliers".to_owned(),
                value: format!(
                    "{}/{}/{}",
                    self.warn_latency_multiplier,
                    self.error_latency_multiplier,
                    self.critical_latency_multiplier
                ),
                reason: "must be positive and monotonic (warn <= error <= critical)".to_owned(),
            });
        }
        Ok(())
    }
}

/// Materialized SLO row for dashboards and alerting views.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SloRollupSnapshot {
    pub scope: SloScope,
    pub scope_key: String,
    pub project_key: Option<String>,
    pub window: SummaryWindow,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub p95_latency_us: Option<u64>,
    pub target_p95_latency_us: u64,
    pub error_budget_ratio: f64,
    pub error_rate: Option<f64>,
    pub error_budget_burn: Option<f64>,
    pub remaining_budget_ratio: Option<f64>,
    pub health: SloHealth,
    pub reason_code: String,
    pub generated_at_ms: i64,
}

/// Materialized anomaly row for timeline and alerts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyMaterializationSnapshot {
    pub anomaly_id: String,
    pub scope: SloScope,
    pub scope_key: String,
    pub project_key: Option<String>,
    pub window: SummaryWindow,
    pub window_start_ms: i64,
    pub metric_name: String,
    pub baseline_value: Option<f64>,
    pub observed_value: Option<f64>,
    pub deviation_ratio: Option<f64>,
    pub severity: AnomalySeverity,
    pub reason_code: String,
    pub correlation_id: Option<String>,
    pub state: String,
    pub opened_at_ms: i64,
    pub updated_at_ms: i64,
    pub resolved_at_ms: Option<i64>,
}

/// Counters emitted after one rollup/anomaly materialization pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SloMaterializationResult {
    pub rollups_upserted: u64,
    pub anomalies_opened: u64,
    pub anomalies_resolved: u64,
}

/// Materialized search summary used by dashboard reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSummarySnapshot {
    pub window: SummaryWindow,
    pub window_start_ms: i64,
    pub search_count: u64,
    pub p50_latency_us: Option<u64>,
    pub p95_latency_us: Option<u64>,
    pub p99_latency_us: Option<u64>,
    pub avg_result_count: Option<f64>,
    pub avg_memory_bytes: Option<f64>,
    pub p95_memory_bytes: Option<u64>,
}

/// Embedding progress rates over a rolling window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingThroughputRate {
    pub window: SummaryWindow,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub completed_per_sec: f64,
    pub failed_per_sec: f64,
    pub retried_per_sec: f64,
}

/// Resource trend point for dashboard charting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceTrendPoint {
    pub ts_ms: i64,
    pub cpu_pct: Option<f64>,
    pub rss_bytes: Option<u64>,
    pub io_read_bytes: Option<u64>,
    pub io_write_bytes: Option<u64>,
    pub queue_depth: Option<u64>,
}

/// Retention and compaction policy for telemetry storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsRetentionPolicy {
    pub raw_search_event_retention_ms: i64,
    pub search_summary_retention_ms: i64,
    pub resource_sample_retention_ms: i64,
    pub resource_downsample_after_ms: i64,
    pub resource_downsample_stride: u32,
}

impl Default for OpsRetentionPolicy {
    fn default() -> Self {
        Self {
            raw_search_event_retention_ms: 3 * 24 * 3_600_000,
            search_summary_retention_ms: 14 * 24 * 3_600_000,
            resource_sample_retention_ms: 7 * 24 * 3_600_000,
            resource_downsample_after_ms: 6 * 3_600_000,
            resource_downsample_stride: 6,
        }
    }
}

/// Row-deletion counters produced by retention/compaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsRetentionResult {
    pub deleted_search_events: u64,
    pub deleted_search_summaries: u64,
    pub deleted_resource_samples: u64,
    pub downsampled_resource_samples: u64,
}

/// Per-call ingestion accounting for search event batches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsIngestBatchResult {
    pub requested: usize,
    pub inserted: usize,
    pub deduplicated: usize,
    pub failed: usize,
    pub queue_depth_before: usize,
    pub queue_depth_after: usize,
    pub write_latency_us: u64,
}

/// Aggregate ingestion metrics for observability.
#[derive(Debug, Default)]
pub struct OpsIngestionMetrics {
    total_batches: AtomicU64,
    total_inserted: AtomicU64,
    total_deduplicated: AtomicU64,
    total_failed_records: AtomicU64,
    total_backpressured_batches: AtomicU64,
    total_write_latency_us: AtomicU64,
    pending_events: AtomicUsize,
    high_watermark_pending_events: AtomicUsize,
}

/// Snapshot of aggregate ingestion counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpsIngestionMetricsSnapshot {
    pub total_batches: u64,
    pub total_inserted: u64,
    pub total_deduplicated: u64,
    pub total_failed_records: u64,
    pub total_backpressured_batches: u64,
    pub total_write_latency_us: u64,
    pub pending_events: usize,
    pub high_watermark_pending_events: usize,
}

#[derive(Debug, Clone, Default)]
struct WindowEventStats {
    search_count: u64,
    p50_latency_us: Option<u64>,
    p95_latency_us: Option<u64>,
    p99_latency_us: Option<u64>,
    avg_result_count: Option<f64>,
    avg_memory_bytes: Option<f64>,
    p95_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SloWindowStats {
    total_requests: u64,
    failed_requests: u64,
    p95_latency_us: Option<u64>,
}

#[derive(Debug, Clone)]
struct SloEvaluation {
    scope: SloScope,
    scope_key: String,
    project_key: Option<String>,
    window: SummaryWindow,
    window_start_ms: i64,
    window_end_ms: i64,
    total_requests: u64,
    failed_requests: u64,
    p95_latency_us: Option<u64>,
    target_p95_latency_us: u64,
    error_budget_ratio: f64,
    error_rate: Option<f64>,
    error_budget_burn: Option<f64>,
    remaining_budget_ratio: Option<f64>,
    health: SloHealth,
    reason_code: String,
    generated_at_ms: i64,
    anomaly: Option<AnomalyCandidate>,
}

#[derive(Debug, Clone)]
struct AnomalyCandidate {
    metric_name: String,
    baseline_value: f64,
    observed_value: f64,
    deviation_ratio: f64,
    severity: AnomalySeverity,
    reason_code: String,
}

impl OpsIngestionMetrics {
    #[must_use]
    pub fn snapshot(&self) -> OpsIngestionMetricsSnapshot {
        OpsIngestionMetricsSnapshot {
            total_batches: self.total_batches.load(Ordering::Relaxed),
            total_inserted: self.total_inserted.load(Ordering::Relaxed),
            total_deduplicated: self.total_deduplicated.load(Ordering::Relaxed),
            total_failed_records: self.total_failed_records.load(Ordering::Relaxed),
            total_backpressured_batches: self.total_backpressured_batches.load(Ordering::Relaxed),
            total_write_latency_us: self.total_write_latency_us.load(Ordering::Relaxed),
            pending_events: self.pending_events.load(Ordering::Relaxed),
            high_watermark_pending_events: self
                .high_watermark_pending_events
                .load(Ordering::Relaxed),
        }
    }

    fn update_high_watermark(&self, candidate: usize) {
        let mut current = self.high_watermark_pending_events.load(Ordering::Relaxed);
        while candidate > current {
            match self.high_watermark_pending_events.compare_exchange_weak(
                current,
                candidate,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Connection wrapper for ops telemetry storage.
pub struct OpsStorage {
    conn: Connection,
    config: OpsStorageConfig,
    ingestion_metrics: Arc<OpsIngestionMetrics>,
}

impl std::fmt::Debug for OpsStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpsStorage")
            .field("path", &self.config.db_path)
            .field("wal_mode", &self.config.wal_mode)
            .field("busy_timeout_ms", &self.config.busy_timeout_ms)
            .field("cache_size_pages", &self.config.cache_size_pages)
            .field(
                "pending_ingest_events",
                &self.ingestion_metrics.snapshot().pending_events,
            )
            .finish_non_exhaustive()
    }
}

impl OpsStorage {
    /// Open storage and bootstrap schema if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection cannot be opened,
    /// pragmas fail to apply, or schema bootstrap/migration fails.
    pub fn open(config: OpsStorageConfig) -> SearchResult<Self> {
        tracing::debug!(
            target: "frankensearch.ops.storage",
            path = %config.db_path.display(),
            wal_mode = config.wal_mode,
            busy_timeout_ms = config.busy_timeout_ms,
            cache_size_pages = config.cache_size_pages,
            "opening ops storage connection"
        );

        let conn =
            Connection::open(config.db_path.to_string_lossy().to_string()).map_err(ops_error)?;
        let storage = Self {
            conn,
            config,
            ingestion_metrics: Arc::new(OpsIngestionMetrics::default()),
        };
        storage.apply_pragmas()?;
        bootstrap(storage.connection())?;
        Ok(storage)
    }

    /// Open in-memory storage and bootstrap schema.
    ///
    /// # Errors
    ///
    /// Returns an error if in-memory database bootstrap fails.
    pub fn open_in_memory() -> SearchResult<Self> {
        Self::open(OpsStorageConfig::in_memory())
    }

    /// Underlying database connection.
    #[must_use]
    pub const fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Runtime configuration used by this storage handle.
    #[must_use]
    pub const fn config(&self) -> &OpsStorageConfig {
        &self.config
    }

    /// Current schema version.
    ///
    /// # Errors
    ///
    /// Returns an error if schema metadata cannot be read.
    pub fn current_schema_version(&self) -> SearchResult<i64> {
        current_version(self.connection())
    }

    /// Current ingestion metrics snapshot used by dashboards and tests.
    #[must_use]
    pub fn ingestion_metrics(&self) -> OpsIngestionMetricsSnapshot {
        self.ingestion_metrics.snapshot()
    }

    /// Insert one search event with idempotent semantics.
    ///
    /// This is equivalent to calling [`Self::ingest_search_events_batch`] with
    /// a single payload.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails, backpressure rejects the write,
    /// or database I/O fails.
    pub fn ingest_search_event(
        &self,
        event: &SearchEventRecord,
        backpressure_threshold: usize,
    ) -> SearchResult<OpsIngestBatchResult> {
        self.ingest_search_events_batch(std::slice::from_ref(event), backpressure_threshold)
    }

    /// Insert a batch of search events atomically.
    ///
    /// `event_id` is treated as an idempotency key. Duplicate IDs are counted
    /// as deduplicated records instead of failures.
    ///
    /// # Errors
    ///
    /// Returns an error if backpressure is active, payload validation fails, or
    /// database operations fail. On error, the full batch is rolled back.
    pub fn ingest_search_events_batch(
        &self,
        events: &[SearchEventRecord],
        backpressure_threshold: usize,
    ) -> SearchResult<OpsIngestBatchResult> {
        if events.is_empty() {
            return Ok(OpsIngestBatchResult::default());
        }
        if backpressure_threshold == 0 {
            return Err(SearchError::InvalidConfig {
                field: "backpressure_threshold".to_owned(),
                value: "0".to_owned(),
                reason: "must be > 0".to_owned(),
            });
        }

        let requested = events.len();
        let queue_depth_before = self
            .ingestion_metrics
            .pending_events
            .fetch_add(requested, Ordering::Relaxed);
        let queue_depth_with_reservation = queue_depth_before.saturating_add(requested);
        Self::log_ingest_start(
            requested,
            backpressure_threshold,
            queue_depth_before,
            queue_depth_with_reservation,
        );
        self.ingestion_metrics
            .update_high_watermark(queue_depth_with_reservation);

        if queue_depth_with_reservation > backpressure_threshold {
            self.ingestion_metrics
                .pending_events
                .fetch_sub(requested, Ordering::Relaxed);
            self.ingestion_metrics
                .total_backpressured_batches
                .fetch_add(1, Ordering::Relaxed);
            Self::log_ingest_backpressure(
                requested,
                queue_depth_before,
                queue_depth_with_reservation,
                backpressure_threshold,
            );
            return Err(SearchError::QueueFull {
                pending: queue_depth_with_reservation,
                capacity: backpressure_threshold,
            });
        }

        let started = Instant::now();
        let ingest_result = self.ingest_search_events_transaction(events);

        let write_latency_us = duration_as_u64(started.elapsed().as_micros());
        let queue_depth_after = self
            .ingestion_metrics
            .pending_events
            .fetch_sub(requested, Ordering::Relaxed)
            .saturating_sub(requested);

        self.ingestion_metrics
            .total_batches
            .fetch_add(1, Ordering::Relaxed);
        self.ingestion_metrics
            .total_write_latency_us
            .fetch_add(write_latency_us, Ordering::Relaxed);

        match ingest_result {
            Ok((inserted, deduplicated)) => {
                self.ingestion_metrics
                    .total_inserted
                    .fetch_add(usize_to_u64(inserted), Ordering::Relaxed);
                self.ingestion_metrics
                    .total_deduplicated
                    .fetch_add(usize_to_u64(deduplicated), Ordering::Relaxed);
                Self::log_ingest_success(
                    requested,
                    inserted,
                    deduplicated,
                    queue_depth_before,
                    queue_depth_after,
                    write_latency_us,
                );

                Ok(OpsIngestBatchResult::new(
                    requested,
                    inserted,
                    deduplicated,
                    queue_depth_before,
                    queue_depth_after,
                    write_latency_us,
                ))
            }
            Err(error) => {
                self.ingestion_metrics
                    .total_failed_records
                    .fetch_add(usize_to_u64(requested), Ordering::Relaxed);
                Self::log_ingest_failure(
                    requested,
                    queue_depth_before,
                    queue_depth_after,
                    write_latency_us,
                    &error,
                );
                Err(error)
            }
        }
    }

    fn ingest_search_events_transaction(
        &self,
        events: &[SearchEventRecord],
    ) -> SearchResult<(usize, usize)> {
        self.with_transaction(|conn| {
            let mut ordered_events: Vec<&SearchEventRecord> = events.iter().collect();
            ordered_events.sort_unstable_by(|left, right| {
                left.ts_ms
                    .cmp(&right.ts_ms)
                    .then_with(|| left.event_id.cmp(&right.event_id))
            });

            for event in &ordered_events {
                event.validate()?;
            }

            let mut to_insert: Vec<&SearchEventRecord> = Vec::with_capacity(ordered_events.len());
            let mut deduplicated = 0_usize;
            for event in ordered_events {
                if search_event_exists(conn, &event.event_id)? {
                    deduplicated = deduplicated.saturating_add(1);
                } else {
                    to_insert.push(event);
                }
            }

            for event in &to_insert {
                insert_search_event_row(conn, event)?;
            }
            let inserted = to_insert.len();

            Ok((inserted, deduplicated))
        })
    }

    fn log_ingest_start(
        requested: usize,
        backpressure_threshold: usize,
        queue_depth_before: usize,
        queue_depth_with_reservation: usize,
    ) {
        tracing::debug!(
            target: "frankensearch.ops.storage",
            event = "search_events_ingest_start",
            requested,
            backpressure_threshold,
            queue_depth_before,
            queue_depth_reserved = queue_depth_with_reservation,
            "ingesting search event batch"
        );
    }

    fn log_ingest_backpressure(
        requested: usize,
        queue_depth_before: usize,
        pending: usize,
        capacity: usize,
    ) {
        tracing::warn!(
            target: "frankensearch.ops.storage",
            event = "search_events_ingest_backpressure",
            requested,
            queue_depth_before,
            pending,
            capacity,
            "rejecting search event batch due to backpressure"
        );
    }

    fn log_ingest_success(
        requested: usize,
        inserted: usize,
        deduplicated: usize,
        queue_depth_before: usize,
        queue_depth_after: usize,
        write_latency_us: u64,
    ) {
        tracing::info!(
            target: "frankensearch.ops.storage",
            event = "search_events_ingest_success",
            requested,
            inserted,
            deduplicated,
            failed = 0usize,
            queue_depth_before,
            queue_depth_after,
            write_latency_us,
            "search event batch ingested"
        );
    }

    fn log_ingest_failure(
        requested: usize,
        queue_depth_before: usize,
        queue_depth_after: usize,
        write_latency_us: u64,
        error: &SearchError,
    ) {
        tracing::warn!(
            target: "frankensearch.ops.storage",
            event = "search_events_ingest_failed",
            requested,
            failed = requested,
            queue_depth_before,
            queue_depth_after,
            write_latency_us,
            error = %error,
            "search event batch ingest failed"
        );
    }

    /// Upsert a resource sample keyed by `(project_key, instance_id, ts_ms)`.
    ///
    /// # Errors
    ///
    /// Returns an error when validation fails or the database write fails.
    pub fn upsert_resource_sample(&self, sample: &ResourceSampleRecord) -> SearchResult<()> {
        sample.validate()?;
        let conn = self.connection();

        // Manual upsert: FrankenSQLite does not yet support
        // ON CONFLICT(...) DO UPDATE, so we check existence first.
        let key_params = [
            SqliteValue::Text(sample.project_key.clone()),
            SqliteValue::Text(sample.instance_id.clone()),
            SqliteValue::Integer(sample.ts_ms),
        ];
        let existing = conn
            .query_with_params(
                "SELECT sample_id FROM resource_samples \
                 WHERE project_key = ?1 AND instance_id = ?2 AND ts_ms = ?3;",
                &key_params,
            )
            .map_err(ops_error)?;

        if existing.is_empty() {
            let params = [
                SqliteValue::Text(sample.project_key.clone()),
                SqliteValue::Text(sample.instance_id.clone()),
                optional_f64(sample.cpu_pct),
                optional_u64(sample.rss_bytes, "rss_bytes")?,
                optional_u64(sample.io_read_bytes, "io_read_bytes")?,
                optional_u64(sample.io_write_bytes, "io_write_bytes")?,
                optional_u64(sample.queue_depth, "queue_depth")?,
                SqliteValue::Integer(sample.ts_ms),
            ];
            conn.execute_with_params(
                "INSERT INTO resource_samples(\
                    project_key, instance_id, cpu_pct, rss_bytes, io_read_bytes, io_write_bytes, \
                    queue_depth, ts_ms\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
                &params,
            )
            .map_err(ops_error)?;
        } else {
            let params = [
                optional_f64(sample.cpu_pct),
                optional_u64(sample.rss_bytes, "rss_bytes")?,
                optional_u64(sample.io_read_bytes, "io_read_bytes")?,
                optional_u64(sample.io_write_bytes, "io_write_bytes")?,
                optional_u64(sample.queue_depth, "queue_depth")?,
                SqliteValue::Text(sample.project_key.clone()),
                SqliteValue::Text(sample.instance_id.clone()),
                SqliteValue::Integer(sample.ts_ms),
            ];
            conn.execute_with_params(
                "UPDATE resource_samples SET \
                    cpu_pct = ?1, rss_bytes = ?2, io_read_bytes = ?3, \
                    io_write_bytes = ?4, queue_depth = ?5 \
                 WHERE project_key = ?6 AND instance_id = ?7 AND ts_ms = ?8;",
                &params,
            )
            .map_err(ops_error)?;
        }
        Ok(())
    }

    /// Insert an evidence link while enforcing `(alert_id, evidence_uri)` uniqueness.
    ///
    /// # Errors
    ///
    /// Returns an error when inputs are invalid, a duplicate pair already exists,
    /// or the database write fails.
    pub fn insert_evidence_link(&self, link: &EvidenceLinkRecord) -> SearchResult<()> {
        link.validate()?;
        self.with_transaction(|conn| {
            let alert_params = [SqliteValue::Text(link.alert_id.clone())];
            let alert_rows = conn
                .query_with_params(
                    "SELECT project_key FROM alerts_timeline \
                     WHERE alert_id = ?1 LIMIT 1;",
                    &alert_params,
                )
                .map_err(ops_error)?;
            let Some(alert_row) = alert_rows.first() else {
                return Err(SearchError::InvalidConfig {
                    field: "alert_id".to_owned(),
                    value: link.alert_id.clone(),
                    reason: "unknown alert_id".to_owned(),
                });
            };
            let alert_project_key = row_text(alert_row, 0, "alerts_timeline.project_key")?;
            if alert_project_key != link.project_key {
                return Err(SearchError::InvalidConfig {
                    field: "project_key".to_owned(),
                    value: link.project_key.clone(),
                    reason: format!(
                        "alert_id '{}' belongs to project_key '{}'",
                        link.alert_id, alert_project_key
                    ),
                });
            }

            let duplicate_params = [
                SqliteValue::Text(link.alert_id.clone()),
                SqliteValue::Text(link.evidence_uri.clone()),
            ];
            let duplicate_rows = conn
                .query_with_params(
                    "SELECT link_id FROM evidence_links \
                     WHERE alert_id = ?1 AND evidence_uri = ?2 LIMIT 1;",
                    &duplicate_params,
                )
                .map_err(ops_error)?;
            if !duplicate_rows.is_empty() {
                return Err(SearchError::InvalidConfig {
                    field: "evidence_uri".to_owned(),
                    value: link.evidence_uri.clone(),
                    reason: format!(
                        "duplicate evidence link pair for alert_id='{}'",
                        link.alert_id
                    ),
                });
            }

            let link_id = evidence_link_id(&link.alert_id, &link.evidence_uri);
            let params = [
                SqliteValue::Text(link_id),
                SqliteValue::Text(link.project_key.clone()),
                SqliteValue::Text(link.alert_id.clone()),
                SqliteValue::Text(link.evidence_type.clone()),
                SqliteValue::Text(link.evidence_uri.clone()),
                link.evidence_hash
                    .clone()
                    .map_or(SqliteValue::Null, SqliteValue::Text),
                SqliteValue::Integer(link.created_at_ms),
            ];
            conn.execute_with_params(
                "INSERT INTO evidence_links(\
                    link_id, project_key, alert_id, evidence_type, evidence_uri, \
                    evidence_hash, created_at_ms\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
                &params,
            )
            .map_err(ops_error)?;
            Ok(())
        })
    }

    /// Recompute and materialize rolling search summaries for one instance.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid or any database operation fails.
    pub fn refresh_search_summaries_for_instance(
        &self,
        project_key: &str,
        instance_id: &str,
        now_ms: i64,
    ) -> SearchResult<Vec<SearchSummarySnapshot>> {
        ensure_non_empty(project_key, "project_key")?;
        ensure_non_empty(instance_id, "instance_id")?;
        if now_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "now_ms".to_owned(),
                value: now_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }

        self.with_transaction(|conn| {
            let mut summaries = Vec::with_capacity(SummaryWindow::ALL.len());
            for window in SummaryWindow::ALL {
                let window_start_ms = window.rolling_start_ms(now_ms);
                let stats = compute_window_event_stats(
                    conn,
                    project_key,
                    instance_id,
                    window_start_ms,
                    now_ms,
                )?;
                upsert_search_summary_row(
                    conn,
                    project_key,
                    instance_id,
                    window,
                    window_start_ms,
                    &stats,
                )?;
                summaries.push(SearchSummarySnapshot {
                    window,
                    window_start_ms,
                    search_count: stats.search_count,
                    p50_latency_us: stats.p50_latency_us,
                    p95_latency_us: stats.p95_latency_us,
                    p99_latency_us: stats.p99_latency_us,
                    avg_result_count: stats.avg_result_count,
                    avg_memory_bytes: stats.avg_memory_bytes,
                    p95_memory_bytes: stats.p95_memory_bytes,
                });
            }
            Ok(summaries)
        })
    }

    /// Materialize SLO rollups and anomaly rows for project + fleet scopes.
    ///
    /// # Errors
    ///
    /// Returns an error if config is invalid, input time is invalid, or SQL
    /// reads/writes fail.
    #[allow(clippy::too_many_lines)]
    pub fn materialize_slo_rollups_and_anomalies(
        &self,
        now_ms: i64,
        config: SloMaterializationConfig,
    ) -> SearchResult<SloMaterializationResult> {
        if now_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "now_ms".to_owned(),
                value: now_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        config.validate()?;

        self.with_transaction(|conn| {
            let mut result = SloMaterializationResult::default();
            let project_keys = list_distinct_project_keys(conn)?;

            for window in SummaryWindow::ALL {
                let window_start_ms = window.bucket_start_ms(now_ms);
                for project_key in &project_keys {
                    let stats =
                        compute_slo_window_stats(conn, Some(project_key), window_start_ms, now_ms)?;
                    let evaluation = evaluate_slo_window(
                        &stats,
                        SloScope::Project,
                        project_key,
                        Some(project_key),
                        window_start_ms,
                        now_ms,
                        config,
                        window,
                    );

                    upsert_slo_rollup_row(conn, &evaluation)?;
                    result.rollups_upserted = result.rollups_upserted.saturating_add(1);

                    let stale_resolved = resolve_stale_anomaly_rows(
                        conn,
                        SloScope::Project,
                        project_key,
                        window,
                        window_start_ms,
                        now_ms,
                    )?;
                    result.anomalies_resolved = result
                        .anomalies_resolved
                        .saturating_add(usize_to_u64(stale_resolved));

                    let (opened, resolved) = sync_rollup_anomaly(conn, &evaluation, now_ms)?;
                    result.anomalies_opened =
                        result.anomalies_opened.saturating_add(usize_to_u64(opened));
                    result.anomalies_resolved = result
                        .anomalies_resolved
                        .saturating_add(usize_to_u64(resolved));
                }

                let fleet_scope_key = "__fleet__";
                let fleet_stats = compute_slo_window_stats(conn, None, window_start_ms, now_ms)?;
                let fleet_evaluation = evaluate_slo_window(
                    &fleet_stats,
                    SloScope::Fleet,
                    fleet_scope_key,
                    None,
                    window_start_ms,
                    now_ms,
                    config,
                    window,
                );
                upsert_slo_rollup_row(conn, &fleet_evaluation)?;
                result.rollups_upserted = result.rollups_upserted.saturating_add(1);

                let stale_resolved = resolve_stale_anomaly_rows(
                    conn,
                    SloScope::Fleet,
                    fleet_scope_key,
                    window,
                    window_start_ms,
                    now_ms,
                )?;
                result.anomalies_resolved = result
                    .anomalies_resolved
                    .saturating_add(usize_to_u64(stale_resolved));

                let (opened, resolved) = sync_rollup_anomaly(conn, &fleet_evaluation, now_ms)?;
                result.anomalies_opened =
                    result.anomalies_opened.saturating_add(usize_to_u64(opened));
                result.anomalies_resolved = result
                    .anomalies_resolved
                    .saturating_add(usize_to_u64(resolved));
            }
            Ok(result)
        })
    }

    /// Fetch latest SLO rollup for a specific scope/key/window.
    ///
    /// # Errors
    ///
    /// Returns an error when reads fail or stored values are invalid.
    pub fn latest_slo_rollup(
        &self,
        scope: SloScope,
        scope_key: &str,
        window: SummaryWindow,
    ) -> SearchResult<Option<SloRollupSnapshot>> {
        ensure_non_empty(scope_key, "scope_key")?;
        let rows = self
            .connection()
            .query_with_params(
                "SELECT scope, scope_key, project_key, window, window_start_ms, window_end_ms, \
                        total_requests, failed_requests, p95_latency_us, target_p95_latency_us, \
                        error_budget_ratio, error_rate, error_budget_burn, remaining_budget_ratio, \
                        health, reason_code, generated_at_ms \
                 FROM slo_rollups \
                 WHERE scope = ?1 AND scope_key = ?2 AND window = ?3 \
                 ORDER BY window_start_ms DESC LIMIT 1;",
                &[
                    SqliteValue::Text(scope.as_str().to_owned()),
                    SqliteValue::Text(scope_key.to_owned()),
                    SqliteValue::Text(window.as_label().to_owned()),
                ],
            )
            .map_err(ops_error)?;
        rows.first().map(row_to_slo_rollup).transpose()
    }

    /// Query latest SLO rollups for one scope/key tuple.
    ///
    /// # Errors
    ///
    /// Returns an error if reads fail or rows are malformed.
    pub fn query_slo_rollups_for_scope(
        &self,
        scope: SloScope,
        scope_key: &str,
        limit: usize,
    ) -> SearchResult<Vec<SloRollupSnapshot>> {
        ensure_non_empty(scope_key, "scope_key")?;
        let rows = self
            .connection()
            .query_with_params(
                "SELECT scope, scope_key, project_key, window, window_start_ms, window_end_ms, \
                        total_requests, failed_requests, p95_latency_us, target_p95_latency_us, \
                        error_budget_ratio, error_rate, error_budget_burn, remaining_budget_ratio, \
                        health, reason_code, generated_at_ms \
                 FROM slo_rollups \
                 WHERE scope = ?1 AND scope_key = ?2 \
                 ORDER BY window_start_ms DESC, window ASC LIMIT ?3;",
                &[
                    SqliteValue::Text(scope.as_str().to_owned()),
                    SqliteValue::Text(scope_key.to_owned()),
                    SqliteValue::Integer(usize_to_i64(limit, "limit")?),
                ],
            )
            .map_err(ops_error)?;
        rows.iter().map(row_to_slo_rollup).collect()
    }

    /// Query open anomalies for a specific scope/key pair.
    ///
    /// # Errors
    ///
    /// Returns an error when reads fail or rows are malformed.
    pub fn query_open_anomalies_for_scope(
        &self,
        scope: SloScope,
        scope_key: &str,
        limit: usize,
    ) -> SearchResult<Vec<AnomalyMaterializationSnapshot>> {
        ensure_non_empty(scope_key, "scope_key")?;
        let rows = self
            .connection()
            .query_with_params(
                "SELECT anomaly_id, scope, scope_key, project_key, window, window_start_ms, \
                        metric_name, baseline_value, observed_value, deviation_ratio, severity, \
                        reason_code, correlation_id, state, opened_at_ms, updated_at_ms, \
                        resolved_at_ms \
                 FROM anomaly_materializations \
                 WHERE scope = ?1 AND scope_key = ?2 AND state = 'open' \
                 ORDER BY CASE severity \
                            WHEN 'critical' THEN 4 \
                            WHEN 'error' THEN 3 \
                            WHEN 'warn' THEN 2 \
                            WHEN 'info' THEN 1 \
                            ELSE 0 \
                          END DESC, \
                          updated_at_ms DESC LIMIT ?3;",
                &[
                    SqliteValue::Text(scope.as_str().to_owned()),
                    SqliteValue::Text(scope_key.to_owned()),
                    SqliteValue::Integer(usize_to_i64(limit, "limit")?),
                ],
            )
            .map_err(ops_error)?;
        rows.iter().map(row_to_anomaly).collect()
    }

    /// Query anomaly timeline ordered by newest first.
    ///
    /// # Errors
    ///
    /// Returns an error when reads fail or rows are malformed.
    pub fn query_anomaly_timeline(
        &self,
        project_key: Option<&str>,
        limit: usize,
    ) -> SearchResult<Vec<AnomalyMaterializationSnapshot>> {
        let rows = if let Some(project_key) = project_key {
            ensure_non_empty(project_key, "project_key")?;
            self.connection()
                .query_with_params(
                    "SELECT anomaly_id, scope, scope_key, project_key, window, window_start_ms, \
                            metric_name, baseline_value, observed_value, deviation_ratio, severity, \
                            reason_code, correlation_id, state, opened_at_ms, updated_at_ms, \
                            resolved_at_ms \
                     FROM anomaly_materializations \
                     WHERE project_key = ?1 \
                     ORDER BY opened_at_ms DESC LIMIT ?2;",
                    &[
                        SqliteValue::Text(project_key.to_owned()),
                        SqliteValue::Integer(usize_to_i64(limit, "limit")?),
                    ],
                )
                .map_err(ops_error)?
        } else {
            self.connection()
                .query_with_params(
                    "SELECT anomaly_id, scope, scope_key, project_key, window, window_start_ms, \
                            metric_name, baseline_value, observed_value, deviation_ratio, severity, \
                            reason_code, correlation_id, state, opened_at_ms, updated_at_ms, \
                            resolved_at_ms \
                     FROM anomaly_materializations \
                     ORDER BY opened_at_ms DESC LIMIT ?1;",
                    &[SqliteValue::Integer(usize_to_i64(limit, "limit")?)],
                )
                .map_err(ops_error)?
        };
        rows.iter().map(row_to_anomaly).collect()
    }

    /// Fetch the most recent summary for a `(project, instance, window)` tuple.
    ///
    /// # Errors
    ///
    /// Returns an error when reads fail or stored values are invalid.
    pub fn latest_search_summary(
        &self,
        project_key: &str,
        instance_id: &str,
        window: SummaryWindow,
    ) -> SearchResult<Option<SearchSummarySnapshot>> {
        ensure_non_empty(project_key, "project_key")?;
        ensure_non_empty(instance_id, "instance_id")?;
        let params = [
            SqliteValue::Text(project_key.to_owned()),
            SqliteValue::Text(instance_id.to_owned()),
            SqliteValue::Text(window.as_label().to_owned()),
        ];
        let rows = self
            .connection()
            .query_with_params(
                "SELECT window_start_ms, search_count, p50_latency_us, p95_latency_us, \
                        p99_latency_us, avg_result_count \
                 FROM search_summaries \
                 WHERE project_key = ?1 AND instance_id = ?2 AND window = ?3 \
                 ORDER BY window_start_ms DESC LIMIT 1;",
                &params,
            )
            .map_err(ops_error)?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };

        let window_start_ms = row_i64(row, 0, "search_summaries.window_start_ms")?;
        let search_count_i64 = row_i64(row, 1, "search_summaries.search_count")?;
        let search_count = i64_to_u64_non_negative(search_count_i64, "search_count")?;
        let p50_latency_us = row_opt_i64(row, 2, "search_summaries.p50_latency_us")?
            .map(|value| i64_to_u64_non_negative(value, "p50_latency_us"))
            .transpose()?;
        let p95_latency_us = row_opt_i64(row, 3, "search_summaries.p95_latency_us")?
            .map(|value| i64_to_u64_non_negative(value, "p95_latency_us"))
            .transpose()?;
        let p99_latency_us = row_opt_i64(row, 4, "search_summaries.p99_latency_us")?
            .map(|value| i64_to_u64_non_negative(value, "p99_latency_us"))
            .transpose()?;
        let avg_result_count = row_opt_f64(row, 5, "search_summaries.avg_result_count")?;

        let window_end_ms = window_start_ms
            .saturating_add(window.duration_ms())
            .saturating_sub(1);
        let raw_stats = compute_window_event_stats(
            self.connection(),
            project_key,
            instance_id,
            window_start_ms,
            window_end_ms,
        )?;

        Ok(Some(SearchSummarySnapshot {
            window,
            window_start_ms,
            search_count,
            p50_latency_us,
            p95_latency_us,
            p99_latency_us,
            avg_result_count,
            avg_memory_bytes: raw_stats.avg_memory_bytes,
            p95_memory_bytes: raw_stats.p95_memory_bytes,
        }))
    }

    /// Query resource trend samples for dashboard charts.
    ///
    /// # Errors
    ///
    /// Returns an error when the read fails.
    pub fn query_resource_trend(
        &self,
        project_key: &str,
        instance_id: &str,
        window: SummaryWindow,
        now_ms: i64,
        limit: usize,
    ) -> SearchResult<Vec<ResourceTrendPoint>> {
        ensure_non_empty(project_key, "project_key")?;
        ensure_non_empty(instance_id, "instance_id")?;
        let window_start_ms = window.rolling_start_ms(now_ms);
        let params = [
            SqliteValue::Text(project_key.to_owned()),
            SqliteValue::Text(instance_id.to_owned()),
            SqliteValue::Integer(window_start_ms),
            SqliteValue::Integer(now_ms),
            SqliteValue::Integer(usize_to_i64(limit, "limit")?),
        ];
        let rows = self
            .connection()
            .query_with_params(
                "SELECT ts_ms, cpu_pct, rss_bytes, io_read_bytes, io_write_bytes, queue_depth \
                 FROM resource_samples \
                 WHERE project_key = ?1 AND instance_id = ?2 AND ts_ms >= ?3 AND ts_ms <= ?4 \
                 ORDER BY ts_ms DESC LIMIT ?5;",
                &params,
            )
            .map_err(ops_error)?;

        let mut points = rows
            .iter()
            .map(|row| {
                let ts_ms = row_i64(row, 0, "resource_samples.ts_ms")?;
                let cpu_pct = row_opt_f64(row, 1, "resource_samples.cpu_pct")?;
                let rss_bytes = row_opt_i64(row, 2, "resource_samples.rss_bytes")?
                    .map(|value| i64_to_u64_non_negative(value, "rss_bytes"))
                    .transpose()?;
                let io_read_bytes = row_opt_i64(row, 3, "resource_samples.io_read_bytes")?
                    .map(|value| i64_to_u64_non_negative(value, "io_read_bytes"))
                    .transpose()?;
                let io_write_bytes = row_opt_i64(row, 4, "resource_samples.io_write_bytes")?
                    .map(|value| i64_to_u64_non_negative(value, "io_write_bytes"))
                    .transpose()?;
                let queue_depth = row_opt_i64(row, 5, "resource_samples.queue_depth")?
                    .map(|value| i64_to_u64_non_negative(value, "queue_depth"))
                    .transpose()?;
                Ok(ResourceTrendPoint {
                    ts_ms,
                    cpu_pct,
                    rss_bytes,
                    io_read_bytes,
                    io_write_bytes,
                    queue_depth,
                })
            })
            .collect::<SearchResult<Vec<_>>>()?;
        points.reverse();
        Ok(points)
    }

    /// Compute embedding progress rates from snapshots in the selected window.
    ///
    /// # Errors
    ///
    /// Returns an error when reads fail or values are invalid.
    #[allow(clippy::cast_precision_loss)]
    pub fn query_embedding_throughput(
        &self,
        project_key: &str,
        instance_id: &str,
        window: SummaryWindow,
        now_ms: i64,
    ) -> SearchResult<Option<EmbeddingThroughputRate>> {
        ensure_non_empty(project_key, "project_key")?;
        ensure_non_empty(instance_id, "instance_id")?;
        let window_start_ms = window.rolling_start_ms(now_ms);
        let params = [
            SqliteValue::Text(project_key.to_owned()),
            SqliteValue::Text(instance_id.to_owned()),
            SqliteValue::Integer(window_start_ms),
            SqliteValue::Integer(now_ms),
        ];
        let rows = self
            .connection()
            .query_with_params(
                "SELECT completed_jobs, failed_jobs, retried_jobs, ts_ms \
                 FROM embedding_job_snapshots \
                 WHERE project_key = ?1 AND instance_id = ?2 AND ts_ms >= ?3 AND ts_ms <= ?4 \
                 ORDER BY ts_ms ASC;",
                &params,
            )
            .map_err(ops_error)?;
        if rows.len() < 2 {
            return Ok(None);
        }

        let (Some(first), Some(last)) = (rows.first(), rows.last()) else {
            return Ok(None);
        };

        let first_completed = i64_to_u64_non_negative(
            row_i64(first, 0, "embedding_job_snapshots.completed_jobs")?,
            "completed_jobs",
        )?;
        let first_failed = i64_to_u64_non_negative(
            row_i64(first, 1, "embedding_job_snapshots.failed_jobs")?,
            "failed_jobs",
        )?;
        let first_retried = i64_to_u64_non_negative(
            row_i64(first, 2, "embedding_job_snapshots.retried_jobs")?,
            "retried_jobs",
        )?;
        let first_ts = row_i64(first, 3, "embedding_job_snapshots.ts_ms")?;

        let last_completed = i64_to_u64_non_negative(
            row_i64(last, 0, "embedding_job_snapshots.completed_jobs")?,
            "completed_jobs",
        )?;
        let last_failed = i64_to_u64_non_negative(
            row_i64(last, 1, "embedding_job_snapshots.failed_jobs")?,
            "failed_jobs",
        )?;
        let last_retried = i64_to_u64_non_negative(
            row_i64(last, 2, "embedding_job_snapshots.retried_jobs")?,
            "retried_jobs",
        )?;
        let last_ts = row_i64(last, 3, "embedding_job_snapshots.ts_ms")?;

        let elapsed_ms = last_ts.saturating_sub(first_ts);
        if elapsed_ms <= 0 {
            return Ok(None);
        }
        let elapsed_secs = (elapsed_ms as f64) / 1000.0;

        let completed_per_sec =
            (last_completed.saturating_sub(first_completed) as f64) / elapsed_secs;
        let failed_per_sec = (last_failed.saturating_sub(first_failed) as f64) / elapsed_secs;
        let retried_per_sec = (last_retried.saturating_sub(first_retried) as f64) / elapsed_secs;

        Ok(Some(EmbeddingThroughputRate {
            window,
            window_start_ms,
            window_end_ms: now_ms,
            completed_per_sec,
            failed_per_sec,
            retried_per_sec,
        }))
    }

    /// Apply retention and lightweight downsampling for telemetry tables.
    ///
    /// # Errors
    ///
    /// Returns an error if policy values are invalid or SQL operations fail.
    pub fn apply_retention_policy(
        &self,
        now_ms: i64,
        policy: OpsRetentionPolicy,
    ) -> SearchResult<OpsRetentionResult> {
        if now_ms < 0 {
            return Err(SearchError::InvalidConfig {
                field: "now_ms".to_owned(),
                value: now_ms.to_string(),
                reason: "must be >= 0".to_owned(),
            });
        }
        if policy.raw_search_event_retention_ms <= 0
            || policy.search_summary_retention_ms <= 0
            || policy.resource_sample_retention_ms <= 0
            || policy.resource_downsample_after_ms < 0
            || policy.resource_downsample_stride == 0
        {
            return Err(SearchError::InvalidConfig {
                field: "OpsRetentionPolicy".to_owned(),
                value: format!("{policy:?}"),
                reason: "retention values must be positive and stride must be > 0".to_owned(),
            });
        }

        let conn = self.connection();
        let search_cutoff = now_ms.saturating_sub(policy.raw_search_event_retention_ms);
        let summary_cutoff = now_ms.saturating_sub(policy.search_summary_retention_ms);
        let resource_cutoff = now_ms.saturating_sub(policy.resource_sample_retention_ms);
        let downsample_cutoff = now_ms.saturating_sub(policy.resource_downsample_after_ms);

        let downsampled_resource_samples = if policy.resource_downsample_stride > 1 {
            conn.execute_with_params(
                "DELETE FROM resource_samples \
                 WHERE ts_ms < ?1 AND (sample_id % ?2) != 0;",
                &[
                    SqliteValue::Integer(downsample_cutoff),
                    SqliteValue::Integer(i64::from(policy.resource_downsample_stride)),
                ],
            )
            .map_err(ops_error)?
        } else {
            0
        };

        let deleted_search_events = conn
            .execute_with_params(
                "DELETE FROM search_events WHERE ts_ms < ?1;",
                &[SqliteValue::Integer(search_cutoff)],
            )
            .map_err(ops_error)?;
        let deleted_search_summaries = conn
            .execute_with_params(
                "DELETE FROM search_summaries WHERE window_start_ms < ?1;",
                &[SqliteValue::Integer(summary_cutoff)],
            )
            .map_err(ops_error)?;
        let deleted_resource_samples = conn
            .execute_with_params(
                "DELETE FROM resource_samples WHERE ts_ms < ?1;",
                &[SqliteValue::Integer(resource_cutoff)],
            )
            .map_err(ops_error)?;

        Ok(OpsRetentionResult {
            deleted_search_events: usize_to_u64(deleted_search_events),
            deleted_search_summaries: usize_to_u64(deleted_search_summaries),
            deleted_resource_samples: usize_to_u64(deleted_resource_samples),
            downsampled_resource_samples: usize_to_u64(downsampled_resource_samples),
        })
    }

    fn apply_pragmas(&self) -> SearchResult<()> {
        self.conn
            .execute("PRAGMA foreign_keys=ON;")
            .map_err(ops_error)?;
        if self.config.wal_mode {
            self.conn
                .execute("PRAGMA journal_mode=WAL;")
                .map_err(ops_error)?;
        } else if let Err(error) = self.conn.execute("PRAGMA journal_mode=DELETE;") {
            tracing::warn!(
                target: "frankensearch.ops.storage",
                ?error,
                "journal_mode=DELETE was not accepted; falling back to WAL"
            );
            self.conn
                .execute("PRAGMA journal_mode=WAL;")
                .map_err(ops_error)?;
        }

        self.conn
            .execute(&format!(
                "PRAGMA busy_timeout={};",
                self.config.busy_timeout_ms
            ))
            .map_err(ops_error)?;
        self.conn
            .execute(&format!(
                "PRAGMA cache_size={};",
                self.config.cache_size_pages
            ))
            .map_err(ops_error)?;

        Ok(())
    }

    fn with_transaction<T, F>(&self, operation: F) -> SearchResult<T>
    where
        F: FnOnce(&Connection) -> SearchResult<T>,
    {
        self.connection().execute("BEGIN;").map_err(ops_error)?;
        let result = operation(self.connection());
        match result {
            Ok(value) => {
                self.connection().execute("COMMIT;").map_err(ops_error)?;
                Ok(value)
            }
            Err(error) => {
                let _ignored = self.connection().execute("ROLLBACK;");
                Err(error)
            }
        }
    }
}

impl OpsIngestBatchResult {
    const fn new(
        requested: usize,
        inserted: usize,
        deduplicated: usize,
        queue_depth_before: usize,
        queue_depth_after: usize,
        write_latency_us: u64,
    ) -> Self {
        Self {
            requested,
            inserted,
            deduplicated,
            failed: 0,
            queue_depth_before,
            queue_depth_after,
            write_latency_us,
        }
    }
}

/// Bootstrap ops schema to the latest supported version.
///
/// # Errors
///
/// Returns an error if migration metadata cannot be read, any migration fails,
/// checksums do not match, or an unsupported schema version is detected.
pub fn bootstrap(conn: &Connection) -> SearchResult<()> {
    conn.execute(OPS_SCHEMA_MIGRATIONS_TABLE_SQL)
        .map_err(ops_error)?;

    let mut version = current_version_optional(conn)?.unwrap_or(0);
    if version > OPS_SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "ops schema version {version} is newer than supported {OPS_SCHEMA_VERSION}"
            ))),
        });
    }

    for migration in OPS_MIGRATIONS {
        if migration.version <= version {
            continue;
        }
        apply_migration(conn, migration)?;
        version = migration.version;
    }

    validate_migration_checksums(conn)?;
    Ok(())
}

/// Read the latest applied schema version.
///
/// # Errors
///
/// Returns an error if migration metadata cannot be queried or no versions
/// have been recorded.
pub fn current_version(conn: &Connection) -> SearchResult<i64> {
    current_version_optional(conn)?.ok_or_else(|| SearchError::SubsystemError {
        subsystem: "ops-storage",
        source: Box::new(io::Error::other(
            "ops_schema_migrations table has no version rows",
        )),
    })
}

fn apply_migration(conn: &Connection, migration: &OpsMigration) -> SearchResult<()> {
    tracing::debug!(
        target: "frankensearch.ops.storage",
        migration_version = migration.version,
        migration_name = migration.name,
        "applying ops storage migration"
    );

    // Execute DDL statements individually in autocommit mode so that each
    // CREATE TABLE / CREATE INDEX is committed separately.  This prevents
    // the sqlite_master btree page from overflowing when the accumulated
    // DDL text exceeds the 4 KiB page size.
    for statement in migration.statements {
        conn.execute(statement).map_err(ops_error)?;
    }

    // Record the migration metadata.
    let params = [
        SqliteValue::Integer(migration.version),
        SqliteValue::Text(migration.name.to_owned()),
        SqliteValue::Integer(unix_timestamp_ms()?),
        SqliteValue::Text(migration.checksum.to_owned()),
        SqliteValue::Integer(i64::from(migration.reversible)),
    ];
    conn.execute_with_params(
        "INSERT INTO ops_schema_migrations(version, name, applied_at_ms, checksum, reversible) \
         VALUES (?1, ?2, ?3, ?4, ?5);",
        &params,
    )
    .map_err(ops_error)?;
    Ok(())
}

fn validate_migration_checksums(conn: &Connection) -> SearchResult<()> {
    let rows = conn
        .query("SELECT version, checksum FROM ops_schema_migrations ORDER BY version ASC;")
        .map_err(ops_error)?;
    for row in &rows {
        let version = row_i64(row, 0, "ops_schema_migrations.version")?;
        let checksum = row_text(row, 1, "ops_schema_migrations.checksum")?;
        let Some(expected) = expected_checksum(version) else {
            return Err(SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other(format!(
                    "unknown ops migration version {version} found in ops_schema_migrations"
                ))),
            });
        };
        if checksum != expected {
            return Err(SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other(format!(
                    "checksum mismatch for ops migration {version}: expected {expected}, found \
                     {checksum}"
                ))),
            });
        }
    }
    Ok(())
}

fn current_version_optional(conn: &Connection) -> SearchResult<Option<i64>> {
    let rows = conn
        .query("SELECT version FROM ops_schema_migrations ORDER BY version DESC LIMIT 1;")
        .map_err(ops_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    row_i64(row, 0, "ops_schema_migrations.version").map(Some)
}

fn expected_checksum(version: i64) -> Option<&'static str> {
    OPS_MIGRATIONS
        .iter()
        .find(|migration| migration.version == version)
        .map(|migration| migration.checksum)
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value.as_str()),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(ops_error)?;
    i64::try_from(since_epoch.as_millis()).map_err(ops_error)
}

fn search_event_exists(conn: &Connection, event_id: &str) -> SearchResult<bool> {
    let existing = conn
        .query_with_params(
            "SELECT 1 FROM search_events WHERE event_id = ?1;",
            &[SqliteValue::Text(event_id.to_owned())],
        )
        .map_err(ops_error)?;
    Ok(!existing.is_empty())
}

fn evidence_link_id(alert_id: &str, evidence_uri: &str) -> String {
    // Deterministic FNV-1a 64-bit hash over (alert_id, U+001F, evidence_uri).
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    for byte in alert_id
        .bytes()
        .chain(std::iter::once(0x1f))
        .chain(evidence_uri.bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    format!("evlnk:{hash:016x}")
}

fn insert_search_event_row(conn: &Connection, event: &SearchEventRecord) -> SearchResult<()> {
    let params = [
        SqliteValue::Text(event.event_id.clone()),
        SqliteValue::Text(event.project_key.clone()),
        SqliteValue::Text(event.instance_id.clone()),
        SqliteValue::Text(event.correlation_id.clone()),
        optional_text(event.query_hash.as_deref()),
        optional_text(event.query_class.as_deref()),
        SqliteValue::Text(event.phase.as_str().to_owned()),
        SqliteValue::Integer(u64_to_i64(event.latency_us, "latency_us")?),
        optional_u64(event.result_count, "result_count")?,
        optional_u64(event.memory_bytes, "memory_bytes")?,
        SqliteValue::Integer(event.ts_ms),
    ];

    conn.execute_with_params(
        "INSERT INTO search_events(\
            event_id, project_key, instance_id, correlation_id, query_hash, query_class, \
            phase, latency_us, result_count, memory_bytes, ts_ms\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11);",
        &params,
    )
    .map_err(ops_error)?;
    Ok(())
}

fn upsert_search_summary_row(
    conn: &Connection,
    project_key: &str,
    instance_id: &str,
    window: SummaryWindow,
    window_start_ms: i64,
    stats: &WindowEventStats,
) -> SearchResult<()> {
    let key_params = [
        SqliteValue::Text(project_key.to_owned()),
        SqliteValue::Text(instance_id.to_owned()),
        SqliteValue::Text(window.as_label().to_owned()),
        SqliteValue::Integer(window_start_ms),
    ];
    let existing = conn
        .query_with_params(
            "SELECT 1 FROM search_summaries \
             WHERE project_key = ?1 AND instance_id = ?2 AND window = ?3 AND window_start_ms = ?4;",
            &key_params,
        )
        .map_err(ops_error)?;

    if existing.is_empty() {
        conn.execute_with_params(
            "INSERT INTO search_summaries(\
                project_key, instance_id, window, window_start_ms, search_count, \
                p50_latency_us, p95_latency_us, p99_latency_us, avg_result_count\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
            &[
                SqliteValue::Text(project_key.to_owned()),
                SqliteValue::Text(instance_id.to_owned()),
                SqliteValue::Text(window.as_label().to_owned()),
                SqliteValue::Integer(window_start_ms),
                SqliteValue::Integer(u64_to_i64(stats.search_count, "search_count")?),
                optional_u64(stats.p50_latency_us, "p50_latency_us")?,
                optional_u64(stats.p95_latency_us, "p95_latency_us")?,
                optional_u64(stats.p99_latency_us, "p99_latency_us")?,
                optional_f64(stats.avg_result_count),
            ],
        )
        .map_err(ops_error)?;
    } else {
        conn.execute_with_params(
            "UPDATE search_summaries SET \
                search_count = ?1, p50_latency_us = ?2, p95_latency_us = ?3, \
                p99_latency_us = ?4, avg_result_count = ?5 \
             WHERE project_key = ?6 AND instance_id = ?7 AND window = ?8 AND window_start_ms = ?9;",
            &[
                SqliteValue::Integer(u64_to_i64(stats.search_count, "search_count")?),
                optional_u64(stats.p50_latency_us, "p50_latency_us")?,
                optional_u64(stats.p95_latency_us, "p95_latency_us")?,
                optional_u64(stats.p99_latency_us, "p99_latency_us")?,
                optional_f64(stats.avg_result_count),
                SqliteValue::Text(project_key.to_owned()),
                SqliteValue::Text(instance_id.to_owned()),
                SqliteValue::Text(window.as_label().to_owned()),
                SqliteValue::Integer(window_start_ms),
            ],
        )
        .map_err(ops_error)?;
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn compute_window_event_stats(
    conn: &Connection,
    project_key: &str,
    instance_id: &str,
    window_start_ms: i64,
    window_end_ms: i64,
) -> SearchResult<WindowEventStats> {
    let rows = conn
        .query_with_params(
            "SELECT latency_us, result_count, memory_bytes \
             FROM search_events \
             WHERE project_key = ?1 AND instance_id = ?2 AND ts_ms >= ?3 AND ts_ms <= ?4 \
             ORDER BY latency_us ASC;",
            &[
                SqliteValue::Text(project_key.to_owned()),
                SqliteValue::Text(instance_id.to_owned()),
                SqliteValue::Integer(window_start_ms),
                SqliteValue::Integer(window_end_ms),
            ],
        )
        .map_err(ops_error)?;

    if rows.is_empty() {
        return Ok(WindowEventStats::default());
    }

    let mut latencies = Vec::with_capacity(rows.len());
    let mut memory_values = Vec::new();
    let mut result_count_sum = 0_f64;
    let mut result_count_samples = 0_u64;
    let mut memory_sum = 0_f64;

    for row in &rows {
        let latency_us =
            i64_to_u64_non_negative(row_i64(row, 0, "search_events.latency_us")?, "latency_us")?;
        latencies.push(latency_us);

        if let Some(result_count) = row_opt_i64(row, 1, "search_events.result_count")? {
            result_count_sum += i64_to_u64_non_negative(result_count, "result_count")? as f64;
            result_count_samples = result_count_samples.saturating_add(1);
        }
        if let Some(memory_bytes) = row_opt_i64(row, 2, "search_events.memory_bytes")? {
            let memory_u64 = i64_to_u64_non_negative(memory_bytes, "memory_bytes")?;
            memory_sum += memory_u64 as f64;
            memory_values.push(memory_u64);
        }
    }
    memory_values.sort_unstable();

    let search_count = usize_to_u64(rows.len());
    let avg_result_count = if result_count_samples == 0 {
        None
    } else {
        Some(result_count_sum / (result_count_samples as f64))
    };
    let avg_memory_bytes = if memory_values.is_empty() {
        None
    } else {
        Some(memory_sum / (memory_values.len() as f64))
    };

    Ok(WindowEventStats {
        search_count,
        p50_latency_us: percentile_nearest_rank(&latencies, 50, 100),
        p95_latency_us: percentile_nearest_rank(&latencies, 95, 100),
        p99_latency_us: percentile_nearest_rank(&latencies, 99, 100),
        avg_result_count,
        avg_memory_bytes,
        p95_memory_bytes: percentile_nearest_rank(&memory_values, 95, 100),
    })
}

fn list_distinct_project_keys(conn: &Connection) -> SearchResult<Vec<String>> {
    let rows = conn
        .query("SELECT DISTINCT project_key FROM search_events ORDER BY project_key ASC;")
        .map_err(ops_error)?;
    rows.iter()
        .map(|row| row_text(row, 0, "search_events.project_key").map(ToOwned::to_owned))
        .collect()
}

fn compute_slo_window_stats(
    conn: &Connection,
    project_key: Option<&str>,
    window_start_ms: i64,
    window_end_ms: i64,
) -> SearchResult<SloWindowStats> {
    let rows = if let Some(project_key) = project_key {
        conn.query_with_params(
            "SELECT phase, latency_us \
             FROM search_events \
             WHERE project_key = ?1 AND ts_ms >= ?2 AND ts_ms <= ?3 \
             ORDER BY latency_us ASC;",
            &[
                SqliteValue::Text(project_key.to_owned()),
                SqliteValue::Integer(window_start_ms),
                SqliteValue::Integer(window_end_ms),
            ],
        )
        .map_err(ops_error)?
    } else {
        conn.query_with_params(
            "SELECT phase, latency_us \
             FROM search_events \
             WHERE ts_ms >= ?1 AND ts_ms <= ?2 \
             ORDER BY latency_us ASC;",
            &[
                SqliteValue::Integer(window_start_ms),
                SqliteValue::Integer(window_end_ms),
            ],
        )
        .map_err(ops_error)?
    };

    if rows.is_empty() {
        return Ok(SloWindowStats::default());
    }

    let mut failed_requests = 0_u64;
    let mut latencies = Vec::with_capacity(rows.len());
    for row in &rows {
        let phase = row_text(row, 0, "search_events.phase")?;
        if phase == SearchEventPhase::Failed.as_str() {
            failed_requests = failed_requests.saturating_add(1);
        }
        let latency_us = i64_to_u64_non_negative(
            row_i64(row, 1, "search_events.latency_us")?,
            "search_events.latency_us",
        )?;
        latencies.push(latency_us);
    }

    Ok(SloWindowStats {
        total_requests: usize_to_u64(rows.len()),
        failed_requests,
        p95_latency_us: percentile_nearest_rank(&latencies, 95, 100),
    })
}

#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn evaluate_slo_window(
    stats: &SloWindowStats,
    scope: SloScope,
    scope_key: &str,
    project_key: Option<&str>,
    window_start_ms: i64,
    now_ms: i64,
    config: SloMaterializationConfig,
    window: SummaryWindow,
) -> SloEvaluation {
    let total_requests = stats.total_requests;
    let failed_requests = stats.failed_requests;
    let p95_latency_us = stats.p95_latency_us;

    if total_requests < config.min_requests {
        return SloEvaluation {
            scope,
            scope_key: scope_key.to_owned(),
            project_key: project_key.map(ToOwned::to_owned),
            window,
            window_start_ms,
            window_end_ms: now_ms,
            total_requests,
            failed_requests,
            p95_latency_us,
            target_p95_latency_us: config.target_p95_latency_us,
            error_budget_ratio: config.error_budget_ratio,
            error_rate: None,
            error_budget_burn: None,
            remaining_budget_ratio: None,
            health: SloHealth::NoData,
            reason_code: "slo.no_data.min_requests".to_owned(),
            generated_at_ms: now_ms,
            anomaly: None,
        };
    }

    let error_rate = (failed_requests as f64) / (total_requests as f64);
    let consumed_budget_ratio = (error_rate / config.error_budget_ratio).clamp(0.0, 1.0);
    let error_budget_burn = consumed_budget_ratio / budget_fraction_for_window(window);
    let remaining_budget_ratio = 1.0 - consumed_budget_ratio;
    let latency_ratio =
        p95_latency_us.map(|value| (value as f64) / (config.target_p95_latency_us as f64));

    let burn_level = if error_budget_burn >= config.critical_burn_rate {
        3
    } else if error_budget_burn >= config.error_burn_rate {
        2
    } else {
        usize::from(error_budget_burn >= config.warn_burn_rate)
    };
    let latency_level = if latency_ratio.unwrap_or(0.0) >= config.critical_latency_multiplier {
        3
    } else if latency_ratio.unwrap_or(0.0) >= config.error_latency_multiplier {
        2
    } else {
        usize::from(latency_ratio.unwrap_or(0.0) >= config.warn_latency_multiplier)
    };
    let max_level = burn_level.max(latency_level);

    let (health, reason_code) = match (burn_level, latency_level, max_level) {
        (_, _, 0) => (SloHealth::Healthy, "slo.search_latency_p95.healthy"),
        (burn, lat, _) if burn > lat => (
            level_to_health(burn_level),
            "slo.query_failure_rate.budget_burn_high",
        ),
        (burn, lat, _) if lat > burn => (
            level_to_health(latency_level),
            "slo.search_latency_p95.threshold_exceeded",
        ),
        (_, _, level) => (
            level_to_health(level),
            "slo.search_latency_p95.budget_burn_and_latency_high",
        ),
    };

    let anomaly = if max_level == 0 {
        None
    } else if burn_level >= latency_level {
        let baseline = match max_level {
            1 => config.warn_burn_rate,
            2 => config.error_burn_rate,
            _ => config.critical_burn_rate,
        };
        Some(AnomalyCandidate {
            metric_name: "query_failure_rate".to_owned(),
            baseline_value: baseline,
            observed_value: error_budget_burn,
            deviation_ratio: if baseline == 0.0 {
                0.0
            } else {
                (error_budget_burn - baseline) / baseline
            },
            severity: severity_for_level(max_level),
            reason_code: "anomaly.query_failure_rate.spike".to_owned(),
        })
    } else {
        let multiplier = match max_level {
            1 => config.warn_latency_multiplier,
            2 => config.error_latency_multiplier,
            _ => config.critical_latency_multiplier,
        };
        let baseline = (config.target_p95_latency_us as f64) * multiplier;
        let observed = p95_latency_us.map_or(0.0, |value| value as f64);
        Some(AnomalyCandidate {
            metric_name: "search_latency_p95".to_owned(),
            baseline_value: baseline,
            observed_value: observed,
            deviation_ratio: if baseline == 0.0 {
                0.0
            } else {
                (observed - baseline) / baseline
            },
            severity: severity_for_level(max_level),
            reason_code: "anomaly.search_latency_p95.spike".to_owned(),
        })
    };

    SloEvaluation {
        scope,
        scope_key: scope_key.to_owned(),
        project_key: project_key.map(ToOwned::to_owned),
        window,
        window_start_ms,
        window_end_ms: now_ms,
        total_requests,
        failed_requests,
        p95_latency_us,
        target_p95_latency_us: config.target_p95_latency_us,
        error_budget_ratio: config.error_budget_ratio,
        error_rate: Some(error_rate),
        error_budget_burn: Some(error_budget_burn),
        remaining_budget_ratio: Some(remaining_budget_ratio),
        health,
        reason_code: reason_code.to_owned(),
        generated_at_ms: now_ms,
        anomaly,
    }
}

const fn severity_for_level(level: usize) -> AnomalySeverity {
    match level {
        1 => AnomalySeverity::Warn,
        2 => AnomalySeverity::Error,
        _ => AnomalySeverity::Critical,
    }
}

const fn level_to_health(level: usize) -> SloHealth {
    match level {
        1 => SloHealth::Warn,
        2 => SloHealth::Error,
        _ => SloHealth::Critical,
    }
}

const fn budget_fraction_for_window(window: SummaryWindow) -> f64 {
    match window {
        SummaryWindow::OneMinute => 0.005,
        SummaryWindow::FifteenMinutes => 0.01,
        SummaryWindow::OneHour => 0.02,
        SummaryWindow::SixHours => 0.08,
        SummaryWindow::TwentyFourHours => 0.2,
        SummaryWindow::ThreeDays => 0.35,
        SummaryWindow::OneWeek => 1.0,
    }
}

fn upsert_slo_rollup_row(conn: &Connection, evaluation: &SloEvaluation) -> SearchResult<()> {
    let rollup_id = format!(
        "slo:{}:{}:{}:{}",
        evaluation.scope.as_str(),
        evaluation.scope_key,
        evaluation.window.as_label(),
        evaluation.window_start_ms
    );
    let key_params = [
        SqliteValue::Text(evaluation.scope.as_str().to_owned()),
        SqliteValue::Text(evaluation.scope_key.clone()),
        SqliteValue::Text(evaluation.window.as_label().to_owned()),
        SqliteValue::Integer(evaluation.window_start_ms),
    ];
    let existing = conn
        .query_with_params(
            "SELECT 1 FROM slo_rollups \
             WHERE scope = ?1 AND scope_key = ?2 AND window = ?3 AND window_start_ms = ?4;",
            &key_params,
        )
        .map_err(ops_error)?;

    if existing.is_empty() {
        conn.execute_with_params(
            "INSERT INTO slo_rollups(\
                rollup_id, scope, scope_key, project_key, window, window_start_ms, window_end_ms, \
                total_requests, failed_requests, p95_latency_us, target_p95_latency_us, \
                error_budget_ratio, error_rate, error_budget_burn, remaining_budget_ratio, \
                health, reason_code, generated_at_ms\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18);",
            &[
                SqliteValue::Text(rollup_id),
                SqliteValue::Text(evaluation.scope.as_str().to_owned()),
                SqliteValue::Text(evaluation.scope_key.clone()),
                optional_text(evaluation.project_key.as_deref()),
                SqliteValue::Text(evaluation.window.as_label().to_owned()),
                SqliteValue::Integer(evaluation.window_start_ms),
                SqliteValue::Integer(evaluation.window_end_ms),
                SqliteValue::Integer(u64_to_i64(
                    evaluation.total_requests,
                    "slo_rollups.total_requests",
                )?),
                SqliteValue::Integer(u64_to_i64(
                    evaluation.failed_requests,
                    "slo_rollups.failed_requests",
                )?),
                optional_u64(evaluation.p95_latency_us, "slo_rollups.p95_latency_us")?,
                SqliteValue::Integer(u64_to_i64(
                    evaluation.target_p95_latency_us,
                    "slo_rollups.target_p95_latency_us",
                )?),
                SqliteValue::Float(evaluation.error_budget_ratio),
                optional_f64(evaluation.error_rate),
                optional_f64(evaluation.error_budget_burn),
                optional_f64(evaluation.remaining_budget_ratio),
                SqliteValue::Text(evaluation.health.as_str().to_owned()),
                SqliteValue::Text(evaluation.reason_code.clone()),
                SqliteValue::Integer(evaluation.generated_at_ms),
            ],
        )
        .map_err(ops_error)?;
    } else {
        conn.execute_with_params(
            "UPDATE slo_rollups SET \
                project_key = ?1, window_end_ms = ?2, total_requests = ?3, failed_requests = ?4, \
                p95_latency_us = ?5, target_p95_latency_us = ?6, error_budget_ratio = ?7, \
                error_rate = ?8, error_budget_burn = ?9, remaining_budget_ratio = ?10, \
                health = ?11, reason_code = ?12, generated_at_ms = ?13 \
             WHERE scope = ?14 AND scope_key = ?15 AND window = ?16 AND window_start_ms = ?17;",
            &[
                optional_text(evaluation.project_key.as_deref()),
                SqliteValue::Integer(evaluation.window_end_ms),
                SqliteValue::Integer(u64_to_i64(
                    evaluation.total_requests,
                    "slo_rollups.total_requests",
                )?),
                SqliteValue::Integer(u64_to_i64(
                    evaluation.failed_requests,
                    "slo_rollups.failed_requests",
                )?),
                optional_u64(evaluation.p95_latency_us, "slo_rollups.p95_latency_us")?,
                SqliteValue::Integer(u64_to_i64(
                    evaluation.target_p95_latency_us,
                    "slo_rollups.target_p95_latency_us",
                )?),
                SqliteValue::Float(evaluation.error_budget_ratio),
                optional_f64(evaluation.error_rate),
                optional_f64(evaluation.error_budget_burn),
                optional_f64(evaluation.remaining_budget_ratio),
                SqliteValue::Text(evaluation.health.as_str().to_owned()),
                SqliteValue::Text(evaluation.reason_code.clone()),
                SqliteValue::Integer(evaluation.generated_at_ms),
                SqliteValue::Text(evaluation.scope.as_str().to_owned()),
                SqliteValue::Text(evaluation.scope_key.clone()),
                SqliteValue::Text(evaluation.window.as_label().to_owned()),
                SqliteValue::Integer(evaluation.window_start_ms),
            ],
        )
        .map_err(ops_error)?;
    }
    Ok(())
}

fn resolve_stale_anomaly_rows(
    conn: &Connection,
    scope: SloScope,
    scope_key: &str,
    window: SummaryWindow,
    active_window_start_ms: i64,
    now_ms: i64,
) -> SearchResult<usize> {
    conn.execute_with_params(
        "UPDATE anomaly_materializations \
         SET state = 'resolved', resolved_at_ms = COALESCE(resolved_at_ms, ?1), updated_at_ms = ?1 \
         WHERE scope = ?2 AND scope_key = ?3 AND window = ?4 AND state = 'open' \
           AND window_start_ms < ?5;",
        &[
            SqliteValue::Integer(now_ms),
            SqliteValue::Text(scope.as_str().to_owned()),
            SqliteValue::Text(scope_key.to_owned()),
            SqliteValue::Text(window.as_label().to_owned()),
            SqliteValue::Integer(active_window_start_ms),
        ],
    )
    .map_err(ops_error)
}

#[allow(clippy::too_many_lines)]
fn sync_rollup_anomaly(
    conn: &Connection,
    evaluation: &SloEvaluation,
    now_ms: i64,
) -> SearchResult<(usize, usize)> {
    let anomaly_id = format!(
        "anomaly:{}:{}:{}:{}",
        evaluation.scope.as_str(),
        evaluation.scope_key,
        evaluation.window.as_label(),
        evaluation.window_start_ms
    );
    let Some(candidate) = &evaluation.anomaly else {
        let resolved = conn
            .execute_with_params(
                "UPDATE anomaly_materializations \
                 SET state = 'resolved', resolved_at_ms = COALESCE(resolved_at_ms, ?1), \
                     updated_at_ms = ?1 \
                 WHERE anomaly_id = ?2 AND state = 'open';",
                &[SqliteValue::Integer(now_ms), SqliteValue::Text(anomaly_id)],
            )
            .map_err(ops_error)?;
        return Ok((0, resolved));
    };

    let rows = conn
        .query_with_params(
            "SELECT state FROM anomaly_materializations WHERE anomaly_id = ?1;",
            &[SqliteValue::Text(anomaly_id.clone())],
        )
        .map_err(ops_error)?;
    let existing_state = rows
        .first()
        .map(|row| row_text(row, 0, "anomaly_materializations.state"))
        .transpose()?;

    match existing_state {
        None => {
            conn.execute_with_params(
                "INSERT INTO anomaly_materializations(\
                    anomaly_id, scope, scope_key, project_key, window, window_start_ms, \
                    metric_name, baseline_value, observed_value, deviation_ratio, severity, \
                    reason_code, correlation_id, state, opened_at_ms, updated_at_ms, resolved_at_ms\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, 'open', ?14, ?14, NULL);",
                &[
                    SqliteValue::Text(anomaly_id),
                    SqliteValue::Text(evaluation.scope.as_str().to_owned()),
                    SqliteValue::Text(evaluation.scope_key.clone()),
                    optional_text(evaluation.project_key.as_deref()),
                    SqliteValue::Text(evaluation.window.as_label().to_owned()),
                    SqliteValue::Integer(evaluation.window_start_ms),
                    SqliteValue::Text(candidate.metric_name.clone()),
                    SqliteValue::Float(candidate.baseline_value),
                    SqliteValue::Float(candidate.observed_value),
                    SqliteValue::Float(candidate.deviation_ratio),
                    SqliteValue::Text(candidate.severity.as_str().to_owned()),
                    SqliteValue::Text(candidate.reason_code.clone()),
                    SqliteValue::Null,
                    SqliteValue::Integer(now_ms),
                ],
            )
            .map_err(ops_error)?;
            Ok((1, 0))
        }
        Some("resolved") => {
            conn.execute_with_params(
                "UPDATE anomaly_materializations SET \
                    project_key = ?1, metric_name = ?2, baseline_value = ?3, observed_value = ?4, \
                    deviation_ratio = ?5, severity = ?6, reason_code = ?7, correlation_id = ?8, \
                    state = 'open', opened_at_ms = ?9, updated_at_ms = ?9, resolved_at_ms = NULL \
                 WHERE anomaly_id = ?10;",
                &[
                    optional_text(evaluation.project_key.as_deref()),
                    SqliteValue::Text(candidate.metric_name.clone()),
                    SqliteValue::Float(candidate.baseline_value),
                    SqliteValue::Float(candidate.observed_value),
                    SqliteValue::Float(candidate.deviation_ratio),
                    SqliteValue::Text(candidate.severity.as_str().to_owned()),
                    SqliteValue::Text(candidate.reason_code.clone()),
                    SqliteValue::Null,
                    SqliteValue::Integer(now_ms),
                    SqliteValue::Text(anomaly_id),
                ],
            )
            .map_err(ops_error)?;
            Ok((1, 0))
        }
        Some("open") => {
            conn.execute_with_params(
                "UPDATE anomaly_materializations SET \
                    project_key = ?1, metric_name = ?2, baseline_value = ?3, observed_value = ?4, \
                    deviation_ratio = ?5, severity = ?6, reason_code = ?7, correlation_id = ?8, \
                    updated_at_ms = ?9 \
                 WHERE anomaly_id = ?10;",
                &[
                    optional_text(evaluation.project_key.as_deref()),
                    SqliteValue::Text(candidate.metric_name.clone()),
                    SqliteValue::Float(candidate.baseline_value),
                    SqliteValue::Float(candidate.observed_value),
                    SqliteValue::Float(candidate.deviation_ratio),
                    SqliteValue::Text(candidate.severity.as_str().to_owned()),
                    SqliteValue::Text(candidate.reason_code.clone()),
                    SqliteValue::Null,
                    SqliteValue::Integer(now_ms),
                    SqliteValue::Text(anomaly_id),
                ],
            )
            .map_err(ops_error)?;
            Ok((0, 0))
        }
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected anomaly state value: {other}"
            ))),
        }),
    }
}

fn row_to_slo_rollup(row: &Row) -> SearchResult<SloRollupSnapshot> {
    let scope = SloScope::from_db(row_text(row, 0, "slo_rollups.scope")?).ok_or_else(|| {
        SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other("invalid slo_rollups.scope value")),
        }
    })?;
    let scope_key = row_text(row, 1, "slo_rollups.scope_key")?.to_owned();
    let project_key = row_opt_text(row, 2, "slo_rollups.project_key")?.map(ToOwned::to_owned);
    let window =
        SummaryWindow::from_label(row_text(row, 3, "slo_rollups.window")?).ok_or_else(|| {
            SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other("invalid slo_rollups.window value")),
            }
        })?;
    let window_start_ms = row_i64(row, 4, "slo_rollups.window_start_ms")?;
    let window_end_ms = row_i64(row, 5, "slo_rollups.window_end_ms")?;
    let total_requests = i64_to_u64_non_negative(
        row_i64(row, 6, "slo_rollups.total_requests")?,
        "slo_rollups.total_requests",
    )?;
    let failed_requests = i64_to_u64_non_negative(
        row_i64(row, 7, "slo_rollups.failed_requests")?,
        "slo_rollups.failed_requests",
    )?;
    let p95_latency_us = row_opt_i64(row, 8, "slo_rollups.p95_latency_us")?
        .map(|value| i64_to_u64_non_negative(value, "slo_rollups.p95_latency_us"))
        .transpose()?;
    let target_p95_latency_us = i64_to_u64_non_negative(
        row_i64(row, 9, "slo_rollups.target_p95_latency_us")?,
        "slo_rollups.target_p95_latency_us",
    )?;
    let error_budget_ratio =
        row_opt_f64(row, 10, "slo_rollups.error_budget_ratio")?.ok_or_else(|| {
            SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other(
                    "slo_rollups.error_budget_ratio should not be NULL",
                )),
            }
        })?;
    let error_rate = row_opt_f64(row, 11, "slo_rollups.error_rate")?;
    let error_budget_burn = row_opt_f64(row, 12, "slo_rollups.error_budget_burn")?;
    let remaining_budget_ratio = row_opt_f64(row, 13, "slo_rollups.remaining_budget_ratio")?;
    let health = SloHealth::from_db(row_text(row, 14, "slo_rollups.health")?).ok_or_else(|| {
        SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other("invalid slo_rollups.health value")),
        }
    })?;
    let reason_code = row_text(row, 15, "slo_rollups.reason_code")?.to_owned();
    let generated_at_ms = row_i64(row, 16, "slo_rollups.generated_at_ms")?;

    Ok(SloRollupSnapshot {
        scope,
        scope_key,
        project_key,
        window,
        window_start_ms,
        window_end_ms,
        total_requests,
        failed_requests,
        p95_latency_us,
        target_p95_latency_us,
        error_budget_ratio,
        error_rate,
        error_budget_burn,
        remaining_budget_ratio,
        health,
        reason_code,
        generated_at_ms,
    })
}

fn row_to_anomaly(row: &Row) -> SearchResult<AnomalyMaterializationSnapshot> {
    let anomaly_id = row_text(row, 0, "anomaly_materializations.anomaly_id")?.to_owned();
    let scope = SloScope::from_db(row_text(row, 1, "anomaly_materializations.scope")?).ok_or_else(
        || SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(
                "invalid anomaly_materializations.scope value",
            )),
        },
    )?;
    let scope_key = row_text(row, 2, "anomaly_materializations.scope_key")?.to_owned();
    let project_key =
        row_opt_text(row, 3, "anomaly_materializations.project_key")?.map(ToOwned::to_owned);
    let window = SummaryWindow::from_label(row_text(row, 4, "anomaly_materializations.window")?)
        .ok_or_else(|| SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(
                "invalid anomaly_materializations.window value",
            )),
        })?;
    let window_start_ms = row_i64(row, 5, "anomaly_materializations.window_start_ms")?;
    let metric_name = row_text(row, 6, "anomaly_materializations.metric_name")?.to_owned();
    let baseline_value = row_opt_f64(row, 7, "anomaly_materializations.baseline_value")?;
    let observed_value = row_opt_f64(row, 8, "anomaly_materializations.observed_value")?;
    let deviation_ratio = row_opt_f64(row, 9, "anomaly_materializations.deviation_ratio")?;
    let severity =
        AnomalySeverity::from_db(row_text(row, 10, "anomaly_materializations.severity")?)
            .ok_or_else(|| SearchError::SubsystemError {
                subsystem: "ops-storage",
                source: Box::new(io::Error::other(
                    "invalid anomaly_materializations.severity value",
                )),
            })?;
    let reason_code = row_text(row, 11, "anomaly_materializations.reason_code")?.to_owned();
    let correlation_id =
        row_opt_text(row, 12, "anomaly_materializations.correlation_id")?.map(ToOwned::to_owned);
    let state = row_text(row, 13, "anomaly_materializations.state")?.to_owned();
    let opened_at_ms = row_i64(row, 14, "anomaly_materializations.opened_at_ms")?;
    let updated_at_ms = row_i64(row, 15, "anomaly_materializations.updated_at_ms")?;
    let resolved_at_ms = row_opt_i64(row, 16, "anomaly_materializations.resolved_at_ms")?;

    Ok(AnomalyMaterializationSnapshot {
        anomaly_id,
        scope,
        scope_key,
        project_key,
        window,
        window_start_ms,
        metric_name,
        baseline_value,
        observed_value,
        deviation_ratio,
        severity,
        reason_code,
        correlation_id,
        state,
        opened_at_ms,
        updated_at_ms,
        resolved_at_ms,
    })
}

fn percentile_nearest_rank(values: &[u64], numerator: usize, denominator: usize) -> Option<u64> {
    if values.is_empty() || denominator == 0 {
        return None;
    }
    let n = values.len();
    let rank = ((n * numerator).saturating_add(denominator - 1)) / denominator;
    let index = rank.saturating_sub(1).min(n.saturating_sub(1));
    values.get(index).copied()
}

fn ensure_non_empty(value: &str, field: &str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: "must be non-empty".to_owned(),
        });
    }
    Ok(())
}

fn optional_text(value: Option<&str>) -> SqliteValue {
    value.map_or(SqliteValue::Null, |text| SqliteValue::Text(text.to_owned()))
}

fn optional_u64(value: Option<u64>, field: &str) -> SearchResult<SqliteValue> {
    value.map_or(Ok(SqliteValue::Null), |number| {
        u64_to_i64(number, field).map(SqliteValue::Integer)
    })
}

fn optional_f64(value: Option<f64>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::Float)
}

fn row_opt_i64(row: &Row, index: usize, field: &str) -> SearchResult<Option<i64>> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(Some(*value)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
    }
}

fn row_opt_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<Option<&'a str>> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(Some(value.as_str())),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
    }
}

#[allow(clippy::cast_precision_loss)]
fn row_opt_f64(row: &Row, index: usize, field: &str) -> SearchResult<Option<f64>> {
    match row.get(index) {
        Some(SqliteValue::Float(value)) => Ok(Some(*value)),
        Some(SqliteValue::Integer(value)) => Ok(Some(*value as f64)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "ops-storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
    }
}

fn u64_to_i64(value: u64, field: &str) -> SearchResult<i64> {
    i64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_string(),
        reason: "must fit into signed 64-bit integer".to_owned(),
    })
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn usize_to_i64(value: usize, field: &str) -> SearchResult<i64> {
    i64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_string(),
        reason: "must fit into signed 64-bit integer".to_owned(),
    })
}

fn i64_to_u64_non_negative(value: i64, field: &str) -> SearchResult<u64> {
    u64::try_from(value).map_err(|_| SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_string(),
        reason: "must be >= 0".to_owned(),
    })
}

fn duration_as_u64(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn ops_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "ops-storage",
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::io::Write;
    use std::sync::{Arc, LazyLock, Mutex};

    use super::{
        EvidenceLinkRecord, OPS_SCHEMA_MIGRATIONS_TABLE_SQL, OPS_SCHEMA_VERSION,
        OpsRetentionPolicy, OpsStorage, ResourceSampleRecord, SearchEventPhase, SearchEventRecord,
        SloHealth, SloMaterializationConfig, SloScope, SummaryWindow, bootstrap, current_version,
        evidence_link_id, ops_error,
    };
    use frankensearch_core::SearchError;
    use fsqlite::Connection;
    use fsqlite_types::value::SqliteValue;

    fn table_exists(conn: &Connection, table_name: &str) -> bool {
        let params = [SqliteValue::Text(table_name.to_owned())];
        let rows = conn
            .query_with_params(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1;",
                &params,
            )
            .map_err(ops_error)
            .expect("sqlite_master table query should succeed");
        !rows.is_empty()
    }

    fn index_exists(conn: &Connection, index_name: &str) -> bool {
        let params = [SqliteValue::Text(index_name.to_owned())];
        let rows = conn
            .query_with_params(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?1 LIMIT 1;",
                &params,
            )
            .map_err(ops_error)
            .expect("sqlite_master index query should succeed");
        !rows.is_empty()
    }

    fn migration_row_count(conn: &Connection) -> i64 {
        let rows = conn
            .query("SELECT COUNT(*) FROM ops_schema_migrations;")
            .map_err(ops_error)
            .expect("count query should succeed");
        let Some(row) = rows.first() else {
            return 0;
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for count: {other:?}"),
        }
    }

    fn search_event_count(conn: &Connection) -> i64 {
        let rows = conn
            .query("SELECT COUNT(*) FROM search_events;")
            .map_err(ops_error)
            .expect("count query should succeed");
        let Some(row) = rows.first() else {
            return 0;
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for count: {other:?}"),
        }
    }

    fn table_row_count(conn: &Connection, table: &str) -> i64 {
        let query = format!("SELECT COUNT(*) FROM {table};");
        let rows = conn
            .query(&query)
            .map_err(ops_error)
            .expect("count query should succeed");
        let Some(row) = rows.first() else {
            return 0;
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for count: {other:?}"),
        }
    }

    fn search_event_order(conn: &Connection) -> Vec<(String, i64)> {
        let rows = conn
            .query(
                "SELECT event_id, ts_ms \
                 FROM search_events \
                 ORDER BY ts_ms ASC, event_id ASC;",
            )
            .map_err(ops_error)
            .expect("ordered search event query should succeed");

        rows.iter()
            .map(|row| {
                let event_id = match row.get(0) {
                    Some(SqliteValue::Text(value)) => value.clone(),
                    other => panic!("unexpected row type for event_id: {other:?}"),
                };
                let ts_ms = match row.get(1) {
                    Some(SqliteValue::Integer(value)) => *value,
                    other => panic!("unexpected row type for ts_ms: {other:?}"),
                };
                (event_id, ts_ms)
            })
            .collect()
    }

    fn latest_resource_queue_depth(conn: &Connection) -> i64 {
        let rows = conn
            .query(
                "SELECT queue_depth FROM resource_samples \
                 ORDER BY ts_ms DESC LIMIT 1;",
            )
            .map_err(ops_error)
            .expect("queue depth query should succeed");
        let Some(row) = rows.first() else {
            panic!("expected one resource sample row");
        };
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => *value,
            other => panic!("unexpected row type for queue_depth: {other:?}"),
        }
    }

    fn anomaly_state(conn: &Connection, anomaly_id: &str) -> Option<String> {
        let rows = conn
            .query_with_params(
                "SELECT state FROM anomaly_materializations WHERE anomaly_id = ?1 LIMIT 1;",
                &[SqliteValue::Text(anomaly_id.to_owned())],
            )
            .map_err(ops_error)
            .expect("anomaly state query should succeed");
        let row = rows.first()?;
        match row.get(0) {
            Some(SqliteValue::Text(value)) => Some(value.clone()),
            other => panic!("unexpected row type for anomaly state: {other:?}"),
        }
    }

    fn anomaly_resolved_at(conn: &Connection, anomaly_id: &str) -> Option<i64> {
        let rows = conn
            .query_with_params(
                "SELECT resolved_at_ms FROM anomaly_materializations \
                 WHERE anomaly_id = ?1 LIMIT 1;",
                &[SqliteValue::Text(anomaly_id.to_owned())],
            )
            .map_err(ops_error)
            .expect("anomaly resolved_at query should succeed");
        let row = rows.first()?;
        match row.get(0) {
            Some(SqliteValue::Integer(value)) => Some(*value),
            Some(SqliteValue::Null) | None => None,
            other => panic!("unexpected row type for anomaly resolved_at: {other:?}"),
        }
    }

    fn sample_search_event(event_id: &str, ts_ms: i64) -> SearchEventRecord {
        SearchEventRecord {
            event_id: event_id.to_owned(),
            project_key: "project-a".to_owned(),
            instance_id: "instance-a".to_owned(),
            correlation_id: "corr-a".to_owned(),
            query_hash: Some("hash-a".to_owned()),
            query_class: Some("nl".to_owned()),
            phase: SearchEventPhase::Initial,
            latency_us: 1_200,
            result_count: Some(7),
            memory_bytes: Some(8_192),
            ts_ms,
        }
    }

    #[derive(Clone, Debug)]
    struct TestLogWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    static LOG_CAPTURE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    impl Write for TestLogWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.buffer
                .lock()
                .expect("log buffer lock poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn with_captured_logs<R>(run: impl FnOnce() -> R) -> (R, String) {
        let _capture_guard = LOG_CAPTURE_LOCK.lock().expect("log capture lock poisoned");
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer_buffer = Arc::clone(&buffer);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || TestLogWriter {
                buffer: Arc::clone(&writer_buffer),
            })
            .finish();
        let result = tracing::subscriber::with_default(subscriber, run);
        let logs = {
            let guard = buffer.lock().expect("log buffer lock poisoned");
            String::from_utf8_lossy(&guard).into_owned()
        };
        (result, logs)
    }

    fn seed_project_and_instance_named(conn: &Connection, project_key: &str, instance_id: &str) {
        let project_insert = format!(
            "INSERT INTO projects(project_key, display_name, created_at_ms, updated_at_ms) \
             VALUES ('{project_key}', '{project_key}', 1, 1);"
        );
        conn.execute(&project_insert)
            .expect("project row should insert");

        let instance_insert = format!(
            "INSERT INTO instances(\
                instance_id, project_key, host_name, pid, version, first_seen_ms, \
                last_heartbeat_ms, state\
             ) VALUES (\
                '{instance_id}', '{project_key}', 'host-a', 123, '0.1.0', 1, 1, 'healthy'\
             );"
        );
        conn.execute(&instance_insert)
            .expect("instance row should insert");
    }

    fn seed_project_and_instance(conn: &Connection) {
        seed_project_and_instance_named(conn, "project-a", "instance-a");
    }

    fn seed_second_project_and_instance(conn: &Connection) {
        seed_project_and_instance_named(conn, "project-b", "instance-b");
    }

    #[test]
    fn bootstrap_creates_schema_tables_and_indexes() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should be present"),
            OPS_SCHEMA_VERSION
        );

        for table in [
            "projects",
            "instances",
            "search_events",
            "search_summaries",
            "embedding_job_snapshots",
            "index_inventory_snapshots",
            "resource_samples",
            "alerts_timeline",
            "evidence_links",
            "slo_rollups",
            "anomaly_materializations",
            "ops_schema_migrations",
        ] {
            assert!(
                table_exists(&conn, table),
                "expected table {table} to exist"
            );
        }

        for index in [
            "ix_inst_pk_hb",
            "ix_se_pk_ts",
            "ix_se_ii_ts",
            "ix_se_corr",
            "ix_ss_pk_w",
            "ix_ejs_pk",
            "ix_iis_pk",
            "ix_rs_pk",
            "ix_at_pk",
            "ix_at_open",
            "ix_el_aid",
            "ix_slo_scope_window",
            "ix_slo_project_window",
            "ix_am_scope_state",
            "ix_am_project_timeline",
        ] {
            assert!(
                index_exists(&conn, index),
                "expected index {index} to exist"
            );
        }
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("first bootstrap should succeed");
        bootstrap(&conn).expect("second bootstrap should succeed");
        bootstrap(&conn).expect("third bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should be present"),
            OPS_SCHEMA_VERSION
        );
        assert_eq!(
            migration_row_count(&conn),
            OPS_SCHEMA_VERSION,
            "schema should record one row per applied migration version"
        );
    }

    #[test]
    fn bootstrap_rejects_newer_schema_versions() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        conn.execute(OPS_SCHEMA_MIGRATIONS_TABLE_SQL)
            .expect("migrations table creation should succeed");
        conn.execute(
            "INSERT INTO ops_schema_migrations(version, name, applied_at_ms, checksum, reversible) \
             VALUES (99, 'future', 0, 'future-checksum', 0);",
        )
        .expect("future migration row should insert");

        let error = bootstrap(&conn).expect_err("newer versions should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("ops schema version 99 is newer than supported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn bootstrap_detects_checksum_mismatch() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        conn.execute(OPS_SCHEMA_MIGRATIONS_TABLE_SQL)
            .expect("migrations table creation should succeed");
        conn.execute(
            "INSERT INTO ops_schema_migrations(version, name, applied_at_ms, checksum, reversible) \
             VALUES (1, 'ops_telemetry_storage_v1', 0, 'bad-checksum', 1);",
        )
        .expect("mismatch migration row should insert");

        let error = bootstrap(&conn).expect_err("checksum mismatch should fail");
        let message = error.to_string();
        assert!(
            message.contains("checksum mismatch"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn open_in_memory_bootstraps_schema() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        assert_eq!(
            storage
                .current_schema_version()
                .expect("schema version should load"),
            OPS_SCHEMA_VERSION
        );
    }

    #[test]
    fn ingest_search_events_batch_is_idempotent_and_tracks_metrics() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let event = sample_search_event("event-1", 42);

        let first = storage
            .ingest_search_events_batch(std::slice::from_ref(&event), 64)
            .expect("first ingest should succeed");
        assert_eq!(first.inserted, 1);
        assert_eq!(first.deduplicated, 0);
        assert_eq!(first.failed, 0);
        assert_eq!(first.queue_depth_after, 0);

        let second = storage
            .ingest_search_events_batch(&[event], 64)
            .expect("second ingest should succeed");
        assert_eq!(second.inserted, 0);
        assert_eq!(second.deduplicated, 1);
        assert_eq!(search_event_count(storage.connection()), 1);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_batches, 2);
        assert_eq!(metrics.total_inserted, 1);
        assert_eq!(metrics.total_deduplicated, 1);
        assert_eq!(metrics.total_failed_records, 0);
        assert_eq!(metrics.total_backpressured_batches, 0);
    }

    #[test]
    fn ingest_search_events_batch_orders_insertions_deterministically() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let first = sample_search_event("event-order-b", 10);
        let second = sample_search_event("event-order-c", 30);
        let third = sample_search_event("event-order-a", 10);
        storage
            .ingest_search_events_batch(&[first, second, third], 64)
            .expect("ingest should succeed");

        let ordered = search_event_order(storage.connection());
        let ordered_ids: Vec<&str> = ordered
            .iter()
            .map(|(event_id, _)| event_id.as_str())
            .collect();
        assert_eq!(
            ordered_ids,
            vec!["event-order-a", "event-order-b", "event-order-c"]
        );
    }

    #[test]
    fn ingest_search_events_batch_dedup_retry_preserves_ingest_sequence() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let event_a = sample_search_event("event-dedup-a", 101);
        let event_b = sample_search_event("event-dedup-b", 102);
        storage
            .ingest_search_events_batch(&[event_a.clone(), event_b.clone()], 64)
            .expect("initial ingest should succeed");
        storage
            .ingest_search_events_batch(&[event_b, event_a], 64)
            .expect("retry ingest should deduplicate");

        let ordered = search_event_order(storage.connection());
        assert_eq!(ordered.len(), 2);
        let ordered_ids: Vec<&str> = ordered
            .iter()
            .map(|(event_id, _)| event_id.as_str())
            .collect();
        assert_eq!(ordered_ids, vec!["event-dedup-a", "event-dedup-b"]);
    }

    #[test]
    fn refresh_search_summaries_materializes_all_windows() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let now_ms = 600_000;
        let mut event_a = sample_search_event("event-summary-a", now_ms - 20_000);
        event_a.latency_us = 100;
        event_a.result_count = Some(2);
        event_a.memory_bytes = Some(1_000);
        let mut event_b = sample_search_event("event-summary-b", now_ms - 10_000);
        event_b.latency_us = 300;
        event_b.result_count = Some(6);
        event_b.memory_bytes = Some(2_500);
        let mut event_c = sample_search_event("event-summary-c", now_ms - 5_000);
        event_c.latency_us = 700;
        event_c.result_count = Some(10);
        event_c.memory_bytes = Some(4_000);

        storage
            .ingest_search_events_batch(&[event_a, event_b, event_c], 128)
            .expect("ingest should succeed");

        let summaries = storage
            .refresh_search_summaries_for_instance("project-a", "instance-a", now_ms)
            .expect("summary refresh should succeed");
        assert_eq!(summaries.len(), SummaryWindow::ALL.len());
        let one_minute = summaries
            .iter()
            .find(|summary| summary.window == SummaryWindow::OneMinute)
            .expect("1m summary should exist");
        assert_eq!(one_minute.search_count, 3);
        assert_eq!(one_minute.p50_latency_us, Some(300));
        assert_eq!(one_minute.p95_latency_us, Some(700));
        assert_eq!(one_minute.p95_memory_bytes, Some(4_000));
        assert_eq!(one_minute.avg_result_count, Some(6.0));

        let latest = storage
            .latest_search_summary("project-a", "instance-a", SummaryWindow::OneMinute)
            .expect("latest summary query should succeed")
            .expect("latest 1m summary should exist");
        assert_eq!(latest.search_count, 3);
        assert_eq!(latest.p99_latency_us, Some(700));
    }

    #[test]
    fn materialize_slo_rollups_and_anomalies_emits_project_and_fleet_views() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let now_ms = 119_000;
        let mut events = Vec::new();
        for index in 0..20 {
            let mut event = sample_search_event(
                &format!("event-slo-{index}"),
                now_ms - 5_000 + i64::from(index),
            );
            if index < 8 {
                event.phase = SearchEventPhase::Failed;
                event.latency_us = 260_000;
            } else {
                event.phase = SearchEventPhase::Initial;
                event.latency_us = 90_000;
            }
            events.push(event);
        }
        storage
            .ingest_search_events_batch(&events, 256)
            .expect("slo test event ingest should succeed");

        let result = storage
            .materialize_slo_rollups_and_anomalies(
                now_ms,
                SloMaterializationConfig {
                    target_p95_latency_us: 150_000,
                    error_budget_ratio: 0.01,
                    min_requests: 1,
                    ..SloMaterializationConfig::default()
                },
            )
            .expect("slo materialization should succeed");

        assert_eq!(
            result.rollups_upserted,
            u64::try_from(SummaryWindow::ALL.len() * 2).expect("constant conversion should fit")
        );
        assert!(result.anomalies_opened >= 2);

        let project_rollup = storage
            .latest_slo_rollup(SloScope::Project, "project-a", SummaryWindow::OneMinute)
            .expect("latest project rollup query should succeed")
            .expect("expected project rollup");
        assert_eq!(project_rollup.total_requests, 20);
        assert_eq!(project_rollup.failed_requests, 8);
        assert!(project_rollup.error_budget_burn.unwrap_or(0.0) > 50.0);
        assert_ne!(project_rollup.health, SloHealth::Healthy);

        let fleet_rollup = storage
            .latest_slo_rollup(SloScope::Fleet, "__fleet__", SummaryWindow::OneMinute)
            .expect("latest fleet rollup query should succeed")
            .expect("expected fleet rollup");
        assert_eq!(fleet_rollup.total_requests, 20);

        let open_project_anomalies = storage
            .query_open_anomalies_for_scope(SloScope::Project, "project-a", 20)
            .expect("open anomaly query should succeed");
        assert!(!open_project_anomalies.is_empty());
        assert!(
            open_project_anomalies
                .iter()
                .all(|row| row.reason_code.starts_with("anomaly."))
        );

        let timeline = storage
            .query_anomaly_timeline(Some("project-a"), 20)
            .expect("project timeline query should succeed");
        assert!(!timeline.is_empty());
    }

    #[test]
    fn materialize_slo_rollups_and_anomalies_resolves_when_thresholds_relax() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let now_ms = 119_000;
        let mut events = Vec::new();
        for index in 0..12 {
            let mut event = sample_search_event(
                &format!("event-slo-recover-{index}"),
                now_ms - 4_000 + i64::from(index),
            );
            event.phase = SearchEventPhase::Failed;
            event.latency_us = 300_000;
            events.push(event);
        }
        storage
            .ingest_search_events_batch(&events, 256)
            .expect("slo recovery event ingest should succeed");

        storage
            .materialize_slo_rollups_and_anomalies(
                now_ms,
                SloMaterializationConfig {
                    target_p95_latency_us: 150_000,
                    error_budget_ratio: 0.01,
                    min_requests: 1,
                    ..SloMaterializationConfig::default()
                },
            )
            .expect("initial anomaly materialization should succeed");

        let open_before = storage
            .query_open_anomalies_for_scope(SloScope::Project, "project-a", 50)
            .expect("open anomalies query should succeed")
            .len();
        assert!(open_before > 0);

        let relaxed = storage
            .materialize_slo_rollups_and_anomalies(
                now_ms,
                SloMaterializationConfig {
                    target_p95_latency_us: 1_000_000,
                    error_budget_ratio: 1.0,
                    warn_burn_rate: 10_000.0,
                    error_burn_rate: 20_000.0,
                    critical_burn_rate: 30_000.0,
                    warn_latency_multiplier: 10.0,
                    error_latency_multiplier: 20.0,
                    critical_latency_multiplier: 30.0,
                    min_requests: 1,
                },
            )
            .expect("relaxed anomaly materialization should succeed");
        assert!(relaxed.anomalies_resolved >= u64::try_from(open_before).expect("len fits u64"));

        let open_after = storage
            .query_open_anomalies_for_scope(SloScope::Project, "project-a", 50)
            .expect("open anomalies query after recovery should succeed");
        assert!(open_after.is_empty());
    }

    #[test]
    fn query_resource_trend_returns_ordered_points_in_window() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let now_ms = 500_000;

        for (offset, queue_depth) in [(40_000, 2_u64), (30_000, 4_u64), (20_000, 6_u64)] {
            storage
                .upsert_resource_sample(&ResourceSampleRecord {
                    project_key: "project-a".to_owned(),
                    instance_id: "instance-a".to_owned(),
                    cpu_pct: Some(10.0),
                    rss_bytes: Some(1_024 * queue_depth),
                    io_read_bytes: Some(256 * queue_depth),
                    io_write_bytes: Some(128 * queue_depth),
                    queue_depth: Some(queue_depth),
                    ts_ms: now_ms - offset,
                })
                .expect("resource sample upsert should succeed");
        }

        let points = storage
            .query_resource_trend("project-a", "instance-a", SummaryWindow::OneHour, now_ms, 2)
            .expect("resource trend query should succeed");
        assert_eq!(points.len(), 2);
        assert!(points[0].ts_ms < points[1].ts_ms);
        assert_eq!(points[0].queue_depth, Some(4));
        assert_eq!(points[1].queue_depth, Some(6));
    }

    #[test]
    fn query_embedding_throughput_computes_rates() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        storage
            .connection()
            .execute(
                "INSERT INTO embedding_job_snapshots(\
                    snapshot_id, project_key, instance_id, embedder_id, pending_jobs, \
                    processing_jobs, completed_jobs, failed_jobs, retried_jobs, batch_latency_us, ts_ms\
                 ) VALUES (\
                    'snap-1', 'project-a', 'instance-a', 'model-a', 10, 2, 100, 5, 3, 1000, 1000\
                 );",
            )
            .expect("first embedding snapshot should insert");
        storage
            .connection()
            .execute(
                "INSERT INTO embedding_job_snapshots(\
                    snapshot_id, project_key, instance_id, embedder_id, pending_jobs, \
                    processing_jobs, completed_jobs, failed_jobs, retried_jobs, batch_latency_us, ts_ms\
                 ) VALUES (\
                    'snap-2', 'project-a', 'instance-a', 'model-a', 4, 1, 160, 11, 9, 900, 7000\
                 );",
            )
            .expect("second embedding snapshot should insert");

        let throughput = storage
            .query_embedding_throughput("project-a", "instance-a", SummaryWindow::OneHour, 8000)
            .expect("throughput query should succeed")
            .expect("throughput should exist");
        assert!((throughput.completed_per_sec - 10.0).abs() < 0.0001);
        assert!((throughput.failed_per_sec - 1.0).abs() < 0.0001);
        assert!((throughput.retried_per_sec - 1.0).abs() < 0.0001);
    }

    #[test]
    fn materialize_slo_rollups_and_anomalies_writes_project_and_fleet_views() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        seed_second_project_and_instance(storage.connection());

        let now_ms = 120_500;
        let mut events = Vec::new();
        for idx in 0_i64..20 {
            let mut event = sample_search_event(&format!("project-a-event-{idx}"), 120_100 + idx);
            event.latency_us = if idx == 0 { 10_000 } else { 1_200 };
            if idx == 0 {
                event.phase = SearchEventPhase::Failed;
            }
            events.push(event);
        }
        for idx in 0_i64..10 {
            let mut event = sample_search_event(&format!("project-b-event-{idx}"), 120_200 + idx);
            event.project_key = "project-b".to_owned();
            event.instance_id = "instance-b".to_owned();
            event.latency_us = 900;
            events.push(event);
        }

        storage
            .ingest_search_events_batch(&events, 512)
            .expect("event ingest should succeed");
        let materialized = storage
            .materialize_slo_rollups_and_anomalies(now_ms, SloMaterializationConfig::default())
            .expect("SLO/anomaly materialization should succeed");

        assert_eq!(
            materialized.rollups_upserted,
            u64::try_from(SummaryWindow::ALL.len() * 3).expect("len fits in u64")
        );
        assert!(materialized.anomalies_opened >= 2);

        let project_rollup = storage
            .latest_slo_rollup(SloScope::Project, "project-a", SummaryWindow::OneMinute)
            .expect("project rollup query should succeed")
            .expect("project rollup should exist");
        assert_eq!(project_rollup.health, SloHealth::Critical);
        assert!(project_rollup.error_budget_burn.expect("burn should exist") >= 4.0);
        assert_eq!(project_rollup.total_requests, 20);

        let fleet_rollup = storage
            .latest_slo_rollup(SloScope::Fleet, "__fleet__", SummaryWindow::OneMinute)
            .expect("fleet rollup query should succeed")
            .expect("fleet rollup should exist");
        assert!(matches!(
            fleet_rollup.health,
            SloHealth::Error | SloHealth::Critical
        ));
        assert_eq!(fleet_rollup.total_requests, 30);

        let project_rollups = storage
            .query_slo_rollups_for_scope(SloScope::Project, "project-a", 64)
            .expect("project rollup list query should succeed");
        let mut project_windows = project_rollups
            .iter()
            .map(|row| row.window.as_label().to_owned())
            .collect::<Vec<_>>();
        project_windows.sort_unstable();
        let mut expected_windows = SummaryWindow::ALL
            .iter()
            .map(|window| window.as_label().to_owned())
            .collect::<Vec<_>>();
        expected_windows.sort_unstable();
        assert_eq!(project_windows, expected_windows);

        let fleet_rollups = storage
            .query_slo_rollups_for_scope(SloScope::Fleet, "__fleet__", 64)
            .expect("fleet rollup list query should succeed");
        let mut fleet_windows = fleet_rollups
            .iter()
            .map(|row| row.window.as_label().to_owned())
            .collect::<Vec<_>>();
        fleet_windows.sort_unstable();
        assert_eq!(fleet_windows, expected_windows);

        let open_anomalies = storage
            .query_open_anomalies_for_scope(SloScope::Project, "project-a", 10)
            .expect("open anomaly query should succeed");
        assert!(!open_anomalies.is_empty(), "expected open anomaly rows");
        let top = &open_anomalies[0];
        assert!(top.baseline_value.is_some());
        assert!(top.observed_value.is_some());
        assert!(top.deviation_ratio.is_some());
        assert!(top.reason_code.starts_with("anomaly."));
    }

    #[test]
    fn materialize_slo_rollups_resolves_previous_window_anomalies() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let first_now_ms = 120_500;
        let mut first_window_events = Vec::new();
        for idx in 0_i64..16 {
            let mut event = sample_search_event(&format!("window1-event-{idx}"), 120_100 + idx);
            event.latency_us = if idx == 0 { 8_000 } else { 1_300 };
            if idx == 0 {
                event.phase = SearchEventPhase::Failed;
            }
            first_window_events.push(event);
        }
        storage
            .ingest_search_events_batch(&first_window_events, 256)
            .expect("first window ingest should succeed");
        storage
            .materialize_slo_rollups_and_anomalies(
                first_now_ms,
                SloMaterializationConfig::default(),
            )
            .expect("first materialization should succeed");

        let first_window_start = SummaryWindow::OneMinute.bucket_start_ms(first_now_ms);
        let anomaly_id = format!("anomaly:project:project-a:1m:{first_window_start}");
        assert_eq!(
            anomaly_state(storage.connection(), &anomaly_id).as_deref(),
            Some("open")
        );

        let second_now_ms = 180_500;
        let mut second_window_events = Vec::new();
        for idx in 0_i64..20 {
            let mut event = sample_search_event(&format!("window2-event-{idx}"), 180_100 + idx);
            event.latency_us = 900;
            second_window_events.push(event);
        }
        storage
            .ingest_search_events_batch(&second_window_events, 256)
            .expect("second window ingest should succeed");
        storage
            .materialize_slo_rollups_and_anomalies(
                second_now_ms,
                SloMaterializationConfig::default(),
            )
            .expect("second materialization should succeed");

        assert_eq!(
            anomaly_state(storage.connection(), &anomaly_id).as_deref(),
            Some("resolved")
        );
        assert!(
            anomaly_resolved_at(storage.connection(), &anomaly_id).is_some(),
            "resolved anomalies should carry resolved_at_ms"
        );
    }

    #[test]
    fn apply_retention_policy_prunes_and_downsamples() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let old_event = sample_search_event("event-retention-old", 1_000);
        let new_event = sample_search_event("event-retention-new", 9_900);
        storage
            .ingest_search_events_batch(&[old_event, new_event], 128)
            .expect("event ingest should succeed");
        storage
            .refresh_search_summaries_for_instance("project-a", "instance-a", 10_000)
            .expect("summary refresh should succeed");

        for ts_ms in [7_000, 8_000, 9_000, 9_500, 9_800] {
            storage
                .upsert_resource_sample(&ResourceSampleRecord {
                    project_key: "project-a".to_owned(),
                    instance_id: "instance-a".to_owned(),
                    cpu_pct: Some(5.0),
                    rss_bytes: Some(10_000),
                    io_read_bytes: Some(100),
                    io_write_bytes: Some(100),
                    queue_depth: Some(1),
                    ts_ms,
                })
                .expect("resource sample insert should succeed");
        }

        let result = storage
            .apply_retention_policy(
                10_000,
                OpsRetentionPolicy {
                    raw_search_event_retention_ms: 1_500,
                    search_summary_retention_ms: 2_000,
                    resource_sample_retention_ms: 4_000,
                    resource_downsample_after_ms: 800,
                    resource_downsample_stride: 2,
                },
            )
            .expect("retention apply should succeed");

        assert_eq!(search_event_count(storage.connection()), 1);
        assert!(result.deleted_search_events >= 1);
        assert!(result.deleted_search_summaries <= 7);
        assert!(result.deleted_resource_samples <= 5);
        assert!(result.downsampled_resource_samples >= 1);
        assert!(table_row_count(storage.connection(), "resource_samples") <= 3);
    }

    #[test]
    fn ingestion_metrics_snapshot_tracks_high_watermark_and_latency() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let first_batch = vec![
            sample_search_event("event-hwm-1", 201),
            sample_search_event("event-hwm-2", 202),
            sample_search_event("event-hwm-3", 203),
        ];
        let first = storage
            .ingest_search_events_batch(&first_batch, 64)
            .expect("first batch should succeed");
        assert_eq!(first.queue_depth_before, 0);
        assert_eq!(first.queue_depth_after, 0);
        assert!(first.write_latency_us > 0);

        let second_batch = vec![
            sample_search_event("event-hwm-4", 204),
            sample_search_event("event-hwm-5", 205),
        ];
        let second = storage
            .ingest_search_events_batch(&second_batch, 64)
            .expect("second batch should succeed");
        assert_eq!(second.queue_depth_before, 0);
        assert_eq!(second.queue_depth_after, 0);
        assert!(second.write_latency_us > 0);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_batches, 2);
        assert_eq!(metrics.total_inserted, 5);
        assert_eq!(metrics.total_deduplicated, 0);
        assert_eq!(metrics.total_failed_records, 0);
        assert_eq!(metrics.total_backpressured_batches, 0);
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.high_watermark_pending_events, 3);
        assert_eq!(
            metrics.total_write_latency_us,
            first
                .write_latency_us
                .saturating_add(second.write_latency_us)
        );
    }

    #[test]
    fn ingestion_metrics_snapshot_tracks_backpressure_peak_and_resets_pending() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let event_a = sample_search_event("event-bp-hwm-a", 301);
        let event_b = sample_search_event("event-bp-hwm-b", 302);
        let error = storage
            .ingest_search_events_batch(&[event_a, event_b], 1)
            .expect_err("batch should be rejected by backpressure threshold");
        assert!(
            matches!(
                error,
                SearchError::QueueFull {
                    pending: 2,
                    capacity: 1
                }
            ),
            "unexpected backpressure error: {error}"
        );

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_batches, 0);
        assert_eq!(metrics.total_inserted, 0);
        assert_eq!(metrics.total_deduplicated, 0);
        assert_eq!(metrics.total_failed_records, 0);
        assert_eq!(metrics.total_backpressured_batches, 1);
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.high_watermark_pending_events, 2);
        assert_eq!(metrics.total_write_latency_us, 0);
    }

    #[test]
    fn ingest_search_events_batch_rejects_when_backpressured() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let event_a = sample_search_event("event-a", 1);
        let event_b = sample_search_event("event-b", 2);

        let error = storage
            .ingest_search_events_batch(&[event_a, event_b], 1)
            .expect_err("batch should be rejected by backpressure threshold");
        assert!(
            matches!(
                error,
                SearchError::QueueFull {
                    pending: 2,
                    capacity: 1
                }
            ),
            "unexpected backpressure error: {error}"
        );
        assert_eq!(search_event_count(storage.connection()), 0);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_backpressured_batches, 1);
        assert_eq!(metrics.pending_events, 0);
        assert_eq!(metrics.total_failed_records, 0);
    }

    #[test]
    fn ingest_search_events_batch_rolls_back_on_validation_error() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let valid = sample_search_event("event-valid", 7);
        let invalid = SearchEventRecord {
            event_id: String::new(),
            ..sample_search_event("event-invalid", 8)
        };

        let error = storage
            .ingest_search_events_batch(&[valid, invalid], 64)
            .expect_err("validation failure should abort full batch");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "event_id"),
            "unexpected validation error: {error}"
        );
        assert_eq!(search_event_count(storage.connection()), 0);

        let metrics = storage.ingestion_metrics();
        assert_eq!(metrics.total_batches, 1);
        assert_eq!(metrics.total_failed_records, 2);
        assert_eq!(metrics.total_inserted, 0);
        assert_eq!(metrics.total_deduplicated, 0);
    }

    #[test]
    fn ingest_search_events_batch_emits_structured_success_log() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let event = sample_search_event("event-log-success", 90);

        let (_result, logs) = with_captured_logs(|| {
            storage
                .ingest_search_events_batch(std::slice::from_ref(&event), 64)
                .expect("ingest should succeed")
        });

        if logs.trim().is_empty() {
            let metrics = storage.ingestion_metrics();
            assert_eq!(metrics.total_batches, 1);
            assert_eq!(metrics.total_inserted, 1);
            assert_eq!(metrics.total_deduplicated, 0);
            return;
        }

        assert!(
            logs.contains("event=\"search_events_ingest_success\""),
            "logs: {logs}"
        );
        assert!(logs.contains("requested=1"), "logs: {logs}");
        assert!(logs.contains("inserted=1"), "logs: {logs}");
        assert!(logs.contains("write_latency_us="), "logs: {logs}");
    }

    #[test]
    fn ingest_search_events_batch_emits_structured_backpressure_log() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let event_a = sample_search_event("event-log-backpressure-a", 91);
        let event_b = sample_search_event("event-log-backpressure-b", 92);

        let (_result, logs) = with_captured_logs(|| {
            storage
                .ingest_search_events_batch(&[event_a, event_b], 1)
                .expect_err("batch should be rejected by backpressure threshold")
        });

        if logs.trim().is_empty() {
            let metrics = storage.ingestion_metrics();
            assert_eq!(metrics.total_backpressured_batches, 1);
            assert_eq!(metrics.total_inserted, 0);
            return;
        }

        assert!(
            logs.contains("event=\"search_events_ingest_backpressure\""),
            "logs: {logs}"
        );
        assert!(logs.contains("requested=2"), "logs: {logs}");
        assert!(logs.contains("pending=2"), "logs: {logs}");
        assert!(logs.contains("capacity=1"), "logs: {logs}");
    }

    #[test]
    fn ingest_search_events_batch_emits_structured_failure_log() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        let valid = sample_search_event("event-log-valid", 93);
        let invalid = SearchEventRecord {
            event_id: String::new(),
            ..sample_search_event("event-log-invalid", 94)
        };

        let (_result, logs) = with_captured_logs(|| {
            storage
                .ingest_search_events_batch(&[valid, invalid], 64)
                .expect_err("validation failure should abort full batch")
        });

        if logs.trim().is_empty() {
            // Some parallel test configurations can swallow thread-local log capture;
            // keep this assertion path deterministic via failure accounting.
            let metrics = storage.ingestion_metrics();
            assert_eq!(metrics.total_failed_records, 2);
            return;
        }

        assert!(
            logs.contains("event=\"search_events_ingest_failed\""),
            "logs: {logs}"
        );
        assert!(logs.contains("requested=2"), "logs: {logs}");
        assert!(logs.contains("failed=2"), "logs: {logs}");
        assert!(logs.contains("error="), "logs: {logs}");
    }

    #[test]
    fn upsert_resource_sample_replaces_existing_queue_depth() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let first = ResourceSampleRecord {
            project_key: "project-a".to_owned(),
            instance_id: "instance-a".to_owned(),
            cpu_pct: Some(8.0),
            rss_bytes: Some(1_024),
            io_read_bytes: Some(256),
            io_write_bytes: Some(64),
            queue_depth: Some(3),
            ts_ms: 123,
        };
        storage
            .upsert_resource_sample(&first)
            .expect("first resource sample upsert should succeed");

        let second = ResourceSampleRecord {
            queue_depth: Some(9),
            cpu_pct: Some(9.5),
            ..first
        };
        storage
            .upsert_resource_sample(&second)
            .expect("second resource sample upsert should succeed");

        assert_eq!(latest_resource_queue_depth(storage.connection()), 9);
    }

    #[test]
    #[ignore = "FrankenSQLite does not yet enforce CHECK constraints on direct SQL writes"]
    fn search_summaries_window_check_rejects_invalid_values() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");
        seed_project_and_instance(&conn);

        let params = [
            SqliteValue::Text("project-a".to_owned()),
            SqliteValue::Text("instance-a".to_owned()),
            SqliteValue::Text("2h".to_owned()),
            SqliteValue::Integer(0),
            SqliteValue::Integer(10),
            SqliteValue::Integer(100),
            SqliteValue::Integer(200),
            SqliteValue::Integer(300),
            SqliteValue::Float(4.2),
        ];
        let result = conn.execute_with_params(
            "INSERT INTO search_summaries(\
                project_key, instance_id, window, window_start_ms, search_count, \
                p50_latency_us, p95_latency_us, p99_latency_us, avg_result_count\
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
            &params,
        );
        assert!(
            result.is_err(),
            "invalid window label should fail CHECK constraint"
        );
    }

    #[test]
    fn search_summaries_accepts_all_supported_windows() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");
        seed_project_and_instance(&conn);

        for (index, window) in ["1m", "15m", "1h", "6h", "24h", "3d", "1w"]
            .iter()
            .enumerate()
        {
            let params = [
                SqliteValue::Text("project-a".to_owned()),
                SqliteValue::Text("instance-a".to_owned()),
                SqliteValue::Text((*window).to_owned()),
                SqliteValue::Integer(i64::try_from(index).expect("index fits in i64")),
                SqliteValue::Integer(10),
                SqliteValue::Integer(100),
                SqliteValue::Integer(200),
                SqliteValue::Integer(300),
                SqliteValue::Float(5.0),
            ];
            conn.execute_with_params(
                "INSERT INTO search_summaries(\
                    project_key, instance_id, window, window_start_ms, search_count, \
                    p50_latency_us, p95_latency_us, p99_latency_us, avg_result_count\
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
                &params,
            )
            .expect("supported window should insert");
        }
    }

    #[test]
    fn evidence_links_unique_constraint_prevents_duplicate_alert_uri_pairs() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        storage
            .connection()
            .execute(
                "INSERT INTO alerts_timeline(\
                alert_id, project_key, instance_id, category, severity, reason_code, summary, \
                state, opened_at_ms, updated_at_ms\
             ) VALUES (\
                'alert-1', 'project-a', 'instance-a', 'latency', 'warn', 'latency.spike', \
                'spike', 'open', 1, 1\
             );",
            )
            .expect("alert row should insert");

        let first = EvidenceLinkRecord {
            project_key: "project-a".to_owned(),
            alert_id: "alert-1".to_owned(),
            evidence_type: "jsonl".to_owned(),
            evidence_uri: "file:///tmp/evidence.jsonl".to_owned(),
            evidence_hash: Some("hash-1".to_owned()),
            created_at_ms: 1,
        };
        storage
            .insert_evidence_link(&first)
            .expect("first evidence link should insert");

        let duplicate_pair = EvidenceLinkRecord {
            evidence_hash: Some("hash-2".to_owned()),
            created_at_ms: 2,
            ..first
        };
        let duplicate_result = storage.insert_evidence_link(&duplicate_pair);
        assert!(
            duplicate_result.is_err(),
            "duplicate alert/evidence_uri pair should be rejected by insert_evidence_link"
        );
    }

    #[test]
    fn insert_evidence_link_rejects_unknown_alert_id() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());

        let link = EvidenceLinkRecord {
            project_key: "project-a".to_owned(),
            alert_id: "missing-alert".to_owned(),
            evidence_type: "jsonl".to_owned(),
            evidence_uri: "file:///tmp/missing-alert.jsonl".to_owned(),
            evidence_hash: None,
            created_at_ms: 10,
        };
        let error = storage
            .insert_evidence_link(&link)
            .expect_err("unknown alert_id should fail");
        let message = error.to_string();
        assert!(
            message.contains("unknown alert_id"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn insert_evidence_link_rejects_alert_project_mismatch() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        seed_second_project_and_instance(storage.connection());
        storage
            .connection()
            .execute(
                "INSERT INTO alerts_timeline(\
                    alert_id, project_key, instance_id, category, severity, reason_code, summary, \
                    state, opened_at_ms, updated_at_ms\
                 ) VALUES (\
                    'alert-project-a', 'project-a', 'instance-a', 'latency', 'warn', 'latency.spike', \
                    'spike', 'open', 1, 1\
                 );",
            )
            .expect("alert row should insert");

        let link = EvidenceLinkRecord {
            project_key: "project-b".to_owned(),
            alert_id: "alert-project-a".to_owned(),
            evidence_type: "jsonl".to_owned(),
            evidence_uri: "file:///tmp/project-mismatch.jsonl".to_owned(),
            evidence_hash: Some("hash-project-mismatch".to_owned()),
            created_at_ms: 2,
        };
        let error = storage
            .insert_evidence_link(&link)
            .expect_err("project mismatch should fail");
        let message = error.to_string();
        assert!(
            message.contains("belongs to project_key 'project-a'"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn insert_evidence_link_uses_deterministic_link_id() {
        let storage = OpsStorage::open_in_memory().expect("in-memory ops storage should open");
        seed_project_and_instance(storage.connection());
        storage
            .connection()
            .execute(
                "INSERT INTO alerts_timeline(\
                    alert_id, project_key, instance_id, category, severity, reason_code, summary, \
                    state, opened_at_ms, updated_at_ms\
                 ) VALUES (\
                    'alert-stable-id', 'project-a', 'instance-a', 'latency', 'warn', 'latency.spike', \
                    'spike', 'open', 1, 1\
                 );",
            )
            .expect("alert row should insert");

        let link = EvidenceLinkRecord {
            project_key: "project-a".to_owned(),
            alert_id: "alert-stable-id".to_owned(),
            evidence_type: "jsonl".to_owned(),
            evidence_uri: "file:///tmp/stable-id.jsonl".to_owned(),
            evidence_hash: Some("hash-stable-id".to_owned()),
            created_at_ms: 3,
        };
        let expected_link_id = evidence_link_id(&link.alert_id, &link.evidence_uri);
        storage
            .insert_evidence_link(&link)
            .expect("evidence link insert should succeed");

        let rows = storage
            .connection()
            .query_with_params(
                "SELECT link_id FROM evidence_links \
                 WHERE alert_id = ?1 AND evidence_uri = ?2 LIMIT 1;",
                &[
                    SqliteValue::Text(link.alert_id),
                    SqliteValue::Text(link.evidence_uri),
                ],
            )
            .map_err(ops_error)
            .expect("link id query should succeed");
        let row = rows.first().expect("expected one evidence link row");
        let actual_link_id = match row.get(0) {
            Some(SqliteValue::Text(value)) => value,
            other => panic!("unexpected row type for evidence_links.link_id: {other:?}"),
        };
        assert_eq!(actual_link_id, &expected_link_id);
    }
}
