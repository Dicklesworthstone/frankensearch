//! FrankenSQLite-backed storage bootstrap for ops telemetry.
//!
//! This module provides the schema contract for the control-plane database
//! (`frankensearch-ops.db`) and a small connection wrapper that applies
//! pragmas, runs migrations, and validates migration checksums.

use std::io;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

/// Current schema version for the ops telemetry database.
pub const OPS_SCHEMA_VERSION: i64 = 1;

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
    r#"
CREATE INDEX IF NOT EXISTS idx_instances_project_heartbeat
    ON instances(project_key, last_heartbeat_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_search_events_project_time
    ON search_events(project_key, ts_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_search_events_instance_time
    ON search_events(instance_id, ts_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_search_events_corr
    ON search_events(project_key, correlation_id);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_search_summaries_project_window_start
    ON search_summaries(project_key, window, window_start_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_embedding_snapshots_project_time
    ON embedding_job_snapshots(project_key, ts_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_index_inventory_project_time
    ON index_inventory_snapshots(project_key, ts_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_resource_samples_project_time
    ON resource_samples(project_key, ts_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_alerts_project_time
    ON alerts_timeline(project_key, opened_at_ms DESC);
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_alerts_open
    ON alerts_timeline(project_key, state, severity, updated_at_ms DESC)
    WHERE state != 'resolved';
"#,
    r#"
CREATE INDEX IF NOT EXISTS idx_evidence_alert
    ON evidence_links(alert_id, created_at_ms DESC);
"#,
];

struct OpsMigration {
    version: i64,
    name: &'static str,
    checksum: &'static str,
    reversible: bool,
    statements: &'static [&'static str],
}

const OPS_MIGRATIONS: &[OpsMigration] = &[OpsMigration {
    version: 1,
    name: OPS_SCHEMA_V1_NAME,
    checksum: OPS_SCHEMA_V1_CHECKSUM,
    reversible: true,
    statements: OPS_SCHEMA_V1_STATEMENTS,
}];

/// Configuration for the ops telemetry storage connection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpsStorageConfig {
    /// Path to the ops telemetry database.
    pub db_path: PathBuf,
    /// Enable `WAL` journaling mode when true.
    pub wal_mode: bool,
    /// SQLite busy timeout in milliseconds.
    pub busy_timeout_ms: u64,
    /// SQLite cache size in pages.
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

/// Connection wrapper for ops telemetry storage.
pub struct OpsStorage {
    conn: Connection,
    config: OpsStorageConfig,
}

impl std::fmt::Debug for OpsStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpsStorage")
            .field("path", &self.config.db_path)
            .field("wal_mode", &self.config.wal_mode)
            .field("busy_timeout_ms", &self.config.busy_timeout_ms)
            .field("cache_size_pages", &self.config.cache_size_pages)
            .finish_non_exhaustive()
    }
}

impl OpsStorage {
    /// Open storage and bootstrap schema if needed.
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
        let storage = Self { conn, config };
        storage.apply_pragmas()?;
        bootstrap(storage.connection())?;
        Ok(storage)
    }

    /// Open in-memory storage and bootstrap schema.
    pub fn open_in_memory() -> SearchResult<Self> {
        Self::open(OpsStorageConfig::in_memory())
    }

    /// Underlying database connection.
    #[must_use]
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Runtime configuration used by this storage handle.
    #[must_use]
    pub fn config(&self) -> &OpsStorageConfig {
        &self.config
    }

    /// Current schema version.
    pub fn current_schema_version(&self) -> SearchResult<i64> {
        current_version(self.connection())
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
}

/// Bootstrap ops schema to the latest supported version.
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

    conn.execute("BEGIN;").map_err(ops_error)?;

    let apply_result = (|| -> SearchResult<()> {
        for statement in migration.statements {
            conn.execute(statement).map_err(ops_error)?;
        }

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
    })();

    match apply_result {
        Ok(()) => conn.execute("COMMIT;").map(|_| ()).map_err(ops_error),
        Err(error) => {
            let _ = conn.execute("ROLLBACK;");
            Err(error)
        }
    }
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
                "unexpected type for {field}: {:?}",
                other
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
                "unexpected type for {field}: {:?}",
                other
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
    use super::{
        OPS_SCHEMA_MIGRATIONS_TABLE_SQL, OPS_SCHEMA_VERSION, OpsStorage, bootstrap,
        current_version, ops_error,
    };
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
            other => panic!("unexpected row type for count: {:?}", other),
        }
    }

    #[test]
    fn bootstrap_creates_v1_schema_tables_and_indexes() {
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
            "ops_schema_migrations",
        ] {
            assert!(
                table_exists(&conn, table),
                "expected table {table} to exist"
            );
        }

        for index in [
            "idx_instances_project_heartbeat",
            "idx_search_events_project_time",
            "idx_search_events_instance_time",
            "idx_search_events_corr",
            "idx_search_summaries_project_window_start",
            "idx_embedding_snapshots_project_time",
            "idx_index_inventory_project_time",
            "idx_resource_samples_project_time",
            "idx_alerts_project_time",
            "idx_alerts_open",
            "idx_evidence_alert",
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
            1,
            "schema should record a single applied migration"
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
}
