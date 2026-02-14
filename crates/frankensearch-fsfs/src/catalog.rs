//! fsfs catalog/changelog schema and replay semantics.
//!
//! This module defines the persistent `FrankenSQLite` model for fsfs incremental
//! indexing:
//! - `fsfs_catalog_files`: current file identity + indexing state
//! - `fsfs_catalog_changelog`: append-only mutation stream for replay
//! - `fsfs_catalog_replay_checkpoint`: deterministic resume cursor per consumer

use std::io;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;

pub const CATALOG_SCHEMA_VERSION: i64 = 1;
const SUBSYSTEM: &str = "fsfs_catalog";

const LATEST_SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_files (\
        file_key TEXT PRIMARY KEY,\
        mount_id TEXT NOT NULL,\
        canonical_path TEXT NOT NULL,\
        device INTEGER,\
        inode INTEGER,\
        content_hash BLOB,\
        revision INTEGER NOT NULL CHECK (revision >= 0),\
        ingestion_class TEXT NOT NULL CHECK (ingestion_class IN ('full_semantic_lexical', 'lexical_only', 'metadata_only', 'skip')),\
        pipeline_status TEXT NOT NULL CHECK (pipeline_status IN ('discovered', 'queued', 'embedding', 'indexed', 'failed', 'skipped', 'tombstoned')),\
        eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),\
        first_seen_ts INTEGER NOT NULL,\
        last_seen_ts INTEGER NOT NULL,\
        updated_ts INTEGER NOT NULL,\
        deleted_ts INTEGER,\
        last_error TEXT,\
        metadata_json TEXT,\
        UNIQUE(mount_id, canonical_path)\
    );",
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_changelog (\
        change_id INTEGER PRIMARY KEY AUTOINCREMENT,\
        stream_seq INTEGER NOT NULL UNIQUE,\
        file_key TEXT NOT NULL REFERENCES fsfs_catalog_files(file_key) ON DELETE CASCADE,\
        revision INTEGER NOT NULL CHECK (revision >= 0),\
        change_kind TEXT NOT NULL CHECK (change_kind IN ('upsert', 'reclassified', 'status', 'tombstone')),\
        ingestion_class TEXT NOT NULL CHECK (ingestion_class IN ('full_semantic_lexical', 'lexical_only', 'metadata_only', 'skip')),\
        pipeline_status TEXT NOT NULL CHECK (pipeline_status IN ('discovered', 'queued', 'embedding', 'indexed', 'failed', 'skipped', 'tombstoned')),\
        content_hash BLOB,\
        event_ts INTEGER NOT NULL,\
        correlation_id TEXT NOT NULL,\
        replay_token TEXT NOT NULL UNIQUE,\
        applied_ts INTEGER,\
        UNIQUE(file_key, revision, change_kind)\
    );",
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_replay_checkpoint (\
        consumer_id TEXT PRIMARY KEY,\
        last_applied_seq INTEGER NOT NULL,\
        updated_ts INTEGER NOT NULL\
    );",
    "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_dirty_lookup ON fsfs_catalog_files(pipeline_status, ingestion_class, last_seen_ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_revisions ON fsfs_catalog_files(file_key, revision DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_cleanup ON fsfs_catalog_files(deleted_ts, pipeline_status);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_catalog_content_hash ON fsfs_catalog_files(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_changelog_replay ON fsfs_catalog_changelog(stream_seq ASC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_changelog_file_revision ON fsfs_catalog_changelog(file_key, revision DESC);",
    "CREATE INDEX IF NOT EXISTS idx_fsfs_changelog_pending_apply ON fsfs_catalog_changelog(applied_ts, stream_seq ASC);",
];

pub const INDEX_CATALOG_DIRTY_LOOKUP: &str = "idx_fsfs_catalog_dirty_lookup";
pub const INDEX_CATALOG_REVISIONS: &str = "idx_fsfs_catalog_revisions";
pub const INDEX_CATALOG_CLEANUP: &str = "idx_fsfs_catalog_cleanup";
pub const INDEX_CATALOG_CONTENT_HASH: &str = "idx_fsfs_catalog_content_hash";
pub const INDEX_CHANGELOG_REPLAY: &str = "idx_fsfs_changelog_replay";
pub const INDEX_CHANGELOG_FILE_REVISION: &str = "idx_fsfs_changelog_file_revision";
pub const INDEX_CHANGELOG_PENDING_APPLY: &str = "idx_fsfs_changelog_pending_apply";

/// Incremental workload query: pick files that still require indexing work.
pub const DIRTY_CATALOG_LOOKUP_SQL: &str = "SELECT file_key, revision, ingestion_class, pipeline_status, last_seen_ts \
    FROM fsfs_catalog_files \
    WHERE pipeline_status IN ('discovered', 'queued', 'failed') \
      AND ingestion_class != 'skip' \
    ORDER BY last_seen_ts DESC \
    LIMIT ?1;";

/// Incremental workload query: stream changelog rows after a checkpoint.
pub const CHANGELOG_REPLAY_BATCH_SQL: &str = "SELECT stream_seq, file_key, revision, change_kind, ingestion_class, pipeline_status, event_ts \
    FROM fsfs_catalog_changelog \
    WHERE stream_seq > ?1 \
    ORDER BY stream_seq ASC \
    LIMIT ?2;";

/// Incremental workload query: purge old tombstones once retention allows it.
pub const CLEANUP_TOMBSTONES_SQL: &str = "DELETE FROM fsfs_catalog_files \
    WHERE deleted_ts IS NOT NULL \
      AND deleted_ts <= ?1 \
      AND pipeline_status = 'tombstoned';";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogIngestionClass {
    FullSemanticLexical,
    LexicalOnly,
    MetadataOnly,
    Skip,
}

impl CatalogIngestionClass {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullSemanticLexical => "full_semantic_lexical",
            Self::LexicalOnly => "lexical_only",
            Self::MetadataOnly => "metadata_only",
            Self::Skip => "skip",
        }
    }
}

impl From<crate::config::IngestionClass> for CatalogIngestionClass {
    fn from(value: crate::config::IngestionClass) -> Self {
        match value {
            crate::config::IngestionClass::FullSemanticLexical => Self::FullSemanticLexical,
            crate::config::IngestionClass::LexicalOnly => Self::LexicalOnly,
            crate::config::IngestionClass::MetadataOnly => Self::MetadataOnly,
            crate::config::IngestionClass::Skip => Self::Skip,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogPipelineStatus {
    Discovered,
    Queued,
    Embedding,
    Indexed,
    Failed,
    Skipped,
    Tombstoned,
}

impl CatalogPipelineStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Queued => "queued",
            Self::Embedding => "embedding",
            Self::Indexed => "indexed",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Tombstoned => "tombstoned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogChangeKind {
    Upsert,
    Reclassified,
    Status,
    Tombstone,
}

impl CatalogChangeKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upsert => "upsert",
            Self::Reclassified => "reclassified",
            Self::Status => "status",
            Self::Tombstone => "tombstone",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDecision {
    /// Incoming row is exactly the next expected sequence and should be applied.
    ApplyNext { next_checkpoint: i64 },
    /// Incoming row was already applied or superseded.
    Duplicate { checkpoint: i64 },
    /// Replay stream has a gap and should pause for deterministic recovery.
    Gap {
        checkpoint: i64,
        expected_next: i64,
        observed: i64,
    },
}

/// Deterministic replay classifier used by consumers resuming after crash or
/// restart.
#[must_use]
pub const fn classify_replay_sequence(last_applied_seq: i64, incoming_seq: i64) -> ReplayDecision {
    if incoming_seq <= last_applied_seq {
        return ReplayDecision::Duplicate {
            checkpoint: last_applied_seq,
        };
    }

    let expected_next = last_applied_seq.saturating_add(1);
    if incoming_seq == expected_next {
        return ReplayDecision::ApplyNext {
            next_checkpoint: incoming_seq,
        };
    }

    ReplayDecision::Gap {
        checkpoint: last_applied_seq,
        expected_next,
        observed: incoming_seq,
    }
}

/// Bootstrap the fsfs catalog/changelog schema to the supported latest version.
///
/// # Errors
///
/// Returns an error if schema DDL fails, the version marker is invalid, or the
/// transaction cannot be committed.
pub fn bootstrap_catalog_schema(conn: &Connection) -> SearchResult<()> {
    conn.execute("BEGIN IMMEDIATE;").map_err(catalog_error)?;
    let result = bootstrap_catalog_schema_inner(conn);
    match result {
        Ok(()) => conn.execute("COMMIT;").map(|_| ()).map_err(catalog_error),
        Err(error) => {
            if let Err(rollback_err) = conn.execute("ROLLBACK;") {
                tracing::warn!(
                    target: "frankensearch.fsfs.catalog",
                    error = %rollback_err,
                    "rollback failed after catalog schema bootstrap error"
                );
            }
            Err(error)
        }
    }
}

fn bootstrap_catalog_schema_inner(conn: &Connection) -> SearchResult<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
    )
    .map_err(catalog_error)?;

    let mut version = current_catalog_schema_version_optional(conn)?.unwrap_or(0);
    if version == 0 {
        for statement in LATEST_SCHEMA {
            conn.execute(statement).map_err(catalog_error)?;
        }

        let params = [SqliteValue::Integer(CATALOG_SCHEMA_VERSION)];
        conn.execute_with_params(
            "INSERT OR REPLACE INTO fsfs_catalog_schema_version(version) VALUES (?1);",
            &params,
        )
        .map_err(catalog_error)?;
        version = current_catalog_schema_version(conn)?;
    }

    if version > CATALOG_SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "catalog schema version {version} is newer than supported {CATALOG_SCHEMA_VERSION}"
            ))),
        });
    }

    if version < CATALOG_SCHEMA_VERSION {
        return Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "legacy catalog schema version {version} is unsupported; rebuild catalog to schema version {CATALOG_SCHEMA_VERSION}"
            ))),
        });
    }

    let params = [SqliteValue::Integer(CATALOG_SCHEMA_VERSION)];
    conn.execute_with_params(
        "INSERT OR REPLACE INTO fsfs_catalog_schema_version(version) VALUES (?1);",
        &params,
    )
    .map_err(catalog_error)?;

    Ok(())
}

/// Return the current catalog schema version marker.
///
/// # Errors
///
/// Returns an error if the version table is missing/corrupt or cannot be
/// queried.
pub fn current_catalog_schema_version(conn: &Connection) -> SearchResult<i64> {
    current_catalog_schema_version_optional(conn)?.ok_or_else(|| SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(io::Error::other(
            "fsfs_catalog_schema_version table has no rows",
        )),
    })
}

fn current_catalog_schema_version_optional(conn: &Connection) -> SearchResult<Option<i64>> {
    let rows = conn
        .query("SELECT version FROM fsfs_catalog_schema_version ORDER BY version DESC LIMIT 1;")
        .map_err(catalog_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    row_i64(row, 0, "fsfs_catalog_schema_version.version").map(Some)
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}",
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: SUBSYSTEM,
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn catalog_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: SUBSYSTEM,
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CATALOG_SCHEMA_VERSION, CHANGELOG_REPLAY_BATCH_SQL, CLEANUP_TOMBSTONES_SQL,
        DIRTY_CATALOG_LOOKUP_SQL, INDEX_CATALOG_CLEANUP, INDEX_CATALOG_CONTENT_HASH,
        INDEX_CATALOG_DIRTY_LOOKUP, INDEX_CATALOG_REVISIONS, INDEX_CHANGELOG_FILE_REVISION,
        INDEX_CHANGELOG_PENDING_APPLY, INDEX_CHANGELOG_REPLAY, ReplayDecision,
        bootstrap_catalog_schema, catalog_error, classify_replay_sequence,
        current_catalog_schema_version,
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
            .map_err(catalog_error)
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
            .map_err(catalog_error)
            .expect("sqlite_master index query should succeed");
        !rows.is_empty()
    }

    #[test]
    fn bootstrap_creates_catalog_tables_and_indexes() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap_catalog_schema(&conn).expect("catalog bootstrap should succeed");

        assert_eq!(
            current_catalog_schema_version(&conn).expect("version should exist"),
            CATALOG_SCHEMA_VERSION
        );
        assert!(table_exists(&conn, "fsfs_catalog_files"));
        assert!(table_exists(&conn, "fsfs_catalog_changelog"));
        assert!(table_exists(&conn, "fsfs_catalog_replay_checkpoint"));

        for index in [
            INDEX_CATALOG_DIRTY_LOOKUP,
            INDEX_CATALOG_REVISIONS,
            INDEX_CATALOG_CLEANUP,
            INDEX_CATALOG_CONTENT_HASH,
            INDEX_CHANGELOG_REPLAY,
            INDEX_CHANGELOG_FILE_REVISION,
            INDEX_CHANGELOG_PENDING_APPLY,
        ] {
            assert!(index_exists(&conn, index), "missing required index {index}");
        }
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");

        bootstrap_catalog_schema(&conn).expect("first bootstrap should succeed");
        bootstrap_catalog_schema(&conn).expect("second bootstrap should succeed");
        bootstrap_catalog_schema(&conn).expect("third bootstrap should succeed");

        assert_eq!(
            current_catalog_schema_version(&conn).expect("version should exist"),
            CATALOG_SCHEMA_VERSION
        );
    }

    #[test]
    fn bootstrap_rejects_legacy_schema_versions() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("schema version table should create");
        conn.execute("INSERT INTO fsfs_catalog_schema_version(version) VALUES (-1);")
            .expect("legacy row should insert");

        let error = bootstrap_catalog_schema(&conn).expect_err("legacy schema should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("legacy catalog schema version -1 is unsupported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn bootstrap_rejects_future_schema_versions() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS fsfs_catalog_schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("schema version table should create");
        let params = [SqliteValue::Integer(CATALOG_SCHEMA_VERSION + 10)];
        conn.execute_with_params(
            "INSERT INTO fsfs_catalog_schema_version(version) VALUES (?1);",
            &params,
        )
        .expect("future version row should insert");

        let error = bootstrap_catalog_schema(&conn).expect_err("future schema should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("newer than supported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn replay_classifier_is_deterministic_for_next_duplicate_and_gap() {
        assert_eq!(
            classify_replay_sequence(41, 42),
            ReplayDecision::ApplyNext {
                next_checkpoint: 42
            }
        );
        assert_eq!(
            classify_replay_sequence(41, 41),
            ReplayDecision::Duplicate { checkpoint: 41 }
        );
        assert_eq!(
            classify_replay_sequence(41, 45),
            ReplayDecision::Gap {
                checkpoint: 41,
                expected_next: 42,
                observed: 45
            }
        );
    }

    #[test]
    fn incremental_workload_queries_execute_and_have_index_support() {
        let conn = Connection::open(":memory:".to_owned()).expect("in-memory connection");
        bootstrap_catalog_schema(&conn).expect("catalog bootstrap should succeed");

        for index in [
            INDEX_CATALOG_DIRTY_LOOKUP,
            INDEX_CATALOG_REVISIONS,
            INDEX_CATALOG_CLEANUP,
            INDEX_CATALOG_CONTENT_HASH,
        ] {
            assert!(
                index_exists(&conn, index),
                "catalog index {index} should exist in sqlite_master"
            );
        }

        for index in [
            INDEX_CHANGELOG_REPLAY,
            INDEX_CHANGELOG_FILE_REVISION,
            INDEX_CHANGELOG_PENDING_APPLY,
        ] {
            assert!(
                index_exists(&conn, index),
                "changelog index {index} should exist in sqlite_master"
            );
        }

        let now = 1_710_000_000_000_i64;
        let file_params = [
            SqliteValue::Text("home:/tmp/a.txt".to_owned()),
            SqliteValue::Text("home".to_owned()),
            SqliteValue::Text("/tmp/a.txt".to_owned()),
            SqliteValue::Blob(vec![7_u8; 32]),
            SqliteValue::Integer(3),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("queued".to_owned()),
            SqliteValue::Integer(1),
            SqliteValue::Integer(now - 1_000),
            SqliteValue::Integer(now),
            SqliteValue::Integer(now),
        ];
        conn.execute_with_params(
            "INSERT INTO fsfs_catalog_files \
             (file_key, mount_id, canonical_path, content_hash, revision, ingestion_class, pipeline_status, eligible, first_seen_ts, last_seen_ts, updated_ts) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11);",
            &file_params,
        )
        .expect("catalog seed row should insert");

        let changelog_params = [
            SqliteValue::Integer(1),
            SqliteValue::Text("home:/tmp/a.txt".to_owned()),
            SqliteValue::Integer(3),
            SqliteValue::Text("upsert".to_owned()),
            SqliteValue::Text("full_semantic_lexical".to_owned()),
            SqliteValue::Text("queued".to_owned()),
            SqliteValue::Blob(vec![7_u8; 32]),
            SqliteValue::Integer(now),
            SqliteValue::Text("corr-1".to_owned()),
            SqliteValue::Text("token-1".to_owned()),
        ];
        conn.execute_with_params(
            "INSERT INTO fsfs_catalog_changelog \
             (stream_seq, file_key, revision, change_kind, ingestion_class, pipeline_status, content_hash, event_ts, correlation_id, replay_token) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);",
            &changelog_params,
        )
        .expect("changelog seed row should insert");

        let dirty_rows = conn
            .query_with_params(DIRTY_CATALOG_LOOKUP_SQL, &[SqliteValue::Integer(50)])
            .expect("dirty catalog lookup should execute");
        assert_eq!(dirty_rows.len(), 1);

        let replay_rows = conn
            .query_with_params(
                CHANGELOG_REPLAY_BATCH_SQL,
                &[SqliteValue::Integer(0), SqliteValue::Integer(100)],
            )
            .expect("replay batch query should execute");
        assert_eq!(replay_rows.len(), 1);

        conn.execute_with_params(
            "UPDATE fsfs_catalog_files SET pipeline_status = 'tombstoned', deleted_ts = ?1 WHERE file_key = ?2;",
            &[SqliteValue::Integer(now), SqliteValue::Text("home:/tmp/a.txt".to_owned())],
        )
        .expect("tombstone update should succeed");
        conn.execute_with_params(CLEANUP_TOMBSTONES_SQL, &[SqliteValue::Integer(now)])
            .expect("cleanup query should execute");

        let remaining = conn
            .query("SELECT file_key FROM fsfs_catalog_files;")
            .expect("remaining rows query should execute");
        assert!(remaining.is_empty(), "tombstone cleanup should remove row");
    }
}
