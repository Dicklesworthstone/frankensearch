use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{AsyncConnection, FrankenError, Row};
use fsqlite_types::value::SqliteValue;

use crate::connection::{map_storage_error_at, retry_transient_storage};

pub const SCHEMA_VERSION: i64 = 7;

/// Governed marker table for the FTS5 Porter tokenizer rebuild.
///
/// This belongs to the storage schema rather than to the rebuild operation so
/// that a failed or cancelled rebuild cannot leave behind ad-hoc schema.
#[cfg(feature = "fts5")]
pub(crate) const PORTER_FTS5_REBUILD_TABLE: &str = "frankensearch_fts5_rebuild_version";

struct Migration {
    version: i64,
    statements: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SchemaVersionState {
    MissingTable,
    Empty,
    Present(i64),
}

/// Canonical latest schema for brand-new databases.
///
/// We intentionally bootstrap directly to latest instead of replaying historical
/// migrations (which include create/drop churn from early prototypes).
const LATEST_SCHEMA: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS documents (\
        doc_id TEXT PRIMARY KEY,\
        source_path TEXT,\
        content_preview TEXT NOT NULL,\
        content_hash BLOB NOT NULL,\
        content_length INTEGER NOT NULL,\
        created_at INTEGER NOT NULL,\
        updated_at INTEGER NOT NULL,\
        metadata_json TEXT\
    );",
    "CREATE TABLE IF NOT EXISTS embedding_jobs (\
        job_id INTEGER PRIMARY KEY AUTOINCREMENT,\
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
        embedder_id TEXT NOT NULL,\
        priority INTEGER NOT NULL DEFAULT 0,\
        submitted_at INTEGER NOT NULL,\
        started_at INTEGER,\
        completed_at INTEGER,\
        status TEXT NOT NULL DEFAULT 'pending',\
        retry_count INTEGER NOT NULL DEFAULT 0,\
        max_retries INTEGER NOT NULL DEFAULT 3,\
        error_message TEXT,\
        content_hash BLOB,\
        worker_id TEXT,\
        UNIQUE(doc_id, embedder_id, status)\
    );",
    "CREATE TABLE IF NOT EXISTS embedding_status (\
        doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
        embedder_id TEXT NOT NULL,\
        embedder_revision TEXT,\
        status TEXT NOT NULL DEFAULT 'pending',\
        embedded_at INTEGER,\
        error_message TEXT,\
        retry_count INTEGER NOT NULL DEFAULT 0,\
        PRIMARY KEY(doc_id, embedder_id)\
    );",
    "CREATE TABLE IF NOT EXISTS content_hashes (\
        content_hash TEXT PRIMARY KEY,\
        first_doc_id TEXT NOT NULL,\
        seen_count INTEGER NOT NULL DEFAULT 1,\
        first_seen_at INTEGER NOT NULL,\
        last_seen_at INTEGER NOT NULL\
    );",
    "CREATE TABLE IF NOT EXISTS index_metadata (\
        index_name TEXT PRIMARY KEY,\
        index_type TEXT NOT NULL,\
        embedder_id TEXT NOT NULL,\
        embedder_revision TEXT,\
        dimension INTEGER NOT NULL,\
        record_count INTEGER NOT NULL DEFAULT 0,\
        file_path TEXT,\
        file_size_bytes INTEGER,\
        file_hash TEXT,\
        schema_version INTEGER,\
        built_at INTEGER,\
        build_duration_ms INTEGER,\
        source_doc_count INTEGER NOT NULL DEFAULT 0,\
        config_json TEXT,\
        fec_path TEXT,\
        fec_size_bytes INTEGER,\
        last_verified_at INTEGER,\
        last_repair_at INTEGER,\
        repair_count INTEGER NOT NULL DEFAULT 0,\
        mean_norm REAL,\
        variance REAL\
    );",
    "CREATE TABLE IF NOT EXISTS index_build_history (\
        build_id INTEGER PRIMARY KEY AUTOINCREMENT,\
        index_name TEXT NOT NULL,\
        built_at INTEGER NOT NULL,\
        build_duration_ms INTEGER NOT NULL,\
        record_count INTEGER NOT NULL,\
        source_doc_count INTEGER NOT NULL,\
        \"trigger\" TEXT NOT NULL,\
        config_json TEXT,\
        notes TEXT,\
        mean_norm REAL,\
        variance REAL\
    );",
    "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
    "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_embedding_status_pending ON embedding_status(status, doc_id) WHERE status = 'pending';",
    "CREATE INDEX IF NOT EXISTS idx_jobs_pending ON embedding_jobs(status, priority DESC, submitted_at ASC) WHERE status = 'pending';",
    "CREATE INDEX IF NOT EXISTS idx_jobs_processing ON embedding_jobs(status, started_at) WHERE status = 'processing';",
    "CREATE INDEX IF NOT EXISTS idx_build_history_index ON index_build_history(index_name, built_at DESC);",
    "CREATE TABLE IF NOT EXISTS search_history (\
        id INTEGER PRIMARY KEY AUTOINCREMENT,\
        query TEXT NOT NULL,\
        query_class TEXT,\
        result_count INTEGER,\
        phase1_latency_ms INTEGER,\
        phase2_latency_ms INTEGER,\
        top_results_json TEXT,\
        searched_at INTEGER NOT NULL\
    );",
    "CREATE INDEX IF NOT EXISTS idx_history_query ON search_history(query);",
    "CREATE INDEX IF NOT EXISTS idx_history_ts ON search_history(searched_at DESC);",
    "CREATE TABLE IF NOT EXISTS bookmarks (\
        id INTEGER PRIMARY KEY AUTOINCREMENT,\
        doc_id TEXT NOT NULL,\
        query TEXT,\
        note TEXT,\
        created_at INTEGER NOT NULL\
    );",
    "CREATE INDEX IF NOT EXISTS idx_bookmarks_doc_query ON bookmarks(doc_id, query);",
    "CREATE TABLE IF NOT EXISTS frankensearch_fts5_rebuild_version (\
        table_name TEXT PRIMARY KEY,\
        rebuild_version INTEGER NOT NULL\
    );",
];

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        statements: &[
            "CREATE TABLE IF NOT EXISTS documents (\
                id TEXT PRIMARY KEY,\
                title TEXT,\
                content TEXT NOT NULL,\
                created_at INTEGER NOT NULL,\
                doc_type TEXT,\
                source TEXT,\
                metadata_json TEXT,\
                content_hash TEXT NOT NULL,\
                updated_at INTEGER NOT NULL\
            );",
            "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
            "CREATE TABLE IF NOT EXISTS embedding_jobs (\
                doc_id TEXT PRIMARY KEY,\
                status TEXT NOT NULL,\
                attempts INTEGER NOT NULL DEFAULT 0,\
                queued_at INTEGER NOT NULL,\
                started_at INTEGER,\
                finished_at INTEGER,\
                last_error TEXT,\
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE\
            );",
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_queued ON embedding_jobs(status, queued_at);",
            "CREATE TABLE IF NOT EXISTS content_hashes (\
                content_hash TEXT PRIMARY KEY,\
                first_doc_id TEXT NOT NULL,\
                seen_count INTEGER NOT NULL DEFAULT 1,\
                first_seen_at INTEGER NOT NULL,\
                last_seen_at INTEGER NOT NULL\
            );",
        ],
    },
    Migration {
        version: 2,
        statements: &[
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_started ON embedding_jobs(status, started_at);",
        ],
    },
    Migration {
        version: 3,
        statements: &[
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_queued;",
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_started;",
            "DROP TABLE IF EXISTS embedding_jobs;",
            "DROP INDEX IF EXISTS idx_documents_updated_at;",
            "DROP INDEX IF EXISTS idx_documents_content_hash;",
            "DROP TABLE IF EXISTS documents;",
            "CREATE TABLE IF NOT EXISTS documents (\
                doc_id TEXT PRIMARY KEY,\
                source_path TEXT,\
                content_preview TEXT NOT NULL,\
                content_hash BLOB NOT NULL,\
                content_length INTEGER NOT NULL,\
                created_at INTEGER NOT NULL,\
                updated_at INTEGER NOT NULL,\
                metadata_json TEXT\
            );",
            "CREATE TABLE IF NOT EXISTS embedding_jobs (\
                doc_id TEXT PRIMARY KEY,\
                status TEXT NOT NULL,\
                attempts INTEGER NOT NULL DEFAULT 0,\
                queued_at INTEGER NOT NULL,\
                started_at INTEGER,\
                finished_at INTEGER,\
                last_error TEXT,\
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE\
            );",
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_queued ON embedding_jobs(status, queued_at);",
            "CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_started ON embedding_jobs(status, started_at);",
            "CREATE TABLE IF NOT EXISTS embedding_status (\
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
                embedder_id TEXT NOT NULL,\
                embedder_revision TEXT,\
                status TEXT NOT NULL DEFAULT 'pending',\
                embedded_at INTEGER,\
                error_message TEXT,\
                retry_count INTEGER NOT NULL DEFAULT 0,\
                PRIMARY KEY(doc_id, embedder_id)\
            );",
            "CREATE INDEX IF NOT EXISTS idx_embedding_status_pending ON embedding_status(status, doc_id) WHERE status = 'pending';",
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);",
            "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at DESC);",
            "CREATE TABLE IF NOT EXISTS content_hashes (\
                content_hash TEXT PRIMARY KEY,\
                first_doc_id TEXT NOT NULL,\
                seen_count INTEGER NOT NULL DEFAULT 1,\
                first_seen_at INTEGER NOT NULL,\
                last_seen_at INTEGER NOT NULL\
            );",
        ],
    },
    Migration {
        version: 4,
        statements: &[
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_queued;",
            "DROP INDEX IF EXISTS idx_embedding_jobs_status_started;",
            "DROP TABLE IF EXISTS embedding_jobs;",
            "CREATE TABLE IF NOT EXISTS embedding_jobs (\
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,\
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
                embedder_id TEXT NOT NULL,\
                priority INTEGER NOT NULL DEFAULT 0,\
                submitted_at INTEGER NOT NULL,\
                started_at INTEGER,\
                completed_at INTEGER,\
                status TEXT NOT NULL DEFAULT 'pending',\
                retry_count INTEGER NOT NULL DEFAULT 0,\
                max_retries INTEGER NOT NULL DEFAULT 3,\
                error_message TEXT,\
                content_hash BLOB,\
                worker_id TEXT,\
                UNIQUE(doc_id, embedder_id, status)\
            );",
            "CREATE INDEX IF NOT EXISTS idx_jobs_pending ON embedding_jobs(status, priority DESC, submitted_at ASC) WHERE status = 'pending';",
            "CREATE INDEX IF NOT EXISTS idx_jobs_processing ON embedding_jobs(status, started_at) WHERE status = 'processing';",
        ],
    },
    Migration {
        version: 5,
        statements: &[
            "CREATE TABLE IF NOT EXISTS index_metadata (\
                index_name TEXT PRIMARY KEY,\
                index_type TEXT NOT NULL,\
                embedder_id TEXT NOT NULL,\
                embedder_revision TEXT,\
                dimension INTEGER NOT NULL,\
                record_count INTEGER NOT NULL DEFAULT 0,\
                file_path TEXT,\
                file_size_bytes INTEGER,\
                file_hash TEXT,\
                schema_version INTEGER,\
                built_at INTEGER,\
                build_duration_ms INTEGER,\
                source_doc_count INTEGER NOT NULL DEFAULT 0,\
                config_json TEXT,\
                fec_path TEXT,\
                fec_size_bytes INTEGER,\
                last_verified_at INTEGER,\
                last_repair_at INTEGER,\
                repair_count INTEGER NOT NULL DEFAULT 0,\
                mean_norm REAL,\
                variance REAL\
            );",
            "CREATE TABLE IF NOT EXISTS index_build_history (\
                build_id INTEGER PRIMARY KEY AUTOINCREMENT,\
                index_name TEXT NOT NULL,\
                built_at INTEGER NOT NULL,\
                build_duration_ms INTEGER NOT NULL,\
                record_count INTEGER NOT NULL,\
                source_doc_count INTEGER NOT NULL,\
                \"trigger\" TEXT NOT NULL,\
                config_json TEXT,\
                notes TEXT,\
                mean_norm REAL,\
                variance REAL\
            );",
            "CREATE INDEX IF NOT EXISTS idx_build_history_index ON index_build_history(index_name, built_at DESC);",
        ],
    },
    Migration {
        version: 6,
        statements: &[
            "CREATE TABLE IF NOT EXISTS search_history (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                query TEXT NOT NULL,\
                query_class TEXT,\
                result_count INTEGER,\
                phase1_latency_ms INTEGER,\
                phase2_latency_ms INTEGER,\
                top_results_json TEXT,\
                searched_at INTEGER NOT NULL\
            );",
            "CREATE INDEX IF NOT EXISTS idx_history_query ON search_history(query);",
            "CREATE INDEX IF NOT EXISTS idx_history_ts ON search_history(searched_at DESC);",
            "CREATE TABLE IF NOT EXISTS bookmarks (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                doc_id TEXT NOT NULL,\
                query TEXT,\
                note TEXT,\
                created_at INTEGER NOT NULL\
            );",
            "CREATE INDEX IF NOT EXISTS idx_bookmarks_doc_query ON bookmarks(doc_id, query);",
        ],
    },
    Migration {
        version: 7,
        statements: &[
            "CREATE TABLE IF NOT EXISTS frankensearch_fts5_rebuild_version (\
                table_name TEXT PRIMARY KEY,\
                rebuild_version INTEGER NOT NULL\
            );",
        ],
    },
];

pub fn bootstrap(conn: &AsyncConnection) -> SearchResult<()> {
    match schema_version_state(conn, "schema preflight")? {
        SchemaVersionState::Present(version) => {
            reject_future_version(version, "schema preflight")?;
            if version == SCHEMA_VERSION {
                tracing::trace!(
                    target: "frankensearch.storage",
                    schema_version = version,
                    "storage schema is current; skipping write transaction"
                );
                return Ok(());
            }
        }
        SchemaVersionState::MissingTable | SchemaVersionState::Empty => {}
    }

    retry_transient_storage(|| bootstrap_write_transaction(conn), "schema bootstrap")
}

fn bootstrap_write_transaction(conn: &AsyncConnection) -> SearchResult<()> {
    conn.execute_sync("BEGIN IMMEDIATE;")
        .map_err(|error| map_storage_error_at("schema transaction begin", error))?;
    let result = catch_unwind(AssertUnwindSafe(|| bootstrap_inner(conn)));
    match result {
        Ok(Ok(())) => conn.commit_transaction_sync().map_err(|commit_err| {
            match conn.rollback_transaction_sync() {
                Ok(()) => map_storage_error_at("schema transaction commit", commit_err),
                Err(rollback_err) => {
                    tracing::warn!(
                        target: "frankensearch.storage",
                        stage = "schema transaction rollback after commit",
                        error = %rollback_err,
                        "rollback failed after schema bootstrap commit error"
                    );
                    crate::connection::unretryable_rollback_error(
                        &map_storage_error_at("schema transaction commit", commit_err),
                        &rollback_err,
                    )
                }
            }
        }),
        Ok(Err(error)) => match conn.rollback_transaction_sync() {
            Ok(()) => Err(error),
            Err(rollback_err) => {
                tracing::warn!(
                    target: "frankensearch.storage",
                    stage = "schema transaction rollback after bootstrap",
                    error = %rollback_err,
                    "rollback failed after schema bootstrap error"
                );
                Err(crate::connection::unretryable_rollback_error(
                    &error,
                    &rollback_err,
                ))
            }
        },
        Err(payload) => {
            if let Err(rollback_err) = conn.rollback_transaction_sync() {
                tracing::error!(
                    target: "frankensearch.storage",
                    stage = "schema transaction rollback after panic",
                    error = %rollback_err,
                    "critical: rollback failed during schema bootstrap panic recovery"
                );
            }
            resume_unwind(payload);
        }
    }
}

fn bootstrap_inner(conn: &AsyncConnection) -> SearchResult<()> {
    conn.execute_sync("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);")
        .map_err(|error| map_storage_error_at("schema transaction marker table", error))?;

    let mut version = current_version_optional_at(conn, "schema transaction recheck")?.unwrap_or(0);
    reject_future_version(version, "schema transaction recheck")?;

    if version == 0 {
        tracing::debug!(
            target: "frankensearch.storage",
            to_version = SCHEMA_VERSION,
            "bootstrapping fresh storage database directly to latest schema"
        );

        for statement in LATEST_SCHEMA {
            conn.execute_sync(statement)
                .map_err(|error| map_storage_error_at("fresh schema application", error))?;
        }

        let params = [SqliteValue::Integer(SCHEMA_VERSION)];
        conn.execute_with_params_sync(
            "INSERT OR REPLACE INTO schema_version(version) VALUES (?1);",
            &params,
        )
        .map_err(|error| map_storage_error_at("fresh schema marker write", error))?;
        version = SCHEMA_VERSION;
    }

    while version < SCHEMA_VERSION {
        let next_version = version.saturating_add(1);
        let Some(migration) = MIGRATIONS
            .iter()
            .find(|migration| migration.version == next_version)
        else {
            return Err(schema_contract_error(
                "schema migration selection",
                format!("missing migration path from schema version {version} to {next_version}"),
            ));
        };

        tracing::debug!(
            target: "frankensearch.storage",
            from_version = version,
            to_version = migration.version,
            "applying storage schema migration"
        );

        for statement in migration.statements {
            conn.execute_sync(statement)
                .map_err(|error| map_storage_error_at("schema migration application", error))?;
        }

        let params = [SqliteValue::Integer(migration.version)];
        conn.execute_with_params_sync(
            "INSERT OR REPLACE INTO schema_version(version) VALUES (?1);",
            &params,
        )
        .map_err(|error| map_storage_error_at("schema migration marker write", error))?;
        version = migration.version;
    }

    tracing::debug!(
        target: "frankensearch.storage",
        schema_version = version,
        "storage schema bootstrap complete"
    );

    Ok(())
}

pub fn current_version(conn: &AsyncConnection) -> SearchResult<i64> {
    current_version_at(conn, "schema version read")
}

pub(crate) fn current_version_at(conn: &AsyncConnection, stage: &'static str) -> SearchResult<i64> {
    retry_transient_storage(
        || {
            current_version_optional_at(conn, stage)?
                .ok_or_else(|| schema_contract_error(stage, "schema_version table has no rows"))
        },
        stage,
    )
}

#[cfg(test)]
fn current_version_optional(conn: &AsyncConnection) -> SearchResult<Option<i64>> {
    current_version_optional_at(conn, "schema version read")
}

fn current_version_optional_at(
    conn: &AsyncConnection,
    stage: &'static str,
) -> SearchResult<Option<i64>> {
    match schema_version_state(conn, stage)? {
        SchemaVersionState::MissingTable => Err(map_storage_error_at(
            stage,
            FrankenError::NoSuchTable {
                name: "schema_version".to_owned(),
            },
        )),
        SchemaVersionState::Empty => Ok(None),
        SchemaVersionState::Present(version) => Ok(Some(version)),
    }
}

fn schema_version_state(
    conn: &AsyncConnection,
    stage: &'static str,
) -> SearchResult<SchemaVersionState> {
    let table_columns = conn
        .query_sync("PRAGMA table_info(schema_version);")
        .map_err(|error| map_storage_error_at(stage, error))?;
    if table_columns.is_empty() {
        return Ok(SchemaVersionState::MissingTable);
    }

    let rows = match conn
        .query_sync("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;")
    {
        Ok(rows) => rows,
        Err(FrankenError::NoSuchTable { name }) if name == "schema_version" => {
            return Ok(SchemaVersionState::MissingTable);
        }
        Err(error) => return Err(map_storage_error_at(stage, error)),
    };
    let Some(row) = rows.first() else {
        return Ok(SchemaVersionState::Empty);
    };
    row_i64_at(row, 0, "schema_version.version", stage).map(SchemaVersionState::Present)
}

fn reject_future_version(version: i64, stage: &'static str) -> SearchResult<()> {
    if version > SCHEMA_VERSION {
        return Err(schema_contract_error(
            stage,
            format!("schema version {version} is newer than supported {SCHEMA_VERSION}"),
        ));
    }
    Ok(())
}

pub(crate) fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    row_i64_at(row, index, field, "schema row decode")
}

fn row_i64_at(row: &Row, index: usize, field: &str, stage: &'static str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(schema_contract_error(
            stage,
            format!("unexpected type for {field}: {:?}", other),
        )),
        None => Err(schema_contract_error(
            stage,
            format!("missing column for {field}"),
        )),
    }
}

fn schema_contract_error(stage: &'static str, message: impl Into<String>) -> SearchError {
    map_storage_error_at(stage, io::Error::other(message.into()))
}

#[cfg(test)]
fn storage_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    map_storage_error_at("storage operation", source)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        SCHEMA_VERSION, bootstrap, current_version, current_version_optional, storage_error,
    };
    use fsqlite::AsyncConnection;
    use fsqlite_types::value::SqliteValue;

    fn table_exists(conn: &AsyncConnection, table_name: &str) -> bool {
        // Probe table existence with a zero-row SELECT instead of
        // querying sqlite_master: FrankenSQLite's VDBE cannot open a
        // storage cursor on sqlite_master's btree root page.
        conn.query_sync(&format!("SELECT 1 FROM \"{table_name}\" LIMIT 0"))
            .is_ok()
    }

    fn index_exists(conn: &AsyncConnection, table_name: &str, index_name: &str) -> bool {
        // Probe index existence via INDEXED BY hint instead of querying
        // sqlite_master: FrankenSQLite's VDBE cannot open a storage
        // cursor on sqlite_master's btree root page. If the index
        // doesn't exist, the query errors with "no such index".
        conn.query_sync(&format!(
            "SELECT 1 FROM \"{table_name}\" INDEXED BY \"{index_name}\" LIMIT 0"
        ))
        .is_ok()
    }

    fn seed_historical_schema(conn: &AsyncConnection, version: i64) {
        conn.execute_sync("CREATE TABLE schema_version (version INTEGER PRIMARY KEY);")
            .expect("schema_version table should be creatable");

        for migration in super::MIGRATIONS
            .iter()
            .filter(|migration| migration.version <= version)
        {
            for statement in migration.statements {
                conn.execute_sync(statement).unwrap_or_else(|error| {
                    panic!("seed migration {}: {error}", migration.version)
                });
            }
            let params = [SqliteValue::Integer(migration.version)];
            conn.execute_with_params_sync(
                "INSERT OR REPLACE INTO schema_version(version) VALUES (?1);",
                &params,
            )
            .unwrap_or_else(|error| panic!("seed migration {} marker: {error}", migration.version));
        }
    }

    #[test]
    fn bootstrap_sets_latest_version_for_fresh_database() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("bootstrap should succeed");
        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
        assert!(
            index_exists(&conn, "embedding_status", "idx_embedding_status_pending"),
            "latest schema should include pending-status index"
        );
        assert!(
            index_exists(&conn, "embedding_jobs", "idx_jobs_pending"),
            "latest schema should include queue pending index"
        );
        assert!(
            index_exists(&conn, "embedding_jobs", "idx_jobs_processing"),
            "latest schema should include queue processing index"
        );
    }

    #[test]
    fn bootstrap_accepts_read_only_confirmation_that_marker_table_is_absent() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        assert_eq!(
            super::schema_version_state(&conn, "test absence probe")
                .expect("absence probe should be readable"),
            super::SchemaVersionState::MissingTable
        );

        bootstrap(&conn).expect("confirmed absent marker table should bootstrap");
        assert_eq!(
            current_version(&conn).expect("bootstrapped marker should be readable"),
            SCHEMA_VERSION
        );
    }

    #[test]
    fn bootstrap_does_not_swallow_malformed_marker_state_as_absence() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        conn.execute_sync("CREATE TABLE schema_version (version TEXT PRIMARY KEY);")
            .expect("malformed marker table should be creatable");
        conn.execute_sync("INSERT INTO schema_version(version) VALUES ('not-an-integer');")
            .expect("malformed marker should insert");

        let error = bootstrap(&conn).expect_err("malformed marker state must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("schema preflight"),
            "error should retain the preflight stage: {message}"
        );
        assert!(
            message.contains("unexpected type"),
            "malformed marker must not be reclassified as absence: {message}"
        );
        assert!(
            !conn.in_transaction(),
            "read-only preflight rejection must not leave a transaction active"
        );
    }

    #[test]
    fn bootstrap_migrates_legacy_schema_versions() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        let previous = SCHEMA_VERSION - 1;
        seed_historical_schema(&conn, previous);

        assert_eq!(
            current_version_optional(&conn).expect("version should read"),
            Some(previous)
        );
        bootstrap(&conn).expect("legacy schema should migrate to latest");
        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
        assert!(
            index_exists(&conn, "search_history", "idx_history_query"),
            "v6 search-history index must survive the v7 rebuild-marker migration"
        );
        assert!(
            table_exists(&conn, "frankensearch_fts5_rebuild_version"),
            "v7 migration must create the FTS5 rebuild-version marker table"
        );
    }

    #[test]
    fn bootstrap_migrates_every_historical_schema_edge() {
        for starting_version in 1..SCHEMA_VERSION {
            let conn =
                AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
            seed_historical_schema(&conn, starting_version);

            assert_eq!(
                current_version(&conn).expect("seed marker should be readable"),
                starting_version,
                "fixture should represent schema v{starting_version}"
            );
            bootstrap(&conn).unwrap_or_else(|error| {
                panic!("schema v{starting_version} should migrate to latest: {error}")
            });
            assert_eq!(
                current_version(&conn).expect("migrated marker should be readable"),
                SCHEMA_VERSION,
                "schema v{starting_version} should reach latest"
            );
        }
    }

    #[test]
    fn bootstrap_rejects_future_schema_versions() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");

        conn.execute_sync(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("schema_version should be creatable");
        let future_version = SCHEMA_VERSION + 100;
        let params = [SqliteValue::Integer(future_version)];
        conn.execute_with_params_sync("INSERT INTO schema_version(version) VALUES (?1);", &params)
            .expect("future version marker should insert");

        let error = bootstrap(&conn).expect_err("future schemas should be rejected");
        let message = error.to_string();
        assert!(
            message.contains("newer than supported"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn bootstrap_is_idempotent_at_latest_version() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");

        bootstrap(&conn).expect("first bootstrap should succeed");
        bootstrap(&conn).expect("second bootstrap should succeed");
        bootstrap(&conn).expect("third bootstrap should succeed");

        assert_eq!(
            current_version(&conn).expect("schema version should exist"),
            SCHEMA_VERSION
        );
    }

    #[test]
    fn concurrent_bootstrap_on_disk_is_race_safe() {
        const THREADS: usize = 6;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let db_path = std::env::temp_dir().join(format!(
            "frankensearch-schema-bootstrap-{}-{nanos}.sqlite3",
            process::id()
        ));

        let barrier = Arc::new(Barrier::new(THREADS));
        let (tx, rx) = mpsc::channel::<i64>();
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let gate = Arc::clone(&barrier);
            let sender = tx.clone();
            let path = db_path.clone();
            handles.push(thread::spawn(move || {
                gate.wait();
                let conn = AsyncConnection::open_sync(path.to_string_lossy().to_string())
                    .expect("connection should open");
                // Retry bootstrap to handle SQLITE_BUSY under contention.
                let mut last_err = None;
                for _ in 0..10 {
                    match bootstrap(&conn) {
                        Ok(()) => {
                            last_err = None;
                            break;
                        }
                        Err(error) => {
                            assert!(
                                !conn.in_transaction(),
                                "failed bootstrap must not leave an active transaction: {error}"
                            );
                            let message = error.to_string().to_ascii_lowercase();
                            if message.contains("busy") || message.contains("locked") {
                                last_err = Some(error);
                                thread::sleep(std::time::Duration::from_millis(5));
                                continue;
                            }
                            panic!("bootstrap should not fail with non-retryable error: {error}");
                        }
                    }
                }
                if let Some(error) = last_err {
                    panic!("bootstrap should succeed after retries: {error}");
                }
                let mut version = None;
                for _ in 0..10 {
                    match current_version(&conn) {
                        Ok(value) => {
                            version = Some(value);
                            break;
                        }
                        Err(error) => {
                            let message = error.to_string().to_ascii_lowercase();
                            if message.contains("busy")
                                || message.contains("locked")
                                || message.contains("no rows")
                            {
                                thread::sleep(std::time::Duration::from_millis(5));
                                continue;
                            }
                            panic!(
                                "schema version lookup should not fail with non-retryable error: {error}"
                            );
                        }
                    }
                }
                let version = version.expect("schema version should exist after retries");
                sender.send(version).expect("sender should send version");
            }));
        }
        drop(tx);

        let versions: Vec<i64> = rx.iter().collect();
        for handle in handles {
            handle.join().expect("bootstrap thread should join");
        }

        assert_eq!(versions.len(), THREADS);
        assert!(
            versions.iter().all(|version| *version == SCHEMA_VERSION),
            "all concurrent bootstraps should resolve to latest schema"
        );

        let db_path_display = db_path.display().to_string();
        let cleanup_targets = [
            db_path,
            PathBuf::from(format!("{db_path_display}.wal")),
            PathBuf::from(format!("{db_path_display}.shm")),
        ];
        for target in cleanup_targets {
            let _ = std::fs::remove_file(target);
        }
    }

    // ── Schema version constant ─────────────────────────────────────────

    #[test]
    fn schema_version_is_seven() {
        assert_eq!(SCHEMA_VERSION, 7);
    }

    // ── Migration array invariants ──────────────────────────────────────

    #[test]
    fn migrations_cover_all_versions_one_through_latest() {
        for version in 1..=SCHEMA_VERSION {
            assert!(
                super::MIGRATIONS.iter().any(|m| m.version == version),
                "missing migration for version {version}"
            );
        }
    }

    #[test]
    fn migrations_are_ascending_order() {
        for window in super::MIGRATIONS.windows(2) {
            assert!(
                window[0].version < window[1].version,
                "migration versions not ascending: {} >= {}",
                window[0].version,
                window[1].version
            );
        }
    }

    #[test]
    fn migrations_have_no_empty_statements() {
        for migration in super::MIGRATIONS {
            assert!(
                !migration.statements.is_empty(),
                "migration {} has empty statements",
                migration.version
            );
        }
    }

    // ── LATEST_SCHEMA invariants ────────────────────────────────────────

    #[test]
    fn latest_schema_is_nonempty() {
        assert!(
            !super::LATEST_SCHEMA.is_empty(),
            "LATEST_SCHEMA must have at least one statement"
        );
    }

    // ── All tables exist after bootstrap ─────────────────────────────────

    #[test]
    fn bootstrap_creates_all_expected_tables() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");

        let expected_tables = [
            "documents",
            "embedding_jobs",
            "embedding_status",
            "content_hashes",
            "index_metadata",
            "index_build_history",
            "search_history",
            "bookmarks",
            "schema_version",
        ];

        for table in expected_tables {
            assert!(
                table_exists(&conn, table),
                "table '{table}' should exist after bootstrap"
            );
        }
    }

    // ── All indices exist after bootstrap ────────────────────────────────

    #[test]
    fn bootstrap_creates_all_expected_indices() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");

        let expected_indices: &[(&str, &str)] = &[
            ("documents", "idx_documents_content_hash"),
            ("documents", "idx_documents_updated_at"),
            ("embedding_status", "idx_embedding_status_pending"),
            ("embedding_jobs", "idx_jobs_pending"),
            ("embedding_jobs", "idx_jobs_processing"),
            ("index_build_history", "idx_build_history_index"),
            ("search_history", "idx_history_query"),
            ("search_history", "idx_history_ts"),
            ("bookmarks", "idx_bookmarks_doc_query"),
        ];

        for &(table, idx) in expected_indices {
            assert!(
                index_exists(&conn, table, idx),
                "index '{idx}' should exist on '{table}' after bootstrap"
            );
        }
    }

    // ── row_i64 edge cases ──────────────────────────────────────────────

    #[test]
    fn row_i64_wrong_type_returns_error() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        conn.execute_sync("CREATE TABLE test_row (val TEXT);")
            .expect("create");
        let params = [SqliteValue::Text("hello".to_owned().into())];
        conn.execute_with_params_sync("INSERT INTO test_row(val) VALUES (?1);", &params)
            .expect("insert");

        let rows = conn
            .query_sync("SELECT val FROM test_row LIMIT 1;")
            .expect("query");
        let row = rows.first().expect("should have a row");
        let err = super::row_i64(row, 0, "test_field").expect_err("text should not parse as i64");
        let msg = err.to_string();
        assert!(
            msg.contains("unexpected type") || msg.contains("test_field"),
            "error should mention type mismatch: {msg}"
        );
    }

    #[test]
    fn row_i64_missing_column_returns_error() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        conn.execute_sync("CREATE TABLE test_row2 (val INTEGER);")
            .expect("create");
        let params = [SqliteValue::Integer(42)];
        conn.execute_with_params_sync("INSERT INTO test_row2(val) VALUES (?1);", &params)
            .expect("insert");

        let rows = conn
            .query_sync("SELECT val FROM test_row2 LIMIT 1;")
            .expect("query");
        let row = rows.first().expect("should have a row");
        let err = super::row_i64(row, 99, "missing_col").expect_err("out-of-bounds index");
        let msg = err.to_string();
        assert!(
            msg.contains("missing column") || msg.contains("missing_col"),
            "error should mention missing column: {msg}"
        );
    }

    #[test]
    fn row_i64_valid_integer_returns_value() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        conn.execute_sync("CREATE TABLE test_row3 (val INTEGER);")
            .expect("create");
        let params = [SqliteValue::Integer(42)];
        conn.execute_with_params_sync("INSERT INTO test_row3(val) VALUES (?1);", &params)
            .expect("insert");

        let rows = conn
            .query_sync("SELECT val FROM test_row3 LIMIT 1;")
            .expect("query");
        let row = rows.first().expect("should have a row");
        let value = super::row_i64(row, 0, "val").expect("should extract integer");
        assert_eq!(value, 42);
    }

    // ── current_version on empty table ──────────────────────────────────

    #[test]
    fn current_version_optional_empty_table_returns_none() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        conn.execute_sync(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("create table");
        let result = current_version_optional(&conn).expect("should not error");
        assert_eq!(result, None);
    }

    #[test]
    fn current_version_empty_table_returns_error() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        conn.execute_sync(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);",
        )
        .expect("create table");
        let err = current_version(&conn).expect_err("should fail on empty table");
        let msg = err.to_string();
        assert!(
            msg.contains("no rows"),
            "error should mention no rows: {msg}"
        );
    }

    #[test]
    fn current_version_missing_table_preserves_typed_cause_and_stage() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        let error = current_version(&conn).expect_err("missing marker table should fail");
        assert!(
            error.to_string().contains("schema version read"),
            "error should identify the failed stage: {error}"
        );

        let frankensearch_core::SearchError::SubsystemError { source, .. } = &error else {
            panic!("expected storage subsystem error, got {error:?}");
        };
        let typed_cause = std::error::Error::source(source.as_ref())
            .and_then(|cause| cause.downcast_ref::<fsqlite::FrankenError>());
        assert!(
            matches!(
                typed_cause,
                Some(fsqlite::FrankenError::NoSuchTable { name }) if name == "schema_version"
            ),
            "error chain should retain exact NoSuchTable(schema_version), got {typed_cause:?}"
        );
    }

    // ── storage_error ───────────────────────────────────────────────────

    #[test]
    fn storage_error_wraps_io_error() {
        let io_err = std::io::Error::other("test error");
        let err = storage_error(io_err);
        match &err {
            frankensearch_core::SearchError::SubsystemError { subsystem, .. } => {
                assert_eq!(*subsystem, "storage");
            }
            other => panic!("expected SubsystemError, got: {other:?}"),
        }
    }

    // ── Tables are queryable after bootstrap ─────────────────────────────

    #[test]
    fn tables_accept_basic_queries_after_bootstrap() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");

        // Verify each table is queryable
        for query in [
            "SELECT COUNT(*) FROM documents;",
            "SELECT COUNT(*) FROM embedding_jobs;",
            "SELECT COUNT(*) FROM embedding_status;",
            "SELECT COUNT(*) FROM content_hashes;",
            "SELECT COUNT(*) FROM index_metadata;",
            "SELECT COUNT(*) FROM index_build_history;",
            "SELECT COUNT(*) FROM search_history;",
            "SELECT COUNT(*) FROM bookmarks;",
        ] {
            conn.query_sync(query)
                .unwrap_or_else(|_| panic!("query should succeed: {query}"));
        }
    }

    // ── Schema version is always latest after bootstrap ──────────────────

    #[test]
    fn schema_version_row_contains_latest_after_fresh_bootstrap() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");

        let rows = conn
            .query_sync("SELECT version FROM schema_version ORDER BY version DESC;")
            .expect("query schema_version");
        assert!(!rows.is_empty(), "schema_version table should have rows");

        let version = super::row_i64(&rows[0], 0, "version").expect("extract version");
        assert_eq!(version, SCHEMA_VERSION);
    }

    // ── index_exists helper returns false for nonexistent ─────────────────

    #[test]
    fn index_exists_returns_false_for_nonexistent_index() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");
        // Use a nonexistent table name: FrankenSQLite's VDBE does not
        // validate INDEXED BY hints at query-planning time, so probing
        // a real table with a fake index incorrectly succeeds.  A
        // nonexistent table reliably triggers the expected failure.
        assert!(!index_exists(
            &conn,
            "table_that_does_not_exist",
            "idx_nonexistent_never_created"
        ));
    }

    // ── Insert and query documents table ─────────────────────────────────

    #[test]
    fn documents_table_accepts_insert_and_select() {
        let conn = AsyncConnection::open_sync(":memory:".to_owned()).expect("in-memory connection");
        bootstrap(&conn).expect("bootstrap should succeed");

        let params = [
            SqliteValue::Text("doc-001".to_owned().into()),
            SqliteValue::Text("/path/to/file.rs".to_owned().into()),
            SqliteValue::Text("fn main() {}".to_owned().into()),
            SqliteValue::Blob(vec![0xAB, 0xCD].into()),
            SqliteValue::Integer(12),
            SqliteValue::Integer(1000),
            SqliteValue::Integer(1000),
        ];
        conn.execute_with_params_sync("INSERT INTO documents(doc_id, source_path, content_preview, content_hash, content_length, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7);",
        &params,)
        .expect("insert should succeed");

        let rows = conn
            .query_sync("SELECT doc_id FROM documents WHERE doc_id = 'doc-001';")
            .expect("select should succeed");
        assert_eq!(rows.len(), 1);
    }
}
