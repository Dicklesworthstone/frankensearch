//! Index metadata persistence for tracking built indices, build history,
//! and staleness signals.
//!
//! Provides the storage layer for bd-3w1.11: every index build is recorded
//! with its embedder, dimension, timing, and optional FEC sidecar info.
//! Build history is retained (last 100 per index) for staleness detection
//! and KL drift analysis.

use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

use crate::connection::{Storage, map_storage_error};

/// What triggered an index build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildTrigger {
    /// First-time build on startup.
    Initial,
    /// Scheduled periodic rebuild.
    Scheduled,
    /// Content changes detected (new/modified documents).
    ContentChanged,
    /// Explicit user or API request.
    Manual,
    /// Embedder model or config changed.
    ConfigChanged,
    /// Repair pipeline triggered a rebuild.
    Repair,
}

impl BuildTrigger {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Scheduled => "scheduled",
            Self::ContentChanged => "content_changed",
            Self::Manual => "manual",
            Self::ConfigChanged => "config_changed",
            Self::Repair => "repair",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "initial" => Some(Self::Initial),
            "scheduled" => Some(Self::Scheduled),
            "content_changed" => Some(Self::ContentChanged),
            "manual" => Some(Self::Manual),
            "config_changed" => Some(Self::ConfigChanged),
            "repair" => Some(Self::Repair),
            _ => None,
        }
    }
}

/// Why an index is considered stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StalenessReason {
    /// Documents have been modified since last build.
    ContentChanged,
    /// New documents were added since last build.
    NewDocuments,
    /// Index has never been built.
    NeverBuilt,
    /// Embedder revision changed since last build.
    EmbedderChanged,
    /// Index age exceeded the configured maximum.
    AgeExceeded,
    /// Storage schema version differs from the index build schema version.
    SchemaChanged,
}

/// Result of a staleness check for a given index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessCheck {
    pub index_name: String,
    pub is_stale: bool,
    pub reasons: Vec<StalenessReason>,
    /// Number of documents modified since last build.
    pub docs_modified: i64,
    /// Number of new documents since last build.
    pub docs_added: i64,
    /// Timestamp of the last build (None if never built).
    pub last_built_at: Option<i64>,
}

/// Metadata about a single index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexMetadata {
    pub index_name: String,
    pub index_type: String,
    pub embedder_id: String,
    pub embedder_revision: Option<String>,
    pub dimension: i64,
    pub record_count: i64,
    pub file_path: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub file_hash: Option<String>,
    pub schema_version: Option<i64>,
    pub built_at: Option<i64>,
    pub build_duration_ms: Option<i64>,
    pub source_doc_count: i64,
    pub config_json: Option<String>,
    pub fec_path: Option<String>,
    pub fec_size_bytes: Option<i64>,
    pub last_verified_at: Option<i64>,
    pub last_repair_at: Option<i64>,
    pub repair_count: i64,
    pub mean_norm: Option<f64>,
    pub variance: Option<f64>,
}

/// A single entry in the build history log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexBuildRecord {
    pub build_id: i64,
    pub index_name: String,
    pub built_at: i64,
    pub build_duration_ms: i64,
    pub record_count: i64,
    pub source_doc_count: i64,
    pub trigger: BuildTrigger,
    pub config_json: Option<String>,
    pub notes: Option<String>,
    pub mean_norm: Option<f64>,
    pub variance: Option<f64>,
}

/// Parameters for recording an index build.
#[derive(Debug, Clone)]
pub struct RecordBuildParams {
    pub index_name: String,
    pub index_type: String,
    pub embedder_id: String,
    pub embedder_revision: Option<String>,
    pub dimension: i64,
    pub record_count: i64,
    pub source_doc_count: i64,
    pub build_duration_ms: i64,
    pub trigger: BuildTrigger,
    pub file_path: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub file_hash: Option<String>,
    pub schema_version: Option<i64>,
    pub config_json: Option<String>,
    pub fec_path: Option<String>,
    pub fec_size_bytes: Option<i64>,
    pub notes: Option<String>,
    pub mean_norm: Option<f64>,
    pub variance: Option<f64>,
}

// ─── Storage methods ────────────────────────────────────────────────────────

impl Storage {
    /// Record a completed index build. Upserts the `index_metadata` row and
    /// appends to `index_build_history`. Retains at most 100 history entries
    /// per index (oldest are pruned).
    pub fn record_index_build(&self, params: &RecordBuildParams) -> SearchResult<()> {
        ensure_non_empty(&params.index_name, "index_name")?;
        ensure_non_empty(&params.index_type, "index_type")?;
        ensure_non_empty(&params.embedder_id, "embedder_id")?;

        let built_at = unix_timestamp_ms()?;

        self.transaction(|conn| {
            upsert_index_metadata(conn, params, built_at)?;
            insert_build_history(conn, params, built_at)?;
            prune_build_history(conn, &params.index_name, 100)?;
            Ok(())
        })?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "record_index_build",
            index_name = %params.index_name,
            index_type = %params.index_type,
            embedder_id = %params.embedder_id,
            record_count = params.record_count,
            trigger = params.trigger.as_str(),
            "index build recorded"
        );

        Ok(())
    }

    /// Fetch metadata for a specific index by name.
    pub fn get_index_metadata(&self, index_name: &str) -> SearchResult<Option<IndexMetadata>> {
        ensure_non_empty(index_name, "index_name")?;

        let params = [SqliteValue::Text(index_name.to_owned())];
        let rows = self
            .connection()
            .query_with_params(
                "SELECT index_name, index_type, embedder_id, embedder_revision, \
                    dimension, record_count, file_path, file_size_bytes, file_hash, \
                    schema_version, built_at, build_duration_ms, source_doc_count, \
                    config_json, fec_path, fec_size_bytes, last_verified_at, \
                    last_repair_at, repair_count, mean_norm, variance \
                 FROM index_metadata WHERE index_name = ?1 LIMIT 1;",
                &params,
            )
            .map_err(map_storage_error)?;

        let Some(row) = rows.first() else {
            return Ok(None);
        };

        Ok(Some(parse_index_metadata(row)?))
    }

    /// List all known index metadata entries.
    pub fn list_index_metadata(&self) -> SearchResult<Vec<IndexMetadata>> {
        let rows = self
            .connection()
            .query(
                "SELECT index_name, index_type, embedder_id, embedder_revision, \
                    dimension, record_count, file_path, file_size_bytes, file_hash, \
                    schema_version, built_at, build_duration_ms, source_doc_count, \
                    config_json, fec_path, fec_size_bytes, last_verified_at, \
                    last_repair_at, repair_count, mean_norm, variance \
                 FROM index_metadata ORDER BY index_name;",
            )
            .map_err(map_storage_error)?;

        let mut result = Vec::with_capacity(rows.len());
        for row in &rows {
            result.push(parse_index_metadata(row)?);
        }
        Ok(result)
    }

    /// Check whether an index is stale by comparing document timestamps
    /// and embedder revision against the stored metadata.
    pub fn check_index_staleness(
        &self,
        index_name: &str,
        current_embedder_revision: Option<&str>,
    ) -> SearchResult<StalenessCheck> {
        ensure_non_empty(index_name, "index_name")?;

        let meta = self.get_index_metadata(index_name)?;

        let Some(meta) = meta else {
            return Ok(StalenessCheck {
                index_name: index_name.to_owned(),
                is_stale: true,
                reasons: vec![StalenessReason::NeverBuilt],
                docs_modified: 0,
                docs_added: 0,
                last_built_at: None,
            });
        };

        let mut reasons = Vec::new();
        let built_at = meta.built_at.unwrap_or(0);

        // Check for modified documents since last build.
        let current_docs = count_total_documents(self.connection())?;
        let docs_removed = if current_docs < meta.source_doc_count {
            meta.source_doc_count - current_docs
        } else {
            0
        };
        let docs_modified =
            count_modified_since(self.connection(), built_at)?.saturating_add(docs_removed);
        if docs_modified > 0 {
            reasons.push(StalenessReason::ContentChanged);
        }

        // Check for new documents since last build.
        let docs_added = count_added_since(self.connection(), built_at)?;
        if docs_added > 0 {
            reasons.push(StalenessReason::NewDocuments);
        }

        // Check embedder revision change.
        if let Some(current_rev) = current_embedder_revision {
            match &meta.embedder_revision {
                Some(stored_rev) if stored_rev != current_rev => {
                    reasons.push(StalenessReason::EmbedderChanged);
                }
                None => {
                    // No stored revision implies unknown — treat as changed.
                    reasons.push(StalenessReason::EmbedderChanged);
                }
                _ => {}
            }
        }

        Ok(StalenessCheck {
            index_name: index_name.to_owned(),
            is_stale: !reasons.is_empty(),
            reasons,
            docs_modified,
            docs_added,
            last_built_at: meta.built_at,
        })
    }

    /// Record a successful integrity verification of an index file.
    pub fn record_verification(&self, index_name: &str) -> SearchResult<bool> {
        ensure_non_empty(index_name, "index_name")?;

        let now = unix_timestamp_ms()?;
        let params = [
            SqliteValue::Integer(now),
            SqliteValue::Text(index_name.to_owned()),
        ];

        let affected = self
            .connection()
            .execute_with_params(
                "UPDATE index_metadata SET last_verified_at = ?1 WHERE index_name = ?2;",
                &params,
            )
            .map_err(map_storage_error)?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "record_verification",
            index_name,
            updated = affected > 0,
            "index verification recorded"
        );

        Ok(affected > 0)
    }

    /// Record a successful repair of an index file.
    pub fn record_repair(&self, index_name: &str) -> SearchResult<bool> {
        ensure_non_empty(index_name, "index_name")?;

        let now = unix_timestamp_ms()?;
        let params = [
            SqliteValue::Integer(now),
            SqliteValue::Text(index_name.to_owned()),
        ];

        let affected = self
            .connection()
            .execute_with_params(
                "UPDATE index_metadata SET last_repair_at = ?1, \
                    repair_count = repair_count + 1 \
                 WHERE index_name = ?2;",
                &params,
            )
            .map_err(map_storage_error)?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "record_repair",
            index_name,
            updated = affected > 0,
            "index repair recorded"
        );

        Ok(affected > 0)
    }

    /// Fetch the build history for an index, most recent first.
    pub fn get_build_history(
        &self,
        index_name: &str,
        limit: usize,
    ) -> SearchResult<Vec<IndexBuildRecord>> {
        ensure_non_empty(index_name, "index_name")?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        // NOTE: Uses format!() with sql_escape_single_quoted instead of
        // query_with_params because FrankenSQLite's query_with_params
        // currently returns at most one row for multi-row result sets.
        let escaped_index = sql_escape_single_quoted(index_name);
        let sql = format!(
            "SELECT build_id, index_name, built_at, build_duration_ms, \
                record_count, source_doc_count, \"trigger\", config_json, notes, \
                mean_norm, variance \
             FROM index_build_history \
             WHERE index_name = '{escaped_index}' \
             ORDER BY built_at DESC, build_id DESC;"
        );
        let rows = self.connection().query(&sql).map_err(map_storage_error)?;

        let mut history = Vec::with_capacity(rows.len());
        for row in &rows {
            history.push(parse_build_record(row)?);
        }
        if history.len() > limit {
            history.truncate(limit);
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "get_build_history",
            index_name,
            limit,
            count = history.len(),
            "build history query completed"
        );

        Ok(history)
    }

    /// Delete metadata and build history for an index.
    pub fn delete_index_metadata(&self, index_name: &str) -> SearchResult<bool> {
        ensure_non_empty(index_name, "index_name")?;

        let (history_deleted, metadata_deleted) = self.transaction(|conn| {
            let params = [SqliteValue::Text(index_name.to_owned())];
            let history_deleted = conn
                .execute_with_params(
                    "DELETE FROM index_build_history WHERE index_name = ?1;",
                    &params,
                )
                .map_err(map_storage_error)?;
            let metadata_deleted = conn
                .execute_with_params("DELETE FROM index_metadata WHERE index_name = ?1;", &params)
                .map_err(map_storage_error)?;
            Ok((history_deleted, metadata_deleted))
        })?;
        let deleted_any = history_deleted > 0 || metadata_deleted > 0;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "delete_index_metadata",
            index_name,
            history_deleted,
            metadata_deleted,
            deleted = deleted_any,
            "index metadata deleted"
        );

        Ok(deleted_any)
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

fn upsert_index_metadata(
    conn: &Connection,
    params: &RecordBuildParams,
    built_at: i64,
) -> SearchResult<()> {
    let update_params = [
        SqliteValue::Text(params.index_type.clone()),
        SqliteValue::Text(params.embedder_id.clone()),
        sqlite_text_opt(params.embedder_revision.as_deref()),
        SqliteValue::Integer(params.dimension),
        SqliteValue::Integer(params.record_count),
        sqlite_text_opt(params.file_path.as_deref()),
        sqlite_i64_opt(params.file_size_bytes),
        sqlite_text_opt(params.file_hash.as_deref()),
        sqlite_i64_opt(params.schema_version),
        SqliteValue::Integer(built_at),
        SqliteValue::Integer(params.build_duration_ms),
        SqliteValue::Integer(params.source_doc_count),
        sqlite_text_opt(params.config_json.as_deref()),
        sqlite_text_opt(params.fec_path.as_deref()),
        sqlite_i64_opt(params.fec_size_bytes),
        sqlite_f64_opt(params.mean_norm),
        sqlite_f64_opt(params.variance),
        SqliteValue::Text(params.index_name.clone()),
    ];

    let updated = conn
        .execute_with_params(
            "UPDATE index_metadata SET \
                index_type = ?1, \
                embedder_id = ?2, \
                embedder_revision = ?3, \
                dimension = ?4, \
                record_count = ?5, \
                file_path = ?6, \
                file_size_bytes = ?7, \
                file_hash = ?8, \
                schema_version = ?9, \
                built_at = ?10, \
                build_duration_ms = ?11, \
                source_doc_count = ?12, \
                config_json = ?13, \
                fec_path = ?14, \
                fec_size_bytes = ?15, \
                mean_norm = ?16, \
                variance = ?17 \
             WHERE index_name = ?18;",
            &update_params,
        )
        .map_err(map_storage_error)?;
    if updated > 0 {
        return Ok(());
    }

    let insert_params = [
        SqliteValue::Text(params.index_name.clone()),
        SqliteValue::Text(params.index_type.clone()),
        SqliteValue::Text(params.embedder_id.clone()),
        sqlite_text_opt(params.embedder_revision.as_deref()),
        SqliteValue::Integer(params.dimension),
        SqliteValue::Integer(params.record_count),
        sqlite_text_opt(params.file_path.as_deref()),
        sqlite_i64_opt(params.file_size_bytes),
        sqlite_text_opt(params.file_hash.as_deref()),
        sqlite_i64_opt(params.schema_version),
        SqliteValue::Integer(built_at),
        SqliteValue::Integer(params.build_duration_ms),
        SqliteValue::Integer(params.source_doc_count),
        sqlite_text_opt(params.config_json.as_deref()),
        sqlite_text_opt(params.fec_path.as_deref()),
        sqlite_i64_opt(params.fec_size_bytes),
        SqliteValue::Null,
        SqliteValue::Null,
        SqliteValue::Integer(0),
        sqlite_f64_opt(params.mean_norm),
        sqlite_f64_opt(params.variance),
    ];

    conn.execute_with_params(
        "INSERT INTO index_metadata (\
            index_name, index_type, embedder_id, embedder_revision, \
            dimension, record_count, file_path, file_size_bytes, file_hash, \
            schema_version, built_at, build_duration_ms, source_doc_count, \
            config_json, fec_path, fec_size_bytes, last_verified_at, \
            last_repair_at, repair_count, mean_norm, variance\
         ) VALUES (\
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, \
            ?15, ?16, ?17, ?18, ?19, ?20, ?21\
         );",
        &insert_params,
    )
    .map_err(map_storage_error)?;

    Ok(())
}

fn insert_build_history(
    conn: &Connection,
    params: &RecordBuildParams,
    built_at: i64,
) -> SearchResult<()> {
    let sql_params = [
        SqliteValue::Text(params.index_name.clone()),
        SqliteValue::Integer(built_at),
        SqliteValue::Integer(params.build_duration_ms),
        SqliteValue::Integer(params.record_count),
        SqliteValue::Integer(params.source_doc_count),
        SqliteValue::Text(params.trigger.as_str().to_owned()),
        sqlite_text_opt(params.config_json.as_deref()),
        sqlite_text_opt(params.notes.as_deref()),
        sqlite_f64_opt(params.mean_norm),
        sqlite_f64_opt(params.variance),
    ];

    conn.execute_with_params(
        "INSERT INTO index_build_history (\
            index_name, built_at, build_duration_ms, record_count, \
            source_doc_count, \"trigger\", config_json, notes, mean_norm, variance\
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10);",
        &sql_params,
    )
    .map_err(map_storage_error)?;

    Ok(())
}

fn prune_build_history(conn: &Connection, index_name: &str, keep: i64) -> SearchResult<()> {
    let escaped_index = sql_escape_single_quoted(index_name);

    if keep <= 0 {
        conn.execute(&format!(
            "DELETE FROM index_build_history WHERE index_name = '{escaped_index}';"
        ))
        .map_err(map_storage_error)?;
        return Ok(());
    }
    let keep = usize::try_from(keep).map_err(|error| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!(
            "invalid keep value for prune_build_history: {error}"
        ))),
    })?;

    // NOTE: Uses format!() because query_with_params returns at most one
    // row for multi-row result sets (FrankenSQLite limitation).
    let rows = conn
        .query(&format!(
            "SELECT build_id FROM index_build_history \
             WHERE index_name = '{escaped_index}' \
             ORDER BY build_id DESC;"
        ))
        .map_err(map_storage_error)?;
    if rows.len() <= keep {
        return Ok(());
    }

    let cutoff_row = &rows[keep - 1];
    let cutoff = row_i64(cutoff_row, 0, "index_build_history.build_id")?;

    conn.execute(&format!(
        "DELETE FROM index_build_history \
         WHERE index_name = '{escaped_index}' \
           AND build_id < {cutoff};"
    ))
    .map_err(map_storage_error)?;
    Ok(())
}

fn count_modified_since(conn: &Connection, since: i64) -> SearchResult<i64> {
    let params = [SqliteValue::Integer(since), SqliteValue::Integer(since)];
    let rows = conn
        .query_with_params(
            "SELECT COUNT(*) FROM documents WHERE updated_at > ?1 AND created_at <= ?2;",
            &params,
        )
        .map_err(map_storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(0);
    };
    row_i64(row, 0, "count_modified")
}

fn count_added_since(conn: &Connection, since: i64) -> SearchResult<i64> {
    let params = [SqliteValue::Integer(since)];
    let rows = conn
        .query_with_params(
            "SELECT COUNT(*) FROM documents WHERE created_at > ?1;",
            &params,
        )
        .map_err(map_storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(0);
    };
    row_i64(row, 0, "count_added")
}

fn count_total_documents(conn: &Connection) -> SearchResult<i64> {
    let row = conn
        .query_row("SELECT COUNT(*) FROM documents;")
        .map_err(map_storage_error)?;
    row_i64(&row, 0, "documents.count")
}

fn parse_index_metadata(row: &Row) -> SearchResult<IndexMetadata> {
    Ok(IndexMetadata {
        index_name: row_text(row, 0, "index_metadata.index_name")?.to_owned(),
        index_type: row_text(row, 1, "index_metadata.index_type")?.to_owned(),
        embedder_id: row_text(row, 2, "index_metadata.embedder_id")?.to_owned(),
        embedder_revision: row_optional_text(row, 3)?,
        dimension: row_i64(row, 4, "index_metadata.dimension")?,
        record_count: row_i64(row, 5, "index_metadata.record_count")?,
        file_path: row_optional_text(row, 6)?,
        file_size_bytes: row_optional_i64(row, 7)?,
        file_hash: row_optional_text(row, 8)?,
        schema_version: row_optional_i64(row, 9)?,
        built_at: row_optional_i64(row, 10)?,
        build_duration_ms: row_optional_i64(row, 11)?,
        source_doc_count: row_i64(row, 12, "index_metadata.source_doc_count")?,
        config_json: row_optional_text(row, 13)?,
        fec_path: row_optional_text(row, 14)?,
        fec_size_bytes: row_optional_i64(row, 15)?,
        last_verified_at: row_optional_i64(row, 16)?,
        last_repair_at: row_optional_i64(row, 17)?,
        repair_count: row_i64(row, 18, "index_metadata.repair_count")?,
        mean_norm: row_optional_f64(row, 19)?,
        variance: row_optional_f64(row, 20)?,
    })
}

fn parse_build_record(row: &Row) -> SearchResult<IndexBuildRecord> {
    let trigger_str = row_text(row, 6, "index_build_history.trigger")?;
    let trigger =
        BuildTrigger::from_str(trigger_str).ok_or_else(|| SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unknown build trigger: {trigger_str}"
            ))),
        })?;

    Ok(IndexBuildRecord {
        build_id: row_i64(row, 0, "index_build_history.build_id")?,
        index_name: row_text(row, 1, "index_build_history.index_name")?.to_owned(),
        built_at: row_i64(row, 2, "index_build_history.built_at")?,
        build_duration_ms: row_i64(row, 3, "index_build_history.build_duration_ms")?,
        record_count: row_i64(row, 4, "index_build_history.record_count")?,
        source_doc_count: row_i64(row, 5, "index_build_history.source_doc_count")?,
        trigger,
        config_json: row_optional_text(row, 7)?,
        notes: row_optional_text(row, 8)?,
        mean_norm: row_optional_f64(row, 9)?,
        variance: row_optional_f64(row, 10)?,
    })
}

// ─── Row extraction helpers ─────────────────────────────────────────────────

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {other:?}"
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn row_optional_text(row: &Row, index: usize) -> SearchResult<Option<String>> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(Some(value.clone())),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected optional text type: {other:?}"
            ))),
        }),
    }
}

fn row_optional_i64(row: &Row, index: usize) -> SearchResult<Option<i64>> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(Some(*value)),
        Some(SqliteValue::Null) | None => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected optional i64 type: {other:?}"
            ))),
        }),
    }
}

fn row_optional_f64(row: &Row, index: usize) -> SearchResult<Option<f64>> {
    match row.get(index) {
        Some(SqliteValue::Float(value)) => Ok(Some(*value)),
        Some(SqliteValue::Null) | None => Ok(None),
        // Integer 0 is sometimes stored for "no value" — coerce.
        #[allow(clippy::cast_precision_loss)]
        Some(SqliteValue::Integer(value)) => Ok(Some(*value as f64)),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected optional f64 type: {other:?}"
            ))),
        }),
    }
}

fn sqlite_text_opt(value: Option<&str>) -> SqliteValue {
    value.map_or(SqliteValue::Null, |v| SqliteValue::Text(v.to_owned()))
}

fn sqlite_i64_opt(value: Option<i64>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::Integer)
}

fn sqlite_f64_opt(value: Option<f64>) -> SqliteValue {
    value.map_or(SqliteValue::Null, SqliteValue::Float)
}

/// Escape single quotes for SQL string literals.
///
/// Used as a workaround for multi-row SELECT queries where
/// `query_with_params` currently returns at most one row (`FrankenSQLite`
/// limitation).  Single-row queries and DML (INSERT/UPDATE/DELETE) use
/// parameterized queries instead.
fn sql_escape_single_quoted(value: &str) -> String {
    value.replace('\'', "''")
}

fn ensure_non_empty(value: &str, field: &'static str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!("{field} must not be empty"))),
        });
    }
    Ok(())
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(map_storage_error)?;
    i64::try_from(duration.as_millis()).map_err(|error| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!(
            "unix timestamp overflow: {error}"
        ))),
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::connection::Storage;
    use crate::document::DocumentRecord;

    use super::*;

    fn sample_build_params(name: &str) -> RecordBuildParams {
        RecordBuildParams {
            index_name: name.to_owned(),
            index_type: "fsvi".to_owned(),
            embedder_id: "potion-128m".to_owned(),
            embedder_revision: Some("v1.0".to_owned()),
            dimension: 256,
            record_count: 100,
            source_doc_count: 0,
            build_duration_ms: 500,
            trigger: BuildTrigger::Initial,
            file_path: Some("/tmp/index.fsvi".to_owned()),
            file_size_bytes: Some(102_400),
            file_hash: Some("abc123".to_owned()),
            schema_version: Some(1),
            config_json: Some(r#"{"tier":"fast"}"#.to_owned()),
            fec_path: None,
            fec_size_bytes: None,
            notes: None,
            mean_norm: Some(1.0),
            variance: Some(0.05),
        }
    }

    fn insert_test_document(storage: &Storage, doc_id: &str, created_at: i64, updated_at: i64) {
        let doc = DocumentRecord::new(
            doc_id,
            "test content",
            [0x42; 32],
            64,
            created_at,
            updated_at,
        );
        storage.upsert_document(&doc).expect("doc insert");
    }

    #[test]
    fn record_and_get_round_trip() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("fast-index");

        storage.record_index_build(&params).expect("record build");

        let meta = storage
            .get_index_metadata("fast-index")
            .expect("get metadata")
            .expect("should exist");

        assert_eq!(meta.index_name, "fast-index");
        assert_eq!(meta.index_type, "fsvi");
        assert_eq!(meta.embedder_id, "potion-128m");
        assert_eq!(meta.embedder_revision.as_deref(), Some("v1.0"));
        assert_eq!(meta.dimension, 256);
        assert_eq!(meta.record_count, 100);
        assert_eq!(meta.source_doc_count, 0);
        assert_eq!(meta.file_path.as_deref(), Some("/tmp/index.fsvi"));
        assert_eq!(meta.file_size_bytes, Some(102_400));
        assert_eq!(meta.mean_norm, Some(1.0));
        assert_eq!(meta.variance, Some(0.05));
        assert!(meta.built_at.is_some());
        assert_eq!(meta.repair_count, 0);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let storage = Storage::open_in_memory().expect("storage");
        let result = storage
            .get_index_metadata("nonexistent")
            .expect("get should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn upsert_updates_existing_metadata() {
        let storage = Storage::open_in_memory().expect("storage");

        let params_v1 = sample_build_params("idx");
        storage.record_index_build(&params_v1).expect("first build");

        let mut params_v2 = sample_build_params("idx");
        params_v2.record_count = 200;
        params_v2.embedder_revision = Some("v2.0".to_owned());
        params_v2.trigger = BuildTrigger::ContentChanged;
        storage
            .record_index_build(&params_v2)
            .expect("second build");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        assert_eq!(meta.record_count, 200);
        assert_eq!(meta.embedder_revision.as_deref(), Some("v2.0"));
    }

    #[test]
    fn build_history_is_appended() {
        let storage = Storage::open_in_memory().expect("storage");

        for i in 0..3 {
            let mut params = sample_build_params("idx");
            params.record_count = i64::from(i) * 50;
            params.trigger = if i == 0 {
                BuildTrigger::Initial
            } else {
                BuildTrigger::ContentChanged
            };
            storage.record_index_build(&params).expect("build");
        }

        let history = storage.get_build_history("idx", 10).expect("history query");
        assert_eq!(history.len(), 3);
        // Most recent first.
        assert!(history[0].built_at >= history[1].built_at);
    }

    #[test]
    fn build_history_respects_limit() {
        let storage = Storage::open_in_memory().expect("storage");

        for _ in 0..5 {
            let params = sample_build_params("idx");
            storage.record_index_build(&params).expect("build");
        }

        let history = storage.get_build_history("idx", 2).expect("history");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn build_history_zero_limit_returns_empty() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let history = storage.get_build_history("idx", 0).expect("history");
        assert!(history.is_empty());
    }

    #[test]
    fn prune_keeps_at_most_100_entries() {
        let storage = Storage::open_in_memory().expect("storage");

        for _ in 0..110 {
            let params = sample_build_params("idx");
            storage.record_index_build(&params).expect("build");
        }

        let history = storage.get_build_history("idx", 200).expect("history");
        assert_eq!(history.len(), 100);
    }

    #[test]
    fn record_verification_updates_timestamp() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let updated = storage.record_verification("idx").expect("verify");
        assert!(updated);

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        assert!(meta.last_verified_at.is_some());
    }

    #[test]
    fn record_verification_missing_index_returns_false() {
        let storage = Storage::open_in_memory().expect("storage");
        let updated = storage.record_verification("missing").expect("verify");
        assert!(!updated);
    }

    #[test]
    fn record_repair_increments_count() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        storage.record_repair("idx").expect("repair 1");
        storage.record_repair("idx").expect("repair 2");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        assert_eq!(meta.repair_count, 2);
        assert!(meta.last_repair_at.is_some());
    }

    #[test]
    fn staleness_never_built() {
        let storage = Storage::open_in_memory().expect("storage");
        let check = storage
            .check_index_staleness("missing", None)
            .expect("staleness check");

        assert!(check.is_stale);
        assert_eq!(check.reasons, vec![StalenessReason::NeverBuilt]);
        assert!(check.last_built_at.is_none());
    }

    #[test]
    fn staleness_fresh_index_is_not_stale() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let check = storage
            .check_index_staleness("idx", Some("v1.0"))
            .expect("staleness check");

        assert!(!check.is_stale);
        assert!(check.reasons.is_empty());
        assert_eq!(check.docs_modified, 0);
        assert_eq!(check.docs_added, 0);
    }

    #[test]
    fn staleness_detects_new_documents() {
        let storage = Storage::open_in_memory().expect("storage");

        // Build the index first.
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        let built_at = meta.built_at.unwrap();

        // Add a document after the build.
        insert_test_document(&storage, "new-doc", built_at + 1000, built_at + 1000);

        let check = storage
            .check_index_staleness("idx", Some("v1.0"))
            .expect("staleness check");

        assert!(check.is_stale);
        assert!(check.reasons.contains(&StalenessReason::NewDocuments));
        assert!(check.docs_added > 0);
    }

    #[test]
    fn staleness_detects_modified_documents() {
        let storage = Storage::open_in_memory().expect("storage");

        // Insert a document before the build.
        insert_test_document(&storage, "old-doc", 1_000_000, 1_000_000);

        // Build the index.
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let meta = storage
            .get_index_metadata("idx")
            .expect("get")
            .expect("exists");
        let built_at = meta.built_at.unwrap();

        // Modify the document (update updated_at to after build time).
        let mut updated_doc = DocumentRecord::new(
            "old-doc",
            "updated content",
            [0x99; 32],
            64,
            1_000_000,
            built_at + 1000,
        );
        updated_doc.source_path = None;
        storage.upsert_document(&updated_doc).expect("update doc");

        let check = storage
            .check_index_staleness("idx", Some("v1.0"))
            .expect("staleness check");

        assert!(check.is_stale);
        assert!(check.reasons.contains(&StalenessReason::ContentChanged));
        assert!(check.docs_modified > 0);
    }

    #[test]
    fn staleness_detects_embedder_change() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        let check = storage
            .check_index_staleness("idx", Some("v2.0"))
            .expect("staleness check");

        assert!(check.is_stale);
        assert!(check.reasons.contains(&StalenessReason::EmbedderChanged));
    }

    #[test]
    fn staleness_detects_missing_documents_from_build_snapshot() {
        let storage = Storage::open_in_memory().expect("storage");

        insert_test_document(&storage, "doc-1", 1_000_000, 1_000_000);

        let mut params = sample_build_params("idx");
        params.source_doc_count = 2;
        storage.record_index_build(&params).expect("build");

        let check = storage
            .check_index_staleness("idx", Some("v1.0"))
            .expect("staleness check");

        assert!(check.is_stale);
        assert!(check.reasons.contains(&StalenessReason::ContentChanged));
        assert!(check.docs_modified >= 1);
    }

    #[test]
    fn staleness_embedder_none_stored_marks_changed() {
        let storage = Storage::open_in_memory().expect("storage");
        let mut params = sample_build_params("idx");
        params.embedder_revision = None;
        storage.record_index_build(&params).expect("build");

        let check = storage
            .check_index_staleness("idx", Some("v1.0"))
            .expect("staleness check");

        assert!(check.is_stale);
        assert!(check.reasons.contains(&StalenessReason::EmbedderChanged));
    }

    #[test]
    fn delete_index_metadata_cascades_history() {
        let storage = Storage::open_in_memory().expect("storage");
        let params = sample_build_params("idx");
        storage.record_index_build(&params).expect("build");

        assert!(
            storage
                .delete_index_metadata("idx")
                .expect("delete should succeed")
        );
        assert!(
            storage
                .get_index_metadata("idx")
                .expect("get should succeed")
                .is_none()
        );
        assert!(
            storage
                .get_build_history("idx", 10)
                .expect("history should succeed")
                .is_empty()
        );
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let storage = Storage::open_in_memory().expect("storage");
        let deleted = storage
            .delete_index_metadata("nonexistent")
            .expect("delete");
        assert!(!deleted);
    }

    #[test]
    fn list_index_metadata_returns_all() {
        let storage = Storage::open_in_memory().expect("storage");

        let fast = sample_build_params("fast-index");
        let quality = {
            let mut p = sample_build_params("quality-index");
            p.embedder_id = "minilm-l6-v2".to_owned();
            p.dimension = 384;
            p
        };

        storage.record_index_build(&fast).expect("fast build");
        storage.record_index_build(&quality).expect("quality build");

        let all = storage.list_index_metadata().expect("list");
        assert_eq!(all.len(), 2);
        let names: Vec<&str> = all.iter().map(|m| m.index_name.as_str()).collect();
        assert!(names.contains(&"fast-index"));
        assert!(names.contains(&"quality-index"));
    }

    #[test]
    fn build_trigger_round_trip() {
        let triggers = [
            BuildTrigger::Initial,
            BuildTrigger::Scheduled,
            BuildTrigger::ContentChanged,
            BuildTrigger::Manual,
            BuildTrigger::ConfigChanged,
            BuildTrigger::Repair,
        ];
        for trigger in triggers {
            let s = trigger.as_str();
            let parsed = BuildTrigger::from_str(s);
            assert_eq!(parsed, Some(trigger), "round-trip failed for {s}");
        }
    }

    #[test]
    fn empty_index_name_is_rejected() {
        let storage = Storage::open_in_memory().expect("storage");
        let mut params = sample_build_params("");
        params.index_name = String::new();

        let err = storage
            .record_index_build(&params)
            .expect_err("empty name should fail");
        assert!(err.to_string().contains("must not be empty"));
    }
}
