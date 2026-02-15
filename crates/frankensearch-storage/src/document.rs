use std::collections::HashSet;
use std::fmt;
use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};

use crate::connection::Storage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrudErrorKind {
    NotFound,
    Conflict,
    Validation,
}

impl CrudErrorKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Validation => "validation",
        }
    }
}

#[derive(Debug)]
struct CrudError {
    kind: CrudErrorKind,
    message: String,
}

impl fmt::Display for CrudError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for CrudError {}

#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub source_path: Option<String>,
    pub content_preview: String,
    pub content_hash: [u8; 32],
    pub content_length: usize,
    pub created_at: i64,
    pub updated_at: i64,
    pub metadata: Option<serde_json::Value>,
}

impl DocumentRecord {
    #[must_use]
    pub fn new(
        doc_id: impl Into<String>,
        content_preview: impl Into<String>,
        content_hash: [u8; 32],
        content_length: usize,
        created_at: i64,
        updated_at: i64,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            source_path: None,
            content_preview: content_preview.into(),
            content_hash,
            content_length,
            created_at,
            updated_at,
            metadata: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingStatus {
    Pending,
    Embedded,
    Failed,
    Skipped,
}

impl EmbeddingStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Embedded => "embedded",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    pub(crate) fn from_str(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "embedded" => Some(Self::Embedded),
            "failed" => Some(Self::Failed),
            "skipped" => Some(Self::Skipped),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StatusCounts {
    pub pending: u64,
    pub embedded: u64,
    pub failed: u64,
    pub skipped: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BatchResult {
    pub inserted: u64,
    pub updated: u64,
    pub unchanged: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpsertOutcome {
    Inserted,
    Updated,
    Unchanged,
}

impl Storage {
    pub fn upsert_document(&self, doc: &DocumentRecord) -> SearchResult<bool> {
        let outcome = self.transaction(|conn| upsert_document_with_outcome(conn, doc))?;
        let changed = !matches!(outcome, UpsertOutcome::Unchanged);

        tracing::debug!(
            target: "frankensearch.storage",
            op = "upsert_document",
            doc_id = %doc.doc_id,
            changed,
            "document upsert completed"
        );

        Ok(changed)
    }

    pub fn get_document(&self, doc_id: &str) -> SearchResult<Option<DocumentRecord>> {
        ensure_non_empty(doc_id, "doc_id")?;

        let params = [SqliteValue::Text(doc_id.to_owned())];
        let rows = self
            .connection()
            .query_with_params(
                "SELECT doc_id, source_path, content_preview, content_hash, content_length, \
                    created_at, updated_at, metadata_json \
                 FROM documents WHERE doc_id = ?1 LIMIT 1;",
                &params,
            )
            .map_err(storage_error)?;

        let Some(row) = rows.first() else {
            tracing::debug!(
                target: "frankensearch.storage",
                op = "get_document",
                doc_id,
                found = false,
                "document fetch completed"
            );
            return Ok(None);
        };

        let metadata_json = row_optional_text(row, 7)?;
        let metadata = metadata_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(storage_error)?;

        let document = DocumentRecord {
            doc_id: row_text(row, 0, "documents.doc_id")?.to_owned(),
            source_path: row_optional_text(row, 1)?,
            content_preview: row_text(row, 2, "documents.content_preview")?.to_owned(),
            content_hash: row_blob_32(row, 3, "documents.content_hash")?,
            content_length: i64_to_usize(row_i64(row, 4, "documents.content_length")?)?,
            created_at: row_i64(row, 5, "documents.created_at")?,
            updated_at: row_i64(row, 6, "documents.updated_at")?,
            metadata,
        };

        tracing::debug!(
            target: "frankensearch.storage",
            op = "get_document",
            doc_id,
            found = true,
            "document fetch completed"
        );

        Ok(Some(document))
    }

    pub fn list_pending_embeddings(
        &self,
        embedder_id: &str,
        limit: usize,
    ) -> SearchResult<Vec<String>> {
        ensure_non_empty(embedder_id, "embedder_id")?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        let sql = format!(
            "SELECT d.doc_id, d.updated_at \
             FROM documents d \
             LEFT JOIN embedding_status e \
               ON d.doc_id = e.doc_id AND e.embedder_id = ?1 \
             WHERE e.status IS NULL OR e.status = 'pending' \
             ORDER BY d.updated_at DESC \
             LIMIT {limit};"
        );

        let params = [SqliteValue::Text(embedder_id.to_owned())];
        let rows = self
            .connection()
            .query_with_params(&sql, &params)
            .map_err(storage_error)?;
        let mut doc_ids = Vec::with_capacity(rows.len());
        for row in &rows {
            doc_ids.push(row_text(row, 0, "documents.doc_id")?.to_owned());
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "list_pending_embeddings",
            embedder_id,
            limit,
            count = doc_ids.len(),
            "pending embedding query completed"
        );

        Ok(doc_ids)
    }

    pub fn mark_embedded(&self, doc_id: &str, embedder_id: &str) -> SearchResult<()> {
        ensure_non_empty(doc_id, "doc_id")?;
        ensure_non_empty(embedder_id, "embedder_id")?;

        let finished_at = unix_timestamp_ms()?;
        self.transaction(|conn| {
            if !document_exists(conn, doc_id)? {
                return Err(not_found_error("documents", doc_id));
            }

            let params = [
                SqliteValue::Text(doc_id.to_owned()),
                SqliteValue::Text(embedder_id.to_owned()),
                SqliteValue::Text(EmbeddingStatus::Embedded.as_str().to_owned()),
                SqliteValue::Integer(finished_at),
            ];
            conn.execute_with_params(
                "INSERT INTO embedding_status \
                 (doc_id, embedder_id, embedder_revision, status, embedded_at, error_message, retry_count) \
                 VALUES (?1, ?2, NULL, ?3, ?4, NULL, 0) \
                 ON CONFLICT(doc_id, embedder_id) DO UPDATE SET \
                 status = excluded.status, \
                 embedded_at = excluded.embedded_at, \
                 error_message = NULL;",
                &params,
            )
            .map_err(storage_error)?;
            Ok(())
        })?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "mark_embedded",
            doc_id,
            embedder_id,
            "embedding status updated to embedded"
        );

        Ok(())
    }

    pub fn mark_failed(
        &self,
        doc_id: &str,
        embedder_id: &str,
        error_message: &str,
    ) -> SearchResult<()> {
        ensure_non_empty(doc_id, "doc_id")?;
        ensure_non_empty(embedder_id, "embedder_id")?;
        ensure_non_empty(error_message, "error_message")?;

        self.transaction(|conn| {
            if !document_exists(conn, doc_id)? {
                return Err(not_found_error("documents", doc_id));
            }

            let params = [
                SqliteValue::Text(doc_id.to_owned()),
                SqliteValue::Text(embedder_id.to_owned()),
                SqliteValue::Text(EmbeddingStatus::Failed.as_str().to_owned()),
                SqliteValue::Text(error_message.to_owned()),
            ];
            conn.execute_with_params(
                "INSERT INTO embedding_status \
                 (doc_id, embedder_id, embedder_revision, status, embedded_at, error_message, retry_count) \
                 VALUES (?1, ?2, NULL, ?3, NULL, ?4, 1) \
                 ON CONFLICT(doc_id, embedder_id) DO UPDATE SET \
                 status = excluded.status, \
                 embedded_at = NULL, \
                 error_message = excluded.error_message, \
                 retry_count = embedding_status.retry_count + 1;",
                &params,
            )
            .map_err(storage_error)?;
            Ok(())
        })?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "mark_failed",
            doc_id,
            embedder_id,
            "embedding status updated to failed"
        );

        Ok(())
    }

    pub fn count_by_status(&self, embedder_id: &str) -> SearchResult<StatusCounts> {
        ensure_non_empty(embedder_id, "embedder_id")?;

        let params = [SqliteValue::Text(embedder_id.to_owned())];
        let rows = self
            .connection()
            .query_with_params(
                "SELECT e.status, COUNT(*) \
                 FROM embedding_status e \
                 INNER JOIN documents d ON d.doc_id = e.doc_id \
                 WHERE e.embedder_id = ?1 \
                 GROUP BY status;",
                &params,
            )
            .map_err(storage_error)?;

        let total_docs = i64_to_u64(count_documents(self.connection())?)?;
        let mut counts = StatusCounts::default();
        for row in &rows {
            let status = row_text(row, 0, "embedding_status.status")?;
            let count = i64_to_u64(row_i64(row, 1, "embedding_status.count")?)?;
            match EmbeddingStatus::from_str(status) {
                Some(EmbeddingStatus::Pending) => {
                    counts.pending = counts.pending.saturating_add(count);
                }
                Some(EmbeddingStatus::Embedded) => {
                    counts.embedded = counts.embedded.saturating_add(count);
                }
                Some(EmbeddingStatus::Failed) => {
                    counts.failed = counts.failed.saturating_add(count);
                }
                Some(EmbeddingStatus::Skipped) => {
                    counts.skipped = counts.skipped.saturating_add(count);
                }
                None => {}
            }
        }

        let classified = counts
            .pending
            .saturating_add(counts.embedded)
            .saturating_add(counts.failed)
            .saturating_add(counts.skipped);
        if total_docs > classified {
            counts.pending = counts.pending.saturating_add(total_docs - classified);
        }

        tracing::debug!(
            target: "frankensearch.storage",
            op = "count_by_status",
            embedder_id,
            pending = counts.pending,
            embedded = counts.embedded,
            failed = counts.failed,
            skipped = counts.skipped,
            "embedding status count query completed"
        );

        Ok(counts)
    }

    pub fn delete_document(&self, doc_id: &str) -> SearchResult<bool> {
        ensure_non_empty(doc_id, "doc_id")?;

        let params = [SqliteValue::Text(doc_id.to_owned())];
        let deleted = self
            .connection()
            .execute_with_params("DELETE FROM documents WHERE doc_id = ?1;", &params)
            .map_err(storage_error)?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "delete_document",
            doc_id,
            deleted = deleted > 0,
            "document delete completed"
        );

        Ok(deleted > 0)
    }

    pub fn upsert_batch(&self, docs: &[DocumentRecord]) -> SearchResult<BatchResult> {
        if docs.is_empty() {
            return Ok(BatchResult::default());
        }

        let mut seen = HashSet::with_capacity(docs.len());
        for doc in docs {
            if !seen.insert(doc.doc_id.as_str()) {
                return Err(conflict_error(
                    "documents",
                    &doc.doc_id,
                    "duplicate doc_id in batch payload",
                ));
            }
        }

        let result = self.transaction(|conn| {
            let mut batch = BatchResult::default();
            for doc in docs {
                match upsert_document_with_outcome(conn, doc)? {
                    UpsertOutcome::Inserted => {
                        batch.inserted += 1;
                    }
                    UpsertOutcome::Updated => {
                        batch.updated += 1;
                    }
                    UpsertOutcome::Unchanged => {
                        batch.unchanged += 1;
                    }
                }
            }
            Ok(batch)
        })?;

        tracing::debug!(
            target: "frankensearch.storage",
            op = "upsert_batch",
            inserted = result.inserted,
            updated = result.updated,
            unchanged = result.unchanged,
            count = docs.len(),
            "batch upsert completed"
        );

        Ok(result)
    }
}

pub fn upsert_document(conn: &Connection, doc: &DocumentRecord) -> SearchResult<usize> {
    validate_document(doc)?;
    let content_length = usize_to_i64(doc.content_length, "content_length")?;
    let metadata_json = metadata_to_json(doc.metadata.as_ref())?;

    if document_exists(conn, &doc.doc_id)? {
        let params = [
            SqliteValue::Text(doc.doc_id.clone()),
            sqlite_text_opt(doc.source_path.as_deref()),
            SqliteValue::Text(doc.content_preview.clone()),
            SqliteValue::Blob(doc.content_hash.to_vec()),
            SqliteValue::Integer(content_length),
            SqliteValue::Integer(doc.updated_at),
            sqlite_text_opt(metadata_json.as_deref()),
        ];
        conn.execute_with_params(
            "UPDATE documents SET \
             source_path = ?2, \
             content_preview = ?3, \
             content_hash = ?4, \
             content_length = ?5, \
             updated_at = ?6, \
             metadata_json = ?7 \
             WHERE doc_id = ?1;",
            &params,
        )
        .map_err(storage_error)
    } else {
        let params = [
            SqliteValue::Text(doc.doc_id.clone()),
            sqlite_text_opt(doc.source_path.as_deref()),
            SqliteValue::Text(doc.content_preview.clone()),
            SqliteValue::Blob(doc.content_hash.to_vec()),
            SqliteValue::Integer(content_length),
            SqliteValue::Integer(doc.created_at),
            SqliteValue::Integer(doc.updated_at),
            sqlite_text_opt(metadata_json.as_deref()),
        ];

        conn.execute_with_params(
            "INSERT INTO documents \
             (doc_id, source_path, content_preview, content_hash, content_length, created_at, updated_at, metadata_json) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
            &params,
        )
        .map_err(storage_error)
    }
}

pub fn list_document_ids(conn: &Connection, limit: usize) -> SearchResult<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let sql = format!("SELECT doc_id FROM documents ORDER BY updated_at DESC LIMIT {limit};");
    let rows = conn.query(&sql).map_err(storage_error)?;
    let mut ids = Vec::with_capacity(rows.len());

    for row in &rows {
        ids.push(row_text(row, 0, "documents.doc_id")?.to_owned());
    }

    Ok(ids)
}

pub fn count_documents(conn: &Connection) -> SearchResult<i64> {
    let row = conn
        .query_row("SELECT COUNT(*) FROM documents;")
        .map_err(storage_error)?;
    row_i64(&row, 0, "documents.count")
}

fn upsert_document_with_outcome(
    conn: &Connection,
    doc: &DocumentRecord,
) -> SearchResult<UpsertOutcome> {
    validate_document(doc)?;

    let existing_hash = fetch_content_hash(conn, &doc.doc_id)?;
    if let Some(hash) = existing_hash {
        if hash == doc.content_hash {
            return Ok(UpsertOutcome::Unchanged);
        }
    }

    upsert_document(conn, doc)?;

    if existing_hash.is_some() {
        reset_embedding_status(conn, &doc.doc_id)?;
        Ok(UpsertOutcome::Updated)
    } else {
        Ok(UpsertOutcome::Inserted)
    }
}

fn validate_document(doc: &DocumentRecord) -> SearchResult<()> {
    ensure_non_empty(&doc.doc_id, "doc_id")?;
    if doc.content_preview.chars().count() > 400 {
        return Err(validation_error(
            "content_preview",
            "must be 400 characters or fewer",
        ));
    }
    if doc.updated_at < doc.created_at {
        return Err(validation_error(
            "updated_at",
            "must be greater than or equal to created_at",
        ));
    }
    if let Some(source_path) = doc.source_path.as_deref() {
        ensure_non_empty(source_path, "source_path")?;
    }
    let _ = usize_to_i64(doc.content_length, "content_length")?;
    Ok(())
}

fn metadata_to_json(metadata: Option<&serde_json::Value>) -> SearchResult<Option<String>> {
    metadata
        .map(serde_json::to_string)
        .transpose()
        .map_err(storage_error)
}

fn ensure_non_empty(value: &str, field: &'static str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(validation_error(field, "must not be empty"));
    }
    Ok(())
}

fn sqlite_text_opt(value: Option<&str>) -> SqliteValue {
    value.map_or(SqliteValue::Null, |v| SqliteValue::Text(v.to_owned()))
}

fn document_exists(conn: &Connection, doc_id: &str) -> SearchResult<bool> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    let rows = conn
        .query_with_params(
            "SELECT doc_id FROM documents WHERE doc_id = ?1 LIMIT 1;",
            &params,
        )
        .map_err(storage_error)?;
    Ok(!rows.is_empty())
}

fn fetch_content_hash(conn: &Connection, doc_id: &str) -> SearchResult<Option<[u8; 32]>> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    let rows = conn
        .query_with_params(
            "SELECT content_hash FROM documents WHERE doc_id = ?1 LIMIT 1;",
            &params,
        )
        .map_err(storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    row_blob_32(row, 0, "documents.content_hash").map(Some)
}

fn reset_embedding_status(conn: &Connection, doc_id: &str) -> SearchResult<()> {
    let params = [SqliteValue::Text(doc_id.to_owned())];
    conn.execute_with_params("DELETE FROM embedding_status WHERE doc_id = ?1;", &params)
        .map_err(storage_error)?;
    Ok(())
}

fn unix_timestamp_ms() -> SearchResult<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(storage_error)?;
    i64::try_from(duration.as_millis()).map_err(|error| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!(
            "unix timestamp milliseconds overflow i64: {error}"
        ))),
    })
}

fn usize_to_i64(value: usize, field: &'static str) -> SearchResult<i64> {
    i64::try_from(value).map_err(|error| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!(
            "value for {field} exceeds i64 range: {error}"
        ))),
    })
}

fn i64_to_usize(value: i64) -> SearchResult<usize> {
    usize::try_from(value).map_err(|error| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!(
            "value {value} cannot convert to usize: {error}"
        ))),
    })
}

fn i64_to_u64(value: i64) -> SearchResult<u64> {
    u64::try_from(value).map_err(|error| SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!(
            "value {value} cannot convert to u64: {error}"
        ))),
    })
}

fn row_text<'a>(row: &'a Row, index: usize, field: &str) -> SearchResult<&'a str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
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
                "unexpected optional text type: {:?}",
                other
            ))),
        }),
    }
}

fn row_blob_32(row: &Row, index: usize, field: &str) -> SearchResult<[u8; 32]> {
    let blob = match row.get(index) {
        Some(SqliteValue::Blob(blob)) => blob.as_slice(),
        Some(other) => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other(format!(
                    "unexpected type for {field}: {:?}",
                    other
                ))),
            });
        }
        None => {
            return Err(SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other(format!("missing column for {field}"))),
            });
        }
    };

    if blob.len() != 32 {
        return Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "expected 32-byte blob for {field}, found {}",
                blob.len()
            ))),
        });
    }

    let mut out = [0_u8; 32];
    out.copy_from_slice(blob);
    Ok(out)
}

fn row_i64(row: &Row, index: usize, field: &str) -> SearchResult<i64> {
    match row.get(index) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected type for {field}: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!("missing column for {field}"))),
        }),
    }
}

fn validation_error(field: &'static str, reason: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(CrudError {
            kind: CrudErrorKind::Validation,
            message: format!("{field}: {reason}"),
        }),
    }
}

fn conflict_error(entity: &'static str, id: &str, reason: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(CrudError {
            kind: CrudErrorKind::Conflict,
            message: format!("{entity}({id}): {reason}"),
        }),
    }
}

fn not_found_error(entity: &'static str, id: &str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(CrudError {
            kind: CrudErrorKind::NotFound,
            message: format!("{entity}({id})"),
        }),
    }
}

fn storage_error<E>(source: E) -> SearchError
where
    E: std::error::Error + Send + Sync + 'static,
{
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::{DocumentRecord, EmbeddingStatus, count_documents, list_document_ids};
    use crate::connection::{Storage, StorageConfig};

    fn hash_with(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn sample_document(doc_id: &str, hash_byte: u8) -> DocumentRecord {
        let mut doc = DocumentRecord::new(
            doc_id,
            "short preview",
            hash_with(hash_byte),
            128,
            1_739_499_200,
            1_739_499_200,
        );
        doc.source_path = Some("/tmp/source.txt".to_owned());
        doc.metadata = Some(serde_json::json!({"cluster": "alpha"}));
        doc
    }

    #[test]
    fn upsert_and_get_round_trip() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-1", 1);

        assert!(
            storage
                .upsert_document(&doc)
                .expect("upsert should succeed on new doc")
        );

        let fetched = storage
            .get_document("doc-1")
            .expect("get should succeed")
            .expect("document should exist");
        assert_eq!(fetched, doc);
        assert_eq!(
            list_document_ids(storage.connection(), 10).expect("list ids should succeed"),
            vec!["doc-1".to_owned()]
        );
        assert_eq!(
            count_documents(storage.connection()).expect("count should succeed"),
            1
        );
    }

    #[test]
    fn upsert_same_hash_is_noop() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-1", 1);

        assert!(
            storage
                .upsert_document(&doc)
                .expect("initial insert should succeed")
        );
        assert!(
            !storage
                .upsert_document(&doc)
                .expect("same-hash upsert should succeed")
        );
        assert_eq!(
            count_documents(storage.connection()).expect("count should succeed"),
            1
        );
    }

    #[test]
    fn changed_hash_resets_embedding_status_to_pending() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let first = sample_document("doc-1", 1);

        assert!(
            storage
                .upsert_document(&first)
                .expect("initial insert should succeed")
        );
        storage
            .mark_embedded("doc-1", "fast-tier")
            .expect("mark_embedded should succeed");

        let before = storage
            .count_by_status("fast-tier")
            .expect("status counts should succeed");
        assert_eq!(before.embedded, 1);
        assert_eq!(before.pending, 0);

        let mut updated = first;
        updated.content_hash = hash_with(2);
        updated.updated_at += 1;
        assert!(
            storage
                .upsert_document(&updated)
                .expect("changed-hash upsert should succeed")
        );

        let after = storage
            .count_by_status("fast-tier")
            .expect("status counts should succeed");
        assert_eq!(after.pending, 1);
        assert_eq!(after.embedded, 0);
    }

    #[test]
    fn pending_embeddings_include_missing_status_rows() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc_a = sample_document("doc-a", 1);
        let doc_b = sample_document("doc-b", 2);

        storage
            .upsert_document(&doc_a)
            .expect("doc-a insert should succeed");
        storage
            .upsert_document(&doc_b)
            .expect("doc-b insert should succeed");
        storage
            .mark_embedded("doc-a", "fast-tier")
            .expect("doc-a should be marked embedded");

        let pending = storage
            .list_pending_embeddings("fast-tier", 10)
            .expect("pending list should succeed");
        assert_eq!(pending, vec!["doc-b".to_owned()]);
    }

    #[test]
    fn mark_embedded_missing_doc_is_not_found_category() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let err = storage
            .mark_embedded("missing-doc", "fast-tier")
            .expect_err("missing doc should return not_found category");
        let msg = err.to_string();
        assert!(msg.contains("not_found"));
    }

    #[test]
    fn upsert_batch_rejects_duplicate_doc_ids() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc_a = sample_document("doc-a", 1);
        let doc_b = sample_document("doc-a", 2);

        let err = storage
            .upsert_batch(&[doc_a, doc_b])
            .expect_err("duplicate doc_ids should produce conflict category");
        assert!(err.to_string().contains("conflict"));
    }

    #[test]
    fn upsert_batch_reports_insert_update_and_unchanged_counts() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");

        let doc_a = sample_document("doc-a", 1);
        let doc_b = sample_document("doc-b", 2);
        let first = storage
            .upsert_batch(&[doc_a.clone(), doc_b.clone()])
            .expect("initial batch insert should succeed");
        assert_eq!(first.inserted, 2);
        assert_eq!(first.updated, 0);
        assert_eq!(first.unchanged, 0);

        let mut doc_b_updated = doc_b;
        doc_b_updated.content_hash = hash_with(3);
        doc_b_updated.updated_at += 1;
        let doc_c = sample_document("doc-c", 4);

        let second = storage
            .upsert_batch(&[doc_a, doc_b_updated, doc_c])
            .expect("mixed-outcome batch should succeed");
        assert_eq!(second.inserted, 1);
        assert_eq!(second.updated, 1);
        assert_eq!(second.unchanged, 1);
        assert_eq!(
            count_documents(storage.connection()).expect("count should succeed"),
            3
        );
    }

    #[test]
    fn upsert_batch_large_payload_is_atomic_and_complete() {
        const DOC_COUNT: usize = 500;

        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let docs: Vec<DocumentRecord> = (0..DOC_COUNT)
            .map(|idx| {
                let hash_seed = u8::try_from(idx % 251).expect("hash seed should fit into u8");
                sample_document(&format!("doc-large-{idx}"), hash_seed)
            })
            .collect();

        let first = storage
            .upsert_batch(&docs)
            .expect("large initial batch should succeed");
        let expected = u64::try_from(DOC_COUNT).expect("doc count should fit u64");
        assert_eq!(first.inserted, expected);
        assert_eq!(first.updated, 0);
        assert_eq!(first.unchanged, 0);
        assert_eq!(
            count_documents(storage.connection()).expect("count should succeed"),
            i64::try_from(DOC_COUNT).expect("doc count should fit i64")
        );

        let second = storage
            .upsert_batch(&docs)
            .expect("repeated large batch should succeed");
        assert_eq!(second.inserted, 0);
        assert_eq!(second.updated, 0);
        assert_eq!(second.unchanged, expected);
    }

    #[test]
    fn delete_document_cascades_embedding_status() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-1", 3);
        storage
            .upsert_document(&doc)
            .expect("insert should succeed");
        storage
            .mark_failed("doc-1", "fast-tier", "network timeout")
            .expect("mark_failed should succeed");

        assert!(
            storage
                .delete_document("doc-1")
                .expect("delete should succeed")
        );
        assert_eq!(
            count_documents(storage.connection()).expect("count should succeed"),
            0
        );

        let counts = storage
            .count_by_status("fast-tier")
            .expect("status counts should succeed");
        assert_eq!(counts.pending, 0);
        assert_eq!(counts.embedded, 0);
        assert_eq!(counts.failed, 0);
        assert_eq!(counts.skipped, 0);
    }

    #[test]
    fn count_by_status_reports_failed_and_pending() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        storage
            .upsert_document(&sample_document("doc-a", 1))
            .expect("doc-a insert should succeed");
        storage
            .upsert_document(&sample_document("doc-b", 2))
            .expect("doc-b insert should succeed");

        storage
            .mark_failed("doc-a", "fast-tier", "oops")
            .expect("doc-a mark_failed should succeed");

        let counts = storage
            .count_by_status("fast-tier")
            .expect("status counts should succeed");
        assert_eq!(counts.failed, 1);
        assert_eq!(counts.pending, 1);
    }

    #[test]
    fn embedding_status_from_str_parses_supported_values() {
        assert_eq!(
            EmbeddingStatus::from_str("pending"),
            Some(EmbeddingStatus::Pending)
        );
        assert_eq!(
            EmbeddingStatus::from_str("embedded"),
            Some(EmbeddingStatus::Embedded)
        );
        assert_eq!(EmbeddingStatus::from_str("bogus"), None);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let result = storage
            .get_document("does-not-exist")
            .expect("get should succeed even for missing doc");
        assert!(result.is_none());
    }

    #[test]
    fn content_preview_round_trips_correctly() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let long_preview = "A".repeat(400);
        let mut doc = DocumentRecord::new(
            "doc-preview",
            long_preview.clone(),
            hash_with(10),
            2048,
            1_739_500_000,
            1_739_500_000,
        );
        doc.source_path = Some("/test/preview.txt".to_owned());

        storage
            .upsert_document(&doc)
            .expect("upsert should succeed");

        let fetched = storage
            .get_document("doc-preview")
            .expect("get should succeed")
            .expect("document should exist");
        assert_eq!(fetched.content_preview, long_preview);
        assert_eq!(fetched.content_length, 2048);
    }

    #[test]
    fn metadata_json_round_trips_correctly() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let metadata = serde_json::json!({
            "tags": ["rust", "search"],
            "priority": 1,
            "nested": {"key": "value"},
            "flag": true
        });

        let mut doc = DocumentRecord::new(
            "doc-meta",
            "metadata test",
            hash_with(20),
            64,
            1_739_500_100,
            1_739_500_100,
        );
        doc.metadata = Some(metadata.clone());

        storage
            .upsert_document(&doc)
            .expect("upsert should succeed");

        let fetched = storage
            .get_document("doc-meta")
            .expect("get should succeed")
            .expect("document should exist");
        assert_eq!(fetched.metadata, Some(metadata));
    }

    #[test]
    fn unicode_doc_id_works_correctly() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let unicode_ids = [
            "\u{1F980}crab-doc",
            "\u{00E9}l\u{00E8}ve",
            "\u{4E16}\u{754C}",
            "path/with\u{2003}em-space",
        ];

        for id in &unicode_ids {
            let doc = DocumentRecord::new(
                *id,
                "unicode test content",
                hash_with(42),
                32,
                1_739_500_200,
                1_739_500_200,
            );
            storage
                .upsert_document(&doc)
                .unwrap_or_else(|e| panic!("upsert for {id:?} should succeed: {e}"));

            let fetched = storage
                .get_document(id)
                .unwrap_or_else(|e| panic!("get for {id:?} should succeed: {e}"))
                .unwrap_or_else(|| panic!("document {id:?} should exist"));
            assert_eq!(fetched.doc_id, *id);
        }

        assert_eq!(
            count_documents(storage.connection()).expect("count should succeed"),
            i64::try_from(unicode_ids.len()).expect("len fits i64")
        );
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let deleted = storage
            .delete_document("ghost-doc")
            .expect("delete of missing doc should not error");
        assert!(!deleted);
    }

    #[test]
    fn multi_tier_embedding_status_tracked_independently() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-multi", 5);
        storage
            .upsert_document(&doc)
            .expect("insert should succeed");

        storage
            .mark_embedded("doc-multi", "fast-tier")
            .expect("fast-tier mark_embedded should succeed");
        storage
            .mark_failed("doc-multi", "quality-tier", "onnx timeout")
            .expect("quality-tier mark_failed should succeed");

        let fast_counts = storage
            .count_by_status("fast-tier")
            .expect("fast-tier status counts should succeed");
        assert_eq!(fast_counts.embedded, 1);
        assert_eq!(fast_counts.pending, 0);

        let quality_counts = storage
            .count_by_status("quality-tier")
            .expect("quality-tier status counts should succeed");
        assert_eq!(quality_counts.failed, 1);
        assert_eq!(quality_counts.pending, 0);
    }

    #[test]
    fn mark_failed_succeeds_on_repeated_calls() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = sample_document("doc-retry", 6);
        storage
            .upsert_document(&doc)
            .expect("insert should succeed");

        for i in 0..3 {
            storage
                .mark_failed("doc-retry", "fast-tier", &format!("attempt {i}"))
                .expect("mark_failed should succeed on each call");
        }

        let counts = storage
            .count_by_status("fast-tier")
            .expect("status counts should succeed");
        assert_eq!(counts.pending, 0, "no pending after mark_failed");
        assert!(counts.failed > 0, "failed count should be positive");
        assert_eq!(counts.embedded, 0, "no embedded after mark_failed");
    }

    #[test]
    fn reopened_database_persists_all_data() {
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let db_path = dir.path().join("test.db");

        {
            let storage = Storage::open(StorageConfig {
                db_path: db_path.clone(),
                ..StorageConfig::default()
            })
            .expect("initial open should succeed");
            storage
                .upsert_document(&sample_document("persisted-1", 1))
                .expect("insert should succeed");
            storage
                .upsert_document(&sample_document("persisted-2", 2))
                .expect("insert should succeed");
            storage
                .mark_embedded("persisted-1", "fast-tier")
                .expect("mark_embedded should succeed");
        }

        {
            let storage = Storage::open(StorageConfig {
                db_path,
                ..StorageConfig::default()
            })
            .expect("reopen should succeed");
            let doc1 = storage
                .get_document("persisted-1")
                .expect("get should succeed")
                .expect("persisted-1 should survive reopen");
            assert_eq!(doc1.doc_id, "persisted-1");

            let doc2 = storage
                .get_document("persisted-2")
                .expect("get should succeed")
                .expect("persisted-2 should survive reopen");
            assert_eq!(doc2.doc_id, "persisted-2");

            let counts = storage
                .count_by_status("fast-tier")
                .expect("status counts should succeed");
            assert_eq!(counts.embedded, 1);
            assert_eq!(counts.pending, 1);

            assert_eq!(
                count_documents(storage.connection()).expect("count should succeed"),
                2
            );
        }
    }

    #[test]
    fn upsert_batch_empty_is_noop() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let result = storage
            .upsert_batch(&[])
            .expect("empty batch should succeed");
        assert_eq!(result.inserted, 0);
        assert_eq!(result.updated, 0);
        assert_eq!(result.unchanged, 0);
    }

    #[test]
    fn metadata_none_round_trips_as_none() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = DocumentRecord::new(
            "doc-no-meta",
            "no metadata",
            hash_with(30),
            16,
            1_739_500_300,
            1_739_500_300,
        );

        storage
            .upsert_document(&doc)
            .expect("upsert should succeed");

        let fetched = storage
            .get_document("doc-no-meta")
            .expect("get should succeed")
            .expect("document should exist");
        assert!(fetched.metadata.is_none());
        assert!(fetched.source_path.is_none());
    }

    // ─── bd-2z2j tests begin ───

    use super::*;

    #[test]
    fn crud_error_kind_as_str() {
        assert_eq!(CrudErrorKind::NotFound.as_str(), "not_found");
        assert_eq!(CrudErrorKind::Conflict.as_str(), "conflict");
        assert_eq!(CrudErrorKind::Validation.as_str(), "validation");
    }

    #[test]
    fn crud_error_kind_debug_clone_copy_eq() {
        let kind = CrudErrorKind::NotFound;
        let cloned = kind;
        assert_eq!(kind, cloned);
        let dbg = format!("{kind:?}");
        assert!(dbg.contains("NotFound"));
    }

    #[test]
    fn crud_error_kind_serde_roundtrip() {
        for kind in [
            CrudErrorKind::NotFound,
            CrudErrorKind::Conflict,
            CrudErrorKind::Validation,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: CrudErrorKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn crud_error_display() {
        let err = CrudError {
            kind: CrudErrorKind::NotFound,
            message: "doc-123 not found".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("not_found"));
        assert!(msg.contains("doc-123 not found"));
    }

    #[test]
    fn crud_error_is_std_error() {
        let err = CrudError {
            kind: CrudErrorKind::Conflict,
            message: "duplicate".into(),
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn document_record_new_defaults() {
        let doc = DocumentRecord::new("test-id", "preview", [0_u8; 32], 100, 1000, 2000);
        assert_eq!(doc.doc_id, "test-id");
        assert_eq!(doc.content_preview, "preview");
        assert_eq!(doc.content_hash, [0_u8; 32]);
        assert_eq!(doc.content_length, 100);
        assert_eq!(doc.created_at, 1000);
        assert_eq!(doc.updated_at, 2000);
        assert!(doc.source_path.is_none());
        assert!(doc.metadata.is_none());
    }

    #[test]
    fn document_record_serde_roundtrip() {
        let mut doc =
            DocumentRecord::new("serde-doc", "serde preview", hash_with(99), 512, 100, 200);
        doc.source_path = Some("/test/path".into());
        doc.metadata = Some(serde_json::json!({"key": "value"}));

        let json = serde_json::to_string(&doc).unwrap();
        let back: DocumentRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, doc);
    }

    #[test]
    fn embedding_status_as_str_all_variants() {
        assert_eq!(EmbeddingStatus::Pending.as_str(), "pending");
        assert_eq!(EmbeddingStatus::Embedded.as_str(), "embedded");
        assert_eq!(EmbeddingStatus::Failed.as_str(), "failed");
        assert_eq!(EmbeddingStatus::Skipped.as_str(), "skipped");
    }

    #[test]
    fn embedding_status_from_str_all_variants() {
        assert_eq!(
            EmbeddingStatus::from_str("pending"),
            Some(EmbeddingStatus::Pending)
        );
        assert_eq!(
            EmbeddingStatus::from_str("embedded"),
            Some(EmbeddingStatus::Embedded)
        );
        assert_eq!(
            EmbeddingStatus::from_str("failed"),
            Some(EmbeddingStatus::Failed)
        );
        assert_eq!(
            EmbeddingStatus::from_str("skipped"),
            Some(EmbeddingStatus::Skipped)
        );
        assert_eq!(EmbeddingStatus::from_str("unknown"), None);
        assert_eq!(EmbeddingStatus::from_str(""), None);
    }

    #[test]
    fn embedding_status_serde_roundtrip() {
        for status in [
            EmbeddingStatus::Pending,
            EmbeddingStatus::Embedded,
            EmbeddingStatus::Failed,
            EmbeddingStatus::Skipped,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: EmbeddingStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn status_counts_default() {
        let counts = StatusCounts::default();
        assert_eq!(counts.pending, 0);
        assert_eq!(counts.embedded, 0);
        assert_eq!(counts.failed, 0);
        assert_eq!(counts.skipped, 0);
    }

    #[test]
    fn status_counts_serde_roundtrip() {
        let counts = StatusCounts {
            pending: 10,
            embedded: 20,
            failed: 3,
            skipped: 1,
        };
        let json = serde_json::to_string(&counts).unwrap();
        let back: StatusCounts = serde_json::from_str(&json).unwrap();
        assert_eq!(back, counts);
    }

    #[test]
    fn batch_result_default() {
        let result = BatchResult::default();
        assert_eq!(result.inserted, 0);
        assert_eq!(result.updated, 0);
        assert_eq!(result.unchanged, 0);
    }

    #[test]
    fn batch_result_serde_roundtrip() {
        let result = BatchResult {
            inserted: 5,
            updated: 3,
            unchanged: 2,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: BatchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back, result);
    }

    #[test]
    fn validate_empty_doc_id_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = DocumentRecord::new("", "preview", hash_with(1), 10, 100, 100);
        let err = storage
            .upsert_document(&doc)
            .expect_err("empty doc_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("doc_id"));
    }

    #[test]
    fn validate_whitespace_doc_id_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = DocumentRecord::new("   ", "preview", hash_with(1), 10, 100, 100);
        let err = storage
            .upsert_document(&doc)
            .expect_err("whitespace doc_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("doc_id"));
    }

    #[test]
    fn validate_content_preview_too_long() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let long_preview = "A".repeat(401);
        let doc = DocumentRecord::new("doc-long", long_preview, hash_with(1), 10, 100, 100);
        let err = storage
            .upsert_document(&doc)
            .expect_err("long preview should fail");
        let msg = err.to_string();
        assert!(msg.contains("content_preview") || msg.contains("400"));
    }

    #[test]
    fn validate_updated_before_created() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let doc = DocumentRecord::new("doc-time", "preview", hash_with(1), 10, 200, 100);
        let err = storage
            .upsert_document(&doc)
            .expect_err("updated < created should fail");
        let msg = err.to_string();
        assert!(msg.contains("updated_at") || msg.contains("created_at"));
    }

    #[test]
    fn validate_empty_source_path_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let mut doc = DocumentRecord::new("doc-sp", "preview", hash_with(1), 10, 100, 100);
        doc.source_path = Some(String::new());
        let err = storage
            .upsert_document(&doc)
            .expect_err("empty source_path should fail");
        let msg = err.to_string();
        assert!(msg.contains("source_path"));
    }

    #[test]
    fn list_document_ids_limit_zero() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        storage
            .upsert_document(&sample_document("doc-1", 1))
            .expect("insert should succeed");
        let ids = list_document_ids(storage.connection(), 0).expect("limit 0 should succeed");
        assert!(ids.is_empty());
    }

    #[test]
    fn list_pending_embeddings_limit_zero() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        storage
            .upsert_document(&sample_document("doc-1", 1))
            .expect("insert should succeed");
        let pending = storage
            .list_pending_embeddings("fast-tier", 0)
            .expect("limit 0 should succeed");
        assert!(pending.is_empty());
    }

    #[test]
    fn list_pending_embeddings_empty_embedder_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let err = storage
            .list_pending_embeddings("", 10)
            .expect_err("empty embedder_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("embedder_id"));
    }

    #[test]
    fn get_document_empty_doc_id_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let err = storage
            .get_document("")
            .expect_err("empty doc_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("doc_id"));
    }

    #[test]
    fn mark_failed_empty_error_message_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        storage
            .upsert_document(&sample_document("doc-1", 1))
            .expect("insert should succeed");
        let err = storage
            .mark_failed("doc-1", "fast-tier", "")
            .expect_err("empty error_message should fail");
        let msg = err.to_string();
        assert!(msg.contains("error_message"));
    }

    #[test]
    fn delete_document_empty_doc_id_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let err = storage
            .delete_document("")
            .expect_err("empty doc_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("doc_id"));
    }

    #[test]
    fn count_by_status_empty_embedder_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        let err = storage
            .count_by_status("")
            .expect_err("empty embedder_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("embedder_id"));
    }

    #[test]
    fn mark_embedded_empty_embedder_rejected() {
        let storage = Storage::open_in_memory().expect("in-memory storage should open");
        storage
            .upsert_document(&sample_document("doc-1", 1))
            .expect("insert should succeed");
        let err = storage
            .mark_embedded("doc-1", "")
            .expect_err("empty embedder_id should fail");
        let msg = err.to_string();
        assert!(msg.contains("embedder_id"));
    }

    // ─── bd-2z2j tests end ───
}
