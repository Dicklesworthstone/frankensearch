//! Durable, revision-bound embedding input, kept separate from display metadata.
//!
//! Previews are not embedding input: they may contain only the first 400
//! characters. A content row keeps the complete canonical text and its digest
//! in the same database as the document catalog and persistent job queue.

use std::io;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::AsyncConnection;
use fsqlite_types::value::SqliteValue;

use crate::Storage;
use crate::connection::map_storage_error;
use crate::content_hash::ContentHasher;
use crate::document::{DocumentRecord, get_document_inner};

/// Governed by schema version 9, shared by fresh bootstrap and migration.
pub(crate) const CREATE_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS document_contents (\
    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,\
    content_hash BLOB NOT NULL,\
    canonical_text TEXT NOT NULL\
);";

impl Storage {
    /// Retain complete canonical text for an already catalogued document.
    ///
    /// The digest and Unicode character count must match the current catalog
    /// revision. This method does not enqueue work or change document metadata.
    /// The job runner uses the same inner operation inside its ingest transaction
    /// so content, catalog, and queue publication are atomic.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing document, a mismatched revision, or a
    /// database failure. A failed write preserves the previous content row.
    pub fn set_document_content(&self, doc_id: &str, canonical_text: &str) -> SearchResult<()> {
        self.transaction(|conn| {
            let document = get_document_inner(conn, doc_id)?
                .ok_or_else(|| content_error("cannot store content for a missing document"))?;
            upsert_document_content(conn, &document, canonical_text)
        })
    }

    /// Read verified complete text for the current document revision.
    ///
    /// Returns `None` when the document or content is absent, or the content row
    /// belongs to an older revision. A preview is never returned in its place.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt retained bytes or a database failure.
    pub fn get_document_content(&self, doc_id: &str) -> SearchResult<Option<String>> {
        self.transaction(|conn| {
            let Some(document) = get_document_inner(conn, doc_id)? else {
                return Ok(None);
            };
            read_document_content(conn, &document)
        })
    }
}

/// Write on the caller's transaction connection; never starts a nested transaction.
pub(crate) fn upsert_document_content(
    conn: &AsyncConnection,
    document: &DocumentRecord,
    canonical_text: &str,
) -> SearchResult<()> {
    verify_document_text(document, canonical_text)?;
    conn.execute_with_params_sync(
        "INSERT INTO document_contents (doc_id, content_hash, canonical_text) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(doc_id) DO UPDATE SET content_hash = excluded.content_hash, \
             canonical_text = excluded.canonical_text;",
        &[
            SqliteValue::Text(document.doc_id.clone().into()),
            SqliteValue::Blob(document.content_hash.to_vec().into()),
            SqliteValue::Text(canonical_text.to_owned().into()),
        ],
    )
    .map_err(map_storage_error)?;
    Ok(())
}

/// Read against an explicit catalog revision, including a runner's claimed one.
pub(crate) fn read_document_content(
    conn: &AsyncConnection,
    document: &DocumentRecord,
) -> SearchResult<Option<String>> {
    let rows = conn
        .query_with_params_sync(
            "SELECT content_hash, canonical_text FROM document_contents WHERE doc_id = ?1;",
            &[SqliteValue::Text(document.doc_id.clone().into())],
        )
        .map_err(map_storage_error)?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let hash = match row.get(0) {
        Some(SqliteValue::Blob(hash)) if hash.len() == 32 => hash,
        _ => {
            return Err(content_error(
                "retained document content has an invalid digest",
            ));
        }
    };
    if &hash[..] != document.content_hash.as_slice() {
        return Ok(None);
    }
    let text = match row.get(1) {
        Some(SqliteValue::Text(text)) => text.to_string(),
        _ => return Err(content_error("retained document content is not text")),
    };
    verify_document_text(document, &text)?;
    Ok(Some(text))
}

/// Validate actual input bytes, not only a caller-supplied digest or preview length.
pub(crate) fn verify_document_text(
    document: &DocumentRecord,
    canonical_text: &str,
) -> SearchResult<()> {
    if ContentHasher::hash(canonical_text) != document.content_hash
        || canonical_text.chars().count() != document.content_length
    {
        return Err(content_error(
            "embedding input does not match the catalogued document revision; re-ingest the document",
        ));
    }
    Ok(())
}

fn content_error(message: &'static str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "storage",
        // Do not include source contents in diagnostics or persisted job errors.
        source: Box::new(io::Error::new(io::ErrorKind::InvalidData, message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StorageConfig;

    fn document(doc_id: &str, text: &str) -> DocumentRecord {
        DocumentRecord {
            doc_id: doc_id.to_owned(),
            source_path: None,
            content_preview: text.chars().take(400).collect(),
            content_hash: ContentHasher::hash(text),
            content_length: text.chars().count(),
            created_at: 1,
            updated_at: 1,
            metadata: None,
        }
    }

    #[test]
    fn complete_unicode_content_survives_reopen_without_expanding_preview() {
        let directory = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            db_path: directory.path().join("content.sqlite3"),
            ..StorageConfig::default()
        };
        let text = "café 日本語 🚀 \0 tail ".repeat(150);
        {
            let storage = Storage::open(config.clone()).unwrap();
            storage.upsert_document(&document("durable", &text)).unwrap();
            storage.set_document_content("durable", &text).unwrap();
        }
        let storage = Storage::open(config).unwrap();
        assert_eq!(storage.get_document_content("durable").unwrap(), Some(text));
        assert_eq!(
            storage
                .get_document("durable")
                .unwrap()
                .unwrap()
                .content_preview
                .chars()
                .count(),
            400
        );
    }

    #[test]
    fn mismatched_write_preserves_current_content_and_diagnostics_omit_text() {
        let storage = Storage::open_in_memory().unwrap();
        storage
            .upsert_document(&document("doc", "original"))
            .unwrap();
        storage.set_document_content("doc", "original").unwrap();
        let error = storage
            .set_document_content("doc", "private replacement")
            .unwrap_err();
        assert!(!error.to_string().contains("private replacement"));
        assert_eq!(
            storage.get_document_content("doc").unwrap().as_deref(),
            Some("original")
        );
        assert!(storage.set_document_content("missing", "original").is_err());
    }

    #[test]
    fn missing_and_old_revision_content_are_not_substituted_for_current_text() {
        let storage = Storage::open_in_memory().unwrap();
        assert_eq!(storage.get_document_content("doc").unwrap(), None);
        storage.upsert_document(&document("doc", "old")).unwrap();
        assert_eq!(storage.get_document_content("doc").unwrap(), None);
        storage.set_document_content("doc", "old").unwrap();
        storage.upsert_document(&document("doc", "new")).unwrap();
        assert_eq!(storage.get_document_content("doc").unwrap(), None);
        storage.set_document_content("doc", "new").unwrap();
        assert_eq!(
            storage.get_document_content("doc").unwrap().as_deref(),
            Some("new")
        );
    }

    #[test]
    fn retained_content_corruption_fails_closed() {
        for mutation in [
            "UPDATE document_contents SET canonical_text = 'corrupt';",
            "UPDATE document_contents SET content_hash = X'01';",
            "UPDATE documents SET content_length = content_length + 1;",
        ] {
            let storage = Storage::open_in_memory().unwrap();
            storage
                .upsert_document(&document("doc", "original"))
                .unwrap();
            storage.set_document_content("doc", "original").unwrap();
            storage.connection().execute_sync(mutation).unwrap();
            assert!(storage.get_document_content("doc").is_err(), "{mutation}");
        }
    }

    #[test]
    fn content_and_catalog_changes_roll_back_together() {
        let storage = Storage::open_in_memory().unwrap();
        storage
            .upsert_document(&document("doc", "original"))
            .unwrap();
        storage.set_document_content("doc", "original").unwrap();
        let result: SearchResult<()> = storage.transaction(|conn| {
            let updated = document("doc", "replacement");
            crate::document::upsert_document(conn, &updated)?;
            upsert_document_content(conn, &updated, "replacement")?;
            Err(content_error("injected later failure"))
        });
        assert!(result.is_err());
        assert_eq!(
            storage.get_document_content("doc").unwrap().as_deref(),
            Some("original")
        );
        assert_eq!(
            storage.get_document("doc").unwrap().unwrap().content_hash,
            ContentHasher::hash("original")
        );
    }

    #[test]
    fn deleting_document_cascades_to_retained_content() {
        let storage = Storage::open_in_memory().unwrap();
        storage
            .upsert_document(&document("doc", "original"))
            .unwrap();
        storage.set_document_content("doc", "original").unwrap();
        storage
            .transaction(|conn| {
                conn.execute_sync("DELETE FROM documents WHERE doc_id = 'doc';")
                    .map_err(map_storage_error)?;
                Ok(())
            })
            .unwrap();
        assert!(
            storage
                .connection()
                .query_sync("SELECT doc_id FROM document_contents;")
                .unwrap()
                .is_empty()
        );
        storage
            .upsert_document(&document("doc", "original"))
            .unwrap();
        assert_eq!(storage.get_document_content("doc").unwrap(), None);
    }
}
