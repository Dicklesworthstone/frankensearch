use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::io;

use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{Connection, Row};
use fsqlite_types::value::SqliteValue;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::connection::Storage;
use crate::document::EmbeddingStatus;

#[derive(Debug, Clone, Copy, Default)]
pub struct ContentHasher;

impl ContentHasher {
    #[must_use]
    pub fn hash(canonical_text: &str) -> [u8; 32] {
        let digest = Sha256::digest(canonical_text.as_bytes());
        digest.into()
    }

    #[must_use]
    pub fn hash_hex(canonical_text: &str) -> String {
        let digest = Self::hash(canonical_text);
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            write!(&mut out, "{byte:02x}").expect("write to String is infallible");
        }
        out
    }

    #[must_use]
    pub fn matches(a: &[u8; 32], b: &[u8; 32]) -> bool {
        a == b
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentHashRecord {
    pub content_hash: String,
    pub first_doc_id: String,
    pub seen_count: i64,
    pub first_seen_at: i64,
    pub last_seen_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeduplicationDecision {
    Skip {
        doc_id: String,
        reason: &'static str,
    },
    New {
        doc_id: String,
    },
    Changed {
        doc_id: String,
        old_hash: [u8; 32],
        new_hash: [u8; 32],
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DedupRow {
    content_hash: [u8; 32],
    status: Option<EmbeddingStatus>,
}

#[must_use]
pub fn sha256_hex(content: &str) -> String {
    ContentHasher::hash_hex(content)
}

pub fn record_content_hash(
    conn: &Connection,
    content_hash: &str,
    doc_id: &str,
    seen_at: i64,
) -> SearchResult<usize> {
    let sql = "INSERT INTO content_hashes \
        (content_hash, first_doc_id, seen_count, first_seen_at, last_seen_at) \
        VALUES (?1, ?2, 1, ?3, ?4) \
        ON CONFLICT(content_hash) DO UPDATE SET \
            seen_count = content_hashes.seen_count + 1, \
            last_seen_at = excluded.last_seen_at;";

    let params = [
        SqliteValue::Text(content_hash.to_owned()),
        SqliteValue::Text(doc_id.to_owned()),
        SqliteValue::Integer(seen_at),
        SqliteValue::Integer(seen_at),
    ];

    conn.execute_with_params(sql, &params)
        .map_err(storage_error)
}

pub fn lookup_content_hash(
    conn: &Connection,
    content_hash: &str,
) -> SearchResult<Option<ContentHashRecord>> {
    let params = [SqliteValue::Text(content_hash.to_owned())];
    let rows = conn
        .query_with_params(
            "SELECT content_hash, first_doc_id, seen_count, first_seen_at, last_seen_at \
             FROM content_hashes WHERE content_hash = ?1;",
            &params,
        )
        .map_err(storage_error)?;

    let Some(row) = rows.first() else {
        return Ok(None);
    };

    Ok(Some(ContentHashRecord {
        content_hash: row_text(row, 0, "content_hashes.content_hash")?.to_owned(),
        first_doc_id: row_text(row, 1, "content_hashes.first_doc_id")?.to_owned(),
        seen_count: row_i64(row, 2, "content_hashes.seen_count")?,
        first_seen_at: row_i64(row, 3, "content_hashes.first_seen_at")?,
        last_seen_at: row_i64(row, 4, "content_hashes.last_seen_at")?,
    }))
}

impl Storage {
    pub fn check_dedup(
        &self,
        doc_id: &str,
        new_hash: &[u8; 32],
        embedder_id: &str,
    ) -> SearchResult<DeduplicationDecision> {
        ensure_non_empty(doc_id, "doc_id")?;
        ensure_non_empty(embedder_id, "embedder_id")?;

        let items = [(doc_id.to_owned(), *new_hash)];
        self.check_dedup_batch(&items, embedder_id)?
            .into_iter()
            .next()
            .ok_or_else(|| SearchError::SubsystemError {
                subsystem: "storage",
                source: Box::new(io::Error::other("check_dedup expected one decision result")),
            })
    }

    pub fn check_dedup_batch(
        &self,
        items: &[(String, [u8; 32])],
        embedder_id: &str,
    ) -> SearchResult<Vec<DeduplicationDecision>> {
        ensure_non_empty(embedder_id, "embedder_id")?;
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let mut seen_doc_ids = HashSet::with_capacity(items.len());
        for (doc_id, _) in items {
            ensure_non_empty(doc_id, "doc_id")?;
            if !seen_doc_ids.insert(doc_id.as_str()) {
                return Err(validation_error(
                    "doc_id",
                    format!("duplicate doc_id in dedup payload: {doc_id}"),
                ));
            }
        }

        self.transaction(|conn| {
            let existing = fetch_existing_dedup_rows(conn, items, embedder_id)?;
            let mut decisions = Vec::with_capacity(items.len());
            let mut skipped_docs = 0_u64;
            let mut new_docs = 0_u64;
            let mut changed_docs = 0_u64;

            for (doc_id, new_hash) in items {
                let decision = build_dedup_decision(doc_id, *new_hash, existing.get(doc_id));
                match &decision {
                    DeduplicationDecision::Skip { .. } => {
                        skipped_docs += 1;
                    }
                    DeduplicationDecision::New { .. } => {
                        new_docs += 1;
                    }
                    DeduplicationDecision::Changed { .. } => {
                        changed_docs += 1;
                    }
                }
                decisions.push(decision);
            }

            tracing::debug!(
                target: "frankensearch.storage",
                op = "check_dedup_batch",
                embedder_id,
                requested = items.len(),
                skipped_docs,
                new_docs,
                changed_docs,
                "dedup decisions computed"
            );

            Ok(decisions)
        })
    }
}

fn fetch_existing_dedup_rows(
    conn: &Connection,
    items: &[(String, [u8; 32])],
    embedder_id: &str,
) -> SearchResult<HashMap<String, DedupRow>> {
    let mut sql = String::from(
        "SELECT d.doc_id, d.content_hash, e.status \
         FROM documents d \
         LEFT JOIN embedding_status e \
           ON d.doc_id = e.doc_id AND e.embedder_id = ?1 \
         WHERE d.doc_id IN (",
    );

    for index in 0..items.len() {
        if index > 0 {
            sql.push_str(", ");
        }
        let _ = write!(&mut sql, "?{}", index + 2);
    }
    sql.push_str(");");

    let mut params = Vec::with_capacity(items.len() + 1);
    params.push(SqliteValue::Text(embedder_id.to_owned()));
    for (doc_id, _) in items {
        ensure_non_empty(doc_id, "doc_id")?;
        params.push(SqliteValue::Text(doc_id.clone()));
    }

    let rows = conn
        .query_with_params(&sql, &params)
        .map_err(storage_error)?;
    let mut existing = HashMap::with_capacity(rows.len());

    for row in &rows {
        let doc_id = row_text(row, 0, "documents.doc_id")?.to_owned();
        let content_hash = row_blob_32(row, 1, "documents.content_hash")?;
        let raw_status = row_optional_text(row, 2)?;
        let status = raw_status.as_deref().and_then(EmbeddingStatus::from_str);

        if raw_status.is_some() && status.is_none() {
            tracing::warn!(
                target: "frankensearch.storage",
                doc_id,
                raw_status = ?raw_status,
                "unknown embedding status while computing dedup"
            );
        }

        existing.insert(
            doc_id,
            DedupRow {
                content_hash,
                status,
            },
        );
    }

    Ok(existing)
}

fn build_dedup_decision(
    doc_id: &str,
    new_hash: [u8; 32],
    existing: Option<&DedupRow>,
) -> DeduplicationDecision {
    let Some(existing) = existing else {
        return DeduplicationDecision::New {
            doc_id: doc_id.to_owned(),
        };
    };

    if existing.content_hash != new_hash {
        return DeduplicationDecision::Changed {
            doc_id: doc_id.to_owned(),
            old_hash: existing.content_hash,
            new_hash,
        };
    }

    match existing.status {
        Some(EmbeddingStatus::Embedded) => DeduplicationDecision::Skip {
            doc_id: doc_id.to_owned(),
            reason: "unchanged_content_already_embedded",
        },
        Some(EmbeddingStatus::Pending) => DeduplicationDecision::Skip {
            doc_id: doc_id.to_owned(),
            reason: "unchanged_content_already_pending",
        },
        Some(EmbeddingStatus::Skipped) => DeduplicationDecision::Skip {
            doc_id: doc_id.to_owned(),
            reason: "unchanged_content_previously_skipped",
        },
        Some(EmbeddingStatus::Failed) | None => DeduplicationDecision::New {
            doc_id: doc_id.to_owned(),
        },
    }
}

fn ensure_non_empty(value: &str, field: &'static str) -> SearchResult<()> {
    if value.trim().is_empty() {
        return Err(validation_error(field, "must not be empty"));
    }
    Ok(())
}

fn validation_error(field: &'static str, reason: impl AsRef<str>) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::other(format!("{field}: {}", reason.as_ref()))),
    }
}

fn row_optional_text(row: &Row, index: usize) -> SearchResult<Option<String>> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(Some(value.clone())),
        Some(SqliteValue::Null) => Ok(None),
        Some(other) => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "unexpected optional text type: {:?}",
                other
            ))),
        }),
        None => Err(SearchError::SubsystemError {
            subsystem: "storage",
            source: Box::new(io::Error::other(format!(
                "missing optional text column at index {index}"
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
    use super::{ContentHasher, DeduplicationDecision, sha256_hex};
    use crate::Storage;
    use crate::document::DocumentRecord;
    use crate::document::EmbeddingStatus;
    use fsqlite_types::value::SqliteValue;

    fn hash_with(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn sample_document(doc_id: &str, hash_byte: u8) -> DocumentRecord {
        DocumentRecord::new(
            doc_id,
            "short preview",
            hash_with(hash_byte),
            128,
            1_739_499_200,
            1_739_499_200,
        )
    }

    fn upsert_status_row(
        storage: &Storage,
        doc_id: &str,
        embedder_id: &str,
        status: EmbeddingStatus,
    ) {
        let params = [
            SqliteValue::Text(doc_id.to_owned()),
            SqliteValue::Text(embedder_id.to_owned()),
            SqliteValue::Text(status.as_str().to_owned()),
        ];
        storage
            .connection()
            .execute_with_params(
                "INSERT INTO embedding_status \
                 (doc_id, embedder_id, embedder_revision, status, embedded_at, error_message, retry_count) \
                 VALUES (?1, ?2, NULL, ?3, NULL, NULL, 0) \
                 ON CONFLICT(doc_id, embedder_id) DO UPDATE SET status = excluded.status;",
                &params,
            )
            .expect("embedding_status upsert should succeed");
    }

    #[test]
    fn content_hasher_is_stable() {
        let first = ContentHasher::hash("hello world");
        let second = ContentHasher::hash("hello world");
        let different = ContentHasher::hash("hello world!");

        assert_eq!(first, second);
        assert_ne!(first, different);
        assert!(ContentHasher::matches(&first, &second));
        assert!(!ContentHasher::matches(&first, &different));
        assert_eq!(
            sha256_hex("hello world"),
            ContentHasher::hash_hex("hello world")
        );
    }

    #[test]
    fn check_dedup_new_document() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let hash = ContentHasher::hash("new document");

        let decision = storage
            .check_dedup("doc-new", &hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::New {
                doc_id: "doc-new".to_owned()
            }
        );
    }

    #[test]
    fn check_dedup_skip_when_unchanged_and_embedded() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let doc = sample_document("doc-1", 1);

        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");
        storage
            .mark_embedded("doc-1", "fast-tier")
            .expect("mark_embedded should succeed");

        let decision = storage
            .check_dedup("doc-1", &doc.content_hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::Skip {
                doc_id: "doc-1".to_owned(),
                reason: "unchanged_content_already_embedded",
            }
        );
    }

    #[test]
    fn check_dedup_new_when_unchanged_but_not_embedded() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let doc = sample_document("doc-2", 2);
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");

        let decision = storage
            .check_dedup("doc-2", &doc.content_hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::New {
                doc_id: "doc-2".to_owned()
            }
        );
    }

    #[test]
    fn check_dedup_skip_when_unchanged_and_pending() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let doc = sample_document("doc-pending", 7);
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");
        upsert_status_row(
            &storage,
            "doc-pending",
            "fast-tier",
            EmbeddingStatus::Pending,
        );

        let decision = storage
            .check_dedup("doc-pending", &doc.content_hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::Skip {
                doc_id: "doc-pending".to_owned(),
                reason: "unchanged_content_already_pending",
            }
        );
    }

    #[test]
    fn check_dedup_skip_when_unchanged_and_previously_skipped() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let doc = sample_document("doc-skipped", 8);
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");
        upsert_status_row(
            &storage,
            "doc-skipped",
            "fast-tier",
            EmbeddingStatus::Skipped,
        );

        let decision = storage
            .check_dedup("doc-skipped", &doc.content_hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::Skip {
                doc_id: "doc-skipped".to_owned(),
                reason: "unchanged_content_previously_skipped",
            }
        );
    }

    #[test]
    fn check_dedup_new_when_unchanged_and_failed() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let doc = sample_document("doc-failed", 9);
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");
        storage
            .mark_failed("doc-failed", "fast-tier", "transient failure")
            .expect("mark_failed should succeed");

        let decision = storage
            .check_dedup("doc-failed", &doc.content_hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::New {
                doc_id: "doc-failed".to_owned()
            }
        );
    }

    #[test]
    fn check_dedup_changed_when_hash_differs() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let doc = sample_document("doc-3", 3);
        storage
            .upsert_document(&doc)
            .expect("document insert should succeed");

        let new_hash = hash_with(4);
        let decision = storage
            .check_dedup("doc-3", &new_hash, "fast-tier")
            .expect("check_dedup should succeed");

        assert_eq!(
            decision,
            DeduplicationDecision::Changed {
                doc_id: "doc-3".to_owned(),
                old_hash: doc.content_hash,
                new_hash,
            }
        );
    }

    #[test]
    fn check_dedup_batch_preserves_input_order() {
        let storage = Storage::open_in_memory().expect("storage should open");
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
            .expect("doc-a should be embedded");

        let decisions = storage
            .check_dedup_batch(
                &[
                    ("doc-missing".to_owned(), hash_with(9)),
                    ("doc-a".to_owned(), doc_a.content_hash),
                    ("doc-b".to_owned(), hash_with(7)),
                ],
                "fast-tier",
            )
            .expect("batch dedup should succeed");

        assert_eq!(
            decisions[0],
            DeduplicationDecision::New {
                doc_id: "doc-missing".to_owned()
            }
        );
        assert_eq!(
            decisions[1],
            DeduplicationDecision::Skip {
                doc_id: "doc-a".to_owned(),
                reason: "unchanged_content_already_embedded",
            }
        );
        assert_eq!(
            decisions[2],
            DeduplicationDecision::Changed {
                doc_id: "doc-b".to_owned(),
                old_hash: doc_b.content_hash,
                new_hash: hash_with(7),
            }
        );
    }

    #[test]
    fn content_hasher_hex_is_64_chars() {
        let hex = ContentHasher::hash_hex("anything");
        assert_eq!(hex.len(), 64);
        assert!(
            hex.chars().all(|c| c.is_ascii_hexdigit()),
            "hex output should contain only hex digits"
        );
    }

    #[test]
    fn content_hasher_empty_string() {
        let hash = ContentHasher::hash("");
        let hex = ContentHasher::hash_hex("");
        assert_eq!(hex.len(), 64);
        // Empty string always produces the same well-known SHA-256 hash.
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert!(ContentHasher::matches(&hash, &hash));
    }

    #[test]
    fn check_dedup_rejects_empty_doc_id() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let hash = ContentHasher::hash("some content");
        let err = storage
            .check_dedup("", &hash, "fast-tier")
            .expect_err("empty doc_id should be rejected");
        assert!(err.to_string().contains("doc_id"));
    }

    #[test]
    fn check_dedup_rejects_empty_embedder_id() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let hash = ContentHasher::hash("some content");
        let err = storage
            .check_dedup("doc-1", &hash, "")
            .expect_err("empty embedder_id should be rejected");
        assert!(err.to_string().contains("embedder_id"));
    }

    #[test]
    fn check_dedup_batch_empty_returns_empty() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let decisions = storage
            .check_dedup_batch(&[], "fast-tier")
            .expect("empty batch should succeed");
        assert!(decisions.is_empty());
    }

    #[test]
    fn check_dedup_batch_rejects_duplicate_doc_ids() {
        let storage = Storage::open_in_memory().expect("storage should open");
        let duplicate = hash_with(3);

        let err = storage
            .check_dedup_batch(
                &[
                    ("doc-dup".to_owned(), duplicate),
                    ("doc-dup".to_owned(), duplicate),
                ],
                "fast-tier",
            )
            .expect_err("duplicate doc_id should be rejected");

        let msg = err.to_string();
        assert!(msg.contains("doc_id"));
        assert!(msg.contains("duplicate"));
    }
}
