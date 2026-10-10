//! Atomic publication of identity-bound embeddings in the catalog database.
//!
//! This is a durable alternative to an independent `EmbeddingVectorSink` write.
//! One transaction admits the exact queue attempt and document revision, stores
//! the vector and producer identity, and completes the job and catalog status.
//! It does not publish FSVI/ANN files or make external sinks transactional.

use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch_core::{EmbeddingIdentityBundleV1, IdentityBoundEmbedding, SearchError, SearchResult};
use fsqlite::{AsyncConnection, Row};
use fsqlite_types::value::SqliteValue;
use sha2::{Digest, Sha256};

use crate::connection::map_storage_error;
use crate::document::{DocumentRecord, get_document_inner, mark_embedded_inner};
use crate::{ClaimOutcome, ClaimedJob, Storage};

/// Shared by fresh bootstrap and the additive v10 migration.
pub(crate) const CREATE_TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS published_embeddings (\
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,\
    embedder_id TEXT NOT NULL,\
    content_hash BLOB NOT NULL,\
    identity_json TEXT NOT NULL,\
    vector_le BLOB NOT NULL,\
    record_hash BLOB NOT NULL,\
    source_job_id INTEGER NOT NULL,\
    claim_epoch INTEGER NOT NULL,\
    published_at_ms INTEGER NOT NULL,\
    record_version INTEGER NOT NULL DEFAULT 1,\
    PRIMARY KEY(doc_id, embedder_id)\
);";

const OWNED_ATTEMPT_SQL: &str = "SELECT job_id FROM embedding_jobs \
    WHERE job_id = ?1 AND status = 'processing' AND claim_epoch = ?2 \
      AND worker_id = ?3 AND doc_id = ?4 AND embedder_id = ?5 \
      AND (content_hash = ?6 OR (content_hash IS NULL AND ?6 IS NULL));";
const RETIRE_ATTEMPT_SQL: &str = "DELETE FROM embedding_jobs \
    WHERE job_id = ?1 AND status = 'processing' AND claim_epoch = ?2 \
      AND worker_id = ?3 AND doc_id = ?4 AND embedder_id = ?5 \
      AND (content_hash = ?6 OR (content_hash IS NULL AND ?6 IS NULL));";
const WRITE_EMBEDDING_SQL: &str = "INSERT INTO published_embeddings \
    (doc_id, embedder_id, content_hash, identity_json, vector_le, record_hash, \
     source_job_id, claim_epoch, published_at_ms, record_version) \
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1) \
    ON CONFLICT(doc_id, embedder_id) DO UPDATE SET \
      content_hash = excluded.content_hash, identity_json = excluded.identity_json, \
      vector_le = excluded.vector_le, record_hash = excluded.record_hash, \
      source_job_id = excluded.source_job_id, claim_epoch = excluded.claim_epoch, \
      published_at_ms = excluded.published_at_ms, record_version = excluded.record_version;";
const DELETE_HISTORY_SQL: &str = "DELETE FROM embedding_jobs \
    WHERE doc_id = ?1 AND embedder_id = ?2 AND status = 'completed';";
const COMPLETE_ATTEMPT_SQL: &str = "UPDATE embedding_jobs \
    SET status = 'completed', completed_at = ?7, worker_id = NULL, error_message = NULL \
    WHERE job_id = ?1 AND status = 'processing' AND claim_epoch = ?2 \
      AND worker_id = ?3 AND doc_id = ?4 AND embedder_id = ?5 \
      AND (content_hash = ?6 OR (content_hash IS NULL AND ?6 IS NULL));";
const READ_EMBEDDING_SQL: &str = "SELECT v.content_hash, v.identity_json, v.vector_le, \
    v.record_hash, v.source_job_id, v.claim_epoch, v.published_at_ms, v.record_version, d.content_hash \
    FROM published_embeddings v \
    JOIN documents d ON d.doc_id = v.doc_id \
    JOIN embedding_status e ON e.doc_id = v.doc_id AND e.embedder_id = v.embedder_id \
    WHERE v.doc_id = ?1 AND v.embedder_id = ?2 AND e.status = 'embedded';";

/// Verified vector data for one catalog revision and complete producer identity.
///
/// The values retain their original f32 bits; the database representation is
/// explicitly little-endian. The identity describes the producer's in-memory
/// output, not an invented FSVI generation or a new embedding space.
#[derive(Debug, Clone, PartialEq)]
pub struct PublishedEmbedding {
    pub doc_id: String,
    pub embedder_id: String,
    pub content_hash: [u8; 32],
    pub source_job_id: i64,
    pub claim_epoch: i64,
    pub published_at_ms: i64,
    pub embedding: IdentityBoundEmbedding,
}

impl Storage {
    /// Publish a prepared response and complete its exact attempt atomically.
    ///
    /// `expected_identity` must be captured from the intended producer before
    /// inference, not copied from an untrusted response. The document hash is
    /// the revision whose actual input was embedded, including for legacy jobs
    /// whose claim hash is unknown. This method replaces `queue.complete` for
    /// database-backed output: do not complete the same claim a second time.
    ///
    /// A lost claim has no side effects. A superseded document retires only the
    /// owned obsolete attempt, preserving replacement work and existing vectors.
    /// Any publication/catalog/queue error rolls back all publication mutations.
    /// No callback, inference, external file operation, or await runs in the
    /// transaction. Instance-local `JobQueueMetrics` are not updated by this
    /// Storage API; its durable queue state and storage transaction metrics are.
    pub fn publish_claimed_embedding(
        &self,
        claim: &ClaimedJob,
        expected_document_hash: &[u8; 32],
        expected_identity: &EmbeddingIdentityBundleV1,
        response: &IdentityBoundEmbedding,
    ) -> SearchResult<ClaimOutcome<()>> {
        self.immediate_transaction(|conn| {
            match owned_document(conn, claim, Some(expected_document_hash))? {
                ClaimOutcome::Applied(_) => {}
                ClaimOutcome::LostClaim => return Ok(ClaimOutcome::LostClaim),
                ClaimOutcome::Superseded => return Ok(ClaimOutcome::Superseded),
            }
            // Admit ownership first: even a malformed late response cannot
            // fail a successor or hide the fact that this attempt lost its claim.
            validate_response(expected_identity, response)?;
            let identity_json = serde_json::to_string(expected_identity).map_err(map_storage_error)?;
            let vector_le = encode_vector(&response.values)?;
            let now_ms = unix_ms()?;
            let record_hash = publication_hash(
                &claim.doc_id,
                &claim.embedder_id,
                expected_document_hash,
                &identity_json,
                &vector_le,
                [claim.job_id, claim.claim_epoch, now_ms],
            )?;
            conn.execute_with_params_sync(
                WRITE_EMBEDDING_SQL,
                &[
                    text_value(&claim.doc_id),
                    text_value(&claim.embedder_id),
                    SqliteValue::Blob(expected_document_hash.to_vec().into()),
                    text_value(&identity_json),
                    SqliteValue::Blob(vector_le.into()),
                    SqliteValue::Blob(record_hash.to_vec().into()),
                    SqliteValue::Integer(claim.job_id),
                    SqliteValue::Integer(claim.claim_epoch),
                    SqliteValue::Integer(now_ms),
                ],
            )
            .map_err(map_storage_error)?;
            conn.execute_with_params_sync(
                DELETE_HISTORY_SQL,
                &[text_value(&claim.doc_id), text_value(&claim.embedder_id)],
            )
            .map_err(map_storage_error)?;
            let mut params = claim_params(claim).to_vec();
            params.push(SqliteValue::Integer(now_ms));
            let updated = conn
                .execute_with_params_sync(COMPLETE_ATTEMPT_SQL, &params)
                .map_err(map_storage_error)?;
            if updated != 1 {
                return Err(publication_error("attempt changed during atomic publication"));
            }
            mark_embedded_inner(conn, &claim.doc_id, &claim.embedder_id, now_ms)?;
            Ok(ClaimOutcome::Applied(()))
        })
    }

    /// Read a published vector only for the current, embedded catalog revision.
    ///
    /// Absent, changed, failed, pending, skipped, or deleted documents return
    /// `None`. A producer mismatch or damaged record is an error, never a vector
    /// silently interpreted in a different space. Completed-job history may be
    /// pruned independently: durable vectors do not reference disposable jobs.
    pub fn get_published_embedding(
        &self,
        doc_id: &str,
        embedder_id: &str,
        expected_identity: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<Option<PublishedEmbedding>> {
        if doc_id.trim().is_empty() || embedder_id.trim().is_empty() {
            return Err(publication_error("document and embedder IDs must not be empty"));
        }
        expected_identity.validate()?;
        self.transaction(|conn| {
            let rows = conn
                .query_with_params_sync(
                    READ_EMBEDDING_SQL,
                    &[text_value(doc_id), text_value(embedder_id)],
                )
                .map_err(map_storage_error)?;
            let Some(row) = rows.first() else {
                return Ok(None);
            };
            let content_hash = hash_column(row, 0)?;
            if content_hash != hash_column(row, 8)? {
                return Ok(None);
            }
            let identity_json = text_column(row, 1)?;
            let vector_le = blob_column(row, 2)?;
            let record_hash = hash_column(row, 3)?;
            let source_job_id = integer_column(row, 4)?;
            let claim_epoch = integer_column(row, 5)?;
            let published_at_ms = integer_column(row, 6)?;
            if integer_column(row, 7)? != 1 || source_job_id <= 0 || claim_epoch <= 0 {
                return Err(publication_error("invalid embedding publication version or attempt"));
            }
            if publication_hash(
                doc_id,
                embedder_id,
                &content_hash,
                identity_json,
                vector_le,
                [source_job_id, claim_epoch, published_at_ms],
            )? != record_hash
            {
                return Err(publication_error("embedding publication checksum mismatch"));
            }
            let identity: EmbeddingIdentityBundleV1 = serde_json::from_str(identity_json)
                .map_err(|_| publication_error("invalid stored embedding identity"))?;
            // Never return coordinates or provider fields under the wrong contract.
            if &identity != expected_identity {
                return Err(publication_error("published embedding producer identity mismatch"));
            }
            let dimension = usize::try_from(identity.space.dimension)
                .map_err(|_| publication_error("embedding dimension does not fit usize"))?;
            let expected_bytes = dimension
                .checked_mul(4)
                .ok_or_else(|| publication_error("embedding byte length overflow"))?;
            if vector_le.len() != expected_bytes {
                return Err(publication_error("published embedding has an invalid byte length"));
            }
            let mut values = Vec::new();
            values.try_reserve_exact(dimension).map_err(map_storage_error)?;
            for bytes in vector_le.chunks_exact(4) {
                values.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            let embedding = IdentityBoundEmbedding { values, identity };
            validate_response(expected_identity, &embedding)?;
            Ok(Some(PublishedEmbedding {
                doc_id: doc_id.to_owned(),
                embedder_id: embedder_id.to_owned(),
                content_hash,
                source_job_id,
                claim_epoch,
                published_at_ms,
                embedding,
            }))
        })
    }
}

fn owned_document(
    conn: &AsyncConnection,
    claim: &ClaimedJob,
    expected_hash: Option<&[u8; 32]>,
) -> SearchResult<ClaimOutcome<DocumentRecord>> {
    if claim.job_id <= 0 || claim.claim_epoch <= 0 {
        return Ok(ClaimOutcome::LostClaim);
    }
    let params = claim_params(claim);
    if conn
        .query_with_params_sync(OWNED_ATTEMPT_SQL, &params)
        .map_err(map_storage_error)?
        .is_empty()
    {
        return Ok(ClaimOutcome::LostClaim);
    }
    if let Some(document) = get_document_inner(conn, &claim.doc_id)?
        && claim.content_hash.as_ref().is_none_or(|hash| *hash == document.content_hash)
        && expected_hash.is_none_or(|hash| *hash == document.content_hash)
    {
        return Ok(ClaimOutcome::Applied(document));
    }
    conn.execute_with_params_sync(RETIRE_ATTEMPT_SQL, &params)
        .map_err(map_storage_error)?;
    Ok(ClaimOutcome::Superseded)
}

fn claim_params(claim: &ClaimedJob) -> [SqliteValue; 6] {
    [
        SqliteValue::Integer(claim.job_id),
        SqliteValue::Integer(claim.claim_epoch),
        text_value(&claim.worker_id),
        text_value(&claim.doc_id),
        text_value(&claim.embedder_id),
        claim.content_hash.map_or(SqliteValue::Null, |hash| SqliteValue::Blob(hash.to_vec().into())),
    ]
}

fn validate_response(
    expected: &EmbeddingIdentityBundleV1,
    response: &IdentityBoundEmbedding,
) -> SearchResult<()> {
    if &response.identity != expected {
        return Err(publication_error("embedding response differs from the admitted producer"));
    }
    response.validate()?;
    let norm_sq: f32 = response.values.iter().map(|value| value * value).sum();
    if norm_sq == 0.0 || !norm_sq.is_finite() {
        return Err(publication_error("embedding must have finite, non-zero squared norm"));
    }
    Ok(())
}

fn encode_vector(values: &[f32]) -> SearchResult<Vec<u8>> {
    let length = values.len().checked_mul(4)
        .ok_or_else(|| publication_error("embedding byte length overflow"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(map_storage_error)?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn publication_hash(
    doc_id: &str,
    embedder_id: &str,
    content_hash: &[u8; 32],
    identity_json: &str,
    vector_le: &[u8],
    attempt: [i64; 3],
) -> SearchResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.published-embedding.v1\0");
    // Length framing prevents ambiguous field boundaries and binds row keys,
    // producer, source revision, vector bytes, attempt, and publication time.
    for part in [doc_id.as_bytes(), embedder_id.as_bytes(), content_hash.as_slice(), identity_json.as_bytes(), vector_le] {
        let length = u64::try_from(part.len())
            .map_err(|_| publication_error("embedding record field length overflow"))?;
        hasher.update(length.to_le_bytes());
        hasher.update(part);
    }
    for value in attempt {
        hasher.update(value.to_le_bytes());
    }
    Ok(hasher.finalize().into())
}

fn text_value(value: &str) -> SqliteValue {
    SqliteValue::Text(value.to_owned().into())
}

fn text_column(row: &Row, column: usize) -> SearchResult<&str> {
    match row.get(column) {
        Some(SqliteValue::Text(value)) => Ok(value),
        _ => Err(publication_error("embedding record text column has an invalid type")),
    }
}

fn blob_column(row: &Row, column: usize) -> SearchResult<&[u8]> {
    match row.get(column) {
        Some(SqliteValue::Blob(value)) => Ok(value),
        _ => Err(publication_error("embedding record blob column has an invalid type")),
    }
}

fn hash_column(row: &Row, column: usize) -> SearchResult<[u8; 32]> {
    blob_column(row, column)?.try_into()
        .map_err(|_| publication_error("embedding record digest must have 32 bytes"))
}

fn integer_column(row: &Row, column: usize) -> SearchResult<i64> {
    match row.get(column) {
        Some(SqliteValue::Integer(value)) => Ok(*value),
        _ => Err(publication_error("embedding record integer column has an invalid type")),
    }
}

fn unix_ms() -> SearchResult<i64> {
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).map_err(map_storage_error)?;
    i64::try_from(duration.as_millis())
        .map_err(|_| publication_error("embedding publication timestamp overflow"))
}

fn publication_error(message: &'static str) -> SearchError {
    // Do not include input text, vector values, or untrusted identity fields.
    SearchError::SubsystemError {
        subsystem: "storage",
        source: Box::new(io::Error::new(io::ErrorKind::InvalidData, message)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::sync::Arc;

    use super::*;
    use crate::{ContentHasher, JobQueueConfig, PersistentJobQueue, StorageConfig};

    fn document(id: &str, body: &str) -> DocumentRecord {
        DocumentRecord {
            doc_id: id.to_owned(),
            source_path: None,
            content_preview: body.chars().take(400).collect(),
            content_hash: ContentHasher::hash(body),
            content_length: body.chars().count(),
            created_at: 1,
            updated_at: 1,
            metadata: None,
        }
    }

    fn response(model: &str, values: Vec<f32>) -> IdentityBoundEmbedding {
        IdentityBoundEmbedding {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(
                model,
                u32::try_from(values.len()).unwrap(),
            ),
            values,
        }
    }

    fn fixture() -> (Arc<Storage>, PersistentJobQueue, ClaimedJob, IdentityBoundEmbedding) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        let doc = document("doc", "original body");
        storage.upsert_document(&doc).unwrap();
        storage.set_document_content("doc", "original body").unwrap();
        queue.enqueue("doc", "model", &doc.content_hash, 0).unwrap();
        let claim = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        (storage, queue, claim, response("model", vec![0.6, 0.8]))
    }

    fn rows(storage: &Storage, sql: &str, columns: usize) -> Vec<Vec<SqliteValue>> {
        storage.connection().query_sync(sql).unwrap().iter().map(|row| {
            (0..columns).map(|column| row.get(column).unwrap().clone()).collect()
        }).collect()
    }

    fn state(storage: &Storage) -> Vec<Vec<Vec<SqliteValue>>> {
        vec![
            rows(storage, "SELECT * FROM embedding_jobs ORDER BY job_id;", 14),
            rows(storage, "SELECT * FROM embedding_status ORDER BY doc_id, embedder_id;", 7),
            rows(storage, "SELECT * FROM published_embeddings ORDER BY doc_id, embedder_id;", 10),
        ]
    }

    fn publish(storage: &Storage, claim: &ClaimedJob, output: &IdentityBoundEmbedding) -> ClaimOutcome<()> {
        storage.publish_claimed_embedding(
            claim,
            &ContentHasher::hash("original body"),
            &output.identity,
            output,
        ).unwrap()
    }

    #[test]
    fn atomic_publication_completes_catalog_and_preserves_all_f32_bits_on_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let config = StorageConfig {
            db_path: directory.path().join("embeddings.sqlite3"),
            ..StorageConfig::default()
        };
        let output = response("model", vec![0.6, 0.8, -0.0, f32::from_bits(1)]);
        let claim;
        {
            let storage = Arc::new(Storage::open(config.clone()).unwrap());
            let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
            let doc = document("doc", "original body");
            storage.upsert_document(&doc).unwrap();
            queue.enqueue("doc", "model", &doc.content_hash, 0).unwrap();
            claim = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
            assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::Applied(()));
            assert_eq!(queue.queue_depth().unwrap().completed, 1);
            assert_eq!(queue.queue_depth().unwrap().processing, 0);
            assert_eq!(storage.count_by_status("model").unwrap().embedded, 1);
        }
        let storage = Storage::open(config).unwrap();
        let retained = storage.get_published_embedding("doc", "model", &output.identity).unwrap().unwrap();
        assert_eq!(retained.source_job_id, claim.job_id);
        assert_eq!(retained.claim_epoch, claim.claim_epoch);
        assert_eq!(retained.content_hash, ContentHasher::hash("original body"));
        assert_eq!(retained.embedding.identity, output.identity);
        assert_eq!(
            retained.embedding.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            output.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::LostClaim);
    }

    #[test]
    fn reclaimed_same_worker_cannot_overwrite_successor_vector() {
        let (storage, queue, old, old_output) = fixture();
        storage.connection().execute_sync("UPDATE embedding_jobs SET started_at = 0;").unwrap();
        assert_eq!(queue.reclaim_stale_jobs().unwrap(), 1);
        let current = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        assert_eq!(current.job_id, old.job_id);
        assert_eq!(current.claim_epoch, old.claim_epoch + 1);
        let output = response("model", vec![0.8, 0.6]);
        assert_eq!(publish(&storage, &current, &output), ClaimOutcome::Applied(()));
        let before = state(&storage);
        assert_eq!(publish(&storage, &old, &old_output), ClaimOutcome::LostClaim);
        assert_eq!(state(&storage), before);
        assert_eq!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().unwrap().embedding, output);
    }

    #[test]
    fn forged_claim_identity_cannot_publish_or_modify_any_history() {
        let (storage, _queue, claim, output) = fixture();
        let before = state(&storage);
        for field in ["job", "epoch", "worker", "document", "embedder", "hash"] {
            let mut forged = claim.clone();
            match field {
                "job" => forged.job_id += 1,
                "epoch" => forged.claim_epoch += 1,
                "worker" => forged.worker_id = "other-worker".to_owned(),
                "document" => forged.doc_id = "other-doc".to_owned(),
                "embedder" => forged.embedder_id = "other-model".to_owned(),
                "hash" => forged.content_hash = None,
                _ => unreachable!(),
            }
            assert_eq!(publish(&storage, &forged, &output), ClaimOutcome::LostClaim, "{field}");
            assert_eq!(state(&storage), before, "{field}");
        }
        let mut malformed = output.clone();
        malformed.values = vec![f32::NAN];
        let mut stale = claim.clone();
        stale.claim_epoch = 0;
        assert_eq!(publish(&storage, &stale, &malformed), ClaimOutcome::LostClaim);
        assert_eq!(state(&storage), before);
    }

    #[test]
    fn superseded_revision_retires_only_its_claim_and_keeps_replacement_runnable() {
        let (storage, queue, old, output) = fixture();
        let current_doc = document("doc", "replacement body");
        storage.upsert_document(&current_doc).unwrap();
        queue.enqueue("doc", "model", &current_doc.content_hash, 0).unwrap();
        let catalog_before = rows(&storage, "SELECT * FROM embedding_status;", 7);
        assert_eq!(publish(&storage, &old, &output), ClaimOutcome::Superseded);
        assert_eq!(rows(&storage, "SELECT * FROM embedding_status;", 7), catalog_before);
        assert!(rows(&storage, "SELECT * FROM published_embeddings;", 10).is_empty());
        assert_eq!(queue.queue_depth().unwrap().pending, 1);
        let current = queue.claim_batch("next", 1).unwrap().pop().unwrap();
        assert_eq!(storage.publish_claimed_embedding(
            &current, &current_doc.content_hash, &output.identity, &output,
        ).unwrap(), ClaimOutcome::Applied(()));
        assert_eq!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().unwrap().content_hash, current_doc.content_hash);
    }

    #[test]
    fn invalid_outputs_leave_claim_catalog_and_previous_vectors_unchanged() {
        let (storage, queue, first, output) = fixture();
        assert_eq!(publish(&storage, &first, &output), ClaimOutcome::Applied(()));
        queue.enqueue("doc", "model", &ContentHasher::hash("original body"), 0).unwrap();
        let claim = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        let before = state(&storage);
        let mut candidates = Vec::new();
        for values in [vec![], vec![0.5], vec![f32::NAN, 1.0], vec![f32::INFINITY, 1.0], vec![0.0, -0.0], vec![f32::MAX, 1.0]] {
            let mut invalid = output.clone();
            invalid.values = values;
            candidates.push(invalid);
        }
        candidates.push(response("different-producer", vec![0.6, 0.8]));
        for invalid in candidates {
            assert!(storage.publish_claimed_embedding(
                &claim, &ContentHasher::hash("original body"), &output.identity, &invalid,
            ).is_err());
            assert_eq!(state(&storage), before);
        }
    }

    #[test]
    fn final_catalog_failure_rolls_back_vector_queue_and_completed_history() {
        let (storage, queue, first, output) = fixture();
        assert_eq!(publish(&storage, &first, &output), ClaimOutcome::Applied(()));
        queue.enqueue("doc", "model", &ContentHasher::hash("original body"), 0).unwrap();
        let claim = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        let before = state(&storage);
        // Owned-attempt admission does not read embedding_status. This fails
        // after both the vector UPSERT and queue/history transitions occurred.
        storage.connection().execute_sync(
            "ALTER TABLE embedding_status RENAME COLUMN status TO unavailable_status;",
        ).unwrap();
        let replacement = response("model", vec![0.8, 0.6]);
        assert!(storage.publish_claimed_embedding(
            &claim, &ContentHasher::hash("original body"), &replacement.identity, &replacement,
        ).is_err());
        assert_eq!(state(&storage), before);
        storage.connection().execute_sync(
            "ALTER TABLE embedding_status RENAME COLUMN unavailable_status TO status;",
        ).unwrap();
        assert_eq!(publish(&storage, &claim, &replacement), ClaimOutcome::Applied(()));
        assert_eq!(queue.queue_depth().unwrap().completed, 1);
    }

    #[test]
    fn vector_and_metadata_corruption_are_rejected_even_when_coordinates_are_finite() {
        for mutation in [
            "UPDATE published_embeddings SET vector_le = X'0000803F00000000';",
            "UPDATE published_embeddings SET identity_json = '{}';",
            "UPDATE published_embeddings SET record_hash = X'01';",
            "UPDATE published_embeddings SET source_job_id = source_job_id + 1;",
            "UPDATE published_embeddings SET claim_epoch = claim_epoch + 1;",
            "UPDATE published_embeddings SET published_at_ms = published_at_ms + 1;",
            "UPDATE published_embeddings SET record_version = 2;",
        ] {
            let (storage, _queue, claim, output) = fixture();
            assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::Applied(()));
            storage.connection().execute_sync(mutation).unwrap();
            assert!(storage.get_published_embedding("doc", "model", &output.identity).is_err(), "{mutation}");
        }
    }

    #[test]
    fn producer_mismatch_is_not_silent_reinterpretation() {
        let (storage, _queue, claim, output) = fixture();
        assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::Applied(()));
        let foreign = response("foreign", vec![0.6, 0.8]);
        assert!(storage.get_published_embedding("doc", "model", &foreign.identity).is_err());
        assert_eq!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().unwrap().embedding, output);
    }

    #[test]
    fn current_revision_and_catalog_status_gate_vector_visibility() {
        for status in ["pending", "failed", "skipped"] {
            let (storage, _queue, claim, output) = fixture();
            assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::Applied(()));
            storage.connection().execute_with_params_sync(
                "UPDATE embedding_status SET status = ?1;", &[text_value(status)],
            ).unwrap();
            assert!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().is_none());
        }
        let (storage, _queue, claim, output) = fixture();
        assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::Applied(()));
        storage.upsert_document(&document("doc", "new body")).unwrap();
        // Even an out-of-band mark_embedded cannot bless the old vector's hash.
        storage.mark_embedded("doc", "model").unwrap();
        assert!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().is_none());
    }

    #[test]
    fn deleting_document_cascades_vectors_but_pruning_jobs_does_not() {
        let (storage, _queue, claim, output) = fixture();
        assert_eq!(publish(&storage, &claim, &output), ClaimOutcome::Applied(()));
        storage.connection().execute_sync("DELETE FROM embedding_jobs;").unwrap();
        assert!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().is_some());
        assert!(storage.delete_document("doc").unwrap());
        assert!(rows(&storage, "SELECT * FROM published_embeddings;", 10).is_empty());
        storage.upsert_document(&document("doc", "original body")).unwrap();
        storage.mark_embedded("doc", "model").unwrap();
        assert!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().is_none());
    }

    #[test]
    fn legacy_unknown_claim_hash_is_bound_to_the_captured_input_revision() {
        let (storage, queue, old, output) = fixture();
        storage.connection().execute_sync("UPDATE embedding_jobs SET content_hash = NULL, started_at = 0;").unwrap();
        assert_eq!(queue.reclaim_stale_jobs().unwrap(), 1);
        let legacy = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        assert_eq!(legacy.content_hash, None);
        assert_eq!(publish(&storage, &old, &output), ClaimOutcome::LostClaim);
        assert_eq!(publish(&storage, &legacy, &output), ClaimOutcome::Applied(()));
        assert_eq!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().unwrap().content_hash, ContentHasher::hash("original body"));
    }

    #[test]
    fn fast_and_quality_publications_do_not_replace_each_other() {
        let (storage, queue, fast, output) = fixture();
        assert_eq!(publish(&storage, &fast, &output), ClaimOutcome::Applied(()));
        queue.enqueue("doc", "quality", &ContentHasher::hash("original body"), 0).unwrap();
        let quality = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        let quality_output = response("quality", vec![1.0, 0.0, 0.0]);
        assert_eq!(publish(&storage, &quality, &quality_output), ClaimOutcome::Applied(()));
        assert_eq!(storage.get_published_embedding("doc", "model", &output.identity).unwrap().unwrap().embedding, output);
        assert_eq!(storage.get_published_embedding("doc", "quality", &quality_output.identity).unwrap().unwrap().embedding, quality_output);
        assert_eq!(rows(&storage, "SELECT * FROM published_embeddings;", 10).len(), 2);
    }
}
