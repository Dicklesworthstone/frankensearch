//! Verified, bounded reads of a single publication snapshot.

use super::{
    EmbeddingIdentityBundleV1, IdentityBoundEmbedding, PublishedEmbedding, QuantizationFormat,
    READ_EMBEDDING_SQL, blob_column, hash_column, integer_column, map_storage_error,
    publication_checkpoint, publication_error, publication_hash, text_column, text_value,
    validate_response,
};
use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use fsqlite::{AsyncConnection, Row};
use fsqlite_types::value::SqliteValue;

use crate::Storage;

// Read lengths before fetching any identity/vector payload. Keyset pagination
// does not repeatedly materialize all IDs or use an OFFSET into a growing scan.
const SCAN_PAGE_SQL: &str = "SELECT v.doc_id, \
    LENGTH(CAST(v.identity_json AS BLOB)), LENGTH(CAST(v.vector_le AS BLOB)) \
    FROM published_embeddings v \
    JOIN documents d ON d.doc_id = v.doc_id \
    JOIN embedding_status e ON e.doc_id = v.doc_id AND e.embedder_id = v.embedder_id \
    WHERE v.embedder_id = ?1 AND e.status = 'embedded' \
      AND v.content_hash = d.content_hash \
      AND (?2 IS NULL OR v.doc_id COLLATE BINARY > ?2) \
    ORDER BY v.doc_id COLLATE BINARY LIMIT ?3;";
const METADATA_PAGE_SIZE: usize = 64;

/// Limits for a complete read, not permission to return an incomplete prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishedEmbeddingReadLimits {
    /// Maximum current, embedded records admitted from the selected tier.
    pub max_records: usize,
    /// Maximum total stored identity-JSON and vector bytes admitted.
    /// This bounds admitted payload, not RSS or SQL execution time. Driver
    /// caches, row keys, decoded objects, and container overhead are additional.
    pub max_payload_bytes: usize,
}

impl Default for PublishedEmbeddingReadLimits {
    fn default() -> Self {
        Self {
            max_records: 10_000,
            max_payload_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Successfully verified records and stored payload bytes in a complete scan.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PublishedEmbeddingScanStats {
    pub records: usize,
    pub payload_bytes: usize,
}

/// Complete bounded materialization from one database transaction.
///
/// Records are ordered by binary document ID and carry the revision needed
/// for subsequent hydration. Later catalog changes do not change this value.
#[derive(Debug, Clone, PartialEq)]
pub struct PublishedEmbeddingSnapshot {
    pub records: Vec<PublishedEmbedding>,
    pub stats: PublishedEmbeddingScanStats,
}

impl Storage {
    /// Load all current embedded publications for one operational tier.
    ///
    /// Every record must match the expected complete producer identity and
    /// pass the same checksum, dimension, and coordinate checks as a point
    /// lookup. A current foreign-producer or corrupt row fails the entire read;
    /// it is never silently omitted to fabricate a complete export.
    ///
    /// Pending, failed, skipped, missing, and obsolete document revisions are
    /// not published coverage. Reaching either limit is an error, not truncation.
    /// Zero limits therefore succeed only on an empty publication set.
    ///
    /// The scan is synchronous, with cancellation checkpoints between bounded
    /// metadata pages and individual records. It holds one read transaction
    /// across the scan, with no await, provider call, or user callback. Schedule
    /// large scans on the caller's blocking execution path. The transaction
    /// uses the configured `Storage` begin mode; a writer-locking configuration
    /// can block writers during this read. External FSVI/ANN files are unchanged.
    pub fn load_published_embeddings(
        &self,
        cx: &Cx,
        embedder_id: &str,
        expected_identity: &EmbeddingIdentityBundleV1,
        limits: PublishedEmbeddingReadLimits,
    ) -> SearchResult<PublishedEmbeddingSnapshot> {
        validate_read_request(cx, embedder_id, expected_identity)?;
        let result = self.transaction(|conn| {
            // A retried transaction must start with fresh accumulators; it must
            // not append duplicate records to a previous failed snapshot.
            let mut records = Vec::new();
            let stats = scan_on(conn, cx, embedder_id, expected_identity, limits, |record| {
                records.try_reserve(1).map_err(map_storage_error)?;
                records.push(record);
                Ok(())
            })?;
            Ok(PublishedEmbeddingSnapshot { records, stats })
        });
        publication_checkpoint(cx)?;
        result
    }
}

pub(super) fn validate_read_identity(identity: &EmbeddingIdentityBundleV1) -> SearchResult<()> {
    identity.validate()?;
    if identity.storage.quantization != QuantizationFormat::F32
        || !identity.storage.format.starts_with("in-memory-")
        || !matches!(
            identity.storage.endianness.as_str(),
            "native-f32-values" | "native-test-only"
        )
    {
        return Err(publication_error("published reads require an in-memory f32 identity"));
    }
    Ok(())
}

fn validate_read_request(
    cx: &Cx,
    embedder_id: &str,
    identity: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    publication_checkpoint(cx)?;
    if embedder_id.trim().is_empty() {
        return Err(publication_error("embedder ID must not be empty"));
    }
    validate_read_identity(identity)
}

// Private visits are pure result accumulation/scoring, never caller callbacks.
// The caller owns the transaction and initializes all visit state inside it.
fn scan_on(
    conn: &AsyncConnection,
    cx: &Cx,
    embedder_id: &str,
    identity: &EmbeddingIdentityBundleV1,
    limits: PublishedEmbeddingReadLimits,
    mut visit: impl FnMut(PublishedEmbedding) -> SearchResult<()>,
) -> SearchResult<PublishedEmbeddingScanStats> {
    let mut stats = PublishedEmbeddingScanStats::default();
    let mut after: Option<String> = None;
    loop {
        publication_checkpoint(cx)?;
        // One extra metadata record distinguishes an exact boundary from a
        // truncated scan without ever fetching the over-budget payload.
        let page_size = METADATA_PAGE_SIZE.min(
            limits.max_records.saturating_sub(stats.records).saturating_add(1),
        );
        let rows = conn.query_with_params_sync(
            SCAN_PAGE_SQL,
            &[
                text_value(embedder_id),
                after.as_deref().map_or(SqliteValue::Null, text_value),
                SqliteValue::Integer(i64::try_from(page_size).map_err(map_storage_error)?),
            ],
        ).map_err(map_storage_error)?;
        publication_checkpoint(cx)?;
        if rows.is_empty() {
            return Ok(stats);
        }
        for row in &rows {
            publication_checkpoint(cx)?;
            if stats.records == limits.max_records {
                return Err(read_limit_error("max_records", limits.max_records));
            }
            let id = text_column(row, 0)?;
            if id.trim().is_empty() || after.as_deref().is_some_and(|previous| id <= previous) {
                return Err(publication_error("published scan returned an invalid document order"));
            }
            let identity_bytes = usize::try_from(integer_column(row, 1)?)
                .map_err(|_| publication_error("invalid stored identity byte length"))?;
            let vector_bytes = usize::try_from(integer_column(row, 2)?)
                .map_err(|_| publication_error("invalid stored vector byte length"))?;
            let payload_bytes = stats.payload_bytes.checked_add(identity_bytes)
                .and_then(|value| value.checked_add(vector_bytes))
                .filter(|value| *value <= limits.max_payload_bytes)
                .ok_or_else(|| read_limit_error("max_payload_bytes", limits.max_payload_bytes))?;
            let records = conn.query_with_params_sync(
                READ_EMBEDDING_SQL, &[text_value(id), text_value(embedder_id)],
            ).map_err(map_storage_error)?;
            publication_checkpoint(cx)?;
            let record = records.first()
                .ok_or_else(|| publication_error("publication disappeared within a read snapshot"))?;
            let record = decode_row(record, id, embedder_id, identity)?
                .ok_or_else(|| publication_error("publication revision changed within a read snapshot"))?;
            publication_checkpoint(cx)?;
            visit(record)?;
            stats.records += 1;
            stats.payload_bytes = payload_bytes;
            after = Some(id.to_owned());
            publication_checkpoint(cx)?;
        }
    }
}

fn read_limit_error(field: &str, limit: usize) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("published_embeddings.{field}"),
        value: limit.to_string(),
        reason: "complete publication scan exceeds the limit; no partial result returned".to_owned(),
    }
}

// Shared by point lookups, snapshot export, and the exact read backend.
pub(super) fn decode_row(
    row: &Row,
    doc_id: &str,
    embedder_id: &str,
    expected_identity: &EmbeddingIdentityBundleV1,
) -> SearchResult<Option<PublishedEmbedding>> {
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
        doc_id, embedder_id, &content_hash, identity_json, vector_le,
        [source_job_id, claim_epoch, published_at_ms],
    )? != record_hash {
        return Err(publication_error("embedding publication checksum mismatch"));
    }
    let identity: EmbeddingIdentityBundleV1 = serde_json::from_str(identity_json)
        .map_err(|_| publication_error("invalid stored embedding identity"))?;
    if &identity != expected_identity {
        return Err(publication_error("published embedding producer identity mismatch"));
    }
    let dimension = usize::try_from(identity.space.dimension)
        .map_err(|_| publication_error("embedding dimension does not fit usize"))?;
    let expected_bytes = dimension.checked_mul(4)
        .ok_or_else(|| publication_error("embedding byte length overflow"))?;
    if vector_le.len() != expected_bytes {
        return Err(publication_error("published embedding has an invalid byte length"));
    }
    let mut values = Vec::new();
    values.try_reserve_exact(dimension).map_err(map_storage_error)?;
    for bytes in vector_le.as_chunks::<4>().0 {
        values.push(f32::from_le_bytes(*bytes));
    }
    let embedding = IdentityBoundEmbedding { values, identity };
    validate_response(expected_identity, &embedding)?;
    Ok(Some(PublishedEmbedding {
        doc_id: doc_id.to_owned(),
        embedder_id: embedder_id.to_owned(),
        content_hash, source_job_id, claim_epoch, published_at_ms, embedding,
    }))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::sync::Arc;

    use super::*;
    use crate::{ClaimOutcome, ContentHasher, DocumentRecord, JobQueueConfig, PersistentJobQueue};

    fn fixture() -> (Arc<Storage>, PersistentJobQueue) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        (storage, queue)
    }

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model(name, 2)
    }

    fn publish(
        storage: &Storage,
        queue: &PersistentJobQueue,
        id: &str,
        producer: &str,
        values: Vec<f32>,
    ) {
        let body = format!("complete body for {id}");
        let hash = ContentHasher::hash(&body);
        let document = DocumentRecord::new(id, &body, hash, body.chars().count(), 1, 1);
        storage.upsert_document(&document).unwrap();
        storage.set_document_content(id, &body).unwrap();
        queue.enqueue(id, "model", &hash, 0).unwrap();
        let claim = queue.claim_batch("publisher", 1).unwrap().pop().unwrap();
        assert_eq!(claim.doc_id, id);
        let response = IdentityBoundEmbedding { values, identity: identity(producer) };
        assert_eq!(
            storage.publish_claimed_embedding(&claim, &hash, &response.identity, &response).unwrap(),
            ClaimOutcome::Applied(())
        );
    }

    #[test]
    fn snapshot_crosses_metadata_pages_and_preserves_order_revision_and_float_bits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let mut ids: Vec<_> = (0..67).map(|n| format!("doc-{n:03}")).collect();
            ids.extend(["a\0tail".to_owned(), "数据".to_owned(), "é".to_owned()]);
            for id in ids.iter().rev() {
                publish(&storage, &queue, id, "model", vec![1.0, -0.0]);
            }
            // Publication is independent of disposable completed-job history.
            storage.connection().execute_sync("DELETE FROM embedding_jobs;").unwrap();
            let snapshot = storage.load_published_embeddings(
                &cx, "model", &identity("model"), PublishedEmbeddingReadLimits::default(),
            ).unwrap();
            ids.sort();
            assert_eq!(snapshot.records.iter().map(|row| &row.doc_id).collect::<Vec<_>>(), ids.iter().collect::<Vec<_>>());
            assert_eq!(snapshot.stats.records, ids.len());
            let payload_size = serde_json::to_vec(&identity("model")).unwrap().len() + 8;
            assert_eq!(snapshot.stats.payload_bytes, payload_size * ids.len());
            for record in &snapshot.records {
                assert_eq!(record.embedding.values[0].to_bits(), 1.0_f32.to_bits());
                assert_eq!(record.embedding.values[1].to_bits(), (-0.0_f32).to_bits());
                assert_eq!(record.content_hash, ContentHasher::hash(&format!("complete body for {}", record.doc_id)));
            }
        });
    }

    #[test]
    fn snapshot_excludes_unpublished_statuses_deleted_docs_and_old_revisions() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in ["current", "pending", "failed", "skipped", "obsolete", "deleted"] {
                publish(&storage, &queue, id, "model", vec![1.0, 0.0]);
            }
            storage.connection().execute_sync(
                "UPDATE embedding_status SET status = doc_id WHERE doc_id IN ('pending', 'failed', 'skipped');",
            ).unwrap();
            let mut obsolete = storage.get_document("obsolete").unwrap().unwrap();
            obsolete.content_hash = ContentHasher::hash("replacement");
            obsolete.content_preview = "replacement".to_owned();
            obsolete.content_length = 11;
            storage.upsert_document(&obsolete).unwrap();
            storage.connection().execute_sync("DELETE FROM documents WHERE doc_id = 'deleted';").unwrap();
            let snapshot = storage.load_published_embeddings(
                &cx, "model", &identity("model"), PublishedEmbeddingReadLimits::default(),
            ).unwrap();
            assert_eq!(snapshot.stats.records, 1);
            assert_eq!(snapshot.records[0].doc_id, "current");
            assert!(storage.load_published_embeddings(
                &cx, "other-tier", &identity("model"), PublishedEmbeddingReadLimits::default(),
            ).unwrap().records.is_empty());
        });
    }

    #[test]
    fn count_and_payload_limits_fail_instead_of_returning_a_prefix() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let zero = PublishedEmbeddingReadLimits { max_records: 0, max_payload_bytes: 0 };
            assert!(storage.load_published_embeddings(&cx, "model", &identity("model"), zero).unwrap().records.is_empty());
            publish(&storage, &queue, "a", "model", vec![1.0, 0.0]);
            publish(&storage, &queue, "b", "model", vec![0.0, 1.0]);
            let bytes = 2 * (serde_json::to_vec(&identity("model")).unwrap().len() + 8);
            let exact = PublishedEmbeddingReadLimits { max_records: 2, max_payload_bytes: bytes };
            assert_eq!(storage.load_published_embeddings(&cx, "model", &identity("model"), exact).unwrap().stats.records, 2);
            for (limits, field) in [
                (PublishedEmbeddingReadLimits { max_records: 1, ..exact }, "published_embeddings.max_records"),
                (PublishedEmbeddingReadLimits { max_payload_bytes: bytes - 1, ..exact }, "published_embeddings.max_payload_bytes"),
                (zero, "published_embeddings.max_records"),
            ] {
                let error = storage.load_published_embeddings(&cx, "model", &identity("model"), limits).unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { field: actual, .. } if actual == field));
                assert!(!storage.connection().in_transaction());
            }
        });
    }

    #[test]
    fn oversized_payload_is_refused_before_it_is_loaded_or_decoded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "oversized", "model", vec![1.0, 0.0]);
            storage.connection().execute_with_params_sync(
                "UPDATE published_embeddings SET identity_json = ?1;",
                &[text_value(&"private-invalid-json".repeat(5000))],
            ).unwrap();
            let error = storage.load_published_embeddings(
                &cx, "model", &identity("model"),
                PublishedEmbeddingReadLimits { max_records: 1, max_payload_bytes: 1024 },
            ).unwrap_err();
            // Decoding the record instead would report a checksum/JSON error.
            assert!(matches!(error, SearchError::InvalidConfig { field, .. } if field == "published_embeddings.max_payload_bytes"));
        });
    }

    #[test]
    fn a_current_foreign_producer_or_corrupt_record_fails_the_entire_snapshot() {
        for mutation in [None, Some("UPDATE published_embeddings SET vector_le = X'0000' WHERE doc_id = 'z';"), Some("UPDATE published_embeddings SET record_version = 99 WHERE doc_id = 'z';")] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "a", "model", vec![1.0, 0.0]);
                publish(&storage, &queue, "z", if mutation.is_some() { "model" } else { "foreign" }, vec![0.0, 1.0]);
                if let Some(sql) = mutation {
                    storage.connection().execute_sync(sql).unwrap();
                }
                assert!(storage.load_published_embeddings(
                    &cx, "model", &identity("model"), PublishedEmbeddingReadLimits::default(),
                ).is_err());
                assert_eq!(queue.queue_depth().unwrap().completed, 2);
                assert!(!storage.connection().in_transaction());
            });
        }
    }

    #[test]
    fn cancellation_before_a_snapshot_does_not_begin_a_transaction() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "doc", "model", vec![1.0, 0.0]);
            let before = storage.metrics_snapshot();
            cx.cancel_with(asupersync::CancelKind::User, Some("stop read"));
            let error = storage.load_published_embeddings(
                &cx, "model", &identity("model"), PublishedEmbeddingReadLimits::default(),
            ).unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }

    #[test]
    fn cancellation_during_scan_discards_the_partial_snapshot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in ["a", "b"] {
                publish(&storage, &queue, id, "model", vec![1.0, 0.0]);
            }
            let mut visited = 0;
            let result = storage.transaction(|conn| {
                scan_on(conn, &cx, "model", &identity("model"), PublishedEmbeddingReadLimits::default(), |_| {
                    visited += 1;
                    cx.cancel_with(asupersync::CancelKind::User, Some("stop during read"));
                    Ok(())
                })
            });
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(visited, 1);
            assert_eq!(queue.queue_depth().unwrap().completed, 2);
            assert!(!storage.connection().in_transaction());
        });
    }
}
