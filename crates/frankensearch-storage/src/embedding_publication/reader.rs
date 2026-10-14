//! Verified, bounded reads of a single publication snapshot.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{
    EmbeddingIdentityBundleV1, IdentityBoundEmbedding, PublishedEmbedding, QuantizationFormat,
    READ_EMBEDDING_SQL, blob_column, hash_column, integer_column, map_storage_error,
    publication_checkpoint, publication_error, publication_hash, text_column, text_value,
    validate_response,
};
use asupersync::Cx;
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::types::{BoundQueryEmbedding, VectorHit};
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

impl PublishedEmbeddingSnapshot {
    /// Search this captured publication set by exact cosine similarity.
    ///
    /// This is an in-memory exact backend, not ANN or live database discovery.
    /// It scans every captured vector, keeps at most `limit` hits, and orders
    /// them by descending score then binary document ID. Negative similarities
    /// are retained. `VectorHit::index` addresses `self.records`, so callers
    /// can hydrate using the exact content hash that produced each result.
    /// Reload the snapshot to observe later publications or document changes.
    ///
    /// The query must carry the same complete in-memory producer contract as
    /// every record. Hash-control spaces are not admitted as semantic search.
    /// Publicly mutable vector contracts, tier membership, and document order
    /// are revalidated on every search. Malformed or mixed-producer vectors
    /// fail the whole search even when
    /// they would not enter the top k. Counts in `stats` are not trusted as
    /// evidence of corpus coverage. A zero limit validates only the query.
    ///
    /// Norms and dot products use f64 intermediates without rewriting vectors.
    /// The finite cosine is clamped to [-1, 1] and then rounded to f32 for the
    /// existing hit type. Cancellation is checked during coordinate scans and
    /// between records. This synchronous CPU operation creates no runtime or
    /// task; schedule large searches on the caller's blocking execution path.
    pub fn search(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        limit: usize,
    ) -> SearchResult<Vec<VectorHit>> {
        publication_checkpoint(cx)?;
        validate_read_identity(query.identity())?;
        if matches!(
            query.identity().space.kind,
            EmbeddingSpaceKindV1::HashControl
        ) {
            return Err(publication_error(
                "hash-control vectors are not semantic search input",
            ));
        }
        let query_norm = query_norm(cx, query.vector())?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut heap = BinaryHeap::new();
        let mut previous: Option<&str> = None;
        let tier = self
            .records
            .first()
            .map(|record| record.embedder_id.as_str());
        for (index, record) in self.records.iter().enumerate() {
            publication_checkpoint(cx)?;
            if record.doc_id.trim().is_empty()
                || record.embedder_id.trim().is_empty()
                || previous.is_some_and(|id| record.doc_id.as_str() <= id)
                || tier != Some(record.embedder_id.as_str())
            {
                return Err(publication_error(
                    "snapshot must contain one tier in unique document order",
                ));
            }
            validate_response(query.identity(), &record.embedding)?;
            let index = u32::try_from(index).map_err(|_| {
                publication_error("snapshot position exceeds vector-hit index width")
            })?;
            let score = cosine(cx, query.vector(), &record.embedding.values, query_norm)?;
            let hit = RankedHit(VectorHit {
                index,
                score,
                doc_id: record.doc_id.clone().into(),
            });
            retain_hit(&mut heap, hit, limit)?;
            previous = Some(record.doc_id.as_str());
        }
        publication_checkpoint(cx)?;
        // The greatest heap element is the worst hit. Ascending heap order
        // therefore yields the public best-first order, including tie breaks.
        let hits = heap
            .into_sorted_vec()
            .into_iter()
            .map(|hit| hit.0)
            .collect();
        publication_checkpoint(cx)?;
        Ok(hits)
    }
}

fn query_norm(cx: &Cx, values: &[f32]) -> SearchResult<f64> {
    let mut squared = 0.0_f64;
    for (index, &value) in values.iter().enumerate() {
        if index % 256 == 0 {
            publication_checkpoint(cx)?;
        }
        if !value.is_finite() {
            return Err(publication_error("query coordinates must be finite"));
        }
        let value = f64::from(value);
        squared += value * value;
    }
    publication_checkpoint(cx)?;
    if squared == 0.0 || !squared.is_finite() {
        return Err(publication_error("query must have finite non-zero norm"));
    }
    Ok(squared.sqrt())
}

fn cosine(cx: &Cx, query: &[f32], vector: &[f32], query_norm: f64) -> SearchResult<f32> {
    if query.len() != vector.len() {
        return Err(publication_error("query and publication dimensions differ"));
    }
    let mut dot = 0.0_f64;
    let mut squared = 0.0_f64;
    for (index, (&left, &right)) in query.iter().zip(vector).enumerate() {
        if index % 256 == 0 {
            publication_checkpoint(cx)?;
        }
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        squared += right * right;
    }
    let score = dot / (query_norm * squared.sqrt());
    if !score.is_finite() {
        return Err(publication_error("cosine similarity is not finite"));
    }
    // [-1, 1] cannot overflow f32. Rounding is the public VectorHit contract.
    #[allow(clippy::cast_possible_truncation)]
    let score = score.clamp(-1.0, 1.0) as f32;
    Ok(if score == 0.0 { 0.0 } else { score })
}

#[derive(Debug)]
struct RankedHit(VectorHit);

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.doc_id.cmp(&other.0.doc_id))
            .then_with(|| self.0.index.cmp(&other.0.index))
    }
}

fn retain_hit(heap: &mut BinaryHeap<RankedHit>, hit: RankedHit, limit: usize) -> SearchResult<()> {
    if heap.len() < limit {
        heap.try_reserve(1).map_err(map_storage_error)?;
        heap.push(hit);
    } else if let Some(mut worst) = heap.peek_mut()
        && hit < *worst
    {
        *worst = hit;
    }
    Ok(())
}

impl Storage {
    /// Capture verified vectors, canonical source text, and catalog metadata.
    ///
    /// All records are read in the SAME transaction, not hydrated by a later
    /// current-document lookup. Missing, stale, or corrupt source input fails
    /// the entire capture. Vector and source budgets are independent and both
    /// reject overflow rather than return an incomplete prefix. No external
    /// files, providers, caller callbacks, or awaits enter this transaction.
    /// The configured storage transaction mode may hold a writer lock.
    pub fn load_published_search_snapshot(
        &self,
        cx: &Cx,
        embedder_id: &str,
        expected_identity: &EmbeddingIdentityBundleV1,
        limits: crate::publication_search::PublishedSearchReadLimits,
    ) -> SearchResult<crate::publication_search::PublishedSearchSnapshot> {
        validate_read_request(cx, embedder_id, expected_identity)?;
        let result = self.transaction(|conn| {
            let mut records = Vec::new();
            let mut documents = Vec::new();
            let mut source_payload_bytes = 0_usize;
            let stats = scan_on(
                conn,
                cx,
                embedder_id,
                expected_identity,
                limits.embeddings,
                |record| {
                    let (document, bytes) = crate::publication_search::read_search_document(
                        conn,
                        cx,
                        &record,
                        limits.max_source_bytes - source_payload_bytes,
                    )?;
                    records.try_reserve(1).map_err(map_storage_error)?;
                    documents.try_reserve(1).map_err(map_storage_error)?;
                    source_payload_bytes += bytes;
                    records.push(record);
                    documents.push(document);
                    Ok(())
                },
            )?;
            Ok(crate::publication_search::PublishedSearchSnapshot {
                vectors: PublishedEmbeddingSnapshot { records, stats },
                documents,
                identity: expected_identity.clone(),
                embedder_id: embedder_id.to_owned(),
                source_payload_bytes,
            })
        });
        publication_checkpoint(cx)?;
        result
    }

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
        return Err(publication_error(
            "published reads require an in-memory f32 identity",
        ));
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
            limits
                .max_records
                .saturating_sub(stats.records)
                .saturating_add(1),
        );
        let rows = conn
            .query_with_params_sync(
                SCAN_PAGE_SQL,
                &[
                    text_value(embedder_id),
                    after.as_deref().map_or(SqliteValue::Null, text_value),
                    SqliteValue::Integer(i64::try_from(page_size).map_err(map_storage_error)?),
                ],
            )
            .map_err(map_storage_error)?;
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
                return Err(publication_error(
                    "published scan returned an invalid document order",
                ));
            }
            let identity_bytes = usize::try_from(integer_column(row, 1)?)
                .map_err(|_| publication_error("invalid stored identity byte length"))?;
            let vector_bytes = usize::try_from(integer_column(row, 2)?)
                .map_err(|_| publication_error("invalid stored vector byte length"))?;
            let payload_bytes = stats
                .payload_bytes
                .checked_add(identity_bytes)
                .and_then(|value| value.checked_add(vector_bytes))
                .filter(|value| *value <= limits.max_payload_bytes)
                .ok_or_else(|| read_limit_error("max_payload_bytes", limits.max_payload_bytes))?;
            let records = conn
                .query_with_params_sync(
                    READ_EMBEDDING_SQL,
                    &[text_value(id), text_value(embedder_id)],
                )
                .map_err(map_storage_error)?;
            publication_checkpoint(cx)?;
            let record = records.first().ok_or_else(|| {
                publication_error("publication disappeared within a read snapshot")
            })?;
            let record = decode_row(record, id, embedder_id, identity)?.ok_or_else(|| {
                publication_error("publication revision changed within a read snapshot")
            })?;
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
        reason: "complete publication scan exceeds the limit; no partial result returned"
            .to_owned(),
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
        return Err(publication_error(
            "invalid embedding publication version or attempt",
        ));
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
    if &identity != expected_identity {
        return Err(publication_error(
            "published embedding producer identity mismatch",
        ));
    }
    let dimension = usize::try_from(identity.space.dimension)
        .map_err(|_| publication_error("embedding dimension does not fit usize"))?;
    let expected_bytes = dimension
        .checked_mul(4)
        .ok_or_else(|| publication_error("embedding byte length overflow"))?;
    if vector_le.len() != expected_bytes {
        return Err(publication_error(
            "published embedding has an invalid byte length",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(dimension)
        .map_err(map_storage_error)?;
    for bytes in vector_le.as_chunks::<4>().0 {
        values.push(f32::from_le_bytes(*bytes));
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
        publish_with_identity(storage, queue, id, identity(producer), values);
    }

    fn publish_with_identity(
        storage: &Storage,
        queue: &PersistentJobQueue,
        id: &str,
        identity: EmbeddingIdentityBundleV1,
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
        let response = IdentityBoundEmbedding { values, identity };
        assert_eq!(
            storage
                .publish_claimed_embedding(&claim, &hash, &response.identity, &response)
                .unwrap(),
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
            storage
                .connection()
                .execute_sync("DELETE FROM embedding_jobs;")
                .unwrap();
            let snapshot = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                )
                .unwrap();
            ids.sort();
            assert_eq!(
                snapshot
                    .records
                    .iter()
                    .map(|row| &row.doc_id)
                    .collect::<Vec<_>>(),
                ids.iter().collect::<Vec<_>>()
            );
            assert_eq!(snapshot.stats.records, ids.len());
            let payload_size = serde_json::to_vec(&identity("model")).unwrap().len() + 8;
            assert_eq!(snapshot.stats.payload_bytes, payload_size * ids.len());
            for record in &snapshot.records {
                assert_eq!(record.embedding.values[0].to_bits(), 1.0_f32.to_bits());
                assert_eq!(record.embedding.values[1].to_bits(), (-0.0_f32).to_bits());
                assert_eq!(
                    record.content_hash,
                    ContentHasher::hash(&format!("complete body for {}", record.doc_id))
                );
            }
        });
    }

    #[test]
    fn snapshot_excludes_unpublished_statuses_deleted_docs_and_old_revisions() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in [
                "current", "pending", "failed", "skipped", "obsolete", "deleted",
            ] {
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
            storage
                .connection()
                .execute_sync("DELETE FROM documents WHERE doc_id = 'deleted';")
                .unwrap();
            let snapshot = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                )
                .unwrap();
            assert_eq!(snapshot.stats.records, 1);
            assert_eq!(snapshot.records[0].doc_id, "current");
            assert!(
                storage
                    .load_published_embeddings(
                        &cx,
                        "other-tier",
                        &identity("model"),
                        PublishedEmbeddingReadLimits::default(),
                    )
                    .unwrap()
                    .records
                    .is_empty()
            );
        });
    }

    #[test]
    fn count_and_payload_limits_fail_instead_of_returning_a_prefix() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let zero = PublishedEmbeddingReadLimits {
                max_records: 0,
                max_payload_bytes: 0,
            };
            assert!(
                storage
                    .load_published_embeddings(&cx, "model", &identity("model"), zero)
                    .unwrap()
                    .records
                    .is_empty()
            );
            publish(&storage, &queue, "a", "model", vec![1.0, 0.0]);
            publish(&storage, &queue, "b", "model", vec![0.0, 1.0]);
            let bytes = 2 * (serde_json::to_vec(&identity("model")).unwrap().len() + 8);
            let exact = PublishedEmbeddingReadLimits {
                max_records: 2,
                max_payload_bytes: bytes,
            };
            assert_eq!(
                storage
                    .load_published_embeddings(&cx, "model", &identity("model"), exact)
                    .unwrap()
                    .stats
                    .records,
                2
            );
            for (limits, field) in [
                (
                    PublishedEmbeddingReadLimits {
                        max_records: 1,
                        ..exact
                    },
                    "published_embeddings.max_records",
                ),
                (
                    PublishedEmbeddingReadLimits {
                        max_payload_bytes: bytes - 1,
                        ..exact
                    },
                    "published_embeddings.max_payload_bytes",
                ),
                (zero, "published_embeddings.max_records"),
            ] {
                let error = storage
                    .load_published_embeddings(&cx, "model", &identity("model"), limits)
                    .unwrap_err();
                assert!(
                    matches!(error, SearchError::InvalidConfig { field: actual, .. } if actual == field)
                );
                assert!(!storage.connection().in_transaction());
            }
        });
    }

    #[test]
    fn oversized_payload_is_refused_before_it_is_loaded_or_decoded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "oversized", "model", vec![1.0, 0.0]);
            storage
                .connection()
                .execute_with_params_sync(
                    "UPDATE published_embeddings SET identity_json = ?1;",
                    &[text_value(&"private-invalid-json".repeat(5000))],
                )
                .unwrap();
            let error = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &identity("model"),
                    PublishedEmbeddingReadLimits {
                        max_records: 1,
                        max_payload_bytes: 1024,
                    },
                )
                .unwrap_err();
            // Decoding the record instead would report a checksum/JSON error.
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "published_embeddings.max_payload_bytes")
            );
        });
    }

    #[test]
    fn a_current_foreign_producer_or_corrupt_record_fails_the_entire_snapshot() {
        for mutation in [
            None,
            Some("UPDATE published_embeddings SET vector_le = X'0000' WHERE doc_id = 'z';"),
            Some("UPDATE published_embeddings SET record_version = 99 WHERE doc_id = 'z';"),
        ] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "a", "model", vec![1.0, 0.0]);
                publish(
                    &storage,
                    &queue,
                    "z",
                    if mutation.is_some() {
                        "model"
                    } else {
                        "foreign"
                    },
                    vec![0.0, 1.0],
                );
                if let Some(sql) = mutation {
                    storage.connection().execute_sync(sql).unwrap();
                }
                assert!(
                    storage
                        .load_published_embeddings(
                            &cx,
                            "model",
                            &identity("model"),
                            PublishedEmbeddingReadLimits::default(),
                        )
                        .is_err()
                );
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
            let error = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                )
                .unwrap_err();
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
                scan_on(
                    conn,
                    &cx,
                    "model",
                    &identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                    |_| {
                        visited += 1;
                        cx.cancel_with(asupersync::CancelKind::User, Some("stop during read"));
                        Ok(())
                    },
                )
            });
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(visited, 1);
            assert_eq!(queue.queue_depth().unwrap().completed, 2);
            assert!(!storage.connection().in_transaction());
        });
    }

    fn semantic_identity(name: &str) -> EmbeddingIdentityBundleV1 {
        use frankensearch_core::generation::EmbeddingArtifactIdentityV1;

        // Synthetic admission fixture, not evidence of model quality. Keep the
        // production hash-control rejection and rebind every changed identity
        // component, following native_ann/builder/scope.rs's test provider.
        let mut identity = identity(name);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        identity
    }

    fn captured_record(id: &str, values: Vec<f32>) -> PublishedEmbedding {
        PublishedEmbedding {
            doc_id: id.to_owned(),
            embedder_id: "model".to_owned(),
            content_hash: ContentHasher::hash(id),
            source_job_id: 1,
            claim_epoch: 1,
            published_at_ms: 1,
            embedding: IdentityBoundEmbedding {
                values,
                identity: semantic_identity("model"),
            },
        }
    }

    fn captured(rows: &[(&str, [f32; 2])]) -> PublishedEmbeddingSnapshot {
        PublishedEmbeddingSnapshot {
            records: rows
                .iter()
                .map(|(id, values)| captured_record(id, values.to_vec()))
                .collect(),
            // Deliberately not a coverage witness: search must inspect records
            // rather than trust the mutable stats field or invent missing rows.
            stats: PublishedEmbeddingScanStats::default(),
        }
    }

    #[test]
    fn exact_search_normalizes_vectors_keeps_negative_scores_and_resolves_ties() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let snapshot = captured(&[
                ("a-near", [2.0, 0.0]),
                ("b-angled", [100.0, 100.0]),
                ("c-near", [4.0, -0.0]),
                ("d-orthogonal", [0.0, 3.0]),
                ("e-opposite", [-5.0, 0.0]),
            ]);
            let before = snapshot.clone();
            let query =
                BoundQueryEmbedding::new(vec![7.0, 0.0], semantic_identity("model")).unwrap();
            let hits = snapshot.search(&cx, &query, usize::MAX).unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["a-near", "c-near", "b-angled", "d-orthogonal", "e-opposite"]
            );
            for (hit, expected) in
                hits.iter()
                    .zip([1.0, 1.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, -1.0])
            {
                assert!((hit.score - expected).abs() < 1.0e-6);
                let row = &snapshot.records[usize::try_from(hit.index).unwrap()];
                assert_eq!(row.doc_id.as_str(), hit.doc_id.as_str());
                assert_eq!(row.content_hash, ContentHasher::hash(hit.doc_id.as_str()));
            }
            assert_eq!(hits[3].score.to_bits(), 0.0_f32.to_bits());
            assert_eq!(snapshot, before, "scoring does not rewrite the snapshot");
            assert_eq!(
                snapshot.records[2].embedding.values[1].to_bits(),
                (-0.0_f32).to_bits()
            );
        });
    }

    #[test]
    fn bounded_heap_matches_an_independent_full_sort_for_every_limit() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut snapshot = captured(&[]);
            for x in -3_i16..=3 {
                for y in -3_i16..=3 {
                    if x == 0 && y == 0 {
                        continue;
                    }
                    let id = format!("doc-{:03}", snapshot.records.len());
                    snapshot
                        .records
                        .push(captured_record(&id, vec![f32::from(x), f32::from(y)]));
                }
            }
            let query =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("model")).unwrap();
            let mut reference: Vec<_> = snapshot
                .records
                .iter()
                .enumerate()
                .map(|(index, row)| {
                    // Independent 2-D reference: normalized x coordinate from
                    // hypot, not the production reduction or heap comparator.
                    let x = f64::from(row.embedding.values[0]);
                    let y = f64::from(row.embedding.values[1]);
                    #[allow(clippy::cast_possible_truncation)]
                    let score = (x / x.hypot(y)) as f32;
                    VectorHit {
                        index: u32::try_from(index).unwrap(),
                        score: if score == 0.0 { 0.0 } else { score },
                        doc_id: row.doc_id.clone().into(),
                    }
                })
                .collect();
            reference.sort_by(VectorHit::cmp_rank);
            for limit in (0..=reference.len() + 1).chain(std::iter::once(usize::MAX)) {
                let actual = snapshot.search(&cx, &query, limit).unwrap();
                assert_eq!(
                    actual,
                    reference[..limit.min(reference.len())],
                    "limit={limit}"
                );
            }
        });
    }

    #[test]
    fn search_accepts_extreme_finite_query_scales_but_rejects_zero_and_foreign_queries() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let snapshot = captured(&[("aligned", [1.0, 0.0]), ("opposite", [-1.0, 0.0])]);
            for scale in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1), 0.5] {
                let query =
                    BoundQueryEmbedding::new(vec![scale, 0.0], semantic_identity("model")).unwrap();
                let hits = snapshot.search(&cx, &query, 2).unwrap();
                assert_eq!(hits[0].doc_id.as_str(), "aligned");
                assert!((hits[0].score - 1.0).abs() < f32::EPSILON);
                assert!((hits[1].score + 1.0).abs() < f32::EPSILON);
            }
            let zero =
                BoundQueryEmbedding::new(vec![0.0, -0.0], semantic_identity("model")).unwrap();
            assert!(snapshot.search(&cx, &zero, 1).is_err());
            assert!(snapshot.search(&cx, &zero, 0).is_err());
            let foreign =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("foreign")).unwrap();
            assert!(snapshot.search(&cx, &foreign, 1).is_err());
        });
    }

    #[test]
    fn malformed_tail_records_fail_even_when_the_best_hit_is_already_known() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let original = captured(&[("a-best", [1.0, 0.0]), ("z-last", [-1.0, 0.0])]);
            let query =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("model")).unwrap();
            for corruption in ["nan", "zero", "dimension", "producer", "duplicate", "tier"] {
                let mut snapshot = original.clone();
                let tail = &mut snapshot.records[1];
                match corruption {
                    "nan" => tail.embedding.values[0] = f32::NAN,
                    "zero" => tail.embedding.values.fill(0.0),
                    "dimension" => {
                        tail.embedding.values.pop();
                    }
                    "producer" => tail.embedding.identity = semantic_identity("foreign"),
                    "duplicate" => tail.doc_id = "a-best".to_owned(),
                    "tier" => tail.embedder_id = "other-tier".to_owned(),
                    _ => unreachable!(),
                }
                assert!(snapshot.search(&cx, &query, 1).is_err(), "{corruption}");
            }
        });
    }

    #[test]
    fn search_cancellation_precedes_empty_and_zero_limit_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let query =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("model")).unwrap();
            cx.cancel_with(asupersync::CancelKind::User, Some("cancel snapshot search"));
            for snapshot in [captured(&[]), captured(&[("doc", [1.0, 0.0])])] {
                for limit in [0, 1] {
                    assert!(matches!(
                        snapshot.search(&cx, &query, limit),
                        Err(SearchError::Cancelled { .. })
                    ));
                }
            }
        });
    }

    #[test]
    fn snapshot_search_keeps_its_captured_revision_until_reloaded() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish_with_identity(
                &storage,
                &queue,
                "doc",
                semantic_identity("model"),
                vec![1.0, 0.0],
            );
            let old = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &semantic_identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                )
                .unwrap();
            let query =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("model")).unwrap();
            let changed =
                DocumentRecord::new("doc", "new body", ContentHasher::hash("new body"), 8, 1, 2);
            storage.upsert_document(&changed).unwrap();
            storage.set_document_content("doc", "new body").unwrap();
            queue
                .enqueue("doc", "model", &changed.content_hash, 0)
                .unwrap();
            let claim = queue.claim_batch("publisher", 1).unwrap().pop().unwrap();
            let response = IdentityBoundEmbedding {
                values: vec![-1.0, 0.0],
                identity: semantic_identity("model"),
            };
            assert_eq!(
                storage
                    .publish_claimed_embedding(
                        &claim,
                        &changed.content_hash,
                        &response.identity,
                        &response,
                    )
                    .unwrap(),
                ClaimOutcome::Applied(())
            );
            let new = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &semantic_identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                )
                .unwrap();
            assert!((old.search(&cx, &query, 1).unwrap()[0].score - 1.0).abs() < f32::EPSILON);
            assert!((new.search(&cx, &query, 1).unwrap()[0].score + 1.0).abs() < f32::EPSILON);
            assert_ne!(old.records[0].content_hash, new.records[0].content_hash);
            assert_eq!(new.records[0].content_hash, changed.content_hash);
        });
    }

    #[test]
    fn database_worker_output_is_searchable_after_reopen_without_an_external_index() {
        use crate::{InMemoryVectorSink, IngestRequest, PersistentEmbeddingWorker, StorageConfig};
        use frankensearch_core::canonicalize::DefaultCanonicalizer;
        use frankensearch_core::traits::{ModelCategory, SearchFuture};

        struct Probe {
            identity: EmbeddingIdentityBundleV1,
        }
        impl frankensearch_core::Embedder for Probe {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                Ok(&self.identity)
            }
            fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                let values = if text.contains("orchard") {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                };
                Box::pin(async move { Ok(values) })
            }
            fn dimension(&self) -> usize {
                2
            }
            fn id(&self) -> &'static str {
                "model"
            }
            fn model_name(&self) -> &'static str {
                "snapshot-search-probe"
            }
            fn is_semantic(&self) -> bool {
                true
            }
            fn category(&self) -> ModelCategory {
                ModelCategory::StaticEmbedder
            }
        }

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let config = StorageConfig {
                db_path: directory.path().join("search.sqlite3"),
                ..StorageConfig::default()
            };
            let sink = Arc::new(InMemoryVectorSink::default());
            {
                let storage = Arc::new(Storage::open(config.clone()).unwrap());
                let worker = PersistentEmbeddingWorker::new(
                    storage,
                    JobQueueConfig::default(),
                    Arc::new(DefaultCanonicalizer::default()),
                    Arc::new(Probe {
                        identity: semantic_identity("model"),
                    }),
                    sink.clone(),
                )
                .unwrap()
                .with_database_publication();
                worker
                    .runner()
                    .ingest(IngestRequest::new("fruit", "apple orchard"))
                    .unwrap();
                worker
                    .runner()
                    .ingest(IngestRequest::new("systems", "database transaction"))
                    .unwrap();
                let processed = worker.process_batch(&cx, "publisher").await.unwrap();
                assert_eq!(processed.jobs_completed, 2);
                assert_eq!(processed.jobs_failed + processed.jobs_suppressed, 0);
                assert!(sink.entries().is_empty());
            }
            let storage = Storage::open(config).unwrap();
            let snapshot = storage
                .load_published_embeddings(
                    &cx,
                    "model",
                    &semantic_identity("model"),
                    PublishedEmbeddingReadLimits::default(),
                )
                .unwrap();
            let query =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("model")).unwrap();
            let hits = snapshot.search(&cx, &query, 1).unwrap();
            assert_eq!(snapshot.stats.records, 2);
            assert_eq!(hits[0].doc_id.as_str(), "fruit");
            let row = &snapshot.records[usize::try_from(hits[0].index).unwrap()];
            assert_eq!(row.content_hash, ContentHasher::hash("apple orchard"));
            assert_eq!(
                storage
                    .get_document_content(&row.doc_id)
                    .unwrap()
                    .as_deref(),
                Some("apple orchard")
            );
            assert!(sink.entries().is_empty());
        });
    }

    #[test]
    fn hash_control_queries_cannot_be_reported_as_semantic_search() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            assert_eq!(
                query.identity().space.kind,
                EmbeddingSpaceKindV1::HashControl
            );
            for limit in [0, 1] {
                let error = captured(&[]).search(&cx, &query, limit).unwrap_err();
                assert!(error.to_string().contains("hash-control"));
            }
        });
    }
}
