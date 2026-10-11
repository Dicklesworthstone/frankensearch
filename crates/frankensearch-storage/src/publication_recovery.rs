//! Resumable recovery of database publication coverage from retained input.
//!
//! Catalog `embedded` status alone is not proof that a database vector exists:
//! the document may have been processed through an external sink. Recovery
//! schedules missing/obsolete publications without resurrecting failed work,
//! replacing live attempts, or reading possibly changed source files.

use std::time::{SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use frankensearch_core::generation::QuantizationFormat;
use frankensearch_core::{EmbeddingIdentityBundleV1, SearchError, SearchResult};
use fsqlite::{AsyncConnection, Row};
use fsqlite_types::value::SqliteValue;

use crate::Storage;
use crate::connection::map_storage_error;
use crate::document::get_document_inner;
use crate::document_content::read_document_content;
use crate::job_queue::{EnqueueOutcome, EnqueueRequest, enqueue_inner};
use crate::schema::row_i64;

const PAGE_SQL: &str = "SELECT doc_id FROM documents \
    WHERE (?1 IS NULL OR doc_id COLLATE BINARY > ?1) \
    ORDER BY doc_id COLLATE BINARY LIMIT ?2;";
const STATUS_SQL: &str = "SELECT status FROM embedding_status \
    WHERE doc_id = ?1 AND embedder_id = ?2;";
const PUBLICATION_SQL: &str = "SELECT content_hash, identity_json FROM published_embeddings \
    WHERE doc_id = ?1 AND embedder_id = ?2;";
const ACTIVE_SQL: &str = "SELECT content_hash, status FROM embedding_jobs \
    WHERE doc_id = ?1 AND embedder_id = ?2 AND status IN ('pending', 'processing');";
const PENDING_COUNT_SQL: &str = "SELECT COUNT(*) FROM embedding_jobs WHERE status = 'pending';";
const MARK_PENDING_SQL: &str = "INSERT INTO embedding_status \
    (doc_id, embedder_id, embedder_revision, status, embedded_at, error_message, retry_count) \
    VALUES (?1, ?2, NULL, 'pending', NULL, NULL, 0) \
    ON CONFLICT(doc_id, embedder_id) DO UPDATE SET \
      embedder_revision = NULL, status = 'pending', embedded_at = NULL, \
      error_message = NULL, retry_count = 0;";

/// Admission and scheduling limits for one atomic recovery page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicationRecoveryConfig {
    /// Documents inspected per page, in the inclusive range 1..=256.
    /// This is not a byte/RSS limit; a retained document may be large.
    pub page_size: usize,
    /// Ceiling for ALL pending jobs, including delayed jobs and other tiers.
    /// A replacement consumes its existing pending slot rather than another.
    pub max_pending_jobs: usize,
    /// Persisted retry budget for newly scheduled attempts only.
    pub max_retries: u32,
    /// Queue priority for newly scheduled attempts only.
    pub priority: i32,
}

impl Default for PublicationRecoveryConfig {
    fn default() -> Self {
        Self {
            page_size: 128,
            max_pending_jobs: 10_000,
            max_retries: 3,
            priority: 0,
        }
    }
}

/// Result of a committed page, not a claim of complete semantic coverage.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PublicationRecoveryPage {
    pub examined_documents: usize,
    pub enqueued_jobs: usize,
    /// Current revision and producer metadata already match. This operation
    /// does NOT verify vector checksums; use the publication reader for that.
    pub already_published: usize,
    /// Current-revision or legacy unknown-hash work preserved without mutation.
    pub already_active: usize,
    /// Failed/skipped catalog entries deliberately left alone.
    pub terminal_documents: usize,
    /// Documents requiring re-ingestion of their complete canonical input.
    /// No job or catalog change is made for these IDs.
    pub missing_input: Vec<String>,
    /// Pass this cursor to the next call. `None` means this page observed the
    /// end of the catalog, not that all documents successfully embedded.
    pub next_after: Option<String>,
}

impl Storage {
    /// Schedule missing or obsolete database publications from retained input.
    ///
    /// Call once per configured embedding tier when switching from an external
    /// sink, restoring publication coverage, or changing a producer under the
    /// same operational embedder ID. Drain with the intended database-publishing
    /// worker after scheduling. Merely selecting worker database mode cannot
    /// migrate documents already marked embedded by an external sink.
    ///
    /// The page, input verification, enqueue and catalog transition share one
    /// immediate transaction. Current/legacy active jobs and their retry delays
    /// are preserved; obsolete pending work may be replaced, but an obsolete
    /// processing claim keeps its fence. Failed and deliberately skipped
    /// documents are never resurrected. Missing input is reported for explicit
    /// re-ingestion; corrupt retained input and database errors abort the page.
    ///
    /// Existing current publication metadata is compared with the FULL expected
    /// identity, not its model name or dimension. This is a coverage repair,
    /// not a vector integrity scrub. Already-present coordinates are not read,
    /// blessed, repaired, or made searchable by this operation.
    ///
    /// Cancellation before commit, queue capacity exhaustion or any error rolls
    /// back the whole page. Retry the SAME input cursor after making room; do
    /// not advance it on error. State is rebuilt for storage transaction retries.
    /// There are no provider callbacks, external file reads, or awaits in the
    /// transaction. Instance-local queue/runner counters are not updated.
    ///
    /// Pages are independent snapshots, not an ingestion barrier. Repeat from
    /// `None` to catch concurrent changes behind the cursor, or active attempts
    /// that finish later under an older producer. Jobs bind document revisions;
    /// the caller must run the intended provider, and readers still verify it.
    #[allow(clippy::too_many_lines)]
    pub fn reconcile_published_embeddings(
        &self,
        cx: &Cx,
        embedder_id: &str,
        expected_identity: &EmbeddingIdentityBundleV1,
        after: Option<&str>,
        config: PublicationRecoveryConfig,
    ) -> SearchResult<PublicationRecoveryPage> {
        checkpoint(cx)?;
        validate_request(embedder_id, expected_identity, config)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_err(map_storage_error)?;
        let now_ms = i64::try_from(now.as_millis()).map_err(map_storage_error)?;
        self.immediate_transaction(|conn| {
            checkpoint(cx)?;
            let rows = conn.query_with_params_sync(
                PAGE_SQL,
                &[
                    after.map_or(SqliteValue::Null, text),
                    SqliteValue::Integer(i64::try_from(config.page_size + 1).map_err(map_storage_error)?),
                ],
            ).map_err(map_storage_error)?;
            let mut page = PublicationRecoveryPage::default();
            let mut pending = pending_count(conn)?;
            let mut previous = after;
            for row in rows.iter().take(config.page_size) {
                checkpoint(cx)?;
                let doc_id = row_text(row, 0)?;
                if doc_id.trim().is_empty() || previous.is_some_and(|id| doc_id <= id) {
                    return Err(invalid("cursor", "invalid document order"));
                }
                previous = Some(doc_id);
                page.examined_documents += 1;
                let params = [text(doc_id), text(embedder_id)];
                let statuses = conn.query_with_params_sync(STATUS_SQL, &params).map_err(map_storage_error)?;
                let status = statuses.first().map(|row| row_text(row, 0)).transpose()?;
                match status {
                    Some("failed" | "skipped") => {
                        page.terminal_documents += 1;
                        continue;
                    }
                    None | Some("pending" | "embedded") => {}
                    _ => return Err(invalid("catalog", "unknown embedding status")),
                }
                let document = get_document_inner(conn, doc_id)?
                    .ok_or_else(|| invalid("catalog", "document disappeared inside recovery transaction"))?;
                if status == Some("embedded")
                    && publication_matches(conn, &params, &document.content_hash, expected_identity)?
                {
                    page.already_published += 1;
                    continue;
                }
                let active = conn.query_with_params_sync(ACTIVE_SQL, &params).map_err(map_storage_error)?;
                let mut covered = false;
                let mut replaces_pending = false;
                for attempt in &active {
                    let hash = optional_hash(attempt, 0)?;
                    covered |= hash.is_none_or(|hash| hash == document.content_hash);
                    replaces_pending |= row_text(attempt, 1)? == "pending";
                }
                if covered {
                    page.already_active += 1;
                    continue;
                }
                // Complete input is verified before changing catalog authority.
                // Never turn a 400-character preview into a replacement vector.
                if read_document_content(conn, &document)?.is_none() {
                    page.missing_input.push(doc_id.to_owned());
                    continue;
                }
                checkpoint(cx)?;
                let projected = pending.checked_add(usize::from(!replaces_pending))
                    .ok_or_else(|| invalid("capacity", "pending count overflow"))?;
                if projected > config.max_pending_jobs {
                    return Err(SearchError::QueueFull {
                        pending,
                        capacity: config.max_pending_jobs,
                    });
                }
                let request = EnqueueRequest::new(doc_id, embedder_id, document.content_hash, config.priority);
                match enqueue_inner(conn, &request, now_ms, config.max_retries)? {
                    EnqueueOutcome::Inserted | EnqueueOutcome::Replaced => {}
                    // The immediate transaction excludes a new claimant between
                    // the active-work check and enqueue. Do not mask a violation.
                    _ => return Err(invalid("enqueue", "unexpected recovery enqueue outcome")),
                }
                conn.execute_with_params_sync(MARK_PENDING_SQL, &params).map_err(map_storage_error)?;
                pending = projected;
                page.enqueued_jobs += 1;
                checkpoint(cx)?;
            }
            if rows.len() > config.page_size {
                page.next_after = previous.map(str::to_owned);
            }
            checkpoint(cx)?;
            Ok(page)
        })
    }
}

fn validate_request(
    embedder_id: &str,
    identity: &EmbeddingIdentityBundleV1,
    config: PublicationRecoveryConfig,
) -> SearchResult<()> {
    if embedder_id.trim().is_empty() || frankensearch_core::is_hash_generation_id(embedder_id) {
        return Err(invalid("embedder_id", "requires a nonempty, persisted embedder ID"));
    }
    if !(1..=256).contains(&config.page_size) {
        return Err(invalid("page_size", "must be in 1..=256"));
    }
    identity.validate()?;
    if identity.storage.quantization != QuantizationFormat::F32
        || !identity.storage.format.starts_with("in-memory-")
        || !matches!(identity.storage.endianness.as_str(), "native-f32-values" | "native-test-only")
    {
        return Err(invalid("identity", "requires the producer's in-memory f32 identity"));
    }
    Ok(())
}

fn publication_matches(
    conn: &AsyncConnection,
    params: &[SqliteValue; 2],
    hash: &[u8; 32],
    expected: &EmbeddingIdentityBundleV1,
) -> SearchResult<bool> {
    let rows = conn.query_with_params_sync(PUBLICATION_SQL, params).map_err(map_storage_error)?;
    let Some(row) = rows.first() else { return Ok(false); };
    let stored_hash = optional_hash(row, 0)?
        .ok_or_else(|| invalid("publication", "missing publication revision"))?;
    if &stored_hash != hash {
        return Ok(false);
    }
    let identity: EmbeddingIdentityBundleV1 = serde_json::from_str(row_text(row, 1)?)
        .map_err(|_| invalid("publication", "malformed publication identity"))?;
    identity.validate()?;
    Ok(&identity == expected)
}

fn pending_count(conn: &AsyncConnection) -> SearchResult<usize> {
    let rows = conn.query_sync(PENDING_COUNT_SQL).map_err(map_storage_error)?;
    let row = rows.first().ok_or_else(|| invalid("capacity", "missing pending count"))?;
    usize::try_from(row_i64(row, 0, "pending count")?).map_err(map_storage_error)
}

fn optional_hash(row: &Row, index: usize) -> SearchResult<Option<[u8; 32]>> {
    match row.get(index) {
        Some(SqliteValue::Null) => Ok(None),
        Some(SqliteValue::Blob(bytes)) => bytes.as_ref().try_into()
            .map(Some).map_err(|_| invalid("revision", "expected a 32-byte content hash")),
        _ => Err(invalid("revision", "invalid content hash type")),
    }
}

fn row_text(row: &Row, index: usize) -> SearchResult<&str> {
    match row.get(index) {
        Some(SqliteValue::Text(value)) => Ok(value.as_ref()),
        _ => Err(invalid("row", "expected text column")),
    }
}

fn text(value: &str) -> SqliteValue { SqliteValue::Text(value.to_owned().into()) }

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.publication_recovery".to_owned(),
        reason: error.to_string(),
    })
}

fn invalid(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("publication_recovery.{field}"),
        value: "rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync, clippy::too_many_lines)]

    use std::sync::{Arc, Mutex, OnceLock};
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use frankensearch_core::{
        Canonicalizer, Embedder, IdentityBoundEmbedding, ModelCategory, SearchFuture,
    };

    use super::*;
    use crate::{
        ClaimOutcome, ContentHasher, DocumentRecord, InMemoryVectorSink, IngestAction,
        IngestRequest, JobQueueConfig, PersistentEmbeddingWorker, PersistentJobQueue,
        PublishedEmbeddingReadLimits, StorageConfig, WorkerDrainOutcome,
    };

    const MODEL: &str = "recovery-model";

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model(name, 2)
    }

    fn fixture() -> (Arc<Storage>, PersistentJobQueue) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        (storage, queue)
    }

    fn seed(storage: &Storage, id: &str, body: &str, retain: bool) -> [u8; 32] {
        let hash = ContentHasher::hash(body);
        let doc = DocumentRecord::new(id, body, hash, body.chars().count(), 1, 1);
        storage.upsert_document(&doc).unwrap();
        if retain {
            storage.set_document_content(id, body).unwrap();
        }
        hash
    }

    fn publish(storage: &Storage, queue: &PersistentJobQueue, id: &str, producer: &str) {
        let hash = seed(storage, id, "retained body", true);
        assert!(queue.enqueue(id, MODEL, &hash, 0).unwrap());
        let claim = queue.claim_batch("publisher", 1).unwrap().pop().unwrap();
        assert_eq!(claim.doc_id, id);
        let value = IdentityBoundEmbedding { values: vec![1.0, -0.0], identity: identity(producer) };
        assert_eq!(
            storage.publish_claimed_embedding(&claim, &hash, &value.identity, &value).unwrap(),
            ClaimOutcome::Applied(())
        );
    }

    fn recover(storage: &Storage, cx: &Cx) -> PublicationRecoveryPage {
        storage.reconcile_published_embeddings(
            cx, MODEL, &identity(MODEL), None, PublicationRecoveryConfig::default(),
        ).unwrap()
    }

    fn rows(storage: &Storage, sql: &str, width: usize) -> Vec<Vec<SqliteValue>> {
        storage.connection().query_sync(sql).unwrap().iter().map(|row| {
            (0..width).map(|column| row.get(column).unwrap().clone()).collect()
        }).collect()
    }

    #[test]
    fn schedules_only_missing_or_obsolete_publications_and_is_idempotent() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "current", MODEL);
            publish(&storage, &queue, "old-revision", MODEL);
            seed(&storage, "old-revision", "replacement body", true);
            seed(&storage, "external", "retained external body", true);
            storage.mark_embedded("external", MODEL).unwrap();
            seed(&storage, "unqueued", "retained catalog only", true);
            let before_current = storage.get_published_embedding("current", MODEL, &identity(MODEL)).unwrap();
            let page = recover(&storage, &cx);
            assert_eq!(page.examined_documents, 4);
            assert_eq!(page.enqueued_jobs, 3);
            assert_eq!(page.already_published, 1);
            assert!(page.missing_input.is_empty());
            assert_eq!(queue.queue_depth().unwrap().pending, 3);
            assert_eq!(storage.get_published_embedding("current", MODEL, &identity(MODEL)).unwrap(), before_current);
            assert!(storage.get_published_embedding("old-revision", MODEL, &identity(MODEL)).unwrap().is_none());
            let before = rows(&storage, "SELECT * FROM embedding_jobs ORDER BY job_id;", 14);
            let again = recover(&storage, &cx);
            assert_eq!(again.enqueued_jobs, 0);
            assert_eq!(again.already_active, 3);
            assert_eq!(rows(&storage, "SELECT * FROM embedding_jobs ORDER BY job_id;", 14), before);
        });
    }

    #[test]
    fn producer_change_under_the_same_operational_id_schedules_replacement() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "upgrade", "old-producer");
            let before = rows(&storage, "SELECT * FROM published_embeddings;", 10);
            let page = recover(&storage, &cx);
            assert_eq!(page.enqueued_jobs, 1);
            assert_eq!(rows(&storage, "SELECT * FROM published_embeddings;", 10), before);
            assert!(storage.get_published_embedding("upgrade", MODEL, &identity("old-producer")).unwrap().is_none());
            let claim = queue.claim_batch("new-producer", 1).unwrap().pop().unwrap();
            let response = IdentityBoundEmbedding { values: vec![0.0, 1.0], identity: identity(MODEL) };
            assert_eq!(storage.publish_claimed_embedding(
                &claim, &claim.content_hash.unwrap(), &response.identity, &response,
            ).unwrap(), ClaimOutcome::Applied(()));
            assert_eq!(recover(&storage, &cx).already_published, 1);
        });
    }

    #[test]
    fn preserves_terminal_catalog_entries_and_live_or_delayed_attempts() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in ["failed", "skipped", "delayed", "processing", "legacy"] {
                seed(&storage, id, "retained body", true);
            }
            storage.mark_failed("failed", MODEL, "retry budget exhausted").unwrap();
            storage.mark_skipped("skipped", MODEL, "intentional").unwrap();
            let hash = ContentHasher::hash("retained body");
            queue.enqueue("processing", MODEL, &hash, 0).unwrap();
            let claim = queue.claim_batch("live", 1).unwrap().pop().unwrap();
            queue.enqueue("delayed", MODEL, &hash, 0).unwrap();
            queue.enqueue("legacy", MODEL, &hash, 0).unwrap();
            storage.connection().execute_sync(
                "UPDATE embedding_jobs SET retry_count = 2, submitted_at = 9223372036854775807 WHERE status = 'pending';",
            ).unwrap();
            storage.connection().execute_sync("UPDATE embedding_jobs SET content_hash = NULL WHERE doc_id = 'legacy';").unwrap();
            let before_jobs = rows(&storage, "SELECT * FROM embedding_jobs ORDER BY job_id;", 14);
            let before_status = rows(&storage, "SELECT * FROM embedding_status ORDER BY doc_id;", 7);
            let page = recover(&storage, &cx);
            assert_eq!(page.terminal_documents, 2);
            assert_eq!(page.already_active, 3);
            assert_eq!(page.enqueued_jobs, 0);
            assert_eq!(rows(&storage, "SELECT * FROM embedding_jobs ORDER BY job_id;", 14), before_jobs);
            assert_eq!(rows(&storage, "SELECT * FROM embedding_status ORDER BY doc_id;", 7), before_status);
            assert!(matches!(queue.check_claim(&claim, None).unwrap(), ClaimOutcome::Applied(_)));
        });
    }

    #[test]
    fn obsolete_processing_fence_survives_while_a_current_replacement_is_queued() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let old_hash = seed(&storage, "changing", "old body", true);
            queue.enqueue("changing", MODEL, &old_hash, 0).unwrap();
            let old = queue.claim_batch("old", 1).unwrap().pop().unwrap();
            let new_hash = seed(&storage, "changing", "new body", true);
            assert_eq!(recover(&storage, &cx).enqueued_jobs, 1);
            assert_eq!(queue.queue_depth().unwrap().processing, 1);
            assert!(queue.claim_batch("new", 1).unwrap().is_empty());
            assert_eq!(queue.complete(&old, &old_hash).unwrap(), ClaimOutcome::Superseded);
            let replacement = queue.claim_batch("new", 1).unwrap().pop().unwrap();
            assert_eq!(replacement.content_hash, Some(new_hash));
            assert_ne!(replacement.job_id, old.job_id);
            assert_eq!(queue.fail(&old, &old_hash, "late").unwrap(), ClaimOutcome::LostClaim);
        });
    }

    #[test]
    fn missing_full_input_is_reported_without_using_a_complete_preview() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            seed(&storage, "missing", "complete but unretained preview", false);
            storage.mark_embedded("missing", MODEL).unwrap();
            let page = recover(&storage, &cx);
            assert_eq!(page.missing_input, ["missing"]);
            assert_eq!(page.enqueued_jobs, 0);
            assert_eq!(queue.queue_depth().unwrap().pending, 0);
            assert_eq!(storage.count_by_status(MODEL).unwrap().embedded, 1);
            storage.set_document_content("missing", "complete but unretained preview").unwrap();
            assert_eq!(recover(&storage, &cx).enqueued_jobs, 1);
        });
    }

    #[test]
    fn page_capacity_failure_rolls_back_and_delayed_jobs_count_against_the_limit() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in ["a", "b"] {
                seed(&storage, id, "body", true);
                storage.mark_embedded(id, MODEL).unwrap();
            }
            let other = seed(&storage, "z-other-tier", "body", true);
            queue.enqueue("z-other-tier", "other-tier", &other, 0).unwrap();
            storage.connection().execute_sync("UPDATE embedding_jobs SET submitted_at = 9223372036854775807;").unwrap();
            let before = rows(&storage, "SELECT * FROM embedding_status ORDER BY doc_id;", 7);
            let error = storage.reconcile_published_embeddings(
                &cx, MODEL, &identity(MODEL), None,
                PublicationRecoveryConfig { page_size: 2, max_pending_jobs: 2, ..PublicationRecoveryConfig::default() },
            ).unwrap_err();
            assert!(matches!(error, SearchError::QueueFull { pending: 2, capacity: 2 }));
            assert_eq!(queue.queue_depth().unwrap().pending, 1);
            assert_eq!(rows(&storage, "SELECT * FROM embedding_status ORDER BY doc_id;", 7), before);
            assert!(!storage.connection().in_transaction());
            // Retrying the original cursor with a smaller page makes progress.
            let page = storage.reconcile_published_embeddings(
                &cx, MODEL, &identity(MODEL), None,
                PublicationRecoveryConfig { page_size: 1, max_pending_jobs: 2, ..PublicationRecoveryConfig::default() },
            ).unwrap();
            assert_eq!(page.next_after.as_deref(), Some("a"));
            assert_eq!(page.enqueued_jobs, 1);
        });
    }

    #[test]
    fn obsolete_pending_replacement_reuses_capacity_and_preserves_history() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "doc", MODEL);
            let history = rows(&storage, "SELECT * FROM embedding_jobs WHERE status = 'completed';", 14);
            let old = ContentHasher::hash("retained body");
            queue.enqueue("doc", MODEL, &old, 0).unwrap();
            let new = seed(&storage, "doc", "new body", true);
            let page = storage.reconcile_published_embeddings(
                &cx, MODEL, &identity(MODEL), None,
                PublicationRecoveryConfig { max_pending_jobs: 1, max_retries: 7, priority: 42, ..PublicationRecoveryConfig::default() },
            ).unwrap();
            assert_eq!(page.enqueued_jobs, 1);
            assert_eq!(queue.queue_depth().unwrap().pending, 1);
            assert_eq!(rows(&storage, "SELECT * FROM embedding_jobs WHERE status = 'completed';", 14), history);
            let claim = queue.claim_batch("new", 1).unwrap().pop().unwrap();
            assert_eq!(claim.content_hash, Some(new));
            assert_eq!(claim.max_retries, 7);
            assert_eq!(claim.priority, 42);
            assert_eq!(claim.retry_count, 0);
        });
    }

    #[test]
    fn corruption_after_an_earlier_enqueue_rolls_back_the_whole_page() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in ["a", "z"] {
                seed(&storage, id, "retained body", true);
                storage.mark_embedded(id, MODEL).unwrap();
            }
            storage.connection().execute_sync("UPDATE document_contents SET canonical_text = 'corrupt' WHERE doc_id = 'z';").unwrap();
            assert!(storage.reconcile_published_embeddings(
                &cx, MODEL, &identity(MODEL), None, PublicationRecoveryConfig::default(),
            ).is_err());
            assert_eq!(queue.queue_depth().unwrap().pending, 0);
            assert_eq!(storage.count_by_status(MODEL).unwrap().embedded, 2);
            assert!(!storage.connection().in_transaction());
        });
    }

    #[test]
    fn binary_keyset_pages_cover_unicode_and_nul_ids_once() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let mut ids = vec!["z", "a\0tail", "a", "é", "数据"];
            for id in &ids { seed(&storage, id, "body", true); }
            ids.sort_unstable();
            let mut after = None;
            let mut examined = 0;
            let mut enqueued = 0;
            let mut cursors = Vec::new();
            loop {
                let page = storage.reconcile_published_embeddings(
                    &cx, MODEL, &identity(MODEL), after.as_deref(),
                    PublicationRecoveryConfig { page_size: 2, ..PublicationRecoveryConfig::default() },
                ).unwrap();
                examined += page.examined_documents;
                enqueued += page.enqueued_jobs;
                after = page.next_after;
                let Some(cursor) = after.as_ref() else { break; };
                cursors.push(cursor.clone());
            }
            assert_eq!(examined, ids.len());
            assert_eq!(enqueued, ids.len());
            assert_eq!(cursors, [ids[1], ids[3]]);
            assert_eq!(queue.queue_depth().unwrap().pending, ids.len());
            assert_eq!(recover(&storage, &cx).already_active, ids.len());
        });
    }

    #[test]
    fn cancelled_and_invalid_requests_do_not_start_a_transaction() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, _) = fixture();
            seed(&storage, "doc", "body", true);
            let before = storage.metrics_snapshot();
            for size in [0, 257, usize::MAX] {
                assert!(storage.reconcile_published_embeddings(
                    &cx, MODEL, &identity(MODEL), None,
                    PublicationRecoveryConfig { page_size: size, ..PublicationRecoveryConfig::default() },
                ).is_err());
            }
            cx.cancel_with(asupersync::CancelKind::User, Some("stop recovery"));
            assert!(matches!(storage.reconcile_published_embeddings(
                &cx, MODEL, &identity(MODEL), None, PublicationRecoveryConfig::default(),
            ), Err(SearchError::Cancelled { .. })));
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }

    struct Canonical;
    impl Canonicalizer for Canonical {
        fn canonicalize(&self, text: &str) -> String { text.to_owned() }
        fn canonicalize_query(&self, query: &str) -> String { query.to_owned() }
    }

    struct Probe(Arc<Mutex<Vec<String>>>);
    impl Embedder for Probe {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            static IDENTITY: OnceLock<EmbeddingIdentityBundleV1> = OnceLock::new();
            Ok(IDENTITY.get_or_init(|| identity(MODEL)))
        }
        fn embed<'a>(&'a self, _cx: &'a Cx, value: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.0.lock().unwrap().push(value.to_owned());
            Box::pin(async { Ok(vec![1.0, 0.0]) })
        }
        fn dimension(&self) -> usize { 2 }
        fn id(&self) -> &'static str { MODEL }
        fn model_name(&self) -> &'static str { MODEL }
        fn is_semantic(&self) -> bool { true }
        fn category(&self) -> ModelCategory { ModelCategory::StaticEmbedder }
    }

    #[test]
    fn external_sink_migration_reopens_and_drains_into_complete_database_vectors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let config = StorageConfig { db_path: dir.path().join("recovery.sqlite3"), ..StorageConfig::default() };
            let observed = Arc::new(Mutex::new(Vec::new()));
            let sink = Arc::new(InMemoryVectorSink::default());
            let input = "Unicode café 数据 full source. ".repeat(40);
            {
                let storage = Arc::new(Storage::open(config.clone()).unwrap());
                let worker = PersistentEmbeddingWorker::new(
                    storage.clone(), JobQueueConfig::default(), Arc::new(Canonical),
                    Arc::new(Probe(observed.clone())), sink.clone(),
                ).unwrap();
                worker.runner().ingest(IngestRequest::new("migrate", &input)).unwrap();
                assert_eq!(worker.process_batch(&cx, "external").await.unwrap().jobs_completed, 1);
                let worker = worker.with_database_publication();
                assert_eq!(worker.runner().ingest(IngestRequest::new("migrate", &input)).unwrap().action, IngestAction::Unchanged);
                assert_eq!(worker.queue().queue_depth().unwrap().pending, 0);
                assert!(storage.get_published_embedding("migrate", MODEL, &identity(MODEL)).unwrap().is_none());
                assert_eq!(recover(&storage, &cx).enqueued_jobs, 1);
            }
            let storage = Arc::new(Storage::open(config).unwrap());
            let worker = PersistentEmbeddingWorker::new(
                storage.clone(), JobQueueConfig::default(), Arc::new(Canonical),
                Arc::new(Probe(observed.clone())), sink.clone(),
            ).unwrap().with_database_publication();
            let outcome = worker.drain(&cx, "database", &AtomicBool::new(false), Duration::from_secs(5)).await.unwrap();
            assert!(matches!(outcome, WorkerDrainOutcome::Drained { .. }));
            assert_eq!(outcome.report().work.jobs_completed, 1);
            assert_eq!(sink.entries().len(), 1, "recovery must never call the old external sink");
            assert_eq!(observed.lock().unwrap().as_slice(), [input.clone(), input]);
            let snapshot = storage.load_published_embeddings(
                &cx, MODEL, &identity(MODEL), PublishedEmbeddingReadLimits::default(),
            ).unwrap();
            assert_eq!(snapshot.records.len(), 1);
            assert_eq!(snapshot.records[0].doc_id, "migrate");
            assert_eq!(recover(&storage, &cx).already_published, 1);
        });
    }
}
