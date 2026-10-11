//! Exact search with source text retained beside the vectors it describes.
//!
//! Load through `Storage::load_published_search_snapshot`, then search the
//! immutable value. Results borrow their source from that value, never from a
//! later catalog lookup or a changed external file. This is a bounded exact
//! backend for one publication tier, not ANN or a live-refreshing index.

use asupersync::Cx;
use frankensearch_core::types::{BoundQueryEmbedding, VectorHit};
use frankensearch_core::{EmbeddingIdentityBundleV1, SearchError, SearchResult};
use fsqlite::AsyncConnection;
use fsqlite_types::value::SqliteValue;

use crate::connection::map_storage_error;
use crate::document::get_document_inner;
use crate::document_content::read_document_content;
use crate::schema::row_i64;
use crate::{
    DocumentRecord, PublishedEmbedding, PublishedEmbeddingReadLimits,
    PublishedEmbeddingScanStats, PublishedEmbeddingSnapshot,
};

// Measure stored bytes, not SQLite's character count (which also stops at NUL
// for TEXT). No source/metadata payload is fetched before this admission step.
const SOURCE_LENGTHS_SQL: &str = "SELECT \
    LENGTH(CAST(c.canonical_text AS BLOB)), \
    LENGTH(CAST(d.content_preview AS BLOB)), \
    LENGTH(CAST(d.source_path AS BLOB)), \
    LENGTH(CAST(d.metadata_json AS BLOB)) \
    FROM documents d LEFT JOIN document_contents c ON c.doc_id = d.doc_id \
    WHERE d.doc_id = ?1 AND d.content_hash = ?2;";

/// Independent budgets for vectors and the source payload retained beside them.
///
/// Limits reject an incomplete result rather than silently truncating a corpus.
/// Stored source bytes include canonical text, preview, path, and metadata JSON.
/// They are not an RSS or SQL-work ceiling: decoded values, IDs, containers, and
/// the database driver's own caches consume additional memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishedSearchReadLimits {
    pub embeddings: PublishedEmbeddingReadLimits,
    pub max_source_bytes: usize,
}

impl Default for PublishedSearchReadLimits {
    fn default() -> Self {
        Self {
            embeddings: PublishedEmbeddingReadLimits::default(),
            max_source_bytes: 64 * 1024 * 1024,
        }
    }
}

/// One catalog record and the exact canonical text verified against its hash.
///
/// The text is embedding input, not necessarily the original raw file: the
/// ingestion canonicalizer may have deliberately transformed or shortened it.
#[derive(Clone)]
pub struct PublishedSearchDocument {
    catalog: DocumentRecord,
    canonical_text: String,
}

impl std::fmt::Debug for PublishedSearchDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedSearchDocument")
            .field("doc_id", &self.catalog.doc_id)
            .field("canonical_bytes", &self.canonical_text.len())
            .finish_non_exhaustive()
    }
}

impl PublishedSearchDocument {
    /// Metadata captured in the same transaction as the publication and text.
    #[must_use]
    pub const fn catalog(&self) -> &DocumentRecord {
        &self.catalog
    }

    /// Complete verified canonical embedding input, never a preview fallback.
    #[must_use]
    pub fn canonical_text(&self) -> &str {
        &self.canonical_text
    }
}

/// Immutable, source-coherent corpus for one exact-search tier.
///
/// All fields are read-only to callers; vector positions cannot be rebound to
/// other documents. Later updates/deletions do not mutate this captured value.
/// Holding it is not a revocation mechanism: reload or drop old snapshots when
/// application policy requires new membership or metadata.
#[derive(Clone)]
pub struct PublishedSearchSnapshot {
    pub(crate) vectors: PublishedEmbeddingSnapshot,
    pub(crate) documents: Vec<PublishedSearchDocument>,
    pub(crate) identity: EmbeddingIdentityBundleV1,
    pub(crate) embedder_id: String,
    pub(crate) source_payload_bytes: usize,
}

impl std::fmt::Debug for PublishedSearchSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedSearchSnapshot")
            .field("embedder_id", &self.embedder_id)
            .field("identity", &self.identity.fingerprint())
            .field("records", &self.documents.len())
            .field("source_payload_bytes", &self.source_payload_bytes)
            .finish_non_exhaustive()
    }
}

/// A ranked hit borrowing its publication and verified source from one owner.
#[derive(Debug)]
pub struct PublishedSearchHit<'a> {
    hit: VectorHit,
    publication: &'a PublishedEmbedding,
    document: &'a PublishedSearchDocument,
}

impl PublishedSearchHit<'_> {
    #[must_use]
    pub const fn hit(&self) -> &VectorHit {
        &self.hit
    }

    #[must_use]
    pub const fn publication(&self) -> &PublishedEmbedding {
        self.publication
    }

    #[must_use]
    pub const fn document(&self) -> &PublishedSearchDocument {
        self.document
    }
}

impl PublishedSearchSnapshot {
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    #[must_use]
    pub fn embedder_id(&self) -> &str {
        &self.embedder_id
    }

    #[must_use]
    pub const fn identity(&self) -> &EmbeddingIdentityBundleV1 {
        &self.identity
    }

    #[must_use]
    pub fn documents(&self) -> &[PublishedSearchDocument] {
        &self.documents
    }

    #[must_use]
    pub const fn vector_stats(&self) -> PublishedEmbeddingScanStats {
        self.vectors.stats
    }

    #[must_use]
    pub const fn source_payload_bytes(&self) -> usize {
        self.source_payload_bytes
    }

    /// Exact cosine search with source hydration from this same captured owner.
    ///
    /// The complete query contract must match even for an empty corpus or a
    /// zero limit. Ordinary vector validation, hash-control refusal, deterministic
    /// ties, and coordinate-level cancellation use the existing exact backend.
    /// Results require no subsequent database or filesystem access.
    pub fn search(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        limit: usize,
    ) -> SearchResult<Vec<PublishedSearchHit<'_>>> {
        checkpoint(cx)?;
        if query.identity() != &self.identity {
            return Err(invalid("identity", "query does not match the captured producer"));
        }
        let hits = self.vectors.search(cx, query, limit)?;
        let mut results = Vec::new();
        results.try_reserve_exact(hits.len()).map_err(map_storage_error)?;
        for hit in hits {
            checkpoint(cx)?;
            let index = usize::try_from(hit.index).map_err(map_storage_error)?;
            let publication = self.vectors.records.get(index)
                .ok_or_else(|| invalid("hit", "result is outside the captured vector set"))?;
            let document = self.documents.get(index)
                .ok_or_else(|| invalid("hit", "result has no captured source"))?;
            if hit.doc_id.as_str() != document.catalog.doc_id.as_str()
                || publication.doc_id != document.catalog.doc_id
                || publication.content_hash != document.catalog.content_hash
            {
                return Err(invalid("hit", "result and source revisions disagree"));
            }
            results.push(PublishedSearchHit { hit, publication, document });
        }
        checkpoint(cx)?;
        Ok(results)
    }
}

/// Internal source read on the publication scan's transaction. No callbacks,
/// nested transactions, source-file reads, or awaits are involved.
pub(crate) fn read_search_document(
    conn: &AsyncConnection,
    cx: &Cx,
    publication: &PublishedEmbedding,
    remaining_bytes: usize,
) -> SearchResult<(PublishedSearchDocument, usize)> {
    checkpoint(cx)?;
    let rows = conn.query_with_params_sync(
        SOURCE_LENGTHS_SQL,
        &[
            SqliteValue::Text(publication.doc_id.clone().into()),
            SqliteValue::Blob(publication.content_hash.to_vec().into()),
        ],
    ).map_err(map_storage_error)?;
    checkpoint(cx)?;
    let row = rows.first()
        .ok_or_else(|| invalid("source", "publication has no matching catalog revision"))?;
    if matches!(row.get(0), Some(SqliteValue::Null)) {
        return Err(invalid("source", "complete canonical input is missing; re-ingest the document"));
    }
    let mut bytes = 0_usize;
    for index in 0..4 {
        if index >= 2 && matches!(row.get(index), Some(SqliteValue::Null)) {
            continue;
        }
        let length = usize::try_from(row_i64(row, index, "publication_source.byte_length")?)
            .map_err(|_| invalid("source", "invalid source byte length"))?;
        bytes = bytes.checked_add(length)
            .filter(|total| *total <= remaining_bytes)
            .ok_or_else(|| invalid("max_source_bytes", "complete source snapshot exceeds its byte budget"))?;
    }
    checkpoint(cx)?;
    let catalog = get_document_inner(conn, &publication.doc_id)?
        .ok_or_else(|| invalid("source", "catalog record disappeared within a snapshot"))?;
    if catalog.content_hash != publication.content_hash {
        return Err(invalid("source", "catalog revision changed within a snapshot"));
    }
    let canonical_text = read_document_content(conn, &catalog)?
        .ok_or_else(|| invalid("source", "retained input does not cover the publication revision"))?;
    checkpoint(cx)?;
    Ok((PublishedSearchDocument { catalog, canonical_text }, bytes))
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.publication_search".to_owned(),
        reason: error.to_string(),
    })
}

fn invalid(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("publication_search.{field}"),
        value: "rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::sync::Arc;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::IdentityBoundEmbedding;
    use crate::{ClaimOutcome, ContentHasher, JobQueueConfig, PersistentJobQueue, Storage};
    use super::*;

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
        // Synthetic semantic contract for API tests, not real-model evidence.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 2);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(), sha256: "b".repeat(64), size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        identity
    }

    fn fixture() -> (Arc<Storage>, PersistentJobQueue) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        (storage, queue)
    }

    fn publish(storage: &Storage, queue: &PersistentJobQueue, id: &str, body: &str, values: [f32; 2]) {
        let hash = ContentHasher::hash(body);
        let mut document = DocumentRecord::new(
            id, body.chars().take(400).collect::<String>(), hash, body.chars().count(), 1, 2,
        );
        document.source_path = Some("missing://never-read".to_owned());
        document.metadata = Some(serde_json::json!({"tag": body}));
        storage.upsert_document(&document).unwrap();
        storage.set_document_content(id, body).unwrap();
        queue.enqueue(id, "model", &hash, 0).unwrap();
        let claim = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        let response = IdentityBoundEmbedding { values: values.to_vec(), identity: identity("model") };
        assert_eq!(storage.publish_claimed_embedding(&claim, &hash, &response.identity, &response).unwrap(), ClaimOutcome::Applied(()));
    }

    fn load(storage: &Storage, cx: &Cx) -> PublishedSearchSnapshot {
        storage.load_published_search_snapshot(cx, "model", &identity("model"), PublishedSearchReadLimits::default()).unwrap()
    }

    #[test]
    fn hits_keep_full_unicode_input_metadata_and_revision_after_replacement_and_deletion() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let body = "café 数据 \0 preserved tail ".repeat(50);
            publish(&storage, &queue, "doc", &body, [1.0, 0.0]);
            let old = load(&storage, &cx);
            publish(&storage, &queue, "doc", "replacement body", [-1.0, 0.0]);
            let new = load(&storage, &cx);
            storage.connection().execute_sync("DELETE FROM documents WHERE doc_id = 'doc';").unwrap();
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            let old_hits = old.search(&cx, &query, 1).unwrap();
            let new_hits = new.search(&cx, &query, 1).unwrap();
            assert_eq!(old_hits[0].document().canonical_text(), body);
            assert_eq!(old_hits[0].document().catalog().metadata.as_ref().unwrap()["tag"].as_str(), Some(body.as_str()));
            assert_eq!(old_hits[0].publication().content_hash, ContentHasher::hash(&body));
            assert_eq!(new_hits[0].document().canonical_text(), "replacement body");
            assert!(old_hits[0].hit().score > 0.99 && new_hits[0].hit().score < -0.99);
            assert!(load(&storage, &cx).is_empty());
        });
    }

    #[test]
    fn exact_source_budget_and_empty_zero_budget_do_not_truncate() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let zero = PublishedSearchReadLimits { max_source_bytes: 0, ..PublishedSearchReadLimits::default() };
            assert!(storage.load_published_search_snapshot(&cx, "model", &identity("model"), zero).unwrap().is_empty());
            for (id, body) in [("a\0tail", "数据\0tail"), ("é", "second body")] {
                publish(&storage, &queue, id, body, [1.0, 0.0]);
            }
            let captured = load(&storage, &cx);
            assert_eq!(captured.len(), 2);
            let expected_bytes: usize = ["数据\0tail", "second body"].iter().map(|body| {
                body.len() * 2 + "missing://never-read".len()
                    + serde_json::to_string(&serde_json::json!({"tag": body})).unwrap().len()
            }).sum();
            assert_eq!(captured.source_payload_bytes(), expected_bytes);
            let exact = PublishedSearchReadLimits { max_source_bytes: expected_bytes, ..zero };
            assert_eq!(storage.load_published_search_snapshot(&cx, "model", &identity("model"), exact).unwrap().len(), 2);
            for budget in [0, exact.max_source_bytes - 1] {
                let error = storage.load_published_search_snapshot(&cx, "model", &identity("model"), PublishedSearchReadLimits { max_source_bytes: budget, ..exact }).unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { field, .. } if field == "publication_search.max_source_bytes"));
                assert!(!storage.connection().in_transaction());
            }
        });
    }

    #[test]
    fn source_admission_precedes_large_or_malformed_payload_decoding() {
        for column in ["canonical_text", "metadata_json", "source_path", "content_preview"] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "doc", "original", [1.0, 0.0]);
                let table = if column == "canonical_text" { "document_contents" } else { "documents" };
                storage.connection().execute_with_params_sync(
                    &format!("UPDATE {table} SET {column} = ?1;"),
                    &[SqliteValue::Text("private-invalid-json\0数据".repeat(1000).into())],
                ).unwrap();
                let error = storage.load_published_search_snapshot(&cx, "model", &identity("model"), PublishedSearchReadLimits { max_source_bytes: 100, ..PublishedSearchReadLimits::default() }).unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { field, .. } if field == "publication_search.max_source_bytes"));
                assert!(!storage.connection().in_transaction());
            });
        }
    }

    #[test]
    fn missing_stale_or_corrupt_input_never_falls_back_to_a_preview() {
        for sql in [
            "DELETE FROM document_contents WHERE doc_id = 'z';",
            "UPDATE document_contents SET content_hash = zeroblob(32) WHERE doc_id = 'z';",
            "UPDATE document_contents SET canonical_text = 'wrong body' WHERE doc_id = 'z';",
            "UPDATE document_contents SET canonical_text = X'0000' WHERE doc_id = 'z';",
        ] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                for id in ["a", "z"] { publish(&storage, &queue, id, "valid preview", [1.0, 0.0]); }
                storage.connection().execute_sync(sql).unwrap();
                assert!(storage.load_published_search_snapshot(&cx, "model", &identity("model"), PublishedSearchReadLimits::default()).is_err());
                assert_eq!(queue.queue_depth().unwrap().completed, 2);
                assert!(!storage.connection().in_transaction());
            });
        }
    }

    #[test]
    fn unpublished_rows_do_not_require_source_or_consume_source_budget() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in ["current", "pending", "failed", "skipped", "obsolete"] {
                publish(&storage, &queue, id, "original", [1.0, 0.0]);
            }
            storage.connection().execute_sync("UPDATE embedding_status SET status = doc_id WHERE doc_id IN ('pending', 'failed', 'skipped');").unwrap();
            storage.connection().execute_sync("UPDATE documents SET content_hash = zeroblob(32) WHERE doc_id = 'obsolete';").unwrap();
            storage.connection().execute_sync("DELETE FROM document_contents WHERE doc_id != 'current';").unwrap();
            let captured = load(&storage, &cx);
            assert_eq!(captured.len(), 1);
            assert_eq!(captured.documents()[0].catalog().doc_id, "current");
            assert_eq!(captured.vector_stats().records, 1);
        });
    }

    #[test]
    fn empty_and_zero_limit_searches_still_bind_the_captured_identity() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let empty = load(&storage, &cx);
            publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
            let full = load(&storage, &cx);
            let foreign = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("foreign")).unwrap();
            for corpus in [&empty, &full] {
                for k in [0, 1] { assert!(corpus.search(&cx, &foreign, k).is_err()); }
            }
        });
    }

    #[test]
    fn loading_and_searching_honor_precancellation_without_database_changes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
            let captured = load(&storage, &cx);
            let before = storage.metrics_snapshot();
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            cx.cancel_with(asupersync::CancelKind::User, Some("stop"));
            assert!(matches!(storage.load_published_search_snapshot(&cx, "model", &identity("model"), PublishedSearchReadLimits::default()), Err(SearchError::Cancelled { .. })));
            assert!(matches!(captured.search(&cx, &query, 0), Err(SearchError::Cancelled { .. })));
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }
}
