//! Exact search with source text retained beside the vectors it describes.
//!
//! Load through `Storage::load_published_search_snapshot`, then search the
//! immutable value. Results borrow their source from that value, never from a
//! later catalog lookup or a changed external file. This is a bounded exact
//! backend for one publication tier, not ANN or a live-refreshing index.

use asupersync::Cx;
use frankensearch_core::filter::SearchFilter;
use frankensearch_core::types::{BoundQueryEmbedding, VectorHit};
use frankensearch_core::{
    Canonicalizer, Embedder, EmbeddingIdentityBundleV1, SearchError, SearchResult,
};
use fsqlite::AsyncConnection;
use fsqlite_types::value::SqliteValue;

use crate::connection::map_storage_error;
use crate::document::get_document_inner;
use crate::document_content::read_document_content;
use crate::schema::row_i64;
use crate::{
    DocumentRecord, PublishedEmbedding, PublishedEmbeddingReadLimits, PublishedEmbeddingScanStats,
    PublishedEmbeddingSnapshot,
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

    /// Embed a text query and search this captured corpus with verified sources.
    ///
    /// The caller supplies the query canonicalizer corresponding to the captured
    /// input contract; the `Canonicalizer` trait cannot attest its own policy.
    /// Document canonicalization is never applied to a query here. The actual
    /// `embed_bound` response, not a newly stamped raw vector, supplies the query
    /// identity. Provider ID, dimension and full identity must match before and
    /// after canonicalization and after inference, including provider failures.
    ///
    /// Empty canonical queries, empty corpora and zero limits skip inference but
    /// do not skip producer admission. No database connection/transaction is held
    /// by this snapshot or across the await. Cancellation dominates provider
    /// errors; caller-owned provider deadlines still apply. Exact scoring is
    /// synchronous after inference, as in `search`; this is not an ANN backend.
    pub async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        canonicalizer: &dyn Canonicalizer,
        text: &str,
        limit: usize,
    ) -> SearchResult<Vec<PublishedSearchHit<'_>>> {
        self.search_text_filtered(cx, embedder, canonicalizer, text, limit, None)
            .await
    }

    /// Bound query inference followed by pre-top-k filtering on captured metadata.
    ///
    /// Preserves every producer and response check in [`Self::search_text`]. The
    /// filter runs only after inference succeeds, against the ID and metadata
    /// retained in this snapshot, never a later catalog lookup. No-op queries
    /// still admit the producer but invoke neither inference nor the filter.
    /// Predicate callbacks are synchronous and cannot be preempted; cancellation
    /// is checked around each one. See [`Self::search_filtered`] for the exact
    /// missing-metadata and retained-policy semantics.
    pub async fn search_text_filtered(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        canonicalizer: &dyn Canonicalizer,
        text: &str,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<PublishedSearchHit<'_>>> {
        self.admit_query_producer(cx, embedder)?;
        if limit == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let text = canonicalizer.canonicalize_query(text);
        checkpoint(cx)?;
        // Even an empty-query canonicalizer can run arbitrary callbacks.
        self.admit_query_producer(cx, embedder)?;
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let response = embedder.embed_bound(cx, &text).await;
        checkpoint(cx)?;
        let response = match response {
            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
            response => response,
        };
        // Provider failure is not permission to silently accept producer drift.
        self.admit_query_producer(cx, embedder)?;
        let response = response?;
        if response.identity != self.identity {
            return Err(invalid(
                "identity",
                "query response has a foreign producer contract",
            ));
        }
        let query = BoundQueryEmbedding::new(response.values, response.identity)?;
        checkpoint(cx)?;
        self.search_filtered(cx, &query, limit, filter)
    }

    fn admit_query_producer(&self, cx: &Cx, embedder: &dyn Embedder) -> SearchResult<()> {
        checkpoint(cx)?;
        // Capture errors before checking cancellation: metadata accessors can
        // request cancellation while returning a different error or identity.
        let admitted = (|| {
            if matches!(
                self.identity.space.kind,
                frankensearch_core::generation::EmbeddingSpaceKindV1::HashControl
            ) {
                return Err(invalid(
                    "identity",
                    "hash-control input is not semantic search",
                ));
            }
            let id = embedder.id();
            let dimension = embedder.dimension();
            let identity = embedder.identity()?;
            if id != self.embedder_id.as_str()
                || identity != &self.identity
                || usize::try_from(self.identity.space.dimension).ok() != Some(dimension)
            {
                return Err(invalid(
                    "identity",
                    "query provider does not match the captured producer",
                ));
            }
            Ok(())
        })();
        checkpoint(cx)?;
        admitted
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
        self.search_filtered(cx, query, limit, None)
    }

    /// Exact top-k among matching captured documents, with source-owned metadata.
    ///
    /// Unlike filtering a truncated result list, every eligible record competes
    /// for the bounded heap. The callback receives exactly `catalog().metadata`
    /// from the same capture as its vector and text. Catalog timestamps/paths
    /// are not synthesized into that JSON, and no live database is consulted.
    /// `None` stays `None`: core metadata filters are permissive when metadata
    /// is absent, so callers requiring rejection must supply such a predicate.
    /// A retained snapshot is not a live authorization or revocation mechanism.
    ///
    /// Every vector and source binding is checked before its predicate, including
    /// excluded records. Original positions, scores and tie ordering survive.
    /// Cancellation before/after a callback discards the whole result. Zero
    /// limits admit the query without invoking the filter. No source cloning,
    /// database transaction, hash-hint shortcut or extra candidate pool is used.
    pub fn search_filtered(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<PublishedSearchHit<'_>>> {
        checkpoint(cx)?;
        if query.identity() != &self.identity {
            return Err(invalid(
                "identity",
                "query does not match the captured producer",
            ));
        }
        let hits = if let Some(filter) = filter {
            if limit != 0 && self.documents.len() != self.vectors.records.len() {
                return Err(invalid("filter", "captured vectors and sources disagree"));
            }
            self.vectors
                .search_matching(cx, query, limit, |index, publication| {
                    let document = self
                        .documents
                        .get(index)
                        .ok_or_else(|| invalid("filter", "candidate has no captured source"))?;
                    if publication.doc_id != document.catalog.doc_id
                        || publication.content_hash != document.catalog.content_hash
                        || publication.embedder_id != self.embedder_id
                    {
                        return Err(invalid("filter", "candidate and source revisions disagree"));
                    }
                    Ok(filter.matches(&publication.doc_id, document.catalog.metadata.as_ref()))
                })?
        } else {
            self.vectors.search(cx, query, limit)?
        };
        let mut results = Vec::new();
        results
            .try_reserve_exact(hits.len())
            .map_err(map_storage_error)?;
        for hit in hits {
            checkpoint(cx)?;
            let index = usize::try_from(hit.index).map_err(map_storage_error)?;
            let publication = self
                .vectors
                .records
                .get(index)
                .ok_or_else(|| invalid("hit", "result is outside the captured vector set"))?;
            let document = self
                .documents
                .get(index)
                .ok_or_else(|| invalid("hit", "result has no captured source"))?;
            if hit.doc_id.as_str() != document.catalog.doc_id.as_str()
                || publication.doc_id != document.catalog.doc_id
                || publication.content_hash != document.catalog.content_hash
            {
                return Err(invalid("hit", "result and source revisions disagree"));
            }
            results.push(PublishedSearchHit {
                hit,
                publication,
                document,
            });
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
    let rows = conn
        .query_with_params_sync(
            SOURCE_LENGTHS_SQL,
            &[
                SqliteValue::Text(publication.doc_id.clone().into()),
                SqliteValue::Blob(publication.content_hash.to_vec().into()),
            ],
        )
        .map_err(map_storage_error)?;
    checkpoint(cx)?;
    let row = rows
        .first()
        .ok_or_else(|| invalid("source", "publication has no matching catalog revision"))?;
    if matches!(row.get(0), Some(SqliteValue::Null)) {
        return Err(invalid(
            "source",
            "complete canonical input is missing; re-ingest the document",
        ));
    }
    let mut bytes = 0_usize;
    for index in 0..4 {
        if index >= 2 && matches!(row.get(index), Some(SqliteValue::Null)) {
            continue;
        }
        let length = usize::try_from(row_i64(row, index, "publication_source.byte_length")?)
            .map_err(|_| invalid("source", "invalid source byte length"))?;
        bytes = bytes
            .checked_add(length)
            .filter(|total| *total <= remaining_bytes)
            .ok_or_else(|| {
                invalid(
                    "max_source_bytes",
                    "complete source snapshot exceeds its byte budget",
                )
            })?;
    }
    checkpoint(cx)?;
    let catalog = get_document_inner(conn, &publication.doc_id)?
        .ok_or_else(|| invalid("source", "catalog record disappeared within a snapshot"))?;
    if catalog.content_hash != publication.content_hash {
        return Err(invalid(
            "source",
            "catalog revision changed within a snapshot",
        ));
    }
    let canonical_text = read_document_content(conn, &catalog)?.ok_or_else(|| {
        invalid(
            "source",
            "retained input does not cover the publication revision",
        )
    })?;
    checkpoint(cx)?;
    Ok((
        PublishedSearchDocument {
            catalog,
            canonical_text,
        },
        bytes,
    ))
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

    use super::*;
    use crate::{ClaimOutcome, ContentHasher, JobQueueConfig, PersistentJobQueue, Storage};
    use frankensearch_core::IdentityBoundEmbedding;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
        // Synthetic semantic contract for API tests, not real-model evidence.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 2);
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

    fn fixture() -> (Arc<Storage>, PersistentJobQueue) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        (storage, queue)
    }

    fn publish(
        storage: &Storage,
        queue: &PersistentJobQueue,
        id: &str,
        body: &str,
        values: [f32; 2],
    ) {
        let hash = ContentHasher::hash(body);
        let mut document = DocumentRecord::new(
            id,
            body.chars().take(400).collect::<String>(),
            hash,
            body.chars().count(),
            1,
            2,
        );
        document.source_path = Some("missing://never-read".to_owned());
        document.metadata = Some(serde_json::json!({"tag": body}));
        storage.upsert_document(&document).unwrap();
        storage.set_document_content(id, body).unwrap();
        queue.enqueue(id, "model", &hash, 0).unwrap();
        let claim = queue.claim_batch("worker", 1).unwrap().pop().unwrap();
        let response = IdentityBoundEmbedding {
            values: values.to_vec(),
            identity: identity("model"),
        };
        assert_eq!(
            storage
                .publish_claimed_embedding(&claim, &hash, &response.identity, &response)
                .unwrap(),
            ClaimOutcome::Applied(())
        );
    }

    fn load(storage: &Storage, cx: &Cx) -> PublishedSearchSnapshot {
        storage
            .load_published_search_snapshot(
                cx,
                "model",
                &identity("model"),
                PublishedSearchReadLimits::default(),
            )
            .unwrap()
    }

    #[derive(Clone, Copy)]
    enum QueryReply {
        Normal,
        Foreign,
        BadDimension,
        NonFinite,
        Zero,
        Error,
        DriftSuccess,
        DriftError,
        CancelError,
        ExplicitCancel,
    }

    thread_local! {
        static QUERY_METADATA_CANCEL: std::cell::RefCell<Option<Cx>> = const {
            std::cell::RefCell::new(None)
        };
    }

    struct ClearQueryMetadataCancel;

    impl Drop for ClearQueryMetadataCancel {
        fn drop(&mut self) {
            QUERY_METADATA_CANCEL.with(|slot| {
                let _ = slot.borrow_mut().take();
            });
        }
    }

    struct QueryProbe {
        identity: EmbeddingIdentityBundleV1,
        foreign: EmbeddingIdentityBundleV1,
        drift: Arc<AtomicBool>,
        calls: AtomicUsize,
        metadata_calls: AtomicUsize,
        inputs: Mutex<Vec<String>>,
        reply: QueryReply,
    }

    impl QueryProbe {
        fn new(reply: QueryReply) -> Self {
            let admitted = identity("model");
            let mut foreign = admitted.clone();
            foreign.producer.backend = "foreign-test-backend".to_owned();
            foreign.validate().unwrap();
            Self {
                identity: admitted,
                foreign,
                reply,
                drift: Arc::new(AtomicBool::new(false)),
                calls: AtomicUsize::new(0),
                metadata_calls: AtomicUsize::new(0),
                inputs: Mutex::new(Vec::new()),
            }
        }
    }

    impl Embedder for QueryProbe {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            self.metadata_calls.fetch_add(1, Ordering::SeqCst);
            let cancel = QUERY_METADATA_CANCEL.with(|slot| slot.borrow_mut().take());
            if let Some(cx) = cancel {
                cx.cancel_with(asupersync::CancelKind::User, Some("metadata cancellation"));
                return Err(invalid("provider", "metadata failed"));
            }
            Ok(if self.drift.load(Ordering::SeqCst) {
                &self.foreign
            } else {
                &self.identity
            })
        }
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async { Err(invalid("provider", "raw embedding must not be used")) })
        }
        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inputs.lock().unwrap().push(text.to_owned());
                asupersync::runtime::yield_now().await;
                if matches!(
                    self.reply,
                    QueryReply::DriftSuccess | QueryReply::DriftError
                ) {
                    self.drift.store(true, Ordering::SeqCst);
                }
                if matches!(self.reply, QueryReply::CancelError) {
                    cx.cancel_with(asupersync::CancelKind::User, Some("inference cancellation"));
                }
                if matches!(self.reply, QueryReply::ExplicitCancel) {
                    return Err(SearchError::Cancelled {
                        phase: "probe".to_owned(),
                        reason: "explicit".to_owned(),
                    });
                }
                if matches!(
                    self.reply,
                    QueryReply::Error | QueryReply::DriftError | QueryReply::CancelError
                ) {
                    return Err(invalid("provider", "inference failed"));
                }
                let values = match self.reply {
                    QueryReply::BadDimension => vec![1.0],
                    QueryReply::NonFinite => vec![f32::NAN, 0.0],
                    QueryReply::Zero => vec![0.0, 0.0],
                    _ if text.contains("orchard") => vec![1.0, 0.0],
                    _ => vec![0.0, 1.0],
                };
                Ok(IdentityBoundEmbedding {
                    values,
                    identity: if matches!(self.reply, QueryReply::Foreign) {
                        self.foreign.clone()
                    } else {
                        self.identity.clone()
                    },
                })
            })
        }
        fn dimension(&self) -> usize {
            2
        }
        fn id(&self) -> &'static str {
            "model"
        }
        fn model_name(&self) -> &'static str {
            "source-query-probe"
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    struct QueryCanonicalizer {
        drift: Option<Arc<AtomicBool>>,
        empty: bool,
    }

    impl Canonicalizer for QueryCanonicalizer {
        fn canonicalize(&self, _text: &str) -> String {
            panic!("document canonicalization must not run for search text")
        }
        fn canonicalize_query(&self, text: &str) -> String {
            if let Some(drift) = &self.drift {
                drift.store(true, Ordering::SeqCst);
            }
            if self.empty {
                String::new()
            } else {
                text.trim().to_owned()
            }
        }
    }

    #[test]
    fn text_queries_use_bound_inference_and_captured_sources_after_database_reopen() {
        use crate::{InMemoryVectorSink, IngestRequest, PersistentEmbeddingWorker, StorageConfig};
        use frankensearch_core::canonicalize::DefaultCanonicalizer;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let config = StorageConfig {
                db_path: directory.path().join("query.sqlite3"),
                ..StorageConfig::default()
            };
            let probe = Arc::new(QueryProbe::new(QueryReply::Normal));
            let sink = Arc::new(InMemoryVectorSink::default());
            {
                let storage = Arc::new(Storage::open(config.clone()).unwrap());
                let worker = PersistentEmbeddingWorker::new(
                    storage,
                    JobQueueConfig::default(),
                    Arc::new(DefaultCanonicalizer::default()),
                    probe.clone(),
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
                assert_eq!(
                    worker
                        .process_batch(&cx, "publisher")
                        .await
                        .unwrap()
                        .jobs_completed,
                    2
                );
            }
            let storage = Storage::open(config).unwrap();
            let captured = load(&storage, &cx);
            storage
                .connection()
                .execute_sync("DELETE FROM documents;")
                .unwrap();
            drop(storage);
            let canonicalizer = QueryCanonicalizer {
                drift: None,
                empty: false,
            };
            let hits = captured
                .search_text(&cx, probe.as_ref(), &canonicalizer, "  apple orchard  ", 1)
                .await
                .unwrap();
            assert_eq!(hits[0].hit().doc_id.as_str(), "fruit");
            assert_eq!(hits[0].document().canonical_text(), "apple orchard");
            assert_eq!(
                hits[0].publication().content_hash,
                ContentHasher::hash("apple orchard")
            );
            assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
            assert_eq!(
                probe.inputs.lock().unwrap().last().unwrap(),
                "apple orchard"
            );
            assert!(sink.entries().is_empty());
        });
    }

    #[test]
    fn text_query_noops_skip_inference_but_do_not_hide_a_foreign_provider() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let empty = load(&storage, &cx);
            publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
            let full = load(&storage, &cx);
            let probe = QueryProbe::new(QueryReply::Normal);
            let canonicalizer = QueryCanonicalizer {
                drift: None,
                empty: false,
            };
            for (snapshot, text, k) in [
                (&empty, "orchard", 1),
                (&full, "orchard", 0),
                (&full, " \t ", 1),
            ] {
                assert!(
                    snapshot
                        .search_text(&cx, &probe, &canonicalizer, text, k)
                        .await
                        .unwrap()
                        .is_empty()
                );
            }
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
            probe.drift.store(true, Ordering::SeqCst);
            for (snapshot, k) in [(&empty, 1), (&full, 0)] {
                assert!(
                    snapshot
                        .search_text(&cx, &probe, &canonicalizer, "", k)
                        .await
                        .is_err()
                );
            }
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn text_queries_reject_foreign_malformed_and_drifting_outputs() {
        for reply in [
            QueryReply::Foreign,
            QueryReply::BadDimension,
            QueryReply::NonFinite,
            QueryReply::Zero,
            QueryReply::DriftSuccess,
            QueryReply::DriftError,
            QueryReply::Error,
        ] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
                let captured = load(&storage, &cx);
                let probe = QueryProbe::new(reply);
                let canonicalizer = QueryCanonicalizer {
                    drift: None,
                    empty: false,
                };
                let error = captured
                    .search_text(&cx, &probe, &canonicalizer, "orchard", 1)
                    .await
                    .unwrap_err();
                if matches!(reply, QueryReply::DriftError | QueryReply::DriftSuccess) {
                    assert!(
                        matches!(error, SearchError::InvalidConfig { field, .. } if field == "publication_search.identity")
                    );
                } else if matches!(reply, QueryReply::Error) {
                    assert!(
                        matches!(error, SearchError::InvalidConfig { field, .. } if field == "publication_search.provider")
                    );
                }
                assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
                assert_eq!(queue.queue_depth().unwrap().completed, 1);
            });
        }
    }

    #[test]
    fn canonicalizer_callbacks_cannot_switch_providers_even_for_an_empty_query() {
        for empty in [false, true] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
                let captured = load(&storage, &cx);
                let probe = QueryProbe::new(QueryReply::Normal);
                let canonicalizer = QueryCanonicalizer {
                    drift: Some(Arc::clone(&probe.drift)),
                    empty,
                };
                assert!(
                    captured
                        .search_text(&cx, &probe, &canonicalizer, "orchard", 1)
                        .await
                        .is_err()
                );
                assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
            });
        }
    }

    #[test]
    fn query_cancellation_dominates_provider_and_metadata_failures() {
        for reply in [QueryReply::CancelError, QueryReply::ExplicitCancel] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
                let captured = load(&storage, &cx);
                let probe = QueryProbe::new(reply);
                let canonicalizer = QueryCanonicalizer {
                    drift: None,
                    empty: false,
                };
                assert!(matches!(
                    captured
                        .search_text(&cx, &probe, &canonicalizer, "orchard", 1)
                        .await,
                    Err(SearchError::Cancelled { .. })
                ));
                assert_eq!(
                    probe.metadata_calls.load(Ordering::SeqCst),
                    2,
                    "cancelled inference never calls provider metadata again"
                );
            });
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, _) = fixture();
            let captured = load(&storage, &cx);
            let probe = QueryProbe::new(QueryReply::Normal);
            let _guard = ClearQueryMetadataCancel;
            QUERY_METADATA_CANCEL.with(|slot| {
                *slot.borrow_mut() = Some(cx.clone());
            });
            let canonicalizer = QueryCanonicalizer {
                drift: None,
                empty: false,
            };
            assert!(matches!(
                captured
                    .search_text(&cx, &probe, &canonicalizer, "", 0)
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn precancelled_text_queries_do_not_invoke_any_provider_code() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, _) = fixture();
            let captured = load(&storage, &cx);
            let probe = QueryProbe::new(QueryReply::Normal);
            let canonicalizer = QueryCanonicalizer {
                drift: None,
                empty: false,
            };
            cx.cancel_with(asupersync::CancelKind::User, Some("stop before query"));
            assert!(matches!(
                captured
                    .search_text(&cx, &probe, &canonicalizer, "", 0)
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(probe.metadata_calls.load(Ordering::SeqCst), 0);
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        });
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
            storage
                .connection()
                .execute_sync("DELETE FROM documents WHERE doc_id = 'doc';")
                .unwrap();
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            let old_hits = old.search(&cx, &query, 1).unwrap();
            let new_hits = new.search(&cx, &query, 1).unwrap();
            assert_eq!(old_hits[0].document().canonical_text(), body);
            assert_eq!(
                old_hits[0].document().catalog().metadata.as_ref().unwrap()["tag"].as_str(),
                Some(body.as_str())
            );
            assert_eq!(
                old_hits[0].publication().content_hash,
                ContentHasher::hash(&body)
            );
            assert_eq!(new_hits[0].document().canonical_text(), "replacement body");
            assert!(old_hits[0].hit().score > 0.99 && new_hits[0].hit().score < -0.99);
            assert!(load(&storage, &cx).is_empty());
        });
    }

    #[test]
    fn exact_source_budget_and_empty_zero_budget_do_not_truncate() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let zero = PublishedSearchReadLimits {
                max_source_bytes: 0,
                ..PublishedSearchReadLimits::default()
            };
            assert!(
                storage
                    .load_published_search_snapshot(&cx, "model", &identity("model"), zero)
                    .unwrap()
                    .is_empty()
            );
            for (id, body) in [("a\0tail", "数据\0tail"), ("é", "second body")] {
                publish(&storage, &queue, id, body, [1.0, 0.0]);
            }
            let captured = load(&storage, &cx);
            assert_eq!(captured.len(), 2);
            let expected_bytes: usize = ["数据\0tail", "second body"]
                .iter()
                .map(|body| {
                    body.len() * 2
                        + "missing://never-read".len()
                        + serde_json::to_string(&serde_json::json!({"tag": body}))
                            .unwrap()
                            .len()
                })
                .sum();
            assert_eq!(captured.source_payload_bytes(), expected_bytes);
            let exact = PublishedSearchReadLimits {
                max_source_bytes: expected_bytes,
                ..zero
            };
            assert_eq!(
                storage
                    .load_published_search_snapshot(&cx, "model", &identity("model"), exact)
                    .unwrap()
                    .len(),
                2
            );
            for budget in [0, exact.max_source_bytes - 1] {
                let error = storage
                    .load_published_search_snapshot(
                        &cx,
                        "model",
                        &identity("model"),
                        PublishedSearchReadLimits {
                            max_source_bytes: budget,
                            ..exact
                        },
                    )
                    .unwrap_err();
                assert!(
                    matches!(error, SearchError::InvalidConfig { field, .. } if field == "publication_search.max_source_bytes")
                );
                assert!(!storage.connection().in_transaction());
            }
        });
    }

    #[test]
    fn source_admission_precedes_large_or_malformed_payload_decoding() {
        for column in [
            "canonical_text",
            "metadata_json",
            "source_path",
            "content_preview",
        ] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "doc", "original", [1.0, 0.0]);
                let table = if column == "canonical_text" {
                    "document_contents"
                } else {
                    "documents"
                };
                storage
                    .connection()
                    .execute_with_params_sync(
                        &format!("UPDATE {table} SET {column} = ?1;"),
                        &[SqliteValue::Text(
                            "private-invalid-json\0数据".repeat(1000).into(),
                        )],
                    )
                    .unwrap();
                let error = storage
                    .load_published_search_snapshot(
                        &cx,
                        "model",
                        &identity("model"),
                        PublishedSearchReadLimits {
                            max_source_bytes: 100,
                            ..PublishedSearchReadLimits::default()
                        },
                    )
                    .unwrap_err();
                assert!(
                    matches!(error, SearchError::InvalidConfig { field, .. } if field == "publication_search.max_source_bytes")
                );
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
                for id in ["a", "z"] {
                    publish(&storage, &queue, id, "valid preview", [1.0, 0.0]);
                }
                storage.connection().execute_sync(sql).unwrap();
                assert!(
                    storage
                        .load_published_search_snapshot(
                            &cx,
                            "model",
                            &identity("model"),
                            PublishedSearchReadLimits::default()
                        )
                        .is_err()
                );
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
            storage
                .connection()
                .execute_sync(
                    "UPDATE documents SET content_hash = zeroblob(32) WHERE doc_id = 'obsolete';",
                )
                .unwrap();
            storage
                .connection()
                .execute_sync("DELETE FROM document_contents WHERE doc_id != 'current';")
                .unwrap();
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
                for k in [0, 1] {
                    assert!(corpus.search(&cx, &foreign, k).is_err());
                }
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
            assert!(matches!(
                storage.load_published_search_snapshot(
                    &cx,
                    "model",
                    &identity("model"),
                    PublishedSearchReadLimits::default()
                ),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                captured.search(&cx, &query, 0),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }

    fn set_filter_metadata(storage: &Storage, id: &str, metadata: Option<serde_json::Value>) {
        storage
            .connection()
            .execute_with_params_sync(
                "UPDATE documents SET metadata_json = ?1 WHERE doc_id = ?2;",
                &[
                    metadata.map_or(SqliteValue::Null, |value| {
                        SqliteValue::Text(value.to_string().into())
                    }),
                    SqliteValue::Text(id.to_owned().into()),
                ],
            )
            .unwrap();
    }

    struct SourceFilterProbe<'a> {
        calls: AtomicUsize,
        cancel: Option<&'a Cx>,
        accept: bool,
    }

    impl SearchFilter for SourceFilterProbe<'_> {
        fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
            assert!(metadata.and_then(|value| value.get("tag")).is_some());
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(cx) = self.cancel {
                cx.cancel_with(asupersync::CancelKind::User, Some("filter cancellation"));
            }
            self.accept
        }

        fn name(&self) -> &str {
            panic!("filter diagnostics must not invoke another callback")
        }
    }

    #[test]
    fn captured_metadata_filters_before_top_k_and_survives_catalog_changes() {
        use frankensearch_core::filter::DocTypeFilter;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "a-blocked", "old blocked", [1.0, 0.0]);
            publish(&storage, &queue, "b-keep", "old retained", [1.0, 1.0]);
            set_filter_metadata(
                &storage,
                "a-blocked",
                Some(serde_json::json!({"doc_type": "skip"})),
            );
            set_filter_metadata(
                &storage,
                "b-keep",
                Some(serde_json::json!({"doc_type": "keep"})),
            );
            let old = load(&storage, &cx);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            let filter = DocTypeFilter::new(["keep"]);
            let before = storage.metrics_snapshot();
            assert_eq!(
                old.search(&cx, &query, 1).unwrap()[0].hit().doc_id.as_str(),
                "a-blocked",
            );
            let hits = old.search_filtered(&cx, &query, 1, Some(&filter)).unwrap();
            assert_eq!(
                hits.len(),
                1,
                "excluded top scorer must not consume the only slot",
            );
            assert_eq!(hits[0].hit().doc_id.as_str(), "b-keep");
            assert_eq!(hits[0].hit().index, 1);
            assert!(std::ptr::eq(hits[0].document(), &old.documents()[1]));
            assert_eq!(
                storage.metrics_snapshot(),
                before,
                "filtering does no SQL work",
            );

            set_filter_metadata(
                &storage,
                "a-blocked",
                Some(serde_json::json!({"doc_type": "keep"})),
            );
            set_filter_metadata(
                &storage,
                "b-keep",
                Some(serde_json::json!({"doc_type": "skip"})),
            );
            let fresh = load(&storage, &cx);
            storage
                .connection()
                .execute_sync("DELETE FROM documents;")
                .unwrap();
            drop(queue);
            drop(storage);
            let retained = old.search_filtered(&cx, &query, 1, Some(&filter)).unwrap();
            let refreshed = fresh
                .search_filtered(&cx, &query, 1, Some(&filter))
                .unwrap();
            assert_eq!(retained[0].hit().doc_id.as_str(), "b-keep");
            assert_eq!(retained[0].document().canonical_text(), "old retained");
            assert_eq!(refreshed[0].hit().doc_id.as_str(), "a-blocked");
            assert_eq!(
                retained[0].document().catalog().metadata.as_ref().unwrap()["doc_type"],
                "keep",
            );
            assert_eq!(
                old.search_filtered(&cx, &query, 2, None)
                    .unwrap()
                    .iter()
                    .map(|hit| hit.hit().clone())
                    .collect::<Vec<_>>(),
                old.search(&cx, &query, 2)
                    .unwrap()
                    .iter()
                    .map(|hit| hit.hit().clone())
                    .collect::<Vec<_>>(),
            );
        });
    }

    #[test]
    fn metadata_chains_and_exact_exclusions_keep_the_core_missing_metadata_contract() {
        use frankensearch_core::filter::{
            DateRangeFilter, DocTypeFilter, ExcludeDocIdsFilter, FilterChain, FilterMode,
        };

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            for id in [
                "a-old",
                "b-excluded",
                "c\0keep",
                "d-prose",
                "e-absent",
                "f-empty",
            ] {
                publish(&storage, &queue, id, "body", [-1.0, 0.0]);
            }
            for (id, value) in [
                (
                    "a-old",
                    Some(serde_json::json!({"doc_type": "code", "created_at": 5})),
                ),
                (
                    "b-excluded",
                    Some(serde_json::json!({"doc_type": "code", "created_at": 15})),
                ),
                (
                    "c\0keep",
                    Some(serde_json::json!({"doc_type": "code", "created_at": 15})),
                ),
                (
                    "d-prose",
                    Some(serde_json::json!({"doc_type": "prose", "created_at": 15})),
                ),
                ("e-absent", None),
                ("f-empty", Some(serde_json::json!({}))),
            ] {
                set_filter_metadata(&storage, id, value);
            }
            let snapshot = load(&storage, &cx);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            let chain = FilterChain::new(FilterMode::All)
                .with(Box::new(DocTypeFilter::new(["code"])))
                .with(Box::new(DateRangeFilter::between(10, 20)));
            let filter = ExcludeDocIdsFilter::new(Some(&chain), ["b-excluded"]);
            let hits = snapshot
                .search_filtered(&cx, &query, usize::MAX, Some(&filter))
                .unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.hit().doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["c\0keep", "e-absent"],
                "None is permissive; present-but-incomplete JSON is rejected by these filters",
            );
            assert!(hits.iter().all(|hit| hit.hit().score < -0.99));
            assert_eq!(hits[0].document().catalog().created_at, 1);
            assert_eq!(
                hits[0].document().catalog().metadata.as_ref().unwrap()["created_at"],
                15,
            );
        });
    }

    #[test]
    fn excluded_source_bindings_are_validated_before_filter_callbacks() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "a", "source a", [1.0, 0.0]);
            publish(&storage, &queue, "b", "source b", [0.0, 1.0]);
            let original = load(&storage, &cx);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            for corruption in ["id", "revision", "length", "vector"] {
                let mut snapshot = original.clone();
                match corruption {
                    "id" => snapshot.documents[1].catalog.doc_id = "foreign".to_owned(),
                    "revision" => snapshot.documents[1].catalog.content_hash = [0; 32],
                    "length" => {
                        snapshot.documents.pop();
                    }
                    "vector" => snapshot.vectors.records[1].embedding.values[0] = f32::NAN,
                    _ => unreachable!(),
                }
                let filter = SourceFilterProbe {
                    calls: AtomicUsize::new(0),
                    cancel: None,
                    accept: false,
                };
                assert!(
                    snapshot
                        .search_filtered(&cx, &query, 1, Some(&filter))
                        .is_err(),
                    "{corruption}",
                );
                assert_eq!(
                    filter.calls.load(Ordering::SeqCst),
                    usize::from(corruption != "length"),
                );
            }
        });
    }

    #[test]
    fn filtered_text_queries_keep_bound_dispatch_and_reject_bad_providers_before_filters() {
        use frankensearch_core::filter::DocTypeFilter;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "a-blocked", "blocked", [1.0, 0.0]);
            publish(&storage, &queue, "b-keep", "retained", [0.0, 1.0]);
            set_filter_metadata(
                &storage,
                "a-blocked",
                Some(serde_json::json!({"doc_type": "skip"})),
            );
            set_filter_metadata(
                &storage,
                "b-keep",
                Some(serde_json::json!({"doc_type": "keep"})),
            );
            let captured = load(&storage, &cx);
            drop(queue);
            drop(storage);
            let probe = QueryProbe::new(QueryReply::Normal);
            let canonicalizer = QueryCanonicalizer {
                drift: None,
                empty: false,
            };
            let filter = DocTypeFilter::new(["keep"]);
            let hits = captured
                .search_text_filtered(&cx, &probe, &canonicalizer, "  orchard  ", 1, Some(&filter))
                .await
                .unwrap();
            assert_eq!(hits[0].hit().doc_id.as_str(), "b-keep");
            assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
            assert_eq!(*probe.inputs.lock().unwrap(), ["orchard"]);
            assert!(std::ptr::eq(hits[0].document(), &captured.documents()[1]));
            for reply in [
                QueryReply::Foreign,
                QueryReply::BadDimension,
                QueryReply::NonFinite,
                QueryReply::Zero,
                QueryReply::DriftSuccess,
                QueryReply::DriftError,
                QueryReply::Error,
            ] {
                let probe = QueryProbe::new(reply);
                let filter = SourceFilterProbe {
                    calls: AtomicUsize::new(0),
                    cancel: None,
                    accept: false,
                };
                assert!(
                    captured
                        .search_text_filtered(
                            &cx,
                            &probe,
                            &canonicalizer,
                            "orchard",
                            1,
                            Some(&filter)
                        )
                        .await
                        .is_err()
                );
                assert_eq!(filter.calls.load(Ordering::SeqCst), 0);
                assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
            }
            for empty in [false, true] {
                let probe = QueryProbe::new(QueryReply::Normal);
                let canonicalizer = QueryCanonicalizer {
                    drift: Some(Arc::clone(&probe.drift)),
                    empty,
                };
                let filter = SourceFilterProbe {
                    calls: AtomicUsize::new(0),
                    cancel: None,
                    accept: false,
                };
                assert!(
                    captured
                        .search_text_filtered(
                            &cx,
                            &probe,
                            &canonicalizer,
                            "orchard",
                            1,
                            Some(&filter)
                        )
                        .await
                        .is_err()
                );
                assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
                assert_eq!(filter.calls.load(Ordering::SeqCst), 0);
            }
        });
    }

    #[test]
    fn filtered_noops_and_wrong_queries_never_run_the_predicate() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let empty = load(&storage, &cx);
            publish(&storage, &queue, "doc", "source", [1.0, 0.0]);
            let full = load(&storage, &cx);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap();
            let wrong = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("foreign")).unwrap();
            let probe = QueryProbe::new(QueryReply::Normal);
            let canonicalizer = QueryCanonicalizer {
                drift: None,
                empty: false,
            };
            let filter = SourceFilterProbe {
                calls: AtomicUsize::new(0),
                cancel: None,
                accept: true,
            };
            for snapshot in [&empty, &full] {
                assert!(
                    snapshot
                        .search_filtered(&cx, &query, 0, Some(&filter))
                        .unwrap()
                        .is_empty()
                );
                for k in [0, 1] {
                    assert!(
                        snapshot
                            .search_filtered(&cx, &wrong, k, Some(&filter))
                            .is_err()
                    );
                }
            }
            for (snapshot, text, k) in [
                (&empty, "query", 1),
                (&full, "query", 0),
                (&full, " \t ", 1),
            ] {
                assert!(
                    snapshot
                        .search_text_filtered(&cx, &probe, &canonicalizer, text, k, Some(&filter))
                        .await
                        .unwrap()
                        .is_empty()
                );
            }
            probe.drift.store(true, Ordering::SeqCst);
            assert!(
                full.search_text_filtered(&cx, &probe, &canonicalizer, "", 0, Some(&filter))
                    .await
                    .is_err()
            );
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
            assert_eq!(filter.calls.load(Ordering::SeqCst), 0);
            cx.cancel_with(asupersync::CancelKind::User, Some("before filtered search"));
            assert!(matches!(
                full.search_filtered(&cx, &query, 1, Some(&filter)),
                Err(SearchError::Cancelled { .. }),
            ));
            assert_eq!(filter.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn captured_metadata_callback_cancellation_discards_results_for_accept_and_reject() {
        for accept in [false, true] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                for id in ["a", "b"] {
                    publish(&storage, &queue, id, "source", [1.0, 0.0]);
                }
                let snapshot = load(&storage, &cx);
                let probe = QueryProbe::new(QueryReply::Normal);
                let canonicalizer = QueryCanonicalizer {
                    drift: None,
                    empty: false,
                };
                let filter = SourceFilterProbe {
                    calls: AtomicUsize::new(0),
                    cancel: Some(&cx),
                    accept,
                };
                let outcome = snapshot
                    .search_text_filtered(&cx, &probe, &canonicalizer, "orchard", 1, Some(&filter))
                    .await;
                assert!(matches!(outcome, Err(SearchError::Cancelled { .. })));
                assert_eq!(filter.calls.load(Ordering::SeqCst), 1);
                assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
            });
        }
    }
}
