//! Direct exact queries on the existing publication transaction and scanner.

use super::{
    AsyncConnection, BinaryHeap, BoundQueryEmbedding, Cx, EmbeddingSpaceKindV1, Ordering,
    PublishedEmbedding, PublishedEmbeddingReadLimits, RankedHit, SearchResult, Storage,
    VectorHit, cosine, map_storage_error, publication_checkpoint, publication_error,
    query_norm, scan_on, validate_read_request,
};
use crate::{PublishedEmbeddingQueryHit, PublishedEmbeddingQueryResult};

impl Storage {
    /// Score current publications without first retaining the entire corpus.
    ///
    /// One transaction runs the existing checksum/identity/revision-admitted
    /// keyset scan and exact f64 cosine kernel, keeping at most `limit` vector
    /// publications plus the record being inspected. Losing and displaced
    /// vectors are dropped during the scan. SQL row buffers, metadata pages,
    /// decoded identity sizes and driver caches are additional memory; this is
    /// not an RSS limit. No source text is loaded by this vector-only operation.
    ///
    /// `limits` bound TOTAL scanned records and stored identity/vector bytes,
    /// not just winners or peak retained payload. A corrupt, foreign or
    /// over-budget losing tail fails the WHOLE query; filling top-k never ends
    /// validation early. Pending/failed/skipped and stale publications remain
    /// outside the query, exactly as in `load_published_embeddings`.
    ///
    /// Ordering and score bits match snapshot exact search. Returned indices
    /// are ordinals in the complete scan, not positions in the winner list.
    /// Zero limit still admits the query but opens no transaction and reports
    /// no scan accounting. A completed empty scan reports `Some` zero stats.
    ///
    /// All accumulators are transaction-attempt-local. No user callback,
    /// inference, await, file read, runtime or task enters the transaction.
    /// This synchronous work belongs on the caller's blocking lane. The
    /// configured begin mode can hold a writer lock; cancellation is
    /// cooperative and a large SQL statement itself is not preemptible.
    pub fn query_published_embeddings(
        &self,
        cx: &Cx,
        embedder_id: &str,
        query: &BoundQueryEmbedding,
        limit: usize,
        limits: PublishedEmbeddingReadLimits,
    ) -> SearchResult<PublishedEmbeddingQueryResult> {
        let norm = admit_query(cx, embedder_id, query)?;
        if limit == 0 {
            return Ok(PublishedEmbeddingQueryResult {
                hits: Vec::new(),
                scan_stats: None,
            });
        }
        let result = self.transaction(|conn| {
            select_on(conn, cx, embedder_id, query, norm, limit, limits)
        });
        publication_checkpoint(cx)?;
        result
    }
}

fn admit_query(cx: &Cx, embedder_id: &str, query: &BoundQueryEmbedding) -> SearchResult<f64> {
    validate_read_request(cx, embedder_id, query.identity())?;
    if matches!(
        query.identity().space.kind,
        EmbeddingSpaceKindV1::HashControl
    ) {
        return Err(publication_error(
            "hash-control vectors are not semantic search input",
        ));
    }
    query_norm(cx, query.vector())
}

fn select_on(
    conn: &AsyncConnection,
    cx: &Cx,
    embedder_id: &str,
    query: &BoundQueryEmbedding,
    norm: f64,
    limit: usize,
    limits: PublishedEmbeddingReadLimits,
) -> SearchResult<PublishedEmbeddingQueryResult> {
    let mut heap = BinaryHeap::new();
    let mut ordinal = 0_usize;
    let stats = scan_on(
        conn,
        cx,
        embedder_id,
        query.identity(),
        limits,
        |publication| {
            let index = u32::try_from(ordinal).map_err(|_| {
                publication_error("query position exceeds vector-hit index width")
            })?;
            let score = cosine(cx, query.vector(), &publication.embedding.values, norm)?;
            publication_checkpoint(cx)?;
            retain_candidate(
                &mut heap,
                Candidate {
                    rank: RankedHit { index, score },
                    publication,
                },
                limit,
            )?;
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| publication_error("query record count overflow"))?;
            Ok(())
        },
    )?;
    publication_checkpoint(cx)?;
    let ranked = heap.into_sorted_vec();
    let mut hits = Vec::new();
    hits.try_reserve_exact(ranked.len())
        .map_err(map_storage_error)?;
    for candidate in ranked {
        publication_checkpoint(cx)?;
        hits.push(PublishedEmbeddingQueryHit {
            hit: VectorHit {
                index: candidate.rank.index,
                score: candidate.rank.score,
                doc_id: candidate.publication.doc_id.clone().into(),
            },
            publication: candidate.publication,
        });
    }
    publication_checkpoint(cx)?;
    Ok(PublishedEmbeddingQueryResult {
        hits,
        scan_stats: Some(stats),
    })
}

/// Preserve the scanner's exact binary-ID ordering through its original index.
/// No additional vector/identity clone is made when a candidate enters the heap.
struct Candidate {
    rank: RankedHit,
    publication: PublishedEmbedding,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

fn retain_candidate(
    heap: &mut BinaryHeap<Candidate>,
    candidate: Candidate,
    limit: usize,
) -> SearchResult<()> {
    if heap.len() < limit {
        heap.try_reserve(1).map_err(map_storage_error)?;
        heap.push(candidate);
    } else if let Some(mut worst) = heap.peek_mut()
        && candidate < *worst
    {
        *worst = candidate;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::sync::Arc;

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1};
    use frankensearch_core::{IdentityBoundEmbedding, SearchError};

    use super::*;
    use crate::{ClaimOutcome, ContentHasher, DocumentRecord, JobQueueConfig, PersistentJobQueue};

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
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
        producer: &str,
        values: [f32; 2],
    ) {
        let hash = ContentHasher::hash(body);
        storage
            .upsert_document(&DocumentRecord::new(id, body, hash, body.chars().count(), 1, 1))
            .unwrap();
        storage.set_document_content(id, body).unwrap();
        queue.enqueue(id, "model", &hash, 0).unwrap();
        let claim = queue.claim_batch("publisher", 1).unwrap().pop().unwrap();
        let response = IdentityBoundEmbedding {
            values: values.to_vec(),
            identity: identity(producer),
        };
        assert_eq!(
            storage
                .publish_claimed_embedding(&claim, &hash, &response.identity, &response)
                .unwrap(),
            ClaimOutcome::Applied(())
        );
    }

    fn query() -> BoundQueryEmbedding {
        BoundQueryEmbedding::new(vec![1.0, 0.0], identity("model")).unwrap()
    }

    #[test]
    fn direct_query_agrees_with_full_capture_across_pages_and_result_limits() {
        run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let mut ids: Vec<_> = (0..67).map(|n| format!("doc-{n:03}")).collect();
            ids.extend(["a\0tail".to_owned(), "é".to_owned(), "数据".to_owned()]);
            ids.sort();
            for (index, id) in ids.iter().enumerate().rev() {
                let values = match index % 4 {
                    0 => [2.0, -0.0], 1 => [3.0, 4.0], 2 => [0.0, 2.0], _ => [-5.0, 0.0],
                };
                publish(&storage, &queue, id, "body", "model", values);
            }
            let query = query();
            let limits = PublishedEmbeddingReadLimits::default();
            let captured = storage.load_published_embeddings(&cx, "model", query.identity(), limits).unwrap();
            let full = captured.search(&cx, &query, usize::MAX).unwrap();
            for limit in [0, 1, 2, 17, 64, 70, 71, usize::MAX] {
                let before = storage.metrics_snapshot();
                let result = storage.query_published_embeddings(&cx, "model", &query, limit, limits).unwrap();
                assert_eq!(result.hits().len(), limit.min(ids.len()));
                assert_eq!(result.scan_stats(), if limit == 0 { None } else { Some(captured.stats) });
                for (actual, expected) in result.hits().iter().zip(&full) {
                    assert_eq!(actual.hit(), expected);
                    assert_eq!(actual.publication(), &captured.records[usize::try_from(expected.index).unwrap()]);
                    assert_eq!(actual.hit().score.to_bits(), expected.score.to_bits());
                }
                assert_eq!(storage.metrics_snapshot().tx_commits, before.tx_commits + u64::from(limit != 0));
                assert!(!storage.connection().in_transaction());
            }
        });
    }

    #[test]
    fn a_losing_corrupt_or_foreign_tail_refuses_the_whole_query() {
        for corruption in ["checksum", "version", "producer"] {
            run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "a", "first", "model", [1.0, 0.0]);
                publish(&storage, &queue, "z", "last", if corruption == "producer" { "foreign" } else { "model" }, [-1.0, 0.0]);
                match corruption {
                    "checksum" => { storage.connection().execute_sync("UPDATE published_embeddings SET vector_le = X'0000' WHERE doc_id = 'z';").unwrap(); }
                    "version" => { storage.connection().execute_sync("UPDATE published_embeddings SET record_version = 99 WHERE doc_id = 'z';").unwrap(); }
                    _ => {}
                }
                let before = storage.metrics_snapshot();
                assert!(storage.query_published_embeddings(&cx, "model", &query(), 1, PublishedEmbeddingReadLimits::default()).is_err());
                let after = storage.metrics_snapshot();
                assert_eq!(after.tx_commits, before.tx_commits);
                assert_eq!(after.tx_rollbacks, before.tx_rollbacks + 1);
                assert_eq!(queue.queue_depth().unwrap().completed, 2);
                assert!(!storage.connection().in_transaction());
            });
        }
    }

    #[test]
    fn scan_budgets_cover_losers_and_distinguish_empty_scan_from_noop() {
        run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let query = query();
            let zero = PublishedEmbeddingReadLimits { max_records: 0, max_payload_bytes: 0 };
            let empty = storage.query_published_embeddings(&cx, "model", &query, 1, zero).unwrap();
            assert_eq!(empty.scan_stats().unwrap().records, 0);
            publish(&storage, &queue, "a", "body", "model", [1.0, 0.0]);
            publish(&storage, &queue, "z", "body", "model", [-1.0, 0.0]);
            let reference = storage.load_published_embeddings(&cx, "model", query.identity(), PublishedEmbeddingReadLimits::default()).unwrap();
            let exact = PublishedEmbeddingReadLimits { max_records: 2, max_payload_bytes: reference.stats.payload_bytes };
            assert_eq!(storage.query_published_embeddings(&cx, "model", &query, 1, exact).unwrap().scan_stats(), Some(reference.stats));
            for (limits, field) in [
                (PublishedEmbeddingReadLimits { max_records: 1, ..exact }, "published_embeddings.max_records"),
                (PublishedEmbeddingReadLimits { max_payload_bytes: exact.max_payload_bytes - 1, ..exact }, "published_embeddings.max_payload_bytes"),
                (zero, "published_embeddings.max_records"),
            ] {
                let result = storage.query_published_embeddings(&cx, "model", &query, 1, limits);
                assert!(matches!(result, Err(SearchError::InvalidConfig { field: actual, .. }) if actual == field));
            }
            let before = storage.metrics_snapshot();
            let noop = storage.query_published_embeddings(&cx, "model", &query, 0, zero).unwrap();
            assert!(noop.hits().is_empty());
            assert_eq!(noop.scan_stats(), None);
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }

    #[test]
    fn direct_query_preserves_publication_revisions_after_updates_and_database_drop() {
        run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "doc", "original", "model", [1.0, -0.0]);
            let limits = PublishedEmbeddingReadLimits::default();
            let query = query();
            let old = storage.query_published_embeddings(&cx, "model", &query, 1, limits).unwrap();
            let changed = DocumentRecord::new("doc", "replacement", ContentHasher::hash("replacement"), 11, 1, 2);
            storage.upsert_document(&changed).unwrap();
            storage.set_document_content("doc", "replacement").unwrap();
            assert!(storage.query_published_embeddings(&cx, "model", &query, 1, limits).unwrap().hits().is_empty());
            publish(&storage, &queue, "doc", "replacement", "model", [-1.0, 0.0]);
            let new = storage.query_published_embeddings(&cx, "model", &query, 1, limits).unwrap();
            drop(queue);
            drop(storage);
            assert_eq!(old.hits()[0].hit().score.to_bits(), 1.0_f32.to_bits());
            assert_eq!(new.hits()[0].hit().score.to_bits(), (-1.0_f32).to_bits());
            assert_eq!(old.hits()[0].publication().content_hash, ContentHasher::hash("original"));
            assert_eq!(new.hits()[0].publication().content_hash, changed.content_hash);
            assert_eq!(old.hits()[0].publication().embedding.values[1].to_bits(), (-0.0_f32).to_bits());
        });
    }

    #[test]
    fn direct_selection_moves_vectors_without_growing_past_the_result_limit() {
        let mut heap = BinaryHeap::new();
        for index in 0..137_u32 {
            let score = f32::from(u16::try_from(index % 7).unwrap());
            let publication = PublishedEmbedding {
                doc_id: format!("doc-{index:03}"),
                embedder_id: "model".to_owned(),
                content_hash: ContentHasher::hash("body"),
                source_job_id: 1,
                claim_epoch: 1,
                published_at_ms: 1,
                embedding: IdentityBoundEmbedding {
                    values: vec![1.0, -0.0],
                    identity: identity("model"),
                },
            };
            let values = publication.embedding.values.as_ptr();
            retain_candidate(
                &mut heap,
                Candidate {
                    rank: RankedHit { index, score },
                    publication,
                },
                3,
            )
            .unwrap();
            assert_eq!(heap.len(), usize::try_from(index + 1).unwrap().min(3));
            if let Some(retained) = heap.iter().find(|hit| hit.rank.index == index) {
                assert_eq!(retained.publication.embedding.values.as_ptr(), values);
            }
        }
        let winners = heap.into_sorted_vec();
        assert_eq!(
            winners.iter().map(|hit| hit.rank.index).collect::<Vec<_>>(),
            [6, 13, 20]
        );
        for winner in winners {
            assert_eq!(winner.rank.score.to_bits(), 6.0_f32.to_bits());
            assert_eq!(
                winner.publication.embedding.values[1].to_bits(),
                (-0.0_f32).to_bits()
            );
        }
    }

    #[test]
    fn cancellation_and_invalid_query_admission_precede_transactions_and_noops() {
        run_test_with_cx(|cx| async move {
            let (storage, _) = fixture();
            let zero = BoundQueryEmbedding::new(vec![0.0, -0.0], identity("model")).unwrap();
            let hash = BoundQueryEmbedding::new(vec![1.0, 0.0], EmbeddingIdentityBundleV1::explicit_test_model("model", 2)).unwrap();
            let valid = query();
            let limits = PublishedEmbeddingReadLimits::default();
            let before = storage.metrics_snapshot();
            for limit in [0, 1] {
                for (tier, query) in [("model", &zero), ("model", &hash), (" ", &valid)] {
                    assert!(storage.query_published_embeddings(&cx, tier, query, limit, limits).is_err());
                }
            }
            assert_eq!(storage.metrics_snapshot(), before);
            cx.set_cancel_requested(true);
            for limit in [0, 1] {
                assert!(matches!(storage.query_published_embeddings(&cx, "model", &valid, limit, limits), Err(SearchError::Cancelled { .. })));
            }
            cx.set_cancel_requested(false);
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }
}
