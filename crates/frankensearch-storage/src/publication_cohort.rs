//! Source-coherent capture of independently identified publication tiers.
//!
//! A cohort uses one database transaction for every vector and its source.
//! It does not join separately loaded snapshots or make independently committed
//! embedding jobs atomic. Missing/pending coverage remains explicit per tier.

use asupersync::Cx;
use frankensearch_core::types::BoundQueryEmbedding;
use frankensearch_core::{
    Canonicalizer, Embedder, EmbeddingIdentityBundleV1, SearchError, SearchResult,
};

use crate::connection::map_storage_error;
use crate::{
    PublishedEmbeddingScanStats, PublishedSearchHit, PublishedSearchReadLimits,
    PublishedSearchSnapshot, Storage,
};

const MAX_COHORT_TIERS: usize = 16;

/// One operational tier and its independently supplied full producer contract.
///
/// The same dimensions or model display name do not establish compatibility.
/// Requests are returned in input order; duplicate operational IDs are refused.
#[derive(Debug, Clone, Copy)]
pub struct PublishedSearchTierRequest<'a> {
    pub embedder_id: &'a str,
    pub identity: &'a EmbeddingIdentityBundleV1,
}

/// An independently embedded query for one explicitly named captured tier.
#[derive(Clone, Copy)]
pub struct PublishedSearchTierQuery<'a> {
    pub embedder_id: &'a str,
    pub query: &'a BoundQueryEmbedding,
    pub limit: usize,
}

impl std::fmt::Debug for PublishedSearchTierQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedSearchTierQuery")
            .field("embedder_id", &self.embedder_id)
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

/// Caller-owned bound inference and query input policy for one captured tier.
///
/// A canonicalizer is explicit per tier; this trait cannot attest that the
/// caller selected the policy described by that tier's input contract.
#[derive(Clone, Copy)]
pub struct PublishedSearchTextQuery<'a> {
    pub embedder_id: &'a str,
    pub embedder: &'a dyn Embedder,
    pub canonicalizer: &'a dyn Canonicalizer,
    pub limit: usize,
}

impl std::fmt::Debug for PublishedSearchTextQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Debug must not invoke provider callbacks or expose query vectors.
        f.debug_struct("PublishedSearchTextQuery")
            .field("embedder_id", &self.embedder_id)
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

/// Ranked results and their retained source owner for one requested tier.
///
/// Scores/ranks remain tier-local. Consumers may fuse them using their chosen
/// ranking policy; this type never treats cross-model cosine scores as equal.
pub struct PublishedSearchTierHits<'a> {
    snapshot: &'a PublishedSearchSnapshot,
    hits: Vec<PublishedSearchHit<'a>>,
}

impl std::fmt::Debug for PublishedSearchTierHits<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PublishedSearchTierHits")
            .field("embedder_id", &self.snapshot.embedder_id())
            .field("hits", &self.hits.len())
            .finish_non_exhaustive()
    }
}

impl<'a> PublishedSearchTierHits<'a> {
    #[must_use]
    pub const fn snapshot(&self) -> &'a PublishedSearchSnapshot {
        self.snapshot
    }

    #[must_use]
    pub fn hits(&self) -> &[PublishedSearchHit<'a>] {
        &self.hits
    }
}

/// Immutable publication tiers and sources captured in one database transaction.
///
/// Only `Storage::load_published_search_cohort` constructs this value. A
/// consumer can query each retained tier independently, including quality-only
/// documents, without loading another database revision between search phases.
/// Missing publications are NOT inferred from another tier or presented as
/// complete coverage. Inspect each snapshot's membership before choosing a
/// consumer-specific full/partial-coverage policy.
///
/// This is database snapshot consistency, not the filesystem generation-root
/// authority, an anti-rollback floor, or atomic publication of multiple jobs.
/// Retaining a cohort also retains its old source text after updates/deletions;
/// consumers needing revocation must drop it and capture a new cohort.
#[derive(Debug, Clone)]
pub struct PublishedSearchCohort {
    snapshots: Vec<PublishedSearchSnapshot>,
    vector_stats: PublishedEmbeddingScanStats,
    source_payload_bytes: usize,
}

impl PublishedSearchCohort {
    /// Tiers in the exact request order, including tiers with no publications.
    #[must_use]
    pub fn snapshots(&self) -> &[PublishedSearchSnapshot] {
        &self.snapshots
    }

    /// Resolve an operational tier without guessing from vector dimensions.
    #[must_use]
    pub fn get(&self, embedder_id: &str) -> Option<&PublishedSearchSnapshot> {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.embedder_id() == embedder_id)
    }

    /// Aggregate vector/identity accounting. A document present in two tiers
    /// contributes two records and both stored payloads.
    #[must_use]
    pub const fn vector_stats(&self) -> PublishedEmbeddingScanStats {
        self.vector_stats
    }

    /// Source payload charged for every retained tier, including duplicates.
    #[must_use]
    pub const fn source_payload_bytes(&self) -> usize {
        self.source_payload_bytes
    }

    /// Search named tiers independently with their own identity-bound queries.
    ///
    /// Results follow request order, not capture order. A subset explicitly
    /// selects fast-only or quality-only retrieval; unknown or duplicate IDs
    /// fail before scoring. Each tier searches its entire captured membership,
    /// never a candidate pool selected by the first tier. Zero limits still
    /// validate the corresponding query. Failure discards the whole result.
    /// No database/filesystem access, inference, or runtime/task creation occurs.
    pub fn search(
        &self,
        cx: &Cx,
        requests: &[PublishedSearchTierQuery<'_>],
    ) -> SearchResult<Vec<PublishedSearchTierHits<'_>>> {
        let snapshots = self.resolve_tiers(cx, requests.iter().map(|request| request.embedder_id))?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(snapshots.len())
            .map_err(map_storage_error)?;
        for (snapshot, request) in snapshots.into_iter().zip(requests) {
            checkpoint(cx)?;
            let hits = snapshot.search(cx, request.query, request.limit)?;
            results.push(PublishedSearchTierHits { snapshot, hits });
        }
        checkpoint(cx)?;
        Ok(results)
    }

    /// Canonicalize and embed a text query independently for each named tier.
    ///
    /// Names/duplicates are resolved before inference. Each snapshot's existing
    /// `search_text` path then admits its provider before/after canonicalization
    /// and inference, consumes the actual bound response, and searches only the
    /// captured owner. No raw-vector stamping, shared cross-model query vector,
    /// database refresh, fallback provider or retry is introduced.
    ///
    /// This is buffered, sequential retrieval with caller-owned provider
    /// deadlines. A later failure returns no partial vector of tier results.
    /// For progressive delivery, retain this cohort and call individual
    /// snapshots as each query embedding becomes available; do not recapture
    /// the database between phases. Empty text/tiers/zero limits retain the
    /// single-tier admission and cancellation rules and skip inference.
    pub async fn search_text(
        &self,
        cx: &Cx,
        text: &str,
        requests: &[PublishedSearchTextQuery<'_>],
    ) -> SearchResult<Vec<PublishedSearchTierHits<'_>>> {
        let snapshots = self.resolve_tiers(cx, requests.iter().map(|request| request.embedder_id))?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(snapshots.len())
            .map_err(map_storage_error)?;
        for (snapshot, request) in snapshots.into_iter().zip(requests) {
            checkpoint(cx)?;
            let hits = snapshot
                .search_text(
                    cx,
                    request.embedder,
                    request.canonicalizer,
                    text,
                    request.limit,
                )
                .await?;
            results.push(PublishedSearchTierHits { snapshot, hits });
        }
        checkpoint(cx)?;
        Ok(results)
    }

    fn resolve_tiers<'request>(
        &self,
        cx: &Cx,
        ids: impl ExactSizeIterator<Item = &'request str>,
    ) -> SearchResult<Vec<&PublishedSearchSnapshot>> {
        checkpoint(cx)?;
        if ids.len() > MAX_COHORT_TIERS {
            return Err(invalid(
                "queries",
                "a cohort accepts at most sixteen tier queries",
            ));
        }
        let mut snapshots: Vec<&PublishedSearchSnapshot> = Vec::new();
        snapshots
            .try_reserve_exact(ids.len())
            .map_err(map_storage_error)?;
        for id in ids {
            checkpoint(cx)?;
            if snapshots.iter().any(|snapshot| snapshot.embedder_id() == id) {
                return Err(invalid("queries", "duplicate tier queries are ambiguous"));
            }
            snapshots.push(self.get(id).ok_or_else(|| {
                invalid("queries", "requested tier is not part of the captured cohort")
            })?);
        }
        checkpoint(cx)?;
        Ok(snapshots)
    }
}

impl Storage {
    /// Capture up to sixteen independently identified tiers in ONE transaction.
    ///
    /// Unlike repeated `load_published_search_snapshot` calls, all requested
    /// tiers see the same catalog, statuses, source text and publications. The
    /// single-tier reader's checksum, identity, source-hash and byte-admission
    /// checks remain the only record-admission path. A corrupt or incompatible
    /// later tier fails the entire capture; no usable prefix is returned.
    ///
    /// `limits` apply to the WHOLE cohort, not separately to each tier. Records
    /// and repeated source payloads count once per tier. A zero remaining budget
    /// admits only an empty subsequent tier, never a truncated nonempty one.
    /// This bounds stored payload, not process RSS or database execution time.
    ///
    /// All requests are validated before opening a transaction. An empty
    /// request list does no database work, but still observes cancellation.
    /// No inference, external-file reads, awaits or caller callbacks run in the
    /// transaction. Its configured begin mode can hold a writer lock. Schedule
    /// large captures on the caller's blocking execution path.
    pub fn load_published_search_cohort(
        &self,
        cx: &Cx,
        requests: &[PublishedSearchTierRequest<'_>],
        limits: PublishedSearchReadLimits,
    ) -> SearchResult<PublishedSearchCohort> {
        validate_requests(cx, requests)?;
        if requests.is_empty() {
            return Ok(PublishedSearchCohort {
                snapshots: Vec::new(),
                vector_stats: PublishedEmbeddingScanStats::default(),
                source_payload_bytes: 0,
            });
        }
        let result = self.transaction(|conn| {
            // The transaction may retry. Never retain a previous attempt's
            // snapshots or budget outside this closure.
            let mut remaining = limits;
            let mut snapshots = Vec::new();
            snapshots
                .try_reserve_exact(requests.len())
                .map_err(map_storage_error)?;
            for request in requests {
                checkpoint(cx)?;
                let snapshot = PublishedSearchSnapshot::capture_on(
                    conn,
                    cx,
                    request.embedder_id,
                    request.identity,
                    remaining,
                )?;
                let used = snapshot.vector_stats();
                remaining.embeddings.max_records = remaining
                    .embeddings
                    .max_records
                    .checked_sub(used.records)
                    .ok_or_else(accounting_error)?;
                remaining.embeddings.max_payload_bytes = remaining
                    .embeddings
                    .max_payload_bytes
                    .checked_sub(used.payload_bytes)
                    .ok_or_else(accounting_error)?;
                remaining.max_source_bytes = remaining
                    .max_source_bytes
                    .checked_sub(snapshot.source_payload_bytes())
                    .ok_or_else(accounting_error)?;
                snapshots.push(snapshot);
            }
            checkpoint(cx)?;
            Ok(PublishedSearchCohort {
                snapshots,
                vector_stats: PublishedEmbeddingScanStats {
                    records: limits.embeddings.max_records - remaining.embeddings.max_records,
                    payload_bytes: limits.embeddings.max_payload_bytes
                        - remaining.embeddings.max_payload_bytes,
                },
                source_payload_bytes: limits.max_source_bytes - remaining.max_source_bytes,
            })
        });
        checkpoint(cx)?;
        result
    }
}

fn validate_requests(cx: &Cx, requests: &[PublishedSearchTierRequest<'_>]) -> SearchResult<()> {
    checkpoint(cx)?;
    if requests.len() > MAX_COHORT_TIERS {
        return Err(invalid("tiers", "a cohort accepts at most sixteen tiers"));
    }
    for (index, request) in requests.iter().enumerate() {
        PublishedSearchSnapshot::validate_capture_request(
            cx,
            request.embedder_id,
            request.identity,
        )?;
        if requests[..index]
            .iter()
            .any(|previous| previous.embedder_id == request.embedder_id)
        {
            return Err(invalid(
                "tiers",
                "duplicate operational tier IDs are ambiguous",
            ));
        }
    }
    checkpoint(cx)
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.publication_cohort".to_owned(),
        reason: error.to_string(),
    })
}

fn accounting_error() -> SearchError {
    invalid(
        "limits",
        "captured tier exceeded its admitted remaining budget",
    )
}

fn invalid(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("publication_cohort.{field}"),
        value: "rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use frankensearch_core::IdentityBoundEmbedding;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::traits::{ModelCategory, SearchFuture};

    use super::*;
    use crate::{
        ClaimOutcome, ContentHasher, DocumentRecord, JobQueueConfig, PersistentJobQueue,
        PublishedEmbeddingReadLimits,
    };

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
        // Synthetic semantic contracts exercise identity separation, not model quality.
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
        tier: &str,
        body: &str,
        values: [f32; 2],
    ) {
        let hash = ContentHasher::hash(body);
        let mut document = DocumentRecord::new(id, body, hash, body.chars().count(), 1, 2);
        document.source_path = Some("missing://never-opened".to_owned());
        storage.upsert_document(&document).unwrap();
        storage.set_document_content(id, body).unwrap();
        queue.enqueue(id, tier, &hash, 0).unwrap();
        let claim = queue.claim_batch("publisher", 1).unwrap().pop().unwrap();
        let response = IdentityBoundEmbedding {
            values: values.to_vec(),
            identity: identity(tier),
        };
        assert_eq!(
            storage
                .publish_claimed_embedding(&claim, &hash, &response.identity, &response)
                .unwrap(),
            ClaimOutcome::Applied(())
        );
    }

    fn requests<'a>(
        fast: &'a EmbeddingIdentityBundleV1,
        quality: &'a EmbeddingIdentityBundleV1,
    ) -> [PublishedSearchTierRequest<'a>; 2] {
        [
            PublishedSearchTierRequest {
                embedder_id: "fast",
                identity: fast,
            },
            PublishedSearchTierRequest {
                embedder_id: "quality",
                identity: quality,
            },
        ]
    }

    fn load(storage: &Storage, cx: &Cx) -> PublishedSearchCohort {
        storage
            .load_published_search_cohort(
                cx,
                &requests(&identity("fast"), &identity("quality")),
                PublishedSearchReadLimits::default(),
            )
            .unwrap()
    }

    #[test]
    fn cohort_captures_independent_tiers_and_sources_with_one_commit() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "shared", "fast", "original", [1.0, 0.0]);
            publish(
                &storage, &queue, "shared", "quality", "original", [-1.0, 0.0],
            );
            publish(
                &storage, &queue, "quality-only", "quality", "extra", [1.0, 0.0],
            );
            let before = storage.metrics_snapshot();
            let cohort = load(&storage, &cx);
            let after = storage.metrics_snapshot();
            assert_eq!(after.tx_commits, before.tx_commits + 1);
            assert_eq!(after.tx_rollbacks, before.tx_rollbacks);
            assert!(!storage.connection().in_transaction());
            assert_eq!(cohort.vector_stats().records, 3);
            assert_eq!(cohort.snapshots().len(), 2);
            assert_eq!(cohort.snapshots()[0].embedder_id(), "fast");
            assert_eq!(cohort.snapshots()[1].embedder_id(), "quality");
            assert!(cohort.get("missing").is_none());
            let fast = cohort.get("fast").unwrap();
            let quality = cohort.get("quality").unwrap();
            assert_eq!(fast.len(), 1);
            assert_eq!(quality.len(), 2);
            assert_ne!(fast.identity(), quality.identity());
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("quality")).unwrap();
            let hits = quality.search(&cx, &query, 1).unwrap();
            assert_eq!(hits[0].hit().doc_id.as_str(), "quality-only");
            assert_eq!(hits[0].document().canonical_text(), "extra");
            assert!(fast.search(&cx, &query, 0).is_err());
            assert_eq!(
                cohort.source_payload_bytes(),
                fast.source_payload_bytes() + quality.source_payload_bytes()
            );
            assert_eq!(
                cohort.vector_stats().payload_bytes,
                fast.vector_stats().payload_bytes + quality.vector_stats().payload_bytes
            );
        });
    }

    #[test]
    fn cohort_budgets_are_aggregate_and_refuse_partial_second_tiers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "doc", "fast", "body", [1.0, 0.0]);
            publish(&storage, &queue, "doc", "quality", "body", [0.0, 1.0]);
            let baseline = load(&storage, &cx);
            let exact = PublishedSearchReadLimits {
                embeddings: PublishedEmbeddingReadLimits {
                    max_records: baseline.vector_stats().records,
                    max_payload_bytes: baseline.vector_stats().payload_bytes,
                },
                max_source_bytes: baseline.source_payload_bytes(),
            };
            let fast = identity("fast");
            let quality = identity("quality");
            let requests = requests(&fast, &quality);
            assert_eq!(
                storage
                    .load_published_search_cohort(&cx, &requests, exact)
                    .unwrap()
                    .vector_stats(),
                baseline.vector_stats()
            );
            for budget in ["records", "vectors", "sources"] {
                let mut limits = exact;
                match budget {
                    "records" => limits.embeddings.max_records -= 1,
                    "vectors" => limits.embeddings.max_payload_bytes -= 1,
                    "sources" => limits.max_source_bytes -= 1,
                    _ => unreachable!(),
                }
                let before = storage.metrics_snapshot();
                assert!(
                    storage
                        .load_published_search_cohort(&cx, &requests, limits)
                        .is_err()
                );
                let after = storage.metrics_snapshot();
                assert_eq!(after.tx_commits, before.tx_commits, "{budget}");
                assert_eq!(after.tx_rollbacks, before.tx_rollbacks + 1, "{budget}");
                assert!(!storage.connection().in_transaction());
            }
        });
    }

    #[test]
    fn corrupt_later_tier_discards_the_cohort_without_changing_publications() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "a", "fast", "fast body", [1.0, 0.0]);
            publish(
                &storage, &queue, "z", "quality", "quality body", [0.0, 1.0],
            );
            storage.connection().execute_sync(
                "UPDATE published_embeddings SET vector_le = X'0000' WHERE embedder_id = 'quality';",
            ).unwrap();
            let before = storage.metrics_snapshot();
            let result = storage.load_published_search_cohort(
                &cx,
                &requests(&identity("fast"), &identity("quality")),
                PublishedSearchReadLimits::default(),
            );
            assert!(result.is_err());
            let after = storage.metrics_snapshot();
            assert_eq!(after.tx_commits, before.tx_commits);
            assert_eq!(after.tx_rollbacks, before.tx_rollbacks + 1);
            assert_eq!(queue.queue_depth().unwrap().completed, 2);
            assert!(!storage.connection().in_transaction());
        });
    }

    #[test]
    fn missing_source_or_foreign_identity_in_a_later_tier_is_not_partial_success() {
        for missing_source in [false, true] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, queue) = fixture();
                publish(&storage, &queue, "a", "fast", "fast body", [1.0, 0.0]);
                publish(
                    &storage, &queue, "z", "quality", "quality body", [0.0, 1.0],
                );
                if missing_source {
                    storage
                        .connection()
                        .execute_sync("DELETE FROM document_contents WHERE doc_id = 'z';")
                        .unwrap();
                }
                let quality = identity(if missing_source { "quality" } else { "foreign" });
                assert!(
                    storage
                        .load_published_search_cohort(
                            &cx,
                            &requests(&identity("fast"), &quality),
                            PublishedSearchReadLimits::default(),
                        )
                        .is_err()
                );
                assert!(!storage.connection().in_transaction());
            });
        }
    }

    #[test]
    fn cohort_preserves_old_sources_and_exposes_pending_coverage_after_refresh() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            publish(&storage, &queue, "doc", "fast", "old source", [1.0, 0.0]);
            publish(
                &storage, &queue, "doc", "quality", "old source", [0.0, 1.0],
            );
            let old = load(&storage, &cx);
            publish(&storage, &queue, "doc", "fast", "new source", [-1.0, 0.0]);
            let new = load(&storage, &cx);
            for tier in old.snapshots() {
                assert_eq!(tier.documents()[0].canonical_text(), "old source");
            }
            assert_eq!(
                new.get("fast").unwrap().documents()[0].canonical_text(),
                "new source"
            );
            assert!(new.get("quality").unwrap().is_empty());
            assert_eq!(new.get("quality").unwrap().identity(), &identity("quality"));
            assert_eq!(new.vector_stats().records, 1);
            assert_eq!(old.vector_stats().records, 2);
        });
    }

    #[test]
    fn requests_are_admitted_before_any_transaction_including_empty_tiers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, _) = fixture();
            let fast = identity("fast");
            let quality = identity("quality");
            let valid = requests(&fast, &quality);
            let before = storage.metrics_snapshot();
            let empty = storage
                .load_published_search_cohort(&cx, &[], PublishedSearchReadLimits::default())
                .unwrap();
            assert!(empty.snapshots().is_empty());
            assert_eq!(empty.vector_stats(), PublishedEmbeddingScanStats::default());
            assert_eq!(empty.source_payload_bytes(), 0);
            for invalid in [
                vec![valid[0], valid[0]],
                vec![valid[0]; MAX_COHORT_TIERS + 1],
                vec![
                    valid[0],
                    PublishedSearchTierRequest {
                        embedder_id: " ",
                        identity: &quality,
                    },
                ],
            ] {
                assert!(
                    storage
                        .load_published_search_cohort(
                            &cx,
                            &invalid,
                            PublishedSearchReadLimits::default(),
                        )
                        .is_err()
                );
                assert_eq!(storage.metrics_snapshot(), before);
            }
            let zero = PublishedSearchReadLimits {
                embeddings: PublishedEmbeddingReadLimits {
                    max_records: 0,
                    max_payload_bytes: 0,
                },
                max_source_bytes: 0,
            };
            let captured = storage
                .load_published_search_cohort(&cx, &valid, zero)
                .unwrap();
            assert_eq!(captured.snapshots().len(), 2);
            assert!(
                captured
                    .snapshots()
                    .iter()
                    .all(PublishedSearchSnapshot::is_empty)
            );
            assert_eq!(
                storage.metrics_snapshot().tx_commits,
                before.tx_commits + 1
            );
        });
    }

    #[test]
    fn cancellation_precedes_empty_or_invalid_request_shortcuts() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, _) = fixture();
            let fast = identity("fast");
            let quality = identity("quality");
            let valid = requests(&fast, &quality);
            let before = storage.metrics_snapshot();
            cx.cancel_with(asupersync::CancelKind::User, Some("stop coherent capture"));
            for requests in [
                vec![],
                valid.to_vec(),
                vec![valid[0]; MAX_COHORT_TIERS + 1],
            ] {
                assert!(matches!(
                    storage.load_published_search_cohort(
                        &cx,
                        &requests,
                        PublishedSearchReadLimits::default(),
                    ),
                    Err(SearchError::Cancelled { .. })
                ));
            }
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }

    fn query_fixture(cx: &Cx) -> (Arc<Storage>, PersistentJobQueue, PublishedSearchCohort) {
        let (storage, queue) = fixture();
        publish(&storage, &queue, "shared", "fast", "old body", [1.0, 0.0]);
        publish(&storage, &queue, "shared", "quality", "old body", [1.0, 0.0]);
        publish(&storage, &queue, "quality-only", "quality", "extra", [0.0, 1.0]);
        let cohort = load(&storage, cx);
        (storage, queue, cohort)
    }

    #[test]
    fn cohort_queries_keep_independent_candidate_pools_and_request_order() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, _queue, cohort) = query_fixture(&cx);
            let fast = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("fast")).unwrap();
            let quality = BoundQueryEmbedding::new(vec![0.0, 1.0], identity("quality")).unwrap();
            let requests = [
                PublishedSearchTierQuery {
                    embedder_id: "quality",
                    query: &quality,
                    limit: 1,
                },
                PublishedSearchTierQuery {
                    embedder_id: "fast",
                    query: &fast,
                    limit: 1,
                },
            ];
            let before = storage.metrics_snapshot();
            let results = cohort.search(&cx, &requests).unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].snapshot().embedder_id(), "quality");
            assert_eq!(results[0].hits()[0].hit().doc_id.as_str(), "quality-only");
            assert_eq!(results[0].hits()[0].document().canonical_text(), "extra");
            assert_eq!(results[1].snapshot().embedder_id(), "fast");
            assert_eq!(results[1].hits()[0].hit().doc_id.as_str(), "shared");
            for (result, request) in results.iter().zip(&requests) {
                assert!(std::ptr::eq(
                    result.snapshot(),
                    cohort.get(request.embedder_id).unwrap()
                ));
                assert_eq!(
                    &result.hits()[0].publication().embedding.identity,
                    request.query.identity()
                );
            }
            for request in requests {
                let selected = cohort.search(&cx, &[request]).unwrap();
                assert_eq!(selected.len(), 1);
                assert_eq!(selected[0].snapshot().embedder_id(), request.embedder_id);
                assert_eq!(selected[0].hits().len(), 1);
            }
            assert!(cohort.search(&cx, &[]).unwrap().is_empty());
            assert_eq!(storage.metrics_snapshot(), before);
        });
    }

    #[test]
    fn cohort_query_admission_rejects_unknown_duplicate_and_swapped_tiers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_storage, _queue, cohort) = query_fixture(&cx);
            let fast = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("fast")).unwrap();
            let request = PublishedSearchTierQuery {
                embedder_id: "fast",
                query: &fast,
                limit: 1,
            };
            for requests in [
                vec![request, request],
                vec![request; MAX_COHORT_TIERS + 1],
                vec![
                    request,
                    PublishedSearchTierQuery {
                        embedder_id: "unknown",
                        ..request
                    },
                ],
            ] {
                assert!(matches!(
                    cohort.search(&cx, &requests),
                    Err(SearchError::InvalidConfig { field, .. })
                        if field == "publication_cohort.queries"
                ));
            }
            for limit in [0, 1] {
                let swapped = PublishedSearchTierQuery {
                    embedder_id: "quality",
                    limit,
                    ..request
                };
                // The first tier may already have scored; the whole response
                // is still refused rather than returning that successful prefix.
                assert!(matches!(
                    cohort.search(&cx, &[request, swapped]),
                    Err(SearchError::InvalidConfig { field, .. })
                        if field == "publication_search.identity"
                ));
            }
        });
    }

    #[derive(Clone, Copy)]
    enum QueryReply {
        Normal,
        Error,
        Foreign,
        Drift,
        Cancel,
        ExplicitCancel,
    }

    struct QueryProbe {
        tier: &'static str,
        admitted: EmbeddingIdentityBundleV1,
        foreign: EmbeddingIdentityBundleV1,
        reply: QueryReply,
        drift: AtomicBool,
        calls: AtomicUsize,
        raw_calls: AtomicUsize,
        inputs: Mutex<Vec<String>>,
    }

    impl QueryProbe {
        fn new(tier: &'static str, reply: QueryReply) -> Self {
            Self {
                tier,
                admitted: identity(tier),
                foreign: identity("foreign"),
                reply,
                drift: AtomicBool::new(false),
                calls: AtomicUsize::new(0),
                raw_calls: AtomicUsize::new(0),
                inputs: Mutex::new(Vec::new()),
            }
        }

        fn request<'a>(
            &'a self,
            canonicalizer: &'a dyn Canonicalizer,
            limit: usize,
        ) -> PublishedSearchTextQuery<'a> {
            PublishedSearchTextQuery {
                embedder_id: self.tier,
                embedder: self,
                canonicalizer,
                limit,
            }
        }
    }

    impl Embedder for QueryProbe {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.drift.load(Ordering::SeqCst) {
                &self.foreign
            } else {
                &self.admitted
            })
        }

        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.raw_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err(invalid("probe", "raw inference is forbidden")) })
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
                if matches!(self.reply, QueryReply::Cancel) {
                    cx.cancel_with(asupersync::CancelKind::User, Some("cancel quality query"));
                }
                if matches!(self.reply, QueryReply::ExplicitCancel) {
                    return Err(SearchError::Cancelled {
                        phase: "probe".to_owned(),
                        reason: "explicit cancellation".to_owned(),
                    });
                }
                if matches!(self.reply, QueryReply::Error | QueryReply::Cancel) {
                    return Err(invalid("probe", "bound inference failed"));
                }
                if matches!(self.reply, QueryReply::Drift) {
                    self.drift.store(true, Ordering::SeqCst);
                }
                Ok(IdentityBoundEmbedding {
                    values: if self.tier == "quality" {
                        vec![0.0, 1.0]
                    } else {
                        vec![1.0, 0.0]
                    },
                    identity: if matches!(self.reply, QueryReply::Foreign) {
                        self.foreign.clone()
                    } else {
                        self.admitted.clone()
                    },
                })
            })
        }

        fn dimension(&self) -> usize {
            2
        }

        fn id(&self) -> &'static str {
            self.tier
        }

        fn model_name(&self) -> &'static str {
            "cohort-query-probe"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    struct QueryPolicy(&'static str);

    impl Canonicalizer for QueryPolicy {
        fn canonicalize(&self, _text: &str) -> String {
            panic!("document canonicalization must not run for a query")
        }

        fn canonicalize_query(&self, text: &str) -> String {
            if text.trim().is_empty() {
                String::new()
            } else {
                format!("{}:{}", self.0, text.trim())
            }
        }
    }

    #[test]
    fn text_queries_use_independent_bound_providers_and_captured_sources() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue, cohort) = query_fixture(&cx);
            publish(&storage, &queue, "shared", "fast", "new body", [-1.0, 0.0]);
            let fast = QueryProbe::new("fast", QueryReply::Normal);
            let quality = QueryProbe::new("quality", QueryReply::Normal);
            let fast_policy = QueryPolicy("fast-query");
            let quality_policy = QueryPolicy("quality-query");
            let requests = [
                fast.request(&fast_policy, 1),
                quality.request(&quality_policy, 1),
            ];
            let before = storage.metrics_snapshot();
            let results = cohort
                .search_text(&cx, "  question  ", &requests)
                .await
                .unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].hits()[0].hit().doc_id.as_str(), "shared");
            assert_eq!(results[0].hits()[0].document().canonical_text(), "old body");
            assert_eq!(results[1].hits()[0].hit().doc_id.as_str(), "quality-only");
            assert_eq!(results[1].hits()[0].document().canonical_text(), "extra");
            assert_eq!(storage.metrics_snapshot(), before);
            for (provider, expected) in [
                (&fast, "fast-query:question"),
                (&quality, "quality-query:question"),
            ] {
                assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    provider.inputs.lock().unwrap().as_slice(),
                    &[expected.to_owned()]
                );
            }
            drop(results);
            drop(queue);
            drop(storage);
            // Quality-only retrieval needs neither the fast query nor a live DB.
            let results = cohort
                .search_text(&cx, "question", &requests[1..])
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].snapshot().embedder_id(), "quality");
            assert_eq!(results[0].hits()[0].hit().doc_id.as_str(), "quality-only");
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
        });
    }

    #[test]
    fn text_query_name_failures_are_resolved_before_any_provider_call() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_storage, _queue, cohort) = query_fixture(&cx);
            let fast = QueryProbe::new("fast", QueryReply::Normal);
            let policy = QueryPolicy("query");
            let request = fast.request(&policy, 1);
            for requests in [
                vec![request, request],
                vec![request; MAX_COHORT_TIERS + 1],
                vec![
                    request,
                    PublishedSearchTextQuery {
                        embedder_id: "unknown",
                        ..request
                    },
                ],
            ] {
                assert!(matches!(
                    cohort.search_text(&cx, "question", &requests).await,
                    Err(SearchError::InvalidConfig { field, .. })
                        if field == "publication_cohort.queries"
                ));
            }
            assert!(
                cohort
                    .search_text(&cx, "question", &[])
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(fast.raw_calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn secondary_query_failure_or_drift_never_returns_partial_success_or_retries() {
        for reply in [
            QueryReply::Error,
            QueryReply::Foreign,
            QueryReply::Drift,
            QueryReply::Cancel,
            QueryReply::ExplicitCancel,
        ] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (storage, _queue, cohort) = query_fixture(&cx);
                let fast = QueryProbe::new("fast", QueryReply::Normal);
                let quality = QueryProbe::new("quality", reply);
                let policy = QueryPolicy("query");
                let before = storage.metrics_snapshot();
                let error = cohort
                    .search_text(
                        &cx,
                        "question",
                        &[fast.request(&policy, 1), quality.request(&policy, 1)],
                    )
                    .await
                    .unwrap_err();
                match reply {
                    QueryReply::Cancel | QueryReply::ExplicitCancel => {
                        assert!(matches!(error, SearchError::Cancelled { .. }));
                    }
                    QueryReply::Foreign | QueryReply::Drift => {
                        assert!(matches!(
                            error,
                            SearchError::InvalidConfig { field, .. }
                                if field == "publication_search.identity"
                        ));
                    }
                    QueryReply::Error => {
                        assert!(matches!(
                            error,
                            SearchError::InvalidConfig { field, .. }
                                if field == "publication_cohort.probe"
                        ));
                    }
                    QueryReply::Normal => unreachable!(),
                }
                for provider in [&fast, &quality] {
                    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
                }
                assert_eq!(storage.metrics_snapshot(), before);
            });
        }
    }

    #[test]
    fn text_query_noops_still_admit_each_captured_provider() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, queue) = fixture();
            let empty = load(&storage, &cx);
            publish(&storage, &queue, "doc", "fast", "body", [1.0, 0.0]);
            publish(&storage, &queue, "doc", "quality", "body", [0.0, 1.0]);
            let full = load(&storage, &cx);
            let fast = QueryProbe::new("fast", QueryReply::Normal);
            let quality = QueryProbe::new("quality", QueryReply::Normal);
            let policy = QueryPolicy("query");
            for (cohort, text, limit) in [
                (&empty, "question", 1),
                (&full, "question", 0),
                (&full, " \t ", 1),
            ] {
                let requests = [fast.request(&policy, limit), quality.request(&policy, limit)];
                let results = cohort.search_text(&cx, text, &requests).await.unwrap();
                assert_eq!(results.len(), 2);
                assert!(results.iter().all(|result| result.hits().is_empty()));
                quality.drift.store(true, Ordering::SeqCst);
                assert!(matches!(
                    cohort.search_text(&cx, text, &requests).await,
                    Err(SearchError::InvalidConfig { field, .. })
                        if field == "publication_search.identity"
                ));
                quality.drift.store(false, Ordering::SeqCst);
            }
            for provider in [&fast, &quality] {
                assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
                assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
            }
        });
    }

    #[test]
    fn cohort_query_cancellation_precedes_empty_zero_and_invalid_requests() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_storage, _queue, cohort) = query_fixture(&cx);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("fast")).unwrap();
            let provider = QueryProbe::new("fast", QueryReply::Normal);
            let policy = QueryPolicy("query");
            cx.cancel_with(asupersync::CancelKind::User, Some("stop cohort query"));
            assert!(matches!(
                cohort.search(&cx, &[]),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                cohort.search_text(&cx, "", &[]).await,
                Err(SearchError::Cancelled { .. })
            ));
            for name in ["fast", "unknown"] {
                assert!(matches!(
                    cohort.search(
                        &cx,
                        &[PublishedSearchTierQuery {
                            embedder_id: name,
                            query: &query,
                            limit: 0,
                        }],
                    ),
                    Err(SearchError::Cancelled { .. })
                ));
                let request = PublishedSearchTextQuery {
                    embedder_id: name,
                    ..provider.request(&policy, 0)
                };
                assert!(matches!(
                    cohort.search_text(&cx, "", &[request]).await,
                    Err(SearchError::Cancelled { .. })
                ));
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        });
    }
}
