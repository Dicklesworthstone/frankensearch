//! Source-coherent capture of independently identified publication tiers.
//!
//! A cohort uses one database transaction for every vector and its source.
//! It does not join separately loaded snapshots or make independently committed
//! embedding jobs atomic. Missing/pending coverage remains explicit per tier.

use asupersync::Cx;
use frankensearch_core::{EmbeddingIdentityBundleV1, SearchError, SearchResult};

use crate::connection::map_storage_error;
use crate::{
    PublishedEmbeddingScanStats, PublishedSearchReadLimits, PublishedSearchSnapshot, Storage,
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

    use std::sync::Arc;

    use frankensearch_core::IdentityBoundEmbedding;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::types::BoundQueryEmbedding;

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
}
