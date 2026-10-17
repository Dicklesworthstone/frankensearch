//! Cross-encoder refinement over source-owned candidates from one database cohort.
//!
//! Candidate retrieval remains independent per tier. This module unions those
//! candidates by document ID and supplies the captured canonical text to the
//! existing `Reranker` trait, never a later catalog or external-file lookup.

use std::collections::{BTreeMap, BTreeSet};

use asupersync::Cx;
use frankensearch_core::traits::{RerankDocument, RerankScore, Reranker};
use frankensearch_core::{SearchError, SearchResult};

use crate::connection::map_storage_error;
use crate::{PublishedSearchCohort, PublishedSearchDocument, PublishedSearchTierHits};

/// Admission budgets for one complete candidate union, not permission to
/// truncate an oversized union and silently discard later-tier candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishedRerankLimits {
    /// Maximum distinct documents supplied to the cross-encoder.
    pub max_candidates: usize,
    /// Query bytes plus the ID and canonical-text bytes of each distinct
    /// document. Shared documents are charged once. This is not an RSS limit.
    pub max_input_bytes: usize,
    /// Independent maximum query length in UTF-8 bytes, before any callback.
    pub max_query_bytes: usize,
}

impl Default for PublishedRerankLimits {
    fn default() -> Self {
        Self {
            max_candidates: 512,
            max_input_bytes: 8 * 1024 * 1024,
            max_query_bytes: 64 * 1024,
        }
    }
}

/// One cross-encoder result and the exact retained document it scored.
///
/// The score is not a cosine or an RRF score. `original_rank` is the document's
/// first occurrence in request-tier order, then each tier's ranked hit order.
/// Returned values borrow the cohort, not the temporary candidate hit lists.
#[derive(Debug)]
pub struct PublishedRerankedHit<'cohort> {
    document: &'cohort PublishedSearchDocument,
    score: RerankScore,
}

impl<'cohort> PublishedRerankedHit<'cohort> {
    #[must_use]
    pub const fn document(&self) -> &'cohort PublishedSearchDocument {
        self.document
    }

    #[must_use]
    pub const fn score(&self) -> &RerankScore {
        &self.score
    }
}

impl PublishedSearchCohort {
    /// Refine the complete union of independently retrieved tier candidates.
    ///
    /// Every input tier must belong to THIS retained cohort, including empty
    /// tiers. Separately captured or cloned cohorts cannot be mixed, even if
    /// their operational IDs happen to agree. Duplicate tiers are refused.
    /// Shared documents are scored once; quality-only candidates are retained.
    /// Input order supplies deterministic ties, not cross-model score blending.
    ///
    /// The cross-encoder receives the captured canonical embedding input, which
    /// may already have been shortened during ingestion, not an invented full
    /// raw source. Query preprocessing and the provider's token policy remain
    /// caller-owned. No canonicalizer, database, filesystem, model download,
    /// runtime, or independent task is invoked here.
    ///
    /// Admission covers the WHOLE union before copying text or calling the
    /// provider. Every response must bind one exact input ID and original rank;
    /// duplicates, missing/foreign IDs, mismatched ranks, and nonfinite scores
    /// or raw logits fail the whole refinement, even outside the output limit.
    /// Output is sorted locally by score, then original rank, before truncation.
    ///
    /// Zero limits, empty unions and blank queries skip provider callbacks but
    /// still validate ownership and budgets. Cancellation dominates failures.
    /// Failure leaves the caller's original tier results available for explicit
    /// fallback; it is never labelled successful refinement. Provider deadlines
    /// are caller-owned. Dropping this future drops its inference future; an
    /// internally detached provider must implement its own teardown. Synchronous
    /// callbacks cannot be preempted.
    pub async fn rerank_candidates<'cohort>(
        &'cohort self,
        cx: &Cx,
        reranker: &dyn Reranker,
        query: &str,
        tiers: &[PublishedSearchTierHits<'cohort>],
        limit: usize,
        limits: PublishedRerankLimits,
    ) -> SearchResult<Vec<PublishedRerankedHit<'cohort>>> {
        let candidates = self.rerank_candidate_union(cx, query, tiers, limits)?;
        if limit == 0 || candidates.is_empty() || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        checkpoint(cx)?;
        let available = reranker.is_available();
        checkpoint(cx)?;
        if !available {
            return Err(SearchError::RerankerUnavailable {
                model: "storage.publication_rerank".to_owned(),
            });
        }
        let mut documents = Vec::new();
        documents
            .try_reserve_exact(candidates.len())
            .map_err(map_storage_error)?;
        for document in &candidates {
            checkpoint(cx)?;
            documents.push(RerankDocument {
                doc_id: copy_text(&document.catalog().doc_id)?,
                text: copy_text(document.canonical_text())?,
            });
        }
        checkpoint(cx)?;
        let response = reranker.rerank(cx, query, &documents).await;
        checkpoint(cx)?;
        let scores = response?;
        // Validate before assembling ANY output: providers are not trusted to
        // preserve membership, positions, finiteness or their advertised order.
        validate_scores(cx, &documents, &scores)?;
        let mut results = Vec::new();
        results
            .try_reserve_exact(scores.len())
            .map_err(map_storage_error)?;
        for score in scores {
            checkpoint(cx)?;
            let document = candidates[score.original_rank];
            results.push(PublishedRerankedHit { document, score });
        }
        results.sort_unstable_by(|left, right| {
            right
                .score
                .score
                .total_cmp(&left.score.score)
                .then_with(|| left.score.original_rank.cmp(&right.score.original_rank))
        });
        results.truncate(limit);
        checkpoint(cx)?;
        Ok(results)
    }

    fn rerank_candidate_union<'cohort>(
        &'cohort self,
        cx: &Cx,
        query: &str,
        tiers: &[PublishedSearchTierHits<'cohort>],
        limits: PublishedRerankLimits,
    ) -> SearchResult<Vec<&'cohort PublishedSearchDocument>> {
        checkpoint(cx)?;
        if query.len() > limits.max_query_bytes || query.len() > limits.max_input_bytes {
            return Err(invalid("query exceeds the reranking input budget"));
        }
        if tiers.len() > self.snapshots().len() {
            return Err(invalid("too many candidate tiers for the retained cohort"));
        }
        let mut tier_ids = BTreeSet::new();
        let mut positions: BTreeMap<&str, usize> = BTreeMap::new();
        let mut candidates: Vec<&PublishedSearchDocument> = Vec::new();
        let mut bytes = query.len();
        for tier in tiers {
            checkpoint(cx)?;
            let snapshot = tier.snapshot();
            if !self
                .get(snapshot.embedder_id())
                .is_some_and(|owner| std::ptr::eq(owner, snapshot))
                || !tier_ids.insert(snapshot.embedder_id())
            {
                return Err(invalid("candidate tiers must be distinct owners from this cohort"));
            }
            let mut seen = BTreeSet::new();
            for hit in tier.hits() {
                checkpoint(cx)?;
                let index = usize::try_from(hit.hit().index)
                    .map_err(|_| invalid("candidate position exceeds the address space"))?;
                let document = snapshot
                    .documents()
                    .get(index)
                    .ok_or_else(|| invalid("candidate is outside its retained source owner"))?;
                let id = document.catalog().doc_id.as_str();
                if hit.hit().doc_id.as_str() != id
                    || !std::ptr::eq(hit.document(), document)
                    || hit.publication().content_hash != document.catalog().content_hash
                    || !hit.hit().score.is_finite()
                    || !seen.insert(id)
                {
                    return Err(invalid("candidate does not bind a unique retained document"));
                }
                if let Some(&position) = positions.get(id) {
                    let previous: &PublishedSearchDocument = candidates[position];
                    if previous.catalog().content_hash != document.catalog().content_hash
                        || previous.canonical_text() != document.canonical_text()
                    {
                        return Err(invalid("candidate tiers disagree on their captured source"));
                    }
                    continue;
                }
                if candidates.len() == limits.max_candidates {
                    return Err(invalid("complete candidate union exceeds the count budget"));
                }
                bytes = bytes
                    .checked_add(id.len())
                    .and_then(|total| total.checked_add(document.canonical_text().len()))
                    .filter(|total| *total <= limits.max_input_bytes)
                    .ok_or_else(|| invalid("complete candidate union exceeds the byte budget"))?;
                candidates.try_reserve(1).map_err(map_storage_error)?;
                positions.insert(id, candidates.len());
                candidates.push(document);
            }
        }
        checkpoint(cx)?;
        Ok(candidates)
    }
}

fn copy_text(text: &str) -> SearchResult<String> {
    let mut copied = String::new();
    copied
        .try_reserve_exact(text.len())
        .map_err(map_storage_error)?;
    copied.push_str(text);
    Ok(copied)
}

fn validate_scores(
    cx: &Cx,
    documents: &[RerankDocument],
    scores: &[RerankScore],
) -> SearchResult<()> {
    checkpoint(cx)?;
    if scores.len() != documents.len() {
        return Err(invalid_response("one reranking result is required per candidate"));
    }
    let mut seen = Vec::new();
    seen.try_reserve_exact(documents.len())
        .map_err(map_storage_error)?;
    seen.resize(documents.len(), false);
    for score in scores {
        checkpoint(cx)?;
        let Some(document) = documents.get(score.original_rank) else {
            return Err(invalid_response("reranker returned an out-of-range original rank"));
        };
        if score.doc_id != document.doc_id
            || seen[score.original_rank]
            || !score.score.is_finite()
            || score.raw_logit.is_some_and(|logit| !logit.is_finite())
        {
            return Err(invalid_response("reranker returned invalid membership, rank or score"));
        }
        seen[score.original_rank] = true;
    }
    checkpoint(cx)
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.publication_rerank".to_owned(),
        reason: error.to_string(),
    })
}

fn invalid(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "publication_rerank.input".to_owned(),
        value: "rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

fn invalid_response(reason: &str) -> SearchError {
    SearchError::RerankFailed {
        model: "storage.publication_rerank".to_owned(),
        source: Box::new(std::io::Error::other(reason.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::traits::SearchFuture;
    use frankensearch_core::{BoundQueryEmbedding, EmbeddingIdentityBundleV1, IdentityBoundEmbedding};

    use super::*;
    use crate::{
        ClaimOutcome, ContentHasher, DocumentRecord, JobQueueConfig, PersistentJobQueue,
        PublishedSearchReadLimits, PublishedSearchTierQuery, PublishedSearchTierRequest, Storage,
    };

    fn identity(tier: &str) -> EmbeddingIdentityBundleV1 {
        // Synthetic admission fixtures, not model-quality evidence.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(tier, 2);
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
        document.source_path = Some("missing://must-not-read".to_owned());
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

    fn load(storage: &Storage, cx: &Cx) -> PublishedSearchCohort {
        storage
            .load_published_search_cohort(
                cx,
                &[
                    PublishedSearchTierRequest { embedder_id: "fast", identity: &identity("fast") },
                    PublishedSearchTierRequest { embedder_id: "quality", identity: &identity("quality") },
                ],
                PublishedSearchReadLimits::default(),
            )
            .unwrap()
    }

    fn fixture(cx: &Cx) -> (Arc<Storage>, PublishedSearchCohort) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        publish(&storage, &queue, "a", "fast", "first source", [1.0, 0.0]);
        publish(&storage, &queue, "shared", "fast", "shared α source", [0.0, 1.0]);
        publish(&storage, &queue, "shared", "quality", "shared α source", [1.0, 0.0]);
        publish(&storage, &queue, "quality-only", "quality", "target source", [-1.0, 0.0]);
        let cohort = load(&storage, cx);
        (storage, cohort)
    }

    fn retrieve<'a>(cohort: &'a PublishedSearchCohort, cx: &Cx) -> Vec<PublishedSearchTierHits<'a>> {
        let fast = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("fast")).unwrap();
        let quality = BoundQueryEmbedding::new(vec![1.0, 0.0], identity("quality")).unwrap();
        cohort.search(cx, &[
            PublishedSearchTierQuery { embedder_id: "fast", query: &fast, limit: 10 },
            PublishedSearchTierQuery { embedder_id: "quality", query: &quality, limit: 10 },
        ]).unwrap()
    }

    #[derive(Clone, Copy)]
    enum Reply {
        Normal,
        Ties,
        Missing,
        Extra,
        Duplicate,
        Foreign,
        Rank,
        Nan,
        InfiniteLogit,
        Error,
        Cancel,
        ExplicitCancel,
        Pending,
        Unavailable,
    }

    struct Probe {
        reply: Reply,
        calls: AtomicUsize,
        availability_calls: AtomicUsize,
        inputs: Mutex<Vec<(String, String)>>,
        dropped: Arc<AtomicBool>,
    }

    impl Probe {
        fn new(reply: Reply) -> Self {
            Self {
                reply,
                calls: AtomicUsize::new(0),
                availability_calls: AtomicUsize::new(0),
                inputs: Mutex::new(Vec::new()),
                dropped: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    struct OnDrop(Arc<AtomicBool>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    impl Reranker for Probe {
        fn rerank<'a>(
            &'a self,
            cx: &'a Cx,
            query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                let _guard = OnDrop(Arc::clone(&self.dropped));
                self.calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(query, "query");
                *self.inputs.lock().unwrap() = documents.iter()
                    .map(|document| (document.doc_id.clone(), document.text.clone())).collect();
                if matches!(self.reply, Reply::Pending) {
                    std::future::pending::<()>().await;
                }
                if matches!(self.reply, Reply::Cancel) {
                    cx.cancel_with(asupersync::CancelKind::User, Some("stop refinement"));
                    return Err(invalid_response("ordinary provider error"));
                }
                if matches!(self.reply, Reply::ExplicitCancel) {
                    return Err(SearchError::Cancelled {
                        phase: "test-provider".to_owned(), reason: "explicit stop".to_owned(),
                    });
                }
                if matches!(self.reply, Reply::Error) {
                    return Err(invalid_response("ordinary provider error"));
                }
                let mut scores: Vec<_> = documents.iter().enumerate().map(|(rank, document)| {
                    RerankScore {
                        doc_id: document.doc_id.clone(),
                        score: if matches!(self.reply, Reply::Ties) { -2.0 }
                            else if document.text == "target source" { 7.0 } else { -2.0 },
                        original_rank: rank,
                        raw_logit: None,
                    }
                }).collect();
                match self.reply {
                    Reply::Missing => { scores.pop(); }
                    Reply::Extra => scores.push(scores[0].clone()),
                    Reply::Duplicate => scores[1] = scores[0].clone(),
                    Reply::Foreign => scores[1].doc_id = "private-score-canary".to_owned(),
                    Reply::Rank => scores[1].original_rank = usize::MAX,
                    Reply::Nan => scores[1].score = f32::NAN,
                    Reply::InfiniteLogit => scores[1].raw_logit = Some(f32::INFINITY),
                    _ => {}
                }
                scores.reverse(); // Never trust provider ordering, including ties.
                Ok(scores)
            })
        }
        fn id(&self) -> &str { "retained-rerank-probe" }
        fn model_name(&self) -> &str { self.id() }
        fn is_available(&self) -> bool {
            self.availability_calls.fetch_add(1, Ordering::SeqCst);
            !matches!(self.reply, Reply::Unavailable)
        }
    }

    #[test]
    fn union_reranks_quality_only_candidates_once_with_retained_sources_after_database_drop() {
        run_test_with_cx(|cx| async move {
            let (storage, cohort) = fixture(&cx);
            let tiers = retrieve(&cohort, &cx);
            assert_eq!(tiers[0].hits().len() + tiers[1].hits().len(), 4);
            storage.connection().execute_sync("DELETE FROM documents;").unwrap();
            drop(storage);
            let probe = Probe::new(Reply::Normal);
            let results = cohort.rerank_candidates(
                &cx, &probe, "query", &tiers, 2, PublishedRerankLimits::default(),
            ).await.unwrap();
            drop(tiers); // Result source lifetime is the cohort's, not these lists'.
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].document().catalog().doc_id, "quality-only");
            assert_eq!(results[0].document().canonical_text(), "target source");
            assert_eq!(results[0].score().original_rank, 2);
            assert_eq!(results[1].document().catalog().doc_id, "a");
            assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
            assert_eq!(*probe.inputs.lock().unwrap(), [
                ("a".to_owned(), "first source".to_owned()),
                ("shared".to_owned(), "shared α source".to_owned()),
                ("quality-only".to_owned(), "target source".to_owned()),
            ]);
        });
    }

    #[test]
    fn all_response_rows_are_validated_before_limit_or_any_refined_output() {
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let tiers = retrieve(&cohort, &cx);
            for reply in [Reply::Missing, Reply::Extra, Reply::Duplicate, Reply::Foreign,
                Reply::Rank, Reply::Nan, Reply::InfiniteLogit] {
                let probe = Probe::new(reply);
                let error = cohort.rerank_candidates(
                    &cx, &probe, "query", &tiers, 1, PublishedRerankLimits::default(),
                ).await.unwrap_err();
                assert!(matches!(error, SearchError::RerankFailed { .. }));
                assert!(!error.to_string().contains("private-score-canary"));
                assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
                assert!(probe.dropped.load(Ordering::SeqCst));
                assert_eq!(tiers[0].hits()[0].document().catalog().doc_id, "a");
            }
        });
    }

    #[test]
    fn tied_and_negative_scores_preserve_first_occurrence_not_provider_order() {
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let tiers = retrieve(&cohort, &cx);
            let probe = Probe::new(Reply::Ties);
            let results = cohort.rerank_candidates(
                &cx, &probe, "query", &tiers, usize::MAX, PublishedRerankLimits::default(),
            ).await.unwrap();
            assert_eq!(results.iter().map(|hit| hit.score().original_rank).collect::<Vec<_>>(), [0, 1, 2]);
            assert!(results.iter().all(|hit| hit.score().score.to_bits() == (-2.0_f32).to_bits()));
        });
    }

    #[test]
    fn budgets_charge_utf8_unique_sources_and_fail_before_provider_callbacks() {
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let tiers = retrieve(&cohort, &cx);
            let bytes = "query".len() + "a".len() + "first source".len()
                + "shared".len() + "shared α source".len()
                + "quality-only".len() + "target source".len();
            let exact = PublishedRerankLimits {
                max_candidates: 3, max_input_bytes: bytes, max_query_bytes: 5,
            };
            let probe = Probe::new(Reply::Normal);
            assert_eq!(cohort.rerank_candidates(&cx, &probe, "query", &tiers, 3, exact)
                .await.unwrap().len(), 3);
            for limits in [
                PublishedRerankLimits { max_candidates: 2, ..exact },
                PublishedRerankLimits { max_input_bytes: bytes - 1, ..exact },
                PublishedRerankLimits { max_query_bytes: 4, ..exact },
            ] {
                let refused = Probe::new(Reply::Normal);
                assert!(matches!(cohort.rerank_candidates(&cx, &refused, "query", &tiers, 1, limits)
                    .await, Err(SearchError::InvalidConfig { .. })));
                assert_eq!(refused.availability_calls.load(Ordering::SeqCst), 0);
                assert_eq!(refused.calls.load(Ordering::SeqCst), 0);
            }
        });
    }

    #[test]
    fn foreign_and_duplicate_tier_owners_are_refused_even_for_noop_requests() {
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let clone = cohort.clone();
            let foreign = retrieve(&clone, &cx);
            let probe = Probe::new(Reply::Normal);
            for limit in [0, 2] {
                assert!(cohort.rerank_candidates(
                    &cx, &probe, "query", &foreign, limit, PublishedRerankLimits::default(),
                ).await.is_err());
            }
            let mut duplicate = retrieve(&cohort, &cx);
            duplicate.pop();
            let mut another = retrieve(&cohort, &cx);
            duplicate.push(another.remove(0));
            assert!(cohort.rerank_candidates(
                &cx, &probe, "query", &duplicate, 2, PublishedRerankLimits::default(),
            ).await.is_err());
            assert_eq!(probe.availability_calls.load(Ordering::SeqCst), 0);
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn noops_and_failures_never_silently_retry_or_replace_original_results() {
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let tiers = retrieve(&cohort, &cx);
            let probe = Probe::new(Reply::Unavailable);
            for (text, limit) in [("query", 0), (" \t ", 2)] {
                assert!(cohort.rerank_candidates(
                    &cx, &probe, text, &tiers, limit, PublishedRerankLimits::default(),
                ).await.unwrap().is_empty());
            }
            assert!(cohort.rerank_candidates(
                &cx, &probe, "query", &[], 2, PublishedRerankLimits::default(),
            ).await.unwrap().is_empty());
            assert_eq!(probe.availability_calls.load(Ordering::SeqCst), 0);
            assert!(matches!(cohort.rerank_candidates(
                &cx, &probe, "query", &tiers, 2, PublishedRerankLimits::default(),
            ).await, Err(SearchError::RerankerUnavailable { .. })));
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
            let failing = Probe::new(Reply::Error);
            assert!(matches!(cohort.rerank_candidates(
                &cx, &failing, "query", &tiers, 2, PublishedRerankLimits::default(),
            ).await, Err(SearchError::RerankFailed { .. })));
            assert_eq!(failing.calls.load(Ordering::SeqCst), 1);
            assert_eq!(tiers[1].hits().len(), 2);
        });
    }

    #[test]
    fn cancellation_wins_over_provider_error_and_before_input_admission() {
        for reply in [Reply::Cancel, Reply::ExplicitCancel] {
            run_test_with_cx(|cx| async move {
                let (_storage, cohort) = fixture(&cx);
                let tiers = retrieve(&cohort, &cx);
                let probe = Probe::new(reply);
                assert!(matches!(cohort.rerank_candidates(
                    &cx, &probe, "query", &tiers, 2, PublishedRerankLimits::default(),
                ).await, Err(SearchError::Cancelled { .. })));
                assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
            });
        }
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let probe = Probe::new(Reply::Normal);
            cx.cancel_with(asupersync::CancelKind::User, Some("before admission"));
            assert!(matches!(cohort.rerank_candidates(
                &cx, &probe, "query", &[], 0,
                PublishedRerankLimits { max_query_bytes: 0, ..PublishedRerankLimits::default() },
            ).await, Err(SearchError::Cancelled { .. })));
            assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn abandoning_pending_rerank_drops_provider_future_without_cancelling_caller() {
        run_test_with_cx(|cx| async move {
            let (_storage, cohort) = fixture(&cx);
            let tiers = retrieve(&cohort, &cx);
            let probe = Probe::new(Reply::Pending);
            let mut future = Box::pin(cohort.rerank_candidates(
                &cx, &probe, "query", &tiers, 2, PublishedRerankLimits::default(),
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            assert!(!probe.dropped.load(Ordering::SeqCst));
            drop(future);
            assert!(probe.dropped.load(Ordering::SeqCst));
            assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
            assert!(!cx.is_cancel_requested());
            assert_eq!(tiers[0].hits().len(), 2);
        });
    }
}
