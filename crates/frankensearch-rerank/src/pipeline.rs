//! Rerank pipeline step: integrates cross-encoder reranking into the search pipeline.
//!
//! The [`rerank_step`] function reranks a set of `ScoredResult` candidates using a
//! [`Reranker`] implementation. It looks up document text via a caller-provided closure,
//! gracefully skips reranking on any failure, and supports cancellation via `&Cx`.

use asupersync::Cx;
use tracing::instrument;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{RerankDocument, Reranker};
use frankensearch_core::types::{ScoreSource, ScoredResult};

/// Default maximum number of candidates to rerank per query.
pub const DEFAULT_TOP_K_RERANK: usize = 100;

/// Default minimum number of candidates required to trigger reranking.
pub const DEFAULT_MIN_CANDIDATES: usize = 5;

/// Rerank the top candidates in-place using a cross-encoder model.
///
/// This function converts `ScoredResult` candidates into `RerankDocument` pairs,
/// runs cross-encoder inference, and updates `rerank_score` / `source` on each
/// result. The candidates slice is re-sorted by descending rerank score for the
/// reranked portion.
///
/// # Graceful Failure
///
/// This function **never** prevents search results from being returned:
/// - If `candidates.len() < min_candidates`: returns `Ok(())` unchanged.
/// - If `text_fn` returns `None` for a document: that document is skipped.
/// - If the reranker returns an error: returns `Ok(())` with candidates unchanged.
/// - If fewer than `min_candidates` have text available: returns `Ok(())` unchanged.
///
/// # Parameters
///
/// - `cx`: Capability context for cancellation.
/// - `reranker`: Cross-encoder reranker implementation.
/// - `query`: The original search query.
/// - `candidates`: Mutable slice of scored results to rerank (modified in-place).
/// - `text_fn`: Closure that retrieves document text by `doc_id`. Returns `None` if
///   the text is unavailable for a given document.
/// - `top_k_rerank`: Maximum number of top candidates to rerank.
/// - `min_candidates`: Minimum number of candidates required to trigger reranking.
///
/// # Errors
///
/// Returns `SearchError::Cancelled` if the operation was cancelled via `cx`.
/// All other reranker errors are caught and logged, returning `Ok(())`.
#[instrument(skip_all, fields(
    query_len = query.len(),
    num_candidates = candidates.len(),
    top_k = top_k_rerank,
))]
pub async fn rerank_step(
    cx: &Cx,
    reranker: &dyn Reranker,
    query: &str,
    candidates: &mut [ScoredResult],
    text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    top_k_rerank: usize,
    min_candidates: usize,
) -> SearchResult<()> {
    if candidates.len() < min_candidates {
        tracing::debug!(
            count = candidates.len(),
            min = min_candidates,
            "skipping rerank: too few candidates"
        );
        return Ok(());
    }

    let rerank_count = candidates.len().min(top_k_rerank);

    // Build RerankDocument pairs for candidates that have retrievable text.
    // Track which original indices were included (some may be skipped if text_fn returns None).
    let mut rerank_docs = Vec::with_capacity(rerank_count);
    let mut included_indices = Vec::with_capacity(rerank_count);

    for (i, candidate) in candidates.iter().take(rerank_count).enumerate() {
        if let Some(text) = text_fn(&candidate.doc_id) {
            rerank_docs.push(RerankDocument {
                doc_id: candidate.doc_id.clone(),
                text,
            });
            included_indices.push(i);
        }
    }

    if included_indices.len() < min_candidates {
        tracing::debug!(
            with_text = included_indices.len(),
            min = min_candidates,
            "skipping rerank: too few candidates with available text"
        );
        return Ok(());
    }

    // Run the reranker (graceful failure: catch non-cancellation errors)
    let scores = match reranker.rerank(cx, query, &rerank_docs).await {
        Ok(scores) => scores,
        Err(SearchError::Cancelled { phase, reason }) => {
            // Cancellation propagates up — this is not a graceful skip.
            return Err(SearchError::Cancelled { phase, reason });
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                model = reranker.id(),
                "reranker failed — keeping original scores"
            );
            return Ok(());
        }
    };

    // Validate score count matches
    if scores.len() != rerank_docs.len() {
        tracing::warn!(
            expected = rerank_docs.len(),
            got = scores.len(),
            "reranker score count mismatch — skipping rerank"
        );
        return Ok(());
    }

    // Apply rerank scores to the original candidates using `original_rank`.
    // This keeps scores aligned with the correct doc_id even after the reranker sorts its output.
    for score in scores {
        if let Some(&candidate_idx) = included_indices.get(score.original_rank) {
            if candidates[candidate_idx].doc_id != score.doc_id {
                tracing::warn!(
                    expected = %candidates[candidate_idx].doc_id,
                    got = %score.doc_id,
                    "reranker returned mismatched doc_id for original_rank {}",
                    score.original_rank
                );
            }
            candidates[candidate_idx].rerank_score = Some(score.score);
            candidates[candidate_idx].source = ScoreSource::Reranked;
        } else {
            tracing::warn!(
                rank = score.original_rank,
                "reranker returned original_rank outside included candidates"
            );
        }
    }

    // Re-sort the reranked portion by rerank_score descending (NaN-safe).
    // Non-reranked candidates (beyond top_k_rerank) keep their original order.
    candidates[..rerank_count].sort_by(|a, b| {
        let score_a = a.rerank_score.unwrap_or(f32::NEG_INFINITY);
        let score_b = b.rerank_score.unwrap_or(f32::NEG_INFINITY);
        score_b
            .total_cmp(&score_a)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    tracing::debug!(
        reranked = included_indices.len(),
        model = reranker.id(),
        "rerank step complete"
    );

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unnecessary_literal_bound)]
mod tests {
    use frankensearch_core::traits::{RerankScore, SearchFuture};

    use super::*;

    /// Stub reranker that assigns decreasing scores based on document order.
    struct StubReranker;

    impl Reranker for StubReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                let len = documents.len().max(1);
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| {
                        // Reverse the order: last doc gets highest score
                        #[allow(clippy::cast_precision_loss)]
                        let score = 1.0 - (i as f32 / len as f32);
                        RerankScore {
                            doc_id: doc.doc_id.clone(),
                            score,
                            original_rank: i,
                        }
                    })
                    .collect())
            })
        }

        fn id(&self) -> &str {
            "stub-reranker"
        }

        fn model_name(&self) -> &str {
            "stub-reranker"
        }
    }

    /// Stub reranker that always fails.
    struct FailingReranker;

    impl Reranker for FailingReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            _documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async {
                Err(SearchError::RerankFailed {
                    model: "fail-reranker".into(),
                    source: "intentional test failure".into(),
                })
            })
        }

        fn id(&self) -> &str {
            "fail-reranker"
        }

        fn model_name(&self) -> &str {
            "fail-reranker"
        }
    }

    /// Stub reranker that returns wrong number of scores.
    struct MismatchReranker;

    impl Reranker for MismatchReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            _documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async {
                // Return only 1 score regardless of input
                Ok(vec![RerankScore {
                    doc_id: "only".into(),
                    score: 0.5,
                    original_rank: 0,
                }])
            })
        }

        fn id(&self) -> &str {
            "mismatch-reranker"
        }

        fn model_name(&self) -> &str {
            "mismatch-reranker"
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn make_candidates(n: usize) -> Vec<ScoredResult> {
        (0..n)
            .map(|i| ScoredResult {
                doc_id: format!("doc-{i}"),
                score: (i as f32).mul_add(-0.1, 1.0),
                source: ScoreSource::Hybrid,
                fast_score: None,
                quality_score: None,
                lexical_score: None,
                rerank_score: None,
                metadata: None,
            })
            .collect()
    }

    #[allow(clippy::unnecessary_wraps)]
    fn text_for_doc(doc_id: &str) -> Option<String> {
        Some(format!("Text content for {doc_id}"))
    }

    fn text_for_doc_partial(doc_id: &str) -> Option<String> {
        // Only return text for even-numbered docs
        let num: usize = doc_id.strip_prefix("doc-")?.parse().ok()?;
        if num.is_multiple_of(2) {
            Some(format!("Text for {doc_id}"))
        } else {
            None
        }
    }

    #[test]
    fn rerank_happy_path() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(10);

            rerank_step(
                &cx,
                &reranker,
                "test query",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // All top candidates should have rerank scores
            assert!(candidates.iter().all(|c| c.rerank_score.is_some()));
            assert!(candidates.iter().all(|c| c.source == ScoreSource::Reranked));
        });
    }

    #[test]
    fn rerank_too_few_candidates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(3);
            let original_scores: Vec<f32> = candidates.iter().map(|c| c.score).collect();

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // Candidates should be unchanged
            let current_scores: Vec<f32> = candidates.iter().map(|c| c.score).collect();
            assert_eq!(original_scores, current_scores);
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_empty_candidates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = Vec::new();

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            assert!(candidates.is_empty());
        });
    }

    #[test]
    fn rerank_graceful_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = FailingReranker;
            let mut candidates = make_candidates(10);
            let original_ids: Vec<String> = candidates.iter().map(|c| c.doc_id.clone()).collect();

            // Should NOT return an error
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // Candidates should be unchanged
            let current_ids: Vec<String> = candidates.iter().map(|c| c.doc_id.clone()).collect();
            assert_eq!(original_ids, current_ids);
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_score_count_mismatch() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = MismatchReranker;
            let mut candidates = make_candidates(10);

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // Should skip reranking due to mismatch
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_missing_text() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(10);

            // Only even-numbered docs have text. That's 5 docs (0,2,4,6,8).
            // Exactly meets min_candidates=5.
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc_partial,
                100,
                5,
            )
            .await
            .unwrap();

            // Even-numbered candidates should have rerank scores
            for c in &candidates {
                let num: usize = c.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                if num.is_multiple_of(2) {
                    assert!(
                        c.rerank_score.is_some(),
                        "{} should have rerank score",
                        c.doc_id
                    );
                }
            }
        });
    }

    #[test]
    fn rerank_missing_text_below_threshold() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);

            // Only 3 even-numbered docs (0,2,4) have text — below min_candidates=5
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc_partial,
                100,
                5,
            )
            .await
            .unwrap();

            // Should skip reranking
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_respects_top_k() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(20);

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 10, 5)
                .await
                .unwrap();

            // Only top 10 should have rerank scores
            for (i, c) in candidates.iter().enumerate() {
                if i < 10 {
                    assert!(c.rerank_score.is_some(), "candidate {i} should be reranked");
                } else {
                    assert!(
                        c.rerank_score.is_none(),
                        "candidate {i} should not be reranked"
                    );
                }
            }
        });
    }

    struct OutOfOrderReranker;

    impl Reranker for OutOfOrderReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                let mut scores: Vec<RerankScore> = documents
                    .iter()
                    .enumerate()
                    .map(|(rank, doc)| {
                        let doc_num: usize =
                            doc.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                        #[allow(clippy::cast_precision_loss)]
                        let score = doc_num as f32;
                        RerankScore {
                            doc_id: doc.doc_id.clone(),
                            score,
                            original_rank: rank,
                        }
                    })
                    .collect();
                scores.sort_by(|a, b| b.doc_id.cmp(&a.doc_id));
                Ok(scores)
            })
        }

        fn id(&self) -> &str {
            "out-of-order"
        }

        fn model_name(&self) -> &str {
            "out-of-order"
        }
    }

    #[test]
    fn rerank_original_rank_mapping() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = OutOfOrderReranker;
            let mut candidates = make_candidates(5);

            rerank_step(&cx, &reranker, "order", &mut candidates, text_for_doc, 5, 2)
                .await
                .unwrap();

            for cand in &candidates {
                let doc_num: usize = cand.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                #[allow(clippy::cast_precision_loss)]
                let expected_score = doc_num as f32;
                assert_eq!(cand.rerank_score, Some(expected_score));
                assert_eq!(cand.source, ScoreSource::Reranked);
            }
        });
    }

    // ─── bd-2hf5 tests begin ───

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_TOP_K_RERANK, 100);
        assert_eq!(DEFAULT_MIN_CANDIDATES, 5);
    }

    /// Reranker that returns `SearchError::Cancelled`.
    struct CancellingReranker;

    impl Reranker for CancellingReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            _documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async {
                Err(SearchError::Cancelled {
                    phase: "rerank".into(),
                    reason: "test cancellation".into(),
                })
            })
        }

        fn id(&self) -> &str {
            "cancel-reranker"
        }

        fn model_name(&self) -> &str {
            "cancel-reranker"
        }
    }

    #[test]
    fn rerank_cancellation_propagates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = CancellingReranker;
            let mut candidates = make_candidates(10);

            let err = rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .expect_err("cancellation should propagate");

            assert!(
                matches!(err, SearchError::Cancelled { .. }),
                "should be Cancelled, got: {err:?}"
            );
        });
    }

    /// Reranker that assigns equal scores to all documents.
    struct EqualScoreReranker;

    impl Reranker for EqualScoreReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| RerankScore {
                        doc_id: doc.doc_id.clone(),
                        score: 0.5,
                        original_rank: i,
                    })
                    .collect())
            })
        }

        fn id(&self) -> &str {
            "equal-score"
        }

        fn model_name(&self) -> &str {
            "equal-score"
        }
    }

    #[test]
    fn rerank_sorts_by_rerank_score_descending() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(8);

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // Verify rerank scores are in descending order.
            for pair in candidates.windows(2) {
                let a = pair[0].rerank_score.unwrap_or(f32::NEG_INFINITY);
                let b = pair[1].rerank_score.unwrap_or(f32::NEG_INFINITY);
                assert!(a >= b, "rerank scores should be descending: {a} >= {b}");
            }
        });
    }

    #[test]
    fn rerank_tie_breaks_by_doc_id() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = EqualScoreReranker;
            let mut candidates = make_candidates(6);

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // All scores are equal (0.5), so tie-breaking is by doc_id ascending.
            for pair in candidates.windows(2) {
                assert!(
                    pair[0].doc_id <= pair[1].doc_id,
                    "tie-breaking should be by doc_id: {} <= {}",
                    pair[0].doc_id,
                    pair[1].doc_id
                );
            }
        });
    }

    #[test]
    fn non_reranked_candidates_keep_original_order() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(15);
            let original_tail: Vec<String> =
                candidates[10..].iter().map(|c| c.doc_id.clone()).collect();

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                10, // only rerank top 10
                5,
            )
            .await
            .unwrap();

            // Candidates beyond top_k_rerank should be unchanged.
            let current_tail: Vec<String> =
                candidates[10..].iter().map(|c| c.doc_id.clone()).collect();
            assert_eq!(original_tail, current_tail);
            assert!(candidates[10..].iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_single_candidate_min_one() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(1);

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                1, // min_candidates = 1
            )
            .await
            .unwrap();

            assert_eq!(candidates.len(), 1);
            assert!(candidates[0].rerank_score.is_some());
            assert_eq!(candidates[0].source, ScoreSource::Reranked);
        });
    }

    #[test]
    fn rerank_all_text_missing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(10);

            // text_fn always returns None.
            rerank_step(&cx, &reranker, "test", &mut candidates, |_| None, 100, 5)
                .await
                .unwrap();

            // Should skip reranking (0 < min_candidates=5).
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    /// Reranker that returns an out-of-range `original_rank`.
    struct BadRankReranker;

    impl Reranker for BadRankReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| RerankScore {
                        doc_id: doc.doc_id.clone(),
                        score: 0.8,
                        original_rank: i + 1000, // Out of range
                    })
                    .collect())
            })
        }

        fn id(&self) -> &str {
            "bad-rank"
        }

        fn model_name(&self) -> &str {
            "bad-rank"
        }
    }

    #[test]
    fn rerank_out_of_range_original_rank_does_not_crash() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = BadRankReranker;
            let mut candidates = make_candidates(10);

            // Should not crash — just logs warnings.
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            // No rerank scores applied (all ranks were out of range).
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_min_candidates_exact_threshold() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(5);

            // Exactly at min_candidates=5 should proceed.
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            assert!(candidates.iter().all(|c| c.rerank_score.is_some()));
        });
    }

    #[test]
    fn rerank_min_candidates_one_below_threshold() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(4);

            // 4 < min_candidates=5, should skip.
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                5,
            )
            .await
            .unwrap();

            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    // ─── bd-2hf5 tests end ───

    // ─── bd-1lni tests begin ───

    #[test]
    fn stub_reranker_identity() {
        assert_eq!(StubReranker.id(), "stub-reranker");
        assert_eq!(StubReranker.model_name(), "stub-reranker");
        assert_eq!(FailingReranker.id(), "fail-reranker");
        assert_eq!(FailingReranker.model_name(), "fail-reranker");
        assert_eq!(MismatchReranker.id(), "mismatch-reranker");
        assert_eq!(MismatchReranker.model_name(), "mismatch-reranker");
        assert_eq!(CancellingReranker.id(), "cancel-reranker");
        assert_eq!(CancellingReranker.model_name(), "cancel-reranker");
        assert_eq!(EqualScoreReranker.id(), "equal-score");
        assert_eq!(EqualScoreReranker.model_name(), "equal-score");
        assert_eq!(OutOfOrderReranker.id(), "out-of-order");
        assert_eq!(OutOfOrderReranker.model_name(), "out-of-order");
        assert_eq!(BadRankReranker.id(), "bad-rank");
        assert_eq!(BadRankReranker.model_name(), "bad-rank");
    }

    #[test]
    fn rerank_min_candidates_zero_always_proceeds() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(2);

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 0)
                .await
                .unwrap();

            // min_candidates=0 means always rerank
            assert!(candidates.iter().all(|c| c.rerank_score.is_some()));
            assert!(candidates.iter().all(|c| c.source == ScoreSource::Reranked));
        });
    }

    #[test]
    fn rerank_top_k_zero_reranks_nothing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(10);

            // top_k=0 means rerank_count = min(10, 0) = 0, so no docs sent to reranker
            // but 0 < min_candidates=0 is false (0 >= 0), so we proceed past the first check
            // then included_indices.len() = 0 < min_candidates=0 is false, so we proceed
            // reranker gets empty slice, returns empty scores, 0 == 0 passes mismatch check
            // no scores to apply, sort empty range, done
            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 0, 0)
                .await
                .unwrap();

            // No candidates reranked since top_k=0
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
        });
    }

    #[test]
    fn rerank_top_k_one_reranks_single_candidate() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(10);

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 1, 1)
                .await
                .unwrap();

            // Only first candidate should be reranked
            assert!(candidates[0].rerank_score.is_some());
            assert_eq!(candidates[0].source, ScoreSource::Reranked);
            for c in &candidates[1..] {
                assert!(c.rerank_score.is_none());
                assert_eq!(c.source, ScoreSource::Hybrid);
            }
        });
    }

    #[test]
    fn rerank_preserves_metadata() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);
            // Attach metadata to each candidate
            for c in &mut candidates {
                c.metadata = Some(serde_json::Value::String(c.doc_id.clone()));
            }

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            // All candidates should still have metadata (even though order may change)
            assert!(candidates.iter().all(|c| c.metadata.is_some()));
            // Each tag should appear exactly once
            let mut tags: Vec<String> = candidates
                .iter()
                .map(|c| c.metadata.as_ref().unwrap().as_str().unwrap().to_string())
                .collect();
            tags.sort();
            assert_eq!(
                tags,
                vec!["doc-0", "doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
            );
        });
    }

    #[test]
    fn rerank_preserves_original_score_field() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);
            let original_scores: Vec<f32> = candidates.iter().map(|c| c.score).collect();

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            // Original score field should be unchanged per doc_id
            for c in &candidates {
                let num: usize = c.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                assert!(
                    (c.score - original_scores[num]).abs() < f32::EPSILON,
                    "doc-{num} score should be preserved: expected {}, got {}",
                    original_scores[num],
                    c.score
                );
            }
        });
    }

    #[test]
    fn rerank_overwrites_pre_existing_rerank_score() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);
            // Set pre-existing rerank scores
            for c in &mut candidates {
                c.rerank_score = Some(999.0);
                c.source = ScoreSource::Reranked;
            }

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            // All rerank scores should be overwritten (none should be 999.0)
            assert!(candidates
                .iter()
                .all(|c| c.rerank_score.is_some() && c.rerank_score != Some(999.0)));
        });
    }

    #[test]
    fn rerank_empty_query_succeeds() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);

            // Empty query should not cause any issues
            rerank_step(&cx, &reranker, "", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            assert!(candidates.iter().all(|c| c.rerank_score.is_some()));
        });
    }

    /// Reranker that returns correct count but with swapped doc_ids.
    struct SwappedDocIdReranker;

    impl Reranker for SwappedDocIdReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                // Return scores with correct original_rank but wrong doc_id
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(i, _doc)| RerankScore {
                        doc_id: format!("wrong-{i}"),
                        score: 0.5,
                        original_rank: i,
                    })
                    .collect())
            })
        }

        fn id(&self) -> &str {
            "swapped-docid"
        }

        fn model_name(&self) -> &str {
            "swapped-docid"
        }
    }

    #[test]
    fn rerank_doc_id_mismatch_still_applies_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = SwappedDocIdReranker;
            let mut candidates = make_candidates(6);

            // Mismatched doc_ids trigger warnings but scores are still applied
            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            // Scores should still be applied despite doc_id mismatch (just warns)
            assert!(candidates.iter().all(|c| c.rerank_score.is_some()));
        });
    }

    #[test]
    fn rerank_source_transitions_from_non_hybrid() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);
            // Set varied source types
            candidates[0].source = ScoreSource::SemanticFast;
            candidates[1].source = ScoreSource::SemanticQuality;
            candidates[2].source = ScoreSource::Lexical;

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            // All should now be Reranked regardless of prior source
            assert!(candidates.iter().all(|c| c.source == ScoreSource::Reranked));
        });
    }

    #[test]
    fn rerank_preserves_fast_and_quality_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);
            for (i, c) in candidates.iter_mut().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                {
                    c.fast_score = Some(i as f32 * 0.1);
                    c.quality_score = Some(i as f32 * 0.2);
                    c.lexical_score = Some(i as f32 * 0.3);
                }
            }

            rerank_step(&cx, &reranker, "test", &mut candidates, text_for_doc, 100, 5)
                .await
                .unwrap();

            // fast_score, quality_score, lexical_score should be preserved per doc_id
            for c in &candidates {
                let num: usize = c.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                #[allow(clippy::cast_precision_loss)]
                {
                    assert_eq!(c.fast_score, Some(num as f32 * 0.1));
                    assert_eq!(c.quality_score, Some(num as f32 * 0.2));
                    assert_eq!(c.lexical_score, Some(num as f32 * 0.3));
                }
            }
        });
    }

    // ─── bd-1lni tests end ───
}
