//! Rerank pipeline step: integrates cross-encoder reranking into the search pipeline.
//!
//! The [`rerank_step`] function reranks a set of `ScoredResult` candidates using a
//! [`Reranker`] implementation. It looks up document text via a caller-provided closure,
//! gracefully skips ordinary provider failures, and supports cancellation via `&Cx`.
//! [`rerank_step_with_combine_and_admission`] also checks request eligibility before
//! exposing text to inference and before committing the resulting order.

use std::cmp::Ordering;

use asupersync::Cx;
use tracing::instrument;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::explanation::{ExplainedSource, ScoreComponent};
use frankensearch_core::traits::{RerankDocument, RerankScore, Reranker, SyncRerank};
use frankensearch_core::types::{ScoreSource, ScoredResult};

/// Default maximum number of candidates to rerank per query.
pub const DEFAULT_TOP_K_RERANK: usize = 100;

/// Default minimum number of candidates required to trigger reranking.
pub const DEFAULT_MIN_CANDIDATES: usize = 5;

/// Default RRF constant for [`RerankCombine::RrfCombine`].
///
/// Reciprocal-rank fusion is nearly insensitive to this over the reranked window
/// (measured `k`=10 ≈ `k`=60), so a single fixed value needs no per-corpus tuning.
pub const DEFAULT_RRF_COMBINE_K: f32 = 60.0;

/// How the cross-encoder's rerank scores are combined with the pre-rerank (fused) order.
///
/// The reranked window arrives already sorted by the fused retrieval score, so each
/// candidate's position in that window *is* its pre-rerank rank.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RerankCombine {
    /// Sort the reranked window purely by descending rerank score (legacy default).
    ///
    /// This lets the cross-encoder fully reorder the window — including promoting deep
    /// candidates the retrieval stage ranked low. Simple, but a mismatched/imperfect
    /// reranker can inject false positives; measured net-harmful on some corpora
    /// (`docs/NEGATIVE_EVIDENCE.md`).
    #[default]
    PureReorder,
    /// Rank-fuse the pre-rerank order with the rerank order via reciprocal-rank fusion.
    ///
    /// Ordering key is `1/(k+pre_rank) + 1/(k+rerank_rank)`. A deep false positive must
    /// rank high by **both** retrieval and the reranker to climb, so the retrieval score
    /// vetoes the reranker's mistakes. Measured the safe, best-default integration — never
    /// catastrophic, parameter-light (see `docs/NEGATIVE_EVIDENCE.md`).
    RrfCombine {
        /// RRF constant; see [`DEFAULT_RRF_COMBINE_K`].
        k: f32,
    },
}

/// Rerank the top candidates in-place using a cross-encoder model.
///
/// This function converts `ScoredResult` candidates into `RerankDocument` pairs,
/// runs cross-encoder inference, and updates `rerank_score` / `source` on each
/// result. The candidates slice is re-sorted by descending rerank score for the
/// reranked portion.
///
/// # Graceful Failure
///
/// Except for cancellation, this function preserves usable search results:
/// - If `candidates.len() < min_candidates`: returns `Ok(())` unchanged.
/// - If `text_fn` returns `None` for a document: that document is skipped.
/// - If the reranker returns a non-cancellation error: returns `Ok(())` unchanged.
/// - If any returned score has an invalid rank, document ID, score or logit:
///   rejects the whole response and returns `Ok(())` with candidates unchanged.
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
pub async fn rerank_step(
    cx: &Cx,
    reranker: &dyn Reranker,
    query: &str,
    candidates: &mut [ScoredResult],
    text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    top_k_rerank: usize,
    min_candidates: usize,
) -> SearchResult<()> {
    rerank_step_with_combine(
        cx,
        reranker,
        query,
        candidates,
        text_fn,
        top_k_rerank,
        min_candidates,
        RerankCombine::PureReorder,
    )
    .await
}

/// Rerank the top candidates in-place, choosing how the cross-encoder scores combine
/// with the pre-rerank (fused) order via [`RerankCombine`].
///
/// [`RerankCombine::RrfCombine`] is the measured-safe integration: it rank-fuses the
/// pre-rerank order with the rerank order, so the reranker cannot promote a deep false
/// positive the retrieval stage ranked low. [`rerank_step`] is the stable convenience
/// wrapper that uses [`RerankCombine::PureReorder`] (unchanged legacy behavior).
///
/// Response admission is all-or-nothing, including existing rerank scores and
/// explanations. Every submitted document must have exactly one finite score
/// bound to its input rank and ID; a supplied raw logit must also be finite.
/// Cancellation is checked around text lookup and inference, during admission,
/// and immediately before mutation. The subsequent synchronous score/order
/// update has no await or cancellation return; it is the commit boundary, not
/// a promise to preempt an already-running model call or sort.
///
/// # Errors
///
/// Returns `SearchError::Cancelled` if the operation was cancelled via `cx`.
/// All other reranker errors are caught and logged, returning `Ok(())`.
pub async fn rerank_step_with_combine(
    cx: &Cx,
    reranker: &dyn Reranker,
    query: &str,
    candidates: &mut [ScoredResult],
    text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    top_k_rerank: usize,
    min_candidates: usize,
    combine: RerankCombine,
) -> SearchResult<()> {
    rerank_step_with_combine_and_admission(
        cx,
        reranker,
        query,
        candidates,
        text_fn,
        top_k_rerank,
        min_candidates,
        combine,
        || Ok(()),
    )
    .await
}

/// Rerank candidates with a request-specific eligibility check at callback boundaries.
///
/// This follows [`rerank_step_with_combine`], but also calls `admission` at entry,
/// before and after each text lookup, before and after inference, and immediately
/// before committing scores and order. This lets callers reject an eligibility
/// change caused by a text or provider callback before further text is exposed or
/// results are changed. A missing text or ordinary provider error cannot turn a
/// rejected request into a graceful skip.
///
/// Cancellation is checked before and after `admission`. A provider's explicit
/// cancellation also takes precedence over admission failure. All checks precede
/// the synchronous mutation boundary; a failure leaves candidates unchanged.
///
/// # Errors
///
/// Returns cancellation or an error returned by `admission`. Admission errors are
/// propagated unchanged, while ordinary provider errors retain the graceful-skip
/// behavior of [`rerank_step_with_combine`].
#[instrument(name = "rerank_step_with_combine", skip_all, fields(
    query_len = query.len(),
    num_candidates = candidates.len(),
    top_k = top_k_rerank,
))]
pub async fn rerank_step_with_combine_and_admission(
    cx: &Cx,
    reranker: &dyn Reranker,
    query: &str,
    candidates: &mut [ScoredResult],
    text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    top_k_rerank: usize,
    min_candidates: usize,
    combine: RerankCombine,
    admission: impl Fn() -> SearchResult<()> + Send + Sync,
) -> SearchResult<()> {
    let Some(prepared) = prepare_rerank(
        cx,
        candidates,
        &text_fn,
        top_k_rerank,
        min_candidates,
        &admission,
    )? else {
        return Ok(());
    };
    rerank_admission_checkpoint(cx, "reranker_before_inference", &admission)?;
    let response =
        frankensearch_core::with_cancellation(cx, "reranker_to_score_validation", async {
            reranker.rerank(cx, query, &prepared.documents).await
        })
        .await;
    finish_rerank(
        cx,
        candidates,
        &prepared,
        response,
        combine,
        &admission,
        || reranker.id().to_owned(),
    )
    .map(|_| ())
}

/// Run a synchronous cross-encoder with the same admission and combine rules
/// as [`rerank_step_with_combine_and_admission`].
///
/// Work executes on the caller's thread without an executor or async runtime.
/// The caller supplies `cx` for cancellation checks around text, inference and
/// admission callbacks; an already-running synchronous model is not preempted.
/// Ordinary provider failures, unavailable text and malformed responses leave
/// candidates unchanged and return `Ok(None)`. A complete admitted response
/// updates the window and returns the model identifier used in explanations.
///
/// # Errors
///
/// Returns cancellation or an error returned by `admission`. Cancellation has
/// priority over ordinary provider errors and admission refusal. No error is
/// returned after candidate mutation begins.
#[instrument(skip_all, fields(
    query_len = query.len(),
    num_candidates = candidates.len(),
    top_k = top_k_rerank,
))]
pub fn rerank_step_sync_with_combine_and_admission(
    cx: &Cx,
    reranker: &dyn SyncRerank,
    query: &str,
    candidates: &mut [ScoredResult],
    text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    top_k_rerank: usize,
    min_candidates: usize,
    combine: RerankCombine,
    admission: impl Fn() -> SearchResult<()> + Send + Sync,
) -> SearchResult<Option<String>> {
    let Some(prepared) = prepare_rerank(
        cx,
        candidates,
        &text_fn,
        top_k_rerank,
        min_candidates,
        &admission,
    )? else {
        return Ok(None);
    };
    rerank_admission_checkpoint(cx, "reranker_before_inference", &admission)?;
    let response = reranker.rerank_sync(query, &prepared.documents);
    finish_rerank(
        cx,
        candidates,
        &prepared,
        response,
        combine,
        &admission,
        || reranker.id().to_owned(),
    )
}

struct PreparedRerank {
    documents: Vec<RerankDocument>,
    included_indices: Option<Vec<usize>>,
    count: usize,
}

fn prepare_rerank(
    cx: &Cx,
    candidates: &[ScoredResult],
    text_fn: &impl Fn(&str) -> Option<String>,
    top_k_rerank: usize,
    min_candidates: usize,
    admission: &impl Fn() -> SearchResult<()>,
) -> SearchResult<Option<PreparedRerank>> {
    rerank_admission_checkpoint(cx, "reranker_entry", admission)?;
    if candidates.len() < min_candidates {
        tracing::debug!(
            count = candidates.len(),
            min = min_candidates,
            "skipping rerank: too few candidates"
        );
        return Ok(None);
    }
    let count = candidates.len().min(top_k_rerank);
    if count == 0 {
        return Ok(None);
    }
    let mut documents = Vec::with_capacity(count);
    let mut included_indices: Option<Vec<usize>> = None;
    for (i, candidate) in candidates.iter().take(count).enumerate() {
        rerank_admission_checkpoint(cx, "reranker_before_text", admission)?;
        let text = text_fn(&candidate.doc_id);
        rerank_admission_checkpoint(cx, "reranker_after_text", admission)?;
        if let Some(text) = text {
            documents.push(RerankDocument {
                doc_id: candidate.doc_id.to_string(),
                text,
            });
            if let Some(indices) = included_indices.as_mut() {
                indices.push(i);
            }
        } else if included_indices.is_none() {
            let mut indices = Vec::with_capacity(count);
            indices.extend(0..i);
            included_indices = Some(indices);
        }
    }
    if documents.is_empty() || documents.len() < min_candidates {
        tracing::debug!(
            with_text = documents.len(),
            min = min_candidates,
            "skipping rerank: too few candidates with available text"
        );
        return Ok(None);
    }
    Ok(Some(PreparedRerank {
        documents,
        included_indices,
        count,
    }))
}

/// Both execution modes use this complete-response admission and mutation
/// boundary, so text compaction, explanations and combination cannot drift.
#[allow(clippy::too_many_lines)]
fn finish_rerank(
    cx: &Cx,
    candidates: &mut [ScoredResult],
    prepared: &PreparedRerank,
    response: SearchResult<Vec<RerankScore>>,
    combine: RerankCombine,
    admission: &impl Fn() -> SearchResult<()>,
    model: impl FnOnce() -> String,
) -> SearchResult<Option<String>> {
    rerank_checkpoint(cx, "reranker_to_score_validation")?;
    let response = match response {
        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
        response => response,
    };
    rerank_admission_checkpoint(cx, "reranker_to_score_validation", admission)?;
    let scores = match response {
        Ok(scores) => scores,
        Err(_) => {
            // Provider errors may contain document text or arbitrary formatters.
            tracing::warn!("reranker failed — keeping original scores");
            return Ok(None);
        }
    };
    if let Err(error) = RerankScore::validate_response(cx, &prepared.documents, &scores) {
        if matches!(error, SearchError::Cancelled { .. }) {
            return Err(error);
        }
        tracing::warn!(error = %error, "reranker returned an invalid response — skipping rerank");
        return Ok(None);
    }

    // Resolve the final provider callback before the commit checkpoint. No
    // callback or cancellation return occurs after the first candidate write.
    let model = model();
    rerank_admission_checkpoint(cx, "reranker_commit", admission)?;
    clear_rerank_scores(candidates, prepared.count);
    for score in scores {
        let candidate_idx = prepared
            .included_indices
            .as_ref()
            .map_or(score.original_rank, |indices| indices[score.original_rank]);
        candidates[candidate_idx].rerank_score = Some(score.score);
        candidates[candidate_idx].source = ScoreSource::Reranked;
        if let Some(explanation) = &mut candidates[candidate_idx].explanation {
            explanation.final_score = f64::from(score.score);
            explanation.components.push(ScoreComponent {
                source: ExplainedSource::Rerank {
                    model: model.clone(),
                    logit: f64::from(score.raw_logit.unwrap_or(0.0)),
                    sigmoid: f64::from(score.score),
                },
                raw_score: f64::from(score.score),
                normalized_score: f64::from(score.score),
                rrf_contribution: 0.0,
                weight: 1.0,
            });
        }
    }
    match combine {
        RerankCombine::PureReorder => {
            candidates[..prepared.count].sort_by(compare_by_rerank_score);
        }
        RerankCombine::RrfCombine { k } => {
            apply_rrf_combine(&mut candidates[..prepared.count], k);
        }
    }
    tracing::debug!(
        reranked = prepared.documents.len(),
        model = %model,
        "rerank step complete"
    );
    Ok(Some(model))
}

fn rerank_admission_checkpoint(
    cx: &Cx,
    phase: &str,
    admission: &impl Fn() -> SearchResult<()>,
) -> SearchResult<()> {
    rerank_checkpoint(cx, phase)?;
    let result = admission();
    rerank_checkpoint(cx, phase)?;
    result
}

fn rerank_checkpoint(cx: &Cx, phase: &str) -> SearchResult<()> {
    let checkpoint = cx.checkpoint();
    if checkpoint.is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: cx.cancel_reason().map_or_else(
                || "rerank operation cancelled".to_owned(),
                |reason| reason.to_string(),
            ),
        });
    }
    Ok(())
}

fn clear_rerank_scores(candidates: &mut [ScoredResult], rerank_count: usize) {
    // Drop any stale rerank scores so this run only reflects fresh model output.
    for candidate in candidates.iter_mut().take(rerank_count) {
        candidate.rerank_score = None;
    }
}

fn finite_rerank_sort_score(candidate: &ScoredResult) -> f32 {
    candidate
        .rerank_score
        .filter(|score| score.is_finite())
        .unwrap_or(f32::NEG_INFINITY)
}

fn compare_by_rerank_score(a: &ScoredResult, b: &ScoredResult) -> Ordering {
    let score_a = finite_rerank_sort_score(a);
    let score_b = finite_rerank_sort_score(b);
    score_b
        .total_cmp(&score_a)
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

#[derive(Clone, Copy)]
struct RrfOrder {
    position: usize,
    fused_key: f64,
}

/// Reorder `window` (the reranked candidates, arriving in pre-rerank/fused order — so
/// index `i` is the pre-rerank rank) by reciprocal-rank fusion of the pre-rerank rank
/// and the rerank-score rank: `1/(k+pre_rank) + 1/(k+rerank_rank)`, descending. Ties
/// break on `doc_id` for determinism. Candidates without a finite rerank score sort to
/// the worst rerank rank (they keep their pre-rerank contribution but earn none from the
/// reranker), matching the graceful-skip semantics of the pure-reorder path.
#[allow(clippy::cast_precision_loss)]
fn apply_rrf_combine(window: &mut [ScoredResult], k: f32) {
    let n = window.len();
    if n < 2 {
        return;
    }
    let kf = f64::from(k.max(1.0));

    // One order vector carries both stages: first sort it by rerank score to
    // assign rerank ranks, then overwrite each entry with its fused key and sort
    // by that final key. This preserves the previous rank-fusion semantics while
    // avoiding separate `by_rerank`, `rerank_rank`, `key`, and `perm` vectors.
    let mut order: Vec<RrfOrder> = (0..n)
        .map(|position| RrfOrder {
            position,
            fused_key: 0.0,
        })
        .collect();
    order.sort_by(|a, b| compare_by_rerank_score(&window[a.position], &window[b.position]));
    for (rerank_rank, entry) in order.iter_mut().enumerate() {
        entry.fused_key = 1.0 / (kf + entry.position as f64) + 1.0 / (kf + rerank_rank as f64);
    }
    order.sort_by(|a, b| {
        b.fused_key
            .total_cmp(&a.fused_key)
            .then_with(|| window[a.position].doc_id.cmp(&window[b.position].doc_id))
    });

    // Gather the permuted window into a snapshot (one clone per slot, unavoidable
    // because the gather can read positions we'd otherwise overwrite), then MOVE the
    // snapshot back into `window`. The prior `clone_from_slice(&reordered)` cloned
    // every element a second time; moving halves the `ScoredResult` clones (2N → N).
    let reordered: Vec<ScoredResult> = order
        .into_iter()
        .map(|entry| window[entry.position].clone())
        .collect();
    for (slot, value) in window.iter_mut().zip(reordered) {
        *slot = value;
    }
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
                            raw_logit: None,
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
                    raw_logit: None,
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

    /// Reranker that over-promotes the DEEPEST candidate (the one retrieval ranked last)
    /// to the highest score — a "false positive" the cross-encoder loves. Used to show
    /// that `PureReorder` lets it win the top slot while `RrfCombine` vetoes it.
    struct FalsePositiveReranker;

    impl Reranker for FalsePositiveReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                let n = documents.len();
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| {
                        // Deepest doc gets the top score; the rest stay near retrieval order.
                        #[allow(clippy::cast_precision_loss)]
                        let score = if i + 1 == n {
                            1.0
                        } else {
                            (i as f32).mul_add(-0.01, 0.9)
                        };
                        RerankScore {
                            doc_id: doc.doc_id.clone(),
                            score,
                            original_rank: i,
                            raw_logit: None,
                        }
                    })
                    .collect())
            })
        }

        fn id(&self) -> &str {
            "false-positive-reranker"
        }

        fn model_name(&self) -> &str {
            "false-positive-reranker"
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn make_candidates(n: usize) -> Vec<ScoredResult> {
        (0..n)
            .map(|i| ScoredResult {
                doc_id: format!("doc-{i}").into(),
                score: (i as f32).mul_add(-0.1, 1.0),
                source: ScoreSource::Hybrid,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: None,
                rerank_score: None,
                explanation: None,
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

    #[allow(clippy::cast_precision_loss)]
    fn apply_rrf_combine_reference(window: &mut [ScoredResult], k: f32) {
        let n = window.len();
        if n < 2 {
            return;
        }
        let kf = f64::from(k.max(1.0));

        let mut by_rerank: Vec<usize> = (0..n).collect();
        by_rerank.sort_by(|&a, &b| compare_by_rerank_score(&window[a], &window[b]));
        let mut rerank_rank = vec![0usize; n];
        for (rank, &pos) in by_rerank.iter().enumerate() {
            rerank_rank[pos] = rank;
        }

        let key: Vec<f64> = (0..n)
            .map(|i| 1.0 / (kf + i as f64) + 1.0 / (kf + rerank_rank[i] as f64))
            .collect();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&a, &b| {
            key[b]
                .total_cmp(&key[a])
                .then_with(|| window[a].doc_id.cmp(&window[b].doc_id))
        });

        let reordered: Vec<ScoredResult> = perm.into_iter().map(|i| window[i].clone()).collect();
        window.clone_from_slice(&reordered);
    }

    #[test]
    fn rrf_combine_order_vector_matches_reference_permutation() {
        let mut reference = make_candidates(16);
        let mut candidate = reference.clone();
        for (i, item) in reference.iter_mut().enumerate() {
            let score = ((i * 13 + 7) % 17) as f32 * 0.1;
            item.rerank_score = if i % 7 == 0 { None } else { Some(score) };
        }
        candidate.clone_from_slice(&reference);

        apply_rrf_combine_reference(&mut reference, DEFAULT_RRF_COMBINE_K);
        apply_rrf_combine(&mut candidate, DEFAULT_RRF_COMBINE_K);

        let reference_ids = reference
            .iter()
            .map(|item| item.doc_id.as_str())
            .collect::<Vec<_>>();
        let candidate_ids = candidate
            .iter()
            .map(|item| item.doc_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(candidate_ids, reference_ids);
        for (candidate, reference) in candidate.iter().zip(reference.iter()) {
            assert_eq!(candidate.rerank_score, reference.rerank_score);
        }
    }

    #[test]
    fn rrf_combine_vetoes_deep_false_positive() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // Pure-reorder (default): the reranker's false positive — the deepest
            // candidate, which retrieval ranked last — wins the top slot.
            let mut pure = make_candidates(5);
            rerank_step(
                &cx,
                &FalsePositiveReranker,
                "q",
                &mut pure,
                text_for_doc,
                100,
                2,
            )
            .await
            .unwrap();
            assert_eq!(
                pure[0].doc_id.to_string(),
                "doc-4",
                "pure-reorder lets the reranker promote the deep false positive to #1"
            );

            // RRF-combine: retrieval's best doc is retained at #1 and the deep false
            // positive is vetoed out of the top slot — it ranked last by retrieval, so
            // fusing the two ranks caps how far the reranker alone can lift it.
            let mut rrf = make_candidates(5);
            rerank_step_with_combine(
                &cx,
                &FalsePositiveReranker,
                "q",
                &mut rrf,
                text_for_doc,
                100,
                2,
                RerankCombine::RrfCombine {
                    k: DEFAULT_RRF_COMBINE_K,
                },
            )
            .await
            .unwrap();
            assert_eq!(
                rrf[0].doc_id.to_string(),
                "doc-0",
                "RRF-combine keeps retrieval's best on top"
            );
            assert_ne!(
                rrf[0].doc_id.to_string(),
                "doc-4",
                "RRF-combine vetoes the deep false positive"
            );
            // The reranked window still carries fresh rerank scores in both modes.
            assert!(rrf.iter().all(|c| c.rerank_score.is_some()));
        });
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
            let original_ids: Vec<String> =
                candidates.iter().map(|c| c.doc_id.to_string()).collect();

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
            let current_ids: Vec<String> =
                candidates.iter().map(|c| c.doc_id.to_string()).collect();
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
    fn rerank_missing_text_clears_stale_scores_for_non_reranked_candidates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = StubReranker;
            let mut candidates = make_candidates(6);
            for c in &mut candidates {
                c.rerank_score = Some(999.0);
                c.source = ScoreSource::Reranked;
            }

            // With partial text, only even docs are reranked (0,2,4).
            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc_partial,
                6,
                3,
            )
            .await
            .unwrap();

            for c in &candidates {
                let num: usize = c.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                if num.is_multiple_of(2) {
                    assert!(
                        c.rerank_score.is_some() && c.rerank_score != Some(999.0),
                        "reranked doc should have fresh score: {}",
                        c.doc_id
                    );
                } else {
                    assert_eq!(
                        c.rerank_score, None,
                        "non-reranked doc should not keep stale score: {}",
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
                            raw_logit: None,
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

    /// Reranker that cancels the real invocation context but still returns scores.
    struct CancellingReranker;

    impl Reranker for CancellingReranker {
        fn rerank<'a>(
            &'a self,
            cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            let scores = documents
                .iter()
                .enumerate()
                .map(|(original_rank, document)| RerankScore {
                    doc_id: document.doc_id.clone(),
                    score: if original_rank == 0 { 0.0 } else { 1.0 },
                    original_rank,
                    raw_logit: None,
                })
                .collect();
            Box::pin(async move {
                cx.cancel_with(
                    asupersync::CancelKind::User,
                    Some("cancel after reranker scores"),
                );
                Ok(scores)
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
            let original_ids: Vec<_> = candidates
                .iter()
                .map(|candidate| candidate.doc_id.clone())
                .collect();
            let original_scores: Vec<_> = candidates
                .iter()
                .map(|candidate| candidate.score.to_bits())
                .collect();

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

            assert!(matches!(
                err,
                SearchError::Cancelled { phase, reason }
                    if phase == "reranker_to_score_validation"
                        && reason == "user: cancel after reranker scores"
            ));
            assert_eq!(
                candidates
                    .iter()
                    .map(|candidate| candidate.doc_id.clone())
                    .collect::<Vec<_>>(),
                original_ids,
                "cancellation before score validation must preserve candidate order"
            );
            assert_eq!(
                candidates
                    .iter()
                    .map(|candidate| candidate.score.to_bits())
                    .collect::<Vec<_>>(),
                original_scores,
                "cancellation before score validation must preserve score bits"
            );
            assert!(
                candidates.iter().all(|candidate| {
                    candidate.rerank_score.is_none() && candidate.source == ScoreSource::Hybrid
                }),
                "cancellation before score validation must not mutate rerank state"
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
                        raw_logit: None,
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
            let original_tail: Vec<String> = candidates[10..]
                .iter()
                .map(|c| c.doc_id.to_string())
                .collect();

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
            let current_tail: Vec<String> = candidates[10..]
                .iter()
                .map(|c| c.doc_id.to_string())
                .collect();
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
                        raw_logit: None,
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

            rerank_step(
                &cx,
                &reranker,
                "test",
                &mut candidates,
                text_for_doc,
                100,
                0,
            )
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
                c.metadata = Some(std::sync::Arc::new(serde_json::Value::String(
                    c.doc_id.to_string(),
                )));
            }

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

            // All rerank scores should be overwritten (none should be 999.0)
            assert!(
                candidates
                    .iter()
                    .all(|c| c.rerank_score.is_some() && c.rerank_score != Some(999.0))
            );
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

    /// Reranker that returns correct count but with swapped `doc_ids`.
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
                        raw_logit: None,
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
    fn rerank_doc_id_mismatch_skips_score_application() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let reranker = SwappedDocIdReranker;
            let mut candidates = make_candidates(6);

            // Mismatched doc_ids should be skipped to prevent cross-document
            // score contamination.
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

            // No scores should be applied when doc_ids don't match
            assert!(candidates.iter().all(|c| c.rerank_score.is_none()));
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

    mod response_admission {
        use super::*;
        use std::future::Future;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
        use std::task::{Context, Poll, Waker};

        #[derive(Clone, Copy, Debug)]
        enum Reply {
            Valid,
            Error,
            Short,
            Long,
            Duplicate,
            InvalidRank,
            WrongId,
            InvalidScore(f32),
            InvalidLogit(f32),
            CancelSuccess,
            CancelError,
            ProviderCancelled,
            Pending,
        }

        struct Probe {
            reply: Reply,
            calls: AtomicUsize,
            drops: AtomicUsize,
            cancel_on_id: Option<Cx>,
            revoke_on_response: Option<Arc<AtomicBool>>,
            revoke_on_id: Option<Arc<AtomicBool>>,
        }

        impl Probe {
            fn new(reply: Reply) -> Self {
                Self {
                    reply,
                    calls: AtomicUsize::new(0),
                    drops: AtomicUsize::new(0),
                    cancel_on_id: None,
                    revoke_on_response: None,
                    revoke_on_id: None,
                }
            }
        }

        struct CallGuard<'a>(&'a AtomicUsize);

        impl Drop for CallGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }

        impl Reranker for Probe {
            #[allow(clippy::cast_precision_loss)]
            fn rerank<'a>(
                &'a self,
                cx: &'a Cx,
                _query: &'a str,
                documents: &'a [RerankDocument],
            ) -> SearchFuture<'a, Vec<RerankScore>> {
                self.calls.fetch_add(1, AtomicOrdering::Relaxed);
                Box::pin(async move {
                    let _guard = CallGuard(&self.drops);
                    if matches!(self.reply, Reply::Pending) {
                        return std::future::pending().await;
                    }
                    if let Some(eligible) = &self.revoke_on_response {
                        eligible.store(false, AtomicOrdering::Relaxed);
                    }
                    if matches!(self.reply, Reply::ProviderCancelled) {
                        return Err(SearchError::Cancelled {
                            phase: "fixture.provider".to_owned(),
                            reason: "provider stopped".to_owned(),
                        });
                    }
                    if matches!(self.reply, Reply::Error) {
                        return Err(SearchError::RerankFailed {
                            model: "probe".to_owned(),
                            source: "ordinary provider error".into(),
                        });
                    }
                    let mut scores: Vec<_> = documents
                        .iter()
                        .enumerate()
                        .map(|(original_rank, doc)| RerankScore {
                            doc_id: doc.doc_id.clone(),
                            score: (original_rank + 1) as f32 / (documents.len() + 1) as f32,
                            original_rank,
                            raw_logit: Some(original_rank as f32 - 2.0),
                        })
                        .collect();
                    scores.reverse();
                    if scores.is_empty() {
                        return Ok(scores);
                    }
                    let last = scores.len() - 1;
                    match self.reply {
                        Reply::Short => {
                            let _ = scores.pop();
                        }
                        Reply::Long => scores.push(RerankScore {
                            doc_id: scores[0].doc_id.clone(),
                            score: scores[0].score,
                            original_rank: scores[0].original_rank,
                            raw_logit: scores[0].raw_logit,
                        }),
                        Reply::Duplicate => {
                            scores[last].doc_id = scores[0].doc_id.clone();
                            scores[last].original_rank = scores[0].original_rank;
                        }
                        Reply::InvalidRank => scores[last].original_rank = usize::MAX,
                        Reply::WrongId => scores[last].doc_id = "foreign-document".to_owned(),
                        Reply::InvalidScore(value) => scores[last].score = value,
                        Reply::InvalidLogit(value) => scores[last].raw_logit = Some(value),
                        Reply::CancelSuccess | Reply::CancelError => {
                            cx.cancel_with(
                                asupersync::CancelKind::User,
                                Some("response cancelled"),
                            );
                            if matches!(self.reply, Reply::CancelError) {
                                return Err(SearchError::RerankFailed {
                                    model: "probe".to_owned(),
                                    source: "ordinary provider error after cancellation".into(),
                                });
                            }
                        }
                        Reply::Valid | Reply::Error | Reply::ProviderCancelled | Reply::Pending => {
                        }
                    }
                    Ok(scores)
                })
            }

            fn id(&self) -> &str {
                if let Some(cx) = &self.cancel_on_id {
                    cx.cancel_with(asupersync::CancelKind::User, Some("model label cancelled"));
                }
                if let Some(eligible) = &self.revoke_on_id {
                    eligible.store(false, AtomicOrdering::Relaxed);
                }
                "admission-probe"
            }

            fn model_name(&self) -> &str {
                "admission-probe"
            }
        }

        struct SyncProbe {
            probe: Probe,
            cx: Cx,
        }

        impl SyncProbe {
            fn new(reply: Reply, cx: &Cx) -> Self {
                Self {
                    probe: Probe::new(reply),
                    cx: cx.clone(),
                }
            }
        }

        impl frankensearch_core::traits::SyncRerank for SyncProbe {
            #[allow(clippy::cast_precision_loss)]
            fn rerank_sync(
                &self,
                _query: &str,
                documents: &[RerankDocument],
            ) -> SearchResult<Vec<RerankScore>> {
                self.probe.calls.fetch_add(1, AtomicOrdering::Relaxed);
                let _guard = CallGuard(&self.probe.drops);
                assert!(!matches!(self.probe.reply, Reply::Pending));
                if let Some(eligible) = &self.probe.revoke_on_response {
                    eligible.store(false, AtomicOrdering::Relaxed);
                }
                if matches!(self.probe.reply, Reply::ProviderCancelled) {
                    return Err(SearchError::Cancelled {
                        phase: "fixture.provider".to_owned(),
                        reason: "provider stopped".to_owned(),
                    });
                }
                if matches!(self.probe.reply, Reply::Error) {
                    return Err(SearchError::RerankFailed {
                        model: "probe".to_owned(),
                        source: "ordinary provider error".into(),
                    });
                }
                let mut scores: Vec<_> = documents
                    .iter()
                    .enumerate()
                    .map(|(original_rank, doc)| RerankScore {
                        doc_id: doc.doc_id.clone(),
                        score: (original_rank + 1) as f32 / (documents.len() + 1) as f32,
                        original_rank,
                        raw_logit: Some(original_rank as f32 - 2.0),
                    })
                    .collect();
                scores.reverse();
                if scores.is_empty() {
                    return Ok(scores);
                }
                let last = scores.len() - 1;
                match self.probe.reply {
                    Reply::Short => {
                        let _ = scores.pop();
                    }
                    Reply::Long => scores.push(RerankScore {
                        doc_id: scores[0].doc_id.clone(),
                        score: scores[0].score,
                        original_rank: scores[0].original_rank,
                        raw_logit: scores[0].raw_logit,
                    }),
                    Reply::Duplicate => {
                        scores[last].doc_id = scores[0].doc_id.clone();
                        scores[last].original_rank = scores[0].original_rank;
                    }
                    Reply::InvalidRank => scores[last].original_rank = usize::MAX,
                    Reply::WrongId => scores[last].doc_id = "foreign-document".to_owned(),
                    Reply::InvalidScore(value) => scores[last].score = value,
                    Reply::InvalidLogit(value) => scores[last].raw_logit = Some(value),
                    Reply::CancelSuccess | Reply::CancelError => {
                        self.cx.cancel_with(
                            asupersync::CancelKind::User,
                            Some("response cancelled"),
                        );
                        if matches!(self.probe.reply, Reply::CancelError) {
                            return Err(SearchError::RerankFailed {
                                model: "probe".to_owned(),
                                source: "ordinary provider error after cancellation".into(),
                            });
                        }
                    }
                    Reply::Valid | Reply::Error | Reply::ProviderCancelled | Reply::Pending => {}
                }
                Ok(scores)
            }

            fn id(&self) -> &str {
                Reranker::id(&self.probe)
            }

            fn model_name(&self) -> &str {
                Reranker::model_name(&self.probe)
            }
        }

        fn candidates() -> Vec<ScoredResult> {
            let mut candidates = make_candidates(7);
            for (i, candidate) in candidates.iter_mut().enumerate() {
                candidate.index = Some(u32::try_from(i + 20).unwrap());
                candidate.fast_score = Some(-0.0);
                candidate.quality_score = Some(0.7);
                candidate.lexical_score = Some(4.0);
                candidate.rerank_score = Some(0.125);
                candidate.source = ScoreSource::Reranked;
                candidate.metadata = Some(Arc::new(serde_json::json!({"row": i})));
                candidate.explanation = Some(Box::new(frankensearch_core::HitExplanation {
                    final_score: 0.125,
                    components: vec![ScoreComponent {
                        source: ExplainedSource::Rerank {
                            model: "previous-model".to_owned(),
                            logit: -1.0,
                            sigmoid: 0.125,
                        },
                        raw_score: 0.125,
                        normalized_score: 0.125,
                        rrf_contribution: 0.0,
                        weight: 1.0,
                    }],
                    phase: frankensearch_core::ExplanationPhase::Refined,
                    rank_movement: None,
                }));
            }
            candidates
        }

        fn assert_unchanged(actual: &[ScoredResult], expected: &[ScoredResult]) {
            assert_eq!(
                serde_json::to_vec(actual).unwrap(),
                serde_json::to_vec(expected).unwrap()
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.score.to_bits(), expected.score.to_bits());
                assert_eq!(
                    actual.fast_score.map(f32::to_bits),
                    expected.fast_score.map(f32::to_bits)
                );
                assert!(Arc::ptr_eq(
                    actual.metadata.as_ref().unwrap(),
                    expected.metadata.as_ref().unwrap()
                ));
            }
        }

        fn admit_while_eligible(eligible: &AtomicBool) -> SearchResult<()> {
            if eligible.load(AtomicOrdering::Relaxed) {
                Ok(())
            } else {
                Err(SearchError::InvalidConfig {
                    field: "search.filter".to_owned(),
                    value: "eligibility_changed".to_owned(),
                    reason: "request eligibility was revoked".to_owned(),
                })
            }
        }

        fn assert_admission_rejected(error: SearchError) {
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, value, reason }
                    if field == "search.filter"
                        && value == "eligibility_changed"
                        && reason == "request eligibility was revoked"
            ));
        }

        const STRATEGIES: [RerankCombine; 2] = [
            RerankCombine::PureReorder,
            RerankCombine::RrfCombine {
                k: DEFAULT_RRF_COMBINE_K,
            },
        ];

        #[test]
        fn sync_pipeline_matches_async_for_full_and_compacted_responses() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for combine in STRATEGIES {
                    for partial in [false, true] {
                        for reply in [
                            Reply::Valid,
                            Reply::Error,
                            Reply::Short,
                            Reply::Long,
                            Reply::Duplicate,
                            Reply::InvalidRank,
                            Reply::WrongId,
                            Reply::InvalidScore(f32::NAN),
                            Reply::InvalidScore(f32::INFINITY),
                            Reply::InvalidScore(f32::NEG_INFINITY),
                            Reply::InvalidLogit(f32::NAN),
                            Reply::InvalidLogit(f32::INFINITY),
                            Reply::InvalidLogit(f32::NEG_INFINITY),
                        ] {
                            let before = candidates();
                            let mut asynchronous = before.clone();
                            let mut synchronous = before.clone();
                            let async_provider = Probe::new(reply);
                            let sync_provider = SyncProbe::new(reply, &cx);
                            let text = |id: &str| {
                                if partial {
                                    text_for_doc_partial(id)
                                } else {
                                    text_for_doc(id)
                                }
                            };
                            rerank_step_with_combine(
                                &cx,
                                &async_provider,
                                "query",
                                &mut asynchronous,
                                &text,
                                6,
                                2,
                                combine,
                            )
                            .await
                            .unwrap();
                            let applied = rerank_step_sync_with_combine_and_admission(
                                &cx,
                                &sync_provider,
                                "query",
                                &mut synchronous,
                                &text,
                                6,
                                2,
                                combine,
                                || Ok(()),
                            )
                            .unwrap();
                            assert_unchanged(&synchronous, &asynchronous);
                            assert_unchanged(&synchronous[6..], &before[6..]);
                            if matches!(reply, Reply::Valid) {
                                assert_eq!(applied.as_deref(), Some("admission-probe"));
                            } else {
                                assert!(applied.is_none());
                                assert_unchanged(&synchronous, &before);
                            }
                            assert_eq!(async_provider.calls.load(AtomicOrdering::Relaxed), 1);
                            assert_eq!(sync_provider.probe.calls.load(AtomicOrdering::Relaxed), 1);
                            assert_eq!(sync_provider.probe.drops.load(AtomicOrdering::Relaxed), 1);
                        }
                    }
                }
            });
        }

        #[test]
        fn sync_text_revocation_stops_before_later_lookups_or_inference() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for combine in STRATEGIES {
                    for missing in [false, true] {
                        for revoke_at in [0, 5] {
                            let provider = SyncProbe::new(Reply::Valid, &cx);
                            let mut values = candidates();
                            let before = values.clone();
                            let lookups = AtomicUsize::new(0);
                            let eligible = AtomicBool::new(true);
                            let error = rerank_step_sync_with_combine_and_admission(
                                &cx,
                                &provider,
                                "query",
                                &mut values,
                                |id| {
                                    if lookups.fetch_add(1, AtomicOrdering::Relaxed) == revoke_at {
                                        eligible.store(false, AtomicOrdering::Relaxed);
                                    }
                                    if missing { None } else { text_for_doc(id) }
                                },
                                6,
                                1,
                                combine,
                                || admit_while_eligible(&eligible),
                            )
                            .unwrap_err();
                            assert_admission_rejected(error);
                            assert_unchanged(&values, &before);
                            assert_eq!(lookups.load(AtomicOrdering::Relaxed), revoke_at + 1);
                            assert_eq!(provider.probe.calls.load(AtomicOrdering::Relaxed), 0);
                        }
                    }
                }
            });
        }

        #[test]
        fn sync_provider_revocation_and_cancellation_preserve_candidate_state() {
            for combine in STRATEGIES {
                for reply in [
                    Reply::Valid,
                    Reply::Error,
                    Reply::WrongId,
                    Reply::CancelSuccess,
                    Reply::CancelError,
                    Reply::ProviderCancelled,
                ] {
                    asupersync::test_utils::run_test_with_cx(|cx| async move {
                        let eligible = Arc::new(AtomicBool::new(true));
                        let mut provider = SyncProbe::new(reply, &cx);
                        provider.probe.revoke_on_response = Some(Arc::clone(&eligible));
                        let mut values = candidates();
                        let before = values.clone();
                        let error = rerank_step_sync_with_combine_and_admission(
                            &cx,
                            &provider,
                            "query",
                            &mut values,
                            text_for_doc,
                            6,
                            1,
                            combine,
                            || admit_while_eligible(&eligible),
                        )
                        .unwrap_err();
                        match reply {
                            Reply::ProviderCancelled => assert!(matches!(
                                error,
                                SearchError::Cancelled { phase, reason }
                                    if phase == "fixture.provider" && reason == "provider stopped"
                            )),
                            Reply::CancelSuccess | Reply::CancelError => assert!(matches!(
                                error,
                                SearchError::Cancelled { phase, .. }
                                    if phase == "reranker_to_score_validation"
                            )),
                            _ => assert_admission_rejected(error),
                        }
                        assert!(!eligible.load(AtomicOrdering::Relaxed));
                        assert_unchanged(&values, &before);
                        assert_eq!(provider.probe.calls.load(AtomicOrdering::Relaxed), 1);
                        assert_eq!(provider.probe.drops.load(AtomicOrdering::Relaxed), 1);
                    });
                }
            }
        }

        #[test]
        fn sync_admission_cancellation_precedes_errors_and_noop_shortcuts() {
            for reject in [false, true] {
                for (count, top_k, minimum) in [(0, 10, 0), (7, 0, 0), (7, 7, 100), (7, 7, 1)] {
                    asupersync::test_utils::run_test_with_cx(|cx| async move {
                        let provider = SyncProbe::new(Reply::Valid, &cx);
                        let mut values = candidates();
                        values.truncate(count);
                        let before = values.clone();
                        let lookups = AtomicUsize::new(0);
                        let eligible = AtomicBool::new(!reject);
                        let error = rerank_step_sync_with_combine_and_admission(
                            &cx,
                            &provider,
                            "query",
                            &mut values,
                            |id| {
                                lookups.fetch_add(1, AtomicOrdering::Relaxed);
                                text_for_doc(id)
                            },
                            top_k,
                            minimum,
                            RerankCombine::PureReorder,
                            || {
                                cx.cancel_with(
                                    asupersync::CancelKind::User,
                                    Some("admission cancelled"),
                                );
                                admit_while_eligible(&eligible)
                            },
                        )
                        .unwrap_err();
                        assert!(matches!(
                            error,
                            SearchError::Cancelled { phase, .. } if phase == "reranker_entry"
                        ));
                        assert_unchanged(&values, &before);
                        assert_eq!(lookups.load(AtomicOrdering::Relaxed), 0);
                        assert_eq!(provider.probe.calls.load(AtomicOrdering::Relaxed), 0);
                    });
                }
            }
        }

        #[test]
        fn sync_model_label_revocation_is_observed_before_commit() {
            for combine in STRATEGIES {
                for cancel in [false, true] {
                    asupersync::test_utils::run_test_with_cx(|cx| async move {
                        let eligible = Arc::new(AtomicBool::new(true));
                        let mut provider = SyncProbe::new(Reply::Valid, &cx);
                        provider.probe.revoke_on_id = Some(Arc::clone(&eligible));
                        if cancel {
                            provider.probe.cancel_on_id = Some(cx.clone());
                        }
                        let mut values = candidates();
                        let before = values.clone();
                        let error = rerank_step_sync_with_combine_and_admission(
                            &cx,
                            &provider,
                            "query",
                            &mut values,
                            text_for_doc,
                            6,
                            1,
                            combine,
                            || admit_while_eligible(&eligible),
                        )
                        .unwrap_err();
                        if cancel {
                            assert!(matches!(
                                error,
                                SearchError::Cancelled { phase, .. } if phase == "reranker_commit"
                            ));
                        } else {
                            assert_admission_rejected(error);
                        }
                        assert!(!eligible.load(AtomicOrdering::Relaxed));
                        assert_unchanged(&values, &before);
                        assert_eq!(provider.probe.calls.load(AtomicOrdering::Relaxed), 1);
                    });
                }
            }
        }

        #[test]
        fn rejected_requests_skip_text_and_inference_even_for_noops() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for (count, top_k, minimum) in [(0, 10, 0), (7, 0, 0), (7, 7, 100), (7, 7, 1)] {
                    let provider = Probe::new(Reply::Valid);
                    let mut values = candidates();
                    values.truncate(count);
                    let before = values.clone();
                    let lookups = AtomicUsize::new(0);
                    let eligible = AtomicBool::new(false);
                    let error = rerank_step_with_combine_and_admission(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        |id| {
                            lookups.fetch_add(1, AtomicOrdering::Relaxed);
                            text_for_doc(id)
                        },
                        top_k,
                        minimum,
                        RerankCombine::PureReorder,
                        || admit_while_eligible(&eligible),
                    )
                    .await
                    .unwrap_err();
                    assert_admission_rejected(error);
                    assert_unchanged(&values, &before);
                    assert_eq!(lookups.load(AtomicOrdering::Relaxed), 0);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 0);
                }
            });
        }

        #[test]
        fn text_lookup_revocation_stops_before_later_lookups_or_inference() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for combine in STRATEGIES {
                    for missing in [false, true] {
                        for revoke_at in [0, 5] {
                            let provider = Probe::new(Reply::Valid);
                            let mut values = candidates();
                            let before = values.clone();
                            let lookups = AtomicUsize::new(0);
                            let eligible = AtomicBool::new(true);
                            let error = rerank_step_with_combine_and_admission(
                                &cx,
                                &provider,
                                "query",
                                &mut values,
                                |id| {
                                    if lookups.fetch_add(1, AtomicOrdering::Relaxed) == revoke_at {
                                        eligible.store(false, AtomicOrdering::Relaxed);
                                    }
                                    if missing { None } else { text_for_doc(id) }
                                },
                                6,
                                1,
                                combine,
                                || admit_while_eligible(&eligible),
                            )
                            .await
                            .unwrap_err();
                            assert_admission_rejected(error);
                            assert_unchanged(&values, &before);
                            assert_eq!(lookups.load(AtomicOrdering::Relaxed), revoke_at + 1);
                            assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 0);
                        }
                    }
                }
            });
        }

        #[test]
        fn provider_revocation_rejects_success_and_graceful_skip_but_preserves_cancellation() {
            for combine in STRATEGIES {
                for reply in [
                    Reply::Valid,
                    Reply::Error,
                    Reply::WrongId,
                    Reply::CancelSuccess,
                    Reply::CancelError,
                    Reply::ProviderCancelled,
                ] {
                    asupersync::test_utils::run_test_with_cx(|cx| async move {
                        let eligible = Arc::new(AtomicBool::new(true));
                        let mut provider = Probe::new(reply);
                        provider.revoke_on_response = Some(Arc::clone(&eligible));
                        let mut values = candidates();
                        let before = values.clone();
                        let error = rerank_step_with_combine_and_admission(
                            &cx,
                            &provider,
                            "query",
                            &mut values,
                            text_for_doc,
                            6,
                            1,
                            combine,
                            || admit_while_eligible(&eligible),
                        )
                        .await
                        .unwrap_err();
                        match reply {
                            Reply::ProviderCancelled => assert!(matches!(
                                error,
                                SearchError::Cancelled { phase, reason }
                                    if phase == "fixture.provider" && reason == "provider stopped"
                            )),
                            Reply::CancelSuccess | Reply::CancelError => assert!(matches!(
                                error,
                                SearchError::Cancelled { phase, .. }
                                    if phase == "reranker_to_score_validation"
                            )),
                            _ => assert_admission_rejected(error),
                        }
                        assert!(!eligible.load(AtomicOrdering::Relaxed));
                        assert_unchanged(&values, &before);
                        assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 1);
                        assert_eq!(provider.drops.load(AtomicOrdering::Relaxed), 1);
                    });
                }
            }
        }

        #[test]
        fn cancellation_from_admission_wins_over_its_return_value() {
            for reject in [false, true] {
                asupersync::test_utils::run_test_with_cx(|cx| async move {
                    let provider = Probe::new(Reply::Valid);
                    let mut values = candidates();
                    let before = values.clone();
                    let lookups = AtomicUsize::new(0);
                    let eligible = AtomicBool::new(!reject);
                    let error = rerank_step_with_combine_and_admission(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        |id| {
                            lookups.fetch_add(1, AtomicOrdering::Relaxed);
                            text_for_doc(id)
                        },
                        6,
                        1,
                        RerankCombine::PureReorder,
                        || {
                            cx.cancel_with(
                                asupersync::CancelKind::User,
                                Some("admission cancelled"),
                            );
                            admit_while_eligible(&eligible)
                        },
                    )
                    .await
                    .unwrap_err();
                    assert!(matches!(
                        error,
                        SearchError::Cancelled { phase, .. } if phase == "reranker_entry"
                    ));
                    assert_unchanged(&values, &before);
                    assert_eq!(lookups.load(AtomicOrdering::Relaxed), 0);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 0);
                });
            }
        }

        #[test]
        fn eligibility_revoked_by_model_label_is_observed_before_commit() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for combine in STRATEGIES {
                    let eligible = Arc::new(AtomicBool::new(true));
                    let mut provider = Probe::new(Reply::Valid);
                    provider.revoke_on_id = Some(Arc::clone(&eligible));
                    let mut values = candidates();
                    let before = values.clone();
                    let error = rerank_step_with_combine_and_admission(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        text_for_doc,
                        6,
                        1,
                        combine,
                        || admit_while_eligible(&eligible),
                    )
                    .await
                    .unwrap_err();
                    assert_admission_rejected(error);
                    assert_unchanged(&values, &before);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 1);
                }
            });
        }

        #[test]
        fn malformed_responses_leave_full_and_compacted_windows_unchanged() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for combine in STRATEGIES {
                    for partial in [false, true] {
                        for reply in [
                            Reply::Short,
                            Reply::Long,
                            Reply::Duplicate,
                            Reply::InvalidRank,
                            Reply::WrongId,
                            Reply::InvalidScore(f32::NAN),
                            Reply::InvalidScore(f32::INFINITY),
                            Reply::InvalidScore(f32::NEG_INFINITY),
                            Reply::InvalidLogit(f32::NAN),
                            Reply::InvalidLogit(f32::INFINITY),
                            Reply::InvalidLogit(f32::NEG_INFINITY),
                        ] {
                            let provider = Probe::new(reply);
                            let mut values = candidates();
                            let before = values.clone();
                            rerank_step_with_combine(
                                &cx,
                                &provider,
                                "query",
                                &mut values,
                                |id| {
                                    if partial {
                                        text_for_doc_partial(id)
                                    } else {
                                        text_for_doc(id)
                                    }
                                },
                                6,
                                2,
                                combine,
                            )
                            .await
                            .unwrap();
                            assert_unchanged(&values, &before);
                            assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 1);
                        }
                    }
                }
            });
        }

        #[test]
        fn valid_permuted_scores_keep_compacted_mapping_explanations_and_tail() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for combine in STRATEGIES {
                    let provider = Probe::new(Reply::Valid);
                    let mut values = candidates();
                    let before = values.clone();
                    rerank_step_with_combine(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        text_for_doc_partial,
                        6,
                        2,
                        combine,
                    )
                    .await
                    .unwrap();
                    let mut expected_order = before.clone();
                    for (i, value) in expected_order.iter_mut().take(6).enumerate() {
                        #[allow(clippy::cast_precision_loss)]
                        {
                            value.rerank_score =
                                i.is_multiple_of(2).then_some((i / 2 + 1) as f32 / 4.0);
                        }
                    }
                    match combine {
                        RerankCombine::PureReorder => {
                            expected_order[..6].sort_by(compare_by_rerank_score);
                        }
                        RerankCombine::RrfCombine { k } => {
                            apply_rrf_combine_reference(&mut expected_order[..6], k);
                        }
                    }
                    assert_eq!(
                        values
                            .iter()
                            .map(|value| value.doc_id.as_str())
                            .collect::<Vec<_>>(),
                        expected_order
                            .iter()
                            .map(|value| value.doc_id.as_str())
                            .collect::<Vec<_>>(),
                    );
                    for value in &values[..6] {
                        let i: usize = value.doc_id.strip_prefix("doc-").unwrap().parse().unwrap();
                        let original = &before[i];
                        assert_eq!(value.index, original.index);
                        assert_eq!(value.score.to_bits(), original.score.to_bits());
                        assert_eq!(
                            value.fast_score.map(f32::to_bits),
                            original.fast_score.map(f32::to_bits)
                        );
                        assert_eq!(value.quality_score, original.quality_score);
                        assert_eq!(value.lexical_score, original.lexical_score);
                        assert!(Arc::ptr_eq(
                            value.metadata.as_ref().unwrap(),
                            original.metadata.as_ref().unwrap()
                        ));
                        let explanation = value.explanation.as_ref().unwrap();
                        if i.is_multiple_of(2) {
                            #[allow(clippy::cast_precision_loss)]
                            let expected = (i / 2 + 1) as f32 / 4.0;
                            assert_eq!(value.rerank_score, Some(expected));
                            assert_eq!(explanation.final_score, f64::from(expected));
                            assert_eq!(explanation.components.len(), 2);
                            #[allow(clippy::cast_precision_loss)]
                            let expected_logit = (i / 2) as f64 - 2.0;
                            #[allow(clippy::float_cmp)] // the probe returns exact logits
                            let exact_rerank_source = matches!(
                                &explanation.components[1].source,
                                ExplainedSource::Rerank { model, logit, sigmoid }
                                    if model == "admission-probe" && *logit == expected_logit
                                        && *sigmoid == f64::from(expected)
                            );
                            assert!(exact_rerank_source);
                        } else {
                            assert!(value.rerank_score.is_none());
                            assert_eq!(explanation.components.len(), 1);
                        }
                    }
                    assert_unchanged(&values[6..], &before[6..]);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 1);
                }
            });
        }

        #[test]
        fn pre_cancelled_requests_skip_text_and_inference_even_for_noops() {
            for (count, top_k, minimum) in [(0, 10, 0), (7, 0, 0), (7, 7, 100), (7, 7, 1)] {
                asupersync::test_utils::run_test_with_cx(|cx| async move {
                    let provider = Probe::new(Reply::Valid);
                    let mut values = candidates();
                    values.truncate(count);
                    let before = values.clone();
                    let lookups = AtomicUsize::new(0);
                    cx.cancel_with(asupersync::CancelKind::User, Some("cancel before rerank"));
                    let result = rerank_step(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        |id| {
                            lookups.fetch_add(1, AtomicOrdering::Relaxed);
                            text_for_doc(id)
                        },
                        top_k,
                        minimum,
                    )
                    .await;
                    assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                    assert_unchanged(&values, &before);
                    assert_eq!(lookups.load(AtomicOrdering::Relaxed), 0);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 0);
                });
            }
        }

        #[test]
        fn text_lookup_cancellation_stops_before_later_lookups_or_inference() {
            for missing in [false, true] {
                asupersync::test_utils::run_test_with_cx(|cx| async move {
                    let provider = Probe::new(Reply::Valid);
                    let mut values = candidates();
                    let before = values.clone();
                    let lookups = AtomicUsize::new(0);
                    let result = rerank_step(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        |id| {
                            lookups.fetch_add(1, AtomicOrdering::Relaxed);
                            cx.cancel_with(
                                asupersync::CancelKind::User,
                                Some("text lookup cancelled"),
                            );
                            if missing { None } else { text_for_doc(id) }
                        },
                        6,
                        0,
                    )
                    .await;
                    assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                    assert_unchanged(&values, &before);
                    assert_eq!(lookups.load(AtomicOrdering::Relaxed), 1);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 0);
                });
            }
        }

        #[test]
        fn cancellation_wins_over_success_and_ordinary_provider_failure() {
            for reply in [
                Reply::CancelSuccess,
                Reply::CancelError,
                Reply::ProviderCancelled,
            ] {
                asupersync::test_utils::run_test_with_cx(|cx| async move {
                    let provider = Probe::new(reply);
                    let mut values = candidates();
                    let before = values.clone();
                    let error =
                        rerank_step(&cx, &provider, "query", &mut values, text_for_doc, 6, 1)
                            .await
                            .unwrap_err();
                    let expected = if matches!(reply, Reply::ProviderCancelled) {
                        "fixture.provider"
                    } else {
                        "reranker_to_score_validation"
                    };
                    assert!(
                        matches!(error, SearchError::Cancelled { ref phase, .. } if phase == expected)
                    );
                    assert_unchanged(&values, &before);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 1);
                    assert_eq!(provider.drops.load(AtomicOrdering::Relaxed), 1);
                });
            }
        }

        #[test]
        fn no_work_does_not_infer_or_clear_existing_rerank_state() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for (count, top_k, missing) in [(0, 6, false), (7, 0, false), (7, 6, true)] {
                    let provider = Probe::new(Reply::Valid);
                    let mut values = candidates();
                    values.truncate(count);
                    let before = values.clone();
                    rerank_step(
                        &cx,
                        &provider,
                        "query",
                        &mut values,
                        |id| {
                            if missing { None } else { text_for_doc(id) }
                        },
                        top_k,
                        0,
                    )
                    .await
                    .unwrap();
                    assert_unchanged(&values, &before);
                    assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 0);
                }
            });
        }

        #[test]
        fn abandoned_inference_preserves_candidates_and_allows_explicit_retry() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let pending = Probe::new(Reply::Pending);
                let mut values = candidates();
                let before = values.clone();
                let mut future = Box::pin(rerank_step(
                    &cx,
                    &pending,
                    "query",
                    &mut values,
                    text_for_doc,
                    6,
                    1,
                ));
                let waker = Waker::noop();
                assert!(matches!(
                    future.as_mut().poll(&mut Context::from_waker(waker)),
                    Poll::Pending
                ));
                drop(future);
                assert_unchanged(&values, &before);
                assert_eq!(pending.calls.load(AtomicOrdering::Relaxed), 1);
                assert_eq!(pending.drops.load(AtomicOrdering::Relaxed), 1);
                assert!(!cx.is_cancel_requested());
                let ready = Probe::new(Reply::Valid);
                rerank_step(&cx, &ready, "query", &mut values, text_for_doc, 6, 1)
                    .await
                    .unwrap();
                assert_eq!(values[0].doc_id, "doc-5");
                assert_unchanged(&values[6..], &before[6..]);
                assert_eq!(ready.calls.load(AtomicOrdering::Relaxed), 1);
            });
        }

        #[test]
        fn cancellation_from_model_label_is_observed_before_commit() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let mut provider = Probe::new(Reply::Valid);
                provider.cancel_on_id = Some(cx.clone());
                let mut values = candidates();
                let before = values.clone();
                let error = rerank_step(&cx, &provider, "query", &mut values, text_for_doc, 6, 1)
                    .await
                    .unwrap_err();
                assert!(
                    matches!(error, SearchError::Cancelled { ref phase, .. } if phase == "reranker_commit")
                );
                assert_unchanged(&values, &before);
                assert_eq!(provider.calls.load(AtomicOrdering::Relaxed), 1);
            });
        }
    }
}
