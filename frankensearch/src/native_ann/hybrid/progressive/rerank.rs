//! Transactional cross-encoder ordering shared by native progressive paths.
//!
//! Models implement the ordinary core Reranker trait. This module owns only
//! candidate/text admission and the final permutation, not inference or a runtime.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use frankensearch_core::traits::{RerankDocument, Reranker};

use super::{checkpoint, invalid};
use crate::{Cx, ScoreSource, ScoredResult, SearchError, SearchResult};

#[derive(Clone, Copy)]
pub(in crate::native_ann::hybrid) struct RerankRequest<'a> {
    pub reranker: &'a dyn Reranker,
    pub text: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
    pub limit: usize,
}

/// No caller-visible candidate is changed until the entire response is admitted.
/// Locations and metadata travel with their original result, never with a rank
/// supplied by the provider.
pub(in crate::native_ann::hybrid) struct RerankPlan {
    order: Vec<usize>,
    scores: Vec<Option<f32>>,
    pub evaluated: usize,
}

impl RerankRequest<'_> {
    pub async fn plan<'a>(
        &self,
        cx: &Cx,
        query: &str,
        candidates: impl IntoIterator<Item = &'a ScoredResult>,
    ) -> SearchResult<Option<RerankPlan>> {
        checkpoint(cx, "native_ann.rerank_start")?;
        let candidates: Vec<_> = candidates.into_iter().collect();
        if candidates.is_empty() {
            return Ok(None);
        }
        if !self.reranker.is_available() {
            return Err(SearchError::RerankerUnavailable {
                model: self.reranker.id().to_owned(),
            });
        }
        let mut documents = Vec::with_capacity(self.limit.min(candidates.len()));
        let mut positions = Vec::with_capacity(documents.capacity());
        let mut seen = BTreeSet::new();
        for (position, candidate) in candidates.iter().take(self.limit).enumerate() {
            checkpoint(cx, "native_ann.rerank_text")?;
            if !seen.insert(candidate.doc_id.as_str()) {
                return Err(invalid(
                    "rerank.candidates",
                    "duplicate-id",
                    "reranking requires unique candidate document identities",
                ));
            }
            let text = (self.text)(candidate.doc_id.as_str());
            checkpoint(cx, "native_ann.rerank_text_complete")?;
            if let Some(text) = text {
                documents.push(RerankDocument {
                    doc_id: candidate.doc_id.to_string(),
                    text,
                });
                positions.push(position);
            }
        }
        // Missing text is not a successful rerank. The consumer already has
        // the preceding phase; no model call or invented final phase is needed.
        if documents.is_empty() {
            return Ok(None);
        }
        let response = self.reranker.rerank(cx, query, &documents).await;
        checkpoint(cx, "native_ann.rerank_response")?;
        let response = response?;
        if response.len() != documents.len() {
            return Err(invalid(
                "rerank.response",
                "cardinality",
                "the reranker must return exactly one score for each submitted document",
            ));
        }
        let mut seen_ranks = vec![false; documents.len()];
        let mut scores = vec![None; candidates.len()];
        for score in response {
            checkpoint(cx, "native_ann.rerank_admission")?;
            let Some(document) = documents.get(score.original_rank) else {
                return Err(invalid(
                    "rerank.response",
                    "rank-range",
                    "rerank rank is outside its input batch",
                ));
            };
            if seen_ranks[score.original_rank] || score.doc_id != document.doc_id {
                return Err(invalid(
                    "rerank.response",
                    "identity-permutation",
                    "returned ranks must be a permutation of the exact submitted document identities",
                ));
            }
            if !score.score.is_finite() || score.raw_logit.is_some_and(|value| !value.is_finite()) {
                return Err(invalid(
                    "rerank.response",
                    "non-finite",
                    "rerank scores and supplied logits must be finite",
                ));
            }
            seen_ranks[score.original_rank] = true;
            scores[positions[score.original_rank]] = Some(score.score);
        }
        let mut order: Vec<_> = (0..candidates.len()).collect();
        order.sort_unstable_by(|&left, &right| match (scores[left], scores[right]) {
            (Some(a), Some(b)) => b
                .total_cmp(&a)
                .then_with(|| candidates[left].doc_id.cmp(&candidates[right].doc_id)),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left.cmp(&right),
        });
        checkpoint(cx, "native_ann.rerank_plan_complete")?;
        Ok(Some(RerankPlan {
            order,
            scores,
            evaluated: documents.len(),
        }))
    }
}

impl RerankPlan {
    /// Move the complete candidate envelope, including any per-tier shard rows.
    /// Retrieval scores/explanations remain retrieval evidence; only the separate
    /// rerank score and source change. No fabricated raw logit is synthesized.
    pub fn apply<T>(
        self,
        candidates: Vec<T>,
        result: impl Fn(&mut T) -> &mut ScoredResult,
    ) -> Vec<T> {
        debug_assert_eq!(self.scores.len(), candidates.len());
        let mut candidates: Vec<_> = candidates.into_iter().map(Some).collect();
        let mut ordered = Vec::with_capacity(candidates.len());
        for position in self.order {
            // order is a private, complete permutation constructed above.
            let mut candidate = candidates[position]
                .take()
                .expect("admitted rerank permutation");
            if let Some(score) = self.scores[position] {
                let result = result(&mut candidate);
                result.rerank_score = Some(score);
                result.source = ScoreSource::Reranked;
            }
            ordered.push(candidate);
        }
        ordered
    }
}
