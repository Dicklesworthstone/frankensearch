//! Frozen source membership across every vector partition and global Quill.
//!
//! This uses the existing sharded progressive engine and the ordinary scope's
//! immutable lexical widening adapter. No second fusion or query engine is used.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::{ScopeSource, ScopedLexical, ScopedText};
use crate::native_ann::builder::sharded::NativeBuiltShardedHybridIndex;
use crate::native_ann::{
    NativeShardedProgressiveSearch, NativeShardedResult, NativeShardedSearchPhase, checkpoint,
    invalid,
};
use crate::{Cx, IndexableDocument, Reranker, ScoreSource, SearchResult};

/// A reusable source scope tied to one complete sharded hybrid generation.
///
/// Membership is evaluated once on the retained original source documents.
/// Filtering precedes per-shard top-k merging, tier normalization, lexical
/// fusion, hydration and cross-encoder input. Very selective lexical scopes can
/// examine the whole matching population; global BM25 statistics are unchanged.
/// Native graphs retain their approximate retrieval semantics.
///
/// Borrow a live snapshot and create the scope on `snapshot.index()`. Never
/// transfer its document IDs or result rows to a newer snapshot: repartitioning
/// can move every row, and metadata updates can change eligible membership.
/// An old scope is intentionally not a revocable permission on the latest head.
/// This does not hide an index from code that already owns that index.
pub struct NativeScopedShardedHybridIndex<'a> {
    index: &'a NativeBuiltShardedHybridIndex,
    lexical: ScopedLexical<'a>,
    text: Box<ScopedText<'a>>,
}

impl NativeBuiltShardedHybridIndex {
    /// Freeze a fallible source predicate across the entire retained inventory.
    ///
    /// Predicates can inspect IDs, original content, titles and metadata. They
    /// are not retained or re-evaluated during widening, queries or later phases.
    /// No source, model or vector file is reopened and no inference starts.
    ///
    /// # Errors
    /// Predicate errors and cancellation refuse the whole scope. No partial or
    /// unrestricted fallback is returned, including an empty candidate inventory.
    pub fn scope(
        &self,
        cx: &Cx,
        mut accept: impl FnMut(&IndexableDocument) -> SearchResult<bool>,
    ) -> SearchResult<NativeScopedShardedHybridIndex<'_>> {
        checkpoint(cx, "native_ann.sharded_scope.start")?;
        let mut allowed = BTreeSet::new();
        for document in self.vectors().documents() {
            checkpoint(cx, "native_ann.sharded_scope.document")?;
            let accepted = accept(document);
            checkpoint(cx, "native_ann.sharded_scope.document_complete")?;
            if accepted? {
                allowed.insert(document.id.clone());
            }
        }
        let allowed = Arc::new(allowed);
        let text_ids = Arc::clone(&allowed);
        let text = Box::new(move |id: &str| {
            if text_ids.contains(id) {
                self.vectors()
                    .document(id)
                    .map(|document| document.content.clone())
            } else {
                None
            }
        });
        checkpoint(cx, "native_ann.sharded_scope.complete")?;
        Ok(NativeScopedShardedHybridIndex {
            index: self,
            lexical: ScopedLexical {
                index: ScopeSource::Sharded(self),
                allowed,
            },
            text,
        })
    }
}

impl NativeScopedShardedHybridIndex<'_> {
    /// Eligible source documents, not the number of hits matching a query.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lexical.allowed.len()
    }

    /// Whether no document is eligible in this retained scope.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lexical.allowed.is_empty()
    }

    /// Resolve an eligible document from this scope's original source cohort.
    #[must_use]
    pub fn document(&self, id: &str) -> Option<&IndexableDocument> {
        if self.lexical.allowed.contains(id) {
            self.index.vectors().document(id)
        } else {
            None
        }
    }

    /// Prepare scoped Initial and independent quality refinement lazily.
    ///
    /// The existing shard engine embeds once per requested tier and filters
    /// inside each partition's ANN widening or exact scan. Counts describe the
    /// eligible pools. Empty scopes skip inference but retain normal configured
    /// identity and topology admission, including on zero-result requests.
    ///
    /// # Errors
    /// Propagates native identity, topology, budget and cancellation errors.
    pub fn progressive<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeShardedProgressiveSearch<'a>> {
        let vectors = self.index.vectors();
        let first = &vectors.partitions()[0];
        let quality = match (vectors.quality(), first.quality()) {
            (Some(shards), Some(tier)) => Some((shards, tier.embedder())),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "sharded_scope.quality",
                    "inconsistent",
                    "quality inventory and producing model must travel together",
                ));
            }
        };
        vectors
            .fast()
            .search_hybrid_progressive(cx, first.fast().embedder(), quality, &self.lexical, text, k)?
            .with_allowed_documents(&self.lexical.allowed)
    }

    /// Rerank only eligible candidates using their original retained source text.
    ///
    /// No excluded document reaches the scorer. Ranking moves each whole result
    /// with its separate fast/quality shard coordinates; `result.index` stays
    /// absent. The window may exceed the displayed page. No model runs until the
    /// final phase is requested, and scorer failure retains the preceding page.
    ///
    /// # Errors
    /// Combines scoped progressive admission and existing rerank-window errors.
    pub fn progressive_with_reranker<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        reranker: &'a dyn Reranker,
        window: usize,
    ) -> SearchResult<NativeShardedProgressiveSearch<'a>> {
        self.progressive(cx, text, k)?
            .with_reranker(reranker, self.text.as_ref(), window)
    }

    /// Scoped fast-plus-keyword retrieval without invoking the quality provider.
    ///
    /// # Errors
    /// Propagates required retrieval, widening, hydration and cancellation errors.
    pub async fn search(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        collect(self.primary(cx, text, k, false)?).await
    }

    /// Collect the scoped progressive query, requiring every configured phase.
    ///
    /// # Errors
    /// A failed required quality phase is an error, not Initial-only success.
    pub async fn search_refined(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        collect(self.progressive(cx, text, k)?).await
    }

    /// Independent scoped quality-plus-keyword retrieval without fast inference.
    ///
    /// # Errors
    /// Refuses absent quality before the empty-scope/zero-k fast paths and
    /// propagates ordinary identity, retrieval, hydration and cancellation errors.
    pub async fn search_quality(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        let mut results = collect(self.primary(cx, text, k, true)?).await?;
        for hit in &mut results {
            checkpoint(cx, "native_ann.sharded_scope.quality_result")?;
            hit.quality_row = hit.fast_row.take();
            hit.result.quality_score = hit.result.fast_score.take();
            if hit.result.source == ScoreSource::SemanticFast {
                hit.result.source = ScoreSource::SemanticQuality;
            }
        }
        Ok(results)
    }

    fn primary<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        quality: bool,
    ) -> SearchResult<NativeShardedProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.sharded_scope.primary")?;
        let vectors = self.index.vectors();
        let first = &vectors.partitions()[0];
        let (shards, tier) = if quality {
            match (vectors.quality(), first.quality()) {
                (Some(shards), Some(tier)) => (shards, tier),
                _ => {
                    return Err(invalid(
                        "sharded_scope.quality",
                        "absent",
                        "quality-primary search requires a complete quality tier",
                    ));
                }
            }
        } else {
            (vectors.fast(), first.fast())
        };
        shards
            .search_hybrid_progressive(cx, tier.embedder(), None, &self.lexical, text, k)?
            .with_allowed_documents(&self.lexical.allowed)
    }
}

async fn collect(
    mut query: NativeShardedProgressiveSearch<'_>,
) -> SearchResult<Vec<NativeShardedResult>> {
    let mut page = Vec::new();
    while let Some(phase) = query.next_phase().await? {
        page = match phase {
            NativeShardedSearchPhase::Initial { results, .. }
            | NativeShardedSearchPhase::Refined { results, .. }
            | NativeShardedSearchPhase::Reranked { results, .. } => results,
            NativeShardedSearchPhase::RefinementFailed { error, .. }
            | NativeShardedSearchPhase::RerankFailed { error, .. } => return Err(error),
        };
    }
    Ok(page)
}
