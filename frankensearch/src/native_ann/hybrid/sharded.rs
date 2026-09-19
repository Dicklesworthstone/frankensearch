//! Hybrid and lazy progressive retrieval over independently sharded tiers.

use std::collections::BTreeMap;

use frankensearch_core::LexicalCandidateBatch;
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::traits::Reranker;
use frankensearch_fusion::{RrfConfig, blend_two_tier, rrf_fuse_for_vector_lane};

use super::progressive::rerank::RerankRequest;
use super::refinement::{admit_pair, checked_lexical};
use super::{NativePhaseCandidates, checkpoint, hydrate_winners, invalid, join_sources};
use crate::native_ann::{NativeShardHit, NativeShardRow, NativeShardSet};
use crate::{
    Cx, Embedder, LexicalRead, ScoreSource, ScoredResult, SearchError, SearchResult, VectorHit,
};

/// A fused winner with separately scoped fast and quality physical locations.
///
/// `result.index` is always `None`: one u32 cannot identify a row across both
/// shards and embedding tiers. Use the two explicit locations below instead.
#[derive(Debug, Clone)]
pub struct NativeShardedResult {
    /// Ranked result, raw semantic scores, source and pinned lexical metadata.
    pub result: ScoredResult,
    /// Location in the admitted fast shard set, when it contributed this document.
    pub fast_row: Option<NativeShardRow>,
    /// Location in the admitted quality shard set, when it contributed this document.
    pub quality_row: Option<NativeShardRow>,
}

/// One requested phase of sharded native retrieval.
#[derive(Debug)]
pub enum NativeShardedSearchPhase {
    /// Fast and lexical results; no quality inference has started.
    Initial {
        /// Hydrated winners in fused order.
        results: Vec<NativeShardedResult>,
        /// Actual global candidate pool sizes, not shard visits or coverage.
        candidates: NativePhaseCandidates,
    },
    /// Results after independent quality-shard retrieval and candidate union.
    Refined {
        /// Hydrated winners with per-tier shard coordinates.
        results: Vec<NativeShardedResult>,
        /// Retained fast/lexical and freshly retrieved quality pool sizes.
        candidates: NativePhaseCandidates,
    },
    /// A non-cancellation failure; initial results remain valid and unchanged.
    RefinementFailed {
        /// Unmodified results from the completed initial phase.
        initial_results: Vec<NativeShardedResult>,
        /// Why quality retrieval, fusion or hydration failed.
        error: SearchError,
    },
    /// Cross-encoder winners retaining their complete per-tier shard locations.
    Reranked {
        /// At most the requested `k` results, with separate rerank scores.
        results: Vec<NativeShardedResult>,
        /// Original retrieval pool sizes, not cross-encoder work counts.
        candidates: NativePhaseCandidates,
        /// Candidate documents with available text evaluated by the model.
        evaluated: usize,
    },
    /// Non-cancellation final-stage failure; the previous page remains valid.
    RerankFailed {
        /// Unmodified previous page, including both tiers' row coordinates.
        previous_results: Vec<NativeShardedResult>,
        /// Availability, inference or whole-response admission failure.
        error: SearchError,
    },
}

impl NativeShardSet {
    /// Retrieve one sharded fast tier and lexical candidates, then hydrate winners.
    ///
    /// The query is embedded once, not once per shard. The lexical scoring
    /// snapshot stays pinned through hydration. Independent reader selection
    /// remains the caller's responsibility; this does not establish durable
    /// composite lexical/vector authority.
    ///
    /// # Errors
    ///
    /// Propagates admission, provider, shard, lexical, hydration and cancellation
    /// failures. No failing partition is omitted from a successful result.
    pub async fn search_hybrid_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        let is_hash = admit_tiers(cx, self, embedder, None)?;
        let budget = candidate_budget(k)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let (fast, batch) = join_sources(
            cx,
            self.search_text(cx, embedder, text, budget, None),
            checked_lexical(cx, lexical, text, budget),
        )
        .await?;
        finish(cx, lexical, &batch, &fast, &[], k, is_hash, false).await
    }

    /// Search this shard set as the primary quality tier, without a fast tier.
    ///
    /// # Errors
    ///
    /// Has the same error surface as [`Self::search_hybrid_text`].
    pub async fn search_hybrid_quality_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        let mut results = self
            .search_hybrid_text(cx, embedder, lexical, text, k)
            .await?;
        for hit in &mut results {
            checkpoint(cx, "native_ann.shards.quality_result")?;
            hit.quality_row = hit.fast_row.take();
            hit.result.quality_score = hit.result.fast_score.take();
            if hit.result.source == ScoreSource::SemanticFast {
                hit.result.source = ScoreSource::SemanticQuality;
            }
        }
        Ok(results)
    }

    /// Independently search both shard sets and fuse the union with lexical ranks.
    ///
    /// Fast and quality may have different shard counts, partition boundaries,
    /// dimensions and partial membership. Their complete artifact generation
    /// and canonical document-ID contracts must agree. A quality winner may be
    /// absent from every fast candidate shard. Each tier embeds only once.
    /// The existing 0.7/0.3 normalized union blend precedes lexical RRF.
    ///
    /// # Errors
    ///
    /// Rejects tier generation/identity disagreement before inference. All
    /// provider, shard, lexical, hydration and cancellation errors propagate.
    pub async fn search_hybrid_refined_text(
        &self,
        cx: &Cx,
        fast_embedder: &dyn Embedder,
        quality: (&Self, &dyn Embedder),
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        let is_hash = admit_tiers(cx, self, fast_embedder, Some(quality))?;
        let budget = candidate_budget(k)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let (fast, (quality, batch)) = join_sources(
            cx,
            self.search_text(cx, fast_embedder, text, budget, None),
            join_sources(
                cx,
                quality.0.search_text(cx, quality.1, text, budget, None),
                checked_lexical(cx, lexical, text, budget),
            ),
        )
        .await?;
        finish(cx, lexical, &batch, &fast, &quality, k, is_hash, true).await
    }

    /// Prepare lazy fast-then-quality retrieval without starting provider work.
    ///
    /// The first phase retains global fast candidates and the lexical scoring
    /// snapshot. Only a second requested phase embeds and searches quality.
    /// Dropping a polled pending phase finishes the stream and drops its work;
    /// dropping an unpolled future does not consume a phase. No tasks are spawned.
    /// Attach [`NativeShardedProgressiveSearch::with_reranker`] before the first
    /// phase to add lazy cross-encoder ordering of a larger candidate window.
    ///
    /// # Errors
    ///
    /// Refuses tier admission or candidate-budget overflow, including on noops.
    pub fn search_hybrid_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        fast_embedder: &'a dyn Embedder,
        quality: Option<(&'a Self, &'a dyn Embedder)>,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeShardedProgressiveSearch<'a>> {
        admit_tiers(cx, self, fast_embedder, quality)?;
        Ok(NativeShardedProgressiveSearch {
            fast: self,
            cx,
            fast_embedder,
            quality,
            lexical,
            text,
            k,
            budget: candidate_budget(k)?,
            reranker: None,
            state: State::Initial,
        })
    }
}

struct Pending {
    fast: Vec<NativeShardHit>,
    batch: LexicalCandidateBatch,
    initial: Vec<NativeShardedResult>,
}
struct PendingRerank {
    results: Vec<NativeShardedResult>,
    candidates: NativePhaseCandidates,
    // The scoring pin outlives text resolution and the model's pending future.
    _batch: LexicalCandidateBatch,
}
enum State {
    Initial,
    Refine(Pending),
    Rerank(PendingRerank),
    Done,
}

/// Caller-driven progressive search holding the original shard and lexical pins.
///
/// Cancellation is a terminal error, not successful degradation. Other quality
/// failures carry unchanged initial results. Graph work is synchronous on the
/// polling executor; this type provides no hidden timeout or worker runtime.
pub struct NativeShardedProgressiveSearch<'a> {
    fast: &'a NativeShardSet,
    cx: &'a Cx,
    fast_embedder: &'a dyn Embedder,
    quality: Option<(&'a NativeShardSet, &'a dyn Embedder)>,
    lexical: &'a dyn LexicalRead,
    text: &'a str,
    k: usize,
    budget: usize,
    reranker: Option<RerankRequest<'a>>,
    state: State,
}

impl<'a> NativeShardedProgressiveSearch<'a> {
    /// Add one lazy cross-encoder stage over the globally fused candidate window.
    ///
    /// The model runs once across the tier, not once per partition. `top_k`
    /// bounds its input window independently of the displayed page; retrieval
    /// requests three times `max(k, top_k)` candidates per tier. This may change
    /// initial fusion relative to a smaller unreranked request. Final ordering
    /// moves the complete result envelope, keeping metadata and both row
    /// coordinates attached to their document. `result.index` stays `None`.
    ///
    /// Only requesting the final phase calls `text_fn` or the reranker. Capture
    /// the consumer's immutable source snapshot in `text_fn`: the lexical pin
    /// is retained, but an arbitrary closure's text provenance cannot be proven
    /// by this API. Missing text skips that candidate; no text returns no final
    /// phase. Failed quality refinement still permits reranking the valid
    /// initial pool. Cancellation or dropping a pending stage never retries it.
    ///
    /// The separate `rerank_score` carries the new ordering signal; original
    /// retrieval scores and explanations stay unchanged. No source is reopened,
    /// model substituted, or runtime created. This is not composite authority.
    ///
    /// # Errors
    ///
    /// Refuses zero/overflowing windows, cancellation, or configuration after
    /// any phase started. Model availability is checked when reranking is requested.
    pub fn with_reranker(
        mut self,
        reranker: &'a dyn Reranker,
        text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
        top_k: usize,
    ) -> SearchResult<Self> {
        checkpoint(self.cx, "native_ann.shards.rerank_configuration")?;
        if !matches!(self.state, State::Initial) || top_k == 0 {
            return Err(invalid(
                "shards.rerank.configuration",
                "started-or-empty-window",
                "attach a positive rerank window before requesting any phase",
            ));
        }
        self.budget = candidate_budget(self.k.max(top_k))?;
        self.reranker = Some(RerankRequest {
            reranker,
            text: text_fn,
            limit: top_k,
        });
        Ok(self)
    }

    /// Whether the final phase completed, failed, or was abandoned while pending.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// Execute one requested phase without restarting completed or abandoned work.
    ///
    /// # Errors
    ///
    /// Initial failures and all cancellations are errors. Other refinement
    /// failures become an explicit phase carrying the unchanged initial results.
    /// Optional reranking follows quality success or non-cancellation failure;
    /// a rerank failure carries the unchanged last displayed page.
    pub async fn next_phase(&mut self) -> SearchResult<Option<NativeShardedSearchPhase>> {
        match std::mem::replace(&mut self.state, State::Done) {
            State::Done => Ok(None),
            State::Initial => self.initial().await.map(Some),
            State::Refine(pending) => match self.refine(&pending).await {
                Ok((results, candidates)) => {
                    let results = self.finish_retrieval(results, candidates, pending.batch);
                    Ok(Some(NativeShardedSearchPhase::Refined {
                        results,
                        candidates,
                    }))
                }
                Err(error) => {
                    checkpoint(self.cx, "native_ann.shards.refinement_failure")?;
                    if matches!(error, SearchError::Cancelled { .. }) {
                        return Err(error);
                    }
                    let candidates = NativePhaseCandidates {
                        fast: pending.fast.len(),
                        quality: 0,
                        lexical: pending.batch.results().len(),
                    };
                    let initial_results =
                        self.finish_retrieval(pending.initial, candidates, pending.batch);
                    Ok(Some(NativeShardedSearchPhase::RefinementFailed {
                        initial_results,
                        error,
                    }))
                }
            },
            State::Rerank(pending) => self.rerank(pending).await,
        }
    }

    async fn initial(&mut self) -> SearchResult<NativeShardedSearchPhase> {
        let is_hash = admit_tiers(self.cx, self.fast, self.fast_embedder, self.quality)?;
        if self.k == 0 {
            return Ok(NativeShardedSearchPhase::Initial {
                results: Vec::new(),
                candidates: NativePhaseCandidates::default(),
            });
        }
        let (fast, batch) = join_sources(
            self.cx,
            self.fast
                .search_text(self.cx, self.fast_embedder, self.text, self.budget, None),
            checked_lexical(self.cx, self.lexical, self.text, self.budget),
        )
        .await?;
        let candidates = NativePhaseCandidates {
            fast: fast.len(),
            quality: 0,
            lexical: batch.results().len(),
        };
        let mut results = finish(
            self.cx,
            self.lexical,
            &batch,
            &fast,
            &[],
            self.result_window(),
            is_hash,
            false,
        )
        .await?;
        checkpoint(self.cx, "native_ann.shards.initial_complete")?;
        if self.quality.is_some() {
            self.state = State::Refine(Pending {
                fast,
                batch,
                initial: results.clone(),
            });
            results.truncate(self.k);
        } else {
            results = self.finish_retrieval(results, candidates, batch);
        }
        Ok(NativeShardedSearchPhase::Initial {
            results,
            candidates,
        })
    }

    async fn refine(
        &self,
        pending: &Pending,
    ) -> SearchResult<(Vec<NativeShardedResult>, NativePhaseCandidates)> {
        let is_hash = admit_tiers(self.cx, self.fast, self.fast_embedder, self.quality)?;
        let (quality, embedder) = self.quality.ok_or_else(|| {
            invalid(
                "shards.quality",
                "missing",
                "refinement requires its admitted quality tier",
            )
        })?;
        let quality = quality
            .search_text(self.cx, embedder, self.text, self.budget, None)
            .await?;
        let candidates = NativePhaseCandidates {
            fast: pending.fast.len(),
            quality: quality.len(),
            lexical: pending.batch.results().len(),
        };
        let results = finish(
            self.cx,
            self.lexical,
            &pending.batch,
            &pending.fast,
            &quality,
            self.result_window(),
            is_hash,
            true,
        )
        .await?;
        checkpoint(self.cx, "native_ann.shards.refined_complete")?;
        Ok((results, candidates))
    }

    fn result_window(&self) -> usize {
        self.reranker
            .map_or(self.k, |request| self.k.max(request.limit))
    }

    fn finish_retrieval(
        &mut self,
        mut results: Vec<NativeShardedResult>,
        candidates: NativePhaseCandidates,
        batch: LexicalCandidateBatch,
    ) -> Vec<NativeShardedResult> {
        if self.reranker.is_some() && !results.is_empty() {
            let displayed = results.iter().take(self.k).cloned().collect();
            self.state = State::Rerank(PendingRerank {
                results,
                candidates,
                _batch: batch,
            });
            displayed
        } else {
            results.truncate(self.k);
            results
        }
    }

    async fn rerank(
        &self,
        pending: PendingRerank,
    ) -> SearchResult<Option<NativeShardedSearchPhase>> {
        let request = self.reranker.as_ref().ok_or_else(|| {
            invalid(
                "shards.rerank.configuration",
                "missing",
                "a queued rerank must retain its provider",
            )
        })?;
        match request
            .plan(
                self.cx,
                self.text,
                pending.results.iter().map(|hit| &hit.result),
            )
            .await
        {
            Ok(Some(plan)) => {
                let evaluated = plan.evaluated;
                // Apply the admitted permutation to the WHOLE envelope. A rank
                // supplied by the model cannot be used as a shard/physical row.
                let mut results = plan.apply(pending.results, |hit| &mut hit.result);
                results.truncate(self.k);
                checkpoint(self.cx, "native_ann.shards.rerank_complete")?;
                Ok(Some(NativeShardedSearchPhase::Reranked {
                    results,
                    candidates: pending.candidates,
                    evaluated,
                }))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                checkpoint(self.cx, "native_ann.shards.rerank_failure")?;
                if matches!(error, SearchError::Cancelled { .. }) {
                    return Err(error);
                }
                let mut previous_results = pending.results;
                previous_results.truncate(self.k);
                Ok(Some(NativeShardedSearchPhase::RerankFailed {
                    previous_results,
                    error,
                }))
            }
        }
    }
}

fn candidate_budget(k: usize) -> SearchResult<usize> {
    k.checked_mul(3).ok_or_else(|| {
        invalid(
            "shards.candidate_budget",
            "overflow",
            "three times the result count must fit usize",
        )
    })
}

fn admit_tiers(
    cx: &Cx,
    fast: &NativeShardSet,
    fast_embedder: &dyn Embedder,
    quality: Option<(&NativeShardSet, &dyn Embedder)>,
) -> SearchResult<bool> {
    checkpoint(cx, "native_ann.shards.tiers")?;
    let first = fast.shard(0).ok_or_else(|| {
        invalid(
            "shards.inventory",
            "empty",
            "missing admitted fast identity",
        )
    })?;
    if let Some((quality, embedder)) = quality {
        let quality = quality.shard(0).ok_or_else(|| {
            invalid(
                "shards.inventory",
                "empty",
                "missing admitted quality identity",
            )
        })?;
        // Each immutable set already joined every partition to its first owner.
        // Reuse the existing cross-tier generation/input/kind admission exactly.
        return admit_pair(cx, first, fast_embedder, (quality, embedder));
    }
    let identity = fast_embedder.identity()?;
    first.admit_identity(identity)?;
    Ok(matches!(
        identity.space.kind,
        EmbeddingSpaceKindV1::HashControl
    ))
}

async fn finish(
    cx: &Cx,
    lexical: &dyn LexicalRead,
    batch: &LexicalCandidateBatch,
    fast: &[NativeShardHit],
    quality: &[NativeShardHit],
    k: usize,
    is_hash: bool,
    blend: bool,
) -> SearchResult<Vec<NativeShardedResult>> {
    checkpoint(cx, "native_ann.shards.fusion")?;
    let fast_vectors = ranking_vectors(fast);
    let vectors = if blend {
        blend_two_tier(&fast_vectors, &ranking_vectors(quality), 0.7)
    } else {
        fast_vectors
    };
    let mut fused = rrf_fuse_for_vector_lane(
        batch.results(),
        &vectors,
        k,
        0,
        &RrfConfig::default(),
        is_hash,
    );
    // The ranking kernels use u32 indices but do not interpret them. Erase
    // their local-row placeholders BEFORE any metadata backend sees winners.
    for hit in &mut fused {
        hit.semantic_index = None;
    }
    let winners = hydrate_winners(cx, lexical, fused, batch).await?;
    let fast: BTreeMap<_, _> = fast.iter().map(|hit| (hit.doc_id.as_str(), hit)).collect();
    let quality: BTreeMap<_, _> = quality
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit))
        .collect();
    let mut results = Vec::with_capacity(winners.len());
    for mut result in winners {
        checkpoint(cx, "native_ann.shards.winner")?;
        let fast = fast.get(result.doc_id.as_str());
        let quality = quality.get(result.doc_id.as_str());
        result.index = None;
        result.fast_score = if is_hash {
            None
        } else {
            fast.map(|hit| hit.score)
        };
        result.quality_score = if is_hash {
            None
        } else {
            quality.map(|hit| hit.score)
        };
        if result.lexical_score.is_none() && !is_hash && quality.is_some() {
            result.source = ScoreSource::SemanticQuality;
        }
        results.push(NativeShardedResult {
            fast_row: fast.map(|hit| hit.row),
            quality_row: quality.map(|hit| hit.row),
            result,
        });
    }
    checkpoint(cx, "native_ann.shards.hydrated")?;
    Ok(results)
}

fn ranking_vectors(hits: &[NativeShardHit]) -> Vec<VectorHit> {
    hits.iter()
        .map(|hit| VectorHit {
            doc_id: hit.doc_id.clone(),
            score: hit.score,
            index: hit.row.physical_row,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use crate::native_ann::NativeAnnIndex;
    use frankensearch_core::LexicalHydrationContext;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

    #[derive(Clone, Copy)]
    enum Reply {
        Ready,
        Pending,
        Cancelled,
        Foreign,
        Failed,
    }
    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        values: Vec<f32>,
        calls: AtomicUsize,
        drops: AtomicUsize,
        reply: Reply,
    }
    impl Provider {
        fn new(name: &str, dimension: usize, semantic: bool) -> Self {
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
                name,
                u32::try_from(dimension).unwrap(),
            );
            if semantic {
                // Synthetic semantic-shaped fixture; not evidence of learned retrieval quality.
                identity.space.kind = EmbeddingSpaceKindV1::Semantic;
                identity.space.hash_control = None;
                identity.space.artifact_manifest_fingerprint = "a".repeat(64);
                identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                    role: "weights".to_owned(),
                    sha256: "b".repeat(64),
                    size: 1,
                }];
                identity.producer.space_fingerprint = identity.space.fingerprint();
            }
            identity.validate().unwrap();
            let mut values = vec![0.0; dimension];
            values[0] = 1.0;
            Self {
                identity,
                values,
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
                reply: Reply::Ready,
            }
        }
    }
    struct DropCount<'a>(&'a AtomicUsize);
    impl Drop for DropCount<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move { Ok(self.values.clone()) })
        }
        fn embed_bound<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                let mut identity = self.identity.clone();
                match self.reply {
                    Reply::Ready => {}
                    Reply::Pending => return std::future::pending().await,
                    Reply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "test.sharded_provider".to_owned(),
                            reason: "cancelled".to_owned(),
                        });
                    }
                    Reply::Failed => {
                        return Err(invalid(
                            "test.sharded_provider",
                            "failed",
                            "provider failed",
                        ));
                    }
                    Reply::Foreign => identity.producer.backend = "foreign-response".to_owned(),
                }
                Ok(IdentityBoundEmbedding {
                    values: self.values.clone(),
                    identity,
                })
            })
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn dimension(&self) -> usize {
            self.values.len()
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn is_semantic(&self) -> bool {
            matches!(self.identity.space.kind, EmbeddingSpaceKindV1::Semantic)
        }
        fn category(&self) -> ModelCategory {
            if self.is_semantic() {
                ModelCategory::TransformerEmbedder
            } else {
                ModelCategory::HashEmbedder
            }
        }
    }

    struct Lexical {
        ids: Vec<&'static str>,
        calls: AtomicUsize,
        hydrations: AtomicUsize,
        current_generation: AtomicUsize,
        snapshot: Arc<usize>,
    }
    impl Lexical {
        fn new(ids: &[&'static str]) -> Self {
            Self {
                ids: ids.to_vec(),
                calls: AtomicUsize::new(0),
                hydrations: AtomicUsize::new(0),
                current_generation: AtomicUsize::new(7),
                snapshot: Arc::new(7),
            }
        }
    }
    impl LexicalRead for Lexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(self
                    .ids
                    .iter()
                    .take(limit)
                    .map(|id| ScoredResult {
                        doc_id: (*id).into(),
                        score: 10.0,
                        source: ScoreSource::Lexical,
                        index: None,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some(10.0),
                        rerank_score: None,
                        explanation: None,
                        metadata: None,
                    })
                    .collect())
            })
        }
        fn search_candidates<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, LexicalCandidateBatch> {
            Box::pin(async move {
                Ok(LexicalCandidateBatch::deferred(
                    self.search(cx, text, limit).await?,
                    LexicalHydrationContext::new(
                        "sharded-test",
                        Box::new(Arc::clone(&self.snapshot)),
                    ),
                ))
            })
        }
        fn hydrate_candidates<'a>(
            &'a self,
            _cx: &'a Cx,
            context: Option<&'a LexicalHydrationContext>,
            results: &'a mut [ScoredResult],
        ) -> SearchFuture<'a, ()> {
            Box::pin(async move {
                self.hydrations.fetch_add(1, Ordering::SeqCst);
                let pin = context
                    .and_then(LexicalHydrationContext::downcast_ref::<Arc<usize>>)
                    .unwrap();
                assert!(Arc::ptr_eq(pin, &self.snapshot));
                for result in results {
                    // Ambiguous per-shard u32 indices must never reach hydration.
                    assert!(result.index.is_none());
                    if result.lexical_score.is_some() {
                        result.metadata = Some(Arc::new(serde_json::json!({"generation": **pin})));
                    }
                }
                Ok(())
            })
        }
        fn doc_count(&self) -> SearchResult<usize> {
            Ok(self.ids.len())
        }
    }

    fn partition(
        cx: &Cx,
        provider: &Provider,
        rows: &[(&str, &[f32])],
        generation: u64,
        ann: bool,
    ) -> Arc<NativeAnnIndex> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("part.fsvi");
        let mut bundle = provider.identity.clone();
        bundle.storage.format = "fsvi-v2".to_owned();
        bundle.storage.quantization = QuantizationFormat::F32;
        bundle.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(generation, [0x72; 16]).unwrap(),
            bundle.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(id, values) in rows {
            writer.write_record(id, values).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        Arc::new(if ann {
            NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
        } else {
            NativeAnnIndex::exact(cx, owner).unwrap()
        })
    }
    fn set(cx: &Cx, parts: Vec<Arc<NativeAnnIndex>>) -> NativeShardSet {
        let expected: Vec<_> = parts
            .iter()
            .map(|part| part.owner_witness().clone())
            .collect();
        NativeShardSet::admit(cx, &expected, parts).unwrap()
    }
    fn fast_set(cx: &Cx, provider: &Provider) -> NativeShardSet {
        set(
            cx,
            vec![
                partition(
                    cx,
                    provider,
                    &[("z-fast", &[1.0, 0.0]), ("mid-fast", &[0.9, 0.1])],
                    1,
                    true,
                ),
                partition(
                    cx,
                    provider,
                    &[("near-fast", &[0.8, 0.2]), ("a-late", &[0.0, 1.0])],
                    1,
                    false,
                ),
            ],
        )
    }
    fn quality_set(cx: &Cx, provider: &Provider) -> NativeShardSet {
        set(
            cx,
            vec![
                partition(
                    cx,
                    provider,
                    &[("q-other", &[0.0, 1.0, 0.0]), ("z-fast", &[0.75, 0.0, 0.0])],
                    1,
                    false,
                ),
                partition(cx, provider, &[("a-late", &[1.0, 0.0, 0.0])], 1, true),
            ],
        )
    }

    #[test]
    fn progressive_quality_finds_a_document_outside_every_fast_candidate_shard() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let quality = Provider::new("quality", 3, true);
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = a
                .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
                .unwrap();
            let NativeShardedSearchPhase::Initial {
                results,
                candidates,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial");
            };
            assert_eq!(results[0].result.doc_id, "z-fast");
            assert_eq!(candidates.fast, 3);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            let State::Refine(pending) = &stream.state else {
                panic!("retained candidates");
            };
            assert!(pending.fast.iter().all(|hit| hit.doc_id != "a-late"));
            let NativeShardedSearchPhase::Refined {
                results,
                candidates,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined");
            };
            assert_eq!(results[0].result.doc_id, "a-late");
            assert_eq!(results[0].result.source, ScoreSource::SemanticQuality);
            assert_eq!(
                results[0].quality_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 0
                })
            );
            assert!(results[0].fast_row.is_none() && results[0].result.index.is_none());
            assert_eq!(results[0].result.quality_score, Some(1.0));
            assert_eq!(candidates.quality, 3);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn shared_winners_keep_distinct_fast_quality_locations_and_raw_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let quality = Provider::new("quality", 3, true);
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let hits = a
                .search_hybrid_refined_text(&cx, &fast, (&b, &quality), &lexical, "query", 4)
                .await
                .unwrap();
            let shared = hits
                .iter()
                .find(|hit| hit.result.doc_id == "z-fast")
                .unwrap();
            assert_eq!(
                shared.fast_row,
                Some(NativeShardRow {
                    shard: 0,
                    physical_row: 0
                })
            );
            assert_eq!(
                shared.quality_row,
                Some(NativeShardRow {
                    shard: 0,
                    physical_row: 1
                })
            );
            assert_eq!(
                (shared.result.fast_score, shared.result.quality_score),
                (Some(1.0), Some(0.75))
            );
            assert_eq!(shared.result.metadata.as_deref().unwrap()["generation"], 7);
            assert!(hits.iter().all(|hit| hit.result.index.is_none()));
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn quality_primary_requires_no_fast_shards_and_preserves_quality_coordinates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let quality = Provider::new("quality", 3, true);
            let b = quality_set(&cx, &quality);
            let hits = b
                .search_hybrid_quality_text(&cx, &quality, &Lexical::new(&[]), "query", 1)
                .await
                .unwrap();
            assert_eq!(hits[0].result.source, ScoreSource::SemanticQuality);
            assert!(hits[0].fast_row.is_none() && hits[0].result.fast_score.is_none());
            assert_eq!(
                hits[0].quality_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 0
                })
            );
            assert_eq!(hits[0].result.quality_score, Some(1.0));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn both_phases_hydrate_the_original_lexical_snapshot_without_requery() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let quality = Provider::new("quality", 3, true);
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&["z-fast", "a-late"]);
            let mut stream = a
                .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            lexical.current_generation.store(99, Ordering::SeqCst);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            let NativeShardedSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined");
            };
            assert!(
                results
                    .iter()
                    .all(|hit| hit.result.metadata.as_deref().unwrap()["generation"] == 7)
            );
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            let eager = a
                .search_hybrid_refined_text(&cx, &fast, (&b, &quality), &lexical, "query", 2)
                .await
                .unwrap();
            assert_eq!(
                serde_json::to_value(results.iter().map(|hit| &hit.result).collect::<Vec<_>>())
                    .unwrap(),
                serde_json::to_value(eager.iter().map(|hit| &hit.result).collect::<Vec<_>>())
                    .unwrap()
            );
            assert_eq!(
                results
                    .iter()
                    .map(|hit| (hit.fast_row, hit.quality_row))
                    .collect::<Vec<_>>(),
                eager
                    .iter()
                    .map(|hit| (hit.fast_row, hit.quality_row))
                    .collect::<Vec<_>>()
            );
        });
    }

    #[test]
    fn abandoning_quality_starts_no_work_or_drops_pending_work_without_retry() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for poll_quality in [false, true] {
                let fast = Provider::new("fast", 2, true);
                let mut quality = Provider::new("quality", 3, true);
                quality.reply = Reply::Pending;
                let a = fast_set(&cx, &fast);
                let b = quality_set(&cx, &quality);
                let lexical = Lexical::new(&["z-fast"]);
                let mut stream = a
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&b, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap();
                assert!(stream.next_phase().await.unwrap().is_some());
                if poll_quality {
                    let mut future = Box::pin(stream.next_phase());
                    assert!(matches!(
                        future
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Pending
                    ));
                    drop(future);
                    assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
                }
                drop(stream);
                assert_eq!(
                    quality.calls.load(Ordering::SeqCst),
                    usize::from(poll_quality)
                );
                assert_eq!(
                    quality.drops.load(Ordering::SeqCst),
                    usize::from(poll_quality)
                );
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            }
        });
    }

    #[test]
    fn quality_errors_preserve_initial_results_but_cancellation_is_terminal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for reply in [Reply::Failed, Reply::Foreign, Reply::Cancelled] {
                let fast = Provider::new("fast", 2, true);
                let mut quality = Provider::new("quality", 3, true);
                quality.reply = reply;
                let a = fast_set(&cx, &fast);
                let b = quality_set(&cx, &quality);
                let lexical = Lexical::new(&["z-fast"]);
                let mut stream = a
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&b, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap();
                let NativeShardedSearchPhase::Initial { results, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial");
                };
                let expected = serde_json::to_value(&results[0].result).unwrap();
                let next = stream.next_phase().await;
                if matches!(reply, Reply::Cancelled) {
                    assert!(
                        matches!(next, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.sharded_provider")
                    );
                } else {
                    let NativeShardedSearchPhase::RefinementFailed {
                        initial_results,
                        error,
                    } = next.unwrap().unwrap()
                    else {
                        panic!("explicit failure");
                    };
                    assert_eq!(
                        serde_json::to_value(&initial_results[0].result).unwrap(),
                        expected
                    );
                    assert_eq!(initial_results[0].fast_row, results[0].fast_row);
                    assert!(matches!(error, SearchError::InvalidConfig { .. }));
                }
                assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn tier_generation_mismatch_and_noop_identity_errors_precede_all_provider_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let quality = Provider::new("quality", 3, true);
            let a = fast_set(&cx, &fast);
            let b = set(&cx, vec![partition(&cx, &quality, &[], 2, false)]);
            let lexical = Lexical::new(&[]);
            for k in [0, 1] {
                assert!(
                    matches!(a.search_hybrid_refined_text(&cx, &fast, (&b, &quality), &lexical, "query", k).await, Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.tier_generation")
                );
            }
            assert!(
                a.search_hybrid_text(&cx, &quality, &lexical, "query", 0)
                    .await
                    .is_err()
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn hash_shards_keep_nonsemantic_labels_but_unambiguous_tier_coordinates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("hash-fast", 2, false);
            let quality = Provider::new("hash-quality", 3, false);
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let hits = a
                .search_hybrid_refined_text(
                    &cx,
                    &fast,
                    (&b, &quality),
                    &Lexical::new(&[]),
                    "query",
                    1,
                )
                .await
                .unwrap();
            assert_eq!(hits[0].result.source, ScoreSource::HashControl);
            assert!(
                hits[0].result.fast_score.is_none()
                    && hits[0].result.quality_score.is_none()
                    && hits[0].result.index.is_none()
            );
            assert_eq!(
                hits[0].quality_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 0
                })
            );
        });
    }

    #[test]
    fn empty_fast_shards_still_allow_quality_and_zero_k_finishes_without_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut fast = Provider::new("fast", 2, true);
            fast.reply = Reply::Pending;
            let quality = Provider::new("quality", 3, true);
            let a = set(&cx, vec![partition(&cx, &fast, &[], 1, false)]);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut empty = a
                .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 0)
                .unwrap();
            assert!(
                matches!(empty.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { results, .. }) if results.is_empty())
            );
            assert!(empty.next_phase().await.unwrap().is_none());
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
            let mut stream = a
                .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
                .unwrap();
            drop(stream.next_phase()); // Unpolled futures do not consume a phase.
            assert!(!stream.is_finished());
            assert!(
                matches!(stream.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { results, .. }) if results.is_empty())
            );
            assert!(
                matches!(stream.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Refined { results, .. }) if results[0].result.doc_id == "a-late")
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[derive(Clone, Copy)]
    enum RerankReply {
        Ready,
        Pending,
        Invalid,
        Failed,
        Cancelled,
        CancelWithScores,
        Unavailable,
    }

    struct CrossEncoder {
        winner: &'static str,
        reply: RerankReply,
        calls: AtomicUsize,
        drops: AtomicUsize,
    }

    impl CrossEncoder {
        fn new(winner: &'static str, reply: RerankReply) -> Self {
            Self {
                winner,
                reply,
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
            }
        }
    }

    impl Reranker for CrossEncoder {
        fn rerank<'a>(
            &'a self,
            cx: &'a Cx,
            _query: &'a str,
            documents: &'a [frankensearch_core::traits::RerankDocument],
        ) -> SearchFuture<'a, Vec<frankensearch_core::traits::RerankScore>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                let mut scores: Vec<_> = documents
                    .iter()
                    .enumerate()
                    .map(|(rank, doc)| {
                        assert_eq!(doc.text, format!("generation-seven:{}", doc.doc_id));
                        frankensearch_core::traits::RerankScore {
                            doc_id: doc.doc_id.clone(),
                            score: if doc.doc_id == self.winner {
                                0.95
                            } else {
                                0.05
                            },
                            original_rank: rank,
                            raw_logit: None,
                        }
                    })
                    .collect();
                match self.reply {
                    RerankReply::Ready => {}
                    RerankReply::Pending => return std::future::pending().await,
                    RerankReply::Invalid => scores[1] = scores[0].clone(),
                    RerankReply::Failed => {
                        return Err(invalid("test.sharded_reranker", "failed", "model failed"));
                    }
                    RerankReply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "test.sharded_reranker".to_owned(),
                            reason: "model cancelled".to_owned(),
                        });
                    }
                    RerankReply::CancelWithScores => {
                        cx.cancel_with(
                            asupersync::CancelKind::User,
                            Some("cancel with sharded rerank scores"),
                        );
                    }
                    RerankReply::Unavailable => panic!("unavailable model must not run"),
                }
                // Providers may sort outputs: never confuse output rank with
                // original rank, much less either rank with a physical row.
                scores.reverse();
                Ok(scores)
            })
        }

        // `Reranker::id` declares `-> &str`; a trait impl cannot narrow that to
        // `&'static str`, which is what clippy's suggestion would do.
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "sharded-test-cross-encoder"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn is_available(&self) -> bool {
            !matches!(self.reply, RerankReply::Unavailable)
        }
    }

    // Fed to with_reranker as `&dyn Fn(&str) -> Option<String>`, so the Option
    // is required by the callback type rather than incidental.
    #[allow(clippy::unnecessary_wraps)]
    fn source_text(id: &str) -> Option<String> {
        Some(format!("generation-seven:{id}"))
    }

    #[test]
    fn rerank_promotes_an_undisplayed_second_shard_hit_with_one_global_model_call() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let a = fast_set(&cx, &fast); // one ANN shard, one exact shard
            let lexical = Lexical::new(&["z-fast", "mid-fast", "near-fast", "a-late"]);
            let model = CrossEncoder::new("a-late", RerankReply::Ready);
            let text_calls = AtomicUsize::new(0);
            let text = |id: &str| {
                assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
                text_calls.fetch_add(1, Ordering::SeqCst);
                source_text(id)
            };
            let mut stream = a
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &text, 4)
                .unwrap();
            let NativeShardedSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(results[0].result.doc_id, "z-fast");
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
            assert_eq!(text_calls.load(Ordering::SeqCst), 0);
            let State::Rerank(pending) = &stream.state else {
                panic!("retained final window")
            };
            let original = pending
                .results
                .iter()
                .find(|hit| hit.result.doc_id == "a-late")
                .unwrap()
                .clone();
            lexical.current_generation.store(99, Ordering::SeqCst);
            let NativeShardedSearchPhase::Reranked {
                results,
                evaluated,
                candidates,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("reranked")
            };
            assert_eq!(results.len(), 1);
            let winner = &results[0];
            assert_eq!(winner.result.doc_id, "a-late");
            assert_eq!(
                winner.fast_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 1
                })
            );
            assert_eq!(winner.fast_row, original.fast_row);
            assert!(winner.quality_row.is_none() && winner.result.index.is_none());
            assert_eq!(winner.result.rerank_score, Some(0.95));
            assert_eq!(winner.result.source, ScoreSource::Reranked);
            assert_eq!(
                winner.result.score.to_bits(),
                original.result.score.to_bits()
            );
            assert_eq!(winner.result.fast_score, original.result.fast_score);
            assert!(Arc::ptr_eq(
                winner.result.metadata.as_ref().unwrap(),
                original.result.metadata.as_ref().unwrap()
            ));
            assert_eq!(winner.result.metadata.as_deref().unwrap()["generation"], 7);
            assert_eq!(evaluated, 4);
            assert_eq!(
                candidates,
                NativePhaseCandidates {
                    fast: 4,
                    quality: 0,
                    lexical: 4
                }
            );
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
            assert_eq!(text_calls.load(Ordering::SeqCst), 4);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn final_permutation_preserves_distinct_tier_rows_and_quality_only_provenance() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for target in ["z-fast", "a-late", "q-other"] {
                let fast = Provider::new("fast", 2, true);
                let quality = Provider::new("quality", 3, true);
                let a = fast_set(&cx, &fast);
                let b = quality_set(&cx, &quality);
                let lexical =
                    Lexical::new(&["z-fast", "mid-fast", "near-fast", "a-late", "q-other"]);
                let model = CrossEncoder::new(target, RerankReply::Ready);
                let mut stream = a
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&b, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap()
                    .with_reranker(&model, &source_text, 5)
                    .unwrap();
                assert!(matches!(
                    stream.next_phase().await.unwrap(),
                    Some(NativeShardedSearchPhase::Initial { .. })
                ));
                assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                assert!(matches!(
                    stream.next_phase().await.unwrap(),
                    Some(NativeShardedSearchPhase::Refined { .. })
                ));
                assert_eq!(model.calls.load(Ordering::SeqCst), 0);
                let State::Rerank(pending) = &stream.state else {
                    panic!("retained refined window")
                };
                let original = pending
                    .results
                    .iter()
                    .find(|hit| hit.result.doc_id == target)
                    .unwrap()
                    .clone();
                let NativeShardedSearchPhase::Reranked {
                    results, evaluated, ..
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("final")
                };
                assert_eq!(evaluated, 5);
                let winner = &results[0];
                assert_eq!(winner.result.doc_id, target);
                assert_eq!(
                    (winner.fast_row, winner.quality_row),
                    (original.fast_row, original.quality_row)
                );
                assert_eq!(winner.result.fast_score, original.result.fast_score);
                assert_eq!(winner.result.quality_score, original.result.quality_score);
                assert_eq!(
                    winner.result.score.to_bits(),
                    original.result.score.to_bits()
                );
                assert!(Arc::ptr_eq(
                    winner.result.metadata.as_ref().unwrap(),
                    original.result.metadata.as_ref().unwrap()
                ));
                assert!(winner.result.index.is_none());
                match target {
                    "z-fast" => {
                        assert_eq!(
                            winner.fast_row,
                            Some(NativeShardRow {
                                shard: 0,
                                physical_row: 0
                            })
                        );
                        assert_eq!(
                            winner.quality_row,
                            Some(NativeShardRow {
                                shard: 0,
                                physical_row: 1
                            })
                        );
                        assert_eq!(
                            (winner.result.fast_score, winner.result.quality_score),
                            (Some(1.0), Some(0.75))
                        );
                    }
                    "a-late" => {
                        assert_eq!(
                            winner.fast_row,
                            Some(NativeShardRow {
                                shard: 1,
                                physical_row: 1
                            })
                        );
                        assert_eq!(
                            winner.quality_row,
                            Some(NativeShardRow {
                                shard: 1,
                                physical_row: 0
                            })
                        );
                    }
                    _ => {
                        assert!(winner.fast_row.is_none() && winner.result.fast_score.is_none());
                        assert_eq!(
                            winner.quality_row,
                            Some(NativeShardRow {
                                shard: 0,
                                physical_row: 0
                            })
                        );
                        assert_eq!(winner.result.quality_score, Some(0.0));
                    }
                }
                assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
                assert_eq!(model.calls.load(Ordering::SeqCst), 1);
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            }
        });
    }

    #[test]
    fn quality_failure_keeps_its_page_and_allows_the_requested_final_stage() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let mut quality = Provider::new("quality", 3, true);
            quality.reply = Reply::Failed;
            let a = fast_set(&cx, &fast);
            let b = quality_set(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let model = CrossEncoder::new("near-fast", RerankReply::Ready);
            let mut stream = a
                .search_hybrid_progressive(&cx, &fast, Some((&b, &quality)), &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &source_text, 4)
                .unwrap();
            let NativeShardedSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            let expected = serde_json::to_value(&results[0].result).unwrap();
            let NativeShardedSearchPhase::RefinementFailed {
                initial_results, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("quality failure is explicit")
            };
            assert_eq!(
                serde_json::to_value(&initial_results[0].result).unwrap(),
                expected
            );
            assert_eq!(initial_results[0].fast_row, results[0].fast_row);
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
            let NativeShardedSearchPhase::Reranked { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("independent final stage")
            };
            assert_eq!(results[0].result.doc_id, "near-fast");
            assert_eq!(
                results[0].fast_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 0
                })
            );
            assert!(results[0].quality_row.is_none() && results[0].result.quality_score.is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn invalid_model_response_keeps_the_whole_previous_envelope_unchanged() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let a = fast_set(&cx, &fast);
            for reply in [
                RerankReply::Invalid,
                RerankReply::Failed,
                RerankReply::Unavailable,
            ] {
                let lexical = Lexical::new(&["z-fast", "mid-fast", "near-fast", "a-late"]);
                let model = CrossEncoder::new("a-late", reply);
                let mut stream = a
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 2)
                    .unwrap()
                    .with_reranker(&model, &source_text, 4)
                    .unwrap();
                let NativeShardedSearchPhase::Initial { results, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial")
                };
                let NativeShardedSearchPhase::RerankFailed {
                    previous_results, ..
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("atomic failure")
                };
                assert_eq!(previous_results.len(), results.len());
                for (previous, initial) in previous_results.iter().zip(&results) {
                    assert_eq!(
                        serde_json::to_value(&previous.result).unwrap(),
                        serde_json::to_value(&initial.result).unwrap()
                    );
                    assert_eq!(
                        (previous.fast_row, previous.quality_row),
                        (initial.fast_row, initial.quality_row)
                    );
                    assert!(Arc::ptr_eq(
                        previous.result.metadata.as_ref().unwrap(),
                        initial.result.metadata.as_ref().unwrap()
                    ));
                }
                assert_eq!(
                    model.calls.load(Ordering::SeqCst),
                    usize::from(!matches!(reply, RerankReply::Unavailable))
                );
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
                assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
            }
        });
    }

    #[test]
    fn stopping_before_or_during_rerank_drops_the_pin_without_restarting_any_shard() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for poll_final in [false, true] {
                let fast = Provider::new("fast", 2, true);
                let a = fast_set(&cx, &fast);
                let lexical = Lexical::new(&["z-fast"]);
                let model = CrossEncoder::new("near-fast", RerankReply::Pending);
                let mut stream = a
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                    .unwrap()
                    .with_reranker(&model, &source_text, 4)
                    .unwrap();
                assert!(stream.next_phase().await.unwrap().is_some());
                drop(stream.next_phase()); // Unpolled final work remains unstarted.
                assert_eq!(model.calls.load(Ordering::SeqCst), 0);
                assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
                if poll_final {
                    let mut future = Box::pin(stream.next_phase());
                    assert!(matches!(
                        future
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Pending
                    ));
                    drop(future);
                    assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
                }
                drop(stream);
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
                assert_eq!(model.calls.load(Ordering::SeqCst), usize::from(poll_final));
                assert_eq!(model.drops.load(Ordering::SeqCst), usize::from(poll_final));
                assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn sharded_rerank_cancellation_dominates_both_errors_and_ready_scores() {
        for reply in [RerankReply::Cancelled, RerankReply::CancelWithScores] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fast = Provider::new("fast", 2, true);
                let a = fast_set(&cx, &fast);
                let lexical = Lexical::new(&["z-fast"]);
                let model = CrossEncoder::new("a-late", reply);
                let mut stream = a
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                    .unwrap()
                    .with_reranker(&model, &source_text, 4)
                    .unwrap();
                assert!(stream.next_phase().await.unwrap().is_some());
                assert!(matches!(
                    stream.next_phase().await,
                    Err(SearchError::Cancelled { .. })
                ));
                assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
                assert_eq!(model.calls.load(Ordering::SeqCst), 1);
            });
        }
    }

    #[test]
    fn missing_text_and_hash_tiers_do_not_invent_semantic_scores_or_row_locations() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("hash-fast", 2, false);
            let a = fast_set(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let model = CrossEncoder::new("near-fast", RerankReply::Ready);
            let only_near =
                |id: &str| (id == "near-fast").then(|| format!("generation-seven:{id}"));
            let mut stream = a
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 4)
                .unwrap()
                .with_reranker(&model, &only_near, 3)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let NativeShardedSearchPhase::Reranked {
                results, evaluated, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("one scored candidate")
            };
            assert_eq!(evaluated, 1);
            assert_eq!(results[0].result.doc_id, "near-fast");
            assert_eq!(
                results[0].fast_row,
                Some(NativeShardRow {
                    shard: 1,
                    physical_row: 0
                })
            );
            assert_eq!(results[0].result.source, ScoreSource::Reranked);
            assert!(results.iter().all(|hit| hit.result.fast_score.is_none()
                && hit.result.quality_score.is_none()
                && hit.result.index.is_none()
                && hit.quality_row.is_none()));
            assert_eq!(
                results[1..]
                    .iter()
                    .map(|hit| hit.result.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["z-fast", "mid-fast", "a-late"]
            );
            assert!(
                results[1..]
                    .iter()
                    .all(|hit| hit.result.source == ScoreSource::HashControl
                        && hit.result.rerank_score.is_none())
            );
            let no_text = |_: &str| None;
            let mut empty = a
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &no_text, 3)
                .unwrap();
            assert!(empty.next_phase().await.unwrap().is_some());
            assert!(empty.next_phase().await.unwrap().is_none());
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn sharded_rerank_noops_and_invalid_budgets_start_no_provider_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2, true);
            let a = fast_set(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let model = CrossEncoder::new("a-late", RerankReply::Ready);
            for budget in [0, usize::MAX] {
                assert!(
                    a.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                        .unwrap()
                        .with_reranker(&model, &source_text, budget)
                        .is_err()
                );
            }
            let mut empty = a
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 0)
                .unwrap()
                .with_reranker(&model, &source_text, 4)
                .unwrap();
            assert!(
                matches!(empty.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { results, .. }) if results.is_empty())
            );
            assert!(empty.is_finished() && empty.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
            let mut late = a
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap();
            assert!(late.next_phase().await.unwrap().is_some());
            assert!(late.with_reranker(&model, &source_text, 4).is_err());
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        });
    }
}
