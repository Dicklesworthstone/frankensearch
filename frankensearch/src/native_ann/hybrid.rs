//! Native ANN/lexical fusion with caller-owned execution and snapshot pins.

use std::future::{Future, poll_fn};
use std::task::Poll;

use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::{BoundQueryEmbedding, FusedHit, LexicalCandidateBatch, LexicalRead};
use frankensearch_fusion::{RrfConfig, candidate_count, rrf_fuse_for_vector_lane};

use super::{NativeAnnIndex, checkpoint, invalid};
use crate::{Cx, Embedder, ScoreSource, ScoredResult, SearchResult};

mod progressive;
mod refinement;
mod sharded;

pub use progressive::{NativePhaseCandidates, NativeProgressiveSearch, NativeSearchPhase};
pub use sharded::{NativeShardedProgressiveSearch, NativeShardedResult, NativeShardedSearchPhase};

impl NativeAnnIndex {
    /// Execute native hybrid retrieval and return hydrated final winners.
    ///
    /// Eager metadata is reused from the original lexical candidates. Deferred
    /// metadata is requested only for the winning lexical documents, with the
    /// exact snapshot context returned by that candidate search. No new search,
    /// reader reopen, or current-generation metadata lookup is performed here.
    /// Vector-only winners have no lexical metadata to hydrate.
    ///
    /// Fused rank order, native row indices and lexical scores are preserved.
    /// This single-tier convenience API treats its retained vector owner as
    /// the fast retrieval lane; it does not execute quality-tier refinement.
    /// Hash-only hits are labelled [`ScoreSource::HashControl`]; their raw
    /// control scores are not placed in semantic `fast_score` fields. Use
    /// [`Self::search_hybrid_candidates`] to retain the explicit hash scores and
    /// per-source ranks, which [`ScoredResult`] cannot represent separately.
    /// Execution and cross-reader generation requirements are the same as for
    /// [`Self::search_hybrid_candidates`].
    ///
    /// # Errors
    ///
    /// Propagates retrieval, hydration and cancellation errors. A partially
    /// hydrated result set is never returned after failure or cancellation.
    pub async fn search_hybrid_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        let (hits, batch) = self
            .search_hybrid_candidates(cx, embedder, lexical, text, k)
            .await?;
        hydrate_winners(cx, lexical, hits, &batch).await
    }

    /// Search the native graph and a read-only lexical backend, then fuse ranks.
    ///
    /// Inference and lexical retrieval are polled concurrently in this future;
    /// neither needs a detached task or a backend that completes in one poll.
    /// Both sources must succeed. A provider error or cancellation drops the
    /// sibling future rather than silently changing the retrieval topology.
    /// Graph traversal remains synchronous on the caller's execution lane.
    ///
    /// Returns the lexical batch alongside the ranked hits. Keep that batch
    /// until winner hydration completes: its opaque context pins the exact
    /// lexical snapshot that scored these candidates. The vector owner stays
    /// retained by `self`. This does not establish a composite generation across
    /// independently supplied lexical and vector readers; their selection is
    /// the caller's responsibility.
    ///
    /// Uses the standard three-times candidate budget and default RRF policy.
    /// For independent scheduling or fusion weights, obtain a bound query and
    /// a lexical batch separately, then call [`Self::fuse_candidates`].
    /// Zero-k requests still admit the configured embedding identity, but start
    /// neither provider. An empty vector owner does not suppress lexical hits.
    ///
    /// # Errors
    ///
    /// Propagates identity, inference, lexical, graph, and cancellation errors.
    /// Non-finite lexical scores are rejected before they reach fusion.
    pub async fn search_hybrid_candidates(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<(Vec<FusedHit>, LexicalCandidateBatch)> {
        checkpoint(cx, "native_ann.hybrid_start")?;
        self.admit_identity(embedder.identity()?)?;
        if k == 0 {
            return Ok((Vec::new(), LexicalCandidateBatch::eager(Vec::new())));
        }
        let budget = candidate_count(k, 0, 3);
        if self.live_count() == 0 {
            // The admitted owner proves there is no vector work. This is not
            // a provider-failure fallback and still requires lexical success.
            let batch = lexical.search_candidates(cx, text, budget).await?;
            validate_lexical(cx, &batch)?;
            let hits =
                rrf_fuse_for_vector_lane(batch.results(), &[], k, 0, &RrfConfig::default(), false);
            checkpoint(cx, "native_ann.hybrid_complete")?;
            return Ok((hits, batch));
        }
        let (query, batch) = join_sources(
            cx,
            self.embed_query(cx, embedder, text),
            lexical.search_candidates(cx, text, budget),
        )
        .await?;
        let hits = self.fuse_candidates(cx, &query, &batch, k, None, &RrfConfig::default())?;
        Ok((hits, batch))
    }

    /// Fuse a pinned lexical batch with retrieval from a bound native query.
    ///
    /// The vector budget is three times `k`; a caller needing another candidate
    /// budget can search the native graph directly and use the public RRF
    /// functions. `ef` controls the native graph beam. This method neither
    /// hydrates nor discards the lexical pin, and does not reopen either source.
    /// Hash-control spaces retain hash ranks rather than being relabelled as
    /// semantic retrieval, irrespective of an embedder's display name.
    ///
    /// # Errors
    ///
    /// Returns identity, graph, cancellation, or invalid lexical score errors.
    pub fn fuse_candidates(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        batch: &LexicalCandidateBatch,
        k: usize,
        ef: Option<usize>,
        config: &RrfConfig,
    ) -> SearchResult<Vec<FusedHit>> {
        checkpoint(cx, "native_ann.hybrid_fusion")?;
        self.admit_identity(query.identity())?;
        validate_lexical(cx, batch)?;
        let vectors = self.search(cx, query, candidate_count(k, 0, 3), ef)?;
        let is_hash = matches!(
            query.identity().space.kind,
            EmbeddingSpaceKindV1::HashControl
        );
        let hits = rrf_fuse_for_vector_lane(batch.results(), &vectors, k, 0, config, is_hash);
        checkpoint(cx, "native_ann.hybrid_complete")?;
        Ok(hits)
    }
}

/// Only the private path can pair fused winners with a hydration batch. Public
/// callers cannot pass a fresh batch to this helper and silently lose the pin.
async fn hydrate_winners(
    cx: &Cx,
    lexical: &dyn LexicalRead,
    hits: Vec<FusedHit>,
    batch: &LexicalCandidateBatch,
) -> SearchResult<Vec<ScoredResult>> {
    checkpoint(cx, "native_ann.hybrid_materialize")?;
    let mut results = Vec::with_capacity(hits.len());
    for hit in hits {
        checkpoint(cx, "native_ann.hybrid_winner")?;
        results.push(materialize_winner(hit, batch)?);
    }
    if batch.is_deferred() && results.iter().any(|result| result.lexical_score.is_some()) {
        checkpoint(cx, "native_ann.hybrid_before_hydration")?;
        let response = lexical
            .hydrate_candidates(cx, batch.context(), &mut results)
            .await;
        checkpoint(cx, "native_ann.hybrid_after_hydration")?;
        response?;
    }
    checkpoint(cx, "native_ann.hybrid_results_complete")?;
    Ok(results)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "ScoredResult exposes f32; fusion retains f64 until this final checked conversion"
)]
fn materialize_winner(hit: FusedHit, batch: &LexicalCandidateBatch) -> SearchResult<ScoredResult> {
    let metadata = if let Some(rank) = hit.lexical_rank {
        let candidate = batch
            .results()
            .get(rank)
            .filter(|candidate| candidate.doc_id == hit.doc_id)
            .ok_or_else(|| {
                invalid(
                    "lexical_rank",
                    "batch-mismatch",
                    "fused lexical rank must resolve to the original scoring candidate",
                )
            })?;
        candidate.metadata.clone()
    } else {
        None
    };
    let score = hit.rrf_score as f32;
    if !score.is_finite() {
        return Err(invalid(
            "fused_score",
            "non-finite",
            "fused score must remain finite at the public result precision",
        ));
    }
    let source = if hit.in_both_sources {
        ScoreSource::Hybrid
    } else if hit.lexical_rank.is_some() {
        ScoreSource::Lexical
    } else if hit.hash_rank.is_some() {
        ScoreSource::HashControl
    } else {
        ScoreSource::SemanticFast
    };
    Ok(ScoredResult {
        doc_id: hit.doc_id,
        score,
        source,
        index: hit.semantic_index,
        fast_score: hit.semantic_score,
        quality_score: None,
        lexical_score: hit.lexical_score,
        rerank_score: None,
        explanation: None,
        metadata,
    })
}

fn validate_lexical(cx: &Cx, batch: &LexicalCandidateBatch) -> SearchResult<()> {
    checkpoint(cx, "native_ann.hybrid_lexical_batch")?;
    for result in batch.results() {
        checkpoint(cx, "native_ann.hybrid_lexical_candidate")?;
        if !result.score.is_finite() {
            return Err(invalid(
                "lexical_score",
                "non-finite",
                "lexical candidate scores must be finite before native hybrid fusion",
            ));
        }
    }
    Ok(())
}

/// Poll both caller-owned sources without spawning or requiring ready futures.
/// On any error the still-owned sibling is dropped by this function's return.
async fn join_sources<A, B>(
    cx: &Cx,
    first: impl Future<Output = SearchResult<A>>,
    second: impl Future<Output = SearchResult<B>>,
) -> SearchResult<(A, B)> {
    let mut first = std::pin::pin!(first);
    let mut second = std::pin::pin!(second);
    let mut first_value = None;
    let mut second_value = None;
    poll_fn(|task| {
        if let Err(error) = checkpoint(cx, "native_ann.hybrid_sources") {
            return Poll::Ready(Err(error));
        }
        if first_value.is_none() {
            match first.as_mut().poll(task) {
                Poll::Ready(Ok(value)) => first_value = Some(value),
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Pending => {}
            }
        }
        if let Err(error) = checkpoint(cx, "native_ann.hybrid_between_sources") {
            return Poll::Ready(Err(error));
        }
        if second_value.is_none() {
            match second.as_mut().poll(task) {
                Poll::Ready(Ok(value)) => second_value = Some(value),
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Pending => {}
            }
        }
        if let Some(first) = first_value.take() {
            if let Some(second) = second_value.take() {
                return Poll::Ready(Ok((first, second)));
            }
            first_value = Some(first);
        }
        Poll::Pending
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Waker};

    use frankensearch_core::LexicalHydrationContext;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

    use crate::SearchError;

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
        drops: AtomicUsize,
        pending: bool,
    }

    impl Provider {
        fn new(pending: bool) -> Self {
            Self {
                identity: identity(),
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
                pending,
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
            Box::pin(async { Ok(vec![1.0, 0.0]) })
        }

        fn embed_bound<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                let _guard = DropCount(&self.drops);
                if self.pending {
                    std::future::pending::<()>().await;
                }
                Ok(IdentityBoundEmbedding {
                    values: vec![1.0, 0.0],
                    identity: self.identity.clone(),
                })
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn id(&self) -> &'static str {
            "native-hybrid-provider"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn dimension(&self) -> usize {
            2
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn is_semantic(&self) -> bool {
            false
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    enum BatchMode {
        Deferred,
        Eager,
        Foreign,
    }

    enum HydrationReply {
        Correct,
        Cancelled,
        Pending,
    }

    struct Lexical {
        calls: AtomicUsize,
        cancelled: bool,
        pending: bool,
        snapshot: Arc<usize>,
        latest_generation: AtomicUsize,
        batch_mode: BatchMode,
        hydration_reply: HydrationReply,
        hydrations: AtomicUsize,
        hydrated_docs: AtomicUsize,
        hydration_drops: AtomicUsize,
    }

    impl Lexical {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                cancelled: false,
                pending: false,
                snapshot: Arc::new(1),
                latest_generation: AtomicUsize::new(1),
                batch_mode: BatchMode::Deferred,
                hydration_reply: HydrationReply::Correct,
                hydrations: AtomicUsize::new(0),
                hydrated_docs: AtomicUsize::new(0),
                hydration_drops: AtomicUsize::new(0),
            }
        }
    }

    fn lexical_result(id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            doc_id: id.into(),
            score,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: None,
            metadata: None,
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
                if self.cancelled {
                    return Err(SearchError::Cancelled {
                        phase: "test.lexical".to_owned(),
                        reason: "cancelled lexical source".to_owned(),
                    });
                }
                if self.pending {
                    std::future::pending::<()>().await;
                }
                let mut results = vec![lexical_result("beta", 10.0), lexical_result("alpha", 5.0)];
                results.truncate(limit);
                Ok(results)
            })
        }

        fn search_candidates<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, LexicalCandidateBatch> {
            Box::pin(async move {
                let mut results = self.search(cx, text, limit).await?;
                self.latest_generation.store(2, Ordering::SeqCst);
                if matches!(self.batch_mode, BatchMode::Eager) {
                    for result in &mut results {
                        result.metadata = Some(Arc::new(serde_json::json!({
                            "snapshot": *self.snapshot,
                        })));
                    }
                    return Ok(LexicalCandidateBatch::eager(results));
                }
                let snapshot = if matches!(self.batch_mode, BatchMode::Foreign) {
                    Arc::new(99_usize)
                } else {
                    Arc::clone(&self.snapshot)
                };
                Ok(LexicalCandidateBatch::deferred(
                    results,
                    LexicalHydrationContext::new("native-hybrid-test", Box::new(snapshot)),
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
                let _guard = DropCount(&self.hydration_drops);
                let snapshot = context
                    .and_then(LexicalHydrationContext::downcast_ref::<Arc<usize>>)
                    .filter(|snapshot| Arc::ptr_eq(snapshot, &self.snapshot))
                    .ok_or_else(|| invalid("hydration", "foreign", "foreign scoring snapshot"))?;
                for result in results {
                    if result.lexical_score.is_some() {
                        result.metadata = Some(Arc::new(serde_json::json!({
                            "snapshot": **snapshot,
                        })));
                        self.hydrated_docs.fetch_add(1, Ordering::SeqCst);
                    }
                }
                match self.hydration_reply {
                    HydrationReply::Correct => Ok(()),
                    HydrationReply::Cancelled => Err(SearchError::Cancelled {
                        phase: "test.hydration".to_owned(),
                        reason: "cancelled after partial hydration".to_owned(),
                    }),
                    HydrationReply::Pending => std::future::pending().await,
                }
            })
        }

        fn doc_count(&self) -> SearchResult<usize> {
            Ok(2)
        }
    }

    fn identity() -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model("native-hybrid-test", 2)
    }

    fn index(cx: &Cx, empty: bool) -> NativeAnnIndex {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hybrid.fsvi");
        let mut bundle = identity();
        bundle.storage.format = "fsvi-v2".to_owned();
        bundle.storage.quantization = QuantizationFormat::F32;
        bundle.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(1, [0x23; 16]).unwrap(),
            bundle.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        if !empty {
            writer.write_record("alpha", &[1.0, 0.0]).unwrap();
            writer.write_record("beta", &[0.8, 0.2]).unwrap();
            writer.write_record("gamma", &[0.0, 1.0]).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
    }

    #[test]
    fn native_hybrid_fuses_actual_ranks_and_retains_lexical_snapshot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(false);
            let lexical = Lexical::new();
            let weak = Arc::downgrade(&lexical.snapshot);
            let (hits, batch) = index
                .search_hybrid_candidates(&cx, &provider, &lexical, "query", 3)
                .await
                .unwrap();
            assert_eq!(
                hits.iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["beta", "alpha", "gamma"]
            );
            assert!(hits[0].in_both_sources);
            assert_eq!(hits[0].lexical_rank, Some(0));
            assert_eq!(hits[0].hash_rank, Some(1));
            assert_eq!(hits[0].semantic_rank, None);
            assert_eq!(hits[2].lexical_rank, None);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            drop(lexical);
            assert!(weak.upgrade().is_some());
            drop(batch);
            assert!(weak.upgrade().is_none());
        });
    }

    #[test]
    fn native_hybrid_polls_lexical_while_embedding_is_pending_and_drops_on_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(true);
            let mut lexical = Lexical::new();
            lexical.cancelled = true;
            let error = index
                .search_hybrid_candidates(&cx, &provider, &lexical, "query", 2)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { ref phase, .. }
                if phase == "test.lexical"));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(index.live_count(), 3);
        });
    }

    #[test]
    fn dropping_pending_hybrid_drops_owned_work_without_changing_the_index() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(true);
            let mut lexical = Lexical::new();
            lexical.pending = true;
            let mut future =
                Box::pin(index.search_hybrid_candidates(&cx, &provider, &lexical, "query", 2));
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            drop(future);
            assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
            let ready = Provider::new(false);
            assert_eq!(
                index
                    .search_text(&cx, &ready, "query", 1, None)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "alpha"
            );
        });
    }

    #[test]
    fn hybrid_zero_k_skips_both_sources_but_not_identity_admission() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let mut provider = Provider::new(false);
            let lexical = Lexical::new();
            let (hits, batch) = index
                .search_hybrid_candidates(&cx, &provider, &lexical, "query", 0)
                .await
                .unwrap();
            assert!(hits.is_empty() && batch.results().is_empty());
            provider.identity.producer.backend = "foreign".to_owned();
            assert!(
                index
                    .search_hybrid_candidates(&cx, &provider, &lexical, "query", 0)
                    .await
                    .is_err()
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn empty_native_owner_still_returns_lexical_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, true);
            let provider = Provider::new(true);
            let lexical = Lexical::new();
            let (hits, _) = index
                .search_hybrid_candidates(&cx, &provider, &lexical, "query", 2)
                .await
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert!(
                hits.iter()
                    .all(|hit| hit.hash_rank.is_none() && hit.semantic_rank.is_none())
            );
            assert_eq!(hits[0].doc_id, "beta");
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn native_hybrid_rejects_non_finite_lexical_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity()).unwrap();
            for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let batch = LexicalCandidateBatch::eager(vec![lexical_result("beta", score)]);
                assert!(matches!(
                    index.fuse_candidates(&cx, &query, &batch, 2, None, &RrfConfig::default()),
                    Err(SearchError::InvalidConfig { .. })
                ));
            }
        });
    }

    #[test]
    fn hybrid_text_hydrates_only_winners_from_the_scoring_snapshot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(false);
            let lexical = Lexical::new();
            let results = index
                .search_hybrid_text(&cx, &provider, &lexical, "query", 1)
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            let winner = &results[0];
            assert_eq!(winner.doc_id, "beta");
            assert_eq!(winner.source, ScoreSource::Hybrid);
            assert_eq!(winner.index, Some(1));
            assert_eq!(winner.lexical_score, Some(10.0));
            assert!(winner.fast_score.is_none() && winner.quality_score.is_none());
            assert!((winner.score - (1.0_f32 / 61.0 + 1.0 / 62.0)).abs() < 1e-7);
            assert_eq!(winner.metadata.as_deref().unwrap()["snapshot"], 1);
            assert_eq!(lexical.latest_generation.load(Ordering::SeqCst), 2);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.hydrated_docs.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn eager_metadata_is_reused_and_hash_only_results_stay_nonsemantic() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(false);
            let mut lexical = Lexical::new();
            lexical.batch_mode = BatchMode::Eager;
            let results = index
                .search_hybrid_text(&cx, &provider, &lexical, "query", 3)
                .await
                .unwrap();
            assert_eq!(results[0].metadata.as_deref().unwrap()["snapshot"], 1);
            assert_eq!(results[1].metadata.as_deref().unwrap()["snapshot"], 1);
            let vector_only = &results[2];
            assert_eq!(vector_only.doc_id, "gamma");
            assert_eq!(vector_only.source, ScoreSource::HashControl);
            assert_eq!(vector_only.index, Some(2));
            assert!(vector_only.metadata.is_none());
            assert!(vector_only.fast_score.is_none());
            assert!(vector_only.lexical_score.is_none());
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn foreign_hydration_context_is_an_error_not_a_new_search() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(false);
            let mut lexical = Lexical::new();
            lexical.batch_mode = BatchMode::Foreign;
            let error = index
                .search_hybrid_text(&cx, &provider, &lexical, "query", 2)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                if field == "native_ann.hydration"));
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.hydrated_docs.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn hydration_cancellation_does_not_return_partially_hydrated_success() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(false);
            let mut lexical = Lexical::new();
            lexical.hydration_reply = HydrationReply::Cancelled;
            let error = index
                .search_hybrid_text(&cx, &provider, &lexical, "query", 2)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { ref phase, .. }
                if phase == "test.hydration"));
            assert_eq!(lexical.hydrated_docs.load(Ordering::SeqCst), 2);
            assert_eq!(lexical.hydration_drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert_eq!(index.live_count(), 3);
        });
    }

    #[test]
    fn dropping_pending_hydration_releases_the_pin_and_owned_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let provider = Provider::new(false);
            let mut lexical = Lexical::new();
            lexical.hydration_reply = HydrationReply::Pending;
            let mut future =
                Box::pin(index.search_hybrid_text(&cx, &provider, &lexical, "query", 2));
            let waker = Waker::noop();
            assert!(matches!(
                future.as_mut().poll(&mut Context::from_waker(waker)),
                Poll::Pending
            ));
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(future);
            assert_eq!(lexical.hydration_drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert_eq!(index.live_count(), 3);
        });
    }

    #[test]
    fn text_noops_skip_hydration_and_empty_native_owners_hydrate_lexical_hits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, true);
            let provider = Provider::new(true);
            let lexical = Lexical::new();
            assert!(
                index
                    .search_hybrid_text(&cx, &provider, &lexical, "query", 0)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 0);
            let results = index
                .search_hybrid_text(&cx, &provider, &lexical, "query", 2)
                .await
                .unwrap();
            assert_eq!(results.len(), 2);
            assert!(results.iter().all(|result| {
                result.source == ScoreSource::Lexical
                    && result.index.is_none()
                    && result.fast_score.is_none()
                    && result.metadata.is_some()
            }));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.hydrated_docs.load(Ordering::SeqCst), 2);
        });
    }

    #[test]
    fn duplicate_lexical_ids_keep_the_winning_candidates_metadata_arc() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = index(&cx, false);
            let lexical = Lexical::new();
            let query = BoundQueryEmbedding::new(vec![1.0, 0.0], identity()).unwrap();
            let original = Arc::new(serde_json::json!({"winner": "first"}));
            let mut first = lexical_result("beta", 10.0);
            first.metadata = Some(Arc::clone(&original));
            let mut duplicate = lexical_result("beta", 1.0);
            duplicate.metadata = Some(Arc::new(serde_json::json!({"winner": "second"})));
            let batch = LexicalCandidateBatch::eager(vec![first, duplicate]);
            let hits = index
                .fuse_candidates(&cx, &query, &batch, 1, None, &RrfConfig::default())
                .unwrap();
            let results = hydrate_winners(&cx, &lexical, hits, &batch).await.unwrap();
            assert_eq!(results[0].doc_id, "beta");
            assert!(Arc::ptr_eq(
                results[0].metadata.as_ref().unwrap(),
                &original
            ));
            assert_eq!(lexical.hydrations.load(Ordering::SeqCst), 0);
        });
    }
}
