//! Lazy native refinement over the same retained fast and lexical candidates.

use frankensearch_core::LexicalCandidateBatch;
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_fusion::{RrfConfig, rrf_fuse_for_vector_lane};

use super::refinement::{admit_pair, checked_lexical, refined_winners};
use super::{NativeAnnIndex, checkpoint, hydrate_winners, invalid, join_sources};
use crate::{Cx, Embedder, LexicalRead, ScoredResult, SearchError, SearchResult, VectorHit};

impl NativeAnnIndex {
    /// Prepare native progressive search without starting inference or retrieval.
    ///
    /// The first `next_phase` retrieves this fast owner plus lexical candidates.
    /// Only a second call starts independent retrieval from the optional quality
    /// owner. Stopping after initial results starts no quality inference. With
    /// no quality arm, the stream yields exactly one initial phase. Direct
    /// quality-only retrieval remains [`Self::search_hybrid_quality_text`].
    ///
    /// Generation, producer, space-kind and document-ID joins use the same
    /// admission boundary as [`Self::search_hybrid_refined_text`]. Refinement
    /// uses its exact blend, RRF, raw-score and owner-specific row-index policy,
    /// but reuses the original fast candidate pool and pinned lexical snapshot.
    /// This does not prove a composite lexical/vector generation or change the
    /// default searcher. Graph work remains synchronous on the polling executor.
    ///
    /// # Errors
    ///
    /// Refuses invalid admission or overflow of the three-times candidate
    /// budget. Every configured identity is checked even for zero-k requests.
    pub fn search_hybrid_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        fast_embedder: &'a dyn Embedder,
        quality: Option<(&'a Self, &'a dyn Embedder)>,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let search = NativeProgressiveSearch {
            fast: self,
            cx,
            fast_embedder,
            quality,
            lexical,
            text,
            k,
            budget: k.checked_mul(3).ok_or_else(|| {
                invalid(
                    "progressive.candidate_budget",
                    "overflow",
                    "three times the requested result count must fit usize",
                )
            })?,
            state: State::Initial,
        };
        search.admit()?;
        Ok(search)
    }
}

/// Actual returned candidate counts, not graph visits or corpus-coverage claims.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativePhaseCandidates {
    /// Candidates retrieved from the retained fast owner.
    pub fast: usize,
    /// Candidates independently retrieved from the retained quality owner.
    pub quality: usize,
    /// Candidates in the original pinned lexical scoring batch.
    pub lexical: usize,
}

/// A requested phase of caller-owned native progressive retrieval.
#[derive(Debug)]
pub enum NativeSearchPhase {
    /// Fast-plus-lexical results ready without any quality inference.
    Initial {
        /// Hydrated winners in fused rank order.
        results: Vec<ScoredResult>,
        /// Observed candidate counts; quality is zero in this phase.
        candidates: NativePhaseCandidates,
    },
    /// Winners after independent quality retrieval and candidate union.
    Refined {
        /// Hydrated winners; these may include documents outside the fast pool.
        results: Vec<ScoredResult>,
        /// Counts from retained fast/lexical pools and the new quality pool.
        candidates: NativePhaseCandidates,
    },
    /// A non-cancellation refinement failure; initial results remain valid.
    RefinementFailed {
        /// Original initial results, unchanged by the attempted refinement.
        initial_results: Vec<ScoredResult>,
        /// Failure in admission, quality retrieval, fusion, or hydration.
        error: SearchError,
    },
}

struct PendingRefinement {
    fast: Vec<VectorHit>,
    batch: LexicalCandidateBatch,
    initial_results: Vec<ScoredResult>,
}

enum State {
    Initial,
    Refine(PendingRefinement),
    Done,
}

/// Lazy search retaining the exact fast pool and lexical scoring snapshot.
///
/// Dropping this stream after initial results starts no quality work. Dropping
/// a *polled, still-pending* `next_phase` future permanently finishes the stream
/// and releases its in-flight work and snapshot pin. Subsequent calls return
/// `None` rather than retrying inference, traversal, or lexical retrieval.
/// Dropping an unpolled future starts no phase and leaves the stream unchanged.
///
/// There are no detached tasks or hidden runtimes. Cancellation checkpoints
/// surround synchronous graph work; no preemptive interruption or implicit
/// quality timeout is claimed. A provider cancellation propagates as `Err`,
/// never as a successful refinement or a `RefinementFailed` degradation.
///
/// Construct with [`NativeAnnIndex::search_hybrid_progressive`].
pub struct NativeProgressiveSearch<'a> {
    fast: &'a NativeAnnIndex,
    cx: &'a Cx,
    fast_embedder: &'a dyn Embedder,
    quality: Option<(&'a NativeAnnIndex, &'a dyn Embedder)>,
    lexical: &'a dyn LexicalRead,
    text: &'a str,
    k: usize,
    budget: usize,
    state: State,
}

impl NativeProgressiveSearch<'_> {
    /// Whether every phase finished or a polled phase was abandoned.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// Compute one requested phase, then finish permanently after the last.
    ///
    /// # Errors
    ///
    /// Initial failures and any cancellation propagate directly. After initial
    /// success, other refinement failures produce `RefinementFailed` carrying
    /// the original results. Every error leaves this stream finished.
    pub async fn next_phase(&mut self) -> SearchResult<Option<NativeSearchPhase>> {
        // Own the phase before the first suspension. Cancellation by dropping
        // this future cannot leave a runnable copy of the work in the stream.
        match std::mem::replace(&mut self.state, State::Done) {
            State::Done => Ok(None),
            State::Initial => self.initial().await.map(Some),
            State::Refine(pending) => self.refine(pending).await.map(Some),
        }
    }

    fn admit(&self) -> SearchResult<bool> {
        if let Some(quality) = self.quality {
            return admit_pair(self.cx, self.fast, self.fast_embedder, quality);
        }
        checkpoint(self.cx, "native_ann.progressive_admission")?;
        let identity = self.fast_embedder.identity()?;
        self.fast.admit_identity(identity)?;
        // Classify the same admitted reference, not a second provider call.
        Ok(matches!(
            identity.space.kind,
            EmbeddingSpaceKindV1::HashControl
        ))
    }

    async fn initial(&mut self) -> SearchResult<NativeSearchPhase> {
        let is_hash = self.admit()?;
        if self.k == 0 {
            return Ok(NativeSearchPhase::Initial {
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
        let fused = rrf_fuse_for_vector_lane(
            batch.results(),
            &fast,
            self.k,
            0,
            &RrfConfig::default(),
            is_hash,
        );
        let results = hydrate_winners(self.cx, self.lexical, fused, &batch).await?;
        checkpoint(self.cx, "native_ann.progressive_initial_complete")?;
        if self.quality.is_some() {
            self.state = State::Refine(PendingRefinement {
                fast,
                batch,
                initial_results: results.clone(),
            });
        }
        Ok(NativeSearchPhase::Initial { results, candidates })
    }

    async fn refine(&self, pending: PendingRefinement) -> SearchResult<NativeSearchPhase> {
        match self.refined_results(&pending).await {
            Ok((results, candidates)) => Ok(NativeSearchPhase::Refined { results, candidates }),
            Err(error) => {
                checkpoint(self.cx, "native_ann.progressive_failure")?;
                if matches!(error, SearchError::Cancelled { .. }) {
                    return Err(error);
                }
                Ok(NativeSearchPhase::RefinementFailed {
                    initial_results: pending.initial_results,
                    error,
                })
            }
        }
    }

    async fn refined_results(
        &self,
        pending: &PendingRefinement,
    ) -> SearchResult<(Vec<ScoredResult>, NativePhaseCandidates)> {
        let is_hash = self.admit()?;
        let (quality, embedder) = self.quality.ok_or_else(|| {
            invalid(
                "progressive.quality",
                "missing",
                "refinement requires its admitted arm",
            )
        })?;
        let quality_hits = quality
            .search_text(self.cx, embedder, self.text, self.budget, None)
            .await?;
        let candidates = NativePhaseCandidates {
            fast: pending.fast.len(),
            quality: quality_hits.len(),
            lexical: pending.batch.results().len(),
        };
        let results = refined_winners(
            self.cx,
            self.lexical,
            &pending.batch,
            &pending.fast,
            &quality_hits,
            self.k,
            is_hash,
        )
        .await?;
        checkpoint(self.cx, "native_ann.progressive_refined_complete")?;
        Ok((results, candidates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use frankensearch_core::LexicalHydrationContext;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

    use crate::ScoreSource;

    #[derive(Clone, Copy)]
    enum Reply {
        Correct,
        Failed,
        Cancelled,
        Pending,
        ForeignProducer,
    }

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        foreign_identity: EmbeddingIdentityBundleV1,
        advertise_foreign: AtomicBool,
        calls: AtomicUsize,
        drops: AtomicUsize,
        reply: Reply,
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Self {
            // Synthetic semantic-shaped fixture, not evidence of a real model.
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
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
            let mut foreign_identity = identity.clone();
            foreign_identity.producer.backend = "foreign-backend".to_owned();
            Self {
                identity,
                foreign_identity,
                advertise_foreign: AtomicBool::new(false),
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
                reply: Reply::Correct,
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
            Box::pin(async move {
                let mut values = vec![0.0; self.dimension()];
                values[0] = 1.0;
                Ok(values)
            })
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                let identity = match self.reply {
                    Reply::Correct => self.identity.clone(),
                    Reply::ForeignProducer => self.foreign_identity.clone(),
                    Reply::Failed => {
                        return Err(invalid("test.provider", "failed", "provider failed"));
                    }
                    Reply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "test.provider".to_owned(),
                            reason: "provider cancelled".to_owned(),
                        });
                    }
                    Reply::Pending => return std::future::pending().await,
                };
                Ok(IdentityBoundEmbedding {
                    values: self.embed(cx, text).await?,
                    identity,
                })
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.advertise_foreign.load(Ordering::SeqCst) {
                &self.foreign_identity
            } else {
                &self.identity
            })
        }

        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }

        fn model_name(&self) -> &str {
            self.id()
        }

        fn dimension(&self) -> usize {
            usize::try_from(self.identity.space.dimension).unwrap()
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }
    }

    struct Lexical {
        ids: Vec<&'static str>,
        calls: AtomicUsize,
        hydration_calls: AtomicUsize,
        hydration_drops: AtomicUsize,
        snapshot: Arc<usize>,
        current_generation: AtomicUsize,
        pending_search: bool,
        refinement_hydration: Reply,
    }

    impl Lexical {
        fn new(ids: &[&'static str]) -> Self {
            Self {
                ids: ids.to_vec(),
                calls: AtomicUsize::new(0),
                hydration_calls: AtomicUsize::new(0),
                hydration_drops: AtomicUsize::new(0),
                snapshot: Arc::new(1),
                current_generation: AtomicUsize::new(1),
                pending_search: false,
                refinement_hydration: Reply::Correct,
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
                if self.pending_search {
                    std::future::pending::<()>().await;
                }
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
                let results = self.search(cx, text, limit).await?;
                self.current_generation.store(2, Ordering::SeqCst);
                Ok(LexicalCandidateBatch::deferred(
                    results,
                    LexicalHydrationContext::new(
                        "native-progressive-test",
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
                let call = self.hydration_calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.hydration_drops);
                let pin = context
                    .and_then(LexicalHydrationContext::downcast_ref::<Arc<usize>>)
                    .filter(|pin| Arc::ptr_eq(pin, &self.snapshot))
                    .ok_or_else(|| invalid("test.snapshot", "foreign", "wrong snapshot"))?;
                for result in results {
                    if result.lexical_score.is_some() {
                        result.metadata = Some(Arc::new(serde_json::json!({"generation": **pin})));
                    }
                }
                if call > 0 {
                    match self.refinement_hydration {
                        Reply::Pending => return std::future::pending().await,
                        Reply::Failed => {
                            return Err(invalid("test.hydration", "failed", "partial hydration"));
                        }
                        Reply::Cancelled => {
                            return Err(SearchError::Cancelled {
                                phase: "test.hydration".to_owned(),
                                reason: "cancelled after writing partial metadata".to_owned(),
                            });
                        }
                        Reply::Correct | Reply::ForeignProducer => {}
                    }
                }
                Ok(())
            })
        }

        fn doc_count(&self) -> SearchResult<usize> {
            Ok(self.ids.len())
        }
    }

    fn generation() -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(1, [0x61; 16]).unwrap()
    }

    fn native_index(
        cx: &Cx,
        provider: &Provider,
        generation: ArtifactGenerationIdentityV1,
        rows: &[(&str, &[f32])],
    ) -> NativeAnnIndex {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier.fsvi");
        let mut bundle = provider.identity.clone();
        bundle.storage.format = "fsvi-v2".to_owned();
        bundle.storage.quantization = QuantizationFormat::F32;
        bundle.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(generation, bundle.freeze().unwrap()).unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(id, values) in rows {
            writer.write_record(id, values).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
    }

    fn fast_index(cx: &Cx, provider: &Provider) -> NativeAnnIndex {
        native_index(
            cx,
            provider,
            generation(),
            &[
                ("z-fast", &[1.0, 0.0]),
                ("middle", &[0.8, 0.6]),
                ("near", &[0.6, 0.8]),
                ("a-quality", &[0.0, 1.0]),
            ],
        )
    }

    fn quality_index(cx: &Cx, provider: &Provider) -> NativeAnnIndex {
        native_index(
            cx,
            provider,
            generation(),
            &[
                ("a-quality", &[1.0, 0.0, 0.0]),
                ("near", &[0.0, 1.0, 0.0]),
                ("middle", &[0.0, 0.0, 1.0]),
                ("z-fast", &[-1.0, 0.0, 0.0]),
            ],
        )
    }

    #[test]
    fn lazy_refinement_retrieves_a_winner_outside_the_fast_candidate_pool() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            let NativeSearchPhase::Initial { results, candidates } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("expected initial phase");
            };
            assert_eq!(results[0].doc_id, "z-fast");
            assert_eq!(candidates, NativePhaseCandidates { fast: 3, quality: 0, lexical: 0 });
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            let State::Refine(pending) = &stream.state else { panic!("missing retained pool") };
            assert!(pending.fast.iter().all(|hit| hit.doc_id != "a-quality"));
            let NativeSearchPhase::Refined { results, candidates } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("expected independent refinement");
            };
            assert_eq!(results[0].doc_id, "a-quality");
            assert_eq!(results[0].quality_score, Some(1.0));
            assert!(results[0].fast_score.is_none());
            assert_eq!(results[0].source, ScoreSource::SemanticQuality);
            assert_eq!(results[0].index, Some(0)); // quality owner's row, not the fast row 3
            assert_eq!(candidates, NativePhaseCandidates { fast: 3, quality: 3, lexical: 0 });
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn stopping_after_initial_starts_no_quality_work_and_releases_the_pin() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Pending;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            assert!(matches!(stream.next_phase().await.unwrap(), Some(NativeSearchPhase::Initial { .. })));
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(stream);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn refinement_failure_carries_the_unchanged_initial_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Failed;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("expected initial");
            };
            let expected = serde_json::to_value(results).unwrap();
            let NativeSearchPhase::RefinementFailed { initial_results, error } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("provider failure must be explicit");
            };
            assert_eq!(serde_json::to_value(initial_results).unwrap(), expected);
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.test.provider"));
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn provider_cancellation_is_terminal_not_successful_degradation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Cancelled;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(matches!(stream.next_phase().await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.provider"));
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn dropping_a_polled_refinement_future_fuses_the_stream_and_drops_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Pending;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(future);
            assert!(stream.is_finished());
            assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn fast_only_and_zero_k_have_a_single_initial_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let mut empty = index.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 0).unwrap();
            let NativeSearchPhase::Initial { results, candidates } = empty.next_phase().await.unwrap().unwrap() else {
                panic!("zero k yields an empty initial phase");
            };
            assert!(results.is_empty());
            assert_eq!(candidates, NativePhaseCandidates::default());
            assert!(empty.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
            let mut stream = index.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1).unwrap();
            let NativeSearchPhase::Initial { results, .. } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("fast-only phase");
            };
            assert_eq!(results[0].doc_id, "z-fast");
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn refinement_reuses_the_scoring_snapshot_and_matches_the_existing_refined_api() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["a-quality", "z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            lexical.current_generation.store(42, Ordering::SeqCst);
            let NativeSearchPhase::Refined { results, .. } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("expected refined");
            };
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.current_generation.load(Ordering::SeqCst), 42);
            assert!(results.iter().all(|hit| hit.metadata.as_deref().unwrap()["generation"] == 1));
            let eager = fast_index.search_hybrid_refined_text(&cx, &fast, (&quality_index, &quality), &lexical, "query", 2).await.unwrap();
            assert_eq!(serde_json::to_value(results).unwrap(), serde_json::to_value(eager).unwrap());
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn changed_quality_identity_is_rejected_before_refinement_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            quality.advertise_foreign.store(true, Ordering::SeqCst);
            let NativeSearchPhase::RefinementFailed { initial_results, error } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("changed producer must not be admitted");
            };
            assert_eq!(initial_results[0].doc_id, "z-fast");
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "query_embedding.native_ann.producer_conformance"));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(stream.is_finished());
        });
    }

    #[test]
    fn dropping_a_polled_initial_future_does_not_restart_either_source() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut fast = Provider::new("fast", 2);
            fast.reply = Reply::Pending;
            let index = fast_index(&cx, &fast);
            let mut lexical = Lexical::new(&[]);
            lexical.pending_search = true;
            let mut stream = index.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1).unwrap();
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            drop(future);
            assert_eq!(fast.drops.load(Ordering::SeqCst), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn pair_admission_and_budget_overflow_refuse_without_any_provider_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let foreign_index = native_index(&cx, &quality, ArtifactGenerationIdentityV1::new(1, [0x62; 16]).unwrap(), &[]);
            let lexical = Lexical::new(&[]);
            for k in [0, 1] {
                assert!(matches!(
                    fast_index.search_hybrid_progressive(&cx, &fast, Some((&foreign_index, &quality)), &lexical, "query", k),
                    Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.tier_generation"
                ));
            }
            assert!(matches!(
                fast_index.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", usize::MAX),
                Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.progressive.candidate_budget"
            ));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn dropping_pending_refinement_hydration_releases_the_original_pin() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let mut lexical = Lexical::new(&["a-quality", "z-fast"]);
            lexical.refinement_hydration = Reply::Pending;
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
            assert_eq!(lexical.hydration_calls.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(future);
            assert_eq!(lexical.hydration_drops.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn a_substituted_actual_quality_response_cannot_produce_a_refined_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::ForeignProducer;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let NativeSearchPhase::RefinementFailed { initial_results, error } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("substituted producer must fail");
            };
            assert_eq!(initial_results[0].doc_id, "z-fast");
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "query_embedding.native_ann.producer_conformance"));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn cancellation_after_partial_refinement_hydration_is_not_partial_success() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let mut lexical = Lexical::new(&["a-quality", "z-fast"]);
            lexical.refinement_hydration = Reply::Cancelled;
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(matches!(stream.next_phase().await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.hydration"));
            assert_eq!(lexical.hydration_calls.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn empty_fast_owner_does_not_prevent_direct_quality_retrieval() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut fast = Provider::new("fast", 2);
            fast.reply = Reply::Pending;
            let quality = Provider::new("quality", 3);
            let fast_index = native_index(&cx, &fast, generation(), &[]);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(&cx, &fast, Some((&quality_index, &quality)), &lexical, "query", 1)
                .unwrap();
            let NativeSearchPhase::Initial { results, candidates } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("initial phase");
            };
            assert!(results.is_empty());
            assert_eq!(candidates.fast, 0);
            let NativeSearchPhase::Refined { results, .. } = stream.next_phase().await.unwrap().unwrap() else {
                panic!("quality must retrieve independently");
            };
            assert_eq!(results[0].doc_id, "a-quality");
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn dropping_an_unpolled_phase_does_not_consume_its_work_or_snapshot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            drop(stream.next_phase());
            assert!(!stream.is_finished());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            drop(stream.next_phase());
            assert!(!stream.is_finished());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Refined { .. })
            ));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }
}
