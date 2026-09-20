//! Deadline-bounded delivery of the existing native progressive phase sequence.

use super::NativeSearchDeadline;
use crate::native_ann::builder::NativeBuiltHybridIndex;
use crate::native_ann::{NativeProgressiveSearch, NativeSearchPhase};
use crate::{Cx, Reranker, ScoredResult, SearchError, SearchResult};

#[derive(Clone, Copy)]
enum Stage {
    Initial,
    Refine,
    Rerank,
    Done,
}

/// Progressive retrieval under one caller-clock deadline for the whole query.
///
/// Uses the existing native retrieval, hydration and whole-envelope reranker;
/// only deadline handling and delivery state live here. The same retained build
/// is borrowed through every phase. For live serving, hold one
/// `NativeHybridSnapshot` and call its `index().progressive_before(...)`.
///
/// Expiry before Initial returns a typed timeout error. Expiry during refinement
/// yields `RefinementFailed` with the exact displayed Initial page; expiry during
/// reranking yields `RerankFailed` with the preceding displayed page. An expired
/// total budget ends the stream, including any unstarted reranker. No model is
/// retried and no newer generation is selected. Ordinary non-timeout refinement
/// failures retain the existing optional-rerank behavior while budget remains.
///
/// The deadline starts when its value is created, not on the next phase request.
/// Consumer delays between phases consume that same budget. No phase is polled
/// after observed expiry. A running poll's synchronous/immediately-ready substeps
/// cannot be preempted; late completion is checked when that poll returns. Caller
/// cancellation is always an error, never a successful timeout/degradation phase.
///
/// Dropping an unpolled `next_phase` future leaves the stream unchanged. Dropping
/// a polled pending future permanently finishes it and releases provider work,
/// its timer registration and the lexical scoring pin. Stopping after Initial
/// never starts quality or reranking.
pub struct NativeDeadlineProgressiveSearch<'a> {
    cx: &'a Cx,
    stream: Option<NativeProgressiveSearch<'a>>,
    deadline: NativeSearchDeadline,
    has_quality: bool,
    stage: Stage,
    previous: Vec<ScoredResult>,
}

impl NativeBuiltHybridIndex {
    /// Prepare progressive retrieval with a fixed total query deadline.
    ///
    /// All configured identities are still admitted by the original constructor,
    /// even for zero-k or an already-expired deadline. No inference starts here.
    ///
    /// # Errors
    /// Propagates progressive identity, topology, budget and cancellation errors.
    pub fn progressive_before<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        deadline: NativeSearchDeadline,
    ) -> SearchResult<NativeDeadlineProgressiveSearch<'a>> {
        Ok(NativeDeadlineProgressiveSearch {
            cx,
            stream: Some(self.progressive(cx, text, k)?),
            deadline,
            has_quality: self.vectors().quality().is_some(),
            stage: Stage::Initial,
            previous: Vec::new(),
        })
    }

    /// Prepare bounded progressive retrieval with lazy retained-source reranking.
    ///
    /// The same deadline includes text resolution, reranker readiness, inference
    /// and response admission. Expired quality work does not start a reranker on
    /// a fresh budget. No mutable/current source-text resolver is accepted.
    ///
    /// # Errors
    /// Propagates progressive and rerank configuration/identity admission errors.
    pub fn progressive_with_reranker_before<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        reranker: &'a dyn Reranker,
        window: usize,
        deadline: NativeSearchDeadline,
    ) -> SearchResult<NativeDeadlineProgressiveSearch<'a>> {
        Ok(NativeDeadlineProgressiveSearch {
            cx,
            stream: Some(self.progressive_with_reranker(cx, text, k, reranker, window)?),
            deadline,
            has_quality: self.vectors().quality().is_some(),
            stage: Stage::Initial,
            previous: Vec::new(),
        })
    }
}

impl NativeDeadlineProgressiveSearch<'_> {
    /// Whether retrieval finished, timed out, was cancelled or was abandoned.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.stream.is_none()
    }

    /// Execute the next phase with the original deadline and generation pin.
    ///
    /// Timeout after a displayed phase carries that page in an explicit failure
    /// phase. Such a timeout is terminal: the next call returns `None`.
    ///
    /// # Errors
    /// Initial failures, initial timeout and any caller/provider cancellation
    /// propagate as errors. An error permanently finishes the stream.
    pub async fn next_phase(&mut self) -> SearchResult<Option<NativeSearchPhase>> {
        // Move every in-flight resource out before the first suspension. A
        // dropped future cannot leave an inner stream or replayable phase here.
        let Some(mut stream) = self.stream.take() else {
            return Ok(None);
        };
        let stage = std::mem::replace(&mut self.stage, Stage::Done);
        let previous = std::mem::take(&mut self.previous);
        let response = self.deadline.run(self.cx, stream.next_phase()).await;
        match response {
            Err(error @ SearchError::SearchTimeout { .. }) => match stage {
                Stage::Initial | Stage::Done => Err(error),
                Stage::Refine => Ok(Some(NativeSearchPhase::RefinementFailed {
                    initial_results: previous,
                    error,
                })),
                Stage::Rerank => Ok(Some(NativeSearchPhase::RerankFailed {
                    previous_results: previous,
                    error,
                })),
            },
            Err(error) => Err(error),
            Ok(None) => Ok(None),
            Ok(Some(phase)) => {
                if !stream.is_finished() {
                    let (next, page) = match &phase {
                        NativeSearchPhase::Initial { results, .. } => (
                            if self.has_quality {
                                Stage::Refine
                            } else {
                                Stage::Rerank
                            },
                            results,
                        ),
                        NativeSearchPhase::Refined { results, .. } => (Stage::Rerank, results),
                        NativeSearchPhase::RefinementFailed {
                            initial_results, ..
                        } => (Stage::Rerank, initial_results),
                        NativeSearchPhase::Reranked { .. }
                        | NativeSearchPhase::RerankFailed { .. } => {
                            return Ok(Some(phase));
                        }
                    };
                    self.previous.clone_from(page);
                    self.stage = next;
                    self.stream = Some(stream);
                }
                Ok(Some(phase))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Waker};
    use std::time::Duration;

    use asupersync::time::{TimerDriverHandle, VirtualClock};
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        EmbeddingSpaceKindV1,
    };
    use frankensearch_core::traits::{
        IdentityBoundEmbedding, ModelCategory, RerankDocument, RerankScore,
    };
    use frankensearch_index::native_hnsw::HnswParams;

    use crate::native_ann::builder::live::NativeLiveHybridIndex;
    use crate::native_ann::builder::{
        NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder,
    };
    use crate::{Embedder, IndexableDocument, SearchFuture};

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
        drops: AtomicUsize,
        // 0 = ready, 1 = pending, 2 = ordinary failure, 3 = provider cancellation.
        mode: AtomicUsize,
    }

    struct DropCount<'a>(&'a AtomicUsize);
    impl Drop for DropCount<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Self {
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
            // Synthetic semantic-shaped identity, not real-model quality evidence.
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
            Self {
                identity,
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
                mode: AtomicUsize::new(0),
            }
        }
        fn values(&self, text: &str) -> Vec<f32> {
            let mut values = vec![0.0; self.dimension()];
            values[usize::from(text == "vertical")] = 1.0;
            values
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                match self.mode.load(Ordering::SeqCst) {
                    1 => std::future::pending().await,
                    2 => Err(crate::native_ann::invalid(
                        "test.provider",
                        "failed",
                        "controlled failure",
                    )),
                    3 => Err(SearchError::Cancelled {
                        phase: "test.provider".to_owned(),
                        reason: "cancelled".to_owned(),
                    }),
                    _ => Ok(self.values(text)),
                }
            })
        }
        fn embed_batch_bound<'a>(
            &'a self,
            _cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                Ok(texts
                    .iter()
                    .map(|text| IdentityBoundEmbedding {
                        values: self.values(text),
                        identity: self.identity.clone(),
                    })
                    .collect())
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
            usize::try_from(self.identity.space.dimension).unwrap()
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }
    }

    #[derive(Default)]
    struct RerankProbe {
        calls: AtomicUsize,
        drops: AtomicUsize,
        pending: bool,
    }
    impl Reranker for RerankProbe {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                if self.pending {
                    return std::future::pending().await;
                }
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(rank, document)| RerankScore {
                        doc_id: document.doc_id.clone(),
                        score: if document.doc_id == "b" { 1.0 } else { 0.0 },
                        original_rank: rank,
                        raw_logit: None,
                    })
                    .collect())
            })
        }
        fn id(&self) -> &'static str {
            "deadline-reranker"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
    }

    fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(sequence, [0x53; 16]).unwrap()
    }

    async fn fixture(
        cx: &Cx,
        with_quality: bool,
    ) -> (
        tempfile::TempDir,
        Arc<Provider>,
        Arc<Provider>,
        NativeBuiltHybridIndex,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2));
        let quality = Arc::new(Provider::new("quality", 3));
        let mut builder =
            NativeIndexBuilder::new(directory.path().join("old"), generation(1), fast.clone())
                .unwrap()
                .with_fast_storage(
                    NativeBuildPrecision::F16,
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 7,
                    },
                )
                .add_documents([
                    IndexableDocument::new("z-fast", "horizontal").with_metadata("version", "old"),
                    IndexableDocument::new("b", "vertical").with_metadata("version", "old"),
                    IndexableDocument::new("x", "horizontal").with_metadata("version", "old"),
                ]);
        if with_quality {
            builder = builder.with_quality_embedder(quality.clone()).unwrap();
        }
        let index = builder.build_hybrid(cx).await.unwrap();
        (directory, fast, quality, index)
    }

    fn clock(cx: &Cx) -> (Arc<VirtualClock>, NativeSearchDeadline) {
        let clock = Arc::new(VirtualClock::new());
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let deadline =
            NativeSearchDeadline::with_timer(cx, timer, Duration::from_millis(10)).unwrap();
        (clock, deadline)
    }

    fn assert_same_page(expected: &[ScoredResult], actual: &[ScoredResult]) {
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        for (before, after) in expected.iter().zip(actual) {
            assert_eq!(before.score.to_bits(), after.score.to_bits());
            assert_eq!(
                before.fast_score.map(f32::to_bits),
                after.fast_score.map(f32::to_bits)
            );
            assert_eq!(
                before.quality_score.map(f32::to_bits),
                after.quality_score.map(f32::to_bits)
            );
            assert_eq!(
                before.lexical_score.map(f32::to_bits),
                after.lexical_score.map(f32::to_bits)
            );
            assert_eq!(
                before.rerank_score.map(f32::to_bits),
                after.rerank_score.map(f32::to_bits)
            );
            if let Some(metadata) = &before.metadata {
                assert!(Arc::ptr_eq(metadata, after.metadata.as_ref().unwrap()));
            }
        }
    }

    #[test]
    fn quality_expiry_preserves_initial_and_never_starts_the_queued_reranker() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = fixture(&cx, true).await;
            let (clock, deadline) = clock(&cx);
            let timer = deadline.timer.clone();
            let reranker = RerankProbe::default();
            let mut stream = index
                .progressive_with_reranker_before(&cx, "vertical", 1, &reranker, 3, deadline)
                .unwrap();
            let Some(NativeSearchPhase::Initial { results, .. }) =
                stream.next_phase().await.unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(results[0].doc_id, "b");
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            quality.mode.store(1, Ordering::SeqCst);
            let mut pending = Box::pin(stream.next_phase());
            assert!(
                pending
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(timer.pending_count(), 1);
            clock.advance(10_000_000);
            let _ = timer.process_timers();
            let Some(NativeSearchPhase::RefinementFailed {
                initial_results,
                error,
            }) = pending.await.unwrap()
            else {
                panic!("timeout phase")
            };
            assert!(matches!(
                error,
                SearchError::SearchTimeout {
                    budget_ms: 10,
                    elapsed_ms: 10
                }
            ));
            assert_same_page(&results, &initial_results);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
            assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
            assert!(timer.is_empty() && stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn rerank_expiry_carries_the_exact_refined_page() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, _, _, index) = fixture(&cx, true).await;
            let (clock, deadline) = clock(&cx);
            let timer = deadline.timer.clone();
            let reranker = RerankProbe {
                pending: true,
                ..RerankProbe::default()
            };
            let mut stream = index
                .progressive_with_reranker_before(&cx, "vertical", 1, &reranker, 3, deadline)
                .unwrap();
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            let Some(NativeSearchPhase::Refined { results, .. }) =
                stream.next_phase().await.unwrap()
            else {
                panic!("refined")
            };
            assert!(results[0].fast_score.is_some() && results[0].quality_score.is_some());
            let mut pending = Box::pin(stream.next_phase());
            assert!(
                pending
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            clock.advance(10_000_000);
            let _ = timer.process_timers();
            let Some(NativeSearchPhase::RerankFailed {
                previous_results,
                error,
            }) = pending.await.unwrap()
            else {
                panic!("rerank timeout")
            };
            assert!(matches!(error, SearchError::SearchTimeout { .. }));
            assert_same_page(&results, &previous_results);
            assert_eq!(reranker.calls.load(Ordering::SeqCst), 1);
            assert_eq!(reranker.drops.load(Ordering::SeqCst), 1);
            assert!(timer.is_empty() && stream.is_finished());
        });
    }

    #[test]
    fn consumer_delay_uses_the_same_budget_and_expired_phases_do_not_start() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for after_initial in [false, true] {
                let (_directory, fast, quality, index) = fixture(&cx, true).await;
                let (clock, deadline) = clock(&cx);
                let mut stream = index
                    .progressive_before(&cx, "vertical", 1, deadline)
                    .unwrap();
                if after_initial {
                    assert!(stream.next_phase().await.unwrap().is_some());
                }
                clock.advance(11_000_000);
                let phase = stream.next_phase().await;
                if after_initial {
                    assert!(matches!(
                        phase,
                        Ok(Some(NativeSearchPhase::RefinementFailed {
                            error: SearchError::SearchTimeout { .. },
                            ..
                        }))
                    ));
                } else {
                    assert!(matches!(phase, Err(SearchError::SearchTimeout { .. })));
                }
                assert_eq!(
                    fast.calls.load(Ordering::SeqCst),
                    usize::from(after_initial)
                );
                assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                assert!(stream.is_finished() && stream.next_phase().await.unwrap().is_none());
            }
        });
    }

    #[test]
    fn unpolled_calls_are_lazy_and_dropped_pending_calls_are_terminal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = fixture(&cx, true).await;
            let (_, deadline) = clock(&cx);
            let timer = deadline.timer.clone();
            let mut stream = index
                .progressive_before(&cx, "vertical", 1, deadline)
                .unwrap();
            drop(stream.next_phase());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert!(!stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_some());
            drop(stream.next_phase());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            quality.mode.store(1, Ordering::SeqCst);
            let mut pending = Box::pin(stream.next_phase());
            assert!(
                pending
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            drop(pending);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
            assert!(timer.is_empty() && stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn cancellation_never_becomes_timeout_degradation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for provider_cancel in [false, true] {
                let (_directory, _, quality, index) = fixture(&cx, true).await;
                let (clock, deadline) = clock(&cx);
                let timer = deadline.timer.clone();
                let mut stream = index
                    .progressive_before(&cx, "vertical", 1, deadline)
                    .unwrap();
                assert!(stream.next_phase().await.unwrap().is_some());
                quality
                    .mode
                    .store(if provider_cancel { 3 } else { 1 }, Ordering::SeqCst);
                let mut pending = Box::pin(stream.next_phase());
                if !provider_cancel {
                    assert!(
                        pending
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop()))
                            .is_pending()
                    );
                    cx.set_cancel_requested(true);
                    clock.advance(10_000_000);
                }
                assert!(matches!(pending.await, Err(SearchError::Cancelled { .. })));
                cx.set_cancel_requested(false);
                assert!(timer.is_empty() && stream.is_finished());
                assert!(stream.next_phase().await.unwrap().is_none());
            }
        });
    }

    #[test]
    fn normal_failure_and_fast_only_topologies_keep_the_existing_phase_sequence() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for with_quality in [false, true] {
                let (_directory, fast, quality, index) = fixture(&cx, with_quality).await;
                let (_, deadline) = clock(&cx);
                let reranker = RerankProbe::default();
                let mut stream = index
                    .progressive_with_reranker_before(&cx, "vertical", 1, &reranker, 3, deadline)
                    .unwrap();
                assert!(matches!(
                    stream.next_phase().await.unwrap(),
                    Some(NativeSearchPhase::Initial { .. })
                ));
                if with_quality {
                    quality.mode.store(2, Ordering::SeqCst);
                    assert!(matches!(
                        stream.next_phase().await.unwrap(),
                        Some(NativeSearchPhase::RefinementFailed { .. })
                    ));
                }
                let Some(NativeSearchPhase::Reranked { results, .. }) =
                    stream.next_phase().await.unwrap()
                else {
                    panic!("reranked")
                };
                assert_eq!(results[0].doc_id, "b");
                assert_eq!(results[0].rerank_score, Some(1.0));
                assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    quality.calls.load(Ordering::SeqCst),
                    usize::from(with_quality)
                );
                assert_eq!(reranker.calls.load(Ordering::SeqCst), 1);
                assert!(stream.is_finished());
            }
        });
    }

    #[test]
    fn live_replacement_does_not_change_an_expiring_queries_page_or_owners() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (directory, _, quality, index) = fixture(&cx, true).await;
            let live = NativeLiveHybridIndex::new(&cx, index).unwrap();
            let pin = live.snapshot(&cx).await.unwrap();
            let (clock, deadline) = clock(&cx);
            let mut stream = pin
                .index()
                .progressive_before(&cx, "vertical", 1, deadline)
                .unwrap();
            let Some(NativeSearchPhase::Initial { results, .. }) =
                stream.next_phase().await.unwrap()
            else {
                panic!("initial")
            };
            quality.mode.store(1, Ordering::SeqCst);
            let mut pending = Box::pin(stream.next_phase());
            assert!(
                pending
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            let candidate = pin
                .begin_update(&cx, directory.path().join("next"), generation(2))
                .unwrap()
                .upsert_document(
                    IndexableDocument::new("b", "horizontal").with_metadata("version", "new"),
                )
                .build(&cx)
                .await
                .unwrap();
            live.install(&cx, &candidate).await.unwrap();
            clock.advance(10_000_000);
            let Some(NativeSearchPhase::RefinementFailed {
                initial_results, ..
            }) = pending.await.unwrap()
            else {
                panic!("timeout")
            };
            assert_same_page(&results, &initial_results);
            let row = usize::try_from(initial_results[0].index.unwrap()).unwrap();
            assert_eq!(
                pin.index()
                    .vectors()
                    .fast()
                    .index()
                    .owner
                    .doc_id_at(row)
                    .unwrap(),
                "b"
            );
            assert_eq!(
                pin.index().vectors().document("b").unwrap().content,
                "vertical"
            );
            quality.mode.store(0, Ordering::SeqCst);
            let current = live.search_refined(&cx, "horizontal", 3).await.unwrap();
            assert_eq!(current.snapshot.generation(), generation(2));
            assert_eq!(
                current
                    .snapshot
                    .index()
                    .vectors()
                    .document("b")
                    .unwrap()
                    .metadata["version"],
                "new"
            );
            assert!(stream.is_finished());
        });
    }

    #[test]
    fn initial_timeout_releases_pending_inference_without_fabricating_a_page() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = fixture(&cx, true).await;
            let (clock, deadline) = clock(&cx);
            let timer = deadline.timer.clone();
            fast.mode.store(1, Ordering::SeqCst);
            let mut stream = index
                .progressive_before(&cx, "vertical", 1, deadline)
                .unwrap();
            let mut pending = Box::pin(stream.next_phase());
            assert!(
                pending
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            clock.advance(10_000_000);
            let _ = timer.process_timers();
            assert!(matches!(
                pending.await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(fast.drops.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(timer.is_empty() && stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            fast.mode.store(0, Ordering::SeqCst);
            assert_eq!(
                index.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn bounded_and_unbounded_complete_phases_agree_without_expiry() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for with_quality in [false, true] {
                let (_directory, _, _, index) = fixture(&cx, with_quality).await;
                for k in [0, 1] {
                    let (_, deadline) = clock(&cx);
                    let mut bounded = index
                        .progressive_before(&cx, "vertical", k, deadline)
                        .unwrap();
                    let mut reference = index.progressive(&cx, "vertical", k).unwrap();
                    loop {
                        match (
                            reference.next_phase().await.unwrap(),
                            bounded.next_phase().await.unwrap(),
                        ) {
                            (
                                Some(NativeSearchPhase::Initial {
                                    results: a,
                                    candidates: ca,
                                }),
                                Some(NativeSearchPhase::Initial {
                                    results: b,
                                    candidates: cb,
                                }),
                            )
                            | (
                                Some(NativeSearchPhase::Refined {
                                    results: a,
                                    candidates: ca,
                                }),
                                Some(NativeSearchPhase::Refined {
                                    results: b,
                                    candidates: cb,
                                }),
                            ) => {
                                assert_eq!(
                                    serde_json::to_value(a).unwrap(),
                                    serde_json::to_value(b).unwrap()
                                );
                                assert_eq!(ca, cb);
                            }
                            (None, None) => break,
                            _ => panic!("bounded and unbounded phase sequence differs"),
                        }
                    }
                    assert!(bounded.is_finished());
                }
            }
        });
    }
}
