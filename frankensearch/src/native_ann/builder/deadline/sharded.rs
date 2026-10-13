//! One caller-clock deadline for complete sharded native hybrid queries.
//!
//! These entry points run the existing whole-inventory query. The deadline
//! includes live snapshot acquisition, all configured vector partitions, the
//! global lexical arm and hydration. It is never restarted per shard or tier.
//! A required failure or expiry returns an error, not a successful partial page.
//! Both physical row coordinates remain attached to their originating snapshot.
//!
//! The caller services the timer and owns scheduling. No runtime, task, model,
//! graph, source lookup or generation is substituted. Synchronous work cannot
//! be preempted; the shared deadline runner refuses late completion when its
//! poll returns. Cancellation is an error and does not cancel unrelated queries.

use super::NativeSearchDeadline;
use crate::native_ann::builder::sharded::NativeBuiltShardedHybridIndex;
use crate::native_ann::builder::sharded::live::{
    NativeLiveShardedHybridIndex, NativeShardedResults,
};
use crate::native_ann::{
    NativeShardedProgressiveSearch, NativeShardedResult, NativeShardedSearchPhase, checkpoint,
    invalid,
};
use crate::{Cx, Reranker, SearchError, SearchResult};

#[derive(Clone, Copy)]
enum Stage {
    Initial,
    Refine,
    Rerank,
    Done,
}

/// Sharded progressive delivery under one retained caller-clock deadline.
///
/// All phases borrow the same complete source/vector/global-lexical cohort.
/// For a live index, acquire one snapshot and call its `index()` methods; keep
/// that snapshot while interpreting both physical coordinates. A consumer delay
/// consumes the original budget rather than granting the next stage a new one.
///
/// Expiry before Initial returns an error. Expiry during quality or reranking
/// returns the corresponding failure phase with the exact last displayed page,
/// including metadata and both shard rows, and permanently finishes the stream.
/// Ordinary non-timeout refinement failures may still run the configured
/// reranker while budget remains. Cancellation is always an error.
///
/// Dropping an unpolled phase future changes nothing. Dropping a polled pending
/// phase finishes the sequence, releasing owned provider work and its timer;
/// no later call can replay the abandoned phase. Already-running synchronous
/// work is not preempted. No model, task, runtime or new generation is selected.
pub struct NativeDeadlineShardedProgressiveSearch<'a> {
    cx: &'a Cx,
    stream: Option<NativeShardedProgressiveSearch<'a>>,
    deadline: NativeSearchDeadline,
    has_quality: bool,
    stage: Stage,
    previous: Vec<NativeShardedResult>,
}

impl NativeDeadlineShardedProgressiveSearch<'_> {
    /// Whether the sequence completed, failed, timed out or was abandoned.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.stream.is_none()
    }

    /// Request a phase without resetting the total deadline or its snapshot.
    ///
    /// A timeout failure page is terminal, even when reranking was configured.
    /// It contains the displayed page, never the undisplayed candidate window.
    ///
    /// # Errors
    /// Initial errors/timeouts and every cancellation propagate as errors. Any
    /// such error permanently finishes the sequence. A finished stream is `None`.
    pub async fn next_phase(&mut self) -> SearchResult<Option<NativeShardedSearchPhase>> {
        let Some(mut stream) = self.stream.take() else {
            return Ok(None);
        };
        // Nothing replayable remains in self while the provider is suspended.
        let stage = std::mem::replace(&mut self.stage, Stage::Done);
        let previous = std::mem::take(&mut self.previous);
        let response = self.deadline.run(self.cx, stream.next_phase()).await;
        match response {
            Err(error @ SearchError::SearchTimeout { .. }) => match stage {
                Stage::Initial | Stage::Done => Err(error),
                Stage::Refine => Ok(Some(NativeShardedSearchPhase::RefinementFailed {
                    initial_results: previous,
                    error,
                })),
                Stage::Rerank => Ok(Some(NativeShardedSearchPhase::RerankFailed {
                    previous_results: previous,
                    error,
                })),
            },
            Err(error) => Err(error),
            Ok(None) => Ok(None),
            Ok(Some(phase)) => {
                if !stream.is_finished() {
                    let (next, page) = match &phase {
                        NativeShardedSearchPhase::Initial { results, .. } => (
                            if self.has_quality {
                                Stage::Refine
                            } else {
                                Stage::Rerank
                            },
                            results,
                        ),
                        NativeShardedSearchPhase::Refined { results, .. } => {
                            (Stage::Rerank, results)
                        }
                        NativeShardedSearchPhase::RefinementFailed {
                            initial_results, ..
                        } => (Stage::Rerank, initial_results),
                        NativeShardedSearchPhase::Reranked { .. }
                        | NativeShardedSearchPhase::RerankFailed { .. } => {
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

    /// Collect an unstarted bounded query, including any retained-source reranker.
    ///
    /// A required refinement or configured rerank failure returns its error,
    /// never a successful earlier page. Use `next_phase` to consume explicit
    /// degradation phases instead. This does not construct a different query.
    ///
    /// # Errors
    /// Refuses started/finished sequences before provider work. Propagates every
    /// phase error, timeout and cancellation under the original deadline.
    pub async fn collect(mut self) -> SearchResult<Vec<NativeShardedResult>> {
        checkpoint(self.cx, "native_ann.deadline.sharded.collect")?;
        if !matches!(self.stage, Stage::Initial) || self.stream.is_none() {
            return Err(invalid(
                "deadline.sharded.collect",
                "started",
                "collect requires an unstarted sharded phase sequence",
            ));
        }
        // One timer covers collection itself as well as every underlying phase.
        // There is no displayed fallback page to clone or preserve in this path.
        self.deadline
            .run(self.cx, async {
                let Some(mut stream) = self.stream.take() else {
                    return Err(invalid(
                        "deadline.sharded.collect",
                        "finished",
                        "collect requires an owned unstarted query",
                    ));
                };
                let mut page = Vec::new();
                while let Some(phase) = stream.next_phase().await? {
                    match phase {
                        NativeShardedSearchPhase::Initial { results, .. }
                        | NativeShardedSearchPhase::Refined { results, .. }
                        | NativeShardedSearchPhase::Reranked { results, .. } => page = results,
                        NativeShardedSearchPhase::RefinementFailed { error, .. }
                        | NativeShardedSearchPhase::RerankFailed { error, .. } => {
                            return Err(error);
                        }
                    }
                }
                Ok(page)
            })
            .await
    }
}

impl NativeBuiltShardedHybridIndex {
    /// Prepare lazy sharded retrieval with a fixed total query deadline.
    ///
    /// The ordinary constructor admits all configured identities before this
    /// returns, including zero-k and expired requests. No inference starts here.
    /// For live queries, borrow the built index from one retained snapshot.
    ///
    /// # Errors
    /// Propagates progressive identity, topology, budget and cancellation errors.
    pub fn progressive_before<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        deadline: NativeSearchDeadline,
    ) -> SearchResult<NativeDeadlineShardedProgressiveSearch<'a>> {
        Ok(NativeDeadlineShardedProgressiveSearch {
            cx,
            stream: Some(self.progressive(cx, text, k)?),
            deadline,
            has_quality: self.vectors().quality().is_some(),
            stage: Stage::Initial,
            previous: Vec::new(),
        })
    }

    /// Prepare lazy refinement and retained-source reranking under one deadline.
    ///
    /// A window larger than k remains undisplayed until reranking. Timing out
    /// preserves only the last displayed page, without losing its shard rows.
    /// No fresh budget is granted to source lookup, model readiness or reranking.
    ///
    /// # Errors
    /// Propagates the ordinary progressive/reranker configuration and identity
    /// errors before any inference. Expiry is observed by phase execution.
    pub fn progressive_with_reranker_before<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        reranker: &'a dyn Reranker,
        window: usize,
        deadline: NativeSearchDeadline,
    ) -> SearchResult<NativeDeadlineShardedProgressiveSearch<'a>> {
        Ok(NativeDeadlineShardedProgressiveSearch {
            cx,
            stream: Some(self.progressive_with_reranker(cx, text, k, reranker, window)?),
            deadline,
            has_quality: self.vectors().quality().is_some(),
            stage: Stage::Initial,
            previous: Vec::new(),
        })
    }

    /// Fast-plus-global-lexical retrieval under one absolute deadline.
    ///
    /// Retain this cohort while interpreting either returned row coordinate.
    /// An already-expired request starts no provider or shard work.
    ///
    /// # Errors
    /// Returns the original query failure, cancellation or a total-budget timeout.
    pub async fn search_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        deadline.run(cx, self.search(cx, text, k)).await
    }

    /// Independently retrieve both configured tiers within the original budget.
    ///
    /// A missing quality tier retains the cohort's declared fast-only behavior.
    /// A configured quality failure never becomes a fast-only successful page.
    /// No failing or late partition is silently omitted from the global ranking.
    ///
    /// # Errors
    /// Returns any required query failure, cancellation or total-budget timeout.
    pub async fn search_refined_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        deadline.run(cx, self.search_refined(cx, text, k)).await
    }

    /// Quality-primary retrieval under the same deadline, without fast inference.
    ///
    /// # Errors
    /// Before expiry, an absent quality tier is refused even for zero k. Expired
    /// requests are never polled; cancellation takes precedence over expiry.
    pub async fn search_quality_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<Vec<NativeShardedResult>> {
        deadline.run(cx, self.search_quality(cx, text, k)).await
    }
}

impl NativeLiveShardedHybridIndex {
    /// Bound snapshot acquisition and fast-plus-lexical search by one deadline.
    ///
    /// The page retains its actual source/vector/lexical snapshot. No timeout
    /// retries against a newer generation, and selection is never modified.
    ///
    /// # Errors
    /// Returns acquisition/query failure, cancellation or total-budget timeout.
    pub async fn search_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<NativeShardedResults> {
        deadline.run(cx, self.search(cx, text, k)).await
    }

    /// Bound snapshot acquisition and both retrieval tiers by the original expiry.
    ///
    /// Every shard and global lexical read uses one pin. The selection lock is
    /// released before inference, and a failed query cannot alter the live head.
    ///
    /// # Errors
    /// Returns acquisition, required-phase, cancellation or timeout errors.
    pub async fn search_refined_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<NativeShardedResults> {
        deadline.run(cx, self.search_refined(cx, text, k)).await
    }

    /// Quality-primary live search with snapshot waiting charged to the same budget.
    ///
    /// # Errors
    /// Returns acquisition, absent-quality, query, cancellation or timeout errors.
    pub async fn search_quality_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<NativeShardedResults> {
        deadline.run(cx, self.search_quality(cx, text, k)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Waker};
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;
    use asupersync::time::{TimerDriverHandle, VirtualClock};
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        EmbeddingSpaceKindV1,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, RerankDocument, RerankScore};
    use frankensearch_index::native_hnsw::HnswParams;

    use crate::native_ann::builder::{
        NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder,
    };
    use crate::{Embedder, IndexableDocument, ModelCategory, SearchError, SearchFuture};

    // Synthetic identified models exercise real FSVI, native graphs and Quill;
    // they do not establish real-model quality, recall or performance.
    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
        drops: AtomicUsize,
        // 0 ready, 1 pending, 2 ordinary error, 3 provider cancellation.
        mode: AtomicUsize,
    }

    struct CallDrop<'a>(&'a AtomicUsize);
    impl Drop for CallDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Self {
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

        fn reset(&self) {
            self.calls.store(0, Ordering::SeqCst);
            self.drops.store(0, Ordering::SeqCst);
            self.mode.store(0, Ordering::SeqCst);
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _call = CallDrop(&self.drops);
                match self.mode.load(Ordering::SeqCst) {
                    1 => std::future::pending().await,
                    2 => Err(crate::native_ann::invalid(
                        "deadline.sharded.test",
                        "failed",
                        "controlled required provider failure",
                    )),
                    3 => Err(SearchError::Cancelled {
                        phase: "deadline.sharded.test".to_owned(),
                        reason: "provider cancelled".to_owned(),
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

        fn bound_batch_is_native(&self) -> bool {
            true
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
            ModelCategory::StaticEmbedder
        }
    }

    fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(sequence, [19; 16]).unwrap()
    }

    async fn build(
        cx: &Cx,
        path: &Path,
        fast: &Arc<Provider>,
        quality: Option<&Arc<Provider>>,
        graph: bool,
    ) -> NativeBuiltShardedHybridIndex {
        let retrieval = if graph {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 41,
            }
        } else {
            NativeBuildRetrieval::Exact
        };
        let mut builder = NativeIndexBuilder::new(path, generation(1), fast.clone())
            .unwrap()
            .with_fast_storage(NativeBuildPrecision::F32, retrieval);
        if let Some(quality) = quality {
            builder = builder
                .with_quality_embedder(quality.clone())
                .unwrap()
                .with_quality_storage(NativeBuildPrecision::F32, retrieval)
                .unwrap();
        }
        builder
            .add_documents([
                IndexableDocument::new("a", "horizontal").with_title("first"),
                IndexableDocument::new("b", "vertical").with_metadata("language", "β"),
                IndexableDocument::new("c", "horizontal horizontal"),
            ])
            .build_sharded_hybrid(cx, 1)
            .await
            .unwrap()
    }

    fn deadline(cx: &Cx, millis: u64) -> (Arc<VirtualClock>, NativeSearchDeadline) {
        let clock = Arc::new(VirtualClock::new());
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let deadline =
            NativeSearchDeadline::with_timer(cx, timer, Duration::from_millis(millis)).unwrap();
        (clock, deadline)
    }

    fn same_page(left: &[NativeShardedResult], right: &[NativeShardedResult]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(
                serde_json::to_value(&left.result).unwrap(),
                serde_json::to_value(&right.result).unwrap()
            );
            assert_eq!(left.result.score.to_bits(), right.result.score.to_bits());
            assert_eq!(left.fast_row, right.fast_row);
            assert_eq!(left.quality_row, right.quality_row);
            assert!(left.result.index.is_none());
        }
    }

    #[test]
    fn eager_deadlines_preserve_global_rankings_rows_and_primary_tier_dispatch() {
        run_test_with_cx(|cx| async move {
            for graph in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let fast = Arc::new(Provider::new("fast", 2));
                let quality = Arc::new(Provider::new("quality", 3));
                let index = build(
                    &cx,
                    &root.path().join("index"),
                    &fast,
                    Some(&quality),
                    graph,
                )
                .await;
                assert_eq!(index.vectors().partitions().len(), 3);
                let (_, budget) = deadline(&cx, 100);
                let expected = index.search(&cx, "vertical", 3).await.unwrap();
                fast.reset();
                quality.reset();
                let actual = index
                    .search_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap();
                same_page(&actual, &expected);
                assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                assert!(
                    actual
                        .iter()
                        .any(|hit| hit.fast_row.is_some_and(|row| row.shard > 0))
                );

                let expected = index.search_quality(&cx, "vertical", 3).await.unwrap();
                fast.reset();
                quality.reset();
                let actual = index
                    .search_quality_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap();
                same_page(&actual, &expected);
                assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert!(actual.iter().any(|hit| hit.quality_row.is_some()));
                assert!(actual.iter().all(|hit| hit.fast_row.is_none()));

                let expected = index.search_refined(&cx, "vertical", 3).await.unwrap();
                fast.reset();
                quality.reset();
                let actual = index
                    .search_refined_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap();
                same_page(&actual, &expected);
                assert!(
                    actual
                        .iter()
                        .any(|hit| hit.fast_row.is_some() && hit.quality_row.is_some())
                );
                assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert!(budget.timer.is_empty());

                let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
                let fast_page = live
                    .search_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap();
                let quality_page = live
                    .search_quality_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap();
                let refined_page = live
                    .search_refined_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap();
                same_page(&refined_page.results, &expected);
                for page in [fast_page, quality_page, refined_page] {
                    assert_eq!(page.snapshot.generation(), generation(1));
                    assert_eq!(
                        page.snapshot
                            .index()
                            .vectors()
                            .document("b")
                            .unwrap()
                            .content,
                        "vertical"
                    );
                }
            }
        });
    }

    #[test]
    fn expired_live_queries_start_no_work_and_cancellation_wins() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(
                &cx,
                &root.path().join("index"),
                &fast,
                Some(&quality),
                false,
            )
            .await;
            let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
            let (_, budget) = deadline(&cx, 0);
            assert!(matches!(
                live.search_before(&cx, "vertical", 3, &budget).await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert!(matches!(
                live.search_quality_before(&cx, "vertical", 3, &budget)
                    .await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert!(matches!(
                live.search_refined_before(&cx, "vertical", 3, &budget)
                    .await,
                Err(SearchError::SearchTimeout { .. })
            ));
            cx.set_cancel_requested(true);
            assert!(matches!(
                live.search_refined_before(&cx, "vertical", 3, &budget)
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(budget.timer.is_empty());
            assert_eq!(
                live.snapshot(&cx).await.unwrap().generation(),
                generation(1)
            );
        });
    }

    #[test]
    fn eager_required_failures_and_missing_quality_are_not_successful_partial_pages() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &root.path().join("full"), &fast, Some(&quality), false).await;
            let (_, budget) = deadline(&cx, 100);
            for provider in [&fast, &quality] {
                for mode in [2, 3] {
                    provider.mode.store(mode, Ordering::SeqCst);
                    let error = index
                        .search_refined_before(&cx, "vertical", 3, &budget)
                        .await
                        .unwrap_err();
                    if mode == 2 {
                        assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                            if field == "native_ann.deadline.sharded.test"));
                    } else {
                        assert!(matches!(error, SearchError::Cancelled { ref phase, .. }
                            if phase == "deadline.sharded.test"));
                    }
                }
                provider.reset();
            }
            let fast_only = build(&cx, &root.path().join("fast-only"), &fast, None, false).await;
            for k in [0, 3] {
                assert!(
                    fast_only
                        .search_quality_before(&cx, "vertical", k, &budget)
                        .await
                        .is_err()
                );
            }
            same_page(
                &fast_only
                    .search_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap(),
                &fast_only
                    .search_refined_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap(),
            );
            assert!(budget.timer.is_empty());
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn pending_sharded_work_releases_on_timeout_drop_and_cancellation() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(
                &cx,
                &root.path().join("index"),
                &fast,
                Some(&quality),
                false,
            )
            .await;
            for provider in [&fast, &quality] {
                for ending in 0..3 {
                    fast.reset();
                    quality.reset();
                    provider.mode.store(1, Ordering::SeqCst);
                    let (clock, budget) = deadline(&cx, 10);
                    let mut pending =
                        Box::pin(index.search_refined_before(&cx, "vertical", 3, &budget));
                    assert!(
                        pending
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop()))
                            .is_pending()
                    );
                    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(budget.timer.pending_count(), 1);
                    match ending {
                        0 => {
                            clock.advance(10_000_000);
                            let _ = budget.timer.process_timers();
                            assert!(matches!(
                                pending.await,
                                Err(SearchError::SearchTimeout {
                                    elapsed_ms: 10,
                                    budget_ms: 10
                                })
                            ));
                        }
                        1 => drop(pending),
                        _ => {
                            cx.set_cancel_requested(true);
                            clock.advance(10_000_000);
                            assert!(matches!(pending.await, Err(SearchError::Cancelled { .. })));
                            cx.set_cancel_requested(false);
                        }
                    }
                    assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
                    assert!(budget.timer.is_empty());
                }
            }
            fast.reset();
            quality.reset();
            let (_, budget) = deadline(&cx, 10);
            assert!(
                !index
                    .search_refined_before(&cx, "vertical", 3, &budget)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(!cx.is_cancel_requested());
        });
    }

    #[derive(Default)]
    struct RerankProbe {
        calls: AtomicUsize,
        drops: AtomicUsize,
        mode: AtomicUsize,
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
                let _call = CallDrop(&self.drops);
                match self.mode.load(Ordering::SeqCst) {
                    1 => return std::future::pending().await,
                    2 => {
                        return Err(invalid(
                            "deadline.sharded.rerank_test",
                            "failed",
                            "controlled rerank failure",
                        ));
                    }
                    3 => {
                        return Err(SearchError::Cancelled {
                            phase: "deadline.sharded.rerank_test".to_owned(),
                            reason: "provider cancelled".to_owned(),
                        });
                    }
                    _ => {}
                }
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(original_rank, document)| RerankScore {
                        doc_id: document.doc_id.clone(),
                        score: if document.doc_id == "c" { 1.0 } else { 0.0 },
                        original_rank,
                        raw_logit: None,
                    })
                    .collect())
            })
        }

        fn id(&self) -> &'static str {
            "sharded-deadline-reranker"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
    }

    fn same_phase(left: &NativeShardedSearchPhase, right: &NativeShardedSearchPhase) {
        assert_eq!(std::mem::discriminant(left), std::mem::discriminant(right));
        match (left, right) {
            (
                NativeShardedSearchPhase::Initial {
                    results: left,
                    candidates: lc,
                }
                | NativeShardedSearchPhase::Refined {
                    results: left,
                    candidates: lc,
                },
                NativeShardedSearchPhase::Initial {
                    results: right,
                    candidates: rc,
                }
                | NativeShardedSearchPhase::Refined {
                    results: right,
                    candidates: rc,
                },
            ) => {
                same_page(left, right);
                assert_eq!(
                    (lc.fast, lc.quality, lc.lexical),
                    (rc.fast, rc.quality, rc.lexical)
                );
            }
            (
                NativeShardedSearchPhase::Reranked {
                    results: left,
                    candidates: lc,
                    evaluated: le,
                },
                NativeShardedSearchPhase::Reranked {
                    results: right,
                    candidates: rc,
                    evaluated: re,
                },
            ) => {
                same_page(left, right);
                assert_eq!(
                    (lc.fast, lc.quality, lc.lexical),
                    (rc.fast, rc.quality, rc.lexical)
                );
                assert_eq!(le, re);
            }
            _ => panic!("expected matching successful phases"),
        }
    }

    #[test]
    fn bounded_progressive_delivery_matches_every_original_phase_and_rerank_row() {
        run_test_with_cx(|cx| async move {
            for graph in [false, true] {
                for has_quality in [false, true] {
                    let root = tempfile::tempdir().unwrap();
                    let fast = Arc::new(Provider::new("fast", 2));
                    let quality = Arc::new(Provider::new("quality", 3));
                    let index = build(
                        &cx,
                        &root.path().join("index"),
                        &fast,
                        has_quality.then_some(&quality),
                        graph,
                    )
                    .await;
                    let reranker = RerankProbe::default();
                    let (_, budget) = deadline(&cx, 10);
                    let mut expected = index
                        .progressive_with_reranker(&cx, "vertical", 1, &reranker, 3)
                        .unwrap();
                    let mut actual = index
                        .progressive_with_reranker_before(
                            &cx,
                            "vertical",
                            1,
                            &reranker,
                            3,
                            budget.clone(),
                        )
                        .unwrap();
                    assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                    let mut phases = 0;
                    while let Some(reference) = expected.next_phase().await.unwrap() {
                        let phase = actual.next_phase().await.unwrap().unwrap();
                        same_phase(&phase, &reference);
                        phases += 1;
                        if phases == 1 {
                            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                            assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
                        }
                        if let NativeShardedSearchPhase::Reranked {
                            results, evaluated, ..
                        } = phase
                        {
                            assert_eq!(evaluated, 3);
                            assert_eq!(results.len(), 1);
                            assert_eq!(results[0].result.doc_id, "c");
                            assert_eq!(results[0].fast_row.unwrap().shard, 2);
                            assert_eq!(results[0].quality_row.is_some(), has_quality);
                        }
                    }
                    assert_eq!(phases, if has_quality { 3 } else { 2 });
                    assert!(actual.is_finished());
                    assert!(actual.next_phase().await.unwrap().is_none());
                    assert!(budget.timer.is_empty());
                }
            }
        });
    }

    #[test]
    fn quality_expiry_preserves_only_the_displayed_page_and_never_starts_reranking() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &root.path().join("index"), &fast, Some(&quality), true).await;
            for start_quality in [false, true] {
                fast.reset();
                quality.reset();
                let reranker = RerankProbe::default();
                let (clock, budget) = deadline(&cx, 10);
                let mut query = index
                    .progressive_with_reranker_before(
                        &cx,
                        "vertical",
                        1,
                        &reranker,
                        3,
                        budget.clone(),
                    )
                    .unwrap();
                let NativeShardedSearchPhase::Initial {
                    results: initial, ..
                } = query.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial");
                };
                assert_eq!(initial.len(), 1);
                assert!(initial[0].fast_row.is_some());
                assert!(initial[0].quality_row.is_none());
                clock.advance(6_000_000);
                quality.mode.store(1, Ordering::SeqCst);
                let mut pending = Box::pin(query.next_phase());
                if start_quality {
                    assert!(
                        pending
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop()))
                            .is_pending()
                    );
                    assert_eq!(budget.timer.pending_count(), 1);
                }
                clock.advance(4_000_000);
                let _ = budget.timer.process_timers();
                let NativeShardedSearchPhase::RefinementFailed {
                    initial_results,
                    error,
                } = pending.await.unwrap().unwrap()
                else {
                    panic!("terminal refinement failure");
                };
                same_page(&initial_results, &initial);
                assert!(matches!(
                    error,
                    SearchError::SearchTimeout {
                        elapsed_ms: 10,
                        budget_ms: 10
                    }
                ));
                assert_eq!(
                    quality.calls.load(Ordering::SeqCst),
                    usize::from(start_quality)
                );
                assert_eq!(
                    quality.drops.load(Ordering::SeqCst),
                    usize::from(start_quality)
                );
                assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
                assert!(query.is_finished());
                assert!(query.next_phase().await.unwrap().is_none());
                assert!(budget.timer.is_empty());
                assert!(!cx.is_cancel_requested());
            }
        });
    }

    #[test]
    fn rerank_expiry_keeps_the_previous_sharded_page_without_a_fresh_budget() {
        run_test_with_cx(|cx| async move {
            for has_quality in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let fast = Arc::new(Provider::new("fast", 2));
                let quality = Arc::new(Provider::new("quality", 3));
                let index = build(
                    &cx,
                    &root.path().join("index"),
                    &fast,
                    has_quality.then_some(&quality),
                    true,
                )
                .await;
                let reranker = RerankProbe::default();
                reranker.mode.store(1, Ordering::SeqCst);
                let (clock, budget) = deadline(&cx, 10);
                let mut query = index
                    .progressive_with_reranker_before(
                        &cx,
                        "vertical",
                        1,
                        &reranker,
                        3,
                        budget.clone(),
                    )
                    .unwrap();
                let NativeShardedSearchPhase::Initial {
                    results: mut previous,
                    ..
                } = query.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial");
                };
                clock.advance(5_000_000);
                if has_quality {
                    let NativeShardedSearchPhase::Refined { results, .. } =
                        query.next_phase().await.unwrap().unwrap()
                    else {
                        panic!("refined");
                    };
                    previous = results;
                    assert!(previous[0].quality_row.is_some());
                }
                clock.advance(3_000_000);
                let mut pending = Box::pin(query.next_phase());
                assert!(
                    pending
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
                assert_eq!(reranker.calls.load(Ordering::SeqCst), 1);
                clock.advance(2_000_000);
                let _ = budget.timer.process_timers();
                let NativeShardedSearchPhase::RerankFailed {
                    previous_results,
                    error,
                } = pending.await.unwrap().unwrap()
                else {
                    panic!("terminal rerank failure");
                };
                same_page(&previous_results, &previous);
                assert!(matches!(
                    error,
                    SearchError::SearchTimeout {
                        elapsed_ms: 10,
                        budget_ms: 10
                    }
                ));
                assert_eq!(reranker.drops.load(Ordering::SeqCst), 1);
                assert!(query.is_finished());
                assert!(query.next_phase().await.unwrap().is_none());
                assert!(budget.timer.is_empty());
            }
        });
    }

    #[test]
    fn progressive_cancellation_and_abandoned_phase_are_terminal_not_degraded() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(
                &cx,
                &root.path().join("index"),
                &fast,
                Some(&quality),
                false,
            )
            .await;
            for ending in 0..3 {
                quality.reset();
                let (clock, budget) = deadline(&cx, 10);
                let mut query = index
                    .progressive_before(&cx, "vertical", 1, budget.clone())
                    .unwrap();
                drop(query.next_phase()); // unpolled: does not consume Initial.
                assert!(!query.is_finished());
                assert!(matches!(
                    query.next_phase().await.unwrap(),
                    Some(NativeShardedSearchPhase::Initial { .. })
                ));
                quality
                    .mode
                    .store(if ending == 2 { 3 } else { 1 }, Ordering::SeqCst);
                let mut pending = Box::pin(query.next_phase());
                if ending == 2 {
                    assert!(
                        matches!(pending.await, Err(SearchError::Cancelled { ref phase, .. })
                        if phase == "deadline.sharded.test")
                    );
                } else {
                    assert!(
                        pending
                            .as_mut()
                            .poll(&mut Context::from_waker(Waker::noop()))
                            .is_pending()
                    );
                    assert_eq!(budget.timer.pending_count(), 1);
                    if ending == 0 {
                        drop(pending);
                    } else {
                        cx.set_cancel_requested(true);
                        clock.advance(10_000_000);
                        assert!(matches!(pending.await, Err(SearchError::Cancelled { .. })));
                        cx.set_cancel_requested(false);
                    }
                }
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
                assert!(query.is_finished());
                assert!(query.next_phase().await.unwrap().is_none());
                assert!(budget.timer.is_empty());
            }
        });
    }

    #[test]
    fn collection_is_all_required_phases_or_error_and_rejects_started_streams() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(
                &cx,
                &root.path().join("index"),
                &fast,
                Some(&quality),
                false,
            )
            .await;
            let reranker = RerankProbe::default();
            let (_, budget) = deadline(&cx, 10);
            let mut started = index
                .progressive_before(&cx, "vertical", 1, budget.clone())
                .unwrap();
            let _ = started.next_phase().await.unwrap();
            assert!(
                matches!(started.collect().await, Err(SearchError::InvalidConfig { ref field, .. })
                    if field == "native_ann.deadline.sharded.collect")
            );
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);

            quality.mode.store(2, Ordering::SeqCst);
            let make_query = || {
                index
                    .progressive_with_reranker_before(
                        &cx,
                        "vertical",
                        1,
                        &reranker,
                        3,
                        budget.clone(),
                    )
                    .unwrap()
            };
            assert!(
                matches!(make_query().collect().await, Err(SearchError::InvalidConfig { ref field, .. })
                    if field == "native_ann.deadline.sharded.test")
            );
            assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
            // Explicit phase consumers may still rerank after an ordinary
            // refinement failure; the strict collector above must not do so.
            let mut streamed = make_query();
            let _ = streamed.next_phase().await.unwrap();
            assert!(matches!(
                streamed.next_phase().await.unwrap(),
                Some(NativeShardedSearchPhase::RefinementFailed { .. })
            ));
            assert!(matches!(
                streamed.next_phase().await.unwrap(),
                Some(NativeShardedSearchPhase::Reranked { .. })
            ));
            quality.reset();
            reranker.mode.store(2, Ordering::SeqCst);
            assert!(
                matches!(make_query().collect().await, Err(SearchError::InvalidConfig { ref field, .. })
                    if field == "native_ann.deadline.sharded.rerank_test")
            );
            reranker.mode.store(0, Ordering::SeqCst);
            let page = make_query().collect().await.unwrap();
            assert_eq!(page.len(), 1);
            assert_eq!(page[0].result.doc_id, "c");
            assert_eq!(page[0].fast_row.unwrap().shard, 2);
            assert_eq!(page[0].quality_row.unwrap().shard, 2);
            assert!(budget.timer.is_empty());

            fast.reset();
            quality.reset();
            let (_, expired) = deadline(&cx, 0);
            let mut query = index
                .progressive_before(&cx, "vertical", 1, expired.clone())
                .unwrap();
            assert!(matches!(
                query.next_phase().await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert!(query.is_finished());
            assert!(query.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(expired.timer.is_empty());
        });
    }

    #[test]
    fn collection_timeout_drops_pending_quality_or_reranking_without_returning_a_page() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(
                &cx,
                &root.path().join("index"),
                &fast,
                Some(&quality),
                false,
            )
            .await;
            for pending_reranker in [false, true] {
                quality.reset();
                let reranker = RerankProbe::default();
                if pending_reranker {
                    reranker.mode.store(1, Ordering::SeqCst);
                } else {
                    quality.mode.store(1, Ordering::SeqCst);
                }
                let (clock, budget) = deadline(&cx, 10);
                let query = index
                    .progressive_with_reranker_before(
                        &cx,
                        "vertical",
                        1,
                        &reranker,
                        3,
                        budget.clone(),
                    )
                    .unwrap();
                let mut pending = Box::pin(query.collect());
                assert!(
                    pending
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    reranker.calls.load(Ordering::SeqCst),
                    usize::from(pending_reranker)
                );
                assert_eq!(budget.timer.pending_count(), 1);
                clock.advance(10_000_000);
                let _ = budget.timer.process_timers();
                assert!(matches!(
                    pending.await,
                    Err(SearchError::SearchTimeout {
                        elapsed_ms: 10,
                        budget_ms: 10
                    })
                ));
                assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
                assert_eq!(
                    reranker.drops.load(Ordering::SeqCst),
                    usize::from(pending_reranker)
                );
                assert!(budget.timer.is_empty());
                assert!(!cx.is_cancel_requested());
            }
        });
    }

    #[test]
    fn progressive_timeout_keeps_old_row_provenance_across_live_repartition() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &root.path().join("old"), &fast, Some(&quality), true).await;
            let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
            let old = live.snapshot(&cx).await.unwrap();
            let (clock, budget) = deadline(&cx, 10);
            let mut query = old
                .index()
                .progressive_before(&cx, "vertical", 1, budget.clone())
                .unwrap();
            let NativeShardedSearchPhase::Initial {
                results: initial, ..
            } = query.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial");
            };
            assert_eq!(initial[0].result.doc_id, "b");
            assert_eq!(initial[0].fast_row.unwrap().shard, 1);
            let candidate = old
                .begin_update(&cx, root.path().join("new"), generation(2))
                .unwrap()
                .delete_document("a")
                .build(&cx, 2)
                .await
                .unwrap();
            live.install(&cx, &candidate).await.unwrap();
            let new_page = live.search(&cx, "vertical", 1).await.unwrap();
            assert_eq!(new_page.snapshot.generation(), generation(2));
            assert_eq!(new_page.results[0].fast_row.unwrap().shard, 0);
            assert!(new_page.snapshot.index().vectors().document("a").is_none());
            clock.advance(10_000_000);
            let NativeShardedSearchPhase::RefinementFailed {
                initial_results,
                error,
            } = query.next_phase().await.unwrap().unwrap()
            else {
                panic!("timeout");
            };
            assert!(matches!(error, SearchError::SearchTimeout { .. }));
            same_page(&initial_results, &initial);
            assert_eq!(old.generation(), generation(1));
            assert!(old.index().vectors().document("a").is_some());
            assert_eq!(
                live.snapshot(&cx).await.unwrap().generation(),
                generation(2)
            );
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(query.is_finished());
            assert!(budget.timer.is_empty());
        });
    }
}
