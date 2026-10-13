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
use crate::native_ann::NativeShardedResult;
use crate::native_ann::builder::sharded::NativeBuiltShardedHybridIndex;
use crate::native_ann::builder::sharded::live::{
    NativeLiveShardedHybridIndex, NativeShardedResults,
};
use crate::{Cx, SearchResult};

impl NativeBuiltShardedHybridIndex {
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
    use frankensearch_core::traits::IdentityBoundEmbedding;
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
                let index =
                    build(&cx, &root.path().join("index"), &fast, Some(&quality), graph).await;
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
            let index = build(&cx, &root.path().join("index"), &fast, Some(&quality), false).await;
            let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
            let (_, budget) = deadline(&cx, 0);
            assert!(matches!(
                live.search_before(&cx, "vertical", 3, &budget).await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert!(matches!(
                live.search_quality_before(&cx, "vertical", 3, &budget).await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert!(matches!(
                live.search_refined_before(&cx, "vertical", 3, &budget).await,
                Err(SearchError::SearchTimeout { .. })
            ));
            cx.set_cancel_requested(true);
            assert!(matches!(
                live.search_refined_before(&cx, "vertical", 3, &budget).await,
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
            let index = build(&cx, &root.path().join("index"), &fast, Some(&quality), false).await;
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
}
