//! Caller-owned blocking execution for identity-bound cache admission.
//!
//! Workers only prepare owned resources; they never publish into a live cache.
//! Dropping an awaiting caller cannot turn late file/graph admission into an
//! unobserved cache replacement. The caller retains and drains its pool.

use std::path::Path;

use asupersync::Cx;
use asupersync::runtime::JoinError;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch_core::{SearchError, SearchResult, TwoTierConfig};
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndexPaths};

use super::{IndexCache, NativeCachePolicy, StalenessDetector, absolute_from, checkpoint};

fn worker_error(reason: &'static str) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "index_cache.blocking",
        source: reason.into(),
    }
}

fn require_pool(cx: &Cx, pool: &BlockingPoolHandle) -> SearchResult<()> {
    checkpoint(cx)?;
    if pool.is_shutdown() {
        return Err(worker_error(
            "the caller-owned cache loading pool is shut down",
        ));
    }
    Ok(())
}

async fn on_pool<T, F>(cx: &Cx, pool: BlockingPoolHandle, load: F) -> SearchResult<T>
where
    T: Send + 'static,
    F: FnOnce(&Cx) -> SearchResult<T> + Send + 'static,
{
    require_pool(cx, &pool)?;
    let request = cx.clone();
    let worker_cx = cx.clone().with_blocking_pool_handle(Some(pool));
    let mut worker = worker_cx
        .spawn_blocking(move |child: Cx| {
            checkpoint(&request)?;
            checkpoint(&child)?;
            // Pool shutdown can race dispatch. Asupersync may recover rejected
            // blocking work on an executor task; refuse before touching files.
            if Cx::is_active() {
                return Err(worker_error(
                    "cache loading cannot run inline on the async executor",
                ));
            }
            let result = load(&request);
            checkpoint(&request)?;
            checkpoint(&child)?;
            result
        })
        .map_err(|_| worker_error("cannot admit the cache loading worker"))?;
    let result = worker.join(cx).await;
    // Join does not itself inspect the request Cx. Neither successful loading
    // nor a simultaneous I/O/worker error may overtake observed cancellation.
    checkpoint(cx)?;
    result.map_err(|error| match error {
        JoinError::Cancelled(_) => SearchError::Cancelled {
            phase: "index_cache.admitted_v2".to_owned(),
            reason: "cache loading worker cancelled".to_owned(),
        },
        _ => worker_error("cache loading worker did not complete"),
    })?
}

impl IndexCache {
    /// Open retained v2 vectors and optional persisted native graphs off-executor.
    ///
    /// `native: None` selects exact retrieval. `Some(policy)` admits every graph
    /// in that policy using the existing strict native loader; it never builds
    /// a missing graph or silently substitutes exact search. Both selected
    /// vector tiers remain required, with independently bound producers and
    /// coverage. The ordinary synchronous constructors remain available to
    /// callers that already own a blocking lane.
    ///
    /// Paths (including graph and sentinel paths) freeze against one CWD snapshot
    /// before submission. File reads, vector admission and graph decoding run
    /// on `pool`, not the asynchronous executor. No model, file or sentinel is
    /// written. A stopped pool or inline dispatch fallback is an error.
    ///
    /// The caller must retain and drain the pool. Cancellation is checked around
    /// blocking operations and before returning resources, not inside kernel I/O
    /// or graph decoding. Dropping this future does not install anything; already
    /// running work may drain on its owner before its resources are released.
    /// This is not durable generation authority or hostile-directory admission.
    ///
    /// # Errors
    /// Returns cancellation, unavailable-worker, path, identity, required-tier
    /// or native graph/receipt validation errors. No fallback or repair occurs.
    #[allow(clippy::too_many_arguments)]
    pub async fn open_admitted_v2_with_pool(
        cx: &Cx,
        paths: TwoTierIndexPaths,
        state_dir: &Path,
        config: TwoTierConfig,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
        native: Option<NativeCachePolicy>,
        detector: Box<dyn StalenessDetector>,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Self> {
        require_pool(cx, &pool)?;
        super::super::validate_bindings(&paths, fast_binding, quality_binding)?;
        let base = std::env::current_dir()?;
        let paths = paths.into_absolute_from(&base)?;
        let state_dir = absolute_from(state_dir, &base);
        let native = native.map(|policy| policy.into_absolute(&base)).transpose()?;
        let fast = fast_binding.clone();
        let quality = quality_binding.cloned();
        on_pool(cx, pool, move |request| match native {
            Some(native) => Self::open_admitted_v2_with_native(
                request,
                paths,
                &state_dir,
                config,
                &fast,
                quality.as_ref(),
                native,
                detector,
            ),
            None => Self::open_admitted_v2_with_paths(
                request,
                paths,
                &state_dir,
                config,
                &fast,
                quality.as_ref(),
                detector,
            ),
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::runtime::RuntimeBuilder;
    use asupersync::runtime::blocking_pool::BlockingPool;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use frankensearch_core::{BoundQueryEmbedding, TieredQueryEmbeddings};
    use frankensearch_index::native_hnsw::{HnswParams, native_hnsw_generation_receipt_path};
    use frankensearch_index::{TwoTierIndex, VectorIndex};
    use std::collections::BTreeMap;
    use std::fs;
    use std::future::Future;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::cache::{NativeCacheTier, SentinelFileDetector};

    fn run<F, Fut>(body: F)
    where
        F: FnOnce(Cx, BlockingPoolHandle) -> Fut,
        Fut: Future<Output = ()>,
    {
        let runtime = RuntimeBuilder::current_thread().build().unwrap();
        let pool = BlockingPool::new(0, 2);
        let handle = pool.handle();
        runtime.block_on(async move {
            body(Cx::current().expect("runtime context"), handle).await;
        });
        assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
    }

    fn binding(name: &str, dimension: u32, sequence: u64) -> FsviV2IdentityBinding {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        "fsvi-v2".clone_into(&mut identity.storage.format);
        "little-endian".clone_into(&mut identity.storage.endianness);
        identity.storage.quantization = QuantizationFormat::F32;
        FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(sequence, [7; 16]).unwrap(),
            identity.freeze().unwrap(),
        )
        .unwrap()
    }

    // Real FSVI owners, persisted native graphs and bound queries; small control
    // vectors establish loading/ownership, not real-model recall or latency.
    struct Fixture {
        directory: tempfile::TempDir,
        paths: TwoTierIndexPaths,
        fast: FsviV2IdentityBinding,
        quality: FsviV2IdentityBinding,
        policy: NativeCachePolicy,
    }

    impl Fixture {
        fn new(fast_graph: bool, quality_graph: bool) -> Self {
            let directory = tempfile::tempdir().unwrap();
            let paths = TwoTierIndexPaths::new(directory.path().join("fast.fsvi"))
                .with_quality_index(directory.path().join("quality.fsvi"));
            let graph = |name: &str, seed| NativeCacheTier {
                graph_path: directory.path().join(name),
                params: HnswParams::default(),
                seed,
            };
            let fixture = Self {
                paths,
                fast: binding("pool-fast", 2, 5),
                quality: binding("pool-quality", 3, 5),
                policy: NativeCachePolicy {
                    fast: fast_graph.then(|| graph("fast.fshnsw", 41)),
                    quality: quality_graph.then(|| graph("quality.fshnsw", 43)),
                },
                directory,
            };
            fixture.stage(5, &["fast", "shared"], &["quality-only", "shared"]);
            fixture
        }

        fn stage(
            &self,
            sequence: u64,
            fast_ids: &[&str],
            quality_ids: &[&str],
        ) -> (FsviV2IdentityBinding, FsviV2IdentityBinding) {
            let staged = tempfile::tempdir_in(self.directory.path()).unwrap();
            let fast = binding("pool-fast", 2, sequence);
            let quality = binding("pool-quality", 3, sequence);
            for (name, identity, ids) in [
                ("fast.fsvi", &fast, fast_ids),
                ("quality.fsvi", &quality, quality_ids),
            ] {
                let mut writer =
                    VectorIndex::create_v2(&staged.path().join(name), identity.clone()).unwrap();
                for id in ids {
                    let mut vector = vec![0.0; identity.dimension()];
                    vector[0] = 1.0;
                    writer.write_record(id, &vector).unwrap();
                }
                writer.finish().unwrap();
            }
            let paths = TwoTierIndexPaths::new(staged.path().join("fast.fsvi"))
                .with_quality_index(staged.path().join("quality.fsvi"));
            let mut index = super::super::super::open_exact(
                &paths,
                TwoTierConfig::default(),
                &fast,
                Some(&quality),
            )
            .unwrap();
            for (is_fast, tier) in [(true, &self.policy.fast), (false, &self.policy.quality)] {
                let Some(tier) = tier else { continue };
                let path = staged.path().join(tier.graph_path.file_name().unwrap());
                if is_fast {
                    index
                        .enable_native_fast_hnsw(tier.params, tier.seed)
                        .unwrap();
                    index.save_native_fast_hnsw(&path).unwrap();
                } else {
                    index
                        .enable_native_quality_hnsw(tier.params, tier.seed)
                        .unwrap();
                    index.save_native_quality_hnsw(&path).unwrap();
                }
                fs::rename(
                    native_hnsw_generation_receipt_path(&path).unwrap(),
                    native_hnsw_generation_receipt_path(&tier.graph_path).unwrap(),
                )
                .unwrap();
                fs::rename(path, &tier.graph_path).unwrap();
            }
            drop(index);
            fs::rename(staged.path().join("fast.fsvi"), self.paths.fast_index()).unwrap();
            fs::rename(
                staged.path().join("quality.fsvi"),
                self.paths.quality_index().unwrap(),
            )
            .unwrap();
            (fast, quality)
        }

        fn native(&self) -> Option<NativeCachePolicy> {
            (self.policy.fast.is_some() || self.policy.quality.is_some())
                .then(|| self.policy.clone())
        }

        async fn open(&self, cx: &Cx, pool: BlockingPoolHandle) -> SearchResult<IndexCache> {
            IndexCache::open_admitted_v2_with_pool(
                cx,
                self.paths.clone(),
                self.directory.path(),
                TwoTierConfig::default(),
                &self.fast,
                Some(&self.quality),
                self.native(),
                Box::new(SentinelFileDetector::new()),
                pool,
            )
            .await
        }
    }

    fn image(directory: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        fs::read_dir(directory)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.path(), fs::read(entry.path()).unwrap())
            })
            .collect()
    }

    fn ids(index: &TwoTierIndex, quality: bool) -> Vec<String> {
        let (name, dimension) = if quality {
            ("pool-quality", 3)
        } else {
            ("pool-fast", 2)
        };
        let mut values = vec![0.0; dimension];
        values[0] = 1.0;
        let embedding = BoundQueryEmbedding::new(
            values,
            EmbeddingIdentityBundleV1::explicit_test_model(name, u32::try_from(dimension).unwrap()),
        )
        .unwrap();
        let query = if quality {
            TieredQueryEmbeddings::quality_only(embedding)
        } else {
            TieredQueryEmbeddings::fast_only(embedding)
        };
        let activated = index.activate_owner_backed_search(&query).unwrap();
        let hits = if quality {
            activated.search_quality(10).unwrap()
        } else {
            activated.search_fast(10).unwrap()
        };
        hits.into_iter()
            .map(|hit| hit.doc_id.to_string())
            .collect()
    }

    #[test]
    fn pool_open_preserves_exact_and_native_tiers_and_independent_coverage() {
        run(|cx, pool| async move {
            for (fast_graph, quality_graph) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let fixture = Fixture::new(fast_graph, quality_graph);
                let before = image(fixture.directory.path());
                let cache = fixture.open(&cx, pool.clone()).await.unwrap();
                let index = cache.current();
                assert_eq!(index.has_native_fast_hnsw(), fast_graph);
                assert_eq!(index.has_native_quality_hnsw(), quality_graph);
                assert_eq!(index.fast_admitted_binding(), Some(&fixture.fast));
                assert_eq!(index.quality_admitted_binding(), Some(&fixture.quality));
                assert_eq!(ids(&index, false), ["fast", "shared"]);
                assert_eq!(ids(&index, true), ["quality-only", "shared"]);
                assert_eq!(cache.native_cache_policy(), fixture.native().as_ref());
                assert_eq!(image(fixture.directory.path()), before);
            }
        });
    }

    #[test]
    fn pool_open_never_degrades_a_missing_or_corrupt_required_graph() {
        run(|cx, pool| async move {
            for missing in [false, true] {
                let fixture = Fixture::new(true, true);
                let graph = &fixture.policy.quality.as_ref().unwrap().graph_path;
                if missing {
                    fs::rename(graph, graph.with_extension("saved")).unwrap();
                } else {
                    fs::write(graph, b"corrupt graph").unwrap();
                }
                let before = image(fixture.directory.path());
                assert!(fixture.open(&cx, pool.clone()).await.is_err());
                assert_eq!(image(fixture.directory.path()), before);
            }
        });
    }

    #[test]
    fn pool_open_refuses_stopped_workers_and_precancel_before_artifact_access() {
        run(|cx, _| async move {
            let fixture = Fixture::new(true, true);
            let stopped = BlockingPool::new(0, 1);
            let handle = stopped.handle();
            assert!(stopped.shutdown_and_wait(Duration::from_secs(10)));
            // A missing artifact would be an I/O error if pool refusal came late.
            fs::rename(
                fixture.paths.fast_index(),
                fixture.directory.path().join("saved"),
            )
            .unwrap();
            assert!(matches!(
                fixture.open(&cx, handle.clone()).await,
                Err(SearchError::SubsystemError {
                    subsystem: "index_cache.blocking",
                    ..
                })
            ));
            cx.set_cancel_requested(true);
            let result = fixture.open(&cx, handle).await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        });
    }

    #[test]
    fn pool_boundary_executes_off_executor_and_preserves_request_cancellation() {
        run(|cx, pool| async move {
            let caller = std::thread::current().id();
            let worker = on_pool(&cx, pool.clone(), |_| {
                assert!(
                    !Cx::is_active(),
                    "artifact admission must not occupy the executor"
                );
                Ok(std::thread::current().id())
            })
            .await
            .unwrap();
            assert_ne!(caller, worker);
            for success in [false, true] {
                let calls = Arc::new(AtomicUsize::new(0));
                let observed = calls.clone();
                let result: SearchResult<()> = on_pool(&cx, pool.clone(), move |request| {
                    observed.fetch_add(1, Ordering::Relaxed);
                    request.set_cancel_requested(true);
                    if success {
                        Ok(())
                    } else {
                        Err(worker_error("simultaneous failure"))
                    }
                })
                .await;
                cx.set_cancel_requested(false);
                assert_eq!(calls.load(Ordering::Relaxed), 1);
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            }
        });
    }
}
