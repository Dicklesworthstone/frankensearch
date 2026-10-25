//! Caller-owned blocking execution for identity-bound cache admission.
//!
//! Workers only prepare owned resources; they never publish into a live cache.
//! Dropping an awaiting caller cannot turn late file/graph admission into an
//! unobserved cache replacement. The caller retains and drains its pool.

use std::path::Path;
use std::sync::Arc;

use asupersync::Cx;
use asupersync::runtime::JoinError;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch_core::{SearchError, SearchResult, TwoTierConfig};
use frankensearch_index::{FsviV2IdentityBinding, TwoTierIndex, TwoTierIndexPaths};

use super::{IndexCache, NativeCachePolicy, StalenessDetector, absolute_from, checkpoint};

/// An immutable recipe, not a cache handle or installation capability. Workers
/// cannot swap an index after the awaiting future has been dropped.
struct ReloadInput {
    paths: TwoTierIndexPaths,
    config: TwoTierConfig,
    fast: FsviV2IdentityBinding,
    quality: Option<FsviV2IdentityBinding>,
    native: Option<NativeCachePolicy>,
}

impl ReloadInput {
    fn open(self, cx: &Cx) -> SearchResult<TwoTierIndex> {
        checkpoint(cx)?;
        if let Some(native) = &self.native {
            native.validate_paths(&self.paths)?;
        }
        checkpoint(cx)?;
        let result =
            super::super::open_exact(&self.paths, self.config, &self.fast, self.quality.as_ref());
        checkpoint(cx)?;
        let mut index = result?;
        if let Some(native) = &self.native {
            native.load(Some(cx), &self.paths, &mut index)?;
        }
        checkpoint(cx)?;
        Ok(index)
    }
}

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
    /// Load an exact v2 successor off-executor, then conditionally install it.
    ///
    /// Capture `expected` with [`Self::current`] before selecting the successor.
    /// Bindings come from trusted caller publication, never from discovering
    /// identity in candidate files. A stale token returns `Ok(false)` before
    /// worker admission or file access. Required paths, tier topology, complete
    /// producer identities, monotone generation sequences and same-generation
    /// content witnesses retain the synchronous selected-reload contract.
    ///
    /// Both vector tiers and every configured native graph are loaded on the
    /// caller's pool. Queries can keep using the old snapshot during that work.
    /// Only this awaiting caller performs the final pointer-fenced swap; a
    /// dropped future, cancelled request, failed admission or concurrent winner
    /// cannot be overwritten by a late worker. Native caches never degrade to
    /// exact retrieval. No disk selection or artifact is written or removed.
    ///
    /// The caller retains and drains the pool. Cancellation brackets blocking
    /// work; it does not preempt file I/O or graph decoding, and the final short
    /// cache lock is synchronous. No await or cancellation check follows a
    /// successful swap. This is an in-process fence, not durable authority.
    ///
    /// # Errors
    /// Returns cancellation, worker, identity, topology, graph or file errors.
    /// Every error preserves the installed snapshot; a lost race returns false.
    pub async fn reload_admitted_v2_with_pool(
        &self,
        cx: &Cx,
        expected: &Arc<TwoTierIndex>,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
        pool: BlockingPoolHandle,
    ) -> SearchResult<bool> {
        self.reload_on_pool_with(
            cx,
            expected,
            fast_binding,
            quality_binding,
            pool,
            |request, input| input.open(request),
        )
        .await
    }

    // One implementation for production and deterministic interleaving tests.
    // The loader owns only the frozen recipe, never `self` or `expected`.
    async fn reload_on_pool_with<F>(
        &self,
        cx: &Cx,
        expected: &Arc<TwoTierIndex>,
        fast_binding: &FsviV2IdentityBinding,
        quality_binding: Option<&FsviV2IdentityBinding>,
        pool: BlockingPoolHandle,
        load: F,
    ) -> SearchResult<bool>
    where
        F: FnOnce(&Cx, ReloadInput) -> SearchResult<TwoTierIndex> + Send + 'static,
    {
        checkpoint(cx)?;
        if !Arc::ptr_eq(&self.current(), expected) {
            return Ok(false);
        }
        let paths = self.index_paths().ok_or_else(|| {
            super::super::rejected("reload", "selected v2 reload requires explicit cache paths")
        })?;
        super::super::validate_bindings(paths, fast_binding, quality_binding)?;
        super::super::validate_successor_bindings(expected, fast_binding, quality_binding)?;
        let input = ReloadInput {
            paths: paths.clone(),
            config: self.config.clone(),
            fast: fast_binding.clone(),
            quality: quality_binding.cloned(),
            native: self.native_reload.clone(),
        };
        let candidate = on_pool(cx, pool, move |request| load(request, input)).await?;
        // Installation stays outside the worker. Existing install gates recheck
        // the token, cancellation, all identities and content under the lock.
        if self.native_reload.is_some() {
            self.install_loaded_native(Some(cx), expected, candidate)
        } else {
            self.replace_admitted_v2_if_current(cx, expected, candidate)
        }
    }

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
        let native = native
            .map(|policy| policy.into_absolute(&base))
            .transpose()?;
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
    use frankensearch_index::VectorIndex;
    use frankensearch_index::native_hnsw::{HnswParams, native_hnsw_generation_receipt_path};
    use std::collections::BTreeMap;
    use std::fs;
    use std::future::Future;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::task::{Context, Poll, Waker};
    use std::time::{Duration, Instant};

    use crate::cache::{NativeCacheTier, SentinelFileDetector};

    fn run<F, Fut, T>(body: F) -> T
    where
        F: FnOnce(Cx, BlockingPoolHandle) -> Fut,
        Fut: Future<Output = T>,
    {
        let runtime = RuntimeBuilder::current_thread().build().unwrap();
        let pool = BlockingPool::new(0, 2);
        let handle = pool.handle();
        let result = runtime
            .block_on(async move { body(Cx::current().expect("runtime context"), handle).await });
        assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
        result
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
                for (rank, id) in ids.iter().enumerate() {
                    let mut vector = vec![0.0; identity.dimension()];
                    #[allow(clippy::cast_precision_loss)]
                    let angle = (rank as f32) * 0.25;
                    vector[0] = angle.cos();
                    vector[1] = angle.sin();
                    writer.write_record(id, &vector).unwrap();
                }
                writer.finish().unwrap();
            }
            fs::rename(staged.path().join("fast.fsvi"), self.paths.fast_index()).unwrap();
            fs::rename(
                staged.path().join("quality.fsvi"),
                self.paths.quality_index().unwrap(),
            )
            .unwrap();
            let mut index = super::super::super::open_exact(
                &self.paths,
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
        hits.into_iter().map(|hit| hit.doc_id.to_string()).collect()
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

    fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        future.poll(&mut Context::from_waker(Waker::noop()))
    }

    #[allow(clippy::future_not_send)] // std's mpsc::Receiver is not Sync; test-only helper
    async fn rendezvous(cx: &Cx, receiver: &mpsc::Receiver<()>) {
        let started = Instant::now();
        loop {
            match receiver.try_recv() {
                Ok(()) => return,
                Err(mpsc::TryRecvError::Empty) => {}
                Err(error) => panic!("admission worker exited before rendezvous: {error}"),
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "worker did not reach rendezvous"
            );
            asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
        }
    }

    #[test]
    fn pool_reload_installs_all_required_tiers_and_keeps_old_readers_queryable() {
        run(|cx, pool| async move {
            for (fast_graph, quality_graph) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let fixture = Fixture::new(fast_graph, quality_graph);
                let cache = fixture.open(&cx, pool.clone()).await.unwrap();
                let old = cache.current();
                let (fast, quality) = fixture.stage(6, &["new-fast"], &["new-quality-only"]);
                let before = image(fixture.directory.path());
                assert!(
                    cache
                        .reload_admitted_v2_with_pool(
                            &cx,
                            &old,
                            &fast,
                            Some(&quality),
                            pool.clone()
                        )
                        .await
                        .unwrap()
                );
                let installed = cache.current();
                assert!(!Arc::ptr_eq(&old, &installed));
                assert_eq!(installed.fast_admitted_binding(), Some(&fast));
                assert_eq!(installed.quality_admitted_binding(), Some(&quality));
                assert_eq!(installed.has_native_fast_hnsw(), fast_graph);
                assert_eq!(installed.has_native_quality_hnsw(), quality_graph);
                assert_eq!(ids(&installed, false), ["new-fast"]);
                assert_eq!(ids(&installed, true), ["new-quality-only"]);
                assert_eq!(ids(&old, false), ["fast", "shared"]);
                assert_eq!(ids(&old, true), ["quality-only", "shared"]);
                assert!(
                    cache
                        .reload_admitted_v2_with_pool(
                            &cx,
                            &installed,
                            &fast,
                            Some(&quality),
                            pool.clone(),
                        )
                        .await
                        .unwrap(),
                    "an identical same-generation reload is still admissible"
                );
                assert_eq!(image(fixture.directory.path()), before);
            }
        });
    }

    #[test]
    fn pool_reload_stale_tokens_do_not_touch_files_or_require_a_live_pool() {
        run(|cx, pool| async move {
            let fixture = Fixture::new(true, true);
            let cache = fixture.open(&cx, pool.clone()).await.unwrap();
            let old = cache.current();
            cache.reload().unwrap();
            let installed = cache.current();
            fs::rename(
                fixture.paths.fast_index(),
                fixture.directory.path().join("saved"),
            )
            .unwrap();
            let stopped = BlockingPool::new(0, 1);
            let handle = stopped.handle();
            assert!(stopped.shutdown_and_wait(Duration::from_secs(10)));
            assert!(
                !cache
                    .reload_admitted_v2_with_pool(&cx, &old, &fixture.fast, None, handle.clone())
                    .await
                    .unwrap()
            );
            cx.set_cancel_requested(true);
            let cancelled = cache
                .reload_admitted_v2_with_pool(&cx, &old, &fixture.fast, None, handle)
                .await;
            cx.set_cancel_requested(false);
            assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
            assert!(Arc::ptr_eq(&installed, &cache.current()));
            assert_eq!(ids(&old, true), ["quality-only", "shared"]);
        });
    }

    #[test]
    fn pool_reload_refuses_partial_foreign_rollback_and_rebound_generations() {
        run(|cx, pool| async move {
            for fault in [
                "missing-graph",
                "producer",
                "rollback",
                "rebound",
                "missing-binding",
            ] {
                let fixture = Fixture::new(true, true);
                let cache = fixture.open(&cx, pool.clone()).await.unwrap();
                let old = cache.current();
                let sequence = match fault {
                    "rollback" => 4,
                    "rebound" => 5,
                    _ => 6,
                };
                let (mut fast, quality) =
                    fixture.stage(sequence, &["changed"], &["quality-changed"]);
                if fault == "producer" {
                    fast = binding("foreign-producer", 2, sequence);
                }
                if fault == "missing-graph" {
                    let path = &fixture.policy.quality.as_ref().unwrap().graph_path;
                    fs::rename(path, path.with_extension("saved")).unwrap();
                }
                let required = (fault != "missing-binding").then_some(&quality);
                let before = image(fixture.directory.path());
                assert!(
                    cache
                        .reload_admitted_v2_with_pool(&cx, &old, &fast, required, pool.clone())
                        .await
                        .is_err(),
                    "{fault} was admitted"
                );
                assert!(Arc::ptr_eq(&old, &cache.current()));
                assert_eq!(ids(&old, false), ["fast", "shared"]);
                assert_eq!(ids(&old, true), ["quality-only", "shared"]);
                assert_eq!(image(fixture.directory.path()), before);
            }
        });
    }

    #[test]
    fn parked_pool_reload_cannot_overwrite_a_newer_async_winner() {
        run(|cx, pool| async move {
            let fixture = Fixture::new(true, true);
            let cache = fixture.open(&cx, pool.clone()).await.unwrap();
            let old = cache.current();
            let (fast, quality) = fixture.stage(6, &["slow"], &["slow-quality"]);
            let (started, ready) = mpsc::channel();
            let (release, barrier) = mpsc::channel();
            let mut slow = Box::pin(cache.reload_on_pool_with(
                &cx,
                &old,
                &fast,
                Some(&quality),
                pool.clone(),
                move |request, input| {
                    let candidate = input.open(request)?;
                    started.send(()).unwrap();
                    barrier.recv_timeout(Duration::from_secs(10)).unwrap();
                    Ok(candidate)
                },
            ));
            assert!(poll_once(slow.as_mut()).is_pending());
            rendezvous(&cx, &ready).await;
            assert_eq!(ids(&cache.current(), false), ["fast", "shared"]);
            let (new_fast, new_quality) = fixture.stage(7, &["winner"], &["winner-quality"]);
            assert!(
                cache
                    .reload_admitted_v2_with_pool(&cx, &old, &new_fast, Some(&new_quality), pool)
                    .await
                    .unwrap()
            );
            let winner = cache.current();
            release.send(()).unwrap();
            assert!(
                !slow.await.unwrap(),
                "late loader must lose its original token"
            );
            assert!(Arc::ptr_eq(&winner, &cache.current()));
            assert_eq!(ids(&winner, false), ["winner"]);
            assert_eq!(ids(&winner, true), ["winner-quality"]);
            assert_eq!(ids(&old, false), ["fast", "shared"]);
        });
    }

    #[test]
    fn dropped_pool_reload_cannot_install_after_worker_and_runtime_drain() {
        let (cache, old, fixture, before) = run(|cx, pool| async move {
            let fixture = Fixture::new(true, true);
            let cache = fixture.open(&cx, pool.clone()).await.unwrap();
            let old = cache.current();
            let (fast, quality) = fixture.stage(6, &["late"], &["late-quality"]);
            let before = image(fixture.directory.path());
            let (started, ready) = mpsc::channel();
            let (release, barrier) = mpsc::channel();
            let (finished, drained) = mpsc::channel();
            let mut pending = Box::pin(cache.reload_on_pool_with(
                &cx,
                &old,
                &fast,
                Some(&quality),
                pool,
                move |request, input| {
                    let candidate = input.open(request)?;
                    started.send(()).unwrap();
                    barrier.recv_timeout(Duration::from_secs(10)).unwrap();
                    finished.send(()).unwrap();
                    Ok(candidate)
                },
            ));
            assert!(poll_once(pending.as_mut()).is_pending());
            rendezvous(&cx, &ready).await;
            drop(pending);
            release.send(()).unwrap();
            rendezvous(&cx, &drained).await;
            (cache, old, fixture, before)
        });
        // run() has drained the real worker pool and dropped its runtime.
        assert!(Arc::ptr_eq(&old, &cache.current()));
        assert_eq!(ids(&old, false), ["fast", "shared"]);
        assert_eq!(ids(&old, true), ["quality-only", "shared"]);
        assert_eq!(image(fixture.directory.path()), before);
    }

    #[test]
    fn cancelled_pool_reload_discards_an_already_admitted_candidate() {
        run(|cx, pool| async move {
            let fixture = Fixture::new(true, true);
            let cache = fixture.open(&cx, pool.clone()).await.unwrap();
            let old = cache.current();
            let (fast, quality) = fixture.stage(6, &["cancelled"], &["cancelled-quality"]);
            let before = image(fixture.directory.path());
            let (started, ready) = mpsc::channel();
            let (release, barrier) = mpsc::channel();
            let mut pending = Box::pin(cache.reload_on_pool_with(
                &cx,
                &old,
                &fast,
                Some(&quality),
                pool,
                move |request, input| {
                    let candidate = input.open(request)?;
                    started.send(()).unwrap();
                    barrier.recv_timeout(Duration::from_secs(10)).unwrap();
                    Ok(candidate)
                },
            ));
            assert!(poll_once(pending.as_mut()).is_pending());
            rendezvous(&cx, &ready).await;
            cx.set_cancel_requested(true);
            release.send(()).unwrap();
            let result = pending.await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert!(Arc::ptr_eq(&old, &cache.current()));
            assert_eq!(ids(&old, true), ["quality-only", "shared"]);
            assert_eq!(image(fixture.directory.path()), before);
        });
    }
}
