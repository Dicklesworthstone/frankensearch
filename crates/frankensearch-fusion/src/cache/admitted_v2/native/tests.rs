use super::*;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::{
    ArtifactGenerationIdentityV1, EmbeddingIdentityBundleV1, QuantizationFormat,
};
use frankensearch_core::{BoundQueryEmbedding, SearchError, TieredQueryEmbeddings};
use frankensearch_index::VectorIndex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cache::{SENTINEL_FILENAME, SentinelFileDetector};

// Real FSVI/native graph persistence and querying. These small identified
// control vectors test ownership and lifecycle, not semantic quality/recall.
static NEXT_STAGE: AtomicUsize = AtomicUsize::new(0);

fn binding(name: &str, dimension: u32, sequence: u64, nonce: u8) -> FsviV2IdentityBinding {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
    "fsvi-v2".clone_into(&mut identity.storage.format);
    "little-endian".clone_into(&mut identity.storage.endianness);
    identity.storage.quantization = QuantizationFormat::F32;
    FsviV2IdentityBinding::new(
        ArtifactGenerationIdentityV1::new(sequence, [nonce; 16]).unwrap(),
        identity.freeze().unwrap(),
    )
    .unwrap()
}

fn write_vectors(path: &Path, binding: &FsviV2IdentityBinding, ids: &[&str], axis: usize) {
    let staged = path.with_extension(format!(
        "stage-{}", NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut writer = VectorIndex::create_v2(&staged, binding.clone()).unwrap();
    for id in ids {
        let mut values = vec![0.0; binding.dimension()];
        values[axis] = 1.0;
        writer.write_record(id, &values).unwrap();
    }
    writer.finish().unwrap();
    fs::rename(staged, path).unwrap();
}

fn persist(index: &mut TwoTierIndex, policy: &NativeCachePolicy, fast: bool) {
    let tier = if fast { &policy.fast } else { &policy.quality };
    let Some(tier) = tier else { return };
    let staged = tier.graph_path.with_extension(format!(
        "stage-{}", NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
    ));
    if fast {
        index.enable_native_fast_hnsw(tier.params, tier.seed).unwrap();
        index.save_native_fast_hnsw(&staged).unwrap();
    } else {
        index.enable_native_quality_hnsw(tier.params, tier.seed).unwrap();
        index.save_native_quality_hnsw(&staged).unwrap();
    }
    fs::rename(
        native_hnsw_generation_receipt_path(&staged).unwrap(),
        native_hnsw_generation_receipt_path(&tier.graph_path).unwrap(),
    ).unwrap();
    fs::rename(staged, &tier.graph_path).unwrap();
}

struct Fixture {
    directory: tempfile::TempDir,
    paths: TwoTierIndexPaths,
    fast: FsviV2IdentityBinding,
    quality: Option<FsviV2IdentityBinding>,
    policy: NativeCachePolicy,
}

impl Fixture {
    fn new(fast_graph: bool, quality_graph: bool, quality_vectors: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let fast = binding("cache-fast", 2, 5, 7);
        let quality = quality_vectors.then(|| binding("cache-quality", 3, 5, 7));
        let mut paths = TwoTierIndexPaths::new(directory.path().join("fast.fsvi"));
        write_vectors(paths.fast_index(), &fast, &["old"], 0);
        if let Some(binding) = &quality {
            let path = directory.path().join("quality.fsvi");
            write_vectors(&path, binding, &["old"], 1);
            paths = paths.with_quality_index(path);
        }
        let graph = |name: &str, seed| NativeCacheTier {
            graph_path: directory.path().join(name),
            params: HnswParams::default(),
            seed,
        };
        let policy = NativeCachePolicy {
            fast: fast_graph.then(|| graph("fast.native", 41)),
            quality: quality_graph.then(|| graph("quality.native", 43)),
        };
        let mut index = open_exact(&paths, TwoTierConfig::default(), &fast, quality.as_ref())
            .unwrap();
        persist(&mut index, &policy, true);
        persist(&mut index, &policy, false);
        Self { directory, paths, fast, quality, policy }
    }

    fn open(&self, cx: &Cx) -> IndexCache {
        IndexCache::open_admitted_v2_with_native(
            cx, self.paths.clone(), self.directory.path(), TwoTierConfig::default(),
            &self.fast, self.quality.as_ref(), self.policy.clone(),
            Box::new(SentinelFileDetector::new()),
        ).unwrap()
    }

    fn stage_vectors(&self, sequence: u64, nonce: u8) -> (FsviV2IdentityBinding, FsviV2IdentityBinding) {
        let fast = binding("cache-fast", 2, sequence, nonce);
        let quality = binding("cache-quality", 3, sequence, nonce);
        write_vectors(self.paths.fast_index(), &fast, &["new", "second"], 1);
        write_vectors(self.paths.quality_index().unwrap(), &quality, &["new", "second"], 0);
        (fast, quality)
    }
}

fn image(directory: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fs::read_dir(directory).unwrap().map(|entry| {
        let entry = entry.unwrap();
        (entry.path(), fs::read(entry.path()).unwrap())
    }).collect()
}

fn fast_ids(index: &TwoTierIndex) -> Vec<String> {
    let query = TieredQueryEmbeddings::fast_only(BoundQueryEmbedding::new(
        vec![0.0, 1.0],
        EmbeddingIdentityBundleV1::explicit_test_model("cache-fast", 2),
    ).unwrap());
    index.activate_owner_backed_search(&query).unwrap().search_fast(10).unwrap()
        .into_iter().map(|hit| hit.doc_id).collect()
}

fn quality_ids(index: &TwoTierIndex) -> Vec<String> {
    let query = BoundQueryEmbedding::new(
        vec![0.0, 1.0, 0.0],
        EmbeddingIdentityBundleV1::explicit_test_model("cache-quality", 3),
    ).unwrap();
    index.search_quality_with_producer(&query, 10).unwrap()
        .into_iter().map(|hit| hit.doc_id).collect()
}

#[test]
fn persisted_native_cache_reopens_every_tier_policy_without_writes() {
    run_test_with_cx(|cx| async move {
        for (fast_graph, quality_graph, quality_vectors) in [
            (true, false, false), (true, false, true),
            (false, true, true), (true, true, true),
        ] {
            let fixture = Fixture::new(fast_graph, quality_graph, quality_vectors);
            let before = image(fixture.directory.path());
            let cache = fixture.open(&cx);
            let old = cache.current();
            assert_eq!(fast_ids(&old), ["old"]);
            for _ in 0..2 {
                cache.reload().unwrap();
                let current = cache.current();
                assert!(!Arc::ptr_eq(&old, &current));
                assert_eq!(current.has_native_fast_hnsw(), fast_graph);
                assert_eq!(current.has_native_quality_hnsw(), quality_graph);
                assert_eq!(fast_ids(&current), ["old"]);
                if quality_vectors { assert_eq!(quality_ids(&current), ["old"]); }
                assert_eq!(cache.native_cache_policy(), Some(&fixture.policy));
            }
            drop(cache);
            let restarted = fixture.open(&cx);
            assert_eq!(fast_ids(&restarted.current()), fast_ids(&old));
            assert_eq!(image(fixture.directory.path()), before);
            assert!(!fixture.directory.path().join(SENTINEL_FILENAME).exists());
        }
    });
}

#[test]
fn selected_native_reload_waits_for_both_graphs_and_preserves_old_readers() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(true, true, true);
        let cache = fixture.open(&cx);
        let old = cache.current();
        let (fast, quality) = fixture.stage_vectors(6, 8);
        let mut builder = open_exact(&fixture.paths, cache.config.clone(), &fast, Some(&quality))
            .unwrap();
        persist(&mut builder, &fixture.policy, true);
        let incomplete = image(fixture.directory.path());
        assert!(cache.reload().is_err(), "ordinary reload cannot discover a successor");
        assert!(cache.reload_admitted_v2_if_current(&cx, &old, &fast, Some(&quality)).is_err());
        assert!(Arc::ptr_eq(&old, &cache.current()));
        assert_eq!(image(fixture.directory.path()), incomplete);
        assert_eq!(fast_ids(&old), ["old"]);
        assert_eq!(quality_ids(&old), ["old"]);

        persist(&mut builder, &fixture.policy, false);
        let complete = image(fixture.directory.path());
        assert!(cache.reload_admitted_v2_if_current(&cx, &old, &fast, Some(&quality)).unwrap());
        let current = cache.current();
        assert_eq!(current.fast_admitted_binding(), Some(&fast));
        assert_eq!(current.quality_admitted_binding(), Some(&quality));
        assert!(current.has_native_fast_hnsw() && current.has_native_quality_hnsw());
        assert_eq!(fast_ids(&current), ["new", "second"]);
        assert_eq!(quality_ids(&current), ["new", "second"]);
        cache.reload().unwrap();
        assert_eq!(cache.current().fast_admitted_binding(), Some(&fast));
        assert_eq!(fast_ids(&old), ["old"]);
        assert_eq!(quality_ids(&old), ["old"]);
        assert_eq!(image(fixture.directory.path()), complete);
    });
}

#[test]
fn native_cache_refuses_prebuilt_policy_substitution_through_all_replace_entries() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(true, true, true);
        let cache = fixture.open(&cx);
        let old = cache.current();
        for native_candidate in [false, true] {
            let candidate = || {
                let mut index = open_exact(&fixture.paths, cache.config.clone(),
                    &fixture.fast, fixture.quality.as_ref()).unwrap();
                if native_candidate {
                    // Even a graph with the right owner is not the cache's recipe.
                    index.enable_native_fast_hnsw(HnswParams::default(), 99).unwrap();
                    index.enable_native_quality_hnsw(HnswParams::default(), 100).unwrap();
                }
                index
            };
            assert!(cache.replace(candidate()).is_err());
            assert!(cache.replace_if_current(&old, candidate()).is_err());
            assert!(cache.replace_admitted_v2_if_current(&cx, &old, candidate()).is_err());
            assert!(Arc::ptr_eq(&old, &cache.current()));
        }
        cache.reload().unwrap();
        assert!(cache.current().has_native_fast_hnsw());
        assert!(cache.current().has_native_quality_hnsw());
        assert_eq!(fast_ids(&old), ["old"]);
    });
}

#[test]
fn native_reload_refuses_corrupt_missing_and_wrong_policy_sidecars_without_fallback() {
    run_test_with_cx(|cx| async move {
        for quality_fault in [false, true] {
            for fault in ["graph", "receipt", "missing", "seed", "parameters"] {
                let fixture = Fixture::new(true, true, true);
                let cache = fixture.open(&cx);
                let old = cache.current();
                let target = if quality_fault { fixture.policy.quality.as_ref() }
                    else { fixture.policy.fast.as_ref() }.unwrap();
                match fault {
                    "graph" => fs::write(&target.graph_path, b"corrupt").unwrap(),
                    "receipt" => fs::write(native_hnsw_generation_receipt_path(&target.graph_path)
                        .unwrap(), b"corrupt").unwrap(),
                    "missing" => fs::rename(&target.graph_path, target.graph_path.with_extension("saved"))
                        .unwrap(),
                    _ => {
                        let mut wrong = fixture.policy.clone();
                        let tier = if quality_fault { wrong.quality.as_mut() }
                            else { wrong.fast.as_mut() }.unwrap();
                        if fault == "seed" { tier.seed += 1; }
                        else { tier.params.ef_search += 1; }
                        let mut builder = open_exact(&fixture.paths, cache.config.clone(),
                            &fixture.fast, fixture.quality.as_ref()).unwrap();
                        persist(&mut builder, &wrong, !quality_fault);
                    }
                }
                let before = image(fixture.directory.path());
                assert!(cache.reload().is_err(), "{fault}, quality={quality_fault}");
                assert!(cache.reload_admitted_v2_if_current(
                    &cx, &old, &fixture.fast, fixture.quality.as_ref()).is_err());
                assert!(Arc::ptr_eq(&old, &cache.current()));
                assert_eq!(fast_ids(&old), ["old"]);
                assert_eq!(quality_ids(&old), ["old"]);
                assert_eq!(image(fixture.directory.path()), before);
            }
        }
    });
}

#[test]
fn native_same_generation_vector_rebinding_remains_forbidden_even_with_valid_new_graphs() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(true, true, true);
        let cache = fixture.open(&cx);
        let old = cache.current();
        let (fast, quality) = fixture.stage_vectors(5, 7);
        let mut builder = open_exact(&fixture.paths, cache.config.clone(), &fast, Some(&quality)).unwrap();
        persist(&mut builder, &fixture.policy, true);
        persist(&mut builder, &fixture.policy, false);
        assert!(cache.reload().is_err());
        assert!(cache.reload_admitted_v2_if_current(&cx, &old, &fast, Some(&quality)).is_err());
        assert!(Arc::ptr_eq(&old, &cache.current()));
        assert_eq!(fast_ids(&old), ["old"]);
        assert_eq!(quality_ids(&old), ["old"]);
    });
}

#[test]
fn native_slow_reload_cannot_overwrite_a_concurrent_successful_reload() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(true, true, true);
        let cache = fixture.open(&cx);
        let old = cache.current();
        let mut winner = None;
        let error = cache.reload_with(|current| {
            let candidate = cache.open_for_reload(current)?;
            cache.reload()?;
            winner = Some(cache.current());
            Ok(candidate)
        }).unwrap_err();
        assert!(matches!(error, SearchError::InvalidConfig { ref value, .. } if value == "superseded"));
        assert!(Arc::ptr_eq(&winner.unwrap(), &cache.current()));
        assert_eq!(fast_ids(&old), ["old"]);
        assert_eq!(quality_ids(&old), ["old"]);
        // A stale selection is refused before opening now-damaged disk artifacts.
        fs::write(&fixture.policy.fast.as_ref().unwrap().graph_path, b"corrupt").unwrap();
        assert!(!cache.reload_admitted_v2_if_current(
            &cx, &old, &fixture.fast, fixture.quality.as_ref()).unwrap());
        assert_eq!(fast_ids(&cache.current()), ["old"]);
    });
}

#[test]
fn native_cancelled_and_competing_selected_reloads_keep_one_complete_snapshot() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(true, true, true);
        let cache = fixture.open(&cx);
        let old = cache.current();
        let error = cache.reload_admitted_v2_with(
            &cx, &old, &fixture.fast, fixture.quality.as_ref(), |_| {
                let candidate = cache.open_for_reload(&old)?;
                cx.set_cancel_requested(true);
                Ok(candidate)
            },
        ).unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(error, SearchError::Cancelled { .. }));
        assert!(Arc::ptr_eq(&old, &cache.current()));

        let barrier = std::sync::Barrier::new(2);
        let outcomes = std::thread::scope(|scope| {
            let reload = || {
                barrier.wait();
                cache.reload_admitted_v2_if_current(
                    &cx, &old, &fixture.fast, fixture.quality.as_ref()).unwrap()
            };
            let left = scope.spawn(reload);
            let right = scope.spawn(reload);
            (left.join().unwrap(), right.join().unwrap())
        });
        assert_ne!(outcomes.0, outcomes.1, "exactly one pointer-fenced reload wins");
        let current = cache.current();
        assert!(current.has_native_fast_hnsw() && current.has_native_quality_hnsw());
        assert_eq!(fast_ids(&current), fast_ids(&old));
        assert_eq!(quality_ids(&current), quality_ids(&old));
    });
}

#[test]
fn native_policy_and_role_admission_cannot_change_the_cache_contract() {
    run_test_with_cx(|cx| async move {
        let fixture = Fixture::new(true, true, true);
        let mut shared = fixture.policy.clone();
        shared.quality = shared.fast.clone();
        let mut vector_alias = fixture.policy.clone();
        vector_alias.fast.as_mut().unwrap().graph_path = fixture.paths.fast_index().to_path_buf();
        for policy in [NativeCachePolicy::default(), shared, vector_alias] {
            assert!(IndexCache::open_admitted_v2_with_native(
                &cx, fixture.paths.clone(), fixture.directory.path(), TwoTierConfig::default(),
                &fixture.fast, fixture.quality.as_ref(), policy,
                Box::new(SentinelFileDetector::new()),
            ).is_err());
        }
        let only_fast = TwoTierIndexPaths::new(fixture.paths.fast_index());
        assert!(IndexCache::open_admitted_v2_with_native(
            &cx, only_fast, fixture.directory.path(), TwoTierConfig::default(), &fixture.fast, None,
            fixture.policy.clone(), Box::new(SentinelFileDetector::new()),
        ).is_err());
        cx.set_cancel_requested(true);
        assert!(matches!(IndexCache::open_admitted_v2_with_native(
            &cx, fixture.paths.clone(), fixture.directory.path(), TwoTierConfig::default(),
            &fixture.fast, fixture.quality.as_ref(), fixture.policy.clone(),
            Box::new(SentinelFileDetector::new()),
        ), Err(SearchError::Cancelled { .. })));
        cx.set_cancel_requested(false);
        let cache = fixture.open(&cx);
        let old = cache.current();
        for (sequence, nonce) in [(4, 6), (5, 8)] {
            let fast = binding("cache-fast", 2, sequence, nonce);
            let quality = binding("cache-quality", 3, sequence, nonce);
            assert!(cache.reload_admitted_v2_if_current(&cx, &old, &fast, Some(&quality)).is_err());
        }
        let foreign = binding("other-fast", 2, 6, 8);
        let quality = binding("cache-quality", 3, 6, 8);
        assert!(cache.reload_admitted_v2_if_current(&cx, &old, &foreign, Some(&quality)).is_err());
        assert!(Arc::ptr_eq(&old, &cache.current()));
    });
}
