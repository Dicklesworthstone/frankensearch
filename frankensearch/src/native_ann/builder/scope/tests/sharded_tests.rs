//! The same identified providers/source fixture as the ordinary scoped route,
//! now through real partitioned builders, global Quill and live replacement.
use super::*;

use crate::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
use crate::native_ann::builder::sharded::NativeBuiltShardedHybridIndex;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use crate::native_ann::builder::sharded::NativeShardedHybridReopenLimits;
use crate::native_ann::{NativeShardedResult, NativeShardedSearchPhase};

async fn partitioned(
    cx: &Cx,
    path: &std::path::Path,
    ann: bool,
    size: usize,
    fast: &Arc<Provider>,
    quality: Option<&Arc<Provider>>,
) -> NativeBuiltShardedHybridIndex {
    let mut builder = NativeIndexBuilder::new(path, generation(1), fast.clone())
        .unwrap()
        .add_documents(documents())
        .with_fast_storage(
            NativeBuildPrecision::F16,
            if ann {
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams {
                        ef_search: 1,
                        ..HnswParams::default()
                    },
                    seed: 17,
                }
            } else {
                NativeBuildRetrieval::Exact
            },
        );
    if let Some(quality) = quality {
        builder = builder.with_quality_embedder(quality.clone()).unwrap();
    }
    builder.build_sharded_hybrid(cx, size).await.unwrap()
}

fn rows(index: &NativeBuiltShardedHybridIndex, hits: &[NativeShardedResult]) {
    for hit in hits {
        assert!(hit.result.index.is_none());
        for (set, location) in [
            (Some(index.vectors().fast()), hit.fast_row),
            (index.vectors().quality(), hit.quality_row),
        ] {
            if let Some(row) = location {
                let owner = &set.unwrap().shard(row.shard).unwrap().owner;
                assert_eq!(
                    owner.doc_id_at(usize::try_from(row.physical_row).unwrap()).unwrap(),
                    hit.result.doc_id
                );
            }
        }
    }
}

fn page(hits: &[NativeShardedResult]) -> serde_json::Value {
    serde_json::json!(hits.iter().map(|hit| {
        serde_json::json!({
            "result": hit.result,
            "fast_row": hit.fast_row.map(|row| (row.shard, row.physical_row)),
            "quality_row": hit.quality_row.map(|row| (row.shard, row.physical_row)),
        })
    }).collect::<Vec<_>>())
}

#[test]
fn selective_scopes_fill_each_lane_before_global_sharded_fusion() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            for size in [1, 7] {
                let root = tempfile::tempdir().unwrap();
                let fast = Arc::new(Provider::new("scoped-shards-fast", 2));
                let quality = Arc::new(Provider::new("scoped-shards-quality", 3));
                let index = partitioned(&cx, &root.path().join("index"), ann, size, &fast, Some(&quality)).await;
                let scope = index.scope(&cx, |doc| Ok(public(doc))).unwrap();
                assert_eq!(scope.len(), 4);
                assert!(!scope.is_empty());
                assert!(scope.document("private-00").is_none());
                let lexical = index.lexical().search_candidates(&cx, "needle", 28).await.unwrap();
                assert_eq!(lexical.results().len(), 28);
                assert!(lexical.results()[..3].iter().all(|hit| hit.doc_id.starts_with("private-")));
                let unscoped = index.vectors().search_fast(&cx, "needle", 3).await.unwrap();
                assert!(unscoped.iter().all(|hit| hit.doc_id.starts_with("private-")));
                for hits in [
                    scope.search(&cx, "needle", 3).await.unwrap(),
                    scope.search_refined(&cx, "needle", 3).await.unwrap(),
                    scope.search_quality(&cx, "needle", 3).await.unwrap(),
                ] {
                    assert_eq!(hits.len(), 3, "post-filtering the original top-k is empty");
                    rows(&index, &hits);
                    for hit in &hits {
                        assert!(scope.document(hit.result.doc_id.as_str()).is_some());
                        assert!(hit.result.metadata.is_some());
                        let raw = lexical.results().iter().find(|raw| raw.doc_id == hit.result.doc_id).unwrap();
                        assert_eq!(hit.result.lexical_score.unwrap().to_bits(), raw.score.to_bits());
                    }
                }
            }
        }
    });
}

#[test]
fn independent_scoped_quality_keeps_its_own_partition_coordinates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("scoped-fast", 2));
        let quality = Arc::new(Provider::new("scoped-quality", 3));
        let index = partitioned(&cx, &root.path().join("index"), true, 1, &fast, Some(&quality)).await;
        let scope = index.scope(&cx, |doc| Ok(public(doc))).unwrap();
        let mut query = scope.progressive(&cx, "absentterm", 1).unwrap();
        let Some(NativeShardedSearchPhase::Initial { results, candidates }) = query.next_phase().await.unwrap() else {
            panic!("initial");
        };
        assert_eq!(results[0].result.doc_id, "a-fast");
        assert_eq!(candidates.fast, 3);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        let Some(NativeShardedSearchPhase::Refined { results, candidates }) = query.next_phase().await.unwrap() else {
            panic!("refined");
        };
        assert_eq!(results[0].result.doc_id, "d-quality");
        assert!(results[0].fast_row.is_none());
        assert_eq!(results[0].quality_row.unwrap().shard, 3);
        assert_eq!(results[0].result.quality_score, Some(1.0));
        assert_eq!(candidates.quality, 3);
        rows(&index, &results);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        let quality_page = scope.search_quality(&cx, "absentterm", 1).await.unwrap();
        assert_eq!(quality_page[0].result.doc_id, "d-quality");
        assert!(quality_page[0].fast_row.is_none());
        assert!(quality_page[0].result.fast_score.is_none());
        rows(&index, &quality_page);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 2);
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn all_source_scope_matches_strictly_reopened_sharded_queries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("scoped-fast", 2));
            let quality = Arc::new(Provider::new("scoped-quality", 3));
            let directory = root.path().join("index");
            let built = partitioned(&cx, &directory, ann, 3, &fast, Some(&quality)).await;
            let receipt = built.seal_for_reopen(&cx).unwrap();
            drop(built);
            let index = NativeBuiltShardedHybridIndex::open_selected(
                &cx, &directory, &receipt, fast, Some(quality), NativeShardedHybridReopenLimits::default(),
            ).await.unwrap();
            let scope = index.scope(&cx, |_| Ok(true)).unwrap();
            for k in [0, 1, 5, 40] {
                for (scoped, original) in [
                    (scope.search(&cx, "needle", k).await, index.search(&cx, "needle", k).await),
                    (scope.search_refined(&cx, "needle", k).await, index.search_refined(&cx, "needle", k).await),
                    (scope.search_quality(&cx, "needle", k).await, index.search_quality(&cx, "needle", k).await),
                ] {
                    let scoped = scoped.unwrap();
                    assert_eq!(page(&scoped), page(&original.unwrap()));
                    rows(&index, &scoped);
                }
            }
        }
    });
}

#[test]
fn empty_scope_still_admits_identity_and_errors_never_widen_scope() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("scoped-fast", 2));
        let quality = Arc::new(Provider::new("scoped-quality", 3));
        let index = partitioned(&cx, &root.path().join("index"), false, 3, &fast, Some(&quality)).await;
        let empty = index.scope(&cx, |_| Ok(false)).unwrap();
        assert!(empty.is_empty());
        assert!(empty.search_refined(&cx, "needle", 10).await.unwrap().is_empty());
        assert!(empty.search_quality(&cx, "needle", 10).await.unwrap().is_empty());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        quality.advertise_foreign.store(true, Ordering::SeqCst);
        assert!(empty.progressive(&cx, "needle", 0).is_err());
        quality.advertise_foreign.store(false, Ordering::SeqCst);
        let mut calls = 0;
        let failed = index.scope(&cx, |_| {
            calls += 1;
            if calls == 5 { Err(invalid("test.policy", "failed", "refuse")) } else { Ok(true) }
        });
        assert!(failed.is_err());
        assert_eq!(calls, 5, "failure lies beyond the first partition");
        assert!(matches!(index.scope(&cx, |_| {
            cx.set_cancel_requested(true);
            Ok(true)
        }), Err(SearchError::Cancelled { .. })));
        cx.set_cancel_requested(false);
        let fast_only = partitioned(&cx, &root.path().join("fast-only"), false, 3, &fast, None).await;
        assert!(fast_only.scope(&cx, |_| Ok(false)).unwrap().search_quality(&cx, "needle", 0).await.is_err());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn live_repartition_cannot_rebind_scoped_phases_or_rerank_sources() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("scoped-fast", 2));
        let quality = Arc::new(Provider::new("scoped-quality", 3));
        let index = partitioned(&cx, &root.path().join("old"), true, 7, &fast, Some(&quality)).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let calls = AtomicUsize::new(0);
        let scope = old.index().scope(&cx, |doc| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(public(doc))
        }).unwrap();
        let probe = RerankProbe { inputs: Mutex::new(Vec::new()) };
        let mut query = scope.progressive_with_reranker(&cx, "needle", 1, &probe, 4).unwrap();
        assert!(matches!(query.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { .. })));
        assert!(probe.inputs.lock().unwrap().is_empty());
        let candidate = old.begin_update(&cx, root.path().join("next"), generation(2)).unwrap()
            .upsert_document(IndexableDocument::new("d-quality", "needle private").with_metadata("scope", "private"))
            .upsert_document(IndexableDocument::new("fresh", "needle fast").with_metadata("scope", "public"))
            .build(&cx, 3).await.unwrap();
        let installed = live.install(&cx, &candidate).await.unwrap();
        assert_ne!(old.index().vectors().partitions().len(), installed.index().vectors().partitions().len());
        assert!(matches!(query.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Refined { .. })));
        assert!(probe.inputs.lock().unwrap().is_empty());
        let Some(NativeShardedSearchPhase::Reranked { results, evaluated, .. }) = query.next_phase().await.unwrap() else {
            panic!("reranked");
        };
        assert_eq!(evaluated, 4);
        assert_eq!(results[0].result.doc_id, "d-quality");
        rows(old.index(), &results);
        for input in probe.inputs.lock().unwrap().iter() {
            assert_eq!(input.text, scope.document(&input.doc_id).unwrap().content);
            assert!(!input.doc_id.starts_with("private-"));
        }
        let current_scope = installed.index().scope(&cx, |doc| Ok(public(doc))).unwrap();
        assert!(scope.document("d-quality").is_some());
        assert!(scope.document("fresh").is_none());
        assert!(current_scope.document("d-quality").is_none());
        assert!(current_scope.document("fresh").is_some());
        let hits = current_scope.search_refined(&cx, "needle", 4).await.unwrap();
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|hit| current_scope.document(hit.result.doc_id.as_str()).is_some()));
        rows(installed.index(), &hits);
        assert_eq!(calls.load(Ordering::SeqCst), 28);
    });
}

#[test]
fn failed_and_dropped_quality_phases_preserve_only_the_scoped_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("scoped-fast", 2));
        let quality = Arc::new(Provider::new("scoped-quality", 3));
        let index = partitioned(&cx, &root.path().join("index"), false, 3, &fast, Some(&quality)).await;
        let scope = index.scope(&cx, |doc| Ok(public(doc))).unwrap();
        quality.fail.store(true, Ordering::SeqCst);
        assert!(scope.search_refined(&cx, "needle", 1).await.is_err());
        let mut query = scope.progressive(&cx, "needle", 1).unwrap();
        let Some(NativeShardedSearchPhase::Initial { results, .. }) = query.next_phase().await.unwrap() else {
            panic!("initial");
        };
        let before = page(&results);
        let Some(NativeShardedSearchPhase::RefinementFailed { initial_results, .. }) = query.next_phase().await.unwrap() else {
            panic!("explicit quality failure");
        };
        assert_eq!(page(&initial_results), before);
        quality.fail.store(false, Ordering::SeqCst);
        quality.hold.store(true, Ordering::SeqCst);
        let mut query = scope.progressive(&cx, "needle", 1).unwrap();
        assert!(query.next_phase().await.unwrap().is_some());
        let drops = quality.drops.load(Ordering::SeqCst);
        let mut waiting = Box::pin(query.next_phase());
        assert!(waiting.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        drop(waiting);
        assert_eq!(quality.drops.load(Ordering::SeqCst), drops + 1);
        assert!(query.is_finished());
        assert!(query.next_phase().await.unwrap().is_none());
    });
}

#[test]
fn internal_vector_scope_cannot_silently_leave_keyword_candidates_unrestricted() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("scoped-fast", 2));
        let index = partitioned(&cx, &root.path().join("index"), false, 3, &fast, None).await;
        let allowed = index.vectors().documents().filter(|doc| public(doc)).map(|doc| doc.id.clone()).collect();
        // Fault injection: bypass the public source-scope constructor and pair
        // restricted vector membership with the ordinary unfiltered lexical arm.
        let mut mixed = index.progressive(&cx, "needle", 1).unwrap().with_allowed_documents(&allowed).unwrap();
        assert!(matches!(mixed.next_phase().await,
            Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.shards.scope.lexical"));
        assert!(mixed.is_finished());
        let scope = index.scope(&cx, |doc| Ok(public(doc))).unwrap();
        let recovered = scope.search(&cx, "needle", 1).await.unwrap();
        assert_eq!(recovered.len(), 1);
        assert!(scope.document(recovered[0].result.doc_id.as_str()).is_some());
        rows(&index, &recovered);
        assert!(mixed.with_allowed_documents(&allowed).is_err());
    });
}
