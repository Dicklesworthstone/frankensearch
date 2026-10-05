use super::*;
use asupersync::test_utils::run_test_with_cx;
use std::future::Future;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll, Waker};

use crate::native_ann::builder::NativeBuildPrecision;
use crate::native_ann::builder::tests::{Provider, Reply, documents, generation};
use frankensearch_index::native_hnsw::HnswParams;

fn model(name: &str, dimension: u32) -> Arc<Provider> {
    Arc::new(Provider::new(name, dimension, Reply::Correct))
}

fn configured(
    path: &Path,
    fast: &Arc<Provider>,
    quality: &Arc<Provider>,
    graph: bool,
) -> NativeIndexBuilder {
    let retrieval = if graph {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 101,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    NativeIndexBuilder::new(path, generation(), fast.clone())
        .unwrap()
        .with_quality_embedder(quality.clone())
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .with_quality_storage(NativeBuildPrecision::F16, retrieval)
        .unwrap()
        .with_batch_size(2)
        .unwrap()
        .with_max_batch_input_bytes(64)
        .unwrap()
        .add_documents(documents())
}

#[test]
fn sharded_build_partitions_one_source_cut_and_embeds_queries_once_per_tier() {
    run_test_with_cx(|cx| async move {
        for graph in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let fast = model("sharded-fast", 2);
            let quality = model("sharded-quality", 3);
            let built = configured(&directory.path().join("cohort"), &fast, &quality, graph)
                .build_sharded(&cx, 2)
                .await
                .unwrap();
            assert_eq!(built.document_count(), 5);
            assert_eq!(built.generation(), generation());
            assert_eq!(
                built
                    .partitions
                    .iter()
                    .map(|p| p.documents.len())
                    .collect::<Vec<_>>(),
                [2, 2, 1]
            );
            assert_eq!(
                built.documents().map(|d| d.id.as_str()).collect::<Vec<_>>(),
                ["a", "b", "c", "x", "z"]
            );
            for partition in built.partitions() {
                assert_eq!(
                    partition.fast().index().owner_witness().generation,
                    generation()
                );
                assert_eq!(
                    partition
                        .quality()
                        .unwrap()
                        .index()
                        .owner_witness()
                        .generation,
                    generation()
                );
                assert_eq!(partition.fast().graph_path().is_some(), graph);
                assert_eq!(partition.quality().unwrap().graph_path().is_some(), graph);
            }
            assert_eq!(built.document("b").unwrap().content, "vertical");
            assert_eq!(built.document("z").unwrap().content, "horizontal");
            assert!(built.document("absent").is_none());
            let hits = built.search_fast(&cx, "vertical", 5).await.unwrap();
            assert_eq!(
                fast.queries.load(Ordering::SeqCst),
                1,
                "not one embedding per shard"
            );
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            assert_eq!(hits.len(), 5);
            assert_eq!(hits[0].doc_id, "b");
            for hit in &hits {
                let owner = &built.partitions[hit.row.shard].fast.index.owner;
                assert_eq!(
                    owner
                        .row(usize::try_from(hit.row.physical_row).unwrap())
                        .unwrap()
                        .doc_id(),
                    hit.doc_id.as_str()
                );
            }
            let refined = built.search_quality(&cx, "vertical", 5).await.unwrap();
            assert_eq!(refined[0].doc_id, "b");
            assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        }
    });
}

#[test]
fn partition_preflight_rejects_bad_tail_sources_and_size_before_creating_files() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let fast = model("fast", 2);
        for (ordinal, docs, size, bytes) in [
            (0, documents(), 0, 64),
            (
                1,
                vec![
                    IndexableDocument::new("a", "x"),
                    IndexableDocument::new("a", "y"),
                ],
                1,
                64,
            ),
            (
                2,
                vec![
                    IndexableDocument::new("a", "ok"),
                    IndexableDocument::new("z", "too long"),
                ],
                1,
                2,
            ),
            (3, vec![IndexableDocument::new("", "empty-id")], 1, 64),
        ] {
            let path = parent.path().join(format!("rejected-{ordinal}"));
            let result = NativeIndexBuilder::new(&path, generation(), fast.clone())
                .unwrap()
                .with_max_batch_input_bytes(bytes)
                .unwrap()
                .add_documents(docs)
                .build_sharded(&cx, size)
                .await;
            assert!(result.is_err());
            assert!(!path.exists());
        }
        let path = parent.path().join("too-many");
        let docs = (0..=MAX_NATIVE_BUILD_SHARDS)
            .map(|i| IndexableDocument::new(format!("id-{i:05}"), "x"));
        assert!(
            NativeIndexBuilder::new(&path, generation(), fast)
                .unwrap()
                .add_documents(docs)
                .build_sharded(&cx, 1)
                .await
                .is_err()
        );
        assert!(!path.exists());
    });
}

#[test]
fn empty_cohort_keeps_one_identity_partition_and_starts_no_inference() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let fast = model("empty", 2);
        let built =
            NativeIndexBuilder::new(parent.path().join("empty"), generation(), fast.clone())
                .unwrap()
                .build_sharded(&cx, 10)
                .await
                .unwrap();
        assert_eq!(built.partitions().len(), 1);
        assert_eq!(built.document_count(), 0);
        assert!(built.documents().next().is_none());
        assert!(built.document("any").is_none());
        assert!(
            built
                .search_fast(&cx, "vertical", 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert!(built.search_quality(&cx, "vertical", 0).await.is_err());
    });
}

#[test]
fn a_later_required_tier_failure_and_dropped_build_never_return_partial_cohorts() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let fast = model("fast", 2);
        let quality = Arc::new(Provider::new("fails-second", 3, Reply::FailSecond));
        let path = parent.path().join("failed");
        assert!(
            configured(&path, &fast, &quality, false)
                .build_sharded(&cx, 2)
                .await
                .is_err()
        );
        assert!(path.join(partition_name(0)).is_dir());
        assert!(!path.join(SNAPSHOT_FILE).exists());
        assert!(!path.join(partition_name(2)).exists());

        let path = parent.path().join("dropped");
        let pending = Arc::new(Provider::new("pending", 2, Reply::Pending));
        let builder = NativeIndexBuilder::new(&path, generation(), pending)
            .unwrap()
            .add_documents(documents());
        let mut future = Box::pin(builder.build_sharded(&cx, 2));
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        drop(future);
        assert!(path.join(partition_name(0)).is_dir());
        assert!(!path.join(partition_name(1)).exists());
        assert!(!path.join(SNAPSHOT_FILE).exists());
        assert!(
            NativeIndexBuilder::new(&path, generation(), fast)
                .unwrap()
                .build_sharded(&cx, 2)
                .await
                .is_err(),
            "partial roots are never adopted"
        );
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn image(path: &Path) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
    let mut output = std::collections::BTreeMap::new();
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let bytes = fs::read(&path).unwrap();
                output.insert(path, bytes);
            }
        }
    }
    output
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn exact_receipt_restart_preserves_sources_graph_policy_and_physical_coordinates() {
    run_test_with_cx(|cx| async move {
        for graph in [false, true] {
            let parent = tempfile::tempdir().unwrap();
            let path = parent.path().join("cohort");
            let fast = model("fast", 2);
            let quality = model("quality", 3);
            let built = configured(&path, &fast, &quality, graph)
                .add_document(
                    IndexableDocument::new("metadata", "!!!")
                        .with_title("title")
                        .with_metadata("lang", "β"),
                )
                .build_sharded(&cx, 2)
                .await
                .unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            let before = image(&path);
            let expected = built.search_quality(&cx, "vertical", 20).await.unwrap();
            let query_counts = (
                fast.queries.load(Ordering::SeqCst),
                quality.queries.load(Ordering::SeqCst),
            );
            let reopened = NativeBuiltShardedIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
                NativeShardedReopenLimits::default(),
            )
            .unwrap();
            assert_eq!(
                query_counts,
                (
                    fast.queries.load(Ordering::SeqCst),
                    quality.queries.load(Ordering::SeqCst)
                ),
                "reopen never runs inference"
            );
            assert_eq!(
                reopened.search_quality(&cx, "vertical", 20).await.unwrap(),
                expected
            );
            let document = reopened.document("metadata").unwrap();
            assert_eq!(document.content, "!!!");
            assert_eq!(document.title.as_deref(), Some("title"));
            assert_eq!(document.metadata["lang"], "β");
            assert_eq!(image(&path), before);
            assert!(built.seal_for_reopen(&cx).is_err());
            assert_eq!(image(&path), before);
        }
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn incomplete_or_corrupt_selected_partition_is_never_skipped_or_rebuilt() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("cohort");
        let fast = model("fast", 2);
        let quality = model("quality", 3);
        let built = configured(&path, &fast, &quality, true)
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let before = image(&path);
        for name in [
            "quality.fsvi",
            "fast.fshnsw",
            "native.source.jsonl",
            "native.snapshot.json",
        ] {
            let target = path.join(partition_name(2)).join(name);
            let saved = fs::read(&target).unwrap();
            fs::write(&target, b"corrupt").unwrap();
            assert!(
                NativeBuiltShardedIndex::open_selected(
                    &cx,
                    &path,
                    &receipt,
                    fast.clone(),
                    Some(quality.clone()),
                    NativeShardedReopenLimits::default()
                )
                .is_err(),
                "{name}"
            );
            assert_eq!(
                fs::read(&target).unwrap(),
                b"corrupt",
                "no repair side effects"
            );
            assert_eq!(
                built.search_fast(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            fs::write(target, saved).unwrap();
        }
        let missing = path.join(partition_name(2)).join("quality.fsvi");
        let retained = parent.path().join("saved-quality");
        fs::rename(&missing, &retained).unwrap();
        assert!(
            NativeBuiltShardedIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
                NativeShardedReopenLimits::default()
            )
            .is_err()
        );
        fs::rename(retained, missing).unwrap();
        assert_eq!(image(&path), before);
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn aggregate_limits_and_provider_mismatch_precede_vector_open() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("cohort");
        let fast = model("fast", 2);
        let quality = model("quality", 3);
        let built = configured(&path, &fast, &quality, false)
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        // Missing first vector distinguishes descriptor preflight from ordinary
        // loading: each refusal below must be semantic, not NotFound.
        let original = path.join(partition_name(0)).join("fast.fsvi");
        let saved = parent.path().join("retained-first-vector");
        fs::rename(&original, &saved).unwrap();
        for limits in [
            NativeShardedReopenLimits {
                max_shards: 2,
                ..NativeShardedReopenLimits::default()
            },
            NativeShardedReopenLimits {
                max_documents: 4,
                ..NativeShardedReopenLimits::default()
            },
            NativeShardedReopenLimits {
                max_artifact_bytes: receipt.byte_len + 1,
                ..NativeShardedReopenLimits::default()
            },
        ] {
            let error = NativeBuiltShardedIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
                limits,
            )
            .err()
            .unwrap();
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
        }
        let error = NativeBuiltShardedIndex::open_selected(
            &cx,
            &path,
            &receipt,
            model("foreign-fast", 2),
            Some(quality),
            NativeShardedReopenLimits::default(),
        )
        .err()
        .unwrap();
        assert!(matches!(error, SearchError::InvalidConfig { .. }));
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        fs::rename(saved, original).unwrap();
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn reordered_omitted_and_substituted_partitions_are_rejected_by_complete_selection() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("cohort");
        let fast = model("fast", 2);
        let quality = model("quality", 3);
        let built = configured(&path, &fast, &quality, false)
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let descriptor = path.join(SNAPSHOT_FILE);
        let original = fs::read(&descriptor).unwrap();
        for fault in ["reordered", "omitted", "duplicate", "generation"] {
            let mut json: serde_json::Value = serde_json::from_slice(&original).unwrap();
            match fault {
                "reordered" => json["partitions"].as_array_mut().unwrap().swap(0, 1),
                "omitted" => {
                    json["partitions"].as_array_mut().unwrap().pop();
                }
                "duplicate" => {
                    json["partitions"][1] = json["partitions"][0].clone();
                }
                _ => json["generation"]["sequence"] = serde_json::json!(10),
            }
            let changed = serde_json::to_vec(&json).unwrap();
            fs::write(&descriptor, &changed).unwrap();
            // Even a separately trusted inconsistent root cannot join these
            // children. This is not a claim to forge the original selection.
            let inconsistent = Artifact::from_bytes(&changed).receipt();
            assert!(
                NativeBuiltShardedIndex::open_selected(
                    &cx,
                    &path,
                    &inconsistent,
                    fast.clone(),
                    Some(quality.clone()),
                    NativeShardedReopenLimits::default()
                )
                .is_err(),
                "{fault}"
            );
            assert!(
                NativeBuiltShardedIndex::open_selected(
                    &cx,
                    &path,
                    &receipt,
                    fast.clone(),
                    Some(quality.clone()),
                    NativeShardedReopenLimits::default()
                )
                .is_err()
            );
        }
        fs::write(descriptor, original).unwrap();
        assert!(
            NativeBuiltShardedIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast,
                Some(quality),
                NativeShardedReopenLimits::default()
            )
            .is_ok()
        );
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn cancellation_and_sealed_ancestor_refusal_leave_original_bytes_untouched() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("cohort");
        let fast = model("fast", 2);
        let quality = model("quality", 3);
        let built = configured(&path, &fast, &quality, false)
            .build_sharded(&cx, 2)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let before = image(&path);
        cx.set_cancel_requested(true);
        assert!(matches!(
            NativeBuiltShardedIndex::open_selected(
                &cx,
                &path,
                &receipt,
                fast.clone(),
                Some(quality.clone()),
                NativeShardedReopenLimits::default()
            ),
            Err(SearchError::Cancelled { .. })
        ));
        let cancelled_path = parent.path().join("cancelled");
        assert!(
            configured(&cancelled_path, &fast, &quality, false)
                .build_sharded(&cx, 2)
                .await
                .is_err()
        );
        cx.set_cancel_requested(false);
        assert!(!cancelled_path.exists());
        let nested = path.join("new-child");
        assert!(
            configured(&nested, &fast, &quality, false)
                .build_sharded(&cx, 2)
                .await
                .is_err()
        );
        assert!(!nested.exists());
        assert_eq!(image(&path), before);
    });
}
