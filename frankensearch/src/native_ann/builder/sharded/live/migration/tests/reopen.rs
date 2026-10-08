use super::*;
use crate::native_ann::builder::sharded::NativeShardedHybridReopenLimits;
use frankensearch_core::generation::GenerationComponentReceiptV1;

fn targets() -> (Arc<Provider>, Arc<Provider>) {
    (
        Arc::new(Provider::new("reopen-fast", 4)),
        Arc::new(Provider::new("reopen-quality", 5)),
    )
}

fn requested(
    cx: &Cx,
    base: &NativeShardedSnapshot,
    path: &Path,
    generation: ArtifactGenerationIdentityV1,
    fast: &Arc<Provider>,
    quality: &Arc<Provider>,
) -> NativeLiveShardedMigration {
    base.begin_model_migration(cx, path, generation, fast.clone(), Some(quality.clone()))
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F16, graph(17))
        .with_quality_storage(NativeBuildPrecision::F32, graph(29))
        .unwrap()
}

async fn seal_target(
    cx: &Cx,
    base: &NativeShardedSnapshot,
    path: &Path,
    generation: ArtifactGenerationIdentityV1,
) -> GenerationComponentReceiptV1 {
    let (fast, quality) = targets();
    let candidate = requested(cx, base, path, generation, &fast, &quality)
        .build(cx, 3)
        .await
        .unwrap();
    candidate.index().seal_for_reopen(cx).unwrap()
}

fn assert_no_inference(fast: &Provider, quality: &Provider) {
    for provider in [fast, quality] {
        assert!(provider.submitted().is_empty());
        assert_eq!(provider.queries.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn selected_migration_reopens_every_owner_after_drop_without_inference() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, old_fast, old_quality) = initial(&cx, root.path(), sources(), true).await;
        let old = live.snapshot(&cx).await.unwrap();
        let old_path = old.index().vectors().directory().to_path_buf();
        let old_receipt = old.index().seal_for_reopen(&cx).unwrap();
        let path = root.path().join("sealed-upgrade");
        let receipt = seal_target(&cx, &old, &path, generation(2)).await;
        let old_image = image(&old_path);
        let target_image = image(&path);
        // Release all constructed index handles. This is object-drop/reopen
        // coverage, not a claim of an independently executed fresh process.
        drop(old);
        drop(live);
        let limits = NativeShardedHybridReopenLimits::default();
        let reopened_old = NativeBuiltShardedHybridIndex::open_selected(
            &cx,
            &old_path,
            &old_receipt,
            old_fast,
            Some(old_quality),
            limits,
        )
        .await
        .unwrap();
        let live = NativeLiveShardedHybridIndex::new(&cx, reopened_old).unwrap();
        let base = live.snapshot(&cx).await.unwrap();
        // The old ordinary reopen remains strict: model migration must be explicit.
        assert!(
            base.prepare_selected(&cx, &path, &receipt, limits)
                .await
                .is_err()
        );
        let (fast, quality) = targets();
        let candidate = requested(&cx, &base, &path, generation(2), &fast, &quality)
            .open_selected(&cx, &receipt, limits)
            .await
            .unwrap();
        assert_no_inference(&fast, &quality);
        assert_eq!(candidate.index().vectors().partitions().len(), 3);
        assert_eq!(candidate.index().lexical().doc_count().unwrap(), 7);
        let installed = live.install(&cx, &candidate).await.unwrap();
        assert_eq!(installed.generation(), generation(2));
        let page = live.search_refined(&cx, "legacytoken", 7).await.unwrap();
        assert_rows(&page.snapshot, &page.results);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        assert!(fast.submitted().is_empty());
        assert!(quality.submitted().is_empty());
        assert_eq!(image(&old_path), old_image);
        assert_eq!(image(&path), target_image);
    });
}

#[test]
fn valid_foreign_receipts_cannot_substitute_any_retained_source_field() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let old_image = image(base.index().vectors().directory());
        for field in ["id", "content", "title", "metadata", "count"] {
            let mut documents = sources();
            let last = documents.last_mut().unwrap();
            match field {
                "id" => "unrelated-id".clone_into(&mut last.id),
                "content" => "unrelated-body".clone_into(&mut last.content),
                "title" => last.title = Some("unrelated-title".to_owned()),
                "metadata" => {
                    last.metadata
                        .insert("version".to_owned(), "unrelated".to_owned());
                }
                "count" => {
                    let _ = documents.pop();
                }
                _ => unreachable!(),
            }
            let (built_fast, built_quality) = targets();
            let path = root.path().join(field);
            let built = NativeIndexBuilder::new(&path, generation(2), built_fast)
                .unwrap()
                .with_quality_embedder(built_quality)
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F16, graph(17))
                .with_quality_storage(NativeBuildPrecision::F32, graph(29))
                .unwrap()
                .add_documents(documents)
                .build_sharded_hybrid(&cx, 3)
                .await
                .unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            drop(built);
            let target_image = image(&path);
            let (fast, quality) = targets();
            let error = requested(&cx, &base, &path, generation(2), &fast, &quality)
                .open_selected(&cx, &receipt, NativeShardedHybridReopenLimits::default())
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.sharded_live.migration.source"));
            assert_no_inference(&fast, &quality);
            assert!(Arc::ptr_eq(&base.index, &live.snapshot(&cx).await.unwrap().index));
            assert_eq!(image(&path), target_image);
            assert_eq!(image(base.index().vectors().directory()), old_image);
        }
    });
}

#[test]
fn selected_migration_requires_exact_generation_and_every_storage_parameter() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let path = root.path().join("target");
        let receipt = seal_target(&cx, &base, &path, generation(2)).await;
        let (fast, quality) = targets();
        let limits = NativeShardedHybridReopenLimits::default();
        for mismatch in [
            "nonce",
            "fast-precision",
            "quality-precision",
            "exact",
            "seed",
            "quality-seed",
            "m",
            "m0",
            "construction",
            "search",
        ] {
            let target_generation = if mismatch == "nonce" {
                ArtifactGenerationIdentityV1::new(2, [0x58; 16]).unwrap()
            } else {
                generation(2)
            };
            let mut intent = requested(&cx, &base, &path, target_generation, &fast, &quality);
            match mismatch {
                "nonce" => {}
                "fast-precision" => {
                    intent = intent.with_fast_storage(NativeBuildPrecision::F32, graph(17));
                }
                "quality-precision" => {
                    intent = intent
                        .with_quality_storage(NativeBuildPrecision::F16, graph(29))
                        .unwrap();
                }
                "exact" => {
                    intent = intent.with_fast_storage(
                        NativeBuildPrecision::F16,
                        NativeBuildRetrieval::Exact,
                    );
                }
                "seed" => {
                    intent = intent.with_fast_storage(NativeBuildPrecision::F16, graph(18));
                }
                "quality-seed" => {
                    intent = intent
                        .with_quality_storage(NativeBuildPrecision::F32, graph(30))
                        .unwrap();
                }
                parameter => {
                    let mut params = HnswParams::default();
                    match parameter {
                        "m" => params.m += 1,
                        "m0" => params.m0 += 1,
                        "construction" => params.ef_construction += 1,
                        "search" => params.ef_search += 1,
                        _ => unreachable!(),
                    }
                    intent = intent.with_fast_storage(
                        NativeBuildPrecision::F16,
                        NativeBuildRetrieval::Hnsw { params, seed: 17 },
                    );
                }
            }
            let error = intent.open_selected(&cx, &receipt, limits).await.unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.sharded_live.migration.target"));
            assert_no_inference(&fast, &quality);
        }
        let candidate = requested(&cx, &base, &path, generation(2), &fast, &quality)
            .open_selected(&cx, &receipt, limits)
            .await
            .unwrap();
        assert_no_inference(&fast, &quality);
        live.install(&cx, &candidate).await.unwrap();
    });
}

#[test]
fn selected_migration_refuses_missing_quality_and_same_name_wrong_producer() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let path = root.path().join("target");
        let receipt = seal_target(&cx, &base, &path, generation(2)).await;
        let limits = NativeShardedHybridReopenLimits::default();
        let (fast, quality) = targets();
        assert!(
            base.begin_model_migration(&cx, &path, generation(2), fast.clone(), None)
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F16, graph(17))
                .open_selected(&cx, &receipt, limits)
                .await
                .is_err()
        );
        for change_quality in [false, true] {
            let mut changed = Provider::new(
                if change_quality {
                    "reopen-quality"
                } else {
                    "reopen-fast"
                },
                if change_quality { 5 } else { 4 },
            );
            "different-producer".clone_into(&mut changed.identity.producer.backend);
            let changed = Arc::new(changed);
            let target_fast = if change_quality {
                fast.clone()
            } else {
                changed.clone()
            };
            let target_quality = if change_quality {
                changed.clone()
            } else {
                quality.clone()
            };
            let error = requested(&cx, &base, &path, generation(2), &target_fast, &target_quality)
                .open_selected(&cx, &receipt, limits)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.builder.producer"));
            assert_no_inference(&target_fast, &target_quality);
        }
        assert_no_inference(&fast, &quality);
        assert!(Arc::ptr_eq(&base.index, &live.snapshot(&cx).await.unwrap().index));
    });
}

#[test]
fn selected_migration_rejects_limits_cancellation_and_corrupt_final_partition() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let path = root.path().join("target");
        let (build_fast, build_quality) = targets();
        let built = requested(&cx, &base, &path, generation(2), &build_fast, &build_quality)
            .build(&cx, 3)
            .await
            .unwrap();
        let last = built.index().vectors().partitions().last().unwrap();
        let last_vector = last.fast().vector_path().to_path_buf();
        let last_graph = last.quality().unwrap().graph_path().unwrap().to_path_buf();
        let receipt = built.index().seal_for_reopen(&cx).unwrap();
        drop(built);
        let old_image = image(base.index().vectors().directory());
        let target_image = image(&path);
        let limits = NativeShardedHybridReopenLimits::default();
        let (fast, quality) = targets();
        for bound in ["shards", "documents", "bytes"] {
            let mut limited = limits;
            match bound {
                "shards" => limited.vectors.max_shards = 1,
                "documents" => limited.vectors.max_documents = 6,
                "bytes" => limited.vectors.max_artifact_bytes = 1,
                _ => unreachable!(),
            }
            assert!(
                requested(&cx, &base, &path, generation(2), &fast, &quality)
                    .open_selected(&cx, &receipt, limited)
                    .await
                    .is_err()
            );
        }
        let intent = requested(&cx, &base, &path, generation(2), &fast, &quality);
        cx.set_cancel_requested(true);
        let error = intent.open_selected(&cx, &receipt, limits).await.unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(error, SearchError::Cancelled { .. }));
        for file in [&last_vector, &last_graph] {
            let original = fs::read(file).unwrap();
            let mut damaged = original.clone();
            *damaged.last_mut().unwrap() ^= 1;
            fs::write(file, damaged).unwrap();
            let response = requested(&cx, &base, &path, generation(2), &fast, &quality)
                .open_selected(&cx, &receipt, limits)
                .await;
            // Restore only this test's artifact; no file is removed or repaired
            // by production code, and the candidate is not retained while damaged.
            fs::write(file, original).unwrap();
            assert!(response.is_err());
            assert!(Arc::ptr_eq(&base.index, &live.snapshot(&cx).await.unwrap().index));
        }
        assert_no_inference(&fast, &quality);
        assert_eq!(image(base.index().vectors().directory()), old_image);
        assert_eq!(image(&path), target_image);
    });
}

#[test]
fn selected_model_upgrade_cannot_overwrite_an_intervening_source_commit() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, _, _) = initial(&cx, root.path(), sources(), true).await;
        let base = live.snapshot(&cx).await.unwrap();
        let path = root.path().join("upgrade");
        let receipt = seal_target(&cx, &base, &path, generation(50)).await;
        let (fast, quality) = targets();
        let intent = requested(&cx, &base, &path, generation(50), &fast, &quality);
        let pending =
            intent.open_selected(&cx, &receipt, NativeShardedHybridReopenLimits::default());
        let source_update = base
            .begin_update(&cx, root.path().join("source-edit"), generation(2))
            .unwrap()
            .upsert_document(IndexableDocument::new("doc-6", "newly committed source"))
            .build(&cx, 2)
            .await
            .unwrap();
        let edited = live.install(&cx, &source_update).await.unwrap();
        let candidate = pending.await.unwrap();
        assert_no_inference(&fast, &quality);
        let error = live.install(&cx, &candidate).await.unwrap_err();
        assert!(matches!(error, SearchError::InvalidConfig { field, .. }
            if field == "native_ann.sharded_live.expected_current"));
        assert!(Arc::ptr_eq(&edited.index, &live.snapshot(&cx).await.unwrap().index));
        assert_eq!(
            edited.index().vectors().document("doc-6").unwrap().content,
            "newly committed source"
        );
        assert_eq!(
            base.index().vectors().document("doc-6").unwrap().content,
            "lastdoc"
        );
    });
}
