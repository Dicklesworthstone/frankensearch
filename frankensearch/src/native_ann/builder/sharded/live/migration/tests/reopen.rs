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
            assert!(Arc::ptr_eq(
                &base.index,
                &live.snapshot(&cx).await.unwrap().index
            ));
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
                    intent = intent
                        .with_fast_storage(NativeBuildPrecision::F16, NativeBuildRetrieval::Exact);
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
            let error = intent
                .open_selected(&cx, &receipt, limits)
                .await
                .unwrap_err();
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
            let error = requested(
                &cx,
                &base,
                &path,
                generation(2),
                &target_fast,
                &target_quality,
            )
            .open_selected(&cx, &receipt, limits)
            .await
            .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.builder.producer"));
            assert_no_inference(&target_fast, &target_quality);
        }
        assert_no_inference(&fast, &quality);
        assert!(Arc::ptr_eq(
            &base.index,
            &live.snapshot(&cx).await.unwrap().index
        ));
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
        let built = requested(
            &cx,
            &base,
            &path,
            generation(2),
            &build_fast,
            &build_quality,
        )
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
        let error = intent
            .open_selected(&cx, &receipt, limits)
            .await
            .unwrap_err();
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
            assert!(Arc::ptr_eq(
                &base.index,
                &live.snapshot(&cx).await.unwrap().index
            ));
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
        assert!(Arc::ptr_eq(
            &edited.index,
            &live.snapshot(&cx).await.unwrap().index
        ));
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

#[test]
fn selected_quality_completion_repartitions_with_fast_cache_and_mixed_storage() {
    run_test_with_cx(|cx| async move {
        for fast_hnsw in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("completion-fast", 2));
            let cached = Arc::new(frankensearch_embed::CachedEmbedder::new(fast.clone(), 32));
            let quality = Arc::new(Provider::new("completion-quality", 3));
            let fast_retrieval = if fast_hnsw {
                graph(57)
            } else {
                NativeBuildRetrieval::Exact
            };
            let quality_retrieval = if fast_hnsw {
                NativeBuildRetrieval::Exact
            } else {
                graph(91)
            };
            let initial =
                NativeIndexBuilder::new(root.path().join("initial"), generation(1), cached.clone())
                    .unwrap()
                    .with_fast_storage(NativeBuildPrecision::F16, fast_retrieval)
                    .add_documents(sources())
                    .build_sharded_hybrid(&cx, 2)
                    .await
                    .unwrap();
            let live = NativeLiveShardedHybridIndex::new(&cx, initial).unwrap();
            let base = live.snapshot(&cx).await.unwrap();
            let query = "warm-only horizontal query";
            live.search(&cx, query, 7).await.unwrap();
            assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
            let path = root.path().join("completed");
            let built = base
                .begin_quality_completion(&cx, &path, generation(2), quality.clone())
                .unwrap()
                .with_quality_storage(NativeBuildPrecision::F32, quality_retrieval)
                .unwrap()
                .build(&cx, 3)
                .await
                .unwrap();
            let receipt = built.index().seal_for_reopen(&cx).unwrap();
            drop(built);
            fast.batches.lock().unwrap().clear();
            quality.batches.lock().unwrap().clear();
            let cache_before = cached.cache_stats();
            let old_image = image(base.index().vectors().directory());
            let target_image = image(&path);
            assert!(
                base.prepare_selected(
                    &cx,
                    &path,
                    &receipt,
                    NativeShardedHybridReopenLimits::default()
                )
                .await
                .is_err()
            );
            let candidate = base
                .prepare_quality_completion_selected(
                    &cx,
                    &path,
                    generation(2),
                    &receipt,
                    quality.clone(),
                    (NativeBuildPrecision::F32, quality_retrieval),
                    NativeShardedHybridReopenLimits::default(),
                )
                .await
                .unwrap();
            let original_fast = &base.index().vectors().partitions()[0].fast;
            assert_eq!(base.index().vectors().partitions().len(), 4);
            assert_eq!(candidate.index().vectors().partitions().len(), 3);
            assert_eq!(candidate.index().vectors().document_count(), 7);
            for partition in candidate.index().vectors().partitions() {
                assert!(Arc::ptr_eq(
                    &original_fast.embedder,
                    &partition.fast().embedder
                ));
                assert_eq!(partition.fast().precision, NativeBuildPrecision::F16);
                assert_eq!(partition.fast().graph_receipt.is_some(), fast_hnsw);
                assert_eq!(
                    partition.quality().unwrap().precision,
                    NativeBuildPrecision::F32
                );
                assert_eq!(
                    partition.quality().unwrap().graph_receipt.is_some(),
                    !fast_hnsw
                );
            }
            assert_eq!(cached.cache_stats(), cache_before);
            assert!(fast.submitted().is_empty());
            assert!(quality.submitted().is_empty());
            assert!(Arc::ptr_eq(
                &base.index,
                &live.snapshot(&cx).await.unwrap().index
            ));
            assert_eq!(image(&path), target_image);
            let selected = live.install(&cx, &candidate).await.unwrap();
            let page = live.search_refined(&cx, query, 7).await.unwrap();
            assert_eq!(page.snapshot.generation(), generation(2));
            assert_rows(&page.snapshot, &page.results);
            assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
            assert!(cached.cache_stats().hits > cache_before.hits);
            assert!(base.index().vectors().quality().is_none());
            assert_eq!(image(base.index().vectors().directory()), old_image);
            let error = selected
                .prepare_quality_completion_selected(
                    &cx,
                    root.path().join("not-read"),
                    generation(3),
                    &receipt,
                    quality.clone(),
                    (NativeBuildPrecision::F32, quality_retrieval),
                    NativeShardedHybridReopenLimits::default(),
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "native_ann.sharded_live.quality_completion.topology"));
        }
    });
}

#[test]
fn selected_quality_completion_refuses_foreign_sharded_sources_and_fast_policy() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("completion-fast", 2));
        let quality = Arc::new(Provider::new("completion-quality", 3));
        let initial =
            NativeIndexBuilder::new(root.path().join("initial"), generation(1), fast.clone())
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F16, graph(57))
                .add_documents(sources())
                .build_sharded_hybrid(&cx, 2)
                .await
                .unwrap();
        let live = NativeLiveShardedHybridIndex::new(&cx, initial).unwrap();
        let base = live.snapshot(&cx).await.unwrap();
        let old_image = image(base.index().vectors().directory());
        for variant in [
            "fast-model",
            "fast-precision",
            "fast-exact",
            "fast-seed",
            "quality-model",
            "missing-quality",
            "id",
            "content",
            "title",
            "metadata",
            "count",
        ] {
            let mut documents = sources();
            let last = documents.last_mut().unwrap();
            match variant {
                "id" => last.id = "doc-7".to_owned(),
                "content" => last.content = "another body".to_owned(),
                "title" => last.title = Some("another title".to_owned()),
                "metadata" => {
                    last.metadata
                        .insert("version".to_owned(), "another version".to_owned());
                }
                "count" => {
                    let _ = documents.pop();
                }
                _ => {}
            }
            let target_fast = if variant == "fast-model" {
                Arc::new(Provider::new("foreign-fast", 2))
            } else {
                fast.clone()
            };
            let target_quality = if variant == "quality-model" {
                Arc::new(Provider::new("foreign-quality", 3))
            } else {
                quality.clone()
            };
            let precision = if variant == "fast-precision" {
                NativeBuildPrecision::F32
            } else {
                NativeBuildPrecision::F16
            };
            let retrieval = match variant {
                "fast-exact" => NativeBuildRetrieval::Exact,
                "fast-seed" => graph(58),
                _ => graph(57),
            };
            let path = root.path().join(variant);
            let mut builder = NativeIndexBuilder::new(&path, generation(2), target_fast)
                .unwrap()
                .with_fast_storage(precision, retrieval)
                .add_documents(documents);
            if variant != "missing-quality" {
                builder = builder.with_quality_embedder(target_quality).unwrap();
            }
            let built = builder.build_sharded_hybrid(&cx, 3).await.unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            drop(built);
            fast.batches.lock().unwrap().clear();
            quality.batches.lock().unwrap().clear();
            let outcome = base
                .prepare_quality_completion_selected(
                    &cx,
                    &path,
                    generation(2),
                    &receipt,
                    quality.clone(),
                    (NativeBuildPrecision::F32, NativeBuildRetrieval::Exact),
                    NativeShardedHybridReopenLimits::default(),
                )
                .await;
            assert!(outcome.is_err(), "accepted {variant}");
            assert!(fast.submitted().is_empty());
            assert!(quality.submitted().is_empty());
            assert!(Arc::ptr_eq(
                &base.index,
                &live.snapshot(&cx).await.unwrap().index
            ));
        }
        assert_eq!(image(base.index().vectors().directory()), old_image);
    });
}

#[test]
fn selected_quality_completion_checks_the_last_partition_and_expected_predecessor() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (live, fast, _) = initial(&cx, root.path(), sources(), false).await;
        let quality = Arc::new(Provider::new("completion-quality", 3));
        let base = live.snapshot(&cx).await.unwrap();
        let path = root.path().join("completed");
        let built = base
            .begin_quality_completion(&cx, &path, generation(50), quality.clone())
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, graph(91))
            .unwrap()
            .build(&cx, 3)
            .await
            .unwrap();
        let last = built.index().vectors().partitions().last().unwrap();
        let quality_file = last.quality().unwrap().vector_path().to_path_buf();
        let quality_graph = last.quality().unwrap().graph_path().unwrap().to_path_buf();
        let receipt = built.index().seal_for_reopen(&cx).unwrap();
        drop(built);
        fast.batches.lock().unwrap().clear();
        quality.batches.lock().unwrap().clear();
        let policy = (NativeBuildPrecision::F32, graph(91));
        let limits = NativeShardedHybridReopenLimits::default();
        for requested in [
            generation(1),
            ArtifactGenerationIdentityV1::new(50, [0x60; 16]).unwrap(),
        ] {
            assert!(
                base.prepare_quality_completion_selected(
                    &cx,
                    &path,
                    requested,
                    &receipt,
                    quality.clone(),
                    policy,
                    limits,
                )
                .await
                .is_err()
            );
        }
        let mut limited = limits;
        limited.vectors.max_shards = 1;
        assert!(
            base.prepare_quality_completion_selected(
                &cx,
                &path,
                generation(50),
                &receipt,
                quality.clone(),
                policy,
                limited,
            )
            .await
            .is_err()
        );
        fast.changed.store(true, Ordering::SeqCst);
        let drift = base
            .prepare_quality_completion_selected(
                &cx,
                &path,
                generation(50),
                &receipt,
                quality.clone(),
                policy,
                limits,
            )
            .await;
        fast.changed.store(false, Ordering::SeqCst);
        assert!(drift.is_err());
        cx.set_cancel_requested(true);
        let cancelled = base
            .prepare_quality_completion_selected(
                &cx,
                &path,
                generation(50),
                &receipt,
                quality.clone(),
                policy,
                limits,
            )
            .await;
        cx.set_cancel_requested(false);
        assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
        for file in [&quality_file, &quality_graph] {
            let original = fs::read(file).unwrap();
            let mut damaged = original.clone();
            *damaged.last_mut().unwrap() ^= 1;
            fs::write(file, damaged).unwrap();
            let outcome = base
                .prepare_quality_completion_selected(
                    &cx,
                    &path,
                    generation(50),
                    &receipt,
                    quality.clone(),
                    policy,
                    limits,
                )
                .await;
            fs::write(file, original).unwrap();
            assert!(outcome.is_err());
        }
        assert_no_inference(&fast, &quality);
        assert!(Arc::ptr_eq(
            &base.index,
            &live.snapshot(&cx).await.unwrap().index
        ));
        let pending = base.prepare_quality_completion_selected(
            &cx,
            &path,
            generation(50),
            &receipt,
            quality.clone(),
            policy,
            limits,
        );
        let source_update = base
            .begin_update(&cx, root.path().join("source-edit"), generation(2))
            .unwrap()
            .delete_document("doc-6")
            .build(&cx, 2)
            .await
            .unwrap();
        let edited = live.install(&cx, &source_update).await.unwrap();
        let candidate = pending.await.unwrap();
        let error = live.install(&cx, &candidate).await.unwrap_err();
        assert!(matches!(error, SearchError::InvalidConfig { field, .. }
            if field == "native_ann.sharded_live.expected_current"));
        assert!(Arc::ptr_eq(
            &edited.index,
            &live.snapshot(&cx).await.unwrap().index
        ));
        assert!(edited.index().vectors().document("doc-6").is_none());
        assert!(base.index().vectors().document("doc-6").is_some());
        assert!(base.index().vectors().quality().is_none());
        assert_no_inference(&fast, &quality);
    });
}
