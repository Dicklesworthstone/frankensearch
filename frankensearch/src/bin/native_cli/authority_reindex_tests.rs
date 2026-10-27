use super::*;

const REINDEX_QUERY: &str = "retry private-reindex-query";

struct ReindexProvider {
    inner: Provider,
    queries: AtomicUsize,
    fail_query: AtomicUsize,
    cancel_query: AtomicUsize,
    typed_cancel_query: AtomicUsize,
}

impl Embedder for ReindexProvider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.inner.identity()
    }

    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            if text == REINDEX_QUERY {
                let query = self.queries.fetch_add(1, Ordering::SeqCst) + 1;
                if self.cancel_query.load(Ordering::SeqCst) == query {
                    cx.set_cancel_requested(true);
                    return Err(SearchError::SubsystemError {
                        subsystem: "reindex_test",
                        source: "private cancellation fixture".into(),
                    });
                }
                if self.typed_cancel_query.load(Ordering::SeqCst) == query {
                    return Err(SearchError::Cancelled {
                        phase: "private-reindex-fixture".to_owned(),
                        reason: "provider cancellation without cancelling its caller".to_owned(),
                    });
                }
                if self.fail_query.load(Ordering::SeqCst) == query {
                    return Err(SearchError::SubsystemError {
                        subsystem: "reindex_test",
                        source: "private inference failure must not enter recovery output".into(),
                    });
                }
            }
            self.inner.embed(cx, text).await
        })
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

fn reindex_provider(revision: &str) -> Arc<ReindexProvider> {
    let mut identity = identity().identity;
    identity.producer.implementation_revision = revision.to_owned();
    identity.validate().unwrap();
    Arc::new(ReindexProvider {
        inner: Provider {
            identity,
            calls: AtomicUsize::new(0),
        },
        queries: AtomicUsize::new(0),
        fail_query: AtomicUsize::new(0),
        cancel_query: AtomicUsize::new(0),
        typed_cancel_query: AtomicUsize::new(0),
    })
}

fn reindex_models(
    primary: &Arc<ReindexProvider>,
    quality: Option<&Arc<ReindexProvider>>,
) -> Models {
    Models {
        fast: primary.clone(),
        quality: quality.map(|model| model.clone() as Arc<dyn Embedder>),
    }
}

fn reindex_documents() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("excluded-retry", "retry retry retry network requests")
            .with_metadata("tenant", "excluded"),
        IndexableDocument::new("selected-retry", "retry network requests")
            .with_metadata("tenant", "selected"),
        IndexableDocument::new("selected-garden", "garden flowers")
            .with_metadata("tenant", "selected"),
    ]
}

async fn reindex_build(
    cx: &Cx,
    parent: &Path,
    label: &str,
    sequence: u64,
    topology: NativeGenerationTopology,
    models: &Models,
    ann: bool,
) -> NativeBuiltExactGeneration {
    let retrieval = if ann {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 47,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    let mut builder = NativeIndexBuilder::new(
        parent.join(format!("{label}-work")),
        new_generation(sequence).unwrap(),
        Arc::clone(&models.fast),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F32, retrieval)
    .add_documents(reindex_documents());
    if let Some(quality) = &models.quality {
        builder = builder
            .with_quality_embedder(Arc::clone(quality))
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, retrieval)
            .unwrap();
    }
    let source = CommitRange { low: 1, high: 5 };
    if topology == NativeGenerationTopology::FullProgressive {
        builder
            .build_progressive_generation(cx, source, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap()
    } else {
        builder
            .build_exact_generation(
                cx,
                if topology == NativeGenerationTopology::QualityOnly {
                    EmbedderTierTag::Quality
                } else {
                    EmbedderTierTag::Fast
                },
                source,
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap()
    }
}

struct ReindexFixture {
    parent: PathBuf,
    root: QualifiedGenerationRoot,
    selected: Receipt,
    previous: NativeGenerationSnapshot,
    primary: Arc<ReindexProvider>,
    quality: Option<Arc<ReindexProvider>>,
}

impl ReindexFixture {
    async fn new(cx: &Cx, topology: NativeGenerationTopology, ann: bool) -> Self {
        let parent = private_parent("bound-reindex");
        let primary = reindex_provider("original-primary");
        let quality = (topology == NativeGenerationTopology::FullProgressive)
            .then(|| reindex_provider("original-quality"));
        let models = reindex_models(&primary, quality.as_ref());
        let built = reindex_build(cx, &parent, "before", 1, topology, &models, ann).await;
        let root_path = parent.join("root");
        let initial_root = create_root(&root_path).expect("qualified Linux root is required");
        let published = stage_and_publish(
            cx,
            &initial_root,
            &root_path,
            &parent.join("before.json"),
            &built,
            None,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap();
        // Independent admission observes fresh atime/directory metadata while
        // retaining the same root/LOCK/AUTHORITY security identities.
        let root = admit_root(&root_path).unwrap();
        let previous = published
            .receipt
            .open_on_root(cx, &initial_root, DEFAULT_IMAGE_BYTES)
            .unwrap();
        Self {
            parent,
            root,
            selected: published.receipt,
            previous,
            primary,
            quality,
        }
    }

    fn options(&self) -> Options {
        let mut arguments = retained_rebuild_arguments();
        arguments.extend([
            "--migrate-model",
            "--query",
            REINDEX_QUERY,
            "--rollback-receipt",
            "rollback.json",
            "--limit",
            "2",
            "--filter",
            r#"{"metadata":{"tenant":"selected"}}"#,
        ]);
        let mut options = parse(&arguments).unwrap().unwrap();
        options.from_receipt = Some(self.parent.join("before.json"));
        options.receipt = self.parent.join("after.json");
        options.rollback_receipt = Some(self.parent.join("rollback.json"));
        options.work = Some(self.parent.join("unused-options-work"));
        options
    }

    fn replacements(&self) -> (Arc<ReindexProvider>, Option<Arc<ReindexProvider>>, Models) {
        let primary = reindex_provider("replacement-primary");
        let quality = self
            .quality
            .as_ref()
            .map(|_| reindex_provider("replacement-quality"));
        let models = reindex_models(&primary, quality.as_ref());
        (primary, quality, models)
    }

    async fn build(&self, cx: &Cx, models: &Models, ann: bool) -> NativeBuiltExactGeneration {
        reindex_build(
            cx,
            &self.parent,
            "after",
            2,
            self.previous.topology(),
            models,
            ann,
        )
        .await
    }
}

#[test]
fn bound_reindex_previews_publishes_and_replays_every_complete_topology_and_filter() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for topology in [
            NativeGenerationTopology::FastOnly,
            NativeGenerationTopology::QualityOnly,
            NativeGenerationTopology::FullProgressive,
        ] {
            for ann in [false, true] {
                let fixture = ReindexFixture::new(&cx, topology, ann).await;
                let (primary, quality, loaded) = fixture.replacements();
                let mut options = fixture.options();
                options.stream = topology == NativeGenerationTopology::FullProgressive && !ann;
                let replacement_retrieval = if ann {
                    NativeBuildRetrieval::Exact
                } else {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 47,
                    }
                };
                options.retrieval = Some(replacement_retrieval);
                let mut loaded = Some(loaded);
                let mut requested = Vec::new();
                let mut output = Vec::new();
                Box::pin(rebuild_with_loader(&cx, &options, &mut output, |mode| {
                    requested.push(mode);
                    Ok(loaded.take().expect("replacement models load only once"))
                }))
                .await
                .unwrap();
                assert_eq!(requested, [fixture.selected.query_mode(None).unwrap()]);
                let responses = frames(&output);
                assert_eq!(responses.len(), if options.stream { 2 } else { 1 });
                assert_eq!(primary.queries.load(Ordering::SeqCst), 2);
                if let Some(quality) = quality {
                    assert_eq!(quality.queries.load(Ordering::SeqCst), 2);
                }
                let after = Receipt::read(&options.receipt).unwrap();
                let active = after
                    .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                    .unwrap();
                assert_eq!(
                    active.authority().sequence,
                    fixture.previous.authority().sequence + 1
                );
                assert_eq!(active.topology(), topology);
                assert_eq!(
                    active.metadata().commit_range,
                    fixture.previous.metadata().commit_range
                );
                assert_eq!(
                    after.generation.sequence,
                    fixture.selected.generation.sequence + 1
                );
                assert_eq!(active.retrieval_configuration(), replacement_retrieval);
                assert_eq!(
                    active.quality_retrieval_configuration(),
                    (topology == NativeGenerationTopology::FullProgressive)
                        .then_some(replacement_retrieval)
                );
                for response in &responses {
                    assert_eq!(response["operation"], "reindex");
                    assert_eq!(response["publication"], "committed");
                    assert_eq!(response["replay"], "complete");
                    assert_eq!(
                        response["recovery"]["plan"]["state"]["state"],
                        "identity_mismatch"
                    );
                    assert_eq!(response["recovery_result"]["state"]["state"], "ready");
                    assert_eq!(
                        response["recovery"]["binding"]["candidate_authority"],
                        serde_json::json!(after.authority)
                    );
                    assert_eq!(
                        response["recovery"]["binding"]["expected_authority"],
                        serde_json::json!(fixture.selected.authority)
                    );
                    assert_eq!(response["scope"]["eligible_documents"], 2);
                    assert!(
                        response["results"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|hit| hit["doc_id"].as_str().unwrap().starts_with("selected-"))
                    );
                }
                let text = String::from_utf8(output).unwrap();
                assert!(!text.contains(REINDEX_QUERY));
                assert!(!options.rollback_receipt.as_ref().unwrap().exists());
                let original = fixture
                    .previous
                    .source_documents(&cx, MAX_INPUT_BYTES)
                    .unwrap();
                assert_eq!(original.len(), 3);
                assert_eq!(fixture.primary.queries.load(Ordering::SeqCst), 0);
                if let Some(quality) = fixture.quality {
                    assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
                }
            }
        }
    });
}

#[test]
fn bound_reindex_preview_failure_never_selects_or_saves_an_attempt() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for topology in [
            NativeGenerationTopology::FastOnly,
            NativeGenerationTopology::QualityOnly,
            NativeGenerationTopology::FullProgressive,
        ] {
            let fixture = ReindexFixture::new(&cx, topology, false).await;
            let (primary, quality, loaded) = fixture.replacements();
            quality
                .as_ref()
                .unwrap_or(&primary)
                .fail_query
                .store(1, Ordering::SeqCst);
            let built = fixture.build(&cx, &loaded, false).await;
            let options = fixture.options();
            let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
            let old_receipt = fs::read(fixture.parent.join("before.json")).unwrap();
            let mut output = Vec::new();
            let error = reindex::execute(
                &cx,
                &options,
                &fixture.root,
                &fixture.selected,
                &fixture.previous,
                &built,
                &loaded,
                &mut output,
            )
            .await
            .unwrap_err();
            let report = publication_failure(error.as_ref()).unwrap();
            assert_eq!(report["publication"], "not_attempted");
            assert_eq!(report["reason"], "preview_refused");
            assert!(
                !serde_json::to_string(&report)
                    .unwrap()
                    .contains("private inference failure")
            );
            assert!(output.is_empty());
            assert!(!options.receipt.exists());
            assert!(!options.rollback_receipt.as_ref().unwrap().exists());
            assert_eq!(
                fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                before
            );
            assert_eq!(
                fs::read(fixture.parent.join("before.json")).unwrap(),
                old_receipt
            );
        }
    });
}

#[test]
fn bound_reindex_refuses_empty_scope_no_change_and_bypassed_authorization() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = ReindexFixture::new(&cx, NativeGenerationTopology::FastOnly, false).await;
        let (primary, _, loaded) = fixture.replacements();
        let built = fixture.build(&cx, &loaded, false).await;
        let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
        for refusal in 0..5 {
            let mut options = fixture.options();
            match refusal {
                0 => options.filter = Some(Filter::parse(r#"{"ids":[]}"#).unwrap()),
                1 => options.migrate_model = false,
                2 => options.limit = 0,
                3 => options.query = Some("  ".to_owned()),
                _ => options.mode = Some(QueryMode::Fast),
            }
            let mut output = Vec::new();
            assert!(
                reindex::execute(
                    &cx,
                    &options,
                    &fixture.root,
                    &fixture.selected,
                    &fixture.previous,
                    &built,
                    &loaded,
                    &mut output
                )
                .await
                .is_err()
            );
            assert!(output.is_empty());
            assert!(!options.receipt.exists());
            assert!(!options.rollback_receipt.as_ref().unwrap().exists());
            assert_eq!(
                fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                before
            );
        }
        assert_eq!(primary.queries.load(Ordering::SeqCst), 0);
        let unchanged = reindex_models(&fixture.primary, None);
        let same = reindex_build(
            &cx,
            &fixture.parent,
            "unchanged",
            2,
            fixture.previous.topology(),
            &unchanged,
            false,
        )
        .await;
        let options = fixture.options();
        assert!(
            reindex::execute(
                &cx,
                &options,
                &fixture.root,
                &fixture.selected,
                &fixture.previous,
                &same,
                &unchanged,
                &mut Vec::new()
            )
            .await
            .is_err()
        );
        assert_eq!(fixture.primary.queries.load(Ordering::SeqCst), 0);
        assert!(!options.receipt.exists());
        assert_eq!(
            fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
            before
        );
    });
}

#[test]
fn bound_reindex_replay_failure_rolls_back_once_with_the_new_exact_pair() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for topology in [
            NativeGenerationTopology::FastOnly,
            NativeGenerationTopology::QualityOnly,
            NativeGenerationTopology::FullProgressive,
        ] {
            let fixture = ReindexFixture::new(&cx, topology, true).await;
            let (primary, quality, loaded) = fixture.replacements();
            quality
                .as_ref()
                .unwrap_or(&primary)
                .fail_query
                .store(2, Ordering::SeqCst);
            let built = fixture.build(&cx, &loaded, false).await;
            let options = fixture.options();
            let original_receipt = fs::read(fixture.parent.join("before.json")).unwrap();
            let mut output = Vec::new();
            let error = reindex::execute(
                &cx,
                &options,
                &fixture.root,
                &fixture.selected,
                &fixture.previous,
                &built,
                &loaded,
                &mut output,
            )
            .await
            .unwrap_err();
            let report = publication_failure(error.as_ref()).unwrap();
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["replay"], "failed");
            assert_eq!(report["rollback"]["status"], "committed");
            assert!(
                !serde_json::to_string(&report)
                    .unwrap()
                    .contains("private inference failure")
            );
            assert!(output.is_empty());
            assert_eq!(primary.queries.load(Ordering::SeqCst), 2);
            if let Some(quality) = &quality {
                assert_eq!(quality.queries.load(Ordering::SeqCst), 2);
            }
            let failed = Receipt::read(&options.receipt).unwrap();
            let restored = Receipt::read(options.rollback_receipt.as_ref().unwrap()).unwrap();
            assert_eq!(failed.authority().unwrap().sequence, 2);
            assert_eq!(restored.authority().unwrap().sequence, 3);
            assert_eq!(restored.generation, fixture.selected.generation);
            assert_eq!(restored.identity, fixture.selected.identity);
            assert_eq!(restored.quality_identity, fixture.selected.quality_identity);
            let expected = restored.recovery_attempt().unwrap().expected;
            assert!(
                [expected.first, expected.second]
                    .into_iter()
                    .flatten()
                    .any(|slot| slot.authority == failed.authority().unwrap())
            );
            assert_ne!(
                expected,
                fixture.previous.retained_snapshot().authority_pair()
            );
            let active = restored
                .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            assert_eq!(active.authority(), restored.authority().unwrap());
            assert_eq!(active.metadata(), fixture.previous.metadata());
            assert_eq!(
                fs::read(fixture.parent.join("before.json")).unwrap(),
                original_receipt
            );
            assert!(
                failed
                    .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                    .is_err()
            );
        }
    });
}

#[test]
fn bound_reindex_cancellation_after_commit_keeps_exact_receipts_and_defers_rollback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture = ReindexFixture::new(&cx, NativeGenerationTopology::FastOnly, false).await;
        let (primary, _, loaded) = fixture.replacements();
        primary.cancel_query.store(2, Ordering::SeqCst);
        let built = fixture.build(&cx, &loaded, false).await;
        let options = fixture.options();
        let mut output = Vec::new();
        let error = reindex::execute(
            &cx,
            &options,
            &fixture.root,
            &fixture.selected,
            &fixture.previous,
            &built,
            &loaded,
            &mut output,
        )
        .await
        .unwrap_err();
        assert!(
            cx.is_cancel_requested(),
            "executor must not clear caller cancellation"
        );
        let report = publication_failure(error.as_ref()).unwrap();
        assert_eq!(report["publication"], "committed");
        assert_eq!(report["reason"], "replay_cancelled");
        assert_eq!(report["rollback"]["status"], "deferred");
        assert_eq!(report["rollback"]["reason"], "caller_cancelled");
        assert!(output.is_empty());
        assert!(options.receipt.exists());
        assert!(!options.rollback_receipt.as_ref().unwrap().exists());
        // Reset only in the test harness, after proving the executor preserved
        // cancellation, to inspect the durable state without another mutation.
        cx.set_cancel_requested(false);
        let after = Receipt::read(&options.receipt).unwrap();
        assert_eq!(
            after
                .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                .unwrap()
                .authority()
                .sequence,
            2
        );
        assert_eq!(fixture.previous.authority().sequence, 1);
        assert_eq!(primary.queries.load(Ordering::SeqCst), 2);
    });
}

#[test]
fn bound_reindex_stale_compare_and_swap_does_not_rebase_replay_or_rollback() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture =
            ReindexFixture::new(&cx, NativeGenerationTopology::FullProgressive, false).await;
        let (primary, quality, loaded) = fixture.replacements();
        let built = fixture.build(&cx, &loaded, false).await;
        let original = reindex_models(&fixture.primary, fixture.quality.as_ref());
        let competitor = reindex_build(
            &cx,
            &fixture.parent,
            "competitor",
            2,
            fixture.previous.topology(),
            &original,
            false,
        )
        .await;
        let winning = stage_and_publish(
            &cx,
            &fixture.root,
            &fixture.selected.root,
            &fixture.parent.join("winner.json"),
            &competitor,
            Some(&fixture.previous),
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap();
        let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
        let options = fixture.options();
        let mut output = Vec::new();
        let error = reindex::execute(
            &cx,
            &options,
            &fixture.root,
            &fixture.selected,
            &fixture.previous,
            &built,
            &loaded,
            &mut output,
        )
        .await
        .unwrap_err();
        let report = publication_failure(error.as_ref()).unwrap();
        assert_eq!(report["publication"], "not_committed");
        assert_eq!(report["replay"], "not_attempted");
        assert_eq!(report["rollback"]["status"], "not_required");
        assert_eq!(primary.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.unwrap().queries.load(Ordering::SeqCst), 1);
        assert!(output.is_empty());
        assert!(!options.rollback_receipt.as_ref().unwrap().exists());
        assert_eq!(
            fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
            authority
        );
        assert_eq!(
            winning
                .receipt
                .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                .unwrap()
                .authority(),
            winning.snapshot.authority()
        );
        let refused = Receipt::read(&options.receipt).unwrap();
        assert_eq!(
            refused.recovery_attempt().unwrap().expected,
            fixture.previous.retained_snapshot().authority_pair()
        );
        assert_ne!(refused.authority().unwrap(), winning.snapshot.authority());
    });
}

#[test]
fn bound_reindex_distinguishes_preview_cancellation_and_provider_replay_cancellation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for cancel_during_preview in [true, false] {
            let fixture =
                ReindexFixture::new(&cx, NativeGenerationTopology::FullProgressive, false).await;
            let (primary, quality, loaded) = fixture.replacements();
            let quality = quality.unwrap();
            if cancel_during_preview {
                quality.cancel_query.store(1, Ordering::SeqCst);
            } else {
                quality.typed_cancel_query.store(2, Ordering::SeqCst);
            }
            let built = fixture.build(&cx, &loaded, false).await;
            let options = fixture.options();
            let mut output = Vec::new();
            let error = reindex::execute(
                &cx,
                &options,
                &fixture.root,
                &fixture.selected,
                &fixture.previous,
                &built,
                &loaded,
                &mut output,
            )
            .await
            .unwrap_err();
            let report = publication_failure(error.as_ref()).unwrap();
            assert!(output.is_empty());
            if cancel_during_preview {
                assert_eq!(report["reason"], "preview_cancelled");
                assert_eq!(report["publication"], "not_attempted");
                assert!(!options.receipt.exists());
                assert!(!options.rollback_receipt.as_ref().unwrap().exists());
                assert!(cx.is_cancel_requested());
                cx.set_cancel_requested(false);
                assert_eq!(
                    fixture
                        .selected
                        .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                        .unwrap()
                        .authority()
                        .sequence,
                    1
                );
            } else {
                assert_eq!(report["reason"], "replay_cancelled");
                assert_eq!(report["publication"], "committed");
                assert_eq!(report["rollback"]["status"], "committed");
                assert!(!cx.is_cancel_requested());
                let restored = Receipt::read(options.rollback_receipt.as_ref().unwrap()).unwrap();
                assert_eq!(
                    restored
                        .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                        .unwrap()
                        .authority()
                        .sequence,
                    3
                );
            }
            let expected_queries = if cancel_during_preview { 1 } else { 2 };
            assert_eq!(primary.queries.load(Ordering::SeqCst), expected_queries);
            assert_eq!(quality.queries.load(Ordering::SeqCst), expected_queries);
            assert!(
                !serde_json::to_string(&report)
                    .unwrap()
                    .contains("private-reindex-fixture")
            );
        }
    });
}

#[test]
fn bound_reindex_result_delivery_failure_preserves_successful_selection() {
    struct BrokenOutput;
    impl Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "closed fixture",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let fixture =
            ReindexFixture::new(&cx, NativeGenerationTopology::FullProgressive, false).await;
        let (primary, quality, loaded) = fixture.replacements();
        let built = fixture.build(&cx, &loaded, false).await;
        let mut options = fixture.options();
        options.stream = true;
        let error = reindex::execute(
            &cx,
            &options,
            &fixture.root,
            &fixture.selected,
            &fixture.previous,
            &built,
            &loaded,
            &mut BrokenOutput,
        )
        .await
        .unwrap_err();
        let report = publication_failure(error.as_ref()).unwrap();
        assert_eq!(report["publication"], "committed");
        assert_eq!(report["serving"], "activated");
        assert_eq!(report["replay"], "complete");
        assert_eq!(report["reason"], "result_delivery_failed");
        assert_eq!(primary.queries.load(Ordering::SeqCst), 2);
        assert_eq!(quality.unwrap().queries.load(Ordering::SeqCst), 2);
        let after = Receipt::read(&options.receipt).unwrap();
        assert_eq!(
            after
                .open_on_root(&cx, &fixture.root, DEFAULT_IMAGE_BYTES)
                .unwrap()
                .authority()
                .sequence,
            2
        );
        assert!(!options.rollback_receipt.as_ref().unwrap().exists());
    });
}
