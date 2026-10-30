//! Add a complete independent quality tier using one retained fast authority.

use super::*;

pub(super) async fn run(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    pool: Option<BlockingPoolHandle>,
) -> Result<()> {
    let context = execution_context(cx, pool)?;
    Box::pin(with_loader(&context, options, output, || {
        load_authority_models(
            &context,
            options,
            QueryMode::Full,
            context.blocking_pool_handle(),
        )
    }))
    .await
}

async fn with_loader<F>(
    cx: &Cx,
    options: &Options,
    output: &mut impl Write,
    load_models: F,
) -> Result<()>
where
    F: FnOnce() -> Result<Models>,
{
    checkpoint(cx)?;
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing predecessor receipt"))?,
    )?;
    if selected.topology() != NativeGenerationTopology::FastOnly {
        return Err(bad(
            "quality completion requires a fast-only authority receipt",
        ));
    }
    let work_path = new_path(
        options
            .work
            .as_deref()
            .ok_or_else(|| bad("missing work directory"))?,
    )?;
    let receipt_path = new_path(&options.receipt)?;
    separate_destinations(&[&selected.root, &work_path, &receipt_path])?;
    let root = admit_root(&selected.root)?;
    let snapshot = selected.open_on_root(cx, &root, options.max_image_bytes)?;
    snapshot.authority().next_sequence()?;
    let sequence = selected
        .generation
        .sequence
        .checked_add(1)
        .ok_or_else(|| bad("the artifact generation sequence is exhausted"))?;
    // Enforce the ordinary CLI source/record limits before model loading. The
    // completion plan then retains that same admitted source, never a file path.
    drop(retained_rebuild_documents(
        cx,
        &snapshot,
        options.max_source_bytes,
    )?);
    let loaded = load_models()?;
    let quality = loaded.quality.ok_or_else(|| {
        bad("quality completion requires an independently supplied quality model")
    })?;
    let built = snapshot
        .begin_quality_completion(
            cx,
            &work_path,
            new_generation(sequence)?,
            loaded.fast,
            quality,
            options.max_source_bytes,
        )?
        .with_batch_size(options.batch_size)?
        .with_max_batch_input_bytes(MAX_RECORD_BYTES)?
        .with_quality_storage(NativeBuildPrecision::F32, options.build_retrieval(None))?
        .build(cx, options.max_image_bytes)
        .await?;
    let prepared = prepare_publication_with_intent(
        &built,
        Some(&snapshot),
        options.max_image_bytes,
        PublicationIntent::CompleteQuality,
    )?;
    let published = stage_prepared_and_publish(
        cx,
        &root,
        &selected.root,
        &receipt_path,
        &built,
        prepared,
        PublicationOptions::unscoped(options.max_image_bytes),
    )?;
    emit_indexed(
        output,
        &published.receipt,
        &receipt_path,
        built.retrieval_configuration(),
        built.quality_retrieval_configuration(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> Vec<String> {
        [
            "authority-complete-quality",
            "--from-receipt",
            "fast.json",
            "--work-dir",
            "new-work",
            "--receipt",
            "full.json",
        ]
        .map(str::to_owned)
        .into()
    }

    #[test]
    fn completion_has_explicit_topology_intent_and_never_accepts_replacement_sources() {
        let options = Options::parse(arguments()).unwrap().unwrap();
        assert_eq!(options.command, Command::CompleteQuality);
        assert!(is_command("authority-complete-quality"));
        for extra in [
            vec!["--input", "replacement.jsonl"],
            vec!["--source-low", "1"],
            vec!["--source-high", "2"],
            vec!["--migrate-model"],
            vec!["--retained-source"],
            vec!["--current-receipt", "other.json"],
            vec!["--tier", "full"],
            vec!["--query", "retry"],
        ] {
            let mut args = arguments();
            args.extend(extra.into_iter().map(str::to_owned));
            assert!(Options::parse(args).is_err());
        }
        let mut args = arguments();
        args.extend(
            [
                "--retrieval",
                "hnsw",
                "--batch-size",
                "2",
                "--max-source-bytes",
                "8192",
            ]
            .map(str::to_owned),
        );
        let options = Options::parse(args).unwrap().unwrap();
        assert_eq!(options.batch_size, 2);
        assert_eq!(options.max_source_bytes, 8192);
        assert!(matches!(
            options.retrieval,
            Some(NativeBuildRetrieval::Hnsw { .. })
        ));
    }

    #[test]
    fn registered_fast_and_quality_models_share_the_retained_prepared_input_contract() {
        use frankensearch::embed::model_manifest::ModelArtifactManifestV1;

        // This checks registered contract compatibility, not model availability
        // or real inference. Model-specific tokenizers/normalization remain in
        // the independently authenticated space identities.
        let fast = [
            ModelArtifactManifestV1::potion_128m_native().unwrap(),
            ModelArtifactManifestV1::potion_base_8m_native().unwrap(),
            ModelArtifactManifestV1::potion_base_32m_native().unwrap(),
        ];
        let quality = [
            ModelArtifactManifestV1::minilm_fastembed().unwrap(),
            ModelArtifactManifestV1::minilm_native_frankentorch().unwrap(),
            ModelArtifactManifestV1::minilm_native_frankentorch_f32().unwrap(),
            ModelArtifactManifestV1::multilingual_minilm_native_frankentorch().unwrap(),
        ];
        for fast in &fast {
            for quality in &quality {
                assert_eq!(
                    fast.execution.input_contract,
                    quality.execution.input_contract
                );
            }
        }
    }

    #[cfg(target_os = "linux")]
    mod lifecycle {
        use super::*;
        use frankensearch_core::generation::{
            EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
            GenerationComponentRole,
        };
        use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
        use std::os::unix::fs::DirBuilderExt as _;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        struct Provider {
            identity: EmbeddingIdentityBundleV1,
            foreign: EmbeddingIdentityBundleV1,
            calls: AtomicUsize,
            fault: AtomicUsize,
            drifted: AtomicBool,
        }

        impl Provider {
            fn new(quality: bool) -> Arc<Self> {
                let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(
                    if quality {
                        "completion-quality"
                    } else {
                        "completion-fast"
                    },
                    if quality { 3 } else { 2 },
                );
                // Synthetic providers exercise production binding and authority
                // publication, without claiming real-model relevance evidence.
                identity.space.kind = EmbeddingSpaceKindV1::Semantic;
                identity.space.hash_control = None;
                identity.space.artifact_manifest_fingerprint = "a".repeat(64);
                identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                    role: "weights".to_owned(),
                    sha256: if quality { "c" } else { "b" }.repeat(64),
                    size: 1,
                }];
                identity.producer.space_fingerprint = identity.space.fingerprint();
                identity.validate().unwrap();
                let mut foreign = identity.clone();
                foreign
                    .producer
                    .implementation_revision
                    .push_str("-foreign");
                Arc::new(Self {
                    identity,
                    foreign,
                    calls: AtomicUsize::new(0),
                    fault: AtomicUsize::new(0),
                    drifted: AtomicBool::new(false),
                })
            }
        }

        impl Embedder for Provider {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                Ok(if self.drifted.load(Ordering::SeqCst) {
                    &self.foreign
                } else {
                    &self.identity
                })
            }

            fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                Box::pin(async move {
                    self.calls.fetch_add(1, Ordering::SeqCst);
                    match self.fault.load(Ordering::SeqCst) {
                        1 => {
                            return Err(SearchError::InvalidConfig {
                                field: "completion.test".to_owned(),
                                value: "inference-forbidden".to_owned(),
                                reason: "injected provider failure".to_owned(),
                            });
                        }
                        2 => {
                            cx.set_cancel_requested(true);
                        }
                        3 => self.drifted.store(true, Ordering::SeqCst),
                        _ => {}
                    }
                    let mut values = vec![0.0; self.dimension()];
                    values[usize::from(!text.contains("retry"))] = 1.0;
                    Ok(values)
                })
            }

            fn dimension(&self) -> usize {
                usize::try_from(self.identity.space.dimension).unwrap()
            }
            fn id(&self) -> &str {
                &self.identity.space.logical_model_id
            }
            fn model_name(&self) -> &str {
                self.id()
            }
            fn is_semantic(&self) -> bool {
                true
            }
            fn category(&self) -> ModelCategory {
                ModelCategory::TransformerEmbedder
            }
        }

        struct Fixture {
            parent: PathBuf,
            fast: Arc<Provider>,
            selected: Receipt,
            snapshot: Arc<NativeGenerationSnapshot>,
        }

        async fn fixture(cx: &Cx, ann: bool) -> Fixture {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
                ".frankensearch-quality-completion-{}-{stamp}",
                std::process::id(),
            ));
            fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
            let root = create_root(&parent.join("root")).expect("qualified Linux root required");
            let fast = Provider::new(false);
            let built = NativeIndexBuilder::new(
                parent.join("original-work"),
                new_generation(1).unwrap(),
                fast.clone(),
            )
            .unwrap()
            .with_fast_storage(
                NativeBuildPrecision::F16,
                if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 17,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                },
            )
            .add_documents([
                IndexableDocument::new("retry", "retry network requests")
                    .with_metadata("revision", "old"),
                IndexableDocument::new("garden", "garden flowers"),
            ])
            .build_exact_generation(
                cx,
                EmbedderTierTag::Fast,
                CommitRange { low: 4, high: 9 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let published = stage_and_publish(
                cx,
                &root,
                &parent.join("root"),
                &parent.join("fast.json"),
                &built,
                None,
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap();
            // Completion must use admitted source bytes, not the original build
            // directory or any original corpus path.
            fs::rename(
                parent.join("original-work"),
                parent.join("unavailable-work"),
            )
            .unwrap();
            fast.fault.store(1, Ordering::SeqCst);
            Fixture {
                parent,
                fast,
                selected: published.receipt,
                snapshot: published.snapshot,
            }
        }

        fn options(fixture: &Fixture, label: &str) -> Options {
            let mut options = Options::parse(arguments()).unwrap().unwrap();
            options.from_receipt = Some(fixture.parent.join("fast.json"));
            options.work = Some(fixture.parent.join(format!("{label}-work")));
            options.receipt = fixture.parent.join(format!("{label}.json"));
            options
        }

        #[test]
        fn completion_reuses_fast_inference_and_quill_then_reopens_full_progressive_authority() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    let fixture = fixture(&cx, ann).await;
                    let quality = Provider::new(true);
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    let mut options = options(&fixture, "complete");
                    options.retrieval = Some(if ann {
                        NativeBuildRetrieval::Exact
                    } else {
                        NativeBuildRetrieval::Hnsw {
                            params: HnswParams::default(),
                            seed: 29,
                        }
                    });
                    let mut output = Vec::new();
                    with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await
                    .unwrap();
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
                    let receipt = Receipt::read(&options.receipt).unwrap();
                    let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        reopened.topology(),
                        NativeGenerationTopology::FullProgressive
                    );
                    assert_eq!(receipt.identity, fixture.selected.identity);
                    assert_eq!(receipt.quality_binding().unwrap().unwrap().dimension(), 3);
                    assert_eq!(
                        reopened.metadata().commit_range,
                        fixture.snapshot.metadata().commit_range
                    );
                    assert_eq!(
                        reopened.retrieval_configuration(),
                        fixture.snapshot.retrieval_configuration()
                    );
                    assert_eq!(
                        reopened.quality_retrieval_configuration(),
                        options.retrieval
                    );
                    assert_eq!(
                        reopened
                            .retained_snapshot()
                            .closure()
                            .bytes(GenerationComponentRole::Lexical),
                        fixture
                            .snapshot
                            .retained_snapshot()
                            .closure()
                            .bytes(GenerationComponentRole::Lexical),
                    );
                    assert_eq!(
                        serde_json::to_value(
                            reopened.source_documents(&cx, MAX_INPUT_BYTES).unwrap()
                        )
                        .unwrap(),
                        serde_json::to_value(
                            fixture
                                .snapshot
                                .source_documents(&cx, MAX_INPUT_BYTES)
                                .unwrap()
                        )
                        .unwrap(),
                    );
                    fixture.fast.fault.store(0, Ordering::SeqCst);
                    let results = reopened
                        .search_refined_text(
                            &cx,
                            fixture.fast.as_ref(),
                            quality.as_ref(),
                            "retry",
                            2,
                        )
                        .await
                        .unwrap();
                    assert!(results.iter().any(|result| result.quality_score.is_some()));
                    assert_eq!(
                        fixture.snapshot.topology(),
                        NativeGenerationTopology::FastOnly
                    );
                    assert!(
                        !fixture
                            .snapshot
                            .search_text(&cx, fixture.fast.as_ref(), "retry", 1)
                            .await
                            .unwrap()
                            .is_empty()
                    );
                    assert!(fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                    assert_eq!(report["publication"], "committed");
                }
            });
        }

        #[test]
        fn completion_failure_cancellation_and_quality_identity_drift_preserve_incumbent() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fixture = fixture(&cx, false).await;
                let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let options = options(&fixture, &format!("refused-{fault}"));
                    assert!(
                        with_loader(&cx, &options, &mut Vec::new(), || Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        }))
                        .await
                        .is_err()
                    );
                    cx.set_cancel_requested(false);
                    assert!(!options.receipt.exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        before
                    );
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                }
                let wrong_fast = Provider::new(false);
                wrong_fast.drifted.store(true, Ordering::SeqCst);
                let options = options(&fixture, "wrong-fast");
                let quality = Provider::new(true);
                assert!(
                    with_loader(&cx, &options, &mut Vec::new(), || Ok(Models {
                        fast: wrong_fast.clone(),
                        quality: Some(quality.clone()),
                    }))
                    .await
                    .is_err()
                );
                assert_eq!(wrong_fast.calls.load(Ordering::SeqCst), 0);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                assert!(!options.work.unwrap().exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    before
                );
            });
        }

        #[test]
        fn competing_publication_refuses_a_completed_quality_candidate_without_rebasing() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fixture = fixture(&cx, false).await;
                let quality = Provider::new(true);
                let completed = fixture
                    .snapshot
                    .begin_quality_completion(
                        &cx,
                        fixture.parent.join("completion-work"),
                        new_generation(2).unwrap(),
                        fixture.fast.clone(),
                        quality,
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build(&cx, DEFAULT_IMAGE_BYTES)
                    .await
                    .unwrap();
                assert!(
                    prepare_publication(&completed, Some(&fixture.snapshot), DEFAULT_IMAGE_BYTES)
                        .is_err()
                );
                let prepared = prepare_publication_with_intent(
                    &completed,
                    Some(&fixture.snapshot),
                    DEFAULT_IMAGE_BYTES,
                    PublicationIntent::CompleteQuality,
                )
                .unwrap();
                let winner = fixture
                    .snapshot
                    .begin_update(
                        &cx,
                        fixture.parent.join("winner-work"),
                        new_generation(2).unwrap(),
                        fixture.fast.clone(),
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build_exact_generation(
                        &cx,
                        EmbedderTierTag::Fast,
                        fixture.snapshot.metadata().commit_range.clone(),
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap();
                let root = admit_root(&fixture.selected.root).unwrap();
                let published = stage_and_publish(
                    &cx,
                    &root,
                    &fixture.selected.root,
                    &fixture.parent.join("winner.json"),
                    &winner,
                    Some(&fixture.snapshot),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap();
                let winner_authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                assert!(
                    stage_prepared_and_publish(
                        &cx,
                        &root,
                        &fixture.selected.root,
                        &fixture.parent.join("stale.json"),
                        &completed,
                        prepared,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )
                    .is_err()
                );
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    winner_authority
                );
                published.receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    published.snapshot.topology(),
                    NativeGenerationTopology::FastOnly
                );
                assert_eq!(
                    fixture.snapshot.authority(),
                    fixture.selected.authority().unwrap()
                );
            });
        }
    }
}
