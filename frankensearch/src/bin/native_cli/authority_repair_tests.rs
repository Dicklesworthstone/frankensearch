use super::super::{Command, is_command};
use super::*;

fn arguments() -> Vec<String> {
    [
        "authority-repair",
        "--from-receipt",
        "damaged.json",
        "--work-dir",
        "new-work",
        "--receipt",
        "repaired.json",
        "--retrieval",
        "exact",
    ]
    .map(str::to_owned)
    .into()
}

#[test]
fn repair_requires_explicit_exact_retrieval_and_excludes_source_or_model_migration() {
    let options = Options::parse(arguments()).unwrap().unwrap();
    assert_eq!(options.command, Command::Repair);
    assert!(is_command("authority-repair"));
    assert_eq!(options.retrieval, Some(NativeBuildRetrieval::Exact));
    let mut missing = arguments();
    missing.truncate(missing.len() - 2);
    assert!(Options::parse(missing).is_err());
    let mut ann = arguments();
    *ann.last_mut().unwrap() = "hnsw".to_owned();
    assert!(Options::parse(ann).is_err());
    for extra in [
        vec!["--input", "replacement.jsonl"],
        vec!["--source-low", "1"],
        vec!["--source-high", "2"],
        vec!["--retained-source"],
        vec!["--migrate-model"],
        vec!["--tier", "full"],
        vec!["--current-receipt", "other.json"],
        vec!["--target-receipt", "other.json"],
        vec!["--allow-damaged-current"],
        vec!["--query", "retry"],
        vec!["--filter", "{}"],
        vec!["--root", "another-root"],
    ] {
        let mut args = arguments();
        args.extend(extra.into_iter().map(str::to_owned));
        assert!(Options::parse(args).is_err());
    }
    let mut args = arguments();
    args.extend(
        [
            "--model-dir",
            "models",
            "--quality-backend",
            "native-f32",
            "--quality-model-dir",
            "quality",
            "--batch-size",
            "2",
            "--max-source-bytes",
            "8192",
            "--max-image-bytes",
            "65536",
        ]
        .map(str::to_owned),
    );
    let options = Options::parse(args).unwrap().unwrap();
    assert_eq!(options.batch_size, 2);
    assert_eq!(options.max_source_bytes, 8192);
    assert_eq!(options.max_image_bytes, 65536);
}

#[cfg(target_os = "linux")]
mod lifecycle {
    use super::super::super::{
        CommitRange, DEFAULT_IMAGE_BYTES, HnswParams, IndexableDocument, MAX_INPUT_BYTES,
        NativeBuildPrecision, NativeGenerationSnapshot, NativeGenerationTopology,
        PublicationFailure, SearchError, create_root, recover, save_receipt, stage_and_publish,
        write_immutable,
    };
    use super::*;
    use frankensearch_core::generation::{
        ActivationManifest, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        EmbeddingSpaceKindV1, GenerationComponentRole,
    };
    use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
    use frankensearch_index::generation_root::generation_reader::component_object_name_v1;
    use std::fs;
    use std::io;
    use std::os::unix::fs::DirBuilderExt as _;
    use std::path::PathBuf;
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
                    "repair-quality"
                } else {
                    "repair-fast"
                },
                if quality { 3 } else { 2 },
            );
            // Valid synthetic semantic providers exercise the actual identity,
            // source, builder and authority contracts, not real model quality.
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
            foreign.validate().unwrap();
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
                            field: "repair.test".to_owned(),
                            value: "injected".to_owned(),
                            reason: "provider failed".to_owned(),
                        });
                    }
                    2 => cx.set_cancel_requested(true),
                    3 => self.drifted.store(true, Ordering::SeqCst),
                    _ => {}
                }
                let mut vector = vec![0.0; self.dimension()];
                vector[usize::from(!text.contains("retry"))] = 1.0;
                Ok(vector)
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
        primary: Arc<Provider>,
        quality: Option<Arc<Provider>>,
        selected: Receipt,
        snapshot: Arc<NativeGenerationSnapshot>,
    }

    impl Fixture {
        fn models(&self) -> Models {
            Models {
                fast: self.primary.clone(),
                quality: self
                    .quality
                    .as_ref()
                    .map(|quality| quality.clone() as Arc<dyn Embedder>),
            }
        }

        fn options(&self, label: &str) -> Options {
            let mut options = Options::parse(arguments()).unwrap().unwrap();
            options.from_receipt = Some(self.parent.join("selected.json"));
            options.work = Some(self.parent.join(format!("{label}-work")));
            options.receipt = self.parent.join(format!("{label}.json"));
            options.batch_size = 1;
            options
        }
    }

    async fn fixture(
        cx: &Cx,
        topology: NativeGenerationTopology,
        ann: bool,
        empty: bool,
    ) -> Fixture {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
            ".frankensearch-authority-repair-{}-{stamp}",
            std::process::id(),
        ));
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        let root = create_root(&parent.join("root")).expect("qualified Linux root required");
        let tier = if topology == NativeGenerationTopology::QualityOnly {
            EmbedderTierTag::Quality
        } else {
            EmbedderTierTag::Fast
        };
        let primary = Provider::new(tier == EmbedderTierTag::Quality);
        let quality =
            (topology == NativeGenerationTopology::FullProgressive).then(|| Provider::new(true));
        let retrieval = if ann {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 17,
            }
        } else {
            NativeBuildRetrieval::Exact
        };
        let mut builder = NativeIndexBuilder::new(
            parent.join("original-work"),
            new_generation(1).unwrap(),
            primary.clone(),
        )
        .unwrap()
        .with_fast_storage(
            if tier == EmbedderTierTag::Fast {
                NativeBuildPrecision::F16
            } else {
                NativeBuildPrecision::F32
            },
            retrieval,
        );
        if let Some(quality) = &quality {
            builder = builder
                .with_quality_embedder(quality.clone())
                .unwrap()
                .with_quality_storage(NativeBuildPrecision::F32, retrieval)
                .unwrap();
        }
        if !empty {
            builder = builder.add_documents([
                IndexableDocument::new("retry", "retry network requests — keep source text")
                    .with_title("Retries 🦀")
                    .with_metadata("revision", "before")
                    .with_metadata("quoted", "a\"b\nc"),
                IndexableDocument::new("garden", "garden flowers").with_metadata("empty", ""),
            ]);
        }
        let source = CommitRange { low: 4, high: 9 };
        let built = if quality.is_some() {
            builder
                .build_progressive_generation(cx, source, DEFAULT_IMAGE_BYTES)
                .await
                .unwrap()
        } else {
            builder
                .build_exact_generation(cx, tier, source, DEFAULT_IMAGE_BYTES)
                .await
                .unwrap()
        };
        let published = stage_and_publish(
            cx,
            &root,
            &parent.join("root"),
            &parent.join("selected.json"),
            &built,
            None,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap();
        // Repair cannot depend on the old work tree or original input files.
        fs::rename(
            parent.join("original-work"),
            parent.join("unavailable-work"),
        )
        .unwrap();
        Fixture {
            parent,
            primary,
            quality,
            selected: published.receipt,
            snapshot: published.snapshot,
        }
    }

    fn damage(cx: &Cx, fixture: &Fixture, role: GenerationComponentRole, missing: bool) -> PathBuf {
        let receipts = fixture.snapshot.retained_snapshot().manifest().components();
        let receipt = receipts
            .iter()
            .find_map(|(candidate, receipt)| (candidate == role).then_some(receipt))
            .unwrap();
        let path = fixture
            .selected
            .root
            .join(component_object_name_v1(role, receipt.sha256));
        fs::rename(&path, fixture.parent.join(format!("original-{role:?}"))).unwrap();
        if !missing {
            write_immutable(cx, &path, b"damaged-component-evidence").unwrap();
        }
        path
    }

    #[test]
    fn repair_rebuilds_damaged_or_missing_derived_images_for_every_topology_and_empty_source() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            use NativeGenerationTopology::{FastOnly, FullProgressive, QualityOnly};
            for (topology, ann, role, missing, empty) in [
                (
                    FastOnly,
                    false,
                    GenerationComponentRole::Vector,
                    false,
                    false,
                ),
                (
                    QualityOnly,
                    false,
                    GenerationComponentRole::Vector,
                    true,
                    false,
                ),
                (
                    FullProgressive,
                    false,
                    GenerationComponentRole::Vector,
                    false,
                    false,
                ),
                (FastOnly, true, GenerationComponentRole::Ann, true, false),
                (
                    QualityOnly,
                    true,
                    GenerationComponentRole::Ann,
                    false,
                    false,
                ),
                (
                    FullProgressive,
                    true,
                    GenerationComponentRole::Vector,
                    true,
                    false,
                ),
                (
                    FullProgressive,
                    true,
                    GenerationComponentRole::Ann,
                    false,
                    false,
                ),
                (FastOnly, false, GenerationComponentRole::Vector, true, true),
                (
                    FullProgressive,
                    false,
                    GenerationComponentRole::Vector,
                    false,
                    true,
                ),
            ] {
                let fixture = fixture(&cx, topology, ann, empty).await;
                let original_source = serde_json::to_value(
                    fixture
                        .snapshot
                        .source_documents(&cx, MAX_INPUT_BYTES)
                        .unwrap(),
                )
                .unwrap();
                let original_manifest = fixture.snapshot.retained_snapshot().manifest();
                assert!(
                    original_manifest.predecessor().is_none(),
                    "generation one needs no healthy ancestor"
                );
                let path = damage(&cx, &fixture, role, missing);
                assert!(fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                let before_calls = fixture.primary.calls.load(Ordering::SeqCst);
                let quality_calls = fixture
                    .quality
                    .as_ref()
                    .map(|provider| provider.calls.load(Ordering::SeqCst));
                let options = fixture.options("repaired");
                let mut output = Vec::new();
                let mut loads = 0;
                with_loader(&cx, &options, &mut output, |mode| {
                    loads += 1;
                    assert_eq!(mode, fixture.selected.query_mode(None).unwrap());
                    Ok(fixture.models())
                })
                .await
                .unwrap();
                assert_eq!(loads, 1);
                let rows = usize::try_from(fixture.selected.documents).unwrap();
                assert_eq!(
                    fixture.primary.calls.load(Ordering::SeqCst),
                    before_calls + rows
                );
                if let Some(provider) = &fixture.quality {
                    assert_eq!(
                        provider.calls.load(Ordering::SeqCst),
                        quality_calls.unwrap() + rows
                    );
                }
                let receipt = Receipt::read(&options.receipt).unwrap();
                let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(reopened.topology(), topology);
                assert_eq!(receipt.identity, fixture.selected.identity);
                assert_eq!(receipt.quality_identity, fixture.selected.quality_identity);
                assert_eq!(
                    receipt.generation.sequence,
                    fixture.selected.generation.sequence + 1
                );
                assert_ne!(receipt.generation.nonce, fixture.selected.generation.nonce);
                assert_eq!(
                    receipt.authority().unwrap().sequence,
                    fixture.selected.authority().unwrap().sequence + 1
                );
                assert_eq!(
                    reopened.retrieval_configuration(),
                    NativeBuildRetrieval::Exact
                );
                assert_eq!(
                    reopened.quality_retrieval_configuration(),
                    fixture
                        .quality
                        .as_ref()
                        .map(|_| NativeBuildRetrieval::Exact)
                );
                assert_eq!(
                    serde_json::to_value(reopened.source_documents(&cx, MAX_INPUT_BYTES).unwrap())
                        .unwrap(),
                    original_source
                );
                assert_eq!(
                    reopened.metadata().commit_range,
                    fixture.snapshot.metadata().commit_range
                );
                let repaired = reopened.retained_snapshot().manifest();
                assert_eq!(
                    repaired.predecessor(),
                    Some(fixture.selected.authority().unwrap())
                );
                assert_eq!(
                    repaired.document_set_sha256(),
                    original_manifest.document_set_sha256()
                );
                assert_eq!(
                    repaired.source_checkpoint_sha256(),
                    original_manifest.source_checkpoint_sha256()
                );
                assert!(repaired.components().ann.is_none());
                let action = match repaired {
                    ActivationManifest::V1(manifest) => manifest.action,
                    ActivationManifest::V2(manifest) => manifest.action,
                };
                assert_eq!(action, GenerationAuthorityActionV1::Repair);
                if missing {
                    assert!(!path.exists());
                } else {
                    assert_eq!(fs::read(&path).unwrap(), b"damaged-component-evidence");
                }
                let hits = if let Some(quality) = &fixture.quality {
                    reopened
                        .search_refined_text(
                            &cx,
                            fixture.primary.as_ref(),
                            quality.as_ref(),
                            "retry",
                            2,
                        )
                        .await
                        .unwrap()
                } else {
                    reopened
                        .search_text(&cx, fixture.primary.as_ref(), "retry", 2)
                        .await
                        .unwrap()
                };
                assert_eq!(hits.is_empty(), empty);
                if !empty {
                    assert!(hits.iter().any(|hit| hit.doc_id == "retry"));
                }
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["event"], "repaired");
                assert_eq!(report["publication"], "committed");
                assert_eq!(report["serving"], "activated");
                assert_eq!(report["model_migrated"], false);
                assert_eq!(report["retrieval"], "exact");
            }
        });
    }

    #[test]
    fn invalid_source_receipts_and_source_or_image_limits_refuse_before_model_loading() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for role in [
                GenerationComponentRole::Lexical,
                GenerationComponentRole::Metadata,
            ] {
                let fixture = fixture(&cx, NativeGenerationTopology::FastOnly, true, false).await;
                damage(&cx, &fixture, role, false);
                let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let options = fixture.options("damaged-source");
                let mut loads = 0;
                assert!(
                    with_loader(&cx, &options, &mut Vec::new(), |_| {
                        loads += 1;
                        Ok(fixture.models())
                    })
                    .await
                    .is_err()
                );
                assert_eq!(loads, 0);
                assert!(!options.work.as_ref().unwrap().exists());
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    before
                );
            }
            let fixture =
                fixture(&cx, NativeGenerationTopology::FullProgressive, true, false).await;
            damage(&cx, &fixture, GenerationComponentRole::Vector, true);
            let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
            for case in [
                "source-budget",
                "image-budget",
                "count",
                "generation",
                "primary",
                "quality",
            ] {
                let mut options = fixture.options(case);
                match case {
                    "source-budget" => options.max_source_bytes = 1,
                    "image-budget" => options.max_image_bytes = 1,
                    _ => {
                        let mut receipt = fixture.selected.clone();
                        match case {
                            "count" => receipt.documents += 1,
                            "generation" => receipt.generation = new_generation(1).unwrap(),
                            "primary" => {
                                receipt.identity = fixture.primary.foreign.clone().freeze().unwrap()
                            }
                            "quality" => {
                                receipt.quality_identity = Some(
                                    fixture
                                        .quality
                                        .as_ref()
                                        .unwrap()
                                        .foreign
                                        .clone()
                                        .freeze()
                                        .unwrap(),
                                )
                            }
                            _ => unreachable!(),
                        }
                        // Keep the receipt's persisted storage contract while
                        // changing only the explicitly tested producer.
                        if case == "primary" {
                            let mut identity = receipt.identity.identity.clone();
                            identity.storage = fixture.selected.identity.identity.storage.clone();
                            receipt.identity = identity.freeze().unwrap();
                        } else if case == "quality" {
                            let mut identity =
                                receipt.quality_identity.as_ref().unwrap().identity.clone();
                            identity.storage = fixture
                                .selected
                                .quality_identity
                                .as_ref()
                                .unwrap()
                                .identity
                                .storage
                                .clone();
                            receipt.quality_identity = Some(identity.freeze().unwrap());
                        }
                        let path = fixture.parent.join(format!("wrong-{case}.json"));
                        save_receipt(&receipt, &path).unwrap();
                        options.from_receipt = Some(path);
                    }
                }
                let mut loads = 0;
                assert!(
                    with_loader(&cx, &options, &mut Vec::new(), |_| {
                        loads += 1;
                        Ok(fixture.models())
                    })
                    .await
                    .is_err(),
                    "case {case}"
                );
                assert_eq!(loads, 0, "case {case}");
                assert!(!options.work.as_ref().unwrap().exists());
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    before
                );
            }
        });
    }

    #[test]
    fn model_mismatch_drift_failure_and_cancellation_never_publish_partial_repairs() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture =
                fixture(&cx, NativeGenerationTopology::FullProgressive, true, false).await;
            damage(&cx, &fixture, GenerationComponentRole::Ann, false);
            let before = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
            for case in [
                "wrong-primary",
                "wrong-quality",
                "missing-quality",
                "load-cancel",
                "inference-error",
                "inference-cancel",
                "drift",
            ] {
                let primary = Provider::new(false);
                let quality = Provider::new(true);
                match case {
                    "wrong-primary" => primary.drifted.store(true, Ordering::SeqCst),
                    "wrong-quality" => quality.drifted.store(true, Ordering::SeqCst),
                    "inference-error" => quality.fault.store(1, Ordering::SeqCst),
                    "inference-cancel" => quality.fault.store(2, Ordering::SeqCst),
                    "drift" => quality.fault.store(3, Ordering::SeqCst),
                    _ => {}
                }
                let options = fixture.options(case);
                let error = with_loader(&cx, &options, &mut Vec::new(), |_| {
                    if case == "load-cancel" {
                        cx.set_cancel_requested(true);
                        return Err(bad("loader failure must not hide cancellation"));
                    }
                    Ok(Models {
                        fast: primary.clone(),
                        quality: (case != "missing-quality")
                            .then(|| quality.clone() as Arc<dyn Embedder>),
                    })
                })
                .await
                .unwrap_err();
                if matches!(case, "load-cancel" | "inference-cancel") {
                    assert!(matches!(
                        error.downcast_ref::<SearchError>(),
                        Some(SearchError::Cancelled { .. })
                    ));
                }
                cx.set_cancel_requested(false);
                if matches!(
                    case,
                    "wrong-primary" | "wrong-quality" | "missing-quality" | "load-cancel"
                ) {
                    assert_eq!(primary.calls.load(Ordering::SeqCst), 0);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                    assert!(!options.work.as_ref().unwrap().exists());
                }
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    before
                );
            }
        });
    }

    #[test]
    fn publication_during_model_loading_refuses_original_repair_pair_without_rebasing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = fixture(&cx, NativeGenerationTopology::FastOnly, false, false).await;
            let winner = NativeIndexBuilder::new(
                fixture.parent.join("winner-work"),
                new_generation(2).unwrap(),
                fixture.primary.clone(),
            )
            .unwrap()
            .with_fast_storage(NativeBuildPrecision::F16, NativeBuildRetrieval::Exact)
            .add_documents(
                fixture
                    .snapshot
                    .source_documents(&cx, MAX_INPUT_BYTES)
                    .unwrap(),
            )
            .build_exact_generation(
                &cx,
                EmbedderTierTag::Fast,
                CommitRange { low: 4, high: 10 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            damage(&cx, &fixture, GenerationComponentRole::Vector, true);
            let root = admit_root(&fixture.selected.root).unwrap();
            let original_pair = fixture.snapshot.retained_snapshot().authority_pair();
            let options = fixture.options("stale-repair");
            let mut winning_receipt = None;
            let error = with_loader(&cx, &options, &mut Vec::new(), |_| {
                let published = stage_and_publish(
                    &cx,
                    &root,
                    &fixture.selected.root,
                    &fixture.parent.join("winner.json"),
                    &winner,
                    Some(&fixture.snapshot),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )?;
                winning_receipt = Some(published.receipt);
                Ok(fixture.models())
            })
            .await
            .unwrap_err();
            let failure = error.downcast_ref::<PublicationFailure>().unwrap();
            assert_eq!(failure.0["publication"], "not_committed");
            let winner = winning_receipt.unwrap();
            let current = winner.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(current.metadata().commit_range.high, 10);
            let attempt = Receipt::read(&options.receipt)
                .unwrap()
                .recovery_attempt()
                .unwrap();
            assert_eq!(attempt.expected, original_pair);
            assert_ne!(attempt.candidate, winner.authority().unwrap());
            assert!(
                Receipt::read(&options.receipt)
                    .unwrap()
                    .open(&cx, DEFAULT_IMAGE_BYTES)
                    .is_err()
            );
            // A later invocation with the old receipt must refuse before loading.
            let retry = fixture.options("stale-input");
            let mut loads = 0;
            assert!(
                with_loader(&cx, &retry, &mut Vec::new(), |_| {
                    loads += 1;
                    Ok(fixture.models())
                })
                .await
                .is_err()
            );
            assert_eq!(loads, 0);
            assert!(!retry.receipt.exists());
        });
    }

    #[test]
    fn lost_success_response_keeps_committed_repair_receipt_recoverable_without_models() {
        struct BrokenOutput;
        impl Write for BrokenOutput {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("injected output failure"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = fixture(&cx, NativeGenerationTopology::QualityOnly, true, false).await;
            damage(&cx, &fixture, GenerationComponentRole::Ann, true);
            let options = fixture.options("lost-output");
            let error = with_loader(&cx, &options, &mut BrokenOutput, |_| Ok(fixture.models()))
                .await
                .unwrap_err();
            let failure = error.downcast_ref::<PublicationFailure>().unwrap();
            assert_eq!(failure.0["publication"], "committed");
            assert_eq!(failure.0["serving"], "activated");
            let receipt = Receipt::read(&options.receipt).unwrap();
            receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let authority = fs::read(receipt.root.join("AUTHORITY")).unwrap();
            let calls = fixture.primary.calls.load(Ordering::SeqCst);
            let mut recovery = Options::parse(
                ["authority-recover", "--receipt", "placeholder.json"].map(str::to_owned),
            )
            .unwrap()
            .unwrap();
            recovery.receipt = options.receipt;
            let mut output = Vec::new();
            recover(&cx, &recovery, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["serving"], "activated");
            assert_eq!(fixture.primary.calls.load(Ordering::SeqCst), calls);
            assert_eq!(fs::read(receipt.root.join("AUTHORITY")).unwrap(), authority);
        });
    }
}
