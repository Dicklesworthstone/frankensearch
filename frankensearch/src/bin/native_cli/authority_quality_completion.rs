//! Complete or incrementally expand quality coverage using one retained authority.

use frankensearch_index::vector_cohort::VectorCohortCoverage;

use super::{
    BlockingPoolHandle, Cx, IndexableDocument, MAX_DOCUMENTS, MAX_RECORD_BYTES, Models,
    NativeBuildPrecision, NativeGenerationSnapshot, NativeGenerationTopology, Options,
    PublicationIntent, PublicationOptions, QueryMode, Receipt, Result, Write, admit_root, bad,
    checkpoint, emit_indexed, execution_context, load_authority_models, new_generation, new_path,
    prepare_partial_quality_completion, prepare_publication_with_intent, rebuild_precision,
    retained_rebuild_documents, separate_destinations, stage_prepared_and_publish,
};

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
    if options.quality_ids.is_some() && options.max_new_quality_documents.is_some() {
        return Err(bad(
            "--quality-ids and --max-new-quality-documents cannot be combined",
        ));
    }
    if options
        .max_new_quality_documents
        .is_some_and(|limit| limit == 0 || limit > MAX_DOCUMENTS)
    {
        return Err(bad("the new quality document limit is out of range"));
    }
    let selected = Receipt::read(
        options
            .from_receipt
            .as_deref()
            .ok_or_else(|| bad("missing predecessor receipt"))?,
    )?;
    if !matches!(
        selected.topology(),
        NativeGenerationTopology::FastOnly | NativeGenerationTopology::PartialQuality { .. }
    ) {
        return Err(bad(
            "quality completion requires a fast-only or partial-quality authority receipt",
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
    let documents = retained_rebuild_documents(cx, &snapshot, options.max_source_bytes)?;
    if let Some(ids) = &options.quality_ids {
        validate_backfill_selection(cx, &snapshot, &documents, ids)?;
    }
    if options.max_new_quality_documents.is_some() {
        validate_next_batch_source(cx, &snapshot, &documents)?;
    }
    drop(documents);
    // A partial predecessor already owns independently admitted quality rows.
    // Preserve their encoding so completion can reuse them exactly; only a new
    // quality tier receives the CLI's default F32 storage policy.
    let quality_precision = selected
        .quality_identity
        .as_ref()
        .map(rebuild_precision)
        .transpose()?
        .unwrap_or(NativeBuildPrecision::F32);
    let quality_retrieval = options.build_retrieval(snapshot.quality_retrieval_configuration());
    let loaded = load_models();
    checkpoint(cx)?;
    let loaded = loaded?;
    let quality = loaded.quality.ok_or_else(|| {
        bad("quality completion requires an independently supplied quality model")
    })?;
    let completion = snapshot
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
        .with_quality_storage(quality_precision, quality_retrieval)?;
    let built = if let Some(ids) = &options.quality_ids {
        completion
            .build_partial(cx, ids.iter().cloned(), options.max_image_bytes)
            .await?
    } else if let Some(limit) = options.max_new_quality_documents {
        completion
            .build_next_batch(cx, limit, options.max_image_bytes)
            .await?
    } else {
        completion.build(cx, options.max_image_bytes).await?
    };
    let prepared = if built.coverage() == VectorCohortCoverage::PartialQuality {
        prepare_partial_quality_completion(cx, &built, &snapshot, options.max_image_bytes)?
    } else {
        prepare_publication_with_intent(
            &built,
            Some(&snapshot),
            options.max_image_bytes,
            PublicationIntent::CompleteQuality,
        )?
    };
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

fn validate_next_batch_source(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    documents: &[IndexableDocument],
) -> Result<()> {
    checkpoint(cx)?;
    let covered: std::collections::BTreeSet<_> = match snapshot.topology() {
        NativeGenerationTopology::FastOnly => std::collections::BTreeSet::new(),
        NativeGenerationTopology::PartialQuality { .. } => snapshot
            .partial_quality_document_ids(cx)?
            .into_iter()
            .collect(),
        NativeGenerationTopology::QualityOnly | NativeGenerationTopology::FullProgressive => {
            return Err(bad("a quality batch requires incomplete quality coverage"));
        }
    };
    if !documents
        .iter()
        .any(|document| !covered.contains(&document.id))
    {
        return Err(bad(
            "a quality batch requires at least one uncovered source document",
        ));
    }
    checkpoint(cx)?;
    Ok(())
}

fn validate_backfill_selection(
    cx: &Cx,
    snapshot: &NativeGenerationSnapshot,
    documents: &[IndexableDocument],
    ids: &[String],
) -> Result<()> {
    checkpoint(cx)?;
    let selected: std::collections::BTreeSet<_> = ids.iter().map(String::as_str).collect();
    if selected.is_empty() || selected.len() != ids.len() || selected.len() >= documents.len() {
        return Err(bad(
            "quality backfill requires a unique nonempty proper subset; omit --quality-ids for full completion",
        ));
    }
    let source_ids: std::collections::BTreeSet<_> = documents
        .iter()
        .map(|document| document.id.as_str())
        .collect();
    if !selected.is_subset(&source_ids) {
        return Err(bad(
            "quality backfill IDs must belong to the selected source",
        ));
    }
    let previous = match snapshot.topology() {
        NativeGenerationTopology::FastOnly => Vec::new(),
        NativeGenerationTopology::PartialQuality { .. } => {
            snapshot.partial_quality_document_ids(cx)?
        }
        NativeGenerationTopology::QualityOnly | NativeGenerationTopology::FullProgressive => {
            return Err(bad("quality backfill requires incomplete quality coverage"));
        }
    };
    if selected.len() <= previous.len() || previous.iter().any(|id| !selected.contains(id.as_str()))
    {
        return Err(bad(
            "quality backfill must retain every covered document and add new IDs",
        ));
    }
    checkpoint(cx)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{Command, MAX_DOCUMENTS, NativeBuildRetrieval, is_command};
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
    fn bounded_completion_accepts_only_explicit_unique_quality_membership() {
        let mut args = arguments();
        args.extend(["--quality-ids", r#"["garden","retry"]"#].map(str::to_owned));
        let options = Options::parse(args).unwrap().unwrap();
        assert_eq!(
            options.quality_ids,
            Some(vec!["garden".to_owned(), "retry".to_owned()])
        );
        for selection in [
            "[]",
            r#"["retry","retry"]"#,
            r#"[""]"#,
            r#"["\u0000"]"#,
            r#"{"id":"retry"}"#,
        ] {
            let mut args = arguments();
            args.extend(["--quality-ids", selection].map(str::to_owned));
            assert!(Options::parse(args).is_err(), "{selection}");
        }
    }

    #[test]
    fn automatic_completion_accepts_a_positive_bounded_count_without_explicit_ids() {
        for limit in [1, MAX_DOCUMENTS] {
            let mut args = arguments();
            args.extend(["--max-new-quality-documents".to_owned(), limit.to_string()]);
            let options = Options::parse(args.clone()).unwrap().unwrap();
            assert_eq!(options.max_new_quality_documents, Some(limit));
            assert!(options.quality_ids.is_none());
            args.extend(["--quality-ids", r#"["garden"]"#].map(str::to_owned));
            assert!(Options::parse(args).is_err());
        }
        for value in [
            "0".to_owned(),
            (MAX_DOCUMENTS + 1).to_string(),
            usize::MAX.to_string(),
            "-1".to_owned(),
            "not-a-count".to_owned(),
        ] {
            let mut args = arguments();
            args.extend(["--max-new-quality-documents".to_owned(), value]);
            assert!(Options::parse(args).is_err());
        }
        let mut args = arguments();
        args.push("--max-new-quality-documents".to_owned());
        assert!(Options::parse(args).is_err());
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
        use super::super::super::{
            CommitRange, DEFAULT_IMAGE_BYTES, Embedder, EmbedderTierTag, HnswParams,
            IndexableDocument, MAX_INPUT_BYTES, NativeGenerationSnapshot, NativeIndexBuilder,
            PublicationFailure, SearchError, create_root, prepare_publication, stage_and_publish,
        };
        use super::*;
        use frankensearch_core::generation::{
            EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
            GenerationComponentRole,
        };
        use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
        use std::fs;
        use std::os::unix::fs::DirBuilderExt as _;
        use std::path::PathBuf;
        use std::sync::Arc;
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
            fixture_with_quality(cx, ann, None).await
        }

        async fn fixture_with_quality(
            cx: &Cx,
            ann: bool,
            quality: Option<Arc<Provider>>,
        ) -> Fixture {
            fixture_with_documents(
                cx,
                ann,
                quality,
                vec![
                    IndexableDocument::new("retry", "retry network requests")
                        .with_metadata("revision", "old"),
                    IndexableDocument::new("garden", "garden flowers"),
                ],
            )
            .await
        }

        async fn fixture_with_documents(
            cx: &Cx,
            ann: bool,
            quality: Option<Arc<Provider>>,
            documents: Vec<IndexableDocument>,
        ) -> Fixture {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let parent = PathBuf::from(
                std::env::var_os("FRANKENSEARCH_TEST_TMPDIR")
                    .or_else(|| std::env::var_os("HOME"))
                    .unwrap(),
            )
            .join(format!(
                ".frankensearch-quality-completion-{}-{stamp}",
                std::process::id(),
            ));
            fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
            let root = create_root(&parent.join("root")).expect("qualified Linux root required");
            let fast = Provider::new(false);
            let builder = NativeIndexBuilder::new(
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
            .add_documents(documents);
            let built = if let Some(quality) = quality {
                builder
                    .with_quality_embedder(quality)
                    .unwrap()
                    .with_quality_storage(
                        NativeBuildPrecision::F16,
                        if ann {
                            NativeBuildRetrieval::Hnsw {
                                params: HnswParams::default(),
                                seed: 29,
                            }
                        } else {
                            NativeBuildRetrieval::Exact
                        },
                    )
                    .unwrap()
                    .build_partial_quality_generation(
                        cx,
                        CommitRange { low: 4, high: 9 },
                        ["garden".to_owned()],
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap()
            } else {
                builder
                    .build_exact_generation(
                        cx,
                        EmbedderTierTag::Fast,
                        CommitRange { low: 4, high: 9 },
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap()
            };
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

        fn backfill_documents() -> Vec<IndexableDocument> {
            vec![
                IndexableDocument::new("retry", "retry network requests")
                    .with_metadata("revision", "old"),
                IndexableDocument::new("garden", "garden flowers"),
                IndexableDocument::new("notes", "field notes"),
                IndexableDocument::new("uncovered", "a final document"),
            ]
        }

        #[test]
        fn automatic_completion_resumes_from_actual_membership_and_finishes_full() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    for starts_partial in [false, true] {
                        let quality = Provider::new(true);
                        let mut documents = backfill_documents();
                        // The existing covered ID is not the first source ID.
                        // Resumption must inspect membership, not use its count as a cursor.
                        documents.push(IndexableDocument::new("a-first", "alphabetically first"));
                        let fixture = fixture_with_documents(
                            &cx,
                            ann,
                            starts_partial.then(|| quality.clone()),
                            documents,
                        )
                        .await;
                        let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                        let batches = if starts_partial {
                            vec![
                                (1, vec!["a-first", "garden"]),
                                (2, vec!["a-first", "garden", "notes", "retry"]),
                                (8, vec!["a-first", "garden", "notes", "retry", "uncovered"]),
                            ]
                        } else {
                            vec![
                                (1, vec!["a-first"]),
                                (2, vec!["a-first", "garden", "notes"]),
                                (8, vec!["a-first", "garden", "notes", "retry", "uncovered"]),
                            ]
                        };
                        let mut current_path = fixture.parent.join("fast.json");
                        let mut current_receipt = fixture.selected.clone();
                        let mut current = Arc::clone(&fixture.snapshot);
                        let mut covered = usize::from(starts_partial);
                        for (step, (limit, expected_ids)) in batches.into_iter().enumerate() {
                            let mut batch = options(&fixture, &format!("automatic-{step}"));
                            batch.from_receipt = Some(current_path.clone());
                            batch.max_new_quality_documents = Some(limit);
                            batch.batch_size = 1;
                            let quality_calls = quality.calls.load(Ordering::SeqCst);
                            let mut output = Vec::new();
                            with_loader(&cx, &batch, &mut output, || {
                                Ok(Models {
                                    fast: fixture.fast.clone(),
                                    quality: Some(quality.clone()),
                                })
                            })
                            .await
                            .unwrap();
                            let added = expected_ids.len() - covered;
                            assert!(added > 0 && added <= limit);
                            assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls + added);
                            assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                            let receipt = Receipt::read(&batch.receipt).unwrap();
                            let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                            assert_eq!(receipt.identity, fixture.selected.identity);
                            if current_receipt.quality_identity.is_some() {
                                assert_eq!(
                                    receipt.quality_identity,
                                    current_receipt.quality_identity
                                );
                            }
                            assert_eq!(
                                receipt.generation.sequence,
                                current_receipt.generation.sequence + 1
                            );
                            assert_eq!(
                                reopened.retained_snapshot().manifest().predecessor(),
                                Some(current.authority())
                            );
                            assert_eq!(
                                reopened.retained_snapshot().manifest().components().lexical,
                                fixture
                                    .snapshot
                                    .retained_snapshot()
                                    .manifest()
                                    .components()
                                    .lexical,
                            );
                            assert_eq!(
                                reopened.metadata().commit_range,
                                fixture.snapshot.metadata().commit_range
                            );
                            assert_eq!(
                                reopened.retrieval_configuration(),
                                fixture.snapshot.retrieval_configuration()
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
                            if expected_ids.len() == 5 {
                                assert_eq!(receipt.quality_documents, None);
                                assert_eq!(
                                    reopened.topology(),
                                    NativeGenerationTopology::FullProgressive
                                );
                                assert_eq!(
                                    receipt.query_mode(Some(QueryMode::Full)).unwrap(),
                                    QueryMode::Full
                                );
                                assert_eq!(
                                    receipt.query_mode(Some(QueryMode::Quality)).unwrap(),
                                    QueryMode::Quality
                                );
                            } else {
                                assert_eq!(
                                    receipt.quality_documents,
                                    Some(u64::try_from(expected_ids.len()).unwrap())
                                );
                                assert_eq!(
                                    reopened.partial_quality_document_ids(&cx).unwrap(),
                                    expected_ids
                                );
                                assert!(receipt.query_mode(Some(QueryMode::Full)).is_err());
                                assert_eq!(
                                    receipt.query_mode(Some(QueryMode::Partial)).unwrap(),
                                    QueryMode::Partial
                                );
                            }
                            let report: serde_json::Value =
                                serde_json::from_slice(&output).unwrap();
                            assert_eq!(report["publication"], "committed");
                            assert_eq!(
                                report["selection"]["quality_documents"],
                                serde_json::to_value(receipt.quality_documents).unwrap()
                            );
                            covered = expected_ids.len();
                            current_path = batch.receipt;
                            current_receipt = receipt;
                            current = Arc::new(reopened);
                        }
                        // Full and stale receipts cannot load a model for another batch.
                        for (label, path) in [
                            ("already-full", current_path),
                            ("stale", fixture.parent.join("fast.json")),
                        ] {
                            let mut batch = options(&fixture, label);
                            batch.from_receipt = Some(path);
                            batch.max_new_quality_documents = Some(1);
                            let mut loads = 0;
                            assert!(
                                with_loader(&cx, &batch, &mut Vec::new(), || {
                                    loads += 1;
                                    Ok(Models {
                                        fast: fixture.fast.clone(),
                                        quality: Some(quality.clone()),
                                    })
                                })
                                .await
                                .is_err()
                            );
                            assert_eq!(loads, 0);
                            assert!(!batch.work.as_ref().unwrap().exists());
                            assert!(!batch.receipt.exists());
                        }
                        fixture.fast.fault.store(0, Ordering::SeqCst);
                        let hits = current
                            .search_progressive_with_quality(
                                &cx,
                                fixture.fast.as_ref(),
                                quality.as_ref(),
                                "final document",
                                5,
                            )
                            .unwrap()
                            .collect()
                            .await
                            .unwrap();
                        assert!(
                            hits.iter()
                                .find(|hit| hit.doc_id == "uncovered")
                                .unwrap()
                                .quality_score
                                .is_some()
                        );
                        assert_eq!(fixture.snapshot.topology(), fixture.selected.topology());
                    }
                }
            });
        }

        #[test]
        fn automatic_completion_refuses_empty_or_invalid_work_before_loading_models() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for empty in [false, true] {
                    let fixture = fixture_with_documents(
                        &cx,
                        false,
                        None,
                        if empty {
                            Vec::new()
                        } else {
                            backfill_documents()
                        },
                    )
                    .await;
                    let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    for case in if empty {
                        vec!["empty"]
                    } else {
                        vec![
                            "zero",
                            "too-many",
                            "explicit-ids",
                            "source-budget",
                            "cancelled",
                        ]
                    } {
                        let mut batch = options(&fixture, case);
                        batch.max_new_quality_documents = Some(1);
                        match case {
                            "zero" => batch.max_new_quality_documents = Some(0),
                            "too-many" => batch.max_new_quality_documents = Some(MAX_DOCUMENTS + 1),
                            "explicit-ids" => batch.quality_ids = Some(vec!["garden".to_owned()]),
                            "source-budget" => batch.max_source_bytes = 1,
                            "cancelled" => cx.set_cancel_requested(true),
                            _ => {}
                        }
                        let mut loads = 0;
                        let mut output = Vec::new();
                        let result = with_loader(&cx, &batch, &mut output, || {
                            loads += 1;
                            Err(bad("invalid automatic work must not load models"))
                        })
                        .await;
                        cx.set_cancel_requested(false);
                        assert!(result.is_err(), "{case}");
                        assert_eq!(loads, 0, "{case}");
                        assert!(output.is_empty());
                        assert!(!batch.work.as_ref().unwrap().exists());
                        assert!(!batch.receipt.exists());
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(
                            fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                            authority
                        );
                    }
                    fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                }
            });
        }

        #[test]
        fn automatic_completion_retries_failed_batches_and_fences_a_concurrent_final_batch() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, true, Some(quality.clone()), backfill_documents())
                        .await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let failed_quality = Provider::new(true);
                    failed_quality.fault.store(fault, Ordering::SeqCst);
                    let mut batch = options(&fixture, &format!("automatic-fault-{fault}"));
                    batch.max_new_quality_documents = Some(1);
                    batch.batch_size = 1;
                    let mut output = Vec::new();
                    let result = with_loader(&cx, &batch, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(failed_quality.clone()),
                        })
                    })
                    .await;
                    cx.set_cancel_requested(false);
                    assert!(result.is_err());
                    if fault == 2 {
                        assert!(matches!(
                            result.unwrap_err().downcast_ref::<SearchError>(),
                            Some(SearchError::Cancelled { .. })
                        ));
                    }
                    assert_eq!(failed_quality.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert!(output.is_empty());
                    assert!(!batch.receipt.exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    assert_eq!(
                        fixture
                            .selected
                            .open(&cx, DEFAULT_IMAGE_BYTES)
                            .unwrap()
                            .partial_quality_document_ids(&cx)
                            .unwrap(),
                        ["garden"]
                    );
                }
                // Retry with the same receipt: the failed work consumed no IDs.
                let mut retry = options(&fixture, "automatic-retry");
                retry.max_new_quality_documents = Some(1);
                let quality_calls = quality.calls.load(Ordering::SeqCst);
                with_loader(&cx, &retry, &mut Vec::new(), || {
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality.clone()),
                    })
                })
                .await
                .unwrap();
                assert_eq!(quality.calls.load(Ordering::SeqCst), quality_calls + 1);
                let current_receipt = Receipt::read(&retry.receipt).unwrap();
                let current = current_receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    current.partial_quality_document_ids(&cx).unwrap(),
                    ["garden", "notes"]
                );

                let winner = current
                    .begin_quality_completion(
                        &cx,
                        fixture.parent.join("automatic-winner-work"),
                        new_generation(current_receipt.generation.sequence + 1).unwrap(),
                        fixture.fast.clone(),
                        quality.clone(),
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build_partial(
                        &cx,
                        ["garden", "notes", "retry"].map(str::to_owned),
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap();
                let prepared =
                    prepare_partial_quality_completion(&cx, &winner, &current, DEFAULT_IMAGE_BYTES)
                        .unwrap();
                let root = admit_root(&current_receipt.root).unwrap();
                let expected = current.retained_snapshot().authority_pair();
                let mut losing = options(&fixture, "automatic-stale-final");
                losing.from_receipt = Some(retry.receipt.clone());
                losing.max_new_quality_documents = Some(8);
                let mut winning_receipt = None;
                let mut output = Vec::new();
                let error = with_loader(&cx, &losing, &mut output, || {
                    let published = stage_prepared_and_publish(
                        &cx,
                        &root,
                        &current_receipt.root,
                        &fixture.parent.join("automatic-winner.json"),
                        &winner,
                        prepared,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )?;
                    winning_receipt = Some(published.receipt);
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality.clone()),
                    })
                })
                .await
                .unwrap_err();
                assert_eq!(
                    error.downcast_ref::<PublicationFailure>().unwrap().0["publication"],
                    "not_committed"
                );
                assert!(output.is_empty());
                assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                let winner = winning_receipt.unwrap();
                let reopened = winner.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    reopened.partial_quality_document_ids(&cx).unwrap(),
                    ["garden", "notes", "retry"]
                );
                let attempted = Receipt::read(&losing.receipt).unwrap();
                assert_eq!(
                    attempted.quality_documents, None,
                    "the losing final batch was complete"
                );
                assert_eq!(attempted.recovery_attempt().unwrap().expected, expected);
                assert_ne!(attempted.authority().unwrap(), winner.authority().unwrap());
                assert!(attempted.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                assert_eq!(
                    current.partial_quality_document_ids(&cx).unwrap(),
                    ["garden", "notes"]
                );
            });
        }

        #[test]
        fn bounded_completion_publishes_truthful_partial_receipts_and_resumes_to_full() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    for starts_partial in [false, true] {
                        let quality = Provider::new(true);
                        let fixture = fixture_with_documents(
                            &cx,
                            ann,
                            starts_partial.then(|| quality.clone()),
                            backfill_documents(),
                        )
                        .await;
                        let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                        let mut bounded = options(&fixture, "bounded");
                        bounded.quality_ids =
                            Some(["garden", "notes", "retry"].map(str::to_owned).into());
                        let mut output = Vec::new();
                        with_loader(&cx, &bounded, &mut output, || {
                            Ok(Models {
                                fast: fixture.fast.clone(),
                                quality: Some(quality.clone()),
                            })
                        })
                        .await
                        .unwrap();
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(quality.calls.load(Ordering::SeqCst), 3);
                        let receipt = Receipt::read(&bounded.receipt).unwrap();
                        assert_eq!(receipt.quality_documents, Some(3));
                        assert_eq!(receipt.identity, fixture.selected.identity);
                        assert_eq!(
                            receipt.query_mode(Some(QueryMode::Partial)).unwrap(),
                            QueryMode::Partial
                        );
                        assert!(receipt.query_mode(Some(QueryMode::Full)).is_err());
                        let partial = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                        assert_eq!(
                            partial.topology(),
                            NativeGenerationTopology::PartialQuality {
                                quality_documents: 3,
                                total_documents: 4,
                                coverage_ppm: 750_000,
                            }
                        );
                        assert_eq!(
                            partial.partial_quality_document_ids(&cx).unwrap(),
                            vec!["garden".to_owned(), "notes".to_owned(), "retry".to_owned()]
                        );
                        assert_eq!(
                            partial.retained_snapshot().manifest().components().lexical,
                            fixture
                                .snapshot
                                .retained_snapshot()
                                .manifest()
                                .components()
                                .lexical,
                        );
                        assert_eq!(
                            partial.metadata().commit_range,
                            fixture.snapshot.metadata().commit_range
                        );
                        assert_eq!(
                            partial.retrieval_configuration(),
                            fixture.snapshot.retrieval_configuration()
                        );
                        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                        assert_eq!(report["publication"], "committed");
                        assert_eq!(report["selection"]["quality_documents"], 3);

                        let mut complete = options(&fixture, "resumed-full");
                        complete.from_receipt = Some(bounded.receipt.clone());
                        with_loader(&cx, &complete, &mut Vec::new(), || {
                            Ok(Models {
                                fast: fixture.fast.clone(),
                                quality: Some(quality.clone()),
                            })
                        })
                        .await
                        .unwrap();
                        assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                        assert_eq!(quality.calls.load(Ordering::SeqCst), 4);
                        let complete = Receipt::read(&complete.receipt).unwrap();
                        let full = complete.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                        assert_eq!(full.topology(), NativeGenerationTopology::FullProgressive);
                        assert_eq!(complete.quality_identity, receipt.quality_identity);
                        assert_eq!(complete.quality_documents, None);
                        // Retained readers remain pinned to their original coverage.
                        assert_eq!(partial.partial_quality_document_ids(&cx).unwrap().len(), 3);
                        assert_eq!(fixture.snapshot.topology(), fixture.selected.topology());
                    }
                }
            });
        }

        #[test]
        fn bounded_completion_refuses_invalid_or_regressing_membership_before_loading() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, false, Some(quality.clone()), backfill_documents())
                        .await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                let selections = [
                    vec![],
                    vec!["garden", "garden"],
                    vec!["garden", "unknown"],
                    vec!["garden", "notes", "retry", "uncovered"],
                    vec!["garden"],
                    vec!["notes", "retry"],
                ];
                for (index, selection) in selections.into_iter().enumerate() {
                    let mut options = options(&fixture, &format!("invalid-backfill-{index}"));
                    options.quality_ids = Some(selection.into_iter().map(str::to_owned).collect());
                    let loaded = AtomicUsize::new(0);
                    let mut output = Vec::new();
                    assert!(
                        with_loader(&cx, &options, &mut output, || {
                            loaded.fetch_add(1, Ordering::SeqCst);
                            Ok(Models {
                                fast: fixture.fast.clone(),
                                quality: Some(quality.clone()),
                            })
                        })
                        .await
                        .is_err()
                    );
                    assert_eq!(loaded.load(Ordering::SeqCst), 0);
                    assert!(output.is_empty());
                    assert!(!options.work.as_ref().unwrap().exists());
                    assert!(!options.receipt.exists());
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                }
                assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            });
        }

        #[test]
        fn bounded_completion_publication_authenticates_membership_instead_of_only_counts() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let seed = fixture_with_documents(&cx, false, None, backfill_documents()).await;
                let quality = Provider::new(true);
                // Derive both sides from one admitted fast source. Rebuilding
                // Quill independently would introduce wall-clock manifest bytes
                // and could fail source equality before the membership guard.
                let mut candidates = Vec::new();
                for (label, ids) in [
                    ("same-count", vec!["retry"]),
                    ("dropped-old", vec!["notes", "retry"]),
                ] {
                    let built = seed
                        .snapshot
                        .begin_quality_completion(
                            &cx,
                            seed.parent.join(format!("substituted-{label}")),
                            new_generation(3).unwrap(),
                            seed.fast.clone(),
                            quality.clone(),
                            MAX_INPUT_BYTES,
                        )
                        .unwrap()
                        .build_partial(&cx, ids.into_iter().map(str::to_owned), DEFAULT_IMAGE_BYTES)
                        .await
                        .unwrap();
                    candidates.push((label, built));
                }
                let first = seed
                    .snapshot
                    .begin_quality_completion(
                        &cx,
                        seed.parent.join("selected-partial-work"),
                        new_generation(2).unwrap(),
                        seed.fast.clone(),
                        quality,
                        MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .build_partial(&cx, ["garden".to_owned()], DEFAULT_IMAGE_BYTES)
                    .await
                    .unwrap();
                let prepared = prepare_partial_quality_completion(
                    &cx,
                    &first,
                    &seed.snapshot,
                    DEFAULT_IMAGE_BYTES,
                )
                .unwrap();
                let root = admit_root(&seed.selected.root).unwrap();
                let selected = stage_prepared_and_publish(
                    &cx,
                    &root,
                    &seed.selected.root,
                    &seed.parent.join("selected-partial.json"),
                    &first,
                    prepared,
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap();
                let authority = fs::read(seed.selected.root.join("AUTHORITY")).unwrap();
                for (label, built) in candidates {
                    // A fully valid candidate with matching source and producers
                    // must still prove it retained the actual old quality member.
                    assert_eq!(
                        built.component_bytes(GenerationComponentRole::Lexical),
                        selected
                            .snapshot
                            .retained_snapshot()
                            .closure()
                            .bytes(GenerationComponentRole::Lexical),
                    );
                    prepare_publication_with_intent(
                        &built,
                        Some(&selected.snapshot),
                        DEFAULT_IMAGE_BYTES,
                        PublicationIntent::BackfillQuality,
                    )
                    .unwrap();
                    let error = prepare_partial_quality_completion(
                        &cx,
                        &built,
                        &selected.snapshot,
                        DEFAULT_IMAGE_BYTES,
                    )
                    .err()
                    .expect("substituted quality coverage must refuse publication");
                    assert!(
                        error
                            .to_string()
                            .contains("quality_backfill_coverage_not_expanded"),
                        "{label}: {error}",
                    );
                    assert_eq!(
                        fs::read(seed.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    selected.receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                }
            });
        }

        #[test]
        fn bounded_completion_faults_leave_the_exact_predecessor_selected() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let old_quality = Provider::new(true);
                let fixture =
                    fixture_with_documents(&cx, true, Some(old_quality), backfill_documents())
                        .await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let mut options = options(&fixture, &format!("bounded-fault-{fault}"));
                    options.quality_ids = Some(["garden", "retry"].map(str::to_owned).into());
                    let mut output = Vec::new();
                    let result = with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await;
                    cx.set_cancel_requested(false);
                    assert!(result.is_err());
                    assert!(quality.calls.load(Ordering::SeqCst) > 0);
                    assert!(output.is_empty());
                    assert!(!options.receipt.exists());
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority
                    );
                    let retained = fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        retained.partial_quality_document_ids(&cx).unwrap(),
                        vec!["garden".to_owned()]
                    );
                }
            });
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
        fn partial_completion_reuses_covered_rows_and_inherits_quality_storage_and_retrieval() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for ann in [false, true] {
                    let quality = Provider::new(true);
                    let fixture = fixture_with_quality(&cx, ann, Some(quality.clone())).await;
                    let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                    let options = options(&fixture, "complete-partial");
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
                    assert_eq!(
                        receipt.topology(),
                        NativeGenerationTopology::FullProgressive
                    );
                    assert_eq!(receipt.identity, fixture.selected.identity);
                    assert_eq!(receipt.quality_identity, fixture.selected.quality_identity);
                    assert_eq!(receipt.quality_documents, None);
                    let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        reopened.quality_retrieval_configuration(),
                        fixture.snapshot.quality_retrieval_configuration(),
                    );
                    assert_eq!(
                        reopened.retained_snapshot().manifest().components().lexical,
                        fixture
                            .snapshot
                            .retained_snapshot()
                            .manifest()
                            .components()
                            .lexical,
                    );
                    assert_eq!(
                        reopened.metadata().commit_range,
                        fixture.snapshot.metadata().commit_range,
                    );
                    // Completion makes a previously uncovered document eligible
                    // for direct quality retrieval after an independent reopen.
                    let hits = reopened
                        .search_quality_text(&cx, quality.as_ref(), "retry", 1)
                        .await
                        .unwrap();
                    assert_eq!(hits[0].doc_id, "retry");
                    assert!(matches!(
                        fixture.snapshot.topology(),
                        NativeGenerationTopology::PartialQuality {
                            quality_documents: 1,
                            total_documents: 2,
                            ..
                        }
                    ));
                    fixture.fast.fault.store(0, Ordering::SeqCst);
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
                    assert!(report["selection"]["quality_identity"].is_object());
                    assert!(report["selection"]["quality_documents"].is_null());
                }
            });
        }

        #[test]
        fn partial_completion_refuses_foreign_quality_before_inference_and_preserves_selection() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture = fixture_with_quality(&cx, false, Some(quality.clone())).await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                let foreign = Provider::new(true);
                foreign.drifted.store(true, Ordering::SeqCst);
                let options = options(&fixture, "foreign-quality");
                assert!(
                    with_loader(&cx, &options, &mut Vec::new(), || Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(foreign.clone()),
                    }))
                    .await
                    .is_err()
                );
                assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
                assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert!(!options.work.unwrap().exists());
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    authority,
                );
                fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            });
        }

        #[test]
        fn partial_completion_inference_failure_cancellation_and_drift_preserve_coverage() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let original_quality = Provider::new(true);
                let fixture =
                    fixture_with_quality(&cx, false, Some(original_quality.clone())).await;
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let fast_calls = fixture.fast.calls.load(Ordering::SeqCst);
                for fault in [1, 2, 3] {
                    let quality = Provider::new(true);
                    quality.fault.store(fault, Ordering::SeqCst);
                    let options = options(&fixture, &format!("partial-fault-{fault}"));
                    let mut output = Vec::new();
                    let result = with_loader(&cx, &options, &mut output, || {
                        Ok(Models {
                            fast: fixture.fast.clone(),
                            quality: Some(quality.clone()),
                        })
                    })
                    .await;
                    cx.set_cancel_requested(false);
                    assert!(result.is_err());
                    assert!(output.is_empty());
                    assert!(!options.receipt.exists());
                    assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(fixture.fast.calls.load(Ordering::SeqCst), fast_calls);
                    assert_eq!(original_quality.calls.load(Ordering::SeqCst), 1);
                    assert_eq!(
                        fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                        authority,
                    );
                    let reopened = fixture.selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(reopened.topology(), fixture.snapshot.topology());
                    assert!(matches!(
                        reopened.topology(),
                        NativeGenerationTopology::PartialQuality {
                            quality_documents: 1,
                            total_documents: 2,
                            ..
                        }
                    ));
                }
            });
        }

        #[test]
        fn partial_completion_competing_winner_fences_stale_candidate_and_requires_explicit_intent()
        {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let quality = Provider::new(true);
                let fixture = fixture_with_quality(&cx, true, Some(quality.clone())).await;
                let completed = fixture
                    .snapshot
                    .begin_quality_completion(
                        &cx,
                        fixture.parent.join("stale-work"),
                        new_generation(2).unwrap(),
                        fixture.fast.clone(),
                        quality.clone(),
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
                let winner_options = options(&fixture, "winner-completion");
                with_loader(&cx, &winner_options, &mut Vec::new(), || {
                    Ok(Models {
                        fast: fixture.fast.clone(),
                        quality: Some(quality.clone()),
                    })
                })
                .await
                .unwrap();
                let authority = fs::read(fixture.selected.root.join("AUTHORITY")).unwrap();
                let root = admit_root(&fixture.selected.root).unwrap();
                assert!(
                    stage_prepared_and_publish(
                        &cx,
                        &root,
                        &fixture.selected.root,
                        &fixture.parent.join("stale-completion.json"),
                        &completed,
                        prepared,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )
                    .is_err()
                );
                assert_eq!(
                    fs::read(fixture.selected.root.join("AUTHORITY")).unwrap(),
                    authority,
                );
                let receipt = Receipt::read(&winner_options.receipt).unwrap();
                let reopened = receipt.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    reopened.topology(),
                    NativeGenerationTopology::FullProgressive
                );
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
