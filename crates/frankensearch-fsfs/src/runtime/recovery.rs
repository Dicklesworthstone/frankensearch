// Receipt-directed admission and recovery share the existing runtime's private
// search paths. This file is included in the same module as the retained reader.
// No directory scan, alternate engine or implicit fallback chooses a target.

fn retained_recovery_error(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "complete_generation.recovery".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

/// An explicitly chosen, searchable recovery target bound to the observed head.
///
/// Preparation opens every stored tier with its real producer before returning.
/// It releases publication ownership before model loading or preview queries,
/// so it cannot block a watcher while an operator inspects the target. A later
/// publisher invalidates this preparation rather than having its work replaced.
/// Dropping this value never restores a generation or deletes any artifacts.
///
/// This is intentional rollback/repair for a cooperative complete-generation
/// store, not the fixed-authority or external antirollback-floor protocol.
#[must_use]
pub struct PreparedRetainedRecovery {
    plan: crate::generation_store::PreparedGenerationRestore,
    reader: RetainedSearchReader,
}

impl std::fmt::Debug for PreparedRetainedRecovery {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedRetainedRecovery")
            .field("generation", &self.reader.generation().id())
            .field("manifest_sha256", &self.reader.generation().manifest_sha256())
            // Do not expose the potentially malformed incumbent descriptor.
            .finish_non_exhaustive()
    }
}

impl PreparedRetainedRecovery {
    /// The exact admitted target, not a newly discovered predecessor.
    #[must_use]
    pub const fn generation(&self) -> &crate::generation_store::PublishedGeneration {
        self.reader.generation()
    }

    /// Preview the retained target without changing the active selection.
    ///
    /// # Errors
    /// Returns producer drift, retrieval, configuration or cancellation errors.
    pub async fn search(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        self.reader.validate_retained_recovery_resources(cx)?;
        self.reader.search(cx, query, limit).await
    }

    /// Abandon restoration and keep only the explicitly admitted reader.
    #[must_use]
    pub fn into_retained(self) -> RetainedSearchReader {
        self.reader
    }

    /// Revalidate and select this target, preserving every retained directory.
    ///
    /// The existing store protocol fences the exact incumbent descriptor and
    /// authenticates the target again under publication ownership. Producer
    /// identities are rechecked against the already opened vector resources.
    /// A stale preparation is refused, never automatically rebased or retried.
    ///
    /// The returned reader is the one already admitted, not a second pathname
    /// reopen. Inspect the publication variant before reporting durability:
    /// `VisibleButDurabilityUncertain` means selection is already visible and
    /// must not be described as an aborted restore. No fallible operation or
    /// cancellation checkpoint follows the store's publication result.
    /// Synchronous filesystem barriers belong on the caller's blocking lane.
    ///
    /// # Errors
    /// Before publication, returns contention, changed selection/target/producer,
    /// I/O or cancellation errors without selecting this target.
    pub fn restore(
        self,
        cx: &Cx,
    ) -> SearchResult<(
        crate::generation_store::GenerationPublication,
        RetainedSearchReader,
    )> {
        retained_search_checkpoint(cx)?;
        let Self { plan, reader } = self;
        let publication = plan.restore(cx, |cx, path| {
            if path != reader.generation().path() {
                return Err(retained_recovery_error(
                    "recovery admission and publication name different targets",
                ));
            }
            reader.validate_retained_recovery_resources(cx)
        })?;
        Ok((publication, reader))
    }
}

impl FsfsRuntime {
    /// Prepare a searchable rollback or repair using a trusted publication receipt.
    ///
    /// Capture the expected selection before asynchronous model admission, so a
    /// concurrent publisher cannot be overwritten after loading or previewing.
    /// No active-generation scan or implicit fallback chooses the target. An
    /// absent, malformed or damaged current selection does not prevent recovery
    /// to the explicitly named intact bundle. Unsafe descriptors remain errors.
    ///
    /// Preparation does not switch selection. Preview with the returned plan's
    /// `search`, then call `restore` explicitly or retain the reader without
    /// restoring. Every stored tier must admit its configured producer; a
    /// fast-only query policy cannot waive quality-tier recovery validation.
    ///
    /// # Errors
    /// Returns invalid receipt, contention, corrupt bundle, missing/mismatched
    /// producer, configuration, I/O or cancellation errors.
    pub async fn prepare_retained_recovery(
        &self,
        cx: &Cx,
        store_root: &Path,
        generation_id: &str,
        manifest_sha256: &str,
    ) -> SearchResult<PreparedRetainedRecovery> {
        self.preflight_retained_recovery(cx)?;
        let store = crate::generation_store::CompleteGenerationStore::open(cx, store_root)?;
        let generation = store.open_retained(cx, generation_id, manifest_sha256)?;
        let plan = store.prepare_restore(cx, &generation)?;
        let reader = self
            .open_retained_recovery_target(cx, &store, generation)
            .await?;
        retained_search_checkpoint(cx)?;
        Ok(PreparedRetainedRecovery { plan, reader })
    }

    /// Open an explicitly named retained generation without changing selection.
    ///
    /// The ID and manifest SHA-256 must come from a trusted publication receipt,
    /// not from hashing a damaged directory to make it pass. This works even
    /// when the current selection is absent, malformed or names a damaged bundle.
    /// It never scans for a predecessor or silently restores one.
    ///
    /// Unlike ordinary progressive query admission, this eagerly verifies every
    /// present vector tier and its configured producer, even with `fast_only`.
    /// A valid inventory alone does not establish that a bundle is searchable.
    /// Legacy Tantivy migration and shadow writes are refused; no source rescan,
    /// index mutation, daemon forwarding or persistent query cache is performed.
    /// The cooperative immutable-directory contract of `CompleteGenerationStore`
    /// still applies; this is not the fixed-authority/antirollback protocol.
    ///
    /// # Errors
    /// Returns invalid receipt, corrupt artifact, missing/mismatched producer,
    /// unsupported migration, configuration, I/O or cancellation errors.
    pub async fn open_retained_search_from_receipt(
        &self,
        cx: &Cx,
        store_root: &Path,
        generation_id: &str,
        manifest_sha256: &str,
    ) -> SearchResult<RetainedSearchReader> {
        self.preflight_retained_recovery(cx)?;
        let store = crate::generation_store::CompleteGenerationStore::open(cx, store_root)?;
        let generation = store.open_retained(cx, generation_id, manifest_sha256)?;
        self.open_retained_recovery_target(cx, &store, generation)
            .await
    }

    fn preflight_retained_recovery(&self, cx: &Cx) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        if self.config.search.shadow_mode {
            return Err(retained_recovery_error(
                "shadow observation cannot write into a retained recovery target",
            ));
        }
        Ok(())
    }

    async fn open_retained_recovery_target(
        &self,
        cx: &Cx,
        store: &crate::generation_store::CompleteGenerationStore,
        generation: crate::generation_store::PublishedGeneration,
    ) -> SearchResult<RetainedSearchReader> {
        self.preflight_retained_recovery(cx)?;
        if Self::resolve_lexical_engine(generation.path())?.engine()
            == Some(BlueGreenEngine::Tantivy)
        {
            // The ordinary opener can migrate Tantivy by reading current source
            // files. That would both mutate this seal and restore different data.
            return Err(retained_recovery_error(
                "recovery requires a Quill or vector-only bundle; migrate legacy lexical data through an isolated rebuild",
            ));
        }
        let mut runtime = self.clone();
        runtime.cli_input.index_dir = Some(generation.path().to_path_buf());
        runtime.cli_input.daemon = false;
        runtime.cli_input.daemon_socket = None;
        let mut resources = Box::pin(
            runtime.prepare_search_execution_resources_at_root_with_modes(
                cx,
                generation.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        retained_search_checkpoint(cx)?;
        if let Some(index) = resources.vector_index.as_ref() {
            let embedder = runtime.resolve_fast_embedder()?;
            Self::admit_vector_generation_for_embedder(index, embedder.as_ref())?;
            resources.fast_embedder = Some(AdmittedEmbedder::admit(embedder)?);
            resources.fast_embedder_attempted = true;
        }
        let quality_path = generation.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE);
        let has_quality = match fs::symlink_metadata(&quality_path) {
            Ok(_) => true,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if has_quality {
            if resources.quality_vector_index.is_none() {
                // Ordinary queries may degrade when this optional tier cannot
                // open. A recovery target must not be certified on that fallback.
                return Err(retained_recovery_error(
                    "the retained quality artifact could not be admitted; recovery cannot discard a stored tier",
                ));
            }
            runtime
                .maybe_prepare_quality_embedder(cx, &mut resources)
                .await?;
        }
        let reader = RetainedSearchReader {
            runtime,
            generation,
            resources,
        };
        reader.validate_retained_recovery_resources(cx)?;
        // Model loading can yield. Re-authenticate the exact inventory after it,
        // without requiring the unrelated active selection to remain readable.
        store.open_retained(
            cx,
            reader.generation().id(),
            reader.generation().manifest_sha256(),
        )?;
        retained_search_checkpoint(cx)?;
        Ok(reader)
    }
}

impl RetainedSearchReader {
    fn validate_retained_recovery_resources(&self, cx: &Cx) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        FsfsRuntime::validate_search_generation_fingerprint(
            self.generation.path(),
            &self.resources.generation_fingerprint,
            SearchExecutionMode::Full,
        )?;
        if let Some(index) = self.resources.vector_index.as_ref() {
            let admitted = self.resources.fast_embedder.as_ref().ok_or_else(|| {
                retained_recovery_error("the stored fast tier has no admitted producer")
            })?;
            FsfsRuntime::admit_vector_generation_for_embedder(index, admitted.embedder())?;
            admitted.check_current_identity()?;
        }
        if let Some(index) = self.resources.quality_vector_index.as_ref() {
            let admitted = self.resources.quality_embedder.as_ref().ok_or_else(|| {
                retained_recovery_error("the stored quality tier has no admitted producer")
            })?;
            FsfsRuntime::admit_quality_generation_for_embedder(index, admitted.embedder())?;
            admitted.check_current_identity()?;
        }
        retained_search_checkpoint(cx)
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod retained_recovery_tests {
    use super::*;
    use crate::generation_store::{
        COMPLETE_GENERATION_POINTER, CompleteGenerationStore, GenerationPublication,
        PublishedGeneration,
    };
    use frankensearch_core::EmbeddingIdentityBundleV1;

    // Distinct deterministic providers test both-tier admission and lifecycle,
    // not real-model semantic quality or cross-platform numerical conformance.
    struct RecoveryProducer {
        identity: EmbeddingIdentityBundleV1,
        changed_identity: EmbeddingIdentityBundleV1,
        changed: AtomicBool,
        category: ModelCategory,
    }

    impl RecoveryProducer {
        fn new(name: &str, category: ModelCategory) -> Self {
            let identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 3);
            let mut changed_identity = identity.clone();
            changed_identity
                .producer
                .implementation_revision
                .push_str("-changed");
            Self {
                identity,
                changed_identity,
                changed: AtomicBool::new(false),
                category,
            }
        }
    }

    impl Embedder for RecoveryProducer {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.changed.load(Ordering::Relaxed) {
                &self.changed_identity
            } else {
                &self.identity
            })
        }

        fn embed<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
        ) -> frankensearch_core::SearchFuture<'a, Vec<f32>> {
            Box::pin(async { Ok(vec![1.0, 0.0, 0.0]) })
        }

        fn dimension(&self) -> usize {
            3
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
            self.category
        }
    }

    struct RecoveryModels {
        old_fast: Option<Arc<dyn Embedder>>,
        old_quality: Option<Arc<dyn Embedder>>,
        fast: Arc<RecoveryProducer>,
        quality: Arc<RecoveryProducer>,
    }

    impl RecoveryModels {
        fn install() -> Self {
            let fast = Arc::new(RecoveryProducer::new(
                "recovery-fast",
                ModelCategory::StaticEmbedder,
            ));
            let quality = Arc::new(RecoveryProducer::new(
                "recovery-quality",
                ModelCategory::TransformerEmbedder,
            ));
            let models = Self {
                old_fast: test_fast_embedder_override(),
                old_quality: test_quality_embedder_override(),
                fast,
                quality,
            };
            set_test_fast_embedder(Some(models.fast.clone()));
            set_test_quality_embedder(Some(models.quality.clone()));
            models
        }
    }

    impl Drop for RecoveryModels {
        fn drop(&mut self) {
            set_test_fast_embedder(self.old_fast.take());
            set_test_quality_embedder(self.old_quality.take());
        }
    }

    fn on_runtime<F, Fut>(test: F)
    where
        F: FnOnce(Cx) -> Fut,
        Fut: Future<Output = ()>,
    {
        let scheduler = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        scheduler.block_on(async move {
            test(Cx::current().expect("runtime context")).await;
        });
    }

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), "sharedtoken original alpha document").unwrap();
        let mut config = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        "recovery-quality".clone_into(&mut config.indexing.quality_model);
        config.indexing.offline = true;
        config.search.fast_only = false;
        config.search.rerank = false;
        config.search.quality_timeout_ms = 5_000;
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        (runtime, source, root)
    }

    fn fresh_runtime(runtime: &FsfsRuntime) -> FsfsRuntime {
        // Do not let a previous successful load mask a changed provider fixture.
        FsfsRuntime::new(runtime.config().clone()).with_cli_input(runtime.cli_input.clone())
    }

    async fn publish(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> PublishedGeneration {
        let result = runtime.rebuild_retained_generation(cx, root).await.unwrap();
        let GenerationPublication::Durable(generation) = result else {
            panic!("fixture must publish durably"); // ubs:ignore — cfg(test) assertion.
        };
        generation
    }

    fn file_bytes(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut files = BTreeMap::new();
        let mut directories = vec![root.to_path_buf()];
        while let Some(directory) = directories.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    directories.push(entry.path());
                } else {
                    files.insert(
                        entry.path().strip_prefix(root).unwrap().to_path_buf(),
                        fs::read(entry.path()).unwrap(),
                    );
                }
            }
        }
        files
    }

    #[test]
    fn receipt_reader_reopens_with_corrupt_selection_without_restoring_it() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(source.join("beta.md"), "sharedtoken successor beta document").unwrap();
            let new = publish(&runtime, &cx, &root).await;
            let old_bytes = file_bytes(old.path());
            let new_bytes = file_bytes(new.path());
            fs::rename(&source, directory.path().join("source-unavailable")).unwrap();
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"damaged selection").unwrap();
            let reopened = fresh_runtime(&runtime);
            let mut reader = reopened
                .open_retained_search_from_receipt(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            assert_eq!(reader.generation(), &old);
            assert!(reader.resources.fast_embedder.is_some());
            assert!(reader.resources.quality_embedder.is_some());
            let result = reader.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(result.last().unwrap().hits.len(), 1);
            assert_eq!(result.last().unwrap().hits[0].path, "alpha.md");
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                b"damaged selection"
            );
            assert_eq!(file_bytes(old.path()), old_bytes);
            assert_eq!(file_bytes(new.path()), new_bytes);
            assert!(reopened.open_retained_search(&cx, &root).await.is_err());
        });
    }

    #[test]
    fn receipt_reader_rejects_forged_receipts_and_changed_bundles_without_fallback() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(source.join("beta.md"), "sharedtoken successor beta document").unwrap();
            let new = publish(&runtime, &cx, &root).await;
            let before = file_bytes(&root);
            for (id, digest) in [
                (old.id(), "0".repeat(64)),
                ("../escape", old.manifest_sha256().to_owned()),
            ] {
                assert!(
                    runtime
                        .open_retained_search_from_receipt(&cx, &root, id, &digest)
                        .await
                        .is_err()
                );
                assert_eq!(file_bytes(&root), before);
            }
            fs::write(old.path().join("unsealed-file"), b"tamper").unwrap();
            let damaged = file_bytes(&root);
            assert!(
                runtime
                    .open_retained_search_from_receipt(&cx, &root, old.id(), old.manifest_sha256())
                    .await
                    .is_err()
            );
            assert_eq!(file_bytes(&root), damaged);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(new)
            );
        });
    }

    #[test]
    fn receipt_reader_checks_both_producers_even_when_search_is_fast_only() {
        on_runtime(|cx| async move {
            let models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            assert!(generation.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE).is_file());
            runtime.config.search.fast_only = true;
            let before = file_bytes(&root);
            for producer in [&models.fast, &models.quality] {
                producer.changed.store(true, Ordering::Relaxed);
                let error = fresh_runtime(&runtime)
                    .open_retained_search_from_receipt(
                        &cx,
                        &root,
                        generation.id(),
                        generation.manifest_sha256(),
                    )
                    .await
                    .unwrap_err();
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                producer.changed.store(false, Ordering::Relaxed);
                assert_eq!(file_bytes(&root), before);
            }
            set_test_quality_embedder(None);
            assert!(
                fresh_runtime(&runtime)
                    .open_retained_search_from_receipt(
                        &cx,
                        &root,
                        generation.id(),
                        generation.manifest_sha256(),
                    )
                    .await
                    .is_err()
            );
            assert_eq!(file_bytes(&root), before);
        });
    }

    #[test]
    fn recovery_rejects_a_sealed_but_unreadable_quality_artifact_instead_of_degrading() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            let store = CompleteGenerationStore::create(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            let mut input = runtime.cli_input.clone();
            input.index_dir = Some(build.path().to_path_buf());
            let candidate = runtime.clone().with_cli_input(input);
            candidate
                .run_retained_index_with_reuse(&cx, &store, build.path())
                .await
                .unwrap();
            let quality = build.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE);
            assert!(quality.is_file());
            fs::write(&quality, b"invalid quality vector header").unwrap();
            // Intentionally exercise only the container's inventory seal here.
            // This is NOT a valid engine publication: the recovery adapter must
            // supply the missing engine/producer admission and refuse it.
            let GenerationPublication::Durable(target) =
                build.publish(&cx, |_, _| Ok(())).unwrap()
            else {
                panic!("fixture inventory must be durable"); // ubs:ignore — cfg(test) assertion.
            };
            assert_eq!(store.active(&cx).unwrap(), Some(target.clone()));
            runtime.config.search.fast_only = true;
            let before = file_bytes(target.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            assert!(
                runtime
                    .prepare_retained_recovery(
                        &cx,
                        &root,
                        target.id(),
                        target.manifest_sha256(),
                    )
                    .await
                    .is_err()
            );
            assert_eq!(file_bytes(target.path()), before);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
        });
    }

    #[test]
    fn receipt_reader_refuses_unsafe_configuration_and_cancellation_before_side_effects() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            let before = file_bytes(&root);
            for shadow in [false, true] {
                let mut invalid = fresh_runtime(&runtime);
                if shadow {
                    invalid.config.search.shadow_mode = true;
                } else {
                    "{index_dir}/../outside.sqlite"
                        .clone_into(&mut invalid.config.storage.db_path);
                }
                assert!(
                    invalid
                        .open_retained_search_from_receipt(
                            &cx,
                            &root,
                            generation.id(),
                            generation.manifest_sha256(),
                        )
                        .await
                        .is_err()
                );
                assert_eq!(file_bytes(&root), before);
            }
            cx.set_cancel_requested(true);
            assert!(matches!(
                runtime
                    .open_retained_search_from_receipt(
                        &cx,
                        &root,
                        generation.id(),
                        generation.manifest_sha256(),
                    )
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(file_bytes(&root), before);
            assert!(!root.join("outside.sqlite").exists());
        });
    }

    #[test]
    fn prepared_recovery_previews_then_restores_without_retargeting_pinned_readers() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(source.join("beta.md"), "sharedtoken successor beta document").unwrap();
            let new = publish(&runtime, &cx, &root).await;
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            let mut live = runtime.open_live_retained_search(&cx, &root).await.unwrap();
            assert_eq!(
                live.search(&cx, "sharedtoken", 10).await.unwrap()[0].hits.len(),
                2
            );
            let old_bytes = file_bytes(old.path());
            let new_bytes = file_bytes(new.path());
            let before = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            fs::rename(&source, directory.path().join("source-unavailable")).unwrap();
            let mut plan = fresh_runtime(&runtime)
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            assert_eq!(plan.generation(), &old);
            let preview = plan.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(preview.last().unwrap().hits.len(), 1);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                before
            );
            let (publication, mut recovered) = plan.restore(&cx).unwrap();
            assert!(
                matches!(publication, GenerationPublication::Durable(ref value) if value == &old)
            );
            assert_eq!(recovered.generation(), &old);
            assert_eq!(
                recovered.search(&cx, "sharedtoken", 10).await.unwrap()[0].hits.len(),
                1
            );
            assert!(live.refresh(&cx).await.unwrap());
            assert_eq!(live.generation(), &old);
            assert_eq!(
                live.search(&cx, "sharedtoken", 10).await.unwrap()[0].hits.len(),
                1
            );
            assert_eq!(pinned.generation(), &new);
            assert_eq!(
                pinned.search(&cx, "sharedtoken", 10).await.unwrap()[0].hits.len(),
                2
            );
            assert_eq!(file_bytes(old.path()), old_bytes);
            assert_eq!(file_bytes(new.path()), new_bytes);
            let mut fresh = fresh_runtime(&runtime)
                .open_retained_search(&cx, &root)
                .await
                .unwrap();
            assert_eq!(fresh.generation(), &old);
            assert_eq!(
                fresh.search(&cx, "sharedtoken", 10).await.unwrap()[0].hits.len(),
                1
            );
        });
    }

    #[test]
    fn prepared_recovery_repairs_corrupt_and_missing_selection_without_deleting_evidence() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(source.join("beta.md"), "sharedtoken successor beta document").unwrap();
            let damaged = publish(&runtime, &cx, &root).await;
            fs::write(damaged.path().join("unsealed-damage"), b"retained evidence").unwrap();
            let evidence = file_bytes(damaged.path());
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert!(store.active(&cx).is_err());
            let plan = runtime
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            assert!(matches!(
                plan.restore(&cx).unwrap().0,
                GenerationPublication::Durable(_)
            ));
            assert_eq!(store.active(&cx).unwrap(), Some(old.clone()));
            assert_eq!(file_bytes(damaged.path()), evidence);
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            for missing in [false, true] {
                if missing {
                    fs::rename(&pointer, root.join("saved-recovery-pointer")).unwrap();
                } else {
                    fs::write(&pointer, b"private malformed descriptor").unwrap();
                }
                let plan = fresh_runtime(&runtime)
                    .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                    .await
                    .unwrap();
                assert!(!format!("{plan:?}").contains("private malformed descriptor"));
                assert!(matches!(
                    plan.restore(&cx).unwrap().0,
                    GenerationPublication::Durable(_)
                ));
                assert_eq!(store.active(&cx).unwrap(), Some(old.clone()));
                assert_eq!(file_bytes(damaged.path()), evidence);
            }
            assert!(root.join("saved-recovery-pointer").is_file());
            assert_eq!(fs::read_dir(root.join("generations")).unwrap().count(), 2);
        });
    }

    #[test]
    fn prepared_recovery_does_not_hold_publication_ownership_or_overwrite_a_successor() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let plan = runtime
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            fs::write(source.join("beta.md"), "sharedtoken concurrent publication").unwrap();
            // A real publication while the plan is alive proves it owns no lease.
            let winner = publish(&runtime, &cx, &root).await;
            let before = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let winner_bytes = file_bytes(winner.path());
            assert!(plan.restore(&cx).is_err());
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                before
            );
            assert_eq!(file_bytes(winner.path()), winner_bytes);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(winner)
            );
        });
    }

    #[test]
    fn prepared_recovery_rechecks_both_producers_at_commit_and_can_retry() {
        on_runtime(|cx| async move {
            let models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            runtime.config.search.fast_only = true;
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            fs::write(&pointer, b"recovery pending").unwrap();
            let before = file_bytes(old.path());
            for producer in [&models.fast, &models.quality] {
                let plan = fresh_runtime(&runtime)
                    .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                    .await
                    .unwrap();
                producer.changed.store(true, Ordering::Relaxed);
                let error = plan.restore(&cx).unwrap_err();
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                producer.changed.store(false, Ordering::Relaxed);
                assert_eq!(fs::read(&pointer).unwrap(), b"recovery pending");
                assert_eq!(file_bytes(old.path()), before);
            }
            let plan = fresh_runtime(&runtime)
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            let (publication, reader) = plan.restore(&cx).unwrap();
            assert!(matches!(publication, GenerationPublication::Durable(_)));
            assert_eq!(reader.generation(), &old);
            assert_eq!(file_bytes(old.path()), before);
        });
    }

    #[test]
    fn prepared_recovery_cancellation_and_target_damage_cannot_publish() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(source.join("beta.md"), "sharedtoken successor beta document").unwrap();
            let current = publish(&runtime, &cx, &root).await;
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let plan = runtime
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(plan.restore(&cx), Err(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            let plan = runtime
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            fs::write(old.path().join("changed-after-admission"), b"damage").unwrap();
            let before = file_bytes(old.path());
            assert!(plan.restore(&cx).is_err());
            assert_eq!(file_bytes(old.path()), before);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(current)
            );
        });
    }

    #[test]
    fn abandoning_prepared_recovery_keeps_a_searchable_target_without_selection_change() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, _, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            fs::write(&pointer, b"explicit repair required").unwrap();
            let plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            let mut reader = plan.into_retained();
            assert_eq!(reader.generation(), &generation);
            assert_eq!(
                reader.search(&cx, "sharedtoken", 10).await.unwrap()[0].hits.len(),
                1
            );
            assert_eq!(fs::read(pointer).unwrap(), b"explicit repair required");
        });
    }
}
