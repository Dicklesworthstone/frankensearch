// Receipt-directed admission and recovery share the existing runtime's private
// search paths. This file is included in the same module as the retained reader.
// No directory scan, alternate engine or implicit fallback chooses a target.

mod recovery_query {
    include!("recovery_query.rs");
}

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
    // A source rebuild requires preview; receipt-directed recovery allows none.
    // Every attempted preview must finish successfully.
    // Set false before any await so dropping a polled future cannot authorize a
    // restore, including when an earlier preview completed successfully.
    preview_ready: bool,
}

impl std::fmt::Debug for PreparedRetainedRecovery {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedRetainedRecovery")
            .field("generation", &self.reader.generation().id())
            .field(
                "manifest_sha256",
                &self.reader.generation().manifest_sha256(),
            )
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

    /// Export the original selection guard after admission and any preview.
    ///
    /// Save this fingerprint with `generation().id()` and its manifest digest
    /// in trusted caller storage. [`FsfsRuntime::resume_retained_recovery`] can
    /// then re-admit the same target without recopying or decoding it. Export
    /// never rereads the selection, so it cannot renew a stale preparation.
    /// A failed or abandoned preview must be retried successfully before export.
    ///
    /// This receipt is not a disk-retention pin, a signature, or durable preview
    /// evidence. A resumed process rechecks all producers and performs any preview
    /// it requests anew. A previously exported receipt is not revoked by a later
    /// failure in this process; keep receipt ownership in the caller's workflow.
    ///
    /// # Errors
    /// Returns cancellation, incomplete preview or changed retained resources.
    pub fn expected_selection_sha256(&self, cx: &Cx) -> SearchResult<String> {
        retained_search_checkpoint(cx)?;
        if !self.preview_ready {
            return Err(retained_recovery_error(
                "the recovery preview did not complete successfully; no resume receipt can be exported",
            ));
        }
        self.reader.validate_retained_recovery_resources(cx)?;
        Ok(self.plan.expected_selection_sha256())
    }

    /// Preview the retained target without changing the active selection.
    ///
    /// Once polled, this preview must complete successfully before restoration.
    /// Errors, cancellation, failed refinement and dropping an unfinished future
    /// block `restore`. A successful retry on this same preparation clears that
    /// block without rebasing the captured selection. Merely constructing an
    /// unpolled future has no effect. A nonblank query and positive limit are
    /// required: a no-op query cannot establish a successful recovery preview.
    /// Returned progressive payloads are unchanged.
    ///
    /// # Errors
    /// Returns producer drift, retrieval, configuration or cancellation errors.
    pub async fn search(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        self.preview_ready = false;
        retained_search_checkpoint(cx)?;
        if query.trim().is_empty() || limit == 0 {
            return Err(retained_recovery_error(
                "recovery preview requires a nonblank query and a positive limit",
            ));
        }
        self.reader.validate_retained_recovery_resources(cx)?;
        let result = self.reader.search(cx, query, limit).await;
        retained_search_checkpoint(cx)?;
        let payloads = result?;
        self.preview_ready = match payloads.as_slice() {
            [initial] => initial.phase == SearchOutputPhase::Initial,
            [initial, refined] => {
                initial.phase == SearchOutputPhase::Initial
                    && refined.phase == SearchOutputPhase::Refined
            }
            _ => false,
        };
        Ok(payloads)
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
    /// Any preview that was started must have completed successfully. After a
    /// failed or abandoned preview, retry `search` successfully or prepare a new
    /// explicit recovery; clearing cancellation alone does not authorize restore.
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
    /// incomplete preview, I/O or cancellation errors without selecting this target.
    pub fn restore(
        self,
        cx: &Cx,
    ) -> SearchResult<(
        crate::generation_store::GenerationPublication,
        RetainedSearchReader,
    )> {
        retained_search_checkpoint(cx)?;
        if !self.preview_ready {
            return Err(retained_recovery_error(
                "the recovery preview did not complete successfully; retry the preview or prepare a new explicit recovery",
            ));
        }
        let Self { plan, reader, .. } = self;
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
    /// Rebuild sources into a sealed inactive generation for explicit preview.
    ///
    /// The source directory and query filters/ranking come from this runtime's
    /// CLI input and configuration. Every document is reindexed with locally
    /// available producers; this never downloads models or relabels old vector
    /// bytes. It uses the ordinary source indexer and complete-generation seal.
    /// The store must be outside the source and the catalog generation-local.
    /// A malformed selection or damaged old bundle is captured only as bounded
    /// expected-old evidence; no old artifact is admitted or reused. Ordinary
    /// indexing keeps its strict refusal of a damaged selected generation.
    ///
    /// A successful [`PreparedRetainedRecovery::search`] is mandatory before
    /// exporting a resume receipt or calling `restore`. Until explicit restore,
    /// the predecessor remains selected and readable. The original selection
    /// guard survives indexing, producer admission and preview; a competing
    /// publication is refused, never automatically rebased. Dropping or failing
    /// this future leaves its unselected candidate intact. Reopening the store
    /// after restore uses the exact sealed candidate, not a second source scan.
    ///
    /// This is a cooperative-store source rebuild, not an executor for the
    /// core planner's capability-blocked model-acquisition/authority actions.
    /// Synchronous indexing and filesystem work belongs on a caller's blocking
    /// lane, as for [`Self::rebuild_retained_generation`].
    ///
    /// # Errors
    /// Returns source/configuration, contention, producer, indexing, integrity,
    /// I/O or cancellation errors without changing the active selection.
    #[allow(clippy::future_not_send)]
    pub async fn prepare_retained_rebuild(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<PreparedRetainedRecovery> {
        self.preflight_retained_recovery(cx)?;
        let mut runtime = self.clone();
        runtime.config.indexing.offline = true;
        runtime.cli_input.full_reindex = true;
        let (build, _, candidate_mode) = runtime
            .build_retained_generation(cx, store_root, RetainedBuildKind::Recovery)
            .await?;
        let plan = build.prepare(cx, |_, path| {
            Self::validate_search_generation_at_root(path, candidate_mode)
        })?;
        let store = crate::generation_store::CompleteGenerationStore::open(cx, store_root)?;
        let reader = runtime
            .open_retained_recovery_target(cx, &store, plan.target().clone())
            .await?;
        retained_search_checkpoint(cx)?;
        Ok(PreparedRetainedRecovery {
            plan,
            reader,
            preview_ready: false,
        })
    }

    /// Re-admit a saved preparation without accepting a newer selected head.
    ///
    /// All three receipt fields must come from the same successful preparation:
    /// its target ID, target manifest and exported selection fingerprint. The
    /// store checks that fingerprint under publication ownership BEFORE opening
    /// the target or loading models. The matched plan is retained across every
    /// await, rather than recaptured after model admission. `restore` checks it
    /// again, so publication during loading or preview cannot be overwritten.
    ///
    /// Every stored tier must admit its producer, even for a fast-only preview.
    /// No earlier process's model admission or preview is trusted. No copy,
    /// repair, source rescan or selection write occurs. Dropping this future
    /// leaves selection unchanged; a removed target fails rather than rebuilding.
    /// As with the store's resume API, receipts are caller-owned evidence, not
    /// authentication, persistent retention pins or an antirollback mechanism.
    ///
    /// # Errors
    /// Returns stale/malformed receipt, missing/corrupt target, producer mismatch,
    /// unsupported configuration, I/O or cancellation errors.
    pub async fn resume_retained_recovery(
        &self,
        cx: &Cx,
        store_root: &Path,
        generation_id: &str,
        manifest_sha256: &str,
        expected_selection_sha256: &str,
    ) -> SearchResult<PreparedRetainedRecovery> {
        self.preflight_retained_recovery(cx)?;
        let store = crate::generation_store::CompleteGenerationStore::open(cx, store_root)?;
        let plan = store.resume_restore(
            cx,
            generation_id,
            manifest_sha256,
            expected_selection_sha256,
        )?;
        let generation = plan.target().clone();
        let reader = self
            .open_retained_recovery_target(cx, &store, generation)
            .await?;
        retained_search_checkpoint(cx)?;
        Ok(PreparedRetainedRecovery {
            plan,
            reader,
            preview_ready: true,
        })
    }

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
        Ok(PreparedRetainedRecovery {
            plan,
            reader,
            preview_ready: true,
        })
    }

    /// Repair receipted vector damage in a new generation, then admit every tier.
    ///
    /// This explicitly allocates a complete independent copy; ordinary recovery
    /// and search never invoke it automatically. The original bundle is read-only,
    /// and only the canonical fast/quality vectors can be reconstructed from
    /// intact receipted FEC sidecars. The catalog, lexical and metadata artifacts
    /// must already match the trusted original manifest. No source is reindexed.
    ///
    /// The store captures selection BEFORE copying. Retain that same plan across
    /// asynchronous model admission and optional preview: creating a new restore
    /// plan afterwards could overwrite a publication that won during decoding.
    /// Every configured stored producer is required even for fast-only queries.
    /// Call `restore` explicitly to select the repaired copy; failure or dropping
    /// this future leaves the original selection and damaged evidence intact.
    ///
    /// `max_copy_bytes` limits the observed and original payload inventories;
    /// individual damaged vectors and sidecars retain the store's 128/256 MiB
    /// limits. This is not an RSS bound. Copy/codec work is synchronous and must
    /// run on the caller's blocking lane; cancellation does not preempt a codec
    /// call already executing. Partial unselected copies are retained on failure.
    ///
    /// # Errors
    /// Returns receipt, damage, copy/decode limit, I/O, producer/engine admission,
    /// configuration or cancellation errors. Unix-only directory durability.
    #[cfg(unix)]
    pub async fn prepare_retained_vector_repair(
        &self,
        cx: &Cx,
        store_root: &Path,
        generation_id: &str,
        manifest_sha256: &str,
        max_copy_bytes: u64,
    ) -> SearchResult<PreparedRetainedRecovery> {
        self.preflight_retained_recovery(cx)?;
        let store = crate::generation_store::CompleteGenerationStore::open(cx, store_root)?;
        let plan =
            store.prepare_vector_repair(cx, generation_id, manifest_sha256, max_copy_bytes)?;
        let generation = plan.target().clone();
        let reader = self
            .open_retained_recovery_target(cx, &store, generation)
            .await?;
        retained_search_checkpoint(cx)?;
        Ok(PreparedRetainedRecovery {
            plan,
            reader,
            preview_ready: true,
        })
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
    /// Every query verifies the provider's identity-bound response against that
    /// tier's frozen admission before using its values in the shared ranker.
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
            resources.fast_embedder = Some(AdmittedEmbedder::admit(
                cx,
                recovery_query::bind(embedder)?,
            )?);
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
            if let Some(admitted) = resources.quality_embedder.take() {
                resources.quality_embedder = Some(AdmittedEmbedder::admit(
                    cx,
                    recovery_query::bind(admitted.embedder)?,
                )?);
            }
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
        fs::write(
            source.join("alpha.md"),
            "sharedtoken original alpha document",
        )
        .unwrap();
        let mut config = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        "all-minilm-l6-v2".clone_into(&mut config.indexing.quality_model);
        config.indexing.offline = true;
        config.search.fast_only = false;
        config.search.rerank = false;
        config.search.quality_timeout_ms = 5_000;
        let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        runtime.lexical_only_indexing = false;
        (runtime, source, root)
    }

    fn fresh_runtime(runtime: &FsfsRuntime) -> FsfsRuntime {
        // Do not let a previous successful load mask a changed provider fixture.
        let mut fresh =
            FsfsRuntime::new(runtime.config().clone()).with_cli_input(runtime.cli_input.clone());
        fresh.lexical_only_indexing = runtime.lexical_only_indexing;
        fresh
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
    fn prepared_source_rebuild_requires_preview_then_publishes_exact_queried_snapshot() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let old_bytes = file_bytes(old.path());
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::write(source.join("beta.md"), "sharedtoken rebuilt beta document").unwrap();
            let mut plan = runtime.prepare_retained_rebuild(&cx, &root).await.unwrap();
            let target = plan.generation().clone();
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            assert_eq!(store.active(&cx).unwrap(), Some(old.clone()));
            assert!(plan.expected_selection_sha256(&cx).is_err());
            assert!(plan.search(&cx, "  ", 10).await.is_err());
            assert!(plan.search(&cx, "sharedtoken", 0).await.is_err());
            assert!(plan.expected_selection_sha256(&cx).is_err());
            assert!(plan.reader.resources.fast_embedder.is_some());
            assert!(plan.reader.resources.quality_embedder.is_some());
            let preview = plan.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(preview.last().unwrap().hits.len(), 2);
            assert!(plan.expected_selection_sha256(&cx).is_ok());
            // Preview/restore serves the sealed source snapshot, even when the
            // live source changes after the build. It never rebuilds on replay.
            fs::write(source.join("later.md"), "sharedtoken after preparation").unwrap();
            let (publication, mut recovered) = plan.restore(&cx).unwrap();
            assert!(
                matches!(publication, GenerationPublication::Durable(ref value) if value == &target)
            );
            assert_eq!(store.active(&cx).unwrap(), Some(target));
            assert_eq!(
                recovered
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
            assert_eq!(
                pinned
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            assert_eq!(file_bytes(old.path()), old_bytes);
            let mut fresh = fresh_runtime(&runtime)
                .open_retained_search(&cx, &root)
                .await
                .unwrap();
            assert_eq!(
                fresh
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
        });
    }

    #[test]
    fn prepared_source_rebuild_refuses_skipped_cancelled_and_stale_previews() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let plan = runtime.prepare_retained_rebuild(&cx, &root).await.unwrap();
            assert!(plan.restore(&cx).is_err());
            assert_eq!(store.active(&cx).unwrap(), Some(old.clone()));
            let mut plan = runtime.prepare_retained_rebuild(&cx, &root).await.unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                plan.search(&cx, "sharedtoken", 10).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(plan.restore(&cx).is_err());
            assert_eq!(store.active(&cx).unwrap(), Some(old));
            let mut plan = runtime.prepare_retained_rebuild(&cx, &root).await.unwrap();
            plan.search(&cx, "sharedtoken", 10).await.unwrap();
            fs::write(source.join("winner.md"), "sharedtoken concurrent winner").unwrap();
            let winner = publish(&runtime, &cx, &root).await;
            assert!(plan.restore(&cx).is_err());
            assert_eq!(store.active(&cx).unwrap(), Some(winner));
        });
    }

    #[test]
    fn prepared_source_rebuild_recovers_damaged_selection_without_inheriting_old_bytes() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            for damage in ["fast", "quality", "lexical", "selection"] {
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, root) = fixture(directory.path());
                let old = publish(&runtime, &cx, &root).await;
                let pointer = root.join(COMPLETE_GENERATION_POINTER);
                let damaged_path = match damage {
                    "fast" => old.path().join(FSFS_VECTOR_INDEX_FILE),
                    "quality" => old.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE),
                    "lexical" => {
                        let relative = old
                            .sealed_artifacts()
                            .unwrap()
                            .into_keys()
                            .find(|path| path.ends_with(".fslx"))
                            .expect("fixture has a Quill segment");
                        old.path().join(relative)
                    }
                    "selection" => pointer.clone(),
                    _ => unreachable!(),
                };
                fs::write(&damaged_path, b"corrupt recovery evidence").unwrap();
                let old_bytes = file_bytes(old.path());
                let expected_selection = fs::read(&pointer).unwrap();
                let store = CompleteGenerationStore::open(&cx, &root).unwrap();
                assert!(store.begin(&cx).is_err(), "{damage}");
                fs::write(source.join("beta.md"), "sharedtoken recovered from sources").unwrap();
                let mut plan = runtime.prepare_retained_rebuild(&cx, &root).await.unwrap();
                assert_eq!(fs::read(&pointer).unwrap(), expected_selection, "{damage}");
                assert_eq!(file_bytes(old.path()), old_bytes, "{damage}");
                let preview = plan.search(&cx, "sharedtoken", 10).await.unwrap();
                assert_eq!(preview.last().unwrap().hits.len(), 2, "{damage}");
                let target = plan.generation().clone();
                let (publication, mut reader) = plan.restore(&cx).unwrap();
                assert!(matches!(publication, GenerationPublication::Durable(_)));
                assert_eq!(store.active(&cx).unwrap(), Some(target), "{damage}");
                assert_eq!(file_bytes(old.path()), old_bytes, "{damage}");
                assert_eq!(
                    reader
                        .search(&cx, "sharedtoken", 10)
                        .await
                        .unwrap()
                        .last()
                        .unwrap()
                        .hits
                        .len(),
                    2
                );
            }
        });
    }

    #[test]
    fn receipt_reader_reopens_with_corrupt_selection_without_restoring_it() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(
                source.join("beta.md"),
                "sharedtoken successor beta document",
            )
            .unwrap();
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
            fs::write(
                source.join("beta.md"),
                "sharedtoken successor beta document",
            )
            .unwrap();
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
            assert!(
                generation
                    .path()
                    .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
                    .is_file()
            );
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
            let GenerationPublication::Durable(target) = build.publish(&cx, |_, _| Ok(())).unwrap()
            else {
                panic!("fixture inventory must be durable"); // ubs:ignore — cfg(test) assertion.
            };
            assert_eq!(store.active(&cx).unwrap(), Some(target.clone()));
            runtime.config.search.fast_only = true;
            let before = file_bytes(target.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            assert!(
                runtime
                    .prepare_retained_recovery(&cx, &root, target.id(), target.manifest_sha256(),)
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
                    "{index_dir}/../outside.sqlite".clone_into(&mut invalid.config.storage.db_path);
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
            fs::write(
                source.join("beta.md"),
                "sharedtoken successor beta document",
            )
            .unwrap();
            let new = publish(&runtime, &cx, &root).await;
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            let mut live = runtime.open_live_retained_search(&cx, &root).await.unwrap();
            assert_eq!(
                live.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
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
                recovered.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
                1
            );
            assert!(live.refresh(&cx).await.unwrap());
            assert_eq!(live.generation(), &old);
            assert_eq!(
                live.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
                1
            );
            assert_eq!(pinned.generation(), &new);
            assert_eq!(
                pinned.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
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
                fresh.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
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
            fs::write(
                source.join("beta.md"),
                "sharedtoken successor beta document",
            )
            .unwrap();
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
            fs::write(
                source.join("beta.md"),
                "sharedtoken successor beta document",
            )
            .unwrap();
            let current = publish(&runtime, &cx, &root).await;
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let plan = runtime
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                plan.restore(&cx),
                Err(SearchError::Cancelled { .. })
            ));
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
                reader.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
                1
            );
            assert_eq!(fs::read(pointer).unwrap(), b"explicit repair required");
        });
    }

    async fn protected_publication(
        runtime: &FsfsRuntime,
        cx: &Cx,
        root: &Path,
    ) -> PublishedGeneration {
        use frankensearch_durability::{DefaultSymbolCodec, DurabilityConfig, FileProtector};

        let store = CompleteGenerationStore::create(cx, root).unwrap();
        let build = store.begin(cx).unwrap();
        let mut input = runtime.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        runtime
            .clone()
            .with_cli_input(input)
            .run_retained_index_with_reuse(cx, &store, build.path())
            .await
            .unwrap();
        let protector =
            FileProtector::new(Arc::new(DefaultSymbolCodec), DurabilityConfig::default()).unwrap();
        for relative in [FSFS_VECTOR_INDEX_FILE, FSFS_VECTOR_QUALITY_INDEX_FILE] {
            protector
                .protect_file(&build.path().join(relative))
                .unwrap();
        }
        // The real writers made these files. Admit all engines/producers before
        // introducing damage; the fixture's first seal alone is not that proof.
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("protected fixture must be durable"); // ubs:ignore — test assertion.
        };
        drop(
            runtime
                .open_retained_search_from_receipt(
                    cx,
                    root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap(),
        );
        generation
    }

    fn damage_recovery_vector(generation: &PublishedGeneration, relative: &str) {
        let path = generation.path().join(relative);
        let mut bytes = fs::read(&path).unwrap();
        bytes[10] ^= 0x5a;
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn vector_repair_admits_real_index_arms_and_searches_without_source_files() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let original = protected_publication(&runtime, &cx, &root).await;
            let healthy = file_bytes(original.path());
            damage_recovery_vector(&original, FSFS_VECTOR_INDEX_FILE);
            let quality = original.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE);
            fs::rename(&quality, directory.path().join("preserved-quality.fsvi")).unwrap();
            let damaged = file_bytes(original.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            fs::rename(source, directory.path().join("source-unavailable")).unwrap();
            assert!(runtime.open_retained_search(&cx, &root).await.is_err());
            let mut plan = fresh_runtime(&runtime)
                .prepare_retained_vector_repair(
                    &cx,
                    &root,
                    original.id(),
                    original.manifest_sha256(),
                    1024 * 1024 * 1024,
                )
                .await
                .unwrap();
            let repaired = plan.generation().clone();
            assert_ne!(repaired.id(), original.id());
            assert!(plan.reader.resources.fast_embedder.is_some());
            assert!(plan.reader.resources.quality_embedder.is_some());
            let preview = plan.search(&cx, "sharedtoken", 10).await.unwrap();
            assert!(
                preview
                    .iter()
                    .all(|phase| phase.phase != SearchOutputPhase::RefinementFailed)
            );
            assert_eq!(preview.last().unwrap().hits[0].path, "alpha.md");
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            assert_eq!(file_bytes(original.path()), damaged);
            for relative in [FSFS_VECTOR_INDEX_FILE, FSFS_VECTOR_QUALITY_INDEX_FILE] {
                assert_eq!(
                    fs::read(repaired.path().join(relative)).unwrap(),
                    healthy[Path::new(relative)]
                );
            }
            let (publication, mut retained) = plan.restore(&cx).unwrap();
            assert!(
                matches!(publication, GenerationPublication::Durable(ref value) if value == &repaired)
            );
            assert_eq!(
                retained
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            let mut reopened = fresh_runtime(&runtime)
                .open_retained_search(&cx, &root)
                .await
                .unwrap();
            assert_eq!(reopened.generation(), &repaired);
            assert_eq!(
                reopened
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                1
            );
            assert_eq!(file_bytes(original.path()), damaged);
        });
    }

    #[test]
    fn repaired_copy_still_requires_both_producers_at_admission_and_selection() {
        on_runtime(|cx| async move {
            let models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root) = fixture(directory.path());
            let original = protected_publication(&runtime, &cx, &root).await;
            damage_recovery_vector(&original, FSFS_VECTOR_INDEX_FILE);
            let damaged = file_bytes(original.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            runtime.config.search.fast_only = true;
            for producer in [&models.fast, &models.quality] {
                producer.changed.store(true, Ordering::Relaxed);
                let result = fresh_runtime(&runtime)
                    .prepare_retained_vector_repair(
                        &cx,
                        &root,
                        original.id(),
                        original.manifest_sha256(),
                        1024 * 1024 * 1024,
                    )
                    .await;
                producer.changed.store(false, Ordering::Relaxed);
                assert!(matches!(
                    result,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
                let plan = fresh_runtime(&runtime)
                    .prepare_retained_vector_repair(
                        &cx,
                        &root,
                        original.id(),
                        original.manifest_sha256(),
                        1024 * 1024 * 1024,
                    )
                    .await
                    .unwrap();
                producer.changed.store(true, Ordering::Relaxed);
                let result = plan.restore(&cx);
                producer.changed.store(false, Ordering::Relaxed);
                assert!(matches!(
                    result,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
                assert_eq!(
                    fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                    pointer
                );
                assert_eq!(file_bytes(original.path()), damaged);
            }
        });
    }

    #[test]
    fn repaired_search_plan_keeps_selection_evidence_and_failed_preview_barriers() {
        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let original = protected_publication(&runtime, &cx, &root).await;
            fs::write(source.join("beta.md"), "sharedtoken second source").unwrap();
            let selected = publish(&runtime, &cx, &root).await;
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            damage_recovery_vector(&original, FSFS_VECTOR_INDEX_FILE);
            let plan = fresh_runtime(&runtime)
                .prepare_retained_vector_repair(
                    &cx,
                    &root,
                    original.id(),
                    original.manifest_sha256(),
                    1024 * 1024 * 1024,
                )
                .await
                .unwrap();
            fs::write(source.join("gamma.md"), "sharedtoken concurrent source").unwrap();
            let winner = publish(&runtime, &cx, &root).await;
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            assert!(
                plan.restore(&cx).is_err(),
                "repair cannot overwrite the concurrent publisher"
            );
            let mut plan = fresh_runtime(&runtime)
                .prepare_retained_vector_repair(
                    &cx,
                    &root,
                    original.id(),
                    original.manifest_sha256(),
                    1024 * 1024 * 1024,
                )
                .await
                .unwrap();
            cx.set_cancel_requested(true);
            let preview = plan.search(&cx, "sharedtoken", 10).await;
            cx.set_cancel_requested(false);
            assert!(matches!(preview, Err(SearchError::Cancelled { .. })));
            assert!(
                plan.restore(&cx).is_err(),
                "clearing cancellation cannot authorize a failed preview"
            );
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(winner)
            );
            assert_eq!(pinned.generation(), &selected);
            assert_eq!(
                pinned
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                2
            );
        });
    }

    #[test]
    fn resumed_repair_reuses_the_verified_copy_without_rebuilding_sources() {
        use std::os::unix::fs::MetadataExt;

        on_runtime(|cx| async move {
            let _models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let original = protected_publication(&runtime, &cx, &root).await;
            damage_recovery_vector(&original, FSFS_VECTOR_INDEX_FILE);
            fs::rename(source, directory.path().join("source-unavailable")).unwrap();
            let damaged = file_bytes(original.path());
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let mut plan = runtime
                .prepare_retained_vector_repair(
                    &cx,
                    &root,
                    original.id(),
                    original.manifest_sha256(),
                    1 << 30,
                )
                .await
                .unwrap();
            plan.search(&cx, "sharedtoken", 10).await.unwrap();
            let token = plan.expected_selection_sha256(&cx).unwrap();
            // Retain a reader pin for the fixture, as a live operator session
            // may; a serialized receipt alone does not prevent retention.
            let target = plan.generation().clone();
            let before = file_bytes(target.path());
            let vector_path = target.path().join(FSFS_VECTOR_INDEX_FILE);
            let inode = fs::metadata(&vector_path).unwrap().ino();
            let count = fs::read_dir(root.join("generations")).unwrap().count();
            drop(plan);
            let mut resumed = fresh_runtime(&runtime)
                .resume_retained_recovery(&cx, &root, target.id(), target.manifest_sha256(), &token)
                .await
                .unwrap();
            assert_eq!(resumed.expected_selection_sha256(&cx).unwrap(), token);
            assert_eq!(resumed.generation(), &target);
            assert!(resumed.reader.resources.fast_embedder.is_some());
            assert!(resumed.reader.resources.quality_embedder.is_some());
            let page = resumed.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(page.last().unwrap().hits[0].path, "alpha.md");
            assert_eq!(fs::metadata(&vector_path).unwrap().ino(), inode);
            assert_eq!(file_bytes(target.path()), before);
            assert_eq!(
                fs::read_dir(root.join("generations")).unwrap().count(),
                count
            );
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
            let (publication, mut reader) = resumed.restore(&cx).unwrap();
            assert!(matches!(publication,
                GenerationPublication::Durable(ref value) if value == &target));
            assert_eq!(
                reader.search(&cx, "sharedtoken", 10).await.unwrap()[0]
                    .hits
                    .len(),
                1
            );
            assert_eq!(file_bytes(original.path()), damaged);
            assert!(
                fresh_runtime(&runtime)
                    .resume_retained_recovery(
                        &cx,
                        &root,
                        target.id(),
                        target.manifest_sha256(),
                        &token,
                    )
                    .await
                    .is_err()
            );
        });
    }

    #[test]
    fn runtime_resume_refuses_staleness_before_missing_target_or_changed_models() {
        on_runtime(|cx| async move {
            let models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let plan = runtime
                .prepare_retained_recovery(&cx, &root, old.id(), old.manifest_sha256())
                .await
                .unwrap();
            let token = plan.expected_selection_sha256(&cx).unwrap();
            drop(plan);
            fs::write(source.join("beta.md"), "sharedtoken later publication").unwrap();
            let winner = publish(&runtime, &cx, &root).await;
            let pointer = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            fs::rename(old.path(), directory.path().join("saved-old")).unwrap();
            models.quality.changed.store(true, Ordering::Relaxed);
            let error = fresh_runtime(&runtime)
                .resume_retained_recovery(&cx, &root, old.id(), old.manifest_sha256(), &token)
                .await
                .unwrap_err();
            models.quality.changed.store(false, Ordering::Relaxed);
            assert!(
                matches!(error, SearchError::InvalidConfig { ref reason, .. }
                if reason.contains("cannot rebase"))
            );
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                pointer
            );
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
    fn resume_receipt_export_preserves_preview_and_required_producer_barriers() {
        on_runtime(|cx| async move {
            let models = RecoveryModels::install();
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, source, root) = fixture(directory.path());
            let target = publish(&runtime, &cx, &root).await;
            runtime.config.search.fast_only = true;
            let mut plan = runtime
                .prepare_retained_recovery(&cx, &root, target.id(), target.manifest_sha256())
                .await
                .unwrap();
            let token = plan.expected_selection_sha256(&cx).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                plan.expected_selection_sha256(&cx),
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                plan.search(&cx, "sharedtoken", 10).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(plan.expected_selection_sha256(&cx).is_err());
            plan.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(plan.expected_selection_sha256(&cx).unwrap(), token);
            for producer in [&models.fast, &models.quality] {
                producer.changed.store(true, Ordering::Relaxed);
                assert!(plan.expected_selection_sha256(&cx).is_err());
                let result = fresh_runtime(&runtime)
                    .resume_retained_recovery(
                        &cx,
                        &root,
                        target.id(),
                        target.manifest_sha256(),
                        &token,
                    )
                    .await;
                producer.changed.store(false, Ordering::Relaxed);
                assert!(matches!(
                    result,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
            }
            let resumed = fresh_runtime(&runtime)
                .resume_retained_recovery(&cx, &root, target.id(), target.manifest_sha256(), &token)
                .await
                .unwrap();
            fs::write(source.join("beta.md"), "sharedtoken concurrent update").unwrap();
            let winner = publish(&runtime, &cx, &root).await;
            // Exporting either old plan still carries the old selection. Neither
            // successful re-admission nor export can erase the publication race.
            assert_eq!(plan.expected_selection_sha256(&cx).unwrap(), token);
            assert_eq!(resumed.expected_selection_sha256(&cx).unwrap(), token);
            assert!(resumed.restore(&cx).is_err());
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(winner)
            );
        });
    }
}
