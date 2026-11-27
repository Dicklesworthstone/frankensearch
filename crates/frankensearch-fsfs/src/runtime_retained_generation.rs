// Compose the existing runtime with retained-generation rebuilding in the same
// module, so its private admission and resource-opening paths remain the single
// source of truth. The existing runtime source is left byte-for-byte intact.
include!("runtime.rs");

#[derive(Clone, Copy, PartialEq, Eq)]
enum RetainedBuildKind {
    Index,
    Recovery,
}

enum RetainedRebuildStart<'a> {
    Fresh,
    Observed(crate::generation_store::RecoverySelection),
    Save(&'a mut (dyn FnMut(&RetainedRebuildCheckpoint) -> SearchResult<()> + Send)),
    Resume(&'a RetainedRebuildCheckpoint),
}

#[derive(Default)]
struct RetainedBuildProducers {
    // Keep the same loaded producers through admission and source indexing.
    fast: Option<Arc<dyn Embedder>>,
    quality: Option<IndexingQualityTier>,
    probes_deferred: bool,
    legacy_evidence: Option<retained_reuse::LegacyReuseReceipt>,
}

/// A reusable search reader bound to one admitted complete generation.
///
/// Publishing a successor does not retarget this reader. Open a new reader to
/// observe that successor; an in-flight query and all of its progressive phases
/// continue to use the original lexical, vector and catalog generation.
///
/// Searches use the ordinary ranking and producer-admission paths, but bypass
/// the CLI's persistent query cache and explanation-session writes: neither may
/// mutate the sealed bundle. Model initialization remains shared with the
/// caller's runtime and uses its existing blocking pool.
pub struct RetainedSearchReader {
    runtime: FsfsRuntime,
    generation: crate::generation_store::PublishedGeneration,
    resources: SearchExecutionResources,
}

impl std::fmt::Debug for RetainedSearchReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedSearchReader")
            .field("generation", &self.generation.id())
            .finish_non_exhaustive()
    }
}

impl RetainedSearchReader {
    /// The stable generation selected when this reader was opened.
    #[must_use]
    pub const fn generation(&self) -> &crate::generation_store::PublishedGeneration {
        &self.generation
    }

    /// Execute one query against the pinned generation without persistent writes.
    ///
    /// The returned payloads preserve the ordinary Initial/Refined/failure phase
    /// semantics. This is direct retrieval, not CLI daemon forwarding or query
    /// expansion. Filters and ranking options come from the opening runtime.
    ///
    /// # Errors
    /// Returns the original admission, retrieval, identity or cancellation error.
    pub async fn search(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        self.search_inner(cx, query, limit, None).await
    }

    /// Execute a query and deliver each phase as the existing pipeline emits it.
    ///
    /// The sink can stop delivery by returning an error. A failed sink or a
    /// cancelled future cannot publish an explanation file or a cached result.
    ///
    /// # Errors
    /// Returns retrieval/cancellation errors or the sink's original error.
    pub async fn search_with_phase_sink(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
        sink: &mut (dyn FnMut(&SearchPayload) -> SearchResult<()> + Send),
    ) -> SearchResult<Vec<SearchPayload>> {
        self.search_inner(cx, query, limit, Some(sink)).await
    }

    async fn search_inner(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
        sink: Option<SearchPhaseSink<'_>>,
    ) -> SearchResult<Vec<SearchPayload>> {
        retained_search_checkpoint(cx)?;
        let mode = self
            .runtime
            .search_mode_for_query(query, self.runtime.requested_search_mode())?;
        let artifacts = Box::pin(
            self.runtime
                .execute_search_phase_artifacts_with_mode_using_resources(
                    cx,
                    query,
                    limit,
                    mode,
                    &mut self.resources,
                    SearchExecutionFlags {
                        include_snippets: true,
                        persist_explain_session: false,
                    },
                    sink,
                ),
        )
        .await?;
        Ok(artifacts
            .into_iter()
            .map(|artifact| artifact.payload)
            .collect())
    }

    /// The mode a search on this reader runs in, as for a legacy CLI search:
    /// hybrid, narrowed by the opening runtime's `pressure.degradation_override`
    /// (which may also refuse search), and lexical-only in a build without
    /// semantic loaders, whose generations carry no vector tier.
    fn search_mode(&self) -> SearchResult<SearchExecutionMode> {
        degraded_search_mode(
            self.runtime.config.pressure.degradation_override,
            self.runtime.requested_search_mode(),
        )
    }
}

/// A long-lived reader that admits published successors between queries.
///
/// Each search checks the bounded selection descriptor, reusing the existing
/// lexical/vector/catalog resources when it has not changed. A replacement is
/// fully verified and opened before the old reader is swapped out. Every phase
/// of an admitted query stays on one generation even if publication happens
/// while its sink is running. This does not change legacy CLI/watch routing.
///
/// Invalid selection, failed admission and cancellation are returned to the
/// caller; they never trigger an implicit stale-result or legacy-index fallback.
/// The store directory is retained by identity: replacing its pathname does not
/// authorize this reader to follow an unrelated store or a new symlink.
#[derive(Debug)]
pub struct LiveRetainedSearchReader {
    store: crate::generation_store::CompleteGenerationStore,
    root_directory: Arc<fs::File>,
    reader: RetainedSearchReader,
}

fn validate_live_store_root(root: &Path, directory: &fs::File) -> SearchResult<()> {
    let replaced = || {
        complete_cli::complete_cli_error(
            "root",
            "the live reader's store directory was replaced or is no longer a directory; its original generation remains retained",
        )
    };
    let opened = directory.metadata()?;
    let current = fs::symlink_metadata(root).map_err(|error| {
        if matches!(
            error.kind(),
            std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
        ) {
            replaced()
        } else {
            SearchError::Io(error)
        }
    })?;
    #[cfg(unix)]
    let same_directory = {
        use std::os::unix::fs::MetadataExt;
        (opened.dev(), opened.ino()) == (current.dev(), current.ino())
    };
    // CompleteGenerationStore already refuses non-Unix platforms before I/O.
    // Never substitute a constant matching identity on an unsupported target.
    #[cfg(not(unix))]
    let same_directory = false;
    if !opened.is_dir() || !current.is_dir() || !same_directory {
        return Err(replaced());
    }
    Ok(())
}

impl LiveRetainedSearchReader {
    fn validate_store_root(&self) -> SearchResult<()> {
        validate_live_store_root(self.store.root(), &self.root_directory)
    }

    /// The currently loaded generation, not an assertion about a later pointer.
    #[must_use]
    pub const fn generation(&self) -> &crate::generation_store::PublishedGeneration {
        self.reader.generation()
    }

    /// Admit the selected generation if it differs from the loaded one.
    ///
    /// Returns true only when a different generation was installed. There is
    /// at most one replacement open per call: continuous publication cannot
    /// force an unbounded retry loop. A subsequent query observes later changes.
    /// Errors leave the original reader intact so admission can be retried
    /// after repair, or the caller can explicitly opt into `into_retained`.
    ///
    /// # Errors
    /// Returns the original store-identity, selection, integrity, producer or
    /// cancellation error.
    pub async fn refresh(&mut self, cx: &Cx) -> SearchResult<bool> {
        retained_search_checkpoint(cx)?;
        self.validate_store_root()?;
        let selected = self.store.is_selected(cx, self.reader.generation())?;
        self.validate_store_root()?;
        if selected {
            return Ok(false);
        }
        let replacement = self
            .reader
            .runtime
            .open_retained_search(cx, self.store.root())
            .await;
        retained_search_checkpoint(cx)?;
        // Admission may yield while the store pathname is replaced. Check
        // even when that admission failed, before accepting any new reader.
        self.validate_store_root()?;
        let replacement = replacement?;
        let changed = replacement.generation() != self.reader.generation();
        self.reader = replacement;
        Ok(changed)
    }

    /// Refresh once, then execute the whole query against that retained reader.
    ///
    /// # Errors
    /// Returns refresh, retrieval, producer-identity or cancellation errors.
    pub async fn search(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<SearchPayload>> {
        self.refresh(cx).await?;
        self.reader.search(cx, query, limit).await
    }

    /// Refresh before the first phase and never switch or retry after delivery.
    ///
    /// # Errors
    /// Returns refresh/retrieval/cancellation errors or the sink's original error.
    pub async fn search_with_phase_sink(
        &mut self,
        cx: &Cx,
        query: &str,
        limit: usize,
        sink: &mut (dyn FnMut(&SearchPayload) -> SearchResult<()> + Send),
    ) -> SearchResult<Vec<SearchPayload>> {
        self.refresh(cx).await?;
        self.reader
            .search_with_phase_sink(cx, query, limit, sink)
            .await
    }

    /// Explicitly stop following publication and keep the last admitted reader.
    #[must_use]
    pub fn into_retained(self) -> RetainedSearchReader {
        self.reader
    }
}

fn retained_search_checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|_| SearchError::Cancelled {
        phase: "fsfs.complete_generation.search".to_owned(),
        reason: "retained-generation search cancelled".to_owned(),
    })
}

// Validate path components, not a string prefix: `{index_dir}/../shared.db`
// and `{index_dir}-other/catalog.db` must never escape a retained generation.
// Readers need the same rule as builders, otherwise catalog hydration can come
// from a mutable database that is not authenticated by the selected inventory.
fn validate_retained_catalog_path(value: &str) -> SearchResult<()> {
    use std::path::Component;

    let relative = Path::new(value.trim())
        .strip_prefix(crate::config::STORAGE_DB_PATH_INDEX_DIR_PLACEHOLDER)
        .ok();
    if relative.is_some_and(|path| {
        path.file_name().is_some()
            && path
                .components()
                .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
    }) {
        return Ok(());
    }
    Err(SearchError::InvalidConfig {
        field: "storage.db_path".to_owned(),
        value: value.to_owned(),
        reason:
            "retained generations require a catalog beneath {index_dir}, without parent traversal"
                .to_owned(),
    })
}

const STORE_OVERLAP_REASON: &str = "source and generation-store trees must not overlap, except for a store beneath a hidden directory of the source";

/// Whether a complete-generation store at `store` overlaps the source tree at
/// `source` (both canonical). A store may not hold its source. It may sit
/// inside it only beneath a hidden directory, like the default in-project
/// `.frankensearch`: index discovery never enters hidden entries, and builds
/// and watches also exclude the store by path, so its files never become
/// sources and its publications never look like source changes.
fn store_overlaps_source(store: &Path, source: &Path) -> bool {
    if source.starts_with(store) {
        return true;
    }
    store.strip_prefix(source).is_ok_and(|inside| {
        !inside
            .components()
            .any(|component| component.as_os_str().as_encoded_bytes().starts_with(b"."))
    })
}

/// Canonicalize a store root that may not exist yet through its existing
/// parent, so an overlap check never has to create the store first.
fn canonical_store_root(root: &Path) -> SearchResult<PathBuf> {
    match fs::canonicalize(root) {
        Ok(root) => Ok(root),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let name = root.file_name().ok_or_else(|| {
                complete_cli::complete_cli_error(
                    "store_root",
                    "a store root needs a directory name",
                )
            })?;
            let parent = root
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            Ok(fs::canonicalize(parent)?.join(name))
        }
        Err(error) => Err(error.into()),
    }
}

/// Unlink Quill segments an unpublished candidate no longer references.
///
/// A candidate starts with every file of its predecessor, retired merge inputs
/// included, so every generation used to inherit all earlier dead segments
/// (bd-2op1d). No reader can resolve a candidate before its publication, so a
/// zero window is safe here and nowhere else. Call once every candidate writer
/// has closed, before resources are admitted and the bundle is sealed. A
/// candidate whose one Quill writer was `apply_candidate_lexical_mutations`
/// was already reclaimed by that session.
async fn reclaim_unpublished_lexical_garbage(cx: &Cx, root: &Path) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    let layout = FsfsRuntime::resolve_lexical_engine(root)?;
    let (Some(BlueGreenEngine::Quill), Some(engine)) = (layout.engine(), layout.engine_dir())
    else {
        return Ok(());
    };
    let index = Box::pin(QuillIndex::open(cx, &engine, QuillConfig::default())).await?;
    let report = Box::pin(index.collect_garbage_with(
        cx,
        frankensearch_quill::GarbageCollectionOptions {
            grace_period: Duration::ZERO,
        },
    ))
    .await?;
    drop(index);
    tracing::debug!(
        engine = %engine.display(),
        removed = report.removed.len(),
        "reclaimed unreferenced lexical files before sealing a candidate"
    );
    Ok(())
}

impl FsfsRuntime {
    /// Maintenance preserves the topology recorded by the selected bundle.
    /// An absent fast file never grants quality-only or lexical-only intent:
    /// the durable sentinel must explicitly declare that topology, and its
    /// actual artifacts must agree before a candidate or producer is admitted.
    fn retained_generation_search_mode(index_root: &Path) -> SearchResult<SearchExecutionMode> {
        let sentinel = Self::read_index_sentinel(index_root)?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "topology",
                "selected generation has no completion sentinel",
            )
        })?;
        let declared = |reason| sentinel.reason_codes.iter().any(|code| code == reason);
        let quality_only = declared(REASON_VECTOR_FAST_TIER_SKIPPED_QUALITY_ONLY);
        let lexical_only = declared(REASON_VECTOR_FAST_TIER_SKIPPED_LEXICAL_ONLY);
        let fast_present =
            complete_cli::complete_entry_exists(&index_root.join(FSFS_VECTOR_INDEX_FILE))?;
        let quality_present =
            complete_cli::complete_entry_exists(&index_root.join(FSFS_VECTOR_QUALITY_INDEX_FILE))?;
        let quality_built = declared(REASON_VECTOR_QUALITY_TIER_BUILT);
        if sentinel.schema_version != 1
            || quality_built
                && (!quality_present
                    || sentinel.reason_codes.iter().any(|code| {
                        code.starts_with("vector.quality_tier.")
                            && code != REASON_VECTOR_QUALITY_TIER_BUILT
                    }))
        {
            return Err(complete_cli::complete_cli_error(
                "topology",
                "selected generation has an unsupported sentinel or contradictory quality-tier evidence",
            ));
        }
        let mode = if quality_only {
            if lexical_only
                || fast_present
                || !quality_present
                || !quality_built
                || sentinel.fast_window_max_per_file != 1
            {
                return Err(complete_cli::complete_cli_error(
                    "topology",
                    "quality-only generation intent disagrees with its retained artifacts or input policy",
                ));
            }
            SearchExecutionMode::QualityOnly
        } else if lexical_only {
            if fast_present || quality_present {
                return Err(complete_cli::complete_cli_error(
                    "topology",
                    "lexical-only generation intent disagrees with its retained artifacts",
                ));
            }
            SearchExecutionMode::LexicalOnly
        } else {
            if !fast_present {
                return Err(complete_cli::complete_cli_error(
                    "topology",
                    "selected generation requires its retained fast tier; a missing artifact cannot change retrieval topology",
                ));
            }
            if quality_present {
                SearchExecutionMode::Full
            } else {
                SearchExecutionMode::FastOnly
            }
        };
        Self::validate_search_generation_at_root(index_root, mode)?;
        Ok(mode)
    }

    /// Open a reusable reader that follows complete-generation publication.
    ///
    /// Model state and caller-owned blocking capacity are shared exactly as for
    /// `open_retained_search`. Each query refreshes before admission, while all
    /// of its phases retain one immutable resource set. No daemon, background
    /// worker, persistent result cache or explanation file is created.
    ///
    /// # Errors
    /// Returns the same configuration, selection and admission errors as
    /// `open_retained_search`. Missing selection never falls back to legacy data.
    pub async fn open_live_retained_search(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<LiveRetainedSearchReader> {
        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let store = crate::generation_store::CompleteGenerationStore::open_with_profile(
            cx,
            store_root,
            frankensearch_core::GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
        )?;
        let mut options = fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
        }
        let root_directory = Arc::new(options.open(store.root())?);
        validate_live_store_root(store.root(), &root_directory)?;
        let reader = self.open_retained_search(cx, store.root()).await;
        retained_search_checkpoint(cx)?;
        validate_live_store_root(store.root(), &root_directory)?;
        Ok(LiveRetainedSearchReader {
            store,
            root_directory,
            reader: reader?,
        })
    }

    /// Resolve and open the selected complete generation once for repeated reads.
    ///
    /// Bundle hashes are verified before the ordinary search admission and
    /// resource opens, in the mode the runtime's searches run in (lexical-only
    /// opens no vector tier). Missing or corrupt selection is an error, never an excuse
    /// to fall back to a legacy index or an arbitrary generation directory.
    /// The source store follows the cooperative immutable-directory contract of
    /// `CompleteGenerationStore`; callers must not modify its sealed files.
    /// The catalog must also resolve beneath `{index_dir}`, so hydration cannot
    /// consult an unrelated mutable database outside the selected generation.
    ///
    /// # Errors
    /// Returns missing/corrupt selection, producer mismatch, cancellation or the
    /// ordinary search-resource admission error. The store is not modified.
    pub async fn open_retained_search(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<RetainedSearchReader> {
        self.open_retained_search_selection(cx, store_root, None)
            .await
    }

    /// Open the exact retained generation named by a trusted publication receipt.
    ///
    /// The ID and inventory digest must come from the same receipt. This admits
    /// its entire bundle and serving resources while retaining its reader pin.
    /// Selection is never consulted or changed, so an intact historical bundle
    /// remains searchable even when the active descriptor is damaged.
    ///
    /// # Errors
    /// Returns invalid receipt, missing or changed bundle, producer mismatch,
    /// configuration, or cancellation errors before any query is executed.
    pub async fn open_historical_search(
        &self,
        cx: &Cx,
        store_root: &Path,
        generation_id: &str,
        manifest_sha256: &str,
    ) -> SearchResult<RetainedSearchReader> {
        self.open_retained_search_selection(cx, store_root, Some((generation_id, manifest_sha256)))
            .await
    }

    async fn open_retained_search_selection(
        &self,
        cx: &Cx,
        store_root: &Path,
        selection: Option<(&str, &str)>,
    ) -> SearchResult<RetainedSearchReader> {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        // Shadow observation persists artifacts beneath the opened index.
        // A retained reader must never create them inside its sealed bundle.
        if self.config.search.shadow_mode {
            return Err(SearchError::InvalidConfig {
                field: "search.shadow_mode".to_owned(),
                value: "true".to_owned(),
                reason: "shadow mode writes observation artifacts and cannot run against a sealed complete generation"
                    .to_owned(),
            });
        }
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let store = CompleteGenerationStore::open_with_profile(
            cx,
            store_root,
            frankensearch_core::GenerationRootSecurityProfileV1::ReadOnlyUnanchored,
        )?;
        let generation = match selection {
            Some((id, digest)) => store.open_retained(cx, id, digest)?,
            None => store
                .active(cx)?
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "complete_generation.selection".to_owned(),
                    value: store.root().display().to_string(),
                    reason: "no complete generation has been published; rebuild the store first"
                        .to_owned(),
                })?,
        };
        let mut runtime = self.clone();
        runtime.cli_input.index_dir = Some(generation.path().to_path_buf());
        // Retained readers never start a daemon or write a socket/cache under
        // an immutable generation, even when their caller originated in a CLI.
        runtime.cli_input.daemon = false;
        // Open what this reader's searches read. A lexical-only search (a build
        // without semantic loaders, or `force_lexical_only`) neither maps nor
        // admits the vector tiers, as on the legacy layout. An override that
        // refuses search opens the least; each search then reports the refusal.
        let mode = if matches!(
            self.cli_input.command,
            CliCommand::Serve | CliCommand::Daemon
        ) {
            // A daemon has not received a query yet. Like legacy serve,
            // admit lexical evidence first; each semantic request rebinds
            // its required tiers before caching, attestation or retrieval.
            SearchExecutionMode::LexicalOnly
        } else {
            self.search_mode_for_query(
                self.cli_input.query.as_deref().unwrap_or_default(),
                self.requested_search_mode(),
            )
            .unwrap_or(SearchExecutionMode::LexicalOnly)
        };
        let resources = Box::pin(
            runtime.prepare_search_execution_resources_at_root_with_modes(
                cx,
                generation.path(),
                mode,
                if matches!(
                    self.cli_input.command,
                    CliCommand::Serve | CliCommand::Daemon
                ) {
                    // Match legacy serving: Quill-present generations narrow
                    // themselves to lexical, while standalone vector stores
                    // must open their available tiers before the first query.
                    SearchExecutionMode::Full
                } else {
                    mode
                },
            ),
        )
        .await?;
        retained_search_checkpoint(cx)?;
        Ok(RetainedSearchReader {
            runtime,
            generation,
            resources,
        })
    }

    /// Rebuild into an isolated complete-generation store and activate only
    /// after the ordinary full-search admission succeeds.
    ///
    /// This explicit library entry point does not change legacy CLI/watch root
    /// discovery. Open a `RetainedSearchReader` to serve a selected generation;
    /// existing readers and directories survive subsequent publications.
    /// The store must be outside the source tree, and the catalog must use the
    /// `{index_dir}` layout so no old-generation database is modified.
    /// Successors built by the same executable and configuration can reuse
    /// completed input evidence in an independent copy, including after process
    /// restart on Linux. Other platforms retain process-scoped reuse.
    /// `full_reindex` disables reuse; unproven inputs are recomputed through the
    /// ordinary indexer and its producer checks.
    ///
    /// # Errors
    /// Returns the original indexing/admission error without changing the
    /// active pointer. Inspect the publication outcome for a post-rename sync
    /// failure; it is visible but not confirmed durable, not an aborted build.
    #[allow(clippy::future_not_send)]
    pub async fn rebuild_retained_generation(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<crate::generation_store::GenerationPublication> {
        self.rebuild_retained_generation_with_precommit(cx, store_root, |_| Ok(()))
            .await
            .map(|(publication, _)| publication)
    }

    /// [`Self::rebuild_retained_generation`] plus the build summary the legacy
    /// `fsfs index` receipt reports (counts, index size, embedding outcomes).
    #[allow(clippy::future_not_send)]
    async fn rebuild_retained_generation_with_summary(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<(
        crate::generation_store::GenerationPublication,
        FsfsIndexPayload,
    )> {
        self.rebuild_retained_generation_with_precommit(cx, store_root, |_| Ok(()))
            .await
    }

    // Source authority is checked after sealing, without replacing the real
    // indexing, producer admission, lease or predecessor checks below.
    #[allow(clippy::future_not_send)]
    async fn rebuild_retained_generation_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        precommit: F,
    ) -> SearchResult<(
        crate::generation_store::GenerationPublication,
        FsfsIndexPayload,
    )>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        let (build, summary, candidate_mode) = self
            .build_retained_generation(cx, store_root, RetainedBuildKind::Index)
            .await?;
        let publication = build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, candidate_mode),
            precommit,
        )?;
        Ok((publication, summary))
    }

    // Share the real source indexer, reuse policy and admission between direct
    // indexing and explicit build/preview/publish recovery. The build keeps its
    // original predecessor and lease through every async indexing boundary.
    #[allow(clippy::future_not_send)]
    async fn build_retained_generation(
        &self,
        cx: &Cx,
        store_root: &Path,
        kind: RetainedBuildKind,
    ) -> SearchResult<(
        crate::generation_store::GenerationBuild,
        FsfsIndexPayload,
        SearchExecutionMode,
    )> {
        self.build_retained_generation_with_progress(
            cx,
            store_root,
            kind,
            RetainedRebuildStart::Fresh,
            |_| Ok(()),
        )
        .await
    }

    async fn admit_retained_build_producers(
        &self,
        cx: &Cx,
        root: &Path,
    ) -> SearchResult<RetainedBuildProducers> {
        use crate::generation_store::COMPLETE_GENERATION_POINTER;

        if matches!(
            self.indexing_search_mode(),
            SearchExecutionMode::QualityOnly
        ) {
            self.validate_semantic_model_selection(SearchExecutionMode::QualityOnly)?;
            if !complete_cli::complete_entry_exists(&root.join(COMPLETE_GENERATION_POINTER))?
                && !complete_cli::complete_entry_exists(&root.join("generations"))?
            {
                self.validate_index_source_root(root, &self.resolve_target_root()?)?;
            }
            let mut control = IndexRunControl::new(&self.config)?;
            if matches!(
                self.cli_input.command,
                CliCommand::Index | CliCommand::Watch
            ) && self.cli_input.quality_only
                && self.cli_input.command_source == crate::adapters::cli::CommandSource::Explicit
            {
                // The required primary must be provisioned before store routing
                // changes. Explicit recovery keeps its separate acquisition
                // authorization and never reaches this CLI-only offer.
                control.admit_work(cx, "index.model_provisioning").await?;
                self.maybe_offer_interactive_model_provisioning(cx, true)
                    .await?;
            }
            control.admit_work(cx, "index.quality_model_load").await?;
            // Quality is required for this topology. Prove it before creating
            // `generations`, which would otherwise redirect a legacy reader
            // even if an unavailable or invalid model then aborts the build.
            #[cfg(test)]
            let resolved = self.resolve_quality_embedder();
            #[cfg(not(test))]
            let resolved = if cx.blocking_pool_handle().is_some() {
                let runtime = self.clone();
                Self::search_blocking(cx, move || runtime.resolve_quality_embedder()).await?
            } else {
                self.resolve_quality_embedder()
            };
            let embedder = resolved?.ok_or_else(|| SearchError::EmbedderUnavailable {
                model: self.config.indexing.quality_model.clone(),
                reason: "quality-only indexing requires its verified quality producer".to_owned(),
            })?;
            control.checkpoint(cx, "index.quality_model_loaded", true)?;
            Self::ensure_semantic_embedder_admissible(embedder.as_ref(), false)?;
            Self::probe_indexing_embedder(cx, embedder.as_ref()).await?;
            retained_search_checkpoint(cx)?;
            return Ok(RetainedBuildProducers {
                fast: None,
                quality: Some((Some(embedder), REASON_VECTOR_QUALITY_TIER_BUILT)),
                ..RetainedBuildProducers::default()
            });
        }
        if !matches!(
            build_search_mode(self.indexing_search_mode()),
            SearchExecutionMode::Full | SearchExecutionMode::FastOnly
        ) || matches!(
            self.config.pressure.degradation_override,
            DegradationOverrideMode::ForceLexicalOnly
                | DegradationOverrideMode::ForceMetadataOnly
                | DegradationOverrideMode::ForcePaused
        ) || complete_cli::complete_entry_exists(&root.join(COMPLETE_GENERATION_POINTER))?
            || (complete_cli::complete_entry_exists(&root.join("generations"))?
                && !crate::generation_store::CompleteGenerationStore::legacy_migration_pending(root)?)
            || !Self::read_index_sentinel(root)?
                .is_some_and(|sentinel| sentinel.generation_complete)
        {
            return Ok(RetainedBuildProducers::default());
        }
        self.validate_index_source_root(root, &self.resolve_target_root()?)?;
        self.validate_semantic_model_selection(self.indexing_search_mode())?;
        let mut control = IndexRunControl::new(&self.config)?;
        control.admit_work(cx, "index.model_provisioning").await?;
        // The candidate receives the loaded fast producer and skips its own
        // provisioning step, so offer optional models here as well.
        self.maybe_offer_interactive_model_provisioning(cx, false)
            .await?;
        control.admit_work(cx, "index.fast_model_load").await?;
        let resolved = if cx.blocking_pool_handle().is_some() {
            let runtime = self.clone();
            #[cfg(test)]
            let selected_test_provider = runtime.resolve_fast_embedder();
            Self::search_blocking(cx, move || {
                #[cfg(not(test))]
                {
                    runtime.resolve_fast_embedder()
                }
                #[cfg(test)]
                {
                    selected_test_provider
                }
            })
            .await?
        } else {
            self.resolve_fast_embedder()
        };
        let embedder = resolved?;
        control.checkpoint(cx, "index.fast_model_loaded", true)?;
        Self::ensure_semantic_embedder_admissible(embedder.as_ref(), cfg!(test))?;
        let probes_deferred = self.contract_upgrade_reuse_enabled() && !self.cli_input.full_reindex;
        if !probes_deferred {
            Self::probe_indexing_embedder(cx, embedder.as_ref()).await?;
        }
        let quality = {
            control.admit_work(cx, "index.quality_model_load").await?;
            // Unit tests inject providers on the caller's thread.
            #[cfg(test)]
            let tier = self.resolve_indexing_quality_embedder()?;
            #[cfg(not(test))]
            let tier = if cx.blocking_pool_handle().is_some() {
                let runtime = self.clone();
                Self::search_blocking(cx, move || runtime.resolve_indexing_quality_embedder())
                    .await??
            } else {
                self.resolve_indexing_quality_embedder()?
            };
            control.checkpoint(cx, "index.quality_model_loaded", true)?;
            if !probes_deferred && let Some(quality_embedder) = tier.0.as_ref() {
                Self::probe_indexing_embedder(cx, quality_embedder.as_ref()).await?;
            }
            tier
        };
        retained_search_checkpoint(cx)?;
        Ok(RetainedBuildProducers {
            fast: Some(embedder),
            quality: Some(quality),
            probes_deferred,
            legacy_evidence: None,
        })
    }

    fn validate_first_migration_discovery(
        &self,
        cx: &Cx,
        root: &Path,
        target_root: &Path,
        source_policy: IndexSourcePolicy,
        on_candidate: impl FnMut(IndexCandidate),
    ) -> SearchResult<()> {
        // Creating `generations` takes over routing from the legacy index. Run
        // normal discovery under the root's writer lease before that boundary.
        // The indexer takes a fresh observation after seeding; the lease does
        // not make the source tree immutable.
        let source = IndexSourceRoot::open(target_root)?;
        let mut control = IndexRunControl::new(&self.config)?;
        control.checkpoint(cx, "index.discovery", true)?;
        self.discover_index_candidates(cx, &mut control, target_root, root, source_policy, on_candidate)?;
        source.check()?;
        Ok(())
    }

    async fn complete_legacy_probe_admission(
        &self,
        cx: &Cx,
        root: &Path,
        candidates: &[IndexCandidate],
        source_policy: IndexSourcePolicy,
        has_predecessor: bool,
        producers: &mut RetainedBuildProducers,
    ) -> SearchResult<()> {
        if !producers.probes_deferred {
            return Ok(());
        }
        let fast = producers.fast.as_ref().expect("deferred fast producer");
        let quality = producers.quality.as_ref().and_then(|tier| tier.0.as_ref());
        if !has_predecessor && let Some(quality) = quality {
            retained_reuse::prepare_legacy_reuse(cx).await?;
            if let Some(evidence) = retained_reuse::admitted_legacy_evidence(cx, self, root)? {
                let fast_admission = self.admit_workload_embedder(cx, Arc::clone(fast), WorkloadClass::Embed)?;
                let quality_admission = self.admit_workload_embedder(cx, Arc::clone(quality), WorkloadClass::Embed)?;
                let proven = retained_reuse::prove_legacy_seed_without_inference(
                    cx, root, candidates, source_policy, &evidence, (&fast_admission, &quality_admission),
                ).await?;
                producers.legacy_evidence = Some(evidence);
                if proven {
                    return Ok(());
                }
            }
        }
        Self::probe_indexing_embedder(cx, fast.as_ref()).await?;
        if let Some(quality) = quality {
            Self::probe_indexing_embedder(cx, quality.as_ref()).await?;
        }
        producers.probes_deferred = false;
        Ok(())
    }

    #[allow(clippy::future_not_send)]
    async fn build_retained_generation_with_progress<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        kind: RetainedBuildKind,
        start: RetainedRebuildStart<'_>,
        on_progress: F,
    ) -> SearchResult<(
        crate::generation_store::GenerationBuild,
        FsfsIndexPayload,
        SearchExecutionMode,
    )>
    where
        F: FnMut(&IndexingProgressSnapshot) -> SearchResult<()>,
    {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        let target_root = fs::canonicalize(self.resolve_target_root()?)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let canonical_root = canonical_store_root(store_root)?;
        if store_overlaps_source(&canonical_root, &target_root) {
            return Err(SearchError::InvalidConfig {
                field: "complete_generation.store_root".to_owned(),
                value: canonical_root.display().to_string(),
                reason: STORE_OVERLAP_REASON.to_owned(),
            });
        }
        if !matches!(start, RetainedRebuildStart::Fresh) && kind != RetainedBuildKind::Recovery {
            return Err(retained_recovery_error(
                "inactive build receipts require an independent source recovery rebuild",
            ));
        }
        self.admit_storage_write(cx, &canonical_root, "retained_build.start")?;
        if let RetainedRebuildStart::Resume(receipt) = &start {
            receipt.admit_scope(cx, self).await?;
        }
        // Creating `generations` makes complete-store routing authoritative.
        // Refuse incompatible installed producers before taking over routing
        // from a searchable legacy index. Reuse the ready producers and check
        // their identities against any rows the indexer carries.
        let mut producers =
            if kind == RetainedBuildKind::Index && matches!(start, RetainedRebuildStart::Fresh) {
                self.admit_retained_build_producers(cx, &canonical_root)
                    .await?
            } else {
                RetainedBuildProducers::default()
            };
        // Model admission can await for a long time. Recheck the whole store
        // before creating or resuming candidate state, never just its new path.
        self.admit_storage_write(cx, &canonical_root, "retained_build.allocate")?;
        let resumed = matches!(start, RetainedRebuildStart::Resume(_));
        let store = if resumed {
            CompleteGenerationStore::open(cx, store_root)?
        } else {
            CompleteGenerationStore::create(cx, store_root)?
        };
        let build = match &start {
            RetainedRebuildStart::Resume(receipt) => receipt.resume(cx, &store)?,
            RetainedRebuildStart::Observed(selection) => {
                store.begin_recovery_rebuild_after(cx, Some(selection))?
            }
            _ => match kind {
                RetainedBuildKind::Index => {
                    let admission = store.admit_build(cx)?;
                    let predecessor = admission.predecessor();
                    let root = store.root();
                    self.validate_index_source_root(
                        predecessor.map_or(root, |generation| generation.path()),
                        &target_root,
                    )?;
                    let source_policy = IndexSourcePolicy::for_index_root(root)?;
                    let mut candidates = Vec::new();
                    if predecessor.is_none()
                        && source_policy == IndexSourcePolicy::RequireReadable
                        && (producers.probes_deferred
                            || !complete_cli::complete_entry_exists(&root.join("generations"))?)
                    {
                        self.validate_first_migration_discovery(cx, root, &target_root, source_policy, |candidate| {
                            if producers.probes_deferred {
                                candidates.push(candidate);
                            }
                        })?;
                    }
                    Box::pin(self.complete_legacy_probe_admission(
                        cx, root, &candidates, source_policy, predecessor.is_some(), &mut producers,
                    )).await?;
                    let protect_legacy = predecessor.is_none()
                        && Self::read_index_sentinel(root)
                            .ok()
                            .flatten()
                            .is_some_and(|sentinel| sentinel.generation_complete);
                    self.admit_storage_write(cx, root, "retained_build.allocate")?;
                    admission.allocate(cx, protect_legacy)?
                }
                RetainedBuildKind::Recovery => store.begin_recovery_rebuild(cx)?,
            },
        };
        let mut input = self.cli_input.clone();
        input.command = CliCommand::Index;
        input.target_path = Some(target_root);
        input.index_dir = Some(build.path().to_path_buf());
        input.watch = false;
        input.quiet = true;
        // Preserve caller-owned native capacity, bundled-model materialization
        // and already initialized model slots rather than constructing an
        // unrelated runtime with no blocking pool.
        let mut candidate = self.clone().with_cli_input(input);
        candidate.config.indexing.watch_mode = false;
        if kind == RetainedBuildKind::Recovery
            || build.has_predecessor()
            || IndexSourcePolicy::for_index_root(store.root())?
                == IndexSourcePolicy::RequireReadable
        {
            candidate.source_policy = IndexSourcePolicy::RequireReadable;
        }
        if kind == RetainedBuildKind::Recovery {
            // A fresh recovery never inherits the selected generation. Only
            // an explicitly scoped resume may consult its own inactive
            // checkpoint; that checkpoint grants no completed-receipt carry.
            candidate.cli_input.full_reindex = !resumed;
            candidate.unchanged_sources = None;
            candidate.receipt_seed = None;
        }
        candidate.discovery_excluded_root = Some(store.root().to_path_buf());
        if let RetainedRebuildStart::Save(save) = start {
            let receipt = RetainedRebuildCheckpoint::capture(cx, &candidate, &build).await?;
            save(&receipt)?;
            retained_search_checkpoint(cx)?;
        }
        let summary = Box::pin(candidate.run_retained_index_with_progress(
            cx,
            &store,
            build.path(),
            resumed,
            producers,
            on_progress,
        ))
        .await?;
        reclaim_unpublished_lexical_garbage(cx, build.path()).await?;

        // An indexing command can return success with deferred semantic rows.
        // Apply the same full-generation admission used by actual search, then
        // open its real lexical/vector/producer resources before sealing.
        let candidate_mode = candidate.indexing_search_mode();
        Self::validate_search_generation_at_root(build.path(), candidate_mode)?;
        if candidate.config.pressure.degradation_override
            == crate::config::DegradationOverrideMode::ForceMetadataOnly
        {
            // This private runtime has already finished indexing and recorded
            // the original metadata-only policy. Validate its path-only output
            // as lexical data; no operator search is being admitted here.
            // The caller's runtime still refuses metadata-only searches.
            candidate.config.pressure.degradation_override =
                crate::config::DegradationOverrideMode::ForceLexicalOnly;
        }
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                candidate_mode,
                candidate_mode,
            ),
        )
        .await?;
        drop(resources);
        Ok((build, summary, candidate_mode))
    }

    /// Insert or replace the supplied JSONL documents in a complete successor.
    ///
    /// Duplicate IDs use their last input body and count once. Every present
    /// vector tier must admit its real producer before embedding; missing or
    /// incompatible quality support refuses the whole batch. Bounded canonical
    /// input drives vectors and catalog identity; complete lexical text remains
    /// searchable, including content beyond the models' input budget. A
    /// lexical-only generation has no vector tier and gains keyword rows only.
    /// Bounded native inference groups share workload and pressure admission
    /// with mixed updates; both tiers share a 128 MiB staged-vector limit.
    /// Untouched source rows keep their authenticated reuse evidence, while
    /// every supplied body requires fresh source validation before later reuse.
    /// No source directory is scanned or changed. Empty input publishes nothing.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn append_retained_generation(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<(
        Option<crate::generation_store::GenerationPublication>,
        usize,
    )> {
        self.append_retained_generation_with_precommit(cx, store_root, |_| Ok(()))
            .await
    }

    #[allow(clippy::future_not_send)]
    async fn append_retained_generation_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        precommit: F,
    ) -> SearchResult<(
        Option<crate::generation_store::GenerationPublication>,
        usize,
    )>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        self.validate_content_upserts_allowed()?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        if self.quality_work_deferred() {
            // A supplied body must not cause quality inference or a partial
            // tier update. Refuse before reading stdin/JSONL or owning a build;
            // validate_append_indexing_mode rechecks the selected base later.
            let store = CompleteGenerationStore::open(cx, store_root)?;
            if let Some(predecessor) = store.active(cx)? {
                self.validate_deferred_quality_mutation(predecessor.path())?;
            }
        }
        let documents = retained_reuse::read_append_documents(cx, self).await?;
        if documents.is_empty() {
            return Ok((None, 0));
        }
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let predecessor = store.active(cx)?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "selection",
                "no complete generation has been published",
            )
        })?;
        let result = self
            .apply_retained_append_with_precommit(
                cx,
                store_root,
                &predecessor,
                documents,
                precommit,
            )
            .await?;
        Ok((result.publication, result.upserted))
    }

    /// Delete exact IDs or prefixes from an isolated complete successor.
    ///
    /// The selected bundle stays immutable and searchable for the entire
    /// operation. Both vector tiers, lexical membership, the local storage
    /// catalog and paired manifests change before publication. No source file
    /// is removed; a later source rebuild can index it again.
    ///
    /// A missing match returns no publication. Otherwise the caller must inspect
    /// the publication outcome before reporting confirmed durability.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn delete_retained_generation(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<(
        Option<crate::generation_store::GenerationPublication>,
        usize,
    )> {
        self.delete_retained_generation_with_precommit(cx, store_root, |_| Ok(()))
            .await
    }

    #[allow(clippy::future_not_send, clippy::too_many_lines)]
    async fn delete_retained_generation_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        precommit: F,
    ) -> SearchResult<(
        Option<crate::generation_store::GenerationPublication>,
        usize,
    )>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        if self.cli_input.delete_ids.is_empty() {
            return Err(complete_cli::complete_cli_error(
                "delete_ids",
                "provide at least one document ID or prefix to delete",
            ));
        }
        if self.cli_input.delete_ids.iter().any(|id| id.contains('\0')) {
            return Err(complete_cli::complete_cli_error(
                "delete_ids",
                "document IDs and prefixes cannot contain the reserved semantic-window NUL byte",
            ));
        }
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let admission = store.admit_build(cx)?;
        let predecessor = admission.predecessor().cloned().ok_or_else(|| {
            complete_cli::complete_cli_error(
                "selection",
                "no complete generation has been published",
            )
        })?;
        let sentinel = Self::read_index_sentinel(predecessor.path())?;
        let candidate_mode = Self::retained_generation_search_mode(predecessor.path())?;
        let mut manifests = Self::read_matching_manifest_generation(predecessor.path())?
            .ok_or_else(|| {
                complete_cli::complete_cli_error(
                    "delete_membership",
                    "selected membership manifests disagree",
                )
            })?;
        // Manifests can retain tombstoned entries. Resolve against the actual
        // live source union, including lexical-only and partial quality rows.
        let fast_path = predecessor.path().join(FSFS_VECTOR_INDEX_FILE);
        let fast_rows = if fast_path.exists() {
            let index = VectorIndex::open_read_only(&fast_path)?;
            semantic_windows::load_mapping(predecessor.path(), &index)?;
            index.live_doc_ids()?
        } else {
            HashSet::new()
        };
        let live_ids = Self::live_delete_sources_at_root(
            cx,
            predecessor.path(),
            &fast_rows,
            &self.cli_input.delete_ids,
            sentinel.as_ref().map(|value| Path::new(&value.target_root)),
            self.cli_input.delete_prefix,
        )?;
        let targets = select_delete_targets_for_index(
            &self.cli_input.delete_ids,
            sentinel.as_ref().map(|value| Path::new(&value.target_root)),
            self.cli_input.delete_prefix,
            &live_ids,
        )?;
        if targets.is_empty() {
            return Ok((None, 0));
        }
        retained_search_checkpoint(cx)?;
        self.admit_storage_write(cx, store_root, "retained_delete.start")?;
        let build = admission.allocate(cx, false)?;
        let mut input = self.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        input.daemon = false;
        input.daemon_socket = None;
        let candidate = self.clone().with_cli_input(input);
        if retained_reuse::copy_selected_generation(cx, &candidate, &store, build.path())?
            != predecessor
        {
            return Err(complete_cli::complete_cli_error(
                "delete_selection",
                "selected predecessor changed while preparing deletion",
            ));
        }
        let candidate_lease = crate::lifecycle::PublicationLease::acquire(build.path())?;
        candidate_lease.fence("complete-generation delete lexical mutation")?;
        let lexical_mutations = targets
            .iter()
            .map(|id| {
                LexicalMutation::delete(id.clone(), 0, IngestionClass::Skip, "delete_command")
            })
            .collect::<Vec<_>>();
        let lexical_reclaimed = candidate
            .apply_candidate_lexical_mutations(cx, build.path(), &lexical_mutations)
            .await?;
        let target_sources = targets.iter().map(String::as_str).collect::<HashSet<_>>();
        for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
            retained_search_checkpoint(cx)?;
            candidate_lease.fence("complete-generation delete vector mutation")?;
            let path = build.path().join(relative);
            if path.exists() {
                let mut index = Self::open_vector_index_for_mutation(&path)?;
                let rows = index
                    .live_doc_ids()?
                    .into_iter()
                    .filter(|row| target_sources.contains(semantic_windows::source_id(row)))
                    .collect::<Vec<_>>();
                index.soft_delete_batch(&rows.iter().map(String::as_str).collect::<Vec<_>>())?;
                // Freeze the candidate without a pending WAL. These ordinary
                // rewrites also invalidate old repair symbols before fresh
                // protection, so repair cannot resurrect a deleted document.
                index.compact()?;
                index.vacuum()?;
            }
        }
        retained_search_checkpoint(cx)?;
        candidate_lease.fence("complete-generation delete metadata mutation")?;
        let catalog_path = candidate.resolve_storage_db_path()?;
        if catalog_path.exists() {
            let storage = Storage::open(PipelineStorageConfig {
                db_path: catalog_path,
                ..PipelineStorageConfig::default()
            })?;
            for id in &targets {
                retained_search_checkpoint(cx)?;
                storage.delete_document(id)?;
            }
        }
        for id in &targets {
            manifests.remove(id);
        }
        let manifests = manifests.into_values().collect::<Vec<_>>();
        let layout = Self::resolve_lexical_engine(build.path())?;
        let lexical_manifest_path = if layout.lexical_root() == build.path() {
            layout
                .engine_dir()
                .map(|path| path.join(FSFS_INDEX_MANIFEST_FILE_NAME))
                .unwrap_or_else(|| build.path().join(FSFS_LEXICAL_MANIFEST_FILE))
        } else {
            build.path().join(FSFS_LEXICAL_MANIFEST_FILE)
        };
        candidate.write_index_artifacts(build.path(), &lexical_manifest_path, &manifests)?;
        let mut sentinel = Self::read_index_sentinel(build.path())?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "delete_membership",
                "candidate has no completion sentinel",
            )
        })?;
        "delete".clone_into(&mut sentinel.command);
        sentinel.generated_at_ms = pressure_timestamp_ms();
        sentinel.indexed_files = manifests.len();
        sentinel.skipped_files = sentinel.discovered_files.saturating_sub(manifests.len());
        sentinel.total_canonical_bytes = manifests.iter().fold(0_u64, |total, entry| {
            total.saturating_add(entry.canonical_bytes)
        });
        sentinel.source_hash_hex = index_source_hash_hex(&manifests);
        for reason in protect_vector_generations(build.path(), "complete-generation delete") {
            if !sentinel.reason_codes.contains(&reason) {
                sentinel.reason_codes.push(reason);
            }
        }
        candidate.write_index_sentinel(build.path(), &sentinel)?;
        candidate_lease.fence("complete-generation delete candidate complete")?;
        // Drop clears the lease's owner record. It must precede sealing so no
        // destructor writes through the completed bundle's inventory.
        drop(candidate_lease);
        if !lexical_reclaimed {
            reclaim_unpublished_lexical_garbage(cx, build.path()).await?;
        }
        if Self::retained_generation_search_mode(build.path())? != candidate_mode {
            return Err(complete_cli::complete_cli_error(
                "delete_topology",
                "deletion changed the selected generation's retrieval topology",
            ));
        }
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                candidate_mode,
                candidate_mode,
            ),
        )
        .await?;
        drop(resources);
        let checkpoint =
            retained_batch::retain_unchanged_inputs(cx, &candidate, &predecessor, build.path())?;
        deferred_quality::carry_after_maintenance(
            cx,
            &candidate,
            &predecessor,
            build.path(),
            || Ok(checkpoint),
        )?;
        let publication = build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, candidate_mode),
            precommit,
        )?;
        Ok((Some(publication), targets.len()))
    }

    /// Compact every retained vector tier into a complete successor while old
    /// readers keep their original files. A quality-only generation needs only
    /// its quality producer and reports that tier's counts at the top level,
    /// identified by `primary_tier: "quality"`. Two-tier output keeps its nested
    /// quality statistics. The caller reports success only after checking the
    /// publication outcome. A lexical-only selection has nothing to compact
    /// and returns no publication.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn compact_retained_generation(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<(
        Option<crate::generation_store::GenerationPublication>,
        serde_json::Value,
    )> {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let store = CompleteGenerationStore::open(cx, store_root)?;
        // A lexical-only generation (built without semantic loaders) has no
        // vector tier to compact, and its keyword index merges itself: as on
        // the legacy layout there is nothing to do, so nothing is published.
        let active = store.active(cx)?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "selection",
                "no complete generation has been published",
            )
        })?;
        let candidate_mode = Self::retained_generation_search_mode(active.path())?;
        if candidate_mode == SearchExecutionMode::LexicalOnly {
            return Ok((None, Self::lexical_only_compact_report().1));
        }
        self.admit_storage_write(cx, store_root, "retained_compact.start")?;
        let build = store.begin(cx)?;
        let mut input = self.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        input.daemon = false;
        input.daemon_socket = None;
        let candidate = self.clone().with_cli_input(input);
        let predecessor =
            retained_reuse::copy_selected_generation(cx, &candidate, &store, build.path())?;
        if predecessor != active {
            return Err(complete_cli::complete_cli_error(
                "compact_selection",
                "selected predecessor changed while preparing compaction",
            ));
        }
        let candidate_lease = crate::lifecycle::PublicationLease::acquire(build.path())?;
        let mut payload = serde_json::Value::Null;
        for (tier, relative) in FSFS_VECTOR_GENERATION_FILES {
            retained_search_checkpoint(cx)?;
            candidate_lease.fence("complete-generation vector compaction")?;
            let path = build.path().join(relative);
            if !complete_cli::complete_entry_exists(&path)? {
                continue;
            }
            let mut index = Self::open_vector_index_for_mutation(&path)?;
            let compact = index.compact()?;
            let vacuum = index.vacuum()?;
            let stats = serde_json::json!({
                "main_records_before": compact.main_records_before,
                "wal_records_merged": compact.wal_records,
                "total_records_after": vacuum.records_after,
                "tombstones_removed": vacuum.tombstones_removed,
                "compaction_elapsed_ms": compact.elapsed_ms,
                "vacuum_elapsed_ms": vacuum.duration.as_secs_f64() * 1000.0,
            });
            if tier == "fast" || candidate_mode == SearchExecutionMode::QualityOnly {
                payload = stats;
                if candidate_mode == SearchExecutionMode::QualityOnly
                    && let Some(object) = payload.as_object_mut()
                {
                    object.insert("primary_tier".to_owned(), serde_json::json!("quality"));
                }
            } else if let Some(object) = payload.as_object_mut() {
                object.insert("quality".to_owned(), stats);
            }
        }
        retained_search_checkpoint(cx)?;
        let mut sentinel = Self::read_index_sentinel(build.path())?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "compact_membership",
                "candidate has no completion sentinel",
            )
        })?;
        "compact".clone_into(&mut sentinel.command);
        sentinel.generated_at_ms = pressure_timestamp_ms();
        for reason in protect_vector_generations(build.path(), "complete-generation compact") {
            if !sentinel.reason_codes.contains(&reason) {
                sentinel.reason_codes.push(reason);
            }
        }
        candidate.write_index_sentinel(build.path(), &sentinel)?;
        candidate_lease.fence("complete-generation compact candidate complete")?;
        drop(candidate_lease);
        reclaim_unpublished_lexical_garbage(cx, build.path()).await?;
        if Self::retained_generation_search_mode(build.path())? != candidate_mode {
            return Err(complete_cli::complete_cli_error(
                "compact_topology",
                "compaction changed the selected generation's retrieval topology",
            ));
        }
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                candidate_mode,
                candidate_mode,
            ),
        )
        .await?;
        drop(resources);
        let checkpoint =
            retained_batch::retain_unchanged_inputs(cx, &candidate, &predecessor, build.path())?;
        deferred_quality::carry_after_maintenance(
            cx,
            &candidate,
            &predecessor,
            build.path(),
            || Ok(checkpoint),
        )?;
        let publication = build.publish(cx, |_, path| {
            Self::validate_search_generation_at_root(path, candidate_mode)
        })?;
        Ok((Some(publication), payload))
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod retained_delete_tests {
    use super::*;
    use crate::generation_store::{
        CompleteGenerationStore, GenerationPublication, PublishedGeneration,
    };
    use asupersync::test_utils::run_test_with_cx;

    async fn fixture(
        cx: &Cx,
        parent: &Path,
        with_wal: bool,
    ) -> (FsfsRuntime, PathBuf, PathBuf, PublishedGeneration) {
        fixture_with_quality(cx, parent, with_wal, None).await
    }

    async fn fixture_with_quality(
        cx: &Cx,
        parent: &Path,
        with_wal: bool,
        quality_embedder: Option<&dyn Embedder>,
    ) -> (FsfsRuntime, PathBuf, PathBuf, PublishedGeneration) {
        fixture_with_quality_windows(cx, parent, with_wal, quality_embedder, 1).await
    }

    async fn fixture_with_quality_windows(
        cx: &Cx,
        parent: &Path,
        with_wal: bool,
        quality_embedder: Option<&dyn Embedder>,
        maximum: usize,
    ) -> (FsfsRuntime, PathBuf, PathBuf, PublishedGeneration) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        for name in ["alpha.md", "beta.md", "beta-notes.md"] {
            fs::write(source.join(name), format!("sharedtoken document {name}")).unwrap();
        }
        let mut config = FsfsConfig::default();
        config.indexing.offline = true;
        config.indexing.fast_window_max_per_file = maximum;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        let store = CompleteGenerationStore::create(cx, &root).unwrap();
        let build = store.begin(cx).unwrap();
        let mut input = runtime.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        let candidate = runtime.clone().with_cli_input(input);
        candidate
            .run_retained_index_with_reuse(cx, &store, build.path())
            .await
            .unwrap();
        // An explicitly synthetic independent quality space checks that the
        // mutation reaches both physical tiers, without claiming model parity.
        let quality_path = build.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE);
        let mut quality = quality_embedder.map_or_else(
            || VectorIndex::create(&quality_path, "test-delete-quality", 3).unwrap(),
            |embedder| {
                VectorIndex::create_with_revision(
                    &quality_path,
                    embedder.id(),
                    &embedder.identity().unwrap().fingerprint(),
                    embedder.dimension(),
                    frankensearch_index::Quantization::F16,
                )
                .unwrap()
            },
        );
        for name in ["alpha.md", "beta.md", "beta-notes.md"] {
            if let Some(embedder) = quality_embedder {
                // Preserve a real partial-coverage fixture across append.
                if name == "beta-notes.md" && maximum == 1 {
                    continue;
                }
                let vector = embedder
                    .embed(cx, &format!("sharedtoken document {name}"))
                    .await
                    .unwrap();
                quality.write_record(name, &vector).unwrap();
            } else {
                quality.write_record(name, &[1.0, 0.0, 0.0]).unwrap();
            }
        }
        quality.finish().unwrap();
        if with_wal {
            for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                let mut index =
                    FsfsRuntime::open_vector_index_for_mutation(&build.path().join(relative))
                        .unwrap();
                let id = index.doc_id_at(0).unwrap().to_owned();
                let vector = index.vector_at_f32(0).unwrap();
                index.append(&id, &vector).unwrap();
                assert_eq!(index.wal_record_count(), 1);
            }
        }
        let storage = Storage::open(PipelineStorageConfig {
            db_path: candidate.resolve_storage_db_path().unwrap(),
            ..PipelineStorageConfig::default()
        })
        .unwrap();
        for name in ["alpha.md", "beta.md", "beta-notes.md"] {
            let text = format!("sharedtoken document {name}");
            storage
                .upsert_document(&frankensearch_storage::DocumentRecord::new(
                    name,
                    &text,
                    frankensearch_storage::ContentHasher::hash(&text),
                    text.chars().count(),
                    1,
                    1,
                ))
                .unwrap();
            storage.set_document_content(name, &text).unwrap();
        }
        drop(storage);
        assert!(protect_vector_generations(build.path(), "delete test fixture").is_empty());
        let publication = build
            .publish(cx, |_, path| {
                FsfsRuntime::validate_search_generation_at_root(path, SearchExecutionMode::Full)
            })
            .unwrap();
        let GenerationPublication::Durable(generation) = publication else {
            panic!("test fixture publication must be durable"); // ubs:ignore — cfg(test) assertion.
        };
        if quality_embedder.is_some() {
            // Keep the initial fast-only build and manual quality coverage,
            // then enable mutations to resolve the injected producer through
            // the ordinary registered-model selection.
            "all-MiniLM-L6-v2".clone_into(&mut runtime.config.indexing.quality_model);
            runtime.config.search.fast_only = false;
        }
        (runtime, source, root, generation)
    }

    fn deletion(runtime: &FsfsRuntime, ids: &[&str], prefix: bool) -> FsfsRuntime {
        let mut input = runtime.cli_input.clone();
        input.command = CliCommand::Delete;
        input.delete_ids = ids.iter().map(|id| (*id).to_owned()).collect();
        input.delete_prefix = prefix;
        runtime.clone().with_cli_input(input)
    }

    fn live_ids(root: &Path, relative: &str) -> BTreeSet<String> {
        VectorIndex::open_read_only(&root.join(relative))
            .unwrap()
            .live_doc_ids()
            .unwrap()
            .into_iter()
            .collect()
    }

    fn file_bytes(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut files = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    pending.push(entry.path());
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

    // An explicit deterministic provider fixture, independently identified
    // from the fast hash control. This proves wiring, not real-model quality.
    struct AppendQualityEmbedder {
        changed_revision: bool,
    }

    impl Embedder for AppendQualityEmbedder {
        fn identity(&self) -> SearchResult<&frankensearch_core::EmbeddingIdentityBundleV1> {
            static IDENTITY: std::sync::OnceLock<frankensearch_core::EmbeddingIdentityBundleV1> =
                std::sync::OnceLock::new();
            static CHANGED_IDENTITY: std::sync::OnceLock<
                frankensearch_core::EmbeddingIdentityBundleV1,
            > = std::sync::OnceLock::new();
            let identity = if self.changed_revision {
                &CHANGED_IDENTITY
            } else {
                &IDENTITY
            };
            Ok(identity.get_or_init(|| {
                let mut identity =
                    frankensearch_core::EmbeddingIdentityBundleV1::explicit_test_model(
                        "retained-append-quality",
                        3,
                    );
                if self.changed_revision {
                    "explicit-test-v2".clone_into(&mut identity.producer.implementation_revision);
                }
                identity
            }))
        }

        fn embed<'a>(
            &'a self,
            _cx: &'a Cx,
            text: &'a str,
        ) -> frankensearch_core::SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                if text.contains("qualityfailuretoken") {
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id().to_owned(),
                        source: Box::new(std::io::Error::other("injected quality failure")),
                    });
                }
                let mut vector = vec![0.0; 3];
                vector[text.len() % 3] = 1.0;
                Ok(vector)
            })
        }

        fn dimension(&self) -> usize {
            3
        }
        fn id(&self) -> &'static str {
            "retained-append-quality"
        }
        fn model_name(&self) -> &'static str {
            "Retained append quality control"
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }
    }

    struct RestoreQuality(Option<Arc<dyn Embedder>>);

    impl Drop for RestoreQuality {
        fn drop(&mut self) {
            set_test_quality_embedder(self.0.take());
        }
    }

    struct RestoreFast(Option<Arc<dyn Embedder>>);

    impl Drop for RestoreFast {
        fn drop(&mut self) {
            set_test_fast_embedder(self.0.take());
        }
    }

    struct ForeignBoundAppendEmbedder {
        inner: Arc<dyn Embedder>,
        cancel: bool,
    }

    impl Embedder for ForeignBoundAppendEmbedder {
        fn identity(&self) -> SearchResult<&frankensearch_core::EmbeddingIdentityBundleV1> {
            self.inner.identity()
        }

        fn embed<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> frankensearch_core::SearchFuture<'a, Vec<f32>> {
            // The raw path deliberately succeeds: trusting advertised identity
            // while bypassing the bound response would wrongly publish it.
            self.inner.embed(cx, text)
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> frankensearch_core::SearchFuture<'a, frankensearch_core::IdentityBoundEmbedding>
        {
            Box::pin(async move {
                let mut bound = self.inner.embed_bound(cx, text).await?;
                if self.cancel {
                    cx.set_cancel_requested(true);
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id().to_owned(),
                        source: Box::new(std::io::Error::other("cancelled provider failure")),
                    });
                }
                bound
                    .identity
                    .producer
                    .implementation_revision
                    .push_str("-foreign");
                bound.validate()?;
                Ok(bound)
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
            self.inner.is_semantic()
        }

        fn category(&self) -> ModelCategory {
            self.inner.category()
        }
    }

    /// Observes actual bound dispatch through the CLI append path. Deliberate
    /// drift returns an old, valid response after changing the live producer;
    /// response-only identity checks cannot detect that failure.
    struct ObservedAppendEmbedder {
        inner: Arc<dyn Embedder>,
        foreign: frankensearch_core::EmbeddingIdentityBundleV1,
        drift_after_inference: bool,
        drifted: std::sync::atomic::AtomicBool,
        singles: std::sync::atomic::AtomicUsize,
        batches: std::sync::Mutex<Vec<usize>>,
    }

    impl ObservedAppendEmbedder {
        fn new(inner: Arc<dyn Embedder>, drift_after_inference: bool) -> Self {
            let mut foreign = inner.identity().unwrap().clone();
            foreign
                .producer
                .implementation_revision
                .push_str("-changed");
            Self {
                inner,
                foreign,
                drift_after_inference,
                drifted: std::sync::atomic::AtomicBool::new(false),
                singles: std::sync::atomic::AtomicUsize::new(0),
                batches: std::sync::Mutex::new(Vec::new()),
            }
        }

        async fn respond(
            &self,
            cx: &Cx,
            texts: &[&str],
        ) -> SearchResult<Vec<frankensearch_core::IdentityBoundEmbedding>> {
            let mut rows = Vec::with_capacity(texts.len());
            for text in texts {
                rows.push(self.inner.embed_bound(cx, text).await?);
            }
            if self.drift_after_inference {
                self.drifted
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(rows)
        }
    }

    impl Embedder for ObservedAppendEmbedder {
        fn identity(&self) -> SearchResult<&frankensearch_core::EmbeddingIdentityBundleV1> {
            if self.drifted.load(std::sync::atomic::Ordering::SeqCst) {
                Ok(&self.foreign)
            } else {
                self.inner.identity()
            }
        }

        fn bound_batch_is_native(&self) -> bool {
            true
        }

        fn embed<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> frankensearch_core::SearchFuture<'a, Vec<f32>> {
            self.inner.embed(cx, text)
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> frankensearch_core::SearchFuture<'a, frankensearch_core::IdentityBoundEmbedding>
        {
            Box::pin(async move {
                self.singles
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(self.respond(cx, &[text]).await?.pop().unwrap())
            })
        }

        fn embed_batch_bound<'a>(
            &'a self,
            cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> frankensearch_core::SearchFuture<'a, Vec<frankensearch_core::IdentityBoundEmbedding>>
        {
            Box::pin(async move {
                self.batches.lock().unwrap().push(texts.len());
                self.respond(cx, texts).await
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
            self.inner.is_semantic()
        }

        fn category(&self) -> ModelCategory {
            self.inner.category()
        }
    }

    fn append_input(runtime: &FsfsRuntime, parent: &Path, text: &str) -> FsfsRuntime {
        let path = parent.join("append.jsonl");
        fs::write(&path, text).unwrap();
        let mut input = runtime.cli_input.clone();
        input.command = CliCommand::AppendBatch;
        input.input_file = Some(path);
        runtime.clone().with_cli_input(input)
    }

    fn inspect_catalog_copy(parent: &Path, generation: &Path) -> Storage {
        let copy = parent.join("append-catalog-inspection");
        fs::create_dir(&copy).unwrap();
        for entry in fs::read_dir(generation).unwrap() {
            let entry = entry.unwrap();
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("catalog.sqlite")
            {
                fs::copy(entry.path(), copy.join(entry.file_name())).unwrap();
            }
        }
        Storage::open(PipelineStorageConfig {
            db_path: copy.join("catalog.sqlite"),
            ..PipelineStorageConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn retained_append_replaces_duplicates_across_both_tiers_without_reading_source() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (runtime, source, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let before = file_bytes(predecessor.path());
            let old = runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::rename(source.join("alpha.md"), parent.path().join("old-alpha.md")).unwrap();
            fs::write(source.join("unrequested.md"), "must not be discovered").unwrap();
            let canonicalizer = DefaultCanonicalizer::default();
            let raw = format!("\r\nappendnovel replacement α {}\r\n", "β".repeat(450));
            let expected = canonicalizer.canonicalize(&raw);
            let raw_new = format!("\r\nappendnovel new body {}\r\n", "新".repeat(450));
            let expected_new = canonicalizer.canonicalize(&raw_new);
            assert!(expected.chars().count() > 400);
            assert!(expected_new.chars().count() > 400);
            let input = [
                serde_json::json!({"id": "alpha.md", "text": "discarded duplicate body"}),
                serde_json::json!({"id": "virtual/new.md", "text": raw_new}),
                serde_json::json!({"id": "alpha.md", "text": raw}),
            ]
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
            let command = append_input(&runtime, parent.path(), &input);
            let (publication, count) = command
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 2, "duplicate IDs count once");
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("append must publish durably"); // ubs:ignore — cfg(test) assertion.
            };
            let expected_ids = BTreeSet::from([
                "alpha.md".to_owned(),
                "beta.md".to_owned(),
                "beta-notes.md".to_owned(),
                "virtual/new.md".to_owned(),
            ]);
            assert_eq!(live_ids(next.path(), FSFS_VECTOR_INDEX_FILE), expected_ids);
            assert_eq!(
                live_ids(next.path(), FSFS_VECTOR_QUALITY_INDEX_FILE),
                BTreeSet::from([
                    "alpha.md".to_owned(),
                    "beta.md".to_owned(),
                    "virtual/new.md".to_owned(),
                ])
            );
            let fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert_eq!(
                read_indexed_document_text(
                    &cx,
                    old.resources.lexical_index.as_ref().unwrap(),
                    "alpha.md",
                )
                .unwrap(),
                "sharedtoken document alpha.md"
            );
            assert_eq!(
                read_indexed_document_text(
                    &cx,
                    fresh.resources.lexical_index.as_ref().unwrap(),
                    "alpha.md",
                )
                .unwrap(),
                expected
            );
            assert_eq!(
                read_indexed_document_text(
                    &cx,
                    fresh.resources.lexical_index.as_ref().unwrap(),
                    "virtual/new.md",
                )
                .unwrap(),
                expected_new
            );
            for (relative, embedder) in [
                (
                    FSFS_VECTOR_INDEX_FILE,
                    runtime.resolve_fast_embedder().unwrap(),
                ),
                (FSFS_VECTOR_QUALITY_INDEX_FILE, Arc::clone(&quality)),
            ] {
                let index = VectorIndex::open_read_only(&next.path().join(relative)).unwrap();
                assert_eq!(index.wal_record_count(), 0);
                assert_eq!(index.tombstone_count(), 0);
                let row = (0..index.record_count())
                    .find(|row| index.doc_id_at(*row).unwrap() == "alpha.md")
                    .unwrap();
                let expected_vector = embedder.embed(&cx, &expected).await.unwrap();
                let actual = index.vector_at_f32(row).unwrap();
                assert_eq!(actual.len(), expected_vector.len());
                assert!(
                    actual
                        .iter()
                        .zip(expected_vector)
                        .all(|(actual, expected)| (*actual - expected).abs() < 0.002)
                );
                assert_eq!(
                    fsfs_fsvi_protector()
                        .unwrap()
                        .verify(&next.path().join(relative))
                        .unwrap(),
                    FsviVerifyResult::Intact
                );
            }
            let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
                .unwrap()
                .unwrap();
            assert_eq!(
                manifests.keys().cloned().collect::<BTreeSet<_>>(),
                expected_ids
            );
            assert_eq!(
                manifests["alpha.md"].canonical_bytes,
                u64::try_from(expected.len()).unwrap()
            );
            let sentinel = FsfsRuntime::read_index_sentinel(next.path())
                .unwrap()
                .unwrap();
            assert_eq!(sentinel.indexed_files, 4);
            assert_eq!(
                sentinel.total_canonical_bytes,
                manifests
                    .values()
                    .map(|entry| entry.canonical_bytes)
                    .sum::<u64>()
            );
            let catalog = inspect_catalog_copy(parent.path(), next.path());
            let document = catalog.get_document("alpha.md").unwrap().unwrap();
            assert_eq!(
                document.content_preview,
                expected.chars().take(400).collect::<String>()
            );
            assert_eq!(document.content_length, expected.chars().count());
            assert_eq!(
                document.content_hash,
                frankensearch_storage::ContentHasher::hash(&expected)
            );
            assert_eq!(document.created_at, 1);
            assert!(document.source_path.is_none());
            for (id, canonical) in [("alpha.md", &expected), ("virtual/new.md", &expected_new)] {
                let document = catalog.get_document(id).unwrap().unwrap();
                assert!(document.source_path.is_none());
                assert_eq!(document.content_length, canonical.chars().count());
                assert_eq!(
                    document.content_hash,
                    frankensearch_storage::ContentHasher::hash(canonical),
                );
                assert_eq!(
                    catalog.get_document_content(id).unwrap().as_ref(),
                    Some(canonical),
                    "a preview or an older content row cannot recover a source-less upsert",
                );
            }
            assert_eq!(
                catalog.get_document_content("beta.md").unwrap().as_deref(),
                Some("sharedtoken document beta.md"),
                "untouched durable input survives the batch",
            );
            for embedder in [runtime.resolve_fast_embedder().unwrap(), quality] {
                let pending = catalog.list_pending_embeddings(embedder.id(), 20).unwrap();
                assert!(
                    !pending
                        .iter()
                        .any(|id| id == "alpha.md" || id == "virtual/new.md")
                );
            }
            assert!(!source.join("alpha.md").exists());
            assert!(!source.join("virtual/new.md").exists());
            let old_reuse: serde_json::Value =
                serde_json::from_slice(&before[Path::new("FSFS-REUSE.json")]).unwrap();
            let new_reuse: serde_json::Value =
                serde_json::from_slice(&fs::read(next.path().join("FSFS-REUSE.json")).unwrap())
                    .unwrap();
            for key in [
                "session",
                "executable_sha256",
                "configuration_sha256",
                "indexing_contract_sha256",
            ] {
                assert_eq!(
                    new_reuse[key], old_reuse[key],
                    "preserve the original scope"
                );
            }
            for id in ["beta.md", "beta-notes.md"] {
                assert_eq!(
                    new_reuse["checkpoint"]["files"][id], old_reuse["checkpoint"]["files"][id],
                    "an untouched source keeps its exact witness"
                );
            }
            for id in ["alpha.md", "virtual/new.md"] {
                let entry = &new_reuse["checkpoint"]["files"][id];
                assert_eq!(entry["lexical_indexed"], false);
                assert_eq!(entry["semantic_indexed"], false);
                assert_eq!(entry["content_hash_hex"], "");
                assert_eq!(entry["reason_code"], "append_batch");
            }
            assert_eq!(sentinel.command, "append-batch");
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(next)
            );
        });
    }

    #[test]
    fn retained_append_waits_for_shared_admission_and_uses_bounded_native_batches() {
        use crate::concurrency::{BudgetSchedulerMode, RuntimeWorkloadScheduler, WorkloadClass};
        use std::sync::atomic::Ordering;

        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore_quality = RestoreQuality(test_quality_embedder_override());
            let _restore_fast = RestoreFast(test_fast_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (mut runtime, _, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let before = file_bytes(predecessor.path());
            let fast = Arc::new(ObservedAppendEmbedder::new(
                runtime.resolve_fast_embedder().unwrap(),
                false,
            ));
            let quality = Arc::new(ObservedAppendEmbedder::new(quality, false));
            set_test_fast_embedder(Some(fast.clone()));
            set_test_quality_embedder(Some(quality.clone()));
            runtime.workload_scheduler = Arc::new(RuntimeWorkloadScheduler::new(
                1,
                BudgetSchedulerMode::FairShare,
            ));
            let held = runtime
                .workload_scheduler
                .acquire(&cx, WorkloadClass::Query)
                .await
                .unwrap();
            let input = (0..130)
                .map(|n| {
                    serde_json::json!({"id": format!("new-{n:03}.md"), "text": "new body"})
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join("\n");
            let command = append_input(&runtime, parent.path(), &input);
            let mut pending = Box::pin(command.append_retained_generation(&cx, &root));
            std::future::poll_fn(|task| {
                assert!(pending.as_mut().poll(task).is_pending());
                if runtime.workload_scheduler.pending_for(WorkloadClass::Embed) == 1 {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            })
            .await;
            for producer in [&fast, &quality] {
                assert_eq!(producer.singles.load(Ordering::SeqCst), 0);
                assert!(producer.batches.lock().unwrap().is_empty());
            }
            assert_eq!(file_bytes(predecessor.path()), before);
            cx.set_cancel_requested(true);
            assert!(matches!(pending.await, Err(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
            assert_eq!(
                runtime.workload_scheduler.pending_for(WorkloadClass::Embed),
                0
            );
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor.clone())
            );
            drop(held);

            let (publication, count) = command
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 130);
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("admitted append must publish durably");
            };
            for producer in [&fast, &quality] {
                assert_eq!(producer.singles.load(Ordering::SeqCst), 0);
                assert_eq!(*producer.batches.lock().unwrap(), [64, 64, 2]);
            }
            assert_eq!(live_ids(next.path(), FSFS_VECTOR_INDEX_FILE).len(), 133);
            assert_eq!(
                live_ids(next.path(), FSFS_VECTOR_QUALITY_INDEX_FILE).len(),
                132
            );
            assert_eq!(file_bytes(predecessor.path()), before);
        });
    }

    #[test]
    fn retained_append_rechecks_live_producer_after_an_old_identity_response() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore_quality = RestoreQuality(test_quality_embedder_override());
            let _restore_fast = RestoreFast(test_fast_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (runtime, _, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let fast = runtime.resolve_fast_embedder().unwrap();
            let before = file_bytes(predecessor.path());
            let command = append_input(
                &runtime,
                parent.path(),
                "{\"id\":\"alpha.md\",\"text\":\"must not replace the old document\"}",
            );
            for drifting_tier in ["fast", "quality"] {
                set_test_fast_embedder(Some(Arc::new(ObservedAppendEmbedder::new(
                    Arc::clone(&fast),
                    drifting_tier == "fast",
                ))));
                set_test_quality_embedder(Some(Arc::new(ObservedAppendEmbedder::new(
                    Arc::clone(&quality),
                    drifting_tier == "quality",
                ))));
                assert!(matches!(
                    command.append_retained_generation(&cx, &root).await,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
                assert_eq!(file_bytes(predecessor.path()), before);
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(predecessor.clone())
                );
            }
        });
    }

    #[test]
    fn retained_lexical_append_preserves_complete_catalog_input_after_source_loss() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let source = parent.path().join("source");
            let root = parent.path().join("store");
            fs::create_dir(&source).unwrap();
            let original = "oldsourceword original document";
            fs::write(source.join("alpha.md"), original).unwrap();
            let mut config = FsfsConfig::default();
            config.indexing.offline = true;
            config.indexing.quality_model.clear();
            config.search.fast_only = true;
            config.search.rerank = false;
            config.pressure.degradation_override =
                crate::config::DegradationOverrideMode::ForceLexicalOnly;
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Index,
                target_path: Some(source.clone()),
                index_dir: Some(root.clone()),
                quiet: true,
                ..CliInput::default()
            });
            let store = CompleteGenerationStore::create(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            let mut input = runtime.cli_input.clone();
            input.index_dir = Some(build.path().to_path_buf());
            let candidate = runtime.clone().with_cli_input(input);
            candidate
                .run_retained_index_with_reuse(&cx, &store, build.path())
                .await
                .unwrap();
            let catalog = Storage::open(PipelineStorageConfig {
                db_path: candidate.resolve_storage_db_path().unwrap(),
                ..PipelineStorageConfig::default()
            })
            .unwrap();
            catalog
                .upsert_document(&frankensearch_storage::DocumentRecord::new(
                    "alpha.md",
                    original,
                    frankensearch_storage::ContentHasher::hash(original),
                    original.chars().count(),
                    1,
                    1,
                ))
                .unwrap();
            catalog.set_document_content("alpha.md", original).unwrap();
            drop(catalog);
            let GenerationPublication::Durable(predecessor) = build
                .publish(&cx, |_, path| {
                    FsfsRuntime::validate_search_generation_at_root(
                        path,
                        SearchExecutionMode::LexicalOnly,
                    )
                })
                .unwrap()
            else {
                panic!("lexical fixture must publish durably"); // ubs:ignore — cfg(test) assertion.
            };
            let before = file_bytes(predecessor.path());
            let old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let moved = parent.path().join("moved-source");
            fs::rename(&source, &moved).unwrap();
            let bodies = [
                ("alpha.md", format!("replacementword {}", "é".repeat(600))),
                (
                    "opaque/new.md",
                    format!("newopaqueword {}", "界".repeat(600)),
                ),
            ];
            let jsonl = bodies
                .iter()
                .map(|(id, text)| serde_json::json!({"id": id, "text": text}).to_string())
                .collect::<Vec<_>>()
                .join("\n");
            let command = append_input(&runtime, parent.path(), &jsonl);
            let (publication, count) = command
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 2);
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("lexical append must publish durably"); // ubs:ignore — cfg(test) assertion.
            };
            assert!(!next.path().join(FSFS_VECTOR_INDEX_FILE).exists());
            assert!(!next.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE).exists());
            let fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let catalog = inspect_catalog_copy(parent.path(), next.path());
            for (id, text) in &bodies {
                let canonical = DefaultCanonicalizer::default().canonicalize(text);
                assert!(canonical.chars().count() > 400);
                let document = catalog.get_document(id).unwrap().unwrap();
                assert!(document.source_path.is_none());
                assert_eq!(document.content_preview.chars().count(), 400);
                assert_eq!(document.content_length, canonical.chars().count());
                assert_eq!(
                    document.content_hash,
                    frankensearch_storage::ContentHasher::hash(&canonical),
                );
                assert_eq!(
                    catalog.get_document_content(id).unwrap().as_deref(),
                    Some(canonical.as_str()),
                );
                assert_eq!(
                    read_indexed_document_text(
                        &cx,
                        fresh.resources.lexical_index.as_ref().unwrap(),
                        id,
                    )
                    .unwrap(),
                    LEXICAL_CANONICALIZER.canonicalize(text),
                );
            }
            assert_eq!(
                read_indexed_document_text(
                    &cx,
                    old.resources.lexical_index.as_ref().unwrap(),
                    "alpha.md",
                )
                .unwrap(),
                original,
            );
            assert!(!source.exists());
            assert_eq!(
                fs::read_to_string(moved.join("alpha.md")).unwrap(),
                original
            );
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(store.active(&cx).unwrap(), Some(next));
        });
    }

    #[test]
    fn retained_append_preserves_full_lexical_text_and_bounded_embedding_inputs() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (runtime, source, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let before = file_bytes(predecessor.path());
            let bodies = [
                (
                    "alpha.md",
                    format!("# Café\r\n{}\r\nappendtailsentinel", "context ".repeat(400)),
                    "appendtailsentinel",
                ),
                (
                    "virtual/code.md",
                    format!(
                        "```rust\r\n{}appendmiddlesentinel\r\n{}```\r\n",
                        "let before = 1;\r\n".repeat(25),
                        "let after = 2;\r\n".repeat(25),
                    ),
                    "appendmiddlesentinel",
                ),
            ];
            let input = bodies
                .iter()
                .map(|(id, text, _)| serde_json::json!({"id": id, "text": text}).to_string())
                .collect::<Vec<_>>()
                .join("\n");
            let command = append_input(&runtime, parent.path(), &input);
            let (publication, count) = command
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 2);
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("append must publish durably"); // ubs:ignore — cfg(test) assertion.
            };
            // Reopen all retrieval resources from the persisted successor;
            // the original source still contains the predecessor's body.
            assert_eq!(
                fs::read_to_string(source.join("alpha.md")).unwrap(),
                "sharedtoken document alpha.md",
            );
            assert!(!source.join("virtual/code.md").exists());
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let catalog = inspect_catalog_copy(parent.path(), next.path());
            let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
                .unwrap()
                .unwrap();
            for (id, text, marker) in &bodies {
                let bounded = DefaultCanonicalizer::default().canonicalize(text);
                let lexical = LEXICAL_CANONICALIZER.canonicalize(text);
                assert!(!bounded.contains(*marker));
                assert!(lexical.contains(*marker));
                assert!(bounded.chars().count() <= 2_000);
                assert!(
                    old.resources
                        .lexical_index
                        .as_ref()
                        .unwrap()
                        .search_doc_ids(&cx, marker, 10)
                        .unwrap()
                        .is_empty(),
                );
                let payloads = fresh.search(&cx, marker, 10).await.unwrap();
                assert!(payloads.iter().any(|payload| {
                    payload
                        .hits
                        .iter()
                        .any(|hit| hit.path == *id && hit.lexical_rank.is_some())
                }));
                assert_eq!(
                    read_indexed_document_text(
                        &cx,
                        fresh.resources.lexical_index.as_ref().unwrap(),
                        id,
                    )
                    .unwrap(),
                    lexical,
                );
                for (relative, embedder) in [
                    (
                        FSFS_VECTOR_INDEX_FILE,
                        runtime.resolve_fast_embedder().unwrap(),
                    ),
                    (FSFS_VECTOR_QUALITY_INDEX_FILE, Arc::clone(&quality)),
                ] {
                    let index = VectorIndex::open_read_only(&next.path().join(relative)).unwrap();
                    let row = (0..index.record_count())
                        .find(|row| index.doc_id_at(*row).unwrap() == *id)
                        .unwrap();
                    let expected = embedder.embed(&cx, &bounded).await.unwrap();
                    let actual = index.vector_at_f32(row).unwrap();
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(actual, expected)| { (*actual - expected).abs() < 0.002 })
                    );
                }
                let document = catalog.get_document(id).unwrap().unwrap();
                assert_eq!(document.content_length, bounded.chars().count());
                assert_eq!(
                    document.content_hash,
                    frankensearch_storage::ContentHasher::hash(&bounded),
                );
                assert_eq!(
                    catalog.get_document_content(id).unwrap().as_deref(),
                    Some(bounded.as_str()),
                    "catalog input must match bounded inference, not whole lexical text",
                );
                assert_eq!(
                    manifests[*id].canonical_bytes,
                    u64::try_from(bounded.len()).unwrap(),
                );
            }
            assert_eq!(file_bytes(predecessor.path()), before);
        });
    }

    #[test]
    fn retained_fast_windows_append_shrink_delete_and_keep_old_reader() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (mut runtime, _, root, predecessor) =
                fixture_with_quality_windows(&cx, parent.path(), false, Some(quality.as_ref()), 4)
                    .await;
            // Commands inherit the indexed policy even when a later process
            // uses its default configuration rather than the original file.
            runtime.config.indexing.fast_window_max_per_file = 1;
            let predecessor_bytes = file_bytes(predecessor.path());
            let body = format!(
                "Introduction. {} Deep café tail evidence.",
                "background text ".repeat(1_000)
            );
            let input = ["alpha.md", "virtual/deep.md"]
                .into_iter()
                .map(|id| serde_json::json!({"id":id,"text":body}).to_string())
                .collect::<Vec<_>>()
                .join("\n");
            let (publication, count) = append_input(&runtime, parent.path(), &input)
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 2, "one receipt item per source, not per window");
            let Some(GenerationPublication::Durable(long_generation)) = publication else {
                panic!("window append must publish durably");
            };
            let old_reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            let long_bytes = file_bytes(long_generation.path());
            let manifests = FsfsRuntime::read_matching_manifest_generation(long_generation.path())
                .unwrap()
                .unwrap();
            let canonical = LEXICAL_CANONICALIZER.canonicalize(&body);
            let fast = runtime.resolve_fast_embedder().unwrap();
            {
                let index = VectorIndex::open_read_only(
                    &long_generation.path().join(FSFS_VECTOR_INDEX_FILE),
                )
                .unwrap();
                for id in ["alpha.md", "virtual/deep.md"] {
                    let plan = manifests[id].fast_windows.as_ref().unwrap();
                    assert_eq!(plan.max_per_file, 4);
                    assert_eq!(plan.windows.len(), 4);
                    assert!(
                        plan.texts(&canonical)
                            .unwrap()
                            .last()
                            .unwrap()
                            .contains("tail evidence")
                    );
                    for (row_id, text) in plan
                        .row_ids(id)
                        .into_iter()
                        .zip(plan.texts(&canonical).unwrap())
                    {
                        let row = (0..index.record_count())
                            .find(|row| index.doc_id_at(*row).unwrap() == row_id)
                            .unwrap();
                        let expected = fast.embed(&cx, text).await.unwrap();
                        assert!(
                            index
                                .vector_at_f32(row)
                                .unwrap()
                                .iter()
                                .zip(expected)
                                .all(|(a, b)| (*a - b).abs() < 0.002)
                        );
                    }
                }
                assert_eq!(index.live_doc_ids().unwrap().len(), 10);
            }
            assert_eq!(
                live_ids(long_generation.path(), FSFS_VECTOR_QUALITY_INDEX_FILE).len(),
                4
            );
            assert_eq!(
                FsfsRuntime::collect_quality_generation_doctor_check(
                    long_generation.path(),
                    crate::runtime::SearchExecutionMode::Full,
                    &mut None,
                    &mut None,
                )
                .verdict,
                DoctorVerdict::Pass
            );

            let short =
                serde_json::json!({"id":"alpha.md","text":"short replacement body"}).to_string();
            let (publication, count) = append_input(&runtime, parent.path(), &short)
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 1);
            let Some(GenerationPublication::Durable(short_generation)) = publication else {
                panic!("shorter replacement must publish durably");
            };
            let remaining_alpha = live_ids(short_generation.path(), FSFS_VECTOR_INDEX_FILE)
                .into_iter()
                .filter(|row| semantic_windows::source_id(row) == "alpha.md")
                .collect::<Vec<_>>();
            assert_eq!(remaining_alpha, ["alpha.md"]);
            let manifests = FsfsRuntime::read_matching_manifest_generation(short_generation.path())
                .unwrap()
                .unwrap();
            assert_eq!(
                manifests["alpha.md"]
                    .fast_windows
                    .as_ref()
                    .unwrap()
                    .windows
                    .len(),
                1
            );
            assert_eq!(old_reader.generation(), &long_generation);
            assert_eq!(file_bytes(long_generation.path()), long_bytes);

            let (publication, count) = deletion(&runtime, &["virtual/"], true)
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 1);
            let Some(GenerationPublication::Durable(deleted)) = publication else {
                panic!("window deletion must publish durably");
            };
            for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                assert!(
                    !live_ids(deleted.path(), relative)
                        .iter()
                        .any(|row| semantic_windows::source_id(row).starts_with("virtual/"))
                );
            }
            let manifests = FsfsRuntime::read_matching_manifest_generation(deleted.path())
                .unwrap()
                .unwrap();
            assert_eq!(manifests.len(), 3);
            assert!(!manifests.contains_key("virtual/deep.md"));
            assert_eq!(file_bytes(predecessor.path()), predecessor_bytes);
            assert_eq!(file_bytes(long_generation.path()), long_bytes);
        });
    }

    #[test]
    fn legacy_fast_windows_append_retires_tail_delete_counts_sources_and_watch_refuses() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (mut runtime, _, root, predecessor) =
                fixture_with_quality_windows(&cx, parent.path(), false, Some(quality.as_ref()), 4)
                    .await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let build = store.begin(&cx).unwrap();
            runtime.cli_input.index_dir = Some(build.path().to_path_buf());
            runtime.cli_input.format = OutputFormat::Json;
            runtime.config.indexing.fast_window_max_per_file = 1;
            assert_eq!(
                retained_reuse::copy_selected_generation(&cx, &runtime, &store, build.path())
                    .unwrap(),
                predecessor
            );

            let body =
                serde_json::json!({"id":"alpha.md","text":"long café source ".repeat(1_000)})
                    .to_string();
            let command = append_input(&runtime, parent.path(), &body);
            let mut receipt = Vec::new();
            command
                .run_append_batch_command_with_writer(&cx, &mut receipt)
                .await
                .unwrap();
            let receipt: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
            assert_eq!(receipt["data"]["appended"], 1);
            assert_eq!(live_ids(build.path(), FSFS_VECTOR_INDEX_FILE).len(), 6);
            assert_eq!(
                live_ids(build.path(), FSFS_VECTOR_QUALITY_INDEX_FILE).len(),
                3
            );
            let sentinel = FsfsRuntime::read_index_sentinel(build.path())
                .unwrap()
                .unwrap();
            assert!(sentinel.generation_complete);
            assert_eq!(sentinel.fast_window_max_per_file, 4);

            // Losing the sentinel policy cannot make append or watch silently
            // reinterpret window rows as ordinary one-vector documents.
            let mut downgraded = sentinel.clone();
            downgraded.fast_window_max_per_file = 1;
            runtime
                .write_index_sentinel(build.path(), &downgraded)
                .unwrap();
            let before_refusal = file_bytes(build.path());
            assert!(
                command
                    .run_append_batch_command_with_writer(&cx, &mut Vec::new())
                    .await
                    .is_err()
            );
            assert!(runtime.build_live_ingest_pipeline(&cx).await.is_err());
            // Publication ownership may update its diagnostic lease record;
            // serving vectors, source plans and sentinel remain byte-exact.
            for relative in [
                FSFS_VECTOR_INDEX_FILE,
                FSFS_VECTOR_QUALITY_INDEX_FILE,
                FSFS_VECTOR_MANIFEST_FILE,
            ] {
                assert_eq!(
                    fs::read(build.path().join(relative)).unwrap(),
                    before_refusal[&PathBuf::from(relative)]
                );
            }
            assert_eq!(
                FsfsRuntime::read_index_sentinel(build.path()).unwrap(),
                Some(downgraded)
            );
            runtime
                .write_index_sentinel(build.path(), &sentinel)
                .unwrap();

            let short = serde_json::json!({"id":"alpha.md","text":"short replacement"}).to_string();
            append_input(&runtime, parent.path(), &short)
                .run_append_batch_command_with_writer(&cx, &mut Vec::new())
                .await
                .unwrap();
            assert_eq!(live_ids(build.path(), FSFS_VECTOR_INDEX_FILE).len(), 3);
            let manifests = FsfsRuntime::read_matching_manifest_generation(build.path())
                .unwrap()
                .unwrap();
            assert_eq!(
                manifests["alpha.md"]
                    .fast_windows
                    .as_ref()
                    .unwrap()
                    .windows
                    .len(),
                1
            );
            append_input(&runtime, parent.path(), &body)
                .run_append_batch_command_with_writer(&cx, &mut Vec::new())
                .await
                .unwrap();
            deletion(&runtime, &["alpha.md"], false)
                .run_delete_command(&cx)
                .await
                .unwrap();
            for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                assert!(
                    !live_ids(build.path(), relative)
                        .iter()
                        .any(|row| semantic_windows::source_id(row) == "alpha.md")
                );
            }
            let manifests = FsfsRuntime::read_matching_manifest_generation(build.path())
                .unwrap()
                .unwrap();
            assert_eq!(manifests.len(), 2);
            let sentinel = FsfsRuntime::read_index_sentinel(build.path())
                .unwrap()
                .unwrap();
            assert_eq!(sentinel.indexed_files, 2);
            assert!(sentinel.generation_complete);

            // A genuine lexical writer conflict occurs after append has
            // begun its mutation protocol. It must leave an incomplete
            // sentinel instead of making a partial update searchable.
            let layout = FsfsRuntime::resolve_lexical_engine(build.path()).unwrap();
            let lexical_path = layout.engine_dir().unwrap();
            let blocker = QuillIndex::open(&cx, &lexical_path, QuillConfig::default())
                .await
                .unwrap();
            let rows_before = live_ids(build.path(), FSFS_VECTOR_INDEX_FILE);
            assert!(
                command
                    .run_append_batch_command_with_writer(&cx, &mut Vec::new())
                    .await
                    .is_err()
            );
            assert!(
                !FsfsRuntime::read_index_sentinel(build.path())
                    .unwrap()
                    .unwrap()
                    .generation_complete
            );
            assert!(
                FsfsRuntime::validate_search_generation_at_root(
                    build.path(),
                    SearchExecutionMode::Full
                )
                .is_err()
            );
            assert_eq!(live_ids(build.path(), FSFS_VECTOR_INDEX_FILE), rows_before);
            drop(blocker);
        });
    }

    #[test]
    fn retained_fast_windows_cancelled_append_keeps_published_rows() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (runtime, _, root, predecessor) =
                fixture_with_quality_windows(&cx, parent.path(), false, Some(quality.as_ref()), 4)
                    .await;
            let before = file_bytes(predecessor.path());
            let body =
                serde_json::json!({"id":"alpha.md","text":"detailed replacement ".repeat(1_000)})
                    .to_string();
            let error = append_input(&runtime, parent.path(), &body)
                .append_retained_generation_with_precommit(&cx, &root, |cx| {
                    cx.set_cancel_requested(true);
                    retained_search_checkpoint(cx)
                })
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor.clone())
            );
            assert_eq!(file_bytes(predecessor.path()), before);
            let (publication, count) = append_input(&runtime, parent.path(), &body)
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 1);
            assert!(matches!(
                publication,
                Some(GenerationPublication::Durable(_))
            ));
        });
    }

    #[test]
    fn retained_append_deferred_quality_refuses_before_reading_input() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (mut runtime, _, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let before = file_bytes(predecessor.path());
            runtime.config.pressure.degradation_override =
                crate::config::DegradationOverrideMode::ForceEmbedDeferred;
            runtime.cli_input.command = CliCommand::AppendBatch;
            runtime.cli_input.input_file = Some(parent.path().join("must-not-read.jsonl"));
            set_test_quality_embedder(None);
            let error = runtime
                .append_retained_generation(&cx, &root)
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, value, .. }
                    if field == "pressure.degradation_override"
                        && value == "force_embed_deferred"
            ));
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor)
            );
        });
    }

    #[test]
    fn retained_append_refusal_and_cancelled_publication_preserve_predecessor() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore = RestoreQuality(test_quality_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (runtime, _, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let before = file_bytes(predecessor.path());
            for body in [
                "{\"id\":\"alpha.md\",\"text\":\"valid prefix\"}\n{invalid json}",
                "{\"id\":\"alpha.md\",\"text\":\"   \\n\"}",
                "{\"id\":\"alpha.md\",\"text\":\"qualityfailuretoken\"}",
            ] {
                assert!(
                    append_input(&runtime, parent.path(), body)
                        .append_retained_generation(&cx, &root)
                        .await
                        .is_err()
                );
                assert_eq!(file_bytes(predecessor.path()), before);
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(predecessor.clone())
                );
            }
            let command = append_input(
                &runtime,
                parent.path(),
                "{\"id\":\"new.md\",\"text\":\"appended after retry\"}",
            );
            set_test_quality_embedder(None);
            assert!(matches!(
                command.append_retained_generation(&cx, &root).await,
                Err(SearchError::EmbedderUnavailable { .. })
            ));
            // Same public model label and width must not admit another
            // producer revision into the stored quality vector space.
            set_test_quality_embedder(Some(Arc::new(AppendQualityEmbedder {
                changed_revision: true,
            })));
            assert!(matches!(
                command.append_retained_generation(&cx, &root).await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            set_test_quality_embedder(Some(quality));
            let error = command
                .append_retained_generation_with_precommit(&cx, &root, |cx| {
                    cx.set_cancel_requested(true);
                    Ok(())
                })
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor)
            );
            let (publication, count) = command
                .append_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(count, 1);
            assert!(matches!(
                publication,
                Some(GenerationPublication::Durable(_))
            ));
        });
    }

    #[test]
    fn retained_append_refuses_foreign_bound_responses_from_either_tier() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let _restore_quality = RestoreQuality(test_quality_embedder_override());
            let _restore_fast = RestoreFast(test_fast_embedder_override());
            let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                changed_revision: false,
            });
            set_test_quality_embedder(Some(Arc::clone(&quality)));
            let (runtime, _, root, predecessor) =
                fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
            let fast = runtime.resolve_fast_embedder().unwrap();
            let before = file_bytes(predecessor.path());
            let command = append_input(
                &runtime,
                parent.path(),
                "{\"id\":\"alpha.md\",\"text\":\"must not replace the old document\"}",
            );
            for tier in ["fast", "quality"] {
                for cancel in [false, true] {
                    set_test_fast_embedder(Some(Arc::clone(&fast)));
                    set_test_quality_embedder(Some(Arc::clone(&quality)));
                    let foreign: Arc<dyn Embedder> = Arc::new(ForeignBoundAppendEmbedder {
                        inner: Arc::clone(if tier == "fast" { &fast } else { &quality }),
                        cancel,
                    });
                    if tier == "fast" {
                        set_test_fast_embedder(Some(foreign));
                    } else {
                        set_test_quality_embedder(Some(foreign));
                    }
                    let error = command
                        .append_retained_generation(&cx, &root)
                        .await
                        .unwrap_err();
                    if cancel {
                        assert!(
                            matches!(error, SearchError::Cancelled { .. }),
                            "cancellation must precede a provider failure at either inference boundary"
                        );
                        cx.set_cancel_requested(false);
                    } else {
                        assert!(
                            matches!(error, SearchError::UnverifiableRemoteSpace { .. }),
                            "the actual bound response must match the admitted full identity"
                        );
                    }
                    assert_eq!(file_bytes(predecessor.path()), before);
                    assert_eq!(
                        CompleteGenerationStore::open(&cx, &root)
                            .unwrap()
                            .active(&cx)
                            .unwrap(),
                        Some(predecessor.clone())
                    );
                }
            }
        });
    }

    #[test]
    fn retained_delete_preserves_opaque_absolute_ids_and_refuses_ambiguous_selectors() {
        run_test_with_cx(|cx| async move {
            for prefix in [false, true] {
                let parent = tempfile::tempdir().unwrap();
                let _restore = RestoreQuality(test_quality_embedder_override());
                let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                    changed_revision: false,
                });
                set_test_quality_embedder(Some(Arc::clone(&quality)));
                let (runtime, source, root, _) =
                    fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
                let ambiguous = source.join("alpha.md").display().to_string();
                let inside = source.join("virtual").join("one.md").display().to_string();
                let outside = parent
                    .path()
                    .join("external")
                    .join("one.md")
                    .display()
                    .to_string();
                let input = [&ambiguous, &inside, &outside]
                    .into_iter()
                    .map(|id| {
                        serde_json::json!({"id": id, "text": "sharedtoken opaque document"})
                            .to_string()
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let (publication, appended) = append_input(&runtime, parent.path(), &input)
                    .append_retained_generation(&cx, &root)
                    .await
                    .unwrap();
                assert_eq!(appended, 3);
                let Some(GenerationPublication::Durable(predecessor)) = publication else {
                    panic!("opaque append must publish durably"); // ubs:ignore — cfg(test) assertion.
                };
                let before = file_bytes(predecessor.path());
                let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();

                let invalid = deletion(&runtime, &["beta.md", &ambiguous], prefix);
                assert!(matches!(
                    invalid.delete_retained_generation(&cx, &root).await,
                    Err(SearchError::InvalidConfig { field, reason, .. })
                        if field == "complete_generation.delete_ids"
                            && reason.contains("different root-relative documents")
                ));
                assert_eq!(file_bytes(predecessor.path()), before);
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(predecessor.clone()),
                    "an ambiguous selector must not publish even earlier valid selectors"
                );
                let selectors = if prefix {
                    vec![
                        format!(
                            "{}{sep}",
                            source.join("virtual").display(),
                            sep = std::path::MAIN_SEPARATOR
                        ),
                        format!(
                            "{}{sep}",
                            parent.path().join("external").display(),
                            sep = std::path::MAIN_SEPARATOR
                        ),
                        inside.clone(),
                    ]
                } else {
                    vec![inside.clone(), outside.clone(), inside.clone()]
                };
                let refs = selectors.iter().map(String::as_str).collect::<Vec<_>>();
                let (publication, deleted) = deletion(&runtime, &refs, prefix)
                    .delete_retained_generation(&cx, &root)
                    .await
                    .unwrap();
                assert_eq!(deleted, 2, "overlapping selectors count unique members");
                let Some(GenerationPublication::Durable(next)) = publication else {
                    panic!("opaque delete must publish durably"); // ubs:ignore — cfg(test) assertion.
                };
                let expected = BTreeSet::from([
                    "alpha.md".to_owned(),
                    "beta.md".to_owned(),
                    "beta-notes.md".to_owned(),
                    ambiguous.clone(),
                ]);
                assert_eq!(live_ids(next.path(), FSFS_VECTOR_INDEX_FILE), expected);
                assert_eq!(
                    live_ids(next.path(), FSFS_VECTOR_QUALITY_INDEX_FILE),
                    BTreeSet::from([
                        "alpha.md".to_owned(),
                        "beta.md".to_owned(),
                        ambiguous.clone(),
                    ])
                );
                let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
                    .unwrap()
                    .unwrap();
                assert_eq!(manifests.keys().cloned().collect::<BTreeSet<_>>(), expected);
                let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
                assert_eq!(
                    fresh
                        .search(&cx, "sharedtoken", 10)
                        .await
                        .unwrap()
                        .last()
                        .unwrap()
                        .hits
                        .iter()
                        .map(|hit| hit.path.clone())
                        .collect::<BTreeSet<_>>(),
                    expected
                );
                assert_eq!(
                    old.search(&cx, "sharedtoken", 10)
                        .await
                        .unwrap()
                        .last()
                        .unwrap()
                        .hits
                        .len(),
                    6,
                    "retained readers keep both opaque and root-relative documents"
                );
                assert_eq!(file_bytes(predecessor.path()), before);
                let catalog = inspect_catalog_copy(parent.path(), next.path());
                for id in [&inside, &outside] {
                    assert!(catalog.get_document(id).unwrap().is_none());
                    assert!(catalog.get_document_content(id).unwrap().is_none());
                }
                assert!(catalog.get_document(&ambiguous).unwrap().is_some());
                assert!(catalog.get_document("alpha.md").unwrap().is_some());
                assert_eq!(
                    fs::read_to_string(source.join("alpha.md")).unwrap(),
                    "sharedtoken document alpha.md"
                );
                assert!(!Path::new(&inside).exists());
                assert!(!Path::new(&outside).exists());
            }
        });
    }

    #[test]
    fn retained_delete_updates_both_tiers_catalog_and_manifests_without_retargeting_old_reader() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, source, root, predecessor) = fixture(&cx, parent.path(), false).await;
            let before = file_bytes(predecessor.path());
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let outside = parent.path().join("outside/alpha.md").display().to_string();
            let invalid = deletion(&runtime, &["alpha.md", &outside], false);
            assert!(matches!(
                invalid.delete_retained_generation(&cx, &root).await,
                Err(SearchError::InvalidConfig { field, .. })
                    if field == "complete_generation.delete_ids"
            ));
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor.clone())
            );
            let absolute = source.join("beta.md").display().to_string();
            let moved_source = parent.path().join("moved-beta.md");
            fs::rename(source.join("beta.md"), &moved_source).unwrap();
            let mut command = deletion(&runtime, &[&absolute, "beta.md", "absent.md"], false);
            command.cli_input.target_path = None;
            let (publication, deleted) = command
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(deleted, 1);
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("delete must publish one durable successor"); // ubs:ignore — cfg(test) assertion.
            };
            assert_ne!(next.id(), predecessor.id());
            let expected = BTreeSet::from(["alpha.md".to_owned(), "beta-notes.md".to_owned()]);
            for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                assert_eq!(live_ids(next.path(), relative), expected);
                assert_eq!(live_ids(predecessor.path(), relative).len(), 3);
                assert_eq!(
                    VectorIndex::open_read_only(&next.path().join(relative))
                        .unwrap()
                        .wal_record_count(),
                    0
                );
                assert_eq!(
                    fsfs_fsvi_protector()
                        .unwrap()
                        .verify(&next.path().join(relative))
                        .unwrap(),
                    FsviVerifyResult::Intact
                );
            }
            let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
                .unwrap()
                .unwrap();
            assert_eq!(manifests.keys().cloned().collect::<BTreeSet<_>>(), expected);
            let sentinel = FsfsRuntime::read_index_sentinel(next.path())
                .unwrap()
                .unwrap();
            assert_eq!(sentinel.indexed_files, 2);
            assert_eq!(sentinel.index_root, next.path().display().to_string());
            assert_eq!(
                sentinel.source_hash_hex,
                index_source_hash_hex(&manifests.into_values().collect::<Vec<_>>())
            );
            let previous_receipt: serde_json::Value =
                serde_json::from_slice(&before[Path::new("FSFS-REUSE.json")]).unwrap();
            let projected_receipt: serde_json::Value =
                serde_json::from_slice(&fs::read(next.path().join("FSFS-REUSE.json")).unwrap())
                    .unwrap();
            assert_ne!(projected_receipt, previous_receipt);
            for field in ["version", "session", "configuration_sha256"] {
                assert!(previous_receipt.get(field).is_some());
                assert_eq!(projected_receipt.get(field), previous_receipt.get(field));
            }
            // A session-local fixture can have no executable proof. Preserve
            // that absence exactly rather than minting cross-process scope.
            for field in ["executable_sha256", "indexing_contract_sha256"] {
                assert_eq!(projected_receipt.get(field), previous_receipt.get(field));
            }
            let previous_checkpoint: IndexingCheckpoint =
                serde_json::from_value(previous_receipt["checkpoint"].clone()).unwrap();
            let projected_checkpoint: IndexingCheckpoint =
                serde_json::from_value(projected_receipt["checkpoint"].clone()).unwrap();
            let mut expected_checkpoint = previous_checkpoint.clone();
            expected_checkpoint
                .files
                .retain(|id, _| expected.contains(id));
            expected_checkpoint.index_root = next.path().display().to_string();
            expected_checkpoint
                .source_hash_hex
                .clone_from(&sentinel.source_hash_hex);
            expected_checkpoint.discovered_files = sentinel.discovered_files;
            expected_checkpoint.skipped_files = sentinel.skipped_files;
            expected_checkpoint.updated_at_ms = sentinel.generated_at_ms;
            assert_eq!(projected_checkpoint, expected_checkpoint);
            assert!(
                FsfsRuntime::read_checkpoint_manifest_generation(next.path(), &previous_checkpoint)
                    .unwrap()
                    .is_none()
            );
            assert!(
                FsfsRuntime::read_checkpoint_manifest_generation(
                    next.path(),
                    &projected_checkpoint
                )
                .unwrap()
                .is_some()
            );
            assert_eq!(
                old.search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                3
            );
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let hits = fresh
                .search(&cx, "sharedtoken", 10)
                .await
                .unwrap()
                .pop()
                .unwrap()
                .hits;
            assert_eq!(hits.len(), 2);
            assert!(!hits.iter().any(|hit| hit.path == "beta.md"));
            // Storage::open bootstraps writable connections. Inspect an
            // independent catalog copy, including any WAL sidecars, so the
            // test never opens the selected generation with a database writer.
            let catalog_copy_root = parent.path().join("verify-catalog");
            fs::create_dir(&catalog_copy_root).unwrap();
            for entry in fs::read_dir(next.path()).unwrap() {
                let entry = entry.unwrap();
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("catalog.sqlite")
                {
                    fs::copy(entry.path(), catalog_copy_root.join(entry.file_name())).unwrap();
                }
            }
            let storage = Storage::open(PipelineStorageConfig {
                db_path: catalog_copy_root.join("catalog.sqlite"),
                ..PipelineStorageConfig::default()
            })
            .unwrap();
            assert!(storage.get_document("beta.md").unwrap().is_none());
            assert!(storage.get_document_content("beta.md").unwrap().is_none());
            assert!(storage.get_document("alpha.md").unwrap().is_some());
            assert_eq!(
                storage.get_document_content("alpha.md").unwrap().as_deref(),
                Some("sharedtoken document alpha.md"),
            );
            drop(storage);
            assert!(!source.join("beta.md").exists());
            assert!(
                moved_source.exists(),
                "index deletion must retain source files"
            );
            assert!(source.join("alpha.md").exists());
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(next)
            );
            let (publication, deleted) = deletion(&runtime, &[""], true)
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(deleted, 2);
            let Some(GenerationPublication::Durable(empty)) = publication else {
                panic!("deleting every remaining document must publish"); // ubs:ignore — cfg(test) assertion.
            };
            for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                assert!(live_ids(empty.path(), relative).is_empty());
            }
            let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert!(
                reader
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .is_empty()
            );
        });
    }

    #[test]
    fn retained_delete_prefix_and_no_match_are_truthful() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, source, root, _) = fixture(&cx, parent.path(), false).await;
            let absolute = source.join("beta").display().to_string();
            assert!(!source.join("beta").exists());
            let command = deletion(&runtime, &[&absolute], true);
            let (publication, deleted) = command
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(deleted, 2);
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("prefix delete must publish"); // ubs:ignore — cfg(test) assertion.
            };
            let (noop, deleted) = command
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert!(noop.is_none());
            assert_eq!(deleted, 0);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(next)
            );
        });
    }

    #[test]
    fn retained_delete_prefix_includes_lexical_only_documents_without_vector_rows() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, source, root, _) = fixture(&cx, parent.path(), false).await;
            fs::write(
                source.join("lexical-data.csv"),
                "sharedtoken,csv document\n",
            )
            .unwrap();
            let publication = runtime
                .rebuild_retained_generation(&cx, &root)
                .await
                .unwrap();
            let GenerationPublication::Durable(predecessor) = publication else {
                panic!("lexical-only fixture must publish"); // ubs:ignore — cfg(test) assertion.
            };
            let manifest = FsfsRuntime::read_matching_manifest_generation(predecessor.path())
                .unwrap()
                .unwrap();
            assert_eq!(manifest["lexical-data.csv"].ingestion_class, "lexical_only");
            assert!(
                !live_ids(predecessor.path(), FSFS_VECTOR_INDEX_FILE).contains("lexical-data.csv")
            );
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let command = deletion(&runtime, &["lexical-"], true);
            let (publication, deleted) = command
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(deleted, 1);
            assert!(matches!(
                publication,
                Some(GenerationPublication::Durable(_))
            ));
            assert!(
                old.search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .iter()
                    .any(|hit| hit.path == "lexical-data.csv")
            );
            let mut current = runtime.open_retained_search(&cx, &root).await.unwrap();
            assert!(
                !current
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .iter()
                    .any(|hit| hit.path == "lexical-data.csv")
            );
        });
    }

    #[test]
    fn retained_delete_cancelled_after_sealing_preserves_predecessor_and_retries() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root, predecessor) = fixture(&cx, parent.path(), false).await;
            let before = file_bytes(predecessor.path());
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let command = deletion(&runtime, &["beta.md"], false);
            let error = command
                .delete_retained_generation_with_precommit(&cx, &root, |cx| {
                    cx.set_cancel_requested(true);
                    Ok(())
                })
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            cx.set_cancel_requested(false);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(predecessor.clone())
            );
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                old.search(&cx, "sharedtoken", 10)
                    .await
                    .unwrap()
                    .last()
                    .unwrap()
                    .hits
                    .len(),
                3
            );
            let (publication, deleted) = command
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert!(matches!(
                publication,
                Some(GenerationPublication::Durable(_))
            ));
            assert_eq!(deleted, 1);
        });
    }

    #[test]
    fn retained_delete_ignores_tombstoned_manifest_aliases() {
        run_test_with_cx(|cx| async move {
            for tombstone_literal in [false, true] {
                let parent = tempfile::tempdir().unwrap();
                let _restore = RestoreQuality(test_quality_embedder_override());
                let quality: Arc<dyn Embedder> = Arc::new(AppendQualityEmbedder {
                    changed_revision: false,
                });
                set_test_quality_embedder(Some(Arc::clone(&quality)));
                let (runtime, source, root, _) =
                    fixture_with_quality(&cx, parent.path(), false, Some(quality.as_ref())).await;
                let absolute = source.join("alpha.md").display().to_string();
                let input = serde_json::json!({
                    "id": absolute,
                    "text": "sharedtoken opaque document"
                })
                .to_string();
                append_input(&runtime, parent.path(), &input)
                    .append_retained_generation(&cx, &root)
                    .await
                    .unwrap();

                // Paired manifests and completion flags alone do not prove
                // liveness. Seal a stale-metadata fixture whose chosen alias
                // has been tombstoned in every engine, but still has its old
                // manifest entry. The other spelling remains a live document.
                let store = CompleteGenerationStore::open(&cx, &root).unwrap();
                let build = store.begin(&cx).unwrap();
                let mut input = runtime.cli_input.clone();
                input.index_dir = Some(build.path().to_path_buf());
                let candidate = runtime.clone().with_cli_input(input);
                retained_reuse::copy_selected_generation(&cx, &candidate, &store, build.path())
                    .unwrap();
                let stale = if tombstone_literal {
                    absolute.as_str()
                } else {
                    "alpha.md"
                };
                candidate
                    .apply_candidate_lexical_mutations(
                        &cx,
                        build.path(),
                        &[LexicalMutation::delete(
                            stale.to_owned(),
                            0,
                            IngestionClass::Skip,
                            "stale_manifest_fixture",
                        )],
                    )
                    .await
                    .unwrap();
                for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                    let mut index =
                        FsfsRuntime::open_vector_index_for_mutation(&build.path().join(relative))
                            .unwrap();
                    assert_eq!(index.soft_delete_batch(&[stale]).unwrap(), 1);
                    assert!(!index.live_doc_ids().unwrap().iter().any(|id| id == stale));
                }
                {
                    let storage = Storage::open(PipelineStorageConfig {
                        db_path: candidate.resolve_storage_db_path().unwrap(),
                        ..PipelineStorageConfig::default()
                    })
                    .unwrap();
                    storage.delete_document(stale).unwrap();
                }
                assert!(
                    FsfsRuntime::read_matching_manifest_generation(build.path())
                        .unwrap()
                        .unwrap()
                        .contains_key(stale)
                );
                assert!(
                    protect_vector_generations(build.path(), "stale manifest fixture").is_empty()
                );
                let publication = build
                    .publish(&cx, |_, path| {
                        FsfsRuntime::validate_search_generation_at_root(
                            path,
                            SearchExecutionMode::Full,
                        )
                    })
                    .unwrap();
                let GenerationPublication::Durable(predecessor) = publication else {
                    panic!("stale manifest fixture must publish durably"); // ubs:ignore — cfg(test) assertion.
                };
                let before = file_bytes(predecessor.path());
                let command = deletion(&runtime, &[&absolute], false);
                let (publication, deleted) = command
                    .delete_retained_generation(&cx, &root)
                    .await
                    .expect("a tombstoned alias must not make the live selector ambiguous");
                assert_eq!(deleted, 1);
                let Some(GenerationPublication::Durable(next)) = publication else {
                    panic!("live alias deletion must publish durably"); // ubs:ignore — cfg(test) assertion.
                };
                assert_eq!(
                    live_ids(next.path(), FSFS_VECTOR_INDEX_FILE),
                    BTreeSet::from(["beta.md".to_owned(), "beta-notes.md".to_owned()])
                );
                assert_eq!(
                    live_ids(next.path(), FSFS_VECTOR_QUALITY_INDEX_FILE),
                    BTreeSet::from(["beta.md".to_owned()])
                );
                let live_target = if tombstone_literal {
                    "alpha.md"
                } else {
                    absolute.as_str()
                };
                let manifests = FsfsRuntime::read_matching_manifest_generation(next.path())
                    .unwrap()
                    .unwrap();
                assert!(!manifests.contains_key(live_target));
                assert!(manifests.contains_key(stale));
                assert_eq!(file_bytes(predecessor.path()), before);
                let (publication, deleted) = command
                    .delete_retained_generation(&cx, &root)
                    .await
                    .unwrap();
                assert!(publication.is_none());
                assert_eq!(deleted, 0, "stale manifest entries are not live documents");
            }
        });
    }

    #[test]
    fn storage_pressure_refuses_retained_compaction_before_copying_and_allows_retry() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, _, root, predecessor) = fixture(&cx, directory.path(), true).await;
            runtime.config.storage.disk_budget_bytes = Some(0);
            let before = file_bytes(predecessor.path());
            let allocations = fs::read_dir(root.join("generations")).unwrap().count();
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let phases = old.search(&cx, "sharedtoken", 10).await.unwrap();
            let old_hits = serde_json::to_value(&phases[0].hits).unwrap();
            for emergency in [false, true] {
                let mut blocked = runtime.clone();
                blocked.config.storage.storage_pressure_emergency = emergency;
                blocked.config.storage.disk_budget_bytes = Some(u64::from(!emergency));
                let error = blocked
                    .compact_retained_generation(&cx, &root)
                    .await
                    .unwrap_err();
                assert!(storage_admission::storage_admission_denial(&error).is_some());
                cx.set_cancel_requested(true);
                let cancelled = blocked.compact_retained_generation(&cx, &root).await;
                cx.set_cancel_requested(false);
                assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
                assert_eq!(
                    fs::read_dir(root.join("generations")).unwrap().count(),
                    allocations
                );
                assert_eq!(file_bytes(predecessor.path()), before);
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(predecessor.clone())
                );
            }
            let (publication, stats) = runtime
                .compact_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert!(matches!(publication, Some(GenerationPublication::Durable(_))));
            assert_eq!(stats["wal_records_merged"], 1);
            assert_eq!(stats["quality"]["wal_records_merged"], 1);
            assert_eq!(file_bytes(predecessor.path()), before);
            let phases = old.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(serde_json::to_value(&phases[0].hits).unwrap(), old_hits);
        });
    }

    #[test]
    fn retained_compact_merges_both_wals_while_retaining_readers_and_membership() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, _, root, predecessor) = fixture(&cx, parent.path(), true).await;
            let before = file_bytes(predecessor.path());
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let (publication, stats) = runtime
                .compact_retained_generation(&cx, &root)
                .await
                .unwrap();
            let Some(GenerationPublication::Durable(next)) = publication else {
                panic!("compaction must publish durably"); // ubs:ignore — cfg(test) assertion.
            };
            assert_ne!(next.id(), predecessor.id());
            assert_eq!(stats["wal_records_merged"], 1);
            assert_eq!(stats["quality"]["wal_records_merged"], 1);
            assert_eq!(stats["total_records_after"], 3);
            assert_eq!(stats["quality"]["total_records_after"], 3);
            for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
                assert_eq!(
                    live_ids(next.path(), relative),
                    live_ids(predecessor.path(), relative)
                );
                let index = VectorIndex::open_read_only(&next.path().join(relative)).unwrap();
                assert_eq!(index.wal_record_count(), 0);
                assert_eq!(index.tombstone_count(), 0);
                assert_eq!(
                    fsfs_fsvi_protector()
                        .unwrap()
                        .verify(&next.path().join(relative))
                        .unwrap(),
                    FsviVerifyResult::Intact
                );
            }
            assert_eq!(
                FsfsRuntime::read_matching_manifest_generation(next.path()).unwrap(),
                FsfsRuntime::read_matching_manifest_generation(predecessor.path()).unwrap()
            );
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let old_hits = old
                .search(&cx, "sharedtoken", 10)
                .await
                .unwrap()
                .pop()
                .unwrap()
                .hits;
            let fresh_hits = fresh
                .search(&cx, "sharedtoken", 10)
                .await
                .unwrap()
                .pop()
                .unwrap()
                .hits;
            assert_eq!(
                old_hits.iter().map(|hit| &hit.path).collect::<Vec<_>>(),
                fresh_hits.iter().map(|hit| &hit.path).collect::<Vec<_>>()
            );
            assert_eq!(file_bytes(predecessor.path()), before);
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(next)
            );
        });
    }
}

#[cfg(test)]
mod retained_catalog_tests {
    use super::validate_retained_catalog_path;

    #[test]
    fn retained_catalog_accepts_only_generation_local_paths() {
        for path in [
            "{index_dir}/catalog.sqlite",
            "{index_dir}/nested/catalog.sqlite",
            "{index_dir}/./nested/catalog.sqlite",
        ] {
            validate_retained_catalog_path(path).expect("generation-local catalog");
        }
    }

    #[test]
    fn retained_catalog_rejects_prefix_aliases_and_parent_traversal() {
        for path in [
            "",
            "{index_dir}",
            "{index_dir}/",
            "{index_dir}/.",
            "{index_dir}-other/catalog.sqlite",
            "{index_dir}/../catalog.sqlite",
            "{index_dir}/nested/../../catalog.sqlite",
            "{index_dir}/nested/../catalog.sqlite",
            "catalog.sqlite",
            "/tmp/catalog.sqlite",
        ] {
            let error = validate_retained_catalog_path(path).expect_err(path);
            assert!(
                matches!(error, frankensearch_core::SearchError::InvalidConfig { field, .. } if field == "storage.db_path")
            );
        }
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod retained_search_tests {
    use asupersync::test_utils::run_test_with_cx;

    use super::*;
    use crate::generation_store::{CompleteGenerationStore, GenerationPublication};

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
        let source = parent.join("source");
        let store = parent.join("index");
        fs::create_dir(&source).expect("source directory");
        fs::write(
            source.join("alpha.md"),
            "sharedtoken alpha retained document",
        )
        .expect("first source");
        let mut config = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.search.fast_only = true;
        config.search.rerank = false;
        let input = CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(store.clone()),
            quiet: true,
            ..CliInput::default()
        };
        (
            FsfsRuntime::new(config).with_cli_input(input),
            source,
            store,
        )
    }

    async fn publish(runtime: &FsfsRuntime, cx: &Cx, root: &Path) {
        assert!(matches!(
            runtime
                .rebuild_retained_generation(cx, root)
                .await
                .expect("rebuild"),
            GenerationPublication::Durable(_)
        ));
    }

    #[test]
    fn storage_pressure_refuses_fresh_indexing_before_store_creation_and_allows_retry() {
        run_test_with_cx(|cx| async move {
            for retained in [false, true] {
                let directory = tempfile::tempdir().unwrap();
                let (mut runtime, _, root) = fixture(directory.path());
                runtime.lexical_only_indexing = true;
                runtime.config.pressure.degradation_override =
                    DegradationOverrideMode::ForceLexicalOnly;
                runtime.config.storage.disk_budget_bytes = Some(0);
                runtime.config.storage.storage_pressure_emergency = true;
                let result = if retained {
                    runtime
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .map(|_| ())
                } else {
                    runtime
                        .run_one_shot_index_scaffold_internal(
                            &cx,
                            CliCommand::Index,
                            |_| Ok(()),
                            false,
                            true,
                        )
                        .await
                        .map(|_| ())
                };
                assert!(
                    storage_admission::storage_admission_denial(&result.unwrap_err()).is_some(),
                    "retained={retained}: emergency overrides an unlimited budget"
                );
                assert!(!root.exists(), "refusal must precede store creation");

                cx.set_cancel_requested(true);
                let cancelled = if retained {
                    runtime
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .map(|_| ())
                } else {
                    runtime
                        .run_one_shot_index_scaffold_internal(
                            &cx,
                            CliCommand::Index,
                            |_| Ok(()),
                            false,
                            true,
                        )
                        .await
                        .map(|_| ())
                };
                cx.set_cancel_requested(false);
                assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
                assert!(!root.exists());

                runtime.config.storage.storage_pressure_emergency = false;
                if retained {
                    publish(&runtime, &cx, &root).await;
                    let mut reader = runtime.open_retained_search(&cx, &root).await.unwrap();
                    let phases = reader.search(&cx, "sharedtoken", 10).await.unwrap();
                    assert_eq!(phases[0].hits.len(), 1);
                } else {
                    runtime
                        .run_one_shot_index_scaffold_internal(
                            &cx,
                            CliCommand::Index,
                            |_| Ok(()),
                            false,
                            true,
                        )
                        .await
                        .unwrap();
                    assert!(
                        FsfsRuntime::read_index_sentinel(&root)
                            .unwrap()
                            .is_some_and(|sentinel| sentinel.generation_complete)
                    );
                }
            }
        });
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn storage_pressure_preserves_retained_mutations_and_real_noops() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (mut runtime, source, root) = fixture(directory.path());
            runtime.lexical_only_indexing = true;
            runtime.config.pressure.degradation_override =
                DegradationOverrideMode::ForceLexicalOnly;
            runtime.config.storage.disk_budget_bytes = Some(0);
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let first = store.active(&cx).unwrap().unwrap();
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let phases = old.search(&cx, "sharedtoken", 10).await.unwrap();
            let before = serde_json::to_value(&phases[0].hits).unwrap();
            let bundle = fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap();
            let allocations = fs::read_dir(root.join("generations")).unwrap().count();
            fs::write(source.join("beta.md"), "sharedtoken beta queued source").unwrap();
            let mutations = [
                retained_batch::RetainedMutation::Delete {
                    id: "alpha.md".to_owned(),
                },
                retained_batch::RetainedMutation::Upsert {
                    id: "gamma.md".to_owned(),
                    text: "sharedtoken gamma supplied body".to_owned(),
                },
            ];
            for emergency in [false, true] {
                let mut blocked = runtime.clone();
                blocked.config.storage.storage_pressure_emergency = emergency;
                blocked.config.storage.disk_budget_bytes = Some(u64::from(!emergency));
                let mut input = blocked.cli_input.clone();
                input.command = CliCommand::Delete;
                input.delete_ids = vec!["alpha.md".to_owned()];
                let deletion = blocked.clone().with_cli_input(input.clone());
                let failures = [
                    blocked
                        .rebuild_retained_generation(&cx, &root)
                        .await
                        .unwrap_err(),
                    blocked
                        .apply_retained_batch(&cx, &root, &first, &mutations)
                        .await
                        .unwrap_err(),
                    deletion
                        .delete_retained_generation(&cx, &root)
                        .await
                        .unwrap_err(),
                ];
                for error in failures {
                    assert!(storage_admission::storage_admission_denial(&error).is_some());
                }

                // Resolve absent targets under the publication lease without
                // allocating a candidate or treating a no-op as a blocked write.
                input.delete_ids = vec!["absent.md".to_owned()];
                let no_delete = blocked.clone().with_cli_input(input);
                let (publication, deleted) = no_delete
                    .delete_retained_generation(&cx, &root)
                    .await
                    .unwrap();
                assert!(publication.is_none());
                assert_eq!(deleted, 0);
                for batch in [
                    Vec::new(),
                    vec![retained_batch::RetainedMutation::Delete {
                        id: "absent.md".to_owned(),
                    }],
                ] {
                    let result = blocked
                        .apply_retained_batch(&cx, &root, &first, &batch)
                        .await
                        .unwrap();
                    assert!(result.publication.is_none());
                    assert_eq!((result.upserted, result.deleted), (0, 0));
                }
                assert!(
                    blocked
                        .compact_retained_generation(&cx, &root)
                        .await
                        .unwrap()
                        .0
                        .is_none(),
                    "lexical-only compaction is a no-op"
                );
                cx.set_cancel_requested(true);
                let cancelled = blocked
                    .apply_retained_batch(&cx, &root, &first, &mutations)
                    .await;
                cx.set_cancel_requested(false);
                assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
                assert_eq!(store.active(&cx).unwrap(), Some(first.clone()));
                assert_eq!(
                    fs::read_dir(root.join("generations")).unwrap().count(),
                    allocations
                );
                assert_eq!(fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap(), bundle);
                let phases = old.search(&cx, "sharedtoken", 10).await.unwrap();
                assert_eq!(serde_json::to_value(&phases[0].hits).unwrap(), before);
            }

            // The source rebuild and original mixed batch both succeed after
            // relief, while a previously retained reader keeps its exact result.
            publish(&runtime, &cx, &root).await;
            let rebuilt = store.active(&cx).unwrap().unwrap();
            let result = runtime
                .apply_retained_batch(&cx, &root, &rebuilt, &mutations)
                .await
                .unwrap();
            assert_eq!((result.upserted, result.deleted), (1, 1));
            assert!(matches!(
                result.publication,
                Some(GenerationPublication::Durable(_))
            ));
            let mut fresh = runtime.open_retained_search(&cx, &root).await.unwrap();
            let phases = fresh.search(&cx, "sharedtoken", 10).await.unwrap();
            let mut paths = phases[0].hits.iter().map(|hit| hit.path.as_str()).collect::<Vec<_>>();
            paths.sort_unstable();
            assert_eq!(paths, ["beta.md", "gamma.md"]);
            let mut input = runtime.cli_input.clone();
            input.command = CliCommand::Delete;
            input.delete_ids = vec!["gamma.md".to_owned()];
            let (publication, deleted) = runtime
                .clone()
                .with_cli_input(input)
                .delete_retained_generation(&cx, &root)
                .await
                .unwrap();
            assert_eq!(deleted, 1);
            assert!(matches!(publication, Some(GenerationPublication::Durable(_))));
            let phases = old.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(serde_json::to_value(&phases[0].hits).unwrap(), before);
            assert_eq!(fs::read(first.path().join("FSFS-BUNDLE.json")).unwrap(), bundle);
        });
    }

    /// Each successor starts as a copy of its predecessor. Merge inputs that
    /// a predecessor retired must not ride along into every later generation
    /// (bd-2op1d: 5 edits grew a 111 MB store to 1.69 GB).
    #[test]
    fn successive_rebuilds_do_not_inherit_retired_lexical_segments() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            for round in 0..4 {
                fs::write(
                    source.join("alpha.md"),
                    format!("sharedtoken alpha revision {round}"),
                )
                .expect("edit source");
                fs::write(
                    source.join(format!("round-{round}.md")),
                    format!("sharedtoken round {round} document"),
                )
                .expect("add source");
                publish(&runtime, &cx, &root).await;
            }
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            let generation = store.active(&cx).expect("active").expect("published");
            let engine = FsfsRuntime::resolve_lexical_engine(generation.path())
                .expect("lexical layout")
                .engine_dir()
                .expect("Quill engine");
            let names = fs::read_dir(&engine)
                .expect("engine directory")
                .map(|entry| {
                    entry
                        .expect("engine entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>();
            assert!(
                !names.iter().any(|name| name.ends_with(".retired")),
                "{names:?}"
            );
            let snapshot = KeeperSnapshot::open(&engine, DEFAULT_SCHEMA).expect("snapshot");
            // Every publication is a short single-shard writer session. Each
            // used to open a fresh 65,536-docid lease that the bulk-finish
            // concat stored as DOCLEN/IDMAP holes (bd-k07zw); five sessions
            // over a handful of rows must stay inside the first block.
            let manifest = &snapshot.loaded_manifest().manifest;
            assert!(
                manifest.docid_high_watermark < frankensearch_quill::scribe::DOCID_LEASE_BLOCK,
                "watermark {} after five publications: {:?}",
                manifest.docid_high_watermark,
                manifest
                    .segments
                    .iter()
                    .map(|segment| (segment.docid_lo, segment.docid_hi))
                    .collect::<Vec<_>>()
            );
            let live = snapshot
                .segment_stats()
                .expect("segment stats")
                .sealed_segments;
            let segments = names
                .iter()
                .filter(|name| Path::new(name).extension().is_some_and(|ext| ext == "fslx"))
                .count();
            // MANIFEST.prev may still name one merge's inputs; nothing older.
            assert!(segments <= live + 2, "{live} live, {names:?}");
            let mut reader = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect("reader");
            let phases = reader.search(&cx, "sharedtoken", 10).await.expect("search");
            assert_eq!(phases.last().expect("phase").hits.len(), 5);
        });
    }

    /// A watch names the files it observed unchanged since the selected
    /// generation; the seeded rebuild carries them forward from its checkpoint
    /// instead of rereading them, and applies the edited, added and removed
    /// files (bd-dnqgr). One file is rewritten with its modification time
    /// restored and still named unchanged, as an edit that keeps every stamp
    /// and raises no notification would be: it keeps its old rows, which is
    /// how this test sees the carry. A file the first build skipped for its
    /// content stays skipped the same way. A full rebuild then rereads both.
    #[test]
    fn rebuild_carries_files_observed_unchanged_without_rereading_them() {
        run_test_with_cx(|cx| async move {
            for lexical_only in [false, true] {
                let directory = tempfile::tempdir().expect("fixture");
                let (runtime, source, root) = fixture(directory.path());
                let mut runtime = if lexical_only {
                    runtime.with_lexical_only_indexing()
                } else {
                    runtime
                };
                for (name, text) in [
                    ("bravo.md", "sharedtoken bravo document"),
                    ("charlie.md", "sharedtoken charlie document"),
                ] {
                    fs::write(source.join(name), text).expect("source");
                }
                // Read and skipped for its content (NUL bytes), not indexed.
                fs::write(source.join("echo.md"), b"\0\0\0binary echo\0\0\0").expect("binary");
                publish(&runtime, &cx, &root).await;

                fs::write(source.join("alpha.md"), "sharedtoken alpha zebra revision")
                    .expect("edit alpha");
                fs::remove_file(source.join("bravo.md")).expect("remove bravo");
                fs::write(source.join("delta.md"), "sharedtoken delta document")
                    .expect("add delta");
                let rewrite_keeping_mtime = |name: &str, text: &str| {
                    let path = fs::canonicalize(source.join(name)).expect("source path");
                    let modified = fs::metadata(&path)
                        .and_then(|metadata| metadata.modified())
                        .expect("mtime");
                    fs::write(&path, text).expect("rewrite");
                    fs::File::options()
                        .write(true)
                        .open(&path)
                        .and_then(|file| file.set_modified(modified))
                        .expect("restore mtime");
                    path
                };
                let unchanged = std::collections::BTreeSet::from([
                    rewrite_keeping_mtime("charlie.md", "sharedtoken charlie yak revision"),
                    rewrite_keeping_mtime("echo.md", "walrus echo text"),
                ]);
                publish(
                    &runtime.clone().with_unchanged_sources(unchanged),
                    &cx,
                    &root,
                )
                .await;

                // Keyword results are exact; the semantic lane would match
                // every document for any query.
                runtime.config.pressure.degradation_override =
                    crate::config::DegradationOverrideMode::ForceLexicalOnly;
                let mut reader = runtime
                    .open_retained_search(&cx, &root)
                    .await
                    .expect("reader");
                let mut hits = async |query: &str| {
                    let phases = reader.search(&cx, query, 10).await.expect("search");
                    let mut paths = phases
                        .last()
                        .expect("phase")
                        .hits
                        .iter()
                        .map(|hit| hit.path.clone())
                        .collect::<Vec<_>>();
                    paths.sort_unstable();
                    paths
                };
                assert_eq!(
                    hits("sharedtoken").await,
                    ["alpha.md", "charlie.md", "delta.md"]
                );
                assert_eq!(hits("zebra").await, ["alpha.md"]);
                assert_eq!(hits("charlie").await, ["charlie.md"]);
                assert!(
                    hits("yak").await.is_empty(),
                    "lexical_only={lexical_only}: charlie.md was reread instead of carried"
                );
                assert!(
                    hits("walrus").await.is_empty(),
                    "lexical_only={lexical_only}: echo.md was reread instead of skipped again"
                );
                drop(reader);
                let store = CompleteGenerationStore::open(&cx, &root).expect("store");
                let generation = store.active(&cx).expect("active").expect("published");
                if lexical_only {
                    assert!(FsfsRuntime::is_lexical_only_generation(generation.path()));
                } else {
                    let index = VectorIndex::open_read_only(
                        &generation.path().join(FSFS_VECTOR_INDEX_FILE),
                    )
                    .expect("fast tier");
                    let mut sources = index
                        .live_doc_ids()
                        .expect("live rows")
                        .iter()
                        .map(|row| semantic_windows::source_id(row).to_owned())
                        .collect::<Vec<_>>();
                    sources.sort_unstable();
                    sources.dedup();
                    assert_eq!(
                        sources,
                        ["alpha.md", "charlie.md", "delta.md"],
                        "the carried file keeps its vectors"
                    );
                }
                drop(generation);

                publish(&runtime, &cx, &root).await;
                let mut reader = runtime
                    .open_retained_search(&cx, &root)
                    .await
                    .expect("reader");
                for (query, path) in [("yak", "charlie.md"), ("walrus", "echo.md")] {
                    let phases = reader.search(&cx, query, 10).await.expect("search");
                    assert_eq!(
                        phases.last().expect("phase").hits[0].path,
                        path,
                        "a full rebuild rereads it"
                    );
                }
            }
        });
    }

    /// Without a watch observation a plain rebuild proves files unchanged by
    /// stamp or digest: over an unchanged source it carries every indexed file
    /// from the seeded predecessor, a lexical-only store included; an edit that
    /// restores its modification time is the only file reread; and `--full`
    /// seeds nothing and rereads everything.
    #[test]
    fn plain_rebuild_carries_files_whose_digest_still_matches() {
        run_test_with_cx(|cx| async move {
            for lexical_only in [false, true] {
                let directory = tempfile::tempdir().expect("fixture");
                let (runtime, source, root) = fixture(directory.path());
                let mut runtime = if lexical_only {
                    runtime.with_lexical_only_indexing()
                } else {
                    runtime
                };
                fs::write(source.join("bravo.md"), "sharedtoken bravo document").expect("source");
                let rebuild = async |runtime: &FsfsRuntime| {
                    runtime
                        .rebuild_retained_generation_with_precommit(&cx, &root, |_| Ok(()))
                        .await
                        .expect("rebuild")
                        .1
                };

                let first = rebuild(&runtime).await;
                assert_eq!(first.generation.indexed_files, 2);
                assert_eq!(first.unchanged_files, 0);
                assert_eq!(
                    rebuild(&runtime).await.unchanged_files,
                    2,
                    "lexical_only={lexical_only}: an unchanged source was reread"
                );
                // The rewrite restores its modification time, so only its change
                // time and digest tell.
                let bravo = source.join("bravo.md");
                let modified = fs::metadata(&bravo)
                    .and_then(|metadata| metadata.modified())
                    .expect("mtime");
                fs::write(&bravo, "sharedtoken bravo yak revision").expect("edit bravo");
                fs::File::options()
                    .write(true)
                    .open(&bravo)
                    .and_then(|file| file.set_modified(modified))
                    .expect("restore mtime");
                let edited = rebuild(&runtime).await;
                assert_eq!(edited.generation.indexed_files, 2);
                assert_eq!(edited.unchanged_files, 1, "lexical_only={lexical_only}");

                runtime.config.pressure.degradation_override =
                    crate::config::DegradationOverrideMode::ForceLexicalOnly;
                let mut reader = runtime
                    .open_retained_search(&cx, &root)
                    .await
                    .expect("reader");
                for (query, path) in [("yak", "bravo.md"), ("alpha", "alpha.md")] {
                    let phases = reader.search(&cx, query, 10).await.expect("search");
                    let hits = &phases.last().expect("phase").hits;
                    assert!(
                        hits.len() == 1 && hits[0].path == path,
                        "lexical_only={lexical_only}: {query}: {hits:?}"
                    );
                }
                drop(reader);

                let mut full = runtime.clone();
                full.cli_input.full_reindex = true;
                assert_eq!(rebuild(&full).await.unchanged_files, 0, "--full rereads");
            }
        });
    }

    /// The first complete build in a directory that holds a legacy index
    /// converts it from the legacy completion receipt: an unchanged file is
    /// carried with its rows and vectors, an edited one is reread, and the
    /// converted store answers the same searches.
    #[test]
    fn converting_a_legacy_index_carries_its_unchanged_files() {
        run_test_with_cx(|cx| async move {
            for lexical_only in [false, true] {
                let directory = tempfile::tempdir().expect("fixture");
                let (runtime, source, root) = fixture(directory.path());
                let mut runtime = if lexical_only {
                    runtime.with_lexical_only_indexing()
                } else {
                    runtime
                };
                fs::write(source.join("bravo.md"), "sharedtoken bravo document").expect("source");
                let legacy = Box::pin(runtime.run_one_shot_index_scaffold_internal(
                    &cx,
                    CliCommand::Index,
                    |_| Ok(()),
                    false,
                    true,
                ))
                .await
                .expect("legacy index");
                assert_eq!(legacy.generation.indexed_files, 2);
                assert!(root.join(FSFS_SENTINEL_FILE).is_file(), "a legacy index");

                fs::write(source.join("bravo.md"), "sharedtoken bravo yak revision")
                    .expect("edit bravo");
                let (publication, converted) = runtime
                    .rebuild_retained_generation_with_precommit(&cx, &root, |_| Ok(()))
                    .await
                    .expect("convert");
                assert!(matches!(publication, GenerationPublication::Durable(_)));
                assert_eq!(converted.generation.indexed_files, 2);
                assert_eq!(
                    converted.unchanged_files, 1,
                    "lexical_only={lexical_only}: alpha.md was rebuilt, not carried from the legacy index"
                );

                runtime.config.pressure.degradation_override =
                    crate::config::DegradationOverrideMode::ForceLexicalOnly;
                let mut reader = runtime
                    .open_retained_search(&cx, &root)
                    .await
                    .expect("reader");
                for (query, path) in [("yak", "bravo.md"), ("alpha", "alpha.md")] {
                    let phases = reader.search(&cx, query, 10).await.expect("search");
                    let hits = &phases.last().expect("phase").hits;
                    assert!(
                        hits.len() == 1 && hits[0].path == path,
                        "lexical_only={lexical_only}: {query}: {hits:?}"
                    );
                }
                drop(reader);
                if !lexical_only {
                    let store = CompleteGenerationStore::open(&cx, &root).expect("store");
                    let generation = store.active(&cx).expect("active").expect("published");
                    let index = VectorIndex::open_read_only(
                        &generation.path().join(FSFS_VECTOR_INDEX_FILE),
                    )
                    .expect("fast tier");
                    let mut sources = index
                        .live_doc_ids()
                        .expect("live rows")
                        .iter()
                        .map(|row| semantic_windows::source_id(row).to_owned())
                        .collect::<Vec<_>>();
                    sources.sort_unstable();
                    sources.dedup();
                    assert_eq!(sources, ["alpha.md", "bravo.md"]);
                }
            }
        });
    }

    #[test]
    fn retained_reader_survives_rebuild_and_search_does_not_change_inventory() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            let mut old = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect("old reader");
            let old_generation = old.generation().clone();
            let before = old
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("first query");
            assert_eq!(before.last().expect("initial phase").hits.len(), 1);
            assert_eq!(
                store.active(&cx).expect("query left bundle intact"),
                Some(old_generation.clone())
            );

            fs::write(source.join("beta.md"), "sharedtoken beta new document").expect("new source");
            publish(&runtime, &cx, &root).await;
            let mut current = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect("new reader");
            assert_ne!(current.generation().id(), old_generation.id());
            let retained = old
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("retained query");
            let fresh = current
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("fresh query");
            assert_eq!(retained.last().expect("retained phase").hits.len(), 1);
            assert_eq!(fresh.last().expect("fresh phase").hits.len(), 2);
            assert_eq!(old.generation(), &old_generation);
            assert!(old_generation.path().exists());
            assert_eq!(
                store.active(&cx).expect("both queries leave bundle intact"),
                Some(current.generation().clone())
            );
        });
    }

    #[test]
    fn retained_phase_sink_failure_is_propagated_without_persisting_results() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut reader = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect("reader");
            let mut delivered = 0;
            let mut sink = |_: &SearchPayload| {
                delivered += 1;
                Err(SearchError::InvalidConfig {
                    field: "test.retained_sink".to_owned(),
                    value: "stop".to_owned(),
                    reason: "injected sink failure".to_owned(),
                })
            };
            let error = reader
                .search_with_phase_sink(&cx, "sharedtoken", 10, &mut sink)
                .await
                .expect_err("sink refusal");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "test.retained_sink")
            );
            assert_eq!(delivered, 1);
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            assert_eq!(
                store.active(&cx).expect("failed sink left bundle intact"),
                Some(reader.generation().clone())
            );
            assert!(
                FsfsRuntime::load_complete_explain_session_at_root(reader.generation().path())
                    .unwrap()
                    .is_none()
            );
            reader
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("reader remains usable");
        });
    }

    #[test]
    fn retained_reader_rejects_absent_and_corrupt_selection_without_fallback() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            CompleteGenerationStore::create(&cx, &root).expect("empty store");
            let error = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect_err("no selection");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "complete_generation.selection")
            );
            publish(&runtime, &cx, &root).await;
            fs::write(
                root.join(crate::generation_store::COMPLETE_GENERATION_POINTER),
                b"broken pointer",
            )
            .expect("inject corruption");
            runtime
                .open_retained_search(&cx, &root)
                .await
                .expect_err("no legacy fallback");
        });
    }

    #[test]
    fn cancelled_retained_search_delivers_no_phase_and_leaves_selection_intact() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut reader = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect("reader");
            cx.set_cancel_requested(true);
            let mut delivered = 0;
            let mut sink = |_: &SearchPayload| {
                delivered += 1;
                Ok(())
            };
            let error = reader
                .search_with_phase_sink(&cx, "sharedtoken", 10, &mut sink)
                .await
                .expect_err("cancelled query");
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(delivered, 0);
            cx.set_cancel_requested(false);
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            assert_eq!(
                store.active(&cx).expect("cancel left bundle intact"),
                Some(reader.generation().clone())
            );
        });
    }

    #[test]
    fn retained_catalog_refusal_does_not_create_a_store_or_touch_external_data() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            let external = directory.path().join("catalog.sqlite");
            fs::write(&external, b"unrelated catalog sentinel").expect("external data");
            let mut config = runtime.config().clone();
            config.storage.db_path = "{index_dir}/../../../catalog.sqlite".to_owned();
            let invalid = FsfsRuntime::new(config).with_cli_input(runtime.cli_input.clone());

            let error = invalid
                .rebuild_retained_generation(&cx, &root)
                .await
                .expect_err("escaping rebuild catalog");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "storage.db_path")
            );
            let error = invalid
                .open_retained_search(&cx, &root)
                .await
                .expect_err("escaping reader catalog");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "storage.db_path")
            );
            assert!(!root.exists(), "preflight refusal must not create a store");
            assert_eq!(
                fs::read(external).expect("external data"),
                b"unrelated catalog sentinel"
            );
        });
    }

    #[test]
    fn retained_reader_refuses_shadow_artifact_writes_without_changing_publication() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            let selected = store.active(&cx).expect("published generation");
            let mut config = runtime.config().clone();
            config.search.shadow_mode = true;
            let invalid = FsfsRuntime::new(config).with_cli_input(runtime.cli_input.clone());

            let error = invalid
                .open_retained_search(&cx, &root)
                .await
                .expect_err("shadow observation must not write into a retained generation");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "search.shadow_mode")
            );
            assert_eq!(
                store.active(&cx).expect("sealed inventory remains valid"),
                selected
            );
        });
    }

    #[test]
    fn retained_reader_rejects_a_foreign_catalog_without_changing_publication() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            let selected = store.active(&cx).expect("published generation");
            let external = directory.path().join("foreign.sqlite");
            fs::write(&external, b"foreign catalog sentinel").expect("external data");
            let mut config = runtime.config().clone();
            config.storage.db_path = external.display().to_string();
            let invalid = FsfsRuntime::new(config).with_cli_input(runtime.cli_input.clone());

            let error = invalid
                .open_retained_search(&cx, &root)
                .await
                .expect_err("foreign catalog must not hydrate retained results");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "storage.db_path")
            );
            assert_eq!(
                store.active(&cx).expect("inventory remains valid"),
                selected
            );
            assert_eq!(
                fs::read(external).expect("external data"),
                b"foreign catalog sentinel"
            );
        });
    }

    #[test]
    fn live_retained_search_observes_add_delete_and_rename_without_retargeting_old_reader() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut live = runtime
                .open_live_retained_search(&cx, &root)
                .await
                .expect("live reader");
            let mut pinned = runtime
                .open_retained_search(&cx, &root)
                .await
                .expect("pinned reader");
            let first = live.generation().clone();
            assert!(!live.refresh(&cx).await.expect("unchanged selection"));

            fs::write(source.join("beta.md"), "sharedtoken beta new document").expect("new source");
            publish(&runtime, &cx, &root).await;
            let added = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("live addition");
            assert_eq!(added.last().expect("phase").hits.len(), 2);
            assert_ne!(live.generation(), &first);
            let old = pinned
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("pinned query");
            assert_eq!(old.last().expect("phase").hits.len(), 1);
            assert_eq!(pinned.generation(), &first);

            fs::rename(source.join("alpha.md"), source.join("renamed.md")).expect("rename source");
            fs::remove_file(source.join("beta.md")).expect("delete fixture source");
            publish(&runtime, &cx, &root).await;
            let renamed = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("live rename and delete");
            let hits = &renamed.last().expect("phase").hits;
            assert_eq!(hits.len(), 1);
            assert!(hits[0].path.ends_with("renamed.md"));
            assert!(!live.refresh(&cx).await.expect("reuse admitted resources"));
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            assert_eq!(
                store.active(&cx).expect("sealed inventory"),
                Some(live.generation().clone())
            );
        });
    }

    #[test]
    fn live_retained_root_replacement_refuses_a_foreign_store_and_preserves_the_reader() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut live = runtime
                .open_live_retained_search(&cx, &root)
                .await
                .expect("live reader");
            let first = live.generation().clone();

            let foreign_directory = tempfile::tempdir().expect("foreign fixture");
            let (foreign_runtime, foreign_source, foreign_root) = fixture(foreign_directory.path());
            fs::write(
                foreign_source.join("alpha.md"),
                "foreigntoken unrelated content",
            )
            .expect("foreign source");
            publish(&foreign_runtime, &cx, &foreign_root).await;
            let foreign_pointer =
                fs::read(foreign_root.join(crate::generation_store::COMPLETE_GENERATION_POINTER))
                    .expect("foreign selection");
            let moved = directory.path().join("retained-original-store");
            fs::rename(&root, &moved).expect("retain original store at another path");
            fs::rename(&foreign_root, &root).expect("replace the store pathname");

            let mut delivered = 0;
            let mut sink = |_: &SearchPayload| {
                delivered += 1;
                Ok(())
            };
            let error = live
                .search_with_phase_sink(&cx, "foreigntoken", 10, &mut sink)
                .await
                .expect_err("a foreign directory cannot become the live reader's authority");
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                    if field == "complete_generation.root"));
            assert_eq!(delivered, 0);
            assert_eq!(live.generation(), &first);
            assert_eq!(
                fs::read(root.join(crate::generation_store::COMPLETE_GENERATION_POINTER))
                    .expect("foreign store remains unchanged"),
                foreign_pointer
            );

            fs::rename(&root, &foreign_root).expect("retain foreign store at its original path");
            fs::rename(&moved, &root).expect("restore original authority pathname");
            assert!(
                !live
                    .refresh(&cx)
                    .await
                    .expect("original reader is still installed")
            );
            let original = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("original query");
            assert_eq!(original.last().expect("phase").hits.len(), 1);
            assert_eq!(live.generation(), &first);

            fs::write(source.join("beta.md"), "sharedtoken legitimate successor")
                .expect("new original source");
            publish(&runtime, &cx, &root).await;
            let successor = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("normal successor");
            assert_eq!(successor.last().expect("phase").hits.len(), 2);
            assert_ne!(live.generation(), &first);
        });
    }

    #[test]
    fn live_retained_root_replacement_refuses_copied_selection_and_symlink_routes() {
        run_test_with_cx(|cx| async move {
            for symlink in [false, true] {
                let directory = tempfile::tempdir().expect("fixture");
                let (runtime, _, root) = fixture(directory.path());
                publish(&runtime, &cx, &root).await;
                let mut live = runtime
                    .open_live_retained_search(&cx, &root)
                    .await
                    .expect("live reader");
                let first = live.generation().clone();
                let moved = directory.path().join("retained-original-store");
                fs::rename(&root, &moved).expect("retain original directory");
                if symlink {
                    std::os::unix::fs::symlink(&moved, &root).expect("replace root by alias");
                } else {
                    fs::create_dir(&root).expect("replacement root");
                    fs::copy(
                        moved.join(crate::generation_store::COMPLETE_GENERATION_POINTER),
                        root.join(crate::generation_store::COMPLETE_GENERATION_POINTER),
                    )
                    .expect("copy the exact selection without its original root");
                }
                let error = live
                    .refresh(&cx)
                    .await
                    .expect_err("unchanged selection bytes do not establish root identity");
                assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                        if field == "complete_generation.root"));
                assert_eq!(live.generation(), &first);

                fs::rename(&root, directory.path().join("rejected-replacement"))
                    .expect("retain rejected replacement");
                fs::rename(&moved, &root).expect("restore original directory");
                let mut retained = live.into_retained();
                let phases = retained
                    .search(&cx, "sharedtoken", 10)
                    .await
                    .expect("retained query");
                assert_eq!(phases.last().expect("phase").hits.len(), 1);
                assert_eq!(retained.generation(), &first);
            }
        });
    }

    #[test]
    fn failed_live_admission_emits_nothing_preserves_old_reader_and_can_retry() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut live = runtime
                .open_live_retained_search(&cx, &root)
                .await
                .expect("live reader");
            let first = live.generation().clone();
            fs::write(source.join("beta.md"), "sharedtoken beta new document").expect("new source");
            publish(&runtime, &cx, &root).await;
            let store = CompleteGenerationStore::open(&cx, &root).expect("store");
            let second = store.active(&cx).expect("selection").expect("generation");
            let manifest = second
                .path()
                .join(crate::generation_store::COMPLETE_GENERATION_MANIFEST);
            let original = fs::read(&manifest).expect("save inventory");
            fs::write(&manifest, "corrupt inventory").expect("inject corruption");
            let mut delivered = 0;
            let mut sink = |_: &SearchPayload| {
                delivered += 1;
                Ok(())
            };
            live.search_with_phase_sink(&cx, "sharedtoken", 10, &mut sink)
                .await
                .expect_err("corrupt successor must not yield stale results");
            assert_eq!(delivered, 0);
            assert_eq!(live.generation(), &first);
            let old = live
                .reader
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("old handle intact");
            assert_eq!(old.last().expect("phase").hits.len(), 1);

            fs::write(manifest, original).expect("restore exact inventory");
            let fresh = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("retry repaired selection");
            assert_eq!(fresh.last().expect("phase").hits.len(), 2);
            assert_eq!(live.generation(), &second);
        });
    }

    #[test]
    fn live_query_pins_its_generation_when_publication_changes_inside_phase_sink() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let pointer = root.join(crate::generation_store::COMPLETE_GENERATION_POINTER);
            let first_pointer = fs::read(&pointer).expect("first pointer");
            fs::write(source.join("beta.md"), "sharedtoken beta new document").expect("new source");
            publish(&runtime, &cx, &root).await;
            let second_pointer = fs::read(&pointer).expect("second pointer");
            // Both bundles were built and admitted by the real indexing path.
            // Replay their genuine descriptors to control the exact rename
            // boundary synchronously inside the production phase callback.
            let switch = root.join("test-pointer-switch");
            fs::write(&switch, first_pointer).expect("stage first pointer");
            fs::rename(&switch, &pointer).expect("select first bundle");
            let mut live = runtime
                .open_live_retained_search(&cx, &root)
                .await
                .expect("live reader");
            let first = live.generation().clone();
            fs::write(&switch, second_pointer).expect("stage successor pointer");
            let mut switched = false;
            let mut sink = |phase: &SearchPayload| -> SearchResult<()> {
                assert_eq!(phase.hits.len(), 1);
                assert!(phase.hits[0].path.ends_with("alpha.md"));
                if !switched {
                    fs::rename(&switch, &pointer)?;
                    switched = true;
                }
                Ok(())
            };
            let phases = live
                .search_with_phase_sink(&cx, "sharedtoken", 10, &mut sink)
                .await
                .expect("admitted query completes on original bundle");
            assert!(switched);
            assert!(phases.iter().all(|phase| phase.hits.len() == 1));
            assert_eq!(live.generation(), &first);
            let fresh = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("next query sees successor");
            assert_eq!(fresh.last().expect("phase").hits.len(), 2);
            assert_ne!(live.generation(), &first);
        });
    }

    #[test]
    fn cancelled_live_refresh_keeps_the_old_reader_and_emits_no_phase() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, source, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut live = runtime
                .open_live_retained_search(&cx, &root)
                .await
                .expect("live reader");
            let first = live.generation().clone();
            fs::write(source.join("beta.md"), "sharedtoken beta new document").expect("new source");
            publish(&runtime, &cx, &root).await;
            cx.set_cancel_requested(true);
            let mut delivered = 0;
            let mut sink = |_: &SearchPayload| {
                delivered += 1;
                Ok(())
            };
            let error = live
                .search_with_phase_sink(&cx, "sharedtoken", 10, &mut sink)
                .await
                .expect_err("cancelled refresh");
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(delivered, 0);
            assert_eq!(live.generation(), &first);
            cx.set_cancel_requested(false);
            let fresh = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("retry after cancellation");
            assert_eq!(fresh.last().expect("phase").hits.len(), 2);
        });
    }

    #[test]
    fn missing_live_selection_fails_closed_and_retained_opt_out_is_explicit() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().expect("fixture");
            let (runtime, _, root) = fixture(directory.path());
            publish(&runtime, &cx, &root).await;
            let mut live = runtime
                .open_live_retained_search(&cx, &root)
                .await
                .expect("live reader");
            let first = live.generation().clone();
            fs::rename(
                root.join(crate::generation_store::COMPLETE_GENERATION_POINTER),
                root.join("saved-pointer"),
            )
            .expect("remove selection without removing its bundle");
            let error = live
                .search(&cx, "sharedtoken", 10)
                .await
                .expect_err("missing selection");
            assert!(
                matches!(error, SearchError::InvalidConfig { field, .. } if field == "complete_generation.selection")
            );
            assert_eq!(live.generation(), &first);
            let mut pinned = live.into_retained();
            let result = pinned
                .search(&cx, "sharedtoken", 10)
                .await
                .expect("explicit pinned read");
            assert_eq!(result.last().expect("phase").hits.len(), 1);
            assert_eq!(pinned.generation(), &first);
        });
    }
}

#[path = "runtime/complete_cli.rs"]
mod complete_cli;

#[cfg(unix)]
#[path = "runtime/complete_watch.rs"]
mod complete_watch;

#[path = "runtime/retained_reuse.rs"]
mod retained_reuse;
pub use retained_reuse::RetainedRebuildCheckpoint;

#[path = "runtime/semantic_windows.rs"]
mod semantic_windows;
