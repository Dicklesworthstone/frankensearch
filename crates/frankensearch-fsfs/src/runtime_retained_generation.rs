// Compose the existing runtime with retained-generation rebuilding in the same
// module, so its private admission and resource-opening paths remain the single
// source of truth. The existing runtime source is left byte-for-byte intact.
include!("runtime.rs");

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
        let artifacts = Box::pin(
            self.runtime
                .execute_search_phase_artifacts_with_mode_using_resources(
                    cx,
                    query,
                    limit,
                    SearchExecutionMode::Full,
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
#[derive(Debug)]
pub struct LiveRetainedSearchReader {
    store: crate::generation_store::CompleteGenerationStore,
    reader: RetainedSearchReader,
}

impl LiveRetainedSearchReader {
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
    /// Returns the original selection, integrity, producer or cancellation error.
    pub async fn refresh(&mut self, cx: &Cx) -> SearchResult<bool> {
        retained_search_checkpoint(cx)?;
        if self.store.is_selected(cx, self.reader.generation())? {
            return Ok(false);
        }
        let replacement = self
            .reader
            .runtime
            .open_retained_search(cx, self.store.root())
            .await?;
        retained_search_checkpoint(cx)?;
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

impl FsfsRuntime {
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
        let store = crate::generation_store::CompleteGenerationStore::open(cx, store_root)?;
        let reader = self.open_retained_search(cx, store.root()).await?;
        Ok(LiveRetainedSearchReader { store, reader })
    }

    /// Resolve and open the selected complete generation once for repeated reads.
    ///
    /// Bundle hashes are verified before the ordinary full-search admission and
    /// resource opens. Missing or corrupt selection is an error, never an excuse
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
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let generation = store
            .active(cx)?
            .ok_or_else(|| SearchError::InvalidConfig {
                field: "complete_generation.selection".to_owned(),
                value: store.root().display().to_string(),
                reason: "no complete generation has been published; rebuild the store first"
                    .to_owned(),
            })?;
        let mut runtime = self.clone();
        runtime.cli_input.index_dir = Some(generation.path().to_path_buf());
        // Retained readers never start a daemon or write a socket/cache under
        // an immutable generation, even when their caller originated in a CLI.
        runtime.cli_input.daemon = false;
        let resources = Box::pin(
            runtime.prepare_search_execution_resources_at_root_with_modes(
                cx,
                generation.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
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
    }

    // Source authority is checked after sealing, without replacing the real
    // indexing, producer admission, lease or predecessor checks below.
    #[allow(clippy::future_not_send)]
    async fn rebuild_retained_generation_with_precommit<F>(
        &self,
        cx: &Cx,
        store_root: &Path,
        precommit: F,
    ) -> SearchResult<crate::generation_store::GenerationPublication>
    where
        F: FnOnce(&Cx) -> SearchResult<()> + Send,
    {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        let target_root = fs::canonicalize(self.resolve_target_root()?)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let store = CompleteGenerationStore::create(cx, store_root)?;
        if store.root().starts_with(&target_root) || target_root.starts_with(store.root()) {
            return Err(SearchError::InvalidConfig {
                field: "complete_generation.store_root".to_owned(),
                value: store.root().display().to_string(),
                reason: "source and generation-store trees must not overlap".to_owned(),
            });
        }
        let build = store.begin(cx)?;
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
        Box::pin(candidate.run_retained_index_with_reuse(cx, &store, build.path())).await?;

        // An indexing command can return success with deferred semantic rows.
        // Apply the same full-generation admission used by actual search, then
        // open its real lexical/vector/producer resources before sealing.
        Self::validate_search_generation_at_root(build.path(), SearchExecutionMode::Full)?;
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        drop(resources);
        build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, SearchExecutionMode::Full),
            precommit,
        )
    }

    /// Insert or replace the supplied JSONL documents in a complete successor.
    ///
    /// Duplicate IDs use their last input body and count once. Every present
    /// vector tier must admit its real producer before embedding; missing or
    /// incompatible quality support refuses the whole batch. Bounded canonical
    /// input drives vectors and catalog identity; complete lexical text remains
    /// searchable, including content beyond the models' input budget.
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

    #[allow(clippy::future_not_send, clippy::too_many_lines)]
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
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let documents = retained_reuse::read_append_documents(cx, self).await?;
        if documents.is_empty() {
            return Ok((None, 0));
        }
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let build = store.begin(cx)?;
        let predecessor = store.active(cx)?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "selection",
                "no complete generation has been published",
            )
        })?;
        Self::validate_search_generation_at_root(predecessor.path(), SearchExecutionMode::Full)?;
        let mut manifests = Self::read_matching_manifest_generation(predecessor.path())?
            .ok_or_else(|| {
                complete_cli::complete_cli_error(
                    "append_membership",
                    "selected membership manifests disagree",
                )
            })?;

        let window_maximum = Self::fast_window_policy_at_root(predecessor.path())?;
        let mut window_plans = BTreeMap::new();
        if window_maximum > 1 {
            for (id, document) in &documents {
                semantic_windows::validate_source_id(id)?;
                let plan = semantic_windows::plan(&document.lexical_text, window_maximum)?;
                if plan
                    .row_ids(id)
                    .iter()
                    .any(|row| row.len() > usize::from(u16::MAX))
                {
                    return Err(complete_cli::complete_cli_error(
                        "append_id",
                        "document ID leaves insufficient room for its semantic window row IDs",
                    ));
                }
                window_plans.insert(id.clone(), plan);
            }
        }

        // Use the same producer-resolution and admission checks as the ordinary
        // appender. Do not borrow a different tier's identity or manufacture
        // vectors from copied producer labels. No mutable mapping spans inference.
        let fast_embedder = self.resolve_fast_embedder()?;
        let fast_identity = {
            let index =
                VectorIndex::open_read_only(&predecessor.path().join(FSFS_VECTOR_INDEX_FILE))?;
            semantic_windows::load_mapping(predecessor.path(), &index)?;
            Self::admit_vector_generation_for_embedder(&index, fast_embedder.as_ref())?;
            let identity = fast_embedder.identity()?.clone();
            identity.validate()?;
            if index.embedder_revision() != identity.fingerprint() {
                return Err(SearchError::UnverifiableRemoteSpace {
                    producer: "fsfs.append_batch.fast".to_owned(),
                    reason: "the fast producer identity changed during admission".to_owned(),
                });
            }
            identity
        };
        let quality = if predecessor
            .path()
            .join(FSFS_VECTOR_QUALITY_INDEX_FILE)
            .exists()
        {
            let embedder = self.resolve_quality_embedder()?.ok_or_else(|| {
                SearchError::EmbedderUnavailable {
                    model: self.config.indexing.quality_model.clone(),
                    reason: "the selected generation has a quality tier but no verified quality producer is available to extend it"
                        .to_owned(),
                }
            })?;
            let index = VectorIndex::open_read_only(
                &predecessor.path().join(FSFS_VECTOR_QUALITY_INDEX_FILE),
            )?;
            Self::admit_quality_generation_for_embedder(&index, embedder.as_ref())?;
            let identity = embedder.identity()?.clone();
            identity.validate()?;
            if index.embedder_revision() != identity.fingerprint() {
                return Err(SearchError::UnverifiableRemoteSpace {
                    producer: "fsfs.append_batch.quality".to_owned(),
                    reason: "the quality producer identity changed during admission".to_owned(),
                });
            }
            Some((embedder, identity))
        } else {
            None
        };
        let mut fast_entries = Vec::with_capacity(documents.len());
        let mut quality_entries = Vec::new();
        for (id, document) in &documents {
            let texts = match window_plans.get(id) {
                Some(plan) => plan.texts(&document.lexical_text)?,
                None => vec![document.embedding_text.as_str()],
            };
            for (ordinal, text) in texts.into_iter().enumerate() {
                retained_search_checkpoint(cx)?;
                let response = fast_embedder.embed_bound(cx, text).await;
                retained_search_checkpoint(cx)?;
                let embedding = response?;
                embedding.validate()?;
                if embedding.identity != fast_identity {
                    return Err(SearchError::UnverifiableRemoteSpace {
                        producer: "fsfs.append_batch.fast".to_owned(),
                        reason:
                            "the returned fast embedding does not carry the admitted producer identity"
                                .to_owned(),
                    });
                }
                fast_entries.push((semantic_windows::row_id(id, ordinal), embedding.values));
            }
            if let Some((embedder, identity)) = quality.as_ref() {
                retained_search_checkpoint(cx)?;
                let response = embedder.embed_bound(cx, &document.embedding_text).await;
                retained_search_checkpoint(cx)?;
                let embedding = response?;
                embedding.validate()?;
                if &embedding.identity != identity {
                    return Err(SearchError::UnverifiableRemoteSpace {
                        producer: "fsfs.append_batch.quality".to_owned(),
                        reason: "the returned quality embedding does not carry the admitted producer identity"
                            .to_owned(),
                    });
                }
                quality_entries.push((id.clone(), embedding.values));
            }
        }
        retained_search_checkpoint(cx)?;
        let mut input = self.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        input.daemon = false;
        input.daemon_socket = None;
        let candidate = self.clone().with_cli_input(input);
        if retained_reuse::copy_selected_generation(cx, &candidate, &store, build.path())?
            != predecessor
        {
            return Err(complete_cli::complete_cli_error(
                "append_selection",
                "selected predecessor changed while preparing the batch",
            ));
        }
        let candidate_lease = crate::lifecycle::PublicationLease::acquire(build.path())?;
        let timestamp = pressure_timestamp_ms();
        candidate_lease.fence("complete-generation append lexical mutation")?;
        let lexical_mutations = documents
            .iter()
            .map(|(id, document)| {
                LexicalMutation::upsert(
                    id.clone(),
                    timestamp,
                    IngestionClass::FullSemanticLexical,
                    document.lexical_text.clone(),
                    "append_batch",
                )
            })
            .collect::<Vec<_>>();
        candidate
            .apply_one_shot_lexical_mutations(cx, build.path(), &lexical_mutations)
            .await?;
        for (relative, embedder, entries) in [
            (FSFS_VECTOR_INDEX_FILE, Some(&fast_embedder), &fast_entries),
            (
                FSFS_VECTOR_QUALITY_INDEX_FILE,
                quality.as_ref().map(|(embedder, _)| embedder),
                &quality_entries,
            ),
        ] {
            let Some(embedder) = embedder else {
                continue;
            };
            retained_search_checkpoint(cx)?;
            candidate_lease.fence("complete-generation append vector mutation")?;
            let mut index = Self::open_vector_index_for_mutation(&build.path().join(relative))?;
            if relative == FSFS_VECTOR_INDEX_FILE {
                Self::admit_vector_generation_for_embedder(&index, embedder.as_ref())?;
            } else {
                Self::admit_quality_generation_for_embedder(&index, embedder.as_ref())?;
            }
            // append_batch logs replacement before superseding an older row.
            // Freeze both tiers without pending WALs before sealing the bundle.
            index.append_batch(entries)?;
            if relative == FSFS_VECTOR_INDEX_FILE && window_maximum > 1 {
                // Replacing a document with fewer windows must retire every
                // old trailing row in this private candidate before sealing.
                let new_rows = entries
                    .iter()
                    .map(|(id, _)| id.as_str())
                    .collect::<HashSet<_>>();
                let stale = index
                    .live_doc_ids()?
                    .into_iter()
                    .filter(|row| {
                        documents.contains_key(semantic_windows::source_id(row))
                            && !new_rows.contains(row.as_str())
                    })
                    .collect::<Vec<_>>();
                index.soft_delete_batch(&stale.iter().map(String::as_str).collect::<Vec<_>>())?;
            }
            index.compact()?;
            index.vacuum()?;
        }
        retained_search_checkpoint(cx)?;
        candidate_lease.fence("complete-generation append metadata mutation")?;
        let revision = i64::try_from(timestamp).unwrap_or(i64::MAX);
        let catalog_path = candidate.resolve_storage_db_path()?;
        if catalog_path.exists() {
            let storage = Storage::open(PipelineStorageConfig {
                db_path: catalog_path,
                ..PipelineStorageConfig::default()
            })?;
            for (id, document) in &documents {
                retained_search_checkpoint(cx)?;
                let text = &document.embedding_text;
                let created_at = storage
                    .get_document(id)?
                    .map_or(revision, |document| document.created_at);
                storage.upsert_document(&frankensearch_storage::DocumentRecord::new(
                    id,
                    text.chars().take(400).collect::<String>(),
                    frankensearch_storage::ContentHasher::hash(text),
                    text.chars().count(),
                    created_at,
                    revision.max(created_at),
                ))?;
                storage.mark_embedded(id, fast_embedder.id())?;
                if let Some((embedder, _)) = quality.as_ref() {
                    storage.mark_embedded(id, embedder.id())?;
                }
            }
        }
        for (id, document) in &documents {
            manifests.insert(
                id.clone(),
                IndexManifestEntry {
                    file_key: id.clone(),
                    revision,
                    ingestion_class: ingestion_class_label(IngestionClass::FullSemanticLexical)
                        .to_owned(),
                    canonical_bytes: u64::try_from(document.embedding_text.len())
                        .unwrap_or(u64::MAX),
                    reason_code: "append_batch".to_owned(),
                    fast_windows: window_plans.remove(id),
                },
            );
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
                "append_membership",
                "candidate has no completion sentinel",
            )
        })?;
        "append-batch".clone_into(&mut sentinel.command);
        sentinel.generated_at_ms = timestamp;
        sentinel.indexed_files = manifests.len();
        sentinel.discovered_files = sentinel.discovered_files.max(manifests.len());
        sentinel.skipped_files = sentinel.discovered_files.saturating_sub(manifests.len());
        sentinel.total_canonical_bytes = manifests.iter().fold(0_u64, |total, entry| {
            total.saturating_add(entry.canonical_bytes)
        });
        sentinel.source_hash_hex = index_source_hash_hex(&manifests);
        for reason in protect_vector_generations(build.path(), "complete-generation append") {
            if !sentinel.reason_codes.contains(&reason) {
                sentinel.reason_codes.push(reason);
            }
        }
        candidate.write_index_sentinel(build.path(), &sentinel)?;
        candidate_lease.fence("complete-generation append candidate complete")?;
        drop(candidate_lease);
        retained_search_checkpoint(cx)?;
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        drop(resources);
        let publication = build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, SearchExecutionMode::Full),
            precommit,
        )?;
        Ok((Some(publication), documents.len()))
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
        let build = store.begin(cx)?;
        let predecessor = store.active(cx)?.ok_or_else(|| {
            complete_cli::complete_cli_error(
                "selection",
                "no complete generation has been published",
            )
        })?;
        Self::validate_search_generation_at_root(predecessor.path(), SearchExecutionMode::Full)?;
        let mut manifests = Self::read_matching_manifest_generation(predecessor.path())?
            .ok_or_else(|| {
                complete_cli::complete_cli_error(
                    "delete_membership",
                    "selected membership manifests disagree",
                )
            })?;
        // Use the union, not just the fast vector tier: lexical-only documents
        // and partial quality coverage still belong to the same command.
        let mut live_ids = manifests.keys().cloned().collect::<BTreeSet<_>>();
        for (_, relative) in FSFS_VECTOR_GENERATION_FILES {
            let path = predecessor.path().join(relative);
            if path.exists() {
                let index = VectorIndex::open_read_only(&path)?;
                if relative == FSFS_VECTOR_INDEX_FILE {
                    semantic_windows::load_mapping(predecessor.path(), &index)?;
                }
                live_ids.extend(
                    index
                        .live_doc_ids()?
                        .into_iter()
                        .map(|row| semantic_windows::source_id(&row).to_owned()),
                );
            }
        }
        let targets = live_ids
            .into_iter()
            .filter(|id| {
                self.cli_input.delete_ids.iter().any(|requested| {
                    if self.cli_input.delete_prefix {
                        id.starts_with(requested.as_str())
                    } else {
                        id == requested
                    }
                })
            })
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return Ok((None, 0));
        }
        retained_search_checkpoint(cx)?;
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
        candidate
            .apply_one_shot_lexical_mutations(cx, build.path(), &lexical_mutations)
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
        retained_search_checkpoint(cx)?;
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        drop(resources);
        let publication = build.publish_with_precommit(
            cx,
            |_, path| Self::validate_search_generation_at_root(path, SearchExecutionMode::Full),
            precommit,
        )?;
        Ok((Some(publication), targets.len()))
    }

    /// Compact both vector tiers into a complete successor while old readers
    /// keep their original files. Counts and timings use the ordinary compact
    /// command's payload fields; the caller reports success only after checking
    /// the returned publication outcome.
    #[allow(clippy::future_not_send)]
    pub(crate) async fn compact_retained_generation(
        &self,
        cx: &Cx,
        store_root: &Path,
    ) -> SearchResult<(
        crate::generation_store::GenerationPublication,
        serde_json::Value,
    )> {
        use crate::generation_store::CompleteGenerationStore;

        retained_search_checkpoint(cx)?;
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        let store = CompleteGenerationStore::open(cx, store_root)?;
        let build = store.begin(cx)?;
        let mut input = self.cli_input.clone();
        input.index_dir = Some(build.path().to_path_buf());
        input.daemon = false;
        input.daemon_socket = None;
        let candidate = self.clone().with_cli_input(input);
        retained_reuse::copy_selected_generation(cx, &candidate, &store, build.path())?;
        let candidate_lease = crate::lifecycle::PublicationLease::acquire(build.path())?;
        let mut payload = serde_json::Value::Null;
        for (tier, relative) in FSFS_VECTOR_GENERATION_FILES {
            retained_search_checkpoint(cx)?;
            candidate_lease.fence("complete-generation vector compaction")?;
            let path = build.path().join(relative);
            if tier == "quality" && !path.exists() {
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
            if tier == "fast" {
                payload = stats;
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
        let resources = Box::pin(
            candidate.prepare_search_execution_resources_at_root_with_modes(
                cx,
                build.path(),
                SearchExecutionMode::Full,
                SearchExecutionMode::Full,
            ),
        )
        .await?;
        drop(resources);
        let publication = build.publish(cx, |_, path| {
            Self::validate_search_generation_at_root(path, SearchExecutionMode::Full)
        })?;
        Ok((publication, payload))
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
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
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
            storage
                .upsert_document(&frankensearch_storage::DocumentRecord::new(
                    name,
                    "sharedtoken catalog row",
                    [7; 32],
                    23,
                    1,
                    1,
                ))
                .unwrap();
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
            let input = [
                serde_json::json!({"id": "alpha.md", "text": "discarded duplicate body"}),
                serde_json::json!({"id": "virtual/new.md", "text": "appendnovel new body"}),
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
                "appendnovel new body"
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
            assert!(catalog.get_document("virtual/new.md").unwrap().is_some());
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
            assert!(!next.path().join("FSFS-REUSE.json").exists());
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
                FsfsRuntime::collect_quality_generation_doctor_check(long_generation.path())
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
    fn retained_delete_updates_both_tiers_catalog_and_manifests_without_retargeting_old_reader() {
        run_test_with_cx(|cx| async move {
            let parent = tempfile::tempdir().unwrap();
            let (runtime, source, root, predecessor) = fixture(&cx, parent.path(), false).await;
            let before = file_bytes(predecessor.path());
            let mut old = runtime.open_retained_search(&cx, &root).await.unwrap();
            let command = deletion(&runtime, &["beta.md", "beta.md", "absent.md"], false);
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
            assert!(!next.path().join("FSFS-REUSE.json").exists());
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
            assert!(storage.get_document("alpha.md").unwrap().is_some());
            drop(storage);
            assert!(
                source.join("beta.md").exists(),
                "index deletion must retain source files"
            );
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
            let (runtime, _, root, _) = fixture(&cx, parent.path(), false).await;
            let command = deletion(&runtime, &["beta"], true);
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
            let GenerationPublication::Durable(next) = publication else {
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
                !reader
                    .generation()
                    .path()
                    .join(FSFS_EXPLAIN_SESSION_FILE)
                    .exists()
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

#[path = "runtime/semantic_windows.rs"]
mod semantic_windows;
