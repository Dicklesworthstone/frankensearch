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
            self.runtime.execute_search_phase_artifacts_with_mode_using_resources(
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
        reason: "retained generations require a catalog beneath {index_dir}, without parent traversal"
            .to_owned(),
    })
}

impl FsfsRuntime {
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
        let resources = Box::pin(runtime.prepare_search_execution_resources_at_root_with_modes(
            cx,
            generation.path(),
            SearchExecutionMode::Full,
            SearchExecutionMode::Full,
        ))
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
        use crate::generation_store::CompleteGenerationStore;

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
        let candidate = self.clone().with_cli_input(input);
        Box::pin(candidate.run_one_shot_index_scaffold_internal(
            cx,
            CliCommand::Index,
            |_| Ok(()),
            false,
        ))
        .await?;

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
        build.publish(cx, |_, path| {
            Self::validate_search_generation_at_root(path, SearchExecutionMode::Full)
        })
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
        fs::write(source.join("alpha.md"), "sharedtoken alpha retained document")
            .expect("first source");
        let mut config = FsfsConfig::default();
        config.storage.db_path = "{index_dir}/catalog.sqlite".to_owned();
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
        (FsfsRuntime::new(config).with_cli_input(input), source, store)
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
            let before = old.search(&cx, "sharedtoken", 10).await.expect("first query");
            assert_eq!(before.last().expect("initial phase").hits.len(), 1);
            assert_eq!(
                store.active(&cx).expect("query left bundle intact"),
                Some(old_generation.clone())
            );

            fs::write(source.join("beta.md"), "sharedtoken beta new document")
                .expect("new source");
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
            assert!(!reader.generation().path().join(FSFS_EXPLAIN_SESSION_FILE).exists());
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
            assert_eq!(store.active(&cx).expect("inventory remains valid"), selected);
            assert_eq!(
                fs::read(external).expect("external data"),
                b"foreign catalog sentinel"
            );
        });
    }
}
