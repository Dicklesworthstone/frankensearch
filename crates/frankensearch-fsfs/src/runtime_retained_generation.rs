// Compose the existing runtime with retained-generation rebuilding in the same
// module, so its private admission and resource-opening paths remain the single
// source of truth. The existing runtime source is left byte-for-byte intact.
include!("runtime.rs");

impl FsfsRuntime {
    /// Rebuild into an isolated complete-generation store and activate only
    /// after the ordinary full-search admission succeeds.
    ///
    /// This explicit library entry point does not change legacy CLI/watch root
    /// discovery. Consumers select a generation once with
    /// `CompleteGenerationStore::active` and use its stable physical path for
    /// every component of a read operation. Existing generations are retained.
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
        if !self
            .config
            .storage
            .db_path
            .trim()
            .starts_with(crate::config::STORAGE_DB_PATH_INDEX_DIR_PLACEHOLDER)
        {
            return Err(SearchError::InvalidConfig {
                field: "storage.db_path".to_owned(),
                value: self.config.storage.db_path.clone(),
                reason: "retained rebuilds require a generation-local {index_dir} catalog"
                    .to_owned(),
            });
        }
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
        let candidate = Self::new(self.config.clone()).with_cli_input(input);
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
        let resources = Box::pin(candidate.prepare_search_execution_resources_at_root_with_modes(
            cx,
            build.path(),
            SearchExecutionMode::Full,
            SearchExecutionMode::Full,
        ))
        .await?;
        drop(resources);
        build.publish(cx, |_, path| {
            Self::validate_search_generation_at_root(path, SearchExecutionMode::Full)
        })
    }
}
