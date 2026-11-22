//! Explicit source rebuild followed by one search against its published reader.
//!
//! This is the CLI composition of the cooperative retained-recovery APIs. It
//! optionally acquires the selected models with explicit CLI consent before an
//! offline rebuild. It does not execute the canonical planner's capability-blocked
//! or fixed-authority actions, infer consent from an error, or retry a failed query.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_embed::{
    ConsentSource, DownloadConsent, ModelDownloader, ModelLifecycle, ModelManifest,
};
use frankensearch_index::generation_root::GENERATION_ROOT_AUTHORITY_FILE_NAME;

use super::super::{
    FSFS_RERANKER_MODEL_ID, FsfsRuntime, PreparedRetainedRecovery, RetainedSearchReader,
    STORE_OVERLAP_REASON, SearchExecutionMode, canonical_store_root, degraded_search_mode,
    retained_recovery_error, retained_search_checkpoint,
    store_overlaps_source, validate_retained_catalog_path,
};
#[cfg(feature = "semantic-support")]
use super::super::{select_fast_model, select_quality_model};
use super::{complete_entry_exists, require_durable_publication};
use crate::generation_store::GenerationPublication;

fn validate_cooperative_recovery_root(root: &Path) -> SearchResult<()> {
    // Resolve existing aliases before checking ancestry, just as sealed-bundle
    // write protection does. An AUTHORITY entry, including a dangling symlink,
    // selects a different protocol; never create FSFS-CURRENT inside that root.
    let canonical = canonical_store_root(root)?;
    for ancestor in canonical.ancestors() {
        if complete_entry_exists(&ancestor.join(GENERATION_ROOT_AUTHORITY_FILE_NAME))? {
            return Err(retained_recovery_error(
                "--recover-source supports cooperative complete-generation stores, not a fixed-authority root or its descendants",
            ));
        }
    }
    Ok(())
}

impl FsfsRuntime {
    /// Resolve only producers this exact rebuild and query can execute. In
    /// particular a quality-only rebuild has no fast-model dependency, and a
    /// lexical or fast-only query cannot authorize a cross-encoder download.
    fn search_recovery_model_manifests(
        &self,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<ModelManifest>> {
        let mode = degraded_search_mode(
            self.config.pressure.degradation_override,
            self.requested_search_mode(),
        )?;
        self.validate_semantic_model_selection(mode)?;
        if matches!(mode, SearchExecutionMode::LexicalOnly) {
            return Ok(Vec::new());
        }
        let mut selected = Vec::new();
        #[cfg(feature = "semantic-support")]
        {
            if !matches!(mode, SearchExecutionMode::QualityOnly) {
                selected.push(
                    select_fast_model(&self.config.indexing.fast_model)?
                        .download_manifest()
                        .clone(),
                );
            }
            if !self.quality_disabled_by_policy()
                && (matches!(mode, SearchExecutionMode::QualityOnly)
                    || self.indexing_quality_deferral().is_none())
                && let Some(quality) = select_quality_model(&self.config.indexing.quality_model)?
            {
                Self::ensure_quality_backend(quality)?;
                selected.push(quality.download_manifest());
            }
        }
        if self.search_can_rerank(query, limit, mode) {
            if !cfg!(feature = "rerank") {
                return Err(SearchError::EmbedderUnavailable {
                    model: FSFS_RERANKER_MODEL_ID.to_owned(),
                    reason: "source recovery cannot acquire a reranker this build cannot execute; use a build with the rerank feature".to_owned(),
                });
            }
            selected.push(ModelManifest::ms_marco_reranker());
        }
        Ok(selected)
    }

    /// Qualify the entire request before downloading any one of its models.
    /// A warm cache is read-only; invalid bytes are replaced through the same
    /// verified staging and atomic promotion as the explicit download command.
    fn missing_search_recovery_models(
        manifests: Vec<ModelManifest>,
        model_root: &Path,
    ) -> SearchResult<Vec<(ModelManifest, PathBuf)>> {
        crate::lifecycle::reject_sealed_ancestry(model_root)?;
        let mut missing = Vec::new();
        for manifest in manifests {
            manifest.validate()?;
            if !manifest.is_production_ready() {
                return Err(SearchError::InvalidConfig {
                    field: "cli.search.acquire_models".to_owned(),
                    value: manifest.id.clone(),
                    reason: Self::non_production_manifest_message(&manifest),
                });
            }
            let destination = model_root.join(Self::manifest_install_dir_name(&manifest));
            crate::lifecycle::reject_sealed_ancestry(&destination)?;
            #[cfg(feature = "embedded-models")]
            if manifest == ModelManifest::potion_128m() || manifest == ModelManifest::minilm_v2() {
                // These exact bytes are already inside the full binary. Leave
                // their materialization to the existing offline build boundary;
                // selecting acquisition must not download bundled defaults.
                continue;
            }
            // Inference also accepts a direct installation at model_root.
            // Reuse that exact layout when valid; repairs still publish into
            // the canonical child, never over the root holding their staging.
            let installed = Self::registered_model_cache_path(&manifest, model_root)?;
            let (_, verified, _) = Self::inspect_manifest_installation(&manifest, &installed);
            if verified != Some(true) {
                missing.push((manifest, destination));
            }
        }
        Ok(missing)
    }

    fn search_recovery_model_root(&self) -> SearchResult<PathBuf> {
        let mut acquisition = self.clone();
        // Download-command output overrides are not part of a search request.
        // Install into the same configured cache the offline builder will load.
        acquisition.cli_input.download_output_dir = None;
        acquisition.resolve_download_model_root()
    }

    async fn acquire_search_recovery_models(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
    ) -> SearchResult<()> {
        // The flag grants this one selected-model acquisition; it never
        // overrides offline policy or grants index publication authority.
        if !self.cli_input.acquire_models || self.config.indexing.offline {
            return Err(SearchError::InvalidConfig {
                field: "cli.search.acquire_models".to_owned(),
                value: self.cli_input.acquire_models.to_string(),
                reason: "model acquisition requires --acquire-models and an online policy; omit the flag for offline recovery or explicitly use --online".to_owned(),
            });
        }
        retained_search_checkpoint(cx)?;
        let manifests = self.search_recovery_model_manifests(query, limit)?;
        if manifests.is_empty() {
            return Ok(());
        }
        let model_root = self.search_recovery_model_root()?;
        let missing = Self::missing_search_recovery_models(manifests, &model_root)?;
        let show_progress =
            self.cli_input.format == super::OutputFormat::Table && !self.cli_input.quiet;
        let downloader = ModelDownloader::with_defaults();
        for (manifest, destination) in missing {
            retained_search_checkpoint(cx)?;
            // Recheck publication ancestry after earlier asynchronous model
            // transfers. Neither acquisition nor recovery may modify a seal.
            crate::lifecycle::reject_sealed_ancestry(&model_root)?;
            crate::lifecycle::reject_sealed_ancestry(&destination)?;
            fs::create_dir_all(&model_root)?;
            let mut lifecycle = ModelLifecycle::new(
                manifest.clone(),
                DownloadConsent::granted(ConsentSource::Programmatic),
            );
            let staged = Box::pin(downloader.download_model(
                cx,
                &manifest,
                &model_root,
                &mut lifecycle,
                |progress| {
                    if show_progress {
                        eprintln!("{progress}");
                    }
                },
            ))
            .await?;
            retained_search_checkpoint(cx)?;
            crate::lifecycle::reject_sealed_ancestry(&destination)?;
            manifest.promote_verified_installation(&staged, &destination)?;
        }
        retained_search_checkpoint(cx)
    }

    /// The flag authorizes exactly one fresh source rebuild, even when the
    /// incumbent already searches successfully. No public search phase is
    /// emitted until preview and guarded, durable publication have succeeded.
    #[allow(clippy::future_not_send)]
    pub(super) async fn run_complete_search_recovery_with_writer<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        started: Instant,
        writer: &mut W,
    ) -> SearchResult<()> {
        let prepared = Box::pin(self.prepare_complete_search_recovery(cx, root)).await?;
        // restore retains the original observed selection across source scans,
        // model admission and preview. Contention is an error, never a rebase.
        let restored = prepared.restore(cx)?;
        Box::pin(
            self.finish_complete_search_recovery_with_writer(cx, root, restored, started, writer),
        )
        .await
    }

    // The existing source builder owns its process-local publication lease
    // across awaits. Run it on this command's established runtime lane.
    #[allow(clippy::future_not_send)]
    async fn prepare_complete_search_recovery(
        &self,
        cx: &Cx,
        root: &Path,
    ) -> SearchResult<PreparedRetainedRecovery> {
        self.cli_input.validate_search_recovery()?;
        retained_search_checkpoint(cx)?;
        validate_cooperative_recovery_root(root)?;
        let source = self.cli_input.recover_source.as_ref().ok_or_else(|| {
            retained_recovery_error("source recovery requires an explicit --recover-source")
        })?;
        let query = self.cli_input.query.as_deref().ok_or_else(|| {
            retained_recovery_error("source recovery requires the original search query")
        })?;
        let limit = self
            .cli_input
            .overrides
            .limit
            .unwrap_or(self.config.search.default_limit);
        // Validate query, filters, configured limit and the complete effective
        // search policy before creating a candidate. A no-op cannot authorize
        // replacement of the selected corpus.
        let query_policy = self.recovery_query_policy_sha256(query, limit)?;
        // Acquisition must not mutate the model cache for a rebuild that
        // already fails read-only source, store, catalog, or observation policy.
        validate_retained_catalog_path(&self.config.storage.db_path)?;
        if self.config.search.shadow_mode {
            return Err(retained_recovery_error(
                "shadow observation cannot write into a retained recovery target",
            ));
        }
        let source = fs::canonicalize(source)?;
        if !fs::metadata(&source)?.is_dir() {
            return Err(retained_recovery_error(
                "source recovery requires an existing source directory",
            ));
        }
        self.refuse_foreign_target_root(&source)?;
        let canonical_root = canonical_store_root(root)?;
        crate::lifecycle::reject_sealed_ancestry(&canonical_root)?;
        if store_overlaps_source(&canonical_root, &source) {
            return Err(SearchError::InvalidConfig {
                field: "complete_generation.store_root".to_owned(),
                value: canonical_root.display().to_string(),
                reason: STORE_OVERLAP_REASON.to_owned(),
            });
        }
        crate::generation_store::CompleteGenerationStore::validate_recovery_selection(
            &canonical_root,
        )?;
        if self.cli_input.acquire_models {
            // Provisioning is a standalone cache operation before the rebuild
            // transaction. The rebuild captures its expected-old selection once
            // provisioning completes, then retains it through preview/restore.
            Box::pin(self.acquire_search_recovery_models(cx, query, limit)).await?;
        }
        let mut recovery = self.clone();
        recovery.cli_input.recover_source = None;
        recovery.cli_input.acquire_models = false;
        recovery.cli_input.target_path = Some(source);
        recovery.cli_input.daemon = false;
        recovery.cli_input.daemon_socket = None;
        recovery.config.indexing.offline = true;

        // This existing flow validates source/store disjointness before staging
        // and captures the incumbent selection before indexing. It builds fresh
        // data, verifies every stored producer, and never trusts damaged bytes.
        let mut prepared = Box::pin(recovery.prepare_retained_rebuild(cx, root)).await?;
        Box::pin(prepared.search(cx, query, limit)).await?;
        if prepared
            .reader
            .runtime
            .recovery_query_policy_sha256(query, limit)?
            != query_policy
        {
            return Err(retained_recovery_error(
                "the prepared recovery does not preserve the original search policy",
            ));
        }
        retained_search_checkpoint(cx)?;
        Ok(prepared)
    }

    async fn finish_complete_search_recovery_with_writer<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        restored: (GenerationPublication, RetainedSearchReader),
        started: Instant,
        writer: &mut W,
    ) -> SearchResult<()> {
        let (publication, reader) = restored;
        // A visible-but-uncertain publication must never produce a successful
        // search receipt or trigger restoration of the predecessor.
        let published = require_durable_publication(publication)?;
        if &published != reader.generation() {
            return Err(retained_recovery_error(
                "recovery publication and admitted reader name different targets; the published selection has not been undone",
            ));
        }
        tracing::info!(
            generation = published.id(),
            "source recovery published; executing the original search once"
        );
        // Use the admitted reader returned by restore. Reopening the current
        // pointer here could execute the original request against a concurrent
        // successor that was never previewed. Output errors are returned once;
        // neither publication nor search is retried after a partial write.
        Box::pin(self.run_retained_search_with_writer(cx, root, reader, started, writer)).await
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod tests {
    use std::fs;
    use std::future::Future;
    use std::io;
    use std::path::PathBuf;

    use frankensearch_core::SearchError;

    use super::*;
    use crate::config::DegradationOverrideMode;
    use crate::generation_store::{
        COMPLETE_GENERATION_POINTER, CompleteGenerationStore, PublishedGeneration,
    };
    use crate::output_schema::{OutputEnvelope, SearchHitPayload, SearchPayload};
    use crate::stream_protocol::{StreamEvent, StreamFrame, StreamTerminalStatus};
    use crate::{CliCommand, CliInput, FsfsConfig, OutputFormat};

    const QUERY: &str = "how does sharedtoken recovery preserve source documents";

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
            format!("{QUERY} alpha source documents"),
        )
        .unwrap();
        let mut config = FsfsConfig::default();
        config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
        config.indexing.offline = true;
        config.indexing.quality_model.clear();
        config.indexing.model_dir = parent.join("missing-models").display().to_string();
        config.search.fast_only = true;
        config.search.rerank = false;
        config.search.quality_timeout_ms = 5_000;
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        let input = CliInput {
            command: CliCommand::Index,
            target_path: Some(source.clone()),
            index_dir: Some(root.clone()),
            format: OutputFormat::Json,
            quiet: true,
            ..CliInput::default()
        };
        (FsfsRuntime::new(config).with_cli_input(input), source, root)
    }

    fn recovery_runtime(runtime: &FsfsRuntime, source: &Path) -> FsfsRuntime {
        let mut input = runtime.cli_input.clone();
        input.command = CliCommand::Search;
        input.target_path = None;
        input.query = Some(QUERY.to_owned());
        input.recover_source = Some(source.to_path_buf());
        input.daemon = false;
        input.overrides.limit = Some(1);
        runtime.clone().with_cli_input(input)
    }

    async fn publish(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> PublishedGeneration {
        require_durable_publication(
            Box::pin(runtime.rebuild_retained_generation(cx, root))
                .await
                .unwrap(),
        )
        .unwrap()
    }

    fn generation_count(root: &Path) -> usize {
        fs::read_dir(root.join("generations")).unwrap().count()
    }

    #[test]
    fn search_recovery_acquisition_respects_effective_offline_policy() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let selection = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let mut search = recovery_runtime(&runtime, &source);
            search.cli_input.acquire_models = true;
            // No CLI --offline is present: resolved configuration still wins.
            assert!(search.cli_input.overrides.offline.is_none());
            let mut output = Vec::new();
            let error = Box::pin(search.run_complete_generation_search_with_writer(
                &cx,
                &root,
                &mut output,
            ))
            .await
            .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "cli.search.acquire_models"));
            assert!(output.is_empty());
            assert_eq!(generation_count(&root), 1);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                selection
            );
            assert!(old.path().exists());
            assert!(!directory.path().join("missing-models").exists());
        });
    }

    #[test]
    fn search_recovery_acquisition_cancellation_prevents_cache_and_generation_writes() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut search = recovery_runtime(&runtime, &source);
            search.config.indexing.offline = false;
            search.cli_input.acquire_models = true;
            let mut output = Vec::new();
            cx.set_cancel_requested(true);
            let result = Box::pin(search.run_complete_generation_search_with_writer(
                &cx,
                &root,
                &mut output,
            ))
            .await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert!(output.is_empty());
            assert!(!root.exists());
            assert!(!directory.path().join("missing-models").exists());
        });
    }

    #[test]
    fn search_recovery_acquisition_lexical_rebuild_preserves_one_exact_replay() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let mut search = recovery_runtime(&runtime, &source);
            search.config.indexing.offline = false;
            search.cli_input.acquire_models = true;
            search.cli_input.filter = Some("path:alpha.md".to_owned());
            // None of these producers can execute in a lexical-only request.
            "unused-fast-model".clone_into(&mut search.config.indexing.fast_model);
            "unused-quality-model".clone_into(&mut search.config.indexing.quality_model);
            search.config.search.rerank = true;
            let mut output = Vec::new();
            Box::pin(search.run_complete_generation_search_with_writer(&cx, &root, &mut output))
                .await
                .unwrap();
            let envelope: OutputEnvelope<SearchPayload> = serde_json::from_slice(&output).unwrap();
            let payload = envelope.data.unwrap();
            assert_eq!(payload.query, QUERY);
            assert_eq!(payload.returned_hits, 1);
            assert!(payload.hits[0].path.ends_with("alpha.md"));
            assert_eq!(
                generation_count(&root),
                2,
                "one fresh rebuild and one replay"
            );
            assert!(old.path().exists());
            assert!(!directory.path().join("missing-models").exists());
        });
    }

    #[test]
    fn search_recovery_acquisition_checks_all_rebuild_inputs_before_writes() {
        on_runtime(|cx| async move {
            for case in [
                "missing_source",
                "file_source",
                "source_alias",
                "catalog",
                "shadow",
                "filter",
            ] {
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, mut root) = fixture(directory.path());
                let mut search = recovery_runtime(&runtime, &source);
                search.config.indexing.offline = false;
                search.cli_input.acquire_models = true;
                match case {
                    "missing_source" => {
                        search.cli_input.recover_source = Some(directory.path().join("absent"));
                    }
                    "file_source" => {
                        search.cli_input.recover_source = Some(source.join("alpha.md"));
                    }
                    "source_alias" => {
                        let alias = directory.path().join("alias");
                        std::os::unix::fs::symlink(&source, &alias).unwrap();
                        root = alias.join("store");
                        search.cli_input.index_dir = Some(root.clone());
                    }
                    "catalog" => {
                        search.config.storage.db_path = directory
                            .path()
                            .join("catalog.sqlite")
                            .display()
                            .to_string();
                    }
                    "shadow" => search.config.search.shadow_mode = true,
                    "filter" => search.cli_input.filter = Some("unsupported:value".to_owned()),
                    _ => unreachable!("bounded fixture"), // ubs:ignore — cfg(test) case assertion.
                }
                let mut output = Vec::new();
                assert!(
                    Box::pin(search.run_complete_generation_search_with_writer(
                        &cx,
                        &root,
                        &mut output,
                    ))
                    .await
                    .is_err(),
                    "{case}"
                );
                assert!(output.is_empty(), "{case}: no public replay");
                assert!(!root.exists(), "{case}: no staging");
                assert!(
                    !directory.path().join("missing-models").exists(),
                    "{case}: no model-cache writes"
                );
                assert!(
                    !directory.path().join("catalog.sqlite").exists(),
                    "{case}: no external catalog"
                );
            }
        });
    }

    #[test]
    fn search_recovery_acquisition_uses_the_inference_cache_without_download_overrides() {
        let directory = tempfile::tempdir().unwrap();
        let (runtime, source, _) = fixture(directory.path());
        let mut search = recovery_runtime(&runtime, &source);
        search.cli_input.download_output_dir = Some(directory.path().join("unrelated-cache"));
        assert_eq!(
            search.search_recovery_model_root().unwrap(),
            directory.path().join("missing-models")
        );
        assert!(!directory.path().join("missing-models").exists());
        assert!(!directory.path().join("unrelated-cache").exists());
    }

    #[cfg(feature = "semantic-support")]
    #[test]
    fn search_recovery_acquisition_admits_sources_before_inspecting_missing_models() {
        on_runtime(|cx| async move {
            for (case, expected) in [
                ("file", "existing source directory"),
                ("overlap", "must not overlap"),
                ("shadow", "shadow observation"),
            ] {
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, mut root) = fixture(directory.path());
                let mut search = recovery_runtime(&runtime, &source);
                search.config.indexing.offline = false;
                search.config.indexing.fast_model = "potion-base-8m".to_owned();
                search.config.search.fast_only = true;
                search.config.pressure.degradation_override = DegradationOverrideMode::Auto;
                search.lexical_only_indexing = false;
                search.cli_input.acquire_models = true;
                // This would fail acquisition before any network request. Its
                // distinct error proves invalid build inputs are checked first.
                let cache = PathBuf::from(&search.config.indexing.model_dir);
                fs::create_dir(&cache).unwrap();
                fs::write(
                    cache.join(GENERATION_ROOT_AUTHORITY_FILE_NAME),
                    b"cache tripwire",
                )
                .unwrap();
                match case {
                    "file" => search.cli_input.recover_source = Some(source.join("alpha.md")),
                    "overlap" => {
                        root = source.join("store");
                        search.cli_input.index_dir = Some(root.clone());
                    }
                    "shadow" => search.config.search.shadow_mode = true,
                    _ => unreachable!("bounded fixture"), // ubs:ignore — cfg(test) case assertion.
                }
                let mut output = Vec::new();
                let error = Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output,
                ))
                .await
                .unwrap_err();
                assert!(error.to_string().contains(expected), "{case}: {error}");
                assert!(output.is_empty());
                assert!(!root.exists());
                assert_eq!(fs::read_dir(&cache).unwrap().count(), 1);
            }
        });
    }

    #[test]
    fn search_recovery_acquisition_refuses_unsafe_selection_before_model_cache_writes() {
        on_runtime(|cx| async move {
            for case in ["symlink", "directory", "oversized"] {
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, root) = fixture(directory.path());
                let mut search = recovery_runtime(&runtime, &source);
                search.config.indexing.offline = false;
                search.cli_input.acquire_models = true;
                #[cfg(feature = "semantic-support")]
                {
                    search.config.indexing.fast_model = "potion-base-8m".to_owned();
                    search.config.search.fast_only = true;
                    search.config.pressure.degradation_override = DegradationOverrideMode::Auto;
                    search.lexical_only_indexing = false;
                }
                let cache = PathBuf::from(&search.config.indexing.model_dir);
                fs::create_dir(&cache).unwrap();
                fs::write(
                    cache.join(GENERATION_ROOT_AUTHORITY_FILE_NAME),
                    b"cache tripwire",
                )
                .unwrap();
                fs::create_dir(&root).unwrap();
                let pointer = root.join(COMPLETE_GENERATION_POINTER);
                match case {
                    "symlink" => {
                        std::os::unix::fs::symlink(source.join("alpha.md"), &pointer).unwrap();
                    }
                    "directory" => fs::create_dir(&pointer).unwrap(),
                    "oversized" => fs::write(&pointer, vec![b'x'; 257]).unwrap(),
                    _ => unreachable!("bounded fixture"), // ubs:ignore — cfg(test) case assertion.
                }
                let mut output = Vec::new();
                let error = Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output,
                ))
                .await
                .unwrap_err();
                assert!(
                    error.to_string().contains("regular file"),
                    "{case}: {error}"
                );
                assert!(output.is_empty());
                assert_eq!(
                    fs::read_dir(&root).unwrap().count(),
                    1,
                    "{case}: no staging"
                );
                assert_eq!(
                    fs::read_dir(&cache).unwrap().count(),
                    1,
                    "{case}: no acquisition"
                );
                assert!(fs::symlink_metadata(&pointer).is_ok());
            }
        });
    }

    fn acquisition_manifest() -> ModelManifest {
        use frankensearch_embed::{ModelFile, ModelTier};
        use sha2::Digest as _;

        let bytes = b"verified model fixture";
        ModelManifest {
            id: "search-recovery-fixture".to_owned(),
            version: "fixture-v1".to_owned(),
            display_name: None,
            description: None,
            repo: "fixture/model".to_owned(),
            revision: "a".repeat(40),
            files: vec![ModelFile {
                name: "model.bin".to_owned(),
                sha256: crate::runtime::sha256_digest_hex(sha2::Sha256::digest(bytes)),
                size: u64::try_from(bytes.len()).unwrap(),
                url: None,
            }],
            license: "MIT".to_owned(),
            dimension: Some(3),
            tier: Some(ModelTier::Fast),
            download_size_bytes: u64::try_from(bytes.len()).unwrap(),
        }
    }

    #[test]
    fn search_recovery_acquisition_plan_is_read_only_and_reuses_only_verified_models() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let manifest = acquisition_manifest();
        let destination = root.join(&manifest.id);
        let missing =
            FsfsRuntime::missing_search_recovery_models(vec![manifest.clone()], &root).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].1, destination);
        assert!(!root.exists(), "planning cannot create the cache");

        fs::create_dir_all(&destination).unwrap();
        let artifact = destination.join("model.bin");
        fs::write(&artifact, b"verified model fixture").unwrap();
        assert!(
            FsfsRuntime::missing_search_recovery_models(vec![manifest.clone()], &root)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            fs::read_dir(&destination).unwrap().count(),
            1,
            "warm verification does not publish a cache marker"
        );

        fs::write(&artifact, b"corrupt model fixture").unwrap();
        let missing =
            FsfsRuntime::missing_search_recovery_models(vec![manifest.clone()], &root).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(fs::read(&artifact).unwrap(), b"corrupt model fixture");

        let mut unqualified = manifest;
        "main".clone_into(&mut unqualified.revision);
        assert!(FsfsRuntime::missing_search_recovery_models(vec![unqualified], &root).is_err());
        assert_eq!(fs::read(&artifact).unwrap(), b"corrupt model fixture");
    }

    #[test]
    fn search_recovery_acquisition_refuses_sealed_cache_aliases() {
        for marker in [
            crate::generation_store::COMPLETE_GENERATION_MANIFEST,
            GENERATION_ROOT_AUTHORITY_FILE_NAME,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let sealed = directory.path().join("sealed");
            fs::create_dir(&sealed).unwrap();
            fs::write(sealed.join(marker), b"existing sealed owner").unwrap();
            let alias = directory.path().join("alias");
            std::os::unix::fs::symlink(&sealed, &alias).unwrap();
            assert!(
                FsfsRuntime::missing_search_recovery_models(
                    vec![acquisition_manifest()],
                    &alias.join("models")
                )
                .is_err()
            );
            assert_eq!(fs::read_dir(&sealed).unwrap().count(), 1);
            assert_eq!(
                fs::read(sealed.join(marker)).unwrap(),
                b"existing sealed owner"
            );
        }
    }

    #[test]
    fn search_recovery_acquisition_reuses_direct_cache_and_repairs_the_preferred_child() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("direct-model");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("model.bin"), b"verified model fixture").unwrap();
        let manifest = acquisition_manifest();
        let destination = root.join(&manifest.id);
        assert!(
            FsfsRuntime::missing_search_recovery_models(vec![manifest.clone()], &root)
                .unwrap()
                .is_empty()
        );
        assert!(
            !destination.exists(),
            "a valid direct cache is reused without download"
        );
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);

        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("model.bin"), b"corrupt child").unwrap();
        let missing = FsfsRuntime::missing_search_recovery_models(vec![manifest], &root).unwrap();
        assert_eq!(
            missing.len(),
            1,
            "an existing child takes precedence for inference"
        );
        assert_eq!(missing[0].1, destination);
        assert_eq!(
            fs::read(root.join("model.bin")).unwrap(),
            b"verified model fixture"
        );
        assert_eq!(
            fs::read(destination.join("model.bin")).unwrap(),
            b"corrupt child"
        );
    }

    #[cfg(feature = "semantic-support")]
    #[test]
    fn search_recovery_acquisition_selects_only_the_requested_tiers() {
        let mut config = FsfsConfig::default();
        config.indexing.fast_model = "potion-base-8M".to_owned();
        config.indexing.quality_model = "unused-quality".to_owned();
        config.search.fast_only = true;
        config.search.rerank = true;
        let ids = |runtime: &FsfsRuntime| {
            runtime
                .search_recovery_model_manifests(QUERY, 3)
                .unwrap()
                .into_iter()
                .map(|manifest| manifest.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(&FsfsRuntime::new(config.clone())),
            vec!["potion-base-8m"]
        );
        config.search.fast_only = false;
        config.pressure.degradation_override = DegradationOverrideMode::ForceEmbedDeferred;
        assert_eq!(
            ids(&FsfsRuntime::new(config.clone())),
            vec!["potion-base-8m"]
        );
        config.pressure.degradation_override = DegradationOverrideMode::ForceLexicalOnly;
        "unused-fast".clone_into(&mut config.indexing.fast_model);
        assert!(ids(&FsfsRuntime::new(config)).is_empty());

        #[cfg(any(feature = "semantic-loaders", feature = "rerank"))]
        {
            let mut config = FsfsConfig::default();
            config.indexing.fast_model = "potion-base-8m".to_owned();
            let quality = if cfg!(feature = "semantic-loaders") {
                "all-minilm-l6-v2"
            } else {
                "all-minilm-l6-v2-native"
            };
            config.indexing.quality_model = quality.to_owned();
            config.search.rerank = false;
            assert_eq!(
                ids(&FsfsRuntime::new(config.clone())),
                vec!["potion-base-8m", quality]
            );
            "unused-fast".clone_into(&mut config.indexing.fast_model);
            let mut runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                quality_only: true,
                ..CliInput::default()
            });
            assert_eq!(ids(&runtime), vec![quality]);
            runtime.config.search.rerank = true;
            #[cfg(feature = "rerank")]
            assert_eq!(ids(&runtime), vec![quality, FSFS_RERANKER_MODEL_ID]);
            #[cfg(not(feature = "rerank"))]
            assert!(matches!(runtime.search_recovery_model_manifests(QUERY, 3),
                Err(SearchError::EmbedderUnavailable { model, .. }) if model == FSFS_RERANKER_MODEL_ID));
        }
    }

    #[test]
    fn search_recovery_repairs_damaged_selection_and_preserves_query_filter_and_explain() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"damaged selection").unwrap();
            fs::write(
                source.join("beta.md"),
                format!("{QUERY} new beta source documents"),
            )
            .unwrap();
            fs::write(
                source.join("gamma.md"),
                format!("{QUERY} excluded gamma documents"),
            )
            .unwrap();
            let mut search = recovery_runtime(&runtime, &source);
            search.cli_input.filter = Some("path:beta.md".to_owned());
            search.config.search.ranking_priors = false;
            search.config.search.explain = true;
            assert_eq!(
                search.complete_generation_command_root(false).unwrap(),
                Some(root.clone())
            );

            let mut output = Vec::new();
            Box::pin(search.run_complete_generation_search_with_writer(&cx, &root, &mut output))
                .await
                .unwrap();
            let envelope: OutputEnvelope<SearchPayload> = serde_json::from_slice(&output).unwrap();
            assert!(envelope.ok);
            let payload = envelope.data.unwrap();
            assert_eq!(payload.query, QUERY);
            assert_eq!(payload.returned_hits, 1);
            assert!(payload.hits[0].path.ends_with("beta.md"));
            assert!(payload.hits[0].semantic_rank.is_none());
            assert_eq!(payload.explanations.len(), 1);
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let current = store.active(&cx).unwrap().unwrap();
            assert_ne!(current.id(), old.id());
            assert!(old.path().exists(), "the predecessor remains retained");
            assert_eq!(generation_count(&root), 2, "one new build, no retry");
            assert!(!directory.path().join("missing-models").exists());
            assert!(
                !current
                    .path()
                    .join(crate::runtime::FSFS_COMPLETE_EXPLAIN_SESSION_FILE)
                    .exists()
            );
        });
    }

    #[test]
    fn search_recovery_streams_one_original_request_after_publication() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let mut search = recovery_runtime(&runtime, &source);
            search.cli_input.stream = true;
            search.cli_input.format = OutputFormat::Jsonl;
            assert_eq!(
                search.complete_generation_command_root(false).unwrap(),
                Some(root.clone())
            );
            let mut output = Vec::new();
            Box::pin(search.run_complete_generation_search_with_writer(&cx, &root, &mut output))
                .await
                .unwrap();
            let frames: Vec<StreamFrame<SearchHitPayload>> = output
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect();
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| matches!(frame.event, StreamEvent::Started(_)))
                    .count(),
                1
            );
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| matches!(frame.event, StreamEvent::Terminal(_)))
                    .count(),
                1
            );
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| matches!(frame.event, StreamEvent::Result(_)))
                    .count(),
                1
            );
            for (sequence, frame) in frames.iter().enumerate() {
                assert_eq!(frame.seq, u64::try_from(sequence).unwrap());
                assert_eq!(frame.stream_id, frames[0].stream_id);
                assert!(crate::stream_protocol::validate_stream_frame(frame).valid);
            }
            assert!(
                matches!(&frames.last().unwrap().event, StreamEvent::Terminal(terminal)
                if terminal.status == StreamTerminalStatus::Completed)
            );
            assert!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .is_some()
            );
            assert_eq!(generation_count(&root), 1);
        });
    }

    #[test]
    fn search_recovery_refuses_invalid_policy_and_overlapping_source_before_staging() {
        on_runtime(|cx| async move {
            for invalid in [
                "zero_limit",
                "stream_format",
                "explain_compact",
                "paused",
                "overlap",
            ] {
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, mut root) = fixture(directory.path());
                let mut search = recovery_runtime(&runtime, &source);
                match invalid {
                    "zero_limit" => {
                        search.cli_input.overrides.limit = None;
                        search.config.search.default_limit = 0;
                    }
                    "stream_format" => search.cli_input.stream = true,
                    "explain_compact" => {
                        search.config.search.explain = true;
                        search.cli_input.compact = true;
                    }
                    "paused" => {
                        search.config.pressure.degradation_override =
                            DegradationOverrideMode::ForcePaused;
                    }
                    "overlap" => {
                        root = source.join("unsafe-store");
                        search.cli_input.index_dir = Some(root.clone());
                    }
                    _ => unreachable!("bounded fixture"), // ubs:ignore — cfg(test) case assertion.
                }
                let mut output = Vec::new();
                assert!(
                    Box::pin(search.run_complete_generation_search_with_writer(
                        &cx,
                        &root,
                        &mut output
                    ))
                    .await
                    .is_err(),
                    "{invalid}"
                );
                assert!(output.is_empty(), "{invalid}: no public replay");
                assert!(!root.exists(), "{invalid}: no staging before admission");
            }
        });
    }

    #[test]
    fn search_recovery_refuses_fixed_authority_roots_and_aliases_before_staging() {
        on_runtime(|cx| async move {
            for case in ["root", "ancestor", "dangling_authority", "aliased_ancestor"] {
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, _) = fixture(directory.path());
                let authority_root = directory.path().join("fixed-authority");
                fs::create_dir(&authority_root).unwrap();
                let authority = authority_root.join(GENERATION_ROOT_AUTHORITY_FILE_NAME);
                if case == "dangling_authority" {
                    std::os::unix::fs::symlink(authority_root.join("absent-authority"), &authority)
                        .unwrap();
                } else {
                    fs::write(&authority, b"fixed protocol owns this root").unwrap();
                }
                let root = match case {
                    "ancestor" => authority_root.join("nested-store"),
                    "aliased_ancestor" => {
                        let alias = directory.path().join("alias");
                        std::os::unix::fs::symlink(&authority_root, &alias).unwrap();
                        alias.join("nested-store")
                    }
                    _ => authority_root.clone(),
                };
                let mut search = recovery_runtime(&runtime, &source);
                search.cli_input.index_dir = Some(root.clone());
                let mut output = Vec::new();
                let error = Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output,
                ))
                .await
                .unwrap_err();
                assert!(
                    error.to_string().contains("fixed-authority"),
                    "{case}: {error}"
                );
                assert!(output.is_empty());
                assert!(!root.join("generations").exists(), "{case}: no staging");
                assert!(
                    !root.join(COMPLETE_GENERATION_POINTER).exists(),
                    "{case}: no competing selection"
                );
                assert!(fs::symlink_metadata(&authority).is_ok());
            }
        });
    }

    #[test]
    fn search_recovery_preview_miss_and_cancellation_preserve_selection() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let selection = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
            let mut search = recovery_runtime(&runtime, &source);
            search.cli_input.filter = Some("path:absent.md".to_owned());
            let mut output = Vec::new();
            let error = Box::pin(search.run_complete_generation_search_with_writer(
                &cx,
                &root,
                &mut output,
            ))
            .await
            .unwrap_err();
            assert!(error.to_string().contains("preview returned no hits"));
            assert!(output.is_empty());
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                selection
            );
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(old)
            );

            let before = generation_count(&root);
            cx.set_cancel_requested(true);
            let cancelled = Box::pin(search.run_complete_generation_search_with_writer(
                &cx,
                &root,
                &mut output,
            ))
            .await;
            cx.set_cancel_requested(false);
            assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
            assert_eq!(generation_count(&root), before);
            assert_eq!(
                fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                selection
            );
            assert!(output.is_empty());
        });
    }

    #[test]
    fn search_recovery_uncertain_publication_emits_no_replay_and_does_not_roll_back() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let search = recovery_runtime(&runtime, &source);
            let prepared = Box::pin(search.prepare_complete_search_recovery(&cx, &root))
                .await
                .unwrap();
            let (publication, reader) = prepared.restore(&cx).unwrap();
            let published = require_durable_publication(publication).unwrap();
            assert_ne!(published.id(), old.id());
            // Exercise the production branch for the store's explicit outcome:
            // selection is visible, but its final sync was not confirmed.
            let uncertain = GenerationPublication::VisibleButDurabilityUncertain {
                generation: published.clone(),
                source: io::Error::other("fixture final-directory-sync failure"),
            };
            let mut output = Vec::new();
            let error = Box::pin(search.finish_complete_search_recovery_with_writer(
                &cx,
                &root,
                (uncertain, reader),
                Instant::now(),
                &mut output,
            ))
            .await
            .unwrap_err();
            assert!(error.to_string().contains("already visible"));
            assert!(output.is_empty());
            assert_eq!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap(),
                Some(published)
            );
            assert!(old.path().exists());
        });
    }

    struct BrokenOutput {
        writes: usize,
    }

    impl Write for BrokenOutput {
        fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "fixture output closed",
            ))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn search_recovery_output_failure_keeps_publication_without_retrying() {
        on_runtime(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let (runtime, source, root) = fixture(directory.path());
            let old = publish(&runtime, &cx, &root).await;
            let search = recovery_runtime(&runtime, &source);
            let mut output = BrokenOutput { writes: 0 };
            assert!(
                Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output
                ))
                .await
                .is_err()
            );
            assert_eq!(output.writes, 1);
            let current = CompleteGenerationStore::open(&cx, &root)
                .unwrap()
                .active(&cx)
                .unwrap()
                .unwrap();
            assert_ne!(current.id(), old.id());
            assert_eq!(
                generation_count(&root),
                2,
                "one rebuild despite failed output"
            );
            assert!(old.path().exists());
        });
    }

    #[cfg(feature = "semantic-support")]
    mod semantic {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        use frankensearch_core::{
            Embedder, EmbeddingIdentityBundleV1, ModelCategory, SearchFuture,
        };

        use super::*;
        use crate::output_schema::SearchOutputPhase;
        use crate::runtime::{
            set_test_fast_embedder, set_test_quality_embedder, test_fast_embedder_override,
            test_quality_embedder_override,
        };

        type QueryHook = Box<dyn FnOnce(&Cx) -> SearchResult<()> + Send>;

        // Deterministic semantic producers exercise tier admission and the
        // public command lifecycle. They do not measure real-model relevance.
        struct RecoveryProducer {
            identity: EmbeddingIdentityBundleV1,
            category: ModelCategory,
            calls: AtomicUsize,
            queries: AtomicUsize,
            on_query: Mutex<Option<QueryHook>>,
        }

        impl Embedder for RecoveryProducer {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                Ok(&self.identity)
            }

            fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                Box::pin(async move {
                    self.calls.fetch_add(1, Ordering::Relaxed);
                    if text == QUERY {
                        self.queries.fetch_add(1, Ordering::Relaxed);
                        let hook = self.on_query.lock().unwrap().take();
                        if let Some(hook) = hook {
                            hook(cx)?;
                        }
                    }
                    Ok(vec![1.0, 0.0, 0.0])
                })
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
            previous_fast: Option<Arc<dyn Embedder>>,
            previous_quality: Option<Arc<dyn Embedder>>,
            fast: Arc<RecoveryProducer>,
            quality: Arc<RecoveryProducer>,
        }

        impl RecoveryModels {
            fn install() -> Self {
                let provider = |name, category| {
                    Arc::new(RecoveryProducer {
                        identity: EmbeddingIdentityBundleV1::explicit_test_model(name, 3),
                        category,
                        calls: AtomicUsize::new(0),
                        queries: AtomicUsize::new(0),
                        on_query: Mutex::new(None),
                    })
                };
                let models = Self {
                    previous_fast: test_fast_embedder_override(),
                    previous_quality: test_quality_embedder_override(),
                    fast: provider("search-recovery-fast", ModelCategory::StaticEmbedder),
                    quality: provider(
                        "search-recovery-quality",
                        ModelCategory::TransformerEmbedder,
                    ),
                };
                set_test_fast_embedder(Some(models.fast.clone()));
                set_test_quality_embedder(Some(models.quality.clone()));
                models
            }
        }

        impl Drop for RecoveryModels {
            fn drop(&mut self) {
                set_test_fast_embedder(self.previous_fast.take());
                set_test_quality_embedder(self.previous_quality.take());
            }
        }

        fn semantic_fixture(parent: &Path) -> (FsfsRuntime, PathBuf, PathBuf) {
            let (mut runtime, source, root) = fixture(parent);
            runtime.config.pressure.degradation_override = DegradationOverrideMode::Auto;
            runtime.config.search.fast_only = false;
            "all-minilm-l6-v2".clone_into(&mut runtime.config.indexing.quality_model);
            runtime.lexical_only_indexing = false;
            (runtime, source, root)
        }

        #[test]
        fn search_recovery_preserves_each_semantic_mode_and_replays_exactly_once() {
            on_runtime(|cx| async move {
                for mode in ["full", "fast", "quality"] {
                    let models = RecoveryModels::install();
                    let directory = tempfile::tempdir().unwrap();
                    let (runtime, source, root) = semantic_fixture(directory.path());
                    let mut search = recovery_runtime(&runtime, &source);
                    search.config.search.fast_only = mode == "fast";
                    search.cli_input.quality_only = mode == "quality";
                    if mode == "quality" {
                        "unused-invalid-fast-producer"
                            .clone_into(&mut search.config.indexing.fast_model);
                    }
                    let mut output = Vec::new();
                    Box::pin(search.run_complete_generation_search_with_writer(
                        &cx,
                        &root,
                        &mut output,
                    ))
                    .await
                    .unwrap();
                    let envelope: OutputEnvelope<SearchPayload> =
                        serde_json::from_slice(&output).unwrap();
                    let payload = envelope.data.unwrap();
                    assert_eq!(payload.query, QUERY);
                    assert_eq!(payload.returned_hits, 1);
                    assert!(payload.hits[0].semantic_rank.is_some(), "{mode}");
                    assert!(!payload.vector_generation_is_hash);
                    assert_eq!(
                        payload.phase,
                        if mode == "full" {
                            SearchOutputPhase::Refined
                        } else {
                            SearchOutputPhase::Initial
                        }
                    );
                    assert_eq!(
                        models.fast.queries.load(Ordering::Relaxed),
                        if mode == "quality" { 0 } else { 2 },
                        "one preview and one public replay: {mode}"
                    );
                    assert_eq!(
                        models.quality.queries.load(Ordering::Relaxed),
                        if mode == "fast" { 0 } else { 2 },
                        "one preview and one public replay: {mode}"
                    );
                    assert_eq!(generation_count(&root), 1);
                    if mode == "quality" {
                        assert_eq!(models.fast.calls.load(Ordering::Relaxed), 0);
                        let active = CompleteGenerationStore::open(&cx, &root)
                            .unwrap()
                            .active(&cx)
                            .unwrap()
                            .unwrap();
                        assert!(!active.path().join("vector/index.fsvi").exists());
                        assert!(active.path().join("vector/quality.fsvi").is_file());
                    }
                    assert!(!directory.path().join("missing-models").exists());
                }
            });
        }

        #[test]
        fn search_recovery_concurrent_publication_during_preview_is_not_overwritten() {
            on_runtime(|cx| async move {
                let models = RecoveryModels::install();
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, root) = semantic_fixture(directory.path());
                let old = publish(&runtime, &cx, &root).await;
                let successor = Arc::new(Mutex::new(None));
                let published = Arc::clone(&successor);
                let store_root = root.clone();
                *models.fast.on_query.lock().unwrap() = Some(Box::new(move |cx| {
                    // Publication uses the real store lease, seal and selection
                    // protocol while recovery's candidate is being previewed.
                    let store = CompleteGenerationStore::open(cx, &store_root)?;
                    let build = store.begin(cx)?;
                    fs::write(
                        build.path().join("concurrent-publisher"),
                        b"selected elsewhere",
                    )?;
                    let generation =
                        require_durable_publication(build.publish(cx, |_, _| Ok(()))?)?;
                    *published.lock().unwrap() = Some(generation);
                    Ok(())
                }));
                let search = recovery_runtime(&runtime, &source);
                let mut output = Vec::new();
                let error = Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output,
                ))
                .await
                .unwrap_err();
                assert!(error.to_string().contains("selection changed"), "{error}");
                assert!(output.is_empty(), "preview must not leak public phases");
                let current = CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap();
                assert_eq!(current, *successor.lock().unwrap());
                assert_ne!(current.as_ref().unwrap().id(), old.id());
                assert_eq!(
                    models.fast.queries.load(Ordering::Relaxed),
                    1,
                    "no replay or retry"
                );
                assert_eq!(
                    generation_count(&root),
                    3,
                    "old, inactive recovery, competing publication"
                );
            });
        }

        #[test]
        fn search_recovery_cancelled_preview_never_publishes_or_replays() {
            on_runtime(|cx| async move {
                let models = RecoveryModels::install();
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, root) = semantic_fixture(directory.path());
                let old = publish(&runtime, &cx, &root).await;
                *models.fast.on_query.lock().unwrap() = Some(Box::new(|cx| {
                    cx.set_cancel_requested(true);
                    Ok(())
                }));
                let search = recovery_runtime(&runtime, &source);
                let mut output = Vec::new();
                let result = Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output,
                ))
                .await;
                cx.set_cancel_requested(false);
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                assert!(output.is_empty());
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(old)
                );
                assert_eq!(models.fast.queries.load(Ordering::Relaxed), 1);
                assert_eq!(models.quality.queries.load(Ordering::Relaxed), 0);
            });
        }

        #[test]
        fn search_recovery_required_quality_failure_cannot_publish_fast_fallback() {
            on_runtime(|cx| async move {
                let models = RecoveryModels::install();
                let directory = tempfile::tempdir().unwrap();
                let (runtime, source, root) = semantic_fixture(directory.path());
                let old = publish(&runtime, &cx, &root).await;
                let selection = fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap();
                *models.quality.on_query.lock().unwrap() = Some(Box::new(|_| {
                    Err(SearchError::EmbedderUnavailable {
                        model: "search-recovery-quality".to_owned(),
                        reason: "fixture quality producer unavailable during preview".to_owned(),
                    })
                }));
                let search = recovery_runtime(&runtime, &source);
                let mut output = Vec::new();
                let result = Box::pin(search.run_complete_generation_search_with_writer(
                    &cx,
                    &root,
                    &mut output,
                ))
                .await;
                assert!(
                    result.is_err(),
                    "Full recovery requires successful quality preview"
                );
                assert!(
                    output.is_empty(),
                    "the successful fast phase is not public recovery output"
                );
                assert_eq!(
                    fs::read(root.join(COMPLETE_GENERATION_POINTER)).unwrap(),
                    selection
                );
                assert_eq!(
                    CompleteGenerationStore::open(&cx, &root)
                        .unwrap()
                        .active(&cx)
                        .unwrap(),
                    Some(old)
                );
                assert_eq!(models.fast.queries.load(Ordering::Relaxed), 1);
                assert_eq!(models.quality.queries.load(Ordering::Relaxed), 1);
                assert_eq!(
                    generation_count(&root),
                    2,
                    "the failed candidate stays inactive"
                );
            });
        }
    }
}

#[cfg(all(test, unix, feature = "embedded-models"))]
mod embedded_acquisition_tests {
    use super::*;

    #[test]
    fn search_recovery_acquisition_leaves_exact_bundled_defaults_for_offline_materialization() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let mut different_revision = ModelManifest::potion_128m();
        different_revision.revision = "b".repeat(40);
        let missing = FsfsRuntime::missing_search_recovery_models(
            vec![
                ModelManifest::potion_128m(),
                ModelManifest::minilm_v2(),
                ModelManifest::potion_base_8m(),
                ModelManifest::minilm_v2_native(),
                ModelManifest::ms_marco_reranker(),
                different_revision,
            ],
            &root,
        )
        .unwrap();
        assert_eq!(
            missing
                .iter()
                .map(|(manifest, _)| manifest.id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "potion-base-8m",
                "all-minilm-l6-v2-native",
                FSFS_RERANKER_MODEL_ID,
                "potion-multilingual-128m",
            ]
        );
        assert!(
            !root.exists(),
            "planning neither downloads nor materializes bundled bytes"
        );
    }
}
