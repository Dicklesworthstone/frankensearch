//! Explicit source rebuild followed by one search against its published reader.
//!
//! This is the CLI composition of the cooperative retained-recovery APIs. It
//! does not execute the canonical planner's capability-blocked acquisition or
//! fixed-authority actions, infer consent from an error, or retry a failed query.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::SearchResult;
use frankensearch_index::generation_root::GENERATION_ROOT_AUTHORITY_FILE_NAME;

use super::super::{PreparedRetainedRecovery, retained_recovery_error};
use super::{
    FsfsRuntime, GenerationPublication, RetainedSearchReader, canonical_store_root,
    complete_entry_exists, require_durable_publication, retained_search_checkpoint,
};

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
        let mut recovery = self.clone();
        recovery.cli_input.recover_source = None;
        recovery.cli_input.target_path = Some(source.clone());
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
                    .join(crate::runtime::FSFS_EXPLAIN_SESSION_FILE)
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
            queries: AtomicUsize,
            on_query: Mutex<Option<QueryHook>>,
        }

        impl Embedder for RecoveryProducer {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                Ok(&self.identity)
            }

            fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                Box::pin(async move {
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
