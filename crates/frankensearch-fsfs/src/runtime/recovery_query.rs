// Recovery readers are read-only but the shared ranking engine still requests
// raw vectors. At this boundary, raw requests must use the provider's bound
// operation; attaching advertised metadata after raw inference is not evidence.
use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::{
    Embedder, EmbeddingIdentityBundleV1, IdentityBoundEmbedding, ModelCategory, SearchError,
    SearchFuture, SearchResult,
};

fn identity_error() -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "fsfs.recovery.query".to_owned(),
        reason: "query producer or response identity differs from the admitted recovery tier"
            .to_owned(),
    }
}

fn response_error() -> SearchError {
    SearchError::InvalidConfig {
        field: "complete_generation.recovery.query_response".to_owned(),
        value: String::new(),
        reason: "recovery inference returned an invalid identity, representation, dimension or non-finite vector"
            .to_owned(),
    }
}

/// Freeze a producer that the recovery opener has admitted to its own tier.
/// The opener rechecks the resulting reader against the stored generation.
pub(super) fn bind(inner: Arc<dyn Embedder>) -> SearchResult<Arc<dyn Embedder>> {
    let identity = inner.identity().map_err(|_| identity_error())?.clone();
    identity.validate().map_err(|_| identity_error())?;
    let dimension = usize::try_from(identity.space.dimension).map_err(|_| identity_error())?;
    if dimension != inner.dimension() {
        return Err(identity_error());
    }
    let id = inner.id().to_owned();
    Ok(Arc::new(RecoveryQueryEmbedder {
        inner,
        identity,
        dimension,
        id,
    }))
}

struct RecoveryQueryEmbedder {
    inner: Arc<dyn Embedder>,
    identity: EmbeddingIdentityBundleV1,
    dimension: usize,
    id: String,
}

impl RecoveryQueryEmbedder {
    fn validate_active(&self) -> SearchResult<()> {
        if self.inner.identity().map_err(|_| identity_error())? != &self.identity
            || self.inner.dimension() != self.dimension
            || self.inner.id() != self.id
        {
            return Err(identity_error());
        }
        Ok(())
    }

    async fn checked_response(&self, cx: &Cx, text: &str) -> SearchResult<IdentityBoundEmbedding> {
        super::retained_search_checkpoint(cx)?;
        self.validate_active()?;
        let outcome = self.inner.embed_bound(cx, text).await;
        // Cancellation outranks both a provider error and a completed response.
        super::retained_search_checkpoint(cx)?;
        let response = outcome?;
        self.validate_active()?;
        // Overrides need not have called the trait's default validator. Do not
        // leak arbitrary identity fields or vector coordinates in diagnostics.
        response.validate().map_err(|_| response_error())?;
        if response.identity != self.identity {
            return Err(identity_error());
        }
        super::retained_search_checkpoint(cx)?;
        Ok(response)
    }
}

impl Embedder for RecoveryQueryEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { Ok(self.checked_response(cx, text).await?.values) })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(self.checked_response(cx, text))
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.validate_active()?;
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> frankensearch_core::traits::ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        self.inner.supports_mrl()
    }
}

#[cfg(all(test, unix, not(feature = "embedded-models")))]
mod tests {
    use super::super::{
        FsfsRuntime, set_test_fast_embedder, set_test_quality_embedder,
        test_fast_embedder_override, test_quality_embedder_override,
    };
    use super::*;
    use crate::generation_store::{
        COMPLETE_GENERATION_POINTER, GenerationPublication, PublishedGeneration,
    };
    use crate::output_schema::SearchOutputPhase;
    use crate::{CliCommand, CliInput, FsfsConfig};
    use std::collections::BTreeMap;
    use std::fs;
    use std::future::Future;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

    const QUERY: &str = "how does durable recovery keep documents searchable";
    const GOOD: u8 = 0;
    const FOREIGN_PRODUCER: u8 = 1;
    const FOREIGN_SPACE: u8 = 2;
    const STORAGE: u8 = 3;
    const SHORT: u8 = 4;
    const NAN: u8 = 5;
    const INFINITY: u8 = 6;
    const DRIFT: u8 = 7;
    const CANCEL_OK: u8 = 8;
    const CANCEL_ERROR: u8 = 9;
    const PENDING: u8 = 10;

    struct Producer {
        identity: EmbeddingIdentityBundleV1,
        changed_identity: EmbeddingIdentityBundleV1,
        changed: AtomicBool,
        mode: AtomicU8,
        raw_calls: AtomicUsize,
        bound_calls: AtomicUsize,
        category: ModelCategory,
    }

    impl Producer {
        fn new(id: &str, category: ModelCategory) -> Self {
            let identity = EmbeddingIdentityBundleV1::explicit_test_model(id, 3);
            let mut changed_identity = identity.clone();
            changed_identity
                .producer
                .implementation_revision
                .push_str("-private-canary");
            Self {
                identity,
                changed_identity,
                changed: AtomicBool::new(false),
                mode: AtomicU8::new(GOOD),
                raw_calls: AtomicUsize::new(0),
                bound_calls: AtomicUsize::new(0),
                category,
            }
        }

        fn reset_calls(&self) {
            self.raw_calls.store(0, Ordering::Relaxed);
            self.bound_calls.store(0, Ordering::Relaxed);
        }
    }

    impl Embedder for Producer {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.changed.load(Ordering::Relaxed) {
                &self.changed_identity
            } else {
                &self.identity
            })
        }

        fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.raw_calls.fetch_add(1, Ordering::Relaxed);
                Ok(vec![1.0, 0.0, 0.0])
            })
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            _: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.bound_calls.fetch_add(1, Ordering::Relaxed);
                let mut response = IdentityBoundEmbedding {
                    values: vec![1.0, 0.0, 0.0],
                    identity: self.identity.clone(),
                };
                match self.mode.load(Ordering::Relaxed) {
                    FOREIGN_PRODUCER => response.identity = self.changed_identity.clone(),
                    FOREIGN_SPACE => response
                        .identity
                        .space
                        .logical_model_id
                        .push_str("-private-canary"),
                    STORAGE => "fsvi-v2".clone_into(&mut response.identity.storage.format),
                    SHORT => {
                        response.values.pop();
                    }
                    NAN => response.values[1] = f32::NAN,
                    INFINITY => response.values[1] = f32::INFINITY,
                    DRIFT => self.changed.store(true, Ordering::Relaxed),
                    CANCEL_OK => cx.set_cancel_requested(true),
                    CANCEL_ERROR => {
                        cx.set_cancel_requested(true);
                        return Err(SearchError::EmbeddingFailed {
                            model: "query-fixture".to_owned(),
                            source: "cancelled alongside backend failure".into(),
                        });
                    }
                    PENDING => std::future::pending::<()>().await,
                    _ => {}
                }
                Ok(response)
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

    struct Models {
        old_fast: Option<Arc<dyn Embedder>>,
        old_quality: Option<Arc<dyn Embedder>>,
        fast: Arc<Producer>,
        quality: Arc<Producer>,
    }

    impl Models {
        fn install() -> Self {
            let models = Self {
                old_fast: test_fast_embedder_override(),
                old_quality: test_quality_embedder_override(),
                fast: Arc::new(Producer::new("guard-fast", ModelCategory::StaticEmbedder)),
                quality: Arc::new(Producer::new(
                    "guard-quality",
                    ModelCategory::TransformerEmbedder,
                )),
            };
            set_test_fast_embedder(Some(models.fast.clone()));
            set_test_quality_embedder(Some(models.quality.clone()));
            models
        }

        fn reset_calls(&self) {
            self.fast.reset_calls();
            self.quality.reset_calls();
        }
    }

    impl Drop for Models {
        fn drop(&mut self) {
            set_test_fast_embedder(self.old_fast.take());
            set_test_quality_embedder(self.old_quality.take());
        }
    }

    fn run<F, Fut>(test: F)
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

    fn fixture(parent: &Path) -> (FsfsRuntime, PathBuf) {
        let source = parent.join("source");
        let root = parent.join("store");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("alpha.md"), QUERY).unwrap();
        let mut config = FsfsConfig::default();
        "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
        "guard-quality".clone_into(&mut config.indexing.quality_model);
        config.indexing.offline = true;
        config.search.fast_only = false;
        config.search.rerank = false;
        config.search.quality_timeout_ms = 5_000;
        let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
            command: CliCommand::Index,
            target_path: Some(source),
            index_dir: Some(root.clone()),
            quiet: true,
            ..CliInput::default()
        });
        (runtime, root)
    }

    async fn publish(runtime: &FsfsRuntime, cx: &Cx, root: &Path) -> PublishedGeneration {
        let publication = runtime.rebuild_retained_generation(cx, root).await.unwrap();
        let GenerationPublication::Durable(generation) = publication else {
            panic!("fixture must publish durably"); // ubs:ignore — cfg(test) assertion.
        };
        generation
    }

    fn bytes(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut result = BTreeMap::new();
        let mut dirs = vec![root.to_path_buf()];
        while let Some(dir) = dirs.pop() {
            for entry in fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    dirs.push(entry.path());
                } else {
                    result.insert(
                        entry.path().strip_prefix(root).unwrap().to_path_buf(),
                        fs::read(entry.path()).unwrap(),
                    );
                }
            }
        }
        result
    }

    #[test]
    fn bound_guard_uses_the_providers_bound_operation_once() {
        run(|cx| async move {
            let producer = Arc::new(Producer::new("guard-unit", ModelCategory::StaticEmbedder));
            let guard = bind(producer.clone()).unwrap();
            assert_eq!(guard.embed(&cx, QUERY).await.unwrap(), [1.0, 0.0, 0.0]);
            let response = guard.embed_bound(&cx, QUERY).await.unwrap();
            assert_eq!(response.identity, producer.identity);
            assert_eq!(response.values, [1.0, 0.0, 0.0]);
            assert_eq!(producer.raw_calls.load(Ordering::Relaxed), 0);
            assert_eq!(producer.bound_calls.load(Ordering::Relaxed), 2);
        });
    }

    #[test]
    fn bound_guard_rejects_foreign_and_malformed_responses_without_diagnostic_leaks() {
        run(|cx| async move {
            let producer = Arc::new(Producer::new("guard-unit", ModelCategory::StaticEmbedder));
            let guard = bind(producer.clone()).unwrap();
            for mode in [FOREIGN_PRODUCER, FOREIGN_SPACE, STORAGE, SHORT, NAN, INFINITY] {
                producer.mode.store(mode, Ordering::Relaxed);
                let error = guard.embed(&cx, QUERY).await.unwrap_err();
                assert!(!error.to_string().contains("private-canary"));
                assert!(matches!(
                    error,
                    SearchError::UnverifiableRemoteSpace { .. } | SearchError::InvalidConfig { .. }
                ));
            }
            assert_eq!(producer.raw_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn bound_guard_refuses_provider_drift_before_and_during_inference() {
        run(|cx| async move {
            let producer = Arc::new(Producer::new("guard-unit", ModelCategory::StaticEmbedder));
            let guard = bind(producer.clone()).unwrap();
            producer.changed.store(true, Ordering::Relaxed);
            assert!(matches!(
                guard.embed(&cx, QUERY).await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            assert_eq!(producer.bound_calls.load(Ordering::Relaxed), 0);
            producer.changed.store(false, Ordering::Relaxed);
            producer.mode.store(DRIFT, Ordering::Relaxed);
            assert!(matches!(
                guard.embed(&cx, QUERY).await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            assert_eq!(producer.bound_calls.load(Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn cancellation_beats_both_response_success_and_backend_error() {
        run(|cx| async move {
            let producer = Arc::new(Producer::new("guard-unit", ModelCategory::StaticEmbedder));
            let guard = bind(producer.clone()).unwrap();
            for mode in [CANCEL_OK, CANCEL_ERROR] {
                producer.mode.store(mode, Ordering::Relaxed);
                assert!(matches!(
                    guard.embed(&cx, QUERY).await,
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
            }
            assert_eq!(producer.raw_calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn receipt_reader_checks_both_bound_query_responses_and_preserves_files() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            let mut reader = runtime
                .open_retained_search_from_receipt(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            models.reset_calls();
            let before = bytes(&root);
            let result = reader.search(&cx, QUERY, 10).await.unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].phase, SearchOutputPhase::Initial);
            assert_eq!(result[1].phase, SearchOutputPhase::Refined);
            assert!(result[1].semantic_blend.is_some());
            for producer in [&models.fast, &models.quality] {
                assert_eq!(producer.bound_calls.load(Ordering::Relaxed), 1);
                assert_eq!(producer.raw_calls.load(Ordering::Relaxed), 0);
            }
            assert_eq!(bytes(&root), before);
        });
    }

    #[test]
    fn foreign_fast_response_cannot_emit_initial_or_change_selection() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"repair still required").unwrap();
            let mut reader = runtime
                .open_retained_search_from_receipt(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            models.reset_calls();
            models.fast.mode.store(FOREIGN_PRODUCER, Ordering::Relaxed);
            let before = bytes(&root);
            let mut phases = Vec::new();
            let result = reader
                .search_with_phase_sink(&cx, QUERY, 10, &mut |payload| {
                    phases.push(payload.phase);
                    Ok(())
                })
                .await;
            assert!(matches!(result, Err(SearchError::UnverifiableRemoteSpace { .. })));
            assert!(phases.is_empty());
            assert_eq!(models.quality.bound_calls.load(Ordering::Relaxed), 0);
            assert_eq!(bytes(&root), before);
        });
    }

    #[test]
    fn foreign_quality_response_preserves_initial_without_successful_refinement() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            let mut plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            models.reset_calls();
            models.quality.mode.store(FOREIGN_PRODUCER, Ordering::Relaxed);
            let before = bytes(&root);
            let result = plan.search(&cx, QUERY, 10).await.unwrap();
            assert_eq!(result.len(), 2);
            assert_eq!(result[0].phase, SearchOutputPhase::Initial);
            assert_eq!(result[1].phase, SearchOutputPhase::RefinementFailed);
            assert!(!FsfsRuntime::search_payloads_cacheable(&result));
            assert_eq!(models.fast.bound_calls.load(Ordering::Relaxed), 1);
            assert_eq!(models.quality.bound_calls.load(Ordering::Relaxed), 1);
            assert_eq!(bytes(&root), before);
        });
    }

    #[test]
    fn returned_restored_reader_keeps_its_response_guard() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            let plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            let (_, mut reader) = plan.restore(&cx).unwrap();
            models.fast.mode.store(FOREIGN_PRODUCER, Ordering::Relaxed);
            assert!(matches!(
                reader.search(&cx, QUERY, 10).await,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        });
    }

    #[test]
    fn failed_fast_or_quality_preview_cannot_authorize_restoration() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"preview must complete").unwrap();
            for fast in [true, false] {
                let mut plan = runtime
                    .prepare_retained_recovery(
                        &cx,
                        &root,
                        generation.id(),
                        generation.manifest_sha256(),
                    )
                    .await
                    .unwrap();
                let producer = if fast { &models.fast } else { &models.quality };
                producer.mode.store(FOREIGN_PRODUCER, Ordering::Relaxed);
                let before = bytes(&root);
                let result = plan.search(&cx, QUERY, 10).await;
                if fast {
                    assert!(matches!(
                        result,
                        Err(SearchError::UnverifiableRemoteSpace { .. })
                    ));
                } else {
                    assert_eq!(
                        result.unwrap().last().unwrap().phase,
                        SearchOutputPhase::RefinementFailed
                    );
                }
                // A now-healthy advertised producer cannot erase a failed preview.
                producer.mode.store(GOOD, Ordering::Relaxed);
                let error = plan.restore(&cx).unwrap_err();
                assert!(error.to_string().contains("preview"));
                assert_eq!(bytes(&root), before);
            }
        });
    }

    #[test]
    fn successful_preview_retry_restores_permission_without_repreparing() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"retry preview").unwrap();
            let mut plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            models.quality.mode.store(FOREIGN_PRODUCER, Ordering::Relaxed);
            assert_eq!(
                plan.search(&cx, QUERY, 10).await.unwrap().last().unwrap().phase,
                SearchOutputPhase::RefinementFailed
            );
            models.quality.mode.store(GOOD, Ordering::Relaxed);
            assert_eq!(
                plan.search(&cx, QUERY, 10).await.unwrap().last().unwrap().phase,
                SearchOutputPhase::Refined
            );
            let (publication, reader) = plan.restore(&cx).unwrap();
            assert!(matches!(publication, GenerationPublication::Durable(_)));
            assert_eq!(reader.generation(), &generation);
        });
    }

    #[test]
    fn cancelled_preview_cannot_restore_after_cancellation_is_cleared() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"cancelled preview").unwrap();
            let mut plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            let before = bytes(&root);
            models.quality.mode.store(CANCEL_ERROR, Ordering::Relaxed);
            assert!(matches!(
                plan.search(&cx, QUERY, 10).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            models.quality.mode.store(GOOD, Ordering::Relaxed);
            assert!(plan.restore(&cx).unwrap_err().to_string().contains("preview"));
            assert_eq!(bytes(&root), before);
        });
    }

    #[test]
    fn dropped_preview_revokes_prior_success_but_unpolled_preview_has_no_effect() {
        run(|cx| async move {
            let models = Models::install();
            let directory = tempfile::tempdir().unwrap();
            let (runtime, root) = fixture(directory.path());
            let generation = publish(&runtime, &cx, &root).await;
            fs::write(root.join(COMPLETE_GENERATION_POINTER), b"dropped preview").unwrap();
            let mut plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            assert_eq!(
                plan.search(&cx, QUERY, 10).await.unwrap().last().unwrap().phase,
                SearchOutputPhase::Refined
            );
            models.reset_calls();
            models.fast.mode.store(PENDING, Ordering::Relaxed);
            let before = bytes(&root);
            let mut preview = Box::pin(plan.search(&cx, QUERY, 10));
            std::future::poll_fn(|task_cx| {
                assert!(preview.as_mut().poll(task_cx).is_pending());
                if models.fast.bound_calls.load(Ordering::Relaxed) > 0 {
                    std::task::Poll::Ready(())
                } else {
                    std::task::Poll::Pending
                }
            })
            .await;
            drop(preview);
            models.fast.mode.store(GOOD, Ordering::Relaxed);
            assert!(plan.restore(&cx).unwrap_err().to_string().contains("preview"));
            assert_eq!(bytes(&root), before);

            let mut plan = runtime
                .prepare_retained_recovery(
                    &cx,
                    &root,
                    generation.id(),
                    generation.manifest_sha256(),
                )
                .await
                .unwrap();
            // Async construction alone must not revoke an explicit no-preview restore.
            drop(plan.search(&cx, QUERY, 10));
            assert!(matches!(
                plan.restore(&cx).unwrap().0,
                GenerationPublication::Durable(_)
            ));
        });
    }
}
