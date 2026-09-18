use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use crate::{ModelCategory, SearchError, SearchFuture};

#[derive(Clone, Copy)]
pub(super) enum Reply { Correct, Short, Foreign, NonFinite, WrongDimension, Cancel, Pending, FailSecond }

pub(super) struct Provider {
    identity: EmbeddingIdentityBundleV1,
    reply: Reply,
    batches: AtomicUsize,
    pub(super) queries: AtomicUsize,
    max_batch: AtomicUsize,
    dropped: AtomicUsize,
}

impl Provider {
    pub(super) fn new(name: &str, dimension: u32, reply: Reply) -> Self {
        Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension), reply,
            queries: AtomicUsize::new(0), batches: AtomicUsize::new(0), max_batch: AtomicUsize::new(0), dropped: AtomicUsize::new(0),
        }
    }
    fn values(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(text == "vertical")] = 1.0;
        values
    }
}

struct DropCount<'a>(&'a AtomicUsize);
impl Drop for DropCount<'_> {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            Ok(self.values(text))
        })
    }
    fn embed_batch_bound<'a>(&'a self, cx: &'a Cx, texts: &'a [&'a str]) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let call = self.batches.fetch_add(1, Ordering::SeqCst);
            self.max_batch.fetch_max(texts.len(), Ordering::SeqCst);
            let _guard = DropCount(&self.dropped);
            if matches!(self.reply, Reply::Pending) { return std::future::pending().await; }
            if matches!(self.reply, Reply::FailSecond) && call == 1 {
                return Err(invalid("test.batch", "failed", "second batch failed"));
            }
            let mut outputs: Vec<_> = texts.iter().map(|text| IdentityBoundEmbedding {
                values: self.values(text), identity: self.identity.clone(),
            }).collect();
            match self.reply {
                Reply::Short => { let _ = outputs.pop(); }
                Reply::Foreign => outputs.last_mut().unwrap().identity.producer.backend = "different".to_owned(),
                Reply::NonFinite => outputs.last_mut().unwrap().values[0] = f32::NAN,
                Reply::WrongDimension => { let _ = outputs.last_mut().unwrap().values.pop(); }
                Reply::Cancel => { cx.cancel_with(asupersync::CancelKind::User, Some("cancel native build")); }
                _ => {},
            }
            Ok(outputs)
        })
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { Ok(&self.identity) }
    fn id(&self) -> &str { &self.identity.space.logical_model_id }
    fn model_name(&self) -> &str { self.id() }
    fn dimension(&self) -> usize { usize::try_from(self.identity.space.dimension).unwrap() }
    fn is_semantic(&self) -> bool { false }
    fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
}

pub(super) fn generation() -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(9, [0x46; 16]).unwrap()
}
pub(super) fn documents() -> Vec<IndexableDocument> {
    [ ("z", "horizontal"), ("b", "vertical"), ("a", "horizontal"), ("x", "horizontal"), ("c", "horizontal") ]
        .into_iter().map(|(id, text)| IndexableDocument::new(id, text)).collect()
}
fn builder(path: &Path, provider: &Arc<Provider>) -> NativeIndexBuilder {
    NativeIndexBuilder::new(path, generation(), provider.clone()).unwrap().add_documents(documents())
}

#[test]
fn native_document_build_batches_both_tiers_and_retains_the_source_cohort() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&dir.path().join("generation"), &fast)
            .with_quality_embedder(quality.clone()).unwrap()
            .with_batch_size(2).unwrap().build(&cx).await.unwrap();
        assert_eq!(fast.batches.load(Ordering::SeqCst), 3);
        assert_eq!(quality.batches.load(Ordering::SeqCst), 3);
        assert_eq!(fast.max_batch.load(Ordering::SeqCst), 2);
        assert_eq!(quality.max_batch.load(Ordering::SeqCst), 2);
        assert_eq!(index.documents().iter().map(|d| d.id.as_str()).collect::<Vec<_>>(), ["a", "b", "c", "x", "z"]);
        assert_eq!(index.fast().index().live_count(), 5);
        assert_eq!(index.quality().unwrap().index().live_count(), 5);
        assert_eq!(index.fast().index().owner_witness().generation, index.quality().unwrap().index().owner_witness().generation);
        assert_eq!(index.document("b").unwrap().content, "vertical");
        assert!(index.document("absent").is_none());
        drop(dir); // Search and source hydration must not reopen these paths.
        assert_eq!(index.fast().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
        assert_eq!(index.quality().unwrap().search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
    });
}

#[test]
fn built_native_sidecars_reopen_with_the_exact_written_v2_binding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for precision in [NativeBuildPrecision::F32, NativeBuildPrecision::F16] {
            let dir = tempfile::tempdir().unwrap();
            let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let index = builder(&dir.path().join("generation"), &provider)
                .with_fast_storage(precision, NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 7 })
                .build(&cx).await.unwrap();
            let tier = index.fast();
            let bytes: Arc<[u8]> = std::fs::read(tier.vector_path()).unwrap().into();
            let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, tier.binding()).unwrap());
            assert_eq!(owner.witness(), tier.index().owner_witness());
            let reopened = NativeAnnIndex::load(&cx, owner, tier.graph_path().unwrap()).unwrap();
            assert_eq!(reopened.search_text(&cx, provider.as_ref(), "vertical", 5, None).await.unwrap(), tier.search(&cx, "vertical", 5).await.unwrap());
        }
    });
}

#[test]
fn document_order_does_not_change_vector_bytes_or_row_identity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let first = builder(&dir.path().join("one"), &provider).build(&cx).await.unwrap();
        let mut docs = documents(); docs.reverse();
        let second = NativeIndexBuilder::new(dir.path().join("two"), generation(), provider.clone()).unwrap()
            .add_documents(docs).build(&cx).await.unwrap();
        assert_eq!(std::fs::read(first.fast().vector_path()).unwrap(), std::fs::read(second.fast().vector_path()).unwrap());
        assert_eq!(first.fast().search(&cx, "vertical", 5).await.unwrap(), second.fast().search(&cx, "vertical", 5).await.unwrap());
    });
}

#[test]
fn invalid_source_and_configuration_start_no_provider_or_directory_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("generation");
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        assert!(builder(&path, &provider).with_batch_size(0).is_err());
        for id in ["", "a"] {
            assert!(builder(&path, &provider).add_document(IndexableDocument::new(id, "extra")).build(&cx).await.is_err());
            assert!(!path.exists());
        }
        let params = HnswParams { ef_search: 0, ..HnswParams::default() };
        assert!(builder(&path, &provider).with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Hnsw { params, seed: 7 }).build(&cx).await.is_err());
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
        assert!(!path.exists());
    });
}

#[test]
fn existing_destination_is_never_adopted_or_truncated() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("fast.fsvi"), b"existing vector bytes").unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        assert!(builder(dir.path(), &provider).build(&cx).await.is_err());
        assert_eq!(std::fs::read(dir.path().join("fast.fsvi")).unwrap(), b"existing vector bytes");
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn malformed_batch_or_later_failure_returns_no_partial_generation() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for reply in [Reply::Short, Reply::Foreign, Reply::NonFinite, Reply::WrongDimension, Reply::FailSecond] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("generation");
            let provider = Arc::new(Provider::new("fast", 2, reply));
            assert!(builder(&path, &provider).with_batch_size(2).unwrap().build(&cx).await.is_err());
            // The failed attempt may leave files for diagnostics. Directory
            // existence is not evidence of a completed, selectable generation.
            assert!(path.is_dir());
        }
    });
}

#[test]
fn required_quality_failure_never_degrades_to_a_fast_only_build() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Foreign));
        assert!(builder(&dir.path().join("generation"), &fast).with_quality_embedder(quality.clone()).unwrap().build(&cx).await.is_err());
        assert_eq!(fast.batches.load(Ordering::SeqCst), 1);
        assert_eq!(quality.batches.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn actual_context_cancellation_dominates_successful_batch_output() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Cancel));
        let result = builder(&dir.path().join("generation"), &provider).build(&cx).await;
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn dropping_pending_build_drops_provider_work_without_retry() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("generation");
        let provider = Arc::new(Provider::new("fast", 2, Reply::Pending));
        let mut future = Box::pin(builder(&path, &provider).build(&cx));
        assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
        drop(future);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(provider.dropped.load(Ordering::SeqCst), 1);
        assert!(path.is_dir());
    });
}

#[test]
fn empty_cohort_is_admitted_without_empty_provider_batches() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Pending));
        let index = NativeIndexBuilder::new(dir.path().join("empty"), generation(), provider.clone()).unwrap().build(&cx).await.unwrap();
        assert!(index.documents().is_empty());
        assert!(index.fast().search(&cx, "anything", 10).await.unwrap().is_empty());
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn source_join_uses_document_ids_not_fsvi_physical_sort_positions() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("generation"), &provider)
            .with_batch_size(2).unwrap().build(&cx).await.unwrap();
        let owner = &index.fast.index.owner;
        let physical: Vec<_> = (0..owner.record_count())
            .map(|position| owner.row(position).unwrap().doc_id().to_owned())
            .collect();
        let source: Vec<_> = index.documents().iter().map(|doc| doc.id.clone()).collect();
        assert_ne!(physical, source, "fixture must distinguish physical hash order from source ID order");
        for position in 0..owner.record_count() {
            let row = owner.row(position).unwrap();
            let source = index.document(row.doc_id()).unwrap();
            assert_eq!(owner.vector_at_f32(position).unwrap(), provider.values(&source.content));
        }
    });
}
