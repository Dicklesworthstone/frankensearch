use super::*;
use std::future::Future;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use crate::native_ann::{NativeRetrievalMode, NativeSearchPhase};
use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval};
use crate::{SearchError, RerankDocument, RerankScore, SearchFuture};
use super::super::tests::{Provider, Reply, generation};
use frankensearch_index::native_hnsw::HnswParams;

#[derive(Default)]
struct RerankProbe {
    calls: AtomicUsize,
    dropped: AtomicUsize,
    documents: Mutex<Vec<RerankDocument>>,
    pending: bool,
    cancel: bool,
}
struct DropCount<'a>(&'a AtomicUsize);
impl Drop for DropCount<'_> {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}
impl Reranker for RerankProbe {
    fn rerank<'a>(&'a self, cx: &'a Cx, _query: &'a str, documents: &'a [RerankDocument]) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _guard = DropCount(&self.dropped);
            *self.documents.lock().unwrap() = documents.to_vec();
            if self.pending { return std::future::pending().await; }
            if self.cancel { cx.cancel_with(asupersync::CancelKind::User, Some("cancel built hybrid rerank")); }
            let mut scores: Vec<_> = documents.iter().enumerate().map(|(rank, doc)| RerankScore {
                doc_id: doc.doc_id.clone(), original_rank: rank,
                score: if doc.doc_id == "b" { 1.0 } else { 0.1 }, raw_logit: None,
            }).collect();
            scores.sort_by(|a, b| b.score.total_cmp(&a.score));
            Ok(scores)
        })
    }
    fn id(&self) -> &str { "retained-source-test-reranker" }
    fn model_name(&self) -> &str { self.id() }
}

fn builder(path: &std::path::Path, fast: &Arc<Provider>) -> NativeIndexBuilder {
    NativeIndexBuilder::new(path, generation(), fast.clone()).unwrap()
        .add_documents(super::super::tests::documents())
}

#[test]
fn real_quill_and_native_vectors_are_built_from_the_complete_same_cohort() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast)
            .with_fast_storage(NativeBuildPrecision::F16, NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 7 })
            .with_quality_embedder(quality.clone()).unwrap()
            .with_batch_size(2).unwrap().build_hybrid(&cx).await.unwrap();
        assert_eq!(index.lexical().doc_count().unwrap(), 5);
        assert_eq!(index.vectors().fast().index().retrieval_mode(), NativeRetrievalMode::Ann);
        assert!(matches!(index.vectors().quality().unwrap().index().retrieval_mode(), NativeRetrievalMode::Exact { .. }));
        let lexical = index.lexical().search(&cx, "vertical", 10).await.unwrap();
        assert_eq!(lexical.len(), 1);
        assert_eq!(lexical[0].doc_id, "b");
        assert_eq!(index.search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
        assert_eq!(index.search_refined(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
        assert_eq!(index.search_quality(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
    });
}

#[test]
fn build_to_progressive_path_keeps_quality_inference_lazy() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast).with_quality_embedder(quality.clone()).unwrap().build_hybrid(&cx).await.unwrap();
        let mut stream = index.progressive(&cx, "vertical", 1).unwrap();
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        assert!(matches!(stream.next_phase().await.unwrap(), Some(NativeSearchPhase::Initial { .. })));
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        assert!(matches!(stream.next_phase().await.unwrap(), Some(NativeSearchPhase::Refined { .. })));
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        assert!(stream.next_phase().await.unwrap().is_none());
    });
}

#[test]
fn reranking_promotes_an_undisplayed_document_using_build_time_text_not_current_files() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.txt");
        std::fs::write(&source, "vertical").unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let mut docs = super::super::tests::documents();
        docs.iter_mut().find(|d| d.id == "b").unwrap().content = std::fs::read_to_string(&source).unwrap();
        let index = NativeIndexBuilder::new(dir.path().join("hybrid"), generation(), fast.clone()).unwrap()
            .add_documents(docs).build_hybrid(&cx).await.unwrap();
        std::fs::write(&source, "replacement text that was never indexed").unwrap();
        std::fs::remove_file(index.vectors().fast().vector_path()).unwrap();
        let reranker = RerankProbe::default();
        let mut stream = index.progressive_with_reranker(&cx, "horizontal", 1, &reranker, 5).unwrap();
        let Some(NativeSearchPhase::Initial { results, .. }) = stream.next_phase().await.unwrap() else { panic!("initial"); };
        assert_eq!(results.len(), 1);
        assert_ne!(results[0].doc_id, "b");
        assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
        let Some(NativeSearchPhase::Reranked { results, .. }) = stream.next_phase().await.unwrap() else { panic!("reranked"); };
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[0].rerank_score, Some(1.0));
        let inputs = reranker.documents.lock().unwrap();
        assert_eq!(inputs.len(), 5);
        assert_eq!(inputs.iter().find(|doc| doc.doc_id == "b").unwrap().text, "vertical");
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn stopping_after_initial_results_never_invokes_the_reranker() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast).build_hybrid(&cx).await.unwrap();
        let reranker = RerankProbe::default();
        let mut stream = index.progressive_with_reranker(&cx, "vertical", 1, &reranker, 5).unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        drop(stream);
        assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
        assert!(reranker.documents.lock().unwrap().is_empty());
    });
}

#[test]
fn dropping_pending_built_generation_rerank_releases_work_without_retry() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast).build_hybrid(&cx).await.unwrap();
        let reranker = RerankProbe { pending: true, ..RerankProbe::default() };
        let mut stream = index.progressive_with_reranker(&cx, "vertical", 1, &reranker, 5).unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let mut future = Box::pin(stream.next_phase());
        assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
        drop(future);
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(reranker.calls.load(Ordering::SeqCst), 1);
        assert_eq!(reranker.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(index.search(&cx, "vertical", 1).await.unwrap()[0].doc_id, "b");
    });
}

#[test]
fn context_cancellation_cannot_publish_a_final_reranked_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast).build_hybrid(&cx).await.unwrap();
        let reranker = RerankProbe { cancel: true, ..RerankProbe::default() };
        let mut stream = index.progressive_with_reranker(&cx, "vertical", 1, &reranker, 5).unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        assert!(matches!(stream.next_phase().await, Err(SearchError::Cancelled { .. })));
        assert!(stream.is_finished());
    });
}

#[test]
fn missing_quality_and_zero_k_do_not_start_unrequested_models() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast).build_hybrid(&cx).await.unwrap();
        assert!(matches!(index.search_quality(&cx, "vertical", 0).await, Err(SearchError::InvalidConfig { .. })));
        let reranker = RerankProbe::default();
        let mut stream = index.progressive_with_reranker(&cx, "vertical", 0, &reranker, 5).unwrap();
        assert!(matches!(stream.next_phase().await.unwrap(), Some(NativeSearchPhase::Initial { results, .. }) if results.is_empty()));
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn required_vector_failure_never_creates_a_lexical_only_hybrid_success() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hybrid");
        let fast = Arc::new(Provider::new("fast", 2, Reply::Foreign));
        assert!(builder(&path, &fast).build_hybrid(&cx).await.is_err());
        assert!(!path.join("lexical").exists());
    });
}
