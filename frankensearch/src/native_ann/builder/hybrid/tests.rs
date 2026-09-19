use super::*;
use std::future::Future;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use super::super::tests::{Provider, Reply, generation};
use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval};
use crate::native_ann::{NativeRetrievalMode, NativeSearchPhase};
use crate::{RerankDocument, RerankScore, SearchError, SearchFuture};
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
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl Reranker for RerankProbe {
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        _query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _guard = DropCount(&self.dropped);
            *self.documents.lock().unwrap() = documents.to_vec();
            if self.pending {
                return std::future::pending().await;
            }
            if self.cancel {
                cx.cancel_with(
                    asupersync::CancelKind::User,
                    Some("cancel built hybrid rerank"),
                );
            }
            let mut scores: Vec<_> = documents
                .iter()
                .enumerate()
                .map(|(rank, doc)| RerankScore {
                    doc_id: doc.doc_id.clone(),
                    original_rank: rank,
                    score: if doc.doc_id == "b" { 1.0 } else { 0.1 },
                    raw_logit: None,
                })
                .collect();
            scores.sort_by(|a, b| b.score.total_cmp(&a.score));
            Ok(scores)
        })
    }
    fn id(&self) -> &str {
        "retained-source-test-reranker"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
}

fn builder(path: &std::path::Path, fast: &Arc<Provider>) -> NativeIndexBuilder {
    NativeIndexBuilder::new(path, generation(), fast.clone())
        .unwrap()
        .add_documents(super::super::tests::documents())
}

#[test]
fn real_quill_and_native_vectors_are_built_from_the_complete_same_cohort() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast)
            .with_fast_storage(
                NativeBuildPrecision::F16,
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 7,
                },
            )
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_batch_size(2)
            .unwrap()
            .build_hybrid(&cx)
            .await
            .unwrap();
        assert_eq!(index.lexical().doc_count().unwrap(), 5);
        assert_eq!(
            index.vectors().fast().index().retrieval_mode(),
            NativeRetrievalMode::Ann
        );
        assert!(matches!(
            index.vectors().quality().unwrap().index().retrieval_mode(),
            NativeRetrievalMode::Exact { .. }
        ));
        let lexical = index.lexical().search(&cx, "vertical", 10).await.unwrap();
        assert_eq!(lexical.len(), 1);
        assert_eq!(lexical[0].doc_id, "b");
        assert_eq!(
            index.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
        assert_eq!(
            index.search_refined(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
        assert_eq!(
            index.search_quality(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
    });
}

#[test]
fn build_to_progressive_path_keeps_quality_inference_lazy() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast)
            .with_quality_embedder(quality.clone())
            .unwrap()
            .build_hybrid(&cx)
            .await
            .unwrap();
        let mut stream = index.progressive(&cx, "vertical", 1).unwrap();
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Refined { .. })
        ));
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
        docs.iter_mut().find(|d| d.id == "b").unwrap().content =
            std::fs::read_to_string(&source).unwrap();
        let index = NativeIndexBuilder::new(dir.path().join("hybrid"), generation(), fast.clone())
            .unwrap()
            .add_documents(docs)
            .build_hybrid(&cx)
            .await
            .unwrap();
        std::fs::write(&source, "replacement text that was never indexed").unwrap();
        std::fs::remove_file(index.vectors().fast().vector_path()).unwrap();
        let reranker = RerankProbe::default();
        let mut stream = index
            .progressive_with_reranker(&cx, "horizontal", 1, &reranker, 5)
            .unwrap();
        let Some(NativeSearchPhase::Initial { results, .. }) = stream.next_phase().await.unwrap()
        else {
            panic!("initial");
        };
        assert_eq!(results.len(), 1);
        assert_ne!(results[0].doc_id, "b");
        assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
        let Some(NativeSearchPhase::Reranked { results, .. }) = stream.next_phase().await.unwrap()
        else {
            panic!("reranked");
        };
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[0].rerank_score, Some(1.0));
        let inputs = reranker.documents.lock().unwrap();
        assert_eq!(inputs.len(), 5);
        assert_eq!(
            inputs.iter().find(|doc| doc.doc_id == "b").unwrap().text,
            "vertical"
        );
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn stopping_after_initial_results_never_invokes_the_reranker() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast)
            .build_hybrid(&cx)
            .await
            .unwrap();
        let reranker = RerankProbe::default();
        let mut stream = index
            .progressive_with_reranker(&cx, "vertical", 1, &reranker, 5)
            .unwrap();
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
        let index = builder(&dir.path().join("hybrid"), &fast)
            .build_hybrid(&cx)
            .await
            .unwrap();
        let reranker = RerankProbe {
            pending: true,
            ..RerankProbe::default()
        };
        let mut stream = index
            .progressive_with_reranker(&cx, "vertical", 1, &reranker, 5)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let mut future = Box::pin(stream.next_phase());
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Pending
        ));
        drop(future);
        assert!(stream.is_finished());
        assert!(stream.next_phase().await.unwrap().is_none());
        assert_eq!(reranker.calls.load(Ordering::SeqCst), 1);
        assert_eq!(reranker.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(
            index.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
    });
}

#[test]
fn context_cancellation_cannot_publish_a_final_reranked_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast)
            .build_hybrid(&cx)
            .await
            .unwrap();
        let reranker = RerankProbe {
            cancel: true,
            ..RerankProbe::default()
        };
        let mut stream = index
            .progressive_with_reranker(&cx, "vertical", 1, &reranker, 5)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        assert!(matches!(
            stream.next_phase().await,
            Err(SearchError::Cancelled { .. })
        ));
        assert!(stream.is_finished());
    });
}

#[test]
fn missing_quality_and_zero_k_do_not_start_unrequested_models() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&dir.path().join("hybrid"), &fast)
            .build_hybrid(&cx)
            .await
            .unwrap();
        assert!(matches!(
            index.search_quality(&cx, "vertical", 0).await,
            Err(SearchError::InvalidConfig { .. })
        ));
        let reranker = RerankProbe::default();
        let mut stream = index
            .progressive_with_reranker(&cx, "vertical", 0, &reranker, 5)
            .unwrap();
        assert!(
            matches!(stream.next_phase().await.unwrap(), Some(NativeSearchPhase::Initial { results, .. }) if results.is_empty())
        );
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

#[test]
fn completed_hybrid_releases_writer_lease_but_keeps_its_original_publication() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hybrid");
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
        let original = index.lexical().search(&cx, "vertical", 10).await.unwrap();
        assert_eq!(original.len(), 1);
        assert_eq!(original[0].doc_id, "b");

        // This acquisition fails while a QuillIndex writer remains retained.
        // A successful hybrid build must instead keep a read-only publication.
        let writer = QuillIndex::open(&cx, path.join("lexical"), QuillConfig::default())
            .await
            .expect("serving hybrid must not own the writer lease");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &crate::IndexableDocument::new("new", "newpublication"),
        )
        .await
        .unwrap();
        LexicalWrite::commit(&writer, &cx).await.unwrap();
        assert_eq!(LexicalRead::doc_count(&writer).unwrap(), 6);
        assert_eq!(index.lexical().doc_count().unwrap(), 5);
        assert!(
            index
                .lexical()
                .search(&cx, "newpublication", 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            index.lexical().search(&cx, "vertical", 10).await.unwrap()[0].doc_id,
            "b"
        );
        assert_eq!(
            index.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
            "b"
        );
    });
}

#[test]
fn read_only_hybrid_pins_survive_a_later_lexical_publication_between_phases() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hybrid");
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
        let index = builder(&path, &fast)
            .with_quality_embedder(quality.clone())
            .unwrap()
            .build_hybrid(&cx)
            .await
            .unwrap();
        let reranker = RerankProbe::default();
        let mut stream = index
            .progressive_with_reranker(&cx, "vertical", 1, &reranker, 5)
            .unwrap();
        assert!(matches!(
            stream.next_phase().await.unwrap(),
            Some(NativeSearchPhase::Initial { .. })
        ));
        let writer = QuillIndex::open(&cx, path.join("lexical"), QuillConfig::default())
            .await
            .expect("read-only serving releases the writer lease");
        LexicalWrite::index_document(
            &writer,
            &cx,
            &crate::IndexableDocument::new("new", "vertical vertical vertical"),
        )
        .await
        .unwrap();
        LexicalWrite::commit(&writer, &cx).await.unwrap();
        let Some(NativeSearchPhase::Refined { results, .. }) = stream.next_phase().await.unwrap()
        else {
            panic!("refined");
        };
        assert_eq!(results[0].doc_id, "b");
        let Some(NativeSearchPhase::Reranked { results, .. }) = stream.next_phase().await.unwrap()
        else {
            panic!("reranked");
        };
        assert_eq!(results[0].doc_id, "b");
        let texts = reranker.documents.lock().unwrap();
        assert!(texts.iter().all(|document| document.doc_id != "new"));
        assert_eq!(
            texts
                .iter()
                .find(|document| document.doc_id == "b")
                .unwrap()
                .text,
            "vertical"
        );
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn empty_hybrid_reopens_its_sealed_lexical_publication_without_embedding() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
        let index = NativeIndexBuilder::new(dir.path().join("empty"), generation(), fast.clone())
            .unwrap()
            .build_hybrid(&cx)
            .await
            .unwrap();
        assert_eq!(index.lexical().doc_count().unwrap(), 0);
        assert!(index.search(&cx, "vertical", 10).await.unwrap().is_empty());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod selected_restart {
    use super::*;
    use crate::native_ann::builder::NativeHybridReopenLimits;
    use crate::native_ann::builder::snapshot::Artifact;
    use frankensearch_core::generation::GenerationComponentReceiptV1;
    use std::collections::BTreeMap;
    use std::path::Path;

    fn directory_bytes(path: &Path) -> BTreeMap<std::path::PathBuf, Vec<u8>> {
        let mut files = BTreeMap::new();
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                files.extend(directory_bytes(&entry.path()));
            } else {
                files.insert(entry.path(), std::fs::read(entry.path()).unwrap());
            }
        }
        files
    }

    async fn reopen(
        cx: &Cx,
        path: &Path,
        receipt: &GenerationComponentReceiptV1,
    ) -> SearchResult<NativeBuiltHybridIndex> {
        NativeBuiltHybridIndex::open_selected(
            cx,
            path,
            receipt,
            Arc::new(Provider::new("fast", 2, Reply::Short)),
            None,
        )
        .await
    }

    #[test]
    fn selected_hybrid_restores_quill_sources_all_vector_tiers_and_lazy_phases() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for ann in [false, true] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("hybrid");
                let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
                let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
                let mut documents = super::super::super::tests::documents();
                let position = documents.iter().position(|doc| doc.id == "b").unwrap();
                let mut encoded = serde_json::to_value(&documents[position]).unwrap();
                encoded["title"] = serde_json::json!("original title");
                encoded["metadata"] = serde_json::json!({"revision": 9, "nested": ["kept"]});
                documents[position] = serde_json::from_value(encoded).unwrap();
                let retrieval = if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 7,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                };
                let built = NativeIndexBuilder::new(&path, generation(), fast)
                    .unwrap()
                    .add_documents(documents)
                    .with_fast_storage(NativeBuildPrecision::F16, retrieval)
                    .with_quality_embedder(quality)
                    .unwrap()
                    .build_hybrid(&cx)
                    .await
                    .unwrap();
                let expected = built.search_refined(&cx, "vertical", 5).await.unwrap();
                let source = serde_json::to_value(built.vectors().documents()).unwrap();
                let receipt = built.seal_for_reopen(&cx).unwrap();
                drop(built);
                // These providers would fail a rebuild's bound batch checks.
                // Query inference still works; selected restart must not embed.
                let fast = Arc::new(Provider::new("fast", 2, Reply::Short));
                let quality = Arc::new(Provider::new("quality", 3, Reply::Foreign));
                let before = directory_bytes(&path);
                let restored = NativeBuiltHybridIndex::open_selected(
                    &cx,
                    &path,
                    &receipt,
                    fast.clone(),
                    Some(quality.clone()),
                )
                .await
                .unwrap();
                assert_eq!(directory_bytes(&path), before);
                assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
                assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
                assert_eq!(
                    serde_json::to_value(restored.vectors().documents()).unwrap(),
                    source
                );
                assert_eq!(restored.lexical().doc_count().unwrap(), 5);
                assert_eq!(
                    restored.vectors().fast().index().retrieval_mode() == NativeRetrievalMode::Ann,
                    ann
                );
                let reranker = RerankProbe::default();
                let mut stream = restored
                    .progressive_with_reranker(&cx, "vertical", 5, &reranker, 5)
                    .unwrap();
                assert!(matches!(
                    stream.next_phase().await.unwrap(),
                    Some(NativeSearchPhase::Initial { .. })
                ));
                assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
                let Some(NativeSearchPhase::Refined { results, .. }) =
                    stream.next_phase().await.unwrap()
                else {
                    panic!("refined");
                };
                assert_eq!(results.len(), expected.len());
                for (actual, expected) in results.iter().zip(&expected) {
                    assert_eq!(actual.doc_id, expected.doc_id);
                    assert_eq!(actual.score.to_bits(), expected.score.to_bits());
                    assert_eq!(actual.metadata, expected.metadata);
                }
                assert!(
                    results
                        .iter()
                        .find(|hit| hit.doc_id == "b")
                        .unwrap()
                        .metadata
                        .is_some()
                );
                assert_eq!(reranker.calls.load(Ordering::SeqCst), 0);
                assert!(matches!(
                    stream.next_phase().await.unwrap(),
                    Some(NativeSearchPhase::Reranked { .. })
                ));
                assert_eq!(
                    reranker
                        .documents
                        .lock()
                        .unwrap()
                        .iter()
                        .find(|doc| doc.doc_id == "b")
                        .unwrap()
                        .text,
                    "vertical"
                );
                assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
                assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn relocated_selected_hybrid_supports_independent_readers_without_writer_leases() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("original");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            drop(built);
            let relocated = dir.path().join("relocated");
            std::fs::rename(path, &relocated).unwrap();
            let first = reopen(&cx, &relocated, &receipt).await.unwrap();
            let second = reopen(&cx, &relocated, &receipt).await.unwrap();
            let writer = QuillIndex::open(&cx, relocated.join("lexical"), QuillConfig::default())
                .await
                .expect("neither reopened hybrid retains a writer lease");
            LexicalWrite::index_document(
                &writer,
                &cx,
                &crate::IndexableDocument::new("late", "laterpublication"),
            )
            .await
            .unwrap();
            LexicalWrite::commit(&writer, &cx).await.unwrap();
            assert_eq!(first.lexical().doc_count().unwrap(), 5);
            drop(first);
            assert_eq!(
                second.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            assert!(
                second
                    .lexical()
                    .search(&cx, "laterpublication", 10)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(reopen(&cx, &relocated, &receipt).await.is_err());
        });
    }

    #[test]
    fn same_cardinality_foreign_lexical_directory_cannot_replace_selected_bytes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("original");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            let mut docs = super::super::super::tests::documents();
            for doc in &mut docs {
                doc.content = "foreignpublication".to_owned();
            }
            let other_path = dir.path().join("other");
            let other = NativeIndexBuilder::new(&other_path, generation(), fast.clone())
                .unwrap()
                .add_documents(docs)
                .build_hybrid(&cx)
                .await
                .unwrap();
            assert_eq!(
                other.lexical().doc_count().unwrap(),
                built.lexical().doc_count().unwrap()
            );
            std::fs::rename(path.join("lexical"), dir.path().join("retained-original")).unwrap();
            std::fs::rename(other_path.join("lexical"), path.join("lexical")).unwrap();
            assert!(reopen(&cx, &path, &receipt).await.is_err());
            assert_eq!(
                built.lexical().search(&cx, "vertical", 10).await.unwrap()[0].doc_id,
                "b"
            );
        });
    }

    #[test]
    fn sealing_old_reader_does_not_bless_newer_lexical_publication() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("hybrid");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let writer = QuillIndex::open(&cx, path.join("lexical"), QuillConfig::default())
                .await
                .unwrap();
            LexicalWrite::index_document(
                &writer,
                &cx,
                &crate::IndexableDocument::new("late", "laterpublication"),
            )
            .await
            .unwrap();
            LexicalWrite::commit(&writer, &cx).await.unwrap();
            assert!(built.seal_for_reopen(&cx).is_err());
            assert!(!path.join("native.snapshot.json").exists());
            assert!(!path.join("native.source.jsonl").exists());
            assert!(!path.join("native.hybrid.json").exists());
            assert_eq!(built.lexical().doc_count().unwrap(), 5);
        });
    }

    #[test]
    fn missing_manifest_or_segment_never_becomes_vector_only_or_previous_lexical_success() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for manifest in [true, false] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("hybrid");
                let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
                let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
                let receipt = built.seal_for_reopen(&cx).unwrap();
                let encoded: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(path.join("native.hybrid.json")).unwrap(),
                )
                .unwrap();
                let name = if manifest {
                    "MANIFEST"
                } else {
                    encoded["lexical"]["files"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find_map(|file| {
                            file["name"].as_str().filter(|name| name.ends_with(".fslx"))
                        })
                        .unwrap()
                };
                std::fs::rename(
                    path.join("lexical").join(name),
                    dir.path().join("displaced-artifact"),
                )
                .unwrap();
                assert!(reopen(&cx, &path, &receipt).await.is_err());
                assert_eq!(
                    built.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                    "b"
                );
            }
        });
    }

    #[test]
    fn hybrid_input_limits_and_unsafe_descriptor_names_fail_closed() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("hybrid");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            for limits in [
                NativeHybridReopenLimits {
                    max_lexical_files: 0,
                    ..NativeHybridReopenLimits::default()
                },
                NativeHybridReopenLimits {
                    max_lexical_file_bytes: 1,
                    ..NativeHybridReopenLimits::default()
                },
                NativeHybridReopenLimits {
                    max_lexical_bytes: 1,
                    ..NativeHybridReopenLimits::default()
                },
            ] {
                assert!(
                    NativeBuiltHybridIndex::open_selected_with_limits(
                        &cx,
                        &path,
                        &receipt,
                        fast.clone(),
                        None,
                        limits
                    )
                    .await
                    .is_err()
                );
            }
            let file = path.join("native.hybrid.json");
            let original: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
            for name in ["../fast.fsvi", "subdir/file", "LOCK", "C:stream", ".", ""] {
                let mut changed = original.clone();
                changed["lexical"]["files"][0]["name"] = serde_json::json!(name);
                let bytes = serde_json::to_vec(&changed).unwrap();
                std::fs::write(&file, &bytes).unwrap();
                // Even an explicitly selected malformed descriptor may not
                // escape the lexical directory or name writer-control state.
                let selected = Artifact::from_bytes(&bytes).receipt();
                assert!(reopen(&cx, &path, &selected).await.is_err());
            }
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn changed_lock_is_not_part_of_read_only_hybrid_selection() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("hybrid");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            std::fs::write(
                path.join("lexical").join("LOCK"),
                b"not a writer-admission record",
            )
            .unwrap();
            let restored = reopen(&cx, &path, &receipt).await.unwrap();
            assert_eq!(
                restored.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
        });
    }

    #[test]
    fn partial_hybrid_seal_is_not_overwritten_or_admitted() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("hybrid");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let partial = b"partial descriptor";
            std::fs::write(path.join("native.hybrid.json"), partial).unwrap();
            assert!(built.seal_for_reopen(&cx).is_err());
            assert_eq!(
                std::fs::read(path.join("native.hybrid.json")).unwrap(),
                partial
            );
            assert!(!path.join("native.snapshot.json").exists());
            assert!(
                reopen(&cx, &path, &Artifact::from_bytes(partial).receipt())
                    .await
                    .is_err()
            );
        });
    }

    #[test]
    fn cancelled_hybrid_seal_and_reopen_do_not_write_or_produce_partial_success() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("hybrid");
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let built = builder(&path, &fast).build_hybrid(&cx).await.unwrap();
            let receipt = built.seal_for_reopen(&cx).unwrap();
            let other_path = dir.path().join("unsealed");
            let unsealed = builder(&other_path, &fast).build_hybrid(&cx).await.unwrap();
            let before = directory_bytes(dir.path());
            cx.cancel_with(asupersync::CancelKind::User, Some("cancel hybrid restart"));
            assert!(matches!(
                reopen(&cx, &path, &receipt).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                unsealed.seal_for_reopen(&cx),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(directory_bytes(dir.path()), before);
        });
    }
}
