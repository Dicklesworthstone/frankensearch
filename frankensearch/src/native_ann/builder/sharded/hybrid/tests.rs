use super::*;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::collections::BTreeMap;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::fs;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use asupersync::test_utils::run_test_with_cx;
use crate::native_ann::builder::tests::{Provider, Reply, generation};
use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval};
use crate::native_ann::NativeShardedSearchPhase;
use crate::{ModelCategory, RerankDocument, RerankScore, ScoredResult, SearchFuture};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_index::native_hnsw::HnswParams;

fn docs() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("z", "horizontal").with_title("rare title"),
        IndexableDocument::new("b", "vertical").with_metadata("language", "β"),
        IndexableDocument::new("a", "horizontal horizontal"),
        IndexableDocument::new("punctuation", "!!!\t—"),
        IndexableDocument::new("empty", "").with_title(""),
    ]
}

fn models() -> (Arc<Provider>, Arc<Provider>) {
    (Arc::new(Provider::new("fast", 2, Reply::Correct)),
     Arc::new(Provider::new("quality", 3, Reply::Correct)))
}

fn builder(path: &Path, fast: &Arc<Provider>, quality: &Arc<Provider>, graph: bool) -> NativeIndexBuilder {
    let retrieval = if graph {
        NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 101 }
    } else { NativeBuildRetrieval::Exact };
    NativeIndexBuilder::new(path, generation(), fast.clone()).unwrap()
        .with_quality_embedder(quality.clone()).unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .with_quality_storage(NativeBuildPrecision::F32, retrieval).unwrap()
        .add_documents(docs())
}

fn logical(mut results: Vec<ScoredResult>) -> serde_json::Value {
    // Only physical single-owner indexes differ: every actual score, ID, source,
    // rank and hydrated field remains part of the comparison.
    for result in &mut results { result.index = None; }
    serde_json::to_value(results).unwrap()
}

fn logical_sharded(results: Vec<NativeShardedResult>) -> serde_json::Value {
    assert!(results.iter().all(|hit| hit.result.index.is_none()));
    logical(results.into_iter().map(|hit| hit.result).collect())
}

#[test]
fn global_quill_ranking_matches_single_cohort_across_partition_boundaries() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let (fast, quality) = models();
        let reference = builder(&parent.path().join("single"), &fast, &quality, false)
            .build_hybrid(&cx).await.unwrap();
        for size in [1, 2, 10] {
            let partitioned = builder(&parent.path().join(format!("partition-{size}")), &fast, &quality, false)
                .build_sharded_hybrid(&cx, size).await.unwrap();
            assert_eq!(partitioned.lexical().doc_count().unwrap(), 5);
            for query in ["vertical", "horizontal", "title:rare", "notincorpus"] {
                let keyword = reference.lexical().search(&cx, query, 10).await.unwrap();
                let global = partitioned.lexical().search(&cx, query, 10).await.unwrap();
                assert_eq!(logical(global), logical(keyword), "global keyword population, size={size}, query={query}");
                assert_eq!(logical_sharded(partitioned.search(&cx, query, 10).await.unwrap()),
                    logical(reference.search(&cx, query, 10).await.unwrap()));
                assert_eq!(logical_sharded(partitioned.search_refined(&cx, query, 10).await.unwrap()),
                    logical(reference.search_refined(&cx, query, 10).await.unwrap()));
                assert_eq!(logical_sharded(partitioned.search_quality(&cx, query, 10).await.unwrap()),
                    logical(reference.search_quality(&cx, query, 10).await.unwrap()));
            }
        }
    });
}

struct Directional {
    identity: EmbeddingIdentityBundleV1,
    quality: bool,
    calls: AtomicUsize,
}
impl Embedder for Directional {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { Ok(&self.identity) }
    fn dimension(&self) -> usize { 2 }
    fn id(&self) -> &str { &self.identity.space.logical_model_id }
    fn model_name(&self) -> &str { self.id() }
    fn is_semantic(&self) -> bool { false }
    fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let first = text == "querynotindexed" || (text == "needle") == self.quality;
            Ok(if first { vec![1.0, 0.0] } else { vec![0.0, 1.0] })
        })
    }
}

#[test]
fn independent_quality_finds_a_document_outside_the_global_fast_candidate_pool() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let fast = Arc::new(Directional { identity: EmbeddingIdentityBundleV1::explicit_test_model("directional-fast", 2), quality: false, calls: AtomicUsize::new(0) });
        let quality = Arc::new(Directional { identity: EmbeddingIdentityBundleV1::explicit_test_model("directional-quality", 2), quality: true, calls: AtomicUsize::new(0) });
        let documents = (0..100).map(|i| IndexableDocument::new(format!("a-{i:03}"), "decoy"))
            .chain(std::iter::once(IndexableDocument::new("zz-target", "needle")));
        let built = NativeIndexBuilder::new(parent.path().join("cohort"), generation(), fast.clone()).unwrap()
            .with_quality_embedder(quality.clone()).unwrap().add_documents(documents)
            .build_sharded_hybrid(&cx, 11).await.unwrap();
        fast.calls.store(0, Ordering::SeqCst);
        quality.calls.store(0, Ordering::SeqCst);
        let mut stream = built.progressive(&cx, "querynotindexed", 5).unwrap();
        assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        let Some(NativeShardedSearchPhase::Initial { results, candidates }) = stream.next_phase().await.unwrap() else {
            panic!("Initial required"); // ubs:ignore — test assertion.
        };
        assert!(!results.iter().any(|hit| hit.result.doc_id == "zz-target"));
        assert!(candidates.fast < 100);
        assert_eq!(candidates.lexical, 0);
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 0, "no eager quality across shards");
        let Some(NativeShardedSearchPhase::Refined { results, .. }) = stream.next_phase().await.unwrap() else {
            panic!("Refined required"); // ubs:ignore — test assertion.
        };
        let winner = results.iter().find(|hit| hit.result.doc_id == "zz-target").unwrap();
        assert!(winner.fast_row.is_none());
        assert!(winner.quality_row.is_some());
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        let primary = built.search_quality(&cx, "querynotindexed", 1).await.unwrap();
        assert_eq!(primary[0].result.doc_id, "zz-target");
        assert!(primary[0].fast_row.is_none());
        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
    });
}

#[derive(Default)]
struct Scorer {
    calls: AtomicUsize,
    fail: AtomicBool,
    seen: Mutex<Vec<(String, String)>>,
}
impl Reranker for Scorer {
    fn id(&self) -> &str { "shard-source-reranker" }
    fn model_name(&self) -> &str { self.id() }
    fn rerank<'a>(&'a self, _cx: &'a Cx, _query: &'a str, documents: &'a [RerankDocument]) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.seen.lock().unwrap() = documents.iter().map(|doc| (doc.doc_id.clone(), doc.text.clone())).collect();
            if self.fail.load(Ordering::SeqCst) {
                return Err(SearchError::RerankFailed { model: self.id().to_owned(), source: "injected failure".into() });
            }
            let mut scores = documents.iter().enumerate().map(|(original_rank, doc)| RerankScore {
                doc_id: doc.doc_id.clone(), original_rank,
                score: if doc.doc_id == "z" { 1.0 } else { 0.0 }, raw_logit: None,
            }).collect::<Vec<_>>();
            scores.sort_by(|left, right| right.score.total_cmp(&left.score));
            Ok(scores)
        })
    }
}

#[test]
fn lazy_reranking_uses_retained_bodies_and_preserves_shard_rows_on_success_and_failure() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let (fast, quality) = models();
        let built = builder(&parent.path().join("old"), &fast, &quality, true)
            .build_sharded_hybrid(&cx, 2).await.unwrap();
        // A separate successor with changed text cannot affect old rerank bodies.
        let successor = NativeIndexBuilder::new(parent.path().join("new"),
            ArtifactGenerationIdentityV1::new(10, [88; 16]).unwrap(), fast.clone()).unwrap()
            .with_quality_embedder(quality.clone()).unwrap()
            .add_document(IndexableDocument::new("z", "replacement body"))
            .build_sharded_hybrid(&cx, 1).await.unwrap();
        assert_eq!(successor.vectors().document("z").unwrap().content, "replacement body");
        for fail in [false, true] {
            let scorer = Scorer::default();
            scorer.fail.store(fail, Ordering::SeqCst);
            let mut stream = built.progressive_with_reranker(&cx, "vertical", 5, &scorer, 5).unwrap();
            assert!(matches!(stream.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { .. })));
            assert_eq!(scorer.calls.load(Ordering::SeqCst), 0);
            let Some(NativeShardedSearchPhase::Refined { results: prior, .. }) = stream.next_phase().await.unwrap() else {
                panic!("Refined required"); // ubs:ignore — test assertion.
            };
            assert_eq!(scorer.calls.load(Ordering::SeqCst), 0);
            let next = stream.next_phase().await.unwrap().unwrap();
            match next {
                NativeShardedSearchPhase::Reranked { results, evaluated, .. } if !fail => {
                    assert_eq!(evaluated, 5);
                    assert_eq!(results[0].result.doc_id, "z");
                    for hit in &results {
                        let old = prior.iter().find(|old| old.result.doc_id == hit.result.doc_id).unwrap();
                        assert_eq!(hit.fast_row, old.fast_row);
                        assert_eq!(hit.quality_row, old.quality_row);
                        assert!(hit.result.index.is_none());
                        assert!(hit.result.rerank_score.is_some());
                    }
                }
                NativeShardedSearchPhase::RerankFailed { previous_results, .. } if fail => {
                    assert_eq!(logical_sharded(previous_results), logical_sharded(prior));
                }
                _ => panic!("wrong final phase"), // ubs:ignore — test assertion.
            }
            assert_eq!(scorer.calls.load(Ordering::SeqCst), 1);
            for (id, body) in scorer.seen.lock().unwrap().iter() {
                assert_eq!(&built.vectors().document(id).unwrap().content, body);
            }
            assert!(stream.next_phase().await.unwrap().is_none());
        }
    });
}

#[test]
fn equal_counts_and_valid_lexical_files_cannot_mix_a_different_source_cut() {
    run_test_with_cx(|cx| async move {
        for fault in ["content", "metadata", "title", "id"] {
            let parent = tempfile::tempdir().unwrap();
            let (fast, quality) = models();
            let vectors = builder(&parent.path().join("cohort"), &fast, &quality, false)
                .build_sharded(&cx, 2).await.unwrap();
            let mut different = docs();
            let changed = different.iter_mut().find(|doc| doc.id == "b").unwrap();
            match fault {
                "content" => changed.content = "VERTICAL!".to_owned(),
                "metadata" => { changed.metadata.insert("language".to_owned(), "foreign".to_owned()); }
                "title" => changed.title = Some("foreign".to_owned()),
                _ => changed.id = "foreign".to_owned(),
            }
            let (writer, reader, seal) = create_lexical(&cx, &vectors.directory().join("lexical"), different.iter()).await.unwrap();
            assert_eq!(reader.doc_count().unwrap(), vectors.document_count());
            assert_eq!(reader.keeper_generation(), seal.generation());
            assert!(matches!(NativeBuiltShardedHybridIndex::from_readers(&cx, vectors, reader, seal),
                Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.builder.lexical_source_join"));
            drop(writer);
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn image(path: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut output = BTreeMap::new();
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() { pending.push(path); }
            else {
                let bytes = fs::read(&path).unwrap();
                output.insert(path, bytes);
            }
        }
    }
    output
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn complete_sharded_hybrid_restart_keeps_global_scores_metadata_and_exact_rows() {
    run_test_with_cx(|cx| async move {
        for graph in [false, true] {
            let parent = tempfile::tempdir().unwrap();
            let path = parent.path().join("cohort");
            let (fast, quality) = models();
            let built = builder(&path, &fast, &quality, graph).build_sharded_hybrid(&cx, 2).await.unwrap();
            let selected = built.seal_for_reopen(&cx).unwrap();
            let before = image(&path);
            let query_counts = (fast.queries.load(Ordering::SeqCst), quality.queries.load(Ordering::SeqCst));
            let restarted = NativeBuiltShardedHybridIndex::open_selected(&cx, &path, &selected,
                fast.clone(), Some(quality.clone()), NativeShardedHybridReopenLimits::default()).await.unwrap();
            assert_eq!(query_counts, (fast.queries.load(Ordering::SeqCst), quality.queries.load(Ordering::SeqCst)));
            let old = built.search_refined(&cx, "vertical", 10).await.unwrap();
            let fresh = restarted.search_refined(&cx, "vertical", 10).await.unwrap();
            for (a, b) in old.iter().zip(&fresh) {
                assert_eq!(a.fast_row, b.fast_row);
                assert_eq!(a.quality_row, b.quality_row);
            }
            assert_eq!(logical_sharded(old), logical_sharded(fresh));
            assert_eq!(restarted.vectors().document("b").unwrap().metadata["language"], "β");
            assert_eq!(restarted.vectors().document("empty").unwrap().content, "");
            assert_eq!(image(&path), before);
            assert!(built.seal_for_reopen(&cx).is_err());
            assert_eq!(image(&path), before);
        }
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn missing_lexical_and_last_partition_refuse_restart_without_changing_old_readers() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("cohort");
        let (fast, quality) = models();
        let built = builder(&path, &fast, &quality, true).build_sharded_hybrid(&cx, 2).await.unwrap();
        let selected = built.seal_for_reopen(&cx).unwrap();
        let before = image(&path);
        for (ordinal, relative) in ["lexical/MANIFEST", "shard-000002/quality.fsvi", "native.sharded.json"].iter().enumerate() {
            let original = path.join(relative);
            let saved = parent.path().join(format!("retained-{ordinal}"));
            fs::rename(&original, &saved).unwrap();
            assert!(NativeBuiltShardedHybridIndex::open_selected(&cx, &path, &selected,
                fast.clone(), Some(quality.clone()), NativeShardedHybridReopenLimits::default()).await.is_err());
            assert_eq!(built.search(&cx, "vertical", 1).await.unwrap()[0].result.doc_id, "b");
            assert!(!original.exists(), "no repair or rebuild during admission");
            fs::rename(saved, original).unwrap();
        }
        assert_eq!(image(&path), before);
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn lexical_budgets_and_cancellation_fail_before_open_and_empty_fast_only_round_trips() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("empty");
        let (fast, _) = models();
        let built = NativeIndexBuilder::new(&path, generation(), fast.clone()).unwrap()
            .build_sharded_hybrid(&cx, 5).await.unwrap();
        let selected = built.seal_for_reopen(&cx).unwrap();
        let before = image(&path);
        let too_small = NativeShardedHybridReopenLimits { max_lexical_bytes: 1, ..NativeShardedHybridReopenLimits::default() };
        assert!(NativeBuiltShardedHybridIndex::open_selected(&cx, &path, &selected, fast.clone(), None, too_small).await.is_err());
        cx.set_cancel_requested(true);
        assert!(matches!(NativeBuiltShardedHybridIndex::open_selected(&cx, &path, &selected,
            fast.clone(), None, NativeShardedHybridReopenLimits::default()).await, Err(SearchError::Cancelled { .. })));
        cx.set_cancel_requested(false);
        let reopened = NativeBuiltShardedHybridIndex::open_selected(&cx, &path, &selected,
            fast.clone(), None, NativeShardedHybridReopenLimits::default()).await.unwrap();
        assert!(reopened.search_refined(&cx, "empty", 1).await.unwrap().is_empty());
        assert!(reopened.search_quality(&cx, "empty", 0).await.is_err());
        assert_eq!(reopened.vectors().partitions().len(), 1);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(image(&path), before);
    });
}
