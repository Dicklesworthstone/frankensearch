//! Explicit keyword admission against real native/Quill snapshots. Deterministic
//! providers construct fixtures only; keyword opens retain no provider handles.
use super::*;
use std::collections::BTreeMap;
use std::fs;

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::native_hnsw::HnswParams;

use crate::native_ann::builder::sharded::{
    NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
};
use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder};
use crate::{ModelCategory, ScoredResult, SearchFuture};

struct Provider(EmbeddingIdentityBundleV1);
impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async { Ok(vec![1.0, 0.0]) })
    }
    fn embed_batch_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            Ok(texts
                .iter()
                .map(|_| IdentityBoundEmbedding {
                    values: vec![1.0, 0.0],
                    identity: self.0.clone(),
                })
                .collect())
        })
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.0)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        &self.0.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

struct Fixture {
    path: PathBuf,
    receipt: GenerationComponentReceiptV1,
    generation: ArtifactGenerationIdentityV1,
    expected: Vec<ScoredResult>,
    derived: Vec<PathBuf>,
    sharded: bool,
}

fn documents() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("a", "needle needle network")
            .with_title("Original title")
            .with_metadata("tenant", "private"),
        IndexableDocument::new("b", "needle public network padding")
            .with_metadata("tenant", "public"),
        IndexableDocument::new("c", "!!!")
            .with_title("punctuation")
            .with_metadata("tenant", "public"),
        IndexableDocument::new("d", "").with_metadata("tenant", "public"),
    ]
}

async fn build(
    cx: &Cx,
    path: &Path,
    sharded: bool,
    graphs: bool,
    docs: Vec<IndexableDocument>,
) -> Fixture {
    let fast = Arc::new(Provider(EmbeddingIdentityBundleV1::explicit_test_model(
        "lexical-fixture-fast",
        2,
    )));
    let quality = Arc::new(Provider(EmbeddingIdentityBundleV1::explicit_test_model(
        "lexical-fixture-quality",
        2,
    )));
    let weak = Arc::downgrade(&fast);
    let generation = ArtifactGenerationIdentityV1::new(1, [37; 16]).unwrap();
    let retrieval = if graphs {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 37,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    let builder = NativeIndexBuilder::new(path, generation, fast.clone())
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .with_quality_embedder(quality.clone())
        .unwrap()
        .with_quality_storage(NativeBuildPrecision::F16, retrieval)
        .unwrap()
        .add_documents(docs);
    let paths = |index: &NativeBuiltIndex| {
        let mut paths = Vec::new();
        for tier in std::iter::once(index.fast()).chain(index.quality()) {
            paths.push(tier.vector_path().to_path_buf());
            if let Some(graph) = tier.graph_path() {
                paths.push(graph.to_path_buf());
                let mut receipt = graph.as_os_str().to_os_string();
                receipt.push(".receipt");
                paths.push(PathBuf::from(receipt));
            }
        }
        paths
    };
    let fixture = if sharded {
        let index = builder.build_sharded_hybrid(cx, 2).await.unwrap();
        Fixture {
            path: path.to_path_buf(),
            receipt: index.seal_for_reopen(cx).unwrap(),
            generation,
            expected: index.lexical().search(cx, "needle", 10).await.unwrap(),
            derived: index.vectors().partitions().iter().flat_map(paths).collect(),
            sharded,
        }
    } else {
        let index = builder.build_hybrid(cx).await.unwrap();
        Fixture {
            path: path.to_path_buf(),
            receipt: index.seal_for_reopen(cx).unwrap(),
            generation,
            expected: index.lexical().search(cx, "needle", 10).await.unwrap(),
            derived: paths(index.vectors()),
            sharded,
        }
    };
    drop(fast);
    drop(quality);
    assert!(
        weak.upgrade().is_none(),
        "no model owner survives fixture construction"
    );
    fixture
}

impl Fixture {
    async fn open(
        &self,
        cx: &Cx,
    ) -> SearchResult<(
        ArtifactGenerationIdentityV1,
        QuillSearchIndex,
        Vec<IndexableDocument>,
    )> {
        if self.sharded {
            NativeBuiltShardedHybridIndex::open_selected_lexical(
                cx,
                &self.path,
                &self.receipt,
                NativeShardedHybridReopenLimits::default(),
            )
            .await
        } else {
            NativeBuiltHybridIndex::open_selected_lexical(
                cx,
                &self.path,
                &self.receipt,
                NativeHybridReopenLimits::default(),
            )
            .await
        }
    }
    fn descriptor(&self) -> PathBuf {
        self.path.join(if self.sharded {
            "native.sharded-hybrid.json"
        } else {
            HYBRID_FILE
        })
    }
    fn last_source(&self) -> PathBuf {
        if self.sharded {
            self.path.join("shard-000001/native.source.jsonl")
        } else {
            self.path.join("native.source.jsonl")
        }
    }
}

fn image(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                files.insert(entry.path(), fs::read(entry.path()).unwrap());
            }
        }
    }
    files
}

#[test]
fn keyword_open_needs_no_models_vectors_or_graphs_and_preserves_raw_quill_results() {
    run_test_with_cx(|cx| async move {
        for sharded in [false, true] {
            for graphs in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let f = build(&cx, &root.path().join("index"), sharded, graphs, documents()).await;
                for (n, path) in f.derived.iter().enumerate() {
                    fs::rename(path, root.path().join(format!("saved-{n}"))).unwrap();
                }
                let before = image(root.path());
                let (generation, lexical, sources) = f.open(&cx).await.unwrap();
                assert_eq!(generation, f.generation);
                assert_eq!(
                    serde_json::to_value(sources).unwrap(),
                    serde_json::to_value(documents()).unwrap()
                );
                assert_eq!(LexicalRead::doc_count(&lexical).unwrap(), 4);
                let results = LexicalRead::search(&lexical, &cx, "needle", 10)
                    .await
                    .unwrap();
                // ScoredResult has no PartialEq; compare every field.
                assert_eq!(
                    serde_json::to_value(&results).unwrap(),
                    serde_json::to_value(&f.expected).unwrap()
                );
                assert!(results
                    .iter()
                    .all(|hit| hit.fast_score.is_none() && hit.quality_score.is_none()));
                assert_eq!(
                    image(root.path()), before,
                    "read-only admission must not repair or write files"
                );
            }
        }
    });
}

#[test]
fn every_used_source_lexical_and_descriptor_byte_remains_required() {
    run_test_with_cx(|cx| async move {
        for sharded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut f = build(&cx, &root.path().join("index"), sharded, true, documents()).await;
            let segment = fs::read_dir(f.path.join("lexical"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|path| path.extension().is_some_and(|extension| extension == "fslx"))
                .unwrap();
            for (n, path) in [f.descriptor(), f.last_source(), segment]
                .into_iter()
                .enumerate()
            {
                let bytes = fs::read(&path).unwrap();
                let mut damaged = bytes.clone();
                damaged[0] ^= 1;
                fs::write(&path, damaged).unwrap();
                let before = image(root.path());
                assert!(f.open(&cx).await.is_err());
                assert_eq!(image(root.path()), before);
                fs::write(&path, &bytes).unwrap();
                let saved = root.path().join(format!("missing-{n}"));
                fs::rename(&path, &saved).unwrap();
                assert!(f.open(&cx).await.is_err());
                fs::rename(saved, path).unwrap();
            }
            f.receipt.sha256[0] ^= 1;
            assert!(f.open(&cx).await.is_err());
            f.receipt.sha256[0] ^= 1;
            assert!(f.open(&cx).await.is_ok());
        }
    });
}

#[test]
fn individually_valid_same_count_lexical_population_cannot_replace_selected_source() {
    run_test_with_cx(|cx| async move {
        for sharded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut first = build(&cx, &root.path().join("first"), sharded, false, documents()).await;
            let mut changed = documents();
            changed[2].title = Some("different untokenized title".to_owned());
            changed[3]
                .metadata
                .insert("tenant".to_owned(), "other".to_owned());
            let second = build(&cx, &root.path().join("second"), sharded, false, changed).await;
            assert!(second.open(&cx).await.is_ok());
            // Explicit malformed-selection fixture: authenticate a hybrid descriptor
            // that binds individually valid components from different source cohorts.
            let mut outer: serde_json::Value =
                serde_json::from_slice(&fs::read(first.descriptor()).unwrap()).unwrap();
            let donor: serde_json::Value =
                serde_json::from_slice(&fs::read(second.descriptor()).unwrap()).unwrap();
            outer["lexical"] = donor["lexical"].clone();
            fs::rename(
                first.path.join("lexical"),
                root.path().join("saved-first-lexical"),
            )
            .unwrap();
            fs::rename(second.path.join("lexical"), first.path.join("lexical")).unwrap();
            let bytes = serde_json::to_vec(&outer).unwrap();
            fs::write(first.descriptor(), &bytes).unwrap();
            first.receipt = Artifact::from_bytes(&bytes).receipt();
            // All file receipts and document counts agree. Only the full census
            // sees the altered non-tokenized document fields.
            assert!(first.open(&cx).await.is_err());
        }
    });
}

#[test]
fn cancellation_and_resource_limits_refuse_before_keyword_admission() {
    run_test_with_cx(|cx| async move {
        for sharded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let f = build(&cx, &root.path().join("index"), sharded, false, documents()).await;
            let before = image(root.path());
            cx.set_cancel_requested(true);
            assert!(matches!(
                f.open(&cx).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            if sharded {
                let mut limits = NativeShardedHybridReopenLimits::default();
                limits.vectors.max_documents = 3;
                assert!(NativeBuiltShardedHybridIndex::open_selected_lexical(
                    &cx, &f.path, &f.receipt, limits,
                )
                .await
                .is_err());
                limits.vectors.max_documents = 4;
                limits.max_lexical_bytes = 1;
                assert!(NativeBuiltShardedHybridIndex::open_selected_lexical(
                    &cx, &f.path, &f.receipt, limits,
                )
                .await
                .is_err());
            } else {
                let mut limits = NativeHybridReopenLimits::default();
                limits.vectors.max_document_bytes = 1;
                assert!(NativeBuiltHybridIndex::open_selected_lexical(
                    &cx, &f.path, &f.receipt, limits,
                )
                .await
                .is_err());
                limits.vectors.max_document_bytes = 1024;
                limits.max_lexical_file_bytes = 1;
                assert!(NativeBuiltHybridIndex::open_selected_lexical(
                    &cx, &f.path, &f.receipt, limits,
                )
                .await
                .is_err());
            }
            assert!(f.open(&cx).await.is_ok());
            assert_eq!(image(root.path()), before);
        }
    });
}

#[test]
fn empty_sources_and_old_readers_remain_explicitly_selected() {
    run_test_with_cx(|cx| async move {
        for sharded in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let old = build(&cx, &root.path().join("old"), sharded, false, documents()).await;
            let (_, reader, retained) = old.open(&cx).await.unwrap();
            let empty = build(&cx, &root.path().join("empty"), sharded, true, Vec::new()).await;
            let (_, newer, sources) = empty.open(&cx).await.unwrap();
            assert!(sources.is_empty());
            assert_eq!(LexicalRead::doc_count(&newer).unwrap(), 0);
            assert!(LexicalRead::search(&newer, &cx, "needle", 10)
                .await
                .unwrap()
                .is_empty());
            assert_eq!(
                serde_json::to_value(
                    LexicalRead::search(&reader, &cx, "needle", 10)
                        .await
                        .unwrap()
                )
                .unwrap(),
                serde_json::to_value(&old.expected).unwrap()
            );
            assert_eq!(
                serde_json::to_value(retained).unwrap(),
                serde_json::to_value(documents()).unwrap()
            );
        }
    });
}
