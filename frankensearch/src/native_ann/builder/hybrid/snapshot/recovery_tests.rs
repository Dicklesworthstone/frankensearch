use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::test_utils::run_test_with_cx;
use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder};
use crate::{ModelCategory, SearchFuture};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_index::native_hnsw::HnswParams;

// Deterministic control providers exercise actual source/FSVI/graph/Quill
// persistence. These tests make no semantic-quality or real-model claim.
struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Provider {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, 3),
            calls: AtomicUsize::new(0),
        })
    }
}

impl Embedder for Provider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { Ok(&self.identity) }
    fn id(&self) -> &str { &self.identity.space.logical_model_id }
    fn model_name(&self) -> &str { self.id() }
    fn dimension(&self) -> usize { 3 }
    fn is_semantic(&self) -> bool { false }
    fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1.0, 0.0, 0.0])
        })
    }
}

fn documents() -> Vec<IndexableDocument> {
    let mut first = IndexableDocument::new("a", "alpha recovery content café 🦀");
    first.title = Some("東京 title".to_owned());
    first.metadata.insert("scope".to_owned(), "private".to_owned());
    vec![first, IndexableDocument::new("z", "")]
}

async fn fixture(
    cx: &Cx,
    directory: &Path,
    exact: bool,
    quality: bool,
) -> (NativeBuiltHybridIndex, GenerationComponentReceiptV1, Arc<Provider>, Arc<Provider>) {
    let fast = Provider::new("recovery-fast");
    let slow = Provider::new("recovery-quality");
    let retrieval = if exact {
        NativeBuildRetrieval::Exact
    } else {
        NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 7 }
    };
    let mut builder = NativeIndexBuilder::new(
        directory,
        ArtifactGenerationIdentityV1::new(11, [0x41; 16]).unwrap(),
        fast.clone(),
    ).unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents(documents());
    if quality {
        builder = builder.with_quality_embedder(slow.clone()).unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, retrieval).unwrap();
    }
    let index = builder.build_hybrid(cx).await.unwrap();
    let receipt = index.seal_for_reopen(cx).unwrap();
    (index, receipt, fast, slow)
}

fn image(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, directory: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(path.strip_prefix(root).unwrap().to_path_buf(), fs::read(path).unwrap());
            }
        }
    }
    let mut files = BTreeMap::new();
    visit(root, root, &mut files);
    files
}

fn recover(
    cx: &Cx,
    directory: &Path,
    receipt: &GenerationComponentReceiptV1,
) -> SearchResult<(ArtifactGenerationIdentityV1, Vec<IndexableDocument>)> {
    NativeBuiltHybridIndex::recover_selected_source(
        cx, directory, receipt, NativeHybridReopenLimits::default(),
    )
}

#[test]
fn recovery_needs_only_selected_descriptors_and_sources_not_models_or_search_files() {
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            for quality in [false, true] {
                let temp = tempfile::tempdir().unwrap();
                let directory = temp.path().join("index");
                let (index, receipt, fast, slow) = fixture(&cx, &directory, exact, quality).await;
                let generation = index.vectors().fast().index().owner_witness().generation;
                drop(index); // No mapped lexical reader survives the injected damage.
                let before_calls = (fast.calls.load(Ordering::SeqCst), slow.calls.load(Ordering::SeqCst));
                let displaced = temp.path().join("displaced");
                fs::create_dir(&displaced).unwrap();
                for entry in fs::read_dir(&directory).unwrap() {
                    let entry = entry.unwrap();
                    if !matches!(entry.file_name().to_str(),
                        Some("native.hybrid.json" | "native.snapshot.json" | "native.source.jsonl"))
                    {
                        fs::rename(entry.path(), displaced.join(entry.file_name())).unwrap();
                    }
                }
                assert!(NativeBuiltHybridIndex::open_selected(
                    &cx, &directory, &receipt, fast.clone(),
                    quality.then(|| slow.clone() as Arc<dyn Embedder>),
                ).await.is_err(), "ordinary reader must NOT accept a recovered source as an index");
                let before = image(temp.path());
                let (recovered_generation, recovered) = recover(&cx, &directory, &receipt).unwrap();
                assert_eq!(recovered_generation, generation);
                assert_eq!(serde_json::to_value(&recovered).unwrap(), serde_json::to_value(documents()).unwrap());
                assert_eq!(image(temp.path()), before);
                assert_eq!(fast.calls.load(Ordering::SeqCst), before_calls.0);
                assert_eq!(slow.calls.load(Ordering::SeqCst), before_calls.1);
            }
        }
    });
}

#[test]
fn every_link_in_the_recovery_receipt_chain_is_required_without_fallback() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("index");
        let (index, receipt, _, _) = fixture(&cx, &directory, false, true).await;
        drop(index);
        for name in [HYBRID_FILE, "native.snapshot.json", "native.source.jsonl"] {
            let path = directory.join(name);
            let original = fs::read(&path).unwrap();
            let mut corrupt = original.clone();
            // Same-length bit corruption defeats length-only admission.
            let at = corrupt.len() / 2;
            corrupt[at] ^= 1;
            fs::write(&path, &corrupt).unwrap();
            let before = image(&directory);
            assert!(recover(&cx, &directory, &receipt).is_err(), "{name}");
            assert_eq!(image(&directory), before, "refusal must not repair {name}");
            fs::write(&path, &original).unwrap();
        }
        let wrong = GenerationComponentReceiptV1 {
            byte_len: receipt.byte_len,
            sha256: [0x33; 32],
        };
        assert!(recover(&cx, &directory, &wrong).is_err());
        let (generation, docs) = recover(&cx, &directory, &receipt).unwrap();
        assert_eq!(generation.sequence, 11);
        assert_eq!(docs.len(), 2);
    });
}

#[test]
fn recovery_enforces_source_limits_and_cancellation_without_partial_results() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("index");
        let (index, receipt, _, _) = fixture(&cx, &directory, true, false).await;
        drop(index);
        let before = image(&directory);
        let source_bytes = fs::metadata(directory.join("native.source.jsonl")).unwrap().len();
        for fault in 0..4 {
            let mut limits = NativeHybridReopenLimits::default();
            match fault {
                0 => limits.vectors.max_documents = 1,
                1 => limits.vectors.max_source_bytes = source_bytes - 1,
                2 => limits.vectors.max_document_bytes = 1,
                _ => limits.max_lexical_files = 0,
            }
            assert!(NativeBuiltHybridIndex::recover_selected_source(
                &cx, &directory, &receipt, limits,
            ).is_err());
        }
        cx.set_cancel_requested(true);
        assert!(matches!(recover(&cx, &directory, &receipt), Err(SearchError::Cancelled { .. })));
        cx.set_cancel_requested(false);
        assert_eq!(image(&directory), before);
        assert_eq!(recover(&cx, &directory, &receipt).unwrap().1.len(), 2);
    });
}

#[test]
fn recovery_refuses_a_symlinked_source_and_does_not_follow_another_cohort() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("index");
        let (index, receipt, _, _) = fixture(&cx, &directory, true, false).await;
        drop(index);
        let path = directory.join("native.source.jsonl");
        let saved = temp.path().join("saved-source");
        fs::rename(&path, &saved).unwrap();
        std::os::unix::fs::symlink(&saved, &path).unwrap();
        let before = fs::read(&saved).unwrap();
        assert!(recover(&cx, &directory, &receipt).is_err());
        assert_eq!(fs::read(saved).unwrap(), before);
        assert!(fs::symlink_metadata(path).unwrap().file_type().is_symlink());
    });
}

#[test]
fn an_empty_selected_cohort_recovers_with_a_zero_document_limit() {
    run_test_with_cx(|cx| async move {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("empty");
        let provider = Provider::new("empty-source");
        let generation = ArtifactGenerationIdentityV1::new(17, [0x54; 16]).unwrap();
        let index = NativeIndexBuilder::new(&directory, generation, provider.clone()).unwrap()
            .build_hybrid(&cx).await.unwrap();
        let receipt = index.seal_for_reopen(&cx).unwrap();
        drop(index);
        let mut limits = NativeHybridReopenLimits::default();
        limits.vectors.max_documents = 0;
        let (observed, docs) = NativeBuiltHybridIndex::recover_selected_source(
            &cx, &directory, &receipt, limits,
        ).unwrap();
        assert_eq!(observed, generation);
        assert!(docs.is_empty());
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    });
}
