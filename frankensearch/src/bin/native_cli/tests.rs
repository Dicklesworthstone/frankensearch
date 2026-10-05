use super::*;
use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::{EmbeddingIdentityBundleV1, ModelCategory, SearchFuture, SearchResult};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};

fn options(args: &[&str]) -> Options {
    Options::parse(args.iter().map(|arg| (*arg).to_owned()))
        .unwrap()
        .unwrap()
}

#[test]
fn parser_requires_explicit_destinations_and_rejects_inapplicable_options() {
    let index = options(&["index", "--index-dir", "new", "--receipt", "selected.json"]);
    assert_eq!(index.command, Command::Index);
    assert!(!index.fast_only);
    assert!(!index.exact);
    let search = options(&[
        "search", "--receipt", "selected.json", "--query", "-obsolete retry", "--mode", "fast",
    ]);
    assert_eq!(search.mode, Mode::Fast);
    for args in [
        vec!["index", "--receipt", "selected.json"],
        vec!["index", "--index-dir", "new"],
        vec!["search", "--receipt", "selected.json"],
        vec!["search", "--receipt", "selected.json", "--query", "x", "--exact"],
        vec!["index", "--index-dir", "new", "--receipt", "r", "--batch-size", "0"],
        vec!["search", "--receipt", "r", "--query", "x", "--limit", "1001"],
        vec!["search", "--receipt", "r", "--receipt", "other", "--query", "x"],
        vec!["index", "--index-dir", "--receipt", "r"],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
}

#[test]
fn documents_preserve_source_and_reject_the_entire_invalid_batch() {
    let input = concat!(
        " \r\n",
        "{\"id\":\"東京\",\"content\":\"café 🦀\",\"title\":\"notes\",",
        "\"metadata\":{\"language\":\"rust\"}}\r\n",
        "{\"id\":\"empty\",\"content\":\"\"}"
    );
    let documents = read_documents(&mut Cursor::new(input)).unwrap();
    assert_eq!(documents.len(), 2);
    assert_eq!(documents[0].id, "東京");
    assert_eq!(documents[0].content, "café 🦀");
    assert_eq!(documents[0].title.as_deref(), Some("notes"));
    assert_eq!(documents[0].metadata["language"], "rust");
    assert!(documents[1].content.is_empty());
    for invalid in [
        "{\"id\":\"x\",\"content\":\"first\"}\n{\"id\":\"x\",\"content\":\"second\"}",
        "{\"id\":\" \t\",\"content\":\"invalid\"}",
        "{\"id\":\"x\\u0000y\",\"content\":\"invalid\"}",
        "{\"id\":\"x\",\"content\":\"valid\"}\nnot JSON",
        "{\"id\":\"x\",\"content\":\"valid\",\"ignored_policy\":true}",
        "{\"id\":\"x\",\"text\":\"wrong input field\"}",
    ] {
        assert!(read_documents(&mut Cursor::new(invalid)).is_err());
    }
    let error = read_documents(&mut Cursor::new(
        b"{\"id\":\"x\",\"content\":{\"secret-body\":42}}",
    ))
    .unwrap_err();
    assert!(!error.to_string().contains("secret-body"));
    assert!(read_documents(&mut Cursor::new([0xff])).is_err());
}

#[test]
fn unterminated_input_and_json_output_are_bounded() {
    let oversized = vec![b'x'; MAX_RECORD_BYTES + 1];
    assert!(read_documents(&mut Cursor::new(oversized)).is_err());
    assert_eq!(encode(&"x", 4).unwrap(), b"\"x\"\n");
    assert!(encode(&"x", 3).is_err());
    let mut output = Vec::new();
    assert!(emit(&mut output, &"x".repeat(MAX_OUTPUT_BYTES)).is_err());
    assert!(output.is_empty(), "encode refusal must not expose a partial record");
}

// Identified deterministic providers test actual Quill/FSVI/native-HNSW build,
// persistence and serving, not retrieval quality or production model loading.
struct FixtureModel {
    identity: EmbeddingIdentityBundleV1,
    category: ModelCategory,
    calls: Arc<AtomicUsize>,
}

impl Embedder for FixtureModel {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(if text.contains("retry") {
                vec![1.0, 0.0, 0.0]
            } else if text.contains("garden") {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0]
            })
        })
    }

    fn dimension(&self) -> usize { 3 }
    fn id(&self) -> &str { &self.identity.space.logical_model_id }
    fn model_name(&self) -> &str { self.id() }
    fn is_semantic(&self) -> bool { true }
    fn category(&self) -> ModelCategory { self.category }
}

fn fixture_models(quality: bool) -> (Models, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let fast_calls = Arc::new(AtomicUsize::new(0));
    let quality_calls = Arc::new(AtomicUsize::new(0));
    let fast: Arc<dyn Embedder> = Arc::new(FixtureModel {
        identity: EmbeddingIdentityBundleV1::explicit_test_model("native-cli-fast", 3),
        category: ModelCategory::StaticEmbedder,
        calls: Arc::clone(&fast_calls),
    });
    let quality = quality.then(|| {
        Arc::new(FixtureModel {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("native-cli-quality", 3),
            category: ModelCategory::TransformerEmbedder,
            calls: Arc::clone(&quality_calls),
        }) as Arc<dyn Embedder>
    });
    (Models { fast, quality }, fast_calls, quality_calls)
}

fn source() -> Vec<IndexableDocument> {
    let mut retry = IndexableDocument::new("retry.rs", "retry network requests with backoff");
    retry.title = Some("network retry".to_owned());
    retry.metadata.insert("language".to_owned(), "rust".to_owned());
    vec![retry, IndexableDocument::new("garden.md", "garden flowers in spring")]
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn native_and_exact_builds_reopen_both_tiers_without_reembedding() {
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            let temporary = tempfile::tempdir().unwrap();
            let directory = temporary.path().join("index");
            let receipt_path = temporary.path().join("receipt.json");
            let mut options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
            options.exact = exact;
            let (models, fast_calls, quality_calls) = fixture_models(true);
            let (index, selection) = build(&cx, &options, &directory, source(), models.clone())
                .await
                .unwrap();
            assert_eq!(index.vectors().fast().graph_path().is_some(), !exact);
            assert_eq!(index.vectors().quality().unwrap().graph_path().is_some(), !exact);
            save_selection(&selection, &receipt_path).unwrap();
            let expected = search(&index, &cx, "retry network", Mode::Full, 2).await.unwrap();
            assert_eq!(expected["phase"], "refined");
            assert_eq!(expected["results"][0]["doc_id"], "retry.rs");
            drop(index);
            let counts = (fast_calls.load(Ordering::Relaxed), quality_calls.load(Ordering::Relaxed));
            let selected = Selection::read(&receipt_path).unwrap();
            let reopened = selected.open(&cx, models.clone()).await.unwrap();
            assert_eq!(fast_calls.load(Ordering::Relaxed), counts.0);
            assert_eq!(quality_calls.load(Ordering::Relaxed), counts.1);
            assert_eq!(reopened.vectors().document("retry.rs").unwrap().metadata["language"], "rust");
            let actual = search(&reopened, &cx, "retry network", Mode::Full, 2).await.unwrap();
            assert_eq!(actual, expected);
            let fast_before = fast_calls.load(Ordering::Relaxed);
            search(&reopened, &cx, "retry network", Mode::Quality, 2).await.unwrap();
            assert_eq!(fast_calls.load(Ordering::Relaxed), fast_before);
            let quality_before = quality_calls.load(Ordering::Relaxed);
            search(&reopened, &cx, "retry network", Mode::Fast, 2).await.unwrap();
            assert_eq!(quality_calls.load(Ordering::Relaxed), quality_before);
            assert!(new_path(&directory).is_err());
            assert!(save_selection(&selection, &receipt_path).is_err());
            assert!(selected.open(&cx, fixture_models(false).0).await.is_err());
            fs::write(directory.join("native.hybrid.json"), b"corruption").unwrap();
            assert!(selected.open(&cx, models).await.is_err());
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn fast_only_reopen_refuses_quality_queries_and_cancelled_builds_leave_no_destination() {
    run_test_with_cx(|cx| async move {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().join("fast");
        let options = options(&["index", "--index-dir", "unused", "--receipt", "unused", "--fast-only"]);
        let providers = fixture_models(false).0;
        let (index, selection) = build(&cx, &options, &directory, source(), providers.clone())
            .await
            .unwrap();
        assert!(selection.quality_producer.is_none());
        assert_eq!(search(&index, &cx, "retry", Mode::Full, 2).await.unwrap()["phase"], "initial");
        assert!(search(&index, &cx, "retry", Mode::Quality, 2).await.is_err());
        drop(index);
        assert!(selection.open(&cx, providers.clone()).await.is_ok());
        let cancelled = temporary.path().join("cancelled");
        cx.set_cancel_requested(true);
        assert!(build(&cx, &options, &cancelled, source(), providers).await.is_err());
        cx.set_cancel_requested(false);
        assert!(!cancelled.exists());
    });
}
