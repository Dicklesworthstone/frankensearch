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
        "{\"id\":\"   \",\"content\":\"invalid\"}",
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
            let expected = search(&index, &cx, "retry network", Mode::Full, 2, [None, None]).await.unwrap();
            assert_eq!(expected["phase"], "refined");
            assert_eq!(expected["results"][0]["doc_id"], "retry.rs");
            drop(index);
            let counts = (fast_calls.load(Ordering::Relaxed), quality_calls.load(Ordering::Relaxed));
            let selected = Selection::read(&receipt_path).unwrap();
            let reopened = selected.open(&cx, models.clone()).await.unwrap();
            assert_eq!(fast_calls.load(Ordering::Relaxed), counts.0);
            assert_eq!(quality_calls.load(Ordering::Relaxed), counts.1);
            assert_eq!(reopened.vectors().document("retry.rs").unwrap().metadata["language"], "rust");
            let actual = search(&reopened, &cx, "retry network", Mode::Full, 2, [None, None]).await.unwrap();
            assert_eq!(actual, expected);
            let fast_before = fast_calls.load(Ordering::Relaxed);
            search(&reopened, &cx, "retry network", Mode::Quality, 2, [None, None]).await.unwrap();
            assert_eq!(fast_calls.load(Ordering::Relaxed), fast_before);
            let quality_before = quality_calls.load(Ordering::Relaxed);
            search(&reopened, &cx, "retry network", Mode::Fast, 2, [None, None]).await.unwrap();
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
        assert_eq!(search(&index, &cx, "retry", Mode::Full, 2, [None, None]).await.unwrap()["phase"], "initial");
        assert!(search(&index, &cx, "retry", Mode::Quality, 2, [None, None]).await.is_err());
        drop(index);
        assert!(selection.open(&cx, providers.clone()).await.is_ok());
        let cancelled = temporary.path().join("cancelled");
        cx.set_cancel_requested(true);
        assert!(build(&cx, &options, &cancelled, source(), providers).await.is_err());
        cx.set_cancel_requested(false);
        assert!(!cancelled.exists());
    });
}

fn output_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes.split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

struct ObservedOutput {
    bytes: Vec<u8>,
    quality_calls: Arc<AtomicUsize>,
    before_quality: usize,
    initial_flushes: usize,
}

impl Write for ObservedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let frames = output_frames(&self.bytes);
        let frame = frames.last().unwrap();
        if frame["phase"] == "initial" {
            assert_eq!(
                self.quality_calls.load(Ordering::Relaxed),
                self.before_quality,
                "quality inference must not begin until Initial has been flushed",
            );
            self.initial_flushes += 1;
        }
        if frame["event"] == "terminal" {
            self.before_quality = self.quality_calls.load(Ordering::Relaxed);
        }
        Ok(())
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn warm_serve_flushes_initial_before_quality_and_survives_bad_requests() {
    run_test_with_cx(|cx| async move {
        let temporary = tempfile::tempdir().unwrap();
        let options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (providers, fast_calls, quality_calls) = fixture_models(true);
        let (index, _) = build(&cx, &options, &temporary.path().join("index"), source(), providers)
            .await
            .unwrap();
        let fast_before = fast_calls.load(Ordering::Relaxed);
        let quality_before = quality_calls.load(Ordering::Relaxed);
        let mut output = ObservedOutput {
            bytes: Vec::new(),
            quality_calls: Arc::clone(&quality_calls),
            before_quality: quality_before,
            initial_flushes: 0,
        };
        let mut input = Cursor::new(concat!(
            "{invalid JSON}\n",
            "{\"id\":\"a\",\"query\":\"retry network\"}\n",
            "{\"id\":\"invalid\",\"query\":\"retry\",\"limit\":0}\n",
            "{\"id\":\"b\",\"query\":\"garden flowers\"}\n",
            "quit\n",
            "{\"id\":\"never\",\"query\":\"must not run\"}\n"
        ));
        let live = serve::NativeLiveHybridIndex::new(&cx, index).unwrap();
        serve::run(&live, &cx, &mut input, &mut output, (Mode::Full, 2), false, None)
            .await
            .unwrap();
        assert_eq!(output.initial_flushes, 2);
        assert_eq!(fast_calls.load(Ordering::Relaxed), fast_before + 2);
        assert_eq!(quality_calls.load(Ordering::Relaxed), quality_before + 2);
        let frames = output_frames(&output.bytes);
        assert_eq!(frames[0]["event"], "ready");
        assert_eq!(frames[1]["status"], "failed");
        for id in ["a", "b"] {
            let request = frames.iter().filter(|frame| frame["id"] == id).collect::<Vec<_>>();
            assert_eq!(request.len(), 4);
            assert_eq!(request[0]["event"], "started");
            assert_eq!(request[1]["phase"], "initial");
            assert_eq!(request[1]["candidates"]["quality"], 0);
            assert_eq!(request[2]["phase"], "refined");
            assert!(request[2]["candidates"]["quality"].as_u64().unwrap() > 0);
            assert_eq!(request[3]["status"], "complete");
            for (seq, frame) in request.iter().enumerate() {
                assert_eq!(frame["seq"], seq);
                assert_eq!(frame["generation"], frames[0]["generation"]);
            }
        }
        assert!(frames.iter().any(|frame| frame["id"] == "invalid" && frame["status"] == "failed"));
        assert!(!frames.iter().any(|frame| frame["id"] == "never"));
    });
}

struct FailingQuality {
    inner: Arc<dyn Embedder>,
    fail: Arc<std::sync::atomic::AtomicBool>,
}

impl Embedder for FailingQuality {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { self.inner.identity() }
    fn dimension(&self) -> usize { self.inner.dimension() }
    fn id(&self) -> &str { self.inner.id() }
    fn model_name(&self) -> &str { self.inner.model_name() }
    fn is_semantic(&self) -> bool { true }
    fn category(&self) -> ModelCategory { self.inner.category() }
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            if self.fail.swap(false, Ordering::AcqRel) {
                return Err(frankensearch::SearchError::InvalidConfig {
                    field: "test.quality".to_owned(),
                    value: "injected".to_owned(),
                    reason: "one quality request failed".to_owned(),
                });
            }
            self.inner.embed(cx, text).await
        })
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn warm_serve_retains_initial_on_quality_failure_and_runs_the_next_query() {
    run_test_with_cx(|cx| async move {
        let temporary = tempfile::tempdir().unwrap();
        let options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (mut providers, _, _) = fixture_models(true);
        let fail = Arc::new(std::sync::atomic::AtomicBool::new(false));
        providers.quality = Some(Arc::new(FailingQuality {
            inner: providers.quality.take().unwrap(),
            fail: Arc::clone(&fail),
        }));
        let (index, _) = build(&cx, &options, &temporary.path().join("index"), source(), providers)
            .await
            .unwrap();
        fail.store(true, Ordering::Release);
        let mut input = Cursor::new(concat!(
            "{\"id\":\"first\",\"query\":\"retry network\"}\n",
            "{\"id\":\"second\",\"query\":\"retry network\"}\n"
        ));
        let mut output = Vec::new();
        let live = serve::NativeLiveHybridIndex::new(&cx, index).unwrap();
        serve::run(&live, &cx, &mut input, &mut output, (Mode::Full, 2), false, None).await.unwrap();
        let frames = output_frames(&output);
        let first = frames.iter().filter(|frame| frame["id"] == "first").collect::<Vec<_>>();
        assert_eq!(first[1]["phase"], "initial");
        assert_eq!(first[2]["phase"], "refinement_failed");
        assert_eq!(first[2]["results"], first[1]["results"]);
        assert_eq!(first[3]["status"], "degraded");
        assert_eq!(first[3]["ok"], false);
        assert_eq!(first[3]["partial_results"], true);
        let second = frames.iter().filter(|frame| frame["id"] == "second").collect::<Vec<_>>();
        assert_eq!(second[2]["phase"], "refined");
        assert_eq!(second[3]["status"], "complete");
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn broken_initial_delivery_stops_before_quality_inference() {
    struct ClosedOutput {
        writes: usize,
    }
    impl Write for ClosedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.writes == 1 {
                Ok(bytes.len()) // Deliver Started, then fail the Initial frame.
            } else {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed client"))
            }
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    run_test_with_cx(|cx| async move {
        let temporary = tempfile::tempdir().unwrap();
        let options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (providers, fast_calls, quality_calls) = fixture_models(true);
        let (index, _) = build(&cx, &options, &temporary.path().join("index"), source(), providers)
            .await
            .unwrap();
        let before = (fast_calls.load(Ordering::Relaxed), quality_calls.load(Ordering::Relaxed));
        let request = serve::Request {
            id: None, query: "retry".to_owned(), mode: None, limit: None, filter: None,
        };
        let mut output = ClosedOutput { writes: 0 };
        assert!(serve::stream_one(&index, &cx, &request, 1, (Mode::Full, 2), &mut output, None)
            .await.is_err());
        assert_eq!(output.writes, 2, "no terminal write after failed delivery");
        assert_eq!(fast_calls.load(Ordering::Relaxed), before.0 + 1);
        assert_eq!(quality_calls.load(Ordering::Relaxed), before.1);
    });
}

#[test]
fn update_admission_preserves_last_edit_wins_and_rejects_invalid_records() {
    let options = options(&[
        "update", "--receipt", "old", "--index-dir", "new", "--new-receipt", "new.json",
    ]);
    assert_eq!(options.command, Command::Update);
    let edits = update::read_edits(&mut Cursor::new(concat!(
        "{\"op\":\"upsert\",\"id\":\"a\",\"content\":\"first\"}\n",
        "{\"op\":\"delete\",\"id\":\"a\"}\n",
        "{\"op\":\"upsert\",\"id\":\"a\",\"content\":\"final\"}\n",
        "{\"op\":\"delete\",\"id\":\"missing\"}"
    ))).unwrap();
    assert_eq!(edits.len(), 2);
    assert_eq!(edits["a"].as_ref().unwrap().content, "final");
    assert!(edits["missing"].is_none());
    for input in [
        "{\"op\":\"delete\",\"id\":\"\"}",
        "{\"op\":\"delete\",\"id\":\"a\",\"content\":\"unexpected\"}",
        "{\"op\":\"upsert\",\"id\":\"a\"}",
        "{\"op\":\"delete\",\"id\":\"a\"}\ninvalid",
    ] {
        assert!(update::read_edits(&mut Cursor::new(input)).is_err());
    }
    assert!(Options::parse([
        "update", "--receipt", "old", "--index-dir", "new",
    ].map(str::to_owned)).is_err());
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn selected_update_reuses_unchanged_vectors_and_keeps_old_readers() {
    run_test_with_cx(|cx| async move {
        let temporary = tempfile::tempdir().unwrap();
        let options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (models, fast_calls, quality_calls) = fixture_models(true);
        let (old, old_selection) = build(&cx, &options, &temporary.path().join("old"), source(), models.clone())
            .await.unwrap();
        let before_fast = fs::read(old.vectors().fast().vector_path()).unwrap();
        let before_quality = fs::read(old.vectors().quality().unwrap().vector_path()).unwrap();
        assert!(new_path(&old.vectors().directory().join("nested-index")).is_err());
        let alias = temporary.path().join("old-alias");
        std::os::unix::fs::symlink(old.vectors().directory(), &alias).unwrap();
        assert!(new_path(&alias.join("nested-receipt.json")).is_err());
        assert!(new_path(&temporary.path().join("new-sibling")).is_ok());
        let counts = (fast_calls.load(Ordering::Relaxed), quality_calls.load(Ordering::Relaxed));
        let edits = update::read_edits(&mut Cursor::new(concat!(
            "{\"op\":\"upsert\",\"id\":\"retry.rs\",\"content\":\"retry network requests with backoff\",\"title\":\"Updated title\",\"metadata\":{\"version\":\"two\"}}\n",
            "{\"op\":\"delete\",\"id\":\"garden.md\"}\n",
            "{\"op\":\"upsert\",\"id\":\"river.md\",\"content\":\"river currents\"}\n"
        ))).unwrap();
        let (new, selection) = update::apply(
            &cx, &old_selection, &old, &temporary.path().join("new"), edits, 8,
        ).await.unwrap();
        assert_eq!(selection.generation.sequence, old_selection.generation.sequence + 1);
        assert_eq!(fast_calls.load(Ordering::Relaxed), counts.0 + 1);
        assert_eq!(quality_calls.load(Ordering::Relaxed), counts.1 + 1);
        assert!(new.vectors().document("garden.md").is_none());
        assert_eq!(new.vectors().document("retry.rs").unwrap().metadata["version"], "two");
        assert_eq!(new.vectors().document("retry.rs").unwrap().title.as_deref(), Some("Updated title"));
        assert!(new.lexical().search(&cx, "garden", 10).await.unwrap().is_empty());
        assert_eq!(old.lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
        assert_eq!(new.lexical().search(&cx, "river", 10).await.unwrap().len(), 1);
        assert!(old.vectors().document("river.md").is_none());
        assert_eq!(fs::read(old.vectors().fast().vector_path()).unwrap(), before_fast);
        assert_eq!(fs::read(old.vectors().quality().unwrap().vector_path()).unwrap(), before_quality);
        drop(new);
        let receipt = temporary.path().join("new.json");
        save_selection(&selection, &receipt).unwrap();
        let reopened = Selection::read(&receipt).unwrap().open(&cx, models).await.unwrap();
        assert_eq!(reopened.vectors().documents().len(), 2);
        assert_eq!(reopened.lexical().search(&cx, "river", 10).await.unwrap().len(), 1);
        assert_eq!(old.lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn failed_and_cancelled_updates_do_not_modify_the_selected_predecessor() {
    run_test_with_cx(|cx| async move {
        let temporary = tempfile::tempdir().unwrap();
        let options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (mut models, _, _) = fixture_models(true);
        let fail = Arc::new(std::sync::atomic::AtomicBool::new(false));
        models.quality = Some(Arc::new(FailingQuality {
            inner: models.quality.take().unwrap(), fail: Arc::clone(&fail),
        }));
        let (old, selection) = build(&cx, &options, &temporary.path().join("old"), source(), models.clone())
            .await.unwrap();
        let receipt = temporary.path().join("old.json");
        save_selection(&selection, &receipt).unwrap();
        let before = fs::read(&receipt).unwrap();
        let vectors = fs::read(old.vectors().fast().vector_path()).unwrap();
        let changes = || update::read_edits(&mut Cursor::new(
            "{\"op\":\"upsert\",\"id\":\"new\",\"content\":\"river\"}\n"
        )).unwrap();
        fail.store(true, Ordering::Release);
        assert!(update::apply(&cx, &selection, &old, &temporary.path().join("failed"), changes(), 8)
            .await.is_err());
        cx.set_cancel_requested(true);
        let cancelled = temporary.path().join("cancelled");
        assert!(update::apply(&cx, &selection, &old, &cancelled, changes(), 8).await.is_err());
        cx.set_cancel_requested(false);
        assert!(!cancelled.exists());
        assert_eq!(fs::read(&receipt).unwrap(), before);
        assert_eq!(fs::read(old.vectors().fast().vector_path()).unwrap(), vectors);
        let reopened = Selection::read(&receipt).unwrap().open(&cx, models).await.unwrap();
        assert_eq!(reopened.vectors().documents().len(), 2);
        assert!(reopened.vectors().document("new").is_none());
    });
}

#[path = "activation_tests.rs"]
mod activation_tests;

#[path = "filter_tests.rs"]
mod filter_tests;
