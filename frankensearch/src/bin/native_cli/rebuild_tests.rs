use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

use asupersync::test_utils::run_test_with_cx;
use frankensearch::{ModelCategory, SearchError, SearchFuture, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;

// Explicit control models test the real CLI build/recover/seal path. They are
// not substitutes for real-model relevance, performance or loader qualification.
struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
    drops: AtomicUsize,
    mode: AtomicUsize,
}

impl Provider {
    fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, 3),
            calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            mode: AtomicUsize::new(0),
        })
    }
}

struct CallDrop<'a>(&'a AtomicUsize);

impl Drop for CallDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn dimension(&self) -> usize {
        3
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _drop = CallDrop(&self.drops);
            match self.mode.load(Ordering::SeqCst) {
                1 => Err(SearchError::InvalidConfig {
                    field: "rebuild.test.provider".to_owned(),
                    value: "failed".to_owned(),
                    reason: "injected inference failure".to_owned(),
                }),
                2 => Err(SearchError::Cancelled {
                    phase: "rebuild.test.provider".to_owned(),
                    reason: "injected provider cancellation".to_owned(),
                }),
                3 => std::future::pending().await,
                _ => Ok(if text.contains("retry") {
                    vec![1.0, 0.0, 0.0]
                } else {
                    vec![0.0, 1.0, 0.0]
                }),
            }
        })
    }
}

fn source() -> Vec<IndexableDocument> {
    let mut retry = IndexableDocument::new("retry.rs", "retry network requests café 🦀");
    retry.title = Some("東京 title".to_owned());
    retry.metadata.insert("language".to_owned(), "rust".to_owned());
    vec![retry, IndexableDocument::new("garden.md", "garden flowers")]
}

fn options(root: &Path, name: &str) -> Options {
    Options::parse([
        "rebuild".to_owned(),
        "--receipt".to_owned(),
        root.join("old.json").display().to_string(),
        "--index-dir".to_owned(),
        root.join(name).display().to_string(),
        "--new-receipt".to_owned(),
        root.join(format!("{name}.json")).display().to_string(),
    ]).unwrap().unwrap()
}

struct Fixture {
    index: NativeBuiltHybridIndex,
    selection: Selection,
    models: Models,
    fast: Arc<Provider>,
    quality: Arc<Provider>,
}

async fn fixture(cx: &Cx, root: &Path, quality: bool, sequence: u64) -> Fixture {
    let fast = Provider::new("old-fast");
    let slow = Provider::new("old-quality");
    let models = Models {
        fast: fast.clone(),
        quality: quality.then(|| slow.clone() as Arc<dyn Embedder>),
    };
    let options = options(root, "unused");
    let (index, selection) = build_with_generation(
        cx, &options, &root.join("old"),
        ArtifactGenerationIdentityV1::new(sequence, [0x41; 16]).unwrap(),
        source(), models.clone(),
    ).await.unwrap();
    save_selection(&selection, &root.join("old.json")).unwrap();
    Fixture { index, selection, models, fast, quality: slow }
}

fn image(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                pending.push(path);
            } else {
                files.insert(path.strip_prefix(root).unwrap().to_path_buf(), fs::read(path).unwrap());
            }
        }
    }
    files
}

fn assert_no_candidate(options: &Options, output: &[u8]) {
    assert!(output.is_empty());
    assert!(!options.directory.as_ref().unwrap().exists());
    assert!(!options.new_receipt.as_ref().unwrap().exists());
}

#[test]
fn rebuild_requires_fresh_destinations_and_never_accepts_stdin_or_search_options() {
    let root = tempfile::tempdir().unwrap();
    let parsed = options(root.path(), "new");
    assert_eq!(parsed.command, Command::Rebuild);
    assert!(!parsed.fast_only);
    for flag in ["--input", "--query", "--filter", "--timeout-ms", "--allow-activation"] {
        let args = [
            "rebuild", "--receipt", "old", "--index-dir", "new", "--new-receipt", "new.json",
            flag, "value",
        ];
        assert!(Options::parse(args.map(str::to_owned)).is_err(), "{flag}");
    }
    for args in [
        vec!["rebuild", "--receipt", "old", "--index-dir", "new"],
        vec!["rebuild", "--receipt", "old", "--new-receipt", "new.json"],
        vec!["rebuild", "--index-dir", "new", "--new-receipt", "new.json"],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
    let parsed = Options::parse([
        "rebuild", "--receipt", "old", "--index-dir", "new", "--new-receipt", "new.json",
        "--fast-only", "--exact", "--batch-size", "2",
    ].map(str::to_owned)).unwrap().unwrap();
    assert!(parsed.fast_only && parsed.exact);
    assert_eq!(parsed.batch_size, 2);
}

#[test]
fn rebuild_recovers_missing_search_artifacts_and_reembeds_with_new_producers() {
    run_test_with_cx(|cx| async move {
        for (old_quality, new_quality, exact) in [(true, true, false), (false, true, true), (true, false, false)] {
            let root = tempfile::tempdir().unwrap();
            let old = fixture(&cx, root.path(), old_quality, 41).await;
            drop(old.index); // Do not alter files while a mapped reader lives.
            let displaced = root.path().join("damaged-artifacts");
            fs::create_dir(&displaced).unwrap();
            for entry in fs::read_dir(&old.selection.directory).unwrap() {
                let entry = entry.unwrap();
                if !matches!(entry.file_name().to_str(),
                    Some("native.hybrid.json" | "native.snapshot.json" | "native.source.jsonl"))
                {
                    fs::rename(entry.path(), displaced.join(entry.file_name())).unwrap();
                }
            }
            assert!(old.selection.open(&cx, old.models).await.is_err());
            let before = image(&old.selection.directory);
            let receipt_before = fs::read(root.path().join("old.json")).unwrap();
            let new_fast = Provider::new("replacement-fast");
            let new_slow = Provider::new("replacement-quality");
            let models = Models {
                fast: new_fast.clone(),
                quality: new_quality.then(|| new_slow.clone() as Arc<dyn Embedder>),
            };
            let mut options = options(root.path(), "rebuilt");
            options.fast_only = !new_quality;
            options.exact = exact;
            let mut output = Vec::new();
            rebuild_with_loader(&cx, &options, &mut output, |_, required_quality| {
                assert_eq!(required_quality, new_quality);
                Ok(models.clone())
            }).await.unwrap();
            assert_eq!(new_fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(new_slow.calls.load(Ordering::SeqCst), if new_quality { 2 } else { 0 });
            assert_eq!(old.fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(old.quality.calls.load(Ordering::SeqCst), if old_quality { 2 } else { 0 });
            let receipt = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
            assert_eq!(receipt.generation.sequence, 42);
            assert_ne!(receipt.generation.nonce, old.selection.generation.nonce);
            assert_eq!(receipt.fast_producer, new_fast.identity.fingerprint());
            let reopened = receipt.open(&cx, models).await.unwrap();
            assert_eq!(reopened.vectors().fast().graph_path().is_some(), !exact);
            assert_eq!(reopened.vectors().quality().is_some(), new_quality);
            assert_eq!(reopened.vectors().document("retry.rs").unwrap().metadata["language"], "rust");
            assert_eq!(reopened.vectors().document("retry.rs").unwrap().content, source()[0].content);
            assert_eq!(reopened.vectors().document("retry.rs").unwrap().title, source()[0].title);
            assert_eq!(reopened.lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
            let response: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(response["event"], "rebuilt");
            assert_eq!(response["vectors_reused"], false);
            assert_eq!(response["selection"]["generation"]["sequence"], 42);
            assert_eq!(image(&old.selection.directory), before);
            assert_eq!(fs::read(root.path().join("old.json")).unwrap(), receipt_before);
        }
    });
}

#[test]
fn corrupt_sources_and_receipt_drift_refuse_before_loading_any_model() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = fixture(&cx, root.path(), true, 4).await;
        drop(old.index);
        let receipt_path = root.path().join("old.json");
        let receipt_before = fs::read(&receipt_path).unwrap();
        for fault in ["native.hybrid.json", "native.snapshot.json", "native.source.jsonl", "count", "generation"] {
            let path = if matches!(fault, "count" | "generation") {
                receipt_path.clone()
            } else {
                old.selection.directory.join(fault)
            };
            let saved = fs::read(&path).unwrap();
            if matches!(fault, "count" | "generation") {
                let mut value: serde_json::Value = serde_json::from_slice(&saved).unwrap();
                if fault == "count" {
                    value["documents"] = serde_json::json!(1);
                } else {
                    value["generation"]["sequence"] = serde_json::json!(3);
                }
                fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            } else {
                let mut corrupt = saved.clone();
                let at = corrupt.len() / 2;
                corrupt[at] ^= 1;
                fs::write(&path, corrupt).unwrap();
            }
            let before = image(root.path());
            let loads = AtomicUsize::new(0);
            let options = options(root.path(), "must-not-exist");
            let mut output = Vec::new();
            assert!(rebuild_with_loader(&cx, &options, &mut output, |_, _| {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(old.models.clone())
            }).await.is_err(), "{fault}");
            assert_eq!(loads.load(Ordering::SeqCst), 0, "{fault}");
            assert_no_candidate(&options, &output);
            assert_eq!(image(root.path()), before);
            fs::write(path, saved).unwrap();
        }
        assert_eq!(fs::read(receipt_path).unwrap(), receipt_before);
    });
}

#[test]
fn inference_failure_cancellation_and_abandoned_rebuild_never_save_a_receipt() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = fixture(&cx, root.path(), true, 8).await;
        let before = image(&old.selection.directory);
        for mode in [1, 2, 3] {
            let fast = Provider::new("fresh-fast");
            let slow = Provider::new("fresh-quality");
            slow.mode.store(mode, Ordering::SeqCst);
            let models = Models { fast, quality: Some(slow.clone()) };
            let options = options(root.path(), &format!("failed-{mode}"));
            let mut output = Vec::new();
            let mut future = Box::pin(rebuild_with_loader(&cx, &options, &mut output, |_, _| Ok(models)));
            if mode == 3 {
                assert!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
                assert_eq!(slow.calls.load(Ordering::SeqCst), 1);
                assert_eq!(slow.drops.load(Ordering::SeqCst), 0);
                drop(future);
                assert_eq!(slow.drops.load(Ordering::SeqCst), 1);
            } else {
                let error = future.await.unwrap_err();
                if mode == 2 {
                    assert!(matches!(error.downcast_ref::<SearchError>(), Some(SearchError::Cancelled { .. })));
                }
            }
            assert!(output.is_empty());
            assert!(!options.new_receipt.as_ref().unwrap().exists());
            assert_eq!(image(&old.selection.directory), before);
        }
        let options = options(root.path(), "cancelled-before-read");
        let mut output = Vec::new();
        let loads = AtomicUsize::new(0);
        cx.set_cancel_requested(true);
        let error = rebuild_with_loader(&cx, &options, &mut output, |_, _| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(old.models.clone())
        }).await.unwrap_err();
        cx.set_cancel_requested(false);
        assert!(matches!(error.downcast_ref::<SearchError>(), Some(SearchError::Cancelled { .. })));
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert_no_candidate(&options, &output);
        assert_eq!(old.index.lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
    });
}

#[test]
fn exhausted_generation_cannot_wrap_or_load_models() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = fixture(&cx, root.path(), false, u64::MAX).await;
        let loads = AtomicUsize::new(0);
        let options = options(root.path(), "exhausted");
        let mut output = Vec::new();
        let error = rebuild_with_loader(&cx, &options, &mut output, |_, _| {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(old.models.clone())
        }).await.unwrap_err();
        assert!(error.to_string().contains("sequence exhausted"));
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        assert_no_candidate(&options, &output);
    });
}

#[test]
fn conflicting_and_aliased_destinations_cannot_modify_the_predecessor() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = fixture(&cx, root.path(), true, 1).await;
        let alias = root.path().join("alias");
        std::os::unix::fs::symlink(&old.selection.directory, &alias).unwrap();
        let before = image(&old.selection.directory);
        let receipt_before = fs::read(root.path().join("old.json")).unwrap();
        for fault in 0..4 {
            let mut options = options(root.path(), "new");
            match fault {
                0 => options.directory = Some(old.selection.directory.clone()),
                1 => options.directory = Some(alias.join("nested")),
                2 => options.new_receipt = Some(options.receipt.clone()),
                _ => options.new_receipt = options.directory.clone(),
            }
            let loads = AtomicUsize::new(0);
            let mut output = Vec::new();
            assert!(rebuild_with_loader(&cx, &options, &mut output, |_, _| {
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(old.models.clone())
            }).await.is_err());
            assert_eq!(loads.load(Ordering::SeqCst), 0);
            assert!(output.is_empty());
            assert_eq!(image(&old.selection.directory), before);
            assert_eq!(fs::read(root.path().join("old.json")).unwrap(), receipt_before);
        }
        assert!(!root.path().join("new").exists());
        assert!(!alias.join("nested").exists());
    });
}

#[test]
fn failed_receipt_delivery_keeps_the_rebuilt_snapshot_and_the_old_live_reader() {
    struct ClosedOutput;
    impl Write for ClosedOutput {
        fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "client closed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = fixture(&cx, root.path(), true, 20).await;
        let before = image(&old.selection.directory);
        let options = options(root.path(), "durable");
        let error = rebuild_with_loader(&cx, &options, &mut ClosedOutput, |_, _| {
            Ok(old.models.clone())
        }).await.unwrap_err();
        assert_eq!(error.downcast_ref::<io::Error>().unwrap().kind(), io::ErrorKind::BrokenPipe);
        let selection = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
        assert_eq!(selection.generation.sequence, 21);
        let fresh = selection.open(&cx, old.models).await.unwrap();
        assert_eq!(fresh.lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
        assert_eq!(old.index.lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
        assert_eq!(image(&old.selection.directory), before);
    });
}

#[test]
fn missing_models_topology_mismatch_and_loading_cancellation_create_nothing() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = fixture(&cx, root.path(), true, 7).await;
        let before = image(&old.selection.directory);
        for fault in 0..3 {
            let options = options(root.path(), "no-model");
            let mut output = Vec::new();
            let result = rebuild_with_loader(&cx, &options, &mut output, |_, required_quality| {
                assert!(required_quality);
                match fault {
                    0 => Err(bad("required local models are unavailable")),
                    1 => Ok(Models { fast: old.models.fast.clone(), quality: None }),
                    _ => {
                        cx.set_cancel_requested(true);
                        Ok(old.models.clone())
                    }
                }
            }).await;
            cx.set_cancel_requested(false);
            let error = result.unwrap_err();
            if fault == 1 {
                assert!(error.to_string().contains("model topology"));
            } else if fault == 2 {
                assert!(matches!(error.downcast_ref::<SearchError>(), Some(SearchError::Cancelled { .. })));
            }
            assert_no_candidate(&options, &output);
            assert_eq!(image(&old.selection.directory), before);
            assert_eq!(old.fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(old.quality.calls.load(Ordering::SeqCst), 2);
        }
    });
}
