//! Executable recovery through real partition writers, receipts and search.
//! Reuse the update fixtures; no replacement recovery parser or mock index.
use super::*;
use super::sharded_tests::{Fixture, fixture};
use crate::{Arc, Mode, filter, query};
use asupersync::test_utils::run_test_with_cx;
use frankensearch::{Embedder, ModelCategory, SearchError, SearchFuture, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use std::cell::Cell;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

struct Replacement {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Replacement {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            calls: AtomicUsize::new(0),
        })
    }
}

impl Embedder for Replacement {
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
        usize::try_from(self.identity.space.dimension).unwrap()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut values = vec![0.0; self.dimension()];
            values[0] = 1.0;
            Ok(values)
        })
    }
}

fn options(f: &Fixture, name: &str) -> Options {
    Options::parse([
        "rebuild".to_owned(),
        "--receipt".to_owned(), f.root.path().join("old.json").display().to_string(),
        "--index-dir".to_owned(), f.root.path().join(name).display().to_string(),
        "--new-receipt".to_owned(), f.root.path().join(format!("{name}.json")).display().to_string(),
        "--shard-size".to_owned(), "3".to_owned(),
    ]).unwrap().unwrap()
}

fn no_candidate(options: &Options, output: &[u8]) {
    assert!(output.is_empty());
    assert!(!options.new_receipt.as_ref().unwrap().exists());
}

fn hide_search_artifacts(f: &Fixture) {
    let saved = f.root.path().join("saved-artifacts");
    fs::create_dir(&saved).unwrap();
    for (ordinal, part) in f.index.vectors().partitions().iter().enumerate() {
        for (name, tier) in [("fast", Some(part.fast())), ("quality", part.quality())] {
            if let Some(tier) = tier {
                fs::rename(tier.vector_path(), saved.join(format!("{ordinal}-{name}.fsvi"))).unwrap();
                if let Some(graph) = tier.graph_path() {
                    fs::rename(graph, saved.join(format!("{ordinal}-{name}.hnsw"))).unwrap();
                    let mut receipt = graph.as_os_str().to_os_string();
                    receipt.push(".receipt");
                    fs::rename(receipt, saved.join(format!("{ordinal}-{name}.receipt"))).unwrap();
                }
            }
        }
    }
    fs::rename(f.selection.directory.join("lexical"), saved.join("lexical")).unwrap();
}

#[test]
fn rebuild_repartitions_all_sources_with_new_producers_despite_missing_search_artifacts() {
    run_test_with_cx(|cx| async move {
        for graphs in [false, true] {
            let f = fixture(&cx, graphs, true).await;
            let expected = serde_json::to_value(f.index.vectors().documents().collect::<Vec<_>>()).unwrap();
            let old_receipt = fs::read(f.root.path().join("old.json")).unwrap();
            hide_search_artifacts(&f);
            assert!(sharded::open(&cx, &f.selection, f.models()).await.is_err());
            f.fast.fault.store(1, Ordering::SeqCst);
            f.quality.as_ref().unwrap().fault.store(1, Ordering::SeqCst);
            let old_submitted = f.fast.submitted.load(Ordering::SeqCst);
            let mut options = options(&f, "rebuilt");
            options.exact = !graphs;
            let fast = Replacement::new("replacement-fast", 4);
            let quality = Replacement::new("replacement-quality", 6);
            let models = Models { fast: fast.clone(), quality: Some(quality.clone()) };
            let loads = Cell::new(0);
            let mut output = Vec::new();
            rebuild_with_loader(&cx, &options, &mut output, |_, required| {
                assert!(required);
                assert!(!options.directory.as_ref().unwrap().exists());
                loads.set(loads.get() + 1);
                Ok(models.clone())
            }).await.unwrap();
            assert_eq!(loads.get(), 1);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 5);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 5);
            assert_eq!(f.fast.submitted.load(Ordering::SeqCst), old_submitted);
            let receipt = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
            assert_eq!(receipt.schema, sharded::SELECTION_SCHEMA);
            assert_eq!(receipt.generation.sequence, f.selection.generation.sequence + 1);
            assert_ne!(receipt.generation, f.selection.generation);
            assert_eq!(receipt.fast_producer, fast.identity.fingerprint());
            assert_eq!(receipt.quality_producer, Some(quality.identity.fingerprint()));
            assert_ne!(receipt.fast_producer, f.selection.fast_producer);
            let frame: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(frame["event"], "rebuilt");
            assert_eq!(frame["vectors_reused"], false);
            assert_eq!(frame["selection"], serde_json::to_value(&receipt).unwrap());
            let reopened = sharded::open(&cx, &receipt, models).await.unwrap();
            assert_eq!(fast.calls.load(Ordering::SeqCst), 5, "restart must not infer");
            assert_eq!(reopened.vectors().partitions().len(), 2);
            assert_eq!(serde_json::to_value(reopened.vectors().documents().collect::<Vec<_>>()).unwrap(), expected);
            assert!(sharded::open(&cx, &receipt, f.models()).await.is_err());
            let scope = filter::Filter::parse(r#"{"metadata":{"version":"old"}}"#).unwrap();
            for mode in [Mode::Fast, Mode::Quality, Mode::Full] {
                let page = query::buffered(
                    &reopened, &cx, "common", mode, 10, [None, Some(&scope)],
                    &query::Policy::default(),
                ).await.unwrap();
                assert_eq!(page["layout"], "sharded");
                assert_eq!(page["scope"]["eligible_documents"], 5);
                assert_eq!(page["results"].as_array().unwrap().len(), 5);
            }
            for part in reopened.vectors().partitions() {
                assert_eq!(part.fast().graph_path().is_some(), graphs);
                assert_eq!(part.quality().unwrap().graph_path().is_some(), graphs);
            }
            assert_eq!(fs::read(f.root.path().join("old.json")).unwrap(), old_receipt);
            assert!(!f.selection.directory.join("lexical").exists(), "no repair in place");
            assert_eq!(f.index.lexical().search(&cx, "obsolete", 10).await.unwrap()[0].doc_id, "c");
        }
    });
}

#[test]
fn source_and_receipt_failures_never_load_a_model_or_create_a_candidate() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        for (ordinal, relative) in [
            "native.sharded-hybrid.json", "native.sharded.json",
            "shard-000000/native.snapshot.json", "shard-000002/native.source.jsonl",
        ].into_iter().enumerate() {
            let options = options(&f, &format!("bad-{ordinal}"));
            let path = f.selection.directory.join(relative);
            let before = fs::read(&path).unwrap();
            let mut damaged = before.clone();
            damaged[0] ^= 1;
            fs::write(&path, &damaged).unwrap();
            let loaded = Cell::new(false);
            let mut output = Vec::new();
            assert!(rebuild_with_loader(&cx, &options, &mut output, |_, _| {
                loaded.set(true);
                Err(bad("unexpected model load"))
            }).await.is_err());
            assert!(!loaded.get(), "{relative}");
            assert!(!options.directory.as_ref().unwrap().exists());
            no_candidate(&options, &output);
            assert_eq!(fs::read(&path).unwrap(), damaged);
            fs::write(path, before).unwrap();
        }
        let options = options(&f, "wrong-receipt-count");
        let mut selection: Selection = serde_json::from_slice(
            &fs::read(&options.receipt).unwrap(),
        ).unwrap();
        selection.documents -= 1;
        fs::write(&options.receipt, encode(&selection, crate::MAX_RECEIPT_BYTES).unwrap()).unwrap();
        assert!(rebuild_with_loader(&cx, &options, &mut Vec::new(), |_, _| {
            panic!("count mismatch must precede models"); // ubs:ignore — test assertion.
        }).await.is_err());
        assert!(!options.directory.as_ref().unwrap().exists());
    });
}

#[test]
fn shard_policy_and_destinations_are_explicit_before_model_loading() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, false).await;
        for size in [None, Some(0), Some(MAX_DOCUMENTS + 1)] {
            let mut options = options(&f, "policy");
            options.shard_size = size;
            assert!(rebuild_with_loader(&cx, &options, &mut Vec::new(), |_, _| {
                panic!("invalid shard policy must precede model loading"); // ubs:ignore — test assertion.
            }).await.is_err());
            assert!(!options.directory.as_ref().unwrap().exists());
        }
        for relative in ["nested", "lexical/nested", "shard-000002/nested"] {
            let mut options = options(&f, "overlap");
            options.directory = Some(f.selection.directory.join(relative));
            assert!(rebuild_with_loader(&cx, &options, &mut Vec::new(), |_, _| {
                panic!("sealed ancestors must be refused before model loading"); // ubs:ignore — test assertion.
            }).await.is_err());
            assert!(!options.directory.as_ref().unwrap().exists());
        }
        let mut options = options(&f, "partition-limit");
        options.shard_size = Some(1);
        let mut selected: Selection = serde_json::from_slice(&fs::read(&options.receipt).unwrap()).unwrap();
        selected.documents = 1025;
        fs::write(&options.receipt, encode(&selected, crate::MAX_RECEIPT_BYTES).unwrap()).unwrap();
        let error = rebuild_with_loader(&cx, &options, &mut Vec::new(), |_, _| {
            panic!("partition limit must precede model loading"); // ubs:ignore — test assertion.
        }).await.unwrap_err();
        assert!(error.to_string().contains("1024-partition"));
        assert!(!options.directory.as_ref().unwrap().exists());
    });
}

#[test]
fn missing_models_required_quality_failure_and_cancellation_leave_no_success_receipt() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        let old_receipt = fs::read(f.root.path().join("old.json")).unwrap();
        let missing = options(&f, "missing-models");
        let loaded = Cell::new(false);
        let mut output = Vec::new();
        assert!(rebuild_with_loader(&cx, &missing, &mut output, |_, _| {
            loaded.set(true);
            Err(bad("required local model is missing"))
        }).await.is_err());
        assert!(loaded.get());
        assert!(!missing.directory.as_ref().unwrap().exists());
        no_candidate(&missing, &output);
        let failed = options(&f, "quality-failed");
        f.quality.as_ref().unwrap().fault.store(1, Ordering::SeqCst);
        assert!(rebuild_with_loader(&cx, &failed, &mut output, |_, _| Ok(f.models())).await.is_err());
        no_candidate(&failed, &output);
        f.quality.as_ref().unwrap().fault.store(0, Ordering::SeqCst);
        for during_inference in [false, true] {
            let cancelled = options(&f, if during_inference { "cancel-model" } else { "cancel-entry" });
            if during_inference {
                f.fast.fault.store(3, Ordering::SeqCst);
            } else {
                cx.set_cancel_requested(true);
            }
            let error = rebuild_with_loader(&cx, &cancelled, &mut output, |_, _| Ok(f.models()))
                .await.unwrap_err();
            assert!(matches!(error.downcast_ref::<SearchError>(), Some(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
            f.fast.fault.store(0, Ordering::SeqCst);
            no_candidate(&cancelled, &output);
        }
        let retry = options(&f, "recovered-after-errors");
        rebuild_with_loader(&cx, &retry, &mut output, |_, _| Ok(f.models())).await.unwrap();
        assert!(Selection::read(retry.new_receipt.as_ref().unwrap()).is_ok());
        assert_eq!(fs::read(f.root.path().join("old.json")).unwrap(), old_receipt);
    });
}

#[test]
fn dropping_pending_rebuild_releases_inference_without_persisting_a_receipt() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        let options = options(&f, "pending");
        let before = f.fast.drops.load(Ordering::SeqCst);
        f.fast.fault.store(2, Ordering::SeqCst);
        let mut output = Vec::new();
        let mut work = Box::pin(rebuild_with_loader(&cx, &options, &mut output, |_, _| Ok(f.models())));
        assert!(work.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        drop(work);
        assert_eq!(f.fast.drops.load(Ordering::SeqCst), before + 1);
        no_candidate(&options, &output);
        assert!(f.index.vectors().document("z").is_some());
        assert!(Selection::read(&f.root.path().join("old.json")).is_ok());
    });
}

#[test]
fn empty_source_rebuild_retains_explicit_tiers_and_a_single_empty_partition() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, false).await;
        let mut transaction = f.index.begin_update(
            &cx, f.root.path().join("empty"), crate::new_generation(2).unwrap(),
        ).unwrap();
        for source in f.index.vectors().documents() {
            transaction = transaction.delete_document(&source.id);
        }
        let empty = transaction.build_sharded_hybrid(&cx, 9).await.unwrap();
        let selected = seal_selection(&cx, (&empty).into()).unwrap();
        let receipt = f.root.path().join("empty.json");
        save_selection(&selected, &receipt).unwrap();
        for fast_only in [false, true] {
            let mut options = options(&f, if fast_only { "empty-fast" } else { "empty-both" });
            options.receipt = receipt.clone();
            options.fast_only = fast_only;
            options.shard_size = Some(10);
            let fast = Replacement::new("empty-fast", 4);
            let quality = Replacement::new("empty-quality", 6);
            let models = Models {
                fast: fast.clone(),
                quality: (!fast_only).then(|| quality.clone() as Arc<dyn Embedder>),
            };
            let mut output = Vec::new();
            rebuild_with_loader(&cx, &options, &mut output, |_, required| {
                assert_eq!(required, !fast_only, "old topology cannot silently choose new policy");
                Ok(models.clone())
            }).await.unwrap();
            let selected = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
            assert_eq!(selected.generation.sequence, 3);
            let opened = sharded::open(&cx, &selected, models).await.unwrap();
            assert_eq!(opened.vectors().document_count(), 0);
            assert_eq!(opened.vectors().partitions().len(), 1);
            assert_eq!(opened.vectors().quality().is_some(), !fast_only);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn output_failure_does_not_undo_a_complete_saved_sharded_rebuild() {
    struct BrokenOutput;
    impl Write for BrokenOutput {
        fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "lost receipt acknowledgement"))
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        let options = options(&f, "lost-ack");
        assert!(rebuild_with_loader(&cx, &options, &mut BrokenOutput, |_, _| Ok(f.models()))
            .await.is_err());
        let selected = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
        assert_eq!(selected.generation.sequence, 2);
        let reopened = sharded::open(&cx, &selected, f.models()).await.unwrap();
        assert_eq!(reopened.vectors().document_count(), 5);
        let before = fs::read(options.new_receipt.as_ref().unwrap()).unwrap();
        assert!(rebuild_with_loader(&cx, &options, &mut Vec::new(), |_, _| {
            panic!("retry cannot overwrite a completed generation"); // ubs:ignore — test assertion.
        }).await.is_err());
        assert_eq!(fs::read(options.new_receipt.as_ref().unwrap()).unwrap(), before);
        assert_eq!(f.index.vectors().generation(), f.selection.generation);
    });
}
