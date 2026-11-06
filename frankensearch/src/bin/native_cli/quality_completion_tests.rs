//! Exercise the command with real retained files, builders, receipts and query
//! owners. Counted control producers prove inference reuse, not model relevance.

use super::*;
use crate::{
    Arc, ArtifactGenerationIdentityV1, Command, Embedder, IndexableDocument, Mode,
    NativeIndexBuilder, PathBuf, cohort,
};
use asupersync::test_utils::run_test_with_cx;
use frankensearch::{ModelCategory, SearchFuture, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::ValidatedFsviBytes;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    quality: bool,
    submitted: AtomicUsize,
    fault: AtomicUsize,
}

impl Provider {
    fn new(quality: bool) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(
                if quality {
                    "complete-quality"
                } else {
                    "retained-fast"
                },
                if quality { 3 } else { 2 },
            ),
            quality,
            submitted: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
        })
    }

    fn vector(&self, text: &str) -> Vec<f32> {
        let winner =
            text == "choose" || text.contains(if self.quality { "quality" } else { "fast" });
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(!winner)] = 1.0;
        values
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
        if self.quality { 3 } else { 2 }
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }

    fn bound_batch_is_native(&self) -> bool {
        true
    }

    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { Ok(self.vector(text)) })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.submitted.fetch_add(texts.len(), Ordering::SeqCst);
            let mut identity = self.identity.clone();
            match self.fault.load(Ordering::SeqCst) {
                1 => {
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id().to_owned(),
                        source: "injected completion inference failure".into(),
                    });
                }
                2 => cx.set_cancel_requested(true),
                3 => identity.producer.implementation_revision.push_str("-drift"),
                _ => {}
            }
            Ok(texts
                .iter()
                .map(|text| IdentityBoundEmbedding {
                    values: self.vector(text),
                    identity: identity.clone(),
                })
                .collect())
        })
    }
}

fn source() -> Vec<IndexableDocument> {
    [
        ("a", "fast winner café"),
        ("b", "ordinary document"),
        ("c", "another document"),
        ("d", "quality winner 東京"),
        ("e", "last document"),
    ]
    .map(|(id, content)| {
        let mut document = IndexableDocument::new(id, content).with_metadata("source", "original");
        document.title = Some(format!("title {id}"));
        document
    })
    .into()
}

fn options(root: &Path, sharded: bool) -> Options {
    let mut args = vec![
        "complete-quality".to_owned(),
        "--receipt".to_owned(),
        root.join("old.json").display().to_string(),
        "--index-dir".to_owned(),
        root.join("completed").display().to_string(),
        "--new-receipt".to_owned(),
        root.join("completed.json").display().to_string(),
        "--batch-size".to_owned(),
        "2".to_owned(),
    ];
    if sharded {
        args.extend(["--shard-size".to_owned(), "3".to_owned()]);
    }
    Options::parse(args).unwrap().unwrap()
}

fn image(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                files.insert(
                    entry.path().strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(entry.path()).unwrap(),
                );
            }
        }
    }
    files
}

fn fast_values(index: &sharded::Opened) -> BTreeMap<String, Vec<u32>> {
    let partitions: Vec<_> = match index {
        sharded::Opened::Single(index) => vec![index.vectors()],
        sharded::Opened::Sharded(index) => index.vectors().partitions().iter().collect(),
    };
    let mut values = BTreeMap::new();
    for partition in partitions {
        let tier = partition.fast();
        let owner = ValidatedFsviBytes::from_arc(
            fs::read(tier.vector_path()).unwrap().into(),
            tier.binding(),
        )
        .unwrap();
        for row in 0..owner.record_count() {
            values.insert(
                owner.doc_id_at(row).unwrap().to_owned(),
                owner
                    .vector_at_f32(row)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect(),
            );
        }
    }
    values
}

fn source_values(index: &sharded::Opened) -> BTreeMap<String, serde_json::Value> {
    match index {
        sharded::Opened::Single(index) => index
            .vectors()
            .documents()
            .iter()
            .map(|document| (document.id.clone(), serde_json::to_value(document).unwrap()))
            .collect(),
        sharded::Opened::Sharded(index) => index
            .vectors()
            .documents()
            .map(|document| (document.id.clone(), serde_json::to_value(document).unwrap()))
            .collect(),
    }
}

async fn fixture(
    cx: &Cx,
    root: &Path,
    partitioned: bool,
    graphs: bool,
    documents: Vec<IndexableDocument>,
) -> (sharded::Opened, Arc<Provider>) {
    let fast = Provider::new(false);
    let retrieval = if graphs {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams {
                m: 2,
                ef_construction: 4,
                ..HnswParams::default()
            },
            seed: 197,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    let builder = NativeIndexBuilder::new(
        root.join("old"),
        ArtifactGenerationIdentityV1::new(1, [0x71; 16]).unwrap(),
        fast.clone(),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F16, retrieval)
    .add_documents(documents);
    let opened = if partitioned {
        sharded::Opened::Sharded(Box::new(builder.build_sharded_hybrid(cx, 2).await.unwrap()))
    } else {
        sharded::Opened::Single(Box::new(builder.build_hybrid(cx).await.unwrap()))
    };
    let selected = update::seal_selection(cx, opened.borrow()).unwrap();
    save_selection(&selected, &root.join("old.json")).unwrap();
    fast.submitted.store(0, Ordering::SeqCst);
    // Any accidental attempt to recompute a fast document is now fatal.
    fast.fault.store(1, Ordering::SeqCst);
    (opened, fast)
}

#[test]
fn parser_requires_explicit_completion_and_preserves_its_source() {
    let base = [
        "complete-quality",
        "--receipt",
        "old.json",
        "--index-dir",
        "new",
        "--new-receipt",
        "new.json",
    ];
    let parsed = Options::parse(base.map(str::to_owned)).unwrap().unwrap();
    assert_eq!(parsed.command, Command::CompleteQuality);
    assert!(!parsed.exact);
    for extra in [
        vec!["--input", "replacement.jsonl"],
        vec!["--fast-only"],
        vec!["--exact"],
        vec!["--mode", "quality"],
        vec!["--allow-model-migration"],
    ] {
        assert!(Options::parse(base.into_iter().chain(extra).map(str::to_owned)).is_err());
    }
    assert!(
        Options::parse(["complete-quality", "--receipt", "old.json"].map(str::to_owned)).is_err()
    );
    let parsed = Options::parse(
        base.into_iter()
            .chain(["--quality-exact"])
            .map(str::to_owned),
    )
    .unwrap()
    .unwrap();
    assert!(parsed.exact);
    assert!(
        Options::parse(
            [
                "index",
                "--receipt",
                "new.json",
                "--index-dir",
                "new",
                "--quality-exact",
            ]
            .map(str::to_owned)
        )
        .is_err()
    );
}

#[test]
fn completion_reuses_fast_documents_and_reopens_complete_single_and_sharded_cohorts() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            for graphs in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let (old, fast) = fixture(&cx, root.path(), partitioned, graphs, source()).await;
                let old_receipt = fs::read(root.path().join("old.json")).unwrap();
                let old_bytes = image(&root.path().join("old"));
                let expected_fast = fast_values(&old);
                let expected_source = source_values(&old);
                let quality = Provider::new(true);
                let models = Models {
                    fast: fast.clone(),
                    quality: Some(quality.clone()),
                };
                let mut options = options(root.path(), partitioned);
                // Deliberately choose the opposite graph policy for the new tier.
                options.exact = graphs;
                let mut output = Vec::new();
                execute_with_loader(&cx, &options, &mut output, |_| Ok(models.clone()))
                    .await
                    .unwrap();
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 0);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), source().len());
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["event"], "quality_completed");
                assert_eq!(report["fast_reused_documents"], source().len());
                drop(old);
                let selection = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
                assert_eq!(selection.generation.sequence, 2);
                let reopened = sharded::Opened::open(&cx, &selection, models)
                    .await
                    .unwrap();
                assert!(reopened.borrow().has_quality());
                assert_eq!(reopened.borrow().all_native_hnsw(false), graphs);
                assert_eq!(reopened.borrow().all_native_hnsw(true), !graphs);
                assert_eq!(fast_values(&reopened), expected_fast);
                assert_eq!(source_values(&reopened), expected_source);
                if let sharded::Opened::Sharded(index) = &reopened {
                    assert_eq!(index.vectors().partitions().len(), 2);
                }
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 0);
                assert_eq!(quality.submitted.load(Ordering::SeqCst), source().len());
                let query =
                    cohort::Prepared::prepare(reopened.borrow(), &cx, [None, None]).unwrap();
                let result = query.search(&cx, "choose", Mode::Quality, 1).await.unwrap();
                assert_eq!(result["phase"], "quality");
                assert_eq!(result["results"][0]["doc_id"], "d");
                let result = query.search(&cx, "choose", Mode::Full, 5).await.unwrap();
                assert_eq!(result["phase"], "refined");
                assert_eq!(fs::read(root.path().join("old.json")).unwrap(), old_receipt);
                assert_eq!(image(&root.path().join("old")), old_bytes);
            }
        }
    });
}

#[test]
fn completion_refuses_failed_cancelled_or_drifted_quality_without_selecting_a_partial_tier() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            for fault in [1, 2, 3] {
                let root = tempfile::tempdir().unwrap();
                let (_old, fast) = fixture(&cx, root.path(), partitioned, false, source()).await;
                let original = image(&root.path().join("old"));
                let old_receipt = fs::read(root.path().join("old.json")).unwrap();
                let quality = Provider::new(true);
                quality.fault.store(fault, Ordering::SeqCst);
                let options = options(root.path(), partitioned);
                let mut output = Vec::new();
                assert!(
                    execute_with_loader(&cx, &options, &mut output, |_| {
                        Ok(Models {
                            fast: fast.clone(),
                            quality: Some(quality),
                        })
                    })
                    .await
                    .is_err()
                );
                cx.set_cancel_requested(false);
                assert_eq!(fast.submitted.load(Ordering::SeqCst), 0);
                assert!(!options.new_receipt.as_ref().unwrap().exists());
                assert!(output.is_empty());
                assert_eq!(image(&root.path().join("old")), original);
                assert_eq!(fs::read(root.path().join("old.json")).unwrap(), old_receipt);
            }
        }
    });
}

#[test]
fn completion_refuses_full_receipts_and_wrong_layout_policy_before_loading_models() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (_old, fast) = fixture(&cx, root.path(), false, false, source()).await;
        let calls = Cell::new(0);
        let mut options = options(root.path(), false);
        options.shard_size = Some(2);
        assert!(
            execute_with_loader(&cx, &options, &mut Vec::new(), |_| {
                calls.set(calls.get() + 1);
                Ok(Models {
                    fast: fast.clone(),
                    quality: Some(Provider::new(true)),
                })
            })
            .await
            .is_err()
        );
        assert_eq!(calls.get(), 0);
        options.shard_size = None;
        let mut selection = Selection::read(&options.receipt).unwrap();
        selection.quality_producer = Some(Provider::new(true).identity.fingerprint());
        options.receipt = root.path().join("already-full.json");
        save_selection(&selection, &options.receipt).unwrap();
        assert!(
            execute_with_loader(&cx, &options, &mut Vec::new(), |_| {
                calls.set(calls.get() + 1);
                Ok(Models {
                    fast,
                    quality: Some(Provider::new(true)),
                })
            })
            .await
            .is_err()
        );
        assert_eq!(calls.get(), 0);
        assert!(!options.directory.as_ref().unwrap().exists());
        assert!(!options.new_receipt.as_ref().unwrap().exists());
    });
}

#[test]
fn completion_preserves_an_explicit_empty_source_cohort() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let (_old, fast) = fixture(&cx, root.path(), partitioned, false, Vec::new()).await;
            let quality = Provider::new(true);
            let models = Models {
                fast: fast.clone(),
                quality: Some(quality.clone()),
            };
            let mut options = options(root.path(), partitioned);
            options.exact = true;
            execute_with_loader(&cx, &options, &mut Vec::new(), |_| Ok(models.clone()))
                .await
                .unwrap();
            let selection = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
            let reopened = sharded::Opened::open(&cx, &selection, models)
                .await
                .unwrap();
            assert_eq!(reopened.borrow().document_count(), 0);
            assert!(reopened.borrow().has_quality());
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 0);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 0);
        }
    });
}
