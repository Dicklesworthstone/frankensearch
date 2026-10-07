//! Real command dispatch, complete shard writers and strict receipt reopen.
//! Counted deterministic producers are not model-quality or throughput evidence.
use super::*;
use crate::{Arc, HnswParams, NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder};
use asupersync::test_utils::run_test_with_cx;
use frankensearch::{Embedder, ModelCategory, SearchError, SearchFuture, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::ValidatedFsviBytes;
use std::cell::Cell;
use std::fmt::Write as _;
use std::future::Future;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

pub struct Provider {
    identity: EmbeddingIdentityBundleV1,
    pub(crate) submitted: AtomicUsize,
    pub(crate) fault: AtomicUsize,
    pub(crate) drops: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            submitted: AtomicUsize::new(0),
            fault: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        })
    }

    fn vector(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0; self.dimension()];
        vector[usize::from(text.contains("obsolete"))] = 0.75;
        vector
    }
}

struct DropCount<'a>(&'a AtomicUsize);
impl Drop for DropCount<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { Ok(self.vector(text)) })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let _drop = DropCount(&self.drops);
            self.submitted.fetch_add(texts.len(), Ordering::SeqCst);
            match self.fault.load(Ordering::SeqCst) {
                1 => {
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id().to_owned(),
                        source: "required update tier failed".into(),
                    });
                }
                2 => return std::future::pending().await,
                3 => cx.set_cancel_requested(true),
                _ => {}
            }
            Ok(texts
                .iter()
                .map(|text| IdentityBoundEmbedding {
                    values: self.vector(text),
                    identity: self.identity.clone(),
                })
                .collect())
        })
    }

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
}

fn source() -> Vec<IndexableDocument> {
    [
        ("a", "common stable-a"),
        ("b", "common stable-b"),
        ("c", "obsolete common"),
        ("d", "common stable-d"),
        ("z", "common stable-z"),
    ]
    .map(|(id, body)| IndexableDocument::new(id, body).with_metadata("version", "old"))
    .into()
}

pub struct Fixture {
    pub(crate) root: tempfile::TempDir,
    pub(crate) index: NativeBuiltShardedHybridIndex,
    pub(crate) selection: Selection,
    pub(crate) fast: Arc<Provider>,
    pub(crate) quality: Option<Arc<Provider>>,
}

impl Fixture {
    pub(crate) fn models(&self) -> Models {
        Models {
            fast: self.fast.clone(),
            quality: self
                .quality
                .as_ref()
                .map(|model| model.clone() as Arc<dyn Embedder>),
        }
    }

    fn counts(&self) -> (usize, Option<usize>) {
        (
            self.fast.submitted.load(Ordering::SeqCst),
            self.quality
                .as_ref()
                .map(|q| q.submitted.load(Ordering::SeqCst)),
        )
    }

    fn options(&self, changes: &str, size: usize) -> Options {
        let edits = self.root.path().join("changes.jsonl");
        fs::write(&edits, changes).unwrap();
        Options::parse(vec![
            "update".to_owned(),
            "--receipt".to_owned(),
            self.root.path().join("old.json").display().to_string(),
            "--index-dir".to_owned(),
            self.root.path().join("next").display().to_string(),
            "--new-receipt".to_owned(),
            self.root.path().join("next.json").display().to_string(),
            "--input".to_owned(),
            edits.display().to_string(),
            "--shard-size".to_owned(),
            size.to_string(),
        ])
        .unwrap()
        .unwrap()
    }
}

pub async fn fixture(cx: &Cx, graphs: bool, quality: bool) -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let fast = Provider::new("update-cli-fast", 2);
    let quality = quality.then(|| Provider::new("update-cli-quality", 3));
    let retrieval = if graphs {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 29,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    let mut builder = NativeIndexBuilder::new(
        root.path().join("old"),
        crate::ArtifactGenerationIdentityV1::new(1, [17; 16]).unwrap(),
        fast.clone(),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F16, retrieval)
    .add_documents(source());
    if let Some(quality) = &quality {
        builder = builder
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, retrieval)
            .unwrap();
    }
    let index = builder.build_sharded_hybrid(cx, 2).await.unwrap();
    let selection = seal_selection(cx, (&index).into()).unwrap();
    save_selection(&selection, &root.path().join("old.json")).unwrap();
    Fixture {
        root,
        index,
        selection,
        fast,
        quality,
    }
}

#[test]
fn sharded_update_command_repartitions_reuses_and_reopens_the_complete_successor() {
    run_test_with_cx(|cx| async move {
        for graphs in [false, true] {
            for has_quality in [false, true] {
                let f = fixture(&cx, graphs, has_quality).await;
                let old_receipt = fs::read(f.root.path().join("old.json")).unwrap();
                let options = f.options(concat!(
                    "{\"op\":\"upsert\",\"id\":\"a\",\"content\":\"common stable-a\",\"title\":\"new title\",\"metadata\":{\"version\":\"new\"}}\n",
                    "{\"op\":\"delete\",\"id\":\"c\"}\n",
                    "{\"op\":\"upsert\",\"id\":\"aa\",\"content\":\"arrival common\"}\n",
                    "{\"op\":\"upsert\",\"id\":\"b\",\"content\":\"discarded edit\"}\n",
                    "{\"op\":\"upsert\",\"id\":\"b\",\"content\":\"replacement common\"}\n",
                ), 3);
                let mut output = Vec::new();
                execute_with_loader(&cx, &options, &mut output, |_, required| {
                    assert_eq!(required, has_quality);
                    Ok(f.models())
                })
                .await
                .unwrap();
                let frame: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(frame["event"], "updated");
                assert_eq!(frame["edited_ids"], 4);
                assert_eq!(f.counts(), (7, has_quality.then_some(7)));
                let selection = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
                assert_eq!(selection.schema, sharded::SELECTION_SCHEMA);
                assert_eq!(selection.generation.sequence, 2);
                assert_eq!(
                    frame["selection"],
                    serde_json::to_value(&selection).unwrap()
                );
                let next = sharded::open(&cx, &selection, f.models()).await.unwrap();
                assert_eq!(next.vectors().partitions().len(), 2);
                assert_eq!(
                    next.vectors()
                        .documents()
                        .map(|doc| doc.id.as_str())
                        .collect::<Vec<_>>(),
                    ["a", "aa", "b", "d", "z"]
                );
                assert_eq!(
                    next.vectors().document("a").unwrap().title.as_deref(),
                    Some("new title")
                );
                assert_eq!(
                    next.vectors().document("a").unwrap().metadata["version"],
                    "new"
                );
                assert_eq!(
                    next.vectors().document("b").unwrap().content,
                    "replacement common"
                );
                assert!(
                    next.lexical()
                        .search(&cx, "obsolete", 10)
                        .await
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(
                    next.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id,
                    "aa"
                );
                assert_eq!(
                    f.index.lexical().search(&cx, "obsolete", 10).await.unwrap()[0].doc_id,
                    "c"
                );
                for partition in next.vectors().partitions() {
                    for tier in std::iter::once(partition.fast()).chain(partition.quality()) {
                        assert_eq!(tier.graph_path().is_some(), graphs);
                        let owner = ValidatedFsviBytes::from_arc(
                            fs::read(tier.vector_path()).unwrap().into(),
                            tier.binding(),
                        )
                        .unwrap();
                        for row in 0..owner.record_count() {
                            assert!(
                                next.vectors()
                                    .document(owner.doc_id_at(row).unwrap())
                                    .is_some()
                            );
                        }
                    }
                }
                assert_eq!(
                    f.counts(),
                    (7, has_quality.then_some(7)),
                    "reopen performs no inference"
                );
                assert_eq!(
                    fs::read(f.root.path().join("old.json")).unwrap(),
                    old_receipt
                );
            }
        }
    });
}

#[test]
fn delete_all_keeps_an_empty_required_shard_and_can_be_updated_after_restart() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, true).await;
        f.fast.fault.store(1, Ordering::SeqCst);
        f.quality.as_ref().unwrap().fault.store(1, Ordering::SeqCst);
        let deletes = source().into_iter().fold(String::new(), |mut lines, doc| {
            writeln!(lines, "{{\"op\":\"delete\",\"id\":\"{}\"}}", doc.id).unwrap();
            lines
        });
        let options = f.options(&deletes, 7);
        execute_with_loader(&cx, &options, &mut Vec::new(), |_, _| Ok(f.models()))
            .await
            .unwrap();
        assert_eq!(f.counts(), (5, Some(5)));
        let selected = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
        let empty = sharded::open(&cx, &selected, f.models()).await.unwrap();
        assert_eq!(empty.vectors().partitions().len(), 1);
        assert_eq!(empty.vectors().document_count(), 0);
        assert!(empty.vectors().quality().is_some());
        assert!(
            empty
                .search_refined(&cx, "common", 10)
                .await
                .unwrap()
                .is_empty()
        );
        f.fast.fault.store(0, Ordering::SeqCst);
        f.quality.as_ref().unwrap().fault.store(0, Ordering::SeqCst);
        let edits = read_edits(&mut Cursor::new(
            "{\"op\":\"upsert\",\"id\":\"fresh\",\"content\":\"fresh arrival\"}\n",
        ))
        .unwrap();
        let (next, receipt) = apply_sharded(
            &cx,
            &selected,
            &empty,
            &f.root.path().join("third"),
            edits,
            4,
            2,
        )
        .await
        .unwrap();
        assert_eq!(receipt.generation.sequence, 3);
        assert_eq!(next.vectors().document_count(), 1);
        assert_eq!(f.counts(), (6, Some(6)));
        assert_eq!(f.index.vectors().document_count(), 5);
    });
}

#[test]
fn policy_invalid_input_and_cancelled_entry_do_not_load_models_or_create_a_candidate() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        let mut options = f.options("{\"op\":\"delete\",\"id\":\"a\"}\n", 2);
        options.shard_size = None;
        let loaded = Cell::new(false);
        let mut output = Vec::new();
        let result = execute_with_loader(&cx, &options, &mut output, |_, _| {
            loaded.set(true);
            Ok(f.models())
        })
        .await;
        assert!(result.is_err());
        assert!(!loaded.get());
        options.shard_size = Some(2);
        fs::write(options.input.as_ref().unwrap(), "{\"op\":\"upsert\",\"id\":\"\",\"content\":\"bad\"}\n{\"op\":\"delete\",\"id\":\"\"}\n").unwrap();
        assert!(
            execute_with_loader(&cx, &options, &mut output, |_, _| {
                loaded.set(true);
                Ok(f.models())
            })
            .await
            .is_err()
        );
        assert!(!loaded.get(), "overwritten invalid operations still fail");
        cx.set_cancel_requested(true);
        let error = execute_with_loader(&cx, &options, &mut output, |_, _| {
            loaded.set(true);
            Ok(f.models())
        })
        .await
        .err()
        .unwrap();
        cx.set_cancel_requested(false);
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::Cancelled { .. })
        ));
        assert!(!loaded.get());
        assert!(output.is_empty());
        assert!(!options.directory.as_ref().unwrap().exists());
        assert!(!options.new_receipt.as_ref().unwrap().exists());
        assert!(validate_partition_policy(false, Some(2)).is_err());
        assert!(validate_partition_policy(true, Some(0)).is_err());
    });
}

#[test]
fn final_partition_limit_includes_unchanged_sources_outside_the_delta() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        let changes = (0..1020).fold(String::new(), |mut lines, i| {
            writeln!(
                lines,
                "{{\"op\":\"upsert\",\"id\":\"new-{i}\",\"content\":\"common\"}}"
            )
            .unwrap();
            lines
        });
        let options = f.options(&changes, 1);
        let mut output = Vec::new();
        assert!(
            execute_with_loader(&cx, &options, &mut output, |_, _| Ok(f.models()))
                .await
                .is_err()
        );
        assert_eq!(f.counts(), (5, Some(5)));
        assert!(!options.directory.as_ref().unwrap().exists());
        assert!(!options.new_receipt.as_ref().unwrap().exists());
        assert!(output.is_empty());
    });
}

#[test]
fn missing_selected_partition_and_foreign_producer_cannot_be_updated_as_a_partial_index() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, true).await;
        let options = f.options(
            "{\"op\":\"upsert\",\"id\":\"new\",\"content\":\"common new\"}\n",
            3,
        );
        let mut output = Vec::new();
        assert!(
            execute_with_loader(&cx, &options, &mut output, |_, _| {
                Ok(Models {
                    fast: Provider::new("foreign", 2),
                    quality: f.models().quality,
                })
            })
            .await
            .is_err()
        );
        let last = f
            .index
            .vectors()
            .partitions()
            .last()
            .unwrap()
            .quality()
            .unwrap()
            .vector_path();
        fs::rename(last, f.root.path().join("retained-missing-quality")).unwrap();
        assert!(
            execute_with_loader(&cx, &options, &mut output, |_, _| Ok(f.models()))
                .await
                .is_err()
        );
        assert!(!options.directory.as_ref().unwrap().exists());
        assert!(!options.new_receipt.as_ref().unwrap().exists());
        assert!(output.is_empty());
        assert_eq!(f.counts(), (5, Some(5)));
        assert!(
            !f.index
                .search_quality(&cx, "common", 5)
                .await
                .unwrap()
                .is_empty()
        );
    });
}

#[test]
fn late_quality_failure_cancellation_and_future_drop_leave_no_complete_successor() {
    run_test_with_cx(|cx| async move {
        for fault in [1, 2, 3] {
            let f = fixture(&cx, false, true).await;
            let quality = f.quality.as_ref().unwrap();
            quality.fault.store(fault, Ordering::SeqCst);
            let edits = read_edits(&mut Cursor::new(
                "{\"op\":\"upsert\",\"id\":\"zz-new\",\"content\":\"common new\"}\n",
            ))
            .unwrap();
            let path = f.root.path().join("failed");
            let before_drops = quality.drops.load(Ordering::SeqCst);
            let mut update = Box::pin(apply_sharded(
                &cx,
                &f.selection,
                &f.index,
                &path,
                edits,
                2,
                2,
            ));
            if fault == 2 {
                assert!(
                    update
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
            } else {
                let error = update.as_mut().await.err().unwrap();
                if fault == 3 {
                    assert!(matches!(
                        error.downcast_ref::<SearchError>(),
                        Some(SearchError::Cancelled { .. })
                    ));
                }
            }
            drop(update);
            cx.set_cancel_requested(false);
            assert_eq!(quality.drops.load(Ordering::SeqCst), before_drops + 1);
            assert!(
                path.join("shard-000000/fast.fsvi").is_file(),
                "the successful prefix is real"
            );
            assert!(!path.join("native.sharded-hybrid.json").exists());
            assert!(!path.join("native.sharded.json").exists());
            assert_eq!(f.index.vectors().document_count(), 5);
            assert!(f.index.vectors().document("zz-new").is_none());
            assert!(f.index.search_refined(&cx, "common", 5).await.is_ok());
        }
    });
}

#[test]
fn failed_acknowledgement_does_not_undo_a_saved_complete_update_receipt() {
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, true).await;
        let options = f.options("{\"op\":\"delete\",\"id\":\"c\"}\n", 2);
        assert!(
            execute_with_loader(&cx, &options, &mut Broken, |_, _| Ok(f.models()))
                .await
                .is_err()
        );
        let selected = Selection::read(options.new_receipt.as_ref().unwrap()).unwrap();
        let next = sharded::open(&cx, &selected, f.models()).await.unwrap();
        assert_eq!(selected.generation.sequence, 2);
        assert_eq!(next.vectors().document_count(), 4);
        assert!(f.index.vectors().document("c").is_some());
        assert_eq!(f.counts(), (5, Some(5)));
    });
}
