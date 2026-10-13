//! Whole-cohort CLI migration through real sharded FSVI/Quill/native owners.
//! Synthetic identified providers isolate routing and lifecycle, not model quality.

use super::*;
use std::future::Future;
use std::io::{self, Cursor};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use asupersync::test_utils::run_test_with_cx;
use frankensearch::native_ann::builder::NativeIndexBuilder;
use frankensearch::native_ann::builder::sharded::NativeBuiltShardedHybridIndex;
use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
use frankensearch::{Embedder, IndexableDocument};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};

use crate::serve::{Controls, SessionControls, run_with_controls};
use crate::{filter, fs, save_selection, update};

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    queries: AtomicUsize,
    documents: AtomicUsize,
    offline: AtomicBool,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            queries: AtomicUsize::new(0),
            documents: AtomicUsize::new(0),
            offline: AtomicBool::new(false),
        })
    }

    fn output(&self, text: &str) -> IdentityBoundEmbedding {
        let mut values = vec![0.0; self.dimension()];
        values[usize::from(text.contains("vertical"))] = 1.0;
        IdentityBoundEmbedding {
            identity: self.identity.clone(),
            values,
        }
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async { panic!("migration must not use raw inference") })
    }

    fn embed_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            Ok(self.output(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.documents.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts.iter().map(|text| self.output(text)).collect())
        })
    }

    fn bound_batch_is_native(&self) -> bool {
        true
    }
    fn identity(&self) -> frankensearch::SearchResult<&EmbeddingIdentityBundleV1> {
        if self.offline.load(Ordering::SeqCst) {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: "offline-fixture".to_owned(),
                reason: "old provider is no longer available".to_owned(),
            });
        }
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
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

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [71; 16]).unwrap()
}

fn documents() -> Vec<IndexableDocument> {
    vec![
        IndexableDocument::new("a", "horizontal").with_title("first"),
        IndexableDocument::new("b", "vertical").with_metadata("scope", "keep"),
        IndexableDocument::new("c", "horizontal").with_metadata("scope", "drop"),
    ]
}

async fn sealed(
    cx: &Cx,
    path: &Path,
    sequence: u64,
    models: Models,
    sources: Vec<IndexableDocument>,
    exact: bool,
    shard_size: usize,
) -> (NativeBuiltShardedHybridIndex, Selection) {
    let retrieval = if exact {
        NativeBuildRetrieval::Exact
    } else {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 42,
        }
    };
    let mut builder = NativeIndexBuilder::new(path, generation(sequence), models.fast)
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents(sources);
    if let Some(quality) = models.quality {
        builder = builder
            .with_quality_embedder(quality)
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, retrieval)
            .unwrap();
    }
    let built = builder.build_sharded_hybrid(cx, shard_size).await.unwrap();
    let selection = update::seal_selection(cx, (&built).into()).unwrap();
    (built, selection)
}

fn request(receipt: &Path, exact: bool) -> serde_json::Value {
    serde_json::json!({
        "op": "activate_model_migration", "id": "migration",
        "expected_generation": generation(1), "receipt": receipt,
        "model_dir": "/explicit/replacement/models", "exact": exact,
    })
}

fn input(messages: &[serde_json::Value]) -> Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for message in messages {
        serde_json::to_writer(&mut bytes, message).unwrap();
        bytes.push(b'\n');
    }
    Cursor::new(bytes)
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes.split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

fn authority(models: Models, loads: Arc<AtomicUsize>) -> Authority {
    Authority {
        load: Box::new(move |_, path, quality, _| {
            assert_eq!(path, Path::new("/explicit/replacement/models"));
            assert_eq!(quality, models.quality.is_some());
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(models.clone())
        }),
    }
}

async fn response(
    live: &NativeLiveShardedHybridIndex,
    cx: &Cx,
    request: serde_json::Value,
    authority: Option<&Authority>,
) -> serde_json::Value {
    let request: Request = serde_json::from_value(request).unwrap();
    let mut output = Vec::new();
    execute(live.into(), cx, &request, 1, authority, &mut output)
        .await
        .unwrap();
    let mut frames = frames(&output);
    assert_eq!(frames.len(), 1);
    frames.pop().unwrap()
}

#[test]
fn warm_migration_repartitions_and_changes_tiers_without_reloading_per_query() {
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            for (old_quality, new_quality) in [(false, true), (true, false), (true, true)] {
                let root = tempfile::tempdir().unwrap();
                let old = Provider::new("old", 2);
                let old_q = Provider::new("old-quality", 3);
                let (built, _) = sealed(
                    &cx, &root.path().join("old"), 1,
                    Models {
                        fast: old.clone(),
                        quality: old_quality.then(|| old_q.clone() as Arc<dyn Embedder>),
                    },
                    documents(), true, 1,
                ).await;
                let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
                let pin = live.snapshot(&cx).await.unwrap();
                let old_bytes: Vec<_> = pin.index().vectors().partitions().iter()
                    .map(|part| fs::read(part.fast().vector_path()).unwrap()).collect();
                let fast = Provider::new("replacement-fast", 4);
                let quality = Provider::new("replacement-quality", 5);
                let models = Models {
                    fast: fast.clone(),
                    quality: new_quality.then(|| quality.clone() as Arc<dyn Embedder>),
                };
                let (candidate, selection) = sealed(
                    &cx, &root.path().join("next"), 2, models.clone(), documents(), exact, 2,
                ).await;
                let receipt = root.path().join("next.json");
                save_selection(&selection, &receipt).unwrap();
                drop(candidate);
                let loads = Arc::new(AtomicUsize::new(0));
                let controls = SessionControls {
                    controls: Controls::default(),
                    migration: Some(authority(models, loads.clone())),
                };
                let mut reader = input(&[
                    serde_json::json!({"id":"before", "query":"vertical"}),
                    request(&receipt, exact),
                    serde_json::json!({"id":"after", "query":"vertical"}),
                    serde_json::json!({"id":"again", "query":"vertical"}),
                    serde_json::json!({"op":"status", "id":"status"}),
                ]);
                let scope = filter::Filter::parse(r#"{"ids":["b"]}"#).unwrap();
                let mut output = Vec::new();
                run_with_controls(
                    &live, &cx, &mut reader, &mut output, (crate::Mode::Full, 3),
                    controls, Some(&scope), &query::Policy::new(&cx, None).unwrap(),
                ).await.unwrap();
                let output = frames(&output);
                assert_eq!(output[0]["model_migration_supported"], true);
                assert_eq!(output[0]["model_migration_enabled"], true);
                assert_eq!(output[0]["activation_enabled"], false);
                assert_eq!(output[0]["updates_enabled"], false);
                let ack = output.iter().find(|f| f["operation"] == "activate_model_migration").unwrap();
                assert_eq!(ack["ok"], true);
                assert_eq!(ack["layout"], "sharded");
                assert_eq!(ack["partitions"], 2);
                assert_eq!(ack["selection_changed"], true);
                assert_eq!(ack["generation"], serde_json::json!(generation(2)));
                assert_eq!(ack["fast_producer"], fast.identity.fingerprint());
                for id in ["after", "again"] {
                    let pages: Vec<_> = output.iter()
                        .filter(|f| f["id"] == id && f["event"] == "results").collect();
                    assert_eq!(pages.len(), if new_quality { 2 } else { 1 });
                    for page in pages {
                        assert_eq!(page["generation"], serde_json::json!(generation(2)));
                        assert_eq!(page["results"].as_array().unwrap().len(), 1);
                        assert_eq!(page["results"][0]["doc_id"], "b");
                        assert_eq!(page["results"][0]["metadata"]["scope"], "keep");
                        assert_eq!(page["results"][0]["fast_row"]["shard"], 0);
                        if page["phase"] == "refined" {
                            assert_eq!(page["results"][0]["quality_row"]["shard"], 0);
                        }
                    }
                }
                assert_eq!(loads.load(Ordering::SeqCst), 1);
                assert_eq!(fast.documents.load(Ordering::SeqCst), 3);
                assert_eq!(quality.documents.load(Ordering::SeqCst), if new_quality { 3 } else { 0 });
                assert_eq!(fast.queries.load(Ordering::SeqCst), 2);
                assert_eq!(quality.queries.load(Ordering::SeqCst), if new_quality { 2 } else { 0 });
                assert_eq!(old.queries.load(Ordering::SeqCst), 1);
                assert_eq!(old_q.queries.load(Ordering::SeqCst), usize::from(old_quality));
                let selected = live.snapshot(&cx).await.unwrap();
                assert_eq!(selected.index().vectors().partitions()[0].fast().embedder().dimension(), 4);
                assert_eq!(selected.index().vectors().quality().is_some(), new_quality);
                assert_eq!(pin.generation(), generation(1));
                for (part, bytes) in pin.index().vectors().partitions().iter().zip(old_bytes) {
                    assert_eq!(fs::read(part.fast().vector_path()).unwrap(), bytes);
                }
                let old_scope = pin.index().scope(&cx, |doc| Ok(doc.id == "b")).unwrap();
                let old_page = old_scope.search(&cx, "vertical", 1).await.unwrap();
                assert_eq!(old_page[0].fast_row.unwrap().shard, 1);
            }
        }
    });
}

#[test]
fn permissions_stale_generation_and_foreign_layout_refuse_before_model_loading() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(&cx, &root.path().join("old"), 1,
            Models { fast: old, quality: None }, documents(), true, 1).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
        let pin = live.snapshot(&cx).await.unwrap();
        let denied = Authority { load: Box::new(|_, _, _, _| panic!("must not load models")) };
        let missing = root.path().join("missing.json");
        let answer = response(&live, &cx, request(&missing, true), None).await;
        assert!(answer["error"].as_str().unwrap().contains("model migration is disabled"));
        let mut stale = request(&missing, true);
        stale["expected_generation"] = serde_json::json!(generation(2));
        let answer = response(&live, &cx, stale, Some(&denied)).await;
        assert!(answer["error"].as_str().unwrap().contains("expected_generation"));
        let replacement = Provider::new("new", 3);
        let single = NativeIndexBuilder::new(root.path().join("single"), generation(2), replacement)
            .unwrap().add_documents(documents()).build_hybrid(&cx).await.unwrap();
        let selection = update::seal_selection(&cx, (&single).into()).unwrap();
        let receipt = root.path().join("single.json");
        save_selection(&selection, &receipt).unwrap();
        let answer = response(&live, &cx, request(&receipt, true), Some(&denied)).await;
        assert!(answer["error"].as_str().unwrap().contains("same-layout"));
        assert_eq!(answer["selection_changed"], false);
        assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), pin.index()));
    });
}

#[test]
fn valid_receipts_with_foreign_source_fields_cannot_replace_a_sharded_cohort() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (built, _) = sealed(&cx, &root.path().join("old"), 1,
            Models { fast: Provider::new("old", 2), quality: None }, documents(), true, 1).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
        let pin = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("new", 3);
        let models = Models { fast: replacement.clone(), quality: None };
        let load = authority(models.clone(), Arc::new(AtomicUsize::new(0)));
        for variant in 0..4 {
            let mut sources = documents();
            match variant {
                0 => sources[2].content = "different content".to_owned(),
                1 => sources[2].title = Some("different title".to_owned()),
                2 => { sources[2].metadata.insert("scope".to_owned(), "different".to_owned()); }
                _ => sources[2].id = "different-id".to_owned(),
            }
            let (candidate, selection) = sealed(&cx, &root.path().join(format!("next-{variant}")),
                2, models.clone(), sources, true, 2).await;
            let receipt = root.path().join(format!("next-{variant}.json"));
            save_selection(&selection, &receipt).unwrap();
            drop(candidate);
            let answer = response(&live, &cx, request(&receipt, true), Some(&load)).await;
            assert_eq!(answer["ok"], false);
            assert_eq!(answer["selection_changed"], false);
            assert!(answer["error"].as_str().unwrap().contains("sharded_live.migration.source"));
            assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), pin.index()));
        }
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 12);
        assert_eq!(replacement.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn damaged_final_partition_and_wrong_policy_never_install_a_partial_inventory() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (built, _) = sealed(&cx, &root.path().join("old"), 1,
            Models { fast: Provider::new("old", 2), quality: None }, documents(), true, 1).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
        let pin = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("new", 3);
        let models = Models { fast: replacement.clone(), quality: None };
        let load = authority(models.clone(), Arc::new(AtomicUsize::new(0)));
        let (candidate, selection) = sealed(&cx, &root.path().join("next"), 2,
            models, documents(), false, 2).await;
        let last = candidate.vectors().partitions().last().unwrap().fast();
        let vector = last.vector_path().to_path_buf();
        let graph = last.graph_path().unwrap().to_path_buf();
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let answer = response(&live, &cx, request(&receipt, true), Some(&load)).await;
        assert_eq!(answer["ok"], false, "exact policy must not accept an HNSW target");
        for path in [vector, graph] {
            let original = fs::read(&path).unwrap();
            let mut corrupt = original.clone();
            *corrupt.last_mut().unwrap() ^= 1;
            fs::write(&path, corrupt).unwrap();
            let answer = response(&live, &cx, request(&receipt, false), Some(&load)).await;
            assert_eq!(answer["ok"], false);
            assert_eq!(answer["selection_changed"], false);
            assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), pin.index()));
            fs::write(&path, original).unwrap();
        }
        let answer = response(&live, &cx, request(&receipt, false), Some(&load)).await;
        assert_eq!(answer["ok"], true, "only the exact restored complete target is accepted");
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 3);
        assert_eq!(replacement.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn concurrent_source_update_beats_a_higher_sequence_model_migration() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (built, _) = sealed(&cx, &root.path().join("old"), 1,
            Models { fast: Provider::new("old", 2), quality: None }, documents(), true, 1).await;
        let live = Arc::new(NativeLiveShardedHybridIndex::new(&cx, built).unwrap());
        let pin = live.snapshot(&cx).await.unwrap();
        let winner = Arc::new(pin.begin_update(&cx, root.path().join("winner"), generation(2))
            .unwrap().upsert_document(IndexableDocument::new("winner", "fresh source"))
            .build(&cx, 2).await.unwrap());
        let models = Models { fast: Provider::new("new", 3), quality: None };
        let (candidate, selection) = sealed(&cx, &root.path().join("migration"), 9,
            models.clone(), documents(), true, 2).await;
        let receipt = root.path().join("migration.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let competing = live.clone();
        let load = Authority {
            load: Box::new(move |cx, _, _, _| {
                let mut install = Box::pin(competing.install(cx, &winner));
                let Poll::Ready(Ok(installed)) = install.as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())) else {
                    panic!("uncontended real install must finish");
                };
                assert_eq!(installed.generation(), generation(2));
                Ok(models.clone())
            }),
        };
        let answer = response(&live, &cx, request(&receipt, true), Some(&load)).await;
        assert_eq!(answer["ok"], false);
        assert_eq!(answer["selection_changed"], false);
        assert!(answer["error"].as_str().unwrap().contains("sharded_live.expected_current"));
        let selected = live.snapshot(&cx).await.unwrap();
        assert_eq!(selected.generation(), generation(2));
        assert!(selected.index().vectors().documents().any(|doc| doc.id == "winner"));
        assert!(!pin.index().vectors().documents().any(|doc| doc.id == "winner"));
    });
}

#[test]
fn cancelled_loading_preserves_selection_and_offline_old_models_do_not_block_migration() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(&cx, &root.path().join("old"), 1,
            Models { fast: old.clone(), quality: None }, documents(), true, 1).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
        let pin = live.snapshot(&cx).await.unwrap();
        let models = Models { fast: Provider::new("new", 3), quality: None };
        let (candidate, selection) = sealed(&cx, &root.path().join("next"), 2,
            models.clone(), documents(), true, 2).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        for success in [false, true] {
            let models = models.clone();
            let load = Authority { load: Box::new(move |cx, _, _, _| {
                cx.set_cancel_requested(true);
                if success { Ok(models.clone()) } else { Err(bad("loader failed")) }
            }) };
            let answer = response(&live, &cx, request(&receipt, true), Some(&load)).await;
            cx.set_cancel_requested(false);
            assert_eq!(answer["status"], "cancelled");
            assert_eq!(answer["selection_changed"], false);
            assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), pin.index()));
        }
        old.offline.store(true, Ordering::SeqCst);
        let load = authority(models, Arc::new(AtomicUsize::new(0)));
        let answer = response(&live, &cx, request(&receipt, true), Some(&load)).await;
        assert_eq!(answer["ok"], true);
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), generation(2));
        assert_eq!(old.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn lost_ack_does_not_rollback_a_sharded_install_or_consume_another_request() {
    struct BrokenAck(usize);
    impl Write for BrokenAck {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 += 1;
            if self.0 > 1 { return Err(io::Error::other("lost migration ack")); }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (built, _) = sealed(&cx, &root.path().join("old"), 1,
            Models { fast: Provider::new("old", 2), quality: None }, documents(), true, 1).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
        let model = Provider::new("new", 3);
        let models = Models { fast: model.clone(), quality: None };
        let (candidate, selection) = sealed(&cx, &root.path().join("next"), 2,
            models.clone(), documents(), true, 2).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let mut reader = input(&[request(&receipt, true), serde_json::json!({"query":"vertical"})]);
        let controls = SessionControls {
            controls: Controls::default(),
            migration: Some(authority(models, Arc::new(AtomicUsize::new(0)))),
        };
        let mut writer = BrokenAck(0);
        let error = run_with_controls(&live, &cx, &mut reader, &mut writer,
            (crate::Mode::Full, 3), controls, None, &query::Policy::new(&cx, None).unwrap())
            .await.unwrap_err();
        assert!(error.to_string().contains("lost migration ack"));
        assert_eq!(writer.0, 2);
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), generation(2));
        assert!(reader.position() < u64::try_from(reader.get_ref().len()).unwrap());
        assert_eq!(model.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn empty_migration_retains_one_identity_bearing_partition_without_inference() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("empty-old", 2);
        let (built, _) = sealed(
            &cx,
            &root.path().join("old"),
            1,
            Models { fast: old, quality: None },
            Vec::new(),
            true,
            1,
        ).await;
        let live = NativeLiveShardedHybridIndex::new(&cx, built).unwrap();
        let fast = Provider::new("empty-new", 3);
        let quality = Provider::new("empty-quality", 4);
        let models = Models { fast: fast.clone(), quality: Some(quality.clone()) };
        let (candidate, selection) = sealed(
            &cx,
            &root.path().join("next"),
            2,
            models.clone(),
            Vec::new(),
            false,
            2,
        ).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let load = authority(models, Arc::new(AtomicUsize::new(0)));
        let answer = response(&live, &cx, request(&receipt, false), Some(&load)).await;
        assert_eq!(answer["ok"], true);
        assert_eq!(answer["documents"], 0);
        assert_eq!(answer["partitions"], 1);
        assert_eq!(answer["quality"], true);
        let selected = live.snapshot(&cx).await.unwrap();
        assert_eq!(selected.index().vectors().partitions()[0].fast().embedder().dimension(), 3);
        assert_eq!(fast.documents.load(Ordering::SeqCst), 0);
        assert_eq!(quality.documents.load(Ordering::SeqCst), 0);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
    });
}
