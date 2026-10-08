use super::*;
use std::io::{self, Cursor};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use asupersync::test_utils::run_test_with_cx;
use frankensearch::native_ann::builder::NativeIndexBuilder;
use frankensearch::native_ann::builder::live::NativeLiveHybridIndex;
use frankensearch::{Embedder, IndexableDocument};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};

use crate::serve::{Controls, SessionControls, run_with_controls};
use crate::{Options, cohort, save_selection, update};

// Synthetic control identities exercise actual persisted v2/Quill admission.
// No real-model quality, performance or artifact verification is inferred.
struct Provider {
    identity: EmbeddingIdentityBundleV1,
    queries: AtomicUsize,
    documents: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            queries: AtomicUsize::new(0),
            documents: AtomicUsize::new(0),
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
        Box::pin(async { panic!("CLI model migration must never use raw inference") })
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
    ArtifactGenerationIdentityV1::new(sequence, [0x56; 16]).unwrap()
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
) -> (crate::NativeBuiltHybridIndex, Selection) {
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
    let built = builder.build_hybrid(cx).await.unwrap();
    let selection = update::seal_selection(cx, cohort::Index::from(&built)).unwrap();
    (built, selection)
}

fn request(receipt: &Path, exact: bool) -> serde_json::Value {
    serde_json::json!({
        "op": "activate_model_migration", "id": "migration",
        "expected_generation": generation(1), "receipt": receipt,
        "model_dir": "/explicit/replacement/models", "exact": exact,
    })
}

fn frames(output: &[u8]) -> Vec<serde_json::Value> {
    output
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

fn input(messages: &[serde_json::Value]) -> Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for message in messages {
        serde_json::to_writer(&mut bytes, message).unwrap();
        bytes.push(b'\n');
    }
    Cursor::new(bytes)
}

fn permitted(models: Models, loads: Arc<AtomicUsize>) -> SessionControls {
    SessionControls {
        controls: Controls::default(),
        migration: Some(Authority {
            load: Box::new(move |_cx, path, required, options| {
                assert_eq!(path, Path::new("/explicit/replacement/models"));
                assert_eq!(required, models.quality.is_some());
                assert!(options.backend.is_none());
                loads.fetch_add(1, Ordering::SeqCst);
                Ok(models.clone())
            }),
        }),
    }
}

#[test]
fn migration_grant_is_separate_serve_only_and_refuses_lexical_mode() {
    let parse = |args: &[&str]| Options::parse(args.iter().map(|arg| (*arg).to_owned()));
    let options = parse(&["serve", "--receipt", "saved.json", "--allow-model-migration"])
        .unwrap()
        .unwrap();
    assert!(options.allow_model_migration);
    assert!(!options.allow_updates);
    assert_eq!(options.activation, crate::serve::ActivationPermission::Disabled);
    let ordinary = parse(&["serve", "--receipt", "saved.json", "--allow-activation", "--allow-updates"])
        .unwrap()
        .unwrap();
    assert!(!ordinary.allow_model_migration);
    for args in [
        vec!["serve", "--receipt", "saved.json", "--lexical-only", "--allow-model-migration"],
        vec!["search", "--receipt", "saved.json", "--query", "q", "--allow-model-migration"],
        vec!["serve", "--receipt", "saved.json", "--allow-model-migration", "--allow-model-migration"],
    ] {
        assert!(parse(&args).is_err());
    }
}

#[test]
fn unknown_or_ambiguous_migration_fields_do_not_become_searches() {
    let valid = request(Path::new("/candidate.json"), true);
    assert!(matches!(
        serde_json::from_value::<crate::serve::Message>(valid.clone()).unwrap(),
        crate::serve::Message::ModelMigration(_)
    ));
    for missing in ["expected_generation", "receipt", "model_dir", "exact"] {
        let mut value = valid.clone();
        value.as_object_mut().unwrap().remove(missing);
        assert!(serde_json::from_value::<crate::serve::Message>(value).is_err());
    }
    for field in ["query", "timeout_ms", "allow_model_migration", "shard_size", "unknown"] {
        let mut value = valid.clone();
        value[field] = serde_json::json!("unexpected");
        assert!(serde_json::from_value::<crate::serve::Message>(value).is_err());
    }
}

#[test]
fn warm_loop_loads_replacement_once_and_uses_new_models_after_atomic_install() {
    run_test_with_cx(|cx| async move {
        for exact in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let old = Provider::new("old", 2);
            let (built, _) = sealed(
                &cx, &root.path().join("old"), 1,
                Models { fast: old.clone(), quality: None }, documents(), true,
            ).await;
            let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
            let pinned = live.snapshot(&cx).await.unwrap();
            let old_bytes = crate::fs::read(pinned.index().vectors().fast().vector_path()).unwrap();
            let fast = Provider::new("new-fast", 3);
            let quality = Provider::new("new-quality", 4);
            let models = Models { fast: fast.clone(), quality: Some(quality.clone()) };
            let (candidate, selection) = sealed(
                &cx, &root.path().join("next"), 2, models.clone(), documents(), exact,
            ).await;
            let receipt = root.path().join("next.json");
            save_selection(&selection, &receipt).unwrap();
            drop(candidate);
            let loads = Arc::new(AtomicUsize::new(0));
            let mut reader = input(&[
                serde_json::json!({"id": "before", "query": "vertical"}),
                request(&receipt, exact),
                serde_json::json!({"id": "after", "query": "vertical"}),
                serde_json::json!({"id": "again", "query": "horizontal"}),
                serde_json::json!({"op": "status", "id": "status"}),
            ]);
            let mut output = Vec::new();
            run_with_controls(
                &live, &cx, &mut reader, &mut output, (crate::Mode::Full, 3),
                permitted(models, loads.clone()), None, &query::Policy::new(&cx, None).unwrap(),
            ).await.unwrap();
            let output = frames(&output);
            assert_eq!(output[0]["model_migration_enabled"], true);
            assert_eq!(output[0]["activation_enabled"], false);
            assert_eq!(output[0]["updates_enabled"], false);
            let migration = output.iter().find(|frame| frame["operation"] == "activate_model_migration").unwrap();
            assert_eq!(migration["ok"], true);
            assert_eq!(migration["selection_changed"], true);
            assert_eq!(migration["generation"], serde_json::json!(generation(2)));
            assert_eq!(migration["fast_producer"], fast.identity.fingerprint());
            assert_eq!(migration["quality_producer"], quality.identity.fingerprint());
            for id in ["after", "again"] {
                let selected: Vec<_> = output.iter().filter(|frame| frame["id"] == id).collect();
                assert!(selected.iter().any(|frame| frame["phase"] == "initial"));
                assert!(selected.iter().any(|frame| frame["phase"] == "refined"));
                assert!(selected.iter().all(|frame| frame["generation"] == serde_json::json!(generation(2))));
            }
            assert_eq!(loads.load(Ordering::SeqCst), 1);
            assert_eq!(old.queries.load(Ordering::SeqCst), 1);
            assert_eq!(fast.queries.load(Ordering::SeqCst), 2);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 2);
            assert_eq!(fast.documents.load(Ordering::SeqCst), 3, "no inference during reopen");
            assert_eq!(quality.documents.load(Ordering::SeqCst), 3);
            assert_eq!(pinned.generation(), generation(1));
            assert_eq!(pinned.index().vectors().fast().embedder().dimension(), 2);
            assert_eq!(crate::fs::read(pinned.index().vectors().fast().vector_path()).unwrap(), old_bytes);
            assert_eq!(live.snapshot(&cx).await.unwrap().generation(), generation(2));
        }
    });
}

#[test]
fn ordinary_controller_grants_do_not_authorize_loading_new_models() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let model = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: model.clone(), quality: None }, documents(), true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let mut reader = input(&[
            request(&root.path().join("absent.json"), true),
            serde_json::json!({"query": "horizontal"}),
        ]);
        let mut output = Vec::new();
        run_with_controls(
            &live, &cx, &mut reader, &mut output, (crate::Mode::Fast, 3),
            Controls { activation: true, updates: true }, None,
            &query::Policy::new(&cx, None).unwrap(),
        ).await.unwrap();
        let output = frames(&output);
        assert_eq!(output[0]["model_migration_enabled"], false);
        assert!(output[1]["error"].as_str().unwrap().contains("model migration is disabled"));
        assert_eq!(output[1]["selection_changed"], false);
        assert_eq!(output.last().unwrap()["ok"], true, "session stays usable after refusal");
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), generation(1));
        assert_eq!(model.queries.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn stale_expected_generation_and_invalid_paths_fail_before_loading() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: old, quality: None }, documents(), true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let base = live.snapshot(&cx).await.unwrap();
        let authority = Authority { load: Box::new(|_, _, _, _| panic!("must refuse before loading")) };
        for variant in 0..4 {
            let mut value = request(&root.path().join("missing.json"), true);
            match variant {
                0 => value["expected_generation"] = serde_json::json!(generation(0)),
                1 => value["model_dir"] = serde_json::json!("relative"),
                2 => value["receipt"] = serde_json::json!("bad\0receipt"),
                _ => value["quality_backend"] = serde_json::json!("automatic"),
            }
            let request: Request = serde_json::from_value(value).unwrap();
            let error = prepare(&base, &cx, &request, &authority).await.err().unwrap();
            assert!(error.to_string().contains(match variant {
                0 => "expected_generation", 1 | 2 => "paths must be", _ => "quality backend",
            }));
        }
    });
}

#[test]
fn delivery_failure_after_migration_does_not_rollback_or_consume_next_request() {
    struct BrokenAck {
        writes: usize,
    }
    impl Write for BrokenAck {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.writes > 1 {
                return Err(io::Error::other("lost migration acknowledgement"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: old, quality: None }, documents(), true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let fast = Provider::new("replacement", 3);
        let models = Models { fast: fast.clone(), quality: None };
        let (candidate, selection) = sealed(
            &cx, &root.path().join("next"), 2, models.clone(), documents(), true,
        ).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let mut reader = input(&[
            request(&receipt, true), serde_json::json!({"query": "horizontal"}),
        ]);
        let mut output = BrokenAck { writes: 0 };
        let error = run_with_controls(
            &live, &cx, &mut reader, &mut output, (crate::Mode::Full, 3),
            permitted(models, Arc::new(AtomicUsize::new(0))), None,
            &query::Policy::new(&cx, None).unwrap(),
        ).await.unwrap_err();
        assert!(error.to_string().contains("lost migration acknowledgement"));
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), generation(2));
        assert_eq!(output.writes, 2, "no contradictory failure frame is appended");
        assert!(reader.position() < u64::try_from(reader.get_ref().len()).unwrap());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn valid_but_foreign_source_receipts_never_change_live_selection() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx,
            &root.path().join("old"),
            1,
            Models { fast: old, quality: None },
            documents(),
            true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let base = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("replacement", 3);
        let models = Models { fast: replacement.clone(), quality: None };
        let authority = Authority {
            load: Box::new(move |_, _, _, _| Ok(models.clone())),
        };
        for variant in 0..4 {
            let mut sources = documents();
            match variant {
                0 => sources[0].content = "different content".to_owned(),
                1 => sources[0].title = Some("different title".to_owned()),
                2 => {
                    sources[0].metadata.insert("scope".to_owned(), "different".to_owned());
                }
                _ => sources[0].id = "different-id".to_owned(),
            }
            let (candidate, selection) = sealed(
                &cx,
                &root.path().join(format!("different-{variant}")),
                2,
                Models { fast: replacement.clone(), quality: None },
                sources,
                true,
            ).await;
            let receipt = root.path().join(format!("different-{variant}.json"));
            save_selection(&selection, &receipt).unwrap();
            drop(candidate);
            let request: Request = serde_json::from_value(request(&receipt, true)).unwrap();
            let mut output = Vec::new();
            execute((&live).into(), &cx, &request, 1, Some(&authority), &mut output).await.unwrap();
            let output = frames(&output);
            assert_eq!(output.len(), 1);
            assert_eq!(output[0]["ok"], false);
            assert_eq!(output[0]["selection_changed"], false);
            assert!(output[0]["error"].as_str().unwrap().contains("native_ann.live.migration.source"));
            assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), base.index()));
        }
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 12);
        assert_eq!(replacement.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn source_update_during_model_loading_makes_even_a_higher_migration_stale() {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: old, quality: None }, documents(), true,
        ).await;
        let live = Arc::new(NativeLiveHybridIndex::new(&cx, built).unwrap());
        let base = live.snapshot(&cx).await.unwrap();
        let update = Arc::new(base
            .begin_update(&cx, root.path().join("source-update"), generation(2))
            .unwrap()
            .upsert_document(IndexableDocument::new("winner", "new document"))
            .build(&cx).await.unwrap());
        let fast = Provider::new("new-fast", 3);
        let models = Models { fast, quality: None };
        let (candidate, selection) = sealed(
            &cx, &root.path().join("model-update"), 9, models.clone(), documents(), true,
        ).await;
        let receipt = root.path().join("model-update.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let competing_live = Arc::clone(&live);
        let authority = Authority {
            load: Box::new(move |cx, _, _, _| {
                // A real native install inside the admission seam, not a mock
                // sequence change. No executor, thread or detached task needed.
                let mut install = Box::pin(competing_live.install(cx, &update));
                let Poll::Ready(Ok(installed)) = install
                    .as_mut().poll(&mut Context::from_waker(Waker::noop()))
                else {
                    panic!("uncontended native installation should complete");
                };
                assert_eq!(installed.generation(), generation(2));
                Ok(models.clone())
            }),
        };
        let request: Request = serde_json::from_value(request(&receipt, true)).unwrap();
        let mut output = Vec::new();
        execute(live.as_ref().into(), &cx, &request, 1, Some(&authority), &mut output).await.unwrap();
        let output = frames(&output);
        assert_eq!(output[0]["ok"], false);
        assert_eq!(output[0]["selection_changed"], false);
        assert!(output[0]["error"].as_str().unwrap().contains("native_ann.live.expected_current"));
        let selected = live.snapshot(&cx).await.unwrap();
        assert_eq!(selected.generation(), generation(2));
        assert!(selected.index().vectors().document("winner").is_some());
        assert!(base.index().vectors().document("winner").is_none());
    });
}

#[test]
fn failed_or_cancelled_loading_does_not_fall_back_or_replace_old_models() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: old.clone(), quality: None }, documents(), true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let base = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("replacement", 3);
        let models = Models { fast: replacement.clone(), quality: None };
        let (candidate, selection) = sealed(
            &cx, &root.path().join("next"), 2, models.clone(), documents(), true,
        ).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        for variant in 0..3 {
            let models = models.clone();
            let authority = Authority {
                load: Box::new(move |cx, _, _, _| {
                    if variant > 0 {
                        cx.set_cancel_requested(true);
                    }
                    if variant == 2 {
                        Ok(models.clone())
                    } else {
                        Err(bad("model loader failed"))
                    }
                }),
            };
            let request: Request = serde_json::from_value(request(&receipt, true)).unwrap();
            let mut output = Vec::new();
            execute((&live).into(), &cx, &request, 1, Some(&authority), &mut output).await.unwrap();
            cx.set_cancel_requested(false);
            let output = frames(&output);
            assert_eq!(output[0]["ok"], false);
            assert_eq!(output[0]["selection_changed"], false);
            if variant == 0 {
                assert_eq!(output[0]["status"], "failed");
                assert!(output[0]["error"].as_str().unwrap().contains("model loader failed"));
            } else {
                assert_eq!(output[0]["status"], "cancelled");
                assert_eq!(output[0]["code"], "cancelled");
                assert!(output[0]["error"].as_str().unwrap().contains("native_cli.model_migration"));
                assert!(!output[0]["error"].as_str().unwrap().contains("model loader failed"));
            }
            assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), base.index()));
        }
        assert_eq!(old.queries.load(Ordering::SeqCst), 0);
        assert_eq!(replacement.queries.load(Ordering::SeqCst), 0);
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 3);
    });
}

#[test]
fn quality_removal_is_explicit_and_later_requests_do_not_use_the_old_quality() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let old_quality = Provider::new("old-quality", 3);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: old, quality: Some(old_quality.clone()) }, documents(), true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let fast = Provider::new("replacement", 4);
        let models = Models { fast: fast.clone(), quality: None };
        let (candidate, selection) = sealed(
            &cx, &root.path().join("next"), 2, models.clone(), documents(), true,
        ).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        let mut reader = input(&[
            request(&receipt, true),
            serde_json::json!({"id": "quality", "query": "horizontal", "mode": "quality"}),
            serde_json::json!({"id": "full", "query": "vertical", "mode": "full"}),
        ]);
        let mut output = Vec::new();
        run_with_controls(
            &live, &cx, &mut reader, &mut output, (crate::Mode::Full, 3),
            permitted(models, Arc::new(AtomicUsize::new(0))), None,
            &query::Policy::new(&cx, None).unwrap(),
        ).await.unwrap();
        let output = frames(&output);
        assert_eq!(output[1]["ok"], true);
        assert_eq!(output[1]["quality"], false);
        assert!(output[1]["quality_producer"].is_null());
        let quality = output.iter().find(|frame| frame["id"] == "quality" && frame["event"] == "terminal").unwrap();
        assert_eq!(quality["ok"], false);
        let full: Vec<_> = output.iter().filter(|frame| frame["id"] == "full").collect();
        assert!(full.iter().any(|frame| frame["phase"] == "initial"));
        assert!(!full.iter().any(|frame| frame["phase"] == "refined"));
        assert_eq!(full.last().unwrap()["ok"], true);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(old_quality.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn activation_policy_and_producer_mismatches_preserve_the_predecessor() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let old = Provider::new("old", 2);
        let (built, _) = sealed(
            &cx, &root.path().join("old"), 1,
            Models { fast: old, quality: None }, documents(), true,
        ).await;
        let live = NativeLiveHybridIndex::new(&cx, built).unwrap();
        let base = live.snapshot(&cx).await.unwrap();
        let replacement = Provider::new("replacement", 3);
        let models = Models { fast: replacement.clone(), quality: None };
        let (candidate, selection) = sealed(
            &cx, &root.path().join("next"), 2, models.clone(), documents(), false,
        ).await;
        let receipt = root.path().join("next.json");
        save_selection(&selection, &receipt).unwrap();
        drop(candidate);
        for variant in 0..3 {
            let request: Request = serde_json::from_value(request(&receipt, variant == 0)).unwrap();
            let loaded = if variant == 1 {
                Models { fast: Provider::new("foreign-same-width", 3), quality: None }
            } else if variant == 2 {
                Models { fast: replacement.clone(), quality: Some(Provider::new("extra-quality", 4)) }
            } else {
                models.clone()
            };
            let authority = Authority { load: Box::new(move |_, _, _, _| Ok(loaded.clone())) };
            let mut output = Vec::new();
            execute((&live).into(), &cx, &request, 1, Some(&authority), &mut output).await.unwrap();
            let output = frames(&output);
            assert_eq!(output[0]["ok"], false);
            assert_eq!(output[0]["selection_changed"], false);
            assert!(std::ptr::eq(live.snapshot(&cx).await.unwrap().index(), base.index()));
        }
        assert_eq!(replacement.queries.load(Ordering::SeqCst), 0);
        assert_eq!(replacement.documents.load(Ordering::SeqCst), 3);
    });
}
