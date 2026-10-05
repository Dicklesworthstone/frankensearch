use super::super::{Controls, MAX_REQUEST_BYTES, run_with_controls};
use super::*;
use crate::{
    Arc, Embedder, HnswParams, IndexableDocument, Mode, Models, NativeBuildPrecision,
    NativeBuildRetrieval, NativeIndexBuilder, encode, io, search,
};
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Waker};

use asupersync::test_utils::run_test_with_cx;
use frankensearch::{ModelCategory, SearchFuture};
use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
};
use frankensearch_core::traits::IdentityBoundEmbedding;

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
    fail: AtomicBool,
    hold: AtomicBool,
    drops: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Self {
        // Identified deterministic fixtures, not model-quality evidence.
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        Self {
            identity,
            calls: AtomicUsize::new(0),
            fail: AtomicBool::new(false),
            hold: AtomicBool::new(false),
            drops: AtomicUsize::new(0),
        }
    }
}

struct Waiting<'a>(&'a AtomicUsize);
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "injected provider failure".into(),
                });
            }
            if self.hold.load(Ordering::SeqCst) {
                let _waiting = Waiting(&self.drops);
                return std::future::pending().await;
            }
            let mut vector = vec![0.0; self.dimension()];
            vector[usize::from(text.contains("garden"))] = 1.0;
            Ok(vector)
        })
    }
    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let mut values = Vec::new();
            for text in texts {
                values.push(IdentityBoundEmbedding {
                    values: self.embed(cx, text).await?,
                    identity: self.identity.clone(),
                });
            }
            Ok(values)
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
        true
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

struct Fixture {
    root: tempfile::TempDir,
    fast: Arc<Provider>,
    quality: Arc<Provider>,
    live: NativeLiveHybridIndex,
    old: NativeHybridSnapshot,
}

async fn fixture(cx: &Cx, graphs: bool) -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let fast = Arc::new(Provider::new("warm-fast", 2));
    let quality = Arc::new(Provider::new("warm-quality", 3));
    let retrieval = if graphs {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 41,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    let index = NativeIndexBuilder::new(
        root.path().join("old"),
        ArtifactGenerationIdentityV1::new(1, [1; 16]).unwrap(),
        fast.clone(),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F32, retrieval)
    .with_quality_embedder(quality.clone())
    .unwrap()
    .with_quality_storage(NativeBuildPrecision::F32, retrieval)
    .unwrap()
    .add_documents([
        IndexableDocument::new("changed", "garden old body"),
        IndexableDocument::new("kept", "network stable body"),
        IndexableDocument::new("gone", "garden remove this"),
    ])
    .build_hybrid(cx)
    .await
    .unwrap();
    index.seal_for_reopen(cx).unwrap();
    let live = NativeLiveHybridIndex::new(cx, index).unwrap();
    let old = live.snapshot(cx).await.unwrap();
    Fixture {
        root,
        fast,
        quality,
        live,
        old,
    }
}

fn counts(f: &Fixture) -> (usize, usize) {
    (
        f.fast.calls.load(Ordering::SeqCst),
        f.quality.calls.load(Ordering::SeqCst),
    )
}

fn image(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut result = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else {
                result.insert(entry.path(), fs::read(entry.path()).unwrap());
            }
        }
    }
    result
}

fn message(f: &Fixture, name: &str, changes: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "op":"update", "id":name, "expected_generation":f.old.generation(),
        "index_dir":f.root.path().join(name),
        "new_receipt":f.root.path().join(format!("{name}.json")), "changes":changes,
    })
}

fn changes() -> serde_json::Value {
    serde_json::json!([
        {"op":"upsert","id":"changed","content":"network replacement body"},
        {"op":"delete","id":"gone"},
        {"op":"upsert","id":"new","content":"discarded first value"},
        {"op":"upsert","id":"new","content":"network added body"},
        {"op":"upsert","id":"kept","content":"network stable body","metadata":{"tenant":"blue"}},
    ])
}

fn input(messages: &[serde_json::Value]) -> io::Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for message in messages {
        bytes.extend(encode(message, MAX_REQUEST_BYTES).unwrap());
    }
    io::Cursor::new(bytes)
}

async fn send(
    f: &Fixture,
    cx: &Cx,
    message: serde_json::Value,
    allowed: bool,
) -> Vec<serde_json::Value> {
    let mut output = Vec::new();
    run_with_controls(
        &f.live,
        cx,
        &mut input(&[message]),
        &mut output,
        (Mode::Full, 10),
        Controls {
            activation: false,
            updates: allowed,
        },
        None,
        &query::Policy::default(),
    )
    .await
    .unwrap();
    output
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

#[test]
fn warm_update_reuses_models_and_unchanged_vectors_and_reopens_complete_successor() {
    run_test_with_cx(|cx| async move {
        for graphs in [false, true] {
            let f = fixture(&cx, graphs).await;
            let before = image(f.old.index().vectors().directory());
            let inference = counts(&f);
            let frames = send(&f, &cx, message(&f, "next", &changes()), true).await;
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[0]["updates_enabled"], true);
            assert_eq!(frames[0]["activation_enabled"], false);
            let receipt = &frames[1];
            assert_eq!(receipt["ok"], true);
            assert_eq!(receipt["edited_ids"], 4);
            assert_eq!(receipt["receipt_state"], "durable");
            assert_eq!(receipt["stage"], "installed");
            // Only the final changed/new body is embedded. Metadata-only edit,
            // deleted source and overwritten upsert cause no extra inference.
            assert_eq!(counts(&f), (inference.0 + 2, inference.1 + 2));
            let selected = Selection::read(&f.root.path().join("next.json")).unwrap();
            let now = f.live.snapshot(&cx).await.unwrap();
            assert_eq!(now.generation(), selected.generation);
            assert_eq!(now.generation().sequence, 2);
            assert_eq!(
                now.index()
                    .vectors()
                    .documents()
                    .iter()
                    .map(|d| d.id.as_str())
                    .collect::<Vec<_>>(),
                ["changed", "kept", "new"]
            );
            assert_eq!(
                now.index().vectors().document("kept").unwrap().metadata["tenant"],
                "blue"
            );
            assert_eq!(now.index().vectors().fast().graph_path().is_some(), graphs);
            assert_eq!(
                now.index()
                    .vectors()
                    .quality()
                    .unwrap()
                    .graph_path()
                    .is_some(),
                graphs
            );
            let reopened = selected
                .open(
                    &cx,
                    Models {
                        fast: f.fast.clone(),
                        quality: Some(f.quality.clone()),
                    },
                )
                .await
                .unwrap();
            assert_eq!(
                counts(&f),
                (inference.0 + 2, inference.1 + 2),
                "reopen must not infer"
            );
            for mode in [Mode::Fast, Mode::Quality, Mode::Full] {
                let payload = search(&reopened, &cx, "network", mode, 10, [None, None])
                    .await
                    .unwrap();
                let ids: Vec<_> = payload["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|hit| hit["doc_id"].as_str().unwrap())
                    .collect();
                assert!(ids.contains(&"new") && !ids.contains(&"gone"));
            }
            assert!(f.old.index().vectors().document("new").is_none());
            assert_eq!(
                f.old.index().vectors().document("changed").unwrap().content,
                "garden old body"
            );
            assert_eq!(image(f.old.index().vectors().directory()), before);
        }
    });
}

#[test]
fn refused_updates_do_no_io_or_inference_and_do_not_widen_activation_permission() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        let before = counts(&f);
        let mut cases = vec![(message(&f, "disabled", &changes()), false)];
        let mut stale = message(&f, "stale", &changes());
        stale["expected_generation"] =
            serde_json::json!(ArtifactGenerationIdentityV1::new(1, [2; 16]).unwrap());
        cases.push((stale, true));
        cases.push((message(&f, "empty", &serde_json::json!([])), true));
        cases.push((
            message(
                &f,
                "invalid",
                &serde_json::json!([
                    {"op":"upsert","id":" ","content":"invalid even if overwritten"},
                    {"op":"delete","id":" "}
                ]),
            ),
            true,
        ));
        let mut collision = message(&f, "collision", &changes());
        collision["index_dir"] = serde_json::json!(f.old.index().vectors().directory());
        cases.push((collision, true));
        let mut relative = message(&f, "relative", &changes());
        relative["new_receipt"] = serde_json::json!("relative.json");
        cases.push((relative, true));
        let before_files = image(f.root.path());
        for (request, allowed) in cases {
            let frames = send(&f, &cx, request, allowed).await;
            assert_eq!(frames[1]["ok"], false);
            assert_eq!(frames[1]["receipt_state"], "not_started");
            assert_eq!(counts(&f), before);
            assert_eq!(image(f.root.path()), before_files);
            assert_eq!(
                f.live.snapshot(&cx).await.unwrap().generation(),
                f.old.generation()
            );
        }
    });
}

#[test]
fn required_quality_failure_keeps_old_selection_and_allows_a_fresh_update() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        let before = image(f.old.index().vectors().directory());
        f.quality.fail.store(true, Ordering::SeqCst);
        let failed = send(&f, &cx, message(&f, "failed", &changes()), true).await;
        assert_eq!(failed[1]["ok"], false);
        assert_eq!(failed[1]["stage"], "building");
        assert!(!f.root.path().join("failed.json").exists());
        assert_eq!(
            f.live.snapshot(&cx).await.unwrap().generation(),
            f.old.generation()
        );
        f.quality.fail.store(false, Ordering::SeqCst);
        assert_eq!(
            send(&f, &cx, message(&f, "retry", &changes()), true).await[1]["ok"],
            true
        );
        assert_eq!(image(f.old.index().vectors().directory()), before);
    });
}

async fn prepared(f: &Fixture, cx: &Cx, name: &str, progress: &mut Progress) -> Prepared {
    let Request::Update {
        expected_generation,
        index_dir,
        new_receipt,
        changes,
        ..
    } = serde_json::from_value(message(f, name, &changes())).unwrap();
    prepare(
        &f.old,
        cx,
        expected_generation,
        &index_dir,
        &new_receipt,
        changes,
        progress,
    )
    .await
    .unwrap()
}

#[test]
fn stale_install_preserves_the_saved_complete_candidate_and_competing_winner() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        let mut progress = Progress::default();
        let prepared = prepared(&f, &cx, "loser", &mut progress).await;
        let winner = f
            .old
            .begin_update(
                &cx,
                f.root.path().join("winner"),
                ArtifactGenerationIdentityV1::new(2, [9; 16]).unwrap(),
            )
            .unwrap()
            .upsert_document(IndexableDocument::new("winner", "network winner"))
            .build(&cx)
            .await
            .unwrap();
        let winner = f.live.install(&cx, &winner).await.unwrap();
        assert!(
            commit(&prepared, &f.live, &cx, &mut progress)
                .await
                .is_err()
        );
        assert_eq!(progress.stage, "installing");
        assert_eq!(progress.receipt_state, "durable");
        let saved = Selection::read(&prepared.receipt).unwrap();
        assert_eq!(saved.generation, prepared.selection.generation);
        saved
            .open(
                &cx,
                Models {
                    fast: f.fast.clone(),
                    quality: Some(f.quality.clone()),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            f.live.snapshot(&cx).await.unwrap().generation(),
            winner.generation()
        );
        assert!(winner.index().vectors().document("winner").is_some());
    });
}

#[test]
fn receipt_collision_and_cancel_before_commit_cannot_install_a_candidate() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        for cancel in [false, true] {
            let mut progress = Progress::default();
            let prepared = prepared(
                &f,
                &cx,
                if cancel { "cancel" } else { "occupied" },
                &mut progress,
            )
            .await;
            if cancel {
                cx.set_cancel_requested(true);
            } else {
                fs::write(&prepared.receipt, "owner evidence").unwrap();
            }
            assert!(
                commit(&prepared, &f.live, &cx, &mut progress)
                    .await
                    .is_err()
            );
            cx.set_cancel_requested(false);
            assert_eq!(
                f.live.snapshot(&cx).await.unwrap().generation(),
                f.old.generation()
            );
            if cancel {
                assert!(!prepared.receipt.exists());
                assert_eq!(progress.receipt_state, "not_started");
            } else {
                assert_eq!(
                    fs::read_to_string(&prepared.receipt).unwrap(),
                    "owner evidence"
                );
                assert_eq!(progress.receipt_state, "uncertain");
            }
        }
    });
}

#[test]
fn dropping_pending_update_releases_provider_work_without_receipt_or_install() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        f.fast.hold.store(true, Ordering::SeqCst);
        let Request::Update {
            expected_generation,
            index_dir,
            new_receipt,
            changes,
            ..
        } = serde_json::from_value(message(&f, "pending", &changes())).unwrap();
        let mut progress = Progress::default();
        let mut pending = Box::pin(prepare(
            &f.old,
            &cx,
            expected_generation,
            &index_dir,
            &new_receipt,
            changes,
            &mut progress,
        ));
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(pending);
        assert_eq!(f.fast.drops.load(Ordering::SeqCst), 1);
        assert!(!new_receipt.exists());
        assert_eq!(
            f.live.snapshot(&cx).await.unwrap().generation(),
            f.old.generation()
        );
        f.fast.hold.store(false, Ordering::SeqCst);
        assert_eq!(
            send(&f, &cx, message(&f, "after-drop", &self::changes()), true).await[1]["ok"],
            true
        );
    });
}

#[test]
fn broken_update_acknowledgement_stops_input_without_rolling_back_installation() {
    struct BrokenAck {
        writes: usize,
    }
    impl Write for BrokenAck {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            if self.writes == 2 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "lost acknowledgement",
                ));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        let before = counts(&f);
        let mut output = BrokenAck { writes: 0 };
        assert!(
            run_with_controls(
                &f.live,
                &cx,
                &mut input(&[
                    message(&f, "ack", &changes()),
                    serde_json::json!({"query":"must not run"})
                ]),
                &mut output,
                (Mode::Full, 10),
                Controls {
                    activation: false,
                    updates: true
                },
                None,
                &query::Policy::default(),
            )
            .await
            .is_err()
        );
        assert_eq!(
            output.writes, 2,
            "never append contradictory failure output"
        );
        assert_eq!(counts(&f), (before.0 + 2, before.1 + 2));
        let saved = Selection::read(&f.root.path().join("ack.json")).unwrap();
        assert_eq!(
            f.live.snapshot(&cx).await.unwrap().generation(),
            saved.generation
        );
        let replay = send(&f, &cx, message(&f, "ack", &changes()), true).await;
        assert_eq!(replay[1]["ok"], false);
        assert_eq!(counts(&f), (before.0 + 2, before.1 + 2));
    });
}
