//! The shared warm transaction protocol over real sharded snapshots, using the
//! one-shot update fixture's counted producers rather than another mock index.
use super::*;
use crate::update::sharded_tests::{Provider, fixture as source_fixture};
use crate::{Arc, IndexableDocument, Mode, Models, encode, filter, io, serve};
use asupersync::test_utils::run_test_with_cx;
use frankensearch::native_ann::builder::sharded::live::{
    NativeLiveShardedHybridIndex, NativeShardedSnapshot,
};
use frankensearch::native_ann::{NativeShardedResult, NativeShardedSearchPhase};
use frankensearch::{RerankDocument, RerankScore, Reranker, SearchFuture};
use frankensearch_index::ValidatedFsviBytes;
use std::future::Future;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::task::{Context, Waker};

struct Fixture {
    root: tempfile::TempDir,
    live: NativeLiveShardedHybridIndex,
    old: NativeShardedSnapshot,
    models: Models,
    fast: Arc<Provider>,
    quality: Arc<Provider>,
}

async fn fixture(cx: &Cx, graphs: bool) -> Fixture {
    let source = source_fixture(cx, graphs, true).await;
    let models = source.models();
    let live = NativeLiveShardedHybridIndex::new(cx, source.index).unwrap();
    let old = live.snapshot(cx).await.unwrap();
    Fixture {
        root: source.root,
        live,
        old,
        models,
        fast: source.fast,
        quality: source.quality.unwrap(),
    }
}

fn counts(f: &Fixture) -> (usize, usize) {
    (
        f.fast.submitted.load(Ordering::SeqCst),
        f.quality.submitted.load(Ordering::SeqCst),
    )
}

fn changes() -> serde_json::Value {
    serde_json::json!([
        {"op":"upsert", "id":"a", "content":"common stable-a", "metadata":{"version":"new"}},
        {"op":"delete", "id":"c"},
        {"op":"upsert", "id":"aa", "content":"arrival common"},
        {"op":"upsert", "id":"b", "content":"discarded body"},
        {"op":"upsert", "id":"b", "content":"replacement common"},
    ])
}

fn message(f: &Fixture, name: &str) -> serde_json::Value {
    serde_json::json!({
        "op":"update", "id":name, "expected_generation":f.old.generation(),
        "index_dir":f.root.path().join(name),
        "new_receipt":f.root.path().join(format!("{name}.json")),
        "changes":changes(), "shard_size":3,
    })
}

fn messages(values: &[serde_json::Value]) -> io::Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend(encode(value, super::super::MAX_REQUEST_BYTES).unwrap());
    }
    io::Cursor::new(bytes)
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

async fn send(
    f: &Fixture,
    cx: &Cx,
    value: serde_json::Value,
    allowed: bool,
) -> Vec<serde_json::Value> {
    let mut output = Vec::new();
    serve::run_with_controls(
        &f.live,
        cx,
        &mut messages(&[value]),
        &mut output,
        (Mode::Full, 10),
        serve::Controls {
            activation: false,
            updates: allowed,
        },
        None,
        &query::Policy::default(),
    )
    .await
    .unwrap();
    frames(&output)
}

async fn prepare_request(
    f: &Fixture,
    cx: &Cx,
    value: serde_json::Value,
    progress: &mut Progress,
) -> Result<Prepared> {
    let Request::Update {
        expected_generation,
        index_dir,
        new_receipt,
        changes,
        shard_size,
        ..
    } = serde_json::from_value(value).unwrap();
    prepare_with_partition(
        &live::Snapshot::Sharded(f.old.clone()),
        cx,
        expected_generation,
        &index_dir,
        &new_receipt,
        changes,
        shard_size,
        progress,
    )
    .await
}

fn assert_rows(snapshot: &NativeShardedSnapshot, hits: &[NativeShardedResult]) {
    for hit in hits {
        assert!(hit.result.index.is_none());
        for (quality, row) in [(false, hit.fast_row), (true, hit.quality_row)] {
            let Some(row) = row else {
                continue;
            };
            let partition = &snapshot.index().vectors().partitions()[row.shard];
            let tier = if quality {
                partition.quality().unwrap()
            } else {
                partition.fast()
            };
            let owner = ValidatedFsviBytes::from_arc(
                fs::read(tier.vector_path()).unwrap().into(),
                tier.binding(),
            )
            .unwrap();
            assert_eq!(
                owner
                    .doc_id_at(usize::try_from(row.physical_row).unwrap())
                    .unwrap(),
                hit.result.doc_id,
            );
        }
    }
}

#[derive(Default)]
struct Scorer {
    seen: Mutex<Vec<RerankDocument>>,
}

impl Reranker for Scorer {
    fn id(&self) -> &'static str {
        "warm-sharded-retained-source-probe"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn rerank<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            *self.seen.lock().unwrap() = documents.to_vec();
            Ok(documents
                .iter()
                .enumerate()
                .map(|(original_rank, document)| RerankScore {
                    doc_id: document.doc_id.clone(),
                    original_rank,
                    raw_logit: None,
                    score: if document.doc_id == "c" { 1.0 } else { 0.0 },
                })
                .collect())
        })
    }
}

#[test]
fn warm_sharded_update_repartitions_all_arms_and_old_scoped_rerank_keeps_its_sources() {
    run_test_with_cx(|cx| async move {
        for graphs in [false, true] {
            let f = fixture(&cx, graphs).await;
            let original_receipt = fs::read(f.root.path().join("old.json")).unwrap();
            let scope = f
                .old
                .index()
                .scope(&cx, |doc| Ok(doc.metadata["version"] == "old"))
                .unwrap();
            let scorer = Scorer::default();
            let mut old_query = scope
                .progressive_with_reranker(&cx, "common", 1, &scorer, 5)
                .unwrap();
            assert!(matches!(
                old_query.next_phase().await.unwrap(),
                Some(NativeShardedSearchPhase::Initial { .. })
            ));
            assert!(scorer.seen.lock().unwrap().is_empty());
            let before = counts(&f);
            let base_filter = filter::Filter::parse(r#"{"metadata":{"version":"old"}}"#).unwrap();
            let mut output = Vec::new();
            serve::run_with_controls(
                &f.live,
                &cx,
                &mut messages(&[
                    message(&f, "next"),
                    serde_json::json!({"query":"common", "mode":"quality", "limit":10}),
                    serde_json::json!({"op":"status"}),
                ]),
                &mut output,
                (Mode::Full, 10),
                serve::Controls {
                    activation: false,
                    updates: true,
                },
                Some(&base_filter),
                &query::Policy::default(),
            )
            .await
            .unwrap();
            let frames = frames(&output);
            assert_eq!(frames[0]["updates_enabled"], true);
            assert_eq!(frames[0]["activation_enabled"], false);
            let update = &frames[1];
            assert_eq!(update["ok"], true);
            assert_eq!(update["stage"], "installed");
            assert_eq!(update["receipt_state"], "durable");
            assert_eq!(update["layout"], "sharded");
            assert_eq!(update["partitions"], 2);
            assert_eq!(update["edited_ids"], 4);
            let selected = Selection::read(&f.root.path().join("next.json")).unwrap();
            let current = f.live.snapshot(&cx).await.unwrap();
            assert_eq!(current.generation(), selected.generation);
            assert_eq!(current.generation().sequence, 2);
            assert_eq!(counts(&f), (before.0 + 2, before.1 + 2));
            let page = frames
                .iter()
                .find(|frame| frame["event"] == "results")
                .unwrap();
            assert_eq!(page["scope"]["eligible_documents"], 2);
            assert_eq!(page["results"].as_array().unwrap().len(), 2);
            assert!(
                page["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|hit| hit["doc_id"] == "d" || hit["doc_id"] == "z")
            );
            assert_eq!(
                frames.last().unwrap()["generation"],
                serde_json::to_value(selected.generation).unwrap()
            );
            let reopened = sharded::open(&cx, &selected, f.models.clone())
                .await
                .unwrap();
            assert_eq!(reopened.vectors().partitions().len(), 2);
            assert_eq!(
                reopened.vectors().document("a").unwrap().metadata["version"],
                "new"
            );
            assert!(reopened.vectors().document("c").is_none());
            assert!(
                reopened
                    .lexical()
                    .search(&cx, "obsolete", 10)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert_eq!(
                reopened.lexical().search(&cx, "arrival", 10).await.unwrap()[0].doc_id,
                "aa"
            );
            assert_eq!(counts(&f), (before.0 + 2, before.1 + 2));
            assert!(matches!(
                old_query.next_phase().await.unwrap(),
                Some(NativeShardedSearchPhase::Refined { .. })
            ));
            let Some(NativeShardedSearchPhase::Reranked { results, .. }) =
                old_query.next_phase().await.unwrap()
            else {
                panic!("old retained query must finish reranking");
            };
            assert_eq!(results[0].result.doc_id, "c");
            assert_rows(&f.old, &results);
            let inputs = scorer.seen.lock().unwrap();
            assert_eq!(inputs.len(), 5);
            for input in inputs.iter() {
                assert_eq!(input.text, scope.document(&input.doc_id).unwrap().content);
                assert_ne!(input.doc_id, "aa");
            }
            drop(inputs);
            assert_eq!(
                fs::read(f.root.path().join("old.json")).unwrap(),
                original_receipt
            );
        }
    });
}

#[test]
fn warm_sharded_updates_require_permission_matching_predecessor_and_explicit_valid_size() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        let before = counts(&f);
        let denied = send(&f, &cx, message(&f, "denied"), false).await;
        assert_eq!(denied[1]["stage"], "admission");
        assert!(
            denied[1]["error"]
                .as_str()
                .unwrap()
                .contains("updates are disabled")
        );
        for (name, size) in [
            ("missing", None),
            ("zero", Some(0)),
            ("excess", Some(100_001)),
        ] {
            let mut value = message(&f, name);
            value["shard_size"] = serde_json::json!(size);
            let output = send(&f, &cx, value, true).await;
            assert_eq!(output[1]["ok"], false);
            assert_eq!(output[1]["receipt_state"], "not_started");
            assert!(!f.root.path().join(name).exists());
            assert!(!f.root.path().join(format!("{name}.json")).exists());
        }
        let mut stale = message(&f, "stale");
        stale["expected_generation"] = serde_json::to_value(
            ArtifactGenerationIdentityV1::new(f.old.generation().sequence, [19; 16]).unwrap(),
        )
        .unwrap();
        stale["index_dir"] = serde_json::json!(f.root.path().join("missing-parent/candidate"));
        let refused = send(&f, &cx, stale, true).await;
        assert!(
            refused[1]["error"]
                .as_str()
                .unwrap()
                .contains("expected_generation")
        );
        assert_eq!(refused[1]["selection_changed"], false);
        assert_eq!(counts(&f), before);
        assert_eq!(
            f.live.snapshot(&cx).await.unwrap().generation(),
            f.old.generation()
        );
        assert!(!f.root.path().join("denied").exists());
        // A writing grant is not an external activation grant.
        let control = serde_json::json!({"op":"activate", "receipt":f.root.path().join("absent"),
            "expected_generation":f.old.generation()});
        let refused = send(&f, &cx, control, true).await;
        assert!(
            refused[1]["error"]
                .as_str()
                .unwrap()
                .contains("activation is disabled")
        );
    });
}

#[test]
fn stale_warm_sharded_install_keeps_the_durable_candidate_and_competing_winner() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        let mut progress = Progress::default();
        let prepared = prepare_request(&f, &cx, message(&f, "loser"), &mut progress)
            .await
            .unwrap();
        assert!(!prepared.receipt.exists());
        let winner = f
            .old
            .begin_update(
                &cx,
                f.root.path().join("winner"),
                ArtifactGenerationIdentityV1::new(2, [9; 16]).unwrap(),
            )
            .unwrap()
            .upsert_document(IndexableDocument::new("winner", "winner common"))
            .build(&cx, 4)
            .await
            .unwrap();
        let installed = f.live.install(&cx, &winner).await.unwrap();
        assert!(
            commit(&prepared, &f.live, &cx, &mut progress)
                .await
                .is_err()
        );
        assert_eq!(progress.stage, "installing");
        assert_eq!(progress.receipt_state, "durable");
        let saved = Selection::read(&prepared.receipt).unwrap();
        let candidate = sharded::open(&cx, &saved, f.models.clone()).await.unwrap();
        assert!(candidate.vectors().document("aa").is_some());
        assert!(candidate.vectors().document("winner").is_none());
        let current = f.live.snapshot(&cx).await.unwrap();
        assert_eq!(current.generation(), installed.generation());
        assert!(current.index().vectors().document("winner").is_some());
        assert!(current.index().vectors().document("aa").is_none());
    });
}

#[test]
fn warm_sharded_receipt_collision_and_cancelled_commit_do_not_install_or_overwrite() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false).await;
        for cancel in [false, true] {
            let mut progress = Progress::default();
            let name = if cancel { "cancelled" } else { "occupied" };
            let prepared = prepare_request(&f, &cx, message(&f, name), &mut progress)
                .await
                .unwrap();
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
                assert_eq!(progress.receipt_state, "not_started");
                assert!(!prepared.receipt.exists());
            } else {
                assert_eq!(progress.receipt_state, "uncertain");
                assert_eq!(
                    fs::read_to_string(&prepared.receipt).unwrap(),
                    "owner evidence"
                );
            }
        }
    });
}

#[test]
fn failed_or_dropped_late_warm_shard_leaves_old_head_queryable_without_a_receipt() {
    run_test_with_cx(|cx| async move {
        for fault in [1, 2] {
            let f = fixture(&cx, false).await;
            let mut value = message(&f, "incomplete");
            value["shard_size"] = serde_json::json!(2);
            value["changes"] =
                serde_json::json!([{"op":"upsert", "id":"zz-new", "content":"arrival common"}]);
            f.quality.fault.store(fault, Ordering::SeqCst);
            let before_drops = f.quality.drops.load(Ordering::SeqCst);
            let mut progress = Progress::default();
            let mut pending = Box::pin(prepare_request(&f, &cx, value, &mut progress));
            if fault == 2 {
                assert!(
                    pending
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
                assert!(f.live.search(&cx, "common", 5).await.is_ok());
            } else {
                assert!(pending.as_mut().await.is_err());
            }
            drop(pending);
            assert_eq!(f.quality.drops.load(Ordering::SeqCst), before_drops + 1);
            assert_eq!(progress.receipt_state, "not_started");
            assert!(!f.root.path().join("incomplete.json").exists());
            assert!(
                f.root
                    .path()
                    .join("incomplete/shard-000000/fast.fsvi")
                    .is_file()
            );
            assert!(
                !f.root
                    .path()
                    .join("incomplete/native.sharded-hybrid.json")
                    .exists()
            );
            assert_eq!(
                f.live.snapshot(&cx).await.unwrap().generation(),
                f.old.generation()
            );
            f.quality.fault.store(0, Ordering::SeqCst);
            assert_eq!(
                send(&f, &cx, message(&f, "retry"), true).await[1]["ok"],
                true
            );
        }
    });
}

#[test]
fn lost_warm_sharded_ack_stops_input_and_preserves_the_installed_receipt() {
    struct BrokenAck {
        writes: usize,
    }
    impl Write for BrokenAck {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            let value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            if value["operation"] == "update" {
                assert_eq!(value["ok"], true);
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
        let f = fixture(&cx, true).await;
        let update = message(&f, "ack");
        let consumed = encode(&update, super::super::MAX_REQUEST_BYTES)
            .unwrap()
            .len() as u64;
        let mut input = messages(&[update, serde_json::json!({"query":"must not execute"})]);
        let mut output = BrokenAck { writes: 0 };
        assert!(
            serve::run_with_controls(
                &f.live,
                &cx,
                &mut input,
                &mut output,
                (Mode::Full, 10),
                serve::Controls {
                    activation: false,
                    updates: true
                },
                None,
                &query::Policy::default(),
            )
            .await
            .is_err()
        );
        assert_eq!(input.position(), consumed);
        assert_eq!(
            output.writes, 2,
            "never append a contradictory failure frame"
        );
        let saved = Selection::read(&f.root.path().join("ack.json")).unwrap();
        assert_eq!(
            f.live.snapshot(&cx).await.unwrap().generation(),
            saved.generation
        );
        let reopened = sharded::open(&cx, &saved, f.models.clone()).await.unwrap();
        assert!(reopened.vectors().document("aa").is_some());
        let before = counts(&f);
        let replay = send(&f, &cx, message(&f, "ack"), true).await;
        assert_eq!(replay[1]["ok"], false);
        assert!(
            replay[1]["error"]
                .as_str()
                .unwrap()
                .contains("expected_generation")
        );
        assert_eq!(counts(&f), before);
    });
}
