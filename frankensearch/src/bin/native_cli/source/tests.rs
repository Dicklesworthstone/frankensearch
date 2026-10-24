use super::*;
use asupersync::test_utils::run_test_with_cx;
use asupersync::time::{TimerDriverHandle, VirtualClock};
use frankensearch::native_ann::builder::{
    NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder,
};
use frankensearch::{Embedder, ModelCategory, SearchFuture, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use std::future::Future;
use std::io::{self, Cursor};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [91; 16]).unwrap()
}

#[allow(clippy::needless_pass_by_value)]
fn request(windows: serde_json::Value) -> Request {
    serde_json::from_value(serde_json::json!({
        "op": "fetch", "id": "source-request", "expected_generation": generation(1),
        "documents": windows,
    }))
    .unwrap()
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes
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

fn message(sequence: u64, id: &str) -> serde_json::Value {
    serde_json::json!({
        "op": "fetch", "id": id, "expected_generation": generation(sequence),
        "documents": [{"doc_id":"a", "include_metadata":true}, {"doc_id":"z"}],
    })
}

#[test]
fn fetch_protocol_is_explicit_strict_and_bounded() {
    let valid = message(1, "fetch");
    assert!(matches!(
        serde_json::from_value::<serve::Message>(valid.clone()).unwrap(),
        serve::Message::Fetch(_)
    ));
    for field in ["expected_generation", "documents", "op"] {
        let mut malformed = valid.clone();
        malformed.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<serve::Message>(malformed).is_err());
    }
    for field in ["query", "receipt", "model_dir", "path", "allow_updates"] {
        let mut malformed = valid.clone();
        malformed[field] = serde_json::json!("CANARY_PRIVATE");
        assert!(serde_json::from_value::<serve::Message>(malformed).is_err());
    }
    for windows in [
        serde_json::json!([]),
        serde_json::json!([{"doc_id":" "}]),
        serde_json::json!([{"doc_id":"a\u{0}b"}]),
        serde_json::json!([{"doc_id":"a","max_bytes":3}]),
        serde_json::json!([{"doc_id":"a","max_bytes":65537}]),
        serde_json::json!(vec![serde_json::json!({"doc_id":"a"}); 33]),
        serde_json::json!([{"doc_id":"x".repeat(65536)}]),
        serde_json::json!([{"doc_id":"x".repeat(40000)}, {"doc_id":"y".repeat(40000)}]),
    ] {
        assert!(request(windows).validate().is_err());
    }
    let json = serde_json::to_string(&valid).unwrap();
    let duplicate = json.replacen("\"doc_id\":\"a\"", "\"doc_id\":\"a\",\"doc_id\":\"b\"", 1);
    assert_ne!(duplicate, json);
    assert!(serde_json::from_str::<serve::Message>(&duplicate).is_err());
    assert!(
        request(serde_json::json!([{"doc_id":"a","max_bytes":4}]))
            .validate()
            .is_ok()
    );
    assert!(
        request(serde_json::json!([{"doc_id":"a","max_bytes":65536}]))
            .validate()
            .is_ok()
    );
}

#[test]
fn utf8_windows_reconstruct_exact_source_without_gaps_or_normalization() {
    let cx = Cx::for_testing();
    let source = IndexableDocument::new("a", "  café🦀\n東京\u{0}<script>&\r\n");
    let mut collected = String::new();
    let mut offset = 0;
    loop {
        let request = request(serde_json::json!([{
            "doc_id":"a", "offset_bytes":offset, "max_bytes":4,
        }]));
        let bytes = render(&cx, generation(1), &request, 7, None, |_| Some(&source)).unwrap();
        let page = &frames(&bytes)[0];
        let document = &page["documents"][0];
        assert_eq!(page["generation"], serde_json::json!(generation(1)));
        assert_eq!(document["offset_bytes"], offset);
        let text = document["content"].as_str().unwrap();
        assert!(text.len() <= 4);
        collected.push_str(text);
        let end = usize::try_from(document["end_offset_bytes"].as_u64().unwrap()).unwrap();
        assert_eq!(end, offset + text.len());
        if document["next_offset_bytes"].is_null() {
            assert_eq!(end, source.content.len());
            break;
        }
        assert!(end > offset);
        offset = end;
    }
    assert_eq!(collected, source.content);
    let empty = IndexableDocument::new("empty", "");
    let request = request(serde_json::json!([{"doc_id":"empty"}]));
    let bytes = render(&cx, generation(1), &request, 1, None, |_| Some(&empty)).unwrap();
    assert_eq!(frames(&bytes)[0]["documents"][0]["content"], "");
    assert!(frames(&bytes)[0]["documents"][0]["next_offset_bytes"].is_null());
}

#[test]
fn batch_order_duplicates_and_conjunctive_scope_do_not_disclose_hidden_sources() {
    let cx = Cx::for_testing();
    let docs = [
        IndexableDocument::new("a", "allowed text")
            .with_title("allowed title")
            .with_metadata("tenant", "A"),
        IndexableDocument::new("z", "CANARY_PRIVATE")
            .with_title("CANARY_PRIVATE")
            .with_metadata("tenant", "B"),
    ];
    let base = filter::Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap();
    let request = request(serde_json::json!([
        {"doc_id":"a","include_metadata":true},
        {"doc_id":"z"}, {"doc_id":"absent"}, {"doc_id":"a"},
    ]));
    let bytes = render(&cx, generation(1), &request, 1, Some(&base), |id| {
        docs.iter().find(|doc| doc.id == id)
    })
    .unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("CANARY_PRIVATE"));
    let rows = &frames(&bytes)[0]["documents"];
    assert_eq!(rows[0]["content"], "allowed text");
    assert_eq!(rows[0]["metadata"]["tenant"], "A");
    assert_eq!(rows[0]["title"], "allowed title");
    assert_eq!(
        rows[1],
        serde_json::json!({"status":"unavailable","doc_id":"z"})
    );
    assert_eq!(
        rows[2],
        serde_json::json!({"status":"unavailable","doc_id":"absent"})
    );
    assert_eq!(rows[3]["content"], "allowed text");
    assert!(rows[3].get("metadata").is_none());
    assert!(rows[3].get("title").is_none());
    let request: Request = serde_json::from_value(serde_json::json!({
        "op":"fetch", "expected_generation":generation(1),
        "documents":[{"doc_id":"a"},{"doc_id":"z"}], "filter":{"ids":["z"]},
    }))
    .unwrap();
    let bytes = render(&cx, generation(1), &request, 2, Some(&base), |id| {
        docs.iter().find(|doc| doc.id == id)
    })
    .unwrap();
    assert!(
        frames(&bytes)[0]["documents"]
            .as_array()
            .unwrap()
            .iter()
            .all(|doc| doc["status"] == "unavailable")
    );
}

#[test]
fn stale_generation_foreign_lookup_and_invalid_final_window_never_return_a_prefix() {
    run_test_with_cx(|cx| async move {
        let source = IndexableDocument::new("a", "🦀retained");
        let policy = query::Policy::new(&cx, None).unwrap();
        let request = request(serde_json::json!([{"doc_id":"a"}]));
        let mut output = Vec::new();
        execute(
            &cx,
            generation(2),
            &request,
            1,
            None,
            &policy,
            |_| panic!("stale request must not look up source"),
            &mut output,
        )
        .await
        .unwrap();
        assert_eq!(frames(&output)[0]["ok"], false);
        assert!(frames(&output)[0].get("documents").is_none());
        for offset in [1, source.content.len() + 1, usize::MAX] {
            let request = self::request(serde_json::json!([
                {"doc_id":"a"}, {"doc_id":"a","offset_bytes":offset},
            ]));
            let mut output = Vec::new();
            execute(
                &cx,
                generation(1),
                &request,
                2,
                None,
                &policy,
                |_| Some(&source),
                &mut output,
            )
            .await
            .unwrap();
            assert_eq!(frames(&output).len(), 1);
            assert_eq!(frames(&output)[0]["ok"], false);
            assert!(frames(&output)[0].get("documents").is_none());
            assert!(!String::from_utf8_lossy(&output).contains("retained"));
        }
        let foreign = IndexableDocument::new("foreign", "CANARY_PRIVATE");
        let error = render(&cx, generation(1), &request, 3, None, |_| Some(&foreign)).unwrap_err();
        assert!(!error.to_string().contains("CANARY_PRIVATE"));
    });
}

#[test]
fn encoded_output_is_bounded_before_delivery_including_metadata_and_escaping() {
    run_test_with_cx(|cx| async move {
        let policy = query::Policy::new(&cx, None).unwrap();
        let mut source = IndexableDocument::new("a", "\u{0}".repeat(MAX_CONTENT_BYTES));
        source.metadata.insert(
            "huge".to_owned(),
            "CANARY_PRIVATE".repeat(MAX_RESPONSE_BYTES / 8),
        );
        for windows in [
            serde_json::json!([{"doc_id":"a","include_metadata":true}]),
            serde_json::json!(vec![serde_json::json!({"doc_id":"a","max_bytes":65536}); 3]),
        ] {
            let request = request(windows);
            let mut output = Vec::new();
            execute(
                &cx,
                generation(1),
                &request,
                1,
                None,
                &policy,
                |_| Some(&source),
                &mut output,
            )
            .await
            .unwrap();
            assert!(output.len() < MAX_RESPONSE_BYTES);
            assert_eq!(frames(&output).len(), 1);
            assert_eq!(frames(&output)[0]["ok"], false);
            assert!(frames(&output)[0].get("documents").is_none());
            assert!(!String::from_utf8_lossy(&output).contains("CANARY_PRIVATE"));
        }
        let request = request(serde_json::json!([{"doc_id":"a","max_bytes":4}]));
        let mut output = Vec::new();
        execute(
            &cx,
            generation(1),
            &request,
            2,
            None,
            &policy,
            |_| Some(&source),
            &mut output,
        )
        .await
        .unwrap();
        assert_eq!(
            frames(&output)[0]["ok"],
            true,
            "metadata is opt-in, not copied then discarded"
        );
    });
}

#[test]
fn cancellation_and_original_deadline_dominate_late_source_rendering() {
    let cx = Cx::for_testing();
    let clock = Arc::new(VirtualClock::new());
    let mut policy = query::Policy::new(&cx, Some(10)).unwrap();
    policy.timer = Some(TimerDriverHandle::with_virtual_clock(clock.clone()));
    let request = request(serde_json::json!([{"doc_id":"a"}]));
    let source = IndexableDocument::new("a", "late source");
    for cancelled in [false, true] {
        let mut output = Vec::new();
        let mut future = Box::pin(execute(
            &cx,
            generation(1),
            &request,
            1,
            None,
            &policy,
            |_| {
                clock.advance(11_000_000);
                cx.set_cancel_requested(cancelled);
                Some(&source)
            },
            &mut output,
        ));
        assert!(matches!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(()))
        ));
        drop(future);
        assert_eq!(
            frames(&output)[0]["status"],
            if cancelled { "cancelled" } else { "timed_out" }
        );
        assert!(frames(&output)[0].get("documents").is_none());
        assert!(policy.timer.as_ref().unwrap().is_empty());
        cx.set_cancel_requested(false);
    }
    cx.set_cancel_requested(true);
    assert!(
        render(&cx, generation(1), &request, 1, None, |_| panic!(
            "cancelled lookup"
        ))
        .is_err()
    );
}

struct Model {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Embedder for Model {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        2
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
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1.0, 0.0])
        })
    }
}

async fn fixture(
    cx: &Cx,
    partitioned: bool,
) -> (tempfile::TempDir, crate::live::Serving, Arc<Model>) {
    let root = tempfile::tempdir().unwrap();
    let model = Arc::new(Model {
        identity: EmbeddingIdentityBundleV1::explicit_test_model("source-fetch-test", 2),
        calls: AtomicUsize::new(0),
    });
    let builder = NativeIndexBuilder::new(root.path().join("index"), generation(1), model.clone())
        .unwrap()
        .with_fast_storage(
            NativeBuildPrecision::F32,
            NativeBuildRetrieval::Hnsw {
                params: crate::HnswParams::default(),
                seed: 91,
            },
        )
        .add_documents([
            IndexableDocument::new("a", "first retained source").with_metadata("tenant", "A"),
            IndexableDocument::new("z", "CANARY_PRIVATE").with_metadata("tenant", "B"),
        ]);
    let opened = if partitioned {
        crate::sharded::Opened::Sharded(Box::new(
            builder.build_sharded_hybrid(cx, 1).await.unwrap(),
        ))
    } else {
        crate::sharded::Opened::Single(Box::new(builder.build_hybrid(cx).await.unwrap()))
    };
    let live = crate::live::Serving::new(cx, opened).unwrap();
    (root, live, model)
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn actual_warm_fetch_preserves_snapshots_across_source_updates_and_never_infers() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (root, live, model) = fixture(&cx, partitioned).await;
            let old = live.borrow().snapshot(&cx).await.unwrap();
            let base = filter::Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap();
            let policy = query::Policy::new(&cx, None).unwrap();
            let before = model.calls.load(Ordering::SeqCst);
            let mut reader = input(&[message(1, "first"), message(1, "repeat")]);
            let mut output = Vec::new();
            serve::run_with_controls(
                live.borrow(),
                &cx,
                &mut reader,
                &mut output,
                (crate::Mode::Full, 2),
                serve::Controls::default(),
                Some(&base),
                &policy,
            )
            .await
            .unwrap();
            let replies = frames(&output);
            assert_eq!(replies[0]["source_fetch_supported"], true);
            assert_eq!(replies.len(), 3);
            for reply in &replies[1..] {
                assert_eq!(reply["documents"][0]["content"], "first retained source");
                assert_eq!(reply["documents"][1]["status"], "unavailable");
                assert_eq!(reply["seq"], 0);
            }
            assert_eq!(model.calls.load(Ordering::SeqCst), before);
            let edits = std::iter::once((
                "a".to_owned(),
                Some(
                    IndexableDocument::new("a", "replacement retained source")
                        .with_metadata("tenant", "A"),
                ),
            ))
            .collect();
            let candidate = old
                .prepare_update(
                    &cx,
                    &root.path().join("next"),
                    generation(2),
                    edits,
                    4,
                    partitioned.then_some(1),
                )
                .await
                .unwrap();
            live.borrow().install(&cx, &candidate).await.unwrap();
            let before = model.calls.load(Ordering::SeqCst);
            let mut reader = input(&[message(1, "stale"), message(2, "current")]);
            let mut output = Vec::new();
            serve::run_with_controls(
                live.borrow(),
                &cx,
                &mut reader,
                &mut output,
                (crate::Mode::Full, 2),
                serve::Controls::default(),
                Some(&base),
                &policy,
            )
            .await
            .unwrap();
            let replies = frames(&output);
            assert_eq!(replies[1]["ok"], false);
            assert!(replies[1].get("documents").is_none());
            assert_eq!(
                replies[2]["documents"][0]["content"],
                "replacement retained source"
            );
            assert_eq!(model.calls.load(Ordering::SeqCst), before);
            let request = request(serde_json::json!([{"doc_id":"a"}]));
            let old_index = old.index();
            let old_bytes = render(
                &cx,
                old.generation(),
                &request,
                1,
                None,
                |id| match old_index {
                    crate::cohort::Index::Single(index) => index.vectors().document(id),
                    crate::cohort::Index::Sharded(index) => index.vectors().document(id),
                },
            )
            .unwrap();
            assert_eq!(
                frames(&old_bytes)[0]["documents"][0]["content"],
                "first retained source"
            );
        }
    });
}

#[test]
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn failed_delivery_stops_without_an_extra_frame_or_consuming_the_next_request() {
    struct Broken {
        writes: usize,
    }
    impl Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.writes += 1;
            match self.writes {
                1 => Ok(bytes.len()),        // ready
                2 => Ok(3.min(bytes.len())), // part of the source frame
                _ => Err(io::Error::other("source output unavailable")),
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let (_root, live, model) = fixture(&cx, true).await;
        let before = model.calls.load(Ordering::SeqCst);
        let mut reader = input(&[message(1, "first"), message(1, "unread")]);
        let mut output = Broken { writes: 0 };
        let error = serve::run_with_controls(
            live.borrow(),
            &cx,
            &mut reader,
            &mut output,
            (crate::Mode::Full, 2),
            serve::Controls::default(),
            None,
            &query::Policy::new(&cx, None).unwrap(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("source output unavailable"));
        assert_eq!(output.writes, 3);
        assert!(reader.position() < u64::try_from(reader.get_ref().len()).unwrap());
        assert_eq!(model.calls.load(Ordering::SeqCst), before);
    });
}
