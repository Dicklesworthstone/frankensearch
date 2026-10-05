//! Executable protocol tests over real native/Quill generations and counted
//! deterministic providers. These are not relevance or performance evidence.
use super::*;
use frankensearch::native_ann::NativeSearchPhase;

struct Fixture {
    live: serve::NativeLiveHybridIndex,
    old: Selection,
    next: Selection,
    receipt: PathBuf,
    fast_calls: Arc<AtomicUsize>,
    quality_calls: Arc<AtomicUsize>,
}

async fn fixture(cx: &Cx, root: &Path, exact: bool) -> Fixture {
    let mut options = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
    options.exact = exact;
    let (models, fast_calls, quality_calls) = fixture_models(true);
    let (old_index, old) = build(cx, &options, &root.join("old"), source(), models).await.unwrap();
    save_selection(&old, &root.join("old.json")).unwrap();
    let edits = update::read_edits(&mut Cursor::new(concat!(
        "{\"op\":\"delete\",\"id\":\"garden.md\"}\n",
        "{\"op\":\"upsert\",\"id\":\"river.md\",\"content\":\"river currents\"}\n",
        "{\"op\":\"upsert\",\"id\":\"retry.rs\",\"content\":\"retry network requests with backoff\",\"metadata\":{\"version\":\"two\"}}\n",
    ))).unwrap();
    let (new_index, next) = update::apply(cx, &old, &old_index, &root.join("next"), edits, 8)
        .await.unwrap();
    let receipt = root.join("next.json");
    save_selection(&next, &receipt).unwrap();
    drop(new_index);
    Fixture {
        live: serve::NativeLiveHybridIndex::new(cx, old_index).unwrap(),
        old, next, receipt, fast_calls, quality_calls,
    }
}

fn activate(fixture: &Fixture) -> serde_json::Value {
    serde_json::json!({
        "op": "activate", "id": "switch", "receipt": fixture.receipt,
        "expected_generation": fixture.old.generation,
    })
}

fn input(messages: &[serde_json::Value]) -> Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for message in messages {
        bytes.extend(encode(message, MAX_OUTPUT_BYTES).unwrap());
    }
    Cursor::new(bytes)
}

fn counts(fixture: &Fixture) -> (usize, usize) {
    (fixture.fast_calls.load(Ordering::Relaxed), fixture.quality_calls.load(Ordering::Relaxed))
}

#[test]
fn activation_is_explicitly_opted_in_only_for_serving() {
    assert_eq!(options(&["serve", "--receipt", "r"]).activation, serve::ActivationPermission::Disabled);
    assert_eq!(options(&["serve", "--receipt", "r", "--allow-activation"]).activation, serve::ActivationPermission::Enabled);
    for args in [
        vec!["search", "--receipt", "r", "--query", "x", "--allow-activation"],
        vec!["index", "--receipt", "r", "--index-dir", "new", "--allow-activation"],
        vec!["serve", "--receipt", "r", "--allow-activation", "--allow-activation"],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn activation_switches_every_query_arm_without_running_models_or_mutating_receipts() {
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let f = fixture(&cx, root.path(), exact).await;
            let pinned = f.live.snapshot(&cx).await.unwrap();
            let before = counts(&f);
            let receipt_bytes = fs::read(&f.receipt).unwrap();
            let old_vector = fs::read(pinned.index().vectors().fast().vector_path()).unwrap();
            let mut queries = input(&[
                serde_json::json!({"id":"old-query", "query":"garden retry", "limit":10}),
                activate(&f),
                serde_json::json!({"id":"new-query", "query":"river retry", "limit":10}),
                serde_json::json!({"op":"status", "id":"current"}),
            ]);
            let mut output = ObservedOutput {
                bytes: Vec::new(), quality_calls: Arc::clone(&f.quality_calls),
                before_quality: before.1, initial_flushes: 0,
            };
            serve::run(&f.live, &cx, &mut queries, &mut output, (Mode::Full, 10), true)
                .await.unwrap();
            assert_eq!(counts(&f), (before.0 + 2, before.1 + 2));
            assert_eq!(output.initial_flushes, 2);
            let frames = output_frames(&output.bytes);
            for (id, generation, present, absent) in [
                ("old-query", f.old.generation, "garden.md", "river.md"),
                ("new-query", f.next.generation, "river.md", "garden.md"),
            ] {
                let query = frames.iter().filter(|frame| frame["id"] == id).collect::<Vec<_>>();
                assert_eq!(query.len(), 4);
                for (seq, frame) in query.iter().enumerate() {
                    assert_eq!(frame["seq"], seq);
                    assert_eq!(frame["generation"], serde_json::to_value(generation).unwrap());
                    if let Some(results) = frame["results"].as_array() {
                        assert!(results.iter().any(|hit| hit["doc_id"] == present));
                        assert!(!results.iter().any(|hit| hit["doc_id"] == absent));
                    }
                }
            }
            let switch = frames.iter().find(|frame| frame["id"] == "switch").unwrap();
            assert_eq!(switch["status"], "complete");
            assert_eq!(switch["selection_changed"], true);
            assert_eq!(switch["activation_scope"], "process_local");
            let status = frames.last().unwrap();
            assert_eq!(status["operation"], "status");
            assert_eq!(status["generation"], serde_json::to_value(f.next.generation).unwrap());
            assert_eq!(status["fast_native_hnsw"], !exact);
            assert_eq!(status["quality_native_hnsw"], !exact);
            assert_eq!(status["selection_changed"], false);
            assert_eq!(pinned.generation(), f.old.generation);
            assert_eq!(pinned.index().lexical().search(&cx, "garden", 10).await.unwrap().len(), 1);
            assert_eq!(fs::read(&f.receipt).unwrap(), receipt_bytes);
            assert_eq!(fs::read(pinned.index().vectors().fast().vector_path()).unwrap(), old_vector);
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn disabled_stale_malformed_and_damaged_activations_leave_the_old_cohort_queryable() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let f = fixture(&cx, root.path(), false).await;
        let before = counts(&f);
        let mut disabled = activate(&f);
        disabled["receipt"] = serde_json::json!(root.path().join("not-present"));
        let mut output = Vec::new();
        serve::run(&f.live, &cx, &mut input(&[disabled]), &mut output, (Mode::Full, 10), false)
            .await.unwrap();
        assert!(output_frames(&output)[1]["error"].as_str().unwrap().contains("disabled"));
        for fault in ["count", "identity", "producer", "quality", "snapshot"] {
            let mut receipt = serde_json::to_value(&f.next).unwrap();
            match fault {
                "count" => receipt["documents"] = serde_json::json!(f.next.documents + 1),
                "identity" => receipt["generation"]["nonce"][0] = serde_json::json!(255 ^ f.next.generation.nonce[0]),
                "producer" => receipt["fast_producer"] = serde_json::json!("0".repeat(64)),
                "quality" => receipt["quality_producer"] = serde_json::Value::Null,
                "snapshot" => receipt["snapshot"]["sha256"][0] = serde_json::json!(255 ^ f.next.snapshot.sha256[0]),
                _ => unreachable!("fixture fault"), // ubs:ignore — test fixture assertion.
            }
            let path = root.path().join(format!("bad-{fault}.json"));
            fs::write(&path, encode(&receipt, MAX_RECEIPT_BYTES).unwrap()).unwrap();
            let mut request = activate(&f);
            request["receipt"] = serde_json::json!(path);
            output.clear();
            serve::run(&f.live, &cx, &mut input(&[request]), &mut output, (Mode::Full, 10), true)
                .await.unwrap();
            let frames = output_frames(&output);
            assert_eq!(frames.len(), 2, "{fault}");
            assert_eq!(frames[1]["status"], "failed", "{fault}");
            assert_eq!(frames[1]["selection_changed"], false);
            assert_eq!(f.live.snapshot(&cx).await.unwrap().generation(), f.old.generation);
        }
        // A stale expected identity rejects before even opening the named file.
        let mut stale = activate(&f);
        stale["expected_generation"]["nonce"][0] = serde_json::json!(255 ^ f.old.generation.nonce[0]);
        stale["receipt"] = serde_json::json!(root.path().join("not-present"));
        let mut replay = activate(&f);
        replay["receipt"] = serde_json::json!(root.path().join("old.json"));
        output.clear();
        serve::run(&f.live, &cx, &mut input(&[
            stale, replay,
            serde_json::json!({"op":"activate", "query":"must not become a search"}),
            serde_json::json!({"op":"status", "unknown":true}),
            activate(&f), activate(&f),
        ]), &mut output, (Mode::Full, 10), true).await.unwrap();
        let frames = output_frames(&output);
        assert!(frames[1]["error"].as_str().unwrap().contains("expected_generation"));
        assert!(frames[2]["error"].as_str().unwrap().contains("strictly newer"));
        for row in [1, 2, 3, 4, 6] { assert_eq!(frames[row]["status"], "failed"); }
        assert_eq!(frames[5]["status"], "complete");
        assert_eq!(counts(&f), before, "control requests never embed");
        assert_eq!(f.live.snapshot(&cx).await.unwrap().generation(), f.next.generation);
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn pending_progressive_query_retains_old_rows_after_successor_activation() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let f = fixture(&cx, root.path(), false).await;
        let old = f.live.snapshot(&cx).await.unwrap();
        let mut stream = old.index().progressive(&cx, "garden retry", 10).unwrap();
        assert!(matches!(stream.next_phase().await.unwrap(), Some(NativeSearchPhase::Initial { .. })));
        serve::run(&f.live, &cx, &mut input(&[activate(&f)]), &mut Vec::new(), (Mode::Full, 10), true)
            .await.unwrap();
        let Some(NativeSearchPhase::Refined { results, .. }) = stream.next_phase().await.unwrap() else {
            panic!("original query must refine"); // ubs:ignore — test assertion.
        };
        assert!(results.iter().any(|hit| hit.doc_id == "garden.md"));
        assert!(!results.iter().any(|hit| hit.doc_id == "river.md"));
        let new = f.live.snapshot(&cx).await.unwrap();
        assert_eq!(new.generation(), f.next.generation);
        assert!(new.index().vectors().document("garden.md").is_none());
        assert!(old.index().vectors().document("garden.md").is_some());
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn failed_activation_acknowledgement_stops_session_without_rollback() {
    struct BrokenAck(usize);
    impl Write for BrokenAck {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 += 1;
            if self.0 == 1 { Ok(bytes.len()) } else { Err(io::ErrorKind::BrokenPipe.into()) }
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let f = fixture(&cx, root.path(), true).await;
        let before = counts(&f);
        let mut output = BrokenAck(0);
        assert!(serve::run(&f.live, &cx, &mut input(&[
            activate(&f), serde_json::json!({"query":"must not run"}),
        ]), &mut output, (Mode::Full, 10), true).await.is_err());
        assert_eq!(output.0, 2, "no error frame after a possibly partial acknowledgement");
        assert_eq!(counts(&f), before);
        assert_eq!(f.live.snapshot(&cx).await.unwrap().generation(), f.next.generation);
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn cancellation_before_control_dispatch_cannot_install_a_successor() {
    struct CancelAtReady<'a>(&'a Cx);
    impl Write for CancelAtReady<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> {
            self.0.set_cancel_requested(true);
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let f = fixture(&cx, root.path(), true).await;
        let before = counts(&f);
        let result = serve::run(&f.live, &cx, &mut input(&[activate(&f)]),
            &mut CancelAtReady(&cx), (Mode::Full, 10), true).await;
        cx.set_cancel_requested(false);
        assert!(result.is_err());
        assert_eq!(f.live.snapshot(&cx).await.unwrap().generation(), f.old.generation);
        assert_eq!(counts(&f), before);
        serve::run(&f.live, &cx, &mut input(&[activate(&f)]), &mut Vec::new(), (Mode::Full, 10), true)
            .await.unwrap();
        assert_eq!(f.live.snapshot(&cx).await.unwrap().generation(), f.next.generation);
    });
}
