//! Actual command dispatch and real persisted source/Quill fixtures. Embedding
//! models are dropped and semantic files moved away before keyword execution.
use super::*;
use crate::{Arc, Path, PathBuf, fs, save_selection, update};
use asupersync::test_utils::run_test_with_cx;
use asupersync::time::{TimerDriverHandle, VirtualClock};
use frankensearch::ScoredResult;

struct Fixture {
    root: tempfile::TempDir,
    selection: Selection,
    receipt: PathBuf,
    expected: Vec<ScoredResult>,
    derived: Vec<PathBuf>,
}

/// Stream results normalized through `ScoredResult`, to compare with
/// `serde_json::to_value` of expected hits. The stream and `to_value` spell
/// the same f32 score differently (0.09852762520313264 vs ...263); parsing
/// back to f32 and serializing once more makes one spelling of each value.
fn results(value: &serde_json::Value) -> serde_json::Value {
    let hits: Vec<ScoredResult> = serde_json::from_value(value.clone()).unwrap();
    serde_json::to_value(hits).unwrap()
}

async fn fixture(cx: &Cx, partitioned: bool, graphs: bool) -> Fixture {
    let update::sharded_tests::Fixture {
        root,
        index,
        selection,
        fast,
        quality,
    } = update::sharded_tests::fixture(cx, graphs, true).await;
    let weak = Arc::downgrade(&fast);
    let paths = |partition: &frankensearch::native_ann::builder::NativeBuiltIndex| {
        let mut paths = Vec::new();
        for tier in std::iter::once(partition.fast()).chain(partition.quality()) {
            paths.push(tier.vector_path().to_path_buf());
            if let Some(graph) = tier.graph_path() {
                paths.push(graph.to_path_buf());
                let mut receipt = graph.as_os_str().to_os_string();
                receipt.push(".receipt");
                paths.push(PathBuf::from(receipt));
            }
        }
        paths
    };
    let (selection, expected, derived) = if partitioned {
        (
            selection,
            index.lexical().search(cx, "common", 100).await.unwrap(),
            index.vectors().partitions().iter().flat_map(paths).collect(),
        )
    } else {
        let single = index
            .begin_update(
                cx,
                root.path().join("single"),
                ArtifactGenerationIdentityV1::new(2, [57; 16]).unwrap(),
            )
            .unwrap()
            .build_hybrid(cx)
            .await
            .unwrap();
        (
            update::seal_selection(cx, (&single).into()).unwrap(),
            single.lexical().search(cx, "common", 100).await.unwrap(),
            paths(single.vectors()),
        )
    };
    drop(index);
    drop(fast);
    drop(quality);
    assert!(weak.upgrade().is_none(), "the command fixture owns no models");
    let receipt = root.path().join("keyword.json");
    save_selection(&selection, &receipt).unwrap();
    Fixture {
        root,
        selection,
        receipt,
        expected,
        derived,
    }
}

fn options(path: &Path, extra: &[&str]) -> Options {
    let mut args = vec![
        "search".to_owned(),
        "--receipt".to_owned(),
        path.display().to_string(),
        "--query".to_owned(),
        "common".to_owned(),
        "--lexical-only".to_owned(),
        "--limit".to_owned(),
        "100".to_owned(),
    ];
    args.extend(extra.iter().map(|s| (*s).to_owned()));
    Options::parse(args).unwrap().unwrap()
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

fn hide_semantic_files(f: &Fixture) {
    for (ordinal, path) in f.derived.iter().enumerate() {
        fs::rename(path, f.root.path().join(format!("saved-{ordinal}"))).unwrap();
    }
}

#[test]
fn explicit_keyword_policy_rejects_semantic_options_instead_of_ignoring_them() {
    for flag in [
        "--model-dir",
        "--quality-backend",
        "--quality-model-dir",
        "--mode",
        "--reranker-dir",
        "--rerank-window",
    ] {
        let value = match flag {
            "--quality-backend" => "onnx",
            "--mode" => "full",
            "--rerank-window" => "10",
            _ => "absent",
        };
        assert!(
            Options::parse(
                [
                    "search",
                    "--receipt",
                    "missing",
                    "--query",
                    "x",
                    "--lexical-only",
                    flag,
                    value,
                ]
                .into_iter()
                .map(str::to_owned),
            )
            .is_err()
        );
    }
    for args in [
        vec!["serve", "--receipt", "r", "--lexical-only", "--allow-updates"],
        vec![
            "index",
            "--receipt",
            "r",
            "--index-dir",
            "d",
            "--lexical-only",
        ],
        vec![
            "search",
            "--receipt",
            "r",
            "--query",
            "x",
            "--lexical-only",
            "--lexical-only",
        ],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
    let semantic = Options::parse(
        ["search", "--receipt", "r", "--query", "x"]
            .into_iter()
            .map(str::to_owned),
    )
    .unwrap()
    .unwrap();
    assert!(!semantic.lexical_only);
    assert!(options(Path::new("missing"), &[]).lexical_only);
}

#[test]
fn root_command_searches_both_layouts_without_models_or_semantic_artifacts() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            for graphs in [false, true] {
                let f = fixture(&cx, partitioned, graphs).await;
                hide_semantic_files(&f);
                let original_receipt = fs::read(&f.receipt).unwrap();
                for stream in [false, true] {
                    let mut options = options(&f.receipt, &[]);
                    options.stream = stream;
                    let mut output = Vec::new();
                    // The actual root branch gets no blocking/model loader pool.
                    // A semantic-path regression cannot succeed with this receipt.
                    crate::execute(&cx, options, &mut output, None)
                        .await
                        .unwrap();
                    let frames = frames(&output);
                    let page = if stream { &frames[1] } else { &frames[0] };
                    assert_eq!(page["phase"], "lexical");
                    assert_eq!(page["semantic_components_verified"], false);
                    assert_eq!(
                        page["generation"],
                        serde_json::to_value(f.selection.generation).unwrap()
                    );
                    assert_eq!(
                        page["source_layout"],
                        if partitioned { "sharded" } else { "single" }
                    );
                    let actual: Vec<ScoredResult> =
                        serde_json::from_value(page["results"].clone()).unwrap();
                    // ScoredResult has no PartialEq; compare every field.
                    assert_eq!(
                        serde_json::to_value(&actual).unwrap(),
                        serde_json::to_value(&f.expected).unwrap()
                    );
                    assert!(actual.iter().all(|hit| {
                        hit.index.is_none()
                            && hit.fast_score.is_none()
                            && hit.quality_score.is_none()
                    }));
                    if stream {
                        assert_eq!(frames.len(), 3);
                        for (seq, frame) in frames.iter().enumerate() {
                            assert_eq!(frame["seq"], seq);
                        }
                        assert_eq!(frames[0]["event"], "started");
                        assert_eq!(frames[2]["status"], "complete");
                    }
                }
                assert_eq!(fs::read(&f.receipt).unwrap(), original_receipt);
                assert!(f.derived.iter().all(|path| !path.exists()));
            }
        }
    });
}

#[test]
fn keyword_scope_collects_before_cutoff_and_preserves_raw_scores_and_metadata() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let f = fixture(&cx, partitioned, true).await;
            let index = Opened::open(&cx, &f.selection).await.unwrap();
            assert!(f.expected.len() > 1);
            let target = f.expected.last().unwrap();
            assert_ne!(f.expected[0].doc_id, target.doc_id);
            let filter = filter::Filter::parse(
                &serde_json::json!({
                    "ids": [target.doc_id], "id_prefix": target.doc_id,
                    "metadata": {"version": "old"},
                })
                .to_string(),
            )
            .unwrap();
            let payload = index
                .search(&cx, "common", 1, [None, Some(&filter)])
                .await
                .unwrap();
            let hits: Vec<ScoredResult> =
                serde_json::from_value(payload["results"].clone()).unwrap();
            assert_eq!(
                serde_json::to_value(&hits).unwrap(),
                serde_json::to_value([target]).unwrap()
            );
            assert_eq!(hits[0].score.to_bits(), target.score.to_bits());
            assert_eq!(payload["scope"]["eligible_documents"], 1);
            let deny = filter::Filter::parse(r#"{"ids":[]}"#).unwrap();
            let empty = index
                .search(&cx, "common", 10, [None, Some(&deny)])
                .await
                .unwrap();
            assert_eq!(empty["scope"]["eligible_documents"], 0);
            assert!(empty["results"].as_array().unwrap().is_empty());
        }
    });
}

#[test]
fn bad_keyword_receipts_and_used_artifacts_emit_no_success() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let f = fixture(&cx, partitioned, false).await;
            let source = f.selection.directory.join(if partitioned {
                "shard-000002/native.source.jsonl"
            } else {
                "native.source.jsonl"
            });
            let saved = f.root.path().join("saved-source");
            fs::rename(&source, &saved).unwrap();
            let mut output = Vec::new();
            assert!(
                crate::execute(&cx, options(&f.receipt, &["--stream"]), &mut output, None)
                    .await
                    .is_err()
            );
            assert!(output.is_empty(), "admission failed before started");
            fs::rename(saved, source).unwrap();
            let mut false_receipt: serde_json::Value =
                serde_json::from_slice(&fs::read(&f.receipt).unwrap()).unwrap();
            false_receipt["documents"] = serde_json::json!(999);
            let altered = f.root.path().join("false-count.json");
            fs::write(&altered, serde_json::to_vec(&false_receipt).unwrap()).unwrap();
            assert!(
                crate::execute(&cx, options(&altered, &[]), &mut output, None)
                    .await
                    .is_err()
            );
            assert!(output.is_empty());
            cx.set_cancel_requested(true);
            let error = crate::execute(&cx, options(&f.receipt, &[]), &mut output, None)
                .await
                .unwrap_err();
            cx.set_cancel_requested(false);
            assert!(matches!(
                error.downcast_ref::<SearchError>(),
                Some(SearchError::Cancelled { .. })
            ));
            assert!(output.is_empty());
        }
    });
}

#[test]
fn keyword_deadline_counts_started_delivery_and_later_queries_can_reuse_the_reader() {
    struct ExpireOnStart {
        bytes: Vec<u8>,
        clock: Arc<VirtualClock>,
    }
    impl Write for ExpireOnStart {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if frames(&self.bytes).last().unwrap()["event"] == "started" {
                self.clock.advance(10_000_000);
            }
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, false).await;
        let index = Opened::open(&cx, &f.selection).await.unwrap();
        let clock = Arc::new(VirtualClock::new());
        let policy = query::Policy {
            maximum_ms: Some(10),
            timer: Some(TimerDriverHandle::with_virtual_clock(clock.clone())),
            rerank: None,
        };
        let options = options(&f.receipt, &["--stream"]);
        let mut output = ExpireOnStart {
            bytes: Vec::new(),
            clock,
        };
        let error = execute_opened(&cx, &options, &index, &policy, &mut output)
            .await
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::SearchTimeout { .. })
        ));
        let timed_out = frames(&output.bytes);
        assert_eq!(timed_out.len(), 2);
        assert_eq!(timed_out[1]["status"], "timed_out");
        assert_eq!(timed_out[1]["partial_results"], false);
        assert!(!cx.is_cancel_requested());
        let mut next = Vec::new();
        execute_opened(&cx, &options, &index, &policy, &mut next)
            .await
            .unwrap();
        assert_eq!(frames(&next).last().unwrap()["status"], "complete");
    });
}

#[test]
fn broken_keyword_delivery_never_appends_a_contradictory_terminal_frame() {
    struct Broken {
        fail_at: usize,
        writes: usize,
    }
    impl Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes += 1;
            if self.writes == self.fail_at {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "failed delivery",
                ));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, false).await;
        let index = Opened::open(&cx, &f.selection).await.unwrap();
        let policy = query::Policy::default();
        for fail_at in [1, 2, 3] {
            let mut writer = Broken { fail_at, writes: 0 };
            assert!(
                execute_opened(
                    &cx,
                    &options(&f.receipt, &["--stream"]),
                    &index,
                    &policy,
                    &mut writer,
                )
                .await
                .is_err()
            );
            assert_eq!(writer.writes, fail_at);
        }
        assert_eq!(
            index.search(&cx, "common", 100, [None, None]).await.unwrap()["results"],
            serde_json::to_value(f.expected).unwrap()
        );
    });
}

fn server_options(path: &Path) -> Options {
    Options::parse([
        "serve".to_owned(), "--receipt".to_owned(), path.display().to_string(),
        "--lexical-only".to_owned(), "--limit".to_owned(), "100".to_owned(),
    ]).unwrap().unwrap()
}

fn keyword_input(messages: &[serde_json::Value]) -> std::io::Cursor<Vec<u8>> {
    let mut bytes = Vec::new();
    for message in messages {
        bytes.extend(crate::encode(message, serve::MAX_REQUEST_BYTES).unwrap());
    }
    std::io::Cursor::new(bytes)
}

#[test]
fn keyword_server_policy_is_explicit_read_only_and_model_free() {
    let options = server_options(Path::new("not-opened"));
    assert_eq!(options.command, crate::Command::Serve);
    assert!(options.lexical_only);
    assert!(!options.allow_updates);
    assert_eq!(options.activation, serve::ActivationPermission::Disabled);
    for extra in [
        vec!["--mode", "full"],
        vec!["--model-dir", "missing"],
        vec!["--quality-backend", "onnx"],
        vec!["--quality-model-dir", "missing"],
        vec!["--reranker-dir", "missing"],
        vec!["--rerank-window", "10"],
        vec!["--allow-updates"],
        vec!["--allow-activation"],
        vec!["--stream"],
        vec!["--lexical-only"],
    ] {
        let mut args = vec!["serve", "--receipt", "missing", "--lexical-only"];
        args.extend(extra);
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
}

#[test]
fn keyword_server_retains_admission_after_ready_without_models_or_reopening() {
    struct RetireDescriptors {
        bytes: Vec<u8>,
        receipt: PathBuf,
        descriptor: PathBuf,
        saved_receipt: PathBuf,
        saved_descriptor: PathBuf,
        retired: bool,
    }
    impl Write for RetireDescriptors {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if !self.retired && frames(&self.bytes).last().unwrap()["event"] == "ready" {
                // Admission has finished. Remove only already-consumed selection
                // metadata, never modify any file mapped by the Quill reader.
                fs::rename(&self.receipt, &self.saved_receipt)?;
                fs::rename(&self.descriptor, &self.saved_descriptor)?;
                self.retired = true;
            }
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            for graphs in [false, true] {
                let f = fixture(&cx, partitioned, graphs).await;
                hide_semantic_files(&f);
                let mut output = RetireDescriptors {
                    bytes: Vec::new(),
                    receipt: f.receipt.clone(),
                    descriptor: f.selection.directory.join(if partitioned {
                        "native.sharded-hybrid.json"
                    } else { "native.hybrid.json" }),
                    saved_receipt: f.root.path().join("saved-selection"),
                    saved_descriptor: f.root.path().join("saved-descriptor"),
                    retired: false,
                };
                serve_with_io(
                    &cx, &server_options(&f.receipt),
                    &mut keyword_input(&[
                        serde_json::json!({"id":"first", "query":"common"}),
                        serde_json::json!({"op":"status", "id":"health"}),
                        serde_json::json!({"id":"second", "query":"common"}),
                    ]), &mut output,
                ).await.unwrap();
                assert!(output.retired);
                // A per-query reopen cannot have succeeded against these paths.
                assert!(Selection::read(&f.receipt).is_err());
                assert!(Opened::open(&cx, &f.selection).await.is_err());
                let rows = frames(&output.bytes);
                assert_eq!(rows.len(), 8);
                assert_eq!(rows[0]["event"], "ready");
                assert_eq!(rows[0]["selection_policy"], "fixed_snapshot");
                assert_eq!(rows[0]["activation_enabled"], false);
                assert_eq!(rows[0]["updates_enabled"], false);
                for row in &rows {
                    assert_eq!(row["generation"], serde_json::to_value(f.selection.generation).unwrap());
                    assert_eq!(row["semantic_components_verified"], false);
                }
                for (offset, ordinal, id) in [(1, 1, "first"), (5, 3, "second")] {
                    for seq in 0..3 {
                        assert_eq!(rows[offset + seq]["seq"], seq);
                        assert_eq!(rows[offset + seq]["request"], ordinal);
                        assert_eq!(rows[offset + seq]["id"], id);
                    }
                    assert_eq!(results(&rows[offset + 1]["results"]), serde_json::to_value(&f.expected).unwrap());
                    assert_eq!(rows[offset + 1]["source_layout"], if partitioned { "sharded" } else { "single" });
                }
                assert_eq!(rows[4]["operation"], "status");
                assert_eq!(rows[4]["documents"], f.selection.documents);
                assert_eq!(rows[4]["selection_changed"], false);
                assert!(f.derived.iter().all(|path| !path.exists()));
                fs::rename(&output.saved_receipt, &output.receipt).unwrap();
                fs::rename(&output.saved_descriptor, &output.descriptor).unwrap();
            }
        }
    });
}

#[test]
fn keyword_server_intersects_filters_before_cutoff_and_keeps_raw_quill_results() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let f = fixture(&cx, partitioned, true).await;
            hide_semantic_files(&f);
            let first = &f.expected[0];
            let last = f.expected.last().unwrap();
            assert_ne!(first.doc_id, last.doc_id);
            let mut options = server_options(&f.receipt);
            options.limit = 1;
            options.filter = Some(filter::Filter::parse(
                &serde_json::json!({"ids":[last.doc_id], "metadata":{"version":"old"}}).to_string(),
            ).unwrap());
            let mut output = Vec::new();
            serve_with_io(&cx, &options, &mut keyword_input(&[
                serde_json::json!({"query":"common", "id":"inherited"}),
                serde_json::json!({"query":"common", "id":"wide", "filter":{"ids":[first.doc_id,last.doc_id]}}),
                serde_json::json!({"query":"common", "id":"excluded", "filter":{"ids":[first.doc_id]}}),
                serde_json::json!({"query":"common", "id":"empty", "filter":{"ids":[]}}),
            ]), &mut output).await.unwrap();
            let rows = frames(&output);
            assert_eq!(rows[0]["default_filter_applied"], true);
            let pages = rows.iter().filter(|row| row["event"] == "results").collect::<Vec<_>>();
            assert_eq!(pages.len(), 4);
            for page in &pages[..2] {
                assert_eq!(results(&page["results"]), serde_json::to_value([last]).unwrap());
                assert_eq!(page["scope"]["eligible_documents"], 1);
                let hits: Vec<ScoredResult> = serde_json::from_value(page["results"].clone()).unwrap();
                assert_eq!(hits[0].score.to_bits(), last.score.to_bits());
            }
            for page in &pages[2..] {
                assert_eq!(page["scope"]["eligible_documents"], 0);
                assert!(page["results"].as_array().unwrap().is_empty());
            }
        }
    });
}

#[test]
fn keyword_server_rejects_bad_and_writing_requests_without_poisoning_the_session() {
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, false).await;
        let protected = f.root.path().join("owner-evidence");
        fs::write(&protected, "unchanged").unwrap();
        let destination = f.root.path().join("must-not-exist");
        let mut bytes = b"\n \t\n\xff\n{invalid json\n".to_vec();
        let messages = [
            serde_json::json!({"query":"common", "mode":"full"}),
            serde_json::json!({"query":"common", "id":"bad\0id"}),
            serde_json::json!({"query":"common", "timeout_ms":0}),
            serde_json::json!({"op":"activate", "receipt":protected, "expected_generation":f.selection.generation}),
            serde_json::json!({"op":"update", "index_dir":destination, "new_receipt":protected,
                "expected_generation":f.selection.generation,
                "changes":[{"op":"delete", "id":"a"}]}),
            serde_json::json!({"op":"status", "query":"common"}),
            serde_json::json!({"op":"status", "id":"invalid\0status"}),
            serde_json::json!({"query":"common", "id":"healthy"}),
        ];
        bytes.extend(keyword_input(&messages).into_inner());
        bytes.extend_from_slice(b"quit\n");
        let consumed = bytes.len() as u64;
        bytes.extend_from_slice(b"{\"query\":\"must not be consumed\"}\n");
        let mut input = std::io::Cursor::new(bytes);
        let mut output = Vec::new();
        serve_with_io(&cx, &server_options(&f.receipt), &mut input, &mut output).await.unwrap();
        assert_eq!(input.position(), consumed);
        let rows = frames(&output);
        assert_eq!(rows.len(), 13);
        for (ordinal, row) in rows[1..10].iter().enumerate() {
            assert_eq!(row["ok"], false);
            assert_eq!(row["event"], "terminal");
            assert_eq!(row["seq"], 0);
            assert_eq!(row["request"], ordinal + 1);
            assert_eq!(row["partial_results"], false);
            assert!(row["results"].is_null());
            assert!(row["id"].is_null());
        }
        assert_eq!(rows[10]["id"], "healthy");
        assert_eq!(results(&rows[11]["results"]), serde_json::to_value(f.expected).unwrap());
        assert_eq!(rows[12]["status"], "complete");
        assert_eq!(fs::read_to_string(protected).unwrap(), "unchanged");
        assert!(!destination.exists());
    });
}

#[test]
fn keyword_server_deadlines_cannot_be_widened_and_expired_queries_do_not_poison_followers() {
    struct ExpireSelected {
        bytes: Vec<u8>,
        clock: Arc<VirtualClock>,
    }
    impl Write for ExpireSelected {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            let rows = frames(&self.bytes);
            let last = rows.last().unwrap();
            if last["event"] == "started" {
                match last["id"].as_str() {
                    Some("wide") => { self.clock.advance(10_000_000); }
                    Some("short") => { self.clock.advance(1_000_000); }
                    _ => {}
                }
            }
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, false).await;
        let index = Opened::open(&cx, &f.selection).await.unwrap();
        let clock = Arc::new(VirtualClock::new());
        let policy = query::Policy {
            maximum_ms: Some(10),
            timer: Some(TimerDriverHandle::with_virtual_clock(clock.clone())),
            rerank: None,
        };
        let mut output = ExpireSelected { bytes: Vec::new(), clock };
        run_opened(&cx, &index, &mut keyword_input(&[
            serde_json::json!({"query":"common", "id":"wide", "timeout_ms":100}),
            serde_json::json!({"query":"common", "id":"short", "timeout_ms":1}),
            serde_json::json!({"query":"common", "id":"healthy"}),
        ]), &mut output, 100, None, &policy).await.unwrap();
        let rows = frames(&output.bytes);
        assert_eq!(rows[0]["maximum_timeout_ms"], 10);
        for (position, budget) in [(2, 10), (4, 1)] {
            assert_eq!(rows[position]["status"], "timed_out");
            assert_eq!(rows[position]["code"], "search_timeout");
            assert_eq!(rows[position]["budget_ms"], budget);
            assert_eq!(rows[position]["partial_results"], false);
        }
        assert_eq!(results(&rows[6]["results"]), serde_json::to_value(f.expected).unwrap());
        assert_eq!(rows[7]["status"], "complete");
        assert!(!cx.is_cancel_requested());
    });
}

#[test]
fn keyword_server_broken_delivery_stops_before_reading_another_request() {
    struct Broken {
        writes: usize,
        fail_at: usize,
    }
    impl Write for Broken {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes += 1;
            if self.writes == self.fail_at {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken output"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, false).await;
        let index = Opened::open(&cx, &f.selection).await.unwrap();
        let first = serde_json::json!({"query":"common", "id":"first"});
        let first_len = crate::encode(&first, serve::MAX_REQUEST_BYTES).unwrap().len() as u64;
        for fail_at in [1, 2, 3, 4] {
            let mut output = Broken { writes: 0, fail_at };
            let mut input = keyword_input(&[first.clone(), serde_json::json!({"query":"second"})]);
            assert!(run_opened(&cx, &index, &mut input, &mut output, 100, None, &query::Policy::default()).await.is_err());
            assert_eq!(output.writes, fail_at, "never append a contradictory frame");
            assert_eq!(input.position(), if fail_at == 1 { 0 } else { first_len });
        }
        assert_eq!(index.search(&cx, "common", 100, [None, None]).await.unwrap()["results"],
                   serde_json::to_value(f.expected).unwrap());
    });
}

#[test]
fn keyword_server_admission_failure_precedes_ready_and_input_consumption() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let f = fixture(&cx, partitioned, false).await;
            let source = f.selection.directory.join(if partitioned {
                "shard-000002/native.source.jsonl"
            } else { "native.source.jsonl" });
            let saved = f.root.path().join("saved-source");
            fs::rename(&source, &saved).unwrap();
            let mut input = keyword_input(&[serde_json::json!({"query":"common"})]);
            let mut output = Vec::new();
            assert!(serve_with_io(&cx, &server_options(&f.receipt), &mut input, &mut output).await.is_err());
            assert!(output.is_empty());
            assert_eq!(input.position(), 0);
            fs::rename(saved, source).unwrap();
            cx.set_cancel_requested(true);
            let failure = serve_with_io(&cx, &server_options(&f.receipt), &mut input, &mut output).await;
            cx.set_cancel_requested(false);
            assert!(matches!(failure.unwrap_err().downcast_ref::<SearchError>(), Some(SearchError::Cancelled { .. })));
            assert!(output.is_empty());
            assert_eq!(input.position(), 0);
            serve_with_io(&cx, &server_options(&f.receipt), &mut input, &mut output).await.unwrap();
            assert_eq!(frames(&output).last().unwrap()["status"], "complete");
        }
    });
}

#[test]
fn shared_request_reader_bounds_allocation_and_leaves_oversized_or_post_quit_input() {
    let cx = Cx::for_testing();
    let mut record = Vec::new();
    let maximum = serve::MAX_REQUEST_BYTES;
    let mut too_large = std::io::Cursor::new(vec![b'x'; maximum + 20]);
    assert!(serve::read_request(&cx, &mut too_large, &mut record).is_err());
    assert_eq!(record.len(), maximum + 1);
    assert_eq!(too_large.position(), (maximum + 1) as u64);
    let mut exact = std::io::Cursor::new(vec![b'x'; maximum]);
    assert!(serve::read_request(&cx, &mut exact, &mut record).unwrap());
    assert_eq!(record.len(), maximum);
    assert!(!serve::read_request(&cx, &mut exact, &mut record).unwrap());
    for stop in ["quit", "exit"] {
        let prefix = format!("\n \t\n{stop}\n");
        let mut input = std::io::Cursor::new(format!("{prefix}later\n").into_bytes());
        assert!(!serve::read_request(&cx, &mut input, &mut record).unwrap());
        assert_eq!(input.position(), prefix.len() as u64);
    }
    let mut input = std::io::Cursor::new(b"query\n");
    cx.set_cancel_requested(true);
    let refused = serve::read_request(&cx, &mut input, &mut record);
    cx.set_cancel_requested(false);
    assert!(refused.is_err());
    assert_eq!(input.position(), 0);
}
