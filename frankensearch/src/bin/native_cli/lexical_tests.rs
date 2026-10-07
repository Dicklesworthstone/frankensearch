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
        vec!["serve", "--receipt", "r", "--lexical-only"],
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
                    assert_eq!(actual, f.expected);
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
                .search(&cx, "common", 1, Some(&filter))
                .await
                .unwrap();
            let hits: Vec<ScoredResult> =
                serde_json::from_value(payload["results"].clone()).unwrap();
            assert_eq!(hits, vec![target.clone()]);
            assert_eq!(hits[0].score.to_bits(), target.score.to_bits());
            assert_eq!(payload["scope"]["eligible_documents"], 1);
            let deny = filter::Filter::parse(r#"{"ids":[]}"#).unwrap();
            let empty = index
                .search(&cx, "common", 10, Some(&deny))
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
            index.search(&cx, "common", 100, None).await.unwrap()["results"],
            serde_json::to_value(f.expected).unwrap()
        );
    });
}
