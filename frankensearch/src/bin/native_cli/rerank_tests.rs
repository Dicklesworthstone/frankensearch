//! Full native retrieval and reranking through the executable entry points.
//! Deterministic scorers exercise ordering, source ownership and failure rules;
//! they are not measurements of model relevance, latency or memory.
use super::*;
use frankensearch::{RerankDocument, RerankScore, Reranker};
use std::sync::Mutex;

#[derive(Default)]
struct Scorer {
    calls: AtomicUsize,
    mode: AtomicUsize,
    seen: Mutex<Vec<(String, String)>>,
    clock: Option<Arc<asupersync::time::VirtualClock>>,
}

impl Reranker for Scorer {
    fn id(&self) -> &'static str {
        "native-cli-test-reranker"
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
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.seen.lock().unwrap() = documents
                .iter()
                .map(|doc| (doc.doc_id.clone(), doc.text.clone()))
                .collect();
            let mode = self.mode.swap(0, Ordering::SeqCst);
            if mode == 1 {
                return Err(frankensearch::SearchError::RerankFailed {
                    model: self.id().to_owned(),
                    source: "injected scorer failure".into(),
                });
            }
            if mode == 3 {
                self.clock.as_ref().unwrap().advance(20_000_000);
            }
            Ok(documents
                .iter()
                .enumerate()
                .map(|(original_rank, document)| RerankScore {
                    doc_id: if mode == 2 {
                        documents[0].doc_id.clone()
                    } else {
                        document.doc_id.clone()
                    },
                    original_rank,
                    score: if document.doc_id == "garden.md" {
                        1.0
                    } else {
                        0.0
                    },
                    raw_logit: None,
                })
                .collect())
        })
    }
}

fn reranking(scorer: &Arc<Scorer>) -> query::Policy {
    query::Policy {
        rerank: Some(query::Rerank::new(scorer.clone(), 2).unwrap()),
        ..query::Policy::default()
    }
}

#[test]
fn rerank_options_require_explicit_local_model_and_full_search() {
    let opts = options(&[
        "search",
        "--receipt",
        "r",
        "--query",
        "q",
        "--reranker-dir",
        "model",
        "--rerank-window",
        "25",
    ]);
    assert_eq!(opts.rerank_window, 25);
    assert_eq!(opts.reranker_dir.as_deref(), Some(Path::new("model")));
    assert_eq!(
        options(&["serve", "--receipt", "r", "--reranker-dir", "model"]).rerank_window,
        50
    );
    for args in [
        vec![
            "search",
            "--receipt",
            "r",
            "--query",
            "q",
            "--rerank-window",
            "25",
        ],
        vec![
            "search",
            "--receipt",
            "r",
            "--query",
            "q",
            "--reranker-dir",
            "m",
            "--mode",
            "quality",
        ],
        vec![
            "serve",
            "--receipt",
            "r",
            "--reranker-dir",
            "m",
            "--rerank-window",
            "0",
        ],
        vec![
            "serve",
            "--receipt",
            "r",
            "--reranker-dir",
            "m",
            "--rerank-window",
            "1001",
        ],
        vec![
            "index",
            "--receipt",
            "r",
            "--index-dir",
            "new",
            "--reranker-dir",
            "m",
        ],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
    assert!(query::Rerank::new(Arc::new(Scorer::default()), 0).is_err());
    assert!(query::Rerank::new(Arc::new(Scorer::default()), 1001).is_err());
    #[cfg(not(any(feature = "native", feature = "rerank")))]
    assert!(
        matches!(query::Rerank::load(&Cx::for_testing(), Path::new("absent"), 2, None),
        Err(error) if error.to_string().contains("not compiled"))
    );
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn buffered_reranking_uses_reopened_sources_and_never_scores_outside_scope() {
    run_test_with_cx(|cx| async move {
        for quality in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
            let (models, _, _) = fixture_models(quality);
            let (built, selected) = build(
                &cx,
                &opts,
                &root.path().join("index"),
                source(),
                models.clone(),
            )
            .await
            .unwrap();
            drop(built);
            let index = selected.open(&cx, models).await.unwrap();
            let vector = fs::read(index.vectors().fast().vector_path()).unwrap();
            let scorer = Arc::new(Scorer::default());
            let policy = reranking(&scorer);
            let baseline = search(&index, &cx, "retry network", Mode::Full, 1, [None, None])
                .await
                .unwrap();
            assert_eq!(baseline["results"][0]["doc_id"], "retry.rs");
            let page = query::buffered(
                &index,
                &cx,
                "retry network",
                Mode::Full,
                1,
                [None, None],
                &policy,
            )
            .await
            .unwrap();
            assert_eq!(page["phase"], "reranked");
            assert_eq!(page["results"][0]["doc_id"], "garden.md");
            assert_eq!(page["results"][0]["rerank_score"], 1.0);
            assert_eq!(page["evaluated"], 2);
            assert_eq!(page["rerank_applied"], true);
            assert_eq!(scorer.calls.load(Ordering::SeqCst), 1);
            let seen = scorer.seen.lock().unwrap().clone();
            for (id, text) in seen {
                assert_eq!(text, index.vectors().document(&id).unwrap().content);
            }
            let scope = filter::Filter::parse(r#"{"ids":["retry.rs"]}"#).unwrap();
            let page = query::buffered(
                &index,
                &cx,
                "retry network",
                Mode::Full,
                1,
                [None, Some(&scope)],
                &policy,
            )
            .await
            .unwrap();
            assert_eq!(page["evaluated"], 1);
            assert_eq!(scorer.seen.lock().unwrap()[0].0, "retry.rs");
            assert_eq!(scorer.seen.lock().unwrap().len(), 1);
            let empty = filter::Filter::parse(r#"{"ids":[]}"#).unwrap();
            let count = scorer.calls.load(Ordering::SeqCst);
            let page = query::buffered(
                &index,
                &cx,
                "retry network",
                Mode::Full,
                1,
                [None, Some(&empty)],
                &policy,
            )
            .await
            .unwrap();
            assert_eq!(page["rerank_applied"], false);
            assert_eq!(page["evaluated"], 0);
            assert!(page["results"].as_array().unwrap().is_empty());
            assert_eq!(scorer.calls.load(Ordering::SeqCst), count);
            assert_eq!(
                fs::read(index.vectors().fast().vector_path()).unwrap(),
                vector
            );
        }
    });
}

struct ObserveScorer {
    bytes: Vec<u8>,
    scorer: Arc<Scorer>,
    before: usize,
}
impl Write for ObserveScorer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        let frames = output_frames(&self.bytes);
        let frame = frames.last().unwrap();
        if frame["phase"] == "initial" || frame["phase"] == "refined" {
            assert_eq!(
                self.scorer.calls.load(Ordering::SeqCst),
                self.before,
                "cross-encoder must not start before retrieval pages are flushed"
            );
        }
        if frame["event"] == "terminal" {
            self.before = self.scorer.calls.load(Ordering::SeqCst);
        }
        Ok(())
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn warm_reranking_follows_flushed_retrieval_and_primary_modes_skip_it() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
        let (models, _, _) = fixture_models(true);
        let (index, _) = build(&cx, &opts, &root.path().join("index"), source(), models)
            .await
            .unwrap();
        let live = serve::NativeLiveHybridIndex::new(&cx, index).unwrap();
        let scorer = Arc::new(Scorer::default());
        let policy = reranking(&scorer);
        let mut output = ObserveScorer {
            bytes: Vec::new(),
            scorer: Arc::clone(&scorer),
            before: 0,
        };
        let mut input = Cursor::new(concat!(
            "{\"id\":\"full\",\"query\":\"retry network\"}\n",
            "{\"id\":\"fast\",\"query\":\"retry network\",\"mode\":\"fast\"}\n",
            "{\"id\":\"quality\",\"query\":\"retry network\",\"mode\":\"quality\"}\n",
            "{\"id\":\"again\",\"query\":\"retry network\"}\n",
        ));
        serve::run(
            &live,
            &cx,
            &mut input,
            &mut output,
            (Mode::Full, 1),
            false,
            None,
            &policy,
        )
        .await
        .unwrap();
        assert_eq!(scorer.calls.load(Ordering::SeqCst), 2);
        let frames = output_frames(&output.bytes);
        assert_eq!(frames[0]["full_mode_reranker"], scorer.id());
        for id in ["full", "again"] {
            let request = frames.iter().filter(|f| f["id"] == id).collect::<Vec<_>>();
            assert_eq!(request.len(), 5);
            assert_eq!(request[1]["phase"], "initial");
            assert_eq!(request[2]["phase"], "refined");
            assert_eq!(request[3]["phase"], "reranked");
            assert_eq!(request[3]["results"][0]["doc_id"], "garden.md");
            assert_eq!(request[4]["status"], "complete");
        }
        for id in ["fast", "quality"] {
            assert!(
                !frames
                    .iter()
                    .any(|f| f["id"] == id && f["phase"] == "reranked")
            );
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn failed_or_malformed_reranks_preserve_the_previous_page_and_do_not_poison_serving() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
        let (models, _, _) = fixture_models(true);
        let (index, _) = build(&cx, &opts, &root.path().join("index"), source(), models)
            .await
            .unwrap();
        let scorer = Arc::new(Scorer::default());
        let policy = reranking(&scorer);
        for mode in [1, 2] {
            scorer.mode.store(mode, Ordering::SeqCst);
            assert!(
                query::buffered(
                    &index,
                    &cx,
                    "retry network",
                    Mode::Full,
                    1,
                    [None, None],
                    &policy
                )
                .await
                .is_err()
            );
            scorer.mode.store(mode, Ordering::SeqCst);
            let request: serve::Request =
                serde_json::from_str(r#"{"query":"retry network"}"#).unwrap();
            let mut output = Vec::new();
            assert!(
                !serve::stream_one(
                    &index,
                    &cx,
                    &request,
                    1,
                    (Mode::Full, 1),
                    &mut output,
                    None,
                    &policy
                )
                .await
                .unwrap()
            );
            let frames = output_frames(&output);
            assert_eq!(frames[2]["phase"], "refined");
            assert_eq!(frames[3]["phase"], "rerank_failed");
            assert_eq!(frames[3]["results"], frames[2]["results"]);
            assert_eq!(frames[4]["status"], "degraded");
            assert_eq!(frames[4]["partial_results"], true);
            output.clear();
            assert!(
                serve::stream_one(
                    &index,
                    &cx,
                    &request,
                    2,
                    (Mode::Full, 1),
                    &mut output,
                    None,
                    &policy
                )
                .await
                .unwrap()
            );
            assert_eq!(output_frames(&output)[3]["phase"], "reranked");
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn total_deadline_refuses_a_late_rerank_and_blocks_inference_after_slow_refined_delivery() {
    struct SlowRefined {
        bytes: Vec<u8>,
        clock: Arc<asupersync::time::VirtualClock>,
    }
    impl Write for SlowRefined {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if output_frames(&self.bytes).last().unwrap()["phase"] == "refined" {
                self.clock.advance(10_000_000);
            }
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
        let (models, _, _) = fixture_models(true);
        let (index, _) = build(&cx, &opts, &root.path().join("index"), source(), models)
            .await
            .unwrap();
        let (clock, mut policy) = deadline_policy(10);
        let scorer = Arc::new(Scorer {
            clock: Some(Arc::clone(&clock)),
            ..Scorer::default()
        });
        policy.rerank = Some(query::Rerank::new(scorer.clone(), 2).unwrap());
        let request: serve::Request = serde_json::from_str(r#"{"query":"retry network"}"#).unwrap();
        let mut slow = SlowRefined {
            bytes: Vec::new(),
            clock,
        };
        assert!(
            !serve::stream_one(
                &index,
                &cx,
                &request,
                1,
                (Mode::Full, 1),
                &mut slow,
                None,
                &policy
            )
            .await
            .unwrap()
        );
        assert_eq!(scorer.calls.load(Ordering::SeqCst), 0);
        assert_eq!(output_frames(&slow.bytes)[3]["status"], "timed_out");
        scorer.mode.store(3, Ordering::SeqCst);
        let mut output = Vec::new();
        assert!(
            !serve::stream_one(
                &index,
                &cx,
                &request,
                2,
                (Mode::Full, 1),
                &mut output,
                None,
                &policy
            )
            .await
            .unwrap()
        );
        let frames = output_frames(&output);
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[2]["phase"], "refined");
        assert_eq!(frames[3]["status"], "timed_out");
        assert_eq!(frames[3]["partial_results"], true);
        assert!(!frames.iter().any(|f| f["phase"] == "reranked"));
        output.clear();
        assert!(
            serve::stream_one(
                &index,
                &cx,
                &request,
                3,
                (Mode::Full, 1),
                &mut output,
                None,
                &policy
            )
            .await
            .unwrap()
        );
        assert!(!cx.is_cancel_requested());
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn activation_does_not_retarget_an_inflight_rerank_source_cohort() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
        let (models, _, _) = fixture_models(true);
        let (old, selected) = build(&cx, &opts, &root.path().join("old"), source(), models)
            .await
            .unwrap();
        let edits = update::read_edits(&mut Cursor::new(
            r#"{"op":"upsert","id":"garden.md","content":"garden rewritten successor"}"#,
        ))
        .unwrap();
        let (new, successor) =
            update::apply(&cx, &selected, &old, &root.path().join("new"), edits, 2)
                .await
                .unwrap();
        let receipt = root.path().join("new.json");
        save_selection(&successor, &receipt).unwrap();
        drop(new);
        let live = serve::NativeLiveHybridIndex::new(&cx, old).unwrap();
        let pinned = live.snapshot(&cx).await.unwrap();
        let scorer = Arc::new(Scorer::default());
        let policy = reranking(&scorer);
        let scoped = filter::Query::prepare(pinned.index(), &cx, [None, None]).unwrap();
        let mut stream = scoped
            .progressive_with_reranker(&cx, "retry network", 1, scorer.as_ref(), 2)
            .unwrap();
        assert!(stream.next_phase().await.unwrap().is_some());
        let activation = serde_json::json!({"op":"activate","receipt":receipt,"expected_generation":selected.generation});
        serve::run(
            &live,
            &cx,
            &mut Cursor::new(encode(&activation, MAX_OUTPUT_BYTES).unwrap()),
            &mut Vec::new(),
            (Mode::Full, 1),
            true,
            None,
            &policy,
        )
        .await
        .unwrap();
        while stream.next_phase().await.unwrap().is_some() {}
        assert!(
            scorer
                .seen
                .lock()
                .unwrap()
                .iter()
                .any(|(id, text)| id == "garden.md" && text == "garden flowers in spring")
        );
        assert_eq!(pinned.generation(), selected.generation);
        let current = live.snapshot(&cx).await.unwrap();
        assert_eq!(current.generation(), successor.generation);
        query::buffered(
            current.index(),
            &cx,
            "retry network",
            Mode::Full,
            1,
            [None, None],
            &policy,
        )
        .await
        .unwrap();
        assert!(
            scorer
                .seen
                .lock()
                .unwrap()
                .iter()
                .any(|(id, text)| id == "garden.md" && text == "garden rewritten successor")
        );
    });
}
