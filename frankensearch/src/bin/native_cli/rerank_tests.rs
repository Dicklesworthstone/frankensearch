//! Selected-tier native retrieval and reranking through executable entry points.
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

// Synthetic semantic-shaped producers exercise score provenance, not real-model
// qualification. A changed response identity must be refused before inference.
struct PrimaryProbe {
    inner: FixtureModel,
    changed: EmbeddingIdentityBundleV1,
    substitute: std::sync::atomic::AtomicBool,
}

impl PrimaryProbe {
    fn new(name: &str) -> Arc<Self> {
        use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 3);
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
        let mut changed = identity.clone();
        changed.producer.backend = "substituted-test-producer".to_owned();
        changed.validate().unwrap();
        Arc::new(Self {
            inner: FixtureModel {
                identity,
                category: ModelCategory::TransformerEmbedder,
                calls: Arc::new(AtomicUsize::new(0)),
            },
            changed,
            substitute: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

impl Embedder for PrimaryProbe {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.substitute.load(Ordering::SeqCst) {
            &self.changed
        } else {
            &self.inner.identity
        })
    }
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.inner.embed(cx, text)
    }
    fn id(&self) -> &str {
        self.inner.id()
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn primary_reranking_retains_persisted_tier_rows_scope_and_strict_noop_admission() {
    use frankensearch_core::generation::EmbedderTierTag;
    use frankensearch_index::ValidatedFsviBytes;
    use std::time::Duration;
    run_test_with_cx(|cx| async move {
        let pool = asupersync::runtime::blocking_pool::BlockingPool::new(0, 2);
        for exact in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
            opts.exact = exact;
            let fast = PrimaryProbe::new("primary-fast");
            let quality = PrimaryProbe::new("primary-quality");
            let models = Models {
                fast: fast.clone(),
                quality: Some(quality.clone()),
            };
            let (built, selection) = build(
                &cx,
                &opts,
                &root.path().join("index"),
                source(),
                models.clone(),
            )
            .await
            .unwrap();
            drop(built);
            let index = selection.open(&cx, models).await.unwrap();
            let scorer = Arc::new(Scorer::default());
            let scope = filter::Filter::parse(r#"{"ids":["retry.rs"]}"#).unwrap();
            let empty = filter::Filter::parse(r#"{"ids":[]}"#).unwrap();
            for use_pool in [false, true] {
                let mut policy = reranking(&scorer);
                policy.query_pool = use_pool.then(|| pool.handle());
                for (mode, selected, other) in [
                    (Mode::Fast, &fast, &quality),
                    (Mode::Quality, &quality, &fast),
                ] {
                    let before = (
                        selected.inner.calls.load(Ordering::SeqCst),
                        other.inner.calls.load(Ordering::SeqCst),
                    );
                    let page = query::buffered(
                        &index,
                        &cx,
                        "retry network",
                        mode,
                        1,
                        [None, None],
                        &policy,
                    )
                    .await
                    .unwrap();
                    assert_eq!(page["phase"], "reranked");
                    assert_eq!(page["results"][0]["doc_id"], "garden.md");
                    assert_eq!(page["evaluated"], 2);
                    assert_eq!(selected.inner.calls.load(Ordering::SeqCst), before.0 + 1);
                    assert_eq!(other.inner.calls.load(Ordering::SeqCst), before.1);
                    let hit = &page["results"][0];
                    let (tier, score, absent) = if mode == Mode::Quality {
                        (
                            index.vectors().quality().unwrap(),
                            "quality_score",
                            "fast_score",
                        )
                    } else {
                        (index.vectors().fast(), "fast_score", "quality_score")
                    };
                    assert!(hit[score].is_number());
                    assert!(hit[absent].is_null());
                    let owner = ValidatedFsviBytes::from_arc(
                        fs::read(tier.vector_path()).unwrap().into(),
                        tier.binding(),
                    )
                    .unwrap();
                    assert_eq!(
                        owner
                            .doc_id_at(usize::try_from(hit["index"].as_u64().unwrap()).unwrap())
                            .unwrap(),
                        "garden.md"
                    );
                    for (id, text) in scorer.seen.lock().unwrap().iter() {
                        assert_eq!(*text, index.vectors().document(id).unwrap().content);
                    }
                    let page = query::buffered(
                        &index,
                        &cx,
                        "retry network",
                        mode,
                        1,
                        [Some(&scope), None],
                        &policy,
                    )
                    .await
                    .unwrap();
                    assert_eq!(page["evaluated"], 1);
                    assert_eq!(
                        scorer.seen.lock().unwrap().as_slice(),
                        &[("retry.rs".to_owned(), source()[0].content.clone())]
                    );
                    let before = (
                        selected.inner.calls.load(Ordering::SeqCst),
                        scorer.calls.load(Ordering::SeqCst),
                    );
                    for (text, limit, filter) in [
                        ("retry", 0, None),
                        ("retry", 1, Some(&empty)),
                        ("-retry", 2, None),
                    ] {
                        let page = query::buffered(
                            &index,
                            &cx,
                            text,
                            mode,
                            limit,
                            [filter, None],
                            &policy,
                        )
                        .await
                        .unwrap();
                        assert_eq!(page["rerank_applied"], false);
                        assert_eq!(
                            page["phase"],
                            if mode == Mode::Quality {
                                "quality"
                            } else {
                                "initial"
                            }
                        );
                    }
                    assert_eq!(selected.inner.calls.load(Ordering::SeqCst), before.0);
                    assert_eq!(scorer.calls.load(Ordering::SeqCst), before.1);
                    selected.substitute.store(true, Ordering::SeqCst);
                    for (limit, filter) in [(0, None), (1, Some(&empty))] {
                        assert!(
                            query::buffered(
                                &index,
                                &cx,
                                "retry",
                                mode,
                                limit,
                                [filter, None],
                                &policy
                            )
                            .await
                            .is_err()
                        );
                    }
                    assert_eq!(selected.inner.calls.load(Ordering::SeqCst), before.0);
                    assert_eq!(scorer.calls.load(Ordering::SeqCst), before.1);
                    selected.substitute.store(false, Ordering::SeqCst);
                    let tier = if mode == Mode::Quality {
                        EmbedderTierTag::Quality
                    } else {
                        EmbedderTierTag::Fast
                    };
                    let mut lazy = index
                        .progressive_tier_with_reranker(
                            &cx,
                            "retry network",
                            1,
                            tier,
                            scorer.as_ref(),
                            2,
                            policy.query_pool(),
                        )
                        .unwrap();
                    assert!(matches!(
                        lazy.next_phase().await.unwrap(),
                        Some(frankensearch::native_ann::NativeSearchPhase::Initial { .. })
                    ));
                    drop(lazy);
                    assert_eq!(scorer.calls.load(Ordering::SeqCst), before.1);
                }
            }
        }
        assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
    });
}

#[test]
fn rerank_options_require_explicit_local_model_and_accept_every_query_tier() {
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
    for mode in ["fast", "quality", "full"] {
        let parsed = options(&[
            "search",
            "--receipt",
            "r",
            "--query",
            "q",
            "--reranker-dir",
            "m",
            "--mode",
            mode,
        ]);
        assert_eq!(parsed.mode, Mode::parse(mode).unwrap());
        assert!(parsed.reranker_dir.is_some());
    }
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
        if ["initial", "quality", "refined"]
            .iter()
            .any(|phase| frame["phase"] == *phase)
        {
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
fn warm_reranking_follows_flushed_retrieval_for_every_selected_mode() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
        let (models, fast_calls, quality_calls) = fixture_models(true);
        let (index, _) = build(&cx, &opts, &root.path().join("index"), source(), models)
            .await
            .unwrap();
        let live = serve::NativeLiveHybridIndex::new(&cx, index).unwrap();
        let before = (
            fast_calls.load(Ordering::SeqCst),
            quality_calls.load(Ordering::SeqCst),
        );
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
        assert_eq!(scorer.calls.load(Ordering::SeqCst), 4);
        assert_eq!(fast_calls.load(Ordering::SeqCst), before.0 + 3);
        assert_eq!(quality_calls.load(Ordering::SeqCst), before.1 + 3);
        let frames = output_frames(&output.bytes);
        assert_eq!(frames[0]["full_mode_reranker"], scorer.id());
        assert_eq!(
            frames[0]["rerank_modes"],
            serde_json::json!(["fast", "quality", "full"])
        );
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
            let request = frames.iter().filter(|f| f["id"] == id).collect::<Vec<_>>();
            assert_eq!(request.len(), 4);
            assert_eq!(
                request[1]["phase"],
                if id == "quality" {
                    "quality"
                } else {
                    "initial"
                }
            );
            assert_eq!(request[2]["phase"], "reranked");
            assert_eq!(request[2]["results"][0]["doc_id"], "garden.md");
            assert_eq!(request[3]["status"], "complete");
            let other = if id == "quality" { "fast" } else { "quality" };
            assert_eq!(request[1]["candidates"][other], 0);
            assert!(request[1]["candidates"][id].as_u64().unwrap() > 0);
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
        for (query_mode, last_retrieval) in [(Mode::Fast, 1), (Mode::Quality, 1), (Mode::Full, 2)] {
            for mode in [1, 2] {
                scorer.mode.store(mode, Ordering::SeqCst);
                assert!(
                    query::buffered(
                        &index,
                        &cx,
                        "retry network",
                        query_mode,
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
                        (query_mode, 1),
                        &mut output,
                        None,
                        &policy
                    )
                    .await
                    .unwrap()
                );
                let frames = output_frames(&output);
                assert_eq!(frames[last_retrieval + 1]["phase"], "rerank_failed");
                assert_eq!(
                    frames[last_retrieval + 1]["results"],
                    frames[last_retrieval]["results"]
                );
                assert_eq!(frames[last_retrieval + 2]["status"], "degraded");
                assert_eq!(frames[last_retrieval + 2]["partial_results"], true);
                output.clear();
                assert!(
                    serve::stream_one(
                        &index,
                        &cx,
                        &request,
                        2,
                        (query_mode, 1),
                        &mut output,
                        None,
                        &policy
                    )
                    .await
                    .unwrap()
                );
                assert_eq!(
                    output_frames(&output)[last_retrieval + 1]["phase"],
                    "reranked"
                );
            }
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn total_deadline_and_delivery_boundaries_hold_for_each_tier_and_layout() {
    struct SlowRefined {
        bytes: Vec<u8>,
        clock: Arc<asupersync::time::VirtualClock>,
        phase: &'static str,
        fail: bool,
    }
    impl Write for SlowRefined {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            if output_frames(&self.bytes).last().unwrap()["phase"] == self.phase {
                if self.fail {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "stopped output"));
                }
                self.clock.advance(10_000_000);
            }
            Ok(())
        }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--receipt", "r", "--index-dir", "new"]);
        let (models, _, _) = fixture_models(true);
        let (single, _) = build(
            &cx,
            &opts,
            &root.path().join("index"),
            source(),
            models.clone(),
        )
        .await
        .unwrap();
        let (shards, _) =
            crate::sharded::build(&cx, &opts, &root.path().join("shards"), source(), models, 1)
                .await
                .unwrap();
        for index in [
            crate::cohort::Index::Single(&single),
            crate::cohort::Index::Sharded(&shards),
        ] {
            for (mode, last_phase, last_retrieval) in [
                (Mode::Fast, "initial", 1),
                (Mode::Quality, "quality", 1),
                (Mode::Full, "refined", 2),
            ] {
                let (clock, mut policy) = deadline_policy(10);
                let scorer = Arc::new(Scorer {
                    clock: Some(Arc::clone(&clock)),
                    ..Scorer::default()
                });
                policy.rerank = Some(query::Rerank::new(scorer.clone(), 2).unwrap());
                let request: serve::Request =
                    serde_json::from_str(r#"{"query":"retry network"}"#).unwrap();
                let mut slow = SlowRefined {
                    bytes: Vec::new(),
                    clock: Arc::clone(&clock),
                    phase: last_phase,
                    fail: false,
                };
                assert!(
                    !serve::stream_one(
                        index,
                        &cx,
                        &request,
                        1,
                        (mode, 1),
                        &mut slow,
                        None,
                        &policy
                    )
                    .await
                    .unwrap()
                );
                assert_eq!(scorer.calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    output_frames(&slow.bytes)[last_retrieval + 1]["status"],
                    "timed_out"
                );
                let mut failed = SlowRefined {
                    bytes: Vec::new(),
                    clock: Arc::clone(&clock),
                    phase: last_phase,
                    fail: true,
                };
                assert!(
                    serve::stream_one(
                        index,
                        &cx,
                        &request,
                        2,
                        (mode, 1),
                        &mut failed,
                        None,
                        &policy
                    )
                    .await
                    .is_err()
                );
                assert_eq!(
                    scorer.calls.load(Ordering::SeqCst),
                    0,
                    "failed retrieval delivery must abandon the lazy scorer"
                );
                assert!(
                    !output_frames(&failed.bytes)
                        .iter()
                        .any(|frame| frame["event"] == "terminal")
                );
                scorer.mode.store(3, Ordering::SeqCst);
                let mut output = Vec::new();
                assert!(
                    !serve::stream_one(
                        index,
                        &cx,
                        &request,
                        2,
                        (mode, 1),
                        &mut output,
                        None,
                        &policy
                    )
                    .await
                    .unwrap()
                );
                let frames = output_frames(&output);
                assert_eq!(frames.len(), last_retrieval + 2);
                assert_eq!(frames[last_retrieval]["phase"], last_phase);
                assert_eq!(frames[last_retrieval + 1]["status"], "timed_out");
                assert_eq!(frames[last_retrieval + 1]["partial_results"], true);
                assert!(!frames.iter().any(|f| f["phase"] == "reranked"));
                let mut delivered = SlowRefined {
                    bytes: Vec::new(),
                    clock,
                    phase: "reranked",
                    fail: false,
                };
                assert!(
                    serve::stream_one(
                        index,
                        &cx,
                        &request,
                        3,
                        (mode, 1),
                        &mut delivered,
                        None,
                        &policy
                    )
                    .await
                    .unwrap()
                );
                assert_eq!(
                    output_frames(&delivered.bytes).last().unwrap()["status"],
                    "complete",
                    "delivered final output cannot retroactively time out"
                );
                assert!(!cx.is_cancel_requested());
            }
        }
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
