use super::*;
use asupersync::time::{TimerDriverHandle, VirtualClock};
use frankensearch::{RerankDocument, RerankScore, Reranker};
use std::future::Future as _;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::task::{Context, Waker};

const TUNED: &str = r#"{"candidate_multiplier":2,"fast_ef":8,"quality_ef":7,"rrf_k":0,"semantic_weight":100,"quality_weight":1}"#;
const GRANTED: &str = r#"{"request_overrides":true,"candidate_multiplier":4,"fast_ef":16,"quality_ef":16,"rrf_k":0,"semantic_weight":100,"quality_weight":1}"#;

struct Model {
    identity: EmbeddingIdentityBundleV1,
    quality: bool,
    calls: AtomicUsize,
    delay_ns: AtomicU64,
    clock: Arc<VirtualClock>,
}

impl Embedder for Model {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.clock.advance(self.delay_ns.load(Ordering::SeqCst));
            Ok(match text {
                "needle" => vec![1.0, 0.0],
                "vector source" if !self.quality => vec![1.0, 0.0],
                "quality source" if self.quality => vec![1.0, 0.0],
                "needle keyword" => vec![0.0, 1.0],
                _ => vec![-1.0, 0.0],
            })
        })
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
        true
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

struct Scorer {
    calls: AtomicUsize,
    drops: AtomicUsize,
    mode: AtomicUsize,
    seen: Mutex<Vec<Vec<(String, String)>>>,
}

struct ScorerDrop<'a>(&'a AtomicUsize);

impl Drop for ScorerDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Reranker for Scorer {
    fn id(&self) -> &str {
        "retained-authority-reranker"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn rerank<'a>(
        &'a self,
        _: &'a Cx,
        _: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _owned_work = ScorerDrop(&self.drops);
            self.seen.lock().unwrap().push(
                documents
                    .iter()
                    .map(|document| (document.doc_id.clone(), document.text.clone()))
                    .collect(),
            );
            let mode = self.mode.swap(0, Ordering::SeqCst);
            if mode == 1 {
                return Err(SearchError::RerankFailed {
                    model: self.id().to_owned(),
                    source: "private-rerank-failure-canary".into(),
                });
            }
            if mode == 2 {
                std::future::pending::<()>().await;
            }
            Ok(documents
                .iter()
                .enumerate()
                .map(|(original_rank, document)| RerankScore {
                    doc_id: if mode == 3 {
                        documents[0].doc_id.clone()
                    } else {
                        document.doc_id.clone()
                    },
                    original_rank,
                    score: if document.doc_id == "vector" {
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

struct Fixture {
    session: AuthoritySession,
    fast: Arc<Model>,
    quality: Arc<Model>,
    scorer: Arc<Scorer>,
    clock: Arc<VirtualClock>,
    timer: TimerDriverHandle,
    receipt: PathBuf,
}

async fn fixture(cx: &Cx, full: bool, ann: bool) -> Fixture {
    let parent = private_parent();
    let clock = Arc::new(VirtualClock::new());
    let make_model = |quality| {
        let mut identity = provider_identity();
        if quality {
            identity.producer.implementation_revision = "authority-query-quality".to_owned();
        }
        identity.validate().unwrap();
        Arc::new(Model {
            identity,
            quality,
            calls: AtomicUsize::new(0),
            delay_ns: AtomicU64::new(0),
            clock: Arc::clone(&clock),
        })
    };
    let fast = make_model(false);
    let quality = make_model(true);
    let retrieval = if ann {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams {
                m: 4,
                m0: 8,
                ef_construction: 24,
                ef_search: 12,
            },
            seed: 91,
        }
    } else {
        NativeBuildRetrieval::Exact
    };
    let root_path = parent.join("root");
    let root = create_root(&root_path).expect("qualified Linux root required");
    let mut builder = NativeIndexBuilder::new(
        parent.join("work"),
        new_generation(1).unwrap(),
        fast.clone(),
    )
    .unwrap()
    .with_fast_storage(NativeBuildPrecision::F32, retrieval)
    .add_documents([
        IndexableDocument::new("vector", "vector source").with_metadata("source_id", "vector"),
        IndexableDocument::new("quality", "quality source").with_metadata("source_id", "quality"),
        IndexableDocument::new("keyword", "needle keyword").with_metadata("source_id", "keyword"),
        IndexableDocument::new("excluded", "needle excluded")
            .with_metadata("source_id", "excluded"),
        IndexableDocument::new("huge", "outside source budget ".repeat(1000)),
    ]);
    if full {
        builder = builder
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, retrieval)
            .unwrap();
    }
    let cut = CommitRange { low: 2, high: 7 };
    let built = if full {
        builder
            .build_progressive_generation(cx, cut, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap()
    } else {
        builder
            .build_exact_generation(cx, EmbedderTierTag::Fast, cut, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap()
    };
    let receipt = parent.join("selected.json");
    let published = stage_and_publish(
        cx,
        &root,
        &root_path,
        &receipt,
        &built,
        None,
        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
    )
    .unwrap();
    let mut session = AuthoritySession::new(
        cx,
        root,
        published.receipt,
        published.snapshot,
        Models {
            fast: fast.clone(),
            quality: full.then(|| quality.clone() as Arc<dyn Embedder>),
        },
        (1, DEFAULT_IMAGE_BYTES),
        false,
    )
    .unwrap()
    .with_filter(
        cx,
        Some(Filter::parse(r#"{"ids":["vector","quality","keyword"]}"#).unwrap()),
        MAX_INPUT_BYTES,
    )
    .unwrap();
    let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
    session.query_policy.timer = Some(timer.clone());
    // This bounds candidate source reads; the preadmitted fixed scope remains
    // unchanged. The unrelated stored "huge" document cannot fit this budget.
    session.max_source_bytes = 512;
    let scorer = Arc::new(Scorer {
        calls: AtomicUsize::new(0),
        drops: AtomicUsize::new(0),
        mode: AtomicUsize::new(0),
        seen: Mutex::new(Vec::new()),
    });
    Fixture {
        session,
        fast,
        quality,
        scorer,
        clock,
        timer,
        receipt,
    }
}

fn install(f: &mut Fixture, grant: bool, rerank: bool) {
    f.session.query_policy.full_query =
        Some(query::FullQuery::parse(if grant { GRANTED } else { TUNED }).unwrap());
    f.session.query_policy.rerank =
        rerank.then(|| query::Rerank::new(f.scorer.clone(), 3).unwrap());
}

fn message(stream: bool) -> serde_json::Value {
    serde_json::json!({"query":"needle","mode":"full","limit":1,"stream":stream})
}

fn parsed(extra: &[&str]) -> Result<Option<Options>> {
    let mut arguments = vec![
        "authority-search",
        "--receipt",
        "selected.json",
        "--query",
        "needle",
    ];
    arguments.extend_from_slice(extra);
    parse(&arguments)
}

#[test]
fn query_control_options_and_request_envelopes_are_explicit_bounded_and_strict() {
    let options = parsed(&[
        "--full-query",
        TUNED,
        "--reranker-dir",
        "local-reranker",
        "--rerank-window",
        "3",
    ])
    .unwrap()
    .unwrap();
    assert_eq!(options.rerank_window, 3);
    assert!(options.full_query.is_some());
    for extra in [
        vec!["--rerank-window", "3"],
        vec!["--reranker-dir", "m", "--rerank-window", "0"],
        vec!["--reranker-dir", "m", "--rerank-window", "1001"],
        vec!["--reranker-dir", "m", "--mode", "fast"],
        vec!["--full-query", "{}", "--mode", "quality"],
        vec!["--full-query", "{}", "--lexical-only"],
        vec!["--reranker-dir", "m", "--mode", "lexical"],
        vec!["--full-query", r#"{"candidate_multiplier":0}"#],
        vec!["--full-query", r#"{"quality_weight":2}"#],
        vec!["--full-query", "{}", "--full-query", "{}"],
    ] {
        assert!(parsed(&extra).is_err(), "{extra:?}");
    }
    for mut arguments in [
        index_arguments(),
        rebuild_arguments(),
        update_arguments(),
        rollback_arguments(),
    ] {
        arguments.extend(["--full-query", TUNED]);
        assert!(parse(&arguments).is_err());
    }
    assert!(
        parse(&[
            "authority-serve",
            "--receipt",
            "selected.json",
            "--full-query",
            GRANTED,
            "--reranker-dir",
            "local-reranker"
        ])
        .is_ok()
    );
    let accepted = r#"{"op":"search","request":{"query":"needle"},"full_query":{"rerank":false}}"#;
    assert!(matches!(
        serde_json::from_str::<AuthorityRequest>(accepted).unwrap(),
        AuthorityRequest::TunedSearch(_)
    ));
    for settings in [
        "null",
        "[]",
        r#"{"candidate_multiplier":1,"candidate_multiplier":2}"#,
        r#"{"request_overrides":true}"#,
        r#"{"timeout_ms":100}"#,
        r#"{"filter":{}}"#,
        r#"{"model_dir":"private-canary"}"#,
    ] {
        let raw =
            format!(r#"{{"op":"search","request":{{"query":"needle"}},"full_query":{settings}}}"#);
        assert!(serde_json::from_str::<AuthorityRequest>(&raw).is_err());
    }
    assert!(
        serde_json::from_str::<AuthorityRequest>(r#"{"query":"needle","full_query":{}}"#).is_err()
    );
    assert!(
        serde_json::from_str::<AuthorityRequest>(r#"{"op":"search","request":{"query":"needle"}}"#)
            .is_err()
    );
}

#[test]
fn tuned_buffered_and_streaming_queries_use_the_same_retained_native_ranking() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let mut f = fixture(&cx, true, ann).await;
            let baseline = exchange(&cx, &mut f.session, &[message(false)]).await;
            install(&mut f, false, false);
            let buffered = exchange(&cx, &mut f.session, &[message(false)]).await;
            let streamed = exchange(&cx, &mut f.session, &[message(true)]).await;
            assert_eq!(buffered[1]["phase"], "refined");
            assert_eq!(buffered[1]["results"][0]["doc_id"], "quality");
            assert_ne!(baseline[1]["results"], buffered[1]["results"]);
            assert_eq!(streamed.len(), 3);
            assert_eq!(streamed[1]["phase"], "initial");
            assert_eq!(streamed[2]["phase"], "refined");
            assert_eq!(streamed[2]["results"], buffered[1]["results"]);
            assert_eq!(streamed[2]["full_query"]["candidate_multiplier"], 2);
            assert_eq!(streamed[2]["full_query"]["quality_ef"], 7);
            assert_eq!(
                streamed[2]["results"][0]["metadata"]["source_id"],
                "quality"
            );
            assert_eq!(f.scorer.calls.load(Ordering::SeqCst), 0);

            // The one-shot loader path also reaches the same configuration.
            let mut options = parsed(&["--full-query", TUNED]).unwrap().unwrap();
            options.receipt = f.receipt.clone();
            options.limit = 1;
            options.filter = f.session.fixed_filter.clone();
            let mut output = Vec::new();
            search_with_loader(&cx, &options, &mut output, |tier| {
                Ok(if tier == EmbedderTierTag::Fast {
                    f.fast.clone() as Arc<dyn Embedder>
                } else {
                    f.quality.clone() as Arc<dyn Embedder>
                })
            })
            .await
            .unwrap();
            assert_eq!(frames(&output)[0]["results"], buffered[1]["results"]);
        }
    });
}

#[test]
fn reranking_reads_only_eligible_retained_sources_and_reports_the_actual_final_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, true, true).await;
        install(&mut f, false, true);
        let retained = Arc::clone(&f.session.snapshot);
        let identity = f.session.receipt.identity.clone();
        for stream in [false, true] {
            let mut request = message(stream);
            request["filter"] = serde_json::json!({"ids":["vector","quality","excluded"]});
            let output = exchange(&cx, &mut f.session, &[request]).await;
            assert_eq!(output.len(), if stream { 4 } else { 2 });
            let final_page = output.last().unwrap();
            assert_eq!(final_page["phase"], "reranked");
            assert_eq!(final_page["results"][0]["doc_id"], "vector");
            assert_eq!(final_page["results"][0]["metadata"]["source_id"], "vector");
            assert_eq!(final_page["reranker"], "retained-authority-reranker");
            assert_eq!(final_page["rerank_window"], 3);
            assert_eq!(final_page["scope"]["eligible_documents"], 2);
            assert!(final_page["results"][0]["rerank_score"].is_number());
            let seen = f.scorer.seen.lock().unwrap();
            let mut documents = seen.last().unwrap().clone();
            documents.sort();
            assert_eq!(
                documents,
                vec![
                    ("quality".to_owned(), "quality source".to_owned()),
                    ("vector".to_owned(), "vector source".to_owned())
                ]
            );
        }
        assert!(Arc::ptr_eq(&retained, &f.session.snapshot));
        assert_eq!(f.session.receipt.identity, identity);
        let before = f.scorer.calls.load(Ordering::SeqCst);
        let mut empty = message(true);
        empty["filter"] = serde_json::json!({"ids":["excluded"]});
        let output = exchange(&cx, &mut f.session, &[empty]).await;
        assert_eq!(output.len(), 3);
        assert_eq!(output[1]["phase"], "initial");
        assert_eq!(output[2]["phase"], "refined");
        assert_eq!(output[2]["results"], serde_json::json!([]));
        assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before);
    });
}

#[test]
fn failed_source_or_reranking_never_becomes_a_successful_buffered_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, true, false).await;
        install(&mut f, false, true);
        for stream in [false, true] {
            for mode in [0, 1, 3] {
                f.session.max_source_bytes = if mode == 0 { 1 } else { 512 };
                f.scorer.mode.store(mode, Ordering::SeqCst);
                let before = f.scorer.calls.load(Ordering::SeqCst);
                let output = exchange(&cx, &mut f.session, &[message(stream)]).await;
                assert_eq!(output.len(), if stream { 4 } else { 2 });
                let failure = output.last().unwrap();
                assert_eq!(failure["event"], "failed");
                assert_eq!(failure["code"], "query_or_activation_refused");
                assert!(failure.get("error").is_none());
                assert!(
                    !serde_json::to_string(&output)
                        .unwrap()
                        .contains("private-rerank-failure-canary")
                );
                assert_eq!(
                    f.scorer.calls.load(Ordering::SeqCst),
                    before + usize::from(mode != 0)
                );
                if stream {
                    assert_eq!(output[1]["phase"], "initial");
                    assert_eq!(output[2]["phase"], "refined");
                }
                f.session.max_source_bytes = 512;
                f.scorer.mode.store(0, Ordering::SeqCst);
                let next = exchange(&cx, &mut f.session, &[message(false)]).await;
                assert_eq!(next[1]["phase"], "reranked");
            }
        }
    });
}

#[test]
fn request_overrides_need_the_server_grant_and_cannot_widen_limits_or_source() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, true, false).await;
        install(&mut f, false, true);
        let envelope = |override_policy: serde_json::Value| {
            serde_json::json!({
                "op":"search","request":{"query":"needle","mode":"full","limit":1},
                "full_query":override_policy,
            })
        };
        let denied = exchange(&cx, &mut f.session, &[envelope(serde_json::json!({}))]).await;
        assert_eq!(denied[1]["code"], "query_tuning_refused");
        install(&mut f, true, true);
        let original =
            serde_json::to_value(f.session.query_policy.full_query.as_ref().unwrap()).unwrap();
        let source = Arc::clone(&f.session.snapshot);
        for value in [
            serde_json::json!({"candidate_multiplier":5}),
            serde_json::json!({"fast_ef":17}),
            serde_json::json!({"quality_ef":17}),
            serde_json::json!({"rerank_window":4}),
            serde_json::json!({"rerank":false,"rerank_window":1}),
        ] {
            let before = (
                f.fast.calls.load(Ordering::SeqCst),
                f.quality.calls.load(Ordering::SeqCst),
                f.scorer.calls.load(Ordering::SeqCst),
            );
            let output = exchange(&cx, &mut f.session, &[envelope(value)]).await;
            assert_eq!(output[1]["code"], "query_tuning_refused");
            assert_eq!(
                (
                    f.fast.calls.load(Ordering::SeqCst),
                    f.quality.calls.load(Ordering::SeqCst),
                    f.scorer.calls.load(Ordering::SeqCst)
                ),
                before
            );
        }
        let before = f.scorer.calls.load(Ordering::SeqCst);
        let mut narrowed = envelope(serde_json::json!({"candidate_multiplier":1,"fast_ef":8,
            "quality_ef":null,"rerank":false}));
        narrowed["request"]["filter"] = serde_json::json!({"ids":["quality","excluded"]});
        let output = exchange(&cx, &mut f.session, &[narrowed, message(false)]).await;
        assert_eq!(output[0]["query_tuning_enabled"], true);
        assert_eq!(output[1]["phase"], "refined");
        assert_eq!(output[1]["results"][0]["doc_id"], "quality");
        assert_eq!(output[1]["scope"]["eligible_documents"], 1);
        assert_eq!(output[1]["full_query"]["candidate_multiplier"], 1);
        assert_eq!(output[1]["full_query"]["quality_ef"], 16);
        assert!(output[1].get("reranker").is_none());
        assert_eq!(output[2]["phase"], "reranked");
        assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before + 1);
        assert_eq!(
            serde_json::to_value(f.session.query_policy.full_query.as_ref().unwrap()).unwrap(),
            original
        );
        assert!(Arc::ptr_eq(&source, &f.session.snapshot));
        let mut primary = envelope(serde_json::json!({}));
        primary["request"]["mode"] = serde_json::json!("fast");
        let output = exchange(&cx, &mut f.session, &[primary]).await;
        assert_eq!(output[1]["code"], "query_tuning_requires_full_mode");
        for mode in ["fast", "quality", "lexical"] {
            let before = f.scorer.calls.load(Ordering::SeqCst);
            let output = exchange(
                &cx,
                &mut f.session,
                &[serde_json::json!({"query":"needle","mode":mode})],
            )
            .await;
            assert_eq!(output[1]["event"], "results");
            assert!(output[1].get("full_query").is_none());
            assert!(output[1].get("reranker").is_none());
            assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before);
        }
    });
}

#[test]
fn full_controls_refuse_incomplete_topology_before_loading_any_provider() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, false).await;
        for extra in [
            vec!["--full-query", TUNED],
            vec!["--reranker-dir", "missing-local-reranker"],
        ] {
            let mut options = parsed(&extra).unwrap().unwrap();
            options.receipt = f.receipt.clone();
            let error = search_with_loader(&cx, &options, &mut Vec::new(), |_| {
                panic!("incomplete topology must refuse before loading models")
            })
            .await
            .unwrap_err();
            assert!(error.to_string().contains("complete full generation"));
        }
    });
}

#[test]
fn rerank_pending_work_obeys_the_original_deadline_even_after_a_tuned_request() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for stream in [false, true] {
            let mut f = fixture(&cx, true, false).await;
            install(&mut f, true, true);
            f.session.query_policy.maximum_ms = Some(10);
            f.fast.delay_ns.store(4_000_000, Ordering::SeqCst);
            f.quality.delay_ns.store(4_000_000, Ordering::SeqCst);
            f.scorer.mode.store(2, Ordering::SeqCst);
            let input = serde_json::json!({"op":"search",
                "request":{"id":"slow","query":"needle","stream":stream,"timeout_ms":100},
                "full_query":{"rerank_window":2}});
            let mut input = Cursor::new(format!("{input}\n"));
            let mut output = Vec::new();
            let mut serving = Box::pin(serve_session(&cx, &mut f.session, &mut input, &mut output));
            assert!(
                serving
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(f.scorer.calls.load(Ordering::SeqCst), 1);
            assert_eq!(f.scorer.drops.load(Ordering::SeqCst), 0);
            assert_eq!(f.timer.pending_count(), 1);
            f.clock.advance(2_000_000);
            let _ = f.timer.process_timers();
            serving.await.unwrap();
            let output = frames(&output);
            assert_eq!(output.len(), if stream { 4 } else { 2 });
            let failure = output.last().unwrap();
            assert_eq!(failure["code"], "search_timeout");
            assert_eq!(failure["budget_ms"], 10);
            assert_eq!(failure["elapsed_ms"], 10);
            assert_eq!(f.scorer.drops.load(Ordering::SeqCst), 1);
            assert!(f.timer.is_empty());
            assert!(!cx.is_cancel_requested());
            f.fast.delay_ns.store(0, Ordering::SeqCst);
            f.quality.delay_ns.store(0, Ordering::SeqCst);
            let next = exchange(&cx, &mut f.session, &[message(false)]).await;
            assert_eq!(next[1]["phase"], "reranked");
        }
    });
}

struct PhaseOutput {
    bytes: Vec<u8>,
    clock: Arc<VirtualClock>,
    phase: &'static str,
    fail: bool,
}

impl Write for PhaseOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let frame: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        if frame["event"] == "results" && frame["phase"] == self.phase {
            if self.fail {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "delivery failed",
                ));
            }
            self.clock.advance(20_000_000);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn rerank_delivery_and_lazy_failure_boundaries_preserve_the_phase_contract() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, true, false).await;
        install(&mut f, false, true);
        f.session.query_policy.maximum_ms = Some(10);
        let request: AuthoritySearchRequest = serde_json::from_value(message(true)).unwrap();
        let mut final_output = PhaseOutput {
            bytes: Vec::new(),
            clock: Arc::clone(&f.clock),
            phase: "reranked",
            fail: false,
        };
        f.session
            .query(&cx, &request, 1, &mut final_output)
            .await
            .unwrap();
        assert_eq!(frames(&final_output.bytes).len(), 3);
        let before = f.scorer.calls.load(Ordering::SeqCst);
        let mut delayed = PhaseOutput {
            bytes: Vec::new(),
            clock: Arc::clone(&f.clock),
            phase: "refined",
            fail: false,
        };
        let error = f
            .session
            .query(&cx, &request, 2, &mut delayed)
            .await
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::SearchTimeout { .. })
        ));
        assert_eq!(frames(&delayed.bytes).len(), 2);
        assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before);
        let mut broken = PhaseOutput {
            bytes: Vec::new(),
            clock: Arc::clone(&f.clock),
            phase: "refined",
            fail: true,
        };
        let error = f
            .session
            .query(&cx, &request, 3, &mut broken)
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<AuthorityServeRefusal>().unwrap().0,
            "result_delivery_failed"
        );
        assert_eq!(frames(&broken.bytes).len(), 1);
        assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before);
    });
}

#[test]
fn caller_pool_reaches_every_authority_query_mode_and_preserves_retained_results() {
    use asupersync::runtime::blocking_pool::BlockingPool;
    use std::time::Duration;

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            for scoped in [false, true] {
                let mut f = fixture(&cx, true, ann).await;
                if !scoped {
                    f.session.scope = None;
                    f.session.fixed_filter = None;
                    // Request filters must first admit the full source cohort;
                    // this fixture's small cap only suits its prepared scope.
                    f.session.max_source_bytes = MAX_INPUT_BYTES;
                }
                let pool = BlockingPool::new(0, 2);
                let pooled_cx = cx.clone().with_blocking_pool_handle(Some(pool.handle()));
                let options = parsed(&[]).unwrap().unwrap();
                let policy = query_controls::load_policy(&pooled_cx, &options).unwrap();
                assert!(policy.query_pool().is_some());
                let mut requests = vec![
                    serde_json::json!({"query":"needle","mode":"fast"}),
                    serde_json::json!({"query":"needle","mode":"quality"}),
                    message(false),
                    message(true),
                    serde_json::json!({"query":"needle","mode":"full","stream":true,
                        "filter":{"ids":["vector","quality"]}}),
                ];
                for tuned in [false, true] {
                    if tuned {
                        install(&mut f, true, true);
                        requests.push(serde_json::json!({"op":"search",
                            "request":{"query":"needle","mode":"full"},
                            "full_query":{"candidate_multiplier":1,"rerank":false}}));
                    }
                    for request in &requests {
                        f.session.query_policy.query_pool = None;
                        let inline = exchange(&cx, &mut f.session, &[request.clone()]).await;
                        f.session.query_policy.query_pool = policy.query_pool();
                        let pooled = exchange(&cx, &mut f.session, &[request.clone()]).await;
                        assert_eq!(inline.len(), pooled.len());
                        for (left, right) in inline.iter().zip(&pooled).skip(1) {
                            assert_eq!(right["event"], "results", "{right}");
                            for field in ["event", "mode", "phase", "results", "scope"] {
                                assert_eq!(left[field], right[field], "{field}: {request}");
                            }
                        }
                    }
                }

                // An unusable supplied pool is a failure, never permission to
                // run a retained exact scan or graph traversal on the executor.
                assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
                for request in &requests {
                    let output = exchange(&cx, &mut f.session, &[request.clone()]).await;
                    assert_eq!(output.len(), 2, "{output:?}");
                    assert_eq!(output[1]["event"], "failed");
                    assert_eq!(output[1]["code"], "query_or_activation_refused");
                }
                let lexical = exchange(
                    &cx,
                    &mut f.session,
                    &[serde_json::json!({"query":"needle","mode":"lexical"})],
                )
                .await;
                assert_eq!(lexical[1]["event"], "results");

                // The one-shot path obtains its pool from the same caller Cx.
                let mut options = options;
                options.receipt = f.receipt.clone();
                options.filter = f.session.fixed_filter.clone();
                let mut output = Vec::new();
                let error = search_with_loader(&pooled_cx, &options, &mut output, |tier| {
                    Ok(if tier == EmbedderTierTag::Fast {
                        f.fast.clone() as Arc<dyn Embedder>
                    } else {
                        f.quality.clone() as Arc<dyn Embedder>
                    })
                })
                .await
                .unwrap_err();
                assert!(matches!(
                    error.downcast_ref::<SearchError>(),
                    Some(SearchError::SubsystemError {
                        subsystem: "native_ann.query_worker",
                        ..
                    })
                ));
                assert!(output.is_empty());
            }
        }
    });
}

#[test]
fn authority_deadline_expires_while_vector_work_is_queued_and_session_stays_usable() {
    use asupersync::runtime::blocking_pool::BlockingPool;
    use std::sync::mpsc;
    use std::time::Duration;

    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, true, true).await;
        let pool = BlockingPool::new(0, 1);
        let (started, entered) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let blocker = pool.spawn(move || {
            started.send(()).unwrap();
            gate.recv_timeout(Duration::from_secs(10)).unwrap();
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        f.session.query_policy.query_pool = Some(pool.handle());
        f.session.query_policy.maximum_ms = Some(10);
        let before_quality = f.quality.calls.load(Ordering::SeqCst);
        let request: AuthoritySearchRequest = serde_json::from_value(message(true)).unwrap();
        let mut output = Vec::new();
        let mut pending = Box::pin(f.session.query(&cx, &request, 1, &mut output));
        assert!(
            pending
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert_eq!(f.quality.calls.load(Ordering::SeqCst), before_quality);
        f.clock.advance(10_000_000);
        let _ = f.timer.process_timers();
        let error = pending.await.unwrap_err();
        assert!(matches!(
            error.downcast_ref::<SearchError>(),
            Some(SearchError::SearchTimeout { .. })
        ));
        assert!(output.is_empty());
        assert!(!cx.is_cancel_requested());
        assert!(f.timer.is_empty());
        assert_eq!(f.quality.calls.load(Ordering::SeqCst), before_quality);

        release.send(()).unwrap();
        assert!(blocker.wait_timeout(Duration::from_secs(10)));
        let next = exchange(&cx, &mut f.session, &[message(true)]).await;
        assert_eq!(next.len(), 3);
        assert_eq!(next[1]["phase"], "initial");
        assert_eq!(next[2]["phase"], "refined");
        assert_eq!(f.quality.calls.load(Ordering::SeqCst), before_quality + 1);
        assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
    });
}
