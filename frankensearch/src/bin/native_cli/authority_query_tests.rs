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
    queries: Mutex<Vec<String>>,
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
            self.queries.lock().unwrap().push(text.to_owned());
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
    queries: Mutex<Vec<String>>,
    seen: Mutex<Vec<Vec<(String, String)>>>,
}

struct ScorerDrop<'a>(&'a AtomicUsize);

impl Drop for ScorerDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Reranker for Scorer {
    fn id(&self) -> &'static str {
        "retained-authority-reranker"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn rerank<'a>(
        &'a self,
        _: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.queries.lock().unwrap().push(query.to_owned());
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
    fixture_topology(
        cx,
        if full {
            NativeGenerationTopology::FullProgressive
        } else {
            NativeGenerationTopology::FastOnly
        },
        ann,
    )
    .await
}

async fn fixture_topology(cx: &Cx, topology: NativeGenerationTopology, ann: bool) -> Fixture {
    let parent = private_parent();
    let full = topology == NativeGenerationTopology::FullProgressive;
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
            queries: Mutex::new(Vec::new()),
            delay_ns: AtomicU64::new(0),
            clock: Arc::clone(&clock),
        })
    };
    let fast = make_model(false);
    let quality = make_model(true);
    let primary = if topology == NativeGenerationTopology::QualityOnly {
        quality.clone()
    } else {
        fast.clone()
    };
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
        primary.clone(),
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
            .build_exact_generation(
                cx,
                if topology == NativeGenerationTopology::QualityOnly {
                    EmbedderTierTag::Quality
                } else {
                    EmbedderTierTag::Fast
                },
                cut,
                DEFAULT_IMAGE_BYTES,
            )
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
            fast: primary,
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
        queries: Mutex::new(Vec::new()),
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

#[test]
fn exact_id_scope_refills_global_lexical_ranks_without_exporting_unrelated_bodies() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, false).await;
        let snapshot = &f.session.snapshot;
        let global = snapshot
            .search_lexical_text(&cx, "needle", 10)
            .await
            .unwrap();
        assert!(global.len() >= 2);
        let winner = global.last().unwrap();
        let id = winner.doc_id.as_str();
        let filter = serde_json::json!({"ids":[id,"missing"],"metadata":{"source_id":id}});
        let selected = Filter::parse(&filter.to_string()).unwrap();
        assert!(snapshot.source_documents(&cx, 512).is_err());
        let mut output = Vec::new();
        let mut one_shot = parsed(&["--mode", "lexical"]).unwrap().unwrap();
        one_shot.receipt = f.receipt.clone();
        one_shot.limit = 1;
        one_shot.max_source_bytes = 512;
        one_shot.filter = Some(selected.clone());
        search_with_loader(&cx, &one_shot, &mut output, |_| {
            panic!("lexical query loaded a model")
        })
        .await
        .unwrap();
        let result = &frames(&output)[0];
        assert_eq!(result["scope"]["eligible_documents"], 1);
        assert_eq!(result["results"][0]["doc_id"], id);
        let hit: ScoredResult = serde_json::from_value(result["results"][0].clone()).unwrap();
        assert_eq!(hit.score.to_bits(), winner.score.to_bits());

        let mut session = f.session.with_filter(&cx, Some(selected), 512).unwrap();
        let messages = [
            serde_json::json!({"query":"needle","mode":"full","limit":1}),
            serde_json::json!({"query":"needle","mode":"lexical","filter":{"ids":["huge"]}}),
        ];
        let response = exchange(&cx, &mut session, &messages).await;
        assert_eq!(response[1]["results"][0]["doc_id"], id);
        assert_eq!(response[2]["scope"]["eligible_documents"], 0);
        assert_eq!(response[2]["results"], serde_json::json!([]));
        let missing = Filter::parse(r#"{"ids":["absent"]}"#).unwrap();
        assert!(
            prepare_scope(&cx, &session.snapshot, Some(&missing), 0)
                .unwrap()
                .unwrap()
                .is_empty()
        );
        let huge = Filter::parse(r#"{"ids":["huge"]}"#).unwrap();
        assert!(prepare_scope(&cx, &session.snapshot, Some(&huge), 512).is_err());
    });
}

#[test]
fn literal_prefix_scopes_read_only_live_matching_source_before_querying() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, true, false).await;
        for (filter, expected) in [
            (r#"{"id_prefix":"key"}"#, Some("keyword")),
            (
                r#"{"ids":["keyword","huge"],"id_prefix":"key","metadata":{"source_id":"keyword"}}"#,
                Some("keyword"),
            ),
            (
                r#"{"ids":["keyword","huge"],"id_prefix":"key","metadata":{"source_id":"huge"}}"#,
                None,
            ),
            (r#"{"id_prefix":"Key"}"#, None),
        ] {
            let mut options = parsed(&[
                "--mode",
                "lexical",
                "--filter",
                filter,
                "--max-source-bytes",
                "512",
            ])
            .unwrap()
            .unwrap();
            options.receipt = f.receipt.clone();
            let mut output = Vec::new();
            search_with_loader(&cx, &options, &mut output, |_| {
                panic!("lexical search loaded a model")
            })
            .await
            .unwrap();
            let page = &frames(&output)[0];
            let hits = page["results"].as_array().unwrap();
            assert_eq!(
                page["scope"]["eligible_documents"],
                usize::from(expected.is_some()),
                "{filter}"
            );
            assert_eq!(hits.len(), usize::from(expected.is_some()), "{filter}");
            if let Some(id) = expected {
                assert_eq!(hits[0]["doc_id"], id);
            }
        }
        let missing = Filter::parse(r#"{"id_prefix":"Key"}"#).unwrap();
        assert!(
            prepare_scope(&cx, &f.session.snapshot, Some(&missing), 0)
                .unwrap()
                .unwrap()
                .is_empty()
        );
        for filter in [
            r#"{"id_prefix":"hu"}"#,
            r#"{"metadata":{"source_id":"keyword"}}"#,
        ] {
            let filter = Filter::parse(filter).unwrap();
            assert!(prepare_scope(&cx, &f.session.snapshot, Some(&filter), 512).is_err());
        }
    });
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

/// Observe actual flushes independently of the writer borrowed by the command.
#[derive(Default)]
struct PhaseOutput {
    pending: Vec<u8>,
    flushed: Arc<Mutex<Vec<serde_json::Value>>>,
    fail_flush: bool,
}

impl io::Write for PhaseOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.fail_flush {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"));
        }
        self.flushed
            .lock()
            .unwrap()
            .push(serde_json::from_slice(&self.pending)?);
        self.pending.clear();
        Ok(())
    }
}

#[test]
fn one_shot_streams_load_quality_only_after_a_successful_initial_flush() {
    let clock = Arc::new(VirtualClock::new());
    let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
    let runtime = RuntimeBuilder::current_thread()
        .with_timer_driver(timer)
        .build()
        .unwrap();
    runtime.block_on(async {
        let cx = Cx::current().unwrap().with_blocking_pool_handle(None);
        for mode in ["full", "partial"] {
            let parent = private_parent();
            let fast = provider(false);
            let quality = provider(true);
            let receipt = if mode == "partial" {
                publish_partial_cli(&cx, &parent, &fast, &quality, false)
                    .await
                    .0
                    .receipt
            } else {
                publish_full(
                    &cx,
                    &parent,
                    &fast,
                    &quality,
                    vec![
                        IndexableDocument::new("fast-only", "retry fallback"),
                        IndexableDocument::new("quality", "retry network"),
                    ],
                    [NativeBuildRetrieval::Exact; 2],
                )
                .await;
                parent.join("initial.json")
            };
            for (filter, failure) in [
                (None, "success"),
                (Some(r#"{"ids":["quality"]}"#), "success"),
                (Some(r#"{"ids":[]}"#), "success"),
                (Some(r#"{"ids":[]}"#), "identity"),
                (None, "load"),
                (None, "identity"),
                (None, "timeout"),
                (None, "transport"),
            ] {
                let mut options = parsed(&["--mode", mode, "--stream", "--full-query", TUNED])
                    .unwrap()
                    .unwrap();
                options.receipt = receipt.clone();
                options.query = Some("retry".to_owned());
                options.filter = filter.map(|text| Filter::parse(text).unwrap());
                let empty = filter == Some(r#"{"ids":[]}"#);
                let mut wire = PhaseOutput {
                    fail_flush: failure == "transport",
                    ..Default::default()
                };
                let flushed = Arc::clone(&wire.flushed);
                options.timeout_ms = Some(1000);
                let before = calls(&fast, &quality);
                let mut quality_loads = 0;
                let result = search_with_loader(&cx, &options, &mut wire, |tier| {
                    if tier == EmbedderTierTag::Fast {
                        assert!(flushed.lock().unwrap().is_empty());
                        return Ok(fast.clone() as Arc<dyn Embedder>);
                    }
                    let initial = flushed.lock().unwrap();
                    assert_eq!(initial.len(), 1);
                    assert_eq!(initial[0]["phase"], "initial");
                    assert_eq!(initial[0]["results"].as_array().unwrap().is_empty(), empty);
                    assert_eq!(
                        calls(&fast, &quality),
                        (before.0 + usize::from(!empty), before.1)
                    );
                    quality_loads += 1;
                    match failure {
                        "identity" => Ok(fast.clone() as Arc<dyn Embedder>),
                        "load" => Err(bad("quality unavailable")),
                        "timeout" => {
                            clock.advance(1_000_000_001);
                            Ok(quality.clone() as Arc<dyn Embedder>)
                        }
                        _ => Ok(quality.clone() as Arc<dyn Embedder>),
                    }
                })
                .await;
                let pages = flushed.lock().unwrap();
                assert_eq!(quality_loads, usize::from(failure != "transport"));
                assert_eq!(
                    result.is_ok(),
                    failure == "success",
                    "{mode} {failure}: {result:?}"
                );
                if failure == "success" {
                    assert_eq!(pages.len(), 2);
                    assert_eq!(pages[1]["phase"], "refined");
                    assert_eq!(pages[0]["authority"], pages[1]["authority"]);
                    assert_eq!(pages[1]["results"].as_array().unwrap().is_empty(), empty);
                } else {
                    assert_eq!(pages.len(), usize::from(failure != "transport"));
                    if failure == "timeout" {
                        assert!(matches!(
                            result.unwrap_err().downcast_ref::<SearchError>(),
                            Some(SearchError::SearchTimeout { .. })
                        ));
                    }
                }
                assert_eq!(
                    calls(&fast, &quality),
                    (
                        before.0 + usize::from(!empty),
                        before.1 + usize::from(!empty && failure == "success")
                    )
                );
            }
        }
    });
}

const EXCLUSIONS: &str =
    r#"not "vector source" nOt "quality source" -excluded -"outside source budget""#;

fn exclusion_phases(mode: &str, stream: bool) -> &'static [&'static str] {
    match (mode, stream) {
        ("full", true) => &["initial", "refined"],
        ("full", false) => &["refined"],
        ("quality", _) => &["quality"],
        ("lexical", _) => &["lexical"],
        ("fast", _) => &["initial"],
        _ => panic!("unexpected exclusion test mode"),
    }
}

fn assert_keyword_exclusion_results(output: &[serde_json::Value], phases: &[&str]) {
    assert_eq!(output[0]["event"], "ready");
    assert_eq!(output.len(), phases.len() + 1, "{output:?}");
    for (frame, phase) in output[1..].iter().zip(phases) {
        assert_eq!(frame["event"], "results", "{frame:?}");
        assert_eq!(frame["phase"], *phase);
        let results = frame["results"].as_array().unwrap();
        assert_eq!(results.len(), 1, "{frame:?}");
        assert_eq!(results[0]["doc_id"], "keyword", "{frame:?}");
        assert_eq!(results[0]["metadata"]["source_id"], "keyword");
    }
}

#[test]
fn authority_exclusions_filter_semantic_winners_and_embed_only_positive_text() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            for scoped in [false, true] {
                let mut f = fixture(&cx, true, ann).await;
                if !scoped {
                    f.session.scope = None;
                    f.session.fixed_filter = None;
                }
                let retained = Arc::clone(&f.session.snapshot);
                let identity = f.session.receipt.identity.clone();
                let text = format!("needle {EXCLUSIONS}");
                for mode in ["fast", "quality", "full", "lexical"] {
                    for stream in [false, true] {
                        f.fast.queries.lock().unwrap().clear();
                        f.quality.queries.lock().unwrap().clear();
                        let output = exchange(
                            &cx,
                            &mut f.session,
                            &[serde_json::json!({
                                "query": text, "mode": mode, "limit": 1, "stream": stream,
                            })],
                        )
                        .await;
                        assert_keyword_exclusion_results(&output, exclusion_phases(mode, stream));
                        for (model, expected) in [
                            (&f.fast, usize::from(matches!(mode, "fast" | "full"))),
                            (&f.quality, usize::from(matches!(mode, "quality" | "full"))),
                        ] {
                            let queries = model.queries.lock().unwrap();
                            assert_eq!(queries.len(), expected, "{mode}/{stream}/{scoped}");
                            assert!(queries.iter().all(|query| query == "needle"), "{queries:?}");
                            drop(queries);
                        }
                        if scoped {
                            assert_eq!(output.last().unwrap()["scope"]["eligible_documents"], 3);
                        }
                    }
                }
                assert!(Arc::ptr_eq(&retained, &f.session.snapshot));
                assert_eq!(f.session.receipt.identity, identity);
            }
        }
    });
}

#[test]
fn negative_only_authority_queries_skip_both_models_and_configured_reranking() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            for scoped in [false, true] {
                let mut f = fixture(&cx, true, ann).await;
                if !scoped {
                    f.session.scope = None;
                    f.session.fixed_filter = None;
                }
                install(&mut f, false, true);
                let before = [
                    f.fast.calls.load(Ordering::SeqCst),
                    f.quality.calls.load(Ordering::SeqCst),
                    f.scorer.calls.load(Ordering::SeqCst),
                ];
                for mode in ["fast", "quality", "full", "lexical"] {
                    for stream in [false, true] {
                        let output = exchange(
                            &cx,
                            &mut f.session,
                            &[serde_json::json!({
                                "query": EXCLUSIONS, "mode": mode, "limit": 1, "stream": stream,
                            })],
                        )
                        .await;
                        assert_keyword_exclusion_results(&output, exclusion_phases(mode, stream));
                        for frame in &output[1..] {
                            let result = &frame["results"][0];
                            assert_eq!(result["source"], "Lexical");
                            assert!(result["index"].is_null());
                            assert!(result["fast_score"].is_null());
                            assert!(result["quality_score"].is_null());
                            assert!(result["rerank_score"].is_null());
                        }
                    }
                }
                assert_eq!(f.fast.calls.load(Ordering::SeqCst), before[0]);
                assert_eq!(f.quality.calls.load(Ordering::SeqCst), before[1]);
                assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before[2]);
            }
        }
    });
}

#[test]
fn authority_exclusions_and_request_scope_constrain_retained_rerank_sources() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, true, true).await;
        install(&mut f, false, true);
        let retained = Arc::clone(&f.session.snapshot);
        let identity = f.session.receipt.identity.clone();
        for stream in [false, true] {
            let output = exchange(
                &cx,
                &mut f.session,
                &[serde_json::json!({
                    "query": r#"needle not "vector source" nOt "quality source""#,
                    "mode": "full", "limit": 1, "stream": stream,
                    "filter": {"ids": ["keyword", "excluded"]},
                })],
            )
            .await;
            assert_keyword_exclusion_results(
                &output,
                if stream {
                    &["initial", "refined", "reranked"]
                } else {
                    &["reranked"]
                },
            );
            assert_eq!(output.last().unwrap()["scope"]["eligible_documents"], 1);
            assert_eq!(
                f.scorer.seen.lock().unwrap().last().unwrap(),
                &vec![("keyword".to_owned(), "needle keyword".to_owned())],
            );
        }
        assert_eq!(*f.scorer.queries.lock().unwrap(), ["needle", "needle"]);
        assert!(Arc::ptr_eq(&retained, &f.session.snapshot));
        assert_eq!(f.session.receipt.identity, identity);
    });
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
        vec!["--full-query", "{}", "--mode", "quality"],
        vec!["--full-query", "{}", "--lexical-only"],
        vec!["--reranker-dir", "m", "--mode", "lexical"],
        vec!["--full-query", r#"{"candidate_multiplier":0}"#],
        vec!["--full-query", r#"{"quality_weight":2}"#],
        vec!["--full-query", "{}", "--full-query", "{}"],
    ] {
        assert!(parsed(&extra).is_err(), "{extra:?}");
    }
    for mode in ["fast", "quality"] {
        let selected = parsed(&["--reranker-dir", "m", "--mode", mode])
            .unwrap()
            .unwrap();
        assert!(selected.reranker_dir.is_some());
        query_controls::validate_startup(&selected, selected.mode.unwrap()).unwrap();
    }
    let partial = parsed(&["--mode", "partial", "--full-query", TUNED])
        .unwrap()
        .unwrap();
    query_controls::validate_startup(&partial, QueryMode::Partial).unwrap();
    assert!(query_controls::validate_startup(&partial, QueryMode::Fast).is_err());
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
fn partial_query_tuning_changes_fusion_and_preserves_actual_quality_evidence() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let parent = private_parent();
            let fast = provider(false);
            let quality = provider(true);
            let (built, _) = publish_partial_cli(&cx, &parent, &fast, &quality, ann).await;
            let mut options = partial_search_options(&built.receipt, Some(QueryMode::Partial));
            let mut responses = Vec::new();
            for tuning in [None, Some(TUNED)] {
                options.full_query = tuning.map(|json| query::FullQuery::parse(json).unwrap());
                let mut output = Vec::new();
                search_with_loader(&cx, &options, &mut output, |tier| {
                    Ok(match tier {
                        EmbedderTierTag::Fast => fast.clone(),
                        EmbedderTierTag::Quality => quality.clone(),
                    })
                })
                .await
                .unwrap();
                responses.push(frames(&output));
            }
            assert_eq!(responses[0].len(), 2);
            assert_eq!(responses[1].len(), 2);
            for (baseline, tuned) in responses[0].iter().zip(&responses[1]) {
                assert_eq!(tuned["phase"], baseline["phase"]);
                assert_eq!(tuned["mode"], "partial");
                assert_eq!(tuned["quality_coverage"]["quality_documents"], 1);
                assert_eq!(tuned["full_query"]["candidate_multiplier"], 2);
                assert!(baseline.get("full_query").is_none());
                let hits = tuned["results"].as_array().unwrap();
                assert_eq!(hits.len(), 3);
                assert!(
                    hits[0]["score"].as_f64().unwrap()
                        > baseline["results"][0]["score"].as_f64().unwrap()
                );
                for hit in hits {
                    assert!(hit["fast_score"].is_number());
                    assert_eq!(
                        hit["quality_score"].is_number(),
                        tuned["phase"] == "refined" && hit["doc_id"] == "quality"
                    );
                }
            }
        }
    });
}

#[test]
fn partial_request_overrides_preserve_scope_limits_and_default_fast_mode() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let parent = private_parent();
        let fast = provider(false);
        let quality = provider(true);
        let (_, receipt) = publish_partial_cli(&cx, &parent, &fast, &quality, true).await;
        let root = admit_root(&receipt.root).unwrap();
        let snapshot = receipt
            .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
            .unwrap();
        let mut session = AuthoritySession::new(
            &cx,
            root,
            receipt,
            snapshot,
            models(&fast, &quality),
            (3, DEFAULT_IMAGE_BYTES),
            false,
        )
        .unwrap();
        session.query_policy.full_query = Some(query::FullQuery::parse(GRANTED).unwrap());
        let message = serde_json::json!({
            "op":"search","request":{"query":"retry","mode":"partial",
                "filter":{"ids":["fast-only"]}},"full_query":{"candidate_multiplier":1}
        });
        let mut excessive = message.clone();
        excessive["full_query"]["candidate_multiplier"] = serde_json::json!(5);
        let mut implicit = message.clone();
        implicit["request"].as_object_mut().unwrap().remove("mode");
        let output = exchange(
            &cx,
            &mut session,
            &[
                excessive,
                implicit,
                message,
                serde_json::json!({"query":"retry"}),
            ],
        )
        .await;
        assert_eq!(output.len(), 5);
        assert_eq!(output[0]["default_mode"], "fast");
        assert_eq!(output[1]["code"], "query_tuning_refused");
        assert_eq!(output[2]["code"], "query_tuning_requires_partial_mode");
        assert_eq!(output[3]["mode"], "partial");
        assert_eq!(output[3]["full_query"]["candidate_multiplier"], 1);
        assert_eq!(output[3]["full_query"]["quality_ef"], 16);
        assert_eq!(output[3]["scope"]["eligible_documents"], 1);
        let hits = output[3]["results"].as_array().unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["doc_id"], "fast-only");
        assert!(hits[0]["quality_score"].is_null());
        assert_eq!(output[4]["mode"], "fast");
        assert!(output[4].get("full_query").is_none());
    });
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
            let mut documents = f.scorer.seen.lock().unwrap().last().unwrap().clone();
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
fn authority_primary_reranking_uses_only_selected_owners_and_retained_source_scope() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for topology in [
            NativeGenerationTopology::FastOnly,
            NativeGenerationTopology::QualityOnly,
            NativeGenerationTopology::FullProgressive,
        ] {
            for ann in [false, true] {
                let mut f = fixture_topology(&cx, topology, ann).await;
                f.session.query_policy.rerank =
                    Some(query::Rerank::new(f.scorer.clone(), 3).unwrap());
                let retained = Arc::clone(&f.session.snapshot);
                let modes: &[&str] = match topology {
                    NativeGenerationTopology::FastOnly => &["fast"],
                    NativeGenerationTopology::QualityOnly => &["quality"],
                    NativeGenerationTopology::FullProgressive => &["fast", "quality"],
                    NativeGenerationTopology::PartialQuality { .. } => {
                        unreachable!("this fixture selects complete coverage")
                    }
                };
                for &mode in modes {
                    let mut buffered_results = None;
                    for stream in [false, true] {
                        let before = [
                            f.fast.calls.load(Ordering::SeqCst),
                            f.quality.calls.load(Ordering::SeqCst),
                        ];
                        let output = exchange(
                            &cx,
                            &mut f.session,
                            &[serde_json::json!({
                                "query":"needle", "mode":mode, "limit":1, "stream":stream,
                                "filter":{"ids":["vector","quality","keyword","excluded"]},
                            })],
                        )
                        .await;
                        assert_eq!(output.len(), if stream { 3 } else { 2 }, "{output:?}");
                        assert_eq!(output[0]["progressive_available"], true);
                        assert!(
                            output[0]["rerank_modes"]
                                .as_array()
                                .unwrap()
                                .contains(&serde_json::json!(mode))
                        );
                        let final_page = output.last().unwrap();
                        assert_eq!(final_page["phase"], "reranked");
                        assert_eq!(final_page["results"][0]["doc_id"], "vector");
                        assert_eq!(final_page["results"][0]["metadata"]["source_id"], "vector");
                        assert_eq!(final_page["reranker"], f.scorer.id());
                        assert_eq!(final_page["scope"]["eligible_documents"], 3);
                        assert!(final_page["full_query"].is_null());
                        let hit = &final_page["results"][0];
                        assert_eq!(hit["source"], "Reranked");
                        assert!(hit["rerank_score"].is_number());
                        assert!(hit["index"].is_number());
                        assert!(
                            hit[if mode == "quality" {
                                "fast_score"
                            } else {
                                "quality_score"
                            }]
                            .is_null()
                        );
                        assert!(
                            hit[if mode == "quality" {
                                "quality_score"
                            } else {
                                "fast_score"
                            }]
                            .is_number()
                        );
                        if stream {
                            assert_eq!(
                                output[1]["phase"],
                                if mode == "quality" {
                                    "quality"
                                } else {
                                    "initial"
                                }
                            );
                            assert_ne!(
                                output[1]["results"][0]["doc_id"], "vector",
                                "reranking promotes a candidate outside the displayed page"
                            );
                            assert_eq!(Some(final_page["results"].clone()), buffered_results);
                        } else {
                            buffered_results = Some(final_page["results"].clone());
                        }
                        assert_eq!(
                            f.fast.calls.load(Ordering::SeqCst) - before[0],
                            usize::from(mode == "fast")
                        );
                        assert_eq!(
                            f.quality.calls.load(Ordering::SeqCst) - before[1],
                            usize::from(mode == "quality")
                        );
                        let mut observed = f.scorer.seen.lock().unwrap().last().unwrap().clone();
                        observed.sort();
                        assert_eq!(
                            observed,
                            vec![
                                ("keyword".to_owned(), "needle keyword".to_owned()),
                                ("quality".to_owned(), "quality source".to_owned()),
                                ("vector".to_owned(), "vector source".to_owned()),
                            ]
                        );
                    }
                    let before = [
                        f.fast.calls.load(Ordering::SeqCst),
                        f.quality.calls.load(Ordering::SeqCst),
                        f.scorer.calls.load(Ordering::SeqCst),
                    ];
                    let empty = exchange(
                        &cx,
                        &mut f.session,
                        &[serde_json::json!({
                            "query":"needle", "mode":mode, "stream":true,
                            "filter":{"ids":["excluded"]},
                        })],
                    )
                    .await;
                    assert_eq!(empty.len(), 2);
                    assert!(empty[1]["results"].as_array().unwrap().is_empty());
                    let excluded = exchange(
                        &cx,
                        &mut f.session,
                        &[serde_json::json!({
                            "query":EXCLUSIONS, "mode":mode, "stream":true,
                        })],
                    )
                    .await;
                    assert_keyword_exclusion_results(
                        &excluded,
                        if mode == "quality" {
                            &["quality"]
                        } else {
                            &["initial"]
                        },
                    );
                    assert_eq!(
                        before,
                        [
                            f.fast.calls.load(Ordering::SeqCst),
                            f.quality.calls.load(Ordering::SeqCst),
                            f.scorer.calls.load(Ordering::SeqCst)
                        ]
                    );
                    let tier = if mode == "quality" {
                        EmbedderTierTag::Quality
                    } else {
                        EmbedderTierTag::Fast
                    };
                    let (selected, wrong) = if mode == "quality" {
                        (f.quality.as_ref(), f.fast.as_ref())
                    } else {
                        (f.fast.as_ref(), f.quality.as_ref())
                    };
                    assert!(
                        f.session
                            .snapshot
                            .search_tier_progressive(&cx, tier, wrong, "needle", 0, None)
                            .is_err(),
                        "zero limit must still admit the exact requested producer"
                    );
                    let empty_scope = f
                        .session
                        .scope
                        .as_ref()
                        .unwrap()
                        .narrow(&cx, |_| Ok(false))
                        .unwrap();
                    assert!(
                        empty_scope
                            .search_tier_progressive(&cx, tier, wrong, "needle", 0, None)
                            .is_err(),
                        "empty membership cannot bypass producer admission"
                    );
                    let mut zero = f
                        .session
                        .snapshot
                        .search_tier_progressive(&cx, tier, selected, "needle", 0, None)
                        .unwrap()
                        .with_retained_reranker(f.scorer.as_ref(), 3, 0)
                        .unwrap();
                    assert!(
                        matches!(zero.next_phase().await.unwrap(), Some(NativeSearchPhase::Initial { results, .. }) if results.is_empty())
                    );
                    assert!(zero.is_finished());
                    assert!(zero.next_phase().await.unwrap().is_none());
                    assert_eq!(
                        before,
                        [
                            f.fast.calls.load(Ordering::SeqCst),
                            f.quality.calls.load(Ordering::SeqCst),
                            f.scorer.calls.load(Ordering::SeqCst)
                        ]
                    );
                    let before_rerank = f.scorer.calls.load(Ordering::SeqCst);
                    let mut direct = f
                        .session
                        .scope
                        .as_ref()
                        .unwrap()
                        .search_tier_progressive(&cx, tier, selected, "needle", 3, None)
                        .unwrap()
                        .with_retained_reranker(f.scorer.as_ref(), 3, 512)
                        .unwrap();
                    let Some(NativeSearchPhase::Initial {
                        results: initial,
                        candidates,
                    }) = direct.next_phase().await.unwrap()
                    else {
                        panic!("a selected tier must deliver its first page before reranking");
                    };
                    assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before_rerank);
                    assert_eq!(candidates.fast, if mode == "fast" { 3 } else { 0 });
                    assert_eq!(candidates.quality, if mode == "quality" { 3 } else { 0 });
                    let Some(NativeSearchPhase::Reranked {
                        results,
                        candidates: reranked_candidates,
                        evaluated,
                    }) = direct.next_phase().await.unwrap()
                    else {
                        panic!("a selected tier must reach its retained reranker");
                    };
                    assert_eq!(evaluated, 3);
                    assert_eq!(candidates, reranked_candidates);
                    for hit in &results {
                        let prior = initial
                            .iter()
                            .find(|prior| prior.doc_id == hit.doc_id)
                            .unwrap();
                        assert_eq!(hit.index, prior.index);
                        assert_eq!(hit.fast_score, prior.fast_score);
                        assert_eq!(hit.quality_score, prior.quality_score);
                        assert_eq!(hit.metadata, prior.metadata);
                    }
                    assert!(direct.is_finished());
                }
                if topology != NativeGenerationTopology::FullProgressive {
                    let absent = if topology == NativeGenerationTopology::QualityOnly {
                        EmbedderTierTag::Fast
                    } else {
                        EmbedderTierTag::Quality
                    };
                    assert!(
                        f.session
                            .snapshot
                            .search_tier_progressive(
                                &cx,
                                absent,
                                f.fast.as_ref(),
                                "needle",
                                0,
                                None
                            )
                            .is_err()
                    );
                }
                assert!(Arc::ptr_eq(&retained, &f.session.snapshot));
            }
        }
    });
}

#[test]
fn authority_primary_rerank_failures_and_output_failures_preserve_the_previous_page() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mode in ["fast", "quality"] {
            let mut f = fixture_topology(
                &cx,
                if mode == "quality" {
                    NativeGenerationTopology::QualityOnly
                } else {
                    NativeGenerationTopology::FastOnly
                },
                false,
            )
            .await;
            f.session.query_policy.rerank = Some(query::Rerank::new(f.scorer.clone(), 3).unwrap());
            let selected = f.session.snapshot.authority();
            let request =
                |stream| serde_json::json!({"query":"needle", "mode":mode, "stream":stream});
            for failure in [0, 1, 3] {
                for stream in [false, true] {
                    f.session.max_source_bytes = if failure == 0 { 0 } else { 512 };
                    f.scorer.mode.store(failure, Ordering::SeqCst);
                    let before = f.scorer.calls.load(Ordering::SeqCst);
                    let output = exchange(&cx, &mut f.session, &[request(stream)]).await;
                    assert_eq!(output.len(), if stream { 3 } else { 2 }, "{output:?}");
                    assert_eq!(output.last().unwrap()["event"], "failed");
                    assert!(!output.iter().any(|frame| frame["phase"] == "reranked"));
                    if stream {
                        assert_eq!(output[1]["event"], "results");
                        assert_eq!(
                            output[1]["phase"],
                            if mode == "quality" {
                                "quality"
                            } else {
                                "initial"
                            }
                        );
                    }
                    assert_eq!(
                        f.scorer.calls.load(Ordering::SeqCst) - before,
                        usize::from(failure != 0)
                    );
                    assert_eq!(f.session.snapshot.authority(), selected);
                }
            }
            f.session.max_source_bytes = 512;
            f.session.query_policy.maximum_ms = Some(10);
            let request: AuthoritySearchRequest = serde_json::from_value(request(true)).unwrap();
            let phase = if mode == "quality" {
                "quality"
            } else {
                "initial"
            };
            let before = f.scorer.calls.load(Ordering::SeqCst);
            for fail in [true, false] {
                let mut output = PhaseOutput {
                    bytes: Vec::new(),
                    clock: Arc::clone(&f.clock),
                    phase,
                    fail,
                };
                let error = f
                    .session
                    .query(&cx, &request, 1, &mut output)
                    .await
                    .unwrap_err();
                if fail {
                    assert_eq!(
                        error.downcast_ref::<AuthorityServeRefusal>().unwrap().0,
                        "result_delivery_failed"
                    );
                    assert!(output.bytes.is_empty());
                } else {
                    assert!(matches!(
                        error.downcast_ref::<SearchError>(),
                        Some(SearchError::SearchTimeout { .. })
                    ));
                    assert_eq!(frames(&output.bytes).len(), 1);
                }
                assert_eq!(
                    f.scorer.calls.load(Ordering::SeqCst),
                    before,
                    "no reranking starts after failed or late first-phase delivery"
                );
            }
            let mut final_output = PhaseOutput {
                bytes: Vec::new(),
                clock: Arc::clone(&f.clock),
                phase: "reranked",
                fail: false,
            };
            f.session
                .query(&cx, &request, 2, &mut final_output)
                .await
                .unwrap();
            assert_eq!(
                frames(&final_output.bytes).len(),
                2,
                "slow completed final output is not a retrospective timeout"
            );
        }
    });
}

#[test]
fn authority_primary_pending_rerank_uses_the_original_deadline_and_releases_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mode in ["fast", "quality"] {
            for stream in [false, true] {
                let mut f = fixture_topology(
                    &cx,
                    if mode == "quality" {
                        NativeGenerationTopology::QualityOnly
                    } else {
                        NativeGenerationTopology::FastOnly
                    },
                    false,
                )
                .await;
                f.session.query_policy.rerank =
                    Some(query::Rerank::new(f.scorer.clone(), 3).unwrap());
                f.session.query_policy.maximum_ms = Some(10);
                f.fast.delay_ns.store(4_000_000, Ordering::SeqCst);
                f.quality.delay_ns.store(4_000_000, Ordering::SeqCst);
                f.scorer.mode.store(2, Ordering::SeqCst);
                let request = serde_json::json!({"query":"needle", "mode":mode, "stream":stream, "timeout_ms":100});
                let mut input = Cursor::new(format!("{request}\n"));
                let mut output = Vec::new();
                let mut serving =
                    Box::pin(serve_session(&cx, &mut f.session, &mut input, &mut output));
                assert!(
                    serving
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
                assert_eq!(f.scorer.calls.load(Ordering::SeqCst), 1);
                assert_eq!(f.scorer.drops.load(Ordering::SeqCst), 0);
                f.clock.advance(6_000_000);
                let _ = f.timer.process_timers();
                serving.await.unwrap();
                let output = frames(&output);
                assert_eq!(output.len(), if stream { 3 } else { 2 });
                assert_eq!(output.last().unwrap()["code"], "search_timeout");
                assert_eq!(output.last().unwrap()["budget_ms"], 10);
                assert_eq!(output.last().unwrap()["elapsed_ms"], 10);
                assert_eq!(f.scorer.drops.load(Ordering::SeqCst), 1);
                assert!(f.timer.is_empty());
                assert!(!cx.is_cancel_requested());
                f.fast.delay_ns.store(0, Ordering::SeqCst);
                f.quality.delay_ns.store(0, Ordering::SeqCst);
                let next = exchange(&cx, &mut f.session, &[request]).await;
                assert_eq!(next.last().unwrap()["phase"], "reranked");
            }
        }
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
            if mode == "lexical" {
                assert!(output[1].get("reranker").is_none());
                assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before);
            } else {
                assert_eq!(output[1]["phase"], "reranked");
                assert_eq!(output[1]["reranker"], f.scorer.id());
                assert_eq!(f.scorer.calls.load(Ordering::SeqCst), before + 1);
            }
        }
    });
}

#[test]
fn full_controls_refuse_incomplete_topology_before_loading_any_provider() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, false, false).await;
        let mut options = parsed(&["--full-query", TUNED]).unwrap().unwrap();
        options.receipt = f.receipt.clone();
        let error = search_with_loader(&cx, &options, &mut Vec::new(), |_| {
            panic!("incomplete topology must refuse before loading models")
        })
        .await
        .unwrap_err();
        assert!(error.to_string().contains("complete full generation"));
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

    asupersync::test_utils::run_test_with_cx(|_| async move {
        // Native pool admission requires the host runtime's spawn authority.
        let cx = Cx::current().expect("active test runtime");
        for ann in [false, true] {
            for scoped in [false, true] {
                let mut f = fixture(&cx, true, ann).await;
                let scoped_budget = f.session.max_source_bytes;
                if !scoped {
                    f.session.scope = None;
                    f.session.fixed_filter = None;
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
                        // Unrestricted reranking may select the large document.
                        // Exact-ID requests fit the original small source budget.
                        let unrestricted_rerank = tuned && !scoped && request.get("filter").is_none();
                        f.session.max_source_bytes = if unrestricted_rerank {
                            MAX_INPUT_BYTES
                        } else {
                            scoped_budget
                        };
                        f.session.query_policy.query_pool = None;
                        let inline =
                            exchange(&cx, &mut f.session, std::slice::from_ref(request)).await;
                        f.session.query_policy.query_pool = policy.query_pool();
                        let pooled =
                            exchange(&cx, &mut f.session, std::slice::from_ref(request)).await;
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
                    let output = exchange(&cx, &mut f.session, std::slice::from_ref(request)).await;
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

    asupersync::test_utils::run_test_with_cx(|_| async move {
        // Keep the real runtime authority while the session owns virtual time.
        let cx = Cx::current().expect("active test runtime");
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
