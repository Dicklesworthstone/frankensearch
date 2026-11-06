use super::*;
use asupersync::time::{TimerDriverHandle, VirtualClock};
use std::future::Future as _;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::task::{Context, Waker};

struct QueryProvider {
    inner: Arc<Provider>,
    clock: Arc<VirtualClock>,
    delay_ns: AtomicU64,
    identity_delay_ns: AtomicU64,
    pending_once: AtomicBool,
    cancel_once: AtomicBool,
    calls: AtomicUsize,
    drops: AtomicUsize,
}

struct PendingDrop<'a>(&'a AtomicUsize);

impl Drop for PendingDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl QueryProvider {
    fn new(inner: Arc<Provider>, clock: &Arc<VirtualClock>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            clock: Arc::clone(clock),
            delay_ns: AtomicU64::new(0),
            identity_delay_ns: AtomicU64::new(0),
            pending_once: AtomicBool::new(false),
            cancel_once: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        })
    }
}

impl Embedder for QueryProvider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.clock
            .advance(self.identity_delay_ns.swap(0, Ordering::SeqCst));
        self.inner.identity()
    }

    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _owned_work = PendingDrop(&self.drops);
            if self.pending_once.swap(false, Ordering::SeqCst) {
                // Deliberately never registers or invokes a provider waker.
                // Only the query's captured timer can finish this request.
                std::future::pending::<()>().await;
            }
            self.clock.advance(self.delay_ns.load(Ordering::SeqCst));
            if self.cancel_once.swap(false, Ordering::SeqCst) {
                return Err(SearchError::Cancelled {
                    phase: "authority_timeout_test.provider".to_owned(),
                    reason: "provider cancelled".to_owned(),
                });
            }
            self.inner.embed(cx, text).await
        })
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

struct Fixture {
    session: AuthoritySession,
    fast: Arc<QueryProvider>,
    quality: Arc<QueryProvider>,
    clock: Arc<VirtualClock>,
    timer: TimerDriverHandle,
    receipt_path: PathBuf,
}

async fn fixture(cx: &Cx, maximum_ms: Option<u64>) -> Fixture {
    let parent = private_parent();
    let fast = provider(false);
    let quality = provider(true);
    let selected = publish_full(
        cx,
        &parent,
        &fast,
        &quality,
        vec![
            IndexableDocument::new("retry", "retry network requests")
                .with_metadata("version", "retained"),
            IndexableDocument::new("garden", "garden flowers"),
        ],
        [NativeBuildRetrieval::Exact; 2],
    )
    .await;
    let snapshot = selected.open(cx, DEFAULT_IMAGE_BYTES).unwrap();
    let clock = Arc::new(VirtualClock::new());
    let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
    let fast = QueryProvider::new(fast, &clock);
    let quality = QueryProvider::new(quality, &clock);
    let mut session = AuthoritySession::new(
        cx,
        admit_root(&selected.root).unwrap(),
        selected,
        snapshot,
        Models {
            fast: fast.clone(),
            quality: Some(quality.clone()),
        },
        (2, DEFAULT_IMAGE_BYTES),
        false,
    )
    .unwrap()
    .with_filter(
        cx,
        Some(Filter::parse(r#"{"id_prefix":"retry"}"#).unwrap()),
        MAX_INPUT_BYTES,
    )
    .unwrap();
    session.query_policy = query::Policy {
        maximum_ms,
        timer: Some(timer.clone()),
        ..query::Policy::default()
    };
    Fixture {
        session,
        fast,
        quality,
        clock,
        timer,
        receipt_path: parent.join("initial.json"),
    }
}

fn request(mode: &str, stream: bool, timeout_ms: Option<u64>) -> AuthoritySearchRequest {
    serde_json::from_value(serde_json::json!({
        "query":"retry", "mode":mode, "stream":stream, "timeout_ms":timeout_ms,
        "filter":{"ids":["retry"]},
    }))
    .unwrap()
}

fn assert_timeout(error: &(dyn Error + 'static), budget: u64, elapsed: u64) {
    assert!(matches!(
        error.downcast_ref::<SearchError>(),
        Some(SearchError::SearchTimeout { budget_ms, elapsed_ms })
            if *budget_ms == budget && *elapsed_ms == elapsed
    ));
}

#[test]
fn authority_timeout_options_are_bounded_and_only_apply_to_queries() {
    for command in ["authority-search", "authority-serve"] {
        let mut base = vec![command, "--receipt", "selected.json"];
        if command == "authority-search" {
            base.extend(["--query", "retry"]);
        }
        for value in ["1", "600000"] {
            let mut args = base.clone();
            args.extend(["--timeout-ms", value]);
            assert_eq!(
                parse(&args).unwrap().unwrap().timeout_ms,
                Some(value.parse().unwrap())
            );
        }
        for value in ["0", "600001", "-1", "1.5", "18446744073709551616"] {
            let mut args = base.clone();
            args.extend(["--timeout-ms", value]);
            assert!(parse(&args).is_err(), "{command}: {value}");
        }
        base.extend(["--timeout-ms", "10", "--timeout-ms", "20"]);
        assert!(parse(&base).is_err());
    }
    for mut args in [
        index_arguments(),
        rebuild_arguments(),
        update_arguments(),
        rollback_arguments(),
        vec!["authority-recover", "--receipt", "selected.json"],
        vec![
            "authority-fetch",
            "--receipt",
            "selected.json",
            "--doc-id",
            "retry",
        ],
    ] {
        args.extend(["--timeout-ms", "10"]);
        assert!(parse(&args).is_err());
    }
}

#[test]
fn nonwaking_provider_timeout_drops_work_and_keeps_the_warm_selection_usable() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (mode, stream) in [
            ("fast", false),
            ("quality", false),
            ("full", false),
            ("full", true),
        ] {
            let mut f = fixture(&cx, Some(10)).await;
            let pending = if mode == "fast" { &f.fast } else { &f.quality };
            pending.pending_once.store(true, Ordering::SeqCst);
            let retained = Arc::clone(&f.session.snapshot);
            let model = Arc::clone(&f.session.models.models().unwrap().fast);
            let authority = f.session.receipt.authority.clone();
            let source_scope = f.session.scope.as_ref().unwrap().len();
            let mut input = Cursor::new(format!(
                "{}\n{}\n",
                serde_json::json!({"id":"slow", "query":"retry", "mode":mode,
                    "stream":stream, "timeout_ms":100, "filter":{"ids":["retry"]}}),
                serde_json::json!({"id":"next", "query":"retry", "mode":mode, "stream":stream}),
            ));
            let mut output = Vec::new();
            let mut serving = Box::pin(serve_session(&cx, &mut f.session, &mut input, &mut output));
            assert!(
                serving
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(pending.calls.load(Ordering::SeqCst), 1);
            assert_eq!(pending.drops.load(Ordering::SeqCst), 0);
            assert_eq!(f.timer.pending_count(), 1);
            f.clock.advance(10_000_000);
            let _ = f.timer.process_timers();
            serving.await.unwrap();
            assert_eq!(pending.calls.load(Ordering::SeqCst), 2);
            assert_eq!(pending.drops.load(Ordering::SeqCst), 2);
            assert!(f.timer.is_empty());
            assert!(!cx.is_cancel_requested());
            assert!(Arc::ptr_eq(&retained, &f.session.snapshot));
            assert!(Arc::ptr_eq(
                &model,
                &f.session.models.models().unwrap().fast
            ));
            assert_eq!(f.session.receipt.authority, authority);
            assert_eq!(f.session.scope.as_ref().unwrap().len(), source_scope);
            let output = frames(&output);
            assert_eq!(output[0]["maximum_timeout_ms"], 10);
            let slow: Vec<_> = output
                .iter()
                .filter(|frame| frame["id"] == "slow")
                .collect();
            assert_eq!(slow.len(), if stream { 2 } else { 1 });
            let failed = slow.last().unwrap();
            assert_eq!(failed["event"], "failed");
            assert_eq!(failed["status"], "timed_out");
            assert_eq!(failed["code"], "search_timeout");
            assert_eq!(failed["elapsed_ms"], 10);
            assert_eq!(
                failed["budget_ms"], 10,
                "request cannot extend the server ceiling"
            );
            assert!(
                failed.get("error").is_none(),
                "provider bodies remain redacted"
            );
            let next: Vec<_> = output
                .iter()
                .filter(|frame| frame["id"] == "next")
                .collect();
            assert_eq!(next.len(), if stream { 2 } else { 1 });
            assert!(next.iter().all(|frame| frame["event"] == "results"));
            assert_eq!(next.last().unwrap()["results"][0]["doc_id"], "retry");
        }
    });
}

#[test]
fn each_stage_spends_the_original_budget_and_request_overrides_only_shorten_it() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, Some(10)).await;
        f.fast.delay_ns.store(6_000_000, Ordering::SeqCst);
        f.quality.delay_ns.store(6_000_000, Ordering::SeqCst);
        for stream in [false, true] {
            let mut output = Vec::new();
            let error = f
                .session
                .query(&cx, &request("full", stream, None), 1, &mut output)
                .await
                .unwrap_err();
            assert_timeout(error.as_ref(), 10, 12);
            assert_eq!(frames(&output).len(), usize::from(stream));
        }
        let mut output = Vec::new();
        let error = f
            .session
            .query(&cx, &request("fast", false, Some(5)), 2, &mut output)
            .await
            .unwrap_err();
        assert_timeout(error.as_ref(), 5, 6);
        assert!(output.is_empty());
        let mut output = Vec::new();
        f.session
            .query(&cx, &request("fast", false, Some(50)), 3, &mut output)
            .await
            .unwrap();
        assert_eq!(frames(&output).len(), 1);
        assert!(!cx.is_cancel_requested());
    });
}

struct SlowOutput {
    bytes: Vec<u8>,
    clock: Arc<VirtualClock>,
    phase: &'static str,
}

impl Write for SlowOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        if let Ok(frame) = serde_json::from_slice::<serde_json::Value>(bytes)
            && frame["event"] == "results"
            && frame["phase"] == self.phase
        {
            self.clock.advance(20_000_000);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn output_delay_cannot_reclassify_completed_work_but_consumes_the_next_real_phase_budget() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, Some(10)).await;
        for (mode, stream, delayed_phase) in [
            ("fast", false, "initial"),
            ("quality", false, "quality"),
            ("lexical", false, "lexical"),
            ("full", false, "refined"),
            ("full", true, "refined"),
        ] {
            let mut output = SlowOutput {
                bytes: Vec::new(),
                clock: Arc::clone(&f.clock),
                phase: delayed_phase,
            };
            f.session
                .query(&cx, &request(mode, stream, None), 1, &mut output)
                .await
                .unwrap();
            assert_eq!(frames(&output.bytes).len(), if stream { 2 } else { 1 });
        }
        let before = f.quality.calls.load(Ordering::SeqCst);
        let mut output = SlowOutput {
            bytes: Vec::new(),
            clock: Arc::clone(&f.clock),
            phase: "initial",
        };
        let error = f
            .session
            .query(&cx, &request("full", true, None), 2, &mut output)
            .await
            .unwrap_err();
        assert_timeout(error.as_ref(), 10, 20);
        assert_eq!(frames(&output.bytes).len(), 1);
        assert_eq!(
            f.quality.calls.load(Ordering::SeqCst),
            before,
            "expired quality is never started"
        );

        // Native computation for an empty scope finishes after empty Initial.
        // Its synthetic Refined frame is delivery, not another timed phase.
        let mut empty = request("full", true, None);
        empty.filter = Some(Filter::parse(r#"{"ids":["outside-frozen-scope"]}"#).unwrap());
        let before = (
            f.fast.calls.load(Ordering::SeqCst),
            f.quality.calls.load(Ordering::SeqCst),
        );
        let mut output = SlowOutput {
            bytes: Vec::new(),
            clock: Arc::clone(&f.clock),
            phase: "initial",
        };
        f.session.query(&cx, &empty, 3, &mut output).await.unwrap();
        let output = frames(&output.bytes);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["phase"], "initial");
        assert_eq!(output[1]["phase"], "refined");
        assert!(
            output
                .iter()
                .all(|frame| frame["results"] == serde_json::json!([]))
        );
        assert_eq!(
            (
                f.fast.calls.load(Ordering::SeqCst),
                f.quality.calls.load(Ordering::SeqCst)
            ),
            before
        );
    });
}

#[test]
fn invalid_timeouts_and_expired_admission_do_no_inference_and_keep_refusal_codes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let mut f = fixture(&cx, Some(10)).await;
        let output = exchange(
            &cx,
            &mut f.session,
            &[
                serde_json::json!({"query":"retry", "timeout_ms":0}),
                serde_json::json!({"query":"retry", "timeout_ms":600001}),
                serde_json::json!({"query":"retry", "timeout_ms":"private-invalid-budget"}),
            ],
        )
        .await;
        assert_eq!(output[1]["code"], "invalid_timeout");
        assert_eq!(output[2]["code"], "invalid_timeout");
        assert_eq!(output[3]["code"], "invalid_request");
        assert_eq!(f.fast.calls.load(Ordering::SeqCst), 0);
        assert_eq!(f.quality.calls.load(Ordering::SeqCst), 0);
        assert!(
            !serde_json::to_string(&output)
                .unwrap()
                .contains("private-invalid-budget")
        );

        f.fast.identity_delay_ns.store(10_000_000, Ordering::SeqCst);
        let mut output = Vec::new();
        let error = f
            .session
            .query(&cx, &request("fast", false, None), 4, &mut output)
            .await
            .unwrap_err();
        assert_timeout(error.as_ref(), 10, 10);
        assert!(output.is_empty());
        assert_eq!(f.fast.calls.load(Ordering::SeqCst), 0);

        let mut changed = provider_identity();
        changed.producer.implementation_revision = "foreign-query-provider".to_owned();
        changed.validate().unwrap();
        let foreign = QueryProvider::new(
            Arc::new(Provider {
                identity: changed,
                calls: AtomicUsize::new(0),
            }),
            &f.clock,
        );
        f.session.models = Models {
            fast: foreign.clone(),
            quality: Some(f.quality.clone()),
        }
        .into();
        let error = f
            .session
            .query(&cx, &request("fast", false, None), 5, &mut output)
            .await
            .unwrap_err();
        assert_eq!(
            f.session.failure(5, None, error.as_ref(), "query_refused")["code"],
            "producer_identity_mismatch"
        );
        foreign
            .identity_delay_ns
            .store(10_000_000, Ordering::SeqCst);
        let error = f
            .session
            .query(&cx, &request("fast", false, None), 6, &mut output)
            .await
            .unwrap_err();
        assert_timeout(error.as_ref(), 10, 10);
        assert_eq!(foreign.calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn one_shot_search_times_pending_inference_after_startup_and_before_final_delivery() {
    let clock = Arc::new(VirtualClock::new());
    let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
    let runtime = RuntimeBuilder::current_thread()
        .with_timer_driver(timer.clone())
        .build()
        .unwrap();
    runtime.block_on(async {
        let cx = Cx::current().expect("runtime query context");
        let f = fixture(&cx, None).await;
        let fast = QueryProvider::new(Arc::clone(&f.fast.inner), &clock);
        let quality = QueryProvider::new(Arc::clone(&f.quality.inner), &clock);
        for (mode, stream) in [
            (QueryMode::Fast, false),
            (QueryMode::Quality, false),
            (QueryMode::Full, false),
            (QueryMode::Full, true),
        ] {
            let mut options = parse(&[
                "authority-search",
                "--receipt",
                "unused",
                "--query",
                "retry",
                "--timeout-ms",
                "10",
            ])
            .unwrap()
            .unwrap();
            options.receipt = f.receipt_path.clone();
            options.mode = Some(mode);
            options.stream = stream;
            options.filter = Some(Filter::parse(r#"{"ids":["retry"]}"#).unwrap());
            let pending = if mode == QueryMode::Fast {
                &fast
            } else {
                &quality
            };
            let before = pending.drops.load(Ordering::SeqCst);
            pending.pending_once.store(true, Ordering::SeqCst);
            let loads = AtomicUsize::new(0);
            let mut output = Vec::new();
            let mut searching = Box::pin(search_with_loader(&cx, &options, &mut output, |tier| {
                loads.fetch_add(1, Ordering::SeqCst);
                // Loading is startup work. Both full-mode loads can take
                // longer than the entire subsequent query budget.
                clock.advance(20_000_000);
                let model: Arc<dyn Embedder> = if tier == EmbedderTierTag::Fast {
                    fast.clone()
                } else {
                    quality.clone()
                };
                Ok(model)
            }));
            assert!(
                searching
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(
                loads.load(Ordering::SeqCst),
                if mode == QueryMode::Full { 2 } else { 1 }
            );
            assert_eq!(timer.pending_count(), 1);
            clock.advance(10_000_000);
            let _ = timer.process_timers();
            let error = searching.await.unwrap_err();
            assert_timeout(error.as_ref(), 10, 10);
            assert_eq!(query::failure(error.as_ref())["code"], "search_timeout");
            assert_eq!(frames(&output).len(), usize::from(stream));
            assert_eq!(pending.drops.load(Ordering::SeqCst), before + 1);
            assert!(timer.is_empty());
            assert!(!cx.is_cancel_requested());
        }
        let mut lexical = parse(&[
            "authority-search",
            "--receipt",
            "unused",
            "--query",
            "retry",
            "--lexical-only",
            "--timeout-ms",
            "10",
        ])
        .unwrap()
        .unwrap();
        lexical.receipt = f.receipt_path;
        let mut output = Vec::new();
        search_with_loader(&cx, &lexical, &mut output, |_| {
            panic!("lexical queries never load models")
        })
        .await
        .unwrap();
        assert_eq!(frames(&output)[0]["phase"], "lexical");
    });
}

#[test]
fn provider_cancellation_remains_cancellation_at_the_deadline() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let f = fixture(&cx, Some(10)).await;
        f.fast.delay_ns.store(10_000_000, Ordering::SeqCst);
        f.fast.cancel_once.store(true, Ordering::SeqCst);
        let mut output = Vec::new();
        let error = f
            .session
            .query(&cx, &request("fast", false, None), 1, &mut output)
            .await
            .unwrap_err();
        assert!(
            matches!(error.downcast_ref::<SearchError>(), Some(SearchError::Cancelled { phase, .. }) if phase == "authority_timeout_test.provider")
        );
        assert!(output.is_empty());
        assert!(f.timer.is_empty());
    });
}
