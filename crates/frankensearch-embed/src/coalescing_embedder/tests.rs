use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Clone, Copy)]
enum Mode {
    Good,
    WrongLast,
    BadValues,
    Short,
    Failure,
    CancelSuccess,
    CancelFailure,
}

struct Probe {
    identity: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    native: bool,
    mode: Mode,
    drift: AtomicBool,
    pending: AtomicBool,
    raw: AtomicUsize,
    singleton: AtomicUsize,
    batches: AtomicUsize,
    drops: AtomicUsize,
    formatted: Arc<AtomicUsize>,
}

impl Probe {
    fn new(native: bool, mode: Mode) -> Self {
        Self {
            identity: identity(),
            foreign: EmbeddingIdentityBundleV1::explicit_test_model("other-producer", 2),
            native,
            mode,
            drift: AtomicBool::new(false),
            pending: AtomicBool::new(false),
            raw: AtomicUsize::new(0),
            singleton: AtomicUsize::new(0),
            batches: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            formatted: Arc::new(AtomicUsize::new(0)),
        }
    }

    async fn produce(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        struct Guard<'a>(&'a AtomicUsize);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let _guard = Guard(&self.drops);
        if self.pending.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        let mut outputs: Vec<_> = texts
            .iter()
            .map(|text| IdentityBoundEmbedding {
                identity: self.identity.clone(),
                values: vec![text.parse().unwrap(), -0.0],
            })
            .collect();
        if texts.contains(&"30") {
            match self.mode {
                Mode::WrongLast => outputs.last_mut().unwrap().identity = self.foreign.clone(),
                Mode::BadValues => outputs.last_mut().unwrap().values[0] = f32::NAN,
                Mode::Short => {
                    let _ = outputs.pop();
                }
                _ => {}
            }
        }
        if matches!(self.mode, Mode::CancelSuccess | Mode::CancelFailure) {
            cx.set_cancel_requested(true);
        }
        if matches!(self.mode, Mode::Failure | Mode::CancelFailure) {
            return Err(SearchError::EmbeddingFailed {
                model: "test".to_owned(),
                source: Box::new(Opaque(Arc::clone(&self.formatted))),
            });
        }
        Ok(outputs)
    }
}

struct Opaque(Arc<AtomicUsize>);
impl fmt::Display for Opaque {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(fmt::Error)
    }
}
impl fmt::Debug for Opaque {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for Opaque {}

impl Embedder for Probe {
    fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.raw.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(vec![999.0, 1.0]) })
    }
    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        self.singleton.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let mut outputs = self.produce(cx, &[text]).await?;
            outputs
                .pop()
                .ok_or_else(|| invalid("fixture", "empty singleton"))
        })
    }
    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        self.batches.fetch_add(1, Ordering::SeqCst);
        Box::pin(self.produce(cx, texts))
    }
    fn bound_batch_is_native(&self) -> bool {
        self.native
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.drift.load(Ordering::SeqCst) {
            &self.foreign
        } else {
            &self.identity
        })
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &'static str {
        "coalescing-test"
    }
    fn model_name(&self) -> &'static str {
        "Coalescing test model"
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

fn identity() -> EmbeddingIdentityBundleV1 {
    EmbeddingIdentityBundleV1::explicit_test_model("coalescing-test", 2)
}

fn setup(
    cx: &Cx,
    provider: Arc<Probe>,
    capacity: usize,
) -> (CoalescingEmbedder, CoalescingDispatcher) {
    CoalescingEmbedder::new(
        cx,
        provider,
        identity(),
        CoalescerConfig {
            max_batch_size: 3,
            max_wait_ms: 0,
            min_batch_size: 1,
            use_priority_lanes: true,
        },
        CoalescerLimits {
            max_outstanding_requests: capacity,
            max_outstanding_text_bytes: 128,
        },
    )
    .unwrap()
}

fn pending<T>(future: &mut SearchFuture<'_, T>) {
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ));
}

#[test]
fn concurrent_trait_callers_receive_bound_values_without_raw_inference() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for native in [false, true] {
            let provider = Arc::new(Probe::new(native, Mode::Good));
            let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 3);
            let background = client.with_priority(Priority::Background);
            let mut first = client.embed_bound(&cx, "10");
            let mut second = background.embed_bound(&cx, "20");
            let mut third = client.embed_bound(&cx, "30");
            for future in [&mut first, &mut second, &mut third] {
                pending(future);
            }
            assert_eq!(client.usage().outstanding_requests, 3);
            assert_eq!(dispatcher.dispatch_ready(&cx).await.unwrap(), 3);
            for (future, expected) in [(first, 10.0_f32), (second, 20.0), (third, 30.0)] {
                let result = future.await.unwrap();
                assert_eq!(result.identity, identity());
                assert_eq!(result.values[0].to_bits(), expected.to_bits());
                assert_eq!(result.values[1].to_bits(), (-0.0_f32).to_bits());
            }
            assert_eq!(provider.raw.load(Ordering::SeqCst), 0);
            assert_eq!(provider.batches.load(Ordering::SeqCst), usize::from(native));
            assert_eq!(
                provider.singleton.load(Ordering::SeqCst),
                if native { 0 } else { 3 }
            );
            assert_eq!(client.usage(), CoalescerUsage::default());
            assert_eq!(dispatcher.dispatch_ready(&cx).await.unwrap(), 0);
            assert!(client.bound_batch_is_native());
            assert!(!client.is_semantic());
        }
    });
}

#[test]
fn an_invalid_last_row_never_delivers_a_successful_prefix() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for native in [false, true] {
            for mode in [Mode::WrongLast, Mode::BadValues, Mode::Short] {
                let provider = Arc::new(Probe::new(native, mode));
                let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 3);
                let mut first = client.embed_bound(&cx, "10");
                let mut second = client.embed_bound(&cx, "20");
                let mut third = client.embed_bound(&cx, "30");
                for future in [&mut first, &mut second, &mut third] {
                    pending(future);
                }
                assert!(dispatcher.dispatch_ready(&cx).await.is_err());
                for future in [first, second, third] {
                    let error = future.await.unwrap_err();
                    assert!(matches!(
                        error,
                        SearchError::UnverifiableRemoteSpace { .. }
                            | SearchError::InvalidConfig { .. }
                    ));
                }
                assert_eq!(client.usage(), CoalescerUsage::default());
                assert_eq!(provider.raw.load(Ordering::SeqCst), 0);
            }
        }
    });
}

#[test]
fn overload_and_shutdown_do_not_enqueue_unserviceable_callers() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 2);
        let mut first = client.embed_bound(&cx, "10");
        let mut second = client.embed_bound(&cx, "20");
        pending(&mut first);
        pending(&mut second);
        assert!(matches!(
            client.embed_bound(&cx, "30").await,
            Err(SearchError::QueueFull {
                pending: 2,
                capacity: 2
            })
        ));
        assert!(matches!(
            client.embed_bound(&cx, &"x".repeat(129)).await,
            Err(SearchError::InvalidConfig { .. })
        ));
        assert_eq!(client.usage().outstanding_requests, 2);
        client.shutdown();
        assert!(!client.is_ready());
        assert!(matches!(
            client.embed_bound(&cx, "30").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(dispatcher.dispatch_ready(&cx).await.unwrap(), 2);
        assert!(first.await.is_ok() && second.await.is_ok());
        assert_eq!(client.usage(), CoalescerUsage::default());
    });
}

struct CountWake(AtomicUsize);
impl Wake for CountWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cancelling_one_waiting_client_wakes_it_without_cancelling_the_batch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 3);
        let cancelled = Cx::for_testing();
        let wake = Arc::new(CountWake(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&wake));
        let mut first = client.embed_bound(&cancelled, "10");
        assert!(
            first
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        let mut second = client.embed_bound(&cx, "20");
        pending(&mut second);
        cancelled.cancel_with(asupersync::CancelKind::User, Some("one client stopped"));
        assert!(wake.0.load(Ordering::SeqCst) > 0);
        assert!(matches!(first.await, Err(SearchError::Cancelled { .. })));
        assert!(!cx.is_cancel_requested());
        assert_eq!(dispatcher.dispatch_ready(&cx).await.unwrap(), 2);
        assert_eq!(second.await.unwrap().values[0], 20.0);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(client.usage(), CoalescerUsage::default());
    });
}

#[test]
fn original_provider_failures_and_cancellation_reach_the_correct_owners() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mode in [Mode::Failure, Mode::CancelSuccess, Mode::CancelFailure] {
            let provider = Arc::new(Probe::new(true, mode));
            let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 3);
            let request_cx = Cx::for_testing();
            let dispatch_cx = Cx::for_testing();
            let mut future = client.embed_bound(&request_cx, "10");
            pending(&mut future);
            let error = dispatcher.dispatch_ready(&dispatch_cx).await.unwrap_err();
            let received = future.await.unwrap_err();
            if matches!(mode, Mode::Failure) {
                assert!(
                    matches!(error, SearchError::EmbeddingFailed { ref source, .. } if source.downcast_ref::<Opaque>().is_some())
                );
                assert!(matches!(received, SearchError::EmbeddingFailed { .. }));
                assert_eq!(provider.formatted.load(Ordering::SeqCst), 0);
            } else {
                assert!(matches!(error, SearchError::Cancelled { .. }));
                assert!(matches!(received, SearchError::Cancelled { .. }));
            }
            assert!(!request_cx.is_cancel_requested());
            assert_eq!(client.usage(), CoalescerUsage::default());
        }
        // Cancellation of a recipient also outranks a value already delivered.
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let (client, mut dispatcher) = setup(&cx, provider, 3);
        let request_cx = Cx::for_testing();
        let mut future = client.embed_bound(&request_cx, "10");
        pending(&mut future);
        dispatcher.dispatch_ready(&cx).await.unwrap();
        request_cx.set_cancel_requested(true);
        assert!(matches!(future.await, Err(SearchError::Cancelled { .. })));
    });
}

#[test]
fn abandoning_clients_or_dispatch_releases_only_their_owned_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 3);
        drop(client.embed_bound(&cx, "10"));
        assert_eq!(client.usage(), CoalescerUsage::default());
        let mut gone = client.embed_bound(&cx, "10");
        pending(&mut gone);
        drop(gone);
        assert_eq!(client.usage().outstanding_requests, 1);
        assert_eq!(dispatcher.dispatch_ready(&cx).await.unwrap(), 1);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
        assert_eq!(client.usage(), CoalescerUsage::default());
        assert!(!cx.is_cancel_requested());

        provider.pending.store(true, Ordering::SeqCst);
        let mut request = client.embed_bound(&cx, "20");
        pending(&mut request);
        let mut dispatch = Box::pin(dispatcher.dispatch_ready(&cx));
        assert!(
            dispatch
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(dispatch);
        assert!(matches!(request.await, Err(SearchError::Cancelled { .. })));
        assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
        assert_eq!(client.usage(), CoalescerUsage::default());
        assert!(!cx.is_cancel_requested());

        provider.pending.store(false, Ordering::SeqCst);
        let mut retry = client.embed_bound(&cx, "30");
        pending(&mut retry);
        dispatcher.dispatch_ready(&cx).await.unwrap();
        assert_eq!(retry.await.unwrap().values[0], 30.0);
        let mut undispatched = client.embed_bound(&cx, "10");
        pending(&mut undispatched);
        drop(dispatcher);
        assert!(matches!(
            undispatched.await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(client.usage(), CoalescerUsage::default());
        assert!(!client.is_ready());
    });
}

#[test]
fn frozen_identity_refuses_constructor_mismatch_and_later_provider_drift() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let wrong = EmbeddingIdentityBundleV1::explicit_test_model("wrong", 2);
        assert!(matches!(
            CoalescingEmbedder::new(
                &cx,
                provider.clone(),
                wrong,
                CoalescerConfig::default(),
                CoalescerLimits::default()
            ),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        let mut storage = identity();
        storage.storage.quantization = QuantizationFormat::F16;
        assert!(
            CoalescingEmbedder::new(
                &cx,
                provider.clone(),
                storage,
                CoalescerConfig::default(),
                CoalescerLimits::default()
            )
            .is_err()
        );
        let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 3);
        let mut request = client.embed_bound(&cx, "10");
        pending(&mut request);
        provider.drift.store(true, Ordering::SeqCst);
        assert_eq!(client.identity().unwrap(), &identity());
        assert!(matches!(
            dispatcher.dispatch_ready(&cx).await,
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        assert!(matches!(
            request.await,
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        assert_eq!(provider.batches.load(Ordering::SeqCst), 0);
        assert_eq!(provider.raw.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn new_auto_coalesces_same_turn_concurrent_callers_into_single_batch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let client = CoalescingEmbedder::new_auto(
            &cx,
            Arc::clone(&provider) as Arc<dyn Embedder>,
            identity(),
            CoalescerConfig {
                max_batch_size: 4,
                max_wait_ms: 0,
                min_batch_size: 1,
                use_priority_lanes: true,
            },
            CoalescerLimits {
                max_outstanding_requests: 8,
                max_outstanding_text_bytes: 128,
            },
        )
        .unwrap();

        let mut f1 = client.embed_bound(&cx, "10");
        let mut f2 = client.embed_bound(&cx, "20");
        let mut f3 = client.embed_bound(&cx, "30");
        let mut s1 = None;
        let mut s2 = None;
        let mut s3 = None;
        std::future::poll_fn(|task_cx| {
            if s1.is_none()
                && let Poll::Ready(res) = f1.as_mut().poll(task_cx)
            {
                s1 = Some(res);
            }
            if s2.is_none()
                && let Poll::Ready(res) = f2.as_mut().poll(task_cx)
            {
                s2 = Some(res);
            }
            if s3.is_none()
                && let Poll::Ready(res) = f3.as_mut().poll(task_cx)
            {
                s3 = Some(res);
            }
            if s1.is_some() && s2.is_some() && s3.is_some() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await;

        assert_eq!(s1.unwrap().unwrap().values[0], 10.0);
        assert_eq!(s2.unwrap().unwrap().values[0], 20.0);
        assert_eq!(s3.unwrap().unwrap().values[0], 30.0);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(provider.singleton.load(Ordering::SeqCst), 0);
        assert_eq!(
            client.metrics().total_texts_batched.load(Ordering::Relaxed),
            3
        );

        let batch_out = client.embed_batch_bound(&cx, &["40", "50"]).await.unwrap();
        assert_eq!(batch_out.len(), 2);
        assert_eq!(batch_out[0].values[0], 40.0);
        assert_eq!(batch_out[1].values[0], 50.0);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 2);
        assert_eq!(
            client.metrics().total_texts_batched.load(Ordering::Relaxed),
            5
        );
    });
}

#[test]
fn abandoned_auto_admission_recovers_request_and_byte_capacity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (rows, bytes, count, next) in [
            (2, 128, 2, "30"),
            (8, 4, 2, "30"),
            (8, 12, 6, "123456789012"),
        ] {
            let provider = Arc::new(Probe::new(true, Mode::Good));
            let client = CoalescingEmbedder::new_auto(
                &cx,
                provider.clone(),
                identity(),
                CoalescerConfig {
                    max_batch_size: 3,
                    max_wait_ms: 10_000,
                    min_batch_size: 1,
                    use_priority_lanes: true,
                },
                CoalescerLimits {
                    max_outstanding_requests: rows,
                    max_outstanding_text_bytes: bytes,
                },
            )
            .unwrap();
            for round in 0..3 {
                let mut abandoned: Vec<_> = (0..count)
                    .map(|_| client.embed_bound(&cx, "10"))
                    .collect();
                // Stop during the admission yield, before any task owns
                // dispatch. Previously all later drivers failed at submission.
                for request in &mut abandoned {
                    pending(request);
                }
                assert_eq!(client.usage().outstanding_requests, count);
                assert_eq!(client.usage().outstanding_text_bytes, count * 2);
                drop(abandoned);
                assert_eq!(client.usage().outstanding_requests, count);
                let result = client.embed_bound(&cx, next).await.unwrap();
                assert_eq!(result.identity, identity());
                assert_eq!(
                    result.values[0].to_bits(),
                    next.parse::<f32>().unwrap().to_bits()
                );
                assert_eq!(provider.batches.load(Ordering::SeqCst), round + 1);
                assert_eq!(provider.raw.load(Ordering::SeqCst), 0);
                assert_eq!(client.usage(), CoalescerUsage::default());
            }
        }
    });
}

#[test]
fn abandoned_admission_does_not_discard_a_live_prefix_recipient() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let (_, dispatcher) = setup(&cx, Arc::clone(&provider), 2);
        let client = dispatcher.into_auto_embedder();
        let mut gone = client.embed_bound(&cx, "10");
        let mut live = client.embed_bound(&cx, "20");
        pending(&mut gone);
        pending(&mut live);
        drop(gone);
        // Reclaiming a full batch here would also consume the live request.
        assert!(matches!(
            client.embed_bound(&cx, "30").await,
            Err(SearchError::QueueFull {
                pending: 2,
                capacity: 2
            })
        ));
        assert_eq!(client.usage().outstanding_requests, 2);
        assert_eq!(
            live.await.unwrap().values[0].to_bits(),
            20.0_f32.to_bits()
        );
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(client.usage(), CoalescerUsage::default());
        assert_eq!(
            client.embed_bound(&cx, "30").await.unwrap().values[0].to_bits(),
            30.0_f32.to_bits()
        );
    });
}

#[test]
fn abandoned_front_reclamation_preserves_live_suffix_and_shutdown() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        let (_, dispatcher) = setup(&cx, Arc::clone(&provider), 4);
        let client = dispatcher.into_auto_embedder();
        let mut abandoned: Vec<_> = ["not-a-number", "also-abandoned", "never-infer"]
            .into_iter()
            .map(|text| client.embed_bound(&cx, text))
            .collect();
        for request in &mut abandoned {
            pending(request);
        }
        let mut live = client.embed_bound(&cx, "40");
        pending(&mut live);
        drop(abandoned);
        let result = client.embed_bound(&cx, "30").await.unwrap();
        assert_eq!(result.values[0].to_bits(), 30.0_f32.to_bits());
        assert_eq!(live.await.unwrap().values[0].to_bits(), 40.0_f32.to_bits());
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(client.usage(), CoalescerUsage::default());

        let mut gone = client.embed_bound(&cx, "10");
        pending(&mut gone);
        drop(gone);
        let before = client.usage();
        let cancelled = Cx::for_testing();
        cancelled.set_cancel_requested(true);
        assert!(matches!(
            client.embed_bound(&cancelled, "20").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(client.usage(), before);
        client.shutdown();
        assert!(matches!(
            client.embed_bound(&cx, "20").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(client.usage(), before);
        let shared = Arc::clone(&client.shared);
        drop(client);
        assert_eq!(shared.queue.usage(), CoalescerUsage::default());
    });
}

#[test]
fn dispatch_cancellation_wakes_an_uncooperative_provider_and_releases_capacity() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for native in [false, true] {
            for available in [false, true] {
                let provider = Arc::new(Probe::new(native, Mode::Good));
                provider.pending.store(true, Ordering::SeqCst);
                let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 2);
                let recipient_cx = Cx::for_testing();
                let owner_cx = Cx::for_testing();
                let mut first = client.embed_bound(&recipient_cx, "10");
                let mut second = client.embed_bound(&recipient_cx, "20");
                pending(&mut first);
                pending(&mut second);
                let wake = Arc::new(CountWake(AtomicUsize::new(0)));
                let waker = Waker::from(Arc::clone(&wake));
                let mut dispatch = Box::pin(async {
                    if available {
                        dispatcher.dispatch_available(&owner_cx).await
                    } else {
                        dispatcher.dispatch_ready(&owner_cx).await
                    }
                });
                assert!(
                    dispatch
                        .as_mut()
                        .poll(&mut Context::from_waker(&waker))
                        .is_pending()
                );
                assert_eq!(provider.drops.load(Ordering::SeqCst), 0);
                assert_eq!(client.usage().outstanding_requests, 2);
                let before = wake.0.load(Ordering::SeqCst);
                owner_cx.cancel_with(asupersync::CancelKind::User, Some("stop dispatcher"));
                assert!(wake.0.load(Ordering::SeqCst) > before);
                assert!(matches!(
                    dispatch.as_mut().poll(&mut Context::from_waker(&waker)),
                    Poll::Ready(Err(SearchError::Cancelled { .. }))
                ));
                drop(dispatch);
                assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
                assert_eq!(client.usage(), CoalescerUsage::default());
                for request in [first, second] {
                    assert!(matches!(
                        request.await,
                        Err(SearchError::Cancelled { .. })
                    ));
                }
                assert!(!recipient_cx.is_cancel_requested());
                assert!(!cx.is_cancel_requested());
                // Cancelling this dispatch is not shutdown or a provider retry.
                // Its released credits can serve a newly submitted batch.
                provider.pending.store(false, Ordering::SeqCst);
                let mut next = client.embed_bound(&recipient_cx, "30");
                pending(&mut next);
                assert_eq!(dispatcher.dispatch_available(&cx).await.unwrap(), 1);
                assert_eq!(
                    next.await.unwrap().values[0].to_bits(),
                    30.0_f32.to_bits()
                );
                assert_eq!(provider.raw.load(Ordering::SeqCst), 0);
                assert_eq!(client.usage(), CoalescerUsage::default());
            }
        }
    });
}

#[test]
fn pending_dispatch_observes_its_owner_not_a_cancelled_manual_recipient() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let provider = Arc::new(Probe::new(true, Mode::Good));
        provider.pending.store(true, Ordering::SeqCst);
        let (client, mut dispatcher) = setup(&cx, Arc::clone(&provider), 2);
        let cancelled_cx = Cx::for_testing();
        let mut first = client.embed_bound(&cancelled_cx, "10");
        let mut second = client.embed_bound(&cx, "20");
        pending(&mut first);
        pending(&mut second);
        let mut dispatch = Box::pin(dispatcher.dispatch_available(&cx));
        assert!(
            dispatch
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        cancelled_cx.cancel_with(asupersync::CancelKind::User, Some("one recipient stopped"));
        assert!(matches!(first.await, Err(SearchError::Cancelled { .. })));
        // The healthy recipient still owns its reply, and the service context
        // is unchanged. Its provider future must not have been abandoned.
        assert!(
            dispatch
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert_eq!(provider.drops.load(Ordering::SeqCst), 0);
        assert_eq!(provider.batches.load(Ordering::SeqCst), 1);
        assert_eq!(client.usage().outstanding_requests, 2);
        assert!(!cx.is_cancel_requested());
        drop(dispatch);
        assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
        assert!(matches!(second.await, Err(SearchError::Cancelled { .. })));
        assert_eq!(client.usage(), CoalescerUsage::default());
        assert!(!cx.is_cancel_requested());
    });
}
