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
            assert!(!client.bound_batch_is_native());
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
