use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use asupersync::channel::oneshot;
use asupersync::test_utils::run_test_with_cx;

use super::super::{
    CoalescerConfig, CoalescerLimits, CoalescerUsage, CoalescingDispatcher, CoalescingEmbedder,
    Embedder, EmbeddingIdentityBundleV1, ModelCategory, SearchError,
};
use super::*;

fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
    EmbeddingIdentityBundleV1::explicit_test_model(name, 2)
}

fn expected(text: &str) -> Vec<f32> {
    let sum: u16 = text.bytes().map(u16::from).sum();
    vec![f32::from(sum) + 1.0, -0.0]
}

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    native: bool,
    singles: AtomicUsize,
    widths: Mutex<Vec<usize>>,
}

impl Provider {
    fn new(native: bool) -> Self {
        Self {
            identity: identity("budget-test"),
            native,
            singles: AtomicUsize::new(0),
            widths: Mutex::new(Vec::new()),
        }
    }

    fn response(&self, text: &str) -> IdentityBoundEmbedding {
        IdentityBoundEmbedding {
            identity: if text == "bad" {
                identity("foreign")
            } else {
                self.identity.clone()
            },
            values: expected(text),
        }
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        panic!("custom bound-single inference must never become raw inference")
    }

    fn embed_bound<'a>(
        &'a self,
        _: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.singles.fetch_add(1, Ordering::SeqCst);
            Ok(self.response(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        _: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            assert!(
                self.native,
                "a bound-single override must stay on that path"
            );
            self.widths.lock().unwrap().push(texts.len());
            Ok(texts.iter().map(|text| self.response(text)).collect())
        })
    }

    fn bound_batch_is_native(&self) -> bool {
        self.native
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        "budget-test"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

fn clients(
    cx: &Cx,
    provider: Arc<Provider>,
    rows: usize,
    bytes: usize,
) -> (CoalescingEmbedder, CoalescingEmbedder) {
    let (manual, dispatcher) = manual_clients(cx, provider, rows, bytes);
    (manual, dispatcher.into_auto_embedder())
}

fn manual_clients(
    cx: &Cx,
    provider: Arc<Provider>,
    rows: usize,
    bytes: usize,
) -> (CoalescingEmbedder, CoalescingDispatcher) {
    let expected = provider.identity.clone();
    CoalescingEmbedder::new(
        cx,
        provider,
        expected,
        CoalescerConfig {
            max_batch_size: rows,
            min_batch_size: 1,
            max_wait_ms: 0,
            use_priority_lanes: true,
        },
        CoalescerLimits {
            max_outstanding_requests: rows,
            max_outstanding_text_bytes: bytes,
        },
    )
    .unwrap()
}

// Exercise the public, trait-erased batch path and the real service loop on
// the same caller-owned task. Keep each completed future out of later polls.
// Signal graceful shutdown only after the request returns. A service failure
// remains an error even when it arrives before that signal can be observed.
async fn drive_manual_batch(
    cx: &Cx,
    client: &dyn Embedder,
    dispatcher: &mut CoalescingDispatcher,
    texts: &[&str],
) -> (SearchResult<Vec<IdentityBoundEmbedding>>, SearchResult<usize>) {
    let mut request = client.embed_batch_bound(cx, texts);
    let (stop, mut stopped) = oneshot::channel::<()>();
    let mut stop = Some(stop);
    let mut service = Box::pin(dispatcher.run_until(cx, async {
        let _ = stopped.recv(cx).await;
    }));
    let mut output = None;
    let mut service_result = None;
    std::future::poll_fn(|task_cx| {
        if output.is_none()
            && let Poll::Ready(result) = request.as_mut().poll(task_cx)
        {
            output = Some(result);
            let _ = stop.take().unwrap().send_blocking(());
        }
        if service_result.is_none()
            && let Poll::Ready(result) = service.as_mut().poll(task_cx)
        {
            service_result = Some(result);
        }
        if output.is_some() && service_result.is_some() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;
    (output.unwrap(), service_result.unwrap())
}

#[test]
fn grouping_enforces_exact_utf8_bytes_rows_and_nonempty_progress() {
    run_test_with_cx(|cx| async move {
        let texts = ["é", "abc", "中", "z", ""];
        assert_eq!(prefix_len(&cx, &texts, 4, 5).unwrap(), 2);
        assert_eq!(prefix_len(&cx, &texts[2..], 4, 5).unwrap(), 3);
        assert_eq!(prefix_len(&cx, &texts, 1, usize::MAX).unwrap(), 1);
        assert_eq!(prefix_len(&cx, &["", ""], 1, 0).unwrap(), 1);
        assert_eq!(prefix_len(&cx, &[], 0, 0).unwrap(), 0);
        assert!(prefix_len(&cx, &texts, 0, 100).is_err());
        assert!(prefix_len(&cx, &texts, 4, 1).is_err());
    });
}

#[test]
fn automatic_batch_spans_byte_bounded_groups_in_original_order() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new(true));
        let (_, client) = clients(&cx, Arc::clone(&provider), 4, 5);
        let texts = ["é", "abc", "中", "z", ""];
        let values = client.embed_batch_bound(&cx, &texts).await.unwrap();
        assert_eq!(values.len(), texts.len());
        for (response, text) in values.iter().zip(texts) {
            assert_eq!(response.identity, provider.identity);
            assert_eq!(response.values[0].to_bits(), expected(text)[0].to_bits());
            assert_eq!(response.values[1].to_bits(), (-0.0_f32).to_bits());
        }
        assert_eq!(*provider.widths.lock().unwrap(), [2, 3]);
        assert_eq!(provider.singles.load(Ordering::SeqCst), 0);
        assert_eq!(client.usage().outstanding_requests, 0);
        assert_eq!(client.usage().outstanding_text_bytes, 0);
    });
}

#[test]
fn advertised_auto_batch_keeps_native_and_bound_single_provider_dispatch() {
    run_test_with_cx(|cx| async move {
        for native in [false, true] {
            let provider = Arc::new(Provider::new(native));
            let (manual, client) = clients(&cx, Arc::clone(&provider), 2, 128);
            assert!(manual.bound_batch_is_native());
            assert!(client.bound_batch_is_native());
            let texts = ["a", "b", "c", "d", "e"];
            // Follow the same trait capability decision used by bound-aware
            // indexers, rather than calling a private adapter entry point.
            let producer: &dyn Embedder = &client;
            assert!(producer.bound_batch_is_native());
            let result = producer.embed_batch_bound(&cx, &texts).await.unwrap();
            assert_eq!(result.len(), 5);
            if native {
                assert_eq!(*provider.widths.lock().unwrap(), [2, 2, 1]);
                assert_eq!(provider.singles.load(Ordering::SeqCst), 0);
            } else {
                assert!(provider.widths.lock().unwrap().is_empty());
                assert_eq!(provider.singles.load(Ordering::SeqCst), 5);
            }
            assert_eq!(client.usage().outstanding_requests, 0);
        }
    });
}

#[test]
fn an_oversized_later_input_is_refused_before_admitting_any_prefix() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new(true));
        let (manual, client) = clients(&cx, Arc::clone(&provider), 2, 5);
        let texts = ["a", "b", "too large"];
        for handle in [&manual, &client] {
            let mut work = handle.embed_batch_bound(&cx, &texts);
            assert!(matches!(
                work.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(SearchError::InvalidConfig { field, .. }))
                    if field == "coalescing_embedder.input"
            ));
            assert_eq!(handle.usage().outstanding_requests, 0);
        }
        assert!(provider.widths.lock().unwrap().is_empty());
        assert_eq!(provider.singles.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn a_foreign_response_in_a_later_group_discards_the_whole_call() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new(true));
        let (_, client) = clients(&cx, Arc::clone(&provider), 2, 128);
        let texts = ["a", "b", "c", "bad", "not-started"];
        assert!(matches!(
            client.embed_batch_bound(&cx, &texts).await,
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        assert_eq!(*provider.widths.lock().unwrap(), [2, 2]);
        assert_eq!(client.usage().outstanding_requests, 0);
        assert!(!cx.is_cancel_requested());
    });
}

#[test]
fn cancellation_precedes_batch_preflight_and_empty_success() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new(true));
        let (_, client) = clients(&cx, Arc::clone(&provider), 2, 5);
        cx.set_cancel_requested(true);
        for texts in [&[][..], &["too large"][..]] {
            assert!(matches!(
                client.embed_batch_bound(&cx, texts).await,
                Err(SearchError::Cancelled { .. })
            ));
        }
        assert!(provider.widths.lock().unwrap().is_empty());
        assert_eq!(client.usage().outstanding_requests, 0);
    });
}

#[test]
fn manual_service_batches_full_groups_without_changing_provider_contracts() {
    run_test_with_cx(|cx| async move {
        let texts = ["é", "abc", "中", "z", ""];
        for native in [false, true] {
            for (rows, bytes, expected_widths) in [
                (2, 128, vec![2, 2, 1]),
                (4, 5, vec![2, 3]),
            ] {
                let provider = Arc::new(Provider::new(native));
                let (client, mut dispatcher) =
                    manual_clients(&cx, Arc::clone(&provider), rows, bytes);
                let producer: &dyn Embedder = &client;
                assert!(producer.bound_batch_is_native());
                let (output, discharged) =
                    drive_manual_batch(&cx, producer, &mut dispatcher, &texts).await;
                let output = output.unwrap();
                assert_eq!(discharged.unwrap(), texts.len());
                assert_eq!(output.len(), texts.len());
                for (output, text) in output.iter().zip(texts) {
                    assert_eq!(output.identity, provider.identity);
                    assert_eq!(output.values[0].to_bits(), expected(text)[0].to_bits());
                    assert_eq!(output.values[1].to_bits(), (-0.0_f32).to_bits());
                }
                if native {
                    assert_eq!(*provider.widths.lock().unwrap(), expected_widths);
                    assert_eq!(provider.singles.load(Ordering::SeqCst), 0);
                } else {
                    assert!(provider.widths.lock().unwrap().is_empty());
                    assert_eq!(provider.singles.load(Ordering::SeqCst), texts.len());
                }
                assert_eq!(client.usage(), CoalescerUsage::default());
                assert!(!cx.is_cancel_requested());
            }
        }
    });
}

#[test]
fn manual_service_failure_discards_prior_groups_and_never_submits_the_suffix() {
    run_test_with_cx(|cx| async move {
        for native in [false, true] {
            let provider = Arc::new(Provider::new(native));
            let (client, mut dispatcher) =
                manual_clients(&cx, Arc::clone(&provider), 2, 128);
            let texts = ["a", "b", "c", "bad", "not-started"];
            let (output, service) =
                drive_manual_batch(&cx, &client, &mut dispatcher, &texts).await;
            assert!(matches!(
                output,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            assert!(matches!(
                service,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
            if native {
                assert_eq!(*provider.widths.lock().unwrap(), [2, 2]);
                assert_eq!(provider.singles.load(Ordering::SeqCst), 0);
            } else {
                assert!(provider.widths.lock().unwrap().is_empty());
                assert_eq!(provider.singles.load(Ordering::SeqCst), 4);
            }
            assert_eq!(client.usage(), CoalescerUsage::default());
            assert!(!cx.is_cancel_requested());
        }
    });
}

#[test]
fn manual_batch_overload_does_not_retry_or_displace_an_existing_recipient() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new(true));
        let (client, mut dispatcher) = manual_clients(&cx, Arc::clone(&provider), 2, 128);
        let mut existing = client.embed_bound(&cx, "a");
        assert!(
            existing
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        let texts = ["b", "c"];
        let mut batch = client.embed_batch_bound(&cx, &texts);
        assert!(matches!(
            batch.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(SearchError::QueueFull {
                pending: 2,
                capacity: 2
            }))
        ));
        drop(batch);
        // The abandoned b slot still owns input credit until the service
        // drains it. The existing a recipient is neither cancelled nor evicted.
        assert_eq!(client.usage().outstanding_requests, 2);
        assert!(provider.widths.lock().unwrap().is_empty());
        client.shutdown();
        assert_eq!(dispatcher.run(&cx).await.unwrap(), 2);
        assert_eq!(existing.await.unwrap().values, expected("a"));
        assert_eq!(*provider.widths.lock().unwrap(), [2]);
        assert_eq!(client.usage(), CoalescerUsage::default());
        assert!(!cx.is_cancel_requested());
    });
}

#[test]
fn cancelling_or_dropping_a_manual_batch_keeps_the_service_and_peer_healthy() {
    run_test_with_cx(|cx| async move {
        for cancel in [false, true] {
            let provider = Arc::new(Provider::new(true));
            let (client, mut dispatcher) =
                manual_clients(&cx, Arc::clone(&provider), 4, 128);
            let request_cx = Cx::for_testing();
            let texts = ["a", "b", "c"];
            drop(client.embed_batch_bound(&request_cx, &texts));
            assert_eq!(client.usage(), CoalescerUsage::default());
            let mut batch = client.embed_batch_bound(&request_cx, &texts);
            assert!(
                batch
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(client.usage().outstanding_requests, 3);
            if cancel {
                request_cx.cancel_with(asupersync::CancelKind::User, Some("stop batch caller"));
                assert!(matches!(batch.await, Err(SearchError::Cancelled { .. })));
            } else {
                drop(batch);
            }
            assert_eq!(client.usage().outstanding_requests, 3);
            let mut peer = client.embed_bound(&cx, "z");
            assert!(
                peer.as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            client.shutdown();
            assert_eq!(dispatcher.run(&cx).await.unwrap(), 4);
            assert_eq!(peer.await.unwrap().values, expected("z"));
            assert_eq!(*provider.widths.lock().unwrap(), [4]);
            assert_eq!(client.usage(), CoalescerUsage::default());
            assert!(!cx.is_cancel_requested());
        }
    });
}

#[test]
fn an_abandoned_manual_group_is_drained_without_inference() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new(true));
        let (client, mut dispatcher) = manual_clients(&cx, Arc::clone(&provider), 3, 128);
        let texts = ["a", "b", "c"];
        let mut batch = client.embed_batch_bound(&cx, &texts);
        assert!(
            batch
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert_eq!(client.usage().outstanding_requests, 3);
        drop(batch);
        client.shutdown();
        assert_eq!(dispatcher.run(&cx).await.unwrap(), 3);
        assert!(provider.widths.lock().unwrap().is_empty());
        assert_eq!(provider.singles.load(Ordering::SeqCst), 0);
        assert_eq!(client.usage(), CoalescerUsage::default());
    });
}

#[test]
fn trait_erased_stack_tiers_keep_batching_and_independent_service_shutdown() {
    run_test_with_cx(|cx| async move {
        let fast = Arc::new(Provider::new(true));
        let mut quality = Provider::new(false);
        // Same operational label, different full contracts and service owners.
        quality.identity = identity("quality-budget-test");
        let quality = Arc::new(quality);
        let stack = crate::auto_detect::EmbedderStack::from_parts(
            fast.clone(),
            Some(quality.clone()),
        );
        let (stack, mut fast_service, quality_service) = stack
            .into_coalescing(
                &cx,
                CoalescerConfig {
                    max_batch_size: 2,
                    ..CoalescerConfig::default()
                },
                CoalescerLimits::default(),
            )
            .unwrap();
        let mut quality_service = quality_service.unwrap();
        let texts = ["a", "b", "c", "d", "e"];
        assert!(stack.fast().bound_batch_is_native());
        assert!(stack.quality().unwrap().bound_batch_is_native());
        let (fast_results, discharged) =
            drive_manual_batch(&cx, stack.fast(), &mut fast_service, &texts).await;
        let fast_results = fast_results.unwrap();
        assert_eq!(discharged.unwrap(), texts.len());
        assert!(!stack.fast().is_ready());
        assert!(stack.quality().unwrap().is_ready());
        let (quality_results, discharged) = drive_manual_batch(
            &cx,
            stack.quality().unwrap(),
            &mut quality_service,
            &texts,
        )
        .await;
        let quality_results = quality_results.unwrap();
        assert_eq!(discharged.unwrap(), texts.len());
        assert_eq!(fast_results.len(), texts.len());
        assert_eq!(quality_results.len(), texts.len());
        for ((fast_result, quality_result), text) in
            fast_results.iter().zip(&quality_results).zip(texts)
        {
            assert_eq!(fast_result.identity, fast.identity);
            assert_eq!(quality_result.identity, quality.identity);
            assert_ne!(fast_result.identity, quality_result.identity);
            assert_eq!(fast_result.values, expected(text));
            assert_eq!(quality_result.values, expected(text));
        }
        assert_eq!(*fast.widths.lock().unwrap(), [2, 2, 1]);
        assert_eq!(fast.singles.load(Ordering::SeqCst), 0);
        assert!(quality.widths.lock().unwrap().is_empty());
        assert_eq!(quality.singles.load(Ordering::SeqCst), texts.len());
        assert!(!cx.is_cancel_requested());
    });
}
