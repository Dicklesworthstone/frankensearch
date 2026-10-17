use std::future::Future;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use asupersync::test_utils::run_test_with_cx;

use super::super::{
    CoalescerConfig, CoalescerLimits, CoalescingEmbedder, EmbeddingIdentityBundleV1,
    Embedder, ModelCategory, SearchError,
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
            assert!(self.native, "a bound-single override must stay on that path");
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
    let (manual, dispatcher) = CoalescingEmbedder::new(
        cx,
        provider,
        identity("budget-test"),
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
    .unwrap();
    (manual, dispatcher.into_auto_embedder())
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
            assert!(!manual.bound_batch_is_native());
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
