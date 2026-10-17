use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use asupersync::test_utils::run_test_with_cx;

use super::super::{
    CoalescerConfig, CoalescerLimits, CoalescingEmbedder, EmbeddingIdentityBundleV1,
    Embedder, ModelCategory, SearchError, YieldOnce, identity_mismatch,
};
use super::*;

struct Dropped(Arc<AtomicUsize>);

impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn identity() -> EmbeddingIdentityBundleV1 {
    EmbeddingIdentityBundleV1::explicit_test_model("coalescing-batch-test", 2)
}

fn value(value: f32) -> IdentityBoundEmbedding {
    IdentityBoundEmbedding {
        identity: identity(),
        values: vec![value, -0.0],
    }
}

#[test]
fn a_known_failure_drops_pending_siblings_without_polling_later_work() {
    run_test_with_cx(|cx| async move {
        let drops = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::clone(&drops);
        let futures: Vec<SearchFuture<'_, IdentityBoundEmbedding>> = vec![
            Box::pin(async move {
                let _guard = Dropped(dropped);
                std::future::pending().await
            }),
            Box::pin(async { Err(identity_mismatch()) }),
            Box::pin(async { panic!("later work must not start after a known failure") }),
        ];
        let mut batch = Box::pin(collect(&cx, futures));
        assert!(matches!(
            batch.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(SearchError::UnverifiableRemoteSpace { .. }))
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!cx.is_cancel_requested());
    });
}

#[test]
fn successes_remain_in_input_order_and_completed_futures_are_not_repolled() {
    run_test_with_cx(|cx| async move {
        let calls = AtomicUsize::new(0);
        let futures: Vec<SearchFuture<'_, IdentityBoundEmbedding>> = vec![
            Box::pin(async {
                YieldOnce(false).await;
                Ok(value(10.0))
            }),
            Box::pin(async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(value(20.0))
            }),
        ];
        let mut batch = Box::pin(collect(&cx, futures));
        let mut context = Context::from_waker(Waker::noop());
        assert!(batch.as_mut().poll(&mut context).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let Poll::Ready(Ok(values)) = batch.as_mut().poll(&mut context) else {
            panic!("both results must be available on the second poll");
        };
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(values[0].values[0].to_bits(), 10.0_f32.to_bits());
        assert_eq!(values[1].values[0].to_bits(), 20.0_f32.to_bits());
        assert_eq!(values[0].values[1].to_bits(), (-0.0_f32).to_bits());
    });
}

#[test]
fn cancellation_during_cleanup_dominates_the_observed_error() {
    struct CancelOnDrop(Cx);
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0.set_cancel_requested(true);
        }
    }
    run_test_with_cx(|cx| async move {
        let child_cx = cx.clone();
        let futures: Vec<SearchFuture<'_, IdentityBoundEmbedding>> = vec![
            Box::pin(async move {
                let _guard = CancelOnDrop(child_cx);
                std::future::pending().await
            }),
            Box::pin(async { Err(identity_mismatch()) }),
        ];
        assert!(matches!(collect(&cx, futures).await, Err(SearchError::Cancelled { .. })));
    });
}

#[test]
fn cancellation_precedes_empty_or_unstarted_groups() {
    run_test_with_cx(|cx| async move {
        assert!(collect(&cx, Vec::new()).await.unwrap().is_empty());
        cx.set_cancel_requested(true);
        assert!(matches!(collect(&cx, Vec::new()).await, Err(SearchError::Cancelled { .. })));
        let futures: Vec<SearchFuture<'_, IdentityBoundEmbedding>> = vec![Box::pin(async {
            panic!("cancelled group must not poll a child")
        })];
        assert!(matches!(collect(&cx, futures).await, Err(SearchError::Cancelled { .. })));
    });
}

#[test]
fn dropping_a_pending_group_releases_every_polled_child() {
    run_test_with_cx(|cx| async move {
        let drops = Arc::new(AtomicUsize::new(0));
        let futures: Vec<SearchFuture<'_, IdentityBoundEmbedding>> = (0..3)
            .map(|_| {
                let drops = Arc::clone(&drops);
                Box::pin(async move {
                    let _guard = Dropped(drops);
                    std::future::pending().await
                }) as SearchFuture<'_, IdentityBoundEmbedding>
            })
            .collect();
        let mut batch = Box::pin(collect(&cx, futures));
        assert!(batch.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        drop(batch);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        assert!(!cx.is_cancel_requested());
    });
}

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        panic!("the adapter must never use raw inference")
    }

    fn embed_bound<'a>(
        &'a self,
        _: &'a Cx,
        _: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::pending())
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        "coalescing-batch-test"
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

#[test]
fn automatic_batch_reports_queue_refusal_without_waiting_for_accepted_siblings() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider {
            identity: identity(),
            calls: AtomicUsize::new(0),
        });
        let client = CoalescingEmbedder::new_auto(
            &cx,
            provider.clone(),
            identity(),
            CoalescerConfig {
                max_batch_size: 3,
                min_batch_size: 1,
                max_wait_ms: 0,
                use_priority_lanes: true,
            },
            CoalescerLimits {
                max_outstanding_requests: 3,
                max_outstanding_text_bytes: 128,
            },
        )
        .unwrap();
        // An unrelated admitted client consumes one slot. Two batch members
        // enqueue and yield; the third must expose the capacity refusal now.
        let mut receiver = client.submit(&cx, "outside").unwrap();
        let texts = ["one", "two", "three"];
        let mut batch = client.embed_batch_bound(&cx, &texts);
        assert!(matches!(
            batch.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(SearchError::QueueFull { .. }))
        ));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        {
            let mut unrelated = Box::pin(receiver.recv(&cx));
            assert!(unrelated.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        }
        assert!(!client.shared.lock().unwrap().closed);
        assert!(!cx.is_cancel_requested());
        drop(batch);
        let shared = Arc::clone(&client.shared);
        drop(client);
        assert_eq!(shared.queue.usage().outstanding_requests, 0);
    });
}
