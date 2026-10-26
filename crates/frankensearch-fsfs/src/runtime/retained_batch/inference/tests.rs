use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::{Embedder, EmbeddingIdentityBundleV1, ModelCategory, SearchFuture};

use super::*;

#[derive(Clone, Copy)]
enum Fault {
    None,
    Failure,
    Count,
    Width,
    NonFinite,
    Foreign,
    Drift,
    DriftFailure,
    Dispatch,
    Cancel,
    TypedCancel,
    Pending,
    PendingAfterFirst,
    CancelFactory,
}

struct Backend {
    identity: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    native: AtomicBool,
    drift: AtomicBool,
    fault: Fault,
    calls: Mutex<Vec<Vec<String>>>,
    raw_calls: AtomicUsize,
    dropped: AtomicUsize,
}

impl Backend {
    fn new(native: bool, fault: Fault) -> Self {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model("group-fixture", 3);
        let mut foreign = identity.clone();
        foreign
            .producer
            .implementation_revision
            .push_str("-private-canary");
        Self {
            identity,
            foreign,
            native: AtomicBool::new(native),
            drift: AtomicBool::new(false),
            fault,
            calls: Mutex::new(Vec::new()),
            raw_calls: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }

    fn values(text: &str) -> Vec<f32> {
        vec![text.bytes().next().map_or(0.0, f32::from), -0.0, 0.5]
    }

    fn widths(&self) -> Vec<usize> {
        self.calls.lock().unwrap().iter().map(Vec::len).collect()
    }

    async fn respond(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        let call_count = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(texts.iter().map(|text| (*text).to_owned()).collect());
            calls.len()
        };
        match self.fault {
            Fault::Dispatch => {
                self.native.fetch_xor(true, Ordering::SeqCst);
            }
            Fault::Drift | Fault::DriftFailure => self.drift.store(true, Ordering::SeqCst),
            Fault::Cancel => cx.set_cancel_requested(true),
            Fault::TypedCancel => {
                return Err(SearchError::Cancelled {
                    phase: "provider.terminal".to_owned(),
                    reason: "original cancellation".to_owned(),
                });
            }
            Fault::Pending | Fault::PendingAfterFirst
                if matches!(self.fault, Fault::Pending) || call_count > 1 =>
            {
                struct Guard<'a>(&'a AtomicUsize);
                impl Drop for Guard<'_> {
                    fn drop(&mut self) {
                        self.0.fetch_add(1, Ordering::SeqCst);
                    }
                }
                let _guard = Guard(&self.dropped);
                return std::future::pending().await;
            }
            _ => {}
        }
        if matches!(self.fault, Fault::Failure | Fault::DriftFailure) {
            return Err(SearchError::EmbeddingFailed {
                model: "original-provider".to_owned(),
                source: "original failure".into(),
            });
        }
        let mut rows = texts
            .iter()
            .map(|text| IdentityBoundEmbedding {
                values: Self::values(text),
                identity: self.identity.clone(),
            })
            .collect::<Vec<_>>();
        match self.fault {
            Fault::Count => {
                let _ = rows.pop();
            }
            Fault::Width => {
                let _ = rows.last_mut().unwrap().values.pop();
            }
            Fault::NonFinite => rows.last_mut().unwrap().values[0] = f32::NAN,
            Fault::Foreign => rows.last_mut().unwrap().identity.clone_from(&self.foreign),
            _ => {}
        }
        Ok(rows)
    }
}

impl Embedder for Backend {
    fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![99.0, 99.0, 99.0])
        })
    }
    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move { Ok(self.respond(cx, &[text]).await?.pop().unwrap()) })
    }
    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        if matches!(self.fault, Fault::CancelFactory) {
            struct Unpolled<'a>(&'a AtomicUsize);
            impl Future for Unpolled<'_> {
                type Output = SearchResult<Vec<IdentityBoundEmbedding>>;

                fn poll(
                    self: std::pin::Pin<&mut Self>,
                    _: &mut std::task::Context<'_>,
                ) -> Poll<Self::Output> {
                    panic!("cancelled provider factory's future was polled");
                }
            }
            impl Drop for Unpolled<'_> {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
            cx.cancel_with(asupersync::CancelKind::User, Some("provider factory"));
            return Box::pin(Unpolled(&self.dropped));
        }
        Box::pin(async move {
            assert!(
                self.native.load(Ordering::SeqCst),
                "unadvertised batch dispatch"
            );
            self.respond(cx, texts).await
        })
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.drift.load(Ordering::SeqCst) {
            &self.foreign
        } else {
            &self.identity
        })
    }
    fn bound_batch_is_native(&self) -> bool {
        self.native.load(Ordering::SeqCst)
    }
    fn dimension(&self) -> usize {
        3
    }
    fn id(&self) -> &'static str {
        "group-fixture"
    }
    fn model_name(&self) -> &'static str {
        "Bound group fixture"
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

fn producer(backend: &Arc<Backend>) -> Producer {
    Producer {
        embedder: backend.clone(),
        identity: backend.identity.clone(),
        role: "test.group",
    }
}

#[test]
fn native_groups_cross_document_boundaries_and_preserve_slot_bits() {
    run_test_with_cx(|cx| async move {
        let backend = Arc::new(Backend::new(true, Fault::None));
        let owner = producer(&backend);
        let texts = (0..130)
            .map(|n| format!("{} duplicate body", n % 3))
            .collect::<Vec<_>>();
        let mut batch = InferenceBatch::new(&owner);
        let mut bytes = 0;
        for (n, text) in texts.iter().enumerate() {
            batch
                .push(&cx, format!("doc-{n}"), text, &mut bytes)
                .await
                .unwrap();
        }
        let output = batch.finish(&cx, &mut bytes).await.unwrap();
        assert_eq!(backend.widths(), [64, 64, 2]);
        assert_eq!(bytes, 130 * 3 * 4);
        assert_eq!(output.len(), texts.len());
        for (n, ((id, vector), text)) in output.iter().zip(&texts).enumerate() {
            assert_eq!(id, &format!("doc-{n}"));
            assert_eq!(
                vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                Backend::values(text)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn custom_bound_single_is_never_replaced_by_an_unadvertised_batch() {
    run_test_with_cx(|cx| async move {
        let backend = Arc::new(Backend::new(false, Fault::None));
        let owner = producer(&backend);
        let mut batch = InferenceBatch::new(&owner);
        let mut bytes = 0;
        for n in 0..65 {
            batch
                .push(&cx, format!("doc-{n}"), "same", &mut bytes)
                .await
                .unwrap();
        }
        let output = batch.finish(&cx, &mut bytes).await.unwrap();
        assert_eq!(backend.widths(), vec![1; 65]);
        assert_eq!(output.len(), 65);
        assert!(
            output
                .iter()
                .all(|(_, vector)| vector == &Backend::values("same"))
        );
        assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn bad_native_response_stages_no_prefix_and_is_never_retried() {
    run_test_with_cx(|cx| async move {
        for fault in [
            Fault::Count,
            Fault::Width,
            Fault::NonFinite,
            Fault::Foreign,
            Fault::Failure,
        ] {
            let backend = Arc::new(Backend::new(true, fault));
            let owner = producer(&backend);
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 12; // a completed group belonging to the other tier
            batch
                .push(&cx, "first".to_owned(), "one", &mut bytes)
                .await
                .unwrap();
            batch
                .push(&cx, "last".to_owned(), "two", &mut bytes)
                .await
                .unwrap();
            let error = batch.flush(&cx, &mut bytes).await.unwrap_err();
            assert!(!error.to_string().contains("private-canary"));
            assert_eq!(bytes, 12);
            assert!(batch.output.is_empty());
            assert_eq!(batch.ids, ["first", "last"]);
            assert_eq!(backend.widths(), [2]);
            assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn producer_or_dispatch_drift_is_terminal_even_during_inference_failure() {
    run_test_with_cx(|cx| async move {
        for native in [false, true] {
            for fault in [Fault::Dispatch, Fault::Drift, Fault::DriftFailure] {
                let backend = Arc::new(Backend::new(native, fault));
                let owner = producer(&backend);
                let mut batch = InferenceBatch::new(&owner);
                let mut bytes = 0;
                batch
                    .push(&cx, "first".to_owned(), "one", &mut bytes)
                    .await
                    .unwrap();
                batch
                    .push(&cx, "last".to_owned(), "two", &mut bytes)
                    .await
                    .unwrap();
                assert!(matches!(
                    batch.flush(&cx, &mut bytes).await,
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ));
                assert_eq!(backend.widths(), [if native { 2 } else { 1 }]);
                assert!(batch.output.is_empty());
                assert_eq!(bytes, 0);
            }
        }
    });
}

#[test]
fn pre_cancel_and_late_cancel_never_stage_a_successful_group() {
    run_test_with_cx(|cx| async move {
        for native in [false, true] {
            let backend = Arc::new(Backend::new(native, Fault::None));
            let owner = producer(&backend);
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 0;
            cx.set_cancel_requested(true);
            assert!(matches!(
                batch.push(&cx, "a".to_owned(), "one", &mut bytes).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(backend.widths().is_empty());
            for fault in [Fault::Cancel, Fault::TypedCancel] {
                let backend = Arc::new(Backend::new(native, fault));
                let owner = producer(&backend);
                let mut batch = InferenceBatch::new(&owner);
                batch
                    .push(&cx, "a".to_owned(), "one", &mut bytes)
                    .await
                    .unwrap();
                let error = batch.flush(&cx, &mut bytes).await.unwrap_err();
                cx.set_cancel_requested(false);
                assert!(matches!(error, SearchError::Cancelled { .. }));
                if matches!(fault, Fault::TypedCancel) {
                    assert!(matches!(error, SearchError::Cancelled { phase, reason }
                        if phase == "provider.terminal" && reason == "original cancellation"));
                }
                assert_eq!(bytes, 0);
                assert!(batch.output.is_empty());
                assert_eq!(backend.widths(), [1]);
            }
        }
    });
}

#[test]
fn grouped_vector_budget_is_checked_before_inference_and_shared_by_tiers() {
    run_test_with_cx(|cx| async move {
        let fast = Arc::new(Backend::new(true, Fault::None));
        let quality = Arc::new(Backend::new(true, Fault::None));
        let fast_owner = producer(&fast);
        let quality_owner = producer(&quality);
        let mut fast_batch = InferenceBatch::new(&fast_owner);
        let mut quality_batch = InferenceBatch::new(&quality_owner);
        let mut bytes = MAX_VECTOR_BYTES - 12;
        fast_batch
            .push(&cx, "a".to_owned(), "fast", &mut bytes)
            .await
            .unwrap();
        quality_batch
            .push(&cx, "a".to_owned(), "quality", &mut bytes)
            .await
            .unwrap();
        assert_eq!(fast_batch.finish(&cx, &mut bytes).await.unwrap().len(), 1);
        assert_eq!(bytes, MAX_VECTOR_BYTES);
        assert!(quality_batch.finish(&cx, &mut bytes).await.is_err());
        assert!(quality.widths().is_empty());
        assert_eq!(fast.widths(), [1]);
        assert_eq!(bytes, MAX_VECTOR_BYTES);
    });
}

#[test]
fn text_byte_boundaries_split_groups_but_do_not_truncate_large_single_inputs() {
    run_test_with_cx(|cx| async move {
        for large in [false, true] {
            let backend = Arc::new(Backend::new(true, Fault::None));
            let owner = producer(&backend);
            let text = "x".repeat(if large {
                MAX_TEXT_BYTES + 1
            } else {
                MAX_TEXT_BYTES / 2
            });
            let texts = if large {
                [text.as_str(), "small", "small"]
            } else {
                [text.as_str(), text.as_str(), text.as_str()]
            };
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 0;
            for (n, text) in texts.iter().enumerate() {
                batch
                    .push(&cx, n.to_string(), text, &mut bytes)
                    .await
                    .unwrap();
            }
            assert_eq!(batch.finish(&cx, &mut bytes).await.unwrap().len(), 3);
            assert_eq!(
                backend.widths(),
                if large { vec![1, 2] } else { vec![2, 1] }
            );
            let calls = backend.calls.lock().unwrap();
            assert_eq!(calls[0][0], text);
        }
    });
}

#[test]
fn dropping_a_pending_group_drops_its_provider_future_without_orphan_work() {
    run_test_with_cx(|cx| async move {
        for native in [false, true] {
            let backend = Arc::new(Backend::new(native, Fault::Pending));
            let owner = producer(&backend);
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 0;
            batch
                .push(&cx, "a".to_owned(), "one", &mut bytes)
                .await
                .unwrap();
            let mut future = Box::pin(batch.finish(&cx, &mut bytes));
            {
                let mut task = std::task::Context::from_waker(std::task::Waker::noop());
                assert!(std::future::Future::poll(future.as_mut(), &mut task).is_pending());
            }
            drop(future);
            assert_eq!(backend.dropped.load(Ordering::SeqCst), 1);
            assert_eq!(bytes, 0);
            assert_eq!(backend.widths(), [1]);
        }
    });
}

#[test]
fn empty_inference_groups_never_invoke_a_model() {
    run_test_with_cx(|cx| async move {
        for native in [false, true] {
            let backend = Arc::new(Backend::new(native, Fault::None));
            let owner = producer(&backend);
            let mut bytes = 0;
            assert!(
                InferenceBatch::new(&owner)
                    .finish(&cx, &mut bytes)
                    .await
                    .unwrap()
                    .is_empty()
            );
            assert!(backend.widths().is_empty());
            assert_eq!(bytes, 0);
        }
    });
}

#[derive(Default)]
struct InferenceWake(AtomicUsize);

impl std::task::Wake for InferenceWake {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cancellation_wakes_a_stalled_group_and_preserves_completed_group_accounting() {
    use std::task::{Context, Waker};

    for native in [false, true] {
        run_test_with_cx(|cx| async move {
            let backend = Arc::new(Backend::new(native, Fault::PendingAfterFirst));
            let owner = producer(&backend);
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 0;
            batch
                .push(&cx, "prior".to_owned(), "previous group", &mut bytes)
                .await
                .unwrap();
            batch.flush(&cx, &mut bytes).await.unwrap();
            let prior = batch.output.clone();
            assert_eq!(bytes, 12);
            for id in ["a", "b"] {
                batch
                    .push(&cx, id.to_owned(), id, &mut bytes)
                    .await
                    .unwrap();
            }
            let first = Arc::new(InferenceWake::default());
            let latest = Arc::new(InferenceWake::default());
            let first_waker = Waker::from(Arc::clone(&first));
            let latest_waker = Waker::from(Arc::clone(&latest));
            let mut operation = Box::pin(batch.flush(&cx, &mut bytes));
            assert!(
                operation
                    .as_mut()
                    .poll(&mut Context::from_waker(&first_waker))
                    .is_pending()
            );
            assert!(
                operation
                    .as_mut()
                    .poll(&mut Context::from_waker(&latest_waker))
                    .is_pending()
            );
            assert_eq!(backend.dropped.load(Ordering::SeqCst), 0);
            let before_first = first.0.load(Ordering::SeqCst);
            let before_latest = latest.0.load(Ordering::SeqCst);
            cx.cancel_with(
                asupersync::CancelKind::User,
                Some("cancel stalled retained update"),
            );
            // The provider is std::future::pending(): no provider completion,
            // runtime timer, or forced poll can account for this notification.
            assert!(latest.0.load(Ordering::SeqCst) > before_latest);
            assert_eq!(first.0.load(Ordering::SeqCst), before_first);
            assert!(matches!(
                operation
                    .as_mut()
                    .poll(&mut Context::from_waker(&latest_waker)),
                Poll::Ready(Err(SearchError::Cancelled { .. }))
            ));
            // Cleanup cannot depend on dropping an already-completed future.
            assert_eq!(backend.dropped.load(Ordering::SeqCst), 1);
            drop(operation);
            assert_eq!(batch.output, prior);
            assert_eq!(batch.ids, ["a", "b"]);
            assert_eq!(batch.texts, ["a", "b"]);
            assert_eq!(bytes, 12);
            assert_eq!(backend.widths(), [1, if native { 2 } else { 1 }]);
            assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
        });
    }
}

#[test]
fn cancelling_a_later_singleton_discards_the_groups_successful_prefix() {
    use std::task::{Context, Waker};

    run_test_with_cx(|cx| async move {
        let backend = Arc::new(Backend::new(false, Fault::PendingAfterFirst));
        let owner = producer(&backend);
        let mut batch = InferenceBatch::new(&owner);
        let mut bytes = 0;
        for id in ["first", "second", "never-started"] {
            batch
                .push(&cx, id.to_owned(), id, &mut bytes)
                .await
                .unwrap();
        }
        let wake = Arc::new(InferenceWake::default());
        let waker = Waker::from(Arc::clone(&wake));
        let mut operation = Box::pin(batch.flush(&cx, &mut bytes));
        assert!(
            operation
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        assert_eq!(backend.widths(), [1, 1]);
        let before = wake.0.load(Ordering::SeqCst);
        cx.cancel_with(
            asupersync::CancelKind::User,
            Some("cancel second singleton"),
        );
        assert!(wake.0.load(Ordering::SeqCst) > before);
        assert!(matches!(
            operation.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Ready(Err(SearchError::Cancelled { .. }))
        ));
        assert_eq!(backend.dropped.load(Ordering::SeqCst), 1);
        drop(operation);
        assert!(batch.output.is_empty());
        assert_eq!(batch.ids, ["first", "second", "never-started"]);
        assert_eq!(bytes, 0);
        assert_eq!(backend.widths(), [1, 1]);
        assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn abandoned_inference_registration_does_not_remove_a_siblings_cancellation_wake() {
    use std::task::{Context, Waker};

    run_test_with_cx(|cx| async move {
        let abandoned = Arc::new(InferenceWake::default());
        let sibling = Arc::new(InferenceWake::default());
        let abandoned_waker = Waker::from(Arc::clone(&abandoned));
        let sibling_waker = Waker::from(Arc::clone(&sibling));
        let mut first = Box::pin(with_cancellation(
            &cx,
            "fsfs.retained_batch.inference",
            std::future::pending::<SearchResult<()>>(),
        ));
        let mut second = Box::pin(with_cancellation(
            &cx,
            "fsfs.retained_batch.inference",
            std::future::pending::<SearchResult<()>>(),
        ));
        assert!(
            first
                .as_mut()
                .poll(&mut Context::from_waker(&abandoned_waker))
                .is_pending()
        );
        assert!(
            second
                .as_mut()
                .poll(&mut Context::from_waker(&sibling_waker))
                .is_pending()
        );
        drop(first);
        assert!(!cx.is_cancel_requested());
        let abandoned_before = abandoned.0.load(Ordering::SeqCst);
        let sibling_before = sibling.0.load(Ordering::SeqCst);
        cx.cancel_with(
            asupersync::CancelKind::User,
            Some("only the live group should wake"),
        );
        assert_eq!(abandoned.0.load(Ordering::SeqCst), abandoned_before);
        assert!(sibling.0.load(Ordering::SeqCst) > sibling_before);
        assert!(matches!(
            second
                .as_mut()
                .poll(&mut Context::from_waker(&sibling_waker)),
            Poll::Ready(Err(SearchError::Cancelled { .. }))
        ));
    });
}

#[test]
fn completed_inference_releases_registration_and_preserves_provider_errors() {
    use std::task::{Context, Waker};

    for fails in [false, true] {
        run_test_with_cx(|cx| async move {
            let result = if fails {
                Err(SearchError::Cancelled {
                    phase: "provider.own-scope".to_owned(),
                    reason: "keep the original reason".to_owned(),
                })
            } else {
                Ok(42)
            };
            let wake = Arc::new(InferenceWake::default());
            let waker = Waker::from(Arc::clone(&wake));
            let mut operation = Box::pin(with_cancellation(
                &cx,
                "fsfs.retained_batch.inference",
                std::future::ready(result),
            ));
            match operation.as_mut().poll(&mut Context::from_waker(&waker)) {
                Poll::Ready(Ok(value)) => {
                    assert!(!fails);
                    assert_eq!(value, 42);
                }
                Poll::Ready(Err(SearchError::Cancelled { phase, reason })) => {
                    assert!(fails);
                    assert_eq!(phase, "provider.own-scope");
                    assert_eq!(reason, "keep the original reason");
                }
                other => panic!("unexpected inference outcome: {other:?}"),
            }
            assert!(!cx.is_cancel_requested());
            let before = wake.0.load(Ordering::SeqCst);
            cx.cancel_with(asupersync::CancelKind::User, Some("after group completion"));
            assert_eq!(wake.0.load(Ordering::SeqCst), before);
            // Keep `operation` alive through cancellation to detect registrations
            // retained in a completed future rather than released on completion.
            drop(operation);
        });
    }
}

#[test]
fn cancelling_provider_factory_drops_the_unpolled_future_and_stages_nothing() {
    run_test_with_cx(|cx| async move {
        let backend = Arc::new(Backend::new(true, Fault::CancelFactory));
        let owner = producer(&backend);
        let mut batch = InferenceBatch::new(&owner);
        let mut bytes = 0;
        batch
            .push(&cx, "a".to_owned(), "one", &mut bytes)
            .await
            .unwrap();
        assert!(matches!(
            batch.flush(&cx, &mut bytes).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(backend.dropped.load(Ordering::SeqCst), 1);
        assert!(backend.widths().is_empty());
        assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
        assert!(batch.output.is_empty());
        assert_eq!(batch.ids, ["a"]);
        assert_eq!(bytes, 0);
    });
}

#[test]
fn cancellation_from_inference_destruction_wins_before_any_result_is_returned() {
    use std::task::Context;

    struct CancelOnDrop(Cx);
    impl Future for CancelOnDrop {
        type Output = SearchResult<()>;

        fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(Ok(()))
        }
    }
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0
                .cancel_with(asupersync::CancelKind::User, Some("provider destructor"));
        }
    }

    run_test_with_cx(|cx| async move {
        assert!(matches!(
            with_cancellation(
                &cx,
                "fsfs.retained_batch.inference",
                CancelOnDrop(cx.clone())
            )
            .await,
            Err(SearchError::Cancelled { .. })
        ));
    });
}
