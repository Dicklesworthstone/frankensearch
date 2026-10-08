use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
        foreign.producer.implementation_revision.push_str("-private-canary");
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
        self.calls.lock().unwrap().push(texts.iter().map(|text| (*text).to_owned()).collect());
        match self.fault {
            Fault::Dispatch => { self.native.fetch_xor(true, Ordering::SeqCst); }
            Fault::Drift | Fault::DriftFailure => self.drift.store(true, Ordering::SeqCst),
            Fault::Cancel => cx.set_cancel_requested(true),
            Fault::TypedCancel => return Err(SearchError::Cancelled {
                phase: "provider.terminal".to_owned(), reason: "original cancellation".to_owned(),
            }),
            Fault::Pending => {
                struct Guard<'a>(&'a AtomicUsize);
                impl Drop for Guard<'_> {
                    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
                }
                let _guard = Guard(&self.dropped);
                return std::future::pending().await;
            }
            _ => {}
        }
        if matches!(self.fault, Fault::Failure | Fault::DriftFailure) {
            return Err(SearchError::EmbeddingFailed {
                model: "original-provider".to_owned(), source: "original failure".into(),
            });
        }
        let mut rows = texts.iter().map(|text| IdentityBoundEmbedding {
            values: Self::values(text), identity: self.identity.clone(),
        }).collect::<Vec<_>>();
        match self.fault {
            Fault::Count => { let _ = rows.pop(); }
            Fault::Width => { let _ = rows.last_mut().unwrap().values.pop(); }
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
    fn embed_bound<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move { Ok(self.respond(cx, &[text]).await?.pop().unwrap()) })
    }
    fn embed_batch_bound<'a>(
        &'a self, cx: &'a Cx, texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            assert!(self.native.load(Ordering::SeqCst), "unadvertised batch dispatch");
            self.respond(cx, texts).await
        })
    }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.drift.load(Ordering::SeqCst) { &self.foreign } else { &self.identity })
    }
    fn bound_batch_is_native(&self) -> bool { self.native.load(Ordering::SeqCst) }
    fn dimension(&self) -> usize { 3 }
    fn id(&self) -> &str { "group-fixture" }
    fn model_name(&self) -> &str { "Bound group fixture" }
    fn is_semantic(&self) -> bool { true }
    fn category(&self) -> ModelCategory { ModelCategory::TransformerEmbedder }
}

fn producer(backend: &Arc<Backend>) -> Producer {
    Producer { embedder: backend.clone(), identity: backend.identity.clone(), role: "test.group" }
}

#[test]
fn native_groups_cross_document_boundaries_and_preserve_slot_bits() {
    run_test_with_cx(|cx| async move {
        let backend = Arc::new(Backend::new(true, Fault::None));
        let owner = producer(&backend);
        let texts = (0..130).map(|n| format!("{} duplicate body", n % 3)).collect::<Vec<_>>();
        let mut batch = InferenceBatch::new(&owner);
        let mut bytes = 0;
        for (n, text) in texts.iter().enumerate() {
            batch.push(&cx, format!("doc-{n}"), text, &mut bytes).await.unwrap();
        }
        let output = batch.finish(&cx, &mut bytes).await.unwrap();
        assert_eq!(backend.widths(), [64, 64, 2]);
        assert_eq!(bytes, 130 * 3 * 4);
        assert_eq!(output.len(), texts.len());
        for (n, ((id, vector), text)) in output.iter().zip(&texts).enumerate() {
            assert_eq!(id, &format!("doc-{n}"));
            assert_eq!(vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                Backend::values(text).iter().map(|v| v.to_bits()).collect::<Vec<_>>());
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
            batch.push(&cx, format!("doc-{n}"), "same", &mut bytes).await.unwrap();
        }
        let output = batch.finish(&cx, &mut bytes).await.unwrap();
        assert_eq!(backend.widths(), vec![1; 65]);
        assert_eq!(output.len(), 65);
        assert!(output.iter().all(|(_, vector)| vector == &Backend::values("same")));
        assert_eq!(backend.raw_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn bad_native_response_stages_no_prefix_and_is_never_retried() {
    run_test_with_cx(|cx| async move {
        for fault in [Fault::Count, Fault::Width, Fault::NonFinite, Fault::Foreign, Fault::Failure] {
            let backend = Arc::new(Backend::new(true, fault));
            let owner = producer(&backend);
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 12; // a completed group belonging to the other tier
            batch.push(&cx, "first".to_owned(), "one", &mut bytes).await.unwrap();
            batch.push(&cx, "last".to_owned(), "two", &mut bytes).await.unwrap();
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
                batch.push(&cx, "first".to_owned(), "one", &mut bytes).await.unwrap();
                batch.push(&cx, "last".to_owned(), "two", &mut bytes).await.unwrap();
                assert!(matches!(batch.flush(&cx, &mut bytes).await,
                    Err(SearchError::UnverifiableRemoteSpace { .. })));
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
            assert!(matches!(batch.push(&cx, "a".to_owned(), "one", &mut bytes).await,
                Err(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
            assert!(backend.widths().is_empty());
            for fault in [Fault::Cancel, Fault::TypedCancel] {
                let backend = Arc::new(Backend::new(native, fault));
                let owner = producer(&backend);
                let mut batch = InferenceBatch::new(&owner);
                batch.push(&cx, "a".to_owned(), "one", &mut bytes).await.unwrap();
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
        fast_batch.push(&cx, "a".to_owned(), "fast", &mut bytes).await.unwrap();
        quality_batch.push(&cx, "a".to_owned(), "quality", &mut bytes).await.unwrap();
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
            let text = "x".repeat(if large { MAX_TEXT_BYTES + 1 } else { MAX_TEXT_BYTES / 2 });
            let texts = if large { [text.as_str(), "small", "small"] }
                else { [text.as_str(), text.as_str(), text.as_str()] };
            let mut batch = InferenceBatch::new(&owner);
            let mut bytes = 0;
            for (n, text) in texts.iter().enumerate() {
                batch.push(&cx, n.to_string(), text, &mut bytes).await.unwrap();
            }
            assert_eq!(batch.finish(&cx, &mut bytes).await.unwrap().len(), 3);
            assert_eq!(backend.widths(), if large { vec![1, 2] } else { vec![2, 1] });
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
            batch.push(&cx, "a".to_owned(), "one", &mut bytes).await.unwrap();
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
            assert!(InferenceBatch::new(&owner).finish(&cx, &mut bytes).await.unwrap().is_empty());
            assert!(backend.widths().is_empty());
            assert_eq!(bytes, 0);
        }
    });
}
