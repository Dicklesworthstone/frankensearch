//! Synthetic producer-contract regressions, not model-quality measurements.

use super::*;
use std::future::{Future, pending};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Valid,
    Foreign,
    ForeignLast,
    MalformedForeign,
    Short,
    Long,
    NanLast,
    InfiniteLast,
    WrongWidthLast,
    Drift,
    DimensionDrift,
    Failure,
    ProviderCancelled,
    CancelSuccess,
    CancelFailure,
    CancelDrift,
    UnavailableAfter,
    CancelledUnavailableAfter,
    Pending,
}

struct Producer {
    admitted: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    mode: Mutex<Mode>,
    drifted: AtomicBool,
    unavailable: AtomicBool,
    dimension: AtomicUsize,
    raw_calls: AtomicUsize,
    single_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    identity_calls: AtomicUsize,
    fail_on_identity: AtomicUsize,
    cancel_on_identity: Mutex<Option<(usize, Cx)>>,
    inputs: Mutex<Vec<Vec<String>>>,
}

impl Producer {
    fn new() -> Self {
        Self {
            admitted: EmbeddingIdentityBundleV1::explicit_test_model("cached-batch", 4),
            foreign: EmbeddingIdentityBundleV1::explicit_test_model("CANARY_PRIVATE_FOREIGN", 4),
            mode: Mutex::new(Mode::Valid),
            drifted: AtomicBool::new(false),
            unavailable: AtomicBool::new(false),
            dimension: AtomicUsize::new(4),
            raw_calls: AtomicUsize::new(0),
            single_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
            identity_calls: AtomicUsize::new(0),
            fail_on_identity: AtomicUsize::new(0),
            cancel_on_identity: Mutex::new(None),
            inputs: Mutex::new(Vec::new()),
        }
    }

    fn set_mode(&self, mode: Mode) {
        *self.mode.lock().unwrap() = mode;
    }

    fn failure() -> SearchError {
        SearchError::InvalidConfig {
            field: "CANARY_PRIVATE_FIELD".to_owned(),
            value: "CANARY_PRIVATE_VALUE".to_owned(),
            reason: "CANARY_PRIVATE_PAYLOAD".to_owned(),
        }
    }

    fn response(&self, text: &str) -> IdentityBoundEmbedding {
        IdentityBoundEmbedding {
            values: if text == "alpha" {
                vec![3.0, 4.0, 1.0, 2.0]
            } else {
                vec![4.0, 3.0, 2.0, 1.0]
            },
            identity: self.admitted.clone(),
        }
    }

    async fn invoke(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        let mode = *self.mode.lock().unwrap();
        if mode == Mode::Pending {
            pending::<()>().await;
        }
        if matches!(mode, Mode::Drift | Mode::CancelDrift) {
            self.drifted.store(true, Ordering::Relaxed);
        }
        if mode == Mode::DimensionDrift {
            self.dimension.store(5, Ordering::Relaxed);
        }
        if matches!(mode, Mode::UnavailableAfter | Mode::CancelledUnavailableAfter) {
            self.unavailable.store(true, Ordering::Relaxed);
        }
        if matches!(mode, Mode::CancelSuccess | Mode::CancelFailure | Mode::CancelDrift) {
            cx.set_cancel_requested(true);
        }
        if matches!(mode, Mode::Failure | Mode::CancelFailure) {
            return Err(Self::failure());
        }
        if matches!(mode, Mode::ProviderCancelled | Mode::CancelledUnavailableAfter) {
            return Err(SearchError::Cancelled {
                phase: "synthetic.provider.cancelled".to_owned(),
                reason: "provider cancellation without cancelling caller context".to_owned(),
            });
        }
        let mut out: Vec<_> = texts.iter().map(|text| self.response(text)).collect();
        match mode {
            Mode::Foreign | Mode::MalformedForeign => {
                for response in &mut out {
                    response.identity = self.foreign.clone();
                    if mode == Mode::MalformedForeign {
                        response.identity.space.dimension = 0;
                    }
                }
            }
            Mode::ForeignLast => {
                if let Some(last) = out.last_mut() {
                    last.identity = self.foreign.clone();
                }
            }
            Mode::Short => {
                out.pop();
            }
            Mode::Long => out.push(self.response("extra")),
            Mode::NanLast | Mode::InfiniteLast | Mode::WrongWidthLast => {
                if let Some(last) = out.last_mut() {
                    match mode {
                        Mode::NanLast => last.values[3] = f32::NAN,
                        Mode::InfiniteLast => last.values[3] = f32::INFINITY,
                        Mode::WrongWidthLast => {
                            last.values.pop();
                        }
                        _ => unreachable!(),
                    }
                }
            }
            _ => {}
        }
        Ok(out)
    }
}

impl Embedder for Producer {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.response(text).values)
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            self.invoke(cx, &[text]).await?.pop().ok_or_else(Self::failure)
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            self.inputs
                .lock()
                .unwrap()
                .push(texts.iter().map(|text| (*text).to_owned()).collect());
            self.invoke(cx, texts).await
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        let call = self.identity_calls.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some((at, cx)) = self.cancel_on_identity.lock().unwrap().as_ref() {
            if call == *at {
                cx.set_cancel_requested(true);
            }
        }
        if self.unavailable.load(Ordering::Relaxed)
            || call == self.fail_on_identity.load(Ordering::Relaxed)
        {
            return Err(Self::failure());
        }
        Ok(if self.drifted.load(Ordering::Relaxed) {
            &self.foreign
        } else {
            &self.admitted
        })
    }

    fn dimension(&self) -> usize {
        self.dimension.load(Ordering::Relaxed)
    }
    fn id(&self) -> &str {
        "cached-batch-fixture"
    }
    fn model_name(&self) -> &str {
        "Synthetic cache batch producer"
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
    fn bound_batch_is_native(&self) -> bool {
        true
    }
}

#[test]
fn valid_batches_preserve_identity_order_duplicates_and_empty_dispatch() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(Producer::new());
        let cached = CachedEmbedder::new(inner.clone(), 2);
        let inputs = ["alpha", "beta", "alpha"];
        let before = cached.cache_stats();
        let out = cached.embed_batch_bound(&cx, &inputs).await.unwrap();
        for (text, response) in inputs.iter().zip(&out) {
            response.validate().unwrap();
            assert_eq!(response.identity, inner.admitted);
            assert_eq!(response.values, inner.response(text).values);
        }
        assert!(cached.embed_batch_bound(&cx, &[]).await.unwrap().is_empty());
        assert_eq!(
            *inner.inputs.lock().unwrap(),
            vec![
                vec!["alpha".to_owned(), "beta".to_owned(), "alpha".to_owned()],
                vec![],
            ]
        );
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 2);
        assert_eq!(inner.single_calls.load(Ordering::Relaxed), 0);
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 0);
        assert_eq!(cached.cache_stats(), before);
    });
}

#[test]
fn homogeneous_foreign_batch_is_rejected_even_when_every_row_validates() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for capacity in [0, 2] {
            let inner = Arc::new(Producer::new());
            inner.set_mode(Mode::Foreign);
            let foreign = IdentityBoundEmbedding {
                values: vec![1.0; 4],
                identity: inner.foreign.clone(),
            };
            foreign.validate().unwrap();
            let cached = CachedEmbedder::new(inner.clone(), capacity);
            let before = cached.cache_stats();
            let error = cached.embed_batch_bound(&cx, &["alpha", "beta"]).await.unwrap_err();
            assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            assert!(!error.to_string().contains("CANARY_PRIVATE"));
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
        }
    });
}

#[test]
fn malformed_batches_never_mutate_or_evict_existing_raw_and_bound_entries() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mode in [Mode::ForeignLast, Mode::MalformedForeign, Mode::Short, Mode::Long,
                     Mode::NanLast, Mode::InfiniteLast, Mode::WrongWidthLast] {
            let inner = Arc::new(Producer::new());
            let cached = CachedEmbedder::new(inner.clone(), 2);
            let raw = cached.embed(&cx, "resident-raw").await.unwrap();
            let bound = cached.embed_bound(&cx, "resident-bound").await.unwrap();
            let before = cached.cache_stats();
            inner.set_mode(mode);
            let error = cached.embed_batch_bound(&cx, &["alpha", "beta"]).await.unwrap_err();
            assert!(!error.to_string().contains("CANARY_PRIVATE"));
            assert_eq!(cached.cache_stats(), before);
            inner.set_mode(Mode::Valid);
            assert_eq!(cached.embed(&cx, "resident-raw").await.unwrap(), raw);
            assert_eq!(cached.embed_bound(&cx, "resident-bound").await.unwrap().values, bound.values);
            assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
            assert_eq!(inner.single_calls.load(Ordering::Relaxed), 1);
        }
    });
}

#[test]
fn metadata_drift_and_loss_during_inference_are_rejected() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mode in [Mode::Drift, Mode::DimensionDrift, Mode::UnavailableAfter] {
            let inner = Arc::new(Producer::new());
            inner.set_mode(mode);
            let cached = CachedEmbedder::new(inner.clone(), 2);
            let before = cached.cache_stats();
            let error = cached.embed_batch_bound(&cx, &["alpha"]).await.unwrap_err();
            assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            assert!(!error.to_string().contains("CANARY_PRIVATE"));
            assert_eq!(cached.cache_stats(), before);
        }
    });
}

#[test]
fn invalid_or_missing_initial_metadata_stops_before_inference_even_when_empty() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for fault in 0..4 {
            for inputs in [&[][..], &["alpha"][..]] {
                let mut producer = Producer::new();
                match fault {
                    0 => producer.unavailable.store(true, Ordering::Relaxed),
                    1 => producer.admitted.space.dimension = 0,
                    2 => producer.dimension.store(5, Ordering::Relaxed),
                    _ => producer.admitted.storage.endianness = "not-native".to_owned(),
                }
                let inner = Arc::new(producer);
                let cached = CachedEmbedder::new(inner.clone(), 2);
                let error = cached.embed_batch_bound(&cx, inputs).await.unwrap_err();
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                assert!(!error.to_string().contains("CANARY_PRIVATE"));
                assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 0);
                assert_eq!(cached.cache_stats().entries, 0);
            }
        }
    });
}

#[test]
fn precancelled_batches_and_unpolled_futures_do_no_provider_or_cache_work() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(Producer::new());
        let cached = CachedEmbedder::new(inner.clone(), 2);
        let before = cached.cache_stats();
        let inputs = ["alpha"];
        drop(cached.embed_batch_bound(&cx, &inputs));
        assert_eq!(inner.identity_calls.load(Ordering::Relaxed), 0);
        cx.set_cancel_requested(true);
        assert!(matches!(
            cached.embed_batch_bound(&cx, &inputs).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            cached.embed_batch_bound(&cx, &[]).await,
            Err(SearchError::Cancelled { .. })
        ));
        cx.set_cancel_requested(false);
        assert_eq!(inner.identity_calls.load(Ordering::Relaxed), 0);
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(cached.cache_stats(), before);
    });
}

#[test]
fn cancellation_after_inference_outranks_success_provider_error_and_drift() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for mode in [Mode::CancelSuccess, Mode::CancelFailure, Mode::CancelDrift] {
            let inner = Arc::new(Producer::new());
            inner.set_mode(mode);
            let cached = CachedEmbedder::new(inner.clone(), 2);
            let before = cached.cache_stats();
            assert!(matches!(
                cached.embed_batch_bound(&cx, &["alpha"]).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(inner.identity_calls.load(Ordering::Relaxed), 1);
            assert_eq!(cached.cache_stats(), before);
        }
    });
}

#[test]
fn explicit_provider_cancellation_survives_lost_metadata_for_singles_and_batches() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for batch in [false, true] {
            for mode in [Mode::ProviderCancelled, Mode::CancelledUnavailableAfter] {
                let inner = Arc::new(Producer::new());
                inner.set_mode(mode);
                let cached = CachedEmbedder::new(inner.clone(), 2);
                let error = if batch {
                    cached.embed_batch_bound(&cx, &["alpha"]).await.unwrap_err()
                } else {
                    cached.embed_bound(&cx, "alpha").await.unwrap_err()
                };
                match error {
                    SearchError::Cancelled { phase, .. } => {
                        assert_eq!(phase, "synthetic.provider.cancelled");
                    }
                    other => panic!("expected original cancellation, got {other:?}"),
                }
                assert!(cx.checkpoint().is_ok());
                assert_eq!(inner.identity_calls.load(Ordering::Relaxed), 1);
                assert_eq!(cached.cache_stats().entries, 0);
            }
        }
    });
}

#[test]
fn metadata_callback_cancellation_outranks_its_success_or_error() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for at in [1, 2] {
            for fails in [false, true] {
                let inner = Arc::new(Producer::new());
                *inner.cancel_on_identity.lock().unwrap() = Some((at, cx.clone()));
                if fails {
                    inner.fail_on_identity.store(at, Ordering::Relaxed);
                }
                let cached = CachedEmbedder::new(inner.clone(), 2);
                let before = cached.cache_stats();
                assert!(matches!(
                    cached.embed_batch_bound(&cx, &["alpha"]).await,
                    Err(SearchError::Cancelled { .. })
                ));
                cx.set_cancel_requested(false);
                assert_eq!(inner.batch_calls.load(Ordering::Relaxed), usize::from(at == 2));
                assert_eq!(cached.cache_stats(), before);
            }
        }
    });
}

#[test]
fn dropped_pending_batch_leaves_no_partial_result_and_explicit_retry_succeeds() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let inner = Arc::new(Producer::new());
        inner.set_mode(Mode::Pending);
        let cached = CachedEmbedder::new(inner.clone(), 2);
        let before = cached.cache_stats();
        let inputs = ["alpha", "beta", "alpha"];
        {
            let mut future = cached.embed_batch_bound(&cx, &inputs);
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(Future::poll(future.as_mut(), &mut context), Poll::Pending));
        }
        assert_eq!(cached.cache_stats(), before);
        inner.set_mode(Mode::Valid);
        let out = cached.embed_batch_bound(&cx, &inputs).await.unwrap();
        assert_eq!(out.len(), inputs.len());
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 2);
        assert_eq!(cached.cache_stats(), before);
    });
}

#[test]
fn ordinary_provider_failure_is_preserved_only_while_identity_remains_admitted() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for loses_identity in [false, true] {
            let inner = Arc::new(Producer::new());
            inner.set_mode(Mode::Failure);
            if loses_identity {
                inner.fail_on_identity.store(2, Ordering::Relaxed);
            }
            let cached = CachedEmbedder::new(inner.clone(), 2);
            let before = cached.cache_stats();
            let error = cached.embed_batch_bound(&cx, &["alpha"]).await.unwrap_err();
            if loses_identity {
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                assert!(!error.to_string().contains("CANARY_PRIVATE"));
            } else {
                assert!(matches!(error, SearchError::InvalidConfig { .. }));
            }
            assert_eq!(inner.identity_calls.load(Ordering::Relaxed), 2);
            assert_eq!(cached.cache_stats(), before);
        }
    });
}
