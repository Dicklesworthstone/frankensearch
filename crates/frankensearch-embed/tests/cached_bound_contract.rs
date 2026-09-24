//! Bound inference must never be satisfied by `CachedEmbedder`'s raw-vector cache.
//! Synthetic native overrides are deliberately different from the raw primitive.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use asupersync::test_utils::run_test_with_cx;
use asupersync::{CancelKind, Cx};
use frankensearch_core::traits::{Embedder, IdentityBoundEmbedding, ModelCategory, SearchFuture};
use frankensearch_core::{EmbeddingIdentityBundleV1, SearchError, SearchResult};
use frankensearch_embed::CachedEmbedder;

const NORMAL: u8 = 0;
const FAIL: u8 = 1;
const CANCEL_SUCCESS: u8 = 2;
const CANCEL_FAILURE: u8 = 3;
const YIELD: u8 = 4;

struct Provider {
    advertised: EmbeddingIdentityBundleV1,
    responses: Mutex<Vec<IdentityBoundEmbedding>>,
    inputs: Mutex<Vec<Vec<String>>>,
    mode: AtomicU8,
    raw_calls: AtomicUsize,
    single_calls: AtomicUsize,
    batch_calls: AtomicUsize,
}

impl Provider {
    fn new() -> Self {
        let actual = EmbeddingIdentityBundleV1::explicit_test_model("actual-response", 3);
        Self {
            advertised: EmbeddingIdentityBundleV1::explicit_test_model("advertised", 3),
            responses: Mutex::new(vec![
                IdentityBoundEmbedding {
                    values: vec![0.0, -0.0, f32::from_bits(1)],
                    identity: actual.clone(),
                },
                IdentityBoundEmbedding {
                    values: vec![1.0, 2.0, 3.0],
                    identity: actual.clone(),
                },
                IdentityBoundEmbedding {
                    values: vec![3.0, 2.0, 1.0],
                    identity: actual,
                },
            ]),
            inputs: Mutex::new(Vec::new()),
            mode: AtomicU8::new(NORMAL),
            raw_calls: AtomicUsize::new(0),
            single_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
        }
    }

    async fn response_boundary(&self, cx: &Cx) -> SearchResult<()> {
        let mode = self.mode.load(Ordering::Relaxed);
        if mode == YIELD {
            let mut yielded = false;
            std::future::poll_fn(|task| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    task.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            })
            .await;
        }
        if matches!(mode, CANCEL_SUCCESS | CANCEL_FAILURE) {
            cx.cancel_fast(CancelKind::User);
        }
        if matches!(mode, FAIL | CANCEL_FAILURE) {
            return Err(SearchError::EmbeddingFailed {
                model: "cache-bound-fixture".to_owned(),
                source: "provider operation refused".into(),
            });
        }
        Ok(())
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, cx: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            self.response_boundary(cx).await?;
            Ok(vec![99.0, 0.0, 0.0])
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            self.response_boundary(cx).await?;
            Ok(vec![vec![99.0, 0.0, 0.0]; texts.len()])
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        _: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            let response = self.responses.lock().unwrap()[0].clone();
            self.response_boundary(cx).await?;
            Ok(response)
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
            let responses = self.responses.lock().unwrap().clone();
            self.response_boundary(cx).await?;
            Ok(responses)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.advertised)
    }

    fn dimension(&self) -> usize {
        3
    }

    fn id(&self) -> &'static str {
        "cache-bound-fixture"
    }

    fn model_name(&self) -> &'static str {
        "Cache bound fixture"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

fn assert_responses(actual: &[IdentityBoundEmbedding], expected: &[IdentityBoundEmbedding]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.identity, expected.identity);
        assert_eq!(
            actual
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn warm_raw_cache_cannot_supply_or_relabel_a_bound_batch() {
    run_test_with_cx(|cx| async move {
        let inner = Arc::new(Provider::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        cached.embed_batch(&cx, &["a", "b"]).await.unwrap();
        let before = cached.cache_stats();
        let expected = inner.responses.lock().unwrap().clone();
        assert_ne!(&expected[0].identity, cached.identity().unwrap());
        for _ in 0..2 {
            let responses = cached
                .embed_batch_bound(&cx, &["a", "a", "b"])
                .await
                .unwrap();
            assert_responses(&responses, &expected);
            assert_eq!(cached.cache_stats(), before);
        }
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.single_calls.load(Ordering::Relaxed), 0);
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 2);
        assert_eq!(
            *inner.inputs.lock().unwrap(),
            vec![vec!["a", "a", "b"], vec!["a", "a", "b"]]
        );
        assert_eq!(cached.embed(&cx, "a").await.unwrap(), vec![99.0, 0.0, 0.0]);
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn empty_bound_batches_still_dispatch_and_preserve_native_errors_with_caching_disabled() {
    run_test_with_cx(|cx| async move {
        let inner = Arc::new(Provider::new());
        inner.responses.lock().unwrap().clear();
        let cached = CachedEmbedder::new(inner.clone(), 0);
        let before = cached.cache_stats();
        assert!(cached.embed_batch_bound(&cx, &[]).await.unwrap().is_empty());
        inner.mode.store(FAIL, Ordering::Relaxed);
        let error = cached.embed_batch_bound(&cx, &[]).await.unwrap_err();
        assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
        assert!(error.to_string().contains("provider operation refused"));
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 2);
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 0);
        assert_eq!(cached.cache_stats(), before);
    });
}

#[test]
fn wrong_bound_batch_cardinality_never_changes_the_cache() {
    run_test_with_cx(|cx| async move {
        for output_count in 0..=4 {
            let inner = Arc::new(Provider::new());
            let response = inner.responses.lock().unwrap()[0].clone();
            *inner.responses.lock().unwrap() = vec![response; output_count];
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed(&cx, "a").await.unwrap();
            let before = cached.cache_stats();
            let result = cached.embed_batch_bound(&cx, &["a", "b"]).await;
            if output_count == 2 {
                assert_eq!(result.unwrap().len(), 2);
            } else {
                assert!(matches!(
                    result,
                    Err(SearchError::InvalidConfig { field, .. }) if field == "embedder.batch_length"
                ));
            }
            assert_eq!(cached.cache_stats(), before);
            assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
            assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
        }
    });
}

#[test]
fn malformed_single_or_late_batch_response_is_rejected_without_disclosure() {
    run_test_with_cx(|cx| async move {
        for mutation in 0..5 {
            let inner = Arc::new(Provider::new());
            let mut invalid = inner.responses.lock().unwrap()[0].clone();
            match mutation {
                0 => invalid.values.clear(),
                1 => invalid.values[1] = f32::NAN,
                2 => invalid.values[1] = f32::INFINITY,
                3 => "private-canary".clone_into(&mut invalid.identity.storage.format),
                _ => invalid.identity.space.dimension = 0,
            }
            let cached = CachedEmbedder::new(inner.clone(), 4);
            cached.embed(&cx, "warm").await.unwrap();
            let before = cached.cache_stats();
            inner.responses.lock().unwrap()[2] = invalid.clone();
            let error = cached
                .embed_batch_bound(&cx, &["warm", "new", "new"])
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
            assert!(!error.to_string().contains("private-canary"));
            inner.responses.lock().unwrap()[0] = invalid;
            let error = cached.embed_bound(&cx, "warm").await.unwrap_err();
            // Single-query admission rejects the foreign identity before
            // inspecting its malformed, potentially sensitive response fields.
            assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            assert!(!error.to_string().contains("private-canary"));
            assert_eq!(cached.cache_stats().entries, before.entries);
            assert_eq!(cached.cache_stats().hits, before.hits);
            assert_eq!(cached.cache_stats().misses, before.misses + 1);
        }
    });
}

#[test]
fn mixed_valid_identities_in_one_native_batch_are_not_silently_relabelled() {
    run_test_with_cx(|cx| async move {
        let inner = Arc::new(Provider::new());
        inner.responses.lock().unwrap()[2]
            .identity
            .producer
            .implementation_revision
            .push_str("-private-canary");
        let cached = CachedEmbedder::new(inner.clone(), 4);
        let before = cached.cache_stats();
        let error = cached
            .embed_batch_bound(&cx, &["a", "b", "c"])
            .await
            .unwrap_err();
        assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
        assert!(!error.to_string().contains("private-canary"));
        assert_eq!(cached.cache_stats(), before);
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn bound_futures_are_lazy_and_cancel_before_dispatch_even_with_warm_hits() {
    run_test_with_cx(|cx| async move {
        let inner = Arc::new(Provider::new());
        let cached = CachedEmbedder::new(inner.clone(), 4);
        cached.embed(&cx, "warm").await.unwrap();
        let before = cached.cache_stats();
        drop(cached.embed_bound(&cx, "warm"));
        drop(cached.embed_batch_bound(&cx, &["warm", "warm", "warm"]));
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(inner.single_calls.load(Ordering::Relaxed), 0);
        cx.cancel_fast(CancelKind::User);
        for inputs in [vec![], vec!["warm", "warm", "warm"]] {
            assert!(matches!(
                cached.embed_batch_bound(&cx, &inputs).await,
                Err(SearchError::Cancelled { .. })
            ));
        }
        assert!(matches!(
            cached.embed_bound(&cx, "warm").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(cached.cache_stats(), before);
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 0);
        assert_eq!(inner.single_calls.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn cancellation_beats_success_and_backend_error_on_all_four_cache_operations() {
    for mode in [CANCEL_SUCCESS, CANCEL_FAILURE] {
        for operation in 0..4 {
            run_test_with_cx(|cx| async move {
                let inner = Arc::new(Provider::new());
                let cached = CachedEmbedder::new(inner.clone(), 4);
                cached.embed(&cx, "warm").await.unwrap();
                let before = cached.cache_stats();
                inner.mode.store(mode, Ordering::Relaxed);
                let result = match operation {
                    0 => cached.embed(&cx, "new").await.map(|_| ()),
                    1 => cached.embed_batch(&cx, &["new"]).await.map(|_| ()),
                    2 => cached.embed_bound(&cx, "warm").await.map(|_| ()),
                    _ => cached
                        .embed_batch_bound(&cx, &["warm", "warm", "warm"])
                        .await
                        .map(|_| ()),
                };
                assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                assert_eq!(cached.cache_stats().entries, before.entries);
                if operation == 3 {
                    assert_eq!(cached.cache_stats(), before);
                } else if operation == 2 {
                    assert_eq!(cached.cache_stats().hits, before.hits);
                    assert_eq!(cached.cache_stats().misses, before.misses + 1);
                }
            });
        }
    }
}

#[test]
fn clearing_during_a_bound_batch_cannot_refill_or_evict_new_raw_entries() {
    run_test_with_cx(|cx| async move {
        let inner = Arc::new(Provider::new());
        let cached = CachedEmbedder::new(inner.clone(), 1);
        inner.mode.store(YIELD, Ordering::Relaxed);
        let mut flight = cached.embed_batch_bound(&cx, &["a", "a", "b"]);
        {
            let mut task = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(std::future::Future::poll(flight.as_mut(), &mut task).is_pending());
        }
        assert_eq!(inner.batch_calls.load(Ordering::Relaxed), 1);
        cached.clear_cache();
        inner.mode.store(NORMAL, Ordering::Relaxed);
        cached.embed(&cx, "new").await.unwrap();
        let after_clear = cached.cache_stats();
        let expected = inner.responses.lock().unwrap().clone();
        assert_responses(&flight.await.unwrap(), &expected);
        assert_eq!(cached.cache_stats(), after_clear);
        cached.embed(&cx, "new").await.unwrap();
        assert_eq!(inner.raw_calls.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn nested_cache_wrappers_preserve_native_batch_identity_without_raw_work() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new());
        let inner_cache = Arc::new(CachedEmbedder::new(provider.clone(), 4));
        let outer_cache = CachedEmbedder::new(inner_cache.clone(), 4);
        let inner_before = inner_cache.cache_stats();
        let outer_before = outer_cache.cache_stats();
        let result = outer_cache
            .embed_batch_bound(&cx, &["a", "a", "b"])
            .await
            .unwrap();
        assert_responses(&result, &provider.responses.lock().unwrap());
        assert_eq!(provider.batch_calls.load(Ordering::Relaxed), 1);
        assert_eq!(provider.raw_calls.load(Ordering::Relaxed), 0);
        assert_eq!(inner_cache.cache_stats(), inner_before);
        assert_eq!(outer_cache.cache_stats(), outer_before);
    });
}
