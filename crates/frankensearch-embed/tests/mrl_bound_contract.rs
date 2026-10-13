//! MRL must project the provider's actual bound response, never relabel raw
//! inference. All providers here are synthetic; no model files are loaded.

use std::future::{Future, poll_fn};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use asupersync::test_utils::run_test_with_cx;
use asupersync::{CancelKind, Cx};
use frankensearch_core::traits::{Embedder, IdentityBoundEmbedding, ModelCategory, SearchFuture};
use frankensearch_core::{
    EmbeddingIdentityBundleV1, QuantizationFormat, SearchError, SearchResult,
};
use frankensearch_embed::{CachedEmbedder, DimReduceEmbedder, ModelArtifactManifestV1};

const PRIVATE: &str = "private-mrl-provider-canary";
const DIMENSION: usize = 768;

#[derive(Clone, Copy)]
enum Outcome {
    Success,
    Failure,
    DriftSuccess,
    DriftFailure,
    CancelSuccess,
    CancelFailure,
    ExplicitCancel,
    Yield,
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    Foreign,
    Storage,
    Quantization,
    Short,
    Long,
    NanPrefix,
    NanTail,
    InfinityTail,
    NegativeInfinityTail,
    ShortBatch,
    LongBatch,
}

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    native: AtomicBool,
    drift: AtomicU8,
    outcome: Mutex<Outcome>,
    fault: Mutex<Fault>,
    projection_fault: AtomicU8,
    cancel_metadata: Mutex<Option<Cx>>,
    metadata_reads: AtomicUsize,
    reads_at_explicit_cancel: AtomicUsize,
    raw_calls: AtomicUsize,
    single_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    projections: AtomicUsize,
    inputs: Mutex<Vec<Vec<String>>>,
}

impl Provider {
    fn new(native: bool) -> Self {
        // Use the existing registered MRL fixture contract, not real inference.
        let identity = ModelArtifactManifestV1::nomic_fastembed()
            .unwrap()
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        assert_eq!(
            usize::try_from(identity.space.dimension).unwrap(),
            DIMENSION
        );
        Self {
            identity,
            foreign: EmbeddingIdentityBundleV1::explicit_test_model(PRIVATE, 768),
            native: AtomicBool::new(native),
            drift: AtomicU8::new(0),
            outcome: Mutex::new(Outcome::Success),
            fault: Mutex::new(Fault::None),
            projection_fault: AtomicU8::new(0),
            cancel_metadata: Mutex::new(None),
            metadata_reads: AtomicUsize::new(0),
            reads_at_explicit_cancel: AtomicUsize::new(0),
            raw_calls: AtomicUsize::new(0),
            single_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
            projections: AtomicUsize::new(0),
            inputs: Mutex::new(Vec::new()),
        }
    }

    fn full(&self, text: &str) -> IdentityBoundEmbedding {
        let mut values = vec![0.0; DIMENSION];
        if text != "zero" {
            values[0] = if text == "b" { 4.0 } else { 3.0 };
            values[1] = if text == "b" { 3.0 } else { 4.0 };
            values[2] = 12.0;
        }
        IdentityBoundEmbedding {
            values,
            identity: self.identity.clone(),
        }
    }

    fn corrupt(&self, response: &mut IdentityBoundEmbedding) {
        let fault = *self.fault.lock().unwrap();
        match fault {
            Fault::None | Fault::ShortBatch | Fault::LongBatch => {}
            Fault::Foreign => response.identity.clone_from(&self.foreign),
            Fault::Storage => response.identity.storage.format = format!("in-memory-{PRIVATE}"),
            Fault::Quantization => {
                response.identity.storage.quantization = QuantizationFormat::F16;
            }
            Fault::Short => response.values.truncate(2),
            Fault::Long => response.values.push(0.0),
            Fault::NanPrefix => response.values[0] = f32::NAN,
            Fault::NanTail => response.values[DIMENSION - 1] = f32::NAN,
            Fault::InfinityTail => response.values[DIMENSION - 1] = f32::INFINITY,
            Fault::NegativeInfinityTail => response.values[DIMENSION - 1] = f32::NEG_INFINITY,
        }
    }

    async fn finish(&self, cx: &Cx) -> SearchResult<()> {
        let outcome = *self.outcome.lock().unwrap();
        if matches!(outcome, Outcome::Yield) {
            let mut yielded = false;
            poll_fn(|task| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    task.waker().wake_by_ref();
                    Poll::Pending
                }
            })
            .await;
        }
        if matches!(outcome, Outcome::DriftSuccess | Outcome::DriftFailure) {
            self.drift.store(1, Ordering::Relaxed);
        }
        if matches!(outcome, Outcome::CancelSuccess | Outcome::CancelFailure) {
            cx.cancel_fast(CancelKind::User);
        }
        if matches!(outcome, Outcome::ExplicitCancel) {
            // No Cx cancellation: the returned typed cancellation itself is
            // terminal and must not trigger another provider metadata read.
            self.drift.store(1, Ordering::Relaxed);
            self.reads_at_explicit_cancel.store(
                self.metadata_reads.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            return Err(SearchError::Cancelled {
                phase: "fixture-provider".to_owned(),
                reason: "explicit cancellation".to_owned(),
            });
        }
        if matches!(
            outcome,
            Outcome::Failure | Outcome::DriftFailure | Outcome::CancelFailure
        ) {
            return Err(SearchError::EmbeddingFailed {
                model: "mrl-fixture".to_owned(),
                source: PRIVATE.into(),
            });
        }
        Ok(())
    }

    fn assert_calls(&self, raw: usize, singles: usize, batches: usize, projections: usize) {
        assert_eq!(self.raw_calls.load(Ordering::Relaxed), raw);
        assert_eq!(self.single_calls.load(Ordering::Relaxed), singles);
        assert_eq!(self.batch_calls.load(Ordering::Relaxed), batches);
        assert_eq!(self.projections.load(Ordering::Relaxed), projections);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            let mut raw = vec![0.0; DIMENSION];
            raw[0] = 99.0;
            Ok(raw)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let mut values = Vec::new();
            for text in texts {
                values.push(self.embed(cx, text).await?);
            }
            Ok(values)
        })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            self.inputs.lock().unwrap().push(vec![text.to_owned()]);
            let mut response = self.full(text);
            self.corrupt(&mut response);
            self.finish(cx).await?;
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
            if !self.native.load(Ordering::Relaxed) {
                return Err(SearchError::EmbeddingFailed {
                    model: "mrl-fixture".to_owned(),
                    source: "non-native bound batch must not be dispatched".into(),
                });
            }
            let mut responses = texts.iter().map(|text| self.full(text)).collect::<Vec<_>>();
            if let Some(last) = responses.last_mut() {
                self.corrupt(last);
            }
            let fault = *self.fault.lock().unwrap();
            match fault {
                Fault::ShortBatch => {
                    let _ = responses.pop();
                }
                Fault::LongBatch => responses.push(self.full("extra")),
                _ => {}
            }
            self.finish(cx).await?;
            Ok(responses)
        })
    }

    fn bound_batch_is_native(&self) -> bool {
        self.native.load(Ordering::Relaxed)
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.metadata_reads.fetch_add(1, Ordering::Relaxed);
        if let Some(cx) = self.cancel_metadata.lock().unwrap().as_ref() {
            cx.cancel_fast(CancelKind::User);
        }
        match self.drift.load(Ordering::Relaxed) {
            1 => Ok(&self.foreign),
            2 => Err(SearchError::EmbeddingFailed {
                model: "mrl-fixture".to_owned(),
                source: PRIVATE.into(),
            }),
            _ => Ok(&self.identity),
        }
    }

    fn dimension(&self) -> usize {
        if self.drift.load(Ordering::Relaxed) == 3 {
            DIMENSION + 1
        } else {
            DIMENSION
        }
    }

    fn supports_mrl(&self) -> bool {
        self.drift.load(Ordering::Relaxed) != 4
    }

    fn truncate_embedding(&self, values: &[f32], target_dim: usize) -> SearchResult<Vec<f32>> {
        self.projections.fetch_add(1, Ordering::Relaxed);
        match self.projection_fault.load(Ordering::Relaxed) {
            1 => return Ok(vec![0.0; target_dim + 1]),
            2 => return Ok(vec![f32::NAN; target_dim]),
            3 => self.drift.store(1, Ordering::Relaxed),
            4 => {
                self.drift.store(1, Ordering::Relaxed);
                return Err(SearchError::EmbeddingFailed {
                    model: "mrl-fixture".to_owned(),
                    source: PRIVATE.into(),
                });
            }
            5 => return Ok(vec![99.0; target_dim]),
            _ => {}
        }
        // Invoke the existing trait's canonical default rather than duplicate
        // its floating-point implementation in the test fixture.
        DefaultProjection.truncate_embedding(values, target_dim)
    }

    fn id(&self) -> &'static str {
        "mrl-fixture"
    }

    fn model_name(&self) -> &'static str {
        "MRL fixture"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

struct DefaultProjection;

impl Embedder for DefaultProjection {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async {
            Err(SearchError::EmbeddingFailed {
                model: "projection-only".to_owned(),
                source: "projection helper must never infer".into(),
            })
        })
    }
    fn dimension(&self) -> usize {
        DIMENSION
    }

    fn id(&self) -> &'static str {
        "projection-only"
    }

    fn model_name(&self) -> &'static str {
        "Projection only"
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

async fn infer(
    reduced: &DimReduceEmbedder,
    cx: &Cx,
    batch: bool,
) -> SearchResult<Vec<IdentityBoundEmbedding>> {
    if batch {
        reduced.embed_batch_bound(cx, &["a", "b"]).await
    } else {
        reduced
            .embed_bound(cx, "a")
            .await
            .map(|response| vec![response])
    }
}

fn assert_redacted(error: &SearchError) {
    let rendered = error.to_string();
    assert!(!rendered.contains(PRIVATE));
    assert!(rendered.len() < 512);
}

fn assert_projected(
    actual: &IdentityBoundEmbedding,
    full: &IdentityBoundEmbedding,
    expected_identity: &EmbeddingIdentityBundleV1,
) {
    actual.validate().unwrap();
    assert_eq!(&actual.identity, expected_identity);
    let expected = DefaultProjection
        .truncate_embedding(&full.values, 2)
        .unwrap();
    assert_eq!(
        actual
            .values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn bound_single_uses_the_actual_bound_response_and_the_existing_projection() {
    run_test_with_cx(|cx| async move {
        let source = Arc::new(Provider::new(true));
        let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
        let expected_identity = source
            .identity
            .derive_projection(
                2,
                "prefix-truncate-first-n-dimensions-v1",
                "l2-f32-zero-on-degenerate-v1",
            )
            .unwrap();
        for text in ["a", "b", "zero"] {
            let response = reduced.embed_bound(&cx, text).await.unwrap();
            assert_projected(&response, &source.full(text), &expected_identity);
        }
        source.assert_calls(0, 3, 0, 0);
        // Raw inference remains explicitly raw, and cannot satisfy the bound path.
        assert_eq!(reduced.embed(&cx, "a").await.unwrap(), vec![1.0, 0.0]);
        source.assert_calls(1, 3, 0, 1);
    });
}

#[test]
fn native_batches_preserve_order_duplicates_and_empty_dispatch() {
    run_test_with_cx(|cx| async move {
        let source = Arc::new(Provider::new(true));
        let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
        let texts = ["a", "b", "a", "zero"];
        let responses = reduced.embed_batch_bound(&cx, &texts).await.unwrap();
        assert_eq!(responses.len(), texts.len());
        for (response, text) in responses.iter().zip(texts) {
            assert_projected(response, &source.full(text), reduced.identity().unwrap());
        }
        source.assert_calls(0, 0, 1, 0);
        assert!(
            reduced
                .embed_batch_bound(&cx, &[])
                .await
                .unwrap()
                .is_empty()
        );
        source.assert_calls(0, 0, 2, 0);
        assert_eq!(
            *source.inputs.lock().unwrap(),
            vec![vec!["a", "b", "a", "zero"], vec![]]
        );
    });
}

#[test]
fn non_native_batches_use_bound_singles_and_never_the_raw_backed_batch() {
    run_test_with_cx(|cx| async move {
        let source = Arc::new(Provider::new(false));
        let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
        assert!(reduced.bound_batch_is_native());
        let texts = ["a", "b", "a"];
        let responses = reduced.embed_batch_bound(&cx, &texts).await.unwrap();
        assert_eq!(responses.len(), texts.len());
        for (response, text) in responses.iter().zip(texts) {
            assert_projected(response, &source.full(text), reduced.identity().unwrap());
        }
        assert!(
            reduced
                .embed_batch_bound(&cx, &[])
                .await
                .unwrap()
                .is_empty()
        );
        source.assert_calls(0, 3, 0, 0);
        assert_eq!(
            *source.inputs.lock().unwrap(),
            vec![vec!["a"], vec!["b"], vec!["a"]]
        );
    });
}

#[test]
fn foreign_and_malformed_full_responses_are_refused_before_any_projection() {
    for batch in [false, true] {
        for fault in [
            Fault::Foreign,
            Fault::Storage,
            Fault::Quantization,
            Fault::Short,
            Fault::Long,
            Fault::NanPrefix,
            Fault::NanTail,
            Fault::InfinityTail,
            Fault::NegativeInfinityTail,
        ] {
            run_test_with_cx(|cx| async move {
                let source = Arc::new(Provider::new(true));
                let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
                *source.fault.lock().unwrap() = fault;
                let error = infer(&reduced, &cx, batch).await.unwrap_err();
                assert!(matches!(
                    error,
                    SearchError::UnverifiableRemoteSpace { .. } | SearchError::InvalidConfig { .. }
                ));
                assert_redacted(&error);
                source.assert_calls(0, usize::from(!batch), usize::from(batch), 0);
            });
        }
    }
}

#[test]
fn batch_cardinality_is_all_or_error_including_empty_input() {
    for empty in [false, true] {
        for fault in [Fault::ShortBatch, Fault::LongBatch] {
            if empty && matches!(fault, Fault::ShortBatch) {
                continue;
            }
            run_test_with_cx(|cx| async move {
                let source = Arc::new(Provider::new(true));
                let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
                *source.fault.lock().unwrap() = fault;
                let texts: &[&str] = if empty { &[] } else { &["a", "b"] };
                let error = reduced.embed_batch_bound(&cx, texts).await.unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                    if field == "embedding.mrl.batch_length"));
                source.assert_calls(0, 0, 1, 0);
            });
        }
    }
}

#[test]
fn drift_after_construction_cannot_retarget_the_projection_or_empty_batches() {
    for drift in 1..=4 {
        run_test_with_cx(|cx| async move {
            let source = Arc::new(Provider::new(true));
            let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
            source.drift.store(drift, Ordering::Relaxed);
            for error in [
                reduced.identity().unwrap_err(),
                reduced.embed_bound(&cx, "a").await.unwrap_err(),
                reduced.embed_batch_bound(&cx, &[]).await.unwrap_err(),
            ] {
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                assert_redacted(&error);
            }
            source.assert_calls(0, 0, 0, 0);
        });
    }
}

#[test]
fn source_drift_during_inference_dominates_both_values_and_provider_errors() {
    for batch in [false, true] {
        for outcome in [Outcome::DriftSuccess, Outcome::DriftFailure] {
            run_test_with_cx(|cx| async move {
                let source = Arc::new(Provider::new(true));
                let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
                *source.outcome.lock().unwrap() = outcome;
                let error = infer(&reduced, &cx, batch).await.unwrap_err();
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                assert_redacted(&error);
                source.assert_calls(0, usize::from(!batch), usize::from(batch), 0);
            });
        }
    }
}

#[test]
fn cancellation_outranks_success_failure_and_metadata_loss() {
    for batch in [false, true] {
        for outcome in [Outcome::CancelSuccess, Outcome::CancelFailure] {
            run_test_with_cx(|cx| async move {
                let source = Arc::new(Provider::new(true));
                let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
                *source.outcome.lock().unwrap() = outcome;
                assert!(matches!(
                    infer(&reduced, &cx, batch).await,
                    Err(SearchError::Cancelled { .. })
                ));
                source.assert_calls(0, usize::from(!batch), usize::from(batch), 0);
            });
        }
        run_test_with_cx(|cx| async move {
            let source = Arc::new(Provider::new(true));
            let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
            source.drift.store(2, Ordering::Relaxed);
            *source.cancel_metadata.lock().unwrap() = Some(cx.clone());
            assert!(matches!(
                infer(&reduced, &cx, batch).await,
                Err(SearchError::Cancelled { .. })
            ));
            source.assert_calls(0, 0, 0, 0);
        });
    }
}

#[test]
fn explicit_provider_cancellation_does_not_reenter_metadata_callbacks() {
    for batch in [false, true] {
        run_test_with_cx(|cx| async move {
            let source = Arc::new(Provider::new(true));
            let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
            *source.outcome.lock().unwrap() = Outcome::ExplicitCancel;
            let error = infer(&reduced, &cx, batch).await.unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { ref phase, .. }
                if phase == "fixture-provider"));
            assert_eq!(
                source.metadata_reads.load(Ordering::Relaxed),
                source.reads_at_explicit_cancel.load(Ordering::Relaxed)
            );
            source.assert_calls(0, usize::from(!batch), usize::from(batch), 0);
        });
    }
}

#[test]
fn stable_provider_failures_remain_failures_without_raw_fallback() {
    for batch in [false, true] {
        run_test_with_cx(|cx| async move {
            let source = Arc::new(Provider::new(true));
            let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
            *source.outcome.lock().unwrap() = Outcome::Failure;
            let error = infer(&reduced, &cx, batch).await.unwrap_err();
            assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
            source.assert_calls(0, usize::from(!batch), usize::from(batch), 0);
        });
    }
}

#[test]
fn provider_truncation_overrides_cannot_substitute_for_the_bound_projection() {
    for batch in [false, true] {
        for fault in 1..=5 {
            run_test_with_cx(|cx| async move {
                let source = Arc::new(Provider::new(true));
                let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
                source.projection_fault.store(fault, Ordering::Relaxed);
                let responses = infer(&reduced, &cx, batch).await.unwrap();
                let texts: &[&str] = if batch { &["a", "b"] } else { &["a"] };
                assert_eq!(responses.len(), texts.len());
                for (response, text) in responses.iter().zip(texts) {
                    assert_projected(response, &source.full(text), reduced.identity().unwrap());
                }
                source.assert_calls(0, usize::from(!batch), usize::from(batch), 0);
                assert_eq!(source.drift.load(Ordering::Relaxed), 0);
            });
        }
    }
}

#[test]
fn unpolled_precancelled_and_abandoned_requests_do_not_project() {
    run_test_with_cx(|cx| async move {
        let source = Arc::new(Provider::new(true));
        let reduced = DimReduceEmbedder::new(source.clone(), 2).unwrap();
        let before = source.metadata_reads.load(Ordering::Relaxed);
        drop(reduced.embed_bound(&cx, "a"));
        drop(reduced.embed_batch_bound(&cx, &["a", "b"]));
        assert_eq!(source.metadata_reads.load(Ordering::Relaxed), before);
        source.assert_calls(0, 0, 0, 0);
        *source.outcome.lock().unwrap() = Outcome::Yield;
        let texts = ["a", "b"];
        let mut future = Box::pin(reduced.embed_batch_bound(&cx, &texts));
        assert!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(future);
        source.assert_calls(0, 0, 1, 0);
        cx.cancel_fast(CancelKind::User);
        assert!(matches!(
            reduced.embed_bound(&cx, "a").await,
            Err(SearchError::Cancelled { .. })
        ));
        assert!(matches!(
            reduced.embed_batch_bound(&cx, &texts).await,
            Err(SearchError::Cancelled { .. })
        ));
        source.assert_calls(0, 0, 1, 0);
    });
}

#[test]
fn projected_cache_hits_do_not_hide_later_source_drift() {
    run_test_with_cx(|cx| async move {
        let source = Arc::new(Provider::new(true));
        let reduced = Arc::new(DimReduceEmbedder::new(source.clone(), 2).unwrap());
        let cached = CachedEmbedder::new(reduced, 4);
        let first = cached.embed_bound(&cx, "a").await.unwrap();
        assert_eq!(cached.embed_bound(&cx, "a").await.unwrap(), first);
        source.assert_calls(0, 1, 0, 0);
        assert_eq!(cached.cache_stats().hits, 1);
        source.drift.store(1, Ordering::Relaxed);
        assert!(matches!(
            cached.embed_bound(&cx, "a").await,
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        source.assert_calls(0, 1, 0, 0);
        assert_eq!(cached.cache_stats().hits, 1);
    });
}

#[test]
fn nested_projections_keep_the_complete_source_chain_and_bound_dispatch() {
    run_test_with_cx(|cx| async move {
        let source = Arc::new(Provider::new(false));
        let first = Arc::new(DimReduceEmbedder::new(source.clone(), 3).unwrap());
        let reduced = DimReduceEmbedder::new(first.clone(), 2).unwrap();
        let responses = reduced.embed_batch_bound(&cx, &["a", "b"]).await.unwrap();
        for (response, text) in responses.iter().zip(["a", "b"]) {
            let intermediate = DefaultProjection
                .truncate_embedding(&source.full(text).values, 3)
                .unwrap();
            let expected = DefaultProjection
                .truncate_embedding(&intermediate, 2)
                .unwrap();
            assert_eq!(response.values, expected);
            assert_eq!(&response.identity, reduced.identity().unwrap());
        }
        source.assert_calls(0, 2, 0, 0);
        source.drift.store(1, Ordering::Relaxed);
        assert!(first.identity().is_err());
        assert!(reduced.identity().is_err());
        assert!(reduced.embed_bound(&cx, "a").await.is_err());
        source.assert_calls(0, 2, 0, 0);
    });
}

#[test]
fn constructor_refuses_non_native_source_storage_and_dimension_disagreement() {
    let mut source = Provider::new(true);
    source.identity.storage.quantization = QuantizationFormat::F16;
    assert!(matches!(
        DimReduceEmbedder::new(Arc::new(source), 2),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
    let source = Arc::new(Provider::new(true));
    source.drift.store(3, Ordering::Relaxed);
    assert!(matches!(
        DimReduceEmbedder::new(source.clone(), 2),
        Err(SearchError::UnverifiableRemoteSpace { .. })
    ));
    source.assert_calls(0, 0, 0, 0);
}
