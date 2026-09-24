//! Public-trait regression tests for inference identity and cancellation.
//!
//! These synthetic providers deliberately disagree between raw and bound calls.
//! They exercise adapter dispatch and default contracts, not model quality.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use asupersync::test_utils::run_test_with_cx;
use asupersync::{CancelKind, Cx};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{
    Embedder, IdentityBoundEmbedding, ModelCategory, RerankDocument, RerankScore, Reranker,
    SearchFuture, SyncEmbed, SyncEmbedderAdapter, SyncRerank, SyncRerankerAdapter,
};
use frankensearch_core::{SearchError, SearchResult};

#[derive(Clone, Copy)]
enum Effect {
    None,
    BoundError,
    Cancel,
    CancelAndError,
    IdentityDrift,
    DimensionDrift,
    LostIdentity,
}

struct NativeProvider {
    cx: Cx,
    identity: EmbeddingIdentityBundleV1,
    changed_identity: EmbeddingIdentityBundleV1,
    changed: AtomicBool,
    effect: Effect,
    outputs: Vec<IdentityBoundEmbedding>,
    raw_calls: AtomicUsize,
    bound_calls: AtomicUsize,
}

impl NativeProvider {
    fn new(cx: &Cx) -> Self {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model("bridge-contract", 3);
        let mut changed_identity = identity.clone();
        changed_identity
            .producer
            .implementation_revision
            .push_str("-private-canary");
        Self {
            cx: cx.clone(),
            identity: identity.clone(),
            changed_identity,
            changed: AtomicBool::new(false),
            effect: Effect::None,
            outputs: vec![
                IdentityBoundEmbedding {
                    values: vec![0.0, -0.0, f32::from_bits(1)],
                    identity: identity.clone(),
                },
                IdentityBoundEmbedding {
                    values: vec![2.0, -3.0, 4.0],
                    identity,
                },
            ],
            raw_calls: AtomicUsize::new(0),
            bound_calls: AtomicUsize::new(0),
        }
    }

    fn dispatch(&self, bound: bool) -> SearchResult<()> {
        if bound {
            self.bound_calls.fetch_add(1, Ordering::Relaxed);
        } else {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
        }
        match self.effect {
            Effect::BoundError if bound => return Err(backend_error()),
            Effect::Cancel => {
                self.cx.cancel_fast(CancelKind::User);
            }
            Effect::CancelAndError => {
                self.cx.cancel_fast(CancelKind::User);
                return Err(backend_error());
            }
            Effect::IdentityDrift | Effect::DimensionDrift | Effect::LostIdentity => {
                self.changed.store(true, Ordering::Relaxed);
            }
            Effect::None | Effect::BoundError => {}
        }
        Ok(())
    }

    fn current_identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        if self.changed.load(Ordering::Relaxed) {
            match self.effect {
                Effect::IdentityDrift => return Ok(&self.changed_identity),
                Effect::LostIdentity => return Err(backend_error()),
                _ => {}
            }
        }
        Ok(&self.identity)
    }

    fn current_dimension(&self) -> usize {
        if self.changed.load(Ordering::Relaxed) && matches!(self.effect, Effect::DimensionDrift) {
            4
        } else {
            3
        }
    }

    fn counts(&self) -> (usize, usize) {
        (
            self.raw_calls.load(Ordering::Relaxed),
            self.bound_calls.load(Ordering::Relaxed),
        )
    }
}

fn backend_error() -> SearchError {
    SearchError::EmbeddingFailed {
        model: "bridge-contract".to_owned(),
        source: "native bound operation refused".into(),
    }
}

impl SyncEmbed for NativeProvider {
    fn embed_sync(&self, _: &str) -> SearchResult<Vec<f32>> {
        self.dispatch(false)?;
        Ok(vec![99.0; 3])
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        self.dispatch(false)?;
        Ok(vec![vec![99.0; 3]; texts.len()])
    }

    fn embed_bound_sync(&self, _: &str) -> SearchResult<IdentityBoundEmbedding> {
        self.dispatch(true)?;
        Ok(self.outputs[0].clone())
    }

    fn embed_batch_bound_sync(&self, _: &[&str]) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        self.dispatch(true)?;
        Ok(self.outputs.clone())
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.current_identity()
    }

    fn dimension(&self) -> usize {
        self.current_dimension()
    }

    fn id(&self) -> &'static str {
        "bridge-contract"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

fn signature(response: &IdentityBoundEmbedding) -> Vec<u32> {
    response
        .values
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn assert_identity_error(error: &SearchError) {
    assert!(!error.to_string().contains("private-canary"));
    assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
}

#[test]
fn dyn_adapter_preserves_native_bound_dispatch_values_identity_and_batch_order() {
    run_test_with_cx(|cx| async move {
        let adapter = SyncEmbedderAdapter(NativeProvider::new(&cx));
        let provider: &dyn Embedder = &adapter;
        let single = provider.embed_bound(&cx, "single").await.unwrap();
        let batch = provider
            .embed_batch_bound(&cx, &["first", "second"])
            .await
            .unwrap();
        assert_eq!(signature(&single), signature(&adapter.0.outputs[0]));
        assert_eq!(single.identity, adapter.0.identity);
        assert_eq!(batch.len(), 2);
        for (actual, expected) in batch.iter().zip(&adapter.0.outputs) {
            assert_eq!(signature(actual), signature(expected));
            assert_eq!(actual.identity, expected.identity);
        }
        assert_eq!(adapter.0.counts(), (0, 2));
        assert_eq!(provider.embed(&cx, "raw").await.unwrap(), vec![99.0; 3]);
        assert_eq!(adapter.0.counts(), (1, 2));
    });
}

#[test]
fn native_bound_failure_is_not_replaced_with_successful_raw_inference() {
    run_test_with_cx(|cx| async move {
        let mut native = NativeProvider::new(&cx);
        native.effect = Effect::BoundError;
        let adapter = SyncEmbedderAdapter(native);
        for batch in [false, true] {
            let result = if batch {
                adapter
                    .embed_batch_bound(&cx, &["a", "b"])
                    .await
                    .map(|_| ())
            } else {
                adapter.embed_bound(&cx, "a").await.map(|_| ())
            };
            let error = result.unwrap_err();
            assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
            assert!(error.to_string().contains("native bound operation refused"));
        }
        assert_eq!(adapter.0.counts(), (0, 2));
        assert!(adapter.embed(&cx, "raw still works").await.is_ok());
    });
}

#[test]
fn foreign_bound_contracts_are_rejected_without_disclosing_the_response() {
    run_test_with_cx(|cx| async move {
        for mutation in 0..5 {
            let mut native = NativeProvider::new(&cx);
            let foreign = &mut native.outputs[0].identity;
            match mutation {
                0 => foreign.space.logical_model_id.push_str("-private-canary"),
                1 => foreign
                    .producer
                    .implementation_revision
                    .push_str("-private-canary"),
                2 => "private-canary-fsvi".clone_into(&mut foreign.storage.format),
                3 => foreign.storage.quantization = QuantizationFormat::F16,
                _ => "private-canary-endian".clone_into(&mut foreign.storage.endianness),
            }
            let adapter = SyncEmbedderAdapter(native);
            assert_identity_error(&adapter.embed_bound(&cx, "input").await.unwrap_err());
            assert_identity_error(
                &adapter
                    .embed_batch_bound(&cx, &["a", "b"])
                    .await
                    .unwrap_err(),
            );
            assert_eq!(adapter.0.counts(), (0, 2));
        }
    });
}

#[test]
fn overridden_bound_values_and_late_batch_rows_are_validated_without_repair() {
    run_test_with_cx(|cx| async move {
        for values in [
            vec![],
            vec![1.0; 2],
            vec![1.0; 4],
            vec![1.0, f32::NAN, 0.0],
            vec![1.0, f32::INFINITY, 0.0],
            vec![1.0, f32::NEG_INFINITY, 0.0],
        ] {
            let mut native = NativeProvider::new(&cx);
            native.outputs[0].values = values.clone();
            let adapter = SyncEmbedderAdapter(native);
            assert!(matches!(
                adapter.embed_bound(&cx, "input").await,
                Err(SearchError::InvalidConfig { field, .. })
                    if field == "identity_bound_embedding.values"
            ));
            let mut native = NativeProvider::new(&cx);
            native.outputs[1].values = values;
            let adapter = SyncEmbedderAdapter(native);
            assert!(matches!(
                adapter.embed_batch_bound(&cx, &["valid", "invalid"]).await,
                Err(SearchError::InvalidConfig { field, .. })
                    if field == "identity_bound_embedding.values"
            ));
            assert_eq!(adapter.0.counts(), (0, 1));
        }
    });
}

#[test]
fn bound_override_batch_cardinality_is_checked_even_for_empty_inputs() {
    run_test_with_cx(|cx| async move {
        for input_count in 0..=3 {
            let inputs = vec!["input"; input_count];
            for output_count in 0..=4 {
                let mut native = NativeProvider::new(&cx);
                native.outputs = vec![native.outputs[0].clone(); output_count];
                let adapter = SyncEmbedderAdapter(native);
                let result = adapter.embed_batch_bound(&cx, &inputs).await;
                if input_count == output_count {
                    assert_eq!(result.unwrap().len(), input_count);
                } else {
                    assert!(matches!(
                        result,
                        Err(SearchError::InvalidConfig { field, .. })
                            if field == "embedder.batch_length"
                    ));
                }
                assert_eq!(adapter.0.counts(), (0, 1));
            }
        }
    });
}

#[test]
fn producer_identity_dimension_or_availability_drift_cannot_escape_the_bridge() {
    run_test_with_cx(|cx| async move {
        for effect in [
            Effect::IdentityDrift,
            Effect::DimensionDrift,
            Effect::LostIdentity,
        ] {
            for batch in [false, true] {
                let mut native = NativeProvider::new(&cx);
                native.effect = effect;
                let adapter = SyncEmbedderAdapter(native);
                let error = if batch {
                    adapter
                        .embed_batch_bound(&cx, &["a", "b"])
                        .await
                        .unwrap_err()
                } else {
                    adapter.embed_bound(&cx, "a").await.unwrap_err()
                };
                assert_identity_error(&error);
                assert_eq!(adapter.0.counts(), (0, 1));
            }
        }
    });
}

async fn call_adapter(provider: &dyn Embedder, cx: &Cx, operation: usize) -> SearchResult<()> {
    match operation {
        0 => provider.embed(cx, "input").await.map(|_| ()),
        1 => provider.embed_batch(cx, &["a", "b"]).await.map(|_| ()),
        2 => provider.embed_bound(cx, "input").await.map(|_| ()),
        _ => provider
            .embed_batch_bound(cx, &["a", "b"])
            .await
            .map(|_| ()),
    }
}

#[test]
fn all_adapter_operations_are_lazy_and_precancelled_calls_do_no_inference() {
    run_test_with_cx(|cx| async move {
        let adapter = SyncEmbedderAdapter(NativeProvider::new(&cx));
        drop(adapter.embed(&cx, "raw"));
        drop(adapter.embed_batch(&cx, &["a", "b"]));
        drop(adapter.embed_bound(&cx, "bound"));
        drop(adapter.embed_batch_bound(&cx, &["a", "b"]));
        assert_eq!(adapter.0.counts(), (0, 0));
        cx.cancel_fast(CancelKind::User);
        for operation in 0..4 {
            assert!(matches!(
                call_adapter(&adapter, &cx, operation).await,
                Err(SearchError::Cancelled { .. })
            ));
        }
        assert_eq!(adapter.0.counts(), (0, 0));
    });
}

#[test]
fn all_adapter_operations_give_cancellation_precedence_over_success_and_failure() {
    for effect in [Effect::Cancel, Effect::CancelAndError] {
        for operation in 0..4 {
            run_test_with_cx(|cx| async move {
                let mut native = NativeProvider::new(&cx);
                native.effect = effect;
                let adapter = SyncEmbedderAdapter(native);
                assert!(matches!(
                    call_adapter(&adapter, &cx, operation).await,
                    Err(SearchError::Cancelled { .. })
                ));
                let (raw, bound) = adapter.0.counts();
                assert_eq!(raw + bound, 1);
            });
        }
    }
}

#[test]
fn invalid_advertised_contract_is_rejected_before_native_bound_dispatch() {
    run_test_with_cx(|cx| async move {
        for dimension in [false, true] {
            let mut native = NativeProvider::new(&cx);
            if dimension {
                native.identity.space.dimension = 4;
            } else {
                "fsvi-v2".clone_into(&mut native.identity.storage.format);
            }
            let adapter = SyncEmbedderAdapter(native);
            assert!(adapter.embed_bound(&cx, "a").await.is_err());
            assert!(adapter.embed_batch_bound(&cx, &[]).await.is_err());
            assert_eq!(adapter.0.counts(), (0, 0));
            assert!(adapter.embed(&cx, "raw").await.is_ok());
        }
    });
}

// These wrappers intentionally inherit the default bound operations instead of
// the native overrides above, exposing metadata changes in raw inference.
struct RawDefaults(NativeProvider);

impl SyncEmbed for RawDefaults {
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        self.0.embed_sync(text)
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.0.current_identity()
    }

    fn dimension(&self) -> usize {
        self.0.current_dimension()
    }

    fn id(&self) -> &'static str {
        "raw-defaults"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

struct AsyncDefaults(NativeProvider);

impl Embedder for AsyncDefaults {
    fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { self.0.embed_sync(text) })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.0.current_identity()
    }

    fn dimension(&self) -> usize {
        self.0.current_dimension()
    }

    fn id(&self) -> &'static str {
        "async-defaults"
    }

    fn model_name(&self) -> &'static str {
        "Async default contract fixture"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[test]
fn default_sync_and_async_bound_operations_recheck_the_captured_producer() {
    run_test_with_cx(|cx| async move {
        for effect in [
            Effect::IdentityDrift,
            Effect::DimensionDrift,
            Effect::LostIdentity,
        ] {
            for batch in [false, true] {
                let mut native = NativeProvider::new(&cx);
                native.effect = effect;
                let synchronous = RawDefaults(native);
                let error = if batch {
                    synchronous.embed_batch_bound_sync(&["a", "b"]).unwrap_err()
                } else {
                    synchronous.embed_bound_sync("a").unwrap_err()
                };
                assert_identity_error(&error);
                let mut native = NativeProvider::new(&cx);
                native.effect = effect;
                let asynchronous = AsyncDefaults(native);
                let error = if batch {
                    asynchronous
                        .embed_batch_bound(&cx, &["a", "b"])
                        .await
                        .unwrap_err()
                } else {
                    asynchronous.embed_bound(&cx, "a").await.unwrap_err()
                };
                assert_identity_error(&error);
            }
        }
    });
}

#[test]
fn default_async_operations_preserve_cancellation_when_raw_inference_returns_error() {
    for operation in 1..4 {
        run_test_with_cx(|cx| async move {
            let mut native = NativeProvider::new(&cx);
            native.effect = Effect::CancelAndError;
            let provider = AsyncDefaults(native);
            assert!(matches!(
                call_adapter(&provider, &cx, operation).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(provider.0.counts(), (1, 0));
        });
    }
}

struct NativeReranker(NativeProvider);

impl SyncRerank for NativeReranker {
    fn rerank_sync(&self, _: &str, _: &[RerankDocument]) -> SearchResult<Vec<RerankScore>> {
        self.0.dispatch(true)?;
        Ok(vec![RerankScore {
            doc_id: "document".to_owned(),
            score: 0.5,
            original_rank: 0,
            raw_logit: None,
        }])
    }

    fn id(&self) -> &'static str {
        "native-rerank-contract"
    }

    fn model_name(&self) -> &'static str {
        "Native rerank fixture"
    }
}

#[test]
fn sync_reranking_observes_cancellation_before_work_and_after_success_or_failure() {
    run_test_with_cx(|cx| async move {
        let adapter = SyncRerankerAdapter(NativeReranker(NativeProvider::new(&cx)));
        cx.cancel_fast(CancelKind::User);
        assert!(matches!(
            adapter.rerank(&cx, "query", &[]).await,
            Err(SearchError::Cancelled { .. })
        ));
        assert_eq!(adapter.0.0.counts(), (0, 0));
    });
    for effect in [
        Effect::None,
        Effect::BoundError,
        Effect::Cancel,
        Effect::CancelAndError,
    ] {
        run_test_with_cx(|cx| async move {
            let mut native = NativeProvider::new(&cx);
            native.effect = effect;
            let adapter = SyncRerankerAdapter(NativeReranker(native));
            let result = adapter.rerank(&cx, "query", &[]).await;
            match effect {
                Effect::None => assert_eq!(result.unwrap().len(), 1),
                Effect::BoundError => {
                    assert!(matches!(result, Err(SearchError::EmbeddingFailed { .. })));
                }
                _ => assert!(matches!(result, Err(SearchError::Cancelled { .. }))),
            }
            assert_eq!(adapter.0.0.counts(), (0, 1));
        });
    }
}
