use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::{IdentityBoundEmbedding, ModelCategory, SearchFuture};

use super::*;

fn upsert(id: &str, text: &str) -> RetainedMutation {
    RetainedMutation::Upsert { id: id.to_owned(), text: text.to_owned() }
}

fn delete(id: &str) -> RetainedMutation {
    RetainedMutation::Delete { id: id.to_owned() }
}

#[test]
fn ordered_mutations_are_last_write_wins_per_exact_id() {
    run_test_with_cx(|cx| async move {
        let prepared = prepare(&cx, &[
            upsert("a", "discarded"), delete("a"), upsert("b", "discarded"),
            upsert("a", "final α"), delete("b"), upsert("b-prefix", "kept"),
        ]).unwrap();
        assert_eq!(prepared.len(), 3);
        assert_eq!(prepared["a"].as_ref().unwrap().embedding, "final α");
        assert!(prepared["b"].is_none());
        assert_eq!(prepared["b-prefix"].as_ref().unwrap().lexical, "kept");
    });
}

#[test]
fn invalid_superseded_operations_cannot_be_hidden_by_later_input() {
    run_test_with_cx(|cx| async move {
        for id in ["", "  ", "a\0window"] {
            assert!(prepare(&cx, &[delete(id), upsert("valid", "body")]).is_err());
        }
        let too_long = "x".repeat(usize::from(u16::MAX) + 1);
        assert!(prepare(&cx, &[delete(&too_long)]).is_err());
        assert!(prepare(&cx, &[upsert("a", " \n ")]).is_err());
        // An empty intermediate body is superseded before canonicalization.
        assert!(prepare(&cx, &[upsert("a", ""), upsert("a", "final")]).is_ok());
        assert!(prepare(&cx, &[upsert("a", &"x".repeat(MAX_BODY_BYTES + 1)), delete("a")]).is_err());
    });
}

#[test]
fn operation_bounds_and_vector_accounting_are_exact() {
    run_test_with_cx(|cx| async move {
        let accepted = vec![delete("same"); MAX_OPERATIONS];
        assert_eq!(prepare(&cx, &accepted).unwrap().len(), 1);
        assert!(prepare(&cx, &vec![delete("same"); MAX_OPERATIONS + 1]).is_err());
        let mut bytes = MAX_VECTOR_BYTES - 4;
        charge_vector(&mut bytes, &[1.0]).unwrap();
        assert_eq!(bytes, MAX_VECTOR_BYTES);
        assert!(charge_vector(&mut bytes, &[1.0]).is_err());
        assert_eq!(bytes, MAX_VECTOR_BYTES, "failed admission cannot change its accounting");
    });
}

#[test]
fn empty_and_rejected_batches_do_not_touch_a_store() {
    run_test_with_cx(|cx| async move {
        let parent = tempfile::tempdir().unwrap();
        let store = CompleteGenerationStore::create(&cx, &parent.path().join("base")).unwrap();
        let build = store.begin(&cx).unwrap();
        std::fs::write(build.path().join("fixture"), "receipt only").unwrap();
        let GenerationPublication::Durable(expected) = build.publish(&cx, |_, _| Ok(())).unwrap()
        else { panic!("fixture sync"); }; // ubs:ignore — test assertion.
        let missing = parent.path().join("must-not-exist");
        let runtime = FsfsRuntime::new(crate::FsfsConfig::default());
        let outcome = runtime.apply_retained_batch(&cx, &missing, &expected, &[]).await.unwrap();
        assert!(outcome.publication.is_none());
        assert_eq!((outcome.upserted, outcome.deleted), (0, 0));
        assert!(runtime.apply_retained_batch(&cx, &missing, &expected, &[delete("")]).await.is_err());
        cx.set_cancel_requested(true);
        let cancelled = runtime.apply_retained_batch(&cx, &missing, &expected, &[]).await;
        cx.set_cancel_requested(false);
        assert!(matches!(cancelled, Err(SearchError::Cancelled { .. })));
        assert!(!missing.exists());
    });
}

#[derive(Clone, Copy)]
enum Reply { Good, Error, Foreign, Nan, Infinite, Short, Cancel, CancelError, Drift, DriftError }

struct TestProducer {
    identity: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    reply: Reply,
    drifted: AtomicBool,
    calls: AtomicUsize,
    raw_calls: AtomicUsize,
}

impl TestProducer {
    fn new(reply: Reply) -> Self {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model("retained-batch-test", 3);
        let mut foreign = identity.clone();
        foreign.producer.implementation_revision.push_str("-foreign");
        Self { identity, foreign, reply, drifted: AtomicBool::new(false), calls: AtomicUsize::new(0), raw_calls: AtomicUsize::new(0) }
    }
}

impl Embedder for TestProducer {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.drifted.load(Ordering::SeqCst) { &self.foreign } else { &self.identity })
    }
    fn id(&self) -> &str { "retained-batch-test" }
    fn model_name(&self) -> &str { "Explicit batch wiring fixture" }
    fn dimension(&self) -> usize { 3 }
    fn is_semantic(&self) -> bool { true }
    fn category(&self) -> ModelCategory { ModelCategory::TransformerEmbedder }
    fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![1.0, 0.0, 0.0])
        })
    }
    fn embed_bound<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if matches!(self.reply, Reply::Cancel | Reply::CancelError) { cx.set_cancel_requested(true); }
            if matches!(self.reply, Reply::Drift | Reply::DriftError) { self.drifted.store(true, Ordering::SeqCst); }
            if matches!(self.reply, Reply::Error | Reply::CancelError | Reply::DriftError)
                || text.contains("rejectquality")
            {
                return Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(), source: std::io::Error::other("fixture failure").into(),
                });
            }
            let values = match self.reply {
                Reply::Nan => vec![f32::NAN, 0.0, 0.0],
                Reply::Infinite => vec![0.0, f32::INFINITY, 0.0],
                Reply::Short => vec![1.0],
                _ => vec![1.0, 0.0, 0.0],
            };
            let identity = if matches!(self.reply, Reply::Foreign) { self.foreign.clone() } else { self.identity.clone() };
            Ok(IdentityBoundEmbedding { values, identity })
        })
    }
}

fn producer(reply: Reply) -> (Producer, Arc<TestProducer>) {
    let embedder = Arc::new(TestProducer::new(reply));
    (Producer { identity: embedder.identity.clone(), embedder: embedder.clone(), role: "test.batch" }, embedder)
}

#[test]
fn producer_validation_refuses_foreign_nonfinite_and_wrong_width_responses() {
    run_test_with_cx(|cx| async move {
        for reply in [Reply::Foreign, Reply::Nan, Reply::Infinite, Reply::Short] {
            let (producer, witness) = producer(reply);
            assert!(producer.infer(&cx, "body").await.is_err());
            assert_eq!(witness.calls.load(Ordering::SeqCst), 1);
            assert_eq!(witness.raw_calls.load(Ordering::SeqCst), 0);
        }
        let (producer, witness) = producer(Reply::Good);
        assert_eq!(producer.infer(&cx, "body").await.unwrap(), [1.0, 0.0, 0.0]);
        assert_eq!(witness.calls.load(Ordering::SeqCst), 1);
        assert_eq!(witness.raw_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn producer_drift_during_success_or_error_is_terminal_without_retries() {
    run_test_with_cx(|cx| async move {
        for reply in [Reply::Drift, Reply::DriftError] {
            let (producer, witness) = producer(reply);
            assert!(matches!(producer.infer(&cx, "body").await,
                Err(SearchError::UnverifiableRemoteSpace { .. })));
            assert_eq!(witness.calls.load(Ordering::SeqCst), 1);
            assert_eq!(witness.raw_calls.load(Ordering::SeqCst), 0);
            assert!(producer.infer(&cx, "later").await.is_err());
            assert_eq!(witness.calls.load(Ordering::SeqCst), 1);
            assert_eq!(witness.raw_calls.load(Ordering::SeqCst), 0);
        }
        let (producer, _) = producer(Reply::Error);
        assert!(matches!(producer.infer(&cx, "body").await, Err(SearchError::EmbeddingFailed { .. })));
    });
}

#[test]
fn cancellation_wins_over_provider_success_and_failure() {
    run_test_with_cx(|cx| async move {
        for reply in [Reply::Cancel, Reply::CancelError] {
            let (producer, witness) = producer(reply);
            let result = producer.infer(&cx, "body").await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(witness.calls.load(Ordering::SeqCst), 1);
            assert_eq!(witness.raw_calls.load(Ordering::SeqCst), 0);
        }
        let (producer, witness) = producer(Reply::Good);
        cx.set_cancel_requested(true);
        let result = producer.infer(&cx, "body").await;
        cx.set_cancel_requested(false);
        assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        assert_eq!(witness.calls.load(Ordering::SeqCst), 0);
    });
}

#[cfg(all(feature = "semantic-support", not(feature = "embedded-models")))]
mod persisted;
