use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use asupersync::test_utils::run_test_with_cx;
use frankensearch_core::generation::ArtifactGenerationIdentityV1;

use super::super::{NativeBuildPrecision, NativeIndexBuilder};
use crate::IndexableDocument;

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Foreign,
    Short,
    NonFinite,
    WrongWidth,
    InvalidConfig,
    TypedCancellation,
    CancelSuccess,
    CancelFailure,
    DriftSuccess,
    DriftFailure,
    Pending,
}

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    foreign: EmbeddingIdentityBundleV1,
    changed: AtomicBool,
    max_batch: usize,
    fault: Fault,
    fault_call: usize,
    calls: Mutex<Vec<Vec<String>>>,
    raw_calls: AtomicUsize,
    dropped: AtomicUsize,
}

impl Provider {
    fn new(name: &str, dimension: u32, max_batch: usize) -> Self {
        let identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
        let mut foreign = identity.clone();
        "private-canary-producer".clone_into(&mut foreign.producer.backend);
        Self {
            identity,
            foreign,
            changed: AtomicBool::new(false),
            max_batch,
            fault: Fault::None,
            fault_call: 0,
            calls: Mutex::new(Vec::new()),
            raw_calls: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }

    fn values(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.dimension()];
        values[0] = text.bytes().map(f32::from).sum::<f32>() / 1024.0;
        values[1] = -0.0;
        values
    }

    fn failure() -> SearchError {
        SearchError::EmbeddingFailed {
            model: "bounded-batch-fixture".to_owned(),
            source: "provider rejected this input batch".into(),
        }
    }
}

struct PendingDrop<'a>(&'a AtomicUsize);

impl Drop for PendingDrop<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.raw_calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.values(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let call = {
                let mut calls = self.calls.lock().unwrap();
                let call = calls.len();
                calls.push(texts.iter().map(|text| (*text).to_owned()).collect());
                call
            };
            let fault = if call == self.fault_call {
                self.fault
            } else {
                Fault::None
            };
            match fault {
                Fault::InvalidConfig => {
                    return Err(SearchError::InvalidConfig {
                        field: "test.provider".to_owned(),
                        value: String::new(),
                        reason: "configuration refusal must not be split".to_owned(),
                    });
                }
                Fault::TypedCancellation => {
                    return Err(SearchError::Cancelled {
                        phase: "provider.bound_batch".to_owned(),
                        reason: "provider cancelled".to_owned(),
                    });
                }
                Fault::CancelSuccess | Fault::CancelFailure => cx.set_cancel_requested(true),
                Fault::DriftSuccess | Fault::DriftFailure => {
                    self.changed.store(true, Ordering::SeqCst);
                }
                Fault::Pending => {
                    let _guard = PendingDrop(&self.dropped);
                    return std::future::pending().await;
                }
                _ => {}
            }
            if matches!(fault, Fault::CancelFailure | Fault::DriftFailure)
                || texts.len() > self.max_batch
                || texts.contains(&"poison")
            {
                return Err(Self::failure());
            }
            let mut response: Vec<_> = texts
                .iter()
                .map(|text| IdentityBoundEmbedding {
                    identity: self.identity.clone(),
                    values: self.values(text),
                })
                .collect();
            match fault {
                Fault::Foreign => response
                    .last_mut()
                    .unwrap()
                    .identity
                    .clone_from(&self.foreign),
                Fault::Short => {
                    let _ = response.pop();
                }
                Fault::NonFinite => response.last_mut().unwrap().values[0] = f32::NAN,
                Fault::WrongWidth => {
                    let _ = response.last_mut().unwrap().values.pop();
                }
                _ => {}
            }
            Ok(response)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(if self.changed.load(Ordering::SeqCst) {
            &self.foreign
        } else {
            &self.identity
        })
    }
    fn dimension(&self) -> usize {
        usize::try_from(self.identity.space.dimension).unwrap()
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
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

fn split(provider: &Arc<Provider>) -> Arc<dyn Embedder> {
    wrap(provider.clone(), provider.identity.clone())
}

fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
    ArtifactGenerationIdentityV1::new(sequence, [0x39; 16]).unwrap()
}

fn documents() -> Vec<IndexableDocument> {
    (0..7)
        .rev()
        .map(|number| IndexableDocument::new(format!("doc-{number}"), format!("text-{number}")))
        .collect()
}

#[test]
fn splitting_preserves_order_duplicates_identity_and_bits_without_raw_calls() {
    run_test_with_cx(|cx| async move {
        let provider = Arc::new(Provider::new("fast", 3, 2));
        let texts = ["a", "a", "bb", "ccc", "a"];
        let output = split(&provider)
            .embed_batch_bound(&cx, &texts)
            .await
            .unwrap();
        assert_eq!(output.len(), texts.len());
        for (row, text) in output.iter().zip(texts) {
            assert_eq!(row.identity, provider.identity);
            assert_eq!(
                row.values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                provider
                    .values(text)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            *provider.calls.lock().unwrap(),
            vec![
                vec!["a", "a", "bb", "ccc", "a"],
                vec!["a", "a"],
                vec!["bb", "ccc", "a"],
                vec!["bb"],
                vec!["ccc", "a"],
            ]
        );
        assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn splitting_has_a_finite_request_bound_and_no_retry_for_a_failed_singleton() {
    run_test_with_cx(|cx| async move {
        for length in 0..18 {
            let provider = Arc::new(Provider::new("fast", 2, 1));
            let texts = vec!["same"; length];
            assert_eq!(
                split(&provider)
                    .embed_batch_bound(&cx, &texts)
                    .await
                    .unwrap()
                    .len(),
                length
            );
            assert_eq!(
                provider.calls.lock().unwrap().len(),
                if length == 0 { 1 } else { 2 * length - 1 }
            );
        }
        let provider = Arc::new(Provider::new("fast", 2, 8));
        assert!(matches!(
            split(&provider).embed_batch_bound(&cx, &["poison"]).await,
            Err(SearchError::EmbeddingFailed { .. })
        ));
        assert_eq!(provider.calls.lock().unwrap().len(), 1);
    });
}

#[test]
fn splitting_does_not_retry_contract_failures_or_configuration_refusals() {
    run_test_with_cx(|cx| async move {
        for fault in [
            Fault::Foreign,
            Fault::Short,
            Fault::NonFinite,
            Fault::WrongWidth,
            Fault::InvalidConfig,
        ] {
            let mut provider = Provider::new("fast", 2, 8);
            provider.fault = fault;
            let provider = Arc::new(provider);
            let error = split(&provider)
                .embed_batch_bound(&cx, &["a", "b", "c"])
                .await
                .unwrap_err();
            assert!(!error.to_string().contains("private-canary"));
            assert_eq!(provider.calls.lock().unwrap().len(), 1);
            assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn splitting_observes_cancellation_and_producer_drift_before_retrying() {
    run_test_with_cx(|cx| async move {
        for fault in [
            Fault::CancelSuccess,
            Fault::CancelFailure,
            Fault::TypedCancellation,
            Fault::DriftSuccess,
            Fault::DriftFailure,
        ] {
            let mut provider = Provider::new("fast", 2, 8);
            provider.fault = fault;
            let provider = Arc::new(provider);
            let error = split(&provider)
                .embed_batch_bound(&cx, &["a", "b"])
                .await
                .unwrap_err();
            if matches!(fault, Fault::DriftSuccess | Fault::DriftFailure) {
                assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
            } else {
                assert!(matches!(error, SearchError::Cancelled { .. }));
            }
            cx.set_cancel_requested(false);
            assert_eq!(provider.calls.lock().unwrap().len(), 1);
        }
        let provider = Arc::new(Provider::new("fast", 2, 1));
        let wrapper = split(&provider);
        let texts = ["a", "b"];
        let future = wrapper.embed_batch_bound(&cx, &texts);
        assert!(provider.calls.lock().unwrap().is_empty());
        cx.set_cancel_requested(true);
        assert!(matches!(future.await, Err(SearchError::Cancelled { .. })));
        cx.set_cancel_requested(false);
        assert!(provider.calls.lock().unwrap().is_empty());
    });
}

#[test]
fn dropping_a_split_batch_drops_its_pending_provider_without_detached_work() {
    run_test_with_cx(|cx| async move {
        let mut provider = Provider::new("fast", 2, 2);
        provider.fault = Fault::Pending;
        provider.fault_call = 2; // root refused, left succeeded, right pending
        let provider = Arc::new(provider);
        let wrapper = split(&provider);
        let texts = ["a", "b", "c", "d"];
        let mut future = wrapper.embed_batch_bound(&cx, &texts);
        {
            let mut task = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(std::future::Future::poll(future.as_mut(), &mut task).is_pending());
        }
        assert_eq!(provider.calls.lock().unwrap().len(), 3);
        drop(future);
        assert_eq!(provider.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(provider.calls.lock().unwrap().len(), 3);
        assert_eq!(
            wrapper.embed_batch_bound(&cx, &texts).await.unwrap().len(),
            4
        );
    });
}

#[test]
fn adaptive_native_build_is_opt_in_and_preserves_both_written_tiers() {
    run_test_with_cx(|cx| async move {
        for precision in [NativeBuildPrecision::F32, NativeBuildPrecision::F16] {
            let parent = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2, 2));
            let quality = Arc::new(Provider::new("quality", 3, 1));
            let ordinary =
                NativeIndexBuilder::new(parent.path().join("refused"), generation(1), fast.clone())
                    .unwrap()
                    .add_documents(documents())
                    .build(&cx)
                    .await;
            assert!(matches!(ordinary, Err(SearchError::EmbeddingFailed { .. })));
            assert_eq!(fast.calls.lock().unwrap().len(), 1);
            let adaptive = NativeIndexBuilder::new(
                parent.path().join("adaptive"),
                generation(1),
                fast.clone(),
            )
            .unwrap()
            .with_batch_failure_splitting()
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_fast_storage(precision, super::super::NativeBuildRetrieval::Exact)
            .with_quality_storage(precision, super::super::NativeBuildRetrieval::Exact)
            .unwrap()
            .add_documents(documents())
            .build(&cx)
            .await
            .unwrap();
            let baseline = NativeIndexBuilder::new(
                parent.path().join("baseline"),
                generation(1),
                Arc::new(Provider::new("fast", 2, 32)),
            )
            .unwrap()
            .with_quality_embedder(Arc::new(Provider::new("quality", 3, 32)))
            .unwrap()
            .with_fast_storage(precision, super::super::NativeBuildRetrieval::Exact)
            .with_quality_storage(precision, super::super::NativeBuildRetrieval::Exact)
            .unwrap()
            .add_documents(documents())
            .build(&cx)
            .await
            .unwrap();
            assert_eq!(
                adaptive
                    .documents()
                    .iter()
                    .map(|doc| (&doc.id, &doc.content))
                    .collect::<Vec<_>>(),
                baseline
                    .documents()
                    .iter()
                    .map(|doc| (&doc.id, &doc.content))
                    .collect::<Vec<_>>()
            );
            for (actual, expected) in [
                (adaptive.fast(), baseline.fast()),
                (adaptive.quality().unwrap(), baseline.quality().unwrap()),
            ] {
                assert_eq!(
                    actual.index().owner_witness(),
                    expected.index().owner_witness()
                );
                assert_eq!(
                    std::fs::read(actual.vector_path()).unwrap(),
                    std::fs::read(expected.vector_path()).unwrap()
                );
            }
            assert_eq!(fast.raw_calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.raw_calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn late_split_failure_writes_none_of_the_original_batch() {
    use super::super::TierPlan;
    use frankensearch_index::{ValidatedFsviBytes, VectorIndex};

    run_test_with_cx(|cx| async move {
        for fault in [
            Fault::Foreign,
            Fault::Short,
            Fault::NonFinite,
            Fault::WrongWidth,
            Fault::InvalidConfig,
            Fault::TypedCancellation,
            Fault::CancelSuccess,
            Fault::CancelFailure,
            Fault::DriftSuccess,
            Fault::DriftFailure,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("candidate.fsvi");
            let mut provider = Provider::new("fast", 2, 2);
            provider.fault = fault;
            provider.fault_call = 2; // root fails; left succeeds; right fails
            let provider = Arc::new(provider);
            let tier = TierPlan::new(split(&provider)).unwrap();
            let binding = tier.binding(&generation(1)).unwrap();
            let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
            writer.write_record("earlier-batch", &[0.25, 0.0]).unwrap();
            let documents = ["a", "b", "c", "d"].map(|id| IndexableDocument::new(id, id));
            let failure = tier.write_batch(&cx, &mut writer, &documents).await;
            cx.set_cancel_requested(false);
            assert!(failure.is_err());
            assert_eq!(provider.calls.lock().unwrap().len(), 3);
            // Finish this unselected writer solely to inspect what was staged.
            // No build handle or publication receipt is manufactured by the test.
            writer.finish().unwrap();
            let bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
            let owner = ValidatedFsviBytes::from_arc(bytes, &binding).unwrap();
            assert_eq!(owner.record_count(), 1);
            assert_eq!(owner.row(0).unwrap().doc_id(), "earlier-batch");
            assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
        }
    });
}

#[test]
fn successful_batches_are_not_split_and_empty_builds_do_not_infer() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        for empty in [false, true] {
            let fast = Arc::new(Provider::new("fast", 2, 2));
            let quality = Arc::new(Provider::new("quality", 3, 2));
            let input = if empty { Vec::new() } else { documents() };
            let built = NativeIndexBuilder::new(
                directory
                    .path()
                    .join(if empty { "empty" } else { "populated" }),
                generation(1),
                fast.clone(),
            )
            .unwrap()
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_batch_failure_splitting()
            .with_batch_failure_splitting() // repeated opt-in is idempotent
            .with_batch_size(2)
            .unwrap()
            .add_documents(input)
            .build(&cx)
            .await
            .unwrap();
            for provider in [&fast, &quality] {
                let calls = provider.calls.lock().unwrap().clone();
                assert_eq!(calls.len(), if empty { 0 } else { 4 });
                assert!(calls.iter().all(|batch| batch.len() <= 2));
                assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
            }
            assert_eq!(built.documents().len(), if empty { 0 } else { 7 });
            assert_eq!(built.fast().index().live_count(), built.documents().len());
            assert_eq!(
                built.quality().unwrap().index().live_count(),
                built.documents().len()
            );
        }
    });
}

#[test]
fn incremental_build_retains_splitting_and_reuses_only_unchanged_inputs() {
    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("fast", 2, 2));
        let quality = Arc::new(Provider::new("quality", 3, 1));
        let original = NativeIndexBuilder::new(
            directory.path().join("original"),
            generation(1),
            fast.clone(),
        )
        .unwrap()
        .with_quality_embedder(quality.clone())
        .unwrap()
        .with_batch_failure_splitting()
        .add_documents(documents())
        .build(&cx)
        .await
        .unwrap();
        let old_fast = std::fs::read(original.fast().vector_path()).unwrap();
        let old_quality = std::fs::read(original.quality().unwrap().vector_path()).unwrap();
        fast.calls.lock().unwrap().clear();
        quality.calls.lock().unwrap().clear();
        let updated = original
            .begin_update(&cx, directory.path().join("updated"), generation(2))
            .unwrap()
            .upsert_document(IndexableDocument::new("doc-0", "changed-0"))
            .upsert_document(IndexableDocument::new("doc-1", "changed-1"))
            .upsert_document(IndexableDocument::new("doc-2", "text-2").with_title("new title"))
            .delete_document("doc-6")
            .upsert_document(IndexableDocument::new("doc-8", "new-8"))
            .build(&cx)
            .await
            .unwrap();
        for (provider, expected_calls) in [(&fast, 3), (&quality, 5)] {
            let calls = provider.calls.lock().unwrap().clone();
            assert_eq!(calls.len(), expected_calls);
            assert_eq!(calls[0], ["changed-0", "changed-1", "new-8"]);
            assert!(
                calls
                    .iter()
                    .flatten()
                    .all(|text| { ["changed-0", "changed-1", "new-8"].contains(&text.as_str()) })
            );
            assert_eq!(provider.raw_calls.load(Ordering::SeqCst), 0);
        }
        assert_eq!(updated.documents().len(), 7);
        assert!(updated.document("doc-6").is_none());
        assert_eq!(updated.document("doc-8").unwrap().content, "new-8");
        assert_eq!(original.document("doc-0").unwrap().content, "text-0");
        assert!(original.document("doc-6").is_some());
        assert_eq!(
            std::fs::read(original.fast().vector_path()).unwrap(),
            old_fast
        );
        assert_eq!(
            std::fs::read(original.quality().unwrap().vector_path()).unwrap(),
            old_quality
        );
        let before_failure = std::fs::read(updated.fast().vector_path()).unwrap();
        let quality_before_failure =
            std::fs::read(updated.quality().unwrap().vector_path()).unwrap();
        let result = updated
            .begin_update(&cx, directory.path().join("failed"), generation(3))
            .unwrap()
            .upsert_document(IndexableDocument::new("doc-0", "poison"))
            .build(&cx)
            .await;
        assert!(matches!(result, Err(SearchError::EmbeddingFailed { .. })));
        assert_eq!(
            std::fs::read(updated.fast().vector_path()).unwrap(),
            before_failure
        );
        assert_eq!(
            std::fs::read(updated.quality().unwrap().vector_path()).unwrap(),
            quality_before_failure
        );
        assert_eq!(updated.document("doc-0").unwrap().content, "changed-0");
    });
}

#[cfg(all(feature = "quill", any(target_os = "linux", target_os = "macos")))]
#[test]
fn split_batches_build_seal_and_reopen_the_complete_hybrid_graph_cohort() {
    use super::super::{NativeBuildRetrieval, NativeBuiltHybridIndex};
    use frankensearch_index::native_hnsw::HnswParams;

    run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hybrid");
        let fast = Arc::new(Provider::new("fast", 2, 2));
        let quality = Arc::new(Provider::new("quality", 3, 1));
        let graph = NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 77,
        };
        let built = NativeIndexBuilder::new(&path, generation(1), fast.clone())
            .unwrap()
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_batch_failure_splitting()
            .with_fast_storage(NativeBuildPrecision::F16, graph)
            .with_quality_storage(NativeBuildPrecision::F32, graph)
            .unwrap()
            .add_documents(documents())
            .build_hybrid(&cx)
            .await
            .unwrap();
        let receipt = built.seal_for_reopen(&cx).unwrap();
        let expected = built.search_refined(&cx, "text-1", 3).await.unwrap();
        let expected: Vec<_> = expected
            .iter()
            .map(|hit| (hit.doc_id.to_string(), hit.score.to_bits()))
            .collect();
        assert_eq!(expected.len(), 3);
        let fast_calls = fast.calls.lock().unwrap().len();
        let quality_calls = quality.calls.lock().unwrap().len();
        let fast_witness = built.vectors().fast().index().owner_witness().clone();
        let quality_witness = built
            .vectors()
            .quality()
            .unwrap()
            .index()
            .owner_witness()
            .clone();
        drop(built);
        let reopened = NativeBuiltHybridIndex::open_selected(
            &cx,
            &path,
            &receipt,
            fast.clone(),
            Some(quality.clone()),
        )
        .await
        .unwrap();
        assert_eq!(fast.calls.lock().unwrap().len(), fast_calls);
        assert_eq!(quality.calls.lock().unwrap().len(), quality_calls);
        assert_eq!(
            reopened.vectors().fast().index().owner_witness(),
            &fast_witness
        );
        assert_eq!(
            reopened
                .vectors()
                .quality()
                .unwrap()
                .index()
                .owner_witness(),
            &quality_witness
        );
        assert!(reopened.vectors().fast().graph_path().is_some());
        assert!(reopened.vectors().quality().unwrap().graph_path().is_some());
        assert_eq!(reopened.lexical().doc_count().unwrap(), 7);
        let actual = reopened.search_refined(&cx, "text-1", 3).await.unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|hit| (hit.doc_id.to_string(), hit.score.to_bits()))
                .collect::<Vec<_>>(),
            expected
        );
        let fast_queries = fast.raw_calls.load(Ordering::SeqCst);
        assert_eq!(
            reopened
                .search_quality(&cx, "text-1", 3)
                .await
                .unwrap()
                .len(),
            3
        );
        assert_eq!(fast.raw_calls.load(Ordering::SeqCst), fast_queries);
    });
}
