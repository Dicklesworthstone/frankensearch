//! Native retrieval fenced to an exact, retained Quill scoring publication.
//!
//! This is a reader-composition boundary, not a publisher or a source-checkpoint
//! oracle. Expected receipts must come from the caller's trusted generation
//! selection. Matching document IDs alone does not attest common source text.

use std::sync::Arc;

use frankensearch_core::generation::{
    ExactComponentReceiptV1, GenerationComponentRole, SourceCheckpointV1,
};
use frankensearch_index::FsviV2Witness;
use frankensearch_index::exact_component_adapters::vector_component_receipt;
use frankensearch_quill::{QuillIndex, QuillIndexError, QuillSearchSnapshot};

use super::{NativeAnnIndex, NativeProgressiveSearch, checkpoint, invalid};
use crate::{
    Cx, Embedder, LexicalCandidateBatch, LexicalHydrationContext, LexicalRead, ScoredResult,
    SearchError, SearchFuture, SearchResult,
};

/// A native vector owner and an exact Quill publication admitted together.
///
/// Construction compares the complete selected FSVI witness and the selected
/// lexical byte receipt with the readers actually retained here. It also joins
/// their authenticated canonical live-document sets and source checkpoint.
/// Metadata is served by the original Quill batch context, not a new lookup.
///
/// New queries fail closed after the supplied Quill writer publishes another
/// snapshot. Queries whose lexical batch was already admitted may finish using
/// that original batch, including deferred hydration and lazy reranking. This
/// distinction permits in-flight completion without mixing new lexical state
/// into old vectors. No pathname is reopened and no writer lock is acquired.
///
/// Unsealed Delta epochs are deliberately refused: a Keeper-only receipt does
/// not authenticate a mixed Keeper/Delta view. Use a sealed, reopened writer
/// handle when the current writer still carries Delta epochs. This boundary
/// does not publish a durable composite manifest, discover current generations,
/// prove the supplied source checkpoint, or maintain a rollback floor.
#[derive(Clone)]
pub struct NativeQuillSnapshot {
    fast: Arc<NativeAnnIndex>,
    lexical: FencedQuillRead,
    vector_receipt: ExactComponentReceiptV1,
    quality: Option<(Arc<NativeAnnIndex>, ExactComponentReceiptV1)>,
}

impl NativeQuillSnapshot {
    /// Admit the exact selected readers before starting any provider work.
    ///
    /// `expected_fast` and `expected_lexical` are selection authority supplied
    /// by the caller, not receipts fabricated from the requested readers here.
    /// The checkpoint must describe that same caller-selected source cut.
    ///
    /// # Errors
    ///
    /// Rejects substituted bytes, role/checkpoint/document-set disagreement,
    /// unsealed lexical epochs, publication drift, and cancellation. Existing
    /// Quill reconciliation and artifact-read failures remain typed errors.
    pub fn admit(
        cx: &Cx,
        fast: Arc<NativeAnnIndex>,
        expected_fast: &FsviV2Witness,
        lexical: Arc<QuillIndex>,
        expected_lexical: &ExactComponentReceiptV1,
        source_checkpoint: SourceCheckpointV1,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.quill.admit")?;
        if fast.owner_witness() != expected_fast {
            return Err(invalid(
                "quill.fast",
                "witness-mismatch",
                "native owner differs from the selected complete FSVI witness",
            ));
        }
        expected_lexical
            .validate()
            .map_err(|error| invalid("quill.lexical", "invalid-receipt", &error.to_string()))?;
        if expected_lexical.role != GenerationComponentRole::Lexical {
            return Err(invalid(
                "quill.lexical",
                "wrong-role",
                "the selected lexical receipt must have the lexical role",
            ));
        }
        let snapshot = lexical.search_snapshot().map_err(SearchError::from)?;
        if snapshot.delta_count() != 0 {
            return Err(invalid(
                "quill.lexical",
                "unsealed-delta",
                "a Keeper receipt cannot authenticate unsealed Delta epochs",
            ));
        }
        let receipt = snapshot
            .keeper_snapshot()
            .exact_lexical_component_receipt(source_checkpoint)
            .map_err(QuillIndexError::from)
            .map_err(SearchError::from)?;
        if &receipt != expected_lexical {
            return Err(invalid(
                "quill.lexical",
                "receipt-mismatch",
                "the actual lexical publication differs from its selected exact receipt",
            ));
        }
        let vector_receipt = native_receipt(cx, &fast, source_checkpoint)?;
        join_docsets(&vector_receipt, &receipt)?;
        let lexical = FencedQuillRead {
            index: lexical,
            snapshot,
            receipt,
        };
        lexical.check_current()?;
        checkpoint(cx, "native_ann.quill.admitted")?;
        Ok(Self {
            fast,
            lexical,
            vector_receipt,
            quality: None,
        })
    }

    /// Compose a selected quality owner with this already-admitted source cut.
    ///
    /// The returned view shares the immutable fast/lexical owners. Failure leaves
    /// this view unchanged. The quality owner must cover the same complete live
    /// document set and bind the same full artifact generation as the fast tier.
    /// Its embedding space and dimension may differ; query-time admission checks
    /// each producer separately and the existing native cross-tier input policy.
    /// This method performs no inference, graph construction, or filesystem I/O.
    ///
    /// # Errors
    ///
    /// Rejects selected-witness substitution, generation/document-set/checkpoint
    /// disagreement, superseded lexical publication, and cancellation.
    pub fn with_quality(
        &self,
        cx: &Cx,
        quality: Arc<NativeAnnIndex>,
        expected_quality: &FsviV2Witness,
        source_checkpoint: SourceCheckpointV1,
    ) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.quill.quality_admit")?;
        self.lexical.check_current()?;
        if quality.owner_witness() != expected_quality {
            return Err(invalid(
                "quill.quality",
                "witness-mismatch",
                "quality owner differs from its selected complete FSVI witness",
            ));
        }
        if quality.owner_witness().generation != self.fast.owner_witness().generation {
            return Err(invalid(
                "quill.quality",
                "generation-mismatch",
                "fast and quality must bind the same full artifact generation",
            ));
        }
        let receipt = native_receipt(cx, &quality, source_checkpoint)?;
        join_docsets(&receipt, &self.lexical.receipt)?;
        self.lexical.check_current()?;
        checkpoint(cx, "native_ann.quill.quality_admitted")?;
        let mut view = self.clone();
        view.quality = Some((quality, receipt));
        Ok(view)
    }

    /// The independently admitted quality component, when explicitly attached.
    #[must_use]
    pub fn quality_receipt(&self) -> Option<&ExactComponentReceiptV1> {
        self.quality.as_ref().map(|(_, receipt)| receipt)
    }

    fn quality_index(&self) -> SearchResult<&NativeAnnIndex> {
        self.quality
            .as_ref()
            .map(|(index, _)| index.as_ref())
            .ok_or_else(|| {
                invalid(
                    "quill.quality",
                    "missing",
                    "quality retrieval requires an explicitly admitted quality owner",
                )
            })
    }

    /// Retrieve quality candidates directly, without running the fast embedder.
    ///
    /// # Errors
    ///
    /// Requires an admitted quality component and current lexical publication;
    /// propagates native identity, retrieval, hydration, and cancellation errors.
    pub async fn search_quality_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.quill.quality_search")?;
        self.lexical.check_current()?;
        self.quality_index()?
            .search_hybrid_quality_text(cx, embedder, &self.lexical, text, k)
            .await
    }

    /// Independently retrieve both admitted tiers and fuse their candidate union.
    ///
    /// Quality may promote a document absent from the fast candidate window.
    /// The existing native union/blend policy retains each model's raw score;
    /// lexical ranking and hydration use the fenced publication exclusively.
    ///
    /// # Errors
    ///
    /// Missing quality, mixed publication, identity/provider/retrieval/hydration
    /// failures, and cancellation are errors, not fast-only success.
    pub async fn search_refined_text(
        &self,
        cx: &Cx,
        fast_embedder: &dyn Embedder,
        quality_embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.quill.refined_search")?;
        self.lexical.check_current()?;
        self.fast
            .search_hybrid_refined_text(
                cx,
                fast_embedder,
                (self.quality_index()?, quality_embedder),
                &self.lexical,
                text,
                k,
            )
            .await
    }

    /// Prepare fast-then-quality retrieval over the admitted three-reader bundle.
    ///
    /// Only the requested second phase runs quality inference. Once the initial
    /// lexical batch has passed its fence, refinement and optional reranking may
    /// finish on that original pin even if the writer publishes another view.
    /// No phase reacquires a lexical snapshot or repeats the lexical query.
    /// `search_progressive` remains the explicitly fast-only convenience method.
    ///
    /// # Errors
    ///
    /// Rejects a missing quality owner, superseded publication, invalid query
    /// budget, per-tier identity mismatch, or cancellation before provider work.
    pub fn search_progressive_with_quality<'a>(
        &'a self,
        cx: &'a Cx,
        fast_embedder: &'a dyn Embedder,
        quality_embedder: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.quill.quality_progressive")?;
        self.lexical.check_current()?;
        self.fast.search_hybrid_progressive(
            cx,
            fast_embedder,
            Some((self.quality_index()?, quality_embedder)),
            &self.lexical,
            text,
            k,
        )
    }

    /// The authenticated vector component, derived from retained native rows.
    #[must_use]
    pub const fn vector_receipt(&self) -> &ExactComponentReceiptV1 {
        &self.vector_receipt
    }

    /// Exact lexical bytes and document set used by this composed reader.
    #[must_use]
    pub const fn lexical_receipt(&self) -> &ExactComponentReceiptV1 {
        &self.lexical.receipt
    }

    /// Immutable lexical owner, useful for version-stable stored-field readers.
    ///
    /// Exposing this immutable owner does not expose the writer or its publisher.
    #[must_use]
    pub fn lexical_snapshot(&self) -> &Arc<QuillSearchSnapshot> {
        &self.lexical.snapshot
    }

    /// Retrieve and hydrate against the admitted vector and lexical publication.
    ///
    /// # Errors
    ///
    /// Rejects a superseded lexical publication even for zero-k requests.
    /// Propagates ordinary native retrieval, provider and hydration failures.
    pub async fn search_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        checkpoint(cx, "native_ann.quill.search")?;
        self.lexical.check_current()?;
        self.fast
            .search_hybrid_text(cx, embedder, &self.lexical, text, k)
            .await
    }

    /// Prepare lazy native retrieval using the same publication fence.
    ///
    /// The returned search also supports `with_reranker`; source-text resolvers
    /// must retain their own version-stable source data as for the ordinary
    /// native progressive API. No work starts merely by constructing it.
    ///
    /// # Errors
    ///
    /// Rejects a superseded lexical publication, invalid query configuration,
    /// embedder admission failures and cancellation.
    pub fn search_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        embedder: &'a dyn Embedder,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.quill.progressive")?;
        self.lexical.check_current()?;
        self.fast
            .search_hybrid_progressive(cx, embedder, None, &self.lexical, text, k)
    }
}

fn native_receipt(
    cx: &Cx,
    native: &NativeAnnIndex,
    source_checkpoint: SourceCheckpointV1,
) -> SearchResult<ExactComponentReceiptV1> {
    let mut documents = Vec::new();
    for physical in 0..native.owner.record_count() {
        checkpoint(cx, "native_ann.quill.docset")?;
        let row = native.owner.row(physical)?;
        if row.flags().is_live() {
            documents.push(row.doc_id().to_owned());
        }
    }
    let receipt = vector_component_receipt(native.owner_witness(), documents, source_checkpoint)
        .map_err(|error| invalid("quill.vector", "invalid-receipt", &error.to_string()))?;
    checkpoint(cx, "native_ann.quill.vector_receipt")?;
    Ok(receipt)
}

fn join_docsets(
    vector: &ExactComponentReceiptV1,
    lexical: &ExactComponentReceiptV1,
) -> SearchResult<()> {
    if vector.docset_digest != lexical.docset_digest
        || vector.live_document_count != lexical.live_document_count
        || vector.source_checkpoint != lexical.source_checkpoint
    {
        return Err(invalid(
            "quill.docset",
            "component-mismatch",
            "native and lexical readers must authenticate the same canonical live-document set and source checkpoint",
        ));
    }
    Ok(())
}

/// The publisher on one QuillIndex advances monotonically and rejects epoch
/// reuse. Pointer equality on both sides of candidate scoring therefore proves
/// the batch was scored on this exact Arc, without decoding backend-private
/// hydration payloads (which differ in conformance-internals builds).
#[derive(Clone)]
struct FencedQuillRead {
    index: Arc<QuillIndex>,
    snapshot: Arc<QuillSearchSnapshot>,
    receipt: ExactComponentReceiptV1,
}

impl FencedQuillRead {
    fn check_current(&self) -> SearchResult<()> {
        let current = self.index.search_snapshot().map_err(SearchError::from)?;
        if !Arc::ptr_eq(&current, &self.snapshot) {
            return Err(invalid(
                "quill.publication",
                "changed",
                "Quill published another snapshot; select and admit a new native/lexical bundle",
            ));
        }
        Ok(())
    }
}

impl LexicalRead for FencedQuillRead {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let batch = self.search_candidates(cx, text, limit).await?;
            let mut results = batch.results().to_vec();
            self.hydrate_candidates(cx, batch.context(), &mut results)
                .await?;
            Ok(results)
        })
    }

    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            checkpoint(cx, "native_ann.quill.before_candidates")?;
            self.check_current()?;
            let response = self.index.search_candidates(cx, text, limit).await;
            checkpoint(cx, "native_ann.quill.after_candidates")?;
            let batch = response?;
            self.check_current()?;
            Ok(batch)
        })
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            // Deliberately no check_current: an already admitted query owns its
            // scoring pin and may finish after the live writer moves forward.
            checkpoint(cx, "native_ann.quill.before_hydration")?;
            let response = self.index.hydrate_candidates(cx, context, results).await;
            checkpoint(cx, "native_ann.quill.after_hydration")?;
            response
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        self.check_current()?;
        usize::try_from(self.snapshot.live_doc_count()).map_err(|_| {
            invalid(
                "quill.doc_count",
                "overflow",
                "lexical live count does not fit usize",
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::{Future, poll_fn};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use crate::IndexableDocument;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, CommitRange, EmbeddingIdentityBundleV1, QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory};
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};
    use frankensearch_quill::QuillConfig;

    fn source_cut() -> SourceCheckpointV1 {
        SourceCheckpointV1::derive(&CommitRange { low: 1, high: 7 })
    }

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        ready: AtomicBool,
        calls: AtomicUsize,
    }

    impl Provider {
        fn new() -> Self {
            Self {
                identity: EmbeddingIdentityBundleV1::explicit_test_model("quill-native", 2),
                ready: AtomicBool::new(true),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async { Ok(vec![1.0, 0.0]) })
        }
        fn embed_bound<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                poll_fn(|_| {
                    if self.ready.load(Ordering::SeqCst) {
                        Poll::Ready(())
                    } else {
                        Poll::Pending
                    }
                })
                .await;
                Ok(IdentityBoundEmbedding {
                    values: vec![1.0, 0.0],
                    identity: self.identity.clone(),
                })
            })
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn dimension(&self) -> usize {
            2
        }
        fn id(&self) -> &str {
            "quill-native"
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

    struct Fixture {
        _directory: tempfile::TempDir,
        lexical: Arc<QuillIndex>,
        fast: Arc<NativeAnnIndex>,
        receipt: ExactComponentReceiptV1,
    }

    fn config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            max_ingest_shards: 1,
            ..QuillConfig::default()
        }
    }

    async fn fixture(cx: &Cx, provider: &Provider, vector_ids: &[&str]) -> Fixture {
        let directory = tempfile::tempdir().unwrap();
        let lexical_path = directory.path().join("lexical");
        let writer = QuillIndex::create(cx, &lexical_path, config())
            .await
            .unwrap();
        writer
            .upsert_documents(
                cx,
                &[
                    IndexableDocument::new("alpha", "common alpha").with_metadata("version", "old"),
                    IndexableDocument::new("beta", "common beta").with_metadata("version", "old"),
                ],
            )
            .await
            .unwrap();
        writer.commit(cx).await.unwrap();
        drop(writer);
        // A fresh writer starts with just its sealed Keeper publication.
        let lexical = Arc::new(QuillIndex::open(cx, &lexical_path, config()).await.unwrap());
        let snapshot = lexical.search_snapshot().unwrap();
        assert_eq!(snapshot.delta_count(), 0);
        let receipt = snapshot
            .keeper_snapshot()
            .exact_lexical_component_receipt(source_cut())
            .unwrap();
        let fast = vector_index(cx, provider, directory.path(), vector_ids, 7);
        Fixture {
            _directory: directory,
            lexical,
            fast,
            receipt,
        }
    }

    fn vector_index(
        cx: &Cx,
        provider: &Provider,
        directory: &std::path::Path,
        ids: &[&str],
        generation: u64,
    ) -> Arc<NativeAnnIndex> {
        let path = directory.join(format!("vectors-{generation}.fsvi"));
        let mut identity = provider.identity.clone();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F32;
        identity.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(generation, [0x34; 16]).unwrap(),
            identity.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for id in ids {
            writer.write_record(id, &[1.0, 0.0]).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        Arc::new(NativeAnnIndex::exact(cx, owner).unwrap())
    }

    impl Fixture {
        fn admit(&self, cx: &Cx) -> SearchResult<NativeQuillSnapshot> {
            NativeQuillSnapshot::admit(
                cx,
                Arc::clone(&self.fast),
                self.fast.owner_witness(),
                Arc::clone(&self.lexical),
                &self.receipt,
                source_cut(),
            )
        }
    }

    #[test]
    fn real_quill_and_native_receipts_join_and_search_without_reopening() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            let view = fixture.admit(&cx).unwrap();
            assert_eq!(
                view.vector_receipt().docset_digest,
                view.lexical_receipt().docset_digest
            );
            let hits = view.search_text(&cx, &provider, "common", 2).await.unwrap();
            assert_eq!(hits.len(), 2);
            assert!(
                hits.iter()
                    .all(|hit| hit.lexical_score.is_some() && hit.metadata.is_some())
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let cloned = view.clone();
            assert!(Arc::ptr_eq(
                view.lexical_snapshot(),
                cloned.lexical_snapshot()
            ));
        });
    }

    #[test]
    fn same_count_different_document_sets_are_rejected() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "gamma"]).await;
            assert!(
                matches!(fixture.admit(&cx), Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.quill.docset")
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn complete_native_witness_not_only_generation_or_docset_is_checked() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            let mut expected = fixture.fast.owner_witness().clone();
            expected.whole_image_sha256[0] ^= 1;
            assert!(
                matches!(NativeQuillSnapshot::admit(&cx, Arc::clone(&fixture.fast), &expected, Arc::clone(&fixture.lexical), &fixture.receipt, source_cut()), Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.quill.fast")
            );
            assert!(fixture.admit(&cx).is_ok());
        });
    }

    #[test]
    fn lexical_byte_receipt_role_and_checkpoint_cannot_be_substituted() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            for case in 0..3 {
                let mut expected = fixture.receipt.clone();
                let mut cut = source_cut();
                match case {
                    0 => expected.bytes.sha256[0] ^= 1,
                    1 => expected.role = GenerationComponentRole::Vector,
                    _ => cut = SourceCheckpointV1::derive(&CommitRange { low: 1, high: 8 }),
                }
                assert!(
                    matches!(NativeQuillSnapshot::admit(&cx, Arc::clone(&fixture.fast), fixture.fast.owner_witness(), Arc::clone(&fixture.lexical), &expected, cut), Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.quill.lexical")
                );
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn unsealed_delta_is_not_authenticated_by_a_keeper_only_receipt() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            fixture
                .lexical
                .upsert_documents(&cx, &[IndexableDocument::new("new", "common new")])
                .await
                .unwrap();
            assert!(fixture.lexical.search_snapshot().unwrap().delta_count() > 0);
            assert!(
                matches!(fixture.admit(&cx), Err(SearchError::InvalidConfig { value, .. }) if value == "unsealed-delta")
            );
        });
    }

    #[test]
    fn new_queries_refuse_a_changed_publication_even_when_k_is_zero() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            let view = fixture.admit(&cx).unwrap();
            fixture
                .lexical
                .upsert_documents(&cx, &[IndexableDocument::new("alpha", "common changed")])
                .await
                .unwrap();
            for k in [0, 2] {
                assert!(
                    matches!(view.search_text(&cx, &provider, "common", k).await, Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.quill.publication")
                );
            }
            assert!(
                view.search_progressive(&cx, &provider, "common", 2)
                    .is_err()
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn admitted_batch_finishes_with_old_metadata_after_writer_publication() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            let view = fixture.admit(&cx).unwrap();
            let expected = fixture.lexical.search_results(&cx, "common", 2).unwrap();
            provider.ready.store(false, Ordering::SeqCst);
            let mut future = Box::pin(view.search_text(&cx, &provider, "common", 2));
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            fixture
                .lexical
                .upsert_documents(
                    &cx,
                    &[
                        IndexableDocument::new("alpha", "common changed alpha")
                            .with_metadata("version", "new"),
                        IndexableDocument::new("beta", "common changed beta")
                            .with_metadata("version", "new"),
                    ],
                )
                .await
                .unwrap();
            assert!(!Arc::ptr_eq(
                view.lexical_snapshot(),
                &fixture.lexical.search_snapshot().unwrap()
            ));
            provider.ready.store(true, Ordering::SeqCst);
            let hits = future.await.unwrap();
            for hit in &hits {
                let old = expected
                    .iter()
                    .find(|old| old.doc_id == hit.doc_id)
                    .unwrap();
                assert_eq!(hit.metadata.as_deref(), old.metadata.as_deref());
                assert!(hit.metadata.is_some());
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(view.search_text(&cx, &provider, "common", 2).await.is_err());
        });
    }

    #[test]
    fn cancellation_and_noops_do_not_start_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new();
            let fixture = fixture(&cx, &provider, &["alpha", "beta"]).await;
            let view = fixture.admit(&cx).unwrap();
            assert!(
                view.search_text(&cx, &provider, "common", 0)
                    .await
                    .unwrap()
                    .is_empty()
            );
            cx.cancel_with(
                asupersync::CancelKind::User,
                Some("quill composition cancelled"),
            );
            assert!(matches!(
                view.search_text(&cx, &provider, "common", 2).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                fixture.admit(&cx),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        });
    }

    fn quality_provider() -> Provider {
        Provider {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("quill-quality", 2),
            ready: AtomicBool::new(true),
            calls: AtomicUsize::new(0),
        }
    }

    fn native_rows(
        cx: &Cx,
        provider: &Provider,
        rows: &[(&str, [f32; 2])],
        generation: ArtifactGenerationIdentityV1,
        ann: bool,
    ) -> Arc<NativeAnnIndex> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier.fsvi");
        let mut identity = provider.identity.clone();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = QuantizationFormat::F32;
        identity.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(generation, identity.freeze().unwrap()).unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for (id, values) in rows {
            writer.write_record(id, values).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        Arc::new(if ann {
            NativeAnnIndex::build(
                cx,
                owner,
                frankensearch_index::native_hnsw::HnswParams::default(),
                7,
            )
            .unwrap()
        } else {
            NativeAnnIndex::exact(cx, owner).unwrap()
        })
    }

    fn attach_quality(cx: &Cx, fixture: &Fixture, provider: &Provider) -> NativeQuillSnapshot {
        let quality = native_rows(
            cx,
            provider,
            &[("alpha", [0.0, 1.0]), ("beta", [1.0, 0.0])],
            fixture.fast.owner_witness().generation,
            true,
        );
        fixture
            .admit(cx)
            .unwrap()
            .with_quality(
                cx,
                Arc::clone(&quality),
                quality.owner_witness(),
                source_cut(),
            )
            .unwrap()
    }

    #[test]
    fn composed_quality_retrieves_a_winner_outside_the_fast_candidate_window() {
        use super::super::NativeSearchPhase;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality_provider = quality_provider();
            let dir = tempfile::tempdir().unwrap();
            let lexical_path = dir.path().join("lexical");
            let writer = QuillIndex::create(&cx, &lexical_path, config())
                .await
                .unwrap();
            let docs: Vec<_> = ["a", "b", "c", "z"]
                .into_iter()
                .map(|id| IndexableDocument::new(id, "common"))
                .collect();
            writer.upsert_documents(&cx, &docs).await.unwrap();
            writer.commit(&cx).await.unwrap();
            drop(writer);
            let lexical = Arc::new(
                QuillIndex::open(&cx, &lexical_path, config())
                    .await
                    .unwrap(),
            );
            let receipt = lexical
                .search_snapshot()
                .unwrap()
                .keeper_snapshot()
                .exact_lexical_component_receipt(source_cut())
                .unwrap();
            let generation = ArtifactGenerationIdentityV1::new(7, [0x34; 16]).unwrap();
            let fast_index = native_rows(
                &cx,
                &fast,
                &[
                    ("a", [1.0, 0.0]),
                    ("b", [0.9, 0.1]),
                    ("c", [0.8, 0.2]),
                    ("z", [0.0, 1.0]),
                ],
                generation,
                false,
            );
            let quality = native_rows(
                &cx,
                &quality_provider,
                &[
                    ("a", [0.2, 0.8]),
                    ("b", [0.1, 0.9]),
                    ("c", [0.0, 1.0]),
                    ("z", [1.0, 0.0]),
                ],
                generation,
                true,
            );
            let view = NativeQuillSnapshot::admit(
                &cx,
                Arc::clone(&fast_index),
                fast_index.owner_witness(),
                lexical,
                &receipt,
                source_cut(),
            )
            .unwrap()
            .with_quality(
                &cx,
                Arc::clone(&quality),
                quality.owner_witness(),
                source_cut(),
            )
            .unwrap();
            assert_eq!(
                view.quality_receipt().unwrap().docset_digest,
                view.vector_receipt().docset_digest
            );
            let query =
                frankensearch_core::BoundQueryEmbedding::new(vec![1.0, 0.0], fast.identity.clone())
                    .unwrap();
            assert!(
                fast_index
                    .search(&cx, &query, 3, None)
                    .unwrap()
                    .iter()
                    .all(|hit| hit.doc_id != "z")
            );
            let mut stream = view
                .search_progressive_with_quality(&cx, &fast, &quality_provider, "unmatched", 1)
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial");
            };
            assert_eq!(results[0].doc_id, "a");
            assert_eq!(quality_provider.calls.load(Ordering::SeqCst), 0);
            let NativeSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined");
            };
            assert_eq!(results[0].doc_id, "z");
            assert_eq!(quality_provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn composed_quality_refinement_hydrates_old_metadata_after_publication() {
        use super::super::NativeSearchPhase;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality = quality_provider();
            let fixture = fixture(&cx, &fast, &["alpha", "beta"]).await;
            let view = attach_quality(&cx, &fixture, &quality);
            let expected = fixture.lexical.search_results(&cx, "common", 2).unwrap();
            let mut stream = view
                .search_progressive_with_quality(&cx, &fast, &quality, "common", 2)
                .unwrap();
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            fixture
                .lexical
                .upsert_documents(
                    &cx,
                    &[
                        IndexableDocument::new("alpha", "common changed")
                            .with_metadata("version", "new"),
                        IndexableDocument::new("beta", "common changed")
                            .with_metadata("version", "new"),
                    ],
                )
                .await
                .unwrap();
            let NativeSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined");
            };
            for hit in results {
                let old = expected
                    .iter()
                    .find(|old| old.doc_id == hit.doc_id)
                    .unwrap();
                assert_eq!(hit.metadata.as_deref(), old.metadata.as_deref());
                assert!(hit.metadata.is_some());
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert!(
                view.search_refined_text(&cx, &fast, &quality, "common", 2)
                    .await
                    .is_err()
            );
            assert!(
                view.search_quality_text(&cx, &quality, "common", 0)
                    .await
                    .is_err()
            );
        });
    }

    #[test]
    fn quality_admission_checks_full_witness_generation_docset_and_source_cut() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality = quality_provider();
            let fixture = fixture(&cx, &fast, &["alpha", "beta"]).await;
            let base = fixture.admit(&cx).unwrap();
            for case in 0..5 {
                let mut generation = fixture.fast.owner_witness().generation;
                let mut cut = source_cut();
                if case == 1 {
                    generation.sequence += 1;
                }
                if case == 2 {
                    generation.nonce[0] ^= 1;
                }
                if case == 4 {
                    cut = SourceCheckpointV1::derive(&CommitRange { low: 1, high: 8 });
                }
                let second = if case == 3 { "gamma" } else { "beta" };
                let candidate = native_rows(
                    &cx,
                    &quality,
                    &[("alpha", [0.0, 1.0]), (second, [1.0, 0.0])],
                    generation,
                    false,
                );
                let mut expected = candidate.owner_witness().clone();
                if case == 0 {
                    expected.whole_image_sha256[0] ^= 1;
                }
                assert!(
                    base.with_quality(&cx, candidate, &expected, cut).is_err(),
                    "case {case}"
                );
                assert!(base.quality_receipt().is_none());
            }
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(base.search_text(&cx, &fast, "common", 2).await.is_ok());
        });
    }

    #[test]
    fn missing_quality_and_foreign_query_producer_never_degrade_to_fast_success() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality = quality_provider();
            let fixture = fixture(&cx, &fast, &["alpha", "beta"]).await;
            let base = fixture.admit(&cx).unwrap();
            let view = attach_quality(&cx, &fixture, &quality);
            for k in [0, 2] {
                assert!(
                    base.search_refined_text(&cx, &fast, &quality, "common", k)
                        .await
                        .is_err()
                );
                assert!(
                    base.search_quality_text(&cx, &quality, "common", k)
                        .await
                        .is_err()
                );
                assert!(
                    base.search_progressive_with_quality(&cx, &fast, &quality, "common", k)
                        .is_err()
                );
                assert!(
                    view.search_refined_text(&cx, &fast, &fast, "common", k)
                        .await
                        .is_err()
                );
                assert!(
                    view.search_progressive_with_quality(&cx, &fast, &fast, "common", k)
                        .is_err()
                );
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn quality_primary_skips_fast_inference_and_refined_collect_matches_progressive() {
        use super::super::NativeSearchPhase;
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality = quality_provider();
            let fixture = fixture(&cx, &fast, &["alpha", "beta"]).await;
            let view = attach_quality(&cx, &fixture, &quality);
            let primary = view
                .search_quality_text(&cx, &quality, "unmatched", 1)
                .await
                .unwrap();
            assert_eq!(primary[0].doc_id, "beta");
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            let eager = view
                .search_refined_text(&cx, &fast, &quality, "common", 2)
                .await
                .unwrap();
            let mut stream = view
                .search_progressive_with_quality(&cx, &fast, &quality, "common", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let NativeSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("refined");
            };
            assert_eq!(
                serde_json::to_value(results).unwrap(),
                serde_json::to_value(eager).unwrap()
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 3);
        });
    }

    #[test]
    fn dropping_pending_composed_quality_releases_batch_and_does_not_retry() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality = quality_provider();
            let fixture = fixture(&cx, &fast, &["alpha", "beta"]).await;
            let view = attach_quality(&cx, &fixture, &quality);
            quality.ready.store(false, Ordering::SeqCst);
            let references = Arc::strong_count(view.lexical_snapshot());
            let mut stream = view
                .search_progressive_with_quality(&cx, &fast, &quality, "common", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(Arc::strong_count(view.lexical_snapshot()) > references);
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            drop(future);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(view.lexical_snapshot()), references);
        });
    }

    #[test]
    fn composed_quality_cancellation_is_terminal_and_starts_no_late_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new();
            let quality = quality_provider();
            let fixture = fixture(&cx, &fast, &["alpha", "beta"]).await;
            let view = attach_quality(&cx, &fixture, &quality);
            let mut stream = view
                .search_progressive_with_quality(&cx, &fast, &quality, "common", 2)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            cx.cancel_with(
                asupersync::CancelKind::User,
                Some("quality composition cancelled"),
            );
            assert!(matches!(
                stream.next_phase().await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(stream.is_finished());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        });
    }
}
