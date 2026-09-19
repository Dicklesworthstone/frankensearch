//! Source-scoped retrieval over one complete, immutable native hybrid cohort.
//!
//! Restrict candidates before vector normalization, lexical RRF, winner
//! hydration and reranking. Filtering a displayed page instead would both leak
//! disallowed candidates to later stages and underfill selective queries.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::NativeBuiltHybridIndex;
use crate::native_ann::{NativeProgressiveSearch, NativeSearchPhase, checkpoint, invalid};
use crate::{
    Cx, IndexableDocument, LexicalCandidateBatch, LexicalHydrationContext, LexicalRead,
    Reranker, ScoreSource, ScoredResult, SearchFuture, SearchResult,
};

type ScopedText<'a> = dyn Fn(&str) -> Option<String> + Send + Sync + 'a;

/// A reusable query scope frozen from the source documents of one retained build.
///
/// The predicate runs once per source document during construction, not during
/// query widening or later phases. The scope borrows the exact built/reopened
/// cohort; it cannot be transferred to a newer live generation by document ID.
/// Hold a live snapshot and call `snapshot.index().scope(...)` to scope that pin.
/// Scope membership, source text and row coordinates remain on that snapshot.
///
/// Both vector tiers and lexical candidates are filtered BEFORE fusion. ANN
/// widening uses the existing physical-row policy; lexical widening uses the
/// same private, unrefreshable Quill reader up to the complete corpus bound.
/// Only the final lexical batch's pin reaches hydration. Small ANN beams remain
/// approximate; this does not certify recall or change whole-corpus BM25 stats.
/// Very selective lexical queries may examine the full matching corpus.
///
/// This is query scoping, not revocation of an already-issued capability: a
/// previously retained scope deliberately keeps its original membership. It
/// does not hide the underlying corpus from code that already owns the index.
/// Keep the existing immutable-file/retention contract while readers live.
pub struct NativeScopedHybridIndex<'a> {
    index: &'a NativeBuiltHybridIndex,
    lexical: ScopedLexical<'a>,
    text: Box<ScopedText<'a>>,
}

impl NativeBuiltHybridIndex {
    /// Freeze scope membership from retained IDs, content, titles and metadata.
    ///
    /// A fallible predicate allows a policy/source lookup failure to abort scope
    /// construction rather than accidentally become an unrestricted query. The
    /// predicate is not retained and need not be Send. No provider runs or files
    /// are reopened. Reuse the returned scope for multiple queries on this cohort.
    ///
    /// # Errors
    /// Propagates predicate errors and cancellation, without returning a partial
    /// scope. Predicate panics are not intercepted.
    pub fn scope(
        &self,
        cx: &Cx,
        mut accept: impl FnMut(&IndexableDocument) -> SearchResult<bool>,
    ) -> SearchResult<NativeScopedHybridIndex<'_>> {
        checkpoint(cx, "native_ann.scope.start")?;
        let mut allowed = BTreeSet::new();
        for document in self.vectors().documents() {
            checkpoint(cx, "native_ann.scope.document")?;
            let accepted = accept(document);
            checkpoint(cx, "native_ann.scope.document_complete")?;
            if accepted? {
                allowed.insert(document.id.clone());
            }
        }
        let allowed = Arc::new(allowed);
        let text_ids = Arc::clone(&allowed);
        let text = Box::new(move |id: &str| {
            if text_ids.contains(id) {
                self.vectors()
                    .document(id)
                    .map(|document| document.content.clone())
            } else {
                None
            }
        });
        checkpoint(cx, "native_ann.scope.complete")?;
        Ok(NativeScopedHybridIndex {
            index: self,
            lexical: ScopedLexical {
                index: self,
                allowed,
            },
            text,
        })
    }
}

impl NativeScopedHybridIndex<'_> {
    /// Number of source documents eligible in this scope, not query matches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lexical.allowed.len()
    }

    /// Whether no source document is eligible.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lexical.allowed.is_empty()
    }

    /// Resolve only an eligible document from the original retained source cohort.
    #[must_use]
    pub fn document(&self, id: &str) -> Option<&IndexableDocument> {
        if self.lexical.allowed.contains(id) {
            self.index.vectors().document(id)
        } else {
            None
        }
    }

    /// Prepare scoped Initial and optional independent quality refinement.
    ///
    /// Uses the ordinary native phase engine, row semantics, hydration validation
    /// and cancellation behavior. Candidate counts describe eligible pools, not
    /// rejected rows examined while widening. Empty scopes skip inference but
    /// still admit every configured identity. No scope predicate is called here.
    ///
    /// # Errors
    /// Propagates native identity, topology, candidate-budget and cancellation errors.
    pub fn progressive<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let vectors = self.index.vectors();
        let fast = vectors.fast();
        fast.index()
            .search_hybrid_progressive(
                cx,
                fast.embedder(),
                vectors
                    .quality()
                    .map(|tier| (tier.index(), tier.embedder())),
                &self.lexical,
                text,
                k,
            )?
            .with_allowed_documents(&self.lexical.allowed)
    }

    /// Add lazy reranking of scoped candidates using their retained source text.
    ///
    /// Excluded documents never enter the rerank window, even when their global
    /// retrieval scores are higher. The same ordinary rerank response checks
    /// preserve scores and physical rows with their originating documents.
    ///
    /// # Errors
    /// Combines scoped progressive admission and native rerank configuration errors.
    pub fn progressive_with_reranker<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        reranker: &'a dyn Reranker,
        window: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        self.progressive(cx, text, k)?
            .with_reranker(reranker, self.text.as_ref(), window)
    }

    /// Scoped fast-plus-lexical search; never starts the quality provider.
    ///
    /// # Errors
    /// Propagates native retrieval, lexical expansion, hydration and cancellation errors.
    pub async fn search(&self, cx: &Cx, text: &str, k: usize) -> SearchResult<Vec<ScoredResult>> {
        collect(self.primary(cx, text, k, false)?).await
    }

    /// Collect the scoped progressive pipeline, requiring configured refinement.
    ///
    /// This runs Initial then quality refinement, retaining the original scoped
    /// fast and lexical pools. A configured quality failure is an error, not an
    /// unannounced Initial-only fallback. Use `progressive` for early delivery.
    ///
    /// # Errors
    /// Propagates every required phase's errors, including refinement failure.
    pub async fn search_refined(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        collect(self.progressive(cx, text, k)?).await
    }

    /// Scoped quality-primary retrieval, without running the fast model.
    /// Returned indices belong to the quality owner; hash controls stay nonsemantic.
    ///
    /// # Errors
    /// Refuses an absent quality tier and propagates retrieval/hydration errors.
    pub async fn search_quality(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        let mut results = collect(self.primary(cx, text, k, true)?).await?;
        for result in &mut results {
            checkpoint(cx, "native_ann.scope.quality_result")?;
            result.quality_score = result.fast_score.take();
            if result.source == ScoreSource::SemanticFast {
                result.source = ScoreSource::SemanticQuality;
            }
        }
        Ok(results)
    }

    fn primary<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        k: usize,
        quality: bool,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        checkpoint(cx, "native_ann.scope.primary")?;
        let vectors = self.index.vectors();
        let tier = if quality {
            vectors.quality().ok_or_else(|| {
                invalid(
                    "scope.quality",
                    "absent",
                    "quality-primary search requires a quality tier",
                )
            })?
        } else {
            vectors.fast()
        };
        tier.index()
            .search_hybrid_progressive(cx, tier.embedder(), None, &self.lexical, text, k)?
            .with_allowed_documents(&self.lexical.allowed)
    }
}

async fn collect(mut stream: NativeProgressiveSearch<'_>) -> SearchResult<Vec<ScoredResult>> {
    let mut page = Vec::new();
    while let Some(phase) = stream.next_phase().await? {
        page = match phase {
            NativeSearchPhase::Initial { results, .. }
            | NativeSearchPhase::Refined { results, .. }
            | NativeSearchPhase::Reranked { results, .. } => results,
            NativeSearchPhase::RefinementFailed { error, .. }
            | NativeSearchPhase::RerankFailed { error, .. } => return Err(error),
        };
    }
    Ok(page)
}

// This adapter can ONLY wrap the private unrefreshable reader belonging to a
// complete built/reopened cohort. Do not generalize it to arbitrary LexicalRead:
// repeated search_candidates calls on a mutable reader could mix generations.
struct ScopedLexical<'a> {
    index: &'a NativeBuiltHybridIndex,
    allowed: Arc<BTreeSet<String>>,
}

impl LexicalRead for ScopedLexical<'_> {
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move {
            let batch = self.search_candidates(cx, text, limit).await?;
            let (mut results, context) = batch.into_parts();
            if context.is_some() {
                self.hydrate_candidates(cx, context.as_ref(), &mut results)
                    .await?;
            }
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
            checkpoint(cx, "native_ann.scope.lexical_start")?;
            let target = limit.min(self.allowed.len());
            if target == 0 {
                return Ok(LexicalCandidateBatch::eager(Vec::new()));
            }
            let ceiling = self.index.vectors().documents().len();
            let mut width = limit.min(ceiling);
            loop {
                checkpoint(cx, "native_ann.scope.lexical_window")?;
                let response = self
                    .index
                    .lexical()
                    .search_candidates(cx, text, width)
                    .await;
                checkpoint(cx, "native_ann.scope.lexical_response")?;
                let batch = response?;
                let count = batch.results().len();
                if count > width {
                    return Err(invalid(
                        "scope.lexical",
                        "overfull",
                        "lexical response exceeds its window",
                    ));
                }
                let mut seen = BTreeSet::new();
                for result in batch.results() {
                    checkpoint(cx, "native_ann.scope.lexical_candidate")?;
                    if !result.score.is_finite()
                        || self
                            .index
                            .vectors()
                            .document(result.doc_id.as_str())
                            .is_none()
                        || !seen.insert(result.doc_id.as_str())
                    {
                        return Err(invalid(
                            "scope.lexical",
                            "invalid-candidate",
                            "lexical candidates must be finite, unique members of the retained source cohort",
                        ));
                    }
                }
                drop(seen);
                let (results, context) = batch.into_parts();
                let mut eligible = Vec::new();
                for result in results {
                    checkpoint(cx, "native_ann.scope.lexical_filter")?;
                    if self.allowed.contains(result.doc_id.as_str()) && eligible.len() < target {
                        eligible.push(result);
                    }
                }
                if eligible.len() == target || count < width || width == ceiling {
                    checkpoint(cx, "native_ann.scope.lexical_complete")?;
                    return Ok(match context {
                        Some(pin) => LexicalCandidateBatch::deferred(eligible, pin),
                        None => LexicalCandidateBatch::eager(eligible),
                    });
                }
                // Discard the entire earlier window and its pin. The next query
                // still uses the SAME immutable Quill reader. Never append ranks
                // from overlapping windows or transfer a pin to another reader.
                width = width.saturating_mul(2).max(width + 1).min(ceiling);
            }
        })
    }

    fn hydrate_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            checkpoint(cx, "native_ann.scope.hydration")?;
            for result in results
                .iter()
                .filter(|result| result.lexical_score.is_some())
            {
                checkpoint(cx, "native_ann.scope.hydration_member")?;
                if !self.allowed.contains(result.doc_id.as_str()) {
                    return Err(invalid(
                        "scope.hydration",
                        "excluded",
                        "winner is outside its frozen scope",
                    ));
                }
            }
            let response = self
                .index
                .lexical()
                .hydrate_candidates(cx, context, results)
                .await;
            checkpoint(cx, "native_ann.scope.hydration_complete")?;
            response
        })
    }

    fn doc_count(&self) -> SearchResult<usize> {
        Ok(self.allowed.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::task::{Context, Poll, Waker};

    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        EmbeddingSpaceKindV1,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, RerankDocument, RerankScore};
    use frankensearch_index::native_hnsw::HnswParams;

    use crate::native_ann::builder::live::NativeLiveHybridIndex;
    use crate::native_ann::builder::{NativeBuildPrecision, NativeBuildRetrieval, NativeIndexBuilder};
    use crate::{Embedder, ModelCategory, SearchError};

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        foreign: EmbeddingIdentityBundleV1,
        advertise_foreign: AtomicBool,
        queries: AtomicUsize,
        drops: AtomicUsize,
        hold: AtomicBool,
        fail: AtomicBool,
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Self {
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
            identity.space.kind = EmbeddingSpaceKindV1::Semantic;
            identity.space.hash_control = None;
            identity.space.artifact_manifest_fingerprint = "a".repeat(64);
            identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(), sha256: "b".repeat(64), size: 1,
            }];
            identity.producer.space_fingerprint = identity.space.fingerprint();
            identity.validate().unwrap();
            let mut foreign = identity.clone();
            foreign.producer.backend = "foreign".to_owned();
            Self {
                identity, foreign,
                advertise_foreign: AtomicBool::new(false),
                queries: AtomicUsize::new(0), drops: AtomicUsize::new(0),
                hold: AtomicBool::new(false), fail: AtomicBool::new(false),
            }
        }

        fn values(&self, text: &str) -> Vec<f32> {
            let quality = self.dimension() == 3;
            let score = if text.contains("private") { 2.0 }
                else if text.contains("fast") { if quality { 0.0 } else { 1.0 } }
                else if text.contains("middle") { if quality { 0.5 } else { 0.75 } }
                else if text.contains("near") { if quality { 0.75 } else { 0.5 } }
                else if text.contains("quality") { if quality { 1.0 } else { 0.0 } }
                else { 1.0 };
            let mut values = vec![0.0; self.dimension()];
            values[0] = score;
            // Stored rows may have zero query score but never zero vector norm.
            values[1] = if score <= 0.0 { 1.0 } else { 0.0 };
            values
        }
    }

    struct DropCount<'a>(&'a AtomicUsize);
    impl Drop for DropCount<'_> {
        fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move { Ok(self.values(text)) })
        }
        fn embed_batch_bound<'a>(
            &'a self, _cx: &'a Cx, texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                Ok(texts.iter().map(|text| IdentityBoundEmbedding {
                    values: self.values(text), identity: self.identity.clone(),
                }).collect())
            })
        }
        fn embed_bound<'a>(
            &'a self, _cx: &'a Cx, _text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.queries.fetch_add(1, Ordering::SeqCst);
                let _drop = DropCount(&self.drops);
                if self.hold.load(Ordering::SeqCst) { return std::future::pending().await; }
                if self.fail.load(Ordering::SeqCst) {
                    return Err(invalid("scope.test_provider", "failed", "quality failed"));
                }
                let mut values = vec![0.0; self.dimension()];
                values[0] = 1.0;
                Ok(IdentityBoundEmbedding { values, identity: self.identity.clone() })
            })
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.advertise_foreign.load(Ordering::SeqCst) { &self.foreign } else { &self.identity })
        }
        fn id(&self) -> &str { &self.identity.space.logical_model_id }
        fn model_name(&self) -> &str { self.id() }
        fn dimension(&self) -> usize { usize::try_from(self.identity.space.dimension).unwrap() }
        fn is_ready(&self) -> bool { true }
        fn is_semantic(&self) -> bool { true }
        fn category(&self) -> ModelCategory { ModelCategory::TransformerEmbedder }
    }

    fn generation(sequence: u64) -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(sequence, [0x65; 16]).unwrap()
    }

    fn documents() -> Vec<IndexableDocument> {
        let mut documents: Vec<_> = (0..24).map(|i| {
            IndexableDocument::new(format!("private-{i:02}"), "needle private")
                .with_metadata("scope", "private")
        }).collect();
        for (id, kind) in [("a-fast", "fast"), ("b-middle", "middle"), ("c-near", "near"), ("d-quality", "quality")] {
            documents.push(IndexableDocument::new(id, format!("needle {kind} {}", "padding ".repeat(24)))
                .with_title(format!("retained {id}"))
                .with_metadata("scope", "public"));
        }
        documents
    }

    fn public(document: &IndexableDocument) -> bool {
        document.metadata.get("scope").is_some_and(|scope| scope == "public")
    }

    async fn build(
        cx: &Cx, path: &std::path::Path, ann: bool, fast: &Arc<Provider>, quality: Option<&Arc<Provider>>,
    ) -> NativeBuiltHybridIndex {
        let mut builder = NativeIndexBuilder::new(path, generation(1), fast.clone()).unwrap()
            .add_documents(documents())
            .with_fast_storage(NativeBuildPrecision::F16, if ann {
                NativeBuildRetrieval::Hnsw { params: HnswParams { ef_search: 1, ..HnswParams::default() }, seed: 7 }
            } else { NativeBuildRetrieval::Exact });
        if let Some(quality) = quality {
            builder = builder.with_quality_embedder(quality.clone()).unwrap()
                .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact).unwrap();
        }
        builder.build_hybrid(cx).await.unwrap()
    }

    fn assert_row(index: &NativeBuiltHybridIndex, result: &ScoredResult, quality_primary: bool) {
        if let Some(row) = result.index {
            let tier = if quality_primary || result.fast_score.is_none() {
                index.vectors().quality().unwrap()
            } else { index.vectors().fast() };
            assert_eq!(
                tier.index()
                    .owner
                    .doc_id_at(usize::try_from(row).unwrap())
                    .unwrap(),
                result.doc_id.as_str()
            );
        }
    }

    #[test]
    fn selective_scopes_refill_lexical_and_vector_pools_before_fusion() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for ann in [false, true] {
                let directory = tempfile::tempdir().unwrap();
                let fast = Arc::new(Provider::new("fast", 2));
                let quality = Arc::new(Provider::new("quality", 3));
                let index = build(&cx, &directory.path().join("index"), ann, &fast, Some(&quality)).await;
                let scope = index.scope(&cx, |document| Ok(public(document))).unwrap();
                assert_eq!(scope.len(), 4);
                assert!(!scope.is_empty());
                assert!(scope.document("private-00").is_none());
                let full = index.lexical().search_candidates(&cx, "needle", 28).await.unwrap();
                assert_eq!(full.results().len(), 28);
                assert!(full.results()[..3].iter().all(|r| r.doc_id.starts_with("private-")));
                let expected: Vec<_> = full.results().iter().filter(|r| scope.document(r.doc_id.as_str()).is_some())
                    .take(3).map(|r| (r.doc_id.clone(), r.score.to_bits())).collect();
                let filtered = scope.lexical.search_candidates(&cx, "needle", 3).await.unwrap();
                assert_eq!(filtered.results().iter().map(|r| (r.doc_id.clone(), r.score.to_bits())).collect::<Vec<_>>(), expected);
                assert!(filtered.is_deferred());
                let unfiltered = index.vectors().fast().search(&cx, "needle", 3).await.unwrap();
                assert!(unfiltered.iter().all(|r| r.doc_id.starts_with("private-")));
                let hits = scope.search(&cx, "needle", 3).await.unwrap();
                assert_eq!(hits.len(), 3, "post-filtering the global window would return nothing");
                for hit in &hits {
                    assert!(scope.document(hit.doc_id.as_str()).is_some());
                    assert!(hit.lexical_score.is_some() && hit.metadata.is_some());
                    assert_row(&index, hit, false);
                }
                let all = scope.lexical.search_candidates(&cx, "needle", 100).await.unwrap();
                assert_eq!(all.results().len(), 4);
                assert!(scope.lexical.search_candidates(&cx, "absentterm", 3).await.unwrap().results().is_empty());
            }
        });
    }

    #[test]
    fn quality_retrieves_a_scoped_winner_outside_the_scoped_fast_pool() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &directory.path().join("index"), true, &fast, Some(&quality)).await;
            let scope = index.scope(&cx, |document| Ok(public(document))).unwrap();
            let mut stream = scope.progressive(&cx, "absentterm", 1).unwrap();
            let NativeSearchPhase::Initial { results, candidates } = stream.next_phase().await.unwrap().unwrap() else { panic!("initial"); };
            assert_eq!(results[0].doc_id, "a-fast");
            assert_eq!(candidates.fast, 3);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            let NativeSearchPhase::Refined { results, candidates } = stream.next_phase().await.unwrap().unwrap() else { panic!("refined"); };
            assert_eq!(results[0].doc_id, "d-quality");
            assert_eq!(results[0].quality_score, Some(1.0));
            assert!(results[0].fast_score.is_none());
            assert_eq!(candidates.quality, 3);
            assert_eq!(candidates.lexical, 0);
            assert_row(&index, &results[0], false);
            assert!(stream.is_finished());
            let fast_calls = fast.queries.load(Ordering::SeqCst);
            let quality_only = scope.search_quality(&cx, "absentterm", 1).await.unwrap();
            assert_eq!(quality_only[0].doc_id, "d-quality");
            assert!(quality_only[0].fast_score.is_none());
            assert_eq!(quality_only[0].quality_score, Some(1.0));
            assert_row(&index, &quality_only[0], true);
            assert_eq!(fast.queries.load(Ordering::SeqCst), fast_calls);
        });
    }

    #[test]
    fn all_document_scope_matches_existing_fast_quality_and_refined_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &directory.path().join("index"), false, &fast, Some(&quality)).await;
            let scope = index.scope(&cx, |_| Ok(true)).unwrap();
            for k in [0, 1, 5, 40] {
                for (scoped, original) in [
                    (scope.search(&cx, "needle", k).await, index.search(&cx, "needle", k).await),
                    (scope.search_refined(&cx, "needle", k).await, index.search_refined(&cx, "needle", k).await),
                    (scope.search_quality(&cx, "needle", k).await, index.search_quality(&cx, "needle", k).await),
                ] {
                    assert_eq!(serde_json::to_value(scoped.unwrap()).unwrap(), serde_json::to_value(original.unwrap()).unwrap());
                }
            }
        });
    }

    struct RerankProbe { inputs: Mutex<Vec<RerankDocument>> }
    impl Reranker for RerankProbe {
        fn rerank<'a>(&'a self, _cx: &'a Cx, _query: &'a str, docs: &'a [RerankDocument]) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                *self.inputs.lock().unwrap() = docs.to_vec();
                Ok(docs.iter().enumerate().map(|(original_rank, doc)| RerankScore {
                    doc_id: doc.doc_id.clone(), original_rank, raw_logit: None,
                    score: if doc.doc_id == "d-quality" { 1.0 } else { 0.0 },
                }).collect())
            })
        }
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str { "scoped-reranker" }
        fn model_name(&self) -> &str { self.id() }
    }

    #[test]
    fn reranking_only_receives_eligible_retained_source_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let index = build(&cx, &directory.path().join("index"), false, &fast, None).await;
            let scope = index.scope(&cx, |document| Ok(public(document))).unwrap();
            let probe = RerankProbe { inputs: Mutex::new(Vec::new()) };
            let mut stream = scope.progressive_with_reranker(&cx, "needle", 1, &probe, 4).unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(probe.inputs.lock().unwrap().is_empty());
            let NativeSearchPhase::Reranked { results, evaluated, .. } = stream.next_phase().await.unwrap().unwrap() else { panic!("reranked"); };
            assert_eq!(evaluated, 4);
            assert_eq!(results[0].doc_id, "d-quality");
            assert_eq!(results[0].fast_score, Some(0.0));
            assert_row(&index, &results[0], false);
            let inputs = probe.inputs.lock().unwrap();
            assert_eq!(inputs.len(), 4);
            for input in inputs.iter() {
                assert_eq!(input.text, scope.document(&input.doc_id).unwrap().content);
                assert!(!input.doc_id.starts_with("private-"));
            }
        });
    }

    #[test]
    fn empty_scopes_skip_inference_without_bypassing_identity_or_topology_admission() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &directory.path().join("index"), false, &fast, Some(&quality)).await;
            let empty = index.scope(&cx, |_| Ok(false)).unwrap();
            assert!(empty.is_empty());
            assert!(empty.search_refined(&cx, "needle", 5).await.unwrap().is_empty());
            assert!(empty.search_quality(&cx, "needle", 5).await.unwrap().is_empty());
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            quality.advertise_foreign.store(true, Ordering::SeqCst);
            assert!(empty.progressive(&cx, "needle", 0).is_err());
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            let fast_only = build(&cx, &directory.path().join("fast-only"), false, &fast, None).await;
            assert!(fast_only.scope(&cx, |_| Ok(false)).unwrap().search_quality(&cx, "needle", 0).await.is_err());
        });
    }

    #[test]
    fn scope_predicate_failures_and_cancellation_never_return_a_partial_scope() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let index = build(&cx, &directory.path().join("index"), false, &fast, None).await;
            let mut calls = 0;
            let result = index.scope(&cx, |_| {
                calls += 1;
                if calls == 3 { Err(invalid("scope.test_policy", "failed", "lookup failed")) } else { Ok(true) }
            });
            assert!(result.is_err());
            assert_eq!(calls, 3);
            let result = index.scope(&cx, |_| { cx.set_cancel_requested(true); Ok(true) });
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn failed_and_abandoned_refinement_never_escape_the_original_scope() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &directory.path().join("index"), false, &fast, Some(&quality)).await;
            let scope = index.scope(&cx, |document| Ok(public(document))).unwrap();
            quality.fail.store(true, Ordering::SeqCst);
            assert!(scope.search_refined(&cx, "needle", 1).await.is_err());
            let mut stream = scope.progressive(&cx, "needle", 1).unwrap();
            let NativeSearchPhase::Initial { results, .. } = stream.next_phase().await.unwrap().unwrap() else { panic!("initial"); };
            let before = serde_json::to_value(&results).unwrap();
            let NativeSearchPhase::RefinementFailed { initial_results, .. } = stream.next_phase().await.unwrap().unwrap() else { panic!("failure"); };
            assert_eq!(serde_json::to_value(initial_results).unwrap(), before);
            quality.fail.store(false, Ordering::SeqCst);
            quality.hold.store(true, Ordering::SeqCst);
            let mut stream = scope.progressive(&cx, "needle", 1).unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let drops = quality.drops.load(Ordering::SeqCst);
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
            drop(future);
            assert_eq!(quality.drops.load(Ordering::SeqCst), drops + 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn live_replacement_does_not_rebind_an_existing_scope_or_reevaluate_its_policy() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2));
            let quality = Arc::new(Provider::new("quality", 3));
            let index = build(&cx, &directory.path().join("old"), false, &fast, Some(&quality)).await;
            let live = NativeLiveHybridIndex::new(&cx, index).unwrap();
            let pin = live.snapshot(&cx).await.unwrap();
            let calls = AtomicUsize::new(0);
            let scope = pin.index().scope(&cx, |document| {
                calls.fetch_add(1, Ordering::SeqCst); Ok(public(document))
            }).unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 28);
            let before = scope.search_refined(&cx, "needle", 4).await.unwrap();
            let candidate = pin.begin_update(&cx, directory.path().join("next"), generation(2)).unwrap()
                .upsert_document(IndexableDocument::new("d-quality", "needle private").with_metadata("scope", "private"))
                .upsert_document(IndexableDocument::new("fresh", "needle fast").with_metadata("scope", "public"))
                .build(&cx).await.unwrap();
            let next = live.install(&cx, &candidate).await.unwrap();
            let next_scope = next.index().scope(&cx, |document| Ok(public(document))).unwrap();
            assert!(scope.document("d-quality").is_some());
            assert!(scope.document("fresh").is_none());
            assert!(next_scope.document("d-quality").is_none());
            assert!(next_scope.document("fresh").is_some());
            let after = scope.search_refined(&cx, "needle", 4).await.unwrap();
            assert_eq!(serde_json::to_value(before).unwrap(), serde_json::to_value(&after).unwrap());
            assert_eq!(calls.load(Ordering::SeqCst), 28);
            for hit in &after { assert_row(pin.index(), hit, false); }
            let current = next_scope.search_refined(&cx, "needle", 4).await.unwrap();
            assert_eq!(current.len(), 4);
            assert!(current.iter().all(|hit| next_scope.document(hit.doc_id.as_str()).is_some()));
        });
    }
}
