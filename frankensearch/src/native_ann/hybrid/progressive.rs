//! Lazy native refinement over the same retained fast and lexical candidates.

use std::collections::BTreeSet;

use frankensearch_core::LexicalCandidateBatch;
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::traits::Reranker;
use frankensearch_fusion::{RrfConfig, rrf_fuse_for_vector_lane};

use super::refinement::{RefinementFusion, admit_pair, checked_lexical, refined_winners};
use super::{NativeAnnIndex, checkpoint, hydrate_winners, invalid, join_sources};
use crate::{Cx, Embedder, LexicalRead, ScoredResult, SearchError, SearchResult, VectorHit};

pub(super) mod rerank;
use rerank::RerankRequest;

impl NativeAnnIndex {
    /// Prepare native progressive search without starting inference or retrieval.
    ///
    /// The first `next_phase` retrieves this fast owner plus lexical candidates.
    /// Only a second call starts independent retrieval from the optional quality
    /// owner. Stopping after initial results starts no quality inference. With
    /// no quality arm, the stream yields one initial phase unless an optional
    /// [`NativeProgressiveSearch::with_reranker`] stage is attached. Direct
    /// quality-only retrieval remains [`Self::search_hybrid_quality_text`].
    ///
    /// Generation, producer, space-kind and document-ID joins use the same
    /// admission boundary as [`Self::search_hybrid_refined_text`]. Refinement
    /// uses its exact blend, RRF, raw-score and owner-specific row-index policy,
    /// but reuses the original fast candidate pool and pinned lexical snapshot.
    /// This does not prove a composite lexical/vector generation or change the
    /// default searcher. Graph work remains synchronous on the polling executor.
    ///
    /// # Errors
    ///
    /// Refuses invalid admission or overflow of the three-times candidate
    /// budget. Every configured identity is checked even for zero-k requests.
    pub fn search_hybrid_progressive<'a>(
        &'a self,
        cx: &'a Cx,
        fast_embedder: &'a dyn Embedder,
        quality: Option<(&'a Self, &'a dyn Embedder)>,
        lexical: &'a dyn LexicalRead,
        text: &'a str,
        k: usize,
    ) -> SearchResult<NativeProgressiveSearch<'a>> {
        let search = NativeProgressiveSearch {
            fast: self,
            cx,
            fast_embedder,
            quality,
            lexical,
            text,
            k,
            reranker: None,
            allowed_documents: None,
            candidate_multiplier: 3,
            rrf: RrfConfig::default(),
            quality_weight: 0.7,
            fast_ef: None,
            quality_ef: None,
            budget: k.checked_mul(3).ok_or_else(|| {
                invalid(
                    "progressive.candidate_budget",
                    "overflow",
                    "three times the requested result count must fit usize",
                )
            })?,
            state: State::Initial,
        };
        search.admit()?;
        Ok(search)
    }
}

/// Actual returned candidate counts, not graph visits or corpus-coverage claims.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativePhaseCandidates {
    /// Candidates retrieved from the retained fast owner.
    pub fast: usize,
    /// Candidates independently retrieved from the retained quality owner.
    pub quality: usize,
    /// Candidates in the original pinned lexical scoring batch.
    pub lexical: usize,
}

/// A requested phase of caller-owned native progressive retrieval.
#[derive(Debug)]
pub enum NativeSearchPhase {
    /// Fast-plus-lexical results ready without any quality inference.
    Initial {
        /// Hydrated winners in fused rank order.
        results: Vec<ScoredResult>,
        /// Observed candidate counts; quality is zero in this phase.
        candidates: NativePhaseCandidates,
    },
    /// Winners after independent quality retrieval and candidate union.
    Refined {
        /// Hydrated winners; these may include documents outside the fast pool.
        results: Vec<ScoredResult>,
        /// Counts from retained fast/lexical pools and the new quality pool.
        candidates: NativePhaseCandidates,
    },
    /// A non-cancellation refinement failure; initial results remain valid.
    RefinementFailed {
        /// Original initial results, unchanged by the attempted refinement.
        initial_results: Vec<ScoredResult>,
        /// Failure in admission, quality retrieval, fusion, or hydration.
        error: SearchError,
    },
    /// Final cross-encoder ordering of the retained retrieval candidate window.
    Reranked {
        /// At most the requested `k` winners, with original retrieval evidence.
        results: Vec<ScoredResult>,
        /// Counts from the retrieval phase, not cross-encoder model calls.
        candidates: NativePhaseCandidates,
        /// Number of candidate documents that had text and were scored.
        evaluated: usize,
    },
    /// Non-cancellation rerank failure; the last displayed results remain valid.
    RerankFailed {
        /// The previous displayed page, unchanged by the failed rerank.
        previous_results: Vec<ScoredResult>,
        /// Model availability, response admission, or inference failure.
        error: SearchError,
    },
}

struct PendingRefinement {
    fast: Vec<VectorHit>,
    batch: LexicalCandidateBatch,
    initial_results: Vec<ScoredResult>,
}

struct PendingRerank {
    results: Vec<ScoredResult>,
    candidates: NativePhaseCandidates,
    // Retain the scoring snapshot through text resolution and model inference.
    // The caller's text resolver must itself select the same document versions.
    _batch: LexicalCandidateBatch,
}

enum State {
    Initial,
    Refine(PendingRefinement),
    Rerank(PendingRerank),
    Done,
}

/// Lazy search retaining the exact fast pool and lexical scoring snapshot.
///
/// Dropping this stream after initial results starts no quality work. Dropping
/// a *polled, still-pending* `next_phase` future permanently finishes the stream
/// and releases its in-flight work and snapshot pin. Subsequent calls return
/// `None` rather than retrying inference, traversal, or lexical retrieval.
/// Dropping an unpolled future starts no phase and leaves the stream unchanged.
///
/// There are no detached tasks or hidden runtimes. Cancellation checkpoints
/// surround synchronous graph work; no preemptive interruption or implicit
/// quality timeout is claimed. A provider cancellation propagates as `Err`,
/// never as a successful refinement or a `RefinementFailed` degradation.
///
/// Construct with [`NativeAnnIndex::search_hybrid_progressive`].
pub struct NativeProgressiveSearch<'a> {
    fast: &'a NativeAnnIndex,
    cx: &'a Cx,
    fast_embedder: &'a dyn Embedder,
    quality: Option<(&'a NativeAnnIndex, &'a dyn Embedder)>,
    lexical: &'a dyn LexicalRead,
    text: &'a str,
    k: usize,
    budget: usize,
    candidate_multiplier: usize,
    rrf: RrfConfig,
    quality_weight: f32,
    fast_ef: Option<usize>,
    quality_ef: Option<usize>,
    reranker: Option<RerankRequest<'a>>,
    // Installed only by the built-cohort scope adapter, together with its
    // filtered lexical reader. None preserves the ordinary unscoped path.
    allowed_documents: Option<&'a BTreeSet<String>>,
    state: State,
}

impl<'a> NativeProgressiveSearch<'a> {
    /// Set the fusion policy for BOTH Initial and Refined before either starts.
    ///
    /// Defaults are the existing RRF policy and 0.7 quality blend. `rrf` is
    /// owned by this query, so later caller configuration cannot change a phase.
    /// Its semantic weight also weights a hash-control vector lane without
    /// relabelling that lane. Every source weight must be strictly positive;
    /// zero is not a way to disable a required provider or its admission checks.
    ///
    /// The quality weight applies to documents present in BOTH vector pools.
    /// One-sided candidates keep their normalized score, including at 0 and 1,
    /// as in the existing union-preserving blend. Raw scores, physical rows and
    /// required-quality failure behavior do not change with weighting.
    ///
    /// # Errors
    /// Refuses cancellation, started queries, non-finite/out-of-range policy,
    /// or a maximum two-source RRF score outside public f32 result precision.
    pub fn with_fusion(mut self, rrf: RrfConfig, quality_weight: f32) -> SearchResult<Self> {
        self.configure()?;
        if !quality_weight.is_finite() || !(0.0..=1.0).contains(&quality_weight) {
            return Err(invalid(
                "progressive.quality_weight",
                "out-of-range",
                "quality blend weight must be finite and between zero and one",
            ));
        }
        if !rrf.k.is_finite()
            || rrf.k < 0.0
            || !rrf.lexical_weight.is_finite()
            || rrf.lexical_weight <= 0.0
            || !rrf.semantic_weight.is_finite()
            || rrf.semantic_weight <= 0.0
        {
            return Err(invalid(
                "progressive.rrf",
                "invalid",
                "RRF K must be finite and nonnegative, and source weights finite and positive",
            ));
        }
        let first_rank = 1.0 / (rrf.k + 1.0);
        let maximum_score = first_rank * rrf.lexical_weight + first_rank * rrf.semantic_weight;
        if !maximum_score.is_finite() || maximum_score > f64::from(f32::MAX) {
            return Err(invalid(
                "progressive.rrf",
                "score-overflow",
                "the maximum fused score must fit the public f32 result",
            ));
        }
        self.rrf = rrf;
        self.quality_weight = quality_weight;
        Ok(self)
    }

    /// Request `multiplier * max(k, rerank_window)` candidates from EACH source.
    ///
    /// The default is three. This is independent of the displayed page size and
    /// may change both pool-local normalization and final ranking. Configuring
    /// the reranker before or after this method produces the same budget.
    /// Scoped retrieval retains its own eligible-candidate and widening rules.
    /// This bounds requested hits, not graph visits or provider peak memory.
    ///
    /// # Errors
    /// Refuses zero, checked-arithmetic overflow, cancellation or a started query.
    pub fn with_candidate_multiplier(mut self, multiplier: usize) -> SearchResult<Self> {
        self.configure()?;
        if multiplier == 0 {
            return Err(invalid(
                "progressive.candidate_multiplier",
                "zero",
                "candidate multiplier must be positive",
            ));
        }
        self.budget = self
            .result_window()
            .checked_mul(multiplier)
            .ok_or_else(|| {
                invalid(
                    "progressive.candidate_budget",
                    "overflow",
                    "the configured candidate budget must fit usize",
                )
            })?;
        self.candidate_multiplier = multiplier;
        Ok(self)
    }

    /// Override the fast and quality native graph beams independently.
    ///
    /// None uses that owner's admitted default. The existing search policy
    /// raises a beam to its requested-hit floor, caps it at physical row count,
    /// and widens it on filtered underfill. Exact owners still ignore the beam.
    /// These settings do not rebuild graphs, alter source counts or certify ANN
    /// recall. Every configured provider is still independently admitted.
    ///
    /// # Errors
    /// Refuses a zero beam, cancellation or configuration after a phase starts.
    pub fn with_beam_widths(
        mut self,
        fast: Option<usize>,
        quality: Option<usize>,
    ) -> SearchResult<Self> {
        self.configure()?;
        if fast == Some(0) || quality == Some(0) {
            return Err(invalid(
                "progressive.beam",
                "zero",
                "an explicit graph beam must be positive",
            ));
        }
        self.fast_ef = fast;
        self.quality_ef = quality;
        Ok(self)
    }

    fn configure(&self) -> SearchResult<()> {
        checkpoint(self.cx, "native_ann.progressive_configuration")?;
        if !matches!(self.state, State::Initial) {
            return Err(invalid(
                "progressive.configuration",
                "started",
                "query policy must be fixed before requesting its first phase",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "quill")]
    pub(in crate::native_ann) fn with_allowed_documents(
        mut self,
        allowed: &'a BTreeSet<String>,
    ) -> SearchResult<Self> {
        checkpoint(self.cx, "native_ann.scope.configure")?;
        if !matches!(self.state, State::Initial) {
            return Err(invalid(
                "scope.configuration",
                "started",
                "scope membership must be frozen before the first phase",
            ));
        }
        self.allowed_documents = Some(allowed);
        Ok(self)
    }

    async fn search_tier(
        &self,
        index: &NativeAnnIndex,
        embedder: &dyn Embedder,
        ef: Option<usize>,
    ) -> SearchResult<Vec<VectorHit>> {
        match self.allowed_documents {
            Some(allowed) => {
                index
                    .search_text_filtered(
                        self.cx,
                        embedder,
                        self.text,
                        self.budget.min(allowed.len()),
                        ef,
                        |id| allowed.contains(id),
                    )
                    .await
            }
            None => {
                index
                    .search_text(self.cx, embedder, self.text, self.budget, ef)
                    .await
            }
        }
    }

    /// Add a lazy final cross-encoder stage before requesting the first phase.
    ///
    /// `top_k` bounds the rerank window, independently of the displayed page.
    /// Retrieval budgets expand to the configured multiplier times
    /// `max(k, top_k)` (three by default) so reranking can
    /// promote candidates outside the initial page. Enabling this option may
    /// therefore change initial fusion relative to a smaller, unreranked query.
    /// Only requesting the final phase resolves text or calls the reranker;
    /// abandoning earlier phases incurs no cross-encoder inference.
    ///
    /// `text_fn` must resolve text from the same document versions as retrieval,
    /// for example by capturing the consumer's immutable source snapshot. This
    /// API retains the lexical scoring pin, but cannot prove the provenance of
    /// an arbitrary closure. Missing text skips that candidate. When no text is
    /// available, the final `next_phase` returns `None`, not a fabricated rerank.
    ///
    /// A failed quality refinement still permits reranking the retained initial
    /// pool. Inference is caller-owned and cancellation/drop is terminal. Raw
    /// retrieval scores and their explanations are preserved; `rerank_score`
    /// records the new ordering signal. No model is downloaded or substituted.
    ///
    /// # Errors
    ///
    /// Refuses zero/overflowing budgets, cancellation, or configuration after
    /// a phase has started. Model availability is checked only at the final phase.
    pub fn with_reranker(
        mut self,
        reranker: &'a dyn Reranker,
        text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
        top_k: usize,
    ) -> SearchResult<Self> {
        checkpoint(self.cx, "native_ann.rerank_configuration")?;
        if !matches!(self.state, State::Initial) || top_k == 0 {
            return Err(invalid(
                "rerank.configuration",
                "started-or-empty-window",
                "attach a positive rerank window before requesting any phase",
            ));
        }
        self.budget = self
            .k
            .max(top_k)
            .checked_mul(self.candidate_multiplier)
            .ok_or_else(|| {
                invalid(
                    "rerank.candidate_budget",
                    "overflow",
                    "rerank retrieval budget must fit usize",
                )
            })?;
        self.reranker = Some(RerankRequest {
            reranker,
            text: text_fn,
            limit: top_k,
        });
        Ok(self)
    }

    /// Whether every phase finished or a polled phase was abandoned.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// Execute every configured phase and return only the final successful page.
    ///
    /// This consumes an unstarted query with all of its configured fusion,
    /// candidate, beam and scope settings. Missing rerank text keeps the last
    /// valid retrieval page. Use `next_phase` instead for early delivery and
    /// explicit degradation pages.
    ///
    /// # Errors
    /// Propagates any required phase failure or cancellation, including errors
    /// carried by RefinementFailed/RerankFailed; it never silently returns an
    /// earlier page as successful refinement. Refuses an already-started query.
    pub async fn collect(mut self) -> SearchResult<Vec<ScoredResult>> {
        self.configure()?;
        let mut page = Vec::new();
        while let Some(phase) = self.next_phase().await? {
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

    /// Compute one requested phase, then finish permanently after the last.
    ///
    /// # Errors
    ///
    /// Initial failures and any cancellation propagate directly. After initial
    /// success, other stage failures carry the preceding results. Optional
    /// reranking follows quality success or non-cancellation failure. Every
    /// returned `Err` leaves this stream finished.
    pub async fn next_phase(&mut self) -> SearchResult<Option<NativeSearchPhase>> {
        // Own the phase before the first suspension. Cancellation by dropping
        // this future cannot leave a runnable copy of the work in the stream.
        match std::mem::replace(&mut self.state, State::Done) {
            State::Done => Ok(None),
            State::Initial => self.initial().await.map(Some),
            State::Refine(pending) => self.refine(pending).await.map(Some),
            State::Rerank(pending) => self.rerank(pending).await,
        }
    }

    fn admit(&self) -> SearchResult<bool> {
        if let Some(quality) = self.quality {
            return admit_pair(self.cx, self.fast, self.fast_embedder, quality);
        }
        checkpoint(self.cx, "native_ann.progressive_admission")?;
        let identity = self.fast_embedder.identity()?;
        self.fast.admit_identity(identity)?;
        // Classify the same admitted reference, not a second provider call.
        Ok(matches!(
            identity.space.kind,
            EmbeddingSpaceKindV1::HashControl
        ))
    }

    async fn initial(&mut self) -> SearchResult<NativeSearchPhase> {
        let is_hash = self.admit()?;
        if self.k == 0 {
            return Ok(NativeSearchPhase::Initial {
                results: Vec::new(),
                candidates: NativePhaseCandidates::default(),
            });
        }
        let (fast, batch) = join_sources(
            self.cx,
            self.search_tier(self.fast, self.fast_embedder, self.fast_ef),
            checked_lexical(self.cx, self.lexical, self.text, self.budget),
        )
        .await?;
        let candidates = NativePhaseCandidates {
            fast: fast.len(),
            quality: 0,
            lexical: batch.results().len(),
        };
        let fused = rrf_fuse_for_vector_lane(
            batch.results(),
            &fast,
            self.result_window(),
            0,
            &self.rrf,
            is_hash,
        );
        let mut results = hydrate_winners(self.cx, self.lexical, fused, &batch).await?;
        checkpoint(self.cx, "native_ann.progressive_initial_complete")?;
        if self.quality.is_some() {
            self.state = State::Refine(PendingRefinement {
                fast,
                batch,
                initial_results: results.clone(),
            });
            results.truncate(self.k);
        } else {
            results = self.finish_retrieval(results, candidates, batch);
        }
        Ok(NativeSearchPhase::Initial {
            results,
            candidates,
        })
    }

    async fn refine(&mut self, pending: PendingRefinement) -> SearchResult<NativeSearchPhase> {
        match self.refined_results(&pending).await {
            Ok((results, candidates)) => {
                let results = self.finish_retrieval(results, candidates, pending.batch);
                Ok(NativeSearchPhase::Refined {
                    results,
                    candidates,
                })
            }
            Err(error) => {
                checkpoint(self.cx, "native_ann.progressive_failure")?;
                if matches!(error, SearchError::Cancelled { .. }) {
                    return Err(error);
                }
                let candidates = NativePhaseCandidates {
                    fast: pending.fast.len(),
                    quality: 0,
                    lexical: pending.batch.results().len(),
                };
                let initial_results =
                    self.finish_retrieval(pending.initial_results, candidates, pending.batch);
                Ok(NativeSearchPhase::RefinementFailed {
                    initial_results,
                    error,
                })
            }
        }
    }

    async fn refined_results(
        &self,
        pending: &PendingRefinement,
    ) -> SearchResult<(Vec<ScoredResult>, NativePhaseCandidates)> {
        let is_hash = self.admit()?;
        let (quality, embedder) = self.quality.ok_or_else(|| {
            invalid(
                "progressive.quality",
                "missing",
                "refinement requires its admitted arm",
            )
        })?;
        let quality_hits = self.search_tier(quality, embedder, self.quality_ef).await?;
        let candidates = NativePhaseCandidates {
            fast: pending.fast.len(),
            quality: quality_hits.len(),
            lexical: pending.batch.results().len(),
        };
        let results = refined_winners(
            self.cx,
            self.lexical,
            &pending.batch,
            &pending.fast,
            &quality_hits,
            self.result_window(),
            RefinementFusion {
                quality_weight: self.quality_weight,
                rrf: &self.rrf,
                is_hash,
            },
        )
        .await?;
        checkpoint(self.cx, "native_ann.progressive_refined_complete")?;
        Ok((results, candidates))
    }

    fn result_window(&self) -> usize {
        self.reranker
            .map_or(self.k, |request| self.k.max(request.limit))
    }

    fn finish_retrieval(
        &mut self,
        mut results: Vec<ScoredResult>,
        candidates: NativePhaseCandidates,
        batch: LexicalCandidateBatch,
    ) -> Vec<ScoredResult> {
        if self.reranker.is_some() && !results.is_empty() {
            let displayed = results.iter().take(self.k).cloned().collect();
            self.state = State::Rerank(PendingRerank {
                results,
                candidates,
                _batch: batch,
            });
            displayed
        } else {
            results.truncate(self.k);
            results
        }
    }

    async fn rerank(&self, pending: PendingRerank) -> SearchResult<Option<NativeSearchPhase>> {
        let request = self.reranker.as_ref().ok_or_else(|| {
            invalid(
                "rerank.configuration",
                "missing",
                "a queued rerank must retain its provider",
            )
        })?;
        match request.plan(self.cx, self.text, &pending.results).await {
            Ok(Some(plan)) => {
                let evaluated = plan.evaluated;
                let mut results = plan.apply(pending.results, |result| result);
                results.truncate(self.k);
                checkpoint(self.cx, "native_ann.rerank_complete")?;
                Ok(Some(NativeSearchPhase::Reranked {
                    results,
                    candidates: pending.candidates,
                    evaluated,
                }))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                checkpoint(self.cx, "native_ann.rerank_failure")?;
                if matches!(error, SearchError::Cancelled { .. }) {
                    return Err(error);
                }
                let mut previous_results = pending.results;
                previous_results.truncate(self.k);
                Ok(Some(NativeSearchPhase::RerankFailed {
                    previous_results,
                    error,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    use frankensearch_core::LexicalHydrationContext;
    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

    use crate::ScoreSource;

    #[derive(Clone, Copy)]
    enum Reply {
        Correct,
        Failed,
        Cancelled,
        Pending,
        ForeignProducer,
    }

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        foreign_identity: EmbeddingIdentityBundleV1,
        advertise_foreign: AtomicBool,
        calls: AtomicUsize,
        drops: AtomicUsize,
        reply: Reply,
    }

    impl Provider {
        fn new(name: &str, dimension: u32) -> Self {
            // Synthetic semantic-shaped fixture, not evidence of a real model.
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
            identity.space.kind = EmbeddingSpaceKindV1::Semantic;
            identity.space.hash_control = None;
            identity.space.artifact_manifest_fingerprint = "a".repeat(64);
            identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "b".repeat(64),
                size: 1,
            }];
            identity.producer.space_fingerprint = identity.space.fingerprint();
            identity.validate().unwrap();
            let mut foreign_identity = identity.clone();
            foreign_identity.producer.backend = "foreign-backend".to_owned();
            Self {
                identity,
                foreign_identity,
                advertise_foreign: AtomicBool::new(false),
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
                reply: Reply::Correct,
            }
        }
    }

    struct DropCount<'a>(&'a AtomicUsize);

    impl Drop for DropCount<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                let mut values = vec![0.0; self.dimension()];
                values[0] = 1.0;
                Ok(values)
            })
        }

        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                let identity = match self.reply {
                    Reply::Correct => self.identity.clone(),
                    Reply::ForeignProducer => self.foreign_identity.clone(),
                    Reply::Failed => {
                        return Err(invalid("test.provider", "failed", "provider failed"));
                    }
                    Reply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "test.provider".to_owned(),
                            reason: "provider cancelled".to_owned(),
                        });
                    }
                    Reply::Pending => return std::future::pending().await,
                };
                Ok(IdentityBoundEmbedding {
                    values: self.embed(cx, text).await?,
                    identity,
                })
            })
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(if self.advertise_foreign.load(Ordering::SeqCst) {
                &self.foreign_identity
            } else {
                &self.identity
            })
        }

        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }

        fn model_name(&self) -> &str {
            self.id()
        }

        fn dimension(&self) -> usize {
            usize::try_from(self.identity.space.dimension).unwrap()
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::TransformerEmbedder
        }
    }

    struct Lexical {
        ids: Vec<&'static str>,
        calls: AtomicUsize,
        hydration_calls: AtomicUsize,
        hydration_drops: AtomicUsize,
        snapshot: Arc<usize>,
        current_generation: AtomicUsize,
        pending_search: bool,
        refinement_hydration: Reply,
    }

    impl Lexical {
        fn new(ids: &[&'static str]) -> Self {
            Self {
                ids: ids.to_vec(),
                calls: AtomicUsize::new(0),
                hydration_calls: AtomicUsize::new(0),
                hydration_drops: AtomicUsize::new(0),
                snapshot: Arc::new(1),
                current_generation: AtomicUsize::new(1),
                pending_search: false,
                refinement_hydration: Reply::Correct,
            }
        }
    }

    impl LexicalRead for Lexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if self.pending_search {
                    std::future::pending::<()>().await;
                }
                Ok(self
                    .ids
                    .iter()
                    .take(limit)
                    .map(|id| ScoredResult {
                        doc_id: (*id).into(),
                        score: 10.0,
                        source: ScoreSource::Lexical,
                        index: None,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some(10.0),
                        rerank_score: None,
                        explanation: None,
                        metadata: None,
                    })
                    .collect())
            })
        }

        fn search_candidates<'a>(
            &'a self,
            cx: &'a Cx,
            text: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, LexicalCandidateBatch> {
            Box::pin(async move {
                let results = self.search(cx, text, limit).await?;
                self.current_generation.store(2, Ordering::SeqCst);
                Ok(LexicalCandidateBatch::deferred(
                    results,
                    LexicalHydrationContext::new(
                        "native-progressive-test",
                        Box::new(Arc::clone(&self.snapshot)),
                    ),
                ))
            })
        }

        fn hydrate_candidates<'a>(
            &'a self,
            _cx: &'a Cx,
            context: Option<&'a LexicalHydrationContext>,
            results: &'a mut [ScoredResult],
        ) -> SearchFuture<'a, ()> {
            Box::pin(async move {
                let call = self.hydration_calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.hydration_drops);
                let pin = context
                    .and_then(LexicalHydrationContext::downcast_ref::<Arc<usize>>)
                    .filter(|pin| Arc::ptr_eq(pin, &self.snapshot))
                    .ok_or_else(|| invalid("test.snapshot", "foreign", "wrong snapshot"))?;
                for result in results {
                    if result.lexical_score.is_some() {
                        result.metadata = Some(Arc::new(serde_json::json!({"generation": **pin})));
                    }
                }
                if call > 0 {
                    match self.refinement_hydration {
                        Reply::Pending => return std::future::pending().await,
                        Reply::Failed => {
                            return Err(invalid("test.hydration", "failed", "partial hydration"));
                        }
                        Reply::Cancelled => {
                            return Err(SearchError::Cancelled {
                                phase: "test.hydration".to_owned(),
                                reason: "cancelled after writing partial metadata".to_owned(),
                            });
                        }
                        Reply::Correct | Reply::ForeignProducer => {}
                    }
                }
                Ok(())
            })
        }

        fn doc_count(&self) -> SearchResult<usize> {
            Ok(self.ids.len())
        }
    }

    fn generation() -> ArtifactGenerationIdentityV1 {
        ArtifactGenerationIdentityV1::new(1, [0x61; 16]).unwrap()
    }

    fn native_index(
        cx: &Cx,
        provider: &Provider,
        generation: ArtifactGenerationIdentityV1,
        rows: &[(&str, &[f32])],
    ) -> NativeAnnIndex {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier.fsvi");
        let mut bundle = provider.identity.clone();
        bundle.storage.format = "fsvi-v2".to_owned();
        bundle.storage.quantization = QuantizationFormat::F32;
        bundle.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(generation, bundle.freeze().unwrap()).unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for &(id, values) in rows {
            writer.write_record(id, values).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
    }

    fn fast_index(cx: &Cx, provider: &Provider) -> NativeAnnIndex {
        native_index(
            cx,
            provider,
            generation(),
            &[
                ("z-fast", &[1.0, 0.0]),
                ("middle", &[0.8, 0.6]),
                ("near", &[0.6, 0.8]),
                ("a-quality", &[0.0, 1.0]),
            ],
        )
    }

    fn quality_index(cx: &Cx, provider: &Provider) -> NativeAnnIndex {
        native_index(
            cx,
            provider,
            generation(),
            &[
                ("a-quality", &[1.0, 0.0, 0.0]),
                ("near", &[0.0, 1.0, 0.0]),
                ("middle", &[0.0, 0.0, 1.0]),
                ("z-fast", &[-1.0, 0.0, 0.0]),
            ],
        )
    }

    #[test]
    fn configured_defaults_collect_the_existing_refined_page() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["a-quality", "z-fast"]);
            let expected = index
                .search_hybrid_refined_text(&cx, &fast, (&qindex, &quality), &lexical, "query", 3)
                .await
                .unwrap();
            let actual = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    3,
                )
                .unwrap()
                .with_fusion(RrfConfig::default(), 0.7)
                .unwrap()
                .with_candidate_multiplier(3)
                .unwrap()
                .with_beam_widths(None, None)
                .unwrap()
                .collect()
                .await
                .unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        });
    }

    #[test]
    fn configured_quality_weight_changes_ranking_not_raw_evidence_or_row_space() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            for (weight, winner) in [(0.0, "z-fast"), (1.0, "a-quality")] {
                let results = index
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&qindex, &quality)),
                        &lexical,
                        "query",
                        4,
                    )
                    .unwrap()
                    .with_fusion(RrfConfig::default(), weight)
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
                assert_eq!(results[0].doc_id, winner);
                assert_eq!(results.len(), 4);
                let a = results
                    .iter()
                    .find(|hit| hit.doc_id == "a-quality")
                    .unwrap();
                assert_eq!(a.fast_score, Some(0.0));
                assert_eq!(a.quality_score, Some(1.0));
                assert_eq!(a.source, ScoreSource::SemanticQuality);
                // Zero is a real fast score, so even at quality weight 1 the
                // index belongs to the contributing fast owner, not to a rank.
                assert_eq!(a.index, Some(2));
                assert_eq!(index.owner.doc_id_at(2).unwrap(), a.doc_id);
                let z = results.iter().find(|hit| hit.doc_id == "z-fast").unwrap();
                assert_eq!(z.fast_score, Some(1.0));
                assert_eq!(z.quality_score, Some(-1.0));
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
        });
    }

    #[test]
    fn configured_rrf_weights_and_k_are_applied_to_both_phases() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["lexical-only"]);
            for (k, expected_score) in [(0.0, 2.0_f32), (60.0, 2.0_f32 / 61.0)] {
                for prefer_lexical in [false, true] {
                    let config = RrfConfig {
                        k,
                        lexical_weight: if prefer_lexical { 2.0 } else { 1.0 },
                        semantic_weight: if prefer_lexical { 1.0 } else { 2.0 },
                        ..RrfConfig::default()
                    };
                    let mut stream = index
                        .search_hybrid_progressive(
                            &cx,
                            &fast,
                            Some((&qindex, &quality)),
                            &lexical,
                            "query",
                            1,
                        )
                        .unwrap()
                        .with_fusion(config, 0.7)
                        .unwrap();
                    let NativeSearchPhase::Initial { results, .. } =
                        stream.next_phase().await.unwrap().unwrap()
                    else {
                        panic!("initial")
                    };
                    assert_eq!(
                        results[0].doc_id,
                        if prefer_lexical {
                            "lexical-only"
                        } else {
                            "z-fast"
                        }
                    );
                    assert_eq!(results[0].score.to_bits(), expected_score.to_bits());
                    let NativeSearchPhase::Refined { results, .. } =
                        stream.next_phase().await.unwrap().unwrap()
                    else {
                        panic!("refined")
                    };
                    assert_eq!(
                        results[0].doc_id,
                        if prefer_lexical {
                            "lexical-only"
                        } else {
                            "a-quality"
                        }
                    );
                    assert_eq!(results[0].score.to_bits(), expected_score.to_bits());
                    if prefer_lexical {
                        assert_eq!(results[0].source, ScoreSource::Lexical);
                        assert!(results[0].index.is_none());
                        assert_eq!(results[0].metadata.as_deref().unwrap()["generation"], 1);
                    } else {
                        assert_eq!(results[0].quality_score, Some(1.0));
                        assert!(results[0].fast_score.is_none());
                        assert_eq!(
                            qindex
                                .owner
                                .doc_id_at(results[0].index.unwrap() as usize)
                                .unwrap(),
                            results[0].doc_id
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn candidate_multiplier_is_effective_and_independent_of_reranker_configuration_order() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast", "middle", "near", "a-quality"]);
            for multiplier in [1, 2, 4] {
                let mut stream = index
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&qindex, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap()
                    .with_candidate_multiplier(multiplier)
                    .unwrap();
                let NativeSearchPhase::Initial {
                    candidates,
                    results,
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial")
                };
                assert_eq!(results.len(), 1);
                assert_eq!(
                    candidates,
                    NativePhaseCandidates {
                        fast: multiplier,
                        quality: 0,
                        lexical: multiplier
                    }
                );
                let NativeSearchPhase::Refined { candidates, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("refined")
                };
                assert_eq!(
                    candidates,
                    NativePhaseCandidates {
                        fast: multiplier,
                        quality: multiplier,
                        lexical: multiplier
                    }
                );
            }
            let model = CrossEncoder::new(RerankReply::Winner);
            let mut pages = Vec::new();
            for reranker_first in [false, true] {
                let stream = index
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                    .unwrap();
                let mut stream = if reranker_first {
                    stream
                        .with_reranker(&model, &source_text, 3)
                        .unwrap()
                        .with_candidate_multiplier(1)
                        .unwrap()
                } else {
                    stream
                        .with_candidate_multiplier(1)
                        .unwrap()
                        .with_reranker(&model, &source_text, 3)
                        .unwrap()
                };
                let NativeSearchPhase::Initial { candidates, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial")
                };
                assert_eq!(candidates.fast, 3);
                assert_eq!(candidates.lexical, 3);
                let NativeSearchPhase::Reranked {
                    results, evaluated, ..
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("reranked")
                };
                assert_eq!(evaluated, 3);
                assert_eq!(results[0].doc_id, "near");
                pages.push(serde_json::to_value(results).unwrap());
            }
            assert_eq!(pages[0], pages[1]);
        });
    }

    #[test]
    fn invalid_or_overflowing_query_policy_is_not_sanitized_into_default_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let fresh = |k| {
                index
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", k)
                    .unwrap()
            };
            for k in [0, 2] {
                for weight in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
                    assert!(fresh(k).with_fusion(RrfConfig::default(), weight).is_err());
                }
                for value in [f64::NAN, f64::INFINITY, -1.0] {
                    assert!(
                        fresh(k)
                            .with_fusion(
                                RrfConfig {
                                    k: value,
                                    ..RrfConfig::default()
                                },
                                0.7
                            )
                            .is_err()
                    );
                }
                for value in [f64::NAN, f64::INFINITY, -1.0, 0.0, f64::MAX] {
                    assert!(
                        fresh(k)
                            .with_fusion(
                                RrfConfig {
                                    lexical_weight: value,
                                    ..RrfConfig::default()
                                },
                                0.7
                            )
                            .is_err()
                    );
                    assert!(
                        fresh(k)
                            .with_fusion(
                                RrfConfig {
                                    semantic_weight: value,
                                    ..RrfConfig::default()
                                },
                                0.7
                            )
                            .is_err()
                    );
                }
                assert!(fresh(k).with_candidate_multiplier(0).is_err());
                assert!(fresh(k).with_beam_widths(Some(0), None).is_err());
                assert!(fresh(k).with_beam_widths(None, Some(0)).is_err());
            }
            assert!(fresh(2).with_candidate_multiplier(usize::MAX).is_err());
            let model = CrossEncoder::new(RerankReply::Unavailable);
            assert!(
                fresh(1)
                    .with_candidate_multiplier(2)
                    .unwrap()
                    .with_reranker(&model, &source_text, usize::MAX)
                    .is_err()
            );
            assert!(
                fresh(1)
                    .with_reranker(&model, &source_text, 2)
                    .unwrap()
                    .with_candidate_multiplier(usize::MAX)
                    .is_err()
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn tuning_cannot_change_a_started_query_or_disable_required_quality() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Failed;
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            for setting in 0..4 {
                let mut stream = index
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&qindex, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap();
                assert!(stream.next_phase().await.unwrap().is_some());
                match setting {
                    0 => assert!(stream.with_fusion(RrfConfig::default(), 0.0).is_err()),
                    1 => assert!(stream.with_candidate_multiplier(1).is_err()),
                    2 => assert!(stream.with_beam_widths(Some(1), None).is_err()),
                    _ => assert!(stream.collect().await.is_err()),
                }
            }
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            let error = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap()
                .with_fusion(RrfConfig::default(), 0.0)
                .unwrap()
                .collect()
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.test.provider")
            );
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn beam_overrides_reach_each_tier_before_retained_owner_rescoring() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new("fast", 2);
            let lexical = Lexical::new(&[]);
            let mut changed_fast = false;
            let mut changed_quality = false;
            for reverse in [false, true] {
                // Both graph distances round to 1.0, but their retained f32
                // dot products differ. A full beam must see the better row.
                let low = [1.0e-8, 1.0];
                let high = [2.0e-8, 1.0];
                let rows: [(&str, &[f32]); 2] = if reverse {
                    [("a", &high), ("b", &low)]
                } else {
                    [("a", &low), ("b", &high)]
                };
                let winner = if reverse { "a" } else { "b" };
                let mut index = native_index(&cx, &provider, generation(), &rows);
                index.default_ef_search = 1;
                let narrow = index
                    .search_hybrid_progressive(&cx, &provider, None, &lexical, "query", 1)
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
                let wide = index
                    .search_hybrid_progressive(&cx, &provider, None, &lexical, "query", 1)
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .with_beam_widths(Some(2), None)
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
                assert_eq!(wide[0].doc_id, winner);
                assert_eq!(wide[0].fast_score, Some(high[0]));
                changed_fast |= narrow[0].doc_id != wide[0].doc_id;
                let empty = native_index(&cx, &provider, generation(), &[]);
                let narrow = empty
                    .search_hybrid_progressive(
                        &cx,
                        &provider,
                        Some((&index, &provider)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
                let wide = empty
                    .search_hybrid_progressive(
                        &cx,
                        &provider,
                        Some((&index, &provider)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .with_beam_widths(None, Some(2))
                    .unwrap()
                    .collect()
                    .await
                    .unwrap();
                assert_eq!(wide[0].doc_id, winner);
                assert_eq!(wide[0].quality_score, Some(high[0]));
                assert!(wide[0].fast_score.is_none());
                assert_eq!(
                    index
                        .owner
                        .doc_id_at(wide[0].index.unwrap() as usize)
                        .unwrap(),
                    winner
                );
                changed_quality |= narrow[0].doc_id != wide[0].doc_id;
            }
            assert!(
                changed_fast && changed_quality,
                "the overrides must change actual retrieval, not only stored configuration"
            );
        });
    }

    #[test]
    fn lazy_refinement_retrieves_a_winner_outside_the_fast_candidate_pool() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (padding, quality_row) in [("middle", 2_u32), ("quality-padding", 1)] {
                let fast = Provider::new("fast", 2);
                let quality = Provider::new("quality", 3);
                let fast_index = fast_index(&cx, &fast);
                // FSVI persists hash-sorted rows, not insertion positions. Use a
                // different quality cohort so the winning document really has
                // distinct physical positions in the two retained owners.
                let quality_index = native_index(
                    &cx,
                    &quality,
                    generation(),
                    &[
                        ("a-quality", &[1.0, 0.0, 0.0]),
                        ("near", &[0.0, 1.0, 0.0]),
                        (padding, &[0.0, 0.0, 1.0]),
                        ("z-fast", &[-1.0, 0.0, 0.0]),
                    ],
                );
                assert_eq!(fast_index.owner.doc_id_at(2).unwrap(), "a-quality");
                assert_eq!(
                    quality_index.owner.doc_id_at(quality_row as usize).unwrap(),
                    "a-quality"
                );
                let lexical = Lexical::new(&[]);
                let mut stream = fast_index
                    .search_hybrid_progressive(
                        &cx,
                        &fast,
                        Some((&quality_index, &quality)),
                        &lexical,
                        "query",
                        1,
                    )
                    .unwrap();
                assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
                let NativeSearchPhase::Initial {
                    results,
                    candidates,
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("expected initial phase");
                };
                assert_eq!(results[0].doc_id, "z-fast");
                assert_eq!(
                    candidates,
                    NativePhaseCandidates {
                        fast: 3,
                        quality: 0,
                        lexical: 0
                    }
                );
                assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
                let State::Refine(pending) = &stream.state else {
                    panic!("missing retained pool")
                };
                assert!(pending.fast.iter().all(|hit| hit.doc_id != "a-quality"));
                let NativeSearchPhase::Refined {
                    results,
                    candidates,
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("expected independent refinement");
                };
                assert_eq!(results[0].doc_id, "a-quality");
                assert_eq!(results[0].quality_score, Some(1.0));
                assert!(results[0].fast_score.is_none());
                assert_eq!(results[0].source, ScoreSource::SemanticQuality);
                assert_eq!(results[0].index, Some(quality_row));
                let row = results[0].index.unwrap() as usize;
                assert_eq!(
                    quality_index.owner.doc_id_at(row).unwrap(),
                    results[0].doc_id
                );
                assert_eq!(
                    quality_index.owner.vector_at_f32(row).unwrap(),
                    vec![1.0, 0.0, 0.0]
                );
                assert_eq!(
                    candidates,
                    NativePhaseCandidates {
                        fast: 3,
                        quality: 3,
                        lexical: 0
                    }
                );
                assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
                assert!(stream.is_finished());
                assert!(stream.next_phase().await.unwrap().is_none());
            }
        });
    }

    #[test]
    fn stopping_after_initial_starts_no_quality_work_and_releases_the_pin() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Pending;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(stream);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.drops.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn refinement_failure_carries_the_unchanged_initial_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Failed;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected initial");
            };
            let expected = serde_json::to_value(results).unwrap();
            let NativeSearchPhase::RefinementFailed {
                initial_results,
                error,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("provider failure must be explicit");
            };
            assert_eq!(serde_json::to_value(initial_results).unwrap(), expected);
            assert!(
                matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "native_ann.test.provider")
            );
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn provider_cancellation_is_terminal_not_successful_degradation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Cancelled;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(
                matches!(stream.next_phase().await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.provider")
            );
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn dropping_a_polled_refinement_future_fuses_the_stream_and_drops_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Pending;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(future);
            assert!(stream.is_finished());
            assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn fast_only_and_zero_k_have_a_single_initial_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let mut empty = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 0)
                .unwrap();
            let NativeSearchPhase::Initial {
                results,
                candidates,
            } = empty.next_phase().await.unwrap().unwrap()
            else {
                panic!("zero k yields an empty initial phase");
            };
            assert!(results.is_empty());
            assert_eq!(candidates, NativePhaseCandidates::default());
            assert!(empty.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("fast-only phase");
            };
            assert_eq!(results[0].doc_id, "z-fast");
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn refinement_reuses_the_scoring_snapshot_and_matches_the_existing_refined_api() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["a-quality", "z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    2,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            lexical.current_generation.store(42, Ordering::SeqCst);
            let NativeSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("expected refined");
            };
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.current_generation.load(Ordering::SeqCst), 42);
            assert!(
                results
                    .iter()
                    .all(|hit| hit.metadata.as_deref().unwrap()["generation"] == 1)
            );
            let eager = fast_index
                .search_hybrid_refined_text(
                    &cx,
                    &fast,
                    (&quality_index, &quality),
                    &lexical,
                    "query",
                    2,
                )
                .await
                .unwrap();
            assert_eq!(
                serde_json::to_value(results).unwrap(),
                serde_json::to_value(eager).unwrap()
            );
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn changed_quality_identity_is_rejected_before_refinement_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            quality.advertise_foreign.store(true, Ordering::SeqCst);
            let NativeSearchPhase::RefinementFailed {
                initial_results,
                error,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("changed producer must not be admitted");
            };
            assert_eq!(initial_results[0].doc_id, "z-fast");
            assert!(
                matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "query_embedding.native_ann.producer_conformance")
            );
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(stream.is_finished());
        });
    }

    #[test]
    fn dropping_a_polled_initial_future_does_not_restart_either_source() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut fast = Provider::new("fast", 2);
            fast.reply = Reply::Pending;
            let index = fast_index(&cx, &fast);
            let mut lexical = Lexical::new(&[]);
            lexical.pending_search = true;
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap();
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            drop(future);
            assert_eq!(fast.drops.load(Ordering::SeqCst), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn pair_admission_and_budget_overflow_refuse_without_any_provider_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let foreign_index = native_index(
                &cx,
                &quality,
                ArtifactGenerationIdentityV1::new(1, [0x62; 16]).unwrap(),
                &[],
            );
            let lexical = Lexical::new(&[]);
            for k in [0, 1] {
                assert!(matches!(
                    fast_index.search_hybrid_progressive(&cx, &fast, Some((&foreign_index, &quality)), &lexical, "query", k),
                    Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.tier_generation"
                ));
            }
            assert!(matches!(
                fast_index.search_hybrid_progressive(&cx, &fast, None, &lexical, "query", usize::MAX),
                Err(SearchError::InvalidConfig { ref field, .. }) if field == "native_ann.progressive.candidate_budget"
            ));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn dropping_pending_refinement_hydration_releases_the_original_pin() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let mut lexical = Lexical::new(&["a-quality", "z-fast"]);
            lexical.refinement_hydration = Reply::Pending;
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    2,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            assert_eq!(lexical.hydration_calls.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(future);
            assert_eq!(lexical.hydration_drops.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn a_substituted_actual_quality_response_cannot_produce_a_refined_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::ForeignProducer;
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let NativeSearchPhase::RefinementFailed {
                initial_results,
                error,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("substituted producer must fail");
            };
            assert_eq!(initial_results[0].doc_id, "z-fast");
            assert!(
                matches!(error, SearchError::InvalidConfig { ref field, .. } if field == "query_embedding.native_ann.producer_conformance")
            );
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn cancellation_after_partial_refinement_hydration_is_not_partial_success() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let mut lexical = Lexical::new(&["a-quality", "z-fast"]);
            lexical.refinement_hydration = Reply::Cancelled;
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    2,
                )
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(
                matches!(stream.next_phase().await, Err(SearchError::Cancelled { ref phase, .. }) if phase == "test.hydration")
            );
            assert_eq!(lexical.hydration_calls.load(Ordering::SeqCst), 2);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn empty_fast_owner_does_not_prevent_direct_quality_retrieval() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut fast = Provider::new("fast", 2);
            fast.reply = Reply::Pending;
            let quality = Provider::new("quality", 3);
            let fast_index = native_index(&cx, &fast, generation(), &[]);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            let NativeSearchPhase::Initial {
                results,
                candidates,
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial phase");
            };
            assert!(results.is_empty());
            assert_eq!(candidates.fast, 0);
            let NativeSearchPhase::Refined { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("quality must retrieve independently");
            };
            assert_eq!(results[0].doc_id, "a-quality");
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn dropping_an_unpolled_phase_does_not_consume_its_work_or_snapshot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let fast_index = fast_index(&cx, &fast);
            let quality_index = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast"]);
            let mut stream = fast_index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&quality_index, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap();
            drop(stream.next_phase());
            assert!(!stream.is_finished());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            drop(stream.next_phase());
            assert!(!stream.is_finished());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Refined { .. })
            ));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[derive(Clone, Copy)]
    enum RerankReply {
        Winner,
        Failed,
        Pending,
        Cancelled,
        CancelWithScores,
        DuplicateRank,
        ForeignId,
        OutOfRange,
        MissingScore,
        NonFinite,
        NonFiniteLogit,
        Unavailable,
    }

    struct CrossEncoder {
        reply: RerankReply,
        calls: AtomicUsize,
        drops: AtomicUsize,
    }

    impl CrossEncoder {
        fn new(reply: RerankReply) -> Self {
            Self {
                reply,
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
            }
        }
    }

    impl Reranker for CrossEncoder {
        fn rerank<'a>(
            &'a self,
            cx: &'a Cx,
            _query: &'a str,
            documents: &'a [frankensearch_core::traits::RerankDocument],
        ) -> SearchFuture<'a, Vec<frankensearch_core::traits::RerankScore>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = DropCount(&self.drops);
                let mut scores: Vec<_> = documents
                    .iter()
                    .enumerate()
                    .map(|(rank, doc)| {
                        assert_eq!(doc.text, format!("version-one:{}", doc.doc_id));
                        frankensearch_core::traits::RerankScore {
                            doc_id: doc.doc_id.clone(),
                            score: if doc.doc_id == "near" { 0.9 } else { 0.1 },
                            original_rank: rank,
                            raw_logit: None,
                        }
                    })
                    .collect();
                match self.reply {
                    RerankReply::Winner => {}
                    RerankReply::Pending => return std::future::pending().await,
                    RerankReply::Failed => {
                        return Err(invalid("test.reranker", "failed", "model failed"));
                    }
                    RerankReply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "test.reranker".to_owned(),
                            reason: "model cancelled".to_owned(),
                        });
                    }
                    RerankReply::CancelWithScores => {
                        cx.cancel_with(
                            asupersync::CancelKind::User,
                            Some("native rerank finished after cancellation"),
                        );
                    }
                    RerankReply::DuplicateRank => scores[1] = scores[0].clone(),
                    RerankReply::ForeignId => scores[0].doc_id = "foreign".to_owned(),
                    RerankReply::OutOfRange => scores[0].original_rank = documents.len(),
                    RerankReply::MissingScore => {
                        let _ = scores.pop();
                    }
                    RerankReply::NonFinite => scores[0].score = f32::INFINITY,
                    RerankReply::NonFiniteLogit => scores[0].raw_logit = Some(f32::NAN),
                    RerankReply::Unavailable => panic!("unavailable model must not run"),
                }
                // The admission boundary must use original_rank, not output order.
                scores.reverse();
                Ok(scores)
            })
        }

        // `Reranker::id` declares `-> &str`; a trait impl cannot narrow that to
        // `&'static str`, which is what clippy's suggestion would do.
        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "native-test-cross-encoder"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn is_available(&self) -> bool {
            !matches!(self.reply, RerankReply::Unavailable)
        }
    }

    // Fed to with_reranker as `&dyn Fn(&str) -> Option<String>`, so the Option
    // is required by the callback type rather than incidental.
    #[allow(clippy::unnecessary_wraps)]
    fn source_text(id: &str) -> Option<String> {
        Some(format!("version-one:{id}"))
    }

    #[test]
    fn rerank_is_lazy_and_can_promote_a_candidate_outside_the_displayed_page() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&["z-fast", "middle", "near"]);
            let model = CrossEncoder::new(RerankReply::Winner);
            let text_calls = AtomicUsize::new(0);
            let text = |id: &str| {
                text_calls.fetch_add(1, Ordering::SeqCst);
                source_text(id)
            };
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &text, 3)
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            assert_eq!(results[0].doc_id, "z-fast");
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
            assert_eq!(text_calls.load(Ordering::SeqCst), 0);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            let State::Rerank(pending) = &stream.state else {
                panic!("retained rerank pool")
            };
            let original = pending
                .results
                .iter()
                .find(|hit| hit.doc_id == "near")
                .unwrap()
                .clone();
            let NativeSearchPhase::Reranked {
                results, evaluated, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("reranked")
            };
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "near");
            assert_eq!(results[0].rerank_score, Some(0.9));
            assert_eq!(results[0].source, ScoreSource::Reranked);
            assert_eq!(results[0].score.to_bits(), original.score.to_bits());
            assert_eq!(results[0].fast_score, original.fast_score);
            assert_eq!(results[0].index, original.index);
            assert!(Arc::ptr_eq(
                results[0].metadata.as_ref().unwrap(),
                original.metadata.as_ref().unwrap()
            ));
            assert_eq!(evaluated, 3);
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
            assert_eq!(text_calls.load(Ordering::SeqCst), 3);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.next_phase().await.unwrap().is_none());
        });
    }

    #[test]
    fn rerank_follows_refinement_without_requerying_or_changing_the_snapshot() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let quality = Provider::new("quality", 3);
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&["z-fast", "middle", "near", "a-quality"]);
            let model = CrossEncoder::new(RerankReply::Winner);
            let mut stream = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap()
                .with_reranker(&model, &source_text, 4)
                .unwrap();
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(matches!(
                stream.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Refined { .. })
            ));
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            lexical.current_generation.store(42, Ordering::SeqCst);
            let NativeSearchPhase::Reranked {
                results, evaluated, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("rerank after refinement")
            };
            assert_eq!(results[0].doc_id, "near");
            assert!(results[0].fast_score.is_some() && results[0].quality_score.is_some());
            assert_eq!(results[0].metadata.as_deref().unwrap()["generation"], 1);
            assert_eq!(evaluated, 4);
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
        });
    }

    #[test]
    fn failed_quality_still_allows_explicit_reranking_of_the_valid_initial_pool() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let mut quality = Provider::new("quality", 3);
            quality.reply = Reply::Failed;
            let index = fast_index(&cx, &fast);
            let qindex = quality_index(&cx, &quality);
            let lexical = Lexical::new(&[]);
            let model = CrossEncoder::new(RerankReply::Winner);
            let mut stream = index
                .search_hybrid_progressive(
                    &cx,
                    &fast,
                    Some((&qindex, &quality)),
                    &lexical,
                    "query",
                    1,
                )
                .unwrap()
                .with_reranker(&model, &source_text, 3)
                .unwrap();
            let NativeSearchPhase::Initial { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("initial")
            };
            let expected = serde_json::to_value(results).unwrap();
            let NativeSearchPhase::RefinementFailed {
                initial_results, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("failure must be exposed")
            };
            assert_eq!(serde_json::to_value(initial_results).unwrap(), expected);
            let NativeSearchPhase::Reranked { results, .. } =
                stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("independent final stage")
            };
            assert_eq!(results[0].doc_id, "near");
            assert!(results[0].quality_score.is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn malformed_rerank_responses_never_publish_partial_or_cross_document_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            for reply in [
                RerankReply::DuplicateRank,
                RerankReply::ForeignId,
                RerankReply::OutOfRange,
                RerankReply::MissingScore,
                RerankReply::NonFinite,
                RerankReply::NonFiniteLogit,
                RerankReply::Failed,
            ] {
                let lexical = Lexical::new(&["z-fast", "middle", "near"]);
                let model = CrossEncoder::new(reply);
                let mut stream = index
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 2)
                    .unwrap()
                    .with_reranker(&model, &source_text, 3)
                    .unwrap();
                let NativeSearchPhase::Initial { results, .. } =
                    stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("initial")
                };
                let metadata = Arc::clone(results[0].metadata.as_ref().unwrap());
                let expected = serde_json::to_value(results).unwrap();
                let NativeSearchPhase::RerankFailed {
                    previous_results, ..
                } = stream.next_phase().await.unwrap().unwrap()
                else {
                    panic!("must fail atomically")
                };
                assert!(Arc::ptr_eq(
                    previous_results[0].metadata.as_ref().unwrap(),
                    &metadata
                ));
                assert_eq!(serde_json::to_value(previous_results).unwrap(), expected);
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
                assert!(stream.next_phase().await.unwrap().is_none());
                assert_eq!(model.calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn pending_rerank_drop_releases_provider_and_snapshot_without_retrying() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&["z-fast"]);
            let model = CrossEncoder::new(RerankReply::Pending);
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &source_text, 3)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            drop(stream.next_phase()); // An unpolled final future is still inert.
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
            assert!(!stream.is_finished());
            let mut future = Box::pin(stream.next_phase());
            assert!(matches!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            assert_eq!(Arc::strong_count(&lexical.snapshot), 2);
            drop(future);
            assert_eq!(model.drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
            assert!(stream.is_finished());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn reranker_cancellation_is_not_a_successful_fallback_phase() {
        for reply in [RerankReply::Cancelled, RerankReply::CancelWithScores] {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let fast = Provider::new("fast", 2);
                let index = fast_index(&cx, &fast);
                let lexical = Lexical::new(&["z-fast"]);
                let model = CrossEncoder::new(reply);
                let mut stream = index
                    .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                    .unwrap()
                    .with_reranker(&model, &source_text, 3)
                    .unwrap();
                assert!(stream.next_phase().await.unwrap().is_some());
                assert!(matches!(
                    stream.next_phase().await,
                    Err(SearchError::Cancelled { .. })
                ));
                assert!(stream.is_finished());
                assert_eq!(Arc::strong_count(&lexical.snapshot), 1);
                assert!(stream.next_phase().await.unwrap().is_none());
            });
        }
    }

    #[test]
    fn missing_text_is_skipped_without_false_scores_or_a_false_reranked_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let model = CrossEncoder::new(RerankReply::Winner);
            let only_near = |id: &str| (id == "near").then(|| format!("version-one:{id}"));
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 4)
                .unwrap()
                .with_reranker(&model, &only_near, 3)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            let NativeSearchPhase::Reranked {
                results, evaluated, ..
            } = stream.next_phase().await.unwrap().unwrap()
            else {
                panic!("one scored document")
            };
            assert_eq!(evaluated, 1);
            assert_eq!(
                results
                    .iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["near", "z-fast", "middle", "a-quality"]
            );
            assert!(
                results[1..]
                    .iter()
                    .all(|hit| hit.rerank_score.is_none() && hit.source != ScoreSource::Reranked)
            );
            let no_text = |_: &str| None;
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &no_text, 3)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(stream.next_phase().await.unwrap().is_none());
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn rerank_configuration_and_zero_result_requests_start_no_model_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", 2);
            let index = fast_index(&cx, &fast);
            let lexical = Lexical::new(&[]);
            let model = CrossEncoder::new(RerankReply::Unavailable);
            for budget in [0, usize::MAX] {
                assert!(
                    index
                        .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                        .unwrap()
                        .with_reranker(&model, &source_text, budget)
                        .is_err()
                );
            }
            let mut empty = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 0)
                .unwrap()
                .with_reranker(&model, &source_text, 3)
                .unwrap();
            assert!(matches!(
                empty.next_phase().await.unwrap(),
                Some(NativeSearchPhase::Initial { .. })
            ));
            assert!(empty.next_phase().await.unwrap().is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            let mut stream = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap();
            assert!(stream.next_phase().await.unwrap().is_some());
            assert!(stream.with_reranker(&model, &source_text, 3).is_err());
            let mut unavailable = index
                .search_hybrid_progressive(&cx, &fast, None, &lexical, "query", 1)
                .unwrap()
                .with_reranker(&model, &source_text, 3)
                .unwrap();
            assert!(unavailable.next_phase().await.unwrap().is_some());
            assert!(matches!(
                unavailable.next_phase().await.unwrap(),
                Some(NativeSearchPhase::RerankFailed {
                    error: SearchError::RerankerUnavailable { .. },
                    ..
                })
            ));
            assert_eq!(model.calls.load(Ordering::SeqCst), 0);
        });
    }
}
