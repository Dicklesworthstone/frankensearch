//! Synchronous two-tier search orchestration for low-latency UIs.
//!
//! [`SyncTwoTierSearcher`] mirrors the progressive two-phase contract of
//! [`crate::searcher::TwoTierSearcher`] but operates on precomputed query
//! embeddings and fully in-memory indices.

use std::collections::hash_map::Entry;
use std::path::Path;

// The per-query `&str`-keyed score maps + `seen` dedup set are `.get()`/`.insert()`
// probed only (never iterated for output), so `ahash` is bit-identical to std and
// ~2× faster than SipHash on short doc_ids (`sync_hash_ab` bench: 0.44–0.51 across
// n=30..300), matching the sibling fusion paths (`rrf.rs`, `blend.rs`).
use ahash::{AHashMap, AHashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use frankensearch_core::explanation::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
};
use frankensearch_core::filter::{DocIdHashSet, IncludeDocIdsFilter, SearchFilter, fnv1a_hash};
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::types::{BoundQueryEmbedding, TieredQueryEmbeddings};
use frankensearch_core::{
    DocId, FusedHit, PhaseMetrics, ScoreSource, ScoredResult, SearchError, SearchPhase,
    SearchResult, SpaceIdentityAdmission, TwoTierConfig, TwoTierMetrics, VectorHit,
    ZeroSignalReason,
};
use frankensearch_index::{
    ClassifiedHits, InMemoryTwoTierIndex, InMemoryVectorIndex, SearchParams, ValidatedFsviBytes,
};

use crate::blend::{
    BlendExplanation, blend_two_tier, blend_two_tier_aligned_vector_index,
    complete_quality_evidence, compute_rank_changes, missing_candidate_ids,
};
use crate::mmr::{MmrConfig, mmr_rerank};
use crate::normalize::{AdaptiveNqcDenseWeight, NqcDenseWeight, nqc_cv_iter};
use crate::rrf::{
    FusionExplanation, RrfConfig, RrfTiebreak, candidate_count, fuse_by_strategy,
    fuse_by_strategy_for_vector_lane,
};
use crate::searcher::promoted_quality_source;

#[cfg(feature = "rerank")]
use frankensearch_core::traits::SyncRerank;

#[derive(Clone, Copy)]
struct SyncSearchContext<'a> {
    filter: Option<&'a dyn SearchFilter>,
    cx: Option<&'a asupersync::Cx>,
    rerank_window: Option<usize>,
    #[cfg(feature = "rerank")]
    rerank: Option<SyncRerankInput<'a>>,
}

#[cfg(feature = "rerank")]
#[derive(Clone, Copy)]
struct SyncRerankInput<'a> {
    text: &'a str,
    text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
}

impl<'a> SyncSearchContext<'a> {
    const fn new(filter: Option<&'a dyn SearchFilter>) -> Self {
        Self {
            filter,
            cx: None,
            rerank_window: None,
            #[cfg(feature = "rerank")]
            rerank: None,
        }
    }

    fn checkpoint(self, phase: &str) -> SearchResult<()> {
        if let Some(cx) = self.cx {
            let checkpoint = cx.checkpoint();
            if checkpoint.is_err() || cx.is_cancel_requested() {
                return Err(SearchError::Cancelled {
                    phase: phase.to_owned(),
                    reason: cx.cancel_reason().map_or_else(
                        || "synchronous search cancelled".to_owned(),
                        |reason| reason.to_string(),
                    ),
                });
            }
        }
        Ok(())
    }
}

/// The per-tier query embeddings this search is authorized to read with
/// (bd-sync-searcher-tiered-query-embeddings-dbp10).
///
/// Produced by [`SyncTwoTierSearcher::admit`] BEFORE any vector is touched.
/// Every admitted tier must join the query's space and attested producer.
/// Legacy and caller-declared tiers retain their explicitly weaker identity
/// contract; they do not acquire owner-backed authority by being loaded here.
enum AdmittedSyncQuery<'query> {
    /// Fast results followed by independently bound quality when requested.
    Progressive {
        fast: &'query BoundQueryEmbedding,
        quality: Option<&'query BoundQueryEmbedding>,
        hash_control: bool,
    },
    /// Direct quality retrieval, with no fast query or fast-vector access.
    QualityOnly {
        quality: &'query BoundQueryEmbedding,
    },
}

/// What the quality tier contributed to a synchronous refinement.
///
/// Same distinction the async searcher draws: `Retrieved` is the quality
/// tier's OWN top-k and can name documents the fast tier never selected,
/// supplemented with exact quality scores for retained fast candidates;
/// `RescoredFastPool` can only re-rank what the fast tier already chose.
/// Independent retrieval requires an admitted v2 owner or a persisted v1
/// producer fingerprint joined to the bound query. A caller-declared space
/// alone does not establish that producer contract.
enum SyncQualityPool {
    /// Independent quality retrieval plus available retained-candidate evidence.
    Retrieved(Vec<VectorHit>),
    /// Quality scores aligned positionally to the fast pool. Still computed
    /// from the QUALITY-bound query: the aligned SHAPE was never the bug, the
    /// vector fed into it was.
    RescoredFastPool(Vec<Option<f32>>),
}

/// Join a query against the identity retained from its tier's admitted owner
/// before reading vectors. As on persistent activation, producer conformance
/// permits search only for `SameProducer`; compatible foreign producers are
/// comparison telemetry and cannot authorize retrieval.
///
/// A caller-declared space has no admitted producer to check, so it keeps its
/// space-only join. Persisted v1 revisions bind the complete query identity,
/// as in the async searcher; an unstamped file cannot admit a semantic query.
/// Bare caller-built vectors retain their explicitly unidentified contract.
fn admit_tier_query(
    tier_index: &InMemoryVectorIndex,
    query: &BoundQueryEmbedding,
    tier: &str,
) -> SearchResult<()> {
    if let Some(binding) = tier_index.admitted_source_binding() {
        match query.verify_producer_conformance(&binding.frozen_identity().identity, tier)? {
            SpaceIdentityAdmission::SameProducer => {}
            other => {
                return Err(SearchError::InvalidConfig {
                    field: format!("search_activation.{tier}.producer_conformance"),
                    value: format!("{other:?}"),
                    reason: format!(
                        "the query's producer is not the {tier} index's attested producer; a \
                         certified-compatible pairing is comparison-grade telemetry and never an \
                         admission basis"
                    ),
                });
            }
        }
    } else if let Some(expected) = tier_index.space_fingerprint_hex() {
        query.verify_space_identity(expected, tier)?;
    } else if let Some(revision) = tier_index.embedder_revision()
        && (!revision.is_empty() || query.identity().space.kind == EmbeddingSpaceKindV1::Semantic)
        && revision != query.identity_fingerprint()
    {
        return Err(SearchError::InvalidConfig {
            field: format!("search_activation.{tier}.producer_revision"),
            value: "incompatible_or_missing".to_owned(),
            reason: format!(
                "the stored {tier} vectors do not bind this query's complete producer \
                 identity; rebuild the index with this producer before searching"
            ),
        });
    }
    // Identity refusals take precedence, but every tier must also admit the
    // read's shape. Legacy tiers cannot infer this from a retained identity,
    // and zero-vector or zero-k shortcuts must not bypass the check.
    let expected = tier_index.dimension();
    let found = query.vector().len();
    if found != expected {
        return Err(SearchError::DimensionMismatch { expected, found });
    }
    Ok(())
}

fn independently_retrievable_quality(index: &InMemoryVectorIndex) -> bool {
    index.space_identity_is_attested()
        || (index.space_fingerprint_hex().is_none()
            && index
                .embedder_revision()
                .is_some_and(|revision| !revision.is_empty()))
}

/// Optional synchronous lexical backend used by [`SyncTwoTierSearcher`].
pub trait SyncLexicalSearch: Send + Sync {
    /// Retrieve lexical candidates for the current query.
    ///
    /// Implementations may ignore `query_vec` when they already have external
    /// query context. Return at most `limit` candidates in relevance order;
    /// fewer candidates means the matching corpus is exhausted. The searcher
    /// counts each document ID once and may repeat this call with a larger
    /// limit to fill the eligible candidate budget.
    ///
    /// # Errors
    ///
    /// Returns backend-specific lexical retrieval errors.
    fn search_sync(&self, query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>>;

    /// Retain one reader for a filtered progressive search, when supported.
    ///
    /// The returned source must keep candidate refills and metadata lookups on
    /// one publication until it is dropped, including between iterator phases.
    /// `None` keeps this source's existing per-call retrieval contract.
    ///
    /// # Errors
    /// Returns snapshot-admission or cancellation failures before retrieval.
    fn pin_search(&self) -> SearchResult<Option<Arc<dyn SyncLexicalSearch>>> {
        Ok(None)
    }

    /// Read metadata for selected candidates from the retained reader.
    ///
    /// Results must follow the input order. A source without selected-document
    /// hydration returns `None` entries; absent documents and genuinely absent
    /// metadata also remain unknown. Filters receive that `None` unchanged.
    /// Implementations supplying metadata must bind it to the publication used
    /// by [`Self::search_sync`], normally through [`Self::pin_search`].
    ///
    /// # Errors
    /// Returns hydration failures. They are terminal for filtering, since an
    /// unreadable observation cannot be treated as permissive missing metadata.
    fn metadata_for_documents(
        &self,
        document_ids: &[&str],
    ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
        Ok(vec![None; document_ids.len()])
    }
}

/// Per-query synchronous lexical adapter for Quill.
///
/// [`SyncLexicalSearch`] receives only the semantic query vector, so the text
/// query and its structured-concurrency context are carried explicitly by this
/// adapter. Quill's reader path is synchronous and lock-free; `search_sync`
/// therefore never creates a runtime or blocks on an async operation. The
/// adapter accepts only a [`frankensearch_quill::QuillSearchIndex`], so a
/// synchronous search consumer cannot retain Quill's writer lease or mutation
/// surface.
#[cfg(feature = "quill")]
#[derive(Clone)]
pub struct QuillSyncLexicalSearch {
    index: Arc<frankensearch_quill::QuillSearchIndex>,
    cx: asupersync::Cx,
    query: Arc<str>,
}

#[cfg(feature = "quill")]
impl QuillSyncLexicalSearch {
    /// Bind a Quill index to one consumer-owned request context and text query.
    #[must_use]
    pub fn new(
        index: Arc<frankensearch_quill::QuillSearchIndex>,
        cx: asupersync::Cx,
        query: impl Into<Arc<str>>,
    ) -> Self {
        Self {
            index,
            cx,
            query: query.into(),
        }
    }

    /// Borrow the text query paired with this adapter.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }
}

#[cfg(feature = "quill")]
impl SyncLexicalSearch for QuillSyncLexicalSearch {
    fn search_sync(&self, _query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
        let query =
            frankensearch_core::traits::LexicalRead::query_text(self.index.as_ref(), &self.query);
        self.index
            .search_results(&self.cx, &query, limit)
            .map_err(SearchError::from)
    }

    fn pin_search(&self) -> SearchResult<Option<Arc<dyn SyncLexicalSearch>>> {
        let query =
            frankensearch_core::traits::LexicalRead::query_text(self.index.as_ref(), &self.query);
        Ok(Some(Arc::new(PinnedQuillSyncLexicalSearch {
            index: self.index.pin_search(&self.cx)?,
            cx: self.cx.clone(),
            query: query.into_owned(),
        })))
    }
}

#[cfg(feature = "quill")]
struct PinnedQuillSyncLexicalSearch {
    index: frankensearch_quill::QuillSearchSession,
    cx: asupersync::Cx,
    query: String,
}

#[cfg(feature = "quill")]
impl SyncLexicalSearch for PinnedQuillSyncLexicalSearch {
    fn search_sync(&self, _query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
        self.index
            .search_results(&self.cx, &self.query, limit)
            .map_err(SearchError::from)
    }

    fn metadata_for_documents(
        &self,
        document_ids: &[&str],
    ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
        self.index
            .metadata_for_documents(&self.cx, document_ids)
            .map_err(SearchError::from)
    }
}

/// Former enabled-path NQC shape retained for the same-binary allocation A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
#[allow(clippy::needless_collect)]
pub fn bench_nqc_cv_collect(lexical: &[ScoredResult]) -> f32 {
    let scores: Vec<f32> = lexical.iter().map(|hit| hit.score).collect();
    crate::normalize::nqc_cv(&scores)
}

/// Shipping enabled-path NQC shape retained for the same-binary allocation A/B.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_cv_iter(lexical: &[ScoredResult]) -> f32 {
    nqc_cv_iter(lexical.iter().map(|hit| hit.score))
}

/// 4-accumulator ILP reduction candidate for the enabled-path NQC compute. Reads `.score`
/// strided from the slice in lanes of four so the `sum`/`sum_sq` accumulations run as four
/// independent dependency chains instead of the single serial chain in [`bench_nqc_cv_iter`].
/// Assumes finite scores (the BM25 invariant — no per-element `is_finite` filter), so it is
/// quality-equivalent to `nqc_cv_iter` only up to f64 reassociation (~1e-13), NOT bit-identical.
/// Retained solely to A/B the reduction's compute; not a production path.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_cv_ilp(lexical: &[ScoredResult]) -> f32 {
    // Mirror of `normalize::NUMERIC_EPSILON` (private there); kept in sync with `nqc_cv_iter`.
    const NUMERIC_EPSILON: f32 = 1e-10;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut q0, mut q1, mut q2, mut q3) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (chunks, remainder) = lexical.as_chunks::<4>();
    for chunk in chunks {
        let a = f64::from(chunk[0].score);
        let b = f64::from(chunk[1].score);
        let c = f64::from(chunk[2].score);
        let d = f64::from(chunk[3].score);
        s0 += a;
        s1 += b;
        s2 += c;
        s3 += d;
        q0 += a * a;
        q1 += b * b;
        q2 += c * c;
        q3 += d * d;
    }
    let mut sum = s0 + s1 + s2 + s3;
    let mut sum_sq = q0 + q1 + q2 + q3;
    for hit in remainder {
        let v = f64::from(hit.score);
        sum += v;
        sum_sq += v * v;
    }
    let count = u32::try_from(lexical.len()).unwrap_or(u32::MAX);
    if count == 0 {
        return 0.0;
    }
    let n = f64::from(count);
    let mean = sum / n;
    if mean <= f64::from(NUMERIC_EPSILON) {
        return 0.0;
    }
    let second_moment = sum_sq / n;
    let variance = (second_moment - mean * mean).max(0.0);
    #[allow(clippy::cast_possible_truncation)]
    let cv = (variance.sqrt() / mean) as f32;
    cv
}

/// Enabled-but-empty NQC path before the neutral-sketch early return.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_empty_weight_orig(
    lexical: &[ScoredResult],
    weight: &NqcDenseWeight,
    beta: f32,
    w_min: f32,
    semantic_weight: f64,
) -> f64 {
    let cv = nqc_cv_iter(lexical.iter().map(|hit| hit.score));
    let factor = weight.dense_weight(cv, beta, w_min);
    semantic_weight * f64::from(factor)
}

/// Candidate neutral-sketch early return for the enabled NQC path.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn bench_nqc_empty_weight_early(
    lexical: &[ScoredResult],
    weight: &NqcDenseWeight,
    beta: f32,
    w_min: f32,
    semantic_weight: f64,
) -> f64 {
    if beta <= 0.0 {
        return semantic_weight;
    }
    let cv = if weight.is_empty() {
        0.0
    } else {
        nqc_cv_iter(lexical.iter().map(|hit| hit.score))
    };
    let factor = weight.dense_weight(cv, beta, w_min);
    semantic_weight * f64::from(factor)
}

/// Progressive synchronous searcher backed by [`InMemoryTwoTierIndex`].
#[derive(Clone, Copy, Debug, Default)]
enum SyncFastRetrieval {
    #[default]
    Exact,
    ApproximateInt8ForBench {
        candidate_multiplier: usize,
    },
}

pub struct SyncTwoTierSearcher {
    index: Arc<InMemoryTwoTierIndex>,
    lexical: Option<Arc<dyn SyncLexicalSearch>>,
    search_params: Option<SearchParams>,
    fast_retrieval: SyncFastRetrieval,
    config: TwoTierConfig,
    rrf_lexical_weight: f64,
    rrf_semantic_weight: f64,
    rrf_tiebreak: RrfTiebreak,
    /// Opt-in NQC dense down-weight (default off, `beta = 0.0`). See
    /// [`Self::with_nqc_dense_downweight`].
    nqc_downweight_beta: f32,
    nqc_downweight_w_min: f32,
    nqc_dense_weight: NqcDenseWeight,
    /// Self-driving rolling NQC dense down-weight — **on by default**
    /// ([`AdaptiveNqcDenseWeight::production_default`]). When set, it takes precedence over the
    /// static sketch above and learns the query distribution online. See
    /// [`Self::with_nqc_dense_downweight_adaptive`] / [`Self::with_nqc_dense_downweight_disabled`].
    /// Behind a `Mutex` so per-query observation mutates the rolling state while `search` stays
    /// `&self`.
    nqc_adaptive: Option<Mutex<AdaptiveNqcDenseWeight>>,
    mmr_config: MmrConfig,
    #[cfg(feature = "rerank")]
    reranker: Option<Arc<dyn SyncRerank>>,
    #[cfg(feature = "rerank")]
    rerank_window: usize,
    #[cfg(feature = "rerank")]
    rerank_combine: frankensearch_rerank::RerankCombine,
}

impl SyncTwoTierSearcher {
    /// Create a sync searcher over an in-memory two-tier index.
    ///
    /// The self-driving NQC dense down-weight is **on by default**
    /// ([`AdaptiveNqcDenseWeight::production_default`]) — neutral (byte-identical fusion) during
    /// its 128-query warm-up, then realizing the measured +0.0022 nDCG@10 gain. Opt out with
    /// [`Self::with_nqc_dense_downweight_disabled`]; an explicit
    /// [`Self::with_nqc_dense_downweight`] (static sketch) also overrides it. (No longer `const`
    /// — the default rolling sketch allocates.)
    #[must_use]
    pub fn new(index: Arc<InMemoryTwoTierIndex>, config: TwoTierConfig) -> Self {
        Self {
            index,
            lexical: None,
            search_params: None,
            fast_retrieval: SyncFastRetrieval::Exact,
            config,
            rrf_lexical_weight: 1.0,
            rrf_semantic_weight: 1.0,
            rrf_tiebreak: RrfTiebreak::LexicalThenId,
            nqc_downweight_beta: 0.0,
            nqc_downweight_w_min: 0.0,
            nqc_dense_weight: NqcDenseWeight::new(),
            nqc_adaptive: Some(Mutex::new(AdaptiveNqcDenseWeight::production_default())),
            mmr_config: MmrConfig::default(),
            #[cfg(feature = "rerank")]
            reranker: None,
            #[cfg(feature = "rerank")]
            rerank_window: 100,
            #[cfg(feature = "rerank")]
            rerank_combine: frankensearch_rerank::RerankCombine::default(),
        }
    }

    /// Construct the shipping synchronous search product directly from
    /// admitted FSVI-v2 owners and their generation-keyed residual cache
    /// directories. The in-memory constructor attaches only a bitwise
    /// source-derived sidecar; missing, stale, corrupt, or unavailable cache
    /// artifacts leave the existing exact flat route active for that tier.
    ///
    /// # Errors
    ///
    /// Returns source-vector loading errors. Optional residual-cache failures
    /// are contained by the in-memory exact flat fallback.
    pub fn from_admitted_v2_with_residual_sidecar_cache(
        fast_source: &ValidatedFsviBytes,
        fast_cache_dir: &Path,
        quality_source: Option<(&ValidatedFsviBytes, &Path)>,
        config: TwoTierConfig,
    ) -> SearchResult<Self> {
        let index = InMemoryTwoTierIndex::from_admitted_v2_with_residual_sidecar_cache(
            fast_source,
            fast_cache_dir,
            quality_source,
        )?;
        Ok(Self::new(Arc::new(index), config))
    }

    /// Attach an optional synchronous lexical source for RRF hybrid fusion.
    #[must_use]
    pub fn with_lexical(mut self, lexical: Arc<dyn SyncLexicalSearch>) -> Self {
        self.lexical = Some(lexical);
        self
    }

    /// Attach a cross-encoder for the text-aware synchronous entry points.
    ///
    /// Inference runs inline on the caller's thread only when final ranking is
    /// requested. Existing entry points without text keep their retrieval
    /// budget and never invoke this model.
    #[cfg(feature = "rerank")]
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn SyncRerank>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Set the cross-encoder candidate window, including hidden candidates.
    ///
    /// The default is 100. The text-aware search retains at least this many
    /// eligible candidates, subject to corpus size, and returns only the
    /// requested page after final ranking.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error unless the window is in 1..=1000.
    #[cfg(feature = "rerank")]
    pub fn with_rerank_window(mut self, window: usize) -> SearchResult<Self> {
        if !(1..=1_000).contains(&window) {
            return Err(SearchError::InvalidConfig {
                field: "rerank.window".to_owned(),
                value: window.to_string(),
                reason: "rerank window must be between 1 and 1000".to_owned(),
            });
        }
        self.rerank_window = window;
        Ok(self)
    }

    /// Choose how cross-encoder scores combine with retrieval order.
    ///
    /// Defaults to pure cross-encoder ordering. Reciprocal-rank combination
    /// preserves both signals and shares the asynchronous pipeline's complete
    /// response admission and deterministic tie handling.
    #[cfg(feature = "rerank")]
    #[must_use]
    pub const fn with_rerank_combine(
        mut self,
        combine: frankensearch_rerank::RerankCombine,
    ) -> Self {
        self.rerank_combine = combine;
        self
    }

    /// Override exact-scan parallelism for fast and quality retrieval, including
    /// progressive refinement, filtered refills, and missing candidate evidence.
    #[must_use]
    pub const fn with_search_params(mut self, params: SearchParams) -> Self {
        self.search_params = Some(params);
        self
    }

    /// Enable optional diversity ranking over the last successful candidate pool.
    ///
    /// The search retains up to `max(k, candidate_pool)` fused candidates so
    /// MMR can promote documents outside the displayed page. The broader
    /// retrieval window can also affect fusion in Initial and Refined.
    /// Iteration performs MMR only when the caller requests the final phase.
    /// Fast-only searches use fast vectors; successful refinement prefers a
    /// complete quality-space pool and otherwise uses a complete fast-space
    /// pool. Quality-only requests never consult fast vectors. If no single
    /// eligible tier covers every candidate, the preceding results are kept.
    ///
    /// Disabled MMR and pools of zero or one perform no additional work.
    /// Enabled pools above 1,000 are rejected before retrieval.
    #[must_use]
    pub const fn with_mmr_config(mut self, config: MmrConfig) -> Self {
        self.mmr_config = config;
        self
    }

    fn final_ranking_enabled(&self, context: SyncSearchContext<'_>) -> bool {
        context.rerank_window.is_some()
            || (self.mmr_config.enabled && self.mmr_config.candidate_pool > 1)
    }

    fn final_candidate_target(&self, k: usize, context: SyncSearchContext<'_>) -> usize {
        let target = k.max(context.rerank_window.unwrap_or(k));
        if self.mmr_config.enabled && self.mmr_config.candidate_pool > 1 {
            target.max(self.mmr_config.candidate_pool)
        } else {
            target
        }
    }

    fn retain_final_candidates(
        &self,
        results: &mut Vec<ScoredResult>,
        k: usize,
        space: SyncMmrSpace,
        context: SyncSearchContext<'_>,
    ) -> Option<SyncFinalRanking> {
        if !self.final_ranking_enabled(context)
            || results.is_empty()
            || (context.rerank_window.is_none() && results.len() < 2)
        {
            return None;
        }
        let candidates = std::mem::take(results);
        *results = candidates.iter().take(k).cloned().collect();
        Some(SyncFinalRanking {
            candidates,
            k,
            space,
        })
    }

    /// Select the approximate int8 candidate generator for an isolated
    /// regression test or benchmark only. The public product default remains
    /// exact regardless of sidecar availability; this explicit seam keeps
    /// approximate-vs-exact controls meaningful without restoring an
    /// approximate production route.
    #[doc(hidden)]
    #[must_use]
    pub const fn with_approximate_int8_fast_fetch_for_bench(
        mut self,
        candidate_multiplier: usize,
    ) -> Self {
        self.fast_retrieval = SyncFastRetrieval::ApproximateInt8ForBench {
            candidate_multiplier,
        };
        self
    }

    /// Set per-tier RRF fusion weights (default `1.0` / `1.0` = neutral).
    ///
    /// Up-weighting the *stronger* tier for the workload (~1.3×) makes the hybrid strictly
    /// dominate the best single tier (see `docs/NEGATIVE_EVIDENCE.md`). Non-finite or `≤ 0`
    /// values fall back to `1.0`.
    #[must_use]
    pub const fn with_rrf_weights(mut self, lexical_weight: f64, semantic_weight: f64) -> Self {
        self.rrf_lexical_weight = lexical_weight;
        self.rrf_semantic_weight = semantic_weight;
        self
    }

    /// Set the RRF tie-break strategy (default [`RrfTiebreak::LexicalThenId`]).
    ///
    /// [`RrfTiebreak::Hash`] breaks score ties by an unbiased hash of `doc_id` rather than
    /// favoring the lexical tier (see `docs/NEGATIVE_EVIDENCE.md`).
    #[must_use]
    pub const fn with_rrf_tiebreak(mut self, tiebreak: RrfTiebreak) -> Self {
        self.rrf_tiebreak = tiebreak;
        self
    }

    /// Enable the opt-in **NQC dense down-weight** (default OFF).
    ///
    /// Per query, the dense tier's fusion weight is scaled by
    /// `clip(1 − beta·CDF(nqc_cv(lexical scores)), w_min, 1)`, where `CDF` is the empirical
    /// percentile from `weight` — a [`NqcDenseWeight`] built offline from a sample of
    /// observed NQC values (the query stream). High lexical commitment (high NQC), where the
    /// dense tier tends to add little or hurt, gets a lower dense weight. Measured aggregate
    /// gain +0.0022 nDCG@10 (pooled 95% CI `[+0.0008, +0.0035]`); latency-neutral (the NQC is
    /// a single-pass reduction, only computed when enabled). See `docs/SEARCH_QUALITY_FINDINGS.md`.
    ///
    /// `beta <= 0` (the default) or an empty `weight` leaves fusion **byte-identical**.
    /// Use `w_min > 0` (e.g. the measured `beta ≈ 0.5` already floors the multiplier at
    /// `0.5`): a scaled semantic weight that reaches `<= 0` is treated as neutral `1.0` by the
    /// tier-weight sanitizer, which would *undo* the down-weight rather than maximize it.
    #[must_use]
    pub fn with_nqc_dense_downweight(
        mut self,
        beta: f32,
        w_min: f32,
        weight: NqcDenseWeight,
    ) -> Self {
        self.nqc_downweight_beta = beta;
        self.nqc_downweight_w_min = w_min;
        self.nqc_dense_weight = weight;
        // An explicit static sketch overrides the default-on adaptive down-weight (which would
        // otherwise take precedence in `effective_semantic_weight`).
        self.nqc_adaptive = None;
        self
    }

    /// Enable the **self-driving rolling** NQC dense down-weight (**on by default**; this rebuilds
    /// it with explicit parameters).
    ///
    /// Unlike [`Self::with_nqc_dense_downweight`] (a caller-supplied static sketch), this builds
    /// and refreshes the cv→percentile sketch *online* from the observed query stream via an
    /// [`AdaptiveNqcDenseWeight`] — no external sample management. It takes precedence over the
    /// static sketch when both are set. Cold start is **neutral** (empty sketch → weight `1.0`,
    /// byte-identical fusion) until `min_samples` queries warm it, so it is safe to enable by
    /// default. `beta <= 0` disables it. See [`AdaptiveNqcDenseWeight::new`] for the parameters.
    #[must_use]
    pub fn with_nqc_dense_downweight_adaptive(
        mut self,
        beta: f32,
        w_min: f32,
        capacity: usize,
        min_samples: usize,
        rebuild_every: usize,
    ) -> Self {
        self.nqc_adaptive = Some(Mutex::new(AdaptiveNqcDenseWeight::new(
            beta,
            w_min,
            capacity,
            min_samples,
            rebuild_every,
        )));
        self
    }

    /// Enable the rolling NQC dense down-weight with the blessed production defaults
    /// ([`AdaptiveNqcDenseWeight::production_default`]) — the recommended one-call setup.
    #[must_use]
    pub fn with_nqc_dense_downweight_adaptive_defaults(mut self) -> Self {
        self.nqc_adaptive = Some(Mutex::new(AdaptiveNqcDenseWeight::production_default()));
        self
    }

    /// Disable any NQC dense down-weight (static or adaptive) → **byte-identical** fusion.
    /// The opt-out / A-B escape hatch (also clears a default-on adaptive down-weight).
    #[must_use]
    pub fn with_nqc_dense_downweight_disabled(mut self) -> Self {
        self.nqc_adaptive = None;
        self.nqc_downweight_beta = 0.0;
        self
    }

    /// The dense-tier fusion weight for this query: the static `rrf_semantic_weight`, scaled
    /// by the per-query NQC dense down-weight when enabled (`beta > 0`). Off (default) returns
    /// `rrf_semantic_weight` unchanged with zero extra work.
    fn effective_semantic_weight(&self, lexical: &[ScoredResult]) -> f64 {
        // The self-driving rolling down-weight takes precedence: it observes this query's NQC
        // (warming the online sketch) and scores it against the sketch from prior queries.
        if let Some(adaptive) = &self.nqc_adaptive {
            let cv = nqc_cv_iter(lexical.iter().map(|hit| hit.score));
            let factor = adaptive
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .weight_for_cv(cv);
            return self.rrf_semantic_weight * f64::from(factor);
        }
        if self.nqc_downweight_beta <= 0.0 {
            return self.rrf_semantic_weight;
        }
        let cv = if self.nqc_dense_weight.is_empty() {
            0.0
        } else {
            nqc_cv_iter(lexical.iter().map(|hit| hit.score))
        };
        let factor = self.nqc_dense_weight.dense_weight(
            cv,
            self.nqc_downweight_beta,
            self.nqc_downweight_w_min,
        );
        self.rrf_semantic_weight * f64::from(factor)
    }

    fn search_optional_lexical(
        context: &mut SyncLexicalContext,
        query_vec: &[f32],
        limit: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Option<Vec<ScoredResult>>> {
        let Some(lexical) = &context.source else {
            return Ok(None);
        };
        match search_lexical_candidates(
            lexical.as_ref(),
            query_vec,
            limit,
            filter,
            &mut context.metadata,
            None,
        ) {
            Ok(results) => Ok(Some(results)),
            Err(
                SyncLexicalFailure::Metadata(error)
                | SyncLexicalFailure::Retrieval(error @ SearchError::Cancelled { .. }),
            ) => Err(error),
            Err(SyncLexicalFailure::Retrieval(error)) => {
                // Failed retrieval cannot contribute a partial lexical ranking.
                // A successful prefix can still supply metadata that rejected
                // a semantic candidate. Keep those observations binding for a
                // filtered fallback; an outage cannot make them unknown.
                tracing::warn!(
                    error_code = error.diagnostic_code(),
                    "lexical search failed, continuing without lexical"
                );
                if filter.is_none() {
                    context.metadata.clear();
                }
                Ok(None)
            }
        }
    }

    /// Execute a synchronous search and return the final result set + metrics.
    ///
    /// `query` carries one bound embedding PER TIER. The former signature took
    /// a single `&[f32]` and used it for both tiers, which meant the quality
    /// index was scored with the fast tier's query — a cross-space read that
    /// no dimension check can catch, because the two tiers usually have the
    /// same width. That vector is no longer representable here.
    /// A quality-only query directly retrieves the independently admitted quality
    /// tier without touching fast vectors. It can still fuse an attached lexical
    /// arm; that adapter receives the quality query vector. A quality-only request
    /// requires semantic quality with an admitted owner or persisted producer
    /// fingerprint, and is incompatible with `config.fast_only`.
    /// Failure of an optional lexical backend leaves semantic retrieval active,
    /// including quality refinement. Lexical cancellation still ends the request.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the requested tier is absent
    /// or unauthorized, or when a query fails its retained space and producer
    /// admission; plus dimension/filter errors from vector search and
    /// cancellation from an optional lexical backend, including metadata
    /// hydration during quality refinement. Cancellation never returns the
    /// initial ranking as a successful collected search. Other refinement errors
    /// also propagate when newly available metadata rejects a retained initial
    /// candidate, including candidates reserved for final diversity ranking.
    pub fn search_collect(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_with_filter(query, k, None)
    }

    /// Execute a synchronous search with an optional document filter.
    ///
    /// Available lexical metadata applies to every retrieval lane, including
    /// semantic-only candidates. A snapshot-capable lexical source is pinned
    /// once for both phases and every refill. Unknown metadata is still passed
    /// as `None` to the caller's predicate; hydration errors are not unknowns.
    /// Built-in metadata filters can opt into
    /// [`MissingMetadataPolicy::Reject`](frankensearch_core::filter::MissingMetadataPolicy::Reject)
    /// with `on_missing_metadata`. They then reject a document whose selected
    /// metadata is unavailable, including every vector-only candidate when no
    /// metadata source is attached. With a source, metadata is resolved before
    /// that rejection and selective filters refill the eligible result budget.
    /// When a later lookup resolves an unknown observation, both lexical and
    /// semantic candidate pools are revalidated and refilled before publishing.
    /// Predicates must remain stable for the same document and observation;
    /// repeated readmission and rejection returns a typed configuration error.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::search_collect`], plus an invalid-configuration error
    /// when a filter repeatedly changes eligibility during candidate refill.
    pub fn search_collect_with_filter(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_in_context(query, k, SyncSearchContext::new(filter))
    }

    /// Search and apply an optional synchronous cross-encoder to document text.
    ///
    /// The original text is used only by the cross-encoder; retrieval still
    /// admits each precomputed query embedding against its retained tier.
    /// Inference runs on the caller's thread without an async executor. The
    /// caller's context checks cancellation at phase and inference boundaries;
    /// it cannot preempt an already-running synchronous model.
    ///
    /// # Errors
    ///
    /// Returns the admission and retrieval errors of [`Self::search_collect`],
    /// plus typed cancellation. Missing document text, ordinary provider
    /// failures and malformed responses preserve the preceding eligible order.
    #[cfg(feature = "rerank")]
    pub fn search_collect_with_text(
        &self,
        cx: &asupersync::Cx,
        text: &str,
        query: &TieredQueryEmbeddings,
        k: usize,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_with_text_and_filter(cx, text, query, k, text_fn, None)
    }

    /// Search with text-aware final ranking and an optional document filter.
    ///
    /// Pinned metadata and filter admission match
    /// [`Self::search_collect_with_filter`]. Every retained candidate, including
    /// hidden candidates, is rechecked around text callbacks, model inference
    /// and final publication. Revocation refuses the whole final stage.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::search_collect_with_text`], plus a typed
    /// invalid `search.filter` configuration when retained eligibility changes.
    #[cfg(feature = "rerank")]
    pub fn search_collect_with_text_and_filter(
        &self,
        cx: &asupersync::Cx,
        text: &str,
        query: &TieredQueryEmbeddings,
        k: usize,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_in_context(query, k, self.text_context(cx, text, text_fn, filter))
    }

    #[cfg(feature = "rerank")]
    fn text_context<'a>(
        &self,
        cx: &'a asupersync::Cx,
        text: &'a str,
        text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
        filter: Option<&'a dyn SearchFilter>,
    ) -> SyncSearchContext<'a> {
        SyncSearchContext {
            filter,
            cx: Some(cx),
            rerank_window: self.reranker.as_ref().map(|_| self.rerank_window),
            rerank: Some(SyncRerankInput { text, text_fn }),
        }
    }

    fn search_collect_in_context(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
        context: SyncSearchContext<'_>,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        // Collected searches do not clone intermediate phase payloads.
        context.checkpoint("sync_search.initial")?;
        let prepared = self.search_initial(query, k, context, false);
        context.checkpoint("sync_search.initial")?;
        let prepared = prepared?;
        let outcome = if let Some(refinement) = prepared.refinement {
            let refined = self.search_refined(prepared.initial, refinement, context, false);
            context.checkpoint("sync_search.refinement")?;
            refined.map_err(|failure| failure.error)?
        } else {
            prepared.initial
        };
        let outcome = self.search_final(outcome, context, false);
        context.checkpoint("sync_search.final")?;
        let outcome = outcome?;
        Ok((outcome.final_results, outcome.metrics))
    }

    /// Execute a synchronous search and stream progressive phases via iterator.
    ///
    /// When admission or phase-1 retrieval fails (for example a query bound to
    /// a different embedding space than the index was built in, or a dimension
    /// mismatch), this returns an iterator yielding a single
    /// `RefinementFailed` phase carrying an empty `initial_results` payload.
    /// No retrieval runs until the first `next()`. That call admits every enabled
    /// query tier and returns Initial; quality retrieval runs only when the
    /// caller requests the next phase. Dropping the iterator discards pending
    /// refinement without starting it. Each call uses the caller's thread.
    /// A quality-only request instead returns an Initial containing
    /// quality results, labeled through their quality source, score and phase
    /// metrics. There is no fabricated fast phase or subsequent refinement.
    /// Opt-in MMR can append a lazy Reranked phase after the last successful
    /// retrieval. Dropping the iterator before that phase performs no MMR work.
    #[must_use]
    pub const fn search_iter<'a>(
        &'a self,
        query: &'a TieredQueryEmbeddings,
        k: usize,
    ) -> SyncSearchIterator<'a> {
        self.search_iter_with_filter(query, k, None)
    }

    /// Execute a synchronous filtered search and stream progressive phases.
    ///
    /// Metadata upgrades and stable predicate requirements match
    /// [`Self::search_collect_with_filter`]. Retained candidates are rechecked
    /// before quality refinement and after its retrieval, even without a final
    /// reranker. A policy revocation produces an empty terminal failure.
    /// A terminal refinement failure retains
    /// the previously emitted Initial only while its candidate pool stays
    /// eligible. If new metadata rejects any retained candidate, the failure
    /// payload is empty and no final diversity ranking runs. Final ranking
    /// rechecks all retained candidates when the iterator advances and before
    /// returning its output. A policy change yields an empty terminal
    /// `RefinementFailed` with an invalid `search.filter` configuration.
    #[must_use]
    pub const fn search_iter_with_filter<'a>(
        &'a self,
        query: &'a TieredQueryEmbeddings,
        k: usize,
        filter: Option<&'a dyn SearchFilter>,
    ) -> SyncSearchIterator<'a> {
        SyncSearchIterator {
            searcher: self,
            context: SyncSearchContext::new(filter),
            state: SyncIteratorState::Initial { query, k },
        }
    }

    /// Lazily search and rerank using the original query and document text.
    ///
    /// Text lookup and cross-encoder inference occur only when the caller
    /// advances beyond the last retrieval phase. Dropping the iterator before
    /// final ranking performs neither operation. The borrowed context supplies
    /// cancellation checks without requiring an async executor.
    #[cfg(feature = "rerank")]
    #[must_use]
    pub fn search_iter_with_text<'a>(
        &'a self,
        cx: &'a asupersync::Cx,
        text: &'a str,
        query: &'a TieredQueryEmbeddings,
        k: usize,
        text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
    ) -> SyncSearchIterator<'a> {
        self.search_iter_with_text_and_filter(cx, text, query, k, text_fn, None)
    }

    /// Lazily search with text-aware final ranking and pinned filter admission.
    ///
    /// A revoked retained candidate yields an empty terminal
    /// `RefinementFailed`, including revocation by a text callback before the
    /// document can reach inference. All work runs on the calling thread.
    #[cfg(feature = "rerank")]
    #[must_use]
    pub fn search_iter_with_text_and_filter<'a>(
        &'a self,
        cx: &'a asupersync::Cx,
        text: &'a str,
        query: &'a TieredQueryEmbeddings,
        k: usize,
        text_fn: &'a (dyn Fn(&str) -> Option<String> + Send + Sync),
        filter: Option<&'a dyn SearchFilter>,
    ) -> SyncSearchIterator<'a> {
        SyncSearchIterator {
            searcher: self,
            context: self.text_context(cx, text, text_fn, filter),
            state: SyncIteratorState::Initial { query, k },
        }
    }

    /// Join every tier this search will consult against the query bound to it,
    /// before any vector is read.
    ///
    /// This runs first in [`Self::search_initial`], ahead of the `k == 0`
    /// return and the zero-norm fast-lane check, so a query in the wrong space
    /// is refused even when its vector would never be searched.
    /// A refusal that depended on the search getting far enough to do work
    /// would not be a guard.
    fn admit<'query>(
        &self,
        query: &'query TieredQueryEmbeddings,
    ) -> SearchResult<AdmittedSyncQuery<'query>> {
        let Some(fast) = query.fast() else {
            let quality = query.quality().ok_or_else(|| SearchError::InvalidConfig {
                field: "sync_search.topology".to_owned(),
                value: "empty".to_owned(),
                reason: "at least one independently bound query tier is required".to_owned(),
            })?;
            let index = self
                .index
                .quality_index()
                .ok_or_else(|| SearchError::InvalidConfig {
                    field: "sync_search.topology".to_owned(),
                    value: "quality_only".to_owned(),
                    reason: "quality-only retrieval requires a quality index".to_owned(),
                })?;
            admit_tier_query(index, quality, "quality")?;
            if !independently_retrievable_quality(index) {
                return Err(SearchError::InvalidConfig {
                    field: "search_activation.quality.producer_identity".to_owned(),
                    value: "unidentified".to_owned(),
                    reason: "quality-only retrieval requires an admitted owner or a persisted \
                             complete producer fingerprint"
                        .to_owned(),
                });
            }
            if quality.identity().space.kind != EmbeddingSpaceKindV1::Semantic {
                return Err(SearchError::InvalidConfig {
                    field: "sync_search.quality_kind".to_owned(),
                    value: "non_semantic".to_owned(),
                    reason: "quality-only retrieval requires a semantic quality producer"
                        .to_owned(),
                });
            }
            if self.config.fast_only {
                return Err(SearchError::InvalidConfig {
                    field: "sync_search.topology".to_owned(),
                    value: "quality_only".to_owned(),
                    reason: "quality-only retrieval conflicts with the fast-only search policy"
                        .to_owned(),
                });
            }
            return Ok(AdmittedSyncQuery::QualityOnly { quality });
        };
        let fast_index = self.index.fast_index();
        admit_tier_query(fast_index, fast, "fast")?;
        let hash_control = admitted_tier_is_hash_control(fast_index, fast);

        let quality = if self.config.fast_only {
            // Disabled quality is outside this request's admission contract.
            // A stale precomputed quality embedding must not block valid fast
            // retrieval, just as an absent quality embedding does not.
            None
        } else {
            match (query.quality(), self.index.quality_index()) {
                (Some(bound), Some(quality_index)) => {
                    admit_tier_query(quality_index, bound, "quality")?;
                    if !hash_control && admitted_tier_is_hash_control(quality_index, bound) {
                        return Err(SearchError::InvalidConfig {
                            field: "sync_search.quality_kind".to_owned(),
                            value: "non_semantic".to_owned(),
                            reason: "progressive refinement requires a semantic quality producer"
                                .to_owned(),
                        });
                    }
                    Some(bound)
                }
                // A quality-bound query against an index with no quality tier is
                // not an error — it is the ordinary fast-only index — and a
                // quality tier the caller bound no embedding for is simply never
                // consulted. Neither case licenses reusing the fast embedding.
                _ => None,
            }
        };
        Ok(AdmittedSyncQuery::Progressive {
            fast,
            quality,
            hash_control,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn search_initial<'query>(
        &self,
        query: &'query TieredQueryEmbeddings,
        k: usize,
        context: SyncSearchContext<'_>,
        want_phase: bool,
    ) -> SearchResult<SyncPreparedSearch<'query>> {
        let filter = context.filter;
        // Identity first, before k == 0, before the zero-norm check, and
        // before any index is touched.
        let admitted = self.admit(query)?;
        if self.mmr_config.enabled && self.mmr_config.candidate_pool > 1_000 {
            return Err(SearchError::InvalidConfig {
                field: "mmr.candidate_pool".to_owned(),
                value: self.mmr_config.candidate_pool.to_string(),
                reason: "enabled MMR candidate pool must not exceed 1000".to_owned(),
            });
        }
        let (fast_query, quality_query, hash_control) = match admitted {
            AdmittedSyncQuery::Progressive {
                fast,
                quality,
                hash_control,
            } => (fast, quality, hash_control),
            AdmittedSyncQuery::QualityOnly { quality } => {
                return self.search_quality_only(quality, k, context, want_phase);
            }
        };
        let query_vec = fast_query.vector();
        let mut metrics = TwoTierMetrics::default();
        // Match the async progressive contract: zero requested results is a
        // successful no-op, not a request to scan the corpus and discard the
        // candidate pool afterwards (bd-k3089).
        if k == 0 {
            metrics.zero_signal = Some(ZeroSignalReason::CallerRequestedZeroK);
            return Ok(SyncPreparedSearch {
                initial: SyncSearchOutcome {
                    phase: None,
                    final_results: Vec::new(),
                    metrics,
                    final_ranking: None,
                },
                refinement: None,
            });
        }
        let target = self.final_candidate_target(k, context);
        let fetch = candidate_count(target, 0, self.config.candidate_multiplier.max(1)).max(target);

        let phase1_started = Instant::now();
        let mut lexical_context = SyncLexicalContext::new(self.lexical.clone(), filter.is_some())?;
        let lexical_started = Instant::now();
        let mut lexical_hits =
            Self::search_optional_lexical(&mut lexical_context, query_vec, fetch, filter)?;
        metrics.lexical_search_ms = ms(lexical_started.elapsed());
        let (mut classified, phase1_vectors_searched) =
            if query_vec.iter().all(|&value| value == 0.0) {
                // Zero signal belongs to this semantic lane (bd-tqhc), not the
                // whole query. Lexical matches and an independently admitted
                // quality query can still answer it without any fast-vector work.
                (ClassifiedHits::empty(ZeroSignalReason::ZeroNormQuery), 0)
            } else {
                // Count the vectors evaluated, not the bounded candidate pool.
                (
                    search_filtered_vectors(
                        &mut lexical_context,
                        fetch,
                        self.index.fast_index().record_count(),
                        filter,
                        |limit, candidates| self.search_fast_hits(query_vec, limit, candidates),
                    )?,
                    self.index.doc_count(),
                )
            };
        let (lexical_refill_time, _) = reconcile_filtered_vectors(
            &mut lexical_context,
            query_vec,
            lexical_hits.as_mut(),
            &mut classified,
            fetch,
            self.index.fast_index().record_count(),
            filter,
            &mut AHashSet::new(),
            |limit, candidates| self.search_fast_hits(query_vec, limit, candidates),
        )?;
        metrics.lexical_search_ms += ms(lexical_refill_time);
        let ClassifiedHits {
            hits: fast_hits,
            zero_signal,
        } = classified;
        metrics.lexical_candidates = lexical_hits.as_ref().map_or(0, Vec::len);
        metrics.phase1_vectors_searched = phase1_vectors_searched;
        metrics.zero_signal = zero_signal;
        metrics.publish_vector_candidates(fast_hits.len(), hash_control);
        if hash_control && metrics.skip_reason.is_none() {
            // Same contract as the async searcher: hash vectors still run as
            // an explicit control path, but they are never unlabeled semantic
            // hits and they do not schedule quality refinement.
            metrics.skip_reason = Some("non_semantic_fast_embedder_vector_control".to_owned());
        }
        let rrf_started = Instant::now();
        let query_semantic_weight = lexical_hits
            .as_ref()
            .map_or(self.rrf_semantic_weight, |lexical| {
                self.effective_semantic_weight(lexical)
            });
        let mut initial_results = lexical_hits.as_ref().map_or_else(
            || {
                vector_hits_to_scored_results(
                    &fast_hits,
                    target,
                    if hash_control {
                        ScoreSource::HashControl
                    } else {
                        ScoreSource::SemanticFast
                    },
                    None,
                    None,
                    &self.config,
                    sync_fast_embedder_id(&self.index),
                )
            },
            |lexical| {
                let rrf_config = RrfConfig {
                    k: self.config.rrf_k,
                    lexical_weight: self.rrf_lexical_weight,
                    semantic_weight: query_semantic_weight,
                    tiebreak: self.rrf_tiebreak,
                };
                fused_hits_to_scored_results(
                    fuse_by_strategy_for_vector_lane(
                        self.config.fusion_strategy,
                        lexical,
                        &fast_hits,
                        &[],
                        0.0,
                        target,
                        0,
                        &rrf_config,
                        hash_control,
                    ),
                    lexical,
                    target,
                    self.config.explain.then(|| {
                        FusionExplanation::new(
                            self.config.fusion_strategy,
                            lexical,
                            &fast_hits,
                            &[],
                            0.0,
                            &rrf_config,
                        )
                    }),
                    sync_fast_embedder_id(&self.index),
                    hash_control,
                )
            },
        );
        lexical_context.attach_metadata(&mut initial_results);
        metrics.rrf_fusion_ms = ms(rrf_started.elapsed());

        let phase1_latency = phase1_started.elapsed();
        metrics.vector_search_ms = ms(phase1_latency);
        metrics.phase1_total_ms = ms(phase1_latency);
        metrics.fast_embed_ms = 0.0;

        let final_ranking =
            self.retain_final_candidates(&mut initial_results, k, SyncMmrSpace::Fast, context);

        let phase = want_phase.then(|| SearchPhase::Initial {
            results: initial_results.clone(),
            latency: phase1_latency,
            metrics: PhaseMetrics {
                embedder_id: "sync-fast-query".to_owned(),
                vectors_searched: phase1_vectors_searched,
                lexical_candidates: metrics.lexical_candidates,
                fused_count: initial_results.len(),
                skip_reason: metrics.skip_reason.clone(),
                hash_control_candidates: metrics.hash_control_candidates,
            },
        });
        let mut initial = SyncSearchOutcome {
            phase,
            final_results: initial_results,
            metrics,
            final_ranking,
        };

        // The quality arm needs four things: it must not be switched off, the
        // index must have a quality tier, the caller must have bound a
        // QUALITY-space embedding for it, and the fast lane must be semantic.
        // A hash-control fast lane is a control path, not a first semantic
        // pass, so a bound quality embedding does not schedule refinement.
        let quality_query = quality_query.filter(|_| !self.config.fast_only && !hash_control);
        let Some(quality_query) = quality_query else {
            // Same vocabulary as the async searcher (searcher.rs) — the two
            // sides share one skip_reason contract; "fast_only" is the string
            // the fsfs surfaces document (bd-k3089 parity suite pins this).
            if initial.metrics.skip_reason.is_none() {
                initial.metrics.skip_reason = Some(if self.config.fast_only {
                    "fast_only".to_owned()
                } else if self.index.has_quality_index() {
                    "quality_query_embedding_absent".to_owned()
                } else {
                    "quality_index_unavailable".to_owned()
                });
            }
            return Ok(SyncPreparedSearch {
                initial,
                refinement: None,
            });
        };
        if quality_query.vector().iter().all(|&value| value == 0.0) {
            // A zero query has no quality evidence. Scanning or rescoring it
            // would turn arbitrary zero-score ties into semantic results and
            // could displace the useful fast or lexical answer already found.
            initial.metrics.skip_reason = Some("quality_query_zero_norm".to_owned());
            return Ok(SyncPreparedSearch {
                initial,
                refinement: None,
            });
        }
        Ok(SyncPreparedSearch {
            initial,
            refinement: Some(SyncRefinement {
                fast_query,
                quality_query,
                fast_hits,
                lexical_hits,
                lexical_context,
                query_semantic_weight,
                fetch,
                k,
            }),
        })
    }

    fn search_quality_only<'query>(
        &self,
        query: &'query BoundQueryEmbedding,
        k: usize,
        context: SyncSearchContext<'_>,
        want_phase: bool,
    ) -> SearchResult<SyncPreparedSearch<'query>> {
        let filter = context.filter;
        // Admission has already joined this query to the quality producer.
        // Nothing here consults fast identity, vectors, ordinals or ranking.
        let index = self
            .index
            .quality_index()
            .expect("quality-only admission requires the quality tier");
        let embedder_id = index.embedder_id().unwrap_or("sync-quality-query");
        let mut metrics = TwoTierMetrics {
            quality_embedder_id: Some(embedder_id.to_owned()),
            ..TwoTierMetrics::default()
        };
        if k == 0 {
            metrics.zero_signal = Some(ZeroSignalReason::CallerRequestedZeroK);
            return Ok(SyncPreparedSearch {
                initial: SyncSearchOutcome {
                    phase: None,
                    final_results: Vec::new(),
                    metrics,
                    final_ranking: None,
                },
                refinement: None,
            });
        }
        let started = Instant::now();
        let target = self.final_candidate_target(k, context);
        let fetch = candidate_count(target, 0, self.config.candidate_multiplier.max(1)).max(target);
        let mut lexical_context = SyncLexicalContext::new(self.lexical.clone(), filter.is_some())?;
        let lexical_started = Instant::now();
        let mut lexical_hits =
            Self::search_optional_lexical(&mut lexical_context, query.vector(), fetch, filter)?;
        metrics.lexical_search_ms = ms(lexical_started.elapsed());

        let quality_started = Instant::now();
        let mut classified = search_filtered_vectors(
            &mut lexical_context,
            fetch,
            index.record_count(),
            filter,
            |limit, candidates| {
                index.search_top_k_classified_with_params(
                    query.vector(),
                    limit,
                    candidates,
                    self.search_params.unwrap_or_default(),
                )
            },
        )?;
        let (lexical_refill_time, _) = reconcile_filtered_vectors(
            &mut lexical_context,
            query.vector(),
            lexical_hits.as_mut(),
            &mut classified,
            fetch,
            index.record_count(),
            filter,
            &mut AHashSet::new(),
            |limit, candidates| {
                index.search_top_k_classified_with_params(
                    query.vector(),
                    limit,
                    candidates,
                    self.search_params.unwrap_or_default(),
                )
            },
        )?;
        metrics.quality_search_ms = ms(quality_started
            .elapsed()
            .saturating_sub(lexical_refill_time));
        metrics.lexical_search_ms += ms(lexical_refill_time);
        let ClassifiedHits { hits, zero_signal } = classified;
        metrics.lexical_candidates = lexical_hits.as_ref().map_or(0, Vec::len);
        if zero_signal != Some(ZeroSignalReason::ZeroNormQuery) {
            // These are Phase-1 vectors because direct quality is the first
            // and only phase. The embedder and quality timing identify its tier.
            metrics.phase1_vectors_searched = index.record_count();
        }
        metrics.zero_signal = zero_signal;
        metrics.vector_search_ms = metrics.quality_search_ms;
        metrics.semantic_candidates = hits.len();
        let fusion_started = Instant::now();
        let mut results =
            self.quality_only_results(&hits, lexical_hits.as_deref(), target, embedder_id);
        lexical_context.attach_metadata(&mut results);
        metrics.rrf_fusion_ms = ms(fusion_started.elapsed());
        let latency = started.elapsed();
        metrics.phase1_total_ms = ms(latency);
        let final_ranking =
            self.retain_final_candidates(&mut results, k, SyncMmrSpace::QualityOnly, context);
        let phase = want_phase.then(|| SearchPhase::Initial {
            results: results.clone(),
            latency,
            metrics: PhaseMetrics {
                embedder_id: embedder_id.to_owned(),
                vectors_searched: metrics.phase1_vectors_searched,
                lexical_candidates: metrics.lexical_candidates,
                fused_count: results.len(),
                skip_reason: metrics
                    .zero_signal
                    .map(|reason| reason.reason_code().to_owned()),
                hash_control_candidates: 0,
            },
        });
        Ok(SyncPreparedSearch {
            initial: SyncSearchOutcome {
                phase,
                final_results: results,
                metrics,
                final_ranking,
            },
            refinement: None,
        })
    }

    fn quality_only_results(
        &self,
        hits: &[VectorHit],
        lexical: Option<&[ScoredResult]>,
        k: usize,
        embedder_id: &str,
    ) -> Vec<ScoredResult> {
        let mut results = lexical.map_or_else(
            || {
                vector_hits_to_scored_results(
                    hits,
                    k,
                    ScoreSource::SemanticQuality,
                    None,
                    None,
                    &self.config,
                    embedder_id,
                )
            },
            |lexical| {
                let rrf_config = RrfConfig {
                    k: self.config.rrf_k,
                    lexical_weight: self.rrf_lexical_weight,
                    semantic_weight: self.effective_semantic_weight(lexical),
                    tiebreak: self.rrf_tiebreak,
                };
                fused_hits_to_scored_results(
                    fuse_by_strategy(
                        self.config.fusion_strategy,
                        lexical,
                        hits,
                        &[],
                        0.0,
                        k,
                        0,
                        &rrf_config,
                    ),
                    lexical,
                    k,
                    self.config.explain.then(|| {
                        FusionExplanation::new(
                            self.config.fusion_strategy,
                            lexical,
                            hits,
                            &[],
                            0.0,
                            &rrf_config,
                        )
                    }),
                    embedder_id,
                    false,
                )
            },
        );
        // The shared converters describe their primary vector input as fast.
        // This input was exclusively quality, so retain its raw evidence under
        // that tier and never expose a quality ordinal as a fast-row ordinal.
        for result in &mut results {
            result.index = None;
            result.quality_score = result.fast_score.take();
            if result.source == ScoreSource::SemanticFast {
                result.source = ScoreSource::SemanticQuality;
            }
            if let Some(explanation) = &mut result.explanation {
                for component in &mut explanation.components {
                    if let ExplainedSource::SemanticFast {
                        embedder,
                        cosine_sim,
                    } = &component.source
                    {
                        component.source = ExplainedSource::SemanticQuality {
                            embedder: embedder.clone(),
                            cosine_sim: *cosine_sim,
                        };
                    }
                }
            }
        }
        results
    }

    #[allow(clippy::too_many_lines)]
    fn search_refined(
        &self,
        initial: SyncSearchOutcome,
        refinement: SyncRefinement<'_>,
        context: SyncSearchContext<'_>,
        want_phase: bool,
    ) -> Result<SyncSearchOutcome, Box<SyncRefinementFailure>> {
        let filter = context.filter;
        let SyncSearchOutcome {
            final_results: initial_results,
            mut metrics,
            mut final_ranking,
            ..
        } = initial;
        let SyncRefinement {
            fast_query,
            quality_query,
            mut fast_hits,
            mut lexical_hits,
            mut lexical_context,
            query_semantic_weight,
            fetch,
            k,
        } = refinement;
        // Phase 1 schedules this state only for an independently admitted
        // nonzero quality query and a non-control fast lane. Both retained
        // indices and both bound embeddings stay borrowed through iteration.
        let quality_index = self
            .index
            .quality_index()
            .expect("admission binds a quality query only when a quality tier exists");
        let quality_embedder_id = quality_index.embedder_id().unwrap_or("sync-quality-query");

        let phase2_started = Instant::now();
        // A lazy caller can change policy after Initial. Refuse stale retained
        // inputs before quality work, using only this request's observations.
        if let Err(error) = lexical_context.validate_refinement_inputs(
            context,
            &fast_hits,
            &[],
            lexical_hits.as_deref(),
        ) {
            return Err(Box::new(SyncRefinementFailure {
                initial_results: Vec::new(),
                error,
                latency: phase2_started.elapsed(),
            }));
        }
        // Retrieve independently after joining the quality query to its
        // retained producer. Both admitted v2 and producer-stamped v1 sources
        // can find documents outside the fast pool, matching the async path.
        // The v1 fingerprint does not manufacture v2 generation authority or
        // coverage. Caller-declared spaces and unstamped control fixtures keep
        // their existing rescoring contract.
        let independent_quality = independently_retrievable_quality(quality_index);
        let mut lexical_refill_time = Duration::ZERO;
        let mut reconciled_zero_signal = None;
        let quality_pool = if independent_quality {
            search_filtered_vectors(
                &mut lexical_context,
                fetch,
                quality_index.record_count(),
                filter,
                |limit, candidates| {
                    quality_index.search_top_k_classified_with_params(
                        quality_query.vector(),
                        limit,
                        candidates,
                        self.search_params.unwrap_or_default(),
                    )
                },
            )
            .and_then(|mut classified| {
                let mut fast_classified = ClassifiedHits {
                    hits: std::mem::take(&mut fast_hits),
                    zero_signal: None,
                };
                let mut fast_revoked = AHashSet::new();
                let mut quality_revoked = AHashSet::new();
                loop {
                    // Either tier can resolve an observation used by the other.
                    // Keep both retained pools eligible before combining evidence.
                    let (elapsed, quality_changed) = reconcile_filtered_vectors(
                        &mut lexical_context,
                        fast_query.vector(),
                        lexical_hits.as_mut(),
                        &mut classified,
                        fetch,
                        quality_index.record_count(),
                        filter,
                        &mut quality_revoked,
                        |limit, candidates| {
                            quality_index.search_top_k_classified_with_params(
                                quality_query.vector(),
                                limit,
                                candidates,
                                self.search_params.unwrap_or_default(),
                            )
                        },
                    )?;
                    lexical_refill_time += elapsed;
                    let (elapsed, fast_changed) = reconcile_filtered_vectors(
                        &mut lexical_context,
                        fast_query.vector(),
                        lexical_hits.as_mut(),
                        &mut fast_classified,
                        fetch,
                        self.index.fast_index().record_count(),
                        filter,
                        &mut fast_revoked,
                        |limit, candidates| {
                            self.search_fast_hits(fast_query.vector(), limit, candidates)
                        },
                    )?;
                    lexical_refill_time += elapsed;
                    if !quality_changed && !fast_changed {
                        break;
                    }
                }
                if fast_classified.hits.is_empty()
                    && classified.hits.is_empty()
                    && (!fast_revoked.is_empty() || !quality_revoked.is_empty())
                {
                    // Initial may have had signal that metadata subsequently
                    // revoked. Keep the reason from the exhausted refill.
                    reconciled_zero_signal = fast_classified.zero_signal.or(classified.zero_signal);
                }
                fast_hits = fast_classified.hits;
                let mut hits = classified.hits;
                let mut refined_fast_hits = fast_hits.clone();
                complete_quality_evidence(&fast_hits, &mut hits, |missing| {
                    self.index
                        .quality_scores_for_hits(quality_query.vector(), missing)
                })?;
                let missing = missing_candidate_ids(&fast_hits, &hits);
                if !missing.is_empty() {
                    let candidates = IncludeDocIdsFilter::new(missing);
                    let fast_index = self.index.fast_index();
                    // A physical-row limit of one per ID can lose other
                    // requested IDs behind duplicate rows. The exact-ID
                    // gather scores only the selected candidate union.
                    refined_fast_hits.extend(
                        fast_index
                            .search_top_k_classified_with_params(
                                fast_query.vector(),
                                fast_index.record_count(),
                                Some(&candidates),
                                self.search_params.unwrap_or_default(),
                            )?
                            .hits,
                    );
                }
                Ok((SyncQualityPool::Retrieved(hits), refined_fast_hits))
            })
        } else {
            self.index
                .quality_scores_for_hits(quality_query.vector(), &fast_hits)
                .map(|scores| (SyncQualityPool::RescoredFastPool(scores), fast_hits.clone()))
        };
        let (quality_pool, refined_fast_hits) = match quality_pool {
            Ok(pool) => pool,
            Err(error) => {
                let latency = phase2_started.elapsed();
                // A later metadata observation can revoke the already-published
                // Initial or an undisplayed candidate retained for MMR. Neither
                // pool is a valid fallback after that observation, even if the
                // quality failure itself normally permits graceful degradation.
                let fallback_is_eligible = filter.is_none_or(|filter| {
                    final_ranking
                        .as_ref()
                        .map_or(initial_results.as_slice(), |ranking| {
                            ranking.candidates.as_slice()
                        })
                        .iter()
                        .all(|hit| {
                            filter.matches(
                                &hit.doc_id,
                                lexical_context
                                    .metadata
                                    .get(&hit.doc_id)
                                    .and_then(Option::as_deref)
                                    .or(hit.metadata.as_deref()),
                            )
                        })
                });
                if !fallback_is_eligible || matches!(&error, SearchError::Cancelled { .. }) {
                    // Collection retains the original typed failure. Iterators
                    // keep an eligible Initial for cancellation, but never
                    // republish an invalidated fallback or schedule final MMR.
                    return Err(Box::new(SyncRefinementFailure {
                        initial_results: if fallback_is_eligible {
                            initial_results
                        } else {
                            Vec::new()
                        },
                        error,
                        latency,
                    }));
                }
                // Preserve metadata resolved during the failed refinement for
                // the next lazy ranking call, without rewriting the phase that
                // has already been displayed to the caller.
                if let Some(ranking) = &mut final_ranking {
                    lexical_context.attach_metadata(&mut ranking.candidates);
                }
                metrics.phase2_total_ms = ms(latency);
                metrics.skip_reason = Some(error.diagnostic_code().to_owned());
                let phase = want_phase.then(|| SearchPhase::RefinementFailed {
                    initial_results: initial_results.clone(),
                    error,
                    latency,
                });
                return Ok(SyncSearchOutcome {
                    phase,
                    final_results: initial_results,
                    metrics,
                    final_ranking,
                });
            }
        };
        // Retrieval and metadata callbacks can revoke an earlier admission.
        // Validate the reconciled inputs, including undisplayed candidates,
        // before any of their evidence participates in the refined ranking.
        let quality_hits = match &quality_pool {
            SyncQualityPool::Retrieved(hits) => hits.as_slice(),
            SyncQualityPool::RescoredFastPool(_) => &[],
        };
        if let Err(error) = lexical_context.validate_refinement_inputs(
            context,
            &refined_fast_hits,
            quality_hits,
            lexical_hits.as_deref(),
        ) {
            return Err(Box::new(SyncRefinementFailure {
                initial_results: Vec::new(),
                error,
                latency: phase2_started.elapsed(),
            }));
        }
        metrics.lexical_search_ms += ms(lexical_refill_time);
        metrics.lexical_candidates = lexical_hits.as_ref().map_or(0, Vec::len);
        if let Some(reason) = reconciled_zero_signal {
            metrics.zero_signal = Some(reason);
        }

        let blend_started = Instant::now();
        // Per-tier evidence is keyed by canonical doc_id in BOTH shapes. The
        // retired `AlignedScoreLookup` keyed it by fast-tier row ordinal, which
        // is only expressible while the quality tier is a subset of the fast
        // pool; a doc_id key costs one hash per row and is the only key an
        // independently retrieved pool can share.
        let fast_scores_by_doc: AHashMap<&str, f32> = refined_fast_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();
        let quality_scores_by_doc: AHashMap<&str, f32> = match &quality_pool {
            SyncQualityPool::Retrieved(hits) => hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score))
                .collect(),
            SyncQualityPool::RescoredFastPool(scores) => fast_hits
                .iter()
                .zip(scores.iter())
                .filter_map(|(hit, score)| score.map(|s| (hit.doc_id.as_str(), s)))
                .collect(),
        };
        let quality_count = quality_scores_by_doc.len();
        metrics.phase2_vectors_searched = quality_count;
        metrics.incomplete_embeddings = fast_hits
            .iter()
            .filter(|hit| !quality_scores_by_doc.contains_key(hit.doc_id.as_str()))
            .count();
        if quality_count > 0 {
            // This classifies the combined semantic lane. Independent quality
            // retrieval can supply signal even when the fast lane had none.
            metrics.zero_signal = None;
        }
        let quality_weight = saturating_f64_to_f32(self.config.quality_weight);
        let blended = match &quality_pool {
            SyncQualityPool::RescoredFastPool(scores) => {
                blend_two_tier_aligned_vector_index(&fast_hits, scores, quality_weight)
            }
            SyncQualityPool::Retrieved(hits) => {
                let mut blended = blend_two_tier(&refined_fast_hits, hits, quality_weight);
                // `VectorHit::index` is a FAST-tier row ordinal for every
                // consumer downstream. Only documents still lacking usable
                // fast evidence after completion carry the "no index"
                // sentinel. A quality discovery with a live fast row keeps
                // that row, even when it missed the fast retrieval window.
                let fast_indices: AHashMap<&str, u32> = refined_fast_hits
                    .iter()
                    .map(|hit| (hit.doc_id.as_str(), hit.index))
                    .collect();
                for hit in &mut blended {
                    hit.index = fast_indices
                        .get(hit.doc_id.as_str())
                        .copied()
                        .unwrap_or(u32::MAX);
                }
                blended
            }
        };
        metrics.blend_ms = ms(blend_started.elapsed());
        metrics.quality_search_ms =
            ms(phase2_started.elapsed().saturating_sub(lexical_refill_time));
        metrics.quality_embed_ms = 0.0;

        // Keep this diagnostic on the full phase candidate pools, not only
        // the displayed top-k. The async searcher uses the same definition;
        // truncating here made identical searches report different stable /
        // promoted / demoted counts whenever `k < fetch` (bd-k3089).
        let rank_changes = compute_rank_changes(&fast_hits, &blended);

        let target = self.final_candidate_target(k, context);
        let mut fusion_components = Vec::new();
        let mut refined_results = if let Some(lexical) = lexical_hits.as_ref() {
            let rrf_config = RrfConfig {
                k: self.config.rrf_k,
                lexical_weight: self.rrf_lexical_weight,
                semantic_weight: query_semantic_weight,
                tiebreak: self.rrf_tiebreak,
            };
            let fused = fuse_by_strategy(
                self.config.fusion_strategy,
                lexical,
                &blended,
                &[],
                0.0,
                target,
                0,
                &rrf_config,
            );
            if self.config.explain {
                // The conversion below preserves this fused order.
                let evidence = FusionExplanation::new(
                    self.config.fusion_strategy,
                    lexical,
                    &blended,
                    &[],
                    0.0,
                    &rrf_config,
                );
                fusion_components.extend(fused.iter().map(|hit| evidence.refined_components(hit)));
            }
            fused_hits_to_scored_results(
                fused,
                lexical,
                target,
                None,
                sync_fast_embedder_id(&self.index),
                false,
            )
        } else {
            unique_vector_hits_to_scored_results_owned(blended, target, ScoreSource::SemanticFast)
        };
        lexical_context.attach_metadata(&mut refined_results);

        // Re-fusion ranks on blended semantic scores, but `fast_score` and
        // `quality_score` are evidence fields: they must retain the raw
        // per-tier values. Use the same source classification as async once
        // the raw quality evidence is available, preserving lexical and
        // hybrid provenance.
        for result in &mut refined_results {
            result.fast_score = fast_scores_by_doc.get(result.doc_id.as_str()).copied();
            result.quality_score = quality_scores_by_doc.get(result.doc_id.as_str()).copied();
            result.source = promoted_quality_source(result.source, result.quality_score.is_some());
        }

        if self.config.explain {
            // Async refinement measures semantic-only movement against the
            // full Phase-1 candidate pool, while lexical re-fusion measures
            // against the displayed fused order. Mirroring that distinction
            // keeps promoted candidates explainable even when they were below
            // the initial top-k display cutoff.
            let initial_ranks = if lexical_hits.is_some() {
                initial_results
                    .iter()
                    .enumerate()
                    .map(|(rank, result)| (result.doc_id.as_str(), rank))
                    .collect::<AHashMap<_, _>>()
            } else {
                fast_hits
                    .iter()
                    .enumerate()
                    .map(|(rank, hit)| (hit.doc_id.as_str(), rank))
                    .collect::<AHashMap<_, _>>()
            };
            let blend_explanation = match &quality_pool {
                SyncQualityPool::Retrieved(hits) => BlendExplanation::new(
                    &refined_fast_hits,
                    hits.iter().map(|hit| &hit.score),
                    quality_weight,
                ),
                SyncQualityPool::RescoredFastPool(scores) => BlendExplanation::new(
                    &refined_fast_hits,
                    scores.iter().flatten(),
                    quality_weight,
                ),
            };
            let fast_embedder_id = sync_fast_embedder_id(&self.index);
            let mut fusion_components = fusion_components.into_iter();
            for (rank, result) in refined_results.iter_mut().enumerate() {
                let vector_components = blend_explanation.components(
                    result.fast_score,
                    result.quality_score,
                    // Admission, rather than the model display name, determines this tier.
                    |cosine_sim| ExplainedSource::SemanticFast {
                        embedder: fast_embedder_id.to_owned(),
                        cosine_sim,
                    },
                    |cosine_sim| ExplainedSource::SemanticQuality {
                        embedder: quality_embedder_id.to_owned(),
                        cosine_sim,
                    },
                );
                result.explanation = Some(Box::new(build_refined_explanation(
                    result.score,
                    rank,
                    result.doc_id.as_str(),
                    &initial_ranks,
                    fusion_components.next().unwrap_or_default(),
                    vector_components,
                )));
            }
        }

        metrics.rank_changes = rank_changes.clone();
        metrics.phase2_total_ms = ms(phase2_started.elapsed());
        metrics.kendall_tau = None;
        let final_ranking = self.retain_final_candidates(
            &mut refined_results,
            k,
            SyncMmrSpace::PreferQuality,
            context,
        );

        let phase = want_phase.then(|| SearchPhase::Refined {
            results: refined_results.clone(),
            latency: phase2_started.elapsed(),
            metrics: PhaseMetrics {
                embedder_id: quality_embedder_id.to_owned(),
                vectors_searched: quality_count,
                lexical_candidates: metrics.lexical_candidates,
                fused_count: refined_results.len(),
                skip_reason: None,
                hash_control_candidates: metrics.hash_control_candidates,
            },
            rank_changes,
        });

        Ok(SyncSearchOutcome {
            phase,
            final_results: refined_results,
            metrics,
            final_ranking,
        })
    }

    fn mmr_embeddings(
        &self,
        results: &[ScoredResult],
        space: SyncMmrSpace,
    ) -> SearchResult<Option<Vec<Vec<f32>>>> {
        if !matches!(space, SyncMmrSpace::Fast)
            && let Some(quality) = self.index.quality_index()
            && let Some(embeddings) = sync_tier_embeddings(quality, results)?
        {
            return Ok(Some(embeddings));
        }
        if matches!(space, SyncMmrSpace::QualityOnly) {
            return Ok(None);
        }
        sync_tier_embeddings(self.index.fast_index(), results)
    }

    fn search_final(
        &self,
        outcome: SyncSearchOutcome,
        context: SyncSearchContext<'_>,
        want_phase: bool,
    ) -> SearchResult<SyncSearchOutcome> {
        // Keep only identities and pinned observations. Text and inference
        // callbacks may change policy while this final stage is running.
        let retained: Option<SyncMetadata> = context
            .filter
            .and(outcome.final_ranking.as_ref())
            .map(|ranking| {
                ranking
                    .candidates
                    .iter()
                    .map(|hit| (hit.doc_id.clone(), hit.metadata.clone()))
                    .collect()
            });
        let refusal = || SearchError::InvalidConfig {
            field: "search.filter".to_owned(),
            value: "eligibility_changed".to_owned(),
            reason: "document eligibility changed before final ranking completed".to_owned(),
        };
        let admission = || {
            context.checkpoint("sync_search.final_admission")?;
            let eligible = match (context.filter, retained.as_ref()) {
                (Some(filter), Some(retained)) => retained
                    .iter()
                    .all(|(id, metadata)| filter.matches(id, metadata.as_deref())),
                _ => true,
            };
            context.checkpoint("sync_search.final_admission")?;
            if eligible { Ok(()) } else { Err(refusal()) }
        };
        admission()?;
        let ranked = self.rank_final_candidates(outcome, context, want_phase, &admission);
        // A provider's typed cancellation is terminal even if it also revoked
        // eligibility. Ordinary failures still require an eligible fallback.
        if matches!(&ranked, Err(SearchError::Cancelled { .. })) {
            return ranked;
        }
        admission()?;
        let ranked = ranked?;
        if let (Some(filter), Some(retained)) = (context.filter, retained.as_ref()) {
            let eligible = ranked.final_results.iter().all(|hit| {
                filter.matches(
                    &hit.doc_id,
                    retained
                        .get(&hit.doc_id)
                        .map_or(hit.metadata.as_deref(), Option::as_deref),
                )
            });
            context.checkpoint("sync_search.final_publish")?;
            if !eligible {
                return Err(refusal());
            }
        }
        Ok(ranked)
    }

    #[allow(clippy::too_many_lines)]
    fn rank_final_candidates(
        &self,
        mut outcome: SyncSearchOutcome,
        context: SyncSearchContext<'_>,
        want_phase: bool,
        admission: &(dyn Fn() -> SearchResult<()> + Send + Sync),
    ) -> SearchResult<SyncSearchOutcome> {
        outcome.phase = None;
        let Some(ranking) = outcome.final_ranking.take() else {
            return Ok(outcome);
        };
        let started = Instant::now();
        let SyncFinalRanking {
            mut candidates,
            k,
            space,
        } = ranking;
        let mut ranker_id = None;
        #[cfg(feature = "rerank")]
        if let (Some(reranker), Some(input), Some(cx), Some(window)) = (
            self.reranker.as_deref(),
            context.rerank,
            context.cx,
            context.rerank_window,
        ) {
            ranker_id =
                frankensearch_rerank::pipeline::rerank_step_sync_with_combine_and_admission(
                    cx,
                    reranker,
                    input.text,
                    &mut candidates,
                    input.text_fn,
                    window,
                    1,
                    self.rerank_combine,
                    admission,
                )?;
            outcome.metrics.rerank_ms = ms(started.elapsed());
        }
        // This also protects the MMR-only route and refuses an invalid fallback
        // after a skipped cross-encoder response.
        admission()?;

        if self.mmr_config.enabled && self.mmr_config.candidate_pool > 1 && candidates.len() > 1 {
            let pool = candidates.len().min(self.mmr_config.candidate_pool);
            let embeddings = match self.mmr_embeddings(&candidates[..pool], space) {
                Ok(embeddings) => embeddings,
                Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                Err(error) => {
                    outcome.metrics.rerank_ms = ms(started.elapsed());
                    tracing::warn!(
                        error_code = error.diagnostic_code(),
                        "synchronous diversity ranking failed"
                    );
                    None
                }
            };
            context.checkpoint("sync_search.mmr")?;
            if let Some(embeddings) = embeddings {
                let scores = candidates[..pool]
                    .iter()
                    .enumerate()
                    .map(|(rank, result)| {
                        if ranker_id.is_some() {
                            // Preserve the admitted combination order. Retrieval
                            // scores undo promotions; raw model scores undo RRF.
                            f64::from(u32::try_from(pool - rank).unwrap_or(u32::MAX))
                        } else {
                            f64::from(result.score)
                        }
                    })
                    .collect::<Vec<_>>();
                let refs = embeddings.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let selected = k.min(pool);
                let order = mmr_rerank(&scores, &refs, selected, &self.mmr_config);
                if order.len() == selected {
                    // Only the bounded prefix participates in diversity.
                    let tail = candidates.split_off(pool);
                    let mut slots = candidates.into_iter().map(Some).collect::<Vec<_>>();
                    let mut results = Vec::with_capacity(k.min(selected + tail.len()));
                    for index in order {
                        if let Some(result) = slots.get_mut(index).and_then(Option::take) {
                            results.push(result);
                        }
                    }
                    results.extend(tail.into_iter().take(k.saturating_sub(selected)));
                    candidates = results;
                    ranker_id.get_or_insert_with(|| "mmr".to_owned());
                }
            }
        }

        admission()?;
        let Some(ranker_id) = ranker_id else {
            return Ok(outcome);
        };
        candidates.truncate(k);
        if self.config.explain {
            for (rank, result) in candidates.iter_mut().enumerate() {
                if let Some(explanation) = &mut result.explanation
                    && let Some(movement) = &mut explanation.rank_movement
                {
                    movement.refined_rank = rank;
                    let initial = i64::try_from(movement.initial_rank).unwrap_or(i64::MAX);
                    let final_rank = i64::try_from(rank).unwrap_or(i64::MAX);
                    let delta = final_rank.saturating_sub(initial);
                    movement.delta = i32::try_from(delta).unwrap_or_else(|_| {
                        if delta.is_negative() {
                            i32::MIN
                        } else {
                            i32::MAX
                        }
                    });
                    movement.reason = match movement.delta.cmp(&0) {
                        std::cmp::Ordering::Less => "promoted".to_owned(),
                        std::cmp::Ordering::Greater => "demoted".to_owned(),
                        std::cmp::Ordering::Equal => "stable".to_owned(),
                    };
                }
            }
        }
        let latency = started.elapsed();
        outcome.metrics.rerank_ms = ms(latency);
        outcome.phase = want_phase.then(|| SearchPhase::Reranked {
            results: candidates.clone(),
            latency,
            metrics: PhaseMetrics {
                embedder_id: ranker_id,
                vectors_searched: if matches!(space, SyncMmrSpace::PreferQuality) {
                    outcome.metrics.phase2_vectors_searched
                } else {
                    outcome.metrics.phase1_vectors_searched
                },
                lexical_candidates: outcome.metrics.lexical_candidates,
                fused_count: candidates.len(),
                skip_reason: outcome.metrics.skip_reason.clone(),
                hash_control_candidates: outcome.metrics.hash_control_candidates,
            },
        });
        outcome.final_results = candidates;
        Ok(outcome)
    }

    fn search_fast_hits(
        &self,
        query_vec: &[f32],
        fetch: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<ClassifiedHits> {
        let fast_index = self.index.fast_index();
        if let Some(params) = self.search_params {
            return fast_index
                .search_top_k_classified_with_params(query_vec, fetch, filter, params);
        }
        let hits = match self.fast_retrieval {
            SyncFastRetrieval::Exact => {
                // A verified sidecar may prune exact work without changing
                // candidate selection or visible results.
                return fast_index.search_top_k_classified(query_vec, fetch, filter);
            }
            SyncFastRetrieval::ApproximateInt8ForBench {
                candidate_multiplier,
            } => fast_index.search_top_k_int8_two_pass_filtered(
                query_vec,
                fetch,
                candidate_multiplier,
                filter,
            )?,
        };
        // The benchmark control returns raw hits. Classify an empty batch
        // once, after search_initial has handled zero k and zero-norm input.
        let zero_signal = hits.is_empty().then(|| {
            if query_vec.iter().any(|value| !value.is_finite()) {
                ZeroSignalReason::NonFiniteQuery
            } else {
                fast_index
                    .zero_signal_state()
                    .empty_result_reason(filter.is_some())
            }
        });
        Ok(ClassifiedHits { hits, zero_signal })
    }
}

impl std::fmt::Debug for SyncTwoTierSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncTwoTierSearcher")
            .field("has_lexical", &self.lexical.is_some())
            .field("search_params", &self.search_params)
            .field("fast_retrieval", &self.fast_retrieval)
            .field("has_quality_index", &self.index.has_quality_index())
            .field("config", &self.config)
            .field("rrf_lexical_weight", &self.rrf_lexical_weight)
            .field("rrf_semantic_weight", &self.rrf_semantic_weight)
            .field("rrf_tiebreak", &self.rrf_tiebreak)
            .field("nqc_downweight_beta", &self.nqc_downweight_beta)
            .field("nqc_downweight_w_min", &self.nqc_downweight_w_min)
            .field("nqc_dense_weight", &self.nqc_dense_weight)
            .field("nqc_adaptive", &self.nqc_adaptive)
            .field("mmr_config", &self.mmr_config)
            .finish()
    }
}

#[derive(Debug)]
struct SyncSearchOutcome {
    phase: Option<SearchPhase>,
    final_results: Vec<ScoredResult>,
    metrics: TwoTierMetrics,
    final_ranking: Option<SyncFinalRanking>,
}

/// Terminal refinement failures propagate to collected searches. Iterators keep
/// an eligible published Initial for cancellation, but an invalidated fallback
/// has an empty payload and cannot schedule final ranking. This is boxed only on
/// failure; successful searches allocate no error payload.
struct SyncRefinementFailure {
    initial_results: Vec<ScoredResult>,
    error: SearchError,
    latency: Duration,
}

#[derive(Debug, Clone, Copy)]
enum SyncMmrSpace {
    Fast,
    PreferQuality,
    QualityOnly,
}

#[derive(Debug)]
struct SyncFinalRanking {
    candidates: Vec<ScoredResult>,
    k: usize,
    space: SyncMmrSpace,
}

type SyncMetadata = AHashMap<DocId, Option<Arc<serde_json::Value>>>;

/// Filtering observations and their reader survive from Initial to Refined.
struct SyncLexicalContext {
    source: Option<Arc<dyn SyncLexicalSearch>>,
    metadata: SyncMetadata,
}

impl std::fmt::Debug for SyncLexicalContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SyncLexicalContext")
            .field("has_source", &self.source.is_some())
            .field("metadata_count", &self.metadata.len())
            .finish()
    }
}

impl SyncLexicalContext {
    fn new(source: Option<Arc<dyn SyncLexicalSearch>>, filtered: bool) -> SearchResult<Self> {
        let source = match source {
            Some(source) if filtered => Some(source.pin_search()?.unwrap_or(source)),
            source => source,
        };
        Ok(Self {
            source,
            metadata: AHashMap::new(),
        })
    }

    fn hydrate_candidates(&mut self, hits: &[VectorHit]) -> SearchResult<()> {
        let Some(source) = &self.source else {
            return Ok(());
        };
        let missing: Vec<_> = hits
            .iter()
            .map(|hit| hit.doc_id.as_str())
            // A prior unknown observation cannot suppress an available lookup.
            .filter(|id| self.metadata.get(*id).is_none_or(Option::is_none))
            .collect();
        hydrate_selected_metadata(source.as_ref(), &missing, &mut self.metadata)
    }

    fn revalidate_lexical(
        &mut self,
        query_vec: &[f32],
        target: usize,
        hits: Option<&mut Vec<ScoredResult>>,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Duration> {
        let (Some(source), Some(hits), Some(filter)) = (&self.source, hits, filter) else {
            return Ok(Duration::ZERO);
        };
        // Vector retrieval can resolve an observation that was unknown when
        // lexical retrieval admitted this row. If the original pool had room
        // left, that retrieval already exhausted the lexical results.
        let filled_budget = hits.len() == target;
        hits.retain(|hit| {
            filter.matches(
                &hit.doc_id,
                self.metadata
                    .get(&hit.doc_id)
                    .and_then(Option::as_deref)
                    .or(hit.metadata.as_deref()),
            )
        });
        if !filled_budget || hits.len() == target {
            return Ok(Duration::ZERO);
        }

        let started = Instant::now();
        let mut observations = SyncMetadata::new();
        let outcome = search_lexical_candidates(
            source.as_ref(),
            query_vec,
            target,
            Some(filter),
            &mut observations,
            Some(&self.metadata),
        );
        // A later refill or hydration failure cannot erase observations from a
        // successful prefix. Refinement fallback needs them too when deciding
        // whether the previously published Initial is still eligible.
        self.metadata.extend(observations);
        match outcome {
            Ok(refilled) => {
                *hits = refilled;
            }
            Err(
                SyncLexicalFailure::Metadata(error)
                | SyncLexicalFailure::Retrieval(error @ SearchError::Cancelled { .. }),
            ) => return Err(error),
            Err(SyncLexicalFailure::Retrieval(error)) => tracing::warn!(
                error_code = error.diagnostic_code(),
                "lexical refill failed after metadata revalidation"
            ),
        }
        Ok(started.elapsed())
    }

    fn validate_refinement_inputs(
        &self,
        context: SyncSearchContext<'_>,
        fast_hits: &[VectorHit],
        quality_hits: &[VectorHit],
        lexical_hits: Option<&[ScoredResult]>,
    ) -> SearchResult<()> {
        context.checkpoint("sync_search.refinement_admission")?;
        let Some(filter) = context.filter else {
            return Ok(());
        };
        let candidates = fast_hits
            .iter()
            .chain(quality_hits)
            .map(|hit| (hit.doc_id.as_str(), None))
            .chain(
                lexical_hits
                    .into_iter()
                    .flatten()
                    .map(|hit| (hit.doc_id.as_str(), hit.metadata.as_deref())),
            );
        for (id, metadata) in candidates {
            context.checkpoint("sync_search.refinement_admission")?;
            let eligible = filter.matches(
                id,
                self.metadata
                    .get(id)
                    .and_then(Option::as_deref)
                    .or(metadata),
            );
            context.checkpoint("sync_search.refinement_admission")?;
            if !eligible {
                return Err(SearchError::InvalidConfig {
                    field: "search.filter".to_owned(),
                    value: "eligibility_changed".to_owned(),
                    reason: "document eligibility changed before quality refinement completed"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }

    fn attach_metadata(&self, results: &mut [ScoredResult]) {
        for result in results {
            if result.metadata.is_none() {
                result.metadata = self.metadata.get(result.doc_id.as_str()).cloned().flatten();
            }
        }
    }
}

#[derive(Debug)]
struct SyncRefinement<'query> {
    fast_query: &'query BoundQueryEmbedding,
    quality_query: &'query BoundQueryEmbedding,
    fast_hits: Vec<VectorHit>,
    lexical_hits: Option<Vec<ScoredResult>>,
    lexical_context: SyncLexicalContext,
    query_semantic_weight: f64,
    fetch: usize,
    k: usize,
}

#[derive(Debug)]
struct SyncPreparedSearch<'query> {
    initial: SyncSearchOutcome,
    refinement: Option<SyncRefinement<'query>>,
}

struct SyncPendingRefinement<'query> {
    initial: SyncSearchOutcome,
    refinement: SyncRefinement<'query>,
}

enum SyncIteratorState<'query> {
    Initial {
        query: &'query TieredQueryEmbeddings,
        k: usize,
    },
    Refine(Box<SyncPendingRefinement<'query>>),
    Rank(Box<SyncSearchOutcome>),
    Done,
}

/// Lazy iterator over progressive phases produced by [`SyncTwoTierSearcher`].
///
/// The first `next()` validates both query identities and performs only fast
/// and lexical retrieval. A second `next()` runs quality refinement when that
/// tier is available and requested. Dropping an unconsumed iterator performs
/// no search; dropping after Initial performs no quality traversal or blending.
/// A quality-only request directly retrieves quality on the first `next()` and
/// yields Initial with quality provenance. No fast tier is read.
/// With MMR enabled, one further `next()` may emit Reranked after the last
/// successful retrieval. Text-aware entry points run a configured synchronous
/// cross-encoder at that same final boundary, before optional MMR. Dropping
/// before that call performs no document text lookup, inference or diversity work.
/// The searcher, bound queries and filter remain borrowed until the iterator is
/// dropped, keeping both phases on the same retained in-memory index. Filters
/// should keep their selection policy stable across those phases.
///
/// Work runs synchronously on the thread calling `next()`; no background job is
/// spawned. Phase latency measures that call's work, excluding caller time
/// between phases. The iterator is exhausted after a terminal phase or refusal.
pub struct SyncSearchIterator<'a> {
    searcher: &'a SyncTwoTierSearcher,
    context: SyncSearchContext<'a>,
    state: SyncIteratorState<'a>,
}

impl std::fmt::Debug for SyncSearchIterator<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = match self.state {
            SyncIteratorState::Initial { .. } => "initial",
            SyncIteratorState::Refine(_) => "refinement",
            SyncIteratorState::Rank(_) => "ranking",
            SyncIteratorState::Done => "done",
        };
        formatter
            .debug_struct("SyncSearchIterator")
            .field("state", &state)
            .field("filtered", &self.context.filter.is_some())
            .finish_non_exhaustive()
    }
}

impl Iterator for SyncSearchIterator<'_> {
    type Item = SearchPhase;

    fn next(&mut self) -> Option<Self::Item> {
        if matches!(self.state, SyncIteratorState::Done) {
            return None;
        }
        if let Err(error) = self.context.checkpoint("sync_search.iterator") {
            self.state = SyncIteratorState::Done;
            return Some(SearchPhase::RefinementFailed {
                initial_results: Vec::new(),
                error,
                latency: Duration::ZERO,
            });
        }
        // Consume the pending state before doing work. Errors and terminal
        // phases cannot replay a retrieval on a subsequent call.
        let phase = match std::mem::replace(&mut self.state, SyncIteratorState::Done) {
            SyncIteratorState::Initial { query, k } => {
                match self.searcher.search_initial(query, k, self.context, true) {
                    Ok(mut prepared) => {
                        let phase = prepared.initial.phase.take();
                        if let Some(refinement) = prepared.refinement {
                            self.state =
                                SyncIteratorState::Refine(Box::new(SyncPendingRefinement {
                                    initial: prepared.initial,
                                    refinement,
                                }));
                        } else if prepared.initial.final_ranking.is_some() {
                            self.state = SyncIteratorState::Rank(Box::new(prepared.initial));
                        }
                        phase
                    }
                    Err(error) => Some(SearchPhase::RefinementFailed {
                        initial_results: Vec::new(),
                        error,
                        latency: Duration::ZERO,
                    }),
                }
            }
            SyncIteratorState::Refine(pending) => {
                match self.searcher.search_refined(
                    pending.initial,
                    pending.refinement,
                    self.context,
                    true,
                ) {
                    Ok(mut outcome) => {
                        let phase = outcome.phase.take();
                        if outcome.final_ranking.is_some() {
                            self.state = SyncIteratorState::Rank(Box::new(outcome));
                        }
                        phase
                    }
                    Err(failure) => Some(SearchPhase::RefinementFailed {
                        initial_results: failure.initial_results,
                        error: failure.error,
                        latency: failure.latency,
                    }),
                }
            }
            SyncIteratorState::Rank(pending) => {
                let started = Instant::now();
                match self.searcher.search_final(*pending, self.context, true) {
                    Ok(outcome) => outcome.phase,
                    Err(error) => Some(SearchPhase::RefinementFailed {
                        initial_results: Vec::new(),
                        error,
                        latency: started.elapsed(),
                    }),
                }
            }
            SyncIteratorState::Done => None,
        };
        if matches!(
            &phase,
            Some(SearchPhase::RefinementFailed {
                error: SearchError::Cancelled { .. },
                ..
            })
        ) {
            return phase;
        }
        if let Err(error) = self.context.checkpoint("sync_search.iterator_publish") {
            self.state = SyncIteratorState::Done;
            return Some(SearchPhase::RefinementFailed {
                initial_results: Vec::new(),
                error,
                latency: Duration::ZERO,
            });
        }
        phase
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let ranking = usize::from(self.searcher.final_ranking_enabled(self.context));
        match self.state {
            SyncIteratorState::Initial { k: 0, .. } => (0, Some(1)),
            SyncIteratorState::Initial { query, .. } if query.fast().is_none() => {
                (1, Some(1 + ranking))
            }
            SyncIteratorState::Initial { .. } => (1, Some(2 + ranking)),
            SyncIteratorState::Refine(_) => (1, Some(1 + ranking)),
            SyncIteratorState::Rank(_) => (0, Some(1)),
            SyncIteratorState::Done => (0, Some(0)),
        }
    }
}

impl std::iter::FusedIterator for SyncSearchIterator<'_> {}

fn sync_tier_embeddings(
    index: &InMemoryVectorIndex,
    results: &[ScoredResult],
) -> SearchResult<Option<Vec<Vec<f32>>>> {
    let mut embeddings = Vec::with_capacity(results.len());
    for result in results {
        let Some(vector) = index.vector_for_doc_id(&result.doc_id)? else {
            return Ok(None);
        };
        embeddings.push(vector);
    }
    Ok(Some(embeddings))
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[allow(clippy::cast_possible_truncation)]
fn saturating_f64_to_f32(value: f64) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(f64::from(f32::MIN), f64::from(f32::MAX)) as f32
}

fn build_refined_explanation(
    final_score: f32,
    rank: usize,
    doc_id: &str,
    initial_ranks: &AHashMap<&str, usize>,
    fusion_components: [Option<ScoreComponent>; 2],
    vector_components: impl Iterator<Item = ScoreComponent>,
) -> HitExplanation {
    let components = fusion_components
        .into_iter()
        .flatten()
        .chain(vector_components)
        .collect();
    let rank_movement = initial_ranks.get(doc_id).map(|&initial_rank| {
        let refined_rank = i64::try_from(rank).unwrap_or(i64::MAX);
        let initial_rank_i64 = i64::try_from(initial_rank).unwrap_or(i64::MAX);
        let delta_i64 = refined_rank - initial_rank_i64;
        let delta = i32::try_from(delta_i64).unwrap_or_else(|_| {
            if delta_i64.is_negative() {
                i32::MIN
            } else {
                i32::MAX
            }
        });
        let reason = match delta.cmp(&0) {
            std::cmp::Ordering::Less => "promoted",
            std::cmp::Ordering::Greater => "demoted",
            std::cmp::Ordering::Equal => "stable",
        };
        RankMovement {
            initial_rank,
            refined_rank: rank,
            delta,
            reason: reason.to_owned(),
        }
    });
    HitExplanation {
        final_score: f64::from(final_score),
        components,
        phase: ExplanationPhase::Refined,
        rank_movement,
    }
}

enum SyncLexicalFailure {
    Retrieval(SearchError),
    Metadata(SearchError),
}

fn hydrate_selected_metadata(
    source: &dyn SyncLexicalSearch,
    document_ids: &[&str],
    observations: &mut SyncMetadata,
) -> SearchResult<()> {
    if document_ids.is_empty() {
        return Ok(());
    }
    let metadata = source.metadata_for_documents(document_ids)?;
    if metadata.len() != document_ids.len() {
        return Err(SearchError::InvalidConfig {
            field: "sync_lexical.metadata".to_owned(),
            value: metadata.len().to_string(),
            reason: format!(
                "metadata lookup returned {} observations for {} candidates",
                metadata.len(),
                document_ids.len(),
            ),
        });
    }
    for (id, metadata) in document_ids.iter().zip(metadata) {
        observations.insert(DocId::from(*id), metadata);
    }
    Ok(())
}

fn search_lexical_candidates(
    lexical: &dyn SyncLexicalSearch,
    query_vec: &[f32],
    target: usize,
    filter: Option<&dyn SearchFilter>,
    metadata: &mut SyncMetadata,
    retained_metadata: Option<&SyncMetadata>,
) -> Result<Vec<ScoredResult>, SyncLexicalFailure> {
    let mut limit = retained_metadata.map_or(target, |_| target.saturating_mul(2));
    loop {
        let mut hits = lexical
            .search_sync(query_vec, limit)
            .map_err(SyncLexicalFailure::Retrieval)?;
        let exhausted = hits.len() < limit;
        // Keep rejected observations too: the vector lane must not reintroduce
        // a document whose actual metadata this batch already ruled out.
        // Unpinned custom sources may change between refills, so only the last
        // complete batch supplies observations, just as it supplies scores.
        metadata.clear();
        // Keep the best ranked row for each public ID. A later row can still
        // supply its metadata when earlier rows had none.
        hits.retain_mut(|hit| {
            // A refill must retain known observations already used by the
            // vector lane, but a prior unknown cannot erase new metadata.
            if let Some(Some(observation)) =
                retained_metadata.and_then(|prior| prior.get(&hit.doc_id))
            {
                hit.metadata = Some(Arc::clone(observation));
            }
            match metadata.entry(hit.doc_id.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(hit.metadata.clone());
                    true
                }
                Entry::Occupied(mut entry) => {
                    if entry.get().is_none() {
                        entry.get_mut().clone_from(&hit.metadata);
                    }
                    false
                }
            }
        });
        if filter.is_some() {
            let unknown: Vec<_> = hits
                .iter()
                .filter(|hit| {
                    metadata.get(&hit.doc_id).is_some_and(Option::is_none)
                        && retained_metadata
                            .is_none_or(|prior| prior.get(&hit.doc_id).is_none_or(Option::is_none))
                })
                .map(|hit| hit.doc_id.as_str())
                .collect();
            // Eager None is unknown, not proof that selected-document hydration
            // cannot supply metadata. Hydrate before deciding lexical membership.
            hydrate_selected_metadata(lexical, &unknown, metadata)
                .map_err(SyncLexicalFailure::Metadata)?;
        }
        hits.retain_mut(|hit| {
            if hit.metadata.is_none() {
                hit.metadata = metadata.get(&hit.doc_id).cloned().flatten();
            }
            filter.is_none_or(|filter| filter.matches(&hit.doc_id, hit.metadata.as_deref()))
        });
        hits.truncate(target);
        let next_limit = limit.saturating_mul(2);
        if hits.len() == target || exhausted || next_limit == limit {
            return Ok(hits);
        }
        // Duplicate and rejected rows can hide eligible documents below the
        // requested prefix, even if the backend filled every row in that prefix.
        // Replace the whole batch so scores and metadata always come from one
        // search; never append candidates from different reader generations.
        limit = next_limit;
    }
}

/// Only enumerable identity bounds may reject a row before metadata is read.
/// Calling an arbitrary predicate with None here can reject rows that it would
/// accept with real metadata, or admit rows it would reject. Final membership
/// is always checked below against the retained observation.
struct MetadataCandidateFilter<'a>(&'a dyn SearchFilter);

impl SearchFilter for MetadataCandidateFilter<'_> {
    fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
        self.0.candidate_doc_ids().map_or_else(
            || {
                self.0
                    .candidate_hashes()
                    .is_none_or(|hashes| hashes.contains(&fnv1a_hash(doc_id.as_bytes())))
            },
            |ids| ids.contains(doc_id),
        )
    }

    fn matches_doc_id_hash(
        &self,
        doc_id_hash: u64,
        _metadata: Option<&serde_json::Value>,
    ) -> Option<bool> {
        if self.0.candidate_doc_ids().is_some() {
            None
        } else {
            Some(
                self.0
                    .candidate_hashes()
                    .is_none_or(|hashes| hashes.contains(&doc_id_hash)),
            )
        }
    }

    fn candidate_doc_ids(&self) -> Option<&std::collections::HashSet<String>> {
        self.0.candidate_doc_ids()
    }

    fn candidate_hashes(&self) -> Option<&DocIdHashSet> {
        self.0.candidate_hashes()
    }

    fn name(&self) -> &'static str {
        "sync_metadata_candidates"
    }
}

fn search_filtered_vectors(
    context: &mut SyncLexicalContext,
    target: usize,
    physical_upper_bound: usize,
    filter: Option<&dyn SearchFilter>,
    mut retrieve: impl FnMut(usize, Option<&dyn SearchFilter>) -> SearchResult<ClassifiedHits>,
) -> SearchResult<ClassifiedHits> {
    let Some(filter) = filter.filter(|_| context.source.is_some()) else {
        return retrieve(target, filter);
    };
    let candidates = MetadataCandidateFilter(filter);
    let mut limit = target.min(physical_upper_bound.max(1));
    loop {
        let mut classified = retrieve(limit, Some(&candidates))?;
        let fetched = classified.hits.len();
        context.hydrate_candidates(&classified.hits)?;
        let mut seen = AHashSet::with_capacity(classified.hits.len());
        classified.hits.retain(|hit| {
            filter.matches(
                &hit.doc_id,
                context
                    .metadata
                    .get(hit.doc_id.as_str())
                    .and_then(Option::as_deref),
            ) && seen.insert(hit.doc_id.clone())
        });
        if classified.hits.is_empty() && fetched != 0 {
            classified.zero_signal = Some(ZeroSignalReason::FilterEliminatedAll);
        }
        classified.hits.truncate(target);
        if classified.hits.len() == target || fetched == 0 || limit >= physical_upper_bound {
            return Ok(classified);
        }
        // The requested budget counts eligible public identities, not physical
        // duplicates. The vector owner can deduplicate after taking a physical
        // prefix, so a short nonempty result doesn't prove exhaustion. Replace
        // the prefix at a wider limit while keeping both retained owners fixed.
        limit = limit
            .saturating_mul(2)
            .max(limit.saturating_add(1))
            .min(physical_upper_bound);
    }
}

/// Reconcile a retained semantic window after lexical refill learns metadata.
/// A prior physical scan may already have reached the end before its eligible
/// tail was truncated, so revocation must repeat retrieval even at exhaustion.
#[allow(clippy::too_many_arguments)]
fn reconcile_filtered_vectors(
    context: &mut SyncLexicalContext,
    lexical_query: &[f32],
    mut lexical_hits: Option<&mut Vec<ScoredResult>>,
    classified: &mut ClassifiedHits,
    target: usize,
    physical_upper_bound: usize,
    filter: Option<&dyn SearchFilter>,
    revoked: &mut AHashSet<DocId>,
    mut retrieve: impl FnMut(usize, Option<&dyn SearchFilter>) -> SearchResult<ClassifiedHits>,
) -> SearchResult<(Duration, bool)> {
    let Some(filter) = filter.filter(|_| context.source.is_some()) else {
        return Ok((Duration::ZERO, false));
    };
    let mut lexical_time = Duration::ZERO;
    let mut replaced = false;
    loop {
        lexical_time += context.revalidate_lexical(
            lexical_query,
            target,
            lexical_hits.as_deref_mut(),
            Some(filter),
        )?;
        let mut rejected = false;
        for hit in &classified.hits {
            if !filter.matches(
                &hit.doc_id,
                context
                    .metadata
                    .get(hit.doc_id.as_str())
                    .and_then(Option::as_deref),
            ) {
                // Known observations are retained, so a readmitted document
                // cannot legitimately become ineligible a second time.
                if !revoked.insert(hit.doc_id.clone()) {
                    return Err(SearchError::InvalidConfig {
                        field: "search.filter".to_owned(),
                        value: "unstable_predicate".to_owned(),
                        reason: "filter eligibility changed repeatedly during candidate refill"
                            .to_owned(),
                    });
                }
                rejected = true;
            }
        }
        if !rejected {
            return Ok((lexical_time, replaced));
        }
        *classified = search_filtered_vectors(
            context,
            target,
            physical_upper_bound,
            Some(filter),
            &mut retrieve,
        )?;
        replaced = true;
    }
}

fn fused_hits_to_scored_results(
    hits: Vec<FusedHit>,
    lexical: &[ScoredResult],
    k: usize,
    explanation: Option<FusionExplanation<'_>>,
    fast_embedder_id: &str,
    hash_control: bool,
) -> Vec<ScoredResult> {
    // Keep metadata from the same retained lexical batch that supplied the
    // scores. Borrow candidate payloads and clone only the winning Arcs, as in
    // the async searcher; semantic-only hits have no lexical metadata to attach.
    let metadata_by_doc: AHashMap<&str, &Arc<serde_json::Value>> = lexical
        .iter()
        .filter_map(|hit| {
            hit.metadata
                .as_ref()
                .map(|metadata| (hit.doc_id.as_str(), metadata))
        })
        .collect();
    // Take the `rrf_fuse` result by value and move each `doc_id` into the
    // `ScoredResult` instead of cloning it; the `FusedHit`s are a fresh
    // temporary here, so there is no need to keep them alive.
    hits.into_iter()
        .take(k)
        .map(|mut hit| {
            if hash_control {
                hit.remap_hash_control_ranks();
            }
            let score = saturating_f64_to_f32(hit.rrf_score);
            let source = if hit.in_both_sources {
                ScoreSource::Hybrid
            } else if hit.lexical_rank.is_some() {
                ScoreSource::Lexical
            } else if hit.semantic_rank.is_some() || hit.hash_rank.is_some() {
                if hash_control {
                    ScoreSource::HashControl
                } else {
                    ScoreSource::SemanticFast
                }
            } else {
                ScoreSource::Hybrid
            };
            let explanation = explanation.as_ref().map(|explanation| {
                Box::new(explanation.for_hit(&hit, score, fast_embedder_id, hash_control))
            });
            let metadata = metadata_by_doc.get(hit.doc_id.as_str()).copied().cloned();
            ScoredResult {
                doc_id: hit.doc_id,
                score,
                source,
                index: hit.semantic_index.filter(|&index| index != u32::MAX),
                fast_score: hit.hash_score.or(hit.semantic_score),
                quality_score: None,
                lexical_score: hit.lexical_score,
                rerank_score: None,
                explanation,
                metadata,
            }
        })
        .collect()
}

fn sync_fast_embedder_id(index: &InMemoryTwoTierIndex) -> &str {
    index.fast_embedder_id().unwrap_or("sync-fast-query")
}

/// Admission has already joined the query to this tier's space or complete
/// producer fingerprint. Only unidentified legacy tiers need the name heuristic.
#[must_use]
fn admitted_tier_is_hash_control(index: &InMemoryVectorIndex, query: &BoundQueryEmbedding) -> bool {
    if index.space_fingerprint_hex().is_some()
        || index
            .embedder_revision()
            .is_some_and(|revision| !revision.is_empty())
    {
        query.identity().space.kind == EmbeddingSpaceKindV1::HashControl
    } else {
        frankensearch_core::is_hash_generation_id(index.embedder_id().unwrap_or_default())
    }
}

fn sync_fast_explanation_source(
    embedder_id: &str,
    cosine_sim: f64,
    hash_control: bool,
) -> ExplainedSource {
    if hash_control {
        ExplainedSource::HashControl {
            embedder: embedder_id.to_owned(),
            cosine_sim,
        }
    } else {
        ExplainedSource::SemanticFast {
            embedder: embedder_id.to_owned(),
            cosine_sim,
        }
    }
}

fn vector_hits_to_scored_results(
    hits: &[VectorHit],
    k: usize,
    source: ScoreSource,
    fast_scores: Option<&AHashMap<&str, f32>>,
    quality_scores: Option<&AHashMap<&str, f32>>,
    config: &TwoTierConfig,
    fast_embedder_id: &str,
) -> Vec<ScoredResult> {
    let mut seen = AHashSet::with_capacity(hits.len());
    hits.iter()
        .filter(|hit| seen.insert(hit.doc_id.as_str()))
        .take(k)
        .map(|hit| {
            let fast_score = fast_scores
                .and_then(|scores| scores.get(hit.doc_id.as_str()))
                .copied()
                .or(Some(hit.score));
            let quality_score = quality_scores
                .and_then(|scores| scores.get(hit.doc_id.as_str()))
                .copied();
            let explanation = config.explain.then(|| {
                Box::new(HitExplanation {
                    final_score: f64::from(hit.score),
                    components: vec![ScoreComponent {
                        source: sync_fast_explanation_source(
                            fast_embedder_id,
                            f64::from(hit.score),
                            source == ScoreSource::HashControl,
                        ),
                        raw_score: f64::from(hit.score),
                        normalized_score: f64::from(hit.score),
                        rrf_contribution: 0.0,
                        weight: 1.0,
                    }],
                    phase: ExplanationPhase::Initial,
                    rank_movement: None,
                })
            });
            ScoredResult {
                doc_id: hit.doc_id.clone(),
                score: hit.score,
                source,
                index: Some(hit.index),
                fast_score,
                quality_score,
                lexical_score: None,
                rerank_score: None,
                explanation,
                metadata: None,
            }
        })
        .collect()
}

/// Convert blended union hits into scored results, carrying the blended score
/// and the no-index sentinel through.
///
/// The former `*_aligned_owned` variant recovered per-tier evidence through a
/// numeric `AlignedScoreLookup` keyed by fast-tier row ordinal. That was
/// correct only while the quality tier was a re-scored SUBSET of the fast pool
/// (bd-ctzo C2 retired that): under independent retrieval the pools share no
/// index space, and a quality-only document has no fast ordinal to look up.
/// Per-tier evidence is now attached by the caller from its two
/// `doc_id`-keyed maps, which is the only key both pools agree on.
fn unique_vector_hits_to_scored_results_owned(
    hits: Vec<VectorHit>,
    k: usize,
    source: ScoreSource,
) -> Vec<ScoredResult> {
    hits.into_iter()
        .take(k)
        .map(|hit| ScoredResult {
            doc_id: hit.doc_id,
            score: hit.score,
            source,
            index: (hit.index != u32::MAX).then_some(hit.index),
            fast_score: None,
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            explanation: None,
            metadata: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::ScoreSource;
    use frankensearch_core::generation::{
        EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
    };
    use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Bind one synthetic vector to an explicitly synthetic identity.
    ///
    /// `explicit_test_model` is the sanctioned constructor for exactly this:
    /// its own docs say it is "intentionally named and tagged as synthetic so
    /// it cannot be mistaken for verified semantic availability". These
    /// fixtures build vectors by hand, so a synthetic identity is the true
    /// one — the alternative would be fabricating a model name, which is what
    /// this whole surface exists to prevent.
    fn bound(model: &str, vector: Vec<f32>) -> BoundQueryEmbedding {
        let dimension = u32::try_from(vector.len()).expect("fixture dimension fits u32");
        BoundQueryEmbedding::new(
            vector,
            EmbeddingIdentityBundleV1::explicit_test_model(model, dimension),
        )
        .expect("synthetic fixture query binds")
    }

    /// The fixture indexes below build both tiers from vectors of the same
    /// synthetic space, so one bundle legitimately describes both arms. A
    /// production pair (potion fast + `MiniLM` quality) would bind two DIFFERENT
    /// bundles here, and `admit` would join each against its own tier.
    fn tiered(vector: Vec<f32>) -> TieredQueryEmbeddings {
        TieredQueryEmbeddings::progressive(
            bound("sync-fixture", vector.clone()),
            bound("sync-fixture", vector),
        )
    }

    fn fast_only_query(vector: Vec<f32>) -> TieredQueryEmbeddings {
        TieredQueryEmbeddings::fast_only(bound("sync-fixture", vector))
    }

    /// A declared semantic test fixture, with the same validated shape used
    /// by core identity tests. `explicit_test_model` alone is a hash control,
    /// regardless of the name supplied to it.
    fn semantic_identity(model: &str, dimension: u32) -> EmbeddingIdentityBundleV1 {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model, dimension);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity
            .validate()
            .expect("semantic fixture identity validates");
        identity
    }

    fn persisted_producer_pair(
        dir: &Path,
    ) -> (EmbeddingIdentityBundleV1, EmbeddingIdentityBundleV1) {
        let fast = semantic_identity("persisted-fast", 2);
        let quality = semantic_identity("persisted-quality", 3);
        let mut writer =
            frankensearch_index::TwoTierIndex::create(dir, TwoTierConfig::default()).unwrap();
        writer.set_fast_embedder_id("persisted-fast");
        writer.set_quality_embedder_id("persisted-quality");
        writer.set_fast_identity(&fast).unwrap();
        writer.set_quality_identity(&quality).unwrap();
        writer.add_fast_record("a", &[0.6, 0.8]).unwrap();
        writer.add_quality_record("a", &[0.0, 0.0, 1.0]).unwrap();
        writer.add_quality_record("b", &[1.0, 0.0, 0.0]).unwrap();
        writer.add_quality_record("c", &[0.8, 0.6, 0.0]).unwrap();
        drop(writer.finish().unwrap());
        (fast, quality)
    }

    #[test]
    fn persisted_producer_fingerprints_reject_drift_before_search() {
        struct UnreachableLexical;
        impl SyncLexicalSearch for UnreachableLexical {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                panic!("producer refusal must precede lexical retrieval");
            }
        }

        let dir = temp_dir("persisted-producer-admission");
        let (fast, quality) = persisted_producer_pair(dir.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
            .with_lexical(Arc::new(UnreachableLexical));
        for tier in ["fast", "quality"] {
            for change_space in [false, true] {
                let mut foreign_fast = fast.clone();
                let mut foreign_quality = quality.clone();
                let foreign = if tier == "fast" {
                    &mut foreign_fast
                } else {
                    &mut foreign_quality
                };
                if change_space {
                    "new-weights-same-model".clone_into(&mut foreign.space.immutable_revision);
                    foreign.producer.space_fingerprint = foreign.space.fingerprint();
                } else {
                    "foreign-backend".clone_into(&mut foreign.producer.backend);
                }
                foreign.validate().unwrap();
                for k in [0, 1] {
                    for zero in [false, true] {
                        let signal = if zero { 0.0 } else { 1.0 };
                        let query = TieredQueryEmbeddings::progressive(
                            BoundQueryEmbedding::new(vec![signal, 0.0], foreign_fast.clone())
                                .unwrap(),
                            BoundQueryEmbedding::new(
                                vec![signal, 0.0, 0.0],
                                foreign_quality.clone(),
                            )
                            .unwrap(),
                        );
                        assert_admission_refusal(
                            &searcher,
                            &query,
                            k,
                            &format!("search_activation.{tier}.producer_revision"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn persisted_quality_retrieval_reaches_outside_fast_pool_after_reopen() {
        let dir = temp_dir("persisted-quality-retrieval");
        let (fast, quality) = persisted_producer_pair(dir.path());
        // Drop and reopen every owner: no process-local builder identity can
        // supply the missing admission or the quality-only document here.
        for _ in 0..2 {
            let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
            assert!(index.fast_index().admitted_source_binding().is_none());
            assert!(
                index
                    .quality_index()
                    .unwrap()
                    .space_fingerprint_hex()
                    .is_none()
            );
            let searcher = SyncTwoTierSearcher::new(
                index,
                TwoTierConfig {
                    candidate_multiplier: 1,
                    quality_weight: 1.0,
                    ..TwoTierConfig::default()
                },
            );
            for fast_signal in [1.0, 0.0] {
                let query = TieredQueryEmbeddings::progressive(
                    BoundQueryEmbedding::new(vec![fast_signal, 0.0], fast.clone()).unwrap(),
                    BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality.clone()).unwrap(),
                );
                let phases = searcher.search_iter(&query, 1).collect::<Vec<_>>();
                let [
                    SearchPhase::Initial {
                        results: initial, ..
                    },
                    SearchPhase::Refined { results, .. },
                ] = phases.as_slice()
                else {
                    panic!("both independently bound tiers must run: {phases:?}");
                };
                if fast_signal == 0.0 {
                    assert!(initial.is_empty());
                } else {
                    assert_eq!(initial[0].doc_id, "a");
                }
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].doc_id, "b");
                assert_eq!(results[0].index, None);
                assert_eq!(results[0].fast_score, None);
                assert_eq!(results[0].quality_score, Some(1.0));
                let (filtered, metrics) = searcher
                    .search_collect_with_filter(&query, 1, Some(&ExcludeB))
                    .unwrap();
                assert_eq!(filtered[0].doc_id, "c");
                assert_eq!(filtered[0].fast_score, None);
                assert!(metrics.coverage.is_none(), "v1 does not mint v2 coverage");
                assert!(metrics.zero_signal.is_none());
            }
        }
    }

    #[test]
    fn progressive_quality_scan_obeys_explicit_parallel_policy() {
        struct ObserveQualityScan {
            caller: std::thread::ThreadId,
            caller_visits: AtomicUsize,
            worker_visits: AtomicUsize,
            exclude_b: bool,
        }

        impl SearchFilter for ObserveQualityScan {
            fn matches(&self, doc_id: &str, _: Option<&serde_json::Value>) -> bool {
                // Only the quality owner contains b/c. Worker visits prove
                // parallel retrieval; refinement admission can also validate
                // these candidates on the caller after that scan completes.
                if doc_id != "a" {
                    let visits = if std::thread::current().id() == self.caller {
                        &self.caller_visits
                    } else {
                        &self.worker_visits
                    };
                    visits.fetch_add(1, Ordering::Relaxed);
                }
                !(self.exclude_b && doc_id == "b")
            }

            fn name(&self) -> &'static str {
                "observe-quality-scan"
            }
        }

        for attested in [false, true] {
            let directory = temp_dir("sync-quality-scan-policy");
            let (index, fast, quality) = if attested {
                let (fast, fast_identity) = attested_tier(
                    directory.path(),
                    "fast.fsvi",
                    "persisted-fast",
                    1,
                    &[("a", &[0.6, 0.8])],
                );
                let (quality, quality_identity) = attested_tier(
                    directory.path(),
                    "quality.fsvi",
                    "persisted-quality",
                    1,
                    &[
                        ("a", &[0.0, 0.0, 1.0]),
                        ("b", &[1.0, 0.0, 0.0]),
                        ("c", &[0.8, 0.6, 0.0]),
                    ],
                );
                (
                    InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap(),
                    fast_identity,
                    quality_identity,
                )
            } else {
                let (fast, quality) = persisted_producer_pair(directory.path());
                (
                    InMemoryTwoTierIndex::from_dir(directory.path()).unwrap(),
                    fast,
                    quality,
                )
            };
            let index = Arc::new(index);
            let query = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
            );
            for exclude_b in [false, true] {
                let mut expected_results = None;
                for parallel in [None, Some(false), Some(true)] {
                    let mut searcher = SyncTwoTierSearcher::new(
                        Arc::clone(&index),
                        TwoTierConfig {
                            candidate_multiplier: 1,
                            quality_weight: 1.0,
                            explain: true,
                            ..TwoTierConfig::default()
                        },
                    );
                    if let Some(parallel_enabled) = parallel {
                        searcher = searcher.with_search_params(SearchParams {
                            parallel_enabled,
                            parallel_threshold: 0,
                            parallel_chunk_size: 1,
                        });
                    }
                    let observed = ObserveQualityScan {
                        caller: std::thread::current().id(),
                        caller_visits: AtomicUsize::new(0),
                        worker_visits: AtomicUsize::new(0),
                        exclude_b,
                    };
                    let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&observed));
                    let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                        panic!("Initial must precede independent quality work");
                    };
                    assert_eq!(results[0].doc_id, "a");
                    assert_eq!(observed.caller_visits.load(Ordering::Relaxed), 0);
                    assert_eq!(observed.worker_visits.load(Ordering::Relaxed), 0);
                    let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
                        panic!("independent quality retrieval must refine Initial");
                    };
                    assert_eq!(results[0].doc_id, if exclude_b { "c" } else { "b" });
                    assert!(results[0].fast_score.is_none());
                    assert!(results[0].quality_score.is_some());
                    assert!(results[0].index.is_none());
                    assert!(phases.next().is_none());
                    if parallel == Some(true) {
                        assert!(observed.worker_visits.load(Ordering::Relaxed) > 0);
                    } else {
                        assert!(observed.caller_visits.load(Ordering::Relaxed) > 0);
                        assert_eq!(observed.worker_visits.load(Ordering::Relaxed), 0);
                    }
                    let serialized = serde_json::to_value(&results).unwrap();
                    assert_eq!(
                        *expected_results.get_or_insert_with(|| serialized.clone()),
                        serialized
                    );
                    let (collected, metrics) = searcher
                        .search_collect_with_filter(&query, 1, Some(&observed))
                        .unwrap();
                    assert_eq!(serde_json::to_value(collected).unwrap(), serialized);
                    assert_eq!(metrics.phase2_vectors_searched, 2);
                    assert_eq!(metrics.incomplete_embeddings, 0);
                }
            }
        }
    }

    #[test]
    fn independent_quality_window_retains_exact_fast_candidate_evidence() {
        let dir = temp_dir("persisted-quality-evidence");
        let (fast, quality) = persisted_producer_pair(dir.path());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        for quality_weight in [0.0, 1.0] {
            let searcher = SyncTwoTierSearcher::new(
                Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap()),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    quality_weight,
                    explain: true,
                    ..TwoTierConfig::default()
                },
            );
            let (results, metrics) = searcher.search_collect(&query, 2).unwrap();
            // Quality retrieves b/c, but a has a real quality vector below
            // that top-k window. Its zero score must remain present evidence,
            // rather than becoming an unpenalized missing-source observation.
            assert_eq!(metrics.phase2_vectors_searched, 3);
            assert_eq!(metrics.incomplete_embeddings, 0);
            assert!(metrics.coverage.is_none(), "v1 cannot mint v2 coverage");
            assert!(results.iter().all(|result| result.explanation.is_some()));
            let ids = results
                .iter()
                .map(|result| result.doc_id.as_str())
                .collect::<Vec<_>>();
            if quality_weight == 0.0 {
                assert_eq!(ids, ["a", "b"]);
                assert_eq!(results[0].quality_score, Some(0.0));
                assert!(results[0].fast_score.is_some());
                assert_eq!(results[0].index, Some(0));
            } else {
                assert_eq!(ids, ["b", "c"]);
                assert!(results.iter().all(|result| result.fast_score.is_none()));
                assert!(results.iter().all(|result| result.index.is_none()));
            }
        }
    }

    #[test]
    fn quality_discoveries_receive_fast_evidence_before_union_blending() {
        let dir = temp_dir("persisted-symmetric-candidate-evidence");
        let fast = semantic_identity("persisted-fast", 2);
        let quality = semantic_identity("persisted-quality", 3);
        let mut writer =
            frankensearch_index::TwoTierIndex::create(dir.path(), TwoTierConfig::default())
                .unwrap();
        writer.set_quality_embedder_id("persisted-quality");
        writer.set_fast_identity(&fast).unwrap();
        writer.set_quality_identity(&quality).unwrap();
        for (id, fast_vector, quality_vector) in [
            ("a", [1.0, 0.0], [0.0, 1.0, 0.0]),
            ("b", [0.0, 1.0], [1.0, 0.0, 0.0]),
            ("c", [0.8, 0.6], [0.8, 0.6, 0.0]),
        ] {
            writer
                .add_record(id, &fast_vector, Some(&quality_vector))
                .unwrap();
        }
        drop(writer.finish().unwrap());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
        assert_eq!(index.quality_embedder_id(), Some("persisted-quality"));
        let searcher = SyncTwoTierSearcher::new(
            Arc::clone(&index),
            TwoTierConfig {
                candidate_multiplier: 1,
                quality_weight: 0.7,
                explain: true,
                ..TwoTierConfig::default()
            },
        );
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
        let [
            SearchPhase::Initial {
                results: initial, ..
            },
            SearchPhase::Refined {
                results,
                metrics: refined_metrics,
                ..
            },
        ] = phases.as_slice()
        else {
            panic!("expected progressive refinement: {phases:?}");
        };
        assert_eq!(
            initial
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "c"]
        );
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["c", "b"]
        );
        assert_eq!(results[1].fast_score, Some(0.0));
        assert_eq!(results[1].quality_score, Some(1.0));
        assert!(results[1].index.is_some(), "B has a real fast row");
        assert!((results[1].score - 0.7).abs() < 1e-6);
        for hit in results {
            let explanation = hit.explanation.as_ref().expect("refined explanation");
            let reconstructed = explanation
                .components
                .iter()
                .map(|component| component.normalized_score * component.weight)
                .sum::<f64>();
            assert!(
                (reconstructed - f64::from(hit.score)).abs() < 1e-7,
                "incorrect blend evidence for {}",
                hit.doc_id
            );
            assert!(explanation.components.iter().any(|component| matches!(
                &component.source,
                ExplainedSource::SemanticQuality { embedder, .. } if embedder == "persisted-quality"
            )));
        }
        assert_eq!(refined_metrics.embedder_id, "persisted-quality");
        let (_, metrics) = searcher.search_collect(&query, 2).unwrap();
        assert_eq!(metrics.semantic_candidates, 2);
        assert_eq!(metrics.phase2_vectors_searched, 3);
        assert_eq!(metrics.incomplete_embeddings, 0);
    }

    #[test]
    fn independent_quality_preserves_genuinely_missing_vectors_without_penalty() {
        let dir = temp_dir("persisted-quality-absent-vector");
        let fast = semantic_identity("persisted-fast", 2);
        let quality = semantic_identity("persisted-quality", 3);
        let mut writer =
            frankensearch_index::TwoTierIndex::create(dir.path(), TwoTierConfig::default())
                .unwrap();
        writer.set_fast_embedder_id("persisted-fast");
        writer.set_quality_embedder_id("persisted-quality");
        writer.set_fast_identity(&fast).unwrap();
        writer.set_quality_identity(&quality).unwrap();
        writer.add_fast_record("a-uncovered", &[1.0, 0.0]).unwrap();
        writer
            .add_quality_record("z-quality", &[1.0, 0.0, 0.0])
            .unwrap();
        drop(writer.finish().unwrap());
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap()),
            TwoTierConfig {
                candidate_multiplier: 1,
                quality_weight: 1.0,
                explain: true,
                ..TwoTierConfig::default()
            },
        );
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let (results, metrics) = searcher.search_collect(&query, 1).unwrap();
        assert_eq!(results[0].doc_id, "a-uncovered");
        assert_eq!(results[0].score, 1.0);
        assert_eq!(results[0].fast_score, Some(1.0));
        assert_eq!(results[0].quality_score, None);
        assert_eq!(results[0].source, ScoreSource::SemanticFast);
        assert_eq!(results[0].index, Some(0));
        assert_eq!(metrics.incomplete_embeddings, 1);
        assert_eq!(metrics.phase2_vectors_searched, 1);
        let (both, _) = searcher.search_collect(&query, 2).unwrap();
        assert_eq!(
            both.iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["a-uncovered", "z-quality"]
        );
        for hit in both {
            let explanation = hit.explanation.expect("single-tier explanation");
            let [component] = explanation.components.as_slice() else {
                panic!("expected one source per document");
            };
            assert_eq!(component.weight.to_bits(), 1.0_f64.to_bits());
            assert_eq!(component.normalized_score.to_bits(), 1.0_f64.to_bits());
        }
    }

    #[test]
    fn persisted_legacy_revision_cannot_admit_a_semantic_query() {
        for revision in ["", "old-model-revision"] {
            let dir = temp_dir("persisted-legacy-refusal");
            let path = dir.path().join("vector.fast.idx");
            let mut writer = frankensearch_index::VectorIndex::create_with_revision(
                &path,
                "persisted-fast",
                revision,
                2,
                frankensearch_index::Quantization::F16,
            )
            .unwrap();
            writer.write_record("a", &[1.0, 0.0]).unwrap();
            writer.finish().unwrap();
            let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
            let query = TieredQueryEmbeddings::fast_only(
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("persisted-fast", 2))
                    .unwrap(),
            );
            for k in [0, 1] {
                assert_admission_refusal(
                    &searcher,
                    &query,
                    k,
                    "search_activation.fast.producer_revision",
                );
            }
        }
    }

    /// An in-memory tier loaded through exact FSVI v2 admission, so it carries
    /// an ATTESTED space identity read out of the artifact's own header.
    ///
    /// This is the only way to reach the attested branch: `from_vectors` is
    /// legacy-unidentified by construction, and fabricating a fingerprint on
    /// it would make the identity join pass without proving anything.
    fn attested_tier(
        dir: &std::path::Path,
        file: &str,
        model: &str,
        sequence: u64,
        rows: &[(&str, &[f32])],
    ) -> (InMemoryVectorIndex, EmbeddingIdentityBundleV1) {
        let dimension = u32::try_from(rows[0].1.len()).expect("fixture dimension fits u32");
        let query_identity = semantic_identity(model, dimension);
        let tier = attested_tier_with_identity(dir, file, sequence, rows, &query_identity);
        (tier, query_identity)
    }

    fn attested_tier_with_identity(
        dir: &std::path::Path,
        file: &str,
        sequence: u64,
        rows: &[(&str, &[f32])],
        identity: &EmbeddingIdentityBundleV1,
    ) -> InMemoryVectorIndex {
        let mut artifact = identity.clone();
        "fsvi-v2".clone_into(&mut artifact.storage.format);
        artifact.storage.quantization = frankensearch_core::generation::QuantizationFormat::F16;
        "little-endian".clone_into(&mut artifact.storage.endianness);
        let binding = frankensearch_index::FsviV2IdentityBinding::new(
            frankensearch_core::generation::ArtifactGenerationIdentityV1::new(sequence, [0x7a; 16])
                .expect("test generation"),
            artifact.freeze().expect("freeze artifact identity"),
        )
        .expect("valid FSVI v2 binding");

        let path = dir.join(file);
        let mut writer = frankensearch_index::VectorIndex::create_v2(&path, binding.clone())
            .expect("create_v2 fixture");
        for (doc_id, vector) in rows {
            writer.write_record(doc_id, vector).expect("write v2 row");
        }
        writer.finish().expect("finish v2 fixture");

        let owner = frankensearch_index::VectorIndex::open_admitted_v2(&path, &binding)
            .expect("admit the v2 fixture");
        let tier = InMemoryVectorIndex::from_admitted_v2(&owner).expect("load admitted tier");
        assert!(
            tier.space_identity_is_attested(),
            "the fixture is only meaningful while the tier's identity is header-attested"
        );
        // The query bundle keeps native f32 storage; only the artifact uses
        // f16. Space and producer admission must tolerate that storage change.
        tier
    }

    fn sync_mmr_fixture(
        directory: &Path,
        attested: bool,
        complete_quality: bool,
    ) -> (
        Arc<InMemoryTwoTierIndex>,
        EmbeddingIdentityBundleV1,
        EmbeddingIdentityBundleV1,
    ) {
        let fast = semantic_identity("mmr-fast", 2);
        let quality = semantic_identity("mmr-quality", 3);
        let fast_rows: &[(&str, &[f32])] = &[
            ("a", &[0.8, 0.6]),
            ("b", &[0.75, 0.661_437_8]),
            ("c", &[0.6, -0.8]),
        ];
        // The quality geometry intentionally disagrees with fast, and its
        // document order differs too. Reusing fast ordinals or mixing vectors
        // across tiers changes which candidate is diverse.
        let quality_rows: &[(&str, &[f32])] = &[
            ("c", &[0.6, 0.8, 0.0]),
            ("b", &[0.75, -0.661_437_8, 0.0]),
            ("a", &[0.8, 0.6, 0.0]),
        ];
        let quality_rows = if complete_quality {
            quality_rows
        } else {
            &quality_rows[1..]
        };
        let index = if attested {
            let fast_index =
                attested_tier_with_identity(directory, "fast.fsvi", 1, fast_rows, &fast);
            let quality_index =
                attested_tier_with_identity(directory, "quality.fsvi", 1, quality_rows, &quality);
            InMemoryTwoTierIndex::new(fast_index, Some(quality_index)).unwrap()
        } else {
            let mut writer =
                frankensearch_index::TwoTierIndex::create(directory, TwoTierConfig::default())
                    .unwrap();
            writer.set_fast_embedder_id("mmr-fast");
            writer.set_quality_embedder_id("mmr-quality");
            writer.set_fast_identity(&fast).unwrap();
            writer.set_quality_identity(&quality).unwrap();
            for (id, vector) in fast_rows {
                writer.add_fast_record(*id, vector).unwrap();
            }
            for (id, vector) in quality_rows {
                writer.add_quality_record(*id, vector).unwrap();
            }
            drop(writer.finish().unwrap());
            InMemoryTwoTierIndex::from_dir(directory).unwrap()
        };
        (Arc::new(index), fast, quality)
    }

    fn sync_mmr_config() -> MmrConfig {
        MmrConfig {
            enabled: true,
            lambda: 0.3,
            candidate_pool: 3,
        }
    }

    #[test]
    fn sync_mmr_revalidates_filters_between_retrieval_and_ranking() {
        struct PhaseFilter(Mutex<Option<DocId>>);
        impl SearchFilter for PhaseFilter {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                self.0.lock().unwrap().as_deref() != Some(id)
            }

            fn name(&self) -> &'static str {
                "phase-filter"
            }
        }

        for attested in [false, true] {
            for mode in ["fast", "quality", "progressive"] {
                for revoked in [None, Some("shown"), Some("hidden")] {
                    let directory = temp_dir("mmr-phase-filter");
                    let (index, fast, quality) = sync_mmr_fixture(directory.path(), attested, true);
                    let fast = BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap();
                    let quality = BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap();
                    let query = match mode {
                        "fast" => TieredQueryEmbeddings::fast_only(fast),
                        "quality" => TieredQueryEmbeddings::quality_only(quality),
                        _ => TieredQueryEmbeddings::progressive(fast, quality),
                    };
                    let searcher = SyncTwoTierSearcher::new(
                        index,
                        TwoTierConfig {
                            candidate_multiplier: 1,
                            quality_weight: 0.0,
                            ..TwoTierConfig::default()
                        },
                    )
                    .with_mmr_config(sync_mmr_config());
                    let filter = PhaseFilter(Mutex::new(None));
                    let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
                    let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                        panic!("retrieval must precede optional diversity ranking");
                    };
                    let displayed = if mode == "progressive" {
                        let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
                            panic!("progressive retrieval must refine before ranking");
                        };
                        results
                    } else {
                        results
                    };
                    assert_eq!(displayed.len(), 1);
                    *filter.0.lock().unwrap() = revoked.map(|which| {
                        if which == "shown" {
                            displayed[0].doc_id.clone()
                        } else {
                            ["a", "b", "c"]
                                .into_iter()
                                .find(|id| *id != displayed[0].doc_id.as_str())
                                .expect("the retained pool has an undisplayed candidate")
                                .into()
                        }
                    });
                    let terminal = phases.next().expect("final ranking emits a terminal phase");
                    if revoked.is_some() {
                        let SearchPhase::RefinementFailed {
                            initial_results,
                            error,
                            ..
                        } = terminal
                        else {
                            panic!("revoked retained candidates must not be reranked");
                        };
                        assert!(initial_results.is_empty());
                        assert!(matches!(
                            error,
                            SearchError::InvalidConfig { field, value, .. }
                                if field == "search.filter" && value == "eligibility_changed"
                        ));
                    } else {
                        let SearchPhase::Reranked { results, .. } = terminal else {
                            panic!("unchanged eligibility must preserve diversity ranking");
                        };
                        let (expected, _) = searcher.search_collect(&query, 1).unwrap();
                        let (collected, _) = searcher
                            .search_collect_with_filter(&query, 1, Some(&filter))
                            .unwrap();
                        assert_eq!(
                            serde_json::to_value(&results).unwrap(),
                            serde_json::to_value(expected).unwrap()
                        );
                        assert_eq!(
                            serde_json::to_value(results).unwrap(),
                            serde_json::to_value(collected).unwrap()
                        );
                    }
                    assert!(phases.next().is_none());
                    assert!(phases.next().is_none());
                }
            }
        }
    }

    #[test]
    fn sync_mmr_rechecks_earlier_admissions_before_returning_results() {
        use std::sync::atomic::AtomicBool;

        struct LateRevocation {
            armed: AtomicBool,
            revoked: AtomicBool,
        }
        impl SearchFilter for LateRevocation {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                if id == "c" && self.armed.load(Ordering::Relaxed) {
                    self.revoked.store(true, Ordering::Relaxed);
                }
                id != "a" || !self.revoked.load(Ordering::Relaxed)
            }

            fn name(&self) -> &'static str {
                "late-final-revocation"
            }
        }

        let directory = temp_dir("mmr-final-recheck");
        let (index, fast, _) = sync_mmr_fixture(directory.path(), true, true);
        let query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
        );
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
            .with_mmr_config(sync_mmr_config());
        let filter = LateRevocation {
            armed: AtomicBool::new(false),
            revoked: AtomicBool::new(false),
        };
        let mut phases = searcher.search_iter_with_filter(&query, 2, Some(&filter));
        let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
            panic!("Initial must precede final ranking");
        };
        assert_eq!(results[0].doc_id, "a");
        assert!(!filter.revoked.load(Ordering::Relaxed));
        // Checking a later retained candidate simulates a policy update after
        // the earlier candidate was admitted. Final output must be rechecked.
        filter.armed.store(true, Ordering::Relaxed);
        let Some(SearchPhase::RefinementFailed {
            initial_results,
            error,
            ..
        }) = phases.next()
        else {
            panic!("a final result must not reuse an earlier eligibility decision");
        };
        assert!(filter.revoked.load(Ordering::Relaxed));
        assert!(initial_results.is_empty());
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, value, .. }
                if field == "search.filter" && value == "eligibility_changed"
        ));
        assert!(phases.next().is_none());
    }

    #[test]
    fn sync_mmr_promotes_an_undisplayed_fast_candidate_after_initial() {
        for attested in [false, true] {
            for mode in ["absent", "disabled", "zero"] {
                let directory = temp_dir("mmr-fast-pool");
                let (index, fast, quality) = sync_mmr_fixture(directory.path(), attested, true);
                let fast_query = BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap();
                let query = if mode == "absent" {
                    TieredQueryEmbeddings::fast_only(fast_query)
                } else {
                    TieredQueryEmbeddings::progressive(
                        fast_query,
                        BoundQueryEmbedding::new(
                            if mode == "zero" {
                                vec![0.0; 3]
                            } else {
                                vec![1.0, 0.0, 0.0]
                            },
                            quality,
                        )
                        .unwrap(),
                    )
                };
                let config = TwoTierConfig {
                    fast_only: mode == "disabled",
                    candidate_multiplier: 1,
                    explain: true,
                    ..TwoTierConfig::default()
                };
                let baseline = SyncTwoTierSearcher::new(Arc::clone(&index), config.clone());
                let (original, _) = baseline.search_collect(&query, 2).unwrap();
                let searcher =
                    SyncTwoTierSearcher::new(index, config).with_mmr_config(sync_mmr_config());

                let mut stopped = searcher.search_iter(&query, 2);
                let Some(SearchPhase::Initial { results, .. }) = stopped.next() else {
                    panic!("Initial must precede optional diversity");
                };
                assert_eq!(results[0].doc_id, "a");
                assert_eq!(results[1].doc_id, "b");
                assert_eq!(
                    serde_json::to_value(&results).unwrap(),
                    serde_json::to_value(original).unwrap(),
                    "Initial keeps retrieval order and evidence"
                );
                assert_eq!(stopped.size_hint(), (0, Some(1)));
                drop(stopped);

                let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
                let [
                    SearchPhase::Initial { results: first, .. },
                    SearchPhase::Reranked {
                        results: final_results,
                        metrics,
                        ..
                    },
                ] = phases.as_slice()
                else {
                    panic!("{mode} must rank Initial without a fabricated quality phase");
                };
                assert_eq!(
                    serde_json::to_value(first).unwrap(),
                    serde_json::to_value(results).unwrap()
                );
                assert_eq!(final_results[0].doc_id, "a");
                assert_eq!(final_results[1].doc_id, "c");
                assert_eq!(metrics.embedder_id, "mmr");
                assert_eq!(metrics.vectors_searched, 3);
                assert!(metrics.skip_reason.is_some());
                assert!(final_results.iter().all(|hit| {
                    hit.source == ScoreSource::SemanticFast
                        && hit.quality_score.is_none()
                        && hit.rerank_score.is_none()
                }));
                let (collected, metrics) = searcher.search_collect(&query, 2).unwrap();
                assert_eq!(
                    serde_json::to_value(collected).unwrap(),
                    serde_json::to_value(final_results).unwrap()
                );
                assert_eq!(metrics.phase2_vectors_searched, 0);
                assert!(metrics.skip_reason.is_some());
            }
        }
    }

    #[test]
    fn sync_mmr_uses_one_complete_space_after_quality_refinement() {
        for attested in [false, true] {
            for complete_quality in [false, true] {
                let directory = temp_dir("mmr-refined-space");
                let (index, fast, quality) =
                    sync_mmr_fixture(directory.path(), attested, complete_quality);
                let query = TieredQueryEmbeddings::progressive(
                    BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
                    BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
                );
                let searcher = SyncTwoTierSearcher::new(
                    index,
                    TwoTierConfig {
                        candidate_multiplier: 1,
                        quality_weight: 0.0,
                        explain: true,
                        ..TwoTierConfig::default()
                    },
                )
                .with_mmr_config(sync_mmr_config());
                let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
                let [
                    SearchPhase::Initial { .. },
                    SearchPhase::Refined {
                        results: refined, ..
                    },
                    SearchPhase::Reranked {
                        results, metrics, ..
                    },
                ] = phases.as_slice()
                else {
                    panic!("quality retrieval and final ranking must stay separate");
                };
                assert_eq!(refined[0].doc_id, "a");
                assert_eq!(refined[1].doc_id, "b");
                assert_eq!(results[0].doc_id, "a");
                assert_eq!(results[1].doc_id, if complete_quality { "b" } else { "c" });
                assert_eq!(metrics.embedder_id, "mmr");
                assert_eq!(results[1].quality_score.is_some(), complete_quality);
                assert!(results.iter().all(|result| result.rerank_score.is_none()));
                let movement = results[1]
                    .explanation
                    .as_ref()
                    .unwrap()
                    .rank_movement
                    .as_ref()
                    .unwrap();
                assert_eq!(movement.refined_rank, 1);
                assert_eq!(movement.delta, if complete_quality { 0 } else { -1 });
                let (collected, metrics) = searcher.search_collect(&query, 2).unwrap();
                assert_eq!(
                    serde_json::to_value(collected).unwrap(),
                    serde_json::to_value(results).unwrap()
                );
                assert_eq!(
                    metrics.incomplete_embeddings,
                    usize::from(!complete_quality)
                );
            }
        }
    }

    #[test]
    fn sync_mmr_quality_only_preserves_quality_provenance_and_never_falls_back_to_fast() {
        for attested in [false, true] {
            let directory = temp_dir("mmr-direct-quality");
            let (index, _, quality) = sync_mmr_fixture(directory.path(), attested, true);
            let query = TieredQueryEmbeddings::quality_only(
                BoundQueryEmbedding::new(vec![0.0, 1.0, 0.0], quality).unwrap(),
            );
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
                .with_mmr_config(sync_mmr_config());
            let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
            let [
                SearchPhase::Initial { results: first, .. },
                SearchPhase::Reranked { results, .. },
            ] = phases.as_slice()
            else {
                panic!("quality-only diversity must not invent fast refinement");
            };
            assert_eq!(
                first
                    .iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["c", "a"]
            );
            assert_eq!(
                results
                    .iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                ["c", "b"]
            );
            assert!(
                results
                    .iter()
                    .all(|hit| hit.source == ScoreSource::SemanticQuality
                        && hit.index.is_none()
                        && hit.fast_score.is_none()
                        && hit.quality_score.is_some())
            );
            let (collected, metrics) = searcher.search_collect(&query, 2).unwrap();
            assert_eq!(
                serde_json::to_value(collected).unwrap(),
                serde_json::to_value(results).unwrap()
            );
            assert_eq!(metrics.quality_embedder_id.as_deref(), Some("mmr-quality"));

            let partial_dir = temp_dir("mmr-direct-partial");
            let (index, _, quality) = sync_mmr_fixture(partial_dir.path(), attested, false);
            let query = TieredQueryEmbeddings::quality_only(
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
            );
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
                .with_lexical(Arc::new(StaticLexical {
                    hits: vec![lexical_result("c", 20.0)],
                }))
                .with_mmr_config(sync_mmr_config());
            let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
            let [SearchPhase::Initial { results, .. }] = phases.as_slice() else {
                panic!("a lexical candidate without quality coverage cannot license fast MMR");
            };
            assert!(results.iter().any(|hit| hit.doc_id == "c"));
            let (collected, _) = searcher.search_collect(&query, 2).unwrap();
            assert_eq!(
                serde_json::to_value(collected).unwrap(),
                serde_json::to_value(results).unwrap()
            );
        }
    }

    #[test]
    fn sync_mmr_disjoint_coverage_keeps_refined_results_and_respects_filters() {
        struct ExcludeQualityOnly;
        impl SearchFilter for ExcludeQualityOnly {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                id != "quality-only"
            }
            fn name(&self) -> &'static str {
                "exclude-quality-only"
            }
        }
        let directory = temp_dir("mmr-disjoint-coverage");
        let (fast, fast_identity) = attested_tier(
            directory.path(),
            "fast.fsvi",
            "disjoint-fast",
            1,
            &[("a", &[1.0, 0.0]), ("fast-only", &[0.8, 0.6])],
        );
        let (quality, quality_identity) = attested_tier(
            directory.path(),
            "quality.fsvi",
            "disjoint-quality",
            1,
            &[("a", &[1.0, 0.0, 0.0]), ("quality-only", &[0.8, 0.6, 0.0])],
        );
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality_identity).unwrap(),
        );
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
            .with_mmr_config(sync_mmr_config());
        let phases = searcher.search_iter(&query, 3).collect::<Vec<_>>();
        let [
            SearchPhase::Initial { .. },
            SearchPhase::Refined { results, .. },
        ] = phases.as_slice()
        else {
            panic!("disjoint tiers must not yield a mixed-space diversity phase");
        };
        assert_eq!(results.len(), 3);
        let (collected, _) = searcher.search_collect(&query, 3).unwrap();
        assert_eq!(
            serde_json::to_value(collected).unwrap(),
            serde_json::to_value(results).unwrap()
        );
        let phases = searcher
            .search_iter_with_filter(&query, 3, Some(&ExcludeQualityOnly))
            .collect::<Vec<_>>();
        let [
            SearchPhase::Initial { .. },
            SearchPhase::Refined { .. },
            SearchPhase::Reranked { results, .. },
        ] = phases.as_slice()
        else {
            panic!("the filtered candidate set has complete fast coverage");
        };
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.doc_id != "quality-only"));
        let (collected, _) = searcher
            .search_collect_with_filter(&query, 3, Some(&ExcludeQualityOnly))
            .unwrap();
        assert_eq!(
            serde_json::to_value(collected).unwrap(),
            serde_json::to_value(results).unwrap()
        );
    }

    #[test]
    fn sync_mmr_bounds_identity_and_disabled_controls_precede_optional_work() {
        struct UnreachableWork;
        impl SyncLexicalSearch for UnreachableWork {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                panic!("admission must precede lexical work");
            }
        }
        impl SearchFilter for UnreachableWork {
            fn matches(&self, _: &str, _: Option<&serde_json::Value>) -> bool {
                panic!("admission must precede vector work");
            }
            fn name(&self) -> &'static str {
                "unreachable-mmr-work"
            }
        }
        let directory = temp_dir("mmr-admission");
        let (index, fast, _) = sync_mmr_fixture(directory.path(), true, true);
        let query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast.clone()).unwrap(),
        );
        let invalid = SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default())
            .with_lexical(Arc::new(UnreachableWork))
            .with_mmr_config(MmrConfig {
                candidate_pool: 1001,
                ..sync_mmr_config()
            });
        for k in [0, 1] {
            assert_admission_refusal(&invalid, &query, k, "mmr.candidate_pool");
            assert!(matches!(
                invalid.search_collect_with_filter(&query, k, Some(&UnreachableWork)),
                Err(SearchError::InvalidConfig { .. })
            ));
        }
        let mut foreign = fast;
        "foreign-compatible-producer".clone_into(&mut foreign.producer.backend);
        let foreign_query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], foreign).unwrap(),
        );
        assert_admission_refusal(
            &invalid,
            &foreign_query,
            0,
            "search_activation.fast.producer_conformance",
        );

        let baseline = SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default());
        let (expected, _) = baseline.search_collect(&query, 3).unwrap();
        for config in [
            MmrConfig {
                enabled: false,
                candidate_pool: usize::MAX,
                ..sync_mmr_config()
            },
            MmrConfig {
                candidate_pool: 0,
                ..sync_mmr_config()
            },
            MmrConfig {
                candidate_pool: 1,
                ..sync_mmr_config()
            },
        ] {
            let disabled = SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default())
                .with_mmr_config(config);
            let phases = disabled.search_iter(&query, 3).collect::<Vec<_>>();
            assert!(matches!(phases.as_slice(), [SearchPhase::Initial { .. }]));
            let (results, metrics) = disabled.search_collect(&query, 3).unwrap();
            assert_eq!(
                serde_json::to_value(results).unwrap(),
                serde_json::to_value(&expected).unwrap()
            );
            assert_eq!(metrics.rerank_ms, 0.0);
        }
        let enabled =
            SyncTwoTierSearcher::new(index, TwoTierConfig::default()).with_mmr_config(MmrConfig {
                candidate_pool: 2,
                ..sync_mmr_config()
            });
        let (results, _) = enabled.search_collect(&query, 3).unwrap();
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(enabled.search_iter(&query, 0).count(), 0);
    }

    fn assert_admission_refusal(
        searcher: &SyncTwoTierSearcher,
        query: &TieredQueryEmbeddings,
        k: usize,
        expected_field: &str,
    ) {
        let error = searcher
            .search_collect(query, k)
            .expect_err("query must be refused");
        assert!(
            matches!(error, SearchError::InvalidConfig { ref field, .. } if field == expected_field),
            "expected {expected_field}, got {error:?}"
        );
        let phases = searcher.search_iter(query, k).collect::<Vec<_>>();
        let [
            SearchPhase::RefinementFailed {
                initial_results,
                error: SearchError::InvalidConfig { field, .. },
                ..
            },
        ] = phases.as_slice()
        else {
            panic!("admission refusal must reach streaming callers without an Initial phase");
        };
        assert!(initial_results.is_empty());
        assert_eq!(field, expected_field);
    }

    fn quality_only_index(
        directory: &Path,
        attested: bool,
    ) -> (Arc<InMemoryTwoTierIndex>, EmbeddingIdentityBundleV1) {
        if !attested {
            let (_, quality) = persisted_producer_pair(directory);
            return (
                Arc::new(InMemoryTwoTierIndex::from_dir(directory).unwrap()),
                quality,
            );
        }
        let (fast, _) = attested_tier(
            directory,
            "fast.fsvi",
            "unused-fast",
            1,
            &[("fast-only", &[1.0, 0.0])],
        );
        let (quality, identity) = attested_tier(
            directory,
            "quality.fsvi",
            "direct-quality",
            1,
            &[("b", &[1.0, 0.0, 0.0]), ("c", &[0.8, 0.6, 0.0])],
        );
        (
            Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap()),
            identity,
        )
    }

    #[test]
    fn quality_only_sync_retrieves_quality_without_reading_fast() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct QualityTraversal(AtomicUsize);
        impl SearchFilter for QualityTraversal {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                assert_ne!(
                    id, "fast-only",
                    "quality-only must never traverse fast rows"
                );
                self.0.fetch_add(1, Ordering::SeqCst);
                true
            }

            fn name(&self) -> &'static str {
                "quality-traversal"
            }
        }
        for attested in [false, true] {
            let directory = temp_dir("quality-only-positive");
            let (index, identity) = quality_only_index(directory.path(), attested);
            let query = TieredQueryEmbeddings::quality_only(
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], identity).unwrap(),
            );
            for hybrid in [false, true] {
                let mut searcher = SyncTwoTierSearcher::new(
                    Arc::clone(&index),
                    TwoTierConfig {
                        quality_weight: 0.0,
                        candidate_multiplier: 1,
                        explain: true,
                        ..TwoTierConfig::default()
                    },
                );
                if hybrid {
                    let mut hit = lexical_result("b", 4.0);
                    hit.metadata = Some(Arc::new(serde_json::json!({"snapshot": "retained"})));
                    searcher = searcher.with_lexical(Arc::new(StaticLexical { hits: vec![hit] }));
                }
                let observed = QualityTraversal(AtomicUsize::new(0));
                let unused = searcher.search_iter_with_filter(&query, 1, Some(&observed));
                assert_eq!(unused.size_hint(), (1, Some(1)));
                drop(unused);
                assert_eq!(observed.0.load(Ordering::SeqCst), 0);

                let phases = searcher
                    .search_iter_with_filter(&query, 1, Some(&observed))
                    .collect::<Vec<_>>();
                let [
                    SearchPhase::Initial {
                        results, metrics, ..
                    },
                ] = phases.as_slice()
                else {
                    panic!("direct quality yields one terminal Initial: {phases:?}");
                };
                assert!(observed.0.load(Ordering::SeqCst) > 0);
                assert_eq!(metrics.embedder_id, index.quality_embedder_id().unwrap());
                assert_eq!(
                    metrics.vectors_searched,
                    index.quality_index().unwrap().record_count()
                );
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].doc_id, "b");
                assert_eq!(results[0].index, None);
                assert_eq!(results[0].fast_score, None);
                assert_eq!(results[0].quality_score, Some(1.0));
                assert_eq!(
                    results[0].source,
                    if hybrid {
                        ScoreSource::Hybrid
                    } else {
                        ScoreSource::SemanticQuality
                    }
                );
                let explanation = results[0].explanation.as_ref().unwrap();
                assert!(explanation.rank_movement.is_none());
                assert!(explanation.components.iter().any(|component| matches!(
                    component.source,
                    ExplainedSource::SemanticQuality { .. }
                )));
                assert!(!explanation.components.iter().any(|component| matches!(
                    component.source,
                    ExplainedSource::SemanticFast { .. } | ExplainedSource::HashControl { .. }
                )));
                if hybrid {
                    assert_eq!(
                        results[0].metadata.as_ref().unwrap()["snapshot"],
                        "retained"
                    );
                }
                let (collected, metrics) = searcher.search_collect(&query, 1).unwrap();
                assert_eq!(collected[0].doc_id, "b");
                assert_eq!(collected[0].score.to_bits(), results[0].score.to_bits());
                assert_eq!(metrics.fast_embedder_id, None);
                assert_eq!(
                    metrics.quality_embedder_id.as_deref(),
                    index.quality_embedder_id()
                );
                assert_eq!(metrics.phase2_vectors_searched, 0);
                assert_eq!(metrics.phase2_total_ms, 0.0);
                assert_eq!(metrics.blend_ms, 0.0);
                assert_eq!(metrics.rank_changes.total(), 0);
                assert_eq!(metrics.semantic_candidates, 1);
            }
        }
    }

    struct UnreachableQualityWork;

    impl SyncLexicalSearch for UnreachableQualityWork {
        fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
            panic!("quality admission must precede lexical retrieval");
        }
    }

    impl SearchFilter for UnreachableQualityWork {
        fn matches(&self, _: &str, _: Option<&serde_json::Value>) -> bool {
            panic!("quality admission must precede vector traversal");
        }

        fn name(&self) -> &'static str {
            "unreachable-quality-work"
        }
    }

    #[test]
    fn quality_only_sync_admission_precedes_all_work() {
        for attested in [false, true] {
            let directory = temp_dir("quality-only-admission");
            let (index, identity) = quality_only_index(directory.path(), attested);
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
                .with_lexical(Arc::new(UnreachableQualityWork));
            for producer_change in [false, true] {
                let mut foreign = identity.clone();
                if producer_change {
                    "foreign-implementation".clone_into(&mut foreign.producer.backend);
                } else {
                    "foreign-weights".clone_into(&mut foreign.space.immutable_revision);
                    foreign.producer.space_fingerprint = foreign.space.fingerprint();
                }
                for k in [0, 1] {
                    for signal in [0.0, 1.0] {
                        let query = TieredQueryEmbeddings::quality_only(
                            BoundQueryEmbedding::new(vec![signal, 0.0, 0.0], foreign.clone())
                                .unwrap(),
                        );
                        assert!(matches!(
                            searcher.search_collect_with_filter(&query, k, Some(&UnreachableQualityWork)),
                            Err(SearchError::InvalidConfig { field, .. }) if field.contains("quality")
                        ));
                        let phases = searcher
                            .search_iter_with_filter(&query, k, Some(&UnreachableQualityWork))
                            .collect::<Vec<_>>();
                        assert!(
                            matches!(phases.as_slice(), [SearchPhase::RefinementFailed { initial_results, error: SearchError::InvalidConfig { field, .. }, .. }] if initial_results.is_empty() && field.contains("quality"))
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn quality_only_sync_refuses_missing_unidentified_control_and_disabled_tiers() {
        let raw = || {
            InMemoryVectorIndex::from_vectors(vec!["raw".to_owned()], vec![vec![1.0, 0.0]], 2)
                .unwrap()
        };
        let semantic = semantic_identity("declared-quality", 2);
        let query = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], semantic.clone()).unwrap(),
        );
        let absent = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(raw(), None).unwrap()),
            TwoTierConfig::default(),
        )
        .with_lexical(Arc::new(UnreachableQualityWork));
        assert_admission_refusal(&absent, &query, 0, "sync_search.topology");
        let declared = InMemoryVectorIndex::from_vectors_with_identity(
            vec!["declared".to_owned()],
            vec![vec![1.0, 0.0]],
            2,
            &semantic.space,
        )
        .unwrap();
        let unidentified = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(raw(), Some(declared)).unwrap()),
            TwoTierConfig::default(),
        )
        .with_lexical(Arc::new(UnreachableQualityWork));
        assert_admission_refusal(
            &unidentified,
            &query,
            0,
            "search_activation.quality.producer_identity",
        );

        let directory = temp_dir("quality-only-control");
        let control = EmbeddingIdentityBundleV1::explicit_test_model("explicit-control", 2);
        let fast = attested_tier_with_identity(
            directory.path(),
            "fast.fsvi",
            1,
            &[("control", &[1.0, 0.0])],
            &control,
        );
        let quality = attested_tier_with_identity(
            directory.path(),
            "quality.fsvi",
            1,
            &[("control", &[1.0, 0.0])],
            &control,
        );
        let control_searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap()),
            TwoTierConfig::default(),
        )
        .with_lexical(Arc::new(UnreachableQualityWork));
        let control_query = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], control).unwrap(),
        );
        assert_admission_refusal(
            &control_searcher,
            &control_query,
            0,
            "sync_search.quality_kind",
        );

        let directory = temp_dir("quality-only-disabled");
        let (index, identity) = quality_only_index(directory.path(), true);
        let disabled = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                fast_only: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(UnreachableQualityWork));
        let query = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], identity).unwrap(),
        );
        assert_admission_refusal(&disabled, &query, 0, "sync_search.topology");
    }

    #[test]
    fn quality_only_sync_zero_empty_and_filtered_queries_report_the_actual_lane() {
        let directory = temp_dir("quality-only-empty");
        let identity = semantic_identity("empty-quality", 3);
        let fast = attested_tier_with_identity(
            directory.path(),
            "fast.fsvi",
            1,
            &[("fast-only", &[1.0, 0.0, 0.0])],
            &identity,
        );
        let quality =
            attested_tier_with_identity(directory.path(), "quality.fsvi", 1, &[], &identity);
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap()),
            TwoTierConfig::default(),
        );
        let query = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], identity).unwrap(),
        );
        let (results, metrics) = searcher.search_collect(&query, 4).unwrap();
        assert!(results.is_empty());
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::NewlyCreatedEmpty)
        );
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert!(
            matches!(searcher.search_iter(&query, 4).collect::<Vec<_>>().as_slice(), [SearchPhase::Initial { results, .. }] if results.is_empty())
        );
        let (_, metrics) = searcher.search_collect(&query, 0).unwrap();
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::CallerRequestedZeroK)
        );
        assert!(searcher.search_iter(&query, 0).next().is_none());

        let directory = temp_dir("quality-only-zero");
        let (index, identity) = quality_only_index(directory.path(), true);
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default()).with_lexical(
            Arc::new(StaticLexical {
                hits: vec![lexical_result("lexical-only", 2.0)],
            }),
        );
        let zero = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![0.0; 3], identity.clone()).unwrap(),
        );
        let (results, metrics) = searcher.search_collect(&zero, 4).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, ScoreSource::Lexical);
        assert_eq!(results[0].fast_score, None);
        assert_eq!(results[0].quality_score, None);
        assert_eq!(metrics.zero_signal, Some(ZeroSignalReason::ZeroNormQuery));
        assert_eq!(metrics.phase1_vectors_searched, 0);
        let query = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], identity).unwrap(),
        );
        let reject = frankensearch_core::filter::PredicateFilter::new("reject-all", |_| false);
        let (results, metrics) = searcher
            .search_collect_with_filter(&query, 4, Some(&reject))
            .unwrap();
        assert!(results.is_empty());
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::FilterEliminatedAll)
        );
        assert_eq!(metrics.semantic_candidates, 0);
    }

    #[test]
    fn quality_only_sync_lexical_cancellation_precedes_vector_work() {
        struct CancelledLexical;
        impl SyncLexicalSearch for CancelledLexical {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                Err(SearchError::Cancelled {
                    phase: "quality-only-lexical".to_owned(),
                    reason: "caller cancelled".to_owned(),
                })
            }
        }
        let directory = temp_dir("quality-only-cancelled");
        let (index, identity) = quality_only_index(directory.path(), true);
        let query = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], identity).unwrap(),
        );
        for enabled in [false, true] {
            let searcher = SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default())
                .with_lexical(Arc::new(CancelledLexical))
                .with_mmr_config(MmrConfig {
                    enabled,
                    ..sync_mmr_config()
                });
            assert!(matches!(
                searcher.search_collect_with_filter(&query, 1, Some(&UnreachableQualityWork)),
                Err(SearchError::Cancelled { .. })
            ));
            let phases = searcher
                .search_iter_with_filter(&query, 1, Some(&UnreachableQualityWork))
                .collect::<Vec<_>>();
            assert!(
                matches!(phases.as_slice(), [SearchPhase::RefinementFailed { initial_results, error: SearchError::Cancelled { .. }, .. }] if initial_results.is_empty())
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn admitted_v2_default_product_preserves_exact_result_across_cache_states() {
        // The first record fixes the global int8 scale at one. The remaining
        // tiny orthogonal rows and the final 0.003 f16 row then all quantize to the same
        // int8 pass-1 score for query [0, 1]. The approximate candidate route
        // cannot retain the final exact winner. The default product scan must
        // return that winner both with an attached sidecar and with a hostile
        // unavailable cache: the sidecar may accelerate the exact route, but
        // cache state must not select a different algorithm. This is a route
        // proof by an observable result, not a private-field assertion.
        const CORPUS_ROWS: usize = 10_002;
        let dir = temp_dir("residual-default-product-route");
        let source_path = dir.path().join("source.fsvi");
        let cache_dir = dir.path().join("residual-cache");
        let unavailable_cache = dir.path().join("residual-cache-is-a-file");
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&unavailable_cache)
            .expect("create a non-directory cache obstacle without replacement");

        let mut artifact = EmbeddingIdentityBundleV1::explicit_test_model("residual-route", 2);
        "fsvi-v2".clone_into(&mut artifact.storage.format);
        artifact.storage.quantization = frankensearch_core::generation::QuantizationFormat::F16;
        "little-endian".clone_into(&mut artifact.storage.endianness);
        let binding = frankensearch_index::FsviV2IdentityBinding::new(
            frankensearch_core::generation::ArtifactGenerationIdentityV1::new(73, [0x5e; 16])
                .expect("test generation"),
            artifact.freeze().expect("freeze test identity"),
        )
        .expect("valid FSVI-v2 binding");
        let mut writer = frankensearch_index::VectorIndex::create_v2(&source_path, binding.clone())
            .expect("create admitted-v2 source");
        writer
            .write_record("scale", &[1.0, 0.0])
            .expect("write global scale row");
        for row in 0..CORPUS_ROWS - 2 {
            writer
                .write_record(&format!("zero-{row:05}"), &[0.003, 0.0])
                .expect("write orthogonal candidate");
        }
        writer
            .write_record("exact-winner", &[0.0, 0.003])
            .expect("write exact-only winner");
        writer.finish().expect("seal admitted-v2 source");
        let owner = frankensearch_index::VectorIndex::open_admitted_v2(&source_path, &binding)
            .expect("admit immutable source");
        let query_vector = vec![0.0, 1.0];

        let approximate = InMemoryVectorIndex::from_admitted_v2(&owner)
            .expect("load independently for negative control")
            .search_top_k_int8_two_pass(&query_vector, 1, 3)
            .expect("run approximate negative control");
        assert_ne!(
            approximate[0].doc_id, "exact-winner",
            "the all-tied int8 candidate pass must demonstrate why this route needs exact selection"
        );

        let config = TwoTierConfig {
            fast_only: true,
            ..TwoTierConfig::default()
        };
        let query_identity = EmbeddingIdentityBundleV1::explicit_test_model("residual-route", 2);
        assert!(
            CORPUS_ROWS >= SearchParams::default().parallel_threshold,
            "the product fixture must remain in the default parallel target scale"
        );
        let unavailable = SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
            &owner,
            &unavailable_cache,
            None,
            config.clone(),
        )
        .expect("an unavailable cache preserves the shipping exact product");
        assert!(
            !unavailable.index.fast_index().has_exact_residual_sidecar(),
            "the regular-file cache obstacle must leave no sidecar attached"
        );
        let unavailable_query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(query_vector.clone(), query_identity.clone())
                .expect("bind exact unavailable-cache query identity"),
        );
        let (unavailable_results, _) = unavailable
            .search_collect(&unavailable_query, 1)
            .expect("unavailable-cache default product search");
        assert_eq!(
            unavailable_results[0].doc_id, "exact-winner",
            "cache failure must retain the same exact winner rather than selecting int8 candidates"
        );

        std::fs::create_dir(&cache_dir).expect("create fresh immutable cache directory");
        let searcher = SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
            &owner,
            &cache_dir,
            None,
            config.clone(),
        )
        .expect("construct shipping sync product through the sidecar cache route");
        assert!(
            searcher.search_params.is_none(),
            "the regression exercises the default product search selection"
        );
        let query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(query_vector.clone(), query_identity.clone())
                .expect("bind exact query identity"),
        );
        let (results, _) = searcher
            .search_collect(&query, 1)
            .expect("default product search");
        assert_eq!(
            results[0].doc_id, "exact-winner",
            "an attached sidecar preserves the same shipping exact result"
        );
        let entry_count_after_publish = std::fs::read_dir(&cache_dir)
            .expect("read immutable cache directory")
            .flatten()
            .count();
        assert_eq!(
            entry_count_after_publish, 1,
            "exactly one sidecar was published"
        );

        let reopened = SyncTwoTierSearcher::from_admitted_v2_with_residual_sidecar_cache(
            &owner, &cache_dir, None, config,
        )
        .expect("reopen shipping sync product through the existing cache entry");
        let reopened_query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(query_vector, query_identity).expect("bind reopened query"),
        );
        let (reopened_results, _) = reopened
            .search_collect(&reopened_query, 1)
            .expect("reopened default product search");
        assert_eq!(reopened_results[0].doc_id, "exact-winner");
        assert_eq!(
            std::fs::read_dir(&cache_dir)
                .expect("read reused immutable cache directory")
                .flatten()
                .count(),
            entry_count_after_publish,
            "reopening selects the generation-matched artifact without overwriting or publishing another"
        );
    }

    /// A uniquely named temporary directory (mkdtemp reserves the name
    /// atomically), removed with its contents when the returned guard drops —
    /// at the end of the test, panics included.
    fn temp_dir(label: &str) -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("frankensearch-sync-dbp10-{label}-"))
            .tempdir()
            .expect("create unique temp dir")
    }

    fn make_index() -> Arc<InMemoryTwoTierIndex> {
        let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let fast_vectors = vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]];
        let quality_vectors = vec![vec![0.2, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]];
        let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vectors, 2).unwrap();
        let quality = InMemoryVectorIndex::from_vectors(doc_ids, quality_vectors, 2).unwrap();
        Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"))
    }

    fn lexical_result(doc_id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: None,
            metadata: None,
        }
    }

    struct StaticLexical {
        hits: Vec<ScoredResult>,
    }

    impl SyncLexicalSearch for StaticLexical {
        fn search_sync(&self, _query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
            Ok(self.hits.iter().take(limit).cloned().collect())
        }
    }

    struct UnavailableLexical {
        calls: std::sync::atomic::AtomicUsize,
        fail_after_first: bool,
    }

    impl SyncLexicalSearch for UnavailableLexical {
        fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
            let previous = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail_after_first && previous == 0 {
                // A full prefix with no eligible result forces another fetch.
                return Ok((0..limit)
                    .map(|rank| lexical_result(&format!("excluded-{rank}"), 1.0))
                    .collect());
            }
            Err(SearchError::SubsystemError {
                subsystem: "test.lexical",
                source: "private-lexical-query secret-token\nforged-log-line".into(),
            })
        }
    }

    #[test]
    #[tracing_test::traced_test]
    fn lexical_backend_failures_preserve_sync_semantic_results() {
        use frankensearch_core::filter::BitsetFilter;
        use std::sync::atomic::{AtomicUsize, Ordering};

        for fast_only in [false, true] {
            for fail_after_first in [false, true] {
                let config = TwoTierConfig {
                    fast_only,
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                };
                let index = make_index();
                let baseline = SyncTwoTierSearcher::new(Arc::clone(&index), config.clone());
                let lexical = Arc::new(UnavailableLexical {
                    calls: AtomicUsize::new(0),
                    fail_after_first,
                });
                let searcher =
                    SyncTwoTierSearcher::new(index, config).with_lexical(lexical.clone());
                let query = tiered(vec![1.0, 0.0]);
                let allowed = BitsetFilter::from_doc_ids(["a", "c"]);
                let (expected, expected_metrics) = baseline
                    .search_collect_with_filter(&query, 2, Some(&allowed))
                    .unwrap();
                let (actual, metrics) = searcher
                    .search_collect_with_filter(&query, 2, Some(&allowed))
                    .expect("optional lexical failure must retain the semantic answer");
                assert_eq!(
                    serde_json::to_value(&actual).unwrap(),
                    serde_json::to_value(&expected).unwrap(),
                );
                assert!(!actual.is_empty());
                assert!(actual.iter().all(|hit| hit.doc_id != "b"));
                assert_eq!(metrics.lexical_candidates, 0);
                assert_eq!(
                    metrics.semantic_candidates,
                    expected_metrics.semantic_candidates
                );
                assert_eq!(
                    lexical.calls.load(Ordering::SeqCst),
                    if fail_after_first { 2 } else { 1 }
                );

                lexical.calls.store(0, Ordering::SeqCst);
                let phases = searcher.search_iter_with_filter(&query, 2, Some(&allowed));
                assert_eq!(
                    lexical.calls.load(Ordering::SeqCst),
                    0,
                    "iteration stays lazy"
                );
                let phases = phases.collect::<Vec<_>>();
                assert_eq!(phases.len(), if fast_only { 1 } else { 2 });
                assert!(matches!(phases.first(), Some(SearchPhase::Initial { .. })));
                let final_results = match phases.last().unwrap() {
                    SearchPhase::Initial { results, .. } | SearchPhase::Refined { results, .. } => {
                        results
                    }
                    other => {
                        panic!("a lexical outage must not fail semantic refinement: {other:?}")
                    }
                };
                assert_eq!(
                    serde_json::to_value(final_results).unwrap(),
                    serde_json::to_value(&expected).unwrap(),
                );
            }
        }
        assert!(logs_contain("lexical search failed"));
        assert!(!logs_contain("private-lexical-query"));
        assert!(!logs_contain("secret-token"));
        assert!(!logs_contain("forged-log-line"));
    }

    #[test]
    fn lexical_backend_failures_preserve_direct_quality_results() {
        use frankensearch_core::filter::BitsetFilter;
        use std::sync::atomic::AtomicUsize;

        for attested in [false, true] {
            for fail_after_first in [false, true] {
                let directory = temp_dir("quality-only-lexical-outage");
                let (index, identity) = quality_only_index(directory.path(), attested);
                let query = TieredQueryEmbeddings::quality_only(
                    BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], identity).unwrap(),
                );
                let allowed = BitsetFilter::from_doc_ids(["b", "c"]);
                let baseline =
                    SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default());
                let (expected, _) = baseline
                    .search_collect_with_filter(&query, 2, Some(&allowed))
                    .unwrap();
                let lexical = Arc::new(UnavailableLexical {
                    calls: AtomicUsize::new(0),
                    fail_after_first,
                });
                let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
                    .with_lexical(lexical.clone());
                let (actual, metrics) = searcher
                    .search_collect_with_filter(&query, 2, Some(&allowed))
                    .expect("quality-only retrieval must survive an optional lexical outage");
                assert_eq!(
                    serde_json::to_value(&actual).unwrap(),
                    serde_json::to_value(&expected).unwrap()
                );
                assert!(!actual.is_empty());
                assert_eq!(metrics.lexical_candidates, 0);
                assert!(
                    actual
                        .iter()
                        .all(|hit| hit.source == ScoreSource::SemanticQuality
                            && hit.fast_score.is_none()
                            && hit.quality_score.is_some())
                );

                lexical.calls.store(0, std::sync::atomic::Ordering::SeqCst);
                let phases = searcher
                    .search_iter_with_filter(&query, 2, Some(&allowed))
                    .collect::<Vec<_>>();
                let [SearchPhase::Initial { results, .. }] = phases.as_slice() else {
                    panic!("direct quality must retain exactly one successful phase");
                };
                assert_eq!(
                    serde_json::to_value(results).unwrap(),
                    serde_json::to_value(&expected).unwrap()
                );
            }
        }
    }

    struct ExcludeB;

    impl SearchFilter for ExcludeB {
        fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
            doc_id != "b"
        }

        fn name(&self) -> &'static str {
            "exclude-b"
        }
    }

    fn vector_hit(doc_id: &str, index: u32, score: f32) -> VectorHit {
        VectorHit {
            doc_id: doc_id.into(),
            index,
            score,
        }
    }

    /// The union conversion carries the blended score and the no-index
    /// sentinel through, and fabricates NO per-tier evidence.
    ///
    /// This replaces `aligned_numeric_score_lookup_matches_doc_id_maps`,
    /// whose subject — a numeric lookup keyed by fast-tier row ordinal — was
    /// only sound while the quality tier was a re-scored subset of the fast
    /// pool. Under independent retrieval a quality-only document has no fast
    /// ordinal, and the old helper's `.unwrap_or((hit.score, None))` fallback
    /// would have reported the BLENDED score as that document's fast score.
    #[test]
    fn union_conversion_reports_no_index_and_no_fabricated_evidence() {
        let blended = vec![
            vector_hit("quality-only", u32::MAX, 0.91),
            vector_hit("a", 10, 0.88),
        ];
        let actual =
            unique_vector_hits_to_scored_results_owned(blended, 2, ScoreSource::SemanticQuality);
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].doc_id, "quality-only");
        assert_eq!(
            actual[0].index, None,
            "a document with no fast-tier row must not report one"
        );
        assert_eq!(actual[1].index, Some(10));
        for result in &actual {
            assert_eq!(
                (result.fast_score, result.quality_score),
                (None, None),
                "per-tier evidence is attached from the pools, never invented here"
            );
            assert_eq!(result.source, ScoreSource::SemanticQuality);
        }
    }

    /// bd-dbp10 acceptance 4, THE PLANTED NEGATIVE: the quality tier must be
    /// scored with the QUALITY-bound query, never the fast one.
    ///
    /// The fixture makes the two tiers disagree on purpose. The fast tier puts
    /// `a` first for the fast query; in the quality tier the fast query's
    /// direction points at `a` too, but the QUALITY query points at `c`. So
    /// the refined winner is `c` if and only if the quality tier saw its own
    /// query. Feeding it the fast vector — which is exactly what this code did
    /// before — returns `a`, and the assertion below is what says so.
    ///
    /// Note the two vectors have the SAME WIDTH. That is the whole reason this
    /// bug could live in production: every dimension check passes.
    #[test]
    fn the_quality_tier_is_scored_with_the_quality_query_not_the_fast_one() {
        let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        // Fast space: the fast query [1,0] ranks a > b > c.
        let fast = InMemoryVectorIndex::from_vectors(
            doc_ids.clone(),
            vec![vec![1.0, 0.0], vec![0.7, 0.7], vec![0.0, 1.0]],
            2,
        )
        .unwrap();
        // Quality space: the fast query [1,0] would rank a first here too,
        // but the quality query [0,1] ranks c first.
        let quality = InMemoryVectorIndex::from_vectors(
            doc_ids,
            vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]],
            2,
        )
        .unwrap();
        let index = Arc::new(
            InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"),
        );
        let searcher = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                // Quality-dominant blend so the quality tier's opinion decides
                // the refined order rather than merely nudging it.
                quality_weight: 1.0,
                ..TwoTierConfig::default()
            },
        );

        let query = TieredQueryEmbeddings::progressive(
            bound("sync-fixture", vec![1.0, 0.0]),
            bound("sync-fixture", vec![0.0, 1.0]),
        );
        let (results, _) = searcher.search_collect(&query, 3).unwrap();
        assert_eq!(
            results[0].doc_id,
            "c",
            "the refined winner must come from the QUALITY query's ranking; got {:?}",
            results
                .iter()
                .map(|r| r.doc_id.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// bd-dbp10 acceptance 2: on an attested index every identity join
    /// completes before any vector read, and a query from another space is
    /// refused by field name.
    #[test]
    fn an_attested_tier_refuses_a_foreign_space_query() {
        let dir = temp_dir("attested-refusal");
        let (fast, fast_identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "dbp10-fast",
            11,
            &[("a", &[1.0, 0.0]), ("b", &[0.0, 1.0])],
        );
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, None).expect("fast-only fixture"));
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());

        // Positive control: the space this tier was written in.
        let (results, _) = searcher
            .search_collect(
                &TieredQueryEmbeddings::fast_only(
                    BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).expect("bind"),
                ),
                2,
            )
            .expect("the producing space is admitted");
        assert_eq!(results[0].doc_id, "a");

        // Same width, different space: refused even when zero signal or k=0
        // would otherwise avoid vector work.
        for (vector, k) in [
            (vec![1.0, 0.0], 2),
            (vec![0.0, 0.0], 2),
            (vec![0.0, 0.0], 0),
        ] {
            let error = searcher
                .search_collect(&fast_only_query(vector), k)
                .expect_err("a foreign embedding space must be refused");
            assert!(
                matches!(
                    error,
                    SearchError::InvalidConfig { ref field, .. }
                        if field == "query_embedding.fast.space_identity"
                ),
                "got {error:?}"
            );
        }
    }

    #[test]
    fn admitted_tiers_reject_foreign_producers_before_zero_signal_or_zero_k() {
        let dir = temp_dir("producer-refusal");
        let (fast, fast_identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "producer-fast",
            41,
            &[("a", &[1.0, 0.0])],
        );
        let (quality, quality_identity) = attested_tier(
            dir.path(),
            "vector.quality.idx",
            "producer-quality",
            41,
            &[("a", &[0.0, 1.0])],
        );
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(
                InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"),
            ),
            TwoTierConfig::default(),
        );
        let matching = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity.clone()).unwrap(),
            BoundQueryEmbedding::new(vec![0.0, 1.0], quality_identity.clone()).unwrap(),
        );
        assert_eq!(
            searcher.search_collect(&matching, 1).unwrap().0[0].doc_id,
            "a"
        );

        let foreign_producer = |identity: &EmbeddingIdentityBundleV1| {
            let mut foreign = identity.clone();
            "foreign-backend".clone_into(&mut foreign.producer.backend);
            foreign.producer.golden_vectors.vectors_sha256 = "c".repeat(64);
            foreign.validate().expect("coherent foreign producer");
            assert_eq!(foreign.space.fingerprint(), identity.space.fingerprint());
            assert_ne!(
                foreign.producer.fingerprint(),
                identity.producer.fingerprint()
            );
            foreign
        };
        let foreign_fast = foreign_producer(&fast_identity);
        let foreign_quality = foreign_producer(&quality_identity);
        for (tier, fast_query_identity, quality_query_identity) in [
            ("fast", foreign_fast, quality_identity),
            ("quality", fast_identity, foreign_quality),
        ] {
            for (vector, k) in [
                (vec![1.0, 0.0], 1),
                (vec![0.0, 0.0], 1),
                (vec![0.0, 0.0], 0),
            ] {
                let query = TieredQueryEmbeddings::progressive(
                    BoundQueryEmbedding::new(vector.clone(), fast_query_identity.clone()).unwrap(),
                    BoundQueryEmbedding::new(vector, quality_query_identity.clone()).unwrap(),
                );
                assert_admission_refusal(
                    &searcher,
                    &query,
                    k,
                    &format!("query_embedding.{tier}.producer_conformance"),
                );
            }
        }
    }

    #[test]
    fn admitted_tier_refuses_a_conformance_compatible_foreign_producer() {
        let dir = temp_dir("compatible-producer-refusal");
        let (fast, identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "compatible-producer",
            42,
            &[("a", &[1.0, 0.0])],
        );
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(fast, None).expect("fast-only fixture")),
            TwoTierConfig::default(),
        );
        let mut foreign = identity.clone();
        "compatible-foreign-backend".clone_into(&mut foreign.producer.backend);
        let bound = BoundQueryEmbedding::new(vec![1.0, 0.0], foreign).unwrap();
        assert!(matches!(
            bound
                .verify_producer_conformance(&identity, "fast")
                .unwrap(),
            SpaceIdentityAdmission::ConformanceCompatibleProducer { .. }
        ));
        let query = TieredQueryEmbeddings::fast_only(bound);
        for k in [0, 1] {
            assert_admission_refusal(
                &searcher,
                &query,
                k,
                "search_activation.fast.producer_conformance",
            );
        }
    }

    #[test]
    fn caller_declared_and_legacy_tiers_keep_their_weaker_producer_contract() {
        let identity = semantic_identity("caller-declared", 2);
        let declared = InMemoryVectorIndex::from_vectors_with_identity(
            vec!["a".to_owned()],
            vec![vec![1.0, 0.0]],
            2,
            &identity.space,
        )
        .expect("declared space fixture");
        let legacy =
            InMemoryVectorIndex::from_vectors(vec!["a".to_owned()], vec![vec![1.0, 0.0]], 2)
                .expect("legacy fixture");
        let mut foreign = identity.clone();
        "unattested-backend".clone_into(&mut foreign.producer.backend);
        foreign.producer.golden_vectors.vectors_sha256 = "d".repeat(64);
        let bound = BoundQueryEmbedding::new(vec![1.0, 0.0], foreign).unwrap();
        assert!(
            bound
                .verify_producer_conformance(&identity, "fast")
                .is_err()
        );
        let query = TieredQueryEmbeddings::fast_only(bound);
        for fast in [declared, legacy] {
            assert!(fast.admitted_source_binding().is_none());
            let searcher = SyncTwoTierSearcher::new(
                Arc::new(InMemoryTwoTierIndex::new(fast, None).expect("unattested fixture")),
                TwoTierConfig::default(),
            );
            let (results, _) = searcher.search_collect(&query, 1).unwrap();
            assert_eq!(results[0].doc_id, "a");
        }
    }

    /// bd-ctzo C2 on the sync path: with an ATTESTED quality tier the refined
    /// phase retrieves independently, so a document the fast tier does not
    /// contain at all becomes reachable.
    #[test]
    fn an_attested_quality_tier_reaches_a_document_outside_the_fast_pool() {
        let dir = temp_dir("attested-union");
        let (fast, fast_identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "dbp10-union-fast",
            21,
            &[
                ("doc-near", &[1.0, 0.0, 0.0]),
                ("doc-far", &[0.0, 0.0, 1.0]),
            ],
        );
        // The quality tier holds a document the FAST TIER DOES NOT CONTAIN.
        // No rescoring of a fast-selected pool can produce it.
        let (quality, quality_identity) = attested_tier(
            dir.path(),
            "vector.quality.idx",
            "dbp10-union-quality",
            21,
            &[
                ("doc-near", &[0.0, 0.0, 1.0]),
                ("doc-far", &[0.0, 0.5, 0.0]),
                ("doc-quality-only", &[0.0, 1.0, 0.0]),
            ],
        );
        let index = Arc::new(
            InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"),
        );
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());

        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], fast_identity).expect("bind fast"),
            BoundQueryEmbedding::new(vec![0.0, 1.0, 0.0], quality_identity).expect("bind quality"),
        );
        let (results, _) = searcher.search_collect(&query, 3).unwrap();
        let ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(
            ids.contains(&"doc-quality-only"),
            "independent quality retrieval must reach a document the fast tier lacks; got {ids:?}"
        );
        let quality_only = results
            .iter()
            .find(|r| r.doc_id == "doc-quality-only")
            .expect("present");
        assert_eq!(
            quality_only.index, None,
            "a document with no fast-tier row must not report one"
        );
        assert_eq!(
            quality_only.fast_score, None,
            "the fast tier never saw it, so it has no fast evidence"
        );
    }

    /// bd-dbp10 acceptance 3: a legacy tier carries no identity to join
    /// against, and must stay searchable rather than becoming a refusal.
    #[test]
    fn a_legacy_unidentified_index_stays_searchable() {
        let index = make_index();
        assert!(
            index.fast_index().space_fingerprint_hex().is_none(),
            "the fixture is only meaningful while the index is legacy-unidentified"
        );
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
        let (results, metrics) = searcher
            .search_collect(&tiered(vec![1.0, 0.0]), 3)
            .expect("a legacy index stays searchable");
        assert!(
            !results.is_empty(),
            "the legacy lane must still return hits"
        );
        assert_eq!(
            metrics.skip_reason, None,
            "the legacy lane still refines; it is unidentified, not degraded"
        );
    }

    /// A quality tier the caller bound no quality embedding for is SKIPPED
    /// with a typed reason — never served by reusing the fast embedding.
    #[test]
    fn a_fast_only_query_skips_the_quality_tier_instead_of_reusing_the_fast_vector() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (_, metrics) = searcher
            .search_collect(&fast_only_query(vec![1.0, 0.0]), 3)
            .unwrap();
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("quality_query_embedding_absent"),
            "the index HAS a quality tier; the query simply bound nothing for it"
        );
        assert_eq!(metrics.phase2_vectors_searched, 0);
    }

    #[test]
    fn hash_control_skips_quality_and_does_not_become_semantic_quality() {
        let hash_bundle = EmbeddingIdentityBundleV1::explicit_test_model("fnv1a-256", 2);
        let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let fast = InMemoryVectorIndex::from_vectors_with_identity(
            doc_ids.clone(),
            vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]],
            2,
            &hash_bundle.space,
        )
        .expect("hash-control fast index");
        let quality = InMemoryVectorIndex::from_vectors(
            doc_ids,
            vec![vec![0.2, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]],
            2,
        )
        .expect("quality index");
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(
                InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"),
            ),
            TwoTierConfig::default(),
        );
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], hash_bundle)
                .expect("hash-control query binds"),
            BoundQueryEmbedding::new(
                vec![1.0, 0.0],
                EmbeddingIdentityBundleV1::explicit_test_model("sync-fixture", 2),
            )
            .expect("quality query binds"),
        );
        let (results, metrics) = searcher
            .search_collect(&query, 2)
            .expect("hash-control search still returns hits");
        assert!(!results.is_empty());
        assert!(
            results
                .iter()
                .all(|result| result.source == ScoreSource::HashControl),
            "hash-control hits must not be relabeled SemanticQuality: {:?}",
            results
                .iter()
                .map(|result| result.source)
                .collect::<Vec<_>>()
        );
        assert_eq!(metrics.phase2_vectors_searched, 0);
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("non_semantic_fast_embedder_vector_control")
        );
        assert!(metrics.hash_control_candidates > 0);
        assert_eq!(metrics.semantic_candidates, 0);
        let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
        assert!(
            phases
                .iter()
                .any(|phase| matches!(phase, SearchPhase::Initial { .. }))
        );
        assert!(
            !phases
                .iter()
                .any(|phase| matches!(phase, SearchPhase::Refined { .. })),
            "hash-control Initial must not schedule a Refined phase"
        );
        assert!(
            matches!(phases.first(), Some(SearchPhase::Initial { .. })),
            "first hash-control phase must be Initial, got {:?}",
            phases.first()
        );
        if let Some(SearchPhase::Initial {
            metrics: phase_metrics,
            ..
        }) = phases.first()
        {
            assert_eq!(
                phase_metrics.hash_control_candidates,
                metrics.hash_control_candidates
            );
            assert!(phase_metrics.hash_control_candidates > 0);
        }
    }

    #[test]
    fn persisted_producer_kind_controls_progressive_search_despite_display_names() {
        for (fast_hash, quality_hash) in [(true, false), (false, false), (false, true)] {
            let identity = |hash| {
                if hash {
                    EmbeddingIdentityBundleV1::explicit_test_model("looks-semantic", 2)
                } else {
                    semantic_identity("fnv1a-256", 2)
                }
            };
            let fast = identity(fast_hash);
            let quality = identity(quality_hash);
            let dir = temp_dir("persisted-producer-kind");
            let mut writer =
                frankensearch_index::TwoTierIndex::create(dir.path(), TwoTierConfig::default())
                    .unwrap();
            writer.set_fast_embedder_id(fast.space.logical_model_id.as_str());
            writer.set_quality_embedder_id(quality.space.logical_model_id.as_str());
            writer.set_fast_identity(&fast).unwrap();
            writer.set_quality_identity(&quality).unwrap();
            writer
                .add_record("a", &[1.0, 0.0], Some(&[1.0, 0.0]))
                .unwrap();
            drop(writer.finish().unwrap());
            let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
            let query = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
                BoundQueryEmbedding::new(vec![1.0, 0.0], quality).unwrap(),
            );
            if quality_hash {
                for k in [0, 1] {
                    assert_admission_refusal(&searcher, &query, k, "sync_search.quality_kind");
                }
            } else {
                let (hits, metrics) = searcher.search_collect(&query, 1).unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].doc_id, "a");
                if fast_hash {
                    assert_eq!(hits[0].source, ScoreSource::HashControl);
                    assert_eq!(hits[0].quality_score, None);
                    assert_eq!(metrics.hash_control_candidates, 1);
                    assert_eq!(metrics.semantic_candidates, 0);
                    assert_eq!(metrics.phase2_vectors_searched, 0);
                } else {
                    assert_eq!(hits[0].source, ScoreSource::SemanticQuality);
                    assert_eq!(hits[0].quality_score, Some(1.0));
                    assert_eq!(metrics.hash_control_candidates, 0);
                    assert_eq!(metrics.semantic_candidates, 1);
                    assert_eq!(metrics.phase2_vectors_searched, 1);
                }
            }
        }
    }

    #[test]
    fn admitted_hash_kind_overrides_a_semantic_looking_display_name() {
        let dir = temp_dir("attested-hash-kind");
        let fast_identity =
            EmbeddingIdentityBundleV1::explicit_test_model("semantic-looking-fast", 2);
        let quality_identity =
            EmbeddingIdentityBundleV1::explicit_test_model("semantic-looking-quality", 2);
        assert!(!frankensearch_core::is_hash_generation_id(
            &fast_identity.space.logical_model_id
        ));
        let fast = attested_tier_with_identity(
            dir.path(),
            "vector.fast.idx",
            43,
            &[("a", &[1.0, 0.0]), ("b", &[0.0, 1.0])],
            &fast_identity,
        );
        let quality = attested_tier_with_identity(
            dir.path(),
            "vector.quality.idx",
            43,
            &[("a", &[0.0, 1.0]), ("b", &[1.0, 0.0])],
            &quality_identity,
        );
        let index = Arc::new(
            InMemoryTwoTierIndex::new(fast, Some(quality)).expect("matching control tiers"),
        );
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).unwrap(),
            BoundQueryEmbedding::new(vec![0.0, 1.0], quality_identity).unwrap(),
        );
        for has_lexical in [false, true] {
            let mut searcher = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    explain: true,
                    ..TwoTierConfig::default()
                },
            );
            if has_lexical {
                searcher = searcher.with_lexical(Arc::new(StaticLexical {
                    hits: vec![lexical_result("a", 10.0), lexical_result("lex-only", 9.0)],
                }));
            }
            let (results, metrics) = searcher.search_collect(&query, 3).unwrap();
            assert_eq!(metrics.semantic_candidates, 0);
            assert_eq!(metrics.hash_control_candidates, 2);
            assert_eq!(metrics.phase2_vectors_searched, 0);
            assert_eq!(
                metrics.skip_reason.as_deref(),
                Some("non_semantic_fast_embedder_vector_control")
            );
            for id in ["a", "b"] {
                let hit = results
                    .iter()
                    .find(|hit| hit.doc_id == id)
                    .expect("control hit");
                assert_eq!(
                    hit.source,
                    if has_lexical && id == "a" {
                        ScoreSource::Hybrid
                    } else {
                        ScoreSource::HashControl
                    }
                );
                assert_eq!(hit.quality_score, None);
                let components = &hit
                    .explanation
                    .as_ref()
                    .expect("explained control hit")
                    .components;
                assert!(components.iter().any(|component| matches!(
                    &component.source,
                    ExplainedSource::HashControl { .. }
                )));
                assert!(!components.iter().any(|component| matches!(
                    &component.source,
                    ExplainedSource::SemanticFast { .. } | ExplainedSource::SemanticQuality { .. }
                )));
            }
            if has_lexical {
                assert_eq!(
                    results
                        .iter()
                        .find(|hit| hit.doc_id == "lex-only")
                        .unwrap()
                        .source,
                    ScoreSource::Lexical
                );
            }
            let phases = searcher.search_iter(&query, 3).collect::<Vec<_>>();
            let [
                SearchPhase::Initial {
                    results: initial, ..
                },
            ] = phases.as_slice()
            else {
                panic!("an admitted hash control must never schedule refinement");
            };
            assert_eq!(
                serde_json::to_value(initial).unwrap(),
                serde_json::to_value(&results).unwrap()
            );
        }
    }

    #[test]
    fn admitted_semantic_kind_overrides_a_hash_looking_display_name() {
        let dir = temp_dir("attested-semantic-kind");
        let (fast, fast_identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "fnv1a-semantic-fixture",
            44,
            &[("a", &[1.0, 0.0])],
        );
        let (quality, quality_identity) = attested_tier(
            dir.path(),
            "vector.quality.idx",
            "semantic-quality",
            44,
            &[("quality-only", &[0.0, 1.0])],
        );
        assert!(frankensearch_core::is_hash_generation_id(
            &fast_identity.space.logical_model_id
        ));
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(
                InMemoryTwoTierIndex::new(fast, Some(quality)).expect("matching semantic tiers"),
            ),
            TwoTierConfig {
                explain: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(StaticLexical {
            hits: vec![lexical_result("a", 10.0)],
        }));
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).unwrap(),
            BoundQueryEmbedding::new(vec![0.0, 1.0], quality_identity).unwrap(),
        );
        let (results, metrics) = searcher.search_collect(&query, 2).unwrap();
        assert_eq!(metrics.hash_control_candidates, 0);
        assert_eq!(metrics.semantic_candidates, 1);
        assert_eq!(metrics.phase2_vectors_searched, 1);
        assert_eq!(metrics.skip_reason, None);
        assert_eq!(
            results
                .iter()
                .find(|hit| hit.doc_id == "quality-only")
                .unwrap()
                .quality_score,
            Some(1.0)
        );
        let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
        assert!(matches!(
            phases.as_slice(),
            [SearchPhase::Initial { .. }, SearchPhase::Refined { .. }]
        ));
        for phase in &phases {
            let hits = match phase {
                SearchPhase::Initial { results, .. } | SearchPhase::Refined { results, .. } => {
                    results
                }
                SearchPhase::RefinementFailed { error, .. } => {
                    panic!("semantic refinement failed: {error}")
                }
                SearchPhase::Reranked { .. } => {
                    panic!("synchronous search must not emit a reranked phase")
                }
            };
            for hit in hits {
                assert_ne!(hit.source, ScoreSource::HashControl);
                let components = &hit
                    .explanation
                    .as_ref()
                    .expect("explained semantic hit")
                    .components;
                assert!(!components.iter().any(|component| matches!(
                    &component.source,
                    ExplainedSource::HashControl { .. }
                )));
                assert!(components.iter().any(|component| matches!(
                    &component.source,
                    ExplainedSource::SemanticFast { .. } | ExplainedSource::SemanticQuality { .. }
                )));
            }
        }
    }

    #[test]
    fn search_collect_returns_refined_results() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, metrics) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].source, ScoreSource::SemanticQuality);
        assert!(metrics.phase1_total_ms >= 0.0);
        assert!(metrics.phase2_total_ms >= 0.0);
        // The semantic lane produced hits, so it must not be classified.
        assert_eq!(metrics.zero_signal, None);
    }

    #[test]
    fn empty_index_search_carries_typed_zero_signal() {
        let fast = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 2).unwrap();
        let index =
            Arc::new(InMemoryTwoTierIndex::new(fast, None).expect("empty fast-only fixture"));
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
        let (results, metrics) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 3).unwrap();
        assert!(results.is_empty());
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::NewlyCreatedEmpty),
            "an empty semantic lane must say why, not just return nothing"
        );
    }

    fn write_corrupt_sync_fixture(path: &Path, model: &str, revision: &str, all_unusable: bool) {
        use frankensearch_index::{Quantization, VectorIndex};

        let mut writer =
            VectorIndex::create_with_revision(path, model, revision, 2, Quantization::F16).unwrap();
        for (id, vector) in [
            ("bad-zero", [1.0, 0.0]),
            ("negative-high", [-0.5, 0.5]),
            ("negative-low", [-1.0, 0.0]),
        ] {
            writer.write_record(id, &vector).unwrap();
        }
        writer.finish().unwrap();
        let file = VectorIndex::open_read_only(path).unwrap();
        let offset = usize::try_from(file.metadata().vectors_offset).unwrap();
        let row_bytes = file.dimension() * std::mem::size_of::<u16>();
        let mut bytes = std::fs::read(path).unwrap();
        for row in 0..file.record_count() {
            if all_unusable || file.doc_id_at(row).unwrap() == "bad-zero" {
                // Corrupt a previously valid persisted F16 row.
                bytes[offset + row * row_bytes..offset + (row + 1) * row_bytes].fill(0);
            }
        }
        drop(file);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn persisted_corrupt_vectors_are_not_published_as_sync_candidates() {
        for all_unusable in [false, true] {
            let dir = temp_dir("corrupt-sync-candidates");
            let path = dir.path().join("fast.fsvi");
            write_corrupt_sync_fixture(&path, "test", "", all_unusable);
            let fast = InMemoryVectorIndex::from_fsvi(&path).unwrap();
            let index = Arc::new(InMemoryTwoTierIndex::new(fast, None).unwrap());
            let searcher = SyncTwoTierSearcher::new(
                index,
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            );
            let query = fast_only_query(vec![1.0, 0.0]);
            let expected_ids = if all_unusable {
                vec![]
            } else {
                vec!["negative-high", "negative-low"]
            };
            let expected_reason = all_unusable.then_some(ZeroSignalReason::NoUsableVectors);
            let (results, metrics) = searcher.search_collect(&query, 2).unwrap();
            assert_eq!(
                results
                    .iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                expected_ids
            );
            assert_eq!(metrics.zero_signal, expected_reason);
            let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
            assert_eq!(phases.len(), 1);
            let SearchPhase::Initial {
                results, metrics, ..
            } = &phases[0]
            else {
                panic!("a fast-only fixture must publish an Initial phase");
            };
            assert_eq!(
                results
                    .iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                expected_ids
            );
            assert_eq!(metrics.vectors_searched, 3);

            let tuned = searcher.with_search_params(SearchParams {
                parallel_enabled: true,
                parallel_threshold: 0,
                parallel_chunk_size: 1,
            });
            let (results, metrics) = tuned.search_collect(&query, 2).unwrap();
            assert_eq!(
                results
                    .iter()
                    .map(|hit| hit.doc_id.as_str())
                    .collect::<Vec<_>>(),
                expected_ids
            );
            assert_eq!(metrics.zero_signal, expected_reason);
        }
    }

    #[test]
    fn persisted_corrupt_vectors_are_not_published_by_sync_quality_only() {
        for all_unusable in [false, true] {
            let dir = temp_dir("corrupt-sync-quality-only");
            let path = dir.path().join("quality.fsvi");
            let quality_query =
                BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("quality", 2)).unwrap();
            write_corrupt_sync_fixture(
                &path,
                "quality",
                quality_query.identity_fingerprint(),
                all_unusable,
            );
            let quality = InMemoryVectorIndex::from_fsvi(&path).unwrap();
            let fast = InMemoryVectorIndex::from_vectors(
                vec!["fast-only".to_owned()],
                vec![vec![1.0, 0.0]],
                2,
            )
            .unwrap();
            let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap());
            let query = TieredQueryEmbeddings::quality_only(quality_query);
            let expected_ids: &[&str] = if all_unusable {
                &[]
            } else {
                &["negative-high"]
            };
            let expected_reason = all_unusable.then_some(ZeroSignalReason::NoUsableVectors);
            for params in [
                None,
                Some(SearchParams {
                    parallel_enabled: true,
                    parallel_threshold: 0,
                    parallel_chunk_size: 1,
                }),
            ] {
                let mut searcher = SyncTwoTierSearcher::new(
                    Arc::clone(&index),
                    TwoTierConfig {
                        candidate_multiplier: 1,
                        ..TwoTierConfig::default()
                    },
                );
                if let Some(params) = params {
                    searcher = searcher.with_search_params(params);
                }
                let (results, metrics) = searcher.search_collect(&query, 1).unwrap();
                assert_eq!(
                    results
                        .iter()
                        .map(|hit| hit.doc_id.as_str())
                        .collect::<Vec<_>>(),
                    expected_ids
                );
                assert_eq!(metrics.zero_signal, expected_reason);
                assert_eq!(metrics.phase1_vectors_searched, 3);
                for hit in &results {
                    assert_eq!(hit.source, ScoreSource::SemanticQuality);
                    assert_eq!(hit.fast_score, None);
                    assert_eq!(hit.quality_score, Some(-0.5));
                }

                let phases = searcher.search_iter(&query, 1).collect::<Vec<_>>();
                let [
                    SearchPhase::Initial {
                        results, metrics, ..
                    },
                ] = phases.as_slice()
                else {
                    panic!("quality-only search must publish one Initial phase: {phases:?}");
                };
                assert_eq!(
                    results
                        .iter()
                        .map(|hit| hit.doc_id.as_str())
                        .collect::<Vec<_>>(),
                    expected_ids
                );
                assert_eq!(
                    metrics.skip_reason.as_deref(),
                    expected_reason.map(ZeroSignalReason::reason_code)
                );
            }
        }
    }

    #[test]
    fn persisted_corrupt_vectors_are_not_published_as_sync_refinement_evidence() {
        use frankensearch_index::{Quantization, VectorIndex};

        let dir = temp_dir("corrupt-sync-refinement");
        let fast_path = dir.path().join("fast.fsvi");
        let fast_query =
            BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("fast", 2)).unwrap();
        let mut writer = VectorIndex::create_with_revision(
            &fast_path,
            "fast",
            fast_query.identity_fingerprint(),
            2,
            Quantization::F16,
        )
        .unwrap();
        // This document has valid fast evidence and an unusable quality row.
        writer.write_record("bad-zero", &[1.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let quality_path = dir.path().join("quality.fsvi");
        let quality_query =
            BoundQueryEmbedding::new(vec![1.0, 0.0], semantic_identity("quality", 2)).unwrap();
        write_corrupt_sync_fixture(
            &quality_path,
            "quality",
            quality_query.identity_fingerprint(),
            false,
        );
        let fast = InMemoryVectorIndex::from_fsvi(&fast_path).unwrap();
        let quality = InMemoryVectorIndex::from_fsvi(&quality_path).unwrap();
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap()),
            TwoTierConfig {
                candidate_multiplier: 1,
                ..TwoTierConfig::default()
            },
        );
        let query = TieredQueryEmbeddings::progressive(fast_query, quality_query);
        let (results, metrics) = searcher.search_collect(&query, 3).unwrap();
        assert_eq!(metrics.phase2_vectors_searched, 2);
        assert_eq!(metrics.incomplete_embeddings, 1);
        assert_eq!(results.len(), 3);
        for (id, fast_score, quality_score) in [
            ("bad-zero", Some(1.0), None),
            ("negative-high", None, Some(-0.5)),
            ("negative-low", None, Some(-1.0)),
        ] {
            let hit = results.iter().find(|hit| hit.doc_id == id).unwrap();
            assert_eq!(hit.fast_score, fast_score, "{id}");
            assert_eq!(hit.quality_score, quality_score, "{id}");
        }

        // Neither tier has usable vectors for every retained candidate.
        let searcher = searcher.with_mmr_config(sync_mmr_config());
        let phases = searcher.search_iter(&query, 3).collect::<Vec<_>>();
        let [
            SearchPhase::Initial { .. },
            SearchPhase::Refined {
                results: refined, ..
            },
        ] = phases.as_slice()
        else {
            panic!("a corrupt row must not license diversity reranking: {phases:?}");
        };
        assert_eq!(
            refined
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn filter_eliminating_every_candidate_classifies_as_filter() {
        struct ExcludeAll;
        impl SearchFilter for ExcludeAll {
            fn matches(&self, _doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
                false
            }
            fn name(&self) -> &'static str {
                "exclude-all"
            }
        }
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, metrics) = searcher
            .search_collect_with_filter(&tiered(vec![1.0, 0.0]), 3, Some(&ExcludeAll))
            .unwrap();
        assert!(results.is_empty());
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::FilterEliminatedAll)
        );
    }

    #[test]
    fn zero_norm_query_preserves_request_scoped_reason_without_scanning() {
        // Request-scoped pre-scan classification outranks index state:
        // a zero-norm query is the caller's defect even on an empty index.
        let fast = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 2).unwrap();
        let empty =
            Arc::new(InMemoryTwoTierIndex::new(fast, None).expect("empty fast-only fixture"));
        for index in [empty, make_index()] {
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
            let (results, metrics) = searcher
                .search_collect(&fast_only_query(vec![0.0, 0.0]), 3)
                .unwrap();
            assert!(results.is_empty());
            assert_eq!(metrics.zero_signal, Some(ZeroSignalReason::ZeroNormQuery));
            assert_eq!(metrics.phase1_vectors_searched, 0);
            assert_eq!(metrics.phase2_vectors_searched, 0);
        }
    }

    #[test]
    fn zero_fast_query_keeps_filtered_lexical_results_in_collect_and_stream() {
        let lexical = Arc::new(StaticLexical {
            hits: vec![lexical_result("b", 10.0), lexical_result("lex-only", 9.0)],
        });
        let searcher =
            SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(lexical);

        // Missing and zero-norm quality queries both preserve the lexical
        // answer without inventing semantic evidence or a Refined phase.
        for query in [fast_only_query(vec![0.0, 0.0]), tiered(vec![0.0, 0.0])] {
            let (results, metrics) = searcher
                .search_collect_with_filter(&query, 3, Some(&ExcludeB))
                .expect("zero fast signal still permits lexical retrieval");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "lex-only");
            assert_eq!(results[0].source, ScoreSource::Lexical);
            assert_eq!(results[0].fast_score, None);
            assert_eq!(results[0].quality_score, None);
            assert_eq!(results[0].lexical_score, Some(9.0));
            assert_eq!(metrics.zero_signal, Some(ZeroSignalReason::ZeroNormQuery));
            assert_eq!(metrics.lexical_candidates, 1);
            assert_eq!(metrics.semantic_candidates, 0);
            assert_eq!(metrics.phase1_vectors_searched, 0);
            assert_eq!(metrics.phase2_vectors_searched, 0);

            let phases = searcher
                .search_iter_with_filter(&query, 3, Some(&ExcludeB))
                .collect::<Vec<_>>();
            assert_eq!(phases.len(), 1);
            let SearchPhase::Initial {
                results: initial,
                metrics: initial_metrics,
                ..
            } = &phases[0]
            else {
                panic!("lexical results must be available in the Initial phase");
            };
            assert_eq!(initial[0].doc_id, "lex-only");
            assert_eq!(initial_metrics.vectors_searched, 0);
            assert_eq!(initial_metrics.lexical_candidates, 1);
            assert_eq!(
                serde_json::to_value(initial).unwrap(),
                serde_json::to_value(&results).unwrap(),
                "collect and the last streamed phase must expose identical evidence"
            );
        }
    }

    #[test]
    fn filtered_lexical_budget_widens_and_stops_at_exhaustion() {
        use frankensearch_core::filter::BitsetFilter;

        struct RecordingLexical {
            limits: Mutex<Vec<usize>>,
            fail_after_first: bool,
        }
        impl SyncLexicalSearch for RecordingLexical {
            fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
                let mut limits = self.limits.lock().unwrap();
                limits.push(limit);
                if self.fail_after_first && limits.len() > 1 {
                    return Err(SearchError::Cancelled {
                        phase: "lexical-refill".to_owned(),
                        reason: "cancelled while widening".to_owned(),
                    });
                }
                drop(limits);
                Ok((0..limit.min(72))
                    .map(|rank| lexical_result(&format!("lex-{rank:02}"), (72 - rank) as f32))
                    .collect())
            }
        }
        let lexical = Arc::new(RecordingLexical {
            limits: Mutex::new(Vec::new()),
            fail_after_first: false,
        });
        let searcher = SyncTwoTierSearcher::new(
            make_index(),
            TwoTierConfig {
                candidate_multiplier: 2,
                fast_only: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(lexical.clone());
        let query = fast_only_query(vec![0.0, 0.0]);
        let allowed = BitsetFilter::from_doc_ids([
            "lex-64", "lex-65", "lex-66", "lex-67", "lex-68", "lex-69", "lex-70", "lex-71",
        ]);
        let (results, metrics) = searcher
            .search_collect_with_filter(&query, 2, Some(&allowed))
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["lex-64", "lex-65"]
        );
        assert_eq!(
            metrics.lexical_candidates, 4,
            "keep the eligible pool bounded"
        );
        assert_eq!(*lexical.limits.lock().unwrap(), [4, 8, 16, 32, 64, 128]);

        lexical.limits.lock().unwrap().clear();
        let (unfiltered, _) = searcher.search_collect(&query, 2).unwrap();
        assert_eq!(unfiltered[0].doc_id, "lex-00");
        assert_eq!(*lexical.limits.lock().unwrap(), [4]);

        lexical.limits.lock().unwrap().clear();
        let reject_all = BitsetFilter::from_doc_ids(std::iter::empty::<&str>());
        let (empty, metrics) = searcher
            .search_collect_with_filter(&query, 2, Some(&reject_all))
            .unwrap();
        assert!(empty.is_empty());
        assert_eq!(metrics.lexical_candidates, 0);
        assert_eq!(*lexical.limits.lock().unwrap(), [4, 8, 16, 32, 64, 128]);

        let failed = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(
            Arc::new(RecordingLexical {
                limits: Mutex::new(Vec::new()),
                fail_after_first: true,
            }),
        );
        assert!(matches!(
            failed.search_collect_with_filter(&query, 2, Some(&allowed)),
            Err(SearchError::Cancelled { ref phase, .. }) if phase == "lexical-refill"
        ));
    }

    #[test]
    fn lexical_refill_counts_distinct_documents() {
        use frankensearch_core::filter::DocTypeFilter;

        let mut duplicate = lexical_result("repeat", 9.0);
        duplicate.metadata = Some(Arc::new(serde_json::json!({"doc_type": "keep"})));
        let mut blocked = lexical_result("blocked", 8.0);
        blocked.metadata = Some(Arc::new(serde_json::json!({"doc_type": "blocked"})));
        let mut tail = lexical_result("tail", 7.0);
        tail.metadata = Some(Arc::new(serde_json::json!({"doc_type": "keep"})));
        let searcher = SyncTwoTierSearcher::new(
            make_index(),
            TwoTierConfig {
                candidate_multiplier: 1,
                fast_only: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(StaticLexical {
            hits: vec![lexical_result("repeat", 10.0), duplicate, blocked, tail],
        }));
        let filter = DocTypeFilter::new(["keep"]);
        let query = fast_only_query(vec![0.0, 0.0]);
        let (results, metrics) = searcher
            .search_collect_with_filter(&query, 2, Some(&filter))
            .unwrap();

        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["repeat", "tail"]
        );
        assert_eq!(metrics.lexical_candidates, 2);
        assert_eq!(results[0].lexical_score, Some(10.0));
        assert_eq!(results[0].metadata.as_deref().unwrap()["doc_type"], "keep");

        let (unfiltered, metrics) = searcher.search_collect(&query, 2).unwrap();
        assert_eq!(
            unfiltered
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["repeat", "blocked"]
        );
        assert_eq!(metrics.lexical_candidates, 2);
        assert_eq!(unfiltered[0].lexical_score, Some(10.0));
        assert_eq!(
            unfiltered[0].metadata.as_deref().unwrap()["doc_type"],
            "keep"
        );
    }

    #[test]
    fn metadata_filter_refill_counts_unique_documents_and_keeps_best_rows() {
        use frankensearch_core::filter::DocTypeFilter;

        struct MetadataSource;
        impl SyncLexicalSearch for MetadataSource {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                Ok(Vec::new())
            }

            fn metadata_for_documents(
                &self,
                ids: &[&str],
            ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
                Ok(ids
                    .iter()
                    .map(|id| {
                        Some(Arc::new(serde_json::json!({
                            "doc_type": if *id == "blocked" { "blocked" } else { "keep" }
                        })))
                    })
                    .collect())
            }
        }

        let fast = InMemoryVectorIndex::from_vectors(
            ["repeat", "repeat", "repeat", "repeat", "blocked", "tail"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            [1.0_f32, 0.9, 0.8, 0.7, 0.6, 0.5]
                .into_iter()
                .map(|score| vec![score, (1.0 - score * score).sqrt()])
                .collect(),
            2,
        )
        .unwrap();
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(fast, None).unwrap()),
            TwoTierConfig {
                candidate_multiplier: 1,
                fast_only: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(MetadataSource));
        let filter = DocTypeFilter::new(["keep"]);
        let (results, _) = searcher
            .search_collect_with_filter(&fast_only_query(vec![1.0, 0.0]), 2, Some(&filter))
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["repeat", "tail"],
            "physical duplicate leaders cannot consume the eligible document budget",
        );
        assert_eq!(results[0].fast_score, Some(1.0));
        assert_eq!(results[0].index, Some(0));
        assert!(results.iter().all(|hit| {
            hit.metadata.as_deref().unwrap()["doc_type"] == "keep" && hit.lexical_score.is_none()
        }));
    }

    #[test]
    fn metadata_filter_keeps_unknowns_and_rejects_known_eager_observations() {
        use frankensearch_core::filter::DocTypeFilter;

        let fast = InMemoryVectorIndex::from_vectors(
            vec![
                "blocked".to_owned(),
                "unknown".to_owned(),
                "bare".to_owned(),
            ],
            vec![vec![1.0, 0.0], vec![0.8, 0.6], vec![0.6, 0.8]],
            2,
        )
        .unwrap();
        let lexical = StaticLexical {
            hits: vec![ScoredResult {
                doc_id: "blocked".into(),
                score: 1.0,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(1.0),
                rerank_score: None,
                explanation: None,
                metadata: Some(Arc::new(serde_json::json!({"doc_type": "blocked"}))),
            }],
        };
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::new(fast, None).unwrap()),
            TwoTierConfig {
                candidate_multiplier: 1,
                fast_only: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(lexical));
        let (results, _) = searcher
            .search_collect_with_filter(
                &fast_only_query(vec![1.0, 0.0]),
                3,
                Some(&DocTypeFilter::new(["keep"])),
            )
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["unknown", "bare"],
            "known lexical rejection applies to semantic hits; unknown metadata stays permissive",
        );
        assert!(results.iter().all(|hit| hit.metadata.is_none()));
    }

    #[test]
    fn strict_metadata_filters_refill_one_percent_tail_in_every_sync_phase() {
        use frankensearch_core::filter::{
            DateRangeFilter, DocTypeFilter, FilterChain, FilterMode, MissingMetadataPolicy,
        };

        struct CorpusMetadata {
            lexical_active_matches: bool,
        }
        impl SyncLexicalSearch for CorpusMetadata {
            fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
                let count = if self.lexical_active_matches {
                    1_000
                } else {
                    990
                };
                Ok((0..count.min(limit))
                    .map(|rank| lexical_result(&format!("doc-{rank:04}"), (1_000 - rank) as f32))
                    .collect())
            }

            fn metadata_for_documents(
                &self,
                ids: &[&str],
            ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
                Ok(ids
                    .iter()
                    .map(|id| {
                        let rank = id.strip_prefix("doc-")?.parse::<usize>().ok()?;
                        (rank < 1_000).then(|| {
                            Arc::new(serde_json::json!({
                                "doc_type": if rank < 990 { "archived" } else { "active" },
                                "created_at": if rank < 990 { 1 } else { 200 },
                            }))
                        })
                    })
                    .collect())
            }
        }

        let rows: Vec<_> = (0..=1_000)
            .map(|rank| {
                let id = if rank == 0 {
                    "metadata-unavailable".to_owned()
                } else {
                    format!("doc-{:04}", rank - 1)
                };
                let score = 1.0 - rank as f32 * 0.0004;
                let other = (1.0 - score * score).sqrt();
                (id, vec![score, other], vec![0.0, score, other])
            })
            .collect();
        let fast_rows: Vec<_> = rows
            .iter()
            .map(|(id, fast, _)| (id.as_str(), fast.as_slice()))
            .collect();
        let quality_rows: Vec<_> = rows
            .iter()
            .map(|(id, _, quality)| (id.as_str(), quality.as_slice()))
            .collect();
        let directory = temp_dir("strict-metadata-tail");
        let (fast, fast_identity) =
            attested_tier(directory.path(), "fast.fsvi", "strict-fast", 1, &fast_rows);
        let (quality, quality_identity) = attested_tier(
            directory.path(),
            "quality.fsvi",
            "strict-quality",
            1,
            &quality_rows,
        );
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)).unwrap());
        let fast_query = BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).unwrap();
        let quality_query =
            BoundQueryEmbedding::new(vec![0.0, 1.0, 0.0], quality_identity).unwrap();
        let queries = [
            TieredQueryEmbeddings::fast_only(fast_query.clone()),
            TieredQueryEmbeddings::quality_only(quality_query.clone()),
            TieredQueryEmbeddings::progressive(fast_query, quality_query),
        ];
        let types =
            DocTypeFilter::new(["active"]).on_missing_metadata(MissingMetadataPolicy::Reject);
        let dates =
            DateRangeFilter::between(100, 300).on_missing_metadata(MissingMetadataPolicy::Reject);
        let both = FilterChain::new(FilterMode::All)
            .with(Box::new(types.clone()))
            .with(Box::new(dates));
        let config = TwoTierConfig {
            candidate_multiplier: 1,
            ..TwoTierConfig::default()
        };
        let without_metadata = SyncTwoTierSearcher::new(Arc::clone(&index), config.clone());
        for query in &queries {
            for filter in [&types as &dyn SearchFilter, &dates, &both] {
                let (hits, _) = without_metadata
                    .search_collect_with_filter(query, 5, Some(filter))
                    .unwrap();
                assert!(
                    hits.is_empty(),
                    "strict filters cannot authorize unknown rows"
                );
            }
            let (permissive, _) = without_metadata
                .search_collect_with_filter(query, 5, Some(&DocTypeFilter::new(["active"])))
                .unwrap();
            assert_eq!(permissive.len(), 5, "missing-metadata rejection is opt-in");
            assert_eq!(permissive[0].doc_id, "metadata-unavailable");
        }

        for lexical_active_matches in [false, true] {
            let searcher = SyncTwoTierSearcher::new(Arc::clone(&index), config.clone())
                .with_nqc_dense_downweight_disabled()
                .with_lexical(Arc::new(CorpusMetadata {
                    lexical_active_matches,
                }));
            for query in &queries {
                for filter in [&types as &dyn SearchFilter, &dates, &both] {
                    let assert_hits = |hits: &[ScoredResult]| {
                        assert_eq!(hits.len(), 5, "the eligible tail must fill top-k");
                        assert!(hits.iter().all(|hit| {
                            hit.metadata.as_deref().is_some_and(|metadata| {
                                metadata["doc_type"] == "active" && metadata["created_at"] == 200
                            }) && hit.lexical_score.is_some() == lexical_active_matches
                        }));
                    };
                    let (hits, _) = searcher
                        .search_collect_with_filter(query, 5, Some(filter))
                        .unwrap();
                    assert_hits(&hits);
                    let phases: Vec<_> = searcher
                        .search_iter_with_filter(query, 5, Some(filter))
                        .collect();
                    assert!(!phases.is_empty());
                    for phase in phases {
                        match phase {
                            SearchPhase::Initial { results, .. }
                            | SearchPhase::Refined { results, .. } => assert_hits(&results),
                            other => panic!("unexpected strict-filter phase: {other:?}"),
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn metadata_filter_hydrates_eager_unknowns_before_accepting_lexical_hits() {
        use frankensearch_core::filter::{DateRangeFilter, DocTypeFilter};
        use std::sync::atomic::{AtomicBool, Ordering};

        struct LazyMetadata {
            fail_hydration: bool,
        }
        impl SyncLexicalSearch for LazyMetadata {
            fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
                Ok([
                    lexical_result("b", 4.0),
                    lexical_result("lexical-blocked", 3.0),
                    lexical_result("a", 2.0),
                    lexical_result("c", 1.0),
                ]
                .into_iter()
                .take(limit)
                .collect())
            }

            fn metadata_for_documents(
                &self,
                ids: &[&str],
            ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
                if self.fail_hydration {
                    return Err(SearchError::InvalidConfig {
                        field: "fixture.metadata".to_owned(),
                        value: "unreadable".to_owned(),
                        reason: "metadata decoding failed".to_owned(),
                    });
                }
                Ok(ids
                    .iter()
                    .map(|id| {
                        let blocked = matches!(*id, "b" | "lexical-blocked");
                        Some(Arc::new(serde_json::json!({
                            "doc_type": if blocked { "blocked" } else { "keep" },
                            "created_at": if blocked { 1 } else { 200 },
                        })))
                    })
                    .collect())
            }
        }

        struct RequireMetadata;
        impl SearchFilter for RequireMetadata {
            fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
                metadata
                    .and_then(|value| value.get("doc_type"))
                    .is_some_and(|value| value == "keep")
            }

            fn name(&self) -> &'static str {
                "require-metadata"
            }
        }

        let directory = temp_dir("eager-unknown-metadata");
        let (fast, quality) = persisted_producer_pair(directory.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let progressive = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality.clone()).unwrap(),
        );
        let quality_only = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let types = DocTypeFilter::new(["keep"]);
        let dates = DateRangeFilter::after(100);
        let strict = RequireMetadata;
        for fail_hydration in [false, true] {
            let searcher = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_nqc_dense_downweight_disabled()
            .with_lexical(Arc::new(LazyMetadata { fail_hydration }));
            for filter in [&types as &dyn SearchFilter, &dates, &strict] {
                for query in [&progressive, &quality_only] {
                    if fail_hydration {
                        assert!(matches!(
                            searcher.search_collect_with_filter(query, 2, Some(filter)),
                            Err(SearchError::InvalidConfig { .. }),
                        ));
                        continue;
                    }
                    let assert_hits = |hits: &[ScoredResult]| {
                        let mut ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
                        ids.sort_unstable();
                        assert_eq!(ids, ["a", "c"]);
                        assert!(hits.iter().all(|hit| {
                            filter.matches(&hit.doc_id, hit.metadata.as_deref())
                                && hit.lexical_score.is_some()
                        }));
                    };
                    let (hits, _) = searcher
                        .search_collect_with_filter(query, 2, Some(filter))
                        .unwrap();
                    assert_hits(&hits);
                    for phase in searcher.search_iter_with_filter(query, 2, Some(filter)) {
                        match phase {
                            SearchPhase::Initial { results, .. }
                            | SearchPhase::Refined { results, .. } => assert_hits(&results),
                            other => panic!("unexpected hydrated-metadata phase: {other:?}"),
                        }
                    }
                }
            }
        }

        struct DelayedMetadata {
            available: AtomicBool,
        }
        impl SyncLexicalSearch for DelayedMetadata {
            fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
                Ok(std::iter::once(lexical_result("b", 1.0))
                    .take(limit)
                    .collect())
            }

            fn metadata_for_documents(
                &self,
                ids: &[&str],
            ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
                Ok(ids
                    .iter()
                    .map(|id| {
                        (*id != "b" || self.available.load(Ordering::Relaxed))
                            .then(|| Arc::new(serde_json::json!({"doc_type": "keep"})))
                    })
                    .collect())
            }
        }
        let delayed = Arc::new(DelayedMetadata {
            available: AtomicBool::new(false),
        });
        let searcher = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                candidate_multiplier: 1,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(delayed.clone());
        let mut phases = searcher.search_iter_with_filter(&progressive, 1, Some(&strict));
        let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
            panic!("expected Initial while b's metadata is unknown");
        };
        assert_eq!(results[0].doc_id, "a");
        delayed.available.store(true, Ordering::Relaxed);
        let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
            panic!("expected Refined after selected metadata becomes available");
        };
        assert_eq!(results[0].doc_id, "b");
        assert!(results[0].metadata.is_some());
        assert!(phases.next().is_none());
    }

    #[derive(Default)]
    struct LateRefillLexical {
        limits: Mutex<Vec<usize>>,
        initial_lookups: AtomicUsize,
        cancel_on_tail: bool,
        fail_after_revalidation_prefix: bool,
    }

    impl SyncLexicalSearch for LateRefillLexical {
        fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
            let first_request = {
                let mut limits = self.limits.lock().unwrap();
                limits.push(limit);
                limits.len() == 1
            };
            if first_request {
                return Ok(
                    vec![lexical_result("doc-a", 10.0), lexical_result("doc-b", 9.0)]
                        .into_iter()
                        .take(limit)
                        .collect(),
                );
            }
            if self.fail_after_revalidation_prefix {
                if limit > 4 {
                    return Err(SearchError::SubsystemError {
                        subsystem: "test.lexical",
                        source: "lexical revalidation refill unavailable".into(),
                    });
                }
                return Ok(["doc-c", "doc-d", "blocked-filler-a", "blocked-filler-b"]
                    .into_iter()
                    .take(limit)
                    .map(|id| {
                        let mut hit = lexical_result(id, 8.0);
                        hit.metadata = Some(Arc::new(serde_json::json!({"doc_type": "blocked"})));
                        hit
                    })
                    .collect());
            }
            let mut hits = vec![
                lexical_result("doc-c", 8.0),
                lexical_result("doc-d", 7.0),
                lexical_result("lexical-only", 6.0),
            ];
            for hit in &mut hits {
                let doc_type = if hit.doc_id == "lexical-only" {
                    "keep"
                } else {
                    "blocked"
                };
                hit.metadata = Some(Arc::new(serde_json::json!({"doc_type": doc_type})));
            }
            hits.truncate(limit);
            Ok(hits)
        }

        fn metadata_for_documents(
            &self,
            ids: &[&str],
        ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
            if self.cancel_on_tail && ids.iter().any(|id| id.starts_with("tail-")) {
                return Err(SearchError::Cancelled {
                    phase: "sync-filter-tail".to_owned(),
                    reason: "cancelled while hydrating a replacement candidate".to_owned(),
                });
            }
            Ok(ids
                .iter()
                .map(|id| {
                    let doc_type = match *id {
                        "doc-a" | "doc-b" => {
                            if self.initial_lookups.fetch_add(1, Ordering::Relaxed) < 2 {
                                return None;
                            }
                            "blocked"
                        }
                        // Only the lexical refill supplies these observations.
                        // A retained None must not erase its eager metadata.
                        "doc-c" | "doc-d" => return None,
                        _ => "keep",
                    };
                    Some(Arc::new(serde_json::json!({"doc_type": doc_type})))
                })
                .collect())
        }
    }

    fn late_refill_index(
        mode: &str,
        attested: bool,
        include_tail: bool,
    ) -> (
        tempfile::TempDir,
        Arc<InMemoryTwoTierIndex>,
        TieredQueryEmbeddings,
    ) {
        let directory = temp_dir("late-refill");
        let fast = semantic_identity("late-refill-fast", 2);
        let quality = semantic_identity("late-refill-quality", 3);
        let fast_rows: &[(&str, &[f32])] = &[
            ("doc-a", &[1.0, 0.0]),
            ("doc-b", &[0.9, 0.435_889_9]),
            ("doc-c", &[0.8, 0.6]),
            ("doc-d", &[0.7, 0.714_142_8]),
            ("tail-a", &[0.6, 0.8]),
            ("tail-b", &[0.5, 0.866_025_4]),
        ];
        let progressive_fast_rows: &[(&str, &[f32])] =
            &[("fast-only-a", &[0.8, 0.6]), ("fast-only-b", &[0.6, 0.8])];
        let quality_rows: &[(&str, &[f32])] = &[
            ("doc-a", &[1.0, 0.0, 0.0]),
            ("doc-b", &[0.9, 0.435_889_9, 0.0]),
            ("doc-c", &[0.8, 0.6, 0.0]),
            ("doc-d", &[0.7, 0.714_142_8, 0.0]),
            ("tail-a", &[0.6, 0.8, 0.0]),
            ("tail-b", &[0.5, 0.866_025_4, 0.0]),
            ("fast-only-a", &[0.2, 0.979_795_9, 0.0]),
            ("fast-only-b", &[0.1, 0.994_987_4, 0.0]),
        ];
        let fast_rows = if mode == "progressive" {
            progressive_fast_rows
        } else if include_tail {
            fast_rows
        } else {
            &fast_rows[..4]
        };
        let quality_rows = if mode == "progressive" {
            quality_rows
        } else if include_tail {
            &quality_rows[..6]
        } else {
            &quality_rows[..4]
        };
        let index = if attested {
            let fast_index =
                attested_tier_with_identity(directory.path(), "fast.fsvi", 1, fast_rows, &fast);
            let quality_index = attested_tier_with_identity(
                directory.path(),
                "quality.fsvi",
                1,
                quality_rows,
                &quality,
            );
            InMemoryTwoTierIndex::new(fast_index, Some(quality_index)).unwrap()
        } else {
            let mut writer = frankensearch_index::TwoTierIndex::create(
                directory.path(),
                TwoTierConfig::default(),
            )
            .unwrap();
            writer.set_fast_embedder_id("late-refill-fast");
            writer.set_quality_embedder_id("late-refill-quality");
            writer.set_fast_identity(&fast).unwrap();
            writer.set_quality_identity(&quality).unwrap();
            for (id, vector) in fast_rows {
                writer.add_fast_record(*id, vector).unwrap();
            }
            for (id, vector) in quality_rows {
                writer.add_quality_record(*id, vector).unwrap();
            }
            drop(writer.finish().unwrap());
            InMemoryTwoTierIndex::from_dir(directory.path()).unwrap()
        };
        let fast_query = BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap();
        let quality_query = BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap();
        let query = match mode {
            "fast" => TieredQueryEmbeddings::fast_only(fast_query),
            "quality" => TieredQueryEmbeddings::quality_only(quality_query),
            "progressive" => TieredQueryEmbeddings::progressive(fast_query, quality_query),
            _ => panic!("unknown late-refill fixture mode"),
        };
        (directory, Arc::new(index), query)
    }

    #[test]
    fn lexical_refill_failure_preserves_known_filter_rejections() {
        use frankensearch_core::filter::DocTypeFilter;

        #[derive(Default)]
        struct PrefixOutage {
            limits: Mutex<Vec<usize>>,
        }

        impl SyncLexicalSearch for PrefixOutage {
            fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
                let first = {
                    let mut limits = self.limits.lock().unwrap();
                    limits.push(limit);
                    limits.len() == 1
                };
                if !first {
                    return Err(SearchError::SubsystemError {
                        subsystem: "test.lexical",
                        source: "lexical refill unavailable".into(),
                    });
                }
                Ok(["doc-a", "doc-b"]
                    .into_iter()
                    .take(limit)
                    .map(|id| {
                        let mut hit = lexical_result(id, 10.0);
                        hit.metadata = Some(Arc::new(serde_json::json!({"doc_type": "blocked"})));
                        hit
                    })
                    .collect())
            }
        }

        for mode in ["fast", "quality", "progressive"] {
            for attested in [false, true] {
                let (_directory, index, query) = late_refill_index(mode, attested, true);
                for streamed in [false, true] {
                    let lexical = Arc::new(PrefixOutage::default());
                    let searcher = SyncTwoTierSearcher::new(
                        Arc::clone(&index),
                        TwoTierConfig {
                            candidate_multiplier: 1,
                            quality_weight: 1.0,
                            ..TwoTierConfig::default()
                        },
                    )
                    .with_lexical(lexical.clone());
                    let filter = DocTypeFilter::new(["keep"]);
                    let results = if streamed {
                        let phases: Vec<_> = searcher
                            .search_iter_with_filter(&query, 2, Some(&filter))
                            .collect();
                        assert_eq!(phases.len(), if mode == "progressive" { 2 } else { 1 });
                        for phase in &phases {
                            match phase {
                                SearchPhase::Initial { results, .. }
                                | SearchPhase::Refined { results, .. } => {
                                    assert!(results.iter().all(|hit| {
                                        !matches!(hit.doc_id.as_str(), "doc-a" | "doc-b")
                                            && hit.lexical_score.is_none()
                                    }));
                                }
                                other => panic!("optional lexical outage failed search: {other:?}"),
                            }
                        }
                        match phases.into_iter().last().unwrap() {
                            SearchPhase::Initial { results, .. }
                            | SearchPhase::Refined { results, .. } => results,
                            other => panic!("unexpected final phase: {other:?}"),
                        }
                    } else {
                        let (results, metrics) = searcher
                            .search_collect_with_filter(&query, 2, Some(&filter))
                            .expect("semantic fallback retains known filter observations");
                        assert_eq!(metrics.lexical_candidates, 0);
                        results
                    };
                    assert_eq!(
                        results
                            .iter()
                            .map(|hit| hit.doc_id.as_str())
                            .collect::<Vec<_>>(),
                        ["doc-c", "doc-d"],
                        "{mode}, attested={attested}, streamed={streamed}",
                    );
                    assert!(results.iter().all(|hit| hit.lexical_score.is_none()));
                    assert_eq!(*lexical.limits.lock().unwrap(), [2, 4]);
                }
            }
        }
    }

    #[test]
    fn lexical_revalidation_failure_preserves_new_filter_rejections() {
        use frankensearch_core::filter::DocTypeFilter;

        for mode in ["fast", "quality", "progressive"] {
            for attested in [false, true] {
                let (_directory, index, query) = late_refill_index(mode, attested, true);
                for streamed in [false, true] {
                    let lexical = Arc::new(LateRefillLexical {
                        fail_after_revalidation_prefix: true,
                        ..LateRefillLexical::default()
                    });
                    let searcher = SyncTwoTierSearcher::new(
                        Arc::clone(&index),
                        TwoTierConfig {
                            candidate_multiplier: 1,
                            quality_weight: 1.0,
                            ..TwoTierConfig::default()
                        },
                    )
                    .with_lexical(lexical.clone());
                    let filter = DocTypeFilter::new(["keep"]);
                    let results = if streamed {
                        let phases: Vec<_> = searcher
                            .search_iter_with_filter(&query, 2, Some(&filter))
                            .collect();
                        assert_eq!(phases.len(), if mode == "progressive" { 2 } else { 1 });
                        match phases.into_iter().last().unwrap() {
                            SearchPhase::Initial { results, .. }
                            | SearchPhase::Refined { results, .. } => results,
                            other => panic!("lexical revalidation failed search: {other:?}"),
                        }
                    } else {
                        let (results, metrics) = searcher
                            .search_collect_with_filter(&query, 2, Some(&filter))
                            .expect("semantic refill retains newly learned rejections");
                        assert_eq!(metrics.lexical_candidates, 0);
                        results
                    };
                    assert_eq!(
                        results
                            .iter()
                            .map(|hit| hit.doc_id.as_str())
                            .collect::<Vec<_>>(),
                        ["tail-a", "tail-b"],
                        "{mode}, attested={attested}, streamed={streamed}",
                    );
                    assert!(results.iter().all(|hit| {
                        hit.lexical_score.is_none()
                            && hit.metadata.as_deref().unwrap()["doc_type"] == "keep"
                    }));
                    assert_eq!(*lexical.limits.lock().unwrap(), [2, 4, 8]);
                }
            }
        }
    }

    #[test]
    fn sync_filter_late_lexical_metadata_refills_each_semantic_tier() {
        use frankensearch_core::filter::DocTypeFilter;

        for mode in ["fast", "quality", "progressive"] {
            for attested in [false, true] {
                let (_directory, index, query) = late_refill_index(mode, attested, true);
                for parallel_enabled in [false, true] {
                    for streamed in [false, true] {
                        for cancel_on_tail in [false, true] {
                            let lexical = Arc::new(LateRefillLexical {
                                cancel_on_tail,
                                ..LateRefillLexical::default()
                            });
                            let searcher = SyncTwoTierSearcher::new(
                                Arc::clone(&index),
                                TwoTierConfig {
                                    candidate_multiplier: 1,
                                    quality_weight: 1.0,
                                    ..TwoTierConfig::default()
                                },
                            )
                            .with_search_params(SearchParams {
                                parallel_threshold: 1,
                                parallel_chunk_size: 1,
                                parallel_enabled,
                            })
                            .with_lexical(lexical.clone());
                            let filter = DocTypeFilter::new(["keep"]);
                            let outcome = if streamed {
                                let mut phases: Vec<_> = searcher
                                    .search_iter_with_filter(&query, 2, Some(&filter))
                                    .collect();
                                if cancel_on_tail && mode == "progressive" {
                                    assert!(matches!(
                                        phases.first(),
                                        Some(SearchPhase::Initial { .. })
                                    ));
                                }
                                match phases.pop().expect("search emits a terminal phase") {
                                    SearchPhase::Initial { results, .. }
                                    | SearchPhase::Refined { results, .. } => Ok(results),
                                    SearchPhase::RefinementFailed { error, .. } => Err(error),
                                    other @ SearchPhase::Reranked { .. } => {
                                        panic!("unexpected final phase: {other:?}")
                                    }
                                }
                            } else {
                                searcher
                                    .search_collect_with_filter(&query, 2, Some(&filter))
                                    .map(|(hits, _)| hits)
                            };
                            assert_eq!(*lexical.limits.lock().unwrap(), vec![2, 4]);
                            if cancel_on_tail {
                                assert!(
                                    matches!(outcome, Err(SearchError::Cancelled { .. })),
                                    "{mode}, attested={attested}, parallel={parallel_enabled}, \
                                     streamed={streamed}: {outcome:?}"
                                );
                                continue;
                            }
                            let hits = outcome.expect("stable metadata refill succeeds");
                            assert_eq!(hits.len(), 2, "{mode}, attested={attested}");
                            assert!(hits.iter().any(|hit| hit.doc_id.starts_with("tail-")));
                            assert!(hits.iter().all(|hit| {
                                !matches!(
                                    hit.doc_id.as_str(),
                                    "doc-a" | "doc-b" | "doc-c" | "doc-d"
                                )
                            }));
                            assert!(hits.iter().all(|hit| {
                                hit.metadata.as_deref().unwrap()["doc_type"] == "keep"
                            }));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sync_filter_late_rejection_stops_at_exhaustion_and_leaves_unfiltered_work_unchanged() {
        use frankensearch_core::filter::DocTypeFilter;

        for mode in ["fast", "quality"] {
            for attested in [false, true] {
                let (_directory, index, query) = late_refill_index(mode, attested, false);
                let config = TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                };
                let lexical = Arc::new(LateRefillLexical::default());
                let searcher = SyncTwoTierSearcher::new(Arc::clone(&index), config.clone())
                    .with_lexical(lexical.clone());
                let filter = DocTypeFilter::new(["keep"]);
                let (hits, metrics) = searcher
                    .search_collect_with_filter(&query, 2, Some(&filter))
                    .unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0].doc_id, "lexical-only");
                assert_eq!(metrics.lexical_candidates, 1);
                assert_eq!(metrics.semantic_candidates, 0);
                assert_eq!(
                    metrics.zero_signal,
                    Some(ZeroSignalReason::FilterEliminatedAll)
                );
                assert_eq!(*lexical.limits.lock().unwrap(), vec![2, 4]);

                let unfiltered = Arc::new(LateRefillLexical::default());
                let searcher = SyncTwoTierSearcher::new(Arc::clone(&index), config)
                    .with_lexical(unfiltered.clone());
                let (hits, _) = searcher.search_collect(&query, 2).unwrap();
                let ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
                assert_eq!(ids, ["doc-a", "doc-b"]);
                assert_eq!(*unfiltered.limits.lock().unwrap(), vec![2]);
                assert_eq!(unfiltered.initial_lookups.load(Ordering::Relaxed), 0);
                assert!(hits.iter().all(|hit| hit.metadata.is_none()));
            }
        }
    }

    #[test]
    fn sync_filter_unstable_predicate_cannot_loop_during_reconciliation() {
        struct AlternatingFilter(AtomicUsize);
        impl SearchFilter for AlternatingFilter {
            fn matches(&self, _: &str, _: Option<&serde_json::Value>) -> bool {
                !self.0.fetch_add(1, Ordering::Relaxed).is_multiple_of(2)
            }

            fn name(&self) -> &'static str {
                "alternating-test-filter"
            }
        }

        let mut context =
            SyncLexicalContext::new(Some(Arc::new(StaticLexical { hits: Vec::new() })), true)
                .unwrap();
        context.metadata.insert(
            "candidate".into(),
            Some(Arc::new(serde_json::json!({"doc_type": "keep"}))),
        );
        let hit = VectorHit {
            index: 0,
            score: 1.0,
            doc_id: "candidate".into(),
        };
        let mut classified = ClassifiedHits {
            hits: vec![hit.clone()],
            zero_signal: None,
        };
        let filter = AlternatingFilter(AtomicUsize::new(0));
        let mut retrievals = 0;
        let error = reconcile_filtered_vectors(
            &mut context,
            &[1.0, 0.0],
            None,
            &mut classified,
            1,
            1,
            Some(&filter),
            &mut AHashSet::new(),
            |_, _| {
                retrievals += 1;
                Ok(ClassifiedHits {
                    hits: vec![hit.clone()],
                    zero_signal: None,
                })
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, value, .. }
                if field == "search.filter" && value == "unstable_predicate"
        ));
        assert_eq!(retrievals, 1);
        assert_eq!(filter.0.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn sync_filter_quality_metadata_refills_a_revoked_fast_window() {
        use frankensearch_core::filter::DocTypeFilter;

        for include_tail in [false, true] {
            let directory = temp_dir("late-fast-window");
            let fast = semantic_identity("late-fast-window-fast", 2);
            let quality = semantic_identity("late-fast-window-quality", 3);
            let fast_rows: &[(&str, &[f32])] = &[("a", &[1.0, 0.0]), ("fast-tail", &[0.8, 0.6])];
            let fast_rows = if include_tail {
                fast_rows
            } else {
                &fast_rows[..1]
            };
            let fast_index =
                attested_tier_with_identity(directory.path(), "fast.fsvi", 1, fast_rows, &fast);
            let quality_index = attested_tier_with_identity(
                directory.path(),
                "quality.fsvi",
                1,
                &[("a", &[1.0, 0.0, 0.0])],
                &quality,
            );
            let index =
                Arc::new(InMemoryTwoTierIndex::new(fast_index, Some(quality_index)).unwrap());
            let lexical = Arc::new(LateBlockingLexical {
                blocked_id: "a",
                unknown_reads: AtomicUsize::new(usize::MAX),
                include_tail: false,
            });
            let searcher = SyncTwoTierSearcher::new(
                index,
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_lexical(lexical.clone());
            let query = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
            );
            let filter = DocTypeFilter::new(["keep"]);
            let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
            let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                panic!("expected Initial before quality metadata becomes available");
            };
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "a");
            assert!(results[0].metadata.is_none());

            lexical.unknown_reads.store(0, Ordering::Relaxed);
            let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
                panic!("expected Refined after the fast window is revalidated");
            };
            assert_eq!(results.len(), usize::from(include_tail));
            if include_tail {
                assert_eq!(results[0].doc_id, "fast-tail");
                assert!(results[0].fast_score.is_some());
                assert_eq!(results[0].quality_score, None);
                assert_eq!(results[0].lexical_score, None);
                assert_eq!(results[0].metadata.as_deref().unwrap()["doc_type"], "keep");
            }
            assert!(phases.next().is_none());

            // Collection performs two unknown lookups in Initial (lexical and
            // fast); quality then learns the rejection without a phase handoff.
            lexical.unknown_reads.store(2, Ordering::Relaxed);
            let (collected, metrics) = searcher
                .search_collect_with_filter(&query, 1, Some(&filter))
                .unwrap();
            assert_eq!(
                collected.iter().map(|hit| &hit.doc_id).collect::<Vec<_>>(),
                results.iter().map(|hit| &hit.doc_id).collect::<Vec<_>>()
            );
            assert_eq!(
                metrics.zero_signal,
                if include_tail {
                    None
                } else {
                    Some(ZeroSignalReason::FilterEliminatedAll)
                }
            );
        }
    }

    struct LateBlockingLexical {
        blocked_id: &'static str,
        unknown_reads: AtomicUsize,
        include_tail: bool,
    }

    impl SyncLexicalSearch for LateBlockingLexical {
        fn search_sync(&self, _: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
            let mut leading = lexical_result(self.blocked_id, 10.0);
            if self.include_tail && self.unknown_reads.load(Ordering::Relaxed) == 0 {
                // An unpinned refill may offer an older eager observation.
                leading.metadata = Some(Arc::new(serde_json::json!({"doc_type": "keep"})));
            }
            let tail = self
                .include_tail
                .then(|| lexical_result("lexical-tail", 9.0));
            Ok(std::iter::once(leading).chain(tail).take(limit).collect())
        }

        fn metadata_for_documents(
            &self,
            ids: &[&str],
        ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
            Ok(ids
                .iter()
                .map(|id| {
                    let doc_type = if *id == self.blocked_id {
                        if self
                            .unknown_reads
                            .try_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                                remaining.checked_sub(1)
                            })
                            .is_ok()
                        {
                            return None;
                        }
                        "blocked"
                    } else {
                        "keep"
                    };
                    Some(Arc::new(serde_json::json!({"doc_type": doc_type})))
                })
                .collect())
        }
    }

    #[test]
    fn late_blocking_metadata_updates_initial_lexical_eligibility() {
        use frankensearch_core::filter::DocTypeFilter;

        let directory = temp_dir("late-blocking-initial");
        let (fast, quality) = persisted_producer_pair(directory.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let filter = DocTypeFilter::new(["keep"]);
        let quality_only = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let fast_only = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
        );
        for (query, blocked_id, expected) in
            [(&quality_only, "b", &["c"][..]), (&fast_only, "a", &[])]
        {
            let searcher = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_lexical(Arc::new(LateBlockingLexical {
                blocked_id,
                unknown_reads: AtomicUsize::new(1),
                include_tail: false,
            }));
            let (hits, metrics) = searcher
                .search_collect_with_filter(query, 1, Some(&filter))
                .unwrap();
            let ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
            assert_eq!(ids, expected, "newly known rejection for {blocked_id}");
            assert_eq!(metrics.lexical_candidates, 0);
            assert!(
                hits.iter()
                    .all(|hit| filter.matches(&hit.doc_id, hit.metadata.as_deref()))
            );
        }
    }

    #[test]
    fn late_blocking_metadata_updates_retained_pools_before_refinement() {
        use frankensearch_core::filter::DocTypeFilter;

        let directory = temp_dir("late-blocking-refined");
        let (fast, quality) = persisted_producer_pair(directory.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let lexical = Arc::new(LateBlockingLexical {
            blocked_id: "a",
            unknown_reads: AtomicUsize::new(usize::MAX),
            include_tail: false,
        });
        let searcher = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                candidate_multiplier: 1,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(lexical.clone());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![0.5, 0.0, 0.866], quality).unwrap(),
        );
        let filter = DocTypeFilter::new(["keep"]);
        let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
        let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
            panic!("expected Initial while a's metadata is unknown");
        };
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "a");
        assert!(results[0].fast_score.is_some() && results[0].lexical_score.is_some());
        assert!(results[0].metadata.is_none());

        lexical.unknown_reads.store(0, Ordering::Relaxed);
        let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
            panic!("expected Refined after a's metadata becomes available");
        };
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[0].metadata.as_deref().unwrap()["doc_type"], "keep");
        assert!(results[0].quality_score.is_some());
        assert_eq!(results[0].fast_score, None);
        assert_eq!(results[0].lexical_score, None);
        assert_eq!(results[0].source, ScoreSource::SemanticQuality);
        assert_eq!(results[0].index, None);
        assert!(phases.next().is_none());
    }

    #[test]
    fn late_blocking_metadata_refills_the_vacated_lexical_slot() {
        use frankensearch_core::filter::DocTypeFilter;

        let directory = temp_dir("late-blocking-lexical-tail");
        let (fast, _) = persisted_producer_pair(directory.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let searcher = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                candidate_multiplier: 1,
                fast_only: true,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(LateBlockingLexical {
            blocked_id: "a",
            unknown_reads: AtomicUsize::new(1),
            include_tail: true,
        }));
        let query = TieredQueryEmbeddings::fast_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
        );
        let filter = DocTypeFilter::new(["keep"]);
        let (hits, metrics) = searcher
            .search_collect_with_filter(&query, 1, Some(&filter))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "lexical-tail");
        assert_eq!(hits[0].source, ScoreSource::Lexical);
        assert_eq!(hits[0].metadata.as_deref().unwrap()["doc_type"], "keep");
        assert_eq!(metrics.lexical_candidates, 1);
        assert_eq!(metrics.semantic_candidates, 0);
    }

    /// The initial fast pool is admitted while metadata is unknown. Quality
    /// retrieval then resolves a rejection before a wider prefix's lookup fails.
    struct RefinementMetadataFailure {
        revoked: Option<&'static str>,
        cancelled: bool,
        lookups: AtomicUsize,
    }

    impl RefinementMetadataFailure {
        fn error(&self) -> SearchError {
            if self.cancelled {
                SearchError::Cancelled {
                    phase: "quality-refill-metadata".to_owned(),
                    reason: "fixture cancellation after metadata resolution".to_owned(),
                }
            } else {
                SearchError::InvalidConfig {
                    field: "quality-refill.metadata".to_owned(),
                    value: "unavailable".to_owned(),
                    reason: "fixture failure after metadata resolution".to_owned(),
                }
            }
        }
    }

    impl SyncLexicalSearch for RefinementMetadataFailure {
        fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
            Ok(Vec::new())
        }

        fn metadata_for_documents(
            &self,
            ids: &[&str],
        ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
            let first = self.lookups.fetch_add(1, Ordering::Relaxed) == 0;
            if ids.contains(&"replacement") {
                return Err(self.error());
            }
            Ok(ids
                .iter()
                .map(|id| {
                    if first {
                        return None;
                    }
                    let doc_type = if *id == "quality-blocked" || self.revoked == Some(*id) {
                        "blocked"
                    } else {
                        "keep"
                    };
                    Some(Arc::new(serde_json::json!({ "doc_type": doc_type })))
                })
                .collect())
        }
    }

    #[test]
    fn sync_mmr_preserves_known_metadata_after_refinement_failure() {
        use std::sync::atomic::AtomicBool;

        struct MetadataPolicy(AtomicBool);
        impl SearchFilter for MetadataPolicy {
            fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
                metadata.is_none_or(|value| {
                    value.get("doc_type").is_some_and(|kind| kind == "keep")
                        && !self.0.load(Ordering::Relaxed)
                })
            }

            fn name(&self) -> &'static str {
                "changed-metadata-policy"
            }
        }

        let directory = temp_dir("mmr-fallback-metadata");
        let fast = semantic_identity("fallback-fast", 2);
        let quality = semantic_identity("fallback-quality", 3);
        let mut writer =
            frankensearch_index::TwoTierIndex::create(directory.path(), TwoTierConfig::default())
                .unwrap();
        writer.set_fast_embedder_id("fallback-fast");
        writer.set_quality_embedder_id("fallback-quality");
        writer.set_fast_identity(&fast).unwrap();
        writer.set_quality_identity(&quality).unwrap();
        writer.add_fast_record("shown", &[1.0, 0.0]).unwrap();
        writer.add_fast_record("hidden", &[0.8, 0.6]).unwrap();
        for (id, score) in [
            ("quality-blocked", 1.0_f32),
            ("shown", 0.99),
            ("hidden", 0.8),
            ("replacement", 0.6),
        ] {
            writer
                .add_quality_record(id, &[score, (1.0 - score * score).sqrt(), 0.0])
                .unwrap();
        }
        drop(writer.finish().unwrap());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let searcher = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                candidate_multiplier: 1,
                ..TwoTierConfig::default()
            },
        )
        .with_mmr_config(MmrConfig {
            enabled: true,
            lambda: 0.3,
            candidate_pool: 2,
        })
        .with_lexical(Arc::new(RefinementMetadataFailure {
            revoked: None,
            cancelled: false,
            lookups: AtomicUsize::new(0),
        }));
        let filter = MetadataPolicy(AtomicBool::new(false));
        let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
        let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
            panic!("Initial must precede metadata hydration");
        };
        assert_eq!(results[0].doc_id, "shown");
        assert!(results[0].metadata.is_none());
        let Some(SearchPhase::RefinementFailed { error, .. }) = phases.next() else {
            panic!("the wider metadata lookup must fail after resolving shown");
        };
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, .. } if field == "quality-refill.metadata"
        ));
        // Unknown metadata stays admissible, but the now-known metadata no
        // longer is. Losing that retained observation would wrongly admit MMR.
        filter.0.store(true, Ordering::Relaxed);
        let Some(SearchPhase::RefinementFailed {
            initial_results,
            error,
            ..
        }) = phases.next()
        else {
            panic!("final ranking must use the latest retained metadata");
        };
        assert!(initial_results.is_empty());
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, value, .. }
                if field == "search.filter" && value == "eligibility_changed"
        ));
        assert!(phases.next().is_none());
    }

    #[test]
    fn refinement_failure_revalidates_displayed_and_hidden_initial_candidates() {
        use frankensearch_core::filter::DocTypeFilter;

        for revoked in [None, Some("shown"), Some("hidden")] {
            let dir = temp_dir("sync-refinement-fallback");
            let fast = semantic_identity("fallback-fast", 2);
            let quality = semantic_identity("fallback-quality", 3);
            let mut writer =
                frankensearch_index::TwoTierIndex::create(dir.path(), TwoTierConfig::default())
                    .unwrap();
            writer.set_fast_embedder_id("fallback-fast");
            writer.set_quality_embedder_id("fallback-quality");
            writer.set_fast_identity(&fast).unwrap();
            writer.set_quality_identity(&quality).unwrap();
            writer.add_fast_record("shown", &[1.0, 0.0]).unwrap();
            writer.add_fast_record("hidden", &[0.8, 0.6]).unwrap();
            // The target's rejection is stored in the first quality prefix.
            // The replacement fails only in the subsequent wider prefix.
            let target = revoked.unwrap_or("shown");
            let other = if target == "shown" { "hidden" } else { "shown" };
            for (id, score) in [
                ("quality-blocked", 1.0_f32),
                (target, 0.99),
                (other, 0.8),
                ("replacement", 0.6),
            ] {
                writer
                    .add_quality_record(id, &[score, (1.0 - score * score).sqrt(), 0.0])
                    .unwrap();
            }
            drop(writer.finish().unwrap());
            let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
            let query = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
            );
            let filter = DocTypeFilter::new(["keep"]);

            for mmr in [false, true] {
                for cancelled in [false, true] {
                    // With MMR, "hidden" belongs to the retained ranking pool
                    // but is absent from the already-published Initial.
                    let k = if mmr { 1 } else { 2 };
                    let searcher = || {
                        let lexical = Arc::new(RefinementMetadataFailure {
                            revoked,
                            cancelled,
                            lookups: AtomicUsize::new(0),
                        });
                        let searcher = SyncTwoTierSearcher::new(
                            Arc::clone(&index),
                            TwoTierConfig {
                                candidate_multiplier: 1,
                                ..TwoTierConfig::default()
                            },
                        )
                        .with_nqc_dense_downweight_disabled()
                        .with_mmr_config(MmrConfig {
                            enabled: mmr,
                            lambda: 0.3,
                            candidate_pool: 2,
                        })
                        .with_lexical(lexical.clone());
                        (searcher, lexical)
                    };
                    let (collect_searcher, lexical) = searcher();
                    let collected =
                        collect_searcher.search_collect_with_filter(&query, k, Some(&filter));
                    assert_eq!(lexical.lookups.load(Ordering::Relaxed), 3);
                    if revoked.is_some() || cancelled {
                        let error = collected.expect_err(
                            "a refinement failure cannot return an invalidated Initial as success",
                        );
                        assert_eq!(format!("{error:?}"), format!("{:?}", lexical.error()));
                    } else {
                        let (results, metrics) = collected.unwrap();
                        assert_eq!(results.len(), k);
                        assert_eq!(results[0].doc_id, "shown");
                        assert_eq!(
                            metrics.skip_reason.as_deref(),
                            Some(lexical.error().diagnostic_code()),
                        );
                    }

                    let (iter_searcher, lexical) = searcher();
                    let mut phases =
                        iter_searcher.search_iter_with_filter(&query, k, Some(&filter));
                    let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                        panic!("Initial must be emitted before quality metadata is resolved");
                    };
                    assert_eq!(results.len(), k);
                    assert_eq!(results[0].doc_id, "shown");
                    assert_eq!(lexical.lookups.load(Ordering::Relaxed), 1);
                    let Some(SearchPhase::RefinementFailed {
                        initial_results,
                        error,
                        ..
                    }) = phases.next()
                    else {
                        panic!("the later metadata lookup must retain its original failure");
                    };
                    assert_eq!(format!("{error:?}"), format!("{:?}", lexical.error()));
                    assert_eq!(lexical.lookups.load(Ordering::Relaxed), 3);
                    if revoked.is_some() {
                        assert!(
                            initial_results.is_empty(),
                            "terminal failure must not republish known rejected candidates",
                        );
                    } else {
                        assert_eq!(
                            serde_json::to_value(initial_results).unwrap(),
                            serde_json::to_value(results).unwrap(),
                        );
                        if mmr && !cancelled {
                            assert!(matches!(phases.next(), Some(SearchPhase::Reranked { .. })));
                        }
                    }
                    assert!(
                        phases.next().is_none(),
                        "rejected candidates and cancelled searches must not reach final ranking",
                    );
                    assert!(phases.next().is_none());
                }
            }
        }
    }

    #[cfg(feature = "quill")]
    #[test]
    fn quill_metadata_filters_cover_semantic_lanes_and_eligible_tails() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_core::filter::{
            DocTypeFilter, ExcludeDocIdsFilter, FilterChain, FilterMode,
        };
        use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchIndex};

        struct RequireMetadata;
        impl SearchFilter for RequireMetadata {
            fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
                metadata
                    .and_then(|value| value.get("doc_type"))
                    .is_some_and(|value| value == "keep")
            }
            fn name(&self) -> &'static str {
                "require-metadata"
            }
        }

        struct NegatedType(DocTypeFilter);
        impl SearchFilter for NegatedType {
            fn matches(&self, id: &str, metadata: Option<&serde_json::Value>) -> bool {
                !self.0.matches(id, metadata)
            }

            fn name(&self) -> &'static str {
                "not-blocked"
            }
        }

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let lexical_root = directory.path().join("lexical");
            let vector_root = directory.path().join("vectors");
            let config = QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let writer = QuillIndex::create(&cx, &lexical_root, config.clone())
                .await
                .unwrap();
            let fast_identity = semantic_identity("metadata-fast", 2);
            let quality_identity = semantic_identity("metadata-quality", 3);
            let mut vectors =
                frankensearch_index::TwoTierIndex::create(&vector_root, TwoTierConfig::default())
                    .unwrap();
            vectors.set_fast_embedder_id("metadata-fast");
            vectors.set_quality_embedder_id("metadata-quality");
            vectors.set_fast_identity(&fast_identity).unwrap();
            vectors.set_quality_identity(&quality_identity).unwrap();
            let mut docs = Vec::new();
            for ordinal in 0..32 {
                let id = format!("blocked-{ordinal:02}");
                vectors.add_fast_record(&id, &[1.0, 0.0]).unwrap();
                vectors.add_quality_record(&id, &[1.0, 0.0, 0.0]).unwrap();
                docs.push(
                    IndexableDocument::new(id, if ordinal == 0 { "needle" } else { "unrelated" })
                        .with_metadata("doc_type", "blocked"),
                );
            }
            for (id, fast, quality) in [
                ("allowed-a", [0.6, 0.8], [0.0, 1.0, 0.0]),
                ("allowed-b", [0.0, 1.0], [0.6, 0.8, 0.0]),
            ] {
                vectors.add_fast_record(id, &fast).unwrap();
                vectors.add_quality_record(id, &quality).unwrap();
                docs.push(
                    IndexableDocument::new(id, "unrelated").with_metadata("doc_type", "keep"),
                );
            }
            drop(vectors.finish().unwrap());
            writer.index_documents(&cx, &docs).await.unwrap();
            writer.commit(&cx).await.unwrap();
            let reader = Arc::new(
                QuillSearchIndex::open(&cx, &lexical_root, config)
                    .await
                    .unwrap(),
            );
            let searcher = SyncTwoTierSearcher::new(
                Arc::new(InMemoryTwoTierIndex::from_dir(&vector_root).unwrap()),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_nqc_dense_downweight_disabled()
            .with_lexical(Arc::new(QuillSyncLexicalSearch::new(reader, cx, "needle")));
            let progressive = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast_identity).unwrap(),
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality_identity.clone()).unwrap(),
            );
            let quality_only = TieredQueryEmbeddings::quality_only(
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality_identity).unwrap(),
            );
            let types = DocTypeFilter::new(["keep"]);
            let exact = FilterChain::new(FilterMode::All)
                .with(Box::new(IncludeDocIdsFilter::new([
                    "blocked-00",
                    "allowed-a",
                    "allowed-b",
                ])))
                .with(Box::new(types.clone()));
            let both = FilterChain::new(FilterMode::All)
                .with(Box::new(types.clone()))
                .with(Box::new(RequireMetadata));
            let strict = RequireMetadata;
            let negated = NegatedType(DocTypeFilter::new(["blocked"]));
            for filter in [
                &types as &dyn SearchFilter,
                &exact,
                &both,
                &strict,
                &negated,
            ] {
                for query in [&progressive, &quality_only] {
                    let assert_hits = |hits: &[ScoredResult]| {
                        assert_eq!(
                            hits.len(),
                            2,
                            "eligible tail must fill the requested window"
                        );
                        let mut ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
                        ids.sort_unstable();
                        assert_eq!(ids, ["allowed-a", "allowed-b"]);
                        for hit in hits {
                            assert!(filter.matches(&hit.doc_id, hit.metadata.as_deref()));
                            assert!(hit.lexical_score.is_none(), "both winners are vector-only");
                            assert_eq!(hit.metadata.as_deref().unwrap()["doc_type"], "keep");
                        }
                    };
                    let (hits, _) = searcher
                        .search_collect_with_filter(query, 2, Some(filter))
                        .unwrap();
                    assert_hits(&hits);
                    let phases = searcher
                        .search_iter_with_filter(query, 2, Some(filter))
                        .collect::<Vec<_>>();
                    assert_eq!(phases.len(), if query.fast().is_some() { 2 } else { 1 });
                    for phase in phases {
                        match phase {
                            SearchPhase::Initial { results, .. }
                            | SearchPhase::Refined { results, .. } => assert_hits(&results),
                            other => panic!("unexpected metadata-filter phase: {other:?}"),
                        }
                    }
                }
            }
            let excluded = ExcludeDocIdsFilter::new(Some(&types), ["allowed-a"]);
            for query in [&progressive, &quality_only] {
                let (hits, _) = searcher
                    .search_collect_with_filter(query, 2, Some(&excluded))
                    .unwrap();
                assert_eq!(
                    hits.iter()
                        .map(|hit| hit.doc_id.as_str())
                        .collect::<Vec<_>>(),
                    ["allowed-b"],
                );
            }
            let either = FilterChain::new(FilterMode::Any)
                .with(Box::new(types))
                .with(Box::new(IncludeDocIdsFilter::new(["blocked-00"])));
            let (hits, _) = searcher
                .search_collect_with_filter(&progressive, 2, Some(&either))
                .unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().any(|hit| hit.doc_id == "blocked-00"));
            assert!(
                hits.iter()
                    .all(|hit| either.matches(&hit.doc_id, hit.metadata.as_deref()))
            );
        });
    }

    #[cfg(feature = "quill")]
    #[test]
    fn quill_metadata_filter_pins_unseen_quality_candidates_across_refresh() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_core::filter::DocTypeFilter;
        use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchIndex};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let vector_root = directory.path().join("vectors");
            let (fast, quality) = persisted_producer_pair(&vector_root);
            let lexical_root = directory.path().join("lexical");
            let config = QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let writer = QuillIndex::create(&cx, &lexical_root, config.clone())
                .await
                .unwrap();
            writer
                .index_documents(
                    &cx,
                    &[
                        IndexableDocument::new("a", "unrelated").with_metadata("doc_type", "keep"),
                        IndexableDocument::new("b", "unrelated").with_metadata("doc_type", "keep"),
                        IndexableDocument::new("c", "unrelated")
                            .with_metadata("doc_type", "blocked"),
                    ],
                )
                .await
                .unwrap();
            writer.commit(&cx).await.unwrap();
            let reader = Arc::new(
                QuillSearchIndex::open(&cx, &lexical_root, config)
                    .await
                    .unwrap(),
            );
            let searcher = SyncTwoTierSearcher::new(
                Arc::new(InMemoryTwoTierIndex::from_dir(&vector_root).unwrap()),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_nqc_dense_downweight_disabled()
            .with_lexical(Arc::new(QuillSyncLexicalSearch::new(
                Arc::clone(&reader),
                cx.clone(),
                "needle",
            )));
            let query = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
            );
            let filter = DocTypeFilter::new(["keep"]);
            let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
            let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                panic!("expected Initial");
            };
            assert_eq!(results[0].doc_id, "a");
            writer
                .upsert_documents(
                    &cx,
                    &[
                        IndexableDocument::new("b", "unrelated")
                            .with_metadata("doc_type", "blocked"),
                        IndexableDocument::new("c", "unrelated").with_metadata("doc_type", "keep"),
                    ],
                )
                .await
                .unwrap();
            writer.commit(&cx).await.unwrap();
            assert!(reader.refresh(&cx).await.unwrap());
            let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
                panic!("expected Refined");
            };
            assert_eq!(
                results[0].doc_id, "b",
                "quality must use Initial's pinned metadata"
            );
            assert_eq!(results[0].metadata.as_deref().unwrap()["doc_type"], "keep");
            assert!(phases.next().is_none());
            let (fresh, _) = searcher
                .search_collect_with_filter(&query, 1, Some(&filter))
                .unwrap();
            assert_eq!(
                fresh[0].doc_id, "c",
                "a new search must pin the refreshed publication"
            );
        });
    }

    #[test]
    fn quality_metadata_cancellation_is_terminal_in_collect_and_stream() {
        use frankensearch_core::filter::DocTypeFilter;
        use frankensearch_index::TwoTierIndex;

        struct CancellingQualityMetadata {
            cancellations: AtomicUsize,
        }
        impl SyncLexicalSearch for CancellingQualityMetadata {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                Ok(Vec::new())
            }

            fn metadata_for_documents(
                &self,
                ids: &[&str],
            ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
                if ids.contains(&"quality-only") {
                    self.cancellations.fetch_add(1, Ordering::SeqCst);
                    return Err(SearchError::Cancelled {
                        phase: "quality-metadata".to_owned(),
                        reason: "request cancelled during quality hydration".to_owned(),
                    });
                }
                Ok(vec![
                    Some(Arc::new(serde_json::json!({"doc_type": "keep"})));
                    ids.len()
                ])
            }
        }

        let directory = temp_dir("quality-metadata-cancellation");
        let fast = semantic_identity("persisted-fast", 2);
        let quality = semantic_identity("persisted-quality", 3);
        let mut writer = TwoTierIndex::create(directory.path(), TwoTierConfig::default()).unwrap();
        writer.set_fast_embedder_id("persisted-fast");
        writer.set_quality_embedder_id("persisted-quality");
        writer.set_fast_identity(&fast).unwrap();
        writer.set_quality_identity(&quality).unwrap();
        writer.add_fast_record("a", &[1.0, 0.0]).unwrap();
        writer.add_fast_record("c", &[0.8, 0.6]).unwrap();
        writer.add_quality_record("a", &[0.0, 0.0, 1.0]).unwrap();
        writer
            .add_quality_record("quality-only", &[1.0, 0.0, 0.0])
            .unwrap();
        writer.add_quality_record("c", &[0.8, 0.6, 0.0]).unwrap();
        drop(writer.finish().unwrap());

        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let fast_query = BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap();
        let query = TieredQueryEmbeddings::progressive(
            fast_query.clone(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let filter = DocTypeFilter::new(["keep"]);
        for (enabled, parallel_enabled) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let lexical = Arc::new(CancellingQualityMetadata {
                cancellations: AtomicUsize::new(0),
            });
            let searcher = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_lexical(lexical.clone())
            .with_search_params(SearchParams {
                parallel_enabled,
                parallel_threshold: 0,
                parallel_chunk_size: 1,
            })
            .with_mmr_config(MmrConfig {
                enabled,
                candidate_pool: 2,
                ..sync_mmr_config()
            });
            assert!(matches!(
                searcher.search_collect_with_filter(&query, 1, Some(&filter)),
                Err(SearchError::Cancelled { ref phase, .. }) if phase == "quality-metadata"
            ));
            assert_eq!(lexical.cancellations.load(Ordering::SeqCst), 1);

            let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
            let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                panic!("fast results must precede quality metadata cancellation");
            };
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "a");
            let Some(SearchPhase::RefinementFailed {
                initial_results,
                error: SearchError::Cancelled { phase, .. },
                ..
            }) = phases.next()
            else {
                panic!("quality cancellation must remain a terminal typed failure");
            };
            assert_eq!(phase, "quality-metadata");
            assert_eq!(
                serde_json::to_value(initial_results).unwrap(),
                serde_json::to_value(results).unwrap(),
                "the failure preserves the exact previously published ranking"
            );
            assert!(
                phases.next().is_none(),
                "cancelled searches must not run MMR"
            );
            assert!(phases.next().is_none(), "terminal iteration cannot retry");
            assert_eq!(lexical.cancellations.load(Ordering::SeqCst), 2);

            let (fast_results, _) = searcher
                .search_collect_with_filter(
                    &TieredQueryEmbeddings::fast_only(fast_query.clone()),
                    1,
                    Some(&filter),
                )
                .expect("a separate fast-only request never reaches quality metadata");
            assert!(!fast_results.is_empty());
            assert_eq!(lexical.cancellations.load(Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn metadata_filter_hydration_errors_never_become_unknown_observations() {
        use frankensearch_core::filter::DocTypeFilter;

        struct FailingMetadata {
            malformed: bool,
        }
        impl SyncLexicalSearch for FailingMetadata {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                Ok(Vec::new())
            }

            fn metadata_for_documents(
                &self,
                ids: &[&str],
            ) -> SearchResult<Vec<Option<Arc<serde_json::Value>>>> {
                if ids.contains(&"b") {
                    if self.malformed {
                        return Ok(Vec::new());
                    }
                    return Err(SearchError::InvalidConfig {
                        field: "fixture.metadata".to_owned(),
                        value: "unreadable".to_owned(),
                        reason: "metadata decoding failed".to_owned(),
                    });
                }
                Ok(vec![
                    Some(Arc::new(serde_json::json!({"doc_type": "keep"})));
                    ids.len()
                ])
            }
        }

        let directory = temp_dir("metadata-filter-errors");
        let (fast, quality) = persisted_producer_pair(directory.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality.clone()).unwrap(),
        );
        let quality_only = TieredQueryEmbeddings::quality_only(
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let filter = DocTypeFilter::new(["keep"]);
        for malformed in [false, true] {
            let searcher = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    ..TwoTierConfig::default()
                },
            )
            .with_lexical(Arc::new(FailingMetadata { malformed }));
            let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
            assert!(matches!(phases.next(), Some(SearchPhase::Initial { .. })));
            let Some(SearchPhase::RefinementFailed {
                initial_results,
                error,
                ..
            }) = phases.next()
            else {
                panic!("a failed quality metadata observation must preserve Initial");
            };
            assert_eq!(initial_results.len(), 1);
            assert_eq!(initial_results[0].doc_id, "a");
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
            assert!(phases.next().is_none());
            let (fallback, metrics) = searcher
                .search_collect_with_filter(&query, 1, Some(&filter))
                .expect("ordinary refinement failures preserve the verified initial ranking");
            assert_eq!(fallback.len(), 1);
            assert_eq!(fallback[0].doc_id, "a");
            assert!(metrics.skip_reason.is_some());
            assert!(matches!(
                searcher.search_collect_with_filter(&quality_only, 1, Some(&filter)),
                Err(SearchError::InvalidConfig { .. }),
            ));
            let fast =
                InMemoryVectorIndex::from_vectors(vec!["b".to_owned()], vec![vec![1.0, 0.0]], 2)
                    .unwrap();
            let initial = SyncTwoTierSearcher::new(
                Arc::new(InMemoryTwoTierIndex::new(fast, None).unwrap()),
                TwoTierConfig::default(),
            )
            .with_lexical(Arc::new(FailingMetadata { malformed }));
            assert!(matches!(
                initial.search_collect_with_filter(
                    &fast_only_query(vec![1.0, 0.0]),
                    1,
                    Some(&filter),
                ),
                Err(SearchError::InvalidConfig { .. }),
            ));
        }
    }

    #[cfg(feature = "quill")]
    #[test]
    fn quill_filters_retrieve_lexical_only_tail_in_both_phases() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_core::filter::BitsetFilter;
        use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchIndex};

        struct SourcePathFilter;
        impl SearchFilter for SourcePathFilter {
            fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
                metadata
                    .and_then(|value| value.get("path"))
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|path| path.starts_with("src/"))
            }
            fn name(&self) -> &'static str {
                "source-path"
            }
        }

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let config = QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let writer = QuillIndex::create(&cx, directory.path(), config.clone())
                .await
                .unwrap();
            let mut docs: Vec<_> = (0..64)
                .map(|rank| {
                    IndexableDocument::new(format!("excluded-{rank:02}"), "needle")
                        .with_title("needle")
                        .with_metadata("path", format!("vendor/{rank}.rs"))
                })
                .collect();
            for id in ["tail-a", "tail-b"] {
                docs.push(
                    IndexableDocument::new(id, "needle with some extra surrounding context")
                        .with_metadata("path", format!("src/{id}.rs")),
                );
            }
            writer.index_documents(&cx, &docs).await.unwrap();
            writer.commit(&cx).await.unwrap();
            drop(writer);
            let reader = Arc::new(
                QuillSearchIndex::open(&cx, directory.path(), config)
                    .await
                    .unwrap(),
            );
            let lexical = Arc::new(QuillSyncLexicalSearch::new(reader, cx, "needle"));
            assert!(
                lexical
                    .search_sync(&[], 64)
                    .unwrap()
                    .iter()
                    .all(|hit| hit.doc_id.starts_with("excluded-")),
                "the real BM25 prefix must exclude both eligible tail documents"
            );
            let allowed = BitsetFilter::from_doc_ids(["tail-a", "tail-b"]);
            let query = TieredQueryEmbeddings::progressive(
                bound("sync-fixture", vec![0.0, 0.0]),
                bound("sync-fixture", vec![1.0, 0.0]),
            );
            for fast_only in [false, true] {
                for filter in [&allowed as &dyn SearchFilter, &SourcePathFilter] {
                    let searcher = SyncTwoTierSearcher::new(
                        make_index(),
                        TwoTierConfig {
                            fast_only,
                            explain: true,
                            ..TwoTierConfig::default()
                        },
                    )
                    .with_lexical(lexical.clone());
                    let (results, metrics) = searcher
                        .search_collect_with_filter(&query, 2, Some(filter))
                        .unwrap();
                    assert_eq!(results.len(), 2);
                    assert!(results.iter().all(|hit| hit.doc_id.starts_with("tail-")));
                    assert!(results.iter().all(|hit| hit.source == ScoreSource::Lexical
                        && hit.fast_score.is_none()
                        && hit.quality_score.is_none()));
                    for hit in &results {
                        assert_eq!(
                            hit.metadata.as_ref().unwrap()["path"],
                            format!("src/{}.rs", hit.doc_id),
                        );
                    }
                    assert_eq!(metrics.lexical_candidates, 2);
                    assert_eq!(metrics.phase1_vectors_searched, 0);
                    let phases = searcher
                        .search_iter_with_filter(&query, 2, Some(filter))
                        .collect::<Vec<_>>();
                    assert_eq!(phases.len(), if fast_only { 1 } else { 2 });
                    for phase in &phases {
                        let hits = match phase {
                            SearchPhase::Initial { results, .. }
                            | SearchPhase::Refined { results, .. } => results,
                            other => panic!("unexpected phase: {other:?}"),
                        };
                        assert_eq!(hits.len(), 2);
                        assert!(hits.iter().all(|hit| hit.doc_id.starts_with("tail-")));
                        for hit in hits {
                            assert_eq!(
                                hit.metadata.as_ref().unwrap()["path"],
                                format!("src/{}.rs", hit.doc_id),
                            );
                            let [component] =
                                hit.explanation.as_ref().unwrap().components.as_slice()
                            else {
                                panic!("a lexical-only hit should have one BM25 component");
                            };
                            assert!(matches!(
                                component.source,
                                ExplainedSource::LexicalBm25 { .. }
                            ));
                            assert_eq!(
                                component.raw_score.to_bits(),
                                f64::from(hit.lexical_score.unwrap()).to_bits()
                            );
                            assert!(
                                (component.rrf_contribution - f64::from(hit.score)).abs() < 1e-8
                            );
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn zero_fast_query_still_retrieves_from_an_attested_quality_tier() {
        let dir = temp_dir("zero-fast-quality");
        let (fast, fast_identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "zero-fast",
            31,
            &[("fast-only", &[1.0, 0.0])],
        );
        let (quality, quality_identity) = attested_tier(
            dir.path(),
            "vector.quality.idx",
            "independent-quality",
            31,
            &[
                ("quality-other", &[1.0, 0.0]),
                ("quality-only", &[0.0, 1.0]),
            ],
        );
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(
                InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"),
            ),
            TwoTierConfig::default(),
        );
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![0.0, 0.0], fast_identity).expect("bind zero fast"),
            BoundQueryEmbedding::new(vec![0.0, 1.0], quality_identity).expect("bind quality"),
        );
        let (results, metrics) = searcher
            .search_collect(&query, 1)
            .expect("the independent quality query can answer without fast candidates");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "quality-only");
        assert_eq!(results[0].source, ScoreSource::SemanticQuality);
        assert_eq!(results[0].index, None);
        assert_eq!(results[0].fast_score, None);
        assert_eq!(results[0].quality_score, Some(1.0));
        assert_eq!(
            metrics.zero_signal, None,
            "quality hits make the combined semantic lane nonempty"
        );
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert_eq!(metrics.phase2_vectors_searched, 2);
        assert_eq!(metrics.skip_reason, None);

        let phases = searcher.search_iter(&query, 1).collect::<Vec<_>>();
        let [
            SearchPhase::Initial {
                results: initial,
                metrics: initial_metrics,
                ..
            },
            SearchPhase::Refined {
                results: refined, ..
            },
        ] = phases.as_slice()
        else {
            panic!("independent quality retrieval must produce Initial then Refined");
        };
        assert!(initial.is_empty());
        assert_eq!(initial_metrics.vectors_searched, 0);
        assert_eq!(
            serde_json::to_value(refined).unwrap(),
            serde_json::to_value(&results).unwrap()
        );
    }

    #[test]
    fn zero_quality_query_preserves_fast_and_lexical_results_on_attested_tiers() {
        let dir = temp_dir("zero-quality");
        let (fast, fast_identity) = attested_tier(
            dir.path(),
            "vector.fast.idx",
            "zero-quality-fast",
            32,
            &[("fast-only", &[1.0, 0.0])],
        );
        let (quality, quality_identity) = attested_tier(
            dir.path(),
            "vector.quality.idx",
            "zero-quality-independent",
            32,
            &[("quality-only", &[0.0, 1.0])],
        );
        let index = Arc::new(
            InMemoryTwoTierIndex::new(fast, Some(quality)).expect("compatible fixture tiers"),
        );

        // Both-zero queries must stay empty or lexical-only. A useful fast
        // query must retain its results when only the quality vector is zero.
        for fast_signal in [false, true] {
            let query = TieredQueryEmbeddings::progressive(
                BoundQueryEmbedding::new(
                    vec![if fast_signal { 1.0 } else { 0.0 }, 0.0],
                    fast_identity.clone(),
                )
                .expect("bind fast"),
                BoundQueryEmbedding::new(vec![0.0, 0.0], quality_identity.clone())
                    .expect("bind zero quality"),
            );
            for has_lexical in [false, true] {
                let mut searcher =
                    SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default());
                if has_lexical {
                    searcher = searcher.with_lexical(Arc::new(StaticLexical {
                        hits: vec![lexical_result("lex-only", 10.0)],
                    }));
                }
                let (results, metrics) = searcher.search_collect(&query, 2).unwrap();
                let mut ids: Vec<&str> = results.iter().map(|hit| hit.doc_id.as_str()).collect();
                ids.sort_unstable();
                let mut expected = Vec::new();
                if fast_signal {
                    expected.push("fast-only");
                }
                if has_lexical {
                    expected.push("lex-only");
                }
                assert_eq!(
                    ids, expected,
                    "zero quality must never retrieve quality-only"
                );
                for result in &results {
                    assert_eq!(
                        result.source,
                        if result.doc_id == "fast-only" {
                            ScoreSource::SemanticFast
                        } else {
                            ScoreSource::Lexical
                        }
                    );
                    assert_eq!(result.quality_score, None);
                }
                assert_eq!(
                    metrics.zero_signal,
                    (!fast_signal).then_some(ZeroSignalReason::ZeroNormQuery)
                );
                assert_eq!(metrics.phase1_vectors_searched, usize::from(fast_signal));
                assert_eq!(metrics.phase2_vectors_searched, 0);
                assert_eq!(metrics.lexical_candidates, usize::from(has_lexical));
                assert_eq!(
                    metrics.skip_reason.as_deref(),
                    Some("quality_query_zero_norm")
                );

                let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
                let [
                    SearchPhase::Initial {
                        results: initial, ..
                    },
                ] = phases.as_slice()
                else {
                    panic!("a zero quality query must not publish a Refined phase");
                };
                assert_eq!(
                    serde_json::to_value(initial).unwrap(),
                    serde_json::to_value(&results).unwrap()
                );
            }
        }
    }

    #[test]
    fn zero_quality_query_does_not_rescore_legacy_fast_hits() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let query = TieredQueryEmbeddings::progressive(
            bound("sync-fixture", vec![1.0, 0.0]),
            bound("sync-fixture", vec![0.0, 0.0]),
        );
        let (results, metrics) = searcher.search_collect(&query, 2).unwrap();
        assert_eq!(
            results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert!(
            results.iter().all(|hit| {
                hit.source == ScoreSource::SemanticFast && hit.quality_score.is_none()
            })
        );
        assert_eq!(metrics.zero_signal, None);
        assert_eq!(metrics.phase2_vectors_searched, 0);
        assert_eq!(
            metrics.skip_reason.as_deref(),
            Some("quality_query_zero_norm")
        );
        let phases = searcher.search_iter(&query, 2).collect::<Vec<_>>();
        let [
            SearchPhase::Initial {
                results: initial, ..
            },
        ] = phases.as_slice()
        else {
            panic!("legacy rescoring must also skip a zero quality query");
        };
        assert_eq!(
            serde_json::to_value(initial).unwrap(),
            serde_json::to_value(&results).unwrap()
        );
    }

    #[test]
    fn query_shape_is_admitted_before_zero_shortcuts_and_lexical_work() {
        struct UnreachableLexical;
        impl SyncLexicalSearch for UnreachableLexical {
            fn search_sync(
                &self,
                _query: &[f32],
                _limit: usize,
            ) -> SearchResult<Vec<ScoredResult>> {
                panic!("invalid query shape must be refused before lexical work");
            }
        }
        for declared in [false, true] {
            let index = if declared {
                let identity = EmbeddingIdentityBundleV1::explicit_test_model("sync-fixture", 2);
                let tier = || {
                    InMemoryVectorIndex::from_vectors_with_identity(
                        vec!["a".to_owned()],
                        vec![vec![1.0, 0.0]],
                        2,
                        &identity.space,
                    )
                    .unwrap()
                };
                Arc::new(InMemoryTwoTierIndex::new(tier(), Some(tier())).unwrap())
            } else {
                make_index()
            };
            let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
                .with_lexical(Arc::new(UnreachableLexical));
            for (tier, fast, quality) in [
                ("fast", vec![0.0; 3], vec![1.0, 0.0]),
                ("fast", vec![1.0, 0.0, 0.0], vec![1.0, 0.0]),
                ("quality", vec![1.0, 0.0], vec![0.0; 3]),
                ("quality", vec![1.0, 0.0], vec![1.0, 0.0, 0.0]),
            ] {
                let query = TieredQueryEmbeddings::progressive(
                    bound("sync-fixture", fast),
                    bound("sync-fixture", quality),
                );
                for k in [0, 2] {
                    if declared {
                        // Width is part of the declared space. Its identity
                        // error must win over the common dimension check.
                        assert_admission_refusal(
                            &searcher,
                            &query,
                            k,
                            &format!("query_embedding.{tier}.space_identity"),
                        );
                        continue;
                    }
                    let error = searcher
                        .search_collect(&query, k)
                        .expect_err("wrong-width query");
                    assert!(matches!(
                        error,
                        SearchError::DimensionMismatch {
                            expected: 2,
                            found: 3
                        }
                    ));
                    let phases = searcher.search_iter(&query, k).collect::<Vec<_>>();
                    let [
                        SearchPhase::RefinementFailed {
                            initial_results,
                            error:
                                SearchError::DimensionMismatch {
                                    expected: 2,
                                    found: 3,
                                },
                            ..
                        },
                    ] = phases.as_slice()
                    else {
                        panic!(
                            "zero and nonzero wrong-width queries must have the same shape error"
                        );
                    };
                    assert!(initial_results.is_empty());
                }
            }
        }
    }

    #[test]
    fn zero_k_with_zero_fast_query_does_not_call_lexical_or_refine() {
        struct UnreachableLexical;
        impl SyncLexicalSearch for UnreachableLexical {
            fn search_sync(
                &self,
                _query: &[f32],
                _limit: usize,
            ) -> SearchResult<Vec<ScoredResult>> {
                panic!("a zero-k search must not call the lexical backend");
            }
        }
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(Arc::new(UnreachableLexical));
        let query = tiered(vec![0.0, 0.0]);
        let (results, metrics) = searcher.search_collect(&query, 0).unwrap();
        assert!(results.is_empty());
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::CallerRequestedZeroK)
        );
        assert_eq!(metrics.phase1_vectors_searched, 0);
        assert_eq!(metrics.phase2_vectors_searched, 0);
        assert_eq!(searcher.search_iter(&query, 0).count(), 0);
    }

    #[test]
    fn rrf_weights_flow_through_searcher_to_fusion() {
        // Lexical favors "c" (then "b"); the quality/semantic tier favors a different doc
        // for query [1,0]. Extreme opposite tier weights must therefore flip the top result,
        // proving `with_rrf_weights` / `with_rrf_tiebreak` reach the fusion `RrfConfig`.
        let make_lex = || {
            Arc::new(StaticLexical {
                hits: vec![lexical_result("c", 10.0), lexical_result("b", 5.0)],
            })
        };
        let q = tiered(vec![1.0_f32, 0.0]);

        let sem_heavy = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(0.01, 100.0)
            .with_rrf_tiebreak(crate::rrf::RrfTiebreak::Hash);
        let lex_heavy = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(100.0, 0.01);

        let (sem_res, _) = sem_heavy.search_collect(&q, 3).unwrap();
        let (lex_res, _) = lex_heavy.search_collect(&q, 3).unwrap();
        assert!(!sem_res.is_empty() && !lex_res.is_empty());
        assert_ne!(
            sem_res[0].doc_id, lex_res[0].doc_id,
            "opposite tier weights must change the fused top result (weights reach fusion)"
        );
    }

    #[test]
    fn nqc_dense_downweight_empty_sketch_is_bit_identical_to_base_weight() {
        let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_rrf_weights(1.0, 1.3)
            .with_nqc_dense_downweight(0.5, 0.1, NqcDenseWeight::new());

        assert_eq!(
            searcher.effective_semantic_weight(&hits).to_bits(),
            searcher.rrf_semantic_weight.to_bits(),
        );
    }

    #[test]
    fn nqc_dense_downweight_flows_through_searcher_to_fusion() {
        // Both searchers up-weight the dense tier (semantic 5×) so it dominates by default;
        // enabling the NQC dense down-weight with a sample below the query's NQC drives the
        // dense weight to 0, so lexical (favoring "c") dominates instead. Different top =>
        // the opt-in down-weight reaches the fusion RrfConfig.
        let make_lex = || {
            Arc::new(StaticLexical {
                hits: vec![lexical_result("c", 10.0), lexical_result("b", 5.0)],
            })
        };
        let q = tiered(vec![1.0_f32, 0.0]);

        let neutral = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(1.0, 5.0);
        // Query NQC (cv of lexical scores [10, 5] ≈ 0.333) is above every sampled value, so
        // its percentile is 1.0 and dense_weight(beta=1, w_min=0.05) = clip(1 - 1·1, 0.05, 1)
        // = 0.05 → effective semantic weight 5·0.05 = 0.25 (< lexical 1.0), still > 0 so it is
        // not neutralized by the tier-weight sanitizer.
        let sample = NqcDenseWeight::from_sample(&[0.1, 0.2, 0.3]);
        let downweighted = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_lexical(make_lex())
            .with_rrf_weights(1.0, 5.0)
            .with_nqc_dense_downweight(1.0, 0.05, sample);

        let (neutral_res, _) = neutral.search_collect(&q, 3).unwrap();
        let (down_res, _) = downweighted.search_collect(&q, 3).unwrap();
        assert!(!neutral_res.is_empty() && !down_res.is_empty());
        assert_eq!(
            down_res[0].doc_id, "c",
            "zeroing the dense tier lets lexical dominate"
        );
        assert_ne!(
            neutral_res[0].doc_id, down_res[0].doc_id,
            "the NQC dense down-weight must change the fused top (it reaches fusion)"
        );
    }

    #[test]
    fn nqc_adaptive_cold_start_is_bit_identical_to_base_weight() {
        // A freshly-enabled adaptive down-weight has an empty online sketch, so a cold-start
        // query's effective semantic weight is bit-identical to the base (safe to default-on).
        let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_rrf_weights(1.0, 1.3)
            .with_nqc_dense_downweight_adaptive(0.5, 0.1, 128, 8, 4);
        assert_eq!(
            searcher.effective_semantic_weight(&hits).to_bits(),
            searcher.rrf_semantic_weight.to_bits(),
            "cold-start (empty online sketch) leaves fusion byte-identical"
        );
    }

    #[test]
    fn nqc_adaptive_warms_up_then_down_weights() {
        // Warm the online sketch with observed queries, then a high-NQC query lands at a high
        // percentile and is down-weighted below the base semantic weight (reaches fusion online).
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_rrf_weights(1.0, 1.0)
            .with_nqc_dense_downweight_adaptive(0.5, 0.05, 128, 8, 4);
        let warm = [
            lexical_result("a", 5.0),
            lexical_result("b", 5.0),
            lexical_result("c", 5.0),
        ];
        for _ in 0..12 {
            let _ = searcher.effective_semantic_weight(&warm);
        }
        let peaked = [
            lexical_result("a", 100.0),
            lexical_result("b", 1.0),
            lexical_result("c", 0.5),
        ];
        let weight = searcher.effective_semantic_weight(&peaked);
        assert!(
            weight < searcher.rrf_semantic_weight,
            "a high-NQC query is down-weighted after the online sketch warms up, got {weight}"
        );
    }

    #[test]
    fn nqc_adaptive_defaults_enable_neutral_cold_start_and_disabled_clears_it() {
        let hits = [lexical_result("a", 10.0), lexical_result("b", 1.0)];
        // Blessed defaults: enabled, but cold-start (empty sketch) is byte-identical to base.
        let defaults = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_rrf_weights(1.0, 1.3)
            .with_nqc_dense_downweight_adaptive_defaults();
        assert_eq!(
            defaults.effective_semantic_weight(&hits).to_bits(),
            defaults.rrf_semantic_weight.to_bits(),
            "production-default cold start is neutral"
        );
        // Disabling clears both the adaptive and static paths -> byte-identical, no observation.
        let disabled = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
            .with_rrf_weights(1.0, 1.3)
            .with_nqc_dense_downweight_adaptive_defaults()
            .with_nqc_dense_downweight_disabled();
        assert_eq!(
            disabled.effective_semantic_weight(&hits).to_bits(),
            disabled.rrf_semantic_weight.to_bits(),
            "disabled escape hatch is byte-identical"
        );
    }

    #[test]
    fn search_iter_yields_initial_then_refined() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let phases = searcher
            .search_iter(&tiered(vec![1.0, 0.0]), 2)
            .collect::<Vec<_>>();
        assert_eq!(phases.len(), 2);
        assert!(matches!(phases[0], SearchPhase::Initial { .. }));
        assert!(matches!(phases[1], SearchPhase::Refined { .. }));
    }

    #[test]
    fn search_iter_defers_quality_until_next_and_drop_stops_pending_work() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ObservedFilter {
            total: AtomicUsize,
            quality_only: AtomicUsize,
        }
        impl SearchFilter for ObservedFilter {
            fn matches(&self, doc_id: &str, _: Option<&serde_json::Value>) -> bool {
                self.total.fetch_add(1, Ordering::SeqCst);
                if doc_id == "b" || doc_id == "c" {
                    self.quality_only.fetch_add(1, Ordering::SeqCst);
                }
                true
            }

            fn name(&self) -> &'static str {
                "observe-tier-traversal"
            }
        }
        struct ObservedLexical(Arc<AtomicUsize>);
        impl SyncLexicalSearch for ObservedLexical {
            fn search_sync(&self, _: &[f32], _: usize) -> SearchResult<Vec<ScoredResult>> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            }
        }

        // These persisted, independently identified tiers contain quality-only
        // documents. Observing their filter calls measures real quality scans,
        // not an iterator's own state or a fabricated completion counter.
        let dir = temp_dir("lazy-sync-quality");
        let (fast, quality) = persisted_producer_pair(dir.path());
        let lexical_calls = Arc::new(AtomicUsize::new(0));
        let searcher = SyncTwoTierSearcher::new(
            Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap()),
            TwoTierConfig {
                candidate_multiplier: 1,
                quality_weight: 1.0,
                ..TwoTierConfig::default()
            },
        )
        .with_lexical(Arc::new(ObservedLexical(Arc::clone(&lexical_calls))));
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let observed = ObservedFilter {
            total: AtomicUsize::new(0),
            quality_only: AtomicUsize::new(0),
        };

        let unused = searcher.search_iter_with_filter(&query, 1, Some(&observed));
        assert_eq!(observed.total.load(Ordering::SeqCst), 0);
        assert_eq!(lexical_calls.load(Ordering::SeqCst), 0);
        assert_eq!(unused.size_hint(), (1, Some(2)));
        drop(unused);
        assert_eq!(observed.total.load(Ordering::SeqCst), 0);
        assert_eq!(lexical_calls.load(Ordering::SeqCst), 0);

        let mut stopped = searcher.search_iter_with_filter(&query, 1, Some(&observed));
        let Some(SearchPhase::Initial { results, .. }) = stopped.next() else {
            panic!("the first call must expose Initial");
        };
        assert_eq!(results[0].doc_id, "a");
        assert!(observed.total.load(Ordering::SeqCst) > 0);
        assert_eq!(observed.quality_only.load(Ordering::SeqCst), 0);
        assert_eq!(lexical_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stopped.size_hint(), (1, Some(1)));
        let before_drop = observed.total.load(Ordering::SeqCst);
        drop(stopped);
        assert_eq!(observed.total.load(Ordering::SeqCst), before_drop);
        assert_eq!(observed.quality_only.load(Ordering::SeqCst), 0);
        assert_eq!(lexical_calls.load(Ordering::SeqCst), 1);

        let mut complete = searcher.search_iter_with_filter(&query, 1, Some(&observed));
        assert!(matches!(complete.next(), Some(SearchPhase::Initial { .. })));
        assert_eq!(observed.quality_only.load(Ordering::SeqCst), 0);
        let Some(SearchPhase::Refined { results, .. }) = complete.next() else {
            panic!("the second call must execute independent quality retrieval");
        };
        assert_eq!(results[0].doc_id, "b");
        assert!(observed.quality_only.load(Ordering::SeqCst) > 0);
        assert_eq!(lexical_calls.load(Ordering::SeqCst), 2);
        let after_refinement = observed.total.load(Ordering::SeqCst);
        for _ in 0..3 {
            assert!(complete.next().is_none());
            assert_eq!(complete.size_hint(), (0, Some(0)));
        }
        assert_eq!(observed.total.load(Ordering::SeqCst), after_refinement);
    }

    #[test]
    fn search_iter_lazy_refinement_preserves_collect_results_and_evidence() {
        let dir = temp_dir("lazy-sync-parity");
        let (fast, quality) = persisted_producer_pair(dir.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(dir.path()).unwrap());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        for hybrid in [false, true] {
            let mut searcher = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    candidate_multiplier: 1,
                    quality_weight: 1.0,
                    explain: true,
                    ..TwoTierConfig::default()
                },
            )
            .with_nqc_dense_downweight_disabled();
            if hybrid {
                searcher = searcher.with_lexical(Arc::new(StaticLexical {
                    hits: vec![lexical_result("a", 10.0), lexical_result("lex-only", 9.0)],
                }));
            }
            for filter in [None, Some(&ExcludeB as &dyn SearchFilter)] {
                let (collected, total) = searcher
                    .search_collect_with_filter(&query, 2, filter)
                    .unwrap();
                let mut phases = searcher.search_iter_with_filter(&query, 2, filter);
                let Some(SearchPhase::Initial {
                    metrics: initial_metrics,
                    ..
                }) = phases.next()
                else {
                    panic!("Initial must precede quality work");
                };
                let Some(SearchPhase::Refined {
                    results,
                    metrics,
                    rank_changes,
                    ..
                }) = phases.next()
                else {
                    panic!("a complete query must refine");
                };
                // Includes exact score serialization, ordering, per-tier score
                // provenance, metadata and the optional explanation tree.
                assert_eq!(
                    serde_json::to_value(&results).unwrap(),
                    serde_json::to_value(&collected).unwrap(),
                );
                assert_eq!(
                    initial_metrics.vectors_searched,
                    total.phase1_vectors_searched
                );
                assert_eq!(metrics.vectors_searched, total.phase2_vectors_searched);
                assert_eq!(metrics.lexical_candidates, total.lexical_candidates);
                assert_eq!(
                    metrics.hash_control_candidates,
                    total.hash_control_candidates
                );
                assert_eq!(
                    serde_json::to_value(rank_changes).unwrap(),
                    serde_json::to_value(total.rank_changes).unwrap(),
                );
                assert!(phases.next().is_none());
            }
        }
    }

    #[test]
    fn search_iter_refinement_failure_retains_initial_and_is_terminal() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let query = tiered(vec![1.0, 0.0]);
        let bad_quality = bound("sync-fixture", vec![1.0, 0.0, 0.0]);
        let mut phases = searcher.search_iter(&query, 2);
        let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
            panic!("Initial must be observable before refinement");
        };

        // Public admission rejects this shape before Initial. Inject the
        // internal fault only after admission to exercise the real quality
        // kernel's defensive error and the lazy iterator's terminal handling.
        let SyncIteratorState::Refine(pending) = &mut phases.state else {
            panic!("quality work must still be pending");
        };
        pending.refinement.quality_query = &bad_quality;
        let Some(SearchPhase::RefinementFailed {
            initial_results,
            error,
            ..
        }) = phases.next()
        else {
            panic!("a quality error must preserve the already-delivered Initial");
        };
        assert!(matches!(
            error,
            SearchError::DimensionMismatch {
                expected: 2,
                found: 3
            }
        ));
        assert_eq!(
            serde_json::to_value(initial_results).unwrap(),
            serde_json::to_value(results).unwrap(),
        );
        assert_eq!(phases.size_hint(), (0, Some(0)));
        assert!(phases.next().is_none());
        assert!(phases.next().is_none());
    }

    #[test]
    fn fast_only_mode_skips_phase_two() {
        let config = TwoTierConfig {
            fast_only: true,
            ..TwoTierConfig::default()
        };
        let searcher = SyncTwoTierSearcher::new(make_index(), config);
        let phases = searcher
            .search_iter(&tiered(vec![1.0, 0.0]), 2)
            .collect::<Vec<_>>();
        assert_eq!(phases.len(), 1);
        assert!(matches!(phases[0], SearchPhase::Initial { .. }));
    }

    #[test]
    fn fast_only_policy_ignores_disabled_quality_identity() {
        for attested in [false, true] {
            let directory = temp_dir("fast-only-quality-admission");
            let (index, fast, quality) = sync_mmr_fixture(directory.path(), attested, true);
            let fast_only = SyncTwoTierSearcher::new(
                Arc::clone(&index),
                TwoTierConfig {
                    fast_only: true,
                    ..TwoTierConfig::default()
                },
            );
            let progressive =
                SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default());
            for change_space in [false, true] {
                let foreign = |identity: &EmbeddingIdentityBundleV1| {
                    let mut changed = identity.clone();
                    if change_space {
                        "new-weights-same-model".clone_into(&mut changed.space.immutable_revision);
                        changed.producer.space_fingerprint = changed.space.fingerprint();
                    } else {
                        "foreign-backend".clone_into(&mut changed.producer.backend);
                        changed.producer.golden_vectors.vectors_sha256 = "c".repeat(64);
                    }
                    changed.validate().expect("coherent foreign identity");
                    changed
                };
                let expected_field = |tier: &str| {
                    if !attested {
                        format!("search_activation.{tier}.producer_revision")
                    } else if change_space {
                        format!("query_embedding.{tier}.space_identity")
                    } else {
                        format!("query_embedding.{tier}.producer_conformance")
                    }
                };
                for zero in [false, true] {
                    for k in [0, 2] {
                        let vector = vec![if zero { 0.0 } else { 1.0 }, 0.0];
                        let query = TieredQueryEmbeddings::progressive(
                            BoundQueryEmbedding::new(vector.clone(), fast.clone()).unwrap(),
                            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], foreign(&quality))
                                .unwrap(),
                        );
                        let (results, metrics) = fast_only
                            .search_collect_with_filter(&query, k, Some(&ExcludeB))
                            .expect("disabled quality cannot block admitted fast retrieval");
                        assert_eq!(metrics.phase2_vectors_searched, 0);
                        assert!(results.iter().all(|hit| hit.quality_score.is_none()));
                        if k == 0 || zero {
                            assert!(results.is_empty());
                        } else {
                            assert_eq!(
                                results
                                    .iter()
                                    .map(|hit| hit.doc_id.as_str())
                                    .collect::<Vec<_>>(),
                                ["a", "c"]
                            );
                            assert_eq!(metrics.skip_reason.as_deref(), Some("fast_only"));
                            let phases = fast_only
                                .search_iter_with_filter(&query, k, Some(&ExcludeB))
                                .collect::<Vec<_>>();
                            let [
                                SearchPhase::Initial {
                                    results: streamed, ..
                                },
                            ] = phases.as_slice()
                            else {
                                panic!("fast-only must return one successful phase: {phases:?}");
                            };
                            assert_eq!(
                                serde_json::to_value(streamed).unwrap(),
                                serde_json::to_value(&results).unwrap()
                            );
                        }
                        assert_admission_refusal(
                            &progressive,
                            &query,
                            k,
                            &expected_field("quality"),
                        );
                        let wrong_fast = TieredQueryEmbeddings::progressive(
                            BoundQueryEmbedding::new(vector, foreign(&fast)).unwrap(),
                            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality.clone()).unwrap(),
                        );
                        assert_admission_refusal(
                            &fast_only,
                            &wrong_fast,
                            k,
                            &expected_field("fast"),
                        );
                    }
                }
            }
            let quality_only = TieredQueryEmbeddings::quality_only(
                BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
            );
            assert_admission_refusal(&fast_only, &quality_only, 1, "sync_search.topology");
        }
    }

    #[test]
    fn fast_only_policy_ignores_disabled_quality_shape_on_legacy_indexes() {
        let index = make_index();
        let fast_only = SyncTwoTierSearcher::new(
            Arc::clone(&index),
            TwoTierConfig {
                fast_only: true,
                ..TwoTierConfig::default()
            },
        );
        let progressive = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
        for k in [0, 2] {
            let query = TieredQueryEmbeddings::progressive(
                bound("sync-fixture", vec![1.0, 0.0]),
                bound("sync-fixture", vec![1.0, 0.0, 0.0]),
            );
            let (results, metrics) = fast_only
                .search_collect(&query, k)
                .expect("disabled quality shape is irrelevant to fast retrieval");
            assert_eq!(results.len(), k);
            assert_eq!(metrics.phase2_vectors_searched, 0);
            assert!(matches!(
                progressive.search_collect(&query, k),
                Err(SearchError::DimensionMismatch {
                    expected: 2,
                    found: 3
                })
            ));
            let wrong_fast = TieredQueryEmbeddings::progressive(
                bound("sync-fixture", vec![1.0, 0.0, 0.0]),
                bound("sync-fixture", vec![1.0, 0.0]),
            );
            assert!(matches!(
                fast_only.search_collect(&wrong_fast, k),
                Err(SearchError::DimensionMismatch {
                    expected: 2,
                    found: 3
                })
            ));
        }
    }

    #[test]
    fn sync_refinement_rejects_revoked_retained_fast_and_lexical_inputs() {
        use std::sync::atomic::AtomicBool;

        struct RevocableFilter {
            denied: &'static str,
            revoked: AtomicBool,
        }

        impl SearchFilter for RevocableFilter {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                id != self.denied || !self.revoked.load(Ordering::SeqCst)
            }

            fn name(&self) -> &'static str {
                "refinement-revocation"
            }
        }

        for (hybrid, denied) in [
            (false, "a"),
            (false, "b"),
            (true, "a"),
            (true, "b"),
            (true, "lex-only"),
        ] {
            let mut searcher = SyncTwoTierSearcher::new(
                make_index(),
                TwoTierConfig {
                    candidate_multiplier: 3,
                    quality_weight: 1.0,
                    ..TwoTierConfig::default()
                },
            );
            if hybrid {
                searcher = searcher.with_lexical(Arc::new(StaticLexical {
                    hits: vec![lexical_result("a", 10.0), lexical_result("lex-only", 9.0)],
                }));
            }
            let query = tiered(vec![1.0, 0.0]);
            let filter = RevocableFilter {
                denied,
                revoked: AtomicBool::new(false),
            };
            let mut phases = searcher.search_iter_with_filter(&query, 1, Some(&filter));
            let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                panic!("Initial must precede lazy refinement");
            };
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "a");

            filter.revoked.store(true, Ordering::SeqCst);
            let Some(SearchPhase::RefinementFailed {
                initial_results,
                error,
                ..
            }) = phases.next()
            else {
                panic!("revoked retained input {denied} must refuse refinement");
            };
            assert!(initial_results.is_empty());
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, value, .. }
                    if field == "search.filter" && value == "eligibility_changed"
            ));
            assert!(phases.next().is_none());
            assert!(phases.next().is_none());
        }
    }

    #[test]
    fn sync_refinement_rechecks_retained_fast_inputs_after_quality_callbacks() {
        use std::sync::atomic::AtomicBool;

        struct LateRevocation {
            armed: AtomicBool,
            revoked: AtomicBool,
        }

        impl SearchFilter for LateRevocation {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                if id == "b" && self.armed.load(Ordering::SeqCst) {
                    self.revoked.store(true, Ordering::SeqCst);
                }
                id != "a" || !self.revoked.load(Ordering::SeqCst)
            }

            fn name(&self) -> &'static str {
                "quality-callback-revocation"
            }
        }

        let directory = temp_dir("sync-refinement-callback-revocation");
        let (fast, quality) = persisted_producer_pair(directory.path());
        let index = Arc::new(InMemoryTwoTierIndex::from_dir(directory.path()).unwrap());
        let query = TieredQueryEmbeddings::progressive(
            BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap(),
            BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap(),
        );
        let searcher = SyncTwoTierSearcher::new(
            index,
            TwoTierConfig {
                candidate_multiplier: 1,
                quality_weight: 1.0,
                ..TwoTierConfig::default()
            },
        )
        .with_search_params(SearchParams {
            parallel_enabled: false,
            ..SearchParams::default()
        });
        let filter = LateRevocation {
            armed: AtomicBool::new(false),
            revoked: AtomicBool::new(false),
        };
        let mut phases = searcher.search_iter_with_filter(&query, 3, Some(&filter));
        let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
            panic!("Initial must precede quality callbacks");
        };
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "a");
        assert!(!filter.revoked.load(Ordering::SeqCst));

        filter.armed.store(true, Ordering::SeqCst);
        let Some(SearchPhase::RefinementFailed {
            initial_results,
            error,
            ..
        }) = phases.next()
        else {
            panic!("quality callbacks cannot readmit revoked fast evidence");
        };
        assert!(filter.revoked.load(Ordering::SeqCst));
        assert!(initial_results.is_empty());
        assert!(matches!(
            error,
            SearchError::InvalidConfig { field, value, .. }
                if field == "search.filter" && value == "eligibility_changed"
        ));
        assert!(phases.next().is_none());
        assert!(phases.next().is_none());
    }

    #[cfg(feature = "rerank")]
    #[test]
    fn sync_refinement_admission_cancellation_precedes_filter_refusal() {
        use std::sync::atomic::AtomicBool;

        struct CancelAndReject<'a> {
            cx: &'a asupersync::Cx,
            armed: AtomicBool,
        }

        impl SearchFilter for CancelAndReject<'_> {
            fn matches(&self, _: &str, _: Option<&serde_json::Value>) -> bool {
                if self.armed.load(Ordering::SeqCst) {
                    self.cx.cancel_with(
                        asupersync::CancelKind::User,
                        Some("cancel during refinement admission"),
                    );
                    return false;
                }
                true
            }

            fn name(&self) -> &'static str {
                "cancel-refinement-admission"
            }
        }

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
            let query = tiered(vec![1.0, 0.0]);
            let filter = CancelAndReject {
                cx: &cx,
                armed: AtomicBool::new(false),
            };
            let text = |_: &str| -> Option<String> {
                panic!("an unconfigured ranker cannot request text")
            };
            let mut phases = searcher.search_iter_with_text_and_filter(
                &cx,
                "query",
                &query,
                1,
                &text,
                Some(&filter),
            );
            assert!(matches!(phases.next(), Some(SearchPhase::Initial { .. })));
            filter.armed.store(true, Ordering::SeqCst);
            let Some(SearchPhase::RefinementFailed {
                initial_results,
                error,
                ..
            }) = phases.next()
            else {
                panic!("predicate cancellation must terminate refinement");
            };
            assert!(initial_results.is_empty());
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert!(phases.next().is_none());
        });
    }

    #[test]
    fn filter_is_applied_to_fast_and_refined_results() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect_with_filter(&tiered(vec![1.0, 0.0]), 3, Some(&ExcludeB))
            .unwrap();
        assert!(results.iter().all(|result| result.doc_id != "b"));
    }

    /// An empty query vector is now refused at BIND time, not at search time.
    ///
    /// This test previously called `search_collect(&[], 3)` and asserted the
    /// searcher returned `DimensionMismatch`. It had been failing on trunk:
    /// the empty slice reached the scan and came back `Ok` with a
    /// `ZeroNormQuery` classification instead. The typed query surface makes
    /// the question moot — a zero-length vector cannot be bound to a
    /// two-dimensional space at all, so the refusal now happens strictly
    /// earlier and cannot be reached by a search. Same defect, moved to where
    /// it is provable.
    #[test]
    fn an_empty_query_vector_cannot_even_be_bound() {
        let error = BoundQueryEmbedding::new(
            Vec::new(),
            EmbeddingIdentityBundleV1::explicit_test_model("sync-fixture", 2),
        )
        .expect_err("a zero-length vector is not a member of a 2-dimensional space");
        assert!(
            matches!(error, SearchError::DimensionMismatch { .. }),
            "got {error:?}"
        );
    }

    #[test]
    fn lexical_fusion_can_introduce_lexical_only_hits() {
        let lexical = Arc::new(StaticLexical {
            hits: vec![lexical_result("lex-only", 10.0), lexical_result("a", 9.0)],
        });
        let searcher =
            SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(lexical);
        let (results, _) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 3).unwrap();
        assert!(results.iter().any(|result| result.doc_id == "lex-only"));
        assert_eq!(
            results
                .iter()
                .find(|result| result.doc_id == "lex-only")
                .map(|result| result.source),
            Some(ScoreSource::Lexical),
            "a fused lexical-only hit is not hybrid semantic search"
        );
    }

    #[test]
    fn lexical_metadata_survives_sync_fusion_and_refinement_without_deep_clones() {
        let lexical_only = Arc::new(serde_json::json!({
            "path": "src/lexical.rs",
            "title": "Lexical result",
            "nested": { "revision": 7 },
        }));
        let hybrid = Arc::new(serde_json::json!({
            "path": "src/hybrid.rs",
            "title": "Hybrid result",
        }));
        let mut lexical_hit = lexical_result("lex-only", 10.0);
        lexical_hit.metadata = Some(Arc::clone(&lexical_only));
        let mut shared_hit = lexical_result("a", 9.0);
        shared_hit.metadata = Some(Arc::clone(&hybrid));
        let lexical = Arc::new(StaticLexical {
            hits: vec![lexical_hit, shared_hit],
        });
        let check_metadata = |results: &[ScoredResult]| {
            assert_eq!(results.len(), 4);
            for hit in results {
                let expected = match hit.doc_id.as_str() {
                    "lex-only" => {
                        assert_eq!(hit.source, ScoreSource::Lexical);
                        &lexical_only
                    }
                    "a" => {
                        assert_eq!(hit.source, ScoreSource::Hybrid);
                        &hybrid
                    }
                    "b" | "c" => {
                        assert!(
                            hit.metadata.is_none(),
                            "semantic-only hits have no stored payload"
                        );
                        continue;
                    }
                    id => panic!("unexpected result {id}"),
                };
                let actual = hit.metadata.as_ref().expect("retained lexical payload");
                assert!(Arc::ptr_eq(actual, expected), "clone only a winner's Arc");
                assert_eq!(actual, expected);
            }
        };
        for fast_only in [false, true] {
            let searcher = SyncTwoTierSearcher::new(
                make_index(),
                TwoTierConfig {
                    fast_only,
                    ..TwoTierConfig::default()
                },
            )
            .with_lexical(lexical.clone());
            let query = tiered(vec![1.0, 0.0]);
            let (results, _) = searcher.search_collect(&query, 4).unwrap();
            check_metadata(&results);
            let phases = searcher.search_iter(&query, 4).collect::<Vec<_>>();
            assert_eq!(phases.len(), if fast_only { 1 } else { 2 });
            for phase in phases {
                match phase {
                    SearchPhase::Initial { results, .. } | SearchPhase::Refined { results, .. } => {
                        check_metadata(&results);
                    }
                    other => panic!("unexpected phase: {other:?}"),
                }
            }
        }
    }

    #[cfg(feature = "rerank")]
    mod cross_encoder {
        use super::*;
        use frankensearch_core::traits::{RerankDocument, RerankScore};
        use frankensearch_rerank::RerankCombine;
        use std::sync::atomic::AtomicBool;

        #[derive(Clone, Copy)]
        enum Reply {
            Valid,
            Error,
            Malformed,
            Cancelled,
        }

        struct Probe {
            reply: Reply,
            calls: AtomicUsize,
            inputs: Mutex<Vec<Vec<DocId>>>,
            caller: std::thread::ThreadId,
            on_response: Option<Arc<dyn Fn() + Send + Sync>>,
        }

        impl Probe {
            fn new(reply: Reply) -> Self {
                Self {
                    reply,
                    calls: AtomicUsize::new(0),
                    inputs: Mutex::new(Vec::new()),
                    caller: std::thread::current().id(),
                    on_response: None,
                }
            }
        }

        impl SyncRerank for Probe {
            fn rerank_sync(
                &self,
                query: &str,
                documents: &[RerankDocument],
            ) -> SearchResult<Vec<RerankScore>> {
                assert_eq!(std::thread::current().id(), self.caller);
                assert_eq!(query, "query text");
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.inputs.lock().unwrap().push(
                    documents
                        .iter()
                        .map(|doc| DocId::from(doc.doc_id.as_str()))
                        .collect(),
                );
                for document in documents {
                    assert_eq!(document.text, format!("text-{}", document.doc_id));
                }
                if let Some(on_response) = &self.on_response {
                    on_response();
                }
                match self.reply {
                    Reply::Error => {
                        return Err(SearchError::RerankFailed {
                            model: "sync-probe".to_owned(),
                            source: "ordinary synchronous failure".into(),
                        });
                    }
                    Reply::Cancelled => {
                        return Err(SearchError::Cancelled {
                            phase: "fixture.sync_reranker".to_owned(),
                            reason: "provider stopped".to_owned(),
                        });
                    }
                    Reply::Valid | Reply::Malformed => {}
                }
                let mut scores = documents
                    .iter()
                    .enumerate()
                    .map(|(original_rank, document)| RerankScore {
                        doc_id: document.doc_id.clone(),
                        score: match document.doc_id.as_str() {
                            "c" => 0.9,
                            "a" => 0.7,
                            _ => 0.1,
                        },
                        original_rank,
                        raw_logit: None,
                    })
                    .collect::<Vec<_>>();
                if matches!(self.reply, Reply::Malformed)
                    && let Some(score) = scores.first_mut()
                {
                    score.doc_id = "foreign-document".to_owned();
                }
                scores.reverse();
                Ok(scores)
            }

            fn id(&self) -> &'static str {
                "sync-probe"
            }

            fn model_name(&self) -> &'static str {
                "sync-probe"
            }
        }

        fn fixture(
            mode: &str,
            attested: bool,
        ) -> (
            tempfile::TempDir,
            Arc<InMemoryTwoTierIndex>,
            TieredQueryEmbeddings,
        ) {
            let directory = temp_dir("sync-cross-encoder");
            let (index, fast, quality) = sync_mmr_fixture(directory.path(), attested, true);
            let fast = BoundQueryEmbedding::new(vec![1.0, 0.0], fast).unwrap();
            let quality = BoundQueryEmbedding::new(vec![1.0, 0.0, 0.0], quality).unwrap();
            let query = match mode {
                "fast" => TieredQueryEmbeddings::fast_only(fast),
                "quality" => TieredQueryEmbeddings::quality_only(quality),
                _ => TieredQueryEmbeddings::progressive(fast, quality),
            };
            (directory, index, query)
        }

        fn config() -> TwoTierConfig {
            TwoTierConfig {
                candidate_multiplier: 1,
                quality_weight: 0.0,
                explain: true,
                ..TwoTierConfig::default()
            }
        }

        fn configured(index: Arc<InMemoryTwoTierIndex>, probe: Arc<Probe>) -> SyncTwoTierSearcher {
            SyncTwoTierSearcher::new(index, config())
                .with_reranker(probe)
                .with_rerank_window(3)
                .unwrap()
        }

        fn retrieval(phases: &mut SyncSearchIterator<'_>, mode: &str) -> Vec<ScoredResult> {
            let Some(SearchPhase::Initial { results, .. }) = phases.next() else {
                panic!("text-aware search must first publish retrieval");
            };
            if mode == "progressive" {
                let Some(SearchPhase::Refined { results, .. }) = phases.next() else {
                    panic!("quality retrieval must precede cross-encoder inference");
                };
                results
            } else {
                results
            }
        }

        fn snapshot(results: &[ScoredResult]) -> serde_json::Value {
            serde_json::to_value(results).unwrap()
        }

        #[test]
        fn sync_cross_encoder_is_lazy_and_promotes_hidden_candidates_in_every_topology() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for attested in [false, true] {
                    for mode in ["fast", "quality", "progressive"] {
                        let (_directory, index, query) = fixture(mode, attested);
                        let probe = Arc::new(Probe::new(Reply::Valid));
                        let searcher = configured(index, probe.clone());
                        let lookups = AtomicUsize::new(0);
                        let text = |id: &str| {
                            lookups.fetch_add(1, Ordering::Relaxed);
                            Some(format!("text-{id}"))
                        };
                        drop(searcher.search_iter_with_text(&cx, "query text", &query, 1, &text));
                        assert_eq!(lookups.load(Ordering::Relaxed), 0);
                        let mut stopped =
                            searcher.search_iter_with_text(&cx, "query text", &query, 1, &text);
                        let initial = retrieval(&mut stopped, mode);
                        assert_eq!(initial[0].doc_id, "a");
                        assert_eq!(stopped.size_hint(), (0, Some(1)));
                        drop(stopped);
                        assert_eq!(lookups.load(Ordering::Relaxed), 0);
                        assert_eq!(probe.calls.load(Ordering::Relaxed), 0);

                        let mut phases =
                            searcher.search_iter_with_text(&cx, "query text", &query, 1, &text);
                        assert_eq!(retrieval(&mut phases, mode)[0].doc_id, "a");
                        let Some(SearchPhase::Reranked {
                            results, metrics, ..
                        }) = phases.next()
                        else {
                            panic!("a valid cross-encoder must publish its own final phase");
                        };
                        assert_eq!(results.len(), 1);
                        assert_eq!(results[0].doc_id, "c");
                        assert_eq!(results[0].source, ScoreSource::Reranked);
                        assert_eq!(results[0].rerank_score, Some(0.9));
                        assert_eq!(metrics.embedder_id, "sync-probe");
                        assert_eq!(lookups.load(Ordering::Relaxed), 3);
                        assert_eq!(probe.inputs.lock().unwrap()[0], ["a", "b", "c"]);
                        assert!(phases.next().is_none());
                        assert!(phases.next().is_none());

                        let (collected, _) = searcher
                            .search_collect_with_text(&cx, "query text", &query, 1, &text)
                            .unwrap();
                        assert_eq!(snapshot(&collected), snapshot(&results));
                        assert_eq!(probe.calls.load(Ordering::Relaxed), 2);
                        assert_eq!(lookups.load(Ordering::Relaxed), 6);
                    }
                }
            });
        }

        #[test]
        fn sync_cross_encoder_combination_order_survives_mmr() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for mode in ["fast", "quality", "progressive"] {
                    let (_directory, index, query) = fixture(mode, true);
                    for combine in [
                        RerankCombine::PureReorder,
                        RerankCombine::RrfCombine { k: 60.0 },
                    ] {
                        for mmr in [false, true] {
                            let probe = Arc::new(Probe::new(Reply::Valid));
                            let searcher = configured(Arc::clone(&index), probe)
                                .with_rerank_combine(combine)
                                .with_mmr_config(MmrConfig {
                                    enabled: mmr,
                                    lambda: 1.0,
                                    candidate_pool: 3,
                                });
                            let text = |id: &str| Some(format!("text-{id}"));
                            let (results, _) = searcher
                                .search_collect_with_text(&cx, "query text", &query, 3, &text)
                                .unwrap();
                            let expected = if matches!(combine, RerankCombine::PureReorder) {
                                ["c", "a", "b"]
                            } else {
                                ["a", "c", "b"]
                            };
                            assert_eq!(
                                results
                                    .iter()
                                    .map(|hit| hit.doc_id.as_str())
                                    .collect::<Vec<_>>(),
                                expected
                            );
                            assert!(results.iter().all(|hit| hit.rerank_score.is_some()));
                            if mode == "progressive" {
                                for (rank, hit) in results.iter().enumerate() {
                                    assert_eq!(
                                        hit.explanation
                                            .as_ref()
                                            .unwrap()
                                            .rank_movement
                                            .as_ref()
                                            .unwrap()
                                            .refined_rank,
                                        rank
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }

        #[test]
        fn sync_cross_encoder_skips_preserve_results_and_compact_text_can_promote() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for mode in ["fast", "quality", "progressive"] {
                    let (_directory, index, query) = fixture(mode, true);
                    for case in ["missing", "provider_error", "malformed", "single_text"] {
                        let reply = match case {
                            "provider_error" => Reply::Error,
                            "malformed" => Reply::Malformed,
                            _ => Reply::Valid,
                        };
                        let probe = Arc::new(Probe::new(reply));
                        let searcher = configured(Arc::clone(&index), probe.clone());
                        let text = |id: &str| {
                            if case == "missing" || (case == "single_text" && id != "c") {
                                None
                            } else {
                                Some(format!("text-{id}"))
                            }
                        };
                        let mut phases =
                            searcher.search_iter_with_text(&cx, "query text", &query, 1, &text);
                        let previous = retrieval(&mut phases, mode);
                        let expected = if case == "single_text" {
                            let Some(SearchPhase::Reranked { results, .. }) = phases.next() else {
                                panic!("one available hidden document can be reranked");
                            };
                            assert_eq!(results[0].doc_id, "c");
                            assert_eq!(probe.inputs.lock().unwrap()[0], ["c"]);
                            results
                        } else {
                            assert!(phases.next().is_none(), "skips do not fabricate a phase");
                            previous
                        };
                        assert!(phases.next().is_none());
                        let (collected, _) = searcher
                            .search_collect_with_text(&cx, "query text", &query, 1, &text)
                            .unwrap();
                        assert_eq!(snapshot(&collected), snapshot(&expected));
                        assert_eq!(
                            probe.calls.load(Ordering::Relaxed),
                            if case == "missing" { 0 } else { 2 }
                        );
                    }
                }
            });
        }

        #[test]
        fn sync_cross_encoder_old_apis_bounds_and_admission_keep_their_contracts() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let (_directory, index, query) = fixture("fast", true);
                let probe = Arc::new(Probe::new(Reply::Valid));
                let searcher = configured(Arc::clone(&index), probe.clone());
                let baseline = SyncTwoTierSearcher::new(Arc::clone(&index), config());
                let (expected, _) = baseline.search_collect(&query, 1).unwrap();
                let (old_api, metrics) = searcher.search_collect(&query, 1).unwrap();
                assert_eq!(snapshot(&old_api), snapshot(&expected));
                assert_eq!(metrics.rerank_ms, 0.0);
                assert!(matches!(
                    searcher
                        .search_iter(&query, 1)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    [SearchPhase::Initial { .. }]
                ));
                assert_eq!(probe.calls.load(Ordering::Relaxed), 0);
                let unreachable_text = |_: &str| -> Option<String> {
                    panic!("no model, zero k or refused identity must not request text")
                };
                let (without_model, _) = baseline
                    .search_collect_with_text(&cx, "query text", &query, 1, &unreachable_text)
                    .unwrap();
                assert_eq!(snapshot(&without_model), snapshot(&expected));
                let (empty, _) = searcher
                    .search_collect_with_text(&cx, "query text", &query, 0, &unreachable_text)
                    .unwrap();
                assert!(empty.is_empty());
                assert!(
                    searcher
                        .search_iter_with_text(&cx, "query text", &query, 0, &unreachable_text)
                        .next()
                        .is_none()
                );
                for window in [0, 1_001] {
                    assert!(matches!(
                        SyncTwoTierSearcher::new(Arc::clone(&index), config())
                            .with_rerank_window(window),
                        Err(SearchError::InvalidConfig { ref field, .. })
                            if field == "rerank.window"
                    ));
                }
                let mut foreign = query.fast().unwrap().identity().clone();
                "foreign-producer".clone_into(&mut foreign.producer.backend);
                let foreign = TieredQueryEmbeddings::fast_only(
                    BoundQueryEmbedding::new(vec![1.0, 0.0], foreign).unwrap(),
                );
                for k in [0, 1] {
                    assert!(matches!(
                        searcher.search_collect_with_text(
                            &cx, "query text", &foreign, k, &unreachable_text,
                        ),
                        Err(SearchError::InvalidConfig { ref field, .. })
                            if field == "search_activation.fast.producer_conformance"
                    ));
                    let mut refused = searcher.search_iter_with_text(
                        &cx,
                        "query text",
                        &foreign,
                        k,
                        &unreachable_text,
                    );
                    assert!(matches!(
                        refused.next(),
                        Some(SearchPhase::RefinementFailed {
                            initial_results,
                            error: SearchError::InvalidConfig { .. },
                            ..
                        }) if initial_results.is_empty()
                    ));
                    assert!(refused.next().is_none());
                }
                assert_eq!(probe.calls.load(Ordering::Relaxed), 0);
            });
        }

        struct Policy(Arc<AtomicBool>);

        impl SearchFilter for Policy {
            fn matches(&self, id: &str, _: Option<&serde_json::Value>) -> bool {
                id != "c" || self.0.load(Ordering::Relaxed)
            }

            fn name(&self) -> &'static str {
                "sync-cross-encoder-policy"
            }
        }

        fn assert_terminal_error(error: &SearchError, cancelled: bool) {
            if cancelled {
                assert!(matches!(
                    error,
                    SearchError::Cancelled { phase, .. } if phase == "fixture.sync_reranker"
                ));
            } else {
                assert!(matches!(
                    error,
                    SearchError::InvalidConfig { field, value, .. }
                        if field == "search.filter" && value == "eligibility_changed"
                ));
            }
        }

        #[test]
        #[allow(clippy::too_many_lines)]
        fn sync_cross_encoder_rechecks_hidden_candidates_at_every_callback_boundary() {
            asupersync::test_utils::run_test_with_cx(|cx| async move {
                for mode in ["fast", "quality", "progressive"] {
                    let (_directory, index, query) = fixture(mode, true);
                    for event in [
                        "between",
                        "first_text",
                        "last_text",
                        "response",
                        "provider_error",
                        "malformed",
                        "provider_cancelled",
                    ] {
                        for missing_text in [false, true] {
                            if missing_text && !matches!(event, "first_text" | "last_text") {
                                continue;
                            }
                            let eligible = Arc::new(AtomicBool::new(true));
                            let filter = Policy(Arc::clone(&eligible));
                            let reply = match event {
                                "provider_error" => Reply::Error,
                                "malformed" => Reply::Malformed,
                                "provider_cancelled" => Reply::Cancelled,
                                _ => Reply::Valid,
                            };
                            let mut probe = Probe::new(reply);
                            let during_response = matches!(
                                event,
                                "response" | "provider_error" | "malformed" | "provider_cancelled"
                            );
                            if during_response {
                                let eligible = Arc::clone(&eligible);
                                probe.on_response = Some(Arc::new(move || {
                                    eligible.store(false, Ordering::Relaxed);
                                }));
                            }
                            let probe = Arc::new(probe);
                            let searcher = configured(Arc::clone(&index), probe.clone());
                            let lookups = AtomicUsize::new(0);
                            let text = |id: &str| {
                                let lookup = lookups.fetch_add(1, Ordering::Relaxed) + 1;
                                let revoke = (event == "first_text" && lookup == 1)
                                    || (event == "last_text" && lookup == 3);
                                if revoke {
                                    eligible.store(false, Ordering::Relaxed);
                                }
                                if revoke && missing_text {
                                    None
                                } else {
                                    Some(format!("text-{id}"))
                                }
                            };
                            let mut phases = searcher.search_iter_with_text_and_filter(
                                &cx,
                                "query text",
                                &query,
                                1,
                                &text,
                                Some(&filter),
                            );
                            assert_eq!(retrieval(&mut phases, mode)[0].doc_id, "a");
                            if event == "between" {
                                eligible.store(false, Ordering::Relaxed);
                            }
                            let Some(SearchPhase::RefinementFailed {
                                initial_results,
                                error,
                                ..
                            }) = phases.next()
                            else {
                                panic!("revocation must terminate without a stale fallback");
                            };
                            assert!(initial_results.is_empty());
                            assert_terminal_error(&error, event == "provider_cancelled");
                            let expected_lookups = match event {
                                "between" => 0,
                                "first_text" => 1,
                                _ => 3,
                            };
                            assert_eq!(lookups.load(Ordering::Relaxed), expected_lookups);
                            assert_eq!(
                                probe.calls.load(Ordering::Relaxed),
                                usize::from(during_response)
                            );
                            assert!(phases.next().is_none());
                            drop(phases);

                            if event != "between" {
                                eligible.store(true, Ordering::Relaxed);
                                lookups.store(0, Ordering::Relaxed);
                                probe.calls.store(0, Ordering::Relaxed);
                                let error = searcher
                                    .search_collect_with_text_and_filter(
                                        &cx,
                                        "query text",
                                        &query,
                                        1,
                                        &text,
                                        Some(&filter),
                                    )
                                    .unwrap_err();
                                assert_terminal_error(&error, event == "provider_cancelled");
                                assert_eq!(lookups.load(Ordering::Relaxed), expected_lookups);
                                assert_eq!(
                                    probe.calls.load(Ordering::Relaxed),
                                    usize::from(during_response)
                                );
                            }
                        }
                    }
                }
            });
        }

        #[test]
        fn sync_cross_encoder_cancellation_is_terminal_for_iterators_and_collectors() {
            for route in ["iterator", "collect"] {
                for event in [
                    "before",
                    "pending",
                    "text",
                    "response",
                    "response_error",
                    "provider",
                ] {
                    if route == "collect" && event == "pending" {
                        continue;
                    }
                    asupersync::test_utils::run_test_with_cx(|cx| async move {
                        let (_directory, index, query) = fixture("fast", true);
                        let reply = match event {
                            "response_error" => Reply::Error,
                            "provider" => Reply::Cancelled,
                            _ => Reply::Valid,
                        };
                        let mut probe = Probe::new(reply);
                        if matches!(event, "response" | "response_error") {
                            let cancelled = cx.clone();
                            probe.on_response = Some(Arc::new(move || {
                                cancelled.cancel_with(
                                    asupersync::CancelKind::User,
                                    Some("synchronous response cancelled"),
                                );
                            }));
                        }
                        let probe = Arc::new(probe);
                        let searcher = configured(index, probe.clone());
                        let lookups = AtomicUsize::new(0);
                        let text = |id: &str| {
                            lookups.fetch_add(1, Ordering::Relaxed);
                            if event == "text" {
                                cx.cancel_with(
                                    asupersync::CancelKind::User,
                                    Some("synchronous text cancelled"),
                                );
                            }
                            Some(format!("text-{id}"))
                        };
                        if event == "before" {
                            cx.cancel_with(asupersync::CancelKind::User, Some("before retrieval"));
                        }
                        let error = if route == "collect" {
                            searcher
                                .search_collect_with_text(&cx, "query text", &query, 1, &text)
                                .unwrap_err()
                        } else {
                            let mut phases =
                                searcher.search_iter_with_text(&cx, "query text", &query, 1, &text);
                            if event != "before" {
                                let _ = retrieval(&mut phases, "fast");
                            }
                            if event == "pending" {
                                cx.cancel_with(
                                    asupersync::CancelKind::User,
                                    Some("before final ranking"),
                                );
                            }
                            let Some(SearchPhase::RefinementFailed {
                                initial_results,
                                error,
                                ..
                            }) = phases.next()
                            else {
                                panic!("cancellation must not publish a successful ranking");
                            };
                            assert!(initial_results.is_empty());
                            assert!(phases.next().is_none());
                            error
                        };
                        assert!(matches!(error, SearchError::Cancelled { .. }));
                        assert_eq!(
                            probe.calls.load(Ordering::Relaxed),
                            usize::from(matches!(
                                event,
                                "response" | "response_error" | "provider"
                            ))
                        );
                        assert_eq!(
                            lookups.load(Ordering::Relaxed),
                            match event {
                                "before" | "pending" => 0,
                                "text" => 1,
                                _ => 3,
                            }
                        );
                    });
                }
            }
        }

        #[test]
        fn sync_cross_encoder_ranks_a_singleton_with_pinned_metadata() {
            struct MetadataFilter;

            impl SearchFilter for MetadataFilter {
                fn matches(&self, _: &str, metadata: Option<&serde_json::Value>) -> bool {
                    metadata
                        .and_then(|value| value.get("allowed"))
                        .and_then(serde_json::Value::as_bool)
                        == Some(true)
                }

                fn name(&self) -> &'static str {
                    "require-pinned-metadata"
                }
            }

            asupersync::test_utils::run_test_with_cx(|cx| async move {
                let directory = temp_dir("sync-rerank-singleton");
                let (fast, identity) = attested_tier(
                    directory.path(),
                    "fast.fsvi",
                    "singleton-fast",
                    1,
                    &[("a", &[1.0, 0.0])],
                );
                let index = Arc::new(InMemoryTwoTierIndex::new(fast, None).unwrap());
                let query = TieredQueryEmbeddings::fast_only(
                    BoundQueryEmbedding::new(vec![1.0, 0.0], identity).unwrap(),
                );
                let metadata = Arc::new(serde_json::json!({"allowed": true, "revision": 7}));
                let mut hit = lexical_result("a", 1.0);
                hit.metadata = Some(Arc::clone(&metadata));
                let probe = Arc::new(Probe::new(Reply::Valid));
                let searcher = configured(index, probe.clone())
                    .with_lexical(Arc::new(StaticLexical { hits: vec![hit] }));
                let text = |id: &str| Some(format!("text-{id}"));
                let (results, _) = searcher
                    .search_collect_with_text_and_filter(
                        &cx,
                        "query text",
                        &query,
                        1,
                        &text,
                        Some(&MetadataFilter),
                    )
                    .unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].doc_id, "a");
                assert_eq!(results[0].source, ScoreSource::Reranked);
                assert_eq!(results[0].rerank_score, Some(0.7));
                assert!(Arc::ptr_eq(
                    results[0].metadata.as_ref().unwrap(),
                    &metadata
                ));
                assert_eq!(probe.calls.load(Ordering::Relaxed), 1);
            });
        }
    }

    #[cfg(feature = "quill")]
    #[test]
    fn quill_sync_adapter_is_read_only_and_maps_cancellation() {
        use frankensearch_core::IndexableDocument;
        use frankensearch_quill::{QuillConfig, QuillIndex, QuillSearchIndex};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<QuillSyncLexicalSearch>();

            let directory = tempfile::tempdir().expect("Quill fixture directory");
            let config = QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            };
            let writer = QuillIndex::create(&cx, directory.path(), config.clone())
                .await
                .expect("create durable Quill writer");
            writer
                .index_documents(
                    &cx,
                    &[IndexableDocument::new(
                        "quill-only",
                        "native quill lexical result",
                    )],
                )
                .await
                .expect("index Quill fixture");
            writer.commit(&cx).await.expect("commit Quill fixture");
            drop(writer);

            let quill = Arc::new(
                QuillSearchIndex::open(&cx, directory.path(), config.clone())
                    .await
                    .expect("open read-only Quill search handle"),
            );
            let reopened_writer = QuillIndex::open(&cx, directory.path(), config)
                .await
                .expect("read-only search handle must not retain the writer lease");
            drop(reopened_writer);

            let lexical = Arc::new(QuillSyncLexicalSearch::new(
                Arc::clone(&quill),
                cx.clone(),
                "native quill",
            ));
            assert_eq!(lexical.query(), "native quill");
            let hits = lexical.search_sync(&[f32::NAN], 1).expect("sync search");
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].doc_id, "quill-only");
            assert_eq!(hits[0].source, ScoreSource::Lexical);
            assert_eq!(hits[0].lexical_score, Some(hits[0].score));

            let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default())
                .with_lexical(lexical);
            let (fused, _) = searcher
                .search_collect(&tiered(vec![1.0, 0.0]), 3)
                .expect("hybrid sync search");
            assert!(fused.iter().any(|hit| hit.doc_id == "quill-only"));

            let cancelled = cx.clone();
            cancelled.set_cancel_requested(true);
            let adapter = QuillSyncLexicalSearch::new(quill, cancelled, "native");
            assert!(matches!(
                adapter.search_sync(&[], 1),
                Err(SearchError::Cancelled { ref phase, .. }) if phase == "search"
            ));
        });
    }
}
