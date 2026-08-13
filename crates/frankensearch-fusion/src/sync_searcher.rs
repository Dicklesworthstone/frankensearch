//! Synchronous two-tier search orchestration for low-latency UIs.
//!
//! [`SyncTwoTierSearcher`] mirrors the progressive two-phase contract of
//! [`crate::searcher::TwoTierSearcher`] but operates on precomputed query
//! embeddings and fully in-memory indices.

use std::collections::VecDeque;
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
use frankensearch_core::filter::SearchFilter;
use frankensearch_core::types::{BoundQueryEmbedding, TieredQueryEmbeddings};
use frankensearch_core::{
    FusedHit, PhaseMetrics, ScoreSource, ScoredResult, SearchError, SearchPhase, SearchResult,
    TwoTierConfig, TwoTierMetrics, VectorHit, ZeroSignalReason,
};
use frankensearch_index::{
    InMemoryTwoTierIndex, InMemoryVectorIndex, SearchParams, ValidatedFsviBytes,
};

use crate::blend::{blend_two_tier, blend_two_tier_aligned_vector_index, compute_rank_changes};
use crate::normalize::{AdaptiveNqcDenseWeight, NqcDenseWeight, nqc_cv_iter};
use crate::rrf::{RrfConfig, RrfTiebreak, candidate_count, fuse_by_strategy};

/// The per-tier query embeddings this search is authorized to read with
/// (bd-sync-searcher-tiered-query-embeddings-dbp10).
///
/// Produced by [`SyncTwoTierSearcher::admit`] BEFORE any vector is touched.
/// Holding one is the proof that every tier this search will consult has had
/// its space identity joined against the query bound to it — or has been
/// typed as legacy-unidentified, which is a state, not an omission.
struct AdmittedSyncQuery<'query> {
    /// The fast-tier query. Always present: the sync contract is
    /// fast-then-refine, so a search with no fast arm has no phase 1.
    fast: &'query BoundQueryEmbedding,
    /// The quality-tier query, when the caller bound one AND this index has a
    /// quality tier. `None` is why phase 2 is skipped, not a licence to reuse
    /// `fast` against the quality index.
    quality: Option<&'query BoundQueryEmbedding>,
}

/// What the quality tier contributed to a synchronous refinement.
///
/// Same distinction the async searcher draws: `Retrieved` is the quality
/// tier's OWN top-k and can name documents the fast tier never selected;
/// `RescoredFastPool` can only re-rank what the fast tier already chose.
/// Which one runs is decided by whether the quality tier's space identity is
/// attested — never by convenience — so the two searchers expand the returned
/// candidate set under the same condition.
enum SyncQualityPool {
    /// The quality tier's own top-k for the quality-bound query.
    Retrieved(Vec<VectorHit>),
    /// Quality scores aligned positionally to the fast pool. Still computed
    /// from the QUALITY-bound query: the aligned SHAPE was never the bug, the
    /// vector fed into it was.
    RescoredFastPool(Vec<Option<f32>>),
}

/// Join one bound query against one tier's retained space identity, before
/// any vector read.
///
/// `space_fingerprint_hex` is `None` for a legacy artifact (v1 FSVI, or the
/// identity-less `from_vectors` constructor). That is a legal typed state:
/// there is no identity to join against, so the query is admitted and the
/// search proceeds exactly as it did before this type existed. Fabricating a
/// fingerprint to "have something to check" would be worse than checking
/// nothing, because it would look like a passing join.
///
/// The join is [`BoundQueryEmbedding::verify_space_identity`], the
/// fingerprint-only check — NOT the full bd-9xuj admission law. That is a
/// real limit and it is the index type's, not this function's:
/// [`InMemoryVectorIndex`] retains only the space fingerprint, never the
/// expected identity bundle, so there is nothing here to attest a producer
/// against. A matching space fingerprint is necessary, not sufficient; the
/// persistent `TwoTierIndex` path applies the complete law because it retains
/// the admitted binding.
fn admit_tier_space(
    tier_index: &InMemoryVectorIndex,
    query: &BoundQueryEmbedding,
    tier: &str,
) -> SearchResult<()> {
    tier_index.space_fingerprint_hex().map_or_else(
        || Ok(()),
        |expected| query.verify_space_identity(expected, tier),
    )
}

/// Optional synchronous lexical backend used by [`SyncTwoTierSearcher`].
pub trait SyncLexicalSearch: Send + Sync {
    /// Retrieve lexical candidates for the current query.
    ///
    /// Implementations may ignore `query_vec` when they already have external
    /// query context.
    ///
    /// # Errors
    ///
    /// Returns backend-specific lexical retrieval errors.
    fn search_sync(&self, query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>>;
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
        self.index
            .search_results(&self.cx, &self.query, limit)
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

    /// Override brute-force parallel search parameters for fast-tier retrieval.
    #[must_use]
    pub const fn with_search_params(mut self, params: SearchParams) -> Self {
        self.search_params = Some(params);
        self
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

    /// Execute a synchronous search and return the final result set + metrics.
    ///
    /// `query` carries one bound embedding PER TIER. The former signature took
    /// a single `&[f32]` and used it for both tiers, which meant the quality
    /// index was scored with the fast tier's query — a cross-space read that
    /// no dimension check can catch, because the two tiers usually have the
    /// same width. That vector is no longer representable here.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the query binds no fast
    /// embedding, or when a tier's retained space identity does not match the
    /// query bound to it; plus dimension/filter errors from vector search and
    /// lexical backend failures (when lexical fusion is enabled).
    pub fn search_collect(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_with_filter(query, k, None)
    }

    /// Execute a synchronous search with an optional doc-level filter.
    ///
    /// # Errors
    ///
    /// Identical to [`Self::search_collect`].
    pub fn search_collect_with_filter(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        // `search_collect` discards `outcome.phases`, so skip building them — that
        // avoids cloning the full `Vec<ScoredResult>` (N owned doc_ids each) once
        // per phase (Initial + Refined), pure waste at large `k` (limit_all).
        let outcome = self.search_internal(query, k, filter, false)?;
        Ok((outcome.final_results, outcome.metrics))
    }

    /// Execute a synchronous search and stream progressive phases via iterator.
    ///
    /// When admission or phase-1 retrieval fails (for example a query bound to
    /// a different embedding space than the index was built in, or a dimension
    /// mismatch), this returns an iterator yielding a single
    /// `RefinementFailed` phase carrying an empty `initial_results` payload.
    #[must_use]
    pub fn search_iter(&self, query: &TieredQueryEmbeddings, k: usize) -> SyncSearchIterator {
        self.search_iter_with_filter(query, k, None)
    }

    /// Execute a synchronous filtered search and stream progressive phases.
    #[must_use]
    pub fn search_iter_with_filter(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SyncSearchIterator {
        // The iterator streams the progressive phases, so build them.
        match self.search_internal(query, k, filter, true) {
            Ok(outcome) => SyncSearchIterator::new(outcome.phases),
            Err(error) => SyncSearchIterator::from_error(error),
        }
    }

    /// Join every tier this search will consult against the query bound to it,
    /// before any vector is read.
    ///
    /// This runs first in [`Self::search_internal`], ahead of the `k == 0` and
    /// zero-norm short circuits, so a query in the wrong space is refused even
    /// when the search would otherwise have exited without touching an index.
    /// A refusal that depended on the search getting far enough to do work
    /// would not be a guard.
    fn admit<'query>(
        &self,
        query: &'query TieredQueryEmbeddings,
    ) -> SearchResult<AdmittedSyncQuery<'query>> {
        let fast = query.fast().ok_or_else(|| SearchError::InvalidConfig {
            field: "sync_search.topology".to_owned(),
            value: format!("{:?}", query.supported_topology()),
            reason: "the synchronous progressive contract is fast-then-refine, so it requires a \
                     fast-tier query embedding; a quality-only retrieval has no Initial phase \
                     and belongs on the owner-backed activation surface instead"
                .to_owned(),
        })?;
        admit_tier_space(self.index.fast_index(), fast, "fast")?;

        let quality = match (query.quality(), self.index.quality_index()) {
            (Some(bound), Some(quality_index)) => {
                admit_tier_space(quality_index, bound, "quality")?;
                Some(bound)
            }
            // A quality-bound query against an index with no quality tier is
            // not an error — it is the ordinary fast-only index — and a
            // quality tier the caller bound no embedding for is simply never
            // consulted. Neither case licenses reusing the fast embedding.
            _ => None,
        };
        Ok(AdmittedSyncQuery { fast, quality })
    }

    #[allow(clippy::too_many_lines)]
    fn search_internal(
        &self,
        query: &TieredQueryEmbeddings,
        k: usize,
        filter: Option<&dyn SearchFilter>,
        want_phases: bool,
    ) -> SearchResult<SyncSearchOutcome> {
        // Identity first, before k == 0, before the zero-norm check, and
        // before any index is touched.
        let admitted = self.admit(query)?;
        let query_vec = admitted.fast.vector();
        let mut metrics = TwoTierMetrics::default();
        // Only the streaming iterator path consumes `phases`; `search_collect`
        // discards them. When they are not wanted, skip the allocation and the
        // per-phase `Vec<ScoredResult>` clones entirely (see the guarded pushes).
        let mut phases = if want_phases {
            Vec::with_capacity(2)
        } else {
            Vec::new()
        };
        // Match the async progressive contract: zero requested results is a
        // successful no-op, not a request to scan the corpus and discard the
        // candidate pool afterwards (bd-k3089).
        if k == 0 {
            return Ok(SyncSearchOutcome {
                phases,
                final_results: Vec::new(),
                metrics,
            });
        }
        if query_vec.iter().all(|&value| value == 0.0) {
            metrics.zero_signal = Some(ZeroSignalReason::ZeroNormQuery);
            return Ok(SyncSearchOutcome {
                phases,
                final_results: Vec::new(),
                metrics,
            });
        }
        let fetch = candidate_count(k, 0, self.config.candidate_multiplier.max(1)).max(k);

        let phase1_started = Instant::now();
        let fast_hits = self.search_fast_hits(query_vec, fetch, filter)?;
        // `phase1_vectors_searched` is diagnostic work accounting, not the
        // bounded candidate-pool size. The fast-tier search evaluates the
        // complete index before returning its top-k pool, matching the async
        // searcher's metric contract and `TwoTierMetrics` documentation.
        let phase1_vectors_searched = self.index.doc_count();
        metrics.phase1_vectors_searched = phase1_vectors_searched;
        metrics.semantic_candidates = fast_hits.len();
        // Typed zero-signal classification (bd-tqhc): an empty semantic lane
        // must carry why. Lazy — the non-empty path pays nothing.
        metrics.zero_signal = if fast_hits.is_empty() {
            Some(self.classify_fast_empty(query_vec, fetch, filter))
        } else {
            None
        };

        let lexical_started = Instant::now();
        let lexical_hits = self
            .lexical
            .as_ref()
            .map(|lexical| lexical.search_sync(query_vec, fetch))
            .transpose()?;
        let lexical_hits = lexical_hits.map(|hits| filter_lexical_hits(hits, filter));
        metrics.lexical_search_ms = ms(lexical_started.elapsed());
        metrics.lexical_candidates = lexical_hits.as_ref().map_or(0, Vec::len);

        let rrf_started = Instant::now();
        let query_semantic_weight = lexical_hits
            .as_ref()
            .map_or(self.rrf_semantic_weight, |lexical| {
                self.effective_semantic_weight(lexical)
            });
        let initial_results = lexical_hits.as_ref().map_or_else(
            || {
                vector_hits_to_scored_results(
                    &fast_hits,
                    k,
                    ScoreSource::SemanticFast,
                    None,
                    None,
                    &self.config,
                )
            },
            |lexical| {
                fused_hits_to_scored_results(
                    fuse_by_strategy(
                        self.config.fusion_strategy,
                        lexical,
                        &fast_hits,
                        &[],
                        0.0,
                        k,
                        0,
                        &RrfConfig {
                            k: self.config.rrf_k,
                            lexical_weight: self.rrf_lexical_weight,
                            semantic_weight: query_semantic_weight,
                            tiebreak: self.rrf_tiebreak,
                        },
                    ),
                    k,
                    &self.config,
                )
            },
        );
        metrics.rrf_fusion_ms = ms(rrf_started.elapsed());

        let phase1_latency = phase1_started.elapsed();
        metrics.vector_search_ms = ms(phase1_latency);
        metrics.phase1_total_ms = ms(phase1_latency);
        metrics.fast_embed_ms = 0.0;

        if want_phases {
            phases.push(SearchPhase::Initial {
                results: initial_results.clone(),
                latency: phase1_latency,
                metrics: PhaseMetrics {
                    embedder_id: "sync-fast-query".to_owned(),
                    vectors_searched: phase1_vectors_searched,
                    lexical_candidates: metrics.lexical_candidates,
                    fused_count: initial_results.len(),
                },
            });
        }

        // The quality arm needs three things: it must not be switched off, the
        // index must have a quality tier, and — new here — the caller must
        // have bound a QUALITY-space embedding for it. Without the third, the
        // refined phase used to run anyway, scoring the quality index with the
        // fast tier's vector.
        let quality_query = admitted.quality.filter(|_| !self.config.fast_only);
        let Some(quality_query) = quality_query else {
            // Same vocabulary as the async searcher (searcher.rs) — the two
            // sides share one skip_reason contract; "fast_only" is the string
            // the fsfs surfaces document (bd-k3089 parity suite pins this).
            metrics.skip_reason = Some(if self.config.fast_only {
                "fast_only".to_owned()
            } else if self.index.has_quality_index() {
                "quality_query_embedding_absent".to_owned()
            } else {
                "quality_index_unavailable".to_owned()
            });
            return Ok(SyncSearchOutcome {
                phases,
                final_results: initial_results,
                metrics,
            });
        };
        let quality_index = self
            .index
            .quality_index()
            .expect("admission binds a quality query only when a quality tier exists");

        let phase2_started = Instant::now();
        // TWO THINGS CHANGE HERE, and only one of them is unconditional.
        //
        // Unconditional: the quality tier is scored with the QUALITY-bound
        // query. It used to be scored with the fast tier's vector, which is a
        // cross-space read no dimension check can catch because the two tiers
        // usually have the same width. That was the defect.
        //
        // Conditional: independent RETRIEVAL from the quality tier — which can
        // surface a document the fast pool never selected (bd-ctzo C2) — runs
        // only when the quality tier's space identity is ATTESTED, i.e. read
        // out of an admitted FSVI v2 artifact's own header. That gate is not
        // timidity; it keeps this path behaviourally identical to the async
        // searcher, which can only retrieve through the owner-backed
        // activation and rescores on a legacy artifact. Expanding the returned
        // candidate set is conditioned on being able to prove the pool being
        // expanded into is the right space, on both sides. A legacy pair keeps
        // the aligned rescoring shape — with the correct vector.
        let quality_pool = if quality_index.space_identity_is_attested() {
            quality_index
                .search_top_k(quality_query.vector(), fetch, filter)
                .map(SyncQualityPool::Retrieved)
        } else {
            self.index
                .quality_scores_for_hits(quality_query.vector(), &fast_hits)
                .map(SyncQualityPool::RescoredFastPool)
        };
        let quality_pool = match quality_pool {
            Ok(pool) => pool,
            Err(error) => {
                let latency = phase2_started.elapsed();
                metrics.phase2_total_ms = ms(latency);
                metrics.skip_reason = Some(error.to_string());
                if want_phases {
                    phases.push(SearchPhase::RefinementFailed {
                        initial_results: initial_results.clone(),
                        error,
                        latency,
                    });
                }
                return Ok(SyncSearchOutcome {
                    phases,
                    final_results: initial_results,
                    metrics,
                });
            }
        };

        let blend_started = Instant::now();
        // Per-tier evidence is keyed by canonical doc_id in BOTH shapes. The
        // retired `AlignedScoreLookup` keyed it by fast-tier row ordinal, which
        // is only expressible while the quality tier is a subset of the fast
        // pool; a doc_id key costs one hash per row and is the only key an
        // independently retrieved pool can share.
        let fast_scores_by_doc: AHashMap<&str, f32> = fast_hits
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
        let quality_weight = saturating_f64_to_f32(self.config.quality_weight);
        let blended = match &quality_pool {
            SyncQualityPool::RescoredFastPool(scores) => {
                blend_two_tier_aligned_vector_index(&fast_hits, scores, quality_weight)
            }
            SyncQualityPool::Retrieved(hits) => {
                let mut blended = blend_two_tier(&fast_hits, hits, quality_weight);
                // `VectorHit::index` is a FAST-tier row ordinal for every
                // consumer downstream. A document only the quality tier
                // returned has no fast row, so it carries the "no index"
                // sentinel rather than a quality ordinal that would silently
                // read as a fast one.
                let fast_indices: AHashMap<&str, u32> = fast_hits
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
        metrics.quality_search_ms = ms(phase2_started.elapsed());
        metrics.quality_embed_ms = 0.0;

        // Keep this diagnostic on the full phase candidate pools, not only
        // the displayed top-k. The async searcher uses the same definition;
        // truncating here made identical searches report different stable /
        // promoted / demoted counts whenever `k < fetch` (bd-k3089).
        let rank_changes = compute_rank_changes(&fast_hits, &blended);

        let mut refined_results = if let Some(lexical) = lexical_hits.as_ref() {
            fused_hits_to_scored_results(
                fuse_by_strategy(
                    self.config.fusion_strategy,
                    lexical,
                    &blended,
                    &[],
                    0.0,
                    k,
                    0,
                    &RrfConfig {
                        k: self.config.rrf_k,
                        lexical_weight: self.rrf_lexical_weight,
                        semantic_weight: query_semantic_weight,
                        tiebreak: self.rrf_tiebreak,
                    },
                ),
                k,
                &self.config,
            )
        } else {
            unique_vector_hits_to_scored_results_owned(blended, k, ScoreSource::SemanticQuality)
        };

        // Re-fusion ranks on blended semantic scores, but `fast_score` and
        // `quality_score` are evidence fields: they must retain the raw
        // per-tier values. The async searcher restores this provenance after
        // lexical re-fusion; leaving the blended score in `fast_score` here
        // made the two APIs report different evidence for identical hits.
        // A document present in only one pool now truthfully reports `None`
        // for the tier that never saw it, instead of borrowing a neighbour's
        // row ordinal.
        for result in &mut refined_results {
            result.fast_score = fast_scores_by_doc.get(result.doc_id.as_str()).copied();
            result.quality_score = quality_scores_by_doc.get(result.doc_id.as_str()).copied();
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
            let (fast_min, fast_max) = finite_score_bounds(fast_hits.iter().map(|hit| hit.score));
            let (quality_min, quality_max) =
                finite_score_bounds(quality_scores_by_doc.values().copied());
            for (rank, result) in refined_results.iter_mut().enumerate() {
                result.explanation = Some(Box::new(build_refined_explanation(
                    result.score,
                    rank,
                    result.doc_id.as_str(),
                    result.fast_score,
                    result.quality_score,
                    &initial_ranks,
                    fast_min,
                    fast_max,
                    quality_min,
                    quality_max,
                    quality_weight,
                )));
            }
        }

        metrics.rank_changes = rank_changes.clone();
        metrics.phase2_total_ms = ms(phase2_started.elapsed());
        metrics.kendall_tau = None;

        if want_phases {
            phases.push(SearchPhase::Refined {
                results: refined_results.clone(),
                latency: phase2_started.elapsed(),
                metrics: PhaseMetrics {
                    embedder_id: "sync-quality-query".to_owned(),
                    vectors_searched: quality_count,
                    lexical_candidates: metrics.lexical_candidates,
                    fused_count: refined_results.len(),
                },
                rank_changes,
            });
        }

        Ok(SyncSearchOutcome {
            phases,
            final_results: refined_results,
            metrics,
        })
    }

    fn search_fast_hits(
        &self,
        query_vec: &[f32],
        fetch: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<Vec<VectorHit>> {
        let fast_index = self.index.fast_index();
        self.search_params.map_or_else(
            || match self.fast_retrieval {
                SyncFastRetrieval::Exact => {
                    // The public default is exact whether the optional cache is
                    // attached or unavailable. A verified sidecar may prune only
                    // inside `search_top_k`; it must never select a different
                    // candidate-generation algorithm or change visible results.
                    fast_index.search_top_k(query_vec, fetch, filter)
                }
                SyncFastRetrieval::ApproximateInt8ForBench {
                    candidate_multiplier,
                } => fast_index.search_top_k_int8_two_pass_filtered(
                    query_vec,
                    fetch,
                    candidate_multiplier,
                    filter,
                ),
            },
            // Explicit params: honour the exact scan + parallelism configuration.
            |params| fast_index.search_top_k_with_params(query_vec, fetch, filter, params),
        )
    }

    /// Classify an empty fast-tier result with its typed
    /// [`ZeroSignalReason`] (bd-tqhc).
    ///
    /// Request-scoped defects are classified from the request itself, in the
    /// documented precedence order; state-scoped reasons come from the
    /// index's own classified lane so in-memory and persistent paths agree.
    /// The classified re-scan runs only on the (already cheap) empty path,
    /// and its hits are never used for result production. Classification is
    /// diagnostic and must never fail the search: errors degrade to an
    /// unclassified lane at debug level rather than a fabricated reason.
    fn classify_fast_empty(
        &self,
        query_vec: &[f32],
        fetch: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> ZeroSignalReason {
        // Request-scoped conditions take precedence over index state, in the
        // order documented on `ZeroSignalReason`.
        if fetch == 0 {
            return ZeroSignalReason::CallerRequestedZeroK;
        }
        if query_vec.iter().any(|value| !value.is_finite()) {
            return ZeroSignalReason::NonFiniteQuery;
        }
        if query_vec.iter().all(|&value| value == 0.0) {
            return ZeroSignalReason::ZeroNormQuery;
        }
        // State-scoped reasons come from the index census rather than from a
        // second search. Re-scanning would cost a full extra pass on a path
        // that has already answered. The production and classified routes are
        // both exact, but the second scan could still observe a concurrent
        // index state change and fabricate a reason for a result it did not
        // produce. The census cannot disagree with itself, and always yields
        // a reason.
        self.index
            .fast_index()
            .zero_signal_state()
            .empty_result_reason(filter.is_some())
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
            .finish()
    }
}

#[derive(Debug)]
struct SyncSearchOutcome {
    phases: Vec<SearchPhase>,
    final_results: Vec<ScoredResult>,
    metrics: TwoTierMetrics,
}

/// Iterator over progressive phases produced by [`SyncTwoTierSearcher`].
#[derive(Debug)]
pub struct SyncSearchIterator {
    phases: VecDeque<SearchPhase>,
}

impl SyncSearchIterator {
    fn new(phases: Vec<SearchPhase>) -> Self {
        Self {
            phases: phases.into(),
        }
    }

    fn from_error(error: SearchError) -> Self {
        Self::new(vec![SearchPhase::RefinementFailed {
            initial_results: Vec::new(),
            error,
            latency: Duration::from_millis(0),
        }])
    }
}

impl Iterator for SyncSearchIterator {
    type Item = SearchPhase;

    fn next(&mut self) -> Option<Self::Item> {
        self.phases.pop_front()
    }
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

fn finite_score_bounds(scores: impl IntoIterator<Item = f32>) -> (f32, f32) {
    scores
        .into_iter()
        .filter(|score| score.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), score| {
            (min.min(score), max.max(score))
        })
}

#[allow(clippy::too_many_arguments)]
fn build_refined_explanation(
    final_score: f32,
    rank: usize,
    doc_id: &str,
    fast_score: Option<f32>,
    quality_score: Option<f32>,
    initial_ranks: &AHashMap<&str, usize>,
    fast_min: f32,
    fast_max: f32,
    quality_min: f32,
    quality_max: f32,
    quality_weight: f32,
) -> HitExplanation {
    let normalize = |score: f32, min: f32, max: f32| {
        if !score.is_finite() {
            return 0.0;
        }
        let range = max - min;
        if range > 0.01 {
            f64::from(((score - min) / range).clamp(0.0, 1.0))
        } else {
            f64::from(score.clamp(0.0, 1.0))
        }
    };
    let mut components = Vec::with_capacity(2);
    if let Some(score) = fast_score {
        components.push(ScoreComponent {
            source: ExplainedSource::SemanticFast {
                embedder: "sync-fast-query".to_owned(),
                cosine_sim: f64::from(score),
            },
            raw_score: f64::from(score),
            normalized_score: normalize(score, fast_min, fast_max),
            rrf_contribution: 0.0,
            weight: 1.0 - f64::from(quality_weight),
        });
    }
    if let Some(score) = quality_score {
        components.push(ScoreComponent {
            source: ExplainedSource::SemanticQuality {
                embedder: "sync-quality-query".to_owned(),
                cosine_sim: f64::from(score),
            },
            raw_score: f64::from(score),
            normalized_score: normalize(score, quality_min, quality_max),
            rrf_contribution: 0.0,
            weight: f64::from(quality_weight),
        });
    }
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

fn filter_lexical_hits(
    hits: Vec<ScoredResult>,
    filter: Option<&dyn SearchFilter>,
) -> Vec<ScoredResult> {
    let Some(filter) = filter else {
        return hits;
    };
    hits.into_iter()
        .filter(|hit| filter.matches(&hit.doc_id, hit.metadata.as_deref()))
        .collect()
}

fn fused_hits_to_scored_results(
    hits: Vec<FusedHit>,
    k: usize,
    config: &TwoTierConfig,
) -> Vec<ScoredResult> {
    // Take the `rrf_fuse` result by value and move each `doc_id` into the
    // `ScoredResult` instead of cloning it; the `FusedHit`s are a fresh
    // temporary here, so there is no need to keep them alive.
    hits.into_iter()
        .take(k)
        .map(|hit| {
            let score = saturating_f64_to_f32(hit.rrf_score);
            let explanation = config.explain.then(|| {
                let mut components = Vec::with_capacity(2);
                if let (Some(rank), Some(raw_score)) = (hit.lexical_rank, hit.lexical_score) {
                    components.push(ScoreComponent {
                        source: ExplainedSource::LexicalBm25 {
                            matched_terms: Vec::new(),
                            tf: 0.0,
                            idf: 0.0,
                        },
                        raw_score: f64::from(raw_score),
                        normalized_score: f64::from(raw_score),
                        rrf_contribution: rrf_rank_contribution(config.rrf_k, rank),
                        weight: 1.0,
                    });
                }
                if let (Some(rank), Some(raw_score)) = (hit.semantic_rank, hit.semantic_score) {
                    components.push(ScoreComponent {
                        source: ExplainedSource::SemanticFast {
                            embedder: "sync-fast-query".to_owned(),
                            cosine_sim: f64::from(raw_score),
                        },
                        raw_score: f64::from(raw_score),
                        normalized_score: f64::from(raw_score),
                        rrf_contribution: rrf_rank_contribution(config.rrf_k, rank),
                        weight: 1.0,
                    });
                }
                Box::new(HitExplanation {
                    final_score: f64::from(score),
                    components,
                    phase: ExplanationPhase::Initial,
                    rank_movement: None,
                })
            });
            ScoredResult {
                doc_id: hit.doc_id,
                score,
                source: ScoreSource::Hybrid,
                index: hit.semantic_index,
                fast_score: hit.semantic_score,
                quality_score: None,
                lexical_score: hit.lexical_score,
                rerank_score: None,
                explanation,
                metadata: None,
            }
        })
        .collect()
}

fn rrf_rank_contribution(rrf_k: f64, rank: usize) -> f64 {
    let rank = u32::try_from(rank).unwrap_or(u32::MAX);
    let rrf_k = if rrf_k.is_finite() && rrf_k >= 0.0 {
        rrf_k
    } else {
        60.0
    };
    1.0 / (rrf_k + f64::from(rank) + 1.0)
}

fn vector_hits_to_scored_results(
    hits: &[VectorHit],
    k: usize,
    source: ScoreSource,
    fast_scores: Option<&AHashMap<&str, f32>>,
    quality_scores: Option<&AHashMap<&str, f32>>,
    config: &TwoTierConfig,
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
                        source: ExplainedSource::SemanticFast {
                            embedder: "sync-fast-query".to_owned(),
                            cosine_sim: f64::from(hit.score),
                        },
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
    use frankensearch_core::generation::EmbeddingIdentityBundleV1;
    use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

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
        let mut artifact = EmbeddingIdentityBundleV1::explicit_test_model(model, dimension);
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
        // The query-side bundle of the SAME space: in-process f32 storage,
        // which can never be byte-equal to the persisted f16 one. That is why
        // the join is on the space, not the whole bundle.
        (
            tier,
            EmbeddingIdentityBundleV1::explicit_test_model(model, dimension),
        )
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn admitted_v2_default_product_preserves_exact_result_across_cache_states() {
        // The first record fixes the global int8 scale at one. The remaining
        // zero rows and the final 0.003 f16 row then all quantize to the same
        // int8 pass-1 score for query [0, 1]. The approximate candidate route
        // cannot retain the final exact winner. The default product scan must
        // return that winner both with an attached sidecar and with a hostile
        // unavailable cache: the sidecar may accelerate the exact route, but
        // cache state must not select a different algorithm. This is a route
        // proof by an observable result, not a private-field assertion.
        const CORPUS_ROWS: usize = 10_002;
        let dir = temp_dir("residual-default-product-route");
        let source_path = dir.join("source.fsvi");
        let cache_dir = dir.join("residual-cache");
        let unavailable_cache = dir.join("residual-cache-is-a-file");
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
                .write_record(&format!("zero-{row:05}"), &[0.0, 0.0])
                .expect("write zero candidate");
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

    fn temp_dir(label: &str) -> std::path::PathBuf {
        static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);
        for _ in 0..1024 {
            let nonce = TEMP_NONCE.fetch_add(1, AtomicOrdering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "frankensearch-sync-dbp10-{label}-{}-{}-{nonce}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            match std::fs::create_dir(&dir) {
                Ok(()) => return dir,
                // Falling out of the match is the next attempt: this arm is the
                // last statement in the loop body, so an explicit `continue`
                // says nothing the control flow does not already say.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create unique temp dir: {error}"),
            }
        }
        panic!("exhausted unique temp directory names")
    }

    fn make_index() -> Arc<InMemoryTwoTierIndex> {
        let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let fast_vectors = vec![vec![1.0, 0.0], vec![0.7, 0.3], vec![0.0, 1.0]];
        let quality_vectors = vec![vec![0.2, 0.8], vec![1.0, 0.0], vec![0.0, 1.0]];
        let fast = InMemoryVectorIndex::from_vectors(doc_ids.clone(), fast_vectors, 2).unwrap();
        let quality = InMemoryVectorIndex::from_vectors(doc_ids, quality_vectors, 2).unwrap();
        Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)))
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
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));
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
            &dir,
            "vector.fast.idx",
            "dbp10-fast",
            11,
            &[("a", &[1.0, 0.0]), ("b", &[0.0, 1.0])],
        );
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, None));
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

        // Same width, different space: refused, by the same contract field the
        // async searcher and the owner-backed activation use.
        let error = searcher
            .search_collect(&fast_only_query(vec![1.0, 0.0]), 2)
            .expect_err("a foreign embedding space must be refused");
        assert!(
            matches!(
                error,
                SearchError::InvalidConfig { ref field, .. }
                    if field == "query_embedding.fast.space_identity"
            ),
            "got {error:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// bd-ctzo C2 on the sync path: with an ATTESTED quality tier the refined
    /// phase retrieves independently, so a document the fast tier does not
    /// contain at all becomes reachable.
    #[test]
    fn an_attested_quality_tier_reaches_a_document_outside_the_fast_pool() {
        let dir = temp_dir("attested-union");
        let (fast, fast_identity) = attested_tier(
            &dir,
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
            &dir,
            "vector.quality.idx",
            "dbp10-union-quality",
            21,
            &[
                ("doc-near", &[0.0, 0.0, 1.0]),
                ("doc-far", &[0.0, 0.5, 0.0]),
                ("doc-quality-only", &[0.0, 1.0, 0.0]),
            ],
        );
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));
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

        let _ = std::fs::remove_dir_all(&dir);
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
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, None));
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
        let (results, metrics) = searcher.search_collect(&tiered(vec![1.0, 0.0]), 3).unwrap();
        assert!(results.is_empty());
        assert_eq!(
            metrics.zero_signal,
            Some(ZeroSignalReason::NewlyCreatedEmpty),
            "an empty semantic lane must say why, not just return nothing"
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
    fn zero_norm_query_over_empty_index_prefers_request_scoped_reason() {
        // Request-scoped pre-scan classification outranks index state:
        // a zero-norm query is the caller's defect even on an empty index.
        let fast = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 2).unwrap();
        let index = Arc::new(InMemoryTwoTierIndex::new(fast, None));
        let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());
        let (results, metrics) = searcher
            .search_collect(&fast_only_query(vec![0.0, 0.0]), 3)
            .unwrap();
        assert!(results.is_empty());
        assert_eq!(metrics.zero_signal, Some(ZeroSignalReason::ZeroNormQuery));
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
        assert!(
            results
                .iter()
                .all(|result| result.source == ScoreSource::Hybrid)
        );
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
