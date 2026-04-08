//! Synchronous two-tier search orchestration for low-latency UIs.
//!
//! [`SyncTwoTierSearcher`] mirrors the progressive two-phase contract of
//! [`crate::searcher::TwoTierSearcher`] but operates on precomputed query
//! embeddings and fully in-memory indices.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{
    FusedHit, PhaseMetrics, RankChanges, ScoreSource, ScoredResult, SearchError, SearchPhase,
    SearchResult, TwoTierConfig, TwoTierMetrics, VectorHit,
};
use frankensearch_index::{InMemoryTwoTierIndex, SearchParams};

use crate::blend::{blend_two_tier, build_borrowed_rank_map, compute_rank_changes_with_maps};
use crate::rrf::{RrfConfig, candidate_count, rrf_fuse};

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

/// Progressive synchronous searcher backed by [`InMemoryTwoTierIndex`].
pub struct SyncTwoTierSearcher {
    index: Arc<InMemoryTwoTierIndex>,
    lexical: Option<Arc<dyn SyncLexicalSearch>>,
    search_params: Option<SearchParams>,
    config: TwoTierConfig,
}

impl SyncTwoTierSearcher {
    /// Create a sync searcher over an in-memory two-tier index.
    #[must_use]
    pub const fn new(index: Arc<InMemoryTwoTierIndex>, config: TwoTierConfig) -> Self {
        Self {
            index,
            lexical: None,
            search_params: None,
            config,
        }
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

    /// Execute a synchronous search and return the final result set + metrics.
    ///
    /// # Errors
    ///
    /// Returns dimension/filter errors from vector search and lexical backend
    /// failures (when lexical fusion is enabled).
    pub fn search_collect(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        self.search_collect_with_filter(query_vec, k, None)
    }

    /// Execute a synchronous search with an optional doc-level filter.
    ///
    /// # Errors
    ///
    /// Returns dimension/filter errors from vector search and lexical backend
    /// failures (when lexical fusion is enabled).
    pub fn search_collect_with_filter(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        let outcome = self.search_internal(query_vec, k, filter)?;
        Ok((outcome.final_results, outcome.metrics))
    }

    /// Execute a synchronous search and stream progressive phases via iterator.
    ///
    /// When phase-1 retrieval fails (for example dimension mismatch), this
    /// returns an iterator yielding a single `RefinementFailed` phase carrying
    /// an empty `initial_results` payload.
    #[must_use]
    pub fn search_iter(&self, query_vec: &[f32], k: usize) -> SyncSearchIterator {
        self.search_iter_with_filter(query_vec, k, None)
    }

    /// Execute a synchronous filtered search and stream progressive phases.
    #[must_use]
    pub fn search_iter_with_filter(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SyncSearchIterator {
        match self.search_internal(query_vec, k, filter) {
            Ok(outcome) => SyncSearchIterator::new(outcome.phases),
            Err(error) => SyncSearchIterator::from_error(error),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn search_internal(
        &self,
        query_vec: &[f32],
        k: usize,
        filter: Option<&dyn SearchFilter>,
    ) -> SearchResult<SyncSearchOutcome> {
        let mut metrics = TwoTierMetrics::default();
        let mut phases = Vec::with_capacity(2);
        let fetch = candidate_count(k, 0, self.config.candidate_multiplier.max(1)).max(k);

        let phase1_started = Instant::now();
        let fast_hits = self.search_fast_hits(query_vec, fetch, filter)?;
        metrics.phase1_vectors_searched = fast_hits.len();
        metrics.semantic_candidates = fast_hits.len();

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
        let initial_results = lexical_hits.as_ref().map_or_else(
            || vector_hits_to_scored_results(&fast_hits, k, ScoreSource::SemanticFast, None, None),
            |lexical| {
                fused_hits_to_scored_results(
                    &rrf_fuse(
                        lexical,
                        &fast_hits,
                        k,
                        0,
                        &RrfConfig {
                            k: self.config.rrf_k,
                        },
                    ),
                    k,
                )
            },
        );
        metrics.rrf_fusion_ms = ms(rrf_started.elapsed());

        let phase1_latency = phase1_started.elapsed();
        metrics.vector_search_ms = ms(phase1_latency);
        metrics.phase1_total_ms = ms(phase1_latency);
        metrics.fast_embed_ms = 0.0;

        phases.push(SearchPhase::Initial {
            results: initial_results.clone(),
            latency: phase1_latency,
            metrics: PhaseMetrics {
                embedder_id: "sync-fast-query".to_owned(),
                vectors_searched: fast_hits.len(),
                lexical_candidates: metrics.lexical_candidates,
                fused_count: initial_results.len(),
            },
        });

        if self.config.fast_only || !self.index.has_quality_index() {
            metrics.skip_reason = Some(if self.config.fast_only {
                "fast_only_enabled".to_owned()
            } else {
                "quality_index_unavailable".to_owned()
            });
            return Ok(SyncSearchOutcome {
                phases,
                final_results: initial_results,
                metrics,
            });
        }

        let phase2_started = Instant::now();
        let quality_scores = match self.index.quality_scores_for_hits(query_vec, &fast_hits) {
            Ok(scores) => scores,
            Err(error) => {
                let latency = phase2_started.elapsed();
                metrics.phase2_total_ms = ms(latency);
                metrics.skip_reason = Some(error.to_string());
                phases.push(SearchPhase::RefinementFailed {
                    initial_results: initial_results.clone(),
                    error,
                    latency,
                });
                return Ok(SyncSearchOutcome {
                    phases,
                    final_results: initial_results,
                    metrics,
                });
            }
        };

        metrics.phase2_vectors_searched = quality_scores.len();
        let blend_started = Instant::now();
        let quality_hits = fast_hits
            .iter()
            .zip(quality_scores.iter())
            .filter_map(|(fast, score)| {
                score.map(|s| VectorHit {
                    index: fast.index,
                    doc_id: fast.doc_id.clone(),
                    score: s,
                })
            })
            .collect::<Vec<_>>();
        let blended = blend_two_tier(
            &fast_hits,
            &quality_hits,
            saturating_f64_to_f32(self.config.quality_weight),
        );
        metrics.blend_ms = ms(blend_started.elapsed());
        metrics.quality_search_ms = ms(phase2_started.elapsed());
        metrics.quality_embed_ms = 0.0;

        let refined_results = lexical_hits.as_ref().map_or_else(
            || {
                let fast_scores = fast_hits
                    .iter()
                    .map(|hit| (hit.doc_id.clone(), hit.score))
                    .collect::<HashMap<_, _>>();
                let quality_scores = quality_hits
                    .iter()
                    .map(|hit| (hit.doc_id.clone(), hit.score))
                    .collect::<HashMap<_, _>>();
                vector_hits_to_scored_results(
                    &blended,
                    k,
                    ScoreSource::SemanticQuality,
                    Some(&fast_scores),
                    Some(&quality_scores),
                )
            },
            |lexical| {
                fused_hits_to_scored_results(
                    &rrf_fuse(
                        lexical,
                        &blended,
                        k,
                        0,
                        &RrfConfig {
                            k: self.config.rrf_k,
                        },
                    ),
                    k,
                )
            },
        );

        let rank_changes = compute_rank_changes_for_scored(&initial_results, &refined_results);
        metrics.rank_changes = rank_changes.clone();
        metrics.phase2_total_ms = ms(phase2_started.elapsed());
        metrics.kendall_tau = None;

        phases.push(SearchPhase::Refined {
            results: refined_results.clone(),
            latency: phase2_started.elapsed(),
            metrics: PhaseMetrics {
                embedder_id: "sync-quality-query".to_owned(),
                vectors_searched: quality_hits.len(),
                lexical_candidates: metrics.lexical_candidates,
                fused_count: refined_results.len(),
            },
            rank_changes,
        });

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
            || fast_index.search_top_k(query_vec, fetch, filter),
            |params| fast_index.search_top_k_with_params(query_vec, fetch, filter, params),
        )
    }
}

impl std::fmt::Debug for SyncTwoTierSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncTwoTierSearcher")
            .field("has_lexical", &self.lexical.is_some())
            .field("search_params", &self.search_params)
            .field("has_quality_index", &self.index.has_quality_index())
            .field("config", &self.config)
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

fn filter_lexical_hits(
    hits: Vec<ScoredResult>,
    filter: Option<&dyn SearchFilter>,
) -> Vec<ScoredResult> {
    let Some(filter) = filter else {
        return hits;
    };
    hits.into_iter()
        .filter(|hit| filter.matches(&hit.doc_id, hit.metadata.as_ref()))
        .collect()
}

fn fused_hits_to_scored_results(hits: &[FusedHit], k: usize) -> Vec<ScoredResult> {
    hits.iter()
        .take(k)
        .map(|hit| ScoredResult {
            doc_id: hit.doc_id.clone(),
            score: saturating_f64_to_f32(hit.rrf_score),
            source: ScoreSource::Hybrid,
            index: hit.semantic_index,
            fast_score: hit.semantic_score,
            quality_score: None,
            lexical_score: hit.lexical_score,
            rerank_score: None,
            explanation: None,
            metadata: None,
        })
        .collect()
}

fn vector_hits_to_scored_results(
    hits: &[VectorHit],
    k: usize,
    source: ScoreSource,
    fast_scores: Option<&HashMap<String, f32>>,
    quality_scores: Option<&HashMap<String, f32>>,
) -> Vec<ScoredResult> {
    let mut seen = HashSet::new();
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
            ScoredResult {
                doc_id: hit.doc_id.clone(),
                score: hit.score,
                source,
                index: Some(hit.index),
                fast_score,
                quality_score,
                lexical_score: None,
                rerank_score: None,
                explanation: None,
                metadata: None,
            }
        })
        .collect()
}

fn compute_rank_changes_for_scored(
    initial: &[ScoredResult],
    refined: &[ScoredResult],
) -> RankChanges {
    let initial_hits = initial
        .iter()
        .enumerate()
        .map(|(idx, hit)| VectorHit {
            index: hit
                .index
                .unwrap_or_else(|| u32::try_from(idx).unwrap_or(u32::MAX)),
            score: hit.score,
            doc_id: hit.doc_id.clone(),
        })
        .collect::<Vec<_>>();
    let refined_hits = refined
        .iter()
        .enumerate()
        .map(|(idx, hit)| VectorHit {
            index: hit
                .index
                .unwrap_or_else(|| u32::try_from(idx).unwrap_or(u32::MAX)),
            score: hit.score,
            doc_id: hit.doc_id.clone(),
        })
        .collect::<Vec<_>>();
    let initial_map = build_borrowed_rank_map(&initial_hits);
    let refined_map = build_borrowed_rank_map(&refined_hits);
    compute_rank_changes_with_maps(&initial_map, &refined_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensearch_core::ScoreSource;
    use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};

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
            doc_id: doc_id.to_owned(),
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

    #[test]
    fn search_collect_returns_refined_results() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, metrics) = searcher.search_collect(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].source, ScoreSource::SemanticQuality);
        assert!(metrics.phase1_total_ms >= 0.0);
        assert!(metrics.phase2_total_ms >= 0.0);
    }

    #[test]
    fn search_iter_yields_initial_then_refined() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let phases = searcher.search_iter(&[1.0, 0.0], 2).collect::<Vec<_>>();
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
        let phases = searcher.search_iter(&[1.0, 0.0], 2).collect::<Vec<_>>();
        assert_eq!(phases.len(), 1);
        assert!(matches!(phases[0], SearchPhase::Initial { .. }));
    }

    #[test]
    fn filter_is_applied_to_fast_and_refined_results() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect_with_filter(&[1.0, 0.0], 3, Some(&ExcludeB))
            .unwrap();
        assert!(results.iter().all(|result| result.doc_id != "b"));
    }

    #[test]
    fn empty_query_returns_dimension_mismatch() {
        let searcher = SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default());
        let err = searcher.search_collect(&[], 3).unwrap_err();
        assert!(matches!(err, SearchError::DimensionMismatch { .. }));
    }

    #[test]
    fn lexical_fusion_can_introduce_lexical_only_hits() {
        let lexical = Arc::new(StaticLexical {
            hits: vec![lexical_result("lex-only", 10.0), lexical_result("a", 9.0)],
        });
        let searcher =
            SyncTwoTierSearcher::new(make_index(), TwoTierConfig::default()).with_lexical(lexical);
        let (results, _) = searcher.search_collect(&[1.0, 0.0], 3).unwrap();
        assert!(results.iter().any(|result| result.doc_id == "lex-only"));
        assert!(
            results
                .iter()
                .all(|result| result.source == ScoreSource::Hybrid)
        );
    }
}
