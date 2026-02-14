//! Progressive two-tier search orchestrator.
//!
//! [`TwoTierSearcher`] coordinates fast-tier and quality-tier search phases,
//! delivering results incrementally via an async callback. Consumers see
//! fast results in ~15ms, then optionally receive quality-refined results.
//!
//! # Callback Protocol
//!
//! The `on_phase` callback fires at most twice:
//! 1. [`SearchPhase::Initial`] — fast-tier results (always fired if search starts).
//! 2. [`SearchPhase::Refined`] or [`SearchPhase::RefinementFailed`] — quality tier
//!    result or graceful degradation (only fired when quality embedder is available
//!    and `fast_only` is false).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use asupersync::Cx;
use asupersync::time::{timeout, wall_now};
use tracing::instrument;

use frankensearch_core::ParsedQuery;
use frankensearch_core::canonicalize::{Canonicalizer, DefaultCanonicalizer};
use frankensearch_core::config::{TwoTierConfig, TwoTierMetrics};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::host_adapter::HostAdapter;
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{Embedder, LexicalSearch, ModelCategory, Reranker};
use frankensearch_core::types::{
    EmbeddingMetrics, PhaseMetrics, ScoreSource, ScoredResult, SearchMetrics, SearchMode,
    SearchPhase, VectorHit,
};
use frankensearch_core::{
    EmbedderTier, EmbeddingCollectorSample, EmbeddingStage, EmbeddingStatus, LiveSearchFrame,
    LiveSearchStreamEmitter, RuntimeMetricsCollector, SearchCollectorSample, SearchEventPhase,
    SearchStreamHealth, TelemetryCorrelation, TelemetryInstance,
};
use frankensearch_index::TwoTierIndex;

use crate::blend::{blend_two_tier, compute_rank_changes, kendall_tau};
use crate::rrf::{RrfConfig, candidate_count, rrf_fuse};

static TELEMETRY_EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Progressive two-tier search orchestrator.
///
/// Coordinates fast-tier embedding, optional lexical search, RRF fusion,
/// optional quality-tier refinement, and optional cross-encoder reranking.
///
/// # Usage
///
/// ```rust,ignore
/// let searcher = TwoTierSearcher::new(index, fast_embedder, config)
///     .with_quality_embedder(quality)
///     .with_lexical(tantivy)
///     .with_reranker(reranker);
///
/// let metrics = searcher.search(&cx, "distributed consensus", 10, text_fn, |phase| {
///     match phase {
///         SearchPhase::Initial { results, .. } => display(&results),
///         SearchPhase::Refined { results, .. } => update_display(&results),
///         SearchPhase::RefinementFailed { .. } => { /* keep initial */ }
///     }
/// }).await?;
/// ```
pub struct TwoTierSearcher {
    index: Arc<TwoTierIndex>,
    fast_embedder: Arc<dyn Embedder>,
    quality_embedder: Option<Arc<dyn Embedder>>,
    lexical: Option<Arc<dyn LexicalSearch>>,
    reranker: Option<Arc<dyn Reranker>>,
    host_adapter: Option<Arc<dyn HostAdapter>>,
    runtime_metrics_collector: Arc<RuntimeMetricsCollector>,
    live_search_stream_emitter: Arc<LiveSearchStreamEmitter>,
    canonicalizer: Box<dyn Canonicalizer>,
    config: TwoTierConfig,
}

impl TwoTierSearcher {
    /// Create a new searcher with a fast-tier embedder.
    #[must_use]
    pub fn new(
        index: Arc<TwoTierIndex>,
        fast_embedder: Arc<dyn Embedder>,
        config: TwoTierConfig,
    ) -> Self {
        Self {
            index,
            fast_embedder,
            quality_embedder: None,
            lexical: None,
            reranker: None,
            host_adapter: None,
            runtime_metrics_collector: Arc::new(RuntimeMetricsCollector::default()),
            live_search_stream_emitter: Arc::new(LiveSearchStreamEmitter::default()),
            canonicalizer: Box::new(DefaultCanonicalizer::default()),
            config,
        }
    }

    /// Set the quality-tier embedder for progressive refinement.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.quality_embedder = Some(embedder);
        self
    }

    /// Set the lexical search backend for hybrid RRF fusion.
    #[must_use]
    pub fn with_lexical(mut self, lexical: Arc<dyn LexicalSearch>) -> Self {
        self.lexical = Some(lexical);
        self
    }

    /// Set the cross-encoder reranker for Phase 2.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Set the host adapter used to receive canonical telemetry envelopes.
    #[must_use]
    pub fn with_host_adapter(mut self, host_adapter: Arc<dyn HostAdapter>) -> Self {
        self.host_adapter = Some(host_adapter);
        self
    }

    /// Override the runtime telemetry collector used for canonical envelope assembly.
    #[must_use]
    pub fn with_runtime_metrics_collector(
        mut self,
        collector: Arc<RuntimeMetricsCollector>,
    ) -> Self {
        self.runtime_metrics_collector = collector;
        self
    }

    /// Override the live-search stream emitter used for timeline/live-feed frames.
    #[must_use]
    pub fn with_live_search_stream_emitter(
        mut self,
        emitter: Arc<LiveSearchStreamEmitter>,
    ) -> Self {
        self.live_search_stream_emitter = emitter;
        self
    }

    /// Override the query canonicalizer.
    #[must_use]
    pub fn with_canonicalizer(mut self, canonicalizer: Box<dyn Canonicalizer>) -> Self {
        self.canonicalizer = canonicalizer;
        self
    }

    /// Snapshot live-search stream health counters.
    #[must_use]
    pub fn live_search_stream_health(&self) -> SearchStreamHealth {
        self.live_search_stream_emitter.health()
    }

    /// Drain buffered live-search frames from oldest to newest.
    #[must_use]
    pub fn drain_live_search_stream(&self, max_items: usize) -> Vec<LiveSearchFrame> {
        self.live_search_stream_emitter.drain(max_items)
    }

    /// Execute progressive search, calling `on_phase` as results become available.
    ///
    /// Fires [`SearchPhase::Initial`] first, then optionally
    /// [`SearchPhase::Refined`] or [`SearchPhase::RefinementFailed`].
    ///
    /// Returns collected metrics from all phases.
    ///
    /// # Parameters
    ///
    /// * `cx` — Capability context for cancellation.
    /// * `query` — The search query string.
    /// * `k` — Maximum number of results per phase.
    /// * `text_fn` — Retrieves document text by `doc_id` for reranking and
    ///   exclusion-query filtering.
    ///   Pass `|_| None` only when reranking is not needed and the query does
    ///   not contain exclusions (`-term`, `NOT "phrase"`).
    /// * `on_phase` — Callback invoked once per search phase.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::Cancelled` if the operation is cancelled via `cx`.
    /// Returns `SearchError::EmbeddingFailed` if fast embedding fails and no
    /// lexical backend is available as fallback.
    #[instrument(skip_all, fields(query_len = query.len(), k))]
    #[allow(clippy::too_many_lines)]
    pub async fn search(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
        mut on_phase: impl FnMut(SearchPhase) + Send,
    ) -> SearchResult<TwoTierMetrics> {
        let mut metrics = TwoTierMetrics::default();

        if query.is_empty() || k == 0 {
            return Ok(metrics);
        }

        // Canonicalize query.
        let canon_query = self.canonicalizer.canonicalize_query(query);
        if canon_query.trim().is_empty() {
            return Ok(metrics);
        }
        let parsed_query = ParsedQuery::parse(&canon_query);
        let semantic_query = if parsed_query.is_positive_empty() {
            canon_query.as_str()
        } else {
            parsed_query.positive.as_str()
        };

        tracing::debug!(
            included_terms = parsed_query.positive.split_whitespace().count(),
            excluded_terms = parsed_query.negation_count(),
            has_negations = parsed_query.has_negations(),
            "query_parsed"
        );

        let query_class = QueryClass::classify(semantic_query);
        metrics.query_class = Some(query_class);
        metrics.fast_embedder_id = Some(self.fast_embedder.id().to_owned());
        let telemetry_root_request_id = self
            .host_adapter
            .as_ref()
            .map(|_| next_telemetry_identifier("root"));
        let mut telemetry_initial_event_id: Option<String> = None;

        // Phase 1: Initial (fast tier).
        let phase1_start = Instant::now();
        let initial = self
            .run_phase1(
                cx,
                semantic_query,
                &parsed_query,
                k,
                query_class,
                &text_fn,
                &mut metrics,
                telemetry_root_request_id.as_deref(),
            )
            .await;
        metrics.phase1_total_ms = phase1_start.elapsed().as_secs_f64() * 1000.0;

        let initial_results = match initial {
            Ok(results) => results,
            Err(err) => {
                self.export_error(&err);
                return Err(err);
            }
        };

        let initial_hits = initial_results.clone();
        let phase1_has_fast_candidates = initial_hits
            .iter()
            .any(|result| result.fast_score.is_some());
        let initial_latency = phase1_start.elapsed();

        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
            telemetry_initial_event_id = self.emit_search_telemetry(
                semantic_query,
                query_class,
                SearchEventPhase::Initial,
                initial_hits.len(),
                metrics.lexical_candidates,
                metrics.semantic_candidates,
                initial_latency,
                root_request_id,
                None,
            );
        }

        on_phase(SearchPhase::Initial {
            results: initial_results,
            latency: initial_latency,
            metrics: PhaseMetrics {
                embedder_id: self.fast_embedder.id().to_owned(),
                vectors_searched: self.index.doc_count(),
                lexical_candidates: metrics.lexical_candidates,
                fused_count: initial_hits.len(),
            },
        });
        self.export_search_metrics(query_class, &metrics, initial_hits.len(), false);

        // Phase 2: Quality refinement (optional).
        if self.should_run_quality() && phase1_has_fast_candidates {
            let phase2_start = Instant::now();
            metrics.quality_embedder_id = self.quality_embedder.as_ref().map(|e| e.id().to_owned());

            let phase2_future = Box::pin(self.run_phase2(
                cx,
                semantic_query,
                k,
                &initial_hits,
                &text_fn,
                &mut metrics,
                telemetry_root_request_id.as_deref(),
                telemetry_initial_event_id.clone(),
            ));
            let timeout_budget = Duration::from_millis(self.config.quality_timeout_ms);
            let timeout_start = cx
                .timer_driver()
                .as_ref()
                .map_or_else(wall_now, asupersync::time::TimerDriverHandle::now);
            let phase2_result = timeout(timeout_start, timeout_budget, phase2_future).await;

            match phase2_result {
                Err(_elapsed) => {
                    let phase2_latency = phase2_start.elapsed();
                    metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                    let timeout_error = SearchError::SearchTimeout {
                        elapsed_ms: u64::try_from(phase2_latency.as_millis()).unwrap_or(u64::MAX),
                        budget_ms: self.config.quality_timeout_ms,
                    };
                    metrics.skip_reason = Some(timeout_error.to_string());
                    self.export_error(&timeout_error);
                    if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                        let _ = self.emit_search_telemetry(
                            semantic_query,
                            query_class,
                            SearchEventPhase::RefinementFailed,
                            initial_hits.len(),
                            metrics.lexical_candidates,
                            metrics.semantic_candidates,
                            phase2_latency,
                            root_request_id,
                            telemetry_initial_event_id.clone(),
                        );
                    }
                    on_phase(SearchPhase::RefinementFailed {
                        initial_results: initial_hits,
                        error: timeout_error,
                        latency: phase2_latency,
                    });
                }
                Ok(phase2_outcome) => match phase2_outcome {
                    Ok(refined_results) => {
                        let phase2_latency = phase2_start.elapsed();
                        let refined_count = refined_results.len();
                        metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            let _ = self.emit_search_telemetry(
                                semantic_query,
                                query_class,
                                SearchEventPhase::Refined,
                                refined_count,
                                metrics.lexical_candidates,
                                metrics.semantic_candidates,
                                phase2_latency,
                                root_request_id,
                                telemetry_initial_event_id.clone(),
                            );
                        }
                        self.export_search_metrics(query_class, &metrics, refined_count, true);
                        on_phase(SearchPhase::Refined {
                            results: refined_results,
                            latency: phase2_latency,
                            metrics: PhaseMetrics {
                                embedder_id: self
                                    .quality_embedder
                                    .as_ref()
                                    .map_or("none", |e| e.id())
                                    .to_owned(),
                                vectors_searched: self.index.doc_count(),
                                lexical_candidates: metrics.lexical_candidates,
                                fused_count: refined_count,
                            },
                            rank_changes: metrics.rank_changes.clone(),
                        });
                    }
                    Err(SearchError::Cancelled { phase, reason }) => {
                        return Err(SearchError::Cancelled { phase, reason });
                    }
                    Err(err) => {
                        let phase2_latency = phase2_start.elapsed();
                        metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                        metrics.skip_reason = Some(format!("{err}"));
                        self.export_error(&err);
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            let _ = self.emit_search_telemetry(
                                semantic_query,
                                query_class,
                                SearchEventPhase::RefinementFailed,
                                initial_hits.len(),
                                metrics.lexical_candidates,
                                metrics.semantic_candidates,
                                phase2_latency,
                                root_request_id,
                                telemetry_initial_event_id.clone(),
                            );
                        }
                        on_phase(SearchPhase::RefinementFailed {
                            initial_results: initial_hits,
                            error: err,
                            latency: phase2_latency,
                        });
                    }
                },
            }
        } else if self.should_run_quality() {
            metrics.skip_reason = Some("no_fast_phase_candidates".to_owned());
        } else if self.config.fast_only {
            metrics.skip_reason = Some("fast_only".to_owned());
        } else {
            metrics.skip_reason = Some("no_quality_embedder".to_owned());
        }

        Ok(metrics)
    }

    /// Convenience method that collects all phases and returns the best results.
    ///
    /// Returns the refined results if Phase 2 succeeds, otherwise the initial results.
    ///
    /// This method cannot evaluate exclusion clauses because it does not accept
    /// a document text provider. Use [`search_collect_with_text`](Self::search_collect_with_text)
    /// or [`search`](Self::search) when querying with exclusions.
    ///
    /// # Errors
    ///
    /// Same as [`search`](Self::search).
    pub async fn search_collect(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        let canon_query = self.canonicalizer.canonicalize_query(query);
        if ParsedQuery::parse(&canon_query).has_negations() {
            return Err(SearchError::QueryParseError {
                query: query.to_owned(),
                detail: "search_collect requires a text provider for exclusion syntax; use search_collect_with_text() or search()".to_owned(),
            });
        }

        self.search_collect_with_text(cx, query, k, |_| None).await
    }

    /// Convenience method that collects all phases while using `text_fn` for
    /// reranking and exclusion filtering.
    ///
    /// Returns the refined results if Phase 2 succeeds, otherwise the initial results.
    ///
    /// # Errors
    ///
    /// Same as [`search`](Self::search).
    pub async fn search_collect_with_text(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        text_fn: impl Fn(&str) -> Option<String> + Send + Sync,
    ) -> SearchResult<(Vec<ScoredResult>, TwoTierMetrics)> {
        let mut best_results = Vec::new();
        let metrics = self
            .search(cx, query, k, text_fn, |phase| match phase {
                SearchPhase::Initial { results, .. } | SearchPhase::Refined { results, .. } => {
                    best_results = results;
                }
                SearchPhase::RefinementFailed { .. } => {
                    // Keep the initial results already stored in best_results.
                }
            })
            .await?;
        Ok((best_results, metrics))
    }

    /// Run Phase 1: fast embedding + optional lexical + RRF fusion.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn run_phase1(
        &self,
        cx: &Cx,
        semantic_query: &str,
        parsed_query: &ParsedQuery,
        k: usize,
        query_class: QueryClass,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        metrics: &mut TwoTierMetrics,
        root_request_id: Option<&str>,
    ) -> SearchResult<Vec<ScoredResult>> {
        let base_candidates = candidate_count(k, 0, self.config.candidate_multiplier);

        // Adaptive budgets: identifiers lean lexical, NL leans semantic.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let semantic_budget =
            (base_candidates as f32 * query_class.semantic_budget_multiplier()) as usize;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let lexical_budget =
            (base_candidates as f32 * query_class.lexical_budget_multiplier()) as usize;

        let rrf_config = RrfConfig {
            k: self.config.rrf_k,
        };

        // Fast embedding.
        let embed_start = Instant::now();
        let fast_embed_result = self.fast_embedder.embed(cx, semantic_query).await;
        let fast_embed_elapsed = embed_start.elapsed();
        metrics.fast_embed_ms = fast_embed_elapsed.as_secs_f64() * 1000.0;

        // Lexical search (runs regardless of embedding success).
        let mut lexical_results = self
            .run_lexical(cx, semantic_query, lexical_budget, metrics)
            .await?;
        if parsed_query.has_negations() {
            lexical_results = lexical_results.map(|results| {
                let filtered =
                    filter_scored_results_by_negations(results, parsed_query, text_fn, "lexical");
                metrics.lexical_candidates = filtered.len();
                filtered
            });
        }

        match fast_embed_result {
            Ok(query_vec) => {
                self.export_embedding_metrics(
                    self.fast_embedder.as_ref(),
                    1,
                    metrics.fast_embed_ms,
                );
                if let Some(root_request_id) = root_request_id {
                    let _ = self.emit_embedding_telemetry(
                        self.fast_embedder.as_ref(),
                        EmbeddingStage::Fast,
                        EmbeddingStatus::Completed,
                        fast_embed_elapsed,
                        root_request_id,
                        None,
                    );
                }
                // Vector search.
                let search_start = Instant::now();
                let fast_hits = self.index.search_fast(&query_vec, semantic_budget)?;
                let fast_hits = if parsed_query.has_negations() {
                    filter_vector_hits_by_negations(fast_hits, parsed_query, text_fn, "semantic")
                } else {
                    fast_hits
                };
                metrics.vector_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;
                metrics.semantic_candidates = fast_hits.len();

                // RRF fusion if lexical results are available.
                let fuse_start = Instant::now();
                let results = lexical_results.as_ref().map_or_else(
                    || vector_hits_to_scored_results(&fast_hits, k),
                    |lexical| {
                        let fused = rrf_fuse(lexical, &fast_hits, k, 0, &rrf_config);
                        fused_hits_to_scored_results(&fused)
                    },
                );
                metrics.rrf_fusion_ms = fuse_start.elapsed().as_secs_f64() * 1000.0;

                Ok(results)
            }
            Err(embed_err) => {
                if let Some(root_request_id) = root_request_id {
                    let status = if matches!(&embed_err, SearchError::Cancelled { .. }) {
                        EmbeddingStatus::Cancelled
                    } else {
                        EmbeddingStatus::Failed
                    };
                    let _ = self.emit_embedding_telemetry(
                        self.fast_embedder.as_ref(),
                        EmbeddingStage::Fast,
                        status,
                        fast_embed_elapsed,
                        root_request_id,
                        None,
                    );
                }
                if matches!(embed_err, SearchError::Cancelled { .. }) {
                    return Err(embed_err);
                }
                // Graceful degradation: use lexical-only results if available.
                if let Some(ref lexical) = lexical_results {
                    self.export_error(&embed_err);
                    tracing::warn!(
                        error = %embed_err,
                        "fast embedding failed, falling back to lexical-only results"
                    );
                    Ok(lexical.iter().take(k).cloned().collect())
                } else {
                    Err(embed_err)
                }
            }
        }
    }

    /// Run Phase 2: quality embedding + blend + optional rerank.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    async fn run_phase2(
        &self,
        cx: &Cx,
        query: &str,
        k: usize,
        initial_results: &[ScoredResult],
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        metrics: &mut TwoTierMetrics,
        root_request_id: Option<&str>,
        parent_event_id: Option<String>,
    ) -> SearchResult<Vec<ScoredResult>> {
        let quality_embedder =
            self.quality_embedder
                .as_ref()
                .ok_or_else(|| SearchError::EmbedderUnavailable {
                    model: "quality".into(),
                    reason: "no quality embedder configured".into(),
                })?;

        // Quality embedding.
        let embed_start = Instant::now();
        let quality_vec = match quality_embedder.embed(cx, query).await {
            Ok(quality_vec) => {
                let quality_embed_elapsed = embed_start.elapsed();
                metrics.quality_embed_ms = quality_embed_elapsed.as_secs_f64() * 1000.0;
                self.export_embedding_metrics(
                    quality_embedder.as_ref(),
                    1,
                    metrics.quality_embed_ms,
                );
                if let Some(root_request_id) = root_request_id {
                    let _ = self.emit_embedding_telemetry(
                        quality_embedder.as_ref(),
                        EmbeddingStage::Quality,
                        EmbeddingStatus::Completed,
                        quality_embed_elapsed,
                        root_request_id,
                        parent_event_id.clone(),
                    );
                }
                quality_vec
            }
            Err(err) => {
                let quality_embed_elapsed = embed_start.elapsed();
                metrics.quality_embed_ms = quality_embed_elapsed.as_secs_f64() * 1000.0;
                if let Some(root_request_id) = root_request_id {
                    let status = if matches!(&err, SearchError::Cancelled { .. }) {
                        EmbeddingStatus::Cancelled
                    } else {
                        EmbeddingStatus::Failed
                    };
                    let _ = self.emit_embedding_telemetry(
                        quality_embedder.as_ref(),
                        EmbeddingStage::Quality,
                        status,
                        quality_embed_elapsed,
                        root_request_id,
                        parent_event_id,
                    );
                }
                return Err(err);
            }
        };

        // Get quality scores for top candidates from initial phase.
        let search_start = Instant::now();
        let fast_hits: Vec<VectorHit> = initial_results
            .iter()
            .enumerate()
            .map(|(i, r)| VectorHit {
                #[allow(clippy::cast_possible_truncation)]
                index: i as u32,
                // Preserve phase-1 fusion signal for lexical-only candidates.
                score: r.fast_score.unwrap_or(r.score),
                doc_id: r.doc_id.clone(),
            })
            .collect();

        // Look up indices in the fast index for quality scoring.
        let fast_indices: Vec<usize> = fast_hits
            .iter()
            .filter_map(|h| self.index.fast_index_for_doc_id(&h.doc_id))
            .collect();

        let quality_scores = self
            .index
            .quality_scores_for_indices(&quality_vec, &fast_indices)?;
        metrics.quality_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;

        // Build quality VectorHits for blending.
        let quality_hits: Vec<VectorHit> = fast_indices
            .iter()
            .zip(quality_scores.iter())
            .map(|(&idx, &score)| VectorHit {
                #[allow(clippy::cast_possible_truncation)]
                index: idx as u32,
                score,
                doc_id: self.index.doc_ids()[idx].clone(),
            })
            .collect();

        // Blend fast + quality scores.
        let blend_start = Instant::now();
        #[allow(clippy::cast_possible_truncation)]
        let blend_factor = self.config.quality_weight as f32;
        let blended = blend_two_tier(&fast_hits, &quality_hits, blend_factor);
        metrics.blend_ms = blend_start.elapsed().as_secs_f64() * 1000.0;

        // Compute rank changes (initial vs refined).
        let rank_changes = compute_rank_changes(&fast_hits, &blended);
        let tau = kendall_tau(&fast_hits, &blended);
        metrics.kendall_tau = tau;
        metrics.rank_changes = rank_changes;

        // Convert blended to scored results.
        #[allow(unused_mut)] // mut needed when `rerank` feature is enabled
        let mut results: Vec<ScoredResult> = blended
            .iter()
            .take(k)
            .map(|hit| {
                let fast_score = fast_hits
                    .iter()
                    .find(|h| h.doc_id == hit.doc_id)
                    .map(|h| h.score);
                let quality_score = quality_hits
                    .iter()
                    .find(|h| h.doc_id == hit.doc_id)
                    .map(|h| h.score);
                ScoredResult {
                    doc_id: hit.doc_id.clone(),
                    score: hit.score,
                    source: ScoreSource::SemanticQuality,
                    fast_score,
                    quality_score,
                    lexical_score: None,
                    rerank_score: None,
                    metadata: None,
                }
            })
            .collect();

        // Optional cross-encoder reranking.
        if let Some(ref reranker) = self.reranker {
            let rerank_start = Instant::now();
            #[cfg(feature = "rerank")]
            {
                if let Err(err) = frankensearch_rerank::pipeline::rerank_step(
                    cx,
                    reranker.as_ref(),
                    query,
                    &mut results,
                    text_fn,
                    k.min(100),
                    5,
                )
                .await
                {
                    self.export_error(&err);
                    return Err(err);
                }
            }
            #[cfg(not(feature = "rerank"))]
            {
                let _ = (reranker, text_fn);
                tracing::debug!("reranker configured but `rerank` feature not enabled");
            }
            metrics.rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
        }

        Ok(results)
    }

    /// Run optional lexical search, returning results or None.
    async fn run_lexical(
        &self,
        cx: &Cx,
        query: &str,
        candidates: usize,
        metrics: &mut TwoTierMetrics,
    ) -> SearchResult<Option<Vec<ScoredResult>>> {
        let Some(lexical) = self.lexical.as_ref() else {
            return Ok(None);
        };
        let start = Instant::now();
        match lexical.search(cx, query, candidates).await {
            Ok(results) => {
                metrics.lexical_search_ms = start.elapsed().as_secs_f64() * 1000.0;
                metrics.lexical_candidates = results.len();
                Ok(Some(results))
            }
            Err(err) => {
                metrics.lexical_search_ms = start.elapsed().as_secs_f64() * 1000.0;
                if matches!(err, SearchError::Cancelled { .. }) {
                    return Err(err);
                }
                self.export_error(&err);
                tracing::warn!(error = %err, "lexical search failed, continuing without");
                Ok(None)
            }
        }
    }

    fn export_search_metrics(
        &self,
        query_class: QueryClass,
        metrics: &TwoTierMetrics,
        result_count: usize,
        refined: bool,
    ) {
        let Some(exporter) = self.config.metrics_exporter.as_ref() else {
            return;
        };
        let payload = SearchMetrics {
            mode: SearchMode::TwoTier,
            query_class: Some(query_class),
            total_latency_ms: metrics.phase1_total_ms
                + if refined {
                    metrics.phase2_total_ms
                } else {
                    0.0
                },
            phase1_latency_ms: Some(metrics.phase1_total_ms),
            phase2_latency_ms: if refined {
                Some(metrics.phase2_total_ms)
            } else {
                None
            },
            result_count,
            lexical_candidates: metrics.lexical_candidates,
            semantic_candidates: metrics.semantic_candidates,
            refined,
        };
        exporter.on_search_completed(&payload);
    }

    fn export_embedding_metrics(
        &self,
        embedder: &dyn Embedder,
        batch_size: usize,
        duration_ms: f64,
    ) {
        let Some(exporter) = self.config.metrics_exporter.as_ref() else {
            return;
        };
        let payload = EmbeddingMetrics {
            embedder_id: embedder.id().to_owned(),
            batch_size,
            duration_ms,
            dimension: embedder.dimension(),
            is_semantic: embedder.is_semantic(),
        };
        exporter.on_embedding_completed(&payload);
    }

    fn export_error(&self, error: &SearchError) {
        if let Some(exporter) = self.config.metrics_exporter.as_ref() {
            exporter.on_error(error);
        }
    }

    /// Whether quality refinement should run.
    fn should_run_quality(&self) -> bool {
        !self.config.fast_only && self.quality_embedder.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_search_telemetry(
        &self,
        query_text: &str,
        query_class: QueryClass,
        phase: SearchEventPhase,
        result_count: usize,
        lexical_count: usize,
        semantic_count: usize,
        latency: Duration,
        root_request_id: &str,
        parent_event_id: Option<String>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };

        let envelope = self.runtime_metrics_collector.emit_search(
            telemetry_timestamp_now(),
            telemetry_instance,
            telemetry_correlation,
            SearchCollectorSample {
                query_text: query_text.to_owned(),
                query_class,
                phase,
                result_count,
                lexical_count,
                semantic_count,
                latency_us: u64::try_from(latency.as_micros()).unwrap_or(u64::MAX),
                memory_bytes: None,
            },
        );

        if let Err(err) = self
            .live_search_stream_emitter
            .publish_search(envelope.clone())
        {
            self.export_error(&err);
            tracing::warn!(error = %err, "live search stream publish failed");
        }

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_embedding_telemetry(
        &self,
        embedder: &dyn Embedder,
        stage: EmbeddingStage,
        status: EmbeddingStatus,
        duration: Duration,
        root_request_id: &str,
        parent_event_id: Option<String>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };

        let envelope = self.runtime_metrics_collector.emit_embedding(
            telemetry_timestamp_now(),
            telemetry_instance,
            telemetry_correlation,
            EmbeddingCollectorSample {
                job_id: format!("embed-{event_id}"),
                queue_depth: 0,
                doc_count: 1,
                stage,
                embedder_id: embedder.id().to_owned(),
                tier: embedder_tier_for_stage(stage, embedder.category()),
                dimension: embedder.dimension(),
                status,
                duration_ms: u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            },
        );

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }
}

// Implement Debug manually since trait objects don't derive Debug.
impl std::fmt::Debug for TwoTierSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoTierSearcher")
            .field("fast_embedder", &self.fast_embedder.id())
            .field(
                "quality_embedder",
                &self.quality_embedder.as_ref().map(|e| e.id()),
            )
            .field("has_lexical", &self.lexical.is_some())
            .field("has_reranker", &self.reranker.is_some())
            .field("has_host_adapter", &self.host_adapter.is_some())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn next_telemetry_identifier(prefix: &str) -> String {
    let sequence = TELEMETRY_EVENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{sequence:020}")
}

fn telemetry_timestamp_now() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    millis.to_string()
}

fn telemetry_instance_for_adapter(host_adapter: &dyn HostAdapter) -> TelemetryInstance {
    let identity = host_adapter.identity();
    let host_name = std::env::var("HOSTNAME")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown-host".to_owned());
    let instance_id = identity
        .instance_uuid
        .unwrap_or_else(|| format!("{}-{}", identity.adapter_id, std::process::id()));
    TelemetryInstance {
        instance_id,
        project_key: identity.host_project,
        host_name,
        pid: Some(std::process::id()),
    }
}

const fn embedder_tier_for_stage(stage: EmbeddingStage, category: ModelCategory) -> EmbedderTier {
    match stage {
        EmbeddingStage::Quality => EmbedderTier::Quality,
        EmbeddingStage::Fast | EmbeddingStage::Background => match category {
            ModelCategory::HashEmbedder => EmbedderTier::Hash,
            ModelCategory::StaticEmbedder => EmbedderTier::Fast,
            ModelCategory::TransformerEmbedder => EmbedderTier::Quality,
        },
    }
}

/// Convert `FusedHit` results to `ScoredResult`.
fn fused_hits_to_scored_results(
    fused: &[frankensearch_core::types::FusedHit],
) -> Vec<ScoredResult> {
    fused
        .iter()
        .map(|fh| {
            #[allow(clippy::cast_possible_truncation)]
            let score = fh.rrf_score as f32;
            ScoredResult {
                doc_id: fh.doc_id.clone(),
                score,
                source: if fh.in_both_sources {
                    ScoreSource::Hybrid
                } else if fh.lexical_rank.is_some() {
                    ScoreSource::Lexical
                } else {
                    ScoreSource::SemanticFast
                },
                fast_score: fh.semantic_score,
                quality_score: None,
                lexical_score: fh.lexical_score,
                rerank_score: None,
                metadata: None,
            }
        })
        .collect()
}

/// Convert `VectorHit` results to `ScoredResult` (semantic-only mode).
fn vector_hits_to_scored_results(hits: &[VectorHit], k: usize) -> Vec<ScoredResult> {
    hits.iter()
        .take(k)
        .map(|h| ScoredResult {
            doc_id: h.doc_id.clone(),
            score: h.score,
            source: ScoreSource::SemanticFast,
            fast_score: Some(h.score),
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            metadata: None,
        })
        .collect()
}

fn filter_scored_results_by_negations(
    results: Vec<ScoredResult>,
    parsed_query: &ParsedQuery,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> Vec<ScoredResult> {
    results
        .into_iter()
        .filter(|result| !should_exclude_document(&result.doc_id, parsed_query, text_fn, source))
        .collect()
}

fn filter_vector_hits_by_negations(
    hits: Vec<VectorHit>,
    parsed_query: &ParsedQuery,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> Vec<VectorHit> {
    hits.into_iter()
        .filter(|hit| !should_exclude_document(&hit.doc_id, parsed_query, text_fn, source))
        .collect()
}

fn should_exclude_document(
    doc_id: &str,
    parsed_query: &ParsedQuery,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> bool {
    let Some(text) = text_fn(doc_id) else {
        return false;
    };
    let Some(matched_clause) = find_negative_match(&text, parsed_query) else {
        return false;
    };
    tracing::debug!(
        %doc_id,
        matched_exclusion_term = %matched_clause,
        source,
        "doc_excluded"
    );
    true
}

fn find_negative_match(text: &str, parsed_query: &ParsedQuery) -> Option<String> {
    let lower_text = text.to_lowercase();
    for term in &parsed_query.negative_terms {
        if !term.is_empty() && lower_text.contains(&term.to_lowercase()) {
            return Some(term.clone());
        }
    }
    for phrase in &parsed_query.negative_phrases {
        if !phrase.is_empty() && lower_text.contains(&phrase.to_lowercase()) {
            return Some(phrase.clone());
        }
    }
    None
}

#[cfg(test)]
#[allow(
    clippy::unnecessary_literal_bound,
    clippy::cast_precision_loss,
    clippy::significant_drop_tightening
)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use frankensearch_core::traits::{MetricsExporter, ModelCategory, SearchFuture};
    use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};
    use frankensearch_core::{
        AdapterIdentity, AdapterLifecycleEvent, HostAdapter, TelemetryEnvelope, TelemetryEvent,
    };

    use super::*;

    // ─── Stub Embedder ──────────────────────────────────────────────────

    struct StubEmbedder {
        id: &'static str,
        dimension: usize,
    }

    impl StubEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self { id, dimension }
        }
    }

    impl Embedder for StubEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            let dim = self.dimension;
            Box::pin(async move {
                let mut vec = vec![0.0; dim];
                if !vec.is_empty() {
                    vec[0] = 1.0; // Simple deterministic embedding
                }
                Ok(vec)
            })
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    // ─── Failing Embedder ───────────────────────────────────────────────

    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async {
                Err(SearchError::EmbeddingFailed {
                    model: "failing-embedder".into(),
                    source: Box::new(std::io::Error::other("intentional test failure")),
                })
            })
        }

        fn dimension(&self) -> usize {
            4
        }

        fn id(&self) -> &str {
            "failing-embedder"
        }

        fn model_name(&self) -> &str {
            "failing-embedder"
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    struct CancelledEmbedder;

    impl Embedder for CancelledEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async {
                Err(SearchError::Cancelled {
                    phase: "embed".to_owned(),
                    reason: "test cancellation".to_owned(),
                })
            })
        }

        fn dimension(&self) -> usize {
            4
        }

        fn id(&self) -> &str {
            "cancelled-embedder"
        }

        fn model_name(&self) -> &str {
            "cancelled-embedder"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    struct PendingEmbedder {
        id: &'static str,
        dimension: usize,
    }

    impl PendingEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self { id, dimension }
        }
    }

    impl Embedder for PendingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(std::future::pending())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    // ─── Stub Lexical Search ────────────────────────────────────────────

    struct StubLexical;

    impl LexicalSearch for StubLexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                Ok((0..limit.min(3))
                    .map(|i| ScoredResult {
                        doc_id: format!("lex-doc-{i}"),
                        score: (3 - i) as f32,
                        source: ScoreSource::Lexical,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some((3 - i) as f32),
                        rerank_score: None,
                        metadata: None,
                    })
                    .collect())
            })
        }

        fn index_document<'a>(
            &'a self,
            _cx: &'a Cx,
            _doc: &'a frankensearch_core::types::IndexableDocument,
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn index_documents<'a>(
            &'a self,
            _cx: &'a Cx,
            _docs: &'a [frankensearch_core::types::IndexableDocument],
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn doc_count(&self) -> usize {
            3
        }
    }

    struct CancelledLexical;

    impl LexicalSearch for CancelledLexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            _limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async {
                Err(SearchError::Cancelled {
                    phase: "lexical_search".to_owned(),
                    reason: "test cancellation".to_owned(),
                })
            })
        }

        fn index_document<'a>(
            &'a self,
            _cx: &'a Cx,
            _doc: &'a frankensearch_core::types::IndexableDocument,
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn index_documents<'a>(
            &'a self,
            _cx: &'a Cx,
            _docs: &'a [frankensearch_core::types::IndexableDocument],
        ) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn commit<'a>(&'a self, _cx: &'a Cx) -> SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn doc_count(&self) -> usize {
            0
        }
    }

    #[derive(Debug)]
    struct RecordingHostAdapter {
        identity: AdapterIdentity,
        telemetry: Mutex<Vec<TelemetryEnvelope>>,
        lifecycle: Mutex<Vec<AdapterLifecycleEvent>>,
    }

    impl RecordingHostAdapter {
        fn new(host_project: &str) -> Self {
            Self {
                identity: AdapterIdentity {
                    adapter_id: "test-host-adapter".to_owned(),
                    adapter_version: "1.0.0".to_owned(),
                    host_project: host_project.to_owned(),
                    runtime_role: Some("query".to_owned()),
                    instance_uuid: Some("test-instance-uuid".to_owned()),
                    telemetry_schema_version: 1,
                    redaction_policy_version: "v1".to_owned(),
                },
                telemetry: Mutex::new(Vec::new()),
                lifecycle: Mutex::new(Vec::new()),
            }
        }

        fn telemetry_events(&self) -> Vec<TelemetryEnvelope> {
            self.telemetry.lock().expect("telemetry lock").clone()
        }
    }

    impl HostAdapter for RecordingHostAdapter {
        fn identity(&self) -> AdapterIdentity {
            self.identity.clone()
        }

        fn emit_telemetry(&self, envelope: &TelemetryEnvelope) -> SearchResult<()> {
            self.telemetry
                .lock()
                .expect("telemetry lock")
                .push(envelope.clone());
            Ok(())
        }

        fn on_lifecycle_event(&self, event: &AdapterLifecycleEvent) -> SearchResult<()> {
            self.lifecycle
                .lock()
                .expect("lifecycle lock")
                .push(event.clone());
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct RecordingExporter {
        search: Mutex<Vec<SearchMetrics>>,
        embedding: Mutex<Vec<EmbeddingMetrics>>,
        index: Mutex<Vec<IndexMetrics>>,
        errors: Mutex<Vec<String>>,
    }

    impl MetricsExporter for RecordingExporter {
        fn on_search_completed(&self, metrics: &SearchMetrics) {
            self.search
                .lock()
                .expect("search metrics lock")
                .push(metrics.clone());
        }

        fn on_embedding_completed(&self, metrics: &EmbeddingMetrics) {
            self.embedding
                .lock()
                .expect("embedding metrics lock")
                .push(metrics.clone());
        }

        fn on_index_updated(&self, metrics: &IndexMetrics) {
            self.index
                .lock()
                .expect("index metrics lock")
                .push(metrics.clone());
        }

        fn on_error(&self, error: &SearchError) {
            self.errors
                .lock()
                .expect("error metrics lock")
                .push(error.to_string());
        }
    }

    // ─── Test Helpers ───────────────────────────────────────────────────

    fn build_test_index(dimension: usize) -> Arc<TwoTierIndex> {
        let dir = std::env::temp_dir().join(format!(
            "frankensearch-searcher-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut builder =
            TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
        builder.set_fast_embedder_id("stub-fast");
        for i in 0..10 {
            let mut vec = vec![0.0; dimension];
            vec[i % dimension] = 1.0;
            builder
                .add_fast_record(format!("doc-{i}"), &vec)
                .expect("add record");
        }
        Arc::new(builder.finish().expect("finish index"))
    }

    // ─── Tests ──────────────────────────────────────────────────────────

    #[test]
    fn search_empty_query_returns_no_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phases = Vec::new();
            let metrics = searcher
                .search(&cx, "", 10, |_| None, |p| phases.push(format!("{p:?}")))
                .await
                .unwrap();

            assert!(phases.is_empty());
            assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
        });
    }

    #[test]
    fn search_zero_k_returns_no_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phases = Vec::new();
            searcher
                .search(&cx, "test", 0, |_| None, |p| phases.push(format!("{p:?}")))
                .await
                .unwrap();

            assert!(phases.is_empty());
        });
    }

    #[test]
    fn search_whitespace_query_returns_no_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phases = Vec::new();
            let metrics = searcher
                .search(
                    &cx,
                    "   \t\n  ",
                    10,
                    |_| None,
                    |p| phases.push(format!("{p:?}")),
                )
                .await
                .unwrap();

            assert!(phases.is_empty());
            assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
        });
    }

    #[test]
    fn search_fast_only_yields_initial_phase() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phase_count = 0;
            let mut got_initial = false;
            let metrics = searcher
                .search(
                    &cx,
                    "test query",
                    5,
                    |_| None,
                    |phase| {
                        phase_count += 1;
                        if matches!(phase, SearchPhase::Initial { .. }) {
                            got_initial = true;
                        }
                    },
                )
                .await
                .unwrap();

            assert_eq!(phase_count, 1);
            assert!(got_initial);
            assert!(metrics.phase1_total_ms > 0.0);
            assert!(
                metrics.skip_reason.is_some(),
                "should report skip reason for no quality embedder"
            );
        });
    }

    #[test]
    fn search_with_quality_yields_two_phases() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality);

            let mut phase_count = 0;
            let mut got_initial = false;
            let mut got_refined = false;
            let metrics = searcher
                .search(
                    &cx,
                    "test query",
                    5,
                    |_| None,
                    |phase| {
                        phase_count += 1;
                        match phase {
                            SearchPhase::Initial { .. } => got_initial = true,
                            SearchPhase::Refined { .. } => got_refined = true,
                            SearchPhase::RefinementFailed { .. } => {}
                        }
                    },
                )
                .await
                .unwrap();

            assert_eq!(phase_count, 2);
            assert!(got_initial);
            assert!(got_refined);
            assert!(metrics.quality_embed_ms > 0.0);
        });
    }

    #[test]
    fn refined_phase_metrics_report_actual_fused_count() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4); // 10 docs in fixture index
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality);

            let mut refined_fused_count = None;
            let mut refined_result_len = None;
            searcher
                .search(
                    &cx,
                    "test query",
                    20, // ask for more than available docs to exercise truncation
                    |_| None,
                    |phase| {
                        if let SearchPhase::Refined {
                            metrics, results, ..
                        } = phase
                        {
                            refined_fused_count = Some(metrics.fused_count);
                            refined_result_len = Some(results.len());
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert_eq!(
                refined_fused_count, refined_result_len,
                "refined phase should report fused_count equal to emitted result length"
            );
        });
    }

    #[test]
    fn fast_only_config_skips_quality() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));

            let config = TwoTierConfig {
                fast_only: true,
                ..TwoTierConfig::default()
            };

            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);

            let mut phase_count = 0;
            let metrics = searcher
                .search(&cx, "test", 5, |_| None, |_| phase_count += 1)
                .await
                .unwrap();

            assert_eq!(phase_count, 1, "fast_only should skip quality phase");
            assert_eq!(metrics.skip_reason.as_deref(), Some("fast_only"));
        });
    }

    #[test]
    fn quality_timeout_emits_refinement_failed_with_timeout_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(PendingEmbedder::new("quality-pending", 4));
            let config = TwoTierConfig {
                quality_timeout_ms: 0,
                ..TwoTierConfig::default()
            };
            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);

            let mut saw_initial = false;
            let mut saw_timeout = false;
            let metrics = searcher
                .search(
                    &cx,
                    "timeout me",
                    5,
                    |_| None,
                    |phase| match phase {
                        SearchPhase::Initial { .. } => saw_initial = true,
                        SearchPhase::RefinementFailed { error, .. } => {
                            saw_timeout = matches!(error, SearchError::SearchTimeout { .. });
                        }
                        SearchPhase::Refined { .. } => {}
                    },
                )
                .await
                .expect("search should return metrics even when refinement times out");

            assert!(saw_initial, "phase 1 should still run");
            assert!(
                saw_timeout,
                "phase 2 timeout should degrade to RefinementFailed with SearchTimeout"
            );
            assert!(
                metrics
                    .skip_reason
                    .as_ref()
                    .is_some_and(|reason| reason.contains("Search timed out")),
                "timeout should be recorded in skip reason"
            );
        });
    }

    #[test]
    fn fast_embed_failure_with_lexical_degrades_gracefully() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);

            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default())
                .with_lexical(lexical);

            let mut got_initial = false;
            let mut initial_count = 0;
            searcher
                .search(
                    &cx,
                    "test",
                    5,
                    |_| None,
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            got_initial = true;
                            initial_count = results.len();
                        }
                    },
                )
                .await
                .unwrap();

            assert!(got_initial, "should fall back to lexical-only results");
            assert!(initial_count > 0, "should have lexical results");
        });
    }

    #[test]
    fn fast_embed_failure_with_quality_configured_skips_refinement() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let quality: Arc<dyn Embedder> = Arc::new(StubEmbedder::new("quality", 4));
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality)
                .with_lexical(lexical);

            let mut phase_count = 0;
            let mut saw_initial = false;
            let mut saw_refined = false;
            let mut saw_refinement_failed = false;
            let metrics = searcher
                .search(
                    &cx,
                    "test",
                    5,
                    |_| None,
                    |phase| {
                        phase_count += 1;
                        match phase {
                            SearchPhase::Initial { .. } => saw_initial = true,
                            SearchPhase::Refined { .. } => saw_refined = true,
                            SearchPhase::RefinementFailed { .. } => saw_refinement_failed = true,
                        }
                    },
                )
                .await
                .expect("search should degrade gracefully");

            assert_eq!(phase_count, 1);
            assert!(saw_initial, "initial lexical fallback should be emitted");
            assert!(
                !saw_refined,
                "quality phase must be skipped without fast candidates"
            );
            assert!(
                !saw_refinement_failed,
                "skipping refinement should not emit refinement failure"
            );
            assert_eq!(
                metrics.skip_reason.as_deref(),
                Some("no_fast_phase_candidates")
            );
        });
    }

    #[test]
    fn fast_embed_failure_without_lexical_returns_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);

            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let result = searcher.search(&cx, "test", 5, |_| None, |_| {}).await;

            assert!(
                result.is_err(),
                "should propagate error without lexical fallback"
            );
        });
    }

    #[test]
    fn fast_embed_cancellation_propagates_even_with_lexical() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(CancelledEmbedder);
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default())
                .with_lexical(lexical);

            let err = searcher
                .search(&cx, "test", 5, |_| None, |_| {})
                .await
                .expect_err("cancelled embed should propagate");

            assert!(matches!(err, SearchError::Cancelled { .. }));
        });
    }

    #[test]
    fn lexical_cancellation_propagates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder = Arc::new(StubEmbedder::new("fast", 4));
            let lexical: Arc<dyn LexicalSearch> = Arc::new(CancelledLexical);
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default())
                .with_lexical(lexical);

            let err = searcher
                .search(&cx, "test", 5, |_| None, |_| {})
                .await
                .expect_err("cancelled lexical search should propagate");

            assert!(matches!(err, SearchError::Cancelled { .. }));
        });
    }

    #[test]
    fn search_collect_returns_best_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let (results, metrics) = searcher.search_collect(&cx, "test", 5).await.unwrap();

            assert!(!results.is_empty());
            assert!(metrics.phase1_total_ms > 0.0);
        });
    }

    #[test]
    fn search_collect_rejects_negations_without_text_provider() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let err = searcher
                .search_collect(&cx, "ownership -unsafe", 5)
                .await
                .expect_err("search_collect should reject exclusion queries without text provider");

            assert!(matches!(err, SearchError::QueryParseError { .. }));
            if let SearchError::QueryParseError { detail, .. } = err {
                assert!(detail.contains("text provider"));
            }
        });
    }

    #[test]
    fn search_collect_with_text_applies_negations() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let (results, _) = searcher
                .search_collect_with_text(&cx, "ownership -unsafe", 10, |doc_id| {
                    let text = if doc_id == "doc-0" {
                        "unsafe ownership example"
                    } else {
                        "safe ownership example"
                    };
                    Some(text.to_owned())
                })
                .await
                .expect("search should succeed");

            assert!(!results.is_empty());
            assert!(!results.iter().any(|r| r.doc_id == "doc-0"));
        });
    }

    #[test]
    fn exclusion_filters_semantic_results_case_insensitive() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let mut initial_results = Vec::new();
            searcher
                .search(
                    &cx,
                    "ownership -RuSt",
                    10,
                    |doc_id| {
                        let text = match doc_id {
                            "doc-0" => "Rust ownership and borrowing",
                            "doc-1" => "RUST lifetimes and traits",
                            _ => "safe memory patterns",
                        };
                        Some(text.to_owned())
                    },
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            initial_results = results;
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert_eq!(initial_results.len(), 8);
            assert!(!initial_results.iter().any(|r| r.doc_id == "doc-0"));
            assert!(!initial_results.iter().any(|r| r.doc_id == "doc-1"));
        });
    }

    #[test]
    fn exclusion_filters_lexical_and_semantic_candidates_before_fusion() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let searcher =
                TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_lexical(lexical);

            let mut initial_results = Vec::new();
            searcher
                .search(
                    &cx,
                    r#"query -unsafe NOT "danger zone""#,
                    10,
                    |doc_id| {
                        let text = match doc_id {
                            "doc-0" => "unsafe pointer dance",
                            "doc-1" => "all checks passed",
                            "lex-doc-0" => "contains danger zone marker",
                            "lex-doc-1" => "safe lexical candidate",
                            "lex-doc-2" => "UNSAFE lexical candidate",
                            _ => "safe semantic content",
                        };
                        Some(text.to_owned())
                    },
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            initial_results = results;
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert!(!initial_results.iter().any(|r| r.doc_id == "doc-0"));
            assert!(!initial_results.iter().any(|r| r.doc_id == "lex-doc-0"));
            assert!(!initial_results.iter().any(|r| r.doc_id == "lex-doc-2"));
            assert!(initial_results.iter().any(|r| r.doc_id == "lex-doc-1"));
        });
    }

    #[test]
    fn exclusion_can_eliminate_all_results_without_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let mut initial_results = Vec::new();
            let metrics = searcher
                .search(
                    &cx,
                    "-unsafe",
                    10,
                    |_doc_id| Some("unsafe across all docs".to_owned()),
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            initial_results = results;
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert!(initial_results.is_empty());
            assert!(metrics.phase1_total_ms > 0.0);
        });
    }

    #[test]
    fn refined_phase_preserves_fast_signal_for_lexical_only_candidates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality)
                .with_lexical(lexical);

            let mut refined_results = Vec::new();
            searcher
                .search(
                    &cx,
                    "query",
                    12,
                    |_| None,
                    |phase| {
                        if let SearchPhase::Refined { results, .. } = phase {
                            refined_results = results;
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert!(
                !refined_results.is_empty(),
                "refined phase should produce results"
            );
            let lexical_only = refined_results
                .iter()
                .find(|result| result.doc_id.starts_with("lex-doc-"))
                .expect("at least one lexical-only candidate should survive top-k");
            assert!(
                lexical_only.fast_score.is_some_and(|score| score > 0.0_f32),
                "lexical-only refined result should preserve positive phase-1 signal"
            );
        });
    }

    #[test]
    fn host_adapter_receives_initial_and_refined_search_events() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality)
                .with_host_adapter(adapter.clone());

            let _ = searcher
                .search(&cx, "test query", 5, |_| None, |_| {})
                .await
                .expect("search should succeed");

            let events = adapter.telemetry_events();
            let search_events: Vec<_> = events
                .iter()
                .filter_map(|event| match &event.event {
                    TelemetryEvent::Search {
                        correlation, query, ..
                    } => Some((correlation, query)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                search_events.len(),
                2,
                "expected initial + refined search telemetry"
            );

            let (initial_event_id, root_request_id) = {
                let (correlation, query) = search_events[0];
                assert_eq!(query.phase, SearchEventPhase::Initial);
                (
                    correlation.event_id.clone(),
                    correlation.root_request_id.clone(),
                )
            };
            assert!(
                !initial_event_id.is_empty(),
                "initial event id should be present"
            );
            assert!(
                !root_request_id.is_empty(),
                "root request id should be present"
            );

            let saw_refined_event = {
                let (correlation, query) = search_events[1];
                assert_eq!(query.phase, SearchEventPhase::Refined);
                assert_eq!(correlation.root_request_id, root_request_id);
                assert_eq!(
                    correlation.parent_event_id.as_deref(),
                    Some(initial_event_id.as_str())
                );
                true
            };
            assert!(saw_refined_event, "second event should be a search event");
        });
    }

    #[test]
    fn host_adapter_receives_refinement_failed_search_event() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(PendingEmbedder::new("quality-pending", 4));
            let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));
            let config = TwoTierConfig {
                quality_timeout_ms: 0,
                ..TwoTierConfig::default()
            };

            let searcher = TwoTierSearcher::new(index, fast, config)
                .with_quality_embedder(quality)
                .with_host_adapter(adapter.clone());

            let _ = searcher
                .search(&cx, "timeout query", 5, |_| None, |_| {})
                .await
                .expect("search should degrade on timeout without failing");

            let events = adapter.telemetry_events();
            let search_events: Vec<_> = events
                .iter()
                .filter_map(|event| match &event.event {
                    TelemetryEvent::Search { query, results, .. } => Some((query, results)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                search_events.len(),
                2,
                "expected initial + refinement_failed search events"
            );

            let initial_result_count = {
                let (query, results) = search_events[0];
                assert_eq!(query.phase, SearchEventPhase::Initial);
                Some(results.result_count)
            };
            assert!(
                initial_result_count.is_some(),
                "first event should be a search event"
            );
            let initial_result_count = initial_result_count.unwrap_or_default();

            let saw_refinement_failed = {
                let (query, results) = search_events[1];
                assert_eq!(query.phase, SearchEventPhase::RefinementFailed);
                assert_eq!(results.result_count, initial_result_count);
                true
            };
            assert!(
                saw_refinement_failed,
                "second event should be refinement_failed"
            );
        });
    }

    #[test]
    fn host_adapter_receives_fast_and_quality_embedding_events() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality)
                .with_host_adapter(adapter.clone());

            let _ = searcher
                .search(&cx, "embed telemetry", 5, |_| None, |_| {})
                .await
                .expect("search should succeed");

            let events = adapter.telemetry_events();
            let mut root_request_id = String::new();
            for event in &events {
                if let TelemetryEvent::Search { correlation, .. } = &event.event {
                    root_request_id = correlation.root_request_id.clone();
                    break;
                }
            }
            assert!(
                !root_request_id.is_empty(),
                "search events should provide root request id"
            );

            let embedding_events: Vec<_> = events
                .iter()
                .filter_map(|event| match &event.event {
                    TelemetryEvent::Embedding {
                        correlation,
                        job,
                        embedder,
                        status,
                        ..
                    } => Some((correlation, job, embedder, status)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                embedding_events.len(),
                2,
                "expected fast + quality embedding events"
            );

            let fast_event = embedding_events
                .iter()
                .find(|(_, job, _, _)| job.stage == EmbeddingStage::Fast);
            assert!(fast_event.is_some(), "fast embedding event missing");
            let (fast_correlation, fast_job, fast_embedder, fast_status) =
                fast_event.expect("fast event should exist");
            assert_eq!(**fast_status, EmbeddingStatus::Completed);
            assert_eq!(fast_embedder.id, "fast");
            assert_eq!(fast_embedder.tier, EmbedderTier::Fast);
            assert_eq!(fast_job.doc_count, 1);
            assert_eq!(fast_job.queue_depth, 0);
            assert_eq!(fast_correlation.root_request_id, root_request_id);
            assert!(!fast_job.job_id.is_empty(), "fast job id should be present");

            let quality_event = embedding_events
                .iter()
                .find(|(_, job, _, _)| job.stage == EmbeddingStage::Quality);
            assert!(quality_event.is_some(), "quality embedding event missing");
            let (quality_correlation, quality_job, quality_embedder, quality_status) =
                quality_event.expect("quality event should exist");
            assert_eq!(**quality_status, EmbeddingStatus::Completed);
            assert_eq!(quality_embedder.id, "quality");
            assert_eq!(quality_embedder.tier, EmbedderTier::Quality);
            assert_eq!(quality_job.doc_count, 1);
            assert_eq!(quality_job.queue_depth, 0);
            assert_eq!(quality_correlation.root_request_id, root_request_id);
            assert!(
                !quality_job.job_id.is_empty(),
                "quality job id should be present"
            );

            let snapshot = searcher.runtime_metrics_collector.snapshot();
            assert_eq!(snapshot.embedding_events_emitted, 2);
        });
    }

    #[test]
    fn host_adapter_receives_failed_fast_embedding_event() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_lexical(lexical)
                .with_host_adapter(adapter.clone());

            let _ = searcher
                .search(&cx, "fallback query", 5, |_| None, |_| {})
                .await
                .expect("search should fall back to lexical-only results");

            let events = adapter.telemetry_events();
            let embedding_events: Vec<_> = events
                .iter()
                .filter_map(|event| match &event.event {
                    TelemetryEvent::Embedding {
                        job,
                        embedder,
                        status,
                        ..
                    } => Some((job, embedder, status)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                embedding_events.len(),
                1,
                "expected one failed fast embedding event"
            );

            let (job, embedder, status) = embedding_events[0];
            assert_eq!(job.stage, EmbeddingStage::Fast);
            assert_eq!(embedder.id, "failing-embedder");
            assert_eq!(embedder.tier, EmbedderTier::Hash);
            assert_eq!(*status, EmbeddingStatus::Failed);

            let snapshot = searcher.runtime_metrics_collector.snapshot();
            assert_eq!(snapshot.embedding_events_emitted, 1);
        });
    }

    #[test]
    fn live_search_stream_health_reflects_emitted_search_events() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_host_adapter(adapter);

            let _ = searcher
                .search(&cx, "stream health", 5, |_| None, |_| {})
                .await
                .expect("search should succeed");

            let health = searcher.live_search_stream_health();
            assert_eq!(health.emitted_total, 1);
            assert_eq!(health.buffered, 1);

            let drained = searcher.drain_live_search_stream(10);
            assert_eq!(drained.len(), 1);
            let drained_health = searcher.live_search_stream_health();
            assert_eq!(drained_health.buffered, 0);
        });
    }

    #[test]
    fn metrics_track_query_class() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let (_, metrics) = searcher
                .search_collect(&cx, "how does distributed consensus work", 5)
                .await
                .unwrap();

            assert!(metrics.query_class.is_some());
            assert!(metrics.fast_embedder_id.is_some());
        });
    }

    #[test]
    fn debug_impl_works() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        let debug_str = format!("{searcher:?}");
        assert!(debug_str.contains("TwoTierSearcher"));
        assert!(debug_str.contains("fast"));
    }

    #[test]
    fn metrics_exporter_receives_search_and_embedding_callbacks() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let exporter = Arc::new(RecordingExporter::default());
            let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
            let _ = searcher
                .search(&cx, "test query", 5, |_| None, |_| {})
                .await
                .unwrap();

            {
                let search_events = exporter.search.lock().expect("search lock");
                assert_eq!(search_events.len(), 2);
                assert!(search_events.iter().any(|m| !m.refined));
                assert!(search_events.iter().any(|m| m.refined));
            }
            {
                let embedding_events = exporter.embedding.lock().expect("embedding lock");
                assert!(embedding_events.len() >= 2);
            }
            {
                let errors = exporter.errors.lock().expect("errors lock");
                assert!(errors.is_empty());
            }
        });
    }

    #[test]
    fn metrics_exporter_receives_degradation_errors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let exporter = Arc::new(RecordingExporter::default());
            let config = TwoTierConfig::default().with_metrics_exporter(exporter.clone());

            let searcher = TwoTierSearcher::new(index, embedder, config).with_lexical(lexical);
            let _ = searcher
                .search(&cx, "test", 5, |_| None, |_| {})
                .await
                .unwrap();

            {
                let errors = exporter.errors.lock().expect("errors lock");
                assert!(!errors.is_empty());
            }
            {
                let search_events = exporter.search.lock().expect("search lock");
                assert_eq!(search_events.len(), 1);
                assert!(!search_events[0].refined);
            }
        });
    }
}
