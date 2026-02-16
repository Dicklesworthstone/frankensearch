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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::time::{timeout, wall_now};
use tracing::instrument;
use unicode_normalization::UnicodeNormalization;

use time::{OffsetDateTime, format_description::well_known::Rfc3339};

#[cfg(feature = "graph")]
use frankensearch_core::DocumentGraph;
use frankensearch_core::ParsedQuery;
use frankensearch_core::canonicalize::{Canonicalizer, DefaultCanonicalizer};
use frankensearch_core::config::{TwoTierConfig, TwoTierMetrics};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::explanation::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
};
use frankensearch_core::host_adapter::{AdapterLifecycleEvent, HostAdapter};
use frankensearch_core::query_class::QueryClass;
use frankensearch_core::traits::{Embedder, LexicalSearch, ModelCategory, Reranker};
use frankensearch_core::types::{
    EmbeddingMetrics, PhaseMetrics, ScoreSource, ScoredResult, SearchMetrics, SearchMode,
    SearchPhase, VectorHit,
};
use frankensearch_core::{
    EmbedderTier, EmbeddingCollectorSample, EmbeddingStage, EmbeddingStatus, LifecycleSeverity,
    LifecycleState, LiveSearchFrame, LiveSearchStreamEmitter, ResourceCollectorSample,
    RuntimeMetricsCollector, SearchCollectorSample, SearchEventPhase, SearchStreamHealth,
    TelemetryCorrelation, TelemetryEnvelope, TelemetryEvent, TelemetryInstance,
};
use frankensearch_embed::CachedEmbedder;
use frankensearch_index::TwoTierIndex;

use crate::blend::{
    blend_two_tier, build_borrowed_rank_map, compute_rank_changes_with_maps,
    kendall_tau_with_refined_rank,
};
#[cfg(feature = "graph")]
use crate::graph_rank::GraphRanker;
use crate::rrf::{RrfConfig, candidate_count, rrf_fuse, rrf_fuse_with_graph};

static TELEMETRY_EVENT_COUNTER: AtomicU64 = AtomicU64::new(1);

struct NormalizedExclusions {
    terms: Vec<String>,
    phrases: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct CpuJiffiesSnapshot {
    process_jiffies: u64,
    total_jiffies: u64,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn scaled_budget(base_candidates: usize, multiplier: f32) -> usize {
    if base_candidates == 0 || !multiplier.is_finite() || multiplier <= 0.0 {
        return 0;
    }
    let scaled = (base_candidates as f32 * multiplier).ceil() as usize;
    scaled.max(1)
}

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
    #[cfg(feature = "graph")]
    document_graph: Option<Arc<DocumentGraph>>,
    /// When set, `with_quality_embedder` auto-wraps with `CachedEmbedder`.
    embedding_cache_capacity: Option<usize>,
    resource_cpu_state: Mutex<Option<CpuJiffiesSnapshot>>,
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
            #[cfg(feature = "graph")]
            document_graph: None,
            embedding_cache_capacity: None,
            resource_cpu_state: Mutex::new(None),
        }
    }

    /// Set the quality-tier embedder for progressive refinement.
    ///
    /// If `with_embedding_cache` was called first, the quality embedder is
    /// automatically wrapped with a `CachedEmbedder` at the same capacity.
    #[must_use]
    pub fn with_quality_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        if let Some(cap) = self.embedding_cache_capacity {
            self.quality_embedder = Some(Arc::new(CachedEmbedder::new(embedder, cap)));
        } else {
            self.quality_embedder = Some(embedder);
        }
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

    /// Attach an optional document graph used for phase-1 graph ranking.
    #[cfg(feature = "graph")]
    #[must_use]
    pub fn with_document_graph(mut self, graph: Arc<DocumentGraph>) -> Self {
        self.document_graph = Some(graph);
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

    /// Wrap the fast (and quality, if set) embedders with a query embedding cache.
    ///
    /// Repeated queries will return cached vectors instead of re-running inference.
    /// `capacity` controls the maximum number of cached embeddings per embedder
    /// (FIFO eviction when full).
    ///
    /// Safe to call in any builder order: if `with_quality_embedder` is called
    /// later, the quality embedder is automatically wrapped at the same capacity.
    #[must_use]
    pub fn with_embedding_cache(mut self, capacity: usize) -> Self {
        self.embedding_cache_capacity = Some(capacity);
        self.fast_embedder = Arc::new(CachedEmbedder::new(self.fast_embedder, capacity));
        if let Some(qe) = self.quality_embedder.take() {
            self.quality_embedder = Some(Arc::new(CachedEmbedder::new(qe, capacity)));
        }
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
        let normalized_exclusions = if parsed_query.has_negations() {
            Some(NormalizedExclusions {
                terms: parsed_query
                    .negative_terms
                    .iter()
                    .map(|t| normalize_for_negation_match(t))
                    .collect(),
                phrases: parsed_query
                    .negative_phrases
                    .iter()
                    .map(|p| normalize_for_negation_match(p))
                    .collect(),
            })
        } else {
            None
        };
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
        let mut telemetry_last_event_id: Option<String> = None;
        let search_started_at = Instant::now();

        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
            self.emit_host_lifecycle_hook(&AdapterLifecycleEvent::SessionStart {
                ts: telemetry_timestamp_now(),
            });
            telemetry_last_event_id = self.emit_lifecycle_telemetry(
                root_request_id,
                None,
                LifecycleState::Started,
                LifecycleSeverity::Info,
                None,
                Some(0),
            );
        }

        // Phase 1: Initial (fast tier).
        let phase1_start = Instant::now();
        let initial = self
            .run_phase1(
                cx,
                semantic_query,
                &parsed_query,
                normalized_exclusions.as_ref(),
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
                if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                    self.emit_session_stop_telemetry(
                        root_request_id,
                        telemetry_last_event_id.clone(),
                        LifecycleState::Degraded,
                        LifecycleSeverity::Error,
                        Some(err.to_string()),
                        search_started_at.elapsed(),
                    );
                }
                return Err(err);
            }
        };

        let initial_hits = initial_results.clone();
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
            if telemetry_initial_event_id.is_some() {
                telemetry_last_event_id = telemetry_initial_event_id.clone();
            }
            self.emit_phase_health_telemetry(
                root_request_id,
                telemetry_initial_event_id
                    .clone()
                    .or_else(|| telemetry_last_event_id.clone()),
                search_started_at.elapsed(),
            );
        }

        on_phase(SearchPhase::Initial {
            results: initial_results,
            latency: initial_latency,
            metrics: PhaseMetrics {
                embedder_id: self.fast_embedder.id().to_owned(),
                vectors_searched: metrics.phase1_vectors_searched,
                lexical_candidates: metrics.lexical_candidates,
                fused_count: initial_hits.len(),
            },
        });
        self.export_search_metrics(query_class, &metrics, initial_hits.len(), false);

        // Phase 2: Quality refinement (optional).
        // Runs even if Phase 1 was lexical-only (fast_score is None), effectively
        // performing a "lexical -> quality rerank" flow.  Skipped when fast
        // embedding failed entirely (phase1_vectors_searched == 0) because the
        // embedding pipeline is degraded and quality refinement would be unreliable.
        if self.should_run_quality()
            && !initial_hits.is_empty()
            && metrics.phase1_vectors_searched > 0
        {
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
                        let refinement_failed_event_id = self.emit_search_telemetry(
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
                        if let Some(event_id) = &refinement_failed_event_id {
                            telemetry_last_event_id = Some(event_id.clone());
                        }
                        self.emit_phase_health_telemetry(
                            root_request_id,
                            refinement_failed_event_id.or_else(|| telemetry_last_event_id.clone()),
                            search_started_at.elapsed(),
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
                            let refined_event_id = self.emit_search_telemetry(
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
                            if let Some(event_id) = &refined_event_id {
                                telemetry_last_event_id = Some(event_id.clone());
                            }
                            self.emit_phase_health_telemetry(
                                root_request_id,
                                refined_event_id.or_else(|| telemetry_last_event_id.clone()),
                                search_started_at.elapsed(),
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
                                vectors_searched: metrics.phase2_vectors_searched,
                                lexical_candidates: metrics.lexical_candidates,
                                fused_count: refined_count,
                            },
                            rank_changes: metrics.rank_changes.clone(),
                        });
                    }
                    Err(SearchError::Cancelled { phase, reason }) => {
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            self.emit_session_stop_telemetry(
                                root_request_id,
                                telemetry_last_event_id.clone(),
                                LifecycleState::Degraded,
                                LifecycleSeverity::Warn,
                                Some(format!("cancelled:{phase}:{reason}")),
                                search_started_at.elapsed(),
                            );
                        }
                        return Err(SearchError::Cancelled { phase, reason });
                    }
                    Err(err) => {
                        let phase2_latency = phase2_start.elapsed();
                        metrics.phase2_total_ms = phase2_latency.as_secs_f64() * 1000.0;
                        metrics.skip_reason = Some(format!("{err}"));
                        self.export_error(&err);
                        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
                            let refinement_failed_event_id = self.emit_search_telemetry(
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
                            if let Some(event_id) = &refinement_failed_event_id {
                                telemetry_last_event_id = Some(event_id.clone());
                            }
                            self.emit_phase_health_telemetry(
                                root_request_id,
                                refinement_failed_event_id
                                    .or_else(|| telemetry_last_event_id.clone()),
                                search_started_at.elapsed(),
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
            if initial_hits.is_empty() {
                metrics.skip_reason = Some("no_fast_phase_candidates".to_owned());
            } else {
                metrics.skip_reason = Some("vector_index_unavailable".to_owned());
            }
        } else if self.config.fast_only {
            metrics.skip_reason = Some("fast_only".to_owned());
        } else {
            metrics.skip_reason = Some("no_quality_embedder".to_owned());
        }

        if let Some(root_request_id) = telemetry_root_request_id.as_deref() {
            self.emit_session_stop_telemetry(
                root_request_id,
                telemetry_last_event_id,
                LifecycleState::Stopped,
                LifecycleSeverity::Info,
                metrics.skip_reason.clone(),
                search_started_at.elapsed(),
            );
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
        _parsed_query: &ParsedQuery,
        normalized_exclusions: Option<&NormalizedExclusions>,
        k: usize,
        query_class: QueryClass,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        metrics: &mut TwoTierMetrics,
        root_request_id: Option<&str>,
    ) -> SearchResult<Vec<ScoredResult>> {
        let base_candidates = candidate_count(k, 0, self.config.candidate_multiplier);

        // Adaptive budgets: identifiers lean lexical, NL leans semantic.
        let semantic_budget =
            scaled_budget(base_candidates, query_class.semantic_budget_multiplier());
        let lexical_budget =
            scaled_budget(base_candidates, query_class.lexical_budget_multiplier());

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
        if let Some(exclusions) = normalized_exclusions {
            lexical_results = lexical_results.map(|results| {
                let filtered =
                    filter_scored_results_by_negations(results, exclusions, text_fn, "lexical");
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
                let fast_hits = if let Some(exclusions) = normalized_exclusions {
                    filter_vector_hits_by_negations(fast_hits, exclusions, text_fn, "semantic")
                } else {
                    fast_hits
                };
                metrics.vector_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;
                metrics.semantic_candidates = fast_hits.len();
                metrics.phase1_vectors_searched = self.index.doc_count();

                let graph_candidates: Option<Vec<ScoredResult>> = if self
                    .config
                    .graph_ranking_enabled
                    && self.config.graph_ranking_weight > 0.0
                {
                    #[cfg(feature = "graph")]
                    {
                        self.document_graph.as_ref().and_then(|graph| {
                            GraphRanker::new().rank_phase1(
                                cx,
                                semantic_query,
                                graph.as_ref(),
                                &fast_hits,
                                semantic_budget,
                            )
                        })
                    }
                    #[cfg(not(feature = "graph"))]
                    {
                        None
                    }
                } else {
                    None
                };
                let graph_candidates = graph_candidates.map(|results| {
                    if let Some(exclusions) = normalized_exclusions {
                        filter_scored_results_by_negations(results, exclusions, text_fn, "graph")
                    } else {
                        results
                    }
                });

                // RRF fusion if lexical results are available.
                let fuse_start = Instant::now();
                let results = lexical_results.as_ref().map_or_else(
                    || {
                        graph_candidates.as_ref().map_or_else(
                            || {
                                vector_hits_to_scored_results(
                                    &fast_hits,
                                    k,
                                    &self.config,
                                    self.fast_embedder.id(),
                                )
                            },
                            |graph| {
                                let fused = rrf_fuse_with_graph(
                                    &[],
                                    &fast_hits,
                                    graph,
                                    self.config.graph_ranking_weight,
                                    k,
                                    0,
                                    &rrf_config,
                                );
                                fused_hits_to_scored_results(
                                    &fused,
                                    &[],
                                    self.config.explain,
                                    self.fast_embedder.id(),
                                    self.config.rrf_k,
                                )
                            },
                        )
                    },
                    |lexical| {
                        let fused = graph_candidates.as_ref().map_or_else(
                            || rrf_fuse(lexical, &fast_hits, k, 0, &rrf_config),
                            |graph| {
                                rrf_fuse_with_graph(
                                    lexical,
                                    &fast_hits,
                                    graph,
                                    self.config.graph_ranking_weight,
                                    k,
                                    0,
                                    &rrf_config,
                                )
                            },
                        );
                        fused_hits_to_scored_results(
                            &fused,
                            lexical,
                            self.config.explain,
                            self.fast_embedder.id(),
                            self.config.rrf_k,
                        )
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
                // Keep missing semantic-fast source at 0.0 so blending semantics
                // remain consistent with blend_two_tier contract.
                score: r.fast_score.unwrap_or(0.0_f32),
                doc_id: r.doc_id.clone(),
            })
            .collect();

        // Look up indices in the fast index for quality scoring.
        let mut fast_indices = Vec::with_capacity(fast_hits.len());
        for hit in &fast_hits {
            if let Some(index) = self.index.fast_index_for_doc_id(&hit.doc_id)? {
                fast_indices.push(index);
            }
        }
        metrics.phase2_vectors_searched = fast_indices.len();

        let quality_scores = self
            .index
            .quality_scores_for_indices(&quality_vec, &fast_indices)?;
        metrics.quality_search_ms = search_start.elapsed().as_secs_f64() * 1000.0;

        // Build quality VectorHits for blending.
        let mut quality_hits = Vec::with_capacity(fast_indices.len());
        for (&idx, &score) in fast_indices.iter().zip(quality_scores.iter()) {
            let doc_id = self.index.doc_id_at(idx)?.to_owned();
            quality_hits.push(VectorHit {
                #[allow(clippy::cast_possible_truncation)]
                index: idx as u32,
                score,
                doc_id,
            });
        }

        // Blend fast + quality scores.
        let blend_start = Instant::now();
        #[allow(clippy::cast_possible_truncation)]
        let blend_factor = self.config.quality_weight as f32;
        let blended = blend_two_tier(&fast_hits, &quality_hits, blend_factor);
        metrics.blend_ms = blend_start.elapsed().as_secs_f64() * 1000.0;

        // Compute rank changes (initial vs refined).
        // Precompute rank maps once, then pass to both functions.
        let initial_rank = build_borrowed_rank_map(&fast_hits);
        let refined_rank = build_borrowed_rank_map(&blended);
        let rank_changes = compute_rank_changes_with_maps(&initial_rank, &refined_rank);
        let tau = kendall_tau_with_refined_rank(&fast_hits, &refined_rank);
        metrics.kendall_tau = tau;
        metrics.rank_changes = rank_changes;

        let initial_by_doc: HashMap<&str, &ScoredResult> = initial_results
            .iter()
            .map(|result| (result.doc_id.as_str(), result))
            .collect();
        let fast_scores_by_doc: HashMap<&str, f32> = fast_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();
        let quality_scores_by_doc: HashMap<&str, f32> = quality_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();

        // Convert blended to scored results.
        #[allow(unused_mut)] // mut needed when `rerank` feature is enabled
        let mut results: Vec<ScoredResult> = blended
            .iter()
            .enumerate()
            .take(k)
            .map(|(rank, hit)| {
                let initial = initial_by_doc.get(hit.doc_id.as_str()).copied();
                let fast_score = fast_scores_by_doc.get(hit.doc_id.as_str()).copied();
                let quality_score = quality_scores_by_doc.get(hit.doc_id.as_str()).copied();
                let source = if quality_score.is_some() {
                    ScoreSource::SemanticQuality
                } else {
                    initial.map_or(ScoreSource::SemanticFast, |result| result.source)
                };

                let explanation = if self.config.explain {
                    let mut components = Vec::new();

                    // Fast component
                    if let Some(s) = fast_score {
                        components.push(ScoreComponent {
                            source: ExplainedSource::SemanticFast {
                                embedder: self.fast_embedder.id().to_owned(),
                                cosine_sim: f64::from(s),
                            },
                            raw_score: f64::from(s),
                            normalized_score: f64::from(s),
                            rrf_contribution: 0.0,
                            weight: 1.0 - f64::from(blend_factor),
                        });
                    }

                    // Quality component
                    if let Some(s) = quality_score {
                        components.push(ScoreComponent {
                            source: ExplainedSource::SemanticQuality {
                                embedder: self.quality_embedder.as_ref().map_or_else(
                                    || "quality-embedder".to_owned(),
                                    |e| e.id().to_owned(),
                                ),
                                cosine_sim: f64::from(s),
                            },
                            raw_score: f64::from(s),
                            normalized_score: f64::from(s),
                            rrf_contribution: 0.0,
                            weight: f64::from(blend_factor),
                        });
                    }

                    // Rank movement
                    let rank_movement = initial_rank.get(hit.doc_id.as_str()).map(|&i_rank| {
                        let refined_rank = i64::try_from(rank).unwrap_or(i64::MAX);
                        let initial_rank = i64::try_from(i_rank).unwrap_or(i64::MAX);
                        let delta_i64 = refined_rank - initial_rank;
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
                            initial_rank: i_rank,
                            refined_rank: rank,
                            delta,
                            reason: reason.to_owned(),
                        }
                    });

                    Some(HitExplanation {
                        final_score: f64::from(hit.score),
                        components,
                        phase: ExplanationPhase::Refined,
                        rank_movement,
                    })
                } else {
                    None
                };

                ScoredResult {
                    doc_id: hit.doc_id.clone(),
                    score: hit.score,
                    source,
                    fast_score,
                    quality_score,
                    lexical_score: initial.and_then(|result| result.lexical_score),
                    rerank_score: None,
                    explanation,
                    metadata: initial.and_then(|result| result.metadata.clone()),
                }
            })
            .collect();

        // Optional cross-encoder reranking.
        if let Some(ref reranker) = self.reranker {
            #[cfg(feature = "rerank")]
            {
                let rerank_start = Instant::now();
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
                metrics.rerank_ms = rerank_start.elapsed().as_secs_f64() * 1000.0;
            }
            #[cfg(not(feature = "rerank"))]
            {
                let _ = (reranker, text_fn);
                tracing::debug!("reranker configured but `rerank` feature not enabled");
            }
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

    fn emit_host_lifecycle_hook(&self, event: &AdapterLifecycleEvent) {
        let Some(host_adapter) = self.host_adapter.as_ref() else {
            return;
        };
        if let Err(err) = host_adapter.on_lifecycle_event(event) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter lifecycle hook failed");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_lifecycle_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
        state: LifecycleState,
        severity: LifecycleSeverity,
        reason: Option<String>,
        uptime_ms: Option<u64>,
    ) -> Option<String> {
        let host_adapter = self.host_adapter.as_ref()?;

        let event_id = next_telemetry_identifier("evt");
        let telemetry_instance = telemetry_instance_for_adapter(host_adapter.as_ref());
        let telemetry_correlation = TelemetryCorrelation {
            event_id: event_id.clone(),
            root_request_id: root_request_id.to_owned(),
            parent_event_id,
        };
        let envelope = TelemetryEnvelope::new(
            telemetry_timestamp_now(),
            TelemetryEvent::Lifecycle {
                instance: telemetry_instance,
                correlation: telemetry_correlation,
                state,
                severity,
                reason,
                uptime_ms,
            },
        );

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    fn emit_resource_telemetry(
        &self,
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
        let sample = self.collect_resource_sample();
        let envelope = self.runtime_metrics_collector.emit_resource(
            telemetry_timestamp_now(),
            telemetry_instance,
            telemetry_correlation,
            sample,
        );

        if let Err(err) = host_adapter.emit_telemetry(&envelope) {
            self.export_error(&err);
            tracing::warn!(error = %err, "host adapter telemetry emission failed");
        }

        Some(event_id)
    }

    fn collect_resource_sample(&self) -> ResourceCollectorSample {
        let interval_ms = self
            .runtime_metrics_collector
            .config()
            .collection_interval_ms;
        let process_jiffies = read_proc_process_jiffies();
        let total_jiffies = read_proc_total_jiffies();
        let cpu_pct = self.update_cpu_pct_estimate(process_jiffies, total_jiffies);
        let rss_bytes = read_proc_status_rss_bytes().unwrap_or(0);
        let (io_read_bytes, io_write_bytes) = read_proc_io_bytes().unwrap_or((0, 0));
        let load_avg_1m = read_proc_load_avg_1m();

        ResourceCollectorSample {
            cpu_pct,
            rss_bytes,
            io_read_bytes,
            io_write_bytes,
            interval_ms,
            load_avg_1m,
            pressure_profile: None,
        }
    }

    fn update_cpu_pct_estimate(
        &self,
        process_jiffies: Option<u64>,
        total_jiffies: Option<u64>,
    ) -> f64 {
        let Some(process_jiffies) = process_jiffies else {
            return 0.0;
        };
        let Some(total_jiffies) = total_jiffies else {
            return 0.0;
        };

        let current = CpuJiffiesSnapshot {
            process_jiffies,
            total_jiffies,
        };
        let mut cpu_state = self
            .resource_cpu_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cpu_pct = cpu_pct_from_jiffies(
            *cpu_state,
            current,
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        );
        *cpu_state = Some(current);
        cpu_pct
    }

    fn emit_phase_health_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
        uptime: Duration,
    ) {
        self.emit_host_lifecycle_hook(&AdapterLifecycleEvent::HealthTick {
            ts: telemetry_timestamp_now(),
        });
        let uptime_ms = Some(u64::try_from(uptime.as_millis()).unwrap_or(u64::MAX));
        let _ = self.emit_lifecycle_telemetry(
            root_request_id,
            parent_event_id.clone(),
            LifecycleState::Healthy,
            LifecycleSeverity::Info,
            None,
            uptime_ms,
        );
        let _ = self.emit_resource_telemetry(root_request_id, parent_event_id);
    }

    fn emit_session_stop_telemetry(
        &self,
        root_request_id: &str,
        parent_event_id: Option<String>,
        state: LifecycleState,
        severity: LifecycleSeverity,
        reason: Option<String>,
        uptime: Duration,
    ) {
        self.emit_host_lifecycle_hook(&AdapterLifecycleEvent::SessionStop {
            ts: telemetry_timestamp_now(),
        });
        let uptime_ms = Some(u64::try_from(uptime.as_millis()).unwrap_or(u64::MAX));
        let _ = self.emit_lifecycle_telemetry(
            root_request_id,
            parent_event_id,
            state,
            severity,
            reason,
            uptime_ms,
        );
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
    const CROCKFORD_BASE32: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    const TIMESTAMP_SHIFTS: [u32; 10] = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0];
    const RANDOM_SHIFTS: [u32; 16] = [75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0];

    let _ = prefix;

    // ULID timestamp component is 48 bits of milliseconds since Unix epoch.
    let timestamp_ms = telemetry_timestamp_ms() & 0x0000_FFFF_FFFF_FFFF;

    // Deterministic, process-local entropy for uniqueness without extra deps.
    let sequence = TELEMETRY_EVENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = u64::from(std::process::id());
    let entropy = sequence ^ timestamp_ms.rotate_left(17) ^ (pid << 32);

    let pid_low16 = u16::try_from(pid % (u64::from(u16::MAX) + 1)).unwrap_or_default();
    let mut random_bytes = [0_u8; 10];
    random_bytes[..2].copy_from_slice(&pid_low16.to_be_bytes());
    random_bytes[2..].copy_from_slice(&entropy.to_be_bytes());

    let random_component = random_bytes
        .iter()
        .fold(0_u128, |acc, byte| (acc << 8) | u128::from(*byte));

    let mut id = String::with_capacity(26);
    for shift in TIMESTAMP_SHIFTS {
        let index = usize::try_from((timestamp_ms >> shift) & 0x1F).unwrap_or_default();
        id.push(char::from(CROCKFORD_BASE32[index]));
    }
    for shift in RANDOM_SHIFTS {
        let index = usize::try_from((random_component >> shift) & 0x1F).unwrap_or_default();
        id.push(char::from(CROCKFORD_BASE32[index]));
    }

    id
}

const TELEMETRY_TIMESTAMP_FALLBACK_RFC3339: &str = "1970-01-01T00:00:00Z";

fn telemetry_timestamp_ms() -> u64 {
    let nanos = OffsetDateTime::now_utc().unix_timestamp_nanos();
    if nanos <= 0 {
        return 0;
    }
    let nanos_u128 = u128::try_from(nanos).unwrap_or_default();
    let millis = nanos_u128 / 1_000_000;
    u64::try_from(millis).unwrap_or(u64::MAX)
}

fn telemetry_timestamp_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| TELEMETRY_TIMESTAMP_FALLBACK_RFC3339.to_owned())
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

#[allow(clippy::cast_precision_loss)]
fn cpu_pct_from_jiffies(
    previous: Option<CpuJiffiesSnapshot>,
    current: CpuJiffiesSnapshot,
    available_cores: usize,
) -> f64 {
    let Some(previous) = previous else {
        return 0.0;
    };

    let process_delta = current
        .process_jiffies
        .saturating_sub(previous.process_jiffies);
    let total_delta = current.total_jiffies.saturating_sub(previous.total_jiffies);
    if process_delta == 0 || total_delta == 0 || available_cores == 0 {
        return 0.0;
    }

    let raw = (process_delta as f64 / total_delta as f64) * available_cores as f64 * 100.0;
    raw.clamp(0.0, 100.0)
}

fn read_proc_total_jiffies() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/stat").ok()?;
        parse_proc_total_jiffies(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_process_jiffies() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/self/stat").ok()?;
        parse_proc_process_jiffies(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_status_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/self/status").ok()?;
        parse_proc_status_rss_bytes(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_io_bytes() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/self/io").ok()?;
        parse_proc_io_bytes(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn read_proc_load_avg_1m() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string("/proc/loadavg").ok()?;
        parse_proc_load_avg_1m(raw.as_str())
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn parse_proc_total_jiffies(raw: &str) -> Option<u64> {
    let cpu_line = raw.lines().find(|line| line.starts_with("cpu "))?;
    let mut total_jiffies = 0_u64;
    for token in cpu_line.split_whitespace().skip(1) {
        let value = token.parse::<u64>().ok()?;
        total_jiffies = total_jiffies.checked_add(value)?;
    }
    Some(total_jiffies)
}

fn parse_proc_process_jiffies(raw: &str) -> Option<u64> {
    let close_paren = raw.rfind(')')?;
    let stats_after_comm = raw.get(close_paren + 2..)?;
    let mut stats = stats_after_comm.split_whitespace();
    let utime = stats.nth(11)?.parse::<u64>().ok()?;
    let stime = stats.next()?.parse::<u64>().ok()?;
    utime.checked_add(stime)
}

fn parse_proc_status_rss_bytes(raw: &str) -> Option<u64> {
    let line = raw
        .lines()
        .find(|line| line.trim_start().starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

fn parse_proc_io_bytes(raw: &str) -> Option<(u64, u64)> {
    let mut read_bytes = None;
    let mut write_bytes = None;
    for line in raw.lines() {
        if let Some((key, value)) = line.split_once(':') {
            let parsed = value.trim().parse::<u64>().ok()?;
            match key.trim() {
                "read_bytes" => read_bytes = Some(parsed),
                "write_bytes" => write_bytes = Some(parsed),
                _ => {}
            }
        }
    }
    Some((read_bytes?, write_bytes?))
}

fn parse_proc_load_avg_1m(raw: &str) -> Option<f64> {
    let value = raw.split_whitespace().next()?.parse::<f64>().ok()?;
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
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
    lexical_results: &[ScoredResult],
    explain: bool,
    fast_embedder_id: &str,
    rrf_k: f64,
) -> Vec<ScoredResult> {
    let lexical_metadata_by_doc: HashMap<&str, serde_json::Value> = lexical_results
        .iter()
        .filter_map(|result| {
            result
                .metadata
                .as_ref()
                .map(|metadata| (result.doc_id.as_str(), metadata.clone()))
        })
        .collect();

    fused
        .iter()
        .map(|fh| {
            #[allow(clippy::cast_possible_truncation)]
            let score = fh.rrf_score as f32;
            let source = if fh.in_both_sources {
                ScoreSource::Hybrid
            } else if fh.lexical_rank.is_some() {
                ScoreSource::Lexical
            } else if fh.semantic_rank.is_some() {
                ScoreSource::SemanticFast
            } else {
                // Graph-only candidates (no lexical/semantic rank) still come from hybrid fusion.
                ScoreSource::Hybrid
            };
            let explanation = explain.then(|| {
                let mut components = Vec::new();

                if let (Some(rank), Some(raw_score)) = (fh.lexical_rank, fh.lexical_score) {
                    components.push(ScoreComponent {
                        source: ExplainedSource::LexicalBm25 {
                            matched_terms: Vec::new(),
                            tf: 0.0,
                            idf: 0.0,
                        },
                        raw_score: f64::from(raw_score),
                        normalized_score: f64::from(raw_score),
                        rrf_contribution: rank_contribution(rrf_k, rank),
                        weight: 1.0,
                    });
                }

                if let (Some(rank), Some(raw_score)) = (fh.semantic_rank, fh.semantic_score) {
                    components.push(ScoreComponent {
                        source: ExplainedSource::SemanticFast {
                            embedder: fast_embedder_id.to_owned(),
                            cosine_sim: f64::from(raw_score),
                        },
                        raw_score: f64::from(raw_score),
                        normalized_score: f64::from(raw_score),
                        rrf_contribution: rank_contribution(rrf_k, rank),
                        weight: 1.0,
                    });
                }

                HitExplanation {
                    final_score: f64::from(score),
                    components,
                    phase: ExplanationPhase::Initial,
                    rank_movement: None,
                }
            });
            ScoredResult {
                doc_id: fh.doc_id.clone(),
                score,
                source,
                index: fh.semantic_index,
                fast_score: fh.semantic_score,
                quality_score: None,
                lexical_score: fh.lexical_score,
                rerank_score: None,
                explanation,
                metadata: lexical_metadata_by_doc.get(fh.doc_id.as_str()).cloned(),
            }
        })
        .collect()
}

fn rank_contribution(rrf_k: f64, rank: usize) -> f64 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (sanitize_rrf_k(rrf_k) + f64::from(rank_u32) + 1.0)
}

fn sanitize_rrf_k(k: f64) -> f64 {
    const DEFAULT_RRF_K: f64 = 60.0;
    if k.is_finite() && k >= 0.0 {
        k
    } else {
        DEFAULT_RRF_K
    }
}

/// Convert `VectorHit` results to `ScoredResult` (semantic-only mode).
fn vector_hits_to_scored_results(
    hits: &[VectorHit],
    k: usize,
    config: &TwoTierConfig,
    fast_embedder_id: &str,
) -> Vec<ScoredResult> {
    let mut seen = std::collections::HashSet::new();
    hits.iter()
        .filter(|h| seen.insert(&h.doc_id))
        .take(k)
        .map(|h| {
            let explanation = if config.explain {
                Some(HitExplanation {
                    final_score: f64::from(h.score),
                    components: vec![ScoreComponent {
                        source: ExplainedSource::SemanticFast {
                            embedder: fast_embedder_id.to_owned(),
                            cosine_sim: f64::from(h.score),
                        },
                        raw_score: f64::from(h.score),
                        normalized_score: f64::from(h.score),
                        rrf_contribution: 0.0,
                        weight: 1.0,
                    }],
                    phase: ExplanationPhase::Initial,
                    rank_movement: None,
                })
            } else {
                None
            };

            ScoredResult {
                doc_id: h.doc_id.clone(),
                score: h.score,
                source: ScoreSource::SemanticFast,
                index: Some(h.index),
                fast_score: Some(h.score),
                quality_score: None,
                lexical_score: None,
                rerank_score: None,
                explanation,
                metadata: None,
            }
        })
        .collect()
}

fn filter_scored_results_by_negations(
    results: Vec<ScoredResult>,
    exclusions: &NormalizedExclusions,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> Vec<ScoredResult> {
    results
        .into_iter()
        .filter(|result| !should_exclude_document(&result.doc_id, exclusions, text_fn, source))
        .collect()
}

fn filter_vector_hits_by_negations(
    hits: Vec<VectorHit>,
    exclusions: &NormalizedExclusions,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> Vec<VectorHit> {
    hits.into_iter()
        .filter(|hit| !should_exclude_document(&hit.doc_id, exclusions, text_fn, source))
        .collect()
}

fn should_exclude_document(
    doc_id: &str,
    exclusions: &NormalizedExclusions,
    text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    source: &'static str,
) -> bool {
    let Some(text) = text_fn(doc_id) else {
        return false;
    };
    let Some(matched_clause) = find_negative_match(&text, exclusions) else {
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

fn find_negative_match(text: &str, exclusions: &NormalizedExclusions) -> Option<String> {
    let normalized_text = normalize_for_negation_match(text);
    for term in &exclusions.terms {
        if !term.is_empty() && contains_negative_term(&normalized_text, term) {
            return Some(term.clone());
        }
    }
    for phrase in &exclusions.phrases {
        if !phrase.is_empty() && normalized_text.contains(phrase) {
            return Some(phrase.clone());
        }
    }
    None
}

fn normalize_for_negation_match(value: &str) -> String {
    value.nfc().collect::<String>().to_lowercase()
}

fn contains_negative_term(normalized_text: &str, normalized_term: &str) -> bool {
    if term_is_word_like(normalized_term) {
        contains_term_with_word_boundaries(normalized_text, normalized_term)
    } else {
        normalized_text.contains(normalized_term)
    }
}

fn term_is_word_like(term: &str) -> bool {
    term.chars().all(is_word_char)
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

fn contains_term_with_word_boundaries(text: &str, term: &str) -> bool {
    let mut search_from = 0_usize;
    while let Some(relative_index) = text[search_from..].find(term) {
        let start = search_from + relative_index;
        let end = start + term.len();
        let prev = text[..start].chars().next_back();
        let next = text[end..].chars().next();
        let start_boundary = prev.is_none_or(|ch| !is_word_char(ch));
        let end_boundary = next.is_none_or(|ch| !is_word_char(ch));
        if start_boundary && end_boundary {
            return true;
        }
        search_from = end;
    }
    false
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

    fn is_valid_ulid_like(candidate: &str) -> bool {
        if candidate.len() != 26 {
            return false;
        }
        candidate.bytes().all(|byte| {
            matches!(
                byte,
                b'0'..=b'9'
                    | b'A'..=b'H'
                    | b'J'..=b'K'
                    | b'M'..=b'N'
                    | b'P'..=b'T'
                    | b'V'..=b'Z'
            )
        })
    }

    #[test]
    fn scaled_budget_clamps_positive_budget_to_at_least_one() {
        assert_eq!(scaled_budget(1, 0.5), 1);
        assert_eq!(scaled_budget(1, 0.25), 1);
        assert_eq!(scaled_budget(2, 0.5), 1);
        assert_eq!(scaled_budget(0, 0.5), 0);
        assert_eq!(scaled_budget(4, 0.0), 0);
    }

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
                        explanation: None,
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

        fn lifecycle_events(&self) -> Vec<AdapterLifecycleEvent> {
            self.lifecycle.lock().expect("lifecycle lock").clone()
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
    fn initial_phase_metrics_report_fast_index_scope() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4); // 10 docs in fixture index
            let expected_doc_count = index.doc_count();
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let mut phase1_vectors = None;
            let mut phase1_semantic_candidates = None;
            searcher
                .search(
                    &cx,
                    "test query",
                    3, // keep k below index size to ensure hit count != scan scope
                    |_| None,
                    |phase| {
                        if let SearchPhase::Initial { metrics, .. } = phase {
                            phase1_vectors = Some(metrics.vectors_searched);
                            phase1_semantic_candidates = Some(metrics.fused_count);
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert_eq!(
                phase1_vectors,
                Some(expected_doc_count),
                "phase-1 vectors_searched should describe fast-tier search scope"
            );
            assert!(
                phase1_semantic_candidates.is_some_and(|count| count <= expected_doc_count),
                "fused candidates should not exceed index size"
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
    fn graph_ranking_enabled_stub_is_noop() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let lexical: Arc<dyn LexicalSearch> = Arc::new(StubLexical);
            let config = TwoTierConfig {
                graph_ranking_enabled: true,
                graph_ranking_weight: 0.9,
                ..TwoTierConfig::default()
            };
            let searcher = TwoTierSearcher::new(index, fast, config).with_lexical(lexical);

            let mut saw_initial = false;
            let metrics = searcher
                .search(
                    &cx,
                    "graph ranking stub query",
                    5,
                    |_| None,
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            saw_initial = true;
                            assert!(
                                !results.is_empty(),
                                "phase-1 results should still be present"
                            );
                        }
                    },
                )
                .await
                .expect("search should succeed with stubbed graph path");

            assert!(saw_initial, "phase-1 callback should fire");
            assert!(
                metrics.phase1_total_ms >= 0.0,
                "phase-1 latency should remain well-formed"
            );
        });
    }

    #[cfg(feature = "graph")]
    #[test]
    fn graph_ranking_with_document_graph_can_add_graph_only_candidate() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let config = TwoTierConfig {
                graph_ranking_enabled: true,
                graph_ranking_weight: 1.0,
                ..TwoTierConfig::default()
            };
            let mut graph = DocumentGraph::new();
            graph.add_edge(
                "doc-0",
                "doc-extra",
                frankensearch_core::EdgeType::Reference,
                1.0,
            );
            let searcher =
                TwoTierSearcher::new(index, fast, config).with_document_graph(Arc::new(graph));

            let mut initial_docs = Vec::new();
            searcher
                .search(
                    &cx,
                    "graph only candidate query",
                    10,
                    |_| None,
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            initial_docs =
                                results.into_iter().map(|result| result.doc_id).collect();
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert!(
                initial_docs.iter().any(|doc_id| doc_id == "doc-extra"),
                "graph channel should be able to contribute graph-only candidates",
            );
        });
    }

    #[cfg(feature = "graph")]
    #[test]
    fn graph_ranking_candidates_respect_negation_filters() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let config = TwoTierConfig {
                graph_ranking_enabled: true,
                graph_ranking_weight: 1.0,
                ..TwoTierConfig::default()
            };
            let mut graph = DocumentGraph::new();
            graph.add_edge(
                "doc-0",
                "doc-extra",
                frankensearch_core::EdgeType::Reference,
                1.0,
            );
            let searcher =
                TwoTierSearcher::new(index, fast, config).with_document_graph(Arc::new(graph));

            let mut initial_docs_without_negation = Vec::new();
            searcher
                .search(
                    &cx,
                    "graph only candidate query",
                    10,
                    |doc_id| {
                        if doc_id == "doc-extra" {
                            Some("unsafe edge content".to_owned())
                        } else {
                            Some("safe content".to_owned())
                        }
                    },
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            initial_docs_without_negation =
                                results.into_iter().map(|result| result.doc_id).collect();
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert!(
                initial_docs_without_negation
                    .iter()
                    .any(|doc_id| doc_id == "doc-extra"),
                "graph channel should contribute graph-only candidate without negation",
            );

            let mut initial_docs_with_negation = Vec::new();
            searcher
                .search(
                    &cx,
                    "graph only candidate query -unsafe",
                    10,
                    |doc_id| {
                        if doc_id == "doc-extra" {
                            Some("unsafe edge content".to_owned())
                        } else {
                            Some("safe content".to_owned())
                        }
                    },
                    |phase| {
                        if let SearchPhase::Initial { results, .. } = phase {
                            initial_docs_with_negation =
                                results.into_iter().map(|result| result.doc_id).collect();
                        }
                    },
                )
                .await
                .expect("search should succeed");

            assert!(
                !initial_docs_with_negation
                    .iter()
                    .any(|doc_id| doc_id == "doc-extra"),
                "graph candidates matching negation should be filtered before fusion",
            );
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
    fn fast_embed_failure_without_lexical_returns_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder);
            let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

            let mut phase_count = 0;
            let err = searcher
                .search(
                    &cx,
                    "test",
                    5,
                    |_| None,
                    |_| {
                        phase_count += 1;
                    },
                )
                .await
                .expect_err("semantic-only fast embed failure should return an error");

            assert!(matches!(err, SearchError::EmbeddingFailed { .. }));
            assert_eq!(
                phase_count, 0,
                "hard failure in phase 1 should not emit any search phases"
            );
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
                Some("vector_index_unavailable")
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
    fn exclusion_full_pipeline_rust_unsafe_returns_safe_docs() {
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
                    "rust -unsafe",
                    12,
                    |doc_id| {
                        let text = match doc_id {
                            "doc-0" => "unsafe rust pointer tricks",
                            "doc-1" => "rust ownership and borrowing",
                            "lex-doc-0" => "unsafe lexical result",
                            "lex-doc-1" => "safe rust lexical result",
                            "lex-doc-2" => "safe rust patterns",
                            _ => "safe rust systems programming",
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

            assert!(
                !initial_results
                    .iter()
                    .any(|result| result.doc_id == "doc-0"),
                "unsafe semantic result should be filtered"
            );
            assert!(
                !initial_results
                    .iter()
                    .any(|result| result.doc_id == "lex-doc-0"),
                "unsafe lexical result should be filtered"
            );
            assert!(
                initial_results
                    .iter()
                    .all(|result| result.doc_id != "doc-0" && result.doc_id != "lex-doc-0")
            );
            assert!(
                !initial_results.is_empty(),
                "expected at least one safe rust result to remain"
            );
        });
    }

    #[test]
    fn exclusion_overhead_is_sub_millisecond_for_typical_query() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

            let baseline = searcher
                .search(
                    &cx,
                    "rust systems",
                    10,
                    |_| Some("safe rust systems".to_owned()),
                    |_| {},
                )
                .await
                .expect("baseline search should succeed")
                .phase1_total_ms;
            let negated = searcher
                .search(
                    &cx,
                    "rust systems -unsafe",
                    10,
                    |doc_id| {
                        let text = if doc_id == "doc-0" {
                            "unsafe rust systems"
                        } else {
                            "safe rust systems"
                        };
                        Some(text.to_owned())
                    },
                    |_| {},
                )
                .await
                .expect("negated search should succeed")
                .phase1_total_ms;

            let overhead_ms = (negated - baseline).max(0.0);
            assert!(
                overhead_ms < 1.0,
                "expected exclusion overhead <1ms, observed {overhead_ms:.4}ms (baseline={baseline:.4}ms, negated={negated:.4}ms)"
            );
        });
    }

    /// Convert a `ParsedQuery` to `NormalizedExclusions` for testing.
    fn to_exclusions(parsed: &ParsedQuery) -> NormalizedExclusions {
        NormalizedExclusions {
            terms: parsed
                .negative_terms
                .iter()
                .map(|t| normalize_for_negation_match(t))
                .collect(),
            phrases: parsed
                .negative_phrases
                .iter()
                .map(|p| normalize_for_negation_match(p))
                .collect(),
        }
    }

    #[test]
    fn exclusion_matching_normalizes_unicode_forms() {
        let parsed = ParsedQuery::parse("rust -café");
        let exclusions = to_exclusions(&parsed);
        let decomposed_text = "safe docs with caf\u{0065}\u{0301} references";
        let matched = find_negative_match(decomposed_text, &exclusions);
        assert_eq!(matched, Some("café".to_owned()));
    }

    #[test]
    fn exclusion_term_matching_requires_word_boundaries_for_word_terms() {
        let parsed = ParsedQuery::parse("query -he");
        let exclusions = to_exclusions(&parsed);
        let text = "the theorem should stay included";
        let matched = find_negative_match(text, &exclusions);
        assert_eq!(matched, None);
    }

    #[test]
    fn exclusion_term_matching_excludes_whole_word_occurrences() {
        let parsed = ParsedQuery::parse("query -he");
        let exclusions = to_exclusions(&parsed);
        let text = "we saw he walk home";
        let matched = find_negative_match(text, &exclusions);
        assert_eq!(matched, Some("he".to_owned()));
    }

    #[test]
    fn exclusion_term_matching_keeps_substring_behavior_for_path_like_terms() {
        let parsed = ParsedQuery::parse("query -src/main.rs");
        let exclusions = to_exclusions(&parsed);
        let text = "candidate path=/workspace/src/main.rs.bak";
        let matched = find_negative_match(text, &exclusions);
        assert_eq!(matched, Some("src/main.rs".to_owned()));
    }

    #[test]
    fn refined_phase_uses_zero_fast_score_for_lexical_only_candidates() {
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
                lexical_only
                    .fast_score
                    .is_some_and(|score| score == 0.0_f32),
                "lexical-only refined result should keep missing fast-source score at 0.0"
            );
            assert_eq!(
                lexical_only.source,
                ScoreSource::Lexical,
                "lexical-only refined result should retain lexical provenance when quality score is absent"
            );
            assert!(
                lexical_only
                    .lexical_score
                    .is_some_and(|score| score > 0.0_f32),
                "lexical-only refined result should preserve lexical score for diagnostics"
            );
        });
    }

    #[test]
    fn fused_hits_to_scored_results_preserves_lexical_metadata() {
        let fused = vec![
            frankensearch_core::types::FusedHit {
                doc_id: "lex-doc-1".to_owned(),
                rrf_score: 1.5,
                lexical_rank: Some(0),
                semantic_rank: None,
                lexical_score: Some(3.0),
                semantic_score: None,
                in_both_sources: false,
            },
            frankensearch_core::types::FusedHit {
                doc_id: "sem-doc-1".to_owned(),
                rrf_score: 1.0,
                lexical_rank: None,
                semantic_rank: Some(0),
                lexical_score: None,
                semantic_score: Some(0.8),
                in_both_sources: false,
            },
        ];
        let lexical_results = vec![ScoredResult {
            doc_id: "lex-doc-1".to_owned(),
            score: 3.0,
            source: ScoreSource::Lexical,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(3.0),
            rerank_score: None,
            explanation: None,
            metadata: Some(serde_json::json!({
                "title": "Lexical doc",
                "section": "api",
            })),
        }];

        let scored =
            fused_hits_to_scored_results(&fused, &lexical_results, false, "fast-test", 60.0);
        let lexical = scored
            .iter()
            .find(|result| result.doc_id == "lex-doc-1")
            .expect("lexical fused result must exist");
        assert_eq!(
            lexical.metadata,
            Some(serde_json::json!({
                "title": "Lexical doc",
                "section": "api",
            })),
            "lexical metadata should be preserved through fused conversion"
        );

        let semantic = scored
            .iter()
            .find(|result| result.doc_id == "sem-doc-1")
            .expect("semantic fused result must exist");
        assert!(
            semantic.metadata.is_none(),
            "semantic-only fused result should not synthesize metadata"
        );
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
                is_valid_ulid_like(&initial_event_id),
                "initial event id should be ULID-like"
            );
            assert!(
                is_valid_ulid_like(&root_request_id),
                "root request id should be ULID-like"
            );

            let saw_refined_event = {
                let (correlation, query) = search_events[1];
                assert_eq!(query.phase, SearchEventPhase::Refined);
                assert!(is_valid_ulid_like(&correlation.event_id));
                assert_eq!(correlation.root_request_id, root_request_id);
                assert_eq!(
                    correlation.parent_event_id.as_deref(),
                    Some(initial_event_id.as_str())
                );
                assert_ne!(correlation.event_id, initial_event_id);
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
                    TelemetryEvent::Search {
                        correlation,
                        query,
                        results,
                        ..
                    } => Some((correlation, query, results)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                search_events.len(),
                2,
                "expected initial + refinement_failed search events"
            );

            let (initial_event_id, root_request_id, initial_result_count) = {
                let (correlation, query, results) = search_events[0];
                assert_eq!(query.phase, SearchEventPhase::Initial);
                assert!(is_valid_ulid_like(&correlation.event_id));
                assert!(is_valid_ulid_like(&correlation.root_request_id));
                (
                    correlation.event_id.clone(),
                    correlation.root_request_id.clone(),
                    results.result_count,
                )
            };

            let saw_refinement_failed = {
                let (correlation, query, results) = search_events[1];
                assert_eq!(query.phase, SearchEventPhase::RefinementFailed);
                assert_eq!(results.result_count, initial_result_count);
                assert!(is_valid_ulid_like(&correlation.event_id));
                assert_eq!(correlation.root_request_id, root_request_id);
                assert_eq!(
                    correlation.parent_event_id.as_deref(),
                    Some(initial_event_id.as_str())
                );
                assert_ne!(correlation.event_id, initial_event_id);
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
    fn host_adapter_receives_lifecycle_and_resource_events_with_runtime_hooks() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let adapter = Arc::new(RecordingHostAdapter::new("coding_agent_session_search"));

            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality)
                .with_host_adapter(adapter.clone());

            let _ = searcher
                .search(&cx, "lifecycle telemetry", 5, |_| None, |_| {})
                .await
                .expect("search should succeed");

            let emitted = adapter.telemetry_events();
            let lifecycle_events: Vec<_> = emitted
                .iter()
                .filter_map(|envelope| match &envelope.event {
                    TelemetryEvent::Lifecycle {
                        correlation,
                        state,
                        severity,
                        uptime_ms,
                        ..
                    } => Some((correlation, state, severity, uptime_ms)),
                    _ => None,
                })
                .collect();
            assert!(
                lifecycle_events.len() >= 3,
                "expected start/health/stop lifecycle telemetry events"
            );
            assert!(lifecycle_events.iter().any(|(_, state, severity, _)| {
                **state == LifecycleState::Started && **severity == LifecycleSeverity::Info
            }));
            assert!(
                lifecycle_events
                    .iter()
                    .any(|(_, state, severity, uptime_ms)| {
                        **state == LifecycleState::Stopped
                            && **severity == LifecycleSeverity::Info
                            && uptime_ms.is_some()
                    })
            );
            assert!(
                lifecycle_events
                    .iter()
                    .any(|(_, state, _, _)| **state == LifecycleState::Healthy)
            );

            let resource_events: Vec<_> = emitted
                .iter()
                .filter_map(|envelope| match &envelope.event {
                    TelemetryEvent::Resource { sample, .. } => Some(sample),
                    _ => None,
                })
                .collect();
            assert!(
                !resource_events.is_empty(),
                "expected resource telemetry from phase health emission"
            );
            assert!(
                resource_events.iter().all(|sample| sample.interval_ms > 0),
                "resource interval should always be non-zero"
            );
            assert!(resource_events.iter().all(|sample| {
                sample.cpu_pct.is_finite() && (0.0..=100.0).contains(&sample.cpu_pct)
            }));
            let collector_snapshot = searcher.runtime_metrics_collector.snapshot();
            assert!(
                collector_snapshot.resource_events_emitted >= 1,
                "phase health should emit at least one resource event"
            );

            let lifecycle_hooks = adapter.lifecycle_events();
            assert!(matches!(
                lifecycle_hooks.first(),
                Some(AdapterLifecycleEvent::SessionStart { .. })
            ));
            assert!(
                lifecycle_hooks
                    .iter()
                    .any(|event| matches!(event, AdapterLifecycleEvent::HealthTick { .. }))
            );
            assert!(matches!(
                lifecycle_hooks.last(),
                Some(AdapterLifecycleEvent::SessionStop { .. })
            ));
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

    // ─── Counting Embedder (for cache-wiring tests) ────────────────────

    /// Embedder that counts inner `embed()` invocations via an external counter.
    struct CountingEmbedder {
        id: &'static str,
        dimension: usize,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CountingEmbedder {
        fn new(
            id: &'static str,
            dimension: usize,
            calls: Arc<std::sync::atomic::AtomicUsize>,
        ) -> Self {
            Self {
                id,
                dimension,
                calls,
            }
        }
    }

    impl Embedder for CountingEmbedder {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dim = self.dimension;
            Box::pin(async move {
                let mut vec = vec![0.0; dim];
                if !vec.is_empty() {
                    vec[0] = 1.0;
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

    // ─── Cache-wiring tests ────────────────────────────────────────────

    #[test]
    fn embedding_cache_wraps_fast_tier() {
        let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls.clone()));

        let index = build_test_index(4);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_embedding_cache(64);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // First search: fast embedder called
            let _ = searcher
                .search(&cx, "hello world", 5, |_| None, |_| {})
                .await
                .unwrap();
            let after_first = fast_calls.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                after_first >= 1,
                "fast embedder should be called at least once"
            );

            // Same query again: should hit cache, no additional inner calls
            let _ = searcher
                .search(&cx, "hello world", 5, |_| None, |_| {})
                .await
                .unwrap();
            let after_second = fast_calls.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(
                after_first, after_second,
                "repeated query should hit cache (fast tier)"
            );
        });
    }

    #[test]
    fn embedding_cache_wraps_quality_tier_when_set_before() {
        let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls));
        let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

        let index = build_test_index(4);
        // quality set BEFORE cache — both should be wrapped
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality)
            .with_embedding_cache(64);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let _ = searcher
                .search(&cx, "cache test", 5, |_| None, |_| {})
                .await
                .unwrap();
            let q_after_first = quality_calls.load(std::sync::atomic::Ordering::Relaxed);

            let _ = searcher
                .search(&cx, "cache test", 5, |_| None, |_| {})
                .await
                .unwrap();
            let q_after_second = quality_calls.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(
                q_after_first, q_after_second,
                "repeated query should hit cache (quality tier, set before cache)"
            );
        });
    }

    #[test]
    fn embedding_cache_wraps_quality_tier_when_set_after() {
        let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let quality_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls));
        let quality = Arc::new(CountingEmbedder::new("quality", 4, quality_calls.clone()));

        let index = build_test_index(4);
        // cache set BEFORE quality — quality should still be auto-wrapped
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_embedding_cache(64)
            .with_quality_embedder(quality);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let _ = searcher
                .search(&cx, "order test", 5, |_| None, |_| {})
                .await
                .unwrap();
            let q_after_first = quality_calls.load(std::sync::atomic::Ordering::Relaxed);

            let _ = searcher
                .search(&cx, "order test", 5, |_| None, |_| {})
                .await
                .unwrap();
            let q_after_second = quality_calls.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(
                q_after_first, q_after_second,
                "repeated query should hit cache (quality tier, set after cache)"
            );
        });
    }

    #[test]
    fn different_queries_are_cache_misses() {
        let fast_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let fast = Arc::new(CountingEmbedder::new("fast", 4, fast_calls.clone()));

        let index = build_test_index(4);
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_embedding_cache(64);

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let _ = searcher
                .search(&cx, "query alpha", 5, |_| None, |_| {})
                .await
                .unwrap();
            let after_alpha = fast_calls.load(std::sync::atomic::Ordering::Relaxed);

            let _ = searcher
                .search(&cx, "query beta", 5, |_| None, |_| {})
                .await
                .unwrap();
            let after_beta = fast_calls.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                after_beta > after_alpha,
                "different query should be a cache miss"
            );
        });
    }

    fn phase2_lookup_checksum_linear(
        blended: &[VectorHit],
        fast_hits: &[VectorHit],
        quality_hits: &[VectorHit],
    ) -> f32 {
        blended.iter().fold(0.0_f32, |acc, hit| {
            let fast_score = fast_hits
                .iter()
                .find(|h| h.doc_id == hit.doc_id)
                .map_or(0.0_f32, |h| h.score);
            let quality_score = quality_hits
                .iter()
                .find(|h| h.doc_id == hit.doc_id)
                .map_or(0.0_f32, |h| h.score);
            acc + fast_score + quality_score + hit.score
        })
    }

    fn phase2_lookup_checksum_mapped(
        blended: &[VectorHit],
        fast_scores_by_doc: &HashMap<&str, f32>,
        quality_scores_by_doc: &HashMap<&str, f32>,
    ) -> f32 {
        blended.iter().fold(0.0_f32, |acc, hit| {
            let fast_score = fast_scores_by_doc
                .get(hit.doc_id.as_str())
                .copied()
                .unwrap_or(0.0_f32);
            let quality_score = quality_scores_by_doc
                .get(hit.doc_id.as_str())
                .copied()
                .unwrap_or(0.0_f32);
            acc + fast_score + quality_score + hit.score
        })
    }

    fn build_phase2_lookup_fixture(
        doc_count: usize,
    ) -> (Vec<VectorHit>, Vec<VectorHit>, Vec<VectorHit>) {
        let fast_hits: Vec<VectorHit> = (0..doc_count)
            .map(|idx| VectorHit {
                index: u32::try_from(idx).expect("idx fits in u32 for test fixture"),
                score: idx as f32 * 0.001_f32,
                doc_id: format!("doc-{idx}"),
            })
            .collect();
        let quality_hits: Vec<VectorHit> = (0..doc_count)
            .step_by(3)
            .map(|idx| VectorHit {
                index: u32::try_from(idx).expect("idx fits in u32 for test fixture"),
                score: idx as f32 * 0.002_f32,
                doc_id: format!("doc-{idx}"),
            })
            .collect();
        let blended: Vec<VectorHit> = (0..doc_count)
            .map(|idx| VectorHit {
                index: u32::try_from(idx).expect("idx fits in u32 for test fixture"),
                doc_id: format!("doc-{idx}"),
                score: idx as f32 * 0.0005_f32,
            })
            .collect();
        (fast_hits, quality_hits, blended)
    }

    #[test]
    fn phase2_lookup_maps_match_linear_scan_oracle() {
        let (fast_hits, quality_hits, blended) = build_phase2_lookup_fixture(10_000);
        let fast_scores_by_doc: HashMap<&str, f32> = fast_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();
        let quality_scores_by_doc: HashMap<&str, f32> = quality_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();

        let linear = phase2_lookup_checksum_linear(&blended, &fast_hits, &quality_hits);
        let mapped =
            phase2_lookup_checksum_mapped(&blended, &fast_scores_by_doc, &quality_scores_by_doc);

        let diff = (linear - mapped).abs();
        assert!(
            diff <= 0.0001_f32,
            "mapped lookup diverged from linear oracle: diff={diff}"
        );
    }

    // ─── bd-3a7q tests begin ───

    #[test]
    fn scaled_budget_negative_multiplier_returns_zero() {
        assert_eq!(scaled_budget(10, -1.0), 0);
        assert_eq!(scaled_budget(100, -0.5), 0);
    }

    #[test]
    fn scaled_budget_exact_one_multiplier() {
        assert_eq!(scaled_budget(7, 1.0), 7);
        assert_eq!(scaled_budget(1, 1.0), 1);
    }

    #[test]
    fn scaled_budget_large_values_do_not_panic() {
        let result = scaled_budget(usize::MAX / 2, 2.0);
        assert!(result >= 1);
    }

    #[test]
    fn scaled_budget_nan_multiplier_returns_zero() {
        assert_eq!(scaled_budget(10, f32::NAN), 0);
    }

    #[test]
    fn scaled_budget_infinity_multiplier_returns_zero() {
        assert_eq!(scaled_budget(10, f32::INFINITY), 0);
        assert_eq!(scaled_budget(10, f32::NEG_INFINITY), 0);
    }

    #[test]
    fn scaled_budget_fractional_rounds_up() {
        // 3 * 0.4 = 1.2 → ceil = 2
        assert_eq!(scaled_budget(3, 0.4), 2);
        // 5 * 0.3 = 1.5 → ceil = 2
        assert_eq!(scaled_budget(5, 0.3), 2);
    }

    #[test]
    fn embedder_tier_for_stage_quality_always_returns_quality() {
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Quality, ModelCategory::HashEmbedder),
            EmbedderTier::Quality
        );
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Quality, ModelCategory::StaticEmbedder),
            EmbedderTier::Quality
        );
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Quality, ModelCategory::TransformerEmbedder),
            EmbedderTier::Quality
        );
    }

    #[test]
    fn embedder_tier_for_stage_fast_maps_category() {
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Fast, ModelCategory::HashEmbedder),
            EmbedderTier::Hash
        );
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Fast, ModelCategory::StaticEmbedder),
            EmbedderTier::Fast
        );
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Fast, ModelCategory::TransformerEmbedder),
            EmbedderTier::Quality
        );
    }

    #[test]
    fn embedder_tier_for_stage_background_maps_same_as_fast() {
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Background, ModelCategory::HashEmbedder),
            EmbedderTier::Hash
        );
        assert_eq!(
            embedder_tier_for_stage(EmbeddingStage::Background, ModelCategory::StaticEmbedder),
            EmbedderTier::Fast
        );
        assert_eq!(
            embedder_tier_for_stage(
                EmbeddingStage::Background,
                ModelCategory::TransformerEmbedder
            ),
            EmbedderTier::Quality
        );
    }

    #[test]
    fn next_telemetry_identifier_is_ulid_like_and_unique() {
        let id1 = next_telemetry_identifier("root");
        let id2 = next_telemetry_identifier("root");
        assert!(is_valid_ulid_like(&id1));
        assert!(is_valid_ulid_like(&id2));
        assert_ne!(id1, id2);
    }

    #[test]
    fn next_telemetry_identifier_is_uppercase_crockford_base32() {
        let id = next_telemetry_identifier("evt");
        assert_eq!(id.len(), 26, "ULID identifiers are 26 chars");
        assert!(
            id.chars()
                .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit()),
            "generated identifier should use uppercase Crockford base32"
        );
    }

    #[test]
    fn telemetry_timestamp_now_is_valid_rfc3339() {
        let ts = telemetry_timestamp_now();
        assert!(!ts.is_empty());
        assert!(
            OffsetDateTime::parse(&ts, &Rfc3339).is_ok(),
            "should be RFC3339"
        );
    }

    #[test]
    fn telemetry_timestamp_fallback_constant_is_valid_rfc3339() {
        assert!(
            OffsetDateTime::parse(TELEMETRY_TIMESTAMP_FALLBACK_RFC3339, &Rfc3339).is_ok(),
            "fallback timestamp must remain RFC3339"
        );
    }

    #[test]
    fn cpu_pct_from_jiffies_first_sample_returns_zero() {
        let current = CpuJiffiesSnapshot {
            process_jiffies: 500,
            total_jiffies: 10_000,
        };
        assert!(
            (cpu_pct_from_jiffies(None, current, 8) - 0.0).abs() < f64::EPSILON,
            "first sample should not report synthetic CPU utilization"
        );
    }

    #[test]
    fn cpu_pct_from_jiffies_clamps_to_conformance_range() {
        let previous = CpuJiffiesSnapshot {
            process_jiffies: 100,
            total_jiffies: 1_000,
        };
        let current = CpuJiffiesSnapshot {
            process_jiffies: 200,
            total_jiffies: 1_100,
        };
        let cpu_pct = cpu_pct_from_jiffies(Some(previous), current, 4);
        assert!(
            (cpu_pct - 100.0).abs() < f64::EPSILON,
            "high deltas should clamp at 100%"
        );
    }

    #[test]
    fn parse_proc_total_jiffies_extracts_aggregate_sum() {
        let fixture = "cpu  10 20 30 40 50 60 70 80 90 100\ncpu0 1 2 3 4\n";
        assert_eq!(parse_proc_total_jiffies(fixture), Some(550));
    }

    #[test]
    fn parse_proc_process_jiffies_handles_command_with_spaces() {
        let fixture = "1234 (two tier worker) S 1 2 3 4 5 6 7 8 9 10 200 300 0 0 0 0 0 0 0";
        assert_eq!(parse_proc_process_jiffies(fixture), Some(500));
    }

    #[test]
    fn parse_proc_status_rss_bytes_extracts_vm_rss() {
        let fixture = "Name:\tproc\nVmRSS:\t   4096 kB\nThreads:\t8\n";
        assert_eq!(parse_proc_status_rss_bytes(fixture), Some(4_194_304));
    }

    #[test]
    fn parse_proc_io_bytes_extracts_read_and_write_bytes() {
        let fixture = "rchar: 1\nwchar: 2\nread_bytes: 333\nwrite_bytes: 444\n";
        assert_eq!(parse_proc_io_bytes(fixture), Some((333, 444)));
    }

    #[test]
    fn parse_proc_load_avg_1m_extracts_first_value() {
        let fixture = "1.42 0.99 0.55 2/321 9999\n";
        assert_eq!(parse_proc_load_avg_1m(fixture), Some(1.42));
    }

    #[test]
    fn vector_hits_to_scored_results_truncates_to_k() {
        let hits: Vec<VectorHit> = (0..10)
            .map(|i| VectorHit {
                index: i,
                score: (10 - i) as f32,
                doc_id: format!("doc-{i}"),
            })
            .collect();
        let config = TwoTierConfig::default();
        let results = vector_hits_to_scored_results(&hits, 3, &config, "test-fast");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, "doc-0");
        assert_eq!(results[2].doc_id, "doc-2");
    }

    #[test]
    fn vector_hits_to_scored_results_empty_hits() {
        let config = TwoTierConfig::default();
        let results = vector_hits_to_scored_results(&[], 5, &config, "test-fast");
        assert!(results.is_empty());
    }

    #[test]
    fn vector_hits_to_scored_results_sets_correct_fields() {
        let hits = vec![VectorHit {
            index: 0,
            score: 0.95,
            doc_id: "my-doc".to_owned(),
        }];
        let config = TwoTierConfig::default();
        let results = vector_hits_to_scored_results(&hits, 10, &config, "test-fast");
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.doc_id, "my-doc");
        assert!((r.score - 0.95).abs() < f32::EPSILON);
        assert_eq!(r.source, ScoreSource::SemanticFast);
        assert!(
            r.fast_score
                .is_some_and(|s| (s - 0.95).abs() < f32::EPSILON)
        );
        assert!(r.quality_score.is_none());
        assert!(r.lexical_score.is_none());
        assert!(r.rerank_score.is_none());
        assert!(r.metadata.is_none());
    }

    #[test]
    fn fused_hits_hybrid_source_classification() {
        let fused = vec![frankensearch_core::types::FusedHit {
            doc_id: "hybrid-doc".to_owned(),
            rrf_score: 2.0,
            lexical_rank: Some(0),
            semantic_rank: Some(1),
            lexical_score: Some(3.0),
            semantic_score: Some(0.8),
            in_both_sources: true,
        }];
        let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
        assert_eq!(results[0].source, ScoreSource::Hybrid);
        assert_eq!(results[0].fast_score, Some(0.8));
        assert_eq!(results[0].lexical_score, Some(3.0));
    }

    #[test]
    fn fused_hits_semantic_only_source() {
        let fused = vec![frankensearch_core::types::FusedHit {
            doc_id: "sem-only".to_owned(),
            rrf_score: 1.0,
            lexical_rank: None,
            semantic_rank: Some(0),
            lexical_score: None,
            semantic_score: Some(0.9),
            in_both_sources: false,
        }];
        let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
        assert_eq!(results[0].source, ScoreSource::SemanticFast);
    }

    #[test]
    fn fused_hits_lexical_only_source() {
        let fused = vec![frankensearch_core::types::FusedHit {
            doc_id: "lex-only".to_owned(),
            rrf_score: 1.0,
            lexical_rank: Some(0),
            semantic_rank: None,
            lexical_score: Some(2.5),
            semantic_score: None,
            in_both_sources: false,
        }];
        let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
        assert_eq!(results[0].source, ScoreSource::Lexical);
    }

    #[test]
    fn fused_hits_graph_only_source_defaults_to_hybrid() {
        let fused = vec![frankensearch_core::types::FusedHit {
            doc_id: "graph-only".to_owned(),
            rrf_score: 1.0,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        }];
        let results = fused_hits_to_scored_results(&fused, &[], false, "fast-test", 60.0);
        assert_eq!(results[0].source, ScoreSource::Hybrid);
    }

    #[test]
    fn fused_hits_include_explanation_when_enabled() {
        let fused = vec![frankensearch_core::types::FusedHit {
            doc_id: "explained".to_owned(),
            rrf_score: 1.25,
            lexical_rank: Some(1),
            semantic_rank: Some(3),
            lexical_score: Some(2.0),
            semantic_score: Some(0.5),
            in_both_sources: true,
        }];

        let results = fused_hits_to_scored_results(&fused, &[], true, "fast-test", 60.0);
        let explanation = results[0]
            .explanation
            .as_ref()
            .expect("explain=true should populate explanation");
        assert_eq!(explanation.phase, ExplanationPhase::Initial);
        assert_eq!(explanation.components.len(), 2);
        assert!(explanation.components[0].rrf_contribution > 0.0);
        assert!(explanation.components[1].rrf_contribution > 0.0);
    }

    #[test]
    fn normalize_for_negation_match_lowercases() {
        assert_eq!(normalize_for_negation_match("HELLO"), "hello");
        assert_eq!(normalize_for_negation_match("MiXeD"), "mixed");
    }

    #[test]
    fn normalize_for_negation_match_nfc_composing() {
        // e + combining acute = NFC café
        let decomposed = "caf\u{0065}\u{0301}";
        let result = normalize_for_negation_match(decomposed);
        assert_eq!(result, "café");
    }

    #[test]
    fn term_is_word_like_alphanumeric_and_underscore() {
        assert!(term_is_word_like("hello"));
        assert!(term_is_word_like("hello_world"));
        assert!(term_is_word_like("abc123"));
        assert!(term_is_word_like("_"));
    }

    #[test]
    fn term_is_word_like_false_for_special_chars() {
        assert!(!term_is_word_like("hello.world"));
        assert!(!term_is_word_like("src/main"));
        assert!(!term_is_word_like("a-b"));
        assert!(!term_is_word_like("foo bar"));
    }

    #[test]
    fn term_is_word_like_empty_is_true() {
        // all chars satisfy predicate vacuously
        assert!(term_is_word_like(""));
    }

    #[test]
    fn is_word_char_boundaries() {
        assert!(is_word_char('a'));
        assert!(is_word_char('Z'));
        assert!(is_word_char('5'));
        assert!(is_word_char('_'));
        assert!(!is_word_char(' '));
        assert!(!is_word_char('.'));
        assert!(!is_word_char('-'));
        assert!(!is_word_char('/'));
    }

    #[test]
    fn contains_term_with_word_boundaries_exact_match() {
        assert!(contains_term_with_word_boundaries("hello world", "hello"));
        assert!(contains_term_with_word_boundaries("hello world", "world"));
    }

    #[test]
    fn contains_term_with_word_boundaries_rejects_substring() {
        assert!(!contains_term_with_word_boundaries("theorem", "he"));
        assert!(!contains_term_with_word_boundaries("unhelpful", "help"));
    }

    #[test]
    fn contains_term_with_word_boundaries_punctuation_boundary() {
        assert!(contains_term_with_word_boundaries("(hello) world", "hello"));
        assert!(contains_term_with_word_boundaries("say hello!", "hello"));
    }

    #[test]
    fn contains_term_with_word_boundaries_start_and_end() {
        assert!(contains_term_with_word_boundaries("he", "he"));
        assert!(contains_term_with_word_boundaries("he said", "he"));
        assert!(contains_term_with_word_boundaries("said he", "he"));
    }

    #[test]
    fn contains_negative_term_word_like_uses_boundaries() {
        assert!(!contains_negative_term("the theorem proves it", "he"));
        assert!(contains_negative_term("he went home", "he"));
    }

    #[test]
    fn contains_negative_term_non_word_uses_substring() {
        assert!(contains_negative_term(
            "path=/workspace/src/main.rs",
            "src/main.rs"
        ));
    }

    #[test]
    fn find_negative_match_empty_term_skipped() {
        let mut parsed = ParsedQuery::parse("query");
        parsed.negative_terms.push(String::new());
        let exclusions = to_exclusions(&parsed);
        let result = find_negative_match("any document text", &exclusions);
        assert!(result.is_none());
    }

    #[test]
    fn find_negative_match_phrase_substring() {
        let parsed = ParsedQuery::parse(r#"query NOT "danger zone""#);
        let exclusions = to_exclusions(&parsed);
        let matched = find_negative_match("entering the danger zone now", &exclusions);
        assert_eq!(matched, Some("danger zone".to_owned()));
    }

    #[test]
    fn find_negative_match_phrase_not_found() {
        let parsed = ParsedQuery::parse(r#"query NOT "exact phrase""#);
        let exclusions = to_exclusions(&parsed);
        let matched = find_negative_match("different text entirely", &exclusions);
        assert!(matched.is_none());
    }

    #[test]
    fn should_exclude_document_returns_false_when_text_fn_returns_none() {
        let parsed = ParsedQuery::parse("query -unsafe");
        let exclusions = to_exclusions(&parsed);
        let result = should_exclude_document("doc-1", &exclusions, &|_| None, "test");
        assert!(!result);
    }

    #[test]
    fn should_exclude_document_returns_true_when_text_matches_negation() {
        let parsed = ParsedQuery::parse("query -unsafe");
        let exclusions = to_exclusions(&parsed);
        let result = should_exclude_document(
            "doc-1",
            &exclusions,
            &|_| Some("unsafe code".to_owned()),
            "test",
        );
        assert!(result);
    }

    #[test]
    fn should_exclude_document_returns_false_when_text_doesnt_match() {
        let parsed = ParsedQuery::parse("query -unsafe");
        let exclusions = to_exclusions(&parsed);
        let result = should_exclude_document(
            "doc-1",
            &exclusions,
            &|_| Some("safe code".to_owned()),
            "test",
        );
        assert!(!result);
    }

    #[test]
    fn filter_scored_results_by_negations_empty_input() {
        let parsed = ParsedQuery::parse("query -foo");
        let exclusions = to_exclusions(&parsed);
        let results = filter_scored_results_by_negations(vec![], &exclusions, &|_| None, "test");
        assert!(results.is_empty());
    }

    #[test]
    fn filter_vector_hits_by_negations_empty_input() {
        let parsed = ParsedQuery::parse("query -foo");
        let exclusions = to_exclusions(&parsed);
        let results = filter_vector_hits_by_negations(vec![], &exclusions, &|_| None, "test");
        assert!(results.is_empty());
    }

    #[test]
    fn builder_with_lexical_sets_lexical() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_lexical(Arc::new(StubLexical));
        let debug = format!("{searcher:?}");
        assert!(debug.contains("has_lexical: true"));
    }

    #[test]
    fn builder_without_lexical() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        let debug = format!("{searcher:?}");
        assert!(debug.contains("has_lexical: false"));
    }

    #[test]
    fn builder_with_host_adapter_shows_in_debug() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let adapter = Arc::new(RecordingHostAdapter::new("test-project"));
        let searcher =
            TwoTierSearcher::new(index, fast, TwoTierConfig::default()).with_host_adapter(adapter);
        let debug = format!("{searcher:?}");
        assert!(debug.contains("has_host_adapter: true"));
    }

    #[test]
    fn builder_with_reranker_shows_in_debug() {
        struct DummyReranker;
        impl Reranker for DummyReranker {
            fn rerank<'a>(
                &'a self,
                _cx: &'a Cx,
                _query: &'a str,
                _docs: &'a [frankensearch_core::traits::RerankDocument],
            ) -> SearchFuture<'a, Vec<frankensearch_core::traits::RerankScore>> {
                Box::pin(async { Ok(vec![]) })
            }
            fn id(&self) -> &str {
                "dummy"
            }
            fn model_name(&self) -> &str {
                "dummy-reranker"
            }
        }
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_reranker(Arc::new(DummyReranker));
        let debug = format!("{searcher:?}");
        assert!(debug.contains("has_reranker: true"));
    }

    #[test]
    fn should_run_quality_false_when_fast_only() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let config = TwoTierConfig {
            fast_only: true,
            ..TwoTierConfig::default()
        };
        let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
        assert!(!searcher.should_run_quality());
    }

    #[test]
    fn should_run_quality_false_when_no_quality_embedder() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        assert!(!searcher.should_run_quality());
    }

    #[test]
    fn should_run_quality_true_when_quality_embedder_and_not_fast_only() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("quality", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);
        assert!(searcher.should_run_quality());
    }

    #[test]
    fn live_search_stream_initially_empty() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        let health = searcher.live_search_stream_health();
        assert_eq!(health.emitted_total, 0);
        assert_eq!(health.buffered, 0);
        let drained = searcher.drain_live_search_stream(10);
        assert!(drained.is_empty());
    }

    #[test]
    fn debug_shows_quality_embedder_id() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let quality = Arc::new(StubEmbedder::new("my-quality", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);
        let debug = format!("{searcher:?}");
        assert!(debug.contains("my-quality"));
    }

    #[test]
    fn debug_shows_none_quality_when_not_set() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
        let debug = format!("{searcher:?}");
        assert!(debug.contains("quality_embedder: None"));
    }

    #[test]
    fn search_collect_with_text_returns_refined_when_quality_available() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_test_index(4);
            let fast = Arc::new(StubEmbedder::new("fast", 4));
            let quality = Arc::new(StubEmbedder::new("quality", 4));
            let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality);

            let (results, metrics) = searcher
                .search_collect_with_text(&cx, "test", 5, |_| None)
                .await
                .unwrap();

            assert!(!results.is_empty());
            assert!(metrics.quality_embed_ms > 0.0);
            // When quality refinement succeeds, results should contain quality scores
            assert!(results.iter().any(|r| r.quality_score.is_some()));
        });
    }

    #[test]
    fn with_runtime_metrics_collector_replaces_default() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let custom_collector = Arc::new(RuntimeMetricsCollector::default());
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_runtime_metrics_collector(custom_collector.clone());

        // Verify via Arc pointer equality
        assert!(Arc::ptr_eq(
            &searcher.runtime_metrics_collector,
            &custom_collector
        ));
    }

    #[test]
    fn with_live_search_stream_emitter_replaces_default() {
        let index = build_test_index(4);
        let fast = Arc::new(StubEmbedder::new("fast", 4));
        let custom_emitter = Arc::new(LiveSearchStreamEmitter::default());
        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_live_search_stream_emitter(custom_emitter.clone());

        assert!(Arc::ptr_eq(
            &searcher.live_search_stream_emitter,
            &custom_emitter
        ));
    }

    // ─── bd-3a7q tests end ───

    #[test]
    #[ignore = "performance probe"]
    fn perf_probe_phase2_lookup_map_vs_linear_scan() {
        let doc_count = std::env::var("PHASE2_LOOKUP_PERF_DOCS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(30_000);
        let iterations = std::env::var("PHASE2_LOOKUP_PERF_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(80);

        let (fast_hits, quality_hits, blended) = build_phase2_lookup_fixture(doc_count);
        let fast_scores_by_doc: HashMap<&str, f32> = fast_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();
        let quality_scores_by_doc: HashMap<&str, f32> = quality_hits
            .iter()
            .map(|hit| (hit.doc_id.as_str(), hit.score))
            .collect();

        let mut linear_checksum = 0.0_f32;
        let linear_start = std::time::Instant::now();
        for _ in 0..iterations {
            linear_checksum += std::hint::black_box(phase2_lookup_checksum_linear(
                &blended,
                &fast_hits,
                &quality_hits,
            ));
        }
        let linear_ms = linear_start.elapsed().as_secs_f64() * 1000.0;

        let mut mapped_checksum = 0.0_f32;
        let mapped_start = std::time::Instant::now();
        for _ in 0..iterations {
            mapped_checksum += std::hint::black_box(phase2_lookup_checksum_mapped(
                &blended,
                &fast_scores_by_doc,
                &quality_scores_by_doc,
            ));
        }
        let mapped_ms = mapped_start.elapsed().as_secs_f64() * 1000.0;

        let checksum_diff = (linear_checksum - mapped_checksum).abs();
        assert!(
            checksum_diff <= 0.01_f32,
            "lookup checksum mismatch: linear={linear_checksum} mapped={mapped_checksum}"
        );
        println!(
            "PHASE2_LOOKUP_PERF map_ms={mapped_ms:.3} linear_ms={linear_ms:.3} speedup={:.3} doc_count={doc_count} iterations={iterations}",
            if mapped_ms > 0.0 {
                linear_ms / mapped_ms
            } else {
                f64::INFINITY
            }
        );
    }
}
