//! Federated search across multiple independent [`TwoTierSearcher`] instances.
//!
//! A single query fans out to multiple indices, then gathered results are fused
//! into one ranked list. Fusion accumulates contributions in index-name order,
//! not response order, so scheduling cannot change scores for the same replies.

use std::future::Future;
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use ahash::{AHashMap, AHashSet};
use asupersync::Cx;
use asupersync::time::timeout;
use frankensearch_core::filter::SearchFilter;
use frankensearch_core::{ScoreSource, ScoredResult, SearchError, SearchPhase, SearchResult};
use tracing::{debug, warn};

use crate::normalize::{NormalizationMethod, normalize_scores_with_method};
use crate::searcher::TwoTierSearcher;

const DEFAULT_FEDERATED_RRF_K: f64 = 60.0;

/// Fusion methods supported by [`FederatedSearcher`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FederatedFusion {
    /// Reciprocal Rank Fusion across indices.
    ///
    /// Score contribution:
    /// `weight / (k + rank + 1)`.
    Rrf { k: f64 },

    /// Weighted sum of normalized per-index scores.
    WeightedScore {
        /// Normalization applied independently per index before weighted sum.
        normalization: NormalizationMethod,
    },

    /// `CombMNZ`: weighted normalized sum multiplied by appearance count.
    ///
    /// `comb_mnz = weighted_sum * count(indices_containing_doc)`.
    CombMnz {
        /// Normalization applied independently per index before weighted sum.
        normalization: NormalizationMethod,
    },
}

impl Default for FederatedFusion {
    fn default() -> Self {
        Self::Rrf {
            k: DEFAULT_FEDERATED_RRF_K,
        }
    }
}

/// Configuration for federated search behavior.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// How shard results are fused.
    pub fusion_method: FederatedFusion,
    /// Per-index timeout budget in milliseconds.
    pub per_index_timeout_ms: u64,
    /// Minimum number of indices that must return a valid result phase — a
    /// FLOOR enforced after the gather completes. This includes a shard whose
    /// optional refinement exceeded its deadline after publishing results.
    /// Fewer responses fail with [`SearchError::FederatedInsufficientResponses`].
    /// This knob never stops the gather early; that is `wait_for_indices`.
    pub min_indices: usize,
    /// Candidate multiplier per index before global fusion.
    pub candidate_pool_factor: usize,
    /// Maximum number of indices to query.
    pub max_indices: usize,
    /// Latency/recall trade-off: stop gathering (cancelling still-in-flight
    /// shards) once this many shards have responded successfully. `usize::MAX`
    /// (the default) and `0` both mean "wait for every dispatched shard"
    /// (bounded per shard by `per_index_timeout_ms`). Cancelled shards are
    /// reported in [`FederatedCoverage::cancelled_in_flight`] (bd-3zh67).
    pub wait_for_indices: usize,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            fusion_method: FederatedFusion::default(),
            per_index_timeout_ms: 500,
            min_indices: 1,
            candidate_pool_factor: 3,
            max_indices: usize::MAX,
            wait_for_indices: usize::MAX,
        }
    }
}

#[derive(Debug, Clone)]
struct FederatedIndex {
    name: String,
    searcher: Arc<TwoTierSearcher>,
    weight: f32,
}

#[derive(Debug)]
struct ShardResult {
    name: String,
    weight: f32,
    hits: Vec<ScoredResult>,
}

#[derive(Debug)]
enum ShardCompletion {
    Completed {
        shard: ShardResult,
        refinement_timed_out: bool,
    },
    Failed {
        index: String,
        error: SearchError,
    },
    TimedOut {
        index: String,
    },
    Cancelled {
        phase: String,
        reason: String,
    },
}

type ShardFuture<'a> = Pin<Box<dyn Future<Output = ShardCompletion> + Send + 'a>>;

#[derive(Debug, Clone)]
struct AggregateDoc {
    template: ScoredResult,
    primary_index: String,
    primary_rank: usize,
    primary_contribution: f32,
    fused_score: f32,
    /// Interned shard ids (rank in the sorted distinct-shard-name list) this doc
    /// appeared in. Was a `BTreeSet<String>` cloning a shard-name `String` on every
    /// `accumulate_doc` call (thousands per fuse); interning to integer ids and
    /// deduping once at output (`into_ranked_hits`) drops that per-call churn.
    /// Ids are the sorted-name rank, so sorting them ascending == sorting the names
    /// lexicographically (the `BTreeSet<String>` iteration order the output kept).
    appeared_in: Vec<u32>,
}

/// A fused hit returned by federated search.
#[derive(Debug, Clone)]
pub struct FederatedHit {
    /// The fused result payload.
    pub result: ScoredResult,
    /// Index where the strongest contribution came from.
    pub source_index: String,
    /// Rank in `source_index` (0-based).
    pub source_rank: usize,
    /// All indices where this document appeared.
    pub appeared_in: Vec<String>,
}

/// A shard that returned an error during a federated query.
#[derive(Debug, Clone)]
pub struct FederatedShardError {
    /// Index name as registered via `add_index`.
    pub index: String,
    /// Rendered shard error (the underlying `SearchError` display).
    pub error: String,
}

/// Per-query record of which shards were dispatched and how each one ended.
///
/// Partial federation was previously invisible: shard failures and timeouts
/// were swallowed into `tracing::warn!` and a healthy-looking `Ok(hits)` came
/// back from whatever subset answered (bd-3zh67 HALF B). Callers that care
/// about recall must check [`FederatedCoverage::is_complete`].
#[derive(Debug, Clone, Default)]
pub struct FederatedCoverage {
    /// Shards dispatched for this query (bounded by `max_indices`), in
    /// registration order.
    pub queried: Vec<String>,
    /// Shards whose results are included in the fused output.
    pub answered: Vec<String>,
    /// Shards that returned an error.
    pub failed: Vec<FederatedShardError>,
    /// Shards that exceeded `per_index_timeout_ms` before publishing any results.
    pub timed_out: Vec<String>,
    /// Answered shards whose optional quality refinement or final ranking
    /// exceeded the deadline. Their last published result phase is included
    /// in fusion and counts toward `min_indices`; no unfinished phase is used.
    pub refinement_timed_out: Vec<String>,
    /// Shards still in flight when the gather stopped early because
    /// `wait_for_indices` was satisfied.
    pub cancelled_in_flight: Vec<String>,
}

impl FederatedCoverage {
    /// True when every dispatched shard completed within its deadline.
    ///
    /// A shard with valid fast results but unfinished refinement contributes
    /// to the answer while keeping this completeness signal false.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.answered.len() == self.queried.len() && self.refinement_timed_out.is_empty()
    }
}

/// Fused hits plus the shard coverage that produced them.
#[derive(Debug, Clone)]
pub struct FederatedResponse {
    /// Globally fused, truncated-to-limit hits.
    pub hits: Vec<FederatedHit>,
    /// Which shards were queried / answered / failed / timed out / were
    /// cancelled for this query.
    pub coverage: FederatedCoverage,
}

/// Multi-index search orchestrator with scatter-gather fusion.
#[derive(Debug, Default)]
pub struct FederatedSearcher {
    indices: Vec<FederatedIndex>,
    config: FederatedConfig,
}

impl FederatedSearcher {
    /// Create a federated searcher with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the federated config.
    #[must_use]
    pub const fn with_config(mut self, config: FederatedConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an index by name with a fusion weight.
    ///
    /// `weight <= 0.0` effectively disables score contribution from this index.
    /// Names must be unique among dispatched indices: they identify coverage,
    /// fusion membership and the primary source. Search rejects duplicates
    /// before starting any shard work.
    #[must_use]
    pub fn add_index(
        mut self,
        name: impl Into<String>,
        searcher: Arc<TwoTierSearcher>,
        weight: f32,
    ) -> Self {
        self.indices.push(FederatedIndex {
            name: name.into(),
            searcher,
            weight,
        });
        self
    }

    /// Number of configured indices.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true when no indices are configured.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    fn validate_shard_names(&self) -> SearchResult<()> {
        let mut names = AHashSet::with_capacity(self.indices.len().min(self.config.max_indices));
        for index in self.indices.iter().take(self.config.max_indices) {
            if !names.insert(index.name.as_str()) {
                return Err(SearchError::InvalidConfig {
                    field: "federated.indices".to_owned(),
                    value: "duplicate_name".to_owned(),
                    reason:
                        "dispatched index names must be unique for unambiguous fusion and coverage"
                            .to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Execute federated search using a shared document-text provider.
    ///
    /// `text_fn` resolves global document IDs for exclusion filtering and
    /// reranking. For separate per-index stores, use
    /// [`Self::search_with_index_text`] to receive the originating index name.
    ///
    /// # Errors
    ///
    /// Same as [`Self::search_with_index_text`].
    pub async fn search<F>(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        text_fn: F,
    ) -> SearchResult<FederatedResponse>
    where
        F: Fn(&str) -> Option<String> + Send + Sync,
    {
        self.search_with_index_text(cx, query, limit, |_index: &str, doc_id: &str| {
            text_fn(doc_id)
        })
        .await
    }

    /// Execute federated search with one request-scoped document filter.
    ///
    /// Each shard applies `filter` within its own search, using that shard's
    /// metadata and filter semantics before the federated cutoff. Rejected
    /// documents never reach text hydration or reranking. Each shard resolves
    /// metadata independently, even for equal document IDs.
    ///
    /// `text_fn` resolves global document IDs. Use
    /// [`Self::search_with_index_filter`] when filters or text stores differ
    /// between indices. Both the filter and provider may borrow request data.
    ///
    /// # Errors
    ///
    /// Same as [`Self::search_with_index_text`].
    pub async fn search_with_filter<F>(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        filter: Option<&dyn SearchFilter>,
        text_fn: F,
    ) -> SearchResult<FederatedResponse>
    where
        F: Fn(&str) -> Option<String> + Send + Sync,
    {
        self.search_with_index_filter(
            cx,
            query,
            limit,
            |_index| filter,
            |_index, doc_id| text_fn(doc_id),
        )
        .await
    }

    /// Execute federated search with text resolved from each originating index.
    ///
    /// `text_fn(index_name, doc_id)` receives the exact name registered with
    /// [`Self::add_index`]. Each shard's exclusion filtering and optional
    /// reranking use its own bound provider, even when multiple indices return
    /// the same document ID. The callback may borrow request-scoped stores; it
    /// need not be `'static`. It is not called for undispatched indices.
    ///
    /// Document IDs still denote global fusion identity: equal IDs merge into
    /// one hit. Applications with unrelated index-local IDs must namespace IDs
    /// when indexing; selecting a text store does not change deduplication.
    ///
    /// The gather waits for every dispatched shard (each bounded by
    /// `per_index_timeout_ms`) unless `wait_for_indices` requests an early
    /// stop. `min_indices` is a floor validated after the gather. When a shard
    /// publishes results before its deadline, those results survive a timeout
    /// of its optional quality or final ranking stage; coverage identifies
    /// that incomplete refinement. A timeout before any phase has no results
    /// to retain. Cancellation and returned errors never take this fallback.
    /// Contributions are accumulated in index-name order, making scores and
    /// cutoff membership independent of completion order for the same shard
    /// replies. Explicit early stopping can still change which shards reply.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` before dispatch for duplicate names.
    /// Returns `SearchError::Cancelled` when cancellation is requested via `cx`.
    /// Returns the first shard error when no shard completes successfully.
    /// Returns `SearchError::FederatedInsufficientResponses` when fewer than
    /// `min_indices` shards return a valid result phase.
    pub async fn search_with_index_text<F>(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        text_fn: F,
    ) -> SearchResult<FederatedResponse>
    where
        F: Fn(&str, &str) -> Option<String> + Send + Sync,
    {
        self.search_with_index_filter(cx, query, limit, |_index| None, text_fn)
            .await
    }

    /// Execute federated search with filters and text bound to each index.
    ///
    /// `filter_fn(index_name)` is called once per dispatched index and returns
    /// its request-scoped filter, or `None` for an unfiltered shard. Filtering
    /// runs before that shard's candidate cutoff, text hydration and reranking,
    /// using that shard's own metadata and filter semantics. The filter and
    /// both callbacks may borrow request data and need not be `'static`.
    ///
    /// `text_fn(index_name, doc_id)` has the same source binding as
    /// [`Self::search_with_index_text`]. Equal document IDs still denote one
    /// global fusion identity, but only shards whose filter admits the document
    /// contribute its score or appear in [`FederatedHit::appeared_in`]. An
    /// answered shard with no matching documents still counts toward coverage.
    /// Neither callback is called for an undispatched index or a no-op request.
    ///
    /// # Errors
    ///
    /// Same as [`Self::search_with_index_text`]. Duplicate dispatched names are
    /// rejected before either callback runs. Per-shard metadata failures
    /// follow the ordinary shard error and minimum-response policy.
    pub async fn search_with_index_filter<'filter, F, P>(
        &self,
        cx: &Cx,
        query: &str,
        limit: usize,
        filter_fn: P,
        text_fn: F,
    ) -> SearchResult<FederatedResponse>
    where
        F: Fn(&str, &str) -> Option<String> + Send + Sync,
        P: Fn(&str) -> Option<&'filter dyn SearchFilter> + Send + Sync,
    {
        if query.is_empty() || limit == 0 || self.indices.is_empty() {
            return Ok(FederatedResponse {
                hits: Vec::new(),
                coverage: FederatedCoverage::default(),
            });
        }
        self.validate_shard_names()?;

        let candidate_pool_factor = self.config.candidate_pool_factor.max(1);
        let per_index_limit = limit.saturating_mul(candidate_pool_factor);
        let timeout_budget = Duration::from_millis(self.config.per_index_timeout_ms);
        let (shard_results, coverage) = self
            .collect_shard_results(
                cx,
                query,
                per_index_limit,
                timeout_budget,
                &filter_fn,
                &text_fn,
            )
            .await?;

        if shard_results.len() < self.config.min_indices {
            return Err(SearchError::FederatedInsufficientResponses {
                required: self.config.min_indices,
                received: shard_results.len(),
            });
        }

        let mut hits = match self.config.fusion_method {
            FederatedFusion::Rrf { k } => fuse_rrf(&shard_results, k),
            FederatedFusion::WeightedScore { normalization } => {
                fuse_weighted(&shard_results, normalization, false)
            }
            FederatedFusion::CombMnz { normalization } => {
                fuse_weighted(&shard_results, normalization, true)
            }
        };

        hits.truncate(limit);
        Ok(FederatedResponse { hits, coverage })
    }

    #[allow(clippy::too_many_lines)]
    async fn collect_shard_results<'filter, F, P>(
        &self,
        cx: &Cx,
        query: &str,
        per_index_limit: usize,
        timeout_budget: Duration,
        filter_fn: &P,
        text_fn: &F,
    ) -> SearchResult<(Vec<ShardResult>, FederatedCoverage)>
    where
        F: Fn(&str, &str) -> Option<String> + Send + Sync,
        P: Fn(&str) -> Option<&'filter dyn SearchFilter> + Send + Sync,
    {
        let mut coverage = FederatedCoverage {
            queried: self
                .indices
                .iter()
                .take(self.config.max_indices)
                .map(|index| index.name.clone())
                .collect(),
            ..FederatedCoverage::default()
        };
        let mut pending: Vec<ShardFuture<'_>> = self
            .indices
            .iter()
            .take(self.config.max_indices)
            .map(|index| {
                let index_name = index.name.clone();
                let index_weight = index.weight;
                let searcher = Arc::clone(&index.searcher);
                let filter = filter_fn(index.name.as_str());
                Box::pin(async move {
                    let timeout_start = cx.now();
                    let mut last_results = None;
                    let response = {
                        // Bind before the shard starts: exclusion and rerank
                        // lookups must never hydrate from another index's store.
                        let shard_text = |doc_id: &str| text_fn(index_name.as_str(), doc_id);
                        let future = Box::pin(searcher.search_with_filter(
                            cx,
                            query,
                            per_index_limit,
                            filter,
                            shard_text,
                            |phase| match phase {
                                SearchPhase::Initial { results, .. }
                                | SearchPhase::Refined { results, .. }
                                | SearchPhase::Reranked { results, .. } => {
                                    last_results = Some(results);
                                }
                                SearchPhase::RefinementFailed { .. } => {}
                            },
                        ));
                        timeout(timeout_start, timeout_budget, future).await
                    };
                    // The timeout has dropped the unfinished future. A provider
                    // may request cancellation during that drop; neither a late
                    // deadline nor already-published hits may hide it.
                    if let Err(error) = cx.checkpoint() {
                        return ShardCompletion::Cancelled {
                            phase: "federated_shard_to_result".to_owned(),
                            reason: cx
                                .cancel_reason()
                                .map_or_else(|| error.to_string(), |reason| reason.to_string()),
                        };
                    }
                    match response {
                        Ok(Ok(_metrics)) => ShardCompletion::Completed {
                            shard: ShardResult {
                                name: index_name,
                                weight: index_weight,
                                hits: last_results.unwrap_or_default(),
                            },
                            refinement_timed_out: false,
                        },
                        Ok(Err(SearchError::Cancelled { phase, reason })) => {
                            ShardCompletion::Cancelled { phase, reason }
                        }
                        Ok(Err(error)) => ShardCompletion::Failed {
                            index: index_name,
                            error,
                        },
                        Err(_elapsed) => match last_results {
                            Some(hits) => ShardCompletion::Completed {
                                shard: ShardResult {
                                    name: index_name,
                                    weight: index_weight,
                                    hits,
                                },
                                refinement_timed_out: true,
                            },
                            None => ShardCompletion::TimedOut { index: index_name },
                        },
                    }
                }) as ShardFuture<'_>
            })
            .collect();

        let mut shard_results = Vec::new();
        let mut first_shard_error: Option<SearchError> = None;
        while !pending.is_empty() {
            let ready_batch = poll_fn(|task_cx| {
                let mut ready = Vec::new();
                let mut idx = 0;
                while idx < pending.len() {
                    let poll = pending[idx].as_mut().poll(task_cx);
                    if let Poll::Ready(completion) = poll {
                        ready.push(completion);
                        drop(pending.swap_remove(idx));
                    } else {
                        idx += 1;
                    }
                }
                if ready.is_empty() {
                    Poll::Pending
                } else {
                    Poll::Ready(ready)
                }
            })
            .await;

            for completion in ready_batch {
                match completion {
                    ShardCompletion::Completed {
                        shard,
                        refinement_timed_out,
                    } => {
                        debug!(
                            index = %shard.name,
                            hit_count = shard.hits.len(),
                            refinement_timed_out,
                            "federated shard results collected"
                        );
                        if refinement_timed_out {
                            coverage.refinement_timed_out.push(shard.name.clone());
                        }
                        coverage.answered.push(shard.name.clone());
                        shard_results.push(shard);
                    }
                    ShardCompletion::Cancelled { phase, reason } => {
                        return Err(SearchError::Cancelled { phase, reason });
                    }
                    ShardCompletion::Failed { index, error } => {
                        warn!(
                            index = %index,
                            error = %error,
                            "federated shard search failed; continuing with remaining indices"
                        );
                        coverage.failed.push(FederatedShardError {
                            index,
                            error: error.to_string(),
                        });
                        if first_shard_error.is_none() {
                            first_shard_error = Some(error);
                        }
                    }
                    ShardCompletion::TimedOut { index } => {
                        warn!(
                            index = %index,
                            timeout_ms = self.config.per_index_timeout_ms,
                            "federated shard timed out; continuing with remaining indices"
                        );
                        coverage.timed_out.push(index);
                    }
                }
            }

            // bd-3zh67: the early stop is gated on `wait_for_indices` alone.
            // `min_indices` is a floor (validated by the caller after the
            // gather) and must never cancel in-flight shards.
            if self.config.wait_for_indices > 0
                && shard_results.len() >= self.config.wait_for_indices
            {
                break;
            }
        }

        // Anything dropped from `pending` on the early break was cancelled
        // in flight: dispatched, but neither answered nor failed nor timed out.
        if !pending.is_empty() {
            drop(pending);
            coverage.cancelled_in_flight = coverage
                .queried
                .iter()
                .filter(|name| {
                    !coverage.answered.iter().any(|n| n == *name)
                        && !coverage.failed.iter().any(|f| &f.index == *name)
                        && !coverage.timed_out.iter().any(|n| n == *name)
                })
                .cloned()
                .collect();
        }

        // Early stopping also drops unfinished providers. Observe cancellation
        // requested by their teardown before returning any surviving shard hits.
        if let Err(error) = cx.checkpoint() {
            return Err(SearchError::Cancelled {
                phase: "federated_gather_to_fusion".to_owned(),
                reason: cx
                    .cancel_reason()
                    .map_or_else(|| error.to_string(), |reason| reason.to_string()),
            });
        }
        if shard_results.is_empty()
            && let Some(error) = first_shard_error
        {
            return Err(error);
        }

        Ok((shard_results, coverage))
    }
}

/// Floating-point addition is not associative. Sorting only the final hits
/// cannot repair a sum that changed with shard completion order, including a
/// changed top-k boundary. Keep every fusion method on one canonical traversal.
fn ordered_shards(shards: &[ShardResult]) -> Vec<&ShardResult> {
    let mut ordered: Vec<_> = shards.iter().collect();
    ordered.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    ordered
}

/// Intern the distinct shard names to their **sorted rank**. Returns the sorted
/// distinct names (indexable by id) and a `name → id` lookup. Because ids are the
/// sorted-name rank, sorting a doc's `appeared_in` ids ascending reproduces the
/// lexicographic name order the old `BTreeSet<String>` iteration gave — so the
/// public `FederatedHit::appeared_in` order is unchanged.
#[allow(clippy::cast_possible_truncation)]
fn intern_shard_names(shards: &[ShardResult]) -> (Vec<&str>, AHashMap<&str, u32>) {
    let mut names: Vec<&str> = shards.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    let name_to_id: AHashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i as u32))
        .collect();
    (names, name_to_id)
}

fn fuse_rrf(shards: &[ShardResult], k: f64) -> Vec<FederatedHit> {
    let k = sanitize_rrf_k(k);
    // aHash (not SipHash): the cross-shard merge keys on owned `doc_id` strings;
    // the DoS-resistant default hasher is the bottleneck here, and the sibling
    // single-node `rrf_fuse_with_graph` already uses `AHashMap`. Output is
    // unchanged — `into_ranked_hits` sorts by a strict total order (the final
    // `doc_id` tiebreak), so map drain order never reaches the result.
    let (names, name_to_id) = intern_shard_names(shards);
    let mut docs: AHashMap<String, AggregateDoc> = AHashMap::new();
    for shard in ordered_shards(shards) {
        // NaN.max(0.0) propagates NaN — guard explicitly.
        if !shard.weight.is_finite() || shard.weight <= 0.0 {
            continue;
        }
        let weight = shard.weight;
        let shard_id = name_to_id[shard.name.as_str()];

        for (rank, hit) in shard.hits.iter().enumerate() {
            let contribution = weight * rank_contribution(k, rank);
            accumulate_doc(
                &mut docs,
                hit,
                &shard.name,
                shard_id,
                rank,
                contribution,
                contribution,
            );
        }
    }

    into_ranked_hits(docs, &names, false)
}

fn fuse_weighted(
    shards: &[ShardResult],
    normalization: NormalizationMethod,
    apply_comb_mnz: bool,
) -> Vec<FederatedHit> {
    let (names, name_to_id) = intern_shard_names(shards);
    let mut docs: AHashMap<String, AggregateDoc> = AHashMap::new();

    for shard in ordered_shards(shards) {
        // NaN.max(0.0) propagates NaN — guard explicitly.
        if !shard.weight.is_finite() || shard.weight <= 0.0 || shard.hits.is_empty() {
            continue;
        }
        let weight = shard.weight;
        let shard_id = name_to_id[shard.name.as_str()];

        let raw_scores: Vec<f32> = shard.hits.iter().map(|hit| hit.score).collect();
        let normalized = normalize_scores_with_method(&raw_scores, normalization);
        for (rank, (hit, normalized_score)) in shard.hits.iter().zip(normalized).enumerate() {
            let contribution = weight * normalized_score.max(0.0);
            accumulate_doc(
                &mut docs,
                hit,
                &shard.name,
                shard_id,
                rank,
                contribution,
                contribution,
            );
        }
    }

    into_ranked_hits(docs, &names, apply_comb_mnz)
}

#[inline]
fn sanitize_rrf_k(k: f64) -> f64 {
    if k.is_finite() && k >= 0.0 {
        k
    } else {
        DEFAULT_FEDERATED_RRF_K
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn rank_contribution(k: f64, rank: usize) -> f32 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    (1.0 / (k + f64::from(rank_u32) + 1.0)) as f32
}

#[allow(clippy::too_many_arguments)]
fn accumulate_doc(
    docs: &mut AHashMap<String, AggregateDoc>,
    hit: &ScoredResult,
    shard_name: &str,
    shard_id: u32,
    rank: usize,
    contribution: f32,
    primary_signal: f32,
) {
    let entry = docs.entry(hit.doc_id.to_string()).or_insert_with(|| {
        let mut template = hit.clone();
        template.score = 0.0;
        AggregateDoc {
            template,
            primary_index: shard_name.to_owned(),
            primary_rank: rank,
            primary_contribution: primary_signal,
            fused_score: 0.0,
            appeared_in: Vec::new(),
        }
    });

    entry.fused_score += contribution;
    // Push the interned id (deduped once at output). A shard contributes a given
    // doc at most once, so duplicates only arise from a shard with a repeated
    // doc_id — `into_ranked_hits` sort+dedups, matching the old `BTreeSet` semantics.
    entry.appeared_in.push(shard_id);

    let update_primary = match primary_signal.total_cmp(&entry.primary_contribution) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            rank < entry.primary_rank
                || (rank == entry.primary_rank && shard_name < entry.primary_index.as_str())
        }
    };
    if update_primary {
        shard_name.clone_into(&mut entry.primary_index);
        entry.primary_rank = rank;
        entry.primary_contribution = primary_signal;
        entry.template = hit.clone();
    }
}

#[allow(clippy::cast_precision_loss)]
fn into_ranked_hits(
    mut docs: AHashMap<String, AggregateDoc>,
    names: &[&str],
    apply_comb_mnz: bool,
) -> Vec<FederatedHit> {
    let mut output: Vec<FederatedHit> = docs
        .drain()
        .map(|(_, mut aggregate)| {
            // Dedup the interned ids once here (the old `BTreeSet` deduped on
            // insert). Ascending id order == lexicographic name order, so the
            // materialized `appeared_in` names are sorted, as before.
            aggregate.appeared_in.sort_unstable();
            aggregate.appeared_in.dedup();
            let appearance_count = aggregate.appeared_in.len();
            if apply_comb_mnz {
                aggregate.fused_score *= appearance_count as f32;
            }
            aggregate.template.score = aggregate.fused_score;
            if appearance_count > 1 {
                aggregate.template.source = ScoreSource::Hybrid;
            }
            let appeared_in: Vec<String> = aggregate
                .appeared_in
                .iter()
                .map(|&id| names[id as usize].to_owned())
                .collect();
            FederatedHit {
                result: aggregate.template,
                source_index: aggregate.primary_index,
                source_rank: aggregate.primary_rank,
                appeared_in,
            }
        })
        .collect();

    // Unstable sort (no scratch alloc): the comparator is a strict total order
    // ending in a `doc_id` tiebreak, and `doc_id` is the unique map key, so no two
    // entries compare Equal — identical output to a stable sort. Same bit-identical
    // sibling fix as `blend_two_tier` / `rrf_fuse` (this cross-shard path is niche,
    // not separately benched).
    output.sort_unstable_by(|left, right| {
        right
            .result
            .score
            .total_cmp(&left.result.score)
            .then(right.appeared_in.len().cmp(&left.appeared_in.len()))
            .then(left.source_rank.cmp(&right.source_rank))
            .then_with(|| left.result.doc_id.cmp(&right.result.doc_id))
    });

    output
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};
    use frankensearch_core::{SearchError, TwoTierConfig};
    use frankensearch_index::TwoTierIndex;

    use super::{FederatedConfig, FederatedFusion, FederatedSearcher};
    use crate::normalize::NormalizationMethod;
    use crate::searcher::TwoTierSearcher;

    // Reserve the directory atomically: parallel tests share both PID and
    // TMPDIR, and a wall-clock timestamp is not a uniqueness primitive.
    // mkdtemp never reopens an existing directory as a writable fixture, and
    // the returned guard removes the directory when the test drops it,
    // panics included.
    fn reserve_index_fixture_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("frankensearch-federated-test-")
            .tempdir()
            .expect("reserve federated test fixture")
    }

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
        fn embed<'a>(
            &'a self,
            _cx: &'a asupersync::Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
            let dimension = self.dimension;
            Box::pin(async move {
                let mut vector = vec![0.0; dimension];
                if !vector.is_empty() {
                    vector[0] = 1.0;
                }
                Ok(vector)
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

    struct PendingEmbedder {
        id: &'static str,
        dimension: usize,
        started: AtomicU64,
        dropped: AtomicU64,
        cancel_on_drop: bool,
    }

    struct PendingEmbeddingDrop<'a> {
        provider: &'a PendingEmbedder,
        cx: &'a asupersync::Cx,
    }

    impl Drop for PendingEmbeddingDrop<'_> {
        fn drop(&mut self) {
            self.provider.dropped.fetch_add(1, Ordering::SeqCst);
            if self.provider.cancel_on_drop {
                self.cx.cancel_with(
                    asupersync::CancelKind::User,
                    Some("cancel while dropping timed-out federated inference"),
                );
            }
        }
    }

    impl PendingEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self {
                id,
                dimension,
                started: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
                cancel_on_drop: false,
            }
        }
    }

    impl Embedder for PendingEmbedder {
        fn embed<'a>(
            &'a self,
            cx: &'a asupersync::Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                let _drop = PendingEmbeddingDrop { provider: self, cx };
                self.started.fetch_add(1, Ordering::SeqCst);
                std::future::pending().await
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
            // ApiEmbedder keeps the searcher on its genuinely-async await path
            // so the shard stays pending until the federated per-index timeout
            // fires — the condition these fixtures exist to produce. The rayon
            // path would poll_immediate the pending embed into an instant
            // failure instead.
            ModelCategory::ApiEmbedder
        }
    }

    /// Completes successfully after yielding to the executor a fixed number of
    /// times. Forces its shard to finish in a later poll batch than a shard
    /// backed by [`StubEmbedder`], without needing timers — the deterministic
    /// stand-in for "healthy but slower shard".
    struct YieldingEmbedder {
        id: &'static str,
        dimension: usize,
        yields: u32,
    }

    impl YieldingEmbedder {
        const fn new(id: &'static str, dimension: usize, yields: u32) -> Self {
            Self {
                id,
                dimension,
                yields,
            }
        }
    }

    impl Embedder for YieldingEmbedder {
        fn embed<'a>(
            &'a self,
            _cx: &'a asupersync::Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
            let dimension = self.dimension;
            let total_yields = self.yields;
            Box::pin(async move {
                let mut remaining = total_yields;
                std::future::poll_fn(move |task_cx| {
                    if remaining == 0 {
                        std::task::Poll::Ready(())
                    } else {
                        remaining -= 1;
                        task_cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
                let mut vector = vec![0.0; dimension];
                if !vector.is_empty() {
                    vector[0] = 1.0;
                }
                Ok(vector)
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
            // ApiEmbedder routes through the searcher's genuinely-async await
            // path; the rayon path poll_immediates non-API embedders and would
            // convert the deliberate yields into an embed failure.
            ModelCategory::ApiEmbedder
        }
    }

    struct FailingEmbedder {
        id: &'static str,
        dimension: usize,
    }

    impl FailingEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self { id, dimension }
        }
    }

    impl Embedder for FailingEmbedder {
        fn embed<'a>(
            &'a self,
            _cx: &'a asupersync::Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                Err(SearchError::EmbeddingFailed {
                    model: self.id.to_owned(),
                    source: Box::new(io::Error::other("simulated embed failure")),
                })
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

    /// A searcher together with the guard owning its index directory; keep
    /// the guard bound for as long as the searcher is used.
    type SearcherFixture = (tempfile::TempDir, Arc<TwoTierSearcher>);

    fn build_index(records: &[(&str, &[f32])]) -> (tempfile::TempDir, Arc<TwoTierIndex>) {
        let dir = reserve_index_fixture_dir();
        let mut builder =
            TwoTierIndex::create(dir.path(), TwoTierConfig::default()).expect("create index");
        builder.set_fast_embedder_id("stub-fast");
        for (doc_id, vector) in records {
            builder
                .add_fast_record((*doc_id).to_owned(), vector)
                .expect("add record");
        }
        let index = Arc::new(builder.finish().expect("finish index"));
        (dir, index)
    }

    fn build_searcher(records: &[(&str, &[f32])]) -> SearcherFixture {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let (dir, index) = build_index(records);
        let embedder: Arc<dyn Embedder> = Arc::new(StubEmbedder::new("stub-fast", dimension));
        let searcher = Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ));
        (dir, searcher)
    }

    fn build_pending_searcher(records: &[(&str, &[f32])]) -> SearcherFixture {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let (dir, index) = build_index(records);
        let embedder: Arc<dyn Embedder> = Arc::new(PendingEmbedder::new("stub-pending", dimension));
        let searcher = Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ));
        (dir, searcher)
    }

    fn build_failing_searcher(records: &[(&str, &[f32])]) -> SearcherFixture {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let (dir, index) = build_index(records);
        let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder::new("stub-failing", dimension));
        let searcher = Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ));
        (dir, searcher)
    }

    fn build_yielding_searcher(records: &[(&str, &[f32])]) -> SearcherFixture {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let (dir, index) = build_index(records);
        let embedder: Arc<dyn Embedder> =
            Arc::new(YieldingEmbedder::new("stub-yielding", dimension, 32));
        let searcher = Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ));
        (dir, searcher)
    }

    fn build_refining_searcher(
        records: &[(&str, &[f32], &[f32])],
        quality: Arc<dyn Embedder>,
    ) -> SearcherFixture {
        let dimension = records.first().map_or(1, |(_, fast, _)| fast.len());
        let dir = reserve_index_fixture_dir();
        let mut builder =
            TwoTierIndex::create(dir.path(), TwoTierConfig::default()).expect("create index");
        builder.set_fast_embedder_id("stub-fast");
        builder.set_quality_embedder_id(quality.id());
        for (doc_id, fast, refined) in records {
            builder
                .add_fast_record((*doc_id).to_owned(), fast)
                .expect("add fast record");
            builder
                .add_quality_record((*doc_id).to_owned(), refined)
                .expect("add quality record");
        }
        let searcher = TwoTierSearcher::new(
            Arc::new(builder.finish().expect("finish index")),
            Arc::new(StubEmbedder::new("stub-fast", dimension)),
            TwoTierConfig {
                // The federation deadline, not the optional quality stage's
                // own timeout, owns the degradation exercised below.
                quality_timeout_ms: 60_000,
                quality_weight: 1.0,
                ..TwoTierConfig::default()
            },
        )
        .with_quality_embedder(quality);
        (dir, Arc::new(searcher))
    }

    /// bd-3zh67 red proof: an out-of-the-box `FederatedSearcher` must return
    /// results from EVERY healthy shard, not just whichever answered first.
    #[test]
    fn default_config_returns_results_from_all_healthy_shards() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_fast_dir, fast) = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let (_slow_dir, slow) = build_yielding_searcher(&[("doc-slow", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .add_index("fast", fast, 1.0)
                .add_index("slow", slow, 1.0);

            let response = federated.search(&cx, "query", 10, |_| None).await.unwrap();
            let ids: std::collections::BTreeSet<_> = response
                .hits
                .iter()
                .map(|hit| hit.result.doc_id.as_str())
                .collect();
            assert!(ids.contains("doc-fast"), "fast shard doc must be present");
            assert!(
                ids.contains("doc-slow"),
                "slow-but-healthy shard doc must be present: the default config \
                 must not cancel in-flight shards after the first reply (bd-3zh67)"
            );
            assert!(response.coverage.is_complete());
            assert_eq!(response.coverage.answered.len(), 2);
            assert!(response.coverage.cancelled_in_flight.is_empty());
        });
    }

    /// bd-3zh67 HALF B: failures and timeouts must be visible per query, and
    /// the first shard error must not be silently discarded just because some
    /// other shard answered.
    #[test]
    fn coverage_reports_failed_and_timed_out_shards() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_healthy_dir, healthy) = build_searcher(&[("doc-healthy", &[1.0, 0.0])]);
            let (_failing_dir, failing) = build_failing_searcher(&[("doc-failing", &[1.0, 0.0])]);
            let (_pending_dir, pending) = build_pending_searcher(&[("doc-pending", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: 80,
                    ..FederatedConfig::default()
                })
                .add_index("healthy", healthy, 1.0)
                .add_index("failing", failing, 1.0)
                .add_index("pending", pending, 1.0);

            let response = federated.search(&cx, "query", 10, |_| None).await.unwrap();

            assert_eq!(response.hits.len(), 1);
            assert_eq!(response.hits[0].result.doc_id, "doc-healthy");

            let coverage = &response.coverage;
            assert_eq!(
                coverage.queried,
                vec![
                    "healthy".to_owned(),
                    "failing".to_owned(),
                    "pending".to_owned()
                ]
            );
            assert_eq!(coverage.answered, vec!["healthy".to_owned()]);
            assert_eq!(coverage.failed.len(), 1, "failed shard must be recorded");
            assert_eq!(coverage.failed[0].index, "failing");
            assert!(
                !coverage.failed[0].error.is_empty(),
                "shard error text must be preserved for the caller"
            );
            assert_eq!(coverage.timed_out, vec!["pending".to_owned()]);
            assert!(coverage.cancelled_in_flight.is_empty());
            assert!(!coverage.is_complete());
        });
    }

    #[test]
    fn deadline_retains_initial_results_for_the_response_floor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_healthy_dir, healthy) = build_searcher(&[("doc-healthy", &[1.0, 0.0])]);
            let quality = Arc::new(PendingEmbedder::new("stub-quality", 2));
            let (_slow_dir, slow) =
                build_refining_searcher(&[("doc-slow", &[1.0, 0.0], &[1.0, 0.0])], quality.clone());
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: 80,
                    min_indices: 2,
                    ..FederatedConfig::default()
                })
                .add_index("healthy", healthy, 1.0)
                .add_index("slow-quality", slow, 1.0);

            let response = federated.search(&cx, "query", 10, |_| None).await.unwrap();
            let ids: std::collections::BTreeSet<_> = response
                .hits
                .iter()
                .map(|hit| hit.result.doc_id.as_str())
                .collect();
            assert_eq!(
                ids,
                std::collections::BTreeSet::from(["doc-healthy", "doc-slow"])
            );
            assert_eq!(response.coverage.answered.len(), 2);
            assert_eq!(
                response.coverage.refinement_timed_out,
                vec!["slow-quality".to_owned()]
            );
            assert!(response.coverage.timed_out.is_empty());
            assert!(response.coverage.failed.is_empty());
            assert!(response.coverage.cancelled_in_flight.is_empty());
            assert!(!response.coverage.is_complete());
            assert_eq!(quality.started.load(Ordering::SeqCst), 1);
            assert_eq!(quality.dropped.load(Ordering::SeqCst), 1);
            let retained = response
                .hits
                .iter()
                .find(|hit| hit.result.doc_id == "doc-slow")
                .unwrap();
            assert_eq!(retained.source_index, "slow-quality");
            assert_eq!(retained.result.fast_score, Some(1.0));
            assert_eq!(retained.result.quality_score, None);
        });
    }

    #[test]
    fn deadline_without_initial_cannot_manufacture_a_partial_shard_response() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Arc::new(PendingEmbedder::new("stub-pending", 2));
            let (_dir, index) = build_index(&[("doc", &[1.0, 0.0])]);
            let searcher = Arc::new(TwoTierSearcher::new(
                index,
                provider.clone(),
                TwoTierConfig::default(),
            ));
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: 80,
                    ..FederatedConfig::default()
                })
                .add_index("pending-fast", searcher, 1.0);
            let error = federated
                .search(&cx, "query", 1, |_| None)
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                SearchError::FederatedInsufficientResponses {
                    required: 1,
                    received: 0,
                }
            ));
            assert_eq!(provider.started.load(Ordering::SeqCst), 1);
            assert_eq!(provider.dropped.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn cancellation_during_timeout_drop_discards_retained_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let quality = Arc::new(PendingEmbedder {
                cancel_on_drop: true,
                ..PendingEmbedder::new("stub-quality", 2)
            });
            let (_dir, searcher) =
                build_refining_searcher(&[("doc", &[1.0, 0.0], &[1.0, 0.0])], quality.clone());
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: 80,
                    ..FederatedConfig::default()
                })
                .add_index("cancelled", searcher, 1.0);
            let error = federated
                .search(&cx, "query", 1, |_| None)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(quality.started.load(Ordering::SeqCst), 1);
            assert_eq!(quality.dropped.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn early_stop_observes_cancellation_from_unfinished_shard_drop() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_healthy_dir, healthy) = build_searcher(&[("healthy", &[1.0, 0.0])]);
            let (_pending_dir, index) = build_index(&[("pending", &[1.0, 0.0])]);
            let provider = Arc::new(PendingEmbedder {
                cancel_on_drop: true,
                ..PendingEmbedder::new("stub-pending", 2)
            });
            let pending = Arc::new(TwoTierSearcher::new(
                index,
                provider.clone(),
                TwoTierConfig::default(),
            ));
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    wait_for_indices: 1,
                    ..FederatedConfig::default()
                })
                // Poll the pending provider before the ready shard satisfies
                // the early-stop policy, so its teardown is part of gathering.
                .add_index("pending", pending, 1.0)
                .add_index("healthy", healthy, 1.0);

            let error = federated
                .search(&cx, "query", 1, |_| None)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::Cancelled { .. }));
            assert_eq!(provider.started.load(Ordering::SeqCst), 1);
            assert_eq!(provider.dropped.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn successful_refinement_replaces_fast_results_before_fusion() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_dir, searcher) = build_refining_searcher(
                &[
                    ("fast-winner", &[1.0, 0.0], &[0.0, 1.0]),
                    ("quality-winner", &[0.0, 1.0], &[1.0, 0.0]),
                ],
                Arc::new(StubEmbedder::new("stub-quality", 2)),
            );
            let federated = FederatedSearcher::new().add_index("complete", searcher, 1.0);
            let response = federated.search(&cx, "query", 1, |_| None).await.unwrap();
            assert_eq!(response.hits.len(), 1);
            assert_eq!(response.hits[0].result.doc_id, "quality-winner");
            assert_eq!(response.hits[0].result.quality_score, Some(1.0));
            assert!(response.coverage.refinement_timed_out.is_empty());
            assert!(response.coverage.is_complete());
        });
    }

    #[test]
    fn single_index_returns_ranked_hits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) =
                build_searcher(&[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.2, 0.0])]);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    fusion_method: FederatedFusion::Rrf { k: 60.0 },
                    ..FederatedConfig::default()
                })
                .add_index("primary", index, 1.0);

            let results = federated
                .search(&cx, "query", 2, |_| None)
                .await
                .unwrap()
                .hits;
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].result.doc_id, "doc-a");
            assert_eq!(results[0].source_index, "primary");
            assert_eq!(results[0].appeared_in, vec!["primary".to_owned()]);
        });
    }

    #[test]
    fn weighted_score_respects_index_weights() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_a_dir, index_a) =
                build_searcher(&[("shared", &[1.0, 0.0]), ("a-only", &[0.8, 0.0])]);
            let (_index_b_dir, index_b) =
                build_searcher(&[("b-only", &[1.0, 0.0]), ("shared", &[0.2, 0.0])]);

            let config = FederatedConfig {
                fusion_method: FederatedFusion::WeightedScore {
                    normalization: NormalizationMethod::MinMax,
                },
                ..FederatedConfig::default()
            };

            let prefer_a = FederatedSearcher::new()
                .with_config(config.clone())
                .add_index("a", Arc::clone(&index_a), 2.0)
                .add_index("b", Arc::clone(&index_b), 1.0);
            let prefer_b = FederatedSearcher::new()
                .with_config(config)
                .add_index("a", index_a, 1.0)
                .add_index("b", index_b, 2.0);

            let results_a = prefer_a
                .search(&cx, "query", 3, |_| None)
                .await
                .unwrap()
                .hits;
            let results_b = prefer_b
                .search(&cx, "query", 3, |_| None)
                .await
                .unwrap()
                .hits;

            assert_eq!(results_a[0].result.doc_id, "shared");
            assert_eq!(results_b[0].result.doc_id, "b-only");
        });
    }

    #[test]
    fn weighted_score_normalizes_disparate_shard_scales() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_large_scale_dir, large_scale) =
                build_searcher(&[("large-top", &[100.0, 0.0]), ("large-low", &[50.0, 0.0])]);
            let (_small_scale_dir, small_scale) =
                build_searcher(&[("small-top", &[1.0, 0.0]), ("small-low", &[0.5, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    fusion_method: FederatedFusion::WeightedScore {
                        normalization: NormalizationMethod::MinMax,
                    },
                    ..FederatedConfig::default()
                })
                .add_index("large", large_scale, 1.0)
                .add_index("small", small_scale, 1.0);

            let results = federated
                .search(&cx, "query", 4, |_| None)
                .await
                .unwrap()
                .hits;
            let top_ids: std::collections::BTreeSet<_> = results
                .iter()
                .take(2)
                .map(|hit| hit.result.doc_id.as_str())
                .collect();

            assert!(top_ids.contains("large-top"));
            assert!(top_ids.contains("small-top"));
            assert!(!top_ids.contains("large-low"));
        });
    }

    #[test]
    fn comb_mnz_boosts_multi_index_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // The low-relevance decoys point away from the query direction but
            // keep a nonzero norm: the index writer rejects zero-norm
            // embeddings outright (bd-tqhc), since they can never match.
            let (_index_a_dir, index_a) = build_searcher(&[
                ("a-only", &[1.0, 0.0]),
                ("shared", &[0.9, 0.0]),
                ("a-low", &[0.05, 1.0]),
            ]);
            let (_index_b_dir, index_b) = build_searcher(&[
                ("b-only", &[1.0, 0.0]),
                ("shared", &[0.9, 0.0]),
                ("b-low", &[0.05, 1.0]),
            ]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    fusion_method: FederatedFusion::CombMnz {
                        normalization: NormalizationMethod::MinMax,
                    },
                    ..FederatedConfig::default()
                })
                .add_index("a", index_a, 1.0)
                .add_index("b", index_b, 1.0);

            let results = federated
                .search(&cx, "query", 3, |_| None)
                .await
                .unwrap()
                .hits;
            assert_eq!(results[0].result.doc_id, "shared");
            assert_eq!(results[0].appeared_in.len(), 2);
        });
    }

    #[test]
    fn zero_weight_disables_index_contribution() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_a_dir, index_a) = build_searcher(&[("a-only", &[1.0, 0.0])]);
            let (_index_b_dir, index_b) = build_searcher(&[("b-only", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    fusion_method: FederatedFusion::WeightedScore {
                        normalization: NormalizationMethod::MinMax,
                    },
                    ..FederatedConfig::default()
                })
                .add_index("a", index_a, 1.0)
                .add_index("b", index_b, 0.0);

            let results = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .unwrap()
                .hits;
            assert!(results.iter().any(|hit| hit.result.doc_id == "a-only"));
            assert!(!results.iter().any(|hit| hit.result.doc_id == "b-only"));
        });
    }

    #[test]
    fn min_indices_enforced_when_shard_times_out() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            // A zero deadline expires before any shard is polled under current
            // timer semantics. A bounded positive budget lets the synchronous
            // shard complete while the intentionally pending shard times out.
            let timeout_ms = 80_u64;
            let (_fast_dir, fast) = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let (_pending_dir, pending) = build_pending_searcher(&[("doc-pending", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: timeout_ms,
                    min_indices: 2,
                    ..FederatedConfig::default()
                })
                .add_index("fast", fast, 1.0)
                .add_index("pending", pending, 1.0);

            let error = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .expect_err("insufficient shard responses should fail");

            assert!(matches!(
                error,
                SearchError::FederatedInsufficientResponses {
                    required: 2,
                    received: 1
                }
            ));
        });
    }

    #[test]
    fn failed_shard_does_not_abort_when_min_indices_met() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_healthy_dir, healthy) = build_searcher(&[("doc-healthy", &[1.0, 0.0])]);
            let (_failing_dir, failing) = build_failing_searcher(&[("doc-failing", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    min_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("healthy", healthy, 1.0)
                .add_index("failing", failing, 1.0);

            let results = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .unwrap()
                .hits;
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].result.doc_id, "doc-healthy");
            assert_eq!(results[0].appeared_in, vec!["healthy".to_owned()]);
        });
    }

    #[test]
    fn underlying_error_is_preserved_when_all_shards_fail() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_first_dir, first) = build_failing_searcher(&[("doc-first", &[1.0, 0.0])]);
            let (_second_dir, second) = build_failing_searcher(&[("doc-second", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    min_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("first", first, 1.0)
                .add_index("second", second, 1.0);

            let error = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .expect_err("all shard failures should preserve an underlying error");

            assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
        });
    }

    #[test]
    fn filtered_shard_can_yield_zero_hits_without_failing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_full_dir, full) = build_searcher(&[("doc-full", &[1.0, 0.0])]);
            let (_filtered_dir, filtered) = build_searcher(&[("doc-filtered", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    min_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("full", full, 1.0)
                .add_index("filtered", filtered, 1.0);

            let results = federated
                .search(&cx, "query -dropme", 10, |doc_id| {
                    if doc_id == "doc-filtered" {
                        Some("dropme".to_owned())
                    } else {
                        Some("keep".to_owned())
                    }
                })
                .await
                .unwrap()
                .hits;
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].result.doc_id, "doc-full");
            assert_eq!(results[0].appeared_in, vec!["full".to_owned()]);
        });
    }

    #[test]
    fn comb_mnz_tracks_all_source_indices_for_duplicate_doc() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_a_dir, index_a) =
                build_searcher(&[("shared", &[1.0, 0.0]), ("a-only", &[1.0, 0.0])]);
            let (_index_b_dir, index_b) =
                build_searcher(&[("shared", &[1.0, 0.0]), ("b-only", &[1.0, 0.0])]);
            let (_index_c_dir, index_c) =
                build_searcher(&[("shared", &[1.0, 0.0]), ("c-only", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    fusion_method: FederatedFusion::CombMnz {
                        normalization: NormalizationMethod::MinMax,
                    },
                    ..FederatedConfig::default()
                })
                .add_index("a", index_a, 1.0)
                .add_index("b", index_b, 1.0)
                .add_index("c", index_c, 1.0);

            let results = federated
                .search(&cx, "query", 5, |_| None)
                .await
                .unwrap()
                .hits;
            let shared = results
                .iter()
                .find(|hit| hit.result.doc_id == "shared")
                .expect("shared should be present");
            assert_eq!(shared.appeared_in, vec!["a", "b", "c"]);
            assert_eq!(results[0].result.doc_id, "shared");
        });
    }

    #[test]
    fn max_indices_limits_scatter_fanout() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_first_dir, first) = build_searcher(&[("doc-first", &[1.0, 0.0])]);
            let (_second_dir, second) = build_searcher(&[("doc-second", &[1.0, 0.0])]);
            let (_third_dir, third) = build_searcher(&[("doc-third", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    max_indices: 2,
                    fusion_method: FederatedFusion::WeightedScore {
                        normalization: NormalizationMethod::MinMax,
                    },
                    ..FederatedConfig::default()
                })
                .add_index("first", first, 1.0)
                .add_index("second", second, 1.0)
                .add_index("third", third, 1.0);

            let results = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .unwrap()
                .hits;
            let ids: std::collections::BTreeSet<_> = results
                .iter()
                .map(|hit| hit.result.doc_id.as_str())
                .collect();

            assert!(ids.contains("doc-first"));
            assert!(ids.contains("doc-second"));
            assert!(!ids.contains("doc-third"));
        });
    }

    #[test]
    fn empty_query_returns_empty_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) = build_searcher(&[("doc-a", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new().add_index("primary", index, 1.0);
            let results = federated.search(&cx, "", 10, |_| None).await.unwrap().hits;
            assert!(results.is_empty());
        });
    }

    #[test]
    fn zero_limit_returns_empty_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) = build_searcher(&[("doc-a", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new().add_index("primary", index, 1.0);
            let results = federated
                .search(&cx, "query", 0, |_| None)
                .await
                .unwrap()
                .hits;
            assert!(results.is_empty());
        });
    }

    #[test]
    fn no_indices_returns_empty_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let federated = FederatedSearcher::new();
            assert!(federated.is_empty());
            assert_eq!(federated.len(), 0);
            let results = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .unwrap()
                .hits;
            assert!(results.is_empty());
        });
    }

    #[test]
    fn federated_fusion_default_is_rrf() {
        let fusion = super::FederatedFusion::default();
        assert!(
            matches!(fusion, super::FederatedFusion::Rrf { k } if (k - 60.0).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn federated_config_defaults() {
        let config = FederatedConfig::default();
        assert_eq!(config.per_index_timeout_ms, 500);
        assert_eq!(config.min_indices, 1);
        assert_eq!(config.candidate_pool_factor, 3);
        assert_eq!(config.max_indices, usize::MAX);
        assert_eq!(
            config.wait_for_indices,
            usize::MAX,
            "default must wait for every dispatched shard (bd-3zh67)"
        );
    }

    #[test]
    fn rank_contribution_decreases_with_rank() {
        let c0 = super::rank_contribution(60.0, 0);
        let c1 = super::rank_contribution(60.0, 1);
        let c10 = super::rank_contribution(60.0, 10);
        assert!(c0 > c1);
        assert!(c1 > c10);
        assert!(c0 > 0.0);
    }

    #[test]
    fn rank_contribution_large_rank() {
        // Should not panic even with very large rank values
        let c = super::rank_contribution(60.0, usize::MAX);
        assert!(c >= 0.0);
        assert!(c.is_finite());
    }

    #[test]
    fn rank_contribution_invalid_k_falls_back_to_default() {
        let expected = super::rank_contribution(super::DEFAULT_FEDERATED_RRF_K, 0);
        for invalid_k in [f64::NAN, f64::INFINITY, -1.0, -100.0] {
            let sanitized = super::sanitize_rrf_k(invalid_k);
            let contribution = super::rank_contribution(sanitized, 0);
            assert!(
                (contribution - expected).abs() < f32::EPSILON,
                "invalid k={invalid_k} should fall back to default",
            );
        }
    }

    #[test]
    fn scatter_gather_runs_shard_timeouts_concurrently() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let timeout_ms = 80_u64;
            let (_fast_dir, fast) = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let (_pending_a_dir, pending_a) =
                build_pending_searcher(&[("doc-pending-a", &[1.0, 0.0])]);
            let (_pending_b_dir, pending_b) =
                build_pending_searcher(&[("doc-pending-b", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: timeout_ms,
                    min_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("pending-a", pending_a, 1.0)
                .add_index("pending-b", pending_b, 1.0)
                .add_index("fast", fast, 1.0);

            let start = Instant::now();
            let results = federated
                .search(&cx, "query", 10, |_| None)
                .await
                .unwrap()
                .hits;
            let elapsed = start.elapsed();

            assert!(
                results.iter().any(|hit| hit.result.doc_id == "doc-fast"),
                "fast shard result should survive pending shard timeouts"
            );
            assert!(
                elapsed < Duration::from_millis(130),
                "scatter-gather should bound latency near one timeout budget; elapsed={elapsed:?}"
            );
        });
    }

    /// bd-3zh67: the early stop is an explicit opt-in via `wait_for_indices`;
    /// `min_indices` (the floor) must not trigger it. Cancelled shards are
    /// visible in coverage.
    #[test]
    fn scatter_gather_returns_early_when_wait_for_indices_is_satisfied() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let timeout_ms = 200_u64;
            let (_fast_dir, fast) = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let (_pending_dir, pending) = build_pending_searcher(&[("doc-pending", &[1.0, 0.0])]);

            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: timeout_ms,
                    wait_for_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("pending", pending, 1.0)
                .add_index("fast", fast, 1.0);

            let start = Instant::now();
            let response = federated.search(&cx, "query", 10, |_| None).await.unwrap();
            let elapsed = start.elapsed();

            assert!(
                response
                    .hits
                    .iter()
                    .any(|hit| hit.result.doc_id == "doc-fast"),
                "fast shard result should be returned"
            );
            assert!(
                elapsed < Duration::from_millis(120),
                "search should return once wait_for_indices is satisfied without waiting for timeout; elapsed={elapsed:?}"
            );
            assert_eq!(response.coverage.answered, vec!["fast".to_owned()]);
            assert_eq!(
                response.coverage.cancelled_in_flight,
                vec!["pending".to_owned()],
                "the shard cancelled by the early stop must be visible in coverage"
            );
            assert!(!response.coverage.is_complete());
        });
    }

    // ─── bd-fj0q tests begin ───

    #[test]
    fn federated_fusion_debug() {
        let fusion = FederatedFusion::Rrf { k: 60.0 };
        let debug_str = format!("{fusion:?}");
        assert!(debug_str.contains("Rrf"));
    }

    #[test]
    fn federated_fusion_clone_and_eq() {
        let a = FederatedFusion::Rrf { k: 42.0 };
        let b = a;
        assert_eq!(a, b);

        let ws = FederatedFusion::WeightedScore {
            normalization: NormalizationMethod::MinMax,
        };
        let ws2 = ws;
        assert_eq!(ws, ws2);

        let mnz = FederatedFusion::CombMnz {
            normalization: NormalizationMethod::MinMax,
        };
        assert_ne!(a, mnz);
    }

    #[test]
    fn federated_fusion_variants_are_distinct() {
        let rrf = FederatedFusion::Rrf { k: 60.0 };
        let ws = FederatedFusion::WeightedScore {
            normalization: NormalizationMethod::MinMax,
        };
        let mnz = FederatedFusion::CombMnz {
            normalization: NormalizationMethod::MinMax,
        };
        assert_ne!(rrf, ws);
        assert_ne!(ws, mnz);
        assert_ne!(rrf, mnz);
    }

    #[test]
    fn federated_config_debug_and_clone() {
        let config = FederatedConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("FederatedConfig"));
        let cloned = config.clone();
        assert_eq!(cloned.per_index_timeout_ms, config.per_index_timeout_ms);
        assert_eq!(cloned.min_indices, config.min_indices);
    }

    #[test]
    fn federated_searcher_debug_and_default() {
        let searcher = FederatedSearcher::default();
        let debug_str = format!("{searcher:?}");
        assert!(debug_str.contains("FederatedSearcher"));
    }

    #[test]
    fn federated_searcher_new_is_empty() {
        let searcher = FederatedSearcher::new();
        assert!(searcher.is_empty());
        assert_eq!(searcher.len(), 0);
    }

    #[test]
    fn federated_searcher_len_after_add() {
        let (_index_dir, index) = build_searcher(&[("doc", &[1.0, 0.0])]);
        let searcher = FederatedSearcher::new()
            .add_index("a", Arc::clone(&index), 1.0)
            .add_index("b", Arc::clone(&index), 0.5)
            .add_index("c", index, 2.0);
        assert_eq!(searcher.len(), 3);
        assert!(!searcher.is_empty());
    }

    #[test]
    fn with_config_overrides_defaults() {
        let config = FederatedConfig {
            per_index_timeout_ms: 1000,
            min_indices: 5,
            candidate_pool_factor: 10,
            max_indices: 42,
            fusion_method: FederatedFusion::Rrf { k: 30.0 },
            wait_for_indices: 7,
        };
        let searcher = FederatedSearcher::new().with_config(config);
        // Verify it compiles and constructs (config is private, so we test behavior)
        assert!(searcher.is_empty());
    }

    #[test]
    fn sanitize_rrf_k_zero_is_valid() {
        let k = super::sanitize_rrf_k(0.0);
        assert!((k - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_rrf_k_positive_is_valid() {
        let k = super::sanitize_rrf_k(30.0);
        assert!((k - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_rrf_k_negative_falls_back() {
        let k = super::sanitize_rrf_k(-1.0);
        assert!((k - super::DEFAULT_FEDERATED_RRF_K).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_rrf_k_nan_falls_back() {
        let k = super::sanitize_rrf_k(f64::NAN);
        assert!((k - super::DEFAULT_FEDERATED_RRF_K).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_rrf_k_infinity_falls_back() {
        let k = super::sanitize_rrf_k(f64::INFINITY);
        assert!((k - super::DEFAULT_FEDERATED_RRF_K).abs() < f64::EPSILON);
    }

    #[test]
    fn rank_contribution_k_zero_no_damping() {
        // k=0: contribution = 1/(0 + rank + 1) = 1/(rank+1)
        let c0 = super::rank_contribution(0.0, 0);
        assert!((c0 - 1.0).abs() < f32::EPSILON); // 1/(0+0+1) = 1.0
        let c1 = super::rank_contribution(0.0, 1);
        assert!((c1 - 0.5).abs() < f32::EPSILON); // 1/(0+1+1) = 0.5
    }

    #[test]
    fn federated_hit_debug_and_clone() {
        let hit = super::FederatedHit {
            result: frankensearch_core::ScoredResult {
                doc_id: "doc-test".into(),
                score: 0.9,
                source: frankensearch_core::ScoreSource::SemanticFast,
                index: Some(0),
                fast_score: Some(0.9),
                quality_score: None,
                lexical_score: None,
                rerank_score: None,
                explanation: None,
                metadata: None,
            },
            source_index: "primary".into(),
            source_rank: 0,
            appeared_in: vec!["primary".into(), "secondary".into()],
        };
        let debug_str = format!("{hit:?}");
        assert!(debug_str.contains("FederatedHit"));
        assert_eq!(hit.source_index, "primary");
        assert_eq!(hit.source_rank, 0);
        assert_eq!(hit.appeared_in.len(), 2);
    }

    #[test]
    fn rrf_with_k_zero_produces_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) =
                build_searcher(&[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.5, 0.0])]);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    fusion_method: FederatedFusion::Rrf { k: 0.0 },
                    ..FederatedConfig::default()
                })
                .add_index("primary", index, 1.0);

            let results = federated
                .search(&cx, "query", 2, |_| None)
                .await
                .unwrap()
                .hits;
            assert_eq!(results.len(), 2);
            // With k=0, rank 0 gets score 1.0, rank 1 gets 0.5
            assert!(results[0].result.score > results[1].result.score);
        });
    }

    #[test]
    fn default_rrf_k_constant_value() {
        assert!((super::DEFAULT_FEDERATED_RRF_K - 60.0).abs() < f64::EPSILON);
    }

    // ─── bd-fj0q tests end ───

    fn rounding_sensitive_shards(order: [usize; 3]) -> Vec<super::ShardResult> {
        let names = ["a-heavy", "b-light", "c-light"];
        let weights = [1.0, f32::EPSILON / 2.0, f32::EPSILON / 2.0];
        let hit = |doc_id: &str, name: &str| frankensearch_core::ScoredResult {
            doc_id: doc_id.into(),
            score: 1.0,
            source: frankensearch_core::ScoreSource::SemanticFast,
            index: Some(0),
            fast_score: Some(1.0),
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            explanation: None,
            metadata: Some(Arc::new(serde_json::json!({"shard": name}))),
        };
        let mut shards: Vec<_> = order
            .into_iter()
            .map(|index| super::ShardResult {
                name: names[index].to_owned(),
                weight: weights[index],
                hits: vec![hit("shared", names[index])],
            })
            .collect();
        shards.push(super::ShardResult {
            name: "z-competitor".to_owned(),
            weight: 1.0 + f32::EPSILON,
            hits: vec![hit("competitor", "z-competitor")],
        });
        shards
    }

    fn assert_same_federated_hits(
        actual: &[super::FederatedHit],
        expected: &[super::FederatedHit],
    ) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(
                actual.result.score.to_bits(),
                expected.result.score.to_bits()
            );
            assert_eq!(
                serde_json::to_value(&actual.result).unwrap(),
                serde_json::to_value(&expected.result).unwrap(),
            );
            assert_eq!(actual.source_index, expected.source_index);
            assert_eq!(actual.source_rank, expected.source_rank);
            assert_eq!(actual.appeared_in, expected.appeared_in);
        }
    }

    #[test]
    fn every_fusion_method_is_bit_identical_across_completion_orders() {
        let canonical = rounding_sensitive_shards([0, 1, 2]);
        let expected_rrf = super::fuse_rrf(&canonical, 0.0);
        assert_eq!(expected_rrf[0].result.doc_id, "competitor");
        let expected_weighted = super::fuse_weighted(&canonical, NormalizationMethod::None, false);
        let expected_mnz = super::fuse_weighted(&canonical, NormalizationMethod::None, true);
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let shards = rounding_sensitive_shards(order);
            assert_same_federated_hits(&super::fuse_rrf(&shards, 0.0), &expected_rrf);
            assert_same_federated_hits(
                &super::fuse_weighted(&shards, NormalizationMethod::None, false),
                &expected_weighted,
            );
            assert_same_federated_hits(
                &super::fuse_weighted(&shards, NormalizationMethod::None, true),
                &expected_mnz,
            );
        }
    }

    #[test]
    fn delayed_shard_cannot_change_the_top_one_cutoff() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for delay_heavy in [false, true] {
                let (_heavy_dir, heavy) = if delay_heavy {
                    build_yielding_searcher(&[("shared", &[1.0, 0.0])])
                } else {
                    build_searcher(&[("shared", &[1.0, 0.0])])
                };
                let (_light_dir, light) = build_searcher(&[("shared", &[1.0, 0.0])]);
                let (_competitor_dir, competitor) = build_searcher(&[("competitor", &[1.0, 0.0])]);
                let federated = FederatedSearcher::new()
                    .with_config(FederatedConfig {
                        fusion_method: FederatedFusion::Rrf { k: 0.0 },
                        min_indices: 4,
                        ..FederatedConfig::default()
                    })
                    .add_index("a-heavy", heavy, 1.0)
                    .add_index("b-light", Arc::clone(&light), f32::EPSILON / 2.0)
                    .add_index("c-light", light, f32::EPSILON / 2.0)
                    .add_index("z-competitor", competitor, 1.0 + f32::EPSILON);
                let response = federated.search(&cx, "query", 1, |_| None).await.unwrap();
                assert!(response.coverage.is_complete());
                assert_eq!(response.coverage.answered.len(), 4);
                assert_eq!(response.hits.len(), 1);
                assert_eq!(response.hits[0].result.doc_id, "competitor");
                assert_eq!(
                    response.hits[0].result.score.to_bits(),
                    (1.0 + f32::EPSILON).to_bits(),
                );
            }
        });
    }

    #[test]
    fn duplicate_dispatched_names_are_rejected_before_gathering() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_pending_dir, pending) = build_pending_searcher(&[("doc", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    per_index_timeout_ms: 0,
                    ..FederatedConfig::default()
                })
                .add_index("duplicate", Arc::clone(&pending), 1.0)
                .add_index("duplicate", pending, 1.0);
            let error = federated
                .search(&cx, "query", 1, |_| None)
                .await
                .expect_err("duplicate names must fail before shard timeouts");
            assert!(matches!(
                error,
                SearchError::InvalidConfig { field, value, .. }
                    if field == "federated.indices" && value == "duplicate_name"
            ));
        });
    }

    #[test]
    fn undispatched_duplicate_does_not_change_the_selected_fanout() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) = build_searcher(&[("doc", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    max_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("selected", Arc::clone(&index), 1.0)
                .add_index("selected", index, 1.0);
            let response = federated.search(&cx, "query", 1, |_| None).await.unwrap();
            assert_eq!(response.hits.len(), 1);
            assert_eq!(response.coverage.queried, vec!["selected".to_owned()]);
            assert!(response.coverage.is_complete());
        });
    }

    #[test]
    fn indexed_text_filters_equal_ids_using_the_originating_store() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_keep_dir, keep) = build_searcher(&[("shared", &[1.0, 0.0])]);
            let (_drop_dir, drop) = build_yielding_searcher(&[("shared", &[1.0, 0.0])]);
            let stores = std::collections::BTreeMap::from([
                (
                    "keep".to_owned(),
                    std::collections::BTreeMap::from([("shared".to_owned(), "keep".to_owned())]),
                ),
                (
                    "drop".to_owned(),
                    std::collections::BTreeMap::from([("shared".to_owned(), "dropme".to_owned())]),
                ),
            ]);
            let federated = FederatedSearcher::new()
                .add_index("keep", keep, 1.0)
                .add_index("drop", drop, 1.0);
            let response = federated
                .search_with_index_text(&cx, "query -dropme", 10, |index, doc_id| {
                    stores
                        .get(index)
                        .and_then(|store| store.get(doc_id))
                        .cloned()
                })
                .await
                .unwrap();
            assert!(response.coverage.is_complete());
            assert_eq!(response.coverage.answered.len(), 2);
            assert_eq!(response.hits.len(), 1);
            assert_eq!(response.hits[0].result.doc_id, "shared");
            assert_eq!(response.hits[0].source_index, "keep");
            assert_eq!(response.hits[0].appeared_in, vec!["keep".to_owned()]);
        });
    }

    #[test]
    fn indexed_text_preserves_shared_provider_results_and_global_fusion() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) =
                build_searcher(&[("shared", &[1.0, 0.0]), ("excluded", &[0.5, 0.5])]);
            let texts = std::collections::BTreeMap::from([
                ("shared".to_owned(), "keep".to_owned()),
                ("excluded".to_owned(), "dropme".to_owned()),
            ]);
            let federated = FederatedSearcher::new()
                .add_index("a", Arc::clone(&index), 1.0)
                .add_index("b", index, 2.0);
            let shared = federated
                .search(&cx, "query -dropme", 10, |doc_id| {
                    texts.get(doc_id).cloned()
                })
                .await
                .unwrap();
            let indexed = federated
                .search_with_index_text(&cx, "query -dropme", 10, |_index, doc_id| {
                    texts.get(doc_id).cloned()
                })
                .await
                .unwrap();
            assert_same_federated_hits(&indexed.hits, &shared.hits);
            assert_eq!(indexed.hits.len(), 1);
            assert_eq!(indexed.hits[0].result.doc_id, "shared");
            assert_eq!(indexed.hits[0].appeared_in, vec!["a", "b"]);
            assert_eq!(indexed.hits[0].source_index, "b");
            assert!(indexed.coverage.is_complete());
        });
    }

    #[test]
    fn indexed_text_never_reads_an_undispatched_store() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) = build_searcher(&[("doc", &[1.0, 0.0])]);
            let calls = AtomicU64::new(0);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    max_indices: 1,
                    ..FederatedConfig::default()
                })
                .add_index("selected", Arc::clone(&index), 1.0)
                .add_index("undispatched", index, 1.0);
            let response = federated
                .search_with_index_text(&cx, "query -dropme", 10, |name, doc_id| {
                    assert_eq!(name, "selected");
                    assert_eq!(doc_id, "doc");
                    calls.fetch_add(1, Ordering::Relaxed);
                    Some("keep".to_owned())
                })
                .await
                .unwrap();
            assert!(calls.load(Ordering::Relaxed) > 0);
            assert_eq!(response.hits.len(), 1);
            assert_eq!(response.coverage.queried, vec!["selected".to_owned()]);
            assert!(response.coverage.is_complete());
        });
    }

    #[test]
    fn filters_select_local_tail_before_the_federated_cutoff() {
        use frankensearch_core::filter::IncludeDocIdsFilter;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut documents = (0..96)
                .map(|number| (format!("denied-{number}"), vec![1.0, 0.0]))
                .collect::<Vec<_>>();
            documents.push(("allowed-tail".to_owned(), vec![0.1, 0.9]));
            let records = documents
                .iter()
                .map(|(id, vector)| (id.as_str(), vector.as_slice()))
                .collect::<Vec<_>>();
            let (_directory, searcher) = build_searcher(&records);
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    candidate_pool_factor: 1,
                    ..FederatedConfig::default()
                })
                .add_index("primary", searcher, 1.0);
            let filter = IncludeDocIdsFilter::new(["allowed-tail"]);
            let hydrated = AtomicU64::new(0);
            let response = federated
                .search_with_filter(&cx, "query -omit", 1, Some(&filter), |doc_id| {
                    assert_eq!(doc_id, "allowed-tail", "denied text must not be read");
                    hydrated.fetch_add(1, Ordering::Relaxed);
                    Some("permitted source".to_owned())
                })
                .await
                .unwrap();
            assert!(hydrated.load(Ordering::Relaxed) > 0);
            assert!(response.coverage.is_complete());
            assert_eq!(response.hits.len(), 1);
            assert_eq!(response.hits[0].result.doc_id, "allowed-tail");
            assert_eq!(response.hits[0].source_rank, 0);
        });
    }

    #[test]
    fn indexed_filters_borrow_request_state_and_keep_equal_ids_source_local() {
        use frankensearch_core::filter::{IncludeDocIdsFilter, SearchFilter};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_first_directory, first) = build_searcher(&[("shared", &[1.0, 0.0])]);
            let (_second_directory, second) = build_yielding_searcher(&[("shared", &[1.0, 0.0])]);
            let deny = IncludeDocIdsFilter::new(std::iter::empty::<&str>());
            let allow = IncludeDocIdsFilter::new(["shared"]);
            let filters = std::collections::BTreeMap::from([
                ("first", &deny as &dyn SearchFilter),
                ("second", &allow as &dyn SearchFilter),
            ]);
            let first_calls = AtomicU64::new(0);
            let second_calls = AtomicU64::new(0);
            let text_calls = AtomicU64::new(0);
            let request_text = String::from("allowed source from second");
            let federated = FederatedSearcher::new()
                .with_config(FederatedConfig {
                    candidate_pool_factor: 1,
                    min_indices: 2,
                    max_indices: 2,
                    ..FederatedConfig::default()
                })
                .add_index("first", Arc::clone(&first), 10.0)
                .add_index("second", second, 2.0)
                .add_index("undispatched", first, 100.0);
            let response = federated
                .search_with_index_filter(
                    &cx,
                    "query -omit",
                    1,
                    |name| {
                        match name {
                            "first" => first_calls.fetch_add(1, Ordering::Relaxed),
                            "second" => second_calls.fetch_add(1, Ordering::Relaxed),
                            _ => panic!("an undispatched shard cannot resolve a filter"),
                        };
                        filters.get(name).copied()
                    },
                    |name, doc_id| {
                        assert_eq!(name, "second", "denied shards cannot hydrate this ID");
                        assert_eq!(doc_id, "shared");
                        text_calls.fetch_add(1, Ordering::Relaxed);
                        Some(request_text.clone())
                    },
                )
                .await
                .unwrap();
            assert_eq!(first_calls.load(Ordering::Relaxed), 1);
            assert_eq!(second_calls.load(Ordering::Relaxed), 1);
            assert!(text_calls.load(Ordering::Relaxed) > 0);
            assert!(response.coverage.is_complete());
            assert_eq!(response.coverage.answered.len(), 2);
            assert_eq!(response.hits.len(), 1);
            let hit = &response.hits[0];
            assert_eq!(hit.result.doc_id, "shared");
            assert_eq!(hit.source_index, "second");
            assert_eq!(hit.source_rank, 0);
            assert_eq!(hit.appeared_in, vec!["second"]);
            assert_eq!(hit.result.score.to_bits(), (2.0_f32 / 61.0).to_bits());
        });
    }

    #[test]
    fn absent_federated_filters_preserve_existing_fusion_and_text_behavior() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, searcher) =
                build_searcher(&[("shared", &[1.0, 0.0]), ("excluded", &[0.5, 0.5])]);
            let texts =
                std::collections::BTreeMap::from([("shared", "keep"), ("excluded", "omit")]);
            let federated = FederatedSearcher::new()
                .add_index("first", Arc::clone(&searcher), 1.0)
                .add_index("second", searcher, 2.0);
            let expected = federated
                .search_with_index_text(&cx, "query -omit", 10, |_, id| {
                    texts.get(id).map(|text| (*text).to_owned())
                })
                .await
                .unwrap();
            let shared = federated
                .search_with_filter(&cx, "query -omit", 10, None, |id| {
                    texts.get(id).map(|text| (*text).to_owned())
                })
                .await
                .unwrap();
            let indexed = federated
                .search_with_index_filter(
                    &cx,
                    "query -omit",
                    10,
                    |_| None,
                    |_, id| texts.get(id).map(|text| (*text).to_owned()),
                )
                .await
                .unwrap();
            assert_same_federated_hits(&shared.hits, &expected.hits);
            assert_same_federated_hits(&indexed.hits, &expected.hits);
            assert!(shared.coverage.is_complete());
            assert!(indexed.coverage.is_complete());
            assert_eq!(indexed.hits.len(), 1);
            assert_eq!(indexed.hits[0].appeared_in, vec!["first", "second"]);
        });
    }

    #[test]
    fn federated_filters_preserve_shard_failure_and_minimum_coverage() {
        use frankensearch_core::filter::IncludeDocIdsFilter;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_healthy_directory, healthy) =
                build_searcher(&[("denied", &[1.0, 0.0]), ("allowed", &[0.1, 0.9])]);
            let (_failed_directory, failed) = build_failing_searcher(&[("allowed", &[1.0, 0.0])]);
            let filter = IncludeDocIdsFilter::new(["allowed"]);
            for required in [1, 2] {
                let federated = FederatedSearcher::new()
                    .with_config(FederatedConfig {
                        min_indices: required,
                        candidate_pool_factor: 1,
                        ..FederatedConfig::default()
                    })
                    .add_index("healthy", Arc::clone(&healthy), 1.0)
                    .add_index("failed", Arc::clone(&failed), 1.0);
                let result = federated
                    .search_with_filter(&cx, "query", 1, Some(&filter), |_| None)
                    .await;
                if required == 1 {
                    let response = result.unwrap();
                    assert_eq!(response.hits.len(), 1);
                    assert_eq!(response.hits[0].result.doc_id, "allowed");
                    assert_eq!(response.coverage.answered, vec!["healthy"]);
                    assert_eq!(response.coverage.failed.len(), 1);
                    assert_eq!(response.coverage.failed[0].index, "failed");
                    assert!(!response.coverage.is_complete());
                } else {
                    assert!(matches!(
                        result,
                        Err(SearchError::FederatedInsufficientResponses {
                            required: 2,
                            received: 1,
                        })
                    ));
                }
            }
        });
    }

    #[test]
    fn filtered_timeout_retains_only_admitted_hits_and_never_hides_cancellation() {
        use frankensearch_core::filter::IncludeDocIdsFilter;

        for cancel_on_drop in [false, true] {
            asupersync::test_utils::run_test_with_cx(move |cx| async move {
                let quality = Arc::new(PendingEmbedder {
                    cancel_on_drop,
                    ..PendingEmbedder::new("pending-quality", 2)
                });
                let (_directory, searcher) = build_refining_searcher(
                    &[
                        ("denied", &[1.0, 0.0], &[1.0, 0.0]),
                        ("allowed", &[0.1, 0.9], &[0.1, 0.9]),
                    ],
                    quality.clone(),
                );
                let federated = FederatedSearcher::new()
                    .with_config(FederatedConfig {
                        per_index_timeout_ms: 1_000,
                        candidate_pool_factor: 1,
                        ..FederatedConfig::default()
                    })
                    .add_index("pending-quality", searcher, 1.0);
                let filter = IncludeDocIdsFilter::new(["allowed"]);
                let result = federated
                    .search_with_filter(
                        &cx,
                        "find related retry behavior",
                        1,
                        Some(&filter),
                        |doc_id| {
                            assert_eq!(doc_id, "allowed", "denied text must not be read");
                            Some("permitted source".to_owned())
                        },
                    )
                    .await;
                assert_eq!(quality.started.load(Ordering::SeqCst), 1);
                assert_eq!(quality.dropped.load(Ordering::SeqCst), 1);
                if cancel_on_drop {
                    assert!(matches!(result, Err(SearchError::Cancelled { .. })));
                } else {
                    let response = result.unwrap();
                    assert_eq!(response.hits.len(), 1);
                    assert_eq!(response.hits[0].result.doc_id, "allowed");
                    assert_eq!(response.coverage.answered, vec!["pending-quality"]);
                    assert_eq!(
                        response.coverage.refinement_timed_out,
                        vec!["pending-quality"]
                    );
                    assert!(response.coverage.timed_out.is_empty());
                    assert!(!response.coverage.is_complete());
                }
            });
        }
    }

    #[test]
    fn federated_filter_providers_do_no_work_before_noop_or_duplicate_refusal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, searcher) = build_searcher(&[("doc", &[1.0, 0.0])]);
            let duplicate = FederatedSearcher::new()
                .add_index("duplicate", Arc::clone(&searcher), 1.0)
                .add_index("duplicate", searcher, 1.0);
            for (query, limit) in [("", 10), ("query", 0), ("query", 1)] {
                let result = duplicate
                    .search_with_index_filter(
                        &cx,
                        query,
                        limit,
                        |_| panic!("no-op and invalid requests cannot resolve filters"),
                        |_, _| panic!("no-op and invalid requests cannot read text"),
                    )
                    .await;
                if query.is_empty() || limit == 0 {
                    let response = result.unwrap();
                    assert!(response.hits.is_empty());
                    assert!(response.coverage.queried.is_empty());
                } else {
                    assert!(matches!(
                        result,
                        Err(SearchError::InvalidConfig { field, .. })
                            if field == "federated.indices"
                    ));
                }
            }
            let empty = FederatedSearcher::new()
                .search_with_index_filter(
                    &cx,
                    "query",
                    1,
                    |_| panic!("an empty federation cannot resolve filters"),
                    |_, _| panic!("an empty federation cannot read text"),
                )
                .await
                .unwrap();
            assert!(empty.hits.is_empty());
            assert!(empty.coverage.queried.is_empty());
        });
    }

    #[test]
    fn indexed_text_noop_queries_do_not_access_stores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_index_dir, index) = build_searcher(&[("doc", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new().add_index("primary", index, 1.0);
            for (query, limit) in [("", 10), ("query -dropme", 0)] {
                let response = federated
                    .search_with_index_text(&cx, query, limit, |_, _| {
                        panic!("no-op queries must not read document stores")
                    })
                    .await
                    .unwrap();
                assert!(response.hits.is_empty());
                assert!(response.coverage.queried.is_empty());
            }
        });
    }
}
