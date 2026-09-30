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
use frankensearch_core::{ScoreSource, ScoredResult, SearchError, SearchResult};
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
    /// Minimum number of indices that must respond successfully — a FLOOR,
    /// enforced after the gather completes. Fewer successful responses fail the
    /// query with [`SearchError::FederatedInsufficientResponses`]. This knob
    /// never stops the gather early; that is `wait_for_indices`.
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
    Completed(ShardResult),
    Failed { index: String, error: SearchError },
    TimedOut { index: String },
    Cancelled { phase: String, reason: String },
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
    /// Shards that exceeded `per_index_timeout_ms`.
    pub timed_out: Vec<String>,
    /// Shards still in flight when the gather stopped early because
    /// `wait_for_indices` was satisfied.
    pub cancelled_in_flight: Vec<String>,
}

impl FederatedCoverage {
    /// True when every dispatched shard contributed results.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.answered.len() == self.queried.len()
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
                    reason: "dispatched index names must be unique for unambiguous fusion and coverage"
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
    /// stop. `min_indices` is a floor validated after the gather.
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
    /// `min_indices` shards complete successfully.
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
            .collect_shard_results(cx, query, per_index_limit, timeout_budget, &text_fn)
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
    async fn collect_shard_results<F>(
        &self,
        cx: &Cx,
        query: &str,
        per_index_limit: usize,
        timeout_budget: Duration,
        text_fn: &F,
    ) -> SearchResult<(Vec<ShardResult>, FederatedCoverage)>
    where
        F: Fn(&str, &str) -> Option<String> + Send + Sync,
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
                Box::pin(async move {
                    let timeout_start = cx.now();
                    let response = {
                        // Bind before the shard starts: exclusion and rerank
                        // lookups must never hydrate from another index's store.
                        let shard_text = |doc_id: &str| text_fn(index_name.as_str(), doc_id);
                        let future = Box::pin(searcher.search_collect_with_text(
                            cx,
                            query,
                            per_index_limit,
                            shard_text,
                        ));
                        timeout(timeout_start, timeout_budget, future).await
                    };
                    match response {
                        Ok(Ok((hits, _metrics))) => ShardCompletion::Completed(ShardResult {
                            name: index_name,
                            weight: index_weight,
                            hits,
                        }),
                        Ok(Err(SearchError::Cancelled { phase, reason })) => {
                            ShardCompletion::Cancelled { phase, reason }
                        }
                        Ok(Err(error)) => ShardCompletion::Failed {
                            index: index_name,
                            error,
                        },
                        Err(_elapsed) => ShardCompletion::TimedOut { index: index_name },
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
                    ShardCompletion::Completed(shard) => {
                        debug!(
                            index = %shard.name,
                            hit_count = shard.hits.len(),
                            "federated shard search completed"
                        );
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
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};
    use frankensearch_core::{SearchError, TwoTierConfig};
    use frankensearch_index::TwoTierIndex;

    use super::{FederatedConfig, FederatedFusion, FederatedSearcher};
    use crate::normalize::NormalizationMethod;
    use crate::searcher::TwoTierSearcher;

    static NEXT_INDEX_FIXTURE: AtomicU64 = AtomicU64::new(0);

    // Reserve the directory atomically: parallel tests share both PID and
    // TMPDIR, and a wall-clock timestamp is not a uniqueness primitive.
    // Retained directories from an earlier process incarnation are skipped,
    // never reopened as a writable fixture for the current test.
    fn reserve_index_fixture_dir() -> io::Result<PathBuf> {
        loop {
            let sequence = NEXT_INDEX_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "frankensearch-federated-test-{}-{sequence}",
                std::process::id()
            ));
            match std::fs::create_dir(&candidate) {
                Ok(()) => return Ok(candidate),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
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
    }

    impl PendingEmbedder {
        const fn new(id: &'static str, dimension: usize) -> Self {
            Self { id, dimension }
        }
    }

    impl Embedder for PendingEmbedder {
        fn embed<'a>(
            &'a self,
            _cx: &'a asupersync::Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, Vec<f32>> {
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

    fn build_index(records: &[(&str, &[f32])]) -> Arc<TwoTierIndex> {
        let dir = reserve_index_fixture_dir().expect("reserve federated test fixture");
        let mut builder =
            TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
        builder.set_fast_embedder_id("stub-fast");
        for (doc_id, vector) in records {
            builder
                .add_fast_record((*doc_id).to_owned(), vector)
                .expect("add record");
        }
        Arc::new(builder.finish().expect("finish index"))
    }

    fn build_searcher(records: &[(&str, &[f32])]) -> Arc<TwoTierSearcher> {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let index = build_index(records);
        let embedder: Arc<dyn Embedder> = Arc::new(StubEmbedder::new("stub-fast", dimension));
        Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ))
    }

    fn build_pending_searcher(records: &[(&str, &[f32])]) -> Arc<TwoTierSearcher> {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let index = build_index(records);
        let embedder: Arc<dyn Embedder> = Arc::new(PendingEmbedder::new("stub-pending", dimension));
        Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ))
    }

    fn build_failing_searcher(records: &[(&str, &[f32])]) -> Arc<TwoTierSearcher> {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let index = build_index(records);
        let embedder: Arc<dyn Embedder> = Arc::new(FailingEmbedder::new("stub-failing", dimension));
        Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ))
    }

    fn build_yielding_searcher(records: &[(&str, &[f32])]) -> Arc<TwoTierSearcher> {
        let dimension = records.first().map_or(1, |(_, vector)| vector.len());
        let index = build_index(records);
        let embedder: Arc<dyn Embedder> =
            Arc::new(YieldingEmbedder::new("stub-yielding", dimension, 32));
        Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ))
    }

    /// bd-3zh67 red proof: an out-of-the-box `FederatedSearcher` must return
    /// results from EVERY healthy shard, not just whichever answered first.
    #[test]
    fn default_config_returns_results_from_all_healthy_shards() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let slow = build_yielding_searcher(&[("doc-slow", &[1.0, 0.0])]);

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
            let healthy = build_searcher(&[("doc-healthy", &[1.0, 0.0])]);
            let failing = build_failing_searcher(&[("doc-failing", &[1.0, 0.0])]);
            let pending = build_pending_searcher(&[("doc-pending", &[1.0, 0.0])]);

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
    fn single_index_returns_ranked_hits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_searcher(&[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.2, 0.0])]);
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
            let index_a = build_searcher(&[("shared", &[1.0, 0.0]), ("a-only", &[0.8, 0.0])]);
            let index_b = build_searcher(&[("b-only", &[1.0, 0.0]), ("shared", &[0.2, 0.0])]);

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
            let large_scale =
                build_searcher(&[("large-top", &[100.0, 0.0]), ("large-low", &[50.0, 0.0])]);
            let small_scale =
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
            let index_a = build_searcher(&[
                ("a-only", &[1.0, 0.0]),
                ("shared", &[0.9, 0.0]),
                ("a-low", &[0.05, 1.0]),
            ]);
            let index_b = build_searcher(&[
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
            let index_a = build_searcher(&[("a-only", &[1.0, 0.0])]);
            let index_b = build_searcher(&[("b-only", &[1.0, 0.0])]);
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
            let fast = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let pending = build_pending_searcher(&[("doc-pending", &[1.0, 0.0])]);

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
            let healthy = build_searcher(&[("doc-healthy", &[1.0, 0.0])]);
            let failing = build_failing_searcher(&[("doc-failing", &[1.0, 0.0])]);

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
            let first = build_failing_searcher(&[("doc-first", &[1.0, 0.0])]);
            let second = build_failing_searcher(&[("doc-second", &[1.0, 0.0])]);

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
            let full = build_searcher(&[("doc-full", &[1.0, 0.0])]);
            let filtered = build_searcher(&[("doc-filtered", &[1.0, 0.0])]);

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
            let index_a = build_searcher(&[("shared", &[1.0, 0.0]), ("a-only", &[1.0, 0.0])]);
            let index_b = build_searcher(&[("shared", &[1.0, 0.0]), ("b-only", &[1.0, 0.0])]);
            let index_c = build_searcher(&[("shared", &[1.0, 0.0]), ("c-only", &[1.0, 0.0])]);

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
            let first = build_searcher(&[("doc-first", &[1.0, 0.0])]);
            let second = build_searcher(&[("doc-second", &[1.0, 0.0])]);
            let third = build_searcher(&[("doc-third", &[1.0, 0.0])]);

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
            let index = build_searcher(&[("doc-a", &[1.0, 0.0])]);
            let federated = FederatedSearcher::new().add_index("primary", index, 1.0);
            let results = federated.search(&cx, "", 10, |_| None).await.unwrap().hits;
            assert!(results.is_empty());
        });
    }

    #[test]
    fn zero_limit_returns_empty_results() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_searcher(&[("doc-a", &[1.0, 0.0])]);
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
            let fast = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let pending_a = build_pending_searcher(&[("doc-pending-a", &[1.0, 0.0])]);
            let pending_b = build_pending_searcher(&[("doc-pending-b", &[1.0, 0.0])]);

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
            let fast = build_searcher(&[("doc-fast", &[1.0, 0.0])]);
            let pending = build_pending_searcher(&[("doc-pending", &[1.0, 0.0])]);

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
        let index = build_searcher(&[("doc", &[1.0, 0.0])]);
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
            let index = build_searcher(&[("doc-a", &[1.0, 0.0]), ("doc-b", &[0.5, 0.0])]);
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

    fn assert_same_federated_hits(actual: &[super::FederatedHit], expected: &[super::FederatedHit]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.result.score.to_bits(), expected.result.score.to_bits());
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
                let heavy = if delay_heavy {
                    build_yielding_searcher(&[("shared", &[1.0, 0.0])])
                } else {
                    build_searcher(&[("shared", &[1.0, 0.0])])
                };
                let light = build_searcher(&[("shared", &[1.0, 0.0])]);
                let competitor = build_searcher(&[("competitor", &[1.0, 0.0])]);
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
            let pending = build_pending_searcher(&[("doc", &[1.0, 0.0])]);
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
            let index = build_searcher(&[("doc", &[1.0, 0.0])]);
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
            let keep = build_searcher(&[("shared", &[1.0, 0.0])]);
            let drop = build_yielding_searcher(&[("shared", &[1.0, 0.0])]);
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
                    stores.get(index).and_then(|store| store.get(doc_id)).cloned()
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
            let index = build_searcher(&[
                ("shared", &[1.0, 0.0]),
                ("excluded", &[0.5, 0.5]),
            ]);
            let texts = std::collections::BTreeMap::from([
                ("shared".to_owned(), "keep".to_owned()),
                ("excluded".to_owned(), "dropme".to_owned()),
            ]);
            let federated = FederatedSearcher::new()
                .add_index("a", Arc::clone(&index), 1.0)
                .add_index("b", index, 2.0);
            let shared = federated
                .search(&cx, "query -dropme", 10, |doc_id| texts.get(doc_id).cloned())
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
            let index = build_searcher(&[("doc", &[1.0, 0.0])]);
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
    fn indexed_text_noop_queries_do_not_access_stores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let index = build_searcher(&[("doc", &[1.0, 0.0])]);
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
