//! Graph-ranking phase-1 hook (feature-gated).
//!
//! This module implements a lightweight query-biased `PageRank` variant on the
//! optional `DocumentGraph` supplied by the caller.

use std::collections::HashMap;

use asupersync::Cx;
use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_core::{DocumentGraph, GraphDocId};

const DEFAULT_RESTART_PROBABILITY: f64 = 0.15;
const DEFAULT_MAX_ITERATIONS: usize = 20;
const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Query-biased `PageRank` engine for optional graph signal generation.
#[derive(Debug, Clone, Copy)]
pub struct GraphRanker {
    restart_probability: f64,
    max_iterations: usize,
    tolerance: f64,
}

impl Default for GraphRanker {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphRanker {
    /// Construct a ranker with conservative defaults for phase-1 latency.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            restart_probability: DEFAULT_RESTART_PROBABILITY,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn finalize_scores(
        mut ranks: HashMap<GraphDocId, f64>,
        limit: usize,
    ) -> Option<Vec<ScoredResult>> {
        if limit == 0 {
            return None;
        }
        let total_rank = ranks.values().sum::<f64>();
        if total_rank > 0.0 {
            for value in ranks.values_mut() {
                *value /= total_rank;
            }
        }
        let mut results: Vec<ScoredResult> = ranks
            .into_iter()
            .filter(|(_, score)| score.is_finite() && *score > 0.0)
            .map(|(doc_id, score)| {
                let score_f32 = score as f32;
                ScoredResult {
                    doc_id,
                    score: score_f32,
                    source: ScoreSource::SemanticFast,
                    index: None,
                    fast_score: Some(score_f32),
                    quality_score: None,
                    lexical_score: None,
                    rerank_score: None,
                    explanation: None,
                    metadata: None,
                }
            })
            .collect();
        results.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        results.truncate(limit);
        (!results.is_empty()).then_some(results)
    }

    fn personalization_from_seed_hits(
        graph: &DocumentGraph,
        seed_hits: &[VectorHit],
    ) -> HashMap<GraphDocId, f64> {
        let mut seed_weights: HashMap<GraphDocId, f64> = HashMap::new();
        for hit in seed_hits {
            if !graph.contains_node(&hit.doc_id) {
                continue;
            }
            let score = f64::from(hit.score);
            if !score.is_finite() || score <= 0.0 {
                continue;
            }
            let entry = seed_weights.entry(hit.doc_id.clone()).or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }
        let total = seed_weights.values().sum::<f64>();
        if total <= 0.0 {
            return HashMap::new();
        }
        for weight in seed_weights.values_mut() {
            *weight /= total;
        }
        seed_weights
    }

    fn outgoing_weight_sums(graph: &DocumentGraph) -> HashMap<GraphDocId, f64> {
        graph
            .adjacency()
            .iter()
            .map(|(doc_id, edges)| {
                let sum = edges
                    .iter()
                    .map(|edge| f64::from(edge.weight))
                    .filter(|weight| weight.is_finite() && *weight > 0.0)
                    .sum::<f64>();
                (doc_id.clone(), sum)
            })
            .collect()
    }

    /// Compute graph-ranked candidates for phase-1 fusion.
    ///
    /// Seeds come from current semantic hits (query-matched docs).
    #[must_use]
    pub fn rank_phase1(
        &self,
        _cx: &Cx,
        _query: &str,
        graph: &DocumentGraph,
        seed_hits: &[VectorHit],
        limit: usize,
    ) -> Option<Vec<ScoredResult>> {
        if graph.is_empty() || limit == 0 {
            return None;
        }

        let personalization = Self::personalization_from_seed_hits(graph, seed_hits);
        if personalization.is_empty() {
            return None;
        }

        let mut ranks: HashMap<GraphDocId, f64> = graph
            .adjacency()
            .keys()
            .cloned()
            .map(|doc_id| (doc_id, 0.0))
            .collect();
        for (doc_id, score) in &personalization {
            ranks.insert(doc_id.clone(), *score);
        }

        let outgoing_sum = Self::outgoing_weight_sums(graph);
        let teleport_scale = self.restart_probability.clamp(0.0, 1.0);
        let walk_scale = 1.0 - teleport_scale;

        for _ in 0..self.max_iterations {
            let mut next: HashMap<GraphDocId, f64> = graph
                .adjacency()
                .keys()
                .cloned()
                .map(|doc_id| (doc_id, 0.0))
                .collect();

            for (doc_id, seed_weight) in &personalization {
                *next.entry(doc_id.clone()).or_insert(0.0) += teleport_scale * seed_weight;
            }

            let dangling_mass = ranks
                .iter()
                .filter_map(|(doc_id, rank)| {
                    (outgoing_sum.get(doc_id).copied().unwrap_or(0.0) <= f64::EPSILON)
                        .then_some(*rank)
                })
                .sum::<f64>();

            if dangling_mass > 0.0 {
                for (doc_id, seed_weight) in &personalization {
                    *next.entry(doc_id.clone()).or_insert(0.0) +=
                        walk_scale * dangling_mass * seed_weight;
                }
            }

            for (doc_id, edges) in graph.adjacency() {
                let rank = ranks.get(doc_id).copied().unwrap_or(0.0);
                if rank <= 0.0 {
                    continue;
                }
                let out_total = outgoing_sum.get(doc_id).copied().unwrap_or(0.0);
                if out_total <= f64::EPSILON {
                    continue;
                }
                let base = walk_scale * rank / out_total;
                for edge in edges {
                    let weight = f64::from(edge.weight);
                    if !weight.is_finite() || weight <= 0.0 {
                        continue;
                    }
                    *next.entry(edge.neighbor_doc_id.clone()).or_insert(0.0) += base * weight;
                }
            }

            let l1_delta = ranks
                .iter()
                .map(|(doc_id, old_rank)| (old_rank - next.get(doc_id).unwrap_or(&0.0)).abs())
                .sum::<f64>();
            ranks = next;
            if l1_delta < self.tolerance {
                break;
            }
        }

        Self::finalize_scores(ranks, limit)
    }
}

#[cfg(test)]
mod tests {
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::types::VectorHit;
    use frankensearch_core::{DocumentGraph, EdgeType};

    use super::GraphRanker;

    #[test]
    fn returns_none_for_empty_graph() {
        run_test_with_cx(|cx| async move {
            let graph = DocumentGraph::new();
            let seed_hits = vec![VectorHit {
                index: 0,
                score: 1.0,
                doc_id: "doc-a".to_owned(),
            }];
            let results = GraphRanker::new().rank_phase1(&cx, "query", &graph, &seed_hits, 5);
            assert!(results.is_none());
        });
    }

    #[test]
    fn returns_none_when_no_seed_doc_is_in_graph() {
        run_test_with_cx(|cx| async move {
            let mut graph = DocumentGraph::new();
            graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 1.0);
            let seed_hits = vec![VectorHit {
                index: 0,
                score: 1.0,
                doc_id: "outside-graph".to_owned(),
            }];
            let results = GraphRanker::new().rank_phase1(&cx, "query", &graph, &seed_hits, 5);
            assert!(results.is_none());
        });
    }

    #[test]
    fn rank_phase1_propagates_signal_to_neighbors() {
        run_test_with_cx(|cx| async move {
            let mut graph = DocumentGraph::new();
            graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 1.0);
            graph.add_edge("doc-b", "doc-c", EdgeType::Reference, 1.0);
            let seed_hits = vec![VectorHit {
                index: 0,
                score: 1.0,
                doc_id: "doc-a".to_owned(),
            }];

            let results = GraphRanker::new()
                .rank_phase1(&cx, "query", &graph, &seed_hits, 10)
                .expect("graph rank should yield scores");

            assert!(
                results.iter().any(|result| result.doc_id == "doc-b"),
                "neighbor of seed doc should get non-zero graph score"
            );
            assert!(
                results.iter().any(|result| result.doc_id == "doc-c"),
                "second hop should get propagated graph signal"
            );
        });
    }
}
