//! Graph-ranking phase-1 hook (feature-gated).
//!
//! This module implements a lightweight query-biased `PageRank` variant on the
//! optional `DocumentGraph` supplied by the caller.

use std::collections::HashMap;
use std::hash::BuildHasher;

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
                    doc_id: doc_id.into(),
                    score: score_f32,
                    source: ScoreSource::Hybrid,
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
            let entry = seed_weights.entry(hit.doc_id.to_string()).or_insert(0.0);
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

    /// Compute graph-ranked candidates for phase-1 fusion.
    ///
    /// Seeds come from current semantic hits (query-matched docs).
    #[must_use]
    pub fn rank_phase1(
        &self,
        cx: &Cx,
        query: &str,
        graph: &DocumentGraph,
        seed_hits: &[VectorHit],
        limit: usize,
    ) -> Option<Vec<ScoredResult>> {
        self.rank_phase1_with_hasher::<ahash::RandomState>(cx, query, graph, seed_hits, limit)
    }

    /// Test/bench-only legacy arm for the dense doc-id index's standard `SipHash` hasher.
    ///
    /// The production path uses aHash for this lookup-only map. Both arms share this function's
    /// implementation, and neither iterates the map, so changing the hasher cannot affect node
    /// numbering, edge visitation, floating-point accumulation, ties, or final ordering.
    #[cfg(any(test, feature = "bench-internals"))]
    #[must_use]
    pub fn rank_phase1_siphash(
        &self,
        cx: &Cx,
        query: &str,
        graph: &DocumentGraph,
        seed_hits: &[VectorHit],
        limit: usize,
    ) -> Option<Vec<ScoredResult>> {
        self.rank_phase1_with_hasher::<std::collections::hash_map::RandomState>(
            cx, query, graph, seed_hits, limit,
        )
    }

    fn rank_phase1_with_hasher<S: BuildHasher + Default>(
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

        // Dense-index the graph ONCE and run the PageRank power iteration over
        // `Vec<f64>` buffers instead of rebuilding a `HashMap<String, f64>` every
        // iteration. `add_edge` inserts both endpoints, so every node — including
        // dangling sinks reached only as a neighbor — is an adjacency key; the
        // node set is therefore exactly `adjacency().keys()`. This eliminates,
        // per iteration: a fresh HashMap allocation, a `String` clone of every
        // doc_id touched (all `entry(_.clone())` keys were already present, so the
        // clones were dead work), and a hash probe per edge — all replaced by
        // array indexing. The per-edge weight-finiteness filter and `f64::from`
        // widen are also hoisted out of the iteration into the one-time CSR build
        // (the old loop re-checked them every pass). Ranking is equivalent: the
        // power iteration converges to the same fixed point within `tolerance`,
        // and the prior `std::HashMap` accumulation order was already run-to-run
        // non-deterministic.
        let adjacency = graph.adjacency();
        let n = adjacency.len();
        let mut nodes: Vec<&GraphDocId> = Vec::with_capacity(n);
        // This map is lookup-only after construction; node order comes from `adjacency.keys()`.
        // The hasher is therefore a pure cost choice and cannot perturb result order or scores.
        let mut idx: HashMap<&str, usize, S> = HashMap::with_capacity_and_hasher(n, S::default());
        for doc_id in adjacency.keys() {
            idx.insert(doc_id.as_str(), nodes.len());
            nodes.push(doc_id);
        }

        // CSR edges + per-node outgoing weight sums (valid edges only), once.
        let mut out_sum = vec![0.0_f64; n];
        let mut edges_csr: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (doc_id, edges) in adjacency {
            let src = idx[doc_id.as_str()];
            let mut sum = 0.0_f64;
            let mut row = Vec::with_capacity(edges.len());
            for edge in edges {
                let weight = f64::from(edge.weight);
                if !weight.is_finite() || weight <= 0.0 {
                    continue;
                }
                sum += weight;
                if let Some(&dst) = idx.get(edge.neighbor_doc_id.as_str()) {
                    row.push((dst, weight));
                }
            }
            out_sum[src] = sum;
            edges_csr[src] = row;
        }

        // Personalization (teleport) vector in dense form.
        let seeds: Vec<(usize, f64)> = personalization
            .iter()
            .filter_map(|(doc_id, w)| idx.get(doc_id.as_str()).map(|&i| (i, *w)))
            .collect();

        let teleport_scale = self.restart_probability.clamp(0.0, 1.0);
        let walk_scale = 1.0 - teleport_scale;

        let mut ranks = vec![0.0_f64; n];
        for &(i, w) in &seeds {
            ranks[i] = w;
        }
        let mut next = vec![0.0_f64; n];

        for _ in 0..self.max_iterations {
            next.fill(0.0);

            for &(i, w) in &seeds {
                next[i] += teleport_scale * w;
            }

            let dangling_mass: f64 = (0..n)
                .filter(|&i| out_sum[i] <= f64::EPSILON)
                .map(|i| ranks[i])
                .sum();

            if dangling_mass > 0.0 {
                for &(i, w) in &seeds {
                    next[i] += walk_scale * dangling_mass * w;
                }
            }

            for src in 0..n {
                let rank = ranks[src];
                if rank <= 0.0 {
                    continue;
                }
                let out_total = out_sum[src];
                if out_total <= f64::EPSILON {
                    continue;
                }
                let base = walk_scale * rank / out_total;
                for &(dst, weight) in &edges_csr[src] {
                    next[dst] += base * weight;
                }
            }

            let l1_delta: f64 = (0..n).map(|i| (ranks[i] - next[i]).abs()).sum();
            std::mem::swap(&mut ranks, &mut next);
            if l1_delta < self.tolerance {
                break;
            }
        }

        // Map dense ranks back to doc ids and reuse the existing finalize path.
        let ranks_map: HashMap<GraphDocId, f64> = nodes
            .iter()
            .zip(ranks.iter())
            .map(|(&doc_id, &rank)| (doc_id.clone(), rank))
            .collect();
        Self::finalize_scores(ranks_map, limit)
    }

    /// **Bench-only twin of [`Self::rank_phase1`] using a true flat CSR.** Not a shipping path.
    ///
    /// Exists so `benches/graph_rank.rs` can A/B the edge layout as a **single variable** against the
    /// shipped function: same private personalization, same power iteration, same `finalize_scores`,
    /// same edge-visit order — the *only* difference is `Vec<Vec<(usize, f64)>>` (one heap block per
    /// node) versus one contiguous `edges_flat` plus an `offsets` array.
    ///
    /// It lives here rather than in the bench because the helpers it must share are private. The
    /// 2026-07-09 flat-CSR REJECT row was decided on a bench-local *copy* of the whole ranker, whose
    /// per-call cost is 11–13% below production's (`bd-i40y`); a copy cannot settle a layout question
    /// about the shipped path. Keeping both arms in-tree is what makes that row reproducible.
    ///
    /// Because the edge-visit order per `src` is unchanged, `next[dst]` accumulates in the same order,
    /// so this is **bit-identical** to `rank_phase1` — the bench asserts that, and a divergence would
    /// mean the two arms are no longer measuring one variable.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn rank_phase1_flat(
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

        let adjacency = graph.adjacency();
        let n = adjacency.len();
        let mut nodes: Vec<&GraphDocId> = Vec::with_capacity(n);
        // Match the shipped lookup hasher so this twin still differs only in edge layout.
        let mut idx: ahash::AHashMap<&str, usize> = ahash::AHashMap::with_capacity(n);
        for doc_id in adjacency.keys() {
            idx.insert(doc_id.as_str(), nodes.len());
            nodes.push(doc_id);
        }

        // Flat CSR build. This is the hypothesis under test: two heap allocations instead of `n`,
        // and a sequential sweep of `edges_flat` in the inner loop. The cost is a COUNTING pass,
        // which re-probes `idx` for every edge — the per-edge `doc_id`-string hash probe is paid
        // twice, once to size each row and once to fill it.
        let mut out_sum = vec![0.0_f64; n];
        let mut offsets = vec![0_usize; n + 1];
        for (doc_id, edges) in adjacency {
            let src = idx[doc_id.as_str()];
            let mut count = 0_usize;
            for edge in edges {
                let weight = f64::from(edge.weight);
                if !weight.is_finite() || weight <= 0.0 {
                    continue;
                }
                if idx.contains_key(edge.neighbor_doc_id.as_str()) {
                    count += 1;
                }
            }
            offsets[src + 1] = count;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut edges_flat = vec![(0_usize, 0.0_f64); offsets[n]];
        let mut cursor = offsets.clone();
        for (doc_id, edges) in adjacency {
            let src = idx[doc_id.as_str()];
            let mut sum = 0.0_f64;
            for edge in edges {
                let weight = f64::from(edge.weight);
                if !weight.is_finite() || weight <= 0.0 {
                    continue;
                }
                // `sum` counts every finite positive edge, including any whose destination is not
                // an indexed node — matching `rank_phase1` exactly.
                sum += weight;
                if let Some(&dst) = idx.get(edge.neighbor_doc_id.as_str()) {
                    edges_flat[cursor[src]] = (dst, weight);
                    cursor[src] += 1;
                }
            }
            out_sum[src] = sum;
        }

        let seeds: Vec<(usize, f64)> = personalization
            .iter()
            .filter_map(|(doc_id, w)| idx.get(doc_id.as_str()).map(|&i| (i, *w)))
            .collect();

        let teleport_scale = self.restart_probability.clamp(0.0, 1.0);
        let walk_scale = 1.0 - teleport_scale;

        let mut ranks = vec![0.0_f64; n];
        for &(i, w) in &seeds {
            ranks[i] = w;
        }
        let mut next = vec![0.0_f64; n];

        for _ in 0..self.max_iterations {
            next.fill(0.0);

            for &(i, w) in &seeds {
                next[i] += teleport_scale * w;
            }

            let dangling_mass: f64 = (0..n)
                .filter(|&i| out_sum[i] <= f64::EPSILON)
                .map(|i| ranks[i])
                .sum();

            if dangling_mass > 0.0 {
                for &(i, w) in &seeds {
                    next[i] += walk_scale * dangling_mass * w;
                }
            }

            for src in 0..n {
                let rank = ranks[src];
                if rank <= 0.0 {
                    continue;
                }
                let out_total = out_sum[src];
                if out_total <= f64::EPSILON {
                    continue;
                }
                let base = walk_scale * rank / out_total;
                for &(dst, weight) in &edges_flat[offsets[src]..offsets[src + 1]] {
                    next[dst] += base * weight;
                }
            }

            let l1_delta: f64 = (0..n).map(|i| (ranks[i] - next[i]).abs()).sum();
            std::mem::swap(&mut ranks, &mut next);
            if l1_delta < self.tolerance {
                break;
            }
        }

        let ranks_map: HashMap<GraphDocId, f64> = nodes
            .iter()
            .zip(ranks.iter())
            .map(|(&doc_id, &rank)| (doc_id.clone(), rank))
            .collect();
        Self::finalize_scores(ranks_map, limit)
    }

    /// **Bench-only twin (bd-5hlw): single-pass FLAT CSR sized from the `edges.len()` UPPER BOUND.**
    ///
    /// The "strongest form" of the flat-CSR lever the ledger left untested. Unlike
    /// [`Self::rank_phase1_flat`] — which pays a COUNTING pass (a second `idx` probe per edge) to size
    /// each row exactly — this sizes `offsets` from each row's raw `edges.len()` (NO probe), then fills
    /// in ONE pass paying a single `idx.get` probe per edge, exactly matching the shipped build's probe
    /// count. Dropped (non-node) neighbors leave a gap after each row, so `row_end[src]` records where
    /// the live edges end; the power iteration reads only `edges_flat[offsets[src]..row_end[src]]` — no
    /// compaction pass. Trades the shipped build's `n` small row allocs for 3 arena allocs plus an
    /// `edges_flat` over-allocated to the total RAW edge count. Edge-visit order per `src` is unchanged
    /// ⇒ **bit-identical** to [`Self::rank_phase1`] (the bench asserts it), so the arms differ by edge
    /// layout and build strategy alone.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn rank_phase1_flat_upper(
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

        let adjacency = graph.adjacency();
        let n = adjacency.len();
        let mut nodes: Vec<&GraphDocId> = Vec::with_capacity(n);
        // Match the shipped lookup hasher so this twin still differs only in edge layout/build.
        let mut idx: ahash::AHashMap<&str, usize> = ahash::AHashMap::with_capacity(n);
        for doc_id in adjacency.keys() {
            idx.insert(doc_id.as_str(), nodes.len());
            nodes.push(doc_id);
        }

        // UPPER-BOUND flat CSR: size `offsets` from raw edge counts (NO probe), so the single fill
        // pass pays exactly one `idx` probe per edge like the shipped build (the rejected two-pass
        // `rank_phase1_flat` paid two). `row_end[src]` marks the end of the LIVE edges written into
        // row `src`'s upper-bound slot; the gap `[row_end[src], offsets[src + 1])` is never read.
        let mut out_sum = vec![0.0_f64; n];
        let mut offsets = vec![0_usize; n + 1];
        for (doc_id, edges) in adjacency {
            let src = idx[doc_id.as_str()];
            offsets[src + 1] = edges.len();
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut edges_flat = vec![(0_usize, 0.0_f64); offsets[n]];
        let mut row_end = vec![0_usize; n];
        for (doc_id, edges) in adjacency {
            let src = idx[doc_id.as_str()];
            let mut cursor = offsets[src];
            let mut sum = 0.0_f64;
            for edge in edges {
                let weight = f64::from(edge.weight);
                if !weight.is_finite() || weight <= 0.0 {
                    continue;
                }
                // `sum` counts every finite positive edge (incl. non-node destinations) — matches
                // `rank_phase1` / `rank_phase1_flat` exactly.
                sum += weight;
                if let Some(&dst) = idx.get(edge.neighbor_doc_id.as_str()) {
                    edges_flat[cursor] = (dst, weight);
                    cursor += 1;
                }
            }
            out_sum[src] = sum;
            row_end[src] = cursor;
        }

        let seeds: Vec<(usize, f64)> = personalization
            .iter()
            .filter_map(|(doc_id, w)| idx.get(doc_id.as_str()).map(|&i| (i, *w)))
            .collect();

        let teleport_scale = self.restart_probability.clamp(0.0, 1.0);
        let walk_scale = 1.0 - teleport_scale;

        let mut ranks = vec![0.0_f64; n];
        for &(i, w) in &seeds {
            ranks[i] = w;
        }
        let mut next = vec![0.0_f64; n];

        for _ in 0..self.max_iterations {
            next.fill(0.0);

            for &(i, w) in &seeds {
                next[i] += teleport_scale * w;
            }

            let dangling_mass: f64 = (0..n)
                .filter(|&i| out_sum[i] <= f64::EPSILON)
                .map(|i| ranks[i])
                .sum();

            if dangling_mass > 0.0 {
                for &(i, w) in &seeds {
                    next[i] += walk_scale * dangling_mass * w;
                }
            }

            for src in 0..n {
                let rank = ranks[src];
                if rank <= 0.0 {
                    continue;
                }
                let out_total = out_sum[src];
                if out_total <= f64::EPSILON {
                    continue;
                }
                let base = walk_scale * rank / out_total;
                for &(dst, weight) in &edges_flat[offsets[src]..row_end[src]] {
                    next[dst] += base * weight;
                }
            }

            let l1_delta: f64 = (0..n).map(|i| (ranks[i] - next[i]).abs()).sum();
            std::mem::swap(&mut ranks, &mut next);
            if l1_delta < self.tolerance {
                break;
            }
        }

        let ranks_map: HashMap<GraphDocId, f64> = nodes
            .iter()
            .zip(ranks.iter())
            .map(|(&doc_id, &rank)| (doc_id.clone(), rank))
            .collect();
        Self::finalize_scores(ranks_map, limit)
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
                doc_id: "doc-a".into(),
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
                doc_id: "outside-graph".into(),
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
                doc_id: "doc-a".into(),
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
            assert!(
                results
                    .iter()
                    .all(|result| result.source == frankensearch_core::types::ScoreSource::Hybrid),
                "graph PageRank is not a semantic embedding score"
            );
        });
    }

    #[test]
    fn dense_idx_ahash_matches_siphash_bits_and_ordering() {
        run_test_with_cx(|cx| async move {
            let mut graph = DocumentGraph::new();
            graph.add_edge("doc-a", "doc-b", EdgeType::Reference, 1.0);
            graph.add_edge("doc-a", "doc-c", EdgeType::Reference, 1.0);
            graph.add_edge("doc-b", "doc-d", EdgeType::Reference, 1.0);
            graph.add_edge("doc-c", "doc-d", EdgeType::Reference, 1.0);
            let seed_hits = vec![VectorHit {
                index: 0,
                score: 1.0,
                doc_id: "doc-a".into(),
            }];
            let ranker = GraphRanker::new();
            let ahash = ranker
                .rank_phase1(&cx, "query", &graph, &seed_hits, 10)
                .expect("aHash graph rank");
            let siphash = ranker
                .rank_phase1_siphash(&cx, "query", &graph, &seed_hits, 10)
                .expect("SipHash graph rank");

            assert_eq!(ahash.len(), siphash.len());
            for (fast, legacy) in ahash.iter().zip(&siphash) {
                assert_eq!(fast.doc_id, legacy.doc_id, "hasher changed ordering");
                assert_eq!(
                    fast.score.to_bits(),
                    legacy.score.to_bits(),
                    "hasher changed score bits for {}",
                    fast.doc_id
                );
            }
        });
    }

    /// Reference: the pre-dense `PageRank` using an ordered (`BTreeMap`)
    /// accumulator — deterministic, mirroring the original `HashMap` algorithm.
    fn rank_phase1_reference(
        graph: &DocumentGraph,
        seed_hits: &[VectorHit],
        limit: usize,
    ) -> Option<Vec<frankensearch_core::types::ScoredResult>> {
        use std::collections::BTreeMap;
        if graph.is_empty() || limit == 0 {
            return None;
        }
        let personalization = GraphRanker::personalization_from_seed_hits(graph, seed_hits);
        if personalization.is_empty() {
            return None;
        }
        let r = GraphRanker::new();
        let mut ranks: BTreeMap<String, f64> = graph
            .adjacency()
            .keys()
            .cloned()
            .map(|d| (d, 0.0))
            .collect();
        for (d, s) in &personalization {
            ranks.insert(d.clone(), *s);
        }
        let out_sum: BTreeMap<String, f64> = graph
            .adjacency()
            .iter()
            .map(|(d, edges)| {
                let s: f64 = edges
                    .iter()
                    .map(|e| f64::from(e.weight))
                    .filter(|w| w.is_finite() && *w > 0.0)
                    .sum();
                (d.clone(), s)
            })
            .collect();
        let teleport = r.restart_probability.clamp(0.0, 1.0);
        let walk = 1.0 - teleport;
        for _ in 0..r.max_iterations {
            let mut next: BTreeMap<String, f64> = graph
                .adjacency()
                .keys()
                .cloned()
                .map(|d| (d, 0.0))
                .collect();
            for (d, w) in &personalization {
                *next.entry(d.clone()).or_insert(0.0) += teleport * w;
            }
            let dangling: f64 = ranks
                .iter()
                .filter_map(|(d, rk)| {
                    (out_sum.get(d).copied().unwrap_or(0.0) <= f64::EPSILON).then_some(*rk)
                })
                .sum();
            if dangling > 0.0 {
                for (d, w) in &personalization {
                    *next.entry(d.clone()).or_insert(0.0) += walk * dangling * w;
                }
            }
            for (d, edges) in graph.adjacency() {
                let rk = ranks.get(d).copied().unwrap_or(0.0);
                if rk <= 0.0 {
                    continue;
                }
                let ot = out_sum.get(d).copied().unwrap_or(0.0);
                if ot <= f64::EPSILON {
                    continue;
                }
                let base = walk * rk / ot;
                for e in edges {
                    let w = f64::from(e.weight);
                    if !w.is_finite() || w <= 0.0 {
                        continue;
                    }
                    *next.entry(e.neighbor_doc_id.clone()).or_insert(0.0) += base * w;
                }
            }
            let l1: f64 = ranks
                .iter()
                .map(|(d, old)| (old - next.get(d).unwrap_or(&0.0)).abs())
                .sum();
            ranks = next;
            if l1 < r.tolerance {
                break;
            }
        }
        GraphRanker::finalize_scores(ranks.into_iter().collect(), limit)
    }

    #[test]
    fn dense_rank_matches_reference_ranking() {
        run_test_with_cx(|cx| async move {
            // Deterministic pseudo-random graph: 40 nodes, ~4 out-edges each.
            let mut state = 0x1234_5678_9abc_def1_u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let n = 40usize;
            let n_u64 = u64::try_from(n).expect("test graph size fits u64");
            let mut graph = DocumentGraph::new();
            for i in 0..n {
                for _ in 0..4 {
                    let j = usize::try_from(next() % n_u64).expect("reduced node index fits usize");
                    if j != i {
                        let w = 0.25 + (next() % 1000) as f32 / 1000.0;
                        graph.add_edge(
                            format!("d{i:03}"),
                            format!("d{j:03}"),
                            EdgeType::Reference,
                            w,
                        );
                    }
                }
            }
            let seed_hits: Vec<VectorHit> = (0..5usize)
                .map(|s| VectorHit {
                    index: u32::try_from(s).expect("test seed index fits u32"),
                    score: 0.5 + (s as f32) * 0.1,
                    doc_id: format!("d{:03}", (s * 7) % n).into(),
                })
                .collect();

            let got = GraphRanker::new()
                .rank_phase1(&cx, "q", &graph, &seed_hits, 25)
                .expect("dense graph rank");
            let want = rank_phase1_reference(&graph, &seed_hits, 25).expect("reference rank");

            let got_ids: Vec<&str> = got.iter().map(|r| r.doc_id.as_str()).collect();
            let want_ids: Vec<&str> = want.iter().map(|r| r.doc_id.as_str()).collect();
            assert_eq!(
                got_ids, want_ids,
                "dense ranking order must match reference"
            );
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (f64::from(g.score) - f64::from(w.score)).abs() < 1e-4,
                    "score mismatch for {}: {} vs {}",
                    g.doc_id,
                    g.score,
                    w.score
                );
            }
        });
    }
}
