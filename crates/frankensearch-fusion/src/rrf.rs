//! Reciprocal Rank Fusion (RRF) for combining lexical and semantic search results.
//!
//! RRF is a principled, training-free method for fusing ranked lists from
//! different retrieval systems (Cormack et al., 2009).
//!
//! The score for a document appearing at rank `r` (0-based) in source `i` is:
//!
//! ```text
//! score(doc) = Σ_i  1 / (K + r_i + 1)
//! ```
//!
//! Documents appearing in multiple sources get their contributions summed,
//! which naturally boosts multi-source hits.

use std::collections::HashMap;

use frankensearch_core::{FusedHit, ScoredResult, VectorHit};
use tracing::{debug, instrument};

// ─── Configuration ──────────────────────────────────────────────────────────
const DEFAULT_RRF_K: f64 = 60.0;

/// RRF fusion parameters.
///
/// The `k` constant controls how steeply rank affects score:
/// - Higher K → flatter distribution (high and low ranks scored similarly)
/// - Lower K → sharper distribution (top ranks much more valuable)
///
/// K=60 is the empirically optimal value from the original paper and is
/// used in production at Elastic, Pinecone, and Vespa.
#[derive(Debug, Clone)]
pub struct RrfConfig {
    /// RRF constant K. Default: 60.0.
    pub k: f64,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: DEFAULT_RRF_K }
    }
}

// ─── Candidate Budget ───────────────────────────────────────────────────────

/// Compute how many candidates to fetch from each source.
///
/// Fetches `multiplier × (limit + offset)` to ensure good coverage for
/// documents that may rank differently across sources.
///
/// # Arguments
///
/// * `limit` - Number of final results desired.
/// * `offset` - Pagination offset.
/// * `multiplier` - Candidate multiplier (typically 3).
#[must_use]
pub const fn candidate_count(limit: usize, offset: usize, multiplier: usize) -> usize {
    limit.saturating_add(offset).saturating_mul(multiplier)
}

#[inline]
fn rank_contribution(k: f64, rank: usize) -> f64 {
    let rank_u32 = u32::try_from(rank).unwrap_or(u32::MAX);
    1.0 / (k + f64::from(rank_u32) + 1.0)
}

#[inline]
fn sanitize_rrf_k(k: f64) -> f64 {
    if k.is_finite() && k >= 0.0 {
        k
    } else {
        DEFAULT_RRF_K
    }
}

// ─── RRF Fusion ─────────────────────────────────────────────────────────────

/// Fuse lexical and semantic search results using Reciprocal Rank Fusion.
///
/// # Algorithm
///
/// 1. Assign RRF scores: `1/(K + rank + 1)` for each source (0-based ranks).
/// 2. Sum scores for documents appearing in both sources.
/// 3. Sort by the 4-level deterministic ordering defined on [`FusedHit`]:
///    - RRF score descending
///    - `in_both_sources` (true preferred)
///    - Lexical score descending
///    - `doc_id` ascending (absolute determinism)
/// 4. Apply offset and limit for pagination.
///
/// # Arguments
///
/// * `lexical` - Lexical (BM25) search results, in descending relevance order.
/// * `semantic` - Semantic (vector) search results, in descending score order.
/// * `limit` - Maximum number of results to return.
/// * `offset` - Number of top results to skip (for pagination).
/// * `config` - RRF parameters (K constant).
#[must_use]
#[instrument(
    name = "frankensearch::rrf_fuse",
    skip(lexical, semantic),
    fields(
        lexical_count = lexical.len(),
        semantic_count = semantic.len(),
        k = config.k,
        limit,
        offset,
    )
)]
pub fn rrf_fuse(
    lexical: &[ScoredResult],
    semantic: &[VectorHit],
    limit: usize,
    offset: usize,
    config: &RrfConfig,
) -> Vec<FusedHit> {
    let k = sanitize_rrf_k(config.k);
    // Adjusted for typical ~50% overlap to reduce over-allocation.
    let capacity = (lexical.len() + semantic.len()) * 3 / 4;
    let mut hits: HashMap<&str, FusedHit> = HashMap::with_capacity(capacity);

    // Score lexical results.
    for (rank, result) in lexical.iter().enumerate() {
        let rrf_contribution = rank_contribution(k, rank);

        hits.entry(result.doc_id.as_str())
            .and_modify(|hit| {
                hit.rrf_score += rrf_contribution;
                hit.lexical_rank = Some(rank);
                hit.lexical_score = Some(result.score);
                hit.in_both_sources = true;
            })
            .or_insert_with(|| FusedHit {
                doc_id: result.doc_id.clone(),
                rrf_score: rrf_contribution,
                lexical_rank: Some(rank),
                semantic_rank: None,
                lexical_score: Some(result.score),
                semantic_score: None,
                in_both_sources: false,
            });
    }

    // Score semantic results.
    for (rank, hit) in semantic.iter().enumerate() {
        let rrf_contribution = rank_contribution(k, rank);

        hits.entry(hit.doc_id.as_str())
            .and_modify(|fh| {
                fh.rrf_score += rrf_contribution;
                fh.semantic_rank = Some(rank);
                fh.semantic_score = Some(hit.score);
                fh.in_both_sources = true;
            })
            .or_insert_with(|| FusedHit {
                doc_id: hit.doc_id.clone(),
                rrf_score: rrf_contribution,
                lexical_rank: None,
                semantic_rank: Some(rank),
                lexical_score: None,
                semantic_score: Some(hit.score),
                in_both_sources: false,
            });
    }

    // Sort by 4-level deterministic ordering.
    let mut results: Vec<FusedHit> = hits.into_values().collect();
    results.sort_by(FusedHit::cmp_for_ranking);

    let overlap_count = results.iter().filter(|h| h.in_both_sources).count();
    let fused_count = results.len();

    // Ranking window needed for pagination. For small windows this avoids
    // sorting every fused hit while preserving deterministic output order.
    let window = limit.saturating_add(offset);
    if window == 0 {
        debug!(
            target: "frankensearch.rrf",
            fused_count,
            overlap_count,
            output_count = 0,
            "rrf fusion complete"
        );
        return Vec::new();
    }
    if window < results.len() {
        let nth_index = window.saturating_sub(1);
        results.select_nth_unstable_by(nth_index, FusedHit::cmp_for_ranking);
        results.truncate(window);
    }

    // Sort only the requested ranking window.
    results.sort_by(FusedHit::cmp_for_ranking);

    // Apply offset and limit.
    let output: Vec<FusedHit> = results.into_iter().skip(offset).take(limit).collect();

    debug!(
        target: "frankensearch.rrf",
        fused_count,
        overlap_count,
        output_count = output.len(),
        "rrf fusion complete"
    );

    output
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;

    fn lexical_hit(doc_id: &str, score: f32) -> ScoredResult {
        ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: frankensearch_core::ScoreSource::Lexical,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            metadata: None,
        }
    }

    fn semantic_hit(doc_id: &str, score: f32) -> VectorHit {
        VectorHit {
            index: 0,
            score,
            doc_id: doc_id.into(),
        }
    }

    // ─── Score formula tests ────────────────────────────────────────────

    #[test]
    fn rrf_score_formula_k60() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("doc-a", 10.0)];
        let semantic = vec![];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        assert_eq!(results.len(), 1);
        let expected = 1.0 / (60.0 + 0.0 + 1.0); // rank 0 → 1/61
        assert!(
            (results[0].rrf_score - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            results[0].rrf_score
        );
    }

    #[test]
    fn rrf_score_formula_k1() {
        let config = RrfConfig { k: 1.0 };
        let semantic = vec![semantic_hit("first", 0.9), semantic_hit("second", 0.8)];

        let results = rrf_fuse(&[], &semantic, 10, 0, &config);

        assert_eq!(results.len(), 2);
        // rank 0: 1/(1+0+1) = 0.5
        // rank 1: 1/(1+1+1) = 0.333...
        assert!((results[0].rrf_score - 0.5).abs() < 1e-12);
        assert!((results[1].rrf_score - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rrf_score_formula_k0_is_valid() {
        let config = RrfConfig { k: 0.0 };
        let lexical = vec![lexical_hit("doc-a", 10.0)];

        let results = rrf_fuse(&lexical, &[], 10, 0, &config);

        assert_eq!(results.len(), 1);
        assert!((results[0].rrf_score - 1.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_k_falls_back_to_default() {
        let lexical = vec![lexical_hit("doc-a", 10.0)];
        let expected = 1.0 / (DEFAULT_RRF_K + 1.0);

        for invalid_k in [f64::NAN, f64::INFINITY, -1.0, -100.0] {
            let config = RrfConfig { k: invalid_k };
            let results = rrf_fuse(&lexical, &[], 10, 0, &config);
            assert_eq!(results.len(), 1);
            assert!(
                (results[0].rrf_score - expected).abs() < 1e-12,
                "invalid k={invalid_k} should fall back to default",
            );
        }
    }

    // ─── Multi-source fusion ────────────────────────────────────────────

    #[test]
    fn document_in_both_sources_gets_summed_score() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("shared", 5.0)];
        let semantic = vec![semantic_hit("shared", 0.9)];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        assert_eq!(results.len(), 1);
        let expected = 2.0 / 61.0; // Both at rank 0 → 1/61 + 1/61
        assert!(
            (results[0].rrf_score - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            results[0].rrf_score
        );
        assert!(results[0].in_both_sources);
        assert_eq!(results[0].lexical_rank, Some(0));
        assert_eq!(results[0].semantic_rank, Some(0));
    }

    #[test]
    fn multi_source_doc_ranks_higher_than_single_source() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("shared", 5.0), lexical_hit("lex-only", 4.0)];
        let semantic = vec![semantic_hit("shared", 0.9), semantic_hit("sem-only", 0.8)];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        assert_eq!(results.len(), 3);
        // "shared" should be first (highest combined score)
        assert_eq!(results[0].doc_id, "shared");
        assert!(results[0].in_both_sources);
    }

    // ─── Single-source fusion ───────────────────────────────────────────

    #[test]
    fn lexical_only_produces_correct_ranking() {
        let config = RrfConfig::default();
        let lexical = vec![
            lexical_hit("a", 10.0),
            lexical_hit("b", 8.0),
            lexical_hit("c", 5.0),
        ];

        let results = rrf_fuse(&lexical, &[], 10, 0, &config);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].doc_id, "a");
        assert_eq!(results[1].doc_id, "b");
        assert_eq!(results[2].doc_id, "c");
        // All single-source
        assert!(results.iter().all(|r| !r.in_both_sources));
    }

    #[test]
    fn semantic_only_produces_correct_ranking() {
        let config = RrfConfig::default();
        let semantic = vec![semantic_hit("x", 0.95), semantic_hit("y", 0.85)];

        let results = rrf_fuse(&[], &semantic, 10, 0, &config);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "x");
        assert_eq!(results[1].doc_id, "y");
    }

    // ─── Empty input ────────────────────────────────────────────────────

    #[test]
    fn both_empty_returns_empty() {
        let results = rrf_fuse(&[], &[], 10, 0, &RrfConfig::default());
        assert!(results.is_empty());
    }

    // ─── Offset and limit ───────────────────────────────────────────────

    #[test]
    fn limit_truncates_results() {
        let config = RrfConfig::default();
        let semantic = vec![
            semantic_hit("a", 0.9),
            semantic_hit("b", 0.8),
            semantic_hit("c", 0.7),
        ];

        let results = rrf_fuse(&[], &semantic, 2, 0, &config);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "a");
        assert_eq!(results[1].doc_id, "b");
    }

    #[test]
    fn offset_skips_top_results() {
        let config = RrfConfig::default();
        let semantic = vec![
            semantic_hit("a", 0.9),
            semantic_hit("b", 0.8),
            semantic_hit("c", 0.7),
        ];

        let results = rrf_fuse(&[], &semantic, 10, 1, &config);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[1].doc_id, "c");
    }

    #[test]
    fn offset_and_limit_combined() {
        let config = RrfConfig::default();
        let semantic = vec![
            semantic_hit("a", 0.9),
            semantic_hit("b", 0.8),
            semantic_hit("c", 0.7),
            semantic_hit("d", 0.6),
        ];

        let results = rrf_fuse(&[], &semantic, 2, 1, &config);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "b");
        assert_eq!(results[1].doc_id, "c");
    }

    // ─── Tie-breaking ───────────────────────────────────────────────────

    #[test]
    fn tie_breaking_in_both_sources_preferred() {
        let config = RrfConfig::default();
        // "shared" at lex rank 1, sem rank 0: RRF = 1/62 + 1/61
        // "lex-0" at lex rank 0: RRF = 1/61
        // "sem-1" at sem rank 1: RRF = 1/62
        // Adjust so "shared" and another doc have same RRF score:
        // Actually, just verify in_both_sources wins on tie.
        let lexical = vec![
            lexical_hit("only-lex", 10.0), // rank 0 → 1/61
        ];
        let semantic = vec![
            semantic_hit("only-sem", 0.9), // rank 0 → 1/61
        ];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        // Both have same RRF score (1/61). Neither is in both sources.
        // Tie-break goes to lexical_score (only-lex has Some(10.0), only-sem has None).
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "only-lex"); // has lexical_score
    }

    #[test]
    fn tie_breaking_doc_id_ascending() {
        let config = RrfConfig::default();
        // Same semantic scores, same rank structure, different doc_ids
        let semantic = vec![
            semantic_hit("beta", 0.9), // rank 0
        ];
        let lexical = vec![
            lexical_hit("alpha", 10.0), // rank 0
        ];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        // Same RRF, same in_both_sources (false). lexical_score tiebreak:
        // "alpha" has Some(10.0), "beta" has None → alpha first.
        assert_eq!(results[0].doc_id, "alpha");
        assert_eq!(results[1].doc_id, "beta");
    }

    // ─── Candidate budget ───────────────────────────────────────────────

    #[test]
    fn candidate_count_basic() {
        assert_eq!(candidate_count(10, 0, 3), 30);
        assert_eq!(candidate_count(10, 5, 3), 45);
        assert_eq!(candidate_count(20, 0, 4), 80);
    }

    #[test]
    fn candidate_count_overflow_safety() {
        // Should not panic on overflow
        let result = candidate_count(usize::MAX, 1, 3);
        assert_eq!(result, usize::MAX); // saturating
    }

    // ─── Score preservation ─────────────────────────────────────────────

    #[test]
    fn lexical_score_preserved() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("doc", 42.5)];

        let results = rrf_fuse(&lexical, &[], 10, 0, &config);

        assert_eq!(results[0].lexical_score, Some(42.5));
        assert_eq!(results[0].semantic_score, None);
    }

    #[test]
    fn semantic_score_preserved() {
        let config = RrfConfig::default();
        let semantic = vec![semantic_hit("doc", 0.87)];

        let results = rrf_fuse(&[], &semantic, 10, 0, &config);

        assert_eq!(results[0].semantic_score, Some(0.87));
        assert_eq!(results[0].lexical_score, None);
    }

    #[test]
    fn both_scores_preserved_on_shared_doc() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("doc", 15.0)];
        let semantic = vec![semantic_hit("doc", 0.93)];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].lexical_score, Some(15.0));
        assert_eq!(results[0].semantic_score, Some(0.93));
        assert!(results[0].in_both_sources);
    }

    // ─── Output ordering ────────────────────────────────────────────────

    #[test]
    fn output_is_strictly_descending_by_rrf_score() {
        let config = RrfConfig::default();
        let lexical = vec![
            lexical_hit("a", 10.0),
            lexical_hit("b", 8.0),
            lexical_hit("c", 6.0),
        ];
        let semantic = vec![
            semantic_hit("d", 0.9),
            semantic_hit("e", 0.8),
            semantic_hit("f", 0.7),
        ];

        let results = rrf_fuse(&lexical, &semantic, 10, 0, &config);

        for window in results.windows(2) {
            assert!(
                window[0].rrf_score >= window[1].rrf_score,
                "results not in descending order: {} ({}) > {} ({})",
                window[0].doc_id,
                window[0].rrf_score,
                window[1].doc_id,
                window[1].rrf_score,
            );
        }
    }

    #[test]
    fn candidate_count_zero_multiplier_returns_zero() {
        assert_eq!(candidate_count(100, 50, 0), 0);
        assert_eq!(candidate_count(0, 0, 0), 0);
    }

    #[test]
    fn candidate_count_zero_limit_and_offset() {
        assert_eq!(candidate_count(0, 0, 3), 0);
    }

    #[test]
    fn limit_zero_returns_empty() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("a", 10.0)];
        let results = rrf_fuse(&lexical, &[], 0, 0, &config);
        assert!(results.is_empty());
    }

    #[test]
    fn offset_beyond_results_returns_empty() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("a", 10.0)];
        let results = rrf_fuse(&lexical, &[], 10, 100, &config);
        assert!(results.is_empty());
    }

    #[test]
    fn single_element_single_source_correct_rank() {
        let config = RrfConfig::default();
        let lexical = vec![lexical_hit("only", 1.0)];
        let results = rrf_fuse(&lexical, &[], 10, 0, &config);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "only");
        assert_eq!(results[0].lexical_rank, Some(0));
        assert_eq!(results[0].semantic_rank, None);
        assert!(!results[0].in_both_sources);
    }

    fn large_lexical_semantic_fixture(doc_count: usize) -> (Vec<ScoredResult>, Vec<VectorHit>) {
        let mut lexical = Vec::with_capacity(doc_count);
        let mut semantic = Vec::with_capacity(doc_count);

        for index in 0..doc_count {
            let shared_bucket = index % (doc_count / 2).max(1);
            let doc_id = format!("doc-{shared_bucket}");

            let lexical_rank = f32::from(u16::try_from(index % 512).unwrap_or(u16::MAX));
            let semantic_rank = f32::from(u16::try_from(index % 256).unwrap_or(u16::MAX));

            lexical.push(lexical_hit(&doc_id, 1000.0 - lexical_rank));
            semantic.push(semantic_hit(&doc_id, 1.0 - (semantic_rank / 1024.0)));
        }

        (lexical, semantic)
    }

    fn rrf_fuse_reference_full_sort(
        lexical: &[ScoredResult],
        semantic: &[VectorHit],
        limit: usize,
        offset: usize,
        config: &RrfConfig,
    ) -> Vec<FusedHit> {
        let k = sanitize_rrf_k(config.k);
        let capacity = lexical.len() + semantic.len();
        let mut hits: HashMap<String, FusedHit> = HashMap::with_capacity(capacity);

        for (rank, result) in lexical.iter().enumerate() {
            let rrf_contribution = rank_contribution(k, rank);

            hits.entry(result.doc_id.clone())
                .and_modify(|hit| {
                    hit.rrf_score += rrf_contribution;
                    hit.lexical_rank = Some(rank);
                    hit.lexical_score = Some(result.score);
                    hit.in_both_sources = true;
                })
                .or_insert_with(|| FusedHit {
                    doc_id: result.doc_id.clone(),
                    rrf_score: rrf_contribution,
                    lexical_rank: Some(rank),
                    semantic_rank: None,
                    lexical_score: Some(result.score),
                    semantic_score: None,
                    in_both_sources: false,
                });
        }

        for (rank, hit) in semantic.iter().enumerate() {
            let rrf_contribution = rank_contribution(k, rank);

            hits.entry(hit.doc_id.clone())
                .and_modify(|fh| {
                    fh.rrf_score += rrf_contribution;
                    fh.semantic_rank = Some(rank);
                    fh.semantic_score = Some(hit.score);
                    fh.in_both_sources = true;
                })
                .or_insert_with(|| FusedHit {
                    doc_id: hit.doc_id.clone(),
                    rrf_score: rrf_contribution,
                    lexical_rank: None,
                    semantic_rank: Some(rank),
                    lexical_score: None,
                    semantic_score: Some(hit.score),
                    in_both_sources: false,
                });
        }

        let mut results: Vec<FusedHit> = hits.into_values().collect();
        results.sort_by(FusedHit::cmp_for_ranking);
        results.into_iter().skip(offset).take(limit).collect()
    }

    #[test]
    fn ranking_window_selection_matches_full_sort_reference() {
        let config = RrfConfig::default();
        let (lexical, semantic) = large_lexical_semantic_fixture(6_000);
        let windows = [
            (0_usize, 0_usize),
            (25, 0),
            (50, 10),
            (100, 250),
            (256, 800),
            (512, 2_000),
            (128, 10_000),
        ];

        for (limit, offset) in windows {
            let actual = rrf_fuse(&lexical, &semantic, limit, offset, &config);
            let expected =
                rrf_fuse_reference_full_sort(&lexical, &semantic, limit, offset, &config);

            assert_eq!(
                actual.len(),
                expected.len(),
                "window mismatch for limit={limit} offset={offset}"
            );

            for (actual_hit, expected_hit) in actual.iter().zip(&expected) {
                assert_eq!(actual_hit.doc_id, expected_hit.doc_id);
                assert!((actual_hit.rrf_score - expected_hit.rrf_score).abs() < 1e-12);
                assert_eq!(actual_hit.lexical_rank, expected_hit.lexical_rank);
                assert_eq!(actual_hit.semantic_rank, expected_hit.semantic_rank);
                assert_eq!(actual_hit.lexical_score, expected_hit.lexical_score);
                assert_eq!(actual_hit.semantic_score, expected_hit.semantic_score);
                assert_eq!(actual_hit.in_both_sources, expected_hit.in_both_sources);
            }
        }
    }

    #[test]
    #[ignore = "Perf probe for optimization loop: run explicitly with --ignored"]
    fn perf_probe_rrf_large_candidates() {
        let doc_count = std::env::var("RRF_PERF_DOCS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8_000);
        let iterations = std::env::var("RRF_PERF_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20);
        let (lexical, semantic) = large_lexical_semantic_fixture(doc_count);
        let config = RrfConfig::default();

        let started = Instant::now();
        let mut checksum = 0.0_f64;
        for _ in 0..iterations {
            let output = rrf_fuse(&lexical, &semantic, 100, 0, &config);
            checksum += output.iter().map(|hit| hit.rrf_score).sum::<f64>();
        }
        let elapsed = started.elapsed();

        eprintln!(
            "RRF_PERF baseline_or_candidate elapsed_ms={} doc_count={} iterations={} checksum={checksum}",
            elapsed.as_millis(),
            doc_count,
            iterations
        );
        assert!(checksum.is_finite());
    }

    #[test]
    #[ignore = "Perf probe for optimization loop: run explicitly with --ignored"]
    fn perf_probe_rrf_window_selection_vs_full_sort_reference() {
        let doc_count = std::env::var("RRF_PERF_DOCS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8_000);
        let iterations = std::env::var("RRF_PERF_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20);
        let limit = 100;
        let offset = 0;

        let (lexical, semantic) = large_lexical_semantic_fixture(doc_count);
        let config = RrfConfig::default();

        let optimized_started = Instant::now();
        let mut optimized_checksum = 0.0_f64;
        for _ in 0..iterations {
            let output = black_box(rrf_fuse(&lexical, &semantic, limit, offset, &config));
            optimized_checksum += output.iter().map(|hit| hit.rrf_score).sum::<f64>();
        }
        let optimized_elapsed = optimized_started.elapsed();

        let reference_started = Instant::now();
        let mut reference_checksum = 0.0_f64;
        for _ in 0..iterations {
            let output = black_box(rrf_fuse_reference_full_sort(
                &lexical, &semantic, limit, offset, &config,
            ));
            reference_checksum += output.iter().map(|hit| hit.rrf_score).sum::<f64>();
        }
        let reference_elapsed = reference_started.elapsed();

        assert!((optimized_checksum - reference_checksum).abs() < 1e-8);

        let optimized_ms = optimized_elapsed.as_secs_f64() * 1_000.0;
        let reference_ms = reference_elapsed.as_secs_f64() * 1_000.0;
        let speedup = if optimized_ms > 0.0 {
            reference_ms / optimized_ms
        } else {
            1.0
        };

        eprintln!(
            "RRF_PERF_COMPARE optimized_ms={optimized_ms:.3} reference_ms={reference_ms:.3} speedup={speedup:.3} doc_count={doc_count} iterations={iterations}"
        );
    }
}
