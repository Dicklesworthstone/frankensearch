//! Maximum Marginal Relevance (MMR) for diversified ranking.
//!
//! Re-ranks search results to balance relevance with diversity, preventing
//! near-duplicate results from dominating the top-k. Based on
//! Carbonell & Goldberg (1998).
//!
//! # Algorithm
//!
//! ```text
//! MMR(d) = lambda * Rel(d, q) - (1 - lambda) * max_{d' in S} Sim(d, d')
//! ```
//!
//! Where `Rel(d, q)` is the existing relevance score (RRF/blend) and
//! `Sim(d, d')` is cosine similarity between document embeddings.
//!
//! # Example
//!
//! ```
//! use frankensearch_fusion::mmr::{MmrConfig, mmr_rerank};
//!
//! // Relevance scores and embeddings for 4 candidates.
//! let scores = vec![0.9, 0.85, 0.84, 0.5];
//! let embeddings: Vec<Vec<f32>> = vec![
//!     vec![1.0, 0.0, 0.0],
//!     vec![0.99, 0.1, 0.0],  // near-duplicate of doc 0
//!     vec![0.0, 1.0, 0.0],   // diverse
//!     vec![0.0, 0.0, 1.0],   // diverse
//! ];
//! let emb_refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
//!
//! let config = MmrConfig { enabled: true, lambda: 0.7, candidate_pool: 30 };
//! let selected = mmr_rerank(&scores, &emb_refs, 3, &config);
//!
//! // Doc 0 selected first (highest relevance).
//! assert_eq!(selected[0], 0);
//! // Doc 1 (near-duplicate) should be penalized; doc 2 (diverse) preferred.
//! assert_eq!(selected[1], 2);
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for MMR diversified ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmrConfig {
    /// Enable MMR reranking. Default: false.
    pub enabled: bool,

    /// Relevance vs diversity tradeoff.
    /// `1.0` = pure relevance (no diversity), `0.0` = pure diversity.
    /// Clamped to `[0.0, 1.0]`. Default: 0.7.
    pub lambda: f64,

    /// Maximum candidate pool size for MMR consideration.
    /// Only the top `candidate_pool` candidates (by relevance score) are
    /// considered for MMR reranking. Default: 30.
    pub candidate_pool: usize,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lambda: 0.7,
            candidate_pool: 30,
        }
    }
}

impl MmrConfig {
    /// Returns lambda clamped to `[0.0, 1.0]`.
    #[must_use]
    pub const fn clamped_lambda(&self) -> f64 {
        // NaN comparisons are always false, so NaN would fall through to
        // the else branch returning NaN. Guard explicitly.
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            0.0
        } else if self.lambda > 1.0 {
            1.0
        } else {
            self.lambda
        }
    }
}

/// Greedy MMR selection.
///
/// Given `n` candidates with relevance scores and embedding vectors, selects
/// `k` candidates that balance relevance with diversity.
///
/// Returns a `Vec<usize>` of indices into the input arrays, in MMR-selected order.
///
/// # Arguments
///
/// * `scores` - Relevance scores for each candidate (higher is better).
/// * `embeddings` - Embedding vectors for each candidate (for inter-doc similarity).
/// * `k` - Number of results to select.
/// * `config` - MMR configuration.
///
/// # Panics
///
/// Panics if `scores.len() != embeddings.len()`.
#[must_use]
pub fn mmr_rerank(
    scores: &[f64],
    embeddings: &[&[f32]],
    k: usize,
    config: &MmrConfig,
) -> Vec<usize> {
    assert_eq!(
        scores.len(),
        embeddings.len(),
        "scores and embeddings must have the same length"
    );

    let n = scores.len().min(config.candidate_pool);
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let k = k.min(n);
    let lambda = config.clamped_lambda();
    let diversity_weight = 1.0 - lambda;

    // Normalize relevance scores to [0, 1] for fair comparison with cosine sim.
    let (min_score, max_score) = scores[..n]
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &s| {
            (mn.min(s), mx.max(s))
        });
    let score_range = max_score - min_score;
    let norm_scores: Vec<f64> = scores[..n]
        .iter()
        .map(|&s| {
            if score_range < f64::EPSILON {
                1.0 // all scores equal
            } else {
                (s - min_score) / score_range
            }
        })
        .collect();

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<bool> = vec![true; n];

    // Select first document: pure relevance (highest score).
    let first = norm_scores[..n]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i);

    selected.push(first);
    remaining[first] = false;

    // Greedy selection for remaining k-1 slots.
    for _ in 1..k {
        let mut best_idx = usize::MAX;
        let mut best_mmr = f64::NEG_INFINITY;

        for i in 0..n {
            if !remaining[i] {
                continue;
            }

            // Max similarity to any already-selected document.
            let max_sim = selected
                .iter()
                .map(|&j| cosine_sim(embeddings[i], embeddings[j]))
                .fold(f64::NEG_INFINITY, f64::max);

            let mmr = lambda.mul_add(norm_scores[i], -(diversity_weight * max_sim));

            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = i;
            }
        }

        if best_idx == usize::MAX {
            break;
        }

        selected.push(best_idx);
        remaining[best_idx] = false;
    }

    selected
}

/// Cosine similarity between two vectors, as f64.
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for i in 0..len {
        let ai = f64::from(a[i]);
        let bi = f64::from(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }

    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    // ── Basic selection ──────────────────────────────────────────────────

    #[test]
    fn first_selected_is_highest_relevance() {
        let scores = vec![0.5, 0.9, 0.7];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.7,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 1, &config);
        assert_eq!(selected, vec![1]); // index 1 has highest score (0.9)
    }

    #[test]
    fn lambda_one_is_pure_relevance() {
        let scores = vec![0.9, 0.85, 0.84, 0.5];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0], // near-duplicate
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 1.0,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 4, &config);
        // Pure relevance ordering: 0, 1, 2, 3
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    // ── Diversity effect ─────────────────────────────────────────────────

    #[test]
    fn diversity_penalizes_near_duplicates() {
        let scores = vec![0.9, 0.85, 0.84, 0.5];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0], // near-dup of 0
            vec![0.0, 1.0, 0.0],  // diverse
            vec![0.0, 0.0, 1.0],  // diverse
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.5, // 50/50 relevance/diversity
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        assert_eq!(selected[0], 0); // highest relevance
        // Doc 1 (near-dup, score 0.85) should lose to doc 2 (diverse, score 0.84)
        assert_eq!(selected[1], 2);
    }

    #[test]
    fn low_lambda_maximizes_diversity() {
        let scores = vec![1.0, 0.9, 0.8, 0.7];
        // Two clusters: {0,1} similar, {2,3} similar
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.95, 0.05],
            vec![0.0, 1.0],
            vec![0.05, 0.95],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.1, // heavily favor diversity
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 2, &config);
        assert_eq!(selected[0], 0); // first is always highest relevance
        // Second should be from the other cluster (2 or 3)
        assert!(selected[1] == 2 || selected[1] == 3);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn empty_input() {
        let config = MmrConfig::default();
        assert!(mmr_rerank(&[], &[], 5, &config).is_empty());
    }

    #[test]
    fn k_zero() {
        let scores = vec![1.0];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig::default();

        assert!(mmr_rerank(&scores, &refs, 0, &config).is_empty());
    }

    #[test]
    fn k_greater_than_n() {
        let scores = vec![0.9, 0.5];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.7,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 10, &config);
        assert_eq!(selected.len(), 2); // capped at n
    }

    #[test]
    fn single_candidate() {
        let scores = vec![0.8];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.7,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 1, &config);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn equal_scores_all_diverse() {
        let scores = vec![0.5, 0.5, 0.5];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.5,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        assert_eq!(selected.len(), 3);
        // All should be selected (all equally relevant and diverse)
    }

    #[test]
    fn identical_embeddings_still_works() {
        let scores = vec![0.9, 0.85, 0.8];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0], // identical to 0
            vec![1.0, 0.0], // identical to 0
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.5,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        assert_eq!(selected.len(), 3);
        // All identical, so MMR degrades to relevance ordering
        assert_eq!(selected[0], 0);
    }

    #[test]
    fn candidate_pool_limits_consideration() {
        let scores = vec![0.9, 0.85, 0.8, 0.7, 0.6];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![0.3, 0.7],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.7,
            candidate_pool: 3, // only consider top 3
        };

        let selected = mmr_rerank(&scores, &refs, 5, &config);
        // Can only select from indices 0..3
        assert_eq!(selected.len(), 3);
        for &idx in &selected {
            assert!(idx < 3);
        }
    }

    // ── Cosine similarity helper ─────────────────────────────────────────

    #[test]
    fn cosine_sim_identical() {
        let a = vec![1.0_f32, 0.0, 0.0];
        assert!(approx_eq(cosine_sim(&a, &a), 1.0));
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert!(approx_eq(cosine_sim(&a, &b), 0.0));
    }

    #[test]
    fn cosine_sim_opposite() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        assert!(approx_eq(cosine_sim(&a, &b), -1.0));
    }

    #[test]
    fn cosine_sim_empty() {
        let empty: Vec<f32> = vec![];
        assert!(approx_eq(cosine_sim(&empty, &empty), 0.0));
    }

    #[test]
    fn cosine_sim_zero_vector() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        assert!(approx_eq(cosine_sim(&a, &b), 0.0));
    }

    // ── Config ───────────────────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let config = MmrConfig::default();
        assert!(!config.enabled);
        assert!((config.lambda - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.candidate_pool, 30);
    }

    #[test]
    fn clamped_lambda_bounds() {
        let config = MmrConfig {
            lambda: -0.5,
            ..Default::default()
        };
        assert!(approx_eq(config.clamped_lambda(), 0.0));

        let config = MmrConfig {
            lambda: 1.5,
            ..Default::default()
        };
        assert!(approx_eq(config.clamped_lambda(), 1.0));

        let config = MmrConfig {
            lambda: 0.7,
            ..Default::default()
        };
        assert!(approx_eq(config.clamped_lambda(), 0.7));
    }

    #[test]
    fn serde_roundtrip() {
        let config = MmrConfig {
            enabled: true,
            lambda: 0.6,
            candidate_pool: 50,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: MmrConfig = serde_json::from_str(&json).unwrap();
        assert!(decoded.enabled);
        assert!((decoded.lambda - 0.6).abs() < f64::EPSILON);
        assert_eq!(decoded.candidate_pool, 50);
    }

    // ── No-duplicate guarantee ───────────────────────────────────────────

    #[test]
    fn no_duplicates_in_output() {
        let scores = vec![0.9, 0.88, 0.87, 0.85, 0.84];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.9, 0.1],
            vec![0.0, 0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.5,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 5, &config);
        let mut seen = std::collections::HashSet::new();
        for &idx in &selected {
            assert!(seen.insert(idx), "duplicate index {idx} in output");
        }
    }

    // ─── bd-2gcl tests begin ───

    #[test]
    fn mmr_config_debug_format() {
        let config = MmrConfig {
            enabled: true,
            lambda: 0.65,
            candidate_pool: 25,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("true"));
        assert!(debug.contains("0.65"));
        assert!(debug.contains("25"));
    }

    #[test]
    fn clamped_lambda_at_exact_boundaries() {
        let zero = MmrConfig {
            lambda: 0.0,
            ..Default::default()
        };
        assert!(approx_eq(zero.clamped_lambda(), 0.0));

        let one = MmrConfig {
            lambda: 1.0,
            ..Default::default()
        };
        assert!(approx_eq(one.clamped_lambda(), 1.0));
    }

    #[test]
    fn lambda_zero_is_pure_diversity() {
        let scores = vec![0.9, 0.5, 0.8];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],  // orthogonal to 0
            vec![0.99, 0.1], // near-duplicate of 0
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.0, // pure diversity
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        // First pick: highest normalized score (index 0, norm=1.0)
        assert_eq!(selected[0], 0);
        // Second pick: doc 1 (orthogonal) should beat doc 2 (near-dup) on diversity
        assert_eq!(selected[1], 1);
    }

    #[test]
    fn cosine_sim_different_lengths_uses_min() {
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0];
        // Only first 2 elements used: dot=1, norm_a=1, norm_b=1 -> cosine=1.0
        assert!(approx_eq(cosine_sim(&a, &b), 1.0));
    }

    #[test]
    fn all_identical_scores_normalized_to_one() {
        // When all scores are identical, norm_scores should all be 1.0
        // and MMR should still work (degrades to diversity ordering after first pick)
        let scores = vec![0.5, 0.5, 0.5];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.5,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn negative_scores_normalize_correctly() {
        let scores = vec![-0.5, -0.1, -0.9];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0, 1.0]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 1.0, // pure relevance
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        // Highest score is -0.1 (index 1)
        assert_eq!(selected[0], 1);
    }

    #[test]
    fn candidate_pool_one_selects_best_only() {
        let scores = vec![0.9, 0.8, 0.7];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.7,
            candidate_pool: 1,
        };

        let selected = mmr_rerank(&scores, &refs, 3, &config);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], 0);
    }

    #[test]
    fn two_candidates_k_two_selects_both() {
        let scores = vec![0.9, 0.5];
        let embeddings: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.7,
            ..Default::default()
        };

        let selected = mmr_rerank(&scores, &refs, 2, &config);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn cosine_sim_non_unit_vectors() {
        // Scaling vectors shouldn't change cosine similarity
        let a = vec![2.0_f32, 0.0];
        let b = vec![3.0_f32, 0.0];
        assert!(approx_eq(cosine_sim(&a, &b), 1.0));

        let c = vec![5.0_f32, 5.0];
        let d = vec![1.0_f32, 1.0];
        assert!(approx_eq(cosine_sim(&c, &d), 1.0));
    }

    #[test]
    fn selected_indices_within_pool_bounds() {
        let scores = vec![0.9, 0.85, 0.8, 0.75, 0.7, 0.65];
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
            vec![0.0, 0.5, 0.5],
            vec![0.5, 0.0, 0.5],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(std::vec::Vec::as_slice).collect();
        let config = MmrConfig {
            enabled: true,
            lambda: 0.6,
            candidate_pool: 4,
        };

        let selected = mmr_rerank(&scores, &refs, 6, &config);
        for &idx in &selected {
            assert!(idx < 4, "index {idx} exceeds candidate_pool=4");
        }
    }

    // ─── bd-2gcl tests end ───
}
