//! Pseudo-relevance feedback (PRF) query expansion.
//!
//! After Phase 1 returns initial results, PRF computes a weighted centroid
//! of top-k feedback embeddings and interpolates it with the original quality-tier
//! query embedding. This nudges the Phase 2 query toward the neighborhood of
//! relevant documents found in Phase 1.
//!
//! Based on the Rocchio algorithm (1971), adapted for two-tier neural search.
//!
//! # Example
//!
//! ```
//! use frankensearch_fusion::prf::{PrfConfig, prf_expand};
//!
//! let original = vec![1.0, 0.0, 0.0];
//! let feedback = vec![
//!     (vec![0.0, 1.0, 0.0], 0.9),
//!     (vec![0.0, 0.0, 1.0], 0.5),
//! ];
//! let refs: Vec<(&[f32], f64)> = feedback.iter()
//!     .map(|(e, w)| (e.as_slice(), *w))
//!     .collect();
//!
//! let config = PrfConfig::default();
//! let expanded = prf_expand(&original, &refs, config.alpha);
//! assert!(expanded.is_some());
//! // Expanded embedding is L2-normalized.
//! let v = expanded.unwrap();
//! let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
//! assert!((norm - 1.0).abs() < 1e-5);
//! ```

use serde::{Deserialize, Serialize};

use frankensearch_core::QueryClass;

/// Configuration for pseudo-relevance feedback query expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrfConfig {
    /// Enable PRF expansion. Default: false.
    pub enabled: bool,

    /// Interpolation weight for the original embedding.
    /// `expanded = alpha * original + (1 - alpha) * centroid`.
    /// Clamped to `[0.5, 1.0]`. Default: 0.8.
    pub alpha: f64,

    /// Number of Phase 1 results to use as feedback.
    /// Default: 5.
    pub top_k_feedback: usize,

    /// Minimum feedback documents required to attempt expansion.
    /// If fewer than this many results are available, expansion is skipped.
    /// Default: 3.
    pub min_feedback_docs: usize,

    /// Weight the centroid by RRF scores (true) or use uniform weights (false).
    /// Default: true.
    pub score_weighted: bool,
}

impl Default for PrfConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alpha: 0.8,
            top_k_feedback: 5,
            min_feedback_docs: 3,
            score_weighted: true,
        }
    }
}

impl PrfConfig {
    /// Returns alpha clamped to `[0.5, 1.0]`.
    #[must_use]
    pub const fn clamped_alpha(&self) -> f64 {
        // f64::clamp propagates NaN; guard explicitly to prevent
        // NaN-poisoned embeddings from escaping as valid results.
        if !self.alpha.is_finite() {
            0.8
        } else if self.alpha < 0.5 {
            0.5
        } else if self.alpha > 1.0 {
            1.0
        } else {
            self.alpha
        }
    }

    /// Check whether PRF should activate for this query class.
    ///
    /// PRF only activates for `NaturalLanguage` queries. Short keywords and
    /// identifiers already have precise embeddings that don't benefit from
    /// centroid nudging.
    #[must_use]
    pub const fn should_expand(&self, query_class: &QueryClass) -> bool {
        self.enabled && matches!(query_class, QueryClass::NaturalLanguage)
    }
}

/// Compute PRF-expanded embedding by interpolating with a weighted centroid.
///
/// Returns `None` if:
/// - `feedback_embeddings` is empty
/// - All weights are zero
/// - The resulting embedding has zero magnitude (degenerate input)
///
/// The returned embedding is L2-normalized for cosine similarity compatibility.
///
/// # Arguments
///
/// * `original_embedding` - The quality-tier query embedding.
/// * `feedback_embeddings` - Pairs of (embedding, weight). Weights are typically
///   RRF scores from Phase 1. If all weights are equal, the centroid is uniform.
/// * `alpha` - Interpolation weight for the original embedding, clamped to `[0.5, 1.0]`.
///   `expanded = alpha * original + (1 - alpha) * centroid`.
#[must_use]
pub fn prf_expand(
    original_embedding: &[f32],
    feedback_embeddings: &[(&[f32], f64)],
    alpha: f64,
) -> Option<Vec<f32>> {
    if feedback_embeddings.is_empty() {
        return None;
    }

    let dims = original_embedding.len();
    let alpha = alpha.clamp(0.5, 1.0);
    let beta = 1.0 - alpha;

    // Compute weighted centroid.
    let total_weight: f64 = feedback_embeddings.iter().map(|(_, w)| w.max(0.0)).sum();
    if total_weight < f64::EPSILON {
        return None;
    }

    let mut centroid = vec![0.0_f32; dims];
    for (emb, weight) in feedback_embeddings {
        #[allow(clippy::cast_possible_truncation)]
        let w = (weight.max(0.0) / total_weight) as f32;
        let len = emb.len().min(dims);
        for j in 0..len {
            centroid[j] = emb[j].mul_add(w, centroid[j]);
        }
    }

    // Interpolate: expanded = alpha * original + beta * centroid.
    #[allow(clippy::cast_possible_truncation)]
    let alpha_f32 = alpha as f32;
    #[allow(clippy::cast_possible_truncation)]
    let beta_f32 = beta as f32;
    let mut expanded = vec![0.0_f32; dims];
    for i in 0..dims {
        expanded[i] = alpha_f32 * original_embedding[i] + beta_f32 * centroid[i];
    }

    // L2-normalize the result.
    let norm_sq: f32 = expanded.iter().map(|x| x * x).sum();
    if norm_sq < f32::EPSILON {
        return None;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    for v in &mut expanded {
        *v *= inv_norm;
    }

    Some(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn is_l2_normalized(v: &[f32]) -> bool {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        (norm - 1.0).abs() < 1e-5
    }

    // ── alpha=1.0 produces no expansion ──────────────────────────────────

    #[test]
    fn alpha_one_returns_original() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 1.0)];
        let result = prf_expand(&original, &feedback, 1.0).unwrap();

        // alpha=1.0 means 100% original, 0% centroid.
        // After L2 normalization, should be [1, 0, 0].
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 0.0));
        assert!(approx_eq(result[2], 0.0));
    }

    // ── alpha=0.5 (minimum) produces max expansion ──────────────────────

    #[test]
    fn alpha_half_max_expansion() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 1.0)];
        let result = prf_expand(&original, &feedback, 0.5).unwrap();

        // 0.5 * [1,0,0] + 0.5 * [0,1,0] = [0.5, 0.5, 0] → normalized
        assert!(is_l2_normalized(&result));
        assert!(approx_eq(result[0], result[1])); // equal contribution
    }

    // ── alpha below 0.5 is clamped ──────────────────────────────────────

    #[test]
    fn alpha_clamped_below() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 1.0)];

        // alpha=0.0 should be clamped to 0.5
        let result_clamped = prf_expand(&original, &feedback, 0.0).unwrap();
        let result_half = prf_expand(&original, &feedback, 0.5).unwrap();

        for (a, b) in result_clamped.iter().zip(result_half.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    // ── alpha above 1.0 is clamped ──────────────────────────────────────

    #[test]
    fn alpha_clamped_above() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 1.0)];

        let result_clamped = prf_expand(&original, &feedback, 2.0).unwrap();
        let result_one = prf_expand(&original, &feedback, 1.0).unwrap();

        for (a, b) in result_clamped.iter().zip(result_one.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    // ── Score-weighted vs uniform centroid ────────────────────────────────

    #[test]
    fn score_weighted_centroid_differs_from_uniform() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback_weighted: Vec<(&[f32], f64)> = vec![
            (&[0.0, 1.0, 0.0], 0.9), // high weight
            (&[0.0, 0.0, 1.0], 0.1), // low weight
        ];
        let feedback_uniform: Vec<(&[f32], f64)> = vec![
            (&[0.0, 1.0, 0.0], 1.0), // equal
            (&[0.0, 0.0, 1.0], 1.0), // equal
        ];

        let result_w = prf_expand(&original, &feedback_weighted, 0.5).unwrap();
        let result_u = prf_expand(&original, &feedback_uniform, 0.5).unwrap();

        // Weighted should lean more toward [0,1,0] (weight 0.9 vs 0.1).
        assert!(result_w[1] > result_u[1]);
        assert!(result_w[2] < result_u[2]);
    }

    // ── Empty feedback returns None ──────────────────────────────────────

    #[test]
    fn empty_feedback_returns_none() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![];
        assert!(prf_expand(&original, &feedback, 0.8).is_none());
    }

    // ── All-zero weights returns None ────────────────────────────────────

    #[test]
    fn zero_weights_returns_none() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 0.0)];
        assert!(prf_expand(&original, &feedback, 0.8).is_none());
    }

    // ── Negative weights treated as zero ─────────────────────────────────

    #[test]
    fn negative_weights_treated_as_zero() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], -1.0), (&[0.0, 0.0, 1.0], -0.5)];
        assert!(prf_expand(&original, &feedback, 0.8).is_none());
    }

    // ── Output is always L2-normalized ───────────────────────────────────

    #[test]
    fn output_is_l2_normalized() {
        let original = vec![3.0, 4.0, 0.0]; // not normalized
        let feedback: Vec<(&[f32], f64)> = vec![(&[1.0, 2.0, 3.0], 0.8), (&[0.5, 0.5, 0.5], 0.3)];
        let result = prf_expand(&original, &feedback, 0.7).unwrap();
        assert!(is_l2_normalized(&result));
    }

    // ── Multiple feedback documents ──────────────────────────────────────

    #[test]
    fn multiple_feedback_docs() {
        let original = vec![1.0, 0.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![
            (&[0.0, 1.0, 0.0, 0.0], 0.9),
            (&[0.0, 0.0, 1.0, 0.0], 0.7),
            (&[0.0, 0.0, 0.0, 1.0], 0.5),
        ];
        let result = prf_expand(&original, &feedback, 0.8).unwrap();

        assert!(is_l2_normalized(&result));
        assert_eq!(result.len(), 4);
        // Original dimension should dominate (alpha=0.8).
        assert!(result[0] > result[1]);
        assert!(result[0] > result[2]);
        assert!(result[0] > result[3]);
    }

    // ── Dimension mismatch truncates to original ─────────────────────────

    #[test]
    fn feedback_shorter_than_original() {
        let original = vec![1.0, 0.0, 0.0, 0.0];
        let short = vec![0.0, 1.0]; // only 2 dims
        let feedback: Vec<(&[f32], f64)> = vec![(short.as_slice(), 1.0)];
        let result = prf_expand(&original, &feedback, 0.5).unwrap();

        assert_eq!(result.len(), 4);
        assert!(is_l2_normalized(&result));
        // dims 2,3 get no centroid contribution, so they stay near zero.
    }

    // ── PrfConfig defaults ───────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let config = PrfConfig::default();
        assert!(!config.enabled);
        assert!((config.alpha - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.top_k_feedback, 5);
        assert_eq!(config.min_feedback_docs, 3);
        assert!(config.score_weighted);
    }

    // ── clamped_alpha ────────────────────────────────────────────────────

    #[test]
    fn clamped_alpha_works() {
        let mut config = PrfConfig::default();
        assert!((config.clamped_alpha() - 0.8).abs() < f64::EPSILON);

        config.alpha = 0.3;
        assert!((config.clamped_alpha() - 0.5).abs() < f64::EPSILON);

        config.alpha = 1.5;
        assert!((config.clamped_alpha() - 1.0).abs() < f64::EPSILON);
    }

    // ── should_expand query class guard ──────────────────────────────────

    #[test]
    fn should_expand_only_natural_language() {
        let config = PrfConfig {
            enabled: true,
            ..Default::default()
        };

        assert!(config.should_expand(&QueryClass::NaturalLanguage));
        assert!(!config.should_expand(&QueryClass::Identifier));
        assert!(!config.should_expand(&QueryClass::ShortKeyword));
        assert!(!config.should_expand(&QueryClass::Empty));
    }

    #[test]
    fn should_expand_disabled() {
        let config = PrfConfig::default(); // enabled: false
        assert!(!config.should_expand(&QueryClass::NaturalLanguage));
    }

    // ── Serde roundtrip ──────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let config = PrfConfig {
            enabled: true,
            alpha: 0.75,
            top_k_feedback: 10,
            min_feedback_docs: 5,
            score_weighted: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: PrfConfig = serde_json::from_str(&json).unwrap();
        assert!(decoded.enabled);
        assert!((decoded.alpha - 0.75).abs() < f64::EPSILON);
        assert_eq!(decoded.top_k_feedback, 10);
        assert_eq!(decoded.min_feedback_docs, 5);
        assert!(!decoded.score_weighted);
    }

    // ── Single feedback document ─────────────────────────────────────────

    #[test]
    fn single_feedback_doc() {
        let original = vec![1.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0], 1.0)];
        let result = prf_expand(&original, &feedback, 0.8).unwrap();

        assert!(is_l2_normalized(&result));
        // 0.8 * [1,0] + 0.2 * [0,1] = [0.8, 0.2] → normalized
        let expected_norm = 0.8_f32.hypot(0.2);
        assert!(approx_eq(result[0], 0.8 / expected_norm));
        assert!(approx_eq(result[1], 0.2 / expected_norm));
    }

    // ─── bd-2fuy tests begin ───

    #[test]
    fn prf_config_debug_format() {
        let config = PrfConfig {
            enabled: true,
            alpha: 0.75,
            top_k_feedback: 8,
            min_feedback_docs: 4,
            score_weighted: false,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("true"));
        assert!(debug.contains("0.75"));
        assert!(debug.contains('8'));
        assert!(debug.contains('4'));
        assert!(debug.contains("false"));
    }

    #[test]
    fn zero_magnitude_original_with_feedback_works() {
        let original = vec![0.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 1.0)];
        let result = prf_expand(&original, &feedback, 0.5).unwrap();
        // 0.5*[0,0,0] + 0.5*[0,1,0] = [0, 0.5, 0] → normalized to [0, 1, 0]
        assert!(is_l2_normalized(&result));
        assert!(approx_eq(result[1], 1.0));
    }

    #[test]
    fn identical_original_and_feedback_returns_original_direction() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[1.0, 0.0, 0.0], 1.0)];
        let result = prf_expand(&original, &feedback, 0.8).unwrap();
        // alpha*[1,0,0] + (1-alpha)*[1,0,0] = [1,0,0] → normalized
        assert!(is_l2_normalized(&result));
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 0.0));
    }

    #[test]
    fn clamped_alpha_at_exact_boundaries() {
        let config = PrfConfig {
            alpha: 0.5,
            ..Default::default()
        };
        assert!((config.clamped_alpha() - 0.5).abs() < f64::EPSILON);

        let config = PrfConfig {
            alpha: 1.0,
            ..Default::default()
        };
        assert!((config.clamped_alpha() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn expanded_has_same_dimensionality() {
        let original = vec![1.0, 0.5, 0.3, 0.1, 0.7];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.2, 0.8, 0.1, 0.9, 0.4], 0.7)];
        let result = prf_expand(&original, &feedback, 0.8).unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn mixed_positive_and_negative_weights() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![
            (&[0.0, 1.0, 0.0], -0.5), // negative → treated as 0
            (&[0.0, 0.0, 1.0], 1.0),  // positive → only this counts
        ];
        let result = prf_expand(&original, &feedback, 0.5).unwrap();
        assert!(is_l2_normalized(&result));
        // Centroid should only come from the second doc [0,0,1]
        // so result[1] should be ~0 (no contribution from dim 1)
        assert!(result[1].abs() < 1e-5);
    }

    #[test]
    fn feedback_longer_than_original_truncated() {
        let original = vec![1.0, 0.0]; // 2 dims
        let long_feedback = vec![0.0, 1.0, 0.5, 0.3]; // 4 dims
        let feedback: Vec<(&[f32], f64)> = vec![(long_feedback.as_slice(), 1.0)];
        let result = prf_expand(&original, &feedback, 0.5).unwrap();
        // Output should have 2 dims (same as original)
        assert_eq!(result.len(), 2);
        assert!(is_l2_normalized(&result));
    }

    #[test]
    fn high_dimensionality_384d_works() {
        let original: Vec<f32> = (0_u16..384).map(|i| (f32::from(i) * 0.01).sin()).collect();
        let feedback_emb: Vec<f32> = (0_u16..384).map(|i| (f32::from(i) * 0.02).cos()).collect();
        let feedback: Vec<(&[f32], f64)> = vec![(feedback_emb.as_slice(), 0.9)];
        let result = prf_expand(&original, &feedback, 0.8).unwrap();
        assert_eq!(result.len(), 384);
        assert!(is_l2_normalized(&result));
    }

    #[test]
    fn very_small_weights_still_produce_result() {
        let original = vec![1.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 1.0, 0.0], 1e-8)];
        let result = prf_expand(&original, &feedback, 0.8).unwrap();
        assert!(is_l2_normalized(&result));
    }

    #[test]
    fn all_zero_original_and_centroid_returns_none() {
        let original = vec![0.0, 0.0, 0.0];
        let feedback: Vec<(&[f32], f64)> = vec![(&[0.0, 0.0, 0.0], 1.0)];
        // 0.5*[0,0,0] + 0.5*[0,0,0] = [0,0,0] → zero magnitude → None
        assert!(prf_expand(&original, &feedback, 0.5).is_none());
    }

    // ─── bd-2fuy tests end ───
}
