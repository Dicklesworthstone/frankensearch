//! Score normalization utilities for fusion and blending.
//!
//! Different retrieval sources emit scores on different scales (for example
//! unbounded BM25 versus bounded cosine similarity), so normalization is often
//! required before weighted blending. RRF itself is rank-based and does not
//! require normalization.

use serde::{Deserialize, Serialize};

const NON_FINITE_FALLBACK: f32 = 0.0;
const DEGENERATE_VALUE: f32 = 0.5;
const Z_SCORE_CLIP_SIGMAS: f32 = 3.0;
const NUMERIC_EPSILON: f32 = 1e-10;

/// Supported normalization strategies for score vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NormalizationMethod {
    /// Min-max normalization into `[0, 1]`.
    #[default]
    MinMax,
    /// Z-score normalization mapped into `[0, 1]` after clipping to ±3σ.
    ZScore,
    /// Leave scores unchanged.
    None,
}

/// In-place min-max normalization.
///
/// Finite values are scaled into `[0, 1]`. Non-finite values (`NaN`/`±∞`) are
/// mapped to `0.0`. If all finite values are effectively identical, finite
/// values are mapped to `0.5`.
pub fn min_max_normalize(scores: &mut [f32]) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut saw_finite = false;

    for &value in scores.iter() {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
            saw_finite = true;
        }
    }

    if !saw_finite {
        scores.fill(NON_FINITE_FALLBACK);
        return;
    }

    let range = max - min;
    if range.abs() <= NUMERIC_EPSILON {
        for score in scores.iter_mut() {
            *score = if score.is_finite() {
                DEGENERATE_VALUE
            } else {
                NON_FINITE_FALLBACK
            };
        }
        return;
    }

    for score in scores.iter_mut() {
        if score.is_finite() {
            *score = ((*score - min) / range).clamp(0.0, 1.0);
        } else {
            *score = NON_FINITE_FALLBACK;
        }
    }
}

/// In-place z-score normalization.
///
/// Finite values are standardized and then linearly mapped to `[0, 1]` by
/// clipping z-scores to ±3σ and applying `(z + 3) / 6`. Non-finite values are
/// mapped to `0.0`. If standard deviation is effectively zero, finite values
/// are mapped to `0.5`.
pub fn z_score_normalize(scores: &mut [f32]) {
    let mut count = 0.0_f32;
    let mut mean = 0.0_f32;
    let mut m2 = 0.0_f32;

    // Welford running variance for numerical stability.
    for &value in scores.iter() {
        if value.is_finite() {
            count += 1.0;
            let delta = value - mean;
            mean += delta / count;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }
    }

    if count <= NUMERIC_EPSILON {
        scores.fill(NON_FINITE_FALLBACK);
        return;
    }

    let std_dev = (m2 / count).sqrt();
    if std_dev <= NUMERIC_EPSILON {
        for score in scores.iter_mut() {
            *score = if score.is_finite() {
                DEGENERATE_VALUE
            } else {
                NON_FINITE_FALLBACK
            };
        }
        return;
    }

    let denominator = 2.0 * Z_SCORE_CLIP_SIGMAS;
    for score in scores.iter_mut() {
        if score.is_finite() {
            let z = (*score - mean) / std_dev;
            let clipped = z.clamp(-Z_SCORE_CLIP_SIGMAS, Z_SCORE_CLIP_SIGMAS);
            *score = (clipped + Z_SCORE_CLIP_SIGMAS) / denominator;
        } else {
            *score = NON_FINITE_FALLBACK;
        }
    }
}

/// Applies a selected normalization method in-place.
pub fn normalize_in_place(scores: &mut [f32], method: NormalizationMethod) {
    match method {
        NormalizationMethod::MinMax => min_max_normalize(scores),
        NormalizationMethod::ZScore => z_score_normalize(scores),
        NormalizationMethod::None => {}
    }
}

/// Returns min-max normalized scores.
#[must_use]
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    normalize_scores_with_method(scores, NormalizationMethod::MinMax)
}

/// Returns normalized scores using the provided method.
#[must_use]
pub fn normalize_scores_with_method(scores: &[f32], method: NormalizationMethod) -> Vec<f32> {
    let mut normalized = scores.to_vec();
    normalize_in_place(&mut normalized, method);
    normalized
}

#[cfg(test)]
mod tests {
    use super::{
        NormalizationMethod, min_max_normalize, normalize_in_place, normalize_scores,
        normalize_scores_with_method, z_score_normalize,
    };

    const EPSILON: f32 = 1e-6;

    fn assert_approx_slice(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*lhs - *rhs).abs() <= EPSILON,
                "index {idx}: {lhs} != {rhs} within {EPSILON}"
            );
        }
    }

    #[test]
    fn min_max_normalize_spans_unit_interval() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn min_max_normalize_identical_values_to_midpoint() {
        let mut scores = vec![3.0, 3.0, 3.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn min_max_normalize_handles_non_finite_values() {
        let mut scores = vec![5.0, f32::NAN, f32::INFINITY, 10.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn z_score_normalize_zero_variance_to_midpoint() {
        let mut scores = vec![42.0, 42.0, 42.0];
        z_score_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn z_score_normalize_symmetric_distribution_centers_at_half() {
        let mut scores = vec![-1.0, 0.0, 1.0];
        z_score_normalize(&mut scores);

        let mean = scores.iter().copied().sum::<f32>() / 3.0;
        assert!((mean - 0.5).abs() <= EPSILON);
        assert!(scores[0] < scores[1] && scores[1] < scores[2]);
    }

    #[test]
    fn normalize_in_place_none_keeps_scores() {
        let original = vec![0.1, 0.4, 0.9];
        let mut scores = original.clone();

        normalize_in_place(&mut scores, NormalizationMethod::None);
        assert_approx_slice(&scores, &original);
    }

    #[test]
    fn normalize_scores_uses_min_max() {
        let scores = vec![10.0, 20.0, 30.0];
        let normalized = normalize_scores(&scores);
        assert_approx_slice(&normalized, &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn normalize_scores_with_method_respects_selection() {
        let scores = vec![1.0, 2.0, 3.0];
        let min_max = normalize_scores_with_method(&scores, NormalizationMethod::MinMax);
        let z_score = normalize_scores_with_method(&scores, NormalizationMethod::ZScore);

        assert!(min_max[0] < min_max[1] && min_max[1] < min_max[2]);
        assert!(z_score[0] < z_score[1] && z_score[1] < z_score[2]);
    }

    #[test]
    fn min_max_normalize_all_negative_values() {
        let mut scores = vec![-5.0, -3.0, -1.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn z_score_normalize_all_negative_values_preserves_order() {
        let mut scores = vec![-10.0, -5.0, -1.0];
        z_score_normalize(&mut scores);
        assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)));
        assert!(scores[0] < scores[1] && scores[1] < scores[2]);
    }

    #[test]
    fn min_max_normalize_single_element() {
        let mut scores = vec![42.0];
        min_max_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5]);
    }

    #[test]
    fn z_score_normalize_single_element() {
        let mut scores = vec![42.0];
        z_score_normalize(&mut scores);
        assert_approx_slice(&scores, &[0.5]);
    }

    #[test]
    fn min_max_normalize_empty_is_noop() {
        let mut scores: Vec<f32> = vec![];
        min_max_normalize(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn z_score_normalize_empty_is_noop() {
        let mut scores: Vec<f32> = vec![];
        z_score_normalize(&mut scores);
        assert!(scores.is_empty());
    }

    #[test]
    fn min_max_normalize_all_nan_maps_to_zero() {
        let mut scores = vec![f32::NAN, f32::NAN, f32::NAN];
        min_max_normalize(&mut scores);
        assert!(scores.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn z_score_normalize_all_nan_maps_to_zero() {
        let mut scores = vec![f32::NAN, f32::NAN, f32::NAN];
        z_score_normalize(&mut scores);
        assert!(scores.iter().all(|s| *s == 0.0));
    }
}
