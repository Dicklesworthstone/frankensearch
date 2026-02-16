//! Information retrieval evaluation metrics.
//!
//! Provides standard IR metrics for evaluating search quality:
//! - **nDCG@K**: Normalized Discounted Cumulative Gain
//! - **MAP@K**: Mean Average Precision at K
//! - **MRR**: Mean Reciprocal Rank
//! - **Recall@K**: Fraction of relevant documents retrieved in top-K

use std::collections::HashSet;

#[inline]
#[allow(clippy::cast_precision_loss)]
const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Normalized Discounted Cumulative Gain at K.
///
/// Measures ranking quality, giving higher weight to relevant documents
/// appearing earlier in the result list. Uses binary relevance (1.0 if in
/// `relevant`, 0.0 otherwise).
///
/// Returns 0.0 when `relevant` is empty or `k` is 0.
#[must_use]
pub fn ndcg_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() || k == 0 {
        return 0.0;
    }

    let limit = k.min(retrieved.len());
    let mut seen = HashSet::with_capacity(limit);

    // DCG: sum of 1/log2(rank+1) for relevant docs in retrieved
    let dcg: f64 = retrieved[..limit]
        .iter()
        .enumerate()
        .filter_map(|(i, doc)| {
            if !seen.insert(*doc) {
                return None;
            }
            if relevant_set.contains(doc) {
                Some(1.0 / (usize_to_f64(i) + 2.0).log2())
            } else {
                None
            }
        })
        .sum();

    // Ideal DCG: all relevant docs at top positions
    let ideal_count = k.min(relevant_set.len());
    let idcg: f64 = (0..ideal_count)
        .map(|i| 1.0 / (usize_to_f64(i) + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return 0.0;
    }

    dcg / idcg
}

/// Mean Average Precision at K.
///
/// Computes precision at each rank position where a relevant document appears,
/// then averages over the total number of relevant documents (capped at K).
///
/// Returns 0.0 when `relevant` is empty or `k` is 0.
#[must_use]
pub fn map_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() || k == 0 {
        return 0.0;
    }

    let limit = k.min(retrieved.len());
    let mut hits = 0_u32;
    let mut sum_precision = 0.0;
    let mut seen = HashSet::with_capacity(limit);

    for (i, doc) in retrieved[..limit].iter().enumerate() {
        if !seen.insert(*doc) {
            continue;
        }
        if relevant_set.contains(doc) {
            hits += 1;
            sum_precision += f64::from(hits) / (usize_to_f64(i) + 1.0);
        }
    }

    let denominator = usize_to_f64(k.min(relevant_set.len()));
    sum_precision / denominator
}

/// Mean Reciprocal Rank.
///
/// Returns 1/(rank of first relevant document). Returns 0.0 if no relevant
/// document appears in the retrieved list.
#[must_use]
pub fn mrr(retrieved: &[&str], relevant: &[&str]) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() {
        return 0.0;
    }

    let mut seen = HashSet::new();
    for (i, doc) in retrieved.iter().enumerate() {
        if !seen.insert(*doc) {
            continue;
        }
        if relevant_set.contains(doc) {
            return 1.0 / (usize_to_f64(i) + 1.0);
        }
    }
    0.0
}

/// Recall at K.
///
/// Fraction of relevant documents that appear in the top-K retrieved results.
/// Returns 0.0 when `relevant` is empty or `k` is 0.
#[must_use]
pub fn recall_at_k(retrieved: &[&str], relevant: &[&str], k: usize) -> f64 {
    let relevant_set: HashSet<&str> = relevant.iter().copied().collect();
    if relevant_set.is_empty() || k == 0 {
        return 0.0;
    }

    let limit = k.min(retrieved.len());
    let mut seen = HashSet::with_capacity(limit);
    let mut found = 0_usize;

    for doc in &retrieved[..limit] {
        if !seen.insert(*doc) {
            continue;
        }
        if relevant_set.contains(doc) {
            found += 1;
        }
    }

    usize_to_f64(found) / usize_to_f64(relevant_set.len())
}

// ---------------------------------------------------------------------------
// Bootstrap confidence intervals and statistical comparison
// ---------------------------------------------------------------------------

/// Confidence interval estimated via bootstrap resampling.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapCi {
    /// Mean of the observed scores.
    pub mean: f64,
    /// Standard error (std dev of bootstrap means).
    pub std_error: f64,
    /// Lower bound of the confidence interval.
    pub lower: f64,
    /// Upper bound of the confidence interval.
    pub upper: f64,
    /// Confidence level used (e.g. 0.95).
    pub confidence: f64,
    /// Number of bootstrap resamples performed.
    pub n_resamples: usize,
}

/// Comparison of two paired score distributions via bootstrap.
#[derive(Debug, Clone, Copy)]
pub struct BootstrapComparison {
    /// Mean of the first system's scores.
    pub mean_a: f64,
    /// Mean of the second system's scores.
    pub mean_b: f64,
    /// Observed mean difference (a − b).
    pub mean_diff: f64,
    /// Lower bound of CI on the difference.
    pub ci_lower: f64,
    /// Upper bound of CI on the difference.
    pub ci_upper: f64,
    /// Two-sided empirical p-value (shift method).
    pub p_value: f64,
    /// Whether the difference is significant at the given confidence level.
    pub significant: bool,
    /// Confidence level used.
    pub confidence: f64,
    /// Number of bootstrap resamples performed.
    pub n_resamples: usize,
}

/// Supported metric kinds for multi-metric quality comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMetric {
    /// Normalized Discounted Cumulative Gain at K.
    NdcgAtK(usize),
    /// Mean Average Precision at K.
    MapAtK(usize),
    /// Mean Reciprocal Rank.
    Mrr,
    /// Recall at K.
    RecallAtK(usize),
}

impl std::fmt::Display for QualityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NdcgAtK(k) => write!(f, "nDCG@{k}"),
            Self::MapAtK(k) => write!(f, "MAP@{k}"),
            Self::Mrr => write!(f, "MRR"),
            Self::RecallAtK(k) => write!(f, "Recall@{k}"),
        }
    }
}

/// Per-metric paired score samples for quality comparison.
#[derive(Debug, Clone, Copy)]
pub struct QualityMetricSamples<'a> {
    /// Metric represented by these score vectors.
    pub metric: QualityMetric,
    /// Scores for system A (e.g., fast model), one score per query.
    pub scores_a: &'a [f64],
    /// Scores for system B (e.g., quality model), one score per query.
    pub scores_b: &'a [f64],
}

/// Comparison result for a single quality metric.
#[derive(Debug, Clone, Copy)]
pub struct QualityMetricComparison {
    /// Metric that was compared.
    pub metric: QualityMetric,
    /// Bootstrap comparison result for this metric.
    pub comparison: BootstrapComparison,
}

/// Multi-metric quality comparison report.
#[derive(Debug, Clone)]
pub struct QualityComparison {
    /// Number of paired queries included in each metric comparison.
    pub query_count: usize,
    /// Confidence level used for all metric comparisons.
    pub confidence: f64,
    /// Number of bootstrap resamples used for all metric comparisons.
    pub n_resamples: usize,
    /// Per-metric bootstrap comparisons.
    pub metrics: Vec<QualityMetricComparison>,
}

impl QualityComparison {
    /// Render a deterministic tab-separated report for terminal/CI logs.
    #[must_use]
    pub fn render_tsv_report(&self) -> String {
        let mut out = String::from(
            "metric\tmean_a\tmean_b\tmean_diff\tci_lower\tci_upper\tp_value\tsignificant\n",
        );

        for item in &self.metrics {
            let row = format!(
                "{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\n",
                item.metric,
                item.comparison.mean_a,
                item.comparison.mean_b,
                item.comparison.mean_diff,
                item.comparison.ci_lower,
                item.comparison.ci_upper,
                item.comparison.p_value,
                item.comparison.significant
            );
            out.push_str(&row);
        }

        out
    }
}

/// Deterministic xorshift64 PRNG for reproducible bootstrap resampling.
///
/// Only needs uniformly-distributed indices — not cryptographic randomness.
struct Xorshift64(u64);

impl Xorshift64 {
    const fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x5EED_CAFE_BABE_D00D
        } else {
            seed
        })
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[allow(clippy::cast_possible_truncation)]
    const fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound as u64)) as usize
    }
}

fn slice_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / usize_to_f64(values.len())
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Linear interpolation percentile on a pre-sorted slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * usize_to_f64(sorted.len() - 1);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let lo = idx.floor() as usize;
    let frac = idx - usize_to_f64(lo);
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo].mul_add(1.0 - frac, sorted[hi] * frac)
}

/// Compute a bootstrap confidence interval for the mean of `scores`.
///
/// Uses the percentile method with `n_resamples` bootstrap iterations.
/// `confidence` is in (0, 1), e.g. 0.95 for 95% CI. `seed` provides
/// deterministic reproducibility.
///
/// Returns `None` if inputs are invalid (empty scores, bad confidence,
/// zero resamples).
#[must_use]
pub fn bootstrap_ci(
    scores: &[f64],
    confidence: f64,
    n_resamples: usize,
    seed: u64,
) -> Option<BootstrapCi> {
    if scores.is_empty()
        || !all_finite(scores)
        || n_resamples == 0
        || !(confidence > 0.0 && confidence < 1.0)
    {
        return None;
    }

    let observed_mean = slice_mean(scores);
    let n = scores.len();
    let mut rng = Xorshift64::new(seed);

    let mut bootstrap_means = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += scores[rng.next_index(n)];
        }
        bootstrap_means.push(sum / usize_to_f64(n));
    }

    bootstrap_means.sort_unstable_by(f64::total_cmp);

    let alpha = 1.0 - confidence;
    let lower = percentile_sorted(&bootstrap_means, alpha / 2.0);
    let upper = percentile_sorted(&bootstrap_means, 1.0 - alpha / 2.0);

    let bm = slice_mean(&bootstrap_means);
    let variance = if bootstrap_means.len() > 1 {
        bootstrap_means
            .iter()
            .map(|x| (x - bm).powi(2))
            .sum::<f64>()
            / usize_to_f64(bootstrap_means.len() - 1)
    } else {
        0.0
    };

    Some(BootstrapCi {
        mean: observed_mean,
        std_error: variance.sqrt(),
        lower,
        upper,
        confidence,
        n_resamples,
    })
}

/// Compare two paired score distributions via bootstrap.
///
/// `scores_a` and `scores_b` must have the same length (one score per query).
/// Computes a confidence interval on the paired difference (a − b) and an
/// empirical two-sided p-value via the shift method.
///
/// Returns `None` if inputs are invalid (empty, mismatched lengths, bad
/// confidence, zero resamples).
#[must_use]
pub fn bootstrap_compare(
    scores_a: &[f64],
    scores_b: &[f64],
    confidence: f64,
    n_resamples: usize,
    seed: u64,
) -> Option<BootstrapComparison> {
    if scores_a.is_empty()
        || scores_a.len() != scores_b.len()
        || !all_finite(scores_a)
        || !all_finite(scores_b)
        || n_resamples == 0
        || !(confidence > 0.0 && confidence < 1.0)
    {
        return None;
    }

    let diffs: Vec<f64> = scores_a
        .iter()
        .zip(scores_b.iter())
        .map(|(a, b)| a - b)
        .collect();
    let observed_diff = slice_mean(&diffs);
    let n = diffs.len();
    let mut rng = Xorshift64::new(seed);

    let mut bootstrap_diffs = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += diffs[rng.next_index(n)];
        }
        bootstrap_diffs.push(sum / usize_to_f64(n));
    }

    bootstrap_diffs.sort_unstable_by(f64::total_cmp);

    let alpha = 1.0 - confidence;
    let ci_lower = percentile_sorted(&bootstrap_diffs, alpha / 2.0);
    let ci_upper = percentile_sorted(&bootstrap_diffs, 1.0 - alpha / 2.0);

    // P-value via shift method: under H0 (mean_diff = 0), the null bootstrap
    // mean is (bootstrap_diff - observed_diff). Count how often the null
    // statistic is at least as extreme as the observed.
    let abs_obs = observed_diff.abs();
    let count_extreme = bootstrap_diffs
        .iter()
        .filter(|&&d| (d - observed_diff).abs() >= abs_obs)
        .count();
    // Plus-one correction (Davison & Hinkley, 1997) prevents p=0.0 from
    // finite samples: p = (count_extreme + 1) / (n_resamples + 1).
    let p_value = usize_to_f64(count_extreme + 1) / usize_to_f64(n_resamples + 1);

    let significant = p_value < alpha;

    Some(BootstrapComparison {
        mean_a: slice_mean(scores_a),
        mean_b: slice_mean(scores_b),
        mean_diff: observed_diff,
        ci_lower,
        ci_upper,
        p_value,
        significant,
        confidence,
        n_resamples,
    })
}

/// Produce a multi-metric quality comparison report using paired bootstrap tests.
///
/// Returns `None` if:
/// - no metrics are provided,
/// - score vectors are empty,
/// - score lengths mismatch across metrics or between systems,
/// - bootstrap parameters are invalid.
#[must_use]
pub fn quality_comparison(
    metric_samples: &[QualityMetricSamples<'_>],
    confidence: f64,
    n_resamples: usize,
    seed: u64,
) -> Option<QualityComparison> {
    let first = metric_samples.first()?;
    let query_count = first.scores_a.len();
    if query_count == 0 || first.scores_b.len() != query_count {
        return None;
    }

    let mut metrics = Vec::with_capacity(metric_samples.len());
    for (index, sample) in metric_samples.iter().enumerate() {
        if sample.scores_a.len() != query_count || sample.scores_b.len() != query_count {
            return None;
        }

        #[allow(clippy::cast_possible_truncation)]
        let metric_seed = seed.wrapping_add(index as u64);
        let comparison = bootstrap_compare(
            sample.scores_a,
            sample.scores_b,
            confidence,
            n_resamples,
            metric_seed,
        )?;
        metrics.push(QualityMetricComparison {
            metric: sample.metric,
            comparison,
        });
    }

    Some(QualityComparison {
        query_count,
        confidence,
        n_resamples,
        metrics,
    })
}

// ---------------------------------------------------------------------------
// Run-stability detection and outlier trimming (bd-2hz.9.9)
// ---------------------------------------------------------------------------

/// Verdict from a run-stability check.
///
/// Used as a pre-gate before statistical comparison: if a benchmark run has
/// too much variance or too few samples, the comparison is unreliable and
/// should be flagged rather than trusted.
#[derive(Debug, Clone, PartialEq)]
pub struct RunStabilityVerdict {
    /// Whether the run is stable enough for meaningful comparison.
    pub stable: bool,
    /// Coefficient of variation (`std_dev` / mean). `None` if mean is zero.
    pub cv: Option<f64>,
    /// Number of samples after outlier removal (if trimming was applied).
    pub effective_sample_count: usize,
    /// Number of outliers detected (before removal).
    pub outlier_count: usize,
    /// Human-readable reason if `stable` is false.
    pub reason: String,
}

/// Coefficient of variation: `std_dev / |mean|`.
///
/// Returns `None` if the sample is empty or the mean is zero (CV is undefined
/// when the mean is zero because it would require division by zero).
#[must_use]
pub fn coefficient_of_variation(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() || !all_finite(samples) {
        return None;
    }
    let mean = slice_mean(samples);
    if mean.abs() < f64::EPSILON {
        return None;
    }
    let n = usize_to_f64(samples.len());
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    Some(variance.sqrt() / mean.abs())
}

/// Detect outlier indices using the IQR fence method.
///
/// An observation is an outlier if it falls outside `[Q1 - k*IQR, Q3 + k*IQR]`.
/// The standard choice is `k = 1.5` (mild outliers) or `k = 3.0` (extreme).
///
/// Returns sorted indices of outliers in the original `samples` slice.
/// Returns an empty vec if `samples` has fewer than 4 elements (IQR is
/// unreliable with very small samples).
#[must_use]
pub fn detect_outliers_iqr(samples: &[f64], iqr_factor: f64) -> Vec<usize> {
    if samples.len() < 4 || !iqr_factor.is_finite() || iqr_factor < 0.0 {
        return Vec::new();
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);

    let q1 = percentile_sorted(&sorted, 0.25);
    let q3 = percentile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;

    let lower_fence = iqr_factor.mul_add(-iqr, q1);
    let upper_fence = iqr_factor.mul_add(iqr, q3);

    let mut outliers: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter(|(_, v)| **v < lower_fence || **v > upper_fence)
        .map(|(i, _)| i)
        .collect();
    outliers.sort_unstable();
    outliers
}

/// Remove IQR outliers from samples and return the trimmed set.
///
/// Uses [`detect_outliers_iqr`] with the given `iqr_factor` to identify
/// outliers, then returns only the non-outlier values (preserving order).
///
/// Returns all samples unchanged if fewer than 4 elements (IQR unreliable).
#[must_use]
pub fn trim_outliers(samples: &[f64], iqr_factor: f64) -> Vec<f64> {
    let outlier_indices = detect_outliers_iqr(samples, iqr_factor);
    if outlier_indices.is_empty() {
        return samples.to_vec();
    }
    let outlier_set: HashSet<usize> = outlier_indices.into_iter().collect();
    samples
        .iter()
        .enumerate()
        .filter(|(i, _)| !outlier_set.contains(i))
        .map(|(_, &v)| v)
        .collect()
}

/// Verify that a benchmark run is stable enough for meaningful comparison.
///
/// Checks two conditions:
/// 1. **Minimum sample count**: at least `min_samples` observations after
///    outlier removal (with IQR factor 1.5).
/// 2. **Maximum CV**: coefficient of variation on trimmed samples must not
///    exceed `max_cv` (e.g. 0.15 for 15% relative spread).
///
/// If either check fails, the verdict explains why and `stable` is `false`.
#[must_use]
pub fn verify_run_stability(
    samples: &[f64],
    max_cv: f64,
    min_samples: usize,
) -> RunStabilityVerdict {
    if samples.is_empty() {
        return RunStabilityVerdict {
            stable: false,
            cv: None,
            effective_sample_count: 0,
            outlier_count: 0,
            reason: "no samples provided".to_owned(),
        };
    }

    if !all_finite(samples) {
        return RunStabilityVerdict {
            stable: false,
            cv: None,
            effective_sample_count: samples.iter().filter(|x| x.is_finite()).count(),
            outlier_count: 0,
            reason: "samples contain non-finite values".to_owned(),
        };
    }

    let outlier_indices = detect_outliers_iqr(samples, 1.5);
    let outlier_count = outlier_indices.len();
    let trimmed = if outlier_indices.is_empty() {
        samples.to_vec()
    } else {
        let outlier_set: HashSet<usize> = outlier_indices.into_iter().collect();
        samples
            .iter()
            .enumerate()
            .filter(|(i, _)| !outlier_set.contains(i))
            .map(|(_, &v)| v)
            .collect()
    };
    let effective_count = trimmed.len();

    if effective_count < min_samples {
        return RunStabilityVerdict {
            stable: false,
            cv: coefficient_of_variation(&trimmed),
            effective_sample_count: effective_count,
            outlier_count,
            reason: format!(
                "insufficient samples after outlier removal: {effective_count} < {min_samples} \
                 ({outlier_count} outliers removed from {} total)",
                samples.len()
            ),
        };
    }

    let cv = coefficient_of_variation(&trimmed);
    match cv {
        Some(cv_val) if cv_val > max_cv => RunStabilityVerdict {
            stable: false,
            cv: Some(cv_val),
            effective_sample_count: effective_count,
            outlier_count,
            reason: format!(
                "coefficient of variation {cv_val:.4} exceeds threshold {max_cv:.4} \
                 ({effective_count} samples, {outlier_count} outliers removed)"
            ),
        },
        _ => RunStabilityVerdict {
            stable: true,
            cv,
            effective_sample_count: effective_count,
            outlier_count,
            reason: String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── nDCG@K ─────────────────────────────────────────────────────────

    #[test]
    fn ndcg_perfect_ranking() {
        let retrieved = vec!["a", "b", "c"];
        let relevant = vec!["a", "b", "c"];
        let score = ndcg_at_k(&retrieved, &relevant, 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "perfect ranking should be 1.0, got {score}"
        );
    }

    #[test]
    fn ndcg_reversed_ranking() {
        // Relevant docs at bottom positions should score less than at top
        let good = ndcg_at_k(&["a", "b", "x"], &["a", "b"], 3);
        let bad = ndcg_at_k(&["x", "a", "b"], &["a", "b"], 3);
        assert!(
            good > bad,
            "top-ranked relevant docs should score higher: {good} vs {bad}"
        );
    }

    #[test]
    fn ndcg_empty_relevant() {
        assert!((ndcg_at_k(&["a", "b"], &[], 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_empty_retrieved() {
        assert!((ndcg_at_k(&[], &["a", "b"], 3)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_k_zero() {
        assert!((ndcg_at_k(&["a"], &["a"], 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_single_relevant_at_rank_1() {
        let score = ndcg_at_k(&["a"], &["a"], 10);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "single relevant doc at rank 1 should be 1.0"
        );
    }

    #[test]
    fn ndcg_no_overlap() {
        let score = ndcg_at_k(&["x", "y", "z"], &["a", "b"], 3);
        assert!(
            score.abs() < f64::EPSILON,
            "no overlap should be 0.0, got {score}"
        );
    }

    #[test]
    fn ndcg_duplicate_retrievals_count_once() {
        let score = ndcg_at_k(&["a", "a", "b"], &["a", "b"], 3);
        let expected = 1.5 / (1.0 + 1.0 / 3f64.log2());
        assert!(
            (score - expected).abs() < 1e-12,
            "duplicates in the ranking should not push nDCG above the ideal"
        );
    }

    // ─── MAP@K ──────────────────────────────────────────────────────────

    #[test]
    fn map_perfect_ranking() {
        let score = map_at_k(&["a", "b", "c"], &["a", "b", "c"], 3);
        // P@1=1, P@2=1, P@3=1 → AP = 3/3 = 1.0
        assert!(
            (score - 1.0).abs() < 1e-10,
            "perfect ranking should be 1.0, got {score}"
        );
    }

    #[test]
    fn map_one_relevant_at_top() {
        let score = map_at_k(&["a", "x", "y"], &["a"], 3);
        // P@1=1 → AP = 1/1 = 1.0
        assert!((score - 1.0).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn map_one_relevant_at_rank_3() {
        let score = map_at_k(&["x", "y", "a"], &["a"], 3);
        // P@3=1/3 → AP = (1/3)/1 = 0.333...
        assert!(
            (score - 1.0 / 3.0).abs() < 1e-10,
            "expected 0.333, got {score}"
        );
    }

    #[test]
    fn map_empty_relevant() {
        assert!(map_at_k(&["a", "b"], &[], 3).abs() < f64::EPSILON);
    }

    #[test]
    fn map_k_zero() {
        assert!(map_at_k(&["a"], &["a"], 0).abs() < f64::EPSILON);
    }

    #[test]
    fn map_no_overlap() {
        let score = map_at_k(&["x", "y"], &["a", "b"], 3);
        assert!(
            score.abs() < f64::EPSILON,
            "no overlap should be 0.0, got {score}"
        );
    }

    #[test]
    fn map_ignores_duplicate_docs() {
        let score = map_at_k(&["a", "a", "b"], &["a", "b"], 3);
        let expected = f64::midpoint(1.0, 2.0 / 3.0);
        assert!(
            (score - expected).abs() < 1e-12,
            "duplicate doc hit should only contribute once to average precision"
        );
    }

    // ─── MRR ────────────────────────────────────────────────────────────

    #[test]
    fn mrr_first_relevant_at_rank_1() {
        let score = mrr(&["a", "b", "c"], &["a"]);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mrr_first_relevant_at_rank_3() {
        let score = mrr(&["x", "y", "a"], &["a"]);
        assert!((score - 1.0 / 3.0).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn mrr_no_relevant() {
        let score = mrr(&["x", "y", "z"], &["a"]);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_empty_retrieved() {
        assert!(mrr(&[], &["a"]).abs() < f64::EPSILON);
    }

    // ─── Recall@K ───────────────────────────────────────────────────────

    #[test]
    fn recall_perfect() {
        let score = recall_at_k(&["a", "b", "c"], &["a", "b"], 3);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn recall_partial() {
        let score = recall_at_k(&["a", "x", "y"], &["a", "b"], 3);
        assert!((score - 0.5).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn recall_none() {
        let score = recall_at_k(&["x", "y", "z"], &["a", "b"], 3);
        assert!(score.abs() < f64::EPSILON);
    }

    #[test]
    fn recall_empty_relevant() {
        assert!(recall_at_k(&["a"], &[], 3).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_k_zero() {
        assert!(recall_at_k(&["a"], &["a"], 0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_k_limits_retrieved() {
        // Only look at top-2, so "b" at position 3 doesn't count
        let score = recall_at_k(&["a", "x", "b"], &["a", "b"], 2);
        assert!((score - 0.5).abs() < 1e-10, "got {score}");
    }

    #[test]
    fn recall_duplicate_documents_count_once() {
        let score = recall_at_k(&["a", "a", "b"], &["a", "b"], 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "duplicate hits should not inflate recall beyond 1.0"
        );
    }

    // ─── Bootstrap CI ─────────────────────────────────────────────────

    #[test]
    #[allow(clippy::float_cmp)]
    fn bootstrap_ci_deterministic() {
        let scores = vec![0.8, 0.6, 0.9, 0.7, 0.85];
        let ci1 = bootstrap_ci(&scores, 0.95, 1000, 42).unwrap();
        let ci2 = bootstrap_ci(&scores, 0.95, 1000, 42).unwrap();
        assert_eq!(ci1.mean, ci2.mean);
        assert_eq!(ci1.lower, ci2.lower);
        assert_eq!(ci1.upper, ci2.upper);
        assert_eq!(ci1.std_error, ci2.std_error);
    }

    #[test]
    fn bootstrap_ci_contains_mean() {
        let scores = vec![0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55, 0.65, 0.75, 0.85];
        let ci = bootstrap_ci(&scores, 0.95, 2000, 123).unwrap();
        assert!(
            ci.lower <= ci.mean && ci.mean <= ci.upper,
            "CI [{}, {}] should contain mean {}",
            ci.lower,
            ci.upper,
            ci.mean
        );
    }

    #[test]
    fn bootstrap_ci_identical_scores_narrow() {
        let scores = vec![0.5; 20];
        let ci = bootstrap_ci(&scores, 0.95, 1000, 99).unwrap();
        assert!((ci.lower - 0.5).abs() < 1e-10);
        assert!((ci.upper - 0.5).abs() < 1e-10);
        assert!(ci.std_error < 1e-10);
    }

    #[test]
    fn bootstrap_ci_rejects_empty() {
        assert!(bootstrap_ci(&[], 0.95, 1000, 42).is_none());
    }

    #[test]
    fn bootstrap_ci_rejects_bad_confidence() {
        let scores = vec![0.5, 0.6];
        assert!(bootstrap_ci(&scores, 0.0, 1000, 42).is_none());
        assert!(bootstrap_ci(&scores, 1.0, 1000, 42).is_none());
        assert!(bootstrap_ci(&scores, -0.1, 1000, 42).is_none());
        assert!(bootstrap_ci(&scores, f64::NAN, 1000, 42).is_none());
        assert!(bootstrap_ci(&scores, f64::INFINITY, 1000, 42).is_none());
    }

    #[test]
    fn bootstrap_ci_rejects_non_finite_scores() {
        assert!(bootstrap_ci(&[0.5, f64::NAN, 0.7], 0.95, 1000, 42).is_none());
        assert!(bootstrap_ci(&[0.5, f64::INFINITY, 0.7], 0.95, 1000, 42).is_none());
        assert!(bootstrap_ci(&[0.5, f64::NEG_INFINITY, 0.7], 0.95, 1000, 42).is_none());
    }

    #[test]
    fn bootstrap_ci_rejects_zero_resamples() {
        assert!(bootstrap_ci(&[0.5, 0.6], 0.95, 0, 42).is_none());
    }

    #[test]
    fn bootstrap_ci_single_score() {
        let ci = bootstrap_ci(&[0.75], 0.95, 1000, 42).unwrap();
        assert!((ci.mean - 0.75).abs() < 1e-10);
        assert!((ci.lower - 0.75).abs() < 1e-10);
        assert!((ci.upper - 0.75).abs() < 1e-10);
    }

    #[test]
    fn bootstrap_ci_wider_at_higher_confidence() {
        let scores = vec![0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.35, 0.55, 0.65, 0.45];
        let ci_99 = bootstrap_ci(&scores, 0.99, 2000, 42).unwrap();
        let ci_90 = bootstrap_ci(&scores, 0.90, 2000, 42).unwrap();
        let width_99 = ci_99.upper - ci_99.lower;
        let width_90 = ci_90.upper - ci_90.lower;
        assert!(
            width_99 > width_90,
            "99% CI width ({width_99}) should be wider than 90% ({width_90})"
        );
    }

    // ─── Bootstrap Compare ────────────────────────────────────────────

    #[test]
    fn bootstrap_compare_identical_not_significant() {
        let scores = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let cmp = bootstrap_compare(&scores, &scores, 0.95, 2000, 42).unwrap();
        assert!(
            !cmp.significant,
            "identical distributions should not be significant, p={}",
            cmp.p_value
        );
        assert!(cmp.mean_diff.abs() < 1e-10);
    }

    #[test]
    fn bootstrap_compare_clearly_different() {
        let better = vec![0.95, 0.80, 0.92, 0.75, 0.88, 0.90, 0.85, 0.93, 0.78, 0.87];
        let worse = vec![0.40, 0.30, 0.35, 0.25, 0.38, 0.42, 0.33, 0.28, 0.31, 0.37];
        let cmp = bootstrap_compare(&better, &worse, 0.95, 2000, 42).unwrap();
        assert!(
            cmp.significant,
            "clearly different distributions should be significant, p={}",
            cmp.p_value
        );
        assert!(cmp.mean_diff > 0.0);
        assert!(
            cmp.ci_lower > 0.0,
            "CI lower {} should be > 0 for clear difference",
            cmp.ci_lower
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn bootstrap_compare_deterministic() {
        let a = vec![0.8, 0.6, 0.9, 0.7];
        let b = vec![0.5, 0.4, 0.6, 0.3];
        let c1 = bootstrap_compare(&a, &b, 0.95, 1000, 77).unwrap();
        let c2 = bootstrap_compare(&a, &b, 0.95, 1000, 77).unwrap();
        assert_eq!(c1.p_value, c2.p_value);
        assert_eq!(c1.ci_lower, c2.ci_lower);
        assert_eq!(c1.ci_upper, c2.ci_upper);
    }

    #[test]
    fn bootstrap_compare_rejects_mismatched_lengths() {
        assert!(bootstrap_compare(&[0.5, 0.6], &[0.5], 0.95, 1000, 42).is_none());
    }

    #[test]
    fn bootstrap_compare_rejects_non_finite_scores() {
        assert!(bootstrap_compare(&[0.5, f64::NAN], &[0.4, 0.3], 0.95, 1000, 42).is_none());
        assert!(bootstrap_compare(&[0.5, 0.6], &[0.4, f64::INFINITY], 0.95, 1000, 42).is_none());
    }

    #[test]
    fn bootstrap_compare_rejects_empty() {
        assert!(bootstrap_compare(&[], &[], 0.95, 1000, 42).is_none());
    }

    #[test]
    fn bootstrap_compare_ci_contains_zero_for_similar() {
        let a = vec![0.50, 0.55, 0.60, 0.45, 0.50, 0.52, 0.48, 0.53, 0.47, 0.51];
        let b = vec![0.51, 0.54, 0.59, 0.46, 0.49, 0.53, 0.47, 0.52, 0.48, 0.50];
        let cmp = bootstrap_compare(&a, &b, 0.95, 2000, 42).unwrap();
        assert!(
            cmp.ci_lower <= 0.0 && cmp.ci_upper >= 0.0,
            "CI [{}, {}] should contain 0 for similar distributions",
            cmp.ci_lower,
            cmp.ci_upper
        );
    }

    #[test]
    fn bootstrap_compare_pvalue_never_zero() {
        // With clearly separated distributions, naive p = count/n would be 0.0.
        // Plus-one correction guarantees p >= 1/(n_resamples+1).
        let a = vec![0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90];
        let b = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10];
        let cmp = bootstrap_compare(&a, &b, 0.95, 1000, 42).unwrap();
        assert!(
            cmp.p_value > 0.0,
            "p-value must never be exactly 0.0 for finite samples, got {}",
            cmp.p_value
        );
        // Minimum p with plus-one correction: 1/1001
        let min_p = 1.0 / 1001.0;
        assert!(
            (cmp.p_value - min_p).abs() < 1e-10,
            "expected minimum p-value {min_p}, got {}",
            cmp.p_value
        );
    }

    // ─── Quality Comparison ───────────────────────────────────────────

    #[test]
    fn quality_comparison_multi_metric_report() {
        let ndcg_fast = [0.45, 0.50, 0.40, 0.55, 0.48, 0.52];
        let ndcg_quality = [0.70, 0.75, 0.68, 0.74, 0.72, 0.73];
        let recall_fast = [0.30, 0.40, 0.35, 0.42, 0.38, 0.36];
        let recall_quality = [0.65, 0.70, 0.66, 0.72, 0.68, 0.69];

        let samples = [
            QualityMetricSamples {
                metric: QualityMetric::NdcgAtK(10),
                scores_a: &ndcg_fast,
                scores_b: &ndcg_quality,
            },
            QualityMetricSamples {
                metric: QualityMetric::RecallAtK(10),
                scores_a: &recall_fast,
                scores_b: &recall_quality,
            },
        ];

        let report = quality_comparison(&samples, 0.95, 2_000, 42).unwrap();
        assert_eq!(report.query_count, 6);
        assert_eq!(report.metrics.len(), 2);
        assert_eq!(report.metrics[0].metric, QualityMetric::NdcgAtK(10));
        assert_eq!(report.metrics[1].metric, QualityMetric::RecallAtK(10));
        assert!(
            report
                .metrics
                .iter()
                .all(|row| row.comparison.mean_diff < 0.0)
        );
    }

    #[test]
    fn quality_comparison_rejects_empty_metrics() {
        assert!(quality_comparison(&[], 0.95, 1_000, 42).is_none());
    }

    #[test]
    fn quality_comparison_rejects_length_mismatch() {
        let samples = [QualityMetricSamples {
            metric: QualityMetric::Mrr,
            scores_a: &[0.5, 0.6, 0.7],
            scores_b: &[0.6, 0.7],
        }];
        assert!(quality_comparison(&samples, 0.95, 1_000, 42).is_none());
    }

    #[test]
    fn quality_comparison_deterministic_with_seed() {
        let map_fast = [0.30, 0.45, 0.40, 0.35, 0.50];
        let map_quality = [0.55, 0.62, 0.58, 0.57, 0.63];
        let mrr_fast = [0.35, 0.40, 0.38, 0.42, 0.39];
        let mrr_quality = [0.60, 0.65, 0.61, 0.66, 0.64];
        let samples = [
            QualityMetricSamples {
                metric: QualityMetric::MapAtK(10),
                scores_a: &map_fast,
                scores_b: &map_quality,
            },
            QualityMetricSamples {
                metric: QualityMetric::Mrr,
                scores_a: &mrr_fast,
                scores_b: &mrr_quality,
            },
        ];

        let one = quality_comparison(&samples, 0.95, 1_000, 123).unwrap();
        let two = quality_comparison(&samples, 0.95, 1_000, 123).unwrap();
        assert_eq!(one.query_count, two.query_count);
        assert!((one.confidence - two.confidence).abs() < f64::EPSILON);
        assert_eq!(one.n_resamples, two.n_resamples);
        assert_eq!(one.metrics.len(), two.metrics.len());
        for (a, b) in one.metrics.iter().zip(two.metrics.iter()) {
            assert_eq!(a.metric, b.metric);
            assert!((a.comparison.mean_a - b.comparison.mean_a).abs() < f64::EPSILON);
            assert!((a.comparison.mean_b - b.comparison.mean_b).abs() < f64::EPSILON);
            assert!((a.comparison.mean_diff - b.comparison.mean_diff).abs() < f64::EPSILON);
            assert!((a.comparison.ci_lower - b.comparison.ci_lower).abs() < f64::EPSILON);
            assert!((a.comparison.ci_upper - b.comparison.ci_upper).abs() < f64::EPSILON);
            assert!((a.comparison.p_value - b.comparison.p_value).abs() < f64::EPSILON);
            assert_eq!(a.comparison.significant, b.comparison.significant);
        }
    }

    #[test]
    fn quality_comparison_report_contains_metric_rows() {
        let samples = [QualityMetricSamples {
            metric: QualityMetric::Mrr,
            scores_a: &[0.5, 0.6, 0.7, 0.8],
            scores_b: &[0.4, 0.5, 0.6, 0.7],
        }];

        let report = quality_comparison(&samples, 0.95, 1_000, 99).unwrap();
        let rendered = report.render_tsv_report();
        assert!(rendered.contains("metric\tmean_a\tmean_b"));
        assert!(rendered.contains("MRR"));
    }

    #[test]
    fn quality_comparison_single_query_pair() {
        let samples = [QualityMetricSamples {
            metric: QualityMetric::NdcgAtK(10),
            scores_a: &[0.9],
            scores_b: &[0.3],
        }];
        let result = quality_comparison(&samples, 0.95, 500, 42);
        assert!(
            result.is_some(),
            "single-query comparison should produce a result"
        );
        let cmp = result.unwrap();
        assert_eq!(cmp.query_count, 1);
        assert_eq!(cmp.metrics.len(), 1);
        // With one pair, bootstrap resamples always pick the same pair,
        // so CI should be degenerate (lower == upper == mean_diff).
        let m = &cmp.metrics[0].comparison;
        assert!((m.ci_lower - m.ci_upper).abs() < f64::EPSILON);
    }

    #[test]
    fn quality_comparison_identical_systems_not_significant() {
        let scores = [0.5, 0.6, 0.7, 0.8, 0.9];
        let samples = [
            QualityMetricSamples {
                metric: QualityMetric::NdcgAtK(10),
                scores_a: &scores,
                scores_b: &scores,
            },
            QualityMetricSamples {
                metric: QualityMetric::Mrr,
                scores_a: &scores,
                scores_b: &scores,
            },
        ];
        let cmp = quality_comparison(&samples, 0.95, 1_000, 42).unwrap();
        for metric_cmp in &cmp.metrics {
            assert!(
                !metric_cmp.comparison.significant,
                "{} should not be significant for identical systems",
                metric_cmp.metric
            );
            assert!(
                metric_cmp.comparison.mean_diff.abs() < f64::EPSILON,
                "{} mean_diff should be 0 for identical systems, got {}",
                metric_cmp.metric,
                metric_cmp.comparison.mean_diff
            );
        }
    }

    #[test]
    fn quality_comparison_cross_metric_length_mismatch_rejected() {
        // First metric has 4 queries, second has 3 — should return None.
        let samples = [
            QualityMetricSamples {
                metric: QualityMetric::NdcgAtK(10),
                scores_a: &[0.5, 0.6, 0.7, 0.8],
                scores_b: &[0.4, 0.5, 0.6, 0.7],
            },
            QualityMetricSamples {
                metric: QualityMetric::Mrr,
                scores_a: &[0.5, 0.6, 0.7],
                scores_b: &[0.4, 0.5, 0.6],
            },
        ];
        assert!(
            quality_comparison(&samples, 0.95, 500, 42).is_none(),
            "cross-metric length mismatch should return None"
        );
    }

    #[test]
    fn bootstrap_compare_all_zero_scores() {
        let zeros = [0.0; 10];
        let cmp = bootstrap_compare(&zeros, &zeros, 0.95, 500, 42).unwrap();
        assert!(
            !cmp.significant,
            "all-zero scores should not be significant"
        );
        assert!(cmp.mean_diff.abs() < f64::EPSILON);
    }

    #[test]
    fn bootstrap_compare_all_one_scores() {
        let ones = [1.0; 10];
        let cmp = bootstrap_compare(&ones, &ones, 0.95, 500, 42).unwrap();
        assert!(!cmp.significant, "all-one scores should not be significant");
        assert!(cmp.mean_a >= 1.0 - f64::EPSILON);
    }

    #[test]
    fn quality_comparison_tsv_report_row_count_matches_metrics() {
        let samples = [
            QualityMetricSamples {
                metric: QualityMetric::NdcgAtK(5),
                scores_a: &[0.5, 0.6, 0.7],
                scores_b: &[0.4, 0.5, 0.6],
            },
            QualityMetricSamples {
                metric: QualityMetric::Mrr,
                scores_a: &[0.5, 0.6, 0.7],
                scores_b: &[0.4, 0.5, 0.6],
            },
            QualityMetricSamples {
                metric: QualityMetric::RecallAtK(10),
                scores_a: &[0.5, 0.6, 0.7],
                scores_b: &[0.4, 0.5, 0.6],
            },
        ];
        let cmp = quality_comparison(&samples, 0.95, 500, 42).unwrap();
        let tsv = cmp.render_tsv_report();
        let data_rows = tsv.lines().count() - 1; // subtract header
        assert_eq!(
            data_rows,
            cmp.metrics.len(),
            "TSV data rows should match metric count"
        );
    }

    // ─── Coefficient of Variation ───────────────────────────────────────

    #[test]
    fn cv_stable_samples() {
        let samples = vec![100.0, 101.0, 99.0, 100.5, 99.5];
        let cv = coefficient_of_variation(&samples).unwrap();
        assert!(cv < 0.01, "stable samples should have low CV, got {cv}");
    }

    #[test]
    fn cv_high_variance_samples() {
        let samples = vec![10.0, 100.0, 50.0, 200.0, 5.0];
        let cv = coefficient_of_variation(&samples).unwrap();
        assert!(
            cv > 0.5,
            "high-variance samples should have high CV, got {cv}"
        );
    }

    #[test]
    fn cv_returns_none_for_empty() {
        assert!(coefficient_of_variation(&[]).is_none());
    }

    #[test]
    fn cv_returns_none_for_zero_mean() {
        let samples = vec![-1.0, 1.0, -1.0, 1.0];
        assert!(
            coefficient_of_variation(&samples).is_none(),
            "CV undefined when mean is zero"
        );
    }

    #[test]
    fn cv_returns_none_for_non_finite_samples() {
        assert!(coefficient_of_variation(&[1.0, f64::NAN, 2.0]).is_none());
        assert!(coefficient_of_variation(&[1.0, f64::INFINITY, 2.0]).is_none());
    }

    #[test]
    fn cv_identical_values_is_zero() {
        let samples = vec![42.0; 10];
        let cv = coefficient_of_variation(&samples).unwrap();
        assert!(
            cv.abs() < 1e-10,
            "identical values should have CV=0, got {cv}"
        );
    }

    #[test]
    fn cv_single_sample() {
        let cv = coefficient_of_variation(&[5.0]).unwrap();
        assert!(cv.abs() < 1e-10, "single sample should have CV=0, got {cv}");
    }

    // ─── IQR Outlier Detection ──────────────────────────────────────────

    #[test]
    fn iqr_detects_clear_outlier() {
        let samples = vec![10.0, 11.0, 10.5, 10.2, 10.8, 100.0];
        let outliers = detect_outliers_iqr(&samples, 1.5);
        assert_eq!(
            outliers,
            vec![5],
            "index 5 (value 100.0) should be an outlier"
        );
    }

    #[test]
    fn iqr_no_outliers_in_tight_distribution() {
        let samples = vec![50.0, 51.0, 49.0, 50.5, 49.5, 50.2];
        let outliers = detect_outliers_iqr(&samples, 1.5);
        assert!(
            outliers.is_empty(),
            "tight distribution should have no outliers"
        );
    }

    #[test]
    fn iqr_returns_empty_for_small_samples() {
        assert!(detect_outliers_iqr(&[1.0, 2.0, 3.0], 1.5).is_empty());
        assert!(detect_outliers_iqr(&[1.0, 2.0], 1.5).is_empty());
        assert!(detect_outliers_iqr(&[1.0], 1.5).is_empty());
        assert!(detect_outliers_iqr(&[], 1.5).is_empty());
    }

    #[test]
    fn iqr_detects_both_low_and_high_outliers() {
        let samples = vec![1.0, 49.0, 50.0, 51.0, 50.5, 49.5, 200.0];
        let outliers = detect_outliers_iqr(&samples, 1.5);
        assert!(
            outliers.contains(&0),
            "index 0 (value 1.0) should be a low outlier"
        );
        assert!(
            outliers.contains(&6),
            "index 6 (value 200.0) should be a high outlier"
        );
    }

    #[test]
    fn iqr_stricter_factor_catches_more() {
        let samples = vec![10.0, 11.0, 10.5, 10.2, 10.8, 14.0, 7.0];
        let mild = detect_outliers_iqr(&samples, 1.5);
        let strict = detect_outliers_iqr(&samples, 0.5);
        assert!(
            strict.len() >= mild.len(),
            "stricter factor should catch at least as many outliers"
        );
    }

    #[test]
    fn iqr_invalid_factor_returns_empty() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(detect_outliers_iqr(&samples, f64::NAN).is_empty());
        assert!(detect_outliers_iqr(&samples, f64::INFINITY).is_empty());
        assert!(detect_outliers_iqr(&samples, -1.0).is_empty());
    }

    // ─── Trim Outliers ──────────────────────────────────────────────────

    #[test]
    fn trim_removes_outlier_values() {
        let samples = vec![10.0, 11.0, 10.5, 10.2, 10.8, 100.0];
        let trimmed = trim_outliers(&samples, 1.5);
        assert!(!trimmed.contains(&100.0), "outlier should be removed");
        assert_eq!(trimmed.len(), 5);
    }

    #[test]
    fn trim_preserves_order() {
        let samples = vec![10.0, 11.0, 10.5, 10.2, 10.8, 100.0];
        let trimmed = trim_outliers(&samples, 1.5);
        assert_eq!(trimmed, vec![10.0, 11.0, 10.5, 10.2, 10.8]);
    }

    #[test]
    fn trim_returns_all_when_no_outliers() {
        let samples = vec![50.0, 51.0, 49.0, 50.5];
        let trimmed = trim_outliers(&samples, 1.5);
        assert_eq!(trimmed, samples);
    }

    #[test]
    fn trim_returns_all_for_small_samples() {
        let samples = vec![1.0, 100.0, 50.0];
        let trimmed = trim_outliers(&samples, 1.5);
        assert_eq!(
            trimmed, samples,
            "fewer than 4 samples should not be trimmed"
        );
    }

    // ─── Run Stability Verdict ──────────────────────────────────────────

    #[test]
    fn stability_passes_for_tight_distribution() {
        // Values within a 1% range — no IQR outliers at k=1.5.
        let samples = vec![
            100.0, 100.3, 100.1, 100.4, 100.2, 100.5, 100.15, 100.35, 100.25, 100.45,
        ];
        let verdict = verify_run_stability(&samples, 0.05, 5);
        assert!(
            verdict.stable,
            "tight distribution should pass: {}",
            verdict.reason
        );
        assert_eq!(verdict.outlier_count, 0);
        assert_eq!(verdict.effective_sample_count, 10);
        assert!(verdict.cv.unwrap() < 0.05);
    }

    #[test]
    fn stability_fails_for_high_cv() {
        let samples = vec![10.0, 50.0, 100.0, 5.0, 200.0, 15.0, 80.0, 45.0];
        let verdict = verify_run_stability(&samples, 0.05, 3);
        assert!(!verdict.stable, "high-variance run should fail stability");
        assert!(verdict.reason.contains("coefficient of variation"));
    }

    #[test]
    fn stability_fails_for_too_few_samples_after_trimming() {
        // 10 inliers + 2 extreme outliers. IQR detects the 2 outliers,
        // leaving 10 effective samples. Require min_samples=11 to trigger fail.
        let samples = vec![
            10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 999.0, 1000.0,
        ];
        let verdict = verify_run_stability(&samples, 0.10, 11);
        assert!(
            !verdict.stable,
            "should fail when effective samples < min after trimming: {}",
            verdict.reason
        );
        assert!(verdict.reason.contains("insufficient samples"));
        assert_eq!(verdict.outlier_count, 2);
        assert_eq!(verdict.effective_sample_count, 10);
    }

    #[test]
    fn stability_fails_for_empty_samples() {
        let verdict = verify_run_stability(&[], 0.10, 5);
        assert!(!verdict.stable);
        assert!(verdict.reason.contains("no samples"));
    }

    #[test]
    fn stability_fails_for_non_finite_samples() {
        let verdict = verify_run_stability(&[10.0, f64::NAN, 10.2, 10.1], 0.10, 3);
        assert!(!verdict.stable);
        assert!(verdict.reason.contains("non-finite"));
    }

    #[test]
    fn stability_passes_after_outlier_removal() {
        let mut samples = vec![100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9];
        samples.push(999.0);
        let verdict = verify_run_stability(&samples, 0.05, 5);
        assert!(
            verdict.stable,
            "should pass after removing the outlier: {}",
            verdict.reason
        );
        assert_eq!(verdict.outlier_count, 1);
        assert_eq!(verdict.effective_sample_count, 9);
    }

    #[test]
    fn stability_cv_none_for_zero_mean_passes() {
        let samples = vec![0.0; 10];
        let verdict = verify_run_stability(&samples, 0.05, 5);
        assert!(verdict.stable, "zero-mean, zero-variance should pass");
        assert!(verdict.cv.is_none());
    }

    #[test]
    fn stability_verdict_reports_diagnostics() {
        let samples = vec![10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7];
        let verdict = verify_run_stability(&samples, 0.10, 5);
        assert!(verdict.stable);
        assert!(verdict.cv.is_some());
        assert_eq!(verdict.effective_sample_count, 8);
        assert_eq!(verdict.outlier_count, 0);
        assert!(
            verdict.reason.is_empty(),
            "stable verdict should have empty reason"
        );
    }
}
