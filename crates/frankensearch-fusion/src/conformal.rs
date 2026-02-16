//! Conformal prediction wrappers for distribution-free search coverage.
//!
//! This module implements lightweight conformal utilities around empirical
//! nonconformity scores (here: relevant-document ranks) so consumers can ask:
//! - "How many results do I need for coverage `1 - alpha`?" (`required_k`)
//! - "What rank interval is plausible at confidence `1 - alpha`?"
//! - "How surprising is an observed rank?" (`p_value`)
//!
//! The design keeps calibration storage and query-time operations simple:
//! - Calibration is `O(n log n)` once (sort scores).
//! - Query operations are `O(1)` plus integer indexing.

use std::collections::HashMap;

use frankensearch_core::query_class::QueryClass;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Deserializer, Serialize};

/// Empirical conformal calibration based on observed nonconformity ranks.
///
/// Ranks are 1-indexed (`1` = best/top hit). Higher ranks are worse.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ConformalSearchCalibration {
    nonconformity_scores: Vec<usize>,
    n_calibration: usize,
}

#[derive(Deserialize)]
struct ConformalSearchCalibrationWire {
    nonconformity_scores: Vec<usize>,
    n_calibration: usize,
}

impl<'de> Deserialize<'de> for ConformalSearchCalibration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ConformalSearchCalibrationWire::deserialize(deserializer)?;
        if wire.nonconformity_scores.is_empty() {
            return Err(serde::de::Error::custom(
                "conformal.nonconformity_scores must not be empty",
            ));
        }
        if wire.nonconformity_scores.contains(&0) {
            return Err(serde::de::Error::custom(
                "conformal.nonconformity_scores ranks must be >= 1",
            ));
        }
        if wire
            .nonconformity_scores
            .windows(2)
            .any(|pair| pair[0] > pair[1])
        {
            return Err(serde::de::Error::custom(
                "conformal.nonconformity_scores must be sorted ascending",
            ));
        }
        if wire.n_calibration != wire.nonconformity_scores.len() {
            return Err(serde::de::Error::custom(format!(
                "conformal.n_calibration ({}) must equal nonconformity_scores length ({})",
                wire.n_calibration,
                wire.nonconformity_scores.len()
            )));
        }

        Ok(Self {
            nonconformity_scores: wire.nonconformity_scores,
            n_calibration: wire.n_calibration,
        })
    }
}

impl ConformalSearchCalibration {
    /// Build a calibration object from observed relevant-document ranks.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when:
    /// - `nonconformity_scores` is empty
    /// - any rank is `0` (ranks must be 1-indexed)
    pub fn calibrate(nonconformity_scores: &[usize]) -> SearchResult<Self> {
        if nonconformity_scores.is_empty() {
            return Err(invalid_config(
                "conformal.nonconformity_scores",
                "[]",
                "calibration set must contain at least one rank",
            ));
        }
        if let Some(invalid) = nonconformity_scores.iter().copied().find(|&r| r == 0) {
            return Err(invalid_config(
                "conformal.nonconformity_scores",
                &invalid.to_string(),
                "ranks must be 1-indexed (minimum value is 1)",
            ));
        }

        let mut scores = nonconformity_scores.to_vec();
        scores.sort_unstable();
        let n_calibration = scores.len();
        Ok(Self {
            nonconformity_scores: scores,
            n_calibration,
        })
    }

    /// Number of calibration observations.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n_calibration
    }

    /// Always false once constructed (calibration requires non-empty input).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n_calibration == 0
    }

    /// Sorted nonconformity ranks used for calibration.
    #[must_use]
    pub fn nonconformity_scores(&self) -> &[usize] {
        &self.nonconformity_scores
    }

    /// Required `k` to achieve target coverage `1 - alpha`.
    ///
    /// Invalid alphas return the most conservative fallback (`max(rank)`).
    #[must_use]
    pub fn required_k(&self, alpha: f32) -> usize {
        self.required_k_checked(alpha).unwrap_or_else(|_| {
            // Fallback: last element (max rank), guarded against empty state
            // which can occur if the struct is constructed via Deserialize
            // bypassing the calibrate() constructor.
            self.nonconformity_scores.last().copied().unwrap_or(1)
        })
    }

    /// Checked variant of [`required_k`](Self::required_k).
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `alpha` is not finite or outside `[0, 1)`,
    /// or if the calibration data is empty (e.g. from invalid deserialization).
    pub fn required_k_checked(&self, alpha: f32) -> SearchResult<usize> {
        if self.nonconformity_scores.is_empty() {
            return Err(invalid_config(
                "conformal.nonconformity_scores",
                "[]",
                "calibration data is empty (possibly from invalid deserialization)",
            ));
        }
        let alpha = validate_alpha(alpha)?;
        let coverage = 1.0 - f64::from(alpha);
        let idx = quantile_index(self.n_calibration, coverage);
        Ok(self.nonconformity_scores[idx.min(self.nonconformity_scores.len() - 1)])
    }

    /// Two-sided rank prediction interval at confidence `1 - alpha`.
    ///
    /// Invalid alphas fall back to the full empirical support.
    #[must_use]
    pub fn rank_prediction_interval(&self, alpha: f32) -> (usize, usize) {
        self.rank_prediction_interval_checked(alpha)
            .unwrap_or_else(|_| {
                // Guarded against empty state from invalid deserialization.
                let lo = self.nonconformity_scores.first().copied().unwrap_or(1);
                let hi = self.nonconformity_scores.last().copied().unwrap_or(1);
                (lo, hi)
            })
    }

    /// Checked variant of [`rank_prediction_interval`](Self::rank_prediction_interval).
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `alpha` is not finite or outside `[0, 1)`,
    /// or if the calibration data is empty.
    pub fn rank_prediction_interval_checked(&self, alpha: f32) -> SearchResult<(usize, usize)> {
        if self.nonconformity_scores.is_empty() {
            return Err(invalid_config(
                "conformal.nonconformity_scores",
                "[]",
                "calibration data is empty (possibly from invalid deserialization)",
            ));
        }
        let alpha = validate_alpha(alpha)?;
        let tail = f64::from(alpha) / 2.0;
        let n = self.nonconformity_scores.len();
        let lower_idx = quantile_index(n, tail).min(n - 1);
        let upper_idx = quantile_index(n, 1.0 - tail).min(n - 1);
        Ok((
            self.nonconformity_scores[lower_idx],
            self.nonconformity_scores[upper_idx],
        ))
    }

    /// Conformal p-value for an observed rank.
    ///
    /// Invalid ranks (`0`) return `0.0`.
    #[must_use]
    pub fn p_value(&self, observed_rank: usize) -> f32 {
        self.p_value_checked(observed_rank).unwrap_or(0.0)
    }

    /// Checked variant of [`p_value`](Self::p_value).
    ///
    /// Uses the standard conformal finite-sample correction:
    /// `p = (#{score >= observed_rank} + 1) / (n + 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `observed_rank == 0`.
    #[allow(clippy::cast_precision_loss)]
    pub fn p_value_checked(&self, observed_rank: usize) -> SearchResult<f32> {
        if observed_rank == 0 {
            return Err(invalid_config(
                "conformal.observed_rank",
                "0",
                "rank must be 1-indexed (minimum value is 1)",
            ));
        }

        let first_geq = self
            .nonconformity_scores
            .partition_point(|&rank| rank < observed_rank);
        let exceed = self.n_calibration - first_geq;
        #[allow(clippy::cast_precision_loss)]
        let numerator = (exceed + 1) as f32;
        #[allow(clippy::cast_precision_loss)]
        let denominator = (self.n_calibration + 1) as f32;
        Ok(numerator / denominator)
    }
}

/// Query-class-conditional (Mondrian) conformal calibration.
///
/// When a query class has enough calibration examples, it gets an independent
/// conformal model; otherwise the global model is used as deterministic fallback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MondrianConformalCalibration {
    global: ConformalSearchCalibration,
    by_class: HashMap<QueryClass, ConformalSearchCalibration>,
    min_examples_per_class: usize,
}

impl MondrianConformalCalibration {
    /// Calibrate global + per-query-class conformal models from `(query, rank)` examples.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when:
    /// - `examples` is empty
    /// - `min_examples_per_class == 0`
    /// - any rank is `0`
    pub fn calibrate(
        examples: &[(String, usize)],
        min_examples_per_class: usize,
    ) -> SearchResult<Self> {
        if examples.is_empty() {
            return Err(invalid_config(
                "mondrian.examples",
                "[]",
                "calibration set must contain at least one query/rank pair",
            ));
        }
        if min_examples_per_class == 0 {
            return Err(invalid_config(
                "mondrian.min_examples_per_class",
                "0",
                "minimum per-class sample count must be >= 1",
            ));
        }

        let mut global_ranks = Vec::with_capacity(examples.len());
        let mut class_ranks: HashMap<QueryClass, Vec<usize>> = HashMap::new();

        for (query, rank) in examples {
            if *rank == 0 {
                return Err(invalid_config(
                    "mondrian.rank",
                    "0",
                    "ranks must be 1-indexed (minimum value is 1)",
                ));
            }

            let class = QueryClass::classify(query);
            class_ranks.entry(class).or_default().push(*rank);
            global_ranks.push(*rank);
        }

        let global = ConformalSearchCalibration::calibrate(&global_ranks)?;
        let mut by_class = HashMap::new();
        for (class, ranks) in class_ranks {
            if ranks.len() >= min_examples_per_class {
                by_class.insert(class, ConformalSearchCalibration::calibrate(&ranks)?);
            }
        }

        Ok(Self {
            global,
            by_class,
            min_examples_per_class,
        })
    }

    /// Required `k` for a query at target coverage `1 - alpha`.
    ///
    /// Uses class-specific calibration when available, otherwise global fallback.
    #[must_use]
    pub fn required_k(&self, query: &str, alpha: f32) -> usize {
        self.calibration_for_query(query).required_k(alpha)
    }

    /// Checked variant of [`required_k`](Self::required_k).
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `alpha` is invalid.
    pub fn required_k_checked(&self, query: &str, alpha: f32) -> SearchResult<usize> {
        self.calibration_for_query(query).required_k_checked(alpha)
    }

    /// Returns true when this query class has its own calibration model.
    #[must_use]
    pub fn has_class_calibration(&self, class: QueryClass) -> bool {
        self.by_class.contains_key(&class)
    }

    /// Global fallback calibration.
    #[must_use]
    pub const fn global(&self) -> &ConformalSearchCalibration {
        &self.global
    }

    /// Optional class-specific calibration.
    #[must_use]
    pub fn class_calibration(&self, class: QueryClass) -> Option<&ConformalSearchCalibration> {
        self.by_class.get(&class)
    }

    /// Minimum number of class examples required before enabling class-specific calibration.
    #[must_use]
    pub const fn min_examples_per_class(&self) -> usize {
        self.min_examples_per_class
    }

    fn calibration_for_query(&self, query: &str) -> &ConformalSearchCalibration {
        let class = QueryClass::classify(query);
        self.by_class.get(&class).unwrap_or(&self.global)
    }
}

/// Online adaptive alpha state for non-stationary distributions.
///
/// Update rule:
/// `alpha_t = clamp(alpha_{t-1} + gamma * (err_t - alpha_{t-1}), ε, 1-ε)`
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveConformalState {
    /// Current target miscoverage (`alpha`).
    pub alpha: f32,
    /// Adaptation rate in `(0, 1]`.
    pub gamma: f32,
}

/// Structured telemetry for one adaptive-alpha update.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveConformalUpdate {
    /// Alpha before update.
    pub alpha_before: f32,
    /// Alpha after update.
    pub alpha_after: f32,
    /// Observed error rate driving the update.
    pub observed_error_rate: f32,
    /// Updated `required_k` under `alpha_after`.
    pub required_k: usize,
}

impl AdaptiveConformalState {
    /// Create adaptive state with validated parameters.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `alpha` or `gamma` are invalid.
    pub fn new(alpha: f32, gamma: f32) -> SearchResult<Self> {
        let alpha = validate_alpha(alpha)?;
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) || gamma == 0.0 {
            return Err(invalid_config(
                "conformal.gamma",
                &gamma.to_string(),
                "gamma must be finite and in (0, 1]",
            ));
        }
        Ok(Self { alpha, gamma })
    }

    /// Update alpha from observed error and return structured telemetry.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for non-finite/out-of-range error rates.
    pub fn update(
        &mut self,
        observed_error_rate: f32,
        calibration: &ConformalSearchCalibration,
    ) -> SearchResult<AdaptiveConformalUpdate> {
        if !observed_error_rate.is_finite() || !(0.0..=1.0).contains(&observed_error_rate) {
            return Err(invalid_config(
                "conformal.observed_error_rate",
                &observed_error_rate.to_string(),
                "observed error rate must be finite and in [0, 1]",
            ));
        }
        // Guard against NaN alpha/gamma from deserialized state that bypassed
        // constructor validation. Without this, NaN propagates through mul_add
        // and clamp (which propagates NaN), permanently poisoning self.alpha.
        if !self.alpha.is_finite() || !self.gamma.is_finite() {
            return Err(invalid_config(
                "conformal.state",
                &format!("alpha={}, gamma={}", self.alpha, self.gamma),
                "adaptive state has non-finite alpha or gamma (possibly from invalid deserialization)",
            ));
        }

        let alpha_before = self.alpha;
        let drift = observed_error_rate - self.alpha;
        // Compute into local first so self.alpha is not corrupted if
        // required_k_checked fails (state-before-error preservation).
        let new_alpha = self
            .gamma
            .mul_add(drift, self.alpha)
            .clamp(1e-6, 1.0 - 1e-6);
        let required_k = calibration.required_k_checked(new_alpha)?;
        self.alpha = new_alpha;

        Ok(AdaptiveConformalUpdate {
            alpha_before,
            alpha_after: self.alpha,
            observed_error_rate,
            required_k,
        })
    }
}

fn invalid_config(field: &str, value: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

fn validate_alpha(alpha: f32) -> SearchResult<f32> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err(invalid_config(
            "conformal.alpha",
            &alpha.to_string(),
            "alpha must be finite and in [0, 1)",
        ));
    }
    Ok(alpha)
}

/// Quantile index using conformal finite-sample correction.
///
/// `quantile` is clamped to `[0, 1]`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn quantile_index(n: usize, quantile: f64) -> usize {
    // Guard: clamp() propagates NaN, and `as usize` on NaN is saturating-to-zero
    // but produces a meaningless index. Callers validate via validate_alpha(),
    // but defense-in-depth: treat non-finite as quantile=1.0 (conservative).
    let q = if quantile.is_finite() {
        quantile.clamp(0.0, 1.0)
    } else {
        1.0
    };
    #[allow(clippy::cast_precision_loss)]
    let adjusted = ((n as f64 + 1.0) * q).ceil();
    #[allow(clippy::cast_precision_loss)]
    let bounded = adjusted.max(1.0).min(n as f64);
    (bounded as usize).saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrate_rejects_empty_input() {
        let err = ConformalSearchCalibration::calibrate(&[]).expect_err("must reject empty");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn calibrate_rejects_zero_rank() {
        let err = ConformalSearchCalibration::calibrate(&[1, 0, 2]).expect_err("must reject zero");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn required_k_decreases_as_alpha_increases() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3, 5, 8, 13]).expect("calibrate");
        let k_tight = cal.required_k(0.01);
        let k_looser = cal.required_k(0.20);
        assert!(k_tight >= k_looser);
    }

    #[test]
    fn rank_prediction_interval_is_ordered() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3, 4, 7, 9]).expect("calibrate");
        let (lower, upper) = cal.rank_prediction_interval(0.1);
        assert!(lower <= upper);
        assert!(lower >= 1);
    }

    #[test]
    fn p_value_is_bounded_and_monotone() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3, 4, 6, 9]).expect("calibrate");
        let p_rank_2 = cal.p_value(2);
        let p_rank_8 = cal.p_value(8);
        assert!((0.0..=1.0).contains(&p_rank_2));
        assert!((0.0..=1.0).contains(&p_rank_8));
        assert!(p_rank_8 <= p_rank_2);
    }

    #[test]
    fn checked_methods_validate_inputs() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        assert!(cal.required_k_checked(1.0).is_err());
        assert!(cal.rank_prediction_interval_checked(f32::NAN).is_err());
        assert!(cal.p_value_checked(0).is_err());
    }

    #[test]
    fn adaptive_update_adjusts_alpha_and_returns_required_k() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 2, 3, 5, 8]).expect("calibrate");
        let mut state = AdaptiveConformalState::new(0.1, 0.2).expect("state");
        let update = state.update(0.3, &cal).expect("update");
        assert!(update.alpha_after > update.alpha_before);
        assert!(update.required_k >= 1);
    }

    #[test]
    fn serde_roundtrip_preserves_scores() {
        let cal = ConformalSearchCalibration::calibrate(&[3, 1, 2, 5]).expect("calibrate");
        let encoded = serde_json::to_string(&cal).expect("serialize");
        let decoded: ConformalSearchCalibration =
            serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(decoded, cal);
    }

    #[test]
    fn serde_rejects_mismatched_n_calibration() {
        let payload = r#"{"nonconformity_scores":[1,2,3],"n_calibration":2}"#;
        let err = serde_json::from_str::<ConformalSearchCalibration>(payload)
            .expect_err("mismatched n_calibration must fail");
        assert!(err.to_string().contains("n_calibration"));
    }

    #[test]
    fn serde_rejects_unsorted_nonconformity_scores() {
        let payload = r#"{"nonconformity_scores":[2,1,3],"n_calibration":3}"#;
        let err = serde_json::from_str::<ConformalSearchCalibration>(payload)
            .expect_err("unsorted scores must fail");
        assert!(err.to_string().contains("sorted"));
    }

    #[test]
    fn serde_rejects_zero_rank() {
        let payload = r#"{"nonconformity_scores":[0,1,2],"n_calibration":3}"#;
        let err = serde_json::from_str::<ConformalSearchCalibration>(payload)
            .expect_err("zero rank must fail");
        assert!(err.to_string().contains(">= 1"));
    }

    #[test]
    fn empirical_coverage_hits_target_within_sampling_tolerance() {
        let mut calibration = Vec::with_capacity(200);
        for _ in 0..10 {
            calibration.extend(1..=20);
        }
        let cal = ConformalSearchCalibration::calibrate(&calibration).expect("calibrate");
        let alpha = 0.10;
        let required_k = cal.required_k(alpha);

        let heldout: Vec<usize> = (0..120).map(|i| (i % 20) + 1).collect();
        let covered = heldout.iter().filter(|&&rank| rank <= required_k).count();
        #[allow(clippy::cast_precision_loss)]
        let empirical_coverage = covered as f32 / heldout.len() as f32;

        assert!(empirical_coverage >= (1.0 - alpha - 0.03));
    }

    #[test]
    fn mondrian_uses_class_specific_calibration_when_sufficient() {
        let examples = vec![
            ("vector search".to_owned(), 1),
            ("error handling".to_owned(), 2),
            ("index format".to_owned(), 2),
            ("ranking logic".to_owned(), 3),
            ("why does ranking drift under pressure".to_owned(), 9),
            (
                "how does calibration improve coverage guarantees".to_owned(),
                10,
            ),
            (
                "what causes retrieval quality degradation over time".to_owned(),
                11,
            ),
            (
                "when should quality phase be skipped by controller".to_owned(),
                12,
            ),
        ];
        let mondrian = MondrianConformalCalibration::calibrate(&examples, 3).expect("calibrate");
        assert!(mondrian.has_class_calibration(QueryClass::ShortKeyword));
        assert!(mondrian.has_class_calibration(QueryClass::NaturalLanguage));

        let short_k = mondrian.required_k("vector search", 0.10);
        let nl_k = mondrian.required_k("how does conformal coverage work in search", 0.10);
        assert!(nl_k > short_k);
    }

    #[test]
    fn mondrian_falls_back_to_global_for_sparse_classes() {
        let examples = vec![
            ("src/main.rs".to_owned(), 1),
            ("bd-123".to_owned(), 2),
            ("vector search".to_owned(), 4),
            ("error handling".to_owned(), 5),
            ("hybrid ranking".to_owned(), 6),
            ("fusion behavior".to_owned(), 7),
        ];
        let mondrian = MondrianConformalCalibration::calibrate(&examples, 3).expect("calibrate");

        assert!(!mondrian.has_class_calibration(QueryClass::Identifier));
        let global_k = mondrian.global().required_k(0.20);
        let identifier_k = mondrian.required_k("src/lib.rs", 0.20);
        assert_eq!(identifier_k, global_k);
    }

    #[test]
    fn mondrian_rejects_zero_min_examples() {
        let err = MondrianConformalCalibration::calibrate(&[("q".to_owned(), 1)], 0)
            .expect_err("must reject min_examples=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn adaptive_state_rejects_zero_gamma() {
        let err = AdaptiveConformalState::new(0.1, 0.0).expect_err("must reject gamma=0");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn adaptive_state_rejects_nan_gamma() {
        let err = AdaptiveConformalState::new(0.1, f32::NAN).expect_err("must reject NaN gamma");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn adaptive_state_rejects_negative_gamma() {
        let err = AdaptiveConformalState::new(0.1, -0.5).expect_err("must reject negative gamma");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn adaptive_update_rejects_nan_error_rate() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        let mut state = AdaptiveConformalState::new(0.1, 0.5).expect("state");
        let err = state
            .update(f32::NAN, &cal)
            .expect_err("must reject NaN error rate");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn adaptive_update_rejects_out_of_range_error_rate() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        let mut state = AdaptiveConformalState::new(0.1, 0.5).expect("state");
        let err = state
            .update(1.5, &cal)
            .expect_err("must reject >1 error rate");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn p_value_for_rank_beyond_all_calibration() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        let p = cal.p_value(100);
        // Rank 100 >> max calibration rank 3, so p-value should be very small.
        assert!((0.0..=1.0).contains(&p));
        assert!(p < 0.5);
    }

    #[test]
    fn single_element_calibration_works() {
        let cal = ConformalSearchCalibration::calibrate(&[5]).expect("calibrate");
        assert_eq!(cal.len(), 1);
        assert!(!cal.is_empty());
        assert_eq!(cal.required_k(0.1), 5);
        let (lo, hi) = cal.rank_prediction_interval(0.1);
        assert_eq!(lo, 5);
        assert_eq!(hi, 5);
    }

    #[test]
    fn mondrian_rejects_empty_examples() {
        let err = MondrianConformalCalibration::calibrate(&[], 3)
            .expect_err("must reject empty examples");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn mondrian_rejects_zero_rank_in_examples() {
        let err = MondrianConformalCalibration::calibrate(&[("query".to_owned(), 0)], 1)
            .expect_err("must reject zero rank");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn adaptive_update_decreases_alpha_when_error_below_target() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3, 5, 8]).expect("calibrate");
        let mut state = AdaptiveConformalState::new(0.3, 0.5).expect("state");
        let update = state.update(0.05, &cal).expect("update");
        assert!(update.alpha_after < update.alpha_before);
    }

    // ─── bd-1epq tests begin ───

    #[test]
    fn nonconformity_scores_are_sorted_after_calibration() {
        let cal = ConformalSearchCalibration::calibrate(&[9, 1, 5, 3, 7]).expect("calibrate");
        let scores = cal.nonconformity_scores();
        assert_eq!(scores, &[1, 3, 5, 7, 9]);
    }

    #[test]
    fn mondrian_class_calibration_accessor() {
        let examples: Vec<(String, usize)> = vec![
            ("vector search".into(), 1),
            ("error handling".into(), 2),
            ("index format".into(), 3),
            ("ranking logic".into(), 4),
        ];
        let mondrian = MondrianConformalCalibration::calibrate(&examples, 3).expect("calibrate");
        // ShortKeyword should have 4 examples >= min 3
        let class_cal = mondrian.class_calibration(QueryClass::ShortKeyword);
        assert!(class_cal.is_some());
        assert_eq!(class_cal.unwrap().len(), 4);
        // Identifier class has no examples
        assert!(mondrian.class_calibration(QueryClass::Identifier).is_none());
    }

    #[test]
    fn mondrian_min_examples_per_class_accessor() {
        let examples = vec![("test query".to_owned(), 5)];
        let mondrian = MondrianConformalCalibration::calibrate(&examples, 7).expect("calibrate");
        assert_eq!(mondrian.min_examples_per_class(), 7);
    }

    #[test]
    fn mondrian_required_k_checked_validates_alpha() {
        let examples = vec![
            ("search query".to_owned(), 1),
            ("another query".to_owned(), 3),
        ];
        let mondrian = MondrianConformalCalibration::calibrate(&examples, 1).expect("calibrate");
        // Valid alpha should work
        assert!(mondrian.required_k_checked("search query", 0.1).is_ok());
        // Invalid alpha (1.0) should fail
        assert!(mondrian.required_k_checked("search query", 1.0).is_err());
        // NaN alpha should fail
        assert!(mondrian.required_k_checked("test", f32::NAN).is_err());
    }

    #[test]
    fn adaptive_conformal_update_debug_format() {
        let update = AdaptiveConformalUpdate {
            alpha_before: 0.1,
            alpha_after: 0.15,
            observed_error_rate: 0.2,
            required_k: 5,
        };
        let debug = format!("{update:?}");
        assert!(debug.contains("0.1"));
        assert!(debug.contains("0.15"));
        assert!(debug.contains("0.2"));
        assert!(debug.contains('5'));
    }

    #[test]
    fn adaptive_conformal_state_serde_roundtrip() {
        let state = AdaptiveConformalState::new(0.15, 0.3).expect("state");
        let json = serde_json::to_string(&state).expect("serialize");
        let decoded: AdaptiveConformalState = serde_json::from_str(&json).expect("deserialize");
        assert!((decoded.alpha - 0.15).abs() < f32::EPSILON);
        assert!((decoded.gamma - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn p_value_for_best_rank_is_high() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3, 4, 5]).expect("calibrate");
        let p = cal.p_value(1);
        // All 5 scores >= 1, so p = (5+1)/(5+1) = 1.0
        assert!((p - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn required_k_with_alpha_zero_returns_max() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3, 10, 20]).expect("calibrate");
        let k = cal.required_k(0.0);
        // alpha=0 means coverage=1.0, should need max rank
        assert_eq!(k, 20);
    }

    #[test]
    fn adaptive_update_clamps_alpha_to_valid_range() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        // Start near 0, push error rate to 0 -> alpha decreases toward epsilon
        let mut state = AdaptiveConformalState::new(0.01, 1.0).expect("state");
        let update = state.update(0.0, &cal).expect("update");
        // Alpha should be clamped to >= 1e-6
        assert!(update.alpha_after >= 1e-6);
        assert!(update.alpha_after.is_finite());
    }

    #[test]
    fn rank_prediction_interval_checked_negative_alpha_error() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        let err = cal
            .rank_prediction_interval_checked(-0.1)
            .expect_err("must reject negative alpha");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    // ─── bd-1epq tests end ───

    // ─── MistyLark session 14: quantile_index NaN defense ───

    #[test]
    fn quantile_index_nan_falls_back_to_conservative() {
        // NaN quantile should produce a valid index, not garbage.
        // Defense-in-depth: treat NaN as 1.0 (most conservative = max index).
        let idx = quantile_index(10, f64::NAN);
        assert!(
            idx < 10,
            "NaN quantile should produce valid index, got {idx}"
        );
        // With q=1.0: adjusted = (10+1)*1.0 = 11.0, bounded = min(11,10) = 10, result = 9
        assert_eq!(idx, 9);
    }

    #[test]
    fn quantile_index_infinity_falls_back_to_conservative() {
        let idx = quantile_index(10, f64::INFINITY);
        assert_eq!(idx, 9);
        let idx_neg = quantile_index(10, f64::NEG_INFINITY);
        assert_eq!(idx_neg, 9);
    }

    // ─── bd-3gi9 tests begin ───

    #[test]
    fn adaptive_update_rejects_nan_alpha_from_deserialization() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        // Simulate deserialized state with NaN alpha (bypassing constructor).
        let mut state = AdaptiveConformalState {
            alpha: f32::NAN,
            gamma: 0.5,
        };
        let err = state.update(0.1, &cal).expect_err("must reject NaN alpha");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        // Alpha should still be NaN (not corrupted further — error before mutation).
        assert!(state.alpha.is_nan());
    }

    #[test]
    fn adaptive_update_rejects_nan_gamma_from_deserialization() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        // Simulate deserialized state with NaN gamma (bypassing constructor).
        let mut state = AdaptiveConformalState {
            alpha: 0.1,
            gamma: f32::NAN,
        };
        let err = state.update(0.1, &cal).expect_err("must reject NaN gamma");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
        // Alpha should NOT have been mutated (error before computation).
        assert!((state.alpha - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn adaptive_update_preserves_state_on_error() {
        let cal = ConformalSearchCalibration::calibrate(&[1, 2, 3]).expect("calibrate");
        let mut state = AdaptiveConformalState::new(0.1, 0.5).expect("state");
        let alpha_before = state.alpha;
        // Invalid error rate should fail without mutating alpha.
        let _ = state.update(f32::NAN, &cal);
        assert!(
            (state.alpha - alpha_before).abs() < f32::EPSILON,
            "alpha should not change on error"
        );
    }

    // ─── bd-3gi9 tests end ───
}
