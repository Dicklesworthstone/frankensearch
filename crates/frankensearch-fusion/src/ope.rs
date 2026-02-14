//! Off-policy evaluation (OPE) for safe ranking changes.
//!
//! Estimates the impact of ranking algorithm changes using historical search
//! logs **before** deploying online. Implements Inverse Propensity Scoring
//! (IPS) and Doubly Robust (DR) estimators.
//!
//! This is an offline analysis tool with zero runtime cost to search.
//!
//! # References
//!
//! - Dudik et al. (2011) "Doubly Robust Policy Evaluation"
//! - Swaminathan & Joachims (2015) "Batch Learning from Logged Bandit Feedback"
//!
//! # Example
//!
//! ```
//! use frankensearch_fusion::ope::*;
//!
//! let observations = vec![
//!     LoggedObservation {
//!         query: "rust async".into(),
//!         doc_id: "doc-1".into(),
//!         rank: 0,
//!         reward: 1.0,
//!         logging_propensity: 0.5,
//!     },
//!     LoggedObservation {
//!         query: "rust async".into(),
//!         doc_id: "doc-2".into(),
//!         rank: 1,
//!         reward: 0.0,
//!         logging_propensity: 0.3,
//!     },
//! ];
//!
//! let target_propensities = vec![0.6, 0.4];
//!
//! let config = OpeConfig::default();
//! let result = ips_estimate(&observations, &target_propensities, &config);
//! assert!(result.effective_sample_size > 0.0);
//! ```

use serde::{Deserialize, Serialize};

/// A single logged observation from the search pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggedObservation {
    /// The search query.
    pub query: String,
    /// The document that was displayed/clicked.
    pub doc_id: String,
    /// The rank at which this document was displayed (0-indexed).
    pub rank: usize,
    /// The observed reward (1.0 = click/relevant, 0.0 = skip/irrelevant).
    pub reward: f64,
    /// The probability that the logging policy would show this doc at this rank.
    pub logging_propensity: f64,
}

/// Configuration for off-policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeConfig {
    /// Maximum importance weight before clipping. Default: 100.0.
    /// Clipping reduces variance at the cost of introducing slight bias.
    pub clipping_threshold: f64,

    /// Minimum effective sample size to trust the estimate. Default: 10.0.
    pub min_effective_sample_size: f64,
}

impl Default for OpeConfig {
    fn default() -> Self {
        Self {
            clipping_threshold: 100.0,
            min_effective_sample_size: 10.0,
        }
    }
}

/// Result of an off-policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpeResult {
    /// Estimated expected reward under the target policy.
    pub estimated_reward: f64,

    /// Effective sample size (ESS). Higher = more reliable estimate.
    /// `ESS = (sum w_i)^2 / sum(w_i^2)`
    pub effective_sample_size: f64,

    /// Number of observations used.
    pub n_observations: usize,

    /// Whether the estimate is considered reliable (ESS >= min threshold).
    pub reliable: bool,

    /// 95% confidence interval half-width (approximate, using CLT).
    pub confidence_interval_95: f64,
}

/// Inverse Propensity Scoring (IPS) estimator.
///
/// Estimates the expected reward under a target policy using importance
/// weighting. Each observation is re-weighted by the ratio of target
/// propensity to logging propensity.
///
/// `estimated_reward = (1/N) * sum(reward_i * w_i)`
/// where `w_i = target_propensity_i / logging_propensity_i` (clipped).
///
/// # Arguments
///
/// * `observations` - Historical search log entries.
/// * `target_propensities` - Propensity of the target policy for each observation.
///   Must have the same length as `observations`.
/// * `config` - Evaluation configuration.
///
/// # Panics
///
/// Panics if `observations` and `target_propensities` have different lengths.
#[must_use]
pub fn ips_estimate(
    observations: &[LoggedObservation],
    target_propensities: &[f64],
    config: &OpeConfig,
) -> OpeResult {
    assert_eq!(
        observations.len(),
        target_propensities.len(),
        "observations and target propensities must have the same length"
    );

    let n = observations.len();
    if n == 0 {
        return OpeResult {
            estimated_reward: 0.0,
            effective_sample_size: 0.0,
            n_observations: 0,
            reliable: false,
            confidence_interval_95: f64::INFINITY,
        };
    }

    // Compute clipped importance weights.
    let weights: Vec<f64> = observations
        .iter()
        .zip(target_propensities.iter())
        .map(|(obs, &target_p)| {
            let logging_p = obs.logging_propensity.max(1e-10); // avoid division by zero
            let w = target_p / logging_p;
            w.clamp(0.0, config.clipping_threshold)
        })
        .collect();

    // IPS estimate.
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    let weighted_sum: f64 = observations
        .iter()
        .zip(weights.iter())
        .map(|(obs, &w)| obs.reward * w)
        .sum();
    let estimated_reward = weighted_sum / n_f;

    // Effective sample size.
    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let ess = if sum_w2 > f64::EPSILON {
        (sum_w * sum_w) / sum_w2
    } else {
        0.0
    };

    // Approximate 95% CI using CLT.
    let mean = estimated_reward;
    let variance: f64 = observations
        .iter()
        .zip(weights.iter())
        .map(|(obs, &w)| {
            let diff = obs.reward.mul_add(w, -mean);
            diff * diff
        })
        .sum::<f64>()
        / n_f;
    let std_err = (variance / n_f).sqrt();
    let ci_95 = 1.96 * std_err;

    let reliable = ess >= config.min_effective_sample_size;

    OpeResult {
        estimated_reward,
        effective_sample_size: ess,
        n_observations: n,
        reliable,
        confidence_interval_95: ci_95,
    }
}

/// Doubly Robust (DR) estimator.
///
/// Combines IPS with a reward model to reduce variance. The DR estimator
/// has variance <= IPS variance (provable).
///
/// `DR = (1/N) * sum( reward_model(x_i) + w_i * (reward_i - reward_model(x_i)) )`
///
/// # Arguments
///
/// * `observations` - Historical search log entries.
/// * `target_propensities` - Propensity of the target policy for each observation.
/// * `reward_predictions` - Reward model predictions for each observation (e.g., NDCG@10).
/// * `config` - Evaluation configuration.
///
/// # Panics
///
/// Panics if `observations`, `target_propensities`, and `reward_predictions`
/// have different lengths.
#[must_use]
pub fn dr_estimate(
    observations: &[LoggedObservation],
    target_propensities: &[f64],
    reward_predictions: &[f64],
    config: &OpeConfig,
) -> OpeResult {
    assert_eq!(observations.len(), target_propensities.len());
    assert_eq!(observations.len(), reward_predictions.len());

    let n = observations.len();
    if n == 0 {
        return OpeResult {
            estimated_reward: 0.0,
            effective_sample_size: 0.0,
            n_observations: 0,
            reliable: false,
            confidence_interval_95: f64::INFINITY,
        };
    }

    // Compute clipped importance weights.
    let weights: Vec<f64> = observations
        .iter()
        .zip(target_propensities.iter())
        .map(|(obs, &target_p)| {
            let logging_p = obs.logging_propensity.max(1e-10);
            let w = target_p / logging_p;
            w.clamp(0.0, config.clipping_threshold)
        })
        .collect();

    // DR estimate.
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;
    let dr_sum: f64 = observations
        .iter()
        .zip(weights.iter())
        .zip(reward_predictions.iter())
        .map(|((obs, &w), &pred)| {
            // control variate: reward_model + importance_weighted_residual
            w.mul_add(obs.reward - pred, pred)
        })
        .sum();
    let estimated_reward = dr_sum / n_f;

    // ESS (same as IPS).
    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let ess = if sum_w2 > f64::EPSILON {
        (sum_w * sum_w) / sum_w2
    } else {
        0.0
    };

    // Approximate 95% CI.
    let mean = estimated_reward;
    let variance: f64 = observations
        .iter()
        .zip(weights.iter())
        .zip(reward_predictions.iter())
        .map(|((obs, &w), &pred)| {
            let term = w.mul_add(obs.reward - pred, pred);
            let diff = term - mean;
            diff * diff
        })
        .sum::<f64>()
        / n_f;
    let std_err = (variance / n_f).sqrt();
    let ci_95 = 1.96 * std_err;

    let reliable = ess >= config.min_effective_sample_size;

    OpeResult {
        estimated_reward,
        effective_sample_size: ess,
        n_observations: n,
        reliable,
        confidence_interval_95: ci_95,
    }
}

/// Compute effective sample size from importance weights.
///
/// `ESS = (sum w_i)^2 / sum(w_i^2)`
///
/// Higher ESS indicates better overlap between logging and target policies.
#[must_use]
pub fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    if sum_w2 > f64::EPSILON {
        (sum_w * sum_w) / sum_w2
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_observations(n: usize) -> Vec<LoggedObservation> {
        (0..n)
            .map(|i| LoggedObservation {
                query: format!("query-{i}"),
                doc_id: format!("doc-{i}"),
                rank: i,
                reward: if i % 2 == 0 { 1.0 } else { 0.0 },
                logging_propensity: 0.5,
            })
            .collect()
    }

    // ── IPS ──────────────────────────────────────────────────────────────

    #[test]
    fn ips_uniform_weights() {
        let obs = make_observations(10);
        let target_p = vec![0.5; 10]; // same as logging -> weights all 1.0
        let config = OpeConfig::default();

        let result = ips_estimate(&obs, &target_p, &config);
        // All weights = 1.0, reward = alternating 1,0
        // Expected: 0.5
        assert!((result.estimated_reward - 0.5).abs() < 1e-10);
        assert_eq!(result.n_observations, 10);
        assert!((result.effective_sample_size - 10.0).abs() < 1e-10);
        assert!(result.reliable);
    }

    #[test]
    fn ips_double_weights() {
        let obs = make_observations(4);
        // Target propensity = 2x logging -> weights = 2.0
        let target_p = vec![1.0; 4];
        let config = OpeConfig::default();

        let result = ips_estimate(&obs, &target_p, &config);
        // rewards: [1, 0, 1, 0], weights: [2, 2, 2, 2]
        // IPS = (1*2 + 0*2 + 1*2 + 0*2) / 4 = 1.0
        assert!((result.estimated_reward - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ips_clipping() {
        let obs = vec![LoggedObservation {
            query: "q".into(),
            doc_id: "d".into(),
            rank: 0,
            reward: 1.0,
            logging_propensity: 0.001, // very low -> huge weight
        }];
        let target_p = vec![1.0]; // weight = 1000 before clipping
        let config = OpeConfig {
            clipping_threshold: 10.0,
            ..Default::default()
        };

        let result = ips_estimate(&obs, &target_p, &config);
        // Weight clipped to 10.0
        assert!((result.estimated_reward - 10.0).abs() < 1e-10);
    }

    #[test]
    fn ips_empty() {
        let config = OpeConfig::default();
        let result = ips_estimate(&[], &[], &config);
        assert!((result.estimated_reward).abs() < f64::EPSILON);
        assert!(!result.reliable);
        assert_eq!(result.n_observations, 0);
    }

    // ── DR ────────────────────────────────────────────────────────────────

    #[test]
    fn dr_with_perfect_reward_model() {
        let obs = make_observations(10);
        let target_p = vec![0.5; 10];
        // Perfect reward model predicts exact rewards.
        let predictions: Vec<f64> = obs.iter().map(|o| o.reward).collect();
        let config = OpeConfig::default();

        let result = dr_estimate(&obs, &target_p, &predictions, &config);
        // DR with perfect model and weight=1 should equal mean reward.
        assert!((result.estimated_reward - 0.5).abs() < 1e-10);
    }

    #[test]
    fn dr_produces_valid_estimate() {
        // DR should produce a valid estimate with correlated reward predictions.
        let obs: Vec<LoggedObservation> = (0..100)
            .map(|i| LoggedObservation {
                query: format!("q-{i}"),
                doc_id: format!("d-{i}"),
                rank: i,
                reward: if i % 3 == 0 { 1.0 } else { 0.0 },
                #[allow(clippy::cast_precision_loss)]
                logging_propensity: (i as f64 % 5.0).mul_add(0.1, 0.3),
            })
            .collect();
        let target_p: Vec<f64> = (0..100)
            .map(|i| (f64::from(i) % 3.0).mul_add(0.1, 0.4))
            .collect();
        let predictions: Vec<f64> = obs
            .iter()
            .map(|o| if o.reward > 0.5 { 0.8 } else { 0.2 })
            .collect();
        let config = OpeConfig::default();

        let ips = ips_estimate(&obs, &target_p, &config);
        let dr = dr_estimate(&obs, &target_p, &predictions, &config);

        // Both should be reliable with 100 observations.
        assert!(ips.reliable);
        assert!(dr.reliable);

        // Both should produce finite estimates.
        assert!(dr.estimated_reward.is_finite());
        assert!(dr.confidence_interval_95.is_finite());
        assert!(dr.confidence_interval_95 >= 0.0);
    }

    #[test]
    fn dr_empty() {
        let config = OpeConfig::default();
        let result = dr_estimate(&[], &[], &[], &config);
        assert!(!result.reliable);
    }

    // ── ESS ──────────────────────────────────────────────────────────────

    #[test]
    fn ess_uniform_weights() {
        let weights = vec![1.0; 100];
        assert!((effective_sample_size(&weights) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn ess_single_dominant_weight() {
        let mut weights = vec![0.01; 100];
        weights[0] = 100.0;
        let ess = effective_sample_size(&weights);
        // ESS should be close to 1 (dominated by one observation).
        assert!(ess < 2.0);
    }

    #[test]
    fn ess_empty() {
        assert!(effective_sample_size(&[]).abs() < f64::EPSILON);
    }

    // ── Reliability ──────────────────────────────────────────────────────

    #[test]
    fn low_ess_not_reliable() {
        let obs = make_observations(3);
        let target_p = vec![0.5; 3];
        let config = OpeConfig {
            min_effective_sample_size: 10.0,
            ..Default::default()
        };

        let result = ips_estimate(&obs, &target_p, &config);
        // ESS = 3 (uniform weights) < 10 -> not reliable
        assert!(!result.reliable);
    }

    // ── Serde ────────────────────────────────────────────────────────────

    #[test]
    fn observation_serde_roundtrip() {
        let obs = LoggedObservation {
            query: "test".into(),
            doc_id: "doc-1".into(),
            rank: 3,
            reward: 0.8,
            logging_propensity: 0.4,
        };
        let json = serde_json::to_string(&obs).unwrap();
        let decoded: LoggedObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.query, "test");
        assert!((decoded.reward - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = OpeConfig {
            clipping_threshold: 50.0,
            min_effective_sample_size: 20.0,
        };
        let json = serde_json::to_string(&config).unwrap();
        let decoded: OpeConfig = serde_json::from_str(&json).unwrap();
        assert!((decoded.clipping_threshold - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn result_serde_roundtrip() {
        let result = OpeResult {
            estimated_reward: 0.42,
            effective_sample_size: 85.0,
            n_observations: 200,
            reliable: true,
            confidence_interval_95: 0.05,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: OpeResult = serde_json::from_str(&json).unwrap();
        assert!((decoded.estimated_reward - 0.42).abs() < f64::EPSILON);
        assert!(decoded.reliable);
    }

    #[test]
    fn ips_single_observation_reward_one() {
        let obs = vec![LoggedObservation {
            query: "q".into(),
            doc_id: "d".into(),
            rank: 0,
            reward: 1.0,
            logging_propensity: 0.5,
        }];
        let target_p = vec![0.5];
        let config = OpeConfig::default();
        let result = ips_estimate(&obs, &target_p, &config);
        assert!((result.estimated_reward - 1.0).abs() < 1e-10);
        assert_eq!(result.n_observations, 1);
    }

    #[test]
    fn ips_all_zero_rewards() {
        let obs: Vec<LoggedObservation> = (0..10)
            .map(|i| LoggedObservation {
                query: format!("q-{i}"),
                doc_id: format!("d-{i}"),
                rank: i,
                reward: 0.0,
                logging_propensity: 0.5,
            })
            .collect();
        let target_p = vec![0.5; 10];
        let config = OpeConfig::default();
        let result = ips_estimate(&obs, &target_p, &config);
        assert!(result.estimated_reward.abs() < 1e-10);
    }

    #[test]
    fn ips_all_one_rewards() {
        let obs: Vec<LoggedObservation> = (0..10)
            .map(|i| LoggedObservation {
                query: format!("q-{i}"),
                doc_id: format!("d-{i}"),
                rank: i,
                reward: 1.0,
                logging_propensity: 0.5,
            })
            .collect();
        let target_p = vec![0.5; 10];
        let config = OpeConfig::default();
        let result = ips_estimate(&obs, &target_p, &config);
        assert!((result.estimated_reward - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ess_all_zero_weights() {
        let weights = vec![0.0; 10];
        let ess = effective_sample_size(&weights);
        assert!(ess.abs() < f64::EPSILON);
    }

    #[test]
    fn dr_zero_predictions_degrades_to_ips_like() {
        let obs = make_observations(20);
        let target_p = vec![0.5; 20];
        let predictions = vec![0.0; 20];
        let config = OpeConfig::default();

        let ips = ips_estimate(&obs, &target_p, &config);
        let dr = dr_estimate(&obs, &target_p, &predictions, &config);

        // With zero predictions and weight=1, DR = (1/N) * sum(0 + 1*(reward - 0)) = IPS.
        assert!((dr.estimated_reward - ips.estimated_reward).abs() < 1e-10);
    }

    #[test]
    fn ips_ci_is_non_negative() {
        let obs = make_observations(50);
        let target_p = vec![0.5; 50];
        let config = OpeConfig::default();
        let result = ips_estimate(&obs, &target_p, &config);
        assert!(result.confidence_interval_95 >= 0.0);
        assert!(result.confidence_interval_95.is_finite());
    }
}
