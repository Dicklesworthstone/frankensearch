//! Adaptive fusion parameters via Bayesian online learning.
//!
//! Maintains conjugate priors for the RRF K constant and blend factor that
//! update from implicit relevance feedback, improving hybrid ranking over
//! time while remaining auditable and safe.
//!
//! # Mathematical Foundation
//!
//! - **Blend factor**: Beta-Bernoulli model. Prior `Beta(7, 3)` encodes
//!   the initial 0.7 blend factor. Each observation of whether quality-tier
//!   reranking improved results updates the posterior.
//! - **RRF K**: Normal-Normal model. Prior `N(60, 10²)` encodes the default
//!   K=60 with uncertainty. Observations of optimal K from NDCG evaluation
//!   update the posterior via conjugate update rules.
//!
//! Both models support per-[`QueryClass`] posteriors with a global fallback.
//!
//! [`QueryClass`]: frankensearch_core::QueryClass

use std::collections::HashMap;
use std::sync::Mutex;

use frankensearch_core::QueryClass;
use serde::{Deserialize, Serialize};
use tracing::debug;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for adaptive fusion parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Minimum observations before posterior values are used (default: 50).
    pub min_samples: u64,
    /// Lower safety clamp for blend factor (default: 0.1).
    pub blend_min: f64,
    /// Upper safety clamp for blend factor (default: 0.95).
    pub blend_max: f64,
    /// Lower safety clamp for RRF K (default: 1.0).
    pub k_min: f64,
    /// Upper safety clamp for RRF K (default: 200.0).
    pub k_max: f64,
    /// Enable Thompson sampling for exploration (default: false).
    pub thompson_sampling: bool,
    /// Deterministic seed for Thompson sampling (default: `None` → random).
    pub seed: Option<u64>,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_samples: 50,
            blend_min: 0.1,
            blend_max: 0.95,
            k_min: 1.0,
            k_max: 200.0,
            thompson_sampling: false,
            seed: None,
        }
    }
}

// ─── Beta-Bernoulli Posterior (Blend Factor) ─────────────────────────────────

/// Beta-Bernoulli posterior for the blend factor.
///
/// The expected blend factor is `alpha / (alpha + beta)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendPosterior {
    /// Beta distribution alpha parameter (successes + prior).
    pub alpha: f64,
    /// Beta distribution beta parameter (failures + prior).
    pub beta: f64,
    /// Number of observations.
    pub n: u64,
}

impl Default for BlendPosterior {
    fn default() -> Self {
        // Prior Beta(7, 3) → E[blend] = 0.7.
        Self {
            alpha: 7.0,
            beta: 3.0,
            n: 0,
        }
    }
}

impl BlendPosterior {
    /// Update the posterior with a Bernoulli observation.
    ///
    /// `success = true` means quality-tier reranking improved results.
    pub fn update(&mut self, success: bool) {
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
        self.n += 1;
    }

    /// Posterior mean (expected blend factor).
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }
}

// ─── Normal-Normal Posterior (RRF K) ─────────────────────────────────────────

/// Normal-Normal posterior for the RRF K constant.
///
/// Uses conjugate update: prior `N(mu, sigma²)` with known observation
/// noise `sigma_obs²` yields a Normal posterior after each observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPosterior {
    /// Posterior mean.
    pub mu: f64,
    /// Posterior variance.
    pub sigma_sq: f64,
    /// Observation noise variance (fixed).
    pub sigma_obs_sq: f64,
    /// Number of observations.
    pub n: u64,
}

impl Default for KPosterior {
    fn default() -> Self {
        // Prior N(60, 100) → E[K] = 60, std = 10.
        Self {
            mu: 60.0,
            sigma_sq: 100.0,
            sigma_obs_sq: 225.0, // sigma_obs = 15
            n: 0,
        }
    }
}

impl KPosterior {
    /// Update the posterior with an observation of optimal K.
    pub fn update(&mut self, observed_k: f64) {
        let precision_prior = 1.0 / self.sigma_sq;
        let precision_obs = 1.0 / self.sigma_obs_sq;
        let precision_post = precision_prior + precision_obs;
        self.mu = precision_prior.mul_add(self.mu, precision_obs * observed_k) / precision_post;
        self.sigma_sq = 1.0 / precision_post;
        self.n += 1;
    }

    /// Posterior mean (expected K).
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// Posterior standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.sigma_sq.sqrt()
    }
}

// ─── Evidence Record ─────────────────────────────────────────────────────────

/// Structured evidence emitted per adaptive update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceEvent {
    /// Query class that produced this evidence.
    pub query_class: QueryClass,
    /// Blend factor used for this query.
    pub blend_used: f64,
    /// RRF K used for this query.
    pub k_used: f64,
    /// Blend posterior after update (alpha, beta).
    pub blend_posterior: (f64, f64),
    /// K posterior after update (mu, sigma_sq).
    pub k_posterior: (f64, f64),
    /// Source of the relevance signal.
    pub signal_source: SignalSource,
}

/// Source of the implicit relevance feedback signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalSource {
    /// User selected a result (click).
    Click,
    /// User skipped this result.
    Skip,
    /// User dwelled on the result for a notable duration.
    Dwell,
    /// Automated NDCG evaluation against ground truth.
    NdcgEval,
}

// ─── Adaptive Fusion State ──────────────────────────────────────────────────

/// Per-query-class adaptive fusion state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ClassState {
    blend: BlendPosterior,
    k: KPosterior,
}

/// Adaptive fusion parameters with Bayesian online learning.
///
/// Thread-safe via internal `Mutex`. All operations are O(1).
pub struct AdaptiveFusion {
    config: AdaptiveConfig,
    state: Mutex<AdaptiveState>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AdaptiveState {
    /// Per-query-class posteriors.
    per_class: HashMap<QueryClass, ClassState>,
    /// Global fallback posteriors (used when per-class has insufficient data).
    global: ClassState,
}

impl std::fmt::Debug for AdaptiveFusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveFusion")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl AdaptiveFusion {
    /// Create adaptive fusion with the given configuration.
    #[must_use]
    pub fn new(config: AdaptiveConfig) -> Self {
        Self {
            config,
            state: Mutex::new(AdaptiveState::default()),
        }
    }

    /// Create adaptive fusion with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(AdaptiveConfig::default())
    }

    /// Get the current blend factor for a query class.
    ///
    /// Returns the posterior mean, clamped to `[blend_min, blend_max]`.
    /// Falls back to the global posterior when per-class data is insufficient.
    #[must_use]
    pub fn blend_factor(&self, query_class: QueryClass) -> f64 {
        let state = self.state.lock().expect("adaptive lock poisoned");
        let blend = self.resolve_blend(&state, query_class);
        blend.clamp(self.config.blend_min, self.config.blend_max)
    }

    /// Get the current RRF K for a query class.
    ///
    /// Returns the posterior mean, clamped to `[k_min, k_max]`.
    /// Falls back to the global posterior when per-class data is insufficient.
    #[must_use]
    pub fn rrf_k(&self, query_class: QueryClass) -> f64 {
        let state = self.state.lock().expect("adaptive lock poisoned");
        let k = self.resolve_k(&state, query_class);
        k.clamp(self.config.k_min, self.config.k_max)
    }

    /// Update the blend posterior with a Bernoulli observation.
    ///
    /// `success = true` means quality-tier reranking improved the ranking.
    pub fn update_blend(
        &self,
        query_class: QueryClass,
        success: bool,
        signal: SignalSource,
    ) -> EvidenceEvent {
        let mut state = self.state.lock().expect("adaptive lock poisoned");

        // Update per-class posterior.
        let class_state = state.per_class.entry(query_class).or_default();
        class_state.blend.update(success);

        // Update global posterior.
        state.global.blend.update(success);

        let event = EvidenceEvent {
            query_class,
            blend_used: class_state.blend.mean(),
            k_used: class_state.k.mean(),
            blend_posterior: (class_state.blend.alpha, class_state.blend.beta),
            k_posterior: (class_state.k.mu, class_state.k.sigma_sq),
            signal_source: signal,
        };

        debug!(
            query_class = ?query_class,
            blend_alpha = class_state.blend.alpha,
            blend_beta = class_state.blend.beta,
            blend_mean = class_state.blend.mean(),
            signal = ?signal,
            "blend posterior updated"
        );

        event
    }

    /// Update the RRF K posterior with an observed optimal K value.
    pub fn update_k(
        &self,
        query_class: QueryClass,
        observed_k: f64,
        signal: SignalSource,
    ) -> EvidenceEvent {
        let mut state = self.state.lock().expect("adaptive lock poisoned");

        // Update per-class posterior.
        let class_state = state.per_class.entry(query_class).or_default();
        class_state.k.update(observed_k);

        // Update global posterior.
        state.global.k.update(observed_k);

        let event = EvidenceEvent {
            query_class,
            blend_used: class_state.blend.mean(),
            k_used: class_state.k.mean(),
            blend_posterior: (class_state.blend.alpha, class_state.blend.beta),
            k_posterior: (class_state.k.mu, class_state.k.sigma_sq),
            signal_source: signal,
        };

        debug!(
            query_class = ?query_class,
            k_mu = class_state.k.mu,
            k_sigma = class_state.k.std_dev(),
            observed_k,
            signal = ?signal,
            "K posterior updated"
        );

        event
    }

    /// Reset all learned parameters to their priors.
    ///
    /// Clears all per-class and global observations, returning the fusion
    /// state to its initial configuration.
    pub fn reset(&self) {
        let mut state = self.state.lock().expect("adaptive lock poisoned");
        state.per_class.clear();
        state.global = ClassState::default();
        debug!("adaptive fusion state reset to prior");
    }

    /// Snapshot the current state for serialization or inspection.
    #[must_use]
    pub fn snapshot(&self) -> AdaptiveSnapshot {
        let state = self.state.lock().expect("adaptive lock poisoned");
        AdaptiveSnapshot {
            global_blend: state.global.blend.clone(),
            global_k: state.global.k.clone(),
            per_class: state
                .per_class
                .iter()
                .map(|(&qc, cs)| (qc, cs.blend.clone(), cs.k.clone()))
                .collect(),
        }
    }

    fn resolve_blend(&self, state: &AdaptiveState, query_class: QueryClass) -> f64 {
        if let Some(cs) = state.per_class.get(&query_class) {
            if cs.blend.n >= self.config.min_samples {
                return cs.blend.mean();
            }
        }
        // Fallback to global if per-class has insufficient data.
        if state.global.blend.n >= self.config.min_samples {
            return state.global.blend.mean();
        }
        // Default prior mean.
        BlendPosterior::default().mean()
    }

    fn resolve_k(&self, state: &AdaptiveState, query_class: QueryClass) -> f64 {
        if let Some(cs) = state.per_class.get(&query_class) {
            if cs.k.n >= self.config.min_samples {
                return cs.k.mean();
            }
        }
        if state.global.k.n >= self.config.min_samples {
            return state.global.k.mean();
        }
        KPosterior::default().mean()
    }
}

/// Serializable snapshot of adaptive state for inspection or persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSnapshot {
    /// Global blend posterior.
    pub global_blend: BlendPosterior,
    /// Global K posterior.
    pub global_k: KPosterior,
    /// Per-query-class states: `(query_class, blend_posterior, k_posterior)`.
    pub per_class: Vec<(QueryClass, BlendPosterior, KPosterior)>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AdaptiveConfig {
        AdaptiveConfig {
            min_samples: 5, // Low threshold for tests.
            ..AdaptiveConfig::default()
        }
    }

    // --- BlendPosterior ---

    #[test]
    fn blend_prior_encodes_0_7() {
        let bp = BlendPosterior::default();
        assert!((bp.mean() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn blend_update_successes_increase_mean() {
        let mut bp = BlendPosterior::default();
        let initial = bp.mean();
        for _ in 0..20 {
            bp.update(true);
        }
        assert!(bp.mean() > initial);
    }

    #[test]
    fn blend_update_failures_decrease_mean() {
        let mut bp = BlendPosterior::default();
        let initial = bp.mean();
        for _ in 0..20 {
            bp.update(false);
        }
        assert!(bp.mean() < initial);
    }

    #[test]
    fn blend_variance_decreases_with_observations() {
        let mut bp = BlendPosterior::default();
        let initial_var = bp.variance();
        for _ in 0..50 {
            bp.update(true);
        }
        assert!(bp.variance() < initial_var);
    }

    #[test]
    fn blend_converges_toward_observed_rate() {
        let mut bp = BlendPosterior::default();
        // 100 observations: 80% success rate.
        for i in 0..100 {
            bp.update(i % 5 != 0); // 80% true, 20% false
        }
        // Posterior mean should converge toward ~0.8.
        assert!((bp.mean() - 0.8).abs() < 0.05);
    }

    // --- KPosterior ---

    #[test]
    fn k_prior_encodes_60() {
        let kp = KPosterior::default();
        assert!((kp.mean() - 60.0).abs() < 1e-10);
    }

    #[test]
    fn k_converges_to_observed_value() {
        let mut kp = KPosterior::default();
        // 100 observations of K=80.
        for _ in 0..100 {
            kp.update(80.0);
        }
        // Posterior mean should converge toward ~80.
        assert!((kp.mean() - 80.0).abs() < 2.0);
    }

    #[test]
    fn k_variance_decreases_with_observations() {
        let mut kp = KPosterior::default();
        let initial_var = kp.sigma_sq;
        for _ in 0..50 {
            kp.update(60.0);
        }
        assert!(kp.sigma_sq < initial_var);
    }

    #[test]
    fn k_zero_observations_returns_prior() {
        let kp = KPosterior::default();
        assert!((kp.mu - 60.0).abs() < f64::EPSILON);
        assert!((kp.sigma_sq - 100.0).abs() < f64::EPSILON);
    }

    // --- AdaptiveFusion ---

    #[test]
    fn default_blend_is_0_7() {
        let af = AdaptiveFusion::with_defaults();
        assert!((af.blend_factor(QueryClass::NaturalLanguage) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn default_k_is_60() {
        let af = AdaptiveFusion::with_defaults();
        assert!((af.rrf_k(QueryClass::NaturalLanguage) - 60.0).abs() < 1e-10);
    }

    #[test]
    fn blend_adapts_after_min_samples() {
        let af = AdaptiveFusion::new(test_config());

        // Below min_samples: returns prior default.
        for _ in 0..4 {
            af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);
        }
        assert!((af.blend_factor(QueryClass::NaturalLanguage) - 0.7).abs() < 1e-10);

        // Cross min_samples threshold.
        af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);

        // Now per-class posterior is used (should be > 0.7 with all successes).
        assert!(af.blend_factor(QueryClass::NaturalLanguage) > 0.7);
    }

    #[test]
    fn k_adapts_after_min_samples() {
        let af = AdaptiveFusion::new(test_config());

        for _ in 0..5 {
            af.update_k(QueryClass::ShortKeyword, 80.0, SignalSource::NdcgEval);
        }

        // Per-class posterior used: should shift toward 80.
        let k = af.rrf_k(QueryClass::ShortKeyword);
        assert!(k > 60.0);
    }

    #[test]
    fn per_class_independence() {
        let af = AdaptiveFusion::new(test_config());

        // Update Identifier class.
        for _ in 0..10 {
            af.update_blend(QueryClass::Identifier, false, SignalSource::Skip);
        }

        // NaturalLanguage class should still be at prior default.
        assert!((af.blend_factor(QueryClass::NaturalLanguage) - 0.7).abs() < 1e-10);

        // Identifier should have adapted downward.
        assert!(af.blend_factor(QueryClass::Identifier) < 0.7);
    }

    #[test]
    fn blend_clamped_to_safety_bounds() {
        let config = AdaptiveConfig {
            min_samples: 0,
            blend_min: 0.2,
            blend_max: 0.9,
            ..AdaptiveConfig::default()
        };
        let af = AdaptiveFusion::new(config);

        // Push blend very high.
        for _ in 0..200 {
            af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);
        }
        assert!(af.blend_factor(QueryClass::NaturalLanguage) <= 0.9);

        // Push blend very low for a different class.
        for _ in 0..200 {
            af.update_blend(QueryClass::Identifier, false, SignalSource::Skip);
        }
        assert!(af.blend_factor(QueryClass::Identifier) >= 0.2);
    }

    #[test]
    fn k_clamped_to_safety_bounds() {
        let config = AdaptiveConfig {
            min_samples: 0,
            k_min: 10.0,
            k_max: 150.0,
            ..AdaptiveConfig::default()
        };
        let af = AdaptiveFusion::new(config);

        // Push K very high.
        for _ in 0..200 {
            af.update_k(QueryClass::NaturalLanguage, 500.0, SignalSource::NdcgEval);
        }
        assert!(af.rrf_k(QueryClass::NaturalLanguage) <= 150.0);
    }

    #[test]
    fn global_fallback_when_class_insufficient() {
        let config = AdaptiveConfig {
            min_samples: 10,
            ..AdaptiveConfig::default()
        };
        let af = AdaptiveFusion::new(config);

        // Feed global via one class (crosses min_samples).
        for _ in 0..15 {
            af.update_blend(QueryClass::ShortKeyword, true, SignalSource::Click);
        }

        // Query a different class with no data — should use global fallback.
        // Global has 15 observations (>= 10 min_samples).
        let blend = af.blend_factor(QueryClass::Identifier);
        // Global posterior mean should be > prior 0.7 (all successes).
        assert!(blend > 0.7);
    }

    #[test]
    fn evidence_event_emitted() {
        let af = AdaptiveFusion::with_defaults();
        let event = af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);

        assert_eq!(event.query_class, QueryClass::NaturalLanguage);
        assert_eq!(event.signal_source, SignalSource::Click);
        assert!(event.blend_posterior.0 > 7.0); // alpha increased
    }

    #[test]
    fn snapshot_captures_state() {
        let af = AdaptiveFusion::new(test_config());
        af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);
        af.update_k(QueryClass::ShortKeyword, 70.0, SignalSource::NdcgEval);

        let snap = af.snapshot();
        assert!(snap.global_blend.n > 0);
        assert!(snap.global_k.n > 0);
        assert!(!snap.per_class.is_empty());
    }

    #[test]
    fn snapshot_serde_roundtrip() {
        let af = AdaptiveFusion::new(test_config());
        af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);

        let snap = af.snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let decoded: AdaptiveSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert!((decoded.global_blend.alpha - snap.global_blend.alpha).abs() < f64::EPSILON);
    }

    #[test]
    fn concurrent_updates() {
        use std::sync::Arc;

        let af = Arc::new(AdaptiveFusion::new(test_config()));
        let mut handles = Vec::new();

        for t in 0..4 {
            let af = Arc::clone(&af);
            handles.push(std::thread::spawn(move || {
                let qc = match t % 4 {
                    0 => QueryClass::NaturalLanguage,
                    1 => QueryClass::ShortKeyword,
                    2 => QueryClass::Identifier,
                    _ => QueryClass::Empty,
                };
                for _ in 0..50 {
                    af.update_blend(qc, true, SignalSource::Click);
                    af.update_k(qc, 60.0 + f64::from(t), SignalSource::NdcgEval);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        // Global should have 200 blend observations (4 threads × 50).
        let snap = af.snapshot();
        assert_eq!(snap.global_blend.n, 200);
        assert_eq!(snap.global_k.n, 200);
    }

    #[test]
    fn debug_output() {
        let af = AdaptiveFusion::with_defaults();
        let debug = format!("{af:?}");
        assert!(debug.contains("AdaptiveFusion"));
    }

    // --- Reset ---

    #[test]
    fn reset_clears_observations_to_prior() {
        let af = AdaptiveFusion::new(test_config());

        // Feed many observations so posterior diverges from prior.
        for _ in 0..20 {
            af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);
            af.update_k(QueryClass::NaturalLanguage, 80.0, SignalSource::NdcgEval);
        }

        // Verify parameters have shifted.
        let snap_before = af.snapshot();
        assert!(snap_before.global_blend.n > 0);
        assert!(snap_before.global_k.n > 0);

        // Reset.
        af.reset();

        // After reset, should be back at prior.
        let snap_after = af.snapshot();
        assert_eq!(snap_after.global_blend.n, 0);
        assert_eq!(snap_after.global_k.n, 0);
        assert!((snap_after.global_blend.alpha - 7.0).abs() < f64::EPSILON);
        assert!((snap_after.global_blend.beta - 3.0).abs() < f64::EPSILON);
        assert!((snap_after.global_k.mu - 60.0).abs() < f64::EPSILON);
        assert!(snap_after.per_class.is_empty());

        // Blend factor should return to default 0.7.
        assert!((af.blend_factor(QueryClass::NaturalLanguage) - 0.7).abs() < 1e-10);
        assert!((af.rrf_k(QueryClass::NaturalLanguage) - 60.0).abs() < 1e-10);
    }

    // --- Determinism ---

    #[test]
    fn determinism_same_sequence_same_result() {
        let run = || {
            let af = AdaptiveFusion::new(test_config());
            let observations = [
                (QueryClass::NaturalLanguage, true),
                (QueryClass::NaturalLanguage, false),
                (QueryClass::ShortKeyword, true),
                (QueryClass::ShortKeyword, true),
                (QueryClass::Identifier, false),
                (QueryClass::NaturalLanguage, true),
                (QueryClass::NaturalLanguage, true),
            ];
            for &(qc, success) in &observations {
                af.update_blend(qc, success, SignalSource::Click);
            }
            let k_observations = [
                (QueryClass::NaturalLanguage, 55.0),
                (QueryClass::ShortKeyword, 70.0),
                (QueryClass::NaturalLanguage, 45.0),
                (QueryClass::Identifier, 80.0),
            ];
            for &(qc, k) in &k_observations {
                af.update_k(qc, k, SignalSource::NdcgEval);
            }
            af.snapshot()
        };

        let snap_a = run();
        let snap_b = run();

        assert!((snap_a.global_blend.alpha - snap_b.global_blend.alpha).abs() < f64::EPSILON);
        assert!((snap_a.global_blend.beta - snap_b.global_blend.beta).abs() < f64::EPSILON);
        assert!((snap_a.global_k.mu - snap_b.global_k.mu).abs() < f64::EPSILON);
        assert!((snap_a.global_k.sigma_sq - snap_b.global_k.sigma_sq).abs() < f64::EPSILON);
        assert_eq!(snap_a.per_class.len(), snap_b.per_class.len());
    }

    // --- Numerical stability ---

    #[test]
    fn numerical_stability_no_nan_or_inf() {
        let af = AdaptiveFusion::new(AdaptiveConfig {
            min_samples: 0,
            ..AdaptiveConfig::default()
        });

        // Many alternating observations to stress the posteriors.
        for i in 0..1_000 {
            let success = i % 3 != 0;
            af.update_blend(QueryClass::NaturalLanguage, success, SignalSource::Click);
            af.update_k(
                QueryClass::NaturalLanguage,
                if i % 2 == 0 { 1.0 } else { 200.0 },
                SignalSource::NdcgEval,
            );
        }

        let blend = af.blend_factor(QueryClass::NaturalLanguage);
        let k = af.rrf_k(QueryClass::NaturalLanguage);
        assert!(blend.is_finite(), "blend factor must be finite: {blend}");
        assert!(k.is_finite(), "RRF K must be finite: {k}");
        assert!(!blend.is_nan(), "blend factor must not be NaN");
        assert!(!k.is_nan(), "RRF K must not be NaN");

        let snap = af.snapshot();
        assert!(snap.global_blend.alpha.is_finite());
        assert!(snap.global_blend.beta.is_finite());
        assert!(snap.global_k.mu.is_finite());
        assert!(snap.global_k.sigma_sq.is_finite());
        assert!(snap.global_k.sigma_sq > 0.0, "K variance must stay positive");
    }

    // --- Conflicting observations edge case ---

    #[test]
    fn conflicting_observations_stay_near_prior() {
        let af = AdaptiveFusion::new(AdaptiveConfig {
            min_samples: 0,
            ..AdaptiveConfig::default()
        });

        // Equal success and failure → blend stays near prior.
        for _ in 0..100 {
            af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);
            af.update_blend(QueryClass::NaturalLanguage, false, SignalSource::Skip);
        }

        let blend = af.blend_factor(QueryClass::NaturalLanguage);
        // 200 observations: 100 success + 7 prior alpha, 100 failure + 3 prior beta
        // Expected mean: 107 / 210 ≈ 0.5095
        assert!(
            (blend - 0.5).abs() < 0.1,
            "conflicting 50/50 observations should pull blend near 0.5, got {blend}"
        );

        let snap = af.snapshot();
        let var = snap.global_blend.alpha * snap.global_blend.beta
            / ((snap.global_blend.alpha + snap.global_blend.beta).powi(2)
                * (snap.global_blend.alpha + snap.global_blend.beta + 1.0));
        assert!(var > 0.0, "variance should remain positive with conflicting data");
    }

    // --- Identical observations edge case ---

    #[test]
    fn identical_observations_converge_precisely() {
        let af = AdaptiveFusion::new(AdaptiveConfig {
            min_samples: 0,
            ..AdaptiveConfig::default()
        });

        // All successes.
        for _ in 0..200 {
            af.update_blend(QueryClass::ShortKeyword, true, SignalSource::Click);
        }

        let blend = af.blend_factor(QueryClass::ShortKeyword);
        // (7 + 200) / (7 + 200 + 3) = 207/210 ≈ 0.9857, clamped to 0.95
        assert!(
            blend >= 0.95 - f64::EPSILON,
            "all-success should hit safety clamp, got {blend}"
        );

        // All observations of K=100.
        for _ in 0..200 {
            af.update_k(QueryClass::ShortKeyword, 100.0, SignalSource::NdcgEval);
        }

        let k = af.rrf_k(QueryClass::ShortKeyword);
        assert!(
            (k - 100.0).abs() < 1.0,
            "identical K observations should converge precisely, got {k}"
        );
    }

    // --- Config serde roundtrip ---

    #[test]
    fn config_serde_roundtrip() {
        let config = AdaptiveConfig {
            min_samples: 42,
            blend_min: 0.15,
            blend_max: 0.85,
            k_min: 5.0,
            k_max: 100.0,
            thompson_sampling: true,
            seed: Some(12345),
        };
        let json = serde_json::to_string(&config).expect("serialize config");
        let decoded: AdaptiveConfig = serde_json::from_str(&json).expect("deserialize config");
        assert_eq!(decoded.min_samples, 42);
        assert!((decoded.blend_min - 0.15).abs() < f64::EPSILON);
        assert!(decoded.thompson_sampling);
        assert_eq!(decoded.seed, Some(12345));
    }

    // --- Blend posterior variance ---

    #[test]
    fn blend_posterior_variance_formulas() {
        let bp = BlendPosterior {
            alpha: 10.0,
            beta: 5.0,
            n: 5,
        };
        // Variance = alpha*beta / (alpha+beta)^2 / (alpha+beta+1)
        let expected = (10.0 * 5.0) / (15.0 * 15.0 * 16.0);
        assert!((bp.variance() - expected).abs() < 1e-12);
    }

    // --- K posterior std_dev ---

    #[test]
    fn k_posterior_std_dev() {
        let kp = KPosterior {
            mu: 60.0,
            sigma_sq: 25.0,
            sigma_obs_sq: 225.0,
            n: 0,
        };
        assert!((kp.std_dev() - 5.0).abs() < f64::EPSILON);
    }

    // --- Signal source serde ---

    #[test]
    fn signal_source_serde_all_variants() {
        for signal in [
            SignalSource::Click,
            SignalSource::Skip,
            SignalSource::Dwell,
            SignalSource::NdcgEval,
        ] {
            let json = serde_json::to_string(&signal).expect("serialize signal");
            let decoded: SignalSource = serde_json::from_str(&json).expect("deserialize signal");
            assert_eq!(decoded, signal);
        }
    }

    // --- Evidence event serde ---

    #[test]
    fn evidence_event_serde_roundtrip() {
        let event = EvidenceEvent {
            query_class: QueryClass::NaturalLanguage,
            blend_used: 0.72,
            k_used: 61.5,
            blend_posterior: (8.0, 3.0),
            k_posterior: (61.5, 90.0),
            signal_source: SignalSource::Click,
        };
        let json = serde_json::to_string(&event).expect("serialize event");
        let decoded: EvidenceEvent = serde_json::from_str(&json).expect("deserialize event");
        assert_eq!(decoded.query_class, QueryClass::NaturalLanguage);
        assert!((decoded.blend_used - 0.72).abs() < f64::EPSILON);
    }

    // --- Multiple query classes in snapshot ---

    #[test]
    fn snapshot_captures_all_query_classes() {
        let af = AdaptiveFusion::new(test_config());

        af.update_blend(QueryClass::NaturalLanguage, true, SignalSource::Click);
        af.update_blend(QueryClass::ShortKeyword, false, SignalSource::Skip);
        af.update_blend(QueryClass::Identifier, true, SignalSource::Dwell);
        af.update_blend(QueryClass::Empty, false, SignalSource::NdcgEval);

        let snap = af.snapshot();
        assert_eq!(snap.per_class.len(), 4);

        let classes: Vec<QueryClass> = snap.per_class.iter().map(|(qc, _, _)| *qc).collect();
        assert!(classes.contains(&QueryClass::NaturalLanguage));
        assert!(classes.contains(&QueryClass::ShortKeyword));
        assert!(classes.contains(&QueryClass::Identifier));
        assert!(classes.contains(&QueryClass::Empty));
    }
}
