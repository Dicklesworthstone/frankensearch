//! Sequential testing phase gate using e-processes (anytime-valid).
//!
//! An e-process accumulates evidence over queries and can be checked at **any**
//! time without multiple-testing correction. When the running e-value crosses
//! `1/alpha`, the decision is statistically guaranteed (Ville's inequality).
//!
//! # Integration
//!
//! The phase gate sits alongside `TwoTierSearcher`:
//! - Before quality embedding: check `gate.decision()`.
//! - If `SkipQuality`: skip Phase 2 entirely.
//! - After each query: `gate.update(observation)`.
//! - If `AlwaysRefine`: always run Phase 2.
//!
//! # References
//!
//! - Ramdas et al. (2020) "Admissible Anytime-Valid Sequential Testing"
//! - Grünwald et al. (2019) "Safe Testing"

use serde::{Deserialize, Serialize};

use frankensearch_core::decision_plane::{
    EvidenceEventType, EvidenceRecord, PipelineAction, PipelineState, ReasonCode, Severity,
};

// ─── Decision ─────────────────────────────────────────────────────────────────

/// Phase transition decision reached by accumulated evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseDecision {
    /// Evidence: fast tier is sufficient, quality tier adds negligible value.
    SkipQuality,
    /// Evidence: quality tier materially improves results.
    AlwaysRefine,
}

// ─── Observation ──────────────────────────────────────────────────────────────

/// Per-query observation fed into the e-process.
#[derive(Debug, Clone)]
pub struct PhaseObservation {
    /// Best score from fast tier only.
    pub fast_score: f64,
    /// Best score from quality tier (after blend/rerank).
    pub quality_score: f64,
    /// Whether the user interacted with a quality-promoted result.
    /// `None` if no feedback is available (e.g. batch mode).
    pub user_signal: Option<bool>,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the sequential testing phase gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseGateConfig {
    /// Significance level. Decision is reached when e-value > 1/alpha.
    /// Default: 0.05 (95% confidence).
    pub alpha: f64,
    /// Maximum queries before forcing a decision and resetting.
    /// Default: 500.
    pub timeout_queries: u64,
    /// Minimum score delta considered meaningful. Observations with
    /// `|quality_score - fast_score|` below this are treated as ties.
    /// Default: 0.01.
    pub min_delta: f64,
    /// Whether the gate is enabled. When disabled, no decision is ever
    /// reached and the searcher always refines.
    /// Default: true.
    pub enabled: bool,
    /// Time-based expiry in milliseconds. If set, the accumulated evidence and
    /// any permanent decision will be reset if no updates occur within this window,
    /// allowing recovery from transient quality-tier lockouts.
    pub expiry_ms: Option<u64>,
}

impl Default for PhaseGateConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            timeout_queries: 500,
            min_delta: 0.01,
            enabled: true,
            expiry_ms: Some(1000 * 60 * 60), // 1 hour default expiry
        }
    }
}

// ─── Phase Gate ──────────────────────────────────────────────────────────────

/// Anytime-valid sequential testing gate for phase transition decisions.
///
/// Accumulates a running e-value (product of per-query e-factors). The gate
/// remains undecided until sufficient evidence accumulates. Decisions are
/// statistically guaranteed: under the null hypothesis,
/// `P(e-value ever exceeds 1/alpha) <= alpha`.
///
/// Thread-safety: this struct is **not** `Sync`. The `TwoTierSearcher` should
/// hold it behind a `Mutex` or update it from a single thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGate {
    config: PhaseGateConfig,
    /// Running e-value (product of per-query e-factors).
    /// Starts at 1.0 (no evidence).
    e_value: f64,
    /// Current decision, if any.
    decision: Option<PhaseDecision>,
    /// Number of observations since last reset.
    observations: u64,
    /// Running count of observations where quality was better.
    quality_wins: u64,
    /// Running count of observations where fast was better (or tied).
    fast_wins: u64,
    /// Timestamp of the last observation, used for expiry.
    #[serde(skip)]
    last_updated: Option<std::time::Instant>,
}

impl PhaseGate {
    /// Create a new phase gate with the given configuration.
    #[must_use]
    pub fn new(config: PhaseGateConfig) -> Self {
        Self {
            config,
            e_value: 1.0,
            decision: None,
            observations: 0,
            quality_wins: 0,
            fast_wins: 0,
            last_updated: None,
        }
    }

    /// Create a new phase gate with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(PhaseGateConfig::default())
    }

    /// The current decision, if any.
    ///
    /// Returns `None` when evidence is insufficient or if the gate has expired.
    /// The caller should default to always refining when `None`.
    #[must_use]
    pub fn decision(&self) -> Option<PhaseDecision> {
        if self.is_expired() {
            None
        } else {
            self.decision
        }
    }

    /// Whether the gate recommends skipping the quality tier.
    #[must_use]
    pub fn should_skip_quality(&self) -> bool {
        self.decision() == Some(PhaseDecision::SkipQuality)
    }

    /// Returns `true` if the gate has expired due to lack of updates.
    fn is_expired(&self) -> bool {
        if let (Some(expiry), Some(last)) = (self.config.expiry_ms, self.last_updated) {
            last.elapsed().as_millis() as u64 > expiry
        } else {
            false
        }
    }

    /// Update the gate with a new observation.
    ///
    /// Returns evidence records for any state transitions (decision reached
    /// or timeout-forced reset).
    pub fn update(&mut self, obs: &PhaseObservation) -> Vec<EvidenceRecord> {
        let mut evidence = Vec::new();
        if self.is_expired() {
            evidence.extend(self.reset());
        }
        self.last_updated = Some(std::time::Instant::now());

        if !self.config.enabled || self.decision.is_some() {
            return evidence;
        }

        self.observations += 1;

        let delta = obs.quality_score - obs.fast_score;
        // NaN delta silently biases toward SkipQuality — discard the observation.
        if !delta.is_finite() {
            return Vec::new();
        }
        let e_factor = self.compute_e_factor(delta, obs.user_signal);

        // Clamp e_factor to avoid numerical explosion.
        let e_factor = e_factor.clamp(0.01, 100.0);
        self.e_value *= e_factor;

        // Track win counts for diagnostics.
        // NaN min_delta → treat as 0.0 (any positive delta is a quality win).
        let min_delta = sanitize_min_delta(self.config.min_delta);
        if delta > min_delta {
            self.quality_wins += 1;
        } else {
            self.fast_wins += 1;
        }

        let threshold = 1.0 / sanitize_alpha(self.config.alpha);

        // Check for decision: quality adds value.
        if self.e_value >= threshold {
            self.decision = Some(PhaseDecision::AlwaysRefine);
            return vec![self.make_decision_evidence(PhaseDecision::AlwaysRefine)];
        }

        // Check for inverse decision: fast is sufficient.
        // The inverse e-value is 1/e_value.
        if self.e_value > 0.0 && (1.0 / self.e_value) >= threshold {
            self.decision = Some(PhaseDecision::SkipQuality);
            return vec![self.make_decision_evidence(PhaseDecision::SkipQuality)];
        }

        // Timeout: force a decision based on accumulated evidence.
        if self.observations >= self.config.timeout_queries {
            let forced = if self.quality_wins > self.fast_wins {
                PhaseDecision::AlwaysRefine
            } else {
                PhaseDecision::SkipQuality
            };
            self.decision = Some(forced);

            let mut evidence = self.make_decision_evidence(forced);
            evidence.reason_human = format!(
                "Phase gate timeout after {} queries; forced decision: {forced:?} \
                 (quality_wins={}, fast_wins={}, e_value={:.4})",
                self.observations, self.quality_wins, self.fast_wins, self.e_value,
            );
            return vec![evidence];
        }

        Vec::new()
    }

    /// Reset the gate, clearing accumulated evidence and decision.
    ///
    /// Call this after index rebuilds or significant distribution shifts.
    pub fn reset(&mut self) -> Vec<EvidenceRecord> {
        let had_decision = self.decision.is_some();
        self.e_value = 1.0;
        self.decision = None;
        self.observations = 0;
        self.quality_wins = 0;
        self.fast_wins = 0;

        if had_decision {
            vec![
                EvidenceRecord::new(
                    EvidenceEventType::Transition,
                    ReasonCode::DECISION_REFINE_NOMINAL,
                    "Phase gate reset; returning to undecided state",
                    Severity::Info,
                    PipelineState::Nominal,
                    "phase_gate",
                )
                .with_action(PipelineAction::Refine),
            ]
        } else {
            Vec::new()
        }
    }

    /// Current running e-value.
    #[must_use]
    pub const fn e_value(&self) -> f64 {
        self.e_value
    }

    /// Number of observations since last reset.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Snapshot of gate diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> PhaseGateDiagnostics {
        PhaseGateDiagnostics {
            e_value: self.e_value,
            decision: self.decision,
            observations: self.observations,
            quality_wins: self.quality_wins,
            fast_wins: self.fast_wins,
            threshold: 1.0 / sanitize_alpha(self.config.alpha),
        }
    }

    /// Current configuration.
    #[must_use]
    pub const fn config(&self) -> &PhaseGateConfig {
        &self.config
    }

    // ─── Internal ─────────────────────────────────────────────────────

    /// Compute the e-factor for a single observation.
    ///
    /// Uses a simple likelihood ratio: under H1 (quality adds value),
    /// observations where quality > fast are more likely. Under H0 (fast
    /// is sufficient), the scores are exchangeable.
    ///
    /// The e-factor is:
    /// - `> 1.0` when quality beats fast (evidence for `AlwaysRefine`)
    /// - `< 1.0` when fast beats quality (evidence for `SkipQuality`)
    /// - `= 1.0` when tied (no evidence)
    fn compute_e_factor(&self, delta: f64, user_signal: Option<bool>) -> f64 {
        // Base e-factor from score delta.
        let min_delta = sanitize_min_delta(self.config.min_delta);
        let base = if delta.abs() < min_delta {
            // Tie: no evidence.
            1.0
        } else if delta > 0.0 {
            // Quality better: evidence for AlwaysRefine.
            // Smoothly increasing with delta magnitude.
            1.0 + delta.min(1.0)
        } else {
            // Fast better: evidence for SkipQuality.
            // Smoothly decreasing with delta magnitude.
            1.0 / (1.0 + delta.abs().min(1.0))
        };

        // Incorporate user signal if available.
        match user_signal {
            Some(true) => {
                // User engaged with quality-promoted result: strong evidence.
                base * 1.5
            }
            Some(false) => {
                // User did NOT engage: evidence against quality tier.
                base * 0.7
            }
            None => base,
        }
    }

    fn make_decision_evidence(&self, decision: PhaseDecision) -> EvidenceRecord {
        let (reason_code, description, state, action) = match decision {
            PhaseDecision::SkipQuality => (
                ReasonCode::DECISION_SKIP_HIGH_LOSS,
                format!(
                    "Phase gate decided: skip quality (e_value={:.4}, obs={}, \
                     fast_wins={}, quality_wins={})",
                    self.e_value, self.observations, self.fast_wins, self.quality_wins,
                ),
                PipelineState::DegradedQuality,
                PipelineAction::SkipRefinement,
            ),
            PhaseDecision::AlwaysRefine => (
                ReasonCode::DECISION_REFINE_NOMINAL,
                format!(
                    "Phase gate decided: always refine (e_value={:.4}, obs={}, \
                     quality_wins={}, fast_wins={})",
                    self.e_value, self.observations, self.quality_wins, self.fast_wins,
                ),
                PipelineState::Nominal,
                PipelineAction::Refine,
            ),
        };

        EvidenceRecord::new(
            EvidenceEventType::Decision,
            reason_code,
            description,
            Severity::Info,
            state,
            "phase_gate",
        )
        .with_action(action)
    }
}

/// Diagnostic snapshot of the phase gate state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGateDiagnostics {
    /// Current running e-value.
    pub e_value: f64,
    /// Current decision, if reached.
    pub decision: Option<PhaseDecision>,
    /// Total observations since last reset.
    pub observations: u64,
    /// Observations where quality tier won.
    pub quality_wins: u64,
    /// Observations where fast tier won (or tied).
    pub fast_wins: u64,
    /// Decision threshold (1/alpha).
    pub threshold: f64,
}

// ─── Sanitization Helpers ────────────────────────────────────────────────────

/// Sanitize alpha to a valid significance level in `(0, 1)`.
/// Returns the default 0.05 for non-finite, zero, negative, or >= 1.0 values.
fn sanitize_alpha(alpha: f64) -> f64 {
    if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        alpha
    } else {
        0.05
    }
}

/// Sanitize `min_delta` to a non-negative finite value.
/// Returns 0.0 for NaN, negative, or infinite values (any positive delta is a win).
fn sanitize_min_delta(min_delta: f64) -> f64 {
    if min_delta.is_finite() && min_delta >= 0.0 {
        min_delta
    } else {
        0.0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PhaseGateConfig {
        PhaseGateConfig {
            alpha: 0.05,
            timeout_queries: 100,
            min_delta: 0.01,
            enabled: true,
            expiry_ms: None,
        }
    }

    fn quality_better_obs(delta: f64) -> PhaseObservation {
        PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.5 + delta,
            user_signal: None,
        }
    }

    fn fast_better_obs(delta: f64) -> PhaseObservation {
        PhaseObservation {
            fast_score: 0.5 + delta,
            quality_score: 0.5,
            user_signal: None,
        }
    }

    fn tied_obs() -> PhaseObservation {
        PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.5,
            user_signal: None,
        }
    }

    // ─── Initial State ──────────────────────────────────────────────

    #[test]
    fn initial_state_undecided() {
        let gate = PhaseGate::with_defaults();
        assert!(gate.decision().is_none());
        assert!(!gate.should_skip_quality());
        assert!((gate.e_value() - 1.0).abs() < f64::EPSILON);
        assert_eq!(gate.observations(), 0);
    }

    #[test]
    fn default_config_values() {
        let config = PhaseGateConfig::default();
        assert!((config.alpha - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.timeout_queries, 500);
        assert!((config.min_delta - 0.01).abs() < f64::EPSILON);
        assert!(config.enabled);
    }

    // ─── Quality Wins → AlwaysRefine ────────────────────────────────

    #[test]
    fn consistent_quality_wins_decides_always_refine() {
        let mut gate = PhaseGate::new(test_config());

        // Feed many observations where quality is consistently better.
        for _ in 0..50 {
            let evidence = gate.update(&quality_better_obs(0.3));
            if gate.decision().is_some() {
                assert_eq!(gate.decision(), Some(PhaseDecision::AlwaysRefine));
                assert!(!evidence.is_empty());
                return;
            }
        }

        // If we get here with no decision, the e-value should at least be > 1.
        assert!(gate.e_value() > 1.0);
    }

    // ─── Fast Wins → SkipQuality ─────────────────────────────────────

    #[test]
    fn consistent_fast_wins_decides_skip_quality() {
        let mut gate = PhaseGate::new(test_config());

        for _ in 0..50 {
            let evidence = gate.update(&fast_better_obs(0.3));
            if gate.decision().is_some() {
                assert_eq!(gate.decision(), Some(PhaseDecision::SkipQuality));
                assert!(gate.should_skip_quality());
                assert!(!evidence.is_empty());
                return;
            }
        }

        // E-value should be < 1 (evidence against quality).
        assert!(gate.e_value() < 1.0);
    }

    // ─── Ties → No Decision ────────────────────────────────────────

    #[test]
    fn ties_produce_no_evidence() {
        let mut gate = PhaseGate::new(test_config());

        for _ in 0..20 {
            let evidence = gate.update(&tied_obs());
            assert!(evidence.is_empty());
        }

        assert!(gate.decision().is_none());
        assert!((gate.e_value() - 1.0).abs() < f64::EPSILON);
    }

    // ─── User Signal Amplification ──────────────────────────────────

    #[test]
    fn user_positive_signal_amplifies_quality_evidence() {
        let mut gate_with_signal = PhaseGate::new(test_config());
        let mut gate_without = PhaseGate::new(test_config());

        let obs_with = PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.6,
            user_signal: Some(true),
        };
        let obs_without = PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.6,
            user_signal: None,
        };

        gate_with_signal.update(&obs_with);
        gate_without.update(&obs_without);

        // With positive user signal, e-value should be higher.
        assert!(gate_with_signal.e_value() > gate_without.e_value());
    }

    #[test]
    fn user_negative_signal_dampens_evidence() {
        let mut gate_with_signal = PhaseGate::new(test_config());
        let mut gate_without = PhaseGate::new(test_config());

        let obs_with = PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.6,
            user_signal: Some(false),
        };
        let obs_without = PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.6,
            user_signal: None,
        };

        gate_with_signal.update(&obs_with);
        gate_without.update(&obs_without);

        // With negative signal, e-value should be lower.
        assert!(gate_with_signal.e_value() < gate_without.e_value());
    }

    // ─── Timeout ────────────────────────────────────────────────────

    #[test]
    fn timeout_forces_decision() {
        let config = PhaseGateConfig {
            timeout_queries: 5,
            ..test_config()
        };
        let mut gate = PhaseGate::new(config);

        // Feed mixed observations that won't trigger the threshold.
        for i in 0..5 {
            if i % 2 == 0 {
                gate.update(&quality_better_obs(0.02));
            } else {
                gate.update(&fast_better_obs(0.02));
            }
        }

        // Should have forced a decision at observation 5.
        assert!(gate.decision().is_some());
    }

    #[test]
    fn timeout_favors_quality_when_quality_wins_more() {
        let config = PhaseGateConfig {
            timeout_queries: 5,
            min_delta: 0.001,
            ..test_config()
        };
        let mut gate = PhaseGate::new(config);

        // 3 quality wins, 2 fast wins.
        gate.update(&quality_better_obs(0.1));
        gate.update(&fast_better_obs(0.1));
        gate.update(&quality_better_obs(0.1));
        gate.update(&quality_better_obs(0.1));
        gate.update(&fast_better_obs(0.1));

        assert_eq!(gate.decision(), Some(PhaseDecision::AlwaysRefine));
    }

    #[test]
    fn timeout_favors_fast_when_fast_wins_more() {
        let config = PhaseGateConfig {
            timeout_queries: 5,
            min_delta: 0.001,
            ..test_config()
        };
        let mut gate = PhaseGate::new(config);

        // 2 quality wins, 3 fast wins.
        gate.update(&fast_better_obs(0.1));
        gate.update(&quality_better_obs(0.1));
        gate.update(&fast_better_obs(0.1));
        gate.update(&fast_better_obs(0.1));
        gate.update(&quality_better_obs(0.1));

        assert_eq!(gate.decision(), Some(PhaseDecision::SkipQuality));
    }

    // ─── Reset ──────────────────────────────────────────────────────

    #[test]
    fn reset_clears_decision_and_evidence() {
        let mut gate = PhaseGate::new(test_config());

        // Reach a decision.
        for _ in 0..50 {
            gate.update(&quality_better_obs(0.5));
            if gate.decision().is_some() {
                break;
            }
        }
        assert!(gate.decision().is_some());

        // Reset.
        let evidence = gate.reset();
        assert!(!evidence.is_empty());
        assert!(gate.decision().is_none());
        assert!((gate.e_value() - 1.0).abs() < f64::EPSILON);
        assert_eq!(gate.observations(), 0);
    }

    #[test]
    fn reset_without_decision_produces_no_evidence() {
        let mut gate = PhaseGate::new(test_config());
        gate.update(&quality_better_obs(0.1));

        let evidence = gate.reset();
        assert!(evidence.is_empty());
    }

    // ─── Disabled Gate ──────────────────────────────────────────────

    #[test]
    fn disabled_gate_never_decides() {
        let config = PhaseGateConfig {
            enabled: false,
            ..test_config()
        };
        let mut gate = PhaseGate::new(config);

        for _ in 0..100 {
            let evidence = gate.update(&quality_better_obs(0.9));
            assert!(evidence.is_empty());
        }
        assert!(gate.decision().is_none());
    }

    // ─── No Updates After Decision ──────────────────────────────────

    #[test]
    fn no_updates_after_decision() {
        let mut gate = PhaseGate::new(test_config());

        // Reach a decision.
        for _ in 0..50 {
            gate.update(&quality_better_obs(0.5));
            if gate.decision().is_some() {
                break;
            }
        }
        let decision = gate.decision();
        assert!(decision.is_some());

        let e_val = gate.e_value();
        let obs = gate.observations();

        // Further updates should be no-ops.
        let evidence = gate.update(&fast_better_obs(0.9));
        assert!(evidence.is_empty());
        assert!((gate.e_value() - e_val).abs() < f64::EPSILON);
        assert_eq!(gate.observations(), obs);
    }

    // ─── Diagnostics ────────────────────────────────────────────────

    #[test]
    fn diagnostics_reflect_state() {
        let mut gate = PhaseGate::new(test_config());

        gate.update(&quality_better_obs(0.2));
        gate.update(&fast_better_obs(0.2));

        let diag = gate.diagnostics();
        assert_eq!(diag.observations, 2);
        assert_eq!(diag.quality_wins, 1);
        assert_eq!(diag.fast_wins, 1);
        assert!(diag.decision.is_none());
        assert!((diag.threshold - 20.0).abs() < f64::EPSILON); // 1/0.05
    }

    // ─── Evidence Records ───────────────────────────────────────────

    #[test]
    fn decision_evidence_has_correct_fields() {
        let mut gate = PhaseGate::new(test_config());

        for _ in 0..50 {
            let evidence = gate.update(&quality_better_obs(0.5));
            if !evidence.is_empty() {
                let record = &evidence[0];
                assert_eq!(record.event_type, EvidenceEventType::Decision);
                assert_eq!(record.pipeline_state, PipelineState::Nominal);
                assert!(record.reason_human.contains("always refine"));
                return;
            }
        }

        panic!("Expected decision within 50 observations");
    }

    // ─── E-value Monotonicity ───────────────────────────────────────

    #[test]
    fn e_value_increases_with_consistent_quality_wins() {
        let mut gate = PhaseGate::new(test_config());
        let mut prev = gate.e_value();

        for _ in 0..5 {
            gate.update(&quality_better_obs(0.2));
            let curr = gate.e_value();
            assert!(curr >= prev, "e-value should increase: {curr} < {prev}");
            prev = curr;
            if gate.decision().is_some() {
                break;
            }
        }
    }

    #[test]
    fn e_value_decreases_with_consistent_fast_wins() {
        let mut gate = PhaseGate::new(test_config());
        let mut prev = gate.e_value();

        for _ in 0..5 {
            gate.update(&fast_better_obs(0.2));
            let curr = gate.e_value();
            assert!(curr <= prev, "e-value should decrease: {curr} > {prev}");
            prev = curr;
            if gate.decision().is_some() {
                break;
            }
        }
    }

    // ─── Serde ──────────────────────────────────────────────────────

    #[test]
    fn config_serde_roundtrip() {
        let config = test_config();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: PhaseGateConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, config);
    }

    #[test]
    fn gate_serde_roundtrip() {
        let mut gate = PhaseGate::new(test_config());
        gate.update(&quality_better_obs(0.3));
        gate.update(&fast_better_obs(0.1));

        let json = serde_json::to_string(&gate).unwrap();
        let decoded: PhaseGate = serde_json::from_str(&json).unwrap();

        assert!((decoded.e_value() - gate.e_value()).abs() < 1e-10);
        assert_eq!(decoded.observations(), gate.observations());
        assert_eq!(decoded.decision(), gate.decision());
    }

    #[test]
    fn diagnostics_serde_roundtrip() {
        let gate = PhaseGate::new(test_config());
        let diag = gate.diagnostics();
        let json = serde_json::to_string(&diag).unwrap();
        let decoded: PhaseGateDiagnostics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.observations, diag.observations);
    }

    // ─── E-factor Clamping ──────────────────────────────────────────

    #[test]
    fn extreme_observations_are_clamped() {
        let mut gate = PhaseGate::new(test_config());

        // Extreme quality win — e-factor clamped to 100.
        let obs = PhaseObservation {
            fast_score: 0.0,
            quality_score: 100.0,
            user_signal: Some(true),
        };
        gate.update(&obs);

        // E-value should be clamped, not infinity.
        assert!(gate.e_value().is_finite());
        assert!(gate.e_value() <= 100.0);
    }

    // ─── bd-f3la tests begin ───

    #[test]
    fn phase_decision_serde_roundtrip() {
        for decision in [PhaseDecision::SkipQuality, PhaseDecision::AlwaysRefine] {
            let json = serde_json::to_string(&decision).unwrap();
            let back: PhaseDecision = serde_json::from_str(&json).unwrap();
            assert_eq!(decision, back);
        }
    }

    #[test]
    fn phase_observation_debug_format() {
        let obs = PhaseObservation {
            fast_score: 0.8,
            quality_score: 0.9,
            user_signal: Some(true),
        };
        let debug = format!("{obs:?}");
        assert!(debug.contains("PhaseObservation"));
        assert!(debug.contains("0.8"));
        assert!(debug.contains("0.9"));
    }

    #[test]
    fn should_skip_quality_false_for_always_refine() {
        let mut gate = PhaseGate::new(test_config());

        // Drive to AlwaysRefine decision.
        for _ in 0..50 {
            gate.update(&quality_better_obs(0.5));
            if gate.decision() == Some(PhaseDecision::AlwaysRefine) {
                assert!(
                    !gate.should_skip_quality(),
                    "AlwaysRefine should NOT skip quality"
                );
                return;
            }
        }
        panic!("Expected AlwaysRefine within 50 observations");
    }

    #[test]
    fn config_accessor_matches_construction() {
        let config = PhaseGateConfig {
            alpha: 0.10,
            timeout_queries: 200,
            min_delta: 0.05,
            enabled: true,
            expiry_ms: None,
        };
        let gate = PhaseGate::new(config.clone());
        assert_eq!(*gate.config(), config);
    }

    #[test]
    fn phase_gate_debug_format() {
        let gate = PhaseGate::with_defaults();
        let debug = format!("{gate:?}");
        assert!(debug.contains("PhaseGate"));
        assert!(debug.contains("e_value"));
    }

    #[test]
    fn timeout_evidence_message_contains_stats() {
        let config = PhaseGateConfig {
            timeout_queries: 3,
            min_delta: 0.001,
            ..test_config()
        };
        let mut gate = PhaseGate::new(config);

        gate.update(&quality_better_obs(0.1));
        gate.update(&quality_better_obs(0.1));
        let evidence = gate.update(&quality_better_obs(0.1));

        // If decision was reached before timeout, skip this check.
        if gate.observations() >= 3 && !evidence.is_empty() {
            let record = &evidence[0];
            // Timeout evidence should contain stats in the message.
            let msg = &record.reason_human;
            assert!(
                msg.contains("quality_wins") || msg.contains("always refine"),
                "expected stats or decision in message: {msg}"
            );
        }
    }

    #[test]
    fn reset_evidence_has_transition_event_type() {
        let mut gate = PhaseGate::new(test_config());

        // Reach a decision first.
        for _ in 0..50 {
            gate.update(&quality_better_obs(0.5));
            if gate.decision().is_some() {
                break;
            }
        }
        assert!(gate.decision().is_some());

        let evidence = gate.reset();
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].event_type, EvidenceEventType::Transition);
        assert_eq!(evidence[0].severity, Severity::Info);
    }

    #[test]
    fn small_negative_delta_counted_as_fast_win() {
        let mut gate = PhaseGate::new(PhaseGateConfig {
            min_delta: 0.1,
            ..test_config()
        });

        // Delta = -0.05, abs(delta) < min_delta → counted as fast win (tie goes to fast).
        let obs = PhaseObservation {
            fast_score: 0.55,
            quality_score: 0.50,
            user_signal: None,
        };
        gate.update(&obs);

        let diag = gate.diagnostics();
        assert_eq!(diag.fast_wins, 1);
        assert_eq!(diag.quality_wins, 0);
        // E-value should stay at 1.0 (tie).
        assert!((diag.e_value - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn observations_counter_increments_per_update() {
        let mut gate = PhaseGate::new(test_config());
        assert_eq!(gate.observations(), 0);

        gate.update(&quality_better_obs(0.1));
        assert_eq!(gate.observations(), 1);

        gate.update(&fast_better_obs(0.1));
        assert_eq!(gate.observations(), 2);

        gate.update(&tied_obs());
        assert_eq!(gate.observations(), 3);
    }

    #[test]
    fn quality_wins_fast_wins_accurate_counting() {
        let mut gate = PhaseGate::new(test_config());

        gate.update(&quality_better_obs(0.2)); // quality wins
        gate.update(&fast_better_obs(0.2)); // fast wins
        gate.update(&tied_obs()); // tie → fast wins
        gate.update(&quality_better_obs(0.3)); // quality wins

        let diag = gate.diagnostics();
        assert_eq!(diag.quality_wins, 2);
        assert_eq!(diag.fast_wins, 2); // 1 explicit + 1 tie
    }

    // ─── bd-f3la tests end ───
}
