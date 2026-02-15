//! Decision Plane Contract: unified adaptive ranking control.
//!
//! This module defines the canonical types shared by all adaptive ranking
//! components in frankensearch:
//!
//! - **Adaptive fusion** (bd-21g): Bayesian online learning for blend weights and RRF K.
//! - **Score calibration** (bd-22k): Platt/isotonic/temperature scaling.
//! - **Sequential testing** (bd-2ps): E-process gates for phase transitions.
//! - **Conformal prediction** (bd-2yj): Distribution-free quality guarantees.
//! - **Circuit breaker** (bd-1do): Quality-tier health monitoring.
//! - **Relevance feedback** (bd-2tv): Implicit boost maps from user behavior.
//!
//! # Expected-Loss Model
//!
//! Adaptive decisions are modelled as expected-loss minimization over
//! [`PipelineState`] x [`PipelineAction`] pairs. Each [`LossVector`]
//! decomposes cost into quality, latency, and resource dimensions so
//! that consumers can weight them according to their SLO profile.
//!
//! # Evidence Ledger
//!
//! Every adaptive component emits [`EvidenceRecord`] entries using
//! machine-stable [`ReasonCode`] values. These records are compatible
//! with the JSONL envelope schema (`schemas/evidence-jsonl-v1.schema.json`).
//!
//! # Budgeted Mode
//!
//! [`ResourceBudget`] caps control how much compute the pipeline may
//! spend on quality refinement within a single query. When a budget
//! is exhausted, [`ExhaustionPolicy`] determines whether to degrade
//! gracefully, circuit-break, or fail open.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::query_class::QueryClass;

// ─── Pipeline State Machine ──────────────────────────────────────────────────

/// Observable state of the search pipeline at decision time.
///
/// The decision plane evaluates actions relative to the current state.
/// State transitions follow the diagram:
///
/// ```text
///   Nominal ──(quality slow/error)──> DegradedQuality
///   Nominal ──(circuit trip)────────> CircuitOpen
///   DegradedQuality ──(recovery)───> Nominal
///   DegradedQuality ──(threshold)──> CircuitOpen
///   CircuitOpen ──(half-open probe)─> Probing
///   Probing ──(probe succeeds)──────> Nominal
///   Probing ──(probe fails)─────────> CircuitOpen
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineState {
    /// Both fast and quality tiers are healthy and operating within SLO.
    Nominal,
    /// Quality tier is responding but outside SLO (high latency or elevated errors).
    DegradedQuality,
    /// Quality tier circuit breaker is open; only fast tier is serving.
    CircuitOpen,
    /// Circuit breaker is half-open; a single probe query is testing quality tier.
    Probing,
}

impl fmt::Display for PipelineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nominal => write!(f, "nominal"),
            Self::DegradedQuality => write!(f, "degraded_quality"),
            Self::CircuitOpen => write!(f, "circuit_open"),
            Self::Probing => write!(f, "probing"),
        }
    }
}

/// Actions the decision plane may select in response to a query.
///
/// Each action implies a different cost profile (latency, compute, quality).
/// The expected-loss model evaluates all feasible actions for the current
/// [`PipelineState`] and selects the minimum-loss option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineAction {
    /// Proceed with full two-tier refinement (fast + quality + blend).
    Refine,
    /// Return fast-tier results without quality refinement.
    SkipRefinement,
    /// Trip the quality-tier circuit breaker.
    OpenCircuit,
    /// Reset the circuit breaker after successful probe.
    CloseCircuit,
    /// Send a single probe query through the quality tier to test recovery.
    ProbeQuality,
    /// Adjust the blend weight between fast and quality tiers.
    AdjustBlend {
        /// New quality weight (0.0-1.0).
        quality_weight: u8, // Stored as percentage 0-100 for Eq/Hash.
    },
}

impl fmt::Display for PipelineAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refine => write!(f, "refine"),
            Self::SkipRefinement => write!(f, "skip_refinement"),
            Self::OpenCircuit => write!(f, "open_circuit"),
            Self::CloseCircuit => write!(f, "close_circuit"),
            Self::ProbeQuality => write!(f, "probe_quality"),
            Self::AdjustBlend { quality_weight } => {
                write!(f, "adjust_blend({quality_weight}%)")
            }
        }
    }
}

// ─── Expected-Loss Model ─────────────────────────────────────────────────────

/// Three-dimensional loss vector for decision evaluation.
///
/// Loss is decomposed into orthogonal dimensions so consumers can
/// weight them according to their SLO profile. All values are
/// non-negative; lower is better.
///
/// The total weighted loss is computed as:
/// ```text
/// total = w_quality * quality + w_latency * latency + w_resource * resource
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossVector {
    /// Relevance quality loss (0.0 = perfect, 1.0 = worst).
    ///
    /// Estimated from the expected NDCG/MRR degradation of choosing
    /// this action versus the oracle-best action.
    pub quality: f64,

    /// Latency cost (0.0 = instant, 1.0 = at SLO budget).
    ///
    /// Normalized to the configured `quality_timeout_ms` so that
    /// a value of 1.0 means "at the edge of the timeout".
    pub latency: f64,

    /// Resource/compute cost (0.0 = free, 1.0 = at budget cap).
    ///
    /// Tracks embedding calls, reranker invocations, and CPU time
    /// against [`ResourceBudget`] caps.
    pub resource: f64,
}

impl LossVector {
    /// A zero-loss vector (oracle-optimal action).
    pub const ZERO: Self = Self {
        quality: 0.0,
        latency: 0.0,
        resource: 0.0,
    };

    /// Compute weighted scalar loss.
    ///
    /// Weights should sum to 1.0 but this is not enforced.
    #[must_use]
    pub fn weighted_total(&self, w_quality: f64, w_latency: f64, w_resource: f64) -> f64 {
        // Any non-finite component or weight is treated as an invalid action
        // and mapped to worst-case loss so it is never silently preferred.
        if !self.quality.is_finite()
            || !self.latency.is_finite()
            || !self.resource.is_finite()
            || !w_quality.is_finite()
            || !w_latency.is_finite()
            || !w_resource.is_finite()
        {
            return f64::MAX;
        }

        let quality = self.quality.max(0.0);
        let latency = self.latency.max(0.0);
        let resource = self.resource.max(0.0);
        let w_quality = w_quality.max(0.0);
        let w_latency = w_latency.max(0.0);
        let w_resource = w_resource.max(0.0);

        let total = quality.mul_add(w_quality, latency.mul_add(w_latency, resource * w_resource));
        if total.is_finite() { total } else { f64::MAX }
    }
}

/// Weights for the three loss dimensions.
///
/// Consumers set these based on their SLO profile:
/// - **Quality-first** (e.g., research): high `quality`, low `latency`.
/// - **Latency-first** (e.g., autocomplete): low `quality`, high `latency`.
/// - **Balanced** (default): equal weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossWeights {
    /// Weight for quality dimension.
    pub quality: f64,
    /// Weight for latency dimension.
    pub latency: f64,
    /// Weight for resource dimension.
    pub resource: f64,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            quality: 0.5,
            latency: 0.3,
            resource: 0.2,
        }
    }
}

impl LossWeights {
    /// Latency-first profile (autocomplete, instant search).
    pub const LATENCY_FIRST: Self = Self {
        quality: 0.2,
        latency: 0.6,
        resource: 0.2,
    };

    /// Quality-first profile (research, deep search).
    pub const QUALITY_FIRST: Self = Self {
        quality: 0.7,
        latency: 0.1,
        resource: 0.2,
    };

    /// Compute weighted total loss from a [`LossVector`].
    #[must_use]
    pub fn apply(&self, loss: &LossVector) -> f64 {
        loss.weighted_total(self.quality, self.latency, self.resource)
    }
}

/// Outcome of a decision plane evaluation.
///
/// Produced by the decision plane after evaluating all feasible actions
/// for the current state. Components should log this as an [`EvidenceRecord`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// The state at decision time.
    pub state: PipelineState,
    /// The action selected by the decision plane.
    pub action: PipelineAction,
    /// Expected loss of the chosen action.
    pub expected_loss: LossVector,
    /// Machine-stable reason code explaining why this action was chosen.
    pub reason: ReasonCode,
    /// Query classification that influenced the decision.
    pub query_class: Option<QueryClass>,
}

// ─── Calibration ─────────────────────────────────────────────────────────────

/// Status of a score calibration model.
///
/// Score calibrators (Platt scaling, isotonic regression, temperature scaling)
/// must be trained on observed data before they can be used. This enum tracks
/// the lifecycle of a calibrator so that consumers know when to fall back
/// to uncalibrated scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationStatus {
    /// No calibration model has been trained yet.
    /// Fall back to raw scores.
    Uncalibrated,

    /// Calibration is in progress (collecting observations).
    Calibrating {
        /// Number of observations collected so far.
        observations: usize,
        /// Minimum observations required before calibration is usable.
        target: usize,
    },

    /// Calibration model is trained and active.
    Calibrated {
        /// Expected calibration error (ECE) of the current model.
        /// Lower is better; typical threshold for "good" is < 0.05.
        ece: f64,
        /// Number of observations the model was trained on.
        observations: usize,
    },

    /// Calibration model exists but is stale (input distribution has shifted).
    /// Fall back to raw scores until recalibrated.
    Stale {
        /// Machine-stable reason for staleness.
        reason: CalibrationFallbackReason,
        /// Number of observations since the model was trained.
        observations_since_train: usize,
    },
}

impl CalibrationStatus {
    /// Whether calibrated scores should be used (vs. falling back to raw).
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        matches!(self, Self::Calibrated { .. })
    }
}

/// Why a calibration model cannot be used or has been invalidated.
///
/// These reasons map to evidence ledger reason codes under the
/// `calibration.fallback.*` namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibrationFallbackReason {
    /// Not enough observations to train a calibrator.
    InsufficientData,
    /// Input score distribution has shifted beyond the drift threshold.
    DistributionShift,
    /// Calibration error (ECE) exceeds the acceptable threshold.
    ErrorTooHigh,
    /// The underlying model (embedder or reranker) has changed.
    ModelChanged,
    /// Calibration was explicitly reset by the operator.
    ManualReset,
}

impl fmt::Display for CalibrationFallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "insufficient_data"),
            Self::DistributionShift => write!(f, "distribution_shift"),
            Self::ErrorTooHigh => write!(f, "error_too_high"),
            Self::ModelChanged => write!(f, "model_changed"),
            Self::ManualReset => write!(f, "manual_reset"),
        }
    }
}

/// Configuration for calibration fallback triggers.
///
/// When any trigger fires, the calibrator transitions to
/// [`CalibrationStatus::Stale`] and consumers fall back to raw scores.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationThresholds {
    /// Minimum observations before calibration is considered usable.
    /// Default: 100.
    pub min_observations: usize,

    /// Maximum expected calibration error (ECE) before fallback.
    /// Default: 0.05.
    pub max_ece: f64,

    /// KL divergence threshold for detecting distribution shift.
    /// Default: 0.1.
    pub drift_kl_threshold: f64,

    /// Number of queries between periodic recalibration checks.
    /// Default: 1000.
    pub recalibration_interval: usize,
}

impl Default for CalibrationThresholds {
    fn default() -> Self {
        Self {
            min_observations: 100,
            max_ece: 0.05,
            drift_kl_threshold: 0.1,
            recalibration_interval: 1000,
        }
    }
}

// ─── Evidence Ledger ─────────────────────────────────────────────────────────

/// Machine-stable reason code for evidence records.
///
/// Format: `namespace.subject.detail` (matches the JSONL schema pattern
/// `^[a-z0-9]+\.[a-z0-9_]+\.[a-z0-9_]+$`).
///
/// Constants are provided for all reason codes used by the decision plane
/// and its consumers. New codes should be added here (not invented ad-hoc)
/// to maintain cross-bead consistency.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReasonCode(pub String);

impl ReasonCode {
    // ── Decision Plane ───────────────────────────────────────────────
    /// Quality refinement was skipped because the pipeline is in fast-only mode.
    pub const DECISION_SKIP_FAST_ONLY: &str = "decision.skip.fast_only";
    /// Quality refinement was skipped because the circuit breaker is open.
    pub const DECISION_SKIP_CIRCUIT_OPEN: &str = "decision.skip.circuit_open";
    /// Quality refinement was skipped because the resource budget is exhausted.
    pub const DECISION_SKIP_BUDGET_EXHAUSTED: &str = "decision.skip.budget_exhausted";
    /// Quality refinement was skipped because expected loss is too high.
    pub const DECISION_SKIP_HIGH_LOSS: &str = "decision.skip.high_loss";
    /// Quality refinement was skipped for an empty query.
    pub const DECISION_SKIP_EMPTY_QUERY: &str = "decision.skip.empty_query";
    /// Quality refinement proceeded normally.
    pub const DECISION_REFINE_NOMINAL: &str = "decision.refine.nominal";
    /// A probe query was sent to test quality tier recovery.
    pub const DECISION_PROBE_SENT: &str = "decision.probe.sent";
    /// Probe succeeded; circuit breaker will close.
    pub const DECISION_PROBE_SUCCESS: &str = "decision.probe.success";
    /// Probe failed; circuit breaker remains open.
    pub const DECISION_PROBE_FAILURE: &str = "decision.probe.failure";

    // ── Circuit Breaker ──────────────────────────────────────────────
    /// Circuit breaker opened due to consecutive failures.
    pub const CIRCUIT_OPEN_FAILURES: &str = "circuit.open.consecutive_failures";
    /// Circuit breaker opened due to sustained high latency.
    pub const CIRCUIT_OPEN_LATENCY: &str = "circuit.open.sustained_latency";
    /// Circuit breaker closed after successful probe.
    pub const CIRCUIT_CLOSE_RECOVERY: &str = "circuit.close.recovery";

    // ── Calibration ──────────────────────────────────────────────────
    /// Calibration fell back due to insufficient data.
    pub const CALIBRATION_FALLBACK_DATA: &str = "calibration.fallback.insufficient_data";
    /// Calibration fell back due to distribution shift.
    pub const CALIBRATION_FALLBACK_DRIFT: &str = "calibration.fallback.distribution_shift";
    /// Calibration fell back due to high calibration error.
    pub const CALIBRATION_FALLBACK_ERROR: &str = "calibration.fallback.error_too_high";
    /// Calibration fell back because the model changed.
    pub const CALIBRATION_FALLBACK_MODEL: &str = "calibration.fallback.model_changed";
    /// Calibration was completed successfully.
    pub const CALIBRATION_TRAINED: &str = "calibration.lifecycle.trained";
    /// Calibration model was reset.
    pub const CALIBRATION_RESET: &str = "calibration.lifecycle.reset";

    // ── Adaptive Fusion ──────────────────────────────────────────────
    /// Blend weight was adjusted by Bayesian update.
    pub const FUSION_BLEND_ADJUSTED: &str = "fusion.blend.adjusted";
    /// RRF K parameter was adjusted by Bayesian update.
    pub const FUSION_RRF_K_ADJUSTED: &str = "fusion.rrf_k.adjusted";
    /// Adaptive fusion fell back to defaults (insufficient evidence).
    pub const FUSION_FALLBACK_DEFAULT: &str = "fusion.fallback.default";

    // ── Sequential Testing ───────────────────────────────────────────
    /// E-process rejected the null hypothesis (phase transition warranted).
    pub const TESTING_REJECT: &str = "testing.gate.rejected";
    /// E-process did not reject (stay in current phase).
    pub const TESTING_CONTINUE: &str = "testing.gate.continue";
    /// E-process evidence was reset (start of new evaluation window).
    pub const TESTING_RESET: &str = "testing.gate.reset";

    // ── Conformal Prediction ─────────────────────────────────────────
    /// Conformal prediction set was constructed with valid coverage.
    pub const CONFORMAL_VALID: &str = "conformal.coverage.valid";
    /// Coverage guarantee was violated (empirical coverage below target).
    pub const CONFORMAL_VIOLATION: &str = "conformal.coverage.violation";
    /// Conformal calibration was updated with new observations.
    pub const CONFORMAL_UPDATE: &str = "conformal.calibration.updated";

    // ── Relevance Feedback ───────────────────────────────────────────
    /// Boost map was updated from implicit feedback.
    pub const FEEDBACK_BOOST_UPDATED: &str = "feedback.boost.updated";
    /// Boost map entry decayed below threshold and was removed.
    pub const FEEDBACK_BOOST_DECAYED: &str = "feedback.boost.decayed";

    /// Create a new reason code from a string.
    ///
    /// The caller is responsible for ensuring the code matches the
    /// `namespace.subject.detail` pattern.
    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into())
    }

    /// The reason code string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validate that this reason code matches the canonical pattern.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let parts: Vec<&str> = self.0.split('.').collect();
        if parts.len() != 3 {
            return false;
        }
        parts.iter().all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        })
    }
}

impl fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ReasonCode {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Severity level for evidence records.
///
/// Matches the `severity` field in the JSONL evidence schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Informational: normal operations, state transitions.
    Info,
    /// Warning: degraded behavior, fallback activated.
    Warn,
    /// Error: component failure, data loss risk.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warn => write!(f, "warn"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// Event types for evidence records.
///
/// Matches the `type` field in the JSONL evidence schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceEventType {
    /// An adaptive component made a decision.
    Decision,
    /// An alert condition was detected.
    Alert,
    /// Pipeline entered a degraded state.
    Degradation,
    /// A state transition occurred.
    Transition,
    /// Marker for deterministic replay synchronization.
    ReplayMarker,
}

impl fmt::Display for EvidenceEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decision => write!(f, "decision"),
            Self::Alert => write!(f, "alert"),
            Self::Degradation => write!(f, "degradation"),
            Self::Transition => write!(f, "transition"),
            Self::ReplayMarker => write!(f, "replay_marker"),
        }
    }
}

/// A structured evidence record emitted by adaptive ranking components.
///
/// This is the Rust-side representation of an evidence JSONL event payload.
/// Components call [`EvidenceRecord::new`] to create records and emit them
/// via the [`MetricsExporter`](crate::traits::MetricsExporter) or tracing.
///
/// # Integration Requirement
///
/// All adaptive beads (bd-21g, bd-22k, bd-2ps, bd-2yj, bd-1do, bd-2tv)
/// MUST emit evidence records for:
/// - Every state transition.
/// - Every fallback trigger.
/// - Every parameter adjustment.
///
/// This enables offline replay and cross-bead consistency auditing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRecord {
    /// Event classification.
    pub event_type: EvidenceEventType,
    /// Machine-stable reason code.
    pub reason_code: ReasonCode,
    /// Human-readable explanation.
    pub reason_human: String,
    /// Severity level.
    pub severity: Severity,
    /// Pipeline state at the time of the event.
    pub pipeline_state: PipelineState,
    /// Action taken (if applicable).
    pub action: Option<PipelineAction>,
    /// Expected loss of the action (if applicable).
    pub expected_loss: Option<LossVector>,
    /// Query classification that influenced the decision.
    pub query_class: Option<QueryClass>,
    /// Component that emitted this record.
    pub source_component: String,
}

impl EvidenceRecord {
    /// Create a new evidence record.
    #[must_use]
    pub fn new(
        event_type: EvidenceEventType,
        reason_code: impl Into<ReasonCode>,
        reason_human: impl Into<String>,
        severity: Severity,
        pipeline_state: PipelineState,
        source_component: impl Into<String>,
    ) -> Self {
        Self {
            event_type,
            reason_code: reason_code.into(),
            reason_human: reason_human.into(),
            severity,
            pipeline_state,
            action: None,
            expected_loss: None,
            query_class: None,
            source_component: source_component.into(),
        }
    }

    /// Attach the action taken.
    #[must_use]
    pub const fn with_action(mut self, action: PipelineAction) -> Self {
        self.action = Some(action);
        self
    }

    /// Attach the expected loss.
    #[must_use]
    pub const fn with_expected_loss(mut self, loss: LossVector) -> Self {
        self.expected_loss = Some(loss);
        self
    }

    /// Attach query classification context.
    #[must_use]
    pub const fn with_query_class(mut self, qc: QueryClass) -> Self {
        self.query_class = Some(qc);
        self
    }
}

// ─── Budgeted Mode ───────────────────────────────────────────────────────────

/// Per-query resource budget for the search pipeline.
///
/// When any cap is reached, the pipeline consults [`ExhaustionPolicy`]
/// to decide how to proceed. A `None` value means unlimited for that
/// resource dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Maximum embedding calls (fast + quality) per query.
    pub max_embed_calls: Option<usize>,
    /// Maximum reranker invocations per query.
    pub max_rerank_calls: Option<usize>,
    /// Maximum milliseconds for phase 2 (quality refinement).
    pub max_phase2_ms: Option<u64>,
    /// Maximum total search latency in milliseconds.
    pub max_total_ms: Option<u64>,
}

impl ResourceBudget {
    /// An unlimited budget (no caps).
    pub const UNLIMITED: Self = Self {
        max_embed_calls: None,
        max_rerank_calls: None,
        max_phase2_ms: None,
        max_total_ms: None,
    };
}

/// Tracks consumed resources against a [`ResourceBudget`].
///
/// Updated incrementally during query processing. Check [`is_exhausted`](Self::is_exhausted)
/// before each expensive operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Number of embedding calls made so far.
    pub embed_calls: usize,
    /// Number of reranker invocations made so far.
    pub rerank_calls: usize,
    /// Milliseconds spent in phase 2 so far.
    pub phase2_ms: u64,
    /// Total milliseconds elapsed so far.
    pub total_ms: u64,
}

impl ResourceUsage {
    /// Check whether any budget cap has been reached.
    #[must_use]
    pub const fn is_exhausted(&self, budget: &ResourceBudget) -> bool {
        if let Some(max) = budget.max_embed_calls
            && self.embed_calls >= max
        {
            return true;
        }
        if let Some(max) = budget.max_rerank_calls
            && self.rerank_calls >= max
        {
            return true;
        }
        if let Some(max) = budget.max_phase2_ms
            && self.phase2_ms >= max
        {
            return true;
        }
        if let Some(max) = budget.max_total_ms
            && self.total_ms >= max
        {
            return true;
        }
        false
    }

    /// Which budget dimension was exhausted first, if any.
    #[must_use]
    pub fn exhausted_dimension(&self, budget: &ResourceBudget) -> Option<&'static str> {
        if budget
            .max_embed_calls
            .is_some_and(|max| self.embed_calls >= max)
        {
            return Some("embed_calls");
        }
        if budget
            .max_rerank_calls
            .is_some_and(|max| self.rerank_calls >= max)
        {
            return Some("rerank_calls");
        }
        if budget
            .max_phase2_ms
            .is_some_and(|max| self.phase2_ms >= max)
        {
            return Some("phase2_ms");
        }
        if budget.max_total_ms.is_some_and(|max| self.total_ms >= max) {
            return Some("total_ms");
        }
        None
    }
}

/// What to do when a [`ResourceBudget`] cap is reached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExhaustionPolicy {
    /// Skip the remaining expensive phase and return current results.
    /// This is the gentlest option: no state machine transitions.
    #[default]
    Degrade,
    /// Trip the circuit breaker for subsequent queries.
    /// Use when budget exhaustion signals a systemic issue.
    CircuitBreak,
    /// Return whatever results are available immediately.
    /// Equivalent to `Degrade` but also skips any in-progress work.
    FailOpen,
}

impl fmt::Display for ExhaustionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Degrade => write!(f, "degrade"),
            Self::CircuitBreak => write!(f, "circuit_break"),
            Self::FailOpen => write!(f, "fail_open"),
        }
    }
}

// ─── Decision Context ────────────────────────────────────────────────────────

/// Input context provided to the decision plane for each query.
///
/// Aggregates all signals needed for the expected-loss evaluation.
/// Consumers construct this from their runtime state before calling
/// the decision evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    /// Current pipeline state.
    pub state: PipelineState,
    /// Calibration status of the active score calibrator.
    pub calibration: CalibrationStatus,
    /// Per-query resource budget.
    pub budget: ResourceBudget,
    /// Resources consumed so far in this query.
    pub usage: ResourceUsage,
    /// Query classification.
    pub query_class: QueryClass,
    /// Loss weights for the current consumer profile.
    pub loss_weights: LossWeights,
    /// Rolling average latency of recent quality-tier operations (ms).
    /// Used to estimate the latency component of the loss vector.
    pub recent_quality_latency_ms: f64,
    /// Number of consecutive quality-tier failures (for circuit breaker).
    pub consecutive_failures: usize,
}

// ─── Integration Criteria ────────────────────────────────────────────────────

/// Cross-bead consistency requirements that MUST hold at all times.
///
/// These are not runtime-checked (that would be too expensive) but are
/// tested via the integration test suite (bd-3un.32) and the evidence
/// replay validator.
///
/// 1. **Single source of truth**: All adaptive parameters flow through
///    this decision plane contract. No component may invent its own
///    state/action enums.
///
/// 2. **Evidence completeness**: Every state transition emits an
///    [`EvidenceRecord`]. Missing records are flagged by the replay
///    validator.
///
/// 3. **Monotonic reason codes**: New reason codes are added to
///    [`ReasonCode`] constants. Ad-hoc string codes are forbidden.
///
/// 4. **Budget monotonicity**: [`ResourceUsage`] counters only increase
///    within a single query. Resetting mid-query is a contract violation.
///
/// 5. **Calibration lifecycle**: Transitions between [`CalibrationStatus`]
///    variants must follow:
///    `Uncalibrated -> Calibrating -> Calibrated -> Stale -> Calibrating`
///    (loops allowed, skipping not allowed).
///
/// 6. **Circuit breaker protocol**: Only [`PipelineAction::ProbeQuality`]
///    may transition from `CircuitOpen` to `Probing`. Direct jumps from
///    `CircuitOpen` to `Nominal` are a contract violation.
///
/// This struct intentionally has no fields; it exists only to anchor
/// the doc comment above in `rustdoc`.
pub struct IntegrationCriteria;

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PipelineState ───────────────────────────────────────────────

    #[test]
    fn pipeline_state_display() {
        assert_eq!(PipelineState::Nominal.to_string(), "nominal");
        assert_eq!(
            PipelineState::DegradedQuality.to_string(),
            "degraded_quality"
        );
        assert_eq!(PipelineState::CircuitOpen.to_string(), "circuit_open");
        assert_eq!(PipelineState::Probing.to_string(), "probing");
    }

    #[test]
    fn pipeline_state_serde_roundtrip() {
        for state in [
            PipelineState::Nominal,
            PipelineState::DegradedQuality,
            PipelineState::CircuitOpen,
            PipelineState::Probing,
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let decoded: PipelineState = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, state);
        }
    }

    // ─── PipelineAction ──────────────────────────────────────────────

    #[test]
    fn pipeline_action_display() {
        assert_eq!(PipelineAction::Refine.to_string(), "refine");
        assert_eq!(
            PipelineAction::SkipRefinement.to_string(),
            "skip_refinement"
        );
        assert_eq!(PipelineAction::OpenCircuit.to_string(), "open_circuit");
        assert_eq!(PipelineAction::CloseCircuit.to_string(), "close_circuit");
        assert_eq!(PipelineAction::ProbeQuality.to_string(), "probe_quality");
        assert_eq!(
            PipelineAction::AdjustBlend { quality_weight: 70 }.to_string(),
            "adjust_blend(70%)"
        );
    }

    #[test]
    fn pipeline_action_serde_roundtrip() {
        let actions = [
            PipelineAction::Refine,
            PipelineAction::SkipRefinement,
            PipelineAction::OpenCircuit,
            PipelineAction::CloseCircuit,
            PipelineAction::ProbeQuality,
            PipelineAction::AdjustBlend { quality_weight: 80 },
        ];
        for action in actions {
            let json = serde_json::to_string(&action).unwrap();
            let decoded: PipelineAction = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, action);
        }
    }

    // ─── LossVector ──────────────────────────────────────────────────

    #[test]
    fn loss_vector_zero() {
        let zero = LossVector::ZERO;
        assert!(zero.quality.abs() < f64::EPSILON);
        assert!(zero.latency.abs() < f64::EPSILON);
        assert!(zero.resource.abs() < f64::EPSILON);
    }

    #[test]
    fn loss_vector_weighted_total() {
        let loss = LossVector {
            quality: 0.5,
            latency: 0.3,
            resource: 0.2,
        };
        let total = loss.weighted_total(1.0, 1.0, 1.0);
        assert!((total - 1.0).abs() < 1e-10);

        // Quality-only weight
        let q_only = loss.weighted_total(1.0, 0.0, 0.0);
        assert!((q_only - 0.5).abs() < 1e-10);
    }

    #[test]
    fn loss_weights_default() {
        let w = LossWeights::default();
        assert!((w.quality - 0.5).abs() < 1e-10);
        assert!((w.latency - 0.3).abs() < 1e-10);
        assert!((w.resource - 0.2).abs() < 1e-10);
    }

    #[test]
    fn loss_weights_apply() {
        let loss = LossVector {
            quality: 1.0,
            latency: 0.0,
            resource: 0.0,
        };
        let w = LossWeights::QUALITY_FIRST;
        let total = w.apply(&loss);
        assert!((total - 0.7).abs() < 1e-10);
    }

    #[test]
    fn loss_weights_serde_roundtrip() {
        let w = LossWeights::LATENCY_FIRST;
        let json = serde_json::to_string(&w).unwrap();
        let decoded: LossWeights = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, w);
    }

    // ─── CalibrationStatus ───────────────────────────────────────────

    #[test]
    fn calibration_status_usable() {
        assert!(!CalibrationStatus::Uncalibrated.is_usable());
        assert!(
            !CalibrationStatus::Calibrating {
                observations: 50,
                target: 100
            }
            .is_usable()
        );
        assert!(
            CalibrationStatus::Calibrated {
                ece: 0.03,
                observations: 200
            }
            .is_usable()
        );
        assert!(
            !CalibrationStatus::Stale {
                reason: CalibrationFallbackReason::DistributionShift,
                observations_since_train: 5000,
            }
            .is_usable()
        );
    }

    #[test]
    fn calibration_status_serde_roundtrip() {
        let statuses = [
            CalibrationStatus::Uncalibrated,
            CalibrationStatus::Calibrating {
                observations: 42,
                target: 100,
            },
            CalibrationStatus::Calibrated {
                ece: 0.02,
                observations: 500,
            },
            CalibrationStatus::Stale {
                reason: CalibrationFallbackReason::ErrorTooHigh,
                observations_since_train: 3000,
            },
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let decoded: CalibrationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, status);
        }
    }

    #[test]
    fn calibration_fallback_reason_display() {
        assert_eq!(
            CalibrationFallbackReason::InsufficientData.to_string(),
            "insufficient_data"
        );
        assert_eq!(
            CalibrationFallbackReason::DistributionShift.to_string(),
            "distribution_shift"
        );
        assert_eq!(
            CalibrationFallbackReason::ErrorTooHigh.to_string(),
            "error_too_high"
        );
        assert_eq!(
            CalibrationFallbackReason::ModelChanged.to_string(),
            "model_changed"
        );
        assert_eq!(
            CalibrationFallbackReason::ManualReset.to_string(),
            "manual_reset"
        );
    }

    // ─── CalibrationThresholds ───────────────────────────────────────

    #[test]
    fn calibration_thresholds_default() {
        let t = CalibrationThresholds::default();
        assert_eq!(t.min_observations, 100);
        assert!((t.max_ece - 0.05).abs() < 1e-10);
        assert!((t.drift_kl_threshold - 0.1).abs() < 1e-10);
        assert_eq!(t.recalibration_interval, 1000);
    }

    #[test]
    fn calibration_thresholds_serde_roundtrip() {
        let t = CalibrationThresholds {
            min_observations: 50,
            max_ece: 0.1,
            drift_kl_threshold: 0.2,
            recalibration_interval: 500,
        };
        let json = serde_json::to_string(&t).unwrap();
        let decoded: CalibrationThresholds = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, t);
    }

    // ─── ReasonCode ──────────────────────────────────────────────────

    #[test]
    fn reason_code_validation() {
        assert!(ReasonCode::new("decision.skip.fast_only").is_valid());
        assert!(ReasonCode::new("calibration.fallback.insufficient_data").is_valid());
        assert!(ReasonCode::new("circuit.open.consecutive_failures").is_valid());

        // Invalid: too few parts
        assert!(!ReasonCode::new("decision.skip").is_valid());
        // Invalid: too many parts
        assert!(!ReasonCode::new("a.b.c.d").is_valid());
        // Invalid: uppercase
        assert!(!ReasonCode::new("Decision.skip.fast_only").is_valid());
        // Invalid: empty part
        assert!(!ReasonCode::new("decision..fast_only").is_valid());
    }

    #[test]
    fn all_reason_code_constants_are_valid() {
        let codes = [
            ReasonCode::DECISION_SKIP_FAST_ONLY,
            ReasonCode::DECISION_SKIP_CIRCUIT_OPEN,
            ReasonCode::DECISION_SKIP_BUDGET_EXHAUSTED,
            ReasonCode::DECISION_SKIP_HIGH_LOSS,
            ReasonCode::DECISION_SKIP_EMPTY_QUERY,
            ReasonCode::DECISION_REFINE_NOMINAL,
            ReasonCode::DECISION_PROBE_SENT,
            ReasonCode::DECISION_PROBE_SUCCESS,
            ReasonCode::DECISION_PROBE_FAILURE,
            ReasonCode::CIRCUIT_OPEN_FAILURES,
            ReasonCode::CIRCUIT_OPEN_LATENCY,
            ReasonCode::CIRCUIT_CLOSE_RECOVERY,
            ReasonCode::CALIBRATION_FALLBACK_DATA,
            ReasonCode::CALIBRATION_FALLBACK_DRIFT,
            ReasonCode::CALIBRATION_FALLBACK_ERROR,
            ReasonCode::CALIBRATION_FALLBACK_MODEL,
            ReasonCode::CALIBRATION_TRAINED,
            ReasonCode::CALIBRATION_RESET,
            ReasonCode::FUSION_BLEND_ADJUSTED,
            ReasonCode::FUSION_RRF_K_ADJUSTED,
            ReasonCode::FUSION_FALLBACK_DEFAULT,
            ReasonCode::TESTING_REJECT,
            ReasonCode::TESTING_CONTINUE,
            ReasonCode::TESTING_RESET,
            ReasonCode::CONFORMAL_VALID,
            ReasonCode::CONFORMAL_VIOLATION,
            ReasonCode::CONFORMAL_UPDATE,
            ReasonCode::FEEDBACK_BOOST_UPDATED,
            ReasonCode::FEEDBACK_BOOST_DECAYED,
        ];
        for code_str in codes {
            let code = ReasonCode::new(code_str);
            assert!(code.is_valid(), "invalid reason code: {code_str}");
        }
    }

    #[test]
    fn reason_code_display() {
        let code = ReasonCode::new("decision.skip.fast_only");
        assert_eq!(code.to_string(), "decision.skip.fast_only");
    }

    #[test]
    fn reason_code_from_str() {
        let code: ReasonCode = "circuit.open.latency".into();
        assert_eq!(code.as_str(), "circuit.open.latency");
    }

    #[test]
    fn reason_code_serde_roundtrip() {
        let code = ReasonCode::new("fusion.blend.adjusted");
        let json = serde_json::to_string(&code).unwrap();
        let decoded: ReasonCode = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, code);
    }

    // ─── Severity ────────────────────────────────────────────────────

    #[test]
    fn severity_display() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warn.to_string(), "warn");
        assert_eq!(Severity::Error.to_string(), "error");
    }

    #[test]
    fn severity_serde_roundtrip() {
        for sev in [Severity::Info, Severity::Warn, Severity::Error] {
            let json = serde_json::to_string(&sev).unwrap();
            let decoded: Severity = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, sev);
        }
    }

    // ─── EvidenceEventType ───────────────────────────────────────────

    #[test]
    fn evidence_event_type_display() {
        assert_eq!(EvidenceEventType::Decision.to_string(), "decision");
        assert_eq!(EvidenceEventType::Alert.to_string(), "alert");
        assert_eq!(EvidenceEventType::Degradation.to_string(), "degradation");
        assert_eq!(EvidenceEventType::Transition.to_string(), "transition");
        assert_eq!(EvidenceEventType::ReplayMarker.to_string(), "replay_marker");
    }

    #[test]
    fn evidence_event_type_serde_roundtrip() {
        for evt in [
            EvidenceEventType::Decision,
            EvidenceEventType::Alert,
            EvidenceEventType::Degradation,
            EvidenceEventType::Transition,
            EvidenceEventType::ReplayMarker,
        ] {
            let json = serde_json::to_string(&evt).unwrap();
            let decoded: EvidenceEventType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, evt);
        }
    }

    // ─── EvidenceRecord ──────────────────────────────────────────────

    #[test]
    fn evidence_record_builder() {
        let record = EvidenceRecord::new(
            EvidenceEventType::Decision,
            ReasonCode::DECISION_REFINE_NOMINAL,
            "Proceeding with quality refinement",
            Severity::Info,
            PipelineState::Nominal,
            "two_tier_searcher",
        )
        .with_action(PipelineAction::Refine)
        .with_expected_loss(LossVector {
            quality: 0.0,
            latency: 0.3,
            resource: 0.2,
        })
        .with_query_class(QueryClass::NaturalLanguage);

        assert_eq!(record.event_type, EvidenceEventType::Decision);
        assert_eq!(record.reason_code.as_str(), "decision.refine.nominal");
        assert_eq!(record.severity, Severity::Info);
        assert_eq!(record.pipeline_state, PipelineState::Nominal);
        assert_eq!(record.action, Some(PipelineAction::Refine));
        assert!(record.expected_loss.is_some());
        assert_eq!(record.query_class, Some(QueryClass::NaturalLanguage));
        assert_eq!(record.source_component, "two_tier_searcher");
    }

    #[test]
    fn evidence_record_serde_roundtrip() {
        let record = EvidenceRecord::new(
            EvidenceEventType::Transition,
            ReasonCode::CIRCUIT_OPEN_FAILURES,
            "Quality tier failed 5 consecutive times",
            Severity::Warn,
            PipelineState::CircuitOpen,
            "circuit_breaker",
        )
        .with_action(PipelineAction::OpenCircuit);

        let json = serde_json::to_string(&record).unwrap();
        let decoded: EvidenceRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type, EvidenceEventType::Transition);
        assert_eq!(
            decoded.reason_code.as_str(),
            "circuit.open.consecutive_failures"
        );
        assert_eq!(decoded.severity, Severity::Warn);
    }

    // ─── ResourceBudget ──────────────────────────────────────────────

    #[test]
    fn resource_budget_unlimited() {
        let b = ResourceBudget::UNLIMITED;
        assert!(b.max_embed_calls.is_none());
        assert!(b.max_rerank_calls.is_none());
        assert!(b.max_phase2_ms.is_none());
        assert!(b.max_total_ms.is_none());
    }

    #[test]
    fn resource_budget_default_is_unlimited() {
        assert_eq!(ResourceBudget::default(), ResourceBudget::UNLIMITED);
    }

    #[test]
    fn resource_budget_serde_roundtrip() {
        let b = ResourceBudget {
            max_embed_calls: Some(4),
            max_rerank_calls: Some(1),
            max_phase2_ms: Some(300),
            max_total_ms: Some(500),
        };
        let json = serde_json::to_string(&b).unwrap();
        let decoded: ResourceBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, b);
    }

    // ─── ResourceUsage ───────────────────────────────────────────────

    #[test]
    fn resource_usage_not_exhausted_with_unlimited() {
        let usage = ResourceUsage {
            embed_calls: 100,
            rerank_calls: 50,
            phase2_ms: 999,
            total_ms: 9999,
        };
        assert!(!usage.is_exhausted(&ResourceBudget::UNLIMITED));
    }

    #[test]
    fn resource_usage_exhausted_embed_calls() {
        let usage = ResourceUsage {
            embed_calls: 5,
            ..Default::default()
        };
        let budget = ResourceBudget {
            max_embed_calls: Some(5),
            ..Default::default()
        };
        assert!(usage.is_exhausted(&budget));
        assert_eq!(usage.exhausted_dimension(&budget), Some("embed_calls"));
    }

    #[test]
    fn resource_usage_exhausted_rerank_calls() {
        let usage = ResourceUsage {
            rerank_calls: 2,
            ..Default::default()
        };
        let budget = ResourceBudget {
            max_rerank_calls: Some(1),
            ..Default::default()
        };
        assert!(usage.is_exhausted(&budget));
        assert_eq!(usage.exhausted_dimension(&budget), Some("rerank_calls"));
    }

    #[test]
    fn resource_usage_exhausted_phase2_ms() {
        let usage = ResourceUsage {
            phase2_ms: 350,
            ..Default::default()
        };
        let budget = ResourceBudget {
            max_phase2_ms: Some(300),
            ..Default::default()
        };
        assert!(usage.is_exhausted(&budget));
        assert_eq!(usage.exhausted_dimension(&budget), Some("phase2_ms"));
    }

    #[test]
    fn resource_usage_exhausted_total_ms() {
        let usage = ResourceUsage {
            total_ms: 600,
            ..Default::default()
        };
        let budget = ResourceBudget {
            max_total_ms: Some(500),
            ..Default::default()
        };
        assert!(usage.is_exhausted(&budget));
        assert_eq!(usage.exhausted_dimension(&budget), Some("total_ms"));
    }

    #[test]
    fn resource_usage_not_exhausted_below_caps() {
        let usage = ResourceUsage {
            embed_calls: 3,
            rerank_calls: 0,
            phase2_ms: 200,
            total_ms: 400,
        };
        let budget = ResourceBudget {
            max_embed_calls: Some(5),
            max_rerank_calls: Some(1),
            max_phase2_ms: Some(300),
            max_total_ms: Some(500),
        };
        assert!(!usage.is_exhausted(&budget));
        assert!(usage.exhausted_dimension(&budget).is_none());
    }

    // ─── ExhaustionPolicy ────────────────────────────────────────────

    #[test]
    fn exhaustion_policy_default() {
        assert_eq!(ExhaustionPolicy::default(), ExhaustionPolicy::Degrade);
    }

    #[test]
    fn exhaustion_policy_display() {
        assert_eq!(ExhaustionPolicy::Degrade.to_string(), "degrade");
        assert_eq!(ExhaustionPolicy::CircuitBreak.to_string(), "circuit_break");
        assert_eq!(ExhaustionPolicy::FailOpen.to_string(), "fail_open");
    }

    #[test]
    fn exhaustion_policy_serde_roundtrip() {
        for policy in [
            ExhaustionPolicy::Degrade,
            ExhaustionPolicy::CircuitBreak,
            ExhaustionPolicy::FailOpen,
        ] {
            let json = serde_json::to_string(&policy).unwrap();
            let decoded: ExhaustionPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, policy);
        }
    }

    // ─── DecisionOutcome ─────────────────────────────────────────────

    #[test]
    fn decision_outcome_serde_roundtrip() {
        let outcome = DecisionOutcome {
            state: PipelineState::Nominal,
            action: PipelineAction::Refine,
            expected_loss: LossVector {
                quality: 0.0,
                latency: 0.4,
                resource: 0.2,
            },
            reason: ReasonCode::new(ReasonCode::DECISION_REFINE_NOMINAL),
            query_class: Some(QueryClass::NaturalLanguage),
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let decoded: DecisionOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.state, PipelineState::Nominal);
        assert_eq!(decoded.action, PipelineAction::Refine);
        assert_eq!(decoded.query_class, Some(QueryClass::NaturalLanguage));
    }

    // ─── DecisionContext ─────────────────────────────────────────────

    #[test]
    fn decision_context_construction() {
        let ctx = DecisionContext {
            state: PipelineState::DegradedQuality,
            calibration: CalibrationStatus::Calibrated {
                ece: 0.03,
                observations: 500,
            },
            budget: ResourceBudget {
                max_total_ms: Some(500),
                ..Default::default()
            },
            usage: ResourceUsage::default(),
            query_class: QueryClass::ShortKeyword,
            loss_weights: LossWeights::default(),
            recent_quality_latency_ms: 250.0,
            consecutive_failures: 0,
        };
        assert_eq!(ctx.state, PipelineState::DegradedQuality);
        assert!(ctx.calibration.is_usable());
        assert!(!ctx.usage.is_exhausted(&ctx.budget));
    }

    #[test]
    fn decision_context_serde_roundtrip() {
        let ctx = DecisionContext {
            state: PipelineState::Nominal,
            calibration: CalibrationStatus::Uncalibrated,
            budget: ResourceBudget::UNLIMITED,
            usage: ResourceUsage::default(),
            query_class: QueryClass::NaturalLanguage,
            loss_weights: LossWeights::QUALITY_FIRST,
            recent_quality_latency_ms: 128.0,
            consecutive_failures: 0,
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let decoded: DecisionContext = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.state, PipelineState::Nominal);
        assert_eq!(decoded.query_class, QueryClass::NaturalLanguage);
    }

    // ─── bd-2jqx tests begin ───

    #[test]
    fn loss_vector_serde_roundtrip() {
        let loss = LossVector {
            quality: 0.42,
            latency: 0.31,
            resource: 0.19,
        };
        let json = serde_json::to_string(&loss).unwrap();
        let decoded: LossVector = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, loss);
    }

    #[test]
    fn loss_weights_presets_exact_values() {
        let lf = LossWeights::LATENCY_FIRST;
        assert!((lf.quality - 0.2).abs() < 1e-10);
        assert!((lf.latency - 0.6).abs() < 1e-10);
        assert!((lf.resource - 0.2).abs() < 1e-10);

        let qf = LossWeights::QUALITY_FIRST;
        assert!((qf.quality - 0.7).abs() < 1e-10);
        assert!((qf.latency - 0.1).abs() < 1e-10);
        assert!((qf.resource - 0.2).abs() < 1e-10);
    }

    #[test]
    fn resource_usage_default_is_zero() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.embed_calls, 0);
        assert_eq!(usage.rerank_calls, 0);
        assert_eq!(usage.phase2_ms, 0);
        assert_eq!(usage.total_ms, 0);
    }

    #[test]
    fn resource_usage_serde_roundtrip() {
        let usage = ResourceUsage {
            embed_calls: 3,
            rerank_calls: 1,
            phase2_ms: 150,
            total_ms: 320,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let decoded: ResourceUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, usage);
    }

    #[test]
    fn evidence_record_without_optional_fields() {
        let record = EvidenceRecord::new(
            EvidenceEventType::Alert,
            ReasonCode::CIRCUIT_OPEN_LATENCY,
            "High latency detected",
            Severity::Warn,
            PipelineState::DegradedQuality,
            "circuit_breaker",
        );
        assert!(record.action.is_none());
        assert!(record.expected_loss.is_none());
        assert!(record.query_class.is_none());

        // Should still serialize/deserialize cleanly with None fields.
        let json = serde_json::to_string(&record).unwrap();
        let decoded: EvidenceRecord = serde_json::from_str(&json).unwrap();
        assert!(decoded.action.is_none());
        assert!(decoded.expected_loss.is_none());
        assert!(decoded.query_class.is_none());
    }

    #[test]
    fn reason_code_with_numbers_and_underscores() {
        // Numbers in parts are valid.
        assert!(ReasonCode::new("ns1.sub2.detail3").is_valid());
        // Underscores in parts are valid.
        assert!(ReasonCode::new("long_ns.sub_part.detail_code").is_valid());
        // Hyphens are NOT valid.
        assert!(!ReasonCode::new("ns.sub.detail-code").is_valid());
        // Spaces are NOT valid.
        assert!(!ReasonCode::new("ns.sub.detail code").is_valid());
    }

    #[test]
    fn calibration_fallback_reason_serde_roundtrip() {
        let reasons = [
            CalibrationFallbackReason::InsufficientData,
            CalibrationFallbackReason::DistributionShift,
            CalibrationFallbackReason::ErrorTooHigh,
            CalibrationFallbackReason::ModelChanged,
            CalibrationFallbackReason::ManualReset,
        ];
        for reason in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            let decoded: CalibrationFallbackReason = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, reason);
        }
    }

    #[test]
    fn exhausted_dimension_returns_first_by_check_order() {
        let usage = ResourceUsage {
            embed_calls: 10,
            rerank_calls: 5,
            phase2_ms: 400,
            total_ms: 600,
        };
        let budget = ResourceBudget {
            max_embed_calls: Some(5),
            max_rerank_calls: Some(3),
            max_phase2_ms: Some(200),
            max_total_ms: Some(500),
        };
        // All dimensions are exhausted; should return the first checked.
        assert_eq!(usage.exhausted_dimension(&budget), Some("embed_calls"));
    }

    #[test]
    fn pipeline_state_hash_distinct() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelineState::Nominal);
        set.insert(PipelineState::DegradedQuality);
        set.insert(PipelineState::CircuitOpen);
        set.insert(PipelineState::Probing);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn loss_vector_weighted_total_zero_weights() {
        let loss = LossVector {
            quality: 1.0,
            latency: 1.0,
            resource: 1.0,
        };
        let total = loss.weighted_total(0.0, 0.0, 0.0);
        assert!(total.abs() < 1e-10);
    }

    #[test]
    fn loss_vector_weighted_total_non_finite_inputs_return_worst_case() {
        let loss = LossVector {
            quality: f64::NAN,
            latency: 0.2,
            resource: 0.3,
        };
        assert_eq!(
            loss.weighted_total(0.5, 0.3, 0.2).to_bits(),
            f64::MAX.to_bits()
        );
        assert_eq!(
            LossVector::ZERO
                .weighted_total(f64::NAN, 0.3, 0.2)
                .to_bits(),
            f64::MAX.to_bits()
        );
    }

    #[test]
    fn loss_vector_weighted_total_negative_values_are_clamped() {
        let loss = LossVector {
            quality: -1.0,
            latency: 0.4,
            resource: -0.2,
        };
        let total = loss.weighted_total(-0.5, 1.0, 0.5);
        assert!((total - 0.4).abs() < 1e-10);
    }

    // ─── bd-2jqx tests end ───
}
