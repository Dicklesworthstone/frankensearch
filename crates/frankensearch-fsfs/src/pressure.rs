//! Host pressure sensing and control-state derivation.
//!
//! This module turns raw host signals into stable control states for fsfs
//! scheduling and UX layers:
//! - Collect CPU/memory/IO/load signals from `/proc` on Linux.
//! - Smooth noisy readings via EWMA.
//! - Derive `normal`/`constrained`/`degraded`/`emergency` states.
//! - Apply hysteresis + anti-flap consecutive-reading guards.

use std::fs;
use std::path::Path;
use std::time::Duration;

use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use crate::config::PressureProfile;

const DEFAULT_EWMA_ALPHA: f64 = 0.3;
const DEFAULT_HYSTERESIS_PCT: f64 = 5.0;
const DEFAULT_CONSECUTIVE_REQUIRED: u8 = 3;
const DEFAULT_IO_CEILING_MIB_PER_SEC: f64 = 64.0;
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

/// Stable control states consumed by scheduler and UX layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureState {
    Normal,
    Constrained,
    Degraded,
    Emergency,
}

impl PressureState {
    const fn severity(self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Constrained => 1,
            Self::Degraded => 2,
            Self::Emergency => 3,
        }
    }
}

/// Explicit feature-shedding ladder for graceful degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationStage {
    /// Normal operation: hybrid retrieval + full indexing throughput.
    Full,
    /// Quality/expensive embedding work is deferred; lexical+fast search stays up.
    EmbedDeferred,
    /// Query serving remains lexical-only; semantic retrieval is disabled.
    LexicalOnly,
    /// Search is reduced to metadata/status operations; heavy retrieval is disabled.
    MetadataOnly,
    /// Writes and query serving are paused until explicit recovery criteria are met.
    Paused,
}

impl DegradationStage {
    const fn severity(self) -> u8 {
        match self {
            Self::Full => 0,
            Self::EmbedDeferred => 1,
            Self::LexicalOnly => 2,
            Self::MetadataOnly => 3,
            Self::Paused => 4,
        }
    }

    const fn stage_reason_code(self) -> &'static str {
        match self {
            Self::Full => "degrade.stage.full",
            Self::EmbedDeferred => "degrade.stage.embed_deferred",
            Self::LexicalOnly => "degrade.stage.lexical_only",
            Self::MetadataOnly => "degrade.stage.metadata_only",
            Self::Paused => "degrade.stage.paused",
        }
    }

    const fn step_toward_full(self) -> Self {
        match self {
            Self::Paused => Self::MetadataOnly,
            Self::MetadataOnly => Self::LexicalOnly,
            Self::LexicalOnly => Self::EmbedDeferred,
            Self::EmbedDeferred | Self::Full => Self::Full,
        }
    }

    const fn contract(self) -> DegradationContract {
        match self {
            Self::Full => DegradationContract {
                stage: Self::Full,
                query_mode: QueryCapabilityMode::Hybrid,
                indexing_mode: IndexingCapabilityMode::Full,
                semantic_search: CapabilityState::Enabled,
                lexical_search: CapabilityState::Enabled,
                metadata_integrity: IntegrityState::Preserved,
                writes: WriteState::Enabled,
                user_banner: "Normal mode: hybrid retrieval and indexing are fully enabled.",
            },
            Self::EmbedDeferred => DegradationContract {
                stage: Self::EmbedDeferred,
                query_mode: QueryCapabilityMode::Hybrid,
                indexing_mode: IndexingCapabilityMode::DeferEmbedding,
                semantic_search: CapabilityState::Enabled,
                lexical_search: CapabilityState::Enabled,
                metadata_integrity: IntegrityState::Preserved,
                writes: WriteState::Enabled,
                user_banner: "Constrained mode: expensive embedding is deferred to protect latency.",
            },
            Self::LexicalOnly => DegradationContract {
                stage: Self::LexicalOnly,
                query_mode: QueryCapabilityMode::LexicalOnly,
                indexing_mode: IndexingCapabilityMode::DeferEmbedding,
                semantic_search: CapabilityState::Disabled,
                lexical_search: CapabilityState::Enabled,
                metadata_integrity: IntegrityState::Preserved,
                writes: WriteState::Enabled,
                user_banner: "Degraded mode: serving lexical-only results while preserving correctness.",
            },
            Self::MetadataOnly => DegradationContract {
                stage: Self::MetadataOnly,
                query_mode: QueryCapabilityMode::MetadataOnly,
                indexing_mode: IndexingCapabilityMode::MetadataOnly,
                semantic_search: CapabilityState::Disabled,
                lexical_search: CapabilityState::Disabled,
                metadata_integrity: IntegrityState::Preserved,
                writes: WriteState::Enabled,
                user_banner: "Safe mode: metadata operations only while search pipelines stabilize.",
            },
            Self::Paused => DegradationContract {
                stage: Self::Paused,
                query_mode: QueryCapabilityMode::Paused,
                indexing_mode: IndexingCapabilityMode::Paused,
                semantic_search: CapabilityState::Disabled,
                lexical_search: CapabilityState::Disabled,
                metadata_integrity: IntegrityState::Preserved,
                writes: WriteState::Paused,
                user_banner: "Emergency pause: write/query execution halted until recovery gates pass.",
            },
        }
    }

    const fn recovery_gate(self, consecutive_healthy_required: u8) -> Option<RecoveryGate> {
        match self {
            Self::Full => None,
            Self::EmbedDeferred => Some(RecoveryGate {
                max_pressure_for_recovery: PressureState::Normal,
                require_quality_circuit_closed: true,
                require_pause_cleared: false,
                consecutive_healthy_required,
            }),
            Self::LexicalOnly => Some(RecoveryGate {
                max_pressure_for_recovery: PressureState::Constrained,
                require_quality_circuit_closed: false,
                require_pause_cleared: false,
                consecutive_healthy_required,
            }),
            Self::MetadataOnly => Some(RecoveryGate {
                max_pressure_for_recovery: PressureState::Degraded,
                require_quality_circuit_closed: false,
                require_pause_cleared: false,
                consecutive_healthy_required,
            }),
            Self::Paused => Some(RecoveryGate {
                max_pressure_for_recovery: PressureState::Degraded,
                require_quality_circuit_closed: false,
                require_pause_cleared: true,
                consecutive_healthy_required,
            }),
        }
    }
}

/// User/operator override mode for degradation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationOverride {
    Auto,
    ForceFull,
    ForceEmbedDeferred,
    ForceLexicalOnly,
    ForceMetadataOnly,
    ForcePaused,
}

impl DegradationOverride {
    const fn forced_stage(self) -> Option<DegradationStage> {
        match self {
            Self::Auto => None,
            Self::ForceFull => Some(DegradationStage::Full),
            Self::ForceEmbedDeferred => Some(DegradationStage::EmbedDeferred),
            Self::ForceLexicalOnly => Some(DegradationStage::LexicalOnly),
            Self::ForceMetadataOnly => Some(DegradationStage::MetadataOnly),
            Self::ForcePaused => Some(DegradationStage::Paused),
        }
    }
}

/// Query-serving behavior for each degradation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryCapabilityMode {
    Hybrid,
    LexicalOnly,
    MetadataOnly,
    Paused,
}

/// Indexing/write behavior for each degradation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexingCapabilityMode {
    Full,
    DeferEmbedding,
    MetadataOnly,
    Paused,
}

/// Binary capability state for retrieval/indexing components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityState {
    Enabled,
    Disabled,
}

/// Metadata integrity guarantee mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrityState {
    Preserved,
}

/// Write-path capability mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteState {
    Enabled,
    Paused,
}

/// Correctness-preserving contract for a degradation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradationContract {
    pub stage: DegradationStage,
    pub query_mode: QueryCapabilityMode,
    pub indexing_mode: IndexingCapabilityMode,
    pub semantic_search: CapabilityState,
    pub lexical_search: CapabilityState,
    pub metadata_integrity: IntegrityState,
    pub writes: WriteState,
    pub user_banner: &'static str,
}

/// Inputs used to derive target degradation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradationSignal {
    pub pressure_state: PressureState,
    pub quality_circuit_open: bool,
    pub hard_pause_requested: bool,
}

impl DegradationSignal {
    #[must_use]
    pub const fn new(
        pressure_state: PressureState,
        quality_circuit_open: bool,
        hard_pause_requested: bool,
    ) -> Self {
        Self {
            pressure_state,
            quality_circuit_open,
            hard_pause_requested,
        }
    }

    #[must_use]
    const fn auto_target_stage(self) -> DegradationStage {
        if self.hard_pause_requested {
            return DegradationStage::Paused;
        }
        match self.pressure_state {
            PressureState::Normal => {
                if self.quality_circuit_open {
                    DegradationStage::EmbedDeferred
                } else {
                    DegradationStage::Full
                }
            }
            PressureState::Constrained => DegradationStage::EmbedDeferred,
            PressureState::Degraded => DegradationStage::LexicalOnly,
            PressureState::Emergency => DegradationStage::MetadataOnly,
        }
    }
}

/// Recovery gate used when stepping toward less severe degradation stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryGate {
    pub max_pressure_for_recovery: PressureState,
    pub require_quality_circuit_closed: bool,
    pub require_pause_cleared: bool,
    pub consecutive_healthy_required: u8,
}

impl RecoveryGate {
    #[must_use]
    const fn is_satisfied(self, signal: DegradationSignal) -> bool {
        if signal.pressure_state.severity() > self.max_pressure_for_recovery.severity() {
            return false;
        }
        if self.require_quality_circuit_closed && signal.quality_circuit_open {
            return false;
        }
        if self.require_pause_cleared && signal.hard_pause_requested {
            return false;
        }
        true
    }
}

/// Control-loop configuration for degradation transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradationControllerConfig {
    pub consecutive_healthy_required: u8,
}

impl Default for DegradationControllerConfig {
    fn default() -> Self {
        Self {
            consecutive_healthy_required: 3,
        }
    }
}

impl DegradationControllerConfig {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `consecutive_healthy_required`
    /// is zero.
    pub fn validate(self) -> SearchResult<Self> {
        if self.consecutive_healthy_required == 0 {
            return Err(SearchError::InvalidConfig {
                field: "degrade.consecutive_healthy_required".to_owned(),
                value: "0".to_owned(),
                reason: "must be >= 1".to_owned(),
            });
        }
        Ok(self)
    }
}

/// Trigger category attached to degradation transitions for audit events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationTrigger {
    Stable,
    PressureEscalation,
    QualityCircuitOpen,
    HardPause,
    Recovery,
    CalibrationBreach,
    OperatorOverride,
}

/// User-visible status projection for CLI/TUI surfaces.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradationStatus {
    pub timestamp_ms: u64,
    pub stage: DegradationStage,
    pub stage_reason_code: &'static str,
    pub transition_reason_code: &'static str,
    pub pressure_state: PressureState,
    pub quality_circuit_open: bool,
    pub hard_pause_requested: bool,
    pub query_mode: QueryCapabilityMode,
    pub indexing_mode: IndexingCapabilityMode,
    pub override_mode: DegradationOverride,
    pub override_allowed: bool,
    pub user_banner: &'static str,
}

/// Degradation transition emitted by the state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradationTransition {
    pub from: DegradationStage,
    pub to: DegradationStage,
    pub changed: bool,
    pub trigger: DegradationTrigger,
    pub reason_code: &'static str,
    pub pending_recovery_observations: u8,
    pub status: DegradationStatus,
}

/// Calibration guard evaluation status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationGuardStatus {
    Healthy,
    Watch,
    Breach,
}

/// Deterministic breach categories for adaptive-control calibration guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationBreachKind {
    CoverageBelowTarget,
    EValueBelowThreshold,
    DriftAboveThreshold,
    ConfidenceBelowThreshold,
}

/// Runtime calibration metrics used for guard evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub sample_count: u64,
    pub observed_coverage_pct: f64,
    pub e_value: f64,
    pub drift_pct: f64,
    pub confidence_pct: f64,
}

impl CalibrationMetrics {
    #[must_use]
    pub fn new(
        sample_count: u64,
        observed_coverage_pct: f64,
        e_value: f64,
        drift_pct: f64,
        confidence_pct: f64,
    ) -> Self {
        Self {
            sample_count,
            observed_coverage_pct: normalize_pct(observed_coverage_pct),
            e_value: if e_value.is_finite() {
                e_value.max(0.0)
            } else {
                0.0
            },
            drift_pct: normalize_pct(drift_pct),
            confidence_pct: normalize_pct(confidence_pct),
        }
    }
}

/// Configuration for calibration/anytime guard rails.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationGuardConfig {
    pub min_sample_count: u64,
    pub target_coverage_pct: f64,
    pub min_e_value: f64,
    pub max_drift_pct: f64,
    pub min_confidence_pct: f64,
    pub breach_consecutive_required: u8,
    pub fallback_stage: DegradationStage,
}

impl Default for CalibrationGuardConfig {
    fn default() -> Self {
        Self {
            min_sample_count: 200,
            target_coverage_pct: 95.0,
            min_e_value: 0.05,
            max_drift_pct: 15.0,
            min_confidence_pct: 70.0,
            breach_consecutive_required: 2,
            fallback_stage: DegradationStage::LexicalOnly,
        }
    }
}

impl CalibrationGuardConfig {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when calibration thresholds are
    /// outside valid ranges.
    pub fn validate(self) -> SearchResult<Self> {
        if !(1..=10_000_000).contains(&self.min_sample_count) {
            return Err(SearchError::InvalidConfig {
                field: "calibration.min_sample_count".to_owned(),
                value: self.min_sample_count.to_string(),
                reason: "must be in [1, 10000000]".to_owned(),
            });
        }
        if !(0.0..=100.0).contains(&self.target_coverage_pct) {
            return Err(SearchError::InvalidConfig {
                field: "calibration.target_coverage_pct".to_owned(),
                value: self.target_coverage_pct.to_string(),
                reason: "must be in [0, 100]".to_owned(),
            });
        }
        if !self.min_e_value.is_finite() || self.min_e_value < 0.0 {
            return Err(SearchError::InvalidConfig {
                field: "calibration.min_e_value".to_owned(),
                value: self.min_e_value.to_string(),
                reason: "must be finite and >= 0".to_owned(),
            });
        }
        if !(0.0..=100.0).contains(&self.max_drift_pct) {
            return Err(SearchError::InvalidConfig {
                field: "calibration.max_drift_pct".to_owned(),
                value: self.max_drift_pct.to_string(),
                reason: "must be in [0, 100]".to_owned(),
            });
        }
        if !(0.0..=100.0).contains(&self.min_confidence_pct) {
            return Err(SearchError::InvalidConfig {
                field: "calibration.min_confidence_pct".to_owned(),
                value: self.min_confidence_pct.to_string(),
                reason: "must be in [0, 100]".to_owned(),
            });
        }
        if self.breach_consecutive_required == 0 {
            return Err(SearchError::InvalidConfig {
                field: "calibration.breach_consecutive_required".to_owned(),
                value: "0".to_owned(),
                reason: "must be >= 1".to_owned(),
            });
        }
        Ok(self)
    }
}

/// Replay/audit evidence emitted by each calibration guard evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEvidence {
    pub sample_count: u64,
    pub observed_coverage_pct: f64,
    pub target_coverage_pct: f64,
    pub e_value: f64,
    pub min_e_value: f64,
    pub drift_pct: f64,
    pub max_drift_pct: f64,
    pub confidence_pct: f64,
    pub min_confidence_pct: f64,
    pub consecutive_breach_count: u8,
    pub fallback_triggered: bool,
}

/// Deterministic guard-evaluation output with fallback instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationGuardDecision {
    pub status: CalibrationGuardStatus,
    pub reason_code: &'static str,
    pub breach_kind: Option<CalibrationBreachKind>,
    pub fallback_stage: Option<DegradationStage>,
    pub evidence: CalibrationEvidence,
}

/// Stateful calibration guard for adaptive controllers.
#[derive(Debug, Clone, Default)]
pub struct CalibrationGuardState {
    config: CalibrationGuardConfig,
    consecutive_breach_count: u8,
}

pub const REASON_CALIBRATION_HEALTHY: &str = "calibration.guard.healthy";
pub const REASON_CALIBRATION_INSUFFICIENT_SAMPLES: &str = "calibration.guard.insufficient_samples";
pub const REASON_CALIBRATION_COVERAGE_BREACH: &str = "calibration.guard.coverage_breach";
pub const REASON_CALIBRATION_EVALUE_BREACH: &str = "calibration.guard.evalue_breach";
pub const REASON_CALIBRATION_DRIFT_BREACH: &str = "calibration.guard.drift_breach";
pub const REASON_CALIBRATION_CONFIDENCE_BREACH: &str = "calibration.guard.confidence_breach";

/// State machine that enforces explicit degradation and recovery behavior.
#[derive(Debug, Clone)]
pub struct DegradationStateMachine {
    config: DegradationControllerConfig,
    stage: DegradationStage,
    override_mode: DegradationOverride,
    pending_recovery_observations: u8,
}

pub const REASON_DEGRADE_STABLE: &str = "degrade.state.stable";
pub const REASON_DEGRADE_ESCALATED: &str = "degrade.transition.escalated";
pub const REASON_DEGRADE_RECOVERY_PENDING: &str = "degrade.transition.recovery_pending";
pub const REASON_DEGRADE_RECOVERED: &str = "degrade.transition.recovered";
pub const REASON_DEGRADE_OVERRIDE: &str = "degrade.transition.override";
pub const REASON_DEGRADE_CALIBRATION_FALLBACK: &str = "degrade.transition.calibration_fallback";

impl CalibrationGuardState {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `config` is invalid.
    pub fn new(config: CalibrationGuardConfig) -> SearchResult<Self> {
        Ok(Self {
            config: config.validate()?,
            consecutive_breach_count: 0,
        })
    }

    #[must_use]
    pub const fn config(&self) -> CalibrationGuardConfig {
        self.config
    }

    #[must_use]
    pub const fn consecutive_breach_count(&self) -> u8 {
        self.consecutive_breach_count
    }

    #[must_use]
    pub fn evaluate(&mut self, metrics: CalibrationMetrics) -> CalibrationGuardDecision {
        if metrics.sample_count < self.config.min_sample_count {
            self.consecutive_breach_count = 0;
            return CalibrationGuardDecision {
                status: CalibrationGuardStatus::Watch,
                reason_code: REASON_CALIBRATION_INSUFFICIENT_SAMPLES,
                breach_kind: None,
                fallback_stage: None,
                evidence: CalibrationEvidence {
                    sample_count: metrics.sample_count,
                    observed_coverage_pct: metrics.observed_coverage_pct,
                    target_coverage_pct: self.config.target_coverage_pct,
                    e_value: metrics.e_value,
                    min_e_value: self.config.min_e_value,
                    drift_pct: metrics.drift_pct,
                    max_drift_pct: self.config.max_drift_pct,
                    confidence_pct: metrics.confidence_pct,
                    min_confidence_pct: self.config.min_confidence_pct,
                    consecutive_breach_count: self.consecutive_breach_count,
                    fallback_triggered: false,
                },
            };
        }

        let breach = if metrics.observed_coverage_pct < self.config.target_coverage_pct {
            Some((
                CalibrationBreachKind::CoverageBelowTarget,
                REASON_CALIBRATION_COVERAGE_BREACH,
            ))
        } else if metrics.e_value < self.config.min_e_value {
            Some((
                CalibrationBreachKind::EValueBelowThreshold,
                REASON_CALIBRATION_EVALUE_BREACH,
            ))
        } else if metrics.drift_pct > self.config.max_drift_pct {
            Some((
                CalibrationBreachKind::DriftAboveThreshold,
                REASON_CALIBRATION_DRIFT_BREACH,
            ))
        } else if metrics.confidence_pct < self.config.min_confidence_pct {
            Some((
                CalibrationBreachKind::ConfidenceBelowThreshold,
                REASON_CALIBRATION_CONFIDENCE_BREACH,
            ))
        } else {
            None
        };

        let (status, reason_code, breach_kind, fallback_stage) =
            if let Some((kind, reason)) = breach {
                self.consecutive_breach_count = self.consecutive_breach_count.saturating_add(1);
                let fallback = (self.consecutive_breach_count
                    >= self.config.breach_consecutive_required)
                    .then_some(self.config.fallback_stage);
                (CalibrationGuardStatus::Breach, reason, Some(kind), fallback)
            } else {
                self.consecutive_breach_count = 0;
                (
                    CalibrationGuardStatus::Healthy,
                    REASON_CALIBRATION_HEALTHY,
                    None,
                    None,
                )
            };

        CalibrationGuardDecision {
            status,
            reason_code,
            breach_kind,
            fallback_stage,
            evidence: CalibrationEvidence {
                sample_count: metrics.sample_count,
                observed_coverage_pct: metrics.observed_coverage_pct,
                target_coverage_pct: self.config.target_coverage_pct,
                e_value: metrics.e_value,
                min_e_value: self.config.min_e_value,
                drift_pct: metrics.drift_pct,
                max_drift_pct: self.config.max_drift_pct,
                confidence_pct: metrics.confidence_pct,
                min_confidence_pct: self.config.min_confidence_pct,
                consecutive_breach_count: self.consecutive_breach_count,
                fallback_triggered: fallback_stage.is_some(),
            },
        }
    }
}

impl Default for DegradationStateMachine {
    fn default() -> Self {
        Self {
            config: DegradationControllerConfig::default(),
            stage: DegradationStage::Full,
            override_mode: DegradationOverride::Auto,
            pending_recovery_observations: 0,
        }
    }
}

impl DegradationStateMachine {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `config` is invalid.
    pub fn new(config: DegradationControllerConfig) -> SearchResult<Self> {
        Ok(Self {
            config: config.validate()?,
            stage: DegradationStage::Full,
            override_mode: DegradationOverride::Auto,
            pending_recovery_observations: 0,
        })
    }

    #[must_use]
    pub const fn stage(&self) -> DegradationStage {
        self.stage
    }

    #[must_use]
    pub const fn override_mode(&self) -> DegradationOverride {
        self.override_mode
    }

    pub const fn set_override(&mut self, override_mode: DegradationOverride) {
        self.override_mode = override_mode;
        self.pending_recovery_observations = 0;
    }

    #[must_use]
    pub fn observe(
        &mut self,
        signal: DegradationSignal,
        timestamp_ms: u64,
    ) -> DegradationTransition {
        let from = self.stage;
        let mut trigger = DegradationTrigger::Stable;
        let mut reason_code = REASON_DEGRADE_STABLE;
        let mut changed = false;

        if let Some(forced) = self.override_mode.forced_stage() {
            self.pending_recovery_observations = 0;
            if forced != self.stage {
                self.stage = forced;
                changed = true;
            }
            trigger = DegradationTrigger::OperatorOverride;
            reason_code = REASON_DEGRADE_OVERRIDE;
        } else {
            let target = signal.auto_target_stage();
            match target.severity().cmp(&self.stage.severity()) {
                std::cmp::Ordering::Greater => {
                    self.stage = target;
                    self.pending_recovery_observations = 0;
                    changed = self.stage != from;
                    trigger = if signal.hard_pause_requested {
                        DegradationTrigger::HardPause
                    } else if signal.quality_circuit_open
                        && matches!(signal.pressure_state, PressureState::Normal)
                        && matches!(target, DegradationStage::EmbedDeferred)
                    {
                        DegradationTrigger::QualityCircuitOpen
                    } else {
                        DegradationTrigger::PressureEscalation
                    };
                    reason_code = REASON_DEGRADE_ESCALATED;
                }
                std::cmp::Ordering::Less => {
                    if let Some(gate) = self
                        .stage
                        .recovery_gate(self.config.consecutive_healthy_required)
                    {
                        if gate.is_satisfied(signal) {
                            self.pending_recovery_observations =
                                self.pending_recovery_observations.saturating_add(1);
                            trigger = DegradationTrigger::Recovery;
                            if self.pending_recovery_observations
                                >= gate.consecutive_healthy_required
                            {
                                self.stage = self.stage.step_toward_full();
                                self.pending_recovery_observations = 0;
                                changed = self.stage != from;
                                reason_code = REASON_DEGRADE_RECOVERED;
                            } else {
                                reason_code = REASON_DEGRADE_RECOVERY_PENDING;
                            }
                        } else {
                            self.pending_recovery_observations = 0;
                        }
                    }
                }
                std::cmp::Ordering::Equal => {
                    self.pending_recovery_observations = 0;
                }
            }
        }

        let contract = self.stage.contract();
        DegradationTransition {
            from,
            to: self.stage,
            changed,
            trigger,
            reason_code,
            pending_recovery_observations: self.pending_recovery_observations,
            status: DegradationStatus {
                timestamp_ms,
                stage: self.stage,
                stage_reason_code: self.stage.stage_reason_code(),
                transition_reason_code: reason_code,
                pressure_state: signal.pressure_state,
                quality_circuit_open: signal.quality_circuit_open,
                hard_pause_requested: signal.hard_pause_requested,
                query_mode: contract.query_mode,
                indexing_mode: contract.indexing_mode,
                override_mode: self.override_mode,
                override_allowed: true,
                user_banner: contract.user_banner,
            },
        }
    }

    /// Apply deterministic fallback when calibration guards breach.
    ///
    /// Returns `None` when no fallback is required (or the fallback is less
    /// restrictive than the current stage).
    #[must_use]
    pub fn apply_calibration_guard(
        &mut self,
        signal: DegradationSignal,
        decision: &CalibrationGuardDecision,
        timestamp_ms: u64,
    ) -> Option<DegradationTransition> {
        let fallback = decision.fallback_stage?;
        if fallback.severity() <= self.stage.severity() {
            return None;
        }

        let from = self.stage;
        self.stage = fallback;
        self.pending_recovery_observations = 0;
        let contract = self.stage.contract();
        Some(DegradationTransition {
            from,
            to: self.stage,
            changed: self.stage != from,
            trigger: DegradationTrigger::CalibrationBreach,
            reason_code: REASON_DEGRADE_CALIBRATION_FALLBACK,
            pending_recovery_observations: self.pending_recovery_observations,
            status: DegradationStatus {
                timestamp_ms,
                stage: self.stage,
                stage_reason_code: self.stage.stage_reason_code(),
                transition_reason_code: decision.reason_code,
                pressure_state: signal.pressure_state,
                quality_circuit_open: signal.quality_circuit_open,
                hard_pause_requested: signal.hard_pause_requested,
                query_mode: contract.query_mode,
                indexing_mode: contract.indexing_mode,
                override_mode: self.override_mode,
                override_allowed: true,
                user_banner: contract.user_banner,
            },
        })
    }
}

/// One pressure reading normalized to percentages.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureSignal {
    pub cpu_pct: f64,
    pub memory_pct: f64,
    pub io_pct: f64,
    pub load_pct: f64,
}

impl PressureSignal {
    #[must_use]
    pub fn new(cpu_pct: f64, memory_pct: f64, io_pct: f64, load_pct: f64) -> Self {
        Self {
            cpu_pct: normalize_pct(cpu_pct),
            memory_pct: normalize_pct(memory_pct),
            io_pct: normalize_pct(io_pct),
            load_pct: normalize_pct(load_pct),
        }
    }

    #[must_use]
    pub fn ewma(self, previous: Self, alpha: f64) -> Self {
        let alpha = if alpha.is_finite() {
            alpha.clamp(0.0, 1.0)
        } else {
            0.3 // Default alpha for non-finite input
        };
        let blend = |current: f64, prior: f64| alpha.mul_add(current, (1.0 - alpha) * prior);
        Self::new(
            blend(self.cpu_pct, previous.cpu_pct),
            blend(self.memory_pct, previous.memory_pct),
            blend(self.io_pct, previous.io_pct),
            blend(self.load_pct, previous.load_pct),
        )
    }

    #[must_use]
    pub const fn score(self) -> f64 {
        self.cpu_pct
            .max(self.memory_pct)
            .max(self.io_pct)
            .max(self.load_pct)
    }
}

/// Snapshot emitted by the controller for telemetry/UX use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureSnapshot {
    pub timestamp_ms: u64,
    pub raw: PressureSignal,
    pub smoothed: PressureSignal,
    pub score: f64,
    pub state: PressureState,
}

/// Transition decision from one observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureTransition {
    pub from: PressureState,
    pub to: PressureState,
    pub changed: bool,
    pub reason_code: &'static str,
    pub consecutive_required: u8,
    pub consecutive_observed: u8,
    pub snapshot: PressureSnapshot,
}

/// Controller configuration for smoothing and anti-flap behavior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureControllerConfig {
    pub profile: PressureProfile,
    pub ewma_alpha: f64,
    pub hysteresis_pct: f64,
    pub consecutive_required: u8,
}

impl Default for PressureControllerConfig {
    fn default() -> Self {
        Self {
            profile: PressureProfile::Performance,
            ewma_alpha: DEFAULT_EWMA_ALPHA,
            hysteresis_pct: DEFAULT_HYSTERESIS_PCT,
            consecutive_required: DEFAULT_CONSECUTIVE_REQUIRED,
        }
    }
}

impl PressureControllerConfig {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when alpha/hysteresis/consecutive
    /// settings are invalid.
    pub fn validate(self) -> SearchResult<Self> {
        if !(0.0..=1.0).contains(&self.ewma_alpha) {
            return Err(SearchError::InvalidConfig {
                field: "pressure.ewma_alpha".to_owned(),
                value: self.ewma_alpha.to_string(),
                reason: "must be between 0.0 and 1.0".to_owned(),
            });
        }
        if !(0.0..=30.0).contains(&self.hysteresis_pct) {
            return Err(SearchError::InvalidConfig {
                field: "pressure.hysteresis_pct".to_owned(),
                value: self.hysteresis_pct.to_string(),
                reason: "must be between 0.0 and 30.0".to_owned(),
            });
        }
        if self.consecutive_required == 0 {
            return Err(SearchError::InvalidConfig {
                field: "pressure.consecutive_required".to_owned(),
                value: "0".to_owned(),
                reason: "must be >= 1".to_owned(),
            });
        }
        Ok(self)
    }
}

/// Stateful controller that derives stable pressure states.
#[derive(Debug, Clone)]
pub struct PressureController {
    config: PressureControllerConfig,
    current_state: PressureState,
    smoothed_signal: Option<PressureSignal>,
    pending_state: Option<PressureState>,
    pending_consecutive: u8,
}

impl PressureController {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when `config` is invalid.
    pub fn new(config: PressureControllerConfig) -> SearchResult<Self> {
        Ok(Self {
            config: config.validate()?,
            current_state: PressureState::Normal,
            smoothed_signal: None,
            pending_state: None,
            pending_consecutive: 0,
        })
    }

    /// Build a controller from profile defaults.
    #[must_use]
    pub fn from_profile(profile: PressureProfile) -> Self {
        Self {
            config: PressureControllerConfig {
                profile,
                ..PressureControllerConfig::default()
            },
            current_state: PressureState::Normal,
            smoothed_signal: None,
            pending_state: None,
            pending_consecutive: 0,
        }
    }

    #[must_use]
    pub const fn state(&self) -> PressureState {
        self.current_state
    }

    #[must_use]
    pub const fn config(&self) -> PressureControllerConfig {
        self.config
    }

    /// Observe one signal and derive a stable pressure transition.
    #[must_use]
    pub fn observe(&mut self, raw: PressureSignal, timestamp_ms: u64) -> PressureTransition {
        let smoothed = self
            .smoothed_signal
            .map_or(raw, |previous| raw.ewma(previous, self.config.ewma_alpha));
        self.smoothed_signal = Some(smoothed);

        let thresholds = PressureThresholds::for_profile(self.config.profile);
        let score = smoothed.score();
        let target = self.target_state(score, thresholds);
        let from_state = self.current_state;

        let (changed, reason_code) = if target == self.current_state {
            self.pending_state = None;
            self.pending_consecutive = 0;
            (false, "pressure.state.stable")
        } else {
            if self.pending_state == Some(target) {
                self.pending_consecutive = self.pending_consecutive.saturating_add(1);
            } else {
                self.pending_state = Some(target);
                self.pending_consecutive = 1;
            }

            if self.pending_consecutive >= self.config.consecutive_required {
                self.current_state = target;
                self.pending_state = None;
                self.pending_consecutive = 0;
                (true, "pressure.transition.applied")
            } else {
                (false, "pressure.transition.pending")
            }
        };

        PressureTransition {
            from: from_state,
            to: self.current_state,
            changed,
            reason_code,
            consecutive_required: self.config.consecutive_required,
            consecutive_observed: self.pending_consecutive,
            snapshot: PressureSnapshot {
                timestamp_ms,
                raw,
                smoothed,
                score,
                state: self.current_state,
            },
        }
    }

    fn target_state(&self, score: f64, thresholds: PressureThresholds) -> PressureState {
        let upward = thresholds.state_for_up(score);
        if upward.severity() > self.current_state.severity() {
            return upward;
        }

        if upward.severity() < self.current_state.severity() {
            return thresholds.state_for_down(score, self.config.hysteresis_pct);
        }

        self.current_state
    }
}

#[derive(Debug, Clone, Copy)]
struct PressureThresholds {
    constrained: f64,
    degraded: f64,
    emergency: f64,
}

impl PressureThresholds {
    const fn for_profile(profile: PressureProfile) -> Self {
        match profile {
            PressureProfile::Strict => Self {
                constrained: 60.0,
                degraded: 75.0,
                emergency: 90.0,
            },
            PressureProfile::Performance => Self {
                constrained: 70.0,
                degraded: 85.0,
                emergency: 95.0,
            },
            PressureProfile::Degraded => Self {
                constrained: 80.0,
                degraded: 90.0,
                emergency: 98.0,
            },
        }
    }

    const fn state_for_up(self, score: f64) -> PressureState {
        if score >= self.emergency {
            return PressureState::Emergency;
        }
        if score >= self.degraded {
            return PressureState::Degraded;
        }
        if score >= self.constrained {
            return PressureState::Constrained;
        }
        PressureState::Normal
    }

    fn state_for_down(self, score: f64, hysteresis_pct: f64) -> PressureState {
        let margin = hysteresis_pct.max(0.0);
        let emergency_down = (self.emergency - margin).max(0.0);
        let degraded_down = (self.degraded - margin).max(0.0);
        let constrained_down = (self.constrained - margin).max(0.0);

        if score >= emergency_down {
            return PressureState::Emergency;
        }
        if score >= degraded_down {
            return PressureState::Degraded;
        }
        if score >= constrained_down {
            return PressureState::Constrained;
        }
        PressureState::Normal
    }
}

/// Raw `/proc/self/io` counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcIoCounters {
    pub read_bytes: u64,
    pub write_bytes: u64,
}

/// Linux host pressure collector backed by `/proc`.
#[derive(Debug, Clone)]
pub struct HostPressureCollector {
    io_ceiling_mib_per_sec: f64,
    previous_io: Option<ProcIoCounters>,
}

impl Default for HostPressureCollector {
    fn default() -> Self {
        Self {
            io_ceiling_mib_per_sec: DEFAULT_IO_CEILING_MIB_PER_SEC,
            previous_io: None,
        }
    }
}

impl HostPressureCollector {
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] if `io_ceiling_mib_per_sec <= 0`.
    pub fn new(io_ceiling_mib_per_sec: f64) -> SearchResult<Self> {
        if !io_ceiling_mib_per_sec.is_finite() || io_ceiling_mib_per_sec <= 0.0 {
            return Err(SearchError::InvalidConfig {
                field: "pressure.io_ceiling_mib_per_sec".to_owned(),
                value: io_ceiling_mib_per_sec.to_string(),
                reason: "must be > 0".to_owned(),
            });
        }
        Ok(Self {
            io_ceiling_mib_per_sec,
            previous_io: None,
        })
    }

    /// Collect one host pressure signal sample.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Io`] or [`SearchError::InvalidConfig`] when proc
    /// files cannot be read/parsed (Linux only).
    #[allow(clippy::cast_precision_loss)]
    pub fn collect(
        &mut self,
        interval: Duration,
        memory_ceiling_mb: usize,
    ) -> SearchResult<PressureSignal> {
        #[cfg(target_os = "linux")]
        {
            let load_avg_1m = read_load_avg_1m(Path::new("/proc/loadavg"))?;
            let cpu_slots =
                std::thread::available_parallelism().map_or(1_usize, std::num::NonZeroUsize::get);
            let load_pct = normalize_pct((load_avg_1m / cpu_slots as f64) * 100.0);
            let cpu_pct = load_pct;

            let rss_mb = read_self_rss_mb(Path::new("/proc/self/status"))?;
            let effective_memory_ceiling_mb = memory_ceiling_mb.max(1);
            let memory_pct =
                normalize_pct((rss_mb as f64 / effective_memory_ceiling_mb as f64) * 100.0);

            let io_now = read_proc_self_io(Path::new("/proc/self/io"))?;
            let io_pct = self.io_pct_for_interval(interval, io_now);
            self.previous_io = Some(io_now);

            Ok(PressureSignal::new(cpu_pct, memory_pct, io_pct, load_pct))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (interval, memory_ceiling_mb);
            Ok(PressureSignal::new(0.0, 0.0, 0.0, 0.0))
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn io_pct_for_interval(&self, interval: Duration, current: ProcIoCounters) -> f64 {
        let Some(previous) = self.previous_io else {
            return 0.0;
        };
        let elapsed_secs = interval.as_secs_f64();
        if elapsed_secs <= f64::EPSILON {
            return 0.0;
        }

        let previous_total = previous.read_bytes.saturating_add(previous.write_bytes);
        let current_total = current.read_bytes.saturating_add(current.write_bytes);
        let delta_bytes = current_total.saturating_sub(previous_total);
        let mib_per_sec = (delta_bytes as f64 / BYTES_PER_MIB) / elapsed_secs;
        normalize_pct((mib_per_sec / self.io_ceiling_mib_per_sec) * 100.0)
    }
}

fn normalize_pct(value: f64) -> f64 {
    if !value.is_finite() || value <= 0.0 {
        return 0.0;
    }
    value.min(200.0)
}

#[cfg(target_os = "linux")]
fn read_proc_self_io(path: &Path) -> SearchResult<ProcIoCounters> {
    let contents = fs::read_to_string(path)?;
    parse_proc_self_io(&contents)
}

#[cfg(target_os = "linux")]
fn parse_proc_self_io(contents: &str) -> SearchResult<ProcIoCounters> {
    let mut read_bytes = None;
    let mut write_bytes = None;

    for line in contents.lines() {
        let mut fields = line.split_whitespace();
        let key = fields.next().unwrap_or_default();
        let value = fields.next().unwrap_or_default();
        if key == "read_bytes:" {
            read_bytes = Some(parse_u64_field("pressure.proc_io.read_bytes", value)?);
        } else if key == "write_bytes:" {
            write_bytes = Some(parse_u64_field("pressure.proc_io.write_bytes", value)?);
        }
    }

    let read_bytes = read_bytes.ok_or_else(|| SearchError::InvalidConfig {
        field: "pressure.proc_io.read_bytes".to_owned(),
        value: "<missing>".to_owned(),
        reason: "read_bytes field missing in /proc/self/io".to_owned(),
    })?;
    let write_bytes = write_bytes.ok_or_else(|| SearchError::InvalidConfig {
        field: "pressure.proc_io.write_bytes".to_owned(),
        value: "<missing>".to_owned(),
        reason: "write_bytes field missing in /proc/self/io".to_owned(),
    })?;

    Ok(ProcIoCounters {
        read_bytes,
        write_bytes,
    })
}

#[cfg(target_os = "linux")]
fn read_self_rss_mb(path: &Path) -> SearchResult<u64> {
    let contents = fs::read_to_string(path)?;
    parse_self_status_rss_mb(&contents)
}

#[cfg(target_os = "linux")]
fn parse_self_status_rss_mb(contents: &str) -> SearchResult<u64> {
    for line in contents.lines() {
        if !line.starts_with("VmRSS:") {
            continue;
        }
        let mut fields = line.split_whitespace();
        let _ = fields.next();
        let value = fields.next().ok_or_else(|| SearchError::InvalidConfig {
            field: "pressure.self_status.rss_kb".to_owned(),
            value: "<missing>".to_owned(),
            reason: "VmRSS value missing from /proc/self/status".to_owned(),
        })?;
        let rss_kb = parse_u64_field("pressure.self_status.rss_kb", value)?;
        return Ok(rss_kb / 1024);
    }

    Err(SearchError::InvalidConfig {
        field: "pressure.self_status.rss_kb".to_owned(),
        value: "<missing>".to_owned(),
        reason: "VmRSS field missing in /proc/self/status".to_owned(),
    })
}

#[cfg(target_os = "linux")]
fn read_load_avg_1m(path: &Path) -> SearchResult<f64> {
    let contents = fs::read_to_string(path)?;
    parse_load_avg_1m(&contents)
}

fn parse_load_avg_1m(contents: &str) -> SearchResult<f64> {
    let first = contents
        .split_whitespace()
        .next()
        .ok_or_else(|| SearchError::InvalidConfig {
            field: "pressure.loadavg.1m".to_owned(),
            value: "<missing>".to_owned(),
            reason: "loadavg missing first field".to_owned(),
        })?;
    first
        .parse::<f64>()
        .map_err(|err| SearchError::InvalidConfig {
            field: "pressure.loadavg.1m".to_owned(),
            value: first.to_owned(),
            reason: format!("invalid loadavg value: {err}"),
        })
}

fn parse_u64_field(field: &str, value: &str) -> SearchResult<u64> {
    value
        .parse::<u64>()
        .map_err(|err| SearchError::InvalidConfig {
            field: field.to_owned(),
            value: value.to_owned(),
            reason: format!("invalid integer value: {err}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controller_requires_consecutive_readings_before_transition() {
        let mut controller = PressureController::from_profile(PressureProfile::Performance);
        let sample = PressureSignal::new(72.0, 30.0, 20.0, 40.0);

        let first = controller.observe(sample, 1);
        assert_eq!(first.from, PressureState::Normal);
        assert_eq!(first.to, PressureState::Normal);
        assert_eq!(first.reason_code, "pressure.transition.pending");
        assert_eq!(first.consecutive_observed, 1);

        let second = controller.observe(sample, 2);
        assert_eq!(second.to, PressureState::Normal);
        assert_eq!(second.reason_code, "pressure.transition.pending");
        assert_eq!(second.consecutive_observed, 2);

        let third = controller.observe(sample, 3);
        assert_eq!(third.from, PressureState::Normal);
        assert_eq!(third.to, PressureState::Constrained);
        assert!(third.changed);
        assert_eq!(third.reason_code, "pressure.transition.applied");
    }

    #[test]
    fn controller_hysteresis_prevents_flapping() {
        let mut controller = PressureController::from_profile(PressureProfile::Performance);
        let high = PressureSignal::new(74.0, 30.0, 20.0, 40.0);
        let near_boundary = PressureSignal::new(67.0, 30.0, 20.0, 40.0);
        let low = PressureSignal::new(40.0, 30.0, 20.0, 20.0);

        for ts in 0_u64..3_u64 {
            let _ = controller.observe(high, ts);
        }
        assert_eq!(controller.state(), PressureState::Constrained);

        for ts in 3_u64..9_u64 {
            let transition = controller.observe(near_boundary, ts);
            assert_eq!(transition.to, PressureState::Constrained);
            assert!(!transition.changed);
        }

        for ts in 9_u64..12_u64 {
            let _ = controller.observe(low, ts);
        }
        assert_eq!(controller.state(), PressureState::Normal);
    }

    #[test]
    fn controller_reaches_emergency_under_extreme_pressure() {
        let mut controller = PressureController::from_profile(PressureProfile::Performance);
        let emergency = PressureSignal::new(150.0, 130.0, 90.0, 140.0);

        for ts in 0_u64..3_u64 {
            let _ = controller.observe(emergency, ts);
        }

        assert_eq!(controller.state(), PressureState::Emergency);
    }

    #[test]
    fn degradation_signal_maps_to_feature_shedding_ladder() {
        assert_eq!(
            DegradationSignal::new(PressureState::Normal, false, false).auto_target_stage(),
            DegradationStage::Full
        );
        assert_eq!(
            DegradationSignal::new(PressureState::Normal, true, false).auto_target_stage(),
            DegradationStage::EmbedDeferred
        );
        assert_eq!(
            DegradationSignal::new(PressureState::Constrained, false, false).auto_target_stage(),
            DegradationStage::EmbedDeferred
        );
        assert_eq!(
            DegradationSignal::new(PressureState::Degraded, false, false).auto_target_stage(),
            DegradationStage::LexicalOnly
        );
        assert_eq!(
            DegradationSignal::new(PressureState::Emergency, false, false).auto_target_stage(),
            DegradationStage::MetadataOnly
        );
        assert_eq!(
            DegradationSignal::new(PressureState::Normal, false, true).auto_target_stage(),
            DegradationStage::Paused
        );
    }

    #[test]
    fn degradation_contracts_define_correctness_preserving_behavior() {
        let full = DegradationStage::Full.contract();
        assert_eq!(full.query_mode, QueryCapabilityMode::Hybrid);
        assert_eq!(full.indexing_mode, IndexingCapabilityMode::Full);
        assert_eq!(full.semantic_search, CapabilityState::Enabled);
        assert_eq!(full.lexical_search, CapabilityState::Enabled);
        assert_eq!(full.metadata_integrity, IntegrityState::Preserved);
        assert_eq!(full.writes, WriteState::Enabled);

        let lexical_only = DegradationStage::LexicalOnly.contract();
        assert_eq!(lexical_only.query_mode, QueryCapabilityMode::LexicalOnly);
        assert_eq!(lexical_only.semantic_search, CapabilityState::Disabled);
        assert_eq!(lexical_only.lexical_search, CapabilityState::Enabled);
        assert_eq!(lexical_only.metadata_integrity, IntegrityState::Preserved);
        assert_eq!(lexical_only.writes, WriteState::Enabled);

        let paused = DegradationStage::Paused.contract();
        assert_eq!(paused.query_mode, QueryCapabilityMode::Paused);
        assert_eq!(paused.indexing_mode, IndexingCapabilityMode::Paused);
        assert_eq!(paused.semantic_search, CapabilityState::Disabled);
        assert_eq!(paused.lexical_search, CapabilityState::Disabled);
        assert_eq!(paused.metadata_integrity, IntegrityState::Preserved);
        assert_eq!(paused.writes, WriteState::Paused);
    }

    #[test]
    fn degradation_state_machine_escalates_immediately_and_recovers_stepwise() {
        let mut machine = DegradationStateMachine::default();
        let constrained = DegradationSignal::new(PressureState::Constrained, false, false);
        let degraded = DegradationSignal::new(PressureState::Degraded, false, false);
        let normal = DegradationSignal::new(PressureState::Normal, false, false);

        let first = machine.observe(constrained, 1);
        assert_eq!(first.to, DegradationStage::EmbedDeferred);
        assert_eq!(first.reason_code, REASON_DEGRADE_ESCALATED);
        assert_eq!(first.trigger, DegradationTrigger::PressureEscalation);

        let second = machine.observe(degraded, 2);
        assert_eq!(second.to, DegradationStage::LexicalOnly);
        assert_eq!(second.reason_code, REASON_DEGRADE_ESCALATED);

        let p1 = machine.observe(normal, 3);
        assert_eq!(p1.to, DegradationStage::LexicalOnly);
        assert_eq!(p1.reason_code, REASON_DEGRADE_RECOVERY_PENDING);
        assert_eq!(p1.pending_recovery_observations, 1);

        let p2 = machine.observe(normal, 4);
        assert_eq!(p2.to, DegradationStage::LexicalOnly);
        assert_eq!(p2.reason_code, REASON_DEGRADE_RECOVERY_PENDING);
        assert_eq!(p2.pending_recovery_observations, 2);

        let recovered_once = machine.observe(normal, 5);
        assert_eq!(recovered_once.to, DegradationStage::EmbedDeferred);
        assert_eq!(recovered_once.reason_code, REASON_DEGRADE_RECOVERED);
        assert_eq!(recovered_once.trigger, DegradationTrigger::Recovery);

        let _ = machine.observe(normal, 6);
        let _ = machine.observe(normal, 7);
        let recovered_full = machine.observe(normal, 8);
        assert_eq!(recovered_full.to, DegradationStage::Full);
        assert_eq!(recovered_full.reason_code, REASON_DEGRADE_RECOVERED);
    }

    #[test]
    fn degradation_state_machine_honors_operator_override() {
        let mut machine = DegradationStateMachine::default();
        machine.set_override(DegradationOverride::ForceMetadataOnly);
        let normal = DegradationSignal::new(PressureState::Normal, false, false);

        let forced = machine.observe(normal, 1);
        assert_eq!(forced.to, DegradationStage::MetadataOnly);
        assert_eq!(forced.trigger, DegradationTrigger::OperatorOverride);
        assert_eq!(forced.reason_code, REASON_DEGRADE_OVERRIDE);
        assert_eq!(
            forced.status.override_mode,
            DegradationOverride::ForceMetadataOnly
        );
        assert_eq!(forced.status.query_mode, QueryCapabilityMode::MetadataOnly);
        assert_eq!(
            forced.status.stage_reason_code,
            "degrade.stage.metadata_only"
        );

        machine.set_override(DegradationOverride::Auto);
        let _ = machine.observe(normal, 2);
        let _ = machine.observe(normal, 3);
        let recovered = machine.observe(normal, 4);
        assert_eq!(recovered.to, DegradationStage::LexicalOnly);
        assert_eq!(recovered.reason_code, REASON_DEGRADE_RECOVERED);
    }

    #[test]
    fn hard_pause_must_clear_before_recovery_can_start() {
        let mut machine = DegradationStateMachine::default();
        let pause_signal = DegradationSignal::new(PressureState::Emergency, false, true);

        let paused = machine.observe(pause_signal, 1);
        assert_eq!(paused.to, DegradationStage::Paused);
        assert_eq!(paused.trigger, DegradationTrigger::HardPause);

        let still_paused = machine.observe(
            DegradationSignal::new(PressureState::Normal, false, true),
            2,
        );
        assert_eq!(still_paused.to, DegradationStage::Paused);
        assert_eq!(still_paused.reason_code, REASON_DEGRADE_STABLE);

        let _ = machine.observe(
            DegradationSignal::new(PressureState::Normal, false, false),
            3,
        );
        let _ = machine.observe(
            DegradationSignal::new(PressureState::Normal, false, false),
            4,
        );
        let recovered = machine.observe(
            DegradationSignal::new(PressureState::Normal, false, false),
            5,
        );
        assert_eq!(recovered.to, DegradationStage::MetadataOnly);
        assert_eq!(recovered.reason_code, REASON_DEGRADE_RECOVERED);
    }

    #[test]
    fn calibration_guard_reports_watch_until_sample_floor() {
        let mut guard = CalibrationGuardState::default();
        let decision = guard.evaluate(CalibrationMetrics::new(42, 92.0, 0.10, 4.0, 88.0));

        assert_eq!(decision.status, CalibrationGuardStatus::Watch);
        assert_eq!(
            decision.reason_code,
            REASON_CALIBRATION_INSUFFICIENT_SAMPLES
        );
        assert!(decision.fallback_stage.is_none());
        assert_eq!(decision.evidence.consecutive_breach_count, 0);
        assert!(!decision.evidence.fallback_triggered);
    }

    #[test]
    fn calibration_guard_triggers_fallback_after_consecutive_breaches() {
        let mut guard = CalibrationGuardState::new(CalibrationGuardConfig {
            breach_consecutive_required: 2,
            fallback_stage: DegradationStage::MetadataOnly,
            ..CalibrationGuardConfig::default()
        })
        .expect("guard config");
        let breach_metrics = CalibrationMetrics::new(500, 70.0, 0.80, 2.0, 90.0);

        let first = guard.evaluate(breach_metrics);
        assert_eq!(first.status, CalibrationGuardStatus::Breach);
        assert_eq!(first.reason_code, REASON_CALIBRATION_COVERAGE_BREACH);
        assert!(first.fallback_stage.is_none());
        assert_eq!(first.evidence.consecutive_breach_count, 1);

        let second = guard.evaluate(breach_metrics);
        assert_eq!(second.status, CalibrationGuardStatus::Breach);
        assert_eq!(second.reason_code, REASON_CALIBRATION_COVERAGE_BREACH);
        assert_eq!(second.fallback_stage, Some(DegradationStage::MetadataOnly));
        assert!(second.evidence.fallback_triggered);
        assert_eq!(second.evidence.consecutive_breach_count, 2);
    }

    #[test]
    fn calibration_guard_resets_breach_streak_after_healthy_metrics() {
        let mut guard = CalibrationGuardState::default();
        let _ = guard.evaluate(CalibrationMetrics::new(500, 80.0, 0.90, 2.0, 90.0));
        assert_eq!(guard.consecutive_breach_count(), 1);

        let healthy = guard.evaluate(CalibrationMetrics::new(500, 97.0, 0.90, 2.0, 92.0));
        assert_eq!(healthy.status, CalibrationGuardStatus::Healthy);
        assert_eq!(healthy.reason_code, REASON_CALIBRATION_HEALTHY);
        assert!(healthy.fallback_stage.is_none());
        assert_eq!(guard.consecutive_breach_count(), 0);
    }

    #[test]
    fn degradation_machine_applies_calibration_fallback_deterministically() {
        let mut machine = DegradationStateMachine::default();
        let mut guard = CalibrationGuardState::new(CalibrationGuardConfig {
            breach_consecutive_required: 1,
            fallback_stage: DegradationStage::LexicalOnly,
            ..CalibrationGuardConfig::default()
        })
        .expect("guard config");

        let decision = guard.evaluate(CalibrationMetrics::new(800, 70.0, 0.90, 1.0, 90.0));
        assert_eq!(decision.fallback_stage, Some(DegradationStage::LexicalOnly));

        let signal = DegradationSignal::new(PressureState::Normal, false, false);
        let fallback = machine
            .apply_calibration_guard(signal, &decision, 99)
            .expect("fallback transition");
        assert_eq!(fallback.from, DegradationStage::Full);
        assert_eq!(fallback.to, DegradationStage::LexicalOnly);
        assert_eq!(fallback.trigger, DegradationTrigger::CalibrationBreach);
        assert_eq!(fallback.reason_code, REASON_DEGRADE_CALIBRATION_FALLBACK);
        assert_eq!(fallback.status.transition_reason_code, decision.reason_code);

        let no_second = machine.apply_calibration_guard(signal, &decision, 100);
        assert!(no_second.is_none());
    }

    #[test]
    fn parse_proc_io_extracts_read_and_write_bytes() {
        let parsed = parse_proc_self_io(
            "rchar: 12\nwchar: 15\nsyscr: 2\nsyscw: 3\nread_bytes: 4096\nwrite_bytes: 8192\n",
        )
        .expect("proc io parse");
        assert_eq!(parsed.read_bytes, 4_096);
        assert_eq!(parsed.write_bytes, 8_192);
    }

    #[test]
    fn parse_proc_io_rejects_missing_fields() {
        let err = parse_proc_self_io("read_bytes: 100\n").expect_err("missing write_bytes");
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn parse_self_status_rss_converts_kb_to_mb() {
        let parsed = parse_self_status_rss_mb("Name:\tfsfs\nVmRSS:\t2048 kB\n").expect("rss parse");
        assert_eq!(parsed, 2);
    }

    #[test]
    fn parse_load_avg_reads_first_field() {
        let parsed = parse_load_avg_1m("1.50 1.21 0.80 2/199 1234\n").expect("loadavg parse");
        assert!((parsed - 1.5).abs() < f64::EPSILON);
    }

    // ─── bd-29u4 tests begin ───

    // --- PressureState ---

    #[test]
    fn pressure_state_severity_ordering() {
        assert_eq!(PressureState::Normal.severity(), 0);
        assert_eq!(PressureState::Constrained.severity(), 1);
        assert_eq!(PressureState::Degraded.severity(), 2);
        assert_eq!(PressureState::Emergency.severity(), 3);
        assert!(PressureState::Normal.severity() < PressureState::Emergency.severity());
    }

    #[test]
    fn pressure_state_serde_roundtrip() {
        for state in [
            PressureState::Normal,
            PressureState::Constrained,
            PressureState::Degraded,
            PressureState::Emergency,
        ] {
            let json = serde_json::to_string(&state).unwrap();
            let back: PressureState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }
    }

    #[test]
    fn pressure_state_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&PressureState::Normal).unwrap(),
            "\"normal\""
        );
        assert_eq!(
            serde_json::to_string(&PressureState::Emergency).unwrap(),
            "\"emergency\""
        );
    }

    // --- DegradationStage ---

    #[test]
    fn degradation_stage_severity_ordering() {
        assert_eq!(DegradationStage::Full.severity(), 0);
        assert_eq!(DegradationStage::EmbedDeferred.severity(), 1);
        assert_eq!(DegradationStage::LexicalOnly.severity(), 2);
        assert_eq!(DegradationStage::MetadataOnly.severity(), 3);
        assert_eq!(DegradationStage::Paused.severity(), 4);
    }

    #[test]
    fn degradation_stage_step_toward_full_all_variants() {
        assert_eq!(
            DegradationStage::Paused.step_toward_full(),
            DegradationStage::MetadataOnly
        );
        assert_eq!(
            DegradationStage::MetadataOnly.step_toward_full(),
            DegradationStage::LexicalOnly
        );
        assert_eq!(
            DegradationStage::LexicalOnly.step_toward_full(),
            DegradationStage::EmbedDeferred
        );
        assert_eq!(
            DegradationStage::EmbedDeferred.step_toward_full(),
            DegradationStage::Full
        );
        assert_eq!(
            DegradationStage::Full.step_toward_full(),
            DegradationStage::Full
        );
    }

    #[test]
    fn degradation_stage_reason_code_all_variants() {
        assert_eq!(
            DegradationStage::Full.stage_reason_code(),
            "degrade.stage.full"
        );
        assert_eq!(
            DegradationStage::EmbedDeferred.stage_reason_code(),
            "degrade.stage.embed_deferred"
        );
        assert_eq!(
            DegradationStage::LexicalOnly.stage_reason_code(),
            "degrade.stage.lexical_only"
        );
        assert_eq!(
            DegradationStage::MetadataOnly.stage_reason_code(),
            "degrade.stage.metadata_only"
        );
        assert_eq!(
            DegradationStage::Paused.stage_reason_code(),
            "degrade.stage.paused"
        );
    }

    #[test]
    fn degradation_stage_recovery_gate_full_is_none() {
        assert!(DegradationStage::Full.recovery_gate(3).is_none());
    }

    #[test]
    fn degradation_stage_recovery_gate_embed_deferred_requires_quality_circuit_closed() {
        let gate = DegradationStage::EmbedDeferred.recovery_gate(3).unwrap();
        assert_eq!(gate.max_pressure_for_recovery, PressureState::Normal);
        assert!(gate.require_quality_circuit_closed);
        assert!(!gate.require_pause_cleared);
        assert_eq!(gate.consecutive_healthy_required, 3);
    }

    #[test]
    fn degradation_stage_recovery_gate_paused_requires_pause_cleared() {
        let gate = DegradationStage::Paused.recovery_gate(5).unwrap();
        assert!(gate.require_pause_cleared);
        assert_eq!(gate.consecutive_healthy_required, 5);
    }

    #[test]
    fn degradation_stage_serde_roundtrip() {
        for stage in [
            DegradationStage::Full,
            DegradationStage::EmbedDeferred,
            DegradationStage::LexicalOnly,
            DegradationStage::MetadataOnly,
            DegradationStage::Paused,
        ] {
            let json = serde_json::to_string(&stage).unwrap();
            let back: DegradationStage = serde_json::from_str(&json).unwrap();
            assert_eq!(back, stage);
        }
    }

    // --- DegradationOverride ---

    #[test]
    fn degradation_override_forced_stage_all_variants() {
        assert_eq!(DegradationOverride::Auto.forced_stage(), None);
        assert_eq!(
            DegradationOverride::ForceFull.forced_stage(),
            Some(DegradationStage::Full)
        );
        assert_eq!(
            DegradationOverride::ForceEmbedDeferred.forced_stage(),
            Some(DegradationStage::EmbedDeferred)
        );
        assert_eq!(
            DegradationOverride::ForceLexicalOnly.forced_stage(),
            Some(DegradationStage::LexicalOnly)
        );
        assert_eq!(
            DegradationOverride::ForceMetadataOnly.forced_stage(),
            Some(DegradationStage::MetadataOnly)
        );
        assert_eq!(
            DegradationOverride::ForcePaused.forced_stage(),
            Some(DegradationStage::Paused)
        );
    }

    #[test]
    fn degradation_override_serde_roundtrip() {
        for over in [
            DegradationOverride::Auto,
            DegradationOverride::ForceFull,
            DegradationOverride::ForceEmbedDeferred,
            DegradationOverride::ForceLexicalOnly,
            DegradationOverride::ForceMetadataOnly,
            DegradationOverride::ForcePaused,
        ] {
            let json = serde_json::to_string(&over).unwrap();
            let back: DegradationOverride = serde_json::from_str(&json).unwrap();
            assert_eq!(back, over);
        }
    }

    // --- normalize_pct ---

    #[test]
    fn normalize_pct_nan_returns_zero() {
        assert!(normalize_pct(f64::NAN).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_pct_negative_returns_zero() {
        assert!(normalize_pct(-5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_pct_negative_infinity_returns_zero() {
        assert!(normalize_pct(f64::NEG_INFINITY).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_pct_positive_infinity_returns_zero() {
        assert!(normalize_pct(f64::INFINITY).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_pct_zero_returns_zero() {
        assert!(normalize_pct(0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_pct_normal_value_passes_through() {
        assert!((normalize_pct(50.0) - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalize_pct_caps_at_200() {
        assert!((normalize_pct(999.0) - 200.0).abs() < f64::EPSILON);
        assert!((normalize_pct(200.0) - 200.0).abs() < f64::EPSILON);
        assert!((normalize_pct(201.0) - 200.0).abs() < f64::EPSILON);
    }

    // --- PressureSignal ---

    #[test]
    fn pressure_signal_new_normalizes_fields() {
        let sig = PressureSignal::new(-1.0, f64::NAN, 999.0, 50.0);
        assert!(sig.cpu_pct.abs() < f64::EPSILON);
        assert!(sig.memory_pct.abs() < f64::EPSILON);
        assert!((sig.io_pct - 200.0).abs() < f64::EPSILON);
        assert!((sig.load_pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_signal_score_returns_max_field() {
        let sig = PressureSignal::new(10.0, 80.0, 30.0, 50.0);
        assert!((sig.score() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_signal_ewma_blends_with_previous() {
        let prev = PressureSignal::new(100.0, 100.0, 100.0, 100.0);
        let curr = PressureSignal::new(0.0, 0.0, 0.0, 0.0);
        let blended = curr.ewma(prev, 0.5);
        // 0.5 * 0.0 + 0.5 * 100.0 = 50.0
        assert!((blended.cpu_pct - 50.0).abs() < f64::EPSILON);
        assert!((blended.memory_pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_signal_ewma_alpha_clamped() {
        let prev = PressureSignal::new(100.0, 100.0, 100.0, 100.0);
        let curr = PressureSignal::new(0.0, 0.0, 0.0, 0.0);
        // alpha > 1.0 clamped to 1.0 → fully current
        let full_current = curr.ewma(prev, 5.0);
        assert!(full_current.cpu_pct.abs() < f64::EPSILON);
        // alpha < 0.0 clamped to 0.0 → fully previous
        let full_prev = curr.ewma(prev, -1.0);
        assert!((full_prev.cpu_pct - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pressure_signal_ewma_nan_alpha_uses_default() {
        let prev = PressureSignal::new(100.0, 100.0, 100.0, 100.0);
        let curr = PressureSignal::new(50.0, 50.0, 50.0, 50.0);
        let blended = curr.ewma(prev, f64::NAN);
        // NaN alpha should fallback to 0.3, so: 0.3 * 50 + 0.7 * 100 = 85.0
        assert!(
            (blended.cpu_pct - 85.0).abs() < 0.01,
            "expected ~85.0, got {}",
            blended.cpu_pct,
        );
    }

    #[test]
    fn pressure_signal_serde_roundtrip() {
        let sig = PressureSignal::new(42.5, 55.0, 10.0, 33.3);
        let json = serde_json::to_string(&sig).unwrap();
        let back: PressureSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sig);
    }

    // --- CalibrationMetrics ---

    #[test]
    fn calibration_metrics_new_normalizes_pct_fields() {
        let m = CalibrationMetrics::new(100, -5.0, 0.5, f64::NAN, 300.0);
        assert!(m.observed_coverage_pct.abs() < f64::EPSILON);
        assert!((m.e_value - 0.5).abs() < f64::EPSILON);
        assert!(m.drift_pct.abs() < f64::EPSILON);
        assert!((m.confidence_pct - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn calibration_metrics_e_value_nan_becomes_zero() {
        let m = CalibrationMetrics::new(100, 95.0, f64::NAN, 5.0, 80.0);
        assert!(m.e_value.abs() < f64::EPSILON);
    }

    #[test]
    fn calibration_metrics_e_value_infinity_becomes_zero() {
        let m = CalibrationMetrics::new(100, 95.0, f64::INFINITY, 5.0, 80.0);
        assert!(m.e_value.abs() < f64::EPSILON);
    }

    #[test]
    fn calibration_metrics_e_value_negative_becomes_zero() {
        let m = CalibrationMetrics::new(100, 95.0, -0.5, 5.0, 80.0);
        assert!(m.e_value.abs() < f64::EPSILON);
    }

    #[test]
    fn calibration_metrics_serde_roundtrip() {
        let m = CalibrationMetrics::new(500, 95.0, 0.10, 4.0, 88.0);
        let json = serde_json::to_string(&m).unwrap();
        let back: CalibrationMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    // --- Config defaults and validation ---

    #[test]
    fn degradation_controller_config_default() {
        let cfg = DegradationControllerConfig::default();
        assert_eq!(cfg.consecutive_healthy_required, 3);
    }

    #[test]
    fn degradation_controller_config_validate_zero_consecutive() {
        let cfg = DegradationControllerConfig {
            consecutive_healthy_required: 0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn degradation_controller_config_validate_ok() {
        let cfg = DegradationControllerConfig {
            consecutive_healthy_required: 1,
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn pressure_controller_config_default_values() {
        let cfg = PressureControllerConfig::default();
        assert_eq!(cfg.profile, PressureProfile::Performance);
        assert!((cfg.ewma_alpha - 0.3).abs() < f64::EPSILON);
        assert!((cfg.hysteresis_pct - 5.0).abs() < f64::EPSILON);
        assert_eq!(cfg.consecutive_required, 3);
    }

    #[test]
    fn pressure_controller_config_validate_alpha_out_of_range() {
        let cfg = PressureControllerConfig {
            ewma_alpha: 1.5,
            ..PressureControllerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(
            matches!(err, SearchError::InvalidConfig { ref field, .. } if field == "pressure.ewma_alpha")
        );
    }

    #[test]
    fn pressure_controller_config_validate_hysteresis_out_of_range() {
        let cfg = PressureControllerConfig {
            hysteresis_pct: 31.0,
            ..PressureControllerConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(
            matches!(err, SearchError::InvalidConfig { ref field, .. } if field == "pressure.hysteresis_pct")
        );
    }

    #[test]
    fn pressure_controller_config_validate_consecutive_zero() {
        let cfg = PressureControllerConfig {
            consecutive_required: 0,
            ..PressureControllerConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_default_values() {
        let cfg = CalibrationGuardConfig::default();
        assert_eq!(cfg.min_sample_count, 200);
        assert!((cfg.target_coverage_pct - 95.0).abs() < f64::EPSILON);
        assert!((cfg.min_e_value - 0.05).abs() < f64::EPSILON);
        assert!((cfg.max_drift_pct - 15.0).abs() < f64::EPSILON);
        assert!((cfg.min_confidence_pct - 70.0).abs() < f64::EPSILON);
        assert_eq!(cfg.breach_consecutive_required, 2);
        assert_eq!(cfg.fallback_stage, DegradationStage::LexicalOnly);
    }

    #[test]
    fn calibration_guard_config_validate_min_sample_count_zero() {
        let cfg = CalibrationGuardConfig {
            min_sample_count: 0,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_validate_target_coverage_out_of_range() {
        let cfg = CalibrationGuardConfig {
            target_coverage_pct: 101.0,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_validate_e_value_negative() {
        let cfg = CalibrationGuardConfig {
            min_e_value: -0.1,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_validate_e_value_nan() {
        let cfg = CalibrationGuardConfig {
            min_e_value: f64::NAN,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_validate_max_drift_out_of_range() {
        let cfg = CalibrationGuardConfig {
            max_drift_pct: -1.0,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_validate_min_confidence_out_of_range() {
        let cfg = CalibrationGuardConfig {
            min_confidence_pct: 101.0,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn calibration_guard_config_validate_breach_consecutive_zero() {
        let cfg = CalibrationGuardConfig {
            breach_consecutive_required: 0,
            ..CalibrationGuardConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- CalibrationGuardState breach kinds ---

    #[test]
    fn calibration_guard_evalue_breach() {
        let mut guard = CalibrationGuardState::default();
        // coverage ok (96 > 95), e_value too low (0.01 < 0.05)
        let d = guard.evaluate(CalibrationMetrics::new(500, 96.0, 0.01, 5.0, 80.0));
        assert_eq!(d.status, CalibrationGuardStatus::Breach);
        assert_eq!(d.reason_code, REASON_CALIBRATION_EVALUE_BREACH);
        assert_eq!(
            d.breach_kind,
            Some(CalibrationBreachKind::EValueBelowThreshold)
        );
    }

    #[test]
    fn calibration_guard_drift_breach() {
        let mut guard = CalibrationGuardState::default();
        // coverage ok, e_value ok (0.10 > 0.05), drift too high (20 > 15)
        let d = guard.evaluate(CalibrationMetrics::new(500, 96.0, 0.10, 20.0, 80.0));
        assert_eq!(d.status, CalibrationGuardStatus::Breach);
        assert_eq!(d.reason_code, REASON_CALIBRATION_DRIFT_BREACH);
        assert_eq!(
            d.breach_kind,
            Some(CalibrationBreachKind::DriftAboveThreshold)
        );
    }

    #[test]
    fn calibration_guard_confidence_breach() {
        let mut guard = CalibrationGuardState::default();
        // coverage ok, e_value ok, drift ok (5 < 15), confidence too low (60 < 70)
        let d = guard.evaluate(CalibrationMetrics::new(500, 96.0, 0.10, 5.0, 60.0));
        assert_eq!(d.status, CalibrationGuardStatus::Breach);
        assert_eq!(d.reason_code, REASON_CALIBRATION_CONFIDENCE_BREACH);
        assert_eq!(
            d.breach_kind,
            Some(CalibrationBreachKind::ConfidenceBelowThreshold)
        );
    }

    #[test]
    fn calibration_guard_state_new_with_invalid_config() {
        let cfg = CalibrationGuardConfig {
            breach_consecutive_required: 0,
            ..CalibrationGuardConfig::default()
        };
        assert!(CalibrationGuardState::new(cfg).is_err());
    }

    #[test]
    fn calibration_guard_state_config_accessor() {
        let guard = CalibrationGuardState::default();
        assert_eq!(guard.config().min_sample_count, 200);
    }

    // --- HostPressureCollector ---

    #[test]
    fn host_pressure_collector_default() {
        let c = HostPressureCollector::default();
        assert!((c.io_ceiling_mib_per_sec - 64.0).abs() < f64::EPSILON);
        assert!(c.previous_io.is_none());
    }

    #[test]
    fn host_pressure_collector_new_rejects_zero() {
        assert!(HostPressureCollector::new(0.0).is_err());
    }

    #[test]
    fn host_pressure_collector_new_rejects_negative() {
        assert!(HostPressureCollector::new(-10.0).is_err());
    }

    #[test]
    fn host_pressure_collector_new_rejects_nan() {
        assert!(HostPressureCollector::new(f64::NAN).is_err());
    }

    #[test]
    fn host_pressure_collector_new_rejects_infinity() {
        assert!(HostPressureCollector::new(f64::INFINITY).is_err());
    }

    #[test]
    fn host_pressure_collector_new_valid() {
        let c = HostPressureCollector::new(128.0).unwrap();
        assert!((c.io_ceiling_mib_per_sec - 128.0).abs() < f64::EPSILON);
    }

    #[test]
    fn io_pct_no_previous_returns_zero() {
        let c = HostPressureCollector::default();
        let current = ProcIoCounters {
            read_bytes: 1_000_000,
            write_bytes: 1_000_000,
        };
        assert!(c.io_pct_for_interval(Duration::from_secs(1), current).abs() < f64::EPSILON);
    }

    #[test]
    fn io_pct_zero_elapsed_returns_zero() {
        let c = HostPressureCollector {
            io_ceiling_mib_per_sec: 64.0,
            previous_io: Some(ProcIoCounters {
                read_bytes: 0,
                write_bytes: 0,
            }),
        };
        let current = ProcIoCounters {
            read_bytes: 1_000_000,
            write_bytes: 1_000_000,
        };
        assert!(c.io_pct_for_interval(Duration::ZERO, current).abs() < f64::EPSILON);
    }

    // --- parse functions ---

    #[test]
    fn parse_proc_self_io_missing_read_bytes() {
        let err = parse_proc_self_io("write_bytes: 100\n").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn parse_proc_self_io_invalid_value() {
        let err = parse_proc_self_io("read_bytes: abc\nwrite_bytes: 100\n").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn parse_self_status_rss_missing_vmrss() {
        let err = parse_self_status_rss_mb("Name:\tfsfs\nVmSize:\t1000 kB\n").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn parse_load_avg_empty() {
        let err = parse_load_avg_1m("").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn parse_load_avg_invalid_value() {
        let err = parse_load_avg_1m("abc 1.0 0.5\n").unwrap_err();
        assert!(matches!(err, SearchError::InvalidConfig { .. }));
    }

    #[test]
    fn parse_u64_field_valid() {
        assert_eq!(parse_u64_field("test", "42").unwrap(), 42);
    }

    #[test]
    fn parse_u64_field_invalid() {
        assert!(parse_u64_field("test", "not_a_number").is_err());
    }

    // --- RecoveryGate ---

    #[test]
    fn recovery_gate_satisfied_when_pressure_at_max() {
        let gate = RecoveryGate {
            max_pressure_for_recovery: PressureState::Constrained,
            require_quality_circuit_closed: false,
            require_pause_cleared: false,
            consecutive_healthy_required: 3,
        };
        let signal = DegradationSignal::new(PressureState::Constrained, false, false);
        assert!(gate.is_satisfied(signal));
    }

    #[test]
    fn recovery_gate_not_satisfied_when_pressure_too_high() {
        let gate = RecoveryGate {
            max_pressure_for_recovery: PressureState::Normal,
            require_quality_circuit_closed: false,
            require_pause_cleared: false,
            consecutive_healthy_required: 3,
        };
        let signal = DegradationSignal::new(PressureState::Constrained, false, false);
        assert!(!gate.is_satisfied(signal));
    }

    #[test]
    fn recovery_gate_not_satisfied_when_quality_circuit_open() {
        let gate = RecoveryGate {
            max_pressure_for_recovery: PressureState::Normal,
            require_quality_circuit_closed: true,
            require_pause_cleared: false,
            consecutive_healthy_required: 3,
        };
        let signal = DegradationSignal::new(PressureState::Normal, true, false);
        assert!(!gate.is_satisfied(signal));
    }

    #[test]
    fn recovery_gate_not_satisfied_when_pause_not_cleared() {
        let gate = RecoveryGate {
            max_pressure_for_recovery: PressureState::Degraded,
            require_quality_circuit_closed: false,
            require_pause_cleared: true,
            consecutive_healthy_required: 3,
        };
        let signal = DegradationSignal::new(PressureState::Normal, false, true);
        assert!(!gate.is_satisfied(signal));
    }

    // --- DegradationSignal ---

    #[test]
    fn degradation_signal_serde_roundtrip() {
        let sig = DegradationSignal::new(PressureState::Degraded, true, false);
        let json = serde_json::to_string(&sig).unwrap();
        let back: DegradationSignal = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sig);
    }

    // --- DegradationStateMachine ---

    #[test]
    fn degradation_state_machine_new_with_invalid_config() {
        let cfg = DegradationControllerConfig {
            consecutive_healthy_required: 0,
        };
        assert!(DegradationStateMachine::new(cfg).is_err());
    }

    #[test]
    fn degradation_state_machine_accessors() {
        let machine = DegradationStateMachine::default();
        assert_eq!(machine.stage(), DegradationStage::Full);
        assert_eq!(machine.override_mode(), DegradationOverride::Auto);
    }

    // --- PressureController ---

    #[test]
    fn pressure_controller_new_with_invalid_config() {
        let cfg = PressureControllerConfig {
            consecutive_required: 0,
            ..PressureControllerConfig::default()
        };
        assert!(PressureController::new(cfg).is_err());
    }

    #[test]
    fn pressure_controller_config_accessor() {
        let c = PressureController::from_profile(PressureProfile::Strict);
        assert_eq!(c.config().profile, PressureProfile::Strict);
    }

    #[test]
    fn pressure_controller_from_profile_strict() {
        let c = PressureController::from_profile(PressureProfile::Strict);
        assert_eq!(c.state(), PressureState::Normal);
        assert_eq!(c.config().profile, PressureProfile::Strict);
    }

    #[test]
    fn pressure_controller_from_profile_degraded() {
        let c = PressureController::from_profile(PressureProfile::Degraded);
        assert_eq!(c.state(), PressureState::Normal);
        assert_eq!(c.config().profile, PressureProfile::Degraded);
    }

    // --- DegradationTrigger serde ---

    #[test]
    fn degradation_trigger_serde_roundtrip() {
        for trigger in [
            DegradationTrigger::Stable,
            DegradationTrigger::PressureEscalation,
            DegradationTrigger::QualityCircuitOpen,
            DegradationTrigger::HardPause,
            DegradationTrigger::Recovery,
            DegradationTrigger::CalibrationBreach,
            DegradationTrigger::OperatorOverride,
        ] {
            let json = serde_json::to_string(&trigger).unwrap();
            let back: DegradationTrigger = serde_json::from_str(&json).unwrap();
            assert_eq!(back, trigger);
        }
    }

    // --- CalibrationGuardStatus / CalibrationBreachKind serde ---

    #[test]
    fn calibration_guard_status_serde_roundtrip() {
        for status in [
            CalibrationGuardStatus::Healthy,
            CalibrationGuardStatus::Watch,
            CalibrationGuardStatus::Breach,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: CalibrationGuardStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn calibration_breach_kind_serde_roundtrip() {
        for kind in [
            CalibrationBreachKind::CoverageBelowTarget,
            CalibrationBreachKind::EValueBelowThreshold,
            CalibrationBreachKind::DriftAboveThreshold,
            CalibrationBreachKind::ConfidenceBelowThreshold,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: CalibrationBreachKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    // --- Capability enums serde ---

    #[test]
    fn query_capability_mode_serde_roundtrip() {
        for mode in [
            QueryCapabilityMode::Hybrid,
            QueryCapabilityMode::LexicalOnly,
            QueryCapabilityMode::MetadataOnly,
            QueryCapabilityMode::Paused,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: QueryCapabilityMode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, mode);
        }
    }

    #[test]
    fn indexing_capability_mode_serde_roundtrip() {
        for mode in [
            IndexingCapabilityMode::Full,
            IndexingCapabilityMode::DeferEmbedding,
            IndexingCapabilityMode::MetadataOnly,
            IndexingCapabilityMode::Paused,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: IndexingCapabilityMode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, mode);
        }
    }

    #[test]
    fn capability_state_serde_roundtrip() {
        for state in [CapabilityState::Enabled, CapabilityState::Disabled] {
            let json = serde_json::to_string(&state).unwrap();
            let back: CapabilityState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }
    }

    #[test]
    fn write_state_serde_roundtrip() {
        for state in [WriteState::Enabled, WriteState::Paused] {
            let json = serde_json::to_string(&state).unwrap();
            let back: WriteState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }
    }

    #[test]
    fn integrity_state_serde_roundtrip() {
        let json = serde_json::to_string(&IntegrityState::Preserved).unwrap();
        let back: IntegrityState = serde_json::from_str(&json).unwrap();
        assert_eq!(back, IntegrityState::Preserved);
    }

    // --- DegradationContract embed_deferred and metadata_only ---

    #[test]
    fn degradation_contract_embed_deferred() {
        let c = DegradationStage::EmbedDeferred.contract();
        assert_eq!(c.query_mode, QueryCapabilityMode::Hybrid);
        assert_eq!(c.indexing_mode, IndexingCapabilityMode::DeferEmbedding);
        assert_eq!(c.semantic_search, CapabilityState::Enabled);
        assert_eq!(c.writes, WriteState::Enabled);
    }

    #[test]
    fn degradation_contract_metadata_only() {
        let c = DegradationStage::MetadataOnly.contract();
        assert_eq!(c.query_mode, QueryCapabilityMode::MetadataOnly);
        assert_eq!(c.indexing_mode, IndexingCapabilityMode::MetadataOnly);
        assert_eq!(c.semantic_search, CapabilityState::Disabled);
        assert_eq!(c.lexical_search, CapabilityState::Disabled);
        assert_eq!(c.writes, WriteState::Enabled);
    }

    // --- DegradationSignal auto_target_stage edge case ---

    #[test]
    fn degradation_signal_constrained_with_quality_circuit_stays_embed_deferred() {
        // quality_circuit_open + constrained → still EmbedDeferred (constrained dominates)
        let sig = DegradationSignal::new(PressureState::Constrained, true, false);
        assert_eq!(sig.auto_target_stage(), DegradationStage::EmbedDeferred);
    }

    // --- PressureController stable reason code ---

    #[test]
    fn pressure_controller_stable_when_already_at_target() {
        let mut controller = PressureController::from_profile(PressureProfile::Performance);
        let low = PressureSignal::new(10.0, 10.0, 10.0, 10.0);
        let t = controller.observe(low, 1);
        assert_eq!(t.reason_code, "pressure.state.stable");
        assert!(!t.changed);
        assert_eq!(t.from, PressureState::Normal);
        assert_eq!(t.to, PressureState::Normal);
    }

    // --- Quality circuit open trigger identification ---

    #[test]
    fn degradation_machine_quality_circuit_trigger() {
        let mut machine = DegradationStateMachine::default();
        let sig = DegradationSignal::new(PressureState::Normal, true, false);
        let t = machine.observe(sig, 1);
        assert_eq!(t.to, DegradationStage::EmbedDeferred);
        assert_eq!(t.trigger, DegradationTrigger::QualityCircuitOpen);
    }

    // --- Calibration guard fallback not applied when already at higher severity ---

    #[test]
    fn calibration_fallback_skipped_when_already_more_severe() {
        let mut machine = DegradationStateMachine::default();
        // First escalate to MetadataOnly
        let sig = DegradationSignal::new(PressureState::Emergency, false, false);
        let _ = machine.observe(sig, 1);
        assert_eq!(machine.stage(), DegradationStage::MetadataOnly);

        // Calibration fallback to LexicalOnly (less severe) → should be skipped
        let mut guard = CalibrationGuardState::new(CalibrationGuardConfig {
            breach_consecutive_required: 1,
            fallback_stage: DegradationStage::LexicalOnly,
            ..CalibrationGuardConfig::default()
        })
        .unwrap();
        let decision = guard.evaluate(CalibrationMetrics::new(800, 70.0, 0.90, 1.0, 90.0));
        let result = machine.apply_calibration_guard(sig, &decision, 99);
        assert!(result.is_none());
    }

    // ─── bd-29u4 tests end ───

    #[test]
    fn io_pct_is_derived_from_byte_delta_and_interval() {
        let collector = HostPressureCollector {
            io_ceiling_mib_per_sec: 10.0,
            previous_io: Some(ProcIoCounters {
                read_bytes: 0,
                write_bytes: 0,
            }),
        };
        let current = ProcIoCounters {
            read_bytes: 5 * 1024 * 1024,
            write_bytes: 5 * 1024 * 1024,
        };
        let io_pct = collector.io_pct_for_interval(Duration::from_secs(1), current);
        assert!((io_pct - 100.0).abs() < f64::EPSILON);
    }
}
