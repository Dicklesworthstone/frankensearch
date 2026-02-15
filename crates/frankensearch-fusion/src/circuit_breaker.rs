//! Quality-tier circuit breaker for the two-tier search pipeline.
//!
//! When the quality tier (Phase 2) is consistently slow, failing, or not
//! improving results, the circuit breaker trips and subsequent queries skip
//! Phase 2 entirely, returning only Phase 1 (fast) results.
//!
//! # State Machine
//!
//! ```text
//!   Closed ──(failure_threshold consecutive failures)──> Open
//!   Open ──(half_open_interval elapsed)──> HalfOpen
//!   HalfOpen ──(reset_threshold consecutive successes)──> Closed
//!   HalfOpen ──(any failure)──> Open
//! ```
//!
//! # Failure Conditions
//!
//! Any of these counts as a quality-tier failure:
//! 1. Quality tier exceeds `latency_threshold_ms`.
//! 2. Quality tier returns an error.
//! 3. Kendall tau improvement < `improvement_threshold` (quality didn't help).
//!
//! # Performance
//!
//! The circuit breaker adds <1 microsecond per query (one atomic load + comparison).
//! When open, it saves ~128ms per query by skipping quality-tier embedding.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use frankensearch_core::decision_plane::{
    EvidenceEventType, EvidenceRecord, PipelineAction, PipelineState, ReasonCode, Severity,
};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the quality-tier circuit breaker.
///
/// All thresholds are configurable to allow tuning for different workloads.
/// The defaults are conservative: 5 consecutive failures before tripping,
/// 30-second cooldown, 3 successes to reset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Whether the circuit breaker is enabled. When disabled, quality tier
    /// is always attempted (unless `fast_only` is set in `TwoTierConfig`).
    /// Default: `true`.
    pub enabled: bool,

    /// Number of consecutive failures required to trip the circuit breaker.
    /// Default: 5.
    pub failure_threshold: u32,

    /// Quality-tier latency (ms) above which a query counts as "slow".
    /// Default: 500.
    pub latency_threshold_ms: u64,

    /// Minimum Kendall tau improvement for quality tier to count as "useful".
    /// If the quality tier re-ranking produces tau < this vs fast-only results,
    /// it counts as a failure (quality didn't materially improve ranking).
    /// Default: 0.05.
    pub improvement_threshold: f64,

    /// Milliseconds to wait before transitioning from `Open` to `HalfOpen`.
    /// Default: 30,000 (30 seconds).
    pub half_open_interval_ms: u64,

    /// Number of consecutive successes in `HalfOpen` state to fully close
    /// the circuit breaker.
    /// Default: 3.
    pub reset_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            latency_threshold_ms: 500,
            improvement_threshold: 0.05,
            half_open_interval_ms: 30_000,
            reset_threshold: 3,
        }
    }
}

// ─── State ──────────────────────────────────────────────────────────────────

/// Internal circuit breaker state, stored as an atomic u8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum CircuitState {
    /// Normal: quality tier is attempted for every query.
    Closed = 0,
    /// Tripped: quality tier is skipped.
    Open = 1,
    /// Testing: quality tier is tried on the next query to probe recovery.
    HalfOpen = 2,
}

impl CircuitState {
    const fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Open,
            2 => Self::HalfOpen,
            _ => Self::Closed,
        }
    }
}

// ─── Metrics ────────────────────────────────────────────────────────────────

/// Observable metrics from the circuit breaker.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitMetrics {
    /// Total number of times the circuit breaker has tripped (Closed → Open).
    pub trips: u64,
    /// Total number of times the circuit breaker has reset (`HalfOpen` → `Closed`).
    pub resets: u64,
    /// Total number of queries where quality tier was skipped due to open circuit.
    pub queries_skipped: u64,
    /// Total number of probe queries attempted in `HalfOpen` state.
    pub probes_attempted: u64,
    /// Total number of probe queries that succeeded.
    pub probes_succeeded: u64,
}

// ─── Outcome ────────────────────────────────────────────────────────────────

/// Result of a quality-tier attempt, used to feed the circuit breaker.
#[derive(Debug)]
pub enum QualityOutcome {
    /// Quality tier completed successfully within latency budget,
    /// and produced meaningful rank improvement.
    Success {
        /// Latency of the quality tier in milliseconds.
        latency_ms: u64,
        /// Kendall tau improvement over fast-only results.
        tau_improvement: f64,
    },
    /// Quality tier completed but was too slow.
    Slow {
        /// Actual latency in milliseconds.
        latency_ms: u64,
    },
    /// Quality tier completed but didn't improve rankings.
    NotUseful {
        /// Kendall tau improvement (below threshold).
        tau_improvement: f64,
    },
    /// Quality tier returned an error.
    Error,
}

// ─── Circuit Breaker ────────────────────────────────────────────────────────

/// Quality-tier circuit breaker with atomic state transitions.
///
/// Thread-safe: all state transitions use atomic operations. Multiple search
/// threads can call `should_skip_quality()` and `record_outcome()` concurrently
/// without locks.
///
/// # Evidence Records
///
/// Every state transition emits an [`EvidenceRecord`] via the returned
/// `Vec<EvidenceRecord>`. The caller is responsible for forwarding these
/// to the evidence ledger or tracing subscriber.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    /// Current state: 0=Closed, 1=Open, 2=HalfOpen.
    state: AtomicU32,
    /// Consecutive failure count.
    consecutive_failures: AtomicU32,
    /// Consecutive success count (used in `HalfOpen` state).
    consecutive_successes: AtomicU32,
    /// Timestamp (ms since epoch-ish) when the circuit was last tripped.
    last_trip_time_ms: AtomicU64,
    /// Reference instant for computing elapsed time.
    epoch: Instant,
    /// Trip count (for metrics).
    trip_count: AtomicU64,
    /// Reset count (for metrics).
    reset_count: AtomicU64,
    /// Queries skipped count.
    skip_count: AtomicU64,
    /// Probes attempted.
    probe_count: AtomicU64,
    /// Probes succeeded.
    probe_success_count: AtomicU64,
}

impl CircuitBreaker {
    /// Creates a new circuit breaker with the given configuration.
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: AtomicU32::new(CircuitState::Closed as u32),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            last_trip_time_ms: AtomicU64::new(0),
            epoch: Instant::now(),
            trip_count: AtomicU64::new(0),
            reset_count: AtomicU64::new(0),
            skip_count: AtomicU64::new(0),
            probe_count: AtomicU64::new(0),
            probe_success_count: AtomicU64::new(0),
        }
    }

    /// Creates a circuit breaker with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Check whether the quality tier should be skipped for this query.
    ///
    /// Returns `true` if the circuit is open (quality tier should be skipped),
    /// along with any evidence records for state transitions.
    ///
    /// When the circuit is Open and the half-open interval has elapsed,
    /// this method transitions to `HalfOpen` and returns `false` to allow
    /// a single probe query.
    pub fn should_skip_quality(&self) -> (bool, Vec<EvidenceRecord>) {
        if !self.config.enabled {
            return (false, Vec::new());
        }

        let state = self.current_state();
        match state {
            CircuitState::Closed => (false, Vec::new()),

            CircuitState::Open => {
                let elapsed_ms = self.elapsed_ms();
                let trip_time = self.last_trip_time_ms.load(Ordering::Acquire);

                if elapsed_ms.saturating_sub(trip_time) >= self.config.half_open_interval_ms {
                    // Transition Open → HalfOpen for a probe
                    self.state
                        .store(CircuitState::HalfOpen as u32, Ordering::Release);
                    self.consecutive_successes.store(0, Ordering::Release);
                    self.probe_count.fetch_add(1, Ordering::Relaxed);

                    let evidence = EvidenceRecord::new(
                        EvidenceEventType::Transition,
                        ReasonCode::DECISION_PROBE_SENT,
                        "Circuit breaker half-open interval elapsed; sending probe query",
                        Severity::Info,
                        PipelineState::Probing,
                        "circuit_breaker",
                    )
                    .with_action(PipelineAction::ProbeQuality);

                    (false, vec![evidence])
                } else {
                    self.skip_count.fetch_add(1, Ordering::Relaxed);
                    (true, Vec::new())
                }
            }

            CircuitState::HalfOpen => {
                // Allow the probe query through
                (false, Vec::new())
            }
        }
    }

    /// Record the outcome of a quality-tier attempt.
    ///
    /// Call this after every quality-tier operation (whether it succeeded or
    /// failed) to update the circuit breaker state machine.
    ///
    /// Returns evidence records for any state transitions.
    pub fn record_outcome(&self, outcome: &QualityOutcome) -> Vec<EvidenceRecord> {
        if !self.config.enabled {
            return Vec::new();
        }

        let is_failure = match outcome {
            QualityOutcome::Success {
                latency_ms,
                tau_improvement,
            } => {
                *latency_ms > self.config.latency_threshold_ms
                    || *tau_improvement < self.config.improvement_threshold
            }
            QualityOutcome::Slow { .. }
            | QualityOutcome::NotUseful { .. }
            | QualityOutcome::Error => true,
        };

        let state = self.current_state();

        match state {
            CircuitState::Closed => {
                if is_failure {
                    let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
                    if failures >= self.config.failure_threshold {
                        return self.trip();
                    }
                } else {
                    // Reset failure count on success
                    self.consecutive_failures.store(0, Ordering::Release);
                }
                Vec::new()
            }

            CircuitState::HalfOpen => {
                if is_failure {
                    // Probe failed → back to Open
                    self.state
                        .store(CircuitState::Open as u32, Ordering::Release);
                    self.last_trip_time_ms
                        .store(self.elapsed_ms(), Ordering::Release);
                    self.consecutive_failures.store(0, Ordering::Release);

                    let evidence = EvidenceRecord::new(
                        EvidenceEventType::Transition,
                        ReasonCode::DECISION_PROBE_FAILURE,
                        "Probe query failed; circuit breaker re-opened",
                        Severity::Warn,
                        PipelineState::CircuitOpen,
                        "circuit_breaker",
                    )
                    .with_action(PipelineAction::OpenCircuit);

                    vec![evidence]
                } else {
                    self.probe_success_count.fetch_add(1, Ordering::Relaxed);
                    let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;
                    if successes >= self.config.reset_threshold {
                        return self.reset();
                    }
                    Vec::new()
                }
            }

            CircuitState::Open => {
                // Shouldn't normally happen (quality tier was skipped),
                // but be defensive.
                Vec::new()
            }
        }
    }

    /// Get the current state of the circuit breaker.
    #[must_use]
    pub fn pipeline_state(&self) -> PipelineState {
        match self.current_state() {
            CircuitState::Closed => PipelineState::Nominal,
            CircuitState::Open => PipelineState::CircuitOpen,
            CircuitState::HalfOpen => PipelineState::Probing,
        }
    }

    /// Whether the circuit is currently open (quality tier should be skipped).
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.current_state() == CircuitState::Open
    }

    /// Whether the circuit is in half-open (probing) state.
    #[must_use]
    pub fn is_half_open(&self) -> bool {
        self.current_state() == CircuitState::HalfOpen
    }

    /// Whether the circuit is closed (normal operation).
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.current_state() == CircuitState::Closed
    }

    /// Snapshot of observable circuit breaker metrics.
    #[must_use]
    pub fn metrics(&self) -> CircuitMetrics {
        CircuitMetrics {
            trips: self.trip_count.load(Ordering::Relaxed),
            resets: self.reset_count.load(Ordering::Relaxed),
            queries_skipped: self.skip_count.load(Ordering::Relaxed),
            probes_attempted: self.probe_count.load(Ordering::Relaxed),
            probes_succeeded: self.probe_success_count.load(Ordering::Relaxed),
        }
    }

    /// Current configuration.
    #[must_use]
    pub const fn config(&self) -> &CircuitBreakerConfig {
        &self.config
    }

    /// Manually trip the circuit breaker (for testing or operator override).
    ///
    /// Returns evidence records for the state transition.
    pub fn force_open(&self) -> Vec<EvidenceRecord> {
        self.trip()
    }

    /// Manually reset the circuit breaker (for testing or operator override).
    ///
    /// Returns evidence records for the state transition.
    pub fn force_close(&self) -> Vec<EvidenceRecord> {
        self.reset()
    }

    // ─── Internal ───────────────────────────────────────────────────

    fn current_state(&self) -> CircuitState {
        CircuitState::from_u32(self.state.load(Ordering::Acquire))
    }

    #[allow(clippy::cast_possible_truncation)] // u128→u64: uptime >584M years would truncate
    fn elapsed_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn trip(&self) -> Vec<EvidenceRecord> {
        self.state
            .store(CircuitState::Open as u32, Ordering::Release);
        self.last_trip_time_ms
            .store(self.elapsed_ms(), Ordering::Release);
        self.consecutive_failures.store(0, Ordering::Release);
        self.trip_count.fetch_add(1, Ordering::Relaxed);

        let evidence = EvidenceRecord::new(
            EvidenceEventType::Transition,
            ReasonCode::CIRCUIT_OPEN_FAILURES,
            format!(
                "Quality tier failed {} consecutive times; circuit breaker tripped",
                self.config.failure_threshold,
            ),
            Severity::Warn,
            PipelineState::CircuitOpen,
            "circuit_breaker",
        )
        .with_action(PipelineAction::OpenCircuit);

        tracing::warn!(
            failures = self.config.failure_threshold,
            "circuit breaker tripped: quality tier disabled"
        );

        vec![evidence]
    }

    fn reset(&self) -> Vec<EvidenceRecord> {
        self.state
            .store(CircuitState::Closed as u32, Ordering::Release);
        self.consecutive_failures.store(0, Ordering::Release);
        self.consecutive_successes.store(0, Ordering::Release);
        self.reset_count.fetch_add(1, Ordering::Relaxed);

        let evidence = EvidenceRecord::new(
            EvidenceEventType::Transition,
            ReasonCode::CIRCUIT_CLOSE_RECOVERY,
            format!(
                "Quality tier recovered after {} consecutive successful probes; circuit breaker reset",
                self.config.reset_threshold,
            ),
            Severity::Info,
            PipelineState::Nominal,
            "circuit_breaker",
        )
        .with_action(PipelineAction::CloseCircuit);

        tracing::info!("circuit breaker reset: quality tier re-enabled");

        vec![evidence]
    }
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.current_state())
            .field(
                "consecutive_failures",
                &self.consecutive_failures.load(Ordering::Relaxed),
            )
            .field(
                "consecutive_successes",
                &self.consecutive_successes.load(Ordering::Relaxed),
            )
            .field("trips", &self.trip_count.load(Ordering::Relaxed))
            .field("resets", &self.reset_count.load(Ordering::Relaxed))
            .field("queries_skipped", &self.skip_count.load(Ordering::Relaxed))
            .field(
                "probes_attempted",
                &self.probe_count.load(Ordering::Relaxed),
            )
            .field(
                "probes_succeeded",
                &self.probe_success_count.load(Ordering::Relaxed),
            )
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            enabled: true,
            failure_threshold: 3,
            latency_threshold_ms: 200,
            improvement_threshold: 0.05,
            half_open_interval_ms: 50, // Short for testing
            reset_threshold: 2,
        }
    }

    // ─── Initial State ──────────────────────────────────────────────

    #[test]
    fn initial_state_is_closed() {
        let cb = CircuitBreaker::new(test_config());
        assert!(cb.is_closed());
        assert!(!cb.is_open());
        assert!(!cb.is_half_open());
        assert_eq!(cb.pipeline_state(), PipelineState::Nominal);
    }

    #[test]
    fn closed_does_not_skip_quality() {
        let cb = CircuitBreaker::new(test_config());
        let (skip, evidence) = cb.should_skip_quality();
        assert!(!skip);
        assert!(evidence.is_empty());
    }

    // ─── Tripping ───────────────────────────────────────────────────

    #[test]
    fn trips_after_consecutive_failures() {
        let cb = CircuitBreaker::new(test_config());

        // Record failures below threshold
        for _ in 0..2 {
            let evidence = cb.record_outcome(&QualityOutcome::Error);
            assert!(evidence.is_empty());
            assert!(cb.is_closed());
        }

        // Third failure trips the breaker
        let evidence = cb.record_outcome(&QualityOutcome::Error);
        assert!(!evidence.is_empty());
        assert!(cb.is_open());
        assert_eq!(cb.pipeline_state(), PipelineState::CircuitOpen);
        assert_eq!(
            evidence[0].reason_code.as_str(),
            "circuit.open.consecutive_failures"
        );
    }

    #[test]
    fn success_resets_failure_count() {
        let cb = CircuitBreaker::new(test_config());

        // Two failures
        cb.record_outcome(&QualityOutcome::Error);
        cb.record_outcome(&QualityOutcome::Error);

        // One success resets the counter
        cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 100,
            tau_improvement: 0.1,
        });

        // Two more failures shouldn't trip (counter was reset)
        cb.record_outcome(&QualityOutcome::Error);
        cb.record_outcome(&QualityOutcome::Error);
        assert!(cb.is_closed());

        // Third failure after reset trips
        let evidence = cb.record_outcome(&QualityOutcome::Error);
        assert!(!evidence.is_empty());
        assert!(cb.is_open());
    }

    #[test]
    fn slow_latency_counts_as_failure() {
        let cb = CircuitBreaker::new(test_config());

        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Slow { latency_ms: 300 });
        }
        assert!(cb.is_open());
    }

    #[test]
    fn low_tau_improvement_counts_as_failure() {
        let cb = CircuitBreaker::new(test_config());

        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::NotUseful {
                tau_improvement: 0.01,
            });
        }
        assert!(cb.is_open());
    }

    #[test]
    fn success_with_high_latency_counts_as_failure() {
        let cb = CircuitBreaker::new(test_config());

        // Success outcome but latency exceeds threshold
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Success {
                latency_ms: 300, // > 200ms threshold
                tau_improvement: 0.2,
            });
        }
        assert!(cb.is_open());
    }

    #[test]
    fn success_with_low_tau_counts_as_failure() {
        let cb = CircuitBreaker::new(test_config());

        // Success outcome but tau below threshold
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Success {
                latency_ms: 100,
                tau_improvement: 0.01, // < 0.05 threshold
            });
        }
        assert!(cb.is_open());
    }

    // ─── Open State ─────────────────────────────────────────────────

    #[test]
    fn open_skips_quality() {
        let cb = CircuitBreaker::new(test_config());

        // Trip the breaker
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        assert!(cb.is_open());

        // Should skip quality
        let (skip, _) = cb.should_skip_quality();
        assert!(skip);
    }

    // ─── Half-Open Transition ───────────────────────────────────────

    #[test]
    fn transitions_to_half_open_after_interval() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10, // Very short for testing
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Trip the breaker
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        assert!(cb.is_open());

        // Wait for the half-open interval
        std::thread::sleep(std::time::Duration::from_millis(20));

        // Should transition to half-open
        let (skip, evidence) = cb.should_skip_quality();
        assert!(!skip);
        assert!(cb.is_half_open());
        assert!(!evidence.is_empty());
        assert_eq!(evidence[0].pipeline_state, PipelineState::Probing);
    }

    // ─── Half-Open Recovery ─────────────────────────────────────────

    #[test]
    fn half_open_recovers_after_successes() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            reset_threshold: 2,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Trip the breaker
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }

        // Wait and transition to half-open
        std::thread::sleep(std::time::Duration::from_millis(20));
        let (skip, _) = cb.should_skip_quality();
        assert!(!skip);
        assert!(cb.is_half_open());

        // First success
        let evidence = cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 100,
            tau_improvement: 0.1,
        });
        assert!(evidence.is_empty()); // Not enough successes yet
        assert!(cb.is_half_open());

        // Second success resets
        let evidence = cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 100,
            tau_improvement: 0.1,
        });
        assert!(!evidence.is_empty());
        assert!(cb.is_closed());
        assert_eq!(evidence[0].reason_code.as_str(), "circuit.close.recovery");
    }

    #[test]
    fn half_open_failure_reopens_circuit() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Trip the breaker
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }

        // Wait and transition to half-open
        std::thread::sleep(std::time::Duration::from_millis(20));
        cb.should_skip_quality();
        assert!(cb.is_half_open());

        // Failure in half-open → back to open
        let evidence = cb.record_outcome(&QualityOutcome::Error);
        assert!(cb.is_open());
        assert!(!evidence.is_empty());
        assert_eq!(evidence[0].reason_code.as_str(), "decision.probe.failure");
    }

    // ─── Disabled ───────────────────────────────────────────────────

    #[test]
    fn disabled_never_skips() {
        let config = CircuitBreakerConfig {
            enabled: false,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Record many failures
        for _ in 0..10 {
            let evidence = cb.record_outcome(&QualityOutcome::Error);
            assert!(evidence.is_empty());
        }

        let (skip, _) = cb.should_skip_quality();
        assert!(!skip);
    }

    // ─── Metrics ────────────────────────────────────────────────────

    #[test]
    fn metrics_track_trips_and_skips() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Initial metrics
        let m = cb.metrics();
        assert_eq!(m.trips, 0);
        assert_eq!(m.queries_skipped, 0);

        // Trip the breaker
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        assert_eq!(cb.metrics().trips, 1);

        // Skip some queries
        cb.should_skip_quality();
        cb.should_skip_quality();
        assert_eq!(cb.metrics().queries_skipped, 2);
    }

    #[test]
    fn metrics_track_probes() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            reset_threshold: 2,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Trip
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }

        // Wait and probe
        std::thread::sleep(std::time::Duration::from_millis(20));
        cb.should_skip_quality();
        assert_eq!(cb.metrics().probes_attempted, 1);

        // Successful probe
        cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 100,
            tau_improvement: 0.1,
        });
        assert_eq!(cb.metrics().probes_succeeded, 1);
    }

    #[test]
    fn metrics_track_resets() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            reset_threshold: 1,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Trip and recover
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }

        std::thread::sleep(std::time::Duration::from_millis(20));
        cb.should_skip_quality();
        cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 100,
            tau_improvement: 0.1,
        });

        assert_eq!(cb.metrics().resets, 1);
        assert_eq!(cb.metrics().trips, 1);
    }

    // ─── Force Operations ───────────────────────────────────────────

    #[test]
    fn force_open_trips_immediately() {
        let cb = CircuitBreaker::new(test_config());
        assert!(cb.is_closed());

        let evidence = cb.force_open();
        assert!(cb.is_open());
        assert!(!evidence.is_empty());
    }

    #[test]
    fn force_close_resets_immediately() {
        let cb = CircuitBreaker::new(test_config());

        // Trip first
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        assert!(cb.is_open());

        let evidence = cb.force_close();
        assert!(cb.is_closed());
        assert!(!evidence.is_empty());
    }

    // ─── Edge Cases ─────────────────────────────────────────────────

    #[test]
    fn default_config_values() {
        let config = CircuitBreakerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.latency_threshold_ms, 500);
        assert!((config.improvement_threshold - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.half_open_interval_ms, 30_000);
        assert_eq!(config.reset_threshold, 3);
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = test_config();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: CircuitBreakerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, config);
    }

    #[test]
    fn metrics_serde_roundtrip() {
        let metrics = CircuitMetrics {
            trips: 3,
            resets: 1,
            queries_skipped: 42,
            probes_attempted: 4,
            probes_succeeded: 2,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: CircuitMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.trips, metrics.trips);
        assert_eq!(decoded.resets, metrics.resets);
        assert_eq!(decoded.queries_skipped, metrics.queries_skipped);
    }

    #[test]
    fn debug_format() {
        let cb = CircuitBreaker::new(test_config());
        let debug = format!("{cb:?}");
        assert!(debug.contains("CircuitBreaker"));
        assert!(debug.contains("Closed"));
    }

    // ─── Full Lifecycle ─────────────────────────────────────────────

    #[test]
    fn full_lifecycle_closed_open_halfopen_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            half_open_interval_ms: 10,
            reset_threshold: 1,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Phase 1: Closed
        assert!(cb.is_closed());
        let (skip, _) = cb.should_skip_quality();
        assert!(!skip);

        // Phase 2: Trip → Open
        cb.record_outcome(&QualityOutcome::Error);
        cb.record_outcome(&QualityOutcome::Error);
        assert!(cb.is_open());

        // Phase 3: Skip queries while open
        let (skip, _) = cb.should_skip_quality();
        assert!(skip);

        // Phase 4: Wait → HalfOpen
        std::thread::sleep(std::time::Duration::from_millis(20));
        let (skip, _) = cb.should_skip_quality();
        assert!(!skip);
        assert!(cb.is_half_open());

        // Phase 5: Probe succeeds → Closed
        cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 50,
            tau_improvement: 0.2,
        });
        assert!(cb.is_closed());

        // Verify metrics tell the full story
        let m = cb.metrics();
        assert_eq!(m.trips, 1);
        assert_eq!(m.resets, 1);
        assert_eq!(m.queries_skipped, 1);
        assert_eq!(m.probes_attempted, 1);
        assert_eq!(m.probes_succeeded, 1);
    }

    // ─── bd-3gmt tests begin ───

    #[test]
    fn config_debug_clone() {
        let config = test_config();
        let cloned = config.clone();
        assert_eq!(config, cloned);
        let dbg = format!("{config:?}");
        assert!(dbg.contains("CircuitBreakerConfig"));
        assert!(dbg.contains("failure_threshold"));
    }

    #[test]
    fn circuit_metrics_debug_clone_default() {
        let m = CircuitMetrics::default();
        assert_eq!(m.trips, 0);
        assert_eq!(m.resets, 0);
        assert_eq!(m.queries_skipped, 0);
        assert_eq!(m.probes_attempted, 0);
        assert_eq!(m.probes_succeeded, 0);

        let cloned = m.clone();
        assert_eq!(cloned.trips, 0);

        let dbg = format!("{m:?}");
        assert!(dbg.contains("CircuitMetrics"));
    }

    #[test]
    fn quality_outcome_debug() {
        let outcomes = vec![
            QualityOutcome::Success {
                latency_ms: 100,
                tau_improvement: 0.1,
            },
            QualityOutcome::Slow { latency_ms: 500 },
            QualityOutcome::NotUseful {
                tau_improvement: 0.01,
            },
            QualityOutcome::Error,
        ];
        for o in &outcomes {
            let dbg = format!("{o:?}");
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn with_defaults_constructor() {
        let cb = CircuitBreaker::with_defaults();
        assert!(cb.is_closed());
        assert_eq!(cb.config().failure_threshold, 5);
        assert_eq!(cb.config().latency_threshold_ms, 500);
        assert_eq!(cb.config().half_open_interval_ms, 30_000);
        assert_eq!(cb.config().reset_threshold, 3);
    }

    #[test]
    fn config_accessor() {
        let config = test_config();
        let cb = CircuitBreaker::new(config.clone());
        assert_eq!(cb.config().failure_threshold, config.failure_threshold);
        assert_eq!(
            cb.config().latency_threshold_ms,
            config.latency_threshold_ms
        );
        assert!(cb.config().enabled);
    }

    #[test]
    fn force_close_when_already_closed() {
        let cb = CircuitBreaker::new(test_config());
        assert!(cb.is_closed());
        let evidence = cb.force_close();
        assert!(cb.is_closed());
        // Still emits evidence (reset transition)
        assert!(!evidence.is_empty());
    }

    #[test]
    fn force_open_when_already_open() {
        let cb = CircuitBreaker::new(test_config());
        cb.force_open();
        assert!(cb.is_open());
        let evidence = cb.force_open();
        assert!(cb.is_open());
        // Still emits evidence
        assert!(!evidence.is_empty());
        assert_eq!(cb.metrics().trips, 2);
    }

    #[test]
    fn multiple_trip_reset_cycles() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            reset_threshold: 1,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        for cycle in 0..3u64 {
            // Trip
            for _ in 0..3 {
                cb.record_outcome(&QualityOutcome::Error);
            }
            assert!(cb.is_open());

            // Wait and recover
            std::thread::sleep(std::time::Duration::from_millis(20));
            cb.should_skip_quality();
            assert!(cb.is_half_open());
            cb.record_outcome(&QualityOutcome::Success {
                latency_ms: 50,
                tau_improvement: 0.2,
            });
            assert!(cb.is_closed());

            assert_eq!(cb.metrics().trips, cycle + 1);
            assert_eq!(cb.metrics().resets, cycle + 1);
        }
    }

    #[test]
    fn record_outcome_in_open_state() {
        let cb = CircuitBreaker::new(test_config());

        // Trip the breaker
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        assert!(cb.is_open());

        // Recording outcome while open is a no-op (defensive path)
        let evidence = cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 50,
            tau_improvement: 0.5,
        });
        assert!(evidence.is_empty());
        assert!(cb.is_open()); // Still open
    }

    #[test]
    fn pipeline_state_all_variants() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Closed -> Nominal
        assert_eq!(cb.pipeline_state(), PipelineState::Nominal);

        // Trip -> CircuitOpen
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        assert_eq!(cb.pipeline_state(), PipelineState::CircuitOpen);

        // Wait -> HalfOpen -> Probing
        std::thread::sleep(std::time::Duration::from_millis(20));
        cb.should_skip_quality();
        assert_eq!(cb.pipeline_state(), PipelineState::Probing);
    }

    #[test]
    fn half_open_does_not_skip() {
        let config = CircuitBreakerConfig {
            half_open_interval_ms: 10,
            ..test_config()
        };
        let cb = CircuitBreaker::new(config);

        // Trip and transition to half-open
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
        cb.should_skip_quality(); // transitions to HalfOpen

        // Second call in HalfOpen should still not skip
        let (skip, evidence) = cb.should_skip_quality();
        assert!(!skip);
        assert!(evidence.is_empty());
    }

    #[test]
    fn debug_format_shows_open_state() {
        let cb = CircuitBreaker::new(test_config());
        for _ in 0..3 {
            cb.record_outcome(&QualityOutcome::Error);
        }
        let dbg = format!("{cb:?}");
        assert!(dbg.contains("Open"));
        assert!(dbg.contains("trips"));
    }

    #[test]
    fn metrics_serde_all_fields() {
        let metrics = CircuitMetrics {
            trips: 10,
            resets: 5,
            queries_skipped: 100,
            probes_attempted: 8,
            probes_succeeded: 5,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: CircuitMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.trips, 10);
        assert_eq!(decoded.resets, 5);
        assert_eq!(decoded.queries_skipped, 100);
        assert_eq!(decoded.probes_attempted, 8);
        assert_eq!(decoded.probes_succeeded, 5);
    }

    // ─── bd-3gmt tests end ───
}
