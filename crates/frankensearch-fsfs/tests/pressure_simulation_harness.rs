//! Deterministic simulation harness for pressure/degradation controllers.
//!
//! Provides:
//! - Synthetic workload and pressure scenario generation
//! - Deterministic timing/state replay with golden-output comparison
//! - Oracle checks for transition correctness, hysteresis, anti-flap,
//!   recovery gates, and calibration-guard fallback triggers
//! - Structured JSONL artifact emission for triage and replay
//!
//! Bead: bd-2hz.10.3

use std::fs;
use std::io::Write as IoWrite;
use std::path::Path;

use serde::{Deserialize, Serialize};

use frankensearch_fsfs::config::PressureProfile;
use frankensearch_fsfs::pressure::{
    CalibrationGuardConfig, CalibrationGuardState, CalibrationGuardStatus, CalibrationMetrics,
    DegradationControllerConfig, DegradationSignal, DegradationStage, DegradationStateMachine,
    DegradationTrigger, PressureController, PressureControllerConfig, PressureSignal,
    PressureState, REASON_DEGRADE_RECOVERED,
};

// ─── Scenario types ──────────────────────────────────────────────────────

/// A timestamped pressure input for the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PressureEvent {
    timestamp_ms: u64,
    signal: PressureSignalInput,
}

/// Raw pressure signal values for scenario definition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct PressureSignalInput {
    cpu: f64,
    memory: f64,
    io: f64,
    load: f64,
}

impl PressureSignalInput {
    const fn new(cpu: f64, memory: f64, io: f64, load: f64) -> Self {
        Self {
            cpu,
            memory,
            io,
            load,
        }
    }

    fn to_signal(self) -> PressureSignal {
        PressureSignal::new(self.cpu, self.memory, self.io, self.load)
    }
}

/// A timestamped degradation control input.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct DegradationEvent {
    timestamp_ms: u64,
    pressure_state: PressureState,
    quality_circuit_open: bool,
    hard_pause_requested: bool,
}

/// A timestamped calibration metrics input.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct CalibrationEvent {
    timestamp_ms: u64,
    sample_count: u64,
    observed_coverage_pct: f64,
    e_value: f64,
    drift_pct: f64,
    confidence_pct: f64,
}

/// A complete simulation scenario combining all input streams.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationScenario {
    name: String,
    description: String,
    seed: u64,
    pressure_config: PressureControllerConfig,
    degradation_config: DegradationControllerConfig,
    calibration_config: CalibrationGuardConfig,
    pressure_events: Vec<PressureEvent>,
    degradation_events: Vec<DegradationEvent>,
    calibration_events: Vec<CalibrationEvent>,
}

// ─── Recorded outputs ────────────────────────────────────────────────────

/// Recorded pressure controller output for one timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PressureRecord {
    timestamp_ms: u64,
    from: PressureState,
    to: PressureState,
    changed: bool,
    reason_code: String,
    score: f64,
    consecutive_observed: u8,
}

/// Recorded degradation state machine output for one timestep.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DegradationRecord {
    timestamp_ms: u64,
    from: DegradationStage,
    to: DegradationStage,
    changed: bool,
    trigger: DegradationTrigger,
    reason_code: String,
    pending_recovery_observations: u8,
}

/// Recorded calibration guard output for one evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationRecord {
    timestamp_ms: u64,
    status: CalibrationGuardStatus,
    reason_code: String,
    fallback_stage: Option<DegradationStage>,
    consecutive_breach_count: u8,
    fallback_triggered: bool,
}

/// Complete simulation output for golden-comparison and oracle checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationTrace {
    scenario_name: String,
    pressure_trace: Vec<PressureRecord>,
    degradation_trace: Vec<DegradationRecord>,
    calibration_trace: Vec<CalibrationRecord>,
    oracle_results: Vec<OracleResult>,
}

/// Result of one oracle invariant check.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OracleResult {
    oracle_name: String,
    passed: bool,
    detail: String,
}

// ─── Scenario generators ─────────────────────────────────────────────────

fn scenario_gradual_ramp_up() -> SimulationScenario {
    let mut events = Vec::new();
    // Start normal, gradually increase CPU pressure over 30 ticks
    let mut cpu = 40.0;
    for i in 0_u64..30 {
        events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(cpu, 30.0, 20.0, 35.0),
        });
        cpu += 2.0; // 40 → 98
    }
    // Hold at 98% for 10 more ticks so EWMA (alpha=0.3) catches up past
    // the Emergency threshold (95% on Performance profile).
    for i in 30_u64..40 {
        events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(98.0, 30.0, 20.0, 35.0),
        });
    }
    SimulationScenario {
        name: "gradual_ramp_up".to_owned(),
        description: "CPU pressure gradually increases from 40% to 98%".to_owned(),
        seed: 42,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig::default(),
        pressure_events: events,
        degradation_events: Vec::new(),
        calibration_events: Vec::new(),
    }
}

fn scenario_spike_and_recovery() -> SimulationScenario {
    let mut pressure_events = Vec::new();
    let mut degradation_events = Vec::new();

    // Phase 1: Normal operation (5 ticks)
    for i in 0_u64..5 {
        pressure_events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(30.0, 25.0, 15.0, 20.0),
        });
    }
    // Phase 2: Sudden CPU spike to Emergency (5 ticks)
    for i in 5_u64..10 {
        pressure_events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(150.0, 40.0, 30.0, 120.0),
        });
    }
    // Phase 3: Back to low (15 ticks — enough for stepwise recovery)
    for i in 10_u64..25 {
        pressure_events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(20.0, 15.0, 10.0, 15.0),
        });
    }

    // Degradation events driven by mapped pressure states
    for i in 0_u64..5 {
        degradation_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }
    for i in 5_u64..10 {
        degradation_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Emergency,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }
    for i in 10_u64..25 {
        degradation_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }

    SimulationScenario {
        name: "spike_and_recovery".to_owned(),
        description: "Normal → Emergency spike → stepwise recovery to Full".to_owned(),
        seed: 99,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig::default(),
        pressure_events,
        degradation_events,
        calibration_events: Vec::new(),
    }
}

fn scenario_hysteresis_oscillation() -> SimulationScenario {
    let mut events = Vec::new();
    // Oscillate around the Performance-profile constrained threshold (70%)
    // with hysteresis margin (5%), cycling between 67% and 73%
    for i in 0_u64..30 {
        let cpu = if i % 2 == 0 { 73.0 } else { 67.0 };
        events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(cpu, 30.0, 20.0, 30.0),
        });
    }
    SimulationScenario {
        name: "hysteresis_oscillation".to_owned(),
        description:
            "Oscillates between 67% and 73% CPU — should not flap between Normal/Constrained"
                .to_owned(),
        seed: 77,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig::default(),
        pressure_events: events,
        degradation_events: Vec::new(),
        calibration_events: Vec::new(),
    }
}

fn scenario_hard_pause_and_clear() -> SimulationScenario {
    let mut events = Vec::new();

    // Phase 1: Normal with hard pause request
    for i in 0_u64..3 {
        events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: true,
        });
    }
    // Phase 2: Hard pause cleared, but still paused until recovery gate
    for i in 3_u64..15 {
        events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }

    SimulationScenario {
        name: "hard_pause_and_clear".to_owned(),
        description: "Hard pause → clear → stepwise recovery through all stages".to_owned(),
        seed: 55,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig::default(),
        pressure_events: Vec::new(),
        degradation_events: events,
        calibration_events: Vec::new(),
    }
}

fn scenario_calibration_breach_fallback() -> SimulationScenario {
    let mut cal_events = Vec::new();
    let mut deg_events = Vec::new();

    // Phase 1: Healthy calibration (5 evaluations)
    for i in 0_u64..5 {
        cal_events.push(CalibrationEvent {
            timestamp_ms: i * 1000,
            sample_count: 500,
            observed_coverage_pct: 97.0,
            e_value: 0.10,
            drift_pct: 3.0,
            confidence_pct: 85.0,
        });
        deg_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }
    // Phase 2: Coverage drops below target (3 evaluations — breach after 2)
    for i in 5_u64..8 {
        cal_events.push(CalibrationEvent {
            timestamp_ms: i * 1000,
            sample_count: 500,
            observed_coverage_pct: 70.0, // below 95% target
            e_value: 0.10,
            drift_pct: 3.0,
            confidence_pct: 85.0,
        });
        deg_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }
    // Phase 3: Recovery to healthy (5 evaluations)
    for i in 8_u64..13 {
        cal_events.push(CalibrationEvent {
            timestamp_ms: i * 1000,
            sample_count: 600,
            observed_coverage_pct: 97.0,
            e_value: 0.15,
            drift_pct: 2.0,
            confidence_pct: 92.0,
        });
        deg_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }

    SimulationScenario {
        name: "calibration_breach_fallback".to_owned(),
        description: "Healthy → coverage breach triggers fallback → healthy recovery".to_owned(),
        seed: 88,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig {
            breach_consecutive_required: 2,
            fallback_stage: DegradationStage::LexicalOnly,
            ..CalibrationGuardConfig::default()
        },
        pressure_events: Vec::new(),
        degradation_events: deg_events,
        calibration_events: cal_events,
    }
}

fn scenario_quality_circuit_breaker() -> SimulationScenario {
    let mut events = Vec::new();
    // Phase 1: Normal with quality circuit closed
    for i in 0_u64..3 {
        events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }
    // Phase 2: Quality circuit opens
    for i in 3_u64..8 {
        events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: true,
            hard_pause_requested: false,
        });
    }
    // Phase 3: Quality circuit closes → recovery needs consecutive healthy
    for i in 8_u64..18 {
        events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }

    SimulationScenario {
        name: "quality_circuit_breaker".to_owned(),
        description: "Normal → circuit open (EmbedDeferred) → circuit close → recovery".to_owned(),
        seed: 33,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig::default(),
        pressure_events: Vec::new(),
        degradation_events: events,
        calibration_events: Vec::new(),
    }
}

fn scenario_multi_profile_comparison() -> SimulationScenario {
    let mut events = Vec::new();
    // Slowly ramp from 50 to 99 to test all profile thresholds
    let mut cpu = 50.0;
    for i in 0_u64..50 {
        events.push(PressureEvent {
            timestamp_ms: i * 1000,
            signal: PressureSignalInput::new(cpu, 30.0, 20.0, 30.0),
        });
        cpu += 1.0;
    }
    SimulationScenario {
        name: "multi_profile_ramp".to_owned(),
        description: "Linear ramp 50→99 CPU for cross-profile threshold comparison".to_owned(),
        seed: 11,
        pressure_config: PressureControllerConfig {
            profile: PressureProfile::Strict,
            ..PressureControllerConfig::default()
        },
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig::default(),
        pressure_events: events,
        degradation_events: Vec::new(),
        calibration_events: Vec::new(),
    }
}

fn scenario_insufficient_samples_watch() -> SimulationScenario {
    let mut cal_events = Vec::new();
    let mut deg_events = Vec::new();

    // Phase 1: Few samples (below min_sample_count=200) with "bad" metrics
    // Guard should stay Watch, not breach
    for i in 0_u64..5 {
        cal_events.push(CalibrationEvent {
            timestamp_ms: i * 1000,
            sample_count: 50 + i * 10, // 50, 60, 70, 80, 90
            observed_coverage_pct: 60.0,
            e_value: 0.01,
            drift_pct: 25.0,
            confidence_pct: 40.0,
        });
        deg_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }
    // Phase 2: Hit sample threshold with bad metrics → breach
    for i in 5_u64..8 {
        cal_events.push(CalibrationEvent {
            timestamp_ms: i * 1000,
            sample_count: 300,
            observed_coverage_pct: 60.0,
            e_value: 0.01,
            drift_pct: 25.0,
            confidence_pct: 40.0,
        });
        deg_events.push(DegradationEvent {
            timestamp_ms: i * 1000,
            pressure_state: PressureState::Normal,
            quality_circuit_open: false,
            hard_pause_requested: false,
        });
    }

    SimulationScenario {
        name: "insufficient_samples_watch".to_owned(),
        description: "Bad metrics but insufficient samples → Watch, then breach after threshold"
            .to_owned(),
        seed: 66,
        pressure_config: PressureControllerConfig::default(),
        degradation_config: DegradationControllerConfig::default(),
        calibration_config: CalibrationGuardConfig {
            breach_consecutive_required: 2,
            ..CalibrationGuardConfig::default()
        },
        pressure_events: Vec::new(),
        degradation_events: deg_events,
        calibration_events: cal_events,
    }
}

// ─── Simulation runner ──────────────────────────────────────────────────

fn run_pressure_simulation(
    config: PressureControllerConfig,
    events: &[PressureEvent],
) -> Vec<PressureRecord> {
    let mut controller = PressureController::new(config).expect("valid pressure controller config");
    let mut records = Vec::with_capacity(events.len());

    for event in events {
        let transition = controller.observe(event.signal.to_signal(), event.timestamp_ms);
        records.push(PressureRecord {
            timestamp_ms: event.timestamp_ms,
            from: transition.from,
            to: transition.to,
            changed: transition.changed,
            reason_code: transition.reason_code.to_owned(),
            score: transition.snapshot.score,
            consecutive_observed: transition.consecutive_observed,
        });
    }
    records
}

fn run_degradation_simulation(
    config: DegradationControllerConfig,
    events: &[DegradationEvent],
) -> Vec<DegradationRecord> {
    let mut machine =
        DegradationStateMachine::new(config).expect("valid degradation controller config");
    let mut records = Vec::with_capacity(events.len());

    for event in events {
        let signal = DegradationSignal::new(
            event.pressure_state,
            event.quality_circuit_open,
            event.hard_pause_requested,
        );
        let transition = machine.observe(signal, event.timestamp_ms);
        records.push(DegradationRecord {
            timestamp_ms: event.timestamp_ms,
            from: transition.from,
            to: transition.to,
            changed: transition.changed,
            trigger: transition.trigger,
            reason_code: transition.reason_code.to_owned(),
            pending_recovery_observations: transition.pending_recovery_observations,
        });
    }
    records
}

fn run_calibration_simulation(
    cal_config: CalibrationGuardConfig,
    deg_config: DegradationControllerConfig,
    cal_events: &[CalibrationEvent],
    deg_events: &[DegradationEvent],
) -> (Vec<CalibrationRecord>, Vec<DegradationRecord>) {
    let mut guard = CalibrationGuardState::new(cal_config).expect("valid calibration guard config");
    let mut machine =
        DegradationStateMachine::new(deg_config).expect("valid degradation controller config");
    let mut cal_records = Vec::with_capacity(cal_events.len());
    let mut deg_records = Vec::with_capacity(deg_events.len());

    for (cal_event, deg_event) in cal_events.iter().zip(deg_events.iter()) {
        let signal = DegradationSignal::new(
            deg_event.pressure_state,
            deg_event.quality_circuit_open,
            deg_event.hard_pause_requested,
        );

        // Observe degradation first
        let deg_transition = machine.observe(signal, deg_event.timestamp_ms);
        deg_records.push(DegradationRecord {
            timestamp_ms: deg_event.timestamp_ms,
            from: deg_transition.from,
            to: deg_transition.to,
            changed: deg_transition.changed,
            trigger: deg_transition.trigger,
            reason_code: deg_transition.reason_code.to_owned(),
            pending_recovery_observations: deg_transition.pending_recovery_observations,
        });

        // Evaluate calibration guard
        let metrics = CalibrationMetrics::new(
            cal_event.sample_count,
            cal_event.observed_coverage_pct,
            cal_event.e_value,
            cal_event.drift_pct,
            cal_event.confidence_pct,
        );
        let decision = guard.evaluate(metrics);
        cal_records.push(CalibrationRecord {
            timestamp_ms: cal_event.timestamp_ms,
            status: decision.status,
            reason_code: decision.reason_code.to_owned(),
            fallback_stage: decision.fallback_stage,
            consecutive_breach_count: decision.evidence.consecutive_breach_count,
            fallback_triggered: decision.evidence.fallback_triggered,
        });

        // Apply calibration fallback if triggered
        if let Some(fallback_transition) =
            machine.apply_calibration_guard(signal, &decision, cal_event.timestamp_ms)
        {
            deg_records.push(DegradationRecord {
                timestamp_ms: cal_event.timestamp_ms,
                from: fallback_transition.from,
                to: fallback_transition.to,
                changed: fallback_transition.changed,
                trigger: fallback_transition.trigger,
                reason_code: fallback_transition.reason_code.to_owned(),
                pending_recovery_observations: fallback_transition.pending_recovery_observations,
            });
        }
    }
    (cal_records, deg_records)
}

// ─── Oracle checks ──────────────────────────────────────────────────────

fn oracle_anti_flap(records: &[PressureRecord], consecutive_required: u8) -> OracleResult {
    // Invariant: no state transition occurs before `consecutive_required`
    // observations at the new state.
    for record in records {
        if record.changed && record.consecutive_observed != 0 {
            // When changed=true, consecutive_observed should have been reset to 0
            // after the transition applied. The controller resets to 0 after applying.
            // This is fine — check that transitions never happen at counts < required.
        }
    }

    // Alternative: check that between stable periods and transitions, the count
    // of pending readings is at least consecutive_required - 1 preceding entries.
    let mut violations = Vec::new();
    let mut pending_count = 0_u8;
    let mut pending_target: Option<PressureState> = None;

    for (i, record) in records.iter().enumerate() {
        if record.changed {
            // Before the transition, there should have been consecutive_required - 1
            // pending readings. The transition itself is the N-th reading.
            if pending_count < consecutive_required.saturating_sub(1) {
                violations.push(format!(
                    "tick {i}: transition with only {} pending (need {})",
                    pending_count,
                    consecutive_required - 1
                ));
            }
            pending_count = 0;
            pending_target = None;
        } else if record.reason_code == "pressure.transition.pending" {
            if pending_target == Some(record.to) || pending_target.is_none() {
                pending_count = pending_count.saturating_add(1);
                pending_target = Some(record.from); // target is implicit from context
            }
        } else {
            pending_count = 0;
            pending_target = None;
        }
    }

    OracleResult {
        oracle_name: "anti_flap_guard".to_owned(),
        passed: violations.is_empty(),
        detail: if violations.is_empty() {
            format!("No anti-flap violations across {} records", records.len())
        } else {
            format!("{} violations: {}", violations.len(), violations.join("; "))
        },
    }
}

fn oracle_monotonic_escalation(records: &[DegradationRecord]) -> OracleResult {
    // Invariant: escalation is always immediate (single-step), never skips states
    // in the recovery direction, and recovery is always stepwise.
    let mut violations = Vec::new();

    for (i, record) in records.iter().enumerate() {
        if !record.changed {
            continue;
        }

        let from_sev = stage_severity(record.from);
        let to_sev = stage_severity(record.to);

        if to_sev > from_sev {
            // Escalation: should be immediate. Multi-step jumps are fine
            // (e.g., Full → MetadataOnly under Emergency pressure).
            // The key invariant is that escalation happens on the very first observation.
            // This is verified by the trigger being PressureEscalation/HardPause/QualityCircuitOpen.
            let valid_escalation_triggers = matches!(
                record.trigger,
                DegradationTrigger::PressureEscalation
                    | DegradationTrigger::HardPause
                    | DegradationTrigger::QualityCircuitOpen
                    | DegradationTrigger::CalibrationBreach
                    | DegradationTrigger::OperatorOverride
            );
            if !valid_escalation_triggers {
                violations.push(format!(
                    "tick {i}: escalation {:?}->{:?} with unexpected trigger {:?}",
                    record.from, record.to, record.trigger
                ));
            }
        }

        if to_sev < from_sev {
            // Recovery: should be exactly one step
            let step_size = from_sev - to_sev;
            if step_size > 1 {
                violations.push(format!(
                    "tick {i}: recovery jumped {} steps ({:?}->{:?}), expected 1",
                    step_size, record.from, record.to
                ));
            }
        }
    }

    OracleResult {
        oracle_name: "monotonic_escalation_stepwise_recovery".to_owned(),
        passed: violations.is_empty(),
        detail: if violations.is_empty() {
            format!("All {} transitions are valid", records.len())
        } else {
            format!("{} violations: {}", violations.len(), violations.join("; "))
        },
    }
}

fn oracle_recovery_gate_respected(records: &[DegradationRecord]) -> OracleResult {
    // Invariant: recovery transitions only happen with trigger=Recovery and
    // reason_code=REASON_DEGRADE_RECOVERED, preceded by recovery_pending entries.
    let mut violations = Vec::new();

    for (i, record) in records.iter().enumerate() {
        if record.changed && stage_severity(record.to) < stage_severity(record.from) {
            // This is a recovery transition
            if record.trigger != DegradationTrigger::Recovery {
                violations.push(format!(
                    "tick {i}: recovery {:?}->{:?} with wrong trigger {:?}",
                    record.from, record.to, record.trigger
                ));
            }
            if record.reason_code != REASON_DEGRADE_RECOVERED {
                violations.push(format!(
                    "tick {i}: recovery with wrong reason: {}",
                    record.reason_code
                ));
            }
        }
    }

    OracleResult {
        oracle_name: "recovery_gate_respected".to_owned(),
        passed: violations.is_empty(),
        detail: if violations.is_empty() {
            "All recovery transitions satisfy gate constraints".to_owned()
        } else {
            format!("{} violations: {}", violations.len(), violations.join("; "))
        },
    }
}

fn oracle_calibration_watch_before_threshold(records: &[CalibrationRecord]) -> OracleResult {
    // Invariant: while sample_count < min_sample_count, status is always Watch
    // and no fallback is triggered.
    let mut violations = Vec::new();
    // min_sample_count is implicit — the guard returns a specific reason_code
    // ("calibration.guard.insufficient_samples") when samples are below threshold.

    for (i, record) in records.iter().enumerate() {
        // We can't directly access sample_count from CalibrationRecord,
        // but the reason_code tells us if samples were insufficient.
        if record.reason_code == "calibration.guard.insufficient_samples" {
            if record.status != CalibrationGuardStatus::Watch {
                violations.push(format!(
                    "tick {i}: insufficient samples but status={:?}",
                    record.status
                ));
            }
            if record.fallback_triggered {
                violations.push(format!(
                    "tick {i}: insufficient samples but fallback triggered"
                ));
            }
        }
    }

    OracleResult {
        oracle_name: "calibration_watch_before_threshold".to_owned(),
        passed: violations.is_empty(),
        detail: if violations.is_empty() {
            "Watch status correctly enforced during insufficient samples".to_owned()
        } else {
            format!("{} violations: {}", violations.len(), violations.join("; "))
        },
    }
}

fn oracle_breach_reset_on_healthy(records: &[CalibrationRecord]) -> OracleResult {
    // Invariant: after a Healthy evaluation, consecutive_breach_count resets to 0
    let mut violations = Vec::new();

    for (i, record) in records.iter().enumerate() {
        if record.status == CalibrationGuardStatus::Healthy && record.consecutive_breach_count != 0
        {
            violations.push(format!(
                "tick {i}: Healthy status but breach_count={}",
                record.consecutive_breach_count
            ));
        }
    }

    OracleResult {
        oracle_name: "breach_reset_on_healthy".to_owned(),
        passed: violations.is_empty(),
        detail: if violations.is_empty() {
            "Breach counter correctly resets on healthy evaluations".to_owned()
        } else {
            format!("{} violations: {}", violations.len(), violations.join("; "))
        },
    }
}

fn oracle_determinism(scenario: &SimulationScenario) -> OracleResult {
    // Invariant: running the same scenario twice produces identical output.
    // Uses run_simulation_core (not run_full_simulation) to avoid infinite recursion:
    // run_full_simulation → oracle_determinism → run_full_simulation → ...
    let trace1 = run_simulation_core(scenario);
    let trace2 = run_simulation_core(scenario);

    let p_match = trace1.pressure_trace.len() == trace2.pressure_trace.len()
        && trace1
            .pressure_trace
            .iter()
            .zip(trace2.pressure_trace.iter())
            .all(|(a, b)| {
                a.from == b.from
                    && a.to == b.to
                    && a.changed == b.changed
                    && a.reason_code == b.reason_code
            });

    let d_match = trace1.degradation_trace.len() == trace2.degradation_trace.len()
        && trace1
            .degradation_trace
            .iter()
            .zip(trace2.degradation_trace.iter())
            .all(|(a, b)| {
                a.from == b.from
                    && a.to == b.to
                    && a.changed == b.changed
                    && a.trigger == b.trigger
                    && a.reason_code == b.reason_code
            });

    let c_match = trace1.calibration_trace.len() == trace2.calibration_trace.len()
        && trace1
            .calibration_trace
            .iter()
            .zip(trace2.calibration_trace.iter())
            .all(|(a, b)| {
                a.status == b.status
                    && a.reason_code == b.reason_code
                    && a.fallback_stage == b.fallback_stage
                    && a.consecutive_breach_count == b.consecutive_breach_count
            });

    let all_match = p_match && d_match && c_match;
    OracleResult {
        oracle_name: "deterministic_replay".to_owned(),
        passed: all_match,
        detail: if all_match {
            "Two runs produce identical traces".to_owned()
        } else {
            format!(
                "Determinism violation: pressure={p_match}, degradation={d_match}, calibration={c_match}"
            )
        },
    }
}

fn oracle_stable_has_no_state_change(records: &[DegradationRecord]) -> OracleResult {
    let mut violations = Vec::new();
    for (i, record) in records.iter().enumerate() {
        if record.trigger == DegradationTrigger::Stable && record.changed {
            violations.push(format!(
                "tick {i}: Stable trigger but changed=true ({:?}->{:?})",
                record.from, record.to
            ));
        }
    }
    OracleResult {
        oracle_name: "stable_no_change".to_owned(),
        passed: violations.is_empty(),
        detail: if violations.is_empty() {
            "Stable trigger never causes state change".to_owned()
        } else {
            format!("{} violations: {}", violations.len(), violations.join("; "))
        },
    }
}

const fn stage_severity(stage: DegradationStage) -> u8 {
    match stage {
        DegradationStage::Full => 0,
        DegradationStage::EmbedDeferred => 1,
        DegradationStage::LexicalOnly => 2,
        DegradationStage::MetadataOnly => 3,
        DegradationStage::Paused => 4,
    }
}

// ─── Full simulation runner ─────────────────────────────────────────────

/// Core simulation runner: produces traces WITHOUT oracle checks.
/// Used by `oracle_determinism` to avoid infinite recursion.
fn run_simulation_core(scenario: &SimulationScenario) -> SimulationTrace {
    let pressure_trace = if scenario.pressure_events.is_empty() {
        Vec::new()
    } else {
        run_pressure_simulation(scenario.pressure_config, &scenario.pressure_events)
    };

    let (calibration_trace, cal_deg_trace) = if scenario.calibration_events.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        run_calibration_simulation(
            scenario.calibration_config,
            scenario.degradation_config,
            &scenario.calibration_events,
            &scenario.degradation_events,
        )
    };

    let degradation_trace =
        if !scenario.degradation_events.is_empty() && scenario.calibration_events.is_empty() {
            run_degradation_simulation(scenario.degradation_config, &scenario.degradation_events)
        } else {
            cal_deg_trace
        };

    SimulationTrace {
        scenario_name: scenario.name.clone(),
        pressure_trace,
        degradation_trace,
        calibration_trace,
        oracle_results: Vec::new(),
    }
}

/// Full simulation: runs core simulation + all oracle invariant checks.
fn run_full_simulation(scenario: &SimulationScenario) -> SimulationTrace {
    let mut trace = run_simulation_core(scenario);

    // Run oracle checks against the produced traces
    if !trace.pressure_trace.is_empty() {
        trace.oracle_results.push(oracle_anti_flap(
            &trace.pressure_trace,
            scenario.pressure_config.consecutive_required,
        ));
    }
    if !trace.degradation_trace.is_empty() {
        trace
            .oracle_results
            .push(oracle_monotonic_escalation(&trace.degradation_trace));
        trace
            .oracle_results
            .push(oracle_recovery_gate_respected(&trace.degradation_trace));
        trace
            .oracle_results
            .push(oracle_stable_has_no_state_change(&trace.degradation_trace));
    }
    if !trace.calibration_trace.is_empty() {
        trace
            .oracle_results
            .push(oracle_calibration_watch_before_threshold(
                &trace.calibration_trace,
            ));
        trace
            .oracle_results
            .push(oracle_breach_reset_on_healthy(&trace.calibration_trace));
    }
    trace.oracle_results.push(oracle_determinism(scenario));

    trace
}

fn emit_trace_artifact(trace: &SimulationTrace, artifact_dir: &Path) {
    fs::create_dir_all(artifact_dir).expect("create artifact dir");

    let events_path = artifact_dir.join("simulation_events.jsonl");
    let mut file = fs::File::create(&events_path).expect("create events file");

    // Emit pressure records
    for record in &trace.pressure_trace {
        let line = serde_json::to_string(record).expect("serialize pressure record");
        writeln!(file, "{line}").expect("write pressure record");
    }

    // Emit degradation records
    for record in &trace.degradation_trace {
        let line = serde_json::to_string(record).expect("serialize degradation record");
        writeln!(file, "{line}").expect("write degradation record");
    }

    // Emit calibration records
    for record in &trace.calibration_trace {
        let line = serde_json::to_string(record).expect("serialize calibration record");
        writeln!(file, "{line}").expect("write calibration record");
    }

    // Emit oracle results
    let oracle_path = artifact_dir.join("oracle_results.json");
    let oracle_json =
        serde_json::to_string_pretty(&trace.oracle_results).expect("serialize oracle results");
    fs::write(&oracle_path, oracle_json).expect("write oracle results");
}

fn emit_trace_artifact_bundle(trace: &SimulationTrace, artifact_root: &Path) {
    let artifact_dir = artifact_root.join(&trace.scenario_name);
    emit_trace_artifact(trace, &artifact_dir);

    let manifest_path = artifact_dir.join("replay_manifest.json");
    let manifest = serde_json::json!({
        "schema_version": 1,
        "scenario": trace.scenario_name,
        "artifact_dir": artifact_dir.to_string_lossy(),
        "artifacts": {
            "events_jsonl": "simulation_events.jsonl",
            "oracle_results_json": "oracle_results.json"
        },
        "replay": {
            "env": {
                "FSFS_PRESSURE_ARTIFACT_DIR": artifact_root.to_string_lossy()
            },
            "command": "cargo test -p frankensearch-fsfs --test pressure_simulation_harness full_simulation_suite_all_oracles_pass -- --exact --nocapture"
        }
    });
    let manifest_json =
        serde_json::to_string_pretty(&manifest).expect("serialize replay manifest json");
    fs::write(&manifest_path, manifest_json).expect("write replay manifest");
}

fn maybe_emit_trace_artifact_bundle(trace: &SimulationTrace) {
    let Some(root) = std::env::var_os("FSFS_PRESSURE_ARTIFACT_DIR") else {
        return;
    };

    let root_path = Path::new(&root);
    emit_trace_artifact_bundle(trace, root_path);
    eprintln!(
        "pressure simulation artifact bundle emitted for '{}' at {}",
        trace.scenario_name,
        root_path.join(&trace.scenario_name).display()
    );
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[test]
fn scenario_gradual_ramp_reaches_emergency() {
    let scenario = scenario_gradual_ramp_up();
    let trace = run_full_simulation(&scenario);

    // Should have progressed through states as CPU ramps up
    let final_state = trace.pressure_trace.last().expect("non-empty trace").to;
    assert_eq!(
        final_state,
        PressureState::Emergency,
        "CPU at 98% should reach Emergency on Performance profile"
    );

    // All transitions should be to increasingly severe states
    let mut max_severity = 0_u8;
    for record in &trace.pressure_trace {
        if record.changed {
            let new_sev = match record.to {
                PressureState::Normal => 0,
                PressureState::Constrained => 1,
                PressureState::Degraded => 2,
                PressureState::Emergency => 3,
            };
            assert!(
                new_sev >= max_severity,
                "Ramp-up should produce monotonically increasing severity"
            );
            max_severity = new_sev;
        }
    }

    // Oracle checks should all pass
    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_spike_has_immediate_escalation_and_stepwise_recovery() {
    let scenario = scenario_spike_and_recovery();
    let trace = run_full_simulation(&scenario);

    // Find first escalation to MetadataOnly
    let first_escalation = trace
        .degradation_trace
        .iter()
        .find(|r| r.changed && r.to == DegradationStage::MetadataOnly);
    assert!(
        first_escalation.is_some(),
        "Emergency pressure should escalate to MetadataOnly"
    );
    let esc = first_escalation.unwrap();
    assert!(
        matches!(
            esc.trigger,
            DegradationTrigger::PressureEscalation
                | DegradationTrigger::HardPause
                | DegradationTrigger::QualityCircuitOpen
                | DegradationTrigger::CalibrationBreach
                | DegradationTrigger::OperatorOverride
        ),
        "Unexpected escalation trigger to MetadataOnly: {:?}",
        esc.trigger
    );

    // Recovery should be stepwise: MetadataOnly → LexicalOnly → EmbedDeferred → Full
    let recovery_transitions: Vec<_> = trace
        .degradation_trace
        .iter()
        .filter(|r| r.changed && r.trigger == DegradationTrigger::Recovery)
        .collect();

    // Should have at least 3 recovery steps
    assert!(
        recovery_transitions.len() >= 3,
        "Expected at least 3 recovery steps, got {}",
        recovery_transitions.len()
    );

    // Verify stepwise: each step reduces severity by exactly 1
    for rt in &recovery_transitions {
        let from_sev = stage_severity(rt.from);
        let to_sev = stage_severity(rt.to);
        assert_eq!(
            from_sev - to_sev,
            1,
            "Recovery step {:?}->{:?} should reduce severity by 1",
            rt.from,
            rt.to
        );
    }

    // All oracles pass
    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_hysteresis_prevents_flapping() {
    let scenario = scenario_hysteresis_oscillation();
    let trace = run_full_simulation(&scenario);

    // Count actual transitions. With oscillation between 67% and 73%
    // around the 70% constrained threshold with 5% hysteresis,
    // the controller should NOT flap continuously.
    let transition_count = trace.pressure_trace.iter().filter(|r| r.changed).count();

    // Should be very few transitions — certainly not one every tick
    assert!(
        transition_count <= 5,
        "Hysteresis should prevent flapping: got {} transitions in {} ticks",
        transition_count,
        trace.pressure_trace.len()
    );

    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_hard_pause_blocks_recovery_until_cleared() {
    let scenario = scenario_hard_pause_and_clear();
    let trace = run_full_simulation(&scenario);

    // First transition should be to Paused due to hard_pause
    let first_change = trace.degradation_trace.iter().find(|r| r.changed);
    assert!(
        first_change.is_some(),
        "Should have at least one transition"
    );
    let fc = first_change.unwrap();
    assert_eq!(fc.to, DegradationStage::Paused);
    assert_eq!(fc.trigger, DegradationTrigger::HardPause);

    // After clearing hard pause (tick 3+), recovery should eventually happen
    let recovery_steps: Vec<_> = trace
        .degradation_trace
        .iter()
        .filter(|r| r.changed && r.trigger == DegradationTrigger::Recovery)
        .collect();
    assert!(
        !recovery_steps.is_empty(),
        "Should recover after hard pause is cleared"
    );

    // Verify stepwise: Paused → MetadataOnly → LexicalOnly → EmbedDeferred → Full
    let expected_recovery = [
        (DegradationStage::Paused, DegradationStage::MetadataOnly),
        (
            DegradationStage::MetadataOnly,
            DegradationStage::LexicalOnly,
        ),
        (
            DegradationStage::LexicalOnly,
            DegradationStage::EmbedDeferred,
        ),
        (DegradationStage::EmbedDeferred, DegradationStage::Full),
    ];
    for (i, step) in recovery_steps.iter().enumerate() {
        if i < expected_recovery.len() {
            assert_eq!(
                (step.from, step.to),
                expected_recovery[i],
                "Recovery step {i} should be {:?}→{:?}",
                expected_recovery[i].0,
                expected_recovery[i].1
            );
        }
    }

    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_calibration_triggers_and_recovers() {
    let scenario = scenario_calibration_breach_fallback();
    let trace = run_full_simulation(&scenario);

    // Phase 1: All calibration evaluations should be Healthy
    let healthy_count = trace
        .calibration_trace
        .iter()
        .take(5)
        .filter(|r| r.status == CalibrationGuardStatus::Healthy)
        .count();
    assert_eq!(healthy_count, 5, "First 5 evaluations should be Healthy");

    // Phase 2: Should have breach(es) and eventually a fallback
    let has_breach = trace
        .calibration_trace
        .iter()
        .any(|record| record.status == CalibrationGuardStatus::Breach);
    assert!(has_breach, "Coverage drop should trigger breach");

    // The fallback should force a degradation transition
    let has_cal_fallback = trace
        .degradation_trace
        .iter()
        .any(|record| record.trigger == DegradationTrigger::CalibrationBreach);
    assert!(
        has_cal_fallback,
        "Calibration breach should trigger degradation fallback"
    );

    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_quality_circuit_escalates_to_embed_deferred() {
    let scenario = scenario_quality_circuit_breaker();
    let trace = run_full_simulation(&scenario);

    // When quality circuit opens at Normal pressure, should go to EmbedDeferred
    let circuit_escalation = trace
        .degradation_trace
        .iter()
        .find(|r| r.changed && r.trigger == DegradationTrigger::QualityCircuitOpen);
    assert!(
        circuit_escalation.is_some(),
        "Quality circuit open should trigger EmbedDeferred"
    );
    let ce = circuit_escalation.unwrap();
    assert_eq!(ce.from, DegradationStage::Full);
    assert_eq!(ce.to, DegradationStage::EmbedDeferred);

    // After circuit closes and quality_circuit_closed requirement met,
    // should eventually recover to Full
    let final_state = trace.degradation_trace.last().expect("non-empty trace").to;
    assert_eq!(
        final_state,
        DegradationStage::Full,
        "Should recover to Full after circuit closes"
    );

    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_strict_profile_triggers_earlier_than_performance() {
    let strict_scenario = scenario_multi_profile_comparison();
    let strict_trace = run_full_simulation(&strict_scenario);

    // Build a Performance profile scenario with the same events
    let mut perf_scenario = strict_scenario;
    perf_scenario.pressure_config.profile = PressureProfile::Performance;
    perf_scenario.name = "multi_profile_ramp_performance".to_owned();
    let perf_trace = run_full_simulation(&perf_scenario);

    // Find first transition to Constrained in each
    let strict_first_constrained = strict_trace
        .pressure_trace
        .iter()
        .position(|r| r.changed && r.to == PressureState::Constrained);
    let perf_first_constrained = perf_trace
        .pressure_trace
        .iter()
        .position(|r| r.changed && r.to == PressureState::Constrained);

    assert!(
        strict_first_constrained.is_some(),
        "Strict should reach Constrained"
    );
    assert!(
        perf_first_constrained.is_some(),
        "Performance should reach Constrained"
    );

    // Strict profile has lower thresholds, so should transition earlier
    assert!(
        strict_first_constrained.unwrap() < perf_first_constrained.unwrap(),
        "Strict profile (threshold=60) should reach Constrained before Performance (threshold=70)"
    );

    for oracle in &strict_trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn scenario_watch_state_protects_against_premature_breach() {
    let scenario = scenario_insufficient_samples_watch();
    let trace = run_full_simulation(&scenario);

    // First 5 evaluations have < 200 samples, so status=Watch despite bad metrics
    for (i, record) in trace.calibration_trace.iter().take(5).enumerate() {
        assert_eq!(
            record.status,
            CalibrationGuardStatus::Watch,
            "Tick {i}: insufficient samples should yield Watch, got {:?}",
            record.status
        );
        assert!(
            !record.fallback_triggered,
            "Tick {i}: no fallback during Watch"
        );
    }

    // After sample threshold is crossed, breaches should appear
    let has_post_threshold_breach = trace
        .calibration_trace
        .iter()
        .skip(5)
        .any(|record| record.status == CalibrationGuardStatus::Breach);
    assert!(
        has_post_threshold_breach,
        "Bad metrics above sample threshold should produce breaches"
    );

    for oracle in &trace.oracle_results {
        assert!(
            oracle.passed,
            "Oracle {} failed: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn all_scenarios_are_deterministic() {
    let scenarios = vec![
        scenario_gradual_ramp_up(),
        scenario_spike_and_recovery(),
        scenario_hysteresis_oscillation(),
        scenario_hard_pause_and_clear(),
        scenario_calibration_breach_fallback(),
        scenario_quality_circuit_breaker(),
        scenario_multi_profile_comparison(),
        scenario_insufficient_samples_watch(),
    ];

    for scenario in &scenarios {
        let result = oracle_determinism(scenario);
        assert!(
            result.passed,
            "Scenario '{}' is not deterministic: {}",
            scenario.name, result.detail
        );
    }
}

#[test]
fn artifact_emission_produces_valid_jsonl() {
    let scenario = scenario_spike_and_recovery();
    let trace = run_full_simulation(&scenario);

    let tmp = tempfile::tempdir().expect("create temp dir");
    let artifact_dir = tmp.path().join(&scenario.name);
    emit_trace_artifact(&trace, &artifact_dir);

    // Verify events file exists and is valid JSONL
    let events_path = artifact_dir.join("simulation_events.jsonl");
    assert!(events_path.exists(), "Events file should be created");
    let contents = fs::read_to_string(&events_path).expect("read events file");
    let line_count = contents.lines().count();
    assert!(
        line_count > 0,
        "Events file should contain at least one line"
    );

    // Every line should be valid JSON
    for (i, line) in contents.lines().enumerate() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Line {i} is not valid JSON: {e}"));
        assert!(parsed.is_object(), "Line {i} should be a JSON object");
    }

    // Verify oracle results file
    let oracle_path = artifact_dir.join("oracle_results.json");
    assert!(
        oracle_path.exists(),
        "Oracle results file should be created"
    );
    let oracle_contents = fs::read_to_string(&oracle_path).expect("read oracle file");
    let oracles: Vec<OracleResult> =
        serde_json::from_str(&oracle_contents).expect("parse oracle results");
    assert!(!oracles.is_empty(), "Should have oracle results");
    for oracle in &oracles {
        assert!(
            oracle.passed,
            "Oracle {} should pass: {}",
            oracle.oracle_name, oracle.detail
        );
    }
}

#[test]
fn artifact_bundle_emission_writes_replay_manifest() {
    let scenario = scenario_gradual_ramp_up();
    let trace = run_full_simulation(&scenario);

    let tmp = tempfile::tempdir().expect("create temp dir");
    emit_trace_artifact_bundle(&trace, tmp.path());

    let scenario_dir = tmp.path().join(&scenario.name);
    let manifest_path = scenario_dir.join("replay_manifest.json");
    assert!(manifest_path.exists(), "Replay manifest should be created");

    let manifest_contents = fs::read_to_string(&manifest_path).expect("read replay manifest");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_contents).expect("parse replay manifest");
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["scenario"], scenario.name);
    assert_eq!(
        manifest["artifacts"]["events_jsonl"],
        "simulation_events.jsonl"
    );
    assert_eq!(
        manifest["artifacts"]["oracle_results_json"],
        "oracle_results.json"
    );
    assert!(
        manifest["replay"]["command"]
            .as_str()
            .expect("replay command as string")
            .contains("full_simulation_suite_all_oracles_pass"),
        "Replay command should target full simulation suite test"
    );
}

#[test]
fn full_simulation_suite_all_oracles_pass() {
    let scenarios = vec![
        scenario_gradual_ramp_up(),
        scenario_spike_and_recovery(),
        scenario_hysteresis_oscillation(),
        scenario_hard_pause_and_clear(),
        scenario_calibration_breach_fallback(),
        scenario_quality_circuit_breaker(),
        scenario_multi_profile_comparison(),
        scenario_insufficient_samples_watch(),
    ];

    let mut total_oracles = 0;
    let mut failed_oracles = Vec::new();

    for scenario in &scenarios {
        let trace = run_full_simulation(scenario);
        maybe_emit_trace_artifact_bundle(&trace);
        for oracle in &trace.oracle_results {
            total_oracles += 1;
            if !oracle.passed {
                failed_oracles.push(format!(
                    "[{}] {}: {}",
                    scenario.name, oracle.oracle_name, oracle.detail
                ));
            }
        }
    }

    assert!(
        failed_oracles.is_empty(),
        "Failed {}/{} oracles:\n{}",
        failed_oracles.len(),
        total_oracles,
        failed_oracles.join("\n")
    );

    eprintln!(
        "All {total_oracles} oracles passed across {} scenarios",
        scenarios.len()
    );
}
