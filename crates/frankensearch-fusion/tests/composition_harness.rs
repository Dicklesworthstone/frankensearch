//! Composition interference harness for controller interaction testing.
//!
//! Tests pairwise and multi-way interactions between the adaptive ranking
//! controllers to verify:
//! 1. No controller corrupts another's state.
//! 2. Timescale separation: controllers operate at intended cadences.
//! 3. Deterministic fallback: when controllers conflict, the pipeline degrades
//!    safely and predictably.
//!
//! # Composition Set
//!
//! - Circuit breaker (bd-1do): binary skip/allow, per-query
//! - Phase gate (bd-2ps): e-process accumulation, per-query
//! - Feedback loop (bd-2tv): boost map, per-signal
//! - Score calibration (bd-22k): offline-trained, per-query application
//! - MMR diversity (bd-z3j): per-result-set reranking

use frankensearch_fusion::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, QualityOutcome};
use frankensearch_fusion::feedback::{FeedbackCollector, FeedbackConfig, FeedbackSignal};
use frankensearch_fusion::phase_gate::{PhaseGate, PhaseGateConfig, PhaseObservation};

use frankensearch_core::decision_plane::PipelineState;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn fast_circuit_breaker() -> CircuitBreaker {
    CircuitBreaker::new(CircuitBreakerConfig {
        enabled: true,
        failure_threshold: 3,
        latency_threshold_ms: 200,
        improvement_threshold: 0.05,
        half_open_interval_ms: 10,
        reset_threshold: 2,
    })
}

fn fast_phase_gate() -> PhaseGate {
    PhaseGate::new(PhaseGateConfig {
        alpha: 0.05,
        timeout_queries: 20,
        min_delta: 0.01,
        enabled: true,
        expiry_ms: None,
    })
}

fn enabled_feedback() -> FeedbackCollector {
    FeedbackCollector::new(FeedbackConfig {
        enabled: true,
        decay_halflife_hours: 168.0,
        max_boost: 2.0,
        min_boost: 0.5,
        max_entries: 1000,
        cleanup_threshold: 0.01,
        ..Default::default()
    })
}

// ─── Pairwise: Circuit Breaker x Phase Gate ──────────────────────────────────

#[test]
fn circuit_breaker_and_phase_gate_do_not_interfere() {
    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();

    // Scenario: quality tier is slow (triggers circuit breaker) but also
    // genuinely improves results (phase gate accumulates quality evidence).
    for _ in 0..3 {
        // Circuit breaker sees slow latency.
        cb.record_outcome(&QualityOutcome::Slow { latency_ms: 300 });

        // Phase gate sees quality improvement.
        gate.update(&PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.8,
            user_signal: None,
        });
    }

    // Circuit breaker should be open (latency exceeded).
    assert!(cb.is_open(), "circuit breaker should trip on slow latency");

    // Phase gate should be accumulating evidence for quality (scores show
    // quality wins), independent of the circuit breaker's latency decision.
    assert!(
        gate.e_value() > 1.0,
        "phase gate should accumulate pro-quality evidence"
    );

    // Key: the two controllers made INDEPENDENT decisions based on different
    // signals. No state leaked between them.
}

#[test]
fn circuit_breaker_skip_overrides_phase_gate_refine() {
    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();

    // Phase gate decides to always refine (lots of quality evidence).
    for _ in 0..20 {
        gate.update(&PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.9,
            user_signal: Some(true),
        });
    }

    // Circuit breaker trips due to errors.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }

    // Even though phase gate says "always refine", circuit breaker should
    // override by skipping quality tier entirely.
    let (skip, _) = cb.should_skip_quality();
    assert!(skip, "circuit breaker should override phase gate");

    // This is the correct composition: safety (circuit breaker) > optimization
    // (phase gate). The consumer should check circuit breaker FIRST.
}

// ─── Pairwise: Circuit Breaker x Feedback ────────────────────────────────────

#[test]
fn feedback_boosts_survive_circuit_breaker_trip() {
    let cb = fast_circuit_breaker();
    let fc = enabled_feedback();

    // Build up feedback boosts.
    fc.record_signal(&FeedbackSignal::Select {
        doc_id: "important_doc".into(),
    });
    let boost_before = fc.get_boost("important_doc");
    assert!(boost_before > 1.0);

    // Trip the circuit breaker.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }
    assert!(cb.is_open());

    // Feedback boosts should be unchanged — they operate on a different layer
    // (post-RRF scoring, not phase decision).
    let boost_after = fc.get_boost("important_doc");
    assert!(
        (boost_before - boost_after).abs() < 0.001,
        "feedback boosts should survive circuit breaker trips"
    );
}

#[test]
fn feedback_applies_even_when_circuit_open() {
    let cb = fast_circuit_breaker();
    let fc = enabled_feedback();

    // Trip circuit breaker.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }
    assert!(cb.is_open());

    // Record feedback signals — should still work.
    fc.record_signal(&FeedbackSignal::Click {
        doc_id: "doc1".into(),
        rank: 1,
    });

    // Apply boosts — should modify scores even with open circuit.
    let mut results = vec![("doc1".to_string(), 0.8)];
    fc.apply_boosts(&mut results);

    assert!(
        results[0].1 > 0.8,
        "feedback boosts should apply regardless of circuit state"
    );
}

// ─── Pairwise: Phase Gate x Feedback ─────────────────────────────────────────

#[test]
fn feedback_signals_inform_phase_gate() {
    let fc = enabled_feedback();
    let mut gate = fast_phase_gate();

    // Simulate: quality tier produces better results AND users engage with them.
    for _ in 0..10 {
        // Phase gate observation: quality wins.
        gate.update(&PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.7,
            user_signal: Some(true), // user engaged
        });

        // Feedback loop records corresponding signal.
        fc.record_signal(&FeedbackSignal::Select {
            doc_id: "quality_doc".into(),
        });
    }

    // Phase gate should strongly favor refining.
    assert!(
        gate.e_value() > 5.0,
        "user engagement should amplify phase gate evidence"
    );

    // Feedback should have boosted the engaged document.
    assert!(fc.get_boost("quality_doc") > 1.0);
}

// ─── Multi-way: All Three Controllers ────────────────────────────────────────

#[test]
fn three_way_composition_deterministic_under_mixed_signals() {
    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();
    let fc = enabled_feedback();

    // Phase 1: Everything nominal. Quality tier works well.
    for i in 0..5 {
        cb.record_outcome(&QualityOutcome::Success {
            latency_ms: 100,
            tau_improvement: 0.2,
        });
        gate.update(&PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.7,
            user_signal: Some(true),
        });
        fc.record_signal(&FeedbackSignal::Click {
            doc_id: format!("doc{i}"),
            rank: 1,
        });
    }

    // All controllers should be in "positive" state.
    assert!(cb.is_closed());
    assert!(gate.e_value() > 1.0);
    assert!(fc.get_boost("doc0") > 1.0);

    // Phase 2: Quality tier starts failing.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
        gate.update(&PhaseObservation {
            fast_score: 0.7,
            quality_score: 0.3, // quality now worse
            user_signal: Some(false),
        });
        fc.record_signal(&FeedbackSignal::Skip {
            doc_id: "bad_doc".into(),
            rank: 1,
        });
    }

    // Circuit breaker should trip.
    assert!(cb.is_open());

    // Phase gate e-value should decrease (evidence shifting toward fast).
    // It may or may not have reached a decision depending on how much evidence
    // accumulated, but the e-value trend should be downward.

    // Feedback should reflect the mixed history.
    assert!(fc.get_boost("bad_doc") < 1.0);

    // Key invariant: each controller's state transition is deterministic
    // and independent. No controller "pulled" another into an incorrect state.
}

// ─── Timescale Separation ────────────────────────────────────────────────────

#[test]
fn timescale_separation_circuit_breaker_fastest() {
    // Circuit breaker decides per-query (fastest cadence).
    // Phase gate accumulates over many queries (slower cadence).
    // Feedback accumulates over user sessions (slowest cadence).

    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();
    let fc = enabled_feedback();

    // After 3 queries, circuit breaker can already make a decision.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }
    assert!(cb.is_open(), "circuit breaker should decide in 3 queries");

    // Phase gate needs more observations to reach statistical significance.
    for _ in 0..3 {
        gate.update(&PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.3,
            user_signal: None,
        });
    }
    // Phase gate should NOT have decided yet with only 3 weak observations.
    assert!(
        gate.decision().is_none(),
        "phase gate should need more observations than circuit breaker"
    );

    // Feedback loop is the slowest: it builds up document-level knowledge.
    for _ in 0..3 {
        fc.record_signal(&FeedbackSignal::Click {
            doc_id: "doc1".into(),
            rank: 1,
        });
    }
    let boost = fc.get_boost("doc1");
    // Feedback effect is gradual (0.1 weight per signal).
    assert!(
        boost < 1.5,
        "feedback should change slowly: boost={boost:.4}"
    );
}

// ─── Deterministic Fallback ──────────────────────────────────────────────────

#[test]
fn all_controllers_disabled_produces_passthrough() {
    let cb = CircuitBreaker::new(CircuitBreakerConfig {
        enabled: false,
        ..CircuitBreakerConfig::default()
    });
    let gate = PhaseGate::new(PhaseGateConfig {
        enabled: false,
        ..PhaseGateConfig::default()
    });
    let fc = FeedbackCollector::new(FeedbackConfig {
        enabled: false,
        ..FeedbackConfig::default()
    });

    // Circuit breaker: never skips.
    let (skip, evidence) = cb.should_skip_quality();
    assert!(!skip);
    assert!(evidence.is_empty());

    // Phase gate: no decision.
    assert!(gate.decision().is_none());
    assert!(!gate.should_skip_quality());

    // Feedback: neutral boosts.
    assert!((fc.get_boost("anything") - 1.0).abs() < f64::EPSILON);

    // Net effect: full two-tier pipeline with no modifications. This is
    // the safe default when all controllers are disabled.
}

#[test]
fn conflicting_decisions_resolve_via_priority() {
    // Simulate a conflict: phase gate says "skip" but circuit breaker is
    // closed (allow). Resolution: circuit breaker is checked first. If it
    // allows, THEN check phase gate.

    let cb = fast_circuit_breaker(); // closed → allows quality
    let mut gate = fast_phase_gate();

    // Force phase gate to decide "skip quality".
    for _ in 0..20 {
        gate.update(&PhaseObservation {
            fast_score: 0.8,
            quality_score: 0.3, // fast consistently better
            user_signal: Some(false),
        });
    }

    let cb_skip = cb.should_skip_quality().0;
    let gate_skip = gate.should_skip_quality();

    // Circuit breaker allows (closed), phase gate says skip.
    assert!(!cb_skip, "circuit breaker should allow");
    assert!(gate_skip, "phase gate should recommend skip");

    // Composition rule: check circuit breaker first.
    // If CB says skip → skip (safety override, strongest signal).
    // If CB allows → defer to phase gate.
    let final_skip = cb_skip || gate_skip;
    assert!(
        final_skip,
        "composed decision should skip (phase gate wins when CB allows)"
    );
}

// ─── State Reset Independence ────────────────────────────────────────────────

#[test]
fn resetting_one_controller_does_not_affect_others() {
    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();
    let fc = enabled_feedback();

    // Build up state in all controllers.
    cb.record_outcome(&QualityOutcome::Error);
    cb.record_outcome(&QualityOutcome::Error);
    gate.update(&PhaseObservation {
        fast_score: 0.5,
        quality_score: 0.7,
        user_signal: None,
    });
    fc.record_signal(&FeedbackSignal::Click {
        doc_id: "doc1".into(),
        rank: 1,
    });

    let gate_e_before = gate.e_value();
    let fc_boost_before = fc.get_boost("doc1");

    // Force-reset the circuit breaker.
    cb.force_close();
    assert!(cb.is_closed());

    // Phase gate and feedback should be completely unaffected.
    assert!(
        (gate.e_value() - gate_e_before).abs() < f64::EPSILON,
        "phase gate e-value should be unchanged after CB reset"
    );
    let fc_boost_after = fc.get_boost("doc1");
    assert!(
        (fc_boost_before - fc_boost_after).abs() < 0.001,
        "feedback boost should be unchanged after CB reset"
    );
}

#[test]
fn phase_gate_reset_does_not_affect_others() {
    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();
    let fc = enabled_feedback();

    // Build up state.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }
    gate.update(&PhaseObservation {
        fast_score: 0.5,
        quality_score: 0.7,
        user_signal: None,
    });
    fc.record_signal(&FeedbackSignal::Click {
        doc_id: "doc1".into(),
        rank: 1,
    });

    // Reset phase gate.
    gate.reset();

    // CB should still be open (tripped from errors).
    assert!(cb.is_open(), "CB should still be open after gate reset");

    // Feedback should be unchanged.
    assert!(
        fc.get_boost("doc1") > 1.0,
        "feedback should survive gate reset"
    );
}

// ─── Pipeline State Consistency ──────────────────────────────────────────────

#[test]
fn pipeline_state_reflects_worst_controller() {
    let cb = fast_circuit_breaker();

    // Nominal: circuit breaker closed.
    assert_eq!(cb.pipeline_state(), PipelineState::Nominal);

    // Trip it.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }
    assert_eq!(cb.pipeline_state(), PipelineState::CircuitOpen);

    // Wait for half-open.
    std::thread::sleep(std::time::Duration::from_millis(20));
    cb.should_skip_quality(); // triggers transition
    assert_eq!(cb.pipeline_state(), PipelineState::Probing);
}

// ─── Evidence Record Non-Interference ────────────────────────────────────────

#[test]
fn evidence_records_carry_correct_source() {
    let cb = fast_circuit_breaker();
    let mut gate = fast_phase_gate();

    // Trip circuit breaker.
    for _ in 0..3 {
        cb.record_outcome(&QualityOutcome::Error);
    }

    // Feed phase gate enough to reach decision.
    let mut gate_evidence = Vec::new();
    for _ in 0..20 {
        let ev = gate.update(&PhaseObservation {
            fast_score: 0.5,
            quality_score: 0.9,
            user_signal: Some(true),
        });
        gate_evidence.extend(ev);
    }

    // CB evidence records should come from "circuit_breaker".
    let cb_trip = cb.force_open();
    for record in &cb_trip {
        assert_eq!(
            record.source_component, "circuit_breaker",
            "CB evidence should have source_component=circuit_breaker"
        );
    }

    // Phase gate evidence should come from "phase_gate".
    for record in &gate_evidence {
        assert_eq!(
            record.source_component, "phase_gate",
            "gate evidence should have source_component=phase_gate"
        );
    }
}
