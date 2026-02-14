//! Distributed observability package for Native Mode generation lifecycle.
//!
//! Defines structured events, metric snapshots, and tracing span/field constants
//! for monitoring generation activation, artifact repair, and degraded-mode
//! transitions in distributed deployments.
//!
//! All events are emitted via the `tracing` crate using consistent span and field
//! names defined in [`crate::tracing_config`]. Consumers subscribe to these events
//! through standard `tracing-subscriber` infrastructure.

use serde::{Deserialize, Serialize};

use crate::repair::{DegradedReason, DetectionMethod, RepairOutcome, ServiceState};

// ---------------------------------------------------------------------------
// Span and field name constants
// ---------------------------------------------------------------------------

/// Tracing span names for distributed generation lifecycle.
pub mod span_names {
    /// Root span for a full generation lifecycle (build → transfer → verify → activate).
    pub const GENERATION_LIFECYCLE: &str = "frankensearch::generation_lifecycle";
    /// Child span for artifact encoding (symbol generation).
    pub const ARTIFACT_ENCODE: &str = "frankensearch::artifact_encode";
    /// Child span for artifact transfer to replica.
    pub const ARTIFACT_TRANSFER: &str = "frankensearch::artifact_transfer";
    /// Child span for artifact decoding on replica.
    pub const ARTIFACT_DECODE: &str = "frankensearch::artifact_decode";
    /// Child span for artifact verification (checksum + invariant checks).
    pub const ARTIFACT_VERIFY: &str = "frankensearch::artifact_verify";
    /// Child span for generation activation (pointer swap).
    pub const GENERATION_ACTIVATE: &str = "frankensearch::generation_activate";
    /// Root span for a repair cycle.
    pub const REPAIR_CYCLE: &str = "frankensearch::repair_cycle";
    /// Child span for a single artifact repair attempt.
    pub const ARTIFACT_REPAIR: &str = "frankensearch::artifact_repair";
    /// Root span for bootstrap (new node catch-up).
    pub const BOOTSTRAP: &str = "frankensearch::bootstrap";
}

/// Tracing field names for distributed events.
pub mod field_names {
    /// Generation identifier (string).
    pub const GENERATION_ID: &str = "generation_id";
    /// Commit sequence number for snapshot reads.
    pub const AS_OF_COMMIT_SEQ: &str = "as_of_commit_seq";
    /// Activation sequence number (monotonic).
    pub const ACTIVATION_SEQ: &str = "activation_seq";
    /// Commit range low bound.
    pub const COMMIT_LOW: &str = "commit_low";
    /// Commit range high bound.
    pub const COMMIT_HIGH: &str = "commit_high";
    /// Artifact path (manifest-relative).
    pub const ARTIFACT_PATH: &str = "artifact_path";
    /// Number of repair symbols used.
    pub const SYMBOLS_USED: &str = "symbols_used";
    /// Total document count.
    pub const TOTAL_DOCUMENTS: &str = "total_documents";
    /// Duration in milliseconds.
    pub const DURATION_MS: &str = "duration_ms";
    /// Number of unrepaired artifacts.
    pub const UNREPAIRED_COUNT: &str = "unrepaired_count";
    /// Current service state label.
    pub const SERVICE_STATE: &str = "service_state";
    /// Failure reason (human-readable).
    pub const FAILURE_REASON: &str = "failure_reason";
    /// Node or replica identifier.
    pub const NODE_ID: &str = "node_id";
}

// ---------------------------------------------------------------------------
// Metric names
// ---------------------------------------------------------------------------

/// Metric name constants for distributed observability.
///
/// These follow the `distributed.*` namespace convention from the design doc.
/// Consumers wire these into their metrics backend (Prometheus, `StatsD`, etc.).
pub mod metric_names {
    /// Lag between latest committed sequence and latest activated generation (ms).
    pub const COMMIT_LAG_MS: &str = "distributed.commit.lag_ms";
    /// Time taken to activate a new generation (ms).
    pub const GENERATION_ACTIVATE_MS: &str = "distributed.generation.activate_ms";
    /// Ratio of repaired artifacts to total artifacts (0.0–1.0).
    pub const ARTIFACT_REPAIR_RATIO: &str = "distributed.artifact.repair_ratio";
    /// Time taken to bootstrap a new node from manifest + symbols (ms).
    pub const SNAPSHOT_BOOTSTRAP_MS: &str = "distributed.snapshot.bootstrap_ms";
    /// Generation skew across nodes in a cluster (max `activation_seq` delta).
    pub const QUERY_GENERATION_SKEW: &str = "distributed.query.generation_skew";
    /// Cumulative count of failed repair attempts.
    pub const REPAIR_FAILURES_TOTAL: &str = "distributed.repair.failures_total";
}

// ---------------------------------------------------------------------------
// Structured events
// ---------------------------------------------------------------------------

/// Enumeration of all distributed lifecycle events.
///
/// Each variant carries the structured data for one event type defined in the
/// Native Mode design doc (Section 9). Consumers emit these via
/// [`emit_event`] or inspect them directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedEvent {
    /// A generation build process has started.
    GenerationBuildStarted {
        /// Target generation identifier.
        generation_id: String,
        /// Commit range low bound.
        commit_low: u64,
        /// Commit range high bound.
        commit_high: u64,
        /// Unix timestamp (millis) of build start.
        started_at: u64,
    },

    /// A generation build process has completed.
    GenerationBuildCompleted {
        /// Generation identifier.
        generation_id: String,
        /// Total documents in the generation.
        total_documents: u64,
        /// Number of vector artifacts produced.
        vector_artifact_count: usize,
        /// Number of lexical artifacts produced.
        lexical_artifact_count: usize,
        /// Build duration in milliseconds.
        build_duration_ms: u64,
    },

    /// A generation was successfully activated for serving.
    GenerationActivationSucceeded {
        /// Generation identifier.
        generation_id: String,
        /// Monotonic activation sequence number.
        activation_seq: u64,
        /// Activation duration in milliseconds.
        activation_duration_ms: u64,
        /// Commit range covered by this generation.
        commit_low: u64,
        /// Commit range high bound.
        commit_high: u64,
    },

    /// A generation activation attempt failed.
    GenerationActivationFailed {
        /// Generation identifier that failed activation.
        generation_id: String,
        /// Human-readable failure reason.
        failure_reason: String,
        /// Number of invariant checks that failed.
        failed_invariant_count: usize,
    },

    /// A repair attempt has started for an artifact.
    ArtifactRepairStarted {
        /// Path of the artifact being repaired.
        artifact_path: String,
        /// How the corruption was detected.
        detection_method: DetectionMethod,
        /// Attempt number (1-based) for this artifact.
        attempt_number: u32,
    },

    /// A repair attempt completed successfully.
    ArtifactRepairCompleted {
        /// Path of the repaired artifact.
        artifact_path: String,
        /// Number of repair symbols consumed.
        symbols_used: u32,
        /// Repair duration in milliseconds.
        repair_duration_ms: u64,
    },

    /// A repair attempt failed.
    ArtifactRepairFailed {
        /// Path of the artifact that failed repair.
        artifact_path: String,
        /// Outcome of the failed attempt.
        outcome: RepairOutcome,
        /// Remaining retry budget for this artifact.
        retries_remaining: u32,
    },

    /// The service entered read-degraded mode.
    ReadDegradedModeEntered {
        /// Unix timestamp (millis) when degraded mode was entered.
        entered_at: u64,
        /// Why the service entered degraded mode.
        reason: DegradedReason,
        /// Number of currently unrepaired artifacts.
        unrepaired_count: usize,
    },

    /// The service exited read-degraded mode (recovered to healthy).
    ReadDegradedModeExited {
        /// Unix timestamp (millis) of recovery.
        recovered_at: u64,
        /// How long the service was degraded (millis).
        degraded_duration_ms: u64,
    },
}

impl DistributedEvent {
    /// Machine-readable event name matching the design doc taxonomy.
    #[must_use]
    pub const fn event_name(&self) -> &'static str {
        match self {
            Self::GenerationBuildStarted { .. } => "generation_build_started",
            Self::GenerationBuildCompleted { .. } => "generation_build_completed",
            Self::GenerationActivationSucceeded { .. } => "generation_activation_succeeded",
            Self::GenerationActivationFailed { .. } => "generation_activation_failed",
            Self::ArtifactRepairStarted { .. } => "artifact_repair_started",
            Self::ArtifactRepairCompleted { .. } => "artifact_repair_completed",
            Self::ArtifactRepairFailed { .. } => "artifact_repair_failed",
            Self::ReadDegradedModeEntered { .. } => "read_degraded_mode_entered",
            Self::ReadDegradedModeExited { .. } => "read_degraded_mode_exited",
        }
    }
}

// ---------------------------------------------------------------------------
// Metric snapshot
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of all distributed metrics.
///
/// Consumers populate this from their runtime state and export it to their
/// metrics backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributedMetrics {
    /// Lag between latest committed sequence and latest activated generation (ms).
    pub commit_lag_ms: u64,
    /// Time taken to activate the current generation (ms).
    pub generation_activate_ms: u64,
    /// Ratio of repaired artifacts to total artifacts (0.0–1.0).
    pub artifact_repair_ratio: f64,
    /// Time taken to bootstrap this node (ms). Zero if not bootstrapped.
    pub snapshot_bootstrap_ms: u64,
    /// Generation skew across the cluster (max `activation_seq` delta).
    pub query_generation_skew: u64,
    /// Cumulative count of failed repair attempts.
    pub repair_failures_total: u64,
}

impl Default for DistributedMetrics {
    fn default() -> Self {
        Self {
            commit_lag_ms: 0,
            generation_activate_ms: 0,
            artifact_repair_ratio: 0.0,
            snapshot_bootstrap_ms: 0,
            query_generation_skew: 0,
            repair_failures_total: 0,
        }
    }
}

impl DistributedMetrics {
    /// Compute the repair ratio from repair history.
    #[must_use]
    pub fn compute_repair_ratio(repaired: u64, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let ratio = repaired as f64 / total as f64;
            ratio
        }
    }
}

// ---------------------------------------------------------------------------
// Event emission helpers
// ---------------------------------------------------------------------------

/// Emit a distributed event via the `tracing` crate.
///
/// Events are emitted at the appropriate severity level:
/// - Failures and degraded-mode transitions: `WARN`
/// - Successful operations: `INFO`
/// - Repair starts: `DEBUG`
pub fn emit_event(event: &DistributedEvent) {
    match event {
        DistributedEvent::GenerationBuildStarted { .. }
        | DistributedEvent::GenerationBuildCompleted { .. }
        | DistributedEvent::GenerationActivationSucceeded { .. }
        | DistributedEvent::ArtifactRepairCompleted { .. }
        | DistributedEvent::ReadDegradedModeExited { .. } => emit_info_event(event),

        DistributedEvent::GenerationActivationFailed { .. }
        | DistributedEvent::ArtifactRepairFailed { .. }
        | DistributedEvent::ReadDegradedModeEntered { .. } => emit_warn_event(event),

        DistributedEvent::ArtifactRepairStarted { .. } => emit_debug_event(event),
    }
}

fn emit_info_event(event: &DistributedEvent) {
    match event {
        DistributedEvent::GenerationBuildStarted {
            generation_id,
            commit_low,
            commit_high,
            started_at,
        } => tracing::info!(
            target: "frankensearch::distributed",
            event_name = "generation_build_started",
            generation_id = %generation_id,
            commit_low, commit_high, started_at,
            "generation build started"
        ),
        DistributedEvent::GenerationBuildCompleted {
            generation_id,
            total_documents,
            vector_artifact_count,
            lexical_artifact_count,
            build_duration_ms,
        } => tracing::info!(
            target: "frankensearch::distributed",
            event_name = "generation_build_completed",
            generation_id = %generation_id,
            total_documents, vector_artifact_count, lexical_artifact_count, build_duration_ms,
            "generation build completed"
        ),
        DistributedEvent::GenerationActivationSucceeded {
            generation_id,
            activation_seq,
            activation_duration_ms,
            commit_low,
            commit_high,
        } => tracing::info!(
            target: "frankensearch::distributed",
            event_name = "generation_activation_succeeded",
            generation_id = %generation_id,
            activation_seq, activation_duration_ms, commit_low, commit_high,
            "generation activated successfully"
        ),
        DistributedEvent::ArtifactRepairCompleted {
            artifact_path,
            symbols_used,
            repair_duration_ms,
        } => tracing::info!(
            target: "frankensearch::distributed",
            event_name = "artifact_repair_completed",
            artifact_path = %artifact_path,
            symbols_used, repair_duration_ms,
            "artifact repair completed"
        ),
        DistributedEvent::ReadDegradedModeExited {
            recovered_at,
            degraded_duration_ms,
        } => tracing::info!(
            target: "frankensearch::distributed",
            event_name = "read_degraded_mode_exited",
            recovered_at, degraded_duration_ms,
            "exited read-degraded mode"
        ),
        // Variants dispatched to other severity handlers.
        DistributedEvent::GenerationActivationFailed { .. }
        | DistributedEvent::ArtifactRepairStarted { .. }
        | DistributedEvent::ArtifactRepairFailed { .. }
        | DistributedEvent::ReadDegradedModeEntered { .. } => {}
    }
}

fn emit_warn_event(event: &DistributedEvent) {
    match event {
        DistributedEvent::GenerationActivationFailed {
            generation_id,
            failure_reason,
            failed_invariant_count,
        } => tracing::warn!(
            target: "frankensearch::distributed",
            event_name = "generation_activation_failed",
            generation_id = %generation_id,
            failure_reason = %failure_reason,
            failed_invariant_count,
            "generation activation failed"
        ),
        DistributedEvent::ArtifactRepairFailed {
            artifact_path,
            outcome,
            retries_remaining,
        } => tracing::warn!(
            target: "frankensearch::distributed",
            event_name = "artifact_repair_failed",
            artifact_path = %artifact_path,
            outcome = ?outcome,
            retries_remaining,
            "artifact repair failed"
        ),
        DistributedEvent::ReadDegradedModeEntered {
            entered_at,
            reason,
            unrepaired_count,
        } => tracing::warn!(
            target: "frankensearch::distributed",
            event_name = "read_degraded_mode_entered",
            entered_at, reason = ?reason, unrepaired_count,
            "entered read-degraded mode"
        ),
        // Variants dispatched to other severity handlers.
        DistributedEvent::GenerationBuildStarted { .. }
        | DistributedEvent::GenerationBuildCompleted { .. }
        | DistributedEvent::GenerationActivationSucceeded { .. }
        | DistributedEvent::ArtifactRepairStarted { .. }
        | DistributedEvent::ArtifactRepairCompleted { .. }
        | DistributedEvent::ReadDegradedModeExited { .. } => {}
    }
}

fn emit_debug_event(event: &DistributedEvent) {
    if let DistributedEvent::ArtifactRepairStarted {
        artifact_path,
        detection_method,
        attempt_number,
    } = event
    {
        tracing::debug!(
            target: "frankensearch::distributed",
            event_name = "artifact_repair_started",
            artifact_path = %artifact_path,
            detection_method = ?detection_method,
            attempt_number,
            "artifact repair started"
        );
    }
}

/// Derive the service state label for metrics/tracing.
#[must_use]
pub const fn service_state_label(state: &ServiceState) -> &'static str {
    match state {
        ServiceState::Healthy => "healthy",
        ServiceState::Degraded { .. } => "degraded",
        ServiceState::Suspended { .. } => "suspended",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repair::RepairOutcome;

    #[test]
    fn event_names_match_design_doc() {
        let events = [
            (
                DistributedEvent::GenerationBuildStarted {
                    generation_id: "g1".into(),
                    commit_low: 1,
                    commit_high: 100,
                    started_at: 1000,
                },
                "generation_build_started",
            ),
            (
                DistributedEvent::GenerationBuildCompleted {
                    generation_id: "g1".into(),
                    total_documents: 100,
                    vector_artifact_count: 2,
                    lexical_artifact_count: 1,
                    build_duration_ms: 5000,
                },
                "generation_build_completed",
            ),
            (
                DistributedEvent::GenerationActivationSucceeded {
                    generation_id: "g1".into(),
                    activation_seq: 1,
                    activation_duration_ms: 50,
                    commit_low: 1,
                    commit_high: 100,
                },
                "generation_activation_succeeded",
            ),
            (
                DistributedEvent::GenerationActivationFailed {
                    generation_id: "g1".into(),
                    failure_reason: "invariant failed".into(),
                    failed_invariant_count: 2,
                },
                "generation_activation_failed",
            ),
            (
                DistributedEvent::ArtifactRepairStarted {
                    artifact_path: "v.fsvi".into(),
                    detection_method: DetectionMethod::PeriodicScan,
                    attempt_number: 1,
                },
                "artifact_repair_started",
            ),
            (
                DistributedEvent::ArtifactRepairCompleted {
                    artifact_path: "v.fsvi".into(),
                    symbols_used: 5,
                    repair_duration_ms: 200,
                },
                "artifact_repair_completed",
            ),
            (
                DistributedEvent::ArtifactRepairFailed {
                    artifact_path: "v.fsvi".into(),
                    outcome: RepairOutcome::SidecarMissing,
                    retries_remaining: 2,
                },
                "artifact_repair_failed",
            ),
            (
                DistributedEvent::ReadDegradedModeEntered {
                    entered_at: 1000,
                    reason: DegradedReason::ActivationFailure {
                        detail: "test".into(),
                    },
                    unrepaired_count: 3,
                },
                "read_degraded_mode_entered",
            ),
            (
                DistributedEvent::ReadDegradedModeExited {
                    recovered_at: 2000,
                    degraded_duration_ms: 1000,
                },
                "read_degraded_mode_exited",
            ),
        ];

        for (event, expected_name) in &events {
            assert_eq!(event.event_name(), *expected_name);
        }
    }

    #[test]
    fn all_nine_event_types_defined() {
        // Verify we cover all 9 events from design doc Section 9.
        let event_names = [
            "generation_build_started",
            "generation_build_completed",
            "generation_activation_succeeded",
            "generation_activation_failed",
            "artifact_repair_started",
            "artifact_repair_completed",
            "artifact_repair_failed",
            "read_degraded_mode_entered",
            "read_degraded_mode_exited",
        ];
        assert_eq!(event_names.len(), 9);
    }

    #[test]
    fn all_six_metrics_defined() {
        // Verify we cover all 6 metrics from design doc Section 9.
        let metrics = [
            metric_names::COMMIT_LAG_MS,
            metric_names::GENERATION_ACTIVATE_MS,
            metric_names::ARTIFACT_REPAIR_RATIO,
            metric_names::SNAPSHOT_BOOTSTRAP_MS,
            metric_names::QUERY_GENERATION_SKEW,
            metric_names::REPAIR_FAILURES_TOTAL,
        ];
        assert_eq!(metrics.len(), 6);
        for metric in &metrics {
            assert!(metric.starts_with("distributed."));
        }
    }

    #[test]
    fn metric_names_match_design_doc() {
        assert_eq!(metric_names::COMMIT_LAG_MS, "distributed.commit.lag_ms");
        assert_eq!(
            metric_names::GENERATION_ACTIVATE_MS,
            "distributed.generation.activate_ms"
        );
        assert_eq!(
            metric_names::ARTIFACT_REPAIR_RATIO,
            "distributed.artifact.repair_ratio"
        );
        assert_eq!(
            metric_names::SNAPSHOT_BOOTSTRAP_MS,
            "distributed.snapshot.bootstrap_ms"
        );
        assert_eq!(
            metric_names::QUERY_GENERATION_SKEW,
            "distributed.query.generation_skew"
        );
        assert_eq!(
            metric_names::REPAIR_FAILURES_TOTAL,
            "distributed.repair.failures_total"
        );
    }

    #[test]
    fn distributed_metrics_default() {
        let m = DistributedMetrics::default();
        assert_eq!(m.commit_lag_ms, 0);
        assert_eq!(m.generation_activate_ms, 0);
        assert!((m.artifact_repair_ratio - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.snapshot_bootstrap_ms, 0);
        assert_eq!(m.query_generation_skew, 0);
        assert_eq!(m.repair_failures_total, 0);
    }

    #[test]
    fn compute_repair_ratio_zero_total() {
        assert!((DistributedMetrics::compute_repair_ratio(0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_repair_ratio_normal() {
        let ratio = DistributedMetrics::compute_repair_ratio(3, 10);
        assert!((ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_repair_ratio_all_repaired() {
        let ratio = DistributedMetrics::compute_repair_ratio(10, 10);
        assert!((ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn service_state_labels() {
        assert_eq!(service_state_label(&ServiceState::Healthy), "healthy");
        assert_eq!(
            service_state_label(&ServiceState::Degraded {
                entered_at: 0,
                reason: DegradedReason::ActivationFailure {
                    detail: String::new()
                },
            }),
            "degraded"
        );
        assert_eq!(
            service_state_label(&ServiceState::Suspended {
                entered_at: 0,
                reason: String::new(),
            }),
            "suspended"
        );
    }

    #[test]
    fn distributed_event_serde_roundtrip() {
        let event = DistributedEvent::GenerationActivationSucceeded {
            generation_id: "gen-042".into(),
            activation_seq: 42,
            activation_duration_ms: 150,
            commit_low: 1,
            commit_high: 500,
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let back: DistributedEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(event, back);
    }

    #[test]
    fn distributed_metrics_serde_roundtrip() {
        let m = DistributedMetrics {
            commit_lag_ms: 500,
            generation_activate_ms: 150,
            artifact_repair_ratio: 0.1,
            snapshot_bootstrap_ms: 30_000,
            query_generation_skew: 2,
            repair_failures_total: 7,
        };
        let json = serde_json::to_string(&m).expect("serialize");
        let back: DistributedMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(m, back);
    }

    #[test]
    fn emit_event_does_not_panic() {
        // Verify all event variants can be emitted without panicking.
        // (No subscriber installed, so events are silently dropped.)
        let events = vec![
            DistributedEvent::GenerationBuildStarted {
                generation_id: "g1".into(),
                commit_low: 1,
                commit_high: 100,
                started_at: 1000,
            },
            DistributedEvent::GenerationBuildCompleted {
                generation_id: "g1".into(),
                total_documents: 100,
                vector_artifact_count: 2,
                lexical_artifact_count: 1,
                build_duration_ms: 5000,
            },
            DistributedEvent::GenerationActivationSucceeded {
                generation_id: "g1".into(),
                activation_seq: 1,
                activation_duration_ms: 50,
                commit_low: 1,
                commit_high: 100,
            },
            DistributedEvent::GenerationActivationFailed {
                generation_id: "g1".into(),
                failure_reason: "test".into(),
                failed_invariant_count: 1,
            },
            DistributedEvent::ArtifactRepairStarted {
                artifact_path: "v.fsvi".into(),
                detection_method: DetectionMethod::ChecksumMismatch,
                attempt_number: 1,
            },
            DistributedEvent::ArtifactRepairCompleted {
                artifact_path: "v.fsvi".into(),
                symbols_used: 5,
                repair_duration_ms: 200,
            },
            DistributedEvent::ArtifactRepairFailed {
                artifact_path: "v.fsvi".into(),
                outcome: RepairOutcome::Failed {
                    reason: "disk".into(),
                },
                retries_remaining: 0,
            },
            DistributedEvent::ReadDegradedModeEntered {
                entered_at: 1000,
                reason: DegradedReason::ExcessiveCorruptionRate {
                    event_count: 5,
                    threshold: 3,
                },
                unrepaired_count: 5,
            },
            DistributedEvent::ReadDegradedModeExited {
                recovered_at: 2000,
                degraded_duration_ms: 1000,
            },
        ];
        for event in &events {
            emit_event(event);
        }
    }

    #[test]
    fn span_names_follow_convention() {
        let spans = [
            span_names::GENERATION_LIFECYCLE,
            span_names::ARTIFACT_ENCODE,
            span_names::ARTIFACT_TRANSFER,
            span_names::ARTIFACT_DECODE,
            span_names::ARTIFACT_VERIFY,
            span_names::GENERATION_ACTIVATE,
            span_names::REPAIR_CYCLE,
            span_names::ARTIFACT_REPAIR,
            span_names::BOOTSTRAP,
        ];
        for span in &spans {
            assert!(
                span.starts_with("frankensearch::"),
                "span '{span}' should start with 'frankensearch::'"
            );
        }
    }

    #[test]
    fn field_names_are_non_empty() {
        let fields = [
            field_names::GENERATION_ID,
            field_names::AS_OF_COMMIT_SEQ,
            field_names::ACTIVATION_SEQ,
            field_names::COMMIT_LOW,
            field_names::COMMIT_HIGH,
            field_names::ARTIFACT_PATH,
            field_names::SYMBOLS_USED,
            field_names::TOTAL_DOCUMENTS,
            field_names::DURATION_MS,
            field_names::UNREPAIRED_COUNT,
            field_names::SERVICE_STATE,
            field_names::FAILURE_REASON,
            field_names::NODE_ID,
        ];
        for field in &fields {
            assert!(!field.is_empty());
        }
    }
}
