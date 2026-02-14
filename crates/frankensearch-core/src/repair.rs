//! Repair orchestration and degraded-mode routing for Native Mode distributed search.
//!
//! When artifact corruption is detected (via periodic checksum scans or read-time
//! verification), the [`RepairOrchestrator`] coordinates repair attempts using
//! `RaptorQ` symbol reconstruction, tracks corruption thresholds, and manages
//! service state transitions between healthy, degraded, and suspended modes.
//!
//! The orchestrator integrates with [`GenerationController`](crate::activation::GenerationController):
//! corruption detection can trigger rollback to the previous generation, and
//! suspension blocks new activation attempts until the operator intervenes.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::generation::{GenerationManifest, RepairDescriptor};

// ---------------------------------------------------------------------------
// Corruption detection
// ---------------------------------------------------------------------------

/// How corruption was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Background periodic checksum scan.
    PeriodicScan,
    /// Detected during a read operation (query-time).
    ReadTimeVerification,
    /// Explicit checksum comparison failed.
    ChecksumMismatch,
    /// The `.fec` sidecar file is missing or unreadable.
    MissingSidecar,
}

/// Record of a detected artifact corruption event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorruptionEvent {
    /// Path of the corrupted artifact (manifest-relative).
    pub artifact_path: String,
    /// How the corruption was detected.
    pub detection_method: DetectionMethod,
    /// Unix timestamp (millis) when corruption was detected.
    pub detected_at: u64,
    /// Human-readable detail about the corruption.
    pub detail: String,
}

// ---------------------------------------------------------------------------
// Repair tracking
// ---------------------------------------------------------------------------

/// Outcome of a single repair attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairOutcome {
    /// Repair succeeded — artifact passes verification after reconstruction.
    Success {
        /// Number of repair symbols consumed during reconstruction.
        symbols_used: u32,
    },
    /// Not enough repair symbols to reconstruct the artifact.
    InsufficientSymbols {
        /// Symbols available from the sidecar.
        available: u32,
        /// Minimum symbols required for reconstruction.
        required: u32,
    },
    /// The `.fec` sidecar file is missing.
    SidecarMissing,
    /// The `.fec` sidecar itself is corrupted.
    SidecarCorrupted,
    /// Repair failed for another reason.
    Failed {
        /// Why the repair failed.
        reason: String,
    },
}

impl RepairOutcome {
    /// Whether the repair was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }
}

/// Record of a repair attempt for a single artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairAttempt {
    /// Path of the artifact that was repaired (manifest-relative).
    pub artifact_path: String,
    /// Unix timestamp (millis) when the repair attempt started.
    pub started_at: u64,
    /// Unix timestamp (millis) when the repair attempt completed.
    pub completed_at: u64,
    /// Outcome of this attempt.
    pub outcome: RepairOutcome,
}

// ---------------------------------------------------------------------------
// Repair provider trait
// ---------------------------------------------------------------------------

/// Consumer-implemented trait for performing artifact repairs.
///
/// The orchestrator calls into this trait when corruption is detected and a
/// repair is warranted by the corruption policy. Implementations know how
/// to locate sidecar files, invoke `RaptorQ` symbol reconstruction, and
/// verify the repaired artifact.
pub trait RepairProvider: Send + Sync {
    /// Attempt to repair a corrupted artifact using its `.fec` sidecar.
    ///
    /// `descriptor` contains the sidecar path and symbol counts from the manifest.
    /// Implementations should:
    /// 1. Back up the corrupted artifact before destructive repair.
    /// 2. Decode source symbols + repair symbols via the `RaptorQ` codec.
    /// 3. Write the reconstructed artifact to the original path.
    /// 4. Return the outcome.
    fn attempt_repair(
        &self,
        artifact_path: &str,
        descriptor: &RepairDescriptor,
        now_millis: u64,
    ) -> RepairAttempt;

    /// Verify an artifact after repair by recomputing its checksum.
    ///
    /// Returns `true` if the repaired file matches `expected_checksum`.
    fn verify_after_repair(&self, artifact_path: &str, expected_checksum: &str) -> bool;
}

// ---------------------------------------------------------------------------
// Service state
// ---------------------------------------------------------------------------

/// Why the service entered degraded mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradedReason {
    /// One or more artifacts could not be repaired.
    UnrepairableCorruption {
        /// Paths of artifacts that failed repair.
        failed_artifacts: Vec<String>,
    },
    /// Corruption rate exceeds the policy threshold.
    ExcessiveCorruptionRate {
        /// Number of corruption events within the tracking window.
        event_count: usize,
        /// Policy threshold that was exceeded.
        threshold: usize,
    },
    /// A generation activation attempt failed.
    ActivationFailure {
        /// Details of the activation failure.
        detail: String,
    },
}

/// Current service state of the repair orchestrator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceState {
    /// All artifacts verified, no outstanding corruption.
    Healthy,
    /// Service is degraded — serving from last healthy generation with known issues.
    Degraded {
        /// When degraded mode was entered (Unix millis).
        entered_at: u64,
        /// Why the service is degraded.
        reason: DegradedReason,
    },
    /// Service is suspended — new activations are blocked until operator intervention.
    Suspended {
        /// When suspension was entered (Unix millis).
        entered_at: u64,
        /// Why the service was suspended.
        reason: String,
    },
}

impl ServiceState {
    /// Whether the service is in a healthy state.
    #[must_use]
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Whether new generation activations should be blocked.
    #[must_use]
    pub const fn is_suspended(&self) -> bool {
        matches!(self, Self::Suspended { .. })
    }
}

// ---------------------------------------------------------------------------
// Corruption policy
// ---------------------------------------------------------------------------

/// Configurable thresholds governing when the orchestrator transitions between
/// service states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorruptionPolicy {
    /// Maximum number of corrupted artifacts before entering degraded mode.
    pub max_corrupted_artifacts: usize,
    /// Maximum repair attempts per artifact before giving up.
    pub max_repair_attempts_per_artifact: u32,
    /// Number of unrepaired artifacts that trigger suspension.
    pub suspension_threshold: usize,
    /// Cooldown period (millis) after suspension before allowing recovery.
    pub cooldown_after_suspension_ms: u64,
}

impl Default for CorruptionPolicy {
    fn default() -> Self {
        Self {
            max_corrupted_artifacts: 3,
            max_repair_attempts_per_artifact: 3,
            suspension_threshold: 5,
            cooldown_after_suspension_ms: 60_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Repair orchestrator
// ---------------------------------------------------------------------------

/// Coordinates corruption detection, repair attempts, and service state transitions.
///
/// Thread-safe: the orchestrator can be queried from any thread (e.g. health checks)
/// while repairs run on a background worker.
///
/// Uses `std::sync` primitives (not asupersync) because state reads must be
/// synchronous on the query hot path, consistent with [`GenerationController`](crate::activation::GenerationController).
pub struct RepairOrchestrator {
    /// Current service state.
    state: RwLock<ServiceState>,
    /// Log of all corruption events (append-only within a generation lifecycle).
    corruption_log: Mutex<Vec<CorruptionEvent>>,
    /// History of all repair attempts.
    repair_history: Mutex<Vec<RepairAttempt>>,
    /// Policy governing state transitions.
    policy: CorruptionPolicy,
}

impl RepairOrchestrator {
    /// Create a new orchestrator with the given corruption policy.
    #[must_use]
    pub const fn new(policy: CorruptionPolicy) -> Self {
        Self {
            state: RwLock::new(ServiceState::Healthy),
            corruption_log: Mutex::new(Vec::new()),
            repair_history: Mutex::new(Vec::new()),
            policy,
        }
    }

    /// Current service state.
    #[must_use]
    pub fn state(&self) -> ServiceState {
        self.state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Number of recorded corruption events.
    #[must_use]
    pub fn corruption_count(&self) -> usize {
        self.corruption_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Number of recorded repair attempts.
    #[must_use]
    pub fn repair_attempt_count(&self) -> usize {
        self.repair_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Snapshot of all corruption events.
    #[must_use]
    pub fn corruption_events(&self) -> Vec<CorruptionEvent> {
        self.corruption_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Snapshot of all repair attempts.
    #[must_use]
    pub fn repair_attempts(&self) -> Vec<RepairAttempt> {
        self.repair_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Count of unique artifacts that are currently unrepaired (corruption detected,
    /// no successful repair).
    #[must_use]
    pub fn unrepaired_artifact_count(&self) -> usize {
        let log = self
            .corruption_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let history = self
            .repair_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        unrepaired_artifacts(&log, &history).len()
    }

    /// Report a newly detected corruption event.
    ///
    /// The orchestrator logs the event and evaluates whether the corruption
    /// threshold has been exceeded (transitioning to degraded mode if so).
    pub fn report_corruption(&self, event: CorruptionEvent) {
        let entered_at = event.detected_at;
        {
            let mut log = self
                .corruption_log
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            log.push(event);
        }

        // Check if we need to transition to degraded.
        let unrepaired = self.unrepaired_artifact_count();
        if unrepaired >= self.policy.suspension_threshold {
            let mut state = self
                .state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !state.is_suspended() {
                *state = ServiceState::Suspended {
                    entered_at,
                    reason: format!(
                        "{unrepaired} unrepaired artifacts exceed suspension threshold ({})",
                        self.policy.suspension_threshold
                    ),
                };
            }
        } else if unrepaired >= self.policy.max_corrupted_artifacts {
            let mut state = self
                .state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.is_healthy() {
                *state = ServiceState::Degraded {
                    entered_at,
                    reason: DegradedReason::ExcessiveCorruptionRate {
                        event_count: unrepaired,
                        threshold: self.policy.max_corrupted_artifacts,
                    },
                };
            }
        }
    }

    /// Attempt to repair all corrupted artifacts in the current generation.
    ///
    /// For each corrupted artifact that has a matching repair descriptor in the
    /// manifest, delegates to the `provider` for symbol reconstruction. Tracks
    /// outcomes and updates service state accordingly.
    ///
    /// Returns the list of repair attempts made during this cycle.
    pub fn run_repair_cycle(
        &self,
        manifest: &GenerationManifest,
        provider: &dyn RepairProvider,
        now_millis: u64,
    ) -> Vec<RepairAttempt> {
        let corrupted_paths = self.corrupted_artifact_paths();
        let mut cycle_attempts = Vec::new();

        for path in &corrupted_paths {
            // Check retry budget.
            if self.attempt_count_since_last_success(path)
                >= self.policy.max_repair_attempts_per_artifact
            {
                continue;
            }

            // Find matching repair descriptor.
            let Some(descriptor) = manifest
                .repair_descriptors
                .iter()
                .find(|rd| rd.protected_artifact == *path)
            else {
                // No sidecar available for this artifact — record as failed.
                let attempt = RepairAttempt {
                    artifact_path: path.clone(),
                    started_at: now_millis,
                    completed_at: now_millis,
                    outcome: RepairOutcome::SidecarMissing,
                };
                cycle_attempts.push(attempt.clone());
                self.record_attempt(attempt);
                continue;
            };

            // Delegate repair to provider.
            let attempt = provider.attempt_repair(path, descriptor, now_millis);
            cycle_attempts.push(attempt.clone());
            self.record_attempt(attempt);
        }

        // Re-evaluate service state after repairs.
        self.reevaluate_state(now_millis);

        cycle_attempts
    }

    /// Manually transition to degraded mode with an explicit reason.
    pub fn enter_degraded(&self, reason: DegradedReason, now_millis: u64) {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = ServiceState::Degraded {
            entered_at: now_millis,
            reason,
        };
    }

    /// Manually transition to suspended mode (blocks new activations).
    pub fn suspend(&self, reason: String, now_millis: u64) {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = ServiceState::Suspended {
            entered_at: now_millis,
            reason,
        };
    }

    /// Attempt to recover from degraded or suspended state.
    ///
    /// Recovery succeeds only if:
    /// - All corrupted artifacts have been successfully repaired, AND
    /// - For suspended state: the cooldown period has elapsed.
    ///
    /// Returns `true` if recovery succeeded.
    pub fn try_recover(&self, now_millis: u64) -> bool {
        let unrepaired = self.unrepaired_artifact_count();
        if unrepaired > 0 {
            return false;
        }

        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        match &*state {
            ServiceState::Healthy => true,
            ServiceState::Degraded { .. } => {
                *state = ServiceState::Healthy;
                true
            }
            ServiceState::Suspended { entered_at, .. } => {
                let elapsed = now_millis.saturating_sub(*entered_at);
                if elapsed >= self.policy.cooldown_after_suspension_ms {
                    *state = ServiceState::Healthy;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Reset the orchestrator, clearing all corruption and repair history.
    ///
    /// Typically called after a successful generation activation to start
    /// tracking fresh for the new generation.
    pub fn reset(&self) {
        {
            let mut state = self
                .state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *state = ServiceState::Healthy;
        }
        {
            let mut log = self
                .corruption_log
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            log.clear();
        }
        {
            let mut history = self
                .repair_history
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            history.clear();
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Unique artifact paths that have corruption events but no successful repair.
    fn corrupted_artifact_paths(&self) -> Vec<String> {
        let log = self
            .corruption_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let history = self
            .repair_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let unrepaired = unrepaired_artifacts(&log, &history);
        let mut paths: Vec<String> = unrepaired.into_iter().collect();
        paths.sort();
        paths
    }

    /// How many repair attempts have occurred for this artifact since its most
    /// recent successful repair.
    fn attempt_count_since_last_success(&self, artifact_path: &str) -> u32 {
        let count = {
            let history = self
                .repair_history
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let latest_success = history
                .iter()
                .filter(|attempt| {
                    attempt.artifact_path == artifact_path && attempt.outcome.is_success()
                })
                .map(|attempt| attempt.completed_at)
                .max()
                .unwrap_or(0);
            history
                .iter()
                .filter(|attempt| {
                    attempt.artifact_path == artifact_path && attempt.started_at > latest_success
                })
                .count()
        };
        u32::try_from(count).unwrap_or(u32::MAX)
    }

    /// Record a repair attempt in the history.
    fn record_attempt(&self, attempt: RepairAttempt) {
        let mut history = self
            .repair_history
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        history.push(attempt);
    }

    /// Re-evaluate service state based on current corruption and repair state.
    fn reevaluate_state(&self, now_millis: u64) {
        let unrepaired = self.unrepaired_artifact_count();

        let mut state = self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if unrepaired == 0 {
            // All repaired — check if we can recover from suspended.
            match &*state {
                ServiceState::Healthy => {}
                ServiceState::Degraded { .. } => {
                    *state = ServiceState::Healthy;
                }
                ServiceState::Suspended { entered_at, .. } => {
                    let elapsed = now_millis.saturating_sub(*entered_at);
                    if elapsed >= self.policy.cooldown_after_suspension_ms {
                        *state = ServiceState::Healthy;
                    }
                }
            }
        } else if unrepaired >= self.policy.suspension_threshold {
            if !state.is_suspended() {
                *state = ServiceState::Suspended {
                    entered_at: now_millis,
                    reason: format!(
                        "{unrepaired} unrepaired artifacts exceed suspension threshold ({})",
                        self.policy.suspension_threshold
                    ),
                };
            }
        } else if unrepaired >= self.policy.max_corrupted_artifacts && state.is_healthy() {
            *state = ServiceState::Degraded {
                entered_at: now_millis,
                reason: DegradedReason::UnrepairableCorruption {
                    failed_artifacts: self.corrupted_artifact_paths_inner(&state),
                },
            };
        }
    }

    /// Helper to get corrupted paths without re-locking `corruption_log` (called
    /// while state write lock is held, but `corruption_log` is a separate `Mutex`).
    fn corrupted_artifact_paths_inner(&self, _state: &ServiceState) -> Vec<String> {
        self.corrupted_artifact_paths()
    }
}

fn unrepaired_artifacts(
    corruption_log: &[CorruptionEvent],
    repair_history: &[RepairAttempt],
) -> HashSet<String> {
    let mut latest_corruption_by_path: HashMap<String, u64> = HashMap::new();
    for event in corruption_log {
        latest_corruption_by_path
            .entry(event.artifact_path.clone())
            .and_modify(|latest| *latest = (*latest).max(event.detected_at))
            .or_insert(event.detected_at);
    }

    let mut latest_success_by_path: HashMap<String, u64> = HashMap::new();
    for attempt in repair_history {
        if attempt.outcome.is_success() {
            latest_success_by_path
                .entry(attempt.artifact_path.clone())
                .and_modify(|latest| *latest = (*latest).max(attempt.completed_at))
                .or_insert(attempt.completed_at);
        }
    }

    latest_corruption_by_path
        .into_iter()
        .filter_map(|(path, detected_at)| {
            let latest_success = latest_success_by_path.get(&path).copied().unwrap_or(0);
            (latest_success < detected_at).then_some(path)
        })
        .collect()
}

impl Default for RepairOrchestrator {
    fn default() -> Self {
        Self::new(CorruptionPolicy::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::*;
    use std::collections::BTreeMap;

    fn sample_manifest() -> GenerationManifest {
        let mut embedders = BTreeMap::new();
        embedders.insert(
            "fast".into(),
            EmbedderRevision {
                model_name: "potion-128M".into(),
                weights_hash: "abcdef1234567890".into(),
                dimension: 256,
                quantization: QuantizationFormat::F16,
            },
        );
        let mut manifest = GenerationManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generation_id: "gen-001".into(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low: 1, high: 100 },
            build_started_at: 1_700_000_000_000,
            build_completed_at: 1_700_000_060_000,
            embedders,
            vector_artifacts: vec![
                VectorArtifact {
                    path: "vectors/shard_0.fsvi".into(),
                    size_bytes: 1024,
                    checksum: "deadbeef".into(),
                    vector_count: 50,
                    dimension: 256,
                    embedder_tier: EmbedderTierTag::Fast,
                },
                VectorArtifact {
                    path: "vectors/shard_1.fsvi".into(),
                    size_bytes: 1024,
                    checksum: "beefdead".into(),
                    vector_count: 50,
                    dimension: 256,
                    embedder_tier: EmbedderTierTag::Fast,
                },
            ],
            lexical_artifacts: vec![LexicalArtifact {
                path: "lexical/segment_0".into(),
                size_bytes: 2048,
                checksum: "cafebabe".into(),
                document_count: 100,
            }],
            repair_descriptors: vec![
                RepairDescriptor {
                    protected_artifact: "vectors/shard_0.fsvi".into(),
                    sidecar_path: "vectors/shard_0.fsvi.fec".into(),
                    source_symbols: 64,
                    repair_symbols: 13,
                    overhead_ratio: 0.2,
                },
                RepairDescriptor {
                    protected_artifact: "vectors/shard_1.fsvi".into(),
                    sidecar_path: "vectors/shard_1.fsvi.fec".into(),
                    source_symbols: 64,
                    repair_symbols: 13,
                    overhead_ratio: 0.2,
                },
            ],
            activation_invariants: vec![],
            total_documents: 100,
            metadata: BTreeMap::new(),
        };
        manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
        manifest
    }

    fn corruption_event(path: &str, at: u64) -> CorruptionEvent {
        CorruptionEvent {
            artifact_path: path.into(),
            detection_method: DetectionMethod::PeriodicScan,
            detected_at: at,
            detail: format!("checksum mismatch for {path}"),
        }
    }

    /// Mock provider that always succeeds.
    struct AlwaysSucceedProvider;

    impl RepairProvider for AlwaysSucceedProvider {
        fn attempt_repair(
            &self,
            artifact_path: &str,
            _descriptor: &RepairDescriptor,
            now_millis: u64,
        ) -> RepairAttempt {
            RepairAttempt {
                artifact_path: artifact_path.into(),
                started_at: now_millis,
                completed_at: now_millis + 10,
                outcome: RepairOutcome::Success { symbols_used: 5 },
            }
        }

        fn verify_after_repair(&self, _artifact_path: &str, _expected_checksum: &str) -> bool {
            true
        }
    }

    /// Mock provider that always fails.
    struct AlwaysFailProvider;

    impl RepairProvider for AlwaysFailProvider {
        fn attempt_repair(
            &self,
            artifact_path: &str,
            _descriptor: &RepairDescriptor,
            now_millis: u64,
        ) -> RepairAttempt {
            RepairAttempt {
                artifact_path: artifact_path.into(),
                started_at: now_millis,
                completed_at: now_millis + 10,
                outcome: RepairOutcome::Failed {
                    reason: "mock failure".into(),
                },
            }
        }

        fn verify_after_repair(&self, _artifact_path: &str, _expected_checksum: &str) -> bool {
            false
        }
    }

    #[test]
    fn starts_healthy() {
        let orch = RepairOrchestrator::default();
        assert!(orch.state().is_healthy());
        assert!(!orch.state().is_suspended());
        assert_eq!(orch.corruption_count(), 0);
        assert_eq!(orch.repair_attempt_count(), 0);
        assert_eq!(orch.unrepaired_artifact_count(), 0);
    }

    #[test]
    fn report_corruption_logs_event() {
        let orch = RepairOrchestrator::default();
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));

        assert_eq!(orch.corruption_count(), 1);
        assert_eq!(orch.unrepaired_artifact_count(), 1);

        let events = orch.corruption_events();
        assert_eq!(events[0].artifact_path, "vectors/shard_0.fsvi");
        assert_eq!(events[0].detection_method, DetectionMethod::PeriodicScan);
    }

    #[test]
    fn single_corruption_stays_healthy() {
        let policy = CorruptionPolicy {
            max_corrupted_artifacts: 3,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));

        // 1 < threshold of 3 — should stay healthy.
        assert!(orch.state().is_healthy());
    }

    #[test]
    fn exceeding_corruption_threshold_enters_degraded() {
        let policy = CorruptionPolicy {
            max_corrupted_artifacts: 2,
            suspension_threshold: 5,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);

        orch.report_corruption(corruption_event("a.fsvi", 1000));
        assert!(orch.state().is_healthy());

        orch.report_corruption(corruption_event("b.fsvi", 1001));
        assert!(!orch.state().is_healthy());
        assert!(matches!(orch.state(), ServiceState::Degraded { .. }));
    }

    #[test]
    fn exceeding_suspension_threshold_enters_suspended() {
        let policy = CorruptionPolicy {
            max_corrupted_artifacts: 2,
            suspension_threshold: 3,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);

        orch.report_corruption(corruption_event("a.fsvi", 1000));
        orch.report_corruption(corruption_event("b.fsvi", 1001));
        orch.report_corruption(corruption_event("c.fsvi", 1002));

        assert!(orch.state().is_suspended());
    }

    #[test]
    fn successful_repair_reduces_unrepaired_count() {
        let orch = RepairOrchestrator::default();
        let manifest = sample_manifest();

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        assert_eq!(orch.unrepaired_artifact_count(), 1);

        let provider = AlwaysSucceedProvider;
        let attempts = orch.run_repair_cycle(&manifest, &provider, 2000);

        assert_eq!(attempts.len(), 1);
        assert!(attempts[0].outcome.is_success());
        assert_eq!(orch.unrepaired_artifact_count(), 0);
    }

    #[test]
    fn successful_repair_restores_healthy() {
        let policy = CorruptionPolicy {
            max_corrupted_artifacts: 1,
            suspension_threshold: 5,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);
        let manifest = sample_manifest();

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        assert!(matches!(orch.state(), ServiceState::Degraded { .. }));

        let provider = AlwaysSucceedProvider;
        orch.run_repair_cycle(&manifest, &provider, 2000);

        assert!(orch.state().is_healthy());
    }

    #[test]
    fn failed_repair_keeps_degraded() {
        let policy = CorruptionPolicy {
            max_corrupted_artifacts: 1,
            max_repair_attempts_per_artifact: 3,
            suspension_threshold: 5,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);
        let manifest = sample_manifest();

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));

        let provider = AlwaysFailProvider;
        orch.run_repair_cycle(&manifest, &provider, 2000);

        assert!(!orch.state().is_healthy());
        assert_eq!(orch.unrepaired_artifact_count(), 1);
    }

    #[test]
    fn retry_budget_exhausted() {
        let policy = CorruptionPolicy {
            max_repair_attempts_per_artifact: 2,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);
        let manifest = sample_manifest();
        let provider = AlwaysFailProvider;

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));

        // First two attempts should proceed.
        let attempts1 = orch.run_repair_cycle(&manifest, &provider, 2000);
        assert_eq!(attempts1.len(), 1);
        let attempts2 = orch.run_repair_cycle(&manifest, &provider, 3000);
        assert_eq!(attempts2.len(), 1);

        // Third attempt should be skipped (budget exhausted).
        let attempts3 = orch.run_repair_cycle(&manifest, &provider, 4000);
        assert_eq!(attempts3.len(), 0);

        assert_eq!(orch.repair_attempt_count(), 2);
    }

    #[test]
    fn retry_budget_resets_after_successful_repair() {
        let policy = CorruptionPolicy {
            max_repair_attempts_per_artifact: 2,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);
        let manifest = sample_manifest();

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        let fail_provider = AlwaysFailProvider;
        let success_provider = AlwaysSucceedProvider;

        // First corruption: fail once, then succeed.
        assert_eq!(
            orch.run_repair_cycle(&manifest, &fail_provider, 2000).len(),
            1
        );
        assert_eq!(
            orch.run_repair_cycle(&manifest, &success_provider, 3000)
                .len(),
            1
        );
        assert_eq!(orch.unrepaired_artifact_count(), 0);

        // New corruption on same artifact should start with a fresh retry budget.
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 4000));
        assert_eq!(
            orch.run_repair_cycle(&manifest, &fail_provider, 5000).len(),
            1
        );
        assert_eq!(
            orch.run_repair_cycle(&manifest, &fail_provider, 6000).len(),
            1
        );
        assert_eq!(
            orch.run_repair_cycle(&manifest, &fail_provider, 7000).len(),
            0
        );
    }

    #[test]
    fn no_repair_descriptor_records_sidecar_missing() {
        let orch = RepairOrchestrator::default();
        let mut manifest = sample_manifest();
        manifest.repair_descriptors.clear();
        let provider = AlwaysSucceedProvider;

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        let attempts = orch.run_repair_cycle(&manifest, &provider, 2000);

        assert_eq!(attempts.len(), 1);
        assert!(matches!(attempts[0].outcome, RepairOutcome::SidecarMissing));
    }

    #[test]
    fn manual_degraded_and_recovery() {
        let orch = RepairOrchestrator::default();

        orch.enter_degraded(
            DegradedReason::ActivationFailure {
                detail: "manifest hash mismatch".into(),
            },
            1000,
        );
        assert!(matches!(orch.state(), ServiceState::Degraded { .. }));

        // Recovery succeeds since there are no unrepaired artifacts.
        assert!(orch.try_recover(2000));
        assert!(orch.state().is_healthy());
    }

    #[test]
    fn recovery_blocked_by_unrepaired_artifacts() {
        let orch = RepairOrchestrator::default();

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        orch.enter_degraded(
            DegradedReason::UnrepairableCorruption {
                failed_artifacts: vec!["vectors/shard_0.fsvi".into()],
            },
            1000,
        );

        assert!(!orch.try_recover(2000));
        assert!(!orch.state().is_healthy());
    }

    #[test]
    fn suspension_cooldown_enforced() {
        let policy = CorruptionPolicy {
            cooldown_after_suspension_ms: 10_000,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);

        orch.suspend("test suspension".into(), 1000);

        // Too early — cooldown hasn't elapsed.
        assert!(!orch.try_recover(5000));

        // After cooldown — should recover.
        assert!(orch.try_recover(11_001));
        assert!(orch.state().is_healthy());
    }

    #[test]
    fn reset_clears_everything() {
        let policy = CorruptionPolicy {
            max_corrupted_artifacts: 1,
            suspension_threshold: 5,
            ..Default::default()
        };
        let orch = RepairOrchestrator::new(policy);

        orch.report_corruption(corruption_event("a.fsvi", 1000));
        assert!(!orch.state().is_healthy());

        orch.reset();

        assert!(orch.state().is_healthy());
        assert_eq!(orch.corruption_count(), 0);
        assert_eq!(orch.repair_attempt_count(), 0);
        assert_eq!(orch.unrepaired_artifact_count(), 0);
    }

    #[test]
    fn multiple_corruptions_same_artifact_counted_once() {
        let orch = RepairOrchestrator::default();

        // Same artifact corrupted twice — still counts as 1 unrepaired.
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1001));

        assert_eq!(orch.corruption_count(), 2);
        assert_eq!(orch.unrepaired_artifact_count(), 1);
    }

    #[test]
    fn repair_cycle_handles_multiple_corrupted_artifacts() {
        let orch = RepairOrchestrator::default();
        let manifest = sample_manifest();
        let provider = AlwaysSucceedProvider;

        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        orch.report_corruption(corruption_event("vectors/shard_1.fsvi", 1001));

        let attempts = orch.run_repair_cycle(&manifest, &provider, 2000);
        assert_eq!(attempts.len(), 2);
        assert!(attempts.iter().all(|a| a.outcome.is_success()));
        assert_eq!(orch.unrepaired_artifact_count(), 0);
    }

    #[test]
    fn detection_methods_serialize() {
        for method in &[
            DetectionMethod::PeriodicScan,
            DetectionMethod::ReadTimeVerification,
            DetectionMethod::ChecksumMismatch,
            DetectionMethod::MissingSidecar,
        ] {
            let json = serde_json::to_string(method).expect("serialize");
            let back: DetectionMethod = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(method, &back);
        }
    }

    #[test]
    fn repair_outcome_serialize() {
        let outcomes = vec![
            RepairOutcome::Success { symbols_used: 5 },
            RepairOutcome::InsufficientSymbols {
                available: 3,
                required: 10,
            },
            RepairOutcome::SidecarMissing,
            RepairOutcome::SidecarCorrupted,
            RepairOutcome::Failed {
                reason: "disk full".into(),
            },
        ];
        for outcome in &outcomes {
            let json = serde_json::to_string(outcome).expect("serialize");
            let back: RepairOutcome = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(outcome, &back);
        }
    }

    #[test]
    fn service_state_serialize() {
        let states = vec![
            ServiceState::Healthy,
            ServiceState::Degraded {
                entered_at: 1000,
                reason: DegradedReason::ActivationFailure {
                    detail: "test".into(),
                },
            },
            ServiceState::Suspended {
                entered_at: 2000,
                reason: "manual".into(),
            },
        ];
        for state in &states {
            let json = serde_json::to_string(state).expect("serialize");
            let back: ServiceState = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(state, &back);
        }
    }

    #[test]
    fn corruption_policy_default_is_reasonable() {
        let policy = CorruptionPolicy::default();
        assert!(policy.max_corrupted_artifacts > 0);
        assert!(policy.max_repair_attempts_per_artifact > 0);
        assert!(policy.suspension_threshold > policy.max_corrupted_artifacts);
        assert!(policy.cooldown_after_suspension_ms > 0);
    }

    #[test]
    fn degraded_reason_variants_serialize() {
        let reasons = vec![
            DegradedReason::UnrepairableCorruption {
                failed_artifacts: vec!["a.fsvi".into(), "b.fsvi".into()],
            },
            DegradedReason::ExcessiveCorruptionRate {
                event_count: 5,
                threshold: 3,
            },
            DegradedReason::ActivationFailure {
                detail: "hash mismatch".into(),
            },
        ];
        for reason in &reasons {
            let json = serde_json::to_string(reason).expect("serialize");
            let back: DegradedReason = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(reason, &back);
        }
    }

    #[test]
    fn repair_after_successful_repair_clears_corruption() {
        let orch = RepairOrchestrator::default();
        let manifest = sample_manifest();
        let provider = AlwaysSucceedProvider;

        // Corrupt, repair, then corrupt again.
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 1000));
        orch.run_repair_cycle(&manifest, &provider, 2000);
        assert_eq!(orch.unrepaired_artifact_count(), 0);

        // New corruption on same artifact after repair.
        orch.report_corruption(corruption_event("vectors/shard_0.fsvi", 3000));
        assert_eq!(orch.unrepaired_artifact_count(), 1);
    }
}
