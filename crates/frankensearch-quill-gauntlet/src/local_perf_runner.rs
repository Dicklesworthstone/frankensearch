//! Typed registered-host performance runner.
//!
//! This producer owns one canonical host-global lease across benchmark
//! compilation, start/end probes, and the measured child. It emits the required
//! bound evidence only after the child exits successfully and every exact
//! artifact re-verifies. The sole process receipt is atomically published last
//! and binds that evidence's exact bytes, so H4 completion requires the pair.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Seek, Write};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;
#[cfg(test)]
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;

use rustix::fs::{
    FileType, FlockOperation, Mode, OFlags, RenameFlags, flock, fstat, mkdirat, open, openat,
    renameat_with,
};
use rustix::io::{FdFlags, fcntl_getfd, fcntl_setfd};
use rustix::process::geteuid;
#[cfg(target_os = "linux")]
use rustix::process::{Pid, Signal, kill_process_group};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(test)]
use crate::machine_class_registry::DefaultFlipDisposition;
use crate::machine_class_registry::{
    ExecutionCapacitySemantics, ExecutionProfileId, HardwareClassId,
    LOCAL_PERF_PRODUCER_CONTRACT_VERSION, MACHINE_CLASS_REGISTRY_SHA256,
    MachineClassAdmissionContext, MachineClassError, MachineClassRegistry,
    MachineProfileAvailability, MachineProfileKey, RUNNER_RECEIPT_SCHEMA_VERSION,
    RunnerArtifactManifest, RunnerBuild, RunnerCompletion, RunnerDurability, RunnerExecution,
    RunnerExecutionRequest, RunnerExecutionSnapshot, RunnerHardware, RunnerProducer, RunnerReceipt,
    VerifiedRunnerIdentity, seal_runner_receipt, sha256_hex,
};
use crate::{
    EvidenceArtifactError, PerfApplicabilityPlan, PerfApplicabilityPlanBinding,
    PerfEvidenceArtifact, PerfGate, PerfGateArtifact, PerfMatrixSpec, command_sha256_from_argv,
};
#[cfg(test)]
use crate::{PerfCellApplicability, PerfCellApplicabilityReason};

const PRODUCER_CONTRACT_SCHEMA_VERSION: &str =
    "frankensearch.quill-local-perf-producer-contract.v1";
/// Strict wire schema for one local-run process-attempt receipt.
pub const LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION: &str = "frankensearch.perf-runner-attempt.v7";
/// Strict schema for the diagnostic inventory retained before runner completion.
pub const PERF_RUN_PRECOMMIT_SCHEMA_VERSION: &str = "frankensearch.perf-run-precommit.v5";
const MAX_IDENTITY_COMPONENT_BYTES: usize = 96;
const MAX_OUTPUT_COMPONENT_BYTES: usize = 128;
const MIN_MEASUREMENT_RUNS: usize = 10;
const MAX_MEASUREMENT_RUNS: usize = 100;
const MEASUREMENT_WARMUP_ROUNDS: &str = "1";
const MEASUREMENT_BOOTSTRAP_SEED: &str = "5860671082138523204";
const WAIT_RECOVERY_POLL_ATTEMPTS: usize = 100;
const WAIT_RECOVERY_POLL_INTERVAL: Duration = Duration::from_millis(10);
const EMBEDDED_PRODUCER_CONTRACT_VERSION: &str = env!("QUILL_PERF_PRODUCER_CONTRACT_VERSION");
const EMBEDDED_PRODUCER_GIT_REVISION: &str = env!("QUILL_PERF_PRODUCER_GIT_REVISION");
const EMBEDDED_PRODUCER_GIT_DIRTY: &str = env!("QUILL_PERF_PRODUCER_GIT_DIRTY");
const EMBEDDED_PRODUCER_CARGO_LOCK_SHA256: &str = env!("QUILL_PERF_PRODUCER_CARGO_LOCK_SHA256");

/// Complete registered-host invocation owned by the typed producer.
#[derive(Debug, Clone)]
pub struct LocalPerfRunConfig {
    /// Gate selected for this invocation.
    pub gate: PerfGate,
    /// Canonical registered hardware/profile identity.
    pub profile: MachineProfileKey,
    /// Unique pass identity.
    pub run_id: String,
    /// Window shared by a candidate and its immediate rerun.
    pub run_window: String,
    /// Predeclared measured block count; never inherited from ambient state.
    pub measurement_runs: usize,
    /// Unique not-yet-created output directory. Existing paths are rejected.
    pub output_dir: PathBuf,
}

/// One exact canonical fixture selected for an isolated partial-shard run.
///
/// Construction validates only the closed text boundary. The selected-run
/// entry point resolves this value against the frozen gate applicability plan,
/// rejects unknown or non-applicable fixtures, and derives the exact ordered
/// cell subset before any benchmark process starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalPerfRunSelection {
    fixture: String,
}

// The dependent H4 assembler consumes this public contract after the isolated
// H2 slice lands.
#[allow(dead_code)]
impl LocalPerfRunSelection {
    /// Construct an exact canonical fixture selector.
    ///
    /// # Errors
    ///
    /// Rejects empty, padded, non-ASCII, control-bearing, or overlong text.
    pub fn for_fixture(fixture: impl Into<String>) -> Result<Self, LocalPerfRunError> {
        let fixture = fixture.into();
        validate_fixture_selector_syntax(&fixture)?;
        Ok(Self { fixture })
    }

    /// Exact fixture string injected into the controlled child environment.
    #[must_use]
    pub fn fixture(&self) -> &str {
        &self.fixture
    }
}

/// Files emitted after a successful self-verifying finalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalPerfRunOutput {
    /// Exact child log.
    pub run_log: PathBuf,
    /// Exact canonical compact artifact manifest.
    pub artifact_manifest: PathBuf,
    /// Canonical controlled-environment policy preimage bound by the receipt.
    pub environment_policy: PathBuf,
    /// Exact strict completion receipt.
    pub runner_receipt: PathBuf,
    /// Exact canonical process-attempt receipt. A completed receipt binds the
    /// SHA-256 of [`Self::bound_evidence`], while every failed attempt returns
    /// this same schema through [`LocalPerfRunError::AttemptFailed`].
    pub attempt_receipt: PathBuf,
    /// Exact raw threshold artifact named by the runner artifact manifest.
    pub threshold_artifact: PathBuf,
    /// Exact pre-binding evidence artifact named by the runner manifest.
    pub prebinding_evidence: PathBuf,
    /// Exact admitted, runner-bound evidence artifact persisted by the
    /// finalizer. This object independently re-admits its embedded runner
    /// receipt and reconstructs the pre-binding evidence bytes on load.
    pub bound_evidence: PathBuf,
    /// Pre-commit inventory written before the receipt commit boundary.
    pub precommit_inventory: PathBuf,
}

/// Typed local-run failure. No promotion receipt is emitted unless all
/// verification succeeds.
///
/// A failed child may leave the separately sealed, non-promotable attempt
/// receipt; a failed final write may otherwise leave diagnostics, the manifest,
/// and the pre-commit inventory.
#[derive(Debug, Error)]
pub enum LocalPerfRunError {
    /// Filesystem or process I/O failed.
    #[error("local performance runner I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// JSON encoding or decoding failed.
    #[error("local performance runner JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    /// Registry admission failed.
    #[error("local performance runner receipt rejected: {0}")]
    Machine(#[from] MachineClassError),
    /// Current evidence did not verify.
    #[error("local performance runner evidence rejected: {0}")]
    Evidence(#[from] EvidenceArtifactError),
    /// A strict producer precondition failed.
    #[error("local performance runner rejected invocation: {0}")]
    Invalid(String),
    /// A measured child attempt reached a durable typed terminal receipt.
    #[error(
        "local performance runner attempt ended as {outcome:?}; sealed receipt preserved at {}",
        receipt_path.display()
    )]
    AttemptFailed {
        /// Exact canonical attempt-receipt path.
        receipt_path: PathBuf,
        /// Typed terminal outcome preserved in the receipt.
        outcome: LocalPerfAttemptOutcome,
    },
    /// The final strict attempt bytes were built and verified, but the atomic
    /// no-replace publication itself could not establish the durable commit
    /// boundary. No successful API result is returned.
    #[error(
        "local performance runner could not durably publish {outcome:?} receipt at {}: {detail}",
        receipt_path.display()
    )]
    AttemptCommitFailed {
        /// Intended final receipt path; it may be absent or nondurable.
        receipt_path: PathBuf,
        /// Typed producer-rejection boundary.
        outcome: LocalPerfAttemptOutcome,
        /// Bounded non-secret publication diagnostic.
        detail: String,
    },
    /// `wait` failed and the bounded force-kill/reap recovery could not prove
    /// a terminal status. No terminal attempt receipt is emitted while the
    /// child might still mutate its log.
    #[error(
        "local performance runner could not prove child reap after wait error {wait_error_kind:?}: {recovery_error_kind:?}"
    )]
    UnreapedChild {
        /// Error returned by the initial blocking wait.
        wait_error_kind: LocalPerfIoErrorKind,
        /// Error or bounded-deadline classification from kill/reap recovery.
        recovery_error_kind: LocalPerfIoErrorKind,
    },
}

#[derive(Debug)]
struct CapturedBuild {
    receipt: RunnerBuild,
    revision: String,
    command: Vec<OsString>,
    measurement_environment: BTreeMap<OsString, OsString>,
    environment_policy_bytes: Vec<u8>,
    tool_identities: Vec<ResolvedTool>,
    cargo_config_candidates: Vec<PathBuf>,
    executable_path: PathBuf,
    executable: File,
    executable_identity: FileIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CleanSourceSnapshot {
    revision: String,
    cargo_lock_sha256: String,
}

#[derive(Debug)]
struct ExecutingProducer {
    receipt: RunnerProducer,
    executable: File,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalPerfProducerContract {
    schema_version: &'static str,
    machine_class_registry_sha256: &'static str,
    producer: RunnerProducer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LeaseFileIdentity {
    device: String,
    inode: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
}

#[derive(Debug)]
struct PinnedDirectory {
    path: PathBuf,
    handle: File,
    identity: FileIdentity,
}

#[derive(Debug)]
struct ExternalRunPaths {
    output_parent: PinnedDirectory,
    target: PinnedDirectory,
}

#[derive(Debug)]
struct ControlledEnvironments {
    build: BTreeMap<OsString, OsString>,
    measurement: BTreeMap<OsString, OsString>,
    policy_sha256: String,
    policy_bytes: Vec<u8>,
    tools: Vec<ResolvedTool>,
    cargo_config_candidates: Vec<PathBuf>,
}

#[derive(Debug)]
struct RunDirectories {
    run: PinnedDirectory,
    artifacts: PinnedDirectory,
}

#[derive(Debug, Clone)]
struct PlatformCapture {
    hardware: RunnerHardware,
    request: RunnerExecutionRequest,
    snapshot: RunnerExecutionSnapshot,
}

#[derive(Debug, Clone)]
struct RunProfileContract {
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    applicability_plan: PerfApplicabilityPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedRunSelection {
    fixture: Option<String>,
    selected_cell_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrecommitInventory {
    schema_version: String,
    gate: String,
    profile: MachineProfileKey,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    applicability_plan: PerfApplicabilityPlanBinding,
    fixture_selector: Option<String>,
    selected_cell_ids: Vec<String>,
    run_id: String,
    run_window: String,
    run_log_sha256: String,
    threshold_artifact_sha256: String,
    evidence_artifact_sha256: String,
    artifact_manifest_sha256: String,
    runner_receipt_sha256: String,
    producer_identity_sha256: String,
    bound_evidence_preview_sha256: String,
    environment_policy_sha256: String,
}

/// Terminal outcome of one runner process attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LocalPerfAttemptOutcome {
    /// The child completed, its evidence was independently admitted, and the
    /// exact bound-evidence digest is carried in the enclosing receipt.
    Completed,
    /// The benchmark image could not be spawned.
    SpawnRejected { error_kind: LocalPerfIoErrorKind },
    /// The first `wait` failed, after which the runner forced termination and
    /// observed a bounded `try_wait` reap before sealing this receipt.
    WaitRecoveredByKill { error_kind: LocalPerfIoErrorKind },
    /// The child was reaped after returning a nonzero status.
    ExitedNonzero { code: i64 },
    /// The child was reaped after a terminating signal.
    Signaled { signal: i32 },
    /// Unix returned a terminal status without an exit code or signal.
    UnknownTerminal,
    /// The child returned zero, but its output failed strict post-exit
    /// verification before any promotion receipt was written.
    PostExitRejected { stage: LocalPerfRejectionStage },
}

/// Bounded OS error class retained for a failed spawn without persisting a
/// host-specific or secret-bearing error string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPerfIoErrorKind {
    NotFound,
    PermissionDenied,
    ResourceBusy,
    OutOfMemory,
    Other,
}

/// Post-exit verification boundary that rejected a zero-exit child.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPerfRejectionStage {
    RunLogSync,
    RunLogRead,
    ExitStatusPersistence,
    FinishedTimestamp,
    EndPlatformCapture,
    RootProcessIdentity,
    PostRunIdentity,
    ArtifactRead,
    ArtifactDurability,
    ArtifactVerification,
    ArtifactManifestSerialization,
    RunnerReceiptSerialization,
    RunnerAdmission,
    BoundEvidenceSerialization,
    PrecommitSerialization,
    PrecommitPersistence,
    RunnerReceiptPersistence,
    AttemptReceiptPersistence,
    BoundEvidencePersistence,
    PersistedPairVerification,
}

/// Concrete typed retry predicate derived from the terminal outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LocalPerfRetryPredicate {
    /// No retry is required for a completed, admitted attempt.
    NotRequired,
    RepairSpawn {
        error_kind: LocalPerfIoErrorKind,
    },
    RepairWait {
        error_kind: LocalPerfIoErrorKind,
    },
    DiagnoseNonzeroExit {
        code: i64,
    },
    DiagnoseSignal {
        signal: i32,
    },
    DiagnoseUnknownTerminal,
    RepairRejectedEvidence {
        stage: LocalPerfRejectionStage,
    },
}

/// Outer child-process lifecycle facts directly observed by this runner.
// These are independent receipt facts rather than mutually exclusive states;
// retaining each observation prevents a coarse enum from implying facts the
// runner did not witness.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPerfProcessLifecycle {
    spawn_attempted: bool,
    spawn_succeeded: bool,
    wait_completed: bool,
    child_reaped: bool,
    run_log_synced: bool,
    run_log_captured: bool,
    process_group_recovery: LocalPerfProcessGroupRecovery,
}

/// Exact bounded recovery action after an OS wait failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPerfProcessGroupRecovery {
    /// No wait recovery was needed for this attempt.
    NotRequired,
    /// A revalidated dedicated child group received SIGKILL before reaping.
    SignaledOwnedGroup,
    /// Group identity was unavailable or changed, so only the owned child
    /// handle received the bounded fallback signal.
    DirectChildFallback,
}

/// Process-tree authority available from one local producer attempt.
///
/// The current runner owns and reaps only the direct child it spawned. It
/// cannot infer descendant completion from that fact because descendants may
/// outlive or be reparented after the direct child exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPerfProcessTreeQuiescence {
    DirectChildOnly,
}

/// Root-process authority retained by a local producer attempt.
///
/// A PID on its own is never authority to signal a later process: Linux can
/// reuse it after the original child exits. The only verified form therefore
/// binds that PID to both a dedicated child-led process group and the kernel's
/// `/proc/<pid>/stat` start-time tick captured immediately after spawn. Other
/// platforms and failed captures are retained as an explicit absence, which
/// rejects completed receipt publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum LocalPerfRootProcessIdentity {
    /// `Command::spawn` did not return a child PID.
    NotSpawned,
    /// Linux PID, dedicated process group, and kernel-reported birth tick.
    LinuxProcStartTime {
        pid: u32,
        process_group_id: u32,
        start_time_ticks: u64,
    },
    /// A child was spawned, but no birth identity was safely captured.
    Unverifiable { pid: u32 },
}

impl LocalPerfRootProcessIdentity {
    /// Whether this value binds a PID to a non-reusable process birth identity.
    #[must_use]
    pub const fn has_verified_birth_identity(self) -> bool {
        matches!(self, Self::LinuxProcStartTime { .. })
    }
}

// This public contract is consumed by the dependent H4 assembler slice; keep
// the isolated H2 commit warning-clean before that re-export lands.
#[allow(dead_code)]
impl LocalPerfProcessLifecycle {
    /// Whether `Command::spawn` was invoked.
    #[must_use]
    pub const fn spawn_attempted(self) -> bool {
        self.spawn_attempted
    }

    /// Whether the child image was successfully spawned.
    #[must_use]
    pub const fn spawn_succeeded(self) -> bool {
        self.spawn_succeeded
    }

    /// Whether `wait`, or bounded recovery through `try_wait`, observed a
    /// terminal status.
    #[must_use]
    pub const fn wait_completed(self) -> bool {
        self.wait_completed
    }

    /// Whether the terminal `wait` reaped the child.
    #[must_use]
    pub const fn child_reaped(self) -> bool {
        self.child_reaped
    }

    /// Whether the exact combined run log was synced before receipt sealing.
    #[must_use]
    pub const fn run_log_synced(self) -> bool {
        self.run_log_synced
    }

    /// Whether the exact combined run-log bytes were captured for hashing.
    #[must_use]
    pub const fn run_log_captured(self) -> bool {
        self.run_log_captured
    }

    /// Exact wait-error recovery authority observed by this receipt.
    #[must_use]
    pub const fn process_group_recovery(self) -> LocalPerfProcessGroupRecovery {
        self.process_group_recovery
    }

    /// Return the strongest process-tree conclusion this receipt can prove.
    #[must_use]
    pub const fn process_tree_quiescence(self) -> LocalPerfProcessTreeQuiescence {
        LocalPerfProcessTreeQuiescence::DirectChildOnly
    }

    /// Whether this receipt proves every descendant reached a terminal state.
    #[must_use]
    pub const fn descendant_process_tree_quiescence_is_proven(self) -> bool {
        false
    }
}

/// Why an engine-internal fact is absent from an outer runner attempt.
// The repeated Child prefix is intentional in the serialized public contract:
// H4 must not mistake an outer-runner observation for an engine-internal one.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPerfInternalLifecycleUnavailable {
    ChildDidNotCompleteSuccessfully,
    ChildEvidenceNotAdmitted,
    /// The child evidence was admitted and cryptographically bound, but this
    /// outer process runner did not independently observe engine internals.
    ChildEvidenceAdmittedButNotIndependentlyObserved,
}

/// Explicit engine-internal receipt gaps.
///
/// These fields prevent configured capacity or outer-process completion from
/// being relabeled as actual work, queue activity, worker join, feed drain, or
/// pending-zero evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPerfInternalLifecycleGaps {
    actual_work: LocalPerfInternalLifecycleUnavailable,
    queue: LocalPerfInternalLifecycleUnavailable,
    workers_joined: LocalPerfInternalLifecycleUnavailable,
    feed_drained: LocalPerfInternalLifecycleUnavailable,
    pending_zero: LocalPerfInternalLifecycleUnavailable,
}

// See the H4 integration note on `LocalPerfProcessLifecycle`.
#[allow(dead_code)]
impl LocalPerfInternalLifecycleGaps {
    /// Typed absence of engine-internal actual-work counters.
    #[must_use]
    pub const fn actual_work(self) -> LocalPerfInternalLifecycleUnavailable {
        self.actual_work
    }

    /// Typed absence of engine-internal queue observations.
    #[must_use]
    pub const fn queue(self) -> LocalPerfInternalLifecycleUnavailable {
        self.queue
    }

    /// Typed absence of engine worker-join evidence.
    #[must_use]
    pub const fn workers_joined(self) -> LocalPerfInternalLifecycleUnavailable {
        self.workers_joined
    }

    /// Typed absence of feed-drain evidence.
    #[must_use]
    pub const fn feed_drained(self) -> LocalPerfInternalLifecycleUnavailable {
        self.feed_drained
    }

    /// Typed absence of pending-zero evidence.
    #[must_use]
    pub const fn pending_zero(self) -> LocalPerfInternalLifecycleUnavailable {
        self.pending_zero
    }
}

/// Runner controls unavailable in the current synchronous local producer.
///
/// These name caller-scheduled controls; the bounded force-kill used only to
/// recover from an OS `wait` error is a fail-closed reap mechanism, not a
/// caller-requested cancellation outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalPerfUnsupportedControl {
    Timeout,
    Cancellation,
}

/// Strict, canonical, independently verifiable process-attempt receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPerfAttemptReceipt {
    schema_version: String,
    mode: String,
    gate: String,
    profile: MachineProfileKey,
    applicability_plan: PerfApplicabilityPlanBinding,
    fixture_selector: Option<String>,
    selected_cell_ids: Vec<String>,
    run_id: String,
    run_window: String,
    registry_sha256: String,
    hardware: RunnerHardware,
    execution_request: RunnerExecutionRequest,
    execution_start: RunnerExecutionSnapshot,
    execution_end: Option<RunnerExecutionSnapshot>,
    end_capture_error: Option<String>,
    build: RunnerBuild,
    durability: RunnerDurability,
    post_run_identity_verified: bool,
    post_run_identity_error: Option<String>,
    outcome: LocalPerfAttemptOutcome,
    retry: LocalPerfRetryPredicate,
    process_lifecycle: LocalPerfProcessLifecycle,
    root_process_identity: LocalPerfRootProcessIdentity,
    internal_lifecycle_gaps: LocalPerfInternalLifecycleGaps,
    unsupported_controls: Vec<LocalPerfUnsupportedControl>,
    run_log_sha256: Option<String>,
    bound_evidence_sha256: Option<String>,
    runner_receipt_sha256: Option<String>,
    runner_artifact_manifest_sha256: Option<String>,
    started_at_utc: String,
    finished_at_utc: String,
    finished_timestamp_error: Option<String>,
    seal_sha256: String,
}

// Same-crate H4 consumption lands separately on the protected train.
#[allow(dead_code)]
impl LocalPerfAttemptReceipt {
    /// Parse exact canonical bytes and verify their self-seal, frozen
    /// applicability plan, pre-spawn machine admission, terminal lifecycle,
    /// retry predicate, and typed internal-evidence gaps.
    ///
    /// # Errors
    ///
    /// Rejects duplicate or unknown fields, noncanonical JSON, a stale plan,
    /// malformed provenance, contradictory lifecycle facts, or any tamper.
    pub fn from_verified_slice(contents: &[u8]) -> Result<Self, LocalPerfRunError> {
        let probe =
            crate::machine_class_registry::parse_strict_json(contents).map_err(|error| {
                LocalPerfRunError::Invalid(format!("attempt receipt is not strict JSON: {error}"))
            })?;
        let receipt: Self = serde_json::from_value(probe.clone()).map_err(|error| {
            LocalPerfRunError::Invalid(format!(
                "attempt receipt does not decode as the current schema: {error}"
            ))
        })?;
        if probe != serde_json::to_value(&receipt)?
            || contents != receipt.to_json_bytes()?.as_slice()
        {
            return Err(LocalPerfRunError::Invalid(
                "attempt receipt bytes are not the exact canonical encoding".to_owned(),
            ));
        }
        receipt.verify()?;
        Ok(receipt)
    }

    /// Load and independently verify one exact attempt receipt.
    ///
    /// # Errors
    ///
    /// Returns a typed I/O or receipt-verification error.
    pub fn load_verified(path: &Path) -> Result<Self, LocalPerfRunError> {
        Self::from_verified_slice(&fs::read(path)?)
    }

    /// Canonical compact JSON bytes used for persistence and exact hashing.
    ///
    /// # Errors
    ///
    /// Returns a JSON error only if the typed schema stops being encodable.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, LocalPerfRunError> {
        serde_json::to_vec(self).map_err(LocalPerfRunError::from)
    }

    /// SHA-256 of the exact canonical receipt bytes.
    ///
    /// # Errors
    ///
    /// Returns a JSON error only if the typed schema stops being encodable.
    pub fn exact_sha256(&self) -> Result<String, LocalPerfRunError> {
        Ok(sha256_hex(&self.to_json_bytes()?))
    }

    /// Prove that exact run-log bytes match the receipt.
    ///
    /// # Errors
    ///
    /// Rejects a substituted, truncated, or otherwise different log.
    pub fn verify_run_log(&self, run_log_bytes: &[u8]) -> Result<(), LocalPerfRunError> {
        let expected = self.run_log_sha256.as_deref().ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "attempt did not capture exact run-log bytes for verification".to_owned(),
            )
        })?;
        if sha256_hex(run_log_bytes) != expected {
            return Err(LocalPerfRunError::Invalid(
                "attempt run-log bytes differ from the sealed receipt".to_owned(),
            ));
        }
        Ok(())
    }

    /// Prove that exact completed bound-evidence bytes are the object named by
    /// this process receipt.
    ///
    /// # Errors
    ///
    /// Rejects a failed-attempt receipt, or substituted, truncated, or
    /// otherwise different bound-evidence bytes.
    pub fn verify_bound_evidence(
        &self,
        bound_evidence_bytes: &[u8],
    ) -> Result<(), LocalPerfRunError> {
        let expected = self.bound_evidence_sha256.as_deref().ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "failed attempt receipt cannot bind completed evidence".to_owned(),
            )
        })?;
        if sha256_hex(bound_evidence_bytes) != expected {
            return Err(LocalPerfRunError::Invalid(
                "bound-evidence bytes differ from the sealed process receipt".to_owned(),
            ));
        }
        let evidence = PerfEvidenceArtifact::from_verified_slice(bound_evidence_bytes)?;
        if evidence.gate.label() != self.gate
            || evidence.applicability_plan != self.applicability_plan
            || evidence.provenance.run_id != self.run_id
            || evidence.provenance.run_window != self.run_window
            || evidence_cell_ids(&evidence) != self.selected_cell_ids
        {
            return Err(LocalPerfRunError::Invalid(
                "bound evidence gate, plan, run identity, or cells differ from the sealed process receipt"
                    .to_owned(),
            ));
        }
        let identity = evidence.machine_class.identity().ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "completed bound evidence has no admitted runner identity".to_owned(),
            )
        })?;
        self.verify_completed_runner_identity(identity)?;
        Ok(())
    }

    fn verify_completed_runner_identity(
        &self,
        identity: &VerifiedRunnerIdentity,
    ) -> Result<(), LocalPerfRunError> {
        identity.verify()?;
        let expected_receipt_sha256 = self.runner_receipt_sha256.as_deref().ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "completed attempt omits the exact admitted runner-receipt digest".to_owned(),
            )
        })?;
        let expected_manifest_sha256 =
            self.runner_artifact_manifest_sha256
                .as_deref()
                .ok_or_else(|| {
                    LocalPerfRunError::Invalid(
                        "completed attempt omits the exact runner artifact-manifest digest"
                            .to_owned(),
                    )
                })?;
        let manifest = identity.artifact_manifest().ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "completed admitted runner identity has no artifact-manifest binding".to_owned(),
            )
        })?;
        let execution_end = self.execution_end.as_ref().ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "completed attempt has no terminal execution snapshot".to_owned(),
            )
        })?;
        let run_log_sha256 = self.run_log_sha256.as_deref().ok_or_else(|| {
            LocalPerfRunError::Invalid("completed attempt has no exact run-log digest".to_owned())
        })?;
        let runner_value =
            crate::machine_class_registry::parse_strict_json(identity.receipt_json().as_bytes())?;
        let runner: RunnerReceipt = serde_json::from_value(runner_value)?;
        let expected_completion = RunnerCompletion {
            verified: true,
            exit_status: 0,
            run_log_sha256: run_log_sha256.to_owned(),
            artifact_manifest_sha256: expected_manifest_sha256.to_owned(),
            artifact_digests_verified: true,
            started_at_utc: self.started_at_utc.clone(),
            finished_at_utc: self.finished_at_utc.clone(),
        };
        let expected_context = MachineClassAdmissionContext {
            gate: self.gate.clone(),
            expected_profile: self.profile,
            destination_basename: self.profile.latest_basename(&self.gate)?,
        };
        if identity.receipt_sha256() != expected_receipt_sha256
            || manifest.manifest_sha256() != expected_manifest_sha256
            || identity.admission_context() != &expected_context
            || identity.profile() != self.profile
            || identity.hardware() != &serde_json::to_value(&self.hardware)?
            || identity.execution_request() != &serde_json::to_value(&self.execution_request)?
            || identity.execution_start() != &serde_json::to_value(&self.execution_start)?
            || identity.execution_end() != &serde_json::to_value(execution_end)?
            || identity.build() != &serde_json::to_value(&self.build)?
            || identity.durability() != &serde_json::to_value(&self.durability)?
            || identity.completion() != &serde_json::to_value(&expected_completion)?
            || runner.requested_profile != self.profile
            || runner.derived_profile != self.profile
            || runner.registry_sha256 != self.registry_sha256
            || runner.hardware != self.hardware
            || runner.execution.request != self.execution_request
            || runner.execution.start != self.execution_start
            || runner.execution.end != *execution_end
            || runner.build != self.build
            || runner.durability != self.durability
            || runner.completion != expected_completion
        {
            return Err(LocalPerfRunError::Invalid(
                "completed attempt facts differ from its exact admitted nested runner receipt"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Gate selected by this attempt.
    #[must_use]
    pub fn gate(&self) -> &str {
        &self.gate
    }

    /// Frozen machine/execution profile selected by this attempt.
    #[must_use]
    pub const fn profile(&self) -> MachineProfileKey {
        self.profile
    }

    /// Exact frozen applicability-plan binding.
    #[must_use]
    pub const fn applicability_plan(&self) -> &PerfApplicabilityPlanBinding {
        &self.applicability_plan
    }

    /// Exact typed fixture selector, or `None` for a full-gate invocation.
    #[must_use]
    pub fn fixture_selector(&self) -> Option<&str> {
        self.fixture_selector.as_deref()
    }

    /// Canonically ordered runnable cells selected by the full-gate producer.
    #[must_use]
    pub fn selected_cell_ids(&self) -> &[String] {
        &self.selected_cell_ids
    }

    /// Process-level run identity.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Candidate/rerun window identity.
    #[must_use]
    pub fn run_window(&self) -> &str {
        &self.run_window
    }

    /// Typed terminal outcome.
    #[must_use]
    pub const fn outcome(&self) -> LocalPerfAttemptOutcome {
        self.outcome
    }

    /// Concrete typed retry predicate derived from [`Self::outcome`].
    #[must_use]
    pub const fn retry(&self) -> LocalPerfRetryPredicate {
        self.retry
    }

    /// Directly observed outer child-process lifecycle.
    #[must_use]
    pub const fn process_lifecycle(&self) -> LocalPerfProcessLifecycle {
        self.process_lifecycle
    }

    /// Root PID/birth identity captured immediately after spawn.
    #[must_use]
    pub const fn root_process_identity(&self) -> LocalPerfRootProcessIdentity {
        self.root_process_identity
    }

    /// Explicit gaps for engine-internal facts the outer runner cannot prove.
    #[must_use]
    pub const fn internal_lifecycle_gaps(&self) -> LocalPerfInternalLifecycleGaps {
        self.internal_lifecycle_gaps
    }

    /// Runner controls that are structurally unsupported in this schema.
    #[must_use]
    pub fn unsupported_controls(&self) -> &[LocalPerfUnsupportedControl] {
        &self.unsupported_controls
    }

    /// SHA-256 of the exact synced combined child log.
    #[must_use]
    pub fn run_log_sha256(&self) -> Option<&str> {
        self.run_log_sha256.as_deref()
    }

    /// SHA-256 of exact admitted bound-evidence bytes for a completed attempt.
    /// Failed attempts carry `None` and cannot be joined to evidence by run ID.
    #[must_use]
    pub fn bound_evidence_sha256(&self) -> Option<&str> {
        self.bound_evidence_sha256.as_deref()
    }

    /// SHA-256 of the exact strict runner-receipt bytes admitted inside a
    /// completed bound-evidence artifact.
    #[must_use]
    pub fn runner_receipt_sha256(&self) -> Option<&str> {
        self.runner_receipt_sha256.as_deref()
    }

    /// SHA-256 of the exact strict artifact-manifest bytes bound into the
    /// completed runner identity.
    #[must_use]
    pub fn runner_artifact_manifest_sha256(&self) -> Option<&str> {
        self.runner_artifact_manifest_sha256.as_deref()
    }

    /// Bounded clock-capture failure retained when a failed attempt had to use
    /// its valid start timestamp as the conservative finish fallback.
    #[must_use]
    pub fn finished_timestamp_error(&self) -> Option<&str> {
        self.finished_timestamp_error.as_deref()
    }

    fn verify(&self) -> Result<(), LocalPerfRunError> {
        if self.schema_version != LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION
            || self.mode != "measurement"
            || self.registry_sha256 != MACHINE_CLASS_REGISTRY_SHA256
            || self
                .run_log_sha256
                .as_deref()
                .is_some_and(|digest| !is_sha256(digest))
            || !is_sha256(&self.seal_sha256)
        {
            return Err(LocalPerfRunError::Invalid(
                "attempt receipt has an invalid schema, mode, registry, or digest".to_owned(),
            ));
        }
        validate_component(&self.run_id, "attempt run ID")?;
        validate_component(&self.run_window, "attempt run window")?;
        let gate = self.gate.parse::<PerfGate>().map_err(|error| {
            LocalPerfRunError::Invalid(format!("attempt receipt names an invalid gate: {error}"))
        })?;
        if gate != self.applicability_plan.gate || self.profile != self.applicability_plan.profile {
            return Err(LocalPerfRunError::Invalid(
                "attempt gate/profile differs from its applicability-plan binding".to_owned(),
            ));
        }
        let registry = MachineClassRegistry::frozen()?;
        let expected_plan = PerfMatrixSpec::complete()
            .applicability_plan(&registry, self.profile, gate)
            .map_err(|error| {
                LocalPerfRunError::Invalid(format!(
                    "attempt applicability plan cannot be reconstructed: {error}"
                ))
            })?;
        let stored_selection = self
            .fixture_selector
            .as_deref()
            .map(|fixture| LocalPerfRunSelection::for_fixture(fixture.to_owned()))
            .transpose()?;
        let expected_selection = resolve_run_selection(&expected_plan, stored_selection.as_ref())?;
        if self.applicability_plan != *expected_plan.binding()
            || self.fixture_selector != expected_selection.fixture
            || self.selected_cell_ids != expected_selection.selected_cell_ids
        {
            return Err(LocalPerfRunError::Invalid(
                "attempt fixture selector or cells differ from the frozen canonical selection"
                    .to_owned(),
            ));
        }
        let context = MachineClassAdmissionContext {
            gate: self.gate.clone(),
            expected_profile: self.profile,
            destination_basename: self.profile.latest_basename(&self.gate)?,
        };
        registry.preflight(
            self.profile,
            self.hardware.clone(),
            self.execution_request.clone(),
            self.execution_start.clone(),
            self.durability.clone(),
            &context,
        )?;
        if let Some(end) = &self.execution_end {
            registry.preflight(
                self.profile,
                self.hardware.clone(),
                self.execution_request.clone(),
                end.clone(),
                self.durability.clone(),
                &context,
            )?;
        }
        if self.execution_end.is_some() == self.end_capture_error.is_some()
            || self
                .end_capture_error
                .as_deref()
                .is_some_and(|error| error.is_empty() || error.len() > 240)
            || self.post_run_identity_verified == self.post_run_identity_error.is_some()
            || self
                .post_run_identity_error
                .as_deref()
                .is_some_and(|error| error.is_empty() || error.len() > 240)
        {
            return Err(LocalPerfRunError::Invalid(
                "attempt end-capture or post-run identity evidence is contradictory".to_owned(),
            ));
        }
        match self.outcome {
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::EndPlatformCapture,
            } if self.execution_end.is_some() || self.end_capture_error.is_none() => {
                return Err(LocalPerfRunError::Invalid(
                    "end-platform-capture rejection must retain an explicit missing-end fact"
                        .to_owned(),
                ));
            }
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::PostRunIdentity,
            } if self.post_run_identity_verified || self.post_run_identity_error.is_none() => {
                return Err(LocalPerfRunError::Invalid(
                    "post-run-identity rejection must retain an explicit verification failure"
                        .to_owned(),
                ));
            }
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::FinishedTimestamp,
            } if self.finished_timestamp_error.is_none()
                || self.finished_at_utc != self.started_at_utc =>
            {
                return Err(LocalPerfRunError::Invalid(
                    "finish-timestamp rejection must retain the conservative start-time fallback"
                        .to_owned(),
                ));
            }
            _ => {}
        }
        validate_attempt_build(&self.build)?;
        let (expected_retry, unavailable) = attempt_derived_facts(self.outcome)?;
        match (
            self.outcome,
            self.bound_evidence_sha256.as_deref(),
            self.runner_receipt_sha256.as_deref(),
            self.runner_artifact_manifest_sha256.as_deref(),
        ) {
            (LocalPerfAttemptOutcome::Completed, Some(bound), Some(runner), Some(manifest))
                if is_sha256(bound) && is_sha256(runner) && is_sha256(manifest) =>
            {
                if self.execution_end.is_none()
                    || self.end_capture_error.is_some()
                    || !self.post_run_identity_verified
                    || self.post_run_identity_error.is_some()
                    || self.run_log_sha256.is_none()
                    || self.finished_timestamp_error.is_some()
                {
                    return Err(LocalPerfRunError::Invalid(
                        "completed attempt requires a terminal snapshot, verified post-run identity, and exact run log"
                            .to_owned(),
                    ));
                }
            }
            (LocalPerfAttemptOutcome::Completed, _, _, _) => {
                return Err(LocalPerfRunError::Invalid(
                    "completed attempt must bind exact bound-evidence, runner-receipt, and artifact-manifest bytes"
                        .to_owned(),
                ));
            }
            (_, None, None, None) => {}
            (_, _, _, _) => {
                return Err(LocalPerfRunError::Invalid(
                    "failed attempt must not claim completed bound evidence or nested runner identities"
                        .to_owned(),
                ));
            }
        }
        let expected_gaps = LocalPerfInternalLifecycleGaps {
            actual_work: unavailable,
            queue: unavailable,
            workers_joined: unavailable,
            feed_drained: unavailable,
            pending_zero: unavailable,
        };
        if self.retry != expected_retry
            || validate_process_lifecycle(
                self.outcome,
                self.process_lifecycle,
                self.run_log_sha256.is_some(),
            )
            .is_err()
            || validate_root_process_identity(
                self.outcome,
                self.process_lifecycle,
                self.root_process_identity,
            )
            .is_err()
            || self.internal_lifecycle_gaps != expected_gaps
            || self.unsupported_controls
                != [
                    LocalPerfUnsupportedControl::Timeout,
                    LocalPerfUnsupportedControl::Cancellation,
                ]
        {
            return Err(LocalPerfRunError::Invalid(
                "attempt terminal, retry, process lifecycle, internal gaps, or unsupported-control facts disagree"
                    .to_owned(),
            ));
        }
        validate_utc_timestamp(&self.started_at_utc, "attempt start")?;
        validate_utc_timestamp(&self.finished_at_utc, "attempt finish")?;
        if self.finished_at_utc < self.started_at_utc {
            return Err(LocalPerfRunError::Invalid(
                "attempt finish timestamp precedes its start timestamp".to_owned(),
            ));
        }
        if self
            .finished_timestamp_error
            .as_deref()
            .is_some_and(|error| error.is_empty() || error.len() > 240)
            || (self.finished_timestamp_error.is_some()
                && self.finished_at_utc != self.started_at_utc)
        {
            return Err(LocalPerfRunError::Invalid(
                "attempt finish-timestamp fallback evidence is contradictory".to_owned(),
            ));
        }
        let mut preimage = self.clone();
        let expected_seal = preimage.seal_sha256.clone();
        preimage.seal_sha256.clear();
        if sha256_hex(&serde_json::to_vec(&preimage)?) != expected_seal {
            return Err(LocalPerfRunError::Invalid(
                "attempt receipt content seal does not verify".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Encode the strict producer contract advertised by the executing finalizer.
///
/// The binary supplies the digest independently computed from its held
/// executable handle; runtime production rehashes another held handle before
/// receipt commit.
///
/// # Errors
///
/// Rejects a malformed executable digest or build-time dirty marker.
pub fn local_perf_producer_contract_json(
    executable_sha256: &str,
) -> Result<String, LocalPerfRunError> {
    if !is_sha256(executable_sha256) {
        return Err(LocalPerfRunError::Invalid(
            "typed producer executable digest is not lowercase SHA-256".to_owned(),
        ));
    }
    let contract = LocalPerfProducerContract {
        schema_version: PRODUCER_CONTRACT_SCHEMA_VERSION,
        machine_class_registry_sha256: MACHINE_CLASS_REGISTRY_SHA256,
        producer: embedded_producer(executable_sha256)?,
    };
    serde_json::to_string(&contract).map_err(LocalPerfRunError::from)
}

/// Execute and finalize one registered-host benchmark invocation.
///
/// The producer derives and acquires the canonical host-global lease before
/// validation or benchmark compilation and holds it until every output is
/// synced. Every completed or supported failed child emits the same strict
/// process-attempt schema; only a completed receipt binds admitted evidence.
///
/// # Errors
///
/// Returns a typed failure for an unavailable lease, unsupported platform,
/// dirty/offloaded source, probe drift, child failure, malformed artifact, or
/// any self-admission mismatch.
pub fn run_local_perf_command(
    config: &LocalPerfRunConfig,
) -> Result<LocalPerfRunOutput, LocalPerfRunError> {
    run_local_perf_command_inner(config, None)
}

/// Execute and finalize one typed exact-fixture partial-shard invocation.
///
/// This is the only supported path that injects `QUILL_PERF_FIXTURE`. The
/// selector is resolved by exact equality against the frozen canonical matrix;
/// ambient selector state remains forbidden and is never inherited.
///
/// # Errors
///
/// Returns the same failures as [`run_local_perf_command`], plus a typed
/// rejection for unknown or non-applicable fixture selection.
#[allow(dead_code)]
pub fn run_selected_local_perf_command(
    config: &LocalPerfRunConfig,
    selection: &LocalPerfRunSelection,
) -> Result<LocalPerfRunOutput, LocalPerfRunError> {
    run_local_perf_command_inner(config, Some(selection))
}

// Each Result match assigns a distinct typed receipt rejection stage. Keeping
// the success value and its error-to-stage mapping together is auditability,
// not an ad-hoc manual let/else conversion.
#[allow(clippy::manual_let_else)]
fn run_local_perf_command_inner(
    config: &LocalPerfRunConfig,
    selection: Option<&LocalPerfRunSelection>,
) -> Result<LocalPerfRunOutput, LocalPerfRunError> {
    validate_bounded_inputs(config)?;
    validate_platform_gate_policy(config)?;
    let registry = MachineClassRegistry::frozen()?;
    let run_profile = resolve_run_profile(config, &registry)?;
    let run_selection = resolve_run_selection(&run_profile.applicability_plan, selection)?;
    let lease_path = stable_lease_path(config.profile.hardware_class_id())?;
    validate_canonical_lease_parent(&lease_path)?;
    let (lease_file, lease_identity) = acquire_family_lease(&lease_path)?;

    require_local_execution()?;
    reject_ambient_git_environment()?;
    reject_ambient_process_injection()?;
    let external_paths = validate_config(config)?;
    let source_before = capture_clean_source()?;
    let producer_before = capture_validated_producer(&source_before)?;
    let run_directories = create_run_directories(config, &external_paths.output_parent)?;
    let artifact_dir = benchmark_artifact_directory_path(&run_directories.artifacts)?;
    let environments = controlled_environments(
        config,
        &run_selection,
        &source_before,
        &external_paths.target,
        &artifact_dir,
    )?;
    let captured_build = prepare_benchmark(
        &source_before,
        &producer_before,
        &external_paths.target,
        environments,
    )?;

    verify_family_lease_path(&lease_path, &lease_identity)?;
    verify_external_paths(&external_paths)?;
    verify_run_directories(config, &run_directories)?;
    verify_prepared_build(&captured_build, &producer_before, &external_paths.target)?;
    let start = capture_platform(config)?;
    let started_at_utc = utc_now()?;
    let context = MachineClassAdmissionContext {
        gate: config.gate.label().to_owned(),
        expected_profile: config.profile,
        destination_basename: config.profile.latest_basename(config.gate.label())?,
    };
    let durability = durability_for_run(config)?;
    let pre_spawn = registry.preflight(
        config.profile,
        start.hardware.clone(),
        start.request.clone(),
        start.snapshot.clone(),
        durability.clone(),
        &context,
    )?;
    verify_family_lease_path(&lease_path, &lease_identity)?;
    verify_external_paths(&external_paths)?;
    verify_run_directories(config, &run_directories)?;
    verify_prepared_build(&captured_build, &producer_before, &external_paths.target)?;

    let environment_policy_path = config.output_dir.join("environment-policy.json");
    write_new_sync_at(
        &run_directories.run.handle,
        "environment-policy.json",
        &captured_build.environment_policy_bytes,
    )?;
    run_directories.run.handle.sync_all()?;
    verify_environment_policy(&run_directories.run.handle, &captured_build)?;
    let run_log_path = config.output_dir.join("run.log");
    let run_log = create_new_file_at(&run_directories.run.handle, "run.log")?;
    let run_log_sync = run_log.try_clone()?;
    let run_log_stderr = run_log.try_clone()?;
    let mut child = Command::new(descriptor_path(&captured_build.executable)?);
    child
        .arg0(&captured_build.command[0])
        .args(&captured_build.command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::from(run_log))
        .stderr(Stdio::from(run_log_stderr));
    configure_benchmark_child(&mut child, &captured_build.measurement_environment);
    let (mut child, root_process_identity) = match child.spawn() {
        Ok(child) => {
            let root_process_identity = capture_root_process_identity(&child);
            (child, root_process_identity)
        }
        Err(error) => {
            let run_log_synced = run_log_sync.sync_all().is_ok();
            let run_log_bytes = read_file_at(&run_directories.run.handle, "run.log").ok();
            let _ = write_new_sync_at(
                &run_directories.run.handle,
                "exit-status",
                b"spawn-rejected\n",
            );
            let outcome = LocalPerfAttemptOutcome::SpawnRejected {
                error_kind: local_perf_io_error_kind(&error),
            };
            let process_lifecycle = LocalPerfProcessLifecycle {
                spawn_attempted: true,
                spawn_succeeded: false,
                wait_completed: false,
                child_reaped: false,
                run_log_synced,
                run_log_captured: run_log_bytes.is_some(),
                process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
            };
            let attempt_path = write_failed_attempt_receipt(
                config,
                &run_profile,
                &run_selection,
                &durability,
                &run_directories,
                &captured_build,
                &producer_before,
                &external_paths,
                &start,
                outcome,
                process_lifecycle,
                LocalPerfRootProcessIdentity::NotSpawned,
                run_log_bytes.as_deref(),
                &started_at_utc,
            )?;
            return Err(LocalPerfRunError::AttemptFailed {
                receipt_path: attempt_path,
                outcome,
            });
        }
    };
    let (status, recovered_wait_error, process_group_recovery) = match child.wait() {
        Ok(status) => (status, None, LocalPerfProcessGroupRecovery::NotRequired),
        Err(error) => {
            let wait_error_kind = local_perf_io_error_kind(&error);
            match force_kill_and_reap(&mut child, root_process_identity) {
                Ok((status, process_group_recovery)) => {
                    (status, Some(wait_error_kind), process_group_recovery)
                }
                Err(recovery_error_kind) => {
                    return Err(LocalPerfRunError::UnreapedChild {
                        wait_error_kind,
                        recovery_error_kind,
                    });
                }
            }
        }
    };
    let run_log_synced = run_log_sync.sync_all().is_ok();
    let run_log_result = read_file_at(&run_directories.run.handle, "run.log");
    let process_lifecycle = LocalPerfProcessLifecycle {
        spawn_attempted: true,
        spawn_succeeded: true,
        wait_completed: true,
        child_reaped: true,
        run_log_synced,
        run_log_captured: run_log_result.is_ok(),
        process_group_recovery,
    };
    if !run_log_synced {
        let outcome = LocalPerfAttemptOutcome::PostExitRejected {
            stage: LocalPerfRejectionStage::RunLogSync,
        };
        let attempt_path = write_failed_attempt_receipt(
            config,
            &run_profile,
            &run_selection,
            &durability,
            &run_directories,
            &captured_build,
            &producer_before,
            &external_paths,
            &start,
            outcome,
            process_lifecycle,
            root_process_identity,
            run_log_result.as_deref().ok(),
            &started_at_utc,
        )?;
        return Err(LocalPerfRunError::AttemptFailed {
            receipt_path: attempt_path,
            outcome,
        });
    }
    let run_log_bytes = match run_log_result {
        Ok(bytes) => bytes,
        Err(_) => {
            let outcome = LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::RunLogRead,
            };
            let attempt_path = write_failed_attempt_receipt(
                config,
                &run_profile,
                &run_selection,
                &durability,
                &run_directories,
                &captured_build,
                &producer_before,
                &external_paths,
                &start,
                outcome,
                process_lifecycle,
                root_process_identity,
                None,
                &started_at_utc,
            )?;
            return Err(LocalPerfRunError::AttemptFailed {
                receipt_path: attempt_path,
                outcome,
            });
        }
    };
    let exit_code = status.code().map_or(-1, i64::from);
    if write_new_sync_at(
        &run_directories.run.handle,
        "exit-status",
        format!("{exit_code}\n").as_bytes(),
    )
    .is_err()
    {
        let outcome = LocalPerfAttemptOutcome::PostExitRejected {
            stage: LocalPerfRejectionStage::ExitStatusPersistence,
        };
        let attempt_path = write_failed_attempt_receipt(
            config,
            &run_profile,
            &run_selection,
            &durability,
            &run_directories,
            &captured_build,
            &producer_before,
            &external_paths,
            &start,
            outcome,
            process_lifecycle,
            root_process_identity,
            Some(&run_log_bytes),
            &started_at_utc,
        )?;
        return Err(LocalPerfRunError::AttemptFailed {
            receipt_path: attempt_path,
            outcome,
        });
    }
    if let Some(error_kind) = recovered_wait_error {
        let outcome = LocalPerfAttemptOutcome::WaitRecoveredByKill { error_kind };
        let attempt_path = write_failed_attempt_receipt(
            config,
            &run_profile,
            &run_selection,
            &durability,
            &run_directories,
            &captured_build,
            &producer_before,
            &external_paths,
            &start,
            outcome,
            process_lifecycle,
            root_process_identity,
            Some(&run_log_bytes),
            &started_at_utc,
        )?;
        return Err(LocalPerfRunError::AttemptFailed {
            receipt_path: attempt_path,
            outcome,
        });
    }
    if !status.success() {
        let outcome = status.code().map_or_else(
            || {
                status
                    .signal()
                    .map_or(LocalPerfAttemptOutcome::UnknownTerminal, |signal| {
                        LocalPerfAttemptOutcome::Signaled { signal }
                    })
            },
            |code| LocalPerfAttemptOutcome::ExitedNonzero {
                code: i64::from(code),
            },
        );
        let attempt_path = write_failed_attempt_receipt(
            config,
            &run_profile,
            &run_selection,
            &durability,
            &run_directories,
            &captured_build,
            &producer_before,
            &external_paths,
            &start,
            outcome,
            process_lifecycle,
            root_process_identity,
            Some(&run_log_bytes),
            &started_at_utc,
        )?;
        return Err(LocalPerfRunError::AttemptFailed {
            receipt_path: attempt_path,
            outcome,
        });
    }

    let post_exit_rejection = |stage| -> Result<LocalPerfRunError, LocalPerfRunError> {
        let outcome = LocalPerfAttemptOutcome::PostExitRejected { stage };
        let receipt_path = write_failed_attempt_receipt(
            config,
            &run_profile,
            &run_selection,
            &durability,
            &run_directories,
            &captured_build,
            &producer_before,
            &external_paths,
            &start,
            outcome,
            process_lifecycle,
            root_process_identity,
            Some(&run_log_bytes),
            &started_at_utc,
        )?;
        Ok(LocalPerfRunError::AttemptFailed {
            receipt_path,
            outcome,
        })
    };
    let end = match capture_platform(config) {
        Ok(end) => end,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::EndPlatformCapture,
            )?);
        }
    };
    if !root_process_identity.has_verified_birth_identity() {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::RootProcessIdentity,
        )?);
    }
    let finished_at_utc = match utc_now() {
        Ok(timestamp) => timestamp,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::FinishedTimestamp,
            )?);
        }
    };
    if verify_prepared_build(&captured_build, &producer_before, &external_paths.target).is_err() {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PostRunIdentity,
        )?);
    }
    if start.hardware != end.hardware
        || start.request != end.request
        || start.snapshot != end.snapshot
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PostRunIdentity,
        )?);
    }

    let threshold_name = format!("{}.json", config.gate.label());
    let evidence_name = format!("{}.evidence.json", config.gate.label());
    let threshold_path = run_directories.artifacts.path.join(&threshold_name);
    let prebinding_evidence_path = run_directories.artifacts.path.join(&evidence_name);
    let durable_child_artifacts = match read_and_sync_child_artifacts(
        &run_directories.artifacts.handle,
        &threshold_name,
        &evidence_name,
    ) {
        Ok(artifacts) => artifacts,
        Err(stage) => return Err(post_exit_rejection(stage)?),
    };
    let threshold_bytes = durable_child_artifacts.threshold_bytes;
    let evidence_bytes = durable_child_artifacts.evidence_bytes;
    let threshold = match read_canonical_threshold(&threshold_bytes) {
        Ok(threshold) => threshold,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::ArtifactVerification,
            )?);
        }
    };
    let evidence = match read_canonical_evidence(&evidence_bytes) {
        Ok(evidence) => evidence,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::ArtifactVerification,
            )?);
        }
    };
    if validate_child_artifacts(
        config,
        &run_profile,
        &run_selection,
        &captured_build,
        &start,
        &threshold,
        &evidence,
    )
    .is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::ArtifactVerification,
        )?);
    }

    let artifact_manifest = match RunnerArtifactManifest::from_artifacts(
        &run_profile.applicability_plan,
        &config.run_id,
        &config.run_window,
        &run_log_bytes,
        &threshold_bytes,
        &evidence_bytes,
    ) {
        Ok(manifest) => manifest,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::ArtifactManifestSerialization,
            )?);
        }
    };
    let artifact_manifest_bytes = match artifact_manifest.to_json_bytes() {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::ArtifactManifestSerialization,
            )?);
        }
    };
    if verify_prepared_build(&captured_build, &producer_before, &external_paths.target).is_err() {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PostRunIdentity,
        )?);
    }
    let producer_identity_sha256 = match serde_json::to_vec(&captured_build.receipt.producer) {
        Ok(bytes) => sha256_hex(&bytes),
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::PrecommitSerialization,
            )?);
        }
    };
    let receipt = RunnerReceipt {
        schema_version: RUNNER_RECEIPT_SCHEMA_VERSION.to_owned(),
        requested_profile: config.profile,
        derived_profile: config.profile,
        registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
        hardware: start.hardware.clone(),
        execution: RunnerExecution {
            request: start.request.clone(),
            start: start.snapshot.clone(),
            end: end.snapshot.clone(),
            identity_sha256: String::new(),
        },
        build: captured_build.receipt.clone(),
        durability: durability.clone(),
        completion: RunnerCompletion {
            verified: true,
            exit_status: exit_code,
            run_log_sha256: sha256_hex(&run_log_bytes),
            artifact_manifest_sha256: sha256_hex(&artifact_manifest_bytes),
            artifact_digests_verified: true,
            started_at_utc: started_at_utc.clone(),
            finished_at_utc: finished_at_utc.clone(),
        },
    };
    let receipt_bytes = match seal_runner_receipt(receipt) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::RunnerReceiptSerialization,
            )?);
        }
    };
    let identity = match registry.admit(&receipt_bytes, &context) {
        Ok(identity) => identity,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::RunnerAdmission,
            )?);
        }
    };
    if pre_spawn.verify_final(&identity).is_err() {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::RunnerAdmission,
        )?);
    }
    let identity = match identity.bind_artifact_manifest(
        &artifact_manifest_bytes,
        &run_log_bytes,
        &threshold_bytes,
        &evidence_bytes,
    ) {
        Ok(identity) => identity,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::RunnerAdmission,
            )?);
        }
    };
    let mut bound_preview = evidence;
    let bound_evidence_bytes = match bound_preview.bind_machine_class_identity_and_seal(
        identity,
        &threshold_bytes,
        &evidence_bytes,
    ) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::BoundEvidenceSerialization,
            )?);
        }
    };
    if PerfEvidenceArtifact::from_verified_slice(&bound_evidence_bytes).is_err() {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::ArtifactVerification,
        )?);
    }

    let manifest_path = config
        .output_dir
        .join(format!("{}.artifacts.json", config.gate.label()));
    let receipt_path = config
        .output_dir
        .join(format!("{}.runner.json", config.gate.label()));
    let bound_evidence_path = config
        .output_dir
        .join(format!("{}.bound.evidence.json", config.gate.label()));
    let inventory_path = config.output_dir.join("PRECOMMIT.json");
    let inventory = PrecommitInventory {
        schema_version: PERF_RUN_PRECOMMIT_SCHEMA_VERSION.to_owned(),
        gate: config.gate.label().to_owned(),
        profile: config.profile,
        execution_capacity: run_profile.execution_capacity,
        max_exercised_cell_width: run_profile.max_exercised_cell_width,
        applicability_plan: run_profile.applicability_plan.binding().clone(),
        fixture_selector: run_selection.fixture.clone(),
        selected_cell_ids: run_selection.selected_cell_ids.clone(),
        run_id: config.run_id.clone(),
        run_window: config.run_window.clone(),
        run_log_sha256: sha256_hex(&run_log_bytes),
        threshold_artifact_sha256: sha256_hex(&threshold_bytes),
        evidence_artifact_sha256: sha256_hex(&evidence_bytes),
        artifact_manifest_sha256: sha256_hex(&artifact_manifest_bytes),
        runner_receipt_sha256: sha256_hex(&receipt_bytes),
        producer_identity_sha256,
        bound_evidence_preview_sha256: sha256_hex(&bound_evidence_bytes),
        environment_policy_sha256: captured_build.receipt.environment_sha256.clone(),
    };
    let inventory_bytes = match serde_json::to_vec_pretty(&inventory) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::PrecommitSerialization,
            )?);
        }
    };
    let completed_attempt_bytes = match completed_attempt_receipt_bytes(
        config,
        &run_profile,
        &run_selection,
        &durability,
        &captured_build,
        &start,
        &end,
        &bound_evidence_bytes,
        &run_log_bytes,
        root_process_identity,
        &started_at_utc,
        &finished_at_utc,
    ) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::AttemptReceiptPersistence,
            )?);
        }
    };

    // The child files and artifact directory were synced before the nested
    // runner receipt was created. Raw diagnostic artifacts, the nested runner
    // receipt, and exact bound evidence are then made durable in that order.
    // Bound evidence alone is orphan-ineligible. The sole attempt receipt is
    // atomically published last, so only the verified receipt/evidence pair is
    // a complete H4 shard.
    let Some(manifest_name) = manifest_path.file_name() else {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PrecommitSerialization,
        )?);
    };
    let Some(inventory_name) = inventory_path.file_name() else {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PrecommitSerialization,
        )?);
    };
    let Some(receipt_name) = receipt_path.file_name() else {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::RunnerReceiptPersistence,
        )?);
    };
    let Some(bound_evidence_name) = bound_evidence_path.file_name() else {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::BoundEvidencePersistence,
        )?);
    };
    if write_new_sync_at(
        &run_directories.run.handle,
        manifest_name,
        &artifact_manifest_bytes,
    )
    .and_then(|()| {
        write_new_sync_at(
            &run_directories.run.handle,
            inventory_name,
            &inventory_bytes,
        )
    })
    .and_then(|()| {
        run_directories
            .run
            .handle
            .sync_all()
            .map_err(LocalPerfRunError::from)
    })
    .is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PrecommitPersistence,
        )?);
    }
    if verify_family_lease_path(&lease_path, &lease_identity)
        .and_then(|()| verify_external_paths(&external_paths))
        .and_then(|()| verify_run_directories(config, &run_directories))
        .and_then(|()| {
            verify_prepared_build(&captured_build, &producer_before, &external_paths.target)
        })
        .and_then(|()| verify_environment_policy(&run_directories.run.handle, &captured_build))
        .is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PostRunIdentity,
        )?);
    }
    if write_new_sync_at(&run_directories.run.handle, receipt_name, &receipt_bytes)
        .and_then(|()| {
            run_directories
                .run
                .handle
                .sync_all()
                .map_err(LocalPerfRunError::from)
        })
        .is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::RunnerReceiptPersistence,
        )?);
    }
    if verify_external_paths(&external_paths)
        .and_then(|()| verify_run_directories(config, &run_directories))
        .and_then(|()| {
            verify_prepared_build(&captured_build, &producer_before, &external_paths.target)
        })
        .and_then(|()| verify_environment_policy(&run_directories.run.handle, &captured_build))
        .is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PostRunIdentity,
        )?);
    }
    if write_new_sync_at(
        &run_directories.run.handle,
        bound_evidence_name,
        &bound_evidence_bytes,
    )
    .and_then(|()| {
        run_directories
            .run
            .handle
            .sync_all()
            .map_err(LocalPerfRunError::from)
    })
    .is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::BoundEvidencePersistence,
        )?);
    }
    let persisted_bound = match read_file_at(&run_directories.run.handle, bound_evidence_name) {
        Ok(bytes) => bytes,
        Err(_) => {
            return Err(post_exit_rejection(
                LocalPerfRejectionStage::PersistedPairVerification,
            )?);
        }
    };
    if persisted_bound != bound_evidence_bytes
        || PerfEvidenceArtifact::from_verified_slice(&persisted_bound).is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PersistedPairVerification,
        )?);
    }
    let completed_attempt =
        match LocalPerfAttemptReceipt::from_verified_slice(&completed_attempt_bytes) {
            Ok(receipt) => receipt,
            Err(_) => {
                return Err(post_exit_rejection(
                    LocalPerfRejectionStage::PersistedPairVerification,
                )?);
            }
        };
    if completed_attempt.verify_run_log(&run_log_bytes).is_err()
        || completed_attempt
            .verify_bound_evidence(&persisted_bound)
            .is_err()
        || verify_external_paths(&external_paths).is_err()
        || verify_run_directories(config, &run_directories).is_err()
    {
        return Err(post_exit_rejection(
            LocalPerfRejectionStage::PersistedPairVerification,
        )?);
    }
    let attempt_receipt_path = config
        .output_dir
        .join(format!("{}.attempt.json", config.gate.label()));
    let attempt_pending_name = format!("{}.attempt.pending", config.gate.label());
    let attempt_name = format!("{}.attempt.json", config.gate.label());
    if let Err(error) = atomically_publish_new_sync_at(
        &run_directories.run.handle,
        &attempt_pending_name,
        &attempt_name,
        &completed_attempt_bytes,
    ) {
        return Err(LocalPerfRunError::AttemptCommitFailed {
            receipt_path: attempt_receipt_path,
            outcome: LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::AttemptReceiptPersistence,
            },
            detail: bounded_diagnostic(&error),
        });
    }
    let persisted_attempt =
        read_file_at(&run_directories.run.handle, &attempt_name).map_err(|error| {
            LocalPerfRunError::AttemptCommitFailed {
                receipt_path: attempt_receipt_path.clone(),
                outcome: LocalPerfAttemptOutcome::PostExitRejected {
                    stage: LocalPerfRejectionStage::PersistedPairVerification,
                },
                detail: bounded_diagnostic(&error),
            }
        })?;
    if persisted_attempt != completed_attempt_bytes
        || LocalPerfAttemptReceipt::from_verified_slice(&persisted_attempt)
            .and_then(|receipt| receipt.verify_bound_evidence(&persisted_bound))
            .is_err()
    {
        return Err(LocalPerfRunError::AttemptCommitFailed {
            receipt_path: attempt_receipt_path.clone(),
            outcome: LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::PersistedPairVerification,
            },
            detail: "persisted final attempt/evidence pair failed exact verification".to_owned(),
        });
    }

    drop(lease_file);
    Ok(LocalPerfRunOutput {
        run_log: run_log_path,
        artifact_manifest: manifest_path,
        environment_policy: environment_policy_path,
        runner_receipt: receipt_path,
        attempt_receipt: attempt_receipt_path,
        threshold_artifact: threshold_path,
        prebinding_evidence: prebinding_evidence_path,
        bound_evidence: bound_evidence_path,
        precommit_inventory: inventory_path,
    })
}

fn validate_config(config: &LocalPerfRunConfig) -> Result<ExternalRunPaths, LocalPerfRunError> {
    validate_platform_gate_policy(config)?;
    validate_external_paths(config)
}

fn validate_bounded_inputs(config: &LocalPerfRunConfig) -> Result<(), LocalPerfRunError> {
    for (name, value) in [
        (
            "hardware class ID",
            config.profile.hardware_class_id().as_str(),
        ),
        (
            "execution profile ID",
            config.profile.execution_profile_id().as_str(),
        ),
        ("run ID", config.run_id.as_str()),
        ("run window", config.run_window.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(LocalPerfRunError::Invalid(format!("{name} is empty")));
        }
    }
    if !(MIN_MEASUREMENT_RUNS..=MAX_MEASUREMENT_RUNS).contains(&config.measurement_runs) {
        return Err(LocalPerfRunError::Invalid(format!(
            "measurement runs must remain within {MIN_MEASUREMENT_RUNS}..={MAX_MEASUREMENT_RUNS}"
        )));
    }
    for (field, value) in [
        (
            "hardware class ID",
            config.profile.hardware_class_id().as_str(),
        ),
        (
            "execution profile ID",
            config.profile.execution_profile_id().as_str(),
        ),
        ("run ID", config.run_id.as_str()),
        ("run window", config.run_window.as_str()),
    ] {
        validate_component(value, field)?;
    }
    Ok(())
}

fn resolve_run_profile(
    config: &LocalPerfRunConfig,
    registry: &MachineClassRegistry,
) -> Result<RunProfileContract, LocalPerfRunError> {
    let profile = registry.execution_profile(config.profile)?;
    if profile.availability() != MachineProfileAvailability::Registered {
        return Err(LocalPerfRunError::Invalid(format!(
            "execution profile {}.{} is unavailable",
            config.profile.hardware_class_id().as_str(),
            config.profile.execution_profile_id().as_str()
        )));
    }
    let execution_capacity = profile.execution_capacity().ok_or_else(|| {
        LocalPerfRunError::Invalid(
            "typed promotion runner requires a frozen non-diagnostic execution capacity".to_owned(),
        )
    })?;
    let max_exercised_cell_width = profile
        .gate_policy(config.gate.label())
        .and_then(|policy| policy.max_exercised_cell_width())
        .ok_or_else(|| {
            LocalPerfRunError::Invalid(format!(
                "execution profile has no runnable width for {}",
                config.gate
            ))
        })?;
    let applicability_plan = PerfMatrixSpec::complete()
        .applicability_plan(registry, config.profile, config.gate)
        .map_err(|error| {
            LocalPerfRunError::Invalid(format!(
                "cannot construct the frozen profile applicability plan: {error}"
            ))
        })?;
    if applicability_plan.execution_capacity != Some(execution_capacity)
        || applicability_plan.max_exercised_cell_width != Some(max_exercised_cell_width)
        || applicability_plan.capacity_semantics != profile.capacity_semantics()
    {
        return Err(LocalPerfRunError::Invalid(
            "registry profile and canonical applicability plan disagree on the execution envelope"
                .to_owned(),
        ));
    }
    Ok(RunProfileContract {
        capacity_semantics: profile.capacity_semantics(),
        execution_capacity,
        max_exercised_cell_width,
        applicability_plan,
    })
}

fn validate_component(value: &str, field: &str) -> Result<(), LocalPerfRunError> {
    if value.is_empty()
        || value.len() > MAX_IDENTITY_COMPONENT_BYTES
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(LocalPerfRunError::Invalid(format!(
            "{field} must be a safe ASCII identity component of at most {MAX_IDENTITY_COMPONENT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_platform_gate_policy(config: &LocalPerfRunConfig) -> Result<(), LocalPerfRunError> {
    if matches!(config.gate, PerfGate::Qg3 | PerfGate::Qg4 | PerfGate::Qg5) {
        return Err(LocalPerfRunError::Invalid(format!(
            "{} cannot run promotion-grade on any host until both benchmark arms emit a non-declarative symmetric durability-treatment witness",
            config.gate
        )));
    }
    if config.profile.hardware_class_id() == HardwareClassId::M4Macos {
        return Err(LocalPerfRunError::Invalid(
            "m4-macos.scheduler-10 remains a required, runnable static registry applicability plan; that static plan does not attest a live M4 host, and this producer cannot emit promotion-admissible M4 evidence until it can attest the actual executing image through a supported O_EXEC or loaded-image mechanism; use the diagnostic Apple profiling path until an attesting producer lands"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_external_paths(
    config: &LocalPerfRunConfig,
) -> Result<ExternalRunPaths, LocalPerfRunError> {
    let repository = capture_repository_root()?;
    if !config.output_dir.is_absolute() {
        return Err(LocalPerfRunError::Invalid(
            "output directory must be absolute".to_owned(),
        ));
    }
    let output_leaf = config.output_dir.file_name().ok_or_else(|| {
        LocalPerfRunError::Invalid("output directory must have a final component".to_owned())
    })?;
    validate_output_component(output_leaf)?;
    match fs::symlink_metadata(&config.output_dir) {
        Ok(_) => {
            return Err(LocalPerfRunError::Invalid(format!(
                "output directory {} already exists",
                config.output_dir.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let output_parent = config
        .output_dir
        .parent()
        .ok_or_else(|| LocalPerfRunError::Invalid("output directory has no parent".to_owned()))?;
    let parent_metadata = fs::symlink_metadata(output_parent).map_err(|error| {
        LocalPerfRunError::Invalid(format!(
            "output parent {} must already exist and resolve cleanly: {error}",
            output_parent.display()
        ))
    })?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(LocalPerfRunError::Invalid(
            "output parent must be an existing non-symlink directory".to_owned(),
        ));
    }
    let resolved_output_parent = fs::canonicalize(output_parent)?;
    if output_parent != resolved_output_parent {
        return Err(LocalPerfRunError::Invalid(
            "output parent must contain no symbolic-link or lexical aliases".to_owned(),
        ));
    }
    if resolved_output_parent.starts_with(&repository) {
        return Err(LocalPerfRunError::Invalid(format!(
            "output parent {} must remain outside the source repository",
            resolved_output_parent.display()
        )));
    }
    let output_parent = pin_directory(&resolved_output_parent, true)?;
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "typed producer requires an explicit CARGO_TARGET_DIR outside the source repository"
                    .to_owned(),
            )
        })?;
    if !target.is_absolute() {
        return Err(LocalPerfRunError::Invalid(
            "CARGO_TARGET_DIR must be absolute".to_owned(),
        ));
    }
    let target_metadata = fs::symlink_metadata(&target).map_err(|error| {
        LocalPerfRunError::Invalid(format!(
            "CARGO_TARGET_DIR {} must already exist and resolve cleanly: {error}",
            target.display()
        ))
    })?;
    if target_metadata.file_type().is_symlink() || !target_metadata.is_dir() {
        return Err(LocalPerfRunError::Invalid(
            "CARGO_TARGET_DIR must be an existing non-symlink directory".to_owned(),
        ));
    }
    let resolved_target = fs::canonicalize(&target)?;
    if target != resolved_target {
        return Err(LocalPerfRunError::Invalid(
            "CARGO_TARGET_DIR must contain no symbolic-link or lexical aliases".to_owned(),
        ));
    }
    if resolved_target.starts_with(&repository) {
        return Err(LocalPerfRunError::Invalid(
            "CARGO_TARGET_DIR must remain outside the source repository".to_owned(),
        ));
    }
    let target = pin_directory(&resolved_target, true)?;
    Ok(ExternalRunPaths {
        output_parent,
        target,
    })
}

fn validate_output_component(value: &OsStr) -> Result<(), LocalPerfRunError> {
    let bytes = value.as_encoded_bytes();
    if bytes.is_empty()
        || bytes.len() > MAX_OUTPUT_COMPONENT_BYTES
        || matches!(bytes, b"." | b"..")
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(LocalPerfRunError::Invalid(format!(
            "output basename must be a safe ASCII component of at most {MAX_OUTPUT_COMPONENT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn pin_directory(path: &Path, close_on_exec: bool) -> Result<PinnedDirectory, LocalPerfRunError> {
    let mut flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW;
    if close_on_exec {
        flags |= OFlags::CLOEXEC;
    }
    let handle = File::from(open(path, flags, Mode::empty()).map_err(std::io::Error::from)?);
    let identity = checked_directory_identity(&handle)?;
    Ok(PinnedDirectory {
        path: path.to_path_buf(),
        handle,
        identity,
    })
}

fn checked_directory_identity(handle: &File) -> Result<FileIdentity, LocalPerfRunError> {
    let metadata = handle.metadata()?;
    if !metadata.is_dir() {
        return Err(LocalPerfRunError::Invalid(
            "pinned run root is not a directory".to_owned(),
        ));
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

fn verify_pinned_directory(directory: &PinnedDirectory) -> Result<(), LocalPerfRunError> {
    let held = checked_directory_identity(&directory.handle)?;
    let metadata = fs::symlink_metadata(&directory.path)?;
    let path_identity = FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    if held != directory.identity
        || path_identity != directory.identity
        || metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || fs::canonicalize(&directory.path)? != directory.path
    {
        return Err(LocalPerfRunError::Invalid(
            "pinned directory path changed identity during the invocation".to_owned(),
        ));
    }
    Ok(())
}

fn verify_external_paths(paths: &ExternalRunPaths) -> Result<(), LocalPerfRunError> {
    verify_pinned_directory(&paths.output_parent)?;
    verify_pinned_directory(&paths.target)
}

fn create_run_directories(
    config: &LocalPerfRunConfig,
    output_parent: &PinnedDirectory,
) -> Result<RunDirectories, LocalPerfRunError> {
    verify_pinned_directory(output_parent)?;
    let output_leaf = config.output_dir.file_name().ok_or_else(|| {
        LocalPerfRunError::Invalid("output directory must have a final component".to_owned())
    })?;
    mkdirat(
        &output_parent.handle,
        output_leaf,
        Mode::from_raw_mode(0o700),
    )
    .map_err(std::io::Error::from)?;
    output_parent.handle.sync_all()?;
    let run_handle = File::from(
        openat(
            &output_parent.handle,
            output_leaf,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?,
    );
    let run = PinnedDirectory {
        path: config.output_dir.clone(),
        identity: checked_directory_identity(&run_handle)?,
        handle: run_handle,
    };
    mkdirat(&run.handle, "artifacts", Mode::from_raw_mode(0o700)).map_err(std::io::Error::from)?;
    let artifact_handle = File::from(
        openat(
            &run.handle,
            "artifacts",
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?,
    );
    let artifacts = PinnedDirectory {
        path: config.output_dir.join("artifacts"),
        identity: checked_directory_identity(&artifact_handle)?,
        handle: artifact_handle,
    };
    let directories = RunDirectories { run, artifacts };
    verify_run_directories(config, &directories)?;
    Ok(directories)
}

fn verify_run_directories(
    config: &LocalPerfRunConfig,
    directories: &RunDirectories,
) -> Result<(), LocalPerfRunError> {
    if directories.run.path != config.output_dir
        || directories.artifacts.path != config.output_dir.join("artifacts")
    {
        return Err(LocalPerfRunError::Invalid(
            "held run directories do not match the requested output identity".to_owned(),
        ));
    }
    verify_pinned_directory(&directories.run)?;
    verify_pinned_directory(&directories.artifacts)
}

fn descriptor_path(file: &File) -> Result<PathBuf, LocalPerfRunError> {
    match std::env::consts::OS {
        "linux" => Ok(PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))),
        "macos" => Err(LocalPerfRunError::Invalid(
            "macOS descriptor execution is not promotion-admissible without a supported O_EXEC or loaded-image attestation mechanism"
                .to_owned(),
        )),
        other => Err(LocalPerfRunError::Invalid(format!(
            "unsupported descriptor-path OS {other:?}"
        ))),
    }
}

fn benchmark_artifact_directory_path(
    directory: &PinnedDirectory,
) -> Result<PathBuf, LocalPerfRunError> {
    verify_pinned_directory(directory)?;
    benchmark_artifact_directory_path_for_os(
        &directory.path,
        directory.handle.as_raw_fd(),
        std::env::consts::OS,
    )
}

fn benchmark_artifact_directory_path_for_os(
    canonical_path: &Path,
    descriptor: i32,
    os: &str,
) -> Result<PathBuf, LocalPerfRunError> {
    match os {
        "linux" => Ok(PathBuf::from(format!("/proc/self/fd/{descriptor}"))),
        "macos" => Ok(canonical_path.to_path_buf()),
        other => Err(LocalPerfRunError::Invalid(format!(
            "unsupported benchmark artifact-directory OS {other:?}"
        ))),
    }
}

fn require_local_execution() -> Result<(), LocalPerfRunError> {
    if std::env::var("RCH_DISABLE").as_deref() != Ok("1")
        || std::env::var("RCH_CARGO_WRAPPER_BYPASS").as_deref() != Ok("1")
    {
        return Err(LocalPerfRunError::Invalid(
            "timed producer requires RCH_DISABLE=1 and RCH_CARGO_WRAPPER_BYPASS=1".to_owned(),
        ));
    }
    for variable in ["RCH_WORKER_ID", "RCH_REMOTE_EXECUTION", "RCH_JOB_ID"] {
        if std::env::var_os(variable).is_some() {
            return Err(LocalPerfRunError::Invalid(format!(
                "timed producer detected offloaded execution marker {variable}"
            )));
        }
    }
    Ok(())
}

fn reject_ambient_git_environment() -> Result<(), LocalPerfRunError> {
    if std::env::vars_os().any(|(name, _)| name.as_encoded_bytes().starts_with(b"GIT_")) {
        return Err(LocalPerfRunError::Invalid(
            "promotion producer rejects ambient GIT_* variables that can redirect repository identity"
                .to_owned(),
        ));
    }
    Ok(())
}

fn reject_ambient_process_injection() -> Result<(), LocalPerfRunError> {
    if let Some(name) = std::env::vars_os()
        .map(|(name, _)| name)
        .find(|name| ambient_variable_is_forbidden(name.as_encoded_bytes()))
    {
        return Err(LocalPerfRunError::Invalid(format!(
            "promotion producer rejects ambient build or workload override {}",
            bounded_diagnostic(&name.to_string_lossy())
        )));
    }
    Ok(())
}

fn ambient_variable_is_forbidden(name: &[u8]) -> bool {
    let exact = [
        b"AR".as_slice(),
        b"CC".as_slice(),
        b"CFLAGS".as_slice(),
        b"CPPFLAGS".as_slice(),
        b"CXX".as_slice(),
        b"CXXFLAGS".as_slice(),
        b"LDFLAGS".as_slice(),
        b"LD_LIBRARY_PATH".as_slice(),
        b"LD_PRELOAD".as_slice(),
        b"MALLOC_ARENA_MAX".as_slice(),
        b"MALLOC_CONF".as_slice(),
        b"RANLIB".as_slice(),
        b"RUSTC".as_slice(),
        b"RUSTC_WRAPPER".as_slice(),
        b"RUSTC_WORKSPACE_WRAPPER".as_slice(),
        b"RUSTFLAGS".as_slice(),
    ];
    exact.contains(&name)
        || (name.starts_with(b"QUILL_PERF_") && name != b"QUILL_PERF_HELD_PRODUCER_FD")
        || name.starts_with(b"RAYON_")
        || (name.starts_with(b"CARGO_") && name != b"CARGO_TARGET_DIR")
        || name.starts_with(b"RUSTUP_")
        || name.starts_with(b"DYLD_")
        || name.starts_with(b"JEMALLOC_")
}

fn controlled_environments(
    config: &LocalPerfRunConfig,
    selection: &ResolvedRunSelection,
    source: &CleanSourceSnapshot,
    target: &PinnedDirectory,
    artifact_dir: &Path,
) -> Result<ControlledEnvironments, LocalPerfRunError> {
    let registry = MachineClassRegistry::frozen()?;
    let run_profile = resolve_run_profile(config, &registry)?;
    let home = canonical_environment_directory("HOME", None)?;
    let cargo_home = canonical_environment_directory("CARGO_HOME", Some(&home.join(".cargo")))?;
    let rustup_home = canonical_environment_directory("RUSTUP_HOME", Some(&home.join(".rustup")))?;
    let repository_root = capture_repository_root()?;
    let cargo_config_candidates = cargo_config_candidates(&repository_root, &cargo_home);
    reject_cargo_config_candidates(&cargo_config_candidates)?;
    let path = required_unicode_environment("PATH")?;
    validate_absolute_path_list(&path)?;
    let rustup = resolve_path_tool("rustup", &path)?;
    let cargo = resolve_rustup_tool(&rustup, "cargo", &home, &rustup_home, &path)?;
    let rustc = resolve_rustup_tool(&rustup, "rustc", &home, &rustup_home, &path)?;
    let git = resolve_path_tool("git", &path)?;

    let mut build = BTreeMap::new();
    insert_environment(&mut build, "HOME", home.as_os_str());
    insert_environment(&mut build, "CARGO_HOME", cargo_home.as_os_str());
    insert_environment(&mut build, "RUSTUP_HOME", rustup_home.as_os_str());
    insert_environment(&mut build, "PATH", OsStr::new(&path));
    insert_environment(&mut build, "LANG", OsStr::new("C"));
    insert_environment(&mut build, "LC_ALL", OsStr::new("C"));
    insert_environment(&mut build, "CARGO_TARGET_DIR", target.path.as_os_str());
    insert_environment(&mut build, "CARGO_TERM_COLOR", OsStr::new("never"));
    bind_build_rustc(&mut build, &rustc);
    insert_environment(
        &mut build,
        "RUSTFLAGS",
        OsStr::new("-C force-frame-pointers=yes"),
    );
    insert_environment(&mut build, "RCH_DISABLE", OsStr::new("1"));
    insert_environment(&mut build, "RCH_CARGO_WRAPPER_BYPASS", OsStr::new("1"));

    let scratch = artifact_dir.join("scratch");
    let mut measurement = BTreeMap::new();
    insert_environment(&mut measurement, "HOME", home.as_os_str());
    insert_environment(&mut measurement, "CARGO_HOME", cargo_home.as_os_str());
    insert_environment(&mut measurement, "RUSTUP_HOME", rustup_home.as_os_str());
    insert_environment(&mut measurement, "PATH", OsStr::new(&path));
    insert_environment(&mut measurement, "LANG", OsStr::new("C"));
    insert_environment(&mut measurement, "LC_ALL", OsStr::new("C"));
    insert_environment(&mut measurement, "CARGO", cargo.path.as_os_str());
    insert_environment(
        &mut measurement,
        "CARGO_TARGET_DIR",
        target.path.as_os_str(),
    );
    insert_environment(&mut measurement, "QUILL_PERF_RUSTC", rustc.path.as_os_str());
    insert_environment(
        &mut measurement,
        "QUILL_PERF_TYPED_PRODUCER",
        OsStr::new("1"),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_GATE",
        OsStr::new(config.gate.label()),
    );
    apply_run_selection_environment(&mut measurement, selection);
    insert_environment(&mut measurement, "QUILL_PERF_SCALE", OsStr::new("full"));
    insert_environment(
        &mut measurement,
        "QUILL_PERF_BUILD_PROFILE",
        OsStr::new("release-perf"),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_GIT_REV",
        OsStr::new(&source.revision),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_RUN_ID",
        OsStr::new(&config.run_id),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_RUN_WINDOW",
        OsStr::new(&config.run_window),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_OUTPUT_DIR",
        artifact_dir.as_os_str(),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_SCRATCH_DIR",
        scratch.as_os_str(),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_RUNS",
        OsStr::new(&config.measurement_runs.to_string()),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_WARMUP_ROUNDS",
        OsStr::new(MEASUREMENT_WARMUP_ROUNDS),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_BOOTSTRAP_SEED",
        OsStr::new(MEASUREMENT_BOOTSTRAP_SEED),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_HARDWARE_CLASS",
        OsStr::new(config.profile.hardware_class_id().as_str()),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_EXECUTION_PROFILE",
        OsStr::new(config.profile.execution_profile_id().as_str()),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_EXECUTION_CAPACITY",
        OsStr::new(&run_profile.execution_capacity.to_string()),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_MAX_EXERCISED_CELL_WIDTH",
        OsStr::new(&run_profile.max_exercised_cell_width.to_string()),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_APPLICABILITY_PLAN_SCHEMA_VERSION",
        OsStr::new(&run_profile.applicability_plan.binding().schema_version),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_APPLICABILITY_PLAN_SHA256",
        OsStr::new(
            &run_profile
                .applicability_plan
                .binding()
                .applicability_plan_sha256,
        ),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_GATE_MATRIX_CONTRACT_SHA256",
        OsStr::new(
            &run_profile
                .applicability_plan
                .binding()
                .gate_matrix_contract_sha256,
        ),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_PROFILE_CONTRACT_SHA256",
        OsStr::new(
            &run_profile
                .applicability_plan
                .binding()
                .profile_contract_sha256,
        ),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_REGISTRY_SCHEMA_VERSION",
        OsStr::new(
            &run_profile
                .applicability_plan
                .binding()
                .registry_schema_version,
        ),
    );
    insert_environment(
        &mut measurement,
        "QUILL_PERF_REGISTRY_SHA256",
        OsStr::new(&run_profile.applicability_plan.binding().registry_sha256),
    );
    insert_environment(
        &mut measurement,
        "RAYON_NUM_THREADS",
        OsStr::new(&run_profile.execution_capacity.to_string()),
    );
    insert_environment(&mut measurement, "RCH_DISABLE", OsStr::new("1"));
    insert_environment(
        &mut measurement,
        "RCH_CARGO_WRAPPER_BYPASS",
        OsStr::new("1"),
    );
    insert_environment(&mut measurement, "TMPDIR", artifact_dir.as_os_str());

    let mut policy = BTreeMap::new();
    for (name, value) in &build {
        policy.insert(
            format!("build.{}", unicode_os(name, "build environment name")?),
            unicode_os(value, "build environment value")?,
        );
    }
    for (name, value) in &measurement {
        let name = unicode_os(name, "measurement environment name")?;
        let normalized = match name.as_str() {
            "QUILL_PERF_RUN_ID" => "<typed-run-id>".to_owned(),
            "QUILL_PERF_RUN_WINDOW" => "<typed-run-window>".to_owned(),
            "QUILL_PERF_GIT_REV" => "<measured-git-revision>".to_owned(),
            "QUILL_PERF_OUTPUT_DIR" | "TMPDIR" => "<held-artifact-directory>".to_owned(),
            "QUILL_PERF_SCRATCH_DIR" => "<held-artifact-directory>/scratch".to_owned(),
            _ => unicode_os(value, "measurement environment value")?,
        };
        policy.insert(format!("measurement.{name}"), normalized);
    }
    policy.insert(
        "policy.hardware_class_id".to_owned(),
        config.profile.hardware_class_id().as_str().to_owned(),
    );
    policy.insert(
        "policy.execution_profile_id".to_owned(),
        config.profile.execution_profile_id().as_str().to_owned(),
    );
    policy.insert(
        "policy.execution_capacity".to_owned(),
        run_profile.execution_capacity.to_string(),
    );
    policy.insert(
        "policy.max_exercised_cell_width".to_owned(),
        run_profile.max_exercised_cell_width.to_string(),
    );
    policy.insert(
        "policy.applicability_plan_sha256".to_owned(),
        run_profile
            .applicability_plan
            .binding()
            .applicability_plan_sha256
            .clone(),
    );
    policy.insert(
        "policy.gate_matrix_contract_sha256".to_owned(),
        run_profile
            .applicability_plan
            .binding()
            .gate_matrix_contract_sha256
            .clone(),
    );
    policy.insert(
        "policy.profile_contract_sha256".to_owned(),
        run_profile
            .applicability_plan
            .binding()
            .profile_contract_sha256
            .clone(),
    );
    policy.insert(
        "policy.fixture_selection".to_owned(),
        "<all-gate-cells>".to_owned(),
    );
    policy.insert(
        "tool.cargo_path".to_owned(),
        cargo.path.display().to_string(),
    );
    policy.insert("tool.cargo_sha256".to_owned(), cargo.sha256.clone());
    policy.insert(
        "tool.rustc_path".to_owned(),
        rustc.path.display().to_string(),
    );
    policy.insert("tool.rustc_sha256".to_owned(), rustc.sha256.clone());
    policy.insert("tool.git_path".to_owned(), git.path.display().to_string());
    policy.insert("tool.git_sha256".to_owned(), git.sha256.clone());
    policy.insert(
        "tool.rustup_path".to_owned(),
        rustup.path.display().to_string(),
    );
    policy.insert("tool.rustup_sha256".to_owned(), rustup.sha256.clone());
    for (index, candidate) in cargo_config_candidates.iter().enumerate() {
        policy.insert(
            format!("cargo_config_absence.{index}"),
            unicode_os(candidate.as_os_str(), "Cargo configuration candidate path")?,
        );
    }
    let policy_bytes = serde_json::to_vec(&policy)?;
    let policy_sha256 = sha256_hex(&policy_bytes);
    insert_environment(
        &mut measurement,
        "QUILL_PERF_ENVIRONMENT_SHA256",
        OsStr::new(&policy_sha256),
    );
    Ok(ControlledEnvironments {
        build,
        measurement,
        policy_sha256,
        policy_bytes,
        tools: vec![cargo, rustc, git, rustup],
        cargo_config_candidates,
    })
}

fn cargo_config_candidates(repository_root: &Path, cargo_home: &Path) -> Vec<PathBuf> {
    let mut candidates = BTreeSet::new();
    for ancestor in repository_root.ancestors() {
        candidates.insert(ancestor.join(".cargo").join("config"));
        candidates.insert(ancestor.join(".cargo").join("config.toml"));
    }
    candidates.insert(cargo_home.join("config"));
    candidates.insert(cargo_home.join("config.toml"));
    candidates.into_iter().collect()
}

fn reject_cargo_config_candidates(candidates: &[PathBuf]) -> Result<(), LocalPerfRunError> {
    reject_cargo_config_candidates_with(candidates, Path::try_exists)
}

fn reject_cargo_config_candidates_with(
    candidates: &[PathBuf],
    mut exists: impl FnMut(&Path) -> std::io::Result<bool>,
) -> Result<(), LocalPerfRunError> {
    for candidate in candidates {
        if exists(candidate)? {
            return Err(LocalPerfRunError::Invalid(
                "promotion build rejects Cargo configuration from the repository, an ancestor directory, or CARGO_HOME"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ResolvedTool {
    path: PathBuf,
    sha256: String,
}

fn verify_resolved_tools(tools: &[ResolvedTool]) -> Result<(), LocalPerfRunError> {
    for tool in tools {
        let canonical = fs::canonicalize(&tool.path)?;
        let metadata = fs::metadata(&canonical)?;
        if !metadata.is_file()
            || metadata.mode() & 0o111 == 0
            || sha256_open_file(&File::open(canonical)?)? != tool.sha256
        {
            return Err(LocalPerfRunError::Invalid(
                "controlled compiler or harness tool changed identity during the invocation"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

fn resolve_path_tool(name: &str, path: &str) -> Result<ResolvedTool, LocalPerfRunError> {
    for directory in std::env::split_paths(OsStr::new(path)) {
        let candidate = directory.join(name);
        let Ok(metadata) = fs::metadata(&candidate) else {
            continue;
        };
        if metadata.is_file() && metadata.mode() & 0o111 != 0 {
            let canonical = fs::canonicalize(&candidate)?;
            let file = File::open(&canonical)?;
            return Ok(ResolvedTool {
                path: canonical,
                sha256: sha256_open_file(&file)?,
            });
        }
    }
    Err(LocalPerfRunError::Invalid(format!(
        "controlled environment could not resolve required tool {name:?}"
    )))
}

fn resolve_rustup_tool(
    rustup: &ResolvedTool,
    name: &str,
    home: &Path,
    rustup_home: &Path,
    path: &str,
) -> Result<ResolvedTool, LocalPerfRunError> {
    let mut command = Command::new(&rustup.path);
    command
        .args(["which", name])
        .env_clear()
        .env("HOME", home)
        .env("RUSTUP_HOME", rustup_home)
        .env("PATH", path);
    let output = command_output_from(&mut command, "rustup which")?;
    let requested = PathBuf::from(output);
    if !requested.is_absolute() {
        return Err(LocalPerfRunError::Invalid(format!(
            "rustup returned a non-absolute {name} path"
        )));
    }
    let canonical = fs::canonicalize(requested)?;
    let metadata = fs::metadata(&canonical)?;
    if !metadata.is_file() || metadata.mode() & 0o111 == 0 {
        return Err(LocalPerfRunError::Invalid(format!(
            "rustup returned a non-executable {name} path"
        )));
    }
    let file = File::open(&canonical)?;
    Ok(ResolvedTool {
        path: canonical,
        sha256: sha256_open_file(&file)?,
    })
}

fn canonical_environment_directory(
    name: &str,
    default: Option<&Path>,
) -> Result<PathBuf, LocalPerfRunError> {
    let requested = match std::env::var_os(name) {
        Some(value) => PathBuf::from(value),
        None => default.map(Path::to_path_buf).ok_or_else(|| {
            LocalPerfRunError::Invalid(format!(
                "controlled environment requires absolute existing {name}"
            ))
        })?,
    };
    if !requested.is_absolute() {
        return Err(LocalPerfRunError::Invalid(format!(
            "controlled environment requires absolute {name}"
        )));
    }
    let canonical = fs::canonicalize(&requested)?;
    if canonical != requested || !fs::metadata(&canonical)?.is_dir() {
        return Err(LocalPerfRunError::Invalid(format!(
            "controlled environment requires canonical directory {name}"
        )));
    }
    Ok(canonical)
}

fn required_unicode_environment(name: &str) -> Result<String, LocalPerfRunError> {
    std::env::var(name).map_err(|_| {
        LocalPerfRunError::Invalid(format!("controlled environment requires a Unicode {name}"))
    })
}

fn validate_absolute_path_list(path: &str) -> Result<(), LocalPerfRunError> {
    if path.is_empty()
        || std::env::split_paths(OsStr::new(path))
            .any(|component| component.as_os_str().is_empty() || !component.is_absolute())
    {
        return Err(LocalPerfRunError::Invalid(
            "controlled PATH must contain only absolute nonempty components".to_owned(),
        ));
    }
    Ok(())
}

fn insert_environment(environment: &mut BTreeMap<OsString, OsString>, name: &str, value: &OsStr) {
    environment.insert(OsString::from(name), value.to_os_string());
}

fn apply_run_selection_environment(
    environment: &mut BTreeMap<OsString, OsString>,
    selection: &ResolvedRunSelection,
) {
    environment.remove(OsStr::new("QUILL_PERF_FIXTURE"));
    if let Some(fixture) = &selection.fixture {
        insert_environment(environment, "QUILL_PERF_FIXTURE", OsStr::new(fixture));
    }
}

fn bind_build_rustc(environment: &mut BTreeMap<OsString, OsString>, rustc: &ResolvedTool) {
    insert_environment(environment, "RUSTC", rustc.path.as_os_str());
}

fn unicode_os(value: &OsStr, field: &str) -> Result<String, LocalPerfRunError> {
    value.to_str().map(str::to_owned).ok_or_else(|| {
        LocalPerfRunError::Invalid(format!("controlled {field} must be valid Unicode"))
    })
}

fn capture_repository_root() -> Result<PathBuf, LocalPerfRunError> {
    let current = fs::canonicalize(std::env::current_dir()?)?;
    let discovered = fs::canonicalize(git_output(&["rev-parse", "--show-toplevel"])?)?;
    if current != discovered {
        return Err(LocalPerfRunError::Invalid(
            "promotion producer must execute from the exact canonical Git worktree root".to_owned(),
        ));
    }
    Ok(discovered)
}

fn capture_clean_source() -> Result<CleanSourceSnapshot, LocalPerfRunError> {
    capture_repository_root()?;
    let assume_unchanged = git_output(&["ls-files", "-v"])?;
    let skip_worktree = git_output(&["ls-files", "-t"])?;
    if assume_unchanged
        .lines()
        .any(|line| line.as_bytes().first().is_some_and(u8::is_ascii_lowercase))
        || skip_worktree.lines().any(|line| line.starts_with("S "))
    {
        return Err(LocalPerfRunError::Invalid(
            "promotion producer rejects assume-unchanged or skip-worktree index entries".to_owned(),
        ));
    }
    let revision = git_output(&["rev-parse", "HEAD"])?;
    if !is_git_revision(&revision) {
        return Err(LocalPerfRunError::Invalid(
            "promotion producer could not resolve a full lowercase Git revision".to_owned(),
        ));
    }
    let status = git_output(&["status", "--porcelain=v1", "--untracked-files=all"])?;
    if !status.trim().is_empty() {
        return Err(LocalPerfRunError::Invalid(
            "promotion producer requires a clean source tree".to_owned(),
        ));
    }
    let cargo_lock = fs::read("Cargo.lock")?;
    Ok(CleanSourceSnapshot {
        revision,
        cargo_lock_sha256: sha256_hex(&cargo_lock),
    })
}

fn git_output(arguments: &[&str]) -> Result<String, LocalPerfRunError> {
    let root = fs::canonicalize(std::env::current_dir()?)?;
    let path = required_unicode_environment("PATH")?;
    validate_absolute_path_list(&path)?;
    let git = resolve_path_tool("git", &path)?;
    let mut command = Command::new(&git.path);
    command.arg("-C").arg(root).args(arguments);
    remove_git_environment(&mut command);
    command_output_from(&mut command, "git")
}

fn remove_git_environment(command: &mut Command) {
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"GIT_") {
            command.env_remove(name);
        }
    }
}

fn capture_executing_producer(executable: &File) -> Result<RunnerProducer, LocalPerfRunError> {
    embedded_producer(&sha256_open_file(executable)?)
}

fn embedded_producer(executable_sha256: &str) -> Result<RunnerProducer, LocalPerfRunError> {
    let source_git_dirty = match EMBEDDED_PRODUCER_GIT_DIRTY {
        "true" => true,
        "false" => false,
        other => {
            return Err(LocalPerfRunError::Invalid(format!(
                "typed producer embeds malformed dirty-state marker {other:?}"
            )));
        }
    };
    Ok(RunnerProducer {
        contract_version: EMBEDDED_PRODUCER_CONTRACT_VERSION.to_owned(),
        source_git_revision: EMBEDDED_PRODUCER_GIT_REVISION.to_owned(),
        source_git_dirty,
        cargo_lock_sha256: EMBEDDED_PRODUCER_CARGO_LOCK_SHA256.to_owned(),
        executable_sha256: executable_sha256.to_owned(),
    })
}

fn capture_validated_producer(
    source: &CleanSourceSnapshot,
) -> Result<ExecutingProducer, LocalPerfRunError> {
    let executable = open_launcher_held_producer()?;
    let producer = capture_executing_producer(&executable)?;
    validate_producer_against_source(&producer, source)?;
    let current_image = open_executing_image()?;
    validate_expected_producer_executable(&producer, &sha256_open_file(&current_image)?)?;
    Ok(ExecutingProducer {
        receipt: producer,
        executable,
    })
}

fn verify_executing_producer(
    expected: &ExecutingProducer,
    source: &CleanSourceSnapshot,
) -> Result<RunnerProducer, LocalPerfRunError> {
    let observed = capture_executing_producer(&expected.executable)?;
    validate_producer_against_source(&observed, source)?;
    if observed != expected.receipt {
        return Err(LocalPerfRunError::Invalid(
            "held typed-producer executable bytes drifted during the invocation".to_owned(),
        ));
    }
    Ok(observed)
}

fn open_launcher_held_producer() -> Result<File, LocalPerfRunError> {
    if std::env::var("QUILL_PERF_HELD_PRODUCER_FD").as_deref() != Ok("9") {
        return Err(LocalPerfRunError::Invalid(
            "typed producer requires launcher-held executable descriptor 9".to_owned(),
        ));
    }
    let path = match std::env::consts::OS {
        "linux" => PathBuf::from("/proc/self/fd/9"),
        "macos" => {
            return Err(LocalPerfRunError::Invalid(
                "macOS held-producer execution is not promotion-admissible without a supported O_EXEC or loaded-image attestation mechanism"
                    .to_owned(),
            ));
        }
        other => {
            return Err(LocalPerfRunError::Invalid(format!(
                "unsupported held-producer descriptor OS {other:?}"
            )));
        }
    };
    File::open(path).map_err(LocalPerfRunError::from)
}

fn open_executing_image() -> Result<File, LocalPerfRunError> {
    let path = match std::env::consts::OS {
        "linux" => PathBuf::from("/proc/self/exe"),
        "macos" => std::env::current_exe()?,
        other => {
            return Err(LocalPerfRunError::Invalid(format!(
                "unsupported typed-producer executable capture OS {other:?}"
            )));
        }
    };
    File::open(path).map_err(LocalPerfRunError::from)
}

fn sha256_open_file(file: &File) -> Result<String, LocalPerfRunError> {
    let mut reader = file.try_clone()?;
    reader.rewind()?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

fn validate_producer_against_source(
    producer: &RunnerProducer,
    source: &CleanSourceSnapshot,
) -> Result<(), LocalPerfRunError> {
    if producer.contract_version != LOCAL_PERF_PRODUCER_CONTRACT_VERSION
        || producer.source_git_dirty
        || !is_git_revision(&producer.source_git_revision)
        || producer.source_git_revision != source.revision
        || !is_sha256(&producer.cargo_lock_sha256)
        || producer.cargo_lock_sha256 != source.cargo_lock_sha256
        || !is_sha256(&producer.executable_sha256)
    {
        return Err(LocalPerfRunError::Invalid(
            "executing typed producer does not match the live clean Git revision and Cargo.lock"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_expected_producer_executable(
    producer: &RunnerProducer,
    expected_executable_sha256: &str,
) -> Result<(), LocalPerfRunError> {
    if !is_sha256(expected_executable_sha256)
        || producer.executable_sha256 != expected_executable_sha256
    {
        return Err(LocalPerfRunError::Invalid(
            "runtime producer handle differs from the launcher-held executable digest".to_owned(),
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn prepare_benchmark(
    source_before: &CleanSourceSnapshot,
    producer_before: &ExecutingProducer,
    target: &PinnedDirectory,
    environments: ControlledEnvironments,
) -> Result<CapturedBuild, LocalPerfRunError> {
    verify_pinned_directory(target)?;
    verify_resolved_tools(&environments.tools)?;
    reject_cargo_config_candidates(&environments.cargo_config_candidates)?;
    let cargo_path = environments
        .measurement
        .get(OsStr::new("CARGO"))
        .ok_or_else(|| {
            LocalPerfRunError::Invalid(
                "controlled measurement environment omitted the Cargo tool".to_owned(),
            )
        })?;
    let mut cargo = Command::new(cargo_path);
    cargo
        .args([
            "build",
            "--frozen",
            "--profile",
            "release-perf",
            "-p",
            "frankensearch-quill-gauntlet",
            "--features",
            "perf-harness",
            "--bench",
            "perf_matrix",
            "--message-format=json-render-diagnostics",
        ])
        .env_clear()
        .envs(&environments.build);
    let output = cargo.output()?;
    if !output.status.success() {
        return Err(LocalPerfRunError::Invalid(format!(
            "typed benchmark build failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    reject_cargo_config_candidates(&environments.cargo_config_candidates)?;
    verify_resolved_tools(&environments.tools)?;

    let mut executables = BTreeSet::new();
    for line in output.stdout.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_slice::<serde_json::Value>(line)?;
        let is_benchmark = value.get("reason").and_then(serde_json::Value::as_str)
            == Some("compiler-artifact")
            && value
                .pointer("/target/name")
                .and_then(serde_json::Value::as_str)
                == Some("perf_matrix")
            && value
                .pointer("/target/kind")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bench")));
        if is_benchmark
            && let Some(executable) = value.get("executable").and_then(serde_json::Value::as_str)
        {
            executables.insert(executable.to_owned());
        }
    }
    let executable = match executables.into_iter().collect::<Vec<_>>().as_slice() {
        [path] => {
            let reported = PathBuf::from(path);
            let canonical = fs::canonicalize(&reported)?;
            if reported != canonical {
                return Err(LocalPerfRunError::Invalid(
                    "Cargo-reported benchmark executable contains a symbolic-link or lexical alias"
                        .to_owned(),
                ));
            }
            canonical
        }
        paths => {
            return Err(LocalPerfRunError::Invalid(format!(
                "Cargo reported {} distinct perf_matrix benchmark executables, expected exactly one",
                paths.len()
            )));
        }
    };
    if !executable.starts_with(&target.path) || executable == target.path {
        return Err(LocalPerfRunError::Invalid(
            "Cargo-reported benchmark executable is outside the pinned CARGO_TARGET_DIR".to_owned(),
        ));
    }
    let executable_handle = File::from(
        open(
            &executable,
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(std::io::Error::from)?,
    );
    let executable_identity = checked_benchmark_identity(&executable_handle)?;
    verify_benchmark_path(&executable, &executable_identity, target)?;
    let source_after = capture_clean_source()?;
    let producer_after = verify_executing_producer(producer_before, &source_after)?;
    if source_before != &source_after {
        return Err(LocalPerfRunError::Invalid(
            "clean source or executing producer identity changed during the typed benchmark build"
                .to_owned(),
        ));
    }
    let command = vec![
        executable.clone().into_os_string(),
        OsString::from("--bench"),
        OsString::from("--noplot"),
    ];
    let command_sha256 =
        command_sha256_from_argv(command.iter().map(|argument| argument.as_encoded_bytes()));
    Ok(CapturedBuild {
        receipt: RunnerBuild {
            git_revision: source_after.revision.clone(),
            git_dirty: false,
            worktree_state_sha256: None,
            cargo_lock_sha256: source_after.cargo_lock_sha256,
            executable_sha256: sha256_open_file(&executable_handle)?,
            command_sha256,
            environment_sha256: environments.policy_sha256,
            producer: producer_after,
        },
        revision: source_after.revision,
        command,
        measurement_environment: environments.measurement,
        environment_policy_bytes: environments.policy_bytes,
        tool_identities: environments.tools,
        cargo_config_candidates: environments.cargo_config_candidates,
        executable_path: executable,
        executable: executable_handle,
        executable_identity,
    })
}

fn verify_prepared_build(
    expected: &CapturedBuild,
    producer_handle: &ExecutingProducer,
    target: &PinnedDirectory,
) -> Result<(), LocalPerfRunError> {
    verify_pinned_directory(target)?;
    verify_resolved_tools(&expected.tool_identities)?;
    reject_cargo_config_candidates(&expected.cargo_config_candidates)?;
    verify_benchmark_path(
        &expected.executable_path,
        &expected.executable_identity,
        target,
    )?;
    let source = capture_clean_source()?;
    let producer = verify_executing_producer(producer_handle, &source)?;
    let command_sha256 = command_sha256_from_argv(
        expected
            .command
            .iter()
            .map(|argument| argument.as_encoded_bytes()),
    );
    let observed = RunnerBuild {
        git_revision: source.revision.clone(),
        git_dirty: false,
        worktree_state_sha256: None,
        cargo_lock_sha256: source.cargo_lock_sha256,
        executable_sha256: sha256_open_file(&expected.executable)?,
        command_sha256,
        environment_sha256: expected.receipt.environment_sha256.clone(),
        producer,
    };
    let measurement_environment_sha256 = expected
        .measurement_environment
        .get(OsStr::new("QUILL_PERF_ENVIRONMENT_SHA256"))
        .and_then(|value| value.to_str());
    if observed != expected.receipt
        || source.revision != expected.revision
        || expected.command.first().map(PathBuf::from).as_ref() != Some(&expected.executable_path)
        || measurement_environment_sha256 != Some(expected.receipt.environment_sha256.as_str())
        || !is_sha256(&expected.receipt.environment_sha256)
    {
        return Err(LocalPerfRunError::Invalid(
            "source, held executable, or fixed benchmark argv drifted after the typed Cargo build"
                .to_owned(),
        ));
    }
    Ok(())
}

fn checked_benchmark_identity(handle: &File) -> Result<FileIdentity, LocalPerfRunError> {
    let metadata = handle.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != geteuid().as_raw()
        || metadata.mode() & 0o111 == 0
    {
        return Err(LocalPerfRunError::Invalid(
            "benchmark image must be an effective-user-owned executable regular single-link file"
                .to_owned(),
        ));
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

fn verify_benchmark_path(
    path: &Path,
    expected: &FileIdentity,
    target: &PinnedDirectory,
) -> Result<(), LocalPerfRunError> {
    let metadata = fs::symlink_metadata(path)?;
    let observed = FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    if observed != *expected
        || metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != geteuid().as_raw()
        || metadata.mode() & 0o111 == 0
        || fs::canonicalize(path)? != path
        || !path.starts_with(&target.path)
        || path == target.path
    {
        return Err(LocalPerfRunError::Invalid(
            "Cargo-reported benchmark path changed or escaped the pinned target root".to_owned(),
        ));
    }
    Ok(())
}

fn configure_benchmark_child(child: &mut Command, environment: &BTreeMap<OsString, OsString>) {
    child.env_clear().envs(environment).process_group(0);
}

fn capture_platform(config: &LocalPerfRunConfig) -> Result<PlatformCapture, LocalPerfRunError> {
    match std::env::consts::OS {
        "linux" => capture_linux(config),
        "macos" => capture_macos(config),
        other => Err(LocalPerfRunError::Invalid(format!(
            "unsupported timed-producer OS {other:?}"
        ))),
    }
}

fn durability_for_run(config: &LocalPerfRunConfig) -> Result<RunnerDurability, LocalPerfRunError> {
    if matches!(config.gate, PerfGate::Qg3 | PerfGate::Qg4 | PerfGate::Qg5) {
        return Err(LocalPerfRunError::Invalid(format!(
            "{} is durability-adjacent and requires an observed symmetric treatment witness",
            config.gate
        )));
    }
    Ok(RunnerDurability {
        adjacent: false,
        control_treatment: "not-applicable".to_owned(),
        candidate_treatment: "not-applicable".to_owned(),
        symmetric: true,
    })
}

fn stable_lease_id(hardware_class_id: HardwareClassId) -> Result<&'static str, LocalPerfRunError> {
    match hardware_class_id {
        HardwareClassId::TrjZen35995wx => Ok("trj-zen3-exclusive"),
        HardwareClassId::M4Macos => Ok("m4-macos-exclusive"),
        HardwareClassId::X86VpsOvh => Ok("x86-vps-ovh-exclusive"),
        HardwareClassId::M5Macos => Err(LocalPerfRunError::Invalid(
            "M5 has no registered host-global lease".to_owned(),
        )),
    }
}

fn stable_lease_path(hardware_class_id: HardwareClassId) -> Result<PathBuf, LocalPerfRunError> {
    stable_lease_id(hardware_class_id)?;
    Ok(PathBuf::from(
        "/tmp/frankensearch-perf-host-global-exclusive.lock",
    ))
}

fn validate_canonical_lease_parent(lease_path: &Path) -> Result<(), LocalPerfRunError> {
    let parent = lease_path.parent().ok_or_else(|| {
        LocalPerfRunError::Invalid("canonical host-global lease has no parent".to_owned())
    })?;
    if parent != Path::new("/tmp") {
        return Err(LocalPerfRunError::Invalid(
            "canonical host-global lease must remain directly under /tmp".to_owned(),
        ));
    }
    let resolved = fs::canonicalize(parent)?;
    let metadata = fs::metadata(&resolved)?;
    let mode = metadata.mode();
    if !metadata.is_dir() || metadata.uid() != 0 || mode & 0o1000 == 0 || mode & 0o002 == 0 {
        return Err(LocalPerfRunError::Invalid(format!(
            "canonical lease parent {} must resolve to a root-owned sticky world-writable directory",
            resolved.display()
        )));
    }
    Ok(())
}

fn acquire_family_lease(
    lease_path: &Path,
) -> Result<(OwnedFd, LeaseFileIdentity), LocalPerfRunError> {
    let lease_file = open(
        lease_path,
        OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(std::io::Error::from)?;
    flock(&lease_file, FlockOperation::NonBlockingLockExclusive).map_err(std::io::Error::from)?;
    let descriptor_flags = fcntl_getfd(&lease_file).map_err(std::io::Error::from)?;
    fcntl_setfd(&lease_file, descriptor_flags - FdFlags::CLOEXEC).map_err(std::io::Error::from)?;
    if fcntl_getfd(&lease_file)
        .map_err(std::io::Error::from)?
        .contains(FdFlags::CLOEXEC)
    {
        return Err(LocalPerfRunError::Invalid(
            "host-global lease descriptor remained close-on-exec".to_owned(),
        ));
    }
    let identity = checked_lease_identity(&lease_file)?;
    verify_family_lease_path(lease_path, &identity)?;
    Ok((lease_file, identity))
}

fn checked_lease_identity(lease_file: &impl AsFd) -> Result<LeaseFileIdentity, LocalPerfRunError> {
    let stat = fstat(lease_file).map_err(std::io::Error::from)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_uid != geteuid().as_raw()
        || stat.st_mode & 0o777 != 0o600
    {
        return Err(LocalPerfRunError::Invalid(
            "canonical host-global lease must be an effective-user-owned 0600 regular single-link file"
                .to_owned(),
        ));
    }
    Ok(LeaseFileIdentity {
        device: stat.st_dev.to_string(),
        inode: stat.st_ino.to_string(),
    })
}

fn verify_family_lease_path(
    lease_path: &Path,
    expected: &LeaseFileIdentity,
) -> Result<(), LocalPerfRunError> {
    let observed_file = open(
        lease_path,
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)?;
    let observed = checked_lease_identity(&observed_file)?;
    if &observed != expected {
        return Err(LocalPerfRunError::Invalid(
            "canonical host-global lease path changed device or inode during the run".to_owned(),
        ));
    }
    Ok(())
}

fn capture_linux(config: &LocalPerfRunConfig) -> Result<PlatformCapture, LocalPerfRunError> {
    if config.profile.hardware_class_id() != HardwareClassId::TrjZen35995wx {
        return Err(LocalPerfRunError::Invalid(
            "promotion-grade Linux producer requires trj-zen3-5995wx".to_owned(),
        ));
    }
    let run_profile = resolve_run_profile(config, &MachineClassRegistry::frozen()?)?;
    let threads_per_core = match config.profile.execution_profile_id() {
        ExecutionProfileId::Physical64 => 1,
        ExecutionProfileId::Smt2_128 => 2,
        other => {
            return Err(LocalPerfRunError::Invalid(format!(
                "unsupported Threadripper execution profile {:?}",
                other.as_str()
            )));
        }
    };
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    let records = parse_cpuinfo(&cpuinfo);
    let first = records.first().ok_or_else(|| {
        LocalPerfRunError::Invalid("/proc/cpuinfo contains no processor record".to_owned())
    })?;
    let field = |name: &str| {
        first
            .get(name)
            .cloned()
            .ok_or_else(|| LocalPerfRunError::Invalid(format!("/proc/cpuinfo is missing {name:?}")))
    };
    let logical_cpus = u64::try_from(records.len())
        .map_err(|_| LocalPerfRunError::Invalid("logical CPU count does not fit u64".to_owned()))?;
    let physical = records
        .iter()
        .map(|record| {
            Ok::<_, LocalPerfRunError>((
                record
                    .get("physical id")
                    .ok_or_else(|| {
                        LocalPerfRunError::Invalid(
                            "/proc/cpuinfo is missing physical id".to_owned(),
                        )
                    })?
                    .clone(),
                record
                    .get("core id")
                    .ok_or_else(|| {
                        LocalPerfRunError::Invalid("/proc/cpuinfo is missing core id".to_owned())
                    })?
                    .clone(),
            ))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let memory_bytes = parse_linux_memory_bytes()?;
    let page_size_bytes = command_output("getconf", &["PAGESIZE"])?
        .parse::<u64>()
        .map_err(|error| LocalPerfRunError::Invalid(format!("invalid page size: {error}")))?;
    let numa_nodes = count_linux_numa_nodes()?;
    let topology_sha256 = linux_topology_sha256(&records);
    let hardware = RunnerHardware {
        os: "linux".to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        cpu_vendor: field("vendor_id")?,
        cpu_family: Some(parse_u64_field(&field("cpu family")?, "cpu family")?),
        cpu_model: Some(parse_u64_field(&field("model")?, "cpu model")?),
        cpu_stepping: Some(parse_u64_field(&field("stepping")?, "cpu stepping")?),
        cpu_model_name: field("model name")?,
        physical_cores: u64::try_from(physical.len()).map_err(|_| {
            LocalPerfRunError::Invalid("physical core count does not fit u64".to_owned())
        })?,
        logical_cpus,
        numa_nodes,
        memory_bytes,
        page_size_bytes,
        performance_cores: None,
        efficiency_cores: None,
        runtime_detected_isa: crate::perf::runtime_detected_isa(),
        topology_sha256,
        fingerprint_sha256: String::new(),
    };
    let observed_logical_cpu_ids = linux_allowed_cpu_ids()?;
    let effective_physical_core_ids = linux_physical_core_ids(&observed_logical_cpu_ids)?;
    validate_linux_numa_binding(&observed_logical_cpu_ids)?;
    let governor = linux_governor(&observed_logical_cpu_ids)?;
    let thermal_pressure = linux_thermal_pressure()?;
    if thermal_pressure {
        return Err(LocalPerfRunError::Invalid(
            "Threadripper CPU thermal sensor is at or above its advertised limit".to_owned(),
        ));
    }
    let request = RunnerExecutionRequest {
        capacity_semantics: run_profile.capacity_semantics,
        execution_capacity: run_profile.execution_capacity,
        max_exercised_cell_width: run_profile.max_exercised_cell_width,
        requested_logical_cpu_ids: observed_logical_cpu_ids.clone(),
        requested_physical_core_width: Some(64),
        requested_worker_pool_width: run_profile.execution_capacity,
        requested_qos: "not-applicable".to_owned(),
    };
    let snapshot = RunnerExecutionSnapshot {
        observed_logical_cpu_ids,
        effective_physical_core_ids,
        cpu_assignment_observability: "affinity-enforced".to_owned(),
        effective_cpuset_sha256: String::new(),
        threads_per_core,
        smt_state: if threads_per_core == 2 { "on" } else { "off" }.to_owned(),
        numa_node_ids: vec![0],
        numa_policy: "bind:0".to_owned(),
        governor,
        thermal_pressure,
        exclusive_lease: true,
        exclusive_lease_id: stable_lease_id(config.profile.hardware_class_id())?.to_owned(),
        local_execution: true,
        observed_hardware_fingerprint_sha256: String::new(),
        snapshot_sha256: String::new(),
    };
    Ok(PlatformCapture {
        hardware,
        request,
        snapshot,
    })
}

fn capture_macos(config: &LocalPerfRunConfig) -> Result<PlatformCapture, LocalPerfRunError> {
    if config.profile.hardware_class_id() != HardwareClassId::M4Macos
        || config.profile.execution_profile_id() != ExecutionProfileId::Scheduler10
    {
        return Err(LocalPerfRunError::Invalid(
            "macOS producer currently recognizes only m4-macos.scheduler-10".to_owned(),
        ));
    }
    let run_profile = resolve_run_profile(config, &MachineClassRegistry::frozen()?)?;
    let sysctl = |name: &str| command_output("sysctl", &["-n", name]);
    let hardware = RunnerHardware {
        os: "macos".to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        cpu_vendor: "Apple".to_owned(),
        cpu_family: None,
        cpu_model: None,
        cpu_stepping: None,
        cpu_model_name: sysctl("machdep.cpu.brand_string")?,
        physical_cores: parse_u64_field(&sysctl("hw.physicalcpu")?, "hw.physicalcpu")?,
        logical_cpus: parse_u64_field(&sysctl("hw.logicalcpu")?, "hw.logicalcpu")?,
        numa_nodes: 1,
        memory_bytes: parse_u64_field(&sysctl("hw.memsize")?, "hw.memsize")?,
        page_size_bytes: parse_u64_field(&sysctl("hw.pagesize")?, "hw.pagesize")?,
        performance_cores: Some(parse_u64_field(
            &sysctl("hw.perflevel0.physicalcpu")?,
            "hw.perflevel0.physicalcpu",
        )?),
        efficiency_cores: Some(parse_u64_field(
            &sysctl("hw.perflevel1.physicalcpu")?,
            "hw.perflevel1.physicalcpu",
        )?),
        runtime_detected_isa: crate::perf::runtime_detected_isa(),
        topology_sha256: macos_topology_sha256()?,
        fingerprint_sha256: String::new(),
    };
    let thermal = command_output("pmset", &["-g", "therm"])?;
    let thermal_pressure = !thermal.to_ascii_lowercase().contains("no thermal warning");
    if thermal_pressure {
        return Err(LocalPerfRunError::Invalid(format!(
            "M4 thermal state is not explicitly clear: {thermal:?}"
        )));
    }
    let request = RunnerExecutionRequest {
        capacity_semantics: run_profile.capacity_semantics,
        execution_capacity: run_profile.execution_capacity,
        max_exercised_cell_width: run_profile.max_exercised_cell_width,
        requested_logical_cpu_ids: Vec::new(),
        requested_physical_core_width: None,
        requested_worker_pool_width: run_profile.execution_capacity,
        requested_qos: "inherit-process-default".to_owned(),
    };
    let snapshot = RunnerExecutionSnapshot {
        observed_logical_cpu_ids: Vec::new(),
        effective_physical_core_ids: Vec::new(),
        cpu_assignment_observability: "unavailable".to_owned(),
        effective_cpuset_sha256: String::new(),
        threads_per_core: 1,
        smt_state: "not-applicable".to_owned(),
        numa_node_ids: vec![0],
        numa_policy: "system".to_owned(),
        governor: "not-applicable".to_owned(),
        thermal_pressure: false,
        exclusive_lease: true,
        exclusive_lease_id: stable_lease_id(config.profile.hardware_class_id())?.to_owned(),
        local_execution: true,
        observed_hardware_fingerprint_sha256: String::new(),
        snapshot_sha256: String::new(),
    };
    Ok(PlatformCapture {
        hardware,
        request,
        snapshot,
    })
}

fn read_canonical_threshold(bytes: &[u8]) -> Result<PerfGateArtifact, LocalPerfRunError> {
    PerfGateArtifact::from_verified_measured_slice(bytes)
        .map_err(|error| LocalPerfRunError::Invalid(error.to_string()))
}

fn read_canonical_evidence(bytes: &[u8]) -> Result<PerfEvidenceArtifact, LocalPerfRunError> {
    let artifact = PerfEvidenceArtifact::from_verified_slice(bytes)?;
    if serde_json::to_vec_pretty(&artifact)? != bytes {
        return Err(LocalPerfRunError::Invalid(
            "evidence artifact is not exact canonical pretty JSON".to_owned(),
        ));
    }
    Ok(artifact)
}

fn validate_child_artifacts(
    config: &LocalPerfRunConfig,
    run_profile: &RunProfileContract,
    run_selection: &ResolvedRunSelection,
    build: &CapturedBuild,
    platform: &PlatformCapture,
    threshold: &PerfGateArtifact,
    evidence: &PerfEvidenceArtifact,
) -> Result<(), LocalPerfRunError> {
    if threshold.gate != config.gate
        || evidence.gate != config.gate
        || threshold.run_id != config.run_id
        || evidence.provenance.run_id != config.run_id
        || threshold.run_window != config.run_window
        || evidence.provenance.run_window != config.run_window
        || threshold.git_rev != build.revision
        || evidence.provenance.build.git_revision != build.revision
        || threshold.bench_elf_sha256 != build.receipt.executable_sha256
        || evidence.provenance.build.executable_sha256 != build.receipt.executable_sha256
        || evidence.provenance.build.command_sha256 != build.receipt.command_sha256
        || evidence.provenance.build.environment_sha256.as_deref()
            != Some(build.receipt.environment_sha256.as_str())
        || evidence.provenance.build.cargo_lock_sha256.as_deref()
            != Some(build.receipt.cargo_lock_sha256.as_str())
        || threshold.applicability_plan.as_ref() != Some(run_profile.applicability_plan.binding())
        || evidence.applicability_plan != *run_profile.applicability_plan.binding()
        || threshold.manifest_sha256
            != run_profile
                .applicability_plan
                .binding()
                .normalized_perf_manifest_sha256
        || evidence.provenance.manifest_sha256
            != run_profile
                .applicability_plan
                .binding()
                .normalized_perf_manifest_sha256
        || evidence_cell_ids(evidence) != run_selection.selected_cell_ids
        || threshold.execution.as_ref() != Some(&evidence.provenance.machine.execution)
        || evidence.provenance.machine.os != platform.hardware.os
        || evidence.provenance.machine.arch != platform.hardware.arch
        || evidence.provenance.machine.execution.physical_cores
            != usize::try_from(platform.hardware.physical_cores).unwrap_or(usize::MAX)
        || evidence.provenance.machine.execution.logical_threads
            != usize::try_from(platform.hardware.logical_cpus).unwrap_or(usize::MAX)
        || evidence.provenance.machine.execution.runtime_detected_isa
            != platform.hardware.runtime_detected_isa
    {
        return Err(LocalPerfRunError::Invalid(
            "child threshold/evidence identity does not equal producer-captured facts".to_owned(),
        ));
    }
    Ok(())
}

fn parse_cpuinfo(contents: &str) -> Vec<BTreeMap<String, String>> {
    contents
        .split("\n\n")
        .filter_map(|block| {
            let fields = block
                .lines()
                .filter_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    Some((name.trim().to_owned(), value.trim().to_owned()))
                })
                .collect::<BTreeMap<_, _>>();
            fields.contains_key("processor").then_some(fields)
        })
        .collect()
}

fn parse_linux_memory_bytes() -> Result<u64, LocalPerfRunError> {
    let contents = fs::read_to_string("/proc/meminfo")?;
    let kib = contents
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_ascii_whitespace().next())
        .ok_or_else(|| LocalPerfRunError::Invalid("/proc/meminfo lacks MemTotal".to_owned()))?
        .parse::<u64>()
        .map_err(|error| LocalPerfRunError::Invalid(format!("invalid MemTotal: {error}")))?;
    kib.checked_mul(1024)
        .ok_or_else(|| LocalPerfRunError::Invalid("MemTotal byte conversion overflowed".to_owned()))
}

fn count_linux_numa_nodes() -> Result<u64, LocalPerfRunError> {
    let count = fs::read_dir("/sys/devices/system/node")?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .strip_prefix("node")
                .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
        })
        .count();
    u64::try_from(count)
        .map_err(|_| LocalPerfRunError::Invalid("NUMA node count does not fit u64".to_owned()))
}

fn linux_topology_sha256(records: &[BTreeMap<String, String>]) -> String {
    let canonical = records
        .iter()
        .map(|record| {
            [
                record.get("processor").map_or("", String::as_str),
                record.get("physical id").map_or("", String::as_str),
                record.get("core id").map_or("", String::as_str),
            ]
            .join(":")
        })
        .collect::<Vec<_>>()
        .join("\n");
    sha256_hex(canonical.as_bytes())
}

fn linux_allowed_cpu_ids() -> Result<Vec<u64>, LocalPerfRunError> {
    let status = fs::read_to_string("/proc/self/status")?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(str::trim)
        .ok_or_else(|| {
            LocalPerfRunError::Invalid("/proc/self/status lacks Cpus_allowed_list".to_owned())
        })?;
    parse_cpu_list(list)
}

fn parse_cpu_list(value: &str) -> Result<Vec<u64>, LocalPerfRunError> {
    let mut ids = Vec::new();
    for component in value.split(',') {
        let component = component.trim();
        let (first, last) = component
            .split_once('-')
            .map_or((component, component), |(first, last)| (first, last));
        let first = parse_u64_field(first, "CPU range start")?;
        let last = parse_u64_field(last, "CPU range end")?;
        if last < first {
            return Err(LocalPerfRunError::Invalid(format!(
                "CPU range {component:?} is reversed"
            )));
        }
        ids.extend(first..=last);
    }
    let unique = ids.iter().copied().collect::<BTreeSet<_>>();
    if ids.is_empty() || unique.len() != ids.len() {
        return Err(LocalPerfRunError::Invalid(
            "CPU affinity list is empty or duplicated".to_owned(),
        ));
    }
    Ok(ids)
}

fn linux_physical_core_ids(ids: &[u64]) -> Result<Vec<String>, LocalPerfRunError> {
    ids.iter()
        .map(|id| {
            let base = PathBuf::from(format!("/sys/devices/system/cpu/cpu{id}/topology"));
            let package = fs::read_to_string(base.join("physical_package_id"))?;
            let core = fs::read_to_string(base.join("core_id"))?;
            Ok(format!("{}:{}", package.trim(), core.trim()))
        })
        .collect()
}

fn validate_linux_numa_binding(ids: &[u64]) -> Result<(), LocalPerfRunError> {
    let show = command_output("numactl", &["--show"])?;
    let policy_bind = show.lines().any(|line| line.trim() == "policy: bind");
    let mem_bind_zero = show
        .lines()
        .find_map(|line| line.trim().strip_prefix("membind:"))
        .is_some_and(|value| value.split_ascii_whitespace().eq(["0"]));
    let allowed = ids.iter().copied().collect::<BTreeSet<_>>();
    let physical_bind = show
        .lines()
        .find_map(|line| line.trim().strip_prefix("physcpubind:"))
        .map(|value| {
            value
                .split_ascii_whitespace()
                .filter_map(|cpu| cpu.parse::<u64>().ok())
                .collect::<BTreeSet<_>>()
        });
    if !policy_bind || !mem_bind_zero || physical_bind.as_ref() != Some(&allowed) {
        return Err(LocalPerfRunError::Invalid(
            "producer must inherit exact numactl policy bind, membind 0, and selected CPUs"
                .to_owned(),
        ));
    }
    Ok(())
}

fn linux_governor(ids: &[u64]) -> Result<String, LocalPerfRunError> {
    for id in ids {
        let path = format!("/sys/devices/system/cpu/cpu{id}/cpufreq/scaling_governor");
        let governor = fs::read_to_string(&path)?;
        if governor.trim() != "performance" {
            return Err(LocalPerfRunError::Invalid(format!(
                "CPU {id} governor is {:?}, expected performance",
                governor.trim()
            )));
        }
    }
    Ok("performance".to_owned())
}

fn linux_thermal_pressure() -> Result<bool, LocalPerfRunError> {
    let mut hwmon_paths = fs::read_dir("/sys/class/hwmon")?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    hwmon_paths.sort();
    let mut samples = Vec::new();
    for hwmon in hwmon_paths {
        let name = fs::read_to_string(hwmon.join("name"))
            .map(|value| value.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if !name.contains("k10temp") && !name.contains("zenpower") {
            continue;
        }
        let mut inputs = fs::read_dir(&hwmon)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        inputs.sort();
        for input_path in inputs {
            let Some(file_name) = input_path.file_name().and_then(OsStr::to_str) else {
                continue;
            };
            let Some(prefix) = file_name.strip_suffix("_input") else {
                continue;
            };
            if !prefix.starts_with("temp") {
                continue;
            }
            let input = read_millidegrees(&input_path)?;
            let mut limits = Vec::new();
            for suffix in ["max", "crit"] {
                let limit_path = hwmon.join(format!("{prefix}_{suffix}"));
                match fs::read_to_string(&limit_path) {
                    Ok(value) => limits.push(parse_millidegrees(&value, &limit_path)?),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
            if !limits.is_empty() {
                samples.push((input, limits));
            }
        }
    }
    thermal_pressure_from_samples(&samples)
}

fn read_millidegrees(path: &Path) -> Result<i64, LocalPerfRunError> {
    let value = fs::read_to_string(path)?;
    parse_millidegrees(&value, path)
}

fn parse_millidegrees(value: &str, path: &Path) -> Result<i64, LocalPerfRunError> {
    value.trim().parse::<i64>().map_err(|error| {
        LocalPerfRunError::Invalid(format!(
            "CPU thermal sensor {} is malformed: {error}",
            path.display()
        ))
    })
}

fn thermal_pressure_from_samples(samples: &[(i64, Vec<i64>)]) -> Result<bool, LocalPerfRunError> {
    if samples.is_empty() {
        return Err(LocalPerfRunError::Invalid(
            "Threadripper promotion requires an observable k10temp or zenpower CPU thermal sensor with max/crit limits"
                .to_owned(),
        ));
    }
    Ok(samples.iter().any(|(input, limits)| {
        limits
            .iter()
            .copied()
            .min()
            .is_some_and(|limit| *input >= limit)
    }))
}

fn macos_topology_sha256() -> Result<String, LocalPerfRunError> {
    let mut values = Vec::new();
    for key in [
        "machdep.cpu.brand_string",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.memsize",
        "hw.pagesize",
        "hw.perflevel0.physicalcpu",
        "hw.perflevel1.physicalcpu",
    ] {
        values.push(format!("{key}={}", command_output("sysctl", &["-n", key])?));
    }
    Ok(sha256_hex(values.join("\n").as_bytes()))
}

fn parse_u64_field(value: &str, field: &str) -> Result<u64, LocalPerfRunError> {
    value.trim().parse::<u64>().map_err(|error| {
        LocalPerfRunError::Invalid(format!("{field} is not an unsigned integer: {error}"))
    })
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, LocalPerfRunError> {
    let mut command = Command::new(program);
    command.args(arguments);
    command_output_from(&mut command, program)
}

fn command_output_from(command: &mut Command, label: &str) -> Result<String, LocalPerfRunError> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(LocalPerfRunError::Invalid(format!(
            "{label} command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| {
            LocalPerfRunError::Invalid(format!("{label} output is not UTF-8: {error}"))
        })
}

fn utc_now() -> Result<String, LocalPerfRunError> {
    command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
}

fn create_new_file_at(directory: &File, name: impl AsRef<Path>) -> Result<File, LocalPerfRunError> {
    let file = openat(
        directory,
        name.as_ref(),
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .map_err(std::io::Error::from)?;
    Ok(File::from(file))
}

fn write_new_sync_at(
    directory: &File,
    name: impl AsRef<Path>,
    bytes: &[u8],
) -> Result<(), LocalPerfRunError> {
    let mut file = create_new_file_at(directory, name)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn atomically_publish_new_sync_at(
    directory: &File,
    pending_name: impl AsRef<Path>,
    final_name: impl AsRef<Path>,
    bytes: &[u8],
) -> Result<(), LocalPerfRunError> {
    write_new_sync_at(directory, pending_name.as_ref(), bytes)?;
    renameat_with(
        directory,
        pending_name.as_ref(),
        directory,
        final_name.as_ref(),
        RenameFlags::NOREPLACE,
    )
    .map_err(std::io::Error::from)?;
    directory.sync_all()?;
    Ok(())
}

fn read_file_at(directory: &File, name: impl AsRef<Path>) -> Result<Vec<u8>, LocalPerfRunError> {
    read_file_with_handle_at(directory, name).map(|(_, bytes)| bytes)
}

fn read_file_with_handle_at(
    directory: &File,
    name: impl AsRef<Path>,
) -> Result<(File, Vec<u8>), LocalPerfRunError> {
    let file = openat(
        directory,
        name.as_ref(),
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)?;
    let mut file = File::from(file);
    let stat = fstat(&file).map_err(std::io::Error::from)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_uid != geteuid().as_raw()
    {
        return Err(LocalPerfRunError::Invalid(
            "runner artifact must be an effective-user-owned regular single-link file".to_owned(),
        ));
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok((file, bytes))
}

#[derive(Debug)]
struct DurableChildArtifacts {
    threshold_bytes: Vec<u8>,
    evidence_bytes: Vec<u8>,
}

fn read_and_sync_child_artifacts(
    artifact_directory: &File,
    threshold_name: &str,
    evidence_name: &str,
) -> Result<DurableChildArtifacts, LocalPerfRejectionStage> {
    let (threshold_file, threshold_bytes) =
        read_file_with_handle_at(artifact_directory, threshold_name)
            .map_err(|_| LocalPerfRejectionStage::ArtifactRead)?;
    threshold_file
        .sync_all()
        .map_err(|_| LocalPerfRejectionStage::ArtifactDurability)?;
    let (evidence_file, evidence_bytes) =
        read_file_with_handle_at(artifact_directory, evidence_name)
            .map_err(|_| LocalPerfRejectionStage::ArtifactRead)?;
    evidence_file
        .sync_all()
        .map_err(|_| LocalPerfRejectionStage::ArtifactDurability)?;
    artifact_directory
        .sync_all()
        .map_err(|_| LocalPerfRejectionStage::ArtifactDurability)?;
    Ok(DurableChildArtifacts {
        threshold_bytes,
        evidence_bytes,
    })
}

fn verify_environment_policy(
    run_directory: &File,
    build: &CapturedBuild,
) -> Result<(), LocalPerfRunError> {
    let bytes = read_file_at(run_directory, "environment-policy.json")?;
    if bytes != build.environment_policy_bytes
        || sha256_hex(&bytes) != build.receipt.environment_sha256
    {
        return Err(LocalPerfRunError::Invalid(
            "persisted controlled-environment policy differs from the receipt-bound preimage"
                .to_owned(),
        ));
    }
    Ok(())
}

fn write_failed_attempt_receipt(
    config: &LocalPerfRunConfig,
    run_profile: &RunProfileContract,
    run_selection: &ResolvedRunSelection,
    durability: &RunnerDurability,
    directories: &RunDirectories,
    build: &CapturedBuild,
    producer: &ExecutingProducer,
    paths: &ExternalRunPaths,
    start: &PlatformCapture,
    outcome: LocalPerfAttemptOutcome,
    process_lifecycle: LocalPerfProcessLifecycle,
    root_process_identity: LocalPerfRootProcessIdentity,
    run_log_bytes: Option<&[u8]>,
    started_at_utc: &str,
) -> Result<PathBuf, LocalPerfRunError> {
    let (finished_at_utc, finished_timestamp_error) = if matches!(
        outcome,
        LocalPerfAttemptOutcome::PostExitRejected {
            stage: LocalPerfRejectionStage::FinishedTimestamp
        }
    ) {
        (
            started_at_utc.to_owned(),
            Some("finish timestamp capture rejected before receipt finalization".to_owned()),
        )
    } else {
        match utc_now() {
            Ok(timestamp) => (timestamp, None),
            Err(error) => (started_at_utc.to_owned(), Some(bounded_diagnostic(&error))),
        }
    };
    let (execution_end, end_capture_error) = if matches!(
        outcome,
        LocalPerfAttemptOutcome::PostExitRejected {
            stage: LocalPerfRejectionStage::EndPlatformCapture
        }
    ) {
        (
            None,
            Some("end platform capture rejected before receipt finalization".to_owned()),
        )
    } else {
        match capture_platform(config) {
            Ok(end) if end.hardware == start.hardware && end.request == start.request => {
                (Some(end.snapshot), None)
            }
            Ok(_) => (
                None,
                Some("end hardware or execution request drifted".to_owned()),
            ),
            Err(error) => (None, Some(bounded_diagnostic(&error))),
        }
    };
    let (post_run_identity_verified, post_run_identity_error) = if matches!(
        outcome,
        LocalPerfAttemptOutcome::PostExitRejected {
            stage: LocalPerfRejectionStage::PostRunIdentity
        }
    ) {
        (
            false,
            Some("post-run source or executable identity verification rejected".to_owned()),
        )
    } else {
        let post_identity = verify_external_paths(paths)
            .and_then(|()| verify_run_directories(config, directories))
            .and_then(|()| verify_prepared_build(build, producer, &paths.target))
            .and_then(|()| verify_environment_policy(&directories.run.handle, build));
        match post_identity {
            Ok(()) => (true, None),
            Err(error) => (false, Some(bounded_diagnostic(&error))),
        }
    };
    let receipt_path = config
        .output_dir
        .join(format!("{}.attempt.json", config.gate.label()));
    persist_attempt_receipt(
        config,
        run_profile,
        durability,
        directories,
        build,
        start,
        execution_end,
        end_capture_error,
        post_run_identity_verified,
        post_run_identity_error,
        outcome,
        run_selection,
        process_lifecycle,
        root_process_identity,
        run_log_bytes,
        None,
        started_at_utc,
        &finished_at_utc,
        finished_timestamp_error,
    )
    .map_err(|error| LocalPerfRunError::AttemptCommitFailed {
        receipt_path,
        outcome,
        detail: bounded_diagnostic(&error),
    })
}

#[allow(clippy::too_many_arguments)]
fn completed_attempt_receipt_bytes(
    config: &LocalPerfRunConfig,
    run_profile: &RunProfileContract,
    run_selection: &ResolvedRunSelection,
    durability: &RunnerDurability,
    build: &CapturedBuild,
    start: &PlatformCapture,
    end: &PlatformCapture,
    bound_evidence_bytes: &[u8],
    run_log_bytes: &[u8],
    root_process_identity: LocalPerfRootProcessIdentity,
    started_at_utc: &str,
    finished_at_utc: &str,
) -> Result<Vec<u8>, LocalPerfRunError> {
    let evidence = PerfEvidenceArtifact::from_verified_slice(bound_evidence_bytes)?;
    if evidence.gate != config.gate
        || evidence.applicability_plan != *run_profile.applicability_plan.binding()
        || evidence.provenance.run_id != config.run_id
        || evidence.provenance.run_window != config.run_window
    {
        return Err(LocalPerfRunError::Invalid(
            "completed bound evidence differs from the runner gate, plan, or run identity"
                .to_owned(),
        ));
    }
    let selected = evidence_cell_ids(&evidence);
    validate_selected_cell_ids(
        &selected,
        &selected_cell_ids(&run_profile.applicability_plan),
    )?;
    if selected != run_selection.selected_cell_ids {
        return Err(LocalPerfRunError::Invalid(
            "completed bound evidence cells differ from the frozen typed selection".to_owned(),
        ));
    }
    let identity = evidence.machine_class.identity().ok_or_else(|| {
        LocalPerfRunError::Invalid(
            "completed bound evidence has no admitted runner identity".to_owned(),
        )
    })?;
    if identity.artifact_manifest().is_none() {
        return Err(LocalPerfRunError::Invalid(
            "completed bound evidence runner identity has no artifact manifest".to_owned(),
        ));
    }
    build_attempt_receipt_bytes(
        config,
        run_profile,
        durability,
        build,
        start,
        Some(end.snapshot.clone()),
        None,
        true,
        None,
        LocalPerfAttemptOutcome::Completed,
        run_selection,
        LocalPerfProcessLifecycle {
            spawn_attempted: true,
            spawn_succeeded: true,
            wait_completed: true,
            child_reaped: true,
            run_log_synced: true,
            run_log_captured: true,
            process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
        },
        root_process_identity,
        Some(run_log_bytes),
        Some(bound_evidence_bytes),
        started_at_utc,
        finished_at_utc,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn persist_attempt_receipt(
    config: &LocalPerfRunConfig,
    run_profile: &RunProfileContract,
    durability: &RunnerDurability,
    directories: &RunDirectories,
    build: &CapturedBuild,
    start: &PlatformCapture,
    execution_end: Option<RunnerExecutionSnapshot>,
    end_capture_error: Option<String>,
    post_run_identity_verified: bool,
    post_run_identity_error: Option<String>,
    outcome: LocalPerfAttemptOutcome,
    run_selection: &ResolvedRunSelection,
    process_lifecycle: LocalPerfProcessLifecycle,
    root_process_identity: LocalPerfRootProcessIdentity,
    run_log_bytes: Option<&[u8]>,
    completed_bound_evidence: Option<&[u8]>,
    started_at_utc: &str,
    finished_at_utc: &str,
    finished_timestamp_error: Option<String>,
) -> Result<PathBuf, LocalPerfRunError> {
    let receipt_bytes = build_attempt_receipt_bytes(
        config,
        run_profile,
        durability,
        build,
        start,
        execution_end,
        end_capture_error,
        post_run_identity_verified,
        post_run_identity_error,
        outcome,
        run_selection,
        process_lifecycle,
        root_process_identity,
        run_log_bytes,
        completed_bound_evidence,
        started_at_utc,
        finished_at_utc,
        finished_timestamp_error,
    )?;
    let receipt_name = format!("{}.attempt.json", config.gate.label());
    let pending_name = format!("{}.attempt.pending", config.gate.label());
    atomically_publish_new_sync_at(
        &directories.run.handle,
        &pending_name,
        &receipt_name,
        &receipt_bytes,
    )?;
    Ok(config.output_dir.join(receipt_name))
}

#[allow(clippy::too_many_arguments)]
fn build_attempt_receipt_bytes(
    config: &LocalPerfRunConfig,
    run_profile: &RunProfileContract,
    durability: &RunnerDurability,
    build: &CapturedBuild,
    start: &PlatformCapture,
    execution_end: Option<RunnerExecutionSnapshot>,
    end_capture_error: Option<String>,
    post_run_identity_verified: bool,
    post_run_identity_error: Option<String>,
    outcome: LocalPerfAttemptOutcome,
    run_selection: &ResolvedRunSelection,
    process_lifecycle: LocalPerfProcessLifecycle,
    root_process_identity: LocalPerfRootProcessIdentity,
    run_log_bytes: Option<&[u8]>,
    completed_bound_evidence: Option<&[u8]>,
    started_at_utc: &str,
    finished_at_utc: &str,
    finished_timestamp_error: Option<String>,
) -> Result<Vec<u8>, LocalPerfRunError> {
    let (retry, unavailable) = attempt_derived_facts(outcome)?;
    let (bound_evidence_sha256, runner_receipt_sha256, runner_artifact_manifest_sha256) =
        if let Some(bound_evidence_bytes) = completed_bound_evidence {
            let evidence = PerfEvidenceArtifact::from_verified_slice(bound_evidence_bytes)?;
            let identity = evidence.machine_class.identity().ok_or_else(|| {
                LocalPerfRunError::Invalid(
                    "completed attempt evidence has no admitted runner identity".to_owned(),
                )
            })?;
            let manifest = identity.artifact_manifest().ok_or_else(|| {
                LocalPerfRunError::Invalid(
                    "completed attempt runner identity has no artifact-manifest binding".to_owned(),
                )
            })?;
            (
                Some(sha256_hex(bound_evidence_bytes)),
                Some(identity.receipt_sha256().to_owned()),
                Some(manifest.manifest_sha256().to_owned()),
            )
        } else {
            (None, None, None)
        };
    let receipt = LocalPerfAttemptReceipt {
        schema_version: LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION.to_owned(),
        mode: "measurement".to_owned(),
        gate: config.gate.label().to_owned(),
        profile: config.profile,
        applicability_plan: run_profile.applicability_plan.binding().clone(),
        fixture_selector: run_selection.fixture.clone(),
        selected_cell_ids: run_selection.selected_cell_ids.clone(),
        run_id: config.run_id.clone(),
        run_window: config.run_window.clone(),
        registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
        hardware: start.hardware.clone(),
        execution_request: start.request.clone(),
        execution_start: start.snapshot.clone(),
        execution_end,
        end_capture_error,
        build: build.receipt.clone(),
        durability: durability.clone(),
        post_run_identity_verified,
        post_run_identity_error,
        outcome,
        retry,
        process_lifecycle,
        root_process_identity,
        internal_lifecycle_gaps: LocalPerfInternalLifecycleGaps {
            actual_work: unavailable,
            queue: unavailable,
            workers_joined: unavailable,
            feed_drained: unavailable,
            pending_zero: unavailable,
        },
        unsupported_controls: vec![
            LocalPerfUnsupportedControl::Timeout,
            LocalPerfUnsupportedControl::Cancellation,
        ],
        run_log_sha256: run_log_bytes.map(sha256_hex),
        bound_evidence_sha256,
        runner_receipt_sha256,
        runner_artifact_manifest_sha256,
        started_at_utc: started_at_utc.to_owned(),
        finished_at_utc: finished_at_utc.to_owned(),
        finished_timestamp_error,
        seal_sha256: String::new(),
    };
    let receipt_bytes = seal_attempt_receipt(receipt)?;
    let verified = LocalPerfAttemptReceipt::from_verified_slice(&receipt_bytes)?;
    if let Some(run_log_bytes) = run_log_bytes {
        verified.verify_run_log(run_log_bytes)?;
    }
    if let Some(bound_evidence_bytes) = completed_bound_evidence {
        verified.verify_bound_evidence(bound_evidence_bytes)?;
    }
    Ok(receipt_bytes)
}

fn selected_cell_ids(plan: &PerfApplicabilityPlan) -> Vec<String> {
    PerfMatrixSpec::complete()
        .for_gate(plan.binding.gate)
        .into_iter()
        .zip(&plan.cells)
        .filter(|(_, entry)| entry.applicability.is_runnable())
        .map(|(cell, _)| format!("{}/{}/{}", plan.binding.gate, cell.fixture, cell.metric))
        .collect()
}

fn validate_fixture_selector_syntax(fixture: &str) -> Result<(), LocalPerfRunError> {
    if fixture.is_empty()
        || fixture.len() > MAX_OUTPUT_COMPONENT_BYTES
        || fixture.trim() != fixture
        || !fixture.bytes().all(|byte| byte.is_ascii_graphic())
    {
        return Err(LocalPerfRunError::Invalid(
            "fixture selector must be bounded nonempty canonical ASCII text".to_owned(),
        ));
    }
    Ok(())
}

fn resolve_run_selection(
    plan: &PerfApplicabilityPlan,
    selection: Option<&LocalPerfRunSelection>,
) -> Result<ResolvedRunSelection, LocalPerfRunError> {
    let Some(selection) = selection else {
        let selected_ids = selected_cell_ids(plan);
        validate_selected_cell_ids(&selected_ids, &selected_ids)?;
        return Ok(ResolvedRunSelection {
            fixture: None,
            selected_cell_ids: selected_ids,
        });
    };
    validate_fixture_selector_syntax(selection.fixture())?;
    let matrix = PerfMatrixSpec::complete();
    let canonical = matrix.for_gate(plan.binding.gate);
    if canonical.len() != plan.cells.len() {
        return Err(LocalPerfRunError::Invalid(
            "selection plan does not classify the complete canonical gate".to_owned(),
        ));
    }
    let mut matched = false;
    let mut selected_ids = Vec::new();
    for (cell, classification) in canonical.into_iter().zip(&plan.cells) {
        if cell.fixture != selection.fixture {
            continue;
        }
        matched = true;
        if !classification.applicability.is_runnable() {
            return Err(LocalPerfRunError::Invalid(format!(
                "fixture selector {:?} names a non-applicable {} cell",
                selection.fixture, plan.binding.gate
            )));
        }
        selected_ids.push(format!(
            "{}/{}/{}",
            plan.binding.gate, cell.fixture, cell.metric
        ));
    }
    if !matched {
        return Err(LocalPerfRunError::Invalid(format!(
            "fixture selector {:?} names no canonical {} cell",
            selection.fixture, plan.binding.gate
        )));
    }
    validate_selected_cell_ids(&selected_ids, &selected_cell_ids(plan))?;
    Ok(ResolvedRunSelection {
        fixture: Some(selection.fixture.clone()),
        selected_cell_ids: selected_ids,
    })
}

fn evidence_cell_ids(evidence: &PerfEvidenceArtifact) -> Vec<String> {
    evidence
        .cells
        .iter()
        .map(|cell| cell.cell_id.clone())
        .collect()
}

fn validate_selected_cell_ids(
    selected: &[String],
    runnable: &[String],
) -> Result<(), LocalPerfRunError> {
    if selected.is_empty() {
        return Err(LocalPerfRunError::Invalid(
            "process receipt must select at least one runnable cell".to_owned(),
        ));
    }
    let selected_set = selected.iter().collect::<BTreeSet<_>>();
    let runnable_set = runnable.iter().collect::<BTreeSet<_>>();
    if selected_set.len() != selected.len() || !selected_set.is_subset(&runnable_set) {
        return Err(LocalPerfRunError::Invalid(
            "process receipt cells must be unique runnable canonical cells".to_owned(),
        ));
    }
    let canonical_subset = runnable
        .iter()
        .filter(|cell_id| selected_set.contains(cell_id))
        .cloned()
        .collect::<Vec<_>>();
    if selected != canonical_subset {
        return Err(LocalPerfRunError::Invalid(
            "process receipt cells are not in canonical matrix order".to_owned(),
        ));
    }
    Ok(())
}

fn local_perf_io_error_kind(error: &std::io::Error) -> LocalPerfIoErrorKind {
    match error.kind() {
        std::io::ErrorKind::NotFound => LocalPerfIoErrorKind::NotFound,
        std::io::ErrorKind::PermissionDenied => LocalPerfIoErrorKind::PermissionDenied,
        std::io::ErrorKind::WouldBlock => LocalPerfIoErrorKind::ResourceBusy,
        _ if error.raw_os_error() == Some(12) => LocalPerfIoErrorKind::OutOfMemory,
        _ => LocalPerfIoErrorKind::Other,
    }
}

fn force_kill_and_reap(
    child: &mut Child,
    root_process_identity: LocalPerfRootProcessIdentity,
) -> Result<(ExitStatus, LocalPerfProcessGroupRecovery), LocalPerfIoErrorKind> {
    let (process_group_recovery, kill_error) =
        match signal_owned_process_group(child, root_process_identity) {
            Ok(()) => (LocalPerfProcessGroupRecovery::SignaledOwnedGroup, None),
            Err(_) => (
                LocalPerfProcessGroupRecovery::DirectChildFallback,
                child
                    .kill()
                    .err()
                    .map(|error| local_perf_io_error_kind(&error)),
            ),
        };
    for _ in 0..WAIT_RECOVERY_POLL_ATTEMPTS {
        match child.try_wait() {
            Ok(Some(status)) => return Ok((status, process_group_recovery)),
            Ok(None) => std::thread::sleep(WAIT_RECOVERY_POLL_INTERVAL),
            Err(error) => return Err(local_perf_io_error_kind(&error)),
        }
    }
    Err(kill_error.unwrap_or(LocalPerfIoErrorKind::ResourceBusy))
}

#[cfg(target_os = "linux")]
fn signal_owned_process_group(
    child: &Child,
    root_process_identity: LocalPerfRootProcessIdentity,
) -> Result<(), LocalPerfIoErrorKind> {
    if capture_root_process_identity(child) != root_process_identity {
        return Err(LocalPerfIoErrorKind::Other);
    }
    kill_process_group(Pid::from_child(child), Signal::KILL)
        .map_err(|error| local_perf_io_error_kind(&std::io::Error::from(error)))
}

#[cfg(not(target_os = "linux"))]
fn signal_owned_process_group(
    _child: &Child,
    _root_process_identity: LocalPerfRootProcessIdentity,
) -> Result<(), LocalPerfIoErrorKind> {
    Err(LocalPerfIoErrorKind::Other)
}

fn attempt_derived_facts(
    outcome: LocalPerfAttemptOutcome,
) -> Result<
    (
        LocalPerfRetryPredicate,
        LocalPerfInternalLifecycleUnavailable,
    ),
    LocalPerfRunError,
> {
    match outcome {
        LocalPerfAttemptOutcome::Completed => Ok((
            LocalPerfRetryPredicate::NotRequired,
            LocalPerfInternalLifecycleUnavailable::ChildEvidenceAdmittedButNotIndependentlyObserved,
        )),
        LocalPerfAttemptOutcome::SpawnRejected { error_kind } => Ok((
            LocalPerfRetryPredicate::RepairSpawn { error_kind },
            LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
        )),
        LocalPerfAttemptOutcome::WaitRecoveredByKill { error_kind } => Ok((
            LocalPerfRetryPredicate::RepairWait { error_kind },
            LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
        )),
        LocalPerfAttemptOutcome::ExitedNonzero { code } if code > 0 => Ok((
            LocalPerfRetryPredicate::DiagnoseNonzeroExit { code },
            LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
        )),
        LocalPerfAttemptOutcome::ExitedNonzero { .. } => Err(LocalPerfRunError::Invalid(
            "nonzero attempt outcome requires a positive exit code".to_owned(),
        )),
        LocalPerfAttemptOutcome::Signaled { signal } if (1..=255).contains(&signal) => Ok((
            LocalPerfRetryPredicate::DiagnoseSignal { signal },
            LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
        )),
        LocalPerfAttemptOutcome::Signaled { .. } => Err(LocalPerfRunError::Invalid(
            "signaled attempt outcome requires a bounded positive signal".to_owned(),
        )),
        LocalPerfAttemptOutcome::UnknownTerminal => Ok((
            LocalPerfRetryPredicate::DiagnoseUnknownTerminal,
            LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
        )),
        LocalPerfAttemptOutcome::PostExitRejected { stage } => Ok((
            LocalPerfRetryPredicate::RepairRejectedEvidence { stage },
            LocalPerfInternalLifecycleUnavailable::ChildEvidenceNotAdmitted,
        )),
    }
}

fn validate_process_lifecycle(
    outcome: LocalPerfAttemptOutcome,
    lifecycle: LocalPerfProcessLifecycle,
    has_run_log_digest: bool,
) -> Result<(), LocalPerfRunError> {
    if !lifecycle.spawn_attempted
        || lifecycle.run_log_captured != has_run_log_digest
        || lifecycle.child_reaped != lifecycle.wait_completed
    {
        return Err(LocalPerfRunError::Invalid(
            "attempt process lifecycle contradicts its captured log or reap facts".to_owned(),
        ));
    }
    let valid = match outcome {
        LocalPerfAttemptOutcome::Completed
        | LocalPerfAttemptOutcome::ExitedNonzero { .. }
        | LocalPerfAttemptOutcome::Signaled { .. }
        | LocalPerfAttemptOutcome::UnknownTerminal => {
            lifecycle.spawn_succeeded
                && lifecycle.wait_completed
                && lifecycle.child_reaped
                && lifecycle.run_log_synced
                && lifecycle.run_log_captured
        }
        LocalPerfAttemptOutcome::SpawnRejected { .. } => {
            !lifecycle.spawn_succeeded && !lifecycle.wait_completed && !lifecycle.child_reaped
        }
        LocalPerfAttemptOutcome::WaitRecoveredByKill { .. } => {
            lifecycle.spawn_succeeded
                && lifecycle.wait_completed
                && lifecycle.child_reaped
                && lifecycle.run_log_synced
                && lifecycle.run_log_captured
        }
        LocalPerfAttemptOutcome::PostExitRejected { stage } => {
            let terminal =
                lifecycle.spawn_succeeded && lifecycle.wait_completed && lifecycle.child_reaped;
            terminal
                && match stage {
                    LocalPerfRejectionStage::RunLogSync => !lifecycle.run_log_synced,
                    LocalPerfRejectionStage::RunLogRead => {
                        lifecycle.run_log_synced && !lifecycle.run_log_captured
                    }
                    _ => lifecycle.run_log_synced && lifecycle.run_log_captured,
                }
        }
    };
    if !valid {
        return Err(LocalPerfRunError::Invalid(
            "attempt process lifecycle disagrees with its typed terminal outcome".to_owned(),
        ));
    }
    let recovery_matches_outcome = match outcome {
        LocalPerfAttemptOutcome::WaitRecoveredByKill { .. } => matches!(
            lifecycle.process_group_recovery,
            LocalPerfProcessGroupRecovery::SignaledOwnedGroup
                | LocalPerfProcessGroupRecovery::DirectChildFallback
        ),
        _ => lifecycle.process_group_recovery == LocalPerfProcessGroupRecovery::NotRequired,
    };
    if !recovery_matches_outcome {
        return Err(LocalPerfRunError::Invalid(
            "attempt process-group recovery authority disagrees with its terminal outcome"
                .to_owned(),
        ));
    }
    Ok(())
}

fn validate_root_process_identity(
    outcome: LocalPerfAttemptOutcome,
    lifecycle: LocalPerfProcessLifecycle,
    root_process_identity: LocalPerfRootProcessIdentity,
) -> Result<(), LocalPerfRunError> {
    let identity_matches_spawn = match (lifecycle.spawn_succeeded, root_process_identity) {
        (false, LocalPerfRootProcessIdentity::NotSpawned) => true,
        (
            true,
            LocalPerfRootProcessIdentity::LinuxProcStartTime {
                pid,
                process_group_id,
                start_time_ticks,
            },
        ) => pid > 0 && process_group_id == pid && start_time_ticks > 0,
        (true, LocalPerfRootProcessIdentity::Unverifiable { pid }) => pid > 0,
        _ => false,
    };
    if !identity_matches_spawn
        || (outcome == LocalPerfAttemptOutcome::Completed
            && !root_process_identity.has_verified_birth_identity())
    {
        return Err(LocalPerfRunError::Invalid(
            "attempt root-process identity contradicts spawn state or completed receipt authority"
                .to_owned(),
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn capture_root_process_identity(child: &Child) -> LocalPerfRootProcessIdentity {
    let pid = child.id();
    let stat_path = format!("/proc/{pid}/stat");
    let identity = fs::read_to_string(stat_path)
        .ok()
        .and_then(|stat| parse_linux_proc_process_identity(&stat))
        .and_then(|(process_group_id, start_time_ticks)| {
            (process_group_id == pid).then_some(LocalPerfRootProcessIdentity::LinuxProcStartTime {
                pid,
                process_group_id,
                start_time_ticks,
            })
        });
    identity.unwrap_or(LocalPerfRootProcessIdentity::Unverifiable { pid })
}

#[cfg(not(target_os = "linux"))]
fn capture_root_process_identity(child: &Child) -> LocalPerfRootProcessIdentity {
    LocalPerfRootProcessIdentity::Unverifiable { pid: child.id() }
}

#[cfg(target_os = "linux")]
fn parse_linux_proc_process_identity(stat: &str) -> Option<(u32, u64)> {
    // `/proc/<pid>/stat`'s second field is parenthesized and may itself carry
    // spaces or parentheses. Split from its final close-paren so field 22
    // (the start-time tick) stays positional even for adversarial comm values.
    let (_, suffix) = stat.rsplit_once(')')?;
    let mut fields = suffix.split_ascii_whitespace();
    let _state = fields.next()?;
    let _parent_pid = fields.next()?;
    let process_group_id = fields
        .next()?
        .parse::<u32>()
        .ok()
        .filter(|process_group_id| *process_group_id > 0)?;
    let start_time_ticks = fields
        .nth(16)
        .and_then(|start_time_ticks| start_time_ticks.parse::<u64>().ok())
        .filter(|start_time_ticks| *start_time_ticks > 0)?;
    Some((process_group_id, start_time_ticks))
}

fn validate_attempt_build(build: &RunnerBuild) -> Result<(), LocalPerfRunError> {
    let producer = &build.producer;
    if !is_git_revision(&build.git_revision)
        || build.git_dirty
        || build.worktree_state_sha256.is_some()
        || !is_sha256(&build.cargo_lock_sha256)
        || !is_sha256(&build.executable_sha256)
        || !is_sha256(&build.command_sha256)
        || !is_sha256(&build.environment_sha256)
        || producer.contract_version != LOCAL_PERF_PRODUCER_CONTRACT_VERSION
        || producer.source_git_revision != build.git_revision
        || producer.source_git_dirty
        || producer.cargo_lock_sha256 != build.cargo_lock_sha256
        || !is_sha256(&producer.executable_sha256)
    {
        return Err(LocalPerfRunError::Invalid(
            "attempt receipt carries a malformed or inconsistent clean build identity".to_owned(),
        ));
    }
    Ok(())
}

fn validate_utc_timestamp(value: &str, field: &str) -> Result<(), LocalPerfRunError> {
    let bytes = value.as_bytes();
    let punctuation = [
        (4, b'-'),
        (7, b'-'),
        (10, b'T'),
        (13, b':'),
        (16, b':'),
        (19, b'Z'),
    ];
    if bytes.len() != 20
        || punctuation
            .iter()
            .any(|(index, expected)| bytes[*index] != *expected)
        || bytes.iter().enumerate().any(|(index, byte)| {
            !punctuation.iter().any(|(position, _)| *position == index) && !byte.is_ascii_digit()
        })
    {
        return Err(LocalPerfRunError::Invalid(format!(
            "{field} timestamp is not bounded canonical UTC"
        )));
    }
    let parse = |start: usize, end: usize| {
        value[start..end]
            .parse::<u32>()
            .map_err(|_| LocalPerfRunError::Invalid(format!("{field} timestamp is not numeric")))
    };
    let year = parse(0, 4)?;
    let month = parse(5, 7)?;
    let day = parse(8, 10)?;
    let hour = parse(11, 13)?;
    let minute = parse(14, 16)?;
    let second = parse(17, 19)?;
    let leap_year =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > days_in_month || hour >= 24 || minute >= 60 || second >= 60 {
        return Err(LocalPerfRunError::Invalid(format!(
            "{field} timestamp is outside canonical UTC calendar/time ranges"
        )));
    }
    Ok(())
}

fn bounded_diagnostic<T: std::fmt::Display + ?Sized>(value: &T) -> String {
    let mut message = value.to_string();
    let mut limit = message.len().min(240);
    while !message.is_char_boundary(limit) {
        limit -= 1;
    }
    message.truncate(limit);
    message
}

fn seal_attempt_receipt(
    mut receipt: LocalPerfAttemptReceipt,
) -> Result<Vec<u8>, LocalPerfRunError> {
    receipt.seal_sha256.clear();
    let preimage = serde_json::to_vec(&receipt)?;
    receipt.seal_sha256 = sha256_hex(&preimage);
    serde_json::to_vec(&receipt).map_err(LocalPerfRunError::from)
}

#[cfg(test)]
pub fn completed_attempt_receipt_for_test(
    artifact: &PerfEvidenceArtifact,
    fixture_selector: Option<&str>,
    run_log_bytes: &[u8],
    threshold_bytes: &[u8],
    prebinding_bytes: &[u8],
    bound_bytes: &[u8],
) -> Vec<u8> {
    let identity = artifact
        .machine_class
        .identity()
        .expect("test evidence has an admitted runner identity");
    let manifest = identity
        .artifact_manifest()
        .expect("test runner identity has an artifact manifest");
    let canonical_bound = artifact
        .canonical_json()
        .expect("canonical test bound evidence");
    assert_eq!(bound_bytes, canonical_bound.as_bytes());
    assert_eq!(
        prebinding_bytes,
        artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstructed test prebinding evidence")
    );
    identity
        .verify_artifact_inputs(run_log_bytes, threshold_bytes, prebinding_bytes)
        .expect("test runner identity binds exact artifact inputs");
    let runner: RunnerReceipt =
        serde_json::from_str(identity.receipt_json()).expect("test runner receipt JSON");
    let registry = MachineClassRegistry::frozen().expect("frozen test registry");
    let plan = PerfMatrixSpec::complete()
        .applicability_plan(&registry, runner.derived_profile, PerfGate::Qg1)
        .expect("canonical QG-1 test plan");
    let selection = fixture_selector
        .map(|fixture| LocalPerfRunSelection::for_fixture(fixture.to_owned()))
        .transpose()
        .expect("valid test fixture selector");
    let resolved =
        resolve_run_selection(&plan, selection.as_ref()).expect("resolved test run selection");
    let artifact_cell_ids = evidence_cell_ids(artifact);
    assert_eq!(
        artifact_cell_ids, resolved.selected_cell_ids,
        "test artifact must exactly equal its typed H2 selection"
    );
    let (retry, unavailable) =
        attempt_derived_facts(LocalPerfAttemptOutcome::Completed).expect("completed test outcome");
    let receipt = LocalPerfAttemptReceipt {
        schema_version: LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION.to_owned(),
        mode: "measurement".to_owned(),
        gate: PerfGate::Qg1.label().to_owned(),
        profile: runner.derived_profile,
        applicability_plan: plan.binding().clone(),
        fixture_selector: resolved.fixture,
        selected_cell_ids: resolved.selected_cell_ids,
        run_id: artifact.provenance.run_id.clone(),
        run_window: artifact.provenance.run_window.clone(),
        registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
        hardware: runner.hardware,
        execution_request: runner.execution.request,
        execution_start: runner.execution.start,
        execution_end: Some(runner.execution.end),
        end_capture_error: None,
        build: runner.build,
        durability: runner.durability,
        post_run_identity_verified: true,
        post_run_identity_error: None,
        outcome: LocalPerfAttemptOutcome::Completed,
        retry,
        process_lifecycle: LocalPerfProcessLifecycle {
            spawn_attempted: true,
            spawn_succeeded: true,
            wait_completed: true,
            child_reaped: true,
            run_log_synced: true,
            run_log_captured: true,
            process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
        },
        root_process_identity: LocalPerfRootProcessIdentity::LinuxProcStartTime {
            pid: 37,
            process_group_id: 37,
            start_time_ticks: 81,
        },
        internal_lifecycle_gaps: LocalPerfInternalLifecycleGaps {
            actual_work: unavailable,
            queue: unavailable,
            workers_joined: unavailable,
            feed_drained: unavailable,
            pending_zero: unavailable,
        },
        unsupported_controls: vec![
            LocalPerfUnsupportedControl::Timeout,
            LocalPerfUnsupportedControl::Cancellation,
        ],
        run_log_sha256: Some(sha256_hex(run_log_bytes)),
        bound_evidence_sha256: Some(sha256_hex(bound_bytes)),
        runner_receipt_sha256: Some(identity.receipt_sha256().to_owned()),
        runner_artifact_manifest_sha256: Some(manifest.manifest_sha256().to_owned()),
        started_at_utc: runner.completion.started_at_utc,
        finished_at_utc: runner.completion.finished_at_utc,
        finished_timestamp_error: None,
        seal_sha256: String::new(),
    };
    let bytes = seal_attempt_receipt(receipt).expect("seal completed H2 test receipt");
    let verified =
        LocalPerfAttemptReceipt::from_verified_slice(&bytes).expect("verify H2 test receipt");
    verified
        .verify_run_log(run_log_bytes)
        .expect("verify H2 test run log");
    verified
        .verify_bound_evidence(bound_bytes)
        .expect("verify H2 test bound evidence");
    bytes
}

#[cfg(test)]
pub fn failed_attempt_receipt_for_test(
    artifact: &PerfEvidenceArtifact,
    fixture_selector: Option<&str>,
    run_log_bytes: &[u8],
    threshold_bytes: &[u8],
    prebinding_bytes: &[u8],
    bound_bytes: &[u8],
    code: i64,
) -> Vec<u8> {
    let completed_bytes = completed_attempt_receipt_for_test(
        artifact,
        fixture_selector,
        run_log_bytes,
        threshold_bytes,
        prebinding_bytes,
        bound_bytes,
    );
    let mut receipt = LocalPerfAttemptReceipt::from_verified_slice(&completed_bytes)
        .expect("parse completed H2 test receipt");
    let outcome = LocalPerfAttemptOutcome::ExitedNonzero { code };
    let (retry, unavailable) =
        attempt_derived_facts(outcome).expect("nonzero test outcome has a typed retry");
    receipt.outcome = outcome;
    receipt.retry = retry;
    receipt.internal_lifecycle_gaps = LocalPerfInternalLifecycleGaps {
        actual_work: unavailable,
        queue: unavailable,
        workers_joined: unavailable,
        feed_drained: unavailable,
        pending_zero: unavailable,
    };
    receipt.bound_evidence_sha256 = None;
    receipt.runner_receipt_sha256 = None;
    receipt.runner_artifact_manifest_sha256 = None;
    let bytes = seal_attempt_receipt(receipt).expect("seal failed H2 test receipt");
    let verified = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
        .expect("verify failed H2 test receipt");
    verified
        .verify_run_log(run_log_bytes)
        .expect("verify failed H2 test run log");
    bytes
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read};

    use super::*;

    fn production_source() -> &'static str {
        const TEST_MODULE_BOUNDARY: &str = "#[cfg(test)]\nmod tests {";

        let source = include_str!("local_perf_runner.rs");
        assert_eq!(
            source.matches(TEST_MODULE_BOUNDARY).count(),
            1,
            "the production/test boundary must be unique"
        );
        let test_module_start = source
            .find(TEST_MODULE_BOUNDARY)
            .expect("unique production/test boundary");
        &source[..test_module_start]
    }

    fn unique_marker_offset(source: &str, marker: &str) -> usize {
        assert_eq!(
            source.matches(marker).count(),
            1,
            "expected one production occurrence of {marker:?}"
        );
        source.find(marker).expect("unique production marker")
    }

    fn profile(
        hardware_class_id: HardwareClassId,
        execution_profile_id: ExecutionProfileId,
    ) -> MachineProfileKey {
        MachineProfileKey::new(hardware_class_id, execution_profile_id)
            .expect("registered test profile")
    }

    fn policy_config(gate: PerfGate) -> LocalPerfRunConfig {
        LocalPerfRunConfig {
            gate,
            profile: profile(HardwareClassId::M4Macos, ExecutionProfileId::Scheduler10),
            run_id: "candidate-1".to_owned(),
            run_window: "window-1".to_owned(),
            measurement_runs: MIN_MEASUREMENT_RUNS,
            output_dir: PathBuf::from("/tmp/frankensearch-perf-run"),
        }
    }

    fn physical_qg1_plan() -> PerfApplicabilityPlan {
        PerfMatrixSpec::complete()
            .applicability_plan(
                &MachineClassRegistry::frozen().expect("frozen registry"),
                profile(
                    HardwareClassId::TrjZen35995wx,
                    ExecutionProfileId::Physical64,
                ),
                PerfGate::Qg1,
            )
            .expect("physical-64 QG-1 plan")
    }

    fn first_runnable_qg1_selection(plan: &PerfApplicabilityPlan) -> ResolvedRunSelection {
        let fixture = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .zip(&plan.cells)
            .find(|(_, classification)| classification.applicability.is_runnable())
            .map(|(cell, _)| cell.fixture.clone())
            .expect("runnable QG-1 fixture");
        let selection =
            LocalPerfRunSelection::for_fixture(fixture).expect("typed runnable fixture selection");
        resolve_run_selection(plan, Some(&selection)).expect("resolved exact fixture selection")
    }

    fn attempt_runner_identity() -> VerifiedRunnerIdentity {
        crate::machine_class_registry::admitted_test_identity_for_artifacts(
            "QG-1",
            &"e".repeat(40),
            &"f".repeat(64),
            &"a".repeat(64),
            &"b".repeat(64),
            &"c".repeat(64),
            "local-perf-attempt",
            "attempt-1",
            "window-1",
            b"threshold artifact",
            b"prebinding evidence artifact",
        )
    }

    fn attempt_fixture(
        outcome: LocalPerfAttemptOutcome,
        bound_evidence_bytes: Option<&[u8]>,
    ) -> (LocalPerfAttemptReceipt, Vec<u8>, Vec<u8>) {
        let run_label = "local-perf-attempt";
        let run_log_bytes = format!("runner-log:{run_label}").into_bytes();
        let identity = attempt_runner_identity();
        let runner: RunnerReceipt =
            serde_json::from_str(identity.receipt_json()).expect("admitted runner fixture");
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let plan = PerfMatrixSpec::complete()
            .applicability_plan(&registry, runner.derived_profile, PerfGate::Qg1)
            .expect("canonical QG-1 plan");
        let (retry, unavailable) = attempt_derived_facts(outcome).expect("valid outcome");
        let process_lifecycle = match outcome {
            LocalPerfAttemptOutcome::SpawnRejected { .. } => LocalPerfProcessLifecycle {
                spawn_attempted: true,
                spawn_succeeded: false,
                wait_completed: false,
                child_reaped: false,
                run_log_synced: true,
                run_log_captured: true,
                process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
            },
            LocalPerfAttemptOutcome::WaitRecoveredByKill { .. } => LocalPerfProcessLifecycle {
                spawn_attempted: true,
                spawn_succeeded: true,
                wait_completed: true,
                child_reaped: true,
                run_log_synced: true,
                run_log_captured: true,
                process_group_recovery: LocalPerfProcessGroupRecovery::SignaledOwnedGroup,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::RunLogSync,
            } => LocalPerfProcessLifecycle {
                spawn_attempted: true,
                spawn_succeeded: true,
                wait_completed: true,
                child_reaped: true,
                run_log_synced: false,
                run_log_captured: true,
                process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::RunLogRead,
            } => LocalPerfProcessLifecycle {
                spawn_attempted: true,
                spawn_succeeded: true,
                wait_completed: true,
                child_reaped: true,
                run_log_synced: true,
                run_log_captured: false,
                process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
            },
            _ => LocalPerfProcessLifecycle {
                spawn_attempted: true,
                spawn_succeeded: true,
                wait_completed: true,
                child_reaped: true,
                run_log_synced: true,
                run_log_captured: true,
                process_group_recovery: LocalPerfProcessGroupRecovery::NotRequired,
            },
        };
        let completed = outcome == LocalPerfAttemptOutcome::Completed;
        let root_process_identity = if process_lifecycle.spawn_succeeded {
            LocalPerfRootProcessIdentity::LinuxProcStartTime {
                pid: 37,
                process_group_id: 37,
                start_time_ticks: 81,
            }
        } else {
            LocalPerfRootProcessIdentity::NotSpawned
        };
        let manifest_sha256 = identity
            .artifact_manifest()
            .expect("artifact-bound runner fixture")
            .manifest_sha256()
            .to_owned();
        let (execution_end, end_capture_error) = match outcome {
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::EndPlatformCapture,
            } => (None, Some("end capture failed".to_owned())),
            _ => (Some(runner.execution.end.clone()), None),
        };
        let (post_run_identity_verified, post_run_identity_error) = match outcome {
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::PostRunIdentity,
            } => (false, Some("post-run identity failed".to_owned())),
            _ => (true, None),
        };
        let started_at_utc = runner.completion.started_at_utc.clone();
        let (finished_at_utc, finished_timestamp_error) = match outcome {
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::FinishedTimestamp,
            } => (
                started_at_utc.clone(),
                Some("finish timestamp failed".to_owned()),
            ),
            _ => (runner.completion.finished_at_utc.clone(), None),
        };
        let receipt = LocalPerfAttemptReceipt {
            schema_version: LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION.to_owned(),
            mode: "measurement".to_owned(),
            gate: "QG-1".to_owned(),
            profile: runner.derived_profile,
            applicability_plan: plan.binding().clone(),
            fixture_selector: None,
            selected_cell_ids: selected_cell_ids(&plan),
            run_id: "attempt-1".to_owned(),
            run_window: "window-1".to_owned(),
            registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
            hardware: runner.hardware,
            execution_request: runner.execution.request,
            execution_start: runner.execution.start,
            execution_end,
            end_capture_error,
            build: runner.build,
            durability: runner.durability,
            post_run_identity_verified,
            post_run_identity_error,
            outcome,
            retry,
            process_lifecycle,
            root_process_identity,
            internal_lifecycle_gaps: LocalPerfInternalLifecycleGaps {
                actual_work: unavailable,
                queue: unavailable,
                workers_joined: unavailable,
                feed_drained: unavailable,
                pending_zero: unavailable,
            },
            unsupported_controls: vec![
                LocalPerfUnsupportedControl::Timeout,
                LocalPerfUnsupportedControl::Cancellation,
            ],
            run_log_sha256: process_lifecycle
                .run_log_captured
                .then(|| sha256_hex(&run_log_bytes)),
            bound_evidence_sha256: bound_evidence_bytes.map(sha256_hex),
            runner_receipt_sha256: completed.then(|| identity.receipt_sha256().to_owned()),
            runner_artifact_manifest_sha256: completed.then_some(manifest_sha256),
            started_at_utc,
            finished_at_utc,
            finished_timestamp_error,
            seal_sha256: String::new(),
        };
        let bytes = seal_attempt_receipt(receipt).expect("seal attempt fixture");
        let verified =
            LocalPerfAttemptReceipt::from_verified_slice(&bytes).expect("verify attempt fixture");
        (verified, bytes, run_log_bytes)
    }

    #[test]
    fn m4_static_registry_plan_remains_required_while_live_promotion_fails_closed() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        assert_eq!(
            PerfGate::ALL.len(),
            10,
            "the static registry proof must cover every normative gate"
        );
        for gate in PerfGate::ALL {
            let config = policy_config(gate);
            let resolved =
                resolve_run_profile(&config, &registry).expect("registered M4 applicability");
            assert_eq!(
                resolved.applicability_plan.default_flip_disposition,
                DefaultFlipDisposition::RequiredForDefaultFlip,
                "{gate} must remain required even while this producer is unavailable"
            );
            assert!(
                resolved
                    .applicability_plan
                    .cells
                    .iter()
                    .all(|cell| cell.reason != PerfCellApplicabilityReason::DiagnosticProfile),
                "{gate} must not relabel the required M4 profile as diagnostic"
            );

            let error = validate_platform_gate_policy(&config)
                .expect_err("every current M4 promotion path must reject");
            if matches!(gate, PerfGate::Qg3 | PerfGate::Qg4 | PerfGate::Qg5) {
                assert!(error.to_string().contains("any host"));
            } else {
                assert!(error.to_string().contains("actual executing image"));
                assert!(
                    error
                        .to_string()
                        .contains("required, runnable static registry applicability plan")
                );
                assert!(
                    error
                        .to_string()
                        .contains("cannot emit promotion-admissible M4 evidence")
                );
                assert!(error.to_string().contains("does not attest a live M4 host"));
            }
        }
    }

    #[test]
    fn smt2_128_static_qg1_plan_pins_the_full_canonical_horizon() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let config = LocalPerfRunConfig {
            gate: PerfGate::Qg1,
            profile: profile(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128),
            run_id: "candidate-1".to_owned(),
            run_window: "window-1".to_owned(),
            measurement_runs: MIN_MEASUREMENT_RUNS,
            output_dir: PathBuf::from("/tmp/frankensearch-perf-run"),
        };
        let resolved =
            resolve_run_profile(&config, &registry).expect("registered SMT2 applicability");
        assert_eq!(
            resolved.capacity_semantics,
            ExecutionCapacitySemantics::LogicalThreads
        );
        assert_eq!(resolved.applicability_plan.cells.len(), 74);
        assert_eq!(
            resolved
                .applicability_plan
                .cell_count(PerfCellApplicability::Required),
            72
        );
        assert_eq!(
            resolved
                .applicability_plan
                .cell_count(PerfCellApplicability::Diagnostic),
            2
        );
        assert_eq!(
            resolved
                .applicability_plan
                .cell_count(PerfCellApplicability::NotApplicable),
            0
        );
        assert_eq!(
            resolved
                .applicability_plan
                .binding()
                .profile_contract_sha256,
            "09a6c596bc866faa72a9e3f34e7ea58c78751eec9172cfe6294eca4b9005d3e1"
        );
        let runnable_widths = resolved
            .applicability_plan
            .cells
            .iter()
            .filter(|cell| cell.applicability.is_runnable())
            .map(|cell| cell.configured_threads)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            runnable_widths,
            [1, 2, 4, 8, 16, 32, 64, 96, 128]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(resolved.execution_capacity, 128);
        assert_eq!(resolved.max_exercised_cell_width, 128);
        assert_eq!(
            resolved.applicability_plan.max_runnable_cell_width(),
            Some(128)
        );
        // This pins the immutable static plan only; it is not a live-host
        // execution or residency attestation.
    }

    #[test]
    fn identity_components_enforce_exact_ascii_and_length_boundaries() {
        validate_component(&"a".repeat(MAX_IDENTITY_COMPONENT_BYTES), "run ID")
            .expect("maximum-length run ID");
        for rejected in [
            String::new(),
            ".".to_owned(),
            "..".to_owned(),
            "with/slash".to_owned(),
            "line\nbreak".to_owned(),
            "a".repeat(MAX_IDENTITY_COMPONENT_BYTES + 1),
        ] {
            assert!(
                validate_component(&rejected, "run ID").is_err(),
                "{rejected:?} admitted"
            );
        }
    }

    #[test]
    fn run_profile_is_derived_from_the_frozen_profile_and_applicability_plan() {
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        for (profile, gate, semantics, capacity, max_width) in [
            (
                profile(
                    HardwareClassId::TrjZen35995wx,
                    ExecutionProfileId::Physical64,
                ),
                PerfGate::Qg1,
                ExecutionCapacitySemantics::PhysicalCores,
                64,
                64,
            ),
            (
                profile(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128),
                PerfGate::Qg1,
                ExecutionCapacitySemantics::LogicalThreads,
                128,
                128,
            ),
            (
                profile(
                    HardwareClassId::TrjZen35995wx,
                    ExecutionProfileId::Physical64,
                ),
                PerfGate::Qg7,
                ExecutionCapacitySemantics::PhysicalCores,
                64,
                8,
            ),
            (
                profile(
                    HardwareClassId::TrjZen35995wx,
                    ExecutionProfileId::Physical64,
                ),
                PerfGate::Qg8,
                ExecutionCapacitySemantics::PhysicalCores,
                64,
                32,
            ),
            (
                profile(HardwareClassId::M4Macos, ExecutionProfileId::Scheduler10),
                PerfGate::Qg1,
                ExecutionCapacitySemantics::SchedulerWorkers,
                10,
                8,
            ),
        ] {
            let config = LocalPerfRunConfig {
                gate,
                profile,
                run_id: "candidate-1".to_owned(),
                run_window: "window-1".to_owned(),
                measurement_runs: MIN_MEASUREMENT_RUNS,
                output_dir: PathBuf::from("/tmp/frankensearch-perf-run"),
            };
            let resolved =
                resolve_run_profile(&config, &registry).expect("resolve registered profile");
            assert_eq!(resolved.capacity_semantics, semantics);
            assert_eq!(resolved.execution_capacity, capacity);
            assert_eq!(resolved.max_exercised_cell_width, max_width);
            assert_eq!(
                resolved.applicability_plan.execution_capacity,
                Some(capacity)
            );
            assert_eq!(
                resolved.applicability_plan.max_exercised_cell_width,
                Some(max_width)
            );
        }
    }

    #[test]
    fn output_component_uses_the_same_closed_ascii_boundary() {
        validate_output_component(OsStr::new(&"a".repeat(MAX_OUTPUT_COMPONENT_BYTES)))
            .expect("maximum-length output component");
        for rejected in [
            String::new(),
            ".".to_owned(),
            "..".to_owned(),
            "with/slash".to_owned(),
            "with\\backslash".to_owned(),
            "unicode-é".to_owned(),
            "line\nbreak".to_owned(),
            "a".repeat(MAX_OUTPUT_COMPONENT_BYTES + 1),
        ] {
            assert!(
                validate_output_component(OsStr::new(&rejected)).is_err(),
                "{rejected:?} admitted"
            );
        }
    }

    #[test]
    fn registry_preflight_precedes_log_creation_and_child_spawn() {
        let source = production_source();
        let preflight = unique_marker_offset(source, "let pre_spawn = registry.preflight(");
        let log_creation = unique_marker_offset(
            source,
            "create_new_file_at(&run_directories.run.handle, \"run.log\")",
        );
        let child_spawn = unique_marker_offset(source, "let mut child = match child.spawn()");
        assert!(preflight < log_creation);
        assert!(log_creation < child_spawn);
    }

    #[test]
    fn pinned_directory_and_held_benchmark_reject_path_replacement() {
        let root = tempfile::tempdir().expect("pinned-root test directory");
        let output = root.path().join("output");
        fs::create_dir(&output).expect("create output root");
        let pinned_output = pin_directory(&output, true).expect("pin output root");
        let output_metadata = fs::symlink_metadata(&output).expect("output metadata");
        assert_eq!(
            pinned_output.identity,
            FileIdentity {
                device: output_metadata.dev(),
                inode: output_metadata.ino(),
            },
            "held directory and path metadata must use one device/inode representation"
        );
        let displaced_output = root.path().join("output-displaced");
        fs::rename(&output, &displaced_output).expect("displace output root");
        fs::create_dir(&output).expect("create replacement output root");
        assert!(
            verify_pinned_directory(&pinned_output).is_err(),
            "replacement output path retained held identity"
        );

        let target_path = root.path().join("target");
        fs::create_dir(&target_path).expect("create target root");
        let target = pin_directory(&target_path, true).expect("pin target root");
        let executable = target_path.join("perf_matrix");
        fs::write(&executable, b"held benchmark bytes").expect("write benchmark");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))
            .expect("make benchmark executable");
        let handle = File::from(
            open(
                &executable,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .expect("open held benchmark"),
        );
        let identity = checked_benchmark_identity(&handle).expect("benchmark identity");
        let executable_metadata =
            fs::symlink_metadata(&executable).expect("benchmark path metadata");
        assert_eq!(
            identity,
            FileIdentity {
                device: executable_metadata.dev(),
                inode: executable_metadata.ino(),
            },
            "held benchmark and path metadata must use one device/inode representation"
        );
        let held_sha = sha256_open_file(&handle).expect("held benchmark digest");
        verify_benchmark_path(&executable, &identity, &target).expect("original benchmark path");

        let displaced_executable = target_path.join("perf_matrix-displaced");
        fs::rename(&executable, &displaced_executable).expect("displace benchmark path");
        fs::write(&executable, b"replacement benchmark bytes").expect("write replacement");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))
            .expect("make replacement executable");
        assert!(verify_benchmark_path(&executable, &identity, &target).is_err());
        assert_eq!(
            sha256_open_file(&handle).expect("held digest after replacement"),
            held_sha
        );
    }

    #[test]
    fn lease_path_is_host_global_while_receipt_identity_tracks_family() {
        assert_eq!(
            stable_lease_id(HardwareClassId::TrjZen35995wx).unwrap(),
            "trj-zen3-exclusive"
        );
        assert_eq!(
            stable_lease_id(HardwareClassId::M4Macos).unwrap(),
            "m4-macos-exclusive"
        );
        assert_eq!(
            stable_lease_path(HardwareClassId::TrjZen35995wx).unwrap(),
            PathBuf::from("/tmp/frankensearch-perf-host-global-exclusive.lock")
        );
        assert_eq!(
            stable_lease_path(HardwareClassId::M4Macos).unwrap(),
            PathBuf::from("/tmp/frankensearch-perf-host-global-exclusive.lock")
        );
        validate_canonical_lease_parent(
            &stable_lease_path(HardwareClassId::TrjZen35995wx)
                .expect("canonical Threadripper lease"),
        )
        .expect("root-owned sticky /tmp lease parent");
    }

    #[test]
    fn lease_rejects_permissive_or_non_owner_substitutable_inode() {
        let directory = tempfile::tempdir().expect("lease mode test directory");
        let lease_path = directory.path().join("permissive.lock");
        fs::write(&lease_path, b"lease").expect("write permissive lease");
        fs::set_permissions(&lease_path, fs::Permissions::from_mode(0o666))
            .expect("set permissive lease mode");
        let error =
            acquire_family_lease(&lease_path).expect_err("permissive lease inode must reject");
        assert!(error.to_string().contains("effective-user-owned 0600"));
    }

    #[test]
    fn lease_process_helper() {
        let Some(path) = std::env::var_os("QUILL_PERF_TEST_LEASE_PATH") else {
            return;
        };
        let (_lease, _identity) =
            acquire_family_lease(Path::new(&path)).expect("helper acquires family lease");
        println!("lease-ready");
        std::io::stdout().flush().expect("flush helper readiness");
        let mut release = Vec::new();
        std::io::stdin()
            .read_to_end(&mut release)
            .expect("wait for parent release");
    }

    #[test]
    fn lease_inherited_fd_child_helper() {
        if std::env::var_os("QUILL_PERF_TEST_INHERITED_LEASE_CHILD").is_none() {
            return;
        }
        println!("inherited-lease-ready");
        std::io::stdout().flush().expect("flush helper readiness");
        let mut release = Vec::new();
        std::io::stdin()
            .read_to_end(&mut release)
            .expect("wait for parent release");
    }

    #[test]
    fn lease_inherited_fd_producer_helper() {
        let Some(lease_path) = std::env::var_os("QUILL_PERF_TEST_INHERITED_LEASE_PRODUCER_PATH")
        else {
            return;
        };
        let lease_path = PathBuf::from(lease_path);
        let (lease, _identity) = acquire_family_lease(&lease_path).expect("acquire parent lease");
        let current_test = std::env::current_exe().expect("current test executable");
        let helper_name = "local_perf_runner::tests::lease_inherited_fd_child_helper";
        let mut child = Command::new(&current_test)
            .args(["--exact", helper_name, "--nocapture"])
            .env("QUILL_PERF_TEST_INHERITED_LEASE_CHILD", "1")
            .env_remove("QUILL_PERF_TEST_INHERITED_LEASE_PRODUCER_PATH")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn inherited-lease child");
        let mut child_output = BufReader::new(child.stdout.take().expect("child stdout"));
        loop {
            let mut line = String::new();
            assert_ne!(
                child_output
                    .read_line(&mut line)
                    .expect("read inherited child readiness"),
                0,
                "child exited before inheriting the lease"
            );
            if line.trim() == "inherited-lease-ready" {
                break;
            }
        }

        drop(lease);
        assert!(
            acquire_family_lease(&lease_path).is_err(),
            "contender acquired while inherited noisy-child lease remained open"
        );

        drop(child.stdin.take());
        assert!(child.wait().expect("wait for inherited child").success());
        let (_reacquired, _identity) =
            acquire_family_lease(&lease_path).expect("lease released after child exit");
    }

    #[test]
    fn family_lease_excludes_a_second_process_before_work_begins() {
        let directory = tempfile::tempdir().expect("lease test directory");
        let lease_path = directory.path().join("family.lock");
        let current_test = std::env::current_exe().expect("current test executable");
        let helper_name = "local_perf_runner::tests::lease_process_helper";
        let mut holder = Command::new(&current_test)
            .args(["--exact", helper_name, "--nocapture"])
            .env("QUILL_PERF_TEST_LEASE_PATH", &lease_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn lease holder");
        let mut holder_output = BufReader::new(holder.stdout.take().expect("holder stdout"));
        loop {
            let mut line = String::new();
            assert_ne!(
                holder_output
                    .read_line(&mut line)
                    .expect("read holder readiness"),
                0,
                "holder exited before acquiring lease"
            );
            if line.trim() == "lease-ready" {
                break;
            }
        }

        let contender = Command::new(&current_test)
            .args(["--exact", helper_name, "--nocapture"])
            .env("QUILL_PERF_TEST_LEASE_PATH", &lease_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("run lease contender");
        assert!(!contender.success(), "second process acquired held lease");

        drop(holder.stdin.take());
        assert!(holder.wait().expect("wait for lease holder").success());
        let mut remainder = String::new();
        holder_output
            .read_to_string(&mut remainder)
            .expect("drain holder output");
    }

    #[test]
    fn inherited_lease_survives_producer_exit_until_noisy_child_exits() {
        let directory = tempfile::tempdir().expect("lease test directory");
        let lease_path = directory.path().join("inherited-family.lock");
        let current_test = std::env::current_exe().expect("current test executable");
        let helper_name = "local_perf_runner::tests::lease_inherited_fd_producer_helper";
        // Keep the inheritable lease out of the parallel parent harness so unrelated
        // sibling test processes cannot extend its open-file-description lifetime.
        let helper = Command::new(&current_test)
            .args(["--exact", helper_name, "--nocapture", "--test-threads=1"])
            .env("QUILL_PERF_TEST_INHERITED_LEASE_PRODUCER_PATH", &lease_path)
            .stdin(Stdio::null())
            .status()
            .expect("run isolated inherited-lease producer");
        assert!(
            helper.success(),
            "isolated inherited-lease producer failed: {helper}"
        );
    }

    #[test]
    fn family_lease_rejects_links_and_path_identity_replacement() {
        let directory = tempfile::tempdir().expect("lease test directory");
        let source = directory.path().join("source.lock");
        fs::write(&source, b"lease").expect("write hard-link source");
        let hard_link = directory.path().join("hard-link.lock");
        fs::hard_link(&source, &hard_link).expect("create hard link");
        assert!(
            acquire_family_lease(&hard_link).is_err(),
            "multi-link lease admitted"
        );

        let symbolic_link = directory.path().join("symbolic-link.lock");
        std::os::unix::fs::symlink(&source, &symbolic_link).expect("create symbolic link");
        assert!(
            acquire_family_lease(&symbolic_link).is_err(),
            "symbolic lease admitted"
        );

        let replaceable = directory.path().join("replaceable.lock");
        let (_lease, identity) =
            acquire_family_lease(&replaceable).expect("acquire replaceable lease");
        let displaced = directory.path().join("displaced.lock");
        fs::rename(&replaceable, &displaced).expect("displace held lease path");
        fs::write(&replaceable, b"replacement").expect("write replacement lease path");
        fs::set_permissions(&replaceable, fs::Permissions::from_mode(0o600))
            .expect("restrict replacement lease mode");
        let error = verify_family_lease_path(&replaceable, &identity)
            .expect_err("replaced lease path must reject");
        assert!(error.to_string().contains("changed device or inode"));
    }

    #[test]
    fn typed_profiles_preserve_capacity_semantics_without_width_aliases() {
        let physical = profile(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        );
        let smt = profile(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128);
        assert_ne!(physical, smt);
        assert_eq!(physical.hardware_class_id(), HardwareClassId::TrjZen35995wx);
        assert_eq!(smt.hardware_class_id(), HardwareClassId::TrjZen35995wx);
        assert!(
            MachineProfileKey::new(HardwareClassId::M4Macos, ExecutionProfileId::Physical64)
                .is_err()
        );
        for obsolete_hardware_id in ["trj-zen3-1c", "trj-zen3-64c", "trj-zen3-64c-smt2"] {
            let json = format!(
                r#"{{"hardware_class_id":"{obsolete_hardware_id}","execution_profile_id":"physical-64"}}"#
            );
            assert!(
                serde_json::from_str::<MachineProfileKey>(&json).is_err(),
                "{obsolete_hardware_id} admitted"
            );
        }
    }

    #[test]
    fn typed_child_overrides_ambient_rayon_configuration_with_attested_budget() {
        let mut child = Command::new("unused-test-child");
        child
            .env("RAYON_NUM_THREADS", "64")
            .env("RAYON_RS_NUM_CPUS", "128")
            .env("RAYON_RS_NUM_THREADS", "256");
        let environment =
            BTreeMap::from([(OsString::from("RAYON_NUM_THREADS"), OsString::from("1"))]);
        configure_benchmark_child(&mut child, &environment);
        let environment = child
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().into_owned(),
                    value.map(|value| value.to_string_lossy().into_owned()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            environment.get("RAYON_NUM_THREADS"),
            Some(&Some("1".to_owned()))
        );
        assert!(!environment.contains_key("RAYON_RS_NUM_CPUS"));
        assert!(!environment.contains_key("RAYON_RS_NUM_THREADS"));
    }

    #[test]
    fn hostile_ambient_build_and_workload_knobs_fail_closed() {
        for name in [
            b"QUILL_PERF_FIXTURE".as_slice(),
            b"QUILL_PERF_RUNS",
            b"QUILL_PERF_WARMUP_ROUNDS",
            b"QUILL_PERF_BOOTSTRAP_SEED",
            b"RAYON_NUM_THREADS",
            b"CARGO_ENCODED_RUSTFLAGS",
            b"CARGO_PROFILE_RELEASE_LTO",
            b"RUSTFLAGS",
            b"RUSTC_WRAPPER",
            b"CC",
            b"CFLAGS",
            b"LDFLAGS",
            b"LD_PRELOAD",
            b"DYLD_INSERT_LIBRARIES",
            b"MALLOC_CONF",
            b"JEMALLOC_SYS_WITH_MALLOC_CONF",
        ] {
            assert!(
                ambient_variable_is_forbidden(name),
                "hostile variable admitted: {}",
                String::from_utf8_lossy(name)
            );
        }
        assert!(!ambient_variable_is_forbidden(b"CARGO_TARGET_DIR"));
        assert!(!ambient_variable_is_forbidden(
            b"QUILL_PERF_HELD_PRODUCER_FD"
        ));
        assert!(!ambient_variable_is_forbidden(b"PATH"));
    }

    #[test]
    fn controlled_build_binds_the_exact_hashed_rustc_path() {
        let rustc = ResolvedTool {
            path: PathBuf::from("/registered/toolchain/bin/rustc"),
            sha256: "a".repeat(64),
        };
        let mut environment = BTreeMap::new();
        bind_build_rustc(&mut environment, &rustc);
        assert_eq!(
            environment.get(OsStr::new("RUSTC")),
            Some(&rustc.path.into_os_string())
        );
    }

    #[test]
    fn cargo_ancestor_configuration_is_rejected_before_build() {
        let candidates = cargo_config_candidates(
            Path::new("/registered/repository"),
            Path::new("/home/perf/.cargo"),
        );
        for hostile in [
            Path::new("/registered/repository/.cargo/config"),
            Path::new("/registered/.cargo/config.toml"),
            Path::new("/.cargo/config"),
            Path::new("/home/perf/.cargo/config.toml"),
        ] {
            assert!(
                candidates.iter().any(|candidate| candidate == hostile),
                "Cargo hierarchy omitted {}",
                hostile.display()
            );
            let error = reject_cargo_config_candidates_with(&candidates, |candidate| {
                Ok(candidate == hostile)
            })
            .expect_err("hostile Cargo configuration must reject");
            assert!(error.to_string().contains("rejects Cargo configuration"));
        }
        reject_cargo_config_candidates_with(&candidates, |_| Ok(false))
            .expect("config-free Cargo hierarchy");
    }

    #[test]
    fn artifact_directory_path_uses_only_traversable_platform_form() {
        let canonical = Path::new("/var/tmp/frankensearch-run/artifacts");
        assert_eq!(
            benchmark_artifact_directory_path_for_os(canonical, 17, "linux")
                .expect("Linux descriptor directory"),
            PathBuf::from("/proc/self/fd/17")
        );
        assert_eq!(
            benchmark_artifact_directory_path_for_os(canonical, 17, "macos")
                .expect("macOS canonical directory"),
            canonical
        );
        assert!(benchmark_artifact_directory_path_for_os(canonical, 17, "windows").is_err());
    }

    #[test]
    fn producer_identity_rejects_stale_dirty_lock_and_substitute_executable_claims() {
        let source = CleanSourceSnapshot {
            revision: "a".repeat(40),
            cargo_lock_sha256: "b".repeat(64),
        };
        let valid = RunnerProducer {
            contract_version: LOCAL_PERF_PRODUCER_CONTRACT_VERSION.to_owned(),
            source_git_revision: source.revision.clone(),
            source_git_dirty: false,
            cargo_lock_sha256: source.cargo_lock_sha256.clone(),
            executable_sha256: "c".repeat(64),
        };
        validate_producer_against_source(&valid, &source).expect("matching producer identity");
        validate_expected_producer_executable(&valid, &valid.executable_sha256)
            .expect("launcher-held digest matches runtime handle");
        assert!(
            validate_expected_producer_executable(&valid, &"f".repeat(64)).is_err(),
            "substituted valid producer digest admitted"
        );

        let mut stale = valid.clone();
        stale.source_git_revision = "d".repeat(40);
        assert!(validate_producer_against_source(&stale, &source).is_err());

        let mut dirty = valid.clone();
        dirty.source_git_dirty = true;
        assert!(validate_producer_against_source(&dirty, &source).is_err());

        let mut wrong_lock = valid.clone();
        wrong_lock.cargo_lock_sha256 = "e".repeat(64);
        assert!(validate_producer_against_source(&wrong_lock, &source).is_err());

        let mut substitute = valid;
        substitute.executable_sha256 = "not-the-executing-elf".to_owned();
        assert!(validate_producer_against_source(&substitute, &source).is_err());
    }

    #[test]
    fn producer_contract_is_closed_and_hashes_the_held_executable() {
        let executing_image = open_executing_image().expect("open executing test image");
        let executing_sha256 =
            sha256_open_file(&executing_image).expect("hash executing test image");
        let contract =
            local_perf_producer_contract_json(&executing_sha256).expect("strict producer contract");
        let value = serde_json::from_str::<serde_json::Value>(&contract).expect("contract JSON");
        assert_eq!(
            value.as_object().map(|object| object.len()),
            Some(3),
            "top-level producer contract is closed"
        );
        assert_eq!(
            value["producer"].as_object().map(|object| object.len()),
            Some(5),
            "nested producer identity is closed"
        );
        assert_eq!(
            value["producer"]["contract_version"],
            LOCAL_PERF_PRODUCER_CONTRACT_VERSION
        );
        assert_eq!(value["producer"]["executable_sha256"], executing_sha256);
        assert!(local_perf_producer_contract_json("substitute").is_err());
    }

    #[test]
    fn build_identity_reexecutes_for_every_tracked_source_input() {
        let build_script = include_str!("../build.rs");
        assert!(build_script.contains(r#"["ls-files", "-z"]"#));
        assert!(build_script.contains(r#""--porcelain=v1", "--untracked-files=all""#));
        assert!(build_script.contains("packed-refs"));
        assert!(build_script.contains("byte.is_ascii_control()"));
        assert!(build_script.contains("must not inject Cargo line-protocol controls"));
        assert!(
            !build_script.contains("rerun-if-changed={repository"),
            "repository-wide directory watching would include target and operational churn"
        );
    }

    #[test]
    fn linux_cpu_list_parser_expands_ranges_without_aliasing() {
        assert_eq!(
            parse_cpu_list("0-3,64-67").unwrap(),
            vec![0, 1, 2, 3, 64, 65, 66, 67]
        );
        assert_eq!(parse_cpu_list("7").unwrap(), vec![7]);
        for rejected in ["", "3-2", "1,1", "1-3,3", "x"] {
            assert!(parse_cpu_list(rejected).is_err(), "{rejected:?} admitted");
        }
    }

    #[test]
    fn cpuinfo_parser_retains_only_complete_processor_records() {
        let records = parse_cpuinfo(
            "processor : 0\nphysical id : 0\ncore id : 0\n\n\
             processor : 1\nphysical id : 0\ncore id : 1\n\n\
             flags : ignored\n",
        );
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].get("processor").map(String::as_str), Some("0"));
        assert_eq!(records[1].get("core id").map(String::as_str), Some("1"));
    }

    #[test]
    fn thermal_samples_fail_closed_and_report_observed_pressure() {
        assert!(thermal_pressure_from_samples(&[]).is_err());
        assert!(!thermal_pressure_from_samples(&[(65_000, vec![80_000, 95_000])]).unwrap());
        assert!(thermal_pressure_from_samples(&[(80_000, vec![80_000, 95_000])]).unwrap());
    }

    #[test]
    fn process_receipts_cover_completed_and_every_supported_failure_outcome() {
        let bound = b"exact completed bound evidence";
        let outcomes = [
            LocalPerfAttemptOutcome::Completed,
            LocalPerfAttemptOutcome::SpawnRejected {
                error_kind: LocalPerfIoErrorKind::PermissionDenied,
            },
            LocalPerfAttemptOutcome::WaitRecoveredByKill {
                error_kind: LocalPerfIoErrorKind::Other,
            },
            LocalPerfAttemptOutcome::ExitedNonzero { code: 17 },
            LocalPerfAttemptOutcome::Signaled { signal: 9 },
            LocalPerfAttemptOutcome::UnknownTerminal,
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::EndPlatformCapture,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::FinishedTimestamp,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::RootProcessIdentity,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::PostRunIdentity,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::ArtifactRead,
            },
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::ArtifactVerification,
            },
        ];
        for outcome in outcomes {
            let completed = outcome == LocalPerfAttemptOutcome::Completed;
            let (receipt, bytes, run_log) =
                attempt_fixture(outcome, completed.then_some(bound.as_slice()));
            assert_eq!(receipt.outcome(), outcome);
            assert_eq!(receipt.applicability_plan().gate, PerfGate::Qg1);
            assert!(receipt.fixture_selector().is_none());
            assert!(!receipt.selected_cell_ids().is_empty());
            assert_eq!(receipt.run_id(), "attempt-1");
            assert_eq!(receipt.run_window(), "window-1");
            let expected_log_sha256 = sha256_hex(&run_log);
            assert_eq!(receipt.run_log_sha256(), Some(expected_log_sha256.as_str()));
            assert_eq!(
                receipt.exact_sha256().expect("exact receipt digest"),
                sha256_hex(&bytes)
            );
            receipt.verify_run_log(&run_log).expect("exact run log");
            let lifecycle = receipt.process_lifecycle();
            assert!(lifecycle.spawn_attempted());
            assert_eq!(
                lifecycle.spawn_succeeded(),
                !matches!(outcome, LocalPerfAttemptOutcome::SpawnRejected { .. })
            );
            assert_eq!(lifecycle.wait_completed(), lifecycle.spawn_succeeded());
            assert_eq!(lifecycle.child_reaped(), lifecycle.wait_completed());
            assert!(lifecycle.run_log_synced());
            assert!(lifecycle.run_log_captured());
            assert_eq!(receipt.unsupported_controls().len(), 2);
            assert_eq!(
                receipt.internal_lifecycle_gaps().actual_work(),
                receipt.internal_lifecycle_gaps().queue()
            );
            assert_eq!(
                receipt.internal_lifecycle_gaps().queue(),
                receipt.internal_lifecycle_gaps().workers_joined()
            );
            assert_eq!(
                receipt.internal_lifecycle_gaps().workers_joined(),
                receipt.internal_lifecycle_gaps().feed_drained()
            );
            assert_eq!(
                receipt.internal_lifecycle_gaps().feed_drained(),
                receipt.internal_lifecycle_gaps().pending_zero()
            );
            if completed {
                assert_eq!(receipt.retry(), LocalPerfRetryPredicate::NotRequired);
                assert!(receipt.runner_receipt_sha256().is_some());
                assert!(receipt.runner_artifact_manifest_sha256().is_some());
                let expected_bound_sha256 = sha256_hex(bound);
                assert_eq!(
                    receipt.bound_evidence_sha256(),
                    Some(expected_bound_sha256.as_str())
                );
                assert!(
                    receipt.verify_bound_evidence(bound).is_err(),
                    "a digest-matching non-artifact must still fail strict evidence verification"
                );
                assert!(receipt.verify_bound_evidence(b"substitute").is_err());
            } else {
                assert!(receipt.bound_evidence_sha256().is_none());
                assert!(receipt.runner_receipt_sha256().is_none());
                assert!(receipt.runner_artifact_manifest_sha256().is_none());
                assert!(receipt.verify_bound_evidence(bound).is_err());
            }
        }
    }

    #[test]
    fn process_receipt_rejects_resealed_semantic_tamper() {
        let bound = b"exact completed bound evidence";
        let (receipt, _, _) = attempt_fixture(LocalPerfAttemptOutcome::Completed, Some(bound));

        let mut retry_tamper = receipt.clone();
        retry_tamper.retry = LocalPerfRetryPredicate::DiagnoseUnknownTerminal;
        let bytes = seal_attempt_receipt(retry_tamper).expect("reseal retry tamper");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut plan_tamper = receipt.clone();
        plan_tamper
            .selected_cell_ids
            .push(plan_tamper.selected_cell_ids[0].clone());
        let bytes = seal_attempt_receipt(plan_tamper).expect("reseal plan tamper");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut selector_tamper = receipt.clone();
        selector_tamper.fixture_selector = Some("bulk/medium/8/on".to_owned());
        let bytes = seal_attempt_receipt(selector_tamper).expect("reseal selector tamper");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut binding_tamper = receipt.clone();
        binding_tamper.bound_evidence_sha256 = None;
        let bytes = seal_attempt_receipt(binding_tamper).expect("reseal binding tamper");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut missing_end = receipt.clone();
        missing_end.execution_end = None;
        missing_end.end_capture_error = Some("capture failed".to_owned());
        let bytes = seal_attempt_receipt(missing_end).expect("reseal missing completed end");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut unverified_identity = receipt.clone();
        unverified_identity.post_run_identity_verified = false;
        unverified_identity.post_run_identity_error = Some("identity failed".to_owned());
        let bytes =
            seal_attempt_receipt(unverified_identity).expect("reseal unverified completion");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut unreaped_completion = receipt.clone();
        unreaped_completion.process_lifecycle.wait_completed = false;
        unreaped_completion.process_lifecycle.child_reaped = false;
        let bytes = seal_attempt_receipt(unreaped_completion).expect("reseal unreaped completion");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut timestamp_fallback = receipt.clone();
        timestamp_fallback.finished_at_utc = timestamp_fallback.started_at_utc.clone();
        timestamp_fallback.finished_timestamp_error = Some("clock failed".to_owned());
        let bytes = seal_attempt_receipt(timestamp_fallback).expect("reseal clock fallback");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut timestamp_tamper = receipt.clone();
        timestamp_tamper.finished_at_utc = "0001-01-01T00:00:00Z".to_owned();
        let bytes = seal_attempt_receipt(timestamp_tamper).expect("reseal timestamp tamper");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut window_tamper = receipt;
        window_tamper.run_window = "x".repeat(MAX_IDENTITY_COMPONENT_BYTES + 1);
        let bytes = seal_attempt_receipt(window_tamper).expect("reseal window tamper");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        for outcome in [
            LocalPerfAttemptOutcome::ExitedNonzero { code: 0 },
            LocalPerfAttemptOutcome::ExitedNonzero { code: -1 },
            LocalPerfAttemptOutcome::Signaled { signal: 0 },
            LocalPerfAttemptOutcome::Signaled { signal: 256 },
        ] {
            assert!(attempt_derived_facts(outcome).is_err());
        }
    }

    #[test]
    fn log_sync_and_capture_failures_are_typed_without_false_durability_claims() {
        let (sync_failure, _, run_log) = attempt_fixture(
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::RunLogSync,
            },
            None,
        );
        assert!(!sync_failure.process_lifecycle().run_log_synced());
        assert!(sync_failure.process_lifecycle().run_log_captured());
        sync_failure
            .verify_run_log(&run_log)
            .expect("captured but unsynced diagnostic log bytes");

        let (read_failure, _, run_log) = attempt_fixture(
            LocalPerfAttemptOutcome::PostExitRejected {
                stage: LocalPerfRejectionStage::RunLogRead,
            },
            None,
        );
        assert!(read_failure.process_lifecycle().run_log_synced());
        assert!(!read_failure.process_lifecycle().run_log_captured());
        assert!(read_failure.run_log_sha256().is_none());
        assert!(read_failure.verify_run_log(&run_log).is_err());

        let mut contradictory_read_failure = read_failure;
        contradictory_read_failure.process_lifecycle.run_log_synced = false;
        let bytes = seal_attempt_receipt(contradictory_read_failure)
            .expect("reseal contradictory run-log read failure");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());
    }

    #[test]
    fn completed_receipt_rejects_nested_runner_substitution_even_after_outer_reseal() {
        let bound = b"exact completed bound evidence";
        let (receipt, _, _) = attempt_fixture(LocalPerfAttemptOutcome::Completed, Some(bound));
        let identity = attempt_runner_identity();
        receipt
            .verify_completed_runner_identity(&identity)
            .expect("exact admitted nested runner identity");
        assert_eq!(
            receipt.runner_receipt_sha256(),
            Some(identity.receipt_sha256())
        );
        assert_eq!(
            receipt.runner_artifact_manifest_sha256(),
            Some(
                identity
                    .artifact_manifest()
                    .expect("bound manifest")
                    .manifest_sha256()
            )
        );

        let mut build_substitution = receipt.clone();
        build_substitution.build.environment_sha256 = "d".repeat(64);
        let bytes = seal_attempt_receipt(build_substitution).expect("reseal build substitution");
        let resealed = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
            .expect("outer receipt remains internally canonical");
        assert!(
            resealed
                .verify_completed_runner_identity(&identity)
                .is_err()
        );

        let mut completion_timestamp_substitution = receipt.clone();
        completion_timestamp_substitution.finished_at_utc = "2026-07-31T23:59:59Z".to_owned();
        let bytes = seal_attempt_receipt(completion_timestamp_substitution)
            .expect("reseal completion timestamp substitution");
        let resealed = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
            .expect("outer receipt accepts a valid later finish timestamp");
        assert!(
            resealed
                .verify_completed_runner_identity(&identity)
                .is_err()
        );

        let mut nested_receipt_substitution = receipt.clone();
        nested_receipt_substitution.runner_receipt_sha256 = Some("d".repeat(64));
        let bytes = seal_attempt_receipt(nested_receipt_substitution)
            .expect("reseal nested receipt substitution");
        let resealed = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
            .expect("syntactically valid nested receipt digest");
        assert!(
            resealed
                .verify_completed_runner_identity(&identity)
                .is_err()
        );

        let mut nested_manifest_substitution = receipt;
        nested_manifest_substitution.runner_artifact_manifest_sha256 = Some("d".repeat(64));
        let bytes = seal_attempt_receipt(nested_manifest_substitution)
            .expect("reseal nested manifest substitution");
        let resealed = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
            .expect("syntactically valid nested manifest digest");
        assert!(
            resealed
                .verify_completed_runner_identity(&identity)
                .is_err()
        );
    }

    #[test]
    fn completed_and_failed_receipts_accept_only_the_exact_typed_fixture_subset() {
        let bound = b"exact completed bound evidence";
        let (receipt, _, _) = attempt_fixture(LocalPerfAttemptOutcome::Completed, Some(bound));
        let plan = physical_qg1_plan();
        let selected = first_runnable_qg1_selection(&plan);

        let mut partial = receipt.clone();
        partial.fixture_selector = selected.fixture.clone();
        partial.selected_cell_ids = selected.selected_cell_ids.clone();
        let bytes = seal_attempt_receipt(partial).expect("seal canonical partial receipt");
        let verified = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
            .expect("canonical runnable partial receipt");
        assert_eq!(verified.fixture_selector(), selected.fixture.as_deref());
        assert_eq!(verified.selected_cell_ids(), selected.selected_cell_ids);

        let mut duplicate = verified.clone();
        duplicate
            .selected_cell_ids
            .push(duplicate.selected_cell_ids[0].clone());
        let bytes = seal_attempt_receipt(duplicate).expect("reseal duplicate selection");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut noncanonical = verified.clone();
        noncanonical.selected_cell_ids[0] = "QG-1/not/a/canonical/cell".to_owned();
        let bytes = seal_attempt_receipt(noncanonical).expect("reseal noncanonical selection");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());

        let mut failed_partial = verified.clone();
        failed_partial.outcome = LocalPerfAttemptOutcome::ExitedNonzero { code: 17 };
        failed_partial.retry = LocalPerfRetryPredicate::DiagnoseNonzeroExit { code: 17 };
        failed_partial.internal_lifecycle_gaps = LocalPerfInternalLifecycleGaps {
            actual_work: LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
            queue: LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
            workers_joined: LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
            feed_drained: LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
            pending_zero: LocalPerfInternalLifecycleUnavailable::ChildDidNotCompleteSuccessfully,
        };
        failed_partial.bound_evidence_sha256 = None;
        failed_partial.runner_receipt_sha256 = None;
        failed_partial.runner_artifact_manifest_sha256 = None;
        let bytes = seal_attempt_receipt(failed_partial).expect("reseal failed partial");
        let failed = LocalPerfAttemptReceipt::from_verified_slice(&bytes)
            .expect("failed attempt keeps the exact typed fixture subset");
        assert_eq!(failed.fixture_selector(), selected.fixture.as_deref());
        assert_eq!(failed.selected_cell_ids(), selected.selected_cell_ids);

        let mut mismatched_failed = failed;
        mismatched_failed.fixture_selector = Some("bulk/medium/8/on".to_owned());
        let bytes = seal_attempt_receipt(mismatched_failed).expect("reseal mismatched failure");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());
    }

    #[test]
    fn typed_fixture_selection_is_exact_and_rejects_unknown_or_not_applicable() {
        let plan = physical_qg1_plan();
        let matrix = PerfMatrixSpec::complete();
        let selected = first_runnable_qg1_selection(&plan);
        let fixture = selected.fixture.as_deref().expect("partial fixture");
        assert!(
            matrix
                .for_gate(PerfGate::Qg1)
                .into_iter()
                .zip(&plan.cells)
                .filter(|(cell, _)| cell.fixture == fixture)
                .all(|(_, classification)| classification.applicability.is_runnable())
        );
        assert!(
            resolve_run_selection(
                &plan,
                Some(
                    &LocalPerfRunSelection::for_fixture("not/a/canonical/fixture")
                        .expect("syntactically valid unknown fixture"),
                ),
            )
            .is_err()
        );
        let prefix = fixture
            .rsplit_once('/')
            .map_or("not-a-fixture", |(prefix, _)| prefix);
        assert!(
            resolve_run_selection(
                &plan,
                Some(
                    &LocalPerfRunSelection::for_fixture(prefix)
                        .expect("syntactically valid fixture prefix"),
                ),
            )
            .is_err(),
            "fixture selection must never use substring or prefix matching"
        );

        let not_applicable_fixture = matrix
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .zip(&plan.cells)
            .find(|(_, classification)| {
                classification.applicability == PerfCellApplicability::NotApplicable
            })
            .map(|(cell, _)| cell.fixture.clone())
            .expect("physical-64 QG-1 must include a NotApplicable fixture");
        assert!(
            resolve_run_selection(
                &plan,
                Some(
                    &LocalPerfRunSelection::for_fixture(not_applicable_fixture)
                        .expect("syntactically valid NotApplicable fixture"),
                ),
            )
            .is_err()
        );
    }

    #[test]
    fn controlled_fixture_environment_replaces_ambient_and_full_gate_removes_it() {
        let plan = physical_qg1_plan();
        let selected = first_runnable_qg1_selection(&plan);
        let mut environment = BTreeMap::from([(
            OsString::from("QUILL_PERF_FIXTURE"),
            OsString::from("hostile-ambient-substring"),
        )]);
        apply_run_selection_environment(&mut environment, &selected);
        assert_eq!(
            environment
                .get(OsStr::new("QUILL_PERF_FIXTURE"))
                .and_then(|value| value.to_str()),
            selected.fixture.as_deref()
        );

        let full = resolve_run_selection(&plan, None).expect("full-gate selection");
        apply_run_selection_environment(&mut environment, &full);
        assert!(!environment.contains_key(OsStr::new("QUILL_PERF_FIXTURE")));
    }

    #[test]
    fn process_receipt_rejects_noncanonical_duplicate_unknown_and_corrupt_bytes() {
        let (_, bytes, _) =
            attempt_fixture(LocalPerfAttemptOutcome::ExitedNonzero { code: 17 }, None);
        let directory = tempfile::tempdir().expect("attempt receipt directory");
        let path = directory.path().join("attempt.json");
        fs::write(&path, &bytes).expect("persist attempt receipt");
        LocalPerfAttemptReceipt::load_verified(&path).expect("strict path reload");
        let receipt: LocalPerfAttemptReceipt =
            serde_json::from_slice(&bytes).expect("typed receipt");
        let pretty = serde_json::to_vec_pretty(&receipt).expect("pretty receipt");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&pretty).is_err());

        let text = std::str::from_utf8(&bytes).expect("receipt UTF-8");
        let duplicate = format!(
            "{{\"schema_version\":\"{}\",{}",
            LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION,
            &text[1..]
        );
        assert!(LocalPerfAttemptReceipt::from_verified_slice(duplicate.as_bytes()).is_err());

        let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("receipt value");
        value
            .as_object_mut()
            .expect("receipt object")
            .insert("unknown".to_owned(), serde_json::Value::Bool(true));
        let unknown = serde_json::to_vec(&value).expect("unknown-field bytes");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&unknown).is_err());

        let mut corrupt = bytes;
        let offset = corrupt.len() / 2;
        corrupt[offset] ^= 1;
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&corrupt).is_err());
    }

    #[test]
    fn failed_process_receipt_can_never_be_registry_promotion_evidence() {
        let (receipt, bytes, _) =
            attempt_fixture(LocalPerfAttemptOutcome::ExitedNonzero { code: 17 }, None);
        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let context = MachineClassAdmissionContext {
            gate: receipt.gate().to_owned(),
            expected_profile: receipt.profile(),
            destination_basename: receipt
                .profile()
                .latest_basename(receipt.gate())
                .expect("typed latest destination"),
        };
        let write_count = std::cell::Cell::new(0_u64);
        assert!(
            registry
                .admit_then(&bytes, &context, |_| write_count.set(write_count.get() + 1))
                .is_err()
        );
        assert_eq!(write_count.get(), 0);
    }

    #[test]
    fn completed_shard_syncs_inputs_then_publishes_verified_attempt_evidence_pair_last() {
        let source = production_source();
        let child_inputs_durable = unique_marker_offset(
            source,
            "let durable_child_artifacts = match read_and_sync_child_artifacts(",
        );
        let nested_runner = unique_marker_offset(source, "let receipt = RunnerReceipt {");
        let bound_write = unique_marker_offset(
            source,
            "bound_evidence_name,\n        &bound_evidence_bytes,",
        );
        let bound_reload =
            unique_marker_offset(source, "let persisted_bound = match read_file_at(");
        let final_attempt_publish = unique_marker_offset(
            source,
            "&attempt_pending_name,\n        &attempt_name,\n        &completed_attempt_bytes,",
        );
        let final_pair_reload = unique_marker_offset(
            source,
            "let persisted_attempt =\n        read_file_at(&run_directories.run.handle",
        );
        assert!(child_inputs_durable < nested_runner);
        assert!(nested_runner < bound_write);
        assert!(bound_write < bound_reload);
        assert!(bound_reload < final_attempt_publish);
        assert!(final_attempt_publish < final_pair_reload);
    }

    #[test]
    fn child_artifact_durability_boundary_reads_and_syncs_exact_owned_files() {
        let directory = tempfile::tempdir().expect("child artifact directory");
        fs::write(directory.path().join("QG-1.json"), b"threshold").expect("threshold fixture");
        fs::write(directory.path().join("QG-1.evidence.json"), b"evidence")
            .expect("evidence fixture");
        let handle = File::open(directory.path()).expect("directory handle");
        let durable = read_and_sync_child_artifacts(&handle, "QG-1.json", "QG-1.evidence.json")
            .expect("durable exact child artifacts");
        assert_eq!(durable.threshold_bytes, b"threshold");
        assert_eq!(durable.evidence_bytes, b"evidence");
        assert_eq!(
            read_and_sync_child_artifacts(&handle, "missing", "QG-1.evidence.json")
                .expect_err("missing threshold must fail closed"),
            LocalPerfRejectionStage::ArtifactRead
        );
    }

    #[test]
    fn attempt_receipt_publication_is_atomic_and_never_replaces_a_final_name() {
        let directory = tempfile::tempdir().expect("attempt publication directory");
        let handle = File::open(directory.path()).expect("directory handle");
        atomically_publish_new_sync_at(
            &handle,
            "QG-1.attempt.pending",
            "QG-1.attempt.json",
            b"first receipt",
        )
        .expect("first atomic attempt publication");
        assert!(!directory.path().join("QG-1.attempt.pending").exists());
        assert_eq!(
            fs::read(directory.path().join("QG-1.attempt.json")).expect("published receipt"),
            b"first receipt"
        );
        assert!(
            atomically_publish_new_sync_at(
                &handle,
                "QG-1.second.pending",
                "QG-1.attempt.json",
                b"replacement",
            )
            .is_err()
        );
        assert_eq!(
            fs::read(directory.path().join("QG-1.attempt.json")).expect("original receipt"),
            b"first receipt"
        );
    }

    #[test]
    fn wait_error_cannot_publish_a_terminal_receipt_before_bounded_kill_and_reap() {
        let source = production_source();
        let wait_start = unique_marker_offset(
            source,
            "let (status, recovered_wait_error) = match child.wait()",
        );
        let wait_tail = &source[wait_start..];
        let log_capture = wait_start
            + unique_marker_offset(
                wait_tail,
                "let run_log_synced = run_log_sync.sync_all().is_ok();",
            );
        let wait_slice = &source[wait_start..log_capture];
        assert_eq!(
            wait_slice
                .matches("force_kill_and_reap(&mut child, root_process_identity)")
                .count(),
            1
        );
        assert_eq!(
            wait_slice
                .matches("LocalPerfRunError::UnreapedChild")
                .count(),
            1
        );
        assert_eq!(
            wait_slice.matches("write_failed_attempt_receipt(").count(),
            0
        );

        let recovered = LocalPerfAttemptOutcome::WaitRecoveredByKill {
            error_kind: LocalPerfIoErrorKind::Other,
        };
        let reaped = LocalPerfProcessLifecycle {
            spawn_attempted: true,
            spawn_succeeded: true,
            wait_completed: true,
            child_reaped: true,
            run_log_synced: true,
            run_log_captured: true,
            process_group_recovery: LocalPerfProcessGroupRecovery::SignaledOwnedGroup,
        };
        assert!(validate_process_lifecycle(recovered, reaped, true).is_ok());
        assert_eq!(
            reaped.process_tree_quiescence(),
            LocalPerfProcessTreeQuiescence::DirectChildOnly,
            "a reaped direct child must never be relabeled as descendant-tree quiescence"
        );
        let mut unreaped = reaped;
        unreaped.wait_completed = false;
        unreaped.child_reaped = false;
        assert!(validate_process_lifecycle(recovered, unreaped, true).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_root_birth_identity_uses_the_final_proc_comm_delimiter() {
        let stat = "37 (benchmark) worker)) R 1 37 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 81";
        assert_eq!(parse_linux_proc_process_identity(stat), Some((37, 81)));
        assert_eq!(
            parse_linux_proc_process_identity("37 (benchmark) R 1 2"),
            None
        );
    }

    #[test]
    fn root_birth_identity_is_captured_before_the_child_can_be_reaped() {
        let source = production_source();
        let spawned = unique_marker_offset(
            source,
            "let (mut child, root_process_identity) = match child.spawn()",
        );
        let captured = unique_marker_offset(
            source,
            "let root_process_identity = capture_root_process_identity(&child);",
        );
        let waited = unique_marker_offset(
            source,
            "let (status, recovered_wait_error) = match child.wait()",
        );
        assert!(spawned < captured);
        assert!(captured < waited);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn configured_benchmark_child_leads_a_dedicated_process_group() {
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "exec sleep 60"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_benchmark_child(&mut command, &BTreeMap::new());
        let mut child = command
            .spawn()
            .expect("spawn dedicated process-group child");
        let root_process_identity = capture_root_process_identity(&child);
        assert!(matches!(
            root_process_identity,
            LocalPerfRootProcessIdentity::LinuxProcStartTime {
                pid,
                process_group_id,
                ..
            } if pid == child.id() && process_group_id == pid
        ));
        let (_, recovery) = force_kill_and_reap(&mut child, root_process_identity)
            .expect("reap dedicated process-group child");
        assert_eq!(recovery, LocalPerfProcessGroupRecovery::SignaledOwnedGroup);
    }

    #[test]
    fn completed_receipt_rejects_an_unverifiable_root_pid() {
        let (receipt, _, _) = attempt_fixture(LocalPerfAttemptOutcome::Completed, Some(b"bound"));
        let mut receipt = receipt;
        receipt.root_process_identity = LocalPerfRootProcessIdentity::Unverifiable { pid: 37 };
        let bytes = seal_attempt_receipt(receipt).expect("reseal root-identity mutation");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());
    }

    #[test]
    fn completed_receipt_rejects_a_non_dedicated_root_process_group() {
        let (receipt, _, _) = attempt_fixture(LocalPerfAttemptOutcome::Completed, Some(b"bound"));
        let mut receipt = receipt;
        receipt.root_process_identity = LocalPerfRootProcessIdentity::LinuxProcStartTime {
            pid: 37,
            process_group_id: 36,
            start_time_ticks: 81,
        };
        let bytes = seal_attempt_receipt(receipt).expect("reseal process-group mutation");
        assert!(LocalPerfAttemptReceipt::from_verified_slice(&bytes).is_err());
    }

    #[test]
    fn bounded_wait_recovery_forces_a_real_child_to_a_reaped_terminal_status() {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "exec sleep 60"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn wait-recovery child");
        let child_pid = child.id();
        let (status, recovery) = force_kill_and_reap(
            &mut child,
            LocalPerfRootProcessIdentity::Unverifiable { pid: child_pid },
        )
        .expect("bounded kill and reap");
        assert!(!status.success());
        assert_eq!(recovery, LocalPerfProcessGroupRecovery::DirectChildFallback);
        assert!(child.try_wait().expect("post-recovery try_wait").is_some());
    }

    #[test]
    fn bounded_diagnostic_preserves_utf8_at_multibyte_boundary() {
        let error = LocalPerfRunError::Invalid(format!("{}é-tail", "a".repeat(239)));
        let bounded = bounded_diagnostic(&error);
        assert!(bounded.len() <= 240);
        assert!(std::str::from_utf8(bounded.as_bytes()).is_ok());
    }

    #[test]
    fn utc_timestamp_validation_rejects_impossible_calendar_and_clock_values() {
        validate_utc_timestamp("2024-02-29T23:59:59Z", "leap timestamp")
            .expect("valid leap-day UTC timestamp");
        for invalid in [
            "0000-01-01T00:00:00Z",
            "2026-00-01T00:00:00Z",
            "2026-13-01T00:00:00Z",
            "2026-01-00T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-02-29T00:00:00Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:60:00Z",
            "2026-01-01T00:00:60Z",
        ] {
            assert!(validate_utc_timestamp(invalid, "hostile timestamp").is_err());
        }
    }
}
