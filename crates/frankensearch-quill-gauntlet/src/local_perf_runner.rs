//! Typed registered-host performance runner.
//!
//! This producer owns one canonical host-global lease across benchmark
//! compilation, start/end probes, and the measured child. It emits the required
//! receipt last, only after the child exits successfully and every exact
//! artifact re-verifies.

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
use std::process::{Command, Stdio};

use rustix::fs::{FileType, FlockOperation, Mode, OFlags, flock, fstat, mkdirat, open, openat};
use rustix::io::{FdFlags, fcntl_getfd, fcntl_setfd};
use rustix::process::geteuid;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::machine_class_registry::{
    LOCAL_PERF_PRODUCER_CONTRACT_VERSION, MACHINE_CLASS_REGISTRY_SHA256,
    MachineClassAdmissionContext, MachineClassError, MachineClassRegistry,
    RUNNER_RECEIPT_SCHEMA_VERSION, RunnerArtifactManifest, RunnerBuild, RunnerCompletion,
    RunnerDurability, RunnerExecution, RunnerExecutionRequest, RunnerExecutionSnapshot,
    RunnerHardware, RunnerProducer, RunnerReceipt, seal_runner_receipt, sha256_hex,
};
use crate::{
    EvidenceArtifactError, PerfEvidenceArtifact, PerfGate, PerfGateArtifact, PerfMatrixSpec,
    command_sha256_from_argv,
};

const PRODUCER_CONTRACT_SCHEMA_VERSION: &str =
    "frankensearch.quill-local-perf-producer-contract.v1";
const ATTEMPT_RECEIPT_SCHEMA_VERSION: &str = "frankensearch.perf-runner-attempt.v1";
const MAX_IDENTITY_COMPONENT_BYTES: usize = 96;
const MAX_OUTPUT_COMPONENT_BYTES: usize = 128;
const MIN_MEASUREMENT_RUNS: usize = 10;
const MAX_MEASUREMENT_RUNS: usize = 100;
const MEASUREMENT_WARMUP_ROUNDS: &str = "1";
const MEASUREMENT_BOOTSTRAP_SEED: &str = "5860671082138523204";
const EMBEDDED_PRODUCER_CONTRACT_VERSION: &str = env!("QUILL_PERF_PRODUCER_CONTRACT_VERSION");
const EMBEDDED_PRODUCER_GIT_REVISION: &str = env!("QUILL_PERF_PRODUCER_GIT_REVISION");
const EMBEDDED_PRODUCER_GIT_DIRTY: &str = env!("QUILL_PERF_PRODUCER_GIT_DIRTY");
const EMBEDDED_PRODUCER_CARGO_LOCK_SHA256: &str = env!("QUILL_PERF_PRODUCER_CARGO_LOCK_SHA256");

/// Complete registered-host invocation owned by the typed producer.
#[derive(Debug, Clone)]
pub struct LocalPerfRunConfig {
    /// Gate selected for this invocation.
    pub gate: PerfGate,
    /// Canonical registered machine class.
    pub class_id: String,
    /// Unique pass identity.
    pub run_id: String,
    /// Window shared by a candidate and its immediate rerun.
    pub run_window: String,
    /// Exact maximum thread width in the selected gate's frozen full matrix.
    pub thread_budget: u64,
    /// Predeclared measured block count; never inherited from ambient state.
    pub measurement_runs: usize,
    /// `not-applicable` or the currently admissible `p-plus-e`.
    pub apple_execution_mode: String,
    /// Unique not-yet-created output directory. Existing paths are rejected.
    pub output_dir: PathBuf,
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrecommitInventory {
    schema_version: String,
    gate: String,
    class_id: String,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum RunnerAttemptTermination {
    Exited { code: i64 },
    Signaled { signal: i32 },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunnerAttemptReceipt {
    schema_version: String,
    mode: String,
    gate: String,
    class_id: String,
    run_id: String,
    run_window: String,
    registry_sha256: String,
    hardware: RunnerHardware,
    execution_request: RunnerExecutionRequest,
    execution_start: RunnerExecutionSnapshot,
    execution_end: Option<RunnerExecutionSnapshot>,
    end_capture_error: Option<String>,
    build: RunnerBuild,
    post_run_identity_verified: bool,
    post_run_identity_error: Option<String>,
    termination: RunnerAttemptTermination,
    run_log_sha256: String,
    started_at_utc: String,
    finished_at_utc: String,
    seal_sha256: String,
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
/// synced. A nonzero or signaled child emits a separately sealed diagnostic
/// attempt receipt that can never be admitted as promotion evidence.
///
/// # Errors
///
/// Returns a typed failure for an unavailable lease, unsupported platform,
/// dirty/offloaded source, probe drift, child failure, malformed artifact, or
/// any self-admission mismatch.
pub fn run_local_perf_command(
    config: &LocalPerfRunConfig,
) -> Result<LocalPerfRunOutput, LocalPerfRunError> {
    validate_bounded_inputs(config)?;
    validate_platform_gate_policy(config)?;
    let lease_path = stable_lease_path(&config.class_id)?;
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
        destination_basename: format!("{}.{}.latest.json", config.gate.label(), config.class_id),
    };
    let durability = durability_for_run(config)?;
    let registry = MachineClassRegistry::frozen()?;
    let pre_spawn = registry.preflight(
        &config.class_id,
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
    let status = child.spawn()?.wait()?;
    run_log_sync.sync_all()?;
    let run_log_bytes = read_file_at(&run_directories.run.handle, "run.log")?;
    let exit_code = status.code().map_or(-1, i64::from);
    write_new_sync_at(
        &run_directories.run.handle,
        "exit-status",
        format!("{exit_code}\n").as_bytes(),
    )?;
    if !status.success() {
        let attempt_path = write_failed_attempt_receipt(
            config,
            &run_directories,
            &captured_build,
            &producer_before,
            &external_paths,
            &start,
            status,
            &run_log_bytes,
            &started_at_utc,
        )?;
        return Err(LocalPerfRunError::Invalid(format!(
            "benchmark child failed; sealed non-promotable attempt receipt preserved at {}",
            attempt_path.display()
        )));
    }

    let end = capture_platform(config)?;
    let finished_at_utc = utc_now()?;
    verify_prepared_build(&captured_build, &producer_before, &external_paths.target)?;
    if start.hardware != end.hardware
        || start.request != end.request
        || start.snapshot != end.snapshot
    {
        return Err(LocalPerfRunError::Invalid(
            "hardware, execution, or clean build identity drifted across the measured child"
                .to_owned(),
        ));
    }

    let threshold_name = format!("{}.json", config.gate.label());
    let evidence_name = format!("{}.evidence.json", config.gate.label());
    let threshold_bytes = read_file_at(&run_directories.artifacts.handle, &threshold_name)?;
    let evidence_bytes = read_file_at(&run_directories.artifacts.handle, &evidence_name)?;
    let threshold = read_canonical_threshold(&threshold_bytes)?;
    let evidence = read_canonical_evidence(&evidence_bytes)?;
    validate_child_artifacts(config, &captured_build, &start, &threshold, &evidence)?;

    let artifact_manifest = RunnerArtifactManifest::from_artifacts(
        config.gate.label(),
        &config.run_id,
        &config.run_window,
        &run_log_bytes,
        &threshold_bytes,
        &evidence_bytes,
    );
    let artifact_manifest_bytes = artifact_manifest.to_json_bytes()?;
    verify_prepared_build(&captured_build, &producer_before, &external_paths.target)?;
    let producer_identity_sha256 =
        sha256_hex(&serde_json::to_vec(&captured_build.receipt.producer)?);
    let receipt = RunnerReceipt {
        schema_version: RUNNER_RECEIPT_SCHEMA_VERSION.to_owned(),
        requested_class_id: config.class_id.clone(),
        derived_class_id: config.class_id.clone(),
        registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
        hardware: start.hardware,
        execution: RunnerExecution {
            request: start.request.clone(),
            start: start.snapshot.clone(),
            end: end.snapshot.clone(),
            identity_sha256: String::new(),
        },
        build: captured_build.receipt.clone(),
        durability,
        completion: RunnerCompletion {
            verified: true,
            exit_status: exit_code,
            run_log_sha256: sha256_hex(&run_log_bytes),
            artifact_manifest_sha256: sha256_hex(&artifact_manifest_bytes),
            artifact_digests_verified: true,
            started_at_utc,
            finished_at_utc,
        },
    };
    let receipt_bytes = seal_runner_receipt(receipt)?;
    let identity = registry.admit(&receipt_bytes, &context)?;
    pre_spawn.verify_final(&identity)?;
    let identity = identity.bind_artifact_manifest(
        &artifact_manifest_bytes,
        &run_log_bytes,
        &threshold_bytes,
        &evidence_bytes,
    )?;
    let mut bound_preview = evidence;
    let bound_evidence_bytes = bound_preview.bind_machine_class_identity_and_seal(
        identity,
        &threshold_bytes,
        &evidence_bytes,
    )?;
    PerfEvidenceArtifact::from_verified_slice(&bound_evidence_bytes)?;

    let manifest_path = config
        .output_dir
        .join(format!("{}.artifacts.json", config.gate.label()));
    let receipt_path = config
        .output_dir
        .join(format!("{}.runner.json", config.gate.label()));
    let inventory_path = config.output_dir.join("PRECOMMIT.json");
    let inventory = PrecommitInventory {
        schema_version: "frankensearch.perf-run-precommit.v3".to_owned(),
        gate: config.gate.label().to_owned(),
        class_id: config.class_id.clone(),
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
    let inventory_bytes = serde_json::to_vec_pretty(&inventory)?;

    // The receipt is the required commit boundary consumed by the ratchet.
    // Everything it binds must be durable before it is created. A crash before
    // the final write can leave diagnostics or PRECOMMIT.json, but cannot leave
    // a promotable run.
    let manifest_name = manifest_path
        .file_name()
        .ok_or_else(|| LocalPerfRunError::Invalid("manifest path has no basename".to_owned()))?;
    let inventory_name = inventory_path
        .file_name()
        .ok_or_else(|| LocalPerfRunError::Invalid("inventory path has no basename".to_owned()))?;
    let receipt_name = receipt_path
        .file_name()
        .ok_or_else(|| LocalPerfRunError::Invalid("receipt path has no basename".to_owned()))?;
    write_new_sync_at(
        &run_directories.run.handle,
        manifest_name,
        &artifact_manifest_bytes,
    )?;
    write_new_sync_at(
        &run_directories.run.handle,
        inventory_name,
        &inventory_bytes,
    )?;
    run_directories.run.handle.sync_all()?;
    verify_family_lease_path(&lease_path, &lease_identity)?;
    verify_external_paths(&external_paths)?;
    verify_run_directories(config, &run_directories)?;
    verify_prepared_build(&captured_build, &producer_before, &external_paths.target)?;
    verify_environment_policy(&run_directories.run.handle, &captured_build)?;
    write_new_sync_at(&run_directories.run.handle, receipt_name, &receipt_bytes)?;
    run_directories.run.handle.sync_all()?;
    verify_external_paths(&external_paths)?;
    verify_run_directories(config, &run_directories)?;

    drop(lease_file);
    Ok(LocalPerfRunOutput {
        run_log: run_log_path,
        artifact_manifest: manifest_path,
        environment_policy: environment_policy_path,
        runner_receipt: receipt_path,
        precommit_inventory: inventory_path,
    })
}

fn validate_config(config: &LocalPerfRunConfig) -> Result<ExternalRunPaths, LocalPerfRunError> {
    validate_platform_gate_policy(config)?;
    validate_external_paths(config)
}

fn validate_bounded_inputs(config: &LocalPerfRunConfig) -> Result<(), LocalPerfRunError> {
    for (name, value) in [
        ("class ID", config.class_id.as_str()),
        ("run ID", config.run_id.as_str()),
        ("run window", config.run_window.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(LocalPerfRunError::Invalid(format!("{name} is empty")));
        }
    }
    let normative_thread_budget = normative_thread_budget(config.gate)?;
    if config.thread_budget != normative_thread_budget {
        return Err(LocalPerfRunError::Invalid(format!(
            "{} requires exact normative thread budget {normative_thread_budget}, received {}",
            config.gate, config.thread_budget
        )));
    }
    if !(MIN_MEASUREMENT_RUNS..=MAX_MEASUREMENT_RUNS).contains(&config.measurement_runs) {
        return Err(LocalPerfRunError::Invalid(format!(
            "measurement runs must remain within {MIN_MEASUREMENT_RUNS}..={MAX_MEASUREMENT_RUNS}"
        )));
    }
    for (field, value) in [
        ("class ID", config.class_id.as_str()),
        ("run ID", config.run_id.as_str()),
        ("run window", config.run_window.as_str()),
    ] {
        validate_component(value, field)?;
    }
    Ok(())
}

fn normative_thread_budget(gate: PerfGate) -> Result<u64, LocalPerfRunError> {
    PerfMatrixSpec::complete()
        .max_thread_width(gate)
        .and_then(|threads| u64::try_from(threads).ok())
        .ok_or_else(|| {
            LocalPerfRunError::Invalid(format!("{} has no positive normative thread width", gate))
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
    if config.class_id == "m4-macos" {
        return Err(LocalPerfRunError::Invalid(
            "m4-macos promotion is unavailable until the producer can attest the actual executing image through a supported O_EXEC or loaded-image mechanism; every current M4 run is diagnostic-only"
                .to_owned(),
        ));
    } else if config.apple_execution_mode != "not-applicable" {
        return Err(LocalPerfRunError::Invalid(
            "non-M4 producer requires apple mode not-applicable".to_owned(),
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

fn checked_directory_identity(handle: &impl AsFd) -> Result<FileIdentity, LocalPerfRunError> {
    let stat = fstat(handle).map_err(std::io::Error::from)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::Directory {
        return Err(LocalPerfRunError::Invalid(
            "pinned run root is not a directory".to_owned(),
        ));
    }
    Ok(FileIdentity {
        device: stat.st_dev,
        inode: stat.st_ino,
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
    source: &CleanSourceSnapshot,
    target: &PinnedDirectory,
    artifact_dir: &Path,
) -> Result<ControlledEnvironments, LocalPerfRunError> {
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
        "RAYON_NUM_THREADS",
        OsStr::new(&config.thread_budget.to_string()),
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
    policy.insert("policy.class_id".to_owned(), config.class_id.clone());
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

fn checked_benchmark_identity(handle: &impl AsFd) -> Result<FileIdentity, LocalPerfRunError> {
    let stat = fstat(handle).map_err(std::io::Error::from)?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_nlink != 1
        || stat.st_uid != geteuid().as_raw()
        || stat.st_mode & 0o111 == 0
    {
        return Err(LocalPerfRunError::Invalid(
            "benchmark image must be an effective-user-owned executable regular single-link file"
                .to_owned(),
        ));
    }
    Ok(FileIdentity {
        device: stat.st_dev,
        inode: stat.st_ino,
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
    child.env_clear().envs(environment);
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

fn stable_lease_id(class_id: &str) -> Result<&'static str, LocalPerfRunError> {
    if class_id == "m4-macos" {
        Ok("m4-macos-exclusive")
    } else if parse_trj_class(class_id).is_ok() {
        Ok("trj-zen3-exclusive")
    } else {
        Err(LocalPerfRunError::Invalid(format!(
            "no stable exclusive-lease family exists for class {class_id:?}"
        )))
    }
}

fn stable_lease_path(class_id: &str) -> Result<PathBuf, LocalPerfRunError> {
    stable_lease_id(class_id)?;
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
    if config.apple_execution_mode != "not-applicable" {
        return Err(LocalPerfRunError::Invalid(
            "Linux producer requires apple mode not-applicable".to_owned(),
        ));
    }
    let class = parse_trj_class(&config.class_id)?;
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
        requested_logical_cpu_ids: observed_logical_cpu_ids.clone(),
        requested_physical_core_width: class.0,
        thread_budget: config.thread_budget,
        apple_execution_mode: config.apple_execution_mode.clone(),
    };
    let snapshot = RunnerExecutionSnapshot {
        observed_logical_cpu_ids,
        effective_physical_core_ids,
        cpu_assignment_observability: "affinity-enforced".to_owned(),
        effective_cpuset_sha256: String::new(),
        threads_per_core: class.1,
        smt_state: if class.1 == 2 { "on" } else { "off" }.to_owned(),
        numa_node_ids: vec![0],
        numa_policy: "bind:0".to_owned(),
        governor,
        thermal_pressure,
        exclusive_lease: true,
        exclusive_lease_id: stable_lease_id(&config.class_id)?.to_owned(),
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
    if config.class_id != "m4-macos" {
        return Err(LocalPerfRunError::Invalid(
            "macOS producer currently admits only m4-macos".to_owned(),
        ));
    }
    if config.apple_execution_mode != "p-plus-e" {
        return Err(LocalPerfRunError::Invalid(
            "M4 diagnostic capture recognizes only p-plus-e execution; promotion remains unavailable"
                .to_owned(),
        ));
    }
    let width = 14;
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
        requested_logical_cpu_ids: Vec::new(),
        requested_physical_core_width: width,
        thread_budget: config.thread_budget,
        apple_execution_mode: config.apple_execution_mode.clone(),
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
        exclusive_lease_id: stable_lease_id(&config.class_id)?.to_owned(),
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
    let artifact = serde_json::from_slice::<PerfGateArtifact>(bytes)?;
    if serde_json::to_vec_pretty(&artifact)? != bytes {
        return Err(LocalPerfRunError::Invalid(
            "threshold artifact is not exact canonical pretty JSON".to_owned(),
        ));
    }
    Ok(artifact)
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

fn parse_trj_class(class_id: &str) -> Result<(u64, u64), LocalPerfRunError> {
    let suffix = class_id.strip_prefix("trj-zen3-").ok_or_else(|| {
        LocalPerfRunError::Invalid(format!("Linux class {class_id:?} is not a TRJ class"))
    })?;
    let (width, threads_per_core) = suffix
        .strip_suffix("c-smt2")
        .map_or_else(
            || suffix.strip_suffix('c').map(|width| (width, 1)),
            |width| Some((width, 2)),
        )
        .ok_or_else(|| LocalPerfRunError::Invalid(format!("invalid TRJ class {class_id:?}")))?;
    let width = width
        .parse::<u64>()
        .map_err(|error| LocalPerfRunError::Invalid(format!("invalid TRJ class width: {error}")))?;
    if !(1..=64).contains(&width) {
        return Err(LocalPerfRunError::Invalid(format!(
            "TRJ class width {width} is outside 1..=64"
        )));
    }
    Ok((width, threads_per_core))
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

fn read_file_at(directory: &File, name: impl AsRef<Path>) -> Result<Vec<u8>, LocalPerfRunError> {
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
    Ok(bytes)
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
    directories: &RunDirectories,
    build: &CapturedBuild,
    producer: &ExecutingProducer,
    paths: &ExternalRunPaths,
    start: &PlatformCapture,
    status: std::process::ExitStatus,
    run_log_bytes: &[u8],
    started_at_utc: &str,
) -> Result<PathBuf, LocalPerfRunError> {
    let finished_at_utc = utc_now()?;
    let end = capture_platform(config);
    let (execution_end, end_capture_error) = match end {
        Ok(end) if end.hardware == start.hardware && end.request == start.request => {
            (Some(end.snapshot), None)
        }
        Ok(_) => (
            None,
            Some("end hardware or execution request drifted".to_owned()),
        ),
        Err(error) => (None, Some(bounded_diagnostic(&error))),
    };
    let post_identity = verify_external_paths(paths)
        .and_then(|()| verify_run_directories(config, directories))
        .and_then(|()| verify_prepared_build(build, producer, &paths.target))
        .and_then(|()| verify_environment_policy(&directories.run.handle, build));
    let (post_run_identity_verified, post_run_identity_error) = match post_identity {
        Ok(()) => (true, None),
        Err(error) => (false, Some(bounded_diagnostic(&error))),
    };
    let termination = status.code().map_or_else(
        || {
            status
                .signal()
                .map_or(RunnerAttemptTermination::Unknown, |signal| {
                    RunnerAttemptTermination::Signaled { signal }
                })
        },
        |code| RunnerAttemptTermination::Exited {
            code: i64::from(code),
        },
    );
    let receipt = RunnerAttemptReceipt {
        schema_version: ATTEMPT_RECEIPT_SCHEMA_VERSION.to_owned(),
        mode: "measurement".to_owned(),
        gate: config.gate.label().to_owned(),
        class_id: config.class_id.clone(),
        run_id: config.run_id.clone(),
        run_window: config.run_window.clone(),
        registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
        hardware: start.hardware.clone(),
        execution_request: start.request.clone(),
        execution_start: start.snapshot.clone(),
        execution_end,
        end_capture_error,
        build: build.receipt.clone(),
        post_run_identity_verified,
        post_run_identity_error,
        termination,
        run_log_sha256: sha256_hex(run_log_bytes),
        started_at_utc: started_at_utc.to_owned(),
        finished_at_utc,
        seal_sha256: String::new(),
    };
    let receipt_bytes = seal_attempt_receipt(receipt)?;
    verify_attempt_receipt(&receipt_bytes)?;
    let receipt_name = format!("{}.attempt.json", config.gate.label());
    write_new_sync_at(&directories.run.handle, &receipt_name, &receipt_bytes)?;
    directories.run.handle.sync_all()?;
    Ok(config.output_dir.join(receipt_name))
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

fn seal_attempt_receipt(mut receipt: RunnerAttemptReceipt) -> Result<Vec<u8>, LocalPerfRunError> {
    receipt.seal_sha256.clear();
    let preimage = serde_json::to_vec(&receipt)?;
    receipt.seal_sha256 = sha256_hex(&preimage);
    serde_json::to_vec(&receipt).map_err(LocalPerfRunError::from)
}

fn verify_attempt_receipt(bytes: &[u8]) -> Result<RunnerAttemptReceipt, LocalPerfRunError> {
    let receipt = serde_json::from_slice::<RunnerAttemptReceipt>(bytes)?;
    if receipt.schema_version != ATTEMPT_RECEIPT_SCHEMA_VERSION
        || receipt.mode != "measurement"
        || !is_sha256(&receipt.run_log_sha256)
        || !is_sha256(&receipt.seal_sha256)
    {
        return Err(LocalPerfRunError::Invalid(
            "diagnostic attempt receipt has an invalid schema, mode, or digest".to_owned(),
        ));
    }
    let mut preimage = receipt.clone();
    let expected = preimage.seal_sha256.clone();
    preimage.seal_sha256.clear();
    if sha256_hex(&serde_json::to_vec(&preimage)?) != expected
        || serde_json::to_vec(&receipt)? != bytes
    {
        return Err(LocalPerfRunError::Invalid(
            "diagnostic attempt receipt seal or canonical bytes do not verify".to_owned(),
        ));
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read};

    use super::*;

    fn policy_config(gate: PerfGate, mode: &str) -> LocalPerfRunConfig {
        LocalPerfRunConfig {
            gate,
            class_id: "m4-macos".to_owned(),
            run_id: "candidate-1".to_owned(),
            run_window: "window-1".to_owned(),
            thread_budget: 14,
            measurement_runs: MIN_MEASUREMENT_RUNS,
            apple_execution_mode: mode.to_owned(),
            output_dir: PathBuf::from("/tmp/frankensearch-perf-run"),
        }
    }

    #[test]
    fn m4_policy_fails_closed_for_every_gate_and_execution_mode() {
        for gate in PerfGate::ALL {
            for mode in ["p-only", "p-plus-e", "not-applicable"] {
                let error = validate_platform_gate_policy(&policy_config(gate, mode))
                    .expect_err("every current M4 promotion path must reject");
                if matches!(gate, PerfGate::Qg3 | PerfGate::Qg4 | PerfGate::Qg5) {
                    assert!(error.to_string().contains("any host"));
                } else {
                    assert!(error.to_string().contains("actual executing image"));
                    assert!(error.to_string().contains("diagnostic-only"));
                }
            }
        }
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
    fn full_gate_thread_budget_is_derived_from_the_frozen_matrix() {
        for (gate, expected) in [
            (PerfGate::Qg1, 128),
            (PerfGate::Qg2, 1),
            (PerfGate::Qg3, 1),
            (PerfGate::Qg4, 1),
            (PerfGate::Qg5, 1),
            (PerfGate::Qg6, 1),
            (PerfGate::Qg7, 8),
            (PerfGate::Qg8, 32),
            (PerfGate::Qg9, 1),
            (PerfGate::Qg10, 1),
        ] {
            assert_eq!(normative_thread_budget(gate).unwrap(), expected);
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
        let source = include_str!("local_perf_runner.rs");
        let preflight = source.find(".preflight(").expect("registry preflight");
        let log_creation = source
            .find("create_new_file_at(&run_directories.run.handle, \"run.log\")")
            .expect("run-log creation");
        let child_spawn = source.find("child.spawn()?").expect("measured child spawn");
        assert!(preflight < log_creation);
        assert!(log_creation < child_spawn);
    }

    #[test]
    fn pinned_directory_and_held_benchmark_reject_path_replacement() {
        let root = tempfile::tempdir().expect("pinned-root test directory");
        let output = root.path().join("output");
        fs::create_dir(&output).expect("create output root");
        let pinned_output = pin_directory(&output, true).expect("pin output root");
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
            stable_lease_id("trj-zen3-1c").unwrap(),
            stable_lease_id("trj-zen3-64c-smt2").unwrap()
        );
        assert_eq!(stable_lease_id("m4-macos").unwrap(), "m4-macos-exclusive");
        assert_eq!(
            stable_lease_path("trj-zen3-1c").unwrap(),
            PathBuf::from("/tmp/frankensearch-perf-host-global-exclusive.lock")
        );
        assert_eq!(
            stable_lease_path("trj-zen3-64c-smt2").unwrap(),
            PathBuf::from("/tmp/frankensearch-perf-host-global-exclusive.lock")
        );
        assert_eq!(
            stable_lease_path("m4-macos").unwrap(),
            PathBuf::from("/tmp/frankensearch-perf-host-global-exclusive.lock")
        );
        validate_canonical_lease_parent(
            &stable_lease_path("trj-zen3-1c").expect("canonical TRJ lease"),
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
    fn trj_class_parser_preserves_physical_width_and_smt_identity() {
        assert_eq!(parse_trj_class("trj-zen3-1c").unwrap(), (1, 1));
        assert_eq!(parse_trj_class("trj-zen3-64c").unwrap(), (64, 1));
        assert_eq!(parse_trj_class("trj-zen3-64c-smt2").unwrap(), (64, 2));
        for rejected in [
            "trj-zen-128c",
            "trj-zen3-0c",
            "trj-zen3-65c",
            "trj-zen3-16c-smt4",
            "x86-vps-ovh",
        ] {
            assert!(parse_trj_class(rejected).is_err(), "{rejected} admitted");
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
    fn diagnostic_attempt_receipt_is_sealed_and_never_registry_admissible() {
        let receipt = RunnerAttemptReceipt {
            schema_version: ATTEMPT_RECEIPT_SCHEMA_VERSION.to_owned(),
            mode: "measurement".to_owned(),
            gate: "QG-2".to_owned(),
            class_id: "trj-zen3-1c".to_owned(),
            run_id: "failed-1".to_owned(),
            run_window: "window-1".to_owned(),
            registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
            hardware: RunnerHardware {
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
                cpu_vendor: "AuthenticAMD".to_owned(),
                cpu_family: Some(25),
                cpu_model: Some(1),
                cpu_stepping: Some(1),
                cpu_model_name: "test".to_owned(),
                physical_cores: 64,
                logical_cpus: 128,
                numa_nodes: 1,
                memory_bytes: 1,
                page_size_bytes: 4096,
                performance_cores: None,
                efficiency_cores: None,
                runtime_detected_isa: vec!["scalar".to_owned()],
                topology_sha256: "a".repeat(64),
                fingerprint_sha256: "b".repeat(64),
            },
            execution_request: RunnerExecutionRequest {
                requested_logical_cpu_ids: vec![0],
                requested_physical_core_width: 1,
                thread_budget: 1,
                apple_execution_mode: "not-applicable".to_owned(),
            },
            execution_start: RunnerExecutionSnapshot {
                observed_logical_cpu_ids: vec![0],
                effective_physical_core_ids: vec!["0:0".to_owned()],
                cpu_assignment_observability: "affinity-enforced".to_owned(),
                effective_cpuset_sha256: "c".repeat(64),
                threads_per_core: 1,
                smt_state: "off".to_owned(),
                numa_node_ids: vec![0],
                numa_policy: "bind:0".to_owned(),
                governor: "performance".to_owned(),
                thermal_pressure: false,
                exclusive_lease: true,
                exclusive_lease_id: "trj-zen3-exclusive".to_owned(),
                local_execution: true,
                observed_hardware_fingerprint_sha256: "b".repeat(64),
                snapshot_sha256: "d".repeat(64),
            },
            execution_end: None,
            end_capture_error: Some("child failed before terminal probe".to_owned()),
            build: RunnerBuild {
                git_revision: "e".repeat(40),
                git_dirty: false,
                worktree_state_sha256: None,
                cargo_lock_sha256: "f".repeat(64),
                executable_sha256: "a".repeat(64),
                command_sha256: "b".repeat(64),
                environment_sha256: "c".repeat(64),
                producer: RunnerProducer {
                    contract_version: LOCAL_PERF_PRODUCER_CONTRACT_VERSION.to_owned(),
                    source_git_revision: "e".repeat(40),
                    source_git_dirty: false,
                    cargo_lock_sha256: "f".repeat(64),
                    executable_sha256: "c".repeat(64),
                },
            },
            post_run_identity_verified: true,
            post_run_identity_error: None,
            termination: RunnerAttemptTermination::Exited { code: 17 },
            run_log_sha256: "d".repeat(64),
            started_at_utc: "2026-07-29T00:00:00Z".to_owned(),
            finished_at_utc: "2026-07-29T00:00:01Z".to_owned(),
            seal_sha256: String::new(),
        };
        let bytes = seal_attempt_receipt(receipt).expect("seal attempt receipt");
        verify_attempt_receipt(&bytes).expect("verify sealed attempt receipt");

        let registry = MachineClassRegistry::frozen().expect("frozen registry");
        let context = MachineClassAdmissionContext {
            gate: "QG-2".to_owned(),
            destination_basename: "QG-2.trj-zen3-1c.latest.json".to_owned(),
        };
        let write_count = std::cell::Cell::new(0_u64);
        assert!(
            registry
                .admit_then(&bytes, &context, |_| write_count.set(write_count.get() + 1))
                .is_err()
        );
        assert_eq!(write_count.get(), 0);

        let mut tampered = bytes;
        let offset = tampered.len() / 2;
        tampered[offset] ^= 1;
        assert!(verify_attempt_receipt(&tampered).is_err());
    }

    #[test]
    fn bounded_diagnostic_preserves_utf8_at_multibyte_boundary() {
        let error = LocalPerfRunError::Invalid(format!("{}é-tail", "a".repeat(239)));
        let bounded = bounded_diagnostic(&error);
        assert!(bounded.len() <= 240);
        assert!(std::str::from_utf8(bounded.as_bytes()).is_ok());
    }
}
