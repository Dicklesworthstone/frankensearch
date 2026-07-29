//! Typed registered-host performance runner.
//!
//! This producer builds the exact benchmark from a clean source snapshot, then
//! owns the exclusive lease and start/end probes across the measured child. It
//! emits the required receipt last, only after the child exits successfully
//! and every exact artifact re-verifies.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rustix::fs::{FlockOperation, flock};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::machine_class_registry::{
    MACHINE_CLASS_REGISTRY_SHA256, MachineClassAdmissionContext, MachineClassError,
    MachineClassRegistry, RunnerArtifactManifest, RunnerBuild, RunnerCompletion, RunnerDurability,
    RunnerExecution, RunnerExecutionRequest, RunnerExecutionSnapshot, RunnerHardware,
    RunnerReceipt, seal_runner_receipt, sha256_hex,
};
use crate::{
    EvidenceArtifactError, PerfEvidenceArtifact, PerfGate, PerfGateArtifact,
    command_sha256_from_argv,
};

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
    /// Maximum engine thread budget admitted for this run.
    pub thread_budget: u64,
    /// `not-applicable` or the currently admissible `p-plus-e`.
    pub apple_execution_mode: String,
    /// Host-local persistent file used for nonblocking `flock`.
    pub lease_path: PathBuf,
    /// Unique output directory. Existing finalization files are never replaced.
    pub output_dir: PathBuf,
}

/// Files emitted after a successful self-verifying finalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalPerfRunOutput {
    /// Exact child log.
    pub run_log: PathBuf,
    /// Exact canonical compact artifact manifest.
    pub artifact_manifest: PathBuf,
    /// Exact strict completion receipt.
    pub runner_receipt: PathBuf,
    /// Pre-commit inventory written before the receipt commit boundary.
    pub precommit_inventory: PathBuf,
}

/// Typed local-run failure. No receipt is emitted unless all verification
/// succeeds. A failed final write may leave only non-admissible diagnostics,
/// the manifest, and the pre-commit inventory.
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedBuild {
    receipt: RunnerBuild,
    revision: String,
    command: Vec<OsString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CleanSourceSnapshot {
    revision: String,
    cargo_lock_sha256: String,
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
    bound_evidence_preview_sha256: String,
}

/// Execute and finalize one registered-host benchmark invocation.
///
/// The benchmark is built by this producer from the clean source snapshot
/// before the lease opens. The lease is then acquired before the start probes
/// and held until every output is synced. Child failure preserves `run.log`
/// and `exit-status` but emits no receipt.
///
/// # Errors
///
/// Returns a typed failure for an unavailable lease, unsupported platform,
/// dirty/offloaded source, probe drift, child failure, malformed artifact, or
/// any self-admission mismatch.
pub fn run_local_perf_command(
    config: &LocalPerfRunConfig,
) -> Result<LocalPerfRunOutput, LocalPerfRunError> {
    validate_config(config)?;
    require_local_execution()?;
    fs::create_dir_all(&config.output_dir)?;
    let artifact_dir = config.output_dir.join("artifacts");
    fs::create_dir_all(&artifact_dir)?;
    let captured_build = prepare_benchmark()?;

    let lease_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&config.lease_path)?;
    flock(&lease_file, FlockOperation::NonBlockingLockExclusive).map_err(std::io::Error::from)?;

    verify_prepared_build(&captured_build)?;
    let start = capture_platform(config)?;
    let started_at_utc = utc_now()?;

    let run_log_path = config.output_dir.join("run.log");
    let exit_status_path = config.output_dir.join("exit-status");
    let run_log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&run_log_path)?;
    let run_log_stderr = run_log.try_clone()?;
    let mut child = Command::new(&captured_build.command[0]);
    child
        .args(&captured_build.command[1..])
        .env("QUILL_PERF_GATE", config.gate.label())
        .env("QUILL_PERF_SCALE", "full")
        .env("QUILL_PERF_BUILD_PROFILE", "release-perf")
        .env("QUILL_PERF_GIT_REV", &captured_build.revision)
        .env("QUILL_PERF_RUN_ID", &config.run_id)
        .env("QUILL_PERF_RUN_WINDOW", &config.run_window)
        .env("QUILL_PERF_OUTPUT_DIR", &artifact_dir)
        .env("RCH_DISABLE", "1")
        .env("RCH_CARGO_WRAPPER_BYPASS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(run_log))
        .stderr(Stdio::from(run_log_stderr));
    let status = child.spawn()?.wait()?;
    sync_existing_file(&run_log_path)?;
    let exit_code = status.code().map_or(-1, i64::from);
    write_new_sync(&exit_status_path, format!("{exit_code}\n").as_bytes())?;
    if !status.success() {
        return Err(LocalPerfRunError::Invalid(format!(
            "benchmark child exited with status {exit_code}; diagnostic log preserved at {}",
            run_log_path.display()
        )));
    }

    let end = capture_platform(config)?;
    let finished_at_utc = utc_now()?;
    let end_build = verify_prepared_build(&captured_build)?;
    if start.hardware != end.hardware
        || start.request != end.request
        || start.snapshot != end.snapshot
        || captured_build != end_build
    {
        return Err(LocalPerfRunError::Invalid(
            "hardware, execution, or clean build identity drifted across the measured child"
                .to_owned(),
        ));
    }

    let threshold_path = artifact_dir.join(format!("{}.json", config.gate.label()));
    let evidence_path = artifact_dir.join(format!("{}.evidence.json", config.gate.label()));
    let run_log_bytes = fs::read(&run_log_path)?;
    let threshold_bytes = fs::read(&threshold_path)?;
    let evidence_bytes = fs::read(&evidence_path)?;
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
    let receipt = RunnerReceipt {
        requested_class_id: config.class_id.clone(),
        derived_class_id: config.class_id.clone(),
        registry_sha256: MACHINE_CLASS_REGISTRY_SHA256.to_owned(),
        hardware: start.hardware,
        execution: RunnerExecution {
            request: start.request,
            start: start.snapshot.clone(),
            end: end.snapshot,
            identity_sha256: String::new(),
        },
        build: captured_build.receipt,
        durability: durability_for_run(config)?,
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
    let context = MachineClassAdmissionContext {
        gate: config.gate.label().to_owned(),
        destination_basename: format!("{}.{}.latest.json", config.gate.label(), config.class_id),
    };
    let identity = MachineClassRegistry::frozen()?
        .admit(&receipt_bytes, &context)?
        .bind_artifact_manifest(
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
        schema_version: "frankensearch.perf-run-precommit.v1".to_owned(),
        gate: config.gate.label().to_owned(),
        class_id: config.class_id.clone(),
        run_id: config.run_id.clone(),
        run_window: config.run_window.clone(),
        run_log_sha256: sha256_hex(&run_log_bytes),
        threshold_artifact_sha256: sha256_hex(&threshold_bytes),
        evidence_artifact_sha256: sha256_hex(&evidence_bytes),
        artifact_manifest_sha256: sha256_hex(&artifact_manifest_bytes),
        runner_receipt_sha256: sha256_hex(&receipt_bytes),
        bound_evidence_preview_sha256: sha256_hex(&bound_evidence_bytes),
    };
    let inventory_bytes = serde_json::to_vec_pretty(&inventory)?;

    // The receipt is the required commit boundary consumed by the ratchet.
    // Everything it binds must be durable before it is created. A crash before
    // the final write can leave diagnostics or PRECOMMIT.json, but cannot leave
    // a promotable run.
    write_new_sync(&manifest_path, &artifact_manifest_bytes)?;
    write_new_sync(&inventory_path, &inventory_bytes)?;
    sync_directory(&config.output_dir)?;
    write_new_sync(&receipt_path, &receipt_bytes)?;
    sync_directory(&config.output_dir)?;

    drop(lease_file);
    Ok(LocalPerfRunOutput {
        run_log: run_log_path,
        artifact_manifest: manifest_path,
        runner_receipt: receipt_path,
        precommit_inventory: inventory_path,
    })
}

fn validate_config(config: &LocalPerfRunConfig) -> Result<(), LocalPerfRunError> {
    for (name, value) in [
        ("class ID", config.class_id.as_str()),
        ("run ID", config.run_id.as_str()),
        ("run window", config.run_window.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(LocalPerfRunError::Invalid(format!("{name} is empty")));
        }
    }
    if config.thread_budget == 0 {
        return Err(LocalPerfRunError::Invalid(
            "thread budget must be positive".to_owned(),
        ));
    }
    for (field, value) in [
        ("class ID", config.class_id.as_str()),
        ("run ID", config.run_id.as_str()),
        ("run window", config.run_window.as_str()),
    ] {
        validate_component(value, field)?;
    }
    validate_platform_gate_policy(config)?;
    validate_external_paths(config)?;
    Ok(())
}

fn validate_component(value: &str, field: &str) -> Result<(), LocalPerfRunError> {
    if value.is_empty()
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(LocalPerfRunError::Invalid(format!(
            "{field} {value:?} is not a safe identity component"
        )));
    }
    Ok(())
}

fn validate_platform_gate_policy(config: &LocalPerfRunConfig) -> Result<(), LocalPerfRunError> {
    if config.class_id == "m4-macos" {
        if config.apple_execution_mode != "p-plus-e" {
            return Err(LocalPerfRunError::Invalid(
                "m4-macos promotion evidence currently requires p-plus-e; p-only is non-admissible and any ad-hoc p-only measurement is diagnostic-only until a scheduler-assignment witness is implemented"
                    .to_owned(),
            ));
        }
        if matches!(config.gate, PerfGate::Qg1 | PerfGate::Qg8) {
            return Err(LocalPerfRunError::Invalid(format!(
                "{} cannot run promotion-grade on m4-macos until the normative matrix has class-specific 10P/14P+E endpoints",
                config.gate
            )));
        }
        if matches!(config.gate, PerfGate::Qg3 | PerfGate::Qg4 | PerfGate::Qg5) {
            return Err(LocalPerfRunError::Invalid(format!(
                "{} cannot run promotion-grade on macOS until both benchmark arms attest symmetric F_FULLFSYNC treatment",
                config.gate
            )));
        }
    } else if config.apple_execution_mode != "not-applicable" {
        return Err(LocalPerfRunError::Invalid(
            "non-M4 producer requires apple mode not-applicable".to_owned(),
        ));
    }
    Ok(())
}

fn validate_external_paths(config: &LocalPerfRunConfig) -> Result<(), LocalPerfRunError> {
    let repository = fs::canonicalize(command_output("git", &["rev-parse", "--show-toplevel"])?)?;
    let output = fs::canonicalize(&config.output_dir).map_err(|error| {
        LocalPerfRunError::Invalid(format!(
            "output directory {} must already exist and resolve cleanly: {error}",
            config.output_dir.display()
        ))
    })?;
    if !output.is_dir() || output.starts_with(&repository) {
        return Err(LocalPerfRunError::Invalid(format!(
            "output directory {} must be an existing directory outside the source repository",
            output.display()
        )));
    }
    validate_external_file_parent("lease path", &config.lease_path, &repository)?;

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
    let target_parent = target.parent().ok_or_else(|| {
        LocalPerfRunError::Invalid("CARGO_TARGET_DIR has no parent directory".to_owned())
    })?;
    let resolved_target_parent = fs::canonicalize(target_parent).map_err(|error| {
        LocalPerfRunError::Invalid(format!(
            "CARGO_TARGET_DIR parent {} must already exist and resolve cleanly: {error}",
            target_parent.display()
        ))
    })?;
    if resolved_target_parent.starts_with(&repository) {
        return Err(LocalPerfRunError::Invalid(
            "CARGO_TARGET_DIR must remain outside the source repository".to_owned(),
        ));
    }
    Ok(())
}

fn validate_external_file_parent(
    label: &str,
    path: &Path,
    repository: &Path,
) -> Result<(), LocalPerfRunError> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err(LocalPerfRunError::Invalid(format!(
            "{label} {} must be an absolute file path",
            path.display()
        )));
    }
    let parent = path.parent().ok_or_else(|| {
        LocalPerfRunError::Invalid(format!("{label} {} has no parent", path.display()))
    })?;
    let resolved_parent = fs::canonicalize(parent).map_err(|error| {
        LocalPerfRunError::Invalid(format!(
            "{label} parent {} must already exist and resolve cleanly: {error}",
            parent.display()
        ))
    })?;
    if resolved_parent.starts_with(repository) {
        return Err(LocalPerfRunError::Invalid(format!(
            "{label} must remain outside the source repository"
        )));
    }
    Ok(())
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

fn capture_clean_source() -> Result<CleanSourceSnapshot, LocalPerfRunError> {
    let revision = command_output("git", &["rev-parse", "HEAD"])?;
    let status = command_output("git", &["status", "--porcelain"])?;
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

fn prepare_benchmark() -> Result<CapturedBuild, LocalPerfRunError> {
    let source_before = capture_clean_source()?;
    let output = Command::new("cargo")
        .args([
            "build",
            "--locked",
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
        .env("RCH_DISABLE", "1")
        .env("RCH_CARGO_WRAPPER_BYPASS", "1")
        .output()?;
    if !output.status.success() {
        return Err(LocalPerfRunError::Invalid(format!(
            "typed benchmark build failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

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
        [path] => fs::canonicalize(path)?,
        paths => {
            return Err(LocalPerfRunError::Invalid(format!(
                "Cargo reported {} distinct perf_matrix benchmark executables, expected exactly one",
                paths.len()
            )));
        }
    };
    if !executable.is_file() {
        return Err(LocalPerfRunError::Invalid(format!(
            "Cargo-reported benchmark executable {} is not a file",
            executable.display()
        )));
    }
    let source_after = capture_clean_source()?;
    if source_before != source_after {
        return Err(LocalPerfRunError::Invalid(
            "clean source revision or Cargo.lock changed during the typed benchmark build"
                .to_owned(),
        ));
    }
    let command = vec![
        executable.clone().into_os_string(),
        OsString::from("--bench"),
        OsString::from("--noplot"),
    ];
    let executable_bytes = fs::read(&executable)?;
    let command_sha256 =
        command_sha256_from_argv(command.iter().map(|argument| argument.as_encoded_bytes()));
    Ok(CapturedBuild {
        receipt: RunnerBuild {
            git_revision: source_after.revision.clone(),
            git_dirty: false,
            worktree_state_sha256: None,
            cargo_lock_sha256: source_after.cargo_lock_sha256,
            executable_sha256: sha256_hex(&executable_bytes),
            command_sha256,
        },
        revision: source_after.revision,
        command,
    })
}

fn verify_prepared_build(expected: &CapturedBuild) -> Result<CapturedBuild, LocalPerfRunError> {
    let source = capture_clean_source()?;
    let executable = fs::read(PathBuf::from(&expected.command[0]))?;
    let command_sha256 = command_sha256_from_argv(
        expected
            .command
            .iter()
            .map(|argument| argument.as_encoded_bytes()),
    );
    let observed = CapturedBuild {
        receipt: RunnerBuild {
            git_revision: source.revision.clone(),
            git_dirty: false,
            worktree_state_sha256: None,
            cargo_lock_sha256: source.cargo_lock_sha256,
            executable_sha256: sha256_hex(&executable),
            command_sha256,
        },
        revision: source.revision,
        command: expected.command.clone(),
    };
    if &observed != expected {
        return Err(LocalPerfRunError::Invalid(
            "source, executable, or fixed benchmark argv drifted after the typed Cargo build"
                .to_owned(),
        ));
    }
    Ok(observed)
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
    if std::env::consts::OS == "macos"
        && matches!(config.gate, PerfGate::Qg3 | PerfGate::Qg4 | PerfGate::Qg5)
    {
        return Err(LocalPerfRunError::Invalid(format!(
            "{} is durability-adjacent on macOS and requires symmetric F_FULLFSYNC attestation",
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

fn capture_macos(config: &LocalPerfRunConfig) -> Result<PlatformCapture, LocalPerfRunError> {
    if config.class_id != "m4-macos" {
        return Err(LocalPerfRunError::Invalid(
            "macOS producer currently admits only m4-macos".to_owned(),
        ));
    }
    if config.apple_execution_mode != "p-plus-e" {
        return Err(LocalPerfRunError::Invalid(
            "m4-macos currently admits only p-plus-e execution".to_owned(),
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
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(LocalPerfRunError::Invalid(format!(
            "{program} {:?} failed: {}",
            arguments,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| {
            LocalPerfRunError::Invalid(format!("{program} output is not UTF-8: {error}"))
        })
}

fn utc_now() -> Result<String, LocalPerfRunError> {
    command_output("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
}

fn write_new_sync(path: &Path, bytes: &[u8]) -> Result<(), LocalPerfRunError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_existing_file(path: &Path) -> Result<(), LocalPerfRunError> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), LocalPerfRunError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy_config(gate: PerfGate, mode: &str) -> LocalPerfRunConfig {
        LocalPerfRunConfig {
            gate,
            class_id: "m4-macos".to_owned(),
            run_id: "candidate-1".to_owned(),
            run_window: "window-1".to_owned(),
            thread_budget: 14,
            apple_execution_mode: mode.to_owned(),
            lease_path: PathBuf::from("/tmp/frankensearch-perf.lock"),
            output_dir: PathBuf::from("/tmp/frankensearch-perf-run"),
        }
    }

    #[test]
    fn m4_policy_fails_closed_for_unverified_modes_and_unavailable_gates() {
        assert!(validate_platform_gate_policy(&policy_config(PerfGate::Qg6, "p-only")).is_err());
        for gate in [PerfGate::Qg1, PerfGate::Qg8] {
            let error = validate_platform_gate_policy(&policy_config(gate, "p-plus-e"))
                .expect_err("M4 matrix-incompatible gate must reject");
            assert!(error.to_string().contains("class-specific 10P/14P+E"));
        }
        for gate in [PerfGate::Qg3, PerfGate::Qg4, PerfGate::Qg5] {
            let error = validate_platform_gate_policy(&policy_config(gate, "p-plus-e"))
                .expect_err("M4 durability-adjacent gate must reject");
            assert!(error.to_string().contains("F_FULLFSYNC"));
        }
    }

    #[test]
    fn m4_policy_admits_only_current_full_pool_nonadjacent_envelope() {
        for gate in [
            PerfGate::Qg2,
            PerfGate::Qg6,
            PerfGate::Qg7,
            PerfGate::Qg9,
            PerfGate::Qg10,
        ] {
            validate_platform_gate_policy(&policy_config(gate, "p-plus-e"))
                .expect("current M4 P+E nonadjacent gate");
        }
    }

    #[test]
    fn lease_identity_is_stable_across_classes_in_one_host_family() {
        assert_eq!(
            stable_lease_id("trj-zen3-1c").unwrap(),
            stable_lease_id("trj-zen3-64c-smt2").unwrap()
        );
        assert_eq!(stable_lease_id("m4-macos").unwrap(), "m4-macos-exclusive");
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
}
