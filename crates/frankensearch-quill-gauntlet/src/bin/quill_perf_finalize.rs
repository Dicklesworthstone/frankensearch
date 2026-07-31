#![forbid(unsafe_code)]
//! Hold the registered-host lease across a benchmark child and finalize its
//! exact artifacts.

use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
#[cfg(test)]
use std::fs::File;
#[cfg(test)]
use std::io::{Read, Seek};
use std::path::PathBuf;
use std::process::ExitCode;

use frankensearch_quill_gauntlet::{
    ExecutionProfileId, HardwareClassId, LocalPerfRunConfig, LocalPerfRunError,
    LocalPerfRunSelection, MACHINE_CLASS_REGISTRY_SCHEMA_VERSION, MACHINE_CLASS_REGISTRY_SHA256,
    MachineProfileKey, PerfEvidenceAssemblyArtifact, PerfEvidenceAssemblyReadiness, PerfGate,
    VerifiedLocalPerfAttemptBundle, run_local_perf_command, run_selected_local_perf_command,
};
#[cfg(test)]
use frankensearch_quill_gauntlet::{
    LOCAL_PERF_PRODUCER_CONTRACT_VERSION, local_perf_producer_contract_json,
};
#[cfg(test)]
use sha2::{Digest, Sha256};

const USAGE: &str = "\
Usage:
  quill-perf-finalize \\
    --gate <QG-N> \\
    --hardware-class <registered-hardware-class> \\
    --execution-profile <registered-execution-profile> \\
    --run-id <unique-pass-id> \\
    --run-window <candidate-rerun-window> \\
    --runs <integer-10-through-100> \\
    [--fixture <exact-canonical-fixture>] \\
    --output-dir <unique-run-directory>

  quill-perf-finalize assemble \\
    --attempt-dir <exact-H2-attempt-directory> \\
    [--attempt-dir <exact-H2-attempt-directory>]... \\
    --output-dir <content-addressed-assembly-directory>

The typed producer builds and resolves perf_matrix itself from a clean source
snapshot. It requires RCH_DISABLE=1, RCH_CARGO_WRAPPER_BYPASS=1, and an
absolute CARGO_TARGET_DIR outside the source repository.

Assembly strictly reloads every sealed input and writes one descriptor-verified,
content-addressed receipt without creating or advancing a latest alias.
Exit status: 0=complete/adjudicable, 2=durable NoClaim, 64=invalid invocation.";

const MAX_LOG_VALUE_BYTES: usize = 240;

#[derive(Debug)]
struct Args {
    gate: PerfGate,
    profile: MachineProfileKey,
    run_id: String,
    run_window: String,
    measurement_runs: usize,
    fixture: Option<String>,
    output_dir: PathBuf,
}

#[derive(Debug)]
struct AssemblyArgs {
    attempt_dirs: Vec<PathBuf>,
    output_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalizeOutcome {
    Success,
    DurableNoClaim,
}

fn main() -> ExitCode {
    match run() {
        Ok(FinalizeOutcome::Success) => ExitCode::SUCCESS,
        Ok(FinalizeOutcome::DurableNoClaim) => ExitCode::from(2),
        Err(error) => {
            eprintln!("quill-perf-finalize: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(64)
        }
    }
}

fn run() -> Result<FinalizeOutcome, Box<dyn Error>> {
    let mut values = env::args_os().skip(1);
    let first = values.next();
    if first.as_deref() == Some(OsStr::new("assemble")) {
        return run_assembly(parse_assembly_args(values)?);
    }
    let args = parse_args(first.into_iter().chain(values))?;
    let config = LocalPerfRunConfig {
        gate: args.gate,
        profile: args.profile,
        run_id: args.run_id,
        run_window: args.run_window,
        measurement_runs: args.measurement_runs,
        output_dir: args.output_dir,
    };
    let result = if let Some(fixture) = args.fixture {
        let selection = LocalPerfRunSelection::for_fixture(fixture)?;
        run_selected_local_perf_command(&config, &selection)
    } else {
        run_local_perf_command(&config)
    };
    let output = match result {
        Ok(output) => output,
        Err(LocalPerfRunError::AttemptFailed {
            receipt_path,
            outcome,
        }) => {
            println!("attempt_receipt={}", receipt_path.display());
            println!("attempt_outcome={outcome:?}");
            return Ok(FinalizeOutcome::DurableNoClaim);
        }
        Err(error) => return Err(error.into()),
    };
    println!("run_log={}", output.run_log.display());
    println!("artifact_manifest={}", output.artifact_manifest.display());
    println!("environment_policy={}", output.environment_policy.display());
    println!("runner_receipt={}", output.runner_receipt.display());
    println!("attempt_receipt={}", output.attempt_receipt.display());
    println!("threshold_artifact={}", output.threshold_artifact.display());
    println!(
        "prebinding_evidence={}",
        output.prebinding_evidence.display()
    );
    println!("bound_evidence={}", output.bound_evidence.display());
    println!(
        "precommit_inventory={}",
        output.precommit_inventory.display()
    );
    Ok(FinalizeOutcome::Success)
}

fn parse_assembly_args<I>(mut values: I) -> Result<AssemblyArgs, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let mut attempt_dirs = Vec::new();
    let mut output_dir = None;
    while let Some(flag) = values.next() {
        match flag.to_string_lossy().as_ref() {
            "-h" | "--help" => return Err(USAGE.into()),
            "--attempt-dir" => {
                attempt_dirs.push(PathBuf::from(next_value(&mut values, "--attempt-dir")?));
            }
            "--output-dir" => {
                if output_dir.is_some() {
                    return Err("assemble repeats singleton --output-dir".into());
                }
                output_dir = Some(PathBuf::from(next_value(&mut values, "--output-dir")?));
            }
            other => return Err(format!("unknown assemble argument {other:?}").into()),
        }
    }
    if attempt_dirs.is_empty() {
        return Err("assemble requires at least one --attempt-dir".into());
    }
    Ok(AssemblyArgs {
        attempt_dirs,
        output_dir: output_dir.ok_or("assemble is missing --output-dir")?,
    })
}

fn run_assembly(args: AssemblyArgs) -> Result<FinalizeOutcome, Box<dyn Error>> {
    let attempts = args
        .attempt_dirs
        .iter()
        .map(|path| VerifiedLocalPerfAttemptBundle::load_verified(path))
        .collect::<Result<Vec<_>, _>>()?;
    let assembly = PerfEvidenceAssemblyArtifact::assemble(attempts)?;
    let output_path = assembly.write_atomic(&args.output_dir)?;
    // write_atomic pins the output directory and independently reopens the
    // owned final inode through that held directory descriptor before it
    // returns. A second pathname-based reload here would discard that proof
    // and reintroduce a symlink/substitution race.
    emit_assembly_logs(&assembly, &output_path)?;
    Ok(
        if assembly.readiness() == PerfEvidenceAssemblyReadiness::ReadyForAdjudication {
            FinalizeOutcome::Success
        } else {
            FinalizeOutcome::DurableNoClaim
        },
    )
}

fn emit_assembly_logs(
    assembly: &PerfEvidenceAssemblyArtifact,
    output_path: &std::path::Path,
) -> Result<(), Box<dyn Error>> {
    let plan = assembly.applicability_plan();
    let profile = plan.profile;
    let counts = assembly.counts();
    let compatibility = assembly.compatibility();
    emit_json_log(serde_json::json!({
        "event": "qg1_assembly_summary",
        "gate": "QG-1",
        "hardware_class": profile.hardware_class_id().as_str(),
        "execution_profile": profile.execution_profile_id().as_str(),
        "run_window": bounded_log_value(assembly.run_window()),
        "required_complete": assembly.is_complete(),
        "full_plan_coverage": assembly.has_full_plan_coverage(),
        "readiness": assembly.readiness(),
        "canonical_cells": counts.canonical_cells(),
        "required_cells": counts.required_cells(),
        "diagnostic_cells": counts.diagnostic_cells(),
        "not_applicable_cells": counts.not_applicable_cells(),
        "measured_cells": counts.measured_cells(),
        "missing_required_cells": assembly.missing_required_cell_ids().len(),
        "missing_diagnostic_cells": assembly.missing_diagnostic_cell_ids().len(),
        "completed_shards": counts.completed_shards(),
        "failed_shards": counts.failed_shards(),
        "applicability_plan_schema_version": plan.schema_version.as_str(),
        "applicability_plan_sha256": plan.applicability_plan_sha256.as_str(),
        "normalized_perf_manifest_sha256": plan.normalized_perf_manifest_sha256.as_str(),
        "matrix_contract_schema_version": plan.matrix_contract_schema_version.as_str(),
        "gate_matrix_contract_sha256": plan.gate_matrix_contract_sha256.as_str(),
        "machine_class_registry_schema_version": MACHINE_CLASS_REGISTRY_SCHEMA_VERSION,
        "machine_class_registry_sha256": MACHINE_CLASS_REGISTRY_SHA256,
        "capacity_semantics": compatibility.map(|value| value.capacity_semantics()),
        "execution_capacity": compatibility.map(|value| value.execution_capacity()),
        "max_exercised_cell_width": compatibility.map(|value| value.max_exercised_cell_width()),
        "matrix_manifest_sha256": assembly.matrix_manifest().matrix_manifest_sha256(),
        "semantic_cell_set_sha256": assembly.semantic_cell_set().semantic_cell_set_sha256(),
        "assembly_sha256": assembly.assembly_sha256(),
        "output": bounded_log_value(&output_path.display().to_string()),
    }))?;
    for (index, source) in assembly.source_shards().iter().enumerate() {
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_shard",
            "shard_index": index,
            "terminal": "completed",
            "run_id": bounded_log_value(source.run_id()),
            "cell_count": source.cell_ids().len(),
            "process_receipt_sha256": source.process().process_receipt_sha256(),
            "bound_evidence_file_sha256": source.bound_evidence_file_sha256(),
            "evidence_artifact_sha256": source.evidence_artifact_sha256(),
            "runner_receipt_sha256": source.runner_receipt_sha256(),
            "runner_artifact_manifest_sha256": source.runner_artifact_manifest_sha256(),
        }))?;
    }
    for cell in assembly.cell_sources() {
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_cell",
            "ordinal": cell.ordinal(),
            "cell_id": cell.cell_id(),
            "role": cell.role(),
            "terminal": cell.terminal_status(),
            "run_id": bounded_log_value(cell.run_id()),
            "evidence_artifact_sha256": cell.evidence_artifact_sha256(),
            "runner_receipt_sha256": cell.runner_receipt_sha256(),
            "runner_artifact_manifest_sha256": cell.runner_artifact_manifest_sha256(),
        }))?;
    }
    for (index, attempt) in assembly.failed_shards().iter().enumerate() {
        let process = attempt.process();
        let receipt = process.receipt();
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_shard",
            "shard_index": assembly.source_shards().len() + index,
            "terminal": receipt.outcome(),
            "run_id": bounded_log_value(receipt.run_id()),
            "run_window": bounded_log_value(receipt.run_window()),
            "selected_cell_count": receipt.selected_cell_ids().len(),
            "process_receipt_sha256": process.process_receipt_sha256(),
            "retry": receipt.retry(),
        }))?;
    }
    for cell_id in assembly.missing_required_cell_ids() {
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_set_difference",
            "cell_id": cell_id,
            "role": "required",
            "terminal": "missing",
        }))?;
    }
    for cell_id in assembly.missing_diagnostic_cell_ids() {
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_set_difference",
            "cell_id": cell_id,
            "role": "diagnostic",
            "terminal": "missing",
        }))?;
    }
    for diagnostic in assembly.non_adjudicable_cells() {
        let reasons = diagnostic
            .reasons()
            .iter()
            .map(|reason| {
                serde_json::json!({
                    "code": reason.code,
                    "severity": reason.severity,
                    "message": bounded_log_value(&reason.message),
                })
            })
            .collect::<Vec<_>>();
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_non_adjudicable_cell",
            "cell_id": diagnostic.cell_id(),
            "ordinal": diagnostic.ordinal(),
            "role": diagnostic.role(),
            "terminal": diagnostic.terminal_status(),
            "reasons": reasons,
        }))?;
    }
    for diagnostic in assembly.non_adjudicable_sources() {
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_non_adjudicable_source",
            "run_id": bounded_log_value(diagnostic.run_id()),
            "evidence_artifact_sha256": diagnostic.evidence_artifact_sha256(),
            "cell_ids": diagnostic.cell_ids(),
            "reason": {
                "code": diagnostic.reason().code.as_str(),
                "severity": diagnostic.reason().severity,
                "message": bounded_log_value(&diagnostic.reason().message),
            },
        }))?;
    }
    for (scope, retry) in [
        ("required_coverage", assembly.retry_predicate()),
        ("diagnostic_coverage", assembly.diagnostic_retry_predicate()),
        ("adjudication", assembly.adjudication_retry_predicate()),
    ] {
        let Some(retry) = retry else {
            continue;
        };
        emit_json_log(serde_json::json!({
            "event": "qg1_assembly_retry",
            "scope": scope,
            "terminal": assembly.readiness(),
            "retry": retry,
        }))?;
    }
    Ok(())
}

fn emit_json_log(value: serde_json::Value) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string(&value)?);
    Ok(())
}

fn bounded_log_value(value: &str) -> &str {
    if value.len() <= MAX_LOG_VALUE_BYTES {
        return value;
    }
    let mut end = MAX_LOG_VALUE_BYTES;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

#[cfg(test)]
fn open_executing_image() -> Result<File, Box<dyn Error>> {
    let path = match env::consts::OS {
        "linux" => PathBuf::from("/proc/self/exe"),
        "macos" => env::current_exe()?,
        other => return Err(format!("unsupported executable capture OS {other:?}").into()),
    };
    File::open(path).map_err(Into::into)
}

#[cfg(test)]
fn sha256_open_file(file: &File) -> Result<String, Box<dyn Error>> {
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

fn parse_args<I>(mut values: I) -> Result<Args, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let mut gate = None;
    let mut hardware_class_id = None;
    let mut execution_profile_id = None;
    let mut run_id = None;
    let mut run_window = None;
    let mut measurement_runs = None;
    let mut fixture = None;
    let mut output_dir = None;

    while let Some(flag) = values.next() {
        let flag_text = flag.to_string_lossy();
        match flag_text.as_ref() {
            "-h" | "--help" => return Err(USAGE.into()),
            "--gate" => {
                let value = next_value(&mut values, "--gate")?;
                set_singleton(
                    &mut gate,
                    "--gate",
                    value.to_string_lossy().parse::<PerfGate>()?,
                )?;
            }
            "--hardware-class" => {
                let value = next_value(&mut values, "--hardware-class")?;
                set_singleton(
                    &mut hardware_class_id,
                    "--hardware-class",
                    parse_hardware_class_id(&value)?,
                )?;
            }
            "--execution-profile" => {
                let value = next_value(&mut values, "--execution-profile")?;
                set_singleton(
                    &mut execution_profile_id,
                    "--execution-profile",
                    parse_execution_profile_id(&value)?,
                )?;
            }
            "--run-id" => {
                set_singleton(
                    &mut run_id,
                    "--run-id",
                    next_value(&mut values, "--run-id")?
                        .to_string_lossy()
                        .into_owned(),
                )?;
            }
            "--run-window" => {
                set_singleton(
                    &mut run_window,
                    "--run-window",
                    next_value(&mut values, "--run-window")?
                        .to_string_lossy()
                        .into_owned(),
                )?;
            }
            "--runs" => {
                let value = next_value(&mut values, "--runs")?;
                set_singleton(
                    &mut measurement_runs,
                    "--runs",
                    value.to_string_lossy().parse::<usize>()?,
                )?;
            }
            "--fixture" => {
                set_singleton(
                    &mut fixture,
                    "--fixture",
                    next_value(&mut values, "--fixture")?
                        .to_string_lossy()
                        .into_owned(),
                )?;
            }
            "--output-dir" => {
                let value = PathBuf::from(next_value(&mut values, "--output-dir")?);
                set_singleton(&mut output_dir, "--output-dir", value)?;
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let profile = MachineProfileKey::new(
        hardware_class_id.ok_or("missing --hardware-class")?,
        execution_profile_id.ok_or("missing --execution-profile")?,
    )?;
    Ok(Args {
        gate: gate.ok_or("missing --gate")?,
        profile,
        run_id: run_id.ok_or("missing --run-id")?,
        run_window: run_window.ok_or("missing --run-window")?,
        measurement_runs: measurement_runs.ok_or("missing --runs")?,
        fixture,
        output_dir: output_dir.ok_or("missing --output-dir")?,
    })
}

fn set_singleton<T>(slot: &mut Option<T>, flag: &str, value: T) -> Result<(), Box<dyn Error>> {
    if slot.is_some() {
        return Err(format!("repeated singleton argument {flag}").into());
    }
    *slot = Some(value);
    Ok(())
}

fn parse_hardware_class_id(value: &OsStr) -> Result<HardwareClassId, Box<dyn Error>> {
    match value.to_str() {
        Some("x86-vps-ovh") => Ok(HardwareClassId::X86VpsOvh),
        Some("trj-zen3-5995wx") => Ok(HardwareClassId::TrjZen35995wx),
        Some("m4-macos") => Ok(HardwareClassId::M4Macos),
        Some("m5-macos") => Ok(HardwareClassId::M5Macos),
        Some(other) => Err(format!("unknown --hardware-class {other:?}").into()),
        None => Err("--hardware-class must be valid UTF-8".into()),
    }
}

fn parse_execution_profile_id(value: &OsStr) -> Result<ExecutionProfileId, Box<dyn Error>> {
    match value.to_str() {
        Some("x86-diagnostic") => Ok(ExecutionProfileId::X86Diagnostic),
        Some("physical-64") => Ok(ExecutionProfileId::Physical64),
        Some("smt2-128") => Ok(ExecutionProfileId::Smt2_128),
        Some("scheduler-10") => Ok(ExecutionProfileId::Scheduler10),
        Some("scheduler-14") => Ok(ExecutionProfileId::Scheduler14),
        Some(other) => Err(format!("unknown --execution-profile {other:?}").into()),
        None => Err("--execution-profile must be valid UTF-8".into()),
    }
}

fn next_value<I>(values: &mut I, flag: &str) -> Result<OsString, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn production_source() -> &'static str {
        const TEST_MODULE_BOUNDARY: &str = "#[cfg(test)]\nmod tests {";

        let source = include_str!("quill_perf_finalize.rs");
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

    #[test]
    fn assembly_parser_accepts_repeated_typed_inputs() {
        let args = parse_assembly_args(
            [
                "--attempt-dir",
                "/tmp/qg1-attempt-a",
                "--attempt-dir",
                "/tmp/qg1-attempt-b",
                "--attempt-dir",
                "/tmp/qg1-attempt-c",
                "--output-dir",
                "/tmp/qg1-assembly",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .expect("complete assembly invocation");
        assert_eq!(args.attempt_dirs.len(), 3);
        assert_eq!(args.output_dir, PathBuf::from("/tmp/qg1-assembly"));
    }

    #[test]
    fn assembly_parser_rejects_empty_or_ambiguous_invocations() {
        assert!(
            parse_assembly_args(
                ["--output-dir", "/tmp/qg1-assembly"]
                    .into_iter()
                    .map(OsString::from)
            )
            .is_err()
        );
        assert!(
            parse_assembly_args(
                [
                    "--attempt-dir",
                    "/tmp/qg1-attempt",
                    "--output-dir",
                    "/tmp/one",
                    "--output-dir",
                    "/tmp/two",
                ]
                .into_iter()
                .map(OsString::from)
            )
            .is_err()
        );
        assert!(
            parse_assembly_args(
                ["--attempt-dir", "/tmp/qg1-attempt"]
                    .into_iter()
                    .map(OsString::from)
            )
            .is_err()
        );
        for legacy_flag in ["--completed-shard", "--failed-shard", "--candidate"] {
            assert!(
                parse_assembly_args(
                    [legacy_flag, "/tmp/legacy.json"]
                        .into_iter()
                        .map(OsString::from)
                )
                .is_err(),
                "legacy assembly flag {legacy_flag} was accepted"
            );
        }
    }

    #[test]
    fn structured_log_values_are_utf8_safe_and_bounded() {
        let value = format!("{}{}", "x".repeat(MAX_LOG_VALUE_BYTES - 1), "éé");
        let bounded = bounded_log_value(&value);
        assert!(bounded.len() <= MAX_LOG_VALUE_BYTES);
        assert!(bounded.is_char_boundary(bounded.len()));
    }

    #[test]
    fn parser_accepts_only_complete_typed_producer_identity() {
        let args = parse_args(
            [
                "--gate",
                "QG-2",
                "--hardware-class",
                "trj-zen3-5995wx",
                "--execution-profile",
                "physical-64",
                "--run-id",
                "candidate-1",
                "--run-window",
                "window-1",
                "--runs",
                "10",
                "--fixture",
                "bulk/small/1/on",
                "--output-dir",
                "/tmp/quill-perf/candidate",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .expect("complete producer invocation");
        assert_eq!(args.gate, PerfGate::Qg2);
        assert_eq!(
            args.profile.hardware_class_id(),
            HardwareClassId::TrjZen35995wx
        );
        assert_eq!(
            args.profile.execution_profile_id(),
            ExecutionProfileId::Physical64
        );
        assert_eq!(args.measurement_runs, 10);
        assert_eq!(args.fixture.as_deref(), Some("bulk/small/1/on"));
    }

    #[test]
    fn parser_rejects_arbitrary_child_commands() {
        let result = parse_args(
            [
                "--gate",
                "QG-2",
                "--hardware-class",
                "trj-zen3-5995wx",
                "--execution-profile",
                "physical-64",
                "--run-id",
                "candidate-1",
                "--run-window",
                "window-1",
                "--output-dir",
                "/tmp/quill-perf/candidate",
                "--",
                "/tmp/stale-perf_matrix",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err());
    }

    #[test]
    fn producer_parser_rejects_repeated_singleton_arguments() {
        let result = parse_args(
            ["--gate", "QG-2", "--gate", "QG-1"]
                .into_iter()
                .map(OsString::from),
        );
        assert!(result.is_err());
    }

    #[test]
    fn parser_rejects_legacy_aliases_and_cross_hardware_profiles() {
        for legacy_flag in ["--class", "--thread-budget", "--apple-mode"] {
            let result = parse_args(
                [legacy_flag, "legacy-value"]
                    .into_iter()
                    .map(OsString::from),
            );
            assert!(result.is_err(), "legacy flag {legacy_flag} was accepted");
        }

        let result = parse_args(
            [
                "--gate",
                "QG-2",
                "--hardware-class",
                "m4-macos",
                "--execution-profile",
                "physical-64",
                "--run-id",
                "candidate-1",
                "--run-window",
                "window-1",
                "--runs",
                "10",
                "--output-dir",
                "/tmp/quill-perf/candidate",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err(), "cross-hardware profile was accepted");
    }

    #[test]
    fn producer_contract_binds_source_lock_registry_and_executing_elf() {
        assert_eq!(MACHINE_CLASS_REGISTRY_SHA256.len(), 64);
        let executing_image = open_executing_image().expect("open current test executable");
        let executing_sha256 =
            sha256_open_file(&executing_image).expect("hash current test executable");
        let contract =
            local_perf_producer_contract_json(&executing_sha256).expect("producer contract JSON");
        let value = serde_json::from_str::<serde_json::Value>(&contract).expect("contract value");
        assert_eq!(
            value["schema_version"],
            "frankensearch.quill-local-perf-producer-contract.v1"
        );
        assert_eq!(
            value["machine_class_registry_sha256"],
            MACHINE_CLASS_REGISTRY_SHA256
        );
        assert_eq!(
            value["producer"]["contract_version"],
            LOCAL_PERF_PRODUCER_CONTRACT_VERSION
        );
        assert!(value["producer"]["source_git_revision"].is_string());
        assert!(value["producer"]["source_git_dirty"].is_boolean());
        assert_eq!(
            value["producer"]["cargo_lock_sha256"]
                .as_str()
                .map(str::len),
            Some(64)
        );
        assert_eq!(value["producer"]["executable_sha256"], executing_sha256);
    }

    #[test]
    fn measurement_mode_enters_the_typed_runner_without_prelock_hashing() {
        let source = production_source();
        let run_body = source
            .split("fn run() -> Result<FinalizeOutcome, Box<dyn Error>>")
            .nth(1)
            .and_then(|suffix| suffix.split("fn parse_assembly_args").next())
            .expect("production run body");
        let config = unique_marker_offset(run_body, "let config = LocalPerfRunConfig");
        let selected = unique_marker_offset(
            run_body,
            "run_selected_local_perf_command(&config, &selection)",
        );
        let full = unique_marker_offset(run_body, "run_local_perf_command(&config)");
        assert!(selected > config);
        assert!(full > config);
        assert!(!run_body.contains("sha256"));
        assert!(!run_body.contains("open_executing_image"));
    }

    #[test]
    fn assembly_uses_the_descriptor_verified_publication_without_path_reload() {
        let source = production_source();
        let assembly_body = source
            .split("fn run_assembly(")
            .nth(1)
            .and_then(|suffix| suffix.split("fn emit_assembly_logs(").next())
            .expect("production assembly body");
        let attempt_loader = unique_marker_offset(
            assembly_body,
            "VerifiedLocalPerfAttemptBundle::load_verified(path)",
        );
        let assembly = unique_marker_offset(
            assembly_body,
            "PerfEvidenceAssemblyArtifact::assemble(attempts)",
        );
        let publication =
            unique_marker_offset(assembly_body, "assembly.write_atomic(&args.output_dir)");
        assert!(attempt_loader < assembly);
        assert!(assembly < publication);
        assert!(
            !assembly_body[publication..].contains("load_verified"),
            "the descriptor-verified publication boundary must not be followed by a pathname reload"
        );
    }

    #[test]
    fn launcher_executes_one_held_producer_without_prelock_source_or_hash_work() {
        let launcher = include_str!("../../../../scripts/perf-runner.sh");
        assert!(
            !launcher.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with("cargo build") || line.starts_with("cargo run")
            }),
            "measurement launcher must never compile"
        );
        let held_executable = launcher
            .find("exec 9<\"$FINALIZER_ELF\"")
            .expect("held finalizer descriptor");
        let producer_launch = launcher
            .find("\"$HELD_FINALIZER_ELF\"")
            .expect("held finalizer launch");
        assert!(held_executable <= producer_launch);
        for forbidden in [
            "--print-producer-contract",
            "sha256_file",
            "jq ",
            "status --porcelain",
            "mkdir \"$RUN_DIR\"",
            "launcher.log",
            "producer.pid",
        ] {
            assert!(
                !launcher.contains(forbidden),
                "launcher performs prelock or prevalidation work: {forbidden}"
            );
        }
        assert!(launcher.contains("QUILL_PERF_HELD_PRODUCER_FD=9"));
        assert!(launcher.contains("[ ! -e \"$RUN_DIR\" ]"));
        assert!(launcher.contains("--hardware-class \"$HARDWARE_CLASS\""));
        assert!(launcher.contains("--execution-profile \"$EXECUTION_PROFILE\""));
        for legacy_flag in ["--class", "--thread-budget", "--apple-mode"] {
            assert!(
                !launcher.contains(legacy_flag),
                "launcher still accepts legacy producer flag {legacy_flag}"
            );
        }
    }
}
