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
    ExecutionProfileId, HardwareClassId, LocalPerfRunConfig, MachineProfileKey, PerfGate,
    run_local_perf_command,
};
#[cfg(test)]
use frankensearch_quill_gauntlet::{
    LOCAL_PERF_PRODUCER_CONTRACT_VERSION, MACHINE_CLASS_REGISTRY_SHA256,
    local_perf_producer_contract_json,
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
    --output-dir <unique-run-directory>

The typed producer builds and resolves perf_matrix itself from a clean source
snapshot. It requires RCH_DISABLE=1, RCH_CARGO_WRAPPER_BYPASS=1, and an
absolute CARGO_TARGET_DIR outside the source repository.";

#[derive(Debug)]
struct Args {
    gate: PerfGate,
    profile: MachineProfileKey,
    run_id: String,
    run_window: String,
    measurement_runs: usize,
    output_dir: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("quill-perf-finalize: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(64)
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let output = run_local_perf_command(&LocalPerfRunConfig {
        gate: args.gate,
        profile: args.profile,
        run_id: args.run_id,
        run_window: args.run_window,
        measurement_runs: args.measurement_runs,
        output_dir: args.output_dir,
    })?;
    println!("run_log={}", output.run_log.display());
    println!("artifact_manifest={}", output.artifact_manifest.display());
    println!("environment_policy={}", output.environment_policy.display());
    println!("runner_receipt={}", output.runner_receipt.display());
    println!(
        "precommit_inventory={}",
        output.precommit_inventory.display()
    );
    Ok(())
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
    let mut output_dir = None;

    while let Some(flag) = values.next() {
        let flag_text = flag.to_string_lossy();
        match flag_text.as_ref() {
            "-h" | "--help" => return Err(USAGE.into()),
            "--gate" => {
                let value = next_value(&mut values, "--gate")?;
                gate = Some(value.to_string_lossy().parse::<PerfGate>()?);
            }
            "--hardware-class" => {
                let value = next_value(&mut values, "--hardware-class")?;
                hardware_class_id = Some(parse_hardware_class_id(&value)?);
            }
            "--execution-profile" => {
                let value = next_value(&mut values, "--execution-profile")?;
                execution_profile_id = Some(parse_execution_profile_id(&value)?);
            }
            "--run-id" => {
                run_id = Some(
                    next_value(&mut values, "--run-id")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--run-window" => {
                run_window = Some(
                    next_value(&mut values, "--run-window")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--runs" => {
                let value = next_value(&mut values, "--runs")?;
                measurement_runs = Some(value.to_string_lossy().parse::<usize>()?);
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_value(&mut values, "--output-dir")?));
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
        output_dir: output_dir.ok_or("missing --output-dir")?,
    })
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
        let source = include_str!("quill_perf_finalize.rs");
        let run_body = source
            .split("fn run() -> Result<(), Box<dyn Error>>")
            .nth(1)
            .and_then(|suffix| suffix.split("#[cfg(test)]").next())
            .expect("production run body");
        let typed_runner = source
            .find("run_local_perf_command(&LocalPerfRunConfig")
            .expect("typed runner invocation");
        assert!(typed_runner > 0);
        assert!(!run_body.contains("sha256"));
        assert!(!run_body.contains("open_executing_image"));
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
