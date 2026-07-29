#![forbid(unsafe_code)]
//! Hold the registered-host lease across a benchmark child and finalize its
//! exact artifacts.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use frankensearch_quill_gauntlet::{LocalPerfRunConfig, PerfGate, run_local_perf_command};

const USAGE: &str = "\
Usage:
  quill-perf-finalize \\
    --gate <QG-N> \\
    --class <registered-class> \\
    --run-id <unique-pass-id> \\
    --run-window <candidate-rerun-window> \\
    --thread-budget <positive-integer> \\
    --apple-mode <not-applicable|p-plus-e> \\
    --lease-path <host-local-lock-file> \\
    --output-dir <unique-run-directory>

The typed producer builds and resolves perf_matrix itself from a clean source
snapshot. It requires RCH_DISABLE=1, RCH_CARGO_WRAPPER_BYPASS=1, and an
absolute CARGO_TARGET_DIR outside the source repository.";

#[derive(Debug)]
struct Args {
    gate: PerfGate,
    class_id: String,
    run_id: String,
    run_window: String,
    thread_budget: u64,
    apple_execution_mode: String,
    lease_path: PathBuf,
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
        class_id: args.class_id,
        run_id: args.run_id,
        run_window: args.run_window,
        thread_budget: args.thread_budget,
        apple_execution_mode: args.apple_execution_mode,
        lease_path: args.lease_path,
        output_dir: args.output_dir,
    })?;
    println!("run_log={}", output.run_log.display());
    println!("artifact_manifest={}", output.artifact_manifest.display());
    println!("runner_receipt={}", output.runner_receipt.display());
    println!(
        "precommit_inventory={}",
        output.precommit_inventory.display()
    );
    Ok(())
}

fn parse_args<I>(mut values: I) -> Result<Args, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let mut gate = None;
    let mut class_id = None;
    let mut run_id = None;
    let mut run_window = None;
    let mut thread_budget = None;
    let mut apple_execution_mode = None;
    let mut lease_path = None;
    let mut output_dir = None;

    while let Some(flag) = values.next() {
        let flag_text = flag.to_string_lossy();
        match flag_text.as_ref() {
            "-h" | "--help" => return Err(USAGE.into()),
            "--gate" => {
                let value = next_value(&mut values, "--gate")?;
                gate = Some(value.to_string_lossy().parse::<PerfGate>()?);
            }
            "--class" => {
                class_id = Some(
                    next_value(&mut values, "--class")?
                        .to_string_lossy()
                        .into_owned(),
                );
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
            "--thread-budget" => {
                let value = next_value(&mut values, "--thread-budget")?;
                thread_budget = Some(value.to_string_lossy().parse::<u64>()?);
            }
            "--apple-mode" => {
                apple_execution_mode = Some(
                    next_value(&mut values, "--apple-mode")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--lease-path" => {
                lease_path = Some(PathBuf::from(next_value(&mut values, "--lease-path")?));
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_value(&mut values, "--output-dir")?));
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    Ok(Args {
        gate: gate.ok_or("missing --gate")?,
        class_id: class_id.ok_or("missing --class")?,
        run_id: run_id.ok_or("missing --run-id")?,
        run_window: run_window.ok_or("missing --run-window")?,
        thread_budget: thread_budget.ok_or("missing --thread-budget")?,
        apple_execution_mode: apple_execution_mode.ok_or("missing --apple-mode")?,
        lease_path: lease_path.ok_or("missing --lease-path")?,
        output_dir: output_dir.ok_or("missing --output-dir")?,
    })
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
                "--class",
                "trj-zen3-16c",
                "--run-id",
                "candidate-1",
                "--run-window",
                "window-1",
                "--thread-budget",
                "16",
                "--apple-mode",
                "not-applicable",
                "--lease-path",
                "/var/lock/quill-perf.lock",
                "--output-dir",
                "/tmp/quill-perf/candidate",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .expect("complete producer invocation");
        assert_eq!(args.gate, PerfGate::Qg2);
        assert_eq!(args.thread_budget, 16);
    }

    #[test]
    fn parser_rejects_arbitrary_child_commands() {
        let result = parse_args(
            [
                "--gate",
                "QG-2",
                "--class",
                "trj-zen3-16c",
                "--run-id",
                "candidate-1",
                "--run-window",
                "window-1",
                "--thread-budget",
                "16",
                "--apple-mode",
                "not-applicable",
                "--lease-path",
                "/var/lock/quill-perf.lock",
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
}
