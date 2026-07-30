#![forbid(unsafe_code)]
//! Evaluate Quill QG artifacts against a committed pass-over-pass baseline.

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};

use frankensearch_quill_gauntlet::{
    MachineClassAdmissionContext, MachineClassRegistry, PerfEvidenceArtifact, PerfEvidenceFile,
    PerfGate, PerfGateArtifact, PerfGateDecision, PerfRatchetMode, PerfRatchetRequest,
    VerifiedRunnerIdentity, evaluate_perf_ratchet, is_explicit_bootstrap_for,
    perf_manifest_contract_sha256,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const USAGE: &str = "\
Usage:
  quill-perf-ratchet \\
    --manifest <docs/contracts/quill-perf-gates.toml> \\
    --baseline <latest-pointer-or-bootstrap/legacy-threshold.json> \\
    [--baseline-evidence <legacy-direct-threshold.evidence.json>] \\
    --candidate <QG-N.json> \\
    [--candidate-evidence <QG-N.evidence.json>] \\
    [--rerun <QG-N.json>] \\
    [--rerun-evidence <QG-N.rerun.evidence.json>] \\
    [--candidate-runner-receipt <candidate.runner.json>] \\
    [--rerun-runner-receipt <rerun.runner.json>] \\
    [--candidate-artifact-manifest <candidate.artifacts.json>] \\
    [--rerun-artifact-manifest <rerun.artifacts.json>] \\
    [--candidate-run-log <candidate/run.log>] \\
    [--rerun-run-log <rerun/run.log>] \\
    [--machine-class <expected-canonical-class>] \\
    --output <ratchet.json> \\
    --mode <promotion|regression-alarm> \\
    [--promote-dir <.bench-history> --date <YYYY-MM-DD>]

Exact unmeasured bootstrap: omit baseline evidence.
Current measured pointer: omit baseline evidence; the pointer binds and resolves it.
Legacy direct measured threshold: supply its already-bound evidence explicitly.
Exit status: 0=Allow, 1=Block, 2=Quarantine, 64=invalid invocation.";

type LoadedEvidence = (PerfEvidenceArtifact, Vec<u8>);
type AdmittedRunnerReceipt = (VerifiedRunnerIdentity, Vec<u8>, Vec<u8>, Vec<u8>);

const HISTORY_POINTER_SCHEMA_VERSION: &str = "frankensearch.perf-history-pointer.v1";

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    baseline: PathBuf,
    baseline_evidence: Option<PathBuf>,
    candidate: PathBuf,
    candidate_evidence: Option<PathBuf>,
    rerun: Option<PathBuf>,
    rerun_evidence: Option<PathBuf>,
    candidate_runner_receipt: Option<PathBuf>,
    rerun_runner_receipt: Option<PathBuf>,
    candidate_artifact_manifest: Option<PathBuf>,
    rerun_artifact_manifest: Option<PathBuf>,
    candidate_run_log: Option<PathBuf>,
    rerun_run_log: Option<PathBuf>,
    output: PathBuf,
    mode: PerfRatchetMode,
    promote_dir: Option<PathBuf>,
    machine_class: Option<String>,
    date: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryPointer {
    schema_version: String,
    gate: PerfGate,
    machine_class: String,
    run_id: String,
    threshold_file: String,
    threshold_sha256: String,
    evidence_file: String,
    evidence_sha256: String,
}

#[derive(Debug)]
struct LoadedBaseline {
    artifact: PerfGateArtifact,
    artifact_bytes: Vec<u8>,
    artifact_path: PathBuf,
    evidence: Option<LoadedEvidence>,
    evidence_path: Option<PathBuf>,
    pointer: Option<(PathBuf, Vec<u8>)>,
}

#[derive(Debug)]
struct HistoryPublicationPlan {
    rolling_threshold: PathBuf,
    threshold_bytes: Vec<u8>,
    rolling_evidence: PathBuf,
    evidence_bytes: Vec<u8>,
    latest_pointer: PathBuf,
    pointer_bytes: Vec<u8>,
}

fn main() -> ExitCode {
    match run() {
        Ok(decision) => match decision {
            PerfGateDecision::Allow => ExitCode::SUCCESS,
            PerfGateDecision::Block => ExitCode::from(1),
            PerfGateDecision::Quarantine => ExitCode::from(2),
        },
        Err(error) => {
            eprintln!("quill-perf-ratchet: {error}");
            eprintln!("{USAGE}");
            ExitCode::from(64)
        }
    }
}

fn run() -> Result<PerfGateDecision, Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    validate_decision_output_is_separate(&args)?;
    let manifest_bytes = fs::read(&args.manifest)?;
    let manifest_text = std::str::from_utf8(&manifest_bytes)?;
    let manifest_sha256 = perf_manifest_contract_sha256(manifest_text);
    let manifest = toml::from_str::<toml::Value>(manifest_text)?;

    let loaded_baseline = read_baseline(&args.baseline, args.baseline_evidence.as_deref())?;
    let baseline = loaded_baseline.artifact;
    let baseline_bytes = loaded_baseline.artifact_bytes;
    let baseline_artifact_path = loaded_baseline.artifact_path;
    let baseline_evidence = loaded_baseline.evidence;
    let baseline_evidence_path = loaded_baseline.evidence_path;
    let baseline_pointer = loaded_baseline.pointer;
    let (candidate, candidate_bytes) = read_artifact(&args.candidate)?;
    let mut candidate_evidence = args
        .candidate_evidence
        .as_deref()
        .map(read_evidence_artifact)
        .transpose()?;
    let baseline_is_bootstrap =
        is_explicit_bootstrap_for(&baseline, candidate.gate, &manifest_sha256);
    validate_baseline_identity_inputs(
        args.mode,
        baseline_is_bootstrap,
        baseline_evidence.is_some(),
    )?;
    let rerun = args.rerun.as_deref().map(read_artifact).transpose()?;
    let mut rerun_evidence = args
        .rerun_evidence
        .as_deref()
        .map(read_evidence_artifact)
        .transpose()?;
    let activated = gate_activated(&manifest, candidate.gate)?;
    let registry = (args.mode == PerfRatchetMode::Promotion)
        .then(MachineClassRegistry::frozen)
        .transpose()?;
    let candidate_runner = read_runner_identity(
        registry.as_ref(),
        args.candidate_runner_receipt.as_deref(),
        args.candidate_artifact_manifest.as_deref(),
        args.candidate_run_log.as_deref(),
        candidate.gate,
        args.machine_class.as_deref(),
        Some(&candidate_bytes),
        candidate_evidence
            .as_ref()
            .map(|(_, bytes)| bytes.as_slice()),
    )?;
    let rerun_runner = read_runner_identity(
        registry.as_ref(),
        args.rerun_runner_receipt.as_deref(),
        args.rerun_artifact_manifest.as_deref(),
        args.rerun_run_log.as_deref(),
        candidate.gate,
        args.machine_class.as_deref(),
        rerun.as_ref().map(|(_, bytes)| bytes.as_slice()),
        rerun_evidence.as_ref().map(|(_, bytes)| bytes.as_slice()),
    )?;
    let candidate_evidence_source = candidate_evidence.as_ref().map(|(_, bytes)| bytes.clone());
    let rerun_evidence_source = rerun_evidence.as_ref().map(|(_, bytes)| bytes.clone());
    if args.mode == PerfRatchetMode::Promotion {
        bind_evidence_to_runner(
            "candidate",
            &candidate_bytes,
            &mut candidate_evidence,
            candidate_runner.as_ref(),
        )?;
        bind_evidence_to_runner(
            "rerun",
            rerun
                .as_ref()
                .map(|(_, bytes)| bytes.as_slice())
                .ok_or("promotion rerun bytes are missing")?,
            &mut rerun_evidence,
            rerun_runner.as_ref(),
        )?;
    }

    let mut evidence_files = vec![
        evidence("manifest", &args.manifest, &manifest_bytes),
        evidence("baseline", &baseline_artifact_path, &baseline_bytes),
        evidence("candidate", &args.candidate, &candidate_bytes),
    ];
    if let Some((pointer_path, pointer_bytes)) = baseline_pointer.as_ref() {
        evidence_files.push(evidence(
            "baseline_history_pointer",
            pointer_path,
            pointer_bytes,
        ));
    }
    if let Some((_, bound_bytes)) = baseline_evidence.as_ref() {
        let evidence_path = baseline_evidence_path
            .as_ref()
            .ok_or("loaded baseline evidence has no source path")?;
        evidence_files.push(evidence(
            "baseline_current_bound",
            evidence_path,
            bound_bytes,
        ));
    }
    if let (Some(rerun_path), Some((_, rerun_bytes))) = (args.rerun.as_deref(), rerun.as_ref()) {
        evidence_files.push(evidence("rerun", rerun_path, rerun_bytes));
    }
    if let (Some(path), Some(source_bytes), Some((_, bound_bytes))) = (
        args.candidate_evidence.as_deref(),
        candidate_evidence_source.as_deref(),
        candidate_evidence.as_ref(),
    ) {
        evidence_files.push(evidence("candidate_current_source", path, source_bytes));
        evidence_files.push(bound_evidence("candidate_current_bound", path, bound_bytes));
    }
    if let (Some(path), Some(source_bytes), Some((_, bound_bytes))) = (
        args.rerun_evidence.as_deref(),
        rerun_evidence_source.as_deref(),
        rerun_evidence.as_ref(),
    ) {
        evidence_files.push(evidence("rerun_current_source", path, source_bytes));
        evidence_files.push(bound_evidence("rerun_current_bound", path, bound_bytes));
    }
    for (role, path, admitted) in [
        (
            "candidate_runner_receipt",
            args.candidate_runner_receipt.as_deref(),
            candidate_runner.as_ref(),
        ),
        (
            "rerun_runner_receipt",
            args.rerun_runner_receipt.as_deref(),
            rerun_runner.as_ref(),
        ),
    ] {
        if let (Some(path), Some((_, bytes, _, _))) = (path, admitted) {
            evidence_files.push(evidence(role, path, bytes));
        }
    }
    for (role, path, admitted) in [
        (
            "candidate_artifact_manifest",
            args.candidate_artifact_manifest.as_deref(),
            candidate_runner.as_ref(),
        ),
        (
            "rerun_artifact_manifest",
            args.rerun_artifact_manifest.as_deref(),
            rerun_runner.as_ref(),
        ),
    ] {
        if let (Some(path), Some((_, _, bytes, _))) = (path, admitted) {
            evidence_files.push(evidence(role, path, bytes));
        }
    }
    for (role, path, admitted) in [
        (
            "candidate_run_log",
            args.candidate_run_log.as_deref(),
            candidate_runner.as_ref(),
        ),
        (
            "rerun_run_log",
            args.rerun_run_log.as_deref(),
            rerun_runner.as_ref(),
        ),
    ] {
        if let (Some(path), Some((_, _, _, bytes))) = (path, admitted) {
            evidence_files.push(evidence(role, path, bytes));
        }
    }

    let mut evaluation = evaluate_perf_ratchet(PerfRatchetRequest {
        baseline: Some(&baseline),
        baseline_evidence: baseline_evidence.as_ref().map(|(artifact, _)| artifact),
        candidate: &candidate,
        rerun: rerun.as_ref().map(|(artifact, _)| artifact),
        candidate_evidence: candidate_evidence.as_ref().map(|(artifact, _)| artifact),
        rerun_evidence: rerun_evidence.as_ref().map(|(artifact, _)| artifact),
        expected_machine_class: args.machine_class.as_deref(),
        candidate_runner_identity: candidate_runner
            .as_ref()
            .map(|(identity, _, _, _)| identity),
        rerun_runner_identity: rerun_runner.as_ref().map(|(identity, _, _, _)| identity),
        gate_activated: activated,
        mode: args.mode,
        expected_manifest_sha256: &manifest_sha256,
        evidence: evidence_files,
    });

    let history_plan = plan_history_if_allowed(
        &args,
        candidate.gate,
        &candidate.run_id,
        &candidate_bytes,
        candidate_evidence
            .as_ref()
            .map(|(_, bytes)| bytes.as_slice()),
        candidate_runner
            .as_ref()
            .map(|(identity, _, _, _)| identity.class_id()),
        &mut evaluation,
    )?;

    let output = serde_json::to_string_pretty(&evaluation)?;
    write_file(&args.output, format!("{output}\n").as_bytes())?;
    if let Some(plan) = history_plan.as_ref() {
        apply_history_plan(plan)?;
    }
    println!(
        "{} {}: {} (evidence {})",
        evaluation.gate,
        mode_label(evaluation.mode),
        evaluation.decision,
        args.output.display()
    );
    for reason in &evaluation.reasons {
        println!("{}: {}", reason.code, reason.message);
    }
    Ok(evaluation.decision)
}

fn parse_args<I>(mut values: I) -> Result<Args, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let mut manifest = None;
    let mut baseline = None;
    let mut baseline_evidence = None;
    let mut candidate = None;
    let mut candidate_evidence = None;
    let mut rerun = None;
    let mut rerun_evidence = None;
    let mut candidate_runner_receipt = None;
    let mut rerun_runner_receipt = None;
    let mut candidate_artifact_manifest = None;
    let mut rerun_artifact_manifest = None;
    let mut candidate_run_log = None;
    let mut rerun_run_log = None;
    let mut output = None;
    let mut mode = None;
    let mut promote_dir = None;
    let mut machine_class = None;
    let mut date = None;

    while let Some(flag) = values.next() {
        match flag.to_string_lossy().as_ref() {
            "-h" | "--help" => return Err(USAGE.into()),
            "--manifest" => manifest = Some(PathBuf::from(next_value(&mut values, "--manifest")?)),
            "--baseline" => baseline = Some(PathBuf::from(next_value(&mut values, "--baseline")?)),
            "--baseline-evidence" => {
                baseline_evidence = Some(PathBuf::from(next_value(
                    &mut values,
                    "--baseline-evidence",
                )?));
            }
            "--candidate" => {
                candidate = Some(PathBuf::from(next_value(&mut values, "--candidate")?));
            }
            "--candidate-evidence" => {
                candidate_evidence = Some(PathBuf::from(next_value(
                    &mut values,
                    "--candidate-evidence",
                )?));
            }
            "--rerun" => rerun = Some(PathBuf::from(next_value(&mut values, "--rerun")?)),
            "--rerun-evidence" => {
                rerun_evidence = Some(PathBuf::from(next_value(&mut values, "--rerun-evidence")?));
            }
            "--candidate-runner-receipt" => {
                candidate_runner_receipt = Some(PathBuf::from(next_value(
                    &mut values,
                    "--candidate-runner-receipt",
                )?));
            }
            "--rerun-runner-receipt" => {
                rerun_runner_receipt = Some(PathBuf::from(next_value(
                    &mut values,
                    "--rerun-runner-receipt",
                )?));
            }
            "--candidate-artifact-manifest" => {
                candidate_artifact_manifest = Some(PathBuf::from(next_value(
                    &mut values,
                    "--candidate-artifact-manifest",
                )?));
            }
            "--rerun-artifact-manifest" => {
                rerun_artifact_manifest = Some(PathBuf::from(next_value(
                    &mut values,
                    "--rerun-artifact-manifest",
                )?));
            }
            "--candidate-run-log" => {
                candidate_run_log = Some(PathBuf::from(next_value(
                    &mut values,
                    "--candidate-run-log",
                )?));
            }
            "--rerun-run-log" => {
                rerun_run_log = Some(PathBuf::from(next_value(&mut values, "--rerun-run-log")?));
            }
            "--output" => output = Some(PathBuf::from(next_value(&mut values, "--output")?)),
            "--mode" => {
                let value = next_value(&mut values, "--mode")?;
                mode = Some(match value.to_string_lossy().as_ref() {
                    "promotion" => PerfRatchetMode::Promotion,
                    "regression-alarm" => PerfRatchetMode::RegressionAlarm,
                    other => return Err(format!("invalid --mode {other:?}").into()),
                });
            }
            "--promote-dir" => {
                promote_dir = Some(PathBuf::from(next_value(&mut values, "--promote-dir")?));
            }
            "--machine-class" => {
                machine_class = Some(
                    next_value(&mut values, "--machine-class")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--date" => {
                date = Some(
                    next_value(&mut values, "--date")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let mode = mode.ok_or("missing --mode")?;
    let history_fields = [promote_dir.is_some(), date.is_some()];
    if history_fields.iter().any(|present| *present)
        && !history_fields.iter().all(|present| *present)
    {
        return Err("--promote-dir and --date must be supplied together".into());
    }
    let receipt_fields = [
        candidate_runner_receipt.is_some(),
        rerun_runner_receipt.is_some(),
        candidate_artifact_manifest.is_some(),
        rerun_artifact_manifest.is_some(),
        candidate_run_log.is_some(),
        rerun_run_log.is_some(),
    ];
    if rerun_evidence.is_some() && rerun.is_none() {
        return Err("--rerun-evidence requires --rerun".into());
    }
    if mode == PerfRatchetMode::Promotion {
        let missing = [
            (candidate_evidence.is_none(), "--candidate-evidence"),
            (rerun.is_none(), "--rerun"),
            (rerun_evidence.is_none(), "--rerun-evidence"),
            (
                candidate_runner_receipt.is_none(),
                "--candidate-runner-receipt",
            ),
            (rerun_runner_receipt.is_none(), "--rerun-runner-receipt"),
            (
                candidate_artifact_manifest.is_none(),
                "--candidate-artifact-manifest",
            ),
            (
                rerun_artifact_manifest.is_none(),
                "--rerun-artifact-manifest",
            ),
            (candidate_run_log.is_none(), "--candidate-run-log"),
            (rerun_run_log.is_none(), "--rerun-run-log"),
            (machine_class.is_none(), "--machine-class"),
        ]
        .into_iter()
        .filter_map(|(missing, flag)| missing.then_some(flag))
        .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "promotion is missing required inputs: {}",
                missing.join(", ")
            )
            .into());
        }
    } else if promote_dir.is_some()
        || machine_class.is_some()
        || receipt_fields.iter().any(|present| *present)
    {
        return Err(
            "regression-alarm mode cannot receive promotion history or runner identity inputs"
                .into(),
        );
    }
    if let Some(label) = machine_class.as_deref() {
        validate_component(label, "machine class")?;
    }
    if let Some(value) = date.as_deref() {
        validate_component(value, "date")?;
    }

    Ok(Args {
        manifest: manifest.ok_or("missing --manifest")?,
        baseline: baseline.ok_or("missing --baseline")?,
        baseline_evidence,
        candidate: candidate.ok_or("missing --candidate")?,
        candidate_evidence,
        rerun,
        rerun_evidence,
        candidate_runner_receipt,
        rerun_runner_receipt,
        candidate_artifact_manifest,
        rerun_artifact_manifest,
        candidate_run_log,
        rerun_run_log,
        output: output.ok_or("missing --output")?,
        mode,
        promote_dir,
        machine_class,
        date,
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

fn validate_component(value: &str, field: &str) -> Result<(), Box<dyn Error>> {
    validate_filename_component(value, field, 112)
}

fn validate_filename_component(
    value: &str,
    field: &str,
    max_len: usize,
) -> Result<(), Box<dyn Error>> {
    if value.is_empty()
        || value.len() > max_len
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{field} {value:?} is not a safe filename component").into());
    }
    Ok(())
}

fn validate_decision_output_is_separate(args: &Args) -> Result<(), Box<dyn Error>> {
    let output = normalize_path(&args.output)?;
    let inputs = [
        Some(args.manifest.as_path()),
        Some(args.baseline.as_path()),
        args.baseline_evidence.as_deref(),
        Some(args.candidate.as_path()),
        args.candidate_evidence.as_deref(),
        args.rerun.as_deref(),
        args.rerun_evidence.as_deref(),
        args.candidate_runner_receipt.as_deref(),
        args.rerun_runner_receipt.as_deref(),
        args.candidate_artifact_manifest.as_deref(),
        args.rerun_artifact_manifest.as_deref(),
        args.candidate_run_log.as_deref(),
        args.rerun_run_log.as_deref(),
    ];
    for input in inputs.into_iter().flatten() {
        if output == normalize_path(input)? {
            return Err(format!(
                "decision output {} aliases input {}",
                args.output.display(),
                input.display()
            )
            .into());
        }
    }
    if let Some(history_dir) = args.promote_dir.as_deref() {
        let history = normalize_path(history_dir)?;
        if output.starts_with(&history) {
            return Err(format!(
                "decision output {} must remain outside promotion history {}",
                args.output.display(),
                history_dir.display()
            )
            .into());
        }
    }
    Ok(())
}

fn normalize_path(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut existing = absolute.as_path();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let file_name = existing
            .file_name()
            .ok_or_else(|| format!("path {} has no existing ancestor", path.display()))?;
        suffix.push(file_name.to_os_string());
        existing = existing
            .parent()
            .ok_or_else(|| format!("path {} has no parent", path.display()))?;
    }
    let mut normalized = fs::canonicalize(existing)?;
    for component in suffix.iter().rev() {
        normalized.push(component);
    }
    Ok(normalized)
}

fn read_baseline(
    path: &Path,
    explicit_evidence_path: Option<&Path>,
) -> Result<LoadedBaseline, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if serde_json::from_slice::<PerfGateArtifact>(&bytes).is_ok() {
        let (artifact, artifact_bytes) = read_artifact(path)?;
        let evidence = explicit_evidence_path
            .map(read_evidence_artifact)
            .transpose()?;
        return Ok(LoadedBaseline {
            artifact,
            artifact_bytes,
            artifact_path: path.to_path_buf(),
            evidence,
            evidence_path: explicit_evidence_path.map(Path::to_path_buf),
            pointer: None,
        });
    }
    if explicit_evidence_path.is_some() {
        return Err(
            "a history-pointer baseline resolves its own evidence; omit --baseline-evidence".into(),
        );
    }

    let pointer = serde_json::from_slice::<HistoryPointer>(&bytes)?;
    if pointer.schema_version != HISTORY_POINTER_SCHEMA_VERSION
        || serde_json::to_vec_pretty(&pointer)? != bytes
    {
        return Err(format!(
            "history pointer {} has a stale schema or noncanonical bytes",
            path.display()
        )
        .into());
    }
    validate_component(&pointer.machine_class, "history pointer machine class")?;
    validate_component(&pointer.run_id, "history pointer run ID")?;
    validate_filename_component(
        &pointer.threshold_file,
        "history pointer threshold file",
        240,
    )?;
    validate_filename_component(&pointer.evidence_file, "history pointer evidence file", 240)?;
    let expected_pointer_name = format!(
        "{}.{}.latest.json",
        pointer.gate.label(),
        pointer.machine_class
    );
    if path.file_name().and_then(|name| name.to_str()) != Some(expected_pointer_name.as_str()) {
        return Err(format!(
            "history pointer {} does not use canonical gate/class latest basename {}",
            path.display(),
            expected_pointer_name
        )
        .into());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let threshold_path = parent.join(&pointer.threshold_file);
    let evidence_path = parent.join(&pointer.evidence_file);
    let (artifact, artifact_bytes) = read_artifact(&threshold_path)?;
    let evidence = read_evidence_artifact(&evidence_path)?;
    if sha256_hex(&artifact_bytes) != pointer.threshold_sha256
        || sha256_hex(&evidence.1) != pointer.evidence_sha256
        || artifact.gate != pointer.gate
        || artifact.run_id != pointer.run_id
        || evidence.0.gate != pointer.gate
        || evidence.0.provenance.run_id != pointer.run_id
    {
        return Err(format!(
            "history pointer {} does not bind its exact threshold/evidence generation",
            path.display()
        )
        .into());
    }
    Ok(LoadedBaseline {
        artifact,
        artifact_bytes,
        artifact_path: threshold_path,
        evidence: Some(evidence),
        evidence_path: Some(evidence_path),
        pointer: Some((path.to_path_buf(), bytes)),
    })
}

fn read_artifact(path: &Path) -> Result<(PerfGateArtifact, Vec<u8>), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let artifact = serde_json::from_slice::<PerfGateArtifact>(&bytes)?;
    let canonical = serde_json::to_vec_pretty(&artifact)?;
    if bytes != canonical {
        return Err(format!(
            "threshold artifact {} is not exact canonical pretty JSON",
            path.display()
        )
        .into());
    }
    Ok((artifact, bytes))
}

fn read_evidence_artifact(path: &Path) -> Result<LoadedEvidence, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let artifact = PerfEvidenceArtifact::from_verified_slice(&bytes)?;
    let canonical = serde_json::to_vec_pretty(&artifact)?;
    if bytes != canonical {
        return Err(format!(
            "evidence artifact {} is not exact canonical pretty JSON",
            path.display()
        )
        .into());
    }
    Ok((artifact, bytes))
}

fn read_runner_identity(
    registry: Option<&MachineClassRegistry>,
    receipt_path: Option<&Path>,
    artifact_manifest_path: Option<&Path>,
    run_log_path: Option<&Path>,
    gate: PerfGate,
    expected_class: Option<&str>,
    threshold_artifact_bytes: Option<&[u8]>,
    evidence_artifact_bytes: Option<&[u8]>,
) -> Result<Option<AdmittedRunnerReceipt>, Box<dyn Error>> {
    if registry.is_none()
        && receipt_path.is_none()
        && artifact_manifest_path.is_none()
        && run_log_path.is_none()
        && expected_class.is_none()
    {
        return Ok(None);
    }
    let (
        Some(registry),
        Some(receipt_path),
        Some(artifact_manifest_path),
        Some(run_log_path),
        Some(expected_class),
        Some(threshold_artifact_bytes),
        Some(evidence_artifact_bytes),
    ) = (
        registry,
        receipt_path,
        artifact_manifest_path,
        run_log_path,
        expected_class,
        threshold_artifact_bytes,
        evidence_artifact_bytes,
    )
    else {
        return Err("runner receipt/artifact-manifest admission inputs are incomplete".into());
    };
    let receipt_bytes = fs::read(receipt_path)?;
    let artifact_manifest_bytes = fs::read(artifact_manifest_path)?;
    let run_log_bytes = fs::read(run_log_path)?;
    let gate = gate.label();
    let context = MachineClassAdmissionContext {
        gate: gate.to_owned(),
        destination_basename: format!("{gate}.{expected_class}.latest.json"),
    };
    let identity = registry
        .admit(&receipt_bytes, &context)?
        .bind_artifact_manifest(
            &artifact_manifest_bytes,
            &run_log_bytes,
            threshold_artifact_bytes,
            evidence_artifact_bytes,
        )?;
    if identity.class_id() != expected_class {
        return Err(format!(
            "runner receipt derives machine class {:?}, expected {expected_class:?}",
            identity.class_id()
        )
        .into());
    }
    Ok(Some((
        identity,
        receipt_bytes,
        artifact_manifest_bytes,
        run_log_bytes,
    )))
}

fn bind_evidence_to_runner(
    role: &str,
    threshold_artifact_bytes: &[u8],
    evidence: &mut Option<LoadedEvidence>,
    runner: Option<&AdmittedRunnerReceipt>,
) -> Result<(), Box<dyn Error>> {
    let (Some((artifact, bytes)), Some((identity, _, _, _))) = (evidence.as_mut(), runner) else {
        return Err(format!("{role} evidence/runner finalization inputs are incomplete").into());
    };
    let prebinding_bytes = bytes.clone();
    *bytes = artifact.bind_machine_class_identity_and_seal(
        identity.clone(),
        threshold_artifact_bytes,
        &prebinding_bytes,
    )?;
    Ok(())
}

fn validate_baseline_identity_inputs(
    mode: PerfRatchetMode,
    baseline_is_bootstrap: bool,
    has_baseline_evidence: bool,
) -> Result<(), Box<dyn Error>> {
    if mode != PerfRatchetMode::Promotion {
        return Ok(());
    }
    if baseline_is_bootstrap && has_baseline_evidence {
        return Err(
            "the exact unmeasured baseline must not receive fabricated current evidence".into(),
        );
    }
    if !baseline_is_bootstrap && !has_baseline_evidence {
        return Err(
            "a measured promotion baseline requires its already-bound committed evidence".into(),
        );
    }
    Ok(())
}

fn gate_activated(manifest: &toml::Value, gate: PerfGate) -> Result<bool, Box<dyn Error>> {
    manifest
        .get("gate")
        .and_then(|gates| gates.get(gate.label()))
        .and_then(|policy| policy.get("activated"))
        .and_then(toml::Value::as_bool)
        .ok_or_else(|| format!("manifest does not define gate.{}.activated", gate.label()).into())
}

fn evidence(role: &str, path: &Path, bytes: &[u8]) -> PerfEvidenceFile {
    PerfEvidenceFile {
        role: role.to_owned(),
        path: path.to_string_lossy().into_owned(),
        sha256: sha256_hex(bytes),
    }
}

fn bound_evidence(role: &str, source_path: &Path, bytes: &[u8]) -> PerfEvidenceFile {
    PerfEvidenceFile {
        role: role.to_owned(),
        path: format!("{}#receipt-bound-in-memory", source_path.to_string_lossy()),
        sha256: sha256_hex(bytes),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn plan_history_if_requested(
    args: &Args,
    gate: PerfGate,
    candidate_run_id: &str,
    candidate_bytes: &[u8],
    candidate_evidence_bytes: Option<&[u8]>,
    verified_machine_class: Option<&str>,
    updates: &mut Vec<PerfEvidenceFile>,
) -> Result<Option<HistoryPublicationPlan>, Box<dyn Error>> {
    let (Some(history_dir), Some(machine_class), Some(date)) = (
        args.promote_dir.as_deref(),
        verified_machine_class,
        args.date.as_deref(),
    ) else {
        return Ok(None);
    };
    let evidence_bytes =
        candidate_evidence_bytes.ok_or("allowed promotion is missing receipt-bound evidence")?;
    validate_component(candidate_run_id, "candidate run ID")?;

    let stem = format!("{}.{}", gate.label(), machine_class);
    let rolling_stem = format!("{stem}.{date}.{candidate_run_id}");
    let threshold_file = format!("{rolling_stem}.json");
    let evidence_file = format!("{rolling_stem}.evidence.json");
    validate_filename_component(&threshold_file, "rolling threshold filename", 240)?;
    validate_filename_component(&evidence_file, "rolling evidence filename", 240)?;
    let rolling_threshold = history_dir.join(&threshold_file);
    let rolling_evidence = history_dir.join(&evidence_file);
    let latest_pointer = history_dir.join(format!("{stem}.latest.json"));
    let pointer = HistoryPointer {
        schema_version: HISTORY_POINTER_SCHEMA_VERSION.to_owned(),
        gate,
        machine_class: machine_class.to_owned(),
        run_id: candidate_run_id.to_owned(),
        threshold_file,
        threshold_sha256: sha256_hex(candidate_bytes),
        evidence_file,
        evidence_sha256: sha256_hex(evidence_bytes),
    };
    let pointer_bytes = serde_json::to_vec_pretty(&pointer)?;
    updates.push(evidence(
        "history_window",
        &rolling_threshold,
        candidate_bytes,
    ));
    updates.push(evidence(
        "history_evidence_window",
        &rolling_evidence,
        evidence_bytes,
    ));
    updates.push(evidence(
        "history_latest_pointer",
        &latest_pointer,
        &pointer_bytes,
    ));
    Ok(Some(HistoryPublicationPlan {
        rolling_threshold,
        threshold_bytes: candidate_bytes.to_vec(),
        rolling_evidence,
        evidence_bytes: evidence_bytes.to_vec(),
        latest_pointer,
        pointer_bytes,
    }))
}

fn plan_history_if_allowed(
    args: &Args,
    gate: PerfGate,
    candidate_run_id: &str,
    candidate_bytes: &[u8],
    candidate_evidence_bytes: Option<&[u8]>,
    verified_machine_class: Option<&str>,
    evaluation: &mut frankensearch_quill_gauntlet::PerfRatchetEvaluation,
) -> Result<Option<HistoryPublicationPlan>, Box<dyn Error>> {
    if evaluation.decision != PerfGateDecision::Allow {
        return Ok(None);
    }
    plan_history_if_requested(
        args,
        gate,
        candidate_run_id,
        candidate_bytes,
        candidate_evidence_bytes,
        verified_machine_class,
        &mut evaluation.history_updates,
    )
}

fn apply_history_plan(plan: &HistoryPublicationPlan) -> Result<(), Box<dyn Error>> {
    write_immutable_file(&plan.rolling_threshold, &plan.threshold_bytes)?;
    write_immutable_file(&plan.rolling_evidence, &plan.evidence_bytes)?;
    // This one atomic pointer replacement is the only baseline advancement.
    // It happens after the complete immutable generation and decision record
    // are durable, so a crash cannot expose a mixed threshold/evidence pair.
    write_file(&plan.latest_pointer, &plan.pointer_bytes)
}

fn write_immutable_file(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(path)?;
            if existing != bytes {
                return Err(format!(
                    "immutable history object {} already exists with different bytes",
                    path.display()
                )
                .into());
            }
            fs::File::open(path)?.sync_all()?;
        }
        Err(error) => return Err(error.into()),
    }
    #[cfg(unix)]
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("output path has no UTF-8 file name")?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{file_name}.tmp-{}-{nonce}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    #[cfg(unix)]
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

const fn mode_label(mode: PerfRatchetMode) -> &'static str {
    match mode {
        PerfRatchetMode::Promotion => "promotion",
        PerfRatchetMode::RegressionAlarm => "regression-alarm",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn test_args(history_dir: &Path) -> Args {
        Args {
            manifest: PathBuf::from("manifest.toml"),
            baseline: PathBuf::from("baseline.json"),
            baseline_evidence: Some(PathBuf::from("baseline.evidence.json")),
            candidate: PathBuf::from("candidate.json"),
            candidate_evidence: Some(PathBuf::from("candidate.evidence.json")),
            rerun: Some(PathBuf::from("rerun.json")),
            rerun_evidence: Some(PathBuf::from("rerun.evidence.json")),
            candidate_runner_receipt: Some(PathBuf::from("candidate.runner.json")),
            rerun_runner_receipt: Some(PathBuf::from("rerun.runner.json")),
            candidate_artifact_manifest: Some(PathBuf::from("candidate.artifacts.json")),
            rerun_artifact_manifest: Some(PathBuf::from("rerun.artifacts.json")),
            candidate_run_log: Some(PathBuf::from("candidate/run.log")),
            rerun_run_log: Some(PathBuf::from("rerun/run.log")),
            output: PathBuf::from("ratchet.json"),
            mode: PerfRatchetMode::Promotion,
            promote_dir: Some(history_dir.to_path_buf()),
            machine_class: Some("trj-zen3-16c-smt2".to_owned()),
            date: Some("2026-07-29".to_owned()),
        }
    }

    fn evaluation(
        decision: PerfGateDecision,
    ) -> frankensearch_quill_gauntlet::PerfRatchetEvaluation {
        frankensearch_quill_gauntlet::PerfRatchetEvaluation {
            schema_version: frankensearch_quill_gauntlet::PERF_RATCHET_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg6,
            mode: PerfRatchetMode::Promotion,
            gate_activated: true,
            decision,
            reasons: Vec::new(),
            comparisons: Vec::new(),
            evidence: Vec::new(),
            history_updates: Vec::new(),
        }
    }

    fn history_files(history_dir: &Path) -> [PathBuf; 3] {
        [
            history_dir.join("QG-6.trj-zen3-16c-smt2.2026-07-29.candidate-1.json"),
            history_dir.join("QG-6.trj-zen3-16c-smt2.2026-07-29.candidate-1.evidence.json"),
            history_dir.join("QG-6.trj-zen3-16c-smt2.latest.json"),
        ]
    }

    fn snapshot(directory: &Path) -> BTreeMap<String, Vec<u8>> {
        fs::read_dir(directory)
            .expect("history directory")
            .map(|entry| {
                let entry = entry.expect("history entry");
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    fs::read(entry.path()).expect("history bytes"),
                )
            })
            .collect()
    }

    #[test]
    fn promotion_options_are_all_or_nothing() {
        let result = parse_args(
            [
                "--manifest",
                "manifest.toml",
                "--baseline",
                "baseline.json",
                "--candidate",
                "candidate.json",
                "--output",
                "out.json",
                "--mode",
                "promotion",
                "--promote-dir",
                ".bench-history",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err());
    }

    #[test]
    fn history_components_reject_path_traversal() {
        assert!(validate_component("../worker", "machine class").is_err());
        assert!(validate_component("github-ubuntu", "machine class").is_ok());
        assert!(validate_component("2026-07-23", "date").is_ok());
    }

    #[test]
    fn runner_receipt_options_are_all_or_nothing() {
        let result = parse_args(
            [
                "--manifest",
                "manifest.toml",
                "--baseline",
                "baseline.json",
                "--candidate",
                "candidate.json",
                "--output",
                "out.json",
                "--mode",
                "promotion",
                "--candidate-runner-receipt",
                "candidate.runner.json",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err());
    }

    #[test]
    fn promotion_parser_allows_exact_bootstrap_to_omit_only_baseline_identity_inputs() {
        let result = parse_args(
            [
                "--manifest",
                "manifest.toml",
                "--baseline",
                "QG-2.unmeasured.latest.json",
                "--candidate",
                "candidate.json",
                "--candidate-evidence",
                "candidate.evidence.json",
                "--rerun",
                "rerun.json",
                "--rerun-evidence",
                "rerun.evidence.json",
                "--candidate-runner-receipt",
                "candidate.runner.json",
                "--rerun-runner-receipt",
                "rerun.runner.json",
                "--candidate-artifact-manifest",
                "candidate.artifacts.json",
                "--rerun-artifact-manifest",
                "rerun.artifacts.json",
                "--candidate-run-log",
                "candidate/run.log",
                "--rerun-run-log",
                "rerun/run.log",
                "--machine-class",
                "trj-zen3-16c-smt2",
                "--output",
                "out.json",
                "--mode",
                "promotion",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .expect("bootstrap-shaped CLI inputs pass structural parsing");

        assert!(result.baseline_evidence.is_none());
        assert!(result.candidate_runner_receipt.is_some());
        assert!(result.rerun_runner_receipt.is_some());
        assert!(result.candidate_artifact_manifest.is_some());
        assert!(result.rerun_artifact_manifest.is_some());
        assert!(result.candidate_run_log.is_some());
        assert!(result.rerun_run_log.is_some());
    }

    #[test]
    fn cli_baseline_binding_exemption_is_narrowly_exact() {
        assert!(validate_baseline_identity_inputs(PerfRatchetMode::Promotion, true, false).is_ok());
        assert!(validate_baseline_identity_inputs(PerfRatchetMode::Promotion, true, true).is_err());
        assert!(
            validate_baseline_identity_inputs(PerfRatchetMode::Promotion, false, false).is_err()
        );
        assert!(validate_baseline_identity_inputs(PerfRatchetMode::Promotion, false, true).is_ok());
    }

    #[test]
    fn cli_recognizes_committed_bootstrap_without_baseline_identity() {
        let baseline = serde_json::from_str::<PerfGateArtifact>(include_str!(
            "../../../../.bench-history/QG-2.unmeasured.latest.json"
        ))
        .expect("committed QG-2 bootstrap artifact");
        let manifest = include_str!("../../../../docs/contracts/quill-perf-gates.toml");
        let manifest_sha256 = perf_manifest_contract_sha256(manifest);

        assert!(is_explicit_bootstrap_for(
            &baseline,
            PerfGate::Qg2,
            &manifest_sha256
        ));
        assert!(validate_baseline_identity_inputs(PerfRatchetMode::Promotion, true, false).is_ok());
    }

    fn assert_denial_leaves_history_unchanged(scenario: &str, decision: PerfGateDecision) {
        let directory = tempfile::tempdir().expect("history directory");
        let args = test_args(directory.path());
        for (index, path) in history_files(directory.path()).iter().enumerate() {
            fs::write(path, format!("original-history-object-{index}\n"))
                .expect("seed history object");
        }
        let before = snapshot(directory.path());
        let mut evaluation = evaluation(decision);
        evaluation
            .reasons
            .push(frankensearch_quill_gauntlet::PerfRatchetReason {
                code: format!("test.{scenario}"),
                message: format!("{scenario} must deny before history mutation"),
            });

        let plan = plan_history_if_allowed(
            &args,
            PerfGate::Qg6,
            "candidate-1",
            b"forbidden-candidate",
            Some(b"forbidden-evidence"),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .expect("denial must be a no-op");

        assert!(plan.is_none());
        assert_eq!(snapshot(directory.path()), before);
        assert!(evaluation.history_updates.is_empty());
        assert_eq!(evaluation.reasons[0].code, format!("test.{scenario}"));
    }

    #[test]
    fn stale_registry_receipt_denial_leaves_history_byte_identical() {
        assert_denial_leaves_history_unchanged("stale_registry_receipt", PerfGateDecision::Block);
    }

    #[test]
    fn mixed_runner_receipt_denial_leaves_history_byte_identical() {
        assert_denial_leaves_history_unchanged("mixed_runner_receipts", PerfGateDecision::Block);
    }

    #[test]
    fn tampered_runner_receipt_denial_leaves_history_byte_identical() {
        assert_denial_leaves_history_unchanged("tampered_runner_receipt", PerfGateDecision::Block);
    }

    #[test]
    fn argv_preimage_drift_denial_leaves_history_byte_identical() {
        assert_denial_leaves_history_unchanged("argv_nul_preimage_drift", PerfGateDecision::Block);
    }

    #[test]
    fn diagnostic_mutation_quarantine_leaves_history_byte_identical() {
        assert_denial_leaves_history_unchanged(
            "candidate_diagnostic_mutation",
            PerfGateDecision::Quarantine,
        );
    }

    #[test]
    fn denial_does_not_create_an_absent_history_directory() {
        let parent = tempfile::tempdir().expect("history parent");
        let history_dir = parent.path().join("not-created");
        let args = test_args(&history_dir);
        let mut evaluation = evaluation(PerfGateDecision::Block);

        let plan = plan_history_if_allowed(
            &args,
            PerfGate::Qg6,
            "candidate-1",
            b"forbidden-candidate",
            Some(b"forbidden-receipt-bound-evidence"),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .expect("denial must not open history");

        assert!(plan.is_none());
        assert!(!history_dir.exists());
        assert!(evaluation.history_updates.is_empty());
    }

    #[test]
    fn allowed_promotion_plans_immutable_generation_then_advances_one_pointer() {
        let directory = tempfile::tempdir().expect("history directory");
        let args = test_args(directory.path());
        let mut evaluation = evaluation(PerfGateDecision::Allow);
        let candidate = b"candidate-threshold-artifact\n";
        let unverified_producer_evidence = b"candidate-unverified-producer-evidence\n";
        let receipt_bound_evidence = b"candidate-receipt-bound-evidence\n";

        let plan = plan_history_if_allowed(
            &args,
            PerfGate::Qg6,
            "candidate-1",
            candidate,
            Some(receipt_bound_evidence),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .expect("valid durable promotion")
        .expect("allow decision plans publication");

        assert!(
            snapshot(directory.path()).is_empty(),
            "planning must not mutate history before the decision record is durable"
        );
        apply_history_plan(&plan).expect("publish planned generation");

        let paths = history_files(directory.path());
        assert_eq!(fs::read(&paths[0]).expect("rolling threshold"), candidate);
        assert_eq!(
            fs::read(&paths[1]).expect("rolling evidence"),
            receipt_bound_evidence
        );
        let pointer_bytes = fs::read(&paths[2]).expect("latest pointer");
        let pointer =
            serde_json::from_slice::<HistoryPointer>(&pointer_bytes).expect("typed latest pointer");
        assert_eq!(pointer.run_id, "candidate-1");
        assert_eq!(
            pointer.threshold_file,
            "QG-6.trj-zen3-16c-smt2.2026-07-29.candidate-1.json"
        );
        assert_eq!(
            pointer.evidence_file,
            "QG-6.trj-zen3-16c-smt2.2026-07-29.candidate-1.evidence.json"
        );
        assert!(
            snapshot(directory.path())
                .values()
                .all(|bytes| bytes.as_slice() != unverified_producer_evidence),
            "promotion must never copy the unverified producer artifact"
        );
        assert_eq!(
            evaluation
                .history_updates
                .iter()
                .map(|update| update.role.as_str())
                .collect::<Vec<_>>(),
            [
                "history_window",
                "history_evidence_window",
                "history_latest_pointer",
            ]
        );
        assert!(
            snapshot(directory.path())
                .keys()
                .all(|name| !name.contains(".tmp-")),
            "durable promotion must leave no temporary siblings"
        );
    }

    #[test]
    fn immutable_generation_retry_is_idempotent() {
        let directory = tempfile::tempdir().expect("history directory");
        let args = test_args(directory.path());
        let mut evaluation = evaluation(PerfGateDecision::Allow);
        let plan = plan_history_if_allowed(
            &args,
            PerfGate::Qg6,
            "candidate-1",
            b"threshold",
            Some(b"bound-evidence"),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .unwrap()
        .unwrap();

        apply_history_plan(&plan).expect("first publication");
        let first = snapshot(directory.path());
        apply_history_plan(&plan).expect("exact publication retry");
        assert_eq!(snapshot(directory.path()), first);
    }

    #[test]
    fn immutable_collision_rejects_without_advancing_latest_pointer() {
        let directory = tempfile::tempdir().expect("history directory");
        let args = test_args(directory.path());
        let mut first_evaluation = evaluation(PerfGateDecision::Allow);
        let first = plan_history_if_allowed(
            &args,
            PerfGate::Qg6,
            "candidate-1",
            b"threshold-one",
            Some(b"evidence-one"),
            Some("trj-zen3-16c-smt2"),
            &mut first_evaluation,
        )
        .unwrap()
        .unwrap();
        apply_history_plan(&first).expect("first publication");
        let pointer_before = fs::read(&first.latest_pointer).expect("first latest pointer");

        let mut second_evaluation = evaluation(PerfGateDecision::Allow);
        let collision = plan_history_if_allowed(
            &args,
            PerfGate::Qg6,
            "candidate-1",
            b"threshold-two",
            Some(b"evidence-two"),
            Some("trj-zen3-16c-smt2"),
            &mut second_evaluation,
        )
        .unwrap()
        .unwrap();
        assert!(apply_history_plan(&collision).is_err());
        assert_eq!(
            fs::read(&first.latest_pointer).expect("latest pointer after rejection"),
            pointer_before
        );
    }

    #[test]
    fn unsafe_or_overlong_run_ids_reject_before_history_opens() {
        let parent = tempfile::tempdir().expect("history parent");
        let history = parent.path().join("absent");
        let args = test_args(&history);
        let overlong = "x".repeat(113);
        for run_id in ["..", "../escape", overlong.as_str()] {
            let mut evaluation = evaluation(PerfGateDecision::Allow);
            assert!(
                plan_history_if_allowed(
                    &args,
                    PerfGate::Qg6,
                    run_id,
                    b"threshold",
                    Some(b"evidence"),
                    Some("trj-zen3-16c-smt2"),
                    &mut evaluation,
                )
                .is_err()
            );
            assert!(!history.exists());
            assert!(evaluation.history_updates.is_empty());
        }
    }

    #[test]
    fn decision_output_cannot_alias_inputs_or_history() {
        let directory = tempfile::tempdir().expect("history directory");
        let mut args = test_args(directory.path());
        args.output = args.candidate.clone();
        assert!(validate_decision_output_is_separate(&args).is_err());

        args.output = directory.path().join("decision.json");
        assert!(validate_decision_output_is_separate(&args).is_err());
    }
}
