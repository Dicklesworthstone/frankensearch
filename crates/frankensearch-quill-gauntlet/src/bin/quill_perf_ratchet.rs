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
    VerifiedRunnerIdentity, evaluate_perf_ratchet, perf_manifest_contract_sha256,
};
use sha2::{Digest, Sha256};

const USAGE: &str = "\
Usage:
  quill-perf-ratchet \\
    --manifest <docs/contracts/quill-perf-gates.toml> \\
    --baseline <.bench-history/QG-N.machine.latest.json> \\
    [--baseline-evidence <.bench-history/QG-N.machine.latest.evidence.json>] \\
    --candidate <QG-N.json> \\
    [--candidate-evidence <QG-N.evidence.json>] \\
    [--rerun <QG-N.json>] \\
    [--rerun-evidence <QG-N.rerun.evidence.json>] \\
    [--baseline-runner-receipt <baseline.runner.json>] \\
    [--candidate-runner-receipt <candidate.runner.json>] \\
    [--rerun-runner-receipt <rerun.runner.json>] \\
    [--machine-class <expected-canonical-class>] \\
    --output <ratchet.json> \\
    --mode <promotion|regression-alarm> \\
    [--promote-dir <.bench-history> --date <YYYY-MM-DD>]

Exit status: 0=Allow, 1=Block, 2=Quarantine, 64=invalid invocation.";

type LoadedEvidence = (PerfEvidenceArtifact, Vec<u8>);
type AdmittedRunnerReceipt = (VerifiedRunnerIdentity, Vec<u8>);

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    baseline: PathBuf,
    baseline_evidence: Option<PathBuf>,
    candidate: PathBuf,
    candidate_evidence: Option<PathBuf>,
    rerun: Option<PathBuf>,
    rerun_evidence: Option<PathBuf>,
    baseline_runner_receipt: Option<PathBuf>,
    candidate_runner_receipt: Option<PathBuf>,
    rerun_runner_receipt: Option<PathBuf>,
    output: PathBuf,
    mode: PerfRatchetMode,
    promote_dir: Option<PathBuf>,
    machine_class: Option<String>,
    date: Option<String>,
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
    let manifest_bytes = fs::read(&args.manifest)?;
    let manifest_text = std::str::from_utf8(&manifest_bytes)?;
    let manifest_sha256 = perf_manifest_contract_sha256(manifest_text);
    let manifest = toml::from_str::<toml::Value>(manifest_text)?;

    let (baseline, baseline_bytes) = read_artifact(&args.baseline)?;
    let mut baseline_evidence = args
        .baseline_evidence
        .as_deref()
        .map(read_evidence_artifact)
        .transpose()?;
    let (candidate, candidate_bytes) = read_artifact(&args.candidate)?;
    let mut candidate_evidence = args
        .candidate_evidence
        .as_deref()
        .map(read_evidence_artifact)
        .transpose()?;
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
    let baseline_runner = read_runner_identity(
        registry.as_ref(),
        args.baseline_runner_receipt.as_deref(),
        candidate.gate,
        args.machine_class.as_deref(),
    )?;
    let candidate_runner = read_runner_identity(
        registry.as_ref(),
        args.candidate_runner_receipt.as_deref(),
        candidate.gate,
        args.machine_class.as_deref(),
    )?;
    let rerun_runner = read_runner_identity(
        registry.as_ref(),
        args.rerun_runner_receipt.as_deref(),
        candidate.gate,
        args.machine_class.as_deref(),
    )?;
    let baseline_evidence_source = baseline_evidence.as_ref().map(|(_, bytes)| bytes.clone());
    let candidate_evidence_source = candidate_evidence.as_ref().map(|(_, bytes)| bytes.clone());
    let rerun_evidence_source = rerun_evidence.as_ref().map(|(_, bytes)| bytes.clone());
    if args.mode == PerfRatchetMode::Promotion {
        bind_evidence_to_runner("baseline", &mut baseline_evidence, baseline_runner.as_ref())?;
        bind_evidence_to_runner(
            "candidate",
            &mut candidate_evidence,
            candidate_runner.as_ref(),
        )?;
        bind_evidence_to_runner("rerun", &mut rerun_evidence, rerun_runner.as_ref())?;
    }

    let mut evidence_files = vec![
        evidence("manifest", &args.manifest, &manifest_bytes),
        evidence("baseline", &args.baseline, &baseline_bytes),
        evidence("candidate", &args.candidate, &candidate_bytes),
    ];
    if let (Some(path), Some(source_bytes), Some((_, bound_bytes))) = (
        args.baseline_evidence.as_deref(),
        baseline_evidence_source.as_deref(),
        baseline_evidence.as_ref(),
    ) {
        evidence_files.push(evidence("baseline_current_source", path, source_bytes));
        evidence_files.push(bound_evidence("baseline_current_bound", path, bound_bytes));
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
            "baseline_runner_receipt",
            args.baseline_runner_receipt.as_deref(),
            baseline_runner.as_ref(),
        ),
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
        if let (Some(path), Some((_, bytes))) = (path, admitted) {
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
        baseline_runner_identity: baseline_runner.as_ref().map(|(identity, _)| identity),
        candidate_runner_identity: candidate_runner.as_ref().map(|(identity, _)| identity),
        rerun_runner_identity: rerun_runner.as_ref().map(|(identity, _)| identity),
        gate_activated: activated,
        mode: args.mode,
        expected_manifest_sha256: &manifest_sha256,
        evidence: evidence_files,
    });

    promote_history_if_allowed(
        &args,
        candidate.gate,
        &candidate_bytes,
        candidate_evidence
            .as_ref()
            .map(|(_, bytes)| bytes.as_slice()),
        candidate_runner
            .as_ref()
            .map(|(identity, _)| identity.class_id()),
        &mut evaluation,
    )?;

    let output = serde_json::to_string_pretty(&evaluation)?;
    write_file(&args.output, format!("{output}\n").as_bytes())?;
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
    let mut baseline_runner_receipt = None;
    let mut candidate_runner_receipt = None;
    let mut rerun_runner_receipt = None;
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
            "--baseline-runner-receipt" => {
                baseline_runner_receipt = Some(PathBuf::from(next_value(
                    &mut values,
                    "--baseline-runner-receipt",
                )?));
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
        baseline_runner_receipt.is_some(),
        candidate_runner_receipt.is_some(),
        rerun_runner_receipt.is_some(),
    ];
    if receipt_fields.iter().any(|present| *present)
        && !receipt_fields.iter().all(|present| *present)
    {
        return Err(
            "baseline, candidate, and rerun runner receipts must be supplied together".into(),
        );
    }
    if rerun_evidence.is_some() && rerun.is_none() {
        return Err("--rerun-evidence requires --rerun".into());
    }
    if mode == PerfRatchetMode::Promotion {
        let missing = [
            (baseline_evidence.is_none(), "--baseline-evidence"),
            (candidate_evidence.is_none(), "--candidate-evidence"),
            (rerun.is_none(), "--rerun"),
            (rerun_evidence.is_none(), "--rerun-evidence"),
            (
                baseline_runner_receipt.is_none(),
                "--baseline-runner-receipt",
            ),
            (
                candidate_runner_receipt.is_none(),
                "--candidate-runner-receipt",
            ),
            (rerun_runner_receipt.is_none(), "--rerun-runner-receipt"),
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
        baseline_runner_receipt,
        candidate_runner_receipt,
        rerun_runner_receipt,
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
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{field} {value:?} is not a safe filename component").into());
    }
    Ok(())
}

fn read_artifact(path: &Path) -> Result<(PerfGateArtifact, Vec<u8>), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let artifact = serde_json::from_slice::<PerfGateArtifact>(&bytes)?;
    Ok((artifact, bytes))
}

fn read_evidence_artifact(path: &Path) -> Result<LoadedEvidence, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let artifact = PerfEvidenceArtifact::from_verified_slice(&bytes)?;
    Ok((artifact, bytes))
}

fn read_runner_identity(
    registry: Option<&MachineClassRegistry>,
    path: Option<&Path>,
    gate: PerfGate,
    expected_class: Option<&str>,
) -> Result<Option<AdmittedRunnerReceipt>, Box<dyn Error>> {
    let (Some(registry), Some(path), Some(expected_class)) = (registry, path, expected_class)
    else {
        if registry.is_none() && path.is_none() && expected_class.is_none() {
            return Ok(None);
        }
        return Err("runner receipt admission inputs are incomplete".into());
    };
    let bytes = fs::read(path)?;
    let gate = gate.label();
    let context = MachineClassAdmissionContext {
        gate: gate.to_owned(),
        destination_basename: format!("{gate}.{expected_class}.latest.json"),
    };
    let identity = registry.admit(&bytes, &context)?;
    if identity.class_id() != expected_class {
        return Err(format!(
            "runner receipt derives machine class {:?}, expected {expected_class:?}",
            identity.class_id()
        )
        .into());
    }
    Ok(Some((identity, bytes)))
}

fn bind_evidence_to_runner(
    role: &str,
    evidence: &mut Option<LoadedEvidence>,
    runner: Option<&AdmittedRunnerReceipt>,
) -> Result<(), Box<dyn Error>> {
    let (Some((artifact, bytes)), Some((identity, _))) = (evidence.as_mut(), runner) else {
        return Err(format!("{role} evidence/runner finalization inputs are incomplete").into());
    };
    *bytes = artifact.bind_machine_class_identity_and_seal(identity.clone())?;
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

fn promote_history_if_requested(
    args: &Args,
    gate: PerfGate,
    candidate_bytes: &[u8],
    candidate_evidence_bytes: Option<&[u8]>,
    verified_machine_class: Option<&str>,
    updates: &mut Vec<PerfEvidenceFile>,
) -> Result<(), Box<dyn Error>> {
    let (Some(history_dir), Some(machine_class), Some(date)) = (
        args.promote_dir.as_deref(),
        verified_machine_class,
        args.date.as_deref(),
    ) else {
        return Ok(());
    };

    let stem = format!("{}.{}", gate.label(), machine_class);
    let latest = history_dir.join(format!("{stem}.latest.json"));
    let rolling = history_dir.join(format!("{stem}.{date}.json"));
    write_file(&rolling, candidate_bytes)?;
    let evidence_paths = candidate_evidence_bytes.map(|bytes| {
        (
            history_dir.join(format!("{stem}.latest.evidence.json")),
            history_dir.join(format!("{stem}.{date}.evidence.json")),
            bytes,
        )
    });
    if let Some((evidence_latest, evidence_rolling, bytes)) = evidence_paths.as_ref() {
        write_file(evidence_rolling, bytes)?;
        write_file(evidence_latest, bytes)?;
    }
    // Advance the legacy latest pointer last, after every immutable history
    // object and its current-schema evidence have reached stable storage.
    write_file(&latest, candidate_bytes)?;
    updates.push(evidence("history_window", &rolling, candidate_bytes));
    if let Some((evidence_latest, evidence_rolling, bytes)) = evidence_paths {
        updates.push(evidence(
            "history_evidence_window",
            &evidence_rolling,
            bytes,
        ));
        updates.push(evidence("history_evidence_latest", &evidence_latest, bytes));
    }
    updates.push(evidence("history_latest", &latest, candidate_bytes));
    Ok(())
}

fn promote_history_if_allowed(
    args: &Args,
    gate: PerfGate,
    candidate_bytes: &[u8],
    candidate_evidence_bytes: Option<&[u8]>,
    verified_machine_class: Option<&str>,
    evaluation: &mut frankensearch_quill_gauntlet::PerfRatchetEvaluation,
) -> Result<(), Box<dyn Error>> {
    if evaluation.decision != PerfGateDecision::Allow {
        return Ok(());
    }
    promote_history_if_requested(
        args,
        gate,
        candidate_bytes,
        candidate_evidence_bytes,
        verified_machine_class,
        &mut evaluation.history_updates,
    )
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
            baseline_runner_receipt: Some(PathBuf::from("baseline.runner.json")),
            candidate_runner_receipt: Some(PathBuf::from("candidate.runner.json")),
            rerun_runner_receipt: Some(PathBuf::from("rerun.runner.json")),
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

    fn history_files(history_dir: &Path) -> [PathBuf; 4] {
        [
            history_dir.join("QG-6.trj-zen3-16c-smt2.2026-07-29.json"),
            history_dir.join("QG-6.trj-zen3-16c-smt2.2026-07-29.evidence.json"),
            history_dir.join("QG-6.trj-zen3-16c-smt2.latest.evidence.json"),
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
                "--baseline-runner-receipt",
                "baseline.runner.json",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err());
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

        promote_history_if_allowed(
            &args,
            PerfGate::Qg6,
            b"forbidden-candidate",
            Some(b"forbidden-evidence"),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .expect("denial must be a no-op");

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

        promote_history_if_allowed(
            &args,
            PerfGate::Qg6,
            b"forbidden-candidate",
            Some(b"forbidden-receipt-bound-evidence"),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .expect("denial must not open history");

        assert!(!history_dir.exists());
        assert!(evaluation.history_updates.is_empty());
    }

    #[test]
    fn allowed_promotion_durably_writes_immutable_objects_then_latest_last() {
        let directory = tempfile::tempdir().expect("history directory");
        let args = test_args(directory.path());
        let mut evaluation = evaluation(PerfGateDecision::Allow);
        let candidate = b"candidate-threshold-artifact\n";
        let unverified_producer_evidence = b"candidate-unverified-producer-evidence\n";
        let receipt_bound_evidence = b"candidate-receipt-bound-evidence\n";

        promote_history_if_allowed(
            &args,
            PerfGate::Qg6,
            candidate,
            Some(receipt_bound_evidence),
            Some("trj-zen3-16c-smt2"),
            &mut evaluation,
        )
        .expect("valid durable promotion");

        let paths = history_files(directory.path());
        assert_eq!(fs::read(&paths[0]).expect("rolling threshold"), candidate);
        assert_eq!(
            fs::read(&paths[1]).expect("rolling evidence"),
            receipt_bound_evidence
        );
        assert_eq!(
            fs::read(&paths[2]).expect("latest evidence"),
            receipt_bound_evidence
        );
        assert_eq!(fs::read(&paths[3]).expect("latest threshold"), candidate);
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
                "history_evidence_latest",
                "history_latest",
            ]
        );
        assert!(
            snapshot(directory.path())
                .keys()
                .all(|name| !name.contains(".tmp-")),
            "durable promotion must leave no temporary siblings"
        );
    }
}
