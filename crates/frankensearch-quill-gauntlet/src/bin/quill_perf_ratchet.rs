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
    PerfEvidenceArtifact, PerfEvidenceFile, PerfGate, PerfGateArtifact, PerfGateDecision,
    PerfRatchetMode, PerfRatchetRequest, evaluate_perf_ratchet, perf_manifest_contract_sha256,
};
use sha2::{Digest, Sha256};

const USAGE: &str = "\
Usage:
  quill-perf-ratchet \\
    --manifest <docs/contracts/quill-perf-gates.toml> \\
    --baseline <.bench-history/QG-N.machine.latest.json> \\
    --candidate <QG-N.json> \\
    [--candidate-evidence <QG-N.evidence.json>] \\
    [--rerun <QG-N.json>] \\
    [--rerun-evidence <QG-N.rerun.evidence.json>] \\
    --output <ratchet.json> \\
    --mode <promotion|regression-alarm> \\
    [--promote-dir <.bench-history> --machine-class <label> --date <YYYY-MM-DD>]

Exit status: 0=Allow, 1=Block, 2=Quarantine, 64=invalid invocation.";

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    baseline: PathBuf,
    candidate: PathBuf,
    candidate_evidence: Option<PathBuf>,
    rerun: Option<PathBuf>,
    rerun_evidence: Option<PathBuf>,
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
    let (candidate, candidate_bytes) = read_artifact(&args.candidate)?;
    let candidate_evidence = args
        .candidate_evidence
        .as_deref()
        .map(read_evidence_artifact)
        .transpose()?;
    let rerun = args.rerun.as_deref().map(read_artifact).transpose()?;
    let rerun_evidence = args
        .rerun_evidence
        .as_deref()
        .map(read_evidence_artifact)
        .transpose()?;
    let activated = gate_activated(&manifest, candidate.gate)?;

    let mut evidence_files = vec![
        evidence("manifest", &args.manifest, &manifest_bytes),
        evidence("baseline", &args.baseline, &baseline_bytes),
        evidence("candidate", &args.candidate, &candidate_bytes),
    ];
    if let (Some(rerun_path), Some((_, rerun_bytes))) = (args.rerun.as_deref(), rerun.as_ref()) {
        evidence_files.push(evidence("rerun", rerun_path, rerun_bytes));
    }
    if let (Some(path), Some((_, bytes))) = (
        args.candidate_evidence.as_deref(),
        candidate_evidence.as_ref(),
    ) {
        evidence_files.push(evidence("candidate_current", path, bytes));
    }
    if let (Some(path), Some((_, bytes))) =
        (args.rerun_evidence.as_deref(), rerun_evidence.as_ref())
    {
        evidence_files.push(evidence("rerun_current", path, bytes));
    }

    let mut evaluation = evaluate_perf_ratchet(PerfRatchetRequest {
        baseline: Some(&baseline),
        candidate: &candidate,
        rerun: rerun.as_ref().map(|(artifact, _)| artifact),
        candidate_evidence: candidate_evidence.as_ref().map(|(artifact, _)| artifact),
        rerun_evidence: rerun_evidence.as_ref().map(|(artifact, _)| artifact),
        require_current_evidence: args.mode == PerfRatchetMode::Promotion,
        gate_activated: activated,
        mode: args.mode,
        expected_manifest_sha256: &manifest_sha256,
        evidence: evidence_files,
    });

    if evaluation.decision == PerfGateDecision::Allow {
        promote_history_if_requested(
            &args,
            candidate.gate,
            &candidate_bytes,
            candidate_evidence
                .as_ref()
                .map(|(_, bytes)| bytes.as_slice()),
            &mut evaluation.history_updates,
        )?;
    }

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
    let mut candidate = None;
    let mut candidate_evidence = None;
    let mut rerun = None;
    let mut rerun_evidence = None;
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

    let promotion_fields = [
        promote_dir.is_some(),
        machine_class.is_some(),
        date.is_some(),
    ];
    if promotion_fields.iter().any(|present| *present)
        && !promotion_fields.iter().all(|present| *present)
    {
        return Err("--promote-dir, --machine-class, and --date must be supplied together".into());
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
        candidate: candidate.ok_or("missing --candidate")?,
        candidate_evidence,
        rerun,
        rerun_evidence,
        output: output.ok_or("missing --output")?,
        mode: mode.ok_or("missing --mode")?,
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

fn read_evidence_artifact(path: &Path) -> Result<(PerfEvidenceArtifact, Vec<u8>), Box<dyn Error>> {
    let artifact = PerfEvidenceArtifact::load_verified(path)?;
    let bytes = fs::read(path)?;
    Ok((artifact, bytes))
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
    updates: &mut Vec<PerfEvidenceFile>,
) -> Result<(), Box<dyn Error>> {
    let (Some(history_dir), Some(machine_class), Some(date)) = (
        args.promote_dir.as_deref(),
        args.machine_class.as_deref(),
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
}
