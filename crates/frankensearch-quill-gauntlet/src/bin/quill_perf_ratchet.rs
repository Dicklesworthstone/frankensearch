#![forbid(unsafe_code)]
//! Evaluate Quill QG artifacts against a committed pass-over-pass baseline.

use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};

use frankensearch_quill_gauntlet::{
    ExecutionProfileId, HardwareClassId, MachineClassAdmissionContext, MachineClassRegistry,
    MachineProfileKey, PERF_ARTIFACT_SCHEMA_VERSION, PERF_EVIDENCE_SCHEMA_VERSION,
    PERF_HISTORY_POINTER_SCHEMA_VERSION, PerfEvidenceArtifact, PerfEvidenceFile, PerfGate,
    PerfGateArtifact, PerfGateDecision, PerfRatchetMode, PerfRatchetQg1AuthoritySets,
    PerfRatchetRequest, Qg1AuthorityRegisterEntryV1, Qg1ExpectedAuthority, Qg1StartupHandshakeV1,
    Qg1TargetPinV1, VerifiedRunnerIdentity, evaluate_perf_ratchet_against_qg1_authorities,
    is_explicit_bootstrap, is_explicit_bootstrap_for, perf_manifest_contract_sha256,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const USAGE: &str = "\
Usage:
  quill-perf-ratchet \\
    --manifest <docs/contracts/quill-perf-gates.toml> \\
    --baseline <authoritative-latest-pointer-or-bootstrap.json> \\
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
    [--hardware-class <expected-canonical-hardware>] \\
    [--execution-profile <expected-canonical-profile>] \\
    --output <ratchet.json> \\
    --mode <promotion|regression-alarm> \\
    [--promote-dir <.bench-history> --date <YYYY-MM-DD>]

Exact unmeasured bootstrap: omit baseline evidence.
Current measured pointer: omit baseline evidence; the pointer binds and resolves it.
Direct current-schema measured threshold: regression-alarm only; supply its bound evidence.
Exit status: 0=Allow, 1=Block, 2=Quarantine, 64=invalid invocation.";

type LoadedEvidence = (PerfEvidenceArtifact, Vec<u8>);
type AdmittedRunnerReceipt = (VerifiedRunnerIdentity, Vec<u8>, Vec<u8>, Vec<u8>);

fn current_bootstrap_basename(gate: PerfGate) -> String {
    let Some(version) = PERF_ARTIFACT_SCHEMA_VERSION.strip_prefix("quill-perf-artifact-") else {
        return format!("{}.invalid-schema.unmeasured.latest.json", gate.label());
    };
    format!("{}.{version}.unmeasured.latest.json", gate.label())
}

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
    machine_profile: Option<MachineProfileKey>,
    date: Option<String>,
    /// Per-arm QG-1 trust roots. Each pin arrives on its own argument and is
    /// never derived from the evidence it authenticates; an arm that supplies
    /// evidence without its pin admits no QG-1 authority at all.
    baseline_target_pin: Option<PathBuf>,
    baseline_authority_dir: Option<PathBuf>,
    candidate_target_pin: Option<PathBuf>,
    candidate_authority_dir: Option<PathBuf>,
    rerun_target_pin: Option<PathBuf>,
    rerun_authority_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryPointer {
    schema_version: String,
    gate: PerfGate,
    profile: MachineProfileKey,
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

/// Load one arm's complete QG-1 authority set from a pin plus its register
/// directory, binding both to the evidence that will be authenticated.
///
/// Authority is never inferred from evidence. The pin is the trust root: it
/// arrives on its own argument, names the complete required-target set before
/// timing, and every register entry must be one it already expected. The
/// evidence only supplies the run and source identity the pin must agree with,
/// which is why both are parameters here rather than being read back out of the
/// artifacts this set will later admit.
///
/// Descriptor-safe: the register directory is opened once with `O_NOFOLLOW |
/// O_DIRECTORY`, and every entry is opened relative to that descriptor with
/// `O_NOFOLLOW`, so a symlink planted between the scan and the read cannot
/// redirect a load, and the directory cannot be swapped underneath the set.
///
/// Missing, extra, duplicate, and wrong-run sets are all rejected:
/// * extra — an entry the pin does not name refuses inside
///   `to_expected_authority`, which is the pin check itself;
/// * missing — the admitted count is compared against the pin's complete
///   required-target set;
/// * duplicate — a repeated authority digest is refused before conversion;
/// * wrong run/source — the pin's campaign run and source revision must equal
///   the evidence identity supplied by the caller.
/// Returns the verified authorities AND a content-addressed evidence entry for
/// the pin and every register file, in canonical order (pin first, then
/// registers by sorted name).
///
/// The bytes are not discarded after verification on purpose. These are
/// external trust inputs the ratchet decision depends on, so a decision that
/// hashed the artifacts but not the authorities admitting them would archive an
/// incomplete account of what it trusted. Returning them lets the caller
/// attach them to the evaluation under arm-qualified roles.
fn load_qg1_authority_set(
    arm: &str,
    pin_path: &Path,
    authority_dir: &Path,
    evidence_run_id: &str,
    evidence_git_revision: &str,
) -> Result<(Vec<Qg1ExpectedAuthority>, Vec<PerfEvidenceFile>), Box<dyn Error>> {
    use rustix::fs::{Mode, OFlags};

    let pin_bytes = read_no_follow(pin_path, MAX_TARGET_PIN_BYTES)?;
    let mut evidence_files = vec![evidence(
        &format!("{arm}-qg1-target-pin"),
        pin_path,
        &pin_bytes,
    )];
    let pin: Qg1TargetPinV1 = serde_json::from_slice(&pin_bytes).map_err(|error| {
        format!(
            "QG-1 target pin {} does not parse: {error}",
            pin_path.display()
        )
    })?;
    pin.verify()?;

    // Bind the pin to the evidence identity BEFORE any entry is converted. A
    // pin cut for another run or another source tree can never authenticate
    // this arm, and discovering that after reconstituting authorities would
    // mean the refusal came too late to be meaningful.
    if pin.campaign_run_id() != evidence_run_id {
        return Err(format!(
            "QG-1 target pin names campaign run {} but the evidence was produced by run \
             {evidence_run_id}",
            pin.campaign_run_id()
        )
        .into());
    }
    if pin.source_git_revision() != evidence_git_revision {
        return Err(format!(
            "QG-1 target pin names source revision {} but the evidence was built from \
             {evidence_git_revision}",
            pin.source_git_revision()
        )
        .into());
    }

    let directory = rustix::fs::open(
        authority_dir,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| {
        format!(
            "QG-1 authority directory {} is not an openable directory: {error}",
            authority_dir.display()
        )
    })?;

    // Census through the SAME pinned descriptor the reads use. A path-based
    // `read_dir` here would enumerate one directory and open entries in
    // whatever directory the path resolved to a moment later; sharing the
    // descriptor is what makes "the set I counted" and "the set I read" the
    // same set.
    let mut names = Vec::new();
    let mut census = rustix::fs::Dir::read_from(&directory).map_err(|error| {
        format!(
            "QG-1 authority directory {} is not enumerable: {error}",
            authority_dir.display()
        )
    })?;
    while let Some(entry) = census.read() {
        let entry = entry.map_err(|error| {
            format!(
                "QG-1 authority directory {} enumeration failed: {error}",
                authority_dir.display()
            )
        })?;
        let name = entry.file_name().to_str().map_err(|_| {
            format!(
                "QG-1 authority directory {} holds a non-UTF-8 entry",
                authority_dir.display()
            )
        })?;
        if name == "." || name == ".." {
            continue;
        }
        // Every remaining entry must be a register file. Silently skipping
        // unexpected entries while claiming a COMPLETE pinned set is the same
        // class of lie as admitting an extra one: the set would be reported
        // complete without the operator ever learning what else was there.
        if !name.ends_with(".json") {
            return Err(format!(
                "QG-1 authority directory {} holds unexpected entry {name}; a pinned register \
                 directory may contain only its register files",
                authority_dir.display()
            )
            .into());
        }
        names.push(name.to_owned());
    }
    names.sort_unstable();

    let mut authorities = Vec::with_capacity(names.len());
    let mut seen_digests = BTreeSet::new();
    for name in &names {
        let file = rustix::fs::openat(
            &directory,
            name.as_str(),
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|error| {
            format!("QG-1 authority register entry {name} is not a regular readable file: {error}")
        })?;
        // Bounded: one byte past the register cap, so an oversized file is
        // REFUSED rather than truncated into something that might still parse.
        let bytes = read_bounded(
            std::fs::File::from(file),
            Qg1StartupHandshakeV1::MAX_REGISTER_BYTES,
            &format!("QG-1 authority register entry {name}"),
        )?;

        evidence_files.push(evidence(
            &format!("{arm}-qg1-authority-register:{name}"),
            &authority_dir.join(name),
            &bytes,
        ));

        let entry = Qg1AuthorityRegisterEntryV1::from_verified_slice(&bytes)?;
        if !seen_digests.insert(entry.digest().to_owned()) {
            return Err(format!(
                "QG-1 authority register directory {} presents authority {} twice",
                authority_dir.display(),
                entry.digest()
            )
            .into());
        }
        // The pin check lives here: an entry the pin does not name cannot
        // become an expectation, so an extra register file is refused rather
        // than silently widening the admitted set.
        authorities.push(entry.to_expected_authority(&pin)?);
    }

    let required = pin.required_targets().count();
    if authorities.len() != required {
        return Err(format!(
            "QG-1 authority set for run {} admitted {} of {required} pinned required targets",
            pin.campaign_run_id(),
            authorities.len()
        )
        .into());
    }
    Ok((authorities, evidence_files))
}

/// A QG-1 target pin is a small fixed record: run identity, source revision,
/// and one digest per required target. This cap is generous for the canonical
/// matrix and still refuses anything that is not a pin.
const MAX_TARGET_PIN_BYTES: usize = 256 * 1024;

/// Read one file without following a final symlink, bounded.
fn read_no_follow(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    use rustix::fs::{Mode, OFlags};

    let file = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|error| format!("{} is not a regular readable file: {error}", path.display()))?;
    read_bounded(
        std::fs::File::from(file),
        limit,
        &path.display().to_string(),
    )
}

/// Read at most `limit` bytes, refusing anything larger.
///
/// Reads `limit + 1` and refuses on overflow rather than truncating: a
/// truncated register or pin could still parse into a SHORTER valid-looking
/// set, which would silently narrow the admitted authorities.
fn read_bounded(
    mut file: std::fs::File,
    limit: usize,
    context: &str,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(
        &mut std::io::Read::take(&mut file, limit as u64 + 1),
        &mut bytes,
    )?;
    if bytes.len() > limit {
        return Err(format!("{context} exceeds its {limit}-byte bound").into());
    }
    Ok(bytes)
}

const REBASELINE_RETRY_PREDICATE: &str = "rerun the paired candidate and same-window reproduction with the current interleaved-runner schema, then promote the resulting current-schema history pointer";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct BaselineSchemaIncompatibility {
    code: &'static str,
    baseline_path: PathBuf,
    found_schema: String,
    expected_schema: &'static str,
    retry_predicate: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BaselineLoadError {
    incompatibility: BaselineSchemaIncompatibility,
}

impl BaselineLoadError {
    fn stale_threshold_schema(path: &Path, found_schema: &str) -> Self {
        Self {
            incompatibility: BaselineSchemaIncompatibility {
                code: "baseline_schema_incompatible",
                baseline_path: path.to_path_buf(),
                found_schema: found_schema.to_owned(),
                expected_schema: PERF_ARTIFACT_SCHEMA_VERSION,
                retry_predicate: REBASELINE_RETRY_PREDICATE,
            },
        }
    }
}

impl fmt::Display for BaselineLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let incompatibility = &self.incompatibility;
        write!(
            formatter,
            "{}: baseline={} found_schema={:?} expected_schema={:?} retry_predicate={:?}",
            incompatibility.code,
            incompatibility.baseline_path.display(),
            incompatibility.found_schema,
            incompatibility.expected_schema,
            incompatibility.retry_predicate,
        )
    }
}

impl Error for BaselineLoadError {}

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
    validate_manifest_gate_set(&manifest)?;
    validate_manifest_schema_bindings(&manifest)?;

    let _history_lock = acquire_promotion_history_lock(&args)?;
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
    validate_promotion_baseline_authority(
        &args,
        candidate.gate,
        baseline_is_bootstrap,
        baseline_pointer.is_some(),
    )?;
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
        args.machine_profile,
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
        args.machine_profile,
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

    // Each arm's authority set is resolved from its OWN pin/directory pair and
    // bound to that arm's evidence identity. The ratchet is handed the sets; it
    // never derives authority from the artifacts it is about to admit.
    let (baseline_authorities, baseline_authority_evidence) = resolve_arm_authorities(
        "baseline",
        args.baseline_target_pin.as_ref(),
        args.baseline_authority_dir.as_ref(),
        &baseline.run_id,
        &baseline.git_rev,
    )?;
    let (candidate_authorities, candidate_authority_evidence) = resolve_arm_authorities(
        "candidate",
        args.candidate_target_pin.as_ref(),
        args.candidate_authority_dir.as_ref(),
        &candidate.run_id,
        &candidate.git_rev,
    )?;
    let (rerun_authorities, rerun_authority_evidence) = match rerun.as_ref() {
        Some((artifact, _)) => resolve_arm_authorities(
            "rerun",
            args.rerun_target_pin.as_ref(),
            args.rerun_authority_dir.as_ref(),
            &artifact.run_id,
            &artifact.git_rev,
        )?,
        None => (Vec::new(), Vec::new()),
    };
    // The sets borrow these vectors, so they must outlive the call rather than
    // being built inline as temporaries.
    let baseline_authority_refs: Vec<&Qg1ExpectedAuthority> = baseline_authorities.iter().collect();
    let candidate_authority_refs: Vec<&Qg1ExpectedAuthority> =
        candidate_authorities.iter().collect();
    let rerun_authority_refs: Vec<&Qg1ExpectedAuthority> = rerun_authorities.iter().collect();
    let qg1_authorities = PerfRatchetQg1AuthoritySets {
        baseline: &baseline_authority_refs,
        candidate: &candidate_authority_refs,
        rerun: &rerun_authority_refs,
    };

    let mut evaluation = evaluate_perf_ratchet_against_qg1_authorities(
        PerfRatchetRequest {
            baseline: Some(&baseline),
            baseline_evidence: baseline_evidence.as_ref().map(|(artifact, _)| artifact),
            candidate: &candidate,
            rerun: rerun.as_ref().map(|(artifact, _)| artifact),
            candidate_evidence: candidate_evidence.as_ref().map(|(artifact, _)| artifact),
            rerun_evidence: rerun_evidence.as_ref().map(|(artifact, _)| artifact),
            expected_machine_profile: args.machine_profile,
            candidate_runner_identity: candidate_runner
                .as_ref()
                .map(|(identity, _, _, _)| identity),
            rerun_runner_identity: rerun_runner.as_ref().map(|(identity, _, _, _)| identity),
            gate_activated: activated,
            mode: args.mode,
            expected_manifest_sha256: &manifest_sha256,
            evidence: evidence_files,
        },
        qg1_authorities,
    );

    // Archive the trust inputs alongside the artifacts they admitted, in
    // canonical arm order. Without this the decision record would hash every
    // artifact it judged and none of the authorities that let it judge them.
    evaluation
        .evidence
        .extend(baseline_authority_evidence.into_iter());
    evaluation
        .evidence
        .extend(candidate_authority_evidence.into_iter());
    evaluation
        .evidence
        .extend(rerun_authority_evidence.into_iter());

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
            .map(|(identity, _, _, _)| identity.profile()),
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
    let mut hardware_class = None;
    let mut execution_profile = None;
    let mut date = None;
    let mut baseline_target_pin = None;
    let mut baseline_authority_dir = None;
    let mut candidate_target_pin = None;
    let mut candidate_authority_dir = None;
    let mut rerun_target_pin = None;
    let mut rerun_authority_dir = None;

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
            // Each arm's QG-1 trust root arrives on its own pair of arguments.
            // Pin and register directory are deliberately separate flags: a
            // single combined argument would let one path supply both the
            // expectation and the evidence it authenticates.
            "--baseline-target-pin" => {
                baseline_target_pin = Some(PathBuf::from(next_value(
                    &mut values,
                    "--baseline-target-pin",
                )?));
            }
            "--baseline-authority-dir" => {
                baseline_authority_dir = Some(PathBuf::from(next_value(
                    &mut values,
                    "--baseline-authority-dir",
                )?));
            }
            "--candidate-target-pin" => {
                candidate_target_pin = Some(PathBuf::from(next_value(
                    &mut values,
                    "--candidate-target-pin",
                )?));
            }
            "--candidate-authority-dir" => {
                candidate_authority_dir = Some(PathBuf::from(next_value(
                    &mut values,
                    "--candidate-authority-dir",
                )?));
            }
            "--rerun-target-pin" => {
                rerun_target_pin = Some(PathBuf::from(next_value(
                    &mut values,
                    "--rerun-target-pin",
                )?));
            }
            "--rerun-authority-dir" => {
                rerun_authority_dir = Some(PathBuf::from(next_value(
                    &mut values,
                    "--rerun-authority-dir",
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
            "--hardware-class" => {
                hardware_class = Some(parse_hardware_class(&next_value(
                    &mut values,
                    "--hardware-class",
                )?)?);
            }
            "--execution-profile" => {
                execution_profile = Some(parse_execution_profile(&next_value(
                    &mut values,
                    "--execution-profile",
                )?)?);
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
    let machine_profile = match (hardware_class, execution_profile) {
        (Some(hardware_class), Some(execution_profile)) => {
            Some(MachineProfileKey::new(hardware_class, execution_profile)?)
        }
        (None, None) => None,
        (Some(_), None) => {
            return Err("--hardware-class requires --execution-profile".into());
        }
        (None, Some(_)) => {
            return Err("--execution-profile requires --hardware-class".into());
        }
    };
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
            (hardware_class.is_none(), "--hardware-class"),
            (execution_profile.is_none(), "--execution-profile"),
            (promote_dir.is_none(), "--promote-dir"),
            (date.is_none(), "--date"),
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
        || machine_profile.is_some()
        || receipt_fields.iter().any(|present| *present)
    {
        return Err(
            "regression-alarm mode cannot receive promotion history or runner identity inputs"
                .into(),
        );
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
        machine_profile,
        date,
        baseline_target_pin,
        baseline_authority_dir,
        candidate_target_pin,
        candidate_authority_dir,
        rerun_target_pin,
        rerun_authority_dir,
    })
}

/// Resolve one arm's QG-1 authority set from its pin/directory pair.
///
/// Both flags are required together. A pin without its register directory has
/// nothing to admit, and a register directory without its pin has no trust
/// root — accepting either alone would let an arm present authorities that no
/// pre-timing pin ever named, which is the inference this whole path exists to
/// forbid. Supplying neither is honest no-claim: the arm contributes no QG-1
/// authority and downstream admission fails closed on its own terms.
fn resolve_arm_authorities(
    arm: &str,
    pin: Option<&PathBuf>,
    authority_dir: Option<&PathBuf>,
    evidence_run_id: &str,
    evidence_git_revision: &str,
) -> Result<(Vec<Qg1ExpectedAuthority>, Vec<PerfEvidenceFile>), Box<dyn Error>> {
    match (pin, authority_dir) {
        (Some(pin), Some(directory)) => {
            load_qg1_authority_set(arm, pin, directory, evidence_run_id, evidence_git_revision)
        }
        (None, None) => Ok((Vec::new(), Vec::new())),
        (Some(_), None) => Err(format!("--{arm}-target-pin requires --{arm}-authority-dir").into()),
        (None, Some(_)) => Err(format!("--{arm}-authority-dir requires --{arm}-target-pin").into()),
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

fn parse_hardware_class(value: &OsStr) -> Result<HardwareClassId, Box<dyn Error>> {
    match value.to_str() {
        Some("x86-vps-ovh") => Ok(HardwareClassId::X86VpsOvh),
        Some("trj-zen3-5995wx") => Ok(HardwareClassId::TrjZen35995wx),
        Some("m4-macos") => Ok(HardwareClassId::M4Macos),
        Some("m5-macos") => Ok(HardwareClassId::M5Macos),
        Some(other) => Err(format!("invalid --hardware-class {other:?}").into()),
        None => Err("--hardware-class must be valid UTF-8".into()),
    }
}

fn parse_execution_profile(value: &OsStr) -> Result<ExecutionProfileId, Box<dyn Error>> {
    match value.to_str() {
        Some("x86-diagnostic") => Ok(ExecutionProfileId::X86Diagnostic),
        Some("physical-64") => Ok(ExecutionProfileId::Physical64),
        Some("smt2-128") => Ok(ExecutionProfileId::Smt2_128),
        Some("scheduler-10") => Ok(ExecutionProfileId::Scheduler10),
        Some("scheduler-14") => Ok(ExecutionProfileId::Scheduler14),
        Some(other) => Err(format!("invalid --execution-profile {other:?}").into()),
        None => Err("--execution-profile must be valid UTF-8".into()),
    }
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

fn acquire_promotion_history_lock(args: &Args) -> Result<Option<File>, Box<dyn Error>> {
    if args.mode != PerfRatchetMode::Promotion {
        return Ok(None);
    }
    let history_dir = args
        .promote_dir
        .as_deref()
        .ok_or("promotion requires --promote-dir")?;
    let directory = File::open(history_dir).map_err(|error| {
        format!(
            "cannot open promotion history directory {}: {error}",
            history_dir.display()
        )
    })?;
    #[cfg(unix)]
    {
        use rustix::fs::{FlockOperation, flock};

        flock(&directory, FlockOperation::LockExclusive).map_err(|error| {
            format!(
                "cannot lock promotion history directory {}: {}",
                history_dir.display(),
                std::io::Error::from(error)
            )
        })?;
        Ok(Some(directory))
    }
    #[cfg(not(unix))]
    {
        drop(directory);
        Err("performance history promotion requires Unix advisory directory locking".into())
    }
}

fn validate_promotion_baseline_authority(
    args: &Args,
    gate: PerfGate,
    baseline_is_bootstrap: bool,
    baseline_is_history_pointer: bool,
) -> Result<(), Box<dyn Error>> {
    if args.mode != PerfRatchetMode::Promotion {
        return Ok(());
    }
    let history_dir = args
        .promote_dir
        .as_deref()
        .ok_or("promotion requires --promote-dir")?;
    let profile = args
        .machine_profile
        .ok_or("promotion requires a complete machine profile")?;
    let supplied_baseline = normalize_path(&args.baseline)?;
    let authoritative_latest = history_dir.join(profile.latest_basename(gate.label())?);
    let latest_exists = match fs::symlink_metadata(&authoritative_latest) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(format!(
                "cannot inspect authoritative latest pointer {}: {error}",
                authoritative_latest.display()
            )
            .into());
        }
    };

    if latest_exists {
        if supplied_baseline != normalize_path(&authoritative_latest)? {
            return Err(format!(
                "promotion baseline {} is stale or nonauthoritative; use current pointer {}",
                args.baseline.display(),
                authoritative_latest.display()
            )
            .into());
        }
        if !baseline_is_history_pointer || baseline_is_bootstrap {
            return Err(format!(
                "authoritative promotion baseline {} must be a measured v2 history pointer",
                authoritative_latest.display()
            )
            .into());
        }
        return Ok(());
    }

    let authoritative_bootstrap = history_dir.join(current_bootstrap_basename(gate));
    if supplied_baseline != normalize_path(&authoritative_bootstrap)?
        || !baseline_is_bootstrap
        || baseline_is_history_pointer
    {
        return Err(format!(
            "profile {}/{} has no measured latest pointer; first promotion must use exact bootstrap {}",
            profile.hardware_class_id().as_str(),
            profile.execution_profile_id().as_str(),
            authoritative_bootstrap.display()
        )
        .into());
    }
    Ok(())
}

fn read_baseline(
    path: &Path,
    explicit_evidence_path: Option<&Path>,
) -> Result<LoadedBaseline, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let probe = serde_json::from_slice::<serde_json::Value>(&bytes)?;
    let schema_version = probe
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("baseline {} has no string schema_version", path.display()))?;
    if schema_version == PERF_ARTIFACT_SCHEMA_VERSION {
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
    if schema_version.starts_with("quill-perf-artifact-v") {
        return Err(BaselineLoadError::stale_threshold_schema(path, schema_version).into());
    }
    if explicit_evidence_path.is_some() {
        return Err(
            "a history-pointer baseline resolves its own evidence; omit --baseline-evidence".into(),
        );
    }

    if schema_version != PERF_HISTORY_POINTER_SCHEMA_VERSION {
        return Err(format!(
            "baseline {} has unsupported schema {schema_version:?}; expected current threshold {PERF_ARTIFACT_SCHEMA_VERSION:?} or history pointer {PERF_HISTORY_POINTER_SCHEMA_VERSION:?}",
            path.display()
        )
        .into());
    }
    let pointer = serde_json::from_value::<HistoryPointer>(probe)?;
    if pointer.schema_version != PERF_HISTORY_POINTER_SCHEMA_VERSION
        || serde_json::to_vec_pretty(&pointer)? != bytes
    {
        return Err(format!(
            "history pointer {} has a stale schema or noncanonical bytes",
            path.display()
        )
        .into());
    }
    validate_component(&pointer.run_id, "history pointer run ID")?;
    validate_filename_component(
        &pointer.threshold_file,
        "history pointer threshold file",
        240,
    )?;
    validate_filename_component(&pointer.evidence_file, "history pointer evidence file", 240)?;
    let expected_pointer_name = pointer.profile.latest_basename(pointer.gate.label())?;
    if path.file_name().and_then(|name| name.to_str()) != Some(expected_pointer_name.as_str()) {
        return Err(format!(
            "history pointer {} does not use canonical gate/profile latest basename {}",
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
        || artifact
            .applicability_plan
            .as_ref()
            .is_none_or(|binding| binding.profile != pointer.profile)
        || evidence.0.applicability_plan.profile != pointer.profile
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
    let probe = serde_json::from_slice::<serde_json::Value>(&bytes)?;
    if let Some(schema_version) = probe
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        && schema_version.starts_with("quill-perf-artifact-v")
        && schema_version != PERF_ARTIFACT_SCHEMA_VERSION
    {
        return Err(BaselineLoadError::stale_threshold_schema(path, schema_version).into());
    }
    let artifact = serde_json::from_slice::<PerfGateArtifact>(&bytes)?;
    let canonical = serde_json::to_vec_pretty(&artifact)?;
    let mut canonical_bootstrap = canonical.clone();
    canonical_bootstrap.push(b'\n');
    if bytes != canonical && !(is_explicit_bootstrap(&artifact) && bytes == canonical_bootstrap) {
        return Err(format!(
            "threshold artifact {} is not exact canonical pretty JSON (current bootstrap sentinels use exactly one terminal newline)",
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
    expected_profile: Option<MachineProfileKey>,
    threshold_artifact_bytes: Option<&[u8]>,
    evidence_artifact_bytes: Option<&[u8]>,
) -> Result<Option<AdmittedRunnerReceipt>, Box<dyn Error>> {
    if registry.is_none()
        && receipt_path.is_none()
        && artifact_manifest_path.is_none()
        && run_log_path.is_none()
        && expected_profile.is_none()
    {
        return Ok(None);
    }
    let (
        Some(registry),
        Some(receipt_path),
        Some(artifact_manifest_path),
        Some(run_log_path),
        Some(expected_profile),
        Some(threshold_artifact_bytes),
        Some(evidence_artifact_bytes),
    ) = (
        registry,
        receipt_path,
        artifact_manifest_path,
        run_log_path,
        expected_profile,
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
        expected_profile,
        destination_basename: expected_profile.latest_basename(gate)?,
    };
    let identity = registry
        .admit(&receipt_bytes, &context)?
        .bind_artifact_manifest(
            &artifact_manifest_bytes,
            &run_log_bytes,
            threshold_artifact_bytes,
            evidence_artifact_bytes,
        )?;
    if identity.profile() != expected_profile {
        return Err(format!(
            "runner receipt derives machine profile {:?}, expected {expected_profile:?}",
            identity.profile()
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

/// Reject a manifest that omits, renames, or adds a normative performance gate.
///
/// The gate set is the contract from which all admission decisions derive. A
/// selected gate's `activated` flag alone is insufficient: a malformed
/// unrelated table would otherwise evade validation and make the committed
/// manifest an incomplete source of truth.
fn validate_manifest_gate_set(manifest: &toml::Value) -> Result<(), Box<dyn Error>> {
    let gates = manifest
        .get("gate")
        .and_then(toml::Value::as_table)
        .ok_or("manifest does not define a [gate] table")?;

    for gate in PerfGate::ALL {
        let label = gate.label();
        let policy = gates
            .get(label)
            .ok_or_else(|| format!("manifest is missing gate.{label}"))?;
        if !policy.is_table() {
            return Err(format!("manifest gate.{label} is not a table").into());
        }
        for field in ["name", "fixture", "target"] {
            if policy
                .get(field)
                .and_then(toml::Value::as_str)
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err(format!("manifest gate.{label}.{field} is missing or empty").into());
            }
        }
        if policy
            .get("activated")
            .and_then(toml::Value::as_bool)
            .is_none()
        {
            return Err(
                format!("manifest gate.{label}.activated is missing or not boolean").into(),
            );
        }
    }

    for label in gates.keys() {
        if !PerfGate::ALL
            .iter()
            .any(|gate| gate.label() == label.as_str())
        {
            return Err(format!("manifest defines unexpected gate.{label}").into());
        }
    }

    Ok(())
}

/// Bind the manifest declaration to the only artifact schemas this ratchet
/// accepts, so an internally inconsistent manifest fails before admission.
fn validate_manifest_schema_bindings(manifest: &toml::Value) -> Result<(), Box<dyn Error>> {
    let schemas = manifest
        .get("schemas")
        .and_then(toml::Value::as_table)
        .ok_or("manifest does not define a [schemas] table")?;
    const KNOWN_FIELDS: &[&str] = &[
        "threshold_artifact",
        "evidence_artifact",
        "evidence_assembly",
        "machine_registry",
        "applicability_plan",
        "runner_completion_receipt",
        "runner_artifact_manifest",
        "local_producer_contract",
        "history_pointer",
        "runner_attempt_receipt",
        "precommit_inventory",
    ];
    for (field, expected) in [
        ("threshold_artifact", PERF_ARTIFACT_SCHEMA_VERSION),
        ("evidence_artifact", PERF_EVIDENCE_SCHEMA_VERSION),
        ("history_pointer", PERF_HISTORY_POINTER_SCHEMA_VERSION),
    ] {
        let found = schemas
            .get(field)
            .and_then(toml::Value::as_str)
            .ok_or_else(|| format!("manifest schemas.{field} is missing or not a string"))?;
        if found != expected {
            return Err(
                format!("manifest schemas.{field} is {found:?}, expected {expected:?}").into(),
            );
        }
    }
    for field in schemas.keys() {
        if !KNOWN_FIELDS.contains(&field.as_str()) {
            return Err(format!("manifest schemas.{field} is unreviewed").into());
        }
    }
    Ok(())
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
    verified_machine_profile: Option<MachineProfileKey>,
    updates: &mut Vec<PerfEvidenceFile>,
) -> Result<Option<HistoryPublicationPlan>, Box<dyn Error>> {
    let (Some(history_dir), Some(profile), Some(date)) = (
        args.promote_dir.as_deref(),
        verified_machine_profile,
        args.date.as_deref(),
    ) else {
        return Ok(None);
    };
    let evidence_bytes =
        candidate_evidence_bytes.ok_or("allowed promotion is missing receipt-bound evidence")?;
    validate_component(candidate_run_id, "candidate run ID")?;

    let stem = format!(
        "{}.{}.{}",
        gate.label(),
        profile.hardware_class_id().as_str(),
        profile.execution_profile_id().as_str()
    );
    let rolling_stem = format!("{stem}.{date}.{candidate_run_id}");
    let threshold_file = format!("{rolling_stem}.json");
    let evidence_file = format!("{rolling_stem}.evidence.json");
    validate_filename_component(&threshold_file, "rolling threshold filename", 240)?;
    validate_filename_component(&evidence_file, "rolling evidence filename", 240)?;
    let rolling_threshold = history_dir.join(&threshold_file);
    let rolling_evidence = history_dir.join(&evidence_file);
    let latest_pointer = history_dir.join(format!("{stem}.latest.json"));
    let pointer = HistoryPointer {
        schema_version: PERF_HISTORY_POINTER_SCHEMA_VERSION.to_owned(),
        gate,
        profile,
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
    verified_machine_profile: Option<MachineProfileKey>,
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
        verified_machine_profile,
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

    fn test_profile() -> MachineProfileKey {
        MachineProfileKey::new(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128)
            .expect("registered Threadripper SMT profile")
    }

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
            machine_profile: Some(test_profile()),
            date: Some("2026-07-29".to_owned()),
            // Authority-free on purpose: these tests exercise history planning
            // and output validation, never QG-1 authority loading.
            baseline_target_pin: None,
            baseline_authority_dir: None,
            candidate_target_pin: None,
            candidate_authority_dir: None,
            rerun_target_pin: None,
            rerun_authority_dir: None,
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
            history_dir.join("QG-6.trj-zen3-5995wx.smt2-128.2026-07-29.candidate-1.json"),
            history_dir.join("QG-6.trj-zen3-5995wx.smt2-128.2026-07-29.candidate-1.evidence.json"),
            history_dir.join("QG-6.trj-zen3-5995wx.smt2-128.latest.json"),
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
        assert!(validate_component("../worker", "run ID").is_err());
        assert!(validate_component("candidate-1", "run ID").is_ok());
        assert!(validate_component("2026-07-23", "date").is_ok());
    }

    #[test]
    fn manifest_gate_set_is_exact_and_fails_closed_for_missing_or_extra_gates() {
        let manifest_text = include_str!("../../../../docs/contracts/quill-perf-gates.toml");
        let manifest = toml::from_str::<toml::Value>(manifest_text)
            .expect("parse normative performance manifest");
        validate_manifest_gate_set(&manifest)
            .expect("normative manifest has every QG gate exactly once");

        let missing = manifest_text.replacen("[gate.QG-10]", "[omitted.QG-10]", 1);
        let missing = toml::from_str::<toml::Value>(&missing).expect("parse missing-gate mutation");
        let missing_error = validate_manifest_gate_set(&missing)
            .expect_err("missing normative gate must fail closed")
            .to_string();
        assert!(
            missing_error.contains("missing gate.QG-10"),
            "unexpected missing-gate error: {missing_error}"
        );

        let extra = format!("{manifest_text}\n[gate.QG-11]\nactivated = false\n");
        let extra = toml::from_str::<toml::Value>(&extra).expect("parse extra-gate mutation");
        let extra_error = validate_manifest_gate_set(&extra)
            .expect_err("extra normative gate must fail closed")
            .to_string();
        assert!(
            extra_error.contains("unexpected gate.QG-11"),
            "unexpected extra-gate error: {extra_error}"
        );

        let mut missing_target = manifest.clone();
        missing_target
            .get_mut("gate")
            .and_then(toml::Value::as_table_mut)
            .and_then(|gates| gates.get_mut("QG-9"))
            .and_then(toml::Value::as_table_mut)
            .expect("QG-9 policy table")
            .remove("target");
        let missing_target_error = validate_manifest_gate_set(&missing_target)
            .expect_err("missing normative target must fail closed")
            .to_string();
        assert!(
            missing_target_error.contains("gate.QG-9.target is missing or empty"),
            "unexpected missing-target error: {missing_target_error}"
        );
    }

    #[test]
    fn manifest_schema_bindings_reject_stale_or_missing_artifact_versions() {
        let manifest_text = include_str!("../../../../docs/contracts/quill-perf-gates.toml");
        let manifest = toml::from_str::<toml::Value>(manifest_text)
            .expect("parse normative performance manifest");
        validate_manifest_schema_bindings(&manifest)
            .expect("normative manifest declares current ratchet artifact schemas");

        let mut stale_evidence = manifest.clone();
        stale_evidence
            .get_mut("schemas")
            .and_then(toml::Value::as_table_mut)
            .expect("schema table")
            .insert(
                "evidence_artifact".to_owned(),
                toml::Value::String("quill-perf-evidence-v4".to_owned()),
            );
        let stale_error = validate_manifest_schema_bindings(&stale_evidence)
            .expect_err("stale evidence schema must fail closed")
            .to_string();
        assert!(
            stale_error.contains("schemas.evidence_artifact")
                && stale_error.contains("quill-perf-evidence-v5"),
            "unexpected stale-schema error: {stale_error}"
        );

        let mut stale_history_pointer = manifest.clone();
        stale_history_pointer
            .get_mut("schemas")
            .and_then(toml::Value::as_table_mut)
            .expect("schema table")
            .insert(
                "history_pointer".to_owned(),
                toml::Value::String("frankensearch.perf-history-pointer.v1".to_owned()),
            );
        let stale_history_pointer_error = validate_manifest_schema_bindings(&stale_history_pointer)
            .expect_err("stale history-pointer schema must fail closed")
            .to_string();
        assert!(
            stale_history_pointer_error.contains("schemas.history_pointer")
                && stale_history_pointer_error.contains(PERF_HISTORY_POINTER_SCHEMA_VERSION),
            "unexpected stale history-pointer schema error: {stale_history_pointer_error}"
        );

        let mut unreviewed_schema = manifest.clone();
        unreviewed_schema
            .get_mut("schemas")
            .and_then(toml::Value::as_table_mut)
            .expect("schema table")
            .insert(
                "unreviewed_schema".to_owned(),
                toml::Value::String("unreviewed.v1".to_owned()),
            );
        let unreviewed_schema_error = validate_manifest_schema_bindings(&unreviewed_schema)
            .expect_err("unreviewed schema key must fail closed")
            .to_string();
        assert!(
            unreviewed_schema_error.contains("schemas.unreviewed_schema is unreviewed"),
            "unexpected unreviewed-schema error: {unreviewed_schema_error}"
        );

        let mut missing_threshold = manifest;
        missing_threshold
            .get_mut("schemas")
            .and_then(toml::Value::as_table_mut)
            .expect("schema table")
            .remove("threshold_artifact");
        let missing_error = validate_manifest_schema_bindings(&missing_threshold)
            .expect_err("missing threshold schema must fail closed")
            .to_string();
        assert!(
            missing_error.contains("schemas.threshold_artifact is missing"),
            "unexpected missing-schema error: {missing_error}"
        );
    }

    #[test]
    fn cli_profile_parsers_are_closed_and_reject_obsolete_width_classes() {
        assert_eq!(
            parse_hardware_class(OsStr::new("trj-zen3-5995wx")).unwrap(),
            HardwareClassId::TrjZen35995wx
        );
        assert_eq!(
            parse_execution_profile(OsStr::new("smt2-128")).unwrap(),
            ExecutionProfileId::Smt2_128
        );
        for obsolete in ["trj-zen3-1c", "trj-zen3-64c", "trj-zen3-64c-smt2"] {
            assert!(
                parse_hardware_class(OsStr::new(obsolete)).is_err(),
                "{obsolete} admitted"
            );
        }
        assert!(
            MachineProfileKey::new(HardwareClassId::M4Macos, ExecutionProfileId::Smt2_128).is_err()
        );
    }

    #[test]
    fn legacy_machine_class_flag_is_rejected() {
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
                "regression-alarm",
                "--machine-class",
                "trj-zen3-64c-smt2",
            ]
            .into_iter()
            .map(OsString::from),
        );
        assert!(result.is_err());
    }

    #[test]
    fn direct_historical_thresholds_return_typed_rebaseline_incompatibility() {
        let directory = tempfile::tempdir().expect("baseline directory");
        for schema in [
            "quill-perf-artifact-v4",
            "quill-perf-artifact-v5",
            "quill-perf-artifact-v6",
        ] {
            let path = directory.path().join(format!("{schema}.json"));
            fs::write(
                &path,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "schema_version": schema,
                }))
                .expect("historical threshold probe"),
            )
            .expect("write historical threshold probe");
            let error = read_baseline(&path, None)
                .expect_err("historical direct threshold must fail closed")
                .downcast::<BaselineLoadError>()
                .expect("historical threshold denial must be typed");
            assert!(
                error.incompatibility.code == "baseline_schema_incompatible"
                    && error.incompatibility.baseline_path == path
                    && error.incompatibility.found_schema == schema
                    && error.incompatibility.expected_schema == PERF_ARTIFACT_SCHEMA_VERSION
                    && error.incompatibility.retry_predicate == REBASELINE_RETRY_PREDICATE,
                "unexpected typed stale-schema diagnostic: {error}"
            );
        }
    }

    #[test]
    fn historical_threshold_through_current_history_pointer_is_typed() {
        let directory = tempfile::tempdir().expect("baseline directory");
        let profile = test_profile();
        let threshold_file = "historical-threshold.json";
        let threshold_path = directory.path().join(threshold_file);
        fs::write(
            &threshold_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": "quill-perf-artifact-v6",
            }))
            .expect("historical threshold probe"),
        )
        .expect("write historical threshold probe");

        let pointer_path = directory.path().join(
            profile
                .latest_basename(PerfGate::Qg2.label())
                .expect("latest basename"),
        );
        let pointer = HistoryPointer {
            schema_version: PERF_HISTORY_POINTER_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg2,
            profile,
            run_id: "typed-history-pointer".to_owned(),
            threshold_file: threshold_file.to_owned(),
            threshold_sha256: "0".repeat(64),
            evidence_file: "unreached-evidence.json".to_owned(),
            evidence_sha256: "0".repeat(64),
        };
        fs::write(
            &pointer_path,
            serde_json::to_vec_pretty(&pointer).expect("canonical history pointer"),
        )
        .expect("write history pointer");

        let error = read_baseline(&pointer_path, None)
            .expect_err("a pointer to historical schema must fail closed")
            .downcast::<BaselineLoadError>()
            .expect("pointer-resolved historical threshold denial must be typed");
        assert_eq!(error.incompatibility.code, "baseline_schema_incompatible");
        assert_eq!(error.incompatibility.baseline_path, threshold_path);
        assert_eq!(error.incompatibility.found_schema, "quill-perf-artifact-v6");
        assert_eq!(
            error.incompatibility.retry_predicate,
            REBASELINE_RETRY_PREDICATE
        );
    }

    #[test]
    fn current_versioned_bootstrap_loads_without_rebaseline_incompatibility() {
        let directory = tempfile::tempdir().expect("baseline directory");
        let path = directory
            .path()
            .join(current_bootstrap_basename(PerfGate::Qg2));
        fs::write(
            &path,
            include_bytes!("../../../../.bench-history/QG-2.v7.unmeasured.latest.json"),
        )
        .expect("write current bootstrap");

        let loaded = read_baseline(&path, None)
            .expect("current versioned bootstrap must load without a rebaseline incompatibility");
        assert_eq!(loaded.artifact.schema_version, PERF_ARTIFACT_SCHEMA_VERSION);
        assert!(loaded.evidence.is_none());
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
                "QG-2.v7.unmeasured.latest.json",
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
                "--hardware-class",
                "trj-zen3-5995wx",
                "--execution-profile",
                "smt2-128",
                "--promote-dir",
                ".bench-history",
                "--date",
                "2026-07-30",
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
        assert_eq!(result.machine_profile, Some(test_profile()));
    }

    #[cfg(unix)]
    #[test]
    fn promotion_history_lock_serializes_the_entire_history_directory() {
        use rustix::fs::{FlockOperation, flock};

        let directory = tempfile::tempdir().expect("history directory");
        let args = test_args(directory.path());
        let first = acquire_promotion_history_lock(&args)
            .expect("first lock")
            .expect("promotion lock");
        let second = File::open(directory.path()).expect("second directory descriptor");

        assert!(
            flock(&second, FlockOperation::NonBlockingLockExclusive).is_err(),
            "a second promoter must not resolve a baseline concurrently"
        );
        drop(first);
        flock(&second, FlockOperation::NonBlockingLockExclusive)
            .expect("lock becomes available only after the first promoter exits");
    }

    #[test]
    fn promotion_baseline_authority_rejects_copies_stale_bootstrap_and_direct_latest() {
        let bootstrap_directory = tempfile::tempdir().expect("bootstrap history directory");
        let canonical_bootstrap = bootstrap_directory
            .path()
            .join(current_bootstrap_basename(PerfGate::Qg2));
        let bootstrap_bytes =
            include_bytes!("../../../../.bench-history/QG-2.v7.unmeasured.latest.json");
        fs::write(&canonical_bootstrap, bootstrap_bytes).expect("canonical bootstrap");
        let mut bootstrap_args = test_args(bootstrap_directory.path());
        bootstrap_args.baseline = canonical_bootstrap.clone();
        let loaded = read_baseline(&canonical_bootstrap, None)
            .expect("actual CLI baseline loader admits the current versioned bootstrap");
        assert!(is_explicit_bootstrap_for(
            &loaded.artifact,
            PerfGate::Qg2,
            &perf_manifest_contract_sha256(include_str!(
                "../../../../docs/contracts/quill-perf-gates.toml"
            ))
        ));
        assert_eq!(loaded.artifact_bytes, bootstrap_bytes);
        assert!(
            validate_promotion_baseline_authority(&bootstrap_args, PerfGate::Qg2, true, false)
                .is_ok()
        );

        let copied_bootstrap = bootstrap_directory.path().join("copied-bootstrap.json");
        fs::write(&copied_bootstrap, bootstrap_bytes).expect("copied bootstrap");
        bootstrap_args.baseline = copied_bootstrap;
        let before_copy_rejection = snapshot(bootstrap_directory.path());
        assert!(
            validate_promotion_baseline_authority(&bootstrap_args, PerfGate::Qg2, true, false)
                .is_err()
        );
        assert_eq!(
            snapshot(bootstrap_directory.path()),
            before_copy_rejection,
            "rejecting a copied bootstrap must not mutate history"
        );

        let measured_directory = tempfile::tempdir().expect("measured history directory");
        let profile = test_profile();
        let latest = measured_directory
            .path()
            .join(profile.latest_basename(PerfGate::Qg2.label()).unwrap());
        fs::write(&latest, b"new-authoritative-pointer").expect("latest pointer");
        let stale_pointer_copy = measured_directory.path().join("stale-pointer-copy.json");
        fs::write(&stale_pointer_copy, b"old-authoritative-pointer").expect("stale pointer copy");
        let measured_bootstrap = measured_directory
            .path()
            .join(current_bootstrap_basename(PerfGate::Qg2));
        fs::write(&measured_bootstrap, bootstrap_bytes).expect("retained bootstrap");
        let mut measured_args = test_args(measured_directory.path());

        measured_args.baseline = stale_pointer_copy;
        let before_stale_rejection = snapshot(measured_directory.path());
        assert!(
            validate_promotion_baseline_authority(&measured_args, PerfGate::Qg2, false, true)
                .is_err()
        );
        assert_eq!(
            snapshot(measured_directory.path()),
            before_stale_rejection,
            "rejecting a stale copied pointer must leave current history byte-identical"
        );

        measured_args.baseline = measured_bootstrap;
        let before_bootstrap_rejection = snapshot(measured_directory.path());
        assert!(
            validate_promotion_baseline_authority(&measured_args, PerfGate::Qg2, true, false)
                .is_err()
        );
        assert_eq!(
            snapshot(measured_directory.path()),
            before_bootstrap_rejection,
            "bootstrap replay after first activation must not mutate history"
        );

        measured_args.baseline = latest;
        assert!(
            validate_promotion_baseline_authority(&measured_args, PerfGate::Qg2, false, true)
                .is_ok()
        );
        assert!(
            validate_promotion_baseline_authority(&measured_args, PerfGate::Qg2, false, false)
                .is_err(),
            "a direct threshold cannot masquerade at the authoritative pointer path"
        );
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
    fn cli_rejects_preserved_v6_bootstrap_as_stale_after_v7_schema_bump() {
        let baseline = serde_json::from_str::<PerfGateArtifact>(include_str!(
            "../../../../.bench-history/QG-2.unmeasured.latest.json"
        ))
        .expect("committed QG-2 bootstrap artifact");
        let manifest = include_str!("../../../../docs/contracts/quill-perf-gates.toml");
        let manifest_sha256 = perf_manifest_contract_sha256(manifest);

        assert!(!is_explicit_bootstrap_for(
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
            Some(test_profile()),
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
            Some(test_profile()),
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
            Some(test_profile()),
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
        assert_eq!(pointer.profile, test_profile());
        assert_eq!(
            pointer.threshold_file,
            "QG-6.trj-zen3-5995wx.smt2-128.2026-07-29.candidate-1.json"
        );
        assert_eq!(
            pointer.evidence_file,
            "QG-6.trj-zen3-5995wx.smt2-128.2026-07-29.candidate-1.evidence.json"
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
            Some(test_profile()),
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
            Some(test_profile()),
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
            Some(test_profile()),
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
                    Some(test_profile()),
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

    /// A pin without its register directory has nothing to admit, and a
    /// register directory without its pin has no trust root. Accepting either
    /// alone would let an arm present authorities no pre-timing pin named,
    /// which is exactly the inference this path forbids.
    #[test]
    fn a_half_supplied_authority_pair_is_refused_per_arm() {
        for arm in ["baseline", "candidate", "rerun"] {
            let pin_only = resolve_arm_authorities(
                arm,
                Some(&PathBuf::from("pin.json")),
                None,
                "run-1",
                "a".repeat(40).as_str(),
            )
            .expect_err("a pin without its authority directory must refuse");
            assert!(
                pin_only
                    .to_string()
                    .contains(&format!("--{arm}-authority-dir")),
                "the refusal must name the missing flag, got: {pin_only}"
            );

            let dir_only = resolve_arm_authorities(
                arm,
                None,
                Some(&PathBuf::from("authorities")),
                "run-1",
                "a".repeat(40).as_str(),
            )
            .expect_err("an authority directory without its pin must refuse");
            assert!(
                dir_only
                    .to_string()
                    .contains(&format!("--{arm}-target-pin")),
                "the refusal must name the missing flag, got: {dir_only}"
            );
        }
    }

    /// Supplying neither flag is honest no-claim, not an error: the arm
    /// contributes no QG-1 authority and no evidence rows, and downstream
    /// admission fails closed on its own terms rather than here.
    #[test]
    fn an_arm_with_no_authority_inputs_contributes_nothing() {
        let (authorities, evidence_files) =
            resolve_arm_authorities("candidate", None, None, "run-1", "a".repeat(40).as_str())
                .expect("absent authority inputs are honest no-claim");
        assert!(authorities.is_empty());
        assert!(
            evidence_files.is_empty(),
            "an arm that admitted nothing must not archive phantom evidence rows"
        );
    }

    /// The loader must refuse a directory it cannot pin open, rather than
    /// treating an unreadable or absent register directory as an empty one.
    /// An empty admitted set would otherwise read as "no authorities needed".
    #[test]
    fn an_unopenable_authority_directory_is_refused_not_treated_as_empty() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let missing = directory.path().join("absent-authorities");
        let pin = directory.path().join("pin.json");
        std::fs::write(&pin, b"{}").expect("write a placeholder pin");
        assert!(
            load_qg1_authority_set("candidate", &pin, &missing, "run-1", &"a".repeat(40)).is_err(),
            "an absent authority directory must refuse, never admit an empty set"
        );
    }
}
