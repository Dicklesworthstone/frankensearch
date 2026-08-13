//! Deterministic, receipt-preserving assembly of profile-qualified QG-1 shards.
//!
//! A completed benchmark shard is a sealed [`PerfEvidenceArtifact`] with its
//! own runner receipt, artifact manifest, and raw samples. Assembly keeps every
//! source artifact intact. It never rewrites provenance into one fictitious
//! aggregate invocation and never creates an aggregate runner receipt. The
//! typed H2 process/lifecycle receipt is a hard dependency before this module
//! can admit real completed or failed producer shards as claim evidence. The
//! offline receipt records the canonical cell-to-source mapping, exact set
//! differences against the independently reconstructed applicability plan,
//! and durable failed-attempt diagnostics.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::machine_class_registry::sha256_hex;
use crate::{
    BuildIdentity, CorpusIdentity, DistributionSummary, EvidenceArtifactError, EvidenceCell,
    EvidenceCellBody, EvidenceDecisionStatus, EvidencePolicy, EvidenceProvenance, EvidenceReason,
    EvidenceRole, EvidenceSeverity, ExecutionCapacitySemantics, LocalPerfAttemptOutcome,
    LocalPerfAttemptReceipt, LocalPerfRunError, MachineClassCanonicalizationBinding,
    MachineClassError, MachineClassRegistry, MachineProfileKey, PERF_ARTIFACT_SCHEMA_VERSION,
    PairedEstimatorConfig, PerfApplicabilityPlan, PerfApplicabilityPlanBinding,
    PerfApplicabilityPlanError, PerfCellApplicability, PerfCellApplicabilityReason, PerfCellResult,
    PerfCellSpec, PerfEvidenceArtifact, PerfExecutionProvenance, PerfGate, PerfGateArtifact,
    PerfMatrixSpec, PerfMetricSemantics, PerfProducerOs, PerfRawSample, PerfSampleArm,
    Qg1ExpectedAuthority, VerifiedRunnerIdentity,
};

#[cfg(test)]
use crate::{MachineIdentity, PeakRssEvidence};

/// Wire version of the strict offline assembly artifact.
pub const PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION: &str = "quill-perf-evidence-assembly-v2";
/// Wire version of the independently sealed derived QG-1 matrix manifest.
pub const PERF_EVIDENCE_ASSEMBLY_MATRIX_SCHEMA_VERSION: &str =
    "quill-perf-evidence-assembly-matrix-v1";
/// Wire version of the run-label-independent semantic cell-set seal.
pub const PERF_EVIDENCE_SEMANTIC_CELL_SET_SCHEMA_VERSION: &str =
    "quill-perf-evidence-semantic-cell-set-v1";
/// The sole source-level `NoClaim` code assembly may discharge after exact
/// disjoint coverage proves that a producer intentionally emitted one shard.
pub const PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE: &str = "qg1.partial_shard";
/// Exact producer diagnostic assembly may discharge only after the H2 receipt
/// proves the corresponding typed fixture selection is a proper subset.
pub const PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_DETAIL: &str = "the invocation retained one immutable partial QG-1 shard; this source artifact cannot \
     support a publication or ratchet claim until exact disjoint assembly proves full coverage";
/// Maximum UTF-8 byte length of a concrete retry predicate.
pub const PERF_ASSEMBLY_MAX_RETRY_PREDICATE_BYTES: usize = 240;
/// Maximum number of completed and failed shards admitted to one assembly.
pub const PERF_ASSEMBLY_MAX_SHARDS: usize = 512;
/// Maximum exact process receipt, runner receipt, or manifest input.
pub const PERF_ASSEMBLY_MAX_RECEIPT_BYTES: usize = 8 * 1024 * 1024;
/// Maximum retained raw log, threshold artifact, or evidence artifact input.
pub const PERF_ASSEMBLY_MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Stable source reason for missing engine-internal actual-work/lifecycle proof.
pub const PERF_ASSEMBLY_ENGINE_LIFECYCLE_NO_CLAIM_CODE: &str = "qg1.engine_lifecycle_unavailable";
/// Stable source reason for missing descendant/process-tree quiescence proof.
pub const PERF_ASSEMBLY_PROCESS_TREE_NO_CLAIM_CODE: &str =
    "qg1.process_tree_quiescence_unavailable";

const ASSEMBLY_HASH_DOMAIN: &[u8] = b"frankensearch.quill.perf-evidence-assembly.v2\0";
const MATRIX_MANIFEST_HASH_DOMAIN: &[u8] =
    b"frankensearch.quill.perf-evidence-assembly-matrix.v1\0";
const SEMANTIC_CELL_SET_HASH_DOMAIN: &[u8] =
    b"frankensearch.quill.perf-evidence-semantic-cell-set.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AssemblyFileIdentity {
    device: u64,
    inode: u64,
}

fn assembly_file_identity(metadata: &fs::Metadata) -> AssemblyFileIdentity {
    AssemblyFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

fn unsafe_assembly_path(reason: impl Into<String>) -> PerfEvidenceAssemblyError {
    PerfEvidenceAssemblyError::InconsistentAssembly {
        reason: reason.into(),
    }
}

fn verify_pinned_assembly_directory(
    path: &Path,
    directory: &File,
    require_private_owner: bool,
) -> Result<AssemblyFileIdentity, PerfEvidenceAssemblyError> {
    use rustix::process::geteuid;

    if !path.is_absolute() {
        return Err(unsafe_assembly_path(
            "assembly output directory must be an absolute path",
        ));
    }
    let held = directory.metadata()?;
    let path_metadata = fs::symlink_metadata(path)?;
    let identity = assembly_file_identity(&held);
    if !held.is_dir()
        || path_metadata.file_type().is_symlink()
        || !path_metadata.is_dir()
        || assembly_file_identity(&path_metadata) != identity
        || fs::canonicalize(path)? != path
    {
        return Err(unsafe_assembly_path(
            "assembly output directory path changed identity or traverses a symlink",
        ));
    }
    if require_private_owner && (held.uid() != geteuid().as_raw() || held.mode() & 0o022 != 0) {
        return Err(unsafe_assembly_path(
            "assembly output directory must be effective-user-owned and not group/world writable",
        ));
    }
    Ok(identity)
}

fn open_pinned_assembly_directory(
    path: &Path,
    require_private_owner: bool,
) -> Result<File, PerfEvidenceAssemblyError> {
    use rustix::fs::{Mode, OFlags, open};

    let descriptor = open(
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::DIRECTORY | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(std::io::Error::from)?;
    let directory = File::from(descriptor);
    verify_pinned_assembly_directory(path, &directory, require_private_owner)?;
    Ok(directory)
}

fn prepare_assembly_output_directory(path: &Path) -> Result<File, PerfEvidenceAssemblyError> {
    use rustix::fs::{Mode, mkdirat};
    use rustix::io::Errno;

    if !path.is_absolute() {
        return Err(unsafe_assembly_path(
            "assembly output directory must be an absolute path",
        ));
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(unsafe_assembly_path(
                "assembly output path exists but is not a real directory",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .ok_or_else(|| unsafe_assembly_path("assembly output directory has no parent"))?;
            let leaf = path.file_name().ok_or_else(|| {
                unsafe_assembly_path("assembly output directory has no final component")
            })?;
            let mut components = Path::new(leaf).components();
            if !matches!(components.next(), Some(std::path::Component::Normal(_)))
                || components.next().is_some()
            {
                return Err(unsafe_assembly_path(
                    "assembly output directory final component is unsafe",
                ));
            }
            let parent_directory = open_pinned_assembly_directory(parent, false)?;
            match mkdirat(&parent_directory, leaf, Mode::RWXU) {
                Ok(()) | Err(Errno::EXIST) => {}
                Err(error) => return Err(std::io::Error::from(error).into()),
            }
            parent_directory.sync_all()?;
            verify_pinned_assembly_directory(parent, &parent_directory, false)?;
        }
        Err(error) => return Err(error.into()),
    }
    open_pinned_assembly_directory(path, true)
}

fn checked_owned_regular_identity(
    file: &File,
    maximum_len: usize,
    exact_len: bool,
) -> Result<AssemblyFileIdentity, PerfEvidenceAssemblyError> {
    use rustix::process::geteuid;

    let metadata = file.metadata()?;
    let maximum_len = u64::try_from(maximum_len).unwrap_or(u64::MAX);
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != geteuid().as_raw()
        || metadata.len() > maximum_len
        || (exact_len && metadata.len() != maximum_len)
    {
        return Err(unsafe_assembly_path(
            "assembly file must be an effective-user-owned regular single-link inode with the expected bounded length",
        ));
    }
    Ok(assembly_file_identity(&metadata))
}

fn read_owned_regular_at(
    directory: &File,
    name: &OsStr,
    exact_len: usize,
) -> Result<Option<(Vec<u8>, AssemblyFileIdentity)>, PerfEvidenceAssemblyError> {
    use rustix::fs::{Mode, OFlags, openat};
    use rustix::io::Errno;

    let descriptor = match openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(Errno::NOENT) => return Ok(None),
        Err(Errno::LOOP | Errno::NOTDIR) => {
            return Err(unsafe_assembly_path(
                "assembly destination is a symlink or unsafe path alias",
            ));
        }
        Err(error) => return Err(std::io::Error::from(error).into()),
    };
    let file = File::from(descriptor);
    let identity = checked_owned_regular_identity(&file, exact_len, true)?;
    let capacity = exact_len;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|error| {
        unsafe_assembly_path(format!("unable to reserve assembly-file read: {error}"))
    })?;
    (&file)
        .take(
            u64::try_from(exact_len)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut bytes)?;
    if bytes.len() != exact_len
        || checked_owned_regular_identity(&file, exact_len, true)? != identity
    {
        return Err(unsafe_assembly_path(
            "assembly destination changed while its exact bytes were read",
        ));
    }
    file.sync_all()?;
    Ok(Some((bytes, identity)))
}

fn read_bounded_owned_regular_at(
    directory: &File,
    name: &OsStr,
    maximum_len: usize,
) -> Result<Vec<u8>, PerfEvidenceAssemblyError> {
    use rustix::fs::{Mode, OFlags, openat};

    let descriptor = match openat(
        directory,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR) => {
            return Err(unsafe_assembly_path(
                "attempt input is a symlink or unsafe path alias",
            ));
        }
        Err(error) => return Err(std::io::Error::from(error).into()),
    };
    let file = File::from(descriptor);
    let identity = checked_owned_regular_identity(&file, maximum_len, false)?;
    let initial_len = usize::try_from(file.metadata()?.len()).map_err(|_| {
        unsafe_assembly_path("attempt input length cannot be represented in memory")
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(initial_len).map_err(|error| {
        unsafe_assembly_path(format!("unable to reserve attempt-input read: {error}"))
    })?;
    (&file)
        .take(
            u64::try_from(maximum_len)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut bytes)?;
    if bytes.len() > maximum_len
        || bytes.len() != initial_len
        || checked_owned_regular_identity(&file, maximum_len, false)? != identity
        || usize::try_from(file.metadata()?.len()).ok() != Some(initial_len)
    {
        return Err(unsafe_assembly_path(
            "attempt input changed identity or length while its bounded bytes were read",
        ));
    }
    Ok(bytes)
}

fn open_private_child_directory_at(
    parent: &File,
    name: &OsStr,
) -> Result<File, PerfEvidenceAssemblyError> {
    use rustix::fs::{Mode, OFlags, openat};
    use rustix::process::geteuid;

    let descriptor = match openat(
        parent,
        name,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::DIRECTORY | OFlags::NONBLOCK,
        Mode::empty(),
    ) {
        Ok(descriptor) => descriptor,
        Err(rustix::io::Errno::LOOP | rustix::io::Errno::NOTDIR) => {
            return Err(unsafe_assembly_path(
                "attempt artifact directory is a symlink or unsafe path alias",
            ));
        }
        Err(error) => return Err(std::io::Error::from(error).into()),
    };
    let directory = File::from(descriptor);
    let metadata = directory.metadata()?;
    if !metadata.is_dir() || metadata.uid() != geteuid().as_raw() || metadata.mode() & 0o022 != 0 {
        return Err(unsafe_assembly_path(
            "attempt artifact directory must be effective-user-owned and not group/world writable",
        ));
    }
    Ok(directory)
}

fn verify_named_assembly_identity(
    directory: &File,
    name: &OsStr,
    expected: AssemblyFileIdentity,
    exact_len: usize,
) -> Result<(), PerfEvidenceAssemblyError> {
    let (_, observed) = read_owned_regular_at(directory, name, exact_len)?.ok_or_else(|| {
        unsafe_assembly_path("assembly destination disappeared before commit verification")
    })?;
    if observed != expected {
        return Err(unsafe_assembly_path(
            "assembly destination inode changed before commit verification",
        ));
    }
    Ok(())
}

/// Whether the measured union exactly covers every Required plan cell.
///
/// `Complete` is coverage metadata only.  It is not a WIN, MISS, ratchet
/// decision, release authorization, or library-flip authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfEvidenceAssemblyCompleteness {
    /// Every Applicable/Required cell appears once.
    Complete,
    /// At least one Applicable/Required cell is absent.
    Incomplete,
}

/// Whether a structurally authentic assembly may enter downstream adjudication.
///
/// This is deliberately distinct from coverage completeness. Authentic
/// InvalidNull/NoDecision evidence remains durable but cannot become a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfEvidenceAssemblyReadiness {
    /// Every runnable cell is present and every Required cell is independently
    /// claim-eligible.
    ReadyForAdjudication,
    /// One or more Required or Diagnostic cells are absent.
    NoClaimIncomplete,
    /// The profile has no Required cells and therefore cannot make a claim.
    NoClaimNoRequiredCells,
    /// Required coverage is complete, but Required evidence is not claim-eligible.
    NoClaimInvalidEvidence,
}

/// Exact cell-level reason an authentic assembly cannot be adjudicated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyNoClaimCell {
    ordinal: usize,
    cell_id: String,
    role: EvidenceRole,
    terminal_status: EvidenceDecisionStatus,
    reasons: Vec<EvidenceReason>,
}

/// Source-level `NoClaim` input that assembly must preserve and propagate.
///
/// Only [`PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE`] on an actually partial
/// source is assembly-neutral. Every other authentic source `NoClaim` is stored
/// here and blocks downstream adjudication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyNoClaimSource {
    evidence_artifact_sha256: String,
    run_id: String,
    cell_ids: Vec<String>,
    reason: EvidenceReason,
}

impl PerfEvidenceAssemblyNoClaimSource {
    /// Full source-artifact identity retained by the outer envelope seal.
    #[must_use]
    pub fn evidence_artifact_sha256(&self) -> &str {
        &self.evidence_artifact_sha256
    }

    /// Bounded source run label.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Canonical cells whose source scope carried this `NoClaim`.
    #[must_use]
    pub fn cell_ids(&self) -> &[String] {
        &self.cell_ids
    }

    /// Exact persisted source-level reason.
    #[must_use]
    pub const fn reason(&self) -> &EvidenceReason {
        &self.reason
    }
}

impl PerfEvidenceAssemblyNoClaimCell {
    /// Canonical QG-1 matrix ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Stable matrix cell identity.
    #[must_use]
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }

    /// Required or diagnostic role from the independently reconstructed plan.
    #[must_use]
    pub const fn role(&self) -> EvidenceRole {
        self.role
    }

    /// Exact terminal evidence state.
    #[must_use]
    pub const fn terminal_status(&self) -> EvidenceDecisionStatus {
        self.terminal_status
    }

    /// Persisted producer reasons, in their original bounded order.
    #[must_use]
    pub fn reasons(&self) -> &[EvidenceReason] {
        &self.reasons
    }
}

/// One exact canonical matrix cell and its profile-specific applicability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyMatrixCell {
    ordinal: usize,
    cell_id: String,
    cell_contract_sha256: String,
    spec: PerfCellSpec,
    configured_threads: usize,
    applicability: PerfCellApplicability,
    applicability_reason: PerfCellApplicabilityReason,
}

impl PerfEvidenceAssemblyMatrixCell {
    /// Zero-based position in `PerfMatrixSpec::complete()`'s QG-1 slice.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Stable gate/fixture/metric identity.
    #[must_use]
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }

    /// Full authoritative cell contract.
    #[must_use]
    pub const fn spec(&self) -> &PerfCellSpec {
        &self.spec
    }

    /// Profile-specific C0 applicability, never an evidence role.
    #[must_use]
    pub const fn applicability(&self) -> PerfCellApplicability {
        self.applicability
    }
}

/// Independently sealed, self-describing projection of the authoritative
/// QG-1 matrix and the exact C0 applicability plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyMatrixManifest {
    schema_version: String,
    applicability_plan: PerfApplicabilityPlanBinding,
    cells: Vec<PerfEvidenceAssemblyMatrixCell>,
    matrix_manifest_sha256: String,
}

impl PerfEvidenceAssemblyMatrixManifest {
    fn derive(contract: &PlanContract) -> Result<Self, PerfEvidenceAssemblyError> {
        let mut manifest = Self {
            schema_version: PERF_EVIDENCE_ASSEMBLY_MATRIX_SCHEMA_VERSION.to_owned(),
            applicability_plan: contract.plan.binding.clone(),
            cells: contract
                .cells
                .iter()
                .enumerate()
                .map(|(ordinal, cell)| PerfEvidenceAssemblyMatrixCell {
                    ordinal,
                    cell_id: cell.cell_id.clone(),
                    cell_contract_sha256: cell.cell_contract_sha256.clone(),
                    spec: cell.spec.clone(),
                    configured_threads: cell.configured_threads,
                    applicability: cell.applicability,
                    applicability_reason: cell.applicability_reason,
                })
                .collect(),
            matrix_manifest_sha256: String::new(),
        };
        manifest.matrix_manifest_sha256 = manifest.recomputed_sha256()?;
        Ok(manifest)
    }

    /// Exact ordered canonical cells, including typed `NotApplicable` entries.
    #[must_use]
    pub fn cells(&self) -> &[PerfEvidenceAssemblyMatrixCell] {
        &self.cells
    }

    /// Independent content seal of this manifest.
    #[must_use]
    pub fn matrix_manifest_sha256(&self) -> &str {
        &self.matrix_manifest_sha256
    }

    /// Canonical pretty JSON bytes for independent persistence and reload.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if the in-memory value is malformed.
    pub fn to_json_pretty(&self) -> Result<Vec<u8>, PerfEvidenceAssemblyError> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Parse canonical bytes and independently verify the derived matrix seal.
    ///
    /// # Errors
    ///
    /// Rejects duplicate or unknown fields, noncanonical bytes, stale plans,
    /// fabricated cells, and stale matrix seals.
    pub fn from_verified_slice(contents: &[u8]) -> Result<Self, PerfEvidenceAssemblyError> {
        let probe =
            crate::machine_class_registry::parse_strict_json(contents).map_err(|error| {
                PerfEvidenceAssemblyError::Malformed {
                    reason: format!("matrix manifest is not strict JSON: {error}"),
                }
            })?;
        let manifest: Self = serde_json::from_value(probe.clone()).map_err(|error| {
            PerfEvidenceAssemblyError::Malformed {
                reason: format!("matrix manifest does not decode as the current schema: {error}"),
            }
        })?;
        if probe != serde_json::to_value(&manifest)?
            || contents != manifest.to_json_pretty()?.as_slice()
        {
            return Err(PerfEvidenceAssemblyError::Malformed {
                reason: "matrix manifest bytes are not the canonical exact encoding".to_owned(),
            });
        }
        manifest.verify()?;
        Ok(manifest)
    }

    /// Load and independently verify one exact derived matrix manifest.
    ///
    /// # Errors
    ///
    /// Returns a typed I/O or verification error.
    pub fn load_verified(path: &Path) -> Result<Self, PerfEvidenceAssemblyError> {
        Self::from_verified_slice(&fs::read(path)?)
    }

    fn verify(&self) -> Result<(), PerfEvidenceAssemblyError> {
        if self.schema_version != PERF_EVIDENCE_ASSEMBLY_MATRIX_SCHEMA_VERSION {
            return Err(PerfEvidenceAssemblyError::MatrixManifestMismatch {
                reason: format!("unsupported schema {:?}", self.schema_version),
            });
        }
        let contract = PlanContract::reconstruct(&self.applicability_plan)?;
        let expected = Self::derive(&contract)?;
        if &expected != self {
            return Err(PerfEvidenceAssemblyError::MatrixManifestMismatch {
                reason: "stored matrix projection differs from authoritative code".to_owned(),
            });
        }
        Ok(())
    }

    fn recomputed_sha256(&self) -> Result<String, PerfEvidenceAssemblyError> {
        let mut unsealed = self.clone();
        unsealed.matrix_manifest_sha256.clear();
        hash_serialized(MATRIX_MANIFEST_HASH_DOMAIN, &unsealed)
    }
}

/// Explicit semantic seal for the run-label-independent evidence projection.
///
/// Its preimage excludes opaque source run IDs and invocation-local timestamps,
/// logs, and artifact-file seals. It retains measurement samples, verdict
/// inputs, build/ELF/lock/command/environment identity, machine and fixture
/// identity, estimator inputs, cell coordinates, terminal states, and
/// source-level `NoClaim` semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceSemanticCellSetSeal {
    schema_version: String,
    semantic_cell_set_sha256: String,
}

impl PerfEvidenceSemanticCellSetSeal {
    /// Versioned semantic projection identity.
    #[must_use]
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    /// Run-label-independent hash of the canonical measurement projection.
    #[must_use]
    pub fn semantic_cell_set_sha256(&self) -> &str {
        &self.semantic_cell_set_sha256
    }
}

/// Stable host facts that must agree across completed shards.
///
/// Per-shard configured engine widths are intentionally absent: the assembler
/// validates those against each shard's selected canonical cells instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfAssemblyMachineIdentity {
    fingerprint: String,
    os: String,
    arch: String,
    logical_cpus: usize,
    host_identity: String,
    producer_os: PerfProducerOs,
    physical_cores: usize,
    logical_threads: usize,
    process_available_threads: usize,
    runtime_detected_isa: Vec<String>,
    cpu_affinity_allowed_list: Option<String>,
    affinity_or_cpuset_cap: Option<String>,
}

impl PerfAssemblyMachineIdentity {
    fn from_execution(
        fingerprint: &str,
        os: &str,
        arch: &str,
        logical_cpus: usize,
        execution: &PerfExecutionProvenance,
    ) -> Self {
        Self {
            fingerprint: fingerprint.to_owned(),
            os: os.to_owned(),
            arch: arch.to_owned(),
            logical_cpus,
            host_identity: execution.host_identity.clone(),
            producer_os: execution.producer_os,
            physical_cores: execution.physical_cores,
            logical_threads: execution.logical_threads,
            process_available_threads: execution.process_available_threads,
            runtime_detected_isa: execution.runtime_detected_isa.clone(),
            cpu_affinity_allowed_list: execution.cpu_affinity_allowed_list.clone(),
            affinity_or_cpuset_cap: execution.affinity_or_cpuset_cap.clone(),
        }
    }
}

/// Invocation-partition-independent identity of the immutable corpus universe.
///
/// `CorpusIdentity::document_count` and `content_bytes` describe what one
/// process measured, so legitimate disjoint shards may differ on those fields.
/// The assembler retains those exact observations inside each source artifact
/// while comparing only the shared generator and universe seals here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PerfAssemblyCorpusIdentity {
    corpus_sha256: String,
    query_set_sha256: Option<String>,
    qrels_sha256: Option<String>,
    generator_seed: u64,
    generator_revision: String,
}

impl PerfAssemblyCorpusIdentity {
    fn from_observation(corpus: &CorpusIdentity) -> Self {
        Self {
            corpus_sha256: corpus.corpus_sha256.clone(),
            query_set_sha256: corpus.query_set_sha256.clone(),
            qrels_sha256: corpus.qrels_sha256.clone(),
            generator_seed: corpus.generator_seed,
            generator_revision: corpus.generator_revision.clone(),
        }
    }
}

/// Exact compatibility envelope shared by every completed source shard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyCompatibility {
    profile: MachineProfileKey,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    canonicalization: MachineClassCanonicalizationBinding,
    runner_hardware_sha256: String,
    runner_execution_identity_sha256: String,
    runner_durability_sha256: String,
    manifest_sha256: String,
    run_window: String,
    build: BuildIdentity,
    machine: PerfAssemblyMachineIdentity,
    corpus: PerfAssemblyCorpusIdentity,
    policy: EvidencePolicy,
    estimator: PairedEstimatorConfig,
}

impl PerfEvidenceAssemblyCompatibility {
    /// Profile whose shards this envelope admits.
    #[must_use]
    pub const fn profile(&self) -> MachineProfileKey {
        self.profile
    }

    /// Capacity interpretation frozen by the admitted execution profile.
    #[must_use]
    pub const fn capacity_semantics(&self) -> ExecutionCapacitySemantics {
        self.capacity_semantics
    }

    /// Exact frozen execution capacity.
    #[must_use]
    pub const fn execution_capacity(&self) -> u64 {
        self.execution_capacity
    }

    /// Widest canonical QG-1 cell admitted by the profile.
    #[must_use]
    pub const fn max_exercised_cell_width(&self) -> u64 {
        self.max_exercised_cell_width
    }
}

/// Exact H2 process receipt and the raw log bytes it names, when present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfAssemblyProcessReceipt {
    process_receipt_sha256: String,
    receipt: LocalPerfAttemptReceipt,
    run_log_bytes: Option<Vec<u8>>,
}

impl PerfAssemblyProcessReceipt {
    /// SHA-256 of the exact canonical compact H2 process receipt.
    #[must_use]
    pub fn process_receipt_sha256(&self) -> &str {
        &self.process_receipt_sha256
    }

    /// Independently verified typed H2 process receipt.
    #[must_use]
    pub const fn receipt(&self) -> &LocalPerfAttemptReceipt {
        &self.receipt
    }

    /// Exact raw child log retained by the assembly when the receipt names it.
    #[must_use]
    pub fn run_log_bytes(&self) -> Option<&[u8]> {
        self.run_log_bytes.as_deref()
    }

    fn verify(&self) -> Result<(), PerfEvidenceAssemblyError> {
        let receipt_bytes = self.receipt.to_json_bytes()?;
        if LocalPerfAttemptReceipt::from_verified_slice(&receipt_bytes)? != self.receipt
            || sha256_hex(&receipt_bytes) != self.process_receipt_sha256
        {
            return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                reason: "retained H2 process receipt is not its exact canonical sealed object"
                    .to_owned(),
            });
        }
        match (&self.run_log_bytes, self.receipt.run_log_sha256()) {
            (Some(bytes), Some(_)) => self.receipt.verify_run_log(bytes)?,
            (None, None) => {}
            (Some(_), None) => {
                return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                    reason: "assembly retained an unbound run log".to_owned(),
                });
            }
            (None, Some(_)) => {
                return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                    reason: "H2 process receipt binds a run log that the assembly omitted"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PerfAssemblyCompletedInputs {
    threshold_artifact_sha256: String,
    threshold_artifact: PerfGateArtifact,
    prebinding_evidence_file_sha256: String,
    prebinding_artifact: PerfEvidenceArtifact,
    bound_evidence_file_sha256: String,
    evidence_content_seal_sha256: String,
    artifact: PerfEvidenceArtifact,
    runner_receipt_sha256: String,
    runner_artifact_manifest_sha256: String,
    run_id: String,
    cell_ids: Vec<String>,
}

/// Opaque exact attempt-directory input. Callers can obtain one only by
/// descriptor-relative verification of the H2 receipt and every bound file.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedLocalPerfAttemptBundle {
    process: PerfAssemblyProcessReceipt,
    completed: Option<PerfAssemblyCompletedInputs>,
}

impl VerifiedLocalPerfAttemptBundle {
    /// Load one canonical absolute private attempt directory and verify the
    /// exact H2 receipt/log/runner/manifest/threshold/evidence chain.
    ///
    /// Failed H2 outcomes retain only their process receipt and bound log.
    /// Any orphan completed artifacts beside a failed receipt are ignored.
    ///
    /// # Errors
    ///
    /// Rejects path aliases, symlinks, hard links, oversized inputs,
    /// noncanonical bytes, substitutions, or any cross-object mismatch.
    pub fn load_verified(attempt_dir: &Path) -> Result<Self, PerfEvidenceAssemblyError> {
        Self::load_verified_against_qg1_authorities(attempt_dir, &[])
    }

    /// Load one canonical attempt directory against the QG-1 expectations
    /// retained for that exact attempt input.
    ///
    /// An empty authority slice is the legacy authority-free path and therefore
    /// rejects a completed shard carrying authority-bound QG-1 evidence.
    ///
    /// # Errors
    ///
    /// Returns the same strict path, receipt, and evidence failures as
    /// [`Self::load_verified`], including missing, foreign, or duplicate
    /// retained QG-1 authority.
    pub fn load_verified_against_qg1_authorities(
        attempt_dir: &Path,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<Self, PerfEvidenceAssemblyError> {
        let directory = open_pinned_assembly_directory(attempt_dir, true)?;
        let directory_identity = verify_pinned_assembly_directory(attempt_dir, &directory, true)?;
        let attempt_bytes = read_bounded_owned_regular_at(
            &directory,
            OsStr::new("QG-1.attempt.json"),
            PERF_ASSEMBLY_MAX_RECEIPT_BYTES,
        )?;
        let receipt = LocalPerfAttemptReceipt::from_verified_slice(&attempt_bytes)?;
        if receipt.gate() != PerfGate::Qg1.label() {
            return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                reason: "attempt bundle is not an exact QG-1 receipt".to_owned(),
            });
        }
        let run_log_bytes = if receipt.run_log_sha256().is_some() {
            Some(read_bounded_owned_regular_at(
                &directory,
                OsStr::new("run.log"),
                PERF_ASSEMBLY_MAX_ARTIFACT_BYTES,
            )?)
        } else {
            None
        };
        let process = PerfAssemblyProcessReceipt {
            process_receipt_sha256: sha256_hex(&attempt_bytes),
            receipt,
            run_log_bytes,
        };
        process.verify()?;
        let completed = if process.receipt.outcome() == LocalPerfAttemptOutcome::Completed {
            Some(load_completed_attempt(
                &directory,
                &process,
                external_qg1_authorities,
            )?)
        } else {
            None
        };
        if verify_pinned_assembly_directory(attempt_dir, &directory, true)? != directory_identity {
            return Err(unsafe_assembly_path(
                "attempt directory changed identity while its exact bundle was loaded",
            ));
        }
        let bundle = Self { process, completed };
        bundle.verify_against_qg1_authorities(external_qg1_authorities)?;
        Ok(bundle)
    }

    /// Exact verified H2 process receipt shared by successful and failed runs.
    #[must_use]
    pub const fn process(&self) -> &PerfAssemblyProcessReceipt {
        &self.process
    }

    fn verify_against_qg1_authorities(
        &self,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> Result<(), PerfEvidenceAssemblyError> {
        self.process.verify()?;
        match (self.process.receipt.outcome(), &self.completed) {
            (LocalPerfAttemptOutcome::Completed, Some(completed)) => {
                verify_completed_inputs(&self.process, completed, external_qg1_authorities)
            }
            (LocalPerfAttemptOutcome::Completed, None) => {
                Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                    reason: "completed H2 receipt has no exact completed artifact bundle"
                        .to_owned(),
                })
            }
            (_, None) => Ok(()),
            (_, Some(_)) => Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                reason: "failed H2 receipt cannot contribute completed artifacts".to_owned(),
            }),
        }
    }
}

/// Durable failed H2 attempt retained by the assembly. It contributes no
/// measured cells and cannot be caller-authored independently of the receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyFailedAttempt {
    process: PerfAssemblyProcessReceipt,
}

impl PerfEvidenceAssemblyFailedAttempt {
    /// Exact verified H2 process receipt.
    #[must_use]
    pub const fn process(&self) -> &PerfAssemblyProcessReceipt {
        &self.process
    }

    fn verify(&self) -> Result<(), PerfEvidenceAssemblyError> {
        self.process.verify()?;
        if self.process.receipt.outcome() == LocalPerfAttemptOutcome::Completed {
            return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
                reason: "completed H2 receipt was stored as a failed attempt".to_owned(),
            });
        }
        Ok(())
    }
}

fn canonical_evidence_bytes(
    artifact: &PerfEvidenceArtifact,
) -> Result<Vec<u8>, PerfEvidenceAssemblyError> {
    Ok(artifact.canonical_json()?.into_bytes())
}

fn canonical_threshold_bytes(
    artifact: &PerfGateArtifact,
) -> Result<Vec<u8>, PerfEvidenceAssemblyError> {
    artifact
        .to_json_pretty()
        .map(String::into_bytes)
        .map_err(|error| PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: format!("threshold artifact cannot be canonically encoded: {error}"),
        })
}

fn invalid_threshold_join(reason: impl Into<String>) -> PerfEvidenceAssemblyError {
    PerfEvidenceAssemblyError::InvalidAttemptBundle {
        reason: reason.into(),
    }
}

fn projected_ratio_distribution(
    samples: &[PerfRawSample],
) -> Result<DistributionSummary, PerfEvidenceAssemblyError> {
    let mut blocks = BTreeMap::<u64, (Option<f64>, Option<f64>)>::new();
    for sample in samples {
        let elapsed_ns = sample
            .ended_ns
            .checked_sub(sample.started_ns)
            .filter(|elapsed| *elapsed != 0)
            .ok_or_else(|| invalid_threshold_join("threshold projection has invalid timing"))?;
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ns = elapsed_ns as f64;
        let value = match sample.scope.semantics {
            PerfMetricSemantics::Throughput => {
                #[allow(clippy::cast_precision_loss)]
                let work_units = sample.work_units.ok_or_else(|| {
                    invalid_threshold_join("throughput projection omits work units")
                })? as f64;
                work_units * 1_000_000_000.0 / elapsed_ns
            }
            PerfMetricSemantics::Duration => elapsed_ns,
            PerfMetricSemantics::GaugeHigherIsBetter | PerfMetricSemantics::GaugeLowerIsBetter => {
                sample.observed_value.ok_or_else(|| {
                    invalid_threshold_join("gauge projection omits its observation")
                })?
            }
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_threshold_join(
                "threshold projection contains a nonpositive or nonfinite sample",
            ));
        }
        let block = blocks.entry(sample.block_id).or_default();
        let duplicate = match sample.arm {
            PerfSampleArm::Control => block.0.replace(value).is_some(),
            PerfSampleArm::Treatment => block.1.replace(value).is_some(),
        };
        if duplicate {
            return Err(invalid_threshold_join(
                "threshold projection repeats one paired block arm",
            ));
        }
    }
    let ratios = blocks
        .into_values()
        .map(|(control, treatment)| {
            let control = control.ok_or_else(|| {
                invalid_threshold_join("threshold projection has an incomplete paired block")
            })?;
            let treatment = treatment.ok_or_else(|| {
                invalid_threshold_join("threshold projection has an incomplete paired block")
            })?;
            Ok(treatment / control)
        })
        .collect::<Result<Vec<_>, PerfEvidenceAssemblyError>>()?;
    DistributionSummary::from_samples(&ratios).map_err(|error| {
        invalid_threshold_join(format!("threshold ratio projection is invalid: {error}"))
    })
}

fn threshold_projection_from_evidence(
    artifact: &PerfEvidenceArtifact,
) -> Result<Vec<PerfCellResult>, PerfEvidenceAssemblyError> {
    let mut rows = Vec::with_capacity(artifact.cells.len().saturating_mul(5));
    for cell in &artifact.cells {
        let EvidenceCellBody::Paired {
            paired,
            treatment_arm_null,
            ..
        } = &cell.body
        else {
            return Err(invalid_threshold_join(
                "QG-1 threshold projection names non-paired evidence",
            ));
        };
        let treatment_arm_null = treatment_arm_null.as_deref().ok_or_else(|| {
            invalid_threshold_join("QG-1 threshold projection omits the Quill/Quill null")
        })?;
        let (treatment_engine, control_engine) = if cell.spec.metric == "tokenize_docs_per_second" {
            ("quill_tokenizer", "quill_tokenizer_null")
        } else {
            ("quill", "tantivy")
        };
        rows.extend([
            PerfCellResult {
                fixture: cell.spec.fixture.clone(),
                metric: cell.spec.metric.clone(),
                engine: treatment_engine.to_owned(),
                unit: cell.spec.unit.clone(),
                distribution: paired.effect.treatment.clone(),
            },
            PerfCellResult {
                fixture: cell.spec.fixture.clone(),
                metric: cell.spec.metric.clone(),
                engine: control_engine.to_owned(),
                unit: cell.spec.unit.clone(),
                distribution: paired.effect.control.clone(),
            },
            PerfCellResult {
                fixture: cell.spec.fixture.clone(),
                metric: format!("{}_quill_over_tantivy", cell.spec.metric),
                engine: "paired_ab".to_owned(),
                unit: "ratio".to_owned(),
                distribution: projected_ratio_distribution(&paired.effect_samples)?,
            },
            PerfCellResult {
                fixture: cell.spec.fixture.clone(),
                metric: format!("{}_tantivy_over_tantivy", cell.spec.metric),
                engine: "paired_null".to_owned(),
                unit: "ratio".to_owned(),
                distribution: projected_ratio_distribution(&paired.null_samples)?,
            },
            PerfCellResult {
                fixture: cell.spec.fixture.clone(),
                metric: format!("{}_quill_over_quill", cell.spec.metric),
                engine: "paired_null_quill".to_owned(),
                unit: "ratio".to_owned(),
                distribution: projected_ratio_distribution(&treatment_arm_null.null_samples)?,
            },
        ]);
    }
    Ok(rows)
}

fn verify_threshold_evidence_join(
    threshold: &PerfGateArtifact,
    evidence: &PerfEvidenceArtifact,
) -> Result<(), PerfEvidenceAssemblyError> {
    let contract = PlanContract::reconstruct(&evidence.applicability_plan)?;
    let runnable_count = contract
        .cells
        .iter()
        .filter(|cell| cell.applicability.is_runnable())
        .count();
    if threshold.schema_version != PERF_ARTIFACT_SCHEMA_VERSION
        || threshold.gate != PerfGate::Qg1
        || threshold.applicability_plan.as_ref() != Some(&evidence.applicability_plan)
        || threshold.bench_elf_sha256 != evidence.provenance.build.executable_sha256
        || threshold.machine_fingerprint != evidence.provenance.machine.fingerprint
        || threshold.execution.as_ref() != Some(&evidence.provenance.machine.execution)
        || threshold.git_rev != evidence.provenance.build.git_revision
        || threshold.run_window != evidence.provenance.run_window
        || threshold.run_id != evidence.provenance.run_id
        || threshold.corpus_manifest_hash != evidence.provenance.corpus.corpus_sha256
        || threshold.manifest_sha256 != evidence.provenance.manifest_sha256
        || threshold.manifest_sha256 != evidence.applicability_plan.normalized_perf_manifest_sha256
        || threshold.laws_attested != (evidence.cells.len() == runnable_count)
        || threshold.cells != threshold_projection_from_evidence(evidence)?
    {
        return Err(invalid_threshold_join(
            "threshold schema, identity, law scope, or cell projection contradicts bound evidence",
        ));
    }
    Ok(())
}

fn load_completed_attempt(
    directory: &File,
    process: &PerfAssemblyProcessReceipt,
    external_qg1_authorities: &[&Qg1ExpectedAuthority],
) -> Result<PerfAssemblyCompletedInputs, PerfEvidenceAssemblyError> {
    let run_log = process.run_log_bytes.as_deref().ok_or_else(|| {
        PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed H2 attempt omitted its exact bound run log".to_owned(),
        }
    })?;
    let bound_bytes = read_bounded_owned_regular_at(
        directory,
        OsStr::new("QG-1.bound.evidence.json"),
        PERF_ASSEMBLY_MAX_ARTIFACT_BYTES,
    )?;
    let runner_bytes = read_bounded_owned_regular_at(
        directory,
        OsStr::new("QG-1.runner.json"),
        PERF_ASSEMBLY_MAX_RECEIPT_BYTES,
    )?;
    let manifest_bytes = read_bounded_owned_regular_at(
        directory,
        OsStr::new("QG-1.artifacts.json"),
        PERF_ASSEMBLY_MAX_RECEIPT_BYTES,
    )?;
    let artifacts_directory = open_private_child_directory_at(directory, OsStr::new("artifacts"))?;
    let threshold_bytes = read_bounded_owned_regular_at(
        &artifacts_directory,
        OsStr::new("QG-1.json"),
        PERF_ASSEMBLY_MAX_ARTIFACT_BYTES,
    )?;
    let prebinding_bytes = read_bounded_owned_regular_at(
        &artifacts_directory,
        OsStr::new("QG-1.evidence.json"),
        PERF_ASSEMBLY_MAX_ARTIFACT_BYTES,
    )?;

    let artifact = PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
        &bound_bytes,
        external_qg1_authorities,
    )?;
    if canonical_evidence_bytes(&artifact)? != bound_bytes {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "bound evidence is not exact canonical pretty JSON".to_owned(),
        });
    }
    process
        .receipt
        .verify_bound_evidence_against_qg1_authorities(&bound_bytes, external_qg1_authorities)?;
    let prebinding_artifact = PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
        &prebinding_bytes,
        external_qg1_authorities,
    )?;
    if canonical_evidence_bytes(&prebinding_artifact)? != prebinding_bytes
        || artifact.reconstructed_prebinding_bytes()? != prebinding_bytes
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason:
                "pre-binding evidence is not the exact source reconstructed from bound evidence"
                    .to_owned(),
        });
    }
    let threshold_artifact = PerfGateArtifact::from_verified_measured_slice(&threshold_bytes)
        .map_err(|error| PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: format!("threshold artifact failed strict verified reload: {error}"),
        })?;
    verify_threshold_evidence_join(&threshold_artifact, &artifact)?;
    let identity = artifact.machine_class.identity().ok_or_else(|| {
        PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed bound evidence has no admitted runner identity".to_owned(),
        }
    })?;
    identity.verify()?;
    identity.verify_artifact_inputs(run_log, &threshold_bytes, &prebinding_bytes)?;
    if runner_bytes != identity.receipt_json().as_bytes()
        || sha256_hex(&runner_bytes) != identity.receipt_sha256()
        || process.receipt.runner_receipt_sha256() != Some(identity.receipt_sha256())
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "root runner receipt differs from the exact nested admitted receipt".to_owned(),
        });
    }
    let manifest = identity.artifact_manifest().ok_or_else(|| {
        PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed runner identity has no exact artifact manifest".to_owned(),
        }
    })?;
    let canonical_manifest = manifest.manifest().to_json_bytes()?;
    if manifest_bytes != canonical_manifest
        || sha256_hex(&manifest_bytes) != manifest.manifest_sha256()
        || process.receipt.runner_artifact_manifest_sha256() != Some(manifest.manifest_sha256())
        || manifest.manifest().applicability_plan() != &artifact.applicability_plan
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "root artifact manifest differs from the exact nested manifest".to_owned(),
        });
    }
    let cell_ids = artifact
        .cells
        .iter()
        .map(|cell| cell.cell_id.clone())
        .collect::<Vec<_>>();
    if threshold_artifact.gate != PerfGate::Qg1
        || threshold_artifact.applicability_plan.as_ref()
            != Some(process.receipt.applicability_plan())
        || threshold_artifact.run_id != process.receipt.run_id()
        || threshold_artifact.run_window != process.receipt.run_window()
        || threshold_artifact.git_rev != artifact.provenance.build.git_revision
        || threshold_artifact.bench_elf_sha256 != artifact.provenance.build.executable_sha256
        || threshold_artifact.execution.as_ref() != Some(&artifact.provenance.machine.execution)
        || cell_ids != process.receipt.selected_cell_ids()
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "threshold, evidence, selection, or producer identities disagree".to_owned(),
        });
    }
    let runner_receipt_sha256 = identity.receipt_sha256().to_owned();
    let runner_artifact_manifest_sha256 = manifest.manifest_sha256().to_owned();
    Ok(PerfAssemblyCompletedInputs {
        threshold_artifact_sha256: sha256_hex(&threshold_bytes),
        threshold_artifact,
        prebinding_evidence_file_sha256: sha256_hex(&prebinding_bytes),
        prebinding_artifact,
        bound_evidence_file_sha256: sha256_hex(&bound_bytes),
        evidence_content_seal_sha256: artifact.artifact_sha256.clone(),
        artifact,
        runner_receipt_sha256,
        runner_artifact_manifest_sha256,
        run_id: process.receipt.run_id().to_owned(),
        cell_ids,
    })
}

fn verify_completed_inputs(
    process: &PerfAssemblyProcessReceipt,
    completed: &PerfAssemblyCompletedInputs,
    external_qg1_authorities: &[&Qg1ExpectedAuthority],
) -> Result<(), PerfEvidenceAssemblyError> {
    completed
        .artifact
        .verify_integrity_against_qg1_authorities(external_qg1_authorities)?;
    completed
        .prebinding_artifact
        .verify_integrity_against_qg1_authorities(external_qg1_authorities)?;
    let bound_bytes = canonical_evidence_bytes(&completed.artifact)?;
    let prebinding_bytes = canonical_evidence_bytes(&completed.prebinding_artifact)?;
    let threshold_bytes = canonical_threshold_bytes(&completed.threshold_artifact)?;
    process
        .receipt
        .verify_bound_evidence_against_qg1_authorities(&bound_bytes, external_qg1_authorities)?;
    verify_threshold_evidence_join(&completed.threshold_artifact, &completed.artifact)?;
    if completed.artifact.reconstructed_prebinding_bytes()? != prebinding_bytes
        || sha256_hex(&bound_bytes) != completed.bound_evidence_file_sha256
        || completed.artifact.artifact_sha256 != completed.evidence_content_seal_sha256
        || sha256_hex(&prebinding_bytes) != completed.prebinding_evidence_file_sha256
        || sha256_hex(&threshold_bytes) != completed.threshold_artifact_sha256
        || completed.run_id != process.receipt.run_id()
        || completed.cell_ids != process.receipt.selected_cell_ids()
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "retained completed attempt wrapper no longer matches its exact artifacts"
                .to_owned(),
        });
    }
    let run_log = process.run_log_bytes.as_deref().ok_or_else(|| {
        PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed attempt lost its exact run log".to_owned(),
        }
    })?;
    let identity = completed.artifact.machine_class.identity().ok_or_else(|| {
        PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed attempt lost its admitted runner identity".to_owned(),
        }
    })?;
    identity.verify_artifact_inputs(run_log, &threshold_bytes, &prebinding_bytes)?;
    let manifest = identity.artifact_manifest().ok_or_else(|| {
        PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed attempt lost its artifact manifest".to_owned(),
        }
    })?;
    if sha256_hex(identity.receipt_json().as_bytes()) != completed.runner_receipt_sha256
        || identity.receipt_sha256() != completed.runner_receipt_sha256
        || sha256_hex(&manifest.manifest().to_json_bytes()?)
            != completed.runner_artifact_manifest_sha256
        || manifest.manifest_sha256() != completed.runner_artifact_manifest_sha256
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "retained runner or manifest identity no longer verifies".to_owned(),
        });
    }
    Ok(())
}

/// Immutable completed shard retained verbatim inside an assembly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblySource {
    process: PerfAssemblyProcessReceipt,
    threshold_artifact_sha256: String,
    threshold_artifact: PerfGateArtifact,
    prebinding_evidence_file_sha256: String,
    prebinding_artifact: PerfEvidenceArtifact,
    bound_evidence_file_sha256: String,
    evidence_content_seal_sha256: String,
    runner_receipt_sha256: String,
    runner_artifact_manifest_sha256: String,
    run_id: String,
    cell_ids: Vec<String>,
    artifact: PerfEvidenceArtifact,
}

impl PerfEvidenceAssemblySource {
    /// Exact verified H2 process receipt and raw run log.
    #[must_use]
    pub const fn process(&self) -> &PerfAssemblyProcessReceipt {
        &self.process
    }

    /// SHA-256 of exact canonical bound-evidence file bytes.
    #[must_use]
    pub fn bound_evidence_file_sha256(&self) -> &str {
        &self.bound_evidence_file_sha256
    }

    /// Independent content seal stored inside the evidence object.
    #[must_use]
    pub fn evidence_artifact_sha256(&self) -> &str {
        &self.evidence_content_seal_sha256
    }

    /// Exact sealed runner receipt retained by the source artifact.
    #[must_use]
    pub fn runner_receipt_sha256(&self) -> &str {
        &self.runner_receipt_sha256
    }

    /// Exact artifact-manifest seal embedded in the admitted runner receipt.
    #[must_use]
    pub fn runner_artifact_manifest_sha256(&self) -> &str {
        &self.runner_artifact_manifest_sha256
    }

    /// Source run ID.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Canonically ordered cell IDs contributed by this source.
    #[must_use]
    pub fn cell_ids(&self) -> &[String] {
        &self.cell_ids
    }

    /// Full independently verified bound source artifact, including raw
    /// samples and its own runner and artifact-manifest receipts.
    #[must_use]
    pub const fn artifact(&self) -> &PerfEvidenceArtifact {
        &self.artifact
    }
}

/// Canonical provenance pointer for one assembled cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceCellSource {
    ordinal: usize,
    cell_id: String,
    role: EvidenceRole,
    terminal_status: EvidenceDecisionStatus,
    evidence_artifact_sha256: String,
    runner_receipt_sha256: String,
    runner_artifact_manifest_sha256: String,
    run_id: String,
}

impl PerfEvidenceCellSource {
    /// Canonical QG-1 matrix ordinal.
    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Stable `QG-1/fixture/metric` identity.
    #[must_use]
    pub fn cell_id(&self) -> &str {
        &self.cell_id
    }

    /// Required or diagnostic evidence role derived from the plan.
    #[must_use]
    pub const fn role(&self) -> EvidenceRole {
        self.role
    }

    /// Terminal cell status derived from raw evidence.
    #[must_use]
    pub const fn terminal_status(&self) -> EvidenceDecisionStatus {
        self.terminal_status
    }

    /// Exact retained evidence artifact that owns this cell.
    #[must_use]
    pub fn evidence_artifact_sha256(&self) -> &str {
        &self.evidence_artifact_sha256
    }

    /// Exact admitted runner receipt for this cell's source shard.
    #[must_use]
    pub fn runner_receipt_sha256(&self) -> &str {
        &self.runner_receipt_sha256
    }

    /// Exact runner artifact-manifest seal for this cell's source shard.
    #[must_use]
    pub fn runner_artifact_manifest_sha256(&self) -> &str {
        &self.runner_artifact_manifest_sha256
    }

    /// Source-shard run label retained only in the full envelope.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

/// Counts derived from the authoritative matrix and the assembled union.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyCounts {
    canonical_cells: usize,
    required_cells: usize,
    diagnostic_cells: usize,
    not_applicable_cells: usize,
    measured_cells: usize,
    completed_shards: usize,
    failed_shards: usize,
}

impl PerfEvidenceAssemblyCounts {
    /// Total canonical QG-1 cells, including typed `NotApplicable` entries.
    #[must_use]
    pub const fn canonical_cells(&self) -> usize {
        self.canonical_cells
    }

    /// Applicable cells required by this profile.
    #[must_use]
    pub const fn required_cells(&self) -> usize {
        self.required_cells
    }

    /// Applicable diagnostic-only cells for this profile.
    #[must_use]
    pub const fn diagnostic_cells(&self) -> usize {
        self.diagnostic_cells
    }

    /// Number of measured cells in the disjoint source union.
    #[must_use]
    pub const fn measured_cells(&self) -> usize {
        self.measured_cells
    }

    /// Number of canonical cells classified `NotApplicable` by the frozen plan.
    #[must_use]
    pub const fn not_applicable_cells(&self) -> usize {
        self.not_applicable_cells
    }

    /// Independently verified completed source shards.
    #[must_use]
    pub const fn completed_shards(&self) -> usize {
        self.completed_shards
    }

    /// Durable failed source attempts.
    #[must_use]
    pub const fn failed_shards(&self) -> usize {
        self.failed_shards
    }
}

/// Strict, hash-sealed, offline QG-1 assembly receipt.
///
/// The schema deliberately has no aggregate `machine_class`, runner receipt,
/// or promotion decision field.  Every source artifact remains independently
/// verifiable and the assembly itself can only state coverage completeness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerfEvidenceAssemblyArtifact {
    schema_version: String,
    gate: PerfGate,
    applicability_plan: PerfApplicabilityPlanBinding,
    matrix_manifest: PerfEvidenceAssemblyMatrixManifest,
    run_window: String,
    completeness: PerfEvidenceAssemblyCompleteness,
    readiness: PerfEvidenceAssemblyReadiness,
    retry_predicate: Option<String>,
    diagnostic_retry_predicate: Option<String>,
    adjudication_retry_predicate: Option<String>,
    non_adjudicable_cells: Vec<PerfEvidenceAssemblyNoClaimCell>,
    non_adjudicable_sources: Vec<PerfEvidenceAssemblyNoClaimSource>,
    compatibility: Option<PerfEvidenceAssemblyCompatibility>,
    counts: PerfEvidenceAssemblyCounts,
    missing_required_cell_ids: Vec<String>,
    missing_diagnostic_cell_ids: Vec<String>,
    cell_sources: Vec<PerfEvidenceCellSource>,
    source_shards: Vec<PerfEvidenceAssemblySource>,
    failed_shards: Vec<PerfEvidenceAssemblyFailedAttempt>,
    semantic_cell_set: PerfEvidenceSemanticCellSetSeal,
    assembly_sha256: String,
}

impl PerfEvidenceAssemblyArtifact {
    /// Assemble exact H2 attempt-directory bundles.
    ///
    /// Input order has no effect on the sealed output. Missing cells produce a
    /// durable `Incomplete` receipt rather than an error; overlap, incompatible
    /// provenance, or any unverified source fails closed.
    ///
    /// # Errors
    ///
    /// Returns a typed structural or provenance error for invalid inputs.
    pub fn assemble(
        attempts: Vec<VerifiedLocalPerfAttemptBundle>,
    ) -> Result<Self, PerfEvidenceAssemblyError> {
        Self::assemble_against_qg1_authorities(
            attempts
                .into_iter()
                .map(|attempt| (attempt, &[] as &[&Qg1ExpectedAuthority]))
                .collect(),
        )
    }

    /// Assemble exact H2 attempt bundles against their separately retained
    /// QG-1 authorities.
    ///
    /// Each tuple binds one exact input bundle to only the authority slice
    /// retained for that bundle. This deliberately rejects positional or
    /// role-string authority routing: an authority set cannot be borrowed from
    /// a neighboring shard. The authority-free [`Self::assemble`] entry passes
    /// empty slices and consequently fails closed for authority-bearing QG-1
    /// evidence while preserving ordinary non-QG behavior.
    ///
    /// # Errors
    ///
    /// Returns a typed structural or provenance error for invalid inputs,
    /// including absent, foreign, or duplicate retained QG-1 authority.
    pub fn assemble_against_qg1_authorities<'authority>(
        attempts: Vec<(
            VerifiedLocalPerfAttemptBundle,
            &'authority [&'authority Qg1ExpectedAuthority],
        )>,
    ) -> Result<Self, PerfEvidenceAssemblyError> {
        let total = attempts.len();
        if total == 0 {
            return Err(PerfEvidenceAssemblyError::EmptyAssembly);
        }
        if total > PERF_ASSEMBLY_MAX_SHARDS {
            return Err(PerfEvidenceAssemblyError::TooManyShards);
        }

        for (attempt, external_qg1_authorities) in &attempts {
            attempt.verify_against_qg1_authorities(external_qg1_authorities)?;
        }
        let first = attempts
            .first()
            .ok_or(PerfEvidenceAssemblyError::EmptyAssembly)?;
        let applicability_plan = first.0.process.receipt.applicability_plan().clone();
        let run_window = first.0.process.receipt.run_window().to_owned();
        let contract = PlanContract::reconstruct(&applicability_plan)?;
        let matrix_manifest = PerfEvidenceAssemblyMatrixManifest::derive(&contract)?;

        let mut source_shards = Vec::new();
        let mut failed_shards = Vec::new();
        let mut source_authorities = BTreeMap::new();
        for (attempt, external_qg1_authorities) in attempts {
            let VerifiedLocalPerfAttemptBundle { process, completed } = attempt;
            if let Some(completed) = completed {
                let source = source_from_completed_inputs(
                    process,
                    completed,
                    &contract,
                    external_qg1_authorities,
                )?;
                if source_authorities
                    .insert(
                        source.bound_evidence_file_sha256.clone(),
                        external_qg1_authorities,
                    )
                    .is_some()
                {
                    return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                        reason: "assembly repeats one exact bound-evidence source input".to_owned(),
                    });
                }
                source_shards.push(source);
            } else {
                failed_shards.push(PerfEvidenceAssemblyFailedAttempt { process });
            }
        }
        source_shards.sort_by(|left, right| {
            source_sort_key(left, &contract).cmp(&source_sort_key(right, &contract))
        });
        failed_shards.sort_by(|left, right| {
            (
                left.process.receipt.run_id(),
                left.process.process_receipt_sha256.as_str(),
            )
                .cmp(&(
                    right.process.receipt.run_id(),
                    right.process.process_receipt_sha256.as_str(),
                ))
        });
        let source_authorities = source_authorities
            .iter()
            .map(|(source_sha256, authorities)| (source_sha256.as_str(), *authorities))
            .collect::<Vec<_>>();

        let derived = derive_assembly(
            &applicability_plan,
            &run_window,
            &source_shards,
            &failed_shards,
            &source_authorities,
        )?;
        let semantic_cell_set = semantic_cell_set_seal(
            &applicability_plan,
            &matrix_manifest,
            &source_shards,
            &derived.cell_sources,
            &derived.non_adjudicable_sources,
        )?;
        let mut artifact = Self {
            schema_version: PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg1,
            applicability_plan,
            matrix_manifest,
            run_window,
            completeness: derived.completeness,
            readiness: derived.readiness,
            retry_predicate: derived.retry_predicate,
            diagnostic_retry_predicate: derived.diagnostic_retry_predicate,
            adjudication_retry_predicate: derived.adjudication_retry_predicate,
            non_adjudicable_cells: derived.non_adjudicable_cells,
            non_adjudicable_sources: derived.non_adjudicable_sources,
            compatibility: derived.compatibility,
            counts: derived.counts,
            missing_required_cell_ids: derived.missing_required_cell_ids,
            missing_diagnostic_cell_ids: derived.missing_diagnostic_cell_ids,
            cell_sources: derived.cell_sources,
            source_shards,
            failed_shards,
            semantic_cell_set,
            assembly_sha256: String::new(),
        };
        artifact.assembly_sha256 = artifact.recomputed_sha256()?;
        Ok(artifact)
    }

    /// Whether the source union covers every Applicable/Required canonical
    /// cell exactly once. Diagnostic coverage is reported separately. This is
    /// not a performance verdict.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(
            self.completeness,
            PerfEvidenceAssemblyCompleteness::Complete
        )
    }

    /// Exact profile-qualified applicability-plan identity.
    #[must_use]
    pub const fn applicability_plan(&self) -> &PerfApplicabilityPlanBinding {
        &self.applicability_plan
    }

    /// Independently sealed authoritative matrix and C0 applicability
    /// projection used by this assembly.
    #[must_use]
    pub const fn matrix_manifest(&self) -> &PerfEvidenceAssemblyMatrixManifest {
        &self.matrix_manifest
    }

    /// Current fail-closed downstream-adjudication readiness.
    #[must_use]
    pub const fn readiness(&self) -> PerfEvidenceAssemblyReadiness {
        self.readiness
    }

    /// Authentic cells whose terminal evidence cannot support adjudication.
    #[must_use]
    pub fn non_adjudicable_cells(&self) -> &[PerfEvidenceAssemblyNoClaimCell] {
        &self.non_adjudicable_cells
    }

    /// Authentic source-level `NoClaim` inputs that assembly cannot discharge.
    #[must_use]
    pub fn non_adjudicable_sources(&self) -> &[PerfEvidenceAssemblyNoClaimSource] {
        &self.non_adjudicable_sources
    }

    /// Shared candidate/rerun measurement window.
    #[must_use]
    pub fn run_window(&self) -> &str {
        &self.run_window
    }

    /// Derived compatibility envelope, absent only when every attempt failed
    /// before a completed evidence artifact existed.
    #[must_use]
    pub const fn compatibility(&self) -> Option<&PerfEvidenceAssemblyCompatibility> {
        self.compatibility.as_ref()
    }

    /// Canonical set difference against Applicable/Required plan cells.
    #[must_use]
    pub fn missing_required_cell_ids(&self) -> &[String] {
        &self.missing_required_cell_ids
    }

    /// Canonical set difference against Applicable/Diagnostic plan cells.
    #[must_use]
    pub fn missing_diagnostic_cell_ids(&self) -> &[String] {
        &self.missing_diagnostic_cell_ids
    }

    /// Whether every Required and Diagnostic plan cell is present exactly once.
    #[must_use]
    pub fn has_full_plan_coverage(&self) -> bool {
        self.is_complete() && self.missing_diagnostic_cell_ids.is_empty()
    }

    /// Bounded concrete retry predicate for missing coverage.
    #[must_use]
    pub fn retry_predicate(&self) -> Option<&str> {
        self.retry_predicate.as_deref()
    }

    /// Bounded retry predicate for mandatory full Diagnostic coverage.
    #[must_use]
    pub fn diagnostic_retry_predicate(&self) -> Option<&str> {
        self.diagnostic_retry_predicate.as_deref()
    }

    /// Bounded concrete retry predicate for authentic invalid evidence.
    #[must_use]
    pub fn adjudication_retry_predicate(&self) -> Option<&str> {
        self.adjudication_retry_predicate.as_deref()
    }

    /// Canonical cell-to-source receipt mapping.
    #[must_use]
    pub fn cell_sources(&self) -> &[PerfEvidenceCellSource] {
        &self.cell_sources
    }

    /// Independently sealed completed source shards.
    #[must_use]
    pub fn source_shards(&self) -> &[PerfEvidenceAssemblySource] {
        &self.source_shards
    }

    /// Durable failed attempts that contribute no measured cells.
    #[must_use]
    pub fn failed_shards(&self) -> &[PerfEvidenceAssemblyFailedAttempt] {
        &self.failed_shards
    }

    /// Counts derived from the authoritative matrix and plan.
    #[must_use]
    pub const fn counts(&self) -> &PerfEvidenceAssemblyCounts {
        &self.counts
    }

    /// Hash seal over the complete offline assembly.
    #[must_use]
    pub fn assembly_sha256(&self) -> &str {
        &self.assembly_sha256
    }

    /// Run-label-independent semantic measurement seal.
    #[must_use]
    pub const fn semantic_cell_set(&self) -> &PerfEvidenceSemanticCellSetSeal {
        &self.semantic_cell_set
    }

    /// Require exact Required-cell coverage before handing the receipt to a
    /// downstream adjudicator.
    ///
    /// # Errors
    ///
    /// Returns [`PerfEvidenceAssemblyError::IncompleteAssembly`] with the
    /// bounded retry predicate when any Applicable/Required cell is missing.
    pub fn require_complete(&self) -> Result<(), PerfEvidenceAssemblyError> {
        self.verify_integrity()?;
        if self.is_complete() {
            Ok(())
        } else {
            Err(PerfEvidenceAssemblyError::IncompleteAssembly {
                missing: self.missing_required_cell_ids.len(),
                retry_predicate: self.retry_predicate.clone().ok_or_else(|| {
                    PerfEvidenceAssemblyError::InconsistentAssembly {
                        reason: "incomplete assembly lacks its required retry predicate".to_owned(),
                    }
                })?,
            })
        }
    }

    /// Require exact Required and Diagnostic coverage for callers that need
    /// every runnable plan cell rather than only the claim-bearing subset.
    ///
    /// # Errors
    ///
    /// Returns the Required-coverage error first, then a separately typed
    /// Diagnostic-coverage error. Missing diagnostics block adjudication.
    pub fn require_full_plan_coverage(&self) -> Result<(), PerfEvidenceAssemblyError> {
        self.require_complete()?;
        if self.missing_diagnostic_cell_ids.is_empty() {
            Ok(())
        } else {
            Err(PerfEvidenceAssemblyError::IncompleteDiagnosticCoverage {
                missing: self.missing_diagnostic_cell_ids.len(),
                retry_predicate: self.diagnostic_retry_predicate.clone().ok_or_else(|| {
                    PerfEvidenceAssemblyError::InconsistentAssembly {
                        reason: "missing Diagnostic cells lack their retry predicate".to_owned(),
                    }
                })?,
            })
        }
    }

    /// Require complete coverage and independently claim-eligible evidence.
    ///
    /// # Errors
    ///
    /// Returns the ordinary incomplete error for holes, or a typed durable
    /// `NoClaim` error carrying exact cell/status/reason diagnostics.
    pub fn require_adjudicable(&self) -> Result<(), PerfEvidenceAssemblyError> {
        self.require_full_plan_coverage()?;
        if self.readiness == PerfEvidenceAssemblyReadiness::ReadyForAdjudication {
            Ok(())
        } else {
            let diagnostics = self
                .non_adjudicable_cells
                .iter()
                .filter(|diagnostic| diagnostic.role == EvidenceRole::Required)
                .cloned()
                .collect::<Vec<_>>();
            let source_diagnostics = self
                .non_adjudicable_sources
                .iter()
                .filter(|source| {
                    source.cell_ids.iter().any(|cell_id| {
                        self.matrix_manifest.cells.iter().any(|cell| {
                            cell.cell_id == *cell_id
                                && cell.applicability == PerfCellApplicability::Required
                        })
                    })
                })
                .cloned()
                .collect::<Vec<_>>();
            Err(PerfEvidenceAssemblyError::NonAdjudicableAssembly {
                cells: diagnostics.len(),
                sources: source_diagnostics.len(),
                retry_predicate: self.adjudication_retry_predicate.clone().ok_or_else(|| {
                    PerfEvidenceAssemblyError::InconsistentAssembly {
                        reason: "non-adjudicable assembly lacks its retry predicate".to_owned(),
                    }
                })?,
                diagnostics,
                source_diagnostics,
            })
        }
    }

    /// Resolve the assembled cells in canonical matrix order without copying
    /// their raw samples.
    ///
    /// # Errors
    ///
    /// Returns a structural error if an in-memory caller mutated a source after
    /// verification.
    pub fn cells_in_canonical_order(
        &self,
    ) -> Result<Vec<&EvidenceCell>, PerfEvidenceAssemblyError> {
        let sources = self
            .source_shards
            .iter()
            .map(|source| (source.bound_evidence_file_sha256.as_str(), source))
            .collect::<BTreeMap<_, _>>();
        self.cell_sources
            .iter()
            .map(|cell_source| {
                sources
                    .get(cell_source.evidence_artifact_sha256.as_str())
                    .and_then(|source| {
                        source
                            .artifact
                            .cells
                            .iter()
                            .find(|cell| cell.cell_id == cell_source.cell_id)
                    })
                    .ok_or_else(|| PerfEvidenceAssemblyError::InconsistentAssembly {
                        reason: format!(
                            "cell-source mapping for {:?} no longer resolves",
                            cell_source.cell_id
                        ),
                    })
            })
            .collect()
    }

    /// Canonical pretty JSON bytes for persistence or exact-byte comparison.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if an invalid non-finite value was
    /// introduced after verification.
    pub fn to_json_pretty(&self) -> Result<Vec<u8>, PerfEvidenceAssemblyError> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Parse strict bytes and re-verify every nested receipt, raw summary,
    /// applicability projection, set difference, and content seal.
    ///
    /// # Errors
    ///
    /// Returns a typed schema, hash, provenance, or structural error.
    pub fn from_verified_slice(contents: &[u8]) -> Result<Self, PerfEvidenceAssemblyError> {
        Self::from_verified_slice_against_qg1_authorities(contents, &[])
    }

    /// Parse strict assembly bytes against authority slices keyed by each
    /// source shard's exact bound-evidence SHA-256.
    ///
    /// A source may occur at most once in `source_authorities`; an unknown or
    /// duplicate key is rejected before replay. This identity key is the
    /// retained shard input itself, never a role label or positional index.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::from_verified_slice`], including
    /// fail-closed replay of authority-bearing QG-1 sources without their
    /// exact retained authority slice.
    pub fn from_verified_slice_against_qg1_authorities(
        contents: &[u8],
        source_authorities: &[(&str, &[&Qg1ExpectedAuthority])],
    ) -> Result<Self, PerfEvidenceAssemblyError> {
        let probe =
            crate::machine_class_registry::parse_strict_json(contents).map_err(|error| {
                PerfEvidenceAssemblyError::Malformed {
                    reason: format!("assembly is not strict JSON: {error}"),
                }
            })?;
        let found_schema = probe
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| PerfEvidenceAssemblyError::Malformed {
                reason: "assembly has no string schema_version".to_owned(),
            })?;
        if found_schema != PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION {
            return Err(PerfEvidenceAssemblyError::SchemaMismatch {
                found: found_schema.to_owned(),
            });
        }
        let artifact: Self = serde_json::from_value(probe.clone()).map_err(|error| {
            PerfEvidenceAssemblyError::Malformed {
                reason: format!("assembly does not decode as the current schema: {error}"),
            }
        })?;
        if probe != serde_json::to_value(&artifact)? {
            return Err(PerfEvidenceAssemblyError::Malformed {
                reason: "assembly contains unknown fields or a noncanonical persisted shape"
                    .to_owned(),
            });
        }
        if contents != artifact.to_json_pretty()?.as_slice() {
            return Err(PerfEvidenceAssemblyError::Malformed {
                reason: "assembly bytes are not the canonical exact encoding".to_owned(),
            });
        }
        artifact.verify_integrity_against_qg1_authorities(source_authorities)?;
        Ok(artifact)
    }

    /// Load and verify one exact assembly file.
    ///
    /// # Errors
    ///
    /// Returns a typed I/O or verification error.
    pub fn load_verified(path: &Path) -> Result<Self, PerfEvidenceAssemblyError> {
        Self::load_verified_against_qg1_authorities(path, &[])
    }

    /// Load one exact assembly file against source-authority slices keyed by
    /// each retained bound-evidence file SHA-256.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::load_verified`].
    pub fn load_verified_against_qg1_authorities(
        path: &Path,
        source_authorities: &[(&str, &[&Qg1ExpectedAuthority])],
    ) -> Result<Self, PerfEvidenceAssemblyError> {
        Self::from_verified_slice_against_qg1_authorities(&fs::read(path)?, source_authorities)
    }

    /// Verify this in-memory assembly and all retained source artifacts.
    ///
    /// # Errors
    ///
    /// Returns a typed error for schema drift, a stale seal, changed source
    /// evidence, or any derived-field mismatch.
    pub fn verify_integrity(&self) -> Result<(), PerfEvidenceAssemblyError> {
        self.verify_integrity_against_qg1_authorities(&[])
    }

    /// Verify this assembly and every source artifact against authority slices
    /// keyed by each source's exact retained bound-evidence file SHA-256.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::verify_integrity`], including the
    /// fail-closed rejection of an authority-bearing QG-1 source whose slice
    /// is absent, foreign, or duplicated.
    pub fn verify_integrity_against_qg1_authorities(
        &self,
        source_authorities: &[(&str, &[&Qg1ExpectedAuthority])],
    ) -> Result<(), PerfEvidenceAssemblyError> {
        if self.schema_version != PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION
            || self.gate != PerfGate::Qg1
        {
            return Err(PerfEvidenceAssemblyError::SchemaMismatch {
                found: self.schema_version.clone(),
            });
        }
        if self.recomputed_sha256()? != self.assembly_sha256 {
            return Err(PerfEvidenceAssemblyError::HashMismatch);
        }
        if self.matrix_manifest.applicability_plan != self.applicability_plan {
            return Err(PerfEvidenceAssemblyError::MatrixManifestMismatch {
                reason: "matrix manifest names another applicability plan".to_owned(),
            });
        }
        self.matrix_manifest.verify()?;
        let mut sorted_sources = self.source_shards.clone();
        let contract = PlanContract::reconstruct(&self.applicability_plan)?;
        sorted_sources.sort_by(|left, right| {
            source_sort_key(left, &contract).cmp(&source_sort_key(right, &contract))
        });
        let mut sorted_failed = self.failed_shards.clone();
        sorted_failed.sort_by(|left, right| {
            (
                left.process.receipt.run_id(),
                left.process.process_receipt_sha256.as_str(),
            )
                .cmp(&(
                    right.process.receipt.run_id(),
                    right.process.process_receipt_sha256.as_str(),
                ))
        });
        if sorted_sources != self.source_shards || sorted_failed != self.failed_shards {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "source and failed shards are not in canonical order".to_owned(),
            });
        }
        let derived = derive_assembly(
            &self.applicability_plan,
            &self.run_window,
            &self.source_shards,
            &self.failed_shards,
            source_authorities,
        )?;
        if derived.completeness != self.completeness
            || derived.readiness != self.readiness
            || derived.retry_predicate != self.retry_predicate
            || derived.diagnostic_retry_predicate != self.diagnostic_retry_predicate
            || derived.adjudication_retry_predicate != self.adjudication_retry_predicate
            || derived.non_adjudicable_cells != self.non_adjudicable_cells
            || derived.non_adjudicable_sources != self.non_adjudicable_sources
            || derived.compatibility != self.compatibility
            || derived.counts != self.counts
            || derived.missing_required_cell_ids != self.missing_required_cell_ids
            || derived.missing_diagnostic_cell_ids != self.missing_diagnostic_cell_ids
            || derived.cell_sources != self.cell_sources
        {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "assembly fields do not recompute from retained sources".to_owned(),
            });
        }
        let semantic_cell_set = semantic_cell_set_seal(
            &self.applicability_plan,
            &self.matrix_manifest,
            &self.source_shards,
            &self.cell_sources,
            &self.non_adjudicable_sources,
        )?;
        if semantic_cell_set != self.semantic_cell_set {
            return Err(PerfEvidenceAssemblyError::SemanticCellSetMismatch);
        }
        Ok(())
    }

    /// Content-addressed, profile-qualified basename. It intentionally never
    /// uses a `.latest.json` destination.
    ///
    /// # Errors
    ///
    /// Returns a machine-profile error if the frozen destination contract no
    /// longer accepts this profile.
    pub fn destination_basename(&self) -> Result<String, PerfEvidenceAssemblyError> {
        let latest = self
            .applicability_plan
            .profile
            .latest_basename(PerfGate::Qg1.label())?;
        let stem = latest.strip_suffix(".latest.json").ok_or_else(|| {
            PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!("machine-profile destination {latest:?} lacks .latest.json"),
            }
        })?;
        Ok(format!("{stem}.assembly.{}.json", self.assembly_sha256))
    }

    /// Atomically persist one content-addressed assembly without creating or
    /// advancing a latest pointer.
    ///
    /// # Errors
    ///
    /// Returns a verification, serialization, or filesystem error.
    pub fn write_atomic(&self, output_dir: &Path) -> Result<PathBuf, PerfEvidenceAssemblyError> {
        use rustix::fs::{FlockOperation, Mode, OFlags, RenameFlags, flock, openat, renameat_with};

        self.verify_integrity()?;
        let canonical = self.to_json_pretty()?;
        let destination_name = OsString::from(self.destination_basename()?);
        let destination = output_dir.join(&destination_name);
        let directory = prepare_assembly_output_directory(output_dir)?;
        let directory_identity = verify_pinned_assembly_directory(output_dir, &directory, true)?;
        flock(&directory, FlockOperation::LockExclusive).map_err(std::io::Error::from)?;
        if verify_pinned_assembly_directory(output_dir, &directory, true)? != directory_identity {
            return Err(unsafe_assembly_path(
                "assembly output directory changed identity after locking",
            ));
        }
        if let Some((existing_bytes, existing_identity)) =
            read_owned_regular_at(&directory, &destination_name, canonical.len())?
        {
            let existing = Self::from_verified_slice(&existing_bytes)?;
            verify_named_assembly_identity(
                &directory,
                &destination_name,
                existing_identity,
                canonical.len(),
            )?;
            verify_pinned_assembly_directory(output_dir, &directory, true)?;
            if existing == *self && existing_bytes == canonical {
                return Ok(destination);
            }
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "content-addressed assembly destination exists with different bytes"
                    .to_owned(),
            });
        }

        let mut pending_name = OsString::from(".");
        pending_name.push(&destination_name);
        pending_name.push(".pending");
        let temporary = openat(
            &directory,
            &pending_name,
            OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::RUSR | Mode::WUSR,
        )
        .map_err(std::io::Error::from)?;
        let mut temporary = File::from(temporary);
        let temporary_identity =
            checked_owned_regular_identity(&temporary, canonical.len(), false)?;
        let staged_size = temporary.metadata()?.len();
        temporary.seek(SeekFrom::Start(0))?;
        let capacity = usize::try_from(staged_size).map_err(|_| {
            PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "staged assembly length cannot fit in memory".to_owned(),
            }
        })?;
        let mut existing = Vec::new();
        existing.try_reserve_exact(capacity).map_err(|error| {
            PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!("unable to reserve staged-assembly read: {error}"),
            }
        })?;
        (&mut temporary)
            .take(
                u64::try_from(canonical.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
            )
            .read_to_end(&mut existing)?;
        if existing.len() > canonical.len() || !canonical.starts_with(&existing) {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "staged assembly is not a prefix of the canonical bytes".to_owned(),
            });
        }
        temporary.seek(SeekFrom::End(0))?;
        temporary.write_all(&canonical[existing.len()..])?;
        temporary.sync_all()?;
        if checked_owned_regular_identity(&temporary, canonical.len(), true)? != temporary_identity
        {
            return Err(unsafe_assembly_path(
                "staged assembly inode changed before atomic publication",
            ));
        }
        verify_named_assembly_identity(
            &directory,
            &pending_name,
            temporary_identity,
            canonical.len(),
        )?;
        verify_pinned_assembly_directory(output_dir, &directory, true)?;
        renameat_with(
            &directory,
            &pending_name,
            &directory,
            &destination_name,
            RenameFlags::NOREPLACE,
        )
        .map_err(std::io::Error::from)?;
        verify_named_assembly_identity(
            &directory,
            &destination_name,
            temporary_identity,
            canonical.len(),
        )?;
        let (persisted_bytes, persisted_identity) =
            read_owned_regular_at(&directory, &destination_name, canonical.len())?.ok_or_else(
                || unsafe_assembly_path("published assembly disappeared before exact reload"),
            )?;
        if persisted_identity != temporary_identity
            || persisted_bytes != canonical
            || Self::from_verified_slice(&persisted_bytes)? != *self
        {
            return Err(unsafe_assembly_path(
                "published assembly differs from the exact verified staged inode",
            ));
        }
        directory.sync_all()?;
        verify_named_assembly_identity(
            &directory,
            &destination_name,
            temporary_identity,
            canonical.len(),
        )?;
        if verify_pinned_assembly_directory(output_dir, &directory, true)? != directory_identity {
            return Err(unsafe_assembly_path(
                "assembly output directory changed identity before publication returned",
            ));
        }
        Ok(destination)
    }

    fn recomputed_sha256(&self) -> Result<String, PerfEvidenceAssemblyError> {
        let mut unsealed = self.clone();
        unsealed.assembly_sha256.clear();
        hash_serialized(ASSEMBLY_HASH_DOMAIN, &unsealed)
    }
}

#[derive(Debug)]
struct PlanCellContract {
    cell_id: String,
    cell_contract_sha256: String,
    spec: PerfCellSpec,
    applicability: PerfCellApplicability,
    applicability_reason: PerfCellApplicabilityReason,
    configured_threads: usize,
    role: Option<EvidenceRole>,
}

#[derive(Debug)]
struct PlanContract {
    plan: PerfApplicabilityPlan,
    cells: Vec<PlanCellContract>,
    ordinals: BTreeMap<String, usize>,
}

impl PlanContract {
    fn reconstruct(
        binding: &PerfApplicabilityPlanBinding,
    ) -> Result<Self, PerfEvidenceAssemblyError> {
        if binding.gate != PerfGate::Qg1 {
            return Err(PerfEvidenceAssemblyError::UnsupportedGate {
                found: binding.gate,
            });
        }
        let registry = MachineClassRegistry::frozen()?;
        let matrix = PerfMatrixSpec::complete();
        let plan = matrix.applicability_plan(&registry, binding.profile, PerfGate::Qg1)?;
        if plan.binding != *binding {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "applicability_plan",
                detail: "stored plan binding differs from the frozen registry and matrix"
                    .to_owned(),
            });
        }
        let canonical = matrix.for_gate(PerfGate::Qg1);
        if canonical.len() != plan.cells.len() {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "QG-1 matrix and plan lengths differ".to_owned(),
            });
        }
        let mut cells = Vec::with_capacity(canonical.len());
        let mut ordinals = BTreeMap::new();
        for (ordinal, (cell, classification)) in canonical.iter().zip(&plan.cells).enumerate() {
            if classification.ordinal != ordinal {
                return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                    reason: "QG-1 plan ordinals are not canonical".to_owned(),
                });
            }
            let cell_id = format!("{}/{}/{}", PerfGate::Qg1, cell.fixture, cell.metric);
            let role = match classification.applicability {
                PerfCellApplicability::Required => Some(EvidenceRole::Required),
                PerfCellApplicability::Diagnostic => Some(EvidenceRole::Diagnostic),
                PerfCellApplicability::NotApplicable => None,
            };
            if ordinals.insert(cell_id.clone(), ordinal).is_some() {
                return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                    reason: format!("canonical QG-1 matrix repeats cell {cell_id}"),
                });
            }
            cells.push(PlanCellContract {
                cell_id,
                cell_contract_sha256: classification.cell_contract_sha256.clone(),
                spec: (*cell).clone(),
                applicability: classification.applicability,
                applicability_reason: classification.reason,
                configured_threads: classification.configured_threads,
                role,
            });
        }
        Ok(Self {
            plan,
            cells,
            ordinals,
        })
    }

    fn runnable_ordinal(&self, cell_id: &str) -> Option<usize> {
        let ordinal = *self.ordinals.get(cell_id)?;
        self.cells[ordinal]
            .applicability
            .is_runnable()
            .then_some(ordinal)
    }
}

#[derive(Debug)]
struct AssemblyDerived {
    completeness: PerfEvidenceAssemblyCompleteness,
    readiness: PerfEvidenceAssemblyReadiness,
    retry_predicate: Option<String>,
    diagnostic_retry_predicate: Option<String>,
    adjudication_retry_predicate: Option<String>,
    non_adjudicable_cells: Vec<PerfEvidenceAssemblyNoClaimCell>,
    non_adjudicable_sources: Vec<PerfEvidenceAssemblyNoClaimSource>,
    compatibility: Option<PerfEvidenceAssemblyCompatibility>,
    counts: PerfEvidenceAssemblyCounts,
    missing_required_cell_ids: Vec<String>,
    missing_diagnostic_cell_ids: Vec<String>,
    cell_sources: Vec<PerfEvidenceCellSource>,
}

fn source_from_completed_inputs(
    process: PerfAssemblyProcessReceipt,
    completed: PerfAssemblyCompletedInputs,
    contract: &PlanContract,
    external_qg1_authorities: &[&Qg1ExpectedAuthority],
) -> Result<PerfEvidenceAssemblySource, PerfEvidenceAssemblyError> {
    process.verify()?;
    verify_completed_inputs(&process, &completed, external_qg1_authorities)?;
    let PerfAssemblyCompletedInputs {
        threshold_artifact_sha256,
        threshold_artifact,
        prebinding_evidence_file_sha256,
        prebinding_artifact,
        bound_evidence_file_sha256,
        evidence_content_seal_sha256,
        artifact,
        runner_receipt_sha256,
        runner_artifact_manifest_sha256,
        run_id,
        cell_ids: retained_cell_ids,
    } = completed;
    artifact.verify_integrity_against_qg1_authorities(external_qg1_authorities)?;
    let identity = artifact.machine_class.identity().ok_or_else(|| {
        PerfEvidenceAssemblyError::IncompatibleShard {
            field: "runner_receipt",
            detail: format!(
                "source run {} has no verified runner identity",
                artifact.provenance.run_id
            ),
        }
    })?;
    let manifest = identity.artifact_manifest().ok_or_else(|| {
        PerfEvidenceAssemblyError::IncompatibleShard {
            field: "artifact_manifest",
            detail: format!(
                "source run {} has no exact artifact manifest",
                artifact.provenance.run_id
            ),
        }
    })?;
    let mut ordinals = BTreeSet::new();
    for cell in &artifact.cells {
        let ordinal = contract.runnable_ordinal(&cell.cell_id).ok_or_else(|| {
            PerfEvidenceAssemblyError::UnexpectedCell {
                cell_id: cell.cell_id.clone(),
            }
        })?;
        if !ordinals.insert(ordinal) {
            return Err(PerfEvidenceAssemblyError::OverlappingCell {
                cell_id: cell.cell_id.clone(),
            });
        }
    }
    let cell_ids = ordinals
        .into_iter()
        .map(|ordinal| contract.cells[ordinal].cell_id.clone())
        .collect::<Vec<_>>();
    if cell_ids != retained_cell_ids
        || runner_receipt_sha256 != identity.receipt_sha256()
        || runner_artifact_manifest_sha256 != manifest.manifest_sha256()
        || run_id != artifact.provenance.run_id
    {
        return Err(PerfEvidenceAssemblyError::InvalidAttemptBundle {
            reason: "completed source wrapper differs from its exact nested identities".to_owned(),
        });
    }
    Ok(PerfEvidenceAssemblySource {
        process,
        threshold_artifact_sha256,
        threshold_artifact,
        prebinding_evidence_file_sha256,
        prebinding_artifact,
        bound_evidence_file_sha256,
        evidence_content_seal_sha256,
        runner_receipt_sha256,
        runner_artifact_manifest_sha256,
        run_id,
        cell_ids,
        artifact,
    })
}

fn reconstruct_source(
    source: &PerfEvidenceAssemblySource,
    contract: &PlanContract,
    external_qg1_authorities: &[&Qg1ExpectedAuthority],
) -> Result<PerfEvidenceAssemblySource, PerfEvidenceAssemblyError> {
    let completed = PerfAssemblyCompletedInputs {
        threshold_artifact_sha256: source.threshold_artifact_sha256.clone(),
        threshold_artifact: source.threshold_artifact.clone(),
        prebinding_evidence_file_sha256: source.prebinding_evidence_file_sha256.clone(),
        prebinding_artifact: source.prebinding_artifact.clone(),
        bound_evidence_file_sha256: source.bound_evidence_file_sha256.clone(),
        evidence_content_seal_sha256: source.evidence_content_seal_sha256.clone(),
        artifact: source.artifact.clone(),
        runner_receipt_sha256: source.runner_receipt_sha256.clone(),
        runner_artifact_manifest_sha256: source.runner_artifact_manifest_sha256.clone(),
        run_id: source.run_id.clone(),
        cell_ids: source.cell_ids.clone(),
    };
    source_from_completed_inputs(
        source.process.clone(),
        completed,
        contract,
        external_qg1_authorities,
    )
}

fn source_sort_key<'a>(
    source: &'a PerfEvidenceAssemblySource,
    contract: &PlanContract,
) -> (usize, &'a str) {
    let first = source
        .cell_ids
        .iter()
        .filter_map(|cell_id| contract.ordinals.get(cell_id).copied())
        .min()
        .unwrap_or(usize::MAX);
    (first, source.bound_evidence_file_sha256.as_str())
}

fn validate_source_authority_sets(
    sources: &[PerfEvidenceAssemblySource],
    source_authorities: &[(&str, &[&Qg1ExpectedAuthority])],
) -> Result<(), PerfEvidenceAssemblyError> {
    let known_sources = sources
        .iter()
        .map(|source| source.bound_evidence_file_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let mut supplied_sources = BTreeSet::new();
    for (source_sha256, _) in source_authorities {
        if !known_sources.contains(source_sha256) {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!(
                    "authority set names unknown bound-evidence source {source_sha256:?}"
                ),
            });
        }
        if !supplied_sources.insert(*source_sha256) {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!("authority sets repeat bound-evidence source {source_sha256:?}"),
            });
        }
    }
    Ok(())
}

fn authorities_for_source<'a>(
    source: &PerfEvidenceAssemblySource,
    source_authorities: &[(&str, &'a [&'a Qg1ExpectedAuthority])],
) -> &'a [&'a Qg1ExpectedAuthority] {
    source_authorities
        .iter()
        .find_map(|(source_sha256, authorities)| {
            (*source_sha256 == source.bound_evidence_file_sha256).then_some(*authorities)
        })
        .unwrap_or(&[])
}

fn derive_assembly(
    applicability_plan: &PerfApplicabilityPlanBinding,
    run_window: &str,
    sources: &[PerfEvidenceAssemblySource],
    failed: &[PerfEvidenceAssemblyFailedAttempt],
    source_authorities: &[(&str, &[&Qg1ExpectedAuthority])],
) -> Result<AssemblyDerived, PerfEvidenceAssemblyError> {
    let total = sources
        .len()
        .checked_add(failed.len())
        .ok_or(PerfEvidenceAssemblyError::TooManyShards)?;
    if total == 0 {
        return Err(PerfEvidenceAssemblyError::EmptyAssembly);
    }
    if total > PERF_ASSEMBLY_MAX_SHARDS {
        return Err(PerfEvidenceAssemblyError::TooManyShards);
    }
    if run_window.trim().is_empty() {
        return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
            reason: "assembly requires a non-empty run window".to_owned(),
        });
    }
    validate_source_authority_sets(sources, source_authorities)?;
    let contract = PlanContract::reconstruct(applicability_plan)?;
    let mut run_ids = BTreeSet::new();
    let mut measured = BTreeMap::<usize, PerfEvidenceCellSource>::new();
    let mut non_adjudicable_sources = Vec::new();
    let mut compatibility = None;
    let mut reference_identity: Option<&VerifiedRunnerIdentity> = None;
    let runnable_count = contract
        .cells
        .iter()
        .filter(|cell| cell.applicability.is_runnable())
        .count();

    for source in sources {
        let reconstructed = reconstruct_source(
            source,
            &contract,
            authorities_for_source(source, source_authorities),
        )?;
        if reconstructed != *source {
            return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!("source wrapper for run {} is stale", source.run_id),
            });
        }
        let artifact = &source.artifact;
        if artifact.gate != PerfGate::Qg1 {
            return Err(PerfEvidenceAssemblyError::UnsupportedGate {
                found: artifact.gate,
            });
        }
        if artifact.applicability_plan != *applicability_plan {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "applicability_plan",
                detail: format!("source run {} names another plan", source.run_id),
            });
        }
        if artifact.provenance.manifest_sha256 != applicability_plan.normalized_perf_manifest_sha256
        {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "manifest_sha256",
                detail: format!(
                    "source run {} does not bind the plan's normalized normative manifest",
                    source.run_id
                ),
            });
        }
        if artifact.provenance.run_window != run_window {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "run_window",
                detail: format!("source run {} names another window", source.run_id),
            });
        }
        if artifact.gate_decision.is_some() {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "gate_decision",
                detail: format!(
                    "source run {} was already adjudicated and is not an immutable shard input",
                    source.run_id
                ),
            });
        }
        let partial_source = source.cell_ids.len() != runnable_count;
        if partial_source && artifact.admission_no_claim.is_none() {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "partial_no_claim",
                detail: format!(
                    "partial source run {} lacks an explicit admission NoClaim reason",
                    source.run_id
                ),
            });
        }
        let exact_partial_no_claim = partial_source
            && source.process.receipt.fixture_selector().is_some()
            && !source.cell_ids.is_empty()
            && source.cell_ids.len() < runnable_count
            && source.cell_ids == source.process.receipt.selected_cell_ids()
            && artifact.cells.iter().all(EvidenceCell::claim_eligible)
            && artifact.admission_no_claim.as_ref().is_some_and(|reason| {
                reason.code == PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE
                    && reason.message == PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_DETAIL
                    && reason.severity == EvidenceSeverity::NoClaim
            });
        if let Some(reason) = artifact.admission_no_claim.as_ref()
            && !exact_partial_no_claim
        {
            non_adjudicable_sources.push(PerfEvidenceAssemblyNoClaimSource {
                evidence_artifact_sha256: source.bound_evidence_file_sha256.clone(),
                run_id: source.run_id.clone(),
                cell_ids: source.cell_ids.clone(),
                reason: reason.clone(),
            });
        }
        let reason = EvidenceReason::new(
            PERF_ASSEMBLY_ENGINE_LIFECYCLE_NO_CLAIM_CODE,
            "H2 v5 proves the outer direct child only; engine actual-work, queue, worker-join, feed-drain, and pending-zero observations are unavailable",
            EvidenceSeverity::NoClaim,
        );
        non_adjudicable_sources.push(PerfEvidenceAssemblyNoClaimSource {
            evidence_artifact_sha256: source.bound_evidence_file_sha256.clone(),
            run_id: source.run_id.clone(),
            cell_ids: source.cell_ids.clone(),
            reason,
        });
        if !source
            .process
            .receipt
            .process_lifecycle()
            .descendant_process_tree_quiescence_is_proven()
        {
            non_adjudicable_sources.push(PerfEvidenceAssemblyNoClaimSource {
                evidence_artifact_sha256: source.bound_evidence_file_sha256.clone(),
                run_id: source.run_id.clone(),
                cell_ids: source.cell_ids.clone(),
                reason: EvidenceReason::new(
                    PERF_ASSEMBLY_PROCESS_TREE_NO_CLAIM_CODE,
                    "H2 v5 proves only a direct child; descendant/process-group quiescence and inherited-handle closure remain unproven",
                    EvidenceSeverity::NoClaim,
                ),
            });
        }
        if !run_ids.insert(source.run_id.as_str()) {
            return Err(PerfEvidenceAssemblyError::DuplicateRunId {
                run_id: source.run_id.clone(),
            });
        }

        let identity = artifact.machine_class.identity().ok_or_else(|| {
            PerfEvidenceAssemblyError::IncompatibleShard {
                field: "runner_receipt",
                detail: format!("source run {} is not registry-admitted", source.run_id),
            }
        })?;
        if let Some(reference) = reference_identity {
            if !reference.same_execution_identity(identity) {
                return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                    field: "machine_execution_identity",
                    detail: format!("source run {} names another execution", source.run_id),
                });
            }
        } else {
            reference_identity = Some(identity);
        }

        validate_selected_widths(artifact, &contract)?;
        let shard_compatibility = compatibility_from_artifact(artifact)?;
        if let Some(expected) = &compatibility {
            if expected != &shard_compatibility {
                return Err(first_compatibility_difference(
                    expected,
                    &shard_compatibility,
                    &source.run_id,
                ));
            }
        } else {
            compatibility = Some(shard_compatibility);
        }

        for cell in &artifact.cells {
            validate_cell_provenance(cell, artifact)?;
            let ordinal = contract.runnable_ordinal(&cell.cell_id).ok_or_else(|| {
                PerfEvidenceAssemblyError::UnexpectedCell {
                    cell_id: cell.cell_id.clone(),
                }
            })?;
            let expected_role = contract.cells[ordinal].role.ok_or_else(|| {
                PerfEvidenceAssemblyError::InconsistentAssembly {
                    reason: format!(
                        "runnable cell {} has no Required/Diagnostic evidence role",
                        cell.cell_id
                    ),
                }
            })?;
            if cell.spec.role != expected_role {
                return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                    field: "evidence_role",
                    detail: format!(
                        "cell {} is {:?}; plan requires {:?}",
                        cell.cell_id, cell.spec.role, expected_role
                    ),
                });
            }
            let cell_source = PerfEvidenceCellSource {
                ordinal,
                cell_id: cell.cell_id.clone(),
                role: expected_role,
                terminal_status: cell.status,
                evidence_artifact_sha256: source.bound_evidence_file_sha256.clone(),
                runner_receipt_sha256: source.runner_receipt_sha256.clone(),
                runner_artifact_manifest_sha256: source.runner_artifact_manifest_sha256.clone(),
                run_id: source.run_id.clone(),
            };
            if measured.insert(ordinal, cell_source).is_some() {
                return Err(PerfEvidenceAssemblyError::OverlappingCell {
                    cell_id: cell.cell_id.clone(),
                });
            }
        }
    }

    for attempt in failed {
        attempt.verify()?;
        let receipt = &attempt.process.receipt;
        if receipt.applicability_plan() != applicability_plan {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "failed_applicability_plan",
                detail: format!("failed run {} names another plan", receipt.run_id()),
            });
        }
        if receipt.run_window() != run_window {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "failed_run_window",
                detail: format!("failed run {} names another window", receipt.run_id()),
            });
        }
        if !run_ids.insert(receipt.run_id()) {
            return Err(PerfEvidenceAssemblyError::DuplicateRunId {
                run_id: receipt.run_id().to_owned(),
            });
        }
    }

    let missing_required_cell_ids = contract
        .cells
        .iter()
        .enumerate()
        .filter(|(ordinal, cell)| {
            cell.applicability == PerfCellApplicability::Required && !measured.contains_key(ordinal)
        })
        .map(|(_, cell)| cell.cell_id.clone())
        .collect::<Vec<_>>();
    let missing_diagnostic_cell_ids = contract
        .cells
        .iter()
        .enumerate()
        .filter(|(ordinal, cell)| {
            cell.applicability == PerfCellApplicability::Diagnostic
                && !measured.contains_key(ordinal)
        })
        .map(|(_, cell)| cell.cell_id.clone())
        .collect::<Vec<_>>();
    let completeness = if missing_required_cell_ids.is_empty() {
        PerfEvidenceAssemblyCompleteness::Complete
    } else {
        PerfEvidenceAssemblyCompleteness::Incomplete
    };
    let retry_predicate = (!missing_required_cell_ids.is_empty()).then(|| {
        format!(
            "rerun the {} cells listed in missing_required_cell_ids for {}/{} and reassemble",
            missing_required_cell_ids.len(),
            applicability_plan.profile.hardware_class_id().as_str(),
            applicability_plan.profile.execution_profile_id().as_str(),
        )
    });
    if retry_predicate
        .as_deref()
        .is_some_and(|predicate| predicate.len() > PERF_ASSEMBLY_MAX_RETRY_PREDICATE_BYTES)
    {
        return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
            reason: "derived retry predicate exceeds the bounded log contract".to_owned(),
        });
    }
    let diagnostic_retry_predicate = (!missing_diagnostic_cell_ids.is_empty()).then(|| {
        format!(
            "rerun the {} cells listed in missing_diagnostic_cell_ids for {}/{} to obtain full-plan coverage",
            missing_diagnostic_cell_ids.len(),
            applicability_plan.profile.hardware_class_id().as_str(),
            applicability_plan.profile.execution_profile_id().as_str(),
        )
    });
    if diagnostic_retry_predicate
        .as_deref()
        .is_some_and(|predicate| predicate.len() > PERF_ASSEMBLY_MAX_RETRY_PREDICATE_BYTES)
    {
        return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
            reason: "derived Diagnostic retry predicate exceeds the bounded log contract"
                .to_owned(),
        });
    }
    let sources_by_hash = sources
        .iter()
        .map(|source| (source.bound_evidence_file_sha256.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let mut non_adjudicable_cells = Vec::new();
    for cell_source in measured.values() {
        let source = sources_by_hash
            .get(cell_source.evidence_artifact_sha256.as_str())
            .ok_or_else(|| PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!(
                    "cell-source mapping for {:?} has no retained source",
                    cell_source.cell_id
                ),
            })?;
        let cell = source
            .artifact
            .cells
            .iter()
            .find(|cell| cell.cell_id == cell_source.cell_id)
            .ok_or_else(|| PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!(
                    "cell-source mapping for {:?} has no retained evidence cell",
                    cell_source.cell_id
                ),
            })?;
        if !cell.claim_eligible() {
            non_adjudicable_cells.push(PerfEvidenceAssemblyNoClaimCell {
                ordinal: cell_source.ordinal,
                cell_id: cell_source.cell_id.clone(),
                role: cell_source.role,
                terminal_status: cell.status,
                reasons: cell.reasons.clone(),
            });
        }
    }
    let required_cells = contract.plan.cell_count(PerfCellApplicability::Required);
    let required_non_adjudicable_cells = non_adjudicable_cells
        .iter()
        .filter(|diagnostic| diagnostic.role == EvidenceRole::Required)
        .count();
    let required_non_adjudicable_sources = non_adjudicable_sources
        .iter()
        .filter(|source| {
            source.cell_ids.iter().any(|cell_id| {
                contract.ordinals.get(cell_id).is_some_and(|ordinal| {
                    contract.cells[*ordinal].applicability == PerfCellApplicability::Required
                })
            })
        })
        .count();
    let readiness = derive_readiness(
        required_cells,
        missing_required_cell_ids.len(),
        missing_diagnostic_cell_ids.len(),
        required_non_adjudicable_cells,
        required_non_adjudicable_sources,
    );
    let adjudication_retry_predicate = match readiness {
        PerfEvidenceAssemblyReadiness::NoClaimNoRequiredCells => Some(
            "do not adjudicate this diagnostic-only profile; select a profile with Applicable/Required QG-1 cells"
                .to_owned(),
        ),
        PerfEvidenceAssemblyReadiness::NoClaimInvalidEvidence => Some(format!(
            "rerun {required_non_adjudicable_cells} Required cells and resolve {required_non_adjudicable_sources} source NoClaim records listed in this assembly"
        )),
        PerfEvidenceAssemblyReadiness::ReadyForAdjudication
        | PerfEvidenceAssemblyReadiness::NoClaimIncomplete => None,
    };
    if adjudication_retry_predicate
        .as_deref()
        .is_some_and(|predicate| predicate.len() > PERF_ASSEMBLY_MAX_RETRY_PREDICATE_BYTES)
    {
        return Err(PerfEvidenceAssemblyError::InconsistentAssembly {
            reason: "derived adjudication retry predicate exceeds the bounded log contract"
                .to_owned(),
        });
    }
    let counts = PerfEvidenceAssemblyCounts {
        canonical_cells: contract.cells.len(),
        required_cells,
        diagnostic_cells: contract.plan.cell_count(PerfCellApplicability::Diagnostic),
        not_applicable_cells: contract
            .plan
            .cell_count(PerfCellApplicability::NotApplicable),
        measured_cells: measured.len(),
        completed_shards: sources.len(),
        failed_shards: failed.len(),
    };
    Ok(AssemblyDerived {
        completeness,
        readiness,
        retry_predicate,
        diagnostic_retry_predicate,
        adjudication_retry_predicate,
        non_adjudicable_cells,
        non_adjudicable_sources,
        compatibility,
        counts,
        missing_required_cell_ids,
        missing_diagnostic_cell_ids,
        cell_sources: measured.into_values().collect(),
    })
}

const fn derive_readiness(
    required_cells: usize,
    missing_required_cells: usize,
    missing_diagnostic_cells: usize,
    non_adjudicable_required_cells: usize,
    blocking_source_no_claims: usize,
) -> PerfEvidenceAssemblyReadiness {
    if missing_required_cells != 0 || missing_diagnostic_cells != 0 {
        PerfEvidenceAssemblyReadiness::NoClaimIncomplete
    } else if required_cells == 0 {
        PerfEvidenceAssemblyReadiness::NoClaimNoRequiredCells
    } else if non_adjudicable_required_cells == 0 && blocking_source_no_claims == 0 {
        PerfEvidenceAssemblyReadiness::ReadyForAdjudication
    } else {
        PerfEvidenceAssemblyReadiness::NoClaimInvalidEvidence
    }
}

#[derive(Serialize)]
struct SemanticCellSetPreimage {
    schema_version: &'static str,
    applicability_plan: PerfApplicabilityPlanBinding,
    matrix_manifest_sha256: String,
    policy: Option<EvidencePolicy>,
    cells: Vec<SemanticCellProjection>,
    source_no_claims: Vec<SemanticSourceNoClaim>,
}

#[derive(Serialize)]
struct SemanticCellProjection {
    ordinal: usize,
    provenance: SemanticEvidenceProvenance,
    runner_identity: SemanticRunnerIdentity,
    cell: serde_json::Value,
}

#[derive(Serialize)]
struct SemanticSourceNoClaim {
    cell_ids: Vec<String>,
    reason: EvidenceReason,
}

/// Run-label-independent producer facts. The envelope still retains and seals
/// the original `run_id`; only this semantic comparison projection omits it.
#[derive(Serialize)]
struct SemanticEvidenceProvenance {
    manifest_sha256: String,
    build: SemanticBuildIdentity,
    machine: PerfAssemblyMachineIdentity,
    corpus: PerfAssemblyCorpusIdentity,
}

/// Build facts that remain invariant when the same canonical cells are
/// partitioned across several typed producer invocations. Exact argv and
/// environment seals remain in the retained source artifacts and the outer
/// assembly seal; they are deliberately absent from this semantic projection
/// because the typed fixture selector is itself a partitioning choice.
#[derive(Serialize)]
struct SemanticBuildIdentity {
    executable_sha256: String,
    git_revision: String,
    git_dirty: bool,
    worktree_state_sha256: Option<String>,
    cargo_lock_sha256: Option<String>,
    rustc_version: String,
    target_triple: String,
    build_profile: String,
    cargo_features: Vec<String>,
}

/// Exact admitted runner facts that affect measurement meaning, excluding the
/// run-local completion hashes/timestamps and manifest artifact hashes because
/// those seals transitively include the deliberately omitted source run ID.
#[derive(Serialize)]
struct SemanticRunnerIdentity {
    profile: MachineProfileKey,
    capacity_semantics: ExecutionCapacitySemantics,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    canonicalization: MachineClassCanonicalizationBinding,
    hardware: serde_json::Value,
    execution_request: serde_json::Value,
    execution_start: serde_json::Value,
    execution_end: serde_json::Value,
    build: serde_json::Value,
    durability: serde_json::Value,
    completion: serde_json::Value,
    artifact_manifest: serde_json::Value,
}

fn semantic_cell_set_seal(
    applicability_plan: &PerfApplicabilityPlanBinding,
    matrix_manifest: &PerfEvidenceAssemblyMatrixManifest,
    sources: &[PerfEvidenceAssemblySource],
    cell_sources: &[PerfEvidenceCellSource],
    non_adjudicable_sources: &[PerfEvidenceAssemblyNoClaimSource],
) -> Result<PerfEvidenceSemanticCellSetSeal, PerfEvidenceAssemblyError> {
    let sources_by_hash = sources
        .iter()
        .map(|source| (source.bound_evidence_file_sha256.as_str(), source))
        .collect::<BTreeMap<_, _>>();
    let mut cells = Vec::with_capacity(cell_sources.len());
    for mapping in cell_sources {
        let source = sources_by_hash
            .get(mapping.evidence_artifact_sha256.as_str())
            .ok_or_else(|| PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!(
                    "semantic projection cannot resolve source for {:?}",
                    mapping.cell_id
                ),
            })?;
        let artifact = &source.artifact;
        let identity = artifact.machine_class.identity().ok_or_else(|| {
            PerfEvidenceAssemblyError::IncompatibleShard {
                field: "runner_receipt",
                detail: format!(
                    "semantic projection source {} has no admitted identity",
                    source.run_id
                ),
            }
        })?;
        let cell = artifact
            .cells
            .iter()
            .find(|cell| cell.cell_id == mapping.cell_id)
            .ok_or_else(|| PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: format!(
                    "semantic projection cannot resolve evidence cell {:?}",
                    mapping.cell_id
                ),
            })?;
        cells.push(SemanticCellProjection {
            ordinal: mapping.ordinal,
            provenance: semantic_provenance(&artifact.provenance),
            runner_identity: semantic_runner_identity(identity)?,
            cell: semantic_cell(cell)?,
        });
    }
    let preimage = SemanticCellSetPreimage {
        schema_version: PERF_EVIDENCE_SEMANTIC_CELL_SET_SCHEMA_VERSION,
        applicability_plan: applicability_plan.clone(),
        matrix_manifest_sha256: matrix_manifest.matrix_manifest_sha256.clone(),
        policy: sources.first().map(|source| source.artifact.policy.clone()),
        cells,
        source_no_claims: normalized_semantic_source_no_claims(non_adjudicable_sources),
    };
    Ok(PerfEvidenceSemanticCellSetSeal {
        schema_version: PERF_EVIDENCE_SEMANTIC_CELL_SET_SCHEMA_VERSION.to_owned(),
        semantic_cell_set_sha256: hash_serialized(SEMANTIC_CELL_SET_HASH_DOMAIN, &preimage)?,
    })
}

fn normalized_semantic_source_no_claims(
    sources: &[PerfEvidenceAssemblyNoClaimSource],
) -> Vec<SemanticSourceNoClaim> {
    let mut cells_by_reason =
        BTreeMap::<(String, String, crate::EvidenceSeverity), BTreeSet<String>>::new();
    for source in sources {
        cells_by_reason
            .entry((
                source.reason.code.clone(),
                source.reason.message.clone(),
                source.reason.severity,
            ))
            .or_default()
            .extend(source.cell_ids.iter().cloned());
    }
    cells_by_reason
        .into_iter()
        .map(
            |((code, message, severity), cell_ids)| SemanticSourceNoClaim {
                cell_ids: cell_ids.into_iter().collect(),
                reason: EvidenceReason {
                    code,
                    message,
                    severity,
                },
            },
        )
        .collect()
}

fn semantic_provenance(provenance: &EvidenceProvenance) -> SemanticEvidenceProvenance {
    let machine = &provenance.machine;
    SemanticEvidenceProvenance {
        manifest_sha256: provenance.manifest_sha256.clone(),
        build: SemanticBuildIdentity::from(&provenance.build),
        machine: PerfAssemblyMachineIdentity::from_execution(
            &machine.fingerprint,
            &machine.os,
            &machine.arch,
            machine.logical_cpus,
            &machine.execution,
        ),
        corpus: PerfAssemblyCorpusIdentity::from_observation(&provenance.corpus),
    }
}

impl From<&BuildIdentity> for SemanticBuildIdentity {
    fn from(build: &BuildIdentity) -> Self {
        Self {
            executable_sha256: build.executable_sha256.clone(),
            git_revision: build.git_revision.clone(),
            git_dirty: build.git_dirty,
            worktree_state_sha256: build.worktree_state_sha256.clone(),
            cargo_lock_sha256: build.cargo_lock_sha256.clone(),
            rustc_version: build.rustc_version.clone(),
            target_triple: build.target_triple.clone(),
            build_profile: build.build_profile.clone(),
            cargo_features: build.cargo_features.clone(),
        }
    }
}

fn semantic_runner_identity(
    identity: &VerifiedRunnerIdentity,
) -> Result<SemanticRunnerIdentity, PerfEvidenceAssemblyError> {
    let artifact_manifest = identity.artifact_manifest().ok_or_else(|| {
        PerfEvidenceAssemblyError::InconsistentAssembly {
            reason: "verified source identity has no artifact manifest".to_owned(),
        }
    })?;
    Ok(SemanticRunnerIdentity {
        profile: identity.profile(),
        capacity_semantics: identity.capacity_semantics(),
        execution_capacity: identity.execution_capacity(),
        max_exercised_cell_width: identity.max_exercised_cell_width(),
        canonicalization: identity.canonicalization().clone(),
        hardware: identity.hardware().clone(),
        execution_request: identity.execution_request().clone(),
        execution_start: identity.execution_start().clone(),
        execution_end: identity.execution_end().clone(),
        build: semantic_json_object(
            identity.build().clone(),
            &["command_sha256", "environment_sha256"],
        )?,
        durability: identity.durability().clone(),
        completion: semantic_json_object(
            identity.completion().clone(),
            &[
                "run_log_sha256",
                "artifact_manifest_sha256",
                "started_at_utc",
                "finished_at_utc",
            ],
        )?,
        artifact_manifest: semantic_json_object(
            serde_json::to_value(artifact_manifest.manifest())?,
            &[
                "run_id",
                "run_log_sha256",
                "threshold_artifact_sha256",
                "prebinding_evidence_artifact_sha256",
            ],
        )?,
    })
}

fn semantic_json_object(
    mut value: serde_json::Value,
    omitted_fields: &[&str],
) -> Result<serde_json::Value, PerfEvidenceAssemblyError> {
    let object =
        value
            .as_object_mut()
            .ok_or_else(|| PerfEvidenceAssemblyError::InconsistentAssembly {
                reason: "verified runner semantic facts are not JSON objects".to_owned(),
            })?;
    for field in omitted_fields {
        object.remove(*field);
    }
    Ok(value)
}

fn semantic_cell(cell: &EvidenceCell) -> Result<serde_json::Value, PerfEvidenceAssemblyError> {
    let mut semantic = cell.clone();
    if let EvidenceCellBody::Paired {
        paired,
        treatment_arm_null,
        ..
    } = &mut semantic.body
    {
        clear_paired_run_ids(paired);
        if let Some(null) = treatment_arm_null {
            clear_paired_run_ids(null);
        }
    }
    let mut value = serde_json::to_value(semantic)?;
    normalize_qg1_session_commitments(&mut value);
    Ok(value)
}

/// Keep the structural QG-1 authority transcript in the semantic projection
/// while discarding only one-run opaque capability commitments and the
/// receipts derived from them. Those values are independently authenticated
/// before an assembly is admitted; including their random entropy here would
/// make identical measurements compare as different semantic evidence.
fn normalize_qg1_session_commitments(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(authority) = object.get_mut("qg1_lifecycle_authority") {
                normalize_qg1_authority_capabilities(authority);
            }
            if let Some(binding) = object.get_mut("qg1_sample_binding") {
                normalize_qg1_binding_receipts(binding);
            }
            for value in object.values_mut() {
                normalize_qg1_session_commitments(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                normalize_qg1_session_commitments(value);
            }
        }
        _ => {}
    }
}

fn normalize_qg1_authority_capabilities(authority: &mut serde_json::Value) {
    let Some(authority) = authority.as_object_mut() else {
        return;
    };
    authority.insert(
        "authority_sha256".to_owned(),
        serde_json::Value::String(String::new()),
    );
    if let Some(serde_json::Value::Array(rows)) = authority.get_mut("issued_rows") {
        for row in rows {
            if let Some(row) = row.as_object_mut() {
                row.insert(
                    "producer_capability_sha256".to_owned(),
                    serde_json::Value::String(String::new()),
                );
            }
        }
    }
}

fn normalize_qg1_binding_receipts(binding: &mut serde_json::Value) {
    let Some(binding) = binding.as_object_mut() else {
        return;
    };
    for field in [
        "stream_id_sha256",
        "lifecycle_authority_sha256",
        "stream_role_identity_sha256",
        "producer_capability_sha256",
        "producer_capability_tag_sha256",
        "lifecycle_receipt_id_sha256",
        "lifecycle_receipt_sha256",
    ] {
        binding.insert(field.to_owned(), serde_json::Value::String(String::new()));
    }
}

fn clear_paired_run_ids(paired: &mut crate::PairedExperimentResult) {
    paired.provenance.run_id.clear();
    for sample in paired
        .effect_samples
        .iter_mut()
        .chain(&mut paired.null_samples)
    {
        sample.provenance.run_id.clear();
    }
}

fn validate_selected_widths(
    artifact: &PerfEvidenceArtifact,
    contract: &PlanContract,
) -> Result<(), PerfEvidenceAssemblyError> {
    let expected = artifact
        .cells
        .iter()
        .map(|cell| {
            contract
                .runnable_ordinal(&cell.cell_id)
                .map(|ordinal| contract.cells[ordinal].configured_threads)
                .ok_or_else(|| PerfEvidenceAssemblyError::UnexpectedCell {
                    cell_id: cell.cell_id.clone(),
                })
        })
        .collect::<Result<BTreeSet<_>, _>>()?
        .into_iter()
        .collect::<Vec<_>>();
    if artifact
        .provenance
        .machine
        .execution
        .configured_engine_thread_widths
        != expected
    {
        return Err(PerfEvidenceAssemblyError::IncompatibleShard {
            field: "configured_engine_thread_widths",
            detail: format!(
                "source run {} records {:?}; selected canonical cells require {expected:?}",
                artifact.provenance.run_id,
                artifact
                    .provenance
                    .machine
                    .execution
                    .configured_engine_thread_widths
            ),
        });
    }
    Ok(())
}

fn validate_cell_provenance(
    cell: &EvidenceCell,
    artifact: &PerfEvidenceArtifact,
) -> Result<(), PerfEvidenceAssemblyError> {
    let EvidenceCellBody::Paired {
        paired,
        treatment_arm_null,
        ..
    } = &cell.body
    else {
        return Err(PerfEvidenceAssemblyError::IncompatibleShard {
            field: "qg1_estimand",
            detail: format!("QG-1 cell {} is not paired evidence", cell.cell_id),
        });
    };
    let provenance = &paired.provenance;
    let expected = &artifact.provenance;
    let matches = provenance.run_id == expected.run_id
        && provenance.executable_sha256 == expected.build.executable_sha256
        && provenance.corpus_sha256 == expected.corpus.corpus_sha256
        && provenance.worker_id == expected.machine.fingerprint
        && provenance.build_profile == expected.build.build_profile;
    if !matches {
        return Err(PerfEvidenceAssemblyError::IncompatibleShard {
            field: "cell_provenance",
            detail: format!(
                "cell {} does not bind the source run/ELF/corpus/worker/profile identity",
                cell.cell_id
            ),
        });
    }
    if treatment_arm_null
        .as_ref()
        .is_some_and(|null| null.provenance != paired.provenance || null.config != paired.config)
    {
        return Err(PerfEvidenceAssemblyError::IncompatibleShard {
            field: "treatment_arm_null",
            detail: format!("cell {} has a cross-invocation Q/Q null", cell.cell_id),
        });
    }
    Ok(())
}

fn compatibility_from_artifact(
    artifact: &PerfEvidenceArtifact,
) -> Result<PerfEvidenceAssemblyCompatibility, PerfEvidenceAssemblyError> {
    let identity = artifact.machine_class.identity().ok_or_else(|| {
        PerfEvidenceAssemblyError::IncompatibleShard {
            field: "runner_receipt",
            detail: format!(
                "source run {} lacks verified runner identity",
                artifact.provenance.run_id
            ),
        }
    })?;
    let mut estimator = None;
    for cell in &artifact.cells {
        let EvidenceCellBody::Paired {
            paired,
            treatment_arm_null,
            ..
        } = &cell.body
        else {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "estimator",
                detail: format!("QG-1 cell {} is not paired", cell.cell_id),
            });
        };
        paired
            .config
            .validate()
            .map_err(EvidenceArtifactError::from)?;
        if treatment_arm_null
            .as_ref()
            .is_some_and(|null| null.config != paired.config)
        {
            return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "estimator",
                detail: format!("cell {} mixes estimator contracts", cell.cell_id),
            });
        }
        match &estimator {
            Some(expected) if expected != &paired.config => {
                return Err(PerfEvidenceAssemblyError::IncompatibleShard {
                    field: "estimator",
                    detail: format!(
                        "source run {} mixes estimator contracts",
                        artifact.provenance.run_id
                    ),
                });
            }
            None => estimator = Some(paired.config.clone()),
            Some(_) => {}
        }
    }
    let machine = &artifact.provenance.machine;
    Ok(PerfEvidenceAssemblyCompatibility {
        profile: identity.profile(),
        capacity_semantics: identity.capacity_semantics(),
        execution_capacity: identity.execution_capacity(),
        max_exercised_cell_width: identity.max_exercised_cell_width(),
        canonicalization: identity.canonicalization().clone(),
        runner_hardware_sha256: identity.derived_sha256().hardware.clone(),
        runner_execution_identity_sha256: identity.derived_sha256().identity.clone(),
        runner_durability_sha256: sha256_hex(&serde_json::to_vec(identity.durability())?),
        manifest_sha256: artifact.provenance.manifest_sha256.clone(),
        run_window: artifact.provenance.run_window.clone(),
        build: artifact.provenance.build.clone(),
        machine: PerfAssemblyMachineIdentity::from_execution(
            &machine.fingerprint,
            &machine.os,
            &machine.arch,
            machine.logical_cpus,
            &machine.execution,
        ),
        corpus: PerfAssemblyCorpusIdentity::from_observation(&artifact.provenance.corpus),
        policy: artifact.policy.clone(),
        estimator: estimator.ok_or_else(|| PerfEvidenceAssemblyError::IncompatibleShard {
            field: "estimator",
            detail: format!(
                "source run {} contains no QG-1 paired estimator",
                artifact.provenance.run_id
            ),
        })?,
    })
}

fn first_compatibility_difference(
    expected: &PerfEvidenceAssemblyCompatibility,
    actual: &PerfEvidenceAssemblyCompatibility,
    run_id: &str,
) -> PerfEvidenceAssemblyError {
    let field = if expected.profile != actual.profile {
        "profile"
    } else if expected.capacity_semantics != actual.capacity_semantics
        || expected.execution_capacity != actual.execution_capacity
        || expected.max_exercised_cell_width != actual.max_exercised_cell_width
    {
        "capacity_envelope"
    } else if expected.canonicalization != actual.canonicalization
        || expected.runner_hardware_sha256 != actual.runner_hardware_sha256
        || expected.runner_execution_identity_sha256 != actual.runner_execution_identity_sha256
        || expected.runner_durability_sha256 != actual.runner_durability_sha256
    {
        "machine_execution_identity"
    } else if expected.manifest_sha256 != actual.manifest_sha256 {
        "manifest_sha256"
    } else if expected.run_window != actual.run_window {
        "run_window"
    } else if expected.build.git_revision != actual.build.git_revision {
        "source_revision"
    } else if expected.build.cargo_lock_sha256 != actual.build.cargo_lock_sha256 {
        "cargo_lock_sha256"
    } else if expected.build.executable_sha256 != actual.build.executable_sha256 {
        "executable_sha256"
    } else if expected.build != actual.build {
        "build_identity"
    } else if expected.machine != actual.machine {
        "machine_identity"
    } else if expected.corpus != actual.corpus {
        "corpus_identity"
    } else if expected.policy != actual.policy {
        "evidence_policy"
    } else {
        "estimator"
    };
    PerfEvidenceAssemblyError::IncompatibleShard {
        field,
        detail: format!("source run {run_id} differs from the first completed shard"),
    }
}

fn hash_serialized<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, PerfEvidenceAssemblyError> {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(serde_json::to_vec_pretty(value)?);
    Ok(sha256_hex(&hasher.finalize()))
}

/// Typed fail-closed errors for offline evidence assembly.
#[derive(Debug, Error)]
pub enum PerfEvidenceAssemblyError {
    /// No completed or failed shard was supplied.
    #[error("QG-1 assembly requires at least one completed or failed shard")]
    EmptyAssembly,
    /// The bounded shard count was exceeded or overflowed.
    #[error("QG-1 assembly exceeds the bounded shard count")]
    TooManyShards,
    /// Only QG-1 is admitted by this assembler.
    #[error("QG-1 assembler cannot consume {found}")]
    UnsupportedGate {
        /// Gate found in the rejected source.
        found: PerfGate,
    },
    /// A source selected a cell outside the runnable profile projection.
    #[error("QG-1 assembly source contains unknown or NotApplicable cell {cell_id}")]
    UnexpectedCell {
        /// Rejected cell identity.
        cell_id: String,
    },
    /// Two completed shards contributed the same canonical cell.
    #[error("QG-1 assembly source overlap at cell {cell_id}")]
    OverlappingCell {
        /// Duplicated cell identity.
        cell_id: String,
    },
    /// Two terminal attempts reused one process run ID.
    #[error("QG-1 assembly repeats run ID {run_id}")]
    DuplicateRunId {
        /// Duplicated run identity.
        run_id: String,
    },
    /// A source differs from the common provenance contract.
    #[error("QG-1 shard is incompatible on {field}: {detail}")]
    IncompatibleShard {
        /// Stable incompatible field family.
        field: &'static str,
        /// Bounded source-specific detail.
        detail: String,
    },
    /// An H2 attempt bundle is malformed, incomplete, or cross-bound to
    /// different exact artifacts.
    #[error("QG-1 H2 attempt bundle is invalid: {reason}")]
    InvalidAttemptBundle {
        /// Stable diagnostic detail.
        reason: String,
    },
    /// A downstream caller requested completeness from a partial receipt.
    #[error("QG-1 assembly is incomplete ({missing} missing Required cells): {retry_predicate}")]
    IncompleteAssembly {
        /// Number of missing Applicable/Required cells.
        missing: usize,
        /// Concrete bounded rerun condition.
        retry_predicate: String,
    },
    /// A caller explicitly requested all Diagnostic cells, but some are absent.
    #[error(
        "QG-1 assembly lacks {missing} Diagnostic cells for full-plan coverage: {retry_predicate}"
    )]
    IncompleteDiagnosticCoverage {
        /// Number of missing Applicable/Diagnostic cells.
        missing: usize,
        /// Concrete bounded rerun condition.
        retry_predicate: String,
    },
    /// Required coverage is complete, but the profile has no claim cells or
    /// authentic Required evidence cannot support a downstream decision.
    #[error(
        "QG-1 assembly has {cells} non-adjudicable Required cells and {sources} blocking source NoClaims: {retry_predicate}"
    )]
    NonAdjudicableAssembly {
        /// Number of exact Required-cell diagnostics.
        cells: usize,
        /// Number of source-level `NoClaims` affecting Required cells.
        sources: usize,
        /// Concrete bounded rerun condition.
        retry_predicate: String,
        /// Exact persisted Required cell/status/reason diagnostics.
        diagnostics: Vec<PerfEvidenceAssemblyNoClaimCell>,
        /// Exact persisted source-level diagnostics affecting Required cells.
        source_diagnostics: Vec<PerfEvidenceAssemblyNoClaimSource>,
    },
    /// The independently sealed canonical matrix projection is stale or
    /// fabricated.
    #[error("QG-1 assembly matrix manifest is invalid: {reason}")]
    MatrixManifestMismatch {
        /// Stable mismatch detail.
        reason: String,
    },
    /// The run-label-independent semantic projection seal is stale.
    #[error("QG-1 semantic cell-set hash does not match retained measurements")]
    SemanticCellSetMismatch,
    /// The persisted schema is obsolete or names another artifact family.
    #[error("QG-1 assembly schema is {found}; current is quill-perf-evidence-assembly-v2")]
    SchemaMismatch {
        /// Schema found in the input.
        found: String,
    },
    /// The top-level assembly content seal does not match.
    #[error("QG-1 assembly hash seal does not match its contents")]
    HashMismatch,
    /// Strict JSON parsing or current-schema decoding failed.
    #[error("QG-1 assembly is malformed: {reason}")]
    Malformed {
        /// Bounded parse detail.
        reason: String,
    },
    /// Derived assembly fields disagree with retained source artifacts.
    #[error("QG-1 assembly is inconsistent: {reason}")]
    InconsistentAssembly {
        /// Bounded structural detail.
        reason: String,
    },
    /// A nested evidence artifact failed verification.
    #[error(transparent)]
    Evidence(#[from] EvidenceArtifactError),
    /// The exact H2 process receipt or its typed joins failed verification.
    #[error(transparent)]
    LocalPerf(#[from] LocalPerfRunError),
    /// Frozen machine-class admission failed.
    #[error(transparent)]
    MachineClass(#[from] MachineClassError),
    /// Frozen applicability-plan reconstruction failed.
    #[error(transparent)]
    Applicability(#[from] PerfApplicabilityPlanError),
    /// Filesystem persistence or loading failed.
    #[error("QG-1 assembly I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Strict JSON serialization failed.
    #[error("QG-1 assembly serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use std::ops::{Deref, DerefMut};
    use std::os::unix::fs::PermissionsExt as _;
    use std::sync::{Mutex, OnceLock};

    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::{
        EngineConcurrencyObservation, EvidenceCellSpec, ExecutionProfileId, HardwareClassId,
        PairedExperimentResult, PerfConcurrencyEngine, PerfConcurrencyObserver,
        PerfConcurrencyWitness, PerfOperationScope, PerfRawSample, PerfSampleArm, PerfSampleOrder,
        PerfSamplePhase, PerfSampleProvenance, Qg1BatchCoverage, Qg1ExpectedAuthority,
        Qg1LifecycleProducer, Qg1LifecycleWitness, Qg1SampleBinding, estimate_paired_experiment,
        estimate_paired_experiment_against_qg1_authority, seeded_balanced_pair_order,
    };

    const RUN_WINDOW: &str = "qg1-h4-test-window";
    const TEST_MACHINE_FINGERPRINT: &str =
        "linux-x86_64-test-machine-128thread-AMD_Ryzen_Threadripper_PRO_5995WX_64-Cores";

    fn private_tempdir(label: &str) -> TempDir {
        let directory = tempdir().unwrap_or_else(|error| panic!("{label}: {error}"));
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))
            .unwrap_or_else(|error| panic!("set private permissions on {label}: {error}"));
        directory
    }

    #[derive(Clone, Copy)]
    struct TestIdentity {
        executable: char,
        source_revision: char,
        cargo_lock: char,
        manifest: Option<char>,
        corpus: char,
    }

    impl TestIdentity {
        const PRIMARY: Self = Self {
            executable: 'a',
            source_revision: 'd',
            cargo_lock: 'c',
            manifest: None,
            corpus: 'b',
        };
    }

    /// One fixture-exact QG-1 source plus the independently retained
    /// authorities minted before its canonical throughput rows were produced.
    #[derive(Clone)]
    struct AuthorityBoundTestArtifact {
        artifact: PerfEvidenceArtifact,
        expected_authorities: Vec<Qg1ExpectedAuthority>,
    }

    impl AuthorityBoundTestArtifact {
        fn authority_refs(&self) -> Vec<&Qg1ExpectedAuthority> {
            self.expected_authorities.iter().collect()
        }
    }

    impl Deref for AuthorityBoundTestArtifact {
        type Target = PerfEvidenceArtifact;

        fn deref(&self) -> &Self::Target {
            &self.artifact
        }
    }

    impl DerefMut for AuthorityBoundTestArtifact {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.artifact
        }
    }

    fn test_profile() -> MachineProfileKey {
        MachineProfileKey::new(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        )
        .expect("registered physical-64 test profile")
    }

    fn test_plan() -> PerfApplicabilityPlanBinding {
        PerfMatrixSpec::complete()
            .applicability_plan(
                &MachineClassRegistry::frozen().expect("frozen registry"),
                test_profile(),
                PerfGate::Qg1,
            )
            .expect("QG-1 physical-64 plan")
            .binding
    }

    fn test_policy() -> EvidencePolicy {
        EvidencePolicy::predeclared()
    }

    fn estimator_config() -> PairedEstimatorConfig {
        PairedEstimatorConfig::predeclared(0x4834_5eed)
    }

    fn build_identity(identity: TestIdentity) -> BuildIdentity {
        BuildIdentity {
            executable_sha256: identity.executable.to_string().repeat(64),
            git_revision: identity.source_revision.to_string().repeat(40),
            git_dirty: false,
            worktree_state_sha256: None,
            cargo_lock_sha256: Some(identity.cargo_lock.to_string().repeat(64)),
            command_sha256: "f".repeat(64),
            environment_sha256: Some("e".repeat(64)),
            rustc_version: "rustc 1.91.0-nightly".to_owned(),
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            build_profile: "test".to_owned(),
            cargo_features: vec!["perf-harness".to_owned()],
        }
    }

    fn evidence_provenance(
        run_id: &str,
        configured_widths: Vec<usize>,
        document_count: u64,
        identity: TestIdentity,
    ) -> EvidenceProvenance {
        let run_variation = run_id
            .bytes()
            .fold(0_u32, |sum, byte| sum.wrapping_add(u32::from(byte)));
        EvidenceProvenance {
            run_id: run_id.to_owned(),
            run_window: RUN_WINDOW.to_owned(),
            manifest_sha256: identity.manifest.map_or_else(
                || test_plan().normalized_perf_manifest_sha256,
                |byte| byte.to_string().repeat(64),
            ),
            build: build_identity(identity),
            machine: MachineIdentity {
                fingerprint: TEST_MACHINE_FINGERPRINT.to_owned(),
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
                logical_cpus: 64,
                execution: PerfExecutionProvenance {
                    host_identity: "test-machine".to_owned(),
                    producer_os: PerfProducerOs::Linux,
                    physical_cores: 64,
                    logical_threads: 128,
                    process_available_threads: 64,
                    execution_capacity: 64,
                    max_exercised_cell_width: 64,
                    configured_engine_thread_widths: configured_widths,
                    runtime_detected_isa: ["aes", "avx2", "bmi2", "fma", "vaes"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                    cpu_affinity_allowed_list: Some("0-63".to_owned()),
                    affinity_or_cpuset_cap: Some(
                        "Cpus_allowed_list=0-63 (64 of 128 host logical threads)".to_owned(),
                    ),
                },
                cpu_governor: Some("performance".to_owned()),
                load_average_start: Some(f64::from(run_variation % 100) / 10.0),
                load_average_end: Some(f64::from(run_variation % 80) / 10.0),
            },
            peak_rss: PeakRssEvidence {
                method: "linux_vmhwm".to_owned(),
                bytes: Some(4_096 + u64::from(run_variation)),
            },
            corpus: CorpusIdentity {
                corpus_sha256: identity.corpus.to_string().repeat(64),
                query_set_sha256: None,
                qrels_sha256: None,
                document_count,
                content_bytes: None,
                generator_seed: 42,
                generator_revision: "qg1-h4-fixture-v1".to_owned(),
            },
        }
    }

    fn sample_provenance(run_id: &str, identity: TestIdentity) -> PerfSampleProvenance {
        PerfSampleProvenance {
            run_id: run_id.to_owned(),
            executable_sha256: identity.executable.to_string().repeat(64),
            corpus_sha256: identity.corpus.to_string().repeat(64),
            input_identity: None,
            worker_id: TEST_MACHINE_FINGERPRINT.to_owned(),
            build_profile: "test".to_owned(),
        }
    }

    fn threshold_artifact_for(artifact: &PerfEvidenceArtifact) -> PerfGateArtifact {
        PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate: PerfGate::Qg1,
            applicability_plan: Some(artifact.applicability_plan.clone()),
            bench_elf_sha256: artifact.provenance.build.executable_sha256.clone(),
            machine_fingerprint: artifact.provenance.machine.fingerprint.clone(),
            execution: Some(artifact.provenance.machine.execution.clone()),
            git_rev: artifact.provenance.build.git_revision.clone(),
            run_window: artifact.provenance.run_window.clone(),
            run_id: artifact.provenance.run_id.clone(),
            corpus_manifest_hash: artifact.provenance.corpus.corpus_sha256.clone(),
            manifest_sha256: artifact.provenance.manifest_sha256.clone(),
            cells: threshold_projection_from_evidence(artifact)
                .expect("production threshold projection accepts test evidence"),
            laws_attested: artifact.cells.len() == runnable_ordinals().len(),
        }
    }

    fn raw_throughput_stream(
        run_id: &str,
        identity: TestIdentity,
        scope: &PerfOperationScope,
        control_elapsed_ns: u64,
        treatment_elapsed_ns: u64,
        sample_id_base: u64,
    ) -> Vec<PerfRawSample> {
        let provenance = sample_provenance(run_id, identity);
        #[allow(clippy::cast_precision_loss)]
        let control_observed = 10_000.0 * 1_000_000_000.0 / control_elapsed_ns as f64;
        #[allow(clippy::cast_precision_loss)]
        let treatment_observed = 10_000.0 * 1_000_000_000.0 / treatment_elapsed_ns as f64;
        let orders = seeded_balanced_pair_order(30, 0x4834_5eed).expect("balanced pair order");
        let mut samples = Vec::with_capacity(60);
        for (index, first_arm) in orders.into_iter().enumerate() {
            let block_id = u64::try_from(index).expect("test block ID");
            let base = block_id * 1_000_000;
            let control_first = first_arm == PerfSampleArm::Control;
            let (control_start, treatment_start) = if control_first {
                (base, base + control_elapsed_ns + 100)
            } else {
                (base + treatment_elapsed_ns + 100, base)
            };
            let index = u64::try_from(index).expect("test sample index");
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id_base + index * 2,
                arm: PerfSampleArm::Control,
                order: if control_first {
                    PerfSampleOrder::First
                } else {
                    PerfSampleOrder::Second
                },
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: provenance.clone(),
                started_ns: control_start,
                ended_ns: control_start + control_elapsed_ns,
                work_units: None,
                byte_count: Some(1_000_000),
                observed_value: Some(control_observed),
                group_id: None,
                qg6_sample_binding: None,
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
            });
            samples.push(PerfRawSample {
                block_id,
                sample_id: sample_id_base + index * 2 + 1,
                arm: PerfSampleArm::Treatment,
                order: if control_first {
                    PerfSampleOrder::Second
                } else {
                    PerfSampleOrder::First
                },
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: provenance.clone(),
                started_ns: treatment_start,
                ended_ns: treatment_start + treatment_elapsed_ns,
                work_units: None,
                byte_count: Some(1_000_000),
                observed_value: Some(treatment_observed),
                group_id: None,
                qg6_sample_binding: None,
                qg1_sample_binding: None,
                tantivy_config_sha256: None,
            });
        }
        samples
    }

    fn paired_results(
        contract: &PlanCellContract,
        run_id: &str,
        identity: TestIdentity,
        invalid_control_null: bool,
        config: &PairedEstimatorConfig,
    ) -> (
        PairedExperimentResult,
        PairedExperimentResult,
        Option<Qg1ExpectedAuthority>,
    ) {
        const PAIRS: usize = 30;
        const CONTENT_BYTES: u64 = 64_000;

        let scope = crate::perf::perf_operation_scope(
            PerfGate::Qg1,
            &contract.spec.fixture,
            &contract.spec.metric,
        );
        let control_null_elapsed_ns = if invalid_control_null {
            50_000
        } else {
            100_000
        };
        if contract.spec.metric != "docs_per_second" {
            let effect = raw_throughput_stream(run_id, identity, &scope, 100_000, 80_000, 0);
            let control_null = raw_throughput_stream(
                run_id,
                identity,
                &scope,
                100_000,
                control_null_elapsed_ns,
                10_000,
            );
            let treatment_null =
                raw_throughput_stream(run_id, identity, &scope, 80_000, 80_000, 20_000);
            return (
                estimate_paired_experiment(&effect, &control_null, config)
                    .expect("paired non-throughput QG-1 experiment"),
                estimate_paired_experiment(&effect, &treatment_null, config)
                    .expect("non-throughput treatment-arm Q/Q null"),
                None,
            );
        }

        let work_units = contract
            .spec
            .document_count
            .expect("canonical QG-1 throughput cell has a document count");
        let provenance = sample_provenance(run_id, identity);
        let schedule =
            seeded_balanced_pair_order(PAIRS, 0x4834_5eed).expect("QG-1 authority schedule");
        let mut config = config.clone();
        let producer = config
            .install_qg1_lifecycle_authority(
                scope.clone(),
                provenance.corpus_sha256.clone(),
                "a".repeat(64),
                "b".repeat(64),
                work_units,
                CONTENT_BYTES,
                1,
                vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: work_units,
                }],
                format!("synthetic-{:08}", work_units.saturating_sub(1)),
                u64::try_from(PAIRS).expect("QG-1 pair count fits u64"),
                vec![
                    (
                        crate::perf::QG1_STREAM_ROLE_EFFECT.to_owned(),
                        0,
                        0,
                        schedule.clone(),
                    ),
                    (
                        crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL.to_owned(),
                        0,
                        10_000,
                        schedule.clone(),
                    ),
                    (
                        crate::perf::QG1_STREAM_ROLE_QUILL_NULL.to_owned(),
                        0,
                        20_000,
                        schedule,
                    ),
                ],
            )
            .expect("mint QG-1 authority before the first raw row");
        let expected_authority = producer.expected_authority().clone();
        let effect = authority_bound_qg1_stream(
            run_id,
            identity,
            &scope,
            &producer,
            crate::perf::QG1_STREAM_ROLE_EFFECT,
            100_000,
            80_000,
            0,
            work_units,
            CONTENT_BYTES,
        );
        let control_null = authority_bound_qg1_stream(
            run_id,
            identity,
            &scope,
            &producer,
            crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL,
            100_000,
            control_null_elapsed_ns,
            10_000,
            work_units,
            CONTENT_BYTES,
        );
        let treatment_null = authority_bound_qg1_stream(
            run_id,
            identity,
            &scope,
            &producer,
            crate::perf::QG1_STREAM_ROLE_QUILL_NULL,
            80_000,
            80_000,
            20_000,
            work_units,
            CONTENT_BYTES,
        );
        (
            estimate_paired_experiment_against_qg1_authority(
                &effect,
                &control_null,
                &config,
                Some(&expected_authority),
            )
            .expect("authority-bound paired QG-1 experiment"),
            estimate_paired_experiment_against_qg1_authority(
                &effect,
                &treatment_null,
                &config,
                Some(&expected_authority),
            )
            .expect("authority-bound treatment-arm Q/Q null"),
            Some(expected_authority),
        )
    }

    fn evidence_cell(
        contract: &PlanCellContract,
        paired: &PairedExperimentResult,
        treatment_null: &PairedExperimentResult,
        expected_authority: Option<&Qg1ExpectedAuthority>,
        policy: &EvidencePolicy,
    ) -> EvidenceCell {
        let role = contract.role.expect("runnable cell role");
        let concurrency_witness =
            (role == EvidenceRole::Required).then(|| PerfConcurrencyWitness {
                configured_threads: contract.configured_threads,
                observations: vec![
                    EngineConcurrencyObservation {
                        engine: PerfConcurrencyEngine::Quill,
                        observer: PerfConcurrencyObserver::RayonCurrentPoolWidth,
                        observation_count: 30,
                        min_observed_worker_pool_threads: contract.configured_threads,
                        max_observed_worker_pool_threads: contract.configured_threads,
                    },
                    EngineConcurrencyObservation {
                        engine: PerfConcurrencyEngine::Tantivy,
                        observer: PerfConcurrencyObserver::TantivyWriterConstruction,
                        observation_count: 30,
                        min_observed_worker_pool_threads: contract.configured_threads,
                        max_observed_worker_pool_threads: contract.configured_threads,
                    },
                ],
            });
        let mut cell = EvidenceCell::evaluate(
            EvidenceCellSpec {
                gate: PerfGate::Qg1,
                fixture: contract.spec.fixture.clone(),
                metric: contract.spec.metric.clone(),
                unit: "docs/s".to_owned(),
                role,
                input_identity: None,
                qg6_semantic_contract: None,
                cold_cache: None,
                concurrency_witness,
            },
            paired.clone(),
            policy,
        )
        .expect("evaluate QG-1 test cell");
        if let Some(expected_authority) = expected_authority {
            cell.attach_treatment_arm_null_against_qg1_authority(
                treatment_null.clone(),
                policy,
                Some(expected_authority),
            )
            .expect("attach authority-bound same-invocation Q/Q null");
        } else {
            cell.attach_treatment_arm_null(treatment_null.clone(), policy)
                .expect("attach same-invocation Q/Q null");
        }
        cell
    }

    fn shard(
        ordinals: &[usize],
        run_id: &str,
        run_label: &str,
        invalid_ordinal: Option<usize>,
        identity: TestIdentity,
    ) -> Vec<AuthorityBoundTestArtifact> {
        shard_with_partial_code(
            ordinals,
            run_id,
            run_label,
            invalid_ordinal,
            identity,
            PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE,
        )
    }

    fn shard_with_partial_code(
        ordinals: &[usize],
        run_id: &str,
        run_label: &str,
        invalid_ordinal: Option<usize>,
        identity: TestIdentity,
        partial_no_claim_code: &str,
    ) -> Vec<AuthorityBoundTestArtifact> {
        shard_with_contract(
            ordinals,
            run_id,
            run_label,
            invalid_ordinal,
            identity,
            partial_no_claim_code,
            false,
            test_policy(),
            &estimator_config(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn shard_with_contract(
        ordinals: &[usize],
        run_id: &str,
        _run_label: &str,
        invalid_ordinal: Option<usize>,
        identity: TestIdentity,
        partial_no_claim_code: &str,
        force_source_no_claim: bool,
        policy: EvidencePolicy,
        estimator: &PairedEstimatorConfig,
    ) -> Vec<AuthorityBoundTestArtifact> {
        let contract = PlanContract::reconstruct(&test_plan()).expect("test plan contract");
        let runnable_count = contract
            .cells
            .iter()
            .filter(|cell| cell.applicability.is_runnable())
            .count();
        let fixture_groups = if ordinals.len() == runnable_count {
            vec![(run_id.to_owned(), ordinals.to_vec())]
        } else {
            let mut by_fixture = BTreeMap::<String, Vec<usize>>::new();
            for ordinal in ordinals {
                by_fixture
                    .entry(contract.cells[*ordinal].spec.fixture.clone())
                    .or_default()
                    .push(*ordinal);
            }
            by_fixture
                .into_values()
                .enumerate()
                .map(|(index, fixture_ordinals)| {
                    (format!("{run_id}-fixture-{index}"), fixture_ordinals)
                })
                .collect()
        };
        fixture_groups
            .into_iter()
            .map(|(fixture_run_id, fixture_ordinals)| {
                fixture_shard_with_contract(
                    &contract,
                    &fixture_ordinals,
                    &fixture_run_id,
                    invalid_ordinal,
                    identity,
                    partial_no_claim_code,
                    force_source_no_claim,
                    &policy,
                    estimator,
                    runnable_count,
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn fixture_shard_with_contract(
        contract: &PlanContract,
        ordinals: &[usize],
        run_id: &str,
        invalid_ordinal: Option<usize>,
        identity: TestIdentity,
        partial_no_claim_code: &str,
        force_source_no_claim: bool,
        policy: &EvidencePolicy,
        estimator: &PairedEstimatorConfig,
        runnable_count: usize,
    ) -> AuthorityBoundTestArtifact {
        let mut expected_authorities = Vec::new();
        let cells = ordinals
            .iter()
            .map(|ordinal| {
                let contract_cell = &contract.cells[*ordinal];
                let (paired, treatment_null, expected_authority) = paired_results(
                    contract_cell,
                    run_id,
                    identity,
                    invalid_ordinal == Some(*ordinal),
                    estimator,
                );
                let cell = evidence_cell(
                    contract_cell,
                    &paired,
                    &treatment_null,
                    expected_authority.as_ref(),
                    policy,
                );
                if let Some(expected_authority) = expected_authority {
                    expected_authorities.push(expected_authority);
                }
                cell
            })
            .collect::<Vec<_>>();
        let configured_widths = ordinals
            .iter()
            .map(|ordinal| contract.cells[*ordinal].configured_threads)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let document_count = ordinals
            .iter()
            .filter_map(|ordinal| contract.cells[*ordinal].spec.document_count)
            .max()
            .expect("QG-1 shard contains a measured corpus size");
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            test_plan(),
            policy.clone(),
            evidence_provenance(run_id, configured_widths, document_count, identity),
            cells,
        )
        .expect("assemble source evidence shard");
        if ordinals.len() != runnable_count || force_source_no_claim {
            let message = if partial_no_claim_code == PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE {
                PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_DETAIL
            } else {
                "selected canonical subset is durable input for offline assembly only"
            };
            artifact.force_no_claim(partial_no_claim_code, message);
        }
        let authority_refs = expected_authorities.iter().collect::<Vec<_>>();
        let prebinding_bytes = artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstruct canonical authority-bound fixture prebinding bytes");
        artifact = PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
            &prebinding_bytes,
            &authority_refs,
        )
        .expect("reload exact authority-bound fixture prebinding bytes");
        assert_eq!(
            canonical_evidence_bytes(&artifact)
                .expect("serialize reloaded canonical fixture prebinding evidence"),
            prebinding_bytes,
            "fixture runner binding must receive the exact canonical prebinding identity"
        );
        let threshold_bytes = canonical_threshold_bytes(&threshold_artifact_for(&artifact))
            .expect("canonical threshold artifact");
        let runner = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            PerfGate::Qg1.label(),
            &artifact.provenance.build.git_revision,
            artifact
                .provenance
                .build
                .cargo_lock_sha256
                .as_deref()
                .expect("Cargo.lock hash"),
            &artifact.provenance.build.executable_sha256,
            &artifact.provenance.build.command_sha256,
            artifact
                .provenance
                .build
                .environment_sha256
                .as_deref()
                .expect("environment hash"),
            run_id,
            run_id,
            RUN_WINDOW,
            &threshold_bytes,
            &prebinding_bytes,
        );
        let sealed = artifact
            .bind_machine_class_identity_and_seal_against_qg1_authorities(
                runner,
                &threshold_bytes,
                &prebinding_bytes,
                &authority_refs,
            )
            .expect("bind authority-aware source runner receipt");
        AuthorityBoundTestArtifact {
            artifact: PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
                &sealed,
                &authority_refs,
            )
            .expect("reload authority-aware source evidence"),
            expected_authorities,
        }
    }

    fn authority_bound_qg1_stream(
        run_id: &str,
        identity: TestIdentity,
        scope: &PerfOperationScope,
        producer: &Qg1LifecycleProducer,
        stream_role: &str,
        control_elapsed_ns: u64,
        treatment_elapsed_ns: u64,
        sample_id_base: u64,
        work_units: u64,
        content_bytes: u64,
    ) -> Vec<PerfRawSample> {
        let mut samples = raw_throughput_stream(
            run_id,
            identity,
            scope,
            control_elapsed_ns,
            treatment_elapsed_ns,
            sample_id_base,
        );
        for sample in &mut samples {
            sample.scope = scope.clone();
            sample.work_units = Some(work_units);
            sample.byte_count = Some(content_bytes);
            let stream_sequence = sample.block_id.saturating_mul(2)
                + u64::from(sample.order == PerfSampleOrder::Second);
            let tantivy_witness = stream_role == crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL
                || (stream_role == crate::perf::QG1_STREAM_ROLE_EFFECT
                    && sample.arm == PerfSampleArm::Control);
            let lifecycle_witness = if tantivy_witness {
                Qg1LifecycleWitness::Tantivy {
                    searchable_segments_before: 1,
                    searchable_segments_after: 1,
                    join_elapsed_ns: 1,
                    writer_rearmed: false,
                }
            } else {
                Qg1LifecycleWitness::Quill {
                    publication_generation_delta: 1,
                }
            };
            let binding = Qg1SampleBinding {
                schema_version: Qg1SampleBinding::SCHEMA_VERSION.to_owned(),
                stream_role: stream_role.to_owned(),
                stream_id_sha256: String::new(),
                stream_sequence,
                raw_sample_id: sample.sample_id,
                raw_block_id: sample.block_id,
                raw_arm: sample.arm,
                raw_order: sample.order,
                lifecycle_authority_sha256: String::new(),
                stream_role_identity_sha256: String::new(),
                producer_capability_sha256: String::new(),
                producer_capability_tag_sha256: String::new(),
                lifecycle_receipt_id_sha256: String::new(),
                lifecycle_receipt_sha256: String::new(),
                prepared_corpus_sha256: sample.provenance.corpus_sha256.clone(),
                prepared_input_sha256: String::new(),
                prepared_manifest_sha256: "a".repeat(64),
                indexed_content_sha256: "b".repeat(64),
                document_count: work_units,
                content_bytes,
                prepared_batch_count: 1,
                recorded_batch_count: 1,
                batch_coverage: vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: work_units,
                }],
                tail_document_id: format!("synthetic-{:08}", work_units.saturating_sub(1)),
                terminal_endpoint_ns: sample.ended_ns.saturating_sub(sample.started_ns),
                lifecycle_witness,
            };
            sample.qg1_sample_binding = Some(
                producer
                    .consume_lifecycle_receipt(&sample.scope, &sample.provenance, binding)
                    .expect("producer consumes exactly one QG-1 receipt per raw row"),
            );
        }
        samples
    }

    fn authority_bound_qg1_shard(run_id: &str) -> (PerfEvidenceArtifact, Qg1ExpectedAuthority) {
        const PAIRS: usize = 30;
        const CONTENT_BYTES: u64 = 64_000;

        let identity = TestIdentity::PRIMARY;
        let contract = PlanContract::reconstruct(&test_plan()).expect("test plan contract");
        let contract_cell = contract
            .cells
            .iter()
            .find(|cell| {
                cell.spec.fixture == "bulk/tiny/1/positions_on"
                    && cell.spec.metric == "docs_per_second"
            })
            .expect("canonical QG-1 tiny throughput cell");
        let work_units = contract_cell
            .spec
            .document_count
            .expect("QG-1 throughput cell has a document count");
        let scope = PerfOperationScope {
            operation_id: format!(
                "{}.{}.{}",
                PerfGate::Qg1,
                contract_cell.spec.fixture,
                contract_cell.spec.metric
            ),
            version: 1,
            semantics: PerfMetricSemantics::Throughput,
            unit: "docs/s".to_owned(),
        };
        let sample_provenance = sample_provenance(run_id, identity);
        let schedule =
            seeded_balanced_pair_order(PAIRS, 0x4834_5eed).expect("QG-1 authority schedule");
        let mut estimator = estimator_config();
        let producer = estimator
            .install_qg1_lifecycle_authority(
                scope.clone(),
                sample_provenance.corpus_sha256.clone(),
                "a".repeat(64),
                "b".repeat(64),
                work_units,
                CONTENT_BYTES,
                1,
                vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: work_units,
                }],
                format!("synthetic-{:08}", work_units.saturating_sub(1)),
                u64::try_from(PAIRS).expect("QG-1 pair count fits u64"),
                vec![
                    (
                        crate::perf::QG1_STREAM_ROLE_EFFECT.to_owned(),
                        0,
                        0,
                        schedule.clone(),
                    ),
                    (
                        crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL.to_owned(),
                        0,
                        10_000,
                        schedule.clone(),
                    ),
                    (
                        crate::perf::QG1_STREAM_ROLE_QUILL_NULL.to_owned(),
                        0,
                        20_000,
                        schedule,
                    ),
                ],
            )
            .expect("mint QG-1 authority before the first raw row");
        let expected_authority = producer.expected_authority().clone();
        let effect = authority_bound_qg1_stream(
            run_id,
            identity,
            &scope,
            &producer,
            crate::perf::QG1_STREAM_ROLE_EFFECT,
            100_000,
            80_000,
            0,
            work_units,
            CONTENT_BYTES,
        );
        let tantivy_null = authority_bound_qg1_stream(
            run_id,
            identity,
            &scope,
            &producer,
            crate::perf::QG1_STREAM_ROLE_TANTIVY_NULL,
            100_000,
            100_000,
            10_000,
            work_units,
            CONTENT_BYTES,
        );
        let quill_null = authority_bound_qg1_stream(
            run_id,
            identity,
            &scope,
            &producer,
            crate::perf::QG1_STREAM_ROLE_QUILL_NULL,
            80_000,
            80_000,
            20_000,
            work_units,
            CONTENT_BYTES,
        );
        let paired = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &tantivy_null,
            &estimator,
            Some(&expected_authority),
        )
        .expect("authority-bound QG-1 effect estimates");
        let treatment_arm_null = estimate_paired_experiment_against_qg1_authority(
            &effect,
            &quill_null,
            &estimator,
            Some(&expected_authority),
        )
        .expect("authority-bound QG-1 treatment null estimates");
        let mut cell = EvidenceCell::evaluate(
            EvidenceCellSpec {
                gate: PerfGate::Qg1,
                fixture: contract_cell.spec.fixture.clone(),
                metric: contract_cell.spec.metric.clone(),
                unit: "docs/s".to_owned(),
                role: EvidenceRole::Required,
                input_identity: None,
                qg6_semantic_contract: None,
                cold_cache: None,
                concurrency_witness: Some(PerfConcurrencyWitness {
                    configured_threads: contract_cell.configured_threads,
                    observations: vec![
                        EngineConcurrencyObservation {
                            engine: PerfConcurrencyEngine::Quill,
                            observer: PerfConcurrencyObserver::RayonCurrentPoolWidth,
                            observation_count: PAIRS,
                            min_observed_worker_pool_threads: contract_cell.configured_threads,
                            max_observed_worker_pool_threads: contract_cell.configured_threads,
                        },
                        EngineConcurrencyObservation {
                            engine: PerfConcurrencyEngine::Tantivy,
                            observer: PerfConcurrencyObserver::TantivyWriterConstruction,
                            observation_count: PAIRS,
                            min_observed_worker_pool_threads: contract_cell.configured_threads,
                            max_observed_worker_pool_threads: contract_cell.configured_threads,
                        },
                    ],
                }),
            },
            paired,
            &test_policy(),
        )
        .expect("authority-bound QG-1 cell evaluates");
        cell.attach_treatment_arm_null_against_qg1_authority(
            treatment_arm_null,
            &test_policy(),
            Some(&expected_authority),
        )
        .expect("authority-bound QG-1 treatment null attaches");
        let mut artifact = PerfEvidenceArtifact::assemble(
            PerfGate::Qg1,
            test_plan(),
            test_policy(),
            evidence_provenance(
                run_id,
                vec![contract_cell.configured_threads],
                work_units,
                identity,
            ),
            vec![cell],
        )
        .expect("assemble authority-bound QG-1 source artifact");
        artifact.force_no_claim(
            PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE,
            PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_DETAIL,
        );
        let prebinding_bytes = artifact
            .reconstructed_prebinding_bytes()
            .expect("canonical authority prebinding bytes");
        artifact = PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
            &prebinding_bytes,
            &[&expected_authority],
        )
        .expect("reload exact authority prebinding source");
        let threshold_bytes = canonical_threshold_bytes(&threshold_artifact_for(&artifact))
            .expect("authority-bound threshold bytes");
        let runner = crate::machine_class_registry::admitted_test_identity_for_artifacts(
            PerfGate::Qg1.label(),
            &artifact.provenance.build.git_revision,
            artifact
                .provenance
                .build
                .cargo_lock_sha256
                .as_deref()
                .expect("Cargo.lock hash"),
            &artifact.provenance.build.executable_sha256,
            &artifact.provenance.build.command_sha256,
            artifact
                .provenance
                .build
                .environment_sha256
                .as_deref()
                .expect("environment hash"),
            run_id,
            run_id,
            RUN_WINDOW,
            &threshold_bytes,
            &prebinding_bytes,
        );
        let sealed = artifact
            .bind_machine_class_identity_and_seal_against_qg1_authorities(
                runner,
                &threshold_bytes,
                &prebinding_bytes,
                &[&expected_authority],
            )
            .expect("seal authority-bound QG-1 source artifact");
        (
            PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
                &sealed,
                &[&expected_authority],
            )
            .expect("reload authority-bound QG-1 source artifact"),
            expected_authority,
        )
    }

    fn normalize_h2_test_artifact(
        artifact: AuthorityBoundTestArtifact,
    ) -> Vec<AuthorityBoundTestArtifact> {
        vec![artifact]
    }

    fn completed_test_attempt_directory(
        artifact: &PerfEvidenceArtifact,
        external_qg1_authorities: &[&Qg1ExpectedAuthority],
    ) -> TempDir {
        let runnable_count = runnable_ordinals().len();
        let fixture_selector = (artifact.cells.len() != runnable_count).then(|| {
            let fixtures = artifact
                .cells
                .iter()
                .map(|cell| cell.spec.fixture.as_str())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                fixtures.len(),
                1,
                "H2 partial test bundle selects one fixture"
            );
            fixtures.into_iter().next().expect("one fixture").to_owned()
        });
        let run_log_bytes = format!("runner-log:{}", artifact.provenance.run_id).into_bytes();
        let prebinding_bytes = artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstructed prebinding test evidence");
        let bound_bytes = canonical_evidence_bytes(artifact).expect("canonical bound test bytes");
        let threshold_artifact = threshold_artifact_for(artifact);
        let threshold_bytes =
            canonical_threshold_bytes(&threshold_artifact).expect("canonical threshold test bytes");
        let identity = artifact
            .machine_class
            .identity()
            .expect("test bound runner identity");
        let manifest = identity
            .artifact_manifest()
            .expect("test artifact manifest");
        let receipt_bytes = crate::local_perf_runner::completed_attempt_receipt_for_test(
            artifact,
            fixture_selector.as_deref(),
            &run_log_bytes,
            &threshold_bytes,
            &prebinding_bytes,
            &bound_bytes,
            external_qg1_authorities,
        );

        let directory = private_tempdir("completed H2 attempt bundle");
        let artifacts_directory = directory.path().join("artifacts");
        fs::create_dir(&artifacts_directory).expect("create private H2 artifact directory");
        fs::set_permissions(&artifacts_directory, fs::Permissions::from_mode(0o700))
            .expect("make H2 artifact directory private");
        for (path, bytes) in [
            (directory.path().join("QG-1.attempt.json"), receipt_bytes),
            (directory.path().join("run.log"), run_log_bytes),
            (
                directory.path().join("QG-1.runner.json"),
                identity.receipt_json().as_bytes().to_vec(),
            ),
            (
                directory.path().join("QG-1.artifacts.json"),
                manifest
                    .manifest()
                    .to_json_bytes()
                    .expect("canonical test artifact manifest"),
            ),
            (
                directory.path().join("QG-1.bound.evidence.json"),
                bound_bytes,
            ),
            (artifacts_directory.join("QG-1.json"), threshold_bytes),
            (
                artifacts_directory.join("QG-1.evidence.json"),
                prebinding_bytes,
            ),
        ] {
            fs::write(path, bytes).expect("write exact H2 attempt input");
        }
        directory
    }

    fn test_attempt_bundle(
        artifact: &AuthorityBoundTestArtifact,
    ) -> VerifiedLocalPerfAttemptBundle {
        static CACHE: OnceLock<Mutex<BTreeMap<String, VerifiedLocalPerfAttemptBundle>>> =
            OnceLock::new();
        let key = sha256_hex(
            &canonical_evidence_bytes(artifact).expect("canonical cache-key evidence bytes"),
        );
        let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
        let mut cache = cache.lock().expect("test attempt cache lock");
        if let Some(bundle) = cache.get(&key).cloned() {
            return bundle;
        }
        let authority_refs = artifact.authority_refs();
        let directory = completed_test_attempt_directory(artifact, &authority_refs);
        let bundle = VerifiedLocalPerfAttemptBundle::load_verified_against_qg1_authorities(
            directory.path(),
            &authority_refs,
        )
        .expect("load exact completed H2 test bundle through authority-aware production boundary");
        cache.insert(key, bundle.clone());
        bundle
    }

    trait IntoAuthorityBoundTestArtifacts {
        fn into_authority_bound_test_artifacts(self) -> Vec<AuthorityBoundTestArtifact>;
    }

    impl IntoAuthorityBoundTestArtifacts for AuthorityBoundTestArtifact {
        fn into_authority_bound_test_artifacts(self) -> Vec<AuthorityBoundTestArtifact> {
            vec![self]
        }
    }

    impl IntoAuthorityBoundTestArtifacts for Vec<AuthorityBoundTestArtifact> {
        fn into_authority_bound_test_artifacts(self) -> Vec<AuthorityBoundTestArtifact> {
            self
        }
    }

    fn assemble_test<T>(
        completed: Vec<T>,
        failed: Vec<VerifiedLocalPerfAttemptBundle>,
    ) -> Result<PerfEvidenceAssemblyArtifact, PerfEvidenceAssemblyError>
    where
        T: IntoAuthorityBoundTestArtifacts,
    {
        let completed = completed
            .into_iter()
            .flat_map(|artifact| artifact.into_authority_bound_test_artifacts())
            .flat_map(normalize_h2_test_artifact)
            .collect::<Vec<_>>();
        let authority_refs = completed
            .iter()
            .map(AuthorityBoundTestArtifact::authority_refs)
            .collect::<Vec<_>>();
        let mut attempts = completed
            .iter()
            .zip(&authority_refs)
            .map(|(artifact, authorities)| (test_attempt_bundle(artifact), authorities.as_slice()))
            .collect::<Vec<_>>();
        attempts.extend(
            failed
                .into_iter()
                .map(|failed| (failed, &[] as &[&Qg1ExpectedAuthority])),
        );
        PerfEvidenceAssemblyArtifact::assemble_against_qg1_authorities(attempts)
    }

    #[test]
    fn fixture_shard_reloads_canonical_authority_prebinding_before_runner_binding() {
        let ordinal = runnable_ordinals()
            .into_iter()
            .next()
            .expect("test plan has a runnable QG-1 cell");
        let mut fixtures = shard(
            &[ordinal],
            "fixture-prebinding-exact",
            "fixture-prebinding-exact-runner",
            None,
            TestIdentity::PRIMARY,
        );
        assert_eq!(fixtures.len(), 1, "one cell produces one fixture shard");
        let fixture = fixtures.pop().expect("one fixture shard");
        let authority_refs = fixture.authority_refs();
        let prebinding_bytes = fixture
            .artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstruct fixture prebinding identity");
        let prebinding = PerfEvidenceArtifact::from_verified_slice_against_qg1_authorities(
            &prebinding_bytes,
            &authority_refs,
        )
        .expect("external QG-1 authority authenticates exact fixture prebinding bytes");
        assert_eq!(
            canonical_evidence_bytes(&prebinding).expect("canonical fixture prebinding bytes"),
            prebinding_bytes,
            "fixture prebinding identity must survive canonical reconstruction and authority-aware reload"
        );
        assert_ne!(
            canonical_evidence_bytes(&fixture.artifact).expect("canonical bound fixture bytes"),
            prebinding_bytes,
            "fixture runner binding must not be mistaken for the prebinding identity"
        );
        let directory = completed_test_attempt_directory(&fixture.artifact, &authority_refs);
        VerifiedLocalPerfAttemptBundle::load_verified_against_qg1_authorities(
            directory.path(),
            &authority_refs,
        )
        .expect("production attempt reload preserves the authority-bound prebinding identity");
    }

    #[test]
    fn authority_aware_assembly_binds_each_qg1_shard_to_its_retained_authorities() {
        let (artifact, expected_authority) = authority_bound_qg1_shard("assembly-authority-good");
        let exact_authorities = [&expected_authority];
        let directory = completed_test_attempt_directory(&artifact, &exact_authorities);
        assert!(
            VerifiedLocalPerfAttemptBundle::load_verified(directory.path()).is_err(),
            "the authority-free H2 loader must refuse an authority-bearing QG-1 shard"
        );

        let bundle = VerifiedLocalPerfAttemptBundle::load_verified_against_qg1_authorities(
            directory.path(),
            &exact_authorities,
        )
        .expect("the retained authority authenticates its exact H2 input");
        assert!(
            PerfEvidenceAssemblyArtifact::assemble(vec![bundle.clone()]).is_err(),
            "the authority-free assembly entry must fail closed for the QG-1 shard"
        );

        let assembly = PerfEvidenceAssemblyArtifact::assemble_against_qg1_authorities(vec![(
            bundle.clone(),
            &exact_authorities,
        )])
        .expect("the authority-aware assembly admits the honest QG-1 shard");
        let source_sha256 = assembly.source_shards()[0].bound_evidence_file_sha256();
        assembly
            .verify_integrity_against_qg1_authorities(&[(source_sha256, &exact_authorities)])
            .expect("the assembled source replays only against its retained authority slice");
        assert!(
            assembly.verify_integrity().is_err(),
            "the authority-free assembly replay must fail closed"
        );

        let (_foreign_artifact, foreign_authority) =
            authority_bound_qg1_shard("assembly-authority-foreign");
        let wrong_authorities = [&foreign_authority];
        assert!(
            VerifiedLocalPerfAttemptBundle::load_verified_against_qg1_authorities(
                directory.path(),
                &wrong_authorities,
            )
            .is_err(),
            "a foreign authority must be rejected at the H2 loader boundary"
        );
        assert!(
            PerfEvidenceAssemblyArtifact::assemble_against_qg1_authorities(vec![(
                bundle.clone(),
                &wrong_authorities
            ),])
            .is_err(),
            "a foreign QG-1 authority must not authenticate this shard"
        );
        assert!(
            assembly
                .verify_integrity_against_qg1_authorities(&[(source_sha256, &wrong_authorities)])
                .is_err(),
            "a source-keyed authority assignment must not accept another shard's authority"
        );
        let duplicate_authorities = [&expected_authority, &expected_authority];
        assert!(
            VerifiedLocalPerfAttemptBundle::load_verified_against_qg1_authorities(
                directory.path(),
                &duplicate_authorities,
            )
            .is_err(),
            "duplicate authorities must remain ambiguous at the H2 loader boundary"
        );
        assert!(
            PerfEvidenceAssemblyArtifact::assemble_against_qg1_authorities(vec![(
                bundle,
                &duplicate_authorities
            ),])
            .is_err(),
            "duplicate candidate authorities must remain ambiguous and fail closed"
        );
    }

    fn failed_test_attempt_bundle(
        artifact: &PerfEvidenceArtifact,
        code: i64,
    ) -> VerifiedLocalPerfAttemptBundle {
        let fixtures = artifact
            .cells
            .iter()
            .map(|cell| cell.spec.fixture.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            fixtures.len(),
            1,
            "failed H2 test bundle selects one fixture"
        );
        let fixture = fixtures.into_iter().next().expect("one failed fixture");
        let run_log_bytes = format!("runner-log:{}", artifact.provenance.run_id).into_bytes();
        let prebinding_bytes = artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstructed failed-template prebinding evidence");
        let bound_bytes = canonical_evidence_bytes(artifact).expect("failed-template bound bytes");
        let threshold_bytes = canonical_threshold_bytes(&threshold_artifact_for(artifact))
            .expect("failed-template threshold bytes");
        let receipt_bytes = crate::local_perf_runner::failed_attempt_receipt_for_test(
            artifact,
            Some(fixture),
            &run_log_bytes,
            &threshold_bytes,
            &prebinding_bytes,
            &bound_bytes,
            code,
        );
        let directory = private_tempdir("failed H2 attempt bundle");
        fs::write(directory.path().join("QG-1.attempt.json"), receipt_bytes)
            .expect("write exact failed H2 receipt");
        fs::write(directory.path().join("run.log"), run_log_bytes)
            .expect("write exact failed H2 log");
        VerifiedLocalPerfAttemptBundle::load_verified(directory.path())
            .expect("load exact failed H2 test bundle through production boundary")
    }

    fn assert_h2_lifecycle_boundary(assembly: &PerfEvidenceAssemblyArtifact) {
        let codes = assembly
            .non_adjudicable_sources()
            .iter()
            .map(|source| source.reason().code.as_str())
            .collect::<BTreeSet<_>>();
        assert!(codes.contains(PERF_ASSEMBLY_ENGINE_LIFECYCLE_NO_CLAIM_CODE));
        // THE PROCESS-TREE NO-CLAIM IS EXPECTED ABSENT, and that is the H2
        // boundary MOVING rather than relaxing (bd-916qm). An assembly source
        // is only ever built from an attempt that COMPLETED, and since
        // 91c55d5b a completed receipt cannot be sealed at all unless its
        // lifecycle proves descendant-tree quiescence -- `validate` refuses it
        // with "lacks a completed descendant-tree quiescence proof", pinned by
        // `local_perf_runner::tests::
        // completed_receipt_rejects_direct_child_only_or_escaped_tree_claims`.
        // The downstream NoClaim was the mitigation for a gap the producer now
        // refuses to have, so asserting its presence asserted an unreachable
        // state: the test was failing on a strengthened invariant, not a lost
        // one. The emitting branch is kept as a fail-closed guard in case that
        // completion invariant is ever relaxed, which is why this asserts the
        // absence exactly rather than dropping the check.
        assert!(!codes.contains(PERF_ASSEMBLY_PROCESS_TREE_NO_CLAIM_CODE));
        if assembly.has_full_plan_coverage() && assembly.counts().required_cells() != 0 {
            assert_eq!(
                assembly.readiness(),
                PerfEvidenceAssemblyReadiness::NoClaimInvalidEvidence
            );
            assert!(matches!(
                assembly.require_adjudicable(),
                Err(PerfEvidenceAssemblyError::NonAdjudicableAssembly { .. })
            ));
        }
    }

    fn runnable_ordinals() -> Vec<usize> {
        PlanContract::reconstruct(&test_plan())
            .expect("test plan contract")
            .cells
            .iter()
            .enumerate()
            .filter_map(|(ordinal, cell)| cell.applicability.is_runnable().then_some(ordinal))
            .collect()
    }

    fn ordinals_for(applicability: PerfCellApplicability) -> Vec<usize> {
        PlanContract::reconstruct(&test_plan())
            .expect("test plan contract")
            .cells
            .iter()
            .enumerate()
            .filter_map(|(ordinal, cell)| (cell.applicability == applicability).then_some(ordinal))
            .collect()
    }

    fn complete_shards_with_prefix(prefix: &str) -> Vec<AuthorityBoundTestArtifact> {
        let ordinals = runnable_ordinals();
        let midpoint = ordinals.len() / 2;
        [
            shard(
                &ordinals[..midpoint],
                &format!("{prefix}-a"),
                &format!("{prefix}-runner-a"),
                None,
                TestIdentity::PRIMARY,
            ),
            shard(
                &ordinals[midpoint..],
                &format!("{prefix}-b"),
                &format!("{prefix}-runner-b"),
                None,
                TestIdentity::PRIMARY,
            ),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn complete_shards() -> Vec<AuthorityBoundTestArtifact> {
        static SHARDS: OnceLock<Vec<AuthorityBoundTestArtifact>> = OnceLock::new();
        SHARDS
            .get_or_init(|| complete_shards_with_prefix("complete"))
            .clone()
    }

    fn reseal_assembly(assembly: &mut PerfEvidenceAssemblyArtifact) {
        assembly.assembly_sha256.clear();
        assembly.assembly_sha256 = assembly.recomputed_sha256().expect("assembly reseal");
    }

    fn assert_incompatible_field(
        result: Result<PerfEvidenceAssemblyArtifact, PerfEvidenceAssemblyError>,
        expected: &'static str,
    ) {
        let error = result.expect_err("hostile shard must fail closed");
        assert!(
            matches!(
                &error,
                PerfEvidenceAssemblyError::IncompatibleShard { field, .. }
                    if *field == expected
            ),
            "expected incompatible field {expected:?}, got {error}"
        );
    }

    #[test]
    fn canonical_matrix_and_input_permutations_are_byte_identical() {
        let shards = complete_shards();
        let forward = assemble_test(shards.clone(), Vec::new()).expect("forward assembly");
        let reverse = assemble_test(shards.into_iter().rev().collect(), Vec::new())
            .expect("reverse assembly");

        assert_eq!(
            forward.to_json_pretty().unwrap(),
            reverse.to_json_pretty().unwrap()
        );
        assert_eq!(forward.matrix_manifest.cells.len(), 74);
        assert_eq!(forward.counts.canonical_cells, 74);
        assert_eq!(forward.counts.measured_cells, 58);
        assert_eq!(forward.counts.not_applicable_cells, 16);
        assert!(forward.is_complete());
        assert!(forward.has_full_plan_coverage());
        forward.require_complete().expect("complete coverage");
        assert_h2_lifecycle_boundary(&forward);
        forward.verify_integrity().expect("verified assembly");
    }

    #[test]
    fn derived_matrix_manifest_independently_round_trips_and_rejects_tamper() {
        let assembly = assemble_test(complete_shards(), Vec::new()).unwrap();
        let manifest = assembly.matrix_manifest();
        let bytes = manifest.to_json_pretty().expect("canonical matrix bytes");
        assert_eq!(
            PerfEvidenceAssemblyMatrixManifest::from_verified_slice(&bytes)
                .expect("independent matrix reload"),
            *manifest
        );

        let mut tampered = manifest.clone();
        tampered.cells[0].cell_id = "QG-1/fabricated/cell".to_owned();
        assert!(matches!(
            PerfEvidenceAssemblyMatrixManifest::from_verified_slice(
                &tampered.to_json_pretty().expect("tampered matrix bytes")
            ),
            Err(PerfEvidenceAssemblyError::MatrixManifestMismatch { .. })
        ));
    }

    #[test]
    fn missing_diagnostics_block_adjudication_before_h2_lifecycle_no_claims() {
        let ordinals = ordinals_for(PerfCellApplicability::Required);
        let midpoint = ordinals.len() / 2;
        let shards = vec![
            shard(
                &ordinals[..midpoint],
                "required-only-a",
                "required-only-runner-a",
                None,
                TestIdentity::PRIMARY,
            ),
            shard(
                &ordinals[midpoint..],
                "required-only-b",
                "required-only-runner-b",
                None,
                TestIdentity::PRIMARY,
            ),
        ];
        let assembly = assemble_test(shards, Vec::new()).expect("Required-only assembly");

        assert!(assembly.is_complete());
        assert!(!assembly.has_full_plan_coverage());
        assert!(assembly.missing_required_cell_ids().is_empty());
        assert_eq!(assembly.missing_diagnostic_cell_ids().len(), 2);
        assert_eq!(
            assembly.readiness(),
            PerfEvidenceAssemblyReadiness::NoClaimIncomplete
        );
        assert!(matches!(
            assembly.require_full_plan_coverage(),
            Err(PerfEvidenceAssemblyError::IncompleteDiagnosticCoverage { missing: 2, .. })
        ));
        assert!(matches!(
            assembly.require_adjudicable(),
            Err(PerfEvidenceAssemblyError::IncompleteDiagnosticCoverage { missing: 2, .. })
        ));
    }

    #[test]
    fn invalid_diagnostic_is_retained_beside_h2_lifecycle_no_claims() {
        let ordinals = runnable_ordinals();
        let diagnostic = ordinals_for(PerfCellApplicability::Diagnostic)[0];
        let midpoint = ordinals.len() / 2;
        let shards = vec![
            shard(
                &ordinals[..midpoint],
                "invalid-diagnostic-a",
                "invalid-diagnostic-runner-a",
                Some(diagnostic),
                TestIdentity::PRIMARY,
            ),
            shard(
                &ordinals[midpoint..],
                "invalid-diagnostic-b",
                "invalid-diagnostic-runner-b",
                Some(diagnostic),
                TestIdentity::PRIMARY,
            ),
        ];
        let assembly = assemble_test(shards, Vec::new())
            .expect("assembly retaining an invalid Diagnostic cell");

        assert!(assembly.has_full_plan_coverage());
        assert!(assembly.non_adjudicable_cells().iter().any(|cell| {
            cell.ordinal() == diagnostic && cell.role() == EvidenceRole::Diagnostic
        }));
        assert_h2_lifecycle_boundary(&assembly);
    }

    #[test]
    fn arbitrary_partial_source_no_claim_cannot_be_laundered_into_readiness() {
        let ordinals = runnable_ordinals();
        let midpoint = ordinals.len() / 2;
        let shards = vec![
            shard_with_partial_code(
                &ordinals[..midpoint],
                "source-no-claim-a",
                "source-no-claim-runner-a",
                None,
                TestIdentity::PRIMARY,
                "evidence.incomplete_gate_selection",
            ),
            shard(
                &ordinals[midpoint..],
                "source-no-claim-b",
                "source-no-claim-runner-b",
                None,
                TestIdentity::PRIMARY,
            ),
        ];
        let assembly = assemble_test(shards, Vec::new()).expect("source NoClaim remains durable");

        assert!(assembly.is_complete());
        assert!(
            assembly
                .non_adjudicable_sources()
                .iter()
                .any(|source| { source.reason().code == "evidence.incomplete_gate_selection" })
        );
        assert_eq!(
            assembly.readiness(),
            PerfEvidenceAssemblyReadiness::NoClaimInvalidEvidence
        );
        assert!(matches!(
            assembly.require_adjudicable(),
            Err(PerfEvidenceAssemblyError::NonAdjudicableAssembly { .. })
        ));
    }

    #[test]
    fn diagnostic_only_source_no_claim_is_retained_beside_h2_lifecycle_no_claims() {
        let required = ordinals_for(PerfCellApplicability::Required);
        let diagnostics = ordinals_for(PerfCellApplicability::Diagnostic);
        let midpoint = required.len() / 2;
        let shards = vec![
            shard(
                &required[..midpoint],
                "diagnostic-source-required-a",
                "diagnostic-source-required-runner-a",
                None,
                TestIdentity::PRIMARY,
            ),
            shard(
                &required[midpoint..],
                "diagnostic-source-required-b",
                "diagnostic-source-required-runner-b",
                None,
                TestIdentity::PRIMARY,
            ),
            shard_with_partial_code(
                &diagnostics,
                "diagnostic-source-only",
                "diagnostic-source-only-runner",
                None,
                TestIdentity::PRIMARY,
                "evidence.gate_without_required_cells",
            ),
        ];
        let assembly =
            assemble_test(shards, Vec::new()).expect("diagnostic source NoClaim remains durable");

        assert!(assembly.has_full_plan_coverage());
        assert!(
            assembly
                .non_adjudicable_sources()
                .iter()
                .any(|source| { source.reason().code == "evidence.gate_without_required_cells" })
        );
        assert_h2_lifecycle_boundary(&assembly);
    }

    #[test]
    fn zero_required_profile_is_explicit_durable_no_claim() {
        assert_eq!(
            derive_readiness(0, 0, 0, 0, 0),
            PerfEvidenceAssemblyReadiness::NoClaimNoRequiredCells
        );
    }

    #[test]
    fn incomplete_receipt_is_durable_and_never_advances_latest() {
        let first = complete_shards().remove(0);
        let assembly = assemble_test(vec![first], Vec::new()).expect("durable partial assembly");
        assert!(!assembly.is_complete());
        assert_eq!(
            assembly.readiness(),
            PerfEvidenceAssemblyReadiness::NoClaimIncomplete
        );
        assert!(matches!(
            assembly.require_adjudicable(),
            Err(PerfEvidenceAssemblyError::IncompleteAssembly { .. })
        ));

        let directory = private_tempdir("assembly tempdir");
        let path = assembly
            .write_atomic(directory.path())
            .expect("persist assembly");
        let basename = path.file_name().unwrap().to_string_lossy();
        assert!(basename.contains("trj-zen3-5995wx"));
        assert!(basename.contains("physical-64"));
        assert!(!basename.contains("latest"));
        assert_eq!(
            PerfEvidenceAssemblyArtifact::load_verified(&path).unwrap(),
            assembly
        );
    }

    #[test]
    fn content_addressed_publication_is_idempotent_and_never_clobbers() {
        let assembly = assemble_test(complete_shards(), Vec::new()).unwrap();
        let directory = private_tempdir("assembly publication tempdir");
        let first = assembly
            .write_atomic(directory.path())
            .expect("first publication");
        let second = assembly
            .write_atomic(directory.path())
            .expect("idempotent publication");
        assert_eq!(first, second);

        let collision_dir = private_tempdir("assembly collision tempdir");
        let collision = collision_dir.path().join(
            assembly
                .destination_basename()
                .expect("destination basename"),
        );
        fs::write(&collision, b"not an assembly").expect("seed collision");
        assert!(assembly.write_atomic(collision_dir.path()).is_err());
        assert_eq!(
            fs::read(collision).expect("collision remains"),
            b"not an assembly"
        );

        let hardlink_dir = private_tempdir("assembly hardlink tempdir");
        let victim = hardlink_dir.path().join("victim");
        fs::write(&victim, b"do not append").expect("seed hardlink victim");
        let mut pending_name = OsString::from(".");
        pending_name.push(
            assembly
                .destination_basename()
                .expect("hardlink destination basename"),
        );
        pending_name.push(".pending");
        fs::hard_link(&victim, hardlink_dir.path().join(pending_name))
            .expect("seed hardlinked pending path");
        assert!(assembly.write_atomic(hardlink_dir.path()).is_err());
        assert_eq!(
            fs::read(victim).expect("hardlink victim remains"),
            b"do not append",
            "resumable publication must never append through a hardlinked pending path"
        );
    }

    #[test]
    fn publication_rejects_output_and_final_path_aliases() {
        use std::os::unix::fs::{MetadataExt as _, symlink};

        let assembly = assemble_test(complete_shards(), Vec::new()).unwrap();
        let canonical_dir = private_tempdir("canonical assembly directory");
        let canonical_path = assembly
            .write_atomic(canonical_dir.path())
            .expect("publish canonical assembly fixture");
        let canonical_bytes = fs::read(&canonical_path).expect("canonical assembly bytes");
        let destination_name = assembly
            .destination_basename()
            .expect("destination basename");

        let new_output_parent = private_tempdir("new-output parent");
        let new_output = new_output_parent.path().join("assembly");
        let new_output_path = assembly
            .write_atomic(&new_output)
            .expect("securely create one missing output leaf");
        assert_eq!(new_output_path.parent(), Some(new_output.as_path()));
        let new_output_metadata = fs::symlink_metadata(&new_output).expect("new output metadata");
        assert!(new_output_metadata.is_dir());
        assert_eq!(new_output_metadata.mode() & 0o077, 0);

        let symlink_root = private_tempdir("symlink-root parent");
        let aliased_output = symlink_root.path().join("aliased-output");
        symlink(canonical_dir.path(), &aliased_output).expect("seed output-directory symlink");
        assert!(assembly.write_atomic(&aliased_output).is_err());
        assert_eq!(
            fs::read(&canonical_path).expect("canonical bytes after output alias rejection"),
            canonical_bytes
        );

        let final_symlink_dir = private_tempdir("final-symlink directory");
        symlink(
            &canonical_path,
            final_symlink_dir.path().join(&destination_name),
        )
        .expect("seed final symlink");
        assert!(assembly.write_atomic(final_symlink_dir.path()).is_err());
        assert_eq!(
            fs::read(&canonical_path).expect("canonical bytes after final symlink rejection"),
            canonical_bytes
        );

        let final_hardlink_dir = private_tempdir("final-hardlink directory");
        fs::hard_link(
            &canonical_path,
            final_hardlink_dir.path().join(&destination_name),
        )
        .expect("seed final hardlink");
        assert!(assembly.write_atomic(final_hardlink_dir.path()).is_err());
        assert_eq!(
            fs::read(&canonical_path).expect("canonical bytes after final hardlink rejection"),
            canonical_bytes
        );
    }

    #[test]
    fn complete_invalid_null_is_preserved_but_never_adjudicable() {
        let ordinals = runnable_ordinals();
        let midpoint = ordinals.len() / 2;
        let invalid_ordinal = ordinals_for(PerfCellApplicability::Required)[0];
        let shards = vec![
            shard(
                &ordinals[..midpoint],
                "invalid-a",
                "invalid-runner-a",
                Some(invalid_ordinal),
                TestIdentity::PRIMARY,
            ),
            shard(
                &ordinals[midpoint..],
                "invalid-b",
                "invalid-runner-b",
                None,
                TestIdentity::PRIMARY,
            ),
        ];
        let assembly = assemble_test(shards, Vec::new()).expect("authentic invalid-null assembly");
        assembly.require_complete().expect("coverage is complete");
        assert_eq!(
            assembly.readiness(),
            PerfEvidenceAssemblyReadiness::NoClaimInvalidEvidence
        );
        let error = assembly
            .require_adjudicable()
            .expect_err("invalid null cannot be adjudicated");
        let (cells, diagnostics) = match error {
            PerfEvidenceAssemblyError::NonAdjudicableAssembly {
                cells, diagnostics, ..
            } => (cells, diagnostics),
            _ => (usize::MAX, Vec::new()),
        };
        assert_eq!(cells, 1);
        assert_eq!(diagnostics[0].ordinal(), invalid_ordinal);
        assert_eq!(
            diagnostics[0].terminal_status(),
            EvidenceDecisionStatus::InvalidNull
        );
    }

    #[test]
    fn overlapping_shards_fail_closed() {
        let ordinals = runnable_ordinals();
        let midpoint = ordinals.len() / 2;
        let left = shard(
            &ordinals[..midpoint],
            "overlap-a",
            "overlap-runner-a",
            None,
            TestIdentity::PRIMARY,
        );
        let duplicate = shard(
            &ordinals[..midpoint],
            "overlap-b",
            "overlap-runner-b",
            None,
            TestIdentity::PRIMARY,
        );
        assert!(matches!(
            assemble_test(vec![left, duplicate], Vec::new()),
            Err(PerfEvidenceAssemblyError::OverlappingCell { .. })
        ));
    }

    #[test]
    fn nested_raw_tamper_cannot_be_hidden_by_an_outer_reseal() {
        let mut assembly = assemble_test(complete_shards(), Vec::new()).unwrap();
        let body = &mut assembly.source_shards[0].artifact.cells[0].body;
        assert!(matches!(body, EvidenceCellBody::Paired { .. }));
        if let EvidenceCellBody::Paired { paired, .. } = body {
            paired.effect_samples[0].ended_ns += 1;
        }
        assembly.source_shards[0].artifact.artifact_sha256.clear();
        let unsealed = serde_json::to_string_pretty(&assembly.source_shards[0].artifact).unwrap();
        assembly.source_shards[0].artifact.artifact_sha256 = sha256_hex(unsealed.as_bytes());
        reseal_assembly(&mut assembly);

        assert!(matches!(
            assembly.verify_integrity(),
            Err(PerfEvidenceAssemblyError::LocalPerf(_))
        ));
    }

    #[test]
    fn fabricated_matrix_projection_fails_after_inner_and_outer_reseal() {
        let mut assembly = assemble_test(complete_shards(), Vec::new()).unwrap();
        assembly.matrix_manifest.cells[0].spec.fixture = "fabricated/fixture".to_owned();
        assembly.matrix_manifest.matrix_manifest_sha256.clear();
        assembly.matrix_manifest.matrix_manifest_sha256 = assembly
            .matrix_manifest
            .recomputed_sha256()
            .expect("matrix reseal");
        reseal_assembly(&mut assembly);
        assert!(matches!(
            assembly.verify_integrity(),
            Err(PerfEvidenceAssemblyError::MatrixManifestMismatch { .. })
        ));
    }

    #[test]
    fn semantic_seal_excludes_run_ids_but_envelope_seal_retains_them() {
        let first = assemble_test(complete_shards_with_prefix("semantic-one"), Vec::new()).unwrap();
        let second =
            assemble_test(complete_shards_with_prefix("semantic-two"), Vec::new()).unwrap();
        assert_eq!(
            first.semantic_cell_set.semantic_cell_set_sha256,
            second.semantic_cell_set.semantic_cell_set_sha256
        );
        assert_ne!(first.assembly_sha256, second.assembly_sha256);
    }

    #[test]
    fn semantic_seal_is_independent_of_shard_partition_and_observed_size() {
        let split_sources = complete_shards_with_prefix("semantic-split");
        assert_eq!(
            split_sources[0].provenance.corpus.corpus_sha256,
            split_sources[1].provenance.corpus.corpus_sha256
        );
        assert_ne!(
            split_sources[0].provenance.corpus.document_count,
            split_sources[1].provenance.corpus.document_count,
            "disjoint source processes retain their actual measured corpus sizes"
        );
        assert_ne!(
            split_sources[0].provenance.machine.load_average_start,
            split_sources[1].provenance.machine.load_average_start,
            "hostile inputs must carry genuinely process-local load observations"
        );
        assert_ne!(
            split_sources[0].provenance.peak_rss, split_sources[1].provenance.peak_rss,
            "hostile inputs must carry genuinely process-local RSS observations"
        );
        let split = assemble_test(split_sources, Vec::new())
            .expect("assemble differently sized disjoint shards");

        let all_ordinals = runnable_ordinals();
        let single = assemble_test(
            vec![shard(
                &all_ordinals,
                "semantic-single",
                "semantic-single-runner",
                None,
                TestIdentity::PRIMARY,
            )],
            Vec::new(),
        )
        .expect("assemble one full-gate source");

        assert_eq!(
            split.semantic_cell_set.semantic_cell_set_sha256,
            single.semantic_cell_set.semantic_cell_set_sha256,
            "semantic evidence identity must not depend on process partitioning"
        );
        assert_ne!(split.assembly_sha256, single.assembly_sha256);
    }

    #[test]
    fn semantic_source_no_claims_union_cells_by_exact_reason() {
        let reason = EvidenceReason::new(
            "qg1.engine_lifecycle_unavailable",
            "engine-internal terminal lifecycle was not independently observed",
            crate::EvidenceSeverity::NoClaim,
        );
        let normalized = normalized_semantic_source_no_claims(&[
            PerfEvidenceAssemblyNoClaimSource {
                evidence_artifact_sha256: "a".repeat(64),
                run_id: "partial-a".to_owned(),
                cell_ids: vec!["QG-1/a".to_owned(), "QG-1/b".to_owned()],
                reason: reason.clone(),
            },
            PerfEvidenceAssemblyNoClaimSource {
                evidence_artifact_sha256: "b".repeat(64),
                run_id: "partial-b".to_owned(),
                cell_ids: vec!["QG-1/b".to_owned(), "QG-1/c".to_owned()],
                reason: reason.clone(),
            },
        ]);
        assert_eq!(normalized.len(), 1);
        assert_eq!(
            normalized[0].cell_ids,
            ["QG-1/a", "QG-1/b", "QG-1/c"].map(str::to_owned)
        );
        assert_eq!(normalized[0].reason, reason);
    }

    #[test]
    fn exact_byte_reload_rejects_unknown_fields_and_whitespace_aliases() {
        let assembly = assemble_test(complete_shards(), Vec::new()).unwrap();
        let mut whitespace = assembly.to_json_pretty().unwrap();
        whitespace.push(b'\n');
        assert!(matches!(
            PerfEvidenceAssemblyArtifact::from_verified_slice(&whitespace),
            Err(PerfEvidenceAssemblyError::Malformed { .. })
        ));

        let mut value = serde_json::to_value(&assembly).unwrap();
        value["aggregate_runner_receipt"] = serde_json::json!({"fabricated": true});
        assert!(matches!(
            PerfEvidenceAssemblyArtifact::from_verified_slice(
                &serde_json::to_vec_pretty(&value).unwrap()
            ),
            Err(PerfEvidenceAssemblyError::Malformed { .. })
        ));
    }

    #[test]
    fn stale_outer_schema_rejects_before_legacy_nested_shape_decoding() {
        let stale = serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": "quill-perf-evidence-assembly-v1",
            "source_shards": [{
                "threshold_artifact": {"schema_version": "quill-perf-artifact-v6"},
                "artifact": {"schema_version": "quill-perf-evidence-v4"},
            }],
        }))
        .expect("stale assembly probe");
        assert!(matches!(
            PerfEvidenceAssemblyArtifact::from_verified_slice(&stale),
            Err(PerfEvidenceAssemblyError::SchemaMismatch { ref found })
                if found == "quill-perf-evidence-assembly-v1"
        ));
    }

    #[test]
    fn threshold_join_rejects_row_and_law_scope_contradictions() {
        let artifact = normalize_h2_test_artifact(complete_shards().remove(0)).remove(0);
        let threshold = threshold_artifact_for(&artifact);
        verify_threshold_evidence_join(&threshold, &artifact)
            .expect("unaltered production threshold projection joins exact evidence");

        let mut row_contradiction = threshold.clone();
        row_contradiction.cells[0].engine.push_str("-substituted");
        assert!(matches!(
            verify_threshold_evidence_join(&row_contradiction, &artifact),
            Err(PerfEvidenceAssemblyError::InvalidAttemptBundle { ref reason })
                if reason == "threshold schema, identity, law scope, or cell projection contradicts bound evidence"
        ));

        let mut law_contradiction = threshold;
        law_contradiction.laws_attested = !law_contradiction.laws_attested;
        assert!(matches!(
            verify_threshold_evidence_join(&law_contradiction, &artifact),
            Err(PerfEvidenceAssemblyError::InvalidAttemptBundle { ref reason })
                if reason == "threshold schema, identity, law scope, or cell projection contradicts bound evidence"
        ));
    }

    #[test]
    fn attempt_loader_rejects_missing_log_symlink_hardlink_and_oversize() {
        use std::os::unix::fs::symlink;

        let artifact = normalize_h2_test_artifact(complete_shards().remove(0)).remove(0);
        let authority_refs = artifact.authority_refs();
        let valid = completed_test_attempt_directory(&artifact, &authority_refs);
        let receipt_bytes = fs::read(valid.path().join("QG-1.attempt.json"))
            .expect("exact completed attempt receipt");

        let missing_log = private_tempdir("missing-log attempt");
        fs::write(missing_log.path().join("QG-1.attempt.json"), &receipt_bytes)
            .expect("write receipt without its bound log");
        assert!(matches!(
            VerifiedLocalPerfAttemptBundle::load_verified(missing_log.path()),
            Err(PerfEvidenceAssemblyError::Io(ref error))
                if error.kind() == std::io::ErrorKind::NotFound
        ));

        let symlinked = private_tempdir("symlinked attempt");
        fs::write(symlinked.path().join("receipt-target"), &receipt_bytes)
            .expect("write symlink target");
        symlink("receipt-target", symlinked.path().join("QG-1.attempt.json"))
            .expect("seed receipt symlink");
        assert!(matches!(
            VerifiedLocalPerfAttemptBundle::load_verified(symlinked.path()),
            Err(PerfEvidenceAssemblyError::InconsistentAssembly { ref reason })
                if reason == "attempt input is a symlink or unsafe path alias"
        ));

        let hardlinked = private_tempdir("hardlinked attempt");
        let hardlink_target = hardlinked.path().join("receipt-target");
        fs::write(&hardlink_target, &receipt_bytes).expect("write hardlink target");
        fs::hard_link(
            &hardlink_target,
            hardlinked.path().join("QG-1.attempt.json"),
        )
        .expect("seed receipt hardlink");
        assert!(matches!(
            VerifiedLocalPerfAttemptBundle::load_verified(hardlinked.path()),
            Err(PerfEvidenceAssemblyError::InconsistentAssembly { ref reason })
                if reason == "assembly file must be an effective-user-owned regular single-link inode with the expected bounded length"
        ));

        let oversized = private_tempdir("oversized attempt");
        fs::write(
            oversized.path().join("QG-1.attempt.json"),
            vec![b'x'; PERF_ASSEMBLY_MAX_RECEIPT_BYTES + 1],
        )
        .expect("write oversized receipt");
        assert!(matches!(
            VerifiedLocalPerfAttemptBundle::load_verified(oversized.path()),
            Err(PerfEvidenceAssemblyError::InconsistentAssembly { ref reason })
                if reason == "assembly file must be an effective-user-owned regular single-link inode with the expected bounded length"
        ));

        let symlinked_artifacts = private_tempdir("symlinked artifacts attempt");
        for name in [
            "QG-1.attempt.json",
            "run.log",
            "QG-1.runner.json",
            "QG-1.artifacts.json",
            "QG-1.bound.evidence.json",
        ] {
            fs::write(
                symlinked_artifacts.path().join(name),
                fs::read(valid.path().join(name)).expect("read valid root attempt input"),
            )
            .expect("copy valid root attempt input");
        }
        symlink(
            valid.path().join("artifacts"),
            symlinked_artifacts.path().join("artifacts"),
        )
        .expect("seed artifact-directory symlink");
        assert!(matches!(
            VerifiedLocalPerfAttemptBundle::load_verified(symlinked_artifacts.path()),
            Err(PerfEvidenceAssemblyError::InconsistentAssembly { ref reason })
                if reason == "attempt artifact directory is a symlink or unsafe path alias"
        ));
    }

    #[test]
    fn attempt_loader_rejects_substitution_but_failed_receipts_ignore_orphans() {
        let artifact = normalize_h2_test_artifact(complete_shards().remove(0)).remove(0);
        let authority_refs = artifact.authority_refs();
        let valid = completed_test_attempt_directory(&artifact, &authority_refs);
        let substituted = private_tempdir("substituted completed attempt");
        let substituted_artifacts = substituted.path().join("artifacts");
        fs::create_dir(&substituted_artifacts).expect("create substituted artifact directory");
        fs::set_permissions(&substituted_artifacts, fs::Permissions::from_mode(0o700))
            .expect("make substituted artifact directory private");
        for name in [
            "QG-1.attempt.json",
            "run.log",
            "QG-1.runner.json",
            "QG-1.artifacts.json",
        ] {
            fs::write(
                substituted.path().join(name),
                fs::read(valid.path().join(name)).expect("read exact completed root input"),
            )
            .expect("write substituted root input");
        }
        fs::write(
            substituted.path().join("QG-1.bound.evidence.json"),
            fs::read(valid.path().join("artifacts/QG-1.evidence.json"))
                .expect("read exact prebinding evidence"),
        )
        .expect("substitute prebinding bytes for bound evidence");
        for name in ["QG-1.json", "QG-1.evidence.json"] {
            fs::write(
                substituted_artifacts.join(name),
                fs::read(valid.path().join("artifacts").join(name))
                    .expect("read exact completed child input"),
            )
            .expect("write substituted child input");
        }
        assert!(VerifiedLocalPerfAttemptBundle::load_verified(substituted.path()).is_err());

        let run_log_bytes = format!("runner-log:{}", artifact.provenance.run_id).into_bytes();
        let prebinding_bytes = artifact
            .reconstructed_prebinding_bytes()
            .expect("reconstructed orphan prebinding evidence");
        let bound_bytes = canonical_evidence_bytes(&artifact).expect("orphan bound evidence");
        let threshold_bytes = canonical_threshold_bytes(&threshold_artifact_for(&artifact))
            .expect("orphan threshold evidence");
        let failed_receipt = crate::local_perf_runner::failed_attempt_receipt_for_test(
            &artifact,
            Some(&artifact.cells[0].spec.fixture),
            &run_log_bytes,
            &threshold_bytes,
            &prebinding_bytes,
            &bound_bytes,
            23,
        );
        let failed = private_tempdir("failed attempt with completed orphans");
        fs::write(failed.path().join("QG-1.attempt.json"), failed_receipt)
            .expect("write failed receipt");
        fs::write(failed.path().join("run.log"), run_log_bytes).expect("write failed run log");
        for name in [
            "QG-1.runner.json",
            "QG-1.artifacts.json",
            "QG-1.bound.evidence.json",
        ] {
            fs::write(
                failed.path().join(name),
                fs::read(valid.path().join(name)).expect("read orphan completed root input"),
            )
            .expect("write orphan completed root input");
        }
        let failed_artifacts = failed.path().join("artifacts");
        fs::create_dir(&failed_artifacts).expect("create orphan artifact directory");
        fs::set_permissions(&failed_artifacts, fs::Permissions::from_mode(0o700))
            .expect("make orphan artifact directory private");
        for name in ["QG-1.json", "QG-1.evidence.json"] {
            fs::write(
                failed_artifacts.join(name),
                fs::read(valid.path().join("artifacts").join(name))
                    .expect("read orphan completed child input"),
            )
            .expect("write orphan completed child input");
        }
        let loaded = VerifiedLocalPerfAttemptBundle::load_verified(failed.path())
            .expect("failed receipt ignores completed orphans");
        assert_eq!(
            loaded.process().receipt().outcome(),
            LocalPerfAttemptOutcome::ExitedNonzero { code: 23 }
        );
        assert!(loaded.completed.is_none());
    }

    #[test]
    fn failed_attempt_is_sealed_preserved_and_contributes_no_cell() {
        let first = complete_shards().remove(0);
        let missing = ordinals_for(PerfCellApplicability::Required)
            .last()
            .copied()
            .unwrap();
        let contract = PlanContract::reconstruct(&test_plan()).unwrap();
        let failed_fixture = contract.cells[missing].spec.fixture.clone();
        let failed_ordinals = contract
            .cells
            .iter()
            .enumerate()
            .filter_map(|(ordinal, cell)| {
                (cell.applicability.is_runnable() && cell.spec.fixture == failed_fixture)
                    .then_some(ordinal)
            })
            .collect::<Vec<_>>();
        let failed_source = shard(
            &failed_ordinals,
            "failed-shard",
            "failed-shard-runner",
            None,
            TestIdentity::PRIMARY,
        );
        let failed = failed_test_attempt_bundle(
            failed_source
                .first()
                .expect("one fixture-exact failed source"),
            17,
        );
        let assembly =
            assemble_test(vec![first], vec![failed.clone()]).expect("assembly retaining failure");
        assert_eq!(assembly.failed_shards().len(), 1);
        assert_eq!(assembly.failed_shards()[0].process(), failed.process());
        assert_eq!(assembly.counts.failed_shards, 1);
        assert_eq!(
            assembly.counts.measured_cells,
            assembly
                .source_shards
                .iter()
                .map(|source| source.cell_ids.len())
                .sum::<usize>()
        );
        assert!(
            assembly
                .missing_required_cell_ids()
                .contains(&contract.cells[missing].cell_id)
        );
        assembly.verify_integrity().expect("failed attempt reloads");
    }

    #[test]
    fn incompatible_executable_identity_is_rejected() {
        let ordinals = runnable_ordinals();
        let midpoint = ordinals.len() / 2;
        let first = shard(
            &ordinals[..midpoint],
            "exec-a",
            "exec-runner-a",
            None,
            TestIdentity::PRIMARY,
        );
        let second = shard(
            &ordinals[midpoint..],
            "exec-b",
            "exec-runner-b",
            None,
            TestIdentity {
                executable: '9',
                ..TestIdentity::PRIMARY
            },
        );
        assert!(matches!(
            assemble_test(vec![first, second], Vec::new()),
            Err(PerfEvidenceAssemblyError::IncompatibleShard {
                field: "executable_sha256",
                ..
            })
        ));
    }

    #[test]
    fn compatible_receipts_still_reject_source_lock_corpus_policy_and_estimator_drift() {
        let ordinals = runnable_ordinals();
        let midpoint = ordinals.len() / 2;
        let first = shard(
            &ordinals[..midpoint],
            "compatibility-a",
            "compatibility-runner-a",
            None,
            TestIdentity::PRIMARY,
        );

        let source_drift = shard(
            &ordinals[midpoint..],
            "source-drift",
            "source-drift-runner",
            None,
            TestIdentity {
                source_revision: '7',
                ..TestIdentity::PRIMARY
            },
        );
        assert_incompatible_field(
            assemble_test(vec![first.clone(), source_drift], Vec::new()),
            "source_revision",
        );

        let lock_drift = shard(
            &ordinals[midpoint..],
            "lock-drift",
            "lock-drift-runner",
            None,
            TestIdentity {
                cargo_lock: '8',
                ..TestIdentity::PRIMARY
            },
        );
        assert_incompatible_field(
            assemble_test(vec![first.clone(), lock_drift], Vec::new()),
            "cargo_lock_sha256",
        );

        let corpus_drift = shard(
            &ordinals[midpoint..],
            "corpus-drift",
            "corpus-drift-runner",
            None,
            TestIdentity {
                corpus: '6',
                ..TestIdentity::PRIMARY
            },
        );
        assert_incompatible_field(
            assemble_test(vec![first.clone(), corpus_drift], Vec::new()),
            "corpus_identity",
        );

        let mut policy = test_policy();
        policy.warmup_rounds += 1;
        let policy_drift = shard_with_contract(
            &ordinals[midpoint..],
            "policy-drift",
            "policy-drift-runner",
            None,
            TestIdentity::PRIMARY,
            PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE,
            false,
            policy,
            &estimator_config(),
        );
        assert_incompatible_field(
            assemble_test(vec![first.clone(), policy_drift], Vec::new()),
            "evidence_policy",
        );

        let mut estimator = estimator_config();
        estimator.bootstrap_seed ^= 1;
        let estimator_drift = shard_with_contract(
            &ordinals[midpoint..],
            "estimator-drift",
            "estimator-drift-runner",
            None,
            TestIdentity::PRIMARY,
            PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE,
            false,
            test_policy(),
            &estimator,
        );
        assert_incompatible_field(
            assemble_test(vec![first, estimator_drift], Vec::new()),
            "estimator",
        );
    }

    #[test]
    fn profile_and_cross_class_substitution_fail_after_outer_reseal() {
        let original = assemble_test(complete_shards(), Vec::new()).unwrap();
        let substitutions = [
            MachineProfileKey::new(HardwareClassId::TrjZen35995wx, ExecutionProfileId::Smt2_128)
                .unwrap(),
            MachineProfileKey::new(HardwareClassId::M4Macos, ExecutionProfileId::Scheduler10)
                .unwrap(),
        ];
        for profile in substitutions {
            let mut substituted = original.clone();
            substituted.applicability_plan.profile = profile;
            reseal_assembly(&mut substituted);
            assert!(matches!(
                substituted.verify_integrity(),
                Err(PerfEvidenceAssemblyError::MatrixManifestMismatch { .. })
            ));
        }
    }

    #[test]
    fn not_applicable_measurement_injection_is_rejected_before_attempt_admission() {
        let mut source = complete_shards().remove(0);
        let contract = PlanContract::reconstruct(&test_plan()).unwrap();
        let not_applicable = contract
            .cells
            .iter()
            .find(|cell| cell.applicability == PerfCellApplicability::NotApplicable)
            .unwrap();
        source.cells[0].cell_id.clone_from(&not_applicable.cell_id);
        source.cells[0]
            .spec
            .fixture
            .clone_from(&not_applicable.spec.fixture);
        source.cells[0]
            .spec
            .metric
            .clone_from(&not_applicable.spec.metric);
        source.artifact_sha256.clear();
        let unsealed = serde_json::to_string_pretty(&source.artifact).unwrap();
        source.artifact_sha256 = sha256_hex(unsealed.as_bytes());

        let bytes =
            serde_json::to_vec_pretty(&source.artifact).expect("canonical hostile evidence bytes");
        assert!(matches!(
            PerfEvidenceArtifact::from_verified_slice(&bytes),
            Err(EvidenceArtifactError::InconsistentArtifact { ref reason })
                if reason.contains("not applicable to profile")
        ));
    }

    #[test]
    fn full_source_cannot_misuse_the_partial_shard_no_claim_code() {
        let ordinals = runnable_ordinals();
        let source = shard_with_contract(
            &ordinals,
            "false-partial",
            "false-partial-runner",
            None,
            TestIdentity::PRIMARY,
            PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE,
            true,
            test_policy(),
            &estimator_config(),
        );
        let assembly = assemble_test(vec![source], Vec::new()).unwrap();

        assert!(assembly.is_complete());
        assert!(
            assembly.non_adjudicable_sources().iter().any(|source| {
                source.reason().code == PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE
            })
        );
        assert_eq!(
            assembly.readiness(),
            PerfEvidenceAssemblyReadiness::NoClaimInvalidEvidence
        );
    }

    #[test]
    fn stale_normative_manifest_is_rejected_at_source_integrity_boundary() {
        let ordinals = runnable_ordinals();
        let mut sources = shard(
            &ordinals,
            "stale-manifest",
            "stale-manifest-runner",
            None,
            TestIdentity::PRIMARY,
        );
        assert_eq!(
            sources.len(),
            1,
            "full-gate fixture has one source artifact"
        );
        let mut stale = sources.pop().expect("full-gate source artifact");
        let expected_authorities = stale.expected_authorities.clone();
        stale.artifact.provenance.manifest_sha256 = "8".repeat(64);
        stale.artifact.artifact_sha256.clear();
        let unsealed =
            serde_json::to_string_pretty(&stale.artifact).expect("stale unsealed evidence");
        stale.artifact.artifact_sha256 = sha256_hex(unsealed.as_bytes());
        let authority_refs = expected_authorities.iter().collect::<Vec<_>>();
        assert!(matches!(
            stale
                .artifact
                .verify_integrity_against_qg1_authorities(&authority_refs),
            Err(EvidenceArtifactError::InvalidProvenance { ref reason })
                if reason.contains("manifest digest differs")
        ));
    }
}
