//! Same-binary Quill/Tantivy performance matrix for QG-1 through QG-10.
//!
//! The default invocation is deliberately a one-cell smoke slice. A release
//! evidence run selects one gate (and optionally one exact canonical fixture), then
//! lets Criterion self-cap that slice while this harness also emits the raw
//! per-gate JSON and human table required by the E0.6 manifests.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 rch exec -- env \
//!   QUILL_PERF_SCALE=full QUILL_PERF_GATE=QG-1 \
//!   QUILL_PERF_GIT_REV="$(git rev-parse HEAD)" \
//!   QUILL_PERF_OUTPUT_DIR=/tmp/quill-perf-qg1 \
//!   cargo bench -p frankensearch-quill-gauntlet \
//!     --features perf-harness --profile release-perf --bench perf_matrix
//! ```

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::hint::black_box;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use asupersync::{Cx, runtime::Runtime};
use criterion::Criterion;
use frankensearch_core::bench_support::BenchExecutableIdentity;
use frankensearch_core::{IndexableDocument, LexicalRead, LexicalWrite};
use frankensearch_lexical::{
    BenchmarkMaterializedWidth, BenchmarkRetainedTantivyReader, BenchmarkWriterJoinReceipt,
    BenchmarkWriterMode, BenchmarkWriterReceipt, SnippetConfig, TantivyIndex,
};
use frankensearch_quill::scribe::{FrankensearchTokenizer, TokenAnalyzer};
use frankensearch_quill::{
    Analyzer, CompactionPolicy, DEFAULT_SCHEMA, FieldDescriptor, FieldKind, QuillConfig,
    QuillIndex, SchemaDescriptor,
};
use frankensearch_quill_gauntlet::{
    BuildIdentity,
    ColdCacheEvidence,
    ComparatorConfig,
    ComparisonStatus,
    CorpusIdentity,
    CorpusManifest,
    CountState,
    DistributionSummary,
    EngineConcurrencyObservation,
    EngineObservation,
    EvidenceCell,
    EvidenceCellBody,
    EvidenceCellSpec,
    EvidencePolicy,
    EvidenceProvenance,
    EvidenceRole,
    ExecutionProfileId,
    HardwareClassId,
    MachineClassRegistry,
    MachineIdentity,
    MachineProfileKey,
    NativeTieKey,
    PERF_ARTIFACT_SCHEMA_VERSION,
    PERF_MIN_RUNS,
    PairedEstimatorConfig,
    PeakRssEvidence,
    PerfApplicabilityPlan,
    PerfCellApplicability,
    PerfCellResult,
    PerfCellSpec,
    PerfConcurrencyEngine,
    PerfConcurrencyObserver,
    PerfConcurrencyWitness,
    PerfCorpus,
    PerfEvidenceArtifact,
    PerfGate,
    PerfGateArtifact,
    PerfInputIdentity,
    PerfMatrixSpec,
    PerfMetricSemantics,
    PerfOperationScope,
    PerfQueryClass,
    PerfRawSample,
    PerfSampleArm,
    PerfSampleOrder,
    PerfSamplePhase,
    PerfSampleProvenance,
    PerfTopology,
    PositionMode,
    QG1_QUILL_ENGINE_ID,
    QG1_STREAM_ROLE_EFFECT,
    QG1_STREAM_ROLE_QUILL_NULL,
    QG1_STREAM_ROLE_TANTIVY_NULL,
    QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT,
    QG1_STREAM_ROLE_TANTIVY_PILOT_NULL,
    QG1_TANTIVY_ENGINE_ID,
    QG6_QUERY_GROUP_IDS,
    QG6_QUERY_GROUPS,
    Qg1AuthorityRegisterEntryV1,
    Qg1BatchCoverage,
    Qg1ExpectedAuthority,
    Qg1IncumbentScreenEvidence,
    Qg1LifecycleProducer,
    Qg1LifecycleWitness,
    Qg1SampleBinding,
    Qg1StartupHandshakeV1,
    Qg1TantivyBoundStream,
    Qg1TantivyDecisionStreamKind,
    Qg1TantivyIncumbentDecision,
    Qg1TantivyIncumbentPilot,
    Qg1TantivyIncumbentScreen,
    Qg1TantivyIncumbentScreenPlan,
    Qg1TantivySemanticContract,
    Qg1TantivyWriterMode,
    Qg6ArmRole,
    Qg6Comparison,
    Qg6Phase,
    Qg6PreparedExperiment,
    Qg6QuerySpec,
    Qg6SampleBinding,
    Qg6SampleOrder,
    Qg6SearchHit,
    Qg6SearchResult,
    Qg6SemanticContract,
    RankClass,
    RankedHit,
    ScoreEpsilonReason,
    SyntheticCorpus,
    SyntheticCorpusSpec,
    ZipfExponent,
    command_sha256_from_argv,
    // `estimate_paired_experiment` is deliberately NOT imported here: its only
    // callers are `cfg(test)` helpers, which already import it locally, and a
    // root import would be unused in the harness-false production build.
    compare_observations,
    estimate_paired_experiment_against_qg1_authority,
    machine_fingerprint,
    oracle_version_contract,
    peak_rss_bytes,
    perf_manifest_contract_sha256,
    preregister_qg1_tantivy_incumbents,
    seeded_balanced_pair_order,
    validate_matrix,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MANIFEST: &str = include_str!("../../../docs/contracts/quill-perf-gates.toml");
const CORPUS_SEED: u64 = 0x5155_494c_4c50_4552;
const VOCABULARY_SIZE: u32 = 8_192;
const MAX_DOCUMENT_BYTES: u32 = 4_096;
const FULL_BATCH_DOCUMENTS: usize = 5_000;
const SMOKE_BATCH_DOCUMENTS: usize = 250;
const FULL_SEGMENTS: usize = 10;
const SMOKE_SEGMENTS: usize = 4;
// The largest normative QG-6 corpus is 1M documents. Fetching the complete
// incumbent boundary group is preflight-only and is required to distinguish a
// true rank mismatch from a native-order substitution inside a large BM25 tie.
const QG6_TIE_EXPANSION_LIMIT: usize = 1_000_000;
const QG6_TIMED_SEARCHES_PER_SAMPLE: usize = 128;
const QG1_CORPUS_GENERATOR_REVISION: &str = "frankensearch-quill-qg1-synthetic-corpus-v1";
const QG1_TERMINAL_NO_CLAIM_CODE: &str = "qg1.terminal_fact_unproved";
const QG1_INCUMBENT_SCREEN_NO_CLAIM_CODE: &str = "qg1.incumbent_screen_incomplete";
const QG1_TIMING_DIAGNOSTIC_NO_CLAIM_CODE: &str = "qg1.continuous_timing_unbound_diagnostic";
const QG1_LIVE_STARTUP_DISCRIMINATOR_ENV: &str = "QUILL_PERF_QG1_LIVE_STARTUP_DISCRIMINATOR";
const QG1_LIVE_STARTUP_ORDINARY_MARKER: &[u8] = b"qg1-live-startup-work-after-ack\n";
#[cfg(test)]
const QG1_AUTHORITY_SUBPROCESS_ENV: &str = "QUILL_PERF_QG1_AUTHORITY_SUBPROCESS";
#[cfg(test)]
const QG1_AUTHORITY_WORK_MARKER: &[u8] = b"\x1eQG1-WORK-AFTER-ACK\x1f";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Qg1LiveStartupDiscriminatorMode {
    Parent,
    Child,
    Preamble,
    NonQg,
}

fn qg1_live_startup_discriminator_mode() -> Option<Qg1LiveStartupDiscriminatorMode> {
    match std::env::var(QG1_LIVE_STARTUP_DISCRIMINATOR_ENV).as_deref() {
        Err(std::env::VarError::NotPresent) => None,
        Ok("parent") => Some(Qg1LiveStartupDiscriminatorMode::Parent),
        Ok("child") => Some(Qg1LiveStartupDiscriminatorMode::Child),
        Ok("preamble") => Some(Qg1LiveStartupDiscriminatorMode::Preamble),
        Ok("non-qg") => Some(Qg1LiveStartupDiscriminatorMode::NonQg),
        Ok(other) => panic!(
            "{QG1_LIVE_STARTUP_DISCRIMINATOR_ENV} must be parent, child, preamble, or non-qg; got {other:?}"
        ),
        Err(error) => panic!("{QG1_LIVE_STARTUP_DISCRIMINATOR_ENV} is not valid Unicode: {error}"),
    }
}

/// The remote parent sets both values together. A stray handshake value on a
/// non-QG invocation is ignored rather than causing a zero-register protocol.
fn qg1_exact_startup_handshake_for_selected_gate() -> bool {
    std::env::var(Qg1StartupHandshakeV1::ENV).as_deref() == Ok(Qg1StartupHandshakeV1::MODE)
        && std::env::var("QUILL_PERF_GATE").as_deref() == Ok(PerfGate::Qg1.label())
}

/// Hash the executing image before benchmark construction without emitting to
/// stdout. Exact QG-1 runs reserve stdout offset zero for the control protocol.
fn hash_bench_elf_sha256_silently() -> io::Result<BenchExecutableIdentity> {
    let path = std::env::current_exe()?;
    let executable = fs::read(&path)?;
    Ok(BenchExecutableIdentity {
        sha256: lower_hex(&Sha256::digest(&executable)),
        bytes: executable.len(),
        path,
    })
}

/// Emit the canonical benchmark identity only after QG-1 has received the one
/// final parent ACK. Non-handshake invocations retain the historical line-one
/// identity behavior from `main`.
fn emit_bench_elf_sha256(identity: &BenchExecutableIdentity) {
    println!(
        "bench_elf_sha256={} ({} bytes) {}",
        identity.sha256,
        identity.bytes,
        identity.path.display()
    );
}

fn qg1_tail_document_id(document_count: u64) -> String {
    let tail_ordinal = document_count
        .checked_sub(1)
        .expect("QG-1 terminal tail requires one or more documents");
    format!("synthetic-{tail_ordinal:08}")
}

// Frozen by a strict-remote full replay on 2026-07-31. The retired per-shard
// all-count replay generated 4,222,000 documents and took 326,401 ms in the
// unoptimized audit binary. Each full-scale producer now validates only the
// selected count against these full-universe pins; H4 requires assembled
// coverage of every Applicable/Required canonical cell before admitting a claim.
const QG1_FULL_PREFIX_IDENTITY_PINS: [(u64, &str, &str); 4] = [
    (
        500,
        "16b56b9704cfd2234a3fa8ca9fcfce1c935dd8ebd3f20c820e3212a684a7aeb1",
        "59188638fb211394e8c1c3d98a28a2cf3790400de1f229a5c6e1b10b100ee5a8",
    ),
    (
        5_000,
        "4886e04bb07825b130f3ad24801738759cc9d6e63af5adb663cab94a45155e0f",
        "72d977c424bc2f1ab1b08b4fc210dcff7dfd9139100ca59f741759be9856d4fb",
    ),
    (
        50_000,
        "a4cdb819886a56944316cf726237eb4d1216e243e2fdea94b5a08c5bddd266a0",
        "21f76704040e4f2f2cd4d1a0f2c3e261bbb5ca5e86f9e23edca6ed718ec98cfd",
    ),
    (
        1_000_000,
        "0a77def1cf79d6e576bf782250158b09ac49a824796b5ce6e8cee84b4a231d70",
        "b9840b4df07535f8908563bfc6b9c627f7e27edbafd6cbc6483a32a70d3c9f76",
    ),
];

/// Wire-stable producer diagnostic accepted by the H4 assembler only when the
/// retained source artifact is an actual proper subset of the runnable gate.
pub const QG1_PARTIAL_SHARD_NO_CLAIM_CODE: &str = "qg1.partial_shard";
const QG1_PARTIAL_SHARD_NO_CLAIM_DETAIL: &str = "the invocation retained one immutable partial QG-1 shard; this source artifact cannot \
     support a publication or ratchet claim until exact disjoint assembly proves full coverage";

#[derive(Deserialize)]
struct GateManifest {
    gate: BTreeMap<String, GateManifestEntry>,
}

#[derive(Deserialize)]
struct GateManifestEntry {
    name: String,
    fixture: String,
    target: String,
    activated: bool,
}

#[derive(Deserialize)]
struct Qg6Manifest {
    gate: BTreeMap<String, Qg6ManifestEntry>,
}

#[derive(Deserialize)]
struct Qg6ManifestEntry {
    queries_per_class: Option<usize>,
}

fn qg6_queries_per_class(manifest: &str) -> Result<usize, String> {
    let manifest = toml::from_str::<Qg6Manifest>(manifest).map_err(|error| error.to_string())?;
    manifest
        .gate
        .get("QG-6")
        .and_then(|gate| gate.queries_per_class)
        .ok_or_else(|| "gate.QG-6.queries_per_class is missing".to_owned())
}

fn validate_manifest_gate_contract(manifest: &str) -> Result<(), String> {
    let manifest = toml::from_str::<GateManifest>(manifest).map_err(|error| error.to_string())?;
    for gate in PerfGate::ALL {
        let label = gate.label();
        let policy = manifest
            .gate
            .get(label)
            .ok_or_else(|| format!("manifest is missing gate.{label}"))?;
        for (field, value) in [
            ("name", policy.name.as_str()),
            ("fixture", policy.fixture.as_str()),
            ("target", policy.target.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("manifest gate.{label}.{field} is empty"));
            }
        }
        let _activated = policy.activated;
    }
    for label in manifest.gate.keys() {
        if !PerfGate::ALL
            .iter()
            .any(|gate| gate.label() == label.as_str())
        {
            return Err(format!("manifest defines unexpected gate.{label}"));
        }
    }
    Ok(())
}

fn validate_qg6_queries_per_class(manifest: &str) -> Result<(), String> {
    let observed = qg6_queries_per_class(manifest)?;
    if observed == QG6_QUERY_GROUPS {
        Ok(())
    } else {
        Err(format!(
            "gate.QG-6.queries_per_class={observed} differs from runner constant \
             {QG6_QUERY_GROUPS}"
        ))
    }
}

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);
static LIFECYCLE_RECEIPT_COUNTER: AtomicU64 = AtomicU64::new(0);
static LIFECYCLE_RECEIPTS: OnceLock<Mutex<Vec<serde_json::Value>>> = OnceLock::new();
static QG1_CONTINUOUS_TIMING_COUNTER: AtomicU64 = AtomicU64::new(0);
static QG1_CONTINUOUS_TIMING_RECEIPTS: OnceLock<Mutex<Vec<Qg1ContinuousTimingRecord>>> =
    OnceLock::new();
static CONCURRENCY_OBSERVATIONS: OnceLock<
    Mutex<BTreeMap<(String, String), ConcurrencyAccumulator>>,
> = OnceLock::new();
static COLD_CACHE_OBSERVATIONS: OnceLock<Mutex<BTreeMap<String, ColdCacheAccumulator>>> =
    OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct ConcurrencyAccumulator {
    count: usize,
    min: usize,
    max: usize,
}

/// Aggregate the per-arm cache-eviction witnesses before admitting a QG-9 cell.
#[derive(Debug, Clone, Copy, Default)]
struct ColdCacheAccumulator {
    quill_successes: usize,
    quill_failures: usize,
    tantivy_successes: usize,
    tantivy_failures: usize,
}

fn record_cold_cache_eviction(
    spec: &PerfCellSpec,
    arm: EngineArm,
    eviction: Result<usize, String>,
) {
    let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
    // Classify (and report) BEFORE taking the lock: the assertion and the
    // stderr write do not need the mutex, and holding it across them is the
    // contention clippy::significant_drop_tightening is pointing at.
    let succeeded = match eviction {
        Ok(file_count) => {
            assert!(
                file_count > 0,
                "QG-9 cache eviction accepted an empty index"
            );
            true
        }
        Err(error) => {
            eprintln!(
                "[quill-qg9-cold-cache] arm={} eviction_unverified={error}",
                arm.label()
            );
            false
        }
    };
    let mut observations = COLD_CACHE_OBSERVATIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock cold-cache observations");
    record_cold_cache_outcome(observations.entry(cell_id).or_default(), arm, succeeded);
}

/// Apply one classified eviction witness to its per-cell accumulator.
///
/// Split out so the mutex guard above spans exactly one statement.
fn record_cold_cache_outcome(entry: &mut ColdCacheAccumulator, arm: EngineArm, succeeded: bool) {
    let (successes, failures) = match arm {
        EngineArm::Quill => (&mut entry.quill_successes, &mut entry.quill_failures),
        EngineArm::Tantivy => (&mut entry.tantivy_successes, &mut entry.tantivy_failures),
    };
    if succeeded {
        *successes = successes.saturating_add(1);
    } else {
        *failures = failures.saturating_add(1);
    }
}

fn cold_cache_evidence(accumulator: ColdCacheAccumulator) -> ColdCacheEvidence {
    let verified = accumulator.quill_successes > 0
        && accumulator.tantivy_successes > 0
        && accumulator.quill_failures == 0
        && accumulator.tantivy_failures == 0;
    let procedure = if verified {
        "fresh child process; successful posix_fadvise(POSIX_FADV_DONTNEED) on every regular index file before each open"
    } else {
        "fresh child process used, but at least one arm lacked a successful posix_fadvise(POSIX_FADV_DONTNEED) eviction witness"
    };
    ColdCacheEvidence {
        procedure: procedure.to_owned(),
        verified,
    }
}

fn take_cold_cache_evidence(spec: &PerfCellSpec) -> Option<ColdCacheEvidence> {
    (spec.gate == PerfGate::Qg9).then(|| {
        let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
        let accumulator = COLD_CACHE_OBSERVATIONS
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .expect("lock cold-cache observations")
            .remove(&cell_id)
            .expect("missing QG-9 cold-cache eviction witness");
        cold_cache_evidence(accumulator)
    })
}

fn record_concurrency(spec: &PerfCellSpec, arm: EngineArm, observed_threads: usize) {
    assert!(observed_threads > 0, "observed engine pool width is zero");
    let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
    let mut observations = CONCURRENCY_OBSERVATIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock concurrency observations");
    let entry = observations
        .entry((cell_id, arm.label().to_owned()))
        .or_insert(ConcurrencyAccumulator {
            count: 0,
            min: observed_threads,
            max: observed_threads,
        });
    entry.count = entry.count.saturating_add(1);
    entry.min = entry.min.min(observed_threads);
    entry.max = entry.max.max(observed_threads);
    drop(observations);
}

fn take_concurrency_witness(spec: &PerfCellSpec) -> Option<PerfConcurrencyWitness> {
    let required = spec.gate == PerfGate::Qg8
        || (spec.gate == PerfGate::Qg1 && spec.metric != "tokenize_docs_per_second");
    if !required {
        return None;
    }
    let configured_threads = spec.threads.expect("scaling cell thread width");
    let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
    let mut observations = CONCURRENCY_OBSERVATIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock concurrency observations");
    let make_observation = |arm: EngineArm,
                            engine: PerfConcurrencyEngine,
                            observer: PerfConcurrencyObserver,
                            observations: &mut BTreeMap<
        (String, String),
        ConcurrencyAccumulator,
    >| {
        let observed = observations
            .remove(&(cell_id.clone(), arm.label().to_owned()))
            .unwrap_or_else(|| panic!("missing {} concurrency witness for {cell_id}", arm.label()));
        EngineConcurrencyObservation {
            engine,
            observer,
            observation_count: observed.count,
            min_observed_worker_pool_threads: observed.min,
            max_observed_worker_pool_threads: observed.max,
        }
    };
    Some(PerfConcurrencyWitness {
        configured_threads,
        observations: vec![
            make_observation(
                EngineArm::Quill,
                PerfConcurrencyEngine::Quill,
                PerfConcurrencyObserver::RayonCurrentPoolWidth,
                &mut observations,
            ),
            make_observation(
                EngineArm::Tantivy,
                PerfConcurrencyEngine::Tantivy,
                PerfConcurrencyObserver::TantivyWriterConstruction,
                &mut observations,
            ),
        ],
    })
}

fn discard_concurrency_observations(spec: &PerfCellSpec) {
    let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
    let mut observations = CONCURRENCY_OBSERVATIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock concurrency observations");
    observations.remove(&(cell_id.clone(), EngineArm::Quill.label().to_owned()));
    observations.remove(&(cell_id, EngineArm::Tantivy.label().to_owned()));
}

const NO_POSITION_FIELDS: [FieldDescriptor; 5] = [
    FieldDescriptor {
        id: 0,
        name: "id",
        kind: FieldKind::Keyword,
        stored: true,
    },
    FieldDescriptor {
        id: 1,
        name: "content",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 2,
        name: "title",
        kind: FieldKind::Text {
            analyzer: Analyzer::FrankensearchDefault,
            positions: false,
        },
        stored: true,
    },
    FieldDescriptor {
        id: 3,
        name: "metadata_json",
        kind: FieldKind::StoredOnly,
        stored: true,
    },
    FieldDescriptor {
        id: 4,
        name: "ord",
        kind: FieldKind::U64 {
            indexed: false,
            fast: true,
        },
        stored: true,
    },
];

const NO_POSITION_SCHEMA: SchemaDescriptor = SchemaDescriptor {
    name: "frankensearch-default-no-positions-v1",
    fields: &NO_POSITION_FIELDS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixScale {
    Smoke,
    Full,
}

impl MatrixScale {
    fn from_env() -> Self {
        match std::env::var("QUILL_PERF_SCALE").as_deref() {
            Ok("full") => Self::Full,
            Ok("smoke") | Err(_) => Self::Smoke,
            Ok(other) => panic!("QUILL_PERF_SCALE must be smoke or full, got {other:?}"),
        }
    }

    const fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }

    const fn document_count(self, requested: u64) -> u64 {
        match self {
            Self::Full => requested,
            Self::Smoke => {
                if requested < 500 {
                    requested
                } else {
                    500
                }
            }
        }
    }

    const fn batch_documents(self) -> usize {
        match self {
            Self::Smoke => SMOKE_BATCH_DOCUMENTS,
            Self::Full => FULL_BATCH_DOCUMENTS,
        }
    }

    const fn segments(self) -> usize {
        match self {
            Self::Smoke => SMOKE_SEGMENTS,
            Self::Full => FULL_SEGMENTS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RunnerPlanClaims {
    gate: PerfGate,
    hardware_class: HardwareClassId,
    execution_profile: ExecutionProfileId,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
    rayon_num_threads: u64,
    applicability_plan_schema_version: String,
    applicability_plan_sha256: String,
    gate_matrix_contract_sha256: String,
    profile_contract_sha256: String,
    registry_schema_version: String,
    registry_sha256: String,
}

impl RunnerPlanClaims {
    fn from_env() -> Result<Self, String> {
        let gate_text = required_env("QUILL_PERF_GATE")?;
        let gate = gate_text
            .parse::<PerfGate>()
            .map_err(|error| format!("QUILL_PERF_GATE is invalid: {error}"))?;
        if gate_text != gate.label() {
            return Err(format!(
                "QUILL_PERF_GATE must use canonical spelling {:?}, got {gate_text:?}",
                gate.label()
            ));
        }
        Ok(Self {
            gate,
            hardware_class: parse_hardware_class_id(&required_env("QUILL_PERF_HARDWARE_CLASS")?)?,
            execution_profile: parse_execution_profile_id(&required_env(
                "QUILL_PERF_EXECUTION_PROFILE",
            )?)?,
            execution_capacity: canonical_positive_u64_env("QUILL_PERF_EXECUTION_CAPACITY")?,
            max_exercised_cell_width: canonical_positive_u64_env(
                "QUILL_PERF_MAX_EXERCISED_CELL_WIDTH",
            )?,
            rayon_num_threads: canonical_positive_u64_env("RAYON_NUM_THREADS")?,
            applicability_plan_schema_version: required_env(
                "QUILL_PERF_APPLICABILITY_PLAN_SCHEMA_VERSION",
            )?,
            applicability_plan_sha256: required_env("QUILL_PERF_APPLICABILITY_PLAN_SHA256")?,
            gate_matrix_contract_sha256: required_env("QUILL_PERF_GATE_MATRIX_CONTRACT_SHA256")?,
            profile_contract_sha256: required_env("QUILL_PERF_PROFILE_CONTRACT_SHA256")?,
            registry_schema_version: required_env("QUILL_PERF_REGISTRY_SCHEMA_VERSION")?,
            registry_sha256: required_env("QUILL_PERF_REGISTRY_SHA256")?,
        })
    }
}

#[derive(Debug, Clone)]
struct RunnerApplicabilityContext {
    profile: MachineProfileKey,
    plan: PerfApplicabilityPlan,
    execution_capacity: u64,
    max_exercised_cell_width: u64,
}

impl RunnerApplicabilityContext {
    fn reconstruct(matrix: &PerfMatrixSpec, claims: &RunnerPlanClaims) -> Result<Self, String> {
        let profile = MachineProfileKey::new(claims.hardware_class, claims.execution_profile)
            .map_err(|error| format!("runner profile key is invalid: {error}"))?;
        let registry = MachineClassRegistry::frozen()
            .map_err(|error| format!("frozen machine registry is invalid: {error}"))?;
        let registered_profile = registry
            .execution_profile(profile)
            .map_err(|error| format!("runner profile is not registered: {error}"))?;
        let plan = matrix
            .applicability_plan(&registry, profile, claims.gate)
            .map_err(|error| format!("cannot reconstruct runner applicability plan: {error}"))?;
        plan.verify_against(matrix, &registry)
            .map_err(|error| format!("runner applicability plan does not recompute: {error}"))?;
        if plan.capacity_semantics != registered_profile.capacity_semantics() {
            return Err(
                "registry profile and applicability plan disagree on capacity semantics".to_owned(),
            );
        }

        verify_runner_claim(
            "QUILL_PERF_APPLICABILITY_PLAN_SCHEMA_VERSION",
            &claims.applicability_plan_schema_version,
            &plan.binding().schema_version,
        )?;
        verify_runner_claim(
            "QUILL_PERF_APPLICABILITY_PLAN_SHA256",
            &claims.applicability_plan_sha256,
            &plan.binding().applicability_plan_sha256,
        )?;
        verify_runner_claim(
            "QUILL_PERF_GATE_MATRIX_CONTRACT_SHA256",
            &claims.gate_matrix_contract_sha256,
            &plan.binding().gate_matrix_contract_sha256,
        )?;
        verify_runner_claim(
            "QUILL_PERF_PROFILE_CONTRACT_SHA256",
            &claims.profile_contract_sha256,
            &plan.binding().profile_contract_sha256,
        )?;
        verify_runner_claim(
            "QUILL_PERF_REGISTRY_SCHEMA_VERSION",
            &claims.registry_schema_version,
            &plan.binding().registry_schema_version,
        )?;
        verify_runner_claim(
            "QUILL_PERF_REGISTRY_SHA256",
            &claims.registry_sha256,
            &plan.binding().registry_sha256,
        )?;

        let execution_capacity = plan.execution_capacity.ok_or_else(|| {
            "typed benchmark runner requires a frozen execution capacity".to_owned()
        })?;
        if claims.execution_capacity != execution_capacity {
            return Err(format!(
                "QUILL_PERF_EXECUTION_CAPACITY={} differs from frozen profile capacity \
                 {execution_capacity}",
                claims.execution_capacity
            ));
        }
        if claims.rayon_num_threads != execution_capacity {
            return Err(format!(
                "RAYON_NUM_THREADS={} differs from frozen profile capacity {execution_capacity}",
                claims.rayon_num_threads
            ));
        }
        let max_exercised_cell_width = plan.max_exercised_cell_width.ok_or_else(|| {
            "typed benchmark runner requires a frozen maximum exercised cell width".to_owned()
        })?;
        if claims.max_exercised_cell_width != max_exercised_cell_width {
            return Err(format!(
                "QUILL_PERF_MAX_EXERCISED_CELL_WIDTH={} differs from frozen gate maximum \
                 {max_exercised_cell_width}",
                claims.max_exercised_cell_width
            ));
        }
        let planned_max = plan
            .max_runnable_cell_width()
            .and_then(|width| u64::try_from(width).ok())
            .ok_or_else(|| "applicability plan has no representable runnable width".to_owned())?;
        if planned_max > max_exercised_cell_width {
            return Err(format!(
                "applicability plan's maximum runnable literal {planned_max} exceeds frozen gate \
                 maximum {max_exercised_cell_width}"
            ));
        }
        if max_exercised_cell_width > execution_capacity {
            return Err(format!(
                "frozen gate maximum {max_exercised_cell_width} exceeds execution capacity \
                 {execution_capacity}"
            ));
        }
        Ok(Self {
            profile,
            plan,
            execution_capacity,
            max_exercised_cell_width,
        })
    }
}

fn required_env(name: &str) -> Result<String, String> {
    let value = std::env::var(name).map_err(|error| format!("{name} is required: {error}"))?;
    if value.is_empty() || value.trim() != value {
        return Err(format!("{name} must be nonempty canonical text"));
    }
    Ok(value)
}

fn canonical_positive_u64_env(name: &str) -> Result<u64, String> {
    let text = required_env(name)?;
    let value = text
        .parse::<u64>()
        .map_err(|error| format!("{name} must be a positive canonical integer: {error}"))?;
    if value == 0 || value.to_string() != text {
        return Err(format!("{name} must be a positive canonical integer"));
    }
    Ok(value)
}

fn parse_hardware_class_id(value: &str) -> Result<HardwareClassId, String> {
    match value {
        "x86-vps-ovh" => Ok(HardwareClassId::X86VpsOvh),
        "trj-zen3-5995wx" => Ok(HardwareClassId::TrjZen35995wx),
        "m4-macos" => Ok(HardwareClassId::M4Macos),
        "m5-macos" => Ok(HardwareClassId::M5Macos),
        _ => Err(format!(
            "QUILL_PERF_HARDWARE_CLASS names unknown closed hardware class {value:?}"
        )),
    }
}

fn parse_execution_profile_id(value: &str) -> Result<ExecutionProfileId, String> {
    match value {
        "x86-diagnostic" => Ok(ExecutionProfileId::X86Diagnostic),
        "physical-64" => Ok(ExecutionProfileId::Physical64),
        "smt2-128" => Ok(ExecutionProfileId::Smt2_128),
        "scheduler-10" => Ok(ExecutionProfileId::Scheduler10),
        "scheduler-14" => Ok(ExecutionProfileId::Scheduler14),
        _ => Err(format!(
            "QUILL_PERF_EXECUTION_PROFILE names unknown closed execution profile {value:?}"
        )),
    }
}

fn verify_runner_claim(name: &str, supplied: &str, expected: &str) -> Result<(), String> {
    if supplied == expected {
        Ok(())
    } else {
        Err(format!(
            "{name}={supplied:?} differs from frozen applicability-plan value {expected:?}"
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum EngineArm {
    Quill,
    Tantivy,
}

impl EngineArm {
    const fn label(self) -> &'static str {
        match self {
            Self::Quill => "quill",
            Self::Tantivy => "tantivy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Qg1ProducerCoverage {
    EngineIndexingLifecycle,
    TokenizerOnlyDiagnosticNoEngineLifecycle,
}

impl Qg1ProducerCoverage {
    const fn admits_engine_lifecycle_receipt(self) -> bool {
        matches!(self, Self::EngineIndexingLifecycle)
    }
}

fn qg1_producer_coverage(spec: &PerfCellSpec) -> Option<Qg1ProducerCoverage> {
    if spec.gate != PerfGate::Qg1 {
        return None;
    }
    match spec.metric.as_str() {
        "docs_per_second" => Some(Qg1ProducerCoverage::EngineIndexingLifecycle),
        "tokenize_docs_per_second" => {
            Some(Qg1ProducerCoverage::TokenizerOnlyDiagnosticNoEngineLifecycle)
        }
        _ => None,
    }
}

/// The successful terminal proof is deliberately typed by its actual
/// operation.  A generic success string allowed a Quill proof to be relabeled
/// as the distinct retained-Tantivy-reader proof (and vice versa).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Qg1TerminalProof {
    /// An exact prepared-tail membership probe returned the tail alone.
    ExactPreparedTailVisible { tail_document_id: String },
    /// Quill committed and retained its reader through the exact tail probe.
    QuillPublicationThenExactTail {
        tail_document_id: String,
        publication_generation_delta: u64,
    },
    /// Tantivy completed one non-rearming worker join before the retained
    /// reader's exact tail probe.
    TantivyJoinThenExactTail {
        tail_document_id: String,
        terminal_join: BenchmarkWriterJoinReceipt,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Qg1TerminalFact {
    Proven { proof: Qg1TerminalProof },
    NoClaim { code: &'static str, detail: String },
}

impl Qg1TerminalFact {
    fn exact_tail_visible(tail_document_id: impl Into<String>) -> Self {
        Self::Proven {
            proof: Qg1TerminalProof::ExactPreparedTailVisible {
                tail_document_id: tail_document_id.into(),
            },
        }
    }

    fn quill_publication_then_exact_tail(
        tail_document_id: impl Into<String>,
        publication_generation_delta: u64,
    ) -> Self {
        Self::Proven {
            proof: Qg1TerminalProof::QuillPublicationThenExactTail {
                tail_document_id: tail_document_id.into(),
                publication_generation_delta,
            },
        }
    }

    fn tantivy_join_then_exact_tail(
        tail_document_id: impl Into<String>,
        terminal_join: BenchmarkWriterJoinReceipt,
    ) -> Self {
        Self::Proven {
            proof: Qg1TerminalProof::TantivyJoinThenExactTail {
                tail_document_id: tail_document_id.into(),
                terminal_join,
            },
        }
    }

    fn no_claim(detail: impl Into<String>) -> Self {
        Self::NoClaim {
            code: QG1_TERMINAL_NO_CLAIM_CODE,
            detail: detail.into(),
        }
    }

    fn no_claim_detail(&self) -> Option<&str> {
        match self {
            Self::Proven { .. } => None,
            Self::NoClaim { detail, .. } => Some(detail),
        }
    }

    fn exact_tail_document_id(&self) -> Option<&str> {
        match self {
            Self::Proven {
                proof: Qg1TerminalProof::ExactPreparedTailVisible { tail_document_id },
            } => Some(tail_document_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qg1BatchTiming {
    document_start: u64,
    document_count: u64,
    feed_started_ns: u64,
    feed_completed_ns: Option<u64>,
    visibility_commit_completed_ns: Option<u64>,
}

/// Immutable prepared input consumed by exactly one QG-1 engine sample.
///
/// The corpus and each borrowed batch are prepared before the continuous
/// interval starts.  Keeping their identity beside the continuous measurement
/// lets the raw sample denominator come from the data the engine actually
/// received, instead of from a separately regenerated corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qg1PreparedSampleBinding {
    manifest_sha256: String,
    indexed_content_sha256: String,
    document_count: u64,
    content_bytes: u64,
    batch_count: usize,
    /// The actual final prepared document.  A terminal search must prove this
    /// tail rather than an early corpus pathology that would be visible even
    /// after a truncated feed.
    tail_document_id: String,
}

impl Qg1PreparedSampleBinding {
    fn validate(&self) -> Result<(), String> {
        let valid_digest = |digest: &str| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        };
        if !valid_digest(&self.manifest_sha256)
            || !valid_digest(&self.indexed_content_sha256)
            || self.document_count <= 2
            || self.content_bytes == 0
            || self.batch_count == 0
            || self.tail_document_id != qg1_tail_document_id(self.document_count)
        {
            return Err(
                "QG-1 prepared sample binding requires exact corpus identities, positive bytes, \
                 a tail terminal sentinel, and at least one prebuilt batch"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

/// Borrowed QG-1 input and its prebuilt batch schedule.  The owned documents
/// stay in [`PreparedQg1Corpus`], so constructing this sample does not clone
/// or regenerate the corpus.
struct Qg1PreparedSampleInput<'a> {
    documents: &'a [IndexableDocument],
    batches: Vec<&'a [IndexableDocument]>,
    binding: Qg1PreparedSampleBinding,
}

impl<'a> Qg1PreparedSampleInput<'a> {
    fn from_prefix(
        prefix: &'a PreparedQg1Prefix,
        documents: &'a [IndexableDocument],
        batch_documents: usize,
    ) -> Result<Self, String> {
        if batch_documents == 0 {
            return Err("QG-1 prepared batch width must be positive".to_owned());
        }
        let document_count = u64::try_from(documents.len())
            .map_err(|_| "QG-1 prepared document count does not fit u64".to_owned())?;
        if document_count != prefix.manifest.document_count {
            return Err(format!(
                "QG-1 prepared input has {document_count} documents but its manifest names {}",
                prefix.manifest.document_count
            ));
        }
        let content_bytes = documents.iter().try_fold(0_u64, |total, document| {
            total
                .checked_add(
                    u64::try_from(document.content.len())
                        .map_err(|_| "QG-1 content length does not fit u64".to_owned())?,
                )
                .ok_or_else(|| "QG-1 prepared content-byte count overflowed".to_owned())
        })?;
        let observed_content_sha256 = qg1_indexed_content_sha256(document_count, documents.iter())?;
        let batches = documents.chunks(batch_documents).collect::<Vec<_>>();
        let tail_document_id = documents
            .last()
            .ok_or_else(|| "QG-1 prepared input requires a tail document".to_owned())?
            .id
            .to_string();
        let binding = Qg1PreparedSampleBinding {
            manifest_sha256: prefix.manifest_sha256.clone(),
            indexed_content_sha256: prefix.indexed_content_sha256.clone(),
            document_count,
            content_bytes,
            batch_count: batches.len(),
            tail_document_id,
        };
        let prepared = Self {
            documents,
            batches,
            binding,
        };
        prepared.validate()?;
        if observed_content_sha256 != prepared.binding.indexed_content_sha256 {
            return Err(
                "QG-1 prepared input content identity changed before measurement".to_owned(),
            );
        }
        if prepared.binding.content_bytes != prefix.content_bytes {
            return Err(
                "QG-1 prepared input content-byte count changed before measurement".to_owned(),
            );
        }
        Ok(prepared)
    }

    fn validate(&self) -> Result<(), String> {
        self.binding.validate()?;
        if self.documents.len()
            != usize::try_from(self.binding.document_count)
                .map_err(|_| "QG-1 prepared binding count does not fit usize".to_owned())?
            || self.batches.is_empty()
        {
            return Err("QG-1 prepared input does not match its sample binding".to_owned());
        }
        let mut next_document = 0_u64;
        for batch in &self.batches {
            if batch.is_empty() {
                return Err("QG-1 prepared input contains an empty batch".to_owned());
            }
            next_document = next_document
                .checked_add(
                    u64::try_from(batch.len())
                        .map_err(|_| "QG-1 prepared batch count does not fit u64".to_owned())?,
                )
                .ok_or_else(|| "QG-1 prepared batch coverage overflowed".to_owned())?;
        }
        if next_document != self.binding.document_count
            || self.batches.len() != self.binding.batch_count
        {
            return Err("QG-1 prepared batches do not cover their bound input exactly".to_owned());
        }
        Ok(())
    }

    fn verify_binding(&self, binding: &Qg1PreparedSampleBinding) -> Result<(), String> {
        self.validate()?;
        if &self.binding != binding {
            return Err(
                "QG-1 raw sample denominator is not bound to the prepared input it measured"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qg1ContinuousTimingReceipt {
    producer_coverage: Qg1ProducerCoverage,
    arm: EngineArm,
    document_count: u64,
    prepared_input: Qg1PreparedSampleBinding,
    interval_started_ns: u64,
    batches: Vec<Qg1BatchTiming>,
    /// Number of batches that actually returned from the engine feed call.
    /// This is deliberately distinct from the scheduled batch vector so a
    /// receipt cannot name a prepared schedule that was never fully fed.
    recorded_batch_count: usize,
    quill_publication_generation_delta: Option<u64>,
    terminal_commit_completed_ns: u64,
    /// The retired pre-search Tantivy rearm join is retained only so old
    /// receipts fail closed rather than being silently accepted as equivalent.
    pre_search_rearm_join_completed_ns: Option<u64>,
    terminal_worker_join_completed_ns: Option<u64>,
    /// The exact receipt returned by Tantivy's one-shot, non-rearming worker
    /// join. Quill has no corresponding external writer API.
    terminal_tantivy_join: Option<BenchmarkWriterJoinReceipt>,
    /// Captured immediately when the retained read-only handle returns the
    /// exact prepared-tail query. Tantivy workers have already joined and stay
    /// quiescent through that query, so one real boundary replaces fabricated
    /// post-hoc equal timestamps.
    terminal_searchable_quiescence_completed_ns: u64,
    interval_ended_ns: u64,
    terminal_searchability: Qg1TerminalFact,
    terminal_quiescence: Qg1TerminalFact,
}

impl Qg1ContinuousTimingReceipt {
    fn validate(&self) -> Result<(), String> {
        if !self.producer_coverage.admits_engine_lifecycle_receipt() {
            return Err(
                "QG-1 tokenizer-only diagnostics cannot attest engine terminal/quiescence facts"
                    .to_owned(),
            );
        }
        if self.interval_started_ns != 0 {
            return Err("QG-1 continuous interval must begin at offset zero".to_owned());
        }
        if self.document_count <= 2 {
            return Err("QG-1 terminal sentinel requires at least three documents".to_owned());
        }
        self.prepared_input.validate()?;
        if self.prepared_input.document_count != self.document_count {
            return Err(
                "QG-1 continuous receipt work differs from its prepared sample input".to_owned(),
            );
        }
        if let Some(proved_tail) = self.terminal_searchability.exact_tail_document_id()
            && proved_tail != self.prepared_input.tail_document_id
        {
            return Err(
                "QG-1 terminal exact-tail proof names a different prepared document".to_owned(),
            );
        }
        if self.batches.is_empty()
            || self.recorded_batch_count != self.batches.len()
            || self.recorded_batch_count != self.prepared_input.batch_count
        {
            return Err("QG-1 continuous interval contains no prebuilt/feed batches".to_owned());
        }
        if self.batches[0].feed_started_ns != self.interval_started_ns {
            return Err(
                "QG-1 continuous interval must start at the first prepared-batch feed".to_owned(),
            );
        }

        let mut cursor = self.interval_started_ns;
        let mut next_document = 0_u64;
        for batch in &self.batches {
            if batch.document_start != next_document || batch.document_count == 0 {
                return Err("QG-1 batch coverage is not contiguous and positive".to_owned());
            }
            let fed = batch
                .feed_completed_ns
                .ok_or_else(|| "QG-1 batch is missing its feed-complete boundary".to_owned())?;
            if batch.feed_started_ns < cursor || fed < batch.feed_started_ns {
                return Err(
                    "QG-1 prebuilt-batch feed phases escape monotonic interval order".to_owned(),
                );
            }
            cursor = fed;
            if let Some(committed) = batch.visibility_commit_completed_ns {
                if committed < cursor {
                    return Err("QG-1 visibility commit completed before its batch feed".to_owned());
                }
                cursor = committed;
            }
            next_document = next_document
                .checked_add(batch.document_count)
                .ok_or_else(|| "QG-1 batch document coverage overflowed".to_owned())?;
        }
        if next_document != self.document_count {
            return Err(format!(
                "QG-1 batches cover {next_document} documents instead of {}",
                self.document_count
            ));
        }
        if self.terminal_commit_completed_ns < cursor {
            return Err("QG-1 terminal commit escaped the batch interval".to_owned());
        }
        cursor = self.terminal_commit_completed_ns;
        if self.pre_search_rearm_join_completed_ns.is_some() {
            return Err(
                "QG-1 rejects the retired Tantivy rearm join before terminal search".to_owned(),
            );
        }
        match self.arm {
            EngineArm::Quill => {
                if self.terminal_worker_join_completed_ns.is_some()
                    || self.terminal_tantivy_join.is_some()
                    || self.quill_publication_generation_delta.is_none()
                {
                    return Err(
                        "QG-1 Quill receipt names an impossible external worker-join lifecycle"
                            .to_owned(),
                    );
                }
                if self.terminal_searchable_quiescence_completed_ns < cursor {
                    return Err(
                        "QG-1 retained Quill tail search preceded terminal commit".to_owned()
                    );
                }
                if let Qg1TerminalFact::Proven { proof } = &self.terminal_quiescence {
                    let Qg1TerminalProof::QuillPublicationThenExactTail {
                        tail_document_id,
                        publication_generation_delta,
                    } = proof
                    else {
                        return Err(
                            "QG-1 Quill receipt carries a Tantivy or generic terminal proof"
                                .to_owned(),
                        );
                    };
                    if *publication_generation_delta == 0
                        || tail_document_id != &self.prepared_input.tail_document_id
                        || Some(*publication_generation_delta)
                            != self.quill_publication_generation_delta
                    {
                        return Err(
                            "QG-1 Quill lifecycle proof is not bound to this tail/publication"
                                .to_owned(),
                        );
                    }
                }
            }
            EngineArm::Tantivy => {
                let Some(join) = self.terminal_tantivy_join else {
                    return Err(
                        "QG-1 Tantivy receipt lacks the actual terminal worker-join API receipt"
                            .to_owned(),
                    );
                };
                let Some(joined) = self.terminal_worker_join_completed_ns else {
                    return Err(
                        "QG-1 Tantivy receipt lacks the terminal worker-join boundary".to_owned(),
                    );
                };
                if self.quill_publication_generation_delta.is_some() || join.writer_rearmed {
                    return Err(
                        "QG-1 Tantivy receipt names an impossible nonrearming join lifecycle"
                            .to_owned(),
                    );
                }
                if join.searchable_segments_before == 0
                    || join.searchable_segments_after == 0
                    || join.join_elapsed_ns == 0
                {
                    return Err(
                        "QG-1 Tantivy receipt lacks authenticated positive searchable/join facts"
                            .to_owned(),
                    );
                }
                if joined < cursor {
                    return Err("QG-1 Tantivy worker join preceded terminal commit".to_owned());
                }
                if self.terminal_searchable_quiescence_completed_ns < joined {
                    return Err(
                        "QG-1 retained Tantivy tail search preceded the terminal worker join"
                            .to_owned(),
                    );
                }
                if let Qg1TerminalFact::Proven { proof } = &self.terminal_quiescence {
                    let Qg1TerminalProof::TantivyJoinThenExactTail {
                        tail_document_id,
                        terminal_join,
                    } = proof
                    else {
                        return Err(
                            "QG-1 Tantivy receipt carries a Quill or generic terminal proof"
                                .to_owned(),
                        );
                    };
                    if tail_document_id != &self.prepared_input.tail_document_id
                        || terminal_join != &join
                    {
                        return Err(
                            "QG-1 Tantivy lifecycle proof is not bound to this tail/join receipt"
                                .to_owned(),
                        );
                    }
                }
            }
        }
        if self.interval_ended_ns != self.terminal_searchable_quiescence_completed_ns {
            return Err(
                "QG-1 continuous interval must end at the retained-reader searchable-quiescence boundary"
                    .to_owned(),
            );
        }
        Ok(())
    }

    fn no_claim_details(&self) -> impl Iterator<Item = &str> {
        [
            self.terminal_searchability.no_claim_detail(),
            self.terminal_quiescence.no_claim_detail(),
        ]
        .into_iter()
        .flatten()
    }
}

/// Convert one verified QG-1 lifecycle receipt into the compact typed binding
/// retained by the paired estimator.  This is intentionally fallible through
/// `Option`: terminal diagnostics remain serializable as `NoClaim`, but they
/// produce no binding and therefore cannot become a throughput headline.
fn qg1_live_sample_binding(
    continuous: Option<&Qg1ContinuousMeasurement>,
    elapsed_ns: u64,
    scope: &PerfOperationScope,
    provenance: &PerfSampleProvenance,
    estimator_config: &PairedEstimatorConfig,
    producer: &Qg1LifecycleProducer,
    stream_role: &str,
    stream_sequence: u64,
    sample_id: u64,
    block_id: u64,
    arm: PerfSampleArm,
    order: PerfSampleOrder,
) -> Option<Qg1SampleBinding> {
    // Refuse a foreign producer before removing its one-shot capability. This
    // applies equally to the Tantivy null stream: attach-null must consume the
    // same independently retained authority as the effect stream.
    if !estimator_config.qg1_expected_authority_matches(producer.expected_authority()) {
        return None;
    }
    let continuous = continuous?;
    let receipt = &continuous.lifecycle_receipt;
    receipt.validate().ok()?;
    if receipt.interval_ended_ns != elapsed_ns || continuous.elapsed_ns != elapsed_ns {
        return None;
    }
    let tail_document_id = receipt.prepared_input.tail_document_id.clone();
    let lifecycle_witness = match (
        receipt.arm,
        &receipt.terminal_searchability,
        &receipt.terminal_quiescence,
    ) {
        (
            EngineArm::Quill,
            Qg1TerminalFact::Proven {
                proof:
                    Qg1TerminalProof::ExactPreparedTailVisible {
                        tail_document_id: search_tail,
                    },
            },
            Qg1TerminalFact::Proven {
                proof:
                    Qg1TerminalProof::QuillPublicationThenExactTail {
                        tail_document_id: lifecycle_tail,
                        publication_generation_delta,
                    },
            },
        ) if search_tail == &tail_document_id
            && lifecycle_tail == &tail_document_id
            && *publication_generation_delta > 0
            && receipt.quill_publication_generation_delta
                == Some(*publication_generation_delta) =>
        {
            Qg1LifecycleWitness::Quill {
                publication_generation_delta: *publication_generation_delta,
            }
        }
        (
            EngineArm::Tantivy,
            Qg1TerminalFact::Proven {
                proof:
                    Qg1TerminalProof::ExactPreparedTailVisible {
                        tail_document_id: search_tail,
                    },
            },
            Qg1TerminalFact::Proven {
                proof:
                    Qg1TerminalProof::TantivyJoinThenExactTail {
                        tail_document_id: lifecycle_tail,
                        terminal_join,
                    },
            },
        ) if search_tail == &tail_document_id
            && lifecycle_tail == &tail_document_id
            && receipt.terminal_tantivy_join.as_ref() == Some(terminal_join)
            && !terminal_join.writer_rearmed =>
        {
            Qg1LifecycleWitness::Tantivy {
                searchable_segments_before: terminal_join.searchable_segments_before,
                searchable_segments_after: terminal_join.searchable_segments_after,
                join_elapsed_ns: terminal_join.join_elapsed_ns,
                writer_rearmed: terminal_join.writer_rearmed,
            }
        }
        _ => return None,
    };
    let binding = Qg1SampleBinding {
        schema_version: Qg1SampleBinding::SCHEMA_VERSION.to_owned(),
        stream_role: stream_role.to_owned(),
        stream_id_sha256: String::new(),
        stream_sequence,
        raw_sample_id: sample_id,
        raw_block_id: block_id,
        raw_arm: arm,
        raw_order: order,
        lifecycle_authority_sha256: String::new(),
        stream_role_identity_sha256: String::new(),
        producer_capability_sha256: String::new(),
        producer_capability_tag_sha256: String::new(),
        lifecycle_receipt_id_sha256: String::new(),
        lifecycle_receipt_sha256: String::new(),
        prepared_corpus_sha256: provenance.corpus_sha256.clone(),
        prepared_input_sha256: String::new(),
        prepared_manifest_sha256: receipt.prepared_input.manifest_sha256.clone(),
        indexed_content_sha256: receipt.prepared_input.indexed_content_sha256.clone(),
        document_count: receipt.prepared_input.document_count,
        content_bytes: receipt.prepared_input.content_bytes,
        prepared_batch_count: receipt.prepared_input.batch_count,
        recorded_batch_count: receipt.recorded_batch_count,
        batch_coverage: receipt
            .batches
            .iter()
            .map(|batch| Qg1BatchCoverage {
                document_start: batch.document_start,
                document_count: batch.document_count,
            })
            .collect(),
        tail_document_id,
        terminal_endpoint_ns: receipt.interval_ended_ns,
        lifecycle_witness,
    };
    let binding = producer.consume_lifecycle_receipt(scope, provenance, binding)?;
    estimator_config
        .qg1_binding_matches_lifecycle_authority(&binding, scope, provenance)
        .then_some(binding)
}

/// Work completed per elapsed second, derived exactly as
/// [`PerfMetricSemantics::Throughput`] derives it inside the estimator.
///
/// A QG-1 row reports this number twice: once as the sample's `observed_value`,
/// which feeds the published absolutes, and once when the estimator recomputes
/// it from that sample's own work and interval. Sharing one expression is what
/// makes those two numbers identical rather than merely close.
fn throughput_per_second(work_units: u64, elapsed_ns: u64) -> f64 {
    work_units as f64 * 1_000_000_000.0 / elapsed_ns as f64
}

/// The exact continuous engine interval one QG-1 measurement was derived from.
///
/// `origin` is the interval's own zero point. Keeping it lets a caller place the
/// interval inside a longer timeline without the interval needing to know that
/// timeline exists.
#[derive(Clone)]
struct Qg1ContinuousMeasurement {
    work_units: u64,
    origin: Instant,
    elapsed_ns: u64,
    prepared_input: Qg1PreparedSampleBinding,
    lifecycle_receipt: Qg1ContinuousTimingReceipt,
}

/// One continuous QG-2 update interval.
///
/// Deliberately carries no prepared-input binding and no lifecycle receipt: QG-2
/// has neither, and manufacturing them here would file QG-2 work under QG-1
/// attestations it never earned. What it does carry is the only thing the
/// estimator needs to type a rate — the work the interval processed and the
/// single monotonic span it processed that work in.
struct Qg2ContinuousMeasurement {
    work_units: u64,
    origin: Instant,
    elapsed_ns: u64,
    /// The retired shape, measured in this same invocation: the feed's own
    /// timing plus the commit's own timing, summed. Retained as a diagnostic so
    /// tail inclusion can be proved against the interval that produced it,
    /// rather than against a second run whose cache and scheduler state differ.
    /// The published value is never computed from this.
    feed_and_commit_ns: u64,
    /// Why the engine is quiescent at the terminal endpoint. A successful tail
    /// search proves VISIBILITY; it does not by itself prove the writer side
    /// settled, so the basis is recorded separately and asserted separately.
    quiescence: Qg2QuiescenceBasis,
}

/// The engine-specific fact that establishes quiescence at a QG-2 endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qg2QuiescenceBasis {
    /// The terminal commit published a new Keeper generation, so no publication
    /// is still in flight behind the tail that was just read.
    QuillPublishedGeneration { delta: u64 },
    /// Tantivy's writer workers joined without rearming a replacement writer,
    /// so no merge or write thread is still running behind the retained reader.
    TantivyWorkersJoined { rearmed: bool },
}

/// One measured cell value, plus the continuous interval behind it when the
/// producer actually measured one.
///
/// QG-1's engine-indexing cells carry `continuous`; QG-2's update cells carry
/// `qg2_continuous`. Every other cell in this matrix reports a value assembled
/// from independently timed calls, and the absence of an interval here is what
/// stops such a value from being typed as throughput downstream.
struct MetricMeasurement {
    value: f64,
    continuous: Option<Qg1ContinuousMeasurement>,
    qg2_continuous: Option<Qg2ContinuousMeasurement>,
}

impl MetricMeasurement {
    /// A directly observed value with no continuous interval behind it.
    const fn gauge(value: f64) -> Self {
        Self {
            value,
            continuous: None,
            qg2_continuous: None,
        }
    }
}

struct Qg1ContinuousInterval {
    /// Set at the first engine feed, after corpus and batch preparation have
    /// completed.  Nothing before this point contributes to throughput time.
    origin: Option<Instant>,
    arm: EngineArm,
    prepared_input: Qg1PreparedSampleBinding,
    batches: Vec<Qg1BatchTiming>,
    recorded_batch_count: usize,
    terminal_commit_completed_ns: Option<u64>,
    terminal_worker_join_completed_ns: Option<u64>,
    terminal_tantivy_join: Option<BenchmarkWriterJoinReceipt>,
    terminal_searchable_quiescence_completed_ns: Option<u64>,
}

impl Qg1ContinuousInterval {
    fn start(arm: EngineArm, prepared_input: Qg1PreparedSampleBinding) -> Self {
        prepared_input
            .validate()
            .expect("QG-1 continuous sample requires verified prepared input");
        Self {
            origin: None,
            arm,
            prepared_input,
            batches: Vec::new(),
            recorded_batch_count: 0,
            terminal_commit_completed_ns: None,
            terminal_worker_join_completed_ns: None,
            terminal_tantivy_join: None,
            terminal_searchable_quiescence_completed_ns: None,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        u64::try_from(
            self.origin
                .expect("QG-1 elapsed time requires the first engine feed")
                .elapsed()
                .as_nanos(),
        )
        .unwrap_or(u64::MAX)
    }

    fn begin_batch(&mut self, document_start: u64, document_count: u64) -> u64 {
        // Push the first bookkeeping record *before* starting the clock. The
        // first feed is therefore exactly zero, without an `elapsed()` read or
        // any record mutation between its timestamp and the engine call.
        let feed_started_ns = if let Some(origin) = self.origin {
            u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
        } else {
            self.batches.push(Qg1BatchTiming {
                document_start,
                document_count,
                feed_started_ns: 0,
                feed_completed_ns: None,
                visibility_commit_completed_ns: None,
            });
            self.origin = Some(Instant::now());
            return 0;
        };
        self.batches.push(Qg1BatchTiming {
            document_start,
            document_count,
            feed_started_ns,
            feed_completed_ns: None,
            visibility_commit_completed_ns: None,
        });
        feed_started_ns
    }

    fn mark_batch_fed(&mut self) {
        let completed = self.elapsed_ns();
        let batch = self
            .batches
            .last_mut()
            .expect("QG-1 feed completion requires an active batch");
        assert!(
            batch.feed_completed_ns.replace(completed).is_none(),
            "QG-1 batch feed boundary repeated"
        );
        self.recorded_batch_count = self
            .recorded_batch_count
            .checked_add(1)
            .expect("QG-1 completed batch count fits usize");
    }

    fn mark_visibility_commit(&mut self) {
        let completed = self.elapsed_ns();
        let batch = self
            .batches
            .last_mut()
            .expect("QG-1 visibility commit requires an active batch");
        assert!(
            batch
                .visibility_commit_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 batch visibility-commit boundary repeated"
        );
    }

    fn mark_terminal_commit(&mut self) {
        let completed = self.elapsed_ns();
        assert!(
            self.terminal_commit_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 terminal commit boundary repeated"
        );
    }

    fn mark_terminal_searchable_quiescence(&mut self) -> u64 {
        let completed = self.elapsed_ns();
        assert!(
            self.terminal_searchable_quiescence_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 retained-reader searchable-quiescence boundary repeated"
        );
        completed
    }

    fn mark_terminal_worker_join(&mut self, receipt: BenchmarkWriterJoinReceipt) -> u64 {
        let completed = self.elapsed_ns();
        assert!(
            self.terminal_worker_join_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 terminal worker join boundary repeated"
        );
        assert!(
            self.terminal_tantivy_join.replace(receipt).is_none(),
            "QG-1 terminal Tantivy worker-join receipt repeated"
        );
        completed
    }

    fn finish(
        self,
        quill_publication_generation_delta: Option<u64>,
        terminal_searchability: Qg1TerminalFact,
        terminal_quiescence: Qg1TerminalFact,
    ) -> (Qg1ContinuousMeasurement, Qg1ContinuousTimingReceipt) {
        let origin = self
            .origin
            .expect("QG-1 continuous interval includes at least one engine feed");
        let work_units = self.prepared_input.document_count;
        let interval_ended_ns = self
            .terminal_searchable_quiescence_completed_ns
            .expect("QG-1 continuous interval includes retained-reader searchable quiescence");
        let receipt = Qg1ContinuousTimingReceipt {
            producer_coverage: Qg1ProducerCoverage::EngineIndexingLifecycle,
            arm: self.arm,
            document_count: work_units,
            prepared_input: self.prepared_input.clone(),
            interval_started_ns: 0,
            batches: self.batches,
            recorded_batch_count: self.recorded_batch_count,
            quill_publication_generation_delta,
            terminal_commit_completed_ns: self
                .terminal_commit_completed_ns
                .expect("QG-1 continuous interval includes terminal commit"),
            pre_search_rearm_join_completed_ns: None,
            terminal_worker_join_completed_ns: self.terminal_worker_join_completed_ns,
            terminal_tantivy_join: self.terminal_tantivy_join,
            terminal_searchable_quiescence_completed_ns: interval_ended_ns,
            interval_ended_ns,
            terminal_searchability,
            terminal_quiescence,
        };
        receipt
            .validate()
            .expect("invalid QG-1 continuous timing receipt");
        // The measurement and the receipt name one interval, not two: a single
        // `elapsed()` reading is published in both.
        let measurement = Qg1ContinuousMeasurement {
            work_units,
            origin,
            elapsed_ns: receipt.interval_ended_ns,
            prepared_input: receipt.prepared_input.clone(),
            lifecycle_receipt: receipt.clone(),
        };
        (measurement, receipt)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qg1ContinuousTimingRecord {
    schema_version: &'static str,
    admission_status: &'static str,
    admission_no_claim_code: &'static str,
    admission_no_claim_detail: &'static str,
    run_id: String,
    sequence: u64,
    gate: String,
    fixture: String,
    metric: String,
    writer_threads: usize,
    writer_heap_bytes: usize,
    #[serde(flatten)]
    timing: Qg1ContinuousTimingReceipt,
}

enum PreparedQueryArm {
    Quill {
        index: Box<QuillIndex>,
    },
    Tantivy {
        role: Qg6ArmRole,
        index: Box<TantivyIndex>,
    },
}

struct PreparedQueryResult {
    native_hits: Vec<(String, u32)>,
    total_count: u64,
    doc_count: u64,
}

#[derive(Clone)]
struct PreparedQueryPreflight {
    native_hits: Vec<(String, u32)>,
    total_count: u64,
    doc_count: u64,
    public_result_sha256: String,
    observation: Option<EngineObservation>,
}

struct PreparedQg1Prefix {
    manifest: CorpusManifest,
    manifest_sha256: String,
    indexed_content_sha256: String,
    content_bytes: u64,
}

struct PreparedQg1Corpus {
    documents: Arc<[IndexableDocument]>,
    prefixes: BTreeMap<u64, PreparedQg1Prefix>,
}

fn hash_qg1_indexed_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(
        u64::try_from(bytes.len())
            .expect("QG-1 indexed field length fits u64")
            .to_le_bytes(),
    );
    hasher.update(bytes);
}

fn qg1_indexed_content_sha256<I, D>(
    expected_document_count: u64,
    documents: I,
) -> Result<String, String>
where
    I: IntoIterator<Item = D>,
    D: Borrow<IndexableDocument>,
{
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch-quill-qg1-indexable-documents-v1\0");
    hasher.update(expected_document_count.to_le_bytes());
    let mut observed_document_count = 0_u64;
    for document in documents {
        let document = document.borrow();
        hash_qg1_indexed_bytes(&mut hasher, document.id.as_bytes());
        match &document.title {
            Some(title) => {
                hasher.update([1]);
                hash_qg1_indexed_bytes(&mut hasher, title.as_bytes());
            }
            None => hasher.update([0]),
        }
        hash_qg1_indexed_bytes(&mut hasher, document.content.as_bytes());

        hasher.update(
            u64::try_from(document.metadata.len())
                .expect("QG-1 indexed metadata count fits u64")
                .to_le_bytes(),
        );
        let mut previous_metadata = None;
        for _ in 0..document.metadata.len() {
            let (key, value) = document
                .metadata
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .filter(|entry| previous_metadata.is_none_or(|previous| *entry > previous))
                .min()
                .expect("QG-1 indexed metadata cardinality is stable while hashing");
            hash_qg1_indexed_bytes(&mut hasher, key.as_bytes());
            hash_qg1_indexed_bytes(&mut hasher, value.as_bytes());
            previous_metadata = Some((key, value));
        }
        observed_document_count = observed_document_count
            .checked_add(1)
            .ok_or_else(|| "QG-1 indexed document count overflowed".to_owned())?;
    }
    if observed_document_count != expected_document_count {
        return Err(format!(
            "QG-1 indexed content observed {observed_document_count} documents but expected \
             {expected_document_count}"
        ));
    }
    Ok(lower_hex(&hasher.finalize()))
}

impl PreparedQg1Corpus {
    fn for_selected(scale: MatrixScale, selected: &[PerfCellSpec]) -> Option<Self> {
        let effective_counts = selected
            .iter()
            .filter(|spec| spec.gate == PerfGate::Qg1)
            .map(|spec| {
                scale.document_count(
                    spec.document_count
                        .expect("selected QG-1 cell has a document count"),
                )
            });
        Self::from_effective_counts(effective_counts)
    }

    fn from_effective_counts(counts: impl IntoIterator<Item = u64>) -> Option<Self> {
        let mut distinct_counts = BTreeSet::new();
        for count in counts {
            assert!(count > 0, "prepared QG-1 corpus count must be positive");
            distinct_counts.insert(count);
        }
        let largest_count = distinct_counts.last().copied()?;
        let largest_corpus = corpus_for(largest_count);
        let generated = largest_corpus.iter().collect::<Vec<_>>();
        assert_eq!(
            generated.len(),
            usize::try_from(largest_count).expect("largest QG-1 corpus count fits usize"),
            "prepared QG-1 corpus materialized a different document count"
        );

        let mut verified_manifests = BTreeMap::new();
        for count in distinct_counts {
            let prefix_len = usize::try_from(count).expect("QG-1 corpus prefix count fits usize");
            let manifest = corpus_for(count)
                .manifest()
                .expect("build exact prepared QG-1 corpus manifest");
            let generated_prefix = generated
                .get(..prefix_len)
                .expect("prepared QG-1 manifest prefix is within the largest corpus");
            manifest
                .verify_documents(generated_prefix)
                .expect("prepared QG-1 corpus prefix matches its exact manifest");
            assert_eq!(
                manifest.document_count, count,
                "prepared QG-1 prefix manifest count drifted"
            );
            assert!(
                verified_manifests
                    .insert(
                        count,
                        (
                            manifest
                                .manifest_hash()
                                .expect("hash verified QG-1 corpus manifest"),
                            manifest,
                        ),
                    )
                    .is_none(),
                "prepared QG-1 corpus repeated an effective count"
            );
        }

        let documents: Arc<[IndexableDocument]> = generated
            .into_iter()
            .map(IndexableDocument::from)
            .collect::<Vec<_>>()
            .into();
        let prefixes = verified_manifests
            .into_iter()
            .map(|(count, (manifest_sha256, manifest))| {
                let prefix_len =
                    usize::try_from(count).expect("QG-1 corpus prefix count fits usize");
                let indexed_documents = documents
                    .get(..prefix_len)
                    .expect("prepared QG-1 corpus prefix is within the largest corpus");
                let content_bytes = indexed_documents.iter().fold(0_u64, |total, document| {
                    total
                        .checked_add(
                            u64::try_from(document.content.len())
                                .expect("QG-1 document content length fits u64"),
                        )
                        .expect("QG-1 corpus content-byte count fits u64")
                });
                (
                    count,
                    PreparedQg1Prefix {
                        manifest,
                        manifest_sha256,
                        indexed_content_sha256: qg1_indexed_content_sha256(
                            count,
                            indexed_documents.iter(),
                        )
                        .expect("hash prepared QG-1 indexed content"),
                        content_bytes,
                    },
                )
            })
            .collect();
        Some(Self {
            documents,
            prefixes,
        })
    }

    fn prefix(&self, document_count: u64) -> (&PreparedQg1Prefix, &[IndexableDocument]) {
        let prefix = self
            .prefixes
            .get(&document_count)
            .expect("selected QG-1 corpus prefix was prepared");
        let prefix_len = usize::try_from(prefix.manifest.document_count)
            .expect("prepared QG-1 manifest count fits usize");
        assert_eq!(
            prefix.manifest.document_count, document_count,
            "prepared QG-1 prefix identity drifted"
        );
        (
            prefix,
            self.documents
                .get(..prefix_len)
                .expect("prepared QG-1 prefix is within the largest corpus"),
        )
    }

    fn validate_prefix(&self, document_count: u64) -> Result<(&str, &str), String> {
        let prefix = self
            .prefixes
            .get(&document_count)
            .ok_or_else(|| format!("QG-1 corpus prefix {document_count} was not prepared"))?;
        if prefix.manifest.document_count != document_count {
            return Err(format!(
                "QG-1 manifest count {} differs from prepared prefix {document_count}",
                prefix.manifest.document_count
            ));
        }
        let prefix_len = usize::try_from(document_count)
            .map_err(|error| format!("QG-1 corpus prefix count does not fit usize: {error}"))?;
        let indexed_documents = self
            .documents
            .get(..prefix_len)
            .ok_or_else(|| "QG-1 prepared corpus is shorter than its prefix".to_owned())?;
        let observed_manifest_sha256 = prefix
            .manifest
            .manifest_hash()
            .map_err(|error| format!("hash QG-1 corpus manifest: {error}"))?;
        if observed_manifest_sha256 != prefix.manifest_sha256 {
            return Err("QG-1 prepared manifest identity changed after replay verification".into());
        }
        let observed_indexed_content_sha256 =
            qg1_indexed_content_sha256(document_count, indexed_documents.iter())?;
        if observed_indexed_content_sha256 != prefix.indexed_content_sha256 {
            return Err("QG-1 prepared indexed content changed after verification".into());
        }
        Ok((&prefix.manifest_sha256, &prefix.indexed_content_sha256))
    }
}

struct BenchContext {
    runtime: Runtime,
    cx: Cx,
    scale: MatrixScale,
    prepared_qg1: Option<PreparedQg1Corpus>,
}

impl BenchContext {
    fn new(scale: MatrixScale) -> Self {
        Self {
            runtime: asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("QG benchmark runtime"),
            cx: Cx::for_testing(),
            scale,
            prepared_qg1: None,
        }
    }

    fn for_selected(scale: MatrixScale, selected: &[PerfCellSpec]) -> Self {
        let mut context = Self::new(scale);
        context.prepared_qg1 = PreparedQg1Corpus::for_selected(scale, selected);
        context
    }

    fn qg1_prefix(&self, document_count: u64) -> (&PreparedQg1Prefix, &[IndexableDocument]) {
        self.prepared_qg1
            .as_ref()
            .expect("selected QG-1 cells have one prepared corpus")
            .prefix(document_count)
    }

    fn qg1_sample_input(&self, document_count: u64) -> Qg1PreparedSampleInput<'_> {
        let (prefix, documents) = self.qg1_prefix(document_count);
        Qg1PreparedSampleInput::from_prefix(prefix, documents, self.scale.batch_documents())
            .expect("prepare one exact QG-1 sample input before timing")
    }
}

fn synthetic_spec(document_count: u64) -> SyntheticCorpusSpec {
    SyntheticCorpusSpec {
        seed: CORPUS_SEED,
        document_count,
        vocabulary_size: VOCABULARY_SIZE,
        zipf_exponent: ZipfExponent::S11,
        max_document_bytes: MAX_DOCUMENT_BYTES,
    }
}

fn corpus_for(document_count: u64) -> SyntheticCorpus {
    SyntheticCorpus::new(synthetic_spec(document_count)).expect("pinned QG corpus recipe")
}

fn generated_batch(
    corpus: &SyntheticCorpus,
    start: u64,
    count: usize,
    update_generation: Option<u64>,
) -> Vec<IndexableDocument> {
    (0..count)
        .map(|offset| {
            let ordinal = start.saturating_add(u64::try_from(offset).expect("batch ordinal"));
            let mut generated = corpus
                .document_at(ordinal % corpus.len())
                .expect("generated document ordinal");
            if let Some(generation) = update_generation {
                generated.content.push_str(" quill update generation ");
                generated.content.push_str(&generation.to_string());
                generated.content.push_str(" qgupdateg");
                generated.content.push_str(&generation.to_string());
                generated.content.push('d');
                generated.content.push_str(&ordinal.to_string());
            }
            generated.into()
        })
        .collect()
}

fn quill_config(spec: &PerfCellSpec) -> QuillConfig {
    let threads = spec.threads.unwrap_or(1);
    let heap = spec.writer_heap_bytes.unwrap_or(50_000_000);
    pinned_quill_config(heap, threads)
}

fn pinned_quill_config(heap: usize, threads: usize) -> QuillConfig {
    QuillConfig {
        scribe_shard_budget_bytes: (heap / threads.max(1)).max(1),
        max_ingest_shards: threads,
        tier_fanout: 64,
        deterministic_ingest: threads == 1,
        ..QuillConfig::default()
    }
}

fn quill_in_memory(spec: &PerfCellSpec) -> QuillIndex {
    let config = quill_config(spec);
    if spec.positions.unwrap_or(PositionMode::On).enabled() {
        QuillIndex::in_memory(config).expect("QG Quill index")
    } else {
        QuillIndex::in_memory_with_schema(NO_POSITION_SCHEMA, config)
            .expect("QG position-free Quill index")
    }
}

fn tantivy_in_memory(spec: &PerfCellSpec) -> TantivyIndex {
    TantivyIndex::in_memory_with_benchmark_config(
        spec.writer_heap_bytes.unwrap_or(50_000_000),
        spec.threads.unwrap_or(1),
        spec.positions.unwrap_or(PositionMode::On).enabled(),
    )
    .expect("QG Tantivy oracle")
}

fn qg1_tantivy_in_memory(spec: &PerfCellSpec, writer_mode: Qg1TantivyWriterMode) -> TantivyIndex {
    let writer_heap_bytes = spec.writer_heap_bytes.unwrap_or(50_000_000);
    let positions = spec.positions.unwrap_or(PositionMode::On).enabled();
    match writer_mode {
        Qg1TantivyWriterMode::ShippingAuto => {
            TantivyIndex::in_memory_with_shipping_auto_writer(writer_heap_bytes, positions)
                .expect("construct QG-1 Tantivy ShippingAuto incumbent")
        }
        Qg1TantivyWriterMode::Fixed { writer_threads } => {
            TantivyIndex::in_memory_with_benchmark_config(
                writer_heap_bytes,
                writer_threads,
                positions,
            )
            .expect("construct QG-1 fixed-width Tantivy incumbent")
        }
    }
}

fn qg1_expected_materialized_width(writer_mode: Qg1TantivyWriterMode) -> Option<usize> {
    match writer_mode {
        Qg1TantivyWriterMode::ShippingAuto => None,
        Qg1TantivyWriterMode::Fixed { writer_threads } => Some(writer_threads),
    }
}

fn qg1_validate_writer_receipt(
    spec: &PerfCellSpec,
    writer_mode: Qg1TantivyWriterMode,
    receipt: &BenchmarkWriterReceipt,
) {
    let expected_mode = match writer_mode {
        Qg1TantivyWriterMode::ShippingAuto => BenchmarkWriterMode::ShippingAuto,
        Qg1TantivyWriterMode::Fixed { writer_threads } => BenchmarkWriterMode::Fixed {
            threads: writer_threads,
        },
    };
    assert_eq!(
        receipt.mode, expected_mode,
        "QG-1 writer constructor drifted"
    );
    assert_eq!(
        receipt.writer_heap_bytes,
        spec.writer_heap_bytes.unwrap_or(50_000_000),
        "QG-1 writer heap drifted"
    );
    match (writer_mode, receipt.materialized_width) {
        (Qg1TantivyWriterMode::ShippingAuto, BenchmarkMaterializedWidth::Unobservable { .. }) => {}
        (
            Qg1TantivyWriterMode::Fixed { writer_threads },
            BenchmarkMaterializedWidth::Authenticated(observed),
        ) if observed == writer_threads => {}
        _ => panic!("QG-1 writer materialized-width receipt does not match its constructor"),
    }
}

fn qg1_attest_writer_constructor(
    spec: &PerfCellSpec,
    writer_mode: Qg1TantivyWriterMode,
) -> BenchmarkWriterReceipt {
    let mut index = qg1_tantivy_in_memory(spec, writer_mode);
    let attestation = index
        .take_benchmark_writer_attestation()
        .expect("QG-1 benchmark constructor must mint a one-shot attestation");
    qg1_validate_writer_receipt(spec, writer_mode, attestation.receipt());
    let receipt = attestation.receipt().clone();
    black_box(attestation.construction_id());
    index
        .benchmark_join_workers()
        .expect("join QG-1 constructor-attestation writer");
    receipt
}

fn tantivy_create(path: &Path, spec: &PerfCellSpec) -> TantivyIndex {
    TantivyIndex::create_with_benchmark_config(
        path,
        spec.writer_heap_bytes.unwrap_or(50_000_000),
        spec.threads.unwrap_or(1),
        spec.positions.unwrap_or(PositionMode::On).enabled(),
    )
    .expect("create pinned on-disk Tantivy oracle")
}

fn emit_tantivy_lifecycle_receipt(
    spec: &PerfCellSpec,
    phase: &str,
    receipt: &BenchmarkWriterJoinReceipt,
) {
    let sequence = LIFECYCLE_RECEIPT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let run_id =
        std::env::var("QUILL_PERF_RUN_ID").unwrap_or_else(|_| "unidentified-run".to_owned());
    let row = serde_json::json!({
        "schema_version": "quill-tantivy-lifecycle-v1",
        "run_id": run_id,
        "sequence": sequence,
        "gate": spec.gate.to_string(),
        "fixture": spec.fixture,
        "metric": spec.metric,
        "phase": phase,
        "writer_threads": spec.threads.unwrap_or(1),
        "writer_heap_bytes": spec.writer_heap_bytes.unwrap_or(50_000_000),
        "searchable_segments_before": receipt.searchable_segments_before,
        "searchable_segments_after": receipt.searchable_segments_after,
        "join_elapsed_ns": receipt.join_elapsed_ns,
        "indexing_workers_joined": true,
        "merge_worker_joined": true,
        "writer_rearmed": receipt.writer_rearmed,
    });
    LIFECYCLE_RECEIPTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("lock QG Tantivy lifecycle receipts")
        .push(row);
}

fn flush_tantivy_lifecycle_receipts(output_dir: &Path) {
    let Some(receipts) = LIFECYCLE_RECEIPTS.get() else {
        return;
    };
    let (payload, receipt_count) = {
        let receipts = receipts
            .lock()
            .expect("lock QG Tantivy lifecycle receipts for flush");
        if receipts.is_empty() {
            return;
        }
        let mut payload = Vec::new();
        for row in receipts.iter() {
            serde_json::to_writer(&mut payload, row)
                .expect("serialize QG Tantivy lifecycle receipt");
            payload.push(b'\n');
        }
        (payload, receipts.len())
    };
    std::fs::create_dir_all(output_dir).expect("create QG lifecycle receipt directory");
    let path = output_dir.join("tantivy-lifecycle.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("open QG Tantivy lifecycle receipt");
    file.write_all(&payload)
        .expect("write QG Tantivy lifecycle receipts");
    eprintln!(
        "[tantivy-lifecycle] receipts={} sha256={} path={}",
        receipt_count,
        lower_hex(&Sha256::digest(&payload)),
        display_path(&path),
    );
}

fn emit_qg1_continuous_timing_receipt(spec: &PerfCellSpec, timing: Qg1ContinuousTimingReceipt) {
    let sequence = QG1_CONTINUOUS_TIMING_COUNTER.fetch_add(1, Ordering::Relaxed);
    let run_id =
        std::env::var("QUILL_PERF_RUN_ID").unwrap_or_else(|_| "unidentified-run".to_owned());
    let record = Qg1ContinuousTimingRecord {
        schema_version: "quill-qg1-continuous-timing-v1",
        admission_status: "no_claim",
        admission_no_claim_code: QG1_TIMING_DIAGNOSTIC_NO_CLAIM_CODE,
        admission_no_claim_detail: "only the continuous interval this trace names is bound into \
                                    PerfRawSample; the per-phase breakdown reaches no H2 assembler \
                                    and cannot independently support a QG-1 claim",
        run_id,
        sequence,
        gate: spec.gate.to_string(),
        fixture: spec.fixture.clone(),
        metric: spec.metric.clone(),
        writer_threads: spec.threads.unwrap_or(1),
        writer_heap_bytes: spec.writer_heap_bytes.unwrap_or(50_000_000),
        timing,
    };
    QG1_CONTINUOUS_TIMING_RECEIPTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("lock QG-1 continuous timing receipts")
        .push(record);
}

fn flush_qg1_continuous_timing_receipts(output_dir: &Path) {
    let Some(receipts) = QG1_CONTINUOUS_TIMING_RECEIPTS.get() else {
        return;
    };
    let (payload, receipt_count) = {
        let receipts = receipts
            .lock()
            .expect("lock QG-1 continuous timing receipts for flush");
        if receipts.is_empty() {
            return;
        }
        let mut payload = Vec::new();
        for row in receipts.iter() {
            serde_json::to_writer(&mut payload, row)
                .expect("serialize QG-1 continuous timing receipt");
            payload.push(b'\n');
        }
        (payload, receipts.len())
    };
    std::fs::create_dir_all(output_dir).expect("create QG-1 timing receipt directory");
    let path = output_dir.join("qg1-continuous-timing-diagnostic.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("open QG-1 continuous timing receipt");
    file.write_all(&payload)
        .expect("write QG-1 continuous timing receipts");
    eprintln!(
        "[qg1-continuous-timing] admission_status=no_claim admission_code={} receipts={} \
         sha256={} path={}",
        QG1_TIMING_DIAGNOSTIC_NO_CLAIM_CODE,
        receipt_count,
        lower_hex(&Sha256::digest(&payload)),
        display_path(&path),
    );
}

fn qg1_terminal_no_claim_detail() -> Option<String> {
    let receipts = QG1_CONTINUOUS_TIMING_RECEIPTS.get()?;
    let details = {
        let receipts = receipts
            .lock()
            .expect("lock QG-1 continuous timing receipts for terminal facts");
        receipts
            .iter()
            .flat_map(|record| record.timing.no_claim_details())
            .map(ToOwned::to_owned)
            .collect::<BTreeSet<_>>()
    };
    (!details.is_empty()).then(|| {
        format!(
            "one or more QG-1 samples retained an unproved terminal fact: {}",
            details.into_iter().collect::<Vec<_>>().join("; ")
        )
    })
}

fn fence_tantivy_lifecycle(
    index: TantivyIndex,
    spec: &PerfCellSpec,
    phase: &str,
) -> (TantivyIndex, Duration) {
    let (index, receipt) = index
        .benchmark_join_workers_and_rearm(
            spec.writer_heap_bytes.unwrap_or(50_000_000),
            spec.threads.unwrap_or(1),
        )
        .expect("join Tantivy benchmark workers and rearm writer");
    emit_tantivy_lifecycle_receipt(spec, phase, &receipt);
    (index, Duration::from_nanos(receipt.join_elapsed_ns))
}

fn finish_tantivy_lifecycle(index: TantivyIndex, spec: &PerfCellSpec, phase: &str) -> Duration {
    let receipt = index
        .benchmark_join_workers()
        .expect("join Tantivy benchmark workers without rearming");
    assert!(
        !receipt.writer_rearmed,
        "terminal Tantivy lifecycle fence unexpectedly rearmed a writer"
    );
    emit_tantivy_lifecycle_receipt(spec, phase, &receipt);
    Duration::from_nanos(receipt.join_elapsed_ns)
}

fn preflight_index<E: LexicalRead + LexicalWrite>(
    context: &BenchContext,
    index: &E,
    documents: &[IndexableDocument],
) -> Vec<String> {
    context.runtime.block_on(async {
        index
            .index_documents(&context.cx, documents)
            .await
            .expect("QG fixture preflight index");
        index
            .commit(&context.cx)
            .await
            .expect("QG fixture preflight commit");
        index
            .search(&context.cx, "term00001", 3)
            .await
            .expect("QG fixture preflight bare-term query")
            .into_iter()
            .map(|result| result.doc_id.into())
            .collect()
    })
}

/// Construct and query every selected QG-1/QG-2 indexing cell before collecting
/// timing samples. This is deliberately driven by the validated normative
/// matrix rather than a second handwritten fixture list, so a newly added
/// position/schema cell cannot reach release measurement without first proving
/// that both engines indexed the same documents and serve the same ordered
/// result.
fn preflight_indexing_fixtures(
    context: &BenchContext,
    matrix: &PerfMatrixSpec,
    selected: &[PerfCellSpec],
) {
    let documents = [
        IndexableDocument::new(
            "qg-preflight-repeated",
            "term00001 term00001 term00001 qgpreflight",
        ),
        IndexableDocument::new("qg-preflight-single", "term00001 qgpreflight"),
        IndexableDocument::new("qg-preflight-decoy", "term00002 qgpreflight"),
    ];
    for spec in selected
        .iter()
        .filter(|spec| matches!(spec.gate, PerfGate::Qg1 | PerfGate::Qg2))
    {
        assert!(
            matrix.cells.contains(spec),
            "selected {} fixture is absent from the normative matrix: {}",
            spec.gate,
            spec.fixture
        );
        if spec.metric == "tokenize_docs_per_second" {
            let mut tokenizer = FrankensearchTokenizer::default();
            let mut token_count = 0_usize;
            tokenizer.analyze(
                Analyzer::FrankensearchDefault,
                &documents[0].content,
                &mut |_| token_count = token_count.saturating_add(1),
            );
            assert!(token_count > 0, "QG tokenizer preflight emitted no terms");
            eprintln!(
                "[qg-fixture-preflight] manifest={} fixture={} schema=tokenizer-only \
                 operator=tokenize status=ok token_count={token_count}",
                matrix.manifest, spec.fixture,
            );
            continue;
        }

        let positions = spec.positions.unwrap_or(PositionMode::On);
        let expected = if positions.enabled() {
            ["qg-preflight-repeated", "qg-preflight-single"]
        } else {
            // Basic postings canonicalize both term frequencies to one while
            // retaining field lengths. The shorter single-occurrence document
            // must therefore outrank the longer repeated-term document.
            ["qg-preflight-single", "qg-preflight-repeated"]
        };
        let schema = if positions.enabled() {
            DEFAULT_SCHEMA.name
        } else {
            NO_POSITION_SCHEMA.name
        };
        let quill = quill_in_memory(spec);
        let tantivy = tantivy_in_memory(spec);
        let quill_hits = preflight_index(context, &quill, &documents);
        let tantivy_hits = preflight_index(context, &tantivy, &documents);
        assert_eq!(
            quill_hits, expected,
            "Quill QG fixture preflight changed independent expected order for {}",
            spec.fixture
        );
        assert_eq!(
            tantivy_hits, expected,
            "Tantivy QG fixture preflight changed independent expected order for {}",
            spec.fixture
        );
        eprintln!(
            "[qg-fixture-preflight] manifest={} fixture={} schema={schema} \
             operator=bare_term positions={} threads={} status=ok hits={}",
            matrix.manifest,
            spec.fixture,
            positions.label(),
            spec.threads.unwrap_or(1),
            expected.len(),
        );
    }
}

fn index_batches<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    corpus: &SyntheticCorpus,
    document_count: u64,
    update_generation: Option<u64>,
) -> Duration {
    index_batches_observed(
        context,
        index,
        corpus,
        document_count,
        update_generation,
        |_| {},
    )
}

fn index_batches_observed<E, F>(
    context: &BenchContext,
    index: &E,
    corpus: &SyntheticCorpus,
    document_count: u64,
    update_generation: Option<u64>,
    mut observe_batch: F,
) -> Duration
where
    E: LexicalWrite,
    F: FnMut(u64),
{
    let mut measured = Duration::ZERO;
    let batch_documents = context.scale.batch_documents();
    let mut start = 0_u64;
    while start < document_count {
        let remaining = document_count - start;
        let count =
            usize::try_from(remaining.min(batch_documents as u64)).expect("bounded batch count");
        let documents = generated_batch(corpus, start, count, update_generation);
        let timer = Instant::now();
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, &documents)
                .await
                .expect("QG index batch");
        });
        measured += timer.elapsed();
        let count = u64::try_from(count).expect("batch count fits u64");
        observe_batch(count);
        start = start.saturating_add(count);
    }
    measured
}

fn index_prepared_qg1_batches<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    documents: &[IndexableDocument],
) -> Duration {
    let mut measured = Duration::ZERO;
    for batch in documents.chunks(context.scale.batch_documents()) {
        let timer = Instant::now();
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, batch)
                .await
                .expect("QG-1 prepared index batch");
        });
        measured += timer.elapsed();
    }
    measured
}

fn index_prepared_qg1_batches_with_visibility_commits<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    documents: &[IndexableDocument],
    commit_cadence: Duration,
) -> (Duration, usize) {
    let mut measured = Duration::ZERO;
    let mut unpublished_since = None;
    let mut periodic_commits = 0_usize;
    for batch in documents.chunks(context.scale.batch_documents()) {
        let timer = Instant::now();
        let unpublished_started = *unpublished_since.get_or_insert(timer);
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, batch)
                .await
                .expect("QG-1 prepared index batch");
        });
        measured += timer.elapsed();
        if unpublished_started.elapsed() >= commit_cadence {
            measured += commit(context, index);
            periodic_commits = periodic_commits.saturating_add(1);
            unpublished_since = None;
        }
    }
    (measured, periodic_commits)
}

fn commit<E: LexicalWrite>(context: &BenchContext, index: &E) -> Duration {
    let timer = Instant::now();
    context.runtime.block_on(async {
        index.commit(&context.cx).await.expect("QG commit");
    });
    timer.elapsed()
}

fn feed_qg1_prepared_batches<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    prepared_input: &Qg1PreparedSampleInput<'_>,
    manual_visibility_commit_cadence: Option<Duration>,
    interval: &mut Qg1ContinuousInterval,
) -> usize {
    prepared_input
        .verify_binding(&interval.prepared_input)
        .expect("QG-1 interval must consume its exact prepared input");
    let cadence_ns = manual_visibility_commit_cadence
        .map(|cadence| u64::try_from(cadence.as_nanos()).unwrap_or(u64::MAX));
    let mut unpublished_since_ns = None;
    let mut periodic_commits = 0_usize;
    let mut start = 0_u64;
    for documents in &prepared_input.batches {
        let count_u64 = u64::try_from(documents.len()).expect("QG-1 batch count fits u64");
        if cadence_ns.is_some() && unpublished_since_ns.is_none() {
            // The first in-flight batch starts with the same exact zero as its
            // feed boundary. Do this before `begin_batch` arms the clock, so
            // no cadence bookkeeping appears between first-feed=0 and the
            // engine call.
            unpublished_since_ns = Some(if interval.origin.is_none() {
                0
            } else {
                interval.elapsed_ns()
            });
        }
        interval.begin_batch(start, count_u64);
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, documents)
                .await
                .expect("QG-1 continuous index batch");
        });
        interval.mark_batch_fed();
        if cadence_ns.is_some_and(|cadence| {
            interval
                .elapsed_ns()
                .saturating_sub(unpublished_since_ns.expect("QG-1 unpublished boundary"))
                >= cadence
        }) {
            context.runtime.block_on(async {
                index
                    .commit(&context.cx)
                    .await
                    .expect("QG-1 continuous visibility commit");
            });
            interval.mark_visibility_commit();
            periodic_commits = periodic_commits.saturating_add(1);
            unpublished_since_ns = None;
        }
        start = start
            .checked_add(count_u64)
            .expect("QG-1 prepared batch coverage fits u64");
    }
    assert_eq!(
        start, prepared_input.binding.document_count,
        "QG-1 prepared feed must cover its exact sample input"
    );
    periodic_commits
}

fn qg1_terminal_commit<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    interval: &mut Qg1ContinuousInterval,
) {
    context.runtime.block_on(async {
        index
            .commit(&context.cx)
            .await
            .expect("QG-1 continuous terminal commit");
    });
    interval.mark_terminal_commit();
}

fn qg1_quill_terminal_searchability(
    index: &QuillIndex,
    interval: &mut Qg1ContinuousInterval,
) -> Qg1TerminalFact {
    let expected_document_id = interval.prepared_input.tail_document_id.clone();
    let result = index.benchmark_search_exact_id(&expected_document_id);
    // Capture the one terminal boundary immediately when the retained Quill
    // read owner returns. Converting IDs into a proof record must not move the
    // end of the measured searchable-and-quiescent state.
    interval.mark_terminal_searchable_quiescence();
    match result {
        Ok(results) => {
            let document_ids = results
                .into_iter()
                .map(|document_id| String::from(document_id))
                .collect::<Vec<_>>();
            // Borrow rather than move: the identifier is still needed by the
            // no-claim message below and by the error arm.
            let fact = if document_ids.as_slice() == std::slice::from_ref(&expected_document_id) {
                Qg1TerminalFact::exact_tail_visible(expected_document_id.clone())
            } else {
                Qg1TerminalFact::no_claim(format!(
                    "terminal exact-ID probe returned {document_ids:?} instead of \
                     [{expected_document_id:?}]"
                ))
            };
            black_box(document_ids);
            fact
        }
        Err(error) => Qg1TerminalFact::no_claim(format!(
            "terminal exact-ID probe for {expected_document_id:?} failed: {error}"
        )),
    }
}

fn qg1_tantivy_terminal_searchability(
    reader: &BenchmarkRetainedTantivyReader,
    interval: &mut Qg1ContinuousInterval,
) -> Qg1TerminalFact {
    let expected_document_id = interval.prepared_input.tail_document_id.clone();
    let result = reader.benchmark_search_exact_id(&expected_document_id);
    // This is the actual QG-1 endpoint: Tantivy's writer workers have already
    // joined, the retained reader is still alive, and the tail query just
    // returned. No segment metadata, `drop`, assertion, or proof conversion
    // lies between that return and the single terminal timestamp.
    interval.mark_terminal_searchable_quiescence();
    match result {
        Ok(document_ids) => {
            let observed_document_ids = document_ids
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>();
            // Borrow rather than move, for the same reason as the Quill arm.
            let fact = if observed_document_ids.as_slice()
                == std::slice::from_ref(&expected_document_id)
            {
                Qg1TerminalFact::exact_tail_visible(expected_document_id.clone())
            } else {
                Qg1TerminalFact::no_claim(format!(
                    "post-join Tantivy tail lookup returned {observed_document_ids:?} instead of \
                     [{expected_document_id:?}]"
                ))
            };
            black_box(observed_document_ids);
            fact
        }
        Err(error) => Qg1TerminalFact::no_claim(format!(
            "post-join Tantivy tail lookup for {expected_document_id:?} failed: {error}"
        )),
    }
}

fn qg1_tantivy_quiescence_fact(
    terminal_join: &BenchmarkWriterJoinReceipt,
    terminal_searchability: &Qg1TerminalFact,
) -> Qg1TerminalFact {
    if !terminal_join.writer_rearmed
        && let Some(tail_document_id) = terminal_searchability.exact_tail_document_id()
    {
        Qg1TerminalFact::tantivy_join_then_exact_tail(tail_document_id, *terminal_join)
    } else {
        Qg1TerminalFact::no_claim(format!(
            "Tantivy terminal lifecycle did not prove one nonrearming worker join followed by \
             a retained-reader tail search: terminal_join=({},{},rearmed={}), \
             tail_search={terminal_searchability:?}",
            terminal_join.searchable_segments_before,
            terminal_join.searchable_segments_after,
            terminal_join.writer_rearmed,
        ))
    }
}

fn qg1_bulk_metric_continuous(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    count: u64,
    tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
) -> MetricMeasurement {
    assert_eq!(
        qg1_producer_coverage(spec),
        Some(Qg1ProducerCoverage::EngineIndexingLifecycle),
        "continuous QG-1 engine lifecycle is reserved for docs_per_second indexing arms"
    );
    let measurement = match arm {
        EngineArm::Quill => {
            let prepared_input = context.qg1_sample_input(count);
            let index = quill_in_memory(spec);
            let mut interval = Qg1ContinuousInterval::start(arm, prepared_input.binding.clone());
            let periodic_commit_calls =
                feed_qg1_prepared_batches(context, &index, &prepared_input, None, &mut interval);
            let generation_before_terminal = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            qg1_terminal_commit(context, &index, &mut interval);
            let generation_after_terminal = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            // Retain the Quill read owner until its terminal search returns,
            // matching Tantivy's retained-reader endpoint without inventing a
            // writer lifecycle that Quill does not have.
            let terminal_searchability = qg1_quill_terminal_searchability(&index, &mut interval);
            let generation_delta =
                generation_after_terminal.saturating_sub(generation_before_terminal);
            let terminal_quiescence = if generation_delta > 0 {
                Qg1TerminalFact::quill_publication_then_exact_tail(
                    prepared_input.binding.tail_document_id.clone(),
                    generation_delta,
                )
            } else {
                Qg1TerminalFact::no_claim(
                    "QG-1 Quill terminal publishing commit did not advance generation",
                )
            };
            let (measurement, receipt) = interval.finish(
                Some(generation_delta),
                terminal_searchability,
                terminal_quiescence,
            );
            emit_qg1_continuous_timing_receipt(spec, receipt);
            eprintln!(
                "[qg-commit-parity] gate={} fixture={} arm=quill cadence_ms={} \
                 explicit_periodic_commit_calls={} terminal_publication_generation_delta={} \
                 terminal_commit_calls=1 \
                 pre_search_rearm_join_calls=0 terminal_search_calls=1 \
                 terminal_worker_join_calls=0 \
                 durability=in_memory continuous_elapsed_ns={}",
                spec.gate,
                spec.fixture,
                quill_config(spec).max_visibility_lag_ms,
                periodic_commit_calls,
                generation_delta,
                measurement.elapsed_ns,
            );
            measurement
        }
        EngineArm::Tantivy => {
            let prepared_input = context.qg1_sample_input(count);
            let writer_mode = tantivy_writer_mode.unwrap_or(Qg1TantivyWriterMode::Fixed {
                writer_threads: spec.threads.unwrap_or(1),
            });
            let index = qg1_tantivy_in_memory(spec, writer_mode);
            let receipt = index
                .benchmark_writer_receipt()
                .expect("QG-1 Tantivy arm uses an authenticated benchmark constructor");
            qg1_validate_writer_receipt(spec, writer_mode, receipt);
            assert_eq!(
                index.benchmark_materialized_writer_threads(),
                qg1_expected_materialized_width(writer_mode),
                "QG-1 constructor width accessor disagrees with its typed receipt"
            );
            if let Some(observed_threads) = index.benchmark_materialized_writer_threads() {
                record_concurrency(spec, arm, observed_threads);
            }
            let mut interval = Qg1ContinuousInterval::start(arm, prepared_input.binding.clone());
            let periodic_commits = feed_qg1_prepared_batches(
                context,
                &index,
                &prepared_input,
                Some(Duration::from_millis(
                    quill_config(spec).max_visibility_lag_ms,
                )),
                &mut interval,
            );
            qg1_terminal_commit(context, &index, &mut interval);
            let (retained_search_owner, terminal_join_receipt) = index
                .benchmark_join_workers_retaining_reader()
                .expect("join QG-1 Tantivy terminal workers while retaining a read handle");
            assert!(
                !terminal_join_receipt.writer_rearmed,
                "QG-1 terminal Tantivy worker fence must not construct a replacement writer"
            );
            interval.mark_terminal_worker_join(terminal_join_receipt);
            let terminal_searchability =
                qg1_tantivy_terminal_searchability(&retained_search_owner, &mut interval);
            let terminal_quiescence =
                qg1_tantivy_quiescence_fact(&terminal_join_receipt, &terminal_searchability);
            let (measurement, receipt) =
                interval.finish(None, terminal_searchability, terminal_quiescence);
            emit_tantivy_lifecycle_receipt(
                spec,
                "qg1_terminal_worker_join_before_retained_tail_search",
                &terminal_join_receipt,
            );
            emit_qg1_continuous_timing_receipt(spec, receipt);
            eprintln!(
                "[qg-commit-parity] gate={} fixture={} arm=tantivy cadence_ms={} \
                 explicit_periodic_commit_calls={periodic_commits} terminal_commit_calls=1 \
                 pre_search_rearm_join_calls=0 terminal_search_calls=1 \
                 terminal_worker_join_calls=1 \
                 durability=in_memory continuous_elapsed_ns={}",
                spec.gate,
                spec.fixture,
                quill_config(spec).max_visibility_lag_ms,
                measurement.elapsed_ns,
            );
            measurement
        }
    };
    assert_eq!(
        measurement.work_units, count,
        "QG-1 continuous interval must cover the exact requested document count"
    );
    assert!(
        measurement.elapsed_ns > 0,
        "QG-1 continuous interval must span positive monotonic time"
    );
    MetricMeasurement {
        value: throughput_per_second(measurement.work_units, measurement.elapsed_ns),
        continuous: Some(measurement),
        qg2_continuous: None,
    }
}

/// The exact identifier of the last document a generated feed of `count`
/// documents writes, resolved through the same corpus conversion the feed
/// itself uses so the probe cannot drift from what was indexed.
fn generated_tail_document_id(corpus: &SyntheticCorpus, count: u64) -> String {
    let tail_ordinal = count
        .checked_sub(1)
        .expect("a QG-2 continuous feed indexes at least one document");
    let document: IndexableDocument = corpus
        .document_at(tail_ordinal % corpus.len())
        .expect("generated tail document ordinal")
        .into();
    document.id
}

/// Measure one QG-2 update cell as a single continuous interval.
///
/// The defect this replaces summed two independently timed calls — the batch
/// feed and the terminal commit — and stopped there. Two sums are not one
/// interval: the time between them is unattributed, and stopping at commit
/// stops before the work a user is actually waiting for, which is the update
/// becoming *searchable* and the engine going *quiescent*. Everything after the
/// first feed and up to that terminal state is inside the measured span here,
/// and both arms end at the same kind of endpoint: a tail-document search
/// served by a retained reader, after the engine's writer side has settled.
///
/// This deliberately does not reuse `qg1_bulk_metric_continuous`: that path
/// asserts QG-1 producer coverage, consumes a QG-1 prepared input, and emits
/// QG-1 lifecycle receipts. Sharing its SHAPE is correct; sharing its
/// attestations would file QG-2 work under QG-1 authority.
fn qg2_bulk_metric_continuous(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    count: u64,
) -> MetricMeasurement {
    qg2_bulk_metric_continuous_with_planted_tail_delay(context, spec, arm, count, None)
}

/// [`qg2_bulk_metric_continuous`] with a bounded delay planted between the
/// terminal commit and the terminal searchable/quiescent endpoint.
///
/// Production always passes `None`, so the shipping path is byte-identical to
/// the function above. The seam exists because tail inclusion cannot be proved
/// by comparing two runs: scheduler and cache variation can make a run that
/// covers MORE lifecycle finish sooner. Planting a known interval inside the
/// endpoint and measuring both shapes in ONE invocation makes the difference
/// deterministic — the delay is by construction inside the continuous span and
/// outside the summed feed-plus-commit span.
fn qg2_bulk_metric_continuous_with_planted_tail_delay(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    count: u64,
    planted_tail_delay: Option<Duration>,
) -> MetricMeasurement {
    let corpus = corpus_for(count);
    let tail_document_id = generated_tail_document_id(&corpus, count);
    let (origin, elapsed_ns, feed_and_commit_ns, quiescence) = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            // The interval opens at the first feed: corpus and index
            // construction are setup, not update throughput.
            let origin = Instant::now();
            let feed = index_batches(context, &index, &corpus, count, None);
            let commit_elapsed = commit(context, &index);
            let feed_and_commit_ns = u64::try_from((feed + commit_elapsed).as_nanos())
                .expect("summed feed and commit fits u64 ns");
            if let Some(delay) = planted_tail_delay {
                std::thread::sleep(delay);
            }
            let visible = index
                .benchmark_search_exact_id(&tail_document_id)
                .expect("QG-2 Quill terminal exact-ID probe");
            // Closed the instant the retained read owner returns, before any
            // proof bookkeeping, so converting the result into an assertion
            // cannot extend the measured state.
            let elapsed_ns = u64::try_from(origin.elapsed().as_nanos()).expect("monotonic ns");
            let generation_after = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            // Quiescence basis, stated rather than inferred: a tail that reads
            // back proves visibility only. What proves nothing is still in
            // flight behind it is that the terminal commit actually published a
            // generation, so that delta is the recorded basis.
            let delta = generation_after.saturating_sub(generation_before);
            assert!(
                delta > 0,
                "QG-2 Quill terminal commit must publish a new generation; a tail that merely \
                 reads back does not establish quiescence"
            );
            assert_eq!(
                visible
                    .into_iter()
                    .map(String::from)
                    .collect::<Vec<_>>()
                    .as_slice(),
                std::slice::from_ref(&tail_document_id),
                "QG-2 Quill interval must end with the exact tail document searchable"
            );
            (
                origin,
                elapsed_ns,
                feed_and_commit_ns,
                Qg2QuiescenceBasis::QuillPublishedGeneration { delta },
            )
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            let origin = Instant::now();
            let feed = index_batches(context, &index, &corpus, count, None);
            let commit_elapsed = commit(context, &index);
            let feed_and_commit_ns = u64::try_from((feed + commit_elapsed).as_nanos())
                .expect("summed feed and commit fits u64 ns");
            if let Some(delay) = planted_tail_delay {
                std::thread::sleep(delay);
            }
            // Symmetric endpoint: Tantivy's writer workers must have settled
            // before its retained reader answers, which is the same
            // searchable-and-quiescent state the Quill arm ends in.
            let (retained_search_owner, terminal_join_receipt) = index
                .benchmark_join_workers_retaining_reader()
                .expect("join QG-2 Tantivy terminal workers while retaining a read handle");
            assert!(
                !terminal_join_receipt.writer_rearmed,
                "QG-2 terminal Tantivy worker fence must not construct a replacement writer"
            );
            let visible = retained_search_owner
                .benchmark_search_exact_id(&tail_document_id)
                .expect("QG-2 Tantivy terminal exact-ID probe");
            let elapsed_ns = u64::try_from(origin.elapsed().as_nanos()).expect("monotonic ns");
            assert_eq!(
                visible
                    .into_iter()
                    .map(String::from)
                    .collect::<Vec<_>>()
                    .as_slice(),
                std::slice::from_ref(&tail_document_id),
                "QG-2 Tantivy interval must end with the exact tail document searchable"
            );
            (
                origin,
                elapsed_ns,
                feed_and_commit_ns,
                // Symmetric basis: the writer workers joined and no replacement
                // writer was armed, so nothing is still running behind the
                // retained reader that just answered.
                Qg2QuiescenceBasis::TantivyWorkersJoined {
                    rearmed: terminal_join_receipt.writer_rearmed,
                },
            )
        }
    };
    assert!(
        elapsed_ns > 0,
        "QG-2 continuous interval must span positive monotonic time"
    );
    MetricMeasurement {
        value: throughput_per_second(count, elapsed_ns),
        continuous: None,
        qg2_continuous: Some(Qg2ContinuousMeasurement {
            work_units: count,
            origin,
            elapsed_ns,
            feed_and_commit_ns,
            quiescence,
        }),
    }
}

fn bulk_metric_unpooled(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    qg1_tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
) -> MetricMeasurement {
    let requested = spec.document_count.expect("bulk document count");
    let count = context.scale.document_count(requested);
    if spec.gate == PerfGate::Qg1 {
        return qg1_bulk_metric_continuous(context, spec, arm, count, qg1_tantivy_writer_mode);
    }
    let prepared_qg1_documents = (spec.gate == PerfGate::Qg1).then(|| context.qg1_prefix(count).1);
    let generated_corpus = (spec.gate != PerfGate::Qg1).then(|| corpus_for(count));
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            let mut elapsed = prepared_qg1_documents.map_or_else(
                || {
                    index_batches(
                        context,
                        &index,
                        generated_corpus
                            .as_ref()
                            .expect("non-QG-1 bulk cell has a generated corpus"),
                        count,
                        None,
                    )
                },
                |documents| index_prepared_qg1_batches(context, &index, documents),
            );
            let generation_after = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .generation;
            elapsed += commit(context, &index);
            if spec.gate == PerfGate::Qg1 {
                eprintln!(
                    "[qg-commit-parity] gate={} fixture={} arm=quill cadence_ms={} \
                     periodic_commits={} terminal_commit_calls=1 durability=in_memory",
                    spec.gate,
                    spec.fixture,
                    quill_config(spec).max_visibility_lag_ms,
                    generation_after.saturating_sub(generation_before),
                );
            }
            elapsed
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            if matches!(spec.gate, PerfGate::Qg1 | PerfGate::Qg8) {
                let observed_threads = index
                    .benchmark_materialized_writer_threads()
                    .expect("scaling Tantivy arm uses the benchmark writer constructor");
                record_concurrency(spec, arm, observed_threads);
            }
            let (mut elapsed, periodic_commits) = if spec.gate == PerfGate::Qg1 {
                index_prepared_qg1_batches_with_visibility_commits(
                    context,
                    &index,
                    prepared_qg1_documents.expect("QG-1 bulk cell has a prepared immutable corpus"),
                    Duration::from_millis(quill_config(spec).max_visibility_lag_ms),
                )
            } else {
                (
                    index_batches(
                        context,
                        &index,
                        generated_corpus
                            .as_ref()
                            .expect("non-QG-1 bulk cell has a generated corpus"),
                        count,
                        None,
                    ),
                    0,
                )
            };
            elapsed += commit(context, &index);
            if spec.gate == PerfGate::Qg1 {
                eprintln!(
                    "[qg-commit-parity] gate={} fixture={} arm=tantivy cadence_ms={} \
                     periodic_commits={periodic_commits} terminal_commit_calls=1 \
                     durability=in_memory",
                    spec.gate,
                    spec.fixture,
                    quill_config(spec).max_visibility_lag_ms,
                );
            }
            elapsed += finish_tantivy_lifecycle(index, spec, "measured_work");
            elapsed
        }
    };
    // A sum of independently timed calls, so it carries no continuous interval
    // and cannot be published as throughput.
    MetricMeasurement::gauge(count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE))
}

fn bulk_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> MetricMeasurement {
    bulk_metric_with_qg1_writer_mode(context, spec, arm, None)
}

fn bulk_metric_with_qg1_writer_mode(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    qg1_tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
) -> MetricMeasurement {
    if !matches!(spec.gate, PerfGate::Qg1 | PerfGate::Qg8) || arm != EngineArm::Quill {
        return bulk_metric_unpooled(context, spec, arm, qg1_tantivy_writer_mode);
    }

    let threads = spec.threads.expect("QG-1/QG-8 thread count");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build QG-1/QG-8 Quill thread pool")
        .install(|| {
            let observed_threads = rayon::current_num_threads();
            assert_eq!(
                observed_threads, threads,
                "QG-1/QG-8 Quill cell escaped its pinned Rayon pool"
            );
            record_concurrency(spec, arm, observed_threads);
            bulk_metric_unpooled(context, spec, arm, qg1_tantivy_writer_mode)
        })
}

fn tokenize_metric(context: &BenchContext, spec: &PerfCellSpec) -> f64 {
    let count = context
        .scale
        .document_count(spec.document_count.expect("tokenize document count"));
    assert_eq!(
        qg1_producer_coverage(spec),
        Some(Qg1ProducerCoverage::TokenizerOnlyDiagnosticNoEngineLifecycle),
        "prepared tokenizer timing is a QG-1 diagnostic without engine lifecycle proof"
    );
    let (_, documents) = context.qg1_prefix(count);
    let mut tokenizer = FrankensearchTokenizer::default();
    let mut measured = Duration::ZERO;
    for batch in documents.chunks(context.scale.batch_documents()) {
        let timer = Instant::now();
        let mut token_count = 0_usize;
        for document in batch {
            tokenizer.analyze(
                Analyzer::FrankensearchDefault,
                black_box(&document.content),
                &mut |_| token_count = token_count.saturating_add(1),
            );
        }
        measured += timer.elapsed();
        black_box(token_count);
    }
    count as f64 / measured.as_secs_f64().max(f64::MIN_POSITIVE)
}

fn watch_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let warm_count = context
        .scale
        .document_count(PerfCorpus::Medium.document_count());
    let update_count = context
        .scale
        .document_count(spec.document_count.expect("watch update count"));
    let corpus = corpus_for(warm_count);
    let topology = spec.topology.expect("watch topology");
    let (probe_query, expected_doc_id) = update_probe(&corpus, update_count, 1);
    let elapsed = match (arm, topology) {
        (EngineArm::Quill, PerfTopology::InProcess) => {
            let index = quill_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, warm_count, None);
            let _ = commit(context, &index);
            let mut elapsed = index_batches(context, &index, &corpus, update_count, Some(1));
            elapsed += commit(context, &index);
            let timer = Instant::now();
            assert_exact_visibility(context, &index, &probe_query, &expected_doc_id);
            elapsed + timer.elapsed()
        }
        (EngineArm::Tantivy, PerfTopology::InProcess) => {
            let index = tantivy_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, warm_count, None);
            let _ = commit(context, &index);
            let (index, _) = fence_tantivy_lifecycle(index, spec, "warm_fixture");
            let mut elapsed = index_batches(context, &index, &corpus, update_count, Some(1));
            elapsed += commit(context, &index);
            let (index, join_elapsed) = fence_tantivy_lifecycle(index, spec, "measured_update");
            elapsed += join_elapsed;
            let timer = Instant::now();
            assert_exact_visibility(context, &index, &probe_query, &expected_doc_id);
            elapsed + timer.elapsed()
        }
        (EngineArm::Quill, PerfTopology::FreshProcess) => measure_quill_fresh_process(
            context,
            spec,
            &corpus,
            warm_count,
            update_count,
            &probe_query,
            &expected_doc_id,
        ),
        (EngineArm::Tantivy, PerfTopology::FreshProcess) => measure_tantivy_fresh_process(
            context,
            spec,
            &corpus,
            warm_count,
            update_count,
            &probe_query,
            &expected_doc_id,
        ),
    };
    if spec.metric == "updates_per_second" {
        update_count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
    } else {
        elapsed.as_secs_f64() * 1_000.0
    }
}

fn update_probe(corpus: &SyntheticCorpus, update_count: u64, generation: u64) -> (String, String) {
    let ordinal = update_count.saturating_sub(1);
    let expected_doc_id = corpus
        .document_at(ordinal % corpus.len())
        .expect("QG-3 visibility probe document")
        .id;
    (format!("qgupdateg{generation}d{ordinal}"), expected_doc_id)
}

fn assert_exact_visibility<E: LexicalRead>(
    context: &BenchContext,
    index: &E,
    query: &str,
    expected_doc_id: &str,
) {
    let doc_ids = search_doc_ids(context, index, query);
    assert_eq!(
        doc_ids,
        [expected_doc_id.to_owned()],
        "visibility fence accepted stale, missing, or ambiguous state"
    );
    black_box(doc_ids);
}

fn assert_absent<E: LexicalRead>(context: &BenchContext, index: &E, query: &str) {
    let doc_ids = search_doc_ids(context, index, query);
    assert!(
        doc_ids.is_empty(),
        "deleted QG fixture document remained query-visible: {doc_ids:?}"
    );
    black_box(doc_ids);
}

fn search_doc_ids<E: LexicalRead>(context: &BenchContext, index: &E, query: &str) -> Vec<String> {
    context.runtime.block_on(async {
        index
            .search(&context.cx, query, 3)
            .await
            .expect("QG exact-version visibility query")
            .into_iter()
            .map(|result| result.doc_id.into())
            .collect::<Vec<String>>()
    })
}

fn scratch_root() -> PathBuf {
    let root = std::env::var_os("QUILL_PERF_SCRATCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("frankensearch-quill-perf"));
    std::fs::create_dir_all(&root).expect("QG scratch root");
    root
}

fn scratch_path(label: &str) -> PathBuf {
    let root = scratch_root();
    let sequence = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
    root.join(format!("{label}-{}-{sequence}", std::process::id()))
}

fn scratch_tempdir(prefix: &str) -> tempfile::TempDir {
    let root = scratch_root();
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(&root)
        .expect("create QG fixture directory under declared scratch root")
}

fn measure_quill_fresh_process(
    context: &BenchContext,
    spec: &PerfCellSpec,
    corpus: &SyntheticCorpus,
    warm_count: u64,
    update_count: u64,
    probe_query: &str,
    expected_doc_id: &str,
) -> Duration {
    let path = scratch_path("qg3-quill");
    let index =
        context
            .runtime
            .block_on(QuillIndex::create(&context.cx, &path, quill_config(spec)));
    let index = index.expect("create on-disk Quill watch fixture");
    let _ = index_batches(context, &index, corpus, warm_count, None);
    let _ = commit(context, &index);
    let mut elapsed = index_batches(context, &index, corpus, update_count, Some(1));
    elapsed += commit(context, &index);
    drop(index);
    elapsed + fresh_process_search(&path, spec, EngineArm::Quill, probe_query, expected_doc_id)
}

fn measure_tantivy_fresh_process(
    context: &BenchContext,
    spec: &PerfCellSpec,
    corpus: &SyntheticCorpus,
    warm_count: u64,
    update_count: u64,
    probe_query: &str,
    expected_doc_id: &str,
) -> Duration {
    let path = scratch_path("qg3-tantivy");
    let index = tantivy_create(&path, spec);
    let _ = index_batches(context, &index, corpus, warm_count, None);
    let _ = commit(context, &index);
    let (index, _) = fence_tantivy_lifecycle(index, spec, "warm_fixture");
    let mut elapsed = index_batches(context, &index, corpus, update_count, Some(1));
    elapsed += commit(context, &index);
    let (index, join_elapsed) = fence_tantivy_lifecycle(index, spec, "measured_update");
    elapsed += join_elapsed;
    drop(index);
    elapsed
        + fresh_process_search(
            &path,
            spec,
            EngineArm::Tantivy,
            probe_query,
            expected_doc_id,
        )
}

fn fresh_process_search(
    path: &Path,
    spec: &PerfCellSpec,
    arm: EngineArm,
    probe_query: &str,
    expected_doc_id: &str,
) -> Duration {
    let timer = Instant::now();
    let output = Command::new(std::env::current_exe().expect("QG benchmark executable"))
        .env("QUILL_PERF_CHILD_MODE", "search")
        .env("QUILL_PERF_CHILD_ENGINE", arm.label())
        .env("QUILL_PERF_CHILD_PATH", path)
        .env("QUILL_PERF_CHILD_QUERY", probe_query)
        .env("QUILL_PERF_CHILD_EXPECTED_DOC_ID", expected_doc_id)
        .env(
            "QUILL_PERF_CHILD_HEAP",
            spec.writer_heap_bytes.unwrap_or(50_000_000).to_string(),
        )
        .env(
            "QUILL_PERF_CHILD_THREADS",
            spec.threads.unwrap_or(1).to_string(),
        )
        .env(
            "QUILL_PERF_CHILD_POSITIONS",
            spec.positions
                .unwrap_or(PositionMode::On)
                .enabled()
                .to_string(),
        )
        .output()
        .expect("spawn fresh-process reader");
    assert!(
        output.status.success(),
        "fresh-process reader failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    black_box(output.stdout);
    timer.elapsed()
}

fn collect_regular_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(path)
        .map_err(|error| format!("read QG-9 index directory {}: {error}", path.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read QG-9 index directory entry under {}: {error}",
                path.display()
            )
        })?;
        let file_type = entry.file_type().map_err(|error| {
            format!(
                "inspect QG-9 index entry {}: {error}",
                entry.path().display()
            )
        })?;
        if file_type.is_dir() {
            collect_regular_files(&entry.path(), files)?;
        } else if file_type.is_file() {
            files.push(entry.path());
        }
    }
    Ok(())
}

/// Request a per-index-file cold page cache without evicting unrelated workloads.
///
/// QG-9 never treats this advisory as proved unless both engine arms succeed on
/// every sampled fixture. The timed open itself is then performed by a fresh
/// executable, so no reader state from the producer process survives.
fn evict_index_file_cache(path: &Path) -> Result<usize, String> {
    #[cfg(target_os = "linux")]
    {
        let mut files = Vec::new();
        collect_regular_files(path, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Err(format!(
                "QG-9 index {} has no regular files",
                path.display()
            ));
        }
        for file_path in &files {
            let file = fs::File::open(file_path).map_err(|error| {
                format!(
                    "open QG-9 index file {} for cache eviction: {error}",
                    file_path.display()
                )
            })?;
            rustix::fs::fadvise(&file, 0, None, rustix::fs::Advice::DontNeed).map_err(|error| {
                format!(
                    "posix_fadvise(POSIX_FADV_DONTNEED) for QG-9 index file {}: {error}",
                    file_path.display()
                )
            })?;
        }
        Ok(files.len())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err("QG-9 has no registered per-file cold-cache eviction method for this OS".to_owned())
    }
}

fn fresh_process_open(path: &Path, spec: &PerfCellSpec, arm: EngineArm) -> Duration {
    let output = Command::new(std::env::current_exe().expect("QG benchmark executable"))
        .env("QUILL_PERF_CHILD_MODE", "open")
        .env("QUILL_PERF_CHILD_ENGINE", arm.label())
        .env("QUILL_PERF_CHILD_PATH", path)
        .env(
            "QUILL_PERF_CHILD_HEAP",
            spec.writer_heap_bytes.unwrap_or(50_000_000).to_string(),
        )
        .env(
            "QUILL_PERF_CHILD_THREADS",
            spec.threads.unwrap_or(1).to_string(),
        )
        .env(
            "QUILL_PERF_CHILD_POSITIONS",
            spec.positions
                .unwrap_or(PositionMode::On)
                .enabled()
                .to_string(),
        )
        .output()
        .expect("spawn fresh-process QG-9 reader");
    assert!(
        output.status.success(),
        "fresh-process QG-9 reader failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("QG-9 child output UTF-8");
    let nanos = stdout
        .lines()
        .find_map(|line| line.strip_prefix("quill-perf-child\\t"))
        .expect("QG-9 child open measurement")
        .parse::<u64>()
        .expect("QG-9 child open measurement nanoseconds");
    Duration::from_nanos(nanos)
}

fn commit_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let warm_count = context
        .scale
        .document_count(spec.document_count.expect("commit warm count"));
    let corpus = corpus_for(warm_count.saturating_add(1));
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, warm_count, None);
            let _ = commit(context, &index);
            let document = generated_batch(&corpus, warm_count, 1, None);
            context.runtime.block_on(async {
                index
                    .index_documents(&context.cx, &document)
                    .await
                    .expect("stage Quill commit probe");
            });
            commit(context, &index)
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, warm_count, None);
            let _ = commit(context, &index);
            let document = generated_batch(&corpus, warm_count, 1, None);
            context.runtime.block_on(async {
                index
                    .index_documents(&context.cx, &document)
                    .await
                    .expect("stage Tantivy commit probe");
            });
            commit(context, &index)
        }
    };
    elapsed.as_secs_f64() * 1_000.0
}

fn compaction_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let count = context
        .scale
        .document_count(spec.document_count.expect("compaction count"));
    let density = spec
        .tombstone_density_pct
        .expect("nonzero compaction density");
    let corpus = corpus_for(count);
    let segments = context.scale.segments();
    let docs_per_segment = usize::try_from(count)
        .expect("compaction count fits usize")
        .div_ceil(segments);
    let elapsed = match arm {
        EngineArm::Quill => {
            let directory = scratch_tempdir("qg5-quill-");
            let index = context
                .runtime
                .block_on(QuillIndex::create(
                    &context.cx,
                    directory.path(),
                    quill_config(spec),
                ))
                .expect("QG-5 create durable Quill fixture");
            for segment in 0..segments {
                let start =
                    u64::try_from(segment.saturating_mul(docs_per_segment)).expect("segment start");
                if start >= count {
                    break;
                }
                let segment_count = usize::try_from((count - start).min(docs_per_segment as u64))
                    .expect("segment count");
                let documents = generated_batch(&corpus, start, segment_count, Some(5));
                context.runtime.block_on(async {
                    index
                        .index_documents(&context.cx, &documents)
                        .await
                        .expect("Quill compaction fixture batch");
                    index.commit(&context.cx).await.expect("Quill fixture seal");
                });
            }
            stage_deletes(context, &index, &corpus, count, density);
            let threshold = (f64::from(density) / 100.0 - 0.001).max(0.000_001);
            let timer = Instant::now();
            context.runtime.block_on(async {
                black_box(
                    index
                        .compact(&context.cx, CompactionPolicy::new(threshold))
                        .await
                        .expect("Quill full compaction"),
                );
            });
            let elapsed = timer.elapsed();
            validate_compaction_outcome(context, &index, &corpus, count);
            drop(index);
            let reopened = context
                .runtime
                .block_on(QuillIndex::open(
                    &context.cx,
                    directory.path(),
                    quill_config(spec),
                ))
                .expect("reopen compacted Quill fixture");
            validate_compaction_outcome(context, &reopened, &corpus, count);
            elapsed
        }
        EngineArm::Tantivy => {
            let directory = scratch_tempdir("qg5-tantivy-");
            let index = tantivy_create(directory.path(), spec);
            context.runtime.block_on(async {
                index
                    .benchmark_disable_auto_merge(&context.cx)
                    .await
                    .expect("disable Tantivy auto merge");
            });
            for segment in 0..segments {
                let start =
                    u64::try_from(segment.saturating_mul(docs_per_segment)).expect("segment start");
                if start >= count {
                    break;
                }
                let segment_count = usize::try_from((count - start).min(docs_per_segment as u64))
                    .expect("segment count");
                let documents = generated_batch(&corpus, start, segment_count, Some(5));
                context.runtime.block_on(async {
                    index
                        .index_documents(&context.cx, &documents)
                        .await
                        .expect("Tantivy compaction fixture batch");
                    index
                        .commit(&context.cx)
                        .await
                        .expect("Tantivy fixture seal");
                });
            }
            let deleted = count.saturating_mul(u64::from(density)) / 100;
            for ordinal in 0..deleted {
                let source = ordinal.saturating_mul(count / deleted.max(1));
                let id = corpus
                    .document_at(source.min(count.saturating_sub(1)))
                    .expect("Tantivy deleted document")
                    .id;
                context.runtime.block_on(async {
                    index
                        .delete_document(&context.cx, &id)
                        .await
                        .expect("stage Tantivy tombstone");
                });
            }
            let _ = commit(context, &index);
            let timer = Instant::now();
            context.runtime.block_on(async {
                index
                    .benchmark_force_merge(&context.cx)
                    .await
                    .expect("Tantivy force merge");
            });
            let elapsed = timer.elapsed();
            validate_compaction_outcome(context, &index, &corpus, count);
            drop(index);
            let reopened = TantivyIndex::open_with_benchmark_config(
                directory.path(),
                spec.writer_heap_bytes.unwrap_or(50_000_000),
                spec.threads.unwrap_or(1),
                spec.positions.unwrap_or(PositionMode::On).enabled(),
            )
            .expect("reopen compacted Tantivy fixture");
            validate_compaction_outcome(context, &reopened, &corpus, count);
            elapsed
        }
    };
    elapsed.as_secs_f64() * 1_000.0
}

fn validate_compaction_outcome<E: LexicalRead>(
    context: &BenchContext,
    index: &E,
    corpus: &SyntheticCorpus,
    count: u64,
) {
    let deleted_ordinal = 0_u64;
    let live_ordinal = count.saturating_sub(1);
    let live_doc_id = corpus
        .document_at(live_ordinal)
        .expect("QG-5 live probe document")
        .id;
    assert_absent(context, index, &format!("qgupdateg5d{deleted_ordinal}"));
    assert_exact_visibility(
        context,
        index,
        &format!("qgupdateg5d{live_ordinal}"),
        &live_doc_id,
    );
}

fn stage_deletes(
    context: &BenchContext,
    index: &QuillIndex,
    corpus: &SyntheticCorpus,
    count: u64,
    density: u8,
) {
    let deleted = count.saturating_mul(u64::from(density)) / 100;
    const DELETE_BATCH: u64 = 10_000;
    let mut start = 0_u64;
    while start < deleted {
        let end = (start + DELETE_BATCH).min(deleted);
        let ids = (start..end)
            .map(|ordinal| {
                let source = ordinal.saturating_mul(count / deleted.max(1));
                corpus
                    .document_at(source.min(count.saturating_sub(1)))
                    .expect("Quill deleted document")
                    .id
            })
            .collect::<Vec<_>>();
        let id_refs = ids.iter().map(String::as_str).collect::<Vec<_>>();
        context.runtime.block_on(async {
            assert_eq!(
                index
                    .delete_documents(&context.cx, &id_refs)
                    .await
                    .expect("stage Quill tombstones"),
                id_refs.len(),
                "every QG-5 tombstone must target a live document"
            );
        });
        start = end;
    }
}

fn query_specs(query_class: PerfQueryClass) -> Vec<Qg6QuerySpec> {
    Qg6QuerySpec::normative_for_class(query_class)
        .expect("frozen QG-6 query manifest must validate before preparation")
}

fn query_text(query_class: PerfQueryClass) -> String {
    query_specs(query_class)[0].text().to_owned()
}

fn query_metric(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    query_override: Option<&str>,
) -> f64 {
    let count = context
        .scale
        .document_count(spec.document_count.expect("query corpus count"));
    let corpus = corpus_for(count);
    let owned_query = query_override
        .is_none()
        .then(|| query_text(spec.query_class.expect("query class")));
    let query = query_override.unwrap_or_else(|| {
        owned_query
            .as_deref()
            .expect("owned query exists without an override")
    });
    let k = spec.k.expect("query k");
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, count, None);
            let _ = commit(context, &index);
            let timer = Instant::now();
            black_box(
                index
                    .search_doc_ids(&context.cx, black_box(query), black_box(k))
                    .expect("QG Quill query"),
            );
            timer.elapsed()
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, count, None);
            let _ = commit(context, &index);
            let timer = Instant::now();
            black_box(
                index
                    .search_doc_ids(&context.cx, black_box(query), black_box(k))
                    .expect("QG Tantivy query"),
            );
            timer.elapsed()
        }
    };
    elapsed.as_secs_f64() * 1_000.0
}

fn memory_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let count = context
        .scale
        .document_count(spec.document_count.expect("memory corpus count"));
    let executable = std::env::current_exe().expect("QG benchmark executable");
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("/usr/bin/time");
        command.arg("-l").arg(&executable);
        command
    };
    #[cfg(not(target_os = "macos"))]
    let mut command = Command::new(executable);
    let output = command
        .env("QUILL_PERF_CHILD_MODE", "memory")
        .env("QUILL_PERF_CHILD_ENGINE", arm.label())
        .env("QUILL_PERF_CHILD_COUNT", count.to_string())
        .env(
            "QUILL_PERF_CHILD_HEAP",
            spec.writer_heap_bytes.unwrap_or(50_000_000).to_string(),
        )
        .env(
            "QUILL_PERF_CHILD_THREADS",
            spec.threads.unwrap_or(1).to_string(),
        )
        .env(
            "QUILL_PERF_CHILD_POSITIONS",
            spec.positions
                .unwrap_or(PositionMode::On)
                .enabled()
                .to_string(),
        )
        .output()
        .expect("spawn isolated RSS probe");
    assert!(
        output.status.success(),
        "isolated RSS probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("RSS child output UTF-8");
    let measurement = stdout
        .lines()
        .find_map(|line| line.strip_prefix("quill-perf-child\t"))
        .expect("RSS child measurement");
    let measurement_columns = measurement
        .split_once('\t')
        .expect("RSS child measurement columns");
    if spec.metric == "peak_rss_bytes" {
        #[cfg(target_os = "macos")]
        let rss_bytes = frankensearch_quill_gauntlet::parse_macos_time_max_rss_bytes(
            std::str::from_utf8(&output.stderr).expect("macOS time report UTF-8"),
        )
        .expect("macOS time report peak RSS row");
        #[cfg(not(target_os = "macos"))]
        let rss_bytes = measurement_columns
            .0
            .parse::<u64>()
            .expect("RSS child byte count");
        rss_bytes as f64
    } else {
        measurement_columns
            .1
            .parse::<u64>()
            .expect("index child byte count") as f64
            / count as f64
    }
}

fn cold_open_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let count = context
        .scale
        .document_count(spec.document_count.expect("cold-open corpus count"));
    let corpus = corpus_for(count);
    let elapsed = match arm {
        EngineArm::Quill => {
            let path = scratch_path("qg9-quill");
            let index = context.runtime.block_on(QuillIndex::create(
                &context.cx,
                &path,
                quill_config(spec),
            ));
            let index = index.expect("create Quill cold-open fixture");
            let _ = index_batches(context, &index, &corpus, count, None);
            let _ = commit(context, &index);
            drop(index);
            let eviction = evict_index_file_cache(&path);
            record_cold_cache_eviction(spec, arm, eviction);
            fresh_process_open(&path, spec, arm)
        }
        EngineArm::Tantivy => {
            let path = scratch_path("qg9-tantivy");
            let index = tantivy_create(&path, spec);
            let _ = index_batches(context, &index, &corpus, count, None);
            let _ = commit(context, &index);
            drop(index);
            let eviction = evict_index_file_cache(&path);
            record_cold_cache_eviction(spec, arm, eviction);
            fresh_process_open(&path, spec, arm)
        }
    };
    elapsed.as_secs_f64() * 1_000.0
}

fn cargo_tree_line_is_tantivy_family(line: &str) -> bool {
    let mut fields = line.split_whitespace();
    let Some(mut package) = fields.next() else {
        return false;
    };
    for version in fields {
        if (package == "tantivy" || package.starts_with("tantivy-")) && version.starts_with('v') {
            return true;
        }
        package = version;
    }
    false
}

fn dependency_surface_metric() -> f64 {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args([
            "tree",
            "--locked",
            "-p",
            "frankensearch",
            "--features",
            "lexical",
        ])
        .output()
        .expect("run QG-10 cargo tree");
    assert!(output.status.success(), "QG-10 cargo tree failed");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| cargo_tree_line_is_tantivy_family(line))
        .count() as f64
}

fn measure_metric(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
) -> MetricMeasurement {
    measure_metric_with_query(context, spec, arm, None)
}

fn measure_metric_with_query(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    query_override: Option<&str>,
) -> MetricMeasurement {
    measure_metric_with_query_and_qg1_writer_mode(context, spec, arm, query_override, None)
}

fn measure_metric_with_query_and_qg1_writer_mode(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    query_override: Option<&str>,
    qg1_tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
) -> MetricMeasurement {
    match spec.gate {
        PerfGate::Qg1 if spec.metric == "tokenize_docs_per_second" => {
            MetricMeasurement::gauge(tokenize_metric(context, spec))
        }
        PerfGate::Qg1 => {
            bulk_metric_with_qg1_writer_mode(context, spec, arm, qg1_tantivy_writer_mode)
        }
        PerfGate::Qg2 if is_qg2_continuous_update_cell(spec) => qg2_bulk_metric_continuous(
            context,
            spec,
            arm,
            context
                .scale
                .document_count(spec.document_count.expect("QG-2 update document count")),
        ),
        // Any other QG-2 cell keeps its prior summed-call behaviour rather than
        // being retyped by association.
        PerfGate::Qg2 => bulk_metric(context, spec, arm),
        PerfGate::Qg8 => bulk_metric(context, spec, arm),
        PerfGate::Qg3 if spec.metric == "docs_per_second" => bulk_metric(context, spec, arm),
        PerfGate::Qg3 => MetricMeasurement::gauge(watch_metric(context, spec, arm)),
        PerfGate::Qg4 => MetricMeasurement::gauge(commit_metric(context, spec, arm)),
        PerfGate::Qg5 => MetricMeasurement::gauge(compaction_metric(context, spec, arm)),
        PerfGate::Qg6 => MetricMeasurement::gauge(query_metric(context, spec, arm, query_override)),
        PerfGate::Qg7 => MetricMeasurement::gauge(memory_metric(context, spec, arm)),
        PerfGate::Qg9 => MetricMeasurement::gauge(cold_open_metric(context, spec, arm)),
        PerfGate::Qg10 => MetricMeasurement::gauge(dependency_surface_metric()),
    }
}

fn unit(spec: &PerfCellSpec) -> &'static str {
    match spec.metric.as_str() {
        "docs_per_second" | "tokenize_docs_per_second" | "updates_per_second" => "docs/s",
        "commit_latency_ms"
        | "latency_ms"
        | "open_latency_ms"
        | "update_to_searchable_ms"
        | "wall_clock_ms" => "ms",
        "peak_rss_bytes" => "bytes",
        "index_bytes_per_document" => "bytes/doc",
        "tantivy_nodes" => "nodes",
        _ => "ratio",
    }
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    numerator / denominator.max(f64::MIN_POSITIVE)
}

/// Evidence-layer measurement context shared by every cell in one run.
struct EvidenceContext {
    config: PairedEstimatorConfig,
    policy: EvidencePolicy,
    sample_provenance: PerfSampleProvenance,
}

/// Whether this cell is a QG-2 update cell measured as ONE continuous
/// first-feed-through-searchable-and-quiescent interval.
///
/// Routing, semantic typing, and the work denominator all key on this single
/// predicate so they cannot drift apart: a cell that is measured continuously
/// but typed as a gauge would publish a rate no clock ever checks again, and a
/// cell typed as throughput without a continuous interval fails closed in
/// `qg1_sample_window`. It is deliberately narrow — only the update cells that
/// actually route through `qg2_bulk_metric_continuous` qualify, so no other
/// QG-2 cell and no other gate is retyped.
fn is_qg2_continuous_update_cell(spec: &PerfCellSpec) -> bool {
    spec.gate == PerfGate::Qg2 && spec.metric == "docs_per_second" && spec.document_count.is_some()
}

fn metric_semantics(spec: &PerfCellSpec) -> PerfMetricSemantics {
    // A QG-1 engine-indexing cell and a QG-2 update cell each derive their rate
    // from one continuous first-feed-through-searchable-and-quiescent interval
    // over an exact document count, so both are native throughput operations
    // and the estimator recomputes them from the sample itself. Every other rate
    // in this matrix — the QG-1 tokenizer diagnostic, QG-3/QG-8 bulk indexing,
    // QG-3 updates — is still a sum of independently timed calls that excludes
    // the gaps between them. Typing one of those as Throughput would silently
    // redefine it as work over the outer sample window, a different and
    // unmeasured quantity, so they stay gauges.
    if qg1_producer_coverage(spec) == Some(Qg1ProducerCoverage::EngineIndexingLifecycle)
        || is_qg2_continuous_update_cell(spec)
    {
        return PerfMetricSemantics::Throughput;
    }
    match spec.metric.as_str() {
        "docs_per_second" | "tokenize_docs_per_second" | "updates_per_second" => {
            PerfMetricSemantics::GaugeHigherIsBetter
        }
        _ => PerfMetricSemantics::GaugeLowerIsBetter,
    }
}

fn operation_scope(spec: &PerfCellSpec) -> PerfOperationScope {
    PerfOperationScope {
        operation_id: format!("{}.{}.{}", spec.gate, spec.fixture, spec.metric),
        version: 1,
        semantics: metric_semantics(spec),
        unit: unit(spec).to_owned(),
    }
}

fn raw_sample_work(context: &BenchContext, spec: &PerfCellSpec) -> (Option<u64>, Option<u64>) {
    // A QG-2 continuous update cell publishes a real work denominator: the
    // exact document count its one interval processed. It has no prepared-input
    // content-byte binding, so bytes stay absent rather than being invented.
    if is_qg2_continuous_update_cell(spec) {
        let document_count = context
            .scale
            .document_count(spec.document_count.expect("QG-2 update document count"));
        assert!(
            document_count > 0,
            "QG-2 throughput sample requires a positive document denominator"
        );
        return (Some(document_count), None);
    }
    if qg1_producer_coverage(spec).is_none() {
        return (None, None);
    }
    let document_count = context
        .scale
        .document_count(spec.document_count.expect("QG-1 throughput document count"));
    let prepared_input = context.qg1_sample_input(document_count);
    let content_bytes = prepared_input.binding.content_bytes;
    assert!(
        content_bytes > 0,
        "QG-1 throughput sample requires positive immutable content bytes"
    );
    (
        Some(prepared_input.binding.document_count),
        Some(content_bytes),
    )
}

/// Freeze the exact QG-1 cell plan before warmup or measurement. The returned
/// authority is retained by the live estimator rather than synthesized from
/// any raw row.
fn qg1_issued_streams(runs: usize, cell_seed: u64) -> Vec<(String, u64, u64, Vec<PerfSampleArm>)> {
    let issue = |stream_role: &str, seed: u64, block_id_base: u64, sample_id_base: u64| {
        (
            stream_role.to_owned(),
            block_id_base,
            sample_id_base,
            seeded_balanced_pair_order(runs, seed).expect("QG-1 issued randomized arm order"),
        )
    };
    vec![
        issue("qg1.effect.tantivy_vs_quill.v1", cell_seed, 0, 0),
        issue("qg1.null.tantivy.v1", cell_seed ^ 0xaa, 0, 1_000_000),
        issue("qg1.null.quill.v1", cell_seed ^ 0x55, 2_000_000, 2_000_000),
    ]
}

fn qg1_pilot_issued_streams(
    runs: usize,
    candidate_seed: u64,
) -> Vec<(String, u64, u64, Vec<PerfSampleArm>)> {
    vec![
        (
            QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT.to_owned(),
            0,
            0,
            seeded_balanced_pair_order(runs, candidate_seed)
                .expect("QG-1 pilot effect issued order"),
        ),
        (
            QG1_STREAM_ROLE_TANTIVY_PILOT_NULL.to_owned(),
            2_000_000,
            2_000_000,
            seeded_balanced_pair_order(runs, candidate_seed ^ 0xaa)
                .expect("QG-1 pilot null issued order"),
        ),
    ]
}

struct Qg1IncumbentAuthorityProducer {
    estimator_config: PairedEstimatorConfig,
    producer: Qg1LifecycleProducer,
}

struct Qg1IncumbentPilotProducer {
    writer_mode: Qg1TantivyWriterMode,
    authority: Qg1IncumbentAuthorityProducer,
}

struct Qg1IncumbentStartup {
    screen_plan: Qg1TantivyIncumbentScreenPlan,
    pilots: Vec<Qg1IncumbentPilotProducer>,
}

/// The child freezes every engine-cell producer before it performs any
/// preflight, warmup, or timed work.  Keeping the estimator configuration with
/// the producer prevents `collect_cell` from minting a second authority after
/// the parent has acknowledged this startup transcript.
struct Qg1StartupProducer {
    operation_id: String,
    estimator_config: PairedEstimatorConfig,
    producer: Qg1LifecycleProducer,
    incumbent: Option<Qg1IncumbentStartup>,
}

struct Qg1StartupProducers {
    engine_cells: Vec<Qg1StartupProducer>,
}

impl Qg1StartupProducers {
    fn for_spec(&self, spec: &PerfCellSpec) -> Option<&Qg1StartupProducer> {
        if qg1_producer_coverage(spec) != Some(Qg1ProducerCoverage::EngineIndexingLifecycle) {
            return None;
        }
        let operation_id = operation_scope(spec).operation_id;
        self.engine_cells
            .iter()
            .find(|producer| producer.operation_id == operation_id)
    }

    fn retained_authorities(&self) -> Vec<&Qg1ExpectedAuthority> {
        self.engine_cells
            .iter()
            .flat_map(|startup| {
                std::iter::once(startup.producer.expected_authority()).chain(
                    startup.incumbent.iter().flat_map(|incumbent| {
                        incumbent
                            .pilots
                            .iter()
                            .map(|pilot| pilot.authority.producer.expected_authority())
                    }),
                )
            })
            .collect()
    }
}

fn construct_qg1_startup_producers(
    context: &BenchContext,
    selected: &[PerfCellSpec],
    runs: usize,
    evidence: &EvidenceContext,
    machine_profile: MachineProfileKey,
    external_cpu_budget: usize,
    preregistered_widths: &[usize],
) -> Qg1StartupProducers {
    let mut engine_cells = Vec::new();
    for spec in selected {
        if qg1_producer_coverage(spec) != Some(Qg1ProducerCoverage::EngineIndexingLifecycle) {
            continue;
        }
        let scope = operation_scope(spec);
        let cell_seed = evidence.config.bootstrap_seed ^ fixture_seed(&spec.fixture);
        let mut estimator_config = evidence.config.clone();
        let producer = install_qg1_lifecycle_authority(
            &mut estimator_config,
            context,
            spec,
            &scope,
            &evidence.sample_provenance,
            u64::try_from(runs).expect("QG-1 authority pair count fits u64"),
            qg1_issued_streams(runs, cell_seed),
        )
        .expect("selected engine QG-1 cell must mint one startup authority producer");
        let screen_this_cell = context.scale.is_full();
        let incumbent = screen_this_cell.then(|| {
            let prepared = context.qg1_sample_input(
                spec.document_count
                    .expect("QG-1 incumbent requires a document count"),
            );
            let writer_heap_bytes = spec.writer_heap_bytes.unwrap_or(50_000_000);
            let mut widths = preregistered_widths
                .iter()
                .copied()
                .filter(|width| {
                    frankensearch_quill_gauntlet::PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES
                        .saturating_mul(*width)
                        <= writer_heap_bytes
                })
                .collect::<Vec<_>>();
            widths.sort_unstable();
            widths.dedup();
            assert!(!widths.is_empty(), "QG-1 incumbent width universe is empty");
            let screen_plan = Qg1TantivyIncumbentScreenPlan::new(
                machine_profile,
                external_cpu_budget,
                widths.clone(),
                spec,
                prepared.binding.content_bytes,
            )
            .expect("freeze live QG-1 Tantivy incumbent screen plan");
            let writer_modes = std::iter::once(Qg1TantivyWriterMode::ShippingAuto)
                .chain(
                    widths
                        .into_iter()
                        .map(|writer_threads| Qg1TantivyWriterMode::Fixed { writer_threads }),
                )
                .collect::<Vec<_>>();
            let pilots = writer_modes
                .into_iter()
                .enumerate()
                .map(|(candidate_index, writer_mode)| {
                    let mut estimator_config = evidence.config.clone();
                    let candidate_seed = cell_seed
                        ^ u64::try_from(candidate_index)
                            .expect("QG-1 candidate index fits u64")
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let producer = install_qg1_lifecycle_authority(
                        &mut estimator_config,
                        context,
                        spec,
                        &scope,
                        &evidence.sample_provenance,
                        u64::try_from(runs).expect("QG-1 pilot pair count fits u64"),
                        qg1_pilot_issued_streams(runs, candidate_seed),
                    )
                    .expect("freeze one distinct QG-1 candidate-pilot authority");
                    Qg1IncumbentPilotProducer {
                        writer_mode,
                        authority: Qg1IncumbentAuthorityProducer {
                            estimator_config,
                            producer,
                        },
                    }
                })
                .collect::<Vec<_>>();
            Qg1IncumbentStartup {
                screen_plan,
                pilots,
            }
        });
        engine_cells.push(Qg1StartupProducer {
            operation_id: scope.operation_id,
            estimator_config,
            producer,
            incumbent,
        });
    }
    let expected_count = selected
        .iter()
        .filter(|spec| {
            qg1_producer_coverage(spec) == Some(Qg1ProducerCoverage::EngineIndexingLifecycle)
        })
        .count();
    assert_eq!(
        engine_cells.len(),
        expected_count,
        "every selected engine QG-1 cell must contribute one fresh-decision startup producer"
    );
    Qg1StartupProducers { engine_cells }
}

fn qg1_wait_for_authority_ack(
    receiver: mpsc::Receiver<Result<Vec<u8>, String>>,
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    receiver
        .recv_timeout(timeout)
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => {
                "QG-1 authority ACK was not received before the bounded deadline".to_owned()
            }
            mpsc::RecvTimeoutError::Disconnected => {
                "QG-1 authority ACK stream closed before acknowledgement".to_owned()
            }
        })?
}

/// Publish every frozen engine-cell authority before preflight, warmup, or
/// timing, then wait for the parent's single final acknowledgement. The child
/// has no path through which it can self-publish or self-load this authority.
fn require_qg1_pre_timing_authority_ack(selected_qg1: bool, producers: &[Qg1StartupProducer]) {
    if !selected_qg1 {
        return;
    }
    let mode = match std::env::var(Qg1StartupHandshakeV1::ENV) {
        Ok(mode) => mode,
        Err(std::env::VarError::NotPresent) => return,
        Err(error) => panic!(
            "{} is not valid Unicode: {error}",
            Qg1StartupHandshakeV1::ENV
        ),
    };
    assert_eq!(
        mode,
        Qg1StartupHandshakeV1::MODE,
        "typed QG-1 producer requires the exact stdio authority handshake mode"
    );

    let registered_producers = producers
        .iter()
        .flat_map(|startup| {
            startup
                .incumbent
                .iter()
                .flat_map(|incumbent| {
                    incumbent
                        .pilots
                        .iter()
                        .map(|pilot| &pilot.authority.producer)
                })
                .chain(std::iter::once(&startup.producer))
        })
        .collect::<Vec<_>>();
    let mut stdout = std::io::stdout().lock();
    for (offset, producer) in registered_producers.iter().enumerate() {
        let entry = producer.register_entry();
        entry
            .verify()
            .expect("producer must emit one complete verified QG-1 authority register entry");
        let entry_bytes = entry
            .to_json_bytes()
            .expect("producer authority register entry must serialize");
        let sequence = u64::try_from(offset)
            .expect("QG-1 startup register ordinal fits u64")
            .checked_add(1)
            .expect("QG-1 startup register sequence cannot overflow");
        let frame = Qg1StartupHandshakeV1::register_frame(sequence, &entry_bytes)
            .expect("producer authority register entry fits the bounded stdio frame");
        stdout
            .write_all(&frame)
            .expect("write QG-1 authority register frame to parent");
    }
    let register_count =
        u64::try_from(registered_producers.len()).expect("QG-1 startup register count fits u64");
    stdout
        .write_all(&Qg1StartupHandshakeV1::complete_frame(register_count))
        .expect("write QG-1 authority COMPLETE frame to parent");
    stdout
        .flush()
        .expect("flush QG-1 authority startup transcript to parent");
    drop(stdout);

    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut frame = vec![0_u8; Qg1StartupHandshakeV1::final_ack_len()];
        let result = std::io::stdin()
            .read_exact(&mut frame)
            .map(|()| frame)
            .map_err(|error| format!("QG-1 authority final ACK read failed: {error}"));
        let _ = sender.send(result);
    });
    let acknowledgement =
        qg1_wait_for_authority_ack(receiver, Qg1StartupHandshakeV1::STARTUP_TIMEOUT)
            .unwrap_or_else(|error| panic!("{error}"));
    Qg1StartupHandshakeV1::validate_final_ack(&acknowledgement)
        .unwrap_or_else(|error| panic!("{error}"));
}

/// Recompute the exact QG-1 runner claims that the normal harness entry point
/// will validate. The harness=false discriminator never hand-authors a plan.
fn qg1_live_startup_discriminator_claims(gate: PerfGate) -> Result<RunnerPlanClaims, String> {
    let matrix = PerfMatrixSpec::complete();
    let registry = MachineClassRegistry::frozen().map_err(|error| error.to_string())?;
    let profile = MachineProfileKey::new(
        HardwareClassId::TrjZen35995wx,
        ExecutionProfileId::Physical64,
    )
    .map_err(|error| error.to_string())?;
    let plan = matrix
        .applicability_plan(&registry, profile, gate)
        .map_err(|error| error.to_string())?;
    let execution_capacity = plan.execution_capacity.ok_or_else(|| {
        "the live QG-1 discriminator profile has no execution capacity".to_owned()
    })?;
    let max_exercised_cell_width = plan.max_exercised_cell_width.ok_or_else(|| {
        "the live QG-1 discriminator profile has no maximum cell width".to_owned()
    })?;
    let binding = plan.binding();
    Ok(RunnerPlanClaims {
        gate,
        hardware_class: HardwareClassId::TrjZen35995wx,
        execution_profile: ExecutionProfileId::Physical64,
        execution_capacity,
        max_exercised_cell_width,
        rayon_num_threads: execution_capacity,
        applicability_plan_schema_version: binding.schema_version.clone(),
        applicability_plan_sha256: binding.applicability_plan_sha256.clone(),
        gate_matrix_contract_sha256: binding.gate_matrix_contract_sha256.clone(),
        profile_contract_sha256: binding.profile_contract_sha256.clone(),
        registry_schema_version: binding.registry_schema_version.clone(),
        registry_sha256: binding.registry_sha256.clone(),
    })
}

/// Build the exact typed environment consumed by the normal benchmark entry
/// path from independently recomputed claims.
fn qg1_live_startup_discriminator_environment(
    gate: PerfGate,
) -> Result<Vec<(&'static str, String)>, String> {
    let claims = qg1_live_startup_discriminator_claims(gate)?;
    Ok(vec![
        ("QUILL_PERF_GATE", gate.label().to_owned()),
        ("QUILL_PERF_SCALE", "smoke".to_owned()),
        ("QUILL_PERF_RUNS", PERF_MIN_RUNS.to_string()),
        (
            "QUILL_PERF_HARDWARE_CLASS",
            claims.hardware_class.as_str().to_owned(),
        ),
        (
            "QUILL_PERF_EXECUTION_PROFILE",
            claims.execution_profile.as_str().to_owned(),
        ),
        (
            "QUILL_PERF_EXECUTION_CAPACITY",
            claims.execution_capacity.to_string(),
        ),
        (
            "QUILL_PERF_MAX_EXERCISED_CELL_WIDTH",
            claims.max_exercised_cell_width.to_string(),
        ),
        (
            "QUILL_PERF_APPLICABILITY_PLAN_SCHEMA_VERSION",
            claims.applicability_plan_schema_version,
        ),
        (
            "QUILL_PERF_APPLICABILITY_PLAN_SHA256",
            claims.applicability_plan_sha256,
        ),
        (
            "QUILL_PERF_GATE_MATRIX_CONTRACT_SHA256",
            claims.gate_matrix_contract_sha256,
        ),
        (
            "QUILL_PERF_PROFILE_CONTRACT_SHA256",
            claims.profile_contract_sha256,
        ),
        (
            "QUILL_PERF_REGISTRY_SCHEMA_VERSION",
            claims.registry_schema_version,
        ),
        ("QUILL_PERF_REGISTRY_SHA256", claims.registry_sha256),
        ("RAYON_NUM_THREADS", claims.rayon_num_threads.to_string()),
    ])
}

/// Independently recompute the engine-lifecycle operation map from the same
/// frozen selection the normal child will receive. The parser below must match
/// this exact map, not merely observe one nonempty register frame.
fn qg1_live_startup_discriminator_selected_operation_ids() -> Result<BTreeSet<String>, String> {
    let matrix = PerfMatrixSpec::complete();
    let claims = qg1_live_startup_discriminator_claims(PerfGate::Qg1)?;
    let runner = RunnerApplicabilityContext::reconstruct(&matrix, &claims)?;
    let selected = selected_cells(&matrix, &runner, MatrixScale::Smoke, None)?;
    let operation_ids = selected
        .iter()
        .filter(|cell| {
            qg1_producer_coverage(&cell.spec) == Some(Qg1ProducerCoverage::EngineIndexingLifecycle)
        })
        .map(|cell| operation_scope(&cell.spec).operation_id)
        .collect::<BTreeSet<_>>();
    let expected = selected
        .iter()
        .filter(|cell| {
            qg1_producer_coverage(&cell.spec) == Some(Qg1ProducerCoverage::EngineIndexingLifecycle)
        })
        .count();
    if operation_ids.len() != expected {
        return Err(
            "the independently recomputed QG-1 selected engine operation map is not unique"
                .to_owned(),
        );
    }
    Ok(operation_ids)
}

fn qg1_live_startup_discriminator_child(
    mode: Qg1LiveStartupDiscriminatorMode,
) -> Result<std::process::Child, String> {
    let child_mode = match mode {
        Qg1LiveStartupDiscriminatorMode::Child => "child",
        Qg1LiveStartupDiscriminatorMode::Preamble => "preamble",
        Qg1LiveStartupDiscriminatorMode::NonQg => "non-qg",
        Qg1LiveStartupDiscriminatorMode::Parent => {
            return Err("the parent discriminator cannot spawn itself as a child".to_owned());
        }
    };
    let mut command = Command::new(std::env::current_exe().map_err(|error| error.to_string())?);
    command
        .env(QG1_LIVE_STARTUP_DISCRIMINATOR_ENV, child_mode)
        .env(Qg1StartupHandshakeV1::ENV, Qg1StartupHandshakeV1::MODE)
        .env("QUILL_PERF_BUILD_PROFILE", "live-startup-discriminator")
        .env_remove("QUILL_PERF_FIXTURE")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let gate = if mode == Qg1LiveStartupDiscriminatorMode::NonQg {
        PerfGate::Qg2
    } else {
        PerfGate::Qg1
    };
    for (name, value) in qg1_live_startup_discriminator_environment(gate)? {
        command.env(name, value);
    }
    command
        .spawn()
        .map_err(|error| format!("spawn harness=false QG-1 startup child: {error}"))
}

fn qg1_live_startup_discriminator_abort(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn qg1_live_startup_operation_id(entry_bytes: &[u8]) -> Result<(String, String), String> {
    let entry = Qg1AuthorityRegisterEntryV1::from_verified_slice(entry_bytes)
        .map_err(|error| format!("live QG-1 register entry verification failed: {error}"))?;
    let canonical = entry
        .to_json_bytes()
        .map_err(|error| format!("live QG-1 register entry did not re-encode: {error}"))?;
    if canonical != entry_bytes {
        return Err(
            "live QG-1 register entry was not the exact canonical producer bytes".to_owned(),
        );
    }
    let value = serde_json::from_slice::<serde_json::Value>(&canonical)
        .map_err(|error| format!("live QG-1 verified entry cannot be inspected: {error}"))?;
    let operation_id = value
        .pointer("/authority/scope/operation_id")
        .and_then(serde_json::Value::as_str)
        .filter(|operation_id| !operation_id.is_empty())
        .ok_or_else(|| "live QG-1 verified entry has no authority scope operation ID".to_owned())?
        .to_owned();
    Ok((operation_id, entry.digest().to_owned()))
}

fn qg1_live_startup_discriminator_positive_child() -> Result<(), String> {
    let expected_operation_ids = qg1_live_startup_discriminator_selected_operation_ids()?;
    let mut child = qg1_live_startup_discriminator_child(Qg1LiveStartupDiscriminatorMode::Child)?;
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            qg1_live_startup_discriminator_abort(&mut child);
            return Err("live QG-1 startup child lacks stdout".to_owned());
        }
    };
    let transcript = (|| -> Result<(), String> {
        let mut next_sequence = 1_u64;
        let mut received = BTreeMap::<String, String>::new();
        loop {
            match Qg1StartupHandshakeV1::read_control_frame(&mut stdout)? {
                frankensearch_quill_gauntlet::Qg1StartupControlFrameV1::Register {
                    sequence,
                    entry_bytes,
                } => {
                    if sequence != next_sequence {
                        return Err(format!(
                            "live QG-1 startup REGISTER sequence {sequence} is not contiguous; expected {next_sequence}"
                        ));
                    }
                    next_sequence = next_sequence.checked_add(1).ok_or_else(|| {
                        "live QG-1 startup REGISTER sequence overflowed".to_owned()
                    })?;
                    let (operation_id, digest) = qg1_live_startup_operation_id(&entry_bytes)?;
                    if received.insert(operation_id.clone(), digest).is_some() {
                        return Err(format!(
                            "live QG-1 startup transcript repeats selected operation {operation_id:?}"
                        ));
                    }
                }
                frankensearch_quill_gauntlet::Qg1StartupControlFrameV1::Complete {
                    register_count,
                } => {
                    let received_count = u64::try_from(received.len()).map_err(|_| {
                        "live QG-1 startup received map count does not fit u64".to_owned()
                    })?;
                    let expected_count =
                        u64::try_from(expected_operation_ids.len()).map_err(|_| {
                            "live QG-1 startup expected map count does not fit u64".to_owned()
                        })?;
                    if register_count != received_count || register_count != expected_count {
                        return Err(format!(
                            "live QG-1 COMPLETE({register_count}) does not match received map {received_count} or expected selected map {expected_count}"
                        ));
                    }
                    let received_operation_ids = received.into_keys().collect::<BTreeSet<_>>();
                    if received_operation_ids != expected_operation_ids {
                        return Err(format!(
                            "live QG-1 startup operation map differs from independently selected cells; expected {expected_operation_ids:?}, received {received_operation_ids:?}"
                        ));
                    }
                    return Ok(());
                }
            }
        }
    })();
    if let Err(error) = transcript {
        qg1_live_startup_discriminator_abort(&mut child);
        return Err(error);
    }

    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut ordinary = [0_u8; 1];
        let result = stdout
            .read(&mut ordinary)
            .map(|count| (ordinary[..count].to_vec(), stdout))
            .map_err(|error| error.to_string());
        let _ = sender.send(result);
    });
    if !matches!(
        receiver.recv_timeout(Duration::from_millis(200)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ) {
        qg1_live_startup_discriminator_abort(&mut child);
        return Err(
            "live QG-1 startup child emitted an ordinary stdout byte before final ACK".to_owned(),
        );
    }

    let mut stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            qg1_live_startup_discriminator_abort(&mut child);
            return Err("live QG-1 startup child lacks stdin".to_owned());
        }
    };
    stdin
        .write_all(&Qg1StartupHandshakeV1::final_ack_frame())
        .map_err(|error| format!("write live QG-1 final ACK: {error}"))?;
    stdin
        .flush()
        .map_err(|error| format!("flush live QG-1 final ACK: {error}"))?;
    drop(stdin);

    let (mut ordinary, mut stdout) = match receiver.recv_timeout(Duration::from_secs(5)) {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            qg1_live_startup_discriminator_abort(&mut child);
            return Err(error);
        }
        Err(error) => {
            qg1_live_startup_discriminator_abort(&mut child);
            return Err(format!(
                "live QG-1 child did not emit after final ACK: {error}"
            ));
        }
    };
    if let Err(error) = stdout.read_to_end(&mut ordinary) {
        qg1_live_startup_discriminator_abort(&mut child);
        return Err(format!("read live QG-1 child ordinary stdout: {error}"));
    }
    let status = child
        .wait()
        .map_err(|error| format!("reap live QG-1 startup child: {error}"))?;
    if !status.success() {
        return Err(format!(
            "live QG-1 startup child failed after final ACK: {status}"
        ));
    }
    if !ordinary.starts_with(b"bench_elf_sha256=") {
        return Err(format!(
            "the first ordinary byte after final ACK is not the canonical benchmark identity: {ordinary:?}"
        ));
    }
    if !ordinary
        .windows(QG1_LIVE_STARTUP_ORDINARY_MARKER.len())
        .any(|window| window == QG1_LIVE_STARTUP_ORDINARY_MARKER)
    {
        return Err(
            "live QG-1 startup child did not reach its post-ACK ordinary marker".to_owned(),
        );
    }
    Ok(())
}

fn qg1_live_startup_discriminator_rejects_preamble() -> Result<(), String> {
    let mut child =
        qg1_live_startup_discriminator_child(Qg1LiveStartupDiscriminatorMode::Preamble)?;
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            qg1_live_startup_discriminator_abort(&mut child);
            return Err("preamble QG-1 startup child lacks stdout".to_owned());
        }
    };
    if Qg1StartupHandshakeV1::read_control_frame(&mut stdout).is_ok() {
        qg1_live_startup_discriminator_abort(&mut child);
        return Err("planted ordinary QG-1 preamble was accepted as a control frame".to_owned());
    }
    child
        .kill()
        .map_err(|error| format!("kill planted-preamble QG-1 startup child: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("reap planted-preamble QG-1 startup child: {error}"))?;
    if status.success() {
        return Err(
            "planted-preamble QG-1 startup child unexpectedly exited successfully".to_owned(),
        );
    }
    Ok(())
}

fn qg1_live_startup_discriminator_non_qg_child() -> Result<(), String> {
    let mut child = qg1_live_startup_discriminator_child(Qg1LiveStartupDiscriminatorMode::NonQg)?;
    let mut stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            qg1_live_startup_discriminator_abort(&mut child);
            return Err("non-QG startup child lacks stdout".to_owned());
        }
    };
    let mut ordinary = Vec::new();
    stdout
        .read_to_end(&mut ordinary)
        .map_err(|error| format!("read non-QG startup child stdout: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("reap non-QG startup child: {error}"))?;
    if !status.success() {
        return Err(format!("non-QG startup child failed: {status}"));
    }
    if !ordinary.starts_with(b"bench_elf_sha256=") {
        return Err(format!(
            "non-QG child did not preserve the ordinary identity as stdout line one: {ordinary:?}"
        ));
    }
    if ordinary
        .windows(Qg1StartupHandshakeV1::REGISTER_MAGIC.len())
        .any(|window| window == Qg1StartupHandshakeV1::REGISTER_MAGIC)
        || ordinary
            .windows(Qg1StartupHandshakeV1::COMPLETE_MAGIC.len())
            .any(|window| window == Qg1StartupHandshakeV1::COMPLETE_MAGIC)
    {
        return Err(
            "non-QG child emitted QG-1 startup control bytes despite its non-QG selected gate"
                .to_owned(),
        );
    }
    Ok(())
}

fn run_qg1_live_startup_discriminator_parent() -> Result<(), String> {
    qg1_live_startup_discriminator_positive_child()?;
    qg1_live_startup_discriminator_rejects_preamble()?;
    qg1_live_startup_discriminator_non_qg_child()?;
    Ok(())
}

fn install_qg1_lifecycle_authority(
    estimator_config: &mut PairedEstimatorConfig,
    context: &BenchContext,
    spec: &PerfCellSpec,
    scope: &PerfOperationScope,
    provenance: &PerfSampleProvenance,
    expected_pair_count: u64,
    issued_streams: Vec<(String, u64, u64, Vec<PerfSampleArm>)>,
) -> Option<Qg1LifecycleProducer> {
    if qg1_producer_coverage(spec) != Some(Qg1ProducerCoverage::EngineIndexingLifecycle) {
        return None;
    }
    let document_count = context
        .scale
        .document_count(spec.document_count.expect("QG-1 throughput document count"));
    let prepared = context.qg1_sample_input(document_count);
    let mut next_document = 0_u64;
    let batch_coverage = prepared
        .batches
        .iter()
        .map(|batch| {
            let document_count =
                u64::try_from(batch.len()).expect("QG-1 prepared batch length fits u64");
            let coverage = Qg1BatchCoverage {
                document_start: next_document,
                document_count,
            };
            next_document = next_document
                .checked_add(document_count)
                .expect("QG-1 prepared batch coverage fits u64");
            coverage
        })
        .collect::<Vec<_>>();
    assert_eq!(
        next_document, prepared.binding.document_count,
        "QG-1 lifecycle authority must cover the complete prepared input"
    );
    Some(
        estimator_config
            .install_qg1_lifecycle_authority(
                scope.clone(),
                provenance.corpus_sha256.clone(),
                prepared.binding.manifest_sha256.clone(),
                prepared.binding.indexed_content_sha256.clone(),
                prepared.binding.document_count,
                prepared.binding.content_bytes,
                prepared.binding.batch_count,
                batch_coverage,
                prepared.binding.tail_document_id.clone(),
                expected_pair_count,
                issued_streams,
            )
            .expect("freeze one complete pre-timing QG-1 lifecycle authority"),
    )
}

/// Resolve a raw QG-1 denominator from the prepared input the continuous
/// measurement actually consumed.  A separately recomputed work/byte pair is
/// allowed only when it is exactly equal; otherwise the sample is rejected
/// before the estimator sees it.
fn qg1_raw_sample_denominator(
    declared: (Option<u64>, Option<u64>),
    continuous: Option<&Qg1ContinuousMeasurement>,
) -> Result<(Option<u64>, Option<u64>), String> {
    let Some(continuous) = continuous else {
        return Ok(declared);
    };
    continuous.prepared_input.validate()?;
    continuous.lifecycle_receipt.validate()?;
    if continuous.lifecycle_receipt.prepared_input != continuous.prepared_input
        || continuous.lifecycle_receipt.document_count != continuous.work_units
    {
        return Err(
            "QG-1 raw denominator is not bound to the lifecycle receipt produced by this sample"
                .to_owned(),
        );
    }
    let actual = (
        Some(continuous.prepared_input.document_count),
        Some(continuous.prepared_input.content_bytes),
    );
    if declared != actual || continuous.work_units != continuous.prepared_input.document_count {
        return Err(format!(
            "QG-1 raw denominator {declared:?} is not the prepared input actually measured \
             ({actual:?}; continuous work={})",
            continuous.work_units
        ));
    }
    Ok(actual)
}

/// One continuous engine interval, resolved onto the stream's own clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qg1IntervalOffsets {
    work_units: u64,
    started_ns: u64,
    elapsed_ns: u64,
}

/// The timing window one raw sample publishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qg1SampleWindow {
    started_ns: u64,
    ended_ns: u64,
}

/// Decide the `started_ns`/`ended_ns` a [`PerfRawSample`] may publish, failing
/// closed on every way a rate can be attached to time it was not measured over.
///
/// `Throughput` is not a label the harness may apply at will. The estimator
/// recomputes such a sample as `work_units * 1e9 / (ended_ns - started_ns)`, so
/// the published window has to be the continuous engine interval itself and the
/// work has to be the work that interval processed. The converse matters just as
/// much: a continuous interval filed under gauge semantics publishes a rate no
/// clock will ever check again, which is how an unmeasured number survives into
/// an artifact.
fn qg1_sample_window(
    semantics: PerfMetricSemantics,
    declared_work_units: Option<u64>,
    call_started_ns: u64,
    call_ended_ns: u64,
    continuous: Option<Qg1IntervalOffsets>,
) -> Result<Qg1SampleWindow, String> {
    match (semantics, continuous) {
        (PerfMetricSemantics::Throughput, Some(interval)) => {
            if interval.elapsed_ns == 0 {
                return Err("continuous engine interval spans no monotonic time".to_owned());
            }
            if declared_work_units != Some(interval.work_units) {
                return Err(format!(
                    "throughput sample declares work_units {declared_work_units:?} but its \
                     continuous interval processed {}",
                    interval.work_units
                ));
            }
            let ended_ns = interval
                .started_ns
                .checked_add(interval.elapsed_ns)
                .ok_or_else(|| {
                    "continuous engine interval overflows the stream clock".to_owned()
                })?;
            if interval.started_ns < call_started_ns || ended_ns > call_ended_ns {
                return Err(format!(
                    "continuous engine interval [{}, {ended_ns}] escapes the call it was measured \
                     in [{call_started_ns}, {call_ended_ns}]",
                    interval.started_ns
                ));
            }
            Ok(Qg1SampleWindow {
                started_ns: interval.started_ns,
                ended_ns,
            })
        }
        (PerfMetricSemantics::Throughput, None) => Err("throughput samples require one \
                                                        continuous engine interval; a summed \
                                                        per-call duration is not work over \
                                                        elapsed time"
            .to_owned()),
        (
            PerfMetricSemantics::Duration
            | PerfMetricSemantics::GaugeHigherIsBetter
            | PerfMetricSemantics::GaugeLowerIsBetter,
            Some(_),
        ) => Err(format!(
            "a continuous engine interval must be published as Throughput, not as {semantics:?}, \
             which stores the rate without ever recomputing it from that interval"
        )),
        (
            PerfMetricSemantics::Duration
            | PerfMetricSemantics::GaugeHigherIsBetter
            | PerfMetricSemantics::GaugeLowerIsBetter,
            None,
        ) => {
            let ended_ns = if call_ended_ns <= call_started_ns {
                call_started_ns.saturating_add(1)
            } else {
                call_ended_ns
            };
            Ok(Qg1SampleWindow {
                started_ns: call_started_ns,
                ended_ns,
            })
        }
    }
}

fn fixture_seed(fixture: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in fixture.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct StreamPlan<'a> {
    control: EngineArm,
    treatment: EngineArm,
    control_qg1_tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
    treatment_qg1_tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
    rounds: usize,
    seed: u64,
    block_id_base: u64,
    sample_id_base: u64,
    group_id: Option<u64>,
    query_override: Option<&'a str>,
    qg1_stream_role: Option<&'a str>,
}

/// Round-stepping executor for one paired raw-sample stream with a seeded
/// balanced randomized first-arm schedule, warmup separation, and monotonic
/// per-sample intervals.
///
/// bd-yo5by: the A/A null floor bounds the A/B effect, so both must sample the
/// same time window. Construction runs the warmup; `run_round` executes exactly
/// one seeded block, letting `collect_cell` interleave several streams in time
/// while every per-stream schedule, block id, and sample id stays identical to
/// the previous serial layout.
struct PairedStreamRunner<'a> {
    context: &'a BenchContext,
    spec: &'a PerfCellSpec,
    evidence: &'a EvidenceContext,
    scope: &'a PerfOperationScope,
    origin: Instant,
    plan: StreamPlan<'a>,
    order: Vec<PerfSampleArm>,
    work_units: Option<u64>,
    byte_count: Option<u64>,
    estimator_config: &'a PairedEstimatorConfig,
    qg1_producer: Option<&'a Qg1LifecycleProducer>,
    consumed_qg1_sequences: BTreeSet<u64>,
    samples: Vec<PerfRawSample>,
}

impl<'a> PairedStreamRunner<'a> {
    fn new(
        context: &'a BenchContext,
        spec: &'a PerfCellSpec,
        evidence: &'a EvidenceContext,
        scope: &'a PerfOperationScope,
        origin: Instant,
        plan: StreamPlan<'a>,
        estimator_config: &'a PairedEstimatorConfig,
        qg1_producer: Option<&'a Qg1LifecycleProducer>,
    ) -> Self {
        let order =
            seeded_balanced_pair_order(plan.rounds, plan.seed).expect("paired order schedule");
        let (work_units, byte_count) = raw_sample_work(context, spec);
        for _ in 0..evidence.policy.warmup_rounds {
            let _ = black_box(measure_metric_with_query_and_qg1_writer_mode(
                context,
                spec,
                plan.control,
                plan.query_override,
                plan.control_qg1_tantivy_writer_mode,
            ));
            let _ = black_box(measure_metric_with_query_and_qg1_writer_mode(
                context,
                spec,
                plan.treatment,
                plan.query_override,
                plan.treatment_qg1_tantivy_writer_mode,
            ));
        }
        let samples = Vec::with_capacity(plan.rounds * 2);
        Self {
            context,
            spec,
            evidence,
            scope,
            origin,
            plan,
            order,
            work_units,
            byte_count,
            estimator_config,
            qg1_producer,
            consumed_qg1_sequences: BTreeSet::new(),
            samples,
        }
    }

    fn run_round(&mut self, round: usize) {
        let round_index = u64::try_from(round).expect("round fits u64");
        let block_id = self.plan.block_id_base + round_index;
        let control_sample_id = self.plan.sample_id_base + round_index * 2;
        let treatment_sample_id = control_sample_id + 1;
        if self.order[round] == PerfSampleArm::Control {
            let first = self.execute(
                self.plan.control,
                self.plan.control_qg1_tantivy_writer_mode,
                PerfSampleArm::Control,
                PerfSampleOrder::First,
                block_id,
                control_sample_id,
            );
            self.samples.push(first);
            let second = self.execute(
                self.plan.treatment,
                self.plan.treatment_qg1_tantivy_writer_mode,
                PerfSampleArm::Treatment,
                PerfSampleOrder::Second,
                block_id,
                treatment_sample_id,
            );
            self.samples.push(second);
        } else {
            let first = self.execute(
                self.plan.treatment,
                self.plan.treatment_qg1_tantivy_writer_mode,
                PerfSampleArm::Treatment,
                PerfSampleOrder::First,
                block_id,
                treatment_sample_id,
            );
            self.samples.push(first);
            let second = self.execute(
                self.plan.control,
                self.plan.control_qg1_tantivy_writer_mode,
                PerfSampleArm::Control,
                PerfSampleOrder::Second,
                block_id,
                control_sample_id,
            );
            self.samples.push(second);
        }
    }

    fn execute(
        &mut self,
        engine: EngineArm,
        qg1_tantivy_writer_mode: Option<Qg1TantivyWriterMode>,
        sample_arm: PerfSampleArm,
        sample_order: PerfSampleOrder,
        block_id: u64,
        sample_id: u64,
    ) -> PerfRawSample {
        let call_started_ns =
            u64::try_from(self.origin.elapsed().as_nanos()).expect("monotonic ns");
        let measurement = black_box(measure_metric_with_query_and_qg1_writer_mode(
            self.context,
            self.spec,
            engine,
            self.plan.query_override,
            qg1_tantivy_writer_mode,
        ));
        let call_ended_ns = u64::try_from(self.origin.elapsed().as_nanos()).expect("monotonic ns");
        // A QG-2 interval carries its own denominator, so the published work is
        // the work that interval actually processed rather than whatever the
        // plan declared. The window check below then has something to hold it
        // to, which is what stops a QG-2 rate being attached to time it was not
        // measured over.
        let (work_units, byte_count, continuous) = if let Some(interval) =
            measurement.qg2_continuous.as_ref()
        {
            (
                Some(interval.work_units),
                self.byte_count,
                Some(Qg1IntervalOffsets {
                    work_units: interval.work_units,
                    started_ns: u64::try_from(
                        interval.origin.duration_since(self.origin).as_nanos(),
                    )
                    .expect("monotonic ns"),
                    elapsed_ns: interval.elapsed_ns,
                }),
            )
        } else {
            let (work_units, byte_count) = qg1_raw_sample_denominator(
                (self.work_units, self.byte_count),
                measurement.continuous.as_ref(),
            )
            .expect("QG-1 raw sample denominator must bind the prepared measured input");
            let continuous = measurement
                .continuous
                .as_ref()
                .map(|interval| Qg1IntervalOffsets {
                    work_units: interval.work_units,
                    started_ns: u64::try_from(
                        interval.origin.duration_since(self.origin).as_nanos(),
                    )
                    .expect("monotonic ns"),
                    elapsed_ns: interval.elapsed_ns,
                });
            (work_units, byte_count, continuous)
        };
        let window = qg1_sample_window(
            self.scope.semantics,
            work_units,
            call_started_ns,
            call_ended_ns,
            continuous,
        )
        .expect("QG sample timing is not publishable");
        let qg1_sample_binding = if qg1_producer_coverage(self.spec)
            == Some(Qg1ProducerCoverage::EngineIndexingLifecycle)
        {
            let stream_sequence = block_id
                .checked_sub(self.plan.block_id_base)
                .and_then(|round| round.checked_mul(2))
                .and_then(|sequence| {
                    sequence.checked_add(match sample_order {
                        PerfSampleOrder::First => 0,
                        PerfSampleOrder::Second => 1,
                    })
                })
                .expect("QG-1 raw-order stream sequence fits u64");
            assert!(
                self.consumed_qg1_sequences.insert(stream_sequence),
                "QG-1 runner attempted to consume one issued transcript slot twice"
            );
            qg1_live_sample_binding(
                measurement.continuous.as_ref(),
                window.ended_ns - window.started_ns,
                self.scope,
                &self.evidence.sample_provenance,
                self.estimator_config,
                self.qg1_producer
                    .expect("QG-1 runner retains the live producer"),
                self.plan
                    .qg1_stream_role
                    .expect("QG-1 paired stream has one canonical role"),
                stream_sequence,
                sample_id,
                block_id,
                sample_arm,
                sample_order,
            )
        } else {
            None
        };
        if self.scope.semantics == PerfMetricSemantics::Throughput {
            // The published absolutes read `observed_value` while the estimator
            // recomputes the same rate from the published window. A QG-1 row is
            // only coherent if those are one number, not two derivations that
            // happen to agree.
            assert_eq!(
                measurement.value.to_bits(),
                throughput_per_second(
                    work_units.expect("throughput work units"),
                    window.ended_ns - window.started_ns,
                )
                .to_bits(),
                "QG-1 observed throughput must equal the rate derived from its published interval"
            );
        }
        PerfRawSample {
            block_id,
            sample_id,
            arm: sample_arm,
            order: sample_order,
            phase: PerfSamplePhase::Measurement,
            scope: self.scope.clone(),
            provenance: self.evidence.sample_provenance.clone(),
            started_ns: window.started_ns,
            ended_ns: window.ended_ns,
            work_units,
            byte_count,
            observed_value: Some(measurement.value),
            group_id: self.plan.group_id,
            qg6_sample_binding: None,
            qg1_sample_binding,
            tantivy_config_sha256: None,
        }
    }

    fn into_samples(self) -> Vec<PerfRawSample> {
        if qg1_producer_coverage(self.spec) == Some(Qg1ProducerCoverage::EngineIndexingLifecycle) {
            let stream_role = self
                .plan
                .qg1_stream_role
                .expect("QG-1 paired stream has one canonical role");
            assert_eq!(
                self.consumed_qg1_sequences.len(),
                self.samples.len(),
                "QG-1 runner must consume every emitted raw row exactly once"
            );
            assert_eq!(
                self.estimator_config.qg1_issued_stream_row_count(
                    self.scope,
                    &self.evidence.sample_provenance,
                    stream_role,
                ),
                Some(self.samples.len()),
                "QG-1 runner must consume every pre-issued transcript row exactly once"
            );
        }
        self.samples
    }
}

/// Which stream executes its next block, in seeded per-round order (bd-yo5by).
#[derive(Clone, Copy)]
enum StreamSlot {
    OracleNull,
    TreatmentNull,
    Effect,
}

/// The QG-1 effect and both independent A/A streams are one prepared-input
/// experiment.  The generic estimator verifies each effect/null pair; this
/// shipping collector also verifies the three-stream bundle before either
/// result can reach headline evidence.
fn validate_qg1_three_stream_prepared_identity(
    effect: &[PerfRawSample],
    tantivy_null: &[PerfRawSample],
    quill_null: &[PerfRawSample],
) -> Result<(), String> {
    let first = effect
        .first()
        .and_then(|sample| sample.qg1_sample_binding.as_ref())
        .ok_or_else(|| "QG-1 effect stream omitted its lifecycle binding".to_owned())?;
    for sample in effect.iter().chain(tantivy_null).chain(quill_null) {
        let binding = sample.qg1_sample_binding.as_ref().ok_or_else(|| {
            "QG-1 three-stream prepared-input bundle omitted a lifecycle binding".to_owned()
        })?;
        if binding.prepared_corpus_sha256 != first.prepared_corpus_sha256
            || binding.prepared_input_sha256 != first.prepared_input_sha256
            || binding.lifecycle_authority_sha256 != first.lifecycle_authority_sha256
        {
            return Err(
                "QG-1 effect and both null streams disagree on frozen trusted prepared input"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

/// Deterministic per-round permutation of the three stream slots so no stream
/// systematically runs first or last within a round (a fixed intra-round order
/// would reintroduce the carryover asymmetry the interleave exists to spread).
fn interleaved_stream_order(seed: u64, round: usize) -> [StreamSlot; 3] {
    const PERMUTATIONS: [[StreamSlot; 3]; 6] = [
        [
            StreamSlot::OracleNull,
            StreamSlot::TreatmentNull,
            StreamSlot::Effect,
        ],
        [
            StreamSlot::OracleNull,
            StreamSlot::Effect,
            StreamSlot::TreatmentNull,
        ],
        [
            StreamSlot::TreatmentNull,
            StreamSlot::OracleNull,
            StreamSlot::Effect,
        ],
        [
            StreamSlot::TreatmentNull,
            StreamSlot::Effect,
            StreamSlot::OracleNull,
        ],
        [
            StreamSlot::Effect,
            StreamSlot::OracleNull,
            StreamSlot::TreatmentNull,
        ],
        [
            StreamSlot::Effect,
            StreamSlot::TreatmentNull,
            StreamSlot::OracleNull,
        ],
    ];
    let round_index = u64::try_from(round).expect("round fits u64");
    let mut mixed = seed ^ 0x9e37_79b9_7f4a_7c15 ^ round_index.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^= mixed >> 30;
    mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed ^= mixed >> 27;
    mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 31;
    PERMUTATIONS[usize::try_from(mixed % 6).expect("permutation index")]
}

fn arm_values(samples: &[PerfRawSample], arm: PerfSampleArm) -> Vec<f64> {
    samples
        .iter()
        .filter(|sample| sample.arm == arm)
        .map(|sample| sample.observed_value.expect("gauge sample value"))
        .collect()
}

fn block_ratios_treatment_over_control(samples: &[PerfRawSample]) -> Vec<f64> {
    let mut by_block: BTreeMap<u64, (Option<f64>, Option<f64>)> = BTreeMap::new();
    for sample in samples {
        let entry = by_block.entry(sample.block_id).or_default();
        match sample.arm {
            PerfSampleArm::Control => entry.0 = sample.observed_value,
            PerfSampleArm::Treatment => entry.1 = sample.observed_value,
        }
    }
    by_block
        .values()
        .map(|(control, treatment)| {
            ratio(
                treatment.expect("treatment block value"),
                control.expect("control block value"),
            )
        })
        .collect()
}

fn values_checksum(samples: &[PerfRawSample]) -> u64 {
    let mut checksum = 0xcbf2_9ce4_8422_2325_u64;
    for sample in samples {
        checksum ^= sample
            .observed_value
            .expect("gauge sample value")
            .to_bits()
            .rotate_left(13);
        checksum = checksum.wrapping_mul(0x0000_0100_0000_01b3);
    }
    checksum
}

fn qg6_config_contract_sha256(spec: &PerfCellSpec) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch/qg6/semantic-config/v2\0");
    hasher.update(
        Qg6QuerySpec::normative_manifest_sha256()
            .expect("frozen 80-query manifest")
            .as_bytes(),
    );
    hasher.update(Qg6QuerySpec::normative_query_generator_revision().as_bytes());
    hasher.update(Qg6QuerySpec::normative_corpus_generator_revision().as_bytes());
    hasher.update(Qg6QuerySpec::sampling_frame().as_bytes());
    hasher.update([u8::from(
        spec.positions.unwrap_or(PositionMode::On).enabled(),
    )]);
    hasher.update(spec.threads.unwrap_or(1).to_le_bytes());
    hasher.update(spec.writer_heap_bytes.unwrap_or(50_000_000).to_le_bytes());
    hasher.update(spec.k.expect("QG-6 k").to_le_bytes());
    hasher.update(b"frankensearch-default-lexical-schema-and-parser-v1");
    hasher.update(b"qg6-rank-parity/full-corpus-native-tie-envelope-reviewed-score-epsilon-v3");
    hasher.update(
        b"score-epsilon=0.0001;reason=oracle-segment-geometry;\
          exact-membership-and-counts;epsilon-connected-order",
    );
    hasher.update(
        u64::try_from(QG6_TIE_EXPANSION_LIMIT)
            .expect("QG-6 tie expansion fits u64")
            .to_le_bytes(),
    );
    hasher.update(b"qg6-query-p50-subsample/v1");
    hasher.update(
        u64::try_from(QG6_TIMED_SEARCHES_PER_SAMPLE)
            .expect("QG-6 timed search count fits u64")
            .to_le_bytes(),
    );
    lower_hex(&hasher.finalize())
}

fn qg6_public_result_sha256(
    native_hits: &[(String, u32)],
    total_count: u64,
    doc_count: u64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch/qg6/public-total-result/v1\0");
    hasher.update(total_count.to_le_bytes());
    hasher.update(doc_count.to_le_bytes());
    hasher.update(
        u64::try_from(native_hits.len())
            .expect("bounded QG-6 top-k length")
            .to_le_bytes(),
    );
    for (doc_id, score_bits) in native_hits {
        hasher.update(
            u64::try_from(doc_id.len())
                .expect("bounded QG-6 document ID")
                .to_le_bytes(),
        );
        hasher.update(doc_id.as_bytes());
        hasher.update(score_bits.to_le_bytes());
    }
    lower_hex(&hasher.finalize())
}

fn qg6_preflight_result(
    context: &BenchContext,
    arm: &PreparedQueryArm,
    query: &Qg6QuerySpec,
    k: usize,
) -> Result<PreparedQueryPreflight, String> {
    match arm {
        PreparedQueryArm::Tantivy { role, index } => {
            let native = index
                .search_doc_ids(&context.cx, query.text(), k)
                .map_err(|error| error.to_string())?;
            let native_hits = native
                .iter()
                .map(|hit| (hit.doc_id.to_string(), hit.bm25_score.to_bits()))
                .collect::<Vec<_>>();
            let snippet_config = SnippetConfig {
                max_chars: 0,
                ..SnippetConfig::default()
            };
            let observed = index
                .oracle_observe_query(
                    &context.cx,
                    query.text(),
                    k,
                    QG6_TIE_EXPANSION_LIMIT,
                    &snippet_config,
                )
                .map_err(|error| error.to_string())?;
            let observed_native = observed
                .hits
                .iter()
                .map(|hit| (hit.doc_id.clone(), hit.score_bits))
                .collect::<Vec<_>>();
            if native_hits != observed_native {
                return Err(format!(
                    "Tantivy {role:?} native timed query disagrees with its total-result \
                     preflight observation"
                ));
            }
            let total_count = u64::try_from(observed.total_count)
                .map_err(|_| "Tantivy match count does not fit u64")?;
            let doc_count = u64::try_from(observed.doc_count)
                .map_err(|_| "Tantivy document count does not fit u64")?;
            let public_result_sha256 =
                qg6_public_result_sha256(&native_hits, total_count, doc_count);
            let hits = observed
                .hits
                .into_iter()
                .map(|hit| RankedHit {
                    doc_id: hit.doc_id,
                    score_bits: hit.score_bits,
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: hit.segment_ord,
                        doc_id: hit.segment_doc_id,
                    },
                })
                .collect();
            let cutoff_tie_group = observed
                .cutoff_tie_group
                .into_iter()
                .map(|hit| RankedHit {
                    doc_id: hit.doc_id,
                    score_bits: hit.score_bits,
                    native_tie_key: NativeTieKey::TantivyDocAddress {
                        segment_ord: hit.segment_ord,
                        doc_id: hit.segment_doc_id,
                    },
                })
                .collect();
            Ok(PreparedQueryPreflight {
                native_hits,
                total_count,
                doc_count,
                public_result_sha256,
                observation: Some(EngineObservation {
                    hits,
                    cutoff_tie_group,
                    cutoff_tie_complete: observed.cutoff_tie_complete,
                    offset_tie_group: Vec::new(),
                    offset_tie_complete: false,
                    snippets: BTreeMap::new(),
                    match_count: CountState::Value(total_count),
                    doc_count,
                    ast_differences: Vec::new(),
                }),
            })
        }
        PreparedQueryArm::Quill { index, .. } => {
            let native = index
                .search_doc_ids(&context.cx, query.text(), k)
                .map_err(|error| error.to_string())?;
            let native_hits = native
                .iter()
                .map(|hit| (hit.document_id.clone(), hit.score.to_bits()))
                .collect::<Vec<_>>();
            // Preserve the exact shipping count-free path above as the
            // observed native top-k. Exact counting intentionally changes the
            // collector/scorer mode and, under a large exact-score cutoff tie,
            // may choose a different native tie member. Count in an
            // independent zero-limit query so evidence cannot redefine the
            // result whose latency and rank are compared.
            let count_evidence = index
                .search_paginated(&context.cx, query.text(), 0, 0, true)
                .map_err(|error| error.to_string())?;
            let total_count = count_evidence
                .total_count
                .ok_or_else(|| "Quill tie evidence omitted its exact count".to_owned())?;
            let public_result_sha256 =
                qg6_public_result_sha256(&native_hits, total_count, count_evidence.doc_count);
            let hits = native
                .into_iter()
                .map(|hit| RankedHit {
                    doc_id: hit.document_id,
                    score_bits: hit.score.to_bits(),
                    native_tie_key: NativeTieKey::QuillDocId {
                        doc_id: hit.global_docid,
                    },
                })
                .collect();
            Ok(PreparedQueryPreflight {
                native_hits,
                total_count,
                doc_count: count_evidence.doc_count,
                public_result_sha256,
                observation: Some(EngineObservation {
                    hits,
                    cutoff_tie_group: Vec::new(),
                    cutoff_tie_complete: false,
                    offset_tie_group: Vec::new(),
                    offset_tie_complete: false,
                    snippets: BTreeMap::new(),
                    match_count: CountState::Value(total_count),
                    doc_count: count_evidence.doc_count,
                    ast_differences: Vec::new(),
                }),
            })
        }
    }
}

fn qg6_query_specs(spec: &PerfCellSpec) -> Vec<Qg6QuerySpec> {
    query_specs(spec.query_class.expect("QG-6 query class"))
}

fn qg6_raw_sample(
    sample: &frankensearch_quill_gauntlet::Qg6TimedSample,
    provenance: &PerfSampleProvenance,
    scope: &PerfOperationScope,
) -> PerfRawSample {
    debug_assert!(matches!(
        (sample.comparison, sample.arm),
        (
            Qg6Comparison::Null,
            Qg6ArmRole::NullLeft | Qg6ArmRole::NullRight
        ) | (
            Qg6Comparison::Effect,
            Qg6ArmRole::EffectControl | Qg6ArmRole::EffectTreatment
        )
    ));
    let arm = match sample.arm {
        Qg6ArmRole::NullLeft | Qg6ArmRole::EffectControl => PerfSampleArm::Control,
        Qg6ArmRole::NullRight | Qg6ArmRole::EffectTreatment => PerfSampleArm::Treatment,
    };
    let order = match sample.order {
        Qg6SampleOrder::First => PerfSampleOrder::First,
        Qg6SampleOrder::Second => PerfSampleOrder::Second,
    };
    PerfRawSample {
        block_id: sample.block_id,
        sample_id: sample.sample_id,
        arm,
        order,
        phase: PerfSamplePhase::Measurement,
        scope: scope.clone(),
        provenance: provenance.clone(),
        started_ns: sample.started_ns,
        ended_ns: sample.ended_ns,
        work_units: Some(sample.subsample_count),
        byte_count: None,
        observed_value: Some(sample.observed_latency_ns as f64 / 1_000_000.0),
        group_id: Some(u64::try_from(sample.query_index).expect("QG-6 query index")),
        qg6_sample_binding: Some(Qg6SampleBinding {
            query_id: sample.query_id.clone(),
            result_sequence_sha256: sample.result_sha256.clone(),
            timed_sample: sample.clone(),
        }),
        qg1_sample_binding: None,
        tantivy_config_sha256: None,
    }
}

fn prepared_qg6_streams(
    context: &BenchContext,
    spec: &PerfCellSpec,
    runs: usize,
    evidence: &EvidenceContext,
    scope: &PerfOperationScope,
    cell_seed: u64,
) -> (
    Vec<PerfRawSample>,
    Vec<PerfRawSample>,
    PerfInputIdentity,
    Qg6SemanticContract,
) {
    validate_qg6_queries_per_class(MANIFEST)
        .expect("gate manifest and compiled frozen QG-6 query universe agree");
    let count = context
        .scale
        .document_count(spec.document_count.expect("query corpus count"));
    let corpus = corpus_for(count);
    let corpus_sha256 = corpus
        .manifest()
        .expect("QG-6 exact corpus manifest")
        .content_sha256;
    let queries = qg6_query_specs(spec);
    let prepared = Qg6PreparedExperiment::prepare_with(
        corpus_sha256,
        qg6_config_contract_sha256(spec),
        count,
        spec.k.expect("QG-6 k"),
        queries,
        |role, identity, setup| {
            if role == Qg6ArmRole::EffectTreatment {
                let index = quill_in_memory(spec);
                let _ = index_batches_observed(
                    context,
                    &index,
                    &corpus,
                    identity.document_count,
                    None,
                    |batch| setup.record_population_batch(batch),
                );
                let _ = commit(context, &index);
                setup.record_commit();
                Ok(PreparedQueryArm::Quill {
                    index: Box::new(index),
                })
            } else {
                let index = tantivy_in_memory(spec);
                let _ = index_batches_observed(
                    context,
                    &index,
                    &corpus,
                    identity.document_count,
                    None,
                    |batch| setup.record_population_batch(batch),
                );
                let _ = commit(context, &index);
                setup.record_commit();
                Ok(PreparedQueryArm::Tantivy {
                    role,
                    index: Box::new(index),
                })
            }
        },
    )
    .expect("prepare four independent QG-6 arms");
    let mut preflight_counts = BTreeMap::<Qg6ArmRole, BTreeMap<String, (u64, u64)>>::new();
    let mut preflight_search = |arm: &PreparedQueryArm, query: &Qg6QuerySpec, k: usize| {
        let result = qg6_preflight_result(context, arm, query, k)?;
        let role = match arm {
            PreparedQueryArm::Quill { .. } => Qg6ArmRole::EffectTreatment,
            PreparedQueryArm::Tantivy { role, .. } => *role,
        };
        if preflight_counts
            .entry(role)
            .or_default()
            .insert(
                query.id().to_owned(),
                (result.total_count, result.doc_count),
            )
            .is_some()
        {
            return Err("QG-6 preflight repeated one role/query count binding".to_owned());
        }
        Ok(result)
    };
    let mut preflight_normalize = |result: &PreparedQueryPreflight| {
        Qg6SearchResult::from_ranked_hits(
            result
                .native_hits
                .iter()
                .map(|(doc_id, score_bits)| Qg6SearchHit::new(doc_id.clone(), *score_bits))
                .collect(),
            result.total_count,
            result.doc_count,
        )
    };
    let mut semantic_compare = |query: &Qg6QuerySpec,
                                expected_role: Qg6ArmRole,
                                expected: &PreparedQueryPreflight,
                                observed_role: Qg6ArmRole,
                                observed: &PreparedQueryPreflight| {
        if expected_role != Qg6ArmRole::NullLeft {
            return Err("QG-6 semantic comparator baseline is not null-left".to_owned());
        }
        if expected.total_count != observed.total_count || expected.doc_count != observed.doc_count
        {
            return Err(format!(
                "QG-6 total public-result contract differs for query_id={} roles={expected_role:?}/\
                 {observed_role:?}: count={}/{} doc_count={}/{}",
                query.id(),
                expected.total_count,
                observed.total_count,
                expected.doc_count,
                observed.doc_count,
            ));
        }
        if observed_role != Qg6ArmRole::EffectTreatment {
            return if expected.native_hits == observed.native_hits
                && expected.public_result_sha256 == observed.public_result_sha256
            {
                Ok(())
            } else {
                Err(format!(
                    "Tantivy A/A total-result preflight changed for query_id={} \
                     expected_sha256={} observed_sha256={}",
                    query.id(),
                    expected.public_result_sha256,
                    observed.public_result_sha256
                ))
            };
        }
        let subject = observed
            .observation
            .clone()
            .ok_or_else(|| "Quill preflight omitted tie-envelope evidence".to_owned())?;
        let oracle = expected
            .observation
            .clone()
            .ok_or_else(|| "Tantivy preflight omitted tie-envelope evidence".to_owned())?;
        let comparator_config = ComparatorConfig::default()
            .with_score_epsilon_reason(ScoreEpsilonReason::OracleSegmentGeometry);
        let report = compare_observations(subject, oracle, comparator_config)
            .map_err(|error| error.to_string())?;
        let non_exact_rank = !matches!(report.rank_class, RankClass::RankExact);
        if non_exact_rank && !query.allows_reviewed_divergence() {
            return Err(format!(
                "query_id={} produced an unregistered result divergence: rank={:?}",
                query.id(),
                report.rank_class
            ));
        }
        eprintln!(
            "[qg6-semantic-parity] query_id={} status={:?} rank={:?} \
             score_epsilon_reason={:?} score_epsilon_bits={} topk={} \
             count_equal={} doc_count_equal={} control_public_sha256={} \
             treatment_public_sha256={} support={:?}",
            query.id(),
            report.status,
            report.rank_class,
            report.score_epsilon_reason,
            comparator_config.score_epsilon_bits,
            report.subject.hits.len(),
            report.subject.match_count == report.oracle.match_count,
            report.subject.doc_count == report.oracle.doc_count,
            expected.public_result_sha256,
            observed.public_result_sha256,
            query.support_label(),
        );
        if report.status == ComparisonStatus::Failed
            || !matches!(
                report.rank_class,
                RankClass::RankExact | RankClass::TieOrder | RankClass::ScoreEpsilon
            )
        {
            let first_rank = report
                .first_divergence
                .as_deref()
                .and_then(|pointer| pointer.rsplit('/').next())
                .and_then(|index| index.parse::<usize>().ok());
            let subject_hit = first_rank.and_then(|index| report.subject.hits.get(index));
            let oracle_hit = first_rank.and_then(|index| report.oracle.hits.get(index));
            let hashed_doc_id = |hit: Option<&RankedHit>| {
                hit.map(|hit| lower_hex(&Sha256::digest(hit.doc_id.as_bytes())))
                    .unwrap_or_else(|| "absent".to_owned())
            };
            let subject_rank_in_oracle = subject_hit.and_then(|hit| {
                report
                    .oracle
                    .hits
                    .iter()
                    .position(|candidate| candidate.doc_id == hit.doc_id)
            });
            let oracle_rank_in_subject = oracle_hit.and_then(|hit| {
                report
                    .subject
                    .hits
                    .iter()
                    .position(|candidate| candidate.doc_id == hit.doc_id)
            });
            let score_group_bounds =
                |hits: &[RankedHit], index: Option<usize>| -> Option<(usize, usize)> {
                    let index = index?;
                    let score_bits = hits.get(index)?.score_bits;
                    let start = hits[..index]
                        .iter()
                        .rposition(|hit| hit.score_bits != score_bits)
                        .map_or(0, |position| position + 1);
                    let end = hits[index + 1..]
                        .iter()
                        .position(|hit| hit.score_bits != score_bits)
                        .map_or(hits.len(), |position| index + 1 + position);
                    Some((start, end))
                };
            let subject_group = score_group_bounds(&report.subject.hits, first_rank);
            let oracle_group = score_group_bounds(&report.oracle.hits, first_rank);
            let subject_map = report
                .subject
                .hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<BTreeMap<_, _>>();
            let oracle_map = report
                .oracle
                .hits
                .iter()
                .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
                .collect::<BTreeMap<_, _>>();
            let subject_only = subject_map
                .keys()
                .filter(|doc_id| !oracle_map.contains_key(**doc_id))
                .count();
            let oracle_only = oracle_map
                .keys()
                .filter(|doc_id| !subject_map.contains_key(**doc_id))
                .count();
            let common_score_mismatches = subject_map
                .iter()
                .filter(|(doc_id, score_bits)| {
                    oracle_map
                        .get(**doc_id)
                        .is_some_and(|oracle_bits| oracle_bits != *score_bits)
                })
                .count();
            let subject_in_cutoff = subject_hit.is_some_and(|hit| {
                report
                    .oracle
                    .cutoff_tie_group
                    .iter()
                    .any(|candidate| candidate.doc_id == hit.doc_id)
            });
            let oracle_in_cutoff = oracle_hit.is_some_and(|hit| {
                report
                    .oracle
                    .cutoff_tie_group
                    .iter()
                    .any(|candidate| candidate.doc_id == hit.doc_id)
            });
            eprintln!(
                "[qg6-parity-diagnostic] status={:?} rank={:?} first_rank={:?} \
                 subject_doc_sha256={} subject_score_bits={:?} oracle_doc_sha256={} \
                 oracle_score_bits={:?} cutoff_group_len={} cutoff_complete={} \
                 subject_in_cutoff={} oracle_in_cutoff={} subject_rank_in_oracle={:?} \
                 oracle_rank_in_subject={:?} subject_group={:?} oracle_group={:?} \
                 topk_map_equal={} subject_only={} oracle_only={} common_score_mismatches={} \
                 count_equal={} doc_count_equal={}",
                report.status,
                report.rank_class,
                first_rank,
                hashed_doc_id(subject_hit),
                subject_hit.map(|hit| hit.score_bits),
                hashed_doc_id(oracle_hit),
                oracle_hit.map(|hit| hit.score_bits),
                report.oracle.cutoff_tie_group.len(),
                report.oracle.cutoff_tie_complete,
                subject_in_cutoff,
                oracle_in_cutoff,
                subject_rank_in_oracle,
                oracle_rank_in_subject,
                subject_group,
                oracle_group,
                subject_map == oracle_map,
                subject_only,
                oracle_only,
                common_score_mismatches,
                report.subject.match_count == report.oracle.match_count,
                report.subject.doc_count == report.oracle.doc_count,
            );
            return Err(format!(
                "cross-engine result parity failed: status={:?} rank={:?} first={:?}",
                report.status, report.rank_class, report.first_divergence
            ));
        }
        Ok(())
    };
    let validated = prepared
        .validate_semantic_parity_with(
            &mut preflight_search,
            &mut preflight_normalize,
            &mut semantic_compare,
        )
        .expect("QG-6 score/tie-envelope preflight parity");
    let mut search = |arm: &PreparedQueryArm, query: &Qg6QuerySpec, k: usize, phase: Qg6Phase| {
        let role = match arm {
            PreparedQueryArm::Quill { .. } => Qg6ArmRole::EffectTreatment,
            PreparedQueryArm::Tantivy { role, .. } => *role,
        };
        if phase == Qg6Phase::Postflight {
            let result = qg6_preflight_result(context, arm, query, k)?;
            return Ok(PreparedQueryResult {
                native_hits: result.native_hits,
                total_count: result.total_count,
                doc_count: result.doc_count,
            });
        }
        let (total_count, doc_count) = preflight_counts
            .get(&role)
            .and_then(|queries| queries.get(query.id()))
            .copied()
            .ok_or_else(|| "QG-6 timed query has no accepted preflight counts".to_owned())?;
        let native_hits = match arm {
            PreparedQueryArm::Quill { index, .. } => index
                .search_doc_ids(&context.cx, query.text(), k)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|hit| (hit.document_id, hit.score.to_bits()))
                .collect(),
            PreparedQueryArm::Tantivy { index, .. } => index
                .search_doc_ids(&context.cx, query.text(), k)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|hit| (hit.doc_id.to_string(), hit.bm25_score.to_bits()))
                .collect(),
        };
        Ok(PreparedQueryResult {
            native_hits,
            total_count,
            doc_count,
        })
    };
    let mut normalize = |result: PreparedQueryResult| {
        Qg6SearchResult::from_ranked_hits(
            result
                .native_hits
                .into_iter()
                .map(|(doc_id, score_bits)| Qg6SearchHit::new(doc_id, score_bits))
                .collect(),
            result.total_count,
            result.doc_count,
        )
    };
    let rounds_per_query = runs
        .div_ceil(QG6_QUERY_GROUPS)
        .max(evidence.policy.min_group_pairs);
    let measurement = validated
        .measure_query_p50_with_normalizer(
            evidence.policy.warmup_rounds,
            rounds_per_query,
            QG6_TIMED_SEARCHES_PER_SAMPLE,
            cell_seed,
            &mut search,
            &mut normalize,
        )
        .expect("prepared QG-6 measurement");
    let mut result_receipt_hasher = Sha256::new();
    for sample in &measurement.samples {
        result_receipt_hasher.update(sample.result_sha256.as_bytes());
    }
    let input_identity = PerfInputIdentity {
        prepared_corpus_sha256: measurement.identity.corpus_sha256.clone(),
        query_manifest_sha256: measurement.identity.query_manifest_sha256.clone(),
        config_contract_sha256: measurement.identity.config_contract_sha256.clone(),
        semantic_contract_sha256: Some(measurement.semantic_contract.contract_sha256.clone()),
        query_group_count: QG6_QUERY_GROUPS,
        query_group_ids: QG6_QUERY_GROUP_IDS.to_vec(),
    };
    let semantic_contract = measurement.semantic_contract.clone();
    let mut sample_provenance = evidence.sample_provenance.clone();
    sample_provenance.input_identity = Some(input_identity.clone());
    eprintln!(
        "[qg6-prepared] fixture={} query_class={} query_count={} \
         global_query_manifest_sha256={} query_generator_revision={} corpus_generator_revision={} \
         corpus_sha256={} query_manifest_sha256={} \
         config_contract_sha256={} schedule_seed={} warmup_rounds={} rounds_per_query={} \
         searches_per_sample={} \
         sample_input_sha256={} result_receipt_sha256={} lifecycle={}",
        spec.fixture,
        spec.query_class.expect("QG-6 class").label(),
        QG6_QUERY_GROUPS,
        Qg6QuerySpec::normative_manifest_sha256().expect("global QG-6 manifest"),
        Qg6QuerySpec::normative_query_generator_revision(),
        Qg6QuerySpec::normative_corpus_generator_revision(),
        measurement.identity.corpus_sha256,
        measurement.identity.query_manifest_sha256,
        measurement.identity.config_contract_sha256,
        measurement.schedule_seed,
        measurement.warmup_rounds,
        measurement.rounds_per_query,
        measurement.searches_per_sample,
        input_identity.fingerprint_sha256(),
        lower_hex(&result_receipt_hasher.finalize()),
        serde_json::to_string(&measurement.lifecycle).expect("serialize QG-6 lifecycle"),
    );
    let mut null_samples = Vec::new();
    let mut effect_samples = Vec::new();
    for sample in measurement.samples {
        let comparison = sample.comparison;
        let sample = qg6_raw_sample(&sample, &sample_provenance, scope);
        match comparison {
            Qg6Comparison::Null => null_samples.push(sample),
            Qg6Comparison::Effect => effect_samples.push(sample),
        }
    }
    (
        null_samples,
        effect_samples,
        input_identity,
        semantic_contract,
    )
}

fn qg1_incumbent_digest<T: Serialize>(domain: &str, value: &T) -> String {
    let encoded = serde_json::to_vec(value).expect("serialize QG-1 incumbent receipt input");
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch.quill.qg1-live-incumbent.v1\0");
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(encoded);
    lower_hex(&hasher.finalize())
}

fn qg1_live_semantic_contract(
    spec: &PerfCellSpec,
    shipping_receipt: &BenchmarkWriterReceipt,
) -> Qg1TantivySemanticContract {
    let quill = quill_config(spec);
    Qg1TantivySemanticContract {
        tantivy_version: frankensearch_quill_gauntlet::QG1_TANTIVY_INCUMBENT_TANTIVY_VERSION
            .to_owned(),
        schema_sha256: qg1_incumbent_digest("tantivy.schema", &shipping_receipt.schema_fields),
        analyzer_sha256: qg1_incumbent_digest("tantivy.analyzer", &shipping_receipt.tokenizer_name),
        indexed_fields_sha256: qg1_incumbent_digest(
            "tantivy.indexed-fields",
            &(
                shipping_receipt.schema_fields.as_slice(),
                shipping_receipt.positions,
            ),
        ),
        merge_policy_sha256: qg1_incumbent_digest(
            "tantivy.merge-policy",
            &("Index::writer default", shipping_receipt.writer_rearmed),
        ),
        visibility_sha256: qg1_incumbent_digest(
            "shared.visibility",
            &(quill.max_visibility_lag_ms, "periodic-plus-terminal-commit"),
        ),
        searchable_terminal_scope_sha256: qg1_incumbent_digest(
            "shared.searchable-terminal-scope",
            &"worker-join-then-retained-tail-search",
        ),
        durability_sha256: qg1_incumbent_digest("shared.durability", &"in-memory"),
        quill_config_sha256: qg1_incumbent_digest("quill.config", &format!("{quill:?}")),
    }
}

fn qg1_live_observation_ids(label: &str, samples: &[PerfRawSample]) -> Vec<String> {
    samples
        .iter()
        .map(|sample| {
            qg1_incumbent_digest(
                "raw-observation",
                &(
                    label,
                    sample.block_id,
                    sample.sample_id,
                    sample.arm,
                    sample.order,
                ),
            )
        })
        .collect()
}

fn qg1_incumbent_stream_runner<'a>(
    context: &'a BenchContext,
    spec: &'a PerfCellSpec,
    evidence: &'a EvidenceContext,
    scope: &'a PerfOperationScope,
    origin: Instant,
    rounds: usize,
    seed: u64,
    block_id_base: u64,
    sample_id_base: u64,
    stream_role: &'static str,
    control: EngineArm,
    treatment: EngineArm,
    control_writer_mode: Option<Qg1TantivyWriterMode>,
    treatment_writer_mode: Option<Qg1TantivyWriterMode>,
    estimator_config: &'a PairedEstimatorConfig,
    producer: &'a Qg1LifecycleProducer,
) -> PairedStreamRunner<'a> {
    PairedStreamRunner::new(
        context,
        spec,
        evidence,
        scope,
        origin,
        StreamPlan {
            control,
            treatment,
            control_qg1_tantivy_writer_mode: control_writer_mode,
            treatment_qg1_tantivy_writer_mode: treatment_writer_mode,
            rounds,
            seed,
            block_id_base,
            sample_id_base,
            group_id: None,
            query_override: None,
            qg1_stream_role: Some(stream_role),
        },
        estimator_config,
        Some(producer),
    )
}

fn qg1_run_incumbent_streams_round_interleaved<'a, const N: usize>(
    mut runners: [PairedStreamRunner<'a>; N],
    rounds: usize,
) -> [Vec<PerfRawSample>; N] {
    assert!(N >= 2, "QG-1 interleaving requires multiple streams");
    for round in 0..rounds {
        for runner in &mut runners {
            runner.run_round(round);
        }
    }
    runners.map(PairedStreamRunner::into_samples)
}

struct Qg1LiveIncumbentCollection {
    screen: Qg1TantivyIncumbentScreen,
    decision: Option<Qg1TantivyIncumbentDecision>,
    selected_cell: Option<EvidenceCell>,
    selected_results: Vec<PerfCellResult>,
}

fn assert_qg1_selected_outputs_share_decision_source(
    spec: &PerfCellSpec,
    screen: &Qg1TantivyIncumbentScreen,
    decision: &Qg1TantivyIncumbentDecision,
    cell: &EvidenceCell,
    results: &[PerfCellResult],
) {
    assert!(
        matches!(
            screen
                .selected_candidate
                .as_ref()
                .map(|candidate| candidate.writer_mode),
            Some(Qg1TantivyWriterMode::Fixed { .. })
        ),
        "only an authenticated fixed-width selection may emit required QG-1 outputs"
    );
    let EvidenceCellBody::Paired {
        paired,
        treatment_arm_null: Some(treatment_arm_null),
        ..
    } = &cell.body
    else {
        panic!("selected QG-1 evidence must retain both decision nulls");
    };
    assert_eq!(paired.effect_samples, decision.tantivy_vs_quill.samples);
    assert_eq!(paired.null_samples, decision.tantivy_null.samples);
    assert_eq!(
        treatment_arm_null.effect_samples,
        decision.tantivy_vs_quill.samples
    );
    assert_eq!(treatment_arm_null.null_samples, decision.quill_null.samples);
    let expected_distributions = [
        DistributionSummary::from_samples(&arm_values(
            &decision.tantivy_vs_quill.samples,
            PerfSampleArm::Treatment,
        ))
        .expect("decision Quill threshold distribution"),
        DistributionSummary::from_samples(&arm_values(
            &decision.tantivy_vs_quill.samples,
            PerfSampleArm::Control,
        ))
        .expect("decision Tantivy threshold distribution"),
        DistributionSummary::from_samples(&block_ratios_treatment_over_control(
            &decision.tantivy_vs_quill.samples,
        ))
        .expect("decision paired threshold distribution"),
        DistributionSummary::from_samples(&block_ratios_treatment_over_control(
            &decision.tantivy_null.samples,
        ))
        .expect("decision T/T threshold distribution"),
        DistributionSummary::from_samples(&block_ratios_treatment_over_control(
            &decision.quill_null.samples,
        ))
        .expect("decision Q/Q threshold distribution"),
    ];
    assert!(
        results
            .iter()
            .zip(expected_distributions)
            .all(|(result, expected)| result.distribution == expected),
        "selected fixed mode must replace threshold and evidence sources together"
    );
    assert_eq!(
        results
            .iter()
            .map(|result| (result.metric.clone(), result.engine.clone()))
            .collect::<Vec<_>>(),
        vec![
            (spec.metric.clone(), EngineArm::Quill.label().to_owned()),
            (spec.metric.clone(), EngineArm::Tantivy.label().to_owned()),
            (
                format!("{}_quill_over_tantivy", spec.metric),
                "paired_ab".to_owned(),
            ),
            (
                format!("{}_tantivy_over_tantivy", spec.metric),
                "paired_null".to_owned(),
            ),
            (
                format!("{}_quill_over_quill", spec.metric),
                "paired_null_quill".to_owned(),
            ),
        ],
        "selected decision must replace threshold rows without changing canonical keys"
    );
}

fn qg1_collect_live_incumbent(
    context: &BenchContext,
    spec: &PerfCellSpec,
    role: EvidenceRole,
    runs: usize,
    evidence: &EvidenceContext,
    startup: &Qg1StartupProducer,
) -> Qg1LiveIncumbentCollection {
    let incumbent = startup
        .incumbent
        .as_ref()
        .expect("live QG-1 incumbent collection requires a preregistered screen");
    let scope = operation_scope(spec);
    let cell_seed = evidence.config.bootstrap_seed ^ fixture_seed(&spec.fixture);
    let candidates = preregister_qg1_tantivy_incumbents(
        spec,
        &incumbent.screen_plan,
        &qg1_live_semantic_contract(
            spec,
            &qg1_attest_writer_constructor(spec, Qg1TantivyWriterMode::ShippingAuto),
        ),
    )
    .expect("preregister live QG-1 Tantivy incumbents");
    assert_eq!(
        candidates.len(),
        incumbent.pilots.len(),
        "every preregistered candidate must own one pre-issued pilot authority"
    );
    let semantic_contract = candidates[0].semantic_contract.clone();
    let shipping_auto_config_sha256 = candidates[0].config_sha256.clone();
    let origin = Instant::now();
    let pilots = candidates
        .into_iter()
        .zip(&incumbent.pilots)
        .enumerate()
        .map(|(candidate_index, (candidate, pilot_startup))| {
            assert_eq!(candidate.writer_mode, pilot_startup.writer_mode);
            let constructor_receipt = qg1_attest_writer_constructor(spec, candidate.writer_mode);
            let constructor_receipt_sha256 =
                qg1_incumbent_digest("tantivy.writer-constructor", &constructor_receipt);
            let seed = cell_seed
                ^ u64::try_from(candidate_index)
                    .expect("QG-1 candidate index fits u64")
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let [effect, null] = qg1_run_incumbent_streams_round_interleaved(
                [
                    qg1_incumbent_stream_runner(
                        context,
                        spec,
                        evidence,
                        &scope,
                        origin,
                        runs,
                        seed,
                        0,
                        0,
                        QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT,
                        EngineArm::Tantivy,
                        EngineArm::Tantivy,
                        Some(Qg1TantivyWriterMode::ShippingAuto),
                        Some(candidate.writer_mode),
                        &pilot_startup.authority.estimator_config,
                        &pilot_startup.authority.producer,
                    ),
                    qg1_incumbent_stream_runner(
                        context,
                        spec,
                        evidence,
                        &scope,
                        origin,
                        runs,
                        seed ^ 0xaa,
                        2_000_000,
                        2_000_000,
                        QG1_STREAM_ROLE_TANTIVY_PILOT_NULL,
                        EngineArm::Tantivy,
                        EngineArm::Tantivy,
                        Some(candidate.writer_mode),
                        Some(candidate.writer_mode),
                        &pilot_startup.authority.estimator_config,
                        &pilot_startup.authority.producer,
                    ),
                ],
                runs,
            );
            let expected_authority = pilot_startup.authority.producer.expected_authority();
            let experiment = estimate_paired_experiment_against_qg1_authority(
                &effect,
                &null,
                &pilot_startup.authority.estimator_config,
                Some(expected_authority),
            )
            .expect("estimate producer-backed QG-1 candidate pilot");
            Qg1TantivyIncumbentPilot::from_experiment(
                candidate,
                constructor_receipt.materialized_width.authenticated(),
                constructor_receipt_sha256,
                &shipping_auto_config_sha256,
                experiment,
                qg1_live_observation_ids(&format!("pilot-effect-{candidate_index}"), &effect),
                qg1_live_observation_ids(&format!("pilot-null-{candidate_index}"), &null),
            )
            .expect("seal live QG-1 candidate pilot")
        })
        .collect::<Vec<_>>();
    let retained_authorities = incumbent
        .pilots
        .iter()
        .map(|pilot| pilot.authority.producer.expected_authority())
        .chain(std::iter::once(startup.producer.expected_authority()))
        .collect::<Vec<&Qg1ExpectedAuthority>>();
    let screen = Qg1TantivyIncumbentScreen::screen_against_qg1_authorities(
        spec,
        incumbent.screen_plan.clone(),
        &semantic_contract,
        pilots,
        &retained_authorities,
    )
    .expect("screen live QG-1 Tantivy incumbents");
    discard_concurrency_observations(spec);
    let decision = screen.selected_candidate.as_ref().map(|selected| {
        let selected_receipt = qg1_attest_writer_constructor(spec, selected.writer_mode);
        qg1_validate_writer_receipt(spec, selected.writer_mode, &selected_receipt);
        let [effect, tantivy_null, quill_null] = qg1_run_incumbent_streams_round_interleaved(
            [
                qg1_incumbent_stream_runner(
                    context,
                    spec,
                    evidence,
                    &scope,
                    origin,
                    runs,
                    cell_seed,
                    0,
                    0,
                    QG1_STREAM_ROLE_EFFECT,
                    EngineArm::Tantivy,
                    EngineArm::Quill,
                    Some(selected.writer_mode),
                    None,
                    &startup.estimator_config,
                    &startup.producer,
                ),
                qg1_incumbent_stream_runner(
                    context,
                    spec,
                    evidence,
                    &scope,
                    origin,
                    runs,
                    cell_seed ^ 0xaa,
                    0,
                    1_000_000,
                    QG1_STREAM_ROLE_TANTIVY_NULL,
                    EngineArm::Tantivy,
                    EngineArm::Tantivy,
                    Some(selected.writer_mode),
                    Some(selected.writer_mode),
                    &startup.estimator_config,
                    &startup.producer,
                ),
                qg1_incumbent_stream_runner(
                    context,
                    spec,
                    evidence,
                    &scope,
                    origin,
                    runs,
                    cell_seed ^ 0x55,
                    2_000_000,
                    2_000_000,
                    QG1_STREAM_ROLE_QUILL_NULL,
                    EngineArm::Quill,
                    EngineArm::Quill,
                    None,
                    None,
                    &startup.estimator_config,
                    &startup.producer,
                ),
            ],
            runs,
        );
        let bind = |kind,
                    control_engine_id: &str,
                    control_config: String,
                    treatment_engine_id: &str,
                    treatment_config: String,
                    samples: Vec<PerfRawSample>| {
            let observation_ids = qg1_live_observation_ids(&format!("decision-{kind:?}"), &samples);
            Qg1TantivyBoundStream::from_raw_samples(
                kind,
                control_engine_id.to_owned(),
                control_config,
                treatment_engine_id.to_owned(),
                treatment_config,
                samples,
                observation_ids,
            )
            .expect("seal fresh selected-candidate decision stream")
        };
        let decision = Qg1TantivyIncumbentDecision {
            estimator_config: startup.estimator_config.clone(),
            tantivy_vs_quill: bind(
                Qg1TantivyDecisionStreamKind::TantivyVsQuill,
                QG1_TANTIVY_ENGINE_ID,
                selected.config_sha256.clone(),
                QG1_QUILL_ENGINE_ID,
                semantic_contract.quill_config_sha256.clone(),
                effect,
            ),
            tantivy_null: bind(
                Qg1TantivyDecisionStreamKind::TantivyNull,
                QG1_TANTIVY_ENGINE_ID,
                selected.config_sha256.clone(),
                QG1_TANTIVY_ENGINE_ID,
                selected.config_sha256.clone(),
                tantivy_null,
            ),
            quill_null: bind(
                Qg1TantivyDecisionStreamKind::QuillNull,
                QG1_QUILL_ENGINE_ID,
                semantic_contract.quill_config_sha256.clone(),
                QG1_QUILL_ENGINE_ID,
                semantic_contract.quill_config_sha256.clone(),
                quill_null,
            ),
        };
        screen
            .validate_decision_against_qg1_authorities(
                spec,
                &semantic_contract,
                &decision,
                &retained_authorities,
            )
            .expect("validate fresh selected-candidate decision");
        decision
    });
    let (selected_cell, selected_results) = decision.as_ref().map_or_else(
        || (None, Vec::new()),
        |decision| {
            let expected_authority = startup.producer.expected_authority();
            let experiment = estimate_paired_experiment_against_qg1_authority(
                &decision.tantivy_vs_quill.samples,
                &decision.tantivy_null.samples,
                &decision.estimator_config,
                Some(expected_authority),
            )
            .expect("estimate selected-candidate T/Q against its fresh T/T null");
            let treatment_null = estimate_paired_experiment_against_qg1_authority(
                &decision.tantivy_vs_quill.samples,
                &decision.quill_null.samples,
                &decision.estimator_config,
                Some(expected_authority),
            )
            .expect("estimate selected-candidate T/Q against its fresh Q/Q null");
            let mut cell = EvidenceCell::evaluate(
                EvidenceCellSpec {
                    gate: spec.gate,
                    fixture: spec.fixture.clone(),
                    metric: spec.metric.clone(),
                    unit: unit(spec).to_owned(),
                    role,
                    input_identity: None,
                    qg6_semantic_contract: None,
                    cold_cache: None,
                    concurrency_witness: take_concurrency_witness(spec),
                },
                experiment,
                &evidence.policy,
            )
            .expect("evaluate required QG-1 cell from the selected fresh decision");
            cell.attach_treatment_arm_null_against_qg1_authority(
                treatment_null,
                &evidence.policy,
                Some(expected_authority),
            )
            .expect("attach fresh selected-decision Q/Q null");
            let effect = &decision.tantivy_vs_quill.samples;
            let tantivy_null = &decision.tantivy_null.samples;
            let quill_null = &decision.quill_null.samples;
            let quill_distribution =
                DistributionSummary::from_samples(&arm_values(effect, PerfSampleArm::Treatment))
                    .expect("selected-decision Quill distribution");
            let tantivy_distribution =
                DistributionSummary::from_samples(&arm_values(effect, PerfSampleArm::Control))
                    .expect("selected-decision Tantivy distribution");
            let results = vec![
                PerfCellResult {
                    fixture: spec.fixture.clone(),
                    metric: spec.metric.clone(),
                    engine: EngineArm::Quill.label().to_owned(),
                    unit: unit(spec).to_owned(),
                    distribution: quill_distribution,
                },
                PerfCellResult {
                    fixture: spec.fixture.clone(),
                    metric: spec.metric.clone(),
                    engine: EngineArm::Tantivy.label().to_owned(),
                    unit: unit(spec).to_owned(),
                    distribution: tantivy_distribution,
                },
                PerfCellResult {
                    fixture: spec.fixture.clone(),
                    metric: format!("{}_quill_over_tantivy", spec.metric),
                    engine: "paired_ab".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: DistributionSummary::from_samples(
                        &block_ratios_treatment_over_control(effect),
                    )
                    .expect("selected-decision paired distribution"),
                },
                PerfCellResult {
                    fixture: spec.fixture.clone(),
                    metric: format!("{}_tantivy_over_tantivy", spec.metric),
                    engine: "paired_null".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: DistributionSummary::from_samples(
                        &block_ratios_treatment_over_control(tantivy_null),
                    )
                    .expect("selected-decision T/T distribution"),
                },
                PerfCellResult {
                    fixture: spec.fixture.clone(),
                    metric: format!("{}_quill_over_quill", spec.metric),
                    engine: "paired_null_quill".to_owned(),
                    unit: "ratio".to_owned(),
                    distribution: DistributionSummary::from_samples(
                        &block_ratios_treatment_over_control(quill_null),
                    )
                    .expect("selected-decision Q/Q distribution"),
                },
            ];
            assert_qg1_selected_outputs_share_decision_source(
                spec, &screen, decision, &cell, &results,
            );
            (Some(cell), results)
        },
    );
    Qg1LiveIncumbentCollection {
        screen,
        decision,
        selected_cell,
        selected_results,
    }
}

struct CellCollection {
    results: Vec<PerfCellResult>,
    evidence: Option<EvidenceCell>,
    qg1_incumbent_screen: Option<Qg1TantivyIncumbentScreen>,
    qg1_incumbent_decision: Option<Qg1TantivyIncumbentDecision>,
}

fn collect_cell(
    context: &BenchContext,
    spec: &PerfCellSpec,
    role: EvidenceRole,
    runs: usize,
    evidence: &EvidenceContext,
    qg1_startup_producer: Option<&Qg1StartupProducer>,
) -> CellCollection {
    if spec.gate == PerfGate::Qg10 {
        let samples = (0..runs)
            .map(|_| dependency_surface_metric())
            .collect::<Vec<_>>();
        let results = vec![PerfCellResult {
            fixture: spec.fixture.clone(),
            metric: spec.metric.clone(),
            engine: "default_feature_graph".to_owned(),
            unit: unit(spec).to_owned(),
            distribution: DistributionSummary::from_samples(&samples).expect("QG-10 distribution"),
        }];
        let cell = EvidenceCell::facts(
            EvidenceCellSpec {
                gate: spec.gate,
                fixture: spec.fixture.clone(),
                metric: spec.metric.clone(),
                unit: unit(spec).to_owned(),
                role,
                input_identity: None,
                qg6_semantic_contract: None,
                cold_cache: None,
                concurrency_witness: None,
            },
            samples,
            &evidence.policy,
        )
        .expect("QG-10 facts evidence cell");
        return CellCollection {
            results,
            evidence: Some(cell),
            qg1_incumbent_screen: None,
            qg1_incumbent_decision: None,
        };
    }

    let mut qg1_no_decision_screen = None;
    if let Some(startup) = qg1_startup_producer
        && startup.incumbent.is_some()
    {
        let incumbent = qg1_collect_live_incumbent(context, spec, role, runs, evidence, startup);
        if let Some(selected_cell) = incumbent.selected_cell {
            assert!(
                incumbent.decision.is_some() && incumbent.screen.selected_candidate.is_some(),
                "a selected required cell must come from a fresh selected-candidate decision"
            );
            return CellCollection {
                results: incumbent.selected_results,
                evidence: Some(selected_cell),
                qg1_incumbent_screen: Some(incumbent.screen),
                qg1_incumbent_decision: incumbent.decision,
            };
        }
        assert!(
            incumbent.decision.is_none()
                && incumbent.screen.selected_candidate.is_none()
                && incumbent.screen.no_decision_reason.is_some(),
            "an incomplete QG-1 screen must be an explicit NoDecision"
        );
        qg1_no_decision_screen = Some(incumbent.screen);
    }

    let scope = operation_scope(spec);
    // Preserve the legacy origin for every non-QG-1 gate. QG-1 deliberately
    // defers clock creation until its authority has been parent-acknowledged.
    let origin = (spec.gate != PerfGate::Qg1).then(Instant::now);
    let cell_seed = evidence.config.bootstrap_seed ^ fixture_seed(&spec.fixture);
    let (estimator_config, qg1_lifecycle_producer) = match qg1_producer_coverage(spec) {
        Some(Qg1ProducerCoverage::EngineIndexingLifecycle) => {
            let startup = qg1_startup_producer
                .expect("selected engine QG-1 cell must reuse its preflight startup producer");
            (startup.estimator_config.clone(), Some(&startup.producer))
        }
        Some(Qg1ProducerCoverage::TokenizerOnlyDiagnosticNoEngineLifecycle) | None => {
            assert!(
                qg1_startup_producer.is_none(),
                "tokenizer-only and non-QG-1 cells must not receive an engine authority producer"
            );
            (evidence.config.clone(), None)
        }
    };
    let origin = origin.unwrap_or_else(Instant::now);

    // Every non-query gate establishes its A/A floor through the exact paired
    // routine. QG-6 uses the prepared four-arm runner so setup is impossible
    // inside timed samples. For every other gate the null and effect streams
    // are interleaved round-by-round under one seeded schedule (bd-yo5by): a
    // null band sampled in an earlier, quieter phase says nothing about the
    // noise DURING the effect measurement, so the streams must share the
    // measurement window.
    let (
        oracle_null_samples,
        treatment_null_samples,
        effect_samples,
        input_identity,
        qg6_semantic_contract,
    ) = if spec.gate == PerfGate::Qg6 {
        let (null, effect, input_identity, semantic_contract) =
            prepared_qg6_streams(context, spec, runs, evidence, &scope, cell_seed);
        (
            null,
            None,
            effect,
            Some(input_identity),
            Some(semantic_contract),
        )
    } else {
        let mut oracle_null = PairedStreamRunner::new(
            context,
            spec,
            evidence,
            &scope,
            origin,
            StreamPlan {
                control: EngineArm::Tantivy,
                treatment: EngineArm::Tantivy,
                control_qg1_tantivy_writer_mode: None,
                treatment_qg1_tantivy_writer_mode: None,
                rounds: runs,
                seed: cell_seed ^ 0xaa,
                block_id_base: 0,
                sample_id_base: 1_000_000,
                group_id: None,
                query_override: None,
                qg1_stream_role: (spec.gate == PerfGate::Qg1).then_some("qg1.null.tantivy.v1"),
            },
            &estimator_config,
            qg1_lifecycle_producer,
        );
        let mut treatment_null = (spec.gate == PerfGate::Qg1).then(|| {
            PairedStreamRunner::new(
                context,
                spec,
                evidence,
                &scope,
                origin,
                StreamPlan {
                    control: EngineArm::Quill,
                    treatment: EngineArm::Quill,
                    control_qg1_tantivy_writer_mode: None,
                    treatment_qg1_tantivy_writer_mode: None,
                    rounds: runs,
                    seed: cell_seed ^ 0x55,
                    block_id_base: 2_000_000,
                    sample_id_base: 2_000_000,
                    group_id: None,
                    query_override: None,
                    qg1_stream_role: Some("qg1.null.quill.v1"),
                },
                &estimator_config,
                qg1_lifecycle_producer,
            )
        });
        let mut effect = PairedStreamRunner::new(
            context,
            spec,
            evidence,
            &scope,
            origin,
            StreamPlan {
                control: EngineArm::Tantivy,
                treatment: EngineArm::Quill,
                control_qg1_tantivy_writer_mode: None,
                treatment_qg1_tantivy_writer_mode: None,
                rounds: runs,
                seed: cell_seed,
                block_id_base: 0,
                sample_id_base: 0,
                group_id: None,
                query_override: None,
                qg1_stream_role: (spec.gate == PerfGate::Qg1)
                    .then_some("qg1.effect.tantivy_vs_quill.v1"),
            },
            &estimator_config,
            qg1_lifecycle_producer,
        );
        for round in 0..runs {
            for slot in interleaved_stream_order(cell_seed, round) {
                match slot {
                    StreamSlot::OracleNull => oracle_null.run_round(round),
                    StreamSlot::TreatmentNull => {
                        if let Some(runner) = treatment_null.as_mut() {
                            runner.run_round(round);
                        }
                    }
                    StreamSlot::Effect => effect.run_round(round),
                }
            }
        }
        (
            oracle_null.into_samples(),
            treatment_null.map(PairedStreamRunner::into_samples),
            effect.into_samples(),
            None,
            None,
        )
    };

    let quill_distribution =
        DistributionSummary::from_samples(&arm_values(&effect_samples, PerfSampleArm::Treatment))
            .expect("Quill distribution");
    let oracle_distribution =
        DistributionSummary::from_samples(&arm_values(&effect_samples, PerfSampleArm::Control))
            .expect("oracle distribution");
    let paired_distribution =
        DistributionSummary::from_samples(&block_ratios_treatment_over_control(&effect_samples))
            .expect("paired distribution");
    let oracle_null_distribution = DistributionSummary::from_samples(
        &block_ratios_treatment_over_control(&oracle_null_samples),
    )
    .expect("oracle null distribution");
    let treatment_null_distribution = treatment_null_samples.as_ref().map(|samples| {
        DistributionSummary::from_samples(&block_ratios_treatment_over_control(samples))
            .expect("treatment-arm null distribution")
    });
    eprintln!(
        "[quill-perf-paired] fixture={} tantivy_null_median={:.6} \
         tantivy_null_ci95=[{:.6},{:.6}] tantivy_null_cv_pct={:.3} \
         quill_null_median={} quill_null_ci95={} quill_null_cv_pct={} \
         ab_median={:.6} ab_ci95=[{:.6},{:.6}] ab_cv_pct={:.3} checksum={:016x}",
        spec.fixture,
        oracle_null_distribution.p50,
        oracle_null_distribution.median_ci95_low,
        oracle_null_distribution.median_ci95_high,
        oracle_null_distribution.cv_pct,
        treatment_null_distribution
            .as_ref()
            .map_or_else(|| "n/a".to_owned(), |summary| format!("{:.6}", summary.p50)),
        treatment_null_distribution.as_ref().map_or_else(
            || "n/a".to_owned(),
            |summary| {
                format!(
                    "[{:.6},{:.6}]",
                    summary.median_ci95_low, summary.median_ci95_high
                )
            },
        ),
        treatment_null_distribution.as_ref().map_or_else(
            || "n/a".to_owned(),
            |summary| format!("{:.3}", summary.cv_pct)
        ),
        paired_distribution.p50,
        paired_distribution.median_ci95_low,
        paired_distribution.median_ci95_high,
        paired_distribution.cv_pct,
        values_checksum(&oracle_null_samples)
            ^ treatment_null_samples
                .as_deref()
                .map_or(0, values_checksum)
                .rotate_left(17)
            ^ values_checksum(&effect_samples).rotate_left(29),
    );

    if qg1_producer_coverage(spec) == Some(Qg1ProducerCoverage::EngineIndexingLifecycle) {
        validate_qg1_three_stream_prepared_identity(
            &effect_samples,
            &oracle_null_samples,
            treatment_null_samples
                .as_deref()
                .expect("QG-1 requires a Quill/Quill A/A stream"),
        )
        .expect("QG-1 harness-produced streams must share one frozen prepared input");
    }

    let qg1_expected_authority =
        qg1_lifecycle_producer.map(Qg1LifecycleProducer::expected_authority);
    let experiment = estimate_paired_experiment_against_qg1_authority(
        &effect_samples,
        &oracle_null_samples,
        &estimator_config,
        qg1_expected_authority,
    )
    .expect("paired estimator rejected harness-produced streams");
    let treatment_null_experiment = treatment_null_samples.as_ref().map(|samples| {
        estimate_paired_experiment_against_qg1_authority(
            &effect_samples,
            samples,
            &estimator_config,
            qg1_expected_authority,
        )
        .expect("treatment-arm null estimator rejected harness-produced streams")
    });
    let is_tokenizer_null = spec.metric == "tokenize_docs_per_second";
    let cold_cache = take_cold_cache_evidence(spec);
    let mut cell = EvidenceCell::evaluate(
        EvidenceCellSpec {
            gate: spec.gate,
            fixture: spec.fixture.clone(),
            metric: spec.metric.clone(),
            unit: unit(spec).to_owned(),
            role,
            input_identity,
            qg6_semantic_contract,
            cold_cache,
            concurrency_witness: take_concurrency_witness(spec),
        },
        experiment,
        &evidence.policy,
    )
    .expect("evidence cell evaluation");
    if let Some(treatment_null_experiment) = treatment_null_experiment {
        // The QG-1 null is authenticated by the same retained producer
        // expectation as the effect stream. Authority-free attachment refuses
        // a config that carries a sealed QG-1 authority, so the live producer
        // hands its expectation to the attach seam exactly as it does to the
        // estimator above.
        cell.attach_treatment_arm_null_against_qg1_authority(
            treatment_null_experiment,
            &evidence.policy,
            qg1_expected_authority,
        )
        .expect("attach treatment-arm A/A null");
    }

    let absolute_engine = if is_tokenizer_null {
        "quill_tokenizer"
    } else {
        EngineArm::Quill.label()
    };
    let mut results = vec![
        PerfCellResult {
            fixture: spec.fixture.clone(),
            metric: spec.metric.clone(),
            engine: absolute_engine.to_owned(),
            unit: unit(spec).to_owned(),
            distribution: quill_distribution,
        },
        PerfCellResult {
            fixture: spec.fixture.clone(),
            metric: spec.metric.clone(),
            engine: if is_tokenizer_null {
                "quill_tokenizer_null".to_owned()
            } else {
                EngineArm::Tantivy.label().to_owned()
            },
            unit: unit(spec).to_owned(),
            distribution: oracle_distribution,
        },
        PerfCellResult {
            fixture: spec.fixture.clone(),
            metric: format!("{}_quill_over_tantivy", spec.metric),
            engine: "paired_ab".to_owned(),
            unit: "ratio".to_owned(),
            distribution: paired_distribution,
        },
        PerfCellResult {
            fixture: spec.fixture.clone(),
            metric: format!("{}_tantivy_over_tantivy", spec.metric),
            engine: "paired_null".to_owned(),
            unit: "ratio".to_owned(),
            distribution: oracle_null_distribution,
        },
    ];
    if let Some(distribution) = treatment_null_distribution {
        results.push(PerfCellResult {
            fixture: spec.fixture.clone(),
            metric: format!("{}_quill_over_quill", spec.metric),
            engine: "paired_null_quill".to_owned(),
            unit: "ratio".to_owned(),
            distribution,
        });
    }
    CellCollection {
        results,
        evidence: Some(cell),
        qg1_incumbent_screen: qg1_no_decision_screen,
        qg1_incumbent_decision: None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedPerfCell {
    ordinal: usize,
    spec: PerfCellSpec,
    role: EvidenceRole,
}

fn selected_cells(
    matrix: &PerfMatrixSpec,
    runner: &RunnerApplicabilityContext,
    scale: MatrixScale,
    fixture_filter: Option<&str>,
) -> Result<Vec<PlannedPerfCell>, String> {
    let gate = runner.plan.binding().gate;
    let canonical = matrix.for_gate(gate);
    if canonical.len() != runner.plan.cells.len() {
        return Err(format!(
            "canonical {gate} matrix has {} cells but applicability plan classifies {}",
            canonical.len(),
            runner.plan.cells.len()
        ));
    }

    let mut filter_matched = false;
    let mut selected = Vec::new();
    for (ordinal, (spec, classification)) in canonical
        .into_iter()
        .zip(runner.plan.cells.iter())
        .enumerate()
    {
        if classification.ordinal != ordinal {
            return Err(format!(
                "applicability plan ordinal {} occupies canonical {gate} position {ordinal}",
                classification.ordinal
            ));
        }
        let cell_sha256 = spec
            .contract_sha256()
            .map_err(|error| format!("cannot hash canonical {gate} cell {ordinal}: {error}"))?;
        if classification.cell_contract_sha256 != cell_sha256 {
            return Err(format!(
                "applicability plan hash for canonical {gate} cell {ordinal} does not match"
            ));
        }
        if spec.threads != Some(classification.configured_threads) {
            return Err(format!(
                "applicability plan width for canonical {gate} cell {ordinal} does not match"
            ));
        }

        let matches_filter = fixture_filter.is_none_or(|fixture| spec.fixture == fixture);
        if fixture_filter.is_some() && matches_filter {
            filter_matched = true;
        }
        let role = match classification.applicability {
            PerfCellApplicability::Required => EvidenceRole::Required,
            PerfCellApplicability::Diagnostic => EvidenceRole::Diagnostic,
            PerfCellApplicability::NotApplicable => {
                if fixture_filter.is_some() && matches_filter {
                    return Err(format!(
                        "QUILL_PERF_FIXTURE selected non-applicable {gate} cell {:?} for profile \
                         {}.{}",
                        spec.fixture,
                        runner.profile.hardware_class_id().as_str(),
                        runner.profile.execution_profile_id().as_str()
                    ));
                }
                continue;
            }
        };
        if matches_filter {
            selected.push(PlannedPerfCell {
                ordinal,
                spec: spec.clone(),
                role,
            });
        }
    }

    if fixture_filter.is_some() && !filter_matched {
        return Err("QUILL_PERF_FIXTURE matched no canonical gate cell".to_owned());
    }
    if !scale.is_full() {
        selected.truncate(1);
    }
    if selected.is_empty() {
        return Err("QG applicability-plan selection contains no runnable cells".to_owned());
    }
    let configured_max = selected
        .iter()
        .filter_map(|cell| cell.spec.threads)
        .max()
        .ok_or_else(|| "selected runnable cells have no configured width".to_owned())?;
    let configured_max = u64::try_from(configured_max)
        .map_err(|_| "selected configured width is not representable".to_owned())?;
    if configured_max > runner.max_exercised_cell_width
        || configured_max > runner.execution_capacity
    {
        return Err(format!(
            "selected configured width {configured_max} exceeds profile maximum {} or capacity {}",
            runner.max_exercised_cell_width, runner.execution_capacity
        ));
    }
    Ok(selected)
}

fn fixture_filter_from_env() -> Result<Option<String>, String> {
    match std::env::var("QUILL_PERF_FIXTURE") {
        Ok(value) if !value.is_empty() && value.trim() == value => Ok(Some(value)),
        Ok(_) => Err("QUILL_PERF_FIXTURE must be nonempty canonical text".to_owned()),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("QUILL_PERF_FIXTURE is invalid: {error}")),
    }
}

fn configured_engine_widths(selected: &[PlannedPerfCell]) -> Vec<usize> {
    selected
        .iter()
        .filter_map(|cell| cell.spec.threads)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn gate_selection_complete(
    runner: &RunnerApplicabilityContext,
    selected: &[PlannedPerfCell],
    scale: MatrixScale,
    fixture_filter: Option<&str>,
) -> bool {
    if !scale.is_full() || fixture_filter.is_some() {
        return false;
    }
    let expected = runner
        .plan
        .cells
        .iter()
        .filter(|cell| cell.applicability.is_runnable())
        .map(|cell| cell.ordinal)
        .collect::<BTreeSet<_>>();
    let actual = selected
        .iter()
        .map(|cell| cell.ordinal)
        .collect::<BTreeSet<_>>();
    !expected.is_empty() && actual == expected && actual.len() == selected.len()
}

fn partial_shard_no_claim(
    gate: PerfGate,
    selection_complete: bool,
) -> Option<(&'static str, &'static str)> {
    if selection_complete {
        None
    } else if gate == PerfGate::Qg1 {
        Some((
            QG1_PARTIAL_SHARD_NO_CLAIM_CODE,
            QG1_PARTIAL_SHARD_NO_CLAIM_DETAIL,
        ))
    } else {
        Some((
            "evidence.incomplete_gate_selection",
            "the invocation selected only part of the normative gate; durable pre-admission \
             evidence cannot support a publication or ratchet claim",
        ))
    }
}

fn git_revision(scale: MatrixScale) -> String {
    if let Ok(revision) = std::env::var("QUILL_PERF_GIT_REV")
        && !revision.trim().is_empty()
    {
        return revision.trim().to_owned();
    }
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("read benchmark git revision");
    if output.status.success() {
        return String::from_utf8(output.stdout)
            .expect("git revision UTF-8")
            .trim()
            .to_owned();
    }
    assert!(
        !scale.is_full(),
        "full QG evidence requires QUILL_PERF_GIT_REV when the worker snapshot has no .git metadata"
    );
    "unavailable-smoke-snapshot".to_owned()
}

fn manifest_sha256() -> String {
    perf_manifest_contract_sha256(MANIFEST)
}

type Qg1PrefixIdentityMap = BTreeMap<u64, (String, String)>;

fn qg1_effective_counts(scale: MatrixScale, cells: &[PerfCellSpec]) -> BTreeSet<u64> {
    cells
        .iter()
        .filter(|cell| cell.gate == PerfGate::Qg1)
        .map(|cell| scale.document_count(cell.document_count.unwrap_or_default()))
        .collect()
}

fn prepared_qg1_prefix_identities(
    context: &BenchContext,
    cells: &[PerfCellSpec],
) -> Result<Qg1PrefixIdentityMap, String> {
    let qg1_counts = cells
        .iter()
        .filter(|cell| cell.gate == PerfGate::Qg1)
        .map(|cell| {
            context
                .scale
                .document_count(cell.document_count.unwrap_or_default())
        })
        .collect::<BTreeSet<_>>();
    qg1_counts
        .into_iter()
        .map(|document_count| {
            let prepared = context
                .prepared_qg1
                .as_ref()
                .ok_or_else(|| "selected QG-1 cells have no prepared corpus".to_owned())?;
            let (manifest_sha256, indexed_content_sha256) =
                prepared.validate_prefix(document_count)?;
            Ok((
                document_count,
                (
                    manifest_sha256.to_owned(),
                    indexed_content_sha256.to_owned(),
                ),
            ))
        })
        .collect::<Result<_, String>>()
}

fn replay_qg1_prefix_identities(
    scale: MatrixScale,
    cells: &[PerfCellSpec],
) -> Result<Qg1PrefixIdentityMap, String> {
    qg1_effective_counts(scale, cells)
        .into_iter()
        .map(|document_count| {
            let corpus = corpus_for(document_count);
            let manifest = corpus
                .manifest()
                .map_err(|error| format!("build authoritative QG-1 corpus manifest: {error}"))?;
            manifest
                .verify_documents(corpus.iter())
                .map_err(|error| format!("replay authoritative QG-1 corpus manifest: {error}"))?;
            let manifest_sha256 = manifest
                .manifest_hash()
                .map_err(|error| format!("hash authoritative QG-1 corpus manifest: {error}"))?;
            let indexed_content_sha256 = qg1_indexed_content_sha256(
                document_count,
                corpus.iter().map(IndexableDocument::from),
            )?;
            Ok((document_count, (manifest_sha256, indexed_content_sha256)))
        })
        .collect()
}

fn frozen_qg1_full_prefix_identities(
    canonical_cells: &[PerfCellSpec],
) -> Result<Qg1PrefixIdentityMap, String> {
    let expected_counts = qg1_effective_counts(MatrixScale::Full, canonical_cells);
    let mut identities = BTreeMap::new();
    for (document_count, manifest_sha256, indexed_content_sha256) in QG1_FULL_PREFIX_IDENTITY_PINS {
        let is_lower_sha256 = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
        };
        if !is_lower_sha256(manifest_sha256) || !is_lower_sha256(indexed_content_sha256) {
            return Err(format!(
                "frozen QG-1 prefix {document_count} has a malformed identity pin"
            ));
        }
        if identities
            .insert(
                document_count,
                (
                    manifest_sha256.to_owned(),
                    indexed_content_sha256.to_owned(),
                ),
            )
            .is_some()
        {
            return Err(format!(
                "frozen QG-1 prefix identity repeats document count {document_count}"
            ));
        }
    }
    let pinned_counts = identities.keys().copied().collect::<BTreeSet<_>>();
    if pinned_counts != expected_counts {
        return Err(format!(
            "frozen QG-1 prefix counts {pinned_counts:?} differ from canonical counts \
             {expected_counts:?}"
        ));
    }
    Ok(identities)
}

fn validate_selected_qg1_prefixes(
    context: &BenchContext,
    selected: &[PerfCellSpec],
    authoritative: &Qg1PrefixIdentityMap,
) -> Result<(), String> {
    let selected_identities = prepared_qg1_prefix_identities(context, selected)?;
    for (document_count, selected_identity) in selected_identities {
        let authoritative_identity = authoritative.get(&document_count).ok_or_else(|| {
            format!(
                "selected QG-1 corpus prefix {document_count} is absent from the authoritative \
                 full-corpus identity"
            )
        })?;
        if &selected_identity != authoritative_identity {
            return Err(format!(
                "selected QG-1 corpus prefix {document_count} differs from the authoritative \
                 full-corpus identity"
            ));
        }
    }
    Ok(())
}

fn hash_corpus_identity_cells(
    scale: MatrixScale,
    cells: &[PerfCellSpec],
    qg1_identities: &Qg1PrefixIdentityMap,
) -> Result<String, String> {
    let has_qg1 = cells.iter().any(|cell| cell.gate == PerfGate::Qg1);

    let mut hasher = Sha256::new();
    if has_qg1 {
        hasher.update(b"frankensearch-quill-qg1-full-corpus-identity-v1\0");
        hasher.update(PerfMatrixSpec::QG1_CANONICAL_SHA256.as_bytes());
        hasher.update(
            u64::try_from(cells.len())
                .map_err(|_| "QG-1 corpus identity cell count is not representable".to_owned())?
                .to_le_bytes(),
        );
    }
    for cell in cells {
        let requested = cell.document_count.unwrap_or_default();
        let effective = scale.document_count(requested);
        if has_qg1 {
            hasher.update(b"\0corpus-identity-cell-v1\0");
            hash_qg1_indexed_bytes(&mut hasher, cell.fixture.as_bytes());
        } else {
            // Preserve the established identity contract for every non-QG-1
            // gate; this framing revision belongs only to QG-1's new frozen
            // full-universe identity domain.
            hasher.update(cell.fixture.as_bytes());
        }
        hasher.update(effective.to_le_bytes());
        hasher.update(CORPUS_SEED.to_le_bytes());
        hasher.update(VOCABULARY_SIZE.to_le_bytes());
        hasher.update(MAX_DOCUMENT_BYTES.to_le_bytes());
        if cell.gate == PerfGate::Qg1 {
            let cell_contract_sha256 = cell
                .contract_sha256()
                .map_err(|error| format!("hash authoritative QG-1 cell contract: {error}"))?;
            hasher.update(b"\0canonical-qg1-cell-contract-v1\0");
            hasher.update(cell_contract_sha256.as_bytes());
            let (manifest_sha256, indexed_content_sha256) = qg1_identities
                .get(&effective)
                .ok_or_else(|| format!("QG-1 corpus identity {effective} was not verified"))?;
            hasher.update(b"\0prepared-qg1-corpus-manifest-v2\0");
            hasher.update(manifest_sha256.as_bytes());
            hasher.update(b"\0prepared-qg1-indexed-content-v1\0");
            hasher.update(indexed_content_sha256.as_bytes());
        }
    }
    Ok(lower_hex(&hasher.finalize()))
}

fn corpus_manifest_hash(context: &BenchContext, cells: &[PerfCellSpec]) -> Result<String, String> {
    let qg1_identities = prepared_qg1_prefix_identities(context, cells)?;
    hash_corpus_identity_cells(context.scale, cells, &qg1_identities)
}

fn authoritative_qg1_corpus_identity(
    context: &BenchContext,
    matrix: &PerfMatrixSpec,
    selected: &[PerfCellSpec],
) -> Result<(String, Vec<PerfCellSpec>), String> {
    let canonical_sha256 = matrix
        .gate_contract_sha256(PerfGate::Qg1)
        .map_err(|error| format!("hash canonical QG-1 matrix: {error}"))?;
    if canonical_sha256 != PerfMatrixSpec::QG1_CANONICAL_SHA256 {
        return Err(format!(
            "canonical QG-1 matrix identity {canonical_sha256} differs from frozen identity {}",
            PerfMatrixSpec::QG1_CANONICAL_SHA256
        ));
    }
    let canonical_cells = matrix
        .for_gate(PerfGate::Qg1)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    if selected.is_empty() || selected.iter().any(|cell| cell.gate != PerfGate::Qg1) {
        return Err("authoritative QG-1 identity requires a nonempty QG-1 selection".to_owned());
    }

    let authoritative = if context.scale.is_full() {
        frozen_qg1_full_prefix_identities(&canonical_cells)?
    } else {
        replay_qg1_prefix_identities(context.scale, &canonical_cells)?
    };
    validate_selected_qg1_prefixes(context, selected, &authoritative)?;
    let corpus_sha256 =
        hash_corpus_identity_cells(context.scale, &canonical_cells, &authoritative)?;
    Ok((corpus_sha256, canonical_cells))
}

fn lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn build_profile_label(scale: MatrixScale) -> String {
    match std::env::var("QUILL_PERF_BUILD_PROFILE") {
        Ok(profile) if !profile.trim().is_empty() => profile,
        Ok(_) => panic!("QUILL_PERF_BUILD_PROFILE must not be empty"),
        Err(error) => {
            assert!(
                !scale.is_full(),
                "full QG evidence requires QUILL_PERF_BUILD_PROFILE: {error}"
            );
            if cfg!(debug_assertions) {
                "dev-unspecified".to_owned()
            } else {
                "release-unspecified".to_owned()
            }
        }
    }
}

fn command_sha256() -> String {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    command_sha256_from_argv(arguments.iter().map(|argument| argument.as_encoded_bytes()))
}

fn environment_sha256(scale: MatrixScale) -> Option<String> {
    match std::env::var("QUILL_PERF_ENVIRONMENT_SHA256") {
        Ok(value)
            if value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) =>
        {
            Some(value)
        }
        Ok(_) => panic!("QUILL_PERF_ENVIRONMENT_SHA256 must be lowercase SHA-256"),
        Err(error) => {
            assert!(
                !scale.is_full(),
                "full QG evidence requires QUILL_PERF_ENVIRONMENT_SHA256: {error}"
            );
            None
        }
    }
}

fn build_identity(bench_elf_sha256: &str, revision: &str, build_profile: &str) -> BuildIdentity {
    let typed_producer = std::env::var("QUILL_PERF_TYPED_PRODUCER").as_deref() == Ok("1");
    let porcelain = if typed_producer {
        Some(String::new())
    } else {
        Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
    };
    let git_dirty = porcelain
        .as_deref()
        .is_some_and(|status| !status.trim().is_empty());
    let worktree_state_sha256 = git_dirty.then(|| {
        let diff = Command::new("git")
            .args(["diff", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| output.stdout)
            .unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(porcelain.as_deref().unwrap_or_default().as_bytes());
        hasher.update(&diff);
        lower_hex(&hasher.finalize())
    });
    let cargo_lock_sha256 = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
        .ok()
        .map(|bytes| lower_hex(&Sha256::digest(&bytes)));
    let rustc = std::env::var_os("QUILL_PERF_RUSTC").unwrap_or_else(|| "rustc".into());
    let rustc_verbose = Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_else(|| "unavailable".to_owned());
    let rustc_version = rustc_verbose
        .lines()
        .next()
        .unwrap_or("unavailable")
        .to_owned();
    let target_triple = rustc_verbose
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map_or_else(
            || format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            str::to_owned,
        );
    BuildIdentity {
        executable_sha256: bench_elf_sha256.to_owned(),
        git_revision: revision.to_owned(),
        git_dirty,
        worktree_state_sha256,
        cargo_lock_sha256,
        command_sha256: command_sha256(),
        environment_sha256: environment_sha256(MatrixScale::from_env()),
        rustc_version,
        target_triple,
        build_profile: build_profile.to_owned(),
        cargo_features: vec!["perf-harness".to_owned(), "tantivy-oracle".to_owned()],
    }
}

fn corpus_identity(
    context: &BenchContext,
    cells: &[PerfCellSpec],
    corpus_hash: &str,
) -> CorpusIdentity {
    let qg1_native_identity = cells.iter().all(|cell| cell.gate == PerfGate::Qg1);
    let document_count = cells
        .iter()
        .map(|cell| {
            context
                .scale
                .document_count(cell.document_count.unwrap_or_default())
        })
        .max()
        .unwrap_or_default();
    let query_set_sha256 = (!qg1_native_identity).then(|| {
        Qg6QuerySpec::normative_manifest_sha256()
            .expect("frozen 80-query manifest validates before evidence identity")
    });
    CorpusIdentity {
        corpus_sha256: corpus_hash.to_owned(),
        query_set_sha256,
        qrels_sha256: None,
        document_count,
        content_bytes: None,
        generator_seed: CORPUS_SEED,
        generator_revision: if qg1_native_identity {
            QG1_CORPUS_GENERATOR_REVISION.to_owned()
        } else {
            Qg6QuerySpec::normative_corpus_generator_revision().to_owned()
        },
    }
}

fn metric_duration(context: &BenchContext, spec: &PerfCellSpec, value: f64) -> Duration {
    let seconds = match spec.metric.as_str() {
        "docs_per_second" | "tokenize_docs_per_second" => {
            context
                .scale
                .document_count(spec.document_count.expect("throughput document count"))
                as f64
                / value.max(f64::MIN_POSITIVE)
        }
        "updates_per_second" => {
            context
                .scale
                .document_count(spec.document_count.expect("update count")) as f64
                / value.max(f64::MIN_POSITIVE)
        }
        "commit_latency_ms"
        | "latency_ms"
        | "open_latency_ms"
        | "update_to_searchable_ms"
        | "wall_clock_ms" => value / 1_000.0,
        _ => 0.0,
    };
    Duration::from_secs_f64(seconds.max(0.0))
}

fn register_criterion_cell(c: &mut Criterion, context: &BenchContext, spec: &PerfCellSpec) {
    // Full-scale invocations already collect and persist the normative paired
    // blocks above. Registering a second Criterion timing scope would repeat
    // the workload after the decision artifact is sealed, without retaining
    // those samples or their A/A control. Keep Criterion's presentation lane
    // for smoke runs only.
    if context.scale.is_full()
        || matches!(spec.gate, PerfGate::Qg6 | PerfGate::Qg7 | PerfGate::Qg10)
    {
        return;
    }
    let mut group = c.benchmark_group(format!("quill_perf/{}/{}", spec.gate, spec.fixture));
    group.sample_size(PERF_MIN_RUNS);
    for arm in [EngineArm::Tantivy, EngineArm::Quill] {
        group.bench_function(arm.label(), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    let measurement = black_box(measure_metric(context, spec, arm));
                    total += metric_duration(context, spec, measurement.value);
                }
                total
            });
        });
    }
    group.finish();
}

fn output_dir() -> PathBuf {
    std::env::var_os("QUILL_PERF_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| scratch_path("artifacts"))
}

fn evidence_policy_from_env() -> EvidencePolicy {
    let mut policy = EvidencePolicy::predeclared();
    policy.warmup_rounds = match std::env::var("QUILL_PERF_WARMUP_ROUNDS") {
        Ok(value) => value.parse::<usize>().unwrap_or_else(|error| {
            panic!("QUILL_PERF_WARMUP_ROUNDS must be a positive integer: {error}")
        }),
        Err(std::env::VarError::NotPresent) => policy.warmup_rounds,
        Err(error) => panic!("QUILL_PERF_WARMUP_ROUNDS is not valid Unicode: {error}"),
    };
    assert!(
        policy.warmup_rounds > 0,
        "QUILL_PERF_WARMUP_ROUNDS must preserve at least one excluded warmup"
    );
    policy
}

fn bench_matrix(c: &mut Criterion, bench_identity: &BenchExecutableIdentity) {
    let bench_elf_sha256 = &bench_identity.sha256;
    let scale = MatrixScale::from_env();
    let build_profile = build_profile_label(scale);
    let matrix = PerfMatrixSpec::complete();
    validate_matrix(&matrix).expect("normative QG matrix");
    let claims = RunnerPlanClaims::from_env()
        .unwrap_or_else(|error| panic!("typed runner applicability claims rejected: {error}"));
    let runner = RunnerApplicabilityContext::reconstruct(&matrix, &claims)
        .unwrap_or_else(|error| panic!("typed runner applicability plan rejected: {error}"));
    let fixture_filter = fixture_filter_from_env()
        .unwrap_or_else(|error| panic!("typed runner fixture filter rejected: {error}"));
    let selected = selected_cells(&matrix, &runner, scale, fixture_filter.as_deref())
        .unwrap_or_else(|error| panic!("typed runner cell selection rejected: {error}"));
    let selection_complete =
        gate_selection_complete(&runner, &selected, scale, fixture_filter.as_deref());
    let selected_specs = selected
        .iter()
        .map(|cell| cell.spec.clone())
        .collect::<Vec<_>>();
    let context = BenchContext::for_selected(scale, &selected_specs);
    let configured_runs = std::env::var("QUILL_PERF_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            if scale.is_full() && selected.iter().any(|cell| cell.spec.gate == PerfGate::Qg4) {
                100
            } else {
                PERF_MIN_RUNS
            }
        });
    assert!(
        configured_runs >= PERF_MIN_RUNS,
        "QUILL_PERF_RUNS must preserve the >=10-run law"
    );

    let output_dir = output_dir();
    let revision = git_revision(scale);
    let run_window = std::env::var("QUILL_PERF_RUN_WINDOW")
        .unwrap_or_else(|_| format!("manual-window-{}", std::process::id()));
    let run_id = std::env::var("QUILL_PERF_RUN_ID")
        .unwrap_or_else(|_| format!("manual-pass-{}", std::process::id()));
    let manifest_hash = manifest_sha256();
    let corpus_hash = if runner.plan.binding().gate == PerfGate::Qg1 {
        authoritative_qg1_corpus_identity(&context, &matrix, &selected_specs)
            .expect("verify authoritative immutable full QG-1 corpus identity")
            .0
    } else {
        corpus_manifest_hash(&context, &selected_specs)
            .expect("verify exact selected corpus identity")
    };
    let bootstrap_seed = std::env::var("QUILL_PERF_BOOTSTRAP_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0x5155_494c_4c45_5644);
    let evidence_policy = evidence_policy_from_env();
    eprintln!(
        "[quill-perf-policy] warmup_rounds={}",
        evidence_policy.warmup_rounds
    );
    let evidence_context = EvidenceContext {
        config: PairedEstimatorConfig::predeclared(bootstrap_seed),
        policy: evidence_policy,
        sample_provenance: PerfSampleProvenance {
            run_id: run_id.clone(),
            executable_sha256: bench_elf_sha256.to_owned(),
            corpus_sha256: corpus_hash.clone(),
            input_identity: None,
            worker_id: machine_fingerprint(),
            build_profile: build_profile.clone(),
        },
    };
    let configured_widths = configured_engine_widths(&selected);
    // The selected-cell order is the canonical startup transcript order.  All
    // engine producers are frozen and parent-acknowledged before preflight can
    // perform indexing work; tokenizer-only QG-1 diagnostics contribute none.
    let qg1_startup_producers = construct_qg1_startup_producers(
        &context,
        &selected_specs,
        configured_runs,
        &evidence_context,
        runner.profile,
        usize::try_from(runner.execution_capacity).expect("QG-1 execution capacity fits usize"),
        &configured_widths,
    );
    let selected_qg1 = runner.plan.binding().gate == PerfGate::Qg1;
    require_qg1_pre_timing_authority_ack(selected_qg1, &qg1_startup_producers.engine_cells);
    if qg1_exact_startup_handshake_for_selected_gate() {
        assert!(
            selected_qg1,
            "the exact QG-1 startup handshake cannot be active for a non-QG selection"
        );
        emit_bench_elf_sha256(bench_identity);
    }
    if qg1_live_startup_discriminator_mode() == Some(Qg1LiveStartupDiscriminatorMode::NonQg) {
        assert!(
            !selected_qg1 && !qg1_exact_startup_handshake_for_selected_gate(),
            "the non-QG discriminator must keep the QG-1 stdio handshake inactive"
        );
        return;
    }
    if qg1_live_startup_discriminator_mode() == Some(Qg1LiveStartupDiscriminatorMode::Child) {
        assert!(
            selected_qg1 && qg1_exact_startup_handshake_for_selected_gate(),
            "the live QG-1 startup discriminator must reach the normal selected-gate barrier"
        );
        std::io::stdout()
            .write_all(QG1_LIVE_STARTUP_ORDINARY_MARKER)
            .expect("emit live QG-1 post-ACK ordinary marker");
        std::io::stdout()
            .flush()
            .expect("flush live QG-1 post-ACK ordinary marker");
        return;
    }
    preflight_indexing_fixtures(&context, &matrix, &selected_specs);
    let mut machine = MachineIdentity::capture(
        runner.execution_capacity,
        runner.max_exercised_cell_width,
        configured_widths.iter().copied(),
    );
    eprintln!(
        "[quill-perf-execution-provenance] {}",
        serde_json::to_string(&machine.execution).expect("serialize execution provenance")
    );

    let mut by_gate: BTreeMap<PerfGate, Vec<PerfCellResult>> = BTreeMap::new();
    let mut evidence_by_gate: BTreeMap<PerfGate, Vec<EvidenceCell>> = BTreeMap::new();
    let mut qg1_incumbent_evidence = BTreeMap::new();
    for planned in &selected {
        let spec = &planned.spec;
        let collection = collect_cell(
            &context,
            spec,
            planned.role,
            configured_runs,
            &evidence_context,
            qg1_startup_producers.for_spec(spec),
        );
        let CellCollection {
            results,
            evidence,
            qg1_incumbent_screen,
            qg1_incumbent_decision,
        } = collection;
        by_gate.entry(spec.gate).or_default().extend(results);
        if let Some(cell) = evidence {
            evidence_by_gate.entry(spec.gate).or_default().push(cell);
        }
        if let Some(screen) = qg1_incumbent_screen {
            let semantic_contract = screen
                .candidates
                .first()
                .expect("QG-1 screen retains its candidate universe")
                .semantic_contract
                .clone();
            let cell_id = format!("{}/{}/{}", spec.gate, spec.fixture, spec.metric);
            let previous = qg1_incumbent_evidence.insert(
                cell_id.clone(),
                Qg1IncumbentScreenEvidence {
                    cell_id,
                    semantic_contract,
                    screen,
                    decision: qg1_incumbent_decision,
                },
            );
            assert!(
                previous.is_none(),
                "every required QG-1 engine cell must contribute exactly one screen"
            );
        } else {
            assert!(qg1_incumbent_decision.is_none());
        }
        register_criterion_cell(c, &context, spec);
    }
    machine.finish();
    flush_tantivy_lifecycle_receipts(&output_dir);
    flush_qg1_continuous_timing_receipts(&output_dir);

    let provenance = EvidenceProvenance {
        run_id: run_id.clone(),
        run_window: run_window.clone(),
        manifest_sha256: manifest_hash.clone(),
        build: build_identity(bench_elf_sha256, &revision, &build_profile),
        machine: machine.clone(),
        peak_rss: PeakRssEvidence::capture(),
        corpus: corpus_identity(&context, &selected_specs, &corpus_hash),
    };
    let applicability_binding = runner.plan.binding().clone();
    for (gate, cells) in evidence_by_gate {
        let mut artifact = PerfEvidenceArtifact::assemble(
            gate,
            applicability_binding.clone(),
            evidence_context.policy.clone(),
            provenance.clone(),
            cells,
        )
        .expect("assemble QG evidence artifact");
        let retained_qg1_authorities = qg1_startup_producers.retained_authorities();
        let authority_bound_qg1 = gate == PerfGate::Qg1 && !retained_qg1_authorities.is_empty();
        let incumbent_screens = if gate == PerfGate::Qg1 {
            std::mem::take(&mut qg1_incumbent_evidence)
                .into_values()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let required_screen_cell_ids = artifact
            .cells
            .iter()
            .filter(|cell| {
                cell.spec.gate == PerfGate::Qg1 && cell.spec.role == EvidenceRole::Required
            })
            .map(|cell| cell.cell_id.clone())
            .collect::<BTreeSet<_>>();
        let observed_screen_cell_ids = incumbent_screens
            .iter()
            .map(|screen| screen.cell_id.clone())
            .collect::<BTreeSet<_>>();
        let qg1_screen_coverage_exact = gate == PerfGate::Qg1
            && selection_complete
            && !required_screen_cell_ids.is_empty()
            && incumbent_screens.len() == observed_screen_cell_ids.len()
            && observed_screen_cell_ids == required_screen_cell_ids;
        let attach_qg1_screens =
            qg1_screen_coverage_exact && authority_bound_qg1 && !incumbent_screens.is_empty();
        if attach_qg1_screens {
            artifact
                .attach_qg1_incumbent_screens(incumbent_screens)
                .expect("attach complete live QG-1 incumbent screens to durable evidence");
        }
        let selection_no_claim = partial_shard_no_claim(gate, selection_complete);
        let incumbent_no_claim = (gate == PerfGate::Qg1 && !attach_qg1_screens).then(|| {
            (
                QG1_INCUMBENT_SCREEN_NO_CLAIM_CODE,
                if required_screen_cell_ids.is_empty() {
                    "the QG-1 selection contains no required engine lifecycle cells; tokenizer-only evidence cannot claim"
                        .to_owned()
                } else if !authority_bound_qg1 {
                    "QG-1 incumbent screens have no retained engine authority set and were omitted"
                        .to_owned()
                } else {
                    format!(
                        "QG-1 incumbent screen coverage is partial; expected {required_screen_cell_ids:?}, observed {observed_screen_cell_ids:?}"
                    )
                },
            )
        });
        let terminal_no_claim = (gate == PerfGate::Qg1)
            .then(qg1_terminal_no_claim_detail)
            .flatten();
        let mut no_claim_reasons = Vec::new();
        if let Some((code, detail)) = selection_no_claim {
            no_claim_reasons.push((code, detail.to_owned()));
        }
        if let Some(reason) = incumbent_no_claim {
            no_claim_reasons.push(reason);
        }
        if let Some(detail) = terminal_no_claim {
            no_claim_reasons.push((QG1_TERMINAL_NO_CLAIM_CODE, detail));
        }
        if let Some((code, _)) = no_claim_reasons.first() {
            artifact.force_no_claim(
                code,
                no_claim_reasons
                    .iter()
                    .map(|(_, detail)| detail.as_str())
                    .collect::<Vec<_>>()
                    .join("; additionally, "),
            );
        }
        if gate == PerfGate::Qg1 && artifact.qg1_incumbent_screens.is_empty() {
            assert!(
                !artifact.ratchet_admissible(),
                "screenless partial or tokenizer-only QG-1 evidence cannot claim"
            );
        }
        let (paths, reloaded) = if authority_bound_qg1 {
            let paths = artifact
                .write_atomic_against_qg1_authorities(&output_dir, &retained_qg1_authorities)
                .expect("write authority-bound QG-1 evidence artifact");
            let reloaded = PerfEvidenceArtifact::load_verified_against_qg1_authorities(
                &paths.json,
                &retained_qg1_authorities,
            )
            .expect("reload QG-1 incumbent evidence with the retained authority set");
            (paths, reloaded)
        } else {
            let paths = artifact
                .write_atomic(&output_dir)
                .expect("write QG evidence artifact");
            let reloaded = PerfEvidenceArtifact::load_verified(&paths.json)
                .expect("reload and verify persisted QG evidence artifact");
            (paths, reloaded)
        };
        let mut normalized_reloaded = reloaded.clone();
        normalized_reloaded.artifact_sha256 = artifact.artifact_sha256.clone();
        assert_eq!(
            normalized_reloaded, artifact,
            "persisted QG evidence must match every source field except its writer-computed seal"
        );
        black_box(reloaded);
        eprintln!(
            "[quill-evidence] gate={gate} status={} ratchet_admissible={} json={} table={}",
            artifact.gate_status,
            artifact.ratchet_admissible(),
            display_path(&paths.json),
            display_path(&paths.table),
        );
    }
    for (gate, cells) in by_gate {
        let artifact = PerfGateArtifact {
            schema_version: PERF_ARTIFACT_SCHEMA_VERSION.to_owned(),
            gate,
            applicability_plan: Some(applicability_binding.clone()),
            bench_elf_sha256: bench_elf_sha256.to_owned(),
            machine_fingerprint: machine_fingerprint(),
            execution: Some(machine.execution.clone()),
            git_rev: revision.clone(),
            run_window: run_window.clone(),
            run_id: run_id.clone(),
            corpus_manifest_hash: corpus_hash.clone(),
            manifest_sha256: manifest_hash.clone(),
            cells,
            laws_attested: selection_complete,
        };
        let (json, table) = artifact.write_to(&output_dir).expect("write QG artifacts");
        eprintln!("{}", artifact.human_table());
        eprintln!("[quill-perf-json-begin] gate={gate}");
        eprintln!(
            "{}",
            artifact
                .to_json_pretty()
                .expect("serialize QG artifact for remote retrieval")
        );
        eprintln!("[quill-perf-json-end] gate={gate}");
        eprintln!(
            "[quill-perf] gate={gate} json={} table={}",
            display_path(&json),
            display_path(&table)
        );
    }
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn child_env<T>(name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    std::env::var(name)
        .unwrap_or_else(|_| panic!("missing {name}"))
        .parse::<T>()
        .unwrap_or_else(|error| panic!("invalid {name}: {error:?}"))
}

fn child_engine() -> EngineArm {
    match std::env::var("QUILL_PERF_CHILD_ENGINE").as_deref() {
        Ok("quill") => EngineArm::Quill,
        Ok("tantivy") => EngineArm::Tantivy,
        value => panic!("invalid QUILL_PERF_CHILD_ENGINE: {value:?}"),
    }
}

fn run_search_child() {
    let arm = child_engine();
    let path = PathBuf::from(
        std::env::var_os("QUILL_PERF_CHILD_PATH").expect("missing QUILL_PERF_CHILD_PATH"),
    );
    let heap = child_env::<usize>("QUILL_PERF_CHILD_HEAP");
    let threads = child_env::<usize>("QUILL_PERF_CHILD_THREADS");
    let positions = child_env::<bool>("QUILL_PERF_CHILD_POSITIONS");
    let query = std::env::var("QUILL_PERF_CHILD_QUERY").expect("missing QG-3 child query");
    let expected_doc_id =
        std::env::var("QUILL_PERF_CHILD_EXPECTED_DOC_ID").expect("missing QG-3 expected doc ID");
    let context = BenchContext::new(MatrixScale::from_env());
    let doc_ids = match arm {
        EngineArm::Quill => {
            let index = context
                .runtime
                .block_on(QuillIndex::open(
                    &context.cx,
                    &path,
                    pinned_quill_config(heap, threads),
                ))
                .expect("fresh-process Quill open");
            context.runtime.block_on(async {
                index
                    .search(&context.cx, &query, 3)
                    .await
                    .expect("fresh-process Quill query")
                    .into_iter()
                    .map(|result| result.doc_id.into())
                    .collect::<Vec<String>>()
            })
        }
        EngineArm::Tantivy => {
            let index = TantivyIndex::open_with_benchmark_config(&path, heap, threads, positions)
                .expect("fresh-process Tantivy open");
            context.runtime.block_on(async {
                index
                    .search(&context.cx, &query, 3)
                    .await
                    .expect("fresh-process Tantivy query")
                    .into_iter()
                    .map(|result| result.doc_id.into())
                    .collect::<Vec<String>>()
            })
        }
    };
    assert_eq!(
        doc_ids,
        [expected_doc_id],
        "fresh-process QG-3 visibility fence accepted stale, missing, or ambiguous state"
    );
    println!("quill-perf-child\t{}", doc_ids.len());
}

fn run_open_child() {
    let arm = child_engine();
    let path = PathBuf::from(
        std::env::var_os("QUILL_PERF_CHILD_PATH").expect("missing QUILL_PERF_CHILD_PATH"),
    );
    let heap = child_env::<usize>("QUILL_PERF_CHILD_HEAP");
    let threads = child_env::<usize>("QUILL_PERF_CHILD_THREADS");
    let positions = child_env::<bool>("QUILL_PERF_CHILD_POSITIONS");
    let context = BenchContext::new(MatrixScale::from_env());
    let timer = Instant::now();
    match arm {
        EngineArm::Quill => {
            black_box(
                context
                    .runtime
                    .block_on(QuillIndex::open(
                        &context.cx,
                        &path,
                        pinned_quill_config(heap, threads),
                    ))
                    .expect("fresh-process QG-9 Quill open"),
            );
        }
        EngineArm::Tantivy => {
            black_box(
                TantivyIndex::open_with_benchmark_config(&path, heap, threads, positions)
                    .expect("fresh-process QG-9 pinned Tantivy open"),
            );
        }
    }
    println!("quill-perf-child\t{}", timer.elapsed().as_nanos());
}

fn run_memory_child() {
    let arm = child_engine();
    let count = child_env::<u64>("QUILL_PERF_CHILD_COUNT");
    let heap = child_env::<usize>("QUILL_PERF_CHILD_HEAP");
    let threads = child_env::<usize>("QUILL_PERF_CHILD_THREADS");
    let positions = child_env::<bool>("QUILL_PERF_CHILD_POSITIONS");
    let context = BenchContext::new(MatrixScale::from_env());
    let corpus = corpus_for(count);
    let index_bytes = match arm {
        EngineArm::Quill => {
            let config = pinned_quill_config(heap, threads);
            let index = if positions {
                QuillIndex::in_memory(config).expect("RSS Quill index")
            } else {
                QuillIndex::in_memory_with_schema(NO_POSITION_SCHEMA, config)
                    .expect("RSS position-free Quill index")
            };
            let _ = index_batches(&context, &index, &corpus, count, None);
            let _ = commit(&context, &index);
            // bd-enf6z: managed_disk_bytes short-circuits to 0 when the keeper
            // has no directory, so the in-memory Quill arm reported 0 index
            // bytes while the Tantivy arm reported real file lengths — a
            // structural bias in Quill's favor. The manifest's per-segment
            // file_len records the exact FSLX byte image and is populated for
            // in-memory indexes too.
            let bytes: u64 = index
                .snapshot()
                .expect("benchmark snapshot is authoritative")
                .loaded_manifest()
                .manifest
                .segments
                .iter()
                .map(|segment| segment.file_len)
                .sum();
            assert!(
                bytes > 0,
                "QG-7 Quill arm measured a zero-byte index footprint for a \
                 committed non-empty corpus; the footprint seam is broken \
                 (bd-enf6z would regress)"
            );
            let rss = peak_rss_bytes().unwrap_or_default();
            println!("quill-perf-child\t{rss}\t{bytes}");
            return;
        }
        EngineArm::Tantivy => {
            let index = TantivyIndex::in_memory_with_benchmark_config(heap, threads, positions)
                .expect("RSS Tantivy index");
            let _ = index_batches(&context, &index, &corpus, count, None);
            let _ = commit(&context, &index);
            index
                .benchmark_index_layout()
                .expect("RSS Tantivy index layout")
                .1
        }
    };
    let rss = peak_rss_bytes().unwrap_or_default();
    println!("quill-perf-child\t{rss}\t{index_bytes}");
}

fn run_child_mode() -> bool {
    match std::env::var("QUILL_PERF_CHILD_MODE").as_deref() {
        Ok("search") => run_search_child(),
        Ok("open") => run_open_child(),
        Ok("memory") => run_memory_child(),
        Ok(mode) => panic!("unknown QUILL_PERF_CHILD_MODE {mode:?}"),
        Err(_) => return false,
    }
    true
}

/// Prove at RUNTIME that the incumbent arm is the real Tantivy, not us.
///
/// This is the dispatch trap, and it is not hypothetical: a peer repo published
/// a "2.6x faster" claim whose baseline had already been dispatched to its own
/// implementation — genuine upstream was 1.88x *slower*. This repo is mid-
/// migration to Quill, so `frankensearch-lexical` being quietly re-pointed at a
/// Quill backend is exactly the shape that failure would take here, and every
/// QG ratio would silently become Quill measuring itself.
///
/// A manifest pin alone cannot catch that — it describes what Cargo resolved,
/// not what the process linked. So this asserts both: the pinned contract
/// (version, registry source, checksum, lexical package + audited contract
/// revision, `tantivy = "=0.26.1"`)
/// *and* the version string the linked Tantivy reports about itself at run
/// time. Printed so it lands in the evidence log beside the ratios.
fn assert_incumbent_is_genuine_tantivy() -> String {
    let contract = oracle_version_contract().expect(
        "QG oracle version contract must validate before any ratio is measured (dispatch trap)",
    );
    let linked = frankensearch_lexical::tantivy_crate::version_string();
    assert!(
        linked.contains(&contract.tantivy_version),
        "incumbent arm is not the pinned Tantivy: linked runtime reports {linked:?} but the \
         oracle contract pins {:?}. Refusing to measure — a QG ratio against a non-incumbent \
         baseline is worthless.",
        contract.tantivy_version,
    );
    eprintln!(
        "[quill-perf-oracle] incumbent=tantivy linked_runtime={linked} contract_version={} \
         source={} checksum={} lexical={}@{} lexical_contract_audit={}",
        contract.tantivy_version,
        contract.tantivy_source,
        contract.tantivy_checksum_sha256,
        contract.lexical_package,
        contract.lexical_package_version,
        contract.lexical_contract_audit_revision,
    );
    linked.to_owned()
}

fn main() {
    validate_manifest_gate_contract(MANIFEST)
        .expect("normative QG manifest has a complete, closed policy set");
    if qg1_live_startup_discriminator_mode() == Some(Qg1LiveStartupDiscriminatorMode::Parent) {
        run_qg1_live_startup_discriminator_parent().unwrap_or_else(|error| {
            panic!("live harness=false QG-1 startup discriminator failed: {error}")
        });
        return;
    }
    #[cfg(test)]
    if std::env::var_os(QG1_AUTHORITY_SUBPROCESS_ENV).is_some() {
        // This marker remains a cfg(test) barrier unit seam only. It is not
        // evidence for normal harness=false stdout ordering; that proof is the
        // production parent/child discriminator above.
        tests::qg1_authority_subprocess_helper();
        return;
    }
    // QG-2's contract runs in the harness-false binary for the same reason
    // H1's does: a `#[test]` item here is stripped and never executes, so it
    // could never be evidence. This branch is what makes the QG-2 assertions
    // actually run under the central command.
    #[cfg(test)]
    if std::env::var_os("QUILL_PERF_QG2_SELF_CHECK").is_some() {
        qg2_continuous_tests::assert_qg2_continuous_interval_contract();
        qg2_continuous_tests::assert_qg2_summed_shape_excludes_the_planted_tail();
        return;
    }
    #[cfg(test)]
    if std::env::var_os("QUILL_PERF_H1_PRODUCER_SELF_CHECK").is_some() {
        tests::assert_qg1_continuous_interval_contract();
        tests::assert_qg1_throughput_semantics_contract();
        tests::assert_throughput_typing_is_not_cosmetic();
        tests::assert_qg1_real_terminal_visibility_contract();
        tests::assert_qg1_raw_sample_work_contract();
        tests::assert_qg1_disjoint_partial_shard_contract();
        tests::assert_corpus_identity_fixture_framing();
        tests::assert_non_qg1_corpus_identity_preserves_legacy_hash();
        tests::assert_qg9_cache_evidence_contract();
        tests::assert_qg9_cache_eviction_file_discovery();
        tests::assert_qg9_cache_eviction_request();
        tests::assert_manifest_gate_contract();
        tests::assert_qg1_authority_handshake_contract();
        eprintln!(
            "[quill-perf-self-check] H1 immutable producer and continuous-timing contracts passed"
        );
        return;
    }
    if run_child_mode() {
        return;
    }
    let identity = hash_bench_elf_sha256_silently().expect("hash executing QG benchmark");
    let qg1_exact_handshake = qg1_exact_startup_handshake_for_selected_gate();
    if qg1_live_startup_discriminator_mode() == Some(Qg1LiveStartupDiscriminatorMode::Preamble) {
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(b"planted-ordinary-qg1-startup-preamble\n")
            .expect("write planted ordinary QG-1 preamble");
        stdout
            .flush()
            .expect("flush planted ordinary QG-1 preamble");
    }
    if !qg1_exact_handshake {
        // Preserve the established line-one identity for non-QG and
        // non-handshake runs. Exact QG-1 reserves stdout until final ACK.
        emit_bench_elf_sha256(&identity);
    }
    // Fail closed before a single cell is timed.
    let _oracle = assert_incumbent_is_genuine_tantivy();
    let mut criterion = Criterion::default().configure_from_args();
    bench_matrix(&mut criterion, &identity);
    criterion.final_summary();
}

#[cfg(test)]
mod qg2_continuous_tests {
    /// The smallest normative QG-2 update cell, taken from the frozen matrix so
    /// the test cannot drift onto a shape the manifest does not ship.
    fn qg2_spec() -> frankensearch_quill_gauntlet::PerfCellSpec {
        frankensearch_quill_gauntlet::PerfMatrixSpec::complete()
            .for_gate(frankensearch_quill_gauntlet::PerfGate::Qg2)
            .into_iter()
            .find(|spec| spec.metric == "docs_per_second")
            .cloned()
            .expect("the frozen matrix ships a QG-2 throughput cell")
    }

    /// THE PRODUCTION PATH, both arms: one continuous interval that ends at a
    /// searchable, quiescent tail, carrying the work it processed.
    ///
    /// The endpoint assertions live inside `qg2_bulk_metric_continuous` itself,
    /// so a regression that stops at commit — before the update is searchable —
    /// fails there rather than being quietly reported as a faster number here.
    /// Ordinary `cfg(test)` helper, reached from `main` under
    /// `QUILL_PERF_QG2_SELF_CHECK`. A `#[test]` item here would be stripped by
    /// `harness = false` and would therefore never execute, which is not
    /// evidence of anything.
    pub(super) fn assert_qg2_continuous_interval_contract() {
        let spec = qg2_spec();
        let context = super::BenchContext::for_selected(
            super::MatrixScale::Smoke,
            std::slice::from_ref(&spec),
        );
        let count = context
            .scale
            .document_count(spec.document_count.expect("QG-2 document count"));
        assert_eq!(
            super::metric_semantics(&spec),
            frankensearch_quill_gauntlet::PerfMetricSemantics::Throughput,
            "a QG-2 update cell publishes native throughput, not a gauge"
        );
        for arm in [super::EngineArm::Quill, super::EngineArm::Tantivy] {
            let measurement = super::qg2_bulk_metric_continuous(&context, &spec, arm, count);
            assert!(
                measurement.continuous.is_none(),
                "{arm:?} QG-2 interval must not present itself as QG-1 lifecycle evidence"
            );
            let interval = measurement
                .qg2_continuous
                .as_ref()
                .expect("every QG-2 arm publishes its continuous interval");
            assert_eq!(
                interval.work_units, count,
                "{arm:?} QG-2 interval must cover the exact requested document count"
            );
            assert!(
                interval.elapsed_ns > 0,
                "{arm:?} QG-2 interval must span positive monotonic time"
            );
            assert!(
                measurement.value.is_finite() && measurement.value > 0.0,
                "{arm:?} QG-2 cell must return positive finite throughput"
            );
            // Quiescence is asserted on its own basis, not inferred from the
            // tail search succeeding: visibility and a settled writer side are
            // different claims, and only the second one licenses ending the
            // interval here.
            match interval.quiescence {
                super::Qg2QuiescenceBasis::QuillPublishedGeneration { delta } => {
                    assert_eq!(
                        arm,
                        super::EngineArm::Quill,
                        "the published-generation basis belongs to the Quill arm"
                    );
                    assert!(
                        delta > 0,
                        "Quill quiescence requires the terminal commit to have published"
                    );
                }
                super::Qg2QuiescenceBasis::TantivyWorkersJoined { rearmed } => {
                    assert_eq!(
                        arm,
                        super::EngineArm::Tantivy,
                        "the joined-workers basis belongs to the Tantivy arm"
                    );
                    assert!(
                        !rearmed,
                        "Tantivy quiescence requires the terminal join to leave no replacement \
                         writer armed"
                    );
                }
            }
            assert!(
                interval.elapsed_ns > interval.feed_and_commit_ns,
                "{arm:?} unplanted interval must still cover more than its summed calls"
            );

            // DOWNSTREAM CONSUMPTION, asserted exactly rather than assumed. The
            // denominator the runner would publish must be the work this
            // interval processed, and the window the estimator would recompute
            // over must be the interval itself — not the outer call.
            let (declared_work, declared_bytes) = super::raw_sample_work(&context, &spec);
            assert_eq!(
                declared_work,
                Some(interval.work_units),
                "{arm:?} raw sample must declare the work its continuous interval processed"
            );
            assert_eq!(
                declared_bytes, None,
                "{arm:?} QG-2 has no prepared content-byte binding to declare"
            );
            let offsets = super::Qg1IntervalOffsets {
                work_units: interval.work_units,
                started_ns: 0,
                elapsed_ns: interval.elapsed_ns,
            };
            let window = super::qg1_sample_window(
                super::metric_semantics(&spec),
                declared_work,
                0,
                interval.elapsed_ns,
                Some(offsets),
            )
            .expect("QG-2 throughput window must be publishable from its own interval");
            assert_eq!(
                window.ended_ns - window.started_ns,
                interval.elapsed_ns,
                "{arm:?} published window must be the continuous interval, not the outer call"
            );
            // The same window fails closed if the denominator is not the work
            // the interval measured, which is what stops a rate being attached
            // to time it was not measured over.
            assert!(
                super::qg1_sample_window(
                    super::metric_semantics(&spec),
                    Some(interval.work_units + 1),
                    0,
                    interval.elapsed_ns,
                    Some(offsets),
                )
                .is_err(),
                "{arm:?} a mismatched work denominator must refuse to publish"
            );
        }
    }

    /// PLANTED NEGATIVE: the summed-call shape this correction replaced, proved
    /// against a delay planted inside the endpoint that shape drops.
    ///
    /// Comparing two runs would prove nothing: a run covering MORE lifecycle can
    /// finish sooner when cache and scheduler state differ, so a
    /// cross-invocation `>` is flaky and says nothing about inclusion. Both
    /// shapes are therefore measured in ONE invocation, with a known interval
    /// planted after the commit returns and before the terminal
    /// searchable/quiescent endpoint. That interval is inside the continuous
    /// span by construction and outside the summed span by construction, so
    /// their difference is bounded below by the planted delay on a single
    /// monotonic clock.
    /// Ordinary `cfg(test)` helper, reached from `main` under the same
    /// environment switch as the contract above.
    pub(super) fn assert_qg2_summed_shape_excludes_the_planted_tail() {
        let spec = qg2_spec();
        let context = super::BenchContext::for_selected(
            super::MatrixScale::Smoke,
            std::slice::from_ref(&spec),
        );
        let count = context
            .scale
            .document_count(spec.document_count.expect("QG-2 document count"));
        const PLANTED_TAIL_DELAY: std::time::Duration = std::time::Duration::from_millis(150);
        let planted_ns =
            u64::try_from(PLANTED_TAIL_DELAY.as_nanos()).expect("planted delay fits u64 ns");

        for arm in [super::EngineArm::Quill, super::EngineArm::Tantivy] {
            let measurement = super::qg2_bulk_metric_continuous_with_planted_tail_delay(
                &context,
                &spec,
                arm,
                count,
                Some(PLANTED_TAIL_DELAY),
            );
            let interval = measurement
                .qg2_continuous
                .as_ref()
                .expect("the continuous path publishes its interval");

            // The retired shape stops when the commit returns, so it cannot
            // have observed one nanosecond of the planted tail.
            assert!(
                interval.elapsed_ns > interval.feed_and_commit_ns,
                "{arm:?} continuous interval must strictly exceed the summed feed and commit it \
                 contains"
            );
            let tail_ns = interval.elapsed_ns - interval.feed_and_commit_ns;
            assert!(
                tail_ns >= planted_ns,
                "{arm:?} continuous endpoint must include the planted {planted_ns}ns tail, but it \
                 covered only {tail_ns}ns beyond the summed calls, so the terminal \
                 searchable-and-quiescent state was not inside the measured span"
            );

            // The same fact as the consequence that matters, stated on this one
            // invocation's own numbers rather than across runs or engines.
            assert!(
                super::throughput_per_second(count, interval.feed_and_commit_ns)
                    > super::throughput_per_second(count, interval.elapsed_ns),
                "{arm:?} summed shape must report the faster rate it obtains by dropping the tail"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    // A child module inherits none of the root's imports, so every bare name
    // below would otherwise have to be re-imported per function. Bringing the
    // parent scope in is the module-structure fix, not a lint silencer.
    use super::*;

    fn qg1_handshake_test_producer() -> frankensearch_quill_gauntlet::Qg1LifecycleProducer {
        use frankensearch_quill_gauntlet::{
            PairedEstimatorConfig, PerfMetricSemantics, PerfOperationScope, PerfSampleArm,
            Qg1BatchCoverage,
        };

        let mut config = PairedEstimatorConfig::predeclared(0x5147_3148_534b_5445);
        config
            .install_qg1_lifecycle_authority(
                PerfOperationScope {
                    operation_id: "QG-1.bulk/tiny/1/positions_on.docs_per_second".to_owned(),
                    version: 1,
                    semantics: PerfMetricSemantics::Throughput,
                    unit: "docs/s".to_owned(),
                },
                "a".repeat(64),
                "b".repeat(64),
                "c".repeat(64),
                1,
                1,
                1,
                vec![Qg1BatchCoverage {
                    document_start: 0,
                    document_count: 1,
                }],
                "synthetic-00000000".to_owned(),
                1,
                vec![
                    (
                        "qg1.effect.tantivy_vs_quill.v1".to_owned(),
                        0,
                        0,
                        vec![PerfSampleArm::Control],
                    ),
                    (
                        "qg1.null.tantivy.v1".to_owned(),
                        0,
                        1_000_000,
                        vec![PerfSampleArm::Control],
                    ),
                    (
                        "qg1.null.quill.v1".to_owned(),
                        2_000_000,
                        2_000_000,
                        vec![PerfSampleArm::Control],
                    ),
                ],
            )
            .expect("construct one real pre-timing QG-1 producer")
    }

    /// Parent-visible because `main` dispatches to it under the same
    /// `cfg(test)` barrier. A child module's private items are not visible to
    /// its parent, and the honest fix is to widen exactly this one helper
    /// rather than to hide the call site that needs it.
    ///
    /// Ordinary `cfg(test)` helper rather than a `#[test]` item: under
    /// `harness = false` a test-attribute item is stripped, so `main` could not
    /// call it at all.
    pub(super) fn qg1_authority_subprocess_helper() {
        if std::env::var_os(super::QG1_AUTHORITY_SUBPROCESS_ENV).is_none() {
            return;
        }
        let producer = qg1_handshake_test_producer();
        let startup = super::Qg1StartupProducer {
            operation_id: "QG-1.bulk/tiny/1/positions_on.docs_per_second".to_owned(),
            estimator_config: frankensearch_quill_gauntlet::PairedEstimatorConfig::predeclared(
                0x5147_3148_534b_5445,
            ),
            producer,
            incumbent: None,
        };
        super::require_qg1_pre_timing_authority_ack(true, &[startup]);
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(super::QG1_AUTHORITY_WORK_MARKER)
            .expect("emit work marker only after the production authority barrier");
        stdout.flush().expect("flush post-barrier work marker");
    }

    fn qg1_read_subprocess_startup_transcript(
        mut stdout: std::process::ChildStdout,
    ) -> Result<(u64, Vec<u8>, u64, std::process::ChildStdout), String> {
        let register = Qg1StartupHandshakeV1::read_control_frame(&mut stdout)?;
        let (sequence, entry_bytes) = match register {
            frankensearch_quill_gauntlet::Qg1StartupControlFrameV1::Register {
                sequence,
                entry_bytes,
            } => (sequence, entry_bytes),
            frankensearch_quill_gauntlet::Qg1StartupControlFrameV1::Complete { .. } => {
                return Err("subprocess emitted COMPLETE before its register".to_owned());
            }
        };
        let complete = Qg1StartupHandshakeV1::read_control_frame(&mut stdout)?;
        let register_count = match complete {
            frankensearch_quill_gauntlet::Qg1StartupControlFrameV1::Complete { register_count } => {
                register_count
            }
            frankensearch_quill_gauntlet::Qg1StartupControlFrameV1::Register { .. } => {
                return Err("subprocess emitted a second register before COMPLETE".to_owned());
            }
        };
        Ok((sequence, entry_bytes, register_count, stdout))
    }

    fn qg1_start_authority_subprocess() -> (
        std::process::Child,
        mpsc::Receiver<Result<(u64, Vec<u8>, u64, std::process::ChildStdout), String>>,
    ) {
        let current_test = std::env::current_exe().expect("current benchmark test executable");
        let mut child = Command::new(current_test)
            .env(super::QG1_AUTHORITY_SUBPROCESS_ENV, "1")
            .env(Qg1StartupHandshakeV1::ENV, Qg1StartupHandshakeV1::MODE)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Do not retain an undrained diagnostic pipe while the parent is
            // waiting on the bounded startup authority exchange.
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn the same binary with piped production handshake stdio");
        let stdout = child.stdout.take().expect("subprocess stdout");
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let _ = sender.send(qg1_read_subprocess_startup_transcript(stdout));
        });
        (child, receiver)
    }

    fn qg1_receive_subprocess_register(
        child: &mut std::process::Child,
        receiver: mpsc::Receiver<Result<(u64, Vec<u8>, u64, std::process::ChildStdout), String>>,
    ) -> (u64, Vec<u8>, u64, std::process::ChildStdout) {
        match receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(register)) => register,
            Ok(Err(error)) => panic!("same-binary authority subprocess failed: {error}"),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("same-binary authority subprocess did not frame startup control: {error}");
            }
        }
    }

    fn qg1_assert_no_post_register_work_before_ack(
        stdout: std::process::ChildStdout,
    ) -> mpsc::Receiver<Result<(Vec<u8>, std::process::ChildStdout), String>> {
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let mut stdout = stdout;
            let mut byte = [0_u8; 1];
            let result = stdout
                .read(&mut byte)
                .map(|count| (byte[..count].to_vec(), stdout))
                .map_err(|error| error.to_string());
            let _ = sender.send(result);
        });
        assert!(
            matches!(
                receiver.recv_timeout(Duration::from_millis(200)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "the same-binary helper emitted work before the parent final ACK"
        );
        receiver
    }

    fn qg1_finish_subprocess(
        child: &mut std::process::Child,
        receiver: mpsc::Receiver<Result<(Vec<u8>, std::process::ChildStdout), String>>,
    ) -> (std::process::ExitStatus, Vec<u8>) {
        let (mut output, mut stdout) = receiver
            .recv_timeout(Duration::from_secs(15))
            .expect("subprocess stdout closed or emitted after its authority outcome")
            .expect("read subprocess post-register stdout");
        let status = child.wait().expect("wait for authority subprocess");
        stdout
            .read_to_end(&mut output)
            .expect("drain subprocess stdout after authority outcome");
        (status, output)
    }

    // Ordinary `cfg(test)` helper, not a `#[test]` item. This bench is
    // `harness = false`, so test-attribute items are stripped and any ordinary
    // caller of one fails to resolve. `assert_qg1_authority_handshake_contract`
    // is that caller, and it runs from `main` in the harness-false binary.
    fn qg1_authority_ack_is_a_real_pre_sampling_barrier() {
        let (mut child, register_receiver) = qg1_start_authority_subprocess();
        let (sequence, entry_bytes, register_count, stdout) =
            qg1_receive_subprocess_register(&mut child, register_receiver);
        let expected = qg1_handshake_test_producer().register_entry();
        assert_eq!(
            entry_bytes,
            expected
                .to_json_bytes()
                .expect("serialize real register entry"),
            "the child emitted the complete real producer register before work"
        );
        assert_eq!(sequence, 1, "the startup register sequence begins at one");
        assert_eq!(
            register_count, 1,
            "the child completes the exact register count"
        );
        let post_register = qg1_assert_no_post_register_work_before_ack(stdout);
        let mut stdin = child.stdin.take().expect("subprocess stdin");
        stdin
            .write_all(&Qg1StartupHandshakeV1::final_ack_frame())
            .expect("write exact final ACK");
        stdin.flush().expect("flush exact final ACK");
        drop(stdin);
        let (status, output) = qg1_finish_subprocess(&mut child, post_register);
        assert!(status.success(), "valid ACK subprocess failed: {status}");
        assert!(
            output
                .windows(super::QG1_AUTHORITY_WORK_MARKER.len())
                .any(|window| window == super::QG1_AUTHORITY_WORK_MARKER),
            "the helper reached its post-barrier work marker only after the exact ACK"
        );
    }

    // Ordinary `cfg(test)` helper; see the note above.
    fn qg1_authority_ack_refuses_malformed_missing_and_timeout_before_work() {
        enum AckCase {
            Malformed,
            Missing,
            Timeout,
        }
        for case in [AckCase::Malformed, AckCase::Missing, AckCase::Timeout] {
            let (mut child, register_receiver) = qg1_start_authority_subprocess();
            let (_sequence, entry_bytes, register_count, stdout) =
                qg1_receive_subprocess_register(&mut child, register_receiver);
            let expected = qg1_handshake_test_producer().register_entry();
            assert_eq!(
                entry_bytes,
                expected
                    .to_json_bytes()
                    .expect("serialize real register entry"),
                "every negative reaches the actual production barrier before refusal"
            );
            assert_eq!(
                register_count, 1,
                "negative transcript has one complete register"
            );
            let post_register = qg1_assert_no_post_register_work_before_ack(stdout);
            let mut stdin = child.stdin.take().expect("subprocess stdin");
            match case {
                AckCase::Malformed => {
                    let mut acknowledgement = Qg1StartupHandshakeV1::final_ack_frame();
                    acknowledgement[0] ^= 0x01;
                    stdin
                        .write_all(&acknowledgement)
                        .expect("write malformed ACK");
                    stdin.flush().expect("flush malformed ACK");
                    drop(stdin);
                }
                AckCase::Missing => drop(stdin),
                AckCase::Timeout => {
                    let (status, output) = qg1_finish_subprocess(&mut child, post_register);
                    assert!(
                        !status.success(),
                        "withheld ACK must make the same-binary helper refuse before work"
                    );
                    assert!(
                        !output
                            .windows(super::QG1_AUTHORITY_WORK_MARKER.len())
                            .any(|window| window == super::QG1_AUTHORITY_WORK_MARKER),
                        "a withheld ACK must never reach the post-barrier work marker"
                    );
                    drop(stdin);
                    continue;
                }
            }
            let (status, output) = qg1_finish_subprocess(&mut child, post_register);
            assert!(
                !status.success(),
                "malformed and missing final ACKs must make the helper refuse"
            );
            assert!(
                !output
                    .windows(super::QG1_AUTHORITY_WORK_MARKER.len())
                    .any(|window| window == super::QG1_AUTHORITY_WORK_MARKER),
                "a rejected ACK must never reach the post-barrier work marker"
            );
        }
    }

    // Ordinary `cfg(test)` helper; see the note above.
    fn qg1_authority_barrier_remains_before_its_timing_origin_and_runners() {
        const TEST_MODULE_BOUNDARY: &str = "#[cfg(test)]\nmod tests {";
        let source = include_str!("perf_matrix.rs");
        assert_eq!(
            source.matches(TEST_MODULE_BOUNDARY).count(),
            1,
            "the production/test boundary must remain unique"
        );
        let production = &source[..source
            .find(TEST_MODULE_BOUNDARY)
            .expect("production/test boundary")];
        let bench_matrix = &production[production
            .find("fn bench_matrix(")
            .expect("unique production bench_matrix")..];
        let unique_offset = |marker: &str| {
            assert_eq!(
                bench_matrix.matches(marker).count(),
                1,
                "bench_matrix must contain exactly one placement marker {marker:?}"
            );
            bench_matrix
                .find(marker)
                .expect("unique bench_matrix placement marker")
        };
        let install = unique_offset("let qg1_startup_producers =");
        let acknowledgement = unique_offset(
            "require_qg1_pre_timing_authority_ack(selected_qg1, &qg1_startup_producers.engine_cells);",
        );
        let identity = unique_offset("emit_bench_elf_sha256(bench_identity);");
        let preflight =
            unique_offset("preflight_indexing_fixtures(&context, &matrix, &selected_specs);");
        let collection = unique_offset("let collection = collect_cell(");
        assert!(
            install < acknowledgement
                && acknowledgement < identity
                && identity < preflight
                && preflight < collection,
            "placement only: QG-1 must mint every producer, receive the parent final ACK, emit its ordinary identity, preflight, then collect"
        );
    }

    pub fn assert_qg1_authority_handshake_contract() {
        qg1_authority_ack_is_a_real_pre_sampling_barrier();
        qg1_authority_ack_refuses_malformed_missing_and_timeout_before_work();
        qg1_authority_barrier_remains_before_its_timing_origin_and_runners();
    }

    pub fn assert_qg9_cache_evidence_contract() {
        let quill_only = super::ColdCacheAccumulator {
            quill_successes: 1,
            ..super::ColdCacheAccumulator::default()
        };
        assert!(
            !super::cold_cache_evidence(quill_only).verified,
            "one arm cannot prove a comparative cold-open row"
        );

        let failed_tantivy = super::ColdCacheAccumulator {
            quill_successes: 1,
            tantivy_successes: 1,
            tantivy_failures: 1,
            ..super::ColdCacheAccumulator::default()
        };
        assert!(
            !super::cold_cache_evidence(failed_tantivy).verified,
            "an eviction failure must keep QG-9 at NoDecision"
        );

        let verified = super::cold_cache_evidence(super::ColdCacheAccumulator {
            quill_successes: 1,
            tantivy_successes: 1,
            ..super::ColdCacheAccumulator::default()
        });
        assert!(verified.verified);
        assert!(verified.procedure.contains("fresh child process"));
        assert!(verified.procedure.contains("POSIX_FADV_DONTNEED"));
    }

    #[test]
    fn qg9_cache_evidence_requires_a_clean_eviction_witness_from_both_arms() {
        assert_qg9_cache_evidence_contract();
    }

    pub fn assert_qg9_cache_eviction_file_discovery() {
        let fixture = tempfile::tempdir().expect("QG-9 fixture directory");
        let nested = fixture.path().join("segments");
        std::fs::create_dir(&nested).expect("nested QG-9 fixture directory");
        std::fs::write(fixture.path().join("meta.json"), b"metadata")
            .expect("write QG-9 metadata fixture");
        std::fs::write(nested.join("segment.fslx"), b"segment")
            .expect("write QG-9 segment fixture");

        let mut files = Vec::new();
        super::collect_regular_files(fixture.path(), &mut files)
            .expect("discover QG-9 regular index files");
        files.sort();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|path| path.is_file()));
        assert!(files.iter().any(|path| path.ends_with("meta.json")));
        assert!(files.iter().any(|path| path.ends_with("segment.fslx")));
    }

    #[test]
    fn qg9_cache_eviction_discovers_nested_regular_files_without_treating_dirs_as_files() {
        assert_qg9_cache_eviction_file_discovery();
    }

    #[cfg(target_os = "linux")]
    pub fn assert_qg9_cache_eviction_request() {
        let fixture = tempfile::tempdir().expect("QG-9 cache-eviction fixture directory");
        std::fs::write(fixture.path().join("segment.fslx"), b"segment")
            .expect("write QG-9 cache-eviction fixture");
        assert_eq!(
            super::evict_index_file_cache(fixture.path())
                .expect("QG-9 Linux cache eviction request must succeed"),
            1,
            "QG-9 must request eviction for every regular index file"
        );
    }

    #[cfg(not(target_os = "linux"))]
    pub fn assert_qg9_cache_eviction_request() {}

    #[test]
    fn qg9_linux_cache_eviction_request_is_real() {
        assert_qg9_cache_eviction_request();
    }

    fn hostile_tantivy_continuous_receipt() -> super::Qg1ContinuousTimingReceipt {
        super::Qg1ContinuousTimingReceipt {
            producer_coverage: super::Qg1ProducerCoverage::EngineIndexingLifecycle,
            arm: super::EngineArm::Tantivy,
            document_count: 20,
            prepared_input: super::Qg1PreparedSampleBinding {
                manifest_sha256: "a".repeat(64),
                indexed_content_sha256: "b".repeat(64),
                document_count: 20,
                content_bytes: 20_480,
                batch_count: 2,
                tail_document_id: "synthetic-00000019".to_owned(),
            },
            interval_started_ns: 0,
            batches: vec![
                super::Qg1BatchTiming {
                    document_start: 0,
                    document_count: 10,
                    feed_started_ns: 0,
                    feed_completed_ns: Some(20),
                    visibility_commit_completed_ns: None,
                },
                super::Qg1BatchTiming {
                    document_start: 10,
                    document_count: 10,
                    feed_started_ns: 40,
                    feed_completed_ns: Some(65),
                    visibility_commit_completed_ns: Some(75),
                },
            ],
            recorded_batch_count: 2,
            quill_publication_generation_delta: None,
            terminal_commit_completed_ns: 100,
            pre_search_rearm_join_completed_ns: None,
            terminal_worker_join_completed_ns: Some(155),
            terminal_tantivy_join: Some(super::BenchmarkWriterJoinReceipt {
                searchable_segments_before: 1,
                searchable_segments_after: 1,
                join_elapsed_ns: 30,
                writer_rearmed: false,
            }),
            terminal_searchable_quiescence_completed_ns: 180,
            interval_ended_ns: 180,
            terminal_searchability: super::Qg1TerminalFact::exact_tail_visible(
                "synthetic-00000019",
            ),
            terminal_quiescence: super::Qg1TerminalFact::tantivy_join_then_exact_tail(
                "synthetic-00000019",
                super::BenchmarkWriterJoinReceipt {
                    searchable_segments_before: 1,
                    searchable_segments_after: 1,
                    join_elapsed_ns: 30,
                    writer_rearmed: false,
                },
            ),
        }
    }

    pub fn assert_qg1_continuous_interval_contract() {
        let receipt = hostile_tantivy_continuous_receipt();
        receipt
            .validate()
            .expect("hostile timeline is one valid continuous interval");

        // This control is deliberately generous to the retired implementation:
        // it sums every feed/commit/search/join call represented in the hostile
        // timeline. It still loses every gap between calls, exactly the
        // undercount caused by adding independent `Instant::elapsed()` results.
        let old_summed_call_ns =
            (20 - 0) + (65 - 40) + (75 - 65) + (100 - 75) + (155 - 100) + (180 - 155);
        assert_eq!(old_summed_call_ns, 160);
        assert_eq!(receipt.interval_ended_ns, 180);
        assert!(
            old_summed_call_ns < receipt.interval_ended_ns,
            "summing individually timed calls must not masquerade as continuous wall time"
        );

        let assert_escape_rejected = |mutated: super::Qg1ContinuousTimingReceipt| {
            assert!(
                mutated.validate().is_err(),
                "a lifecycle phase outside the interval must invalidate the receipt"
            );
        };
        let mut first_feed_escape = receipt.clone();
        first_feed_escape.batches[0].feed_started_ns = 1;
        assert_escape_rejected(first_feed_escape);
        let mut batch_count_escape = receipt.clone();
        batch_count_escape.recorded_batch_count = 1;
        assert_escape_rejected(batch_count_escape);
        let mut tail_escape = receipt.clone();
        tail_escape.prepared_input.tail_document_id = "synthetic-00000018".to_owned();
        assert_escape_rejected(tail_escape);
        let mut feed_escape = receipt.clone();
        feed_escape.batches[1].feed_completed_ns = Some(171);
        assert_escape_rejected(feed_escape);
        let mut commit_escape = receipt.clone();
        commit_escape.terminal_commit_completed_ns = 69;
        assert_escape_rejected(commit_escape);
        let mut retired_rearm = receipt.clone();
        retired_rearm.pre_search_rearm_join_completed_ns = Some(110);
        assert_escape_rejected(retired_rearm);
        let mut retained_search_escape = receipt.clone();
        retained_search_escape.terminal_searchable_quiescence_completed_ns = 154;
        assert_escape_rejected(retained_search_escape);
        let mut terminal_join_escape = receipt.clone();
        terminal_join_escape.terminal_worker_join_completed_ns = Some(99);
        assert_escape_rejected(terminal_join_escape);
        let mut missing_join_api = receipt.clone();
        missing_join_api.terminal_tantivy_join = None;
        assert_escape_rejected(missing_join_api);
        let mut rearmed_join = receipt.clone();
        rearmed_join
            .terminal_tantivy_join
            .as_mut()
            .expect("hostile receipt names its actual join API result")
            .writer_rearmed = true;
        assert_escape_rejected(rearmed_join);
        let mut relabeled_tantivy_proof = receipt.clone();
        relabeled_tantivy_proof.terminal_quiescence =
            super::Qg1TerminalFact::quill_publication_then_exact_tail("synthetic-00000019", 1);
        assert_escape_rejected(relabeled_tantivy_proof);
        let mut mismatched_tantivy_proof = receipt.clone();
        mismatched_tantivy_proof.terminal_quiescence =
            super::Qg1TerminalFact::tantivy_join_then_exact_tail(
                "synthetic-00000018",
                mismatched_tantivy_proof
                    .terminal_tantivy_join
                    .expect("hostile Tantivy receipt retains its join proof"),
            );
        assert_escape_rejected(mismatched_tantivy_proof);
        let mut quiescence_escape = receipt.clone();
        quiescence_escape.interval_ended_ns = 159;
        assert_escape_rejected(quiescence_escape);
        let mut tokenizer_false_claim = receipt.clone();
        tokenizer_false_claim.producer_coverage =
            super::Qg1ProducerCoverage::TokenizerOnlyDiagnosticNoEngineLifecycle;
        assert_escape_rejected(tokenizer_false_claim);

        let mut unproved = receipt;
        unproved.terminal_searchability = super::Qg1TerminalFact::no_claim(
            "hostile terminal search intentionally lacks an exact sentinel",
        );
        unproved
            .validate()
            .expect("an unproved fact remains structurally typed rather than fabricated");
        assert_eq!(
            unproved.no_claim_details().collect::<Vec<_>>(),
            ["hostile terminal search intentionally lacks an exact sentinel"]
        );
        let encoded = serde_json::to_string(&unproved).expect("serialize typed NoClaim receipt");
        assert!(encoded.contains("\"status\":\"no_claim\""));
        assert!(encoded.contains(super::QG1_TERMINAL_NO_CLAIM_CODE));

        // Receipt-level rejection alone is not enough: an attacker could
        // preserve a serializable `NoClaim` receipt and hope a later layer
        // still publishes a rate. Convert the receipt through the production
        // binding factory, then run its hostile absence through the live
        // paired estimator used by the headline path.
        use frankensearch_quill_gauntlet::{
            PairedEstimatorConfig, PerfOperationScope, PerfRawSample, PerfSampleArm,
            PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance, estimate_paired_experiment,
        };
        let scope = PerfOperationScope {
            operation_id: "QG-1.bulk/tiny/1/positions_on.docs_per_second".to_owned(),
            version: 1,
            semantics: frankensearch_quill_gauntlet::PerfMetricSemantics::Throughput,
            unit: "docs/s".to_owned(),
        };
        let provenance = PerfSampleProvenance {
            run_id: "qg1-hostile-lifecycle-estimator".to_owned(),
            executable_sha256: "a".repeat(64),
            corpus_sha256: "b".repeat(64),
            input_identity: None,
            worker_id: "hostile-test".to_owned(),
            build_profile: "release-perf".to_owned(),
        };
        let authority_receipt = hostile_tantivy_continuous_receipt();
        let mut config = PairedEstimatorConfig::predeclared(0x5147_314c_4946_4543);
        let producer = config
            .install_qg1_lifecycle_authority(
                scope.clone(),
                provenance.corpus_sha256.clone(),
                authority_receipt.prepared_input.manifest_sha256.clone(),
                authority_receipt
                    .prepared_input
                    .indexed_content_sha256
                    .clone(),
                authority_receipt.prepared_input.document_count,
                authority_receipt.prepared_input.content_bytes,
                authority_receipt.prepared_input.batch_count,
                authority_receipt
                    .batches
                    .iter()
                    .map(|batch| Qg1BatchCoverage {
                        document_start: batch.document_start,
                        document_count: batch.document_count,
                    })
                    .collect(),
                authority_receipt.prepared_input.tail_document_id.clone(),
                10,
                vec![
                    (
                        "qg1.effect.tantivy_vs_quill.v1".to_owned(),
                        0,
                        0,
                        vec![PerfSampleArm::Control; 10],
                    ),
                    (
                        "qg1.null.tantivy.v1".to_owned(),
                        0,
                        10_000,
                        vec![PerfSampleArm::Control; 10],
                    ),
                ],
            )
            .expect("freeze hostile test lifecycle authority");
        let binding_from_receipt = |receipt: super::Qg1ContinuousTimingReceipt,
                                    stream_role: &str,
                                    stream_sequence: u64,
                                    sample_id: u64,
                                    block_id: u64,
                                    arm: PerfSampleArm| {
            let continuous = super::Qg1ContinuousMeasurement {
                work_units: receipt.document_count,
                origin: std::time::Instant::now(),
                elapsed_ns: receipt.interval_ended_ns,
                prepared_input: receipt.prepared_input.clone(),
                lifecycle_receipt: receipt,
            };
            super::qg1_live_sample_binding(
                Some(&continuous),
                continuous.elapsed_ns,
                &scope,
                &provenance,
                &config,
                &producer,
                stream_role,
                stream_sequence,
                sample_id,
                block_id,
                arm,
                match arm {
                    PerfSampleArm::Control => PerfSampleOrder::First,
                    PerfSampleArm::Treatment => PerfSampleOrder::Second,
                },
            )
        };
        let proved_binding = binding_from_receipt(
            hostile_tantivy_continuous_receipt(),
            "qg1.null.tantivy.v1",
            0,
            10_000,
            0,
            PerfSampleArm::Control,
        );
        assert!(
            proved_binding.is_some(),
            "a proved terminal lifecycle must create the estimator binding"
        );
        assert!(
            binding_from_receipt(
                hostile_tantivy_continuous_receipt(),
                "qg1.null.tantivy.v1",
                0,
                10_000,
                0,
                PerfSampleArm::Control,
            )
            .is_none(),
            "a consumed QG-1 producer capability must not be reissued"
        );

        // A producer that did not issue this invocation's authority is refused
        // before any capability is removed, so the wrong authority can never
        // burn a live slot. The foreign producer's own slot must therefore
        // still be consumable afterwards.
        let mut foreign_config = PairedEstimatorConfig::predeclared(0x5147_314c_4946_4544);
        let foreign_producer = foreign_config
            .install_qg1_lifecycle_authority(
                scope.clone(),
                provenance.corpus_sha256.clone(),
                authority_receipt.prepared_input.manifest_sha256.clone(),
                authority_receipt
                    .prepared_input
                    .indexed_content_sha256
                    .clone(),
                authority_receipt.prepared_input.document_count,
                authority_receipt.prepared_input.content_bytes,
                authority_receipt.prepared_input.batch_count,
                authority_receipt
                    .batches
                    .iter()
                    .map(|batch| Qg1BatchCoverage {
                        document_start: batch.document_start,
                        document_count: batch.document_count,
                    })
                    .collect(),
                authority_receipt.prepared_input.tail_document_id.clone(),
                10,
                vec![
                    (
                        "qg1.effect.tantivy_vs_quill.v1".to_owned(),
                        0,
                        0,
                        vec![PerfSampleArm::Control; 10],
                    ),
                    (
                        "qg1.null.tantivy.v1".to_owned(),
                        0,
                        10_000,
                        vec![PerfSampleArm::Control; 10],
                    ),
                ],
            )
            .expect("freeze an independent foreign lifecycle authority");
        assert!(
            !config.qg1_expected_authority_matches(foreign_producer.expected_authority()),
            "two independently issued producers must never be interchangeable"
        );
        let foreign_binding = |producer_config: &PairedEstimatorConfig,
                               producer: &Qg1LifecycleProducer| {
            let receipt = hostile_tantivy_continuous_receipt();
            let continuous = super::Qg1ContinuousMeasurement {
                work_units: receipt.document_count,
                origin: std::time::Instant::now(),
                elapsed_ns: receipt.interval_ended_ns,
                prepared_input: receipt.prepared_input.clone(),
                lifecycle_receipt: receipt,
            };
            super::qg1_live_sample_binding(
                Some(&continuous),
                continuous.elapsed_ns,
                &scope,
                &provenance,
                producer_config,
                producer,
                "qg1.effect.tantivy_vs_quill.v1",
                0,
                0,
                0,
                PerfSampleArm::Control,
                PerfSampleOrder::First,
            )
        };
        assert!(
            foreign_binding(&config, &foreign_producer).is_none(),
            "a foreign producer must be refused against this invocation's authority"
        );
        assert!(
            foreign_binding(&foreign_config, &foreign_producer).is_some(),
            "the refused attempt must not have consumed the foreign producer's slot"
        );
        let no_claim_binding = binding_from_receipt(
            unproved.clone(),
            "qg1.effect.tantivy_vs_quill.v1",
            0,
            0,
            0,
            PerfSampleArm::Control,
        );
        assert!(
            no_claim_binding.is_none(),
            "NoClaim lifecycle receipts must not create a headline-eligible binding"
        );

        let mut quill_receipt = hostile_tantivy_continuous_receipt();
        quill_receipt.arm = super::EngineArm::Quill;
        quill_receipt.terminal_worker_join_completed_ns = None;
        quill_receipt.terminal_tantivy_join = None;
        quill_receipt.quill_publication_generation_delta = Some(1);
        quill_receipt.terminal_quiescence =
            super::Qg1TerminalFact::quill_publication_then_exact_tail("synthetic-00000019", 1);
        quill_receipt
            .validate()
            .expect("hostile Quill timeline is a valid proved lifecycle");
        let stream = |control_receipt: Option<super::Qg1ContinuousTimingReceipt>,
                      treatment_receipt: Option<super::Qg1ContinuousTimingReceipt>,
                      stream_role: &str,
                      sample_id_base: u64| {
            (0_u64..10)
                .flat_map(|block_id| {
                    let base = block_id * 1_000;
                    let control_sample_id = sample_id_base + block_id * 2;
                    let treatment_sample_id = control_sample_id + 1;
                    [
                        PerfRawSample {
                            block_id,
                            sample_id: control_sample_id,
                            arm: PerfSampleArm::Control,
                            order: PerfSampleOrder::First,
                            phase: PerfSamplePhase::Measurement,
                            scope: scope.clone(),
                            provenance: provenance.clone(),
                            started_ns: base,
                            ended_ns: base + 180,
                            work_units: Some(20),
                            byte_count: Some(20_480),
                            observed_value: None,
                            group_id: None,
                            qg6_sample_binding: None,
                            qg1_sample_binding: control_receipt.clone().and_then(|receipt| {
                                binding_from_receipt(
                                    receipt,
                                    stream_role,
                                    block_id * 2,
                                    control_sample_id,
                                    block_id,
                                    PerfSampleArm::Control,
                                )
                            }),
                            tantivy_config_sha256: None,
                        },
                        PerfRawSample {
                            block_id,
                            sample_id: treatment_sample_id,
                            arm: PerfSampleArm::Treatment,
                            order: PerfSampleOrder::Second,
                            phase: PerfSamplePhase::Measurement,
                            scope: scope.clone(),
                            provenance: provenance.clone(),
                            started_ns: base + 181,
                            ended_ns: base + 361,
                            work_units: Some(20),
                            byte_count: Some(20_480),
                            observed_value: None,
                            group_id: None,
                            qg6_sample_binding: None,
                            qg1_sample_binding: treatment_receipt.clone().and_then(|receipt| {
                                binding_from_receipt(
                                    receipt,
                                    stream_role,
                                    block_id * 2 + 1,
                                    treatment_sample_id,
                                    block_id,
                                    PerfSampleArm::Treatment,
                                )
                            }),
                            tantivy_config_sha256: None,
                        },
                    ]
                })
                .collect::<Vec<_>>()
        };
        let valid_effect = stream(
            Some(hostile_tantivy_continuous_receipt()),
            Some(quill_receipt.clone()),
            "qg1.effect.tantivy_vs_quill.v1",
            0,
        );
        let valid_null = stream(
            Some(hostile_tantivy_continuous_receipt()),
            Some(hostile_tantivy_continuous_receipt()),
            "qg1.null.tantivy.v1",
            10_000,
        );
        assert!(
            estimate_paired_experiment(&valid_effect, &valid_null, &config).is_err(),
            "the authority-free estimator must refuse canonical QG-1 throughput evidence"
        );
        assert!(
            estimate_paired_experiment_against_qg1_authority(
                &valid_effect,
                &valid_null,
                &config,
                Some(producer.expected_authority()),
            )
            .is_ok(),
            "proved QG-1 lifecycle bindings must reach the live estimator"
        );
        let no_claim_effect = stream(
            no_claim_binding.map(|_| unproved.clone()),
            Some(quill_receipt.clone()),
            "qg1.effect.tantivy_vs_quill.v1",
            20_000,
        );
        assert!(
            estimate_paired_experiment_against_qg1_authority(
                &no_claim_effect,
                &valid_null,
                &config,
                Some(producer.expected_authority()),
            )
            .is_err(),
            "NoClaim lifecycle receipts must be rejected by the live estimator before headline generation"
        );

        let mut relabelled = hostile_tantivy_continuous_receipt();
        relabelled.terminal_quiescence =
            super::Qg1TerminalFact::quill_publication_then_exact_tail("synthetic-00000019", 1);
        let relabelled_effect = stream(
            binding_from_receipt(
                relabelled,
                "qg1.effect.tantivy_vs_quill.v1",
                0,
                30_000,
                0,
                PerfSampleArm::Control,
            )
            .map(|_| hostile_tantivy_continuous_receipt()),
            Some(quill_receipt),
            "qg1.effect.tantivy_vs_quill.v1",
            30_000,
        );
        assert!(
            estimate_paired_experiment_against_qg1_authority(
                &relabelled_effect,
                &valid_null,
                &config,
                Some(producer.expected_authority()),
            )
            .is_err(),
            "an arm-relabeled terminal proof must reach and fail the live estimator"
        );
    }

    #[test]
    fn qg1_continuous_interval_rejects_summed_call_undercount_and_phase_escape() {
        assert_qg1_continuous_interval_contract();
    }

    pub fn assert_qg1_real_terminal_visibility_contract() {
        use frankensearch_quill_gauntlet::{PerfGate, PerfMatrixSpec};

        let spec = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .filter(|spec| spec.metric == "docs_per_second" && spec.threads == Some(1))
            .min_by_key(|spec| spec.document_count)
            .expect("smallest canonical single-thread QG-1 indexing cell")
            .clone();
        assert_eq!(
            spec.document_count,
            Some(500),
            "real terminal test must stay on the smallest normative QG-1 corpus"
        );
        let context = super::BenchContext::for_selected(
            super::MatrixScale::Smoke,
            std::slice::from_ref(&spec),
        );
        let first_sequence =
            super::QG1_CONTINUOUS_TIMING_COUNTER.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            super::metric_semantics(&spec),
            frankensearch_quill_gauntlet::PerfMetricSemantics::Throughput,
            "the real QG-1 indexing cell publishes native throughput"
        );
        let mut measured = Vec::new();
        for arm in [super::EngineArm::Quill, super::EngineArm::Tantivy] {
            let measurement = super::bulk_metric(&context, &spec, arm);
            assert!(
                measurement.value.is_finite() && measurement.value > 0.0,
                "real {arm:?} terminal sample must return positive finite throughput"
            );
            let continuous = measurement
                .continuous
                .expect("a real QG-1 engine arm publishes its continuous interval");
            assert_eq!(
                continuous.work_units, 500,
                "the interval must cover the exact immutable 500-document corpus"
            );
            assert!(continuous.elapsed_ns > 0);
            assert_eq!(
                measurement.value.to_bits(),
                super::throughput_per_second(continuous.work_units, continuous.elapsed_ns)
                    .to_bits(),
                "the reported rate must be exactly work over the continuous interval"
            );
            // The same interval, run through the publication chokepoint the
            // paired stream uses, is admissible as Throughput and nothing else.
            let window = super::qg1_sample_window(
                frankensearch_quill_gauntlet::PerfMetricSemantics::Throughput,
                Some(continuous.work_units),
                0,
                continuous.elapsed_ns.saturating_add(1),
                Some(super::Qg1IntervalOffsets {
                    work_units: continuous.work_units,
                    started_ns: 0,
                    elapsed_ns: continuous.elapsed_ns,
                }),
            )
            .expect("a real continuous interval is publishable as throughput");
            assert_eq!(window.ended_ns - window.started_ns, continuous.elapsed_ns);
            measured.push((arm, continuous));
        }

        let receipts = {
            let receipts = super::QG1_CONTINUOUS_TIMING_RECEIPTS
                .get()
                .expect("real QG-1 terminal samples emitted diagnostic receipts")
                .lock()
                .expect("lock real QG-1 terminal receipts");
            receipts
                .iter()
                .filter(|record| {
                    record.sequence >= first_sequence && record.fixture == spec.fixture
                })
                .cloned()
                .collect::<Vec<_>>()
        };
        for arm in [super::EngineArm::Quill, super::EngineArm::Tantivy] {
            let record = receipts
                .iter()
                .find(|record| {
                    record.sequence >= first_sequence
                        && record.fixture == spec.fixture
                        && record.timing.arm == arm
                })
                .expect("missing real engine terminal timing receipt");
            record
                .timing
                .validate()
                .expect("invalid real engine timing receipt");
            assert!(matches!(
                &record.timing.terminal_searchability,
                super::Qg1TerminalFact::Proven {
                    proof: super::Qg1TerminalProof::ExactPreparedTailVisible { .. }
                }
            ));
            assert!(matches!(
                &record.timing.terminal_quiescence,
                super::Qg1TerminalFact::Proven { .. }
            ));
            assert_eq!(record.admission_status, "no_claim");
            assert_eq!(
                record.admission_no_claim_code,
                super::QG1_TIMING_DIAGNOSTIC_NO_CLAIM_CODE
            );
            assert_eq!(
                record.timing.producer_coverage,
                super::Qg1ProducerCoverage::EngineIndexingLifecycle
            );
            // The receipt and the sample are not two independent observations of
            // the run: the interval the receipt reports is the interval the
            // sample publishes, and the work it names is the work the rate is
            // divided by.
            let (_, continuous) = measured
                .iter()
                .find(|(measured_arm, _)| *measured_arm == arm)
                .expect("every measured arm has a continuous interval");
            assert_eq!(
                record.timing.interval_ended_ns, continuous.elapsed_ns,
                "the published sample window must be the receipt's continuous interval"
            );
            assert_eq!(record.timing.document_count, continuous.work_units);
            assert_eq!(
                record.timing.recorded_batch_count, record.timing.prepared_input.batch_count,
                "the receipt must bind batches that completed the real engine feed"
            );
            assert_eq!(
                record.timing.prepared_input.tail_document_id,
                super::qg1_tail_document_id(continuous.work_units),
                "terminal search must name the final document in this exact measured corpus"
            );
            assert_eq!(
                record.timing.prepared_input, continuous.prepared_input,
                "the receipt and raw-rate interval must name the exact prepared input"
            );
            assert_eq!(record.timing.batches[0].feed_started_ns, 0);
            assert_eq!(
                record.timing.interval_ended_ns,
                record.timing.terminal_searchable_quiescence_completed_ns,
                "the sample must publish the one real retained-reader searchable-quiescence boundary"
            );
            assert_eq!(
                record.timing.pre_search_rearm_join_completed_ns, None,
                "the old replacement-writer rearm is never part of QG-1 timing"
            );
            match arm {
                super::EngineArm::Quill => assert_eq!(
                    (
                        record.timing.terminal_worker_join_completed_ns,
                        record.timing.terminal_tantivy_join,
                    ),
                    (None, None),
                    "Quill has no external Tantivy worker join"
                ),
                super::EngineArm::Tantivy => {
                    let terminal_join_completed = record
                        .timing
                        .terminal_worker_join_completed_ns
                        .expect("Tantivy must finish its one nonrearming terminal worker join");
                    assert!(
                        terminal_join_completed
                            <= record.timing.terminal_searchable_quiescence_completed_ns,
                        "planted negative: a Tantivy tail search cannot certify a boundary before its actual worker join"
                    );
                    let join = record
                        .timing
                        .terminal_tantivy_join
                        .expect("Tantivy must retain the actual one-shot join API receipt");
                    assert!(
                        !join.writer_rearmed,
                        "the retained Tantivy reader must not rearm a replacement writer"
                    );
                }
            }
        }
    }

    #[test]
    fn qg1_smallest_real_fixture_is_terminally_visible_for_both_engines() {
        assert_qg1_real_terminal_visibility_contract();
    }

    pub fn assert_qg1_raw_sample_work_contract() {
        use frankensearch_quill_gauntlet::{PerfGate, PerfMatrixSpec};

        let mut qg1 = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|spec| spec.metric == "docs_per_second")
            .expect("canonical QG-1 indexing cell")
            .clone();
        qg1.document_count = Some(12);
        let context = super::BenchContext::for_selected(super::MatrixScale::Full, &[qg1.clone()]);
        let quill_work = super::raw_sample_work(&context, &qg1);
        let tantivy_work = super::raw_sample_work(&context, &qg1);
        assert_eq!(quill_work, tantivy_work, "paired arms require equal work");
        let (work_units, byte_count) = quill_work;
        assert_eq!(work_units, Some(12));
        let expected_bytes = context
            .qg1_prefix(12)
            .1
            .iter()
            .map(|document| {
                u64::try_from(document.content.len()).expect("test content length fits u64")
            })
            .sum::<u64>();
        assert_eq!(byte_count, Some(expected_bytes));
        assert!(expected_bytes > 0);

        let prepared_input = context.qg1_sample_input(12);
        assert_eq!(prepared_input.binding.document_count, 12);
        assert_eq!(prepared_input.binding.content_bytes, expected_bytes);
        prepared_input
            .verify_binding(&prepared_input.binding)
            .expect("the measured QG-1 input must verify against its own binding");
        let mut unbound_input = prepared_input.binding.clone();
        unbound_input.content_bytes = unbound_input
            .content_bytes
            .checked_add(1)
            .expect("hostile content-byte mutation fits u64");
        assert!(
            prepared_input.verify_binding(&unbound_input).is_err(),
            "planted negative: a separately labeled corpus cannot supply the sample denominator"
        );
        let mut lifecycle_receipt = hostile_tantivy_continuous_receipt();
        lifecycle_receipt.document_count = prepared_input.binding.document_count;
        lifecycle_receipt.prepared_input = prepared_input.binding.clone();
        lifecycle_receipt.batches[0].document_count = 6;
        lifecycle_receipt.batches[1].document_start = 6;
        lifecycle_receipt.batches[1].document_count = 6;
        lifecycle_receipt
            .validate()
            .expect("hostile lifecycle receipt is rebound to this measured prepared input");
        let actual_measurement = super::Qg1ContinuousMeasurement {
            work_units: prepared_input.binding.document_count,
            origin: std::time::Instant::now(),
            elapsed_ns: 1,
            prepared_input: prepared_input.binding.clone(),
            lifecycle_receipt,
        };
        assert_eq!(
            super::qg1_raw_sample_denominator(
                (Some(12), Some(expected_bytes)),
                Some(&actual_measurement),
            )
            .expect("raw denominator derives from the measured prepared input"),
            (Some(12), Some(expected_bytes))
        );
        assert!(
            super::qg1_raw_sample_denominator(
                (Some(12), Some(expected_bytes.saturating_add(1))),
                Some(&actual_measurement),
            )
            .is_err(),
            "planted negative: raw bytes may not come from a separately regenerated input"
        );
        let mut receipt_relabel = actual_measurement.clone();
        receipt_relabel
            .lifecycle_receipt
            .prepared_input
            .indexed_content_sha256 = "c".repeat(64);
        assert!(
            super::qg1_raw_sample_denominator(
                (Some(12), Some(expected_bytes)),
                Some(&receipt_relabel),
            )
            .is_err(),
            "planted negative: one sample cannot borrow another lifecycle receipt under its own input"
        );

        let mut tokenizer = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|spec| spec.metric == "tokenize_docs_per_second")
            .expect("canonical QG-1 tokenizer diagnostic")
            .clone();
        tokenizer.document_count = Some(12);
        assert_eq!(
            super::qg1_producer_coverage(&tokenizer),
            Some(super::Qg1ProducerCoverage::TokenizerOnlyDiagnosticNoEngineLifecycle)
        );
        assert_eq!(
            super::raw_sample_work(&context, &tokenizer),
            quill_work,
            "tokenizer diagnostics retain exact work metadata without engine lifecycle proof"
        );
        assert!(
            !super::qg1_producer_coverage(&tokenizer)
                .expect("typed QG-1 tokenizer coverage")
                .admits_engine_lifecycle_receipt()
        );

        let mut non_qg1 = qg1;
        non_qg1.gate = PerfGate::Qg2;
        assert_eq!(super::raw_sample_work(&context, &non_qg1), (None, None));
    }

    #[test]
    fn qg1_throughput_raw_samples_bind_equal_work_and_content_bytes() {
        assert_qg1_raw_sample_work_contract();
    }

    pub fn assert_qg1_throughput_semantics_contract() {
        use frankensearch_quill_gauntlet::{PerfGate, PerfMatrixSpec, PerfMetricSemantics};

        let matrix = PerfMatrixSpec::complete();
        let indexing = matrix
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|spec| spec.metric == "docs_per_second")
            .expect("canonical QG-1 indexing cell")
            .clone();
        let tokenizer = matrix
            .for_gate(PerfGate::Qg1)
            .into_iter()
            .find(|spec| spec.metric == "tokenize_docs_per_second")
            .expect("canonical QG-1 tokenizer diagnostic")
            .clone();

        assert_eq!(
            super::metric_semantics(&indexing),
            PerfMetricSemantics::Throughput,
            "a continuous QG-1 engine interval is a native throughput operation"
        );
        assert_eq!(
            super::operation_scope(&indexing).semantics,
            PerfMetricSemantics::Throughput,
            "the scope every QG-1 indexing sample carries must say so too"
        );
        assert_eq!(
            super::metric_semantics(&tokenizer),
            PerfMetricSemantics::GaugeHigherIsBetter,
            "the tokenizer diagnostic sums per-batch calls and cannot claim throughput semantics"
        );
        // Same metric name, different producer: QG-8 bulk indexing still sums
        // independently timed batches, so the name alone must not decide.
        let mut summed_bulk = indexing.clone();
        summed_bulk.gate = PerfGate::Qg8;
        assert_eq!(
            super::metric_semantics(&summed_bulk),
            PerfMetricSemantics::GaugeHigherIsBetter
        );
        let latency = matrix
            .for_gate(PerfGate::Qg4)
            .into_iter()
            .next()
            .expect("canonical QG-4 latency cell")
            .clone();
        assert_eq!(
            super::metric_semantics(&latency),
            PerfMetricSemantics::GaugeLowerIsBetter
        );

        let interval = super::Qg1IntervalOffsets {
            work_units: 500,
            started_ns: 40,
            elapsed_ns: 100,
        };
        assert_eq!(
            super::qg1_sample_window(
                PerfMetricSemantics::Throughput,
                Some(500),
                10,
                400,
                Some(interval),
            )
            .expect("a continuous interval publishes its own window"),
            super::Qg1SampleWindow {
                started_ns: 40,
                ended_ns: 140,
            },
            "the published window must be the interval, not the enclosing call"
        );

        // Planted negative: the same real interval filed as a gauge.
        let gauge_error = super::qg1_sample_window(
            PerfMetricSemantics::GaugeHigherIsBetter,
            Some(500),
            10,
            400,
            Some(interval),
        )
        .expect_err("a continuous interval must not be published as a gauge");
        assert!(
            gauge_error.contains("must be published as Throughput"),
            "unexpected gauge rejection: {gauge_error}"
        );

        // Planted negative: throughput semantics offered a summed per-call
        // duration, which is all any non-QG-1 rate producer can offer.
        let summed_error =
            super::qg1_sample_window(PerfMetricSemantics::Throughput, Some(500), 10, 400, None)
                .expect_err("summed per-call timing must not be published as throughput");
        assert!(
            summed_error.contains("summed per-call duration"),
            "unexpected summed-call rejection: {summed_error}"
        );

        let work_error = super::qg1_sample_window(
            PerfMetricSemantics::Throughput,
            Some(499),
            10,
            400,
            Some(interval),
        )
        .expect_err("declared work must equal the work the interval processed");
        assert!(
            work_error.contains("continuous interval processed 500"),
            "unexpected work rejection: {work_error}"
        );
        assert!(
            super::qg1_sample_window(
                PerfMetricSemantics::Throughput,
                None,
                10,
                400,
                Some(interval)
            )
            .is_err(),
            "a throughput sample without work units cannot be derived"
        );

        let overrun = super::Qg1IntervalOffsets {
            elapsed_ns: 400,
            ..interval
        };
        assert!(
            super::qg1_sample_window(
                PerfMetricSemantics::Throughput,
                Some(500),
                10,
                400,
                Some(overrun)
            )
            .is_err(),
            "an interval must not end after the call that measured it"
        );
        let early = super::Qg1IntervalOffsets {
            started_ns: 5,
            ..interval
        };
        assert!(
            super::qg1_sample_window(
                PerfMetricSemantics::Throughput,
                Some(500),
                10,
                400,
                Some(early)
            )
            .is_err(),
            "an interval must not begin before the call that measured it"
        );
        let empty = super::Qg1IntervalOffsets {
            elapsed_ns: 0,
            ..interval
        };
        assert!(
            super::qg1_sample_window(
                PerfMetricSemantics::Throughput,
                Some(500),
                10,
                400,
                Some(empty)
            )
            .is_err(),
            "a zero-length interval cannot carry a rate"
        );

        // Gauge producers keep the enclosing call window, including its
        // degenerate-resolution guard.
        assert_eq!(
            super::qg1_sample_window(
                PerfMetricSemantics::GaugeHigherIsBetter,
                Some(500),
                10,
                400,
                None
            )
            .expect("gauge samples publish their call window"),
            super::Qg1SampleWindow {
                started_ns: 10,
                ended_ns: 400,
            }
        );
        assert_eq!(
            super::qg1_sample_window(PerfMetricSemantics::GaugeLowerIsBetter, None, 10, 10, None)
                .expect("gauge samples publish their call window"),
            super::Qg1SampleWindow {
                started_ns: 10,
                ended_ns: 11,
            }
        );

        // The retired summed-call form drops the gaps between calls. With the
        // corrected end-at-quiescence fixture it reports 135 ns of work inside
        // the actual 155 ns interval, still inflating the same 20 documents.
        let receipt = hostile_tantivy_continuous_receipt();
        let continuous_rate =
            super::throughput_per_second(receipt.document_count, receipt.interval_ended_ns);
        let summed_rate = super::throughput_per_second(receipt.document_count, 135);
        assert!(
            summed_rate > continuous_rate * 1.14 && summed_rate < continuous_rate * 1.15,
            "summed-call timing must overstate the rate: {summed_rate} vs {continuous_rate}"
        );
    }

    #[test]
    fn qg1_indexing_is_throughput_and_summed_or_gauge_timing_is_rejected() {
        assert_qg1_throughput_semantics_contract();
    }

    /// A throughput row is recomputed by the estimator from work and interval; a
    /// gauge row is whatever number the harness stored. This runs one paired
    /// stream through the real estimator twice with identical timings and work,
    /// changing only the declared semantics and the stored observation.
    pub fn assert_throughput_typing_is_not_cosmetic() {
        use frankensearch_quill_gauntlet::{
            PERF_MIN_RUNS, PairedEstimatorConfig, PerfMetricSemantics, PerfOperationScope,
            PerfRawSample, PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance,
            estimate_paired_experiment,
        };

        const WORK_UNITS: u64 = 500;
        const CONTROL_NS: u64 = 2_000_000;
        const TREATMENT_NS: u64 = 1_000_000;
        let rounds = u64::try_from(PERF_MIN_RUNS).expect("min runs fits u64") + 2;

        let scope = |semantics| PerfOperationScope {
            operation_id: "qg1.synthetic.bulk_index_publish".to_owned(),
            version: 1,
            semantics,
            unit: "docs/s".to_owned(),
        };
        let provenance = PerfSampleProvenance {
            run_id: "qg1-semantics-control".to_owned(),
            executable_sha256: "a".repeat(64),
            corpus_sha256: "b".repeat(64),
            input_identity: None,
            worker_id: "synthetic-worker".to_owned(),
            build_profile: "release-perf".to_owned(),
        };
        let sample = |semantics: PerfMetricSemantics,
                      block_id: u64,
                      sample_id: u64,
                      arm: PerfSampleArm,
                      order: PerfSampleOrder,
                      elapsed_ns: u64,
                      observed_value: f64| PerfRawSample {
            block_id,
            sample_id,
            arm,
            order,
            phase: PerfSamplePhase::Measurement,
            scope: scope(semantics),
            provenance: provenance.clone(),
            started_ns: block_id.wrapping_mul(100_000_000),
            ended_ns: block_id.wrapping_mul(100_000_000) + elapsed_ns,
            work_units: Some(WORK_UNITS),
            byte_count: Some(WORK_UNITS * 1_024),
            observed_value: Some(observed_value),
            group_id: None,
            qg6_sample_binding: None,
            qg1_sample_binding: None,
            tantivy_config_sha256: None,
        };
        // `stored_treatment` is the only lever: `None` stores the honest derived
        // rate, `Some(value)` stores a fabricated one over the same interval.
        let stream = |semantics: PerfMetricSemantics,
                      base_block: u64,
                      base_id: u64,
                      treatment_base_ns: u64,
                      stored_treatment: Option<f64>| {
            (0..rounds)
                .flat_map(|round| {
                    // Per-round jitter keeps the distributions non-degenerate
                    // without changing which arm is faster.
                    let control_ns = CONTROL_NS + round;
                    let treatment_ns = treatment_base_ns + round;
                    let (control_order, treatment_order) = if round % 2 == 0 {
                        (PerfSampleOrder::First, PerfSampleOrder::Second)
                    } else {
                        (PerfSampleOrder::Second, PerfSampleOrder::First)
                    };
                    [
                        sample(
                            semantics,
                            base_block + round,
                            base_id + round * 2,
                            PerfSampleArm::Control,
                            control_order,
                            control_ns,
                            super::throughput_per_second(WORK_UNITS, control_ns),
                        ),
                        sample(
                            semantics,
                            base_block + round,
                            base_id + round * 2 + 1,
                            PerfSampleArm::Treatment,
                            treatment_order,
                            treatment_ns,
                            stored_treatment.unwrap_or_else(|| {
                                super::throughput_per_second(WORK_UNITS, treatment_ns)
                            }),
                        ),
                    ]
                })
                .collect::<Vec<_>>()
        };
        let config = PairedEstimatorConfig::predeclared(0x1cec_0a57_5eed_0001);

        let throughput_null = stream(
            PerfMetricSemantics::Throughput,
            1_000,
            10_000,
            CONTROL_NS,
            None,
        );
        let honest = estimate_paired_experiment(
            &stream(PerfMetricSemantics::Throughput, 0, 0, TREATMENT_NS, None),
            &throughput_null,
            &config,
        )
        .expect("throughput stream is structurally valid");
        let tampered = estimate_paired_experiment(
            &stream(
                PerfMetricSemantics::Throughput,
                0,
                0,
                TREATMENT_NS,
                Some(1.0),
            ),
            &throughput_null,
            &config,
        )
        .expect("tampered throughput stream is structurally valid");
        assert_eq!(
            honest.effect.treatment.p50.to_bits(),
            tampered.effect.treatment.p50.to_bits(),
            "throughput is recomputed from work and interval, so a fabricated observation \
             cannot move it"
        );
        assert!(
            honest.effect.treatment.p50 <= super::throughput_per_second(WORK_UNITS, TREATMENT_NS)
                && honest.effect.treatment.p50
                    >= super::throughput_per_second(WORK_UNITS, TREATMENT_NS + rounds),
            "the throughput estimate must fall inside the rates its intervals imply: {}",
            honest.effect.treatment.p50
        );

        // The identical fabricated observation under gauge semantics: nothing
        // recomputes it, so it simply becomes the answer.
        let gauge = estimate_paired_experiment(
            &stream(
                PerfMetricSemantics::GaugeHigherIsBetter,
                0,
                0,
                TREATMENT_NS,
                Some(1.0),
            ),
            &stream(
                PerfMetricSemantics::GaugeHigherIsBetter,
                1_000,
                10_000,
                CONTROL_NS,
                None,
            ),
            &config,
        )
        .expect("gauge stream is structurally valid");
        assert_eq!(
            gauge.effect.treatment.p50.to_bits(),
            1.0_f64.to_bits(),
            "a gauge publishes whatever the harness stored, which is why a rate must not be one"
        );
    }

    #[test]
    fn gauge_typing_would_publish_a_rate_no_clock_ever_checked() {
        assert_throughput_typing_is_not_cosmetic();
    }

    #[test]
    fn runner_profile_parsers_accept_only_closed_canonical_ids() {
        use frankensearch_quill_gauntlet::{ExecutionProfileId, HardwareClassId};

        assert_eq!(
            super::parse_hardware_class_id("trj-zen3-5995wx"),
            Ok(HardwareClassId::TrjZen35995wx)
        );
        assert_eq!(
            super::parse_hardware_class_id("m4-macos"),
            Ok(HardwareClassId::M4Macos)
        );
        assert!(super::parse_hardware_class_id("trj-zen3-64c").is_err());
        assert!(super::parse_hardware_class_id(" M4-macos").is_err());
        assert_eq!(
            super::parse_execution_profile_id("physical-64"),
            Ok(ExecutionProfileId::Physical64)
        );
        assert_eq!(
            super::parse_execution_profile_id("smt2-128"),
            Ok(ExecutionProfileId::Smt2_128)
        );
        assert_eq!(
            super::parse_execution_profile_id("scheduler-10"),
            Ok(ExecutionProfileId::Scheduler10)
        );
        assert!(super::parse_execution_profile_id("64").is_err());
        assert!(super::parse_execution_profile_id("p-plus-e").is_err());
    }

    #[test]
    fn runner_plan_claims_reject_mutation_and_cross_profile_substitution() {
        use frankensearch_quill_gauntlet::{
            ExecutionProfileId, HardwareClassId, MachineClassRegistry, MachineProfileKey, PerfGate,
            PerfMatrixSpec,
        };

        fn claims_for(
            matrix: &PerfMatrixSpec,
            hardware_class: HardwareClassId,
            execution_profile: ExecutionProfileId,
            gate: PerfGate,
        ) -> super::RunnerPlanClaims {
            let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
            let profile = MachineProfileKey::new(hardware_class, execution_profile)
                .expect("canonical typed profile key");
            let plan = matrix
                .applicability_plan(&registry, profile, gate)
                .expect("canonical applicability plan");
            let execution_capacity = plan
                .execution_capacity
                .expect("typed promotion profile capacity");
            let max_exercised_cell_width = plan
                .max_exercised_cell_width
                .expect("typed promotion gate maximum");
            super::RunnerPlanClaims {
                gate,
                hardware_class,
                execution_profile,
                execution_capacity,
                max_exercised_cell_width,
                rayon_num_threads: execution_capacity,
                applicability_plan_schema_version: plan.binding().schema_version.clone(),
                applicability_plan_sha256: plan.binding().applicability_plan_sha256.clone(),
                gate_matrix_contract_sha256: plan.binding().gate_matrix_contract_sha256.clone(),
                profile_contract_sha256: plan.binding().profile_contract_sha256.clone(),
                registry_schema_version: plan.binding().registry_schema_version.clone(),
                registry_sha256: plan.binding().registry_sha256.clone(),
            }
        }

        let matrix = PerfMatrixSpec::complete();
        let claims = claims_for(
            &matrix,
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
            PerfGate::Qg1,
        );
        let context = super::RunnerApplicabilityContext::reconstruct(&matrix, &claims)
            .expect("exact physical-64 claims");
        assert_eq!(
            context.profile,
            MachineProfileKey::new(
                HardwareClassId::TrjZen35995wx,
                ExecutionProfileId::Physical64,
            )
            .expect("canonical typed profile key")
        );
        assert_eq!(context.execution_capacity, 64);
        assert_eq!(context.max_exercised_cell_width, 64);

        let mut plan_hash_mutation = claims.clone();
        plan_hash_mutation.applicability_plan_sha256 = "0".repeat(64);
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &plan_hash_mutation)
                .expect_err("mutated plan hash")
                .contains("QUILL_PERF_APPLICABILITY_PLAN_SHA256")
        );

        let mut matrix_hash_mutation = claims.clone();
        matrix_hash_mutation.gate_matrix_contract_sha256 = "0".repeat(64);
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &matrix_hash_mutation)
                .expect_err("mutated gate matrix hash")
                .contains("QUILL_PERF_GATE_MATRIX_CONTRACT_SHA256")
        );

        let mut capacity_mutation = claims.clone();
        capacity_mutation.execution_capacity = 63;
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &capacity_mutation)
                .expect_err("mutated execution capacity")
                .contains("QUILL_PERF_EXECUTION_CAPACITY")
        );

        let mut max_width_mutation = claims.clone();
        max_width_mutation.max_exercised_cell_width = 32;
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &max_width_mutation)
                .expect_err("mutated gate maximum")
                .contains("QUILL_PERF_MAX_EXERCISED_CELL_WIDTH")
        );

        let mut rayon_mutation = claims.clone();
        rayon_mutation.rayon_num_threads = 63;
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &rayon_mutation)
                .expect_err("mutated Rayon capacity")
                .contains("RAYON_NUM_THREADS")
        );

        let mut cross_profile = claims.clone();
        cross_profile.execution_profile = ExecutionProfileId::Smt2_128;
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &cross_profile)
                .expect_err("cross-profile substitution")
                .contains("QUILL_PERF_APPLICABILITY_PLAN_SHA256")
        );

        let mut cross_hardware = claims;
        cross_hardware.hardware_class = HardwareClassId::M4Macos;
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&matrix, &cross_hardware)
                .expect_err("cross-hardware profile substitution")
                .contains("profile key is invalid")
        );
    }

    #[test]
    fn m4_selection_executes_exact_runnable_plan_and_rejects_na_filters() {
        use frankensearch_quill_gauntlet::{
            EvidenceRole, ExecutionProfileId, HardwareClassId, MachineClassRegistry,
            MachineProfileKey, PerfGate, PerfMatrixSpec,
        };

        fn claims_for(
            matrix: &PerfMatrixSpec,
            hardware_class: HardwareClassId,
            execution_profile: ExecutionProfileId,
            gate: PerfGate,
        ) -> super::RunnerPlanClaims {
            let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
            let profile = MachineProfileKey::new(hardware_class, execution_profile)
                .expect("canonical typed profile key");
            let plan = matrix
                .applicability_plan(&registry, profile, gate)
                .expect("canonical applicability plan");
            let execution_capacity = plan
                .execution_capacity
                .expect("typed promotion profile capacity");
            let max_exercised_cell_width = plan
                .max_exercised_cell_width
                .expect("typed promotion gate maximum");
            super::RunnerPlanClaims {
                gate,
                hardware_class,
                execution_profile,
                execution_capacity,
                max_exercised_cell_width,
                rayon_num_threads: execution_capacity,
                applicability_plan_schema_version: plan.binding().schema_version.clone(),
                applicability_plan_sha256: plan.binding().applicability_plan_sha256.clone(),
                gate_matrix_contract_sha256: plan.binding().gate_matrix_contract_sha256.clone(),
                profile_contract_sha256: plan.binding().profile_contract_sha256.clone(),
                registry_schema_version: plan.binding().registry_schema_version.clone(),
                registry_sha256: plan.binding().registry_sha256.clone(),
            }
        }

        let matrix = PerfMatrixSpec::complete();
        let claims = claims_for(
            &matrix,
            HardwareClassId::M4Macos,
            ExecutionProfileId::Scheduler10,
            PerfGate::Qg1,
        );
        let context = super::RunnerApplicabilityContext::reconstruct(&matrix, &claims)
            .expect("exact scheduler-10 claims");
        let full = super::selected_cells(&matrix, &context, super::MatrixScale::Full, None)
            .expect("complete M4 runnable selection");
        assert_eq!(full.len(), 34);
        assert_eq!(
            full.iter()
                .filter(|cell| cell.role == EvidenceRole::Required)
                .count(),
            32
        );
        assert_eq!(
            full.iter()
                .filter(|cell| cell.role == EvidenceRole::Diagnostic)
                .count(),
            2
        );
        assert_eq!(super::configured_engine_widths(&full), vec![1, 2, 4, 8]);
        assert!(full.iter().all(|cell| cell.spec.threads != Some(10)));
        assert!(super::gate_selection_complete(
            &context,
            &full,
            super::MatrixScale::Full,
            None
        ));

        let smoke = super::selected_cells(&matrix, &context, super::MatrixScale::Smoke, None)
            .expect("smoke selection");
        assert_eq!(smoke.len(), 1);
        assert!(!super::gate_selection_complete(
            &context,
            &smoke,
            super::MatrixScale::Smoke,
            None
        ));

        let filtered = super::selected_cells(
            &matrix,
            &context,
            super::MatrixScale::Full,
            Some("bulk/tiny/1/positions_on"),
        )
        .expect("runnable partial selection");
        assert_eq!(filtered.len(), 1);
        assert!(!super::gate_selection_complete(
            &context,
            &filtered,
            super::MatrixScale::Full,
            Some("bulk/tiny/1/positions_on")
        ));

        assert!(
            super::selected_cells(
                &matrix,
                &context,
                super::MatrixScale::Full,
                Some("bulk/tiny/1/"),
            )
            .expect_err("fixture prefixes must not select a fuzzy shard")
            .contains("matched no canonical")
        );

        assert!(
            super::selected_cells(
                &matrix,
                &context,
                super::MatrixScale::Full,
                Some("bulk/tiny/16/positions_on"),
            )
            .expect_err("explicit NA fixture filter must fail")
            .contains("non-applicable")
        );
        assert!(
            super::selected_cells(
                &matrix,
                &context,
                super::MatrixScale::Full,
                Some("not-a-canonical-fixture"),
            )
            .expect_err("unknown fixture filter must fail")
            .contains("matched no canonical")
        );
    }

    pub fn assert_qg1_disjoint_partial_shard_contract() {
        use frankensearch_quill_gauntlet::{
            ExecutionProfileId, HardwareClassId, MachineClassRegistry, MachineProfileKey, PerfGate,
            PerfMatrixSpec,
        };

        let matrix = PerfMatrixSpec::complete();
        let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
        let profile = MachineProfileKey::new(
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
        )
        .expect("canonical physical-64 profile");
        let plan = matrix
            .applicability_plan(&registry, profile, PerfGate::Qg1)
            .expect("canonical physical-64 QG-1 plan");
        let execution_capacity = plan
            .execution_capacity
            .expect("physical-64 profile has typed capacity");
        let max_exercised_cell_width = plan
            .max_exercised_cell_width
            .expect("physical-64 QG-1 plan has a maximum width");
        let claims = super::RunnerPlanClaims {
            gate: PerfGate::Qg1,
            hardware_class: HardwareClassId::TrjZen35995wx,
            execution_profile: ExecutionProfileId::Physical64,
            execution_capacity,
            max_exercised_cell_width,
            rayon_num_threads: execution_capacity,
            applicability_plan_schema_version: plan.binding().schema_version.clone(),
            applicability_plan_sha256: plan.binding().applicability_plan_sha256.clone(),
            gate_matrix_contract_sha256: plan.binding().gate_matrix_contract_sha256.clone(),
            profile_contract_sha256: plan.binding().profile_contract_sha256.clone(),
            registry_schema_version: plan.binding().registry_schema_version.clone(),
            registry_sha256: plan.binding().registry_sha256.clone(),
        };
        let runner = super::RunnerApplicabilityContext::reconstruct(&matrix, &claims)
            .expect("exact physical-64 applicability plan");
        let full = super::selected_cells(&matrix, &runner, super::MatrixScale::Full, None)
            .expect("complete physical-64 QG-1 selection");
        assert!(super::gate_selection_complete(
            &runner,
            &full,
            super::MatrixScale::Full,
            None
        ));
        assert!(
            super::selected_cells(&matrix, &runner, super::MatrixScale::Full, Some("bulk/"),)
                .expect_err("fixture prefixes must not select a fuzzy shard")
                .contains("matched no canonical")
        );

        let expected_ordinals = full
            .iter()
            .map(|cell| cell.ordinal)
            .collect::<std::collections::BTreeSet<_>>();
        let mut shard_ordinals = std::collections::BTreeSet::new();
        let mut identity_probe_shards: Vec<Vec<super::PlannedPerfCell>> = Vec::new();
        for planned in &full {
            let shard = super::selected_cells(
                &matrix,
                &runner,
                super::MatrixScale::Full,
                Some(planned.spec.fixture.as_str()),
            )
            .expect("exact runnable fixture forms one real partial shard");
            assert_eq!(shard.len(), 1, "fixture filters must be disjoint");
            assert!(
                !super::gate_selection_complete(
                    &runner,
                    &shard,
                    super::MatrixScale::Full,
                    Some(planned.spec.fixture.as_str()),
                ),
                "no individual shard may become ratchet-admissible"
            );
            let (code, detail) = super::partial_shard_no_claim(PerfGate::Qg1, false)
                .expect("partial QG-1 shard has an explicit NoClaim");
            assert_eq!(code, super::QG1_PARTIAL_SHARD_NO_CLAIM_CODE);
            assert!(!detail.is_empty());
            assert!(shard_ordinals.insert(shard[0].ordinal));
            if identity_probe_shards.is_empty()
                || (identity_probe_shards.len() == 1
                    && identity_probe_shards[0][0].spec.document_count
                        != shard[0].spec.document_count)
            {
                identity_probe_shards.push(shard);
            }
        }
        assert_eq!(
            shard_ordinals, expected_ordinals,
            "the disjoint NoClaim shards collectively cover the runnable full gate"
        );
        assert!(
            super::partial_shard_no_claim(PerfGate::Qg1, true).is_none(),
            "only a single complete invocation may omit the partial-shard NoClaim"
        );
        assert_eq!(
            identity_probe_shards.len(),
            2,
            "hostile identity probes require two shards with distinct measured corpus sizes"
        );

        let first_specs = identity_probe_shards[0]
            .iter()
            .map(|cell| cell.spec.clone())
            .collect::<Vec<_>>();
        let second_specs = identity_probe_shards[1]
            .iter()
            .map(|cell| cell.spec.clone())
            .collect::<Vec<_>>();
        let first_context =
            super::BenchContext::for_selected(super::MatrixScale::Full, &first_specs);
        let second_context =
            super::BenchContext::for_selected(super::MatrixScale::Full, &second_specs);
        let (first_authoritative_hash, first_authoritative_specs) =
            super::authoritative_qg1_corpus_identity(&first_context, &matrix, &first_specs)
                .expect("first shard binds the authoritative corpus");
        let (second_authoritative_hash, second_authoritative_specs) =
            super::authoritative_qg1_corpus_identity(&second_context, &matrix, &second_specs)
                .expect("second shard binds the authoritative corpus");
        assert_eq!(
            first_authoritative_specs.len(),
            matrix.for_gate(PerfGate::Qg1).len()
        );
        assert_eq!(first_authoritative_specs, second_authoritative_specs);
        assert_eq!(
            first_authoritative_hash, second_authoritative_hash,
            "runtime shard selection must not alter the full-corpus identity"
        );
        let first_qg1_identity =
            super::corpus_identity(&first_context, &first_specs, &first_authoritative_hash);
        let second_qg1_identity =
            super::corpus_identity(&second_context, &second_specs, &second_authoritative_hash);
        assert_eq!(
            first_qg1_identity.corpus_sha256, second_qg1_identity.corpus_sha256,
            "disjoint shards must retain the same immutable full-corpus-universe seal"
        );
        assert_eq!(
            first_qg1_identity.document_count,
            first_specs[0]
                .document_count
                .expect("first shard has a measured document count")
        );
        assert_eq!(
            second_qg1_identity.document_count,
            second_specs[0]
                .document_count
                .expect("second shard has a measured document count")
        );
        assert_ne!(
            first_qg1_identity.document_count, second_qg1_identity.document_count,
            "shared full-corpus identity must not erase distinct shard-local measured counts"
        );
        assert_eq!(
            first_qg1_identity.generator_revision,
            super::QG1_CORPUS_GENERATOR_REVISION
        );
        assert_eq!(first_qg1_identity.query_set_sha256, None);

        let first_selected_only_hash = super::corpus_manifest_hash(&first_context, &first_specs)
            .expect("hash first selected-only control");
        let second_selected_only_hash = super::corpus_manifest_hash(&second_context, &second_specs)
            .expect("hash second selected-only control");
        assert_ne!(
            first_selected_only_hash, second_selected_only_hash,
            "hostile control proves selected-spec hashing would split the corpus identity"
        );
        assert_ne!(first_authoritative_hash, first_selected_only_hash);
        assert_ne!(second_authoritative_hash, second_selected_only_hash);

        let mut position_mutated_matrix = matrix.clone();
        position_mutated_matrix
            .cells
            .iter_mut()
            .find(|cell| cell.gate == PerfGate::Qg1)
            .expect("canonical QG-1 cell")
            .positions = Some(frankensearch_quill_gauntlet::PositionMode::Off);
        assert!(
            super::authoritative_qg1_corpus_identity(
                &first_context,
                &position_mutated_matrix,
                &first_specs,
            )
            .expect_err("position-contract mutation must change the authoritative identity")
            .contains("differs from frozen identity")
        );
    }

    #[test]
    fn qg1_disjoint_partial_shards_share_full_corpus_identity_and_stay_no_claim() {
        assert_qg1_disjoint_partial_shard_contract();
    }

    pub fn assert_corpus_identity_fixture_framing() {
        use sha2::{Digest, Sha256};

        let mut left_unframed = b"a".to_vec();
        left_unframed.extend_from_slice(b"bc");
        let mut right_unframed = b"ab".to_vec();
        right_unframed.extend_from_slice(b"c");
        assert_eq!(
            left_unframed, right_unframed,
            "hostile variable-field control must collide without boundary framing"
        );

        let framed_hash = |fixture: &[u8], following_field: &[u8]| {
            let mut hasher = Sha256::new();
            hasher.update(b"corpus-identity-framing-test-v1\0");
            super::hash_qg1_indexed_bytes(&mut hasher, fixture);
            hasher.update(following_field);
            super::lower_hex(&hasher.finalize())
        };
        assert_ne!(
            framed_hash(b"a", b"bc"),
            framed_hash(b"ab", b"c"),
            "length-prefixed fixture framing must separate the hostile preimages"
        );
    }

    #[test]
    fn corpus_identity_length_prefixes_variable_fixture_text() {
        assert_corpus_identity_fixture_framing();
    }

    pub fn assert_non_qg1_corpus_identity_preserves_legacy_hash() {
        use frankensearch_quill_gauntlet::{PerfGate, PerfMatrixSpec};
        use sha2::{Digest, Sha256};

        let representative = PerfMatrixSpec::complete()
            .for_gate(PerfGate::Qg6)
            .into_iter()
            .next()
            .expect("canonical QG-6 cell")
            .clone();
        let cells = [representative];
        let mut legacy = Sha256::new();
        for cell in &cells {
            legacy.update(cell.fixture.as_bytes());
            legacy.update(
                super::MatrixScale::Full
                    .document_count(cell.document_count.unwrap_or_default())
                    .to_le_bytes(),
            );
            legacy.update(super::CORPUS_SEED.to_le_bytes());
            legacy.update(super::VOCABULARY_SIZE.to_le_bytes());
            legacy.update(super::MAX_DOCUMENT_BYTES.to_le_bytes());
        }
        let expected = super::lower_hex(&legacy.finalize());
        let actual = super::hash_corpus_identity_cells(
            super::MatrixScale::Full,
            &cells,
            &std::collections::BTreeMap::new(),
        )
        .expect("hash representative non-QG-1 corpus identity");
        assert_eq!(
            actual, expected,
            "the QG-1 identity revision must not alter a non-QG-1 corpus hash"
        );
    }

    #[test]
    fn non_qg1_corpus_identity_preserves_legacy_hash() {
        assert_non_qg1_corpus_identity_preserves_legacy_hash();
    }

    #[test]
    fn runner_selection_rejects_matrix_or_plan_cell_mutation() {
        use frankensearch_quill_gauntlet::{
            ExecutionProfileId, HardwareClassId, MachineClassRegistry, MachineProfileKey, PerfGate,
            PerfMatrixSpec,
        };

        fn claims_for(
            matrix: &PerfMatrixSpec,
            hardware_class: HardwareClassId,
            execution_profile: ExecutionProfileId,
            gate: PerfGate,
        ) -> super::RunnerPlanClaims {
            let registry = MachineClassRegistry::frozen().expect("frozen machine registry");
            let profile = MachineProfileKey::new(hardware_class, execution_profile)
                .expect("canonical typed profile key");
            let plan = matrix
                .applicability_plan(&registry, profile, gate)
                .expect("canonical applicability plan");
            let execution_capacity = plan
                .execution_capacity
                .expect("typed promotion profile capacity");
            let max_exercised_cell_width = plan
                .max_exercised_cell_width
                .expect("typed promotion gate maximum");
            super::RunnerPlanClaims {
                gate,
                hardware_class,
                execution_profile,
                execution_capacity,
                max_exercised_cell_width,
                rayon_num_threads: execution_capacity,
                applicability_plan_schema_version: plan.binding().schema_version.clone(),
                applicability_plan_sha256: plan.binding().applicability_plan_sha256.clone(),
                gate_matrix_contract_sha256: plan.binding().gate_matrix_contract_sha256.clone(),
                profile_contract_sha256: plan.binding().profile_contract_sha256.clone(),
                registry_schema_version: plan.binding().registry_schema_version.clone(),
                registry_sha256: plan.binding().registry_sha256.clone(),
            }
        }

        let matrix = PerfMatrixSpec::complete();
        let claims = claims_for(
            &matrix,
            HardwareClassId::TrjZen35995wx,
            ExecutionProfileId::Physical64,
            PerfGate::Qg1,
        );
        let mut context = super::RunnerApplicabilityContext::reconstruct(&matrix, &claims)
            .expect("exact physical plan");
        context.plan.cells[0].cell_contract_sha256 = "0".repeat(64);
        assert!(
            super::selected_cells(&matrix, &context, super::MatrixScale::Full, None)
                .expect_err("mutated plan cell")
                .contains("does not match")
        );

        let mut mutated_matrix = matrix;
        mutated_matrix.cells[0].fixture.push_str("/mutated");
        assert!(
            super::RunnerApplicabilityContext::reconstruct(&mutated_matrix, &claims)
                .expect_err("mutated canonical matrix")
                .contains("cannot reconstruct runner applicability plan")
        );
    }

    #[test]
    fn qg1_prepared_corpus_prefixes_replay_exact_manifests_and_share_one_materialization() {
        let prepared = super::PreparedQg1Corpus::from_effective_counts([12, 40, 12])
            .expect("prepare distinct QG-1 corpus prefixes");
        let (short_prefix, short_documents) = prepared.prefix(12);
        let (long_prefix, long_documents) = prepared.prefix(40);

        assert_eq!(
            short_prefix.manifest,
            super::corpus_for(12).manifest().expect("short manifest")
        );
        assert_eq!(
            long_prefix.manifest,
            super::corpus_for(40).manifest().expect("long manifest")
        );
        assert_eq!(short_documents.len(), 12);
        assert_eq!(long_documents.len(), 40);
        assert_eq!(
            short_documents.as_ptr(),
            long_documents.as_ptr(),
            "every prepared prefix must borrow the same immutable allocation"
        );
        assert_eq!(
            std::sync::Arc::strong_count(&prepared.documents),
            1,
            "prepared prefixes must not clone the corpus allocation"
        );
        let replayed =
            super::PreparedQg1Corpus::from_effective_counts([40]).expect("replay QG-1 corpus");
        let (replayed_prefix, _) = replayed.prefix(40);
        assert_eq!(
            long_prefix.indexed_content_sha256, replayed_prefix.indexed_content_sha256,
            "indexed-content identity must ignore HashMap iteration order"
        );

        let expected = frankensearch_core::IndexableDocument::from(
            super::corpus_for(40).document_at(11).expect("document 11"),
        );
        let actual = short_documents
            .get(11)
            .expect("short prepared prefix contains document 11");
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.content, expected.content);
        assert_eq!(actual.title, expected.title);
        assert_eq!(actual.metadata, expected.metadata);

        let generated = super::corpus_for(40)
            .document_at(17)
            .expect("document for move proof");
        let id_pointer = generated.id.as_ptr();
        let content_pointer = generated.content.as_ptr();
        let title_pointer = generated.title.as_ref().map(|title| title.as_ptr());
        let converted = frankensearch_core::IndexableDocument::from(generated);
        assert_eq!(converted.id.as_ptr(), id_pointer);
        assert_eq!(converted.content.as_ptr(), content_pointer);
        assert_eq!(
            converted.title.as_ref().map(|title| title.as_ptr()),
            title_pointer
        );
    }

    #[test]
    fn qg1_corpus_identity_rejects_mutated_indexed_documents() {
        let cell = frankensearch_quill_gauntlet::PerfMatrixSpec::complete()
            .cells
            .into_iter()
            .find(|cell| cell.gate == frankensearch_quill_gauntlet::PerfGate::Qg1)
            .expect("normative matrix has a QG-1 cell");
        assert_eq!(cell.fixture, "bulk/tiny/1/positions_on");
        let mut context =
            super::BenchContext::for_selected(super::MatrixScale::Smoke, &[cell.clone()]);
        super::corpus_manifest_hash(&context, &[cell.clone()])
            .expect("verified QG-1 corpus identity");

        let documents = std::sync::Arc::get_mut(
            &mut context
                .prepared_qg1
                .as_mut()
                .expect("prepared QG-1 corpus")
                .documents,
        )
        .expect("QG-1 test corpus has one owner");
        documents
            .first_mut()
            .expect("normative QG-1 corpus is nonempty")
            .content
            .push_str(" adversarial-indexed-content-tamper");

        let error = super::corpus_manifest_hash(&context, &[cell])
            .expect_err("mutated indexed documents must fail closed");
        assert!(error.contains("indexed content"));
    }

    #[test]
    fn qg1_corpus_identity_rejects_mutated_verified_manifest() {
        let cell = frankensearch_quill_gauntlet::PerfMatrixSpec::complete()
            .cells
            .into_iter()
            .find(|cell| cell.gate == frankensearch_quill_gauntlet::PerfGate::Qg1)
            .expect("normative matrix has a QG-1 cell");
        let effective_count = super::MatrixScale::Smoke.document_count(
            cell.document_count
                .expect("normative QG-1 cell has a document count"),
        );
        let mut context =
            super::BenchContext::for_selected(super::MatrixScale::Smoke, &[cell.clone()]);
        super::corpus_manifest_hash(&context, &[cell.clone()])
            .expect("verified QG-1 corpus identity");

        let manifest = &mut context
            .prepared_qg1
            .as_mut()
            .expect("prepared QG-1 corpus")
            .prefixes
            .get_mut(&effective_count)
            .expect("prepared normative QG-1 prefix")
            .manifest;
        let replacement = if manifest.content_sha256.starts_with('0') {
            "1"
        } else {
            "0"
        };
        manifest.content_sha256.replace_range(..1, replacement);
        let error = super::corpus_manifest_hash(&context, &[cell])
            .expect_err("mutated verified manifest must fail closed");
        assert!(error.contains("manifest identity"));
    }

    #[test]
    fn typed_qg6_query_count_contract_accepts_exact_value_and_rejects_mismatch() {
        let exact = "[gate.QG-6]\nqueries_per_class = 16\n";
        super::validate_qg6_queries_per_class(exact).expect("exact typed query count");

        let mismatch = "[gate.QG-6]\nqueries_per_class = 15\n";
        assert!(
            super::validate_qg6_queries_per_class(mismatch)
                .expect_err("mismatched typed query count")
                .contains("differs from runner constant")
        );
        assert!(super::validate_qg6_queries_per_class("[gate.QG-6]\n").is_err());
        assert!(
            super::validate_qg6_queries_per_class(
                "[gate.QG-6]\nqueries_per_class = \"sixteen\"\n",
            )
                .is_err()
        );
    }

    #[test]
    fn normative_manifest_gate_contract_is_complete_and_closed() {
        assert_manifest_gate_contract();
    }

    pub fn assert_manifest_gate_contract() {
        super::validate_manifest_gate_contract(super::MANIFEST)
            .expect("normative manifest has every required QG policy");

        let missing = super::MANIFEST.replacen("[gate.QG-10]", "[omitted.QG-10]", 1);
        let missing_error = super::validate_manifest_gate_contract(&missing)
            .expect_err("missing QG policy must fail closed");
        assert!(
            missing_error.contains("missing gate.QG-10"),
            "unexpected missing-gate error: {missing_error}"
        );

        let extra = format!(
            "{}\n[gate.QG-11]\nname = \"extra\"\nfixture = \"extra\"\ntarget = \"extra\"\nactivated = false\n",
            super::MANIFEST
        );
        let extra_error = super::validate_manifest_gate_contract(&extra)
            .expect_err("extra QG policy must fail closed");
        assert!(
            extra_error.contains("unexpected gate.QG-11"),
            "unexpected extra-gate error: {extra_error}"
        );

        let empty_target = super::MANIFEST.replacen(
            "target = \"open() <= 50ms (manifest + lazy sections) vs oracle reader open\"",
            "target = \" \"",
            1,
        );
        let target_error = super::validate_manifest_gate_contract(&empty_target)
            .expect_err("empty QG target must fail closed");
        assert!(
            target_error.contains("gate.QG-9.target is empty"),
            "unexpected empty-target error: {target_error}"
        );
    }
}
