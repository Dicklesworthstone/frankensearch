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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use asupersync::{Cx, runtime::Runtime};
use criterion::Criterion;
use frankensearch_core::bench_support::print_bench_elf_sha256;
use frankensearch_core::{IndexableDocument, LexicalRead, LexicalWrite};
use frankensearch_lexical::{BenchmarkWriterJoinReceipt, SnippetConfig, TantivyIndex};
use frankensearch_quill::scribe::{FrankensearchTokenizer, TokenAnalyzer};
use frankensearch_quill::{
    Analyzer, CompactionPolicy, DEFAULT_SCHEMA, FieldDescriptor, FieldKind, QuillConfig,
    QuillIndex, SchemaDescriptor,
};
use frankensearch_quill_gauntlet::{
    BuildIdentity, ColdCacheEvidence, ComparatorConfig, ComparisonStatus, CorpusIdentity,
    CorpusManifest, CountState, DistributionSummary, EngineConcurrencyObservation,
    EngineObservation, EvidenceCell, EvidenceCellSpec, EvidencePolicy, EvidenceProvenance,
    EvidenceRole, ExecutionProfileId, HardwareClassId, MachineClassRegistry, MachineIdentity,
    MachineProfileKey, NativeTieKey, PERF_ARTIFACT_SCHEMA_VERSION, PERF_MIN_RUNS,
    PairedEstimatorConfig, PeakRssEvidence, PerfApplicabilityPlan, PerfCellApplicability,
    PerfCellResult, PerfCellSpec, PerfConcurrencyEngine, PerfConcurrencyObserver,
    PerfConcurrencyWitness, PerfCorpus, PerfEvidenceArtifact, PerfGate, PerfGateArtifact,
    PerfInputIdentity, PerfMatrixSpec, PerfMetricSemantics, PerfOperationScope, PerfQueryClass,
    PerfRawSample, PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance,
    PerfTopology, PositionMode, QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS, Qg6ArmRole, Qg6Comparison,
    Qg6Phase, Qg6PreparedExperiment, Qg6QuerySpec, Qg6SampleBinding, Qg6SampleOrder, Qg6SearchHit,
    Qg6SearchResult, Qg6SemanticContract, RankClass, RankedHit, ScoreEpsilonReason,
    SyntheticCorpus, SyntheticCorpusSpec, ZipfExponent, command_sha256_from_argv,
    compare_observations, estimate_paired_experiment, machine_fingerprint, oracle_version_contract,
    peak_rss_bytes, perf_manifest_contract_sha256, seeded_balanced_pair_order, validate_matrix,
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
const QG1_TERMINAL_QUERY: &str = "singleton";
const QG1_TERMINAL_DOCUMENT_ID: &str = "synthetic-00000002";
const QG1_TERMINAL_NO_CLAIM_CODE: &str = "qg1.terminal_fact_unproved";
const QG1_TIMING_DIAGNOSTIC_NO_CLAIM_CODE: &str = "qg1.continuous_timing_unbound_diagnostic";
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
    queries_per_class: Option<usize>,
}

fn qg6_queries_per_class(manifest: &str) -> Result<usize, String> {
    let manifest = toml::from_str::<GateManifest>(manifest).map_err(|error| error.to_string())?;
    manifest
        .gate
        .get("QG-6")
        .and_then(|gate| gate.queries_per_class)
        .ok_or_else(|| "gate.QG-6.queries_per_class is missing".to_owned())
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
    let mut observations = COLD_CACHE_OBSERVATIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock cold-cache observations");
    let entry = observations.entry(cell_id).or_default();
    let (successes, failures) = match arm {
        EngineArm::Quill => (&mut entry.quill_successes, &mut entry.quill_failures),
        EngineArm::Tantivy => (&mut entry.tantivy_successes, &mut entry.tantivy_failures),
    };
    match eviction {
        Ok(file_count) => {
            assert!(
                file_count > 0,
                "QG-9 cache eviction accepted an empty index"
            );
            *successes = successes.saturating_add(1);
        }
        Err(error) => {
            eprintln!(
                "[quill-qg9-cold-cache] arm={} eviction_unverified={error}",
                arm.label()
            );
            *failures = failures.saturating_add(1);
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Qg1TerminalFact {
    Proven { proof: &'static str },
    NoClaim { code: &'static str, detail: String },
}

impl Qg1TerminalFact {
    const fn proven(proof: &'static str) -> Self {
        Self::Proven { proof }
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qg1BatchTiming {
    document_start: u64,
    document_count: u64,
    generation_started_ns: u64,
    generation_completed_ns: Option<u64>,
    feed_completed_ns: Option<u64>,
    visibility_commit_completed_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qg1ContinuousTimingReceipt {
    producer_coverage: Qg1ProducerCoverage,
    arm: EngineArm,
    document_count: u64,
    interval_started_ns: u64,
    corpus_constructed_ns: u64,
    batches: Vec<Qg1BatchTiming>,
    quill_publication_generation_delta: Option<u64>,
    terminal_commit_completed_ns: u64,
    post_commit_join_completed_ns: Option<u64>,
    terminal_search_attempt_completed_ns: u64,
    terminal_idle_join_completed_ns: Option<u64>,
    terminal_quiescence_completed_ns: u64,
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
        if self.batches.is_empty() {
            return Err("QG-1 continuous interval contains no generated/feed batches".to_owned());
        }

        let mut cursor = self.corpus_constructed_ns;
        let mut next_document = 0_u64;
        for batch in &self.batches {
            if batch.document_start != next_document || batch.document_count == 0 {
                return Err("QG-1 batch coverage is not contiguous and positive".to_owned());
            }
            let generated = batch.generation_completed_ns.ok_or_else(|| {
                "QG-1 batch is missing its generation-complete boundary".to_owned()
            })?;
            let fed = batch
                .feed_completed_ns
                .ok_or_else(|| "QG-1 batch is missing its feed-complete boundary".to_owned())?;
            if batch.generation_started_ns < cursor
                || generated < batch.generation_started_ns
                || fed < generated
            {
                return Err(
                    "QG-1 generation/feed phases escape monotonic interval order".to_owned(),
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
        if let Some(joined) = self.post_commit_join_completed_ns {
            if joined < cursor {
                return Err("QG-1 post-commit join preceded terminal commit".to_owned());
            }
            cursor = joined;
        }
        if self.terminal_search_attempt_completed_ns < cursor {
            return Err("QG-1 terminal search preceded commit/merge quiescence".to_owned());
        }
        cursor = self.terminal_search_attempt_completed_ns;
        if let Some(joined) = self.terminal_idle_join_completed_ns {
            if joined < cursor {
                return Err("QG-1 terminal idle-writer join preceded searchability".to_owned());
            }
            cursor = joined;
        }
        if self.terminal_quiescence_completed_ns < cursor
            || self.interval_ended_ns < self.terminal_quiescence_completed_ns
        {
            return Err("QG-1 terminal quiescence escaped the continuous interval".to_owned());
        }
        match self.arm {
            EngineArm::Quill => {
                if self.post_commit_join_completed_ns.is_some()
                    || self.terminal_idle_join_completed_ns.is_some()
                    || self.quill_publication_generation_delta.is_none()
                {
                    return Err(
                        "QG-1 Quill receipt names an impossible external worker-join lifecycle"
                            .to_owned(),
                    );
                }
            }
            EngineArm::Tantivy => {
                if self.post_commit_join_completed_ns.is_none()
                    || self.terminal_idle_join_completed_ns.is_none()
                    || self.quill_publication_generation_delta.is_some()
                {
                    return Err(
                        "QG-1 Tantivy receipt lacks its post-commit and terminal joins".to_owned(),
                    );
                }
            }
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

struct Qg1ContinuousInterval {
    origin: Instant,
    arm: EngineArm,
    document_count: u64,
    corpus_constructed_ns: Option<u64>,
    batches: Vec<Qg1BatchTiming>,
    terminal_commit_completed_ns: Option<u64>,
    post_commit_join_completed_ns: Option<u64>,
    terminal_search_attempt_completed_ns: Option<u64>,
    terminal_idle_join_completed_ns: Option<u64>,
    terminal_quiescence_completed_ns: Option<u64>,
}

impl Qg1ContinuousInterval {
    fn start(arm: EngineArm, document_count: u64) -> Self {
        assert!(
            document_count > 2,
            "QG-1 continuous sample requires its immutable terminal sentinel"
        );
        Self {
            origin: Instant::now(),
            arm,
            document_count,
            corpus_constructed_ns: None,
            batches: Vec::new(),
            terminal_commit_completed_ns: None,
            post_commit_join_completed_ns: None,
            terminal_search_attempt_completed_ns: None,
            terminal_idle_join_completed_ns: None,
            terminal_quiescence_completed_ns: None,
        }
    }

    fn elapsed_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn mark_corpus_constructed(&mut self) {
        assert!(
            self.corpus_constructed_ns
                .replace(self.elapsed_ns())
                .is_none(),
            "QG-1 corpus construction boundary repeated"
        );
    }

    fn begin_batch(&mut self, document_start: u64, document_count: u64) -> u64 {
        let generation_started_ns = self.elapsed_ns();
        self.batches.push(Qg1BatchTiming {
            document_start,
            document_count,
            generation_started_ns,
            generation_completed_ns: None,
            feed_completed_ns: None,
            visibility_commit_completed_ns: None,
        });
        generation_started_ns
    }

    fn mark_batch_generated(&mut self) -> u64 {
        let completed = self.elapsed_ns();
        let batch = self
            .batches
            .last_mut()
            .expect("QG-1 generation completion requires an active batch");
        assert!(
            batch.generation_completed_ns.replace(completed).is_none(),
            "QG-1 batch generation boundary repeated"
        );
        completed
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

    fn mark_post_commit_join(&mut self) {
        let completed = self.elapsed_ns();
        assert!(
            self.post_commit_join_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 post-commit join boundary repeated"
        );
    }

    fn mark_terminal_search_attempt(&mut self) {
        let completed = self.elapsed_ns();
        assert!(
            self.terminal_search_attempt_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 terminal search boundary repeated"
        );
    }

    fn mark_terminal_idle_join(&mut self) {
        let completed = self.elapsed_ns();
        assert!(
            self.terminal_idle_join_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 terminal idle-writer join boundary repeated"
        );
    }

    fn mark_terminal_quiescence(&mut self) {
        let completed = self.elapsed_ns();
        assert!(
            self.terminal_quiescence_completed_ns
                .replace(completed)
                .is_none(),
            "QG-1 terminal quiescence boundary repeated"
        );
    }

    fn finish(
        self,
        quill_publication_generation_delta: Option<u64>,
        terminal_searchability: Qg1TerminalFact,
        terminal_quiescence: Qg1TerminalFact,
    ) -> (Duration, Qg1ContinuousTimingReceipt) {
        let elapsed = self.origin.elapsed();
        let interval_ended_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let receipt = Qg1ContinuousTimingReceipt {
            producer_coverage: Qg1ProducerCoverage::EngineIndexingLifecycle,
            arm: self.arm,
            document_count: self.document_count,
            interval_started_ns: 0,
            corpus_constructed_ns: self
                .corpus_constructed_ns
                .expect("QG-1 continuous interval includes corpus construction"),
            batches: self.batches,
            quill_publication_generation_delta,
            terminal_commit_completed_ns: self
                .terminal_commit_completed_ns
                .expect("QG-1 continuous interval includes terminal commit"),
            post_commit_join_completed_ns: self.post_commit_join_completed_ns,
            terminal_search_attempt_completed_ns: self
                .terminal_search_attempt_completed_ns
                .expect("QG-1 continuous interval includes terminal search"),
            terminal_idle_join_completed_ns: self.terminal_idle_join_completed_ns,
            terminal_quiescence_completed_ns: self
                .terminal_quiescence_completed_ns
                .expect("QG-1 continuous interval includes terminal quiescence"),
            interval_ended_ns,
            terminal_searchability,
            terminal_quiescence,
        };
        receipt
            .validate()
            .expect("invalid QG-1 continuous timing receipt");
        (elapsed, receipt)
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
        admission_no_claim_detail: "diagnostic phase trace is not bound into PerfRawSample or the \
                                    H2 assembler and cannot independently support a QG-1 claim",
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

fn feed_qg1_generated_batches<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    document_count: u64,
    manual_visibility_commit_cadence: Option<Duration>,
    interval: &mut Qg1ContinuousInterval,
) -> usize {
    let corpus = corpus_for(document_count);
    interval.mark_corpus_constructed();
    let cadence_ns = manual_visibility_commit_cadence
        .map(|cadence| u64::try_from(cadence.as_nanos()).unwrap_or(u64::MAX));
    let mut unpublished_since_ns = None;
    let mut periodic_commits = 0_usize;
    let mut start = 0_u64;
    while start < document_count {
        let remaining = document_count - start;
        let count = usize::try_from(remaining.min(context.scale.batch_documents() as u64))
            .expect("bounded QG-1 batch count");
        let count_u64 = u64::try_from(count).expect("QG-1 batch count fits u64");
        interval.begin_batch(start, count_u64);
        let documents = generated_batch(&corpus, start, count, None);
        let generated_ns = interval.mark_batch_generated();
        if cadence_ns.is_some() {
            unpublished_since_ns.get_or_insert(generated_ns);
        }
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, &documents)
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
        start = start.saturating_add(count_u64);
    }
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

fn qg1_terminal_searchability<E: LexicalRead>(
    context: &BenchContext,
    index: &E,
) -> Qg1TerminalFact {
    match context
        .runtime
        .block_on(index.search(&context.cx, QG1_TERMINAL_QUERY, 3))
    {
        Ok(results) => {
            let document_ids = results
                .into_iter()
                .map(|result| String::from(result.doc_id))
                .collect::<Vec<_>>();
            let fact = if document_ids == [QG1_TERMINAL_DOCUMENT_ID] {
                Qg1TerminalFact::proven("exact_immutable_sentinel_visible")
            } else {
                Qg1TerminalFact::no_claim(format!(
                    "terminal query {QG1_TERMINAL_QUERY:?} returned {document_ids:?} instead of \
                     [{QG1_TERMINAL_DOCUMENT_ID:?}]"
                ))
            };
            black_box(document_ids);
            fact
        }
        Err(error) => Qg1TerminalFact::no_claim(format!(
            "terminal query {QG1_TERMINAL_QUERY:?} failed: {error}"
        )),
    }
}

fn qg1_tantivy_quiescence_fact(
    post_commit: &BenchmarkWriterJoinReceipt,
    terminal_idle: &BenchmarkWriterJoinReceipt,
) -> Qg1TerminalFact {
    let post_join_segments = post_commit.searchable_segments_after;
    let terminal_join_segments_before = terminal_idle.searchable_segments_before;
    let terminal_join_segments_after = terminal_idle.searchable_segments_after;
    if post_commit.writer_rearmed
        && !terminal_idle.writer_rearmed
        && post_join_segments > 0
        && terminal_join_segments_before == post_join_segments
        && terminal_join_segments_after == terminal_join_segments_before
    {
        Qg1TerminalFact::proven("post_commit_join_search_then_unchanged_terminal_idle_join")
    } else {
        Qg1TerminalFact::no_claim(format!(
            "Tantivy terminal lifecycle did not prove stable post-join searchable segments: \
             post_commit=({},{},rearmed={}) terminal_idle=({},{},rearmed={})",
            post_commit.searchable_segments_before,
            post_commit.searchable_segments_after,
            post_commit.writer_rearmed,
            terminal_idle.searchable_segments_before,
            terminal_idle.searchable_segments_after,
            terminal_idle.writer_rearmed,
        ))
    }
}

fn qg1_bulk_metric_continuous(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    count: u64,
) -> f64 {
    assert_eq!(
        qg1_producer_coverage(spec),
        Some(Qg1ProducerCoverage::EngineIndexingLifecycle),
        "continuous QG-1 engine lifecycle is reserved for docs_per_second indexing arms"
    );
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
            let mut interval = Qg1ContinuousInterval::start(arm, count);
            let periodic_commit_calls =
                feed_qg1_generated_batches(context, &index, count, None, &mut interval);
            let generation_before_terminal = index.snapshot().loaded_manifest().manifest.generation;
            qg1_terminal_commit(context, &index, &mut interval);
            let terminal_searchability = qg1_terminal_searchability(context, &index);
            interval.mark_terminal_search_attempt();
            interval.mark_terminal_quiescence();
            let generation_delta = generation_before_terminal.saturating_sub(generation_before);
            let (elapsed, receipt) = interval.finish(
                Some(generation_delta),
                terminal_searchability,
                Qg1TerminalFact::proven("awaited_quill_inline_publication_and_tier_merges"),
            );
            emit_qg1_continuous_timing_receipt(spec, receipt);
            eprintln!(
                "[qg-commit-parity] gate={} fixture={} arm=quill cadence_ms={} \
                 explicit_periodic_commit_calls={} automatic_publication_generation_delta={} \
                 terminal_commit_calls=1 \
                 post_commit_join_calls=0 terminal_search_calls=1 terminal_idle_join_calls=0 \
                 durability=in_memory continuous_elapsed_ns={}",
                spec.gate,
                spec.fixture,
                quill_config(spec).max_visibility_lag_ms,
                periodic_commit_calls,
                generation_delta,
                elapsed.as_nanos(),
            );
            elapsed
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            let observed_threads = index
                .benchmark_materialized_writer_threads()
                .expect("QG-1 Tantivy arm uses the benchmark writer constructor");
            record_concurrency(spec, arm, observed_threads);
            let mut interval = Qg1ContinuousInterval::start(arm, count);
            let periodic_commits = feed_qg1_generated_batches(
                context,
                &index,
                count,
                Some(Duration::from_millis(
                    quill_config(spec).max_visibility_lag_ms,
                )),
                &mut interval,
            );
            qg1_terminal_commit(context, &index, &mut interval);
            let (index, post_commit_receipt) = index
                .benchmark_join_workers_and_rearm(
                    spec.writer_heap_bytes.unwrap_or(50_000_000),
                    spec.threads.unwrap_or(1),
                )
                .expect("join QG-1 Tantivy workers after terminal commit and rearm for search");
            interval.mark_post_commit_join();
            let terminal_searchability = qg1_terminal_searchability(context, &index);
            interval.mark_terminal_search_attempt();
            let terminal_idle_receipt = index
                .benchmark_join_workers()
                .expect("join QG-1 Tantivy idle terminal writer without rearming");
            interval.mark_terminal_idle_join();
            let terminal_quiescence =
                qg1_tantivy_quiescence_fact(&post_commit_receipt, &terminal_idle_receipt);
            interval.mark_terminal_quiescence();
            let (elapsed, receipt) =
                interval.finish(None, terminal_searchability, terminal_quiescence);
            emit_tantivy_lifecycle_receipt(spec, "qg1_post_commit_join", &post_commit_receipt);
            emit_tantivy_lifecycle_receipt(spec, "qg1_terminal_idle_join", &terminal_idle_receipt);
            emit_qg1_continuous_timing_receipt(spec, receipt);
            eprintln!(
                "[qg-commit-parity] gate={} fixture={} arm=tantivy cadence_ms={} \
                 explicit_periodic_commit_calls={periodic_commits} terminal_commit_calls=1 \
                 post_commit_join_calls=1 terminal_search_calls=1 terminal_idle_join_calls=1 \
                 durability=in_memory continuous_elapsed_ns={}",
                spec.gate,
                spec.fixture,
                quill_config(spec).max_visibility_lag_ms,
                elapsed.as_nanos(),
            );
            elapsed
        }
    };
    count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
}

fn bulk_metric_unpooled(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let requested = spec.document_count.expect("bulk document count");
    let count = context.scale.document_count(requested);
    if spec.gate == PerfGate::Qg1 {
        return qg1_bulk_metric_continuous(context, spec, arm, count);
    }
    let prepared_qg1_documents = (spec.gate == PerfGate::Qg1).then(|| context.qg1_prefix(count).1);
    let generated_corpus = (spec.gate != PerfGate::Qg1).then(|| corpus_for(count));
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
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
            let generation_after = index.snapshot().loaded_manifest().manifest.generation;
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
    count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
}

fn bulk_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    if !matches!(spec.gate, PerfGate::Qg1 | PerfGate::Qg8) || arm != EngineArm::Quill {
        return bulk_metric_unpooled(context, spec, arm);
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
            bulk_metric_unpooled(context, spec, arm)
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

fn measure_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    measure_metric_with_query(context, spec, arm, None)
}

fn measure_metric_with_query(
    context: &BenchContext,
    spec: &PerfCellSpec,
    arm: EngineArm,
    query_override: Option<&str>,
) -> f64 {
    match spec.gate {
        PerfGate::Qg1 if spec.metric == "tokenize_docs_per_second" => {
            tokenize_metric(context, spec)
        }
        PerfGate::Qg1 | PerfGate::Qg2 | PerfGate::Qg8 => bulk_metric(context, spec, arm),
        PerfGate::Qg3 if spec.metric == "docs_per_second" => bulk_metric(context, spec, arm),
        PerfGate::Qg3 => watch_metric(context, spec, arm),
        PerfGate::Qg4 => commit_metric(context, spec, arm),
        PerfGate::Qg5 => compaction_metric(context, spec, arm),
        PerfGate::Qg6 => query_metric(context, spec, arm, query_override),
        PerfGate::Qg7 => memory_metric(context, spec, arm),
        PerfGate::Qg9 => cold_open_metric(context, spec, arm),
        PerfGate::Qg10 => dependency_surface_metric(),
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

fn metric_semantics(metric: &str) -> PerfMetricSemantics {
    match metric {
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
        semantics: metric_semantics(&spec.metric),
        unit: unit(spec).to_owned(),
    }
}

fn raw_sample_work(context: &BenchContext, spec: &PerfCellSpec) -> (Option<u64>, Option<u64>) {
    if qg1_producer_coverage(spec).is_none() {
        return (None, None);
    }
    let document_count = context
        .scale
        .document_count(spec.document_count.expect("QG-1 throughput document count"));
    let content_bytes = context.qg1_prefix(document_count).0.content_bytes;
    assert!(
        content_bytes > 0,
        "QG-1 throughput sample requires positive immutable content bytes"
    );
    (Some(document_count), Some(content_bytes))
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
    rounds: usize,
    seed: u64,
    block_id_base: u64,
    sample_id_base: u64,
    group_id: Option<u64>,
    query_override: Option<&'a str>,
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
    ) -> Self {
        let order =
            seeded_balanced_pair_order(plan.rounds, plan.seed).expect("paired order schedule");
        let (work_units, byte_count) = raw_sample_work(context, spec);
        for _ in 0..evidence.policy.warmup_rounds {
            let _ = black_box(measure_metric_with_query(
                context,
                spec,
                plan.control,
                plan.query_override,
            ));
            let _ = black_box(measure_metric_with_query(
                context,
                spec,
                plan.treatment,
                plan.query_override,
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
                PerfSampleArm::Control,
                PerfSampleOrder::First,
                block_id,
                control_sample_id,
            );
            self.samples.push(first);
            let second = self.execute(
                self.plan.treatment,
                PerfSampleArm::Treatment,
                PerfSampleOrder::Second,
                block_id,
                treatment_sample_id,
            );
            self.samples.push(second);
        } else {
            let first = self.execute(
                self.plan.treatment,
                PerfSampleArm::Treatment,
                PerfSampleOrder::First,
                block_id,
                treatment_sample_id,
            );
            self.samples.push(first);
            let second = self.execute(
                self.plan.control,
                PerfSampleArm::Control,
                PerfSampleOrder::Second,
                block_id,
                control_sample_id,
            );
            self.samples.push(second);
        }
    }

    fn execute(
        &self,
        engine: EngineArm,
        sample_arm: PerfSampleArm,
        sample_order: PerfSampleOrder,
        block_id: u64,
        sample_id: u64,
    ) -> PerfRawSample {
        let started_ns = u64::try_from(self.origin.elapsed().as_nanos()).expect("monotonic ns");
        let value = black_box(measure_metric_with_query(
            self.context,
            self.spec,
            engine,
            self.plan.query_override,
        ));
        let mut ended_ns = u64::try_from(self.origin.elapsed().as_nanos()).expect("monotonic ns");
        if ended_ns <= started_ns {
            ended_ns = started_ns + 1;
        }
        PerfRawSample {
            block_id,
            sample_id,
            arm: sample_arm,
            order: sample_order,
            phase: PerfSamplePhase::Measurement,
            scope: self.scope.clone(),
            provenance: self.evidence.sample_provenance.clone(),
            started_ns,
            ended_ns,
            work_units: self.work_units,
            byte_count: self.byte_count,
            observed_value: Some(value),
            group_id: self.plan.group_id,
            qg6_sample_binding: None,
        }
    }

    fn into_samples(self) -> Vec<PerfRawSample> {
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
        }),
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

struct CellCollection {
    results: Vec<PerfCellResult>,
    evidence: Option<EvidenceCell>,
}

fn collect_cell(
    context: &BenchContext,
    spec: &PerfCellSpec,
    role: EvidenceRole,
    runs: usize,
    evidence: &EvidenceContext,
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
        };
    }

    let scope = operation_scope(spec);
    let origin = Instant::now();
    let cell_seed = evidence.config.bootstrap_seed ^ fixture_seed(&spec.fixture);

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
                rounds: runs,
                seed: cell_seed ^ 0xaa,
                block_id_base: 0,
                sample_id_base: 1_000_000,
                group_id: None,
                query_override: None,
            },
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
                    rounds: runs,
                    seed: cell_seed ^ 0x55,
                    block_id_base: 2_000_000,
                    sample_id_base: 2_000_000,
                    group_id: None,
                    query_override: None,
                },
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
                rounds: runs,
                seed: cell_seed,
                block_id_base: 0,
                sample_id_base: 0,
                group_id: None,
                query_override: None,
            },
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

    let experiment =
        estimate_paired_experiment(&effect_samples, &oracle_null_samples, &evidence.config)
            .expect("paired estimator rejected harness-produced streams");
    let treatment_null_experiment = treatment_null_samples.as_ref().map(|samples| {
        estimate_paired_experiment(&effect_samples, samples, &evidence.config)
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
        cell.attach_treatment_arm_null(treatment_null_experiment, &evidence.policy)
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
                    let value = black_box(measure_metric(context, spec, arm));
                    total += metric_duration(context, spec, value);
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

fn bench_matrix(c: &mut Criterion, bench_elf_sha256: &str) {
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
    preflight_indexing_fixtures(&context, &matrix, &selected_specs);
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
    for planned in &selected {
        let spec = &planned.spec;
        let collection = collect_cell(
            &context,
            spec,
            planned.role,
            configured_runs,
            &evidence_context,
        );
        by_gate
            .entry(spec.gate)
            .or_default()
            .extend(collection.results);
        if let Some(cell) = collection.evidence {
            evidence_by_gate.entry(spec.gate).or_default().push(cell);
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
        let selection_no_claim = partial_shard_no_claim(gate, selection_complete);
        let terminal_no_claim = (gate == PerfGate::Qg1)
            .then(qg1_terminal_no_claim_detail)
            .flatten();
        match (selection_no_claim, terminal_no_claim) {
            (Some((code, detail)), Some(terminal_detail)) => {
                artifact.force_no_claim(code, format!("{detail}; additionally, {terminal_detail}"));
            }
            (Some((code, detail)), None) => artifact.force_no_claim(code, detail),
            (None, Some(detail)) => artifact.force_no_claim(QG1_TERMINAL_NO_CLAIM_CODE, detail),
            (None, None) => {}
        }
        let paths = artifact
            .write_atomic(&output_dir)
            .expect("write QG evidence artifact");
        let reloaded = PerfEvidenceArtifact::load_verified(&paths.json)
            .expect("reload and verify persisted QG evidence artifact");
        assert_eq!(
            reloaded, artifact,
            "persisted QG evidence artifact must reload as the exact sealed source object"
        );
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
    };
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
    #[cfg(test)]
    if std::env::var_os("QUILL_PERF_H1_PRODUCER_SELF_CHECK").is_some() {
        tests::assert_qg1_continuous_interval_contract();
        tests::assert_qg1_real_terminal_visibility_contract();
        tests::assert_qg1_raw_sample_work_contract();
        tests::assert_qg1_disjoint_partial_shard_contract();
        tests::assert_corpus_identity_fixture_framing();
        tests::assert_non_qg1_corpus_identity_preserves_legacy_hash();
        tests::assert_qg9_cache_evidence_contract();
        tests::assert_qg9_cache_eviction_file_discovery();
        eprintln!(
            "[quill-perf-self-check] H1 immutable producer and continuous-timing contracts passed"
        );
        return;
    }
    if run_child_mode() {
        return;
    }
    let identity = print_bench_elf_sha256().expect("hash executing QG benchmark");
    // Fail closed before a single cell is timed.
    let _oracle = assert_incumbent_is_genuine_tantivy();
    let mut criterion = Criterion::default().configure_from_args();
    bench_matrix(&mut criterion, &identity.sha256);
    criterion.final_summary();
}

#[cfg(test)]
mod tests {
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

    fn hostile_tantivy_continuous_receipt() -> super::Qg1ContinuousTimingReceipt {
        super::Qg1ContinuousTimingReceipt {
            producer_coverage: super::Qg1ProducerCoverage::EngineIndexingLifecycle,
            arm: super::EngineArm::Tantivy,
            document_count: 20,
            interval_started_ns: 0,
            corpus_constructed_ns: 3,
            batches: vec![
                super::Qg1BatchTiming {
                    document_start: 0,
                    document_count: 10,
                    generation_started_ns: 5,
                    generation_completed_ns: Some(15),
                    feed_completed_ns: Some(25),
                    visibility_commit_completed_ns: None,
                },
                super::Qg1BatchTiming {
                    document_start: 10,
                    document_count: 10,
                    generation_started_ns: 40,
                    generation_completed_ns: Some(50),
                    feed_completed_ns: Some(65),
                    visibility_commit_completed_ns: Some(75),
                },
            ],
            quill_publication_generation_delta: None,
            terminal_commit_completed_ns: 100,
            post_commit_join_completed_ns: Some(125),
            terminal_search_attempt_completed_ns: 140,
            terminal_idle_join_completed_ns: Some(155),
            terminal_quiescence_completed_ns: 160,
            interval_ended_ns: 170,
            terminal_searchability: super::Qg1TerminalFact::proven(
                "exact_immutable_sentinel_visible",
            ),
            terminal_quiescence: super::Qg1TerminalFact::proven(
                "post_commit_join_search_then_unchanged_terminal_idle_join",
            ),
        }
    }

    pub fn assert_qg1_continuous_interval_contract() {
        let receipt = hostile_tantivy_continuous_receipt();
        receipt
            .validate()
            .expect("hostile timeline is one valid continuous interval");

        // This control is deliberately generous to the retired implementation:
        // it sums every feed/commit/join/search call represented in the hostile
        // timeline. It still loses corpus construction, both generation phases,
        // and every gap between calls, exactly the undercount caused by adding
        // independent `Instant::elapsed()` results.
        let old_summed_call_ns = (25 - 15)
            + (65 - 50)
            + (75 - 65)
            + (100 - 90)
            + (125 - 100)
            + (140 - 125)
            + (155 - 140);
        assert_eq!(old_summed_call_ns, 100);
        assert_eq!(receipt.interval_ended_ns, 170);
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
        let mut generation_escape = receipt.clone();
        generation_escape.batches[0].generation_started_ns = 171;
        assert_escape_rejected(generation_escape);
        let mut feed_escape = receipt.clone();
        feed_escape.batches[1].feed_completed_ns = Some(171);
        assert_escape_rejected(feed_escape);
        let mut commit_escape = receipt.clone();
        commit_escape.terminal_commit_completed_ns = 69;
        assert_escape_rejected(commit_escape);
        let mut post_commit_join_escape = receipt.clone();
        post_commit_join_escape.post_commit_join_completed_ns = Some(99);
        assert_escape_rejected(post_commit_join_escape);
        let mut search_escape = receipt.clone();
        search_escape.terminal_search_attempt_completed_ns = 124;
        assert_escape_rejected(search_escape);
        let mut terminal_join_escape = receipt.clone();
        terminal_join_escape.terminal_idle_join_completed_ns = Some(139);
        assert_escape_rejected(terminal_join_escape);
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
        for arm in [super::EngineArm::Quill, super::EngineArm::Tantivy] {
            let throughput = super::bulk_metric(&context, &spec, arm);
            assert!(
                throughput.is_finite() && throughput > 0.0,
                "real {arm:?} terminal sample must return positive finite throughput"
            );
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
                    proof: "exact_immutable_sentinel_visible"
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
}
