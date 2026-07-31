//! Same-binary Quill/Tantivy performance matrix for QG-1 through QG-10.
//!
//! The default invocation is deliberately a one-cell smoke slice. A release
//! evidence run selects one gate (and optionally one fixture substring), then
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

use std::collections::{BTreeMap, BTreeSet};
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
    QuillIndex, SchemaDescriptor, SegmentStatsProvider,
};
use frankensearch_quill_gauntlet::{
    BuildIdentity, ByteStageObservation, CONTINUOUS_TIMING_SCHEMA_VERSION, ColdCacheEvidence,
    CollectorBinding, ComparatorConfig, ComparisonStatus, ContinuousCellEvidence,
    ContinuousCorpusManifest, ContinuousPhaseTimeline, ContinuousSampleIdentity,
    ContinuousSampleWindow, ContinuousTimingEvidence, ContinuousWindowReceipt, CorpusIdentity,
    CorpusManifest, CountState, DistributionSummary, EngineByteObservation,
    EngineConcurrencyObservation, EngineObservation, EngineQuiescence, EvidenceCell,
    EvidenceCellSpec, EvidencePolicy, EvidenceProvenance, EvidenceRole, LifecycleObserver,
    LifecyclePhase, MachineIdentity, NativeTieKey, NoopLifecycleObserver,
    PERF_ARTIFACT_SCHEMA_VERSION, PERF_MIN_RUNS, PairedEstimatorConfig, PeakRssEvidence,
    PerfCellResult, PerfCellSpec, PerfConcurrencyEngine, PerfConcurrencyObserver,
    PerfConcurrencyWitness, PerfCorpus, PerfEvidenceArtifact, PerfGate, PerfGateArtifact,
    PerfInputIdentity, PerfMatrixSpec, PerfMetricSemantics, PerfOperationScope, PerfQueryClass,
    PerfRawSample, PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance,
    PerfTopology, PositionMode, QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS, Qg6ArmRole, Qg6Comparison,
    Qg6Phase, Qg6PreparedExperiment, Qg6QuerySpec, Qg6SampleBinding, Qg6SampleOrder, Qg6SearchHit,
    Qg6SearchResult, Qg6SelectionScope, Qg6SemanticContract, QueueObservation, RankClass,
    RankedHit, ScoreEpsilonReason, SyntheticCorpus, SyntheticCorpusSpec, TerminalJoin, TimingMode,
    TimingSource, WORK_RECEIPT_SCHEMA_VERSION, WidthObservation, WorkReceipt,
    WorkReceiptCellEvidence, WorkReceiptCollector, WorkReceiptEvidence, WorkReceiptExpectation,
    WorkReceiptMode, ZipfExponent, command_sha256_from_argv, compare_observations,
    continuous_raw_sample, continuous_throughput_scope, document_bytes, estimate_paired_experiment,
    machine_fingerprint, oracle_version_contract, peak_rss_bytes, perf_manifest_contract_sha256,
    seeded_balanced_pair_order, validate_matrix,
};
use serde::Deserialize;
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
/// Invocation-wide timing mode (QG-1 H1). Read once; per-call by default.
static TIMING_MODE: OnceLock<TimingMode> = OnceLock::new();
/// Continuous-mode per-fixture evidence accumulator.
static CONTINUOUS_CELL_STATE: OnceLock<Mutex<BTreeMap<String, ContinuousCellState>>> =
    OnceLock::new();
/// Hand-off slot: the continuous window measured by the innermost bulk runner,
/// consumed by the paired-stream sample builder on the coordinating thread.
static CONTINUOUS_SAMPLE_CAPTURE: OnceLock<Mutex<Option<ContinuousSampleCapture>>> =
    OnceLock::new();
/// Invocation-wide work-receipt mode (QG-1 H2). Read once; off by default.
static WORK_RECEIPT_MODE: OnceLock<WorkReceiptMode> = OnceLock::new();
/// Monotonic anchor all receipt windows are expressed against.
static WORK_RECEIPT_ORIGIN: OnceLock<Instant> = OnceLock::new();
/// Run identity sealed by `bench_matrix` before any receipted collection.
static WORK_RECEIPT_IDENTITY: OnceLock<WorkReceiptRunIdentity> = OnceLock::new();
/// Per-fixture work-receipt accumulator (bounded: counts + last per arm).
static WORK_RECEIPT_STATE: OnceLock<Mutex<BTreeMap<String, WorkReceiptCellState>>> =
    OnceLock::new();
/// Every sealed receipt of the invocation, flushed as JSONL post-collection.
static WORK_RECEIPT_LOG: OnceLock<Mutex<Vec<serde_json::Value>>> = OnceLock::new();
static CONCURRENCY_OBSERVATIONS: OnceLock<
    Mutex<BTreeMap<(String, String), ConcurrencyAccumulator>>,
> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
struct ConcurrencyAccumulator {
    count: usize,
    min: usize,
    max: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

fn qg1_indexed_content_sha256(documents: &[IndexableDocument]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch-quill-qg1-indexable-documents-v1\0");
    hasher.update(
        u64::try_from(documents.len())
            .expect("QG-1 indexed document count fits u64")
            .to_le_bytes(),
    );
    let mut metadata = Vec::new();
    for document in documents {
        hash_qg1_indexed_bytes(&mut hasher, document.id.as_bytes());
        match &document.title {
            Some(title) => {
                hasher.update([1]);
                hash_qg1_indexed_bytes(&mut hasher, title.as_bytes());
            }
            None => hasher.update([0]),
        }
        hash_qg1_indexed_bytes(&mut hasher, document.content.as_bytes());

        metadata.extend(document.metadata.iter());
        metadata.sort_unstable_by(|(left_key, left_value), (right_key, right_value)| {
            left_key
                .cmp(right_key)
                .then_with(|| left_value.cmp(right_value))
        });
        hasher.update(
            u64::try_from(metadata.len())
                .expect("QG-1 indexed metadata count fits u64")
                .to_le_bytes(),
        );
        for (key, value) in &metadata {
            hash_qg1_indexed_bytes(&mut hasher, key.as_bytes());
            hash_qg1_indexed_bytes(&mut hasher, value.as_bytes());
        }
        metadata.clear();
    }
    lower_hex(&hasher.finalize())
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
                (
                    count,
                    PreparedQg1Prefix {
                        manifest,
                        manifest_sha256,
                        indexed_content_sha256: qg1_indexed_content_sha256(indexed_documents),
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
        let observed_indexed_content_sha256 = qg1_indexed_content_sha256(indexed_documents);
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

fn finish_tantivy_lifecycle(
    index: TantivyIndex,
    spec: &PerfCellSpec,
    phase: &str,
) -> (Duration, BenchmarkWriterJoinReceipt) {
    let receipt = index
        .benchmark_join_workers()
        .expect("join Tantivy benchmark workers without rearming");
    assert!(
        !receipt.writer_rearmed,
        "terminal Tantivy lifecycle fence unexpectedly rearmed a writer"
    );
    emit_tantivy_lifecycle_receipt(spec, phase, &receipt);
    (Duration::from_nanos(receipt.join_elapsed_ns), receipt)
}

// ─── QG-1 H2: actual-work, queue, worker-role, and lifecycle receipts ────────
//
// Opt-in via `QUILL_PERF_WORK_RECEIPTS=on`, scoped to QG-1 docs_per_second
// cells (the only cells with a prepared immutable corpus to bind against).
// The collector observes at the harness↔engine boundary and through the
// engines' real accessors; anything no seam exposes stays a typed gap. The
// same collector implements the H1 `LifecycleObserver` seam and plugs into
// `bulk_metric_continuous` with the same phase clock and
// `timing_mode: "continuous"`.

/// Run identity every receipt binds (sealed once per invocation).
struct WorkReceiptRunIdentity {
    run_id: String,
    machine_fingerprint: String,
    build_profile: String,
    executable_sha256: String,
    git_rev: String,
}

/// Bounded per-fixture accumulator entry.
struct WorkReceiptCellState {
    gate: PerfGate,
    rounds_quill: u64,
    rounds_tantivy: u64,
    last_quill: Option<WorkReceipt>,
    last_tantivy: Option<WorkReceipt>,
    all_validated: bool,
}

fn work_receipt_mode() -> WorkReceiptMode {
    *WORK_RECEIPT_MODE
        .get_or_init(|| WorkReceiptMode::from_env().expect("QUILL_PERF_WORK_RECEIPTS"))
}

/// Whether this cell collects H2 work receipts.
fn work_receipt_cell(spec: &PerfCellSpec) -> bool {
    work_receipt_mode().is_enabled()
        && spec.gate == PerfGate::Qg1
        && spec.metric == "docs_per_second"
}

fn work_receipt_origin_elapsed_ns() -> u64 {
    u64::try_from(
        WORK_RECEIPT_ORIGIN
            .get_or_init(Instant::now)
            .elapsed()
            .as_nanos(),
    )
    .expect("monotonic ns")
}

/// First alphanumeric token of the prepared corpus, lowercased with the same
/// folding the analyzers apply: a term guaranteed searchable after the
/// terminal commit.
fn work_receipt_probe_term(documents: &[IndexableDocument]) -> String {
    documents
        .iter()
        .find_map(|document| {
            document
                .content
                .split(|c: char| !c.is_alphanumeric())
                .find(|token| !token.is_empty())
        })
        .expect("prepared corpus contains at least one searchable token")
        .to_ascii_lowercase()
}

/// Build the receipt collector for one window of one arm.
fn work_receipt_collector(
    spec: &PerfCellSpec,
    arm: EngineArm,
    prefix: &PreparedQg1Prefix,
) -> WorkReceiptCollector {
    let identity = WORK_RECEIPT_IDENTITY
        .get()
        .expect("work-receipt identity sealed before any receipted collection");
    WorkReceiptCollector::new(
        CollectorBinding {
            run_id: identity.run_id.clone(),
            machine_fingerprint: identity.machine_fingerprint.clone(),
            build_profile: identity.build_profile.clone(),
            executable_sha256: identity.executable_sha256.clone(),
            git_rev: identity.git_rev.clone(),
            gate: spec.gate.to_string(),
            fixture: spec.fixture.clone(),
            metric: spec.metric.clone(),
            engine: arm.label().to_owned(),
            timing_mode: timing_mode().label().to_owned(),
            corpus_identity: format!(
                "qg1-native/prepared-prefix-v1/{}",
                prefix.indexed_content_sha256
            ),
            corpus_manifest_sha256: prefix.manifest_sha256.clone(),
        },
        u64::try_from(spec.threads.expect("QG-1 bulk cell thread width")).expect("width fits u64"),
    )
}

fn lifecycle_observer<'a>(
    collector: Option<&'a mut WorkReceiptCollector>,
    noop: &'a mut NoopLifecycleObserver,
) -> &'a mut dyn LifecycleObserver {
    collector.map_or(noop as &mut dyn LifecycleObserver, |collector| {
        collector as &mut dyn LifecycleObserver
    })
}

/// Quill's honest engine-reported footprint: exact FSLX byte lengths from
/// the published manifest (backend-independent — `managed_disk_bytes` is
/// hard-zero on the in-memory backend and must not be used here).
fn quill_index_footprint(index: &QuillIndex) -> (EngineByteObservation, EngineByteObservation) {
    let snapshot = index.snapshot();
    let manifest = &snapshot.loaded_manifest().manifest;
    let bytes: u64 = manifest
        .segments
        .iter()
        .map(|segment| segment.file_len)
        .sum();
    (
        EngineByteObservation::Observed {
            bytes,
            seam: "quill KeeperSnapshot::loaded_manifest().manifest.segments[].file_len \
                   (exact FSLX lengths, backend-independent)"
                .to_owned(),
        },
        EngineByteObservation::Observed {
            bytes: manifest.segments.len() as u64,
            seam: "quill KeeperSnapshot::loaded_manifest().manifest.segments.len()".to_owned(),
        },
    )
}

/// Tantivy's honest engine-reported footprint via the bench layout seam.
fn tantivy_index_footprint(index: &TantivyIndex) -> (EngineByteObservation, EngineByteObservation) {
    match index.benchmark_index_layout() {
        Ok((segments, bytes)) => (
            EngineByteObservation::Observed {
                bytes,
                seam: "TantivyIndex::benchmark_index_layout (managed segment files via \
                       Directory::open_read)"
                    .to_owned(),
            },
            EngineByteObservation::Observed {
                bytes: segments as u64,
                seam: "TantivyIndex::benchmark_index_layout searchable segment count".to_owned(),
            },
        ),
        Err(error) => {
            let seam = format!("TantivyIndex::benchmark_index_layout failed: {error}");
            (
                EngineByteObservation::StructurallyUnobservable { seam: seam.clone() },
                EngineByteObservation::StructurallyUnobservable { seam },
            )
        }
    }
}

/// Finish, validate fail-closed, log, and record one receipt.
fn finalize_work_receipt(
    collector: WorkReceiptCollector,
    spec: &PerfCellSpec,
    arm: EngineArm,
    expected_docs: u64,
    expected_bytes: u64,
    window_started_rel_ns: u64,
    expected_wall_ns: Option<u64>,
) {
    let receipt = collector
        .finish(window_started_rel_ns)
        .expect("assemble QG-1 work receipt");
    if let Some(expected_wall_ns) = expected_wall_ns {
        assert_eq!(
            receipt.concurrency.wall_ns, expected_wall_ns,
            "H1 continuous window and H2 work receipt used different wall clocks"
        );
    }
    let expectation = WorkReceiptExpectation {
        engine: arm.label().to_owned(),
        doc_count: expected_docs,
        total_bytes: expected_bytes,
        configured_threads: u64::try_from(spec.threads.expect("QG-1 bulk cell thread width"))
            .expect("width fits u64"),
    };
    let validated = receipt.validate(&expectation);
    eprintln!("{}", receipt.bounded_log_line());
    let mut state = WORK_RECEIPT_STATE
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock work-receipt state");
    let cell = state
        .entry(spec.fixture.clone())
        .or_insert_with(|| WorkReceiptCellState {
            gate: spec.gate,
            rounds_quill: 0,
            rounds_tantivy: 0,
            last_quill: None,
            last_tantivy: None,
            all_validated: true,
        });
    cell.all_validated &= validated.is_ok();
    match arm {
        EngineArm::Quill => {
            cell.rounds_quill += 1;
            cell.last_quill = Some(receipt.clone());
        }
        EngineArm::Tantivy => {
            cell.rounds_tantivy += 1;
            cell.last_tantivy = Some(receipt.clone());
        }
    }
    drop(state);
    WORK_RECEIPT_LOG
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("lock work-receipt log")
        .push(serde_json::to_value(&receipt).expect("serialize work receipt"));
    validated.expect("work receipt violated the H2 contract");
}

/// Drain the accumulator into per-gate additive artifact evidence.
fn take_work_receipt_evidence() -> BTreeMap<PerfGate, WorkReceiptEvidence> {
    let Some(state) = WORK_RECEIPT_STATE.get() else {
        return BTreeMap::new();
    };
    let drained = {
        let mut state = state.lock().expect("lock work-receipt state for drain");
        std::mem::take(&mut *state)
    };
    let mut by_gate: BTreeMap<PerfGate, WorkReceiptEvidence> = BTreeMap::new();
    for (fixture, cell) in drained {
        by_gate
            .entry(cell.gate)
            .or_insert_with(|| WorkReceiptEvidence {
                schema_version: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
                mode: work_receipt_mode().label().to_owned(),
                cells: Vec::new(),
            })
            .cells
            .push(WorkReceiptCellEvidence {
                schema_version: WORK_RECEIPT_SCHEMA_VERSION.to_owned(),
                fixture,
                rounds_quill: cell.rounds_quill,
                rounds_tantivy: cell.rounds_tantivy,
                last_quill_receipt: cell.last_quill,
                last_tantivy_receipt: cell.last_tantivy,
                all_receipts_validated: cell.all_validated,
            });
    }
    by_gate
}

/// Flush every sealed receipt as JSONL alongside the lifecycle receipts.
fn flush_work_receipts(output_dir: &Path) {
    let Some(log) = WORK_RECEIPT_LOG.get() else {
        return;
    };
    let (payload, receipt_count) = {
        let log = log.lock().expect("lock work-receipt log for flush");
        if log.is_empty() {
            return;
        }
        let mut payload = Vec::new();
        for row in log.iter() {
            serde_json::to_writer(&mut payload, row).expect("serialize work receipt row");
            payload.push(b'\n');
        }
        (payload, log.len())
    };
    std::fs::create_dir_all(output_dir).expect("create work-receipt directory");
    let path = output_dir.join("work-receipts.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("open work-receipt JSONL");
    file.write_all(&payload).expect("write work receipts");
    eprintln!(
        "[qg1-work-receipts] receipts={} sha256={} path={}",
        receipt_count,
        lower_hex(&Sha256::digest(&payload)),
        display_path(&path),
    );
}

// ─── QG-1 H1: continuous first-feed-to-quiescence timing ────────────────────
//
// Corpus immutability is provided by the prepared QG-1 corpus (bd-6oiq): one
// Arc-backed materialization, exact prefix manifests, and a domain-separated
// SHA-256 over the actual indexed documents, all fail-closed rechecked before
// evidence collection. This block adds the remaining H1 contract on top: one
// continuous monotonic window per sample (first feed → terminal searchable
// commit → engine quiescence join), accepted/processed/committed/searchable
// work-equality receipts, and derived-throughput samples — opt-in via
// `QUILL_PERF_TIMING_MODE=continuous`.

/// Continuous-mode per-fixture accumulator entry.
struct ContinuousCellState {
    gate: PerfGate,
    document_count: u64,
    manifest: ContinuousCorpusManifest,
    rounds_quill: u64,
    rounds_tantivy: u64,
    last_quill: Option<ContinuousWindowReceipt>,
    last_tantivy: Option<ContinuousWindowReceipt>,
    all_windows_validated: bool,
}

/// One measured continuous window, handed from the bulk runner to the
/// paired-stream sample builder.
#[derive(Clone, Copy)]
struct ContinuousSampleCapture {
    started: Instant,
    ended: Instant,
    work_units: u64,
    byte_count: u64,
}

fn timing_mode() -> TimingMode {
    *TIMING_MODE.get_or_init(|| TimingMode::from_env().expect("QUILL_PERF_TIMING_MODE"))
}

/// Whether this cell measures bulk indexing under the continuous H1 contract.
///
/// Scoped to QG-1 engine-indexing cells: only QG-1 has the prepared immutable
/// corpus this window feeds from, and the tokenizer-only null keeps gauge
/// semantics because it exercises no engine with background workers.
fn continuous_bulk_cell(spec: &PerfCellSpec) -> bool {
    timing_mode().is_continuous() && spec.gate == PerfGate::Qg1 && spec.metric == "docs_per_second"
}

fn elapsed_ns(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).expect("monotonic ns")
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).expect("duration ns")
}

/// Seal (first use) or fetch the continuous receipt manifest for one fixture.
///
/// The manifest binds document count, byte totals, field occurrences, order,
/// and an xxh3 digest over the exact prepared prefix this cell feeds; its
/// identity embeds the prepared corpus's indexed-content SHA-256 so the
/// receipt chain is anchored to the same bytes bd-6oiq verified.
fn continuous_manifest(
    spec: &PerfCellSpec,
    indexed_content_sha256: &str,
    documents: &[IndexableDocument],
) -> ContinuousCorpusManifest {
    let mut state = CONTINUOUS_CELL_STATE
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock continuous cell state");
    state
        .entry(spec.fixture.clone())
        .or_insert_with(|| {
            let manifest = ContinuousCorpusManifest::seal(
                &format!("qg1-native/prepared-prefix-v1/{indexed_content_sha256}"),
                CORPUS_SEED,
                spec.positions.unwrap_or(PositionMode::On).label(),
                documents,
            )
            .expect("seal QG-1-native continuous corpus manifest");
            eprintln!(
                "[qg1-continuous-corpus] fixture={} identity={} docs={} bytes={} xxh3={}",
                spec.fixture,
                manifest.identity,
                manifest.doc_count,
                manifest.total_bytes,
                manifest.corpus_xxh3,
            );
            ContinuousCellState {
                gate: spec.gate,
                document_count: manifest.doc_count,
                manifest,
                rounds_quill: 0,
                rounds_tantivy: 0,
                last_quill: None,
                last_tantivy: None,
                all_windows_validated: true,
            }
        })
        .manifest
        .clone()
}

fn record_continuous_window(spec: &PerfCellSpec, arm: EngineArm, receipt: ContinuousWindowReceipt) {
    let mut state = CONTINUOUS_CELL_STATE
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .expect("lock continuous cell state");
    let cell = state
        .get_mut(&spec.fixture)
        .expect("continuous window recorded before its manifest was sealed");
    match arm {
        EngineArm::Quill => {
            cell.rounds_quill += 1;
            cell.last_quill = Some(receipt);
        }
        EngineArm::Tantivy => {
            cell.rounds_tantivy += 1;
            cell.last_tantivy = Some(receipt);
        }
    }
    drop(state);
}

/// Drain the continuous accumulator into per-gate artifact evidence.
///
/// Post-collection bookend to the prepared corpus's pre-collection identity
/// recheck: every fixture's manifest is re-verified against the live prefix
/// slice after all windows ran, fail-closed.
fn take_continuous_evidence(
    context: &BenchContext,
) -> BTreeMap<PerfGate, ContinuousTimingEvidence> {
    let Some(state) = CONTINUOUS_CELL_STATE.get() else {
        return BTreeMap::new();
    };
    let drained = {
        let mut state = state.lock().expect("lock continuous cell state for drain");
        std::mem::take(&mut *state)
    };
    let mut by_gate: BTreeMap<PerfGate, ContinuousTimingEvidence> = BTreeMap::new();
    for (fixture, cell) in drained {
        let (_, documents) = context.qg1_prefix(cell.document_count);
        cell.manifest
            .verify(documents)
            .expect("continuous corpus mutated between first window and evidence drain");
        by_gate
            .entry(cell.gate)
            .or_insert_with(|| ContinuousTimingEvidence {
                schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
                timing_mode: timing_mode().label().to_owned(),
                cells: Vec::new(),
            })
            .cells
            .push(ContinuousCellEvidence {
                schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
                fixture,
                timing_mode: timing_mode().label().to_owned(),
                corpus: cell.manifest,
                rounds_quill: cell.rounds_quill,
                rounds_tantivy: cell.rounds_tantivy,
                last_quill_receipt: cell.last_quill,
                last_tantivy_receipt: cell.last_tantivy,
                all_windows_validated: cell.all_windows_validated,
            });
    }
    by_gate
}

fn clear_continuous_capture() {
    CONTINUOUS_SAMPLE_CAPTURE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("lock continuous sample capture")
        .take();
}

fn store_continuous_capture(capture: ContinuousSampleCapture) {
    // Warmup rounds and Criterion presentation iterations measure without
    // consuming the capture, so replacement is deliberate; the paired stream
    // guards pairing by clearing before and take-expecting after each
    // measurement.
    CONTINUOUS_SAMPLE_CAPTURE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("lock continuous sample capture")
        .replace(capture);
}

fn take_continuous_capture() -> Option<ContinuousSampleCapture> {
    CONTINUOUS_SAMPLE_CAPTURE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("lock continuous sample capture")
        .take()
}

/// First alphanumeric token found in the corpus (documents may include
/// pathology anchors with no tokenizable content), lowercased with the same
/// folding the analyzer applies: a term guaranteed searchable after the
/// terminal commit.
fn continuous_probe_term(documents: &[IndexableDocument]) -> String {
    documents
        .iter()
        .find_map(|document| {
            document
                .content
                .split(|c: char| !c.is_alphanumeric())
                .find(|token| !token.is_empty())
        })
        .expect("prepared corpus contains at least one searchable token")
        .to_ascii_lowercase()
}

/// Feed totals accumulated inside one continuous window.
struct ContinuousFeedTotals {
    feed_calls: u64,
    accepted_docs: u64,
    processed_docs: u64,
    accepted_bytes: u64,
    processed_bytes: u64,
    percall_ns: u64,
    periodic_commits: u64,
}

/// Feed every prepared batch back-to-back inside the timed window.
///
/// There is no generation between calls; the only work between two feed
/// calls is the loop bookkeeping itself, so the continuous clock never
/// credits either engine with untimed progress. Per-call durations are still
/// recorded (into `percall_ns`) purely as the side-by-side legacy view.
fn continuous_feed<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    documents: &[IndexableDocument],
    visibility_cadence: Option<Duration>,
    observer: &mut dyn LifecycleObserver,
) -> ContinuousFeedTotals {
    let mut totals = ContinuousFeedTotals {
        feed_calls: 0,
        accepted_docs: 0,
        processed_docs: 0,
        accepted_bytes: 0,
        processed_bytes: 0,
        percall_ns: 0,
        periodic_commits: 0,
    };
    let mut unpublished_since: Option<Instant> = None;
    for chunk in documents.chunks(context.scale.batch_documents()) {
        // Observed at hand-off: bytes summed over the actual documents of
        // this batch, not copied from the manifest denominator.
        let chunk_bytes: u64 = chunk.iter().map(document_bytes).sum();
        totals.feed_calls += 1;
        totals.accepted_docs += chunk.len() as u64;
        totals.accepted_bytes += chunk_bytes;
        let call = Instant::now();
        let publish_started = *unpublished_since.get_or_insert(call);
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, chunk)
                .await
                .expect("QG continuous feed batch");
        });
        totals.percall_ns += duration_ns(call.elapsed());
        totals.processed_docs += chunk.len() as u64;
        totals.processed_bytes += chunk_bytes;
        observer.on_feed_batch(chunk.len() as u64, chunk_bytes);
        if let Some(cadence) = visibility_cadence
            && publish_started.elapsed() >= cadence
        {
            let commit_call = Instant::now();
            context.runtime.block_on(async {
                index
                    .commit(&context.cx)
                    .await
                    .expect("QG continuous visibility commit");
            });
            totals.percall_ns += duration_ns(commit_call.elapsed());
            totals.periodic_commits += 1;
            unpublished_since = None;
        }
    }
    totals
}

/// Phase results of the engine-generic window segment (feed through
/// searchable verification). Quiescence is arm-specific and completed by the
/// caller before the clock stops.
struct ContinuousWindowCore {
    feed: ContinuousFeedTotals,
    feed_complete_ns: u64,
    commit_complete_ns: u64,
    committed_docs: u64,
    searchable_docs: u64,
    searchable_verified_ns: u64,
    probe_hits: u64,
    percall_ns: u64,
}

fn continuous_window_core<E: LexicalWrite + LexicalRead>(
    context: &BenchContext,
    index: &E,
    documents: &[IndexableDocument],
    visibility_cadence: Option<Duration>,
    probe: &str,
    window_started: Instant,
    observer: &mut dyn LifecycleObserver,
    post_feed: &mut dyn FnMut(),
) -> ContinuousWindowCore {
    observer.on_phase(LifecyclePhase::FirstFeed, 0);
    let feed = continuous_feed(context, index, documents, visibility_cadence, observer);
    // Pre-terminal sampling point: arm-specific state (Quill's snapshot
    // generation) must be read here so the terminal commit below is never
    // miscounted as a periodic visibility commit.
    post_feed();
    let feed_complete_ns = elapsed_ns(window_started);
    observer.on_phase(LifecyclePhase::FeedComplete, feed_complete_ns);
    let commit_call = Instant::now();
    context.runtime.block_on(async {
        index
            .commit(&context.cx)
            .await
            .expect("QG continuous terminal commit");
    });
    let commit_ns = duration_ns(commit_call.elapsed());
    let commit_complete_ns = elapsed_ns(window_started);
    observer.on_phase(LifecyclePhase::CommitComplete, commit_complete_ns);
    let committed_docs = u64::try_from(LexicalRead::doc_count(index)).expect("doc count fits u64");
    let probe_hits = context.runtime.block_on(async {
        index
            .search(&context.cx, probe, 1)
            .await
            .expect("QG continuous terminal searchable probe")
            .len() as u64
    });
    let searchable_docs = u64::try_from(LexicalRead::doc_count(index)).expect("doc count fits u64");
    let searchable_verified_ns = elapsed_ns(window_started);
    observer.on_phase(LifecyclePhase::SearchableVerified, searchable_verified_ns);
    ContinuousWindowCore {
        percall_ns: feed.percall_ns + commit_ns,
        feed,
        feed_complete_ns,
        commit_complete_ns,
        committed_docs,
        searchable_docs,
        searchable_verified_ns,
        probe_hits,
    }
}

/// Run one complete continuous bulk window for one arm and return the
/// derived docs/s rate. The window covers first feed through terminal
/// searchable commit *and* engine quiescence; the receipt is validated
/// fail-closed against the sealed corpus manifest before the rate is used.
fn bulk_metric_continuous(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let requested = spec.document_count.expect("bulk document count");
    let count = context.scale.document_count(requested);
    let (prefix, documents) = context.qg1_prefix(count);
    let manifest = continuous_manifest(spec, &prefix.indexed_content_sha256, documents);
    let probe = continuous_probe_term(documents);
    let mut collector = work_receipt_cell(spec).then(|| work_receipt_collector(spec, arm, prefix));
    let mut noop_observer = NoopLifecycleObserver;

    // Committed/searchable byte totals are structurally unobservable at the
    // LexicalRead seam, which exposes only a document count; the typed gap is
    // recorded explicitly and H1 stays open behind it (H2 receipts territory).
    let byte_stage_gap = || ByteStageObservation::StructurallyUnobservable {
        seam: "LexicalRead exposes doc_count only; engine-side committed/searchable byte \
               totals await the H2 actual-work receipts seam"
            .to_owned(),
    };
    let (window_started, window, core, quiescence, periodic_commits, extra_percall_ns) = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
            // Sampled by the post-feed hook, before the terminal commit, so
            // the terminal publication is never counted as periodic.
            let generation_after_feed = std::cell::Cell::new(generation_before);
            let window_started_rel_ns = collector
                .as_ref()
                .map_or(0, |_| work_receipt_origin_elapsed_ns());
            let window_started = Instant::now();
            let core = continuous_window_core(
                context,
                &index,
                documents,
                None,
                &probe,
                window_started,
                lifecycle_observer(collector.as_mut(), &mut noop_observer),
                &mut || {
                    generation_after_feed
                        .set(index.snapshot().loaded_manifest().manifest.generation);
                },
            );
            let periodic_commits = generation_after_feed
                .get()
                .saturating_sub(generation_before);
            if let Some(c) = collector.as_mut() {
                for _ in 0..periodic_commits {
                    c.record_periodic_commit();
                }
                let (index_bytes, segment_count) = quill_index_footprint(&index);
                c.record_committed(core.committed_docs, index_bytes, segment_count);
                c.record_searchable(core.searchable_docs);
                c.record_queue(QueueObservation::SynchronousNoQueue {
                    seam: "QuillIndex::index_documents/commit are synchronous; the ingest \
                           hand-off has no queue"
                        .to_owned(),
                });
                c.record_width(WidthObservation::Observed {
                    threads: u64::try_from(rayon::current_num_threads()).expect("width fits u64"),
                    seam: "rayon::current_num_threads inside the bench-pinned Quill pool"
                        .to_owned(),
                });
                c.record_measured_sum_ns(core.percall_ns);
                c.record_terminal(TerminalJoin::QuillSynchronousCommit, "completed", false);
            }
            // Quill quiescence: the terminal commit is synchronous; when it
            // returned, shards were sealed and the searchable snapshot was
            // published with no background worker still running.
            let window = window_started.elapsed();
            lifecycle_observer(collector.as_mut(), &mut noop_observer)
                .on_phase(LifecyclePhase::QuiescenceJoined, duration_ns(window));
            if let Some(c) = collector.take() {
                finalize_work_receipt(
                    c,
                    spec,
                    arm,
                    count,
                    manifest.total_bytes,
                    window_started_rel_ns,
                    Some(duration_ns(window)),
                );
            }
            eprintln!(
                "[qg-commit-parity] gate={} fixture={} arm=quill cadence_ms={} \
                 periodic_commits={periodic_commits} terminal_commit_calls=1 \
                 durability=in_memory timing_mode=continuous",
                spec.gate,
                spec.fixture,
                quill_config(spec).max_visibility_lag_ms,
            );
            (
                window_started,
                window,
                core,
                EngineQuiescence::QuillSealedSynchronousCommit,
                periodic_commits,
                0_u64,
            )
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            let observed_threads = index
                .benchmark_materialized_writer_threads()
                .expect("scaling Tantivy arm uses the benchmark writer constructor");
            record_concurrency(spec, arm, observed_threads);
            let visibility_cadence =
                Duration::from_millis(quill_config(spec).max_visibility_lag_ms);
            let window_started_rel_ns = collector
                .as_ref()
                .map_or(0, |_| work_receipt_origin_elapsed_ns());
            let window_started = Instant::now();
            let core = continuous_window_core(
                context,
                &index,
                documents,
                Some(visibility_cadence),
                &probe,
                window_started,
                lifecycle_observer(collector.as_mut(), &mut noop_observer),
                &mut || {},
            );
            let periodic_commits = core.feed.periodic_commits;
            if let Some(c) = collector.as_mut() {
                for _ in 0..periodic_commits {
                    c.record_periodic_commit();
                }
                let (index_bytes, segment_count) = tantivy_index_footprint(&index);
                c.record_committed(core.committed_docs, index_bytes, segment_count);
                c.record_searchable(core.searchable_docs);
                c.record_queue(QueueObservation::StructurallyUnobservable {
                    seam: "tantivy 0.26.1 IndexWriter AddOperation channel \
                           (crossbeam bounded(10_000)) exposes no depth accessor; observing \
                           occupancy requires patching tantivy"
                        .to_owned(),
                });
                c.record_width(WidthObservation::Observed {
                    threads: u64::try_from(observed_threads).expect("width fits u64"),
                    seam: "TantivyIndex::benchmark_materialized_writer_threads".to_owned(),
                });
            }
            // Tantivy quiescence: join every indexing worker and the merging
            // thread inside the window, so background merge work started by
            // this workload is charged to this workload.
            let (join, join_receipt) = finish_tantivy_lifecycle(index, spec, "continuous_window");
            if let Some(c) = collector.as_mut() {
                c.record_measured_sum_ns(core.percall_ns + duration_ns(join));
                c.record_terminal(
                    TerminalJoin::TantivyWorkersJoined {
                        join_elapsed_ns: join_receipt.join_elapsed_ns,
                        searchable_segments_before: join_receipt.searchable_segments_before as u64,
                        searchable_segments_after: join_receipt.searchable_segments_after as u64,
                        writer_rearmed: join_receipt.writer_rearmed,
                    },
                    "completed",
                    false,
                );
            }
            let window = window_started.elapsed();
            lifecycle_observer(collector.as_mut(), &mut noop_observer)
                .on_phase(LifecyclePhase::QuiescenceJoined, duration_ns(window));
            if let Some(c) = collector.take() {
                finalize_work_receipt(
                    c,
                    spec,
                    arm,
                    count,
                    manifest.total_bytes,
                    window_started_rel_ns,
                    Some(duration_ns(window)),
                );
            }
            eprintln!(
                "[qg-commit-parity] gate={} fixture={} arm=tantivy cadence_ms={} \
                 periodic_commits={periodic_commits} terminal_commit_calls=1 \
                 durability=in_memory timing_mode=continuous",
                spec.gate,
                spec.fixture,
                quill_config(spec).max_visibility_lag_ms,
            );
            (
                window_started,
                window,
                core,
                EngineQuiescence::TantivyMergingThreadsJoined,
                periodic_commits,
                duration_ns(join),
            )
        }
    };

    let window_total_ns = duration_ns(window);
    let receipt = ContinuousWindowReceipt {
        schema_version: CONTINUOUS_TIMING_SCHEMA_VERSION.to_owned(),
        engine: arm.label().to_owned(),
        timing_source: TimingSource::ContinuousMonotonicWindow,
        quiescence,
        feed_calls: core.feed.feed_calls,
        accepted_docs: core.feed.accepted_docs,
        processed_docs: core.feed.processed_docs,
        committed_docs: core.committed_docs,
        searchable_docs: core.searchable_docs,
        accepted_bytes: core.feed.accepted_bytes,
        processed_bytes: core.feed.processed_bytes,
        committed_bytes: byte_stage_gap(),
        searchable_bytes: byte_stage_gap(),
        periodic_commits,
        percall_sum_ns: core.percall_ns + extra_percall_ns,
        timeline: ContinuousPhaseTimeline {
            feed_complete_ns: core.feed_complete_ns,
            commit_complete_ns: core.commit_complete_ns,
            searchable_verified_ns: core.searchable_verified_ns,
            quiescence_joined_ns: window_total_ns,
            window_total_ns,
        },
        terminal_probe_query: probe,
        terminal_probe_hits: core.probe_hits,
    };
    receipt
        .validate(&manifest)
        .expect("continuous window receipt violated the H1 contract");
    let rate = receipt.docs_per_second();
    eprintln!(
        "[qg1-continuous] fixture={} arm={} docs={} accepted_bytes={} processed_bytes={} \
         window_ns={} percall_sum_ns={} feed_calls={} periodic_commits={} committed={} \
         searchable={} probe={} probe_hits={} quiescence={} manifest_sha256={} \
         indexed_content_sha256={}",
        spec.fixture,
        arm.label(),
        receipt.accepted_docs,
        receipt.accepted_bytes,
        receipt.processed_bytes,
        window_total_ns,
        receipt.percall_sum_ns,
        receipt.feed_calls,
        receipt.periodic_commits,
        receipt.committed_docs,
        receipt.searchable_docs,
        receipt.terminal_probe_query,
        receipt.terminal_probe_hits,
        receipt.quiescence.label(),
        prefix.manifest_sha256,
        prefix.indexed_content_sha256,
    );
    record_continuous_window(spec, arm, receipt);
    store_continuous_capture(ContinuousSampleCapture {
        started: window_started,
        ended: window_started + window,
        work_units: count,
        byte_count: manifest.total_bytes,
    });
    rate
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
    observe_batch: &mut dyn FnMut(&[IndexableDocument]),
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
        // Observed at the hand-off boundary, after the call returned.
        observe_batch(batch);
    }
    measured
}

fn index_prepared_qg1_batches_with_visibility_commits<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    documents: &[IndexableDocument],
    commit_cadence: Duration,
    observe_batch: &mut dyn FnMut(&[IndexableDocument]),
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
        observe_batch(batch);
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

#[allow(clippy::too_many_lines)]
fn bulk_metric_unpooled(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    if continuous_bulk_cell(spec) {
        return bulk_metric_continuous(context, spec, arm);
    }
    let requested = spec.document_count.expect("bulk document count");
    let count = context.scale.document_count(requested);
    let prepared_qg1 = (spec.gate == PerfGate::Qg1).then(|| context.qg1_prefix(count));
    let prepared_qg1_documents = prepared_qg1.map(|(_, documents)| documents);
    let generated_corpus = (spec.gate != PerfGate::Qg1).then(|| corpus_for(count));
    // QG-1 H2 receipts (opt-in): the collector observes at the hand-off
    // boundary and through engine accessors, strictly between the timed
    // regions, identically for both arms.
    let mut collector = work_receipt_cell(spec).then(|| {
        let (prefix, _) = prepared_qg1.expect("QG-1 receipted cell has a prepared corpus");
        work_receipt_collector(spec, arm, prefix)
    });
    let expected_bytes: u64 = collector.as_ref().map_or(0, |_| {
        prepared_qg1_documents
            .expect("QG-1 receipted cell has prepared documents")
            .iter()
            .map(document_bytes)
            .sum()
    });
    let mut window_started_rel_ns = 0_u64;
    let win_ns =
        |origin: Instant| u64::try_from(origin.elapsed().as_nanos()).expect("monotonic ns");
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
            if let Some(c) = collector.as_mut() {
                window_started_rel_ns = work_receipt_origin_elapsed_ns();
                c.begin_window();
            }
            let window_origin = Instant::now();
            if let Some(c) = collector.as_mut() {
                c.on_phase(LifecyclePhase::FirstFeed, 0);
            }
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
                |documents| {
                    index_prepared_qg1_batches(context, &index, documents, &mut |batch| {
                        if let Some(c) = collector.as_mut() {
                            c.record_feed_batch(
                                batch.len() as u64,
                                batch.iter().map(document_bytes).sum(),
                            );
                        }
                    })
                },
            );
            let generation_after = index.snapshot().loaded_manifest().manifest.generation;
            if let Some(c) = collector.as_mut() {
                c.on_phase(LifecyclePhase::FeedComplete, win_ns(window_origin));
            }
            elapsed += commit(context, &index);
            if let Some(c) = collector.as_mut() {
                c.on_phase(LifecyclePhase::CommitComplete, win_ns(window_origin));
            }
            let periodic_commits = generation_after.saturating_sub(generation_before);
            if spec.gate == PerfGate::Qg1 {
                eprintln!(
                    "[qg-commit-parity] gate={} fixture={} arm=quill cadence_ms={} \
                     periodic_commits={periodic_commits} terminal_commit_calls=1 \
                     durability=in_memory",
                    spec.gate,
                    spec.fixture,
                    quill_config(spec).max_visibility_lag_ms,
                );
            }
            if let Some(mut c) = collector.take() {
                let committed =
                    u64::try_from(LexicalRead::doc_count(&index)).expect("doc count fits u64");
                let (index_bytes, segment_count) = quill_index_footprint(&index);
                c.record_committed(committed, index_bytes, segment_count);
                let documents =
                    prepared_qg1_documents.expect("QG-1 receipted cell has prepared documents");
                let probe = work_receipt_probe_term(documents);
                let probe_hits = context.runtime.block_on(async {
                    index
                        .search(&context.cx, &probe, 1)
                        .await
                        .expect("QG-1 receipt terminal searchable probe")
                        .len()
                });
                assert!(probe_hits > 0, "terminal searchable probe returned no hits");
                let searchable =
                    u64::try_from(LexicalRead::doc_count(&index)).expect("doc count fits u64");
                c.record_searchable(searchable);
                c.on_phase(LifecyclePhase::SearchableVerified, win_ns(window_origin));
                for _ in 0..periodic_commits {
                    c.record_periodic_commit();
                }
                c.record_queue(QueueObservation::SynchronousNoQueue {
                    seam: "QuillIndex::index_documents/commit are synchronous; the ingest \
                           hand-off has no queue"
                        .to_owned(),
                });
                c.record_width(WidthObservation::Observed {
                    threads: u64::try_from(rayon::current_num_threads()).expect("width fits u64"),
                    seam: "rayon::current_num_threads inside the bench-pinned Quill pool"
                        .to_owned(),
                });
                // Quill quiescence: the terminal commit is synchronous; when
                // it returned there was no background worker left to join.
                c.record_terminal(TerminalJoin::QuillSynchronousCommit, "completed", false);
                c.on_phase(LifecyclePhase::QuiescenceJoined, win_ns(window_origin));
                c.record_measured_sum_ns(
                    u64::try_from(elapsed.as_nanos()).expect("measured ns fits u64"),
                );
                finalize_work_receipt(
                    c,
                    spec,
                    arm,
                    count,
                    expected_bytes,
                    window_started_rel_ns,
                    None,
                );
            }
            elapsed
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            let observed_threads = if matches!(spec.gate, PerfGate::Qg1 | PerfGate::Qg8) {
                let threads = index
                    .benchmark_materialized_writer_threads()
                    .expect("scaling Tantivy arm uses the benchmark writer constructor");
                record_concurrency(spec, arm, threads);
                Some(threads)
            } else {
                None
            };
            if let Some(c) = collector.as_mut() {
                window_started_rel_ns = work_receipt_origin_elapsed_ns();
                c.begin_window();
            }
            let window_origin = Instant::now();
            if let Some(c) = collector.as_mut() {
                c.on_phase(LifecyclePhase::FirstFeed, 0);
            }
            let (mut elapsed, periodic_commits) = if spec.gate == PerfGate::Qg1 {
                index_prepared_qg1_batches_with_visibility_commits(
                    context,
                    &index,
                    prepared_qg1_documents.expect("QG-1 bulk cell has a prepared immutable corpus"),
                    Duration::from_millis(quill_config(spec).max_visibility_lag_ms),
                    &mut |batch| {
                        if let Some(c) = collector.as_mut() {
                            c.record_feed_batch(
                                batch.len() as u64,
                                batch.iter().map(document_bytes).sum(),
                            );
                        }
                    },
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
            if let Some(c) = collector.as_mut() {
                c.on_phase(LifecyclePhase::FeedComplete, win_ns(window_origin));
            }
            elapsed += commit(context, &index);
            if let Some(c) = collector.as_mut() {
                c.on_phase(LifecyclePhase::CommitComplete, win_ns(window_origin));
            }
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
            if let Some(c) = collector.as_mut() {
                // Pre-join observations: the post-commit worker generation,
                // segment updater, and merge threads are still alive here.
                let committed =
                    u64::try_from(LexicalRead::doc_count(&index)).expect("doc count fits u64");
                let (index_bytes, segment_count) = tantivy_index_footprint(&index);
                c.record_committed(committed, index_bytes, segment_count);
                let documents =
                    prepared_qg1_documents.expect("QG-1 receipted cell has prepared documents");
                let probe = work_receipt_probe_term(documents);
                let probe_hits = context.runtime.block_on(async {
                    index
                        .search(&context.cx, &probe, 1)
                        .await
                        .expect("QG-1 receipt terminal searchable probe")
                        .len()
                });
                assert!(probe_hits > 0, "terminal searchable probe returned no hits");
                let searchable =
                    u64::try_from(LexicalRead::doc_count(&index)).expect("doc count fits u64");
                c.record_searchable(searchable);
                c.on_phase(LifecyclePhase::SearchableVerified, win_ns(window_origin));
                for _ in 0..periodic_commits {
                    c.record_periodic_commit();
                }
                c.record_queue(QueueObservation::StructurallyUnobservable {
                    seam: "tantivy 0.26.1 IndexWriter AddOperation channel \
                           (crossbeam bounded(10_000)) exposes no depth accessor; observing \
                           occupancy requires patching tantivy"
                        .to_owned(),
                });
                c.record_width(WidthObservation::Observed {
                    threads: u64::try_from(
                        observed_threads.expect("QG-1 Tantivy arm recorded its writer width"),
                    )
                    .expect("width fits u64"),
                    seam: "TantivyIndex::benchmark_materialized_writer_threads".to_owned(),
                });
            }
            let (join, join_receipt) = finish_tantivy_lifecycle(index, spec, "measured_work");
            elapsed += join;
            if let Some(mut c) = collector.take() {
                c.record_terminal(
                    TerminalJoin::TantivyWorkersJoined {
                        join_elapsed_ns: join_receipt.join_elapsed_ns,
                        searchable_segments_before: join_receipt.searchable_segments_before as u64,
                        searchable_segments_after: join_receipt.searchable_segments_after as u64,
                        writer_rearmed: join_receipt.writer_rearmed,
                    },
                    "completed",
                    false,
                );
                c.on_phase(LifecyclePhase::QuiescenceJoined, win_ns(window_origin));
                c.record_measured_sum_ns(
                    u64::try_from(elapsed.as_nanos()).expect("measured ns fits u64"),
                );
                finalize_work_receipt(
                    c,
                    spec,
                    arm,
                    count,
                    expected_bytes,
                    window_started_rel_ns,
                    None,
                );
            }
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
        // Named so the H2 worker-role census can attribute pool threads
        // honestly; rayon threads are otherwise anonymous in /proc comm.
        .thread_name(|index| {
            format!(
                "{}{index}",
                frankensearch_quill_gauntlet::BENCH_RAYON_THREAD_PREFIX
            )
        })
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
        spec.gate,
        PerfGate::Qg1,
        "prepared tokenizer corpus is reserved for QG-1"
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
            let timer = Instant::now();
            black_box(
                context
                    .runtime
                    .block_on(QuillIndex::open(&context.cx, &path, quill_config(spec)))
                    .expect("cold-open Quill"),
            );
            timer.elapsed()
        }
        EngineArm::Tantivy => {
            let path = scratch_path("qg9-tantivy");
            let index = tantivy_create(&path, spec);
            let _ = index_batches(context, &index, &corpus, count, None);
            let _ = commit(context, &index);
            drop(index);
            let timer = Instant::now();
            black_box(
                TantivyIndex::open_with_benchmark_config(
                    &path,
                    spec.writer_heap_bytes.unwrap_or(50_000_000),
                    spec.threads.unwrap_or(1),
                    spec.positions.unwrap_or(PositionMode::On).enabled(),
                )
                .expect("cold-open pinned Tantivy"),
            );
            timer.elapsed()
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

/// Measure one paired raw-sample stream with a seeded balanced randomized
/// first-arm schedule, warmup separation, and monotonic per-sample intervals.
fn paired_raw_stream(
    context: &BenchContext,
    spec: &PerfCellSpec,
    evidence: &EvidenceContext,
    scope: &PerfOperationScope,
    origin: Instant,
    plan: &StreamPlan<'_>,
) -> Vec<PerfRawSample> {
    let order = seeded_balanced_pair_order(plan.rounds, plan.seed).expect("paired order schedule");
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
    let mut samples = Vec::with_capacity(plan.rounds * 2);
    for (round, first_arm) in order.into_iter().enumerate() {
        let round_index = u64::try_from(round).expect("round fits u64");
        let block_id = plan.block_id_base + round_index;
        let control_sample_id = plan.sample_id_base + round_index * 2;
        let treatment_sample_id = control_sample_id + 1;
        let control_first = first_arm == PerfSampleArm::Control;
        let run_arm = |engine: EngineArm,
                       sample_arm: PerfSampleArm,
                       sample_order: PerfSampleOrder,
                       sample_id: u64| {
            clear_continuous_capture();
            let started_ns = u64::try_from(origin.elapsed().as_nanos()).expect("monotonic ns");
            let value = black_box(measure_metric_with_query(
                context,
                spec,
                engine,
                plan.query_override,
            ));
            let mut ended_ns = u64::try_from(origin.elapsed().as_nanos()).expect("monotonic ns");
            if ended_ns <= started_ns {
                ended_ns = started_ns + 1;
            }
            if scope.semantics == PerfMetricSemantics::Throughput {
                // Continuous H1 cell: the sample's own interval is the exact
                // first-feed-to-quiescence window, and the estimator derives
                // the rate from equal work units over it. A gauge value is
                // structurally impossible on this path.
                let capture = take_continuous_capture()
                    .expect("continuous bulk cell produced no window capture");
                black_box(value);
                let window = ContinuousSampleWindow {
                    started_ns: duration_ns(capture.started.duration_since(origin)),
                    ended_ns: duration_ns(capture.ended.duration_since(origin)),
                    work_units: capture.work_units,
                    byte_count: capture.byte_count,
                };
                return continuous_raw_sample(
                    ContinuousSampleIdentity {
                        block_id,
                        sample_id,
                        arm: sample_arm,
                        order: sample_order,
                    },
                    scope,
                    &evidence.sample_provenance,
                    &window,
                )
                .expect("continuous raw sample construction");
            }
            PerfRawSample {
                block_id,
                sample_id,
                arm: sample_arm,
                order: sample_order,
                phase: PerfSamplePhase::Measurement,
                scope: scope.clone(),
                provenance: evidence.sample_provenance.clone(),
                started_ns,
                ended_ns,
                work_units: None,
                byte_count: None,
                observed_value: Some(value),
                group_id: plan.group_id,
                qg6_sample_binding: None,
            }
        };
        if control_first {
            samples.push(run_arm(
                plan.control,
                PerfSampleArm::Control,
                PerfSampleOrder::First,
                control_sample_id,
            ));
            samples.push(run_arm(
                plan.treatment,
                PerfSampleArm::Treatment,
                PerfSampleOrder::Second,
                treatment_sample_id,
            ));
        } else {
            samples.push(run_arm(
                plan.treatment,
                PerfSampleArm::Treatment,
                PerfSampleOrder::First,
                treatment_sample_id,
            ));
            samples.push(run_arm(
                plan.control,
                PerfSampleArm::Control,
                PerfSampleOrder::Second,
                control_sample_id,
            ));
        }
    }
    samples
}

/// Presentation value of one raw sample: the stored gauge for legacy cells,
/// or the rate derived from equal work units over the continuous window for
/// Throughput cells (mirroring the estimator's own derivation).
fn sample_value(sample: &PerfRawSample) -> f64 {
    sample.observed_value.unwrap_or_else(|| {
        #[allow(clippy::cast_precision_loss)]
        let work_units = sample
            .work_units
            .expect("throughput sample carries work units") as f64;
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ns = sample.ended_ns.saturating_sub(sample.started_ns).max(1) as f64;
        work_units * 1_000_000_000.0 / elapsed_ns
    })
}

fn arm_values(samples: &[PerfRawSample], arm: PerfSampleArm) -> Vec<f64> {
    samples
        .iter()
        .filter(|sample| sample.arm == arm)
        .map(sample_value)
        .collect()
}

fn block_ratios_treatment_over_control(samples: &[PerfRawSample]) -> Vec<f64> {
    let mut by_block: BTreeMap<u64, (Option<f64>, Option<f64>)> = BTreeMap::new();
    for sample in samples {
        let entry = by_block.entry(sample.block_id).or_default();
        match sample.arm {
            PerfSampleArm::Control => entry.0 = Some(sample_value(sample)),
            PerfSampleArm::Treatment => entry.1 = Some(sample_value(sample)),
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
        checksum ^= sample_value(sample).to_bits().rotate_left(13);
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
                role: EvidenceRole::Diagnostic,
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

    // Continuous H1 bulk cells carry an independent Throughput scope identity
    // (`.continuous`, version 2): continuous and summed-call sample streams
    // can never be merged or substituted for one another.
    let scope = if continuous_bulk_cell(spec) {
        continuous_throughput_scope(
            &format!("{}.{}.{}", spec.gate, spec.fixture, spec.metric),
            unit(spec),
        )
    } else {
        operation_scope(spec)
    };
    let origin = Instant::now();
    let cell_seed = evidence.config.bootstrap_seed ^ fixture_seed(&spec.fixture);

    // Every non-query gate establishes its A/A floor through the exact paired
    // routine before measuring the Quill/Tantivy claim. QG-6 uses the prepared
    // four-arm runner so setup is impossible inside timed samples and null/
    // effect blocks are interleaved.
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
        let oracle_null = paired_raw_stream(
            context,
            spec,
            evidence,
            &scope,
            origin,
            &StreamPlan {
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
        let treatment_null = (spec.gate == PerfGate::Qg1).then(|| {
            paired_raw_stream(
                context,
                spec,
                evidence,
                &scope,
                origin,
                &StreamPlan {
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
        let effect = paired_raw_stream(
            context,
            spec,
            evidence,
            &scope,
            origin,
            &StreamPlan {
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
        (oracle_null, treatment_null, effect, None, None)
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
    let cold_cache = (spec.gate == PerfGate::Qg9).then(|| ColdCacheEvidence {
        procedure: "same-process index drop and reopen; the OS page cache is not dropped"
            .to_owned(),
        verified: false,
    });
    let mut cell = EvidenceCell::evaluate(
        EvidenceCellSpec {
            gate: spec.gate,
            fixture: spec.fixture.clone(),
            metric: spec.metric.clone(),
            unit: unit(spec).to_owned(),
            role: if is_tokenizer_null {
                EvidenceRole::Diagnostic
            } else {
                EvidenceRole::Required
            },
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

fn selected_cells(matrix: &PerfMatrixSpec, scale: MatrixScale) -> Vec<PerfCellSpec> {
    let gate_filter = std::env::var("QUILL_PERF_GATE").unwrap_or_else(|_| "QG-1".to_owned());
    let fixture_filter = std::env::var("QUILL_PERF_FIXTURE").ok();
    let mut selected = if gate_filter.eq_ignore_ascii_case("all") {
        matrix.cells.clone()
    } else {
        let gate = gate_filter.parse::<PerfGate>().expect("QUILL_PERF_GATE");
        matrix
            .for_gate(gate)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
    };
    if let Some(needle) = fixture_filter {
        selected.retain(|cell| cell.fixture.contains(&needle));
    }
    if !scale.is_full() {
        selected.truncate(1);
    }
    assert!(!selected.is_empty(), "QG matrix slice selected no cells");
    selected
}

fn gate_selection_complete(
    matrix: &PerfMatrixSpec,
    selected: &[PerfCellSpec],
    gate: PerfGate,
) -> bool {
    let normative = matrix.for_gate(gate).len();
    let selected = selected.iter().filter(|cell| cell.gate == gate).count();
    if gate == PerfGate::Qg6 {
        return matches!(
            Qg6SelectionScope::from_cell_counts(selected, normative),
            Ok(Qg6SelectionScope::CompleteGate)
        );
    }
    normative != 0 && selected == normative
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

fn corpus_manifest_hash(context: &BenchContext, cells: &[PerfCellSpec]) -> Result<String, String> {
    let qg1_counts = cells
        .iter()
        .filter(|cell| cell.gate == PerfGate::Qg1)
        .map(|cell| {
            context
                .scale
                .document_count(cell.document_count.unwrap_or_default())
        })
        .collect::<BTreeSet<_>>();
    let qg1_identities = qg1_counts
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
        .collect::<Result<BTreeMap<_, _>, String>>()?;

    let mut hasher = Sha256::new();
    for cell in cells {
        let requested = cell.document_count.unwrap_or_default();
        let effective = context.scale.document_count(requested);
        hasher.update(cell.fixture.as_bytes());
        hasher.update(effective.to_le_bytes());
        hasher.update(CORPUS_SEED.to_le_bytes());
        hasher.update(VOCABULARY_SIZE.to_le_bytes());
        hasher.update(MAX_DOCUMENT_BYTES.to_le_bytes());
        if cell.gate == PerfGate::Qg1 {
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
    let document_count = cells
        .iter()
        .map(|cell| {
            context
                .scale
                .document_count(cell.document_count.unwrap_or_default())
        })
        .max()
        .unwrap_or_default();
    let query_set_sha256 = Some(
        Qg6QuerySpec::normative_manifest_sha256()
            .expect("frozen 80-query manifest validates before evidence identity"),
    );
    CorpusIdentity {
        corpus_sha256: corpus_hash.to_owned(),
        query_set_sha256,
        qrels_sha256: None,
        document_count,
        content_bytes: None,
        generator_seed: CORPUS_SEED,
        generator_revision: Qg6QuerySpec::normative_corpus_generator_revision().to_owned(),
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
    //
    // Continuous H1 and receipted H2 cells skip the presentation lane
    // entirely: each Criterion iteration would run another full window and
    // overwrite the recorded final-measurement-round receipt after the
    // evidence was described.
    if context.scale.is_full()
        || matches!(spec.gate, PerfGate::Qg6 | PerfGate::Qg7 | PerfGate::Qg10)
        || continuous_bulk_cell(spec)
        || work_receipt_cell(spec)
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
    let selected = selected_cells(&matrix, scale);
    let context = BenchContext::for_selected(scale, &selected);
    preflight_indexing_fixtures(&context, &matrix, &selected);
    let configured_runs = std::env::var("QUILL_PERF_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            if scale.is_full() && selected.iter().any(|cell| cell.gate == PerfGate::Qg4) {
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
    let corpus_hash = corpus_manifest_hash(&context, &selected)
        .expect("verify exact prepared QG-1 corpus identity");
    let bootstrap_seed = std::env::var("QUILL_PERF_BOOTSTRAP_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0x5155_494c_4c45_5644);
    let evidence_policy = evidence_policy_from_env();
    eprintln!(
        "[quill-perf-policy] warmup_rounds={}",
        evidence_policy.warmup_rounds
    );
    // QG-1 H1: opt-in continuous first-feed-to-quiescence timing. Per-call
    // remains the artifact-compatible default until the fleet flips it.
    eprintln!("[quill-perf-timing-mode] mode={}", timing_mode().label());
    // QG-1 H2: opt-in actual-work/queue/worker-role/lifecycle receipts.
    // Off by default: the artifact byte shape and the measured lanes are
    // untouched until a run opts in.
    eprintln!("[quill-work-receipts] mode={}", work_receipt_mode().label());
    let _ = WORK_RECEIPT_IDENTITY.set(WorkReceiptRunIdentity {
        run_id: run_id.clone(),
        machine_fingerprint: machine_fingerprint(),
        build_profile: build_profile.clone(),
        executable_sha256: bench_elf_sha256.to_owned(),
        git_rev: revision.clone(),
    });
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
    let mut machine = MachineIdentity::capture(selected.iter().filter_map(|spec| spec.threads));
    eprintln!(
        "[quill-perf-execution-provenance] {}",
        serde_json::to_string(&machine.execution).expect("serialize execution provenance")
    );

    let mut by_gate: BTreeMap<PerfGate, Vec<PerfCellResult>> = BTreeMap::new();
    let mut evidence_by_gate: BTreeMap<PerfGate, Vec<EvidenceCell>> = BTreeMap::new();
    for spec in &selected {
        let collection = collect_cell(&context, spec, configured_runs, &evidence_context);
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
    flush_work_receipts(&output_dir);
    let mut continuous_by_gate = take_continuous_evidence(&context);
    for (gate, evidence) in &continuous_by_gate {
        eprintln!(
            "[qg1-continuous-summary] gate={gate} timing_mode={} cells={}",
            evidence.timing_mode,
            evidence.cells.len(),
        );
    }
    let mut work_receipts_by_gate = take_work_receipt_evidence();
    for (gate, evidence) in &work_receipts_by_gate {
        eprintln!(
            "[qg1-work-receipt-summary] gate={gate} mode={} cells={}",
            evidence.mode,
            evidence.cells.len(),
        );
    }

    let provenance = EvidenceProvenance {
        run_id: run_id.clone(),
        run_window: run_window.clone(),
        manifest_sha256: manifest_hash.clone(),
        build: build_identity(bench_elf_sha256, &revision, &build_profile),
        machine: machine.clone(),
        peak_rss: PeakRssEvidence::capture(),
        corpus: corpus_identity(&context, &selected, &corpus_hash),
    };
    for (gate, cells) in evidence_by_gate {
        let mut artifact = PerfEvidenceArtifact::assemble(
            gate,
            evidence_context.policy.clone(),
            provenance.clone(),
            cells,
        )
        .expect("assemble QG evidence artifact");
        let mut admission_no_claims = Vec::new();
        if !gate_selection_complete(&matrix, &selected, gate) {
            admission_no_claims.push((
                "evidence.incomplete_gate_selection",
                "the invocation selected only part of the normative gate; durable pre-admission \
                 evidence cannot support a publication or ratchet claim",
            ));
        }
        // Unconditional guard, not convention: the ratchet does not yet bind
        // timing mode, the `.continuous` operation identity, or the
        // continuous_timing schema, so no continuous-mode invocation may
        // support a claim until a distinct end-to-end mode is validated.
        if timing_mode().is_continuous() {
            admission_no_claims.push((
                "evidence.continuous_timing_mode_unratcheted",
                "QUILL_PERF_TIMING_MODE=continuous emits pre-admission diagnostics only: the \
                 ratchet does not yet bind timing mode, continuous operation identity, or the \
                 continuous_timing schema end-to-end",
            ));
        }
        // Unconditional guard, not convention: receipts are provenance, and
        // enabling them adds observation probes between measured regions
        // (symmetric across arms, but unratcheted), so a receipted run can
        // never support a claim until the ratchet binds the receipts mode.
        if work_receipt_mode().is_enabled() {
            admission_no_claims.push((
                "evidence.work_receipts_mode_unratcheted",
                "QUILL_PERF_WORK_RECEIPTS=on adds symmetric observation probes between \
                 measured regions; the ratchet does not bind the receipts mode, so receipt \
                 runs are provenance only",
            ));
        }
        match admission_no_claims.as_slice() {
            [] => {}
            [(code, message)] => artifact.force_no_claim(code, *message),
            reasons => {
                let codes = reasons
                    .iter()
                    .map(|(code, _)| *code)
                    .collect::<Vec<_>>()
                    .join(", ");
                artifact.force_no_claim(
                    "evidence.multiple_pre_admission_no_claims",
                    format!("multiple pre-admission no-claim conditions apply: {codes}"),
                );
            }
        }
        let paths = artifact
            .write_atomic(&output_dir)
            .expect("write QG evidence artifact");
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
            bench_elf_sha256: bench_elf_sha256.to_owned(),
            machine_fingerprint: machine_fingerprint(),
            execution: Some(machine.execution.clone()),
            git_rev: revision.clone(),
            run_window: run_window.clone(),
            run_id: run_id.clone(),
            corpus_manifest_hash: corpus_hash.clone(),
            manifest_sha256: manifest_hash.clone(),
            cells,
            // Continuous mode can never attest the laws: the ratchet does not
            // yet bind the continuous stream identity (see the matching
            // evidence.continuous_timing_mode_unratcheted force_no_claim).
            laws_attested: scale.is_full()
                && gate_selection_complete(&matrix, &selected, gate)
                && !timing_mode().is_continuous(),
            continuous_timing: continuous_by_gate.remove(&gate),
            work_receipts: work_receipts_by_gate.remove(&gate),
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
            let bytes = index.segment_stats().managed_disk_bytes;
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
/// (version, checksum, lexical package + git revision, `tantivy = "=0.26.1"`)
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
         lexical={}@{} lexical_git={}",
        contract.tantivy_version,
        contract.lexical_package,
        contract.lexical_package_version,
        contract.lexical_git_revision,
    );
    linked.to_owned()
}

fn main() {
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
    fn continuous_quill_periodic_commits_exclude_the_terminal_commit() {
        let cell = super::PerfCellSpec {
            gate: super::PerfGate::Qg1,
            fixture: "bulk/test-periodic-pin/1/positions_on".to_owned(),
            metric: "docs_per_second".to_owned(),
            corpus: Some(super::PerfCorpus::Tiny),
            document_count: Some(12),
            threads: Some(1),
            writer_heap_bytes: Some(50_000_000),
            positions: Some(super::PositionMode::On),
            tombstone_density_pct: None,
            query_class: None,
            k: None,
            topology: None,
        };
        let context = super::BenchContext::for_selected(super::MatrixScale::Smoke, &[cell.clone()]);
        let rate = super::bulk_metric_continuous(&context, &cell, super::EngineArm::Quill);
        assert!(rate > 0.0, "continuous window produced a derived rate");

        let state = super::CONTINUOUS_CELL_STATE
            .get()
            .expect("continuous state recorded")
            .lock()
            .expect("lock continuous state");
        let recorded = state.get(&cell.fixture).expect("cell state recorded");
        let receipt = recorded
            .last_quill
            .as_ref()
            .expect("quill continuous receipt recorded");
        // A 12-document feed completes far below the visibility cadence, so
        // no periodic publish can occur during the feed phase. Before the
        // pre-terminal sampling fix, the terminal commit's snapshot
        // generation bump leaked into this count and made it nonzero.
        assert_eq!(
            receipt.periodic_commits, 0,
            "the terminal commit must never be counted as a periodic visibility commit"
        );
        assert_eq!(receipt.committed_docs, 12);
        assert_eq!(receipt.searchable_docs, 12);
    }
}
