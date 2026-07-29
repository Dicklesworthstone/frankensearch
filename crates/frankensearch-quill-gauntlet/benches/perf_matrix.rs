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

use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
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
    BuildIdentity, ColdCacheEvidence, ComparatorConfig, ComparisonStatus, CorpusIdentity,
    CountState, DistributionSummary, EngineObservation, EvidenceCell, EvidenceCellSpec,
    EvidencePolicy, EvidenceProvenance, EvidenceRole, MachineIdentity, NativeTieKey,
    PERF_ARTIFACT_SCHEMA_VERSION, PERF_MIN_RUNS, PairedEstimatorConfig, PeakRssEvidence,
    PerfCellResult, PerfCellSpec, PerfCorpus, PerfEvidenceArtifact, PerfGate, PerfGateArtifact,
    PerfInputIdentity, PerfMatrixSpec, PerfMetricSemantics, PerfOperationScope, PerfQueryClass,
    PerfRawSample, PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance,
    PerfTopology, PositionMode, QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS, Qg6ArmRole, Qg6Comparison,
    Qg6PreparedExperiment, Qg6QuerySpec, Qg6SampleOrder, Qg6SearchResult, Qg6SelectionScope,
    RankClass, RankedHit, ScoreEpsilonReason, SyntheticCorpus, SyntheticCorpusSpec, ZipfExponent,
    compare_observations, estimate_paired_experiment, machine_fingerprint, oracle_version_contract,
    peak_rss_bytes, perf_manifest_contract_sha256, seeded_balanced_pair_order, validate_matrix,
};
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

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);
static LIFECYCLE_RECEIPT_COUNTER: AtomicU64 = AtomicU64::new(0);
static LIFECYCLE_RECEIPTS: OnceLock<Mutex<Vec<serde_json::Value>>> = OnceLock::new();

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

enum PreparedQueryResult {
    Quill(Vec<frankensearch_quill::QuillHit>),
    Tantivy(Vec<frankensearch_lexical::LexicalIdHit>),
}

struct PreparedQueryPreflight {
    native_hits: Vec<(String, u32)>,
    observation: Option<EngineObservation>,
}

struct BenchContext {
    runtime: Runtime,
    cx: Cx,
    scale: MatrixScale,
}

impl BenchContext {
    fn new(scale: MatrixScale) -> Self {
        Self {
            runtime: asupersync::runtime::RuntimeBuilder::current_thread()
                .build()
                .expect("QG benchmark runtime"),
            cx: Cx::for_testing(),
            scale,
        }
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

fn index_batches_with_visibility_commits<E: LexicalWrite>(
    context: &BenchContext,
    index: &E,
    corpus: &SyntheticCorpus,
    document_count: u64,
    commit_cadence: Duration,
) -> (Duration, usize) {
    let mut measured = Duration::ZERO;
    let mut unpublished_since = None;
    let mut periodic_commits = 0_usize;
    let batch_documents = context.scale.batch_documents();
    let mut start = 0_u64;
    while start < document_count {
        let remaining = document_count - start;
        let count =
            usize::try_from(remaining.min(batch_documents as u64)).expect("bounded batch count");
        let documents = generated_batch(corpus, start, count, None);
        let timer = Instant::now();
        let unpublished_started = *unpublished_since.get_or_insert(timer);
        context.runtime.block_on(async {
            index
                .index_documents(&context.cx, &documents)
                .await
                .expect("QG index batch");
        });
        measured += timer.elapsed();
        if unpublished_started.elapsed() >= commit_cadence {
            measured += commit(context, index);
            periodic_commits = periodic_commits.saturating_add(1);
            unpublished_since = None;
        }
        start = start.saturating_add(u64::try_from(count).expect("batch count fits u64"));
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

fn bulk_metric_unpooled(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    let requested = spec.document_count.expect("bulk document count");
    let count = context.scale.document_count(requested);
    let corpus = corpus_for(count);
    let elapsed = match arm {
        EngineArm::Quill => {
            let index = quill_in_memory(spec);
            let generation_before = index.snapshot().loaded_manifest().manifest.generation;
            let mut elapsed = index_batches(context, &index, &corpus, count, None);
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
            let (mut elapsed, periodic_commits) = if spec.gate == PerfGate::Qg1 {
                index_batches_with_visibility_commits(
                    context,
                    &index,
                    &corpus,
                    count,
                    Duration::from_millis(quill_config(spec).max_visibility_lag_ms),
                )
            } else {
                (index_batches(context, &index, &corpus, count, None), 0)
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
            assert_eq!(
                rayon::current_num_threads(),
                threads,
                "QG-1/QG-8 Quill cell escaped its pinned Rayon pool"
            );
            bulk_metric_unpooled(context, spec, arm)
        })
}

fn tokenize_metric(context: &BenchContext, spec: &PerfCellSpec) -> f64 {
    let count = context
        .scale
        .document_count(spec.document_count.expect("tokenize document count"));
    let corpus = corpus_for(count);
    let mut tokenizer = FrankensearchTokenizer::default();
    let mut measured = Duration::ZERO;
    let mut start = 0_u64;
    while start < count {
        let remaining = count - start;
        let batch_count = usize::try_from(
            remaining.min(u64::try_from(context.scale.batch_documents()).expect("batch size")),
        )
        .expect("tokenize batch count");
        let documents = generated_batch(&corpus, start, batch_count, None);
        let timer = Instant::now();
        let mut token_count = 0_usize;
        for document in &documents {
            tokenizer.analyze(
                Analyzer::FrankensearchDefault,
                black_box(&document.content),
                &mut |_| token_count = token_count.saturating_add(1),
            );
        }
        measured += timer.elapsed();
        black_box(token_count);
        start = start.saturating_add(u64::try_from(batch_count).expect("batch count"));
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

fn query_texts(query_class: PerfQueryClass) -> &'static [&'static str; QG6_QUERY_GROUPS] {
    match query_class {
        PerfQueryClass::Identifier => &["term00042", "term00137", "term00256", "term00301"],
        PerfQueryClass::ShortKeyword => &["term00001", "term00002", "term00005", "term00011"],
        PerfQueryClass::NaturalLanguage => &[
            "term00001 term00007 generated record",
            "term00002 term00013 generated record",
            "term00005 term00011 generated record",
            "term00003 term00017 generated record",
        ],
        PerfQueryClass::Phrase => &[
            "\"term00001 term00002\"",
            "\"term00002 term00003\"",
            "\"term00003 term00004\"",
            "\"term00005 term00006\"",
        ],
        PerfQueryClass::Boolean => &[
            "term00001 OR term00002",
            "term00003 OR term00004",
            "term00002 OR term00005",
            "term00001 OR term00007",
        ],
    }
}

fn query_text(query_class: PerfQueryClass) -> &'static str {
    query_texts(query_class)[0]
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
    let query =
        query_override.unwrap_or_else(|| query_text(spec.query_class.expect("query class")));
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
    hasher.update(b"frankensearch/qg6/semantic-config/v1\0");
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
            if *role != Qg6ArmRole::NullLeft {
                return Ok(PreparedQueryPreflight {
                    native_hits,
                    observation: None,
                });
            }
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
                return Err(
                    "Tantivy native timed query disagrees with its tie-evidence observation"
                        .to_owned(),
                );
            }
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
                observation: Some(EngineObservation {
                    hits,
                    cutoff_tie_group,
                    cutoff_tie_complete: observed.cutoff_tie_complete,
                    offset_tie_group: Vec::new(),
                    offset_tie_complete: false,
                    snippets: BTreeMap::new(),
                    match_count: CountState::Value(
                        u64::try_from(observed.total_count)
                            .map_err(|_| "Tantivy match count does not fit u64")?,
                    ),
                    doc_count: u64::try_from(observed.doc_count)
                        .map_err(|_| "Tantivy document count does not fit u64")?,
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
    let query_class = spec.query_class.expect("QG-6 query class");
    query_texts(query_class)
        .iter()
        .enumerate()
        .map(|(index, text)| {
            Qg6QuerySpec::new(
                format!("{query_class:?}-{index}").to_ascii_lowercase(),
                *text,
            )
            .expect("bounded QG-6 query")
        })
        .collect()
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
    }
}

fn prepared_qg6_streams(
    context: &BenchContext,
    spec: &PerfCellSpec,
    runs: usize,
    evidence: &EvidenceContext,
    scope: &PerfOperationScope,
    cell_seed: u64,
) -> (Vec<PerfRawSample>, Vec<PerfRawSample>, PerfInputIdentity) {
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
    let mut preflight_search = |arm: &PreparedQueryArm, query: &Qg6QuerySpec, k: usize| {
        qg6_preflight_result(context, arm, query, k)
    };
    let mut preflight_normalize = |result: &PreparedQueryPreflight| {
        Qg6SearchResult::from_ordered_doc_ids(
            result
                .native_hits
                .iter()
                .map(|(doc_id, _)| doc_id.clone())
                .collect(),
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
        if observed_role != Qg6ArmRole::EffectTreatment {
            return if expected.native_hits == observed.native_hits {
                Ok(())
            } else {
                Err("Tantivy A/A preflight changed native ranked hits".to_owned())
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
        eprintln!(
            "[qg6-semantic-parity] query_id={} status={:?} rank={:?} \
             score_epsilon_reason={:?} score_epsilon_bits={} topk={} \
             count_equal={} doc_count_equal={}",
            query.id(),
            report.status,
            report.rank_class,
            report.score_epsilon_reason,
            comparator_config.score_epsilon_bits,
            report.subject.hits.len(),
            report.subject.match_count == report.oracle.match_count,
            report.subject.doc_count == report.oracle.doc_count,
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
    let mut search = |arm: &PreparedQueryArm, query: &Qg6QuerySpec, k: usize| match arm {
        PreparedQueryArm::Quill { index, .. } => index
            .search_doc_ids(&context.cx, query.text(), k)
            .map(PreparedQueryResult::Quill)
            .map_err(|error| error.to_string()),
        PreparedQueryArm::Tantivy { index, .. } => index
            .search_doc_ids(&context.cx, query.text(), k)
            .map(PreparedQueryResult::Tantivy)
            .map_err(|error| error.to_string()),
    };
    let mut normalize = |result| {
        let ordered_doc_ids = match result {
            PreparedQueryResult::Quill(hits) => {
                hits.into_iter().map(|hit| hit.document_id).collect()
            }
            PreparedQueryResult::Tantivy(hits) => {
                hits.into_iter().map(|hit| hit.doc_id.to_string()).collect()
            }
        };
        Qg6SearchResult::from_ordered_doc_ids(ordered_doc_ids)
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
        query_group_count: QG6_QUERY_GROUPS,
        query_group_ids: QG6_QUERY_GROUP_IDS.to_vec(),
    };
    let mut sample_provenance = evidence.sample_provenance.clone();
    sample_provenance.input_identity = Some(input_identity.clone());
    eprintln!(
        "[qg6-prepared] fixture={} corpus_sha256={} query_manifest_sha256={} \
         config_contract_sha256={} schedule_seed={} warmup_rounds={} rounds_per_query={} \
         searches_per_sample={} \
         sample_input_sha256={} result_receipt_sha256={} lifecycle={}",
        spec.fixture,
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
    (null_samples, effect_samples, input_identity)
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
                cold_cache: None,
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
    // routine before measuring the Quill/Tantivy claim. QG-6 uses the prepared
    // four-arm runner so setup is impossible inside timed samples and null/
    // effect blocks are interleaved.
    let (oracle_null_samples, treatment_null_samples, effect_samples, input_identity) =
        if spec.gate == PerfGate::Qg6 {
            let (null, effect, input_identity) =
                prepared_qg6_streams(context, spec, runs, evidence, &scope, cell_seed);
            (null, None, effect, Some(input_identity))
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
            (oracle_null, treatment_null, effect, None)
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
            cold_cache,
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

fn corpus_manifest_hash(context: &BenchContext, cells: &[PerfCellSpec]) -> String {
    let mut hasher = Sha256::new();
    for cell in cells {
        let requested = cell.document_count.unwrap_or_default();
        let effective = context.scale.document_count(requested);
        hasher.update(cell.fixture.as_bytes());
        hasher.update(effective.to_le_bytes());
        hasher.update(CORPUS_SEED.to_le_bytes());
        hasher.update(VOCABULARY_SIZE.to_le_bytes());
        hasher.update(MAX_DOCUMENT_BYTES.to_le_bytes());
    }
    lower_hex(&hasher.finalize())
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

fn build_identity(bench_elf_sha256: &str, revision: &str, build_profile: &str) -> BuildIdentity {
    let porcelain = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned());
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
    let rustc_verbose = Command::new("rustc")
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
    let query_set_sha256 = {
        let mut hasher = Sha256::new();
        for class in [
            PerfQueryClass::Identifier,
            PerfQueryClass::ShortKeyword,
            PerfQueryClass::NaturalLanguage,
            PerfQueryClass::Phrase,
            PerfQueryClass::Boolean,
        ] {
            for query in query_texts(class) {
                hasher.update(query.as_bytes());
                hasher.update([0]);
            }
        }
        Some(lower_hex(&hasher.finalize()))
    };
    CorpusIdentity {
        corpus_sha256: corpus_hash.to_owned(),
        query_set_sha256,
        qrels_sha256: None,
        document_count,
        content_bytes: None,
        generator_seed: CORPUS_SEED,
        generator_revision: "synthetic-zipf-s11-vocab8192-doc4096-v1".to_owned(),
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
    let context = BenchContext::new(scale);
    let matrix = PerfMatrixSpec::complete();
    validate_matrix(&matrix).expect("normative QG matrix");
    let selected = selected_cells(&matrix, scale);
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
    let corpus_hash = corpus_manifest_hash(&context, &selected);
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
        if !gate_selection_complete(&matrix, &selected, gate) {
            artifact.force_no_claim(
                "evidence.incomplete_gate_selection",
                "the invocation selected only part of the normative gate; durable pre-admission \
                 evidence cannot support a publication or ratchet claim",
            );
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
            laws_attested: scale.is_full() && gate_selection_complete(&matrix, &selected, gate),
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
