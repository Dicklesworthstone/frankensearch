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
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::{Cx, runtime::Runtime};
use criterion::Criterion;
use frankensearch_core::bench_support::print_bench_elf_sha256;
use frankensearch_core::{IndexableDocument, LexicalSearch};
use frankensearch_lexical::TantivyIndex;
use frankensearch_quill::scribe::{FrankensearchTokenizer, TokenAnalyzer};
use frankensearch_quill::{
    Analyzer, CompactionPolicy, FieldDescriptor, FieldKind, QuillConfig, QuillIndex,
    SchemaDescriptor, SegmentStatsProvider,
};
use frankensearch_quill_gauntlet::{
    BuildIdentity, ColdCacheEvidence, CorpusIdentity, DistributionSummary, EvidenceCell,
    EvidenceCellSpec, EvidencePolicy, EvidenceProvenance, EvidenceRole, MachineIdentity,
    PERF_ARTIFACT_SCHEMA_VERSION, PERF_MIN_RUNS, PairedEstimatorConfig, PeakRssEvidence,
    PerfCellResult, PerfCellSpec, PerfCorpus, PerfEvidenceArtifact, PerfGate, PerfGateArtifact,
    PerfMatrixSpec, PerfMetricSemantics, PerfOperationScope, PerfQueryClass, PerfRawSample,
    PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance, PerfTopology,
    PositionMode, SyntheticCorpus, SyntheticCorpusSpec, ZipfExponent, estimate_paired_experiment,
    machine_fingerprint, peak_rss_bytes, seeded_balanced_pair_order, validate_matrix,
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

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

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

fn index_batches<E: LexicalSearch>(
    context: &BenchContext,
    index: &E,
    corpus: &SyntheticCorpus,
    document_count: u64,
    update_generation: Option<u64>,
) -> Duration {
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
        start = start.saturating_add(u64::try_from(count).expect("batch count fits u64"));
    }
    measured
}

fn commit<E: LexicalSearch>(context: &BenchContext, index: &E) -> Duration {
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
            index_batches(context, &index, &corpus, count, None) + commit(context, &index)
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
            index_batches(context, &index, &corpus, count, None) + commit(context, &index)
        }
    };
    count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
}

fn bulk_metric(context: &BenchContext, spec: &PerfCellSpec, arm: EngineArm) -> f64 {
    if spec.gate != PerfGate::Qg8 || arm != EngineArm::Quill {
        return bulk_metric_unpooled(context, spec, arm);
    }

    let threads = spec.threads.expect("QG-8 thread count");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build QG-8 Quill thread pool")
        .install(|| {
            assert_eq!(
                rayon::current_num_threads(),
                threads,
                "QG-8 Quill cell escaped its pinned Rayon pool"
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
    let elapsed = match (arm, topology) {
        (EngineArm::Quill, PerfTopology::InProcess) => {
            let index = quill_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, warm_count, None);
            let _ = commit(context, &index);
            let mut elapsed = index_batches(context, &index, &corpus, update_count, Some(1));
            elapsed += commit(context, &index);
            let timer = Instant::now();
            black_box(
                index
                    .search_doc_ids(&context.cx, "term00001", 10)
                    .expect("in-process Quill visibility"),
            );
            elapsed + timer.elapsed()
        }
        (EngineArm::Tantivy, PerfTopology::InProcess) => {
            let index = tantivy_in_memory(spec);
            let _ = index_batches(context, &index, &corpus, warm_count, None);
            let _ = commit(context, &index);
            let mut elapsed = index_batches(context, &index, &corpus, update_count, Some(1));
            elapsed += commit(context, &index);
            let timer = Instant::now();
            black_box(
                index
                    .search_doc_ids(&context.cx, "term00001", 10)
                    .expect("in-process Tantivy visibility"),
            );
            elapsed + timer.elapsed()
        }
        (EngineArm::Quill, PerfTopology::FreshProcess) => {
            measure_quill_fresh_process(context, spec, &corpus, warm_count, update_count)
        }
        (EngineArm::Tantivy, PerfTopology::FreshProcess) => {
            measure_tantivy_fresh_process(context, spec, &corpus, warm_count, update_count)
        }
    };
    if spec.metric == "updates_per_second" {
        update_count as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
    } else {
        elapsed.as_secs_f64() * 1_000.0
    }
}

fn scratch_path(label: &str) -> PathBuf {
    let root = std::env::var_os("QUILL_PERF_SCRATCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("frankensearch-quill-perf"));
    std::fs::create_dir_all(&root).expect("QG scratch root");
    let sequence = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
    root.join(format!("{label}-{}-{sequence}", std::process::id()))
}

fn measure_quill_fresh_process(
    context: &BenchContext,
    spec: &PerfCellSpec,
    corpus: &SyntheticCorpus,
    warm_count: u64,
    update_count: u64,
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
    elapsed + fresh_process_search(&path, spec, EngineArm::Quill)
}

fn measure_tantivy_fresh_process(
    context: &BenchContext,
    spec: &PerfCellSpec,
    corpus: &SyntheticCorpus,
    warm_count: u64,
    update_count: u64,
) -> Duration {
    let path = scratch_path("qg3-tantivy");
    let index = tantivy_create(&path, spec);
    let _ = index_batches(context, &index, corpus, warm_count, None);
    let _ = commit(context, &index);
    let mut elapsed = index_batches(context, &index, corpus, update_count, Some(1));
    elapsed += commit(context, &index);
    drop(index);
    elapsed + fresh_process_search(&path, spec, EngineArm::Tantivy)
}

fn fresh_process_search(path: &Path, spec: &PerfCellSpec, arm: EngineArm) -> Duration {
    let timer = Instant::now();
    let output = Command::new(std::env::current_exe().expect("QG benchmark executable"))
        .env("QUILL_PERF_CHILD_MODE", "search")
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
            let index = quill_in_memory(spec);
            for segment in 0..segments {
                let start =
                    u64::try_from(segment.saturating_mul(docs_per_segment)).expect("segment start");
                if start >= count {
                    break;
                }
                let segment_count = usize::try_from((count - start).min(docs_per_segment as u64))
                    .expect("segment count");
                let documents = generated_batch(&corpus, start, segment_count, None);
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
            timer.elapsed()
        }
        EngineArm::Tantivy => {
            let index = tantivy_in_memory(spec);
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
                let documents = generated_batch(&corpus, start, segment_count, None);
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
            timer.elapsed()
        }
    };
    elapsed.as_secs_f64() * 1_000.0
}

fn stage_deletes(
    context: &BenchContext,
    index: &QuillIndex,
    corpus: &SyntheticCorpus,
    count: u64,
    density: u8,
) {
    let deleted = count.saturating_mul(u64::from(density)) / 100;
    for ordinal in 0..deleted {
        let source = ordinal.saturating_mul(count / deleted.max(1));
        let id = corpus
            .document_at(source.min(count.saturating_sub(1)))
            .expect("Quill deleted document")
            .id;
        context.runtime.block_on(async {
            assert!(
                index
                    .delete_document(&context.cx, &id)
                    .await
                    .expect("stage Quill tombstone"),
                "Quill compaction tombstone must target a live document"
            );
        });
    }
}

/// Per-class query groups for hierarchical latency evidence. The first entry
/// of each class is the legacy single-query text, so flat consumers keep the
/// exact workload they always measured.
const QG6_QUERY_GROUPS: usize = 4;

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

    // Contract order is deliberate: establish this invocation's A/A floor
    // through the exact paired routine, then measure the Quill/Tantivy claim.
    let (null_samples, effect_samples) = if spec.gate == PerfGate::Qg6 {
        let per_group = runs
            .div_ceil(QG6_QUERY_GROUPS)
            .max(evidence.policy.min_group_pairs);
        let queries = query_texts(spec.query_class.expect("query class"));
        let mut null_samples = Vec::new();
        let mut effect_samples = Vec::new();
        for (group_index, query) in queries.iter().enumerate() {
            let group = u64::try_from(group_index).expect("group index");
            let group_seed = cell_seed ^ group.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            null_samples.extend(paired_raw_stream(
                context,
                spec,
                evidence,
                &scope,
                origin,
                &StreamPlan {
                    control: EngineArm::Tantivy,
                    treatment: EngineArm::Tantivy,
                    rounds: per_group,
                    seed: group_seed ^ 0xaa,
                    block_id_base: group * 10_000,
                    sample_id_base: 1_000_000 + group * 10_000,
                    group_id: Some(group),
                    query_override: Some(query),
                },
            ));
            effect_samples.extend(paired_raw_stream(
                context,
                spec,
                evidence,
                &scope,
                origin,
                &StreamPlan {
                    control: EngineArm::Tantivy,
                    treatment: EngineArm::Quill,
                    rounds: per_group,
                    seed: group_seed,
                    block_id_base: group * 10_000,
                    sample_id_base: group * 10_000,
                    group_id: Some(group),
                    query_override: Some(query),
                },
            ));
        }
        (null_samples, effect_samples)
    } else {
        let null = paired_raw_stream(
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
        (null, effect)
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
    let null_distribution =
        DistributionSummary::from_samples(&block_ratios_treatment_over_control(&null_samples))
            .expect("null distribution");
    eprintln!(
        "[quill-perf-paired] fixture={} null_median={:.6} null_ci95=[{:.6},{:.6}] \
         null_cv_pct={:.3} ab_median={:.6} ab_ci95=[{:.6},{:.6}] ab_cv_pct={:.3} \
         checksum={:016x}",
        spec.fixture,
        null_distribution.p50,
        null_distribution.median_ci95_low,
        null_distribution.median_ci95_high,
        null_distribution.cv_pct,
        paired_distribution.p50,
        paired_distribution.median_ci95_low,
        paired_distribution.median_ci95_high,
        paired_distribution.cv_pct,
        values_checksum(&null_samples) ^ values_checksum(&effect_samples).rotate_left(29),
    );

    let experiment = estimate_paired_experiment(&effect_samples, &null_samples, &evidence.config)
        .expect("paired estimator rejected harness-produced streams");
    let is_tokenizer_null = spec.metric == "tokenize_docs_per_second";
    let cold_cache = (spec.gate == PerfGate::Qg9).then(|| ColdCacheEvidence {
        procedure: "same-process index drop and reopen; the OS page cache is not dropped"
            .to_owned(),
        verified: false,
    });
    let cell = EvidenceCell::evaluate(
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
            cold_cache,
        },
        experiment,
        &evidence.policy,
    )
    .expect("evidence cell evaluation");

    let absolute_engine = if is_tokenizer_null {
        "quill_tokenizer"
    } else {
        EngineArm::Quill.label()
    };
    let results = vec![
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
            distribution: null_distribution,
        },
    ];
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
    lower_hex(&Sha256::digest(MANIFEST.as_bytes()))
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

fn build_profile_label() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        "release"
    }
}

fn build_identity(bench_elf_sha256: &str, revision: &str) -> BuildIdentity {
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
    let rustc_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned());
    BuildIdentity {
        executable_sha256: bench_elf_sha256.to_owned(),
        git_revision: revision.to_owned(),
        git_dirty,
        worktree_state_sha256,
        cargo_lock_sha256,
        rustc_version,
        target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        build_profile: build_profile_label().to_owned(),
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
    if matches!(spec.gate, PerfGate::Qg7 | PerfGate::Qg10) {
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

fn bench_matrix(c: &mut Criterion, bench_elf_sha256: &str) {
    let scale = MatrixScale::from_env();
    let context = BenchContext::new(scale);
    let matrix = PerfMatrixSpec::complete();
    validate_matrix(&matrix).expect("normative QG matrix");
    let selected = selected_cells(&matrix, scale);
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
    let evidence_context = EvidenceContext {
        config: PairedEstimatorConfig::predeclared(bootstrap_seed),
        policy: EvidencePolicy::predeclared(),
        sample_provenance: PerfSampleProvenance {
            run_id: run_id.clone(),
            executable_sha256: bench_elf_sha256.to_owned(),
            corpus_sha256: corpus_hash.clone(),
            worker_id: machine_fingerprint(),
            build_profile: build_profile_label().to_owned(),
        },
    };
    let mut machine = MachineIdentity::capture();

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

    let provenance = EvidenceProvenance {
        run_id: run_id.clone(),
        run_window: run_window.clone(),
        manifest_sha256: manifest_hash.clone(),
        build: build_identity(bench_elf_sha256, &revision),
        machine,
        peak_rss: PeakRssEvidence::capture(),
        corpus: corpus_identity(&context, &selected, &corpus_hash),
    };
    for (gate, cells) in evidence_by_gate {
        let artifact = PerfEvidenceArtifact::assemble(
            gate,
            evidence_context.policy.clone(),
            provenance.clone(),
            cells,
        )
        .expect("assemble QG evidence artifact");
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
            git_rev: revision.clone(),
            run_window: run_window.clone(),
            run_id: run_id.clone(),
            corpus_manifest_hash: corpus_hash.clone(),
            manifest_sha256: manifest_hash.clone(),
            cells,
            laws_attested: scale.is_full(),
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
    let context = BenchContext::new(MatrixScale::from_env());
    let hit_count = match arm {
        EngineArm::Quill => {
            let index = context
                .runtime
                .block_on(QuillIndex::open(
                    &context.cx,
                    &path,
                    pinned_quill_config(heap, threads),
                ))
                .expect("fresh-process Quill open");
            index
                .search_doc_ids(&context.cx, "term00001", 10)
                .expect("fresh-process Quill query")
                .len()
        }
        EngineArm::Tantivy => {
            TantivyIndex::open_with_benchmark_config(&path, heap, threads, positions)
                .expect("fresh-process Tantivy open")
                .search_doc_ids(&context.cx, "term00001", 10)
                .expect("fresh-process Tantivy query")
                .len()
        }
    };
    println!("quill-perf-child\t{hit_count}");
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

fn main() {
    if run_child_mode() {
        return;
    }
    let identity = print_bench_elf_sha256().expect("hash executing QG benchmark");
    let mut criterion = Criterion::default().configure_from_args();
    bench_matrix(&mut criterion, &identity.sha256);
    criterion.final_summary();
}
