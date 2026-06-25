//! Criterion benchmarks for frankensearch performance-critical paths.
//!
//! Run with: `cargo bench -p frankensearch`
//!
//! Benchmark groups:
//! 1. SIMD dot product (f32, various dimensions)
//! 2. Hash embedder (short/medium/long text)
//! 3. Vector search (brute-force top-k at various corpus sizes)
//! 4. RRF fusion (various result counts)
//! 5. Score normalization (various sizes)
//! 6. Vector index I/O (write/open)
//! 7. BOLD-VERIFY Tantivy-class incumbent vs frankensearch hybrid

#[cfg(feature = "lexical")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "lexical")]
use std::fs::File;
#[cfg(feature = "lexical")]
use std::hash::{Hash, Hasher};
use std::hint::black_box;
#[cfg(feature = "lexical")]
use std::io::{BufWriter, Write};
use std::path::Path;
#[cfg(feature = "lexical")]
use std::path::PathBuf;
#[cfg(feature = "lexical")]
use std::sync::{Arc, OnceLock};
#[cfg(feature = "lexical")]
use std::time::{Duration, Instant};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

#[cfg(feature = "lexical")]
use frankensearch_core::query_class::QueryClass;
#[cfg(feature = "lexical")]
use frankensearch_core::traits::LexicalSearch;
#[cfg(feature = "lexical")]
use frankensearch_core::types::IndexableDocument;
use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_embed::hash_embedder::{HashAlgorithm, HashEmbedder};
use frankensearch_fusion::normalize::{min_max_normalize, z_score_normalize};
use frankensearch_fusion::rrf::{RrfConfig, rrf_fuse};
use frankensearch_index::{VectorIndex, dot_product_f32_f32};
#[cfg(feature = "lexical")]
use frankensearch_lexical::TantivyIndex;

#[cfg(feature = "lexical")]
static BOLD_VERIFY_SUMMARY_ONCE: OnceLock<()> = OnceLock::new();

// ─── Helpers ────────────────────────────────────────────────────────────────

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let x = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64) as f32;
            (x * 1e-10).sin()
        })
        .collect()
}

fn build_corpus(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
    (0..n)
        .map(|i| {
            let doc_id = format!("doc-{i:06}");
            #[allow(clippy::cast_precision_loss)]
            let vec = random_vector(dim, i as u64);
            (doc_id, vec)
        })
        .collect()
}

#[cfg(feature = "lexical")]
#[derive(Clone, Copy)]
struct BoldVerifyQuery {
    class: &'static str,
    text: &'static str,
    limit: usize,
}

#[cfg(feature = "lexical")]
struct BoldVerifyFixture {
    doc_count: usize,
    corpus_hash: String,
    lexical: Arc<TantivyIndex>,
    vector: VectorIndex,
    embedder: HashEmbedder,
    _vector_dir: TempDir,
}

#[cfg(feature = "lexical")]
#[derive(Clone, Copy)]
struct LatencyStats {
    p50: u128,
    p95: u128,
    p99: u128,
}

#[cfg(feature = "lexical")]
const BOLD_VERIFY_TOP10_QUERIES: &[BoldVerifyQuery] = &[
    BoldVerifyQuery {
        class: "exact_identifier",
        text: "doc 000042",
        limit: 10,
    },
    BoldVerifyQuery {
        class: "short_keyword",
        text: "rust ownership",
        limit: 10,
    },
    BoldVerifyQuery {
        class: "quoted_phrase",
        text: "\"reciprocal rank fusion\"",
        limit: 10,
    },
    BoldVerifyQuery {
        class: "natural_language",
        text: "how do vector embeddings improve local search relevance",
        limit: 10,
    },
    BoldVerifyQuery {
        class: "high_fanout",
        text: "search",
        limit: 10,
    },
    BoldVerifyQuery {
        class: "zero_hit",
        text: "zzzz nohit sentinel",
        limit: 10,
    },
];

#[cfg(feature = "lexical")]
const BOLD_VERIFY_LIMIT_ALL_QUERY: BoldVerifyQuery = BoldVerifyQuery {
    class: "limit_all",
    text: "search",
    limit: usize::MAX,
};

#[cfg(feature = "lexical")]
fn bold_verify_content(i: usize) -> String {
    let theme = match i % 6 {
        0 => "Rust ownership borrowing lifetimes Result error handling async future executor",
        1 => {
            "vector embeddings cosine similarity HNSW approximate nearest neighbor f16 quantization"
        }
        2 => "BM25 scoring algorithm reciprocal rank fusion hybrid search Tantivy Lucene",
        3 => "docker compose kubernetes health check prometheus grafana monitoring",
        4 => "transformer text embeddings ONNX Runtime inference semantic ranking",
        _ => "sourdough bread starter fermentation chocolate chip cookies recipe",
    };
    format!(
        "doc {i:06} title document {i:06}. {theme}. common search corpus term. \
         This deterministic benchmark document keeps the same content for Tantivy-only \
         and frankensearch hybrid retrieval."
    )
}

#[cfg(feature = "lexical")]
fn build_bold_verify_docs(doc_count: usize) -> Vec<IndexableDocument> {
    (0..doc_count)
        .map(|i| {
            IndexableDocument::new(format!("doc-{i:06}"), bold_verify_content(i))
                .with_title(format!("document {i:06}"))
                .with_metadata("cluster", (i % 6).to_string())
        })
        .collect()
}

#[cfg(feature = "lexical")]
fn corpus_hash(docs: &[IndexableDocument]) -> String {
    let mut hasher = DefaultHasher::new();
    for doc in docs {
        doc.id.hash(&mut hasher);
        doc.content.hash(&mut hasher);
        doc.title.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

#[cfg(feature = "lexical")]
fn build_bold_verify_fixture(doc_count: usize) -> BoldVerifyFixture {
    let docs = build_bold_verify_docs(doc_count);
    let hash = corpus_hash(&docs);
    let embedder = HashEmbedder::default_384();
    let vector_dir = TempDir::new().expect("create vector tempdir");
    let vector_path = vector_dir.path().join("bold_verify.fast.idx");
    let mut writer = VectorIndex::create(&vector_path, "fnv1a-384", 384)
        .expect("create bold-verify vector index");
    for doc in &docs {
        let embedding = embedder.embed_sync(&doc.content);
        writer
            .write_record(&doc.id, &embedding)
            .expect("write bold-verify vector record");
    }
    writer.finish().expect("finish bold-verify vector index");
    let vector = VectorIndex::open(&vector_path).expect("open bold-verify vector index");

    let lexical = Arc::new(TantivyIndex::in_memory().expect("create tantivy comparator index"));
    let lexical_for_index = Arc::clone(&lexical);
    let docs_for_index = docs.clone();
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        lexical_for_index
            .index_documents(&cx, &docs_for_index)
            .await
            .expect("index tantivy comparator corpus");
        lexical_for_index
            .commit(&cx)
            .await
            .expect("commit tantivy comparator corpus");
    });

    BoldVerifyFixture {
        doc_count,
        corpus_hash: hash,
        lexical,
        vector,
        embedder,
        _vector_dir: vector_dir,
    }
}

fn write_index(dir: &Path, corpus: &[(String, Vec<f32>)], dim: usize) {
    let path = dir.join("vector.fast.idx");
    let mut writer = VectorIndex::create(&path, "bench-embedder", dim).unwrap();
    for (doc_id, vec) in corpus {
        writer.write_record(doc_id, vec).unwrap();
    }
    writer.finish().unwrap();
}

#[allow(clippy::cast_precision_loss)]
fn make_lexical_hits(n: usize) -> Vec<ScoredResult> {
    (0..n)
        .map(|i| ScoredResult {
            doc_id: format!("doc-{i:06}"),
            score: (n - i) as f32,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some((n - i) as f32),
            rerank_score: None,
            explanation: None,
            metadata: None,
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn make_semantic_hits(n: usize) -> Vec<VectorHit> {
    (0..n)
        .map(|i| VectorHit {
            index: u32::try_from(i).unwrap_or(u32::MAX),
            score: 1.0 - (i as f32 / n as f32),
            doc_id: format!("sem-{i:06}"),
        })
        .collect()
}

#[cfg(feature = "lexical")]
fn lexical_doc_ids_as_scored(results: &[frankensearch_lexical::LexicalIdHit]) -> Vec<ScoredResult> {
    results
        .iter()
        .map(|hit| ScoredResult {
            doc_id: hit.doc_id.clone(),
            score: hit.bm25_score,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(hit.bm25_score),
            rerank_score: None,
            explanation: None,
            metadata: None,
        })
        .collect()
}

#[cfg(feature = "lexical")]
fn bold_verify_lexical_short_circuit(
    query: &BoldVerifyQuery,
    lexical_count: usize,
    limit: usize,
) -> bool {
    lexical_count >= limit
        && matches!(
            QueryClass::classify(query.text),
            QueryClass::Identifier | QueryClass::ShortKeyword
        )
}

#[cfg(feature = "lexical")]
fn tantivy_only_search(fixture: &BoldVerifyFixture, cx: &asupersync::Cx, query: &BoldVerifyQuery) {
    let limit = query.limit.min(fixture.doc_count);
    black_box(
        fixture
            .lexical
            .search_doc_ids(cx, black_box(query.text), black_box(limit))
            .expect("tantivy comparator search"),
    );
}

#[cfg(feature = "lexical")]
fn frankensearch_hash_hybrid_search(
    fixture: &BoldVerifyFixture,
    cx: &asupersync::Cx,
    query: &BoldVerifyQuery,
) {
    let limit = query.limit.min(fixture.doc_count);
    let candidate_limit = limit.saturating_mul(3).min(fixture.doc_count).max(limit);
    let lexical_hits = fixture
        .lexical
        .search_doc_ids(cx, query.text, candidate_limit)
        .expect("hybrid lexical candidate search");
    let lexical = lexical_doc_ids_as_scored(&lexical_hits);
    if bold_verify_lexical_short_circuit(query, lexical.len(), limit) {
        black_box(lexical);
        return;
    }
    let query_embedding = fixture.embedder.embed_sync(query.text);
    let semantic = fixture
        .vector
        .search_top_k(&query_embedding, candidate_limit, None)
        .expect("hybrid vector candidate search");
    black_box(rrf_fuse(
        &lexical,
        &semantic,
        black_box(limit),
        0,
        &RrfConfig::default(),
    ));
}

#[cfg(feature = "lexical")]
fn percentile(sorted: &[u128], pct: usize) -> u128 {
    let len = sorted.len();
    let index = len.saturating_mul(pct).div_ceil(100).saturating_sub(1);
    sorted[index.min(len.saturating_sub(1))]
}

#[cfg(feature = "lexical")]
fn measure_latency_us(mut f: impl FnMut(), samples: usize) -> LatencyStats {
    for _ in 0..5 {
        f();
    }
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        f();
        timings.push(start.elapsed().as_micros());
    }
    timings.sort_unstable();
    LatencyStats {
        p50: percentile(&timings, 50),
        p95: percentile(&timings, 95),
        p99: percentile(&timings, 99),
    }
}

#[cfg(feature = "lexical")]
fn current_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(pages.saturating_mul(4096))
}

#[cfg(feature = "lexical")]
fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |sha| sha.trim().to_owned())
}

#[cfg(feature = "lexical")]
fn worker_id() -> String {
    std::env::var("RCH_WORKER")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned())
}

#[cfg(feature = "lexical")]
fn bold_verify_output_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FRANKENSEARCH_BOLD_VERIFY_OUT") {
        return Some(PathBuf::from(path));
    }
    std::env::var_os("FRANKENSEARCH_BOLD_VERIFY_EMIT")?;
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        return Some(PathBuf::from(target_dir).join("criterion/bold_verify"));
    }
    Some(PathBuf::from("target").join("criterion/bold_verify"))
}

#[cfg(feature = "lexical")]
fn emit_bold_verify_summary(fixtures: &[BoldVerifyFixture]) {
    let Some(out_dir) = bold_verify_output_dir() else {
        return;
    };
    std::fs::create_dir_all(&out_dir).expect("create bold-verify output dir");
    let jsonl_path = out_dir.join("summary.jsonl");
    let markdown_path = out_dir.join("summary.md");
    let mut jsonl = BufWriter::new(File::create(&jsonl_path).expect("create JSONL summary"));
    let mut markdown = BufWriter::new(File::create(&markdown_path).expect("create MD summary"));
    writeln!(
        markdown,
        "| workload | corpus_docs | query_class | incumbent_p50_us | frankensearch_p50_us | ratio | p95_ratio | p99_ratio | corpus_hash |"
    )
    .expect("write MD header");
    writeln!(
        markdown,
        "|----------|-------------|-------------|------------------|----------------------|-------|-----------|-----------|-------------|"
    )
    .expect("write MD separator");

    let cx = asupersync::Cx::for_testing();
    let sha = git_sha();
    let worker = worker_id();
    let command = std::env::var("FRANKENSEARCH_BOLD_VERIFY_COMMAND").unwrap_or_else(|_| {
        "CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1".to_owned()
    });

    for fixture in fixtures {
        for query in BOLD_VERIFY_TOP10_QUERIES
            .iter()
            .copied()
            .chain(std::iter::once(BOLD_VERIFY_LIMIT_ALL_QUERY))
        {
            if query.limit == usize::MAX && fixture.doc_count > 10_000 {
                continue;
            }
            let workload = if query.limit == usize::MAX {
                format!("bold_verify/limit_all/{}", fixture.doc_count)
            } else {
                format!("bold_verify/top10/{}", fixture.doc_count)
            };
            let samples = if fixture.doc_count >= 100_000 { 12 } else { 25 };
            let incumbent =
                measure_latency_us(|| tantivy_only_search(fixture, &cx, &query), samples);
            let challenger = measure_latency_us(
                || frankensearch_hash_hybrid_search(fixture, &cx, &query),
                samples,
            );
            #[allow(clippy::cast_precision_loss)]
            let ratio = challenger.p50 as f64 / incumbent.p50.max(1) as f64;
            #[allow(clippy::cast_precision_loss)]
            let p95_ratio = challenger.p95 as f64 / incumbent.p95.max(1) as f64;
            #[allow(clippy::cast_precision_loss)]
            let p99_ratio = challenger.p99 as f64 / incumbent.p99.max(1) as f64;
            let row = serde_json::json!({
                "workload": workload,
                "corpus_docs": fixture.doc_count,
                "query_class": query.class,
                "query": query.text,
                "incumbent": "tantivy_doc_ids",
                "frankensearch": "hash_hybrid_lexical_short_circuit_vector_rrf",
                "incumbent_p50_us": incumbent.p50,
                "incumbent_p95_us": incumbent.p95,
                "incumbent_p99_us": incumbent.p99,
                "frankensearch_p50_us": challenger.p50,
                "frankensearch_p95_us": challenger.p95,
                "frankensearch_p99_us": challenger.p99,
                "ratio": ratio,
                "p95_ratio": p95_ratio,
                "p99_ratio": p99_ratio,
                "rss_bytes": current_rss_bytes(),
                "git_sha": sha,
                "worker": worker,
                "command": command,
                "corpus_hash": fixture.corpus_hash,
            });
            writeln!(jsonl, "{row}").expect("write JSONL row");
            println!("BOLD_VERIFY_JSONL {row}");
            writeln!(
                markdown,
                "| {workload} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {} |",
                fixture.doc_count,
                query.class,
                incumbent.p50,
                challenger.p50,
                ratio,
                p95_ratio,
                p99_ratio,
                fixture.corpus_hash,
            )
            .expect("write MD row");
            println!(
                "BOLD_VERIFY_MD | {workload} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {} |",
                fixture.doc_count,
                query.class,
                incumbent.p50,
                challenger.p50,
                ratio,
                p95_ratio,
                p99_ratio,
                fixture.corpus_hash,
            );
        }
    }
    println!(
        "BOLD_VERIFY_ARTIFACTS jsonl={} markdown={}",
        jsonl_path.display(),
        markdown_path.display()
    );
}

// ─── 1. SIMD Dot Product ────────────────────────────────────────────────────

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_f32");

    for dim in [128, 256, 384, 768] {
        let a = random_vector(dim, 42);
        let b = random_vector(dim, 99);

        group.bench_function(BenchmarkId::from_parameter(dim), |bencher| {
            bencher.iter(|| dot_product_f32_f32(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

// ─── 2. Hash Embedder ───────────────────────────────────────────────────────

fn bench_hash_embedder(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_embedder");
    let embedder = HashEmbedder::new(384, HashAlgorithm::FnvModular);

    let short = "Rust is a systems programming language focused on safety";
    let medium = short.repeat(10);
    let long = short.repeat(100);

    group.bench_function("short_10w", |b| {
        b.iter(|| embedder.embed_sync(black_box(short)));
    });
    group.bench_function("medium_100w", |b| {
        b.iter(|| embedder.embed_sync(black_box(&medium)));
    });
    group.bench_function("long_1000w", |b| {
        b.iter(|| embedder.embed_sync(black_box(&long)));
    });

    group.finish();
}

// ─── 3. Vector Search (brute-force top-k) ───────────────────────────────────

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_topk");
    group.sample_size(20);

    let dim = 384;

    for n in [1_000, 5_000, 10_000] {
        let dir = TempDir::new().unwrap();
        let corpus = build_corpus(n, dim);
        write_index(dir.path(), &corpus, dim);

        let idx_path = dir.path().join("vector.fast.idx");
        let index = VectorIndex::open(&idx_path).unwrap();
        let query = random_vector(dim, 12345);

        group.bench_function(BenchmarkId::new("top10", n), |b| {
            b.iter(|| index.search_top_k(black_box(&query), black_box(10), None));
        });
    }

    group.finish();
}

// ─── 3d. BOLD-VERIFY Tantivy-Class Comparator ──────────────────────────────

/// Warm-query head-to-head against a Tantivy/Lucene-class incumbent.
///
/// This is deliberately a comparator harness, not a dominance claim: both sides
/// use the same generated documents and query stream. The incumbent side is
/// Tantivy BM25 identifiers only; the frankensearch side is hash embedding +
/// FSVI vector search + Tantivy candidates + RRF fusion.
#[cfg(feature = "lexical")]
fn bench_tantivy_class_comparator(c: &mut Criterion) {
    let mut group = c.benchmark_group("bold_verify_tantivy_class");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    let cx = asupersync::Cx::for_testing();
    let fixtures = [
        build_bold_verify_fixture(10_000),
        build_bold_verify_fixture(100_000),
    ];
    BOLD_VERIFY_SUMMARY_ONCE.get_or_init(|| emit_bold_verify_summary(&fixtures));

    for fixture in &fixtures {
        for query in BOLD_VERIFY_TOP10_QUERIES
            .iter()
            .copied()
            .chain(std::iter::once(BOLD_VERIFY_LIMIT_ALL_QUERY))
        {
            if query.limit == usize::MAX && fixture.doc_count > 10_000 {
                continue;
            }
            let workload = if query.limit == usize::MAX {
                format!("limit_all/{}", query.class)
            } else {
                format!("top10/{}", query.class)
            };

            group.bench_function(
                BenchmarkId::new(format!("tantivy_doc_ids/{workload}"), fixture.doc_count),
                |b| {
                    b.iter(|| tantivy_only_search(black_box(fixture), &cx, black_box(&query)));
                },
            );
            group.bench_function(
                BenchmarkId::new(
                    format!("frankensearch_hash_hybrid/{workload}"),
                    fixture.doc_count,
                ),
                |b| {
                    b.iter(|| {
                        frankensearch_hash_hybrid_search(
                            black_box(fixture),
                            &cx,
                            black_box(&query),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "lexical"))]
fn bench_tantivy_class_comparator(c: &mut Criterion) {
    let mut group = c.benchmark_group("bold_verify_tantivy_class");
    group.bench_function("enable_lexical_feature", |b| {
        b.iter(|| black_box("run with --features lexical"));
    });
    group.finish();
}

// ─── 3b. Vector Search Tombstone Overhead ──────────────────────────────────

fn bench_vector_search_tombstone_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_tombstone_overhead");
    group.sample_size(12);

    let dim = 384;
    let n = 10_000usize;

    for (label, delete_count) in [
        ("0pct", 0usize),
        ("10pct", 1_000usize),
        ("50pct", 5_000usize),
        ("90pct", 9_000usize),
    ] {
        let dir = TempDir::new().unwrap();
        let corpus = build_corpus(n, dim);
        write_index(dir.path(), &corpus, dim);

        let idx_path = dir.path().join("vector.fast.idx");
        let mut index = VectorIndex::open(&idx_path).unwrap();
        let tombstoned_doc_ids: Vec<String> =
            (0..delete_count).map(|i| format!("doc-{i:06}")).collect();
        let refs: Vec<&str> = tombstoned_doc_ids.iter().map(String::as_str).collect();
        if !refs.is_empty() {
            index.soft_delete_batch(&refs).unwrap();
        }

        let query = random_vector(dim, 98_765);

        group.bench_function(BenchmarkId::new("top10", label), |b| {
            b.iter(|| index.search_top_k(black_box(&query), black_box(10), None));
        });
    }

    group.finish();
}

// ─── 3c. Vacuum Time by Tombstone Ratio ───────────────────────────────────

fn bench_vector_vacuum_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_vacuum_time");
    group.sample_size(10);

    let dim = 384;
    let n = 2_000usize;

    for (label, delete_count) in [("10pct", n / 10), ("50pct", n / 2), ("90pct", (n * 9) / 10)] {
        group.bench_function(BenchmarkId::new("vacuum", label), |b| {
            b.iter_batched(
                || {
                    let dir = TempDir::new().unwrap();
                    let corpus = build_corpus(n, dim);
                    write_index(dir.path(), &corpus, dim);

                    let idx_path = dir.path().join("vector.fast.idx");
                    let mut index = VectorIndex::open(&idx_path).unwrap();
                    let tombstoned_doc_ids: Vec<String> =
                        (0..delete_count).map(|i| format!("doc-{i:06}")).collect();
                    let refs: Vec<&str> = tombstoned_doc_ids.iter().map(String::as_str).collect();
                    if !refs.is_empty() {
                        index.soft_delete_batch(&refs).unwrap();
                    }
                    (dir, index)
                },
                |(_dir, mut index)| {
                    black_box(index.vacuum().unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ─── 4. RRF Fusion ──────────────────────────────────────────────────────────

fn bench_rrf_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_fusion");
    let config = RrfConfig::default();

    for n in [50, 100, 500, 1000] {
        let lexical = make_lexical_hits(n);
        let semantic = make_semantic_hits(n);

        group.bench_function(BenchmarkId::new("fuse", format!("{n}+{n}")), |b| {
            b.iter(|| {
                rrf_fuse(
                    black_box(&lexical),
                    black_box(&semantic),
                    black_box(10),
                    0,
                    &config,
                )
            });
        });
    }

    group.finish();
}

// ─── 5. Score Normalization ─────────────────────────────────────────────────

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_normalization");

    for n in [100, 1_000, 10_000] {
        #[allow(clippy::cast_precision_loss)]
        let scores: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

        group.bench_function(BenchmarkId::new("min_max", n), |b| {
            b.iter_batched(
                || scores.clone(),
                |mut s| min_max_normalize(black_box(&mut s)),
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("z_score", n), |b| {
            b.iter_batched(
                || scores.clone(),
                |mut s| z_score_normalize(black_box(&mut s)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ─── 6. Vector Index I/O ────────────────────────────────────────────────────

fn bench_index_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_io");
    group.sample_size(10);

    let dim = 384;

    // Write benchmark.
    for n in [1_000, 10_000] {
        let corpus = build_corpus(n, dim);

        group.bench_function(BenchmarkId::new("write", n), |b| {
            b.iter_batched(
                TempDir::new,
                |dir| {
                    let dir = dir.unwrap();
                    write_index(dir.path(), black_box(&corpus), dim);
                },
                BatchSize::PerIteration,
            );
        });
    }

    // Open/read benchmark.
    for n in [1_000, 10_000] {
        let dir = TempDir::new().unwrap();
        let corpus = build_corpus(n, dim);
        write_index(dir.path(), &corpus, dim);
        let idx_path = dir.path().join("vector.fast.idx");

        group.bench_function(BenchmarkId::new("open", n), |b| {
            b.iter(|| VectorIndex::open(black_box(&idx_path)));
        });
    }

    group.finish();
}

// ─── Group Registration ─────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_dot_product,
    bench_hash_embedder,
    bench_vector_search,
    bench_vector_search_tombstone_overhead,
    bench_vector_vacuum_time,
    bench_tantivy_class_comparator,
    bench_rrf_fusion,
    bench_normalization,
    bench_index_io,
);
criterion_main!(benches);
