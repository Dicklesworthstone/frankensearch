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

use std::path::Path;

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use tempfile::TempDir;

use frankensearch_core::types::{ScoreSource, ScoredResult, VectorHit};
use frankensearch_embed::hash_embedder::{HashAlgorithm, HashEmbedder};
use frankensearch_fusion::normalize::{min_max_normalize, z_score_normalize};
use frankensearch_fusion::rrf::{RrfConfig, rrf_fuse};
use frankensearch_index::{VectorIndex, dot_product_f32_f32};

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
            fast_score: None,
            quality_score: None,
            lexical_score: Some((n - i) as f32),
            rerank_score: None,
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
    let n = 10_000usize;

    for (label, delete_count) in [("10pct", 1_000usize), ("50pct", 5_000usize), ("90pct", 9_000usize)] {
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
    bench_rrf_fusion,
    bench_normalization,
    bench_index_io,
);
criterion_main!(benches);
