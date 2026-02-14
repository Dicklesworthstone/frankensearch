//! Quick benchmarks: latency spot-check for key operations (bd-3un.40).
//!
//! NOT a replacement for Criterion benchmarks — this provides a fast
//! developer sanity-check with colorized output.
//!
//! Run with: `cargo run --example bench_quick --release`

use std::sync::Arc;
use std::time::Instant;

use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::Embedder;
use frankensearch_core::types::VectorHit;
use frankensearch_embed::hash_embedder::{HashAlgorithm, HashEmbedder as HashEmbedderDirect};
use frankensearch_fusion::normalize::min_max_normalize;
use frankensearch_fusion::rrf::{RrfConfig, rrf_fuse};
use frankensearch_index::{VectorIndex, dot_product_f32_f32};

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
fn main() {
    println!("\n\x1b[1;36m=== frankensearch Quick Bench ===\x1b[0m\n");
    println!("  \x1b[2m(Tracing noise is expected; run with RUST_LOG=off to suppress)\x1b[0m\n");

    let dir =
        std::env::temp_dir().join(format!("frankensearch-bench-quick-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    // ── 1. Hash Embedder ──────────────────────────────────────────────────
    {
        let embedder = HashEmbedderDirect::new(384, HashAlgorithm::FnvModular);
        let text = "Rust ownership and borrowing prevents data races at compile time";
        let iters = 10_000;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = embedder.embed_sync(text);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / f64::from(iters);
        report("Hash embed (384d)", per_op, "ns/op", 50_000.0);
    }

    // ── 2. SIMD Dot Product ───────────────────────────────────────────────
    for dim in [256, 384] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();
        let iters = 100_000;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = dot_product_f32_f32(&a, &b);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() as f64 / f64::from(iters);
        report(&format!("Dot product ({dim}d)"), per_op, "ns/op", 5_000.0);
    }

    // ── 3. Vector Index Write + Open ──────────────────────────────────────
    {
        let dim = 256;
        let n = 1000;
        let idx_path = dir.join("bench_vectors.idx");

        let corpus: Vec<(String, Vec<f32>)> = (0..n)
            .map(|i| {
                let id = format!("doc-{i:06}");
                let vec: Vec<f32> = (0..dim)
                    .map(|d| ((i * 31 + d) as f32 * 0.001).sin())
                    .collect();
                (id, vec)
            })
            .collect();

        // Write
        let start = Instant::now();
        let mut writer = VectorIndex::create(&idx_path, "bench", dim).expect("create");
        for (id, vec) in &corpus {
            writer.write_record(id, vec).expect("write");
        }
        writer.finish().expect("finish");
        let write_ms = start.elapsed().as_secs_f64() * 1000.0;
        report("Index write (1k x 256d)", write_ms, "ms", 100.0);

        // Open
        let start = Instant::now();
        let iters = 100;
        for _ in 0..iters {
            let _ = VectorIndex::open(&idx_path).expect("open");
        }
        let open_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        report("Index open (1k x 256d)", open_ms, "ms", 10.0);

        // Search
        let index = VectorIndex::open(&idx_path).expect("open");
        let query = &corpus[0].1;
        let start = Instant::now();
        let search_iters = 100;
        for _ in 0..search_iters {
            let _ = index.search_top_k(query, 10, None);
        }
        let search_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(search_iters);
        report("Brute top-10 (1k x 256d)", search_ms, "ms", 5.0);
    }

    // ── 4. RRF Fusion ─────────────────────────────────────────────────────
    {
        let n = 500;
        let lexical: Vec<_> = (0..n)
            .map(|i| frankensearch_core::types::ScoredResult {
                doc_id: format!("doc-{i:06}"),
                score: (n - i) as f32,
                source: frankensearch_core::types::ScoreSource::Lexical,
                fast_score: None,
                quality_score: None,
                lexical_score: Some((n - i) as f32),
                rerank_score: None,
                metadata: None,
            })
            .collect();
        #[allow(clippy::cast_sign_loss)]
        let semantic: Vec<VectorHit> = (0..n)
            .map(|i| VectorHit {
                index: i as u32,
                score: 1.0 - (i as f32 / n as f32),
                doc_id: format!("sem-{i:06}"),
            })
            .collect();
        let config = RrfConfig::default();
        let iters = 1000;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = rrf_fuse(&lexical, &semantic, 10, 0, &config);
        }
        let per_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        report("RRF fuse (500+500 -> 10)", per_ms, "ms", 1.0);
    }

    // ── 5. Score Normalization ────────────────────────────────────────────
    {
        let n = 10_000;
        let iters = 1000;
        let base: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();

        let start = Instant::now();
        for _ in 0..iters {
            let mut scores = base.clone();
            min_max_normalize(&mut scores);
        }
        let per_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
        report("MinMax normalize (10k)", per_ms, "ms", 1.0);
    }

    // ── 6. Full Pipeline (build + search) ─────────────────────────────────
    {
        let corpus = vec![
            (
                "doc-001",
                "Rust ownership and borrowing prevents data races",
            ),
            ("doc-002", "Machine learning models require training data"),
            (
                "doc-003",
                "Distributed consensus algorithms ensure fault tolerance",
            ),
            (
                "doc-004",
                "Database indexing with B-trees provides fast lookups",
            ),
            (
                "doc-005",
                "Container orchestration with Kubernetes manages deployments",
            ),
        ];

        // Build
        let build_dir = dir.join("pipeline");
        std::fs::create_dir_all(&build_dir).expect("mkdir");

        let start = Instant::now();
        asupersync::test_utils::run_test_with_cx(|cx| {
            let build_dir = build_dir.clone();
            let corpus = corpus.clone();
            async move {
                let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
                let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
                let stack = EmbedderStack::from_parts(fast, Some(quality));

                let mut builder = IndexBuilder::new(&build_dir).with_embedder_stack(stack);
                for (id, text) in &corpus {
                    builder = builder.add_document(*id, *text);
                }
                builder.build(&cx).await.expect("build");
            }
        });
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        report("Pipeline build (5 docs)", build_ms, "ms", 200.0);

        // Search
        let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());
        let index =
            Arc::new(TwoTierIndex::open(&build_dir, TwoTierConfig::default()).expect("open"));
        let searcher = TwoTierSearcher::new(
            Arc::clone(&index),
            Arc::clone(&fast),
            TwoTierConfig::default(),
        )
        .with_quality_embedder(quality);

        let search_iters = 100;
        let start = Instant::now();
        for _ in 0..search_iters {
            asupersync::test_utils::run_test_with_cx(|cx| {
                let searcher = &searcher;
                async move {
                    let _ = searcher.search_collect(&cx, "Rust ownership", 5).await;
                }
            });
        }
        let search_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(search_iters);
        report("Two-tier search (5 docs)", search_ms, "ms", 10.0);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&dir);

    println!();
    println!("\x1b[1;36m=== Done ===\x1b[0m");
    println!();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn report(name: &str, value: f64, unit: &str, budget: f64) {
    let status = if value <= budget {
        "\x1b[32mOK\x1b[0m"
    } else {
        "\x1b[33mSLOW\x1b[0m"
    };
    println!("  [{status}] {name:<35} {value:>10.1} {unit:<6} (budget: {budget:.1} {unit})",);
}
