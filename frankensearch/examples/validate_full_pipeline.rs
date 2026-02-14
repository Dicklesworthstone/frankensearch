//! E2E validation: full frankensearch search pipeline (bd-3un.40).
//!
//! Exercises index building, two-tier search, phase progression, and
//! result quality. Requires only the `hash` feature (no ML downloads).
//!
//! Run with: `cargo run --example validate_full_pipeline`

use std::sync::{Arc, Mutex};
use std::time::Instant;

use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::Embedder;
use frankensearch_core::types::SearchPhase;

#[allow(clippy::too_many_lines)]
fn main() {
    let start = Instant::now();
    let mut pass = 0u32;
    let mut fail = 0u32;
    let warn = 0u32;

    println!("\n\x1b[1;36m=== frankensearch E2E: Full Pipeline Validation ===\x1b[0m\n");

    // ── Step 1: Build index ─────────────────────────────────────────────
    log_info("INDEX", "Building index from 20-document test corpus...");
    let dir =
        std::env::temp_dir().join(format!("frankensearch-e2e-pipeline-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let fast_embedder = HashEmbedder::default_256();
    let quality_embedder = HashEmbedder::default_384();

    asupersync::test_utils::run_test_with_cx(|cx| {
        let dir = dir.clone();
        async move {
            let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
            let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
            let stack = EmbedderStack::from_parts(fast, Some(quality));

            let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
            for (id, text) in TEST_CORPUS {
                builder = builder.add_document(*id, *text);
            }
            let stats = builder.build(&cx).await.expect("build index");

            log_info(
                "INDEX",
                &format!(
                    "Built: {} docs, {} errors, quality={}, {:.1}ms",
                    stats.doc_count, stats.error_count, stats.has_quality_index, stats.total_ms
                ),
            );
            assert_eq!(stats.doc_count, TEST_CORPUS.len());
        }
    });
    check(&mut pass, &mut fail, "Index build", true);

    // ── Step 2: Open index and create searcher ──────────────────────────
    log_info("SEARCH", "Opening index and creating two-tier searcher...");
    let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open index"));
    let fast: Arc<dyn Embedder> = Arc::new(fast_embedder);
    let quality_arc: Arc<dyn Embedder> = Arc::new(quality_embedder);

    let searcher = TwoTierSearcher::new(
        Arc::clone(&index),
        Arc::clone(&fast),
        TwoTierConfig::default(),
    )
    .with_quality_embedder(Arc::clone(&quality_arc));

    check(&mut pass, &mut fail, "Searcher creation", true);

    // ── Step 3: Execute queries ─────────────────────────────────────────
    log_info("SEARCH", "Running query suite...");
    let mut total_phase1_ms = 0.0f64;
    let mut total_phase2_ms = 0.0f64;
    let mut query_pass = 0u32;
    let mut query_fail = 0u32;

    for (query, expected_top) in QUERY_SUITE {
        let result = Arc::new(Mutex::new((false, 0.0f64, 0.0f64)));
        let result_inner = result.clone();
        asupersync::test_utils::run_test_with_cx(|cx| {
            let searcher = &searcher;
            async move {
                let metrics = searcher
                    .search(&cx, query, 10, |_| None, |_phase| {})
                    .await
                    .expect("search");

                let (results, _) = searcher
                    .search_collect(&cx, query, 10)
                    .await
                    .expect("collect");
                let top_ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();

                let found = top_ids.iter().any(|id| expected_top.contains(id));
                if found {
                    log_pass(
                        "QUERY",
                        &format!(
                            "\"{}\" -> top: [{}] phase1={:.1}ms phase2={:.1}ms",
                            query,
                            top_ids
                                .iter()
                                .take(3)
                                .copied()
                                .collect::<Vec<_>>()
                                .join(", "),
                            metrics.phase1_total_ms,
                            metrics.phase2_total_ms,
                        ),
                    );
                } else {
                    log_fail(
                        "QUERY",
                        &format!(
                            "\"{query}\" -> expected one of {expected_top:?} in top-10, got: {top_ids:?}",
                        ),
                    );
                }

                *result_inner.lock().unwrap() =
                    (found, metrics.phase1_total_ms, metrics.phase2_total_ms);
            }
        });
        let (found, p1, p2) = *result.lock().unwrap();
        if found {
            query_pass += 1;
        } else {
            query_fail += 1;
        }
        total_phase1_ms += p1;
        total_phase2_ms += p2;
    }

    #[allow(clippy::cast_precision_loss)]
    let n_queries = QUERY_SUITE.len() as f64;
    log_info(
        "SEARCH",
        &format!(
            "{} queries: {} pass, {} fail, avg phase1={:.1}ms phase2={:.1}ms",
            QUERY_SUITE.len(),
            query_pass,
            query_fail,
            total_phase1_ms / n_queries,
            total_phase2_ms / n_queries,
        ),
    );
    pass += query_pass;
    fail += query_fail;

    // ── Step 4: Phase contract tests ────────────────────────────────────
    log_info("PHASE", "Verifying progressive search phase contract...");

    // fast_only → exactly 1 phase (Initial)
    let fast_only_ok = Arc::new(Mutex::new(false));
    let fast_only_ok_inner = fast_only_ok.clone();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let index = Arc::clone(&index);
        let fast = Arc::clone(&fast);
        async move {
            let config = TwoTierConfig {
                fast_only: true,
                ..Default::default()
            };
            let s = TwoTierSearcher::new(index, fast, config);
            let mut count = 0;
            s.search(&cx, "test query", 5, |_| None, |_| count += 1)
                .await
                .expect("search");
            *fast_only_ok_inner.lock().unwrap() = count == 1;
        }
    });
    check(
        &mut pass,
        &mut fail,
        "fast_only yields 1 phase",
        *fast_only_ok.lock().unwrap(),
    );

    // two-tier → exactly 2 phases (Initial + Refined)
    let two_tier_ok = Arc::new(Mutex::new(false));
    let two_tier_ok_inner = two_tier_ok.clone();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let index = Arc::clone(&index);
        let fast = Arc::clone(&fast);
        let quality = Arc::clone(&quality_arc);
        async move {
            let s = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
                .with_quality_embedder(quality);
            let mut phases = Vec::new();
            s.search(
                &cx,
                "test query",
                5,
                |_| None,
                |p| {
                    phases.push(match p {
                        SearchPhase::Initial { .. } => "Initial",
                        SearchPhase::Refined { .. } => "Refined",
                        SearchPhase::RefinementFailed { .. } => "Failed",
                    });
                },
            )
            .await
            .expect("search");
            *two_tier_ok_inner.lock().unwrap() = phases == vec!["Initial", "Refined"];
        }
    });
    check(
        &mut pass,
        &mut fail,
        "two-tier yields Initial+Refined",
        *two_tier_ok.lock().unwrap(),
    );

    // ── Step 5: Persistence round-trip ──────────────────────────────────
    log_info("PERSIST", "Verifying index persistence round-trip...");
    let persist_ok = Arc::new(Mutex::new(false));
    let persist_ok_inner = persist_ok.clone();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let dir = dir.clone();
        let fast = Arc::clone(&fast);
        async move {
            let idx1 = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open1"));
            let s1 = TwoTierSearcher::new(idx1, Arc::clone(&fast), TwoTierConfig::default());
            let (r1, _) = s1
                .search_collect(&cx, "neural networks", 5)
                .await
                .expect("search1");

            drop(s1);
            let idx2 = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open2"));
            let s2 = TwoTierSearcher::new(idx2, fast, TwoTierConfig::default());
            let (r2, _) = s2
                .search_collect(&cx, "neural networks", 5)
                .await
                .expect("search2");

            *persist_ok_inner.lock().unwrap() =
                r1.len() == r2.len() && r1.iter().zip(r2.iter()).all(|(a, b)| a.doc_id == b.doc_id);
        }
    });
    check(
        &mut pass,
        &mut fail,
        "Persistence round-trip",
        *persist_ok.lock().unwrap(),
    );

    // ── Step 6: Empty/edge cases ────────────────────────────────────────
    log_info("EDGE", "Verifying edge case handling...");
    let empty_ok = Arc::new(Mutex::new(false));
    let empty_ok_inner = empty_ok.clone();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let index = Arc::clone(&index);
        let fast = Arc::clone(&fast);
        async move {
            let s = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
            let (r, m) = s.search_collect(&cx, "", 10).await.expect("empty");
            *empty_ok_inner.lock().unwrap() =
                r.is_empty() && m.phase1_total_ms.abs() < f64::EPSILON;
        }
    });
    check(
        &mut pass,
        &mut fail,
        "Empty query → empty results",
        *empty_ok.lock().unwrap(),
    );

    let zero_k_ok = Arc::new(Mutex::new(false));
    let zero_k_ok_inner = zero_k_ok.clone();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let index = Arc::clone(&index);
        let fast = Arc::clone(&fast);
        async move {
            let s = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
            let (r, _) = s.search_collect(&cx, "test", 0).await.expect("zero-k");
            *zero_k_ok_inner.lock().unwrap() = r.is_empty();
        }
    });
    check(
        &mut pass,
        &mut fail,
        "k=0 → empty results",
        *zero_k_ok.lock().unwrap(),
    );

    // ── Cleanup and summary ─────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&dir);

    println!();
    println!("\x1b[1;36m=== Summary ===\x1b[0m");
    println!(
        "  \x1b[32mPassed: {pass}\x1b[0m  \x1b[31mFailed: {fail}\x1b[0m  \x1b[33mWarnings: {warn}\x1b[0m"
    );
    println!(
        "  Total time: {:.1}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    if fail > 0 {
        std::process::exit(1);
    }
}

// ── Logging helpers ─────────────────────────────────────────────────────

fn log_info(step: &str, msg: &str) {
    println!("\x1b[36m[INFO] [{step}]\x1b[0m {msg}");
}

fn log_pass(step: &str, msg: &str) {
    println!("\x1b[32m[PASS] [{step}]\x1b[0m {msg}");
}

fn log_fail(step: &str, msg: &str) {
    println!("\x1b[31m[FAIL] [{step}]\x1b[0m {msg}");
}

fn check(pass: &mut u32, fail: &mut u32, name: &str, ok: bool) {
    if ok {
        log_pass("CHECK", name);
        *pass += 1;
    } else {
        log_fail("CHECK", name);
        *fail += 1;
    }
}

// ── Test data ───────────────────────────────────────────────────────────

const TEST_CORPUS: &[(&str, &str)] = &[
    (
        "doc-001",
        "Rust ownership and borrowing prevents data races at compile time",
    ),
    (
        "doc-002",
        "Machine learning models require large training datasets",
    ),
    (
        "doc-003",
        "Distributed consensus algorithms like Raft ensure fault tolerance",
    ),
    (
        "doc-004",
        "The HTTP/2 protocol supports multiplexed streams over a single connection",
    ),
    (
        "doc-005",
        "Database indexing with B-trees provides logarithmic lookup time",
    ),
    (
        "doc-006",
        "Functional programming emphasizes immutability and pure functions",
    ),
    (
        "doc-007",
        "Container orchestration with Kubernetes manages microservice deployments",
    ),
    (
        "doc-008",
        "Graph neural networks learn representations on structured data",
    ),
    (
        "doc-009",
        "WebAssembly enables near-native performance in web browsers",
    ),
    (
        "doc-010",
        "Zero-knowledge proofs allow verification without revealing data",
    ),
    (
        "doc-011",
        "The Rust borrow checker enforces memory safety without garbage collection",
    ),
    (
        "doc-012",
        "Gradient descent optimization finds local minima in loss landscapes",
    ),
    (
        "doc-013",
        "Byzantine fault tolerance handles malicious nodes in distributed systems",
    ),
    (
        "doc-014",
        "TLS 1.3 reduces handshake latency with zero round-trip resumption",
    ),
    (
        "doc-015",
        "LSM-tree storage engines optimize write-heavy workloads",
    ),
    (
        "doc-016",
        "Type-driven development uses the type system to enforce invariants",
    ),
    (
        "doc-017",
        "Service mesh sidecars provide observability and traffic management",
    ),
    (
        "doc-018",
        "Attention mechanisms in transformers capture long-range dependencies",
    ),
    (
        "doc-019",
        "SIMD instructions accelerate vector operations on modern CPUs",
    ),
    (
        "doc-020",
        "Homomorphic encryption enables computation on encrypted data",
    ),
];

/// Query suite: (query, expected doc IDs in top 10).
const QUERY_SUITE: &[(&str, &[&str])] = &[
    ("Rust ownership borrowing memory", &["doc-001", "doc-011"]),
    ("machine learning training data", &["doc-002", "doc-012"]),
    (
        "distributed consensus fault tolerance",
        &["doc-003", "doc-013"],
    ),
    ("HTTP protocol streams", &["doc-004"]),
    ("database index B-tree lookup", &["doc-005"]),
    ("functional programming immutability", &["doc-006"]),
    ("container kubernetes microservice", &["doc-007"]),
    ("neural network graph structured", &["doc-008"]),
    ("WebAssembly performance browser", &["doc-009"]),
    ("zero knowledge proofs verification", &["doc-010"]),
    ("encryption homomorphic computation", &["doc-020"]),
    ("SIMD vector CPU operations", &["doc-019"]),
    ("TLS handshake latency", &["doc-014"]),
    ("LSM tree write storage engine", &["doc-015"]),
    ("type system invariants development", &["doc-016"]),
    ("service mesh observability traffic", &["doc-017"]),
    ("attention transformer dependencies", &["doc-018"]),
    ("gradient descent optimization loss", &["doc-012"]),
    ("borrow checker garbage collection", &["doc-011"]),
    ("Byzantine malicious nodes distributed", &["doc-013"]),
];
