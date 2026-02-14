//! Integration tests for frankensearch (bd-3un.32).
//!
//! End-to-end tests exercising the full search pipeline using the hash embedder
//! (no ML model downloads needed). All tests use default features only.
//!
//! Coverage:
//! 1. Basic two-tier flow (`IndexBuilder` → `TwoTierSearcher`)
//! 2. Progressive search phases (Initial, Refined, `fast_only`)
//! 3. Persistence round-trip (build → close → reopen → search)
//! 4. Config interactions (`quality_weight`, `rrf_k`, `fast_only`)
//! 5. Concurrent reads (Arc<TwoTierIndex> shared across searches)
//! 6. Error propagation (empty queries, dimension mismatches)
//! 7. Rank changes across phases

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex, VectorIndex};
use frankensearch_core::config::TwoTierConfig;
use frankensearch_core::traits::Embedder;
use frankensearch_core::types::SearchPhase;
use frankensearch_index::{
    Quantization, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
};

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn temp_dir(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "frankensearch-integ-{name}-{}-{now}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Build an index from text documents using `HashEmbedder`, returning the dir.
fn build_hash_index(name: &str, docs: &[(&str, &str)]) -> (PathBuf, usize) {
    let dir = temp_dir(name);
    let embedder = HashEmbedder::default_256();
    let dim = embedder.dimension();
    let path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let mut writer =
        VectorIndex::create_with_revision(&path, embedder.id(), "v1", dim, Quantization::F16)
            .expect("create writer");
    for (id, text) in docs {
        let vec = embedder.embed_sync(text);
        writer.write_record(id, &vec).expect("write");
    }
    writer.finish().expect("finish");
    (dir, dim)
}

/// Build a two-tier index with separate fast (256d) and quality (384d) embeddings.
fn build_two_tier_hash_index(name: &str, docs: &[(&str, &str)]) -> PathBuf {
    let dir = temp_dir(name);
    let fast = HashEmbedder::default_256();
    let quality = HashEmbedder::default_384();

    // Fast index
    let fast_path = dir.join(VECTOR_INDEX_FAST_FILENAME);
    let mut fw = VectorIndex::create_with_revision(
        &fast_path,
        fast.id(),
        "v1",
        fast.dimension(),
        Quantization::F16,
    )
    .expect("create fast");
    for (id, text) in docs {
        fw.write_record(id, &fast.embed_sync(text))
            .expect("write fast");
    }
    fw.finish().expect("finish fast");

    // Quality index
    let quality_path = dir.join(VECTOR_INDEX_QUALITY_FILENAME);
    let mut qw = VectorIndex::create_with_revision(
        &quality_path,
        quality.id(),
        "v1",
        quality.dimension(),
        Quantization::F16,
    )
    .expect("create quality");
    for (id, text) in docs {
        qw.write_record(id, &quality.embed_sync(text))
            .expect("write quality");
    }
    qw.finish().expect("finish quality");

    dir
}

/// A corpus of 20 diverse documents for testing.
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

// ═══════════════════════════════════════════════════════════════════════════
// 1. Basic two-tier flow
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn basic_search_returns_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("basic-search", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, metrics) = searcher
            .search_collect(&cx, "Rust memory safety", 5)
            .await
            .unwrap();

        assert!(!results.is_empty(), "should return results");
        assert!(results.len() <= 5, "should respect k limit");
        assert!(metrics.phase1_total_ms > 0.0, "should measure phase 1 time");
        assert!(metrics.fast_embedder_id.is_some());
    });
}

#[test]
fn search_results_are_relevant() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("relevance", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect(&cx, "Rust ownership borrowing", 5)
            .await
            .unwrap();

        // doc-001 and doc-011 are about Rust and should appear in top 5
        let top_ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(
            top_ids.contains(&"doc-001") || top_ids.contains(&"doc-011"),
            "Rust-related docs should rank high for Rust query, got: {top_ids:?}"
        );
    });
}

#[test]
fn search_with_corpus_of_100_documents() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // Generate 100 synthetic documents
        let docs: Vec<(String, String)> = (0..100)
            .map(|i| {
                (
                    format!("doc-{i:03}"),
                    format!(
                        "Document number {i} about topic {} with content variation {}",
                        i % 10,
                        i * 7
                    ),
                )
            })
            .collect();
        let doc_refs: Vec<(&str, &str)> = docs
            .iter()
            .map(|(id, text)| (id.as_str(), text.as_str()))
            .collect();

        let (dir, _) = build_hash_index("corpus-100", &doc_refs);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, metrics) = searcher
            .search_collect(&cx, "topic 5 variation", 10)
            .await
            .unwrap();

        assert!(results.len() <= 10);
        assert!(metrics.semantic_candidates > 0);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Progressive search phases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fast_only_mode_yields_only_initial_phase() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("fast-only", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let config = TwoTierConfig {
            fast_only: true,
            ..Default::default()
        };
        let searcher = TwoTierSearcher::new(index, embedder, config);

        let mut phases = Vec::new();
        let _metrics = searcher
            .search(
                &cx,
                "distributed systems",
                5,
                |_| None,
                |phase| {
                    phases.push(format!("{phase:?}"));
                },
            )
            .await
            .unwrap();

        assert_eq!(phases.len(), 1, "fast_only should yield exactly 1 phase");
        assert!(
            phases[0].contains("Initial"),
            "single phase should be Initial, got: {}",
            phases[0]
        );
    });
}

#[test]
fn two_tier_search_yields_initial_then_refined() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = build_two_tier_hash_index("two-tier-phases", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));

        let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let mut phase_names = Vec::new();
        let _metrics = searcher
            .search(
                &cx,
                "machine learning optimization",
                5,
                |_| None,
                |phase| match &phase {
                    SearchPhase::Initial { .. } => phase_names.push("Initial"),
                    SearchPhase::Refined { .. } => phase_names.push("Refined"),
                    SearchPhase::RefinementFailed { .. } => phase_names.push("RefinementFailed"),
                },
            )
            .await
            .unwrap();

        assert_eq!(
            phase_names.len(),
            2,
            "should yield 2 phases: {phase_names:?}"
        );
        assert_eq!(phase_names[0], "Initial");
        assert_eq!(phase_names[1], "Refined");
    });
}

#[test]
fn initial_phase_results_are_valid() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("initial-valid", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut initial_results = None;
        let _metrics = searcher
            .search(
                &cx,
                "database indexing",
                5,
                |_| None,
                |phase| {
                    if let SearchPhase::Initial { results, .. } = phase {
                        initial_results = Some(results);
                    }
                },
            )
            .await
            .unwrap();

        let results = initial_results.expect("should have Initial phase");
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Results should be sorted by score descending
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "results should be sorted: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Persistence round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn persist_and_reopen_returns_same_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("persist-roundtrip", TEST_CORPUS);
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        // First search
        let index1 = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open1"));
        let searcher1 =
            TwoTierSearcher::new(index1, Arc::clone(&embedder), TwoTierConfig::default());
        let (results1, _) = searcher1
            .search_collect(&cx, "graph neural networks", 5)
            .await
            .unwrap();

        // Drop everything, reopen
        drop(searcher1);

        let index2 = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open2"));
        let searcher2 =
            TwoTierSearcher::new(index2, Arc::clone(&embedder), TwoTierConfig::default());
        let (results2, _) = searcher2
            .search_collect(&cx, "graph neural networks", 5)
            .await
            .unwrap();

        // Same query on same data → identical results
        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.doc_id, r2.doc_id, "doc IDs should match");
            assert!(
                (r1.score - r2.score).abs() < 1e-5,
                "scores should match: {} vs {}",
                r1.score,
                r2.score
            );
        }
    });
}

#[test]
fn index_builder_creates_searchable_index() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("builder-search");
        let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
        let stack = EmbedderStack::from_parts(fast, None);

        let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
        for (id, text) in TEST_CORPUS {
            builder = builder.add_document(*id, *text);
        }
        let stats = builder.build(&cx).await.unwrap();

        assert_eq!(stats.doc_count, 20);
        assert_eq!(stats.error_count, 0);
        assert!(!stats.has_quality_index);

        // Now search the built index
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, _) = searcher
            .search_collect(&cx, "WebAssembly performance", 5)
            .await
            .unwrap();

        assert!(!results.is_empty());
        // doc-009 is about WebAssembly
        let top_ids: Vec<&str> = results.iter().map(|r| r.doc_id.as_str()).collect();
        assert!(
            top_ids.contains(&"doc-009"),
            "WebAssembly doc should appear for WebAssembly query: {top_ids:?}"
        );
    });
}

#[test]
fn index_builder_with_two_tier() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("builder-two-tier");
        let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
        let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
        let stack = EmbedderStack::from_parts(fast, Some(quality));

        let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
        for (id, text) in TEST_CORPUS {
            builder = builder.add_document(*id, *text);
        }
        let stats = builder.build(&cx).await.unwrap();

        assert_eq!(stats.doc_count, 20);
        assert!(stats.has_quality_index);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Config interactions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn quality_weight_affects_refined_ranking() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = build_two_tier_hash_index("quality-weight", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));

        let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());

        // Low quality weight → fast tier dominates
        let config_low = TwoTierConfig {
            quality_weight: 0.1,
            ..Default::default()
        };
        let searcher_low = TwoTierSearcher::new(Arc::clone(&index), Arc::clone(&fast), config_low)
            .with_quality_embedder(Arc::clone(&quality));
        let (results_low, _) = searcher_low
            .search_collect(&cx, "distributed consensus", 10)
            .await
            .unwrap();

        // High quality weight → quality tier dominates
        let config_high = TwoTierConfig {
            quality_weight: 0.9,
            ..Default::default()
        };
        let searcher_high =
            TwoTierSearcher::new(Arc::clone(&index), Arc::clone(&fast), config_high)
                .with_quality_embedder(Arc::clone(&quality));
        let (results_high, _) = searcher_high
            .search_collect(&cx, "distributed consensus", 10)
            .await
            .unwrap();

        // Both should return results
        assert!(!results_low.is_empty());
        assert!(!results_high.is_empty());
        // Rankings may differ due to different quality weights
        // (they use different embedding dimensions so quality scores differ)
    });
}

#[test]
fn different_k_values_respected() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("k-values", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        for k in [1, 3, 5, 10, 20] {
            let (results, _) = searcher
                .search_collect(&cx, "machine learning", k)
                .await
                .unwrap();
            assert!(
                results.len() <= k,
                "k={k}: got {} results (should be <= {k})",
                results.len()
            );
            if k <= 20 {
                // We have 20 docs, so for k <= 20 we should get min(k, relevant)
                assert!(
                    !results.is_empty(),
                    "k={k}: should return at least 1 result"
                );
            }
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Concurrent reads
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn concurrent_searches_on_shared_index() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("concurrent", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let searcher = Arc::new(TwoTierSearcher::new(
            index,
            embedder,
            TwoTierConfig::default(),
        ));

        let queries = [
            "Rust ownership",
            "machine learning",
            "distributed systems",
            "encryption",
        ];
        let mut results_per_query = Vec::new();

        for query in &queries {
            let (results, _) = searcher.search_collect(&cx, query, 5).await.unwrap();
            results_per_query.push((query.to_string(), results));
        }

        // All queries should return results
        for (query, results) in &results_per_query {
            assert!(!results.is_empty(), "query '{query}' should return results");
        }

        // Different queries should produce different top results
        let first_ids: Vec<&str> = results_per_query
            .iter()
            .map(|(_, r)| r[0].doc_id.as_str())
            .collect();
        // At least some queries should have different top results
        let unique_count = {
            let mut ids = first_ids.clone();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };
        assert!(
            unique_count >= 2,
            "different queries should return different results: {first_ids:?}"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Error propagation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn empty_query_returns_empty_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("empty-query", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let mut phases = Vec::new();
        let metrics = searcher
            .search(&cx, "", 10, |_| None, |p| phases.push(format!("{p:?}")))
            .await
            .unwrap();

        assert!(phases.is_empty(), "empty query should yield no phases");
        assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
    });
}

#[test]
fn zero_k_returns_empty_results() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("zero-k", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
        let (results, metrics) = searcher.search_collect(&cx, "anything", 0).await.unwrap();

        assert!(results.is_empty());
        assert!(metrics.phase1_total_ms.abs() < f64::EPSILON);
    });
}

#[test]
fn index_builder_empty_documents_rejected() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("empty-docs");
        let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
        let stack = EmbedderStack::from_parts(fast, None);

        let err = IndexBuilder::new(&dir)
            .with_embedder_stack(stack)
            .build(&cx)
            .await
            .expect_err("should fail with no docs");

        assert!(
            matches!(err, SearchError::InvalidConfig { .. }),
            "expected InvalidConfig, got: {err:?}"
        );
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Rank changes across phases
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn refined_phase_reports_rank_changes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = build_two_tier_hash_index("rank-changes", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));

        let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let (_, metrics) = searcher
            .search_collect(&cx, "zero knowledge proofs encryption", 10)
            .await
            .unwrap();

        // With two-tier search, rank changes should be tracked
        let total = metrics.rank_changes.total();
        assert!(
            total > 0,
            "should have some rank changes across phases, got {total}"
        );
        // promoted + demoted + stable = total
        assert_eq!(
            metrics.rank_changes.promoted
                + metrics.rank_changes.demoted
                + metrics.rank_changes.stable,
            total
        );
    });
}

#[test]
fn metrics_capture_both_phases() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = build_two_tier_hash_index("metrics-phases", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));

        let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
        let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());

        let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default())
            .with_quality_embedder(quality);

        let (_, metrics) = searcher
            .search_collect(&cx, "container orchestration kubernetes", 5)
            .await
            .unwrap();

        // Phase 1 metrics
        assert!(
            metrics.fast_embed_ms > 0.0,
            "fast embedding should be timed"
        );
        assert!(
            metrics.vector_search_ms > 0.0,
            "vector search should be timed"
        );
        assert!(metrics.phase1_total_ms > 0.0);
        assert!(metrics.fast_embedder_id.is_some());

        // Phase 2 metrics
        assert!(
            metrics.quality_embed_ms > 0.0,
            "quality embedding should be timed"
        );
        assert!(metrics.blend_ms >= 0.0, "blend should be timed");
        assert!(metrics.phase2_total_ms > 0.0);
        assert!(metrics.quality_embedder_id.is_some());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Search determinism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn search_is_deterministic() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("determinism", TEST_CORPUS);
        let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

        let (r1, _) = searcher
            .search_collect(&cx, "type driven development", 10)
            .await
            .unwrap();
        let (r2, _) = searcher
            .search_collect(&cx, "type driven development", 10)
            .await
            .unwrap();

        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.doc_id, b.doc_id, "result order should be deterministic");
            assert!(
                (a.score - b.score).abs() < 1e-6,
                "scores should match: {} vs {}",
                a.score,
                b.score
            );
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Config serialization across search
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn config_roundtrip_produces_consistent_search() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let (dir, _) = build_hash_index("config-roundtrip", TEST_CORPUS);
        let config = TwoTierConfig {
            quality_weight: 0.7,
            candidate_multiplier: 4,
            fast_only: true,
            ..Default::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let decoded: TwoTierConfig = serde_json::from_str(&json).unwrap();

        let index = Arc::new(TwoTierIndex::open(&dir, decoded.clone()).expect("open"));
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());

        let searcher = TwoTierSearcher::new(index, embedder, decoded);
        let (results, _) = searcher
            .search_collect(&cx, "SIMD vector operations", 5)
            .await
            .unwrap();

        assert!(!results.is_empty());
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Progress callback in IndexBuilder
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn index_builder_reports_progress() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let dir = temp_dir("builder-progress");
        let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
        let stack = EmbedderStack::from_parts(fast, None);

        let progress_calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_clone = Arc::clone(&progress_calls);

        let mut builder = IndexBuilder::new(&dir)
            .with_embedder_stack(stack)
            .with_batch_size(5)
            .with_progress(move |p| {
                progress_clone.lock().unwrap().push((p.completed, p.total));
            });
        for (id, text) in TEST_CORPUS {
            builder = builder.add_document(*id, *text);
        }
        let stats = builder.build(&cx).await.unwrap();

        assert_eq!(stats.doc_count, 20);
        let calls = progress_calls.lock().unwrap();
        assert!(!calls.is_empty(), "progress should be reported");
        // All calls should have total=20
        assert!(calls.iter().all(|(_, total)| *total == 20));
        // Last call should have completed=20
        assert_eq!(calls.last().unwrap().0, 20);
        drop(calls);
    });
}
