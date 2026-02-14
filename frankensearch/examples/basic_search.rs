//! Basic search example: build an index and search it with hash embedders.
//!
//! This example requires only the default `hash` feature (no ML model downloads).
//!
//! Run with: `cargo run --example basic_search`

use std::sync::Arc;

use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::Embedder;

#[allow(clippy::too_many_lines)]
fn main() {
    // Documents to index.
    let documents = vec![
        (
            "rust-ownership",
            "Rust ownership and borrowing prevents data races at compile time",
        ),
        (
            "ml-training",
            "Machine learning models require large training datasets",
        ),
        (
            "distributed",
            "Distributed consensus algorithms like Raft ensure fault tolerance",
        ),
        (
            "http2",
            "The HTTP/2 protocol supports multiplexed streams over a single connection",
        ),
        (
            "databases",
            "Database indexing with B-trees provides logarithmic lookup time",
        ),
    ];

    // Create a temporary directory for the index.
    let dir = std::env::temp_dir().join(format!("frankensearch-basic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    // ── Step 1: Build the index ───────────────────────────────────────────
    println!("Building index with {} documents...", documents.len());

    asupersync::test_utils::run_test_with_cx(|cx| {
        let dir = dir.clone();
        let documents = documents.clone();
        async move {
            // Hash embedders are fast (~11μs) and require no model downloads.
            let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
            let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
            let stack = EmbedderStack::from_parts(fast, Some(quality));

            let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
            for (id, text) in &documents {
                builder = builder.add_document(*id, *text);
            }
            let stats = builder.build(&cx).await.expect("build index");
            println!(
                "Index built: {} docs, quality_tier={}, {:.1}ms",
                stats.doc_count, stats.has_quality_index, stats.total_ms
            );
        }
    });

    // ── Step 2: Open and search ───────────────────────────────────────────
    let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
    let quality: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_384());
    let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open index"));

    let searcher = TwoTierSearcher::new(Arc::clone(&index), fast, TwoTierConfig::default())
        .with_quality_embedder(quality);

    let queries = [
        "Rust memory safety",
        "machine learning data",
        "distributed fault tolerance",
        "HTTP protocol",
        "B-tree index lookup",
    ];

    for query in &queries {
        println!("\nQuery: \"{query}\"");

        asupersync::test_utils::run_test_with_cx(|cx| {
            let searcher = &searcher;
            async move {
                // search_collect returns final results after both phases.
                let (results, metrics) = searcher
                    .search_collect(&cx, query, 3)
                    .await
                    .expect("search");

                for (i, result) in results.iter().enumerate() {
                    println!(
                        "  {}. {} (score: {:.4})",
                        i + 1,
                        result.doc_id,
                        result.score
                    );
                }
                println!(
                    "  phase1={:.1}ms phase2={:.1}ms",
                    metrics.phase1_total_ms, metrics.phase2_total_ms
                );
            }
        });
    }

    // ── Step 3: Progressive search (phase callbacks) ──────────────────────
    println!("\n--- Progressive search demo ---");
    println!("Query: \"database indexing\"");

    asupersync::test_utils::run_test_with_cx(|cx| {
        let searcher = &searcher;
        async move {
            searcher
                .search(
                    &cx,
                    "database indexing",
                    3,
                    |_| None, // no text lookup function
                    |phase| match &phase {
                        SearchPhase::Initial { results, .. } => {
                            println!("  [Phase 1 - Initial] {} results", results.len());
                            for r in results.iter().take(3) {
                                println!("    {} (score: {:.4})", r.doc_id, r.score);
                            }
                        }
                        SearchPhase::Refined { results, rank_changes, .. } => {
                            println!(
                                "  [Phase 2 - Refined] {} results (promoted={}, demoted={}, stable={})",
                                results.len(),
                                rank_changes.promoted,
                                rank_changes.demoted,
                                rank_changes.stable,
                            );
                            for r in results.iter().take(3) {
                                println!("    {} (score: {:.4})", r.doc_id, r.score);
                            }
                        }
                        SearchPhase::RefinementFailed { error, .. } => {
                            println!("  [Phase 2 - Failed] {error}");
                        }
                    },
                )
                .await
                .expect("search");
        }
    });

    // ── Cleanup ───────────────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&dir);
    println!("\nDone.");
}
