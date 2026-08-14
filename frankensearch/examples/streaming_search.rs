//! Streaming/progressive search example.
//!
//! The default `hash` feature runs this as an explicit hash-control fixture,
//! not semantic two-tier search. Production code should detect a verified
//! semantic embedder with `EmbedderStack::auto_detect_semantic_with`.
//!
//! Run with:
//! - `cargo run --example streaming_search`

use std::sync::Arc;
use std::time::Instant;

use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::Embedder;

#[allow(clippy::too_many_lines)]
fn main() {
    let documents = vec![
        (
            "retry-strategy",
            "Exponential backoff with jitter avoids thundering herd retries",
        ),
        (
            "raft-overview",
            "Raft replication uses leader election and majority quorum",
        ),
        (
            "http-timeouts",
            "HTTP clients should set connect and read timeout budgets",
        ),
        (
            "memory-safety",
            "Rust ownership prevents use-after-free and data races",
        ),
        (
            "queue-workers",
            "Worker queues need retry policy, dead-lettering, and visibility timeouts",
        ),
    ];

    let dir = std::env::temp_dir().join(format!("frankensearch-streaming-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    asupersync::test_utils::run_test_with_cx(|cx| {
        let dir = dir.clone();
        let documents = documents.clone();
        async move {
            let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
            let stack = EmbedderStack::from_parts(fast, None);

            let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
            for (doc_id, content) in &documents {
                builder = builder.add_document(*doc_id, *content);
            }
            let stats = builder.build(&cx).await.expect("build index");
            println!(
                "index_ready docs={} quality={} total_ms={:.2} (hash control, not semantic)",
                stats.doc_count, stats.has_quality_index, stats.total_ms
            );
        }
    });

    let fast: Arc<dyn Embedder> = Arc::new(HashEmbedder::default_256());
    let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open index"));
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());

    let query = "resilient retry policy for distributed workers";
    println!("query=\"{query}\"");

    asupersync::test_utils::run_test_with_cx(|cx| {
        let searcher = &searcher;
        async move {
            let wall_start = Instant::now();
            let metrics = searcher
                .search(&cx, query, 4, |_| None, |phase| match phase {
                    SearchPhase::Initial {
                        results,
                        latency,
                        metrics,
                    } => {
                        println!(
                            "event=phase name=initial hits={} latency_ms={:.2} vectors_searched={} lexical_candidates={} hash_control_candidates={}",
                            results.len(),
                            latency.as_secs_f64() * 1000.0,
                            metrics.vectors_searched,
                            metrics.lexical_candidates,
                            metrics.hash_control_candidates
                        );
                        for (rank, hit) in results.iter().take(3).enumerate() {
                            println!(
                                "event=hit phase=initial rank={} doc_id={} score={:.4}",
                                rank + 1,
                                hit.doc_id,
                                hit.score
                            );
                        }
                    }
                    SearchPhase::Refined {
                        results,
                        latency,
                        rank_changes,
                        ..
                    } => {
                        println!(
                            "event=phase name=refined hits={} latency_ms={:.2} promoted={} demoted={} stable={}",
                            results.len(),
                            latency.as_secs_f64() * 1000.0,
                            rank_changes.promoted,
                            rank_changes.demoted,
                            rank_changes.stable
                        );
                        for (rank, hit) in results.iter().take(3).enumerate() {
                            println!(
                                "event=hit phase=refined rank={} doc_id={} score={:.4}",
                                rank + 1,
                                hit.doc_id,
                                hit.score
                            );
                        }
                    }
                    SearchPhase::Reranked {
                        results,
                        latency,
                        ..
                    } => {
                        println!(
                            "event=phase name=reranked hits={} latency_ms={:.2}",
                            results.len(),
                            latency.as_secs_f64() * 1000.0
                        );
                        for (rank, hit) in results.iter().take(3).enumerate() {
                            println!(
                                "event=hit phase=reranked rank={} doc_id={} score={:.4}",
                                rank + 1,
                                hit.doc_id,
                                hit.score
                            );
                        }
                    }
                    SearchPhase::RefinementFailed {
                        initial_results,
                        error,
                        latency,
                    } => {
                        println!(
                            "event=phase name=refinement_failed hits={} latency_ms={:.2} error={}",
                            initial_results.len(),
                            latency.as_secs_f64() * 1000.0,
                            error
                        );
                    }
                })
                .await
                .expect("search");

            println!(
                "event=done phase1_ms={:.2} phase2_ms={:.2} total_wall_ms={:.2} skip_reason={:?}",
                metrics.phase1_total_ms,
                metrics.phase2_total_ms,
                wall_start.elapsed().as_secs_f64() * 1000.0,
                metrics.skip_reason
            );
        }
    });

    let _ = std::fs::remove_dir_all(&dir);
}
