//! Custom embedder example: implement the [`Embedder`] trait.
//!
//! Shows how to plug a custom embedding model into the frankensearch pipeline.
//! This example uses a toy "bag of characters" embedder for illustration.
//!
//! Run with: `cargo run --example custom_embedder`

use std::sync::Arc;

use asupersync::Cx;
use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::{ModelCategory, SearchFuture};

// ── Custom Embedder ──────────────────────────────────────────────────────────

/// A toy "bag of characters" embedder that counts ASCII letter frequencies.
///
/// Each dimension corresponds to a letter (a=0, b=1, ..., z=25) plus 6
/// extra dimensions for digits and punctuation, yielding 32 dimensions.
///
/// This is NOT semantically useful — it's purely to show how to implement
/// the [`Embedder`] trait.
struct CharFreqEmbedder;

impl CharFreqEmbedder {
    const DIM: usize = 32;
}

impl Embedder for CharFreqEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let mut counts = vec![0.0f32; Self::DIM];
            for byte in text.bytes() {
                let idx = match byte {
                    b'a'..=b'z' => (byte - b'a') as usize,
                    b'A'..=b'Z' => (byte - b'A') as usize,
                    b'0'..=b'9' => 26,
                    b' ' => 27,
                    b'.' | b',' | b'!' | b'?' => 28,
                    b'-' | b'_' => 29,
                    b'(' | b')' | b'[' | b']' => 30,
                    _ => 31,
                };
                counts[idx] += 1.0;
            }
            // L2-normalize for cosine similarity.
            let norm = counts.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut counts {
                    *x /= norm;
                }
            }
            Ok(counts)
        })
    }

    fn dimension(&self) -> usize {
        Self::DIM
    }

    fn id(&self) -> &'static str {
        "char-freq-32"
    }

    fn model_name(&self) -> &'static str {
        "Character Frequency Embedder"
    }

    fn is_semantic(&self) -> bool {
        false // Not truly semantic
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }

    // Note: model_info() has a default implementation that builds ModelInfo
    // from the other trait methods. Override it only if you need custom fields.
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("Custom embedder example: CharFreqEmbedder (32d)\n");

    let documents = vec![
        (
            "rust-safety",
            "Rust ownership and borrowing prevents data races",
        ),
        (
            "python-gc",
            "Python uses reference counting and garbage collection",
        ),
        (
            "go-goroutines",
            "Go uses goroutines and channels for concurrency",
        ),
        (
            "java-jvm",
            "Java runs on the JVM with automatic memory management",
        ),
    ];

    let dir = std::env::temp_dir().join(format!("frankensearch-custom-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");

    // Build index with custom embedder.
    asupersync::test_utils::run_test_with_cx(|cx| {
        let dir = dir.clone();
        let documents = documents.clone();
        async move {
            let embedder = Arc::new(CharFreqEmbedder) as Arc<dyn Embedder>;
            let stack = EmbedderStack::from_parts(Arc::clone(&embedder), None);

            let mut builder = IndexBuilder::new(&dir).with_embedder_stack(stack);
            for (id, text) in &documents {
                builder = builder.add_document(*id, *text);
            }
            let stats = builder.build(&cx).await.expect("build");
            println!(
                "Built index: {} docs, {:.1}ms\n",
                stats.doc_count, stats.total_ms
            );
        }
    });

    // Search with the custom embedder.
    let embedder = Arc::new(CharFreqEmbedder) as Arc<dyn Embedder>;
    let index = Arc::new(TwoTierIndex::open(&dir, TwoTierConfig::default()).expect("open"));
    let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());

    for query in [
        "Rust memory safety",
        "Python garbage collector",
        "Go channels",
    ] {
        println!("Query: \"{query}\"");
        asupersync::test_utils::run_test_with_cx(|cx| {
            let searcher = &searcher;
            async move {
                let (results, _) = searcher
                    .search_collect(&cx, query, 4)
                    .await
                    .expect("search");
                for (i, r) in results.iter().enumerate() {
                    println!("  {}. {} (score: {:.4})", i + 1, r.doc_id, r.score);
                }
            }
        });
        println!();
    }

    let _ = std::fs::remove_dir_all(&dir);
    println!("Done.");
}
