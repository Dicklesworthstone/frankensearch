# frankensearch

Two-tier hybrid search for Rust: sub-millisecond initial results, quality-refined rankings in ~150ms.

## Overview

`frankensearch` is the main library crate that re-exports and unifies all workspace sub-crates into a single, ergonomic API. It combines lexical (native Quill or the explicit Tantivy oracle) and semantic (vector cosine similarity) search via Reciprocal Rank Fusion (RRF), with a two-tier progressive embedding model that delivers results in two phases:

1. **Phase 1 (Initial):** Fast embedder (potion-128M, 256d, ~0.57ms) produces results immediately via brute-force vector search + optional BM25 fusion.
2. **Phase 2 (Refined):** Quality embedder (MiniLM-L6-v2, 384d, ~128ms) re-scores the top candidates for higher relevance.

Consumers receive results progressively via `SearchPhase` callbacks, so UIs can display fast results while quality refinement runs in the background.

```text
 Query --+-> Fast Embed (256d) -> Vector Search --+-> RRF Fusion -> Phase 1
         |                                        |
         +-> Quill / Tantivy-oracle BM25 ---------+
                                                        |
                                              Quality Embed (384d)
                                                        |
                                                   Score Blend
                                                        |
                                                  Phase 2 Results
```

## Key Types

- `IndexBuilder` / `IndexBuildStats` - build a search index from documents
- `TwoTierSearcher` - progressive two-phase search orchestrator
- `TwoTierConfig` / `TwoTierMetrics` - search configuration and per-search diagnostics
- `SearchPhase` - progressive result delivery (Initial / Refined / RefinementFailed)
- `EmbedderStack` - fast + optional quality embedder pair with auto-detection
- `TwoTierIndex` / `VectorIndex` - two-tier and low-level vector index types
- `ScoredResult` / `FusedHit` / `VectorHit` - result types with provenance tracking
- `FederatedSearcher` / `FederatedFusion` - multi-index federated search
- `Embedder` / `LexicalSearch` / `Reranker` - core traits for pluggable backends
- `QueryClass` - automatic query classification
- `Canonicalizer` / `DocumentFingerprint` - text normalization and change detection

## Feature Flags

| Feature | Description |
|---------|-------------|
| `hash` (default) | FNV-1a hash embedder, zero dependencies |
| `model2vec` | potion-128M static embedder (fast tier) |
| `fastembed` | MiniLM-L6-v2 ONNX embedder (quality tier) |
| `lexical` | Pre-flip Tantivy BM25 compatibility lane |
| `quill` | Native Quill lexical engine and `IndexBuilder` integration |
| `lexical-tantivy` | Explicit Tantivy facade/oracle and comparator lane |
| `cass-compat` | External CASS schema-v8 Tantivy-format interoperability; not a Quill fallback |
| `rerank` | FlashRank cross-encoder reranking |
| `ann` | HNSW approximate nearest-neighbor index |
| `download` | Model auto-download from HuggingFace |
| `storage` | FrankenSQLite document metadata + embedding queue |
| `durability` | RaptorQ self-healing for persistent index artifacts |
| `fts5` | FrankenSQLite FTS5 lexical backend |
| `graph` | Graph-boosted ranking using document relationships |
| `semantic` | `hash` + `model2vec` + `fastembed` |
| `hybrid` | `semantic` + `lexical` |
| `persistent` | `hybrid` + `storage` |
| `durable` | `persistent` + `durability` |
| `full` | `durable` + `rerank` + `ann` + `download` + `graph` |
| `full-fts5` | `full` + `fts5` |

### Recommended Combinations

- **Development/testing:** `default` (hash only, no downloads)
- **Production semantic:** `semantic` + `download`
- **Persistent hybrid search:** `persistent`
- **Maximum durability:** `durable` or `full`

## Usage

```rust,no_run
use std::sync::Arc;
use frankensearch::prelude::*;
use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
use frankensearch_core::traits::Embedder;

asupersync::test_utils::run_test_with_cx(|cx| async move {
    // Build an index
    let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
    let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
    let stack = EmbedderStack::from_parts(fast, Some(quality));

    let stats = IndexBuilder::new("./my_index")
        .with_embedder_stack(stack)
        .add_document("doc-1", "Rust ownership and borrowing")
        .add_document("doc-2", "Python garbage collection")
        .build(&cx)
        .await
        .expect("build index");

    // Search
    let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
    let index = Arc::new(
        TwoTierIndex::open(std::path::Path::new("./my_index"), TwoTierConfig::default()).unwrap()
    );
    let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
    let (results, metrics) = searcher
        .search_collect(&cx, "memory management", 10)
        .await
        .expect("search");

    for result in &results {
        println!("{}: {:.4}", result.doc_id, result.score);
    }

    #[cfg(feature = "quill")]
    {
        let lexical = frankensearch::QuillIndex::open(
            &cx,
            "./my_index/lexical",
            frankensearch::QuillConfig::default(),
        )
        .await
        .expect("open Quill lexical index");
        let lexical_hits = lexical
            .search_results(&cx, "ownership", 10)
            .expect("search Quill lexical index");
        assert!(!lexical_hits.is_empty());
    }
});
```

## Async Runtime

frankensearch uses [asupersync](https://docs.rs/asupersync) exclusively -- not tokio. All async methods take `&Cx` (capability context) as their first parameter. The `Cx` is provided by the consumer's asupersync runtime; frankensearch never creates its own runtime.

## Performance

Measured on a single core (no GPU), 10K document corpus:

| Operation | Embedder | Latency |
|-----------|----------|---------|
| Hash embed (256d) | FNV-1a | ~11 us |
| Fast embed (256d) | potion-128M | ~0.57 ms |
| Quality embed (384d) | MiniLM-L6-v2 | ~128 ms |
| Vector search (10K, top-10) | brute-force | ~2 ms |
| RRF fusion (500+500) | - | ~1 ms |
| Full pipeline (hash, 10K) | hash only | ~3 ms |

## Dependency Graph Position

This is the top-level library crate that unifies the workspace:

```
frankensearch-core       (always)
frankensearch-embed      (always)
frankensearch-index      (always)
frankensearch-fusion     (always)
frankensearch-lexical    (feature: lexical)
frankensearch-quill      (feature: quill)
frankensearch-rerank     (feature: rerank)
frankensearch-storage    (feature: storage)
frankensearch-durability (feature: durability)
```

## License

MIT
