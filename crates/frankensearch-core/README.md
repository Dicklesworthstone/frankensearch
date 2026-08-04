# frankensearch-core

Core traits, types, and error types for the frankensearch hybrid search library.

## Overview

`frankensearch-core` is the foundational crate in the frankensearch workspace. Every other crate depends on it. It defines the shared interfaces, result types, error types, text canonicalization, query classification, telemetry structures, and configuration types used across the entire search pipeline.

This crate has minimal external dependencies and is designed to be a stable, lightweight foundation.

## Key Types

### Traits

- `Embedder` - async trait for text-to-vector embedding
- `LexicalRead` / `LexicalWrite` - split async traits for keyword/BM25 search backends
  (`LexicalRead` also carries generation-pinned candidate hydration)
- `Reranker` - async trait for cross-encoder reranking
- `SyncEmbed` / `SyncRerank` - synchronous adapter wrappers for embedders and rerankers
- `MetricsExporter` - trait for exporting telemetry data
- `HostAdapter` - trait for host environment integration and conformance

### Result and Hit Types

- `ScoredResult` - a search result with doc ID, score, rank, and metadata
- `VectorHit` - raw vector similarity hit (doc ID + cosine score)
- `FusedHit` - merged result after RRF fusion with provenance tracking
- `IndexableDocument` - input document with ID, content, title, and metadata

### Configuration and Metrics

- `TwoTierConfig` - configuration for the two-tier progressive search pipeline
- `TwoTierMetrics` - per-search timing and diagnostic metrics
- `SearchPhase` - enum for progressive result delivery (Initial / Refined / RefinementFailed)
- `QueryClass` - automatic query classification (short keyword, natural language, code, etc.)

### Error Handling

- `SearchError` - unified error type for all frankensearch operations
- `SearchResult<T>` - type alias for `Result<T, SearchError>`

### Infrastructure

- `Canonicalizer` / `DefaultCanonicalizer` - Unicode-aware text normalization
- `DocumentFingerprint` - content-hash fingerprinting for change detection
- `S3FifoCache` - S3-FIFO eviction cache for embeddings
- `RuntimeMetricsCollector` - telemetry collection for embedder, index, and search stages
- `CommitReplayEngine` - append-only commit log with replay for crash recovery
- `RepairOrchestrator` - corruption detection and repair coordination
- `DocumentGraph` - directed document relationship graph for graph-boosted ranking
- `GenerationManifest` - versioned manifest tracking index artifact generations
- `DecisionContext` / `DecisionOutcome` - decision-plane types for pipeline orchestration

## Usage

```rust
use frankensearch_core::{
    SearchError, SearchResult, TwoTierConfig, QueryClass,
    ScoredResult, IndexableDocument, Embedder,
};

// Classify a query
let class = QueryClass::classify("memory management in Rust");

// Create a document
let doc = IndexableDocument {
    id: "doc-1".to_string(),
    content: "Rust ownership and borrowing".to_string(),
    title: Some("Rust Memory Model".to_string()),
    metadata: None,
};

// Use default search configuration
let config = TwoTierConfig::default();
```

## Dependency Graph Position

```
frankensearch-core  (no internal dependencies -- leaf crate)
  ^
  |-- frankensearch-embed
  |-- frankensearch-index
  |-- frankensearch-lexical
  |-- frankensearch-fusion
  |-- frankensearch-rerank
  |-- frankensearch-storage
  |-- frankensearch-durability
  |-- frankensearch-tui
  |-- frankensearch-ops
  |-- frankensearch-fsfs
  |-- frankensearch (root)
```

## License

MIT
