# frankensearch-fusion

RRF fusion, score blending, and two-tier progressive search orchestration for frankensearch.

## Overview

This crate is the search orchestration layer of frankensearch. It combines results from vector search and lexical search using Reciprocal Rank Fusion (RRF), blends fast-tier and quality-tier scores, and manages the progressive two-phase search pipeline. It also provides adaptive fusion, score calibration, circuit breaking, feedback collection, and federated multi-index search.

## Key Types

### Fusion and Blending

- `rrf_fuse` - Reciprocal Rank Fusion (K=60) with 4-level tie-breaking
- `blend_two_tier` - score blending (default 0.7 quality / 0.3 fast)
- `RrfConfig` - RRF parameter configuration
- `MmrConfig` / `mmr_rerank` - Maximal Marginal Relevance for diversity-aware reranking

### Search Orchestration

- `TwoTierSearcher` - asynchronous progressive search delivering `SearchPhase` callbacks
- `SyncTwoTierSearcher` / `SyncSearchIterator` - demand-driven progressive retrieval with precomputed, independently bound query embeddings
- `IncrementalSearcher` - incremental search with configurable strategy and budget
- `FederatedSearcher` / `FederatedFusion` - search across multiple indices with result merging

### Score Calibration

- `ScoreCalibrator` - trait for score normalization
- `PlattScaling` - Platt sigmoid calibration
- `IsotonicRegression` - monotone score calibration
- `TemperatureScaling` - temperature-based softmax calibration

### Adaptive and Quality Control

- `AdaptiveFusion` - Bayesian-adaptive fusion weight tuning per query class
- `CircuitBreaker` - quality-tier health monitoring with automatic skip on failure
- `PhaseGate` - anytime-valid sequential testing for phase transition decisions
- `FeedbackCollector` - implicit relevance feedback with exponentially-decaying boost map

### Caching and Refresh

- `IndexCache` / `SentinelFileDetector` - index staleness detection and cache invalidation
- `RefreshWorker` - background index refresh with configurable polling
- `EmbeddingQueue` - async embedding job queue with priority and backpressure

### Normalization

- `normalize_scores` - min-max, z-score, or identity score normalization
- `NormalizationMethod` - configurable normalization strategy

### Pseudo-Relevance Feedback

- `prf_expand` - pseudo-relevance feedback query expansion

### Graph Ranking (feature: `graph`)

- `GraphRanker` - graph-boosted ranking using document relationships

## Usage

```rust
use std::sync::Arc;
use frankensearch_fusion::{TwoTierSearcher, rrf_fuse, RrfConfig};
use frankensearch_core::{TwoTierConfig, VectorHit, ScoredResult};

// Fuse vector and lexical results
let vector_hits = vec![/* VectorHit results */];
let lexical_hits = vec![/* ScoredResult results */];
let config = RrfConfig::default();
let fused = rrf_fuse(&vector_hits, &lexical_hits, &config);

// Two-tier progressive search
// let searcher = TwoTierSearcher::new(index, embedder, TwoTierConfig::default());
// let (results, metrics) = searcher.search_collect(&cx, "query", 10).await?;
```

### Final ranking with optional quality refinement

`TwoTierSearcher::with_reranker` enables a cross-encoder with the `rerank`
feature. `with_mmr_config` enables diversity ranking without that feature.
Both retain a bounded candidate window beyond the displayed page, so a final
ranker can promote a document outside the Initial top-k.

Final ranking uses the refined pool when quality succeeds and the retained
Initial pool when quality is absent, disabled, skipped, or fails. A fast-only
search can therefore emit `Initial` followed by `Reranked`; a quality failure
can emit `Initial`, `RefinementFailed`, then `Reranked`. Quality skip and failure
reasons remain in the metrics. Cancellation at a phase callback stops later
ranking work, and failed or unapplied ranking preserves the preceding result.
MMR uses only fast vectors when quality did not succeed, and skips a pool that
no single eligible vector space covers. Exclusion-only lexical browsing keeps
its single Initial phase.

Use `search` to observe each phase or `search_collect_with_text` to obtain the
last successful ranking with document text available to the cross-encoder.

### Synchronous progressive retrieval

`SyncTwoTierSearcher::search_iter` starts work when the iterator is consumed.
The first `next()` validates the query identities and returns the fast/lexical
`Initial` phase. Only a subsequent `next()` performs quality retrieval and
blending, so a UI can display the initial results before asking for refinement.
Dropping the iterator after the initial phase avoids that remaining work.

```rust
use frankensearch_core::{SearchPhase, TieredQueryEmbeddings};
use frankensearch_fusion::SyncTwoTierSearcher;

fn display_progressively(searcher: &SyncTwoTierSearcher, query: &TieredQueryEmbeddings) {
    for phase in searcher.search_iter(query, 10) {
        match phase {
            SearchPhase::Initial { results, .. } => println!("Initial results: {results:?}"),
            SearchPhase::Refined { results, .. } => println!("Refined results: {results:?}"),
            SearchPhase::Reranked { results, .. } => println!("Final results: {results:?}"),
            SearchPhase::RefinementFailed { error, .. } => eprintln!("{error}"),
        }
    }
}
```

The iterator borrows its searcher, query and optional filter for its lifetime.
Identity or initial-retrieval failures yield `RefinementFailed` with an empty
initial result set; a refinement failure retains the already delivered results.
By default, fast-only and zero-quality-signal searches stop after the initial phase. Use
`search_collect` when only the final result set and aggregate metrics are needed.

`SyncTwoTierSearcher::with_mmr_config` enables synchronous diversity ranking
with the existing `MmrConfig`. It retains a bounded candidate window beyond
the displayed page and can emit a final `Reranked` phase after Initial or
Refined. MMR runs only when that iterator phase is requested; collection runs
it before returning. It compares every candidate in one admitted vector space,
preferring complete quality coverage after refinement and otherwise complete
fast coverage. A pool with no common eligible space keeps the previous result.
Disabled MMR performs no candidate retention or diversity work, and enabled
candidate pools above 1,000 are refused before retrieval.

`TieredQueryEmbeddings::quality_only` performs direct quality retrieval through
the same APIs. It requires a semantic quality tier with an admitted FSVI v2 owner
or a matching persisted producer fingerprint. The request never searches fast
vectors or blends fast scores, and its `Initial` phase identifies the quality
tier through its results and metrics. Optional lexical fusion preserves
the same metadata and receives the quality query vector. Quality results carry
`quality_score` and no fast-row `index`; malformed identities, absent quality,
control-only producers, and a conflicting `fast_only` configuration are refused
before retrieval. A zero-result limit still validates admission and performs no
search.

With MMR enabled, direct quality can also emit `Reranked`, using quality vectors
exclusively. Missing quality coverage for a lexical candidate skips diversity;
it never authorizes a fallback into the fast embedding space.

### Federation with separate document stores

`FederatedSearcher::search` accepts a shared `doc_id`-to-text callback. Use
`search_with_index_text` when each index has its own store or retained text
snapshot. The callback receives `(index_name, doc_id)` for the originating
shard, so exclusion clauses and optional reranking use that shard's text.
The callback can borrow request-scoped stores without cloning the stores or
requiring a `'static` lifetime.

```rust
use std::collections::BTreeMap;
use asupersync::Cx;
use frankensearch_core::SearchResult;
use frankensearch_fusion::{FederatedResponse, FederatedSearcher};

async fn search_separate_stores(
    federation: &FederatedSearcher,
    cx: &Cx,
    stores: &BTreeMap<String, BTreeMap<String, String>>,
) -> SearchResult<FederatedResponse> {
    federation
        .search_with_index_text(cx, "ownership -unsafe", 10, |index, doc_id| {
            stores.get(index).and_then(|store| store.get(doc_id)).cloned()
        })
        .await
}
```

Index names must be unique within the dispatched fanout. Document IDs remain
**global fusion identities**: equal IDs from multiple indices combine into one
hit with `appeared_in` provenance. Namespace unrelated index-local IDs at ingest;
choosing a text store does not disable deduplication. Check
`response.coverage.is_complete()` when complete shard coverage is required.
For the same shard replies, fusion scores do not depend on response order;
explicit early stopping can still change which shards contribute.

## Dependency Graph Position

```
frankensearch-core
  ^    ^    ^
  |    |    |
  embed index lexical (optional)  rerank (optional)
    \   |   /                      /
     \  |  /                      /
  frankensearch-fusion -----------
        ^
        |-- frankensearch (root)
```

## License

MIT
