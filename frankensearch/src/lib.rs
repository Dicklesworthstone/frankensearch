//! # frankensearch
//!
//! Two-tier hybrid search for Rust: sub-millisecond initial results,
//! quality-refined rankings in ~150ms.
//!
//! frankensearch combines **lexical** (Tantivy BM25) and **semantic** (vector
//! cosine similarity) search via [Reciprocal Rank Fusion][rrf], with a two-tier
//! progressive embedding model that delivers results in two phases:
//!
//! 1. **Phase 1 (Initial):** Fast embedder (potion-128M, 256d, ~0.57ms) produces
//!    results immediately via brute-force vector search + optional BM25 fusion.
//! 2. **Phase 2 (Refined):** Quality embedder (MiniLM-L6-v2, 384d, ~128ms)
//!    re-scores the top candidates for higher relevance.
//!
//! Consumers receive results progressively via [`SearchPhase`] callbacks, so UIs
//! can display fast results while quality refinement runs in the background.
//!
//! [rrf]: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
//!
//! # Quick Start
//!
//! Build an index and search it (requires only the default `hash` feature):
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use frankensearch::prelude::*;
//! use frankensearch::{EmbedderStack, HashEmbedder, IndexBuilder, TwoTierIndex};
//! use frankensearch_core::traits::Embedder;
//!
//! asupersync::test_utils::run_test_with_cx(|cx| async move {
//!     // Build an index
//!     let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
//!     let quality = Arc::new(HashEmbedder::default_384()) as Arc<dyn Embedder>;
//!     let stack = EmbedderStack::from_parts(fast, Some(quality));
//!
//!     let stats = IndexBuilder::new("./my_index")
//!         .with_embedder_stack(stack)
//!         .add_document("doc-1", "Rust ownership and borrowing")
//!         .add_document("doc-2", "Python garbage collection")
//!         .build(&cx)
//!         .await
//!         .expect("build index");
//!
//!     // Search
//!     let fast = Arc::new(HashEmbedder::default_256()) as Arc<dyn Embedder>;
//!     let index = Arc::new(TwoTierIndex::open("./my_index", TwoTierConfig::default()).unwrap());
//!     let searcher = TwoTierSearcher::new(index, fast, TwoTierConfig::default());
//!     let (results, metrics) = searcher
//!         .search_collect(&cx, "memory management", 10)
//!         .await
//!         .expect("search");
//!
//!     for result in &results {
//!         println!("{}: {:.4}", result.doc_id, result.score);
//!     }
//! });
//! ```
//!
//! # Architecture
//!
//! ```text
//!  Query ─┬─► Fast Embed (256d) ─► Vector Search ─┐
//!         │                                         ├─► RRF Fusion ─► Phase 1 Results
//!         └─► Tantivy BM25 (optional) ─────────────┘
//!                                                        │
//!                                              Quality Embed (384d)
//!                                                        │
//!                                                   Score Blend
//!                                                        │
//!                                                  Phase 2 Results
//! ```
//!
//! ## Crate Layout
//!
//! | Crate | Purpose |
//! |-------|---------|
//! | [`frankensearch-core`](core) | Types, traits, errors, config |
//! | [`frankensearch-embed`](embed) | Embedder implementations (hash, model2vec, fastembed) |
//! | [`frankensearch-index`](index) | FSVI vector index format, brute-force + HNSW search |
//! | [`frankensearch-fusion`](fusion) | RRF fusion, blending, [`TwoTierSearcher`] orchestration |
//! | `frankensearch-lexical` | Tantivy BM25 backend (feature-gated) |
//! | `frankensearch-rerank` | `FlashRank` cross-encoder (feature-gated) |
//!
//! ## Key Types
//!
//! - [`IndexBuilder`] — Build a search index from documents
//! - [`TwoTierSearcher`] — Progressive two-phase search orchestrator
//! - [`TwoTierConfig`] — Search configuration (blend factor, budgets, fast-only mode)
//! - [`TwoTierMetrics`] — Per-search timing and diagnostic metrics
//! - [`SearchPhase`] — Progressive result delivery (Initial / Refined / `RefinementFailed`)
//! - [`EmbedderStack`] — Fast + optional quality embedder pair
//! - [`VectorIndex`] — Low-level FSVI vector index reader
//!
//! # Performance
//!
//! Measured on a single core (no GPU), 10K document corpus:
//!
//! | Operation | Embedder | Latency |
//! |-----------|----------|---------|
//! | Hash embed (256d) | FNV-1a | ~11 μs |
//! | Fast embed (256d) | potion-128M | ~0.57 ms |
//! | Quality embed (384d) | MiniLM-L6-v2 | ~128 ms |
//! | Vector search (10K, top-10) | brute-force | ~2 ms |
//! | RRF fusion (500+500) | - | ~1 ms |
//! | Full pipeline (hash, 10K) | hash only | ~3 ms |
//!
//! # Feature Flags
//!
//! | Feature      | Description                                            |
//! |--------------|--------------------------------------------------------|
//! | `hash`       | FNV-1a hash embedder (default, zero dependencies)      |
//! | `model2vec`  | potion-128M static embedder (fast tier, ~0.57ms)       |
//! | `fastembed`  | MiniLM-L6-v2 ONNX embedder (quality tier, ~128ms)      |
//! | `lexical`    | Tantivy BM25 full-text search                          |
//! | `rerank`     | `FlashRank` cross-encoder reranking                    |
//! | `ann`        | HNSW approximate nearest-neighbor index                |
//! | `download`   | Model auto-download from `HuggingFace` via asupersync  |
//! | `storage`    | `FrankenSQLite` document metadata + embedding queue     |
//! | `durability` | `RaptorQ` self-healing for persistent index artifacts   |
//! | `fts5`       | Enables `FrankenSQLite` FTS5 lexical backend wiring     |
//! | `semantic`   | `hash` + `model2vec` + `fastembed`                     |
//! | `hybrid`     | `semantic` + `lexical`                                 |
//! | `persistent` | `hybrid` + `storage`                                   |
//! | `durable`    | `persistent` + `durability`                            |
//! | `full`       | `durable` + `rerank` + `ann` + `download`             |
//! | `full-fts5`  | `full` + `fts5`                                        |
//!
//! ## Recommended Feature Combinations
//!
//! - **Development/testing:** `default` (hash only, no downloads)
//! - **Production semantic:** `semantic` + `download`
//! - **Persistent hybrid search:** `persistent`
//! - **Maximum durability:** `durable` or `full`
//!
//! # Async Runtime
//!
//! frankensearch uses [asupersync](https://docs.rs/asupersync) exclusively — **not
//! tokio**. All async methods take `&Cx` (capability context) as their first
//! parameter. The `Cx` is provided by the consumer's asupersync runtime;
//! frankensearch never creates its own runtime.

#[cfg(all(feature = "fts5", not(feature = "storage")))]
compile_error!("feature `fts5` requires feature `storage`");

#[cfg(all(
    feature = "persistent",
    not(all(feature = "hybrid", feature = "storage"))
))]
compile_error!("feature `persistent` requires both `hybrid` and `storage`");

#[cfg(all(
    feature = "durable",
    not(all(feature = "persistent", feature = "durability"))
))]
compile_error!("feature `durable` requires both `persistent` and `durability`");

#[cfg(all(feature = "full-fts5", not(all(feature = "full", feature = "fts5"))))]
compile_error!("feature `full-fts5` requires both `full` and `fts5`");

// ─── Sub-crate module aliases (advanced access) ─────────────────────────────

/// Core types, traits, and error definitions.
pub use frankensearch_core as core;
/// Embedding model implementations and auto-detection.
pub use frankensearch_embed as embed;
/// RRF fusion, blending, search orchestration, and queue management.
pub use frankensearch_fusion as fusion;
/// Vector index I/O (FSVI format) and brute-force/HNSW search.
pub use frankensearch_index as index;

#[cfg(feature = "lexical")]
/// Tantivy-based lexical (BM25) search backend.
pub use frankensearch_lexical as lexical;

#[cfg(feature = "rerank")]
/// FlashRank cross-encoder reranking.
pub use frankensearch_rerank as rerank;

#[cfg(feature = "storage")]
/// FrankenSQLite storage backend.
pub use frankensearch_storage as storage;

#[cfg(feature = "durability")]
/// RaptorQ self-healing durability layer.
pub use frankensearch_durability as durability;

// ─── Async runtime re-exports ───────────────────────────────────────────────

/// Capability context for structured concurrency (from asupersync).
///
/// All async search methods take `&Cx` as their first parameter. The `Cx` flows
/// down from the consumer's asupersync runtime — frankensearch does not create
/// its own runtime.
pub use asupersync::Cx;

// ─── Core types (always available) ──────────────────────────────────────────

// Error types
pub use frankensearch_core::error::{SearchError, SearchResult};

// Configuration
pub use frankensearch_core::config::{TwoTierConfig, TwoTierMetrics};

// Search result types
pub use frankensearch_core::types::{
    FusedHit, IndexableDocument, PhaseMetrics, RankChanges, ScoreSource, ScoredResult, SearchMode,
    SearchPhase, VectorHit,
};

// Telemetry types
pub use frankensearch_core::types::{EmbeddingMetrics, IndexMetrics, SearchMetrics};

// Traits
pub use frankensearch_core::traits::{
    Embedder, LexicalSearch, MetricsExporter, ModelCategory, ModelInfo, ModelTier,
    NoOpMetricsExporter, Reranker, SearchFuture, SharedMetricsExporter,
};

// Reranker support types
pub use frankensearch_core::traits::{RerankDocument, RerankScore};

// Query classification
pub use frankensearch_core::query_class::QueryClass;

// Text canonicalization
pub use frankensearch_core::canonicalize::{Canonicalizer, DefaultCanonicalizer};
pub use frankensearch_core::fingerprint::{
    DEFAULT_SEMANTIC_CHANGE_THRESHOLD, DocumentFingerprint, SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD,
};

// Utility functions
pub use frankensearch_core::traits::{cosine_similarity, l2_normalize, truncate_embedding};

// ─── Embedder stack (always available) ──────────────────────────────────────

pub use frankensearch_embed::auto_detect::{DimReduceEmbedder, EmbedderStack, TwoTierAvailability};
pub use frankensearch_embed::model_registry::{EmbedderRegistry, RegisteredEmbedder};

// ─── Vector index (always available) ────────────────────────────────────────

pub use frankensearch_index::{TwoTierIndex, TwoTierIndexBuilder, VectorIndex, VectorIndexWriter};

#[cfg(feature = "ann")]
pub use frankensearch_index::{AnnSearchStats, HnswConfig, HnswIndex};

// ─── Fusion and search orchestration (always available) ─────────────────────

pub use frankensearch_fusion::{
    RrfConfig, TwoTierSearcher, blend_two_tier, candidate_count, rrf_fuse,
};

// ─── Feature-gated embedder re-exports ──────────────────────────────────────

#[cfg(feature = "hash")]
pub use frankensearch_embed::hash_embedder::{HashAlgorithm, HashEmbedder};

#[cfg(feature = "model2vec")]
pub use frankensearch_embed::model2vec_embedder::Model2VecEmbedder;

#[cfg(feature = "fastembed")]
pub use frankensearch_embed::fastembed_embedder::FastEmbedEmbedder;

// ─── Feature-gated lexical re-exports ───────────────────────────────────────

#[cfg(feature = "lexical")]
pub use frankensearch_lexical::TantivyIndex;

// ─── Feature-gated reranker re-exports ──────────────────────────────────────

#[cfg(feature = "rerank")]
pub use frankensearch_rerank::{FlashRankReranker, rerank_step};

// ─── IndexBuilder convenience API ────────────────────────────────────────────

mod index_builder;
pub use index_builder::{IndexBuildStats, IndexBuilder, IndexProgress};

// ─── Prelude ────────────────────────────────────────────────────────────────

/// Convenience re-exports for common usage.
///
/// ```rust,ignore
/// use frankensearch::prelude::*;
/// ```
pub mod prelude {
    pub use asupersync::Cx;

    pub use crate::{
        DocumentFingerprint, Embedder, LexicalSearch, Reranker, ScoreSource, ScoredResult,
        SearchError, SearchPhase, SearchResult, TwoTierConfig, TwoTierMetrics, TwoTierSearcher,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_types_accessible() {
        // Verify key types are re-exported and usable.
        let _config = TwoTierConfig::default();
        let _metrics = TwoTierMetrics::default();
        let _rrf = RrfConfig::default();
    }

    #[test]
    fn error_types_accessible() {
        let err: SearchError = SearchError::DurabilityDisabled;
        let result: SearchResult<()> = Err(err);
        assert!(result.is_err());
    }

    #[test]
    fn prelude_provides_essentials() {
        fn _takes_cx(_cx: &Cx) {}

        use crate::prelude::*;

        let _config = TwoTierConfig::default();
        let _metrics = TwoTierMetrics::default();
    }

    #[test]
    fn score_source_accessible() {
        assert_ne!(ScoreSource::Hybrid, ScoreSource::Lexical);
    }

    #[test]
    fn query_class_accessible() {
        let class = QueryClass::classify("hello world");
        assert!(matches!(
            class,
            QueryClass::NaturalLanguage | QueryClass::ShortKeyword
        ));
    }

    #[test]
    fn traits_are_object_safe() {
        fn _takes_embedder(_: &dyn Embedder) {}
        fn _takes_reranker(_: &dyn Reranker) {}
        fn _takes_lexical(_: &dyn LexicalSearch) {}
        fn _takes_metrics(_: &dyn MetricsExporter) {}
    }

    #[test]
    fn utility_functions_accessible() {
        let v = l2_normalize(&[3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn indexable_document_accessible() {
        let doc = IndexableDocument::new("id", "content").with_title("title");
        assert_eq!(doc.id, "id");
    }

    #[test]
    fn embedder_stack_accessible() {
        assert!(matches!(
            TwoTierAvailability::HashOnly,
            TwoTierAvailability::HashOnly
        ));
    }

    #[test]
    fn sub_crate_modules_accessible() {
        // Advanced users can access sub-crate modules directly.
        let _ = core::error::SearchError::DurabilityDisabled;
        let _ = fusion::rrf::RrfConfig::default();
    }

    #[cfg(feature = "hash")]
    #[test]
    fn hash_embedder_accessible() {
        assert!(matches!(
            HashAlgorithm::FnvModular,
            HashAlgorithm::FnvModular
        ));
    }

    #[cfg(feature = "storage")]
    #[test]
    fn storage_reexports_accessible() {
        let schema_version = storage::SCHEMA_VERSION;
        assert!(schema_version >= 1);
    }

    #[cfg(feature = "durability")]
    #[test]
    fn durability_reexports_accessible() {
        let trailer_version = durability::REPAIR_TRAILER_VERSION;
        assert!(trailer_version >= 1);
    }
}
