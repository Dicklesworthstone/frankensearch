//! Core traits, types, and error types for the frankensearch hybrid search library.
//!
//! This crate defines the shared interfaces (`Embedder`, `Reranker`, `LexicalSearch`),
//! result types (`ScoredResult`, `VectorHit`, `FusedHit`), error types (`SearchError`),
//! text canonicalization, and query classification used across all frankensearch crates.
//!
//! It has minimal external dependencies and is intended to be depended on by every
//! other crate in the workspace.

pub mod error;
pub mod traits;
pub mod types;

pub use error::{SearchError, SearchResult};
pub use traits::{
    cosine_similarity, l2_normalize, truncate_embedding, Embedder, LexicalSearch, ModelCategory,
    Reranker, RerankDocument, RerankScore,
};
pub use types::{
    FusedHit, IndexableDocument, PhaseMetrics, RankChanges, ScoreSource, ScoredResult, SearchMode,
    SearchPhase, VectorHit,
};
