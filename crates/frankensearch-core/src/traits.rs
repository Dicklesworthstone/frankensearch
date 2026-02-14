//! Core traits for the frankensearch search pipeline.
//!
//! - [`Embedder`]: Text embedding model interface (hash, model2vec, fastembed).
//! - [`Reranker`]: Cross-encoder reranking model interface.
//! - [`LexicalSearch`]: Full-text search backend interface (Tantivy, FTS5).
//!
//! All traits are object-safe (`dyn`-compatible) and `Send + Sync` for use
//! across async contexts.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::SearchResult;
use crate::types::{IndexableDocument, ScoredResult};

// ─── Model Category ─────────────────────────────────────────────────────────

/// Classification of an embedding model by its speed/quality tradeoff.
///
/// Used by `EmbedderStack` to pair a fast-tier and quality-tier embedder
/// for the two-tier progressive search pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCategory {
    /// Zero-dependency hash embedder (deterministic, no semantic understanding).
    Hash,
    /// Fast static embedder (~0.57ms, good quality). E.g., potion-128M.
    Fast,
    /// Quality transformer embedder (~128ms, excellent quality). E.g., MiniLM-L6-v2.
    Quality,
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hash => write!(f, "hash"),
            Self::Fast => write!(f, "fast"),
            Self::Quality => write!(f, "quality"),
        }
    }
}

// ─── Embedder Trait ─────────────────────────────────────────────────────────

/// Core trait for text embedding models.
///
/// Implementations must be `Send + Sync` for use across async contexts.
/// The trait is object-safe for runtime polymorphism via `Box<dyn Embedder>`.
///
/// # Contract
///
/// - `embed()` and `embed_batch()` are synchronous. Neural models (fastembed,
///   model2vec) do inference on the calling thread. For non-blocking search,
///   wrap calls in `rayon::spawn()` or an asupersync task.
/// - `dimension()` must be constant for the lifetime of the embedder.
/// - `id()` must be stable across process restarts (it's stored in FSVI headers).
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector of f32 floats.
    ///
    /// The returned vector has exactly `self.dimension()` elements.
    fn embed(&self, text: &str) -> SearchResult<Vec<f32>>;

    /// Embed a batch of text strings.
    ///
    /// Default implementation calls `embed` in a loop. Neural models should
    /// override this to exploit batch inference (ONNX has high fixed overhead
    /// but low marginal cost per additional input).
    fn embed_batch(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// The dimensionality of embedding vectors produced by this model.
    fn dimension(&self) -> usize;

    /// A unique, stable identifier for this embedder.
    ///
    /// Examples: `"fnv-hash-384"`, `"potion-multilingual-128M"`, `"all-MiniLM-L6-v2"`.
    /// Stored in FSVI index headers for embedder-revision matching.
    fn id(&self) -> &str;

    /// Whether this embedder produces semantically meaningful vectors.
    ///
    /// Hash embedders return `false`; neural models return `true`.
    fn is_semantic(&self) -> bool;

    /// The speed/quality category of this embedder.
    fn category(&self) -> ModelCategory;

    /// Whether this model supports Matryoshka Representation Learning
    /// (dimension truncation for faster search with controlled quality loss).
    fn supports_mrl(&self) -> bool {
        false
    }
}

// ─── Embedding Utilities ──────────────────────────────────────────────────

/// L2-normalizes a vector to unit length.
///
/// Returns a zero vector if the input has zero norm (avoids division by zero).
#[must_use]
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if norm_sq < f32::EPSILON {
        return vec![0.0; vec.len()];
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    vec.iter().map(|x| x * inv_norm).collect()
}

/// Computes cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
///
/// # Panics
///
/// Panics in debug mode if the vectors have different lengths.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let denom = norm_a * norm_b;
    if denom < f32::EPSILON {
        return 0.0;
    }
    dot / denom
}

/// Truncates an embedding to a target dimension and re-normalizes.
///
/// Only meaningful for models that support Matryoshka Representation Learning (MRL),
/// where the first N dimensions capture most of the variance.
///
/// Returns the original vector unchanged if `target_dim >= embedding.len()`.
#[must_use]
pub fn truncate_embedding(embedding: &[f32], target_dim: usize) -> Vec<f32> {
    if target_dim >= embedding.len() {
        return embedding.to_vec();
    }
    l2_normalize(&embedding[..target_dim])
}

// ─── Reranker Trait ─────────────────────────────────────────────────────────

/// A document for reranking: pairs a document ID with its text content.
///
/// Text must be provided because cross-encoders process query+document
/// pairs through a transformer. `ScoredResult` intentionally does not
/// carry text to avoid memory waste in the common case.
#[derive(Debug, Clone)]
pub struct RerankDocument {
    /// Document identifier.
    pub doc_id: String,
    /// Document text content for cross-encoder input.
    pub text: String,
}

/// A reranking score for a single document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankScore {
    /// Document identifier.
    pub doc_id: String,
    /// Cross-encoder relevance score (typically sigmoid-activated logit).
    pub score: f32,
    /// Position before reranking (for rank-change tracking).
    pub original_rank: usize,
}

/// Core trait for cross-encoder reranking models.
///
/// Cross-encoders process query+document pairs together through a transformer,
/// producing more accurate relevance scores than bi-encoder cosine similarity.
/// This accuracy comes at the cost of not being able to pre-compute anything:
/// every query-document pair requires a full inference pass.
///
/// # Graceful Failure
///
/// The reranking step should never block search results. If the model is
/// unavailable or inference fails, implementations should return
/// `Err(SearchError::RerankFailed { .. })` and callers should fall back
/// to the original RRF scores.
pub trait Reranker: Send + Sync {
    /// Score and re-rank documents against a query.
    ///
    /// Returns documents sorted by descending cross-encoder score.
    fn rerank(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>>;

    /// A unique identifier for this reranker model.
    fn id(&self) -> &str;
}

// ─── Lexical Search Trait ───────────────────────────────────────────────────

/// Trait for full-text lexical search backends.
///
/// Two implementations are planned:
/// - `TantivyIndex` in `frankensearch-lexical` (default, via `lexical` feature)
/// - FTS5 adapter in `frankensearch-storage` (alternative, via `fts5` feature)
///
/// Both produce `ScoredResult` with `source = ScoreSource::Lexical`.
pub trait LexicalSearch: Send + Sync {
    /// Search for documents matching the query, returning up to `limit` results
    /// sorted by BM25 relevance.
    fn search(&self, query: &str, limit: usize) -> SearchResult<Vec<ScoredResult>>;

    /// Index a single document for full-text search.
    fn index_document(&self, doc: &IndexableDocument) -> SearchResult<()>;

    /// Index a batch of documents.
    fn index_documents(&self, docs: &[IndexableDocument]) -> SearchResult<()> {
        for doc in docs {
            self.index_document(doc)?;
        }
        Ok(())
    }

    /// Commit any pending writes to the index.
    fn commit(&self) -> SearchResult<()>;

    /// Number of documents currently indexed.
    fn doc_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_category_display() {
        assert_eq!(ModelCategory::Hash.to_string(), "hash");
        assert_eq!(ModelCategory::Fast.to_string(), "fast");
        assert_eq!(ModelCategory::Quality.to_string(), "quality");
    }

    #[test]
    fn model_category_serialization() {
        let json = serde_json::to_string(&ModelCategory::Fast).unwrap();
        let decoded: ModelCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ModelCategory::Fast);
    }

    #[test]
    fn model_category_equality() {
        assert_eq!(ModelCategory::Hash, ModelCategory::Hash);
        assert_ne!(ModelCategory::Hash, ModelCategory::Fast);
        assert_ne!(ModelCategory::Fast, ModelCategory::Quality);
    }

    #[test]
    fn rerank_document_construction() {
        let doc = RerankDocument {
            doc_id: "doc-1".into(),
            text: "Some content".into(),
        };
        assert_eq!(doc.doc_id, "doc-1");
        assert_eq!(doc.text, "Some content");
    }

    #[test]
    fn rerank_score_serialization() {
        let score = RerankScore {
            doc_id: "doc-1".into(),
            score: 0.92,
            original_rank: 3,
        };

        let json = serde_json::to_string(&score).unwrap();
        let decoded: RerankScore = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.doc_id, "doc-1");
        assert!((decoded.score - 0.92).abs() < 1e-6);
        assert_eq!(decoded.original_rank, 3);
    }

    // Compile-time checks for trait object safety
    #[test]
    fn embedder_trait_is_object_safe() {
        // This function existing and compiling proves object safety.
        fn _takes_dyn_embedder(_: &dyn Embedder) {}
    }

    #[test]
    fn reranker_trait_is_object_safe() {
        fn _takes_dyn_reranker(_: &dyn Reranker) {}
    }

    #[test]
    fn lexical_search_trait_is_object_safe() {
        fn _takes_dyn_lexical(_: &dyn LexicalSearch) {}
    }

    // ─── Utility function tests ─────────────────────────────────────────

    #[test]
    fn l2_normalize_produces_unit_vector() {
        let v = vec![3.0, 4.0];
        let normalized = l2_normalize(&v);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&v);
        assert!(normalized.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    #[test]
    fn truncate_embedding_reduces_dim() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let t = truncate_embedding(&v, 2);
        assert_eq!(t.len(), 2);
        let norm: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncate_embedding_noop_when_larger() {
        let v = vec![1.0, 2.0];
        assert_eq!(truncate_embedding(&v, 10), v);
    }
}
