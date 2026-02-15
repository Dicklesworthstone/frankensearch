//! Core traits for the frankensearch search pipeline.
//!
//! - [`Embedder`]: Text embedding model interface (hash, model2vec, fastembed).
//! - [`Reranker`]: Cross-encoder reranking model interface.
//! - [`LexicalSearch`]: Full-text search backend interface (Tantivy, FTS5).
//!
//! Async operations are represented as boxed futures so the traits remain
//! dyn-compatible for runtime polymorphism (`Box<dyn Embedder>`, etc.).

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use asupersync::Cx;
use serde::{Deserialize, Serialize};

use crate::error::{SearchError, SearchResult};
use crate::types::{
    EmbeddingMetrics, IndexMetrics, IndexableDocument, ScoredResult, SearchMetrics,
};

/// Boxed future carrying a `SearchResult<T>`.
pub type SearchFuture<'a, T> = Pin<Box<dyn Future<Output = SearchResult<T>> + Send + 'a>>;

// ─── Model Category ─────────────────────────────────────────────────────────

/// Classification of an embedding model by its speed/quality tradeoff.
///
/// Used by `EmbedderStack` to pair a fast-tier and quality-tier embedder
/// for the two-tier progressive search pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCategory {
    /// Hash-based (FNV-1a): ultra-fast, deterministic, not semantically meaningful.
    HashEmbedder,
    /// Static token embeddings (Model2Vec/potion): fast with good semantic quality.
    StaticEmbedder,
    /// Transformer inference (MiniLM/BGE): highest quality but slower.
    TransformerEmbedder,
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HashEmbedder => write!(f, "hash_embedder"),
            Self::StaticEmbedder => write!(f, "static_embedder"),
            Self::TransformerEmbedder => write!(f, "transformer_embedder"),
        }
    }
}

impl ModelCategory {
    /// Returns the default progressive tier for this model category.
    #[must_use]
    pub const fn default_tier(self) -> ModelTier {
        match self {
            Self::HashEmbedder | Self::StaticEmbedder => ModelTier::Fast,
            Self::TransformerEmbedder => ModelTier::Quality,
        }
    }

    /// Whether this category is semantically meaningful by default.
    #[must_use]
    pub const fn default_semantic_flag(self) -> bool {
        !matches!(self, Self::HashEmbedder)
    }
}

/// Tier assignment in the progressive two-tier pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTier {
    /// Ultra-fast path for immediate results.
    Fast,
    /// Higher-quality path for deferred refinement.
    Quality,
}

impl fmt::Display for ModelTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fast => write!(f, "fast"),
            Self::Quality => write!(f, "quality"),
        }
    }
}

/// Static metadata describing an embedder implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Stable model identifier used in index metadata.
    pub id: String,
    /// Human-friendly model name.
    pub name: String,
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Embedder category by architecture/performance profile.
    pub category: ModelCategory,
    /// Default tier assignment in progressive search.
    pub tier: ModelTier,
    /// Whether embeddings encode semantic similarity.
    pub is_semantic: bool,
    /// Whether Matryoshka truncation is supported.
    pub supports_mrl: bool,
    /// Optional upstream model id (e.g., `HuggingFace`).
    pub huggingface_id: Option<String>,
    /// Optional model footprint on disk.
    pub size_bytes: Option<u64>,
    /// Optional model license string.
    pub license: Option<String>,
}

// ─── Embedder Trait ─────────────────────────────────────────────────────────

/// Core trait for text embedding models.
///
/// Implementations run under structured concurrency, so each async operation
/// receives a capability context (`&Cx`) as its first parameter.
///
/// # Contract
///
/// - `embed()` and `embed_batch()` are cancel-aware and return boxed futures.
/// - `dimension()` must be constant for the lifetime of the embedder.
/// - `id()` must be stable across process restarts (it's stored in FSVI headers).
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector of f32 floats.
    ///
    /// The returned vector has exactly `self.dimension()` elements.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if embedding inference fails.
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>>;

    /// Embed a batch of text strings.
    ///
    /// Default implementation calls `embed` in a loop. Neural models should
    /// override this to exploit batch inference (ONNX has high fixed overhead
    /// but low marginal cost per additional input).
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any embedding inference fails.
    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                out.push(self.embed(cx, text).await?);
            }
            Ok(out)
        })
    }

    /// The dimensionality of embedding vectors produced by this model.
    fn dimension(&self) -> usize;

    /// A unique, stable identifier for this embedder.
    ///
    /// Examples: `"fnv-hash-384"`, `"potion-multilingual-128M"`, `"all-MiniLM-L6-v2"`.
    /// Stored in FSVI index headers for embedder-revision matching.
    fn id(&self) -> &str;

    /// Human-readable model name.
    fn model_name(&self) -> &str;

    /// Whether this embedder is loaded and operational.
    fn is_ready(&self) -> bool {
        true
    }

    /// Whether this embedder produces semantically meaningful vectors.
    ///
    /// Hash embedders return `false`; neural models return `true`.
    fn is_semantic(&self) -> bool;

    /// The speed/quality category of this embedder.
    fn category(&self) -> ModelCategory;

    /// Default progressive tier assignment.
    fn tier(&self) -> ModelTier {
        self.category().default_tier()
    }

    /// Whether this model supports Matryoshka Representation Learning
    /// (dimension truncation for faster search with controlled quality loss).
    fn supports_mrl(&self) -> bool {
        false
    }

    /// Truncate and re-normalize embedding to `target_dim`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when `target_dim` is zero.
    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> SearchResult<Vec<f32>> {
        if target_dim == 0 {
            return Err(SearchError::InvalidConfig {
                field: "target_dim".to_owned(),
                value: "0".to_owned(),
                reason: "target dimension must be at least 1".to_owned(),
            });
        }

        if target_dim >= embedding.len() {
            return Ok(embedding.to_vec());
        }

        Ok(l2_normalize(&embedding[..target_dim]))
    }
}

// ─── Embedding Utilities ──────────────────────────────────────────────────

/// L2-normalizes a vector to unit length.
///
/// Returns a zero vector if the input has zero norm (avoids division by zero).
#[must_use]
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
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
    // Runtime length check — debug_assert is stripped in release builds,
    // and zip would silently truncate mismatched vectors.
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let denom = norm_a * norm_b;
    if !denom.is_finite() || denom < f32::EPSILON {
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
    ///
    /// # Errors
    ///
    /// Returns `SearchError::RerankFailed` if cross-encoder inference fails.
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>>;

    /// A unique identifier for this reranker model.
    fn id(&self) -> &str;

    /// Human-friendly reranker model name.
    fn model_name(&self) -> &str;

    /// Maximum supported token length for query+document pair input.
    fn max_length(&self) -> usize {
        512
    }

    /// Whether this reranker is loaded and ready for inference.
    fn is_available(&self) -> bool {
        true
    }
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
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or the search backend fails.
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>>;

    /// Index a single document for full-text search.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the document cannot be indexed.
    fn index_document<'a>(&'a self, cx: &'a Cx, doc: &'a IndexableDocument)
    -> SearchFuture<'a, ()>;

    /// Index a batch of documents.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if any document cannot be indexed.
    fn index_documents<'a>(
        &'a self,
        cx: &'a Cx,
        docs: &'a [IndexableDocument],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async move {
            for doc in docs {
                self.index_document(cx, doc).await?;
            }
            Ok(())
        })
    }

    /// Commit any pending writes to the index.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the commit fails (e.g., I/O error).
    fn commit<'a>(&'a self, cx: &'a Cx) -> SearchFuture<'a, ()>;

    /// Number of documents currently indexed.
    fn doc_count(&self) -> usize;
}

// ─── Metrics Exporter Trait ─────────────────────────────────────────────────

/// Trait for exporting search/index/embed telemetry to external consumers.
///
/// Implementations must be non-blocking and fast, because callbacks are invoked
/// directly from hot paths.
pub trait MetricsExporter: fmt::Debug + Send + Sync {
    /// Called when a search request completes.
    fn on_search_completed(&self, metrics: &SearchMetrics);

    /// Called when an embedding operation completes.
    fn on_embedding_completed(&self, metrics: &EmbeddingMetrics);

    /// Called when index state changes after an update/commit.
    fn on_index_updated(&self, metrics: &IndexMetrics);

    /// Called when a search pipeline error is observed.
    fn on_error(&self, error: &SearchError);
}

/// Shared handle for dynamic telemetry exporters.
pub type SharedMetricsExporter = Arc<dyn MetricsExporter>;

/// No-op exporter used when no telemetry sink is attached.
///
/// This is intentionally empty so callers can cheaply opt out of telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpMetricsExporter;

impl MetricsExporter for NoOpMetricsExporter {
    fn on_search_completed(&self, _: &SearchMetrics) {}

    fn on_embedding_completed(&self, _: &EmbeddingMetrics) {}

    fn on_index_updated(&self, _: &IndexMetrics) {}

    fn on_error(&self, _: &SearchError) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_category_display() {
        assert_eq!(ModelCategory::HashEmbedder.to_string(), "hash_embedder");
        assert_eq!(ModelCategory::StaticEmbedder.to_string(), "static_embedder");
        assert_eq!(
            ModelCategory::TransformerEmbedder.to_string(),
            "transformer_embedder"
        );
    }

    #[test]
    fn model_category_serialization() {
        let json = serde_json::to_string(&ModelCategory::StaticEmbedder).unwrap();
        let decoded: ModelCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ModelCategory::StaticEmbedder);
    }

    #[test]
    fn model_category_equality() {
        assert_eq!(ModelCategory::HashEmbedder, ModelCategory::HashEmbedder);
        assert_ne!(ModelCategory::HashEmbedder, ModelCategory::StaticEmbedder);
        assert_ne!(
            ModelCategory::StaticEmbedder,
            ModelCategory::TransformerEmbedder
        );
    }

    #[test]
    fn model_category_default_tier() {
        assert_eq!(ModelCategory::HashEmbedder.default_tier(), ModelTier::Fast);
        assert_eq!(
            ModelCategory::StaticEmbedder.default_tier(),
            ModelTier::Fast
        );
        assert_eq!(
            ModelCategory::TransformerEmbedder.default_tier(),
            ModelTier::Quality
        );
    }

    #[test]
    fn model_tier_display() {
        assert_eq!(ModelTier::Fast.to_string(), "fast");
        assert_eq!(ModelTier::Quality.to_string(), "quality");
    }

    #[test]
    fn model_info_roundtrip() {
        let info = ModelInfo {
            id: "potion-multilingual-128M".to_owned(),
            name: "Potion 128M".to_owned(),
            dimension: 256,
            category: ModelCategory::StaticEmbedder,
            tier: ModelTier::Fast,
            is_semantic: true,
            supports_mrl: false,
            huggingface_id: Some("minishlab/potion-multilingual-128M".to_owned()),
            size_bytes: Some(128_000_000),
            license: Some("apache-2.0".to_owned()),
        };

        let json = serde_json::to_string(&info).unwrap();
        let decoded: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, info);
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

    #[test]
    fn metrics_exporter_trait_is_object_safe() {
        fn _takes_dyn_metrics_exporter(_: &dyn MetricsExporter) {}
    }

    #[test]
    fn noop_metrics_exporter_callbacks_are_noops() {
        let exporter = NoOpMetricsExporter;

        let search_metrics = SearchMetrics {
            mode: crate::types::SearchMode::Hybrid,
            query_class: None,
            total_latency_ms: 10.0,
            phase1_latency_ms: Some(4.0),
            phase2_latency_ms: Some(6.0),
            result_count: 8,
            lexical_candidates: 30,
            semantic_candidates: 25,
            refined: true,
        };
        let embedding_metrics = EmbeddingMetrics {
            embedder_id: "fnv-hash-384".into(),
            batch_size: 1,
            duration_ms: 0.07,
            dimension: 384,
            is_semantic: false,
        };
        let index_metrics = IndexMetrics {
            doc_count: 100,
            index_size_bytes: 4096,
            updated_docs: 1,
            staleness_detected: false,
        };

        exporter.on_search_completed(&search_metrics);
        exporter.on_embedding_completed(&embedding_metrics);
        exporter.on_index_updated(&index_metrics);
        exporter.on_error(&SearchError::SearchTimeout {
            elapsed_ms: 11,
            budget_ms: 10,
        });
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

    #[test]
    fn model_category_default_semantic_flag() {
        assert!(!ModelCategory::HashEmbedder.default_semantic_flag());
        assert!(ModelCategory::StaticEmbedder.default_semantic_flag());
        assert!(ModelCategory::TransformerEmbedder.default_semantic_flag());
    }
}
