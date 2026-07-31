//! Core traits for the frankensearch search pipeline.
//!
//! - [`Embedder`]: Text embedding model interface (hash, model2vec, fastembed).
//! - [`Reranker`]: Cross-encoder reranking model interface.
//! - [`LexicalSearch`]: Full-text search backend interface (Tantivy, FTS5).
//!
//! Async operations are represented as boxed futures so the traits remain
//! dyn-compatible for runtime polymorphism (`Box<dyn Embedder>`, etc.).

use std::any::Any;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use asupersync::Cx;
use serde::{Deserialize, Serialize};

use crate::error::{SearchError, SearchResult};
use crate::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use crate::types::{
    EmbeddingMetrics, IndexMetrics, IndexableDocument, ScoredResult, SearchMetrics,
};

/// Boxed future carrying a `SearchResult<T>`.
pub type SearchFuture<'a, T> = Pin<Box<dyn Future<Output = SearchResult<T>> + Send + 'a>>;

fn bounded_embedder_diagnostic_id(id: &str) -> String {
    if !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        id.to_owned()
    } else {
        "<redacted-embedder-id>".to_owned()
    }
}

/// Vector output bound to the complete space, producer, input, and storage contracts.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityBoundEmbedding {
    /// Raw f32 vector values.
    pub values: Vec<f32>,
    /// Complete validated identity bundle used to produce the vector.
    pub identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for IdentityBoundEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdentityBoundEmbedding")
            .field("dimension", &self.values.len())
            .field("identity", &self.identity.fingerprint())
            .finish_non_exhaustive()
    }
}

impl IdentityBoundEmbedding {
    /// Validate the identity bundle and exact vector dimension.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the identity is malformed or the vector
    /// length does not match its declared mathematical space.
    pub fn validate(&self) -> SearchResult<()> {
        self.identity.validate()?;
        let declared_dimension = usize::try_from(self.identity.space.dimension).map_err(|_| {
            SearchError::InvalidConfig {
                field: "identity_bound_embedding.dimension".to_owned(),
                value: self.identity.space.dimension.to_string(),
                reason: "dimension does not fit usize".to_owned(),
            }
        })?;
        if self.values.len() != declared_dimension {
            return Err(SearchError::InvalidConfig {
                field: "identity_bound_embedding.values".to_owned(),
                value: self.values.len().to_string(),
                reason: format!("expected {declared_dimension} vector elements"),
            });
        }
        if self.identity.storage.quantization != QuantizationFormat::F32 {
            return Err(SearchError::InvalidConfig {
                field: "identity_bound_embedding.storage.quantization".to_owned(),
                value: format!("{:?}", self.identity.storage.quantization),
                reason: "an in-process Vec<f32> output must carry an f32 storage identity"
                    .to_owned(),
            });
        }
        if !self.identity.storage.format.starts_with("in-memory-") {
            return Err(SearchError::InvalidConfig {
                field: "identity_bound_embedding.storage.format".to_owned(),
                value: self.identity.storage.format.clone(),
                reason: "an in-process Vec<f32> output must carry an in-memory storage format"
                    .to_owned(),
            });
        }
        if !matches!(
            self.identity.storage.endianness.as_str(),
            "native-f32-values" | "native-test-only"
        ) {
            return Err(SearchError::InvalidConfig {
                field: "identity_bound_embedding.storage.endianness".to_owned(),
                value: self.identity.storage.endianness.clone(),
                reason: "an in-process Vec<f32> output must carry a native-value contract"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

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
    /// Cloud API embeddings (`OpenAI`, Gemini): high quality, network-dependent latency.
    ApiEmbedder,
}

impl fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HashEmbedder => write!(f, "hash_embedder"),
            Self::StaticEmbedder => write!(f, "static_embedder"),
            Self::TransformerEmbedder => write!(f, "transformer_embedder"),
            Self::ApiEmbedder => write!(f, "api_embedder"),
        }
    }
}

impl ModelCategory {
    /// Returns the default progressive tier for this model category.
    #[must_use]
    pub const fn default_tier(self) -> ModelTier {
        match self {
            Self::HashEmbedder | Self::StaticEmbedder => ModelTier::Fast,
            Self::TransformerEmbedder | Self::ApiEmbedder => ModelTier::Quality,
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
/// - `embed()` and `embed_batch()` are raw inference primitives; any caller
///   persisting, comparing, caching, or transporting vectors must use
///   `embed_bound()` or `embed_batch_bound()` so space and producer identity
///   travel with the values.
/// - `dimension()` must be constant for the lifetime of the embedder.
/// - `id()` must be stable across process restarts for diagnostics and registry
///   selection, but never establishes vector-space compatibility.
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector of f32 floats.
    ///
    /// The returned vector has exactly `self.dimension()` elements.
    /// This raw primitive carries no compatibility proof; use
    /// [`Self::embed_bound`] outside an implementation-local inference path.
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
    /// This raw primitive carries no compatibility proof; use
    /// [`Self::embed_batch_bound`] when values leave the embedder boundary.
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

    /// Embed one input and bind the output to the complete verified identity.
    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            let bound = IdentityBoundEmbedding {
                values: self.embed(cx, text).await?,
                identity: self.identity()?.clone(),
            };
            bound.validate()?;
            Ok(bound)
        })
    }

    /// Embed a batch and bind every output to the same verified identity.
    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            let identity = self.identity()?.clone();
            self.embed_batch(cx, texts)
                .await?
                .into_iter()
                .map(|values| {
                    let bound = IdentityBoundEmbedding {
                        values,
                        identity: identity.clone(),
                    };
                    bound.validate()?;
                    Ok(bound)
                })
                .collect()
        })
    }

    /// Complete immutable identity of this embedder and its output/storage contract.
    ///
    /// Legacy/custom implementations that have not supplied a complete identity
    /// fail closed here; raw model names and dimensions never synthesize
    /// compatibility.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the implementation is not identity-aware.
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Err(SearchError::InvalidConfig {
            field: "embedder.identity".to_owned(),
            value: bounded_embedder_diagnostic_id(self.id()),
            reason: "embedder did not supply a complete immutable identity bundle".to_owned(),
        })
    }

    /// The dimensionality of embedding vectors produced by this model.
    fn dimension(&self) -> usize;

    /// A unique, stable identifier for this embedder.
    ///
    /// Examples: `"fnv-hash-384"`, `"potion-multilingual-128M"`, `"all-MiniLM-L6-v2"`.
    /// This is operational metadata only. Persistence and compatibility checks
    /// must use the complete immutable identity bundle.
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

// ─── Synchronous Embedder Bridge ─────────────────────────────────────────

/// Synchronous embedding interface for host projects that call embedders from
/// non-async contexts.
///
/// Implement this trait for embedders whose `embed` operations are inherently
/// synchronous (e.g., hash embedders, CPU-only ONNX inference). The companion
/// [`SyncEmbedderAdapter`] wraps any `SyncEmbed` implementor into a full
/// async [`Embedder`], suitable for use anywhere frankensearch expects one.
///
/// # Example
///
/// ```ignore
/// use frankensearch_core::traits::{SyncEmbed, SyncEmbedderAdapter, Embedder};
///
/// struct MyHashEmbedder { dim: usize }
///
/// impl SyncEmbed for MyHashEmbedder {
///     fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> { /* ... */ }
///     fn dimension(&self) -> usize { self.dim }
///     fn id(&self) -> &str { "my-hash" }
///     fn model_name(&self) -> &str { "My Hash Embedder" }
///     fn is_semantic(&self) -> bool { false }
///     fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
/// }
///
/// // Use it as a full async Embedder:
/// let adapted: Box<dyn Embedder> = Box::new(SyncEmbedderAdapter(MyHashEmbedder { dim: 256 }));
/// ```
pub trait SyncEmbed: Send + Sync {
    /// Synchronously embed a single text into a vector.
    ///
    /// This raw primitive carries no compatibility proof; callers that persist,
    /// compare, cache, or transport the vector must use
    /// [`Self::embed_bound_sync`].
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] when embedding fails (for example model load,
    /// inference, or input validation failures).
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>>;

    /// Synchronously embed a batch of texts.
    ///
    /// Default implementation calls [`embed_sync`](Self::embed_sync) for each text.
    /// Use [`Self::embed_batch_bound_sync`] when vectors leave an
    /// implementation-local inference path.
    ///
    /// # Errors
    ///
    /// Returns the first [`SearchError`] encountered while embedding any item
    /// in the batch.
    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_sync(t)).collect()
    }

    /// Synchronously embed one input and bind it to the complete identity.
    ///
    /// # Errors
    ///
    /// Returns the embedding error or fails closed when identity/dimension
    /// validation fails.
    fn embed_bound_sync(&self, text: &str) -> SearchResult<IdentityBoundEmbedding> {
        let bound = IdentityBoundEmbedding {
            values: self.embed_sync(text)?,
            identity: self.identity()?.clone(),
        };
        bound.validate()?;
        Ok(bound)
    }

    /// Synchronously embed a batch and bind every output to one identity.
    ///
    /// # Errors
    ///
    /// Returns the first embedding or identity validation error.
    fn embed_batch_bound_sync(&self, texts: &[&str]) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        let identity = self.identity()?.clone();
        self.embed_batch_sync(texts)?
            .into_iter()
            .map(|values| {
                let bound = IdentityBoundEmbedding {
                    values,
                    identity: identity.clone(),
                };
                bound.validate()?;
                Ok(bound)
            })
            .collect()
    }

    /// Complete immutable identity of this embedder and its output/storage contract.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the implementation is not identity-aware.
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Err(SearchError::InvalidConfig {
            field: "sync_embedder.identity".to_owned(),
            value: bounded_embedder_diagnostic_id(self.id()),
            reason: "embedder did not supply a complete immutable identity bundle".to_owned(),
        })
    }

    /// The output dimensionality of embedding vectors.
    fn dimension(&self) -> usize;

    /// Unique, stable operational identifier for this embedder.
    ///
    /// It never substitutes for the immutable identity bundle in persistence or
    /// compatibility checks.
    fn id(&self) -> &str;

    /// Human-readable model name.
    fn model_name(&self) -> &str {
        self.id()
    }

    /// Whether the embedder is loaded and operational.
    fn is_ready(&self) -> bool {
        true
    }

    /// Whether this embedder produces semantically meaningful vectors.
    fn is_semantic(&self) -> bool;

    /// The speed/quality category of this embedder.
    fn category(&self) -> ModelCategory;

    /// Default progressive tier assignment.
    fn tier(&self) -> ModelTier {
        self.category().default_tier()
    }

    /// Whether this model supports Matryoshka Representation Learning.
    fn supports_mrl(&self) -> bool {
        false
    }
}

/// Adapts a [`SyncEmbed`] implementor into a full async [`Embedder`].
///
/// The sync `embed_sync()` call is wrapped in `Box::pin(async move { ... })`,
/// which is zero-cost for pure computation (hash embedders) and acceptable for
/// blocking ONNX inference when called from a `spawn_blocking` context.
pub struct SyncEmbedderAdapter<T: SyncEmbed>(pub T);

impl<T: SyncEmbed + 'static> Embedder for SyncEmbedderAdapter<T> {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { self.0.embed_sync(text) })
    }

    fn embed_batch<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move { self.0.embed_batch_sync(texts) })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        self.0.identity()
    }

    fn dimension(&self) -> usize {
        self.0.dimension()
    }

    fn id(&self) -> &str {
        self.0.id()
    }

    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    fn is_ready(&self) -> bool {
        self.0.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.0.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.0.category()
    }

    fn tier(&self) -> ModelTier {
        self.0.tier()
    }

    fn supports_mrl(&self) -> bool {
        self.0.supports_mrl()
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

/// L2-normalizes a vector to unit length **in place**.
///
/// Bit-identical to [`l2_normalize`] (same `norm_sq` accumulation, same
/// `is_finite`/`EPSILON` guard, same `x * inv_norm` scaling, zero vector on zero
/// norm) but reuses the caller's owned buffer instead of allocating a fresh `Vec` —
/// for callers that already own the vector (e.g. the hash embedder builds its
/// accumulator then normalizes it), this drops one dimension-sized allocation.
pub fn l2_normalize_in_place(vec: &mut [f32]) {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        for x in vec.iter_mut() {
            *x = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    // Element-wise scale — runtime-AVX2 (~1.7× over the SSE2 auto-vec on this
    // no-global-avx2 build); bit-identical (per-element IEEE multiply).
    crate::simd::scale_f32_in_place(vec, inv_norm);
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
    /// Raw pre-sigmoid logit, when the backend exposes it.
    ///
    /// Some cross-encoder implementations only emit a final score (after sigmoid
    /// activation). When the raw logit is unavailable, this field is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_logit: Option<f32>,
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

// ─── Synchronous Reranker Bridge ────────────────────────────────────────────

/// Synchronous reranking interface for host projects that call rerankers from
/// non-async contexts.
///
/// Implement this trait for rerankers whose `rerank` operations are inherently
/// synchronous (e.g., blocking ONNX inference). The companion
/// [`SyncRerankerAdapter`] wraps any `SyncRerank` implementor into a full
/// async [`Reranker`], suitable for use anywhere frankensearch expects one.
pub trait SyncRerank: Send + Sync {
    /// Synchronously rerank documents against a query.
    ///
    /// Returns documents sorted by descending cross-encoder score.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] when reranking fails (for example model load,
    /// inference, or input validation failures).
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>>;

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

/// Adapts a [`SyncRerank`] implementor into a full async [`Reranker`].
///
/// The sync `rerank_sync()` call is wrapped in `Box::pin(async move { ... })`,
/// which is acceptable for blocking ONNX inference when called from a
/// `spawn_blocking` context.
pub struct SyncRerankerAdapter<T: SyncRerank>(pub T);

impl<T: SyncRerank + 'static> Reranker for SyncRerankerAdapter<T> {
    fn rerank<'a>(
        &'a self,
        _cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            let mut scores = self.0.rerank_sync(query, documents)?;
            scores.sort_by(|lhs, rhs| {
                rhs.score
                    .total_cmp(&lhs.score)
                    .then_with(|| lhs.original_rank.cmp(&rhs.original_rank))
                    .then_with(|| lhs.doc_id.cmp(&rhs.doc_id))
            });
            Ok(scores)
        })
    }

    fn id(&self) -> &str {
        self.0.id()
    }

    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    fn max_length(&self) -> usize {
        self.0.max_length()
    }

    fn is_available(&self) -> bool {
        self.0.is_available()
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

    /// Search for lexical candidates that will be consumed by hybrid fusion.
    ///
    /// The default preserves the full [`Self::search`] result. Backends with a
    /// cheaper identifier-only path may override this and defer stored metadata
    /// materialization until [`Self::hydrate_fusion_metadata`] knows which
    /// candidates survived fusion.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` under the same conditions as [`Self::search`].
    fn search_fusion_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        self.search(cx, query, limit)
    }

    /// Whether [`Self::search_fusion_candidates`] omits metadata that must be
    /// restored for the final fused winners.
    fn fusion_metadata_is_deferred(&self) -> bool {
        false
    }

    /// Restore metadata for final hybrid results produced from deferred fusion
    /// candidates.
    ///
    /// Implementations should ignore results without a lexical score: those
    /// results did not survive from the lexical candidate pool. The default is
    /// a no-op because the default candidate path already returns full results.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if winner metadata cannot be materialized.
    fn hydrate_fusion_metadata<'a>(
        &'a self,
        _cx: &'a Cx,
        _results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

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

// ─── Split lexical traits (bd-8nqz.1) ───────────────────────────────────────
//
// [`LexicalSearch`] combines read and write concerns, which forces read-only
// consumers to hold writer-capable backends and lets hydration read a newer
// snapshot than the one that scored a candidate batch. The split below is the
// replacement contract: [`LexicalRead`] for search + generation-pinned
// hydration, [`LexicalWrite`] for mutation. Consumers migrate off
// [`LexicalSearch`] in the coordinated flip (fusion/facade/gauntlet), after
// which the combined trait is removed.

/// Opaque, backend-owned pin of the immutable snapshot that scored a
/// candidate batch (bd-8nqz.1).
///
/// A backend stores whatever it needs — typically an `Arc` of its published
/// search snapshot — and downcasts it back during hydration, so hydrated
/// metadata always comes from the exact generation that produced the scores.
/// Callers cannot forge or relabel a context: the payload is opaque, the
/// backend tag is read-only, and a context only originates from a backend's
/// own [`LexicalRead::search_candidates`].
pub struct LexicalHydrationContext {
    backend: &'static str,
    inner: Box<dyn Any + Send + Sync>,
}

impl LexicalHydrationContext {
    /// Wrap a backend-owned snapshot pin.
    #[must_use]
    pub fn new(backend: &'static str, inner: Box<dyn Any + Send + Sync>) -> Self {
        Self { backend, inner }
    }

    /// Stable tag of the backend that produced this context.
    #[must_use]
    pub const fn backend(&self) -> &'static str {
        self.backend
    }

    /// Downcast the opaque payload back to the backend's snapshot type.
    ///
    /// Returns `None` for a foreign context (wrong backend or wrong payload
    /// type) — backends must treat that as a typed error, never a silent
    /// no-op, so cross-engine mixing cannot pass unnoticed.
    #[must_use]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

impl fmt::Debug for LexicalHydrationContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexicalHydrationContext")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

/// Typed result of a fusion-candidate search: scored candidates plus the
/// snapshot pin required to hydrate them from the exact immutable generation
/// that scored them (bd-8nqz.1).
#[derive(Debug)]
pub struct LexicalCandidateBatch {
    results: Vec<ScoredResult>,
    context: Option<LexicalHydrationContext>,
}

impl LexicalCandidateBatch {
    /// Batch whose results already carry full metadata (no hydration needed).
    #[must_use]
    pub const fn eager(results: Vec<ScoredResult>) -> Self {
        Self {
            results,
            context: None,
        }
    }

    /// Batch with deferred metadata, pinned to the scoring snapshot.
    #[must_use]
    pub const fn deferred(results: Vec<ScoredResult>, context: LexicalHydrationContext) -> Self {
        Self {
            results,
            context: Some(context),
        }
    }

    /// Scored candidates in backend rank order.
    #[must_use]
    pub fn results(&self) -> &[ScoredResult] {
        &self.results
    }

    /// Snapshot pin for hydration; `None` when the batch is eager.
    #[must_use]
    pub const fn context(&self) -> Option<&LexicalHydrationContext> {
        self.context.as_ref()
    }

    /// Whether metadata hydration is required for final winners.
    #[must_use]
    pub const fn is_deferred(&self) -> bool {
        self.context.is_some()
    }

    /// Decompose into candidates and the hydration pin.
    #[must_use]
    pub fn into_parts(self) -> (Vec<ScoredResult>, Option<LexicalHydrationContext>) {
        (self.results, self.context)
    }
}

/// Read-only lexical search surface (bd-8nqz.1).
///
/// Search consumers (`TwoTierSearcher`, the sync searcher, `open_hybrid`
/// callers) depend on this trait alone, so a read-only reader never needs a
/// writer-capable backend or a writer lease.
pub trait LexicalRead: Send + Sync {
    /// Search for documents matching the query, returning up to `limit`
    /// results sorted by BM25 relevance, with full metadata attached.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if the query cannot be parsed or the backend
    /// fails.
    fn search<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>>;

    /// Search for fusion candidates as a typed batch pinned to the scoring
    /// snapshot.
    ///
    /// The default preserves full-metadata results as an eager batch.
    /// Backends with a cheaper deferred-metadata path override this and
    /// return [`LexicalCandidateBatch::deferred`] with their snapshot pin.
    ///
    /// # Errors
    ///
    /// Returns `SearchError` under the same conditions as [`Self::search`].
    fn search_candidates<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, LexicalCandidateBatch> {
        Box::pin(async move {
            Ok(LexicalCandidateBatch::eager(
                self.search(cx, query, limit).await?,
            ))
        })
    }

    /// Restore metadata for final winners produced from a deferred candidate
    /// batch, reading from the pinned scoring snapshot — never from a newer
    /// generation.
    ///
    /// Implementations must ignore results without a lexical score (those
    /// did not survive from the lexical candidate pool) and must reject a
    /// foreign context with a typed error. The default is a no-op for eager
    /// batches (`context == None`).
    ///
    /// # Errors
    ///
    /// Returns `SearchError` if winner metadata cannot be materialized or the
    /// context does not belong to this backend.
    fn hydrate_candidates<'a>(
        &'a self,
        _cx: &'a Cx,
        context: Option<&'a LexicalHydrationContext>,
        _results: &'a mut [ScoredResult],
    ) -> SearchFuture<'a, ()> {
        let _ = context;
        Box::pin(async { Ok(()) })
    }

    /// Number of documents currently searchable.
    fn doc_count(&self) -> usize;
}

/// Mutation/indexing surface of a lexical backend (bd-8nqz.1).
pub trait LexicalWrite: Send + Sync {
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
    use asupersync::test_utils::run_test_with_cx;

    use super::*;

    struct BoundSyncEmbedder {
        identity: EmbeddingIdentityBundleV1,
        output_dimension: usize,
    }

    impl SyncEmbed for BoundSyncEmbedder {
        fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
            Ok(vec![1.0; self.output_dimension])
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn dimension(&self) -> usize {
            self.output_dimension
        }

        fn id(&self) -> &'static str {
            "bound-sync-fixture"
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    struct UnsortedSyncReranker;

    impl SyncRerank for UnsortedSyncReranker {
        fn rerank_sync(
            &self,
            _query: &str,
            _documents: &[RerankDocument],
        ) -> SearchResult<Vec<RerankScore>> {
            Ok(vec![
                RerankScore {
                    doc_id: "doc-a".into(),
                    score: 0.8,
                    original_rank: 2,
                    raw_logit: None,
                },
                RerankScore {
                    doc_id: "doc-b".into(),
                    score: 0.8,
                    original_rank: 1,
                    raw_logit: None,
                },
                RerankScore {
                    doc_id: "doc-c".into(),
                    score: 0.3,
                    original_rank: 0,
                    raw_logit: None,
                },
            ])
        }

        fn id(&self) -> &'static str {
            "unsorted-sync-reranker"
        }

        fn model_name(&self) -> &'static str {
            "Unsorted Sync Reranker"
        }
    }

    struct UnboundSyncEmbedder;

    impl SyncEmbed for UnboundSyncEmbedder {
        fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
            Ok(vec![0.0])
        }

        fn dimension(&self) -> usize {
            1
        }

        fn id(&self) -> &'static str {
            "legacy\nforged-log-line"
        }

        fn is_semantic(&self) -> bool {
            false
        }

        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

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
            raw_logit: None,
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
    fn sync_bound_outputs_carry_identity_and_fail_on_shape_drift() {
        let embedder = BoundSyncEmbedder {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("bound-sync-fixture", 3),
            output_dimension: 3,
        };
        let bound = embedder.embed_bound_sync("text").unwrap();
        assert_eq!(bound.values, vec![1.0; 3]);
        assert_eq!(bound.identity, embedder.identity);
        assert_eq!(
            embedder.embed_batch_bound_sync(&["a", "b"]).unwrap().len(),
            2
        );

        let drifted = BoundSyncEmbedder {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("bound-sync-fixture", 2),
            output_dimension: 3,
        };
        assert!(drifted.embed_bound_sync("text").is_err());
    }

    #[test]
    fn missing_identity_diagnostic_redacts_untrusted_embedder_id() {
        let error = UnboundSyncEmbedder.identity().unwrap_err();
        assert!(error.to_string().contains("<redacted-embedder-id>"));
        assert!(!error.to_string().contains("forged-log-line"));
    }

    #[test]
    fn identity_bound_debug_redacts_vector_values() {
        let bound = IdentityBoundEmbedding {
            values: vec![12_345.5, -9_876.25],
            identity: EmbeddingIdentityBundleV1::explicit_test_model("debug-redaction", 2),
        };
        let debug = format!("{bound:?}");
        assert!(debug.contains("dimension"));
        assert!(debug.contains(&bound.identity.fingerprint()));
        assert!(!debug.contains("12345"));
        assert!(!debug.contains("9876"));
    }

    #[test]
    fn identity_bound_output_rejects_non_memory_f32_storage_claims() {
        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model("bound-storage-fixture", 2);
        identity.storage.quantization = QuantizationFormat::F16;
        let bound = IdentityBoundEmbedding {
            values: vec![1.0, 2.0],
            identity,
        };
        assert!(bound.validate().is_err());

        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model("bound-storage-fixture", 2);
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.endianness = "little-endian".to_owned();
        let bound = IdentityBoundEmbedding {
            values: vec![1.0, 2.0],
            identity,
        };
        assert!(bound.validate().is_err());

        let mut identity =
            EmbeddingIdentityBundleV1::explicit_test_model("bound-storage-fixture", 2);
        identity.storage.endianness = "little-endian".to_owned();
        let bound = IdentityBoundEmbedding {
            values: vec![1.0, 2.0],
            identity,
        };
        assert!(bound.validate().is_err());
    }

    #[test]
    fn async_bound_outputs_carry_forwarded_identity() {
        run_test_with_cx(|cx| async move {
            let identity = EmbeddingIdentityBundleV1::explicit_test_model("bound-async-fixture", 3);
            let adapter = SyncEmbedderAdapter(BoundSyncEmbedder {
                identity: identity.clone(),
                output_dimension: 3,
            });
            let bound = adapter.embed_bound(&cx, "text").await.unwrap();
            assert_eq!(bound.values, vec![1.0; 3]);
            assert_eq!(bound.identity, identity);
            assert_eq!(
                adapter
                    .embed_batch_bound(&cx, &["a", "b"])
                    .await
                    .unwrap()
                    .len(),
                2
            );
        });
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
    fn sync_reranker_adapter_sorts_descending_for_trait_contract() {
        run_test_with_cx(|cx| async move {
            let adapter = SyncRerankerAdapter(UnsortedSyncReranker);
            let docs = vec![
                RerankDocument {
                    doc_id: "doc-a".into(),
                    text: "alpha".to_owned(),
                },
                RerankDocument {
                    doc_id: "doc-b".into(),
                    text: "beta".to_owned(),
                },
                RerankDocument {
                    doc_id: "doc-c".into(),
                    text: "gamma".to_owned(),
                },
            ];
            let scores = adapter
                .rerank(&cx, "query", &docs)
                .await
                .expect("adapter rerank should succeed");
            let ids = scores
                .iter()
                .map(|score| score.doc_id.as_str())
                .collect::<Vec<_>>();
            assert_eq!(ids, vec!["doc-b", "doc-a", "doc-c"]);
        });
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
    fn l2_normalize_in_place_matches_allocating() {
        // The in-place variant must be bit-identical to the allocating one for
        // every input, including zero / near-zero / non-finite norm cases.
        let cases: &[Vec<f32>] = &[
            vec![],
            vec![0.0, 0.0, 0.0],
            vec![3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![-1.5, 0.25, 1e-3, 7.0, -7.0],
            vec![1e-30, 1e-30],       // near-zero norm → zero vector
            vec![f32::MAX, f32::MAX], // non-finite norm_sq → zero vector
        ];
        for v in cases {
            let allocating = l2_normalize(v);
            let mut in_place = v.clone();
            l2_normalize_in_place(&mut in_place);
            assert_eq!(in_place, allocating, "input={v:?}");
        }
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
