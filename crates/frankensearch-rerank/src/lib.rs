//! Cross-encoder reranking for frankensearch.
//!
//! The pure-Rust [`NativeReranker`] (feature `native`) runs a BERT
//! cross-encoder forward pass on frankentorch tensors — no ONNX / `ort` /
//! native C++ dependency — producing `sigmoid(logit)` relevance scores in
//! `[0, 1]`. An optional `FastEmbed` backend is available behind
//! `fastembed-reranker`.
//!
//! ```text
//! (query, document) → tokenize → frankentorch forward → logit → sigmoid → score ∈ [0, 1]
//! ```

pub mod pipeline;

#[cfg(feature = "fastembed-reranker")]
pub mod fastembed_reranker;

pub use pipeline::{
    DEFAULT_MIN_CANDIDATES, DEFAULT_RRF_COMBINE_K, DEFAULT_TOP_K_RERANK, RerankCombine,
    rerank_step, rerank_step_with_combine,
};

#[cfg(feature = "fastembed-reranker")]
pub use fastembed_reranker::FastEmbedReranker;

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "native")]
pub mod native_embedder;

#[cfg(feature = "native")]
pub use native::NativeReranker;

#[cfg(feature = "native")]
pub use native_embedder::{NativeEmbedder, NativeEmbeddingModel};

/// Default model directory name for the cross-encoder reranker.
pub const DEFAULT_MODEL_NAME: &str = "rerank-default-v1";

/// Default maximum input token length for cross-encoder query/document pairs.
pub const DEFAULT_MAX_LENGTH: usize = 512;

/// Convert a token-id slice to `i64`, keeping at most `max_length` ids.
///
/// Truncating with `take(max_length)` **before** the `i64` conversion + collect
/// avoids materializing the discarded tail: for a document that tokenizes to far
/// more than `max_length` tokens (common when reranking full document bodies), this
/// is `O(max_length)` instead of `O(total_tokens)`. Byte-identical to
/// `iter().map(i64::from).collect()` followed by `truncate(max_length)` — both yield
/// the first `min(len, max_length)` ids.
#[must_use]
pub fn ids_to_truncated_i64(ids: &[u32], max_length: usize) -> Vec<i64> {
    ids.iter()
        .take(max_length)
        .map(|&id| i64::from(id))
        .collect()
}
