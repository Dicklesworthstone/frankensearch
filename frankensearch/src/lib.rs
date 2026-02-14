//! frankensearch: Two-tier hybrid search for Rust.
//!
//! Combines lexical (Tantivy BM25) and semantic (vector cosine similarity) search
//! via Reciprocal Rank Fusion, with a two-tier progressive model: fast results in
//! <15ms, quality-refined results in ~150ms.
//!
//! # Feature Flags
//!
//! - `hash` (default): FNV-1a hash embedder (zero dependencies).
//! - `model2vec`: potion-128M static embedder (fast tier, ~0.57ms).
//! - `fastembed`: MiniLM-L6-v2 ONNX embedder (quality tier, ~128ms).
//! - `lexical`: Tantivy BM25 full-text search.
//! - `rerank`: `FlashRank` cross-encoder reranking.
//! - `ann`: HNSW approximate nearest neighbor index.
//! - `download`: Model auto-download from `HuggingFace` via asupersync HTTP.
//! - `semantic`: All embedding models (hash + model2vec + fastembed).
//! - `hybrid`: Semantic + lexical + RRF fusion.
//! - `full`: Everything enabled.

pub use frankensearch_core as core;
pub use frankensearch_embed as embed;
pub use frankensearch_fusion as fusion;
pub use frankensearch_index as index;

#[cfg(feature = "lexical")]
pub use frankensearch_lexical as lexical;

#[cfg(feature = "rerank")]
pub use frankensearch_rerank as rerank;
