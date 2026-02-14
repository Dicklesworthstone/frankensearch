//! `FlashRank` cross-encoder reranking for frankensearch.
//!
//! Provides the `FlashRankReranker` implementation of the `Reranker` trait,
//! using ONNX Runtime for cross-encoder scoring with sigmoid activation.
//! Gracefully falls back to original scores if the model is unavailable.
