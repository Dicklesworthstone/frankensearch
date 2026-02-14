//! Embedder implementations for the frankensearch hybrid search library.
//!
//! Provides three tiers of text embedding:
//! - **Hash** (`hash` feature, default): FNV-1a hash embedder, zero dependencies, always available.
//! - **`Model2Vec`** (`model2vec` feature): potion-128M static embedder, fast tier (~0.57ms).
//! - **`FastEmbed`** (`fastembed` feature): MiniLM-L6-v2 ONNX embedder, quality tier (~128ms).
//!
//! The `EmbedderStack` auto-detection probes for available models and configures
//! the best fast+quality pair automatically.
