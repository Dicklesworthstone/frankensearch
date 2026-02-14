//! FSVI vector index, SIMD dot product, and top-k search for frankensearch.
//!
//! This crate provides:
//! - **FSVI binary format**: Custom vector index format with f16 quantization and memory-mapping.
//! - **SIMD dot product**: `wide::f32x8` portable SIMD across x86 SSE2/AVX2 and ARM NEON.
//! - **Top-k search**: Brute-force search with `BinaryHeap` guard pattern, Rayon parallel.
//! - **HNSW ANN** (`ann` feature): Approximate nearest neighbor index for large-scale search.
