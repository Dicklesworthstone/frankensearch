//! RRF fusion, score blending, and two-tier progressive search for frankensearch.
//!
//! This crate provides:
//! - **RRF**: Reciprocal Rank Fusion (K=60) with 4-level tie-breaking.
//! - **Blending**: Two-tier score blending (0.7 quality / 0.3 fast).
//! - **`TwoTierSearcher`**: Progressive iterator orchestrator yielding `SearchPhase` results.
//! - **Query classification**: Adaptive candidate budgets per query class.
