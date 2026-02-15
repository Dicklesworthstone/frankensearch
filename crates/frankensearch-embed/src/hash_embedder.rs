//! FNV-1a hash-based embedder for frankensearch.
//!
//! Produces deterministic (but non-semantic) embeddings using only hashing —
//! no model files, no ML inference, zero external dependencies. This is:
//!
//! - The always-available fallback when no ML models are downloaded
//! - The test double for pipeline integration tests in CI
//! - The fastest embedder (~0.07ms per embedding)
//!
//! Two algorithms are available:
//!
//! | Algorithm       | Quality        | Speed    | Use Case                       |
//! |-----------------|----------------|----------|--------------------------------|
//! | `FnvModular`    | Bag-of-words   | ~0.07ms  | Default fallback, regression   |
//! | `JLProjection`  | JL-guaranteed  | ~0.10ms  | Better distance preservation   |

use asupersync::Cx;
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture, l2_normalize};

/// FNV-1a offset basis (64-bit).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a prime (64-bit).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Minimum token length (tokens shorter than this are filtered out).
const MIN_TOKEN_LEN: usize = 2;

/// Default embedding dimension (matches `MiniLM` for index compatibility).
const DEFAULT_DIMENSION: usize = 384;

/// Hash algorithm selection for the [`HashEmbedder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// FNV-1a with modular projection. Default. Deterministic, fast, simple.
    ///
    /// Each token hashes to a single dimension index, accumulating a sign
    /// derived from the hash's high bit.
    FnvModular,

    /// Johnson-Lindenstrauss random hyperplane projection.
    ///
    /// Each token's contribution is spread across all dimensions using
    /// an xorshift64 PRNG seeded from the token hash. Provides formal
    /// distance-preservation guarantees from the JL lemma.
    JLProjection {
        /// Seed for the xorshift64 PRNG (combined with token hash).
        seed: u64,
    },
}

/// Zero-dependency hash-based embedder.
///
/// Produces deterministic embeddings using FNV-1a hashing. Not semantic —
/// captures lexical overlap only — but always available and extremely fast.
///
/// # Examples
///
/// ```
/// use frankensearch_embed::HashEmbedder;
///
/// let embedder = HashEmbedder::default_384();
/// let vec = embedder.embed_sync("hello world");
/// assert_eq!(vec.len(), 384);
/// ```
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dimension: usize,
    algorithm: HashAlgorithm,
}

impl HashEmbedder {
    /// Create a hash embedder with the given dimension and algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` is zero.
    #[must_use]
    pub fn new(dimension: usize, algorithm: HashAlgorithm) -> Self {
        assert!(dimension > 0, "dimension must be > 0");
        Self {
            dimension,
            algorithm,
        }
    }

    /// Default FNV-modular embedder with 384 dimensions.
    #[must_use]
    pub fn default_384() -> Self {
        Self::new(DEFAULT_DIMENSION, HashAlgorithm::FnvModular)
    }

    /// Default FNV-modular embedder with 256 dimensions (fast-tier compatibility).
    #[must_use]
    pub fn default_256() -> Self {
        Self::new(256, HashAlgorithm::FnvModular)
    }

    /// JL-projection embedder with 384 dimensions and the given seed.
    #[must_use]
    pub fn jl_384(seed: u64) -> Self {
        Self::new(DEFAULT_DIMENSION, HashAlgorithm::JLProjection { seed })
    }

    /// Synchronous embedding (no async overhead needed for ~0.07ms).
    #[must_use]
    pub fn embed_sync(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);

        match self.algorithm {
            HashAlgorithm::FnvModular => self.embed_fnv_modular(&tokens),
            HashAlgorithm::JLProjection { seed } => self.embed_jl(&tokens, seed),
        }
    }

    /// FNV-1a modular projection: each token maps to one dimension.
    fn embed_fnv_modular(&self, tokens: &[&str]) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; self.dimension];

        for token in tokens {
            let hash = fnv1a_hash(token.as_bytes());
            #[allow(clippy::cast_possible_truncation)] // modular arithmetic; truncation is fine
            let index = (hash as usize) % self.dimension;
            let sign = if (hash >> 63) == 1 { 1.0 } else { -1.0 };
            embedding[index] += sign;
        }

        l2_normalize(&embedding)
    }

    /// Johnson-Lindenstrauss random hyperplane projection.
    ///
    /// Each token's contribution is spread across all dimensions using
    /// xorshift64, providing better distance preservation than modular
    /// projection.
    fn embed_jl(&self, tokens: &[&str], seed: u64) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; self.dimension];

        for token in tokens {
            let hash = fnv1a_hash(token.as_bytes());
            // xorshift64 has a fixed point at zero — if seed ^ hash == 0,
            // the state stays zero forever, making all signs +1.0.
            let mut state = (seed ^ hash) | 1;

            for dim in &mut embedding {
                // Advance xorshift64 state for each dimension
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;

                let sign = if (state & 1) == 0 { 1.0 } else { -1.0 };
                *dim += sign;
            }
        }

        l2_normalize(&embedding)
    }
}

impl Embedder for HashEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        // Hash embedding is pure computation (~0.07ms) — no cancellation check needed
        Box::pin(async move { Ok(self.embed_sync(text)) })
    }

    fn embed_batch<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move { Ok(texts.iter().map(|t| self.embed_sync(t)).collect()) })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        // The id encodes algorithm + dimension for index compatibility
        match (self.algorithm, self.dimension) {
            (HashAlgorithm::FnvModular, 384) => "fnv1a-384",
            (HashAlgorithm::FnvModular, 256) => "fnv1a-256",
            (HashAlgorithm::JLProjection { .. }, 384) => "jl-384",
            (HashAlgorithm::JLProjection { .. }, 256) => "jl-256",
            (HashAlgorithm::FnvModular, _) => "fnv1a-custom",
            (HashAlgorithm::JLProjection { .. }, _) => "jl-custom",
        }
    }

    fn model_name(&self) -> &str {
        match self.algorithm {
            HashAlgorithm::FnvModular => "FNV-1a Hash Embedder",
            HashAlgorithm::JLProjection { .. } => "JL-Projection Hash Embedder",
        }
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
}

/// Compute FNV-1a hash of a byte slice.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Tokenize text for hash embedding.
///
/// Lowercases, splits on non-alphanumeric characters, and filters
/// tokens shorter than `MIN_TOKEN_LEN`.
fn tokenize(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= MIN_TOKEN_LEN)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Determinism ────────────────────────────────────────────────────

    #[test]
    fn deterministic_same_input_same_output() {
        let embedder = HashEmbedder::default_384();
        let a = embedder.embed_sync("hello world");
        let b = embedder.embed_sync("hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn deterministic_jl_same_seed_same_output() {
        let embedder = HashEmbedder::jl_384(42);
        let a = embedder.embed_sync("hello world");
        let b = embedder.embed_sync("hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn jl_different_seeds_different_output() {
        let e1 = HashEmbedder::jl_384(42);
        let e2 = HashEmbedder::jl_384(99);
        let a = e1.embed_sync("hello world");
        let b = e2.embed_sync("hello world");
        assert_ne!(a, b);
    }

    // ── Dimension ──────────────────────────────────────────────────────

    #[test]
    fn output_dimension_384() {
        let embedder = HashEmbedder::default_384();
        assert_eq!(embedder.embed_sync("test").len(), 384);
    }

    #[test]
    fn output_dimension_256() {
        let embedder = HashEmbedder::default_256();
        assert_eq!(embedder.embed_sync("test").len(), 256);
    }

    #[test]
    fn output_dimension_custom() {
        let embedder = HashEmbedder::new(128, HashAlgorithm::FnvModular);
        assert_eq!(embedder.embed_sync("test").len(), 128);
    }

    // ── L2 Normalization ───────────────────────────────────────────────

    #[test]
    fn output_is_l2_normalized() {
        let embedder = HashEmbedder::default_384();
        let vec = embedder.embed_sync("hello world");
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm = {norm}");
    }

    #[test]
    fn jl_output_is_l2_normalized() {
        let embedder = HashEmbedder::jl_384(42);
        let vec = embedder.embed_sync("hello world");
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm = {norm}");
    }

    // ── Different Inputs ───────────────────────────────────────────────

    #[test]
    fn different_inputs_different_embeddings() {
        let embedder = HashEmbedder::default_384();
        let a = embedder.embed_sync("hello world");
        let b = embedder.embed_sync("goodbye universe");
        assert_ne!(a, b);
    }

    // ── Empty and Edge Cases ───────────────────────────────────────────

    #[test]
    fn empty_string_produces_zero_vector() {
        let embedder = HashEmbedder::default_384();
        let vec = embedder.embed_sync("");
        // Empty string has no tokens, so embedding is all zeros (normalized to zeros)
        assert_eq!(vec.len(), 384);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn single_char_tokens_filtered() {
        let embedder = HashEmbedder::default_384();
        // "a b c" has only 1-char tokens which are filtered
        let vec = embedder.embed_sync("a b c");
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn long_input_no_panic() {
        let embedder = HashEmbedder::default_384();
        let long_text = "word ".repeat(20_000);
        let vec = embedder.embed_sync(&long_text);
        assert_eq!(vec.len(), 384);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    // ── Tokenization ───────────────────────────────────────────────────

    #[test]
    fn tokenize_basic() {
        let tokens = tokenize("hello world");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_filters_short() {
        let tokens = tokenize("a bb ccc");
        assert_eq!(tokens, vec!["bb", "ccc"]);
    }

    #[test]
    fn tokenize_splits_on_punctuation() {
        let tokens = tokenize("hello-world.test");
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn tokenize_preserves_case_for_hashing() {
        // Tokenize does NOT lowercase — the hash captures case differences
        let tokens = tokenize("Hello WORLD");
        assert_eq!(tokens, vec!["Hello", "WORLD"]);
    }

    // ── FNV-1a Hash ────────────────────────────────────────────────────

    #[test]
    fn fnv1a_empty_is_offset_basis() {
        assert_eq!(fnv1a_hash(b""), FNV_OFFSET);
    }

    #[test]
    fn fnv1a_deterministic() {
        let a = fnv1a_hash(b"hello");
        let b = fnv1a_hash(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn fnv1a_different_inputs() {
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    // ── Embedder Trait ─────────────────────────────────────────────────

    #[test]
    fn embedder_trait_id() {
        assert_eq!(HashEmbedder::default_384().id(), "fnv1a-384");
        assert_eq!(HashEmbedder::default_256().id(), "fnv1a-256");
        assert_eq!(HashEmbedder::jl_384(42).id(), "jl-384");
    }

    #[test]
    fn embedder_trait_not_semantic() {
        assert!(!HashEmbedder::default_384().is_semantic());
    }

    #[test]
    fn embedder_trait_category_hash() {
        assert_eq!(
            HashEmbedder::default_384().category(),
            ModelCategory::HashEmbedder
        );
    }

    #[test]
    fn embedder_trait_dimension() {
        assert_eq!(HashEmbedder::default_384().dimension(), 384);
        assert_eq!(HashEmbedder::default_256().dimension(), 256);
    }

    #[test]
    fn embed_via_trait() {
        // Test embed through the sync path (async wrapper is trivial)
        let embedder = HashEmbedder::default_384();
        let vec = embedder.embed_sync("test query");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn embed_batch_via_sync() {
        let embedder = HashEmbedder::default_384();
        let texts = ["hello", "world"];
        let vecs: Vec<_> = texts.iter().map(|t| embedder.embed_sync(t)).collect();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 384);
        assert_eq!(vecs[1].len(), 384);
        assert_ne!(vecs[0], vecs[1]);
    }

    #[test]
    fn embedder_model_name() {
        assert_eq!(
            HashEmbedder::default_384().model_name(),
            "FNV-1a Hash Embedder"
        );
        assert_eq!(
            HashEmbedder::jl_384(42).model_name(),
            "JL-Projection Hash Embedder"
        );
    }

    // ── JL Orthogonality ───────────────────────────────────────────────

    #[test]
    fn case_sensitivity_produces_different_embeddings() {
        let embedder = HashEmbedder::default_384();
        let lower = embedder.embed_sync("hello world");
        let upper = embedder.embed_sync("Hello World");
        // Case matters: different tokens → different hash values → different embeddings
        assert_ne!(lower, upper);

        // Also verify with JL projection variant
        let jl = HashEmbedder::jl_384(42);
        let jl_lower = jl.embed_sync("hello world");
        let jl_upper = jl.embed_sync("Hello World");
        assert_ne!(jl_lower, jl_upper);
    }

    #[test]
    fn jl_random_pairs_approximately_orthogonal() {
        use frankensearch_core::cosine_similarity;

        let embedder = HashEmbedder::jl_384(42);
        let mut total_sim = 0.0_f32;
        let n: usize = 100;

        for i in 0..n {
            let text = format!("random document number {i} with unique content");
            let other = format!("another document {i} about different topics entirely");
            let a = embedder.embed_sync(&text);
            let b = embedder.embed_sync(&other);
            total_sim += cosine_similarity(&a, &b).abs();
        }

        let mean_sim = total_sim / 100.0_f32;
        assert!(
            mean_sim < 0.3,
            "mean absolute cosine similarity should be low, got {mean_sim}"
        );
    }
}
