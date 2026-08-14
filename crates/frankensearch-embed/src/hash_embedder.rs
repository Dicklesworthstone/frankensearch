//! FNV-1a hash-based embedder for frankensearch.
//!
//! Produces deterministic (but non-semantic) embeddings using only hashing —
//! no model files and no ML inference. This is:
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
use frankensearch_core::generation::{
    EMBEDDING_INPUT_CONTRACT_SCHEMA_V1, EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
    EMBEDDING_SPACE_IDENTITY_SCHEMA_V1, EmbeddingIdentityBundleV1, EmbeddingInputContractV1,
    EmbeddingProducerAttestationV1, EmbeddingSpaceIdentityV1, EmbeddingSpaceKindV1,
    GoldenVectorCertificateV1, HashControlProfileV1, QuantizationFormat,
    VECTOR_STORAGE_IDENTITY_SCHEMA_V1, VectorStorageIdentityV1,
};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture, l2_normalize_in_place};
use frankensearch_core::{SearchError, SearchResult};
use rayon::prelude::*;

/// FNV-1a offset basis (64-bit).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a prime (64-bit).
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Minimum UTF-8 byte length (tokens shorter than this are filtered out).
const MIN_TOKEN_LEN: usize = 2;

/// Default embedding dimension (the same width as `MiniLM`, but a distinct
/// non-semantic vector space).
const DEFAULT_DIMENSION: usize = 384;

/// Batch size where parallel dispatch amortizes Rayon scheduling for FNV-384.
/// Smaller batches stay serial to preserve the latency of the default size (64).
const PARALLEL_BATCH_MIN: usize = 256;

const HASH_CONFORMANCE_TEXTS_V1: [&str; 4] = [
    "",
    "Frankensearch identity",
    "Case CASE case",
    "unicode café 東京",
];

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

fn hash_identity(dimension: usize, algorithm: HashAlgorithm) -> EmbeddingIdentityBundleV1 {
    let maximum = usize::try_from(u32::MAX).unwrap_or(usize::MAX);
    assert!(
        dimension <= maximum,
        "dimension must fit the u32 identity schema"
    );
    let dimension_u32 = u32::try_from(dimension).unwrap_or(u32::MAX);
    let (logical_model_id, algorithm_name, algorithm_revision, seed, feature_rules, signing_rules) =
        match algorithm {
            HashAlgorithm::FnvModular => (
                "fnv1a-hash-control",
                "fnv1a-modular-signed-bucket",
                "v1",
                0,
                "for each token hash=fnv1a64(UTF-8 bytes); bucket=hash modulo dimension; add one contribution",
                "FNV hash bit 63 set => +1f32; clear => -1f32",
            ),
            HashAlgorithm::JLProjection { seed } => (
                "jl-hash-control",
                "fnv1a-xorshift64-jl-projection",
                "v1",
                seed,
                "for each token state=(seed xor fnv1a64(UTF-8 bytes)) or 1; per output dimension advance xorshift64 by left-13,right-7,left-17; add one contribution",
                "advanced xorshift64 state bit 0 clear => +1f32; set => -1f32",
            ),
        };
    let profile = HashControlProfileV1 {
        algorithm: algorithm_name.to_owned(),
        algorithm_revision: algorithm_revision.to_owned(),
        seed,
        feature_rules: feature_rules.to_owned(),
        tokenization_rules:
            "split unicode alphanumeric runs; preserve case; drop tokens shorter than 2 UTF-8 bytes"
                .to_owned(),
        signing_rules: signing_rules.to_owned(),
        normalization_rules: "l2-f32-zero-on-degenerate-v1".to_owned(),
    };
    let profile_fingerprint = profile.fingerprint();
    let input = EmbeddingInputContractV1 {
        schema_version: EMBEDDING_INPUT_CONTRACT_SCHEMA_V1,
        canonicalization: "caller-utf8-as-is-v1".to_owned(),
        content_selection: "single-caller-supplied-text-v1".to_owned(),
        chunking: "none-at-embedder-boundary-v1".to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        doc_id_semantics: "vector-independent-of-document-id-v1".to_owned(),
    };
    let space = EmbeddingSpaceIdentityV1 {
        schema_version: EMBEDDING_SPACE_IDENTITY_SCHEMA_V1,
        logical_model_id: logical_model_id.to_owned(),
        immutable_revision: format!("{algorithm_revision}:dimension={dimension_u32}"),
        kind: EmbeddingSpaceKindV1::HashControl,
        artifact_manifest_fingerprint: profile_fingerprint.clone(),
        artifacts: Vec::new(),
        tokenizer_fingerprint: profile_fingerprint.clone(),
        vocabulary_fingerprint: profile_fingerprint.clone(),
        model_config_fingerprint: profile_fingerprint.clone(),
        model_preprocessing: "none".to_owned(),
        sequence_policy: "unbounded-caller-string".to_owned(),
        query_instruction: String::new(),
        document_instruction: String::new(),
        pooling: feature_rules.to_owned(),
        output_normalization: "l2-f32-zero-on-degenerate-v1".to_owned(),
        dimension: dimension_u32,
        input_contract_fingerprint: input.fingerprint(),
        hash_control: Some(profile),
        projection: None,
    };
    let golden_vectors = hash_golden_certificate(dimension, algorithm);
    let producer = EmbeddingProducerAttestationV1 {
        schema_version: EMBEDDING_PRODUCER_ATTESTATION_SCHEMA_V1,
        backend: "frankensearch-hash-native".to_owned(),
        implementation_revision: env!("CARGO_PKG_VERSION").to_owned(),
        protocol_revision: "in-process-hash-v1".to_owned(),
        numeric_profile: "deterministic-f32-bit-exact-v1".to_owned(),
        provenance_manifest_fingerprint: profile_fingerprint,
        space_fingerprint: space.fingerprint(),
        golden_vectors,
    };
    let identity = EmbeddingIdentityBundleV1 {
        space,
        producer,
        input,
        storage: VectorStorageIdentityV1 {
            schema_version: VECTOR_STORAGE_IDENTITY_SCHEMA_V1,
            format: "in-memory-f32-v1".to_owned(),
            quantization: QuantizationFormat::F32,
            endianness: "native-f32-values".to_owned(),
            vector_normalization: "l2-f32-zero-on-degenerate-v1".to_owned(),
            dimension: dimension_u32,
        },
    };
    debug_assert!(identity.validate().is_ok());
    identity
}

fn hash_golden_certificate(
    dimension: usize,
    algorithm: HashAlgorithm,
) -> GoldenVectorCertificateV1 {
    let placeholder = EmbeddingIdentityBundleV1::explicit_test_model("hash-golden-probe", 1);
    let probe = HashEmbedder {
        dimension,
        algorithm,
        identity: placeholder,
    };
    let vectors = HASH_CONFORMANCE_TEXTS_V1
        .iter()
        .map(|text| probe.embed_sync(text))
        .collect::<Vec<_>>();
    GoldenVectorCertificateV1::from_exact_f32(&HASH_CONFORMANCE_TEXTS_V1, &vectors)
        .expect("static hash conformance corpus and vectors must be valid")
}

/// Model-free hash-based embedder.
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
    identity: EmbeddingIdentityBundleV1,
}

impl HashEmbedder {
    /// Create a hash embedder with the given dimension and algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` is zero or exceeds the identity schema's u32 limit.
    #[must_use]
    pub fn new(dimension: usize, algorithm: HashAlgorithm) -> Self {
        assert!(dimension > 0, "dimension must be > 0");
        let identity = hash_identity(dimension, algorithm);
        Self {
            dimension,
            algorithm,
            identity,
        }
    }

    /// Default FNV-modular embedder with 384 dimensions.
    #[must_use]
    pub fn default_384() -> Self {
        Self::new(DEFAULT_DIMENSION, HashAlgorithm::FnvModular)
    }

    /// Default FNV-modular embedder with 256 dimensions.
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
        // Each branch consumes the token iterator exactly once, so it is built
        // lazily per branch — no intermediate `Vec<&str>` allocation.
        match self.algorithm {
            HashAlgorithm::FnvModular => self.embed_fnv_modular(tokenize(text)),
            HashAlgorithm::JLProjection { seed } => self.embed_jl(tokenize(text), seed),
        }
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        if texts.len() >= PARALLEL_BATCH_MIN {
            texts.par_iter().map(|text| self.embed_sync(text)).collect()
        } else {
            texts.iter().map(|text| self.embed_sync(text)).collect()
        }
    }

    /// FNV-1a modular projection: each token maps to one dimension.
    fn embed_fnv_modular<'a>(&self, tokens: impl Iterator<Item = &'a str>) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; self.dimension];
        let dimension = u64::try_from(self.dimension)
            .expect("hash-embedder dimension must fit the u64 modulus");

        for token in tokens {
            let hash = fnv1a_hash(token.as_bytes());
            let index = usize::try_from(hash % dimension)
                .expect("modular bucket must fit the configured usize dimension");
            let sign = if (hash >> 63) == 1 { 1.0 } else { -1.0 };
            embedding[index] += sign;
        }

        // Normalize in place: reuse the accumulator we already own instead of
        // allocating a second dimension-sized vector via `l2_normalize`.
        l2_normalize_in_place(&mut embedding);
        embedding
    }

    /// Johnson-Lindenstrauss random hyperplane projection.
    ///
    /// Each token's contribution is spread across all dimensions using
    /// xorshift64, providing better distance preservation than modular
    /// projection.
    ///
    /// This is the compute-bound embedder (O(tokens·dim) xorshift). A single
    /// token's xorshift chain is **latency-bound**: every step's three
    /// shift→xor operations depend on the previous step, so one chain cannot
    /// keep the CPU's shift/ALU ports busy. But each token seeds an
    /// *independent* chain, so we advance [`JL_LANES`] token chains together
    /// per dimension to expose instruction-level parallelism that hides that
    /// latency. The result is **bit-identical** to advancing the chains one at
    /// a time: the per-dimension accumulator only ever holds an exact
    /// integer-valued `f32` (a sum of ±1 contributions, `|value| ≤ token count
    /// ≪ 2^24`), so f32 addition is exact and reordering the lane
    /// contributions cannot change a single bit of the output.
    fn embed_jl<'a>(&self, tokens: impl Iterator<Item = &'a str>, seed: u64) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; self.dimension];

        // Buffer token chain seeds in lanes; flush a full group through the
        // interleaved inner loop, then drain the (< JL_LANES) tail one chain
        // at a time. Token order across the accumulator is preserved.
        let mut states = [0_u64; JL_LANES];
        let mut filled = 0_usize;

        for token in tokens {
            let hash = fnv1a_hash(token.as_bytes());
            // xorshift64 has a fixed point at zero — if seed ^ hash == 0,
            // the state would stay zero forever. The v1 profile's explicit
            // `| 1` seed-mixing rule keeps every chain live.
            states[filled] = (seed ^ hash) | 1;
            filled += 1;
            if filled == JL_LANES {
                jl_accumulate_lanes8(&mut embedding, &states);
                filled = 0;
            }
        }
        for &state in &states[..filled] {
            jl_accumulate_one(&mut embedding, state);
        }

        // Normalize in place (see `embed_fnv_modular`): one fewer allocation.
        l2_normalize_in_place(&mut embedding);
        embedding
    }
}

/// Number of independent xorshift64 token-chains advanced together in the JL
/// inner loop (instruction-level parallelism over a latency-bound recurrence).
const JL_LANES: usize = 8;

/// Advance one token's xorshift64 chain over every dimension, adding its ±1
/// sign per dimension. This is the original single-chain inner loop, used for
/// the (< [`JL_LANES`]) tail. Kept identical so the tail is trivially the same
/// as the pre-ILP scalar path.
#[inline]
fn jl_accumulate_one(embedding: &mut [f32], mut state: u64) {
    for dim in embedding {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *dim += if (state & 1) == 0 { 1.0 } else { -1.0 };
    }
}

/// Advance 4 independent token chains together (one `__m256i` of u64 lanes),
/// folding their per-dimension signs into the accumulator. **Superseded in
/// production by the 8-lane [`jl_accumulate_lanes8`]** (2 `__m256i` → 2-way ILP
/// that hides the xorshift latency, ~1.76× faster); kept as the bench A/B
/// baseline. Runtime-dispatches to an AVX2 kernel; the scalar-ILP version is the
/// portable fallback. Both are bit-identical: the per-dimension contribution is a
/// sum of ±1 lanes (an exact small integer added to an exact-integer accumulator),
/// so neither the SIMD lane grouping nor the `4 - 2·popcount` reformulation can
/// change a single output bit.
#[doc(hidden)]
#[inline]
pub fn jl_accumulate_lanes(embedding: &mut [f32], states: &[u64; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            unsafe {
                jl_accumulate_lanes_avx2(embedding, states);
            }
            return;
        }
    }
    jl_accumulate_lanes_scalar(embedding, states);
}

/// Portable scalar-ILP JL accumulate (the AVX2-dispatch fallback). 4 independent
/// xorshift64 chains pipelined; `#[doc(hidden)] pub` for the bench A/B.
#[doc(hidden)]
#[inline]
pub fn jl_accumulate_lanes_scalar(embedding: &mut [f32], states: &[u64; 4]) {
    let [mut s0, mut s1, mut s2, mut s3] = *states;
    for dim in embedding {
        s0 ^= s0 << 13;
        s0 ^= s0 >> 7;
        s0 ^= s0 << 17;
        s1 ^= s1 << 13;
        s1 ^= s1 >> 7;
        s1 ^= s1 << 17;
        s2 ^= s2 << 13;
        s2 ^= s2 >> 7;
        s2 ^= s2 << 17;
        s3 ^= s3 << 13;
        s3 ^= s3 >> 7;
        s3 ^= s3 << 17;
        let a0 = if (s0 & 1) == 0 { 1.0 } else { -1.0 };
        let a1 = if (s1 & 1) == 0 { 1.0 } else { -1.0 };
        let a2 = if (s2 & 1) == 0 { 1.0 } else { -1.0 };
        let a3 = if (s3 & 1) == 0 { 1.0 } else { -1.0 };
        // Sum of ±1 lanes is an exact small integer; adding it to the exact
        // integer accumulator is bit-identical to four sequential `+=`.
        *dim += a0 + a1 + a2 + a3;
    }
}

/// Hand-written AVX2 JL accumulate: the 4 xorshift64 chains live in one
/// `__m256i` (u64×4), so each chain step is a single vector `slli`/`srli`/`xor`
/// (3 steps = 6 vector ops vs 24 scalar). The per-dimension ±1 sum is recovered
/// branch-free: shift each lane's low bit into the sign position, `movemask_pd`
/// the 4 sign bits, and `4 - 2·popcount(mask)` is the exact integer
/// `a0+a1+a2+a3`. **Bit-identical** to the scalar kernel
/// (`jl_avx2_matches_scalar`); the xorshift is per-lane identical and the sum is
/// an exact integer either way.
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`jl_accumulate_lanes`]
/// guarantees it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
fn jl_accumulate_lanes_avx2(embedding: &mut [f32], states: &[u64; 4]) {
    use core::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_loadu_si256, _mm256_movemask_pd, _mm256_slli_epi64,
        _mm256_srli_epi64, _mm256_xor_si256,
    };
    // SAFETY: avx2 by contract; the load reads exactly the 4 u64 of `states`.
    unsafe {
        let mut s = _mm256_loadu_si256(states.as_ptr().cast::<__m256i>());
        for dim in embedding {
            s = _mm256_xor_si256(s, _mm256_slli_epi64::<13>(s));
            s = _mm256_xor_si256(s, _mm256_srli_epi64::<7>(s));
            s = _mm256_xor_si256(s, _mm256_slli_epi64::<17>(s));
            // Move each lane's low bit to its sign bit, then extract the 4 signs.
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_slli_epi64::<63>(s)));
            // odd lanes → -1, even → +1; Σ = (4-odd) - odd = 4 - 2·odd.
            #[allow(clippy::cast_possible_wrap)]
            let sum = (4 - 2 * mask.cast_unsigned().count_ones() as i32) as f32;
            *dim += sum;
        }
    }
}

/// 8-chain variant of [`jl_accumulate_lanes`]. The 4-lane AVX2 kernel is one
/// `__m256i` whose 3-step xorshift is a *dependency chain* → latency-bound; running
/// TWO independent `__m256i` (8 chains) exposes the 2-way ILP that hides that
/// latency. Bit-identical to the 4-lane / scalar path for the SAME tokens (the
/// per-dim contribution is a sum of ±1 — an exact integer independent of lane
/// grouping). Runtime-dispatched; `#[doc(hidden)] pub` for the bench A/B.
#[doc(hidden)]
#[inline]
pub fn jl_accumulate_lanes8(embedding: &mut [f32], states: &[u64; 8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            unsafe {
                jl_accumulate_lanes8_avx2(embedding, states);
            }
            return;
        }
    }
    jl_accumulate_lanes8_scalar(embedding, states);
}

/// Portable 8-chain scalar fallback for [`jl_accumulate_lanes8`].
#[doc(hidden)]
#[inline]
pub fn jl_accumulate_lanes8_scalar(embedding: &mut [f32], states: &[u64; 8]) {
    let mut s = *states;
    for dim in embedding {
        let mut odd = 0_i32;
        for st in &mut s {
            *st ^= *st << 13;
            *st ^= *st >> 7;
            *st ^= *st << 17;
            odd += i32::from((*st & 1) == 1);
        }
        #[allow(clippy::cast_possible_wrap)]
        let sum = (8 - 2 * odd) as f32;
        *dim += sum;
    }
}

/// AVX2 8-chain JL accumulate — TWO `__m256i` (u64×4 each), xorshift steps
/// interleaved so the CPU pipelines the two independent chains (2-way ILP over the
/// latency-bound recurrence). Per-dim ±1 sum = `8 - 2·(popcount(maskA)+popcount(maskB))`.
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`jl_accumulate_lanes8`]
/// guarantees it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
#[allow(clippy::cast_ptr_alignment)]
fn jl_accumulate_lanes8_avx2(embedding: &mut [f32], states: &[u64; 8]) {
    use core::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_loadu_si256, _mm256_movemask_pd, _mm256_slli_epi64,
        _mm256_srli_epi64, _mm256_xor_si256,
    };
    // SAFETY: avx2 by contract; the two loads read the 8 u64 of `states` (chains
    // 0-3 and 4-7); `states` is exactly 8 u64 = 64 bytes.
    unsafe {
        let mut a = _mm256_loadu_si256(states.as_ptr().cast::<__m256i>());
        let mut b = _mm256_loadu_si256(states.as_ptr().add(4).cast::<__m256i>());
        for dim in embedding {
            a = _mm256_xor_si256(a, _mm256_slli_epi64::<13>(a));
            b = _mm256_xor_si256(b, _mm256_slli_epi64::<13>(b));
            a = _mm256_xor_si256(a, _mm256_srli_epi64::<7>(a));
            b = _mm256_xor_si256(b, _mm256_srli_epi64::<7>(b));
            a = _mm256_xor_si256(a, _mm256_slli_epi64::<17>(a));
            b = _mm256_xor_si256(b, _mm256_slli_epi64::<17>(b));
            let ma = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_slli_epi64::<63>(a)));
            let mb = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_slli_epi64::<63>(b)));
            #[allow(clippy::cast_possible_wrap)]
            let sum = (8 - 2
                * (ma.cast_unsigned().count_ones() + mb.cast_unsigned().count_ones()) as i32)
                as f32;
            *dim += sum;
        }
    }
}

fn embed_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

impl Embedder for HashEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            embed_checkpoint(cx, "hash.embed")?;
            Ok(self.embed_sync(text))
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            embed_checkpoint(cx, "hash.embed_batch")?;
            Ok(self.embed_batch_sync(texts))
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        // The ID is operational metadata; the complete hash-control identity
        // establishes compatibility.
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
/// Splits on non-alphanumeric characters and filters
/// tokens shorter than `MIN_TOKEN_LEN`. Case is intentionally preserved.
fn tokenize(text: &str) -> Tokens<'_> {
    Tokens {
        text,
        offset: 0,
        unicode: false,
    }
}

struct Tokens<'a> {
    text: &'a str,
    offset: usize,
    unicode: bool,
}

impl<'a> Iterator for Tokens<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.unicode {
            return self.next_unicode();
        }
        self.next_ascii_or_promote()
    }
}

impl<'a> Tokens<'a> {
    fn next_ascii_or_promote(&mut self) -> Option<&'a str> {
        let bytes = self.text.as_bytes();
        while self.offset < bytes.len() {
            while self.offset < bytes.len() {
                let byte = bytes[self.offset];
                if byte >= 0x80 {
                    self.unicode = true;
                    return self.next_unicode();
                }
                if byte.is_ascii_alphanumeric() {
                    break;
                }
                self.offset += 1;
            }
            let start = self.offset;
            while self.offset < bytes.len() {
                let byte = bytes[self.offset];
                if byte >= 0x80 {
                    self.unicode = true;
                    self.offset = start;
                    return self.next_unicode();
                }
                if !byte.is_ascii_alphanumeric() {
                    break;
                }
                self.offset += 1;
            }
            if self.offset.saturating_sub(start) >= MIN_TOKEN_LEN {
                return Some(&self.text[start..self.offset]);
            }
        }
        None
    }

    fn next_unicode(&mut self) -> Option<&'a str> {
        while self.offset < self.text.len() {
            let Some((start_rel, _)) = self.text[self.offset..]
                .char_indices()
                .find(|(_, c)| c.is_alphanumeric())
            else {
                self.offset = self.text.len();
                return None;
            };
            let start = self.offset + start_rel;
            let mut end = self.text.len();
            for (idx, c) in self.text[start..].char_indices() {
                if !c.is_alphanumeric() {
                    end = start + idx;
                    break;
                }
            }
            self.offset = end;
            if end.saturating_sub(start) >= MIN_TOKEN_LEN {
                return Some(&self.text[start..end]);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_tokens(text: &str) -> Vec<&str> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|token| token.len() >= MIN_TOKEN_LEN)
            .collect()
    }

    #[test]
    fn tokenizer_matches_unicode_split_reference() {
        let cases = [
            "",
            "a an ant, fox_42 jumps-over C3PO",
            "snake_case HTTP/2 path/to/file.rs",
            "cafe\u{0301} café naïve 日本語 x",
            "a.b::c -- Δelta42",
        ];
        for case in cases {
            let actual: Vec<_> = tokenize(case).collect();
            assert_eq!(actual, reference_tokens(case), "input={case:?}");
        }
    }

    /// The AVX2 JL accumulate must be byte-for-byte identical to the scalar-ILP
    /// kernel across dim shapes + accumulated over many token-groups (exact-integer
    /// accumulator). Skips without AVX2.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn jl_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for &dim in &[1_usize, 7, 8, 16, 31, 64, 100, 256, 384, 385] {
            let mut e_scalar = vec![0.0_f32; dim];
            let mut e_avx2 = vec![0.0_f32; dim];
            // Accumulate many independent token-groups (incl. the zero-state edge).
            for g in 0..25 {
                let states = if g == 0 {
                    [1_u64; 4] // (seed^hash)|1 always odd, never 0
                } else {
                    [next() | 1, next() | 1, next() | 1, next() | 1]
                };
                jl_accumulate_lanes_scalar(&mut e_scalar, &states);
                // SAFETY: avx2 verified present above.
                #[allow(unsafe_code)]
                unsafe {
                    jl_accumulate_lanes_avx2(&mut e_avx2, &states);
                }
            }
            let sb: Vec<u32> = e_scalar.iter().map(|x| x.to_bits()).collect();
            let ab: Vec<u32> = e_avx2.iter().map(|x| x.to_bits()).collect();
            assert_eq!(sb, ab, "dim={dim}");
        }
    }

    /// The 8-chain kernels (AVX2 + scalar) must produce a byte-for-byte identical
    /// accumulated embedding to the 4-chain path for the SAME tokens — the per-dim
    /// ±1 sum is an exact integer independent of lane grouping. Skips without AVX2.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn jl_8lane_matches_4lane() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng | 1
        };
        for &dim in &[1_usize, 8, 31, 100, 384] {
            let states: Vec<u64> = (0..48).map(|_| next()).collect(); // 12×4 = 6×8, no tail
            let mut e4 = vec![0.0_f32; dim];
            let (chunks4, tail4) = states.as_chunks::<4>();
            assert!(tail4.is_empty());
            for g in chunks4 {
                jl_accumulate_lanes_scalar(&mut e4, g);
            }
            let mut e8_avx2 = vec![0.0_f32; dim];
            let mut e8_scalar = vec![0.0_f32; dim];
            let (chunks8, tail8) = states.as_chunks::<8>();
            assert!(tail8.is_empty());
            for g in chunks8 {
                jl_accumulate_lanes8(&mut e8_avx2, g);
                jl_accumulate_lanes8_scalar(&mut e8_scalar, g);
            }
            let b4: Vec<u32> = e4.iter().map(|x| x.to_bits()).collect();
            let ba: Vec<u32> = e8_avx2.iter().map(|x| x.to_bits()).collect();
            let bs: Vec<u32> = e8_scalar.iter().map(|x| x.to_bits()).collect();
            assert_eq!(b4, ba, "8-lane avx2 vs 4-lane, dim={dim}");
            assert_eq!(b4, bs, "8-lane scalar vs 4-lane, dim={dim}");
        }
    }

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

    /// Reference: the pre-ILP single-chain JL accumulation, used to prove the
    /// interleaved [`jl_accumulate_lanes`] path is bit-identical.
    fn embed_jl_scalar_reference(dimension: usize, seed: u64, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; dimension];
        for token in tokenize(text) {
            let hash = fnv1a_hash(token.as_bytes());
            let mut state = (seed ^ hash) | 1;
            for dim in &mut embedding {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *dim += if (state & 1) == 0 { 1.0 } else { -1.0 };
            }
        }
        l2_normalize_in_place(&mut embedding);
        embedding
    }

    #[test]
    fn jl_ilp_matches_scalar_reference_bit_identical() {
        // 13 tokens ≥2 chars → exercises three full JL_LANES groups (12) plus a
        // 1-token tail, covering both the interleaved loop and the drain path.
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa \
                    lambda mu nu";
        for &dim in &[256_usize, 384] {
            for &seed in &[42_u64, 0x9e37_79b9_7f4a_7c15] {
                let embedder = HashEmbedder::new(dim, HashAlgorithm::JLProjection { seed });
                let got = embedder.embed_sync(text);
                let want = embed_jl_scalar_reference(dim, seed, text);
                assert_eq!(
                    got, want,
                    "ILP JL must be bit-identical (dim={dim}, seed={seed})"
                );
            }
        }
    }

    #[test]
    fn jl_ilp_matches_scalar_across_token_counts() {
        // Token counts straddling the JL_LANES boundary (remainders 0..=4+).
        let embedder = HashEmbedder::jl_384(7);
        for n in 0..=9 {
            let text = (0..n)
                .map(|i| format!("tok{i:02}"))
                .collect::<Vec<_>>()
                .join(" ");
            let got = embedder.embed_sync(&text);
            let want = embed_jl_scalar_reference(384, 7, &text);
            assert_eq!(got, want, "ILP JL mismatch at token count {n}");
        }
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
        let tokens: Vec<&str> = tokenize("hello world").collect();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_filters_short() {
        let tokens: Vec<&str> = tokenize("a bb ccc").collect();
        assert_eq!(tokens, vec!["bb", "ccc"]);
    }

    #[test]
    fn tokenize_length_filter_is_exactly_utf8_bytes() {
        let tokens: Vec<&str> = tokenize("a é Δ 京").collect();
        assert_eq!(tokens, vec!["é", "Δ", "京"]);
    }

    #[test]
    fn tokenize_splits_on_punctuation() {
        let tokens: Vec<&str> = tokenize("hello-world.test").collect();
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn tokenize_preserves_case_for_hashing() {
        // Tokenize does NOT lowercase — the hash captures case differences
        let tokens: Vec<&str> = tokenize("Hello WORLD").collect();
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
    fn hash_control_identity_binds_the_actual_golden_vectors() {
        for embedder in [
            HashEmbedder::default_256(),
            HashEmbedder::jl_384(0x9e37_79b9_7f4a_7c15),
        ] {
            let vectors = embedder.embed_batch_sync(&HASH_CONFORMANCE_TEXTS_V1);
            embedder
                .identity()
                .unwrap()
                .producer
                .golden_vectors
                .verify_exact_f32(&HASH_CONFORMANCE_TEXTS_V1, &vectors)
                .unwrap();
            assert!(!embedder.is_semantic());
        }
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
    fn embed_batch_matches_serial_across_parallel_boundary() {
        let embedders = [HashEmbedder::default_384(), HashEmbedder::jl_384(42)];
        for embedder in &embedders {
            for &batch_size in &[0_usize, 1, PARALLEL_BATCH_MIN - 1, PARALLEL_BATCH_MIN, 257] {
                let docs = (0..batch_size)
                    .map(|index| format!("hello café world document {index}"))
                    .collect::<Vec<_>>();
                let texts = docs.iter().map(String::as_str).collect::<Vec<_>>();
                let serial = texts
                    .iter()
                    .map(|text| embedder.embed_sync(text))
                    .collect::<Vec<_>>();
                let batch = embedder.embed_batch_sync(&texts);
                assert_eq!(serial.len(), batch.len());
                for (serial_row, batch_row) in serial.iter().zip(&batch) {
                    assert!(
                        serial_row
                            .iter()
                            .zip(batch_row)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "batch mismatch at size {batch_size}"
                    );
                }
            }
        }
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
