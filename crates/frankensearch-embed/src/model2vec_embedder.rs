//! `Model2Vec` static token embedding for the fast tier.
//!
//! Wraps the exact manifest-registered potion-multilingual-128M artifact set.
//! It looks up pre-computed per-token embeddings and mean-pools them. No
//! transformer inference, no GPU needed.
//!
//! Performance: ~0.57ms per embedding (223x faster than `MiniLM-L6-v2`).
//!
//! Memory: ~32MB resident for a 32K-vocab × 256-dim model.
//!
//! Only two files are required:
//! - `tokenizer.json` (`HuggingFace` BPE tokenizer)
//! - `model.safetensors` (static embedding matrix)

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use asupersync::Cx;
use rayon::prelude::*;
use safetensors::Dtype;
use safetensors::tensor::Metadata;
use tokenizers::Tokenizer;
use tracing::instrument;

use crate::model_manifest::{
    MODEL2VEC_OUTPUT_NORMALIZATION_V1, MODEL2VEC_POOLING_V1, MODEL2VEC_PREPROCESSING_V1,
    MODEL2VEC_SEQUENCE_POLICY_V1, ModelArtifactManifestV1,
};
use crate::model_registry::{ensure_model_storage_layout, model_directory_variants};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};

/// Required files for a `Model2Vec` model.
const REQUIRED_FILES: [&str; 2] = ["tokenizer.json", "model.safetensors"];

/// Batch size at/above which `embed_batch` dispatches per-document embedding across
/// Rayon threads. Each `embed_sync` is ~0.57 ms of independent CPU work (tokenize →
/// static-row gather → mean-pool → normalize), so parallel dispatch amortizes Rayon
/// scheduling at a much smaller batch than the FNV hash embedder needs (256); smaller
/// batches stay serial to preserve single/few-doc latency.
const PARALLEL_BATCH_MIN: usize = 8;

/// Tensor name candidates, tried in order when discovering the embedding matrix.
const TENSOR_NAME_CANDIDATES: [&str; 5] =
    ["embeddings", "embedding", "word_embeddings", "embed", "emb"];

/// Default model name for the primary fast-tier model.
const DEFAULT_MODEL_NAME: &str = "potion-multilingual-128M";

/// Default `HuggingFace` model ID for the primary fast-tier model.
const DEFAULT_HF_ID: &str = "minishlab/potion-multilingual-128M";

/// Static token embedding model (`Model2Vec` / potion).
///
/// After construction, all fields are immutable — no `Mutex` needed.
/// The struct is `Send + Sync` by construction.
///
/// # Loading
///
/// ```rust,ignore
/// let embedder = Model2VecEmbedder::load("/path/to/model")?;
/// let embedding = embedder.embed_sync("hello world");
/// assert_eq!(embedding.len(), 256);
/// ```
pub struct Model2VecEmbedder {
    /// `HuggingFace` BPE tokenizer.
    tokenizer: Tokenizer,
    /// Flat embedding matrix: `embeddings[token_id * dim .. (token_id + 1) * dim]`.
    embeddings: Vec<f32>,
    /// Output dimensionality.
    dimensions: usize,
    /// Vocabulary size (number of rows in the embedding matrix).
    vocab_size: usize,
    /// Human-readable model name.
    name: String,
    /// Directory the model was loaded from.
    model_dir: PathBuf,
    /// Complete identity derived from the verified frozen manifest.
    identity: EmbeddingIdentityBundleV1,
    /// Test-only witness that the shipping tokenizer returned the offset-free encoding shape.
    #[cfg(test)]
    last_tokenizer_route_was_offset_free: AtomicBool,
}

impl fmt::Debug for Model2VecEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model2VecEmbedder")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("vocab_size", &self.vocab_size)
            .field("model_dir", &"<redacted>")
            .field("identity", &self.identity.fingerprint())
            .finish_non_exhaustive()
    }
}

impl Model2VecEmbedder {
    /// Load a `Model2Vec` model from a directory containing `tokenizer.json`
    /// and `model.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` if required files are missing.
    /// Returns `SearchError::ModelLoadFailed` if files exist but cannot be parsed.
    #[instrument(skip_all, fields(model = DEFAULT_MODEL_NAME))]
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// Load the registered potion model with a custom display identifier.
    ///
    /// The supplied name does not select or attest compatibility; production
    /// model bytes and runtime identity still come exclusively from the frozen
    /// potion manifest.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` if required files are missing.
    /// Returns `SearchError::ModelLoadFailed` if files exist but cannot be parsed.
    pub fn load_with_name(model_dir: impl AsRef<Path>, name: &str) -> SearchResult<Self> {
        let model_dir = model_dir.as_ref();
        #[cfg(test)]
        {
            Self::load_explicit_test_model(model_dir, name)
        }
        #[cfg(not(test))]
        {
            // The native manifest is derived from the potion download manifest,
            // whose `.verified` receipt is minted after a full hash pass. Reuse
            // that receipt instead of re-hashing the 512 MB safetensors file on
            // every process start; any mismatch falls back to the full pass.
            let verified = ModelArtifactManifestV1::potion_128m_native()?.verify_dir_cached(
                &crate::model_manifest::ModelManifest::potion_128m(),
                model_dir,
            )?;
            let identity = verified.identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
            validate_registered_execution_contract(&identity)?;
            Self::load_preverified(model_dir, name, identity)
        }
    }

    /// Load the registered potion model, reusing the process-wide instance
    /// when one is already resident.
    ///
    /// Prefer this over [`Self::load`] in anything that may load the model
    /// more than once per process — a daemon, a server, a runtime rebuilt per
    /// request. Building the 500 353-piece Unigram tokenizer and reading the
    /// 512 MB matrix costs seconds and half a gigabyte, and neither depends on
    /// the caller, so the second and later loads in a process should be free
    /// (GH #46).
    ///
    /// Artifact verification runs on EVERY call, before any cached instance is
    /// handed back, and the cache key carries the attested identity
    /// fingerprint — so a model that fails admission is never served from
    /// cache, and a changed model never reuses the old matrix. The cache holds
    /// only a [`Weak`] reference, so the matrix is released as soon as the
    /// last caller drops its `Arc`.
    ///
    /// The key also carries the display name, so two callers that load the
    /// same directory under different names get separate instances rather than
    /// one whose `name` disagrees with what the caller asked for. Callers that
    /// want to share should agree on the name.
    ///
    /// This does NOT amortise across processes; a one-shot CLI invocation
    /// still pays a full load. Cross-process reuse would need a compiled
    /// tokenizer representation that `tokenizers` 0.23 does not provide.
    ///
    /// # Errors
    ///
    /// Same as [`Self::load`].
    #[instrument(skip_all, fields(model = DEFAULT_MODEL_NAME))]
    pub fn load_shared(model_dir: impl AsRef<Path>) -> SearchResult<Arc<Self>> {
        Self::load_shared_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// [`Self::load_shared`] with a custom display identifier, mirroring
    /// [`Self::load_with_name`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::load_with_name`].
    pub fn load_shared_with_name(
        model_dir: impl AsRef<Path>,
        name: &str,
    ) -> SearchResult<Arc<Self>> {
        let model_dir = model_dir.as_ref();
        #[cfg(test)]
        let identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 1);
        #[cfg(not(test))]
        let identity = {
            let verified = ModelArtifactManifestV1::potion_128m_native()?.verify_dir_cached(
                &crate::model_manifest::ModelManifest::potion_128m(),
                model_dir,
            )?;
            let identity = verified.identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
            validate_registered_execution_contract(&identity)?;
            identity
        };
        Self::load_shared_preverified(model_dir, name, identity)
    }

    /// Cache lookup and, on a miss, the full load.
    ///
    /// The cache lock is held ACROSS the load. That serialises concurrent
    /// first loads, which is the point: releasing it would let two threads
    /// each allocate a 512 MB matrix for the same model and then discard one.
    /// Loading is a startup event, and this crate carries a single fast-tier
    /// model, so the contention window is not on any steady-state path.
    // `significant_drop_tightening` wants the guard released before the load.
    // Releasing it is precisely the behaviour this design rejects: it would
    // let two threads each build a 512 MB matrix for the same model and throw
    // one away.
    #[allow(clippy::significant_drop_tightening)]
    fn load_shared_preverified(
        model_dir: &Path,
        name: &str,
        identity: EmbeddingIdentityBundleV1,
    ) -> SearchResult<Arc<Self>> {
        let key = SharedModelKey {
            dir: canonical_model_dir(model_dir),
            name: name.to_owned(),
            identity: identity.fingerprint(),
        };

        let mut cache = shared_model_cache()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        if let Some(existing) = cache.get(&key).and_then(Weak::upgrade) {
            return Ok(existing);
        }

        let embedder = Arc::new(Self::load_preverified(model_dir, name, identity)?);
        // Drop entries whose last strong reference is gone, so a long-lived
        // process that cycles through model directories does not accumulate
        // dead keys.
        cache.retain(|_, weak| weak.strong_count() > 0);
        cache.insert(key, Arc::downgrade(&embedder));
        Ok(embedder)
    }

    fn load_preverified(
        model_dir: &Path,
        name: &str,
        mut identity: EmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        // Validate required files exist
        for filename in &REQUIRED_FILES {
            let path = model_dir.join(filename);
            if !path.exists() {
                return Err(SearchError::ModelNotFound {
                    name: format!("{name} (missing {filename} in {})", model_dir.display()),
                });
            }
        }

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| SearchError::ModelLoadFailed {
                path: tokenizer_path,
                source: format!("failed to load tokenizer: {e}").into(),
            })?;

        // Locate the embedding tensor WITHOUT reading the matrix.
        //
        // This used to be `std::fs::read` of the whole file plus
        // `SafeTensors::deserialize` over that buffer, which meant the raw 512
        // MB of file bytes and the decoded 512 MB `Vec<f32>` were live at the
        // same time — a ~1 GB peak and roughly 2.6 GB of pages touched for a
        // 512 MB artifact (GH #46). Only the header is read here; the matrix
        // is streamed straight into its final `Vec<f32>` below, and only after
        // every admission check has passed.
        let safetensors_path = model_dir.join("model.safetensors");
        let located = SafetensorsF32Matrix::locate(&safetensors_path).map_err(|e| {
            SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: e.into(),
            }
        })?;

        let vocab_size = located.vocab_size;
        let dimensions = located.dimensions;
        let parsed_dimension =
            u32::try_from(dimensions).map_err(|_| SearchError::InvalidConfig {
                field: "model2vec.dimension".to_owned(),
                value: dimensions.to_string(),
                reason: "parsed tensor dimension exceeds the identity schema".to_owned(),
            })?;

        if identity.producer.backend == "explicit-test-backend" {
            identity.space.dimension = parsed_dimension;
            identity.storage.dimension = parsed_dimension;
            identity.producer.golden_vectors.dimension = parsed_dimension;
            identity.producer.space_fingerprint = identity.space.fingerprint();
        }
        if identity.space.dimension != parsed_dimension {
            return Err(SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: format!(
                    "parsed embedding dimension {parsed_dimension} disagrees with attested dimension {}",
                    identity.space.dimension
                )
                .into(),
            });
        }
        identity.validate()?;

        // Stream the matrix into its final buffer. Every check that could
        // reject this artifact — receipt, dtype, shape, attested dimension,
        // identity — has already run, so no borrowed or decoded bytes can
        // outlive a failed admission.
        let embeddings = located
            .read_values()
            .map_err(|e| SearchError::ModelLoadFailed {
                path: safetensors_path,
                source: e.into(),
            })?;

        #[cfg(test)]
        record_model2vec_full_load(model_dir);

        tracing::info!(
            model = DEFAULT_MODEL_NAME,
            vocab_size,
            dimensions,
            manifest = %identity.producer.provenance_manifest_fingerprint,
            identity = %identity.fingerprint(),
            "Model2Vec model loaded"
        );

        Ok(Self {
            tokenizer,
            embeddings,
            dimensions,
            vocab_size,
            name: name.to_owned(),
            model_dir: model_dir.to_owned(),
            identity,
            #[cfg(test)]
            last_tokenizer_route_was_offset_free: AtomicBool::new(false),
        })
    }

    #[cfg(test)]
    fn load_explicit_test_model(model_dir: &Path, name: &str) -> SearchResult<Self> {
        Self::load_preverified(
            model_dir,
            name,
            EmbeddingIdentityBundleV1::explicit_test_model(name, 1),
        )
    }

    /// Synchronous embedding (no async overhead for ~0.57ms operation).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbeddingFailed` if tokenization fails or
    /// all tokens are out-of-vocabulary.
    pub fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        if text.is_empty() {
            // Empty text → return zero vector (consistent with hash embedder)
            return Ok(vec![0.0; self.dimensions]);
        }

        // `Model2Vec` consumes token IDs only. Avoid constructing offsets, token strings, and
        // word IDs that this private boundary immediately discards.
        let encoding =
            self.tokenizer
                .encode_fast(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;

        #[cfg(test)]
        self.last_tokenizer_route_was_offset_free.store(
            !encoding.get_ids().is_empty()
                && encoding.get_tokens().iter().all(String::is_empty)
                && encoding
                    .get_offsets()
                    .iter()
                    .all(|&offset| offset == (0, 0))
                && encoding.get_word_ids().iter().all(Option::is_none),
            Ordering::Relaxed,
        );

        Ok(self.embed_token_ids(encoding.get_ids()))
    }

    #[inline]
    fn embed_token_ids(&self, token_ids: &[u32]) -> Vec<f32> {
        if token_ids.is_empty() {
            return vec![0.0; self.dimensions];
        }

        // Mean pool: accumulate embeddings for in-vocabulary tokens
        let mut sum = vec![0.0_f32; self.dimensions];
        let count = crate::simd::accumulate_model2vec_rows(
            &mut sum,
            &self.embeddings,
            token_ids,
            self.vocab_size,
        );

        if count == 0 {
            // All tokens were OOV — return zero vector
            return vec![0.0; self.dimensions];
        }

        // Store each rounded mean value before adding its square in the same strict
        // left-to-right `Iterator::sum()` order, starting from 0.0, as the former pass.
        finish_mean_pool_and_normalize(&mut sum, count);
        sum
    }

    #[cfg(test)]
    fn last_tokenizer_route_was_offset_free(&self) -> bool {
        self.last_tokenizer_route_was_offset_free
            .load(Ordering::Relaxed)
    }

    /// Former pre-lever `encode` route retained only for the existing internal benchmark.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_embed_sync_former_encode(&self, text: &str) -> SearchResult<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimensions]);
        }

        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;
        Ok(self.embed_token_ids(encoding.get_ids()))
    }

    /// Former pre-lever mean-pool finish retained only for the internal paired benchmark.
    #[cfg(feature = "bench-internals")]
    #[doc(hidden)]
    pub fn benchmark_embed_sync_former_finish(&self, text: &str) -> SearchResult<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dimensions]);
        }

        let encoding =
            self.tokenizer
                .encode_fast(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;
        Ok(self.embed_token_ids_with_former_finish(encoding.get_ids()))
    }

    #[cfg(feature = "bench-internals")]
    fn embed_token_ids_with_former_finish(&self, token_ids: &[u32]) -> Vec<f32> {
        if token_ids.is_empty() {
            return vec![0.0; self.dimensions];
        }

        let mut sum = vec![0.0_f32; self.dimensions];
        let count = crate::simd::accumulate_model2vec_rows(
            &mut sum,
            &self.embeddings,
            token_ids,
            self.vocab_size,
        );
        if count == 0 {
            return vec![0.0; self.dimensions];
        }

        finish_mean_pool_and_normalize_former(&mut sum, count);
        sum
    }

    /// Embed a batch of texts, dispatching per-document `embed_sync` across Rayon
    /// threads once the batch reaches [`PARALLEL_BATCH_MIN`]. Each document is
    /// independent CPU-bound work, so the result is **identical** to the serial loop
    /// (Rayon's indexed `collect` preserves input order); only the wall-clock differs.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::EmbeddingFailed` if tokenization fails or
    /// all tokens are out-of-vocabulary for any text in the batch.
    pub fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        if texts.len() >= PARALLEL_BATCH_MIN {
            texts.par_iter().map(|text| self.embed_sync(text)).collect()
        } else {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed_sync(text)?);
            }
            Ok(results)
        }
    }

    /// The directory this model was loaded from.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Vocabulary size (number of token embeddings in the matrix).
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[inline]
fn finish_mean_pool_and_normalize(sum: &mut [f32], count: usize) {
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    let mut norm_sq = 0.0_f32;
    for value in sum.iter_mut() {
        *value *= inv;
        norm_sq += *value * *value;
    }
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for value in sum {
            *value *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
}

/// Exact former finish sequence retained only as an independently callable oracle.
#[cfg(any(test, feature = "bench-internals"))]
fn finish_mean_pool_and_normalize_former(sum: &mut [f32], count: usize) {
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    for value in sum.iter_mut() {
        *value *= inv;
    }
    let norm_sq: f32 = sum.iter().map(|value| value * value).sum();
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for value in sum {
            *value *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
}

fn validate_registered_execution_contract(
    identity: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    for (field, actual, expected) in [
        (
            "model preprocessing",
            identity.space.model_preprocessing.as_str(),
            MODEL2VEC_PREPROCESSING_V1,
        ),
        (
            "sequence policy",
            identity.space.sequence_policy.as_str(),
            MODEL2VEC_SEQUENCE_POLICY_V1,
        ),
        (
            "pooling",
            identity.space.pooling.as_str(),
            MODEL2VEC_POOLING_V1,
        ),
        (
            "output normalization",
            identity.space.output_normalization.as_str(),
            MODEL2VEC_OUTPUT_NORMALIZATION_V1,
        ),
    ] {
        if actual != expected {
            return Err(SearchError::InvalidConfig {
                field: "model2vec.execution_contract".to_owned(),
                value: identity.space.logical_model_id.clone(),
                reason: format!("registered {field} disagrees with the native Model2Vec backend"),
            });
        }
    }
    Ok(())
}

fn embed_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

impl Embedder for Model2VecEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            embed_checkpoint(cx, "model2vec.embed")?;
            self.embed_sync(text)
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            embed_checkpoint(cx, "model2vec.embed_batch")?;
            self.embed_batch_sync(texts)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn dimension(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &str {
        &self.name
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

/// Discover the embedding tensor name among a safetensors file's tensor names.
///
/// Tries known names first, then falls back to using the only tensor
/// if the file contains exactly one.
fn discover_tensor_name(names: &[&str]) -> Option<String> {
    // Try known candidate names
    for candidate in &TENSOR_NAME_CANDIDATES {
        if names.contains(candidate) {
            return Some((*candidate).to_owned());
        }
    }

    // Fallback: if exactly one tensor exists, use it regardless of name
    if let [only] = names {
        return Some((*only).to_owned());
    }

    None
}

/// Length prefix of a safetensors file: a little-endian `u64` header size.
const SAFETENSORS_HEADER_LEN_PREFIX: usize = 8;

/// Upper bound on the safetensors JSON header, matching the `MAX_HEADER_SIZE`
/// the `safetensors` crate enforces in `SafeTensors::read_metadata`. Kept so a
/// corrupt length prefix cannot ask for a huge allocation.
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

/// Bytes pulled per `read_exact` while streaming the matrix. A multiple of 4,
/// so every chunk decodes into whole `f32` values with no carry-over.
const MATRIX_READ_CHUNK_BYTES: usize = 1 << 20;

/// A located, validated F32 matrix inside a safetensors file, not yet read.
///
/// Splitting "locate" from "read" is what keeps the peak allocation at one
/// copy of the matrix: the header is parsed from a few hundred bytes, the
/// caller runs every admission check against the shape it reports, and only
/// then are the tensor bytes streamed into their final `Vec<f32>`.
struct SafetensorsF32Matrix {
    file: File,
    /// Absolute offset of the tensor's first byte within the file.
    data_start: u64,
    /// Tensor byte length; always `vocab_size * dimensions * 4`.
    byte_len: usize,
    vocab_size: usize,
    dimensions: usize,
}

impl SafetensorsF32Matrix {
    /// Parse the safetensors header and locate the embedding tensor.
    ///
    /// Performs the same structural validation the whole-file path did —
    /// header bounds, offset contiguity (via the `safetensors` crate's own
    /// `Metadata` deserializer), file-length completeness, 2-D shape, exact
    /// byte count — plus an EXPLICIT dtype check. The old path enforced F32
    /// only implicitly, through the byte-count comparison in
    /// `parse_f32_matrix`; a same-shaped tensor of another dtype was rejected
    /// with a size-mismatch message rather than a dtype one.
    fn locate(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("failed to open safetensors: {e}"))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("failed to stat safetensors: {e}"))?
            .len();

        let mut prefix = [0_u8; SAFETENSORS_HEADER_LEN_PREFIX];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("failed to read safetensors header length: {e}"))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_SAFETENSORS_HEADER_BYTES {
            return Err(format!(
                "safetensors header of {header_len} bytes exceeds the {MAX_SAFETENSORS_HEADER_BYTES}-byte limit"
            ));
        }
        // Bound the header against the file too, so a corrupt prefix cannot
        // ask for a 100 MB allocation that the read is then going to fail.
        if header_len > file_len.saturating_sub(SAFETENSORS_HEADER_LEN_PREFIX as u64) {
            return Err(format!(
                "safetensors header of {header_len} bytes exceeds the {file_len}-byte file"
            ));
        }
        let header_len_usize = usize::try_from(header_len)
            .map_err(|_| "safetensors header length exceeds this platform's usize".to_owned())?;

        let mut header = vec![0_u8; header_len_usize];
        file.read_exact(&mut header)
            .map_err(|e| format!("failed to read safetensors header: {e}"))?;
        let metadata: Metadata = serde_json::from_slice(&header)
            .map_err(|e| format!("failed to parse safetensors: {e}"))?;

        // Completeness check, identical in effect to the one
        // `SafeTensors::read_metadata` performs over a whole-file buffer: a
        // truncated or over-long file is rejected rather than silently
        // yielding a short matrix.
        let data_len = u64::try_from(metadata.data_len())
            .map_err(|_| "safetensors data length exceeds u64".to_owned())?;
        let expected_len = header_len
            .checked_add(data_len)
            .and_then(|n| n.checked_add(SAFETENSORS_HEADER_LEN_PREFIX as u64))
            .ok_or_else(|| "safetensors declared length overflows".to_owned())?;
        if expected_len != file_len {
            return Err(format!(
                "safetensors file length mismatch: header declares {expected_len} bytes, file is {file_len}"
            ));
        }

        let names = metadata.offset_keys();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let tensor_name = discover_tensor_name(&name_refs).ok_or_else(|| {
            format!(
                "no embedding tensor found. Tried: {TENSOR_NAME_CANDIDATES:?}. Available: {names:?}"
            )
        })?;
        let info = metadata
            .info(&tensor_name)
            .ok_or_else(|| format!("failed to get tensor '{tensor_name}'"))?;

        if info.dtype != Dtype::F32 {
            return Err(format!(
                "expected an F32 embedding tensor, got {:?} for '{tensor_name}'",
                info.dtype
            ));
        }
        if info.shape.len() != 2 {
            return Err(format!(
                "expected 2D tensor, got {}D with shape {:?}",
                info.shape.len(),
                info.shape
            ));
        }
        let vocab_size = info.shape[0];
        let dimensions = info.shape[1];

        let (start, end) = info.data_offsets;
        let byte_len = end
            .checked_sub(start)
            .ok_or_else(|| format!("tensor '{tensor_name}' has inverted data offsets"))?;
        let expected_bytes = vocab_size
            .checked_mul(dimensions)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| format!("byte size overflow for [{vocab_size} x {dimensions}] f32"))?;
        if byte_len != expected_bytes {
            return Err(format!(
                "tensor data size mismatch: expected {expected_bytes} bytes for [{vocab_size} x {dimensions}] f32, got {byte_len}"
            ));
        }

        let data_start = u64::try_from(start)
            .ok()
            .and_then(|s| s.checked_add(header_len))
            .and_then(|s| s.checked_add(SAFETENSORS_HEADER_LEN_PREFIX as u64))
            .ok_or_else(|| format!("tensor '{tensor_name}' offset overflows the file"))?;

        Ok(Self {
            file,
            data_start,
            byte_len,
            vocab_size,
            dimensions,
        })
    }

    /// Stream the tensor into a freshly allocated `Vec<f32>`.
    ///
    /// Decoding is byte-for-byte what `parse_f32_matrix` did — the same
    /// little-endian `f32::from_le_bytes` over the same bytes in the same
    /// order — so the resulting matrix is bit-identical to the old path's.
    /// Only the buffering changes: one chunk of `MATRIX_READ_CHUNK_BYTES` is
    /// live at a time instead of the whole file.
    fn read_values(mut self) -> Result<Vec<f32>, String> {
        self.file
            .seek(SeekFrom::Start(self.data_start))
            .map_err(|e| format!("failed to seek to the embedding tensor: {e}"))?;

        let expected_elements = self.byte_len / 4;
        let mut values: Vec<f32> = Vec::with_capacity(expected_elements);
        // Never allocate a full chunk for a matrix smaller than one; both this
        // and `byte_len` are multiples of 4, so the invariant below holds
        // either way.
        let mut chunk = vec![0_u8; MATRIX_READ_CHUNK_BYTES.min(self.byte_len.max(4))];
        let mut remaining = self.byte_len;

        while remaining > 0 {
            // Both operands are multiples of 4, so every chunk decodes with no
            // trailing partial value and `as_chunks` leaves no remainder.
            let take = remaining.min(chunk.len());
            debug_assert_eq!(take % 4, 0, "matrix reads must stay f32-aligned");
            self.file
                .read_exact(&mut chunk[..take])
                .map_err(|e| format!("failed to read the embedding tensor: {e}"))?;
            for &bytes in chunk[..take].as_chunks::<4>().0 {
                values.push(f32::from_le_bytes(bytes));
            }
            remaining -= take;
        }

        if values.len() != expected_elements {
            return Err(format!(
                "parsed element count mismatch: expected {expected_elements}, got {}",
                values.len()
            ));
        }
        Ok(values)
    }
}

/// Test hook: how many FULL loads (tokenizer construction plus matrix read)
/// each model directory has cost this process.
///
/// Counted PER DIRECTORY rather than as one global tally so an assertion is
/// unaffected by the other tests loading their own fixtures in parallel.
/// `#[cfg(test)]` — a shipping build carries neither the map nor the lock.
#[cfg(test)]
fn model2vec_full_load_counts() -> &'static Mutex<HashMap<PathBuf, u64>> {
    static COUNTS: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Full loads charged to `model_dir` so far in this process.
#[cfg(test)]
fn model2vec_full_loads_for(model_dir: &Path) -> u64 {
    model2vec_full_load_counts()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(model_dir)
        .copied()
        .unwrap_or(0)
}

#[cfg(test)]
fn record_model2vec_full_load(model_dir: &Path) {
    *model2vec_full_load_counts()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .entry(canonical_model_dir(model_dir))
        .or_insert(0) += 1;
}

/// Canonical form of a model directory, used as the cache key and as the
/// per-directory counter key so both agree.
///
/// Canonicalising makes `./model` and its absolute path one entry — and on
/// macOS, where `/var` is a symlink to `/private/var`, keeps a temp directory
/// from being two. A path that cannot be canonicalised (it may not exist; the
/// load reports that properly) is used as given.
fn canonical_model_dir(model_dir: &Path) -> PathBuf {
    std::fs::canonicalize(model_dir).unwrap_or_else(|_| model_dir.to_owned())
}

/// Key for the process-wide loaded-model cache.
///
/// Includes the attested identity fingerprint, not just the path, so a model
/// directory whose contents changed can never be answered from a cache entry
/// built for the old contents.
#[derive(Clone, PartialEq, Eq, Hash)]
struct SharedModelKey {
    dir: PathBuf,
    name: String,
    identity: String,
}

type SharedModelCache = Mutex<HashMap<SharedModelKey, Weak<Model2VecEmbedder>>>;

fn shared_model_cache() -> &'static SharedModelCache {
    static CACHE: OnceLock<SharedModelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Parse raw bytes from a safetensors tensor into a flat `Vec<f32>` matrix.
///
/// Expects little-endian f32 data with shape `[vocab_size, dimensions]`.
///
/// This is the pre-#46 whole-buffer decoder. The shipping loader streams the
/// tensor instead (`SafetensorsF32Matrix::read_values`) so the file bytes and
/// the decoded matrix are never both resident; this is retained as the
/// BIT-IDENTITY ORACLE that stream decoding is checked against, which is why
/// it must keep decoding exactly the way it always did.
#[cfg(test)]
fn parse_f32_matrix(data: &[u8], vocab_size: usize, dimensions: usize) -> Result<Vec<f32>, String> {
    let expected_elements = vocab_size
        .checked_mul(dimensions)
        .ok_or_else(|| format!("matrix size overflow for [{vocab_size} x {dimensions}]"))?;
    let expected_bytes = expected_elements
        .checked_mul(4)
        .ok_or_else(|| format!("byte size overflow for [{vocab_size} x {dimensions}] f32"))?;
    if data.len() != expected_bytes {
        return Err(format!(
            "tensor data size mismatch: expected {expected_bytes} bytes for [{vocab_size} x {dimensions}] f32, got {}",
            data.len()
        ));
    }

    // Pre-allocate the exact size
    let mut matrix = Vec::with_capacity(expected_elements);

    // Parse bytes in 4-byte chunks (length is validated above to be a
    // multiple of 4, so `as_chunks` leaves no remainder)
    for &bytes in data.as_chunks::<4>().0 {
        matrix.push(f32::from_le_bytes(bytes));
    }

    if matrix.len() != expected_elements {
        return Err(format!(
            "parsed element count mismatch: expected {}, got {}",
            expected_elements,
            matrix.len()
        ));
    }

    Ok(matrix)
}

/// Search for a `Model2Vec` model directory in standard locations.
///
/// Checks these paths in order:
/// 1. `$FRANKENSEARCH_MODEL_DIR/<model_name>/`
/// 2. `$XDG_DATA_HOME/frankensearch/models/<model_name>/`
/// 3. `~/.local/share/frankensearch/models/<model_name>/` (or macOS
///    `~/Library/Application Support/frankensearch/models/<model_name>/`)
/// 4. `~/.cache/huggingface/hub/models--<hf_id>/snapshots/*/`
///
/// Returns `None` if no directory with the required files is found.
#[must_use]
pub fn find_model_dir(model_name: &str) -> Option<PathBuf> {
    find_model_dir_with_hf_id(model_name, DEFAULT_HF_ID)
}

/// Search for a `Model2Vec` model directory with a specific `HuggingFace` ID.
#[must_use]
pub fn find_model_dir_with_hf_id(model_name: &str, hf_id: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    // 1. Explicit env var override
    if let Ok(dir) = std::env::var("FRANKENSEARCH_MODEL_DIR") {
        let base = PathBuf::from(dir);
        for variant in model_directory_variants(model_name) {
            candidates.push(base.join(variant));
        }
        candidates.push(base);
    }

    // 2-3. Standard frankensearch model layout (created on first access)
    let model_root = ensure_model_storage_layout();
    for variant in model_directory_variants(model_name) {
        candidates.push(model_root.join(variant));
    }

    // 4. HuggingFace cache
    if let Some(cache_dir) = frankensearch_core::platform_dirs::cache_dir() {
        let hf_dir = cache_dir
            .join("huggingface/hub")
            .join(format!("models--{}", hf_id.replace('/', "--")));
        if let Ok(snapshots) = std::fs::read_dir(hf_dir.join("snapshots")) {
            for entry in snapshots.flatten() {
                candidates.push(entry.path());
            }
        }
    }

    // Check each candidate for required files
    for candidate in &candidates {
        if has_required_files(candidate) {
            return Some(candidate.clone());
        }
    }

    None
}

/// Check if a directory contains all required `Model2Vec` files.
fn has_required_files(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{Model2VecAccumulationRoute, last_model2vec_accumulation_route_for_test};
    use std::fs;

    /// Create a minimal `Model2Vec` model in a temp directory for testing.
    ///
    /// Creates a tiny tokenizer and a small safetensors file with known values.
    fn create_test_model(dir: &Path, vocab_size: usize, dimensions: usize) {
        // Create a minimal tokenizer.json
        // This is a minimal valid HuggingFace tokenizer config
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "[UNK]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": {
                "type": "Lowercase"
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": create_test_vocab(vocab_size),
                "unk_token": "[UNK]"
            }
        });

        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();

        // Create safetensors file with known embedding values
        create_test_safetensors(dir, vocab_size, dimensions);
    }

    /// Create a tokenizer fixture that exercises added tokens and truncation while retaining
    /// enough rows to pool every emitted ID.
    fn create_tokenizer_parity_model(dir: &Path) {
        let mut vocab = create_test_vocab(16)
            .as_object()
            .expect("test vocabulary is an object")
            .clone();
        vocab.insert("café".to_owned(), serde_json::Value::from(11));

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": {
                "direction": "Right",
                "max_length": 512,
                "strategy": "LongestFirst",
                "stride": 0
            },
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "[UNK]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 12,
                    "content": "<added>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": true,
                    "special": false
                },
                {
                    "id": 13,
                    "content": "[SPECIAL]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": {
                "type": "Lowercase"
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "[UNK]"
            }
        });

        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        create_test_safetensors(dir, 16, 256);
    }

    /// Create a test vocabulary mapping words to token IDs.
    fn create_test_vocab(vocab_size: usize) -> serde_json::Value {
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".to_owned(), serde_json::Value::from(0));

        let test_words = [
            "hello", "world", "test", "rust", "search", "embed", "vector", "model", "fast", "query",
        ];

        for (i, word) in test_words.iter().enumerate() {
            if i + 1 < vocab_size {
                vocab.insert((*word).to_owned(), serde_json::Value::from(i + 1));
            }
        }

        serde_json::Value::Object(vocab)
    }

    /// Create a minimal safetensors file with a known embedding matrix.
    fn create_test_safetensors(dir: &Path, vocab_size: usize, dimensions: usize) {
        use std::collections::HashMap;

        // Build embedding matrix: each row is [row_idx * 0.1, row_idx * 0.1 + 0.01, ...]
        let mut data = Vec::with_capacity(vocab_size * dimensions * 4);
        for row in 0..vocab_size {
            for col in 0..dimensions {
                #[allow(clippy::cast_precision_loss)]
                let val = (row as f32).mul_add(0.1, (col as f32) * 0.01);
                data.extend_from_slice(&val.to_le_bytes());
            }
        }

        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![vocab_size, dimensions],
                &data,
            )
            .unwrap(),
        );

        let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
        fs::write(dir.join("model.safetensors"), serialized).unwrap();
    }

    // ── Loading ────────────────────────────────────────────────────────

    #[test]
    fn load_valid_model() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load_with_name(dir.path(), "test-model").unwrap();
        assert_eq!(embedder.dimensions, 8);
        assert_eq!(embedder.vocab_size, 12);
        assert_eq!(embedder.name, "test-model");
    }

    #[test]
    fn load_preverified_rejects_tensor_dimension_drift() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("dimension-drift", 7);
        identity.producer.backend = "attested-fixture-backend".to_owned();
        identity.validate().unwrap();

        let error = Model2VecEmbedder::load_preverified(dir.path(), "dimension-drift", identity)
            .expect_err("parsed tensor width must agree with the attested identity");
        assert!(matches!(error, SearchError::ModelLoadFailed { .. }));
    }

    #[test]
    fn registered_identity_matches_native_execution_contract() {
        let identity = ModelArtifactManifestV1::potion_128m_native()
            .unwrap()
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        validate_registered_execution_contract(&identity).unwrap();

        let mut drifted = identity;
        drifted.space.model_preprocessing.push_str("-drift");
        assert!(validate_registered_execution_contract(&drifted).is_err());
    }

    #[test]
    fn embed_batch_sync_matches_serial_across_parallel_boundary() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);
        let embedder = Model2VecEmbedder::load_with_name(dir.path(), "test-model").unwrap();

        // Straddle PARALLEL_BATCH_MIN so both the serial and Rayon paths are exercised.
        for &batch_size in &[0_usize, 1, PARALLEL_BATCH_MIN - 1, PARALLEL_BATCH_MIN, 17] {
            let docs: Vec<String> = (0..batch_size)
                .map(|i| format!("hello world test rust search {i}"))
                .collect();
            let texts: Vec<&str> = docs.iter().map(String::as_str).collect();

            let serial: Vec<Vec<f32>> = texts
                .iter()
                .map(|t| former_embed_sync(&embedder, t))
                .collect();
            let batched = embedder.embed_batch_sync(&texts).unwrap();

            assert_eq!(batched.len(), serial.len(), "len at n={batch_size}");
            for (index, (batched, former)) in batched.iter().zip(&serial).enumerate() {
                assert_f32_bits_eq(
                    batched,
                    former,
                    &format!("batch order or output diverged at n={batch_size}, index={index}"),
                );
            }
        }
    }

    #[test]
    fn load_missing_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        // Only create safetensors, not tokenizer
        create_test_safetensors(dir.path(), 10, 4);

        let result = Model2VecEmbedder::load(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SearchError::ModelNotFound { .. }),
            "expected ModelNotFound, got {err:?}"
        );
    }

    #[test]
    fn load_missing_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        // Only create tokenizer
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();

        let result = Model2VecEmbedder::load(dir.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SearchError::ModelNotFound { .. }
        ));
    }

    #[test]
    fn load_nonexistent_directory() {
        let result = Model2VecEmbedder::load("/nonexistent/path/to/model");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SearchError::ModelNotFound { .. }
        ));
    }

    // ── Embedding ──────────────────────────────────────────────────────

    #[test]
    fn embed_produces_correct_dimension() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        let vec = embedder.embed_sync("hello world").unwrap();
        assert_eq!(vec.len(), 8);
    }

    #[test]
    fn embed_output_is_l2_normalized() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        let vec = embedder.embed_sync("hello world").unwrap();

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }

    #[test]
    fn embed_empty_string_returns_zero_vector() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        let vec = embedder.embed_sync("").unwrap();
        assert_eq!(vec.len(), 8);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn embed_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        let a = embedder.embed_sync("hello world").unwrap();
        let b = embedder.embed_sync("hello world").unwrap();
        assert_eq!(a, b, "same input must produce same output");
    }

    #[test]
    fn embed_different_inputs_different_outputs() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        let a = embedder.embed_sync("hello").unwrap();
        let b = embedder.embed_sync("world").unwrap();
        assert_ne!(a, b, "different inputs should produce different embeddings");
    }

    #[test]
    fn embed_sync_observes_offset_free_tokenizer_route() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);
        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();

        let former = embedder.tokenizer.encode("hello world", false).unwrap();
        assert!(
            former.get_tokens().iter().any(|token| !token.is_empty()),
            "the former encode route must materialize token text for this planted route witness"
        );

        embedder.embed_sync("hello world").unwrap();
        assert!(
            embedder.last_tokenizer_route_was_offset_free(),
            "shipping Model2Vec must retain the encode_fast offset-free tokenizer route"
        );
    }

    #[test]
    fn encode_fast_token_ids_and_vectors_match_former_oracle() {
        let dir = tempfile::tempdir().unwrap();
        create_tokenizer_parity_model(dir.path());
        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();

        let mut inputs = crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1
            .iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>();
        inputs.extend([
            String::new(),
            "HELLO world".to_owned(),
            "CAFÉ cafe\u{301}".to_owned(),
            "hello 東京 world".to_owned(),
            "definitely-oov-token".to_owned(),
            "hello <ADDED> [SPECIAL] world".to_owned(),
        ]);

        for tokens in [511_usize, 512, 513] {
            inputs.push(
                std::iter::repeat_n("hello", tokens)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }

        for input in &inputs {
            assert_fast_ids_and_former_vector_bits(&embedder, input, input);
        }

        let special_ids = former_token_ids(&embedder, "hello <ADDED> [SPECIAL] world");
        assert!(
            special_ids.contains(&12) && special_ids.contains(&13),
            "fixture must exercise both the normalized added token and the added special token"
        );
        for tokens in [511_usize, 512, 513] {
            let input = std::iter::repeat_n("hello", tokens)
                .collect::<Vec<_>>()
                .join(" ");
            let expected_len = tokens.min(512);
            let former_ids = former_token_ids(&embedder, &input);
            let fast_ids = embedder
                .tokenizer
                .encode_fast(input.as_str(), false)
                .unwrap()
                .get_ids()
                .to_vec();
            assert_eq!(
                former_ids.len(),
                expected_len,
                "former tokenizer truncation at {tokens} input tokens"
            );
            assert_eq!(
                fast_ids.len(),
                expected_len,
                "fast tokenizer truncation at {tokens} input tokens"
            );
        }

        let texts = inputs.iter().map(String::as_str).collect::<Vec<_>>();
        let former = texts
            .iter()
            .map(|text| former_embed_sync(&embedder, text))
            .collect::<Vec<_>>();
        let batched = embedder.embed_batch_sync(&texts).unwrap();
        assert_eq!(batched.len(), former.len());
        for (index, (actual, expected)) in batched.iter().zip(&former).enumerate() {
            assert_f32_bits_eq(
                actual,
                expected,
                &format!("batch order or vector bits changed at index={index}"),
            );
        }
    }

    /// The former `embed_sync` pooling and finish sequence, retained independently
    /// of the production gather helper for exact native-256 parity checks.
    fn former_embed_sync(embedder: &Model2VecEmbedder, text: &str) -> Vec<f32> {
        if text.is_empty() {
            return vec![0.0; embedder.dimensions];
        }

        let token_ids = former_token_ids(embedder, text);
        former_embed_token_ids(embedder, &token_ids)
    }

    fn former_embed_token_ids(embedder: &Model2VecEmbedder, token_ids: &[u32]) -> Vec<f32> {
        let mut sum = vec![0.0_f32; embedder.dimensions];
        let mut count = 0_usize;
        for &token_id in token_ids {
            let index = token_id as usize;
            if index < embedder.vocab_size {
                let start = index * embedder.dimensions;
                crate::simd::accumulate_f32_into(
                    &mut sum,
                    &embedder.embeddings[start..start + embedder.dimensions],
                );
                count += 1;
            }
        }
        if count == 0 {
            return vec![0.0; embedder.dimensions];
        }

        finish_mean_pool_and_normalize_former(&mut sum, count);
        sum
    }

    fn former_token_ids(embedder: &Model2VecEmbedder, text: &str) -> Vec<u32> {
        embedder
            .tokenizer
            .encode(text, false)
            .unwrap()
            .get_ids()
            .to_vec()
    }

    fn assert_fast_ids_and_former_vector_bits(
        embedder: &Model2VecEmbedder,
        text: &str,
        scenario: &str,
    ) {
        let former_ids = former_token_ids(embedder, text);
        let fast_ids = embedder
            .tokenizer
            .encode_fast(text, false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(fast_ids, former_ids, "token IDs diverged for {scenario}");

        let expected = former_embed_sync(embedder, text);
        let actual = embedder.embed_sync(text).unwrap();
        assert_f32_bits_eq(&actual, &expected, scenario);
    }

    fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], scenario: &str) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{scenario}"
        );
    }

    fn expected_native_256_route(token_count: usize) -> Model2VecAccumulationRoute {
        #[cfg(target_arch = "x86_64")]
        {
            if token_count < 512 {
                if std::is_x86_feature_detected!("avx2") {
                    Model2VecAccumulationRoute::Native256ShortAvx2
                } else {
                    Model2VecAccumulationRoute::Base
                }
            } else {
                Model2VecAccumulationRoute::Prefetched
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = token_count;
            Model2VecAccumulationRoute::Base
        }
    }

    #[test]
    fn native_256_embed_sync_matches_former_pool_and_finish_bits() {
        const DIMENSIONS: usize = 256;
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, DIMENSIONS);
        let mut embedder = Model2VecEmbedder::load(dir.path()).unwrap();

        for &tokens in &[0_usize, 1, 2, 3, 4, 8, 16, 32, 64, 511, 512, 513] {
            let text = (0..tokens)
                .map(|position| match position % 4 {
                    0 => "hello",
                    1 => "world",
                    2 => "missing-token",
                    _ => "hello",
                })
                .collect::<Vec<_>>()
                .join(" ");
            let expected = former_embed_sync(&embedder, &text);
            let actual = embedder.embed_sync(&text).unwrap();
            assert_f32_bits_eq(&actual, &expected, &format!("tokens={tokens}"));
            assert_eq!(
                embedder
                    .tokenizer
                    .encode_fast(text.as_str(), false)
                    .unwrap()
                    .get_ids(),
                former_token_ids(&embedder, &text),
                "token IDs at native-256 boundary tokens={tokens}"
            );
            if !text.is_empty() {
                let token_count = embedder
                    .tokenizer
                    .encode(text.as_str(), false)
                    .unwrap()
                    .len();
                assert_eq!(
                    last_model2vec_accumulation_route_for_test(),
                    expected_native_256_route(token_count),
                    "shipping embed_sync route for {tokens} input words ({token_count} token IDs)"
                );
            }
        }

        for text in [
            "hello caf\u{e9} world",
            "hello \u{6771}\u{4eac} hello",
            "HELLO hello HELLO",
        ] {
            let expected = former_embed_sync(&embedder, text);
            let actual = embedder.embed_sync(text).unwrap();
            assert_f32_bits_eq(&actual, &expected, text);
        }

        let hello = DIMENSIONS..DIMENSIONS * 2;
        embedder.embeddings[hello.clone()].fill(-0.0);
        let expected = former_embed_sync(&embedder, "hello hello");
        let actual = embedder.embed_sync("hello hello").unwrap();
        assert_f32_bits_eq(&actual, &expected, "signed-zero row");

        embedder.embeddings[hello.clone()].fill(1.0e-20);
        let expected = former_embed_sync(&embedder, "hello");
        let actual = embedder.embed_sync("hello").unwrap();
        assert_f32_bits_eq(&actual, &expected, "below normalization guard");

        embedder.embeddings[hello.clone()].fill(1.0e-4);
        let expected = former_embed_sync(&embedder, "hello");
        let actual = embedder.embed_sync("hello").unwrap();
        assert_f32_bits_eq(&actual, &expected, "above normalization guard");

        embedder.embeddings[hello].fill(f32::NAN);
        let expected = former_embed_sync(&embedder, "hello world hello");
        let actual = embedder.embed_sync("hello world hello").unwrap();
        assert_f32_bits_eq(&actual, &expected, "non-finite pooled row");
    }

    #[test]
    fn fused_mean_and_ordered_norm_finish_matches_former_bits() {
        let arbitrary_initial_sum = (0_u32..256)
            .map(|index| {
                let value = f32::from_bits(0x3f80_0000 + index);
                if index % 2 == 0 { value } else { -value }
            })
            .collect::<Vec<_>>();
        let cases = vec![
            ("arbitrary finite sum", arbitrary_initial_sum),
            (
                "normalization guard boundary",
                vec![f32::MIN_POSITIVE, f32::EPSILON.sqrt(), -f32::EPSILON.sqrt()],
            ),
            ("signed zero", vec![-0.0, 0.0, -0.0, 0.0]),
            (
                "non-finite values",
                vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX],
            ),
        ];

        for (label, initial_sum) in cases {
            for &count in &[1_usize, 2, 3, 4, 511, 512, 513] {
                let mut former = initial_sum.clone();
                let mut fused = initial_sum.clone();
                finish_mean_pool_and_normalize_former(&mut former, count);
                finish_mean_pool_and_normalize(&mut fused, count);
                assert_f32_bits_eq(&fused, &former, &format!("{label}, count={count}"));
            }
        }
    }

    #[test]
    fn full_embed_sync_fused_finish_matches_former_bits_across_shapes_and_values() {
        for &dimensions in &[1_usize, 255, 256, 257] {
            let dir = tempfile::tempdir().unwrap();
            create_test_model(dir.path(), 12, dimensions);
            let mut embedder = Model2VecEmbedder::load(dir.path()).unwrap();

            let mut full_embed_sync_corpus =
                vec![String::new(), "hello definitely-oov-token hello".to_owned()];
            for token_count in [1_usize, 2, 3, 4, 511, 512, 513] {
                full_embed_sync_corpus.push(
                    std::iter::repeat_n("hello", token_count)
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }

            let oov_grouping_ids = former_token_ids(&embedder, &full_embed_sync_corpus[1]);
            assert!(
                oov_grouping_ids.contains(&0),
                "the full embed_sync OOV grouping corpus must emit the [UNK] token at dim={dimensions}"
            );

            let hello = dimensions..dimensions * 2;
            for (lane, value) in embedder.embeddings[hello.clone()].iter_mut().enumerate() {
                let lane = u32::try_from(lane).expect("fixture dimension fits u32");
                let magnitude = f32::from_bits(0x3f80_0000 + lane);
                *value = if lane % 2 == 0 { magnitude } else { -magnitude };
            }
            for text in &full_embed_sync_corpus {
                assert_fast_ids_and_former_vector_bits(
                    &embedder,
                    text,
                    &format!("finite full embed_sync corpus dim={dimensions}"),
                );
            }

            #[allow(clippy::cast_precision_loss)]
            let guard_center = (f32::EPSILON / dimensions as f32).sqrt();
            let below_guard = f32::from_bits(guard_center.to_bits() - 1);
            let above_guard = f32::from_bits(guard_center.to_bits() + 1);
            for (scenario, value) in [
                ("subnormal", f32::from_bits(1)),
                ("below guard", below_guard),
                ("above guard", above_guard),
                ("signed zero", -0.0),
                ("NaN", f32::from_bits(0x7fc0_0001)),
                ("infinity", f32::INFINITY),
            ] {
                embedder.embeddings[hello.clone()].fill(value);
                for text in &full_embed_sync_corpus {
                    assert_fast_ids_and_former_vector_bits(
                        &embedder,
                        text,
                        &format!("{scenario} full embed_sync corpus dim={dimensions}"),
                    );
                }
            }

            let invalid_token_ids = [
                1_u32,
                u32::try_from(embedder.vocab_size).expect("fixture vocabulary fits u32"),
                u32::MAX,
                2_u32,
            ];
            let former = former_embed_token_ids(&embedder, &invalid_token_ids);
            let fused = embedder.embed_token_ids(&invalid_token_ids);
            assert_f32_bits_eq(
                &fused,
                &former,
                &format!("mixed valid/OOV token IDs dim={dimensions}"),
            );

            let all_oov_token_ids = [
                u32::try_from(embedder.vocab_size).expect("fixture vocabulary fits u32"),
                u32::MAX,
            ];
            let former = former_embed_token_ids(&embedder, &all_oov_token_ids);
            let fused = embedder.embed_token_ids(&all_oov_token_ids);
            assert_f32_bits_eq(
                &fused,
                &former,
                &format!("all OOV token IDs dim={dimensions}"),
            );
        }
    }

    // ── OOV Handling ───────────────────────────────────────────────────

    #[test]
    fn embed_all_oov_returns_zero_vector() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        // "zzzzzzz" is not in our test vocab
        let vec = embedder.embed_sync("xyzxyzxyz qqqqq").unwrap();
        // All tokens should be OOV → zero vector
        assert_eq!(vec.len(), 8);
    }

    // ── Embedder Trait ─────────────────────────────────────────────────

    #[test]
    fn trait_is_semantic() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert!(embedder.is_semantic());
    }

    #[test]
    fn trait_category_is_static() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert_eq!(embedder.category(), ModelCategory::StaticEmbedder);
    }

    #[test]
    fn trait_dimension() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert_eq!(embedder.dimension(), 8);
    }

    #[test]
    fn trait_does_not_infer_mrl_from_model2vec_backend() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert!(!embedder.supports_mrl());
    }

    #[test]
    fn trait_id_and_name() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load_with_name(dir.path(), "my-model").unwrap();
        assert_eq!(embedder.id(), "my-model");
        assert_eq!(embedder.model_name(), "my-model");
    }

    // ── Thread Safety ──────────────────────────────────────────────────

    #[test]
    fn embedder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Model2VecEmbedder>();
    }

    /// Bit-exact producer conformance proof for the frozen potion manifest.
    #[test]
    #[ignore = "requires a verified potion model dir via POTION_FIXTURE_DIR"]
    fn conformance_certificate_matches_fixture() {
        let dir = std::env::var("POTION_FIXTURE_DIR")
            .expect("set POTION_FIXTURE_DIR to a potion-multilingual-128M directory");
        let manifest = crate::model_manifest::ModelArtifactManifestV1::potion_128m_native()
            .expect("registered potion manifest");
        let verified = manifest
            .verify_dir(Path::new(&dir))
            .expect("verify frozen potion artifacts");
        let expected_identity = verified
            .identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .expect("derive verified potion identity");
        let embedder = Model2VecEmbedder::load_preverified(
            Path::new(&dir),
            DEFAULT_MODEL_NAME,
            expected_identity.clone(),
        )
        .expect("load verified potion embedder");
        assert_eq!(embedder.identity().unwrap(), &expected_identity);

        let added_vocabulary = embedder.tokenizer.get_added_vocabulary().get_vocab();
        let pad_id = *added_vocabulary
            .get("[PAD]")
            .expect("verified Potion tokenizer must retain [PAD] as an added special token");
        let unk_id = *added_vocabulary
            .get("[UNK]")
            .expect("verified Potion tokenizer must retain [UNK] as an added special token");
        let added_special_text = "hello [PAD] [UNK] world";
        let long_over_512_text = std::iter::repeat_n("hello", 1024)
            .collect::<Vec<_>>()
            .join(" ");
        let mut parity_texts = crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1
            .iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>();
        parity_texts.extend([
            "Caf\u{e9} na\u{ef}ve \u{2014} \u{6771}\u{4eac} \u{1f980}".to_owned(),
            added_special_text.to_owned(),
            "\tmetaspace  boundaries\tand unseen-oov-\u{10ffff}".to_owned(),
            long_over_512_text.clone(),
        ]);
        for text in &parity_texts {
            assert_fast_ids_and_former_vector_bits(
                &embedder,
                text,
                "verified Potion tokenizer parity input",
            );
        }

        let added_special_ids = former_token_ids(&embedder, added_special_text);
        assert!(
            added_special_ids.contains(&pad_id) && added_special_ids.contains(&unk_id),
            "verified Potion tokenizer must emit both literal added special-token IDs"
        );
        let former_long = embedder
            .tokenizer
            .encode(long_over_512_text.as_str(), false)
            .expect("encode long verified Potion input");
        let fast_long = embedder
            .tokenizer
            .encode_fast(long_over_512_text.as_str(), false)
            .expect("encode_fast long verified Potion input");
        assert!(
            embedder.tokenizer.get_truncation().is_none(),
            "registered Potion tokenizer must preserve its configured no-truncation policy"
        );
        assert!(
            former_long.len() > 512,
            "long verified Potion input must exceed the former 512-token synthetic boundary"
        );
        assert_eq!(
            fast_long.len(),
            former_long.len(),
            "verified Potion long-input token count diverged"
        );
        assert_eq!(
            fast_long.get_ids(),
            former_long.get_ids(),
            "verified Potion long-input token IDs diverged"
        );
        assert!(
            former_long.get_overflowing().is_empty() && fast_long.get_overflowing().is_empty(),
            "configured no-truncation policy must not emit overflow encodings"
        );

        let texts = &crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        let former_vectors = texts
            .iter()
            .map(|text| {
                assert_fast_ids_and_former_vector_bits(
                    &embedder,
                    text,
                    "verified Potion conformance input",
                );
                former_embed_sync(&embedder, text)
            })
            .collect::<Vec<_>>();
        let vectors = embedder
            .embed_batch_sync(texts)
            .expect("embed bounded conformance corpus");
        assert_eq!(vectors.len(), former_vectors.len());
        for (index, (actual, expected)) in vectors.iter().zip(&former_vectors).enumerate() {
            assert_f32_bits_eq(
                actual,
                expected,
                &format!("verified Potion batch order or vector bits changed at index={index}"),
            );
        }
        let observed = frankensearch_core::generation::GoldenVectorCertificateV1::from_exact_f32(
            texts, &vectors,
        )
        .expect("compute exact conformance certificate");
        let expected = manifest.execution.golden_vectors;
        assert_eq!(
            observed, expected,
            "Model2Vec output bits drifted from the registered producer certificate"
        );
    }

    // ── Debug impl ─────────────────────────────────────────────────────

    #[test]
    fn debug_does_not_dump_embeddings() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        let debug = format!("{embedder:?}");
        assert!(debug.contains("Model2VecEmbedder"));
        assert!(debug.contains("dimensions: 8"));
        assert!(debug.contains("vocab_size: 12"));
        // Must NOT contain actual embedding data
        assert!(!debug.contains("0.1"));
    }

    // ── Tensor Discovery ───────────────────────────────────────────────

    #[test]
    fn tensor_discovery_finds_standard_name() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 4, 2);

        // The test model uses "embeddings" as tensor name
        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert_eq!(embedder.vocab_size, 4);
    }

    #[test]
    fn tensor_discovery_single_tensor_fallback() {
        let dir = tempfile::tempdir().unwrap();

        // Create tokenizer
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [],
            "model": {
                "type": "WordLevel",
                "vocab": {"hello": 0, "world": 1},
                "unk_token": "hello"
            }
        });
        fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::to_string(&tokenizer_json).unwrap(),
        )
        .unwrap();

        // Create safetensors with a non-standard tensor name
        let mut data = vec![0u8; 2 * 3 * 4]; // 2 rows × 3 dims × 4 bytes
        for (i, chunk) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let val = i as f32;
            chunk.copy_from_slice(&val.to_le_bytes());
        }

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "my_custom_tensor_name".to_owned(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &data)
                .unwrap(),
        );

        let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
        fs::write(dir.path().join("model.safetensors"), serialized).unwrap();

        // Should fall back to the single tensor
        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert_eq!(embedder.vocab_size, 2);
        assert_eq!(embedder.dimensions, 3);
    }

    // ── Model Directory Search ─────────────────────────────────────────

    #[test]
    fn has_required_files_positive() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 4, 2);
        assert!(has_required_files(dir.path()));
    }

    #[test]
    fn has_required_files_negative() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!has_required_files(dir.path()));
    }

    // ── Parse Matrix ───────────────────────────────────────────────────

    #[test]
    fn parse_f32_matrix_correct() {
        // 2 rows × 2 dims = 16 bytes
        let data: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let matrix = parse_f32_matrix(&data, 2, 2).unwrap();
        assert_eq!(matrix.len(), 4);
        assert_eq!(&matrix[0..2], &[1.0, 2.0]);
        assert_eq!(&matrix[2..4], &[3.0, 4.0]);
    }

    #[test]
    fn parse_f32_matrix_too_short() {
        let data = vec![0u8; 4]; // Only 1 float, need more
        let result = parse_f32_matrix(&data, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn parse_f32_matrix_too_long() {
        // 16 bytes expected for [2 x 2] f32, plus one trailing garbage byte
        let mut data: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        data.push(0xAA);

        let result = parse_f32_matrix(&data, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn parse_f32_matrix_size_overflow() {
        let result = parse_f32_matrix(&[], usize::MAX, 2);
        assert!(result.is_err());
    }

    // ── GH #46: streamed matrix load and the process-wide instance cache ──

    /// Serialises the real-model tests. They all point at the SAME directory
    /// (`POTION_FIXTURE_DIR`), so the per-directory load counter is only
    /// deterministic while one of them runs at a time.
    fn real_model_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Decode a safetensors file the way the loader did BEFORE #46: read the
    /// whole file, deserialize over that buffer, then materialise the matrix.
    /// This is the oracle every bit-identity assertion below compares against.
    fn whole_buffer_decode(model_dir: &Path) -> (usize, usize, Vec<f32>) {
        let bytes = fs::read(model_dir.join("model.safetensors")).unwrap();
        let safetensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let names = safetensors.names();
        let tensor_name = discover_tensor_name(&names).expect("oracle: no embedding tensor");
        let tensor = safetensors.tensor(&tensor_name).unwrap();
        let shape = tensor.shape();
        assert_eq!(shape.len(), 2, "oracle: expected a 2D tensor");
        let (vocab_size, dimensions) = (shape[0], shape[1]);
        let values = parse_f32_matrix(tensor.data(), vocab_size, dimensions).unwrap();
        (vocab_size, dimensions, values)
    }

    #[test]
    fn streamed_matrix_is_bit_identical_to_whole_buffer_decode() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load_with_name(dir.path(), "test-model").unwrap();
        let (vocab_size, dimensions, expected) = whole_buffer_decode(dir.path());

        assert_eq!(embedder.vocab_size, vocab_size);
        assert_eq!(embedder.dimensions, dimensions);
        assert_f32_bits_eq(
            &embedder.embeddings,
            &expected,
            "streamed matrix must decode to the same bits as the pre-#46 whole-buffer path",
        );
    }

    #[test]
    fn streamed_matrix_survives_a_chunk_boundary() {
        // MATRIX_READ_CHUNK_BYTES is 1 MiB; make the matrix straddle it so the
        // multi-chunk path, not just the single-read path, is exercised.
        let dimensions = 256;
        let vocab_size = (MATRIX_READ_CHUNK_BYTES / (dimensions * 4)) * 2 + 3;
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), vocab_size, dimensions);

        let embedder = Model2VecEmbedder::load_with_name(dir.path(), "chunked").unwrap();
        let (_, _, expected) = whole_buffer_decode(dir.path());
        assert!(
            expected.len() * 4 > MATRIX_READ_CHUNK_BYTES,
            "fixture must be larger than one read chunk"
        );
        assert_f32_bits_eq(
            &embedder.embeddings,
            &expected,
            "a matrix spanning several read chunks must decode identically",
        );
    }

    #[test]
    fn shared_load_builds_the_model_once_per_process() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);
        let key = std::fs::canonicalize(dir.path()).unwrap();

        assert_eq!(model2vec_full_loads_for(&key), 0);

        let first = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared").unwrap();
        assert_eq!(
            model2vec_full_loads_for(&key),
            1,
            "the first shared load must build the model"
        );

        let second = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared").unwrap();
        assert_eq!(
            model2vec_full_loads_for(&key),
            1,
            "the second load in the same process must not re-read the matrix or rebuild the tokenizer"
        );
        assert!(
            Arc::ptr_eq(&first, &second),
            "both callers must share one instance"
        );

        // And a third, to prove the hit is not a one-off.
        let third = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared").unwrap();
        assert_eq!(model2vec_full_loads_for(&key), 1);
        assert!(Arc::ptr_eq(&first, &third));
    }

    #[test]
    fn shared_load_is_bit_identical_to_a_direct_load() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let direct = Model2VecEmbedder::load_with_name(dir.path(), "shared-parity").unwrap();
        let shared = Model2VecEmbedder::load_shared_with_name(dir.path(), "shared-parity").unwrap();

        assert_f32_bits_eq(
            &shared.embeddings,
            &direct.embeddings,
            "a shared load must carry the same matrix bits as a direct one",
        );
        for text in ["hello world", "rust search", "", "totally-unseen-token"] {
            assert_f32_bits_eq(
                &shared.embed_sync(text).unwrap(),
                &direct.embed_sync(text).unwrap(),
                "shared and direct embedders must produce identical embeddings",
            );
        }
    }

    #[test]
    fn shared_load_reloads_after_the_last_reference_drops() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);
        let key = std::fs::canonicalize(dir.path()).unwrap();

        {
            let held = Model2VecEmbedder::load_shared_with_name(dir.path(), "weak").unwrap();
            assert_eq!(model2vec_full_loads_for(&key), 1);
            drop(held);
        }

        // The cache holds only a Weak, so the 512 MB matrix is not pinned for
        // the life of the process once nobody wants it.
        let reloaded = Model2VecEmbedder::load_shared_with_name(dir.path(), "weak").unwrap();
        assert_eq!(
            model2vec_full_loads_for(&key),
            2,
            "a released model must be rebuilt on the next request, not resurrected"
        );
        assert_eq!(reloaded.vocab_size, 12);
    }

    #[test]
    fn shared_load_keeps_distinct_model_directories_apart() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        create_test_model(first_dir.path(), 12, 8);
        create_test_model(second_dir.path(), 10, 4);

        let first = Model2VecEmbedder::load_shared_with_name(first_dir.path(), "shared").unwrap();
        let second = Model2VecEmbedder::load_shared_with_name(second_dir.path(), "shared").unwrap();

        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(first.dimensions, 8);
        assert_eq!(second.dimensions, 4);
    }

    #[test]
    fn shared_load_keeps_distinct_display_names_apart() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        // The display name is part of the cache key on purpose: returning an
        // instance whose `name` disagrees with the caller's request would be
        // worse than loading twice. Callers that want to share agree on it.
        let first = Model2VecEmbedder::load_shared_with_name(dir.path(), "alpha").unwrap();
        let second = Model2VecEmbedder::load_shared_with_name(dir.path(), "beta").unwrap();

        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(first.name, "alpha");
        assert_eq!(second.name, "beta");
    }

    #[test]
    fn load_rejects_a_truncated_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let path = dir.path().join("model.safetensors");
        let bytes = fs::read(&path).unwrap();
        fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();

        let error = Model2VecEmbedder::load_with_name(dir.path(), "truncated").unwrap_err();
        assert!(
            error.to_string().contains("length mismatch"),
            "a truncated artifact must be refused by the completeness check, got: {error}"
        );
    }

    #[test]
    fn load_rejects_a_trailing_garbage_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let path = dir.path().join("model.safetensors");
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0xAA; 16]);
        fs::write(&path, &bytes).unwrap();

        let error = Model2VecEmbedder::load_with_name(dir.path(), "overlong").unwrap_err();
        assert!(
            error.to_string().contains("length mismatch"),
            "an over-long artifact must be refused too, got: {error}"
        );
    }

    #[test]
    fn load_rejects_an_absurd_header_length_prefix() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let path = dir.path().join("model.safetensors");
        let mut bytes = fs::read(&path).unwrap();
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&path, &bytes).unwrap();

        let error = Model2VecEmbedder::load_with_name(dir.path(), "huge-header").unwrap_err();
        assert!(
            error.to_string().contains("exceeds"),
            "a corrupt length prefix must be bounded before it becomes an allocation, got: {error}"
        );
    }

    #[test]
    fn load_rejects_a_non_f32_embedding_tensor() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        // Same name and rank, different dtype. The pre-#46 path caught this
        // only as a byte-count mismatch; the dtype is now checked outright.
        let data = vec![0_u8; 12 * 8 * 4];
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "embeddings".to_owned(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::I32, vec![12, 8], &data)
                .unwrap(),
        );
        let serialized = safetensors::tensor::serialize(&tensors, None).unwrap();
        fs::write(dir.path().join("model.safetensors"), serialized).unwrap();

        let error = Model2VecEmbedder::load_with_name(dir.path(), "int-tensor").unwrap_err();
        assert!(
            error.to_string().contains("F32"),
            "a non-F32 tensor must be named as such, got: {error}"
        );
    }

    /// Bit-identity on the REAL artifact. A synthetic fixture cannot prove
    /// this: it has one small tensor, no added/special tokens and no
    /// multi-chunk read. Run with:
    ///
    /// ```text
    /// POTION_FIXTURE_DIR=$HOME/.local/share/frankensearch/models/potion-multilingual-128M \
    ///   cargo test -p frankensearch-embed --features model2vec --lib -- --ignored real_model
    /// ```
    #[test]
    #[ignore = "requires a real potion model dir via POTION_FIXTURE_DIR"]
    fn real_model_streamed_matrix_is_bit_identical_to_whole_buffer_decode() {
        let _serialized = real_model_test_lock()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let dir = std::env::var("POTION_FIXTURE_DIR")
            .expect("set POTION_FIXTURE_DIR to a potion-multilingual-128M directory");
        let dir = Path::new(&dir);

        let embedder = Model2VecEmbedder::load_with_name(dir, DEFAULT_MODEL_NAME).unwrap();
        let (vocab_size, dimensions, expected) = whole_buffer_decode(dir);

        assert_eq!(embedder.vocab_size, vocab_size);
        assert_eq!(embedder.dimensions, dimensions);
        assert_eq!(embedder.embeddings.len(), expected.len());
        assert_f32_bits_eq(
            &embedder.embeddings,
            &expected,
            "the real 512 MB matrix must stream to exactly the bits the whole-buffer path produced",
        );

        // And the embeddings the matrix feeds, over the registered
        // conformance corpus plus the multilingual/emoji cases.
        let texts = [
            "hello world",
            "semantic search finds related ideas",
            "identifier fsvi_v2",
            "naive cafe Tokyo",
            "Café naïve — 東京 🦀",
            "",
        ];
        let reference = Model2VecEmbedder {
            tokenizer: Tokenizer::from_file(dir.join("tokenizer.json")).unwrap(),
            embeddings: expected,
            dimensions,
            vocab_size,
            name: DEFAULT_MODEL_NAME.to_owned(),
            model_dir: dir.to_owned(),
            identity: embedder.identity.clone(),
            last_tokenizer_route_was_offset_free: AtomicBool::new(false),
        };
        for text in texts {
            assert_f32_bits_eq(
                &embedder.embed_sync(text).unwrap(),
                &reference.embed_sync(text).unwrap(),
                "embeddings must be bit-identical before and after the streamed load",
            );
        }
    }

    /// Reuse on the REAL artifact: the second load in a process must not
    /// re-read the 512 MB matrix or rebuild the 500k-piece tokenizer. Same
    /// env gate as above.
    #[test]
    #[ignore = "requires a real potion model dir via POTION_FIXTURE_DIR"]
    fn real_model_shared_load_is_built_once_per_process() {
        let _serialized = real_model_test_lock()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let dir = std::env::var("POTION_FIXTURE_DIR")
            .expect("set POTION_FIXTURE_DIR to a potion-multilingual-128M directory");
        let dir = Path::new(&dir);
        let key = std::fs::canonicalize(dir).unwrap();

        let before = model2vec_full_loads_for(&key);
        let first = Model2VecEmbedder::load_shared_with_name(dir, DEFAULT_MODEL_NAME).unwrap();
        assert_eq!(model2vec_full_loads_for(&key), before + 1);

        let cold = std::time::Instant::now();
        let second = Model2VecEmbedder::load_shared_with_name(dir, DEFAULT_MODEL_NAME).unwrap();
        let warm_elapsed = cold.elapsed();

        assert_eq!(
            model2vec_full_loads_for(&key),
            before + 1,
            "the second real-model load must be served from the process cache"
        );
        assert!(Arc::ptr_eq(&first, &second));
        assert!(
            warm_elapsed < std::time::Duration::from_millis(250),
            "a cache hit must not cost a load; took {warm_elapsed:?}"
        );
    }
}
