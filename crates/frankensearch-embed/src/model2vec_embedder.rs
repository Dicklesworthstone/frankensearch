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

use std::fmt;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use rayon::prelude::*;
use safetensors::SafeTensors;
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
            let verified = ModelArtifactManifestV1::potion_128m_native()?.verify_dir(model_dir)?;
            let identity = verified.identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
            validate_registered_execution_contract(&identity)?;
            Self::load_preverified(model_dir, name, identity)
        }
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

        // Load safetensors
        let safetensors_path = model_dir.join("model.safetensors");
        let safetensors_data =
            std::fs::read(&safetensors_path).map_err(|e| SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: Box::new(e),
            })?;

        let safetensors = SafeTensors::deserialize(&safetensors_data).map_err(|e| {
            SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: format!("failed to parse safetensors: {e}").into(),
            }
        })?;

        // Discover the embedding tensor
        let tensor_name = discover_tensor_name(&safetensors).ok_or_else(|| {
            let available: Vec<_> = safetensors.names().into_iter().collect();
            SearchError::ModelLoadFailed {
                path: safetensors_path.clone(),
                source: format!(
                    "no embedding tensor found. Tried: {TENSOR_NAME_CANDIDATES:?}. Available: {available:?}"
                )
                .into(),
            }
        })?;

        let tensor =
            safetensors
                .tensor(&tensor_name)
                .map_err(|e| SearchError::ModelLoadFailed {
                    path: safetensors_path.clone(),
                    source: format!("failed to get tensor '{tensor_name}': {e}").into(),
                })?;

        // Validate tensor shape
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(SearchError::ModelLoadFailed {
                path: safetensors_path,
                source: format!(
                    "expected 2D tensor, got {}D with shape {shape:?}",
                    shape.len()
                )
                .into(),
            });
        }

        let vocab_size = shape[0];
        let dimensions = shape[1];
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

        // Parse the raw f32 data into the embedding matrix
        let embeddings = parse_f32_matrix(tensor.data(), vocab_size, dimensions).map_err(|e| {
            SearchError::ModelLoadFailed {
                path: safetensors_path,
                source: e.into(),
            }
        })?;

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

        // Tokenize
        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenization failed: {e}").into(),
                })?;

        let token_ids = encoding.get_ids();
        if token_ids.is_empty() {
            return Ok(vec![0.0; self.dimensions]);
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
            return Ok(vec![0.0; self.dimensions]);
        }

        // Compute mean
        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0 / count as f32;
        for s in &mut sum {
            *s *= inv;
        }

        // L2 normalize to unit length
        normalize_in_place(&mut sum);
        Ok(sum)
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

fn normalize_in_place(vec: &mut [f32]) {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in vec {
            *x *= inv_norm;
        }
    } else {
        vec.fill(0.0);
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

impl Embedder for Model2VecEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        // Model2Vec is pure computation (~0.57ms) — no cancellation check needed
        Box::pin(async move { self.embed_sync(text) })
    }

    fn embed_batch<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move { self.embed_batch_sync(texts) })
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

/// Discover the embedding tensor name in a safetensors file.
///
/// Tries known names first, then falls back to using the only tensor
/// if the file contains exactly one.
fn discover_tensor_name(safetensors: &SafeTensors<'_>) -> Option<String> {
    let names = safetensors.names();

    // Try known candidate names
    for candidate in &TENSOR_NAME_CANDIDATES {
        if names.iter().any(|n| n == candidate) {
            return Some((*candidate).to_owned());
        }
    }

    // Fallback: if exactly one tensor exists, use it regardless of name
    if names.len() == 1 {
        return Some(names[0].to_owned());
    }

    None
}

/// Parse raw bytes from a safetensors tensor into a flat `Vec<f32>` matrix.
///
/// Expects little-endian f32 data with shape `[vocab_size, dimensions]`.
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
                .map(|t| embedder.embed_sync(t).unwrap())
                .collect();
            let batched = embedder.embed_batch_sync(&texts).unwrap();

            assert_eq!(batched.len(), serial.len(), "len at n={batch_size}");
            for (b, s) in batched.iter().zip(&serial) {
                assert_eq!(
                    b, s,
                    "embed_batch_sync diverged from serial at n={batch_size}"
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

    /// The former `embed_sync` pooling and finish sequence, retained independently
    /// of the production gather helper for exact native-256 parity checks.
    fn former_embed_sync(embedder: &Model2VecEmbedder, text: &str) -> Vec<f32> {
        if text.is_empty() {
            return vec![0.0; embedder.dimensions];
        }

        let encoding = embedder.tokenizer.encode(text, false).unwrap();
        let mut sum = vec![0.0_f32; embedder.dimensions];
        let mut count = 0_usize;
        for &token_id in encoding.get_ids() {
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

        #[allow(clippy::cast_precision_loss)]
        let inv = 1.0 / count as f32;
        for value in &mut sum {
            *value *= inv;
        }
        normalize_in_place(&mut sum);
        sum
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
            if !text.is_empty() {
                let token_count = embedder.tokenizer.encode(&text, false).unwrap().len();
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
        let texts = &crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        let vectors = embedder
            .embed_batch_sync(texts)
            .expect("embed bounded conformance corpus");
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
}
