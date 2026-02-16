//! `Model2Vec` static token embedding for the fast tier.
//!
//! Wraps potion-multilingual-128M (and compatible `Model2Vec` models) which are
//! static token embedding models: they look up pre-computed per-token embeddings
//! and mean-pool them. No transformer inference, no GPU needed.
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
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use tracing::instrument;

use crate::model_registry::{ensure_model_storage_layout, model_directory_variants};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture, l2_normalize};

/// Required files for a `Model2Vec` model.
const REQUIRED_FILES: [&str; 2] = ["tokenizer.json", "model.safetensors"];

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
}

impl fmt::Debug for Model2VecEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model2VecEmbedder")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("vocab_size", &self.vocab_size)
            .field("model_dir", &self.model_dir)
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
    #[instrument(skip_all, fields(model_dir = %model_dir.as_ref().display()))]
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// Load a `Model2Vec` model with a custom name.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` if required files are missing.
    /// Returns `SearchError::ModelLoadFailed` if files exist but cannot be parsed.
    pub fn load_with_name(model_dir: impl AsRef<Path>, name: &str) -> SearchResult<Self> {
        let model_dir = model_dir.as_ref();

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

        // Parse the raw f32 data into the embedding matrix
        let embeddings = parse_f32_matrix(tensor.data(), vocab_size, dimensions).map_err(|e| {
            SearchError::ModelLoadFailed {
                path: safetensors_path,
                source: e.into(),
            }
        })?;

        tracing::info!(
            name,
            vocab_size,
            dimensions,
            tensor_name = tensor_name.as_str(),
            "Model2Vec model loaded"
        );

        Ok(Self {
            tokenizer,
            embeddings,
            dimensions,
            vocab_size,
            name: name.to_owned(),
            model_dir: model_dir.to_owned(),
        })
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
        let mut count: usize = 0;

        for &token_id in token_ids {
            let idx = token_id as usize;
            if idx < self.vocab_size {
                let start = idx * self.dimensions;
                let end = start + self.dimensions;
                let row = &self.embeddings[start..end];
                for (s, r) in sum.iter_mut().zip(row.iter()) {
                    *s += r;
                }
                count += 1;
            }
            // Out-of-vocabulary tokens are silently skipped (common in Model2Vec)
        }

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
        Ok(l2_normalize(&sum))
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
        Box::pin(async move {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed_sync(text)?);
            }
            Ok(results)
        })
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

    fn supports_mrl(&self) -> bool {
        // Model2Vec models support Matryoshka truncation
        true
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
        if names.iter().any(|n| n.as_str() == *candidate) {
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

    // Parse bytes in 4-byte chunks
    for chunk in data.chunks_exact(4) {
        let bytes: [u8; 4] = chunk
            .try_into()
            .map_err(|_| "byte slice conversion failed".to_string())?;
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
    if let Some(cache_dir) = dirs::cache_dir() {
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

        let serialized = safetensors::tensor::serialize(&tensors, &None).unwrap();
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
    fn trait_supports_mrl() {
        let dir = tempfile::tempdir().unwrap();
        create_test_model(dir.path(), 12, 8);

        let embedder = Model2VecEmbedder::load(dir.path()).unwrap();
        assert!(embedder.supports_mrl());
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
        for (i, chunk) in data.chunks_exact_mut(4).enumerate() {
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

        let serialized = safetensors::tensor::serialize(&tensors, &None).unwrap();
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
