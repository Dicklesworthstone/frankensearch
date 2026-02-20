//! FastEmbed-based ONNX embedders.
//!
//! This module loads ONNX + tokenizer assets from a local model directory and
//! performs semantic embedding inference through `fastembed`.
//!
//! Supports multiple models via [`OnnxEmbedderConfig`]:
//! - `MiniLM` (baseline, 384 dimensions)
//! - Snowflake Arctic Embed S (bake-off candidate, 384 dimensions)
//! - Nomic Embed Text v1.5 (bake-off candidate, 768 dimensions)
//!
//! Required files:
//! - `onnx/model.onnx` (preferred, current layout) OR `model.onnx` (legacy layout)
//! - `tokenizer.json`
//! - `config.json`
//! - `special_tokens_map.json`
//! - `tokenizer_config.json`

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use asupersync::sync::{LockError, Mutex};
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use tracing::instrument;

use crate::model_registry::{ensure_model_storage_layout, model_directory_variants};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};

/// Default quality-tier model directory name.
pub const DEFAULT_MODEL_NAME: &str = "all-MiniLM-L6-v2";

/// `HuggingFace` model ID for MiniLM-L6-v2.
pub const DEFAULT_HF_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Expected `MiniLM` output dimension.
pub const DEFAULT_DIMENSION: usize = 384;

/// Configuration for loading an ONNX embedder with custom parameters.
#[derive(Debug, Clone)]
pub struct OnnxEmbedderConfig {
    /// Unique embedder ID (e.g., `"minilm-384"`).
    pub embedder_id: String,
    /// Model identifier for logging (e.g., `"all-minilm-l6-v2"`).
    pub model_id: String,
    /// Output embedding dimension.
    pub dimension: usize,
    /// Pooling strategy.
    pub pooling: Pooling,
}

impl Default for OnnxEmbedderConfig {
    fn default() -> Self {
        Self {
            embedder_id: "minilm-384".to_string(),
            model_id: DEFAULT_MODEL_NAME.to_string(),
            dimension: DEFAULT_DIMENSION,
            pooling: Pooling::Mean,
        }
    }
}

impl OnnxEmbedderConfig {
    /// Return a config for a known embedder name, or `None` for unknown names.
    ///
    /// Recognised names: `"minilm"`, `"snowflake-arctic-s"`, `"nomic-embed"`.
    #[must_use]
    pub fn for_name(embedder_name: &str) -> Option<Self> {
        match embedder_name {
            "minilm" => Some(Self::default()),
            "snowflake-arctic-s" => Some(Self {
                embedder_id: "snowflake-arctic-s-384".to_string(),
                model_id: "snowflake-arctic-embed-s".to_string(),
                dimension: 384,
                pooling: Pooling::Mean,
            }),
            "nomic-embed" => Some(Self {
                embedder_id: "nomic-embed-768".to_string(),
                model_id: "nomic-embed-text-v1.5".to_string(),
                dimension: 768,
                pooling: Pooling::Mean,
            }),
            _ => None,
        }
    }
}

const MODEL_ONNX_SUBDIR: &str = "onnx/model.onnx";
const MODEL_ONNX_LEGACY: &str = "model.onnx";

const TOKENIZER_JSON: &str = "tokenizer.json";
const CONFIG_JSON: &str = "config.json";
const SPECIAL_TOKENS_JSON: &str = "special_tokens_map.json";
const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

const REQUIRED_NON_MODEL_FILES: [&str; 4] = [
    TOKENIZER_JSON,
    CONFIG_JSON,
    SPECIAL_TOKENS_JSON,
    TOKENIZER_CONFIG_JSON,
];

/// FastEmbed-backed ONNX embedder.
///
/// Supports any ONNX model that produces fixed-dimension embeddings.
/// `TextEmbedding` is wrapped in a cancel-aware `asupersync::sync::Mutex`
/// because ONNX sessions are not safe for concurrent mutable access.
pub struct FastEmbedEmbedder {
    model: Mutex<TextEmbedding>,
    name: String,
    dimension: usize,
    model_dir: PathBuf,
}

impl fmt::Debug for FastEmbedEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastEmbedEmbedder")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .field("model_dir", &self.model_dir)
            .finish_non_exhaustive()
    }
}

impl FastEmbedEmbedder {
    /// Load `all-MiniLM-L6-v2` from a local directory.
    ///
    /// `model_dir` may be either:
    /// - the model directory itself (contains tokenizer/config + `onnx/model.onnx`)
    /// - a parent directory containing `<model_name>/`
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX/session initialization fails.
    #[instrument(skip_all, fields(model_dir = %model_dir.as_ref().display()))]
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// Load a `FastEmbed` model with a custom model directory name.
    ///
    /// Uses the default `MiniLM` configuration (384 dimensions, mean pooling).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX/session initialization fails.
    pub fn load_with_name(model_dir: impl AsRef<Path>, name: &str) -> SearchResult<Self> {
        Self::load_with_config(
            model_dir,
            OnnxEmbedderConfig {
                embedder_id: name.to_owned(),
                model_id: name.to_owned(),
                ..OnnxEmbedderConfig::default()
            },
        )
    }

    /// Load an ONNX embedder with custom configuration.
    ///
    /// This is the most flexible constructor, allowing callers to specify
    /// dimension, pooling strategy, and identifiers for any supported model.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX/session initialization fails.
    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        config: OnnxEmbedderConfig,
    ) -> SearchResult<Self> {
        let name = &config.model_id;
        let expected_dim = config.dimension;
        let model_dir = resolve_model_dir(model_dir.as_ref(), name)?;
        let model_file =
            select_model_file(&model_dir).ok_or_else(|| SearchError::ModelNotFound {
                name: format!("{name} (missing {MODEL_ONNX_SUBDIR} or {MODEL_ONNX_LEGACY})"),
            })?;

        for filename in &REQUIRED_NON_MODEL_FILES {
            let path = model_dir.join(filename);
            if !path.is_file() {
                return Err(SearchError::ModelNotFound {
                    name: format!("{name} (missing {filename} in {})", model_dir.display()),
                });
            }
        }

        let model_bytes = read_required(&model_file)?;
        let tokenizer_file = read_required(&model_dir.join(TOKENIZER_JSON))?;
        let config_file = read_required(&model_dir.join(CONFIG_JSON))?;
        let special_tokens_map_file = read_required(&model_dir.join(SPECIAL_TOKENS_JSON))?;
        let tokenizer_config_file = read_required(&model_dir.join(TOKENIZER_CONFIG_JSON))?;

        let tokenizer_files = TokenizerFiles {
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
        };

        let mut user_model = UserDefinedEmbeddingModel::new(model_bytes, tokenizer_files);
        user_model.pooling = Some(config.pooling);

        let init_options = InitOptionsUserDefined::new();
        let mut text_embedding = TextEmbedding::try_new_from_user_defined(user_model, init_options)
            .map_err(|e| SearchError::ModelLoadFailed {
                path: model_dir.clone(),
                source: format!("failed to initialize FastEmbed model: {e}").into(),
            })?;

        // Fail fast on model/schema mismatch rather than deferring to first query.
        let probe = text_embedding
            .embed(vec!["dimension probe"], None)
            .map_err(|e| SearchError::ModelLoadFailed {
                path: model_dir.clone(),
                source: format!("failed to run embedding probe: {e}").into(),
            })?;
        let probe_dim = probe.first().map_or(0, Vec::len);
        if probe_dim != expected_dim {
            return Err(SearchError::ModelLoadFailed {
                path: model_dir,
                source: format!(
                    "dimension mismatch for {name}: expected {expected_dim}, got {probe_dim}"
                )
                .into(),
            });
        }

        tracing::info!(
            model = %name,
            dimension = expected_dim,
            model_dir = %model_dir.display(),
            "FastEmbed model loaded"
        );

        Ok(Self {
            model: Mutex::new(text_embedding),
            name: config.embedder_id,
            dimension: expected_dim,
            model_dir,
        })
    }

    /// Embed a single non-empty string.
    async fn embed_non_empty(&self, cx: &Cx, text: &str) -> SearchResult<Vec<f32>> {
        let mut model = self
            .model
            .lock(cx)
            .await
            .map_err(|err| map_lock_error(&self.name, "fastembed.embed", err))?;

        let mut embeddings =
            model
                .embed(vec![text], None)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("fastembed inference failed: {e}").into(),
                })?;

        let mut embedding = embeddings
            .pop()
            .ok_or_else(|| SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: "fastembed returned no embedding".into(),
            })?;

        if embedding.len() != self.dimension {
            return Err(SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dimension,
                    embedding.len()
                )
                .into(),
            });
        }

        normalize_in_place(&mut embedding);
        Ok(embedding)
    }

    /// Embed a batch of non-empty strings.
    async fn embed_batch_non_empty(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock(cx)
            .await
            .map_err(|err| map_lock_error(&self.name, "fastembed.embed_batch", err))?;

        let mut embeddings =
            model
                .embed(texts, None)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("fastembed batch inference failed: {e}").into(),
                })?;

        if embeddings.len() != texts.len() {
            return Err(SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!(
                    "batch size mismatch: requested {}, got {}",
                    texts.len(),
                    embeddings.len()
                )
                .into(),
            });
        }

        for embedding in &mut embeddings {
            if embedding.len() != self.dimension {
                return Err(SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!(
                        "dimension mismatch: expected {}, got {}",
                        self.dimension,
                        embedding.len()
                    )
                    .into(),
                });
            }
            normalize_in_place(embedding);
        }
        Ok(embeddings)
    }

    /// Directory containing model assets.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
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

impl Embedder for FastEmbedEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            if text.is_empty() {
                return Ok(vec![0.0; self.dimension]);
            }
            self.embed_non_empty(cx, text).await
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let mut output = vec![vec![0.0; self.dimension]; texts.len()];
            let mut non_empty_indices = Vec::with_capacity(texts.len());
            let mut non_empty_texts = Vec::with_capacity(texts.len());

            for (idx, text) in texts.iter().enumerate() {
                if !text.is_empty() {
                    non_empty_indices.push(idx);
                    non_empty_texts.push(*text);
                }
            }

            if non_empty_texts.is_empty() {
                return Ok(output);
            }

            let normalized = self.embed_batch_non_empty(cx, &non_empty_texts).await?;
            for (slot_idx, embedding) in non_empty_indices.into_iter().zip(normalized.into_iter()) {
                output[slot_idx] = embedding;
            }
            Ok(output)
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
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
        ModelCategory::TransformerEmbedder
    }
}

/// Search for a `FastEmbed` model directory in standard locations.
///
/// Checks these paths in order:
/// 1. `$FRANKENSEARCH_MODEL_DIR/<model_name>/` then `$FRANKENSEARCH_MODEL_DIR`
/// 2. `$XDG_DATA_HOME/frankensearch/models/<model_name>/`
/// 3. `~/.local/share/frankensearch/models/<model_name>/` (or macOS
///    `~/Library/Application Support/frankensearch/models/<model_name>/`)
/// 4. `~/.cache/huggingface/hub/models--<hf_id>/snapshots/*/`
///
/// Returns `None` if no directory with required files is found.
#[must_use]
pub fn find_model_dir(model_name: &str) -> Option<PathBuf> {
    find_model_dir_with_hf_id(model_name, DEFAULT_HF_ID)
}

/// Search for a `FastEmbed` model directory with a specific `HuggingFace` ID.
#[must_use]
pub fn find_model_dir_with_hf_id(model_name: &str, hf_id: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(dir) = std::env::var("FRANKENSEARCH_MODEL_DIR") {
        let base = PathBuf::from(dir);
        for variant in model_directory_variants(model_name) {
            candidates.push(base.join(variant));
        }
        candidates.push(base);
    }

    let model_root = ensure_model_storage_layout();
    for variant in model_directory_variants(model_name) {
        candidates.push(model_root.join(variant));
    }

    if let Some(cache_dir) = dirs::cache_dir() {
        let hf_dir = cache_dir
            .join("huggingface/hub")
            .join(format!("models--{}", hf_id.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = fs::read_dir(hf_dir) {
            for entry in entries.flatten() {
                candidates.push(entry.path());
            }
        }
    }

    candidates.into_iter().find(|dir| has_required_files(dir))
}

fn map_lock_error(model: &str, phase: &str, error: LockError) -> SearchError {
    match error {
        LockError::Cancelled => SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: "mutex lock cancelled".to_owned(),
        },
        LockError::Poisoned => SearchError::EmbeddingFailed {
            model: model.to_owned(),
            source: "fastembed mutex poisoned".into(),
        },
    }
}

fn read_required(path: &Path) -> SearchResult<Vec<u8>> {
    fs::read(path).map_err(|e| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: Box::new(e),
    })
}

fn resolve_model_dir(base_dir: &Path, model_name: &str) -> SearchResult<PathBuf> {
    if has_required_files(base_dir) {
        return Ok(base_dir.to_path_buf());
    }

    let nested = base_dir.join(model_name);
    if has_required_files(&nested) {
        return Ok(nested);
    }

    Err(SearchError::ModelNotFound {
        name: format!(
            "{model_name} (missing required files in {} or {})",
            base_dir.display(),
            nested.display()
        ),
    })
}

fn select_model_file(model_dir: &Path) -> Option<PathBuf> {
    let modern = model_dir.join(MODEL_ONNX_SUBDIR);
    if modern.is_file() {
        return Some(modern);
    }

    let legacy = model_dir.join(MODEL_ONNX_LEGACY);
    if legacy.is_file() {
        return Some(legacy);
    }

    None
}

fn has_required_files(dir: &Path) -> bool {
    select_model_file(dir).is_some()
        && REQUIRED_NON_MODEL_FILES
            .iter()
            .all(|filename| dir.join(filename).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_stub_model_layout(dir: &Path, use_onnx_subdir: bool) {
        if use_onnx_subdir {
            std::fs::create_dir_all(dir.join("onnx")).unwrap();
            std::fs::write(dir.join("onnx/model.onnx"), b"stub-onnx").unwrap();
        } else {
            std::fs::write(dir.join("model.onnx"), b"stub-onnx").unwrap();
        }

        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("special_tokens_map.json"), "{}").unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    }

    #[test]
    fn has_required_files_accepts_modern_onnx_layout() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model_layout(temp.path(), true);
        assert!(has_required_files(temp.path()));
    }

    #[test]
    fn has_required_files_accepts_legacy_layout() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model_layout(temp.path(), false);
        assert!(has_required_files(temp.path()));
    }

    #[test]
    fn resolve_model_dir_accepts_direct_model_path() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model_layout(temp.path(), true);

        let resolved = resolve_model_dir(temp.path(), DEFAULT_MODEL_NAME).unwrap();
        assert_eq!(resolved, temp.path());
    }

    #[test]
    fn resolve_model_dir_accepts_parent_with_named_child() {
        let temp = tempfile::tempdir().unwrap();
        let child = temp.path().join(DEFAULT_MODEL_NAME);
        std::fs::create_dir_all(&child).unwrap();
        create_stub_model_layout(&child, true);

        let resolved = resolve_model_dir(temp.path(), DEFAULT_MODEL_NAME).unwrap();
        assert_eq!(resolved, child);
    }

    #[test]
    fn resolve_model_dir_errors_when_missing_files() {
        let temp = tempfile::tempdir().unwrap();
        let err = resolve_model_dir(temp.path(), DEFAULT_MODEL_NAME).unwrap_err();

        assert!(matches!(err, SearchError::ModelNotFound { .. }));
    }

    #[test]
    fn map_lock_error_cancelled_to_search_cancelled() {
        let err = map_lock_error("all-MiniLM-L6-v2", "fastembed.embed", LockError::Cancelled);
        assert!(matches!(err, SearchError::Cancelled { .. }));
    }

    #[test]
    fn map_lock_error_poisoned_to_embedding_failed() {
        let err = map_lock_error("all-MiniLM-L6-v2", "fastembed.embed", LockError::Poisoned);
        assert!(matches!(err, SearchError::EmbeddingFailed { .. }));
    }

    #[test]
    fn select_model_file_prefers_modern_path() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model_layout(temp.path(), true);
        std::fs::write(temp.path().join("model.onnx"), b"legacy").unwrap();

        let selected = select_model_file(temp.path()).unwrap();
        assert!(selected.ends_with(MODEL_ONNX_SUBDIR));
    }

    #[test]
    fn has_required_files_rejects_empty_directory() {
        let temp = tempfile::tempdir().unwrap();
        assert!(!has_required_files(temp.path()));
    }

    #[test]
    fn has_required_files_rejects_model_without_tokenizer() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(temp.path().join("onnx")).unwrap();
        std::fs::write(temp.path().join("onnx/model.onnx"), b"stub").unwrap();
        // Missing tokenizer.json, config.json, etc.
        assert!(!has_required_files(temp.path()));
    }

    #[test]
    fn has_required_files_rejects_tokenizer_without_model() {
        let temp = tempfile::tempdir().unwrap();
        // All non-model files present, but no model.onnx
        std::fs::write(temp.path().join("tokenizer.json"), "{}").unwrap();
        std::fs::write(temp.path().join("config.json"), "{}").unwrap();
        std::fs::write(temp.path().join("special_tokens_map.json"), "{}").unwrap();
        std::fs::write(temp.path().join("tokenizer_config.json"), "{}").unwrap();
        assert!(!has_required_files(temp.path()));
    }

    #[test]
    fn select_model_file_returns_none_for_empty_dir() {
        let temp = tempfile::tempdir().unwrap();
        assert!(select_model_file(temp.path()).is_none());
    }

    #[test]
    fn select_model_file_falls_back_to_legacy() {
        let temp = tempfile::tempdir().unwrap();
        // Only legacy model.onnx, no onnx/ subdir
        std::fs::write(temp.path().join("model.onnx"), b"legacy").unwrap();
        let selected = select_model_file(temp.path()).unwrap();
        assert!(selected.ends_with(MODEL_ONNX_LEGACY));
    }

    #[test]
    fn read_required_returns_error_for_missing_file() {
        let path = PathBuf::from("/nonexistent/path/model.onnx");
        let err = read_required(&path).unwrap_err();
        assert!(matches!(err, SearchError::ModelLoadFailed { .. }));
    }

    #[test]
    fn constants_have_expected_values() {
        assert_eq!(DEFAULT_MODEL_NAME, "all-MiniLM-L6-v2");
        assert_eq!(DEFAULT_DIMENSION, 384);
        assert!(DEFAULT_HF_ID.contains("MiniLM"));
    }

    #[test]
    fn map_lock_error_preserves_phase_string() {
        let err = map_lock_error("test-model", "test.phase", LockError::Cancelled);
        match err {
            SearchError::Cancelled { phase, .. } => {
                assert_eq!(phase, "test.phase");
            }
            _ => panic!("expected Cancelled variant"),
        }
    }

    #[test]
    fn map_lock_error_preserves_model_string() {
        let err = map_lock_error("custom-model", "embed", LockError::Poisoned);
        match err {
            SearchError::EmbeddingFailed { model, .. } => {
                assert_eq!(model, "custom-model");
            }
            _ => panic!("expected EmbeddingFailed variant"),
        }
    }

    #[test]
    fn resolve_model_dir_error_message_includes_paths() {
        let temp = tempfile::tempdir().unwrap();
        let err = resolve_model_dir(temp.path(), "my-model").unwrap_err();
        match err {
            SearchError::ModelNotFound { name } => {
                assert!(name.contains("my-model"), "error should include model name");
            }
            _ => panic!("expected ModelNotFound variant"),
        }
    }

    #[test]
    fn onnx_config_default_is_minilm() {
        let config = OnnxEmbedderConfig::default();
        assert_eq!(config.embedder_id, "minilm-384");
        assert_eq!(config.dimension, 384);
    }

    #[test]
    fn onnx_config_for_name_known_models() {
        let minilm = OnnxEmbedderConfig::for_name("minilm").unwrap();
        assert_eq!(minilm.embedder_id, "minilm-384");
        assert_eq!(minilm.dimension, 384);

        let snowflake = OnnxEmbedderConfig::for_name("snowflake-arctic-s").unwrap();
        assert_eq!(snowflake.embedder_id, "snowflake-arctic-s-384");
        assert_eq!(snowflake.dimension, 384);

        let nomic = OnnxEmbedderConfig::for_name("nomic-embed").unwrap();
        assert_eq!(nomic.embedder_id, "nomic-embed-768");
        assert_eq!(nomic.dimension, 768);
    }

    #[test]
    fn onnx_config_for_name_unknown_returns_none() {
        assert!(OnnxEmbedderConfig::for_name("unknown-model").is_none());
        assert!(OnnxEmbedderConfig::for_name("").is_none());
    }
}
