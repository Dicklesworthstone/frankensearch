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
//! - `onnx/model.onnx` at the exact registered manifest path
//! - `tokenizer.json`
//! - `config.json`
//! - `special_tokens_map.json`
//! - `tokenizer_config.json`

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use asupersync::Cx;
use asupersync::sync::{LockError, Mutex, OwnedMutexGuard};
use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use tracing::instrument;

use crate::model_manifest::{
    FASTEMBED_MAX_LENGTH_V1, FASTEMBED_OUTPUT_NORMALIZATION_V1, FASTEMBED_SEQUENCE_POLICY_V1,
    ModelArtifactManifestV1,
};
use crate::model_registry::{ensure_model_storage_layout, model_directory_variants};
use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};

/// Default quality-tier model directory name.
pub const DEFAULT_MODEL_NAME: &str = "all-MiniLM-L6-v2";

/// `HuggingFace` model ID for MiniLM-L6-v2.
pub const DEFAULT_HF_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Expected `MiniLM` output dimension.
pub const DEFAULT_DIMENSION: usize = 384;

/// Configuration for selecting a frozen, manifest-registered ONNX embedder.
///
/// `model_id`, `dimension`, and `pooling` must agree with the registered
/// execution contract. `embedder_id` remains a caller-facing display/registry
/// identifier and never establishes vector-space compatibility.
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
    /// Recognised names include the short registry aliases and complete frozen
    /// logical model IDs for `MiniLM`, Snowflake, and Nomic.
    #[must_use]
    pub fn for_name(embedder_name: &str) -> Option<Self> {
        match embedder_name {
            "minilm" | DEFAULT_MODEL_NAME | "all-minilm-l6-v2" => Some(Self::default()),
            "snowflake-arctic-s" | "snowflake-arctic-embed-s" => Some(Self {
                embedder_id: "snowflake-arctic-s-384".to_string(),
                model_id: "snowflake-arctic-embed-s".to_string(),
                dimension: 384,
                pooling: Pooling::Mean,
            }),
            "nomic-embed" | "nomic-embed-text-v1.5" => Some(Self {
                embedder_id: "nomic-embed-768".to_string(),
                model_id: "nomic-embed-text-v1.5".to_string(),
                dimension: 768,
                pooling: Pooling::Mean,
            }),
            _ => None,
        }
    }
}

fn frozen_manifest_for_config(
    config: &OnnxEmbedderConfig,
) -> SearchResult<ModelArtifactManifestV1> {
    let manifest = match config.model_id.as_str() {
        DEFAULT_MODEL_NAME | "all-minilm-l6-v2" => ModelArtifactManifestV1::minilm_fastembed(),
        "snowflake-arctic-embed-s" => ModelArtifactManifestV1::snowflake_fastembed(),
        "nomic-embed-text-v1.5" => ModelArtifactManifestV1::nomic_fastembed(),
        _model_id => Err(SearchError::InvalidConfig {
            field: "fastembed.model_id".to_owned(),
            value: "unregistered".to_owned(),
            reason: "model has no registered frozen artifact/execution manifest".to_owned(),
        }),
    }?;
    let manifest_dimension =
        usize::try_from(manifest.dimension).map_err(|_| SearchError::InvalidConfig {
            field: "fastembed.manifest.dimension".to_owned(),
            value: manifest.dimension.to_string(),
            reason: "registered dimension does not fit usize".to_owned(),
        })?;
    if config.dimension != manifest_dimension {
        return Err(SearchError::InvalidConfig {
            field: "fastembed.dimension".to_owned(),
            value: config.dimension.to_string(),
            reason: format!(
                "configuration disagrees with frozen manifest dimension {manifest_dimension}"
            ),
        });
    }
    if config.pooling != Pooling::Mean {
        return Err(SearchError::InvalidConfig {
            field: "fastembed.pooling".to_owned(),
            value: format!("{:?}", config.pooling),
            reason: "configuration disagrees with frozen mean-pooling execution contract"
                .to_owned(),
        });
    }
    if manifest.execution.sequence_policy != FASTEMBED_SEQUENCE_POLICY_V1
        || manifest.execution.output_normalization != FASTEMBED_OUTPUT_NORMALIZATION_V1
    {
        return Err(SearchError::InvalidConfig {
            field: "fastembed.execution_contract".to_owned(),
            value: manifest.logical_model_id.clone(),
            reason: "registered tokenizer or normalization semantics disagree with the pinned FastEmbed adapter"
                .to_owned(),
        });
    }
    Ok(manifest)
}

const MODEL_ONNX_SUBDIR: &str = "onnx/model.onnx";

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
    model: Arc<Mutex<TextEmbedding>>,
    name: String,
    dimension: usize,
    model_dir: PathBuf,
    identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for FastEmbedEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FastEmbedEmbedder")
            .field("name", &self.name)
            .field("dimension", &self.dimension)
            .field("model_dir", &"<redacted>")
            .field("identity", &self.identity.fingerprint())
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
    /// Returns `SearchError::InvalidConfig` when model, dimension, or pooling
    /// disagrees with the frozen execution contract.
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX/session initialization fails.
    #[instrument(skip_all, fields(model = DEFAULT_MODEL_NAME))]
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_with_name(model_dir, DEFAULT_MODEL_NAME)
    }

    /// Load a registered `FastEmbed` model by its manifest model identifier.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` for an unregistered model name and
    /// `SearchError::ModelNotFound` when registered files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX/session initialization fails.
    pub fn load_with_name(model_dir: impl AsRef<Path>, name: &str) -> SearchResult<Self> {
        let mut config = OnnxEmbedderConfig::for_name(name).unwrap_or_else(|| OnnxEmbedderConfig {
            embedder_id: name.to_owned(),
            model_id: name.to_owned(),
            ..OnnxEmbedderConfig::default()
        });
        name.clone_into(&mut config.embedder_id);
        Self::load_with_config(model_dir, config)
    }

    /// Load an ONNX embedder with an explicit registered configuration.
    ///
    /// Dimension and pooling are checked against the frozen manifest before any
    /// model bytes are accepted.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::InvalidConfig` when model, dimension, or pooling
    /// disagrees with the frozen execution contract.
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX/session initialization fails.
    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        config: OnnxEmbedderConfig,
    ) -> SearchResult<Self> {
        let name = &config.model_id;
        let expected_dim = config.dimension;
        let frozen_manifest = frozen_manifest_for_config(&config)?;
        let model_dir = resolve_model_dir(model_dir.as_ref(), name)?;
        let verified = frozen_manifest.verify_dir(&model_dir)?;
        let identity = verified.identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
        let model_file =
            select_model_file(&model_dir).ok_or_else(|| SearchError::ModelNotFound {
                name: format!("{name} (missing registered artifact {MODEL_ONNX_SUBDIR})"),
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

        let init_options = InitOptionsUserDefined::new().with_max_length(FASTEMBED_MAX_LENGTH_V1);
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
            manifest = %identity.producer.provenance_manifest_fingerprint,
            identity = %identity.fingerprint(),
            "FastEmbed model loaded"
        );

        Ok(Self {
            model: Arc::new(Mutex::new(text_embedding)),
            name: config.embedder_id,
            dimension: expected_dim,
            model_dir,
            identity,
        })
    }

    /// Embed a single non-empty string.
    async fn embed_non_empty(&self, cx: &Cx, text: &str) -> SearchResult<Vec<f32>> {
        let mut embeddings = self.infer(cx, vec![text.to_owned()]).await?;

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
        let mut embeddings = self
            .infer(cx, texts.iter().map(|text| (*text).to_owned()).collect())
            .await?;

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

    /// ONNX is synchronous and cannot be preempted mid-call. Keep its owned
    /// model lock in a region-owned blocking worker: timeout can stop waiting
    /// without blocking the executor or admitting another call on this model.
    /// The caller must retain and fully drain its blocking pool at shutdown;
    /// a pool with a bounded shutdown wait cannot guarantee that by itself.
    async fn infer(&self, cx: &Cx, texts: Vec<String>) -> SearchResult<Vec<Vec<f32>>> {
        let mut model = OwnedMutexGuard::lock(Arc::clone(&self.model), cx)
            .await
            .map_err(|error| map_lock_error(&self.name, "fastembed.infer", error))?;
        let name = self.name.clone();
        let mut worker = cx
            .spawn_blocking(move |child| {
                embed_checkpoint(&child, "fastembed.infer")?;
                model
                    .embed(texts, None)
                    .map_err(|error| SearchError::EmbeddingFailed {
                        model: name,
                        source: format!("fastembed inference failed: {error}").into(),
                    })
            })
            .map_err(|error| SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!("cannot admit inference worker: {error}").into(),
            })?;
        worker.join(cx).await.map_err(|error| match error {
            asupersync::runtime::JoinError::Cancelled(_) => SearchError::Cancelled {
                phase: "fastembed.infer".to_owned(),
                reason: "inference worker cancelled".to_owned(),
            },
            error => SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!("inference worker failed: {error}").into(),
            },
        })?
    }

    /// Directory containing model assets.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
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
            embed_checkpoint(cx, "fastembed.embed")?;
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
            embed_checkpoint(cx, "fastembed.embed_batch")?;
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
            for (slot_idx, embedding) in non_empty_indices.into_iter().zip(normalized) {
                output[slot_idx] = embedding;
            }
            Ok(output)
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
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

    if let Some(cache_dir) = frankensearch_core::platform_dirs::cache_dir() {
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
        other => {
            let detail = format!("fastembed mutex lock failed during {phase}: {other}");
            SearchError::EmbeddingFailed {
                model: model.to_owned(),
                source: std::io::Error::other(detail).into(),
            }
        }
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
    let registered = model_dir.join(MODEL_ONNX_SUBDIR);
    registered.is_file().then_some(registered)
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
    fn has_required_files_rejects_unregistered_legacy_onnx_path() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model_layout(temp.path(), false);
        assert!(!has_required_files(temp.path()));
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
    fn embed_checkpoint_observes_cancel_before_onnx() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            cx.cancel_fast(asupersync::CancelKind::User);
            let err = super::embed_checkpoint(&cx, "fastembed.embed").unwrap_err();
            match err {
                SearchError::Cancelled { phase, .. } => {
                    assert_eq!(phase, "fastembed.embed");
                }
                other => panic!("expected Cancelled, got {other:?}"),
            }
        });
    }

    #[test]
    fn map_lock_error_poisoned_to_embedding_failed() {
        let err = map_lock_error("all-MiniLM-L6-v2", "fastembed.embed", LockError::Poisoned);
        assert!(matches!(&err, SearchError::EmbeddingFailed { .. }));
    }

    #[test]
    fn map_lock_error_polled_after_completion_to_embedding_failed() {
        let err = map_lock_error(
            "all-MiniLM-L6-v2",
            "fastembed.embed",
            LockError::PolledAfterCompletion,
        );
        assert!(matches!(err, SearchError::EmbeddingFailed { .. }));
        if let SearchError::EmbeddingFailed { source, .. } = err {
            assert!(
                source
                    .to_string()
                    .contains("future polled after completion")
            );
        }
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
    fn select_model_file_rejects_unregistered_legacy_path() {
        let temp = tempfile::tempdir().unwrap();
        // Only unregistered legacy model.onnx, no frozen onnx/ path.
        std::fs::write(temp.path().join("model.onnx"), b"legacy").unwrap();
        assert!(select_model_file(temp.path()).is_none());
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
        assert!(matches!(&err, SearchError::Cancelled { .. }));
        if let SearchError::Cancelled { phase, .. } = err {
            assert_eq!(phase, "test.phase");
        }
    }

    #[test]
    fn map_lock_error_preserves_model_string() {
        let err = map_lock_error("custom-model", "embed", LockError::Poisoned);
        assert!(matches!(&err, SearchError::EmbeddingFailed { .. }));
        if let SearchError::EmbeddingFailed { model, .. } = err {
            assert_eq!(model, "custom-model");
        }
    }

    #[test]
    fn resolve_model_dir_error_message_includes_paths() {
        let temp = tempfile::tempdir().unwrap();
        let err = resolve_model_dir(temp.path(), "my-model").unwrap_err();
        assert!(matches!(&err, SearchError::ModelNotFound { .. }));
        if let SearchError::ModelNotFound { name } = err {
            assert!(name.contains("my-model"), "error should include model name");
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
        assert_eq!(
            OnnxEmbedderConfig::for_name("all-minilm-l6-v2")
                .unwrap()
                .dimension,
            384
        );

        let snowflake = OnnxEmbedderConfig::for_name("snowflake-arctic-s").unwrap();
        assert_eq!(snowflake.embedder_id, "snowflake-arctic-s-384");
        assert_eq!(snowflake.dimension, 384);
        assert_eq!(
            OnnxEmbedderConfig::for_name("snowflake-arctic-embed-s")
                .unwrap()
                .dimension,
            384
        );

        let nomic = OnnxEmbedderConfig::for_name("nomic-embed").unwrap();
        assert_eq!(nomic.embedder_id, "nomic-embed-768");
        assert_eq!(nomic.dimension, 768);
        assert_eq!(
            OnnxEmbedderConfig::for_name("nomic-embed-text-v1.5")
                .unwrap()
                .dimension,
            768
        );
    }

    #[test]
    fn onnx_config_for_name_unknown_returns_none() {
        assert!(OnnxEmbedderConfig::for_name("unknown-model").is_none());
        assert!(OnnxEmbedderConfig::for_name("").is_none());
    }

    #[test]
    fn frozen_manifest_rejects_pooling_and_dimension_drift() {
        let pooling_drift = OnnxEmbedderConfig {
            pooling: Pooling::Cls,
            ..OnnxEmbedderConfig::default()
        };
        assert!(matches!(
            frozen_manifest_for_config(&pooling_drift),
            Err(SearchError::InvalidConfig { .. })
        ));

        let mut dimension_drift = OnnxEmbedderConfig::default();
        dimension_drift.dimension += 1;
        assert!(matches!(
            frozen_manifest_for_config(&dimension_drift),
            Err(SearchError::InvalidConfig { .. })
        ));

        let unregistered = OnnxEmbedderConfig {
            model_id: "unregistered-model".to_owned(),
            ..OnnxEmbedderConfig::default()
        };
        assert!(matches!(
            frozen_manifest_for_config(&unregistered),
            Err(SearchError::InvalidConfig { .. })
        ));
    }

    fn assert_conformance_fixture(
        environment_variable: &str,
        config: OnnxEmbedderConfig,
        manifest: &ModelArtifactManifestV1,
    ) {
        let dir = std::env::var(environment_variable)
            .expect("conformance fixture environment variable must name the verified model dir");
        let expected_identity = manifest
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .expect("derive registered FastEmbed identity");
        let embedder =
            FastEmbedEmbedder::load_with_config(&dir, config).expect("load verified ONNX model");
        assert_eq!(embedder.identity().unwrap(), &expected_identity);
        let texts = &crate::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .expect("build current-thread conformance runtime");
        let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
        let vectors = runtime
            .block_on(embedder.embed_batch(&cx, texts))
            .expect("embed bounded conformance corpus");
        let observed = frankensearch_core::generation::GoldenVectorCertificateV1::from_exact_f32(
            texts, &vectors,
        )
        .expect("compute exact conformance certificate");
        assert_eq!(
            observed, manifest.execution.golden_vectors,
            "FastEmbed output bits drifted from the registered producer certificate"
        );
    }

    #[test]
    #[ignore = "requires verified MiniLM ONNX assets via FASTEMBED_MINILM_FIXTURE_DIR"]
    fn minilm_conformance_certificate_matches_fixture() {
        assert_conformance_fixture(
            "FASTEMBED_MINILM_FIXTURE_DIR",
            OnnxEmbedderConfig::for_name("minilm").unwrap(),
            &ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        );
    }

    #[test]
    #[ignore = "requires verified Snowflake ONNX assets via FASTEMBED_SNOWFLAKE_FIXTURE_DIR"]
    fn snowflake_conformance_certificate_matches_fixture() {
        assert_conformance_fixture(
            "FASTEMBED_SNOWFLAKE_FIXTURE_DIR",
            OnnxEmbedderConfig::for_name("snowflake-arctic-s").unwrap(),
            &ModelArtifactManifestV1::snowflake_fastembed().unwrap(),
        );
    }

    #[test]
    #[ignore = "requires verified Nomic ONNX assets via FASTEMBED_NOMIC_FIXTURE_DIR"]
    fn nomic_conformance_certificate_matches_fixture() {
        assert_conformance_fixture(
            "FASTEMBED_NOMIC_FIXTURE_DIR",
            OnnxEmbedderConfig::for_name("nomic-embed").unwrap(),
            &ModelArtifactManifestV1::nomic_fastembed().unwrap(),
        );
    }
}
