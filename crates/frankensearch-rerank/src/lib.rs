//! `FlashRank` cross-encoder reranking for frankensearch.
//!
//! Provides the [`FlashRankReranker`] implementation of the [`Reranker`] trait,
//! using ONNX Runtime for cross-encoder scoring with sigmoid activation on raw
//! logits. Gracefully falls back to original scores if the model is unavailable.
//!
//! # Model Layout
//!
//! Required files in the model directory:
//! - `onnx/model.onnx` (preferred) OR `model.onnx` (legacy)
//! - `tokenizer.json`
//!
//! # Architecture
//!
//! Cross-encoders differ from bi-encoders: instead of comparing pre-computed
//! embeddings, they process the query and document *together* through a
//! transformer, producing direct token-level attention between them. This is
//! dramatically more accurate but cannot pre-compute anything.
//!
//! ```text
//! (query, document) → tokenize → ONNX → logit → sigmoid → score ∈ [0, 1]
//! ```

pub mod pipeline;

#[cfg(feature = "fastembed-reranker")]
pub mod fastembed_reranker;

pub use pipeline::{DEFAULT_MIN_CANDIDATES, DEFAULT_TOP_K_RERANK, rerank_step};

#[cfg(feature = "fastembed-reranker")]
pub use fastembed_reranker::FastEmbedReranker;

use std::fmt;
use std::path::{Path, PathBuf};

use asupersync::Cx;
use asupersync::sync::{LockError, Mutex};
use ort::session::Session;
use tokenizers::Tokenizer;
use tracing::instrument;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{RerankDocument, RerankScore, Reranker, SearchFuture};

/// Default model directory name for the `FlashRank` nano cross-encoder.
pub const DEFAULT_MODEL_NAME: &str = "flashrank";

/// Default maximum input token length for cross-encoder pairs.
pub const DEFAULT_MAX_LENGTH: usize = 512;

/// Batch size for ONNX inference (number of query-doc pairs per run).
const INFERENCE_BATCH_SIZE: usize = 32;

const MODEL_ONNX_SUBDIR: &str = "onnx/model.onnx";
const MODEL_ONNX_LEGACY: &str = "model.onnx";
const TOKENIZER_JSON: &str = "tokenizer.json";

/// Output tensor name candidates, tried in order.
const OUTPUT_TENSOR_CANDIDATES: [&str; 3] = ["logits", "output", "sentence_embedding"];

/// `FlashRank` cross-encoder reranker backed by ONNX Runtime.
///
/// The ONNX session is wrapped in a cancel-aware [`asupersync::sync::Mutex`]
/// because `Session` is not safe for concurrent mutable access.
///
/// # Sigmoid Activation
///
/// Raw ONNX output is a **logit**, not a probability. This implementation
/// applies sigmoid activation to produce scores in `[0, 1]`:
///
/// ```text
/// score = 1 / (1 + exp(-logit))
/// ```
pub struct FlashRankReranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    name: String,
    model_dir: PathBuf,
}

impl fmt::Debug for FlashRankReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlashRankReranker")
            .field("name", &self.name)
            .field("max_length", &self.max_length)
            .field("model_dir", &self.model_dir)
            .finish_non_exhaustive()
    }
}

impl FlashRankReranker {
    /// Load a `FlashRank` cross-encoder model from a local directory.
    ///
    /// The directory must contain `tokenizer.json` and either
    /// `onnx/model.onnx` (preferred) or `model.onnx` (legacy).
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX session creation fails.
    #[instrument(skip_all, fields(model_dir = %model_dir.as_ref().display()))]
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_with_config(model_dir, DEFAULT_MODEL_NAME, DEFAULT_MAX_LENGTH)
    }

    /// Load a `FlashRank` model with custom name and max token length.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::ModelNotFound` when required files are missing.
    /// Returns `SearchError::ModelLoadFailed` when ONNX session creation fails.
    pub fn load_with_config(
        model_dir: impl AsRef<Path>,
        name: &str,
        max_length: usize,
    ) -> SearchResult<Self> {
        let model_dir = resolve_model_dir(model_dir.as_ref(), name)?;

        let model_file =
            select_model_file(&model_dir).ok_or_else(|| SearchError::ModelNotFound {
                name: format!("{name} (missing {MODEL_ONNX_SUBDIR} or {MODEL_ONNX_LEGACY})"),
            })?;

        let tokenizer_path = model_dir.join(TOKENIZER_JSON);
        if !tokenizer_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{name} (missing {TOKENIZER_JSON} in {})",
                    model_dir.display()
                ),
            });
        }

        // Load ONNX session with Level3 optimization
        let session = Session::builder()
            .and_then(|b| {
                b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            })
            .and_then(|b| b.with_intra_threads(num_cpus()))
            .and_then(|b| b.commit_from_file(&model_file))
            .map_err(|e| SearchError::ModelLoadFailed {
                path: model_file.clone(),
                source: format!("ONNX session creation failed: {e}").into(),
            })?;

        // Load tokenizer
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| SearchError::ModelLoadFailed {
                path: tokenizer_path,
                source: format!("tokenizer load failed: {e}").into(),
            })?;

        tracing::info!(
            model = %name,
            max_length,
            model_dir = %model_dir.display(),
            "FlashRank reranker model loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length,
            name: name.to_owned(),
            model_dir,
        })
    }

    /// Tokenize a (query, document) pair into ONNX-ready input tensors.
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids)` as `Vec<i64>`.
    fn tokenize_pair(&self, query: &str, document: &str) -> SearchResult<TokenizedPair> {
        let encoding = self
            .tokenizer
            .encode((query, document), true)
            .map_err(|e| SearchError::RerankFailed {
                model: self.name.clone(),
                source: format!("tokenization failed: {e}").into(),
            })?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();
        let mut token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&t| i64::from(t))
            .collect();

        // Truncate to max_length
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
            token_type_ids.truncate(self.max_length);
        }

        Ok(TokenizedPair {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    /// Run ONNX inference on a batch of tokenized pairs and return sigmoid-activated scores.
    #[allow(clippy::cast_possible_wrap)]
    fn infer_batch(
        session: &mut Session,
        pairs: &[TokenizedPair],
        model_name: &str,
    ) -> SearchResult<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = pairs.len();
        let seq_len = pairs.iter().map(|p| p.input_ids.len()).max().unwrap_or(0);

        let flat_len =
            batch_size
                .checked_mul(seq_len)
                .ok_or_else(|| SearchError::RerankFailed {
                    model: model_name.to_owned(),
                    source: format!("tensor size overflow: {batch_size} * {seq_len}").into(),
                })?;

        // Pad all sequences to the same length and flatten into contiguous arrays
        let mut flat_input_ids = vec![0_i64; flat_len];
        let mut flat_attention_mask = vec![0_i64; flat_len];
        let mut flat_token_type_ids = vec![0_i64; flat_len];

        for (i, pair) in pairs.iter().enumerate() {
            let offset = i * seq_len;
            flat_input_ids[offset..offset + pair.input_ids.len()].copy_from_slice(&pair.input_ids);
            flat_attention_mask[offset..offset + pair.attention_mask.len()]
                .copy_from_slice(&pair.attention_mask);
            flat_token_type_ids[offset..offset + pair.token_type_ids.len()]
                .copy_from_slice(&pair.token_type_ids);
        }

        let batch_i64 = i64::try_from(batch_size).map_err(|_| SearchError::RerankFailed {
            model: model_name.to_owned(),
            source: format!("batch_size {batch_size} exceeds i64::MAX").into(),
        })?;
        let seq_i64 = i64::try_from(seq_len).map_err(|_| SearchError::RerankFailed {
            model: model_name.to_owned(),
            source: format!("seq_len {seq_len} exceeds i64::MAX").into(),
        })?;
        let shape = [batch_i64, seq_i64];

        let input_ids_tensor = ort::value::Tensor::from_array((shape, flat_input_ids))
            .map_err(|e| rerank_ort_error(model_name, "input_ids tensor", &e))?;
        let attention_mask_tensor = ort::value::Tensor::from_array((shape, flat_attention_mask))
            .map_err(|e| rerank_ort_error(model_name, "attention_mask tensor", &e))?;
        let token_type_ids_tensor = ort::value::Tensor::from_array((shape, flat_token_type_ids))
            .map_err(|e| rerank_ort_error(model_name, "token_type_ids tensor", &e))?;

        let inputs = ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        };

        let outputs = session.run(inputs).map_err(|e| SearchError::RerankFailed {
            model: model_name.to_owned(),
            source: format!("ONNX inference failed: {e}").into(),
        })?;

        // Extract logits from the output tensor using name fallback chain
        let logits = extract_logits(&outputs, model_name, batch_size)?;

        // Apply sigmoid activation to convert logits → probabilities
        Ok(logits.iter().map(|&logit| sigmoid(logit)).collect())
    }

    /// Directory containing model assets.
    #[must_use]
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

impl Reranker for FlashRankReranker {
    fn rerank<'a>(
        &'a self,
        cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            if documents.is_empty() {
                return Ok(Vec::new());
            }

            // Tokenize all (query, document) pairs
            let mut all_pairs = Vec::with_capacity(documents.len());
            for doc in documents {
                all_pairs.push(self.tokenize_pair(query, &doc.text)?);
            }

            // Acquire ONNX session (cancel-aware)
            let mut session = self
                .session
                .lock(cx)
                .await
                .map_err(|err| map_lock_error(&self.name, err))?;

            // Run inference in batches
            let mut all_scores = Vec::with_capacity(documents.len());
            for chunk in all_pairs.chunks(INFERENCE_BATCH_SIZE) {
                let batch_scores = Self::infer_batch(&mut session, chunk, &self.name)?;
                all_scores.extend(batch_scores);
            }

            // Validate inference returned one score per document.
            if all_scores.len() != documents.len() {
                return Err(SearchError::RerankFailed {
                    model: self.name.clone(),
                    source: format!(
                        "inference returned {} scores for {} documents",
                        all_scores.len(),
                        documents.len()
                    )
                    .into(),
                });
            }

            // Build RerankScore results with original rank tracking.
            // Use index lookup (not zip) so score/doc cardinality mismatches
            // can never be silently truncated if this block is modified later.
            let mut results = Vec::with_capacity(documents.len());
            for (rank, doc) in documents.iter().enumerate() {
                let Some(&score) = all_scores.get(rank) else {
                    return Err(SearchError::RerankFailed {
                        model: self.name.clone(),
                        source: format!(
                            "missing score at rank {rank} ({} scores for {} documents)",
                            all_scores.len(),
                            documents.len()
                        )
                        .into(),
                    });
                };
                results.push(RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score,
                    original_rank: rank,
                });
            }

            // Sort by descending cross-encoder score (NaN-safe)
            results.sort_by(|a, b| {
                sanitize_score(b.score)
                    .total_cmp(&sanitize_score(a.score))
                    .then_with(|| a.doc_id.cmp(&b.doc_id))
            });

            Ok(results)
        })
    }

    fn id(&self) -> &str {
        &self.name
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_length(&self) -> usize {
        self.max_length
    }
}

// ─── Internal Helpers ──────────────────────────────────────────────────────

struct TokenizedPair {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

/// Sigmoid activation: converts logit → probability in [0, 1].
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sanitize a score for comparison: NaN/Inf → -infinity.
#[inline]
const fn sanitize_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        f32::NEG_INFINITY
    }
}

/// Extract logits from ONNX session output using a name fallback chain.
///
/// Uses `try_extract_raw_tensor::<f32>()` which returns `(&[i64], &[f32])`.
#[allow(clippy::cast_sign_loss)]
fn extract_logits(
    outputs: &ort::session::SessionOutputs<'_>,
    model_name: &str,
    batch_size: usize,
) -> SearchResult<Vec<f32>> {
    // Try named output tensors in priority order (use .get() to avoid panicking)
    for name in &OUTPUT_TENSOR_CANDIDATES {
        if let Some(value) = outputs.get(*name)
            && let Ok((shape, data)) = value.try_extract_tensor::<f32>()
        {
            return extract_scores_from_raw((&**shape, data), batch_size, model_name);
        }
    }

    // Fallback: use first output by index
    if let Some((_, value)) = outputs.iter().next()
        && let Ok((shape, data)) = value.try_extract_tensor::<f32>()
    {
        return extract_scores_from_raw((&**shape, data), batch_size, model_name);
    }

    Err(SearchError::RerankFailed {
        model: model_name.to_owned(),
        source: "no extractable output tensor found in ONNX session output".into(),
    })
}

/// Extract per-sample scores from a raw tensor `(shape, data)`.
///
/// Cross-encoder models may output:
/// - Shape `[batch, 1]`: one logit per sample
/// - Shape `[batch, 2]`: binary classification (positive class logit at index 1)
/// - Shape `[batch]`: flat logits
/// - Shape `[1, batch]`: transposed single-row output
fn extract_scores_from_raw(
    (shape, data): (&[i64], &[f32]),
    batch_size: usize,
    model_name: &str,
) -> SearchResult<Vec<f32>> {
    // Safe i64→usize helper: negative or out-of-range dimensions never match.
    let dim_eq = |n: &i64| usize::try_from(*n).is_ok_and(|v| v == batch_size);

    match shape {
        // [batch, 1] — single logit per sample
        [n, 1] if dim_eq(n) && data.len() >= batch_size => Ok(data[..batch_size].to_vec()),
        // [batch, 2] — binary classification, take positive class (index 1)
        [n, 2] if dim_eq(n) && data.len() == batch_size * 2 => {
            Ok(data.chunks_exact(2).map(|pair| pair[1]).collect())
        }
        // [batch] — flat logits
        [n] if dim_eq(n) => Ok(data.to_vec()),
        // [1, batch] — transposed single-row output
        [1, n] if dim_eq(n) => Ok(data.to_vec()),
        _ => {
            // Best-effort: take first `batch_size` elements if enough data
            if data.len() >= batch_size {
                Ok(data[..batch_size].to_vec())
            } else {
                Err(SearchError::RerankFailed {
                    model: model_name.to_owned(),
                    source: format!(
                        "unexpected output tensor shape {shape:?} for batch size {batch_size}"
                    )
                    .into(),
                })
            }
        }
    }
}

fn map_lock_error(model: &str, error: LockError) -> SearchError {
    match error {
        LockError::Cancelled => SearchError::Cancelled {
            phase: "rerank".to_owned(),
            reason: "mutex lock cancelled".to_owned(),
        },
        LockError::Poisoned => SearchError::RerankFailed {
            model: model.to_owned(),
            source: "reranker mutex poisoned".into(),
        },
    }
}

fn rerank_ort_error(model: &str, context: &str, error: &ort::Error) -> SearchError {
    SearchError::RerankFailed {
        model: model.to_owned(),
        source: format!("{context}: {error}").into(),
    }
}

fn resolve_model_dir(base_dir: &Path, model_name: &str) -> SearchResult<PathBuf> {
    if has_required_files(base_dir) {
        return Ok(base_dir.to_path_buf());
    }

    // Reject model names containing path traversal sequences.
    if model_name.contains("..") || model_name.starts_with('/') || model_name.starts_with('\\') {
        return Err(SearchError::ModelNotFound {
            name: format!("{model_name} (unsafe model name)"),
        });
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
    select_model_file(dir).is_some() && dir.join(TOKENIZER_JSON).is_file()
}

/// Search for a `FlashRank` model directory in standard locations.
///
/// Checks these paths in order:
/// 1. `$FRANKENSEARCH_MODEL_DIR/<model_name>/` then `$FRANKENSEARCH_MODEL_DIR`
/// 2. `~/.cache/frankensearch/models/<model_name>/`
/// 3. `~/.local/share/frankensearch/models/<model_name>/`
/// 4. `~/.cache/flashrank/<model_name>/`
///
/// Returns `None` if no directory with required files is found.
#[must_use]
pub fn find_model_dir(model_name: &str) -> Option<PathBuf> {
    // Reject model names containing path traversal sequences (same validation
    // as resolve_model_dir). Without this, `Path::join(model_name)` with
    // `../../etc` escapes the intended model directories.
    if model_name.contains("..") || model_name.starts_with('/') || model_name.starts_with('\\') {
        return None;
    }

    let mut candidates = Vec::new();

    if let Ok(dir) = std::env::var("FRANKENSEARCH_MODEL_DIR") {
        let base = PathBuf::from(dir);
        candidates.push(base.join(model_name));
        candidates.push(base);
    }

    if let Some(cache_dir) = dirs::cache_dir() {
        candidates.push(cache_dir.join("frankensearch/models").join(model_name));
        candidates.push(cache_dir.join("flashrank").join(model_name));
    }
    if let Some(data_dir) = dirs::data_local_dir() {
        candidates.push(data_dir.join("frankensearch/models").join(model_name));
    }

    candidates.into_iter().find(|dir| has_required_files(dir))
}

/// Number of logical CPU cores, capped to a reasonable value for ONNX threads.
fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get().min(8))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    // ─── Sigmoid Tests ──────────────────────────────────────────────────

    #[test]
    fn sigmoid_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_large_positive_approaches_one() {
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn sigmoid_large_negative_approaches_zero() {
        assert!(sigmoid(-10.0) < 0.01);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn sigmoid_output_range() {
        for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} out of [0,1]");
        }
    }

    #[test]
    fn sigmoid_symmetry() {
        // sigmoid(x) + sigmoid(-x) == 1
        for x in [0.5, 1.0, 2.5, 5.0] {
            let sum = sigmoid(x) + sigmoid(-x);
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "sigmoid({x}) + sigmoid(-{x}) = {sum}"
            );
        }
    }

    // ─── Score Sanitization ─────────────────────────────────────────────

    #[test]
    fn sanitize_finite_scores() {
        assert!((sanitize_score(0.5) - 0.5).abs() < f32::EPSILON);
        assert!((sanitize_score(1.0) - 1.0).abs() < f32::EPSILON);
        assert!((sanitize_score(-1.0) - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn sanitize_nan_to_neg_infinity() {
        assert!(sanitize_score(f32::NAN) == f32::NEG_INFINITY);
    }

    #[test]
    fn sanitize_infinity_to_neg_infinity() {
        assert!(sanitize_score(f32::INFINITY) == f32::NEG_INFINITY);
        assert!(sanitize_score(f32::NEG_INFINITY) == f32::NEG_INFINITY);
    }

    // ─── Model Directory Discovery ──────────────────────────────────────

    #[test]
    fn has_required_files_with_modern_layout() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model(temp.path(), true);
        assert!(has_required_files(temp.path()));
    }

    #[test]
    fn has_required_files_with_legacy_layout() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model(temp.path(), false);
        assert!(has_required_files(temp.path()));
    }

    #[test]
    fn has_required_files_missing_tokenizer() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("onnx")).unwrap();
        fs::write(temp.path().join("onnx/model.onnx"), b"stub").unwrap();
        // No tokenizer.json
        assert!(!has_required_files(temp.path()));
    }

    #[test]
    fn resolve_model_dir_direct_path() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model(temp.path(), true);
        let resolved = resolve_model_dir(temp.path(), "flashrank").unwrap();
        assert_eq!(resolved, temp.path());
    }

    #[test]
    fn resolve_model_dir_nested_path() {
        let temp = tempfile::tempdir().unwrap();
        let child = temp.path().join("flashrank");
        fs::create_dir_all(&child).unwrap();
        create_stub_model(&child, true);
        let resolved = resolve_model_dir(temp.path(), "flashrank").unwrap();
        assert_eq!(resolved, child);
    }

    #[test]
    fn resolve_model_dir_missing() {
        let temp = tempfile::tempdir().unwrap();
        let err = resolve_model_dir(temp.path(), "flashrank").unwrap_err();
        assert!(matches!(err, SearchError::ModelNotFound { .. }));
    }

    #[test]
    fn select_model_file_prefers_modern() {
        let temp = tempfile::tempdir().unwrap();
        create_stub_model(temp.path(), true);
        fs::write(temp.path().join("model.onnx"), b"legacy").unwrap();
        let selected = select_model_file(temp.path()).unwrap();
        assert!(selected.ends_with(MODEL_ONNX_SUBDIR));
    }

    // ─── Lock Error Mapping ─────────────────────────────────────────────

    #[test]
    fn lock_cancelled_maps_to_search_cancelled() {
        let err = map_lock_error("flashrank", LockError::Cancelled);
        assert!(matches!(err, SearchError::Cancelled { .. }));
    }

    #[test]
    fn lock_poisoned_maps_to_rerank_failed() {
        let err = map_lock_error("flashrank", LockError::Poisoned);
        assert!(matches!(err, SearchError::RerankFailed { .. }));
    }

    // ─── num_cpus ───────────────────────────────────────────────────────

    #[test]
    fn num_cpus_returns_reasonable_value() {
        let n = num_cpus();
        assert!((1..=8).contains(&n));
    }

    // ─── find_model_dir path traversal (bd-124o) ────────────────────────

    #[test]
    fn find_model_dir_rejects_dotdot_traversal() {
        // Even if a model directory existed at a traversed path, the function
        // must reject it to prevent escaping the model directory hierarchy.
        assert!(find_model_dir("../../etc").is_none());
        assert!(find_model_dir("foo/../bar").is_none());
    }

    #[test]
    fn find_model_dir_rejects_absolute_path() {
        assert!(find_model_dir("/etc/passwd").is_none());
    }

    #[test]
    fn find_model_dir_rejects_backslash_prefix() {
        assert!(find_model_dir("\\Windows\\System32").is_none());
    }

    // ─── Test helpers ───────────────────────────────────────────────────

    fn create_stub_model(dir: &Path, use_onnx_subdir: bool) {
        if use_onnx_subdir {
            fs::create_dir_all(dir.join("onnx")).unwrap();
            fs::write(dir.join("onnx/model.onnx"), b"stub-onnx").unwrap();
        } else {
            fs::write(dir.join("model.onnx"), b"stub-onnx").unwrap();
        }
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    }
}
