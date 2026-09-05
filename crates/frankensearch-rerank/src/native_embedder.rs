//! Pure-Rust transformer sentence-embedder (`all-MiniLM-L6-v2`, 384-dim) backed by
//! frankentorch — the embedding counterpart of [`crate::native::NativeReranker`] (no
//! ONNX / no `ort`).
//!
//! It reuses the reranker's validated, SIMD/int8-optimized BERT encoder verbatim
//! (the registered 6- or 12-layer topology, with the same kernels via
//! [`crate::native::Model::embed_forward`]); it differs only at the head —
//! **mean-pool over every token + L2-normalize** instead of the `[CLS]` pooler +
//! classifier — and in tokenization (one text, token-type ids all 0). Because there
//! is no ONNX Runtime, there is no AVX-static-init hazard: the int8 GEMM dispatches
//! NEON (aarch64 SDOT / NR=4 packing) or x86 SIMD at runtime.
//!
//! Feature-gated behind `native`.
//! `NativeEmbeddingModel::AllMiniLmL6V2F32` explicitly retains F32 Linear weights
//! for higher numerical fidelity. It has its own producer identity; it is not
//! substituted for the default int8 producer or an existing index's producer.

use std::path::Path;
use std::sync::{Arc, Mutex};

use asupersync::Cx;
use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use asupersync::sync::{LockError, Mutex as AsyncMutex, OwnedMutexGuard};

use tokenizers::Tokenizer;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{ModelCategory, SearchFuture, SyncEmbed};
use frankensearch_embed::model_manifest::ModelArtifactManifestV1;

use crate::native::{
    DEFAULT_MAX_LENGTH, LinearPrecision, Model, SAFETENSORS_FALLBACK, TOKENIZER_JSON, build_model,
    parse_weights,
};

const DEFAULT_MODEL_NAME: &str = "all-minilm-l6-v2";
const DEFAULT_EMBEDDER_ID: &str = "minilm-384-native";
const MULTILINGUAL_MODEL_NAME: &str = "paraphrase-multilingual-minilm-l12-v2";
const MULTILINGUAL_EMBEDDER_ID: &str = "paraphrase-multilingual-minilm-l12-v2-384-native";
const DIM: usize = 384;
const IDENTITY_DIMENSION: u32 = 384;
const IDENTITY_SEQUENCE_POLICY: &str = "max-length=512;longest-first;no-padding";
const IDENTITY_POOLING: &str = "mean-all-returned-tokens-including-specials-no-padding-v1";
const IDENTITY_OUTPUT_NORMALIZATION: &str = "l2-f32-if-norm-gt-zero-else-unchanged-v1";
/// Token budget per batched forward (mirrors the reranker's chunking) so each
/// forward's attention intermediates stay memory-bounded.
const MAX_BATCH_TOKENS: usize = 2048;

/// Manifest-registered pure-Rust sentence-embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeEmbeddingModel {
    /// English-centric `all-MiniLM-L6-v2` baseline with int8 linears.
    AllMiniLmL6V2,
    /// Explicit full-F32 English `MiniLM` producer; requires its own index identity.
    AllMiniLmL6V2F32,
    /// Opt-in XLM-R/SentencePiece multilingual `MiniLM` L12 model.
    ParaphraseMultilingualMiniLmL12V2,
}

impl NativeEmbeddingModel {
    const fn model_name(self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 | Self::AllMiniLmL6V2F32 => DEFAULT_MODEL_NAME,
            Self::ParaphraseMultilingualMiniLmL12V2 => MULTILINGUAL_MODEL_NAME,
        }
    }

    const fn embedder_id(self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => DEFAULT_EMBEDDER_ID,
            Self::AllMiniLmL6V2F32 => "minilm-384-native-f32",
            Self::ParaphraseMultilingualMiniLmL12V2 => MULTILINGUAL_EMBEDDER_ID,
        }
    }

    const fn encoder_layers(self) -> usize {
        match self {
            Self::AllMiniLmL6V2 | Self::AllMiniLmL6V2F32 => 6,
            Self::ParaphraseMultilingualMiniLmL12V2 => 12,
        }
    }

    const fn linear_precision(self) -> LinearPrecision {
        match self {
            Self::AllMiniLmL6V2F32 => LinearPrecision::F32,
            Self::AllMiniLmL6V2 | Self::ParaphraseMultilingualMiniLmL12V2 => LinearPrecision::Int8,
        }
    }

    fn manifest(self) -> SearchResult<ModelArtifactManifestV1> {
        match self {
            Self::AllMiniLmL6V2 => ModelArtifactManifestV1::minilm_native_frankentorch(),
            Self::AllMiniLmL6V2F32 => ModelArtifactManifestV1::minilm_native_frankentorch_f32(),
            Self::ParaphraseMultilingualMiniLmL12V2 => {
                ModelArtifactManifestV1::multilingual_minilm_native_frankentorch()
            }
        }
    }
}

/// Pure-Rust frankentorch `MiniLM` sentence-embedder.
///
/// Clones share the loaded model and serialize inference. For async [`Embedder`](
/// frankensearch_core::traits::Embedder) use, attach a caller-owned pool with
/// [`Self::with_blocking_pool`]; synchronous [`SyncEmbed`] use needs no runtime.
#[derive(Clone)]
pub struct NativeEmbedder {
    /// One frankentorch session behind a `Mutex` (each forward parallelizes internally
    /// across cores; calls are serialized, so no nested-rayon-under-lock hazard) — same
    /// pattern as [`crate::native::NativeReranker`].
    inner: Arc<Mutex<Model>>,
    tokenizer: Arc<Tokenizer>,
    admission: Arc<AsyncMutex<()>>,
    blocking_pool: Option<BlockingPoolHandle>,
    max_length: usize,
    name: String,
    id: String,
    identity: EmbeddingIdentityBundleV1,
}

impl std::fmt::Debug for NativeEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEmbedder")
            .field("name", &self.name)
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

impl NativeEmbedder {
    /// Load the default `all-MiniLM-L6-v2` model from a verified directory.
    ///
    /// # Errors
    /// [`SearchError::ModelNotFound`] when required files are missing;
    /// [`SearchError::ModelLoadFailed`] when the tokenizer or weights fail to load.
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_model(model_dir, NativeEmbeddingModel::AllMiniLmL6V2)
    }

    /// Load the opt-in multilingual `MiniLM` L12 model from a verified directory.
    ///
    /// This constructor is intentionally explicit: the multilingual model is never
    /// substituted for the default model merely because both output 384 values.
    ///
    /// # Errors
    /// [`SearchError::ModelNotFound`] when required files are missing;
    /// [`SearchError::ModelLoadFailed`] when the tokenizer, topology, or weights fail.
    pub fn load_multilingual(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        Self::load_model(
            model_dir,
            NativeEmbeddingModel::ParaphraseMultilingualMiniLmL12V2,
        )
    }

    /// Load one explicit manifest-registered native embedding model.
    ///
    /// # Errors
    /// [`SearchError::ModelNotFound`] when required files are missing;
    /// [`SearchError::ModelLoadFailed`] when the tokenizer, topology, weights, or
    /// executing producer's registered output certificate fails verification.
    pub fn load_model(
        model_dir: impl AsRef<Path>,
        profile: NativeEmbeddingModel,
    ) -> SearchResult<Self> {
        Self::load_from_manifest(model_dir.as_ref(), profile, &profile.manifest()?)
    }

    fn load_from_manifest(
        dir: &Path,
        profile: NativeEmbeddingModel,
        manifest: &ModelArtifactManifestV1,
    ) -> SearchResult<Self> {
        let model_name = profile.model_name();
        let verified = manifest.verify_dir(dir)?;
        let identity = verified.identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")?;
        if identity.space.dimension != IDENTITY_DIMENSION {
            return Err(SearchError::ModelLoadFailed {
                path: dir.to_path_buf(),
                source: format!(
                    "registered dimension {} disagrees with native backend dimension {DIM}",
                    identity.space.dimension
                )
                .into(),
            });
        }
        for (field, actual, expected) in [
            (
                "sequence policy",
                identity.space.sequence_policy.as_str(),
                IDENTITY_SEQUENCE_POLICY,
            ),
            ("pooling", identity.space.pooling.as_str(), IDENTITY_POOLING),
            (
                "output normalization",
                identity.space.output_normalization.as_str(),
                IDENTITY_OUTPUT_NORMALIZATION,
            ),
        ] {
            if actual != expected {
                return Err(SearchError::ModelLoadFailed {
                    path: dir.to_path_buf(),
                    source: format!(
                        "registered {field} disagrees with the native backend contract"
                    )
                    .into(),
                });
            }
        }

        let tok_path = dir.join(TOKENIZER_JSON);
        if !tok_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{model_name} (missing {TOKENIZER_JSON} in {})",
                    dir.display()
                ),
            });
        }
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("tokenizer load failed: {e}").into(),
            })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("failed to enable truncation: {e}").into(),
            })?;
        // Disable padding: `tokenizer.json` ships a fixed-length padding config, but the
        // embedder mean-pools over EVERY returned token, so any `[PAD]` tokens would
        // corrupt the sentence embedding (they dominate the mean and collapse all
        // embeddings toward each other — anisotropy). Each text/batch element is encoded
        // to its real tokens only; the encoder runs per-document over those, so no
        // padding is needed for either the single or the batched path.
        tokenizer.with_padding(None);

        let weights_path = dir.join(SAFETENSORS_FALLBACK);
        if !weights_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{model_name} (missing verified {SAFETENSORS_FALLBACK} in {})",
                    dir.display()
                ),
            });
        }

        let shared = parse_weights(&weights_path, profile.linear_precision())?;
        let model = build_model(shared)?;
        if model.encoder_layers() != profile.encoder_layers() {
            return Err(SearchError::ModelLoadFailed {
                path: weights_path,
                source: format!(
                    "registered model requires {} encoder layers, weights contain {}",
                    profile.encoder_layers(),
                    model.encoder_layers()
                )
                .into(),
            });
        }

        let embedder = Self {
            inner: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            admission: Arc::new(AsyncMutex::new(())),
            blocking_pool: None,
            max_length: DEFAULT_MAX_LENGTH,
            name: model_name.to_owned(),
            id: profile.embedder_id().to_owned(),
            identity,
        };
        // Verified artifacts alone cannot attest the executing kernels. Exercise
        // the same public batch path before any caller can obtain this identity.
        let texts = &frankensearch_embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        let probe =
            embedder
                .embed_batch_sync(texts)
                .map_err(|error| SearchError::ModelLoadFailed {
                    path: dir.to_path_buf(),
                    source: format!("failed to run native producer conformance probe: {error}")
                        .into(),
                })?;
        embedder
            .identity
            .producer
            .golden_vectors
            .verify_exact_f32(texts, &probe)
            .map_err(|_| SearchError::ModelLoadFailed {
                path: dir.to_path_buf(),
                source: "native execution does not match the registered producer certificate; use a qualified runtime build before rebuilding the index (model files already verified)"
                    .into(),
            })?;

        tracing::info!(
            model = model_name,
            dimension = DIM,
            encoder_layers = profile.encoder_layers(),
            max_length = DEFAULT_MAX_LENGTH,
            manifest = %verified.frozen().fingerprint,
            identity = %embedder.identity.fingerprint(),
            precision = ?profile.linear_precision(),
            "native frankentorch MiniLM embedder loaded (mean-pool + L2)"
        );

        Ok(embedder)
    }

    /// Attach the caller's bounded blocking pool for async inference.
    ///
    /// The caller must retain and fully drain the pool at shutdown. Cancelling
    /// a waiting future cannot preempt an executing tensor kernel; its worker
    /// retains model admission until it finishes. No internal runtime is made.
    /// Without an attached pool, async inference returns an actionable error.
    #[must_use]
    pub fn with_blocking_pool(mut self, pool: BlockingPoolHandle) -> Self {
        self.blocking_pool = Some(pool);
        self
    }

    async fn infer(&self, cx: &Cx, texts: Vec<String>) -> SearchResult<Vec<Vec<f32>>> {
        native_checkpoint(cx)?;
        let pool = self
            .blocking_pool
            .clone()
            .ok_or_else(|| SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: "native async inference requires a caller-owned blocking pool; attach it with NativeEmbedder::with_blocking_pool"
                    .into(),
            })?;
        // Obtain admission before spawning, so cancelled waiters cannot fill
        // the blocking pool with workers waiting on this model's sync mutex.
        let admission = OwnedMutexGuard::lock(Arc::clone(&self.admission), cx)
            .await
            .map_err(|error| match error {
                LockError::Cancelled => SearchError::Cancelled {
                    phase: "native.infer".to_owned(),
                    reason: "native model admission cancelled".to_owned(),
                },
                error => SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("native model admission failed: {error}").into(),
                },
            })?;
        let owner = self.clone();
        let worker_cx = cx.clone().with_blocking_pool_handle(Some(pool));
        let mut worker = worker_cx
            .spawn_blocking(move |child| {
                let _admission = admission;
                let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
                owner.embed_batch_checked(&texts, Some(&child))
            })
            .map_err(|error| SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!("cannot admit native inference worker: {error}").into(),
            })?;
        worker.join(cx).await.map_err(|error| match error {
            asupersync::runtime::JoinError::Cancelled(_) => SearchError::Cancelled {
                phase: "native.infer".to_owned(),
                reason: "native inference worker cancelled".to_owned(),
            },
            error => SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!("native inference worker failed: {error}").into(),
            },
        })?
    }

    /// Tokenize one text to token ids (with `[CLS]`/`[SEP]`), truncated to `max_length`.
    fn tokenize(&self, text: &str) -> SearchResult<Vec<i64>> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: format!("tokenize failed: {e}").into(),
                })?;
        Ok(crate::ids_to_truncated_i64(
            encoding.get_ids(),
            self.max_length,
        ))
    }

    fn lock_model(&self) -> SearchResult<std::sync::MutexGuard<'_, Model>> {
        self.inner.lock().map_err(|e| SearchError::EmbeddingFailed {
            model: self.name.clone(),
            source: format!("embedder mutex poisoned: {e}").into(),
        })
    }

    fn embed_batch_checked(&self, texts: &[&str], cx: Option<&Cx>) -> SearchResult<Vec<Vec<f32>>> {
        if let Some(cx) = cx {
            native_checkpoint(cx)?;
        }
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let token_batches: Vec<Vec<i64>> = texts
            .iter()
            .map(|t| self.tokenize(t))
            .collect::<SearchResult<_>>()?;
        let mut model = self.lock_model()?;
        let mut out = Vec::with_capacity(texts.len());
        // Chunk inputs by total token budget so each forward's intermediates stay
        // bounded; a single over-budget input is still run alone.
        let mut start = 0usize;
        while start < token_batches.len() {
            if let Some(cx) = cx {
                native_checkpoint(cx)?;
            }
            let mut end = start;
            let mut tok = 0usize;
            while end < token_batches.len() {
                let len = token_batches[end].len().max(1);
                if end > start && tok + len > MAX_BATCH_TOKENS {
                    break;
                }
                tok += len;
                end += 1;
            }
            out.extend(model.embed_forward(&token_batches[start..end])?);
            start = end;
        }
        drop(model);
        if let Some(cx) = cx {
            native_checkpoint(cx)?;
        }
        if out.len() != texts.len() || out.iter().any(|vector| vector.len() != DIM) {
            return Err(SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source:
                    "native backend returned a batch shape inconsistent with its attested identity"
                        .into(),
            });
        }
        Ok(out)
    }
}

fn native_checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "native.infer".to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

impl frankensearch_core::traits::Embedder for NativeEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            native_checkpoint(cx)?;
            self.infer(cx, vec![text.to_owned()])
                .await?
                .pop()
                .ok_or_else(|| SearchError::EmbeddingFailed {
                    model: self.name.clone(),
                    source: "native backend returned no embedding".into(),
                })
        })
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            native_checkpoint(cx)?;
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            self.infer(cx, texts.iter().map(|text| (*text).to_owned()).collect())
                .await
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        SyncEmbed::identity(self)
    }

    fn dimension(&self) -> usize {
        DIM
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_ready(&self) -> bool {
        self.blocking_pool.is_some()
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

impl SyncEmbed for NativeEmbedder {
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        let ids = self.tokenize(text)?;
        let mut model = self.lock_model()?;
        let mut out = model.embed_forward(&[ids])?;
        drop(model);
        let vector = out.pop().ok_or_else(|| SearchError::EmbeddingFailed {
            model: self.name.clone(),
            source: "native backend returned no embedding".into(),
        })?;
        if vector.len() != DIM {
            return Err(SearchError::EmbeddingFailed {
                model: self.name.clone(),
                source: format!(
                    "native backend returned dimension {}, expected {DIM}",
                    vector.len()
                )
                .into(),
            });
        }
        Ok(vector)
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        self.embed_batch_checked(texts, None)
    }

    fn dimension(&self) -> usize {
        DIM
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        &self.id
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

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-level proof that NativeEmbedder satisfies the embedder contract.
    const fn assert_sync_embed<T: SyncEmbed>() {}
    const _: () = assert_sync_embed::<NativeEmbedder>();
    const fn assert_async_embed<T: frankensearch_core::traits::Embedder>() {}
    const _: () = assert_async_embed::<NativeEmbedder>();

    #[test]
    #[ignore = "requires verified native MiniLM assets via MINILM_FIXTURE_DIR"]
    fn async_fixture_preserves_vectors_and_requires_explicit_pool() {
        use frankensearch_core::traits::Embedder;
        use frankensearch_embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;

        let dir = std::env::var("MINILM_FIXTURE_DIR").expect("native fixture required");
        let native = NativeEmbedder::load_model(dir, NativeEmbeddingModel::AllMiniLmL6V2F32)
            .expect("load actual F32 producer");
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
        assert!(!Embedder::is_ready(&native));
        let error = runtime
            .block_on(Embedder::embed(&native, &cx, "hello world"))
            .expect_err("a missing pool must never silently run inline");
        assert!(error.to_string().contains("caller-owned blocking pool"));

        let native = native.with_blocking_pool(runtime.blocking_handle().unwrap());
        assert!(Embedder::is_ready(&native));
        assert_eq!(
            Embedder::identity(&native).unwrap(),
            SyncEmbed::identity(&native).unwrap()
        );
        let expected = native
            .embed_batch_sync(&MODEL_CONFORMANCE_TEXTS_V1)
            .unwrap();
        let actual = runtime
            .block_on(Embedder::embed_batch(
                &native,
                &cx,
                &MODEL_CONFORMANCE_TEXTS_V1,
            ))
            .unwrap();
        assert_eq!(
            actual, expected,
            "async batching must preserve exact producer bits"
        );
        for text in ["", "hello world", "identifier fsvi_v2"] {
            assert_eq!(
                runtime
                    .block_on(Embedder::embed(&native, &cx, text))
                    .unwrap(),
                native.embed_sync(text).unwrap(),
                "async single inference must preserve exact producer bits"
            );
        }
        let cancelled = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
        cancelled.cancel_fast(asupersync::CancelKind::User);
        for result in [
            runtime
                .block_on(Embedder::embed(&native, &cancelled, "cancelled"))
                .map(|_| ()),
            runtime
                .block_on(Embedder::embed_batch(&native, &cancelled, &[]))
                .map(|_| ()),
        ] {
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        }
        assert!(native.admission.try_lock().is_ok());
        assert!(runtime.shutdown_timeout(std::time::Duration::from_secs(5)));
    }

    #[test]
    #[ignore = "requires verified native MiniLM assets via MINILM_FIXTURE_DIR"]
    fn async_fixture_timeout_retains_worker_admission_until_drain() {
        use frankensearch_core::traits::Embedder;
        use std::future::poll_fn;
        use std::task::Poll;
        use std::time::{Duration, Instant};

        let dir = std::env::var("MINILM_FIXTURE_DIR").expect("native fixture required");
        let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
            .blocking_threads(0, 2)
            .build()
            .unwrap();
        let pool = runtime.blocking_handle().unwrap();
        let native = NativeEmbedder::load_model(dir, NativeEmbeddingModel::AllMiniLmL6V2F32)
            .unwrap()
            .with_blocking_pool(pool.clone());
        let cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
        let waiting_cx = runtime.request_cx_with_budget(asupersync::types::Budget::INFINITE);
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            // Hold the real model, not a mock inference callback. A finite
            // release fuse makes an inline-execution regression fail, not hang.
            let model = Arc::clone(&native.inner);
            scope.spawn(move || {
                let _model = model.lock().unwrap();
                held_tx.send(()).unwrap();
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            });
            held_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            runtime.block_on(async {
                let mut running = Embedder::embed(&native, &cx, "hello world");
                poll_fn(|task_cx| {
                    assert!(running.as_mut().poll(task_cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                let started = Instant::now();
                while pool.busy_threads() != 1 {
                    assert!(
                        started.elapsed() < Duration::from_secs(2),
                        "worker did not start"
                    );
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
                assert!(native.admission.try_lock().is_err());
                assert!(
                    asupersync::time::timeout(cx.now(), Duration::from_millis(20), running)
                        .await
                        .is_err()
                );
                assert!(
                    started.elapsed() < Duration::from_secs(2),
                    "executor was blocked"
                );
                assert_eq!(pool.busy_threads(), 1);
                assert!(
                    native.admission.try_lock().is_err(),
                    "a timed-out waiter must not release its worker's admission"
                );

                let mut waiting = Embedder::embed(&native, &waiting_cx, "second query");
                poll_fn(|task_cx| {
                    assert!(waiting.as_mut().poll(task_cx).is_pending());
                    Poll::Ready(())
                })
                .await;
                waiting_cx.cancel_fast(asupersync::CancelKind::User);
                assert!(matches!(waiting.await, Err(SearchError::Cancelled { .. })));
                assert_eq!(
                    pool.busy_threads(),
                    1,
                    "cancelled admission must not occupy another worker"
                );
                assert_eq!(pool.pending_count(), 0);
                release_tx.send(()).unwrap();
                let drain_started = Instant::now();
                while pool.busy_threads() != 0 {
                    assert!(
                        drain_started.elapsed() < Duration::from_secs(2),
                        "worker did not drain"
                    );
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
                assert!(native.admission.try_lock().is_ok());
                assert_eq!(
                    Embedder::embed(&native, &cx, "healthy query")
                        .await
                        .unwrap(),
                    native.embed_sync("healthy query").unwrap()
                );
            });
        });
        assert!(runtime.shutdown_timeout(Duration::from_secs(5)));
    }

    #[test]
    fn registered_identity_matches_native_backend_contract() {
        let identity = ModelArtifactManifestV1::minilm_native_frankentorch()
            .expect("registered native MiniLM manifest")
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .expect("derive native MiniLM identity");
        assert_eq!(identity.space.dimension, IDENTITY_DIMENSION);
        assert_eq!(identity.space.sequence_policy, IDENTITY_SEQUENCE_POLICY);
        assert_eq!(identity.space.pooling, IDENTITY_POOLING);
        assert_eq!(
            identity.space.output_normalization,
            IDENTITY_OUTPUT_NORMALIZATION
        );
    }

    #[test]
    fn multilingual_identity_is_distinct_from_same_dimension_minilm() {
        let baseline = ModelArtifactManifestV1::minilm_native_frankentorch()
            .expect("registered native MiniLM manifest")
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .expect("derive native MiniLM identity");
        let multilingual = ModelArtifactManifestV1::multilingual_minilm_native_frankentorch()
            .expect("registered multilingual MiniLM manifest")
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .expect("derive multilingual MiniLM identity");

        assert_eq!(baseline.space.dimension, multilingual.space.dimension);
        assert_ne!(
            baseline.space.fingerprint(),
            multilingual.space.fingerprint()
        );
        assert!(
            baseline.verify_exact_producer_with(&multilingual).is_err(),
            "same dimensionality must not admit vectors from a different model space"
        );
    }

    #[test]
    fn f32_identity_cannot_substitute_for_int8_or_onnx() {
        let f32 = NativeEmbeddingModel::AllMiniLmL6V2F32
            .manifest()
            .unwrap()
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .unwrap();
        for manifest in [
            NativeEmbeddingModel::AllMiniLmL6V2.manifest().unwrap(),
            ModelArtifactManifestV1::minilm_fastembed().unwrap(),
        ] {
            let other = manifest
                .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
                .unwrap();
            assert_eq!(f32.space.dimension, other.space.dimension);
            assert!(
                f32.verify_exact_producer_with(&other).is_err(),
                "same dimensions and close output must not admit a foreign producer"
            );
        }
    }

    #[test]
    #[ignore = "requires verified native MiniLM assets via MINILM_FIXTURE_DIR"]
    fn f32_fixture_proves_certificate_batching_and_repeatability() {
        use frankensearch_core::generation::GoldenVectorCertificateV1;
        use frankensearch_embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        let dir = std::env::var("MINILM_FIXTURE_DIR").expect("native fixture required");
        let profile = NativeEmbeddingModel::AllMiniLmL6V2F32;
        let manifest = profile.manifest().unwrap();
        let embedder =
            NativeEmbedder::load_model(&dir, profile).expect("load verified F32 producer");
        assert_eq!(embedder.id(), "minilm-384-native-f32");
        assert_eq!(
            embedder.identity().unwrap(),
            &manifest
                .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
                .unwrap()
        );
        let vectors = embedder
            .embed_batch_sync(&MODEL_CONFORMANCE_TEXTS_V1)
            .unwrap();
        assert_eq!(
            GoldenVectorCertificateV1::from_exact_f32(&MODEL_CONFORMANCE_TEXTS_V1, &vectors)
                .unwrap(),
            manifest.execution.golden_vectors,
            "F32 producer output bits drifted"
        );
        let long = "Search finds related documents across the library. ".repeat(100);
        let texts = ["", "hello world", "identifier fsvi_v2", long.as_str()];
        assert_eq!(embedder.tokenize(&long).unwrap().len(), DEFAULT_MAX_LENGTH);
        let batch = embedder.embed_batch_sync(&texts).unwrap();
        let repeated = embedder.embed_batch_sync(&texts).unwrap();
        assert_eq!(
            batch, repeated,
            "repeated F32 batches must be deterministic"
        );
        assert_eq!(batch.len(), texts.len());
        assert!(embedder.embed_batch_sync(&[]).unwrap().is_empty());
        for (text, vector) in texts.iter().zip(&batch) {
            let single = embedder.embed_sync(text).unwrap();
            assert_eq!(vector.len(), DIM);
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
            for (&a, b) in vector.iter().zip(single) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "single vs batch F32 drift: {a} vs {b}"
                );
            }
        }

        // Valid, identical model bytes must not let F32 execution attest int8
        // output. Exercise the owning constructor, not just the verifier alone.
        let mut nonconformant = manifest;
        nonconformant.execution.golden_vectors = NativeEmbeddingModel::AllMiniLmL6V2
            .manifest()
            .unwrap()
            .execution
            .golden_vectors;
        let error = NativeEmbedder::load_from_manifest(Path::new(&dir), profile, &nonconformant)
            .expect_err("F32 execution must not attest the original int8 producer");
        let SearchError::ModelLoadFailed { source, .. } = error else {
            panic!("expected execution conformance refusal, got {error}");
        };
        assert!(source.to_string().contains("producer certificate"));
    }

    /// Smoke test against a real `all-MiniLM-L6-v2` directory. Ignored by default
    /// (no model fixture in CI); run with `MINILM_FIXTURE_DIR=<dir> cargo test -p
    /// frankensearch-rerank --features native -- --ignored native_embedder`.
    #[test]
    #[ignore = "requires a local all-MiniLM-L6-v2 model dir via MINILM_FIXTURE_DIR"]
    fn embeds_unit_vector_from_fixture() {
        let dir = std::env::var("MINILM_FIXTURE_DIR")
            .expect("set MINILM_FIXTURE_DIR to an all-MiniLM-L6-v2 model directory");
        let embedder = NativeEmbedder::load(&dir).expect("load native MiniLM embedder");
        assert_eq!(embedder.dimension(), DIM);
        let v = embedder.embed_sync("hello world").expect("embed");
        assert_eq!(v.len(), DIM, "embedding dimensionality");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected L2-normalized unit vector, got norm {norm}"
        );
        // Batch path agrees with the single path.
        let batch = embedder
            .embed_batch_sync(&["hello world", "a second sentence"])
            .expect("batch embed");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), DIM);
        let cos: f32 = v.iter().zip(&batch[0]).map(|(a, b)| a * b).sum();
        assert!(
            cos > 0.999,
            "single vs batch embedding mismatch (cos {cos})"
        );
    }

    /// Bit-exact producer conformance proof for the frozen native `MiniLM`
    /// certificate. This is intentionally fixture-gated because CI does not
    /// provision the 90 MiB model bundle in ordinary unit-test lanes.
    #[test]
    #[ignore = "requires a verified all-MiniLM-L6-v2 model dir via MINILM_FIXTURE_DIR"]
    fn conformance_certificate_matches_fixture() {
        let dir = std::env::var("MINILM_FIXTURE_DIR")
            .expect("set MINILM_FIXTURE_DIR to an all-MiniLM-L6-v2 model directory");
        let manifest = ModelArtifactManifestV1::minilm_native_frankentorch()
            .expect("registered native MiniLM manifest");
        let expected_identity = manifest
            .declared_identity_bundle(QuantizationFormat::F32, "in-memory-f32-v1")
            .expect("derive registered native MiniLM identity");
        let embedder = NativeEmbedder::load(&dir).expect("load native MiniLM embedder");
        assert_eq!(embedder.identity().unwrap(), &expected_identity);
        let texts = &frankensearch_embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
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
            "native MiniLM output bits drifted from the registered producer certificate"
        );
    }

    fn cosine(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    /// Real multilingual proof against the immutable 12-layer fixture. The test
    /// covers native XLM-R/SentencePiece tokenization, exact topology admission,
    /// Chinese-to-English and English-to-Chinese retrieval, mixed code/text, and
    /// bit-exact repeatability.
    #[test]
    #[ignore = "requires a verified multilingual MiniLM model dir via MULTILINGUAL_MINILM_FIXTURE_DIR"]
    fn multilingual_fixture_proves_cross_language_retrieval_and_determinism() {
        let dir = std::env::var("MULTILINGUAL_MINILM_FIXTURE_DIR")
            .expect("set MULTILINGUAL_MINILM_FIXTURE_DIR to paraphrase-multilingual-MiniLM-L12-v2");
        let load_started = std::time::Instant::now();
        let embedder = NativeEmbedder::load_multilingual(&dir)
            .expect("load verified multilingual MiniLM embedder");
        let load_elapsed = load_started.elapsed();
        assert_eq!(embedder.dimension(), DIM);
        assert_eq!(embedder.id(), MULTILINGUAL_EMBEDDER_ID);
        assert_eq!(
            embedder.lock_model().expect("lock model").encoder_layers(),
            12
        );

        let chinese_ids = embedder
            .tokenize("如何修复数据库事务死锁？")
            .expect("tokenize Chinese query");
        assert!(
            chinese_ids.len() > 4,
            "native multilingual tokenizer collapsed Chinese input"
        );

        let texts = [
            "如何在 Rust 中处理任务取消和结构化并发？",
            "In Rust, structured concurrency keeps child tasks scoped and propagates cancellation safely.",
            "A sourdough starter needs flour, water, and a warm kitchen.",
            "How should a database transaction deadlock be resolved?",
            "数据库事务发生死锁时，应回滚其中一个事务，并按固定顺序重试锁操作。",
            "这份食谱介绍如何烤制苹果派和准备奶油馅料。",
            "修复 Rust async cancellation bug in worker_queue.rs",
            "worker_queue.rs 必须在 async 任务取消时归还 reservation，避免消息丢失。",
            "The watercolor landscape uses blue pigment and cold-press paper.",
        ];
        let first_started = std::time::Instant::now();
        let first = embedder
            .embed_batch_sync(&texts)
            .expect("embed multilingual retrieval fixture");
        let first_elapsed = first_started.elapsed();
        let repeat_started = std::time::Instant::now();
        let second = embedder
            .embed_batch_sync(&texts)
            .expect("repeat multilingual retrieval fixture");
        let repeat_elapsed = repeat_started.elapsed();
        assert_eq!(
            first, second,
            "native multilingual output must be bit-exact"
        );

        for (query, relevant, distractor, label) in [
            (0, 1, 2, "Chinese query to English discussion"),
            (3, 4, 5, "English query to Chinese discussion"),
            (6, 7, 8, "mixed Chinese/code query"),
        ] {
            let relevant_score = cosine(&first[query], &first[relevant]);
            let distractor_score = cosine(&first[query], &first[distractor]);
            assert!(
                relevant_score > distractor_score + 0.05,
                "{label} failed: relevant={relevant_score}, distractor={distractor_score}"
            );
        }

        let manifest = ModelArtifactManifestV1::multilingual_minilm_native_frankentorch()
            .expect("registered multilingual MiniLM manifest");
        let conformance_texts = &frankensearch_embed::model_manifest::MODEL_CONFORMANCE_TEXTS_V1;
        let conformance_started = std::time::Instant::now();
        let vectors = embedder
            .embed_batch_sync(conformance_texts)
            .expect("embed bounded conformance corpus");
        let conformance_elapsed = conformance_started.elapsed();
        let observed = frankensearch_core::generation::GoldenVectorCertificateV1::from_exact_f32(
            conformance_texts,
            &vectors,
        )
        .expect("compute exact multilingual conformance certificate");
        assert_eq!(observed, manifest.execution.golden_vectors);
        eprintln!(
            "multilingual_native_metrics load_ms={} first_9_ms={} repeat_9_ms={} conformance_4_ms={}",
            load_elapsed.as_millis(),
            first_elapsed.as_millis(),
            repeat_elapsed.as_millis(),
            conformance_elapsed.as_millis()
        );
    }
}
