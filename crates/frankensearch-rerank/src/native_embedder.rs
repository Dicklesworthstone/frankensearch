//! Pure-Rust transformer sentence-embedder (`all-MiniLM-L6-v2`, 384-dim) backed by
//! frankentorch — the embedding counterpart of [`crate::native::NativeReranker`] (no
//! ONNX / no `ort`).
//!
//! It reuses the reranker's validated, SIMD/int8-optimized BERT encoder verbatim
//! (same 6-layer `MiniLM` forward, same kernels via [`crate::native::Model::embed_forward`]);
//! it differs only at the head — **mean-pool over every token + L2-normalize** instead
//! of the `[CLS]` pooler + classifier — and in tokenization (one text, token-type ids
//! all 0). Because there is no ONNX Runtime, there is no AVX-static-init hazard: the
//! int8 GEMM dispatches NEON (aarch64 SDOT / NR=4 packing) or x86 SIMD at runtime.
//!
//! Feature-gated behind `native`.

use std::path::Path;
use std::sync::Mutex;

use tokenizers::Tokenizer;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{EmbeddingIdentityBundleV1, QuantizationFormat};
use frankensearch_core::traits::{ModelCategory, SyncEmbed};
use frankensearch_embed::model_manifest::ModelArtifactManifestV1;

use crate::native::{
    DEFAULT_MAX_LENGTH, Model, SAFETENSORS_FALLBACK, TOKENIZER_JSON, build_model, parse_weights,
};

const MODEL_NAME: &str = "all-minilm-l6-v2";
const EMBEDDER_ID: &str = "minilm-384-native";
const DIM: usize = 384;
const IDENTITY_DIMENSION: u32 = 384;
const IDENTITY_SEQUENCE_POLICY: &str = "max-length=512;longest-first;no-padding";
const IDENTITY_POOLING: &str = "mean-all-returned-tokens-including-specials-no-padding-v1";
const IDENTITY_OUTPUT_NORMALIZATION: &str = "l2-f32-if-norm-gt-zero-else-unchanged-v1";
/// Token budget per batched forward (mirrors the reranker's chunking) so each
/// forward's attention intermediates stay memory-bounded.
const MAX_BATCH_TOKENS: usize = 2048;

/// Pure-Rust frankentorch `MiniLM` sentence-embedder.
pub struct NativeEmbedder {
    /// One frankentorch session behind a `Mutex` (each forward parallelizes internally
    /// across cores; calls are serialized, so no nested-rayon-under-lock hazard) — same
    /// pattern as [`crate::native::NativeReranker`].
    inner: Mutex<Model>,
    tokenizer: Tokenizer,
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
    /// Load from a model directory containing `tokenizer.json` and a safetensors weight
    /// file (`model.safetensors`) — the standard immutable
    /// `sentence-transformers/all-MiniLM-L6-v2` layout (bare
    /// `embeddings.*`/`encoder.*` keys are normalized to the shared
    /// `bert.`-prefixed scheme during parse).
    ///
    /// # Errors
    /// [`SearchError::ModelNotFound`] when required files are missing;
    /// [`SearchError::ModelLoadFailed`] when the tokenizer or weights fail to load.
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        let dir = model_dir.as_ref();
        let verified = ModelArtifactManifestV1::minilm_native_frankentorch()?.verify_dir(dir)?;
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
                    "{MODEL_NAME} (missing {TOKENIZER_JSON} in {})",
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
                    "{MODEL_NAME} (missing verified {SAFETENSORS_FALLBACK} in {})",
                    dir.display()
                ),
            });
        }

        let shared = parse_weights(&weights_path)?;
        let model = build_model(&shared)?;

        tracing::info!(
            model = MODEL_NAME,
            dimension = DIM,
            max_length = DEFAULT_MAX_LENGTH,
            manifest = %verified.frozen().fingerprint,
            identity = %identity.fingerprint(),
            "native frankentorch MiniLM embedder loaded (int8 linear, mean-pool + L2)"
        );

        Ok(Self {
            inner: Mutex::new(model),
            tokenizer,
            max_length: DEFAULT_MAX_LENGTH,
            name: MODEL_NAME.to_owned(),
            id: EMBEDDER_ID.to_owned(),
            identity,
        })
    }

    /// Tokenize one text to token ids (with `[CLS]`/`[SEP]`), truncated to `max_length`.
    fn tokenize(&self, text: &str) -> SearchResult<Vec<i64>> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|e| SearchError::EmbeddingFailed {
                    model: MODEL_NAME.to_owned(),
                    source: format!("tokenize failed: {e}").into(),
                })?;
        Ok(crate::ids_to_truncated_i64(
            encoding.get_ids(),
            self.max_length,
        ))
    }

    fn lock_model(&self) -> SearchResult<std::sync::MutexGuard<'_, Model>> {
        self.inner.lock().map_err(|e| SearchError::EmbeddingFailed {
            model: MODEL_NAME.to_owned(),
            source: format!("embedder mutex poisoned: {e}").into(),
        })
    }
}

impl SyncEmbed for NativeEmbedder {
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        let ids = self.tokenize(text)?;
        let mut model = self.lock_model()?;
        let mut out = model.embed_forward(&[ids])?;
        drop(model);
        let vector = out.pop().ok_or_else(|| SearchError::EmbeddingFailed {
            model: MODEL_NAME.to_owned(),
            source: "native backend returned no embedding".into(),
        })?;
        if vector.len() != DIM {
            return Err(SearchError::EmbeddingFailed {
                model: MODEL_NAME.to_owned(),
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
        if out.len() != texts.len() || out.iter().any(|vector| vector.len() != DIM) {
            return Err(SearchError::EmbeddingFailed {
                model: MODEL_NAME.to_owned(),
                source:
                    "native backend returned a batch shape inconsistent with its attested identity"
                        .into(),
            });
        }
        Ok(out)
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
}
