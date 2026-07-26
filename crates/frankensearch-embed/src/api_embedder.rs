//! Cloud API embedder implementing the `Embedder` trait.
//!
//! Wraps any [`super::api_provider::ApiProvider`] with HTTP transport, retry
//! logic, rate limiting, and L2 normalization. Gated behind the `api` feature.

use std::fmt;
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::body::{Body, Frame};
use asupersync::http::h1::{HttpClient, HttpClientConfig, Method, RedirectPolicy};
use tracing::{debug, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, FrozenEmbeddingIdentityBundleV1,
    QuantizationFormat,
};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture, l2_normalize_in_place};

use crate::api_provider::{ApiProvider, RemoteEmbeddingAttestationV1};
use crate::cached_embedder::CachedEmbedder;

const API_OUTPUT_NORMALIZATION_V1: &str = "l2-f32-zero-on-degenerate-v1";
const API_STORAGE_FORMAT_V1: &str = "in-memory-f32-v1";
const API_STORAGE_ENDIANNESS_V1: &str = "native-f32-values";

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for API embedder HTTP behavior.
#[derive(Debug, Clone)]
pub struct ApiEmbedderConfig {
    /// Maximum retries on transient failure (429, 5xx).
    pub max_retries: u32,
    /// Base delay for exponential backoff.
    pub retry_base_delay: Duration,
    /// Requests per minute limit (0 = unlimited).
    pub requests_per_minute: u32,
}

impl Default for ApiEmbedderConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_millis(500),
            requests_per_minute: 0,
        }
    }
}

// ─── Rate Limiter ───────────────────────────────────────────────────────────

/// Simple token-bucket rate limiter for API calls.
#[derive(Debug)]
struct RateLimiter {
    state: Mutex<RateLimiterState>,
    requests_per_minute: u32,
}

#[derive(Debug)]
struct RateLimiterState {
    available_permits: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            state: Mutex::new(RateLimiterState {
                available_permits: f64::from(requests_per_minute),
                last_refill: Instant::now(),
            }),
            requests_per_minute,
        }
    }

    /// Returns the duration to wait before making a request, or `None` if
    /// a token is available immediately.
    fn acquire(&self) -> Option<Duration> {
        if self.requests_per_minute == 0 {
            return None;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let requests_per_minute = f64::from(self.requests_per_minute);
        let refill = elapsed * (requests_per_minute / 60.0);
        state.available_permits = (state.available_permits + refill).min(requests_per_minute);
        state.last_refill = now;

        if state.available_permits >= 1.0 {
            state.available_permits -= 1.0;
            drop(state);
            None
        } else {
            let deficit = 1.0 - state.available_permits;
            state.available_permits = 0.0; // consume all partial permits
            drop(state);
            let wait_secs = deficit / (requests_per_minute / 60.0);
            Some(Duration::from_secs_f64(wait_secs))
        }
    }
}

// ─── ApiEmbedder ────────────────────────────────────────────────────────────

/// Cloud API embedder wrapping any [`ApiProvider`].
///
/// Handles HTTP transport, retry with exponential backoff, rate limiting,
/// batch chunking, and L2 normalization. Every successful response must carry
/// a space/producer attestation matching the construction-time epoch; an epoch
/// supplied by the caller does not authenticate the responding service.
pub struct ApiEmbedder {
    provider: Box<dyn ApiProvider>,
    client: HttpClient,
    rate_limiter: RateLimiter,
    config: ApiEmbedderConfig,
    identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for ApiEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiEmbedder")
            .field(
                "provider",
                &bounded_remote_producer_label(self.provider.provider_name()),
            )
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl ApiEmbedder {
    /// Create an API embedder bound to an explicit immutable remote space epoch.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] if the epoch is absent,
    /// malformed, non-semantic, or incompatible with the provider/output
    /// storage contract. Embedding requests also return that error if the
    /// provider response omits or drifts from its identity attestation.
    pub fn new(
        provider: Box<dyn ApiProvider>,
        config: ApiEmbedderConfig,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        let frozen_identity = immutable_space_epoch.ok_or_else(|| {
            unverifiable_remote_space(
                provider.provider_name(),
                "no explicit immutable space epoch was supplied",
            )
        })?;
        frozen_identity.validate().map_err(|_error| {
            unverifiable_remote_space(
                provider.provider_name(),
                "explicit immutable space epoch failed canonical validation",
            )
        })?;
        let identity = frozen_identity.identity;
        if identity.space.kind != EmbeddingSpaceKindV1::Semantic {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "remote API embedders require a semantic space identity",
            ));
        }
        if identity.space.logical_model_id != provider.api_model_id() {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "explicit space epoch model disagrees with the provider contract",
            ));
        }
        if identity.producer.backend != provider.identity_backend() {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "explicit producer backend disagrees with the provider contract",
            ));
        }
        if identity.producer.protocol_revision != provider.identity_protocol_revision() {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "explicit producer protocol disagrees with the provider contract",
            ));
        }
        if identity.space.output_normalization != API_OUTPUT_NORMALIZATION_V1 {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "explicit space epoch normalization disagrees with the API embedder contract",
            ));
        }
        if identity.storage.quantization != QuantizationFormat::F32
            || identity.storage.format != API_STORAGE_FORMAT_V1
            || identity.storage.endianness != API_STORAGE_ENDIANNESS_V1
        {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "explicit storage epoch disagrees with the API embedder's in-memory f32 output contract",
            ));
        }
        let provider_dimension = u32::try_from(provider.dimension()).map_err(|_| {
            unverifiable_remote_space(
                provider.provider_name(),
                "provider dimension does not fit the identity schema",
            )
        })?;
        if identity.space.dimension != provider_dimension {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "explicit space epoch dimension disagrees with the provider contract",
            ));
        }

        let mut client_config = HttpClientConfig::default();
        client_config.redirect_policy = RedirectPolicy::Limited(5);
        client_config.user_agent = Some(format!(
            "frankensearch/{} (api-embedder)",
            env!("CARGO_PKG_VERSION")
        ));
        let rate_limiter = RateLimiter::new(config.requests_per_minute);
        Ok(Self {
            provider,
            client: HttpClient::with_config(client_config),
            rate_limiter,
            config,
            identity,
        })
    }

    /// Create with default configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] when the immutable
    /// epoch is absent or incompatible with the provider.
    pub fn with_defaults(
        provider: Box<dyn ApiProvider>,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        Self::new(
            provider,
            ApiEmbedderConfig::default(),
            immutable_space_epoch,
        )
    }

    /// Wrap this embedder with a cache (convenience).
    #[must_use]
    pub fn cached(self, capacity: usize) -> CachedEmbedder {
        CachedEmbedder::new(Arc::new(self), capacity)
    }

    /// Wrap with the default cache capacity (4096 entries).
    #[must_use]
    pub fn cached_default(self) -> CachedEmbedder {
        CachedEmbedder::new(Arc::new(self), 4096)
    }

    /// Make a single API request for a batch of texts, with retry.
    async fn request_batch(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        let body = self.provider.serialize_request(texts)?;
        let headers = self.provider.request_headers();

        let url = self.provider.request_url();

        let mut last_err = None;
        'retry: for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let backoff = self.config.retry_base_delay * 2u32.pow(attempt - 1);
                debug!(
                    provider = self.provider.provider_name(),
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    "retrying API request"
                );
                asupersync::time::sleep(asupersync::time::wall_now(), backoff).await;
            }

            // Rate limit.
            if let Some(wait) = self.rate_limiter.acquire() {
                asupersync::time::sleep(asupersync::time::wall_now(), wait).await;
            }

            let response = self
                .client
                .request_streaming(
                    cx,
                    Method::Post,
                    &url,
                    headers
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    body.clone(),
                )
                .await;

            let mut response = match response {
                Ok(r) => r,
                Err(_error) => {
                    last_err = Some(SearchError::EmbeddingFailed {
                        model: self.provider.embedder_id().to_owned(),
                        source: "remote HTTP transport failed".into(),
                    });
                    continue;
                }
            };

            let status = response.head.status;

            // Collect response body.
            let mut response_body = Vec::new();
            while let Some(frame) = poll_fn(|cx| Pin::new(&mut response.body).poll_frame(cx)).await
            {
                match frame {
                    Ok(Frame::Data(mut chunk)) => {
                        while chunk.has_remaining() {
                            let bytes = chunk.chunk();
                            if bytes.is_empty() {
                                break;
                            }
                            response_body.extend_from_slice(bytes);
                            chunk.advance(bytes.len());
                        }
                    }
                    Ok(Frame::Trailers(_)) => {}
                    Err(_error) => {
                        last_err = Some(SearchError::EmbeddingFailed {
                            model: self.provider.embedder_id().to_owned(),
                            source: "remote HTTP response body failed".into(),
                        });
                        continue 'retry;
                    }
                }
            }

            // Success.
            if (200..300).contains(&status) {
                let embeddings = self.provider.deserialize_response(&response_body)?;
                let attestation =
                    self.provider
                        .response_attestation(&response_body)
                        .map_err(|_error| {
                            unverifiable_remote_space(
                                self.provider.provider_name(),
                                "per-response space or producer attestation is malformed",
                            )
                        })?;
                self.verify_response_attestation(attestation.as_ref())?;
                self.verify_response_vectors(&embeddings, texts.len())?;
                return Ok(embeddings);
            }

            // Retry on 429 or 5xx.
            if status == 429 || status >= 500 {
                warn!(
                    provider = self.provider.provider_name(),
                    status, attempt, "transient API error"
                );
                last_err = Some(SearchError::EmbeddingFailed {
                    model: self.provider.embedder_id().to_owned(),
                    source: format!("HTTP {status} remote provider error").into(),
                });
                continue;
            }

            // Non-retryable client error (4xx other than 429).
            return Err(SearchError::EmbeddingFailed {
                model: self.provider.embedder_id().to_owned(),
                source: format!("HTTP {status} remote provider error").into(),
            });
        }

        Err(last_err.unwrap_or_else(|| SearchError::EmbeddingFailed {
            model: self.provider.embedder_id().to_owned(),
            source: "all retries exhausted".into(),
        }))
    }

    fn verify_response_attestation(
        &self,
        attestation: Option<&RemoteEmbeddingAttestationV1>,
    ) -> SearchResult<()> {
        let attestation = attestation.ok_or_else(|| {
            unverifiable_remote_space(
                self.provider.provider_name(),
                "remote response carried no space or producer attestation",
            )
        })?;
        let expected = RemoteEmbeddingAttestationV1::from_identity(&self.identity);
        if attestation == &expected {
            return Ok(());
        }
        Err(unverifiable_remote_space(
            self.provider.provider_name(),
            "per-response space or producer attestation drifted from the immutable epoch",
        ))
    }

    fn verify_response_vectors(
        &self,
        embeddings: &[Vec<f32>],
        expected_count: usize,
    ) -> SearchResult<()> {
        if embeddings.len() != expected_count {
            return Err(SearchError::EmbeddingFailed {
                model: self.provider.embedder_id().to_owned(),
                source: format!(
                    "remote response returned {} vectors for {expected_count} inputs",
                    embeddings.len()
                )
                .into(),
            });
        }
        if let Some(vector) = embeddings
            .iter()
            .find(|vector| vector.len() != self.dimension())
        {
            return Err(SearchError::EmbeddingFailed {
                model: self.provider.embedder_id().to_owned(),
                source: format!(
                    "remote response vector dimension {} disagrees with attested dimension {}",
                    vector.len(),
                    self.dimension()
                )
                .into(),
            });
        }
        Ok(())
    }
}

fn bounded_remote_producer_label(provider: &str) -> String {
    if provider.len() <= 128
        && !provider.is_empty()
        && provider
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        provider.to_owned()
    } else {
        "<redacted-remote-producer>".to_owned()
    }
}

fn unverifiable_remote_space(provider: &str, reason: &str) -> SearchError {
    let producer = bounded_remote_producer_label(provider);
    let reason = if reason.len() <= 512 && !reason.chars().any(char::is_control) {
        reason.to_owned()
    } else {
        "remote identity validation failed".to_owned()
    };
    SearchError::UnverifiableRemoteSpace { producer, reason }
}

/// L2-normalize a vector in place.
impl Embedder for ApiEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let results = self.request_batch(cx, &[text]).await?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| SearchError::EmbeddingFailed {
                    model: self.provider.embedder_id().to_owned(),
                    source: "empty response from API".into(),
                })
                .map(|mut v| {
                    l2_normalize_in_place(&mut v);
                    v
                })
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

            let batch_size = self.provider.max_batch_size();
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for chunk in texts.chunks(batch_size) {
                let mut batch = self.request_batch(cx, chunk).await?;
                for v in &mut batch {
                    l2_normalize_in_place(v);
                }
                all_embeddings.extend(batch);
            }

            Ok(all_embeddings)
        })
    }

    fn dimension(&self) -> usize {
        self.provider.dimension()
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        self.provider.embedder_id()
    }

    fn model_name(&self) -> &str {
        self.provider.api_model_id()
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::ApiEmbedder
    }

    fn supports_mrl(&self) -> bool {
        self.provider.supports_mrl()
    }

    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> SearchResult<Vec<f32>> {
        if target_dim == 0 || target_dim > embedding.len() {
            return Err(SearchError::EmbeddingFailed {
                model: self.provider.embedder_id().to_owned(),
                source: format!(
                    "target dimension {target_dim} must be between 1 and embedding dimension {}",
                    embedding.len()
                )
                .into(),
            });
        }
        let mut truncated = embedding[..target_dim].to_vec();
        l2_normalize_in_place(&mut truncated);
        Ok(truncated)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_provider::OpenAiProvider;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};

    fn remote_test_identity(dimension: u32) -> FrozenEmbeddingIdentityBundleV1 {
        let model = "text-embedding-3-small";
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(model, dimension);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "1".repeat(64);
        identity.space.artifacts = vec![
            EmbeddingArtifactIdentityV1 {
                role: "weights".to_owned(),
                sha256: "2".repeat(64),
                size: 1,
            },
            EmbeddingArtifactIdentityV1 {
                role: "tokenizer".to_owned(),
                sha256: "3".repeat(64),
                size: 1,
            },
        ];
        identity.space.tokenizer_fingerprint = "3".repeat(64);
        identity.space.vocabulary_fingerprint = "4".repeat(64);
        identity.space.model_config_fingerprint = "5".repeat(64);
        identity.space.output_normalization = API_OUTPUT_NORMALIZATION_V1.to_owned();
        identity.storage.format = API_STORAGE_FORMAT_V1.to_owned();
        identity.storage.quantization = QuantizationFormat::F32;
        identity.storage.endianness = API_STORAGE_ENDIANNESS_V1.to_owned();
        identity.storage.vector_normalization = API_OUTPUT_NORMALIZATION_V1.to_owned();
        identity.producer.backend = "remote-api-openai".to_owned();
        identity.producer.protocol_revision = "openai-embeddings-json-v1".to_owned();
        identity.producer.provenance_manifest_fingerprint = "6".repeat(64);
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity
            .validate()
            .expect("valid explicit remote test epoch");
        identity
            .freeze()
            .expect("freeze explicit remote test epoch")
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn l2_normalize_general() {
        let mut v = vec![3.0, 4.0];
        l2_normalize_in_place(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize_in_place(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn rate_limiter_unlimited() {
        let rl = RateLimiter::new(0);
        assert!(rl.acquire().is_none());
        assert!(rl.acquire().is_none());
    }

    #[test]
    fn rate_limiter_exhausts_tokens() {
        let rl = RateLimiter::new(2);
        // Should get 2 immediate tokens.
        assert!(rl.acquire().is_none());
        assert!(rl.acquire().is_none());
        // Third should require waiting.
        assert!(rl.acquire().is_some());
    }

    #[test]
    fn api_embedder_properties() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(256)));
        let embedder =
            ApiEmbedder::with_defaults(provider, Some(remote_test_identity(256))).unwrap();
        assert_eq!(embedder.dimension(), 256);
        assert_eq!(embedder.id(), "openai-text-embedding-3-small-256d");
        assert!(embedder.is_semantic());
        assert_eq!(embedder.category(), ModelCategory::ApiEmbedder);
        assert!(embedder.supports_mrl());
    }

    #[test]
    fn truncate_embedding_works() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
        let emb = vec![1.0, 2.0, 3.0, 4.0];
        let truncated = embedder.truncate_embedding(&emb, 2).unwrap();
        assert_eq!(truncated.len(), 2);
        let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncate_embedding_rejects_larger_dim() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
        let emb = vec![1.0, 2.0];
        assert!(embedder.truncate_embedding(&emb, 0).is_err());
        assert!(embedder.truncate_embedding(&emb, 4).is_err());
    }

    #[test]
    fn api_embedder_rejects_unverifiable_or_drifted_epoch() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        assert!(matches!(
            ApiEmbedder::with_defaults(provider, None),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let drifted = remote_test_identity(3);
        assert!(matches!(
            ApiEmbedder::with_defaults(provider, Some(drifted)),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let mut drifted = remote_test_identity(4).identity;
        drifted.space.logical_model_id = "text-embedding-3-large".to_owned();
        drifted.producer.space_fingerprint = drifted.space.fingerprint();
        assert!(matches!(
            ApiEmbedder::with_defaults(provider, Some(drifted.freeze().unwrap())),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let mut drifted = remote_test_identity(4).identity;
        drifted.producer.protocol_revision = "unregistered-wire-protocol".to_owned();
        assert!(matches!(
            ApiEmbedder::with_defaults(provider, Some(drifted.freeze().unwrap())),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let mut drifted = remote_test_identity(4).identity;
        drifted.storage.format = "fsvi-v2".to_owned();
        drifted.storage.quantization = QuantizationFormat::F16;
        drifted.storage.endianness = "little-endian".to_owned();
        assert!(matches!(
            ApiEmbedder::with_defaults(provider, Some(drifted.freeze().unwrap())),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn per_response_attestation_rejects_drift() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
        let mut attestation =
            RemoteEmbeddingAttestationV1::from_identity(embedder.identity().unwrap());
        attestation.producer_fingerprint = "f".repeat(64);
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation)),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn per_response_attestation_accepts_exact_epoch() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
        let attestation = RemoteEmbeddingAttestationV1::from_identity(embedder.identity().unwrap());
        assert!(
            embedder
                .verify_response_attestation(Some(&attestation))
                .is_ok()
        );
    }

    #[test]
    fn per_response_attestation_is_mandatory() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(None),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn unverifiable_remote_space_redacts_untrusted_labels() {
        let error =
            unverifiable_remote_space("provider\nforged-log-line", "reason\nforged-reason-line");
        assert!(matches!(
            &error,
            SearchError::UnverifiableRemoteSpace { producer, reason }
                if producer == "<redacted-remote-producer>"
                    && reason == "remote identity validation failed"
        ));

        let rendered = error.to_string();
        assert!(rendered.contains("Embedding space identity is unverifiable"));
        assert!(!rendered.contains("<redacted-remote-producer>"));
        assert!(!rendered.contains("remote identity validation failed"));
        assert!(!rendered.contains("forged-log-line"));
        assert!(!rendered.contains("forged-reason-line"));
    }
}
