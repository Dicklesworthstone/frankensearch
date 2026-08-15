//! Cloud API embedder implementing the `Embedder` trait.
//!
//! Wraps any [`super::api_provider::ApiProvider`] with HTTP transport, retry
//! logic, rate limiting, and L2 normalization. Gated behind the `api` feature.

use std::fmt;
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::bytes::Buf;
use asupersync::http::body::{Body, Frame};
use asupersync::http::h1::{ClientError, HttpClient, HttpClientConfig, Method, RedirectPolicy};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::{
    EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, FrozenEmbeddingIdentityBundleV1,
    QuantizationFormat,
};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture, l2_normalize_in_place};

use crate::api_provider::{
    ApiProvider, MIN_REMOTE_ATTESTATION_KEY_BYTES, RemoteEmbeddingAttestationV1,
    RemoteEmbeddingChallengeV1, remote_embedding_payload_sha256, remote_endpoint_fingerprint,
    remote_ordered_request_sha256,
};
use crate::cached_embedder::CachedEmbedder;

const API_OUTPUT_NORMALIZATION_V1: &str = "l2-f32-zero-on-degenerate-v1";
const API_STORAGE_FORMAT_V1: &str = "in-memory-f32-v1";
const API_STORAGE_ENDIANNESS_V1: &str = "native-f32-values";
const DEFAULT_MAX_API_RESPONSE_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_API_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
    /// Total timeout for one HTTP attempt.
    pub request_timeout: Duration,
    /// Maximum response bytes admitted before parsing.
    pub max_response_bytes: usize,
}

impl Default for ApiEmbedderConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_millis(500),
            requests_per_minute: 0,
            request_timeout: DEFAULT_API_REQUEST_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_API_RESPONSE_BYTES,
        }
    }
}

/// Explicit trust classification for remote API vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteApiTrustLevelV1 {
    /// Every returned vector passed pinned producer authentication.
    VerifiedRemote,
    /// Caller explicitly requested transient, non-persistable exploration.
    AssumedRemote,
}

/// Explicitly unverified vectors returned only by the transient API.
///
/// This wrapper deliberately does not implement [`Embedder`] and cannot be
/// wrapped in [`CachedEmbedder`]. Callers must unwrap it consciously; verified
/// indexing/search APIs continue to accept only identity-bound `Embedder`
/// implementations.
#[derive(Clone)]
pub struct AssumedRemoteEmbeddingBatchV1 {
    vectors: Vec<Vec<f32>>,
    identity_fingerprint: String,
}

impl fmt::Debug for AssumedRemoteEmbeddingBatchV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssumedRemoteEmbeddingBatchV1")
            .field("trust", &RemoteApiTrustLevelV1::AssumedRemote)
            .field("vector_count", &self.vectors.len())
            .field(
                "vector_dimension",
                &self.vectors.first().map_or(0, Vec::len),
            )
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("vectors", &"<redacted>")
            .finish()
    }
}

impl AssumedRemoteEmbeddingBatchV1 {
    /// Explicit trust label for this non-persistable result.
    #[must_use]
    pub const fn trust_level(&self) -> RemoteApiTrustLevelV1 {
        RemoteApiTrustLevelV1::AssumedRemote
    }

    /// Expected, caller-authored identity fingerprint.
    ///
    /// This is a label, not producer proof.
    #[must_use]
    pub fn assumed_identity_fingerprint(&self) -> &str {
        &self.identity_fingerprint
    }

    /// Borrow transient vectors for immediate exploratory use.
    #[must_use]
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Consume the visibly unverified wrapper.
    #[must_use]
    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.vectors
    }
}

/// Pinned HMAC-SHA256 gateway authority for verified remote embeddings.
///
/// The secret is never formatted or serialized. The generation and key ID are
/// immutable for one `ApiEmbedder`; rotation requires constructing a new
/// verified instance so midstream drift fails closed.
#[derive(Clone)]
pub struct PinnedRemoteAttesterV1 {
    key_id: String,
    authentication_key: Vec<u8>,
    generation: u64,
}

impl fmt::Debug for PinnedRemoteAttesterV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedRemoteAttesterV1")
            .field("key_id", &bounded_remote_identifier(&self.key_id))
            .field("authentication_key", &"<redacted>")
            .field("generation", &self.generation)
            .field("key_fingerprint", &self.key_fingerprint())
            .finish()
    }
}

impl PinnedRemoteAttesterV1 {
    /// Pin one gateway key ID, high-entropy shared key, and generation.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for an unsafe key ID or a key shorter
    /// than 256 bits.
    pub fn new(
        key_id: impl Into<String>,
        authentication_key: impl Into<Vec<u8>>,
        generation: u64,
    ) -> SearchResult<Self> {
        let key_id = key_id.into();
        let authentication_key = authentication_key.into();
        if bounded_remote_identifier(&key_id) != key_id.as_str() {
            return Err(unverifiable_remote_space(
                "remote-api",
                "pinned attestation key ID is malformed",
            ));
        }
        if authentication_key.len() < MIN_REMOTE_ATTESTATION_KEY_BYTES {
            return Err(unverifiable_remote_space(
                "remote-api",
                "pinned attestation key has insufficient entropy",
            ));
        }
        Ok(Self {
            key_id,
            authentication_key,
            generation,
        })
    }

    /// Bounded key identifier expected in every accepted response.
    #[must_use]
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Pinned gateway generation expected in every accepted response.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Credential-safe fingerprint for structured diagnostics.
    #[must_use]
    pub fn key_fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.remote-attestation-key-fingerprint.v1");
        hasher.update(&self.authentication_key);
        encode_lower_hex(&hasher.finalize())
    }

    fn fresh_nonce(&self, sequence: u64, entropy: &[u8; 32]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"frankensearch.remote-attestation-request-nonce.v1");
        hasher.update(entropy);
        hasher.update(sequence.to_be_bytes());
        hasher.update(self.generation.to_be_bytes());
        hasher.update(self.key_id.as_bytes());
        encode_lower_hex(&hasher.finalize())
    }
}

impl Drop for PinnedRemoteAttesterV1 {
    fn drop(&mut self) {
        self.authentication_key.fill(0);
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

#[derive(Debug)]
struct VerifiedProviderContractV1 {
    provider_name: String,
    model: String,
    embedder_id: String,
    endpoint_fingerprint: String,
    dimension: usize,
    max_batch_size: usize,
    supports_mrl: bool,
}

impl VerifiedProviderContractV1 {
    fn capture(provider: &dyn ApiProvider) -> SearchResult<Self> {
        let provider_name = provider.provider_name().to_owned();
        if bounded_remote_producer_label(&provider_name) != provider_name {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "provider name is not a bounded attestation identifier",
            ));
        }
        let model = provider.api_model_id().to_owned();
        if bounded_remote_identifier(&model) != model {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "provider model is not a bounded attestation identifier",
            ));
        }
        let embedder_id = provider.embedder_id().to_owned();
        if bounded_remote_identifier(&embedder_id) != embedder_id {
            return Err(unverifiable_remote_space(
                provider.provider_name(),
                "provider embedder ID is not a bounded diagnostic identifier",
            ));
        }
        Ok(Self {
            provider_name,
            model,
            embedder_id,
            endpoint_fingerprint: remote_endpoint_fingerprint(provider.endpoint_url()),
            dimension: provider.dimension(),
            max_batch_size: provider.max_batch_size(),
            supports_mrl: provider.supports_mrl(),
        })
    }

    fn verify_unchanged(&self, provider: &dyn ApiProvider) -> SearchResult<()> {
        let unchanged = provider.provider_name() == self.provider_name
            && provider.api_model_id() == self.model
            && provider.embedder_id() == self.embedder_id
            && remote_endpoint_fingerprint(provider.endpoint_url()) == self.endpoint_fingerprint
            && provider.dimension() == self.dimension
            && provider.max_batch_size() == self.max_batch_size
            && provider.supports_mrl() == self.supports_mrl;
        if unchanged {
            return Ok(());
        }
        Err(unverifiable_remote_space(
            &self.provider_name,
            "remote provider contract drifted after verified construction",
        ))
    }
}

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
    provider_contract: VerifiedProviderContractV1,
    attester: PinnedRemoteAttesterV1,
    request_sequence: AtomicU64,
}

impl fmt::Debug for ApiEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiEmbedder")
            .field("provider", &self.provider_contract.provider_name)
            .field("config", &self.config)
            .field("trust", &RemoteApiTrustLevelV1::VerifiedRemote)
            .field("attester", &self.attester)
            .finish_non_exhaustive()
    }
}

impl ApiEmbedder {
    /// Reject the legacy caller-asserted construction path.
    ///
    /// A frozen identity supplied by the caller is expected state, not producer
    /// proof. Use [`Self::new_verified`] with a pinned gateway authority.
    ///
    /// # Errors
    ///
    /// Always returns [`SearchError::UnverifiableRemoteSpace`]. The identity is
    /// still validated first so malformed explicit configuration retains its
    /// precise fail-closed boundary.
    pub fn new(
        provider: Box<dyn ApiProvider>,
        _config: ApiEmbedderConfig,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        let _identity = validate_remote_identity(provider.as_ref(), immutable_space_epoch)?;
        let provider_name = provider.provider_name().to_owned();
        drop(provider);
        Err(unverifiable_remote_space(
            &provider_name,
            "verified remote construction requires a pinned producer authentication key",
        ))
    }

    /// Create an API embedder whose every response must authenticate against a
    /// pinned gateway key and immutable generation.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] if the epoch is absent,
    /// malformed, non-semantic, incompatible with the provider/output
    /// contract, or lacks a valid pinned producer authority.
    pub fn new_verified(
        provider: Box<dyn ApiProvider>,
        config: ApiEmbedderConfig,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
        attester: PinnedRemoteAttesterV1,
    ) -> SearchResult<Self> {
        validate_http_config(&config)?;
        let identity = validate_remote_identity(provider.as_ref(), immutable_space_epoch)?;
        let provider_contract = VerifiedProviderContractV1::capture(provider.as_ref())?;
        let client = build_http_client(&config);
        let rate_limiter = RateLimiter::new(config.requests_per_minute);
        Ok(Self {
            provider,
            client,
            rate_limiter,
            config,
            identity,
            provider_contract,
            attester,
            request_sequence: AtomicU64::new(0),
        })
    }

    /// Create a verified embedder with default HTTP bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] for an invalid epoch or
    /// pinned authority.
    pub fn with_verified_defaults(
        provider: Box<dyn ApiProvider>,
        immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
        attester: PinnedRemoteAttesterV1,
    ) -> SearchResult<Self> {
        Self::new_verified(
            provider,
            ApiEmbedderConfig::default(),
            immutable_space_epoch,
            attester,
        )
    }

    /// Trust classification of every vector this type can return.
    #[must_use]
    pub const fn remote_trust_level(&self) -> RemoteApiTrustLevelV1 {
        RemoteApiTrustLevelV1::VerifiedRemote
    }

    /// Pinned gateway key fingerprint for credential-safe telemetry.
    #[must_use]
    pub fn attestation_key_fingerprint(&self) -> String {
        self.attester.key_fingerprint()
    }

    fn build_challenge_with_entropy(
        &self,
        texts: &[&str],
        entropy: &[u8; 32],
    ) -> SearchResult<RemoteEmbeddingChallengeV1> {
        self.provider_contract
            .verify_unchanged(self.provider.as_ref())?;
        let input_count = u32::try_from(texts.len()).map_err(|_| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote request batch exceeds the attestation schema",
            )
        })?;
        let sequence = self.request_sequence.fetch_add(1, Ordering::Relaxed);
        let challenge = RemoteEmbeddingChallengeV1 {
            schema_version: crate::api_provider::REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1,
            request_nonce: self.attester.fresh_nonce(sequence, entropy),
            ordered_request_sha256: remote_ordered_request_sha256(texts),
            input_count,
            endpoint_fingerprint: self.provider_contract.endpoint_fingerprint.clone(),
            identity_fingerprint: self.identity.fingerprint(),
            space_fingerprint: self.identity.space.fingerprint(),
            producer_fingerprint: self.identity.producer.fingerprint(),
        };
        challenge.validate()?;
        Ok(challenge)
    }

    #[cfg(test)]
    fn build_challenge(&self, texts: &[&str]) -> SearchResult<RemoteEmbeddingChallengeV1> {
        self.build_challenge_with_entropy(texts, &[0x42; 32])
    }

    /// Create with default HTTP configuration but without producer authority.
    ///
    /// This legacy entry point is retained only to fail explicit direct
    /// OpenAI/Gemini construction with a typed non-retryable error.
    ///
    /// # Errors
    ///
    /// Always returns [`SearchError::UnverifiableRemoteSpace`] after validating
    /// the caller-supplied epoch.
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

    /// Wrap this verified embedder with a cache (convenience).
    #[must_use]
    pub fn cached(self, capacity: usize) -> CachedEmbedder {
        CachedEmbedder::new(Arc::new(self), capacity)
    }

    /// Wrap with the default cache capacity (4096 entries).
    #[must_use]
    pub fn cached_default(self) -> CachedEmbedder {
        CachedEmbedder::new(Arc::new(self), 4096)
    }

    /// Make a single producer-authenticated API request for a batch, with retry.
    async fn request_batch(&self, cx: &Cx, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        self.provider_contract
            .verify_unchanged(self.provider.as_ref())?;
        let headers = self.provider.request_headers();
        let url = self.provider.request_url();
        let safe_provider = self.provider_contract.provider_name.as_str();

        let mut last_err = None;
        'retry: for attempt in 0..=self.config.max_retries {
            embed_checkpoint(cx, "api.request_batch")?;
            let mut nonce_entropy = [0_u8; 32];
            cx.random_bytes(&mut nonce_entropy);
            let challenge = self.build_challenge_with_entropy(texts, &nonce_entropy)?;
            nonce_entropy.fill(0);
            let body = self
                .provider
                .serialize_attested_request(texts, &challenge)
                .map_err(|_error| {
                    bounded_embedding_failure(
                        self.provider.as_ref(),
                        "remote request serialization failed",
                    )
                })?;

            if attempt > 0 {
                let multiplier = 1_u32.checked_shl((attempt - 1).min(31)).unwrap_or(u32::MAX);
                let backoff = self.config.retry_base_delay.saturating_mul(multiplier);
                debug!(
                    provider = safe_provider,
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    "retrying API request"
                );
                asupersync::time::sleep(cx.now(), backoff).await;
            }

            if let Some(wait) = self.rate_limiter.acquire() {
                asupersync::time::sleep(cx.now(), wait).await;
            }

            let response = self
                .client
                .request_streaming(
                    cx,
                    Method::Post,
                    &url,
                    headers
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                    body,
                )
                .await;

            let mut response = match response {
                Ok(response) => response,
                Err(error) => {
                    let error = api_client_error_to_search(&error, self.provider.as_ref());
                    if matches!(error, SearchError::Cancelled { .. }) {
                        return Err(error);
                    }
                    last_err = Some(error);
                    continue;
                }
            };

            let status = response.head.status;
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
                            if response_body.len().saturating_add(bytes.len())
                                > self.config.max_response_bytes
                            {
                                return Err(bounded_embedding_failure(
                                    self.provider.as_ref(),
                                    "remote response exceeded the configured byte budget",
                                ));
                            }
                            response_body.extend_from_slice(bytes);
                            chunk.advance(bytes.len());
                        }
                    }
                    Ok(Frame::Trailers(_)) => {}
                    Err(_error) => {
                        last_err = Some(bounded_embedding_failure(
                            self.provider.as_ref(),
                            "remote HTTP response body failed",
                        ));
                        continue 'retry;
                    }
                }
            }

            if (200..300).contains(&status) {
                let embeddings =
                    self.provider
                        .deserialize_response(&response_body)
                        .map_err(|_error| {
                            bounded_embedding_failure(
                                self.provider.as_ref(),
                                "remote response decoding failed",
                            )
                        })?;
                let attestation =
                    self.provider
                        .response_attestation(&response_body)
                        .map_err(|_error| {
                            unverifiable_remote_space(
                                &self.provider_contract.provider_name,
                                "per-response producer attestation is malformed",
                            )
                        })?;
                self.verify_response_vectors(&embeddings, texts.len())?;
                self.verify_response_attestation(attestation.as_ref(), &challenge, &embeddings)?;
                return Ok(embeddings);
            }

            if status == 429 || status >= 500 {
                warn!(
                    provider = safe_provider,
                    status, attempt, "transient API error"
                );
                last_err = Some(bounded_embedding_failure(
                    self.provider.as_ref(),
                    &format!("HTTP {status} remote provider error"),
                ));
                continue;
            }

            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                &format!("HTTP {status} remote provider error"),
            ));
        }

        Err(last_err.unwrap_or_else(|| {
            bounded_embedding_failure(self.provider.as_ref(), "all retries exhausted")
        }))
    }

    fn verify_response_attestation(
        &self,
        attestation: Option<&RemoteEmbeddingAttestationV1>,
        challenge: &RemoteEmbeddingChallengeV1,
        embeddings: &[Vec<f32>],
    ) -> SearchResult<()> {
        let attestation = attestation.ok_or_else(|| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response carried no producer-authenticated attestation",
            )
        })?;
        if !attestation.authenticate_hmac_sha256(&self.attester.authentication_key) {
            return Err(unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response producer authentication failed",
            ));
        }

        let vector_count = u32::try_from(embeddings.len()).map_err(|_| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response shape exceeds the attestation schema",
            )
        })?;
        let vector_dimension = embeddings.first().map_or(0, Vec::len);
        let vector_dimension = u32::try_from(vector_dimension).map_err(|_| {
            unverifiable_remote_space(
                &self.provider_contract.provider_name,
                "remote response dimension exceeds the attestation schema",
            )
        })?;

        let exact_match = attestation.key_id == self.attester.key_id
            && attestation.generation == self.attester.generation
            && attestation.request_nonce == challenge.request_nonce
            && attestation.ordered_request_sha256 == challenge.ordered_request_sha256
            && attestation.input_count == challenge.input_count
            && attestation.endpoint_fingerprint == challenge.endpoint_fingerprint
            && attestation.identity_fingerprint == challenge.identity_fingerprint
            && attestation.space_fingerprint == challenge.space_fingerprint
            && attestation.producer_fingerprint == challenge.producer_fingerprint
            && attestation.provider == self.provider_contract.provider_name
            && attestation.model == self.provider_contract.model
            && attestation.producer_backend == self.identity.producer.backend
            && attestation.protocol_revision == self.identity.producer.protocol_revision
            && attestation.vector_count == vector_count
            && attestation.vector_dimension == vector_dimension
            && attestation.response_payload_sha256 == remote_embedding_payload_sha256(embeddings);
        if exact_match {
            return Ok(());
        }
        Err(unverifiable_remote_space(
            &self.provider_contract.provider_name,
            "authenticated remote response drifted from its request or immutable epoch",
        ))
    }

    fn verify_response_vectors(
        &self,
        embeddings: &[Vec<f32>],
        expected_count: usize,
    ) -> SearchResult<()> {
        if embeddings.len() != expected_count {
            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                "remote response vector count disagrees with the request",
            ));
        }
        if embeddings
            .iter()
            .any(|vector| vector.len() != self.dimension())
        {
            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                "remote response vector dimension disagrees with the attested dimension",
            ));
        }
        Ok(())
    }
}

/// Explicit transient-only client for an unattested remote provider.
///
/// This type intentionally does not implement [`Embedder`], does not expose a
/// cache constructor, and returns [`AssumedRemoteEmbeddingBatchV1`] rather than
/// identity-bound vectors. It is suitable only for visibly labeled
/// exploration; persistent indexing and verified semantic readiness require
/// [`ApiEmbedder`].
pub struct AssumedRemoteApi {
    provider: Box<dyn ApiProvider>,
    client: HttpClient,
    rate_limiter: RateLimiter,
    config: ApiEmbedderConfig,
    identity: EmbeddingIdentityBundleV1,
}

impl fmt::Debug for AssumedRemoteApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssumedRemoteApi")
            .field(
                "provider",
                &bounded_remote_producer_label(self.provider.provider_name()),
            )
            .field("config", &self.config)
            .field("trust", &RemoteApiTrustLevelV1::AssumedRemote)
            .finish_non_exhaustive()
    }
}

impl AssumedRemoteApi {
    /// Construct the explicit transient-only path.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] if the caller-authored
    /// expected identity is malformed or disagrees with the provider contract.
    pub fn new(
        provider: Box<dyn ApiProvider>,
        config: ApiEmbedderConfig,
        assumed_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        validate_http_config(&config)?;
        let identity = validate_remote_identity(provider.as_ref(), assumed_space_epoch)?;
        let client = build_http_client(&config);
        let rate_limiter = RateLimiter::new(config.requests_per_minute);
        Ok(Self {
            provider,
            client,
            rate_limiter,
            config,
            identity,
        })
    }

    /// Construct the transient-only path with default HTTP bounds.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] for an invalid expected
    /// identity.
    pub fn with_defaults(
        provider: Box<dyn ApiProvider>,
        assumed_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
    ) -> SearchResult<Self> {
        Self::new(provider, ApiEmbedderConfig::default(), assumed_space_epoch)
    }

    /// This path is always visibly unverified.
    #[must_use]
    pub const fn remote_trust_level(&self) -> RemoteApiTrustLevelV1 {
        RemoteApiTrustLevelV1::AssumedRemote
    }

    /// Embed one text for immediate exploratory use.
    pub fn embed_transient<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, AssumedRemoteEmbeddingBatchV1> {
        Box::pin(async move {
            let texts = [text];
            self.embed_batch_transient(cx, &texts).await
        })
    }

    /// Embed a batch for immediate exploratory use.
    ///
    /// The result remains wrapped in an explicit `AssumedRemote` type and is
    /// not accepted by verified cache/index constructors.
    pub fn embed_batch_transient<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, AssumedRemoteEmbeddingBatchV1> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(AssumedRemoteEmbeddingBatchV1 {
                    vectors: Vec::new(),
                    identity_fingerprint: self.identity.fingerprint(),
                });
            }

            let mut vectors = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(self.provider.max_batch_size()) {
                let mut batch = self.request_batch_unattested(cx, chunk).await?;
                for vector in &mut batch {
                    l2_normalize_in_place(vector);
                }
                vectors.extend(batch);
            }
            Ok(AssumedRemoteEmbeddingBatchV1 {
                vectors,
                identity_fingerprint: self.identity.fingerprint(),
            })
        })
    }

    async fn request_batch_unattested(
        &self,
        cx: &Cx,
        texts: &[&str],
    ) -> SearchResult<Vec<Vec<f32>>> {
        let headers = self.provider.request_headers();
        let url = self.provider.request_url();
        let safe_provider = bounded_remote_producer_label(self.provider.provider_name());
        let body = self.provider.serialize_request(texts).map_err(|_error| {
            bounded_embedding_failure(
                self.provider.as_ref(),
                "remote request serialization failed",
            )
        })?;

        let mut last_err = None;
        'retry: for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let multiplier = 1_u32.checked_shl((attempt - 1).min(31)).unwrap_or(u32::MAX);
                let backoff = self.config.retry_base_delay.saturating_mul(multiplier);
                debug!(
                    provider = safe_provider.as_str(),
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    trust = "assumed-remote",
                    "retrying transient API request"
                );
                asupersync::time::sleep(cx.now(), backoff).await;
            }

            if let Some(wait) = self.rate_limiter.acquire() {
                asupersync::time::sleep(cx.now(), wait).await;
            }

            let response = self
                .client
                .request_streaming(
                    cx,
                    Method::Post,
                    &url,
                    headers
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                    body.clone(),
                )
                .await;
            let mut response = match response {
                Ok(response) => response,
                Err(error) => {
                    let error = api_client_error_to_search(&error, self.provider.as_ref());
                    if matches!(error, SearchError::Cancelled { .. }) {
                        return Err(error);
                    }
                    last_err = Some(error);
                    continue;
                }
            };

            let status = response.head.status;
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
                            if response_body.len().saturating_add(bytes.len())
                                > self.config.max_response_bytes
                            {
                                return Err(bounded_embedding_failure(
                                    self.provider.as_ref(),
                                    "remote response exceeded the configured byte budget",
                                ));
                            }
                            response_body.extend_from_slice(bytes);
                            chunk.advance(bytes.len());
                        }
                    }
                    Ok(Frame::Trailers(_)) => {}
                    Err(_error) => {
                        last_err = Some(bounded_embedding_failure(
                            self.provider.as_ref(),
                            "remote HTTP response body failed",
                        ));
                        continue 'retry;
                    }
                }
            }

            if (200..300).contains(&status) {
                let embeddings =
                    self.provider
                        .deserialize_response(&response_body)
                        .map_err(|_error| {
                            bounded_embedding_failure(
                                self.provider.as_ref(),
                                "remote response decoding failed",
                            )
                        })?;
                verify_response_vectors(
                    self.provider.as_ref(),
                    &embeddings,
                    texts.len(),
                    self.provider.dimension(),
                )?;
                return Ok(embeddings);
            }

            if status == 429 || status >= 500 {
                warn!(
                    provider = safe_provider.as_str(),
                    status,
                    attempt,
                    trust = "assumed-remote",
                    "transient API error"
                );
                last_err = Some(bounded_embedding_failure(
                    self.provider.as_ref(),
                    &format!("HTTP {status} remote provider error"),
                ));
                continue;
            }
            return Err(bounded_embedding_failure(
                self.provider.as_ref(),
                &format!("HTTP {status} remote provider error"),
            ));
        }

        Err(last_err.unwrap_or_else(|| {
            bounded_embedding_failure(self.provider.as_ref(), "all retries exhausted")
        }))
    }
}

fn validate_remote_identity(
    provider: &dyn ApiProvider,
    immutable_space_epoch: Option<FrozenEmbeddingIdentityBundleV1>,
) -> SearchResult<EmbeddingIdentityBundleV1> {
    if provider.max_batch_size() == 0 {
        return Err(unverifiable_remote_space(
            provider.provider_name(),
            "provider maximum batch size must be greater than zero",
        ));
    }
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
    Ok(identity)
}

fn validate_http_config(config: &ApiEmbedderConfig) -> SearchResult<()> {
    if config.request_timeout.is_zero() {
        return Err(SearchError::InvalidConfig {
            field: "api.request_timeout".to_owned(),
            value: "0ms".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }
    if config.max_response_bytes == 0 {
        return Err(SearchError::InvalidConfig {
            field: "api.max_response_bytes".to_owned(),
            value: "0".to_owned(),
            reason: "must be greater than zero".to_owned(),
        });
    }
    Ok(())
}

fn verify_response_vectors(
    provider: &dyn ApiProvider,
    embeddings: &[Vec<f32>],
    expected_count: usize,
    expected_dimension: usize,
) -> SearchResult<()> {
    if embeddings.len() != expected_count {
        return Err(bounded_embedding_failure(
            provider,
            "remote response vector count disagrees with the request",
        ));
    }
    if embeddings
        .iter()
        .any(|vector| vector.len() != expected_dimension)
    {
        return Err(bounded_embedding_failure(
            provider,
            "remote response vector dimension disagrees with the expected dimension",
        ));
    }
    Ok(())
}

fn build_http_client(config: &ApiEmbedderConfig) -> HttpClient {
    let mut client_config = HttpClientConfig::default();
    client_config.redirect_policy = RedirectPolicy::Limited(5);
    client_config.user_agent = Some(format!(
        "frankensearch/{} (api-embedder)",
        env!("CARGO_PKG_VERSION")
    ));
    client_config.max_body_size = Some(config.max_response_bytes);
    client_config.request_timeout = Some(config.request_timeout);
    HttpClient::with_config(client_config)
}

fn bounded_embedding_failure(provider: &dyn ApiProvider, reason: &str) -> SearchError {
    let reason = if reason.len() <= 256 && !reason.chars().any(char::is_control) {
        reason
    } else {
        "remote embedding operation failed"
    };
    SearchError::EmbeddingFailed {
        model: bounded_remote_identifier(provider.embedder_id()),
        source: reason.to_owned().into(),
    }
}

fn api_client_error_to_search(error: &ClientError, provider: &dyn ApiProvider) -> SearchError {
    if matches!(error, ClientError::Cancelled) {
        return SearchError::Cancelled {
            phase: "remote-api-embedding-request".to_owned(),
            reason: "HTTP client observed structured cancellation".to_owned(),
        };
    }
    let reason = match error {
        ClientError::InvalidUrl(_) => "remote transport invalid URL",
        ClientError::DnsError(_) => "remote transport DNS failure",
        ClientError::ConnectError(_) | ClientError::Io(_) => "remote transport connection failure",
        ClientError::TlsError(_) => "remote transport TLS failure",
        ClientError::HttpError(_) => "remote HTTP protocol failure",
        ClientError::TooManyRedirects { .. } => "remote transport redirect limit exceeded",
        ClientError::DeadlineExceeded => "remote request deadline exceeded",
        ClientError::ConnectTunnelRefused { .. } => "remote proxy tunnel refused",
        ClientError::InvalidConnectInput(_) => "remote proxy configuration invalid",
        ClientError::ProxyError(_) => "remote proxy failure",
        ClientError::PoolExhausted { .. } => "remote connection pool exhausted",
        ClientError::Cancelled => unreachable!("handled above"),
    };
    bounded_embedding_failure(provider, reason)
}

fn bounded_remote_identifier(value: &str) -> String {
    if value.len() <= 128
        && !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/'))
    {
        value.to_owned()
    } else {
        "<redacted-remote-identifier>".to_owned()
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

fn encode_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

/// L2-normalize a vector in place.
fn embed_checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: cx
            .cancel_reason()
            .map_or_else(|| error.to_string(), |reason| reason.to_string()),
    })
}

impl Embedder for ApiEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            embed_checkpoint(cx, "api.embed")?;
            let results = self.request_batch(cx, &[text]).await?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| SearchError::EmbeddingFailed {
                    model: self.provider_contract.embedder_id.clone(),
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
            embed_checkpoint(cx, "api.embed_batch")?;
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let batch_size = self.provider_contract.max_batch_size;
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for chunk in texts.chunks(batch_size) {
                embed_checkpoint(cx, "api.embed_batch")?;
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
        self.provider_contract.dimension
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        &self.provider_contract.embedder_id
    }

    fn model_name(&self) -> &str {
        &self.provider_contract.model
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::ApiEmbedder
    }

    fn supports_mrl(&self) -> bool {
        self.provider_contract.supports_mrl
    }

    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> SearchResult<Vec<f32>> {
        if target_dim == 0 || target_dim > embedding.len() {
            return Err(SearchError::EmbeddingFailed {
                model: self.provider_contract.embedder_id.clone(),
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
    use crate::api_provider::{OpenAiProvider, remote_ordered_request_sha256};
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use serde::Deserialize;
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::sync::atomic::AtomicU8;
    use std::thread;

    const TEST_ATTESTATION_KEY: &[u8; 32] = b"0123456789abcdef0123456789abcdef";
    const TEST_KEY_ID: &str = "gateway-key-2026-07";
    const TEST_GENERATION: u64 = 41;

    #[test]
    fn embed_checkpoint_observes_cancel_before_http() {
        run_test_with_cx(|cx| async move {
            cx.cancel_fast(asupersync::CancelKind::User);
            let err = super::embed_checkpoint(&cx, "api.embed").unwrap_err();
            match err {
                SearchError::Cancelled { phase, .. } => {
                    assert_eq!(phase, "api.embed");
                }
                other => panic!("expected Cancelled, got {other:?}"),
            }
        });
    }

    #[derive(Debug, Clone)]
    struct TestGatewayProvider {
        endpoint: String,
        dimension: usize,
    }

    #[derive(Debug)]
    struct MidstreamDriftProvider {
        drift_mode: Arc<AtomicU8>,
    }

    impl ApiProvider for MidstreamDriftProvider {
        fn provider_name(&self) -> &'static str {
            "openai"
        }

        fn api_model_id(&self) -> &'static str {
            if self.drift_mode.load(Ordering::Relaxed) == 1 {
                "same-dimension-wrong-model"
            } else {
                "text-embedding-3-small"
            }
        }

        fn embedder_id(&self) -> &'static str {
            "midstream-drift-test"
        }

        fn identity_backend(&self) -> &'static str {
            "remote-api-openai"
        }

        fn identity_protocol_revision(&self) -> &'static str {
            "openai-embeddings-json-v1"
        }

        fn dimension(&self) -> usize {
            4
        }

        fn max_batch_size(&self) -> usize {
            16
        }

        fn supports_mrl(&self) -> bool {
            false
        }

        fn endpoint_url(&self) -> &'static str {
            if self.drift_mode.load(Ordering::Relaxed) == 2 {
                "http://127.0.0.1:2/changed"
            } else {
                "http://127.0.0.1:1/expected"
            }
        }

        fn request_headers(&self) -> Vec<(String, String)> {
            Vec::new()
        }

        fn serialize_request(&self, _texts: &[&str]) -> SearchResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_response(&self, _body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
            Ok(Vec::new())
        }
    }

    impl ApiProvider for TestGatewayProvider {
        fn provider_name(&self) -> &'static str {
            "openai"
        }

        fn api_model_id(&self) -> &'static str {
            "text-embedding-3-small"
        }

        fn embedder_id(&self) -> &'static str {
            "authenticated-test-gateway"
        }

        fn identity_backend(&self) -> &'static str {
            "remote-api-openai"
        }

        fn identity_protocol_revision(&self) -> &'static str {
            "openai-embeddings-json-v1"
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn max_batch_size(&self) -> usize {
            16
        }

        fn supports_mrl(&self) -> bool {
            false
        }

        fn endpoint_url(&self) -> &str {
            &self.endpoint
        }

        fn request_headers(&self) -> Vec<(String, String)> {
            vec![("content-type".to_owned(), "application/json".to_owned())]
        }

        fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>> {
            serde_json::to_vec(&serde_json::json!({ "inputs": texts })).map_err(|error| {
                SearchError::EmbeddingFailed {
                    model: self.embedder_id().to_owned(),
                    source: error.into(),
                }
            })
        }

        fn serialize_attested_request(
            &self,
            texts: &[&str],
            challenge: &RemoteEmbeddingChallengeV1,
        ) -> SearchResult<Vec<u8>> {
            serde_json::to_vec(&serde_json::json!({
                "inputs": texts,
                "challenge": challenge,
            }))
            .map_err(|error| SearchError::EmbeddingFailed {
                model: self.embedder_id().to_owned(),
                source: error.into(),
            })
        }

        fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
            serde_json::from_slice::<TestGatewayResponse>(body)
                .map(|response| response.vectors)
                .map_err(|error| SearchError::EmbeddingFailed {
                    model: self.embedder_id().to_owned(),
                    source: error.into(),
                })
        }

        fn response_attestation(
            &self,
            body: &[u8],
        ) -> SearchResult<Option<RemoteEmbeddingAttestationV1>> {
            serde_json::from_slice::<TestGatewayResponse>(body)
                .map(|response| response.attestation)
                .map_err(|error| SearchError::EmbeddingFailed {
                    model: self.embedder_id().to_owned(),
                    source: error.into(),
                })
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TestGatewayRequest {
        inputs: Vec<String>,
        challenge: RemoteEmbeddingChallengeV1,
    }

    #[derive(Debug, Deserialize, serde::Serialize)]
    #[serde(deny_unknown_fields)]
    struct TestGatewayResponse {
        vectors: Vec<Vec<f32>>,
        attestation: Option<RemoteEmbeddingAttestationV1>,
    }

    fn spawn_authenticated_gateway(
        identity: EmbeddingIdentityBundleV1,
        vectors: Vec<Vec<f32>>,
        transient_failures: usize,
        include_attestation: bool,
    ) -> (String, thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut nonces = Vec::new();
            for attempt in 0..=transient_failures {
                let (mut stream, _) = listener.accept().unwrap();
                let body = read_http_request_body(&mut stream);
                let request: TestGatewayRequest = serde_json::from_slice(&body).unwrap();
                let input_refs: Vec<&str> = request.inputs.iter().map(String::as_str).collect();
                assert_eq!(
                    request.challenge.ordered_request_sha256,
                    remote_ordered_request_sha256(&input_refs)
                );
                assert_eq!(
                    request.challenge.input_count,
                    u32::try_from(input_refs.len()).unwrap()
                );
                nonces.push(request.challenge.request_nonce.clone());

                if attempt < transient_failures {
                    write_http_response(&mut stream, 503, b"{\"error\":\"transient\"}");
                    continue;
                }

                let attestation = include_attestation
                    .then(|| {
                        RemoteEmbeddingAttestationV1::unsigned(
                            &request.challenge,
                            &identity,
                            "openai",
                            "text-embedding-3-small",
                            TEST_GENERATION,
                            &vectors,
                            TEST_KEY_ID,
                        )
                        .unwrap()
                    })
                    .map(|mut attestation| {
                        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
                        attestation
                    });
                let response = serde_json::to_vec(&TestGatewayResponse {
                    vectors: vectors.clone(),
                    attestation,
                })
                .unwrap();
                write_http_response(&mut stream, 200, &response);
            }
            nonces
        });
        (format!("http://{address}/embeddings"), handle)
    }

    fn read_http_request_body(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        let header_end = loop {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "request ended before HTTP headers");
            request.extend_from_slice(&buffer[..read]);
            if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
            assert!(request.len() <= 64 * 1024, "request headers too large");
        };
        let headers = std::str::from_utf8(&request[..header_end]).unwrap();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap();
        while request.len() - header_end < content_length {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "request ended before declared body");
            request.extend_from_slice(&buffer[..read]);
        }
        request[header_end..header_end + content_length].to_vec()
    }

    fn write_http_response(stream: &mut TcpStream, status: u16, body: &[u8]) {
        let reason = if status == 200 {
            "OK"
        } else {
            "Service Unavailable"
        };
        write!(
            stream,
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .unwrap();
        stream.write_all(body).unwrap();
        stream.flush().unwrap();
        let _ = stream.shutdown(Shutdown::Both);
    }

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

    fn test_attester() -> PinnedRemoteAttesterV1 {
        PinnedRemoteAttesterV1::new(TEST_KEY_ID, TEST_ATTESTATION_KEY.to_vec(), TEST_GENERATION)
            .expect("valid pinned test attester")
    }

    fn verified_test_embedder(dimension: u32) -> ApiEmbedder {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small(
            "redacted-test-key",
            Some(usize::try_from(dimension).unwrap()),
        ));
        ApiEmbedder::with_verified_defaults(
            provider,
            Some(remote_test_identity(dimension)),
            test_attester(),
        )
        .expect("valid verified test embedder")
    }

    fn signed_test_attestation(
        embedder: &ApiEmbedder,
        challenge: &RemoteEmbeddingChallengeV1,
        vectors: &[Vec<f32>],
    ) -> RemoteEmbeddingAttestationV1 {
        let mut attestation = RemoteEmbeddingAttestationV1::unsigned(
            challenge,
            &embedder.identity,
            embedder.provider.provider_name(),
            embedder.provider.api_model_id(),
            TEST_GENERATION,
            vectors,
            TEST_KEY_ID,
        )
        .expect("valid unsigned test attestation");
        attestation
            .sign_hmac_sha256(TEST_ATTESTATION_KEY)
            .expect("sign test attestation");
        attestation
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
        let embedder = verified_test_embedder(256);
        assert_eq!(embedder.dimension(), 256);
        assert_eq!(embedder.id(), "openai-text-embedding-3-small-256d");
        assert!(embedder.is_semantic());
        assert_eq!(embedder.category(), ModelCategory::ApiEmbedder);
        assert!(embedder.supports_mrl());
        assert_eq!(
            embedder.remote_trust_level(),
            RemoteApiTrustLevelV1::VerifiedRemote
        );
        assert_eq!(embedder.attester.key_id(), TEST_KEY_ID);
        assert_eq!(embedder.attester.generation(), TEST_GENERATION);
    }

    #[test]
    fn truncate_embedding_works() {
        let embedder = verified_test_embedder(4);
        let emb = vec![1.0, 2.0, 3.0, 4.0];
        let truncated = embedder.truncate_embedding(&emb, 2).unwrap();
        assert_eq!(truncated.len(), 2);
        let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncate_embedding_rejects_larger_dim() {
        let embedder = verified_test_embedder(4);
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
            ApiEmbedder::with_verified_defaults(provider, Some(drifted), test_attester()),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let mut drifted = remote_test_identity(4).identity;
        drifted.space.logical_model_id = "text-embedding-3-large".to_owned();
        drifted.producer.space_fingerprint = drifted.space.fingerprint();
        assert!(matches!(
            ApiEmbedder::with_verified_defaults(
                provider,
                Some(drifted.freeze().unwrap()),
                test_attester()
            ),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let mut drifted = remote_test_identity(4).identity;
        drifted.producer.protocol_revision = "unregistered-wire-protocol".to_owned();
        assert!(matches!(
            ApiEmbedder::with_verified_defaults(
                provider,
                Some(drifted.freeze().unwrap()),
                test_attester()
            ),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let mut drifted = remote_test_identity(4).identity;
        drifted.storage.format = "fsvi-v2".to_owned();
        drifted.storage.quantization = QuantizationFormat::F16;
        drifted.storage.endianness = "little-endian".to_owned();
        assert!(matches!(
            ApiEmbedder::with_verified_defaults(
                provider,
                Some(drifted.freeze().unwrap()),
                test_attester()
            ),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn direct_provider_identity_without_pinned_authority_fails_closed() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let error = ApiEmbedder::with_defaults(provider, Some(remote_test_identity(4)))
            .expect_err("caller-authored identity must not authenticate a provider");
        assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
    }

    #[test]
    fn per_response_attestation_accepts_exact_authenticated_envelope() {
        let embedder = verified_test_embedder(4);
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let challenge = embedder.build_challenge(&["alpha"]).unwrap();
        let attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        assert!(
            embedder
                .verify_response_attestation(Some(&attestation), &challenge, &vectors)
                .is_ok()
        );
    }

    #[test]
    fn per_response_attestation_rejects_authenticated_epoch_drift() {
        let embedder = verified_test_embedder(4);
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let challenge = embedder.build_challenge(&["alpha"]).unwrap();
        let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        attestation.producer_fingerprint = "f".repeat(64);
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        attestation.endpoint_fingerprint = "e".repeat(64);
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        attestation.model = "same-dimension-wrong-model".to_owned();
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        attestation.generation += 1;
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
        attestation.key_id = "rotated-key".to_owned();
        attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn authenticated_envelope_mutation_matrix_fails_closed() {
        let embedder = verified_test_embedder(4);
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let challenge = embedder.build_challenge(&["alpha"]).unwrap();
        let mutations: [fn(&mut RemoteEmbeddingAttestationV1); 7] = [
            |attestation| attestation.identity_fingerprint = "1".repeat(64),
            |attestation| attestation.space_fingerprint = "2".repeat(64),
            |attestation| attestation.provider = "other-provider".to_owned(),
            |attestation| attestation.producer_backend = "other-backend".to_owned(),
            |attestation| attestation.protocol_revision = "other-protocol-v1".to_owned(),
            |attestation| attestation.ordered_request_sha256 = "3".repeat(64),
            |attestation| attestation.response_payload_sha256 = "4".repeat(64),
        ];

        for mutate in mutations {
            let mut attestation = signed_test_attestation(&embedder, &challenge, &vectors);
            mutate(&mut attestation);
            attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();
            assert!(matches!(
                embedder.verify_response_attestation(Some(&attestation), &challenge, &vectors),
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        }
    }

    #[test]
    fn same_model_and_dimension_reject_every_canonical_space_component_drift() {
        let embedder = verified_test_embedder(4);
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let expected_challenge = embedder.build_challenge(&["alpha"]).unwrap();
        let mutations: [fn(&mut EmbeddingIdentityBundleV1); 5] = [
            |identity| {
                identity.space.artifact_manifest_fingerprint = "7".repeat(64);
                identity.space.artifacts[0].sha256 = "8".repeat(64);
            },
            |identity| {
                identity.space.tokenizer_fingerprint = "7".repeat(64);
                identity.space.vocabulary_fingerprint = "8".repeat(64);
                identity.space.artifacts[1].sha256 = "9".repeat(64);
            },
            |identity| {
                identity.space.model_preprocessing = "changed-preprocessing-v2".to_owned();
            },
            |identity| {
                identity.space.pooling = "changed-pooling-v2".to_owned();
            },
            |identity| {
                identity.space.output_normalization = "changed-normalization-v2".to_owned();
                identity.storage.vector_normalization = "changed-normalization-v2".to_owned();
            },
        ];

        for mutate in mutations {
            let mut drifted_identity = embedder.identity.clone();
            mutate(&mut drifted_identity);
            drifted_identity.producer.space_fingerprint = drifted_identity.space.fingerprint();
            drifted_identity.validate().unwrap();
            assert_eq!(
                drifted_identity.space.logical_model_id,
                embedder.identity.space.logical_model_id
            );
            assert_eq!(
                drifted_identity.space.dimension,
                embedder.identity.space.dimension
            );
            assert_ne!(
                drifted_identity.space.fingerprint(),
                embedder.identity.space.fingerprint()
            );

            let mut drifted_challenge = expected_challenge.clone();
            drifted_challenge.identity_fingerprint = drifted_identity.fingerprint();
            drifted_challenge.space_fingerprint = drifted_identity.space.fingerprint();
            drifted_challenge.producer_fingerprint = drifted_identity.producer.fingerprint();
            let mut attestation = RemoteEmbeddingAttestationV1::unsigned(
                &drifted_challenge,
                &drifted_identity,
                embedder.provider_contract.provider_name.as_str(),
                embedder.provider_contract.model.as_str(),
                TEST_GENERATION,
                &vectors,
                TEST_KEY_ID,
            )
            .unwrap();
            attestation.sign_hmac_sha256(TEST_ATTESTATION_KEY).unwrap();

            assert!(matches!(
                embedder.verify_response_attestation(
                    Some(&attestation),
                    &expected_challenge,
                    &vectors
                ),
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        }
    }

    #[test]
    fn verified_provider_contract_rejects_midstream_model_and_endpoint_drift() {
        let drift_mode = Arc::new(AtomicU8::new(0));
        let provider = Box::new(MidstreamDriftProvider {
            drift_mode: Arc::clone(&drift_mode),
        });
        let embedder = ApiEmbedder::with_verified_defaults(
            provider,
            Some(remote_test_identity(4)),
            test_attester(),
        )
        .unwrap();

        for mode in [1, 2] {
            drift_mode.store(mode, Ordering::Relaxed);
            assert!(matches!(
                embedder.build_challenge(&["alpha"]),
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        }
    }

    #[test]
    fn per_response_attestation_rejects_missing_wrong_key_and_unknown_schema() {
        let embedder = verified_test_embedder(4);
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let challenge = embedder.build_challenge(&["alpha"]).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(None, &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let mut wrong_key = RemoteEmbeddingAttestationV1::unsigned(
            &challenge,
            &embedder.identity,
            embedder.provider.provider_name(),
            embedder.provider.api_model_id(),
            TEST_GENERATION,
            &vectors,
            TEST_KEY_ID,
        )
        .unwrap();
        wrong_key.sign_hmac_sha256(&[0x55; 32]).unwrap();
        assert!(matches!(
            embedder.verify_response_attestation(Some(&wrong_key), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let mut unknown_schema = signed_test_attestation(&embedder, &challenge, &vectors);
        unknown_schema.schema_version = 2;
        assert!(matches!(
            embedder.verify_response_attestation(Some(&unknown_schema), &challenge, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn attestation_rejects_replay_request_reorder_and_payload_tamper() {
        let embedder = verified_test_embedder(4);
        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
        let original = embedder.build_challenge(&["alpha", "beta"]).unwrap();
        let attestation = signed_test_attestation(&embedder, &original, &vectors);

        let replay_target = embedder.build_challenge(&["alpha", "beta"]).unwrap();
        assert_ne!(original.request_nonce, replay_target.request_nonce);
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &replay_target, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let reordered_request = embedder.build_challenge(&["beta", "alpha"]).unwrap();
        assert_ne!(
            original.ordered_request_sha256,
            reordered_request.ordered_request_sha256
        );
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &reordered_request, &vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let reordered_vectors = vec![vectors[1].clone(), vectors[0].clone()];
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &original, &reordered_vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));

        let truncated_vectors = vec![vectors[0].clone()];
        assert!(matches!(
            embedder.verify_response_attestation(Some(&attestation), &original, &truncated_vectors),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn assumed_remote_is_explicit_transient_and_debug_redacts_vectors() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small(
            "assumed-secret",
            Some(4),
        ));
        let assumed =
            AssumedRemoteApi::with_defaults(provider, Some(remote_test_identity(4))).unwrap();
        assert_eq!(
            assumed.remote_trust_level(),
            RemoteApiTrustLevelV1::AssumedRemote
        );
        let batch = AssumedRemoteEmbeddingBatchV1 {
            vectors: vec![vec![123_456.0, 654_321.0, 0.0, 0.0]],
            identity_fingerprint: assumed.identity.fingerprint(),
        };
        let rendered = format!("{batch:?}");
        assert!(rendered.contains("AssumedRemote"));
        assert!(!rendered.contains("123456"));
        assert!(!rendered.contains("654321"));

        let attester = test_attester();
        let rendered = format!("{attester:?}");
        assert!(!rendered.contains("0123456789abcdef"));
    }

    #[test]
    fn authenticated_local_gateway_e2e_admits_exact_vectors() {
        let frozen_identity = remote_test_identity(4);
        let raw_vectors = vec![vec![3.0, 4.0, 0.0, 0.0], vec![0.0, 0.0, 5.0, 12.0]];
        let (endpoint, handle) =
            spawn_authenticated_gateway(frozen_identity.identity.clone(), raw_vectors, 0, true);
        let provider = Box::new(TestGatewayProvider {
            endpoint,
            dimension: 4,
        });
        let embedder =
            ApiEmbedder::with_verified_defaults(provider, Some(frozen_identity), test_attester())
                .unwrap();

        let captured = Arc::new(Mutex::new(None));
        let captured_for_task = Arc::clone(&captured);
        run_test_with_cx(|cx| async move {
            let vectors = embedder.embed_batch(&cx, &["alpha", "beta"]).await.unwrap();
            *captured_for_task.lock().unwrap() = Some(vectors);
        });
        let vectors = captured.lock().unwrap().take().unwrap();
        let nonces = handle.join().unwrap();
        assert_eq!(nonces.len(), 1);
        assert_eq!(vectors.len(), 2);
        for (actual, expected) in vectors[0].iter().zip([0.6, 0.8, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in vectors[1].iter().zip([0.0, 0.0, 5.0 / 13.0, 12.0 / 13.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn successful_gateway_response_without_attestation_fails_strict_mode() {
        let frozen_identity = remote_test_identity(4);
        let raw_vectors = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let (endpoint, handle) =
            spawn_authenticated_gateway(frozen_identity.identity.clone(), raw_vectors, 0, false);
        let provider = Box::new(TestGatewayProvider {
            endpoint,
            dimension: 4,
        });
        let embedder =
            ApiEmbedder::with_verified_defaults(provider, Some(frozen_identity), test_attester())
                .unwrap();

        let rejected = Arc::new(Mutex::new(false));
        let rejected_for_task = Arc::clone(&rejected);
        run_test_with_cx(|cx| async move {
            let error = embedder
                .embed_batch(&cx, &["alpha"])
                .await
                .expect_err("strict mode must reject a successful unattested response");
            *rejected_for_task.lock().unwrap() =
                matches!(error, SearchError::UnverifiableRemoteSpace { .. });
        });
        assert!(*rejected.lock().unwrap());
        assert_eq!(handle.join().unwrap().len(), 1);
    }

    #[test]
    fn authenticated_retry_uses_a_fresh_nonce_and_admits_only_final_response() {
        let frozen_identity = remote_test_identity(4);
        let raw_vectors = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let (endpoint, handle) =
            spawn_authenticated_gateway(frozen_identity.identity.clone(), raw_vectors, 1, true);
        let provider = Box::new(TestGatewayProvider {
            endpoint,
            dimension: 4,
        });
        let config = ApiEmbedderConfig {
            max_retries: 1,
            retry_base_delay: Duration::from_millis(1),
            ..ApiEmbedderConfig::default()
        };
        let embedder =
            ApiEmbedder::new_verified(provider, config, Some(frozen_identity), test_attester())
                .unwrap();

        let captured = Arc::new(Mutex::new(None));
        let captured_for_task = Arc::clone(&captured);
        run_test_with_cx(|cx| async move {
            let vectors = embedder.embed_batch(&cx, &["alpha"]).await.unwrap();
            *captured_for_task.lock().unwrap() = Some(vectors);
        });
        let vectors = captured.lock().unwrap().take().unwrap();
        let nonces = handle.join().unwrap();
        assert_eq!(vectors, vec![vec![1.0, 0.0, 0.0, 0.0]]);
        assert_eq!(nonces.len(), 2);
        assert_ne!(nonces[0], nonces[1]);
    }

    #[test]
    fn transport_cancellation_and_timeout_are_typed_and_redacted() {
        let provider = OpenAiProvider::custom(
            "credential-canary",
            "model",
            4,
            "https://user:password@example.invalid/private?token=secret",
        );
        let cancelled = api_client_error_to_search(&ClientError::Cancelled, &provider);
        assert!(matches!(cancelled, SearchError::Cancelled { .. }));

        let timed_out = api_client_error_to_search(&ClientError::DeadlineExceeded, &provider);
        let rendered = timed_out.to_string();
        assert!(matches!(timed_out, SearchError::EmbeddingFailed { .. }));
        for secret in [
            "credential-canary",
            "password",
            "example.invalid",
            "private",
            "secret",
        ] {
            assert!(!rendered.contains(secret));
        }
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
