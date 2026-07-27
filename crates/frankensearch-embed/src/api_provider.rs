//! Cloud API embedding provider trait and implementations (`OpenAI`, Gemini).
//!
//! Each provider knows its endpoint, auth, JSON format, and batch limits.
//! The shared [`super::api_embedder::ApiEmbedder`] handles HTTP, retry, and
//! rate-limiting generically over any `ApiProvider`.
//!
//! Gated behind the `api` feature flag.

use std::fmt;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema version for the producer-authenticated remote request challenge.
pub const REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1: u16 = 1;
/// Schema version for the producer-authenticated remote response envelope.
pub const REMOTE_EMBEDDING_ATTESTATION_SCHEMA_V1: u16 = 1;
/// Minimum entropy required for a pinned HMAC-SHA256 attestation key.
pub const MIN_REMOTE_ATTESTATION_KEY_BYTES: usize = 32;

/// One fresh, credential-free challenge sent to an authenticated embedding gateway.
///
/// Only digests and bounded identifiers cross this boundary; query text is carried
/// by the provider's ordinary request payload and is never duplicated in logs or
/// diagnostics.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteEmbeddingChallengeV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Fresh lowercase SHA-256 nonce for this one HTTP attempt.
    pub request_nonce: String,
    /// SHA-256 of the ordered, length-prefixed UTF-8 input batch.
    pub ordered_request_sha256: String,
    /// Number of ordered inputs covered by the request digest.
    pub input_count: u32,
    /// Fingerprint of the credential-free configured endpoint identity.
    pub endpoint_fingerprint: String,
    /// Fingerprint of the complete expected embedding identity bundle.
    pub identity_fingerprint: String,
    /// Mathematical embedding-space fingerprint.
    pub space_fingerprint: String,
    /// Concrete producer/backend fingerprint.
    pub producer_fingerprint: String,
}

impl fmt::Debug for RemoteEmbeddingChallengeV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteEmbeddingChallengeV1")
            .field("schema_version", &self.schema_version)
            .field("request_nonce", &"<redacted-nonce>")
            .field("ordered_request_sha256", &"<redacted-request-digest>")
            .field("input_count", &self.input_count)
            .field("endpoint_fingerprint", &self.endpoint_fingerprint)
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("space_fingerprint", &self.space_fingerprint)
            .field("producer_fingerprint", &self.producer_fingerprint)
            .finish()
    }
}

impl RemoteEmbeddingChallengeV1 {
    /// Validate all canonical fields without exposing request content.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for an unknown schema, malformed
    /// digest, or empty batch.
    pub fn validate(&self) -> SearchResult<()> {
        if self.schema_version != REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1 {
            return Err(remote_contract_error(
                "challenge uses an unsupported schema version",
            ));
        }
        for digest in [
            &self.request_nonce,
            &self.ordered_request_sha256,
            &self.endpoint_fingerprint,
            &self.identity_fingerprint,
            &self.space_fingerprint,
            &self.producer_fingerprint,
        ] {
            if !is_canonical_sha256(digest) {
                return Err(remote_contract_error(
                    "challenge contains a malformed canonical digest",
                ));
            }
        }
        if self.input_count == 0 {
            return Err(remote_contract_error(
                "challenge must bind at least one ordered input",
            ));
        }
        Ok(())
    }
}

/// Producer-authenticated envelope carried by one remote embedding response.
///
/// The signature is HMAC-SHA256 over [`Self::canonical_unsigned_bytes`].
/// Deployments provision the same high-entropy key into the trusted gateway
/// and the calling process. A caller-authored identity alone is never proof.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteEmbeddingAttestationV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Bounded identifier of the pinned authentication key.
    pub key_id: String,
    /// Fresh request nonce copied from [`RemoteEmbeddingChallengeV1`].
    pub request_nonce: String,
    /// Ordered request digest copied from the challenge.
    pub ordered_request_sha256: String,
    /// Number of ordered request inputs.
    pub input_count: u32,
    /// Fingerprint of the credential-free endpoint identity.
    pub endpoint_fingerprint: String,
    /// Complete identity-bundle fingerprint.
    pub identity_fingerprint: String,
    /// Mathematical embedding-space fingerprint.
    pub space_fingerprint: String,
    /// Concrete producer/backend fingerprint.
    pub producer_fingerprint: String,
    /// Canonical provider label.
    pub provider: String,
    /// Immutable upstream model identifier.
    pub model: String,
    /// Concrete producer backend from the canonical producer contract.
    pub producer_backend: String,
    /// Wire/inference protocol revision from the canonical producer contract.
    pub protocol_revision: String,
    /// Monotonic gateway generation pinned by the client.
    pub generation: u64,
    /// Number of vectors in the ordered response payload.
    pub vector_count: u32,
    /// Dimension of every vector in the ordered response payload.
    pub vector_dimension: u32,
    /// SHA-256 of the exact ordered f32 vector values and shape.
    pub response_payload_sha256: String,
    /// Lowercase HMAC-SHA256 over every preceding field.
    pub signature_hmac_sha256: String,
}

impl fmt::Debug for RemoteEmbeddingAttestationV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteEmbeddingAttestationV1")
            .field("schema_version", &self.schema_version)
            .field("key_id", &bounded_contract_label(&self.key_id))
            .field("request_nonce", &"<redacted-nonce>")
            .field("ordered_request_sha256", &"<redacted-request-digest>")
            .field("input_count", &self.input_count)
            .field("endpoint_fingerprint", &self.endpoint_fingerprint)
            .field("identity_fingerprint", &self.identity_fingerprint)
            .field("space_fingerprint", &self.space_fingerprint)
            .field("producer_fingerprint", &self.producer_fingerprint)
            .field("provider", &bounded_contract_label(&self.provider))
            .field("model", &"<redacted-model>")
            .field(
                "producer_backend",
                &bounded_contract_label(&self.producer_backend),
            )
            .field(
                "protocol_revision",
                &bounded_contract_label(&self.protocol_revision),
            )
            .field("generation", &self.generation)
            .field("vector_count", &self.vector_count)
            .field("vector_dimension", &self.vector_dimension)
            .field("response_payload_sha256", &"<redacted-response-digest>")
            .field("signature_hmac_sha256", &"<redacted-signature>")
            .finish()
    }
}

impl RemoteEmbeddingAttestationV1 {
    /// Construct an unsigned envelope for a gateway response.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` when the response shape does not fit
    /// the versioned envelope.
    pub fn unsigned(
        challenge: &RemoteEmbeddingChallengeV1,
        identity: &EmbeddingIdentityBundleV1,
        provider: impl Into<String>,
        model: impl Into<String>,
        generation: u64,
        vectors: &[Vec<f32>],
        key_id: impl Into<String>,
    ) -> SearchResult<Self> {
        challenge.validate()?;
        if challenge.identity_fingerprint != identity.fingerprint()
            || challenge.space_fingerprint != identity.space.fingerprint()
            || challenge.producer_fingerprint != identity.producer.fingerprint()
        {
            return Err(remote_contract_error(
                "gateway identity does not match the request challenge",
            ));
        }
        let vector_count = u32::try_from(vectors.len())
            .map_err(|_| remote_contract_error("response vector count exceeds schema capacity"))?;
        let vector_dimension = vectors.first().map_or(0, Vec::len);
        if vectors
            .iter()
            .any(|vector| vector.len() != vector_dimension)
        {
            return Err(remote_contract_error(
                "response vectors do not have one stable dimension",
            ));
        }
        let vector_dimension = u32::try_from(vector_dimension).map_err(|_| {
            remote_contract_error("response vector dimension exceeds schema capacity")
        })?;

        Ok(Self {
            schema_version: REMOTE_EMBEDDING_ATTESTATION_SCHEMA_V1,
            key_id: key_id.into(),
            request_nonce: challenge.request_nonce.clone(),
            ordered_request_sha256: challenge.ordered_request_sha256.clone(),
            input_count: challenge.input_count,
            endpoint_fingerprint: challenge.endpoint_fingerprint.clone(),
            identity_fingerprint: challenge.identity_fingerprint.clone(),
            space_fingerprint: challenge.space_fingerprint.clone(),
            producer_fingerprint: challenge.producer_fingerprint.clone(),
            provider: provider.into(),
            model: model.into(),
            producer_backend: identity.producer.backend.clone(),
            protocol_revision: identity.producer.protocol_revision.clone(),
            generation,
            vector_count,
            vector_dimension,
            response_payload_sha256: remote_embedding_payload_sha256(vectors),
            signature_hmac_sha256: String::new(),
        })
    }

    /// Sign this envelope with a high-entropy gateway/client shared key.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for a short key or malformed envelope.
    pub fn sign_hmac_sha256(&mut self, authentication_key: &[u8]) -> SearchResult<()> {
        if authentication_key.len() < MIN_REMOTE_ATTESTATION_KEY_BYTES {
            return Err(remote_contract_error(
                "remote attestation key has insufficient entropy",
            ));
        }
        self.validate_unsigned()?;
        self.signature_hmac_sha256 = encode_lower_hex(&hmac_sha256(
            authentication_key,
            &self.canonical_unsigned_bytes(),
        ));
        Ok(())
    }

    /// Verify the HMAC in constant time against a pinned shared key.
    #[must_use]
    pub fn authenticate_hmac_sha256(&self, authentication_key: &[u8]) -> bool {
        if authentication_key.len() < MIN_REMOTE_ATTESTATION_KEY_BYTES
            || self.validate_unsigned().is_err()
        {
            return false;
        }
        let Some(observed) = decode_canonical_sha256(&self.signature_hmac_sha256) else {
            return false;
        };
        let expected = hmac_sha256(authentication_key, &self.canonical_unsigned_bytes());
        constant_time_eq(&observed, &expected)
    }

    /// Domain-separated, length-prefixed bytes authenticated by the gateway.
    #[must_use]
    pub fn canonical_unsigned_bytes(&self) -> Vec<u8> {
        let mut bytes = b"frankensearch.remote-embedding-attestation.hmac-sha256.v1".to_vec();
        append_u16(&mut bytes, self.schema_version);
        for value in [
            &self.key_id,
            &self.request_nonce,
            &self.ordered_request_sha256,
        ] {
            append_text(&mut bytes, value);
        }
        append_u32(&mut bytes, self.input_count);
        for value in [
            &self.endpoint_fingerprint,
            &self.identity_fingerprint,
            &self.space_fingerprint,
            &self.producer_fingerprint,
            &self.provider,
            &self.model,
            &self.producer_backend,
            &self.protocol_revision,
        ] {
            append_text(&mut bytes, value);
        }
        append_u64(&mut bytes, self.generation);
        append_u32(&mut bytes, self.vector_count);
        append_u32(&mut bytes, self.vector_dimension);
        append_text(&mut bytes, &self.response_payload_sha256);
        bytes
    }

    fn validate_unsigned(&self) -> SearchResult<()> {
        if self.schema_version != REMOTE_EMBEDDING_ATTESTATION_SCHEMA_V1 {
            return Err(remote_contract_error(
                "response attestation uses an unsupported schema version",
            ));
        }
        for label in [
            &self.key_id,
            &self.provider,
            &self.model,
            &self.producer_backend,
            &self.protocol_revision,
        ] {
            if bounded_contract_label(label) != *label {
                return Err(remote_contract_error(
                    "response attestation contains an invalid bounded identifier",
                ));
            }
        }
        for digest in [
            &self.request_nonce,
            &self.ordered_request_sha256,
            &self.endpoint_fingerprint,
            &self.identity_fingerprint,
            &self.space_fingerprint,
            &self.producer_fingerprint,
            &self.response_payload_sha256,
        ] {
            if !is_canonical_sha256(digest) {
                return Err(remote_contract_error(
                    "response attestation contains a malformed canonical digest",
                ));
            }
        }
        if self.input_count == 0 || self.vector_count == 0 || self.vector_dimension == 0 {
            return Err(remote_contract_error(
                "response attestation contains an empty batch shape",
            ));
        }
        Ok(())
    }
}

/// SHA-256 of one credential-free endpoint identity.
#[must_use]
pub fn remote_endpoint_fingerprint(endpoint_identity: &str) -> String {
    encode_lower_hex(&Sha256::digest(endpoint_identity.as_bytes()))
}

/// SHA-256 of an ordered, length-prefixed UTF-8 request batch.
#[must_use]
pub fn remote_ordered_request_sha256(texts: &[&str]) -> String {
    let mut bytes = b"frankensearch.remote-embedding-request.v1".to_vec();
    append_u64(&mut bytes, u64::try_from(texts.len()).unwrap_or(u64::MAX));
    for text in texts {
        append_bytes(&mut bytes, text.as_bytes());
    }
    encode_lower_hex(&Sha256::digest(&bytes))
}

/// SHA-256 of the exact ordered f32 vector values and their shape.
#[must_use]
pub fn remote_embedding_payload_sha256(vectors: &[Vec<f32>]) -> String {
    let mut bytes = b"frankensearch.remote-embedding-payload.f32.v1".to_vec();
    append_u64(&mut bytes, u64::try_from(vectors.len()).unwrap_or(u64::MAX));
    for vector in vectors {
        append_u64(&mut bytes, u64::try_from(vector.len()).unwrap_or(u64::MAX));
        for value in vector {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
    }
    encode_lower_hex(&Sha256::digest(&bytes))
}

fn remote_contract_error(reason: &str) -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "remote-api".to_owned(),
        reason: reason.to_owned(),
    }
}

fn bounded_contract_label(value: &str) -> String {
    if !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/'))
    {
        value.to_owned()
    } else {
        "<redacted-remote-identifier>".to_owned()
    }
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn decode_canonical_sha256(value: &str) -> Option<[u8; 32]> {
    if !is_canonical_sha256(value) {
        return None;
    }
    let mut decoded = [0_u8; 32];
    for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        decoded[index] = (decode_hex_nibble(pair[0])? << 4) | decode_hex_nibble(pair[1])?;
    }
    Some(decoded)
}

const fn decode_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_BYTES: usize = 64;

    let mut normalized_key = [0_u8; BLOCK_BYTES];
    if key.len() > BLOCK_BYTES {
        let digest = Sha256::digest(key);
        normalized_key[..digest.len()].copy_from_slice(&digest);
    } else {
        normalized_key[..key.len()].copy_from_slice(key);
    }

    let mut inner_pad = [0x36_u8; BLOCK_BYTES];
    let mut outer_pad = [0x5c_u8; BLOCK_BYTES];
    for index in 0..BLOCK_BYTES {
        inner_pad[index] ^= normalized_key[index];
        outer_pad[index] ^= normalized_key[index];
    }

    let mut inner = Sha256::new();
    inner.update(inner_pad);
    inner.update(message);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(outer_pad);
    outer.update(inner_digest);
    outer.finalize().into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    let mut difference = 0_u8;
    for index in 0..left.len() {
        difference |= left[index] ^ right[index];
    }
    difference == 0
}

fn append_bytes(output: &mut Vec<u8>, value: &[u8]) {
    append_u64(output, u64::try_from(value.len()).unwrap_or(u64::MAX));
    output.extend_from_slice(value);
}

fn append_text(output: &mut Vec<u8>, value: &str) {
    append_bytes(output, value.as_bytes());
}

fn append_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn append_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn append_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_be_bytes());
}

// ─── ApiProvider trait ──────────────────────────────────────────────────────

/// Abstraction over cloud embedding API differences.
///
/// Implementors encode provider-specific details (URL, auth scheme, JSON
/// schema, batch limits) so that `ApiEmbedder` can drive any provider
/// uniformly.
pub trait ApiProvider: Send + Sync + fmt::Debug {
    /// Human-readable provider name (e.g. `"openai"`, `"gemini"`).
    fn provider_name(&self) -> &str;

    /// Model ID sent to the API (e.g. `"text-embedding-3-small"`).
    fn api_model_id(&self) -> &str;

    /// Stable operational identifier for registry selection and diagnostics.
    ///
    /// This display-level ID never establishes vector-space compatibility.
    fn embedder_id(&self) -> &str;

    /// Canonical producer backend value required by an explicit immutable
    /// identity epoch.
    fn identity_backend(&self) -> &'static str;

    /// Canonical wire-protocol revision required by an explicit immutable
    /// identity epoch.
    fn identity_protocol_revision(&self) -> &'static str;

    /// Output embedding dimensionality.
    fn dimension(&self) -> usize;

    /// Maximum texts per single API call.
    fn max_batch_size(&self) -> usize;

    /// Whether this model supports Matryoshka Representation Learning.
    fn supports_mrl(&self) -> bool;

    /// Base endpoint URL for the embedding request.
    fn endpoint_url(&self) -> &str;

    /// Full request URL (may include query parameters like API keys).
    /// Defaults to `endpoint_url()`.
    fn request_url(&self) -> String {
        self.endpoint_url().to_owned()
    }

    /// HTTP headers (excluding content-type which is always application/json).
    fn request_headers(&self) -> Vec<(String, String)>;

    /// Serialize a batch of texts into the provider's JSON request body.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbeddingFailed`] when the request cannot be
    /// serialized.
    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>>;

    /// Serialize a verified request and carry its fresh challenge to an
    /// authenticated gateway.
    ///
    /// Direct provider protocols do not support this envelope and retain their
    /// ordinary request shape. Such providers consequently return no response
    /// attestation and cannot satisfy [`ApiEmbedder`](crate::ApiEmbedder).
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingFailed` when request serialization fails.
    fn serialize_attested_request(
        &self,
        texts: &[&str],
        _challenge: &RemoteEmbeddingChallengeV1,
    ) -> SearchResult<Vec<u8>> {
        self.serialize_request(texts)
    }

    /// Deserialize the provider's JSON response into embedding vectors.
    /// The returned vectors MUST be in the same order as the input texts.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::EmbeddingFailed`] when the response is malformed
    /// or reports a provider error.
    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>>;

    /// Extract a per-response immutable space/producer attestation.
    ///
    /// Providers whose wire protocol does not carry this contract return
    /// `None`. [`ApiEmbedder`](crate::ApiEmbedder) rejects such responses
    /// because a caller-supplied epoch cannot authenticate the responding
    /// service.
    ///
    /// # Errors
    ///
    /// Returns `EmbeddingFailed` when an attestation field is present but
    /// malformed.
    fn response_attestation(
        &self,
        _body: &[u8],
    ) -> SearchResult<Option<RemoteEmbeddingAttestationV1>> {
        Ok(None)
    }
}

// ─── OpenAI ─────────────────────────────────────────────────────────────────

/// `OpenAI` embeddings API provider (`text-embedding-3-small`, `text-embedding-3-large`).
#[derive(Clone)]
pub struct OpenAiProvider {
    api_key: String,
    model: String,
    dimension: usize,
    endpoint: String,
    embedder_id: String,
}

impl fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiProvider")
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .field("endpoint", &"<redacted>")
            .field("api_key", &"<redacted>")
            .field("embedder_id", &self.embedder_id)
            .finish()
    }
}

impl OpenAiProvider {
    /// Create an `OpenAI` provider for `text-embedding-3-small`.
    ///
    /// Default dimension is 1536; pass a smaller value for MRL truncation.
    #[must_use]
    pub fn text_embedding_3_small(api_key: impl Into<String>, dimension: Option<usize>) -> Self {
        let dim = dimension.unwrap_or(1536);
        Self {
            api_key: api_key.into(),
            model: "text-embedding-3-small".to_owned(),
            dimension: dim,
            endpoint: "https://api.openai.com/v1/embeddings".to_owned(),
            embedder_id: format!("openai-text-embedding-3-small-{dim}d"),
        }
    }

    /// Create an `OpenAI` provider for `text-embedding-3-large`.
    ///
    /// Default dimension is 3072; pass a smaller value for MRL truncation.
    #[must_use]
    pub fn text_embedding_3_large(api_key: impl Into<String>, dimension: Option<usize>) -> Self {
        let dim = dimension.unwrap_or(3072);
        Self {
            api_key: api_key.into(),
            model: "text-embedding-3-large".to_owned(),
            dimension: dim,
            endpoint: "https://api.openai.com/v1/embeddings".to_owned(),
            embedder_id: format!("openai-text-embedding-3-large-{dim}d"),
        }
    }

    /// Create a fully custom OpenAI-compatible provider.
    #[must_use]
    pub fn custom(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
        endpoint: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let embedder_id = format!("openai-{model}-{dimension}d");
        Self {
            api_key: api_key.into(),
            model,
            dimension,
            endpoint: endpoint.into(),
            embedder_id,
        }
    }
}

impl ApiProvider for OpenAiProvider {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn api_model_id(&self) -> &str {
        &self.model
    }

    fn embedder_id(&self) -> &str {
        &self.embedder_id
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
        2048
    }

    fn supports_mrl(&self) -> bool {
        self.model.starts_with("text-embedding-3-")
    }

    fn endpoint_url(&self) -> &str {
        &self.endpoint
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        vec![
            (
                "authorization".to_owned(),
                format!("Bearer {}", self.api_key),
            ),
            ("content-type".to_owned(), "application/json".to_owned()),
        ]
    }

    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>> {
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
            "dimensions": self.dimension,
            "encoding_format": "float"
        });
        serde_json::to_vec(&body).map_err(|e| SearchError::EmbeddingFailed {
            model: self.embedder_id.clone(),
            source: e.into(),
        })
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "the API's JSON float values are defined to produce f32 embeddings"
    )]
    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
        let v: serde_json::Value =
            serde_json::from_slice(body).map_err(|e| SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("JSON parse error: {e}").into(),
            })?;

        // Check for API-level error.
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown API error");
            return Err(SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("OpenAI API error: {msg}").into(),
            });
        }

        let data = v.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
            SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: "missing 'data' array in response".into(),
            }
        })?;

        // Sort by index field to ensure correct ordering.
        let mut indexed: Vec<(usize, Vec<f32>)> = data
            .iter()
            .map(|item| {
                let raw_index = item
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let idx = usize::try_from(raw_index).map_err(|_| SearchError::EmbeddingFailed {
                    model: self.embedder_id.clone(),
                    source: format!("embedding index {raw_index} exceeds usize::MAX").into(),
                })?;
                let emb = item
                    .get("embedding")
                    .and_then(|e| e.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                Ok((idx, emb))
            })
            .collect::<SearchResult<_>>()?;
        indexed.sort_by_key(|(idx, _)| *idx);

        Ok(indexed.into_iter().map(|(_, emb)| emb).collect())
    }
}

// ─── Gemini ─────────────────────────────────────────────────────────────────

/// Google Gemini embeddings API provider (`text-embedding-004`, `embedding-001`).
#[derive(Clone)]
pub struct GeminiProvider {
    api_key: String,
    model: String,
    dimension: usize,
    embedder_id: String,
}

impl fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("model", &self.model)
            .field("dimension", &self.dimension)
            .field("api_key", &"<redacted>")
            .field("embedder_id", &self.embedder_id)
            .finish()
    }
}

impl GeminiProvider {
    /// Create a Gemini provider for `text-embedding-004` (768-dimensional).
    #[must_use]
    pub fn text_embedding_004(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "text-embedding-004".to_owned(),
            dimension: 768,
            embedder_id: "gemini-text-embedding-004-768d".to_owned(),
        }
    }

    /// Create a Gemini provider for `embedding-001` (768-dimensional).
    #[must_use]
    pub fn embedding_001(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "embedding-001".to_owned(),
            dimension: 768,
            embedder_id: "gemini-embedding-001-768d".to_owned(),
        }
    }
}

impl ApiProvider for GeminiProvider {
    fn provider_name(&self) -> &'static str {
        "gemini"
    }

    fn api_model_id(&self) -> &str {
        &self.model
    }

    fn embedder_id(&self) -> &str {
        &self.embedder_id
    }

    fn identity_backend(&self) -> &'static str {
        "remote-api-gemini"
    }

    fn identity_protocol_revision(&self) -> &'static str {
        "gemini-batch-embed-content-json-v1"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_batch_size(&self) -> usize {
        100
    }

    fn supports_mrl(&self) -> bool {
        false
    }

    fn endpoint_url(&self) -> &'static str {
        "https://generativelanguage.googleapis.com"
    }

    fn request_url(&self) -> String {
        self.batch_embed_url()
    }

    fn request_headers(&self) -> Vec<(String, String)> {
        vec![("content-type".to_owned(), "application/json".to_owned())]
    }

    fn serialize_request(&self, texts: &[&str]) -> SearchResult<Vec<u8>> {
        let requests: Vec<serde_json::Value> = texts
            .iter()
            .map(|text| {
                serde_json::json!({
                    "model": format!("models/{}", self.model),
                    "content": {
                        "parts": [{"text": text}]
                    }
                })
            })
            .collect();

        let body = serde_json::json!({ "requests": requests });
        serde_json::to_vec(&body).map_err(|e| SearchError::EmbeddingFailed {
            model: self.embedder_id.clone(),
            source: e.into(),
        })
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "the API's JSON float values are defined to produce f32 embeddings"
    )]
    fn deserialize_response(&self, body: &[u8]) -> SearchResult<Vec<Vec<f32>>> {
        let v: serde_json::Value =
            serde_json::from_slice(body).map_err(|e| SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("JSON parse error: {e}").into(),
            })?;

        // Check for API-level error.
        if let Some(err) = v.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown API error");
            return Err(SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: format!("Gemini API error: {msg}").into(),
            });
        }

        let embeddings = v
            .get("embeddings")
            .and_then(|e| e.as_array())
            .ok_or_else(|| SearchError::EmbeddingFailed {
                model: self.embedder_id.clone(),
                source: "missing 'embeddings' array in response".into(),
            })?;

        embeddings
            .iter()
            .map(|item| {
                item.get("values")
                    .and_then(|vals| vals.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .ok_or_else(|| SearchError::EmbeddingFailed {
                        model: self.embedder_id.clone(),
                        source: "missing 'values' in embedding entry".into(),
                    })
            })
            .collect()
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

impl GeminiProvider {
    /// Construct the full batch-embed URL including API key.
    #[must_use]
    pub fn batch_embed_url(&self) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
            self.model, self.api_key
        )
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_serialize_request() {
        let p = OpenAiProvider::text_embedding_3_small("test-key", Some(256));
        let body = p.serialize_request(&["hello", "world"]).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], "text-embedding-3-small");
        assert_eq!(v["dimensions"], 256);
        assert_eq!(v["input"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn openai_deserialize_response() {
        let p = OpenAiProvider::text_embedding_3_small("test-key", Some(3));
        let response = serde_json::json!({
            "data": [
                {"index": 1, "embedding": [0.1, 0.2, 0.3]},
                {"index": 0, "embedding": [0.4, 0.5, 0.6]}
            ]
        });
        let embeddings = p
            .deserialize_response(&serde_json::to_vec(&response).unwrap())
            .unwrap();
        // Should be sorted by index.
        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 0.4).abs() < f32::EPSILON);
        assert!((embeddings[1][0] - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn openai_error_response() {
        let p = OpenAiProvider::text_embedding_3_small("test-key", None);
        let response = serde_json::json!({
            "error": {"message": "Invalid API key", "type": "auth_error"}
        });
        let err = p
            .deserialize_response(&serde_json::to_vec(&response).unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("Invalid API key"));
    }

    #[test]
    fn gemini_serialize_request() {
        let p = GeminiProvider::text_embedding_004("test-key");
        let body = p.serialize_request(&["hello"]).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let requests = v["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["model"], "models/text-embedding-004");
    }

    #[test]
    fn gemini_deserialize_response() {
        let p = GeminiProvider::text_embedding_004("test-key");
        let response = serde_json::json!({
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]}
            ]
        });
        let embeddings = p
            .deserialize_response(&serde_json::to_vec(&response).unwrap())
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 3);
    }

    #[test]
    fn gemini_batch_url() {
        let p = GeminiProvider::text_embedding_004("mykey");
        assert!(p.batch_embed_url().contains("text-embedding-004"));
        assert!(p.batch_embed_url().contains("key=mykey"));
    }

    #[test]
    fn openai_supports_mrl() {
        let p = OpenAiProvider::text_embedding_3_small("k", None);
        assert!(p.supports_mrl());
        let p2 = OpenAiProvider::custom("k", "ada-002", 1536, "https://example.com");
        assert!(!p2.supports_mrl());
    }

    #[test]
    fn openai_embedder_id_includes_dimension() {
        let p = OpenAiProvider::text_embedding_3_small("k", Some(512));
        assert_eq!(p.embedder_id(), "openai-text-embedding-3-small-512d");
    }

    #[test]
    fn provider_debug_redacts_credentials_and_endpoint() {
        let openai = OpenAiProvider::custom(
            "openai-secret",
            "model",
            8,
            "https://user:password@example.invalid/private",
        );
        let openai_debug = format!("{openai:?}");
        assert!(!openai_debug.contains("openai-secret"));
        assert!(!openai_debug.contains("password"));
        assert!(!openai_debug.contains("example.invalid"));

        let gemini = GeminiProvider::text_embedding_004("gemini-secret");
        assert!(!format!("{gemini:?}").contains("gemini-secret"));
    }

    #[test]
    fn remote_contract_debug_redacts_request_response_and_authentication_material() {
        let request_nonce = "a".repeat(64);
        let request_digest = "b".repeat(64);
        let response_digest = "c".repeat(64);
        let signature = "d".repeat(64);
        let challenge = RemoteEmbeddingChallengeV1 {
            schema_version: REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1,
            request_nonce: request_nonce.clone(),
            ordered_request_sha256: request_digest.clone(),
            input_count: 1,
            endpoint_fingerprint: "e".repeat(64),
            identity_fingerprint: "f".repeat(64),
            space_fingerprint: "1".repeat(64),
            producer_fingerprint: "2".repeat(64),
        };
        let challenge_debug = format!("{challenge:?}");
        assert!(!challenge_debug.contains(&request_nonce));
        assert!(!challenge_debug.contains(&request_digest));

        let attestation = RemoteEmbeddingAttestationV1 {
            schema_version: REMOTE_EMBEDDING_ATTESTATION_SCHEMA_V1,
            key_id: "key-id".to_owned(),
            request_nonce,
            ordered_request_sha256: request_digest,
            input_count: 1,
            endpoint_fingerprint: challenge.endpoint_fingerprint,
            identity_fingerprint: challenge.identity_fingerprint,
            space_fingerprint: challenge.space_fingerprint,
            producer_fingerprint: challenge.producer_fingerprint,
            provider: "provider".to_owned(),
            model: "model-canary".to_owned(),
            producer_backend: "backend".to_owned(),
            protocol_revision: "protocol-v1".to_owned(),
            generation: 1,
            vector_count: 1,
            vector_dimension: 2,
            response_payload_sha256: response_digest.clone(),
            signature_hmac_sha256: signature.clone(),
        };
        let attestation_debug = format!("{attestation:?}");
        assert!(!attestation_debug.contains("model-canary"));
        assert!(!attestation_debug.contains(&response_digest));
        assert!(!attestation_debug.contains(&signature));
    }

    #[test]
    fn hmac_sha256_matches_rfc_4231_case_one() {
        let observed = encode_lower_hex(&hmac_sha256(&[0x0b; 20], b"Hi There"));
        assert_eq!(
            observed,
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn ordered_request_digest_binds_order_boundaries_and_content() {
        let original = remote_ordered_request_sha256(&["alpha", "beta"]);
        assert_eq!(original.len(), 64);
        assert_ne!(original, remote_ordered_request_sha256(&["beta", "alpha"]));
        assert_ne!(original, remote_ordered_request_sha256(&["alphabeta"]));
        assert_ne!(original, remote_ordered_request_sha256(&["alpha", "beta "]));
    }

    #[test]
    fn response_payload_digest_binds_order_shape_and_exact_f32_bits() {
        let original =
            remote_embedding_payload_sha256(&[vec![1.0, -0.0], vec![f32::from_bits(1), 2.0]]);
        assert_ne!(
            original,
            remote_embedding_payload_sha256(&[vec![f32::from_bits(1), 2.0], vec![1.0, -0.0]])
        );
        assert_ne!(
            original,
            remote_embedding_payload_sha256(&[vec![1.0, 0.0], vec![f32::from_bits(1), 2.0]])
        );
        assert_ne!(
            original,
            remote_embedding_payload_sha256(&[vec![1.0, -0.0, f32::from_bits(1), 2.0]])
        );
    }

    #[test]
    fn challenge_rejects_unknown_schema_and_noncanonical_digests() {
        let valid_digest = "a".repeat(64);
        let mut challenge = RemoteEmbeddingChallengeV1 {
            schema_version: REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1,
            request_nonce: valid_digest.clone(),
            ordered_request_sha256: valid_digest.clone(),
            input_count: 1,
            endpoint_fingerprint: valid_digest.clone(),
            identity_fingerprint: valid_digest.clone(),
            space_fingerprint: valid_digest.clone(),
            producer_fingerprint: valid_digest,
        };
        assert!(challenge.validate().is_ok());

        challenge.schema_version += 1;
        assert!(matches!(
            challenge.validate(),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        challenge.schema_version = REMOTE_EMBEDDING_CHALLENGE_SCHEMA_V1;
        challenge.request_nonce = "ABC".to_owned();
        assert!(matches!(
            challenge.validate(),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }
}
