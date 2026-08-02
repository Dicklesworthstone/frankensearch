//! Daemon client abstraction for warm embedding and reranking.
//!
//! This module defines the protocol-agnostic daemon interfaces shared by
//! host applications and fusion-layer fallback wrappers.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::generation::{EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1};
use crate::traits::ModelCategory;

/// Schema version for a daemon connection identity.
pub const DAEMON_CONNECTION_IDENTITY_SCHEMA_V1: u16 = 1;
/// Schema version for daemon challenges.
pub const DAEMON_CHALLENGE_SCHEMA_V1: u16 = 1;
/// Schema version for producer-authenticated daemon attestations.
pub const DAEMON_ATTESTATION_SCHEMA_V1: u16 = 1;
/// Minimum HMAC key length accepted by verified daemon clients.
pub const MIN_DAEMON_ATTESTATION_KEY_BYTES: usize = 32;

/// Retry/backoff configuration for daemon requests.
#[derive(Debug, Clone)]
pub struct DaemonRetryConfig {
    /// Max attempts per request (including the first try).
    pub max_attempts: u32,
    /// Base backoff delay for the first failure.
    pub base_delay: Duration,
    /// Maximum backoff delay.
    pub max_delay: Duration,
    /// Jitter percentage applied to backoff (0.0..=1.0).
    pub jitter_pct: f64,
}

impl Default for DaemonRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 2,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(5),
            jitter_pct: 0.2,
        }
    }
}

impl DaemonRetryConfig {
    /// Load retry config from environment variables; fall back to defaults.
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(val) = std::env::var("CASS_DAEMON_RETRY_MAX")
            && let Ok(parsed) = val.parse::<u32>()
        {
            cfg.max_attempts = parsed.max(1);
        }

        if let Ok(val) = std::env::var("CASS_DAEMON_BACKOFF_BASE_MS")
            && let Ok(parsed) = val.parse::<u64>()
        {
            cfg.base_delay = Duration::from_millis(parsed.max(1));
        }

        if let Ok(val) = std::env::var("CASS_DAEMON_BACKOFF_MAX_MS")
            && let Ok(parsed) = val.parse::<u64>()
        {
            cfg.max_delay = Duration::from_millis(parsed.max(1));
        }

        if let Ok(val) = std::env::var("CASS_DAEMON_JITTER_PCT")
            && let Ok(parsed) = val.parse::<f64>()
        {
            cfg.jitter_pct = parsed.clamp(0.0, 1.0);
        }

        cfg
    }

    /// Compute backoff for the given failure attempt.
    #[must_use]
    pub fn backoff_for_attempt(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        if let Some(explicit) = retry_after {
            return explicit.min(self.max_delay);
        }

        let exp = 2u32.saturating_pow(attempt.saturating_sub(1));
        let base = self.base_delay.checked_mul(exp).unwrap_or(self.max_delay);
        apply_jitter(base.min(self.max_delay), self.jitter_pct)
    }
}

/// Daemon request failure details.
#[derive(Debug, Clone)]
pub enum DaemonError {
    Unavailable(String),
    Timeout(String),
    Overloaded {
        retry_after: Option<Duration>,
        message: String,
    },
    Failed(String),
    InvalidInput(String),
    /// The remote operation was cancelled. Cancellation is never retried or
    /// converted into a local fallback.
    Cancelled,
    /// The daemon did not provide producer-authenticated embedding-space proof.
    UnverifiableRemoteSpace,
}

impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(msg) => write!(f, "daemon unavailable: {msg}"),
            Self::Timeout(msg) => write!(f, "daemon timeout: {msg}"),
            Self::Overloaded { message, .. } => write!(f, "daemon overloaded: {message}"),
            Self::Failed(msg) => write!(f, "daemon failed: {msg}"),
            Self::InvalidInput(msg) => write!(f, "daemon invalid input: {msg}"),
            Self::Cancelled => write!(f, "daemon request cancelled"),
            Self::UnverifiableRemoteSpace => {
                write!(f, "daemon embedding space is unverifiable")
            }
        }
    }
}

impl std::error::Error for DaemonError {}

/// Operation bound into a daemon challenge and its signed response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonOperationV1 {
    /// Initial connection handshake.
    Handshake,
    /// Liveness/readiness health proof.
    Health,
    /// One embedding input.
    Embed,
    /// An ordered embedding batch.
    EmbedBatch,
    /// An ordered rerank batch: input 0 is the query, the remainder are the
    /// documents. The response is one score row of `input_count - 1` scores.
    Rerank,
}

impl DaemonOperationV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::Handshake => 1,
            Self::Health => 2,
            Self::Embed => 3,
            Self::EmbedBatch => 4,
            Self::Rerank => 5,
        }
    }
}

/// Immutable identity of one authenticated daemon connection generation.
///
/// The endpoint and executable are represented only by canonical SHA-256
/// fingerprints. Raw socket paths, URLs, process arguments, and credentials
/// never enter this contract.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonConnectionIdentityV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Hash of the canonical endpoint identity.
    pub endpoint_fingerprint: String,
    /// Hash of the exact daemon executable/deployment identity.
    pub executable_fingerprint: String,
    /// Authenticated wire-protocol revision.
    pub protocol_revision: String,
    /// Public identifier of the HMAC key generation.
    pub key_id: String,
    /// Monotonic daemon generation. Zero is invalid.
    pub generation: u64,
    /// Complete mathematical-space, producer, input, and storage identity.
    pub embedding_identity: EmbeddingIdentityBundleV1,
    /// Operational model category exposed through the embedder interface.
    pub model_category: ModelCategory,
}

impl DaemonConnectionIdentityV1 {
    /// Validate every field and cross-field semantic classification.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for an unknown schema,
    /// malformed fingerprint, invalid generation, incomplete embedding
    /// identity, or category/space disagreement.
    pub fn validate(&self) -> Result<(), DaemonError> {
        if self.schema_version != DAEMON_CONNECTION_IDENTITY_SCHEMA_V1
            || !is_canonical_sha256(&self.endpoint_fingerprint)
            || !is_canonical_sha256(&self.executable_fingerprint)
            || !is_bounded_protocol_label(&self.protocol_revision)
            || !is_bounded_protocol_label(&self.key_id)
            || self.generation == 0
            || self.embedding_identity.validate().is_err()
        {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        let category_matches = match self.embedding_identity.space.kind {
            EmbeddingSpaceKindV1::HashControl => self.model_category == ModelCategory::HashEmbedder,
            EmbeddingSpaceKindV1::Semantic => self.model_category != ModelCategory::HashEmbedder,
        };
        if !category_matches {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        Ok(())
    }

    /// Canonical domain-separated, length-prefixed connection bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"frankensearch.daemon-connection.v1".to_vec();
        append_u16(&mut bytes, self.schema_version);
        append_text(&mut bytes, &self.endpoint_fingerprint);
        append_text(&mut bytes, &self.executable_fingerprint);
        append_text(&mut bytes, &self.protocol_revision);
        append_text(&mut bytes, &self.key_id);
        append_u64(&mut bytes, self.generation);
        append_bytes(&mut bytes, &self.embedding_identity.canonical_bytes());
        bytes.push(match self.model_category {
            ModelCategory::HashEmbedder => 1,
            ModelCategory::StaticEmbedder => 2,
            ModelCategory::TransformerEmbedder => 3,
            ModelCategory::ApiEmbedder => 4,
        });
        bytes
    }

    /// Lowercase SHA-256 of the complete connection identity.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        sha256_hex(&self.canonical_bytes())
    }
}

impl fmt::Debug for DaemonConnectionIdentityV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DaemonConnectionIdentityV1")
            .field("schema_version", &self.schema_version)
            .field("endpoint_fingerprint", &self.endpoint_fingerprint)
            .field("executable_fingerprint", &self.executable_fingerprint)
            .field(
                "protocol_revision_fingerprint",
                &sha256_hex(self.protocol_revision.as_bytes()),
            )
            .field("key_id_fingerprint", &sha256_hex(self.key_id.as_bytes()))
            .field("generation", &self.generation)
            .field(
                "embedding_identity_fingerprint",
                &self.embedding_identity.fingerprint(),
            )
            .field("model_category", &self.model_category)
            .finish()
    }
}

/// Fresh request challenge sent to the daemon.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonChallengeV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Fresh 32-byte nonce encoded as lowercase hexadecimal.
    pub request_nonce: String,
    /// Protocol operation.
    pub operation: DaemonOperationV1,
    /// Digest of the exact ordered input batch.
    pub ordered_request_sha256: String,
    /// Number of ordered input items.
    pub input_count: u32,
    /// Expected immutable connection fingerprint.
    pub expected_connection_fingerprint: String,
}

impl DaemonChallengeV1 {
    /// Build and validate a challenge for one exact ordered input set.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for malformed nonce,
    /// invalid connection identity, impossible input count, or an
    /// operation/input-shape mismatch.
    pub fn for_inputs(
        request_nonce: String,
        operation: DaemonOperationV1,
        inputs: &[&str],
        expected_connection: &DaemonConnectionIdentityV1,
    ) -> Result<Self, DaemonError> {
        expected_connection.validate()?;
        let input_count =
            u32::try_from(inputs.len()).map_err(|_| DaemonError::UnverifiableRemoteSpace)?;
        let challenge = Self {
            schema_version: DAEMON_CHALLENGE_SCHEMA_V1,
            request_nonce,
            operation,
            ordered_request_sha256: daemon_ordered_request_sha256(operation, inputs),
            input_count,
            expected_connection_fingerprint: expected_connection.fingerprint(),
        };
        challenge.validate()?;
        Ok(challenge)
    }

    /// Validate schema, digests, nonce, and operation shape.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for malformed fields.
    pub fn validate(&self) -> Result<(), DaemonError> {
        let shape_valid = match self.operation {
            DaemonOperationV1::Handshake | DaemonOperationV1::Health => self.input_count == 0,
            DaemonOperationV1::Embed => self.input_count == 1,
            DaemonOperationV1::EmbedBatch => self.input_count > 0,
            // Query plus at least one document.
            DaemonOperationV1::Rerank => self.input_count >= 2,
        };
        if self.schema_version != DAEMON_CHALLENGE_SCHEMA_V1
            || !is_canonical_sha256(&self.request_nonce)
            || !is_canonical_sha256(&self.ordered_request_sha256)
            || !is_canonical_sha256(&self.expected_connection_fingerprint)
            || !shape_valid
        {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"frankensearch.daemon-challenge.v1".to_vec();
        append_u16(&mut bytes, self.schema_version);
        append_text(&mut bytes, &self.request_nonce);
        bytes.push(self.operation.tag());
        append_text(&mut bytes, &self.ordered_request_sha256);
        append_u32(&mut bytes, self.input_count);
        append_text(&mut bytes, &self.expected_connection_fingerprint);
        bytes
    }
}

impl fmt::Debug for DaemonChallengeV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DaemonChallengeV1")
            .field("schema_version", &self.schema_version)
            .field("operation", &self.operation)
            .field("input_count", &self.input_count)
            .field(
                "expected_connection_fingerprint",
                &self.expected_connection_fingerprint,
            )
            .field("request_nonce", &"<redacted>")
            .field("ordered_request_sha256", &"<redacted>")
            .finish()
    }
}

/// Producer-authenticated proof covering a handshake, health response, or
/// embedding response.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DaemonEmbeddingAttestationV1 {
    /// Schema version; unknown versions fail closed.
    pub schema_version: u16,
    /// Exact caller challenge.
    pub challenge: DaemonChallengeV1,
    /// Full responder connection and embedding identity.
    pub connection: DaemonConnectionIdentityV1,
    /// Number of vectors in the response.
    pub vector_count: u32,
    /// Exact vector dimension advertised by the connection identity.
    pub vector_dimension: u32,
    /// Digest of vector order, shape, and exact f32 bits.
    pub response_payload_sha256: String,
    /// Lowercase HMAC-SHA256 over the unsigned canonical envelope.
    pub signature_hmac_sha256: String,
}

impl DaemonEmbeddingAttestationV1 {
    /// Construct an unsigned envelope after validating challenge, identity, and
    /// vector shape.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for any mismatch.
    pub fn unsigned(
        challenge: DaemonChallengeV1,
        connection: DaemonConnectionIdentityV1,
        vectors: &[Vec<f32>],
    ) -> Result<Self, DaemonError> {
        challenge.validate()?;
        connection.validate()?;
        if challenge.expected_connection_fingerprint != connection.fingerprint() {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        validate_response_shape(&challenge, &connection, vectors)?;
        let vector_dimension = expected_attested_dimension(&challenge, &connection)?;
        Ok(Self {
            schema_version: DAEMON_ATTESTATION_SCHEMA_V1,
            challenge,
            vector_count: u32::try_from(vectors.len())
                .map_err(|_| DaemonError::UnverifiableRemoteSpace)?,
            vector_dimension,
            response_payload_sha256: daemon_embedding_payload_sha256(vectors),
            connection,
            signature_hmac_sha256: String::new(),
        })
    }

    /// Sign the envelope with an HMAC-SHA256 key.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for a short key or
    /// invalid unsigned envelope.
    pub fn sign_hmac_sha256(&mut self, key: &[u8]) -> Result<(), DaemonError> {
        if key.len() < MIN_DAEMON_ATTESTATION_KEY_BYTES {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        self.validate_unsigned()?;
        self.signature_hmac_sha256 = encode_lower_hex(&hmac_sha256(key, &self.unsigned_bytes()));
        Ok(())
    }

    /// Authenticate the signature in constant time.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for a malformed
    /// envelope, short key, malformed signature, or authentication failure.
    pub fn authenticate_hmac_sha256(&self, key: &[u8]) -> Result<(), DaemonError> {
        if key.len() < MIN_DAEMON_ATTESTATION_KEY_BYTES {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        self.validate_unsigned()?;
        let observed = decode_canonical_sha256(&self.signature_hmac_sha256)
            .ok_or(DaemonError::UnverifiableRemoteSpace)?;
        let expected = hmac_sha256(key, &self.unsigned_bytes());
        if !constant_time_eq(&observed, &expected) {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        Ok(())
    }

    /// Validate the envelope against exact caller state and response vectors.
    ///
    /// Authentication remains a separate mandatory step so callers cannot
    /// accidentally treat structural validity as producer proof.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for any field, shape,
    /// digest, connection, or challenge mismatch.
    pub fn validate_against(
        &self,
        challenge: &DaemonChallengeV1,
        expected_connection: &DaemonConnectionIdentityV1,
        vectors: &[Vec<f32>],
    ) -> Result<(), DaemonError> {
        self.validate_unsigned()?;
        challenge.validate()?;
        expected_connection.validate()?;
        if self.challenge != *challenge
            || self.connection != *expected_connection
            || self.challenge.expected_connection_fingerprint != expected_connection.fingerprint()
            || self.vector_count
                != u32::try_from(vectors.len()).map_err(|_| DaemonError::UnverifiableRemoteSpace)?
            || self.vector_dimension != expected_attested_dimension(challenge, expected_connection)?
            || self.response_payload_sha256 != daemon_embedding_payload_sha256(vectors)
        {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        validate_response_shape(challenge, expected_connection, vectors)
    }

    fn validate_unsigned(&self) -> Result<(), DaemonError> {
        self.challenge.validate()?;
        self.connection.validate()?;
        let expected_vector_count = match self.challenge.operation {
            DaemonOperationV1::Handshake | DaemonOperationV1::Health => 0,
            DaemonOperationV1::Embed | DaemonOperationV1::Rerank => 1,
            DaemonOperationV1::EmbedBatch => self.challenge.input_count,
        };
        if self.schema_version != DAEMON_ATTESTATION_SCHEMA_V1
            || self.challenge.expected_connection_fingerprint != self.connection.fingerprint()
            || self.vector_count != expected_vector_count
            || self.vector_dimension
                != expected_attested_dimension(&self.challenge, &self.connection)?
            || !is_canonical_sha256(&self.response_payload_sha256)
        {
            return Err(DaemonError::UnverifiableRemoteSpace);
        }
        Ok(())
    }

    fn unsigned_bytes(&self) -> Vec<u8> {
        let mut bytes = b"frankensearch.daemon-attestation.v1".to_vec();
        append_u16(&mut bytes, self.schema_version);
        append_bytes(&mut bytes, &self.challenge.canonical_bytes());
        append_bytes(&mut bytes, &self.connection.canonical_bytes());
        append_u32(&mut bytes, self.vector_count);
        append_u32(&mut bytes, self.vector_dimension);
        append_text(&mut bytes, &self.response_payload_sha256);
        bytes
    }
}

impl fmt::Debug for DaemonEmbeddingAttestationV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DaemonEmbeddingAttestationV1")
            .field("schema_version", &self.schema_version)
            .field("operation", &self.challenge.operation)
            .field("input_count", &self.challenge.input_count)
            .field(
                "endpoint_fingerprint",
                &self.connection.endpoint_fingerprint,
            )
            .field(
                "protocol_revision_fingerprint",
                &sha256_hex(self.connection.protocol_revision.as_bytes()),
            )
            .field(
                "key_id_fingerprint",
                &sha256_hex(self.connection.key_id.as_bytes()),
            )
            .field("generation", &self.connection.generation)
            .field(
                "embedding_identity_fingerprint",
                &self.connection.embedding_identity.fingerprint(),
            )
            .field("vector_count", &self.vector_count)
            .field("vector_dimension", &self.vector_dimension)
            .field("request_nonce", &"<redacted>")
            .field("ordered_request_sha256", &"<redacted>")
            .field("response_payload_sha256", &"<redacted>")
            .field("signature_hmac_sha256", &"<redacted>")
            .finish()
    }
}

/// Vectors returned by a daemon together with their mandatory signed proof.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestedDaemonEmbeddingResponseV1 {
    /// Ordered response vectors.
    pub vectors: Vec<Vec<f32>>,
    /// Producer-authenticated response envelope.
    pub attestation: DaemonEmbeddingAttestationV1,
}

impl AttestedDaemonEmbeddingResponseV1 {
    /// Build and sign a response.
    ///
    /// # Errors
    ///
    /// Returns [`DaemonError::UnverifiableRemoteSpace`] for invalid contract
    /// fields, response shape, or key material.
    pub fn signed(
        challenge: DaemonChallengeV1,
        connection: DaemonConnectionIdentityV1,
        vectors: Vec<Vec<f32>>,
        key: &[u8],
    ) -> Result<Self, DaemonError> {
        let mut attestation =
            DaemonEmbeddingAttestationV1::unsigned(challenge, connection, &vectors)?;
        attestation.sign_hmac_sha256(key)?;
        Ok(Self {
            vectors,
            attestation,
        })
    }
}

impl fmt::Debug for AttestedDaemonEmbeddingResponseV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttestedDaemonEmbeddingResponseV1")
            .field("vector_count", &self.vectors.len())
            .field(
                "vector_dimension",
                &self.vectors.first().map_or(0, Vec::len),
            )
            .field("vectors", &"<redacted>")
            .field("attestation", &self.attestation)
            .finish()
    }
}

/// Abstract daemon client.
///
/// Concrete transports (e.g. UDS/HTTP) are implemented by host applications.
#[allow(clippy::missing_errors_doc)]
pub trait DaemonClient: Send + Sync {
    fn id(&self) -> &str;
    fn is_available(&self) -> bool;

    /// Return a producer-authenticated connection handshake.
    ///
    /// The fail-closed default makes a legacy/raw client ineligible for
    /// verified embedding paths.
    fn handshake_attested(
        &self,
        _challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    /// Return a producer-authenticated health/readiness proof.
    ///
    /// The fail-closed default makes a caller-supplied epoch insufficient.
    fn health_attested(
        &self,
        _challenge: &DaemonChallengeV1,
    ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    /// Embed one input and return a producer-authenticated response envelope.
    fn embed_attested(
        &self,
        _text: &str,
        _challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    /// Embed an ordered batch and return a producer-authenticated response
    /// envelope.
    fn embed_batch_attested(
        &self,
        _texts: &[&str],
        _challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    /// Rerank an ordered document batch against a query and return a
    /// producer-authenticated response envelope holding exactly one score row
    /// of `documents.len()` scores.
    ///
    /// The challenge's ordered inputs are the query followed by the documents
    /// (`DaemonOperationV1::Rerank`). The fail-closed default makes a
    /// legacy/raw client ineligible for verified rerank paths.
    fn rerank_attested(
        &self,
        _query: &str,
        _documents: &[&str],
        _challenge: &DaemonChallengeV1,
    ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
        Err(DaemonError::UnverifiableRemoteSpace)
    }

    /// Raw unverified inference primitive.
    ///
    /// This may only be exposed through an explicit transient assumed-mode
    /// wrapper. Verified embedders never call it.
    fn embed(&self, text: &str, request_id: &str) -> Result<Vec<f32>, DaemonError>;
    /// Raw unverified ordered-batch inference primitive.
    fn embed_batch(&self, texts: &[&str], request_id: &str) -> Result<Vec<Vec<f32>>, DaemonError>;
    fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        request_id: &str,
    ) -> Result<Vec<f32>, DaemonError>;
}

/// Domain-separated fingerprint of a canonical daemon endpoint identifier.
///
/// Callers should hash a resolved UDS identity or normalized URL and retain
/// only this digest in protocol state and logs.
#[must_use]
pub fn daemon_endpoint_fingerprint(endpoint: &str) -> String {
    let mut bytes = b"frankensearch.daemon-endpoint.v1".to_vec();
    append_text(&mut bytes, endpoint);
    sha256_hex(&bytes)
}

/// Domain-separated fingerprint of exact daemon executable/deployment bytes.
#[must_use]
pub fn daemon_executable_fingerprint(executable_identity: &[u8]) -> String {
    let mut bytes = b"frankensearch.daemon-executable.v1".to_vec();
    append_bytes(&mut bytes, executable_identity);
    sha256_hex(&bytes)
}

/// Digest an exact ordered daemon request with boundary-safe length prefixes.
#[must_use]
pub fn daemon_ordered_request_sha256(operation: DaemonOperationV1, inputs: &[&str]) -> String {
    let mut bytes = b"frankensearch.daemon-request.v1".to_vec();
    bytes.push(operation.tag());
    append_u64(&mut bytes, u64::try_from(inputs.len()).unwrap_or(u64::MAX));
    for input in inputs {
        append_text(&mut bytes, input);
    }
    sha256_hex(&bytes)
}

/// Digest vector order, shape, and exact f32 bits.
#[must_use]
pub fn daemon_embedding_payload_sha256(vectors: &[Vec<f32>]) -> String {
    let mut bytes = b"frankensearch.daemon-vector-payload.v1".to_vec();
    append_u64(&mut bytes, u64::try_from(vectors.len()).unwrap_or(u64::MAX));
    for vector in vectors {
        append_u64(&mut bytes, u64::try_from(vector.len()).unwrap_or(u64::MAX));
        for value in vector {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
    }
    sha256_hex(&bytes)
}

/// Expected per-row length of an attested response: the embedding dimension
/// for embed operations, `input_count - 1` scores for a rerank (input 0 is the
/// query, which receives no score).
fn expected_attested_dimension(
    challenge: &DaemonChallengeV1,
    connection: &DaemonConnectionIdentityV1,
) -> Result<u32, DaemonError> {
    match challenge.operation {
        DaemonOperationV1::Rerank => challenge
            .input_count
            .checked_sub(1)
            .ok_or(DaemonError::UnverifiableRemoteSpace),
        DaemonOperationV1::Handshake
        | DaemonOperationV1::Health
        | DaemonOperationV1::Embed
        | DaemonOperationV1::EmbedBatch => Ok(connection.embedding_identity.space.dimension),
    }
}

fn validate_response_shape(
    challenge: &DaemonChallengeV1,
    connection: &DaemonConnectionIdentityV1,
    vectors: &[Vec<f32>],
) -> Result<(), DaemonError> {
    let expected_count = match challenge.operation {
        DaemonOperationV1::Handshake | DaemonOperationV1::Health => 0,
        DaemonOperationV1::Embed | DaemonOperationV1::Rerank => 1,
        DaemonOperationV1::EmbedBatch => challenge.input_count,
    };
    let observed_count =
        u32::try_from(vectors.len()).map_err(|_| DaemonError::UnverifiableRemoteSpace)?;
    let row_len = usize::try_from(expected_attested_dimension(challenge, connection)?)
        .map_err(|_| DaemonError::UnverifiableRemoteSpace)?;
    // A rerank score row may legitimately be all zeros; embedding vectors must
    // carry signal in every row.
    let requires_nonzero = challenge.operation != DaemonOperationV1::Rerank;
    let vectors_valid = vectors.iter().all(|vector| {
        vector.len() == row_len
            && vector.iter().all(|value| value.is_finite())
            && (!requires_nonzero || vector.iter().any(|value| *value != 0.0))
    });
    if observed_count != expected_count || !vectors_valid {
        return Err(DaemonError::UnverifiableRemoteSpace);
    }
    Ok(())
}

fn is_bounded_protocol_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':')
        })
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

fn sha256_hex(bytes: &[u8]) -> String {
    encode_lower_hex(&Sha256::digest(bytes))
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

/// Apply bounded symmetric jitter to a duration.
#[must_use]
pub fn apply_jitter(duration: Duration, jitter_pct: f64) -> Duration {
    if jitter_pct <= 0.0 {
        return duration;
    }
    let unit = next_jitter_unit();
    let delta = unit.mul_add(2.0, -1.0) * jitter_pct;
    #[allow(clippy::cast_precision_loss)]
    let base_ms = duration.as_millis() as f64;
    let jittered = (base_ms * (1.0 + delta)).max(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Duration::from_millis(jittered.round() as u64)
}

/// Generate a stable daemon request id for tracing and retries.
#[must_use]
pub fn next_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("daemon-{id}")
}

fn next_jitter_unit() -> f64 {
    static SEED: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);
    let mut current = SEED.load(Ordering::Relaxed);
    loop {
        let next = current
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add(1);
        match SEED.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => {
                // Use top 53 bits for a uniform f64 in [0, 1).
                let value = next >> 11;
                #[allow(clippy::cast_precision_loss)]
                return (value as f64) / ((1_u64 << 53) as f64);
            }
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

    fn test_connection(model_id: &str) -> DaemonConnectionIdentityV1 {
        DaemonConnectionIdentityV1 {
            schema_version: DAEMON_CONNECTION_IDENTITY_SCHEMA_V1,
            endpoint_fingerprint: daemon_endpoint_fingerprint("unix:/run/frankensearch.sock"),
            executable_fingerprint: daemon_executable_fingerprint(b"fixture-daemon-v1"),
            protocol_revision: "frankensearch-daemon-v1".to_owned(),
            key_id: "fixture-key-v1".to_owned(),
            generation: 7,
            embedding_identity: EmbeddingIdentityBundleV1::explicit_test_model(model_id, 4),
            model_category: ModelCategory::HashEmbedder,
        }
    }

    fn nonce(byte: u8) -> String {
        encode_lower_hex(&[byte; 32])
    }

    #[test]
    fn backoff_respects_retry_after() {
        let cfg = DaemonRetryConfig::default();
        let retry_after = Duration::from_secs(1);
        assert_eq!(cfg.backoff_for_attempt(4, Some(retry_after)), retry_after);
    }

    #[test]
    fn jitter_stays_positive() {
        let base = Duration::from_millis(50);
        for _ in 0..100 {
            let jittered = apply_jitter(base, 0.2);
            assert!(jittered.as_millis() >= 1);
        }
    }

    #[test]
    fn ordered_request_digest_binds_operation_order_boundaries_and_content() {
        let ab = daemon_ordered_request_sha256(DaemonOperationV1::EmbedBatch, &["a", "bc"]);
        let ba = daemon_ordered_request_sha256(DaemonOperationV1::EmbedBatch, &["bc", "a"]);
        let joined = daemon_ordered_request_sha256(DaemonOperationV1::EmbedBatch, &["ab", "c"]);
        let single = daemon_ordered_request_sha256(DaemonOperationV1::Embed, &["a"]);
        let batch = daemon_ordered_request_sha256(DaemonOperationV1::EmbedBatch, &["a"]);
        assert_ne!(ab, ba);
        assert_ne!(ab, joined);
        assert_ne!(single, batch);
        assert_eq!(
            ab,
            daemon_ordered_request_sha256(DaemonOperationV1::EmbedBatch, &["a", "bc"])
        );
    }

    #[test]
    fn payload_digest_binds_order_shape_and_exact_f32_bits() {
        let original = daemon_embedding_payload_sha256(&[vec![1.0, -0.0], vec![2.0, 3.0]]);
        assert_ne!(
            original,
            daemon_embedding_payload_sha256(&[vec![2.0, 3.0], vec![1.0, -0.0]])
        );
        assert_ne!(
            original,
            daemon_embedding_payload_sha256(&[vec![1.0, -0.0, 2.0, 3.0]])
        );
        assert_ne!(
            original,
            daemon_embedding_payload_sha256(&[vec![1.0, 0.0], vec![2.0, 3.0]])
        );
    }

    #[test]
    fn signed_envelope_authenticates_exact_connection_challenge_and_payload() {
        let connection = test_connection("fixture-space");
        let challenge = DaemonChallengeV1::for_inputs(
            nonce(0x11),
            DaemonOperationV1::Embed,
            &["query"],
            &connection,
        )
        .unwrap();
        let response = AttestedDaemonEmbeddingResponseV1::signed(
            challenge.clone(),
            connection.clone(),
            vec![vec![1.0, 2.0, 3.0, 4.0]],
            TEST_KEY,
        )
        .unwrap();

        response
            .attestation
            .validate_against(&challenge, &connection, &response.vectors)
            .unwrap();
        response
            .attestation
            .authenticate_hmac_sha256(TEST_KEY)
            .unwrap();

        let wrong_key = b"abcdef0123456789abcdef0123456789";
        assert!(matches!(
            response.attestation.authenticate_hmac_sha256(wrong_key),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));

        let mut inconsistent_count = response.attestation;
        inconsistent_count.vector_count = 2;
        assert!(matches!(
            inconsistent_count.sign_hmac_sha256(TEST_KEY),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));
    }

    #[test]
    fn replay_payload_tamper_and_same_dimension_space_drift_fail_closed() {
        let connection = test_connection("fixture-space");
        let challenge = DaemonChallengeV1::for_inputs(
            nonce(0x22),
            DaemonOperationV1::EmbedBatch,
            &["first", "second"],
            &connection,
        )
        .unwrap();
        let response = AttestedDaemonEmbeddingResponseV1::signed(
            challenge.clone(),
            connection.clone(),
            vec![vec![1.0; 4], vec![2.0; 4]],
            TEST_KEY,
        )
        .unwrap();

        let replay = DaemonChallengeV1::for_inputs(
            nonce(0x23),
            DaemonOperationV1::EmbedBatch,
            &["first", "second"],
            &connection,
        )
        .unwrap();
        assert!(matches!(
            response
                .attestation
                .validate_against(&replay, &connection, &response.vectors),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));

        let mut tampered_vectors = response.vectors.clone();
        tampered_vectors.swap(0, 1);
        assert!(matches!(
            response
                .attestation
                .validate_against(&challenge, &connection, &tampered_vectors),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));

        let drifted = test_connection("same-dimension-different-space");
        assert!(matches!(
            response
                .attestation
                .validate_against(&challenge, &drifted, &response.vectors),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));
    }

    #[test]
    fn health_envelope_allows_no_vectors_but_embedding_rejects_zero_signal() {
        let connection = test_connection("fixture-space");
        let health =
            DaemonChallengeV1::for_inputs(nonce(0x33), DaemonOperationV1::Health, &[], &connection)
                .unwrap();
        let mut health_attestation =
            DaemonEmbeddingAttestationV1::unsigned(health, connection.clone(), &[]).unwrap();
        health_attestation.sign_hmac_sha256(TEST_KEY).unwrap();
        health_attestation
            .authenticate_hmac_sha256(TEST_KEY)
            .unwrap();

        let embed = DaemonChallengeV1::for_inputs(
            nonce(0x34),
            DaemonOperationV1::Embed,
            &["query"],
            &connection,
        )
        .unwrap();
        assert!(matches!(
            DaemonEmbeddingAttestationV1::unsigned(embed, connection, &[vec![0.0; 4]]),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));
    }

    #[test]
    fn protocol_debug_redacts_nonce_payload_signature_and_model_label() {
        let connection = test_connection("secret-model-label");
        let challenge = DaemonChallengeV1::for_inputs(
            nonce(0x44),
            DaemonOperationV1::Embed,
            &["secret query text"],
            &connection,
        )
        .unwrap();
        let response = AttestedDaemonEmbeddingResponseV1::signed(
            challenge,
            connection,
            vec![vec![1.0, 2.0, 3.0, 4.0]],
            TEST_KEY,
        )
        .unwrap();
        let rendered = format!("{response:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("secret-model-label"));
        assert!(!rendered.contains("secret query text"));
        assert!(!rendered.contains(&nonce(0x44)));
        assert!(!rendered.contains(&response.attestation.signature_hmac_sha256));
        assert!(!rendered.contains("[1.0, 2.0, 3.0, 4.0]"));
    }

    #[test]
    fn serde_rejects_unknown_attestation_fields_and_unknown_schema() {
        let connection = test_connection("fixture-space");
        let challenge = DaemonChallengeV1::for_inputs(
            nonce(0x55),
            DaemonOperationV1::Embed,
            &["query"],
            &connection,
        )
        .unwrap();
        let response = AttestedDaemonEmbeddingResponseV1::signed(
            challenge,
            connection,
            vec![vec![1.0; 4]],
            TEST_KEY,
        )
        .unwrap();
        let mut value = serde_json::to_value(&response.attestation).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("future_field".to_owned(), serde_json::json!(true));
        assert!(serde_json::from_value::<DaemonEmbeddingAttestationV1>(value).is_err());

        let mut unknown_schema = response.attestation;
        unknown_schema.schema_version = u16::MAX;
        assert!(matches!(
            unknown_schema.authenticate_hmac_sha256(TEST_KEY),
            Err(DaemonError::UnverifiableRemoteSpace)
        ));
    }
}
