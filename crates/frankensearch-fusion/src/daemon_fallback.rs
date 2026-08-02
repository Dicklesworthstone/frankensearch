//! Daemon fallback wrappers for sync embedders and rerankers.
//!
//! These wrappers attempt daemon inference first and gracefully fall back to
//! local in-process models with bounded retry and jittered backoff.

use std::fs::File;
use std::io::Read as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use frankensearch_core::{
    AttestedDaemonEmbeddingResponseV1, DaemonChallengeV1, DaemonClient, DaemonConnectionIdentityV1,
    DaemonEmbeddingAttestationV1, DaemonError, DaemonOperationV1, DaemonRetryConfig,
    EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1, MIN_DAEMON_ATTESTATION_KEY_BYTES,
    ModelCategory, RerankDocument, RerankScore, SearchError, SearchResult, SyncEmbed, SyncRerank,
    next_request_id,
};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

/// Trust level carried by daemon embedding APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonTrustLevelV1 {
    /// Every accepted response is producer-authenticated.
    VerifiedRemote,
    /// Raw daemon vectors are explicit, transient, and ineligible for
    /// persistence/cache/index APIs.
    AssumedRemote,
}

/// Pinned HMAC verifier for one public daemon key identifier.
pub struct PinnedDaemonVerifierV1 {
    key_id: String,
    secret_key: Vec<u8>,
}

impl PinnedDaemonVerifierV1 {
    /// Construct a verifier with at least 256 bits of key material.
    ///
    /// # Errors
    ///
    /// Returns `UnverifiableRemoteSpace` for malformed identifiers or short
    /// keys.
    pub fn new(key_id: impl Into<String>, secret_key: Vec<u8>) -> SearchResult<Self> {
        let key_id = key_id.into();
        if !is_bounded_daemon_label(&key_id) || secret_key.len() < MIN_DAEMON_ATTESTATION_KEY_BYTES
        {
            return Err(SearchError::UnverifiableRemoteSpace {
                producer: "<redacted-daemon-producer>".to_owned(),
                reason: "invalid pinned daemon verifier".to_owned(),
            });
        }
        Ok(Self { key_id, secret_key })
    }

    fn validate_for(&self, connection: &DaemonConnectionIdentityV1) -> SearchResult<()> {
        if self.key_id != connection.key_id
            || self.secret_key.len() < MIN_DAEMON_ATTESTATION_KEY_BYTES
        {
            return Err(unverifiable_daemon_space(
                connection,
                "pinned key identifier does not match daemon connection",
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for PinnedDaemonVerifierV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PinnedDaemonVerifierV1")
            .field("key_id_fingerprint", &label_fingerprint(&self.key_id))
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

impl Drop for PinnedDaemonVerifierV1 {
    fn drop(&mut self) {
        self.secret_key.fill(0);
    }
}

/// Explicit wrapper around unverified transient daemon vectors.
///
/// This type deliberately carries no embedding identity and cannot implement
/// [`SyncEmbed`].
pub struct AssumedDaemonEmbeddingBatchV1 {
    vectors: Vec<Vec<f32>>,
}

impl AssumedDaemonEmbeddingBatchV1 {
    /// Explicit trust label.
    #[must_use]
    pub const fn trust_level(&self) -> DaemonTrustLevelV1 {
        DaemonTrustLevelV1::AssumedRemote
    }

    /// Borrow transient vectors.
    #[must_use]
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    /// Consume the transient wrapper.
    #[must_use]
    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.vectors
    }
}

impl std::fmt::Debug for AssumedDaemonEmbeddingBatchV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AssumedDaemonEmbeddingBatchV1")
            .field("trust_level", &DaemonTrustLevelV1::AssumedRemote)
            .field("vector_count", &self.vectors.len())
            .field(
                "vector_dimension",
                &self.vectors.first().map_or(0, Vec::len),
            )
            .field("vectors", &"<redacted>")
            .finish()
    }
}

/// Explicit transient-only access to a raw daemon client.
///
/// It intentionally does not implement [`SyncEmbed`], expose an immutable
/// identity, or compose with verified cache/index APIs.
pub struct AssumedDaemonClient {
    daemon: Arc<dyn DaemonClient>,
}

impl AssumedDaemonClient {
    #[must_use]
    pub fn new(daemon: Arc<dyn DaemonClient>) -> Self {
        Self { daemon }
    }

    /// Embed one input for transient exploration.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or a redacted embedding failure.
    pub fn embed_transient(&self, text: &str) -> SearchResult<AssumedDaemonEmbeddingBatchV1> {
        let request_id = next_request_id();
        let vector = self
            .daemon
            .embed(text, &request_id)
            .map_err(|error| map_assumed_daemon_error(&error))?;
        Ok(AssumedDaemonEmbeddingBatchV1 {
            vectors: vec![vector],
        })
    }

    /// Embed an ordered batch for transient exploration.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or a redacted embedding failure.
    pub fn embed_batch_transient(
        &self,
        texts: &[&str],
    ) -> SearchResult<AssumedDaemonEmbeddingBatchV1> {
        let request_id = next_request_id();
        let vectors = self
            .daemon
            .embed_batch(texts, &request_id)
            .map_err(|error| map_assumed_daemon_error(&error))?;
        Ok(AssumedDaemonEmbeddingBatchV1 { vectors })
    }
}

impl std::fmt::Debug for AssumedDaemonClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AssumedDaemonClient")
            .field("trust_level", &DaemonTrustLevelV1::AssumedRemote)
            .field("daemon", &"<redacted>")
            .finish()
    }
}

/// No-op daemon client used when daemon config is missing.
pub struct NoopDaemonClient {
    id: String,
}

impl NoopDaemonClient {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl DaemonClient for NoopDaemonClient {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_available(&self) -> bool {
        false
    }

    fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "daemon not configured".to_string(),
        ))
    }

    fn embed_batch(
        &self,
        _texts: &[&str],
        _request_id: &str,
    ) -> Result<Vec<Vec<f32>>, DaemonError> {
        Err(DaemonError::Unavailable(
            "daemon not configured".to_string(),
        ))
    }

    fn rerank(
        &self,
        _query: &str,
        _documents: &[&str],
        _request_id: &str,
    ) -> Result<Vec<f32>, DaemonError> {
        Err(DaemonError::Unavailable(
            "daemon not configured".to_string(),
        ))
    }
}

#[derive(Debug)]
struct DaemonState {
    consecutive_failures: u32,
    next_retry_at: Option<Instant>,
}

impl DaemonState {
    const fn new() -> Self {
        Self {
            consecutive_failures: 0,
            next_retry_at: None,
        }
    }

    fn can_attempt(&self, now: Instant) -> bool {
        self.next_retry_at.is_none_or(|at| now >= at)
    }

    const fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.next_retry_at = None;
    }

    fn record_failure(&mut self, config: &DaemonRetryConfig, err: &DaemonError) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let retry_after = match err {
            DaemonError::Overloaded { retry_after, .. } => *retry_after,
            _ => None,
        };
        let backoff = config.backoff_for_attempt(self.consecutive_failures, retry_after);
        self.next_retry_at = Some(Instant::now() + backoff);
    }
}

#[derive(Debug)]
struct DaemonFailure {
    error: DaemonError,
    attempts: u32,
    backoff: bool,
}

#[derive(Debug)]
enum DaemonEmbeddingAttemptError {
    Transport(DaemonError),
    Unverifiable,
    Cancelled,
}

#[derive(Debug)]
struct DaemonEmbeddingFailure {
    error: DaemonEmbeddingAttemptError,
    attempts: u32,
    backoff: bool,
}

fn lock_state(state: &Mutex<DaemonState>) -> std::sync::MutexGuard<'_, DaemonState> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Embedder wrapper that uses the daemon when available and falls back to a local embedder.
pub struct DaemonFallbackEmbedder {
    daemon: Arc<dyn DaemonClient>,
    fallback: Option<Arc<dyn SyncEmbed>>,
    expected_connection: DaemonConnectionIdentityV1,
    verifier: PinnedDaemonVerifierV1,
    model_id: String,
    dimension: usize,
    semantic: bool,
    config: DaemonRetryConfig,
    state: Mutex<DaemonState>,
}

impl DaemonFallbackEmbedder {
    /// Legacy caller-supplied identity cannot authenticate daemon responses.
    ///
    /// # Errors
    ///
    /// Always returns [`SearchError::UnverifiableRemoteSpace`]. Use
    /// [`Self::new_verified`] with a pinned authority, or
    /// [`AssumedDaemonClient`] for explicit transient exploration.
    pub fn new(
        daemon: Arc<dyn DaemonClient>,
        fallback: Arc<dyn SyncEmbed>,
        _config: DaemonRetryConfig,
    ) -> SearchResult<Self> {
        drop(daemon);
        drop(fallback);
        Err(SearchError::UnverifiableRemoteSpace {
            producer: "<redacted-daemon-producer>".to_owned(),
            reason: "caller-supplied daemon identity is not producer proof".to_owned(),
        })
    }

    /// Construct a verified daemon embedder after authenticating handshake and
    /// health boundaries.
    ///
    /// A local fallback is optional. When present, its complete identity must
    /// exactly equal the daemon identity because one `SyncEmbed` object cannot
    /// truthfully expose two producer identities.
    ///
    /// # Errors
    ///
    /// Returns a typed cancellation or `UnverifiableRemoteSpace` for invalid
    /// config, missing proof, identity/key/generation drift, or a mismatched
    /// local fallback.
    pub fn new_verified(
        daemon: Arc<dyn DaemonClient>,
        fallback: Option<Arc<dyn SyncEmbed>>,
        config: DaemonRetryConfig,
        expected_connection: DaemonConnectionIdentityV1,
        verifier: PinnedDaemonVerifierV1,
    ) -> SearchResult<Self> {
        expected_connection
            .validate()
            .map_err(|_| unverifiable_daemon_space(&expected_connection, "invalid connection"))?;
        verifier.validate_for(&expected_connection)?;
        if config.max_attempts == 0 {
            return Err(unverifiable_daemon_space(
                &expected_connection,
                "daemon retry attempts must be non-zero",
            ));
        }
        if let Some(local) = fallback.as_deref() {
            Self::validate_local_fallback(local, &expected_connection)?;
        }
        let model_id = expected_connection
            .embedding_identity
            .space
            .logical_model_id
            .clone();
        let dimension = usize::try_from(expected_connection.embedding_identity.space.dimension)
            .map_err(|_| {
                unverifiable_daemon_space(
                    &expected_connection,
                    "daemon dimension does not fit usize",
                )
            })?;
        let semantic = matches!(
            expected_connection.embedding_identity.space.kind,
            EmbeddingSpaceKindV1::Semantic
        );
        let embedder = Self {
            daemon,
            fallback,
            expected_connection,
            verifier,
            model_id,
            dimension,
            semantic,
            config,
            state: Mutex::new(DaemonState::new()),
        };
        embedder.verify_handshake_and_health()?;
        Ok(embedder)
    }

    /// Construct a verified daemon-only embedder with default retry policy.
    ///
    /// # Errors
    ///
    /// Returns a typed proof or cancellation error from [`Self::new_verified`].
    pub fn with_verified_defaults(
        daemon: Arc<dyn DaemonClient>,
        expected_connection: DaemonConnectionIdentityV1,
        verifier: PinnedDaemonVerifierV1,
    ) -> SearchResult<Self> {
        Self::new_verified(
            daemon,
            None,
            DaemonRetryConfig::default(),
            expected_connection,
            verifier,
        )
    }

    /// Explicit trust label for verified composition.
    #[must_use]
    pub const fn trust_level(&self) -> DaemonTrustLevelV1 {
        DaemonTrustLevelV1::VerifiedRemote
    }

    /// Pinned connection identity authenticated at construction and on every
    /// accepted response.
    #[must_use]
    pub const fn connection_identity(&self) -> &DaemonConnectionIdentityV1 {
        &self.expected_connection
    }

    fn validate_local_fallback(
        fallback: &dyn SyncEmbed,
        expected: &DaemonConnectionIdentityV1,
    ) -> SearchResult<()> {
        let local_identity = fallback
            .identity()
            .map_err(|_| unverifiable_daemon_space(expected, "fallback has no identity"))?;
        local_identity.validate().map_err(|_| {
            unverifiable_daemon_space(expected, "fallback identity failed validation")
        })?;
        let expected_dimension = usize::try_from(expected.embedding_identity.space.dimension)
            .map_err(|_| {
                unverifiable_daemon_space(expected, "daemon dimension does not fit usize")
            })?;
        let expected_semantic = matches!(
            expected.embedding_identity.space.kind,
            EmbeddingSpaceKindV1::Semantic
        );
        if local_identity != &expected.embedding_identity
            || fallback.dimension() != expected_dimension
            || fallback.is_semantic() != expected_semantic
            || fallback.category() != expected.model_category
        {
            return Err(unverifiable_daemon_space(
                expected,
                "fallback metadata or identity differs from daemon identity",
            ));
        }
        Ok(())
    }

    const fn should_retry(err: &DaemonError) -> bool {
        !matches!(
            err,
            DaemonError::InvalidInput(_)
                | DaemonError::Overloaded { .. }
                | DaemonError::Cancelled
                | DaemonError::UnverifiableRemoteSpace
        )
    }

    const fn fallback_reason(err: &DaemonError, backoff_active: bool) -> &'static str {
        if backoff_active {
            return "backoff";
        }
        match err {
            DaemonError::Unavailable(_) => "unavailable",
            DaemonError::Timeout(_) => "timeout",
            DaemonError::Overloaded { .. } => "overloaded",
            DaemonError::Failed(_) => "error",
            DaemonError::InvalidInput(_) => "invalid",
            DaemonError::Cancelled => "cancelled",
            DaemonError::UnverifiableRemoteSpace => "unverifiable",
        }
    }

    fn log_fallback(&self, request_id: &str, retries: u32, reason: &str) {
        warn!(
            daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
            daemon_protocol_hash = label_fingerprint(&self.expected_connection.protocol_revision),
            daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
            daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
            daemon_generation = self.expected_connection.generation,
            request_id = request_id,
            retry_count = retries,
            fallback_reason = reason,
            "Daemon embed failed; using local embedder"
        );
    }

    fn log_attestation_rejection(&self, request_id: &str, reason: &str) {
        warn!(
            daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
            daemon_protocol_hash = label_fingerprint(&self.expected_connection.protocol_revision),
            daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
            daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
            daemon_generation = self.expected_connection.generation,
            request_id,
            rejection = reason,
            "Rejected unverifiable daemon embedding response"
        );
    }

    fn fresh_challenge(
        &self,
        operation: DaemonOperationV1,
        inputs: &[&str],
    ) -> Result<DaemonChallengeV1, DaemonEmbeddingAttemptError> {
        let nonce = fresh_daemon_nonce().map_err(|_| DaemonEmbeddingAttemptError::Unverifiable)?;
        DaemonChallengeV1::for_inputs(nonce, operation, inputs, &self.expected_connection)
            .map_err(|_| DaemonEmbeddingAttemptError::Unverifiable)
    }

    fn verify_attestation(
        &self,
        attestation: &DaemonEmbeddingAttestationV1,
        challenge: &DaemonChallengeV1,
        vectors: &[Vec<f32>],
    ) -> Result<(), DaemonEmbeddingAttemptError> {
        attestation
            .validate_against(challenge, &self.expected_connection, vectors)
            .and_then(|()| attestation.authenticate_hmac_sha256(&self.verifier.secret_key))
            .map_err(|_| DaemonEmbeddingAttemptError::Unverifiable)
    }

    fn verify_control_boundary(
        &self,
        operation: DaemonOperationV1,
    ) -> Result<(), DaemonEmbeddingAttemptError> {
        let challenge = self.fresh_challenge(operation, &[])?;
        let response = match operation {
            DaemonOperationV1::Handshake => self.daemon.handshake_attested(&challenge),
            DaemonOperationV1::Health => self.daemon.health_attested(&challenge),
            DaemonOperationV1::Embed | DaemonOperationV1::EmbedBatch => {
                return Err(DaemonEmbeddingAttemptError::Unverifiable);
            }
        }
        .map_err(classify_daemon_embedding_error)?;
        self.verify_attestation(&response, &challenge, &[])
    }

    fn verify_handshake_and_health(&self) -> SearchResult<()> {
        if !self.daemon.is_available() {
            return Err(unverifiable_daemon_space(
                &self.expected_connection,
                "daemon unavailable before authenticated handshake",
            ));
        }
        for operation in [DaemonOperationV1::Handshake, DaemonOperationV1::Health] {
            self.verify_control_boundary(operation).map_err(|error| {
                self.embedding_attempt_error_to_search(&error, "daemon.control")
            })?;
        }
        Ok(())
    }

    fn verify_embedding_response(
        &self,
        response: AttestedDaemonEmbeddingResponseV1,
        challenge: &DaemonChallengeV1,
    ) -> Result<Vec<Vec<f32>>, DaemonEmbeddingAttemptError> {
        self.verify_attestation(&response.attestation, challenge, &response.vectors)?;
        Ok(response.vectors)
    }

    fn wait_for_backoff(&self) {
        let next_retry_at = lock_state(&self.state).next_retry_at;
        if let Some(next_retry_at) = next_retry_at {
            let sleep_for = next_retry_at.saturating_duration_since(Instant::now());
            if !sleep_for.is_zero() {
                std::thread::sleep(sleep_for);
            }
        }
    }

    fn try_embed(&self, request_id: &str, text: &str) -> Result<Vec<f32>, DaemonEmbeddingFailure> {
        if !self.daemon.is_available() {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "daemon not available".to_owned(),
                )),
                attempts: 0,
                backoff: false,
            });
        }
        if !lock_state(&self.state).can_attempt(Instant::now()) {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "backoff active".to_owned(),
                )),
                attempts: 0,
                backoff: true,
            });
        }

        let mut attempts = 0;
        let mut last_error = DaemonError::Unavailable("daemon embed failed".to_owned());
        while attempts < self.config.max_attempts {
            attempts += 1;
            let challenge = self
                .fresh_challenge(DaemonOperationV1::Embed, &[text])
                .map_err(|error| DaemonEmbeddingFailure {
                    error,
                    attempts,
                    backoff: false,
                })?;
            debug!(
                daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                daemon_protocol_hash =
                    label_fingerprint(&self.expected_connection.protocol_revision),
                daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
                daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
                daemon_generation = self.expected_connection.generation,
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                "Attempting authenticated daemon embed"
            );
            match self.daemon.embed_attested(text, &challenge) {
                Ok(response) => {
                    let mut vectors = self
                        .verify_embedding_response(response, &challenge)
                        .map_err(|error| DaemonEmbeddingFailure {
                            error,
                            attempts,
                            backoff: false,
                        })?;
                    let vector = vectors.pop().ok_or(DaemonEmbeddingFailure {
                        error: DaemonEmbeddingAttemptError::Unverifiable,
                        attempts,
                        backoff: false,
                    })?;
                    lock_state(&self.state).record_success();
                    return Ok(vector);
                }
                Err(error) => {
                    let classified = classify_daemon_embedding_error(error);
                    let error = match classified {
                        DaemonEmbeddingAttemptError::Transport(error) => error,
                        other => {
                            return Err(DaemonEmbeddingFailure {
                                error: other,
                                attempts,
                                backoff: false,
                            });
                        }
                    };
                    let should_retry = Self::should_retry(&error);
                    let should_backoff = !matches!(error, DaemonError::InvalidInput(_));
                    if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &error);
                    }
                    debug!(
                        daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = Self::fallback_reason(&error, false),
                        "Authenticated daemon embed failed"
                    );
                    last_error = error;
                    if !should_retry || attempts >= self.config.max_attempts {
                        break;
                    }
                    if should_backoff {
                        self.wait_for_backoff();
                    }
                    for operation in [DaemonOperationV1::Handshake, DaemonOperationV1::Health] {
                        self.verify_control_boundary(operation).map_err(|error| {
                            DaemonEmbeddingFailure {
                                error,
                                attempts,
                                backoff: false,
                            }
                        })?;
                    }
                }
            }
        }

        Err(DaemonEmbeddingFailure {
            error: DaemonEmbeddingAttemptError::Transport(last_error),
            attempts,
            backoff: false,
        })
    }

    fn try_embed_batch(
        &self,
        request_id: &str,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, DaemonEmbeddingFailure> {
        if !self.daemon.is_available() {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "daemon not available".to_owned(),
                )),
                attempts: 0,
                backoff: false,
            });
        }
        if !lock_state(&self.state).can_attempt(Instant::now()) {
            return Err(DaemonEmbeddingFailure {
                error: DaemonEmbeddingAttemptError::Transport(DaemonError::Unavailable(
                    "backoff active".to_owned(),
                )),
                attempts: 0,
                backoff: true,
            });
        }

        let mut attempts = 0;
        let mut last_error = DaemonError::Unavailable("daemon embed batch failed".to_owned());
        while attempts < self.config.max_attempts {
            attempts += 1;
            let challenge = self
                .fresh_challenge(DaemonOperationV1::EmbedBatch, texts)
                .map_err(|error| DaemonEmbeddingFailure {
                    error,
                    attempts,
                    backoff: false,
                })?;
            debug!(
                daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                daemon_protocol_hash =
                    label_fingerprint(&self.expected_connection.protocol_revision),
                daemon_key_hash = label_fingerprint(&self.expected_connection.key_id),
                daemon_identity_hash = self.expected_connection.embedding_identity.fingerprint(),
                daemon_generation = self.expected_connection.generation,
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                input_count = texts.len(),
                "Attempting authenticated daemon embed batch"
            );
            match self.daemon.embed_batch_attested(texts, &challenge) {
                Ok(response) => {
                    let vectors = self
                        .verify_embedding_response(response, &challenge)
                        .map_err(|error| DaemonEmbeddingFailure {
                            error,
                            attempts,
                            backoff: false,
                        })?;
                    lock_state(&self.state).record_success();
                    return Ok(vectors);
                }
                Err(error) => {
                    let classified = classify_daemon_embedding_error(error);
                    let error = match classified {
                        DaemonEmbeddingAttemptError::Transport(error) => error,
                        other => {
                            return Err(DaemonEmbeddingFailure {
                                error: other,
                                attempts,
                                backoff: false,
                            });
                        }
                    };
                    let should_retry = Self::should_retry(&error);
                    let should_backoff = !matches!(error, DaemonError::InvalidInput(_));
                    if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &error);
                    }
                    debug!(
                        daemon_endpoint_hash = self.expected_connection.endpoint_fingerprint,
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = Self::fallback_reason(&error, false),
                        "Authenticated daemon embed batch failed"
                    );
                    last_error = error;
                    if !should_retry || attempts >= self.config.max_attempts {
                        break;
                    }
                    if should_backoff {
                        self.wait_for_backoff();
                    }
                    for operation in [DaemonOperationV1::Handshake, DaemonOperationV1::Health] {
                        self.verify_control_boundary(operation).map_err(|error| {
                            DaemonEmbeddingFailure {
                                error,
                                attempts,
                                backoff: false,
                            }
                        })?;
                    }
                }
            }
        }

        Err(DaemonEmbeddingFailure {
            error: DaemonEmbeddingAttemptError::Transport(last_error),
            attempts,
            backoff: false,
        })
    }

    fn embedding_attempt_error_to_search(
        &self,
        error: &DaemonEmbeddingAttemptError,
        phase: &str,
    ) -> SearchError {
        match error {
            DaemonEmbeddingAttemptError::Cancelled => SearchError::Cancelled {
                phase: phase.to_owned(),
                reason: "daemon operation cancelled".to_owned(),
            },
            DaemonEmbeddingAttemptError::Unverifiable
            | DaemonEmbeddingAttemptError::Transport(_) => unverifiable_daemon_space(
                &self.expected_connection,
                "daemon could not authenticate the required protocol boundary",
            ),
        }
    }
}

impl SyncEmbed for DaemonFallbackEmbedder {
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        let request_id = next_request_id();
        match self.try_embed(&request_id, text) {
            Ok(vector) => Ok(vector),
            Err(failure) => {
                let retries = failure.attempts.saturating_sub(1);
                match failure.error {
                    DaemonEmbeddingAttemptError::Unverifiable => {
                        self.log_attestation_rejection(&request_id, "attestation");
                        Err(unverifiable_daemon_space(
                            &self.expected_connection,
                            "daemon response failed authenticated identity validation",
                        ))
                    }
                    DaemonEmbeddingAttemptError::Cancelled => Err(SearchError::Cancelled {
                        phase: "daemon.embedding".to_owned(),
                        reason: "daemon operation cancelled".to_owned(),
                    }),
                    DaemonEmbeddingAttemptError::Transport(error) => {
                        let reason = Self::fallback_reason(&error, failure.backoff);
                        self.fallback.as_ref().map_or_else(
                            || Err(map_verified_daemon_transport_error(&self.model_id, &error)),
                            |fallback| {
                                self.log_fallback(&request_id, retries, reason);
                                fallback.embed_sync(text)
                            },
                        )
                    }
                }
            }
        }
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let request_id = next_request_id();
        match self.try_embed_batch(&request_id, texts) {
            Ok(vectors) => Ok(vectors),
            Err(failure) => {
                let retries = failure.attempts.saturating_sub(1);
                match failure.error {
                    DaemonEmbeddingAttemptError::Unverifiable => {
                        self.log_attestation_rejection(&request_id, "attestation");
                        Err(unverifiable_daemon_space(
                            &self.expected_connection,
                            "daemon response failed authenticated identity validation",
                        ))
                    }
                    DaemonEmbeddingAttemptError::Cancelled => Err(SearchError::Cancelled {
                        phase: "daemon.embedding_batch".to_owned(),
                        reason: "daemon operation cancelled".to_owned(),
                    }),
                    DaemonEmbeddingAttemptError::Transport(error) => {
                        let reason = Self::fallback_reason(&error, failure.backoff);
                        self.fallback.as_ref().map_or_else(
                            || Err(map_verified_daemon_transport_error(&self.model_id, &error)),
                            |fallback| {
                                self.log_fallback(&request_id, retries, reason);
                                fallback.embed_batch_sync(texts)
                            },
                        )
                    }
                }
            }
        }
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.expected_connection.embedding_identity)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.model_id
    }

    fn model_name(&self) -> &str {
        &self.model_id
    }

    fn is_ready(&self) -> bool {
        self.verify_handshake_and_health().is_ok()
            || self
                .fallback
                .as_ref()
                .is_some_and(|fallback| fallback.is_ready())
    }

    fn is_semantic(&self) -> bool {
        self.semantic
    }

    fn category(&self) -> ModelCategory {
        self.expected_connection.model_category
    }
}

fn classify_daemon_embedding_error(error: DaemonError) -> DaemonEmbeddingAttemptError {
    match error {
        DaemonError::Cancelled => DaemonEmbeddingAttemptError::Cancelled,
        DaemonError::UnverifiableRemoteSpace => DaemonEmbeddingAttemptError::Unverifiable,
        transport => DaemonEmbeddingAttemptError::Transport(transport),
    }
}

fn fresh_daemon_nonce() -> std::io::Result<String> {
    let mut entropy = [0_u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut entropy)?;
    Ok(encode_lower_hex(&entropy))
}

fn map_assumed_daemon_error(error: &DaemonError) -> SearchError {
    match error {
        DaemonError::Cancelled => SearchError::Cancelled {
            phase: "daemon.assumed_transient".to_owned(),
            reason: "daemon operation cancelled".to_owned(),
        },
        DaemonError::UnverifiableRemoteSpace => SearchError::UnverifiableRemoteSpace {
            producer: "<redacted-daemon-producer>".to_owned(),
            reason: "daemon rejected the transient request".to_owned(),
        },
        _ => SearchError::EmbeddingFailed {
            model: "<assumed-daemon>".to_owned(),
            source: std::io::Error::other("assumed daemon transport failed").into(),
        },
    }
}

fn map_verified_daemon_transport_error(model_id: &str, error: &DaemonError) -> SearchError {
    match error {
        DaemonError::Cancelled => SearchError::Cancelled {
            phase: "daemon.embedding".to_owned(),
            reason: "daemon operation cancelled".to_owned(),
        },
        DaemonError::UnverifiableRemoteSpace => SearchError::UnverifiableRemoteSpace {
            producer: "<redacted-daemon-producer>".to_owned(),
            reason: "daemon embedding space is unverifiable".to_owned(),
        },
        _ => SearchError::EmbeddingFailed {
            model: label_fingerprint(model_id),
            source: std::io::Error::other("verified daemon transport failed").into(),
        },
    }
}

fn unverifiable_daemon_space(connection: &DaemonConnectionIdentityV1, reason: &str) -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: connection.endpoint_fingerprint.clone(),
        reason: if reason.len() <= 256 && !reason.chars().any(char::is_control) {
            reason.to_owned()
        } else {
            "daemon identity validation failed".to_owned()
        },
    }
}

fn is_bounded_daemon_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/' | b':')
        })
}

fn label_fingerprint(value: &str) -> String {
    encode_lower_hex(&Sha256::digest(value.as_bytes()))
}

fn encode_lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

/// Reranker wrapper that uses the daemon when available and falls back to a local reranker.
pub struct DaemonFallbackReranker {
    daemon: Arc<dyn DaemonClient>,
    fallback: Option<Arc<dyn SyncRerank>>,
    config: DaemonRetryConfig,
    state: Mutex<DaemonState>,
}

impl DaemonFallbackReranker {
    #[must_use]
    pub fn new(
        daemon: Arc<dyn DaemonClient>,
        fallback: Option<Arc<dyn SyncRerank>>,
        config: DaemonRetryConfig,
    ) -> Self {
        Self {
            daemon,
            fallback,
            config,
            state: Mutex::new(DaemonState::new()),
        }
    }

    fn log_fallback(&self, request_id: &str, retries: u32, reason: &str) {
        warn!(
            daemon_id_hash = label_fingerprint(self.daemon.id()),
            request_id,
            retry_count = retries,
            fallback_reason = reason,
            "Daemon rerank failed; using local reranker"
        );
    }

    fn try_rerank(
        &self,
        request_id: &str,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<f32>, DaemonFailure> {
        if !self.daemon.is_available() {
            return Err(DaemonFailure {
                error: DaemonError::Unavailable("daemon not available".to_string()),
                attempts: 0,
                backoff: false,
            });
        }
        let now = Instant::now();
        if !lock_state(&self.state).can_attempt(now) {
            return Err(DaemonFailure {
                error: DaemonError::Unavailable("backoff active".to_string()),
                attempts: 0,
                backoff: true,
            });
        }

        let mut attempts = 0;
        let mut last_err: Option<DaemonError> = None;

        while attempts < self.config.max_attempts {
            attempts += 1;
            debug!(
                daemon_id_hash = label_fingerprint(self.daemon.id()),
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                "Attempting daemon rerank"
            );
            match self.daemon.rerank(query, documents, request_id) {
                Ok(scores) => {
                    lock_state(&self.state).record_success();
                    return Ok(scores);
                }
                Err(err) => {
                    let should_retry = DaemonFallbackEmbedder::should_retry(&err);
                    let should_backoff = !matches!(err, DaemonError::InvalidInput(_));
                    let backoff = if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &err);
                        true
                    } else {
                        false
                    };

                    debug!(
                        daemon_id_hash = label_fingerprint(self.daemon.id()),
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = DaemonFallbackEmbedder::fallback_reason(&err, false),
                        "Daemon rerank failed"
                    );

                    last_err = Some(err);
                    if !should_retry || attempts >= self.config.max_attempts {
                        break;
                    }

                    if backoff && let Some(next_retry_at) = lock_state(&self.state).next_retry_at {
                        let sleep_for = next_retry_at.saturating_duration_since(Instant::now());
                        if !sleep_for.is_zero() {
                            std::thread::sleep(sleep_for);
                        }
                    }
                }
            }
        }

        Err(DaemonFailure {
            error: last_err
                .unwrap_or_else(|| DaemonError::Unavailable("daemon rerank failed".to_string())),
            attempts,
            backoff: false,
        })
    }

    /// Reject a structurally invalid daemon rerank response before any score
    /// is attributed to a document.
    ///
    /// `Vec<f32>` carries no length invariant on the wire: a daemon that caps
    /// its batch or truncates on a partial read returns a score prefix, and a
    /// positional zip would assign `0.0` to every document past it — sinking
    /// them to the bottom of a "successful" rerank with no error. Non-finite
    /// scores poison downstream ordering the same way. Both are daemon
    /// contract violations, so they take the existing failed-call fallback
    /// path as `InvalidInput`.
    fn validate_rerank_shape(scores: &[f32], expected: usize) -> Result<(), DaemonError> {
        if scores.len() != expected {
            return Err(DaemonError::InvalidInput(format!(
                "daemon rerank returned {} scores for {expected} documents",
                scores.len()
            )));
        }
        if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
            return Err(DaemonError::InvalidInput(format!(
                "daemon rerank returned a non-finite score at index {index}"
            )));
        }
        Ok(())
    }
}

impl SyncRerank for DaemonFallbackReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        let request_id = next_request_id();

        let outcome = self
            .try_rerank(&request_id, query, &texts)
            .and_then(
                |scores| match Self::validate_rerank_shape(&scores, documents.len()) {
                    Ok(()) => Ok(scores),
                    Err(error) => Err(DaemonFailure {
                        error,
                        attempts: 1,
                        backoff: false,
                    }),
                },
            );
        match outcome {
            Ok(scores) => Ok(documents
                .iter()
                .zip(scores)
                .enumerate()
                .map(|(index, (doc, score))| RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score,
                    original_rank: index,
                    raw_logit: None,
                })
                .collect()),
            Err(failure) => {
                if matches!(&failure.error, DaemonError::Cancelled) {
                    return Err(SearchError::Cancelled {
                        phase: "daemon.rerank".to_owned(),
                        reason: "daemon operation cancelled".to_owned(),
                    });
                }
                let retries = failure.attempts.saturating_sub(1);
                let reason =
                    DaemonFallbackEmbedder::fallback_reason(&failure.error, failure.backoff);
                self.log_fallback(&request_id, retries, reason);
                self.fallback.as_ref().map_or_else(
                    || {
                        Err(SearchError::RerankFailed {
                            model: "daemon-reranker".to_string(),
                            source: std::io::Error::other("no local reranker available").into(),
                        })
                    },
                    |reranker| reranker.rerank_sync(query, documents),
                )
            }
        }
    }

    fn id(&self) -> &str {
        self.fallback
            .as_ref()
            .map_or("daemon-reranker", |fallback| fallback.id())
    }

    fn model_name(&self) -> &str {
        self.fallback
            .as_ref()
            .map_or("daemon-reranker", |fallback| fallback.model_name())
    }

    fn max_length(&self) -> usize {
        self.fallback
            .as_ref()
            .map_or(512, |fallback| fallback.max_length())
    }

    fn is_available(&self) -> bool {
        self.daemon.is_available()
            || self
                .fallback
                .as_ref()
                .is_some_and(|reranker| reranker.is_available())
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::collections::HashSet;
    use std::io::{Read, Write};
    use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use serde::{Deserialize, Serialize};

    use super::*;

    struct ConstEmbedder {
        id: &'static str,
        model_name: &'static str,
        dim: usize,
        value: f32,
        semantic: bool,
        category: ModelCategory,
        identity: EmbeddingIdentityBundleV1,
    }

    impl SyncEmbed for ConstEmbedder {
        fn embed_sync(&self, _text: &str) -> SearchResult<Vec<f32>> {
            Ok(vec![self.value; self.dim])
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.model_name
        }

        fn is_semantic(&self) -> bool {
            self.semantic
        }

        fn category(&self) -> ModelCategory {
            self.category
        }
    }

    struct ConstReranker {
        id: &'static str,
    }

    impl SyncRerank for ConstReranker {
        fn rerank_sync(
            &self,
            _query: &str,
            documents: &[RerankDocument],
        ) -> SearchResult<Vec<RerankScore>> {
            Ok(documents
                .iter()
                .enumerate()
                .map(|(idx, doc)| RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score: 10.0 - idx as f32,
                    original_rank: idx,
                    raw_logit: None,
                })
                .collect())
        }

        fn id(&self) -> &str {
            self.id
        }

        fn model_name(&self) -> &str {
            self.id
        }
    }

    /// Daemon fixture whose rerank call succeeds with a caller-chosen score
    /// vector, so tests can exercise arity/finiteness violations on the Ok
    /// path.
    struct ShapedRerankDaemon {
        scores: Vec<f32>,
        raw_calls: AtomicUsize,
    }

    impl ShapedRerankDaemon {
        fn new(scores: Vec<f32>) -> Self {
            Self {
                scores,
                raw_calls: AtomicUsize::new(0),
            }
        }
    }

    impl DaemonClient for ShapedRerankDaemon {
        fn id(&self) -> &str {
            "shaped-rerank-daemon"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
            Err(DaemonError::Unavailable(
                "shaped rerank fixture has no embedder".to_string(),
            ))
        }

        fn embed_batch(
            &self,
            _texts: &[&str],
            _request_id: &str,
        ) -> Result<Vec<Vec<f32>>, DaemonError> {
            Err(DaemonError::Unavailable(
                "shaped rerank fixture has no embedder".to_string(),
            ))
        }

        fn rerank(
            &self,
            _query: &str,
            _documents: &[&str],
            _request_id: &str,
        ) -> Result<Vec<f32>, DaemonError> {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.scores.clone())
        }
    }

    #[derive(Clone, Copy)]
    enum FailureMode {
        Timeout,
        Overloaded { retry_after: Duration },
        Failed,
        InvalidInput,
    }

    impl FailureMode {
        fn error(&self) -> DaemonError {
            match self {
                Self::Timeout => DaemonError::Timeout("daemon timeout".to_string()),
                Self::Overloaded { retry_after } => DaemonError::Overloaded {
                    retry_after: Some(*retry_after),
                    message: "queue full".to_string(),
                },
                Self::Failed => DaemonError::Failed("daemon failed".to_string()),
                Self::InvalidInput => DaemonError::InvalidInput("invalid input".to_string()),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ResponseMutation {
        None,
        MissingAttestation,
        WrongKey,
        PayloadTamper,
        ResponseReorder,
        SpaceDrift,
        EndpointDrift,
        GenerationDrift,
        KeyIdDrift,
        UnknownSchema,
        ReplayNonce,
        Cancelled,
    }

    const TEST_KEY: &[u8] = b"0123456789abcdef0123456789abcdef";
    const WRONG_KEY: &[u8] = b"abcdef0123456789abcdef0123456789";

    struct FixtureDaemon {
        attested_calls: AtomicUsize,
        raw_calls: AtomicUsize,
        control_calls: AtomicUsize,
        fail_first: usize,
        mode: FailureMode,
        available: AtomicBool,
        embed_value: f32,
        connection: DaemonConnectionIdentityV1,
        key: Vec<u8>,
        mutation: Mutex<ResponseMutation>,
        challenges: Mutex<Vec<DaemonChallengeV1>>,
    }

    impl FixtureDaemon {
        fn new(
            fail_first: usize,
            mode: FailureMode,
            available: bool,
            embed_value: f32,
            connection: DaemonConnectionIdentityV1,
        ) -> Self {
            Self {
                attested_calls: AtomicUsize::new(0),
                raw_calls: AtomicUsize::new(0),
                control_calls: AtomicUsize::new(0),
                fail_first,
                mode,
                available: AtomicBool::new(available),
                embed_value,
                connection,
                key: TEST_KEY.to_vec(),
                mutation: Mutex::new(ResponseMutation::None),
                challenges: Mutex::new(Vec::new()),
            }
        }

        fn exact(connection: DaemonConnectionIdentityV1, embed_value: f32) -> Self {
            Self::new(0, FailureMode::Failed, true, embed_value, connection)
        }

        fn set_mutation(&self, mutation: ResponseMutation) {
            *self
                .mutation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = mutation;
        }

        fn set_available(&self, available: bool) {
            self.available.store(available, Ordering::Relaxed);
        }

        fn record_challenge(&self, challenge: &DaemonChallengeV1) {
            self.challenges
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(challenge.clone());
        }

        fn signed_response(
            &self,
            challenge: &DaemonChallengeV1,
            vectors: Vec<Vec<f32>>,
        ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
            self.record_challenge(challenge);
            let mutation = *self
                .mutation
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match mutation {
                ResponseMutation::MissingAttestation => {
                    return Err(DaemonError::UnverifiableRemoteSpace);
                }
                ResponseMutation::Cancelled => return Err(DaemonError::Cancelled),
                _ => {}
            }
            let signing_key = if mutation == ResponseMutation::WrongKey {
                WRONG_KEY
            } else {
                &self.key
            };
            let mut response = AttestedDaemonEmbeddingResponseV1::signed(
                challenge.clone(),
                self.connection.clone(),
                vectors,
                signing_key,
            )?;
            match mutation {
                ResponseMutation::PayloadTamper => response.vectors[0][0] += 1.0,
                ResponseMutation::ResponseReorder => response.vectors.swap(0, 1),
                ResponseMutation::SpaceDrift => {
                    let mut drifted = self.connection.embedding_identity.clone();
                    drifted.space.tokenizer_fingerprint = "ab".repeat(32);
                    drifted.producer.space_fingerprint = drifted.space.fingerprint();
                    response.attestation.connection.embedding_identity = drifted;
                }
                ResponseMutation::EndpointDrift => {
                    response.attestation.connection.endpoint_fingerprint =
                        frankensearch_core::daemon_endpoint_fingerprint(
                            "unix:/run/other-daemon.sock",
                        );
                }
                ResponseMutation::GenerationDrift => {
                    response.attestation.connection.generation += 1;
                }
                ResponseMutation::KeyIdDrift => {
                    "rotated-key".clone_into(&mut response.attestation.connection.key_id);
                }
                ResponseMutation::UnknownSchema => {
                    response.attestation.schema_version = u16::MAX;
                }
                ResponseMutation::ReplayNonce => {
                    response.attestation.challenge.request_nonce = "11".repeat(32);
                }
                ResponseMutation::None
                | ResponseMutation::MissingAttestation
                | ResponseMutation::WrongKey
                | ResponseMutation::Cancelled => {}
            }
            Ok(response)
        }

        fn control_attestation(
            &self,
            challenge: &DaemonChallengeV1,
        ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
            self.control_calls.fetch_add(1, Ordering::Relaxed);
            self.signed_response(challenge, Vec::new())
                .map(|response| response.attestation)
        }

        fn maybe_fail_attested(&self) -> Result<(), DaemonError> {
            let call = self.attested_calls.fetch_add(1, Ordering::Relaxed);
            if call < self.fail_first {
                Err(self.mode.error())
            } else {
                Ok(())
            }
        }
    }

    impl DaemonClient for FixtureDaemon {
        fn id(&self) -> &str {
            "fixture-daemon"
        }

        fn is_available(&self) -> bool {
            self.available.load(Ordering::Relaxed)
        }

        fn handshake_attested(
            &self,
            challenge: &DaemonChallengeV1,
        ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
            self.control_attestation(challenge)
        }

        fn health_attested(
            &self,
            challenge: &DaemonChallengeV1,
        ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
            self.control_attestation(challenge)
        }

        fn embed_attested(
            &self,
            _text: &str,
            challenge: &DaemonChallengeV1,
        ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
            if let Err(error) = self.maybe_fail_attested() {
                self.record_challenge(challenge);
                return Err(error);
            }
            self.signed_response(challenge, vec![vec![self.embed_value; 4]])
        }

        fn embed_batch_attested(
            &self,
            texts: &[&str],
            challenge: &DaemonChallengeV1,
        ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
            if let Err(error) = self.maybe_fail_attested() {
                self.record_challenge(challenge);
                return Err(error);
            }
            let vectors = (0..texts.len())
                .map(|index| vec![self.embed_value + index as f32; 4])
                .collect();
            self.signed_response(challenge, vectors)
        }

        fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
            let call = self.raw_calls.fetch_add(1, Ordering::Relaxed);
            if call < self.fail_first {
                Err(self.mode.error())
            } else {
                Ok(vec![self.embed_value; 4])
            }
        }

        fn embed_batch(
            &self,
            texts: &[&str],
            _request_id: &str,
        ) -> Result<Vec<Vec<f32>>, DaemonError> {
            let call = self.raw_calls.fetch_add(1, Ordering::Relaxed);
            if call < self.fail_first {
                Err(self.mode.error())
            } else {
                Ok(vec![vec![self.embed_value; 4]; texts.len()])
            }
        }

        fn rerank(
            &self,
            _query: &str,
            documents: &[&str],
            _request_id: &str,
        ) -> Result<Vec<f32>, DaemonError> {
            let call = self.raw_calls.fetch_add(1, Ordering::Relaxed);
            if call < self.fail_first {
                Err(self.mode.error())
            } else {
                Ok((0..documents.len())
                    .map(|idx| (documents.len() - idx) as f32)
                    .collect())
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct WireRequest {
        challenge: DaemonChallengeV1,
        inputs: Vec<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    enum WireResponse {
        Control(DaemonEmbeddingAttestationV1),
        Embedding(AttestedDaemonEmbeddingResponseV1),
    }

    struct TcpDaemonClient {
        address: SocketAddr,
    }

    impl TcpDaemonClient {
        fn round_trip(
            &self,
            challenge: &DaemonChallengeV1,
            inputs: &[&str],
        ) -> Result<WireResponse, DaemonError> {
            let mut stream = TcpStream::connect(self.address)
                .map_err(|_| DaemonError::Failed("tcp connection failed".to_owned()))?;
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .map_err(|_| DaemonError::Failed("tcp timeout setup failed".to_owned()))?;
            let request = WireRequest {
                challenge: challenge.clone(),
                inputs: inputs.iter().map(|input| (*input).to_owned()).collect(),
            };
            let payload = serde_json::to_vec(&request)
                .map_err(|_| DaemonError::Failed("wire request encoding failed".to_owned()))?;
            stream
                .write_all(&payload)
                .map_err(|_| DaemonError::Failed("wire request write failed".to_owned()))?;
            stream
                .shutdown(Shutdown::Write)
                .map_err(|_| DaemonError::Failed("wire request shutdown failed".to_owned()))?;
            let mut response = Vec::new();
            stream
                .read_to_end(&mut response)
                .map_err(|_| DaemonError::Failed("wire response read failed".to_owned()))?;
            serde_json::from_slice(&response)
                .map_err(|_| DaemonError::Failed("wire response decoding failed".to_owned()))
        }
    }

    impl DaemonClient for TcpDaemonClient {
        fn id(&self) -> &str {
            "tcp-test-daemon"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn handshake_attested(
            &self,
            challenge: &DaemonChallengeV1,
        ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
            match self.round_trip(challenge, &[])? {
                WireResponse::Control(attestation) => Ok(attestation),
                WireResponse::Embedding(_) => Err(DaemonError::UnverifiableRemoteSpace),
            }
        }

        fn health_attested(
            &self,
            challenge: &DaemonChallengeV1,
        ) -> Result<DaemonEmbeddingAttestationV1, DaemonError> {
            match self.round_trip(challenge, &[])? {
                WireResponse::Control(attestation) => Ok(attestation),
                WireResponse::Embedding(_) => Err(DaemonError::UnverifiableRemoteSpace),
            }
        }

        fn embed_attested(
            &self,
            text: &str,
            challenge: &DaemonChallengeV1,
        ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
            match self.round_trip(challenge, &[text])? {
                WireResponse::Embedding(response) => Ok(response),
                WireResponse::Control(_) => Err(DaemonError::UnverifiableRemoteSpace),
            }
        }

        fn embed_batch_attested(
            &self,
            texts: &[&str],
            challenge: &DaemonChallengeV1,
        ) -> Result<AttestedDaemonEmbeddingResponseV1, DaemonError> {
            match self.round_trip(challenge, texts)? {
                WireResponse::Embedding(response) => Ok(response),
                WireResponse::Control(_) => Err(DaemonError::UnverifiableRemoteSpace),
            }
        }

        fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
            Err(DaemonError::UnverifiableRemoteSpace)
        }

        fn embed_batch(
            &self,
            _texts: &[&str],
            _request_id: &str,
        ) -> Result<Vec<Vec<f32>>, DaemonError> {
            Err(DaemonError::UnverifiableRemoteSpace)
        }

        fn rerank(
            &self,
            _query: &str,
            _documents: &[&str],
            _request_id: &str,
        ) -> Result<Vec<f32>, DaemonError> {
            Err(DaemonError::Unavailable(
                "test gateway has no reranker".to_owned(),
            ))
        }
    }

    fn spawn_authenticated_tcp_gateway(
        connection: DaemonConnectionIdentityV1,
    ) -> (SocketAddr, thread::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut payload = Vec::new();
                stream.read_to_end(&mut payload).unwrap();
                let request: WireRequest = serde_json::from_slice(&payload).unwrap();
                let input_refs = request
                    .inputs
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let expected = DaemonChallengeV1::for_inputs(
                    request.challenge.request_nonce.clone(),
                    request.challenge.operation,
                    &input_refs,
                    &connection,
                )
                .unwrap();
                assert_eq!(request.challenge, expected);
                let response = match request.challenge.operation {
                    DaemonOperationV1::Handshake | DaemonOperationV1::Health => {
                        let signed = AttestedDaemonEmbeddingResponseV1::signed(
                            request.challenge,
                            connection.clone(),
                            Vec::new(),
                            TEST_KEY,
                        )
                        .unwrap();
                        WireResponse::Control(signed.attestation)
                    }
                    DaemonOperationV1::Embed => {
                        let signed = AttestedDaemonEmbeddingResponseV1::signed(
                            request.challenge,
                            connection.clone(),
                            vec![vec![4.0, 3.0, 2.0, 1.0]],
                            TEST_KEY,
                        )
                        .unwrap();
                        WireResponse::Embedding(signed)
                    }
                    DaemonOperationV1::EmbedBatch => {
                        let vectors = (0..input_refs.len())
                            .map(|index| vec![index as f32 + 1.0; 4])
                            .collect();
                        let signed = AttestedDaemonEmbeddingResponseV1::signed(
                            request.challenge,
                            connection.clone(),
                            vectors,
                            TEST_KEY,
                        )
                        .unwrap();
                        WireResponse::Embedding(signed)
                    }
                };
                stream
                    .write_all(&serde_json::to_vec(&response).unwrap())
                    .unwrap();
            }
        });
        (address, handle)
    }

    fn test_connection(model_id: &str) -> DaemonConnectionIdentityV1 {
        DaemonConnectionIdentityV1 {
            schema_version: frankensearch_core::DAEMON_CONNECTION_IDENTITY_SCHEMA_V1,
            endpoint_fingerprint: frankensearch_core::daemon_endpoint_fingerprint(
                "unix:/run/frankensearch.sock",
            ),
            executable_fingerprint: frankensearch_core::daemon_executable_fingerprint(
                b"fixture-daemon-v1",
            ),
            protocol_revision: "frankensearch-daemon-v1".to_owned(),
            key_id: "fixture-key-v1".to_owned(),
            generation: 9,
            embedding_identity: EmbeddingIdentityBundleV1::explicit_test_model(model_id, 4),
            model_category: ModelCategory::HashEmbedder,
        }
    }

    fn verifier() -> PinnedDaemonVerifierV1 {
        PinnedDaemonVerifierV1::new("fixture-key-v1", TEST_KEY.to_vec()).unwrap()
    }

    fn fallback_embedder(value: f32, identity: EmbeddingIdentityBundleV1) -> Arc<dyn SyncEmbed> {
        Arc::new(ConstEmbedder {
            id: "fixture-space",
            model_name: "fixture-space",
            dim: 4,
            value,
            semantic: false,
            category: ModelCategory::HashEmbedder,
            identity,
        })
    }

    #[test]
    fn caller_supplied_epoch_without_producer_proof_fails_closed() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let fallback = fallback_embedder(1.0, connection.embedding_identity);
        assert!(matches!(
            DaemonFallbackEmbedder::new(daemon, fallback, DaemonRetryConfig::default()),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn verified_daemon_only_accepts_exact_authenticated_single_and_batch() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let embedder = DaemonFallbackEmbedder::with_verified_defaults(
            daemon.clone(),
            connection.clone(),
            verifier(),
        )
        .unwrap();
        assert_eq!(embedder.trust_level(), DaemonTrustLevelV1::VerifiedRemote);
        assert_eq!(
            embedder.identity().unwrap().fingerprint(),
            connection.embedding_identity.fingerprint()
        );
        assert_eq!(embedder.embed_sync("hello").unwrap(), vec![2.0; 4]);
        assert_eq!(
            embedder.embed_batch_sync(&["first", "second"]).unwrap(),
            vec![vec![2.0; 4], vec![3.0; 4]]
        );
        assert_eq!(
            embedder.embed_batch_sync(&[]).unwrap(),
            Vec::<Vec<f32>>::new()
        );
        assert_eq!(daemon.control_calls.load(Ordering::Relaxed), 2);
        assert_eq!(daemon.attested_calls.load(Ordering::Relaxed), 2);

        let challenges = daemon
            .challenges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let nonce_count = challenges
            .iter()
            .map(|challenge| challenge.request_nonce.as_str())
            .collect::<HashSet<_>>()
            .len();
        assert_eq!(nonce_count, challenges.len());
    }

    #[test]
    fn daemon_only_readiness_requires_fresh_authenticated_control_proof() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let embedder =
            DaemonFallbackEmbedder::with_verified_defaults(daemon.clone(), connection, verifier())
                .unwrap();

        assert!(embedder.is_ready());
        daemon.set_mutation(ResponseMutation::GenerationDrift);
        assert!(!embedder.is_ready());
    }

    #[test]
    fn authenticated_tcp_gateway_e2e_admits_exact_vectors_without_a_mock_transport() {
        let connection = test_connection("fixture-space");
        let (address, gateway) = spawn_authenticated_tcp_gateway(connection.clone());
        let daemon: Arc<dyn DaemonClient> = Arc::new(TcpDaemonClient { address });
        let embedder =
            DaemonFallbackEmbedder::with_verified_defaults(daemon, connection, verifier()).unwrap();
        assert_eq!(
            embedder.embed_sync("query over a real TCP socket").unwrap(),
            vec![4.0, 3.0, 2.0, 1.0]
        );
        gateway.join().unwrap();
    }

    #[test]
    fn verified_constructor_rejects_missing_attestation_wrong_key_and_key_id() {
        let connection = test_connection("fixture-space");
        for mutation in [
            ResponseMutation::MissingAttestation,
            ResponseMutation::WrongKey,
        ] {
            let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
            daemon.set_mutation(mutation);
            assert!(matches!(
                DaemonFallbackEmbedder::with_verified_defaults(
                    daemon,
                    connection.clone(),
                    verifier()
                ),
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        }

        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let wrong_key_id = PinnedDaemonVerifierV1::new("other-key", TEST_KEY.to_vec()).unwrap();
        assert!(matches!(
            DaemonFallbackEmbedder::with_verified_defaults(daemon, connection, wrong_key_id),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn every_response_drift_and_tamper_class_fails_typed() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let embedder =
            DaemonFallbackEmbedder::with_verified_defaults(daemon.clone(), connection, verifier())
                .unwrap();
        for mutation in [
            ResponseMutation::MissingAttestation,
            ResponseMutation::WrongKey,
            ResponseMutation::PayloadTamper,
            ResponseMutation::SpaceDrift,
            ResponseMutation::EndpointDrift,
            ResponseMutation::GenerationDrift,
            ResponseMutation::KeyIdDrift,
            ResponseMutation::UnknownSchema,
            ResponseMutation::ReplayNonce,
        ] {
            daemon.set_mutation(mutation);
            assert!(
                matches!(
                    embedder.embed_sync("secret query"),
                    Err(SearchError::UnverifiableRemoteSpace { .. })
                ),
                "mutation {mutation:?} must fail closed"
            );
        }
    }

    #[test]
    fn batch_order_and_shape_are_authenticated() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let embedder =
            DaemonFallbackEmbedder::with_verified_defaults(daemon.clone(), connection, verifier())
                .unwrap();
        daemon.set_mutation(ResponseMutation::ResponseReorder);
        assert!(matches!(
            embedder.embed_batch_sync(&["first", "second"]),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn reconnect_reauthenticates_and_every_attempt_uses_a_fresh_nonce() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::new(
            1,
            FailureMode::Failed,
            true,
            2.0,
            connection.clone(),
        ));
        let config = DaemonRetryConfig {
            max_attempts: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            jitter_pct: 0.0,
        };
        let embedder = DaemonFallbackEmbedder::new_verified(
            daemon.clone(),
            None,
            config,
            connection,
            verifier(),
        )
        .unwrap();

        let result = embedder.embed_sync("hello").unwrap();
        assert_eq!(result, vec![2.0; 4]);
        assert_eq!(daemon.attested_calls.load(Ordering::Relaxed), 2);
        assert_eq!(daemon.control_calls.load(Ordering::Relaxed), 4);
        let challenges = daemon
            .challenges
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let nonces = challenges
            .iter()
            .map(|challenge| challenge.request_nonce.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(nonces.len(), challenges.len());
    }

    #[test]
    fn attestation_failure_and_cancellation_never_use_local_fallback() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
        let embedder = DaemonFallbackEmbedder::new_verified(
            daemon.clone(),
            Some(fallback),
            DaemonRetryConfig::default(),
            connection,
            verifier(),
        )
        .unwrap();

        daemon.set_mutation(ResponseMutation::PayloadTamper);
        assert!(matches!(
            embedder.embed_sync("hello"),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
        daemon.set_mutation(ResponseMutation::Cancelled);
        assert!(matches!(
            embedder.embed_sync("hello"),
            Err(SearchError::Cancelled { .. })
        ));
    }

    #[test]
    fn transport_failure_uses_only_an_exact_identity_local_fallback() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
        let embedder = DaemonFallbackEmbedder::new_verified(
            daemon.clone(),
            Some(fallback),
            DaemonRetryConfig::default(),
            connection.clone(),
            verifier(),
        )
        .unwrap();
        daemon.set_available(false);
        assert_eq!(embedder.embed_sync("hello").unwrap(), vec![1.0; 4]);

        let daemon = Arc::new(FixtureDaemon::exact(connection.clone(), 2.0));
        let drifted_fallback = fallback_embedder(
            1.0,
            EmbeddingIdentityBundleV1::explicit_test_model("different-space", 4),
        );
        assert!(matches!(
            DaemonFallbackEmbedder::new_verified(
                daemon,
                Some(drifted_fallback),
                DaemonRetryConfig::default(),
                connection,
                verifier()
            ),
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn invalid_input_does_not_retry_and_uses_exact_local_fallback() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::new(
            10,
            FailureMode::InvalidInput,
            true,
            2.0,
            connection.clone(),
        ));
        let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
        let config = DaemonRetryConfig {
            max_attempts: 3,
            ..DaemonRetryConfig::default()
        };
        let embedder = DaemonFallbackEmbedder::new_verified(
            daemon.clone(),
            Some(fallback),
            config,
            connection,
            verifier(),
        )
        .unwrap();

        let result = embedder.embed_sync("hello").unwrap();
        assert_eq!(result, vec![1.0; 4]);
        assert_eq!(daemon.attested_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn assumed_daemon_is_explicit_transient_and_debug_redacts_vectors() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::exact(connection, 7.0));
        let assumed = AssumedDaemonClient::new(daemon);
        let batch = assumed.embed_batch_transient(&["first", "second"]).unwrap();
        assert_eq!(batch.trust_level(), DaemonTrustLevelV1::AssumedRemote);
        assert_eq!(batch.vectors(), &[vec![7.0; 4], vec![7.0; 4]]);
        let rendered = format!("{batch:?} {assumed:?}");
        assert!(rendered.contains("AssumedRemote"));
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("[7.0, 7.0, 7.0, 7.0]"));
        assert!(!rendered.contains("fixture-daemon"));
    }

    #[test]
    fn verifier_debug_redacts_secret_material() {
        let verifier = verifier();
        let rendered = format!("{verifier:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("0123456789abcdef"));
    }

    #[test]
    fn reranker_falls_back_when_daemon_fails() {
        let daemon = Arc::new(FixtureDaemon::new(
            10,
            FailureMode::Timeout,
            true,
            2.0,
            test_connection("fixture-space"),
        ));
        let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
            id: "fallback-reranker",
        });
        let reranker = DaemonFallbackReranker::new(
            daemon.clone(),
            Some(fallback.clone()),
            DaemonRetryConfig {
                max_attempts: 1,
                ..DaemonRetryConfig::default()
            },
        );

        let docs = vec![
            RerankDocument {
                doc_id: "a".to_string(),
                text: "doc a".to_string(),
            },
            RerankDocument {
                doc_id: "b".to_string(),
                text: "doc b".to_string(),
            },
        ];
        let result = reranker.rerank_sync("query", &docs).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].doc_id, "a");
        assert_eq!(result[0].score, 10.0);
        assert_eq!(daemon.raw_calls.load(Ordering::Relaxed), 1);
    }

    fn rerank_docs() -> Vec<RerankDocument> {
        vec![
            RerankDocument {
                doc_id: "a".to_string(),
                text: "doc a".to_string(),
            },
            RerankDocument {
                doc_id: "b".to_string(),
                text: "doc b".to_string(),
            },
        ]
    }

    fn shaped_reranker(
        scores: Vec<f32>,
        fallback: Option<Arc<dyn SyncRerank>>,
    ) -> (Arc<ShapedRerankDaemon>, DaemonFallbackReranker) {
        let daemon = Arc::new(ShapedRerankDaemon::new(scores));
        let reranker = DaemonFallbackReranker::new(
            daemon.clone(),
            fallback,
            DaemonRetryConfig {
                max_attempts: 1,
                ..DaemonRetryConfig::default()
            },
        );
        (daemon, reranker)
    }

    #[test]
    fn reranker_uses_daemon_scores_when_shape_is_valid() {
        let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
            id: "fallback-reranker",
        });
        let (daemon, reranker) = shaped_reranker(vec![0.25, 0.75], Some(fallback));
        let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].score, 0.25);
        assert_eq!(result[1].score, 0.75);
        assert_eq!(daemon.raw_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reranker_falls_back_when_daemon_returns_score_prefix() {
        let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
            id: "fallback-reranker",
        });
        let (daemon, reranker) = shaped_reranker(vec![0.9], Some(fallback));
        let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].score, 10.0,
            "fallback scores expected, not the daemon prefix"
        );
        assert_eq!(
            result[1].score, 9.0,
            "fallback scores expected, not a silent 0.0 pad"
        );
        assert_eq!(daemon.raw_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reranker_falls_back_when_daemon_returns_extra_scores() {
        let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
            id: "fallback-reranker",
        });
        let (_daemon, reranker) = shaped_reranker(vec![0.1, 0.2, 0.3], Some(fallback));
        let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
        assert_eq!(result[0].score, 10.0);
        assert_eq!(result[1].score, 9.0);
    }

    #[test]
    fn reranker_falls_back_when_daemon_returns_non_finite_score() {
        let fallback: Arc<dyn SyncRerank> = Arc::new(ConstReranker {
            id: "fallback-reranker",
        });
        let (_daemon, reranker) = shaped_reranker(vec![0.5, f32::NAN], Some(fallback));
        let result = reranker.rerank_sync("query", &rerank_docs()).unwrap();
        assert_eq!(result[0].score, 10.0);
        assert_eq!(result[1].score, 9.0);
    }

    #[test]
    fn reranker_surfaces_shape_violation_when_no_fallback_exists() {
        let (_daemon, reranker) = shaped_reranker(vec![0.9], None);
        let result = reranker.rerank_sync("query", &rerank_docs());
        assert!(
            result.is_err(),
            "a short daemon response must never zero-pad into a successful rerank"
        );
    }

    #[test]
    fn overloaded_sets_backoff_and_skips_immediate_retry() {
        let connection = test_connection("fixture-space");
        let daemon = Arc::new(FixtureDaemon::new(
            1,
            FailureMode::Overloaded {
                retry_after: Duration::from_millis(25),
            },
            true,
            2.0,
            connection.clone(),
        ));
        let fallback = fallback_embedder(1.0, connection.embedding_identity.clone());
        let config = DaemonRetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(50),
            jitter_pct: 0.0,
        };
        let embedder = DaemonFallbackEmbedder::new_verified(
            daemon.clone(),
            Some(fallback),
            config,
            connection,
            verifier(),
        )
        .unwrap();

        let _ = embedder.embed_sync("first").unwrap();
        let calls_after_first = daemon.attested_calls.load(Ordering::Relaxed);
        let _ = embedder.embed_sync("second").unwrap();
        let calls_after_second = daemon.attested_calls.load(Ordering::Relaxed);

        assert_eq!(calls_after_first, calls_after_second);
    }
}
