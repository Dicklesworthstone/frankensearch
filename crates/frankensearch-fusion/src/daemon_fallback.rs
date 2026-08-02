//! Daemon fallback wrappers for sync embedders and rerankers.
//!
//! These wrappers attempt daemon inference first and gracefully fall back to
//! local in-process models with bounded retry and jittered backoff.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use frankensearch_core::{
    DaemonClient, DaemonError, DaemonRetryConfig, EmbeddingIdentityBundleV1,
    FrozenEmbeddingIdentityBundleV1, ModelCategory, RerankDocument, RerankScore, SearchError,
    SearchResult, SyncEmbed, SyncRerank, next_request_id,
};
use tracing::{debug, warn};

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

fn lock_state(state: &Mutex<DaemonState>) -> std::sync::MutexGuard<'_, DaemonState> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Embedder wrapper that uses the daemon when available and falls back to a local embedder.
pub struct DaemonFallbackEmbedder {
    daemon: Arc<dyn DaemonClient>,
    fallback: Arc<dyn SyncEmbed>,
    identity: EmbeddingIdentityBundleV1,
    daemon_identity: Option<EmbeddingIdentityBundleV1>,
    config: DaemonRetryConfig,
    state: Mutex<DaemonState>,
}

impl DaemonFallbackEmbedder {
    /// # Errors
    ///
    /// Returns `InvalidConfig` when the local fallback does not expose a
    /// complete immutable identity.
    pub fn new(
        daemon: Arc<dyn DaemonClient>,
        fallback: Arc<dyn SyncEmbed>,
        config: DaemonRetryConfig,
    ) -> SearchResult<Self> {
        let identity = Self::validated_fallback_identity(fallback.as_ref())?;
        Ok(Self {
            daemon,
            fallback,
            identity,
            daemon_identity: None,
            config,
            state: Mutex::new(DaemonState::new()),
        })
    }

    /// Construct a wrapper that may accept daemon vectors only when the daemon
    /// is pinned to the exact complete identity exposed by the fallback.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::UnverifiableRemoteSpace`] when the epoch is
    /// invalid or differs in space, producer, input, or storage identity. A
    /// dynamic wrapper cannot truthfully expose two producer identities through
    /// one `SyncEmbed` object.
    pub fn with_immutable_daemon_epoch(
        daemon: Arc<dyn DaemonClient>,
        fallback: Arc<dyn SyncEmbed>,
        config: DaemonRetryConfig,
        daemon_epoch: FrozenEmbeddingIdentityBundleV1,
    ) -> SearchResult<Self> {
        daemon_epoch.validate().map_err(|_error| {
            unverifiable_daemon_space(
                daemon.id(),
                "explicit daemon epoch failed canonical validation",
            )
        })?;
        let daemon_identity = daemon_epoch.identity;
        let identity = Self::validated_fallback_identity(fallback.as_ref())?;
        if daemon_identity.fingerprint() != identity.fingerprint() {
            return Err(unverifiable_daemon_space(
                daemon.id(),
                "daemon epoch must exactly equal the wrapper's exposed fallback identity",
            ));
        }
        Ok(Self {
            daemon,
            fallback,
            daemon_identity: Some(identity.clone()),
            identity,
            config,
            state: Mutex::new(DaemonState::new()),
        })
    }

    fn validated_fallback_identity(
        fallback: &dyn SyncEmbed,
    ) -> SearchResult<EmbeddingIdentityBundleV1> {
        let identity = fallback.identity()?.clone();
        identity.validate()?;
        let identity_dimension =
            usize::try_from(identity.space.dimension).map_err(|_| SearchError::InvalidConfig {
                field: "daemon_fallback.identity.dimension".to_owned(),
                value: identity.space.dimension.to_string(),
                reason: "identity dimension does not fit usize".to_owned(),
            })?;
        if fallback.dimension() != identity_dimension {
            return Err(SearchError::InvalidConfig {
                field: "daemon_fallback.dimension".to_owned(),
                value: fallback.dimension().to_string(),
                reason: format!(
                    "fallback dimension disagrees with immutable identity dimension {identity_dimension}"
                ),
            });
        }
        let identity_is_semantic = matches!(
            identity.space.kind,
            frankensearch_core::EmbeddingSpaceKindV1::Semantic
        );
        if fallback.is_semantic() != identity_is_semantic {
            return Err(SearchError::InvalidConfig {
                field: "daemon_fallback.is_semantic".to_owned(),
                value: fallback.is_semantic().to_string(),
                reason: "fallback semantic classification disagrees with immutable identity"
                    .to_owned(),
            });
        }
        Ok(identity)
    }

    const fn should_retry(err: &DaemonError) -> bool {
        !matches!(
            err,
            DaemonError::InvalidInput(_) | DaemonError::Overloaded { .. }
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
            DaemonError::UnverifiableRemoteSpace => "unverifiable_space",
        }
    }

    fn log_fallback(&self, request_id: &str, retries: u32, reason: &str) {
        warn!(
            request_id = request_id,
            retry_count = retries,
            fallback_reason = reason,
            "Daemon embed failed; using local embedder"
        );
    }

    fn try_embed(&self, request_id: &str, text: &str) -> Result<Vec<f32>, DaemonFailure> {
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
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                "Attempting daemon embed"
            );
            match self.daemon.embed(text, request_id) {
                Ok(vector) => {
                    if self.daemon_identity.as_ref() != Some(&self.identity) {
                        return Err(DaemonFailure {
                            error: DaemonError::Failed(
                                "daemon response has no matching immutable embedding epoch"
                                    .to_owned(),
                            ),
                            attempts,
                            backoff: false,
                        });
                    }
                    if vector.len() != self.fallback.dimension() {
                        return Err(DaemonFailure {
                            error: DaemonError::Failed(
                                "daemon response dimension disagrees with immutable embedding epoch"
                                    .to_owned(),
                            ),
                            attempts,
                            backoff: false,
                        });
                    }
                    lock_state(&self.state).record_success();
                    return Ok(vector);
                }
                Err(err) => {
                    let should_retry = Self::should_retry(&err);
                    let should_backoff = !matches!(err, DaemonError::InvalidInput(_));
                    let backoff = if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &err);
                        true
                    } else {
                        false
                    };

                    debug!(
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = Self::fallback_reason(&err, false),
                        "Daemon embed failed"
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
                .unwrap_or_else(|| DaemonError::Unavailable("daemon embed failed".to_string())),
            attempts,
            backoff: false,
        })
    }

    fn try_embed_batch(
        &self,
        request_id: &str,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, DaemonFailure> {
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
                request_id,
                attempt = attempts,
                max_attempts = self.config.max_attempts,
                "Attempting daemon embed batch"
            );
            match self.daemon.embed_batch(texts, request_id) {
                Ok(vectors) => {
                    if self.daemon_identity.as_ref() != Some(&self.identity) {
                        return Err(DaemonFailure {
                            error: DaemonError::Failed(
                                "daemon response has no matching immutable embedding epoch"
                                    .to_owned(),
                            ),
                            attempts,
                            backoff: false,
                        });
                    }
                    if vectors.len() != texts.len()
                        || vectors
                            .iter()
                            .any(|vector| vector.len() != self.fallback.dimension())
                    {
                        return Err(DaemonFailure {
                            error: DaemonError::Failed(
                                "daemon batch shape disagrees with immutable embedding epoch"
                                    .to_owned(),
                            ),
                            attempts,
                            backoff: false,
                        });
                    }
                    lock_state(&self.state).record_success();
                    return Ok(vectors);
                }
                Err(err) => {
                    let should_retry = Self::should_retry(&err);
                    let should_backoff = !matches!(err, DaemonError::InvalidInput(_));
                    let backoff = if should_backoff {
                        lock_state(&self.state).record_failure(&self.config, &err);
                        true
                    } else {
                        false
                    };

                    debug!(
                        request_id,
                        attempt = attempts,
                        max_attempts = self.config.max_attempts,
                        will_retry = should_retry && attempts < self.config.max_attempts,
                        error_kind = Self::fallback_reason(&err, false),
                        "Daemon embed batch failed"
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
                .unwrap_or_else(|| DaemonError::Unavailable("daemon embed failed".to_string())),
            attempts,
            backoff: false,
        })
    }
}

fn unverifiable_daemon_space(producer: &str, reason: &str) -> SearchError {
    let producer = if producer.len() <= 128
        && !producer.is_empty()
        && producer
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        producer.to_owned()
    } else {
        "<redacted-daemon-producer>".to_owned()
    };
    let reason = if reason.len() <= 512 && !reason.chars().any(char::is_control) {
        reason.to_owned()
    } else {
        "daemon identity validation failed".to_owned()
    };
    SearchError::UnverifiableRemoteSpace { producer, reason }
}

impl SyncEmbed for DaemonFallbackEmbedder {
    fn embed_sync(&self, text: &str) -> SearchResult<Vec<f32>> {
        let request_id = next_request_id();
        match self.try_embed(&request_id, text) {
            Ok(vector) => Ok(vector),
            Err(failure) => {
                let retries = failure.attempts.saturating_sub(1);
                let reason = Self::fallback_reason(&failure.error, failure.backoff);
                self.log_fallback(&request_id, retries, reason);
                self.fallback.embed_sync(text)
            }
        }
    }

    fn embed_batch_sync(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>> {
        let request_id = next_request_id();
        match self.try_embed_batch(&request_id, texts) {
            Ok(vectors) => Ok(vectors),
            Err(failure) => {
                let retries = failure.attempts.saturating_sub(1);
                let reason = Self::fallback_reason(&failure.error, failure.backoff);
                self.log_fallback(&request_id, retries, reason);
                self.fallback.embed_batch_sync(texts)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.fallback.dimension()
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }

    fn id(&self) -> &str {
        self.fallback.id()
    }

    fn model_name(&self) -> &str {
        self.fallback.model_name()
    }

    fn is_semantic(&self) -> bool {
        self.fallback.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.fallback.category()
    }
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
}

impl SyncRerank for DaemonFallbackReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        let texts: Vec<&str> = documents.iter().map(|doc| doc.text.as_str()).collect();
        let request_id = next_request_id();

        match self.try_rerank(&request_id, query, &texts) {
            Ok(scores) => Ok(documents
                .iter()
                .enumerate()
                .map(|(index, doc)| RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score: scores.get(index).copied().unwrap_or(0.0),
                    original_rank: index,
                    raw_logit: None,
                })
                .collect()),
            Err(failure) => {
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

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

    #[derive(Clone, Copy)]
    enum FailureMode {
        Unavailable,
        Timeout,
        Overloaded { retry_after: Duration },
        Failed,
        InvalidInput,
    }

    impl FailureMode {
        fn error(&self) -> DaemonError {
            match self {
                Self::Unavailable => DaemonError::Unavailable("daemon down".to_string()),
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

    struct FixtureDaemon {
        calls: AtomicUsize,
        fail_first: usize,
        mode: FailureMode,
        available: bool,
        embed_value: f32,
    }

    impl FixtureDaemon {
        fn new(fail_first: usize, mode: FailureMode, available: bool, embed_value: f32) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                fail_first,
                mode,
                available,
                embed_value,
            }
        }
    }

    impl DaemonClient for FixtureDaemon {
        fn id(&self) -> &str {
            "fixture-daemon"
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn embed(&self, _text: &str, _request_id: &str) -> Result<Vec<f32>, DaemonError> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
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
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
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
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call < self.fail_first {
                Err(self.mode.error())
            } else {
                Ok((0..documents.len())
                    .map(|idx| (documents.len() - idx) as f32)
                    .collect())
            }
        }
    }

    fn fallback_embedder(value: f32) -> Arc<dyn SyncEmbed> {
        Arc::new(ConstEmbedder {
            id: "fallback-embed",
            model_name: "fallback-embed",
            dim: 4,
            value,
            semantic: false,
            category: ModelCategory::HashEmbedder,
            identity: EmbeddingIdentityBundleV1::explicit_test_model("fallback-embed", 4),
        })
    }

    #[test]
    fn embedder_falls_back_when_daemon_unavailable() {
        let daemon = Arc::new(FixtureDaemon::new(1, FailureMode::Unavailable, false, 2.0));
        let fallback = fallback_embedder(1.0);
        let embedder =
            DaemonFallbackEmbedder::new(daemon.clone(), fallback, DaemonRetryConfig::default())
                .unwrap();

        let result = embedder.embed_sync("hello").unwrap();
        assert_eq!(result, vec![1.0; 4]);
        assert_eq!(daemon.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn embedder_rejects_invalid_fallback_identity() {
        let daemon = Arc::new(FixtureDaemon::new(0, FailureMode::Failed, false, 2.0));
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("invalid-fallback", 4);
        identity.storage.dimension = 3;
        let fallback: Arc<dyn SyncEmbed> = Arc::new(ConstEmbedder {
            id: "invalid-fallback",
            model_name: "invalid-fallback",
            dim: 4,
            value: 1.0,
            semantic: false,
            category: ModelCategory::HashEmbedder,
            identity,
        });
        assert!(matches!(
            DaemonFallbackEmbedder::new(daemon, fallback, DaemonRetryConfig::default()),
            Err(SearchError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn embedder_rejects_fallback_metadata_that_disagrees_with_identity() {
        let daemon = Arc::new(FixtureDaemon::new(0, FailureMode::Failed, false, 2.0));
        let fallback: Arc<dyn SyncEmbed> = Arc::new(ConstEmbedder {
            id: "wrong-dimension",
            model_name: "wrong-dimension",
            dim: 3,
            value: 1.0,
            semantic: false,
            category: ModelCategory::HashEmbedder,
            identity: EmbeddingIdentityBundleV1::explicit_test_model("wrong-dimension", 4),
        });
        let result = DaemonFallbackEmbedder::new(daemon, fallback, DaemonRetryConfig::default());
        assert!(result.is_err(), "dimension mismatch must fail closed");
        let error = result.err().expect("asserted error result");
        assert!(error.to_string().contains("fallback dimension disagrees"));

        let daemon = Arc::new(FixtureDaemon::new(0, FailureMode::Failed, false, 2.0));
        let fallback: Arc<dyn SyncEmbed> = Arc::new(ConstEmbedder {
            id: "wrong-semantic-kind",
            model_name: "wrong-semantic-kind",
            dim: 4,
            value: 1.0,
            semantic: true,
            category: ModelCategory::StaticEmbedder,
            identity: EmbeddingIdentityBundleV1::explicit_test_model("wrong-semantic-kind", 4),
        });
        let result = DaemonFallbackEmbedder::new(daemon, fallback, DaemonRetryConfig::default());
        assert!(
            result.is_err(),
            "semantic classification mismatch must fail closed"
        );
        let error = result.err().expect("asserted error result");
        assert!(
            error
                .to_string()
                .contains("semantic classification disagrees")
        );
    }

    #[test]
    fn embedder_retries_then_uses_daemon() {
        let daemon = Arc::new(FixtureDaemon::new(1, FailureMode::Failed, true, 2.0));
        let fallback = fallback_embedder(1.0);
        let daemon_identity = fallback.identity().unwrap().freeze().unwrap();
        let config = DaemonRetryConfig {
            max_attempts: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            jitter_pct: 0.0,
        };
        let embedder = DaemonFallbackEmbedder::with_immutable_daemon_epoch(
            daemon.clone(),
            fallback,
            config,
            daemon_identity,
        )
        .unwrap();

        let result = embedder.embed_sync("hello").unwrap();
        assert_eq!(result, vec![2.0; 4]);
        assert_eq!(daemon.calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn embedder_rejects_drifted_daemon_epoch_with_typed_error() {
        let daemon = Arc::new(FixtureDaemon::new(0, FailureMode::Failed, true, 2.0));
        let fallback = fallback_embedder(1.0);
        let drifted = EmbeddingIdentityBundleV1::explicit_test_model("different-embedder", 4)
            .freeze()
            .unwrap();
        let result = DaemonFallbackEmbedder::with_immutable_daemon_epoch(
            daemon,
            fallback,
            DaemonRetryConfig::default(),
            drifted,
        );
        assert!(matches!(
            result,
            Err(SearchError::UnverifiableRemoteSpace { .. })
        ));
    }

    #[test]
    fn unverifiable_daemon_space_redacts_untrusted_labels() {
        let error =
            unverifiable_daemon_space("daemon\nforged-log-line", "reason\nforged-reason-line");
        let rendered = error.to_string();
        assert!(rendered.contains("<redacted-daemon-producer>"));
        assert!(rendered.contains("daemon identity validation failed"));
        assert!(!rendered.contains("forged-log-line"));
        assert!(!rendered.contains("forged-reason-line"));
    }

    #[test]
    fn embedder_rejects_unattested_daemon_success_and_falls_back() {
        let daemon = Arc::new(FixtureDaemon::new(0, FailureMode::Failed, true, 2.0));
        let fallback = fallback_embedder(1.0);
        let embedder =
            DaemonFallbackEmbedder::new(daemon.clone(), fallback, DaemonRetryConfig::default())
                .unwrap();

        let result = embedder.embed_sync("hello").unwrap();
        assert_eq!(result, vec![1.0; 4]);
        assert_eq!(daemon.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn embedder_invalid_input_does_not_retry() {
        let daemon = Arc::new(FixtureDaemon::new(10, FailureMode::InvalidInput, true, 2.0));
        let fallback = fallback_embedder(1.0);
        let config = DaemonRetryConfig {
            max_attempts: 3,
            ..DaemonRetryConfig::default()
        };
        let embedder = DaemonFallbackEmbedder::new(daemon.clone(), fallback, config).unwrap();

        let result = embedder.embed_sync("hello").unwrap();
        assert_eq!(result, vec![1.0; 4]);
        assert_eq!(daemon.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reranker_falls_back_when_daemon_fails() {
        let daemon = Arc::new(FixtureDaemon::new(10, FailureMode::Timeout, true, 2.0));
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
        assert_eq!(daemon.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn overloaded_sets_backoff_and_skips_immediate_retry() {
        let daemon = Arc::new(FixtureDaemon::new(
            1,
            FailureMode::Overloaded {
                retry_after: Duration::from_millis(25),
            },
            true,
            2.0,
        ));
        let fallback = fallback_embedder(1.0);
        let config = DaemonRetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(50),
            jitter_pct: 0.0,
        };
        let embedder = DaemonFallbackEmbedder::new(daemon.clone(), fallback, config).unwrap();

        let _ = embedder.embed_sync("first").unwrap();
        let calls_after_first = daemon.calls.load(Ordering::Relaxed);
        let _ = embedder.embed_sync("second").unwrap();
        let calls_after_second = daemon.calls.load(Ordering::Relaxed);

        assert_eq!(calls_after_first, calls_after_second);
    }
}
