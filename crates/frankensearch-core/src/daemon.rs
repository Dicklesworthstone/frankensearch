//! Daemon client abstraction for warm embedding and reranking.
//!
//! This module defines the protocol-agnostic daemon interfaces shared by
//! host applications and fusion-layer fallback wrappers.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

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
}

impl fmt::Display for DaemonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(msg) => write!(f, "daemon unavailable: {msg}"),
            Self::Timeout(msg) => write!(f, "daemon timeout: {msg}"),
            Self::Overloaded { message, .. } => write!(f, "daemon overloaded: {message}"),
            Self::Failed(msg) => write!(f, "daemon failed: {msg}"),
            Self::InvalidInput(msg) => write!(f, "daemon invalid input: {msg}"),
        }
    }
}

impl std::error::Error for DaemonError {}

/// Abstract daemon client.
///
/// Concrete transports (e.g. UDS/HTTP) are implemented by host applications.
#[allow(clippy::missing_errors_doc)]
pub trait DaemonClient: Send + Sync {
    fn id(&self) -> &str;
    fn is_available(&self) -> bool;

    fn embed(&self, text: &str, request_id: &str) -> Result<Vec<f32>, DaemonError>;
    fn embed_batch(&self, texts: &[&str], request_id: &str) -> Result<Vec<Vec<f32>>, DaemonError>;
    fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        request_id: &str,
    ) -> Result<Vec<f32>, DaemonError>;
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
}
