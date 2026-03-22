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
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};

use crate::api_provider::ApiProvider;
use crate::cached_embedder::CachedEmbedder;

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
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            state: Mutex::new(RateLimiterState {
                tokens: requests_per_minute as f64,
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
        let refill = elapsed * (self.requests_per_minute as f64 / 60.0);
        state.tokens = (state.tokens + refill).min(self.requests_per_minute as f64);
        state.last_refill = now;

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            None
        } else {
            let deficit = 1.0 - state.tokens;
            state.tokens = 0.0; // consume all partial tokens
            let wait_secs = deficit / (self.requests_per_minute as f64 / 60.0);
            Some(Duration::from_secs_f64(wait_secs))
        }
    }
}

// ─── ApiEmbedder ────────────────────────────────────────────────────────────

/// Cloud API embedder wrapping any [`ApiProvider`].
///
/// Handles HTTP transport, retry with exponential backoff, rate limiting,
/// batch chunking, and L2 normalization.
pub struct ApiEmbedder {
    provider: Box<dyn ApiProvider>,
    client: HttpClient,
    rate_limiter: RateLimiter,
    config: ApiEmbedderConfig,
}

impl fmt::Debug for ApiEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiEmbedder")
            .field("provider", &self.provider)
            .field("config", &self.config)
            .finish()
    }
}

impl ApiEmbedder {
    /// Create a new API embedder with the given provider and configuration.
    #[must_use]
    pub fn new(provider: Box<dyn ApiProvider>, config: ApiEmbedderConfig) -> Self {
        let client_config = HttpClientConfig {
            redirect_policy: RedirectPolicy::Limited(5),
            user_agent: Some(format!(
                "frankensearch/{} (api-embedder)",
                env!("CARGO_PKG_VERSION")
            )),
            ..HttpClientConfig::default()
        };
        let rate_limiter = RateLimiter::new(config.requests_per_minute);
        Self {
            provider,
            client: HttpClient::with_config(client_config),
            rate_limiter,
            config,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults(provider: Box<dyn ApiProvider>) -> Self {
        Self::new(provider, ApiEmbedderConfig::default())
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
                Err(e) => {
                    last_err = Some(SearchError::EmbeddingFailed {
                        model: self.provider.embedder_id().to_owned(),
                        source: format!("HTTP error: {e}").into(),
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
                    Err(e) => {
                        last_err = Some(SearchError::EmbeddingFailed {
                            model: self.provider.embedder_id().to_owned(),
                            source: format!("body read error: {e}").into(),
                        });
                        continue 'retry;
                    }
                }
            }

            // Success.
            if (200..300).contains(&status) {
                return self.provider.deserialize_response(&response_body);
            }

            // Retry on 429 or 5xx.
            if status == 429 || status >= 500 {
                let msg = String::from_utf8_lossy(&response_body);
                warn!(
                    provider = self.provider.provider_name(),
                    status, attempt, "transient API error: {msg}"
                );
                last_err = Some(SearchError::EmbeddingFailed {
                    model: self.provider.embedder_id().to_owned(),
                    source: format!("HTTP {status}: {msg}").into(),
                });
                continue;
            }

            // Non-retryable client error (4xx other than 429).
            let msg = String::from_utf8_lossy(&response_body);
            return Err(SearchError::EmbeddingFailed {
                model: self.provider.embedder_id().to_owned(),
                source: format!("HTTP {status}: {msg}").into(),
            });
        }

        Err(last_err.unwrap_or_else(|| SearchError::EmbeddingFailed {
            model: self.provider.embedder_id().to_owned(),
            source: "all retries exhausted".into(),
        }))
    }
}

/// L2-normalize a vector in place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

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
                    l2_normalize(&mut v);
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
                    l2_normalize(v);
                }
                all_embeddings.extend(batch);
            }

            Ok(all_embeddings)
        })
    }

    fn dimension(&self) -> usize {
        self.provider.dimension()
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
        if target_dim > embedding.len() {
            return Err(SearchError::EmbeddingFailed {
                model: self.provider.embedder_id().to_owned(),
                source: format!(
                    "target dimension {target_dim} exceeds embedding dimension {}",
                    embedding.len()
                )
                .into(),
            });
        }
        let mut truncated = embedding[..target_dim].to_vec();
        l2_normalize(&mut truncated);
        Ok(truncated)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_provider::OpenAiProvider;

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn l2_normalize_general() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
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
        let embedder = ApiEmbedder::with_defaults(provider);
        assert_eq!(embedder.dimension(), 256);
        assert_eq!(embedder.id(), "openai-text-embedding-3-small-256d");
        assert!(embedder.is_semantic());
        assert_eq!(embedder.category(), ModelCategory::ApiEmbedder);
        assert!(embedder.supports_mrl());
    }

    #[test]
    fn truncate_embedding_works() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider);
        let emb = vec![1.0, 2.0, 3.0, 4.0];
        let truncated = embedder.truncate_embedding(&emb, 2).unwrap();
        assert_eq!(truncated.len(), 2);
        let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn truncate_embedding_rejects_larger_dim() {
        let provider = Box::new(OpenAiProvider::text_embedding_3_small("key", Some(4)));
        let embedder = ApiEmbedder::with_defaults(provider);
        let emb = vec![1.0, 2.0];
        assert!(embedder.truncate_embedding(&emb, 4).is_err());
    }
}
