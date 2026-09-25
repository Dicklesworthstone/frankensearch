//! Opt-in recovery of a rejected native inference batch by ordered bisection.
//!
//! Only ordinary `EmbeddingFailed` responses are split, using the SAME bound
//! operation and captured producer. No raw fallback, omitted row, asynchronous
//! worker or model substitution can turn a failed cohort into a success.

use std::sync::Arc;

use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, ModelTier, SearchFuture};

use super::checkpoint;
use crate::{Cx, Embedder, SearchError, SearchResult};

pub(super) fn wrap(
    inner: Arc<dyn Embedder>,
    identity: EmbeddingIdentityBundleV1,
) -> Arc<dyn Embedder> {
    Arc::new(SplittingEmbedder { inner, identity })
}

struct SplittingEmbedder {
    inner: Arc<dyn Embedder>,
    identity: EmbeddingIdentityBundleV1,
}

impl SplittingEmbedder {
    fn identity_error() -> SearchError {
        SearchError::UnverifiableRemoteSpace {
            producer: "native_ann.builder.batch".to_owned(),
            reason: "batch inference no longer matches the admitted producer contract".to_owned(),
        }
    }

    fn admit_producer(&self) -> SearchResult<()> {
        let current = self.inner.identity().map_err(|_| Self::identity_error())?;
        if current != &self.identity
            || usize::try_from(self.identity.space.dimension).ok() != Some(self.inner.dimension())
        {
            return Err(Self::identity_error());
        }
        Ok(())
    }

    fn admit_response(&self, response: &IdentityBoundEmbedding) -> SearchResult<()> {
        // Compare first: a foreign response must not disclose arbitrary fields
        // through the identity validator's more detailed diagnostics.
        if response.identity != self.identity {
            return Err(Self::identity_error());
        }
        response.validate()
    }

    async fn split_batch(
        &self,
        cx: &Cx,
        texts: &[&str],
    ) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        checkpoint(cx, "native_ann.builder.before_batch")?;
        self.admit_producer()?;
        // Keep native empty-batch admission/error behavior. There is no split
        // for an empty input and no manufactured successful empty response.
        // A work stack of index ranges, seeded with the whole batch.
        let mut pending = Vec::new();
        pending.push(0..texts.len());
        let mut accepted = Vec::with_capacity(texts.len());
        while let Some(range) = pending.pop() {
            checkpoint(cx, "native_ann.builder.before_batch")?;
            self.admit_producer()?;
            let outcome = self
                .inner
                .embed_batch_bound(cx, &texts[range.clone()])
                .await;
            // Cancellation outranks both a provider failure and late success.
            checkpoint(cx, "native_ann.builder.after_batch")?;
            let outcome = match outcome {
                Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                outcome => outcome,
            };
            self.admit_producer()?;
            let mut response = match outcome {
                Ok(response) => response,
                Err(SearchError::EmbeddingFailed { .. }) if range.len() > 1 => {
                    let middle = range.start + range.len() / 2;
                    tracing::debug!(
                        batch_size = range.len(),
                        left_size = middle - range.start,
                        "splitting rejected bound embedding batch with the same producer"
                    );
                    // Right first on a LIFO stack preserves original input order.
                    // Each child is strictly smaller: at most 2*N-1 requests,
                    // and O(log N) pending ranges for N nonempty input slots.
                    pending.push(middle..range.end);
                    pending.push(range.start..middle);
                    continue;
                }
                Err(error) => return Err(error),
            };
            if response.len() != range.len() {
                return Err(SearchError::InvalidConfig {
                    field: "native_ann.builder.batch_cardinality".to_owned(),
                    value: response.len().to_string(),
                    reason: format!("expected exactly {} bound outputs", range.len()),
                });
            }
            for value in &response {
                checkpoint(cx, "native_ann.builder.admit_output")?;
                self.admit_response(value)?;
            }
            accepted.append(&mut response);
        }
        checkpoint(cx, "native_ann.builder.batch_complete")?;
        self.admit_producer()?;
        // The caller receives all outputs at once and validates before writing.
        // A later leaf failure drops every previously accepted sibling output.
        Ok(accepted)
    }
}

impl Embedder for SplittingEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.inner.embed(cx, text)
    }

    fn embed_batch<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        self.inner.embed_batch(cx, texts)
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        // Query inference keeps the native single-input operation. The builder
        // never reaches this path when recovering a rejected batch.
        self.inner.embed_bound(cx, text)
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(self.split_batch(cx, texts))
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        // Do not hide drift behind our captured copy. Existing build/query
        // admission must still see the provider's actual current identity.
        self.inner.identity()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }

    fn tier(&self) -> ModelTier {
        self.inner.tier()
    }

    fn supports_mrl(&self) -> bool {
        self.inner.supports_mrl()
    }
}

#[cfg(test)]
mod tests;
