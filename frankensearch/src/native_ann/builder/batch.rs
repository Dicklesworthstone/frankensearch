//! Identity-admitted inference and opt-in recovery by ordered bisection.
//!
//! Native batches require the provider's explicit bound-batch contract. Other
//! providers retain their bound-single operation for every input slot. Only
//! ordinary `EmbeddingFailed` responses from native batches are split, using
//! the SAME operation and captured producer. No raw fallback, omitted row,
//! asynchronous worker or model substitution can rescue a failed cohort.

use std::sync::Arc;

use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, ModelTier, SearchFuture};

use super::checkpoint;
use crate::{Cx, Embedder, SearchError, SearchResult};

fn identity_error() -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "native_ann.builder.batch".to_owned(),
        reason: "batch inference no longer matches the admitted producer contract".to_owned(),
    }
}

fn admit_current_producer(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
    native_batch: bool,
    phase: &'static str,
) -> SearchResult<()> {
    checkpoint(cx, phase)?;
    let identity = embedder.identity();
    let dimension = embedder.dimension();
    let current_native_batch = embedder.bound_batch_is_native();
    // Provider metadata access is user code too. Cancellation during that
    // access must win before another inference call or any writer admission.
    checkpoint(cx, phase)?;
    if identity.is_ok_and(|identity| identity == expected)
        && usize::try_from(expected.space.dimension).ok() == Some(dimension)
        && current_native_batch == native_batch
    {
        Ok(())
    } else {
        Err(identity_error())
    }
}

fn finish_admitted_call<T>(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
    native_batch: bool,
    outcome: SearchResult<T>,
) -> SearchResult<T> {
    checkpoint(cx, "native_ann.builder.after_batch")?;
    match outcome {
        Err(error @ SearchError::Cancelled { .. }) => Err(error),
        outcome => {
            // Check before inspecting an ordinary failure: changed producers
            // must never be made retryable by a failed inference response.
            admit_current_producer(
                cx,
                embedder,
                expected,
                native_batch,
                "native_ann.builder.after_batch",
            )?;
            outcome
        }
    }
}

fn admit_output(
    expected: &EmbeddingIdentityBundleV1,
    response: &IdentityBoundEmbedding,
) -> SearchResult<()> {
    // Compare before validating so untrusted foreign fields are not echoed by
    // more detailed validation errors. Never attach our identity to raw output.
    if &response.identity != expected {
        return Err(identity_error());
    }
    response.validate()
}

/// Produce a complete ordered group under the builder's captured identity.
/// Nothing from this group is handed to a writer until all slots are admitted.
pub(super) async fn infer_admitted(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
    texts: &[&str],
) -> SearchResult<Vec<IdentityBoundEmbedding>> {
    checkpoint(cx, "native_ann.builder.before_batch")?;
    let native_batch = embedder.bound_batch_is_native();
    admit_current_producer(
        cx,
        embedder,
        expected,
        native_batch,
        "native_ann.builder.before_batch",
    )?;
    let response = if native_batch {
        let outcome = embedder.embed_batch_bound(cx, texts).await;
        let response = finish_admitted_call(cx, embedder, expected, native_batch, outcome)?;
        if response.len() != texts.len() {
            return Err(SearchError::InvalidConfig {
                field: "native_ann.builder.batch_cardinality".to_owned(),
                value: response.len().to_string(),
                reason: format!("expected exactly {} bound outputs", texts.len()),
            });
        }
        for bound in &response {
            checkpoint(cx, "native_ann.builder.admit_output")?;
            admit_output(expected, bound)?;
        }
        response
    } else {
        let mut response = Vec::with_capacity(texts.len());
        for text in texts {
            admit_current_producer(
                cx,
                embedder,
                expected,
                native_batch,
                "native_ann.builder.before_batch",
            )?;
            let outcome = embedder.embed_bound(cx, text).await;
            let bound = finish_admitted_call(cx, embedder, expected, native_batch, outcome)?;
            checkpoint(cx, "native_ann.builder.admit_output")?;
            admit_output(expected, &bound)?;
            response.push(bound);
        }
        response
    };
    admit_current_producer(
        cx,
        embedder,
        expected,
        native_batch,
        "native_ann.builder.batch_complete",
    )?;
    Ok(response)
}

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
    async fn split_batch(
        &self,
        cx: &Cx,
        texts: &[&str],
    ) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        checkpoint(cx, "native_ann.builder.before_batch")?;
        let native_batch = self.inner.bound_batch_is_native();
        if !native_batch {
            // A failure of an already-single operation cannot be fixed by
            // bisecting a group. In particular, do not try the inherited raw
            // bound-batch default after a custom bound-single refusal.
            return infer_admitted(cx, self.inner.as_ref(), &self.identity, texts).await;
        }
        // Keep native empty-batch admission/error behavior. There is no split
        // for an empty input and no manufactured successful empty response.
        // A work stack of index ranges, seeded with the whole batch.
        let mut pending = Vec::new();
        pending.push(0..texts.len());
        let mut accepted = Vec::with_capacity(texts.len());
        while let Some(range) = pending.pop() {
            admit_current_producer(
                cx,
                self.inner.as_ref(),
                &self.identity,
                native_batch,
                "native_ann.builder.before_batch",
            )?;
            let outcome = self
                .inner
                .embed_batch_bound(cx, &texts[range.clone()])
                .await;
            let outcome = finish_admitted_call(
                cx,
                self.inner.as_ref(),
                &self.identity,
                native_batch,
                outcome,
            );
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
                admit_output(&self.identity, value)?;
            }
            accepted.append(&mut response);
        }
        admit_current_producer(
            cx,
            self.inner.as_ref(),
            &self.identity,
            native_batch,
            "native_ann.builder.batch_complete",
        )?;
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

    /// Splitting re-sends the inner producer's own bound batches, so it is as
    /// faithful to single inputs as that producer is.
    fn bound_batch_is_native(&self) -> bool {
        self.inner.bound_batch_is_native()
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
