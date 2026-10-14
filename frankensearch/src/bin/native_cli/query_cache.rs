//! Optional warm query embedding cache.

use std::sync::Arc;
use frankensearch::{Cx, Embedder, SearchError, SearchResult};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, ModelTier, SearchFuture};
use frankensearch_embed::CachedEmbedder;
use crate::{Models, Result, bad};

const MAX_ENTRIES: usize = 1024;
const MAX_CACHED_INPUT_BYTES: usize = 64 * 1024;
const MAX_CACHED_DIMENSION: usize = 16 * 1024;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Config { entries: usize }

impl Config {
    pub fn parse(value: &str) -> Result<Self> {
        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad("query cache entries must be an integer between 0 and 1024"));
        }
        let entries = value.parse::<usize>().ok().filter(|n| *n <= MAX_ENTRIES)
            .ok_or_else(|| bad("query cache entries must be an integer between 0 and 1024"))?;
        Ok(Self { entries })
    }
    #[must_use]
    pub const fn enabled(self) -> bool { self.entries != 0 }
    pub fn wrap(self, cx: &Cx, models: Models) -> Result<Models> {
        checkpoint(cx)?;
        if !self.enabled() { return Ok(models); }
        let wrap = |inner: Arc<dyn Embedder>| -> Arc<dyn Embedder> {
            Arc::new(ServingEmbedder { cache: CachedEmbedder::new(Arc::clone(&inner), self.entries), inner })
        };
        Ok(Models { fast: wrap(models.fast), quality: models.quality.map(wrap) })
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|_| SearchError::Cancelled { phase: "native_cli.query_cache".to_owned(), reason: "embedding cache operation cancelled".to_owned() })?;
    if cx.is_cancel_requested() { return Err(SearchError::Cancelled { phase: "native_cli.query_cache".to_owned(), reason: "embedding cache operation cancelled".to_owned() }); }
    Ok(())
}

struct ServingEmbedder { inner: Arc<dyn Embedder>, cache: CachedEmbedder }

impl Embedder for ServingEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> { self.inner.embed(cx, text) }
    fn embed_batch<'a>(&'a self, cx: &'a Cx, texts: &'a [&'a str]) -> SearchFuture<'a, Vec<Vec<f32>>> { self.inner.embed_batch(cx, texts) }
    fn embed_bound<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            checkpoint(cx)?;
            let dimension = self.inner.dimension();
            checkpoint(cx)?;
            let out = if text.len() <= MAX_CACHED_INPUT_BYTES && dimension <= MAX_CACHED_DIMENSION { self.cache.embed_bound(cx, text).await } else { self.inner.embed_bound(cx, text).await };
            checkpoint(cx)?;
            out
        })
    }
    fn embed_batch_bound<'a>(&'a self, cx: &'a Cx, texts: &'a [&'a str]) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> { self.inner.embed_batch_bound(cx, texts) }
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { self.inner.identity() }
    fn bound_batch_is_native(&self) -> bool { self.inner.bound_batch_is_native() }
    fn dimension(&self) -> usize { self.inner.dimension() }
    fn id(&self) -> &str { self.inner.id() }
    fn model_name(&self) -> &str { self.inner.model_name() }
    fn is_ready(&self) -> bool { self.inner.is_ready() }
    fn is_semantic(&self) -> bool { self.inner.is_semantic() }
    fn category(&self) -> ModelCategory { self.inner.category() }
    fn tier(&self) -> ModelTier { self.inner.tier() }
    fn supports_mrl(&self) -> bool { self.inner.supports_mrl() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parser_is_bounded() {
        assert!(!Config::parse("0").unwrap().enabled());
        assert!(Config::parse("1024").unwrap().enabled());
        for value in ["", "-1", "1.0", "1025", " 1"] { assert!(Config::parse(value).is_err()); }
    }
}
