//! Full-mode tuning and bounded, lazy reranking over one retained authority.
//! The ordinary CLI policy owns validation and request limits; this adapter
//! changes only how that policy reaches a fixed generation's native query.

use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use frankensearch::native_ann::builder::scope::NativeScopedGenerationSnapshot;
use frankensearch::native_ann::{NativeGenerationSnapshot, NativeProgressiveSearch};
use frankensearch::{Embedder, ScoredResult};

use super::{
    AuthoritySearchRequest, AuthoritySession, Cx, Deserialize, Options, QueryMode, Result,
    SearchError, Write, bad, checkpoint, query, serve_refused,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Search,
}

/// Match the ordinary native CLI's explicit tuning envelope. Flat search
/// requests keep their existing fields and cannot implicitly grant tuning.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SearchEnvelope {
    #[serde(rename = "op")]
    _operation: Operation,
    pub(super) request: AuthoritySearchRequest,
    #[serde(deserialize_with = "query::deserialize_query_override")]
    full_query: query::FullQueryOverride,
}

pub(super) fn validate_startup(options: &Options, mode: QueryMode) -> Result<()> {
    if (options.full_query.is_some() || options.reranker_dir.is_some()) && mode != QueryMode::Full {
        return Err(bad(
            "full-query tuning and reranking require a complete full generation in full mode",
        ));
    }
    Ok(())
}

pub(super) fn load_policy(cx: &Cx, options: &Options) -> Result<query::Policy> {
    let mut policy = query::Policy::new(cx, options.timeout_ms)?;
    policy.query_pool = cx.blocking_pool_handle();
    policy.full_query.clone_from(&options.full_query);
    policy.rerank = options
        .reranker_dir
        .as_deref()
        .map(|path| query::Rerank::load(cx, path, options.rerank_window, cx.blocking_pool_handle()))
        .transpose()?;
    checkpoint(cx)?;
    Ok(policy)
}

pub(super) fn configured(policy: &query::Policy) -> bool {
    policy.full_query.is_some() || policy.rerank.is_some()
}

/// Route every semantic mode through the same retained selection and optional
/// caller pool. The worker owns vector traversal; inference, lexical work and
/// result delivery keep their existing ownership and deadline boundaries.
pub(super) struct RetainedQuery<'a> {
    snapshot: &'a NativeGenerationSnapshot,
    scope: Option<&'a NativeScopedGenerationSnapshot>,
    pool: Option<BlockingPoolHandle>,
}

impl<'a> RetainedQuery<'a> {
    pub(super) fn new(
        snapshot: &'a NativeGenerationSnapshot,
        scope: Option<&'a NativeScopedGenerationSnapshot>,
        policy: Option<&query::Policy>,
    ) -> Self {
        Self {
            snapshot,
            scope,
            pool: policy.and_then(query::Policy::query_pool),
        }
    }

    pub(super) async fn fast(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>> {
        Ok(match (self.scope, self.pool.clone()) {
            (Some(scope), Some(pool)) => {
                Box::pin(scope.search_text_with_pool(cx, embedder, text, limit, pool)).await?
            }
            (None, Some(pool)) => {
                Box::pin(
                    self.snapshot
                        .search_text_with_pool(cx, embedder, text, limit, pool),
                )
                .await?
            }
            (Some(scope), None) => Box::pin(scope.search_text(cx, embedder, text, limit)).await?,
            (None, None) => Box::pin(self.snapshot.search_text(cx, embedder, text, limit)).await?,
        })
    }

    pub(super) async fn quality(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>> {
        Ok(match (self.scope, self.pool.clone()) {
            (Some(scope), Some(pool)) => {
                Box::pin(scope.search_quality_text_with_pool(cx, embedder, text, limit, pool))
                    .await?
            }
            (None, Some(pool)) => {
                Box::pin(
                    self.snapshot
                        .search_quality_text_with_pool(cx, embedder, text, limit, pool),
                )
                .await?
            }
            (Some(scope), None) => {
                Box::pin(scope.search_quality_text(cx, embedder, text, limit)).await?
            }
            (None, None) => {
                Box::pin(self.snapshot.search_quality_text(cx, embedder, text, limit)).await?
            }
        })
    }

    pub(super) async fn refined(
        &self,
        cx: &Cx,
        fast: &dyn Embedder,
        quality: &dyn Embedder,
        text: &str,
        limit: usize,
    ) -> Result<Vec<ScoredResult>> {
        Ok(match (self.scope, self.pool.clone()) {
            (Some(scope), Some(pool)) => {
                Box::pin(scope.search_refined_text_with_pool(cx, fast, quality, text, limit, pool))
                    .await?
            }
            (None, Some(pool)) => {
                Box::pin(
                    self.snapshot
                        .search_refined_text_with_pool(cx, fast, quality, text, limit, pool),
                )
                .await?
            }
            (Some(scope), None) => {
                Box::pin(scope.search_refined_text(cx, fast, quality, text, limit)).await?
            }
            (None, None) => {
                Box::pin(
                    self.snapshot
                        .search_refined_text(cx, fast, quality, text, limit),
                )
                .await?
            }
        })
    }

    pub(super) fn progressive(
        &self,
        cx: &'a Cx,
        fast: &'a dyn Embedder,
        quality: &'a dyn Embedder,
        text: &'a str,
        limit: usize,
    ) -> Result<NativeProgressiveSearch<'a>> {
        Ok(match (self.scope, self.pool.clone()) {
            (Some(scope), Some(pool)) => scope
                .search_progressive_with_quality_and_pool(cx, fast, quality, text, limit, pool)?,
            (None, Some(pool)) => self
                .snapshot
                .search_progressive_with_quality_and_pool(cx, fast, quality, text, limit, pool)?,
            (Some(scope), None) => {
                scope.search_progressive_with_quality(cx, fast, quality, text, limit)?
            }
            (None, None) => self
                .snapshot
                .search_progressive_with_quality(cx, fast, quality, text, limit)?,
        })
    }
}

pub(super) fn configure<'a>(
    mut search: NativeProgressiveSearch<'a>,
    policy: &'a query::Policy,
    max_source_bytes: usize,
) -> Result<NativeProgressiveSearch<'a>> {
    if let Some(rerank) = &policy.rerank {
        search = search.with_retained_reranker(
            rerank.model.as_ref(),
            rerank.window,
            max_source_bytes,
        )?;
    }
    match &policy.full_query {
        Some(tuning) => tuning.apply_native(search),
        None => Ok(search),
    }
}

pub(super) fn annotate(policy: &query::Policy, mode: QueryMode, response: &mut serde_json::Value) {
    if mode == QueryMode::Full {
        policy.annotate_full_query(crate::Mode::Full, response);
        if let Some(rerank) = &policy.rerank {
            rerank.annotate(response);
        }
    }
}

pub(super) async fn execute(
    session: &AuthoritySession,
    cx: &Cx,
    message: &SearchEnvelope,
    ordinal: u64,
    output: &mut impl Write,
) -> Result<()> {
    checkpoint(cx)?;
    let mode = session
        .models
        .query_mode(&session.receipt, message.request.mode)?;
    if mode != QueryMode::Full {
        return Err(serve_refused("query_tuning_requires_full_mode"));
    }
    let policy = session
        .query_policy
        .for_query_override(cx, &message.full_query)
        .map_err(|error| {
            if error
                .downcast_ref::<SearchError>()
                .is_some_and(|error| matches!(error, SearchError::Cancelled { .. }))
            {
                error
            } else {
                serve_refused("query_tuning_refused")
            }
        })?;
    session
        .query_with_policy(cx, &message.request, ordinal, output, &policy)
        .await
}
