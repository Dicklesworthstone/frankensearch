//! Full-mode tuning and bounded, lazy reranking over one retained authority.
//! The ordinary CLI policy owns validation and request limits; this adapter
//! changes only how that policy reaches a fixed generation's native query.

use frankensearch::native_ann::NativeProgressiveSearch;

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
pub(super) struct Request {
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
    policy.full_query = options.full_query.clone();
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
    message: &Request,
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
