//! Buffered CLI forwarding with a checked configuration and retained resources.
//!
//! The request deliberately has no top-level `query`: an older daemon must
//! refuse it rather than execute the query while ignoring its configuration.
//! Connect/write/read use the existing bounded control transport. The
//! forwarder itself never retrieves locally, starts a daemon, or replays after
//! ambiguous delivery; the search command decides around it whether to start one.

use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use super::super::super::{
    FSFS_DAEMON_RESPONSE_MAX_BYTES, LiveRetainedSearchReader, SearchCacheKey, SearchExecutionMode,
    SearchServeOptions, SearchServeRequest,
};
use super::{
    FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS, FsfsRuntime, PeerOutcome, SearchExecutionFlags,
    complete_cli_error, control, encode_response, pressure_timestamp_ms,
    retained_search_checkpoint, run_request, write_response,
};
use crate::adapters::format_emitter::meta_for_format;
use crate::output_schema::{
    OUTPUT_SCHEMA_VERSION, OutputEnvelope, OutputError, SearchPayload, output_error_from,
};
use crate::{CliCommand, FsfsConfig, OutputFormat};

// Ranking changes invalidate retained peers even when configuration and
// generation identity match. The progressive request shares this version.
// 3: the WAL top-k repair (0dc3df2f).
// 4: request-scoped, retained-reader inline explanations.
// 5: fast scores for quality-discovered Refined documents.
// 6: coverage-scaled quality weight for windowed sources.
// 7: query exclusions apply to the vector lanes.
// 8: `type:`/`lang:` filters accept language names.
// 9: extension filter clauses widen each other.
// 10: colons that name no field reach the lexical lane.
// 11: snippets mask credential tokens and private-key material.
// 12: the search policy travels with each request (`SearchPolicy`).
// 13: shared buffered/progressive cache and configured search delivery budgets.
const VERSION: u32 = 13;
static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[path = "complete_daemon_forward_stream.rs"]
mod progressive;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForwardedSearch {
    fsfs_complete_cli: u32,
    request_id: String,
    search: SearchRequest,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchRequest {
    query: String,
    limit: usize,
    filter: Option<String>,
    explain: bool,
    store_root: PathBuf,
    configuration: serde_json::Value,
    policy: SearchPolicy,
}

impl SearchRequest {
    /// The policy is already installed on the per-request runtime. Omitted
    /// serve overrides preserve that exact policy in the common executor.
    fn as_serve_request(&self, mode: SearchExecutionMode) -> SearchServeRequest {
        SearchServeRequest {
            query: self.query.clone(),
            limit: Some(self.limit),
            mode: Some(mode.label().to_owned()),
            filter: self.filter.clone(),
            rerank: None,
            rerank_timeout_ms: None,
            quality_weight: None,
            quality_timeout_ms: None,
            rrf_k: None,
            fast_only: None,
        }
    }
}

/// The search knobs a warm daemon applies per request, as the legacy query
/// daemon applies its request policy: they change how one query is ranked,
/// not which models, generation or privacy rules serve it. Changing one in a
/// config file or with `--fast-only`/`--rerank` therefore needs no other
/// daemon. They are left out of the configuration contract, which stays
/// exact for everything else.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchPolicy {
    quality_weight: f64,
    rrf_k: f64,
    quality_timeout_ms: u64,
    fast_only: bool,
    rerank: bool,
    rerank_timeout_ms: u64,
}

impl SearchPolicy {
    /// The `[search]` keys this policy carries.
    const KEYS: [&'static str; 6] = [
        "quality_weight",
        "rrf_k",
        "quality_timeout_ms",
        "fast_only",
        "rerank",
        "rerank_timeout_ms",
    ];

    const fn of(config: &FsfsConfig) -> Self {
        Self {
            quality_weight: config.search.quality_weight,
            rrf_k: config.search.rrf_k,
            quality_timeout_ms: config.search.quality_timeout_ms,
            fast_only: config.search.fast_only,
            rerank: config.search.rerank,
            rerank_timeout_ms: config.search.rerank_timeout_ms,
        }
    }

    fn delivery_budget(self) -> Duration {
        FsfsRuntime::search_daemon_delivery_budget(
            self.quality_timeout_ms,
            self.rerank,
            self.rerank_timeout_ms,
        )
    }

    /// Set this policy on `runtime`, which then searches with it. A value a
    /// config file could not hold is refused, as it would be there.
    fn apply_to(self, runtime: &mut FsfsRuntime) -> SearchResult<()> {
        let mut search = runtime.config.search.clone();
        search.quality_weight = self.quality_weight;
        search.rrf_k = self.rrf_k;
        search.quality_timeout_ms = self.quality_timeout_ms;
        search.fast_only = self.fast_only;
        search.rerank = self.rerank;
        search.rerank_timeout_ms = self.rerank_timeout_ms;
        crate::config::validate_search_policy(&search)?;
        runtime.config.search = search;
        runtime.cli_input.overrides.fast_only = Some(self.fast_only);
        runtime.cli_input.overrides.rerank = Some(self.rerank);
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reply {
    fsfs_complete_cli: u32,
    request_id: String,
    query: String,
    result: OutputEnvelope<SearchPayload>,
}

/// Preserve the peer's canonical error as a source, not just the JSON parser's
/// failure or a generic "daemon unavailable" message. Returning this error
/// never authorizes the caller to re-run retrieval in the client process.
#[derive(Debug)]
struct RemoteSearchFailure(OutputError);

impl fmt::Display for RemoteSearchFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "complete-generation daemon refused search: {:?}", self.0)
    }
}

impl std::error::Error for RemoteSearchFailure {}

fn codec_error(error: serde_json::Error) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.daemon_protocol",
        source: Box::new(error),
    }
}

fn configuration_contract(config: &FsfsConfig) -> SearchResult<serde_json::Value> {
    let mut value = serde_json::to_value(config).map_err(codec_error)?;
    // Limit and explanations are explicit request arguments, not properties
    // of warmed models. An ordinary daemon can explain a particular request.
    // The search policy travels with each request and is applied to it.
    // Everything else remains exact: in particular, no model, producer,
    // privacy, or pressure-policy mismatch is silently ignored.
    if let Some(search) = value
        .get_mut("search")
        .and_then(serde_json::Value::as_object_mut)
    {
        search.remove("default_limit");
        search.remove("explain");
        for key in SearchPolicy::KEYS {
            search.remove(key);
        }
    }
    Ok(value)
}

pub(super) fn is_forwarded(bytes: &[u8]) -> bool {
    progressive::is_streamed(bytes)
        || serde_json::from_slice::<serde_json::Value>(bytes)
            .is_ok_and(|value| value.get("fsfs_complete_cli").is_some())
}

fn decode_request(bytes: &[u8]) -> SearchResult<ForwardedSearch> {
    // Decode the original bytes so duplicate fields cannot become last-wins
    // authority. Unknown fields, including attempted control/stream commands,
    // are rejected rather than stripped and sent through another path.
    let request: ForwardedSearch = serde_json::from_slice(bytes).map_err(codec_error)?;
    if request.fsfs_complete_cli != VERSION
        || request.request_id.is_empty()
        || request.request_id.len() > 128
    {
        return Err(complete_cli_error(
            "daemon_protocol",
            "invalid CLI protocol version or request id",
        ));
    }
    if request.search.limit == 0 {
        return Err(complete_cli_error(
            "daemon_limit",
            "forward the resolved positive search limit, not the CLI's zero/unlimited spelling",
        ));
    }
    Ok(request)
}

fn validate_request(
    runtime: &FsfsRuntime,
    store_root: &Path,
    request: &ForwardedSearch,
) -> SearchResult<()> {
    if request.search.store_root != store_root {
        return Err(complete_cli_error(
            "daemon_root",
            "peer serves a different complete-generation store",
        ));
    }
    if request.search.configuration != configuration_contract(runtime.config())? {
        return Err(complete_cli_error(
            "daemon_config",
            "client and daemon configurations differ; start the daemon with the same configuration or use --no-daemon; no settings were discarded and no fallback was attempted",
        ));
    }
    Ok(())
}

fn make_request(
    runtime: &FsfsRuntime,
    root: &Path,
    query: &str,
    limit: usize,
) -> SearchResult<ForwardedSearch> {
    if runtime
        .cli_input
        .overrides
        .fast_only
        .is_some_and(|value| value != runtime.config.search.fast_only)
        || runtime
            .cli_input
            .overrides
            .rerank
            .is_some_and(|value| value != runtime.config.search.rerank)
    {
        return Err(complete_cli_error(
            "daemon_config",
            "search overrides must be resolved into the runtime configuration before forwarding",
        ));
    }
    let sequence = REQUEST_SEQUENCE
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| complete_cli_error("daemon_protocol", "request sequence exhausted"))?;
    Ok(ForwardedSearch {
        fsfs_complete_cli: VERSION,
        request_id: format!(
            "cli-{}-{}-{sequence}",
            std::process::id(),
            pressure_timestamp_ms()
        ),
        search: SearchRequest {
            query: query.to_owned(),
            limit,
            filter: runtime.cli_input.filter.clone(),
            explain: runtime.config.search.explain,
            store_root: std::fs::canonicalize(root)?,
            configuration: configuration_contract(runtime.config())?,
            policy: SearchPolicy::of(runtime.config()),
        },
    })
}

fn decode_reply(bytes: &[u8], request: &ForwardedSearch) -> SearchResult<SearchPayload> {
    let reply: Reply = serde_json::from_slice(bytes).map_err(codec_error)?;
    // Every daemon refuses a request of another protocol version before
    // searching, and answers in its own: another fsfs build started it.
    if reply.fsfs_complete_cli != VERSION {
        return Err(SearchError::InvalidConfig {
            field: "complete_generation.daemon_version".to_owned(),
            value: reply.fsfs_complete_cli.to_string(),
            reason: format!(
                "the query daemon was started by another fsfs build (protocol {}; this client speaks {VERSION}); stop it with `fsfs daemon --stop` or search with --no-daemon",
                reply.fsfs_complete_cli
            ),
        });
    }
    if reply.request_id != request.request_id
        || reply.query != request.search.query
        || reply.result.v != OUTPUT_SCHEMA_VERSION
        || reply.result.meta.command != "search"
    {
        return Err(complete_cli_error(
            "daemon_response",
            "response identity or schema does not match this search",
        ));
    }
    match (reply.result.ok, reply.result.data, reply.result.error) {
        (true, Some(payload), None) => Ok(payload),
        (false, None, Some(error)) if (1..=255).contains(&error.exit_code) => {
            Err(SearchError::SubsystemError {
                subsystem: "fsfs.complete_generation.remote_search",
                source: Box::new(RemoteSearchFailure(error)),
            })
        }
        _ => Err(complete_cli_error(
            "daemon_response",
            "inconsistent search response envelope or query",
        )),
    }
}

impl FsfsRuntime {
    /// Recover a validated forwarding reply's structured error so callers
    /// can preserve its exit status, error code and recovery guidance.
    /// Transport failures and unrelated subsystem errors return `None`.
    #[must_use]
    pub fn forwarded_search_error(error: &SearchError) -> Option<&OutputError> {
        if let Some(error) = progressive::reported_error(error) {
            return Some(error);
        }
        match error {
            SearchError::SubsystemError { subsystem, source }
                if *subsystem == "fsfs.complete_generation.remote_search" =>
            {
                source
                    .downcast_ref::<RemoteSearchFailure>()
                    .map(|failure| &failure.0)
            }
            _ => None,
        }
    }

    /// Whether `error` is a daemon's refusal of a forwarded search because it
    /// runs with another configuration or was started by another fsfs build
    /// (another protocol version). Either refuses before searching, and a
    /// refused request has delivered no output, so the search may still be
    /// served elsewhere.
    pub(crate) fn is_stale_daemon_refusal(error: &SearchError) -> bool {
        match error {
            SearchError::InvalidConfig { field, .. } => {
                field == "complete_generation.daemon_version"
            }
            SearchError::SubsystemError { subsystem, source }
                if *subsystem == "fsfs.complete_generation.remote_search" =>
            {
                source
                    .downcast_ref::<RemoteSearchFailure>()
                    .is_some_and(|failure| {
                        failure.0.field.as_deref() == Some("complete_generation.daemon_config")
                    })
            }
            _ => false,
        }
    }

    /// True when a forwarded stream already emitted its terminal record, or
    /// its output writer failed after possibly exposing part of a record.
    #[must_use]
    pub fn forwarded_search_error_was_emitted(error: &SearchError) -> bool {
        progressive::reported_error(error).is_some()
    }

    /// Query an already-running complete-generation daemon without opening the
    /// catalog, vectors or models in this client. Failure is final for this
    /// request, including malformed replies and an absent socket.
    ///
    /// # Errors
    /// Returns transport, cancellation, protocol or the remote search error.
    pub(crate) async fn query_complete_generation_daemon(
        &self,
        cx: &Cx,
        root: &Path,
        query: &str,
        limit: usize,
    ) -> SearchResult<SearchPayload> {
        retained_search_checkpoint(cx)?;
        if self.cli_input.stream || self.cli_input.expand {
            return Err(complete_cli_error(
                "daemon_options",
                "CLI forwarding currently supports buffered search without expansion; use --no-daemon for --stream",
            ));
        }
        let request = make_request(self, root, query, limit)?;
        let mut bytes = serde_json::to_vec(&request).map_err(codec_error)?;
        bytes.write_all(b"\n")?;
        let socket = self.complete_generation_socket_path(root)?;
        let started = Instant::now();
        let mut peer = control::connect_and_send(
            cx,
            &socket,
            &bytes,
            started,
            Duration::from_millis(FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS),
        )
        .await?;
        let response = control::read_response(
            cx,
            &mut peer,
            FSFS_DAEMON_RESPONSE_MAX_BYTES,
            control::remaining(started, request.search.policy.delivery_budget())?,
        )
        .await?;
        let payload = decode_reply(&response, &request)?;
        retained_search_checkpoint(cx)?;
        Ok(payload)
    }
}

fn query_runtime(
    session: &LiveRetainedSearchReader,
    request: &SearchRequest,
    stream: bool,
) -> SearchResult<FsfsRuntime> {
    // Clone the *admitted* runtime, not the store-root runtime. Hydration and
    // retrieval must point at the same pinned bundle. The filter is request
    // scoped; None explicitly clears any filter from daemon startup.
    let mut query_runtime = session.reader.runtime.clone();
    query_runtime.enable_complete_generation_explanations(
        session.store.root(),
        session.reader.generation(),
    )?;
    query_runtime.cli_input.command = CliCommand::Search;
    query_runtime.cli_input.query = Some(request.query.clone());
    query_runtime.cli_input.filter.clone_from(&request.filter);
    query_runtime.cli_input.daemon = false;
    query_runtime.cli_input.daemon_socket = None;
    query_runtime.cli_input.stream = stream;
    if stream {
        query_runtime.cli_input.format = OutputFormat::Jsonl;
    }
    query_runtime.cli_input.expand = false;
    query_runtime.cli_input.overrides.limit = Some(request.limit);
    request.policy.apply_to(&mut query_runtime)?;
    query_runtime.config.search.explain = request.explain;
    query_runtime.cli_input.overrides.explain = Some(request.explain);
    Ok(query_runtime)
}

async fn execute(
    cx: &Cx,
    runtime: &FsfsRuntime,
    session: &mut LiveRetainedSearchReader,
    request: &ForwardedSearch,
    cache: Option<&mut HashMap<SearchCacheKey, Vec<SearchPayload>>>,
) -> SearchResult<SearchPayload> {
    validate_request(runtime, session.store.root(), request)?;
    let cache_enabled = cache.is_some();
    let mut disabled_cache = HashMap::new();
    let cache = cache.unwrap_or(&mut disabled_cache);
    if session.refresh(cx).await? {
        // A forwarded request can be the first to observe publication. Clear
        // the old generation before either consulting or filling the shared
        // cache; the next raw request may otherwise see no refresh at all.
        cache.clear();
    }
    let query_runtime = query_runtime(session, &request.search, false)?;
    // The daemon applies its own degradation override to every client, as
    // the legacy query daemon does.
    let mode = session.reader.search_mode()?;
    if !request.search.explain {
        // Use the ordinary daemon executor so both transports share complete
        // phase payloads, producer checks, cache keys, bounded admission and
        // the refusal to cache failed refinement. The request policy is already
        // installed on query_runtime; omitted serve overrides preserve it.
        let response = query_runtime
            .execute_search_serve_request_with_sink(
                cx,
                request.search.as_serve_request(mode),
                &mut session.reader.resources,
                cache,
                SearchServeOptions {
                    hot_cache_enabled: cache_enabled,
                    persist_explain_session: true,
                },
                None,
            )
            .await?;
        return response.payloads.into_iter().last().ok_or_else(|| {
            complete_cli_error(
                "daemon_response",
                "search completed without an Initial phase",
            )
        });
    }
    // Inline explanations require the actual per-query artifacts. They neither
    // reuse plain cached payloads nor contaminate them with explanation output.
    let mut phases = Box::pin(
        query_runtime.execute_search_phase_artifacts_with_mode_using_resources(
            cx,
            &request.search.query,
            request.search.limit,
            mode,
            &mut session.reader.resources,
            SearchExecutionFlags {
                include_snippets: true,
                persist_explain_session: true,
            },
            None,
        ),
    )
    .await?;
    let mut last = phases.pop().ok_or_else(|| {
        complete_cli_error(
            "daemon_response",
            "search completed without an Initial phase",
        )
    })?;
    query_runtime.attach_complete_search_explanations(cx, &mut last, &session.reader.resources)?;
    Ok(last.payload)
}

pub(super) async fn serve(
    cx: &Cx,
    runtime: &FsfsRuntime,
    session: &mut LiveRetainedSearchReader,
    peer: &mut UnixStream,
    bytes: &[u8],
    timeout: Duration,
    cache: Option<&mut HashMap<SearchCacheKey, Vec<SearchPayload>>>,
) -> SearchResult<PeerOutcome> {
    if progressive::is_streamed(bytes) {
        return Box::pin(progressive::serve(
            cx, runtime, session, peer, bytes, timeout, cache,
        ))
        .await;
    }
    let request = decode_request(bytes);
    let request_id = request
        .as_ref()
        .map_or_else(|_| String::new(), |value| value.request_id.clone());
    let query = request
        .as_ref()
        .map_or_else(|_| String::new(), |value| value.search.query.clone());
    let result = match request {
        Ok(request) => {
            run_request(
                cx,
                request.search.policy.delivery_budget(),
                execute(cx, runtime, session, &request, cache),
            )
            .await
        }
        Err(error) => Err(error),
    };
    if matches!(result, Err(SearchError::Cancelled { .. })) {
        return result.map(|_| PeerOutcome::Search);
    }
    let envelope = match result {
        Ok(payload) => {
            let warnings = FsfsRuntime::search_generation_warnings(&payload);
            OutputEnvelope::success(
                payload,
                meta_for_format("search", OutputFormat::Json),
                super::super::iso_timestamp_now(),
            )
            .with_warnings(warnings)
        }
        Err(error) => error_envelope(&error),
    };
    let reply = Reply {
        fsfs_complete_cli: VERSION,
        request_id,
        query,
        result: envelope,
    };
    let encoded = match encode_response(&reply) {
        Ok(encoded) => encoded,
        Err(error) => encode_response(&Reply {
            result: error_envelope(&error),
            ..reply
        })?,
    };
    write_response(cx, peer, &encoded, timeout).await?;
    Ok(PeerOutcome::Search)
}

fn error_envelope(error: &SearchError) -> OutputEnvelope<SearchPayload> {
    OutputEnvelope::error(
        output_error_from(error),
        meta_for_format("search", OutputFormat::Json),
        super::super::iso_timestamp_now(),
    )
}

#[cfg(test)]
mod remote_error_tests {
    use super::*;

    fn error_reply(exit_code: Option<i32>) -> (Reply, ForwardedSearch) {
        let root = tempfile::tempdir().unwrap();
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let request = make_request(&runtime, root.path(), "query", 10).unwrap();
        let original = complete_cli_error("test_remote", "original remote diagnosis");
        let mut envelope = error_envelope(&original);
        if let Some(exit_code) = exit_code {
            envelope.error.as_mut().unwrap().exit_code = exit_code;
        }
        let reply = Reply {
            fsfs_complete_cli: VERSION,
            request_id: request.request_id.clone(),
            query: request.search.query.clone(),
            result: envelope,
        };
        (reply, request)
    }

    #[test]
    fn forwarded_error_preserves_the_complete_canonical_error() {
        let (reply, request) = error_reply(None);
        let expected = serde_json::to_value(reply.result.error.as_ref().unwrap()).unwrap();
        let error = decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).unwrap_err();
        let original = FsfsRuntime::forwarded_search_error(&error).unwrap();
        assert_eq!(serde_json::to_value(original).unwrap(), expected);
        assert!(error.to_string().contains("original remote diagnosis"));
    }

    #[test]
    fn failed_remote_reply_cannot_claim_success_or_out_of_range_exit_status() {
        for exit_code in [0, -1, 256] {
            let (reply, request) = error_reply(Some(exit_code));
            let error = decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { .. }));
            assert!(FsfsRuntime::forwarded_search_error(&error).is_none());
        }
    }

    #[test]
    fn subsystem_name_alone_does_not_authorize_a_remote_error() {
        let error = SearchError::SubsystemError {
            subsystem: "fsfs.complete_generation.remote_search",
            source: Box::new(std::io::Error::other("not a decoded forwarding reply")),
        };
        assert!(FsfsRuntime::forwarded_search_error(&error).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::time::Instant;

    fn request(root: &Path) -> (FsfsRuntime, ForwardedSearch) {
        let runtime = FsfsRuntime::new(FsfsConfig::default());
        let request = make_request(&runtime, root, "sharedtoken", 10).unwrap();
        (runtime, request)
    }

    #[test]
    fn forwarded_request_has_no_legacy_query_and_preserves_filter_and_limit() {
        let root = tempfile::tempdir().unwrap();
        let (mut runtime, _) = request(root.path());
        runtime.cli_input.filter = Some("ext:rs".to_owned());
        let request =
            make_request(&runtime, root.path(), "query\nwith newline", usize::MAX).unwrap();
        let bytes = serde_json::to_vec(&request).unwrap();
        assert!(is_forwarded(&bytes));
        assert!(
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .unwrap()
                .get("query")
                .is_none()
        );
        assert!(
            FsfsRuntime::parse_search_serve_request(std::str::from_utf8(&bytes).unwrap()).is_err()
        );
        let decoded = decode_request(&bytes).unwrap();
        assert_eq!(decoded.search.filter.as_deref(), Some("ext:rs"));
        assert_eq!(decoded.search.limit, usize::MAX);
        assert_eq!(decoded.search.query, "query\nwith newline");
        validate_request(
            &runtime,
            &std::fs::canonicalize(root.path()).unwrap(),
            &decoded,
        )
        .unwrap();
    }

    #[test]
    fn forwarded_request_rejects_duplicate_unknown_and_unsupported_fields() {
        let root = tempfile::tempdir().unwrap();
        let (_, request) = request(root.path());
        let text = serde_json::to_string(&request).unwrap();
        let duplicate = format!("{{\"fsfs_complete_cli\":{VERSION},{}", &text[1..]);
        assert!(decode_request(duplicate.as_bytes()).is_err());
        for (field, value) in [
            ("stream", serde_json::json!(true)),
            ("fsfs_complete_daemon", serde_json::json!("shutdown")),
        ] {
            let mut value_map = serde_json::to_value(&request).unwrap();
            value_map[field] = value;
            assert!(decode_request(&serde_json::to_vec(&value_map).unwrap()).is_err());
        }
        let mut value = serde_json::to_value(&request).unwrap();
        // Earlier versions cannot acknowledge request-scoped explanations,
        // 4 ranks quality-discovered documents without fast scores, 5
        // gives long windowed sources the whole quality weight, 6 lets
        // excluded documents back through the vector lanes and 7 reads
        // `lang:rust` as a file extension.
        for unsupported in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, VERSION + 1] {
            value["fsfs_complete_cli"] = serde_json::json!(unsupported);
            assert!(decode_request(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        value["fsfs_complete_cli"] = serde_json::json!(VERSION);
        value["search"]["limit"] = serde_json::json!(0);
        assert!(decode_request(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn daemon_contract_refuses_different_models_ranking_privacy_and_store() {
        let root = tempfile::tempdir().unwrap();
        let (runtime, mut request) = request(root.path());
        let canonical_root = std::fs::canonicalize(root.path()).unwrap();
        validate_request(&runtime, &canonical_root, &request).unwrap();
        let original = request.search.configuration.clone();
        for section in ["indexing", "search", "privacy", "pressure", "storage"] {
            request.search.configuration = original.clone();
            request.search.configuration[section] = serde_json::Value::Null;
            assert!(
                matches!(validate_request(&runtime, &canonical_root, &request),
                    Err(SearchError::InvalidConfig { field, .. })
                        if field == "complete_generation.daemon_config"),
                "{section}"
            );
        }
        request.search.configuration = original;
        request.search.store_root = root.path().join("different-store");
        assert!(
            matches!(validate_request(&runtime, &canonical_root, &request),
            Err(SearchError::InvalidConfig { field, .. })
                if field == "complete_generation.daemon_root")
        );
    }

    /// The resolved limit and the per-request search policy are not part of
    /// the warmed configuration contract; the policy travels in the request
    /// instead. A model choice is.
    #[test]
    fn limit_and_search_policy_are_not_part_of_the_warmed_configuration_contract() {
        let first = FsfsConfig::default();
        let mut second = first.clone();
        second.search.default_limit = 123;
        second.search.fast_only = !first.search.fast_only;
        second.search.rerank = !first.search.rerank;
        second.search.quality_weight = 0.25;
        second.search.rrf_k = 17.0;
        second.search.quality_timeout_ms = first.search.quality_timeout_ms + 100;
        second.search.rerank_timeout_ms = first.search.rerank_timeout_ms + 100;
        assert_eq!(
            configuration_contract(&first).unwrap(),
            configuration_contract(&second).unwrap()
        );
        assert_ne!(SearchPolicy::of(&first), SearchPolicy::of(&second));
        second.indexing.quality_model = "another-quality-model".to_owned();
        assert_ne!(
            configuration_contract(&first).unwrap(),
            configuration_contract(&second).unwrap()
        );
    }

    /// Applying a request's policy sets each knob (and the fast-only and
    /// rerank overrides) on the per-request runtime; a value outside its
    /// config-file range is refused and changes nothing.
    #[test]
    fn a_request_policy_is_applied_or_refused_whole() {
        let mut runtime = FsfsRuntime::new(FsfsConfig::default());
        let mut policy = SearchPolicy::of(&runtime.config);
        policy.quality_weight = 0.25;
        policy.rrf_k = 17.0;
        policy.fast_only = true;
        policy.rerank = true;
        policy.quality_timeout_ms = 2_000;
        policy.rerank_timeout_ms = 10_000;
        policy.apply_to(&mut runtime).unwrap();
        assert_eq!(SearchPolicy::of(&runtime.config), policy);
        assert_eq!(runtime.cli_input.overrides.fast_only, Some(true));
        assert_eq!(runtime.cli_input.overrides.rerank, Some(true));
        // Delivery includes both requested stages instead of inheriting the
        // five-second framing timeout. Disabled reranking contributes no wait.
        assert_eq!(policy.delivery_budget(), Duration::from_secs(42));
        assert_eq!(
            SearchPolicy {
                rerank: false,
                ..policy
            }
            .delivery_budget(),
            Duration::from_secs(32)
        );

        let before = SearchPolicy::of(&runtime.config);
        for (field, invalid) in [
            (
                "search.quality_weight",
                SearchPolicy {
                    quality_weight: 1.5,
                    ..policy
                },
            ),
            (
                "search.rrf_k",
                SearchPolicy {
                    rrf_k: f64::NAN,
                    ..policy
                },
            ),
            (
                "search.quality_timeout_ms",
                SearchPolicy {
                    quality_timeout_ms: 1,
                    ..policy
                },
            ),
        ] {
            let error = invalid.apply_to(&mut runtime).unwrap_err();
            assert!(
                matches!(&error, SearchError::InvalidConfig { field: refused, .. }
                    if refused == field),
                "{field}: {error:?}"
            );
            assert_eq!(SearchPolicy::of(&runtime.config), before, "{field}");
        }
    }

    #[test]
    fn remote_failure_retains_canonical_diagnostic_and_rejects_reply_misassociation() {
        let root = tempfile::tempdir().unwrap();
        let (_, request) = request(root.path());
        let original = complete_cli_error("test_producer", "wrong semantic producer: fixture");
        let mut reply = Reply {
            fsfs_complete_cli: VERSION,
            request_id: request.request_id.clone(),
            query: request.search.query.clone(),
            result: error_envelope(&original),
        };
        let error = decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("wrong semantic producer: fixture")
        );
        assert!(std::error::Error::source(&error).is_some());
        // A reply in another protocol version names another fsfs build's
        // daemon, which a plain search may replace.
        for stale in (1..VERSION).chain([VERSION + 1]) {
            reply.fsfs_complete_cli = stale;
            let error = decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).unwrap_err();
            assert!(
                matches!(&error, SearchError::InvalidConfig { field, value, .. }
                    if field == "complete_generation.daemon_version" && *value == stale.to_string()),
                "{stale}: {error:?}"
            );
            assert!(FsfsRuntime::is_stale_daemon_refusal(&error), "{stale}");
        }
        assert!(
            !FsfsRuntime::is_stale_daemon_refusal(&original),
            "an ordinary refusal is not a stale daemon"
        );
        reply.fsfs_complete_cli = VERSION;
        reply.request_id.push('x');
        assert!(
            matches!(decode_reply(&serde_json::to_vec(&reply).unwrap(), &request),
            Err(SearchError::InvalidConfig { field, .. }) if field == "complete_generation.daemon_response")
        );
        reply.request_id.clone_from(&request.request_id);
        reply.result.ok = true;
        assert!(decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).is_err());
        reply.result.ok = false;
        reply.result.v += 1;
        assert!(decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).is_err());
    }

    #[test]
    fn absent_daemon_never_opens_or_repairs_a_corrupt_store_in_the_client() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let pointer = root
                .path()
                .join(crate::generation_store::COMPLETE_GENERATION_POINTER);
            std::fs::write(&pointer, "do not repair").unwrap();
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            let error = runtime
                .query_complete_generation_daemon(&cx, root.path(), "x", 10)
                .await
                .unwrap_err();
            assert!(
                matches!(error, SearchError::Io(ref source) if source.kind() == std::io::ErrorKind::NotFound)
            );
            assert_eq!(std::fs::read_to_string(pointer).unwrap(), "do not repair");
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        });
    }

    #[test]
    fn malformed_peer_response_is_not_retried_or_followed_by_local_retrieval() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let runtime = FsfsRuntime::new(FsfsConfig::default());
            let path = runtime
                .complete_generation_socket_path(root.path())
                .unwrap();
            let listener = UnixListener::bind(&path).unwrap();
            listener.set_nonblocking(true).unwrap();
            let worker = std::thread::spawn(move || {
                let started = Instant::now();
                let mut peer = loop {
                    match listener.accept() {
                        Ok((peer, _)) => break peer,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(started.elapsed() < Duration::from_secs(5));
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("accept: {error}"), // ubs:ignore — cfg(test) assertion.
                    }
                };
                peer.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                peer.set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                // Consume the actual single framed request before returning a
                // corrupt reply. Returning closes the one owned listener too.
                let mut line = Vec::new();
                loop {
                    let mut byte = [0];
                    peer.read_exact(&mut byte).unwrap();
                    line.push(byte[0]);
                    if byte[0] == b'\n' {
                        break;
                    }
                }
                assert!(decode_request(&line).is_ok());
                peer.write_all(b"{corrupt reply}\n").unwrap();
            });
            let result = runtime
                .query_complete_generation_daemon(&cx, root.path(), "x", 10)
                .await;
            worker.join().unwrap();
            assert!(
                matches!(result, Err(SearchError::SubsystemError { subsystem, .. })
                if subsystem == "fsfs.complete_generation.daemon_protocol")
            );
            assert!(!root.path().join("generations").exists());
        });
    }
}

#[cfg(all(test, not(feature = "embedded-models")))]
mod generation_tests {
    use super::*;
    use crate::CliInput;
    use crate::generation_store::{COMPLETE_GENERATION_POINTER, GenerationPublication};
    use crate::output_schema::SearchHitPayload;
    use crate::stream_protocol::{
        StreamEvent, StreamFrame, StreamTerminalStatus, validate_stream_frame,
    };
    use asupersync::test_utils::run_test_with_cx;
    use std::future::{Future, poll_fn};
    use std::pin::pin;
    use std::task::Poll;

    /// Drive the real daemon command and a client future on the same
    /// test-owned context. No worker is detached, the daemon is cancelled once
    /// the client finishes, and a regression cannot wait indefinitely.
    async fn drive_with_daemon<F>(cx: &Cx, runtime: &FsfsRuntime, root: &Path, client_work: F)
    where
        F: Future<Output = SearchResult<()>>,
    {
        let mut server = Box::pin(runtime.run_complete_generation_daemon(cx, root));
        let mut client_work = Box::pin(client_work);
        let mut deadline = pin!(asupersync::time::sleep(cx.now(), Duration::from_secs(30)));
        let mut server_result = None;
        let mut client_result = None;
        poll_fn(|task| {
            assert!(
                !deadline.as_mut().poll(task).is_ready(),
                "forwarding lifecycle exceeded deadline"
            );
            if server_result.is_none()
                && let Poll::Ready(result) = server.as_mut().poll(task)
            {
                assert!(
                    client_result.is_some(),
                    "daemon stopped before client: {result:?}"
                );
                server_result = Some(result);
            }
            if client_result.is_none()
                && let Poll::Ready(result) = client_work.as_mut().poll(task)
            {
                client_result = Some(result);
                cx.set_cancel_requested(true);
            }
            if client_result.is_some() && server_result.is_some() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await;
        drop(client_work);
        drop(server);
        cx.set_cancel_requested(false);
        client_result.unwrap().unwrap();
        assert!(matches!(
            server_result.unwrap(),
            Err(SearchError::Cancelled { .. })
        ));
    }

    /// One buffered serve request, as `fsfs serve` clients send it: the hit
    /// count of its last phase and whether the daemon answered from its cache.
    async fn raw_search(cx: &Cx, endpoint: &Path) -> SearchResult<(usize, bool)> {
        let reply = control::exchange(
            cx,
            endpoint,
            b"{\"query\":\"sharedtoken\",\"limit\":10}\n",
            FSFS_DAEMON_RESPONSE_MAX_BYTES,
            Duration::from_secs(10),
        )
        .await?;
        let value: serde_json::Value = serde_json::from_slice(&reply).map_err(codec_error)?;
        assert_eq!(value["ok"], true, "{value}");
        let hits = value["payloads"].as_array().unwrap().last().unwrap()["hits"]
            .as_array()
            .unwrap()
            .len();
        Ok((hits, value["cached"] == true))
    }

    async fn stream_search(
        cx: &Cx,
        client: &FsfsRuntime,
        root: &Path,
    ) -> SearchResult<(usize, bool)> {
        let mut client = client.clone();
        client.cli_input.stream = true;
        client.cli_input.format = OutputFormat::Jsonl;
        let mut output = Vec::new();
        client
            .stream_complete_generation_daemon(cx, root, "sharedtoken", 10, &mut output)
            .await?;
        let frames = std::str::from_utf8(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<StreamFrame<SearchHitPayload>>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            frames.first().unwrap().event,
            StreamEvent::Started(_)
        ));
        assert!(
            matches!(frames.last().unwrap().event, StreamEvent::Terminal(ref terminal)
            if terminal.status == StreamTerminalStatus::Completed)
        );
        for (sequence, frame) in frames.iter().enumerate() {
            assert!(validate_stream_frame(frame).valid);
            assert_eq!(frame.seq, u64::try_from(sequence).unwrap());
            assert_eq!(frame.stream_id, frames[0].stream_id);
        }
        let hits = frames
            .iter()
            .filter(|frame| matches!(frame.event, StreamEvent::Result(_)))
            .count();
        let cached = frames.iter().any(|frame| {
            matches!(&frame.event,
            StreamEvent::Progress(progress) if progress.reason_code == "daemon_cache_hit")
        });
        Ok((hits, cached))
    }

    #[test]
    fn forwarded_searches_fill_the_serve_cache_and_replace_it_with_the_generation() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source");
            let root = directory.path().join("store");
            std::fs::create_dir(&source).unwrap();
            std::fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
            let mut config = FsfsConfig::default();
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            config.indexing.offline = true;
            config.indexing.quality_model.clear();
            config.search.fast_only = true;
            config.search.rerank = false;
            let runtime = FsfsRuntime::new(config.clone()).with_cli_input(CliInput {
                command: CliCommand::Daemon,
                target_path: Some(source.clone()),
                index_dir: Some(root.clone()),
                quiet: true,
                ..CliInput::default()
            });
            assert!(matches!(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
                GenerationPublication::Durable(_)
            ));
            let endpoint = runtime.complete_generation_socket_path(&root).unwrap();
            let client = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Search,
                index_dir: Some(root.clone()),
                daemon: true,
                format: OutputFormat::Json,
                ..CliInput::default()
            });
            let caching = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
            let client_work = async {
                for _ in 0..1_000 {
                    if endpoint.exists() {
                        break;
                    }
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
                assert!(endpoint.exists(), "daemon did not become ready");
                let forwarded = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await?;
                assert_eq!(forwarded.hits.len(), 1);
                assert_eq!(
                    raw_search(&cx, &endpoint).await?,
                    (1, caching),
                    "the first forwarded search must populate the ordinary serve cache"
                );
                assert_eq!(raw_search(&cx, &endpoint).await?, (1, caching));
                assert_eq!(stream_search(&cx, &client, &root).await?, (1, caching));

                std::fs::write(source.join("beta.md"), "sharedtoken beta document")?;
                assert!(matches!(
                    runtime.rebuild_retained_generation(&cx, &root).await?,
                    GenerationPublication::Durable(_)
                ));
                // The forwarded search moves the session to the new generation,
                // so the buffered lane's own refresh then reports no change.
                let forwarded = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await?;
                assert_eq!(forwarded.hits.len(), 2);
                assert_eq!(
                    raw_search(&cx, &endpoint).await?,
                    (2, caching),
                    "a forwarded publication refresh must replace cached old results with the new generation's results"
                );
                assert_eq!(raw_search(&cx, &endpoint).await?, (2, caching));

                // A cold progressive query also fills the same cache, without
                // losing its Started/Initial/Terminal framing or generation.
                std::fs::write(source.join("gamma.md"), "sharedtoken gamma document")?;
                assert!(matches!(
                    runtime.rebuild_retained_generation(&cx, &root).await?,
                    GenerationPublication::Durable(_)
                ));
                assert_eq!(stream_search(&cx, &client, &root).await?, (3, false));
                assert_eq!(stream_search(&cx, &client, &root).await?, (3, caching));
                assert_eq!(raw_search(&cx, &endpoint).await?, (3, caching));
                let forwarded = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await?;
                assert_eq!(forwarded.hits.len(), 3);
                assert_eq!(raw_search(&cx, &endpoint).await?, (3, caching));

                // The raw lane can be first to observe a later publication too.
                std::fs::write(source.join("delta.md"), "sharedtoken delta document")?;
                assert!(matches!(
                    runtime.rebuild_retained_generation(&cx, &root).await?,
                    GenerationPublication::Durable(_)
                ));
                assert_eq!(raw_search(&cx, &endpoint).await?, (4, false));
                assert_eq!(stream_search(&cx, &client, &root).await?, (4, caching));
                Ok::<(), SearchError>(())
            };
            Box::pin(drive_with_daemon(&cx, &runtime, &root, client_work)).await;
            assert!(!endpoint.exists(), "server must release its owned endpoint");
        });
    }

    #[test]
    fn cli_forwarding_follows_publication_isolates_filters_and_recovers_after_refusal() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source");
            let root = directory.path().join("store");
            std::fs::create_dir(&source).unwrap();
            std::fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
            std::fs::write(source.join("beta.rs"), "// sharedtoken beta document").unwrap();
            let mut config = FsfsConfig::default();
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            config.indexing.offline = true;
            config.indexing.quality_model.clear();
            config.search.fast_only = true;
            config.search.rerank = false;
            let runtime = FsfsRuntime::new(config.clone()).with_cli_input(CliInput {
                command: CliCommand::Daemon,
                target_path: Some(source.clone()),
                index_dir: Some(root.clone()),
                filter: Some("type:md".to_owned()),
                quiet: true,
                ..CliInput::default()
            });
            assert!(matches!(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
                GenerationPublication::Durable(_)
            ));
            let endpoint = runtime.complete_generation_socket_path(&root).unwrap();
            let mut client = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("sharedtoken".to_owned()),
                index_dir: Some(root.clone()),
                daemon: true,
                format: OutputFormat::Json,
                ..CliInput::default()
            });
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let client_work = async {
                for _ in 0..1_000 {
                    if endpoint.exists() {
                        break;
                    }
                    asupersync::time::sleep(cx.now(), Duration::from_millis(1)).await;
                }
                assert!(endpoint.exists(), "daemon did not become ready");
                // A plain raw request warms the shared cache without saving a
                // CLI explanation session. A cached forwarded result must save
                // that context, while --explain still computes its own detail.
                assert_eq!(raw_search(&cx, &endpoint).await?, (2, false));
                assert!(FsfsRuntime::load_explain_session_at_root(&root)?.is_none());
                let cached = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await?;
                let context = FsfsRuntime::load_explain_session_at_root(&root)?.unwrap();
                assert_eq!(context.query, cached.query);
                assert_eq!(context.hits.len(), 2);
                assert!(context.complete_generation.is_some());
                // Exercise the actual CLI dispatch and formatter, not just a
                // JSON fixture or direct invocation of the server search helper.
                // Explanation is per request: this daemon started without it.
                client.config.search.explain = true;
                let mut output = Vec::new();
                client
                    .run_complete_generation_search_with_writer(&cx, &root, &mut output)
                    .await?;
                let envelope: OutputEnvelope<SearchPayload> =
                    serde_json::from_slice(&output).unwrap();
                assert!(envelope.ok);
                let forwarded = envelope.data.unwrap();
                assert_eq!(
                    forwarded.hits.len(),
                    2,
                    "startup filter leaked into request"
                );
                assert_eq!(forwarded.explanations.len(), forwarded.hits.len());
                let mut direct = client.clone();
                direct.cli_input.daemon = false;
                let mut output = Vec::new();
                direct
                    .run_complete_generation_search_with_writer(&cx, &root, &mut output)
                    .await?;
                let direct: OutputEnvelope<SearchPayload> =
                    serde_json::from_slice(&output).unwrap();
                let direct = direct.data.unwrap();
                assert_eq!(forwarded.explanations, direct.explanations);
                assert_eq!(forwarded.explanation_warnings, direct.explanation_warnings);
                client.config.search.explain = false;
                let ordinary = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await?;
                assert!(
                    ordinary.explanations.is_empty(),
                    "previous request's explanations leaked"
                );
                assert!(ordinary.explanation_warnings.is_empty());
                client.cli_input.filter = Some("type:rs".to_owned());
                let filtered = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await?;
                assert_eq!(filtered.hits.len(), 1);
                assert!(filtered.hits[0].path.ends_with("beta.rs"));
                client.cli_input.filter = None;
                assert_eq!(
                    client
                        .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                        .await?
                        .hits
                        .len(),
                    2
                );
                let limited = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 1)
                    .await?;
                assert_eq!(limited.hits.len(), 1);
                let context = FsfsRuntime::load_explain_session_at_root(&root)?.unwrap();
                assert_eq!(context.query, limited.query);
                assert_eq!(context.hits.len(), 1);
                assert_eq!(context.hits[0].path, limited.hits[0].path);
                assert!(context.complete_generation.is_some());
                let mut explain = client.clone();
                explain.cli_input.command = CliCommand::Explain;
                explain.cli_input.result_id = Some("R0".to_owned());
                let mut explanation = Vec::new();
                explain
                    .run_complete_generation_explain_with_writer(&cx, &root, &mut explanation)
                    .await?;
                let explanation: serde_json::Value = serde_json::from_slice(&explanation).unwrap();
                assert_eq!(
                    explanation["data"]["ranking"]["doc_id"],
                    limited.hits[0].path
                );

                // Search knobs travel per request; the rest of the
                // configuration must still match.
                let mut different = runtime.config().clone();
                different.indexing.embedding_batch_size = 32;
                let mismatched =
                    FsfsRuntime::new(different).with_cli_input(client.cli_input.clone());
                let failure = mismatched
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await
                    .unwrap_err();
                assert!(failure.to_string().contains("configurations differ"));

                std::fs::write(source.join("gamma.md"), "sharedtoken gamma document")?;
                assert!(matches!(
                    runtime.rebuild_retained_generation(&cx, &root).await?,
                    GenerationPublication::Durable(_)
                ));
                assert_eq!(
                    client
                        .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                        .await?
                        .hits
                        .len(),
                    3
                );
                let selected = std::fs::read(&pointer)?;
                std::fs::write(&pointer, "corrupt selected generation")?;
                let failed = client
                    .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                    .await;
                assert!(
                    failed.is_err(),
                    "corruption must not yield a cached or locally rebuilt result"
                );
                assert_eq!(
                    std::fs::read_to_string(&pointer)?,
                    "corrupt selected generation"
                );
                std::fs::write(&pointer, selected)?;
                assert_eq!(
                    client
                        .query_complete_generation_daemon(&cx, &root, "sharedtoken", 10)
                        .await?
                        .hits
                        .len(),
                    3
                );
                Ok::<(), SearchError>(())
            };
            Box::pin(drive_with_daemon(&cx, &runtime, &root, client_work)).await;
            assert!(!endpoint.exists(), "server must release its owned endpoint");
            assert!(
                crate::generation_store::CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .is_some()
            );
        });
    }
}
