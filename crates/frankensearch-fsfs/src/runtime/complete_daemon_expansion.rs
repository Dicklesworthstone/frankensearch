//! One client-owned expansion exchange inside an already admitted search.
//!
//! Only query variants cross the socket. The caller keeps its provider choice
//! and credentials; the daemon keeps one retained generation and its warmed
//! models. Ready is queued behind Initial on the same phase writer, so neither
//! backpressure nor a delayed LLM can reorder the public stream.

use std::future::Future;
use std::io::{ErrorKind, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use super::super::{FSFS_DAEMON_REQUEST_MAX_BYTES, IO_POLL_INTERVAL};
use super::{
    ForwardedSearch, VERSION, complete_cli_error, control, retained_search_checkpoint, run_request,
    streaming, write_response,
};
use crate::output_schema::{OutputError, output_error_from};
use crate::query_expansion::{
    ExpandedQuery, ExpansionResult, ExpansionStrategy, MAX_EXPANDED_QUERIES, MAX_VARIANT_BYTES,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ready {
    fsfs_complete_cli_expansion: u32,
    request_id: String,
    query: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Variants {
    fsfs_complete_cli_expansion: u32,
    request_id: String,
    queries: Vec<ExpandedQuery>,
}

fn protocol_error(reason: &str) -> SearchError {
    complete_cli_error("daemon_expansion", reason)
}

pub(super) fn is_ready(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .is_ok_and(|value| value.get("fsfs_complete_cli_expansion").is_some())
}

pub(super) fn validate_ready(bytes: &[u8], request: &ForwardedSearch) -> SearchResult<()> {
    let ready: Ready = serde_json::from_slice(bytes)
        .map_err(|_| protocol_error("malformed expansion readiness record"))?;
    if !request.search.expand
        || ready.fsfs_complete_cli_expansion != VERSION
        || ready.request_id != request.request_id
        || ready.query != request.search.query
    {
        return Err(protocol_error(
            "expansion readiness does not match this request",
        ));
    }
    Ok(())
}

fn validate_variants(variants: &Variants, request: &ForwardedSearch) -> SearchResult<()> {
    if !request.search.expand
        || variants.fsfs_complete_cli_expansion != VERSION
        || variants.request_id != request.request_id
        || variants.queries.is_empty()
        || variants.queries.len() > MAX_EXPANDED_QUERIES
        || variants.queries[0].strategy != ExpansionStrategy::Original
        || variants.queries[0].text != request.search.query
    {
        return Err(protocol_error(
            "query variants do not match the admitted expansion",
        ));
    }
    for (ordinal, query) in variants.queries.iter().enumerate().skip(1) {
        if query.strategy == ExpansionStrategy::Original
            || query.text.trim().is_empty()
            || query.text.len() > MAX_VARIANT_BYTES
            || variants.queries[..ordinal]
                .iter()
                .any(|prior| prior.strategy == query.strategy)
        {
            return Err(protocol_error(
                "query variants exceed the bounded strategy contract",
            ));
        }
    }
    Ok(())
}

fn encode_variants(request: &ForwardedSearch, expansion: ExpansionResult) -> SearchResult<Vec<u8>> {
    let variants = Variants {
        fsfs_complete_cli_expansion: VERSION,
        request_id: request.request_id.clone(),
        queries: expansion.queries,
    };
    validate_variants(&variants, request)?;
    let mut bytes = serde_json::to_vec(&variants)
        .map_err(|_| protocol_error("cannot encode query variants"))?;
    bytes.push(b'\n');
    if bytes.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
        return Err(protocol_error(
            "query variants exceed the request byte bound",
        ));
    }
    Ok(bytes)
}

fn decode_variants(bytes: &[u8], request: &ForwardedSearch) -> SearchResult<ExpansionResult> {
    if bytes.len() > FSFS_DAEMON_REQUEST_MAX_BYTES || bytes.last() != Some(&b'\n') {
        return Err(protocol_error(
            "expected one bounded, complete variants record",
        ));
    }
    let variants: Variants = serde_json::from_slice(bytes)
        .map_err(|_| protocol_error("malformed or trailing query variants"))?;
    validate_variants(&variants, request)?;
    Ok(ExpansionResult {
        queries: variants.queries,
        // Provider diagnostics belong to the client. They are not server
        // authority and are deliberately absent from this protocol.
        backend_used: None,
        elapsed_ms: 0,
    })
}

/// Request text only after the surrounding executor has pinned its reader.
/// This future is polled after original Initial in the progressive route.
pub(super) async fn receive_variants(
    cx: &Cx,
    request: &ForwardedSearch,
    peer: &mut UnixStream,
    mut output: streaming::PhaseWriter,
) -> SearchResult<ExpansionResult> {
    retained_search_checkpoint(cx)?;
    if !request.search.expand {
        return Err(protocol_error(
            "this request did not authorize query expansion",
        ));
    }
    let ready = Ready {
        fsfs_complete_cli_expansion: VERSION,
        request_id: request.request_id.clone(),
        query: request.search.query.clone(),
    };
    let mut bytes = serde_json::to_vec(&ready)
        .map_err(|_| protocol_error("cannot encode expansion readiness"))?;
    bytes.push(b'\n');
    output.write_all(&bytes)?;
    output.flush()?;
    let bytes = read_variants_to_eof(cx, peer, request.search.delivery_budget()).await?;
    retained_search_checkpoint(cx)?;
    decode_variants(&bytes, request)
}

/// EOF fences the one follow-up. A second record sent in a later socket read
/// must not disappear as it could in a newline-only request reader. The client
/// half-closes writes and keeps receiving; this is not a peer disconnect.
async fn read_variants_to_eof(
    cx: &Cx,
    peer: &mut UnixStream,
    timeout: Duration,
) -> SearchResult<Vec<u8>> {
    let started = Instant::now();
    let mut bytes = Vec::new();
    let mut chunk = [0; 8192];
    let mut turn_bytes = 0;
    let mut turn_calls = 0;
    loop {
        retained_search_checkpoint(cx)?;
        let left = control::remaining(started, timeout)?;
        if turn_bytes >= 64 * 1024 || turn_calls >= 32 {
            super::super::admission::yield_turn(cx).await?;
            turn_bytes = 0;
            turn_calls = 0;
        }
        turn_calls += 1;
        let count_limit = (FSFS_DAEMON_REQUEST_MAX_BYTES + 1 - bytes.len()).min(chunk.len());
        match peer.read(&mut chunk[..count_limit]) {
            Ok(0) => return Ok(bytes),
            Ok(count) => {
                bytes.extend_from_slice(&chunk[..count]);
                turn_bytes += count;
                if bytes.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
                    return Err(protocol_error(
                        "query variants exceed the request byte bound",
                    ));
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                asupersync::time::sleep(cx.now(), left.min(IO_POLL_INTERVAL)).await;
                turn_bytes = 0;
                turn_calls = 0;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

/// Generate only after the peer's correlated admission, using the client's
/// own future. No LLM credentials or environment can enter the wire schema.
pub(super) async fn send_variants<F>(
    cx: &Cx,
    peer: &mut UnixStream,
    request: &ForwardedSearch,
    started: Instant,
    timeout: Duration,
    expansion: F,
) -> SearchResult<()>
where
    F: Future<Output = SearchResult<ExpansionResult>> + Send,
{
    // The daemon can disappear while an LLM future is pending. Observe peer
    // loss without consuming response bytes, then drop only this generator.
    // A response-side half-close remains legal while buffered frames drain.
    let connection = streaming::PhaseWriter::new(peer)?;
    let result = run_request(
        cx,
        control::remaining(started, timeout)?,
        streaming::drive(cx, &connection, expansion),
    )
    .await?;
    retained_search_checkpoint(cx)?;
    let bytes = encode_variants(request, result)?;
    write_response(cx, peer, &bytes, control::remaining(started, timeout)?).await?;
    peer.shutdown(Shutdown::Write)?;
    retained_search_checkpoint(cx)
}

#[derive(Debug)]
struct StartedFailure {
    source: SearchError,
    output: OutputError,
}

impl std::fmt::Display for StartedFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("admitted daemon query expansion failed; this request was not replayed")
    }
}

impl std::error::Error for StartedFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Once the handshake was accepted, even a malformed later version/refusal or
/// connection error cannot authorize reconnecting and replaying this request.
pub(super) fn after_started(error: SearchError) -> SearchError {
    if matches!(error, SearchError::Cancelled { .. }) {
        return error;
    }
    let output = super::FsfsRuntime::forwarded_search_error(&error)
        .cloned()
        .unwrap_or_else(|| output_error_from(&error));
    SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.expansion_started",
        source: Box::new(StartedFailure {
            source: error,
            output,
        }),
    }
}

pub(super) fn reported_error(error: &SearchError) -> Option<&OutputError> {
    if let SearchError::SubsystemError { subsystem, source } = error
        && *subsystem == "fsfs.complete_generation.expansion_started"
    {
        return source
            .downcast_ref::<StartedFailure>()
            .map(|error| &error.output);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::{FsfsRuntime, Reply, decode_reply, error_envelope, make_request};
    use super::*;
    use crate::FsfsConfig;
    use crate::query_expansion::LlmBackend;
    use asupersync::test_utils::run_test_with_cx;
    use std::task::{Context, Waker};

    fn request() -> ForwardedSearch {
        let root = tempfile::tempdir().unwrap();
        let mut runtime = FsfsRuntime::new(FsfsConfig::default());
        runtime.cli_input.expand = true;
        make_request(&runtime, root.path(), "original query", 10).unwrap()
    }

    fn queries(request: &ForwardedSearch) -> ExpansionResult {
        ExpansionResult {
            queries: vec![
                ExpandedQuery {
                    text: request.search.query.clone(),
                    strategy: ExpansionStrategy::Original,
                },
                ExpandedQuery {
                    text: "related terms".to_owned(),
                    strategy: ExpansionStrategy::Keyword,
                },
            ],
            backend_used: Some(LlmBackend::Anthropic),
            elapsed_ms: 17,
        }
    }

    #[test]
    fn expansion_wire_carries_only_correlated_text_and_bounded_strategies() {
        let request = request();
        let bytes = encode_variants(&request, queries(&request)).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 3);
        assert!(value.get("backend_used").is_none());
        assert!(value.get("elapsed_ms").is_none());
        let decoded = decode_variants(&bytes, &request).unwrap();
        assert_eq!(decoded.queries.len(), 2);
        assert_eq!(decoded.queries[0].text, request.search.query);
        assert_eq!(decoded.queries[1].strategy, ExpansionStrategy::Keyword);
        assert!(decoded.backend_used.is_none());

        for (field, invalid) in [
            (
                "fsfs_complete_cli_expansion",
                serde_json::json!(VERSION - 1),
            ),
            ("request_id", serde_json::json!("another request")),
            (
                "environment",
                serde_json::json!({"OPENAI_API_KEY": "never on the wire"}),
            ),
        ] {
            let mut malformed = value.clone();
            malformed[field] = invalid;
            let mut bytes = serde_json::to_vec(&malformed).unwrap();
            bytes.push(b'\n');
            assert!(decode_variants(&bytes, &request).is_err(), "{field}");
        }
        let mut excess = queries(&request);
        excess.queries.extend([
            ExpandedQuery {
                text: "s".to_owned(),
                strategy: ExpansionStrategy::Semantic,
            },
            ExpandedQuery {
                text: "h".to_owned(),
                strategy: ExpansionStrategy::HyDE,
            },
            ExpandedQuery {
                text: "extra".to_owned(),
                strategy: ExpansionStrategy::Keyword,
            },
        ]);
        assert!(encode_variants(&request, excess).is_err());
        for invalid in [
            "".to_owned(),
            " ".to_owned(),
            "x".repeat(MAX_VARIANT_BYTES + 1),
        ] {
            let mut expanded = queries(&request);
            expanded.queries[1].text = invalid;
            assert!(encode_variants(&request, expanded).is_err());
        }
        let mut expanded = queries(&request);
        expanded.queries.push(expanded.queries[1].clone());
        assert!(encode_variants(&request, expanded).is_err());
        let mut expanded = queries(&request);
        expanded.queries[0].text.push('x');
        assert!(encode_variants(&request, expanded).is_err());
        let mut expanded = queries(&request);
        expanded.queries[1].strategy = ExpansionStrategy::Original;
        assert!(encode_variants(&request, expanded).is_err());
        let mut expanded = queries(&request);
        expanded.queries.clear();
        assert!(encode_variants(&request, expanded).is_err());

        let mut trailing = bytes.clone();
        trailing.extend_from_slice(&bytes);
        assert!(decode_variants(&trailing, &request).is_err());
        assert!(decode_variants(&bytes[..bytes.len() - 1], &request).is_err());
        assert!(decode_variants(&vec![b' '; FSFS_DAEMON_REQUEST_MAX_BYTES + 1], &request).is_err());
        let duplicate = format!(
            "{{\"fsfs_complete_cli_expansion\":{VERSION},\"request_id\":\"{}\",\"request_id\":\"{}\",\"queries\":[]}}\n",
            request.request_id, request.request_id,
        );
        assert!(decode_variants(duplicate.as_bytes(), &request).is_err());
    }

    #[test]
    fn readiness_cannot_change_request_identity_or_add_provider_configuration() {
        let request = request();
        let ready = Ready {
            fsfs_complete_cli_expansion: VERSION,
            request_id: request.request_id.clone(),
            query: request.search.query.clone(),
        };
        let value = serde_json::to_value(&ready).unwrap();
        validate_ready(&serde_json::to_vec(&value).unwrap(), &request).unwrap();
        for (field, invalid) in [
            (
                "fsfs_complete_cli_expansion",
                serde_json::json!(VERSION - 1),
            ),
            ("query", serde_json::json!("another query")),
            ("request_id", serde_json::json!("another request")),
            ("api_key", serde_json::json!("must remain client-owned")),
        ] {
            let mut malformed = value.clone();
            malformed[field] = invalid;
            assert!(
                validate_ready(&serde_json::to_vec(&malformed).unwrap(), &request).is_err(),
                "{field}"
            );
        }
        let mut ordinary = request;
        ordinary.search.expand = false;
        assert!(validate_ready(&serde_json::to_vec(&value).unwrap(), &ordinary).is_err());
    }

    #[test]
    fn variants_require_eof_and_reject_a_second_later_record() {
        run_test_with_cx(|cx| async move {
            let request = request();
            let bytes = encode_variants(&request, queries(&request)).unwrap();
            let (mut client, mut server) = UnixStream::pair().unwrap();
            server.set_nonblocking(true).unwrap();
            client.write_all(&bytes).unwrap();
            let mut received = Box::pin(read_variants_to_eof(
                &cx,
                &mut server,
                Duration::from_secs(1),
            ));
            let mut task = Context::from_waker(Waker::noop());
            assert!(
                received.as_mut().poll(&mut task).is_pending(),
                "newline alone does not admit variants"
            );
            client.write_all(&bytes).unwrap();
            client.shutdown(Shutdown::Write).unwrap();
            let received = received.await.unwrap();
            assert!(decode_variants(&received, &request).is_err());
        });
    }

    #[test]
    fn admission_errors_after_readiness_never_authorize_replay() {
        let request = request();
        let refusal = complete_cli_error("daemon_config", "configuration changed");
        let reply = Reply {
            fsfs_complete_cli: VERSION,
            request_id: request.request_id.clone(),
            query: request.search.query.clone(),
            result: error_envelope(&refusal),
        };
        let remote = decode_reply(&serde_json::to_vec(&reply).unwrap(), &request).unwrap_err();
        assert!(FsfsRuntime::is_stale_daemon_refusal(&remote));
        let final_error = after_started(remote);
        assert!(!FsfsRuntime::is_stale_daemon_refusal(&final_error));
        assert_eq!(
            FsfsRuntime::forwarded_search_error(&final_error)
                .unwrap()
                .field
                .as_deref(),
            Some("complete_generation.daemon_config"),
        );
        let version_error = complete_cli_error("daemon_version", "old peer");
        assert!(!FsfsRuntime::is_stale_daemon_refusal(&after_started(
            version_error
        )));
    }
}
