//! Progressive CLI forwarding through the existing retained daemon and emitter.
//!
//! The socket carries canonical JSONL. Local TOON conversion changes only the
//! presentation, never retrieval. There is one request, no reconnect/replay,
//! a bounded partial record, and one deadline across connect, send and receive.

use std::fmt;
use std::io::{self, ErrorKind, Read, Write};
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};

use super::super::{FSFS_DAEMON_REQUEST_MAX_BYTES, IO_POLL_INTERVAL, streaming};
use super::{
    FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS, FSFS_DAEMON_RESPONSE_MAX_BYTES, ForwardedSearch, FsfsRuntime,
    LiveRetainedSearchReader, PeerOutcome, Reply, SearchExecutionFlags, VERSION, codec_error,
    complete_cli_error, control, decode_reply, decode_request, encode_response, error_envelope,
    make_request, retained_search_checkpoint, run_request, validate_request, write_response,
};
use crate::adapters::format_emitter::emit_stream_frame;
use crate::output_schema::{OutputError, SearchHitPayload, output_error_from};
use crate::stream_protocol::{
    StreamEvent, StreamFrame, StreamTerminalStatus, terminal_event_from_error,
    validate_stream_frame,
};
use crate::{CliCommand, OutputFormat};

type Frame = StreamFrame<SearchHitPayload>;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamRequest {
    fsfs_complete_cli_stream: u32,
    request: ForwardedSearch,
    #[serde(default)]
    explain: Option<bool>,
}

#[derive(Debug)]
struct ReportedFailure(OutputError);

impl fmt::Display for ReportedFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "forwarded stream ended: {:?}", self.0)
    }
}

impl std::error::Error for ReportedFailure {}

pub(super) fn reported_error(error: &SearchError) -> Option<&OutputError> {
    match error {
        SearchError::SubsystemError { source, .. } => source
            .downcast_ref::<ReportedFailure>()
            .map(|failure| &failure.0),
        _ => None,
    }
}

fn already_reported(error: OutputError) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.complete_generation.forwarded_stream",
        source: Box::new(ReportedFailure(error)),
    }
}

pub(super) fn is_streamed(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .is_ok_and(|value| value.get("fsfs_complete_cli_stream").is_some())
}

fn parse_request(bytes: &[u8]) -> SearchResult<StreamRequest> {
    // Both wrapper and nested request reject unknown and duplicate fields.
    // In particular, an older daemon sees no top-level query it could execute.
    let request: StreamRequest = serde_json::from_slice(bytes).map_err(codec_error)?;
    if request.fsfs_complete_cli_stream != VERSION || request.request.search.query.trim().is_empty()
    {
        return Err(complete_cli_error(
            "daemon_stream",
            "unsupported streaming CLI protocol",
        ));
    }
    // Reuse the buffered protocol's semantic version/id/limit checks. Duplicate
    // fields were already rejected above before this typed projection existed.
    decode_request(&serde_json::to_vec(&request.request).map_err(codec_error)?)?;
    Ok(request)
}

pub(super) async fn serve(
    cx: &Cx,
    runtime: &FsfsRuntime,
    session: &mut LiveRetainedSearchReader,
    peer: &mut UnixStream,
    bytes: &[u8],
    timeout: Duration,
) -> SearchResult<PeerOutcome> {
    let request = parse_request(bytes)?;
    let output = streaming::PhaseWriter::new(peer)?;
    let mut writer = output.clone();
    let producer = async {
        validate_request(runtime, session.store.root(), &request.request)?;
        session.refresh(cx).await?;
        let mut query_runtime = session.reader.runtime.clone();
        query_runtime.enable_complete_generation_explanations(
            session.store.root(),
            session.reader.generation(),
        )?;
        query_runtime.cli_input.command = CliCommand::Search;
        query_runtime.cli_input.query = Some(request.request.search.query.clone());
        query_runtime
            .cli_input
            .filter
            .clone_from(&request.request.search.filter);
        query_runtime.cli_input.daemon = false;
        query_runtime.cli_input.daemon_socket = None;
        query_runtime.cli_input.expand = false;
        query_runtime.cli_input.stream = true;
        query_runtime.cli_input.format = OutputFormat::Jsonl;
        query_runtime.cli_input.overrides.limit = Some(request.request.search.limit);
        query_runtime.cli_input.overrides.fast_only = Some(query_runtime.config.search.fast_only);
        query_runtime.cli_input.overrides.rerank = Some(query_runtime.config.search.rerank);
        query_runtime.cli_input.overrides.explain = request.explain;
        query_runtime
            .run_search_stream_command_with_writer(
                cx,
                &request.request.search.query,
                request.request.search.limit,
                &request.request.request_id,
                &mut writer,
                Some((
                    &mut session.reader.resources,
                    SearchExecutionFlags {
                        include_snippets: true,
                        persist_explain_session: true,
                    },
                )),
            )
            .await
    };
    let result = Box::pin(run_request(
        cx,
        timeout,
        streaming::drive(cx, &output, producer),
    ))
    .await;
    match result {
        Ok(()) => Ok(PeerOutcome::Search),
        Err(error @ SearchError::Cancelled { .. }) => Err(error),
        Err(error) if output.emitted() => Err(error),
        Err(error) => {
            // Before any canonical frame was exposed, use the same correlated
            // refusal as buffered forwarding. Never switch protocols mid-stream.
            let reply = Reply {
                fsfs_complete_cli: VERSION,
                request_id: request.request.request_id,
                query: request.request.search.query,
                result: error_envelope(&error),
            };
            let bytes = encode_response(&reply)?;
            write_response(cx, peer, &bytes, timeout).await?;
            Ok(PeerOutcome::Search)
        }
    }
}

#[derive(Default)]
struct ReceiveState {
    next_sequence: u64,
    started: bool,
    terminal: Option<Frame>,
    // Conservative: a Write error may have followed a partially written frame.
    // No other envelope or terminal may then be appended to that partial record.
    output_attempted: bool,
    output_failed: bool,
}

impl ReceiveState {
    fn accept(&mut self, frame: Frame, request: &ForwardedSearch) -> SearchResult<Option<Frame>> {
        // Reuse the public protocol validator, including terminal category and
        // retry constraints. Correlation and session order add to that contract,
        // rather than replacing it with a second, more permissive schema.
        if !validate_stream_frame(&frame).valid
            || self.terminal.is_some()
            || frame.command != "search"
            || frame.ts.trim().is_empty()
            || frame.stream_id != request.request_id
            || (self.started && frame.seq != self.next_sequence)
        {
            return Err(complete_cli_error(
                "daemon_stream",
                "invalid stream identity, ordering or trailing frame",
            ));
        }
        match &frame.event {
            StreamEvent::Started(started) => {
                if self.started
                    || started.stream_id != request.request_id
                    || started.query != request.search.query
                    || started.format != "jsonl"
                {
                    return Err(complete_cli_error(
                        "daemon_stream",
                        "invalid or duplicate Started frame",
                    ));
                }
                self.started = true;
            }
            _ if !self.started => {
                return Err(complete_cli_error(
                    "daemon_stream",
                    "stream did not begin with Started",
                ));
            }
            StreamEvent::Terminal(terminal) => {
                let coherent = match terminal.status {
                    StreamTerminalStatus::Completed => {
                        terminal.exit_code == 0 && terminal.error.is_none()
                    }
                    StreamTerminalStatus::Failed | StreamTerminalStatus::Cancelled => {
                        (1..=255).contains(&terminal.exit_code)
                            && terminal
                                .error
                                .as_ref()
                                .is_some_and(|error| error.exit_code == terminal.exit_code)
                    }
                };
                if !coherent {
                    return Err(complete_cli_error(
                        "daemon_stream",
                        "inconsistent terminal status or error",
                    ));
                }
                // Hold success until EOF proves there are no trailing records.
                // Keep this sequence available for a replacement failure terminal.
                self.terminal = Some(frame);
                return Ok(None);
            }
            StreamEvent::Result(result) if result.rank == 0 => {
                return Err(complete_cli_error(
                    "daemon_stream",
                    "result rank must be positive",
                ));
            }
            _ => {}
        }
        // The public frame contract requires monotonic sequence numbers, not
        // a particular initial value. Preserve the producer's starting value.
        self.next_sequence = frame
            .seq
            .checked_add(1)
            .ok_or_else(|| complete_cli_error("daemon_stream", "stream sequence exhausted"))?;
        Ok(Some(frame))
    }

    fn emit<W: Write>(
        &mut self,
        mut frame: Frame,
        format: OutputFormat,
        output: &mut W,
    ) -> SearchResult<()> {
        if let StreamEvent::Started(started) = &mut frame.event {
            started.format = format.to_string();
        }
        self.output_attempted = true;
        let result = emit_stream_frame(&frame, format, output)
            .and_then(|()| output.flush().map_err(SearchError::Io));
        if result.is_err() {
            self.output_failed = true;
        }
        result
    }
}

#[allow(clippy::too_many_arguments)] // Request identity, deadline and output state remain borrowed.
async fn receive<W: Write + Send>(
    cx: &Cx,
    peer: &mut UnixStream,
    request: &ForwardedSearch,
    started: Instant,
    timeout: Duration,
    format: OutputFormat,
    state: &mut ReceiveState,
    output: &mut W,
) -> SearchResult<Option<OutputError>> {
    let mut record = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        retained_search_checkpoint(cx)?;
        let remaining = control::remaining(started, timeout)?;
        match peer.read(&mut chunk) {
            Ok(0) => {
                if !record.is_empty() {
                    return Err(io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "partial daemon stream record",
                    )
                    .into());
                }
                let terminal = state.terminal.take().ok_or_else(|| {
                    SearchError::Io(io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "daemon stream closed without Terminal",
                    ))
                })?;
                let error = match &terminal.event {
                    StreamEvent::Terminal(terminal) => terminal.error.clone(),
                    _ => {
                        return Err(complete_cli_error(
                            "daemon_stream",
                            "missing terminal state",
                        ));
                    }
                };
                state.emit(terminal, format, output)?;
                return Ok(error);
            }
            Ok(count) => {
                for part in chunk[..count].split_inclusive(|byte| *byte == b'\n') {
                    retained_search_checkpoint(cx)?;
                    control::remaining(started, timeout)?;
                    append_record(&mut record, part)?;
                    if part.last() != Some(&b'\n') {
                        continue;
                    }
                    if !state.started
                        && serde_json::from_slice::<serde_json::Value>(&record)
                            .is_ok_and(|value| value.get("fsfs_complete_cli").is_some())
                    {
                        return match decode_reply(&record, request) {
                            Err(error) => Err(error),
                            Ok(_) => Err(complete_cli_error(
                                "daemon_stream",
                                "buffered success cannot satisfy a stream request",
                            )),
                        };
                    }
                    let frame: Frame = serde_json::from_slice(&record).map_err(codec_error)?;
                    record.clear();
                    if let Some(frame) = state.accept(frame, request)? {
                        state.emit(frame, format, output)?;
                    }
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                asupersync::time::sleep(cx.now(), remaining.min(IO_POLL_INTERVAL)).await;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

fn append_record(record: &mut Vec<u8>, part: &[u8]) -> SearchResult<()> {
    if part.len() > FSFS_DAEMON_RESPONSE_MAX_BYTES.saturating_sub(record.len()) {
        return Err(complete_cli_error(
            "daemon_stream",
            "stream record exceeds response bound",
        ));
    }
    record.extend_from_slice(part);
    Ok(())
}

fn finish_receive<W: Write>(
    state: &mut ReceiveState,
    request: &ForwardedSearch,
    format: OutputFormat,
    output: &mut W,
    result: SearchResult<Option<OutputError>>,
) -> SearchResult<()> {
    match result {
        Ok(None) => Ok(()),
        Ok(Some(error)) => Err(already_reported(error)),
        Err(error) if !state.output_attempted => Err(error),
        Err(error) => {
            let original = output_error_from(&error);
            if !state.output_failed {
                let terminal = Frame::new(
                    &request.request_id,
                    state.next_sequence,
                    super::super::super::iso_timestamp_now(),
                    "search",
                    StreamEvent::Terminal(terminal_event_from_error(&error, 0, 0)),
                );
                if let Err(output_error) = state.emit(terminal, format, output) {
                    return Err(already_reported(output_error_from(&output_error)));
                }
            }
            Err(already_reported(original))
        }
    }
}

impl FsfsRuntime {
    /// Forward a progressive search without initializing models in this client.
    ///
    /// # Errors
    /// Returns transport, configuration, protocol, cancellation or remote errors.
    /// Errors after delivery are marked so the process boundary emits no second
    /// envelope. A broken output writer is never written to again.
    pub(crate) async fn stream_complete_generation_daemon<W: Write + Send>(
        &self,
        cx: &Cx,
        root: &Path,
        query: &str,
        limit: usize,
        output: &mut W,
    ) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        if self.cli_input.expand
            || !matches!(
                self.cli_input.format,
                OutputFormat::Jsonl | OutputFormat::Toon
            )
        {
            return Err(complete_cli_error(
                "daemon_options",
                "stream forwarding requires JSONL/TOON without expansion",
            ));
        }
        let request = StreamRequest {
            fsfs_complete_cli_stream: VERSION,
            request: make_request(self, root, query, limit)?,
            explain: self.cli_input.overrides.explain,
        };
        let mut bytes = serde_json::to_vec(&request).map_err(codec_error)?;
        bytes.push(b'\n');
        if bytes.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
            return Err(complete_cli_error(
                "daemon_request",
                "request exceeds its byte limit",
            ));
        }
        parse_request(&bytes)?;
        let path = self.complete_generation_socket_path(root)?;
        if !std::fs::symlink_metadata(&path)?.file_type().is_socket() {
            return Err(complete_cli_error(
                "daemon_socket",
                "expected a non-symlink Unix socket",
            ));
        }
        let timeout = Duration::from_millis(FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS);
        let started = Instant::now();
        let mut peer = control::connect(cx, &path, started, timeout).await?;
        write_response(cx, &mut peer, &bytes, control::remaining(started, timeout)?).await?;
        let mut state = ReceiveState::default();
        let result = receive(
            cx,
            &mut peer,
            &request.request,
            started,
            timeout,
            self.cli_input.format,
            &mut state,
            output,
        )
        .await;
        finish_receive(
            &mut state,
            &request.request,
            self.cli_input.format,
            output,
            result,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FsfsConfig;
    use crate::stream_protocol::{StreamStartedEvent, terminal_event_completed};
    use std::net::Shutdown;

    fn request() -> ForwardedSearch {
        let root = tempfile::tempdir().unwrap();
        make_request(
            &FsfsRuntime::new(FsfsConfig::default()),
            root.path(),
            "query",
            10,
        )
        .unwrap()
    }

    fn started(request: &ForwardedSearch) -> Frame {
        Frame::new(
            &request.request_id,
            0,
            "2026-09-21T00:00:00Z",
            "search",
            StreamEvent::Started(StreamStartedEvent::new(
                &request.request_id,
                &request.search.query,
                "jsonl",
            )),
        )
    }

    fn completed(request: &ForwardedSearch) -> Frame {
        Frame::new(
            &request.request_id,
            1,
            "2026-09-21T00:00:00Z",
            "search",
            StreamEvent::Terminal(terminal_event_completed()),
        )
    }

    fn frames_bytes(frames: &[Frame]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for frame in frames {
            emit_stream_frame(frame, OutputFormat::Jsonl, &mut bytes).unwrap();
        }
        bytes
    }

    async fn read(
        cx: &Cx,
        request: &ForwardedSearch,
        bytes: &[u8],
        format: OutputFormat,
    ) -> (Vec<u8>, SearchResult<()>) {
        let (mut peer, mut server) = UnixStream::pair().unwrap();
        peer.set_nonblocking(true).unwrap();
        server.write_all(bytes).unwrap();
        server.shutdown(Shutdown::Write).unwrap();
        let mut state = ReceiveState::default();
        let mut output = Vec::new();
        let result = receive(
            cx,
            &mut peer,
            request,
            Instant::now(),
            Duration::from_secs(1),
            format,
            &mut state,
            &mut output,
        )
        .await;
        let result = finish_receive(&mut state, request, format, &mut output, result);
        (output, result)
    }

    #[test]
    fn streaming_request_is_versioned_nested_and_preserves_filter_and_explain() {
        let mut request = request();
        request.search.filter = Some("path:src".into());
        let wrapper = StreamRequest {
            fsfs_complete_cli_stream: VERSION,
            request,
            explain: Some(true),
        };
        let bytes = serde_json::to_vec(&wrapper).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("query").is_none());
        assert!(is_streamed(&bytes));
        let parsed = parse_request(&bytes).unwrap();
        assert_eq!(parsed.explain, Some(true));
        assert_eq!(parsed.request.search.filter, wrapper.request.search.filter);
        let mut stale = value.clone();
        stale["fsfs_complete_cli_stream"] = serde_json::json!(1);
        assert!(parse_request(&serde_json::to_vec(&stale).unwrap()).is_err());
        stale = value;
        stale["request"]["fsfs_complete_cli"] = serde_json::json!(1);
        assert!(parse_request(&serde_json::to_vec(&stale).unwrap()).is_err());
        let raw = String::from_utf8(bytes).unwrap();
        for field in [",\"explain\":false}", ",\"query\":\"legacy bypass\"}"] {
            assert!(parse_request(format!("{}{field}", &raw[..raw.len() - 1]).as_bytes()).is_err());
        }
    }

    #[test]
    fn successful_stream_reaches_both_jsonl_and_toon_presentations() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = request();
            let bytes = frames_bytes(&[started(&request), completed(&request)]);
            let (json, result) = read(&cx, &request, &bytes, OutputFormat::Jsonl).await;
            result.unwrap();
            assert_eq!(json, bytes);
            let (toon, result) = read(&cx, &request, &bytes, OutputFormat::Toon).await;
            result.unwrap();
            let toon = String::from_utf8(toon).unwrap();
            assert_eq!(toon.matches('\u{001e}').count(), 2);
            assert!(toon.contains("toon"));
        });
    }

    #[test]
    fn frame_identity_sequence_and_started_are_checked_before_delivery() {
        let request = request();
        for frame in [
            Frame {
                stream_id: "wrong request".into(),
                ..started(&request)
            },
            Frame {
                seq: u64::MAX,
                ..started(&request)
            },
            Frame {
                v: 99,
                ..started(&request)
            },
            Frame {
                ts: String::new(),
                ..started(&request)
            },
            Frame {
                seq: 0,
                ..completed(&request)
            },
        ] {
            assert!(ReceiveState::default().accept(frame, &request).is_err());
        }
        let mut state = ReceiveState::default();
        state.accept(started(&request), &request).unwrap();
        assert!(
            state
                .accept(
                    Frame {
                        seq: 2,
                        ..completed(&request)
                    },
                    &request
                )
                .is_err()
        );
        let duplicate = Frame {
            seq: 1,
            ..started(&request)
        };
        assert!(state.accept(duplicate, &request).is_err());
        let mut offset = ReceiveState::default();
        offset
            .accept(
                Frame {
                    seq: 7,
                    ..started(&request)
                },
                &request,
            )
            .unwrap();
        assert!(
            offset
                .accept(
                    Frame {
                        seq: 8,
                        ..completed(&request)
                    },
                    &request
                )
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn terminal_validation_keeps_the_public_protocol_retry_contract() {
        use crate::stream_protocol::StreamRetryDirective;

        let request = request();
        let mut invalid = completed(&request);
        let StreamEvent::Terminal(terminal) = &mut invalid.event else {
            unreachable!("completed fixture is a terminal"); // ubs:ignore — cfg(test) fixture.
        };
        terminal.retry = StreamRetryDirective::RetryAfterMs {
            delay_ms: 1,
            next_attempt: 1,
            max_attempts: 1,
        };
        let mut state = ReceiveState::default();
        state.accept(started(&request), &request).unwrap();
        assert!(state.accept(invalid, &request).is_err());
        // Refusal leaves the expected terminal sequence available, and the
        // identical terminal without invalid retry guidance is still accepted.
        assert!(
            state
                .accept(completed(&request), &request)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn eof_without_terminal_replaces_truncation_with_one_failed_terminal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = request();
            let mut bytes = frames_bytes(&[started(&request)]);
            bytes.extend_from_slice(b"{partial");
            let (output, result) = read(&cx, &request, &bytes, OutputFormat::Jsonl).await;
            let error = result.unwrap_err();
            assert!(reported_error(&error).is_some());
            assert!(FsfsRuntime::forwarded_search_error_was_emitted(&error));
            let frames: Vec<Frame> = output
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect();
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[1].seq, 1);
            assert!(matches!(&frames[1].event, StreamEvent::Terminal(terminal)
                if terminal.status == StreamTerminalStatus::Failed));
        });
    }

    #[test]
    fn terminal_success_is_not_exposed_when_the_peer_appends_another_frame() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = request();
            let extra = Frame {
                seq: 2,
                ..started(&request)
            };
            let bytes = frames_bytes(&[started(&request), completed(&request), extra]);
            let (output, result) = read(&cx, &request, &bytes, OutputFormat::Jsonl).await;
            assert!(result.is_err());
            let frames: Vec<Frame> = output
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect();
            assert_eq!(frames.len(), 2);
            assert!(matches!(&frames[1].event, StreamEvent::Terminal(terminal)
                if terminal.status == StreamTerminalStatus::Failed));
        });
    }

    #[test]
    fn remote_failed_terminal_retains_its_exit_status_without_a_second_envelope() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = request();
            let original = complete_cli_error("test_remote", "original stream failure");
            let terminal = Frame::new(
                &request.request_id,
                1,
                "2026-09-21T00:00:00Z",
                "search",
                StreamEvent::Terminal(terminal_event_from_error(&original, 0, 0)),
            );
            let bytes = frames_bytes(&[started(&request), terminal]);
            let (output, result) = read(&cx, &request, &bytes, OutputFormat::Jsonl).await;
            let error = result.unwrap_err();
            assert_eq!(output, bytes);
            assert_eq!(
                FsfsRuntime::forwarded_search_error(&error)
                    .unwrap()
                    .exit_code,
                output_error_from(&original).exit_code
            );
        });
    }

    #[test]
    fn record_bound_is_enforced_before_growing_the_partial_buffer() {
        let mut bytes = vec![b'x'; FSFS_DAEMON_RESPONSE_MAX_BYTES - 1];
        append_record(&mut bytes, b"\n").unwrap();
        assert_eq!(bytes.len(), FSFS_DAEMON_RESPONSE_MAX_BYTES);
        assert!(append_record(&mut bytes, b"x").is_err());
        assert_eq!(bytes.len(), FSFS_DAEMON_RESPONSE_MAX_BYTES);
    }

    #[test]
    fn cancellation_and_expired_deadline_emit_nothing_before_started() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = request();
            let (mut peer, _server) = UnixStream::pair().unwrap();
            peer.set_nonblocking(true).unwrap();
            let mut state = ReceiveState::default();
            let mut output = Vec::new();
            cx.set_cancel_requested(true);
            let result = receive(
                &cx,
                &mut peer,
                &request,
                Instant::now(),
                Duration::from_secs(1),
                OutputFormat::Jsonl,
                &mut state,
                &mut output,
            )
            .await;
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            cx.set_cancel_requested(false);
            let result = receive(
                &cx,
                &mut peer,
                &request,
                Instant::now(),
                Duration::ZERO,
                OutputFormat::Jsonl,
                &mut state,
                &mut output,
            )
            .await;
            assert!(
                matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut)
            );
            assert!(output.is_empty());
        });
    }

    #[test]
    fn cancellation_after_started_emits_one_cancelled_terminal() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let request = request();
            let mut state = ReceiveState::default();
            let mut output = Vec::new();
            let first = state.accept(started(&request), &request).unwrap().unwrap();
            state.emit(first, OutputFormat::Jsonl, &mut output).unwrap();
            let (mut peer, _server) = UnixStream::pair().unwrap();
            peer.set_nonblocking(true).unwrap();
            cx.set_cancel_requested(true);
            let result = receive(
                &cx,
                &mut peer,
                &request,
                Instant::now(),
                Duration::from_secs(1),
                OutputFormat::Jsonl,
                &mut state,
                &mut output,
            )
            .await;
            let error = finish_receive(
                &mut state,
                &request,
                OutputFormat::Jsonl,
                &mut output,
                result,
            )
            .unwrap_err();
            cx.set_cancel_requested(false);
            let frames: Vec<Frame> = output
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect();
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[1].seq, 1);
            assert!(matches!(&frames[1].event, StreamEvent::Terminal(terminal)
                if terminal.status == StreamTerminalStatus::Cancelled));
            assert!(FsfsRuntime::forwarded_search_error_was_emitted(&error));
        });
    }

    #[test]
    fn cli_stream_dispatch_reaches_a_socket_without_opening_local_models_or_indexes() {
        use crate::CliInput;
        use std::io::BufRead;
        use std::os::unix::net::UnixListener;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let root = std::fs::canonicalize(directory.path()).unwrap();
            let socket = root.join("forward.sock");
            let listener = UnixListener::bind(&socket).unwrap();
            listener.set_nonblocking(true).unwrap();
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Search,
                query: Some("query".into()),
                daemon: true,
                daemon_socket: Some(socket),
                index_dir: Some(root.clone()),
                stream: true,
                format: OutputFormat::Toon,
                filter: Some("path:src".into()),
                ..CliInput::default()
            });
            let server = std::thread::spawn(move || {
                let deadline = Instant::now();
                let mut peer = loop {
                    match listener.accept() {
                        Ok((peer, _)) => break peer,
                        Err(error) if error.kind() == ErrorKind::WouldBlock => {
                            assert!(deadline.elapsed() < Duration::from_secs(3));
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("accept failed: {error}"), // ubs:ignore — cfg(test) failure.
                    }
                };
                peer.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                peer.set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                std::io::BufReader::new(&mut peer)
                    .take(FSFS_DAEMON_REQUEST_MAX_BYTES as u64 + 1)
                    .read_until(b'\n', &mut bytes)
                    .unwrap();
                assert!(bytes.len() <= FSFS_DAEMON_REQUEST_MAX_BYTES);
                let request = parse_request(&bytes).unwrap();
                assert_eq!(request.request.search.filter.as_deref(), Some("path:src"));
                peer.write_all(&frames_bytes(&[
                    started(&request.request),
                    completed(&request.request),
                ]))
                .unwrap();
            });
            // Deliberately no generation or model artifacts exist. A routing
            // regression into direct retrieval cannot satisfy this request.
            let mut output = Vec::new();
            let result = runtime
                .run_complete_generation_search_with_writer(&cx, &root, &mut output)
                .await;
            server.join().unwrap();
            result.unwrap();
            let output = String::from_utf8(output).unwrap();
            assert_eq!(output.matches('\u{001e}').count(), 2);
            assert!(!root.join("generations").exists());
        });
    }

    #[test]
    fn broken_output_is_never_written_again_by_error_reporting() {
        struct BrokenOutput(usize);
        impl Write for BrokenOutput {
            fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
                self.0 += 1;
                Err(io::Error::new(ErrorKind::BrokenPipe, "consumer gone"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let request = request();
        let mut state = ReceiveState::default();
        let mut output = BrokenOutput(0);
        let error = state
            .emit(started(&request), OutputFormat::Jsonl, &mut output)
            .unwrap_err();
        let writes = output.0;
        let error = finish_receive(
            &mut state,
            &request,
            OutputFormat::Jsonl,
            &mut output,
            Err(error),
        )
        .unwrap_err();
        assert_eq!(output.0, writes);
        assert!(FsfsRuntime::forwarded_search_error_was_emitted(&error));
    }
}

#[cfg(all(test, not(feature = "embedded-models")))]
mod generation_tests {
    use super::*;
    use crate::generation_store::{CompleteGenerationStore, GenerationPublication};
    use crate::{CliInput, FsfsConfig};
    use std::fs;

    #[test]
    fn forwarded_stream_uses_real_retained_retrieval_and_keeps_the_bundle_intact() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source");
            let root = directory.path().join("store");
            fs::create_dir(&source).unwrap();
            fs::write(source.join("alpha.md"), "sharedtoken alpha document").unwrap();
            let mut config = FsfsConfig::default();
            "{index_dir}/catalog.sqlite".clone_into(&mut config.storage.db_path);
            config.indexing.offline = true;
            config.indexing.quality_model.clear();
            config.search.fast_only = true;
            config.search.rerank = false;
            let runtime = FsfsRuntime::new(config).with_cli_input(CliInput {
                command: CliCommand::Search,
                index_dir: Some(root.clone()),
                target_path: Some(source),
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
            let store = CompleteGenerationStore::open(&cx, &root).unwrap();
            let selected = store.active(&cx).unwrap();
            let mut session = runtime.open_live_retained_search(&cx, &root).await.unwrap();
            let request = StreamRequest {
                fsfs_complete_cli_stream: VERSION,
                request: make_request(&runtime, &root, "sharedtoken", 10).unwrap(),
                explain: None,
            };
            let (mut client, mut peer) = UnixStream::pair().unwrap();
            peer.set_nonblocking(true).unwrap();
            client.set_nonblocking(true).unwrap();
            serve(
                &cx,
                &runtime,
                &mut session,
                &mut peer,
                &serde_json::to_vec(&request).unwrap(),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
            drop(peer);
            let mut state = ReceiveState::default();
            let mut output = Vec::new();
            let result = receive(
                &cx,
                &mut client,
                &request.request,
                Instant::now(),
                Duration::from_secs(5),
                OutputFormat::Jsonl,
                &mut state,
                &mut output,
            )
            .await;
            finish_receive(
                &mut state,
                &request.request,
                OutputFormat::Jsonl,
                &mut output,
                result,
            )
            .unwrap();
            let frames: Vec<Frame> = output
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_slice(line).unwrap())
                .collect();
            assert!(matches!(
                &frames.first().unwrap().event,
                StreamEvent::Started(_)
            ));
            assert_eq!(
                frames
                    .iter()
                    .filter(|frame| matches!(&frame.event, StreamEvent::Result(_)))
                    .count(),
                1
            );
            assert!(
                matches!(&frames.last().unwrap().event, StreamEvent::Terminal(terminal)
                if terminal.status == StreamTerminalStatus::Completed)
            );
            assert_eq!(store.active(&cx).unwrap(), selected);
            let context = FsfsRuntime::load_explain_session_at_root(&root)
                .unwrap()
                .unwrap();
            assert_eq!(context.query, "sharedtoken");
            assert_eq!(context.hits.len(), 1);
            assert_eq!(
                context.complete_generation.as_ref().unwrap().id,
                selected.as_ref().unwrap().id()
            );
            assert!(
                !selected
                    .as_ref()
                    .unwrap()
                    .path()
                    .join("explain/last_search_session.json")
                    .exists()
            );
        });
    }
}
