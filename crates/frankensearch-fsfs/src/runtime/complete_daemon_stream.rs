//! Bounded, owned bridge from the synchronous phase writer to a Unix socket.
//!
//! Flush publishes complete JSONL records immediately when the peer is ready.
//! Backpressure is retained under the existing response-byte bound and drained
//! while the owning search future is pending. No thread or detached task owns
//! a search, generation reader, writer, or socket after this call returns.

use std::future::{Future, poll_fn};
use std::io::{self, ErrorKind, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::Poll;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};

use super::{IO_POLL_INTERVAL, complete_cli_error, retained_search_checkpoint};
use super::super::super::FSFS_DAEMON_RESPONSE_MAX_BYTES;

struct Output {
    peer: UnixStream,
    pending: Vec<u8>,
    offset: usize,
    staged: Vec<u8>,
    emitted: bool,
}

impl Output {
    fn drain(&mut self) -> io::Result<()> {
        while self.offset < self.pending.len() {
            match self.peer.write(&self.pending[self.offset..]) {
                Ok(0) => return Err(io::Error::new(ErrorKind::WriteZero, "stream peer closed")),
                Ok(count) => {
                    self.emitted = true;
                    self.offset += count;
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        self.pending.clear();
        self.offset = 0;
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct PhaseWriter(Arc<Mutex<Output>>);

impl PhaseWriter {
    pub(super) fn new(peer: &UnixStream) -> SearchResult<Self> {
        let peer = peer.try_clone()?;
        peer.set_nonblocking(true)?;
        Ok(Self(Arc::new(Mutex::new(Output {
            peer,
            pending: Vec::new(),
            offset: 0,
            staged: Vec::new(),
            emitted: false,
        }))))
    }

    fn lock(&self) -> io::Result<MutexGuard<'_, Output>> {
        self.0.lock().map_err(|_| io::Error::other("stream output lock poisoned"))
    }

    pub(super) fn emitted(&self) -> bool {
        // A poisoned state cannot authorize a buffered fallback after a stream.
        self.0.lock().map_or(true, |output| output.emitted)
    }
}

impl Write for PhaseWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut output = self.lock()?;
        let used = output.pending.len() - output.offset + output.staged.len();
        if bytes.len() > FSFS_DAEMON_RESPONSE_MAX_BYTES.saturating_sub(used) {
            return Err(io::Error::other("complete-generation stream output exceeds 4 MiB bound"));
        }
        output.staged.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut output = self.lock()?;
        if !output.staged.is_empty() {
            if output.staged.last() != Some(&b'\n') {
                return Err(io::Error::new(ErrorKind::InvalidData, "stream flush requires complete JSONL records"));
            }
            if output.offset != 0 {
                let offset = output.offset;
                output.pending.drain(..offset);
                output.offset = 0;
            }
            // Take disjoint mutable fields without holding a borrow of `self`
            // across a socket operation or an async suspension.
            let Output { pending, staged, .. } = &mut *output;
            pending.append(staged);
        }
        output.drain()
    }
}

/// Drive and drain one producer, retaining its completed result until all
/// committed records have been delivered. The caller supplies the overall
/// request deadline through `run_request`; this function adds no retry/replay.
pub(super) async fn drive<T>(
    cx: &Cx,
    output: &PhaseWriter,
    producer: impl Future<Output = SearchResult<T>>,
) -> SearchResult<T> {
    let mut producer = Some(Box::pin(producer));
    let mut finished = None;
    loop {
        let mut tick = Box::pin(asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL));
        let result = poll_fn(|task| {
            if let Err(error) = retained_search_checkpoint(cx) {
                return Poll::Ready(Err(error));
            }
            if let Err(error) = output.lock().and_then(|mut state| state.drain()) {
                return Poll::Ready(Err(SearchError::Io(error)));
            }
            if let Some(future) = producer.as_mut()
                && let Poll::Ready(result) = future.as_mut().poll(task)
            {
                producer = None;
                finished = Some(result);
            }
            let mut state = match output.lock() {
                Ok(state) => state,
                Err(error) => return Poll::Ready(Err(error.into())),
            };
            if finished.as_ref().is_some_and(Result::is_err) {
                // Discard an uncommitted partial record, never an older record
                // already flushed by the producer. No buffered envelope may
                // follow any exposed stream bytes.
                state.staged.clear();
            }
            if let Err(error) = state.drain() {
                return Poll::Ready(Err(error.into()));
            }
            if state.pending.is_empty()
                && let Some(result) = finished.take()
            {
                if result.is_ok() && !state.staged.is_empty() {
                    return Poll::Ready(Err(complete_cli_error(
                        "daemon_stream", "producer finished with an unflushed record",
                    )));
                }
                return Poll::Ready(result.map(Some));
            }
            drop(state);
            if tick.as_mut().poll(task).is_ready() {
                Poll::Ready(Ok(None))
            } else {
                Poll::Pending
            }
        }).await?;
        if let Some(result) = result {
            return Ok(result);
        }
    }
}

/// Streaming uses the daemon's admitted full-search configuration. Until
/// per-request overlays share the ordinary streaming route, refuse them
/// explicitly rather than silently dropping filters or changing model space.
pub(super) fn validate_request(bytes: &[u8]) -> SearchResult<()> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        complete_cli_error("daemon_stream", &format!("invalid streaming request: {error}"))
    })?;
    let object = value.as_object().ok_or_else(|| {
        complete_cli_error("daemon_stream", "streaming request must be a JSON object")
    })?;
    for (key, value) in object {
        match key.as_str() {
            "query" | "limit" | "stream" => {}
            "mode" if value.is_null() || value.as_str() == Some("full") => {}
            _ => return Err(complete_cli_error(
                "daemon_stream",
                &format!("unsupported streaming option {key}; use query, limit, stream and mode=full with the daemon startup configuration"),
            )),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::io::Read;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn stream_flush_delivers_before_producer_completion() {
        run_test_with_cx(|cx| async move {
            let (mut client, server) = UnixStream::pair().unwrap();
            client.set_nonblocking(true).unwrap();
            let output = PhaseWriter::new(&server).unwrap();
            let mut writer = output.clone();
            let result = drive(&cx, &output, async {
                writer.write_all(b"{\"phase\":\"initial\"}\n")?;
                writer.flush()?;
                // The producer is still running and has not produced Refined.
                // Read the actual socket now, not a shared test-side queue.
                let mut first = [0; 20];
                client.read_exact(&mut first)?;
                assert_eq!(&first, b"{\"phase\":\"initial\"}\n");
                let mut yielded = false;
                poll_fn(|task| {
                    if yielded {
                        Poll::Ready(())
                    } else {
                        yielded = true;
                        task.waker().wake_by_ref();
                        Poll::Pending
                    }
                }).await;
                writer.write_all(b"{\"phase\":\"refined\"}\n")?;
                writer.flush()?;
                Ok(42)
            }).await.unwrap();
            assert_eq!(result, 42);
            let mut second = [0; 20];
            client.read_exact(&mut second).unwrap();
            assert_eq!(&second, b"{\"phase\":\"refined\"}\n");
        });
    }

    #[test]
    fn staged_or_oversized_records_are_not_exposed() {
        let (mut client, server) = UnixStream::pair().unwrap();
        client.set_nonblocking(true).unwrap();
        let mut output = PhaseWriter::new(&server).unwrap();
        output.write_all(b"partial").unwrap();
        assert!(output.flush().is_err());
        assert!(output.write_all(&vec![b'x'; FSFS_DAEMON_RESPONSE_MAX_BYTES]).is_err());
        assert!(!output.emitted());
        let error = client.read(&mut [0; 1]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::WouldBlock);
    }

    struct Released(Arc<AtomicBool>);
    impl Drop for Released {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn disconnected_output_drops_producer_and_does_not_cancel_daemon() {
        run_test_with_cx(|cx| async move {
            let (client, server) = UnixStream::pair().unwrap();
            let output = PhaseWriter::new(&server).unwrap();
            let mut writer = output.clone();
            let released = Arc::new(AtomicBool::new(false));
            let owned = Released(Arc::clone(&released));
            drop(client);
            let result: SearchResult<()> = drive(&cx, &output, async move {
                let _owned = owned;
                writer.write_all(b"{}\n")?;
                writer.flush()?;
                std::future::pending().await
            }).await;
            assert!(result.is_err());
            assert!(released.load(Ordering::SeqCst));
            retained_search_checkpoint(&cx).unwrap();
        });
    }

    #[test]
    fn blocked_output_remains_bounded_and_deadline_drops_search() {
        run_test_with_cx(|cx| async move {
            let (_client, mut server) = UnixStream::pair().unwrap();
            server.set_nonblocking(true).unwrap();
            let mut filled = 0;
            loop {
                match server.write(&[b'x'; 8192]) {
                    Ok(count) => {
                        filled += count;
                        assert!(filled < 16 * 1024 * 1024, "socket did not reach backpressure");
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                    other => panic!("unexpected socket fill: {other:?}"),
                }
            }
            let output = PhaseWriter::new(&server).unwrap();
            let mut writer = output.clone();
            let released = Arc::new(AtomicBool::new(false));
            let owned = Released(Arc::clone(&released));
            let result: SearchResult<()> = super::super::run_request(
                &cx, Duration::from_millis(1), drive(&cx, &output, async move {
                    let _owned = owned;
                    writer.write_all(b"{}\n")?;
                    writer.flush()?;
                    std::future::pending().await
                }),
            ).await;
            assert!(matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut));
            assert!(released.load(Ordering::SeqCst));
            assert_eq!(output.lock().unwrap().pending, b"{}\n");
            retained_search_checkpoint(&cx).unwrap();
        });
    }

    #[test]
    fn unfinished_record_is_not_mistaken_for_success() {
        run_test_with_cx(|cx| async move {
            let (_client, server) = UnixStream::pair().unwrap();
            let output = PhaseWriter::new(&server).unwrap();
            let mut writer = output.clone();
            let error = drive(&cx, &output, async {
                writer.write_all(b"unfinished")?;
                Ok(())
            }).await.unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { field, .. }
                if field == "complete_generation.daemon_stream"));
            assert!(!output.emitted());
        });
    }

    #[test]
    fn streaming_refuses_ignored_options_and_nonfull_modes() {
        validate_request(br#"{"query":"x","limit":10,"stream":true,"mode":"full"}"#).unwrap();
        for raw in [
            br#"{"query":"x","stream":true,"mode":"fast"}"#.as_slice(),
            br#"{"query":"x","stream":true,"filter":"secret"}"#,
            br#"{"query":"x","stream":true,"fast_only":false}"#,
            br#"{"query":"x","stream":true,"protocol":"unknown"}"#,
        ] {
            assert!(validate_request(raw).is_err());
        }
    }
}
