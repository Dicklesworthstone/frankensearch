//! Owned, bounded Unix transport for complete-generation buffered search.
//!
//! Index admission/ranking remain in the existing serve handler. The listener
//! and lock live at the store root, never inside a retained generation.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::future::{Future, poll_fn};
use std::io::{self, ErrorKind, Read, Write};
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::task::Poll;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::Serialize;

use super::super::{
    FSFS_DAEMON_CLIENT_TIMEOUT_MS, FSFS_DAEMON_IDLE_TIMEOUT_MS,
    FSFS_DAEMON_REQUEST_MAX_BYTES, FSFS_DAEMON_SOCKET_FILE, SearchServeRequest,
};
use super::{FsfsRuntime, complete_cli_error, emit_complete_serve_line, retained_search_checkpoint};

const IO_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[path = "complete_daemon_control.rs"]
mod control;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeerOutcome {
    Search,
    Shutdown,
}

/// Keep the singleton lock through listener teardown. The lock file is never
/// unlinked: replacing it would let two processes lock different inodes.
struct BoundCompleteSocket {
    listener: UnixListener,
    path: PathBuf,
    identity: (u64, u64),
    _lock: File,
}

impl BoundCompleteSocket {
    fn bind(path: PathBuf) -> SearchResult<Self> {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path.with_extension("lock"))?;
        if !lock.metadata()?.is_file() {
            return Err(complete_cli_error("daemon_lock", "daemon lock is not a regular file"));
        }
        rustix::fs::flock(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive)
            .map_err(|source| SearchError::SubsystemError {
                subsystem: "fsfs.complete_generation.daemon_lock",
                source: Box::new(io::Error::from(source)),
            })?;
        // Never steal a socket from a legacy/non-cooperating server. In
        // particular, a successful lock is not proof that an existing socket
        // is stale. Crash recovery requires explicitly removing a verified
        // stale endpoint, not an unbounded connect probe or blind unlink.
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                return Err(complete_cli_error(
                    "daemon_socket",
                    "socket path already exists; stop its owner or remove only a verified stale socket",
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let listener = UnixListener::bind(&path)?;
        let metadata = fs::symlink_metadata(&path)?;
        let bound = Self {
            listener,
            path,
            identity: (metadata.dev(), metadata.ino()),
            _lock: lock,
        };
        // Construct the guard before fallible setup so setup errors also close
        // and remove precisely this endpoint while retaining singleton ownership.
        fs::set_permissions(&bound.path, fs::Permissions::from_mode(0o600))?;
        bound.listener.set_nonblocking(true)?;
        Ok(bound)
    }
}

impl Drop for BoundCompleteSocket {
    fn drop(&mut self) {
        if let Ok(metadata) = fs::symlink_metadata(&self.path)
            && metadata.file_type().is_socket()
            && (metadata.dev(), metadata.ino()) == self.identity
        {
            let _ = fs::remove_file(&self.path);
        }
    }
}

impl FsfsRuntime {
    /// Serve one buffered v3 request per connection until cancellation or idle
    /// expiry. No detached worker owns a reader, socket, or request after return.
    pub(super) async fn run_complete_generation_daemon(
        &self,
        cx: &Cx,
        root: &Path,
    ) -> SearchResult<()> {
        retained_search_checkpoint(cx)?;
        let socket_path = self.complete_generation_socket_path(root)?;
        if self.cli_input.daemon_stop {
            // Stop must remain usable when selection or model admission fails.
            // It must never acquire a listener lock or start a replacement.
            return control::stop(
                cx,
                &socket_path,
                Duration::from_millis(FSFS_DAEMON_CLIENT_TIMEOUT_MS),
            ).await;
        }
        // Admit before opening the transport: no listening socket claims a
        // ready service while its generation or semantic producer is invalid.
        let mut session = self.open_live_retained_search(cx, root).await?;
        retained_search_checkpoint(cx)?;
        let bound = BoundCompleteSocket::bind(socket_path)?;
        let mut cache = HashMap::new();
        let cache_enabled = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
        let idle_timeout = control::idle_timeout(self.cli_input.daemon_idle_timeout_ms);
        let peer_timeout = Duration::from_millis(FSFS_DAEMON_CLIENT_TIMEOUT_MS);
        let mut last_activity = Instant::now();
        loop {
            retained_search_checkpoint(cx)?;
            match bound.listener.accept() {
                Ok((mut peer, _)) => {
                    peer.set_nonblocking(true)?;
                    let session = &mut session;
                    let cache = &mut cache;
                    let result = serve_peer(cx, &mut peer, peer_timeout, |request| async move {
                        // A cached request must pass selection admission too.
                        // Refresh never swaps on failure and we never consult
                        // old cache/resources after a failed refresh.
                        if session.refresh(cx).await? {
                            cache.clear();
                        }
                        session.reader.runtime.execute_search_serve_request(
                            cx,
                            request,
                            &mut session.reader.resources,
                            cache,
                            cache_enabled,
                        ).await
                    }).await;
                    match result {
                        Ok(PeerOutcome::Shutdown) => return Ok(()),
                        Ok(PeerOutcome::Search) => {}
                        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                        Err(error) => {
                            // A broken/slow client does not take down the listener
                            // or strand a task holding the selected generation.
                            tracing::debug!(%error, "complete-generation daemon client ended");
                        }
                    }
                    last_activity = Instant::now();
                }
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    if idle_timeout.is_some_and(|timeout| last_activity.elapsed() >= timeout) {
                        return Ok(());
                    }
                    asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL).await;
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
    }

    pub(super) fn complete_generation_socket_path(&self, root: &Path) -> SearchResult<PathBuf> {
        let root = fs::canonicalize(root)?;
        let path = self.cli_input.daemon_socket.as_ref().map_or_else(
            || root.join(FSFS_DAEMON_SOCKET_FILE),
            |path| if path.is_absolute() { path.clone() } else { root.join(path) },
        );
        // Custom names are supported at the store root only. Do not create a
        // socket or its lock inside a sealed bundle or an unrelated directory.
        let parent = path.parent().ok_or_else(|| {
            complete_cli_error("daemon_socket", "socket must have a parent directory")
        })?;
        if fs::canonicalize(parent)? != root
            || path.extension().and_then(|extension| extension.to_str()) != Some("sock")
        {
            return Err(complete_cli_error(
                "daemon_socket",
                "complete-generation sockets must end in .sock and live directly in the store root",
            ));
        }
        let name = path.file_name().ok_or_else(|| {
            complete_cli_error("daemon_socket", "socket must have a file name")
        })?;
        Ok(root.join(name))
    }
}

/// The execution closure is shared with the ordinary serve implementation;
/// transport errors never trigger an in-process retry or a stale-reader query.
async fn serve_peer<T, F, Fut>(
    cx: &Cx,
    peer: &mut UnixStream,
    timeout: Duration,
    execute: F,
) -> SearchResult<PeerOutcome>
where
    T: Serialize,
    F: FnOnce(SearchServeRequest) -> Fut,
    Fut: Future<Output = SearchResult<T>>,
{
    let bytes = read_request(cx, peer, timeout).await?;
    match control::is_shutdown_request(&bytes) {
        Ok(true) => {
            // No model/search work and no selection refresh for an explicit
            // control. A failed acknowledgment never claims a successful stop.
            write_response(cx, peer, control::SHUTDOWN_RESPONSE, timeout).await?;
            return Ok(PeerOutcome::Shutdown);
        }
        Ok(false) => {}
        Err(error) => {
            let response = encode_response(&FsfsRuntime::search_serve_error_response(
                "", "full", error.to_string(),
            ))?;
            write_response(cx, peer, &response, timeout).await?;
            return Ok(PeerOutcome::Search);
        }
    }
    let request = std::str::from_utf8(&bytes)
        .map_err(|_| complete_cli_error("daemon_request", "request is not UTF-8"))
        .and_then(|raw| FsfsRuntime::parse_search_serve_request(raw.trim()));
    let response = match request {
        Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
            "", "full", error.to_string(),
        ))?,
        Ok(request) => {
            let query = request.query.clone();
            let mode = request.mode.clone().unwrap_or_else(|| "full".to_owned());
            if request.stream {
                // Never disguise a buffered response as progressive frames.
                encode_response(&FsfsRuntime::search_serve_error_response(
                    query, mode,
                    "complete-generation socket serving currently supports buffered requests; use direct search --no-daemon --stream for progressive phases",
                ))?
            } else {
                retained_search_checkpoint(cx)?;
                match run_request(cx, timeout, execute(request)).await {
                    Ok(response) => match encode_response(&response) {
                        Ok(bytes) => bytes,
                        Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
                            query, mode, error.to_string(),
                        ))?,
                    },
                    Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                    Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
                        query, mode, error.to_string(),
                    ))?,
                }
            }
        }
    };
    write_response(cx, peer, &response, timeout).await?;
    Ok(PeerOutcome::Search)
}

/// Own the request future until completion, cancellation or its deadline.
///
/// Read/write timeouts alone do not bound an asynchronous admission or search
/// that never completes. Periodic checkpoints also observe a shutdown even
/// when that future never wakes itself. Dropping a timed-out future releases
/// its borrows before the next peer; it must not cancel the shared daemon Cx.
/// This is cooperative: a synchronous poll that blocks cannot be preempted.
async fn run_request<T>(
    cx: &Cx,
    timeout: Duration,
    request: impl Future<Output = SearchResult<T>>,
) -> SearchResult<T> {
    let mut request = pin!(request);
    let mut deadline = pin!(asupersync::time::sleep(cx.now(), timeout));
    loop {
        let mut tick = pin!(asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL));
        let result = poll_fn(|task| {
            if let Err(error) = retained_search_checkpoint(cx) {
                return Poll::Ready(Err(error));
            }
            if deadline.as_mut().poll(task).is_ready() {
                return Poll::Ready(Err(io::Error::new(
                    ErrorKind::TimedOut,
                    "complete-generation daemon search deadline exceeded",
                ).into()));
            }
            if let Poll::Ready(result) = request.as_mut().poll(task) {
                return Poll::Ready(retained_search_checkpoint(cx).and(result).map(Some));
            }
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

fn encode_response<T: Serialize>(response: &T) -> SearchResult<Vec<u8>> {
    let mut bytes = Vec::new();
    // Same schema and byte limit as stdio, with no partially encoded record
    // exposed when serialization or the response-size bound fails.
    emit_complete_serve_line(response, &mut bytes)?;
    Ok(bytes)
}

async fn read_request(cx: &Cx, peer: &mut UnixStream, timeout: Duration) -> SearchResult<Vec<u8>> {
    let started = Instant::now();
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        retained_search_checkpoint(cx)?;
        if started.elapsed() >= timeout {
            return Err(io::Error::new(ErrorKind::TimedOut, "daemon request deadline exceeded").into());
        }
        let remaining = (FSFS_DAEMON_REQUEST_MAX_BYTES + 1 - bytes.len()).min(chunk.len());
        match peer.read(&mut chunk[..remaining]) {
            Ok(0) if bytes.is_empty() => {
                return Err(io::Error::new(ErrorKind::UnexpectedEof, "empty daemon request").into());
            }
            Ok(0) => return Ok(bytes),
            Ok(count) => {
                let newline = chunk[..count].iter().position(|byte| *byte == b'\n');
                let end = newline.map_or(count, |offset| offset + 1);
                bytes.extend_from_slice(&chunk[..end]);
                if bytes.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
                    return Err(complete_cli_error("daemon_request", "request exceeds 1 MiB limit"));
                }
                if newline.is_some() {
                    return Ok(bytes);
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL).await;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

async fn write_response(
    cx: &Cx,
    peer: &mut UnixStream,
    mut bytes: &[u8],
    timeout: Duration,
) -> SearchResult<()> {
    let started = Instant::now();
    while !bytes.is_empty() {
        retained_search_checkpoint(cx)?;
        if started.elapsed() >= timeout {
            return Err(io::Error::new(ErrorKind::TimedOut, "daemon response deadline exceeded").into());
        }
        match peer.write(bytes) {
            Ok(0) => return Err(io::Error::new(ErrorKind::WriteZero, "daemon peer closed").into()),
            Ok(count) => bytes = &bytes[count..],
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL).await;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::io::BufRead;
    use std::net::Shutdown;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct MarkReleased(Arc<AtomicBool>);

    impl Drop for MarkReleased {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn request_deadline_releases_an_unresponsive_future_without_cancelling_daemon() {
        run_test_with_cx(|cx| async move {
            let released = Arc::new(AtomicBool::new(false));
            let owned = MarkReleased(Arc::clone(&released));
            let result: SearchResult<()> = run_request(&cx, Duration::from_millis(1), async move {
                let _owned = owned;
                std::future::pending().await
            }).await;
            assert!(matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut));
            assert!(released.load(Ordering::SeqCst));
            retained_search_checkpoint(&cx).expect("one timeout must not cancel the daemon");
            assert_eq!(run_request(&cx, Duration::from_secs(1), async { Ok(42) }).await.unwrap(), 42);
        });
    }

    #[test]
    fn expired_request_is_dropped_without_polling_or_delivering_results() {
        run_test_with_cx(|cx| async move {
            let released = Arc::new(AtomicBool::new(false));
            let owned = MarkReleased(Arc::clone(&released));
            let mut polled = false;
            let result = run_request(&cx, Duration::ZERO, async {
                let _owned = owned;
                polled = true;
                Ok(())
            }).await;
            assert!(matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut));
            assert!(!polled);
            assert!(released.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn request_checkpoint_observes_cancellation_without_an_inner_wakeup() {
        run_test_with_cx(|cx| async move {
            let released = Arc::new(AtomicBool::new(false));
            let owned = MarkReleased(Arc::clone(&released));
            let result: SearchResult<()> = run_request(&cx, Duration::from_secs(1), async {
                let _owned = owned;
                cx.set_cancel_requested(true);
                std::future::pending().await
            }).await;
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert!(released.load(Ordering::SeqCst));
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn request_execution_preserves_the_underlying_failure() {
        run_test_with_cx(|cx| async move {
            let result: SearchResult<()> = run_request(&cx, Duration::from_secs(1), async {
                Err(complete_cli_error("test_admission", "original refusal"))
            }).await;
            assert!(matches!(result, Err(SearchError::InvalidConfig { field, reason, .. })
                if field == "complete_generation.test_admission" && reason == "original refusal"));
        });
    }

    #[test]
    fn socket_reports_search_deadline_and_accepts_a_subsequent_request() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(b"{\"query\":\"stall\"}\n");
            let result = serve_peer::<serde_json::Value, _, _>(
                &cx, &mut server, Duration::from_millis(100), |_| std::future::pending(),
            ).await.unwrap();
            assert_eq!(result, PeerOutcome::Search);
            let failure = response(client);
            assert_eq!(failure["ok"], false);
            assert!(failure.to_string().contains("search deadline exceeded"));

            let (client, mut server) = pair(b"{\"query\":\"healthy\"}\n");
            serve_peer(&cx, &mut server, Duration::from_secs(1), |_| async {
                Ok(serde_json::json!({"ok": true}))
            }).await.unwrap();
            assert_eq!(response(client)["ok"], true);
        });
    }

    fn pair(request: &[u8]) -> (UnixStream, UnixStream) {
        let (mut client, server) = UnixStream::pair().unwrap();
        client.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        client.write_all(request).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        server.set_nonblocking(true).unwrap();
        (client, server)
    }

    fn response(client: UnixStream) -> serde_json::Value {
        let mut line = String::new();
        std::io::BufReader::new(client).read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    #[test]
    fn socket_owner_excludes_duplicates_and_releases_endpoint_not_lock_inode() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.sock");
        let first = BoundCompleteSocket::bind(path.clone()).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(BoundCompleteSocket::bind(path.clone()).is_err());
        let lock_inode = fs::metadata(path.with_extension("lock")).unwrap().ino();
        drop(first);
        assert!(!path.exists());
        assert_eq!(fs::metadata(path.with_extension("lock")).unwrap().ino(), lock_inode);
        let second = BoundCompleteSocket::bind(path.clone()).unwrap();
        drop(second);
        assert!(!path.exists());
    }

    #[test]
    fn socket_startup_does_not_remove_existing_files_or_noncooperating_listeners() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("file.sock");
        fs::write(&file, b"unrelated data").unwrap();
        assert!(BoundCompleteSocket::bind(file.clone()).is_err());
        assert_eq!(fs::read(&file).unwrap(), b"unrelated data");
        let socket = directory.path().join("other.sock");
        let _other = UnixListener::bind(&socket).unwrap();
        let inode = fs::metadata(&socket).unwrap().ino();
        assert!(BoundCompleteSocket::bind(socket.clone()).is_err());
        assert_eq!(fs::metadata(socket).unwrap().ino(), inode);
    }

    #[test]
    fn socket_cleanup_preserves_replaced_endpoint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.sock");
        let owner = BoundCompleteSocket::bind(path.clone()).unwrap();
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"replacement evidence").unwrap();
        drop(owner);
        assert_eq!(fs::read(path).unwrap(), b"replacement evidence");
    }

    #[test]
    fn socket_lock_symlink_is_not_followed() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.sock");
        let outside = directory.path().join("outside");
        fs::write(&outside, b"sentinel").unwrap();
        std::os::unix::fs::symlink(&outside, path.with_extension("lock")).unwrap();
        assert!(BoundCompleteSocket::bind(path.clone()).is_err());
        assert!(!path.exists());
        assert_eq!(fs::read(outside).unwrap(), b"sentinel");
    }

    #[test]
    fn socket_peer_round_trips_one_request_and_does_not_emit_stdio_ready() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(b"{\"query\":\"sharedtoken\",\"limit\":3}\n");
            serve_peer(&cx, &mut server, Duration::from_secs(2), |request| async move {
                assert_eq!(request.query, "sharedtoken");
                assert_eq!(request.limit, Some(3));
                Ok(serde_json::json!({"ok":true,"query":request.query}))
            }).await.unwrap();
            let value = response(client);
            assert_eq!(value["ok"], true);
            assert!(value.get("event").is_none());
        });
    }

    #[test]
    fn socket_bad_input_and_progressive_requests_do_not_execute_search() {
        run_test_with_cx(|cx| async move {
            for raw in [b"\xff\n".as_slice(), b"{invalid\n", b"{\"query\":\"x\",\"stream\":true}\n"] {
                let (client, mut server) = pair(raw);
                serve_peer(&cx, &mut server, Duration::from_secs(2), |_| async {
                    panic!("refused request must not execute");
                    #[allow(unreachable_code)]
                    Ok(serde_json::Value::Null)
                }).await.unwrap();
                assert_eq!(response(client)["ok"], false);
            }
        });
    }

    #[test]
    fn socket_cancelled_read_and_write_return_without_waiting_or_emitting() {
        run_test_with_cx(|cx| async move {
            let (mut client, mut server) = UnixStream::pair().unwrap();
            server.set_nonblocking(true).unwrap();
            client.set_nonblocking(true).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(read_request(&cx, &mut server, Duration::from_secs(60)).await,
                Err(SearchError::Cancelled { .. })));
            assert!(matches!(write_response(&cx, &mut server, b"not visible", Duration::from_secs(60)).await,
                Err(SearchError::Cancelled { .. })));
            let error = client.read(&mut [0_u8; 16]).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::WouldBlock);
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn explicit_shutdown_is_acknowledged_without_executing_search() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(control::SHUTDOWN_REQUEST);
            let outcome = serve_peer(&cx, &mut server, Duration::from_secs(2), |_| async {
                panic!("shutdown must not execute search"); // ubs:ignore — cfg(test) assertion.
                #[allow(unreachable_code)]
                Ok(serde_json::Value::Null)
            }).await.unwrap();
            assert_eq!(outcome, PeerOutcome::Shutdown);
            let acknowledgment = response(client);
            assert_eq!(acknowledgment["ok"], true);
            assert_eq!(acknowledgment["event"], "shutdown");
        });
    }

    #[test]
    fn malformed_control_neither_stops_nor_executes_search() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(
                b"{\"fsfs_complete_daemon\":\"shutdown\",\"version\":2}\n",
            );
            let outcome = serve_peer(&cx, &mut server, Duration::from_secs(2), |_| async {
                panic!("malformed control must not execute search"); // ubs:ignore — cfg(test) assertion.
                #[allow(unreachable_code)]
                Ok(serde_json::Value::Null)
            }).await.unwrap();
            assert_eq!(outcome, PeerOutcome::Search);
            assert_eq!(response(client)["ok"], false);
        });
    }

    #[test]
    fn socket_deadlines_apply_to_idle_read_and_response_write() {
        run_test_with_cx(|cx| async move {
            let (_client, mut server) = UnixStream::pair().unwrap();
            server.set_nonblocking(true).unwrap();
            for result in [
                read_request(&cx, &mut server, Duration::ZERO).await.map(|_| ()),
                write_response(&cx, &mut server, b"response", Duration::ZERO).await,
            ] {
                assert!(matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut));
            }
        });
    }
}

#[cfg(all(test, not(feature = "embedded-models")))]
mod generation_tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use crate::{CliCommand, CliInput, FsfsConfig, InterfaceMode};
    use crate::generation_store::{
        COMPLETE_GENERATION_POINTER, CompleteGenerationStore, GenerationPublication,
    };
    use std::io::BufRead;

    struct CancelOnDrop(Cx);

    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            self.0.set_cancel_requested(true);
        }
    }

    #[test]
    fn complete_daemon_refreshes_real_search_and_cache_after_publication_and_repair() {
        run_test_with_cx(|cx| async move {
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
                command: CliCommand::Daemon,
                target_path: Some(source.clone()),
                index_dir: Some(root.clone()),
                quiet: true,
                ..CliInput::default()
            });
            assert!(matches!(runtime.rebuild_retained_generation(&cx, &root).await.unwrap(),
                GenerationPublication::Durable(_)));
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let first = fs::read(&pointer).unwrap();
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            assert!(matches!(runtime.rebuild_retained_generation(&cx, &root).await.unwrap(),
                GenerationPublication::Durable(_)));
            let successor = fs::read(&pointer).unwrap();
            fs::write(&pointer, &first).unwrap();
            let endpoint = root.join(FSFS_DAEMON_SOCKET_FILE);
            let client_endpoint = endpoint.clone();
            let client_root = root.clone();
            let cancel = CancelOnDrop(cx.clone());
            let worker = std::thread::spawn(move || {
                // Even a client assertion failure cancels the owning server;
                // every socket read/write and startup wait has a deadline.
                let _cancel = cancel;
                let request = || -> serde_json::Value {
                    let started = Instant::now();
                    let mut stream = loop {
                        match UnixStream::connect(&client_endpoint) {
                            Ok(stream) => break stream,
                            Err(error) => {
                                assert!(started.elapsed() < Duration::from_secs(10), "{error}");
                                std::thread::sleep(Duration::from_millis(5));
                            }
                        }
                    };
                    stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                    stream.set_write_timeout(Some(Duration::from_secs(10))).unwrap();
                    stream.write_all(b"{\"query\":\"sharedtoken\",\"limit\":10}\n").unwrap();
                    let mut line = String::new();
                    std::io::BufReader::new(stream).read_line(&mut line).unwrap();
                    serde_json::from_str(&line).unwrap()
                };
                let check = |value: serde_json::Value, count, cached| {
                    assert_eq!(value["ok"], true, "{value}");
                    assert_eq!(value["cached"], cached);
                    let phases = value["payloads"].as_array().unwrap();
                    assert_eq!(phases.last().unwrap()["hits"].as_array().unwrap().len(), count);
                };
                let caching = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
                check(request(), 1, false);
                check(request(), 1, caching);
                let temporary = client_root.join("test-pointer-switch");
                fs::write(&temporary, b"invalid selection").unwrap();
                fs::rename(&temporary, &pointer).unwrap();
                let refused = request();
                assert_eq!(refused["ok"], false);
                assert_eq!(refused["cached"], false);
                assert!(refused["payloads"].as_array().unwrap().is_empty());
                fs::write(&temporary, successor).unwrap();
                fs::rename(&temporary, &pointer).unwrap();
                check(request(), 2, false);
                check(request(), 2, caching);
            });
            let result = runtime.run_mode_with_complete_generations(
                &cx, InterfaceMode::Cli, None, false,
            ).await;
            worker.join().unwrap();
            assert!(matches!(result, Err(SearchError::Cancelled { .. })), "{result:?}");
            cx.set_cancel_requested(false);
            assert!(!endpoint.exists(), "owning command must release the socket");
            let old = pinned.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(old.last().unwrap().hits.len(), 1);
            assert!(CompleteGenerationStore::open(&cx, &root).unwrap().active(&cx).unwrap().is_some());
        });
    }
}
