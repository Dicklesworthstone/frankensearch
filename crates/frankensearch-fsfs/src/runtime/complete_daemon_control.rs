//! Complete-daemon lifecycle messages and the bounded, cancellable client lane.
//!
//! A stop request addresses the owned socket, never a saved PID. It neither
//! creates a listener/lock nor opens a generation or model. Ordinary queries
//! cannot accidentally become control messages, and malformed controls fail
//! closed without reaching search execution.

use std::fs;
use std::io::{self, ErrorKind, Read};
use std::os::unix::fs::FileTypeExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::Deserialize;

use super::{
    FSFS_DAEMON_IDLE_TIMEOUT_MS, FSFS_DAEMON_REQUEST_MAX_BYTES, IO_POLL_INTERVAL,
    complete_cli_error, retained_search_checkpoint, write_response,
};

const CONTROL_PROTOCOL: &str = "fsfs.complete_generation.control";
const CONTROL_RESPONSE_MAX_BYTES: usize = 4096;
pub(super) const SHUTDOWN_REQUEST: &[u8] =
    b"{\"fsfs_complete_daemon\":\"shutdown\",\"version\":1}\n";
pub(super) const SHUTDOWN_RESPONSE: &[u8] =
    b"{\"ok\":true,\"event\":\"shutdown\",\"version\":1,\"protocol\":\"fsfs.complete_generation.control\"}\n";

#[derive(Deserialize)]
enum ShutdownCommand {
    #[serde(rename = "shutdown")]
    Shutdown,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShutdownRequest {
    fsfs_complete_daemon: ShutdownCommand,
    version: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShutdownResponse {
    ok: bool,
    event: ShutdownCommand,
    version: u8,
    protocol: String,
}

/// Absence of the control discriminator leaves the established query parser
/// authoritative. Presence requires the exact, versioned control schema.
pub(super) fn is_shutdown_request(bytes: &[u8]) -> SearchResult<bool> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return Ok(false);
    };
    if value.get("fsfs_complete_daemon").is_none() {
        return Ok(false);
    }
    // Deserialize the original bytes, not `value`: this also rejects duplicate
    // keys instead of letting a last-key-wins object authorize shutdown.
    let request: ShutdownRequest = serde_json::from_slice(bytes).map_err(|error| {
        complete_cli_error(
            "daemon_control",
            &format!("invalid control request: {error}"),
        )
    })?;
    let ShutdownCommand::Shutdown = request.fsfs_complete_daemon;
    if request.version != 1 {
        return Err(complete_cli_error(
            "daemon_control",
            "unsupported control version",
        ));
    }
    Ok(true)
}

pub(super) fn idle_timeout(override_ms: Option<u64>) -> Option<Duration> {
    let milliseconds = override_ms.unwrap_or(FSFS_DAEMON_IDLE_TIMEOUT_MS);
    (milliseconds != 0).then(|| Duration::from_millis(milliseconds))
}

pub(super) async fn stop(cx: &Cx, path: &Path, timeout: Duration) -> SearchResult<()> {
    let bytes = exchange(
        cx,
        path,
        SHUTDOWN_REQUEST,
        CONTROL_RESPONSE_MAX_BYTES,
        timeout,
    )
    .await?;
    let response: ShutdownResponse = serde_json::from_slice(&bytes).map_err(|error| {
        complete_cli_error(
            "daemon_control",
            &format!("invalid shutdown acknowledgment: {error}"),
        )
    })?;
    let ShutdownCommand::Shutdown = response.event;
    if !response.ok || response.version != 1 || response.protocol != CONTROL_PROTOCOL {
        return Err(complete_cli_error(
            "daemon_control",
            "peer did not acknowledge complete-generation daemon shutdown",
        ));
    }
    Ok(())
}

/// One request, one bounded newline-terminated response. Connect, request write
/// and response read share a single deadline. No retry can replay a request
/// whose delivery outcome is unknown, and no failure starts a replacement daemon.
pub(super) async fn exchange(
    cx: &Cx,
    path: &Path,
    request: &[u8],
    response_limit: usize,
    timeout: Duration,
) -> SearchResult<Vec<u8>> {
    retained_search_checkpoint(cx)?;
    let started = Instant::now();
    remaining(started, timeout)?;
    if request.len() > FSFS_DAEMON_REQUEST_MAX_BYTES
        || request.last() != Some(&b'\n')
        || request[..request.len() - 1].contains(&b'\n')
    {
        return Err(complete_cli_error(
            "daemon_request",
            "expected one bounded request frame",
        ));
    }
    if !fs::symlink_metadata(path)?.file_type().is_socket() {
        return Err(complete_cli_error(
            "daemon_socket",
            "expected a non-symlink Unix socket",
        ));
    }
    let mut peer = connect(cx, path, started, timeout).await?;
    write_response(cx, &mut peer, request, remaining(started, timeout)?).await?;
    let bytes = read_response(cx, &mut peer, response_limit, remaining(started, timeout)?).await?;
    retained_search_checkpoint(cx)?;
    remaining(started, timeout)?;
    Ok(bytes)
}

pub(super) async fn connect(
    cx: &Cx,
    path: &Path,
    started: Instant,
    timeout: Duration,
) -> SearchResult<UnixStream> {
    use rustix::net::{AddressFamily, SocketAddrUnix, SocketFlags, SocketType};

    retained_search_checkpoint(cx)?;
    remaining(started, timeout)?;
    let address = SocketAddrUnix::new(path).map_err(io::Error::from)?;
    // Set NONBLOCK before connect: setting it on an already-connected std
    // stream leaves a full accept backlog able to block the owning task.
    let socket = rustix::net::socket_with(
        AddressFamily::UNIX,
        SocketType::STREAM,
        SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
        None,
    )
    .map_err(io::Error::from)?;
    match rustix::net::connect(&socket, &address) {
        Ok(()) => return Ok(UnixStream::from(socket)),
        Err(rustix::io::Errno::INPROGRESS) => {}
        // In particular, Unix EAGAIN means a full listener backlog, not an
        // established connection. Return it rather than inventing success or
        // retrying an operation with an ambiguous delivery outcome.
        Err(error) => return Err(io::Error::from(error).into()),
    }
    let peer = UnixStream::from(socket);
    loop {
        retained_search_checkpoint(cx)?;
        let left = remaining(started, timeout)?;
        if let Some(error) = peer.take_error()? {
            return Err(error.into());
        }
        match peer.peer_addr() {
            Ok(_) => return Ok(peer),
            Err(error) if error.kind() == ErrorKind::NotConnected => {}
            Err(error) => return Err(error.into()),
        }
        asupersync::time::sleep(cx.now(), left.min(IO_POLL_INTERVAL)).await;
    }
}

pub(super) fn remaining(started: Instant, timeout: Duration) -> SearchResult<Duration> {
    timeout
        .checked_sub(started.elapsed())
        .filter(|left| !left.is_zero())
        .ok_or_else(|| {
            SearchError::Io(io::Error::new(
                ErrorKind::TimedOut,
                "complete-daemon client deadline exceeded",
            ))
        })
}

async fn read_response(
    cx: &Cx,
    peer: &mut UnixStream,
    limit: usize,
    timeout: Duration,
) -> SearchResult<Vec<u8>> {
    let started = Instant::now();
    let capacity = limit.checked_add(1).ok_or_else(|| {
        complete_cli_error("daemon_response", "response bound overflows address space")
    })?;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 8192];
    loop {
        retained_search_checkpoint(cx)?;
        let left = remaining(started, timeout)?;
        let count_limit = (capacity - bytes.len()).min(chunk.len());
        match peer.read(&mut chunk[..count_limit]) {
            Ok(0) => {
                return Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "daemon closed before a complete response frame",
                )
                .into());
            }
            Ok(count) => {
                bytes.extend_from_slice(&chunk[..count]);
                if bytes.len() > limit {
                    return Err(complete_cli_error(
                        "daemon_response",
                        "response exceeds its byte limit",
                    ));
                }
                if let Some(offset) = chunk[..count].iter().position(|byte| *byte == b'\n') {
                    let newline = bytes.len() - count + offset;
                    if bytes[newline + 1..]
                        .iter()
                        .any(|byte| !byte.is_ascii_whitespace())
                    {
                        return Err(complete_cli_error(
                            "daemon_response",
                            "multiple response frames on one connection",
                        ));
                    }
                    bytes.truncate(newline + 1);
                    return Ok(bytes);
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                asupersync::time::sleep(cx.now(), left.min(IO_POLL_INTERVAL)).await;
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use std::io::Write;
    use std::net::Shutdown;
    use std::os::unix::net::UnixListener;

    #[test]
    fn lifecycle_controls_are_versioned_and_not_search_queries() {
        assert!(is_shutdown_request(SHUTDOWN_REQUEST).unwrap());
        for query in [
            b"shutdown".as_slice(),
            b"{\"query\":\"shutdown\"}\n",
            b"{broken",
        ] {
            assert!(!is_shutdown_request(query).unwrap());
        }
        for invalid in [
            br#"{"fsfs_complete_daemon":"shutdown","version":2}"#.as_slice(),
            br#"{"fsfs_complete_daemon":"shutdown","version":1,"query":"x"}"#,
            br#"{"fsfs_complete_daemon":"shutdown","version":1,"version":1}"#,
            br#"{"fsfs_complete_daemon":"restart","version":1}"#,
            br#"{"fsfs_complete_daemon":"shutdown"}"#,
        ] {
            assert!(is_shutdown_request(invalid).is_err());
        }
    }

    #[test]
    fn idle_timeout_override_preserves_zero_as_keep_alive() {
        assert_eq!(idle_timeout(Some(0)), None);
        assert_eq!(idle_timeout(Some(17)), Some(Duration::from_millis(17)));
        assert_eq!(
            idle_timeout(None),
            idle_timeout(Some(FSFS_DAEMON_IDLE_TIMEOUT_MS))
        );
    }

    #[test]
    fn stopping_absent_or_non_socket_endpoints_never_creates_or_removes_files() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("search.sock");
            assert!(stop(&cx, &path, Duration::from_secs(1)).await.is_err());
            assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
            fs::write(&path, b"sentinel").unwrap();
            assert!(stop(&cx, &path, Duration::from_secs(1)).await.is_err());
            assert_eq!(fs::read(&path).unwrap(), b"sentinel");
            assert!(!path.with_extension("lock").exists());
        });
    }

    #[test]
    fn cancelled_and_expired_client_never_connects_to_listener() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("search.sock");
            let listener = UnixListener::bind(&path).unwrap();
            listener.set_nonblocking(true).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                stop(&cx, &path, Duration::from_secs(60)).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(matches!(stop(&cx, &path, Duration::ZERO).await,
                Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut));
            assert_eq!(listener.accept().unwrap_err().kind(), ErrorKind::WouldBlock);
        });
    }

    #[test]
    fn response_reader_refuses_truncated_oversized_and_multiple_frames() {
        run_test_with_cx(|cx| async move {
            for (response, limit) in [
                (b"{}".as_slice(), 20),
                (b"12345\n".as_slice(), 5),
                (b"{}\n{}\n".as_slice(), 20),
            ] {
                let (mut peer, mut sender) = UnixStream::pair().unwrap();
                peer.set_nonblocking(true).unwrap();
                sender.write_all(response).unwrap();
                sender.shutdown(Shutdown::Write).unwrap();
                assert!(
                    read_response(&cx, &mut peer, limit, Duration::from_secs(1))
                        .await
                        .is_err()
                );
            }
            let (mut peer, mut sender) = UnixStream::pair().unwrap();
            peer.set_nonblocking(true).unwrap();
            sender.write_all(b"{}\n").unwrap();
            assert_eq!(
                read_response(&cx, &mut peer, 3, Duration::from_secs(1))
                    .await
                    .unwrap(),
                b"{}\n"
            );
        });
    }

    #[test]
    fn response_read_observes_cancellation_and_deadline() {
        run_test_with_cx(|cx| async move {
            let (mut peer, _sender) = UnixStream::pair().unwrap();
            peer.set_nonblocking(true).unwrap();
            cx.set_cancel_requested(true);
            assert!(matches!(
                read_response(&cx, &mut peer, 20, Duration::from_secs(60)).await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert!(
                matches!(read_response(&cx, &mut peer, 20, Duration::ZERO).await,
                Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut)
            );
        });
    }

    #[test]
    fn stop_command_round_trips_without_admitting_a_corrupt_generation() {
        run_test_with_cx(|cx| async move {
            use super::super::BoundCompleteSocket;
            use crate::{CliCommand, CliInput, FsfsConfig, FsfsRuntime};
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().to_path_buf();
            let path = root.join("controlled.sock");
            let pointer = root.join(crate::generation_store::COMPLETE_GENERATION_POINTER);
            fs::write(&pointer, b"intentionally corrupt selection").unwrap();
            let owner = BoundCompleteSocket::bind(path.clone()).unwrap();
            let server = std::thread::spawn(move || {
                let started = Instant::now();
                let mut peer = loop {
                    match owner.listener.accept() {
                        Ok((peer, _)) => break peer,
                        Err(error) if error.kind() == ErrorKind::WouldBlock => {
                            assert!(started.elapsed() < Duration::from_secs(3));
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("accept failed: {error}"), // ubs:ignore — cfg(test) assertion.
                    }
                };
                peer.set_nonblocking(false).unwrap();
                peer.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
                peer.set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = vec![0; SHUTDOWN_REQUEST.len()];
                peer.read_exact(&mut request).unwrap();
                assert_eq!(request, SHUTDOWN_REQUEST);
                assert!(is_shutdown_request(&request).unwrap());
                peer.write_all(SHUTDOWN_RESPONSE).unwrap();
                // The socket guard tears down only its own endpoint. The stop
                // client never removes a socket or replaces the singleton lock.
            });
            let runtime = FsfsRuntime::new(FsfsConfig::default()).with_cli_input(CliInput {
                command: CliCommand::Daemon,
                daemon_stop: true,
                daemon_socket: Some(path.clone()),
                index_dir: Some(root.clone()),
                ..CliInput::default()
            });
            let result = runtime.run_complete_generation_daemon(&cx, &root).await;
            server.join().unwrap();
            result.unwrap();
            assert!(!path.exists());
            assert_eq!(
                fs::read(&pointer).unwrap(),
                b"intentionally corrupt selection"
            );
            assert!(!root.join("generations").exists());
        });
    }
}
