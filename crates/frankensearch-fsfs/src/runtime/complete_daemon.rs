//! Owned, bounded Unix transport for complete-generation search.
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::Serialize;

use super::super::{
    FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS, FSFS_DAEMON_CONNECT_MAX_ATTEMPTS,
    FSFS_DAEMON_CONNECT_RETRY_DELAY_MS, FSFS_DAEMON_IDLE_TIMEOUT_MS, FSFS_DAEMON_REQUEST_MAX_BYTES,
    SearchExecutionFlags, SearchServeRequest, daemon_socket_path_capacity,
};
use super::{
    FsfsRuntime, complete_cli_error, emit_complete_serve_line, pressure_timestamp_ms,
    retained_search_checkpoint,
};

const IO_POLL_INTERVAL: Duration = Duration::from_millis(10);
// Complete stores own their endpoint beside the selection pointer, outside
// every immutable generation. Legacy query sockets use a separate hashed path.
const FSFS_DAEMON_SOCKET_FILE: &str = "fsfs-query.sock";

#[path = "complete_daemon_control.rs"]
mod control;

#[path = "complete_daemon_stream.rs"]
mod streaming;

#[path = "complete_daemon_forward.rs"]
mod forwarding;

static STREAM_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
        // The default socket lives in its store root, so a root too deep for
        // sun_path cannot serve on it. Refuse before creating the lock,
        // instead of leaving it behind on bind()'s raw "shorter than SUN_LEN"
        // error.
        let capacity = daemon_socket_path_capacity();
        let length = path.as_os_str().len();
        if length >= capacity {
            return Err(SearchError::InvalidConfig {
                field: "complete_generation.daemon_socket".to_owned(),
                value: path.display().to_string(),
                reason: format!(
                    "the socket path is {length} bytes, but Unix sockets allow at most {}; name a shorter socket outside the store with --daemon-socket, or serve a store at a shorter path",
                    capacity - 1
                ),
            });
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path.with_extension("lock"))?;
        if !lock.metadata()?.is_file() {
            return Err(complete_cli_error(
                "daemon_lock",
                "daemon lock is not a regular file",
            ));
        }
        rustix::fs::flock(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive).map_err(
            |source| SearchError::SubsystemError {
                subsystem: "fsfs.complete_generation.daemon_lock",
                source: Box::new(io::Error::from(source)),
            },
        )?;
        // Never steal a socket from a legacy/non-cooperating server: a
        // successful lock alone is not proof that an existing socket is
        // stale. A crashed daemon, or a reboot, leaves its socket behind; it
        // is replaced only once verified stale. The lock rules out a
        // cooperating owner, one refused connect rules out any listener, and
        // nobody can bind a path while it exists. Any other file stays.
        match fs::symlink_metadata(&path) {
            Ok(metadata)
                if metadata.file_type().is_socket() && control::refuses_connections(&path) =>
            {
                fs::remove_file(&path)?;
                tracing::info!(
                    socket = %path.display(),
                    "removed a verified stale complete-generation daemon socket"
                );
            }
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

/// Polls whether a started daemon process has already exited.
pub(super) type DaemonExited = Box<dyn FnMut() -> io::Result<bool> + Send>;

/// What a test installs in place of the detached daemon process. It receives
/// the store root and the socket the daemon must bind.
#[cfg(test)]
pub(super) type TestDaemonSpawner = Box<dyn FnMut(&Path, &Path) -> Option<DaemonExited>>;

#[cfg(test)]
thread_local! {
    /// Tests cannot start the test binary as the daemon; with no spawner
    /// installed, nothing is started.
    static TEST_DAEMON_SPAWNER: std::cell::RefCell<Option<TestDaemonSpawner>> =
        std::cell::RefCell::new(None);
}

/// Installs a test spawner for this thread until the guard drops.
#[cfg(test)]
pub(super) struct TestDaemonSpawnerGuard;

#[cfg(test)]
impl TestDaemonSpawnerGuard {
    pub(super) fn install(spawner: TestDaemonSpawner) -> Self {
        TEST_DAEMON_SPAWNER.with(|slot| *slot.borrow_mut() = Some(spawner));
        Self
    }
}

#[cfg(test)]
impl Drop for TestDaemonSpawnerGuard {
    fn drop(&mut self) {
        TEST_DAEMON_SPAWNER.with(|slot| *slot.borrow_mut() = None);
    }
}

impl FsfsRuntime {
    /// Start this store's daemon as a legacy search starts its warm daemon,
    /// and report whether it accepts connections within the startup window.
    /// Model loading often outlasts the window on the first search, which then
    /// runs in process while the daemon finishes starting; later searches
    /// forward to it. A daemon that exits early (no models, a lost lock race,
    /// a refused socket path) ends the wait at once, and a failed spawn only
    /// costs the in-process search it was going to run anyway.
    pub(super) async fn start_complete_generation_daemon(
        &self,
        cx: &Cx,
        root: &Path,
        socket: &Path,
    ) -> SearchResult<bool> {
        if socket.as_os_str().len() >= daemon_socket_path_capacity() {
            return Ok(false);
        }
        let mut exited = match self.spawn_complete_generation_daemon(root, socket) {
            Ok(Some(exited)) => exited,
            Ok(None) => return Ok(false),
            Err(error) => {
                tracing::warn!(%error, "could not start the complete-generation daemon");
                return Ok(false);
            }
        };
        for _ in 0..FSFS_DAEMON_CONNECT_MAX_ATTEMPTS {
            retained_search_checkpoint(cx)?;
            if fs::symlink_metadata(socket).is_ok() && !control::refuses_connections(socket) {
                return Ok(true);
            }
            if exited().unwrap_or(true) {
                return Ok(false);
            }
            asupersync::time::sleep(
                cx.now(),
                Duration::from_millis(FSFS_DAEMON_CONNECT_RETRY_DELAY_MS),
            )
            .await;
        }
        Ok(false)
    }

    /// Ask the daemon on `socket` to stop, so that one started with this
    /// runtime's configuration can replace it, and wait as long as a start
    /// would for it to stop accepting. `false` means it did not stop in time
    /// (or refused the request); the caller then searches in process.
    pub(super) async fn retire_complete_generation_daemon(
        cx: &Cx,
        socket: &Path,
    ) -> SearchResult<bool> {
        let timeout = Duration::from_millis(FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS);
        if let Err(error) = control::stop(cx, socket, timeout).await {
            tracing::warn!(
                %error,
                socket = %socket.display(),
                "could not stop the store's query daemon; searching in process"
            );
            return Ok(false);
        }
        Self::socket_released(cx, socket, FSFS_DAEMON_CONNECT_MAX_ATTEMPTS).await
    }

    /// `fsfs daemon --stop`: ask the daemon on `socket` to stop, wait until it
    /// no longer accepts (so a following start can bind), and print a receipt
    /// in the shape the legacy `--stop` prints.
    async fn stop_complete_generation_daemon(&self, cx: &Cx, socket: &Path) -> SearchResult<()> {
        // As long as the legacy stop waits for its daemon's process to exit.
        const STOP_WAIT_ATTEMPTS: usize = 400;

        let started = Instant::now();
        control::stop(
            cx,
            socket,
            Duration::from_millis(FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS),
        )
        .await?;
        if !Self::socket_released(cx, socket, STOP_WAIT_ATTEMPTS).await? {
            return Err(SearchError::InvalidConfig {
                field: "complete_generation.daemon_stop".to_owned(),
                value: socket.display().to_string(),
                reason: "the query daemon acknowledged the stop but still accepts connections"
                    .to_owned(),
            });
        }
        let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        if self.cli_input.format == crate::OutputFormat::Table {
            println!(
                "daemon: stopped the query daemon on {} after {elapsed_ms}ms",
                socket.display()
            );
        } else {
            let receipt = serde_json::json!({
                "stopped": true,
                "socket": socket.display().to_string(),
                "elapsed_ms": elapsed_ms,
            });
            println!("{receipt}");
        }
        Ok(())
    }

    /// Wait, polling `attempts` times, until nothing accepts on `socket` (it
    /// is gone or refuses connections).
    async fn socket_released(cx: &Cx, socket: &Path, attempts: usize) -> SearchResult<bool> {
        for _ in 0..attempts {
            retained_search_checkpoint(cx)?;
            if fs::symlink_metadata(socket).is_err() || control::refuses_connections(socket) {
                return Ok(true);
            }
            asupersync::time::sleep(
                cx.now(),
                Duration::from_millis(FSFS_DAEMON_CONNECT_RETRY_DELAY_MS),
            )
            .await;
        }
        Ok(false)
    }

    // The daemon binds exactly the socket this client resolved, the store's
    // own or a named one.
    #[cfg(not(test))]
    fn spawn_complete_generation_daemon(
        &self,
        root: &Path,
        socket: &Path,
    ) -> SearchResult<Option<DaemonExited>> {
        let mut child = self.spawn_detached_fsfs([
            "daemon".into(),
            "--index-dir".into(),
            root.into(),
            "--daemon-socket".into(),
            socket.into(),
            "--idle-timeout-ms".into(),
            FSFS_DAEMON_IDLE_TIMEOUT_MS.to_string().into(),
        ])?;
        Ok(Some(Box::new(move || Ok(child.try_wait()?.is_some()))))
    }

    // Same signature as the process spawner it stands in for.
    #[cfg(test)]
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn spawn_complete_generation_daemon(
        &self,
        root: &Path,
        socket: &Path,
    ) -> SearchResult<Option<DaemonExited>> {
        Ok(TEST_DAEMON_SPAWNER.with(|slot| {
            slot.borrow_mut()
                .as_mut()
                .and_then(|spawn| spawn(root, socket))
        }))
    }

    /// Serve one buffered or progressive request per connection until cancellation or idle
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
            return self.stop_complete_generation_daemon(cx, &socket_path).await;
        }
        // Admit before opening the transport: no listening socket claims a
        // ready service while its generation or semantic producer is invalid.
        let mut session = self.open_live_retained_search(cx, root).await?;
        retained_search_checkpoint(cx)?;
        let bound = BoundCompleteSocket::bind(socket_path)?;
        let mut cache = HashMap::new();
        let cache_enabled = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
        let idle_timeout = control::idle_timeout(self.cli_input.daemon_idle_timeout_ms);
        let peer_timeout = Duration::from_millis(FSFS_DAEMON_CLIENT_IO_TIMEOUT_MS);
        let mut last_activity = Instant::now();
        loop {
            retained_search_checkpoint(cx)?;
            match bound.listener.accept() {
                Ok((mut peer, _)) => {
                    peer.set_nonblocking(true)?;
                    let session = &mut session;
                    let cache = &mut cache;
                    let bytes = match read_request(cx, &mut peer, peer_timeout).await {
                        Ok(bytes) => bytes,
                        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                        Err(error) => {
                            tracing::debug!(%error, "complete-generation daemon request ended");
                            last_activity = Instant::now();
                            continue;
                        }
                    };
                    if forwarding::is_forwarded(&bytes) {
                        // Forwarded and raw requests share the same admitted
                        // generation and bounded hot cache. The forwarder clears
                        // it before using a newly selected generation.
                        let result = Box::pin(forwarding::serve(
                            cx,
                            self,
                            session,
                            &mut peer,
                            &bytes,
                            peer_timeout,
                            cache_enabled.then_some(cache),
                        ))
                        .await;
                        match result {
                            Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                            Err(error) => {
                                tracing::debug!(%error, "complete-generation forwarded client ended");
                            }
                            Ok(_) => {}
                        }
                        last_activity = Instant::now();
                        continue;
                    }
                    let result = Box::pin(serve_peer_bytes(
                        cx,
                        &mut peer,
                        peer_timeout,
                        &bytes,
                        |request| self.complete_request_delivery_budget(request),
                        |request, output| async move {
                            // A cached request must pass selection admission too.
                            // Refresh never swaps on failure and we never consult
                            // old cache/resources after a failed refresh.
                            if session.refresh(cx).await? {
                                cache.clear();
                            }
                            if let Some(mut output) = output {
                                // Reuse direct streaming, including its producer
                                // annotations, phase ordering and terminal frames.
                                // Runtime paths and resources retain one generation.
                                let mut runtime = session.reader.runtime.clone();
                                runtime.cli_input.command = crate::CliCommand::Search;
                                runtime.cli_input.format = crate::OutputFormat::Jsonl;
                                runtime.cli_input.stream = true;
                                runtime.cli_input.daemon = false;
                                runtime.cli_input.daemon_socket = None;
                                runtime.cli_input.query = Some(request.query.clone());
                                let limit =
                                    request.limit.unwrap_or(runtime.config.search.default_limit);
                                let stream_id = format!(
                                    "complete-{}-{}-{}",
                                    pressure_timestamp_ms(),
                                    std::process::id(),
                                    STREAM_SEQUENCE.fetch_add(1, Ordering::Relaxed),
                                );
                                runtime
                                    .run_search_stream_command_with_writer(
                                        cx,
                                        &request.query,
                                        limit,
                                        &stream_id,
                                        &mut output,
                                        Some((
                                            &mut session.reader.resources,
                                            SearchExecutionFlags {
                                                include_snippets: true,
                                                persist_explain_session: false,
                                            },
                                        )),
                                    )
                                    .await?;
                                return Ok(None);
                            }
                            session
                                .reader
                                .runtime
                                .execute_search_serve_request(
                                    cx,
                                    request,
                                    &mut session.reader.resources,
                                    cache,
                                    cache_enabled,
                                )
                                .await
                                .map(Some)
                        },
                    ))
                    .await;
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

    fn complete_request_delivery_budget(&self, request: &SearchServeRequest) -> Duration {
        Self::search_daemon_delivery_budget(
            request
                .quality_timeout_ms
                .unwrap_or(self.config.search.quality_timeout_ms),
            request.rerank.unwrap_or(self.config.search.rerank),
            request
                .rerank_timeout_ms
                .unwrap_or(self.config.search.rerank_timeout_ms),
        )
    }

    pub(super) fn complete_generation_socket_path(&self, root: &Path) -> SearchResult<PathBuf> {
        let root = fs::canonicalize(root)?;
        let path = self.cli_input.daemon_socket.as_ref().map_or_else(
            || root.join(FSFS_DAEMON_SOCKET_FILE),
            |path| {
                if path.is_absolute() {
                    path.clone()
                } else {
                    root.join(path)
                }
            },
        );
        // A named socket may live in the store root or, as a legacy
        // --daemon-socket may, in any existing directory outside the store.
        // Never inside the store's own subdirectories: they hold the sealed
        // bundles and reader pins. The .sock name keeps its lock
        // (`<name>.lock`, created next to it) off any file the caller named,
        // and no directory is created for it.
        let parent = path.parent().ok_or_else(|| {
            complete_cli_error("daemon_socket", "socket must have a parent directory")
        })?;
        // Name the refused path and the root it relates to: the shared helper
        // leaves the value empty, which hid both.
        let refusal = |reason: String| SearchError::InvalidConfig {
            field: "complete_generation.daemon_socket".to_owned(),
            value: path.display().to_string(),
            reason,
        };
        let parent = match fs::canonicalize(parent) {
            Ok(parent) => parent,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Err(refusal(format!(
                    "the socket's directory {} does not exist; create it first",
                    parent.display()
                )));
            }
            Err(error) => return Err(error.into()),
        };
        if (parent != root && parent.starts_with(&root))
            || path.extension().and_then(|extension| extension.to_str()) != Some("sock")
        {
            return Err(refusal(format!(
                "complete-generation sockets must end in .sock and live in the store root {} or outside the store, never inside its generation or reader directories",
                root.display()
            )));
        }
        // Another store's root holds that store's own sockets and locks.
        let pointer = parent.join(crate::generation_store::COMPLETE_GENERATION_POINTER);
        if parent != root && fs::symlink_metadata(pointer).is_ok() {
            return Err(refusal(format!(
                "{} is another complete-generation store's root; name a socket in the store root {} or in a directory that is no store's root",
                parent.display(),
                root.display()
            )));
        }
        let name = path
            .file_name()
            .ok_or_else(|| complete_cli_error("daemon_socket", "socket must have a file name"))?;
        Ok(parent.join(name))
    }
}

/// The execution closure is shared with the ordinary serve implementation;
/// transport errors never trigger an in-process retry or a stale-reader query.
#[cfg(test)]
async fn serve_peer<T, F, Fut>(
    cx: &Cx,
    peer: &mut UnixStream,
    timeout: Duration,
    execute: F,
) -> SearchResult<PeerOutcome>
where
    T: Serialize,
    F: FnOnce(SearchServeRequest, Option<streaming::PhaseWriter>) -> Fut,
    Fut: Future<Output = SearchResult<Option<T>>>,
{
    let bytes = read_request(cx, peer, timeout).await?;
    serve_peer_bytes(cx, peer, timeout, &bytes, |_| timeout, execute).await
}

async fn serve_peer_bytes<T, F, Fut>(
    cx: &Cx,
    peer: &mut UnixStream,
    timeout: Duration,
    bytes: &[u8],
    execution_budget: impl FnOnce(&SearchServeRequest) -> Duration,
    execute: F,
) -> SearchResult<PeerOutcome>
where
    T: Serialize,
    F: FnOnce(SearchServeRequest, Option<streaming::PhaseWriter>) -> Fut,
    Fut: Future<Output = SearchResult<Option<T>>>,
{
    match control::is_shutdown_request(bytes) {
        Ok(true) => {
            // No model/search work and no selection refresh for an explicit
            // control. A failed acknowledgment never claims a successful stop.
            write_response(cx, peer, control::SHUTDOWN_RESPONSE, timeout).await?;
            return Ok(PeerOutcome::Shutdown);
        }
        Ok(false) => {}
        Err(error) => {
            let response = encode_response(&FsfsRuntime::search_serve_error_response(
                "",
                "full",
                error.to_string(),
            ))?;
            write_response(cx, peer, &response, timeout).await?;
            return Ok(PeerOutcome::Search);
        }
    }
    let request = parse_request(bytes);
    let response = match request {
        Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
            "",
            "full",
            error.to_string(),
        ))?,
        Ok((request, stream)) => {
            let search_timeout = execution_budget(&request);
            let query = request.query.clone();
            let mode = request.mode.clone().unwrap_or_else(|| "full".to_owned());
            if stream {
                if let Err(error) = streaming::validate_request(bytes) {
                    encode_response(&FsfsRuntime::search_serve_error_response(
                        query,
                        mode,
                        error.to_string(),
                    ))?
                } else {
                    let output = streaming::PhaseWriter::new(peer)?;
                    let result = run_request(
                        cx,
                        search_timeout,
                        streaming::drive(cx, &output, execute(request, Some(output.clone()))),
                    )
                    .await;
                    match result {
                        Ok(None) => return Ok(PeerOutcome::Search),
                        Ok(Some(_)) => {
                            return Err(complete_cli_error(
                                "daemon_stream",
                                "stream producer returned a buffered response",
                            ));
                        }
                        Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                        Err(error) if output.emitted() => return Err(error),
                        Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
                            query,
                            mode,
                            error.to_string(),
                        ))?,
                    }
                }
            } else {
                retained_search_checkpoint(cx)?;
                match run_request(cx, search_timeout, execute(request, None)).await {
                    Ok(Some(response)) => match encode_response(&response) {
                        Ok(bytes) => bytes,
                        Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
                            query,
                            mode,
                            error.to_string(),
                        ))?,
                    },
                    Ok(None) => {
                        return Err(complete_cli_error(
                            "daemon_response",
                            "buffered request completed without a response",
                        ));
                    }
                    Err(error @ SearchError::Cancelled { .. }) => return Err(error),
                    Err(error) => encode_response(&FsfsRuntime::search_serve_error_response(
                        query,
                        mode,
                        error.to_string(),
                    ))?,
                }
            }
        }
    };
    write_response(cx, peer, &response, timeout).await?;
    Ok(PeerOutcome::Search)
}

fn parse_request(bytes: &[u8]) -> SearchResult<(SearchServeRequest, bool)> {
    let raw = std::str::from_utf8(bytes)
        .map_err(|_| complete_cli_error("daemon_request", "request is not UTF-8"))?
        .trim();
    // The established buffered request type intentionally has no stream field.
    // Inspect the transport selector before serde can ignore that extra field.
    let stream = if raw.starts_with('{') {
        let value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| complete_cli_error("daemon_request", &error.to_string()))?;
        match value.get("stream") {
            None => false,
            Some(serde_json::Value::Bool(stream)) => *stream,
            Some(_) => {
                return Err(complete_cli_error(
                    "daemon_request",
                    "stream must be a boolean",
                ));
            }
        }
    } else {
        false
    };
    FsfsRuntime::parse_search_serve_request(raw).map(|request| (request, stream))
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
                )
                .into()));
            }
            if let Poll::Ready(result) = request.as_mut().poll(task) {
                return Poll::Ready(retained_search_checkpoint(cx).and(result).map(Some));
            }
            if tick.as_mut().poll(task).is_ready() {
                Poll::Ready(Ok(None))
            } else {
                Poll::Pending
            }
        })
        .await?;
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
            return Err(
                io::Error::new(ErrorKind::TimedOut, "daemon request deadline exceeded").into(),
            );
        }
        let remaining = (FSFS_DAEMON_REQUEST_MAX_BYTES + 1 - bytes.len()).min(chunk.len());
        match peer.read(&mut chunk[..remaining]) {
            Ok(0) if bytes.is_empty() => {
                return Err(
                    io::Error::new(ErrorKind::UnexpectedEof, "empty daemon request").into(),
                );
            }
            Ok(0) => return Ok(bytes),
            Ok(count) => {
                let newline = chunk[..count].iter().position(|byte| *byte == b'\n');
                let end = newline.map_or(count, |offset| offset + 1);
                bytes.extend_from_slice(&chunk[..end]);
                if bytes.len() > FSFS_DAEMON_REQUEST_MAX_BYTES {
                    return Err(complete_cli_error(
                        "daemon_request",
                        "request exceeds 1 MiB limit",
                    ));
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
            return Err(
                io::Error::new(ErrorKind::TimedOut, "daemon response deadline exceeded").into(),
            );
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
            })
            .await;
            assert!(
                matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut)
            );
            assert!(released.load(Ordering::SeqCst));
            retained_search_checkpoint(&cx).expect("one timeout must not cancel the daemon");
            assert_eq!(
                run_request(&cx, Duration::from_secs(1), async { Ok(42) })
                    .await
                    .unwrap(),
                42
            );
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
            })
            .await;
            assert!(
                matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut)
            );
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
            })
            .await;
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
            })
            .await;
            assert!(
                matches!(result, Err(SearchError::InvalidConfig { field, reason, .. })
                if field == "complete_generation.test_admission" && reason == "original refusal")
            );
        });
    }

    #[test]
    fn socket_reports_search_deadline_and_accepts_a_subsequent_request() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(b"{\"query\":\"stall\"}\n");
            let result = serve_peer::<serde_json::Value, _, _>(
                &cx,
                &mut server,
                Duration::from_millis(100),
                |_, _| std::future::pending(),
            )
            .await
            .unwrap();
            assert_eq!(result, PeerOutcome::Search);
            let failure = response(client);
            assert_eq!(failure["ok"], false);
            assert!(failure.to_string().contains("search deadline exceeded"));

            let (client, mut server) = pair(b"{\"query\":\"healthy\"}\n");
            serve_peer(&cx, &mut server, Duration::from_secs(1), |_, _| async {
                Ok(Some(serde_json::json!({"ok": true})))
            })
            .await
            .unwrap();
            assert_eq!(response(client)["ok"], true);
        });
    }

    #[test]
    fn raw_request_delivery_budget_preserves_startup_and_request_overrides() {
        let mut config = crate::FsfsConfig::default();
        config.search.quality_timeout_ms = 12_000;
        config.search.rerank = true;
        config.search.rerank_timeout_ms = 18_000;
        let runtime = FsfsRuntime::new(config);
        let (mut request, _) = parse_request(b"{\"query\":\"budget\"}").unwrap();
        assert_eq!(
            runtime.complete_request_delivery_budget(&request),
            Duration::from_secs(60)
        );
        request.quality_timeout_ms = Some(9_000);
        request.rerank = Some(false);
        assert_eq!(
            runtime.complete_request_delivery_budget(&request),
            Duration::from_secs(39)
        );
        request.rerank = Some(true);
        request.rerank_timeout_ms = Some(1_000);
        assert_eq!(
            runtime.complete_request_delivery_budget(&request),
            Duration::from_secs(40)
        );
    }

    #[test]
    fn raw_search_compute_budget_can_outlast_its_transport_deadline() {
        run_test_with_cx(|cx| async move {
            let runtime = FsfsRuntime::new(crate::FsfsConfig::default());
            let transport_timeout = Duration::from_millis(100);
            let (client, mut server) = pair(b"{\"query\":\"slow search\"}\n");
            let bytes = read_request(&cx, &mut server, transport_timeout)
                .await
                .unwrap();
            serve_peer_bytes(
                &cx,
                &mut server,
                transport_timeout,
                &bytes,
                |request| runtime.complete_request_delivery_budget(request),
                |_, _| async {
                    // Computation is allowed past the short framing deadline;
                    // the same deadline still bounds the final response write.
                    asupersync::time::sleep(cx.now(), Duration::from_millis(200)).await;
                    Ok(Some(serde_json::json!({"ok": true})))
                },
            )
            .await
            .unwrap();
            assert_eq!(response(client)["ok"], true);
        });
    }

    fn pair(request: &[u8]) -> (UnixStream, UnixStream) {
        let (mut client, server) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client.write_all(request).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        server.set_nonblocking(true).unwrap();
        (client, server)
    }

    fn response(client: UnixStream) -> serde_json::Value {
        let mut line = String::new();
        std::io::BufReader::new(client)
            .read_line(&mut line)
            .unwrap();
        serde_json::from_str(&line).unwrap()
    }

    #[test]
    fn socket_owner_excludes_duplicates_and_releases_endpoint_not_lock_inode() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.sock");
        let first = BoundCompleteSocket::bind(path.clone()).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(BoundCompleteSocket::bind(path.clone()).is_err());
        let lock_inode = fs::metadata(path.with_extension("lock")).unwrap().ino();
        drop(first);
        assert!(!path.exists());
        assert_eq!(
            fs::metadata(path.with_extension("lock")).unwrap().ino(),
            lock_inode
        );
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

    /// Starting the store's daemon succeeds once its socket accepts, ends at
    /// once when the started process has already exited, and starts nothing
    /// without a spawner or for a socket path too long to bind.
    #[test]
    fn starting_the_store_daemon_waits_only_for_a_live_process() {
        use std::cell::Cell;
        use std::rc::Rc;

        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let runtime = FsfsRuntime::new(crate::FsfsConfig::default());
            let socket = directory.path().join("fsfs-query.sock");

            // No spawner installed: nothing starts and nothing waits.
            let started = Instant::now();
            assert!(
                !runtime
                    .start_complete_generation_daemon(&cx, directory.path(), &socket)
                    .await
                    .unwrap()
            );
            assert!(started.elapsed() < Duration::from_secs(1));

            // A daemon that binds its socket is ready.
            let calls = Rc::new(Cell::new(0));
            let listener = Rc::new(std::cell::RefCell::new(None));
            {
                let (calls, listener, bind_at) = (calls.clone(), listener.clone(), socket.clone());
                let _guard = TestDaemonSpawnerGuard::install(Box::new(move |_root, socket| {
                    calls.set(calls.get() + 1);
                    assert_eq!(socket, bind_at, "the daemon binds the resolved socket");
                    *listener.borrow_mut() = Some(UnixListener::bind(socket).unwrap());
                    Some(Box::new(|| Ok(false)) as DaemonExited)
                }));
                assert!(
                    runtime
                        .start_complete_generation_daemon(&cx, directory.path(), &socket)
                        .await
                        .unwrap()
                );
            }
            assert_eq!(calls.get(), 1);
            drop(listener.borrow_mut().take());
            fs::remove_file(&socket).unwrap();

            // A process that already exited ends the wait well inside the window.
            {
                let _guard = TestDaemonSpawnerGuard::install(Box::new(|_root, _socket| {
                    Some(Box::new(|| Ok(true)) as DaemonExited)
                }));
                let started = Instant::now();
                assert!(
                    !runtime
                        .start_complete_generation_daemon(&cx, directory.path(), &socket)
                        .await
                        .unwrap()
                );
                assert!(started.elapsed() < Duration::from_millis(500));
            }

            // A socket path too long to bind starts nothing.
            let mut long = directory.path().to_path_buf();
            while long.join("fsfs-query.sock").as_os_str().len() <= daemon_socket_path_capacity() {
                long.push("deep-store-segment");
            }
            let calls = Rc::new(Cell::new(0));
            {
                let calls = calls.clone();
                let _guard = TestDaemonSpawnerGuard::install(Box::new(move |_root, _socket| {
                    calls.set(calls.get() + 1);
                    None
                }));
                assert!(
                    !runtime
                        .start_complete_generation_daemon(&cx, &long, &long.join("fsfs-query.sock"))
                        .await
                        .unwrap()
                );
            }
            assert_eq!(calls.get(), 0);
        });
    }

    /// A socket left by a crashed daemon (or a reboot) has no listener; a new
    /// daemon replaces it instead of refusing to start until someone deletes
    /// it. Live listeners and other files stay untouched (test above).
    #[test]
    fn socket_startup_replaces_only_a_verified_stale_endpoint() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("search.sock");
        drop(UnixListener::bind(&path).unwrap());
        assert!(path.exists(), "the endpoint outlives its listener");
        assert!(control::refuses_connections(&path));
        let owner = BoundCompleteSocket::bind(path.clone()).expect("a stale endpoint is replaced");
        assert!(
            !control::refuses_connections(&path),
            "the new owner listens"
        );
        drop(owner);
        assert!(!path.exists());
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

    /// A named daemon socket may live in the store root or in an existing
    /// directory outside the store, as a legacy --daemon-socket may. Inside the
    /// store's subdirectories (sealed bundles, reader pins), in another store's
    /// root, without the .sock name, or in a missing directory it is refused,
    /// naming the path.
    #[test]
    fn named_socket_lives_in_the_store_root_or_outside_the_store() {
        let store = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(store.path()).unwrap();
        let outside_dir = fs::canonicalize(elsewhere.path()).unwrap();
        let runtime_for = |socket: PathBuf| {
            FsfsRuntime::new(crate::FsfsConfig::default()).with_cli_input(crate::CliInput {
                daemon_socket: Some(socket),
                ..crate::CliInput::default()
            })
        };
        let resolve =
            |socket: PathBuf| runtime_for(socket).complete_generation_socket_path(store.path());
        assert_eq!(
            resolve(root.join("custom.sock")).unwrap(),
            root.join("custom.sock")
        );
        assert_eq!(
            resolve(elsewhere.path().join("daemon.sock")).unwrap(),
            outside_dir.join("daemon.sock")
        );

        let generations = root.join("generations");
        fs::create_dir(&generations).unwrap();
        let other_store = elsewhere.path().join("other-store");
        fs::create_dir(&other_store).unwrap();
        fs::write(
            other_store.join(crate::generation_store::COMPLETE_GENERATION_POINTER),
            b"",
        )
        .unwrap();
        for refused in [
            generations.join("daemon.sock"),
            elsewhere.path().join("daemon.socket"),
            elsewhere.path().join("missing").join("daemon.sock"),
            other_store.join("fsfs-query.sock"),
        ] {
            let error = resolve(refused.clone()).unwrap_err();
            assert!(
                matches!(&error, SearchError::InvalidConfig { field, value, .. }
                    if field == "complete_generation.daemon_socket"
                        && *value == refused.display().to_string()),
                "{error:?}"
            );
        }
        let error = resolve(generations.join("daemon.sock")).unwrap_err();
        assert!(
            error.to_string().contains(&root.display().to_string()),
            "{error}"
        );
        assert!(!generations.join("daemon.lock").exists());
        assert!(
            !elsewhere.path().join("missing").exists(),
            "no directory is created"
        );
    }

    #[test]
    fn store_root_too_deep_for_a_unix_socket_is_refused_before_locking() {
        let store = tempfile::tempdir().unwrap();
        let mut root = fs::canonicalize(store.path()).unwrap();
        while root.join("fsfs.sock").as_os_str().len() <= daemon_socket_path_capacity() {
            root.push("deep-store-segment");
        }
        fs::create_dir_all(&root).unwrap();
        let path = root.join("fsfs.sock");
        let Err(error) = BoundCompleteSocket::bind(path.clone()) else {
            panic!("an overlong socket path must not bind"); // ubs:ignore — cfg(test) assertion.
        };
        assert!(
            matches!(&error, SearchError::InvalidConfig { field, value, reason }
                if field == "complete_generation.daemon_socket"
                    && *value == path.display().to_string()
                    && reason.contains("--daemon-socket")),
            "{error:?}"
        );
        assert!(
            !path.with_extension("lock").exists(),
            "no lock is left behind"
        );
        assert!(!path.exists());
    }

    #[test]
    fn socket_peer_round_trips_one_request_and_does_not_emit_stdio_ready() {
        run_test_with_cx(|cx| async move {
            for raw in [
                b"{\"query\":\"sharedtoken\",\"limit\":3}\n".as_slice(),
                b"{\"query\":\"sharedtoken\",\"limit\":3,\"stream\":false}\n",
            ] {
                let (client, mut server) = pair(raw);
                serve_peer(
                    &cx,
                    &mut server,
                    Duration::from_secs(2),
                    |request, output| async move {
                        assert!(output.is_none());
                        assert_eq!(request.query, "sharedtoken");
                        assert_eq!(request.limit, Some(3));
                        Ok(Some(serde_json::json!({"ok":true,"query":request.query})))
                    },
                )
                .await
                .unwrap();
                let value = response(client);
                assert_eq!(value["ok"], true);
                assert!(value.get("event").is_none());
            }
        });
    }

    #[test]
    fn socket_stream_forwards_producer_records_without_a_buffered_wrapper() {
        run_test_with_cx(|cx| async move {
            let (mut client, mut server) = pair(b"{\"query\":\"x\",\"stream\":true}\n");
            serve_peer::<serde_json::Value, _, _>(
                &cx,
                &mut server,
                Duration::from_secs(2),
                |request, output| async move {
                    assert_eq!(request.query, "x");
                    let mut writer = output.expect("stream writer");
                    writer.write_all(b"{\"phase\":\"initial\"}\n")?;
                    writer.flush()?;
                    writer.write_all(b"{\"phase\":\"refined\"}\n")?;
                    writer.flush()?;
                    Ok(None)
                },
            )
            .await
            .unwrap();
            drop(server);
            let mut bytes = Vec::new();
            client.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes, b"{\"phase\":\"initial\"}\n{\"phase\":\"refined\"}\n");
        });
    }

    #[test]
    fn socket_stream_failure_after_delivery_never_appends_a_buffered_error() {
        run_test_with_cx(|cx| async move {
            let (mut client, mut server) = pair(b"{\"query\":\"x\",\"stream\":true}\n");
            let result = serve_peer::<serde_json::Value, _, _>(
                &cx,
                &mut server,
                Duration::from_secs(2),
                |_, output| async move {
                    let mut writer = output.unwrap();
                    writer.write_all(b"{\"phase\":\"initial\"}\n")?;
                    writer.flush()?;
                    Err(complete_cli_error("test_stream", "after Initial"))
                },
            )
            .await;
            assert!(
                matches!(result, Err(SearchError::InvalidConfig { field, .. })
                if field == "complete_generation.test_stream")
            );
            drop(server);
            let mut bytes = Vec::new();
            client.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes, b"{\"phase\":\"initial\"}\n");
        });
    }

    #[test]
    fn socket_stream_admission_failure_is_reported_before_any_frames() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(b"{\"query\":\"x\",\"stream\":true}\n");
            serve_peer::<serde_json::Value, _, _>(
                &cx,
                &mut server,
                Duration::from_secs(2),
                |_, _| async {
                    Err(complete_cli_error(
                        "test_admission",
                        "bad selected generation",
                    ))
                },
            )
            .await
            .unwrap();
            let error = response(client);
            assert_eq!(error["ok"], false);
            assert!(error.to_string().contains("bad selected generation"));
            assert!(error.get("event").is_none());
        });
    }

    #[test]
    fn socket_bad_input_and_unsupported_stream_options_do_not_execute_search() {
        run_test_with_cx(|cx| async move {
            for raw in [
                b"\xff\n".as_slice(),
                b"{invalid\n",
                b"{\"query\":\"x\",\"stream\":true,\"mode\":\"fast\"}\n",
                b"{\"query\":\"x\",\"stream\":\"true\"}\n",
                b"{\"query\":\"x\",\"stream\":null}\n",
            ] {
                let (client, mut server) = pair(raw);
                serve_peer(&cx, &mut server, Duration::from_secs(2), |_, _| async {
                    panic!("refused request must not execute"); // ubs:ignore — cfg(test) negative control fails if a refused request reaches execution.
                    #[allow(unreachable_code)]
                    Ok(Some(serde_json::Value::Null))
                })
                .await
                .unwrap();
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
            assert!(matches!(
                read_request(&cx, &mut server, Duration::from_secs(60)).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(matches!(
                write_response(&cx, &mut server, b"not visible", Duration::from_secs(60)).await,
                Err(SearchError::Cancelled { .. })
            ));
            let error = client.read(&mut [0_u8; 16]).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::WouldBlock);
            cx.set_cancel_requested(false);
        });
    }

    #[test]
    fn explicit_shutdown_is_acknowledged_without_executing_search() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) = pair(control::SHUTDOWN_REQUEST);
            let outcome = serve_peer(&cx, &mut server, Duration::from_secs(2), |_, _| async {
                panic!("shutdown must not execute search"); // ubs:ignore — cfg(test) assertion.
                #[allow(unreachable_code)]
                Ok(Some(serde_json::Value::Null))
            })
            .await
            .unwrap();
            assert_eq!(outcome, PeerOutcome::Shutdown);
            let acknowledgment = response(client);
            assert_eq!(acknowledgment["ok"], true);
            assert_eq!(acknowledgment["event"], "shutdown");
        });
    }

    #[test]
    fn malformed_control_neither_stops_nor_executes_search() {
        run_test_with_cx(|cx| async move {
            let (client, mut server) =
                pair(b"{\"fsfs_complete_daemon\":\"shutdown\",\"version\":2}\n");
            let outcome = serve_peer(&cx, &mut server, Duration::from_secs(2), |_, _| async {
                panic!("malformed control must not execute search"); // ubs:ignore — cfg(test) assertion.
                #[allow(unreachable_code)]
                Ok(Some(serde_json::Value::Null))
            })
            .await
            .unwrap();
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
                read_request(&cx, &mut server, Duration::ZERO)
                    .await
                    .map(|_| ()),
                write_response(&cx, &mut server, b"response", Duration::ZERO).await,
            ] {
                assert!(
                    matches!(result, Err(SearchError::Io(error)) if error.kind() == ErrorKind::TimedOut)
                );
            }
        });
    }
}

#[cfg(all(test, not(feature = "embedded-models")))]
mod generation_tests {
    use super::*;
    use crate::generation_store::{
        COMPLETE_GENERATION_POINTER, CompleteGenerationStore, GenerationPublication,
    };
    use crate::{CliCommand, CliInput, FsfsConfig, InterfaceMode};
    use asupersync::test_utils::run_test_with_cx;
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
            assert!(matches!(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
                GenerationPublication::Durable(_)
            ));
            let pointer = root.join(COMPLETE_GENERATION_POINTER);
            let first = fs::read(&pointer).unwrap();
            let mut pinned = runtime.open_retained_search(&cx, &root).await.unwrap();
            fs::write(source.join("beta.md"), "sharedtoken beta document").unwrap();
            assert!(matches!(
                runtime
                    .rebuild_retained_generation(&cx, &root)
                    .await
                    .unwrap(),
                GenerationPublication::Durable(_)
            ));
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
                    stream
                        .set_read_timeout(Some(Duration::from_secs(10)))
                        .unwrap();
                    stream
                        .set_write_timeout(Some(Duration::from_secs(10)))
                        .unwrap();
                    stream
                        .write_all(b"{\"query\":\"sharedtoken\",\"limit\":10}\n")
                        .unwrap();
                    let mut line = String::new();
                    std::io::BufReader::new(stream)
                        .read_line(&mut line)
                        .unwrap();
                    serde_json::from_str(&line).unwrap()
                };
                let check = |value: serde_json::Value, count, cached| {
                    assert_eq!(value["ok"], true, "{value}");
                    assert_eq!(value["cached"], cached);
                    let phases = value["payloads"].as_array().unwrap();
                    assert_eq!(
                        phases.last().unwrap()["hits"].as_array().unwrap().len(),
                        count
                    );
                };
                let streamed = || -> Vec<serde_json::Value> {
                    let mut stream = UnixStream::connect(&client_endpoint).unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(10)))
                        .unwrap();
                    stream
                        .set_write_timeout(Some(Duration::from_secs(10)))
                        .unwrap();
                    stream
                        .write_all(b"{\"query\":\"sharedtoken\",\"limit\":10,\"stream\":true}\n")
                        .unwrap();
                    let mut reader = std::io::BufReader::new(stream);
                    let mut frames = Vec::new();
                    loop {
                        assert!(frames.len() < 256, "missing stream terminal");
                        let mut line = String::new();
                        assert_ne!(
                            reader.read_line(&mut line).unwrap(),
                            0,
                            "stream ended without terminal/refusal"
                        );
                        let frame: serde_json::Value = serde_json::from_str(&line).unwrap();
                        let done = frame["event"] == "terminal" || frame["ok"] == false;
                        frames.push(frame);
                        if done {
                            return frames;
                        }
                    }
                };
                let check_stream = |frames: Vec<serde_json::Value>, count| {
                    assert_eq!(frames.first().unwrap()["event"], "started");
                    assert_eq!(frames.last().unwrap()["event"], "terminal");
                    assert_eq!(frames.last().unwrap()["payload"]["status"], "completed");
                    assert_eq!(
                        frames
                            .iter()
                            .filter(|frame| frame["event"] == "result")
                            .count(),
                        count
                    );
                    for pair in frames.windows(2) {
                        assert_eq!(
                            pair[1]["seq"].as_u64().unwrap(),
                            pair[0]["seq"].as_u64().unwrap() + 1
                        );
                    }
                    for frame in frames {
                        let _: crate::stream_protocol::StreamFrame<
                            crate::output_schema::SearchHitPayload,
                        > = serde_json::from_value(frame).unwrap();
                    }
                };
                let caching = std::env::var_os("FSFS_DISABLE_QUERY_CACHE").is_none();
                check(request(), 1, false);
                check(request(), 1, caching);
                check_stream(streamed(), 1);
                let temporary = client_root.join("test-pointer-switch");
                fs::write(&temporary, b"invalid selection").unwrap();
                fs::rename(&temporary, &pointer).unwrap();
                let refused = request();
                assert_eq!(refused["ok"], false);
                assert_eq!(refused["cached"], false);
                assert!(refused["payloads"].as_array().unwrap().is_empty());
                let stream_refused = streamed();
                assert_eq!(stream_refused.len(), 1);
                assert_eq!(stream_refused[0]["ok"], false);
                fs::write(&temporary, successor).unwrap();
                fs::rename(&temporary, &pointer).unwrap();
                check(request(), 2, false);
                check(request(), 2, caching);
                check_stream(streamed(), 2);
            });
            let result = runtime
                .run_mode_with_complete_generations(&cx, InterfaceMode::Cli, None, false)
                .await;
            worker.join().unwrap();
            assert!(
                matches!(result, Err(SearchError::Cancelled { .. })),
                "{result:?}"
            );
            cx.set_cancel_requested(false);
            assert!(!endpoint.exists(), "owning command must release the socket");
            let old = pinned.search(&cx, "sharedtoken", 10).await.unwrap();
            assert_eq!(old.last().unwrap().hits.len(), 1);
            assert!(
                CompleteGenerationStore::open(&cx, &root)
                    .unwrap()
                    .active(&cx)
                    .unwrap()
                    .is_some()
            );
        });
    }
}
