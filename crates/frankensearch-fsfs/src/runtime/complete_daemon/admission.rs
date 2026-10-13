//! Bounded multi-peer framing around the daemon's single search owner.
//!
//! A peer that has not finished its request cannot monopolize the listener.
//! Framing futures are polled here, never spawned. Search/model/cache access
//! remains serialized by the parent; only transport intake runs alongside it.

use std::collections::VecDeque;
use std::future::{Future, poll_fn};
use std::os::unix::net::{UnixListener, UnixStream};
use std::pin::{Pin, pin};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::SearchResult;

use super::{IO_POLL_INTERVAL, control, read_request, retained_search_checkpoint};

// At most eight framed/framing peers in addition to the executing request.
// The existing 1 MiB request limit therefore also bounds queued payloads.
const MAX_PEERS: usize = 8;
const ACCEPTS_PER_TURN: usize = 4;

pub(super) struct ReadyPeer {
    peer: UnixStream,
    bytes: Vec<u8>,
    accepted: Instant,
    shutdown: bool,
}

impl ReadyPeer {
    pub(super) const fn is_shutdown(&self) -> bool {
        self.shutdown
    }

    pub(super) fn into_parts(self) -> (UnixStream, Vec<u8>) {
        (self.peer, self.bytes)
    }
}

struct ReadingPeer {
    accepted: Instant,
    future: Pin<Box<dyn Future<Output = SearchResult<ReadyPeer>> + Send>>,
}

pub(super) enum DriveOutcome<T> {
    Finished(T),
    // Only the existing strict control decoder can mint this disposition.
    // The parent must drop the active peer and acknowledge before stopping.
    Shutdown(ReadyPeer),
}

pub(super) struct Intake {
    reading: VecDeque<ReadingPeer>,
    ready: VecDeque<ReadyPeer>,
    timeout: Duration,
    last_activity: Instant,
}

impl Intake {
    pub(super) fn new(timeout: Duration) -> Self {
        Self {
            reading: VecDeque::new(),
            ready: VecDeque::new(),
            timeout,
            last_activity: Instant::now(),
        }
    }

    pub(super) fn completed(&mut self) {
        self.last_activity = Instant::now();
    }

    fn len(&self) -> usize {
        self.reading.len() + self.ready.len()
    }

    /// One bounded scheduler turn: four accepts and one poll of each admitted
    /// framing future. read_request itself yields after bounded read work.
    /// Expiry includes time waiting for the single search owner, so queued
    /// clients cannot retain their payload/socket indefinitely during inference.
    fn pump(
        &mut self,
        cx: &Cx,
        listener: &UnixListener,
        task: &mut Context<'_>,
        now: Instant,
    ) -> SearchResult<bool> {
        retained_search_checkpoint(cx)?;
        let before = self.len();
        let timeout = self.timeout;
        self.reading
            .retain(|peer| now.saturating_duration_since(peer.accepted) < timeout);
        self.ready
            .retain(|peer| now.saturating_duration_since(peer.accepted) < timeout);
        if self.len() != before {
            self.last_activity = now;
            tracing::debug!(
                reason_code = "daemon.intake.expired",
                "expired queued request peers"
            );
        }
        let mut accepted = 0;
        for _ in 0..ACCEPTS_PER_TURN {
            if self.len() == MAX_PEERS {
                break;
            }
            retained_search_checkpoint(cx)?;
            let (mut peer, _) = match listener.accept() {
                Ok(peer) => peer,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            };
            self.last_activity = now;
            accepted += 1;
            if let Err(error) = peer.set_nonblocking(true) {
                tracing::debug!(%error, "complete-generation peer setup failed");
                continue;
            }
            let task_cx = cx.clone();
            self.reading.push_back(ReadingPeer {
                accepted: now,
                future: Box::pin(async move {
                    let bytes = read_request(&task_cx, &mut peer, timeout).await?;
                    // Malformed controls remain ordinary refused requests. Do
                    // not turn a decoder error, discriminator alone, duplicate
                    // field or unknown command into shutdown authorization.
                    let shutdown = control::is_shutdown_request(&bytes).unwrap_or(false);
                    Ok(ReadyPeer {
                        peer,
                        bytes,
                        accepted: now,
                        shutdown,
                    })
                }),
            });
        }
        let count = self.reading.len();
        for _ in 0..count {
            retained_search_checkpoint(cx)?;
            let Some(mut peer) = self.reading.pop_front() else {
                break;
            };
            match peer.future.as_mut().poll(task) {
                Poll::Pending => self.reading.push_back(peer),
                Poll::Ready(Ok(peer)) => {
                    self.last_activity = now;
                    self.ready.push_back(peer);
                }
                Poll::Ready(Err(error)) => {
                    retained_search_checkpoint(cx)?;
                    self.last_activity = now;
                    tracing::debug!(%error, "complete-generation request framing failed");
                }
            }
        }
        retained_search_checkpoint(cx)?;
        // Drain a still-readable accept backlog on another scheduler turn,
        // not recursively here. A full queue waits for capacity/the I/O tick.
        Ok(accepted == ACCEPTS_PER_TURN && self.len() < MAX_PEERS)
    }

    fn take_shutdown(&mut self) -> Option<ReadyPeer> {
        let position = self.ready.iter().position(|peer| peer.shutdown)?;
        self.ready.remove(position)
    }

    /// Wait for a complete request, never for a particular incomplete sender.
    /// Valid shutdown controls take priority; all other ready peers retain
    /// completion order. Admission saturation leaves the kernel backlog alone.
    #[allow(clippy::future_not_send)] // Framing futures stay on this owning task.
    pub(super) async fn next(
        &mut self,
        cx: &Cx,
        listener: &UnixListener,
        idle_timeout: Option<Duration>,
    ) -> SearchResult<Option<ReadyPeer>> {
        loop {
            let mut tick = pin!(asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL));
            let result = poll_fn(|task| {
                let now = Instant::now();
                let again = match self.pump(cx, listener, task, now) {
                    Ok(again) => again,
                    Err(error) => return Poll::Ready(Err(error)),
                };
                if let Some(peer) = self.take_shutdown().or_else(|| self.ready.pop_front()) {
                    return Poll::Ready(Ok(Some(Some(peer))));
                }
                if self.len() == 0
                    && idle_timeout.is_some_and(|limit| {
                        now.saturating_duration_since(self.last_activity) >= limit
                    })
                {
                    return Poll::Ready(Ok(Some(None)));
                }
                if again {
                    task.waker().wake_by_ref();
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

    /// Continue bounded intake while one caller-owned search is pending.
    /// Only an exactly decoded shutdown preempts it. The work future is dropped
    /// before returning Shutdown; no shared Cx cancellation, replay, reply or
    /// generation mutation occurs here. Acknowledgment remains the parent's job.
    /// Synchronous work already running inside poll cannot be preempted.
    #[allow(clippy::future_not_send)] // Never spawn or require Send from the search owner.
    pub(super) async fn drive<T>(
        &mut self,
        cx: &Cx,
        listener: &UnixListener,
        work: impl Future<Output = T>,
    ) -> SearchResult<DriveOutcome<T>> {
        let mut work = pin!(work);
        loop {
            let mut tick = pin!(asupersync::time::sleep(cx.now(), IO_POLL_INTERVAL));
            let result = poll_fn(|task| {
                let again = match self.pump(cx, listener, task, Instant::now()) {
                    Ok(again) => again,
                    Err(error) => return Poll::Ready(Err(error)),
                };
                if let Some(peer) = self.take_shutdown() {
                    return Poll::Ready(Ok(Some(DriveOutcome::Shutdown(peer))));
                }
                let response = work.as_mut().poll(task);
                if let Err(error) = retained_search_checkpoint(cx) {
                    return Poll::Ready(Err(error));
                }
                if let Poll::Ready(result) = response {
                    return Poll::Ready(Ok(Some(DriveOutcome::Finished(result))));
                }
                if again {
                    task.waker().wake_by_ref();
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
}

/// One real scheduler yield, not a timer delay or another spawned task.
/// Used by the shared request reader after its per-turn byte/syscall budget.
pub(super) async fn yield_turn(cx: &Cx) -> SearchResult<()> {
    retained_search_checkpoint(cx)?;
    let mut yielded = false;
    poll_fn(|task| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            task.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;
    retained_search_checkpoint(cx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::SearchError;
    use std::cell::Cell;
    use std::io::{Read, Write};
    use std::net::Shutdown;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Waker;

    fn listener() -> (tempfile::TempDir, UnixListener) {
        let directory = tempfile::tempdir().unwrap();
        let listener = UnixListener::bind(directory.path().join("intake.sock")).unwrap();
        listener.set_nonblocking(true).unwrap();
        (directory, listener)
    }

    fn connect(listener: &UnixListener, bytes: &[u8]) -> UnixStream {
        let mut client = UnixStream::connect(listener.local_addr().unwrap().as_pathname().unwrap())
            .unwrap();
        client.write_all(bytes).unwrap();
        client.set_nonblocking(true).unwrap();
        client
    }

    fn pump(intake: &mut Intake, cx: &Cx, listener: &UnixListener, now: Instant) {
        intake
            .pump(cx, listener, &mut Context::from_waker(Waker::noop()), now)
            .unwrap();
    }

    #[test]
    fn stalled_first_sender_does_not_block_later_complete_requests() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            let _stalled = connect(&listener, b"{\"query\":");
            let mut healthy = connect(&listener, b"{\"query\":\"ready\"}\n");
            healthy.shutdown(Shutdown::Write).unwrap();
            pump(&mut intake, &cx, &listener, Instant::now());
            assert_eq!(intake.reading.len(), 1);
            assert_eq!(intake.ready.len(), 1);
            let (mut peer, bytes) = intake
                .next(&cx, &listener, None)
                .await
                .unwrap()
                .unwrap()
                .into_parts();
            assert_eq!(bytes, b"{\"query\":\"ready\"}\n");
            peer.write_all(b"one response\n").unwrap();
            let mut reply = [0; 13];
            healthy.read_exact(&mut reply).unwrap();
            assert_eq!(&reply, b"one response\n");
        });
    }

    #[test]
    fn bounded_intake_leaves_excess_peers_in_backlog_until_capacity_is_released() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            let _clients = (0..MAX_PEERS + 1)
                .map(|_| connect(&listener, b"ready\n"))
                .collect::<Vec<_>>();
            for _ in 0..3 {
                pump(&mut intake, &cx, &listener, Instant::now());
                assert!(intake.len() <= MAX_PEERS);
            }
            assert_eq!(intake.ready.len(), MAX_PEERS);
            drop(intake.ready.pop_front());
            pump(&mut intake, &cx, &listener, Instant::now());
            assert_eq!(intake.ready.len(), MAX_PEERS);
            assert!(intake.reading.is_empty());
        });
    }

    #[test]
    fn queued_deadlines_release_incomplete_and_complete_peers_without_a_search() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let timeout = Duration::from_secs(1);
            let mut intake = Intake::new(timeout);
            let now = Instant::now();
            let mut silent = connect(&listener, b"");
            let mut complete = connect(&listener, b"ready\n");
            pump(&mut intake, &cx, &listener, now);
            assert_eq!(intake.reading.len(), 1);
            assert_eq!(intake.ready.len(), 1);
            pump(&mut intake, &cx, &listener, now + timeout);
            assert_eq!(intake.len(), 0);
            assert_eq!(silent.read(&mut [0; 1]).unwrap(), 0);
            assert_eq!(complete.read(&mut [0; 1]).unwrap(), 0);
            retained_search_checkpoint(&cx).unwrap();
        });
    }

    #[test]
    fn fragmented_and_eof_framing_use_the_existing_request_contract() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            let mut fragmented = connect(&listener, b"sha");
            pump(&mut intake, &cx, &listener, Instant::now());
            assert!(intake.ready.is_empty());
            fragmented.write_all(b"red\n").unwrap();
            let eof = connect(&listener, b"eof query");
            eof.shutdown(Shutdown::Write).unwrap();
            // The partial sender's framing future may still be waiting on its
            // I/O tick. Await completion, without assuming cross-peer ordering.
            let mut received = Vec::new();
            for _ in 0..2 {
                received.push(
                    intake.next(&cx, &listener, None).await.unwrap().unwrap().bytes,
                );
            }
            received.sort();
            assert_eq!(received, [b"eof query".to_vec(), b"shared\n".to_vec()]);
        });
    }

    struct Released(Arc<AtomicBool>);
    impl Drop for Released {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn shutdown_preempts_pending_work_without_a_model_wakeup_or_shared_cancellation() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            let released = Arc::new(AtomicBool::new(false));
            let owned = Released(Arc::clone(&released));
            let polls = Cell::new(0);
            let mut driving = Box::pin(intake.drive(&cx, &listener, async {
                let _owned = owned;
                poll_fn(|_| {
                    polls.set(polls.get() + 1);
                    Poll::<()>::Pending
                })
                .await
            }));
            let mut task = Context::from_waker(Waker::noop());
            assert!(driving.as_mut().poll(&mut task).is_pending());
            assert_eq!(polls.get(), 1);
            let mut client = connect(&listener, control::SHUTDOWN_REQUEST);
            let Poll::Ready(Ok(DriveOutcome::Shutdown(shutdown))) = driving.as_mut().poll(&mut task)
            else {
                panic!("an exact shutdown must preempt pending work"); // ubs:ignore — cfg(test) assertion.
            };
            drop(driving);
            assert!(released.load(Ordering::SeqCst));
            assert_eq!(polls.get(), 1);
            retained_search_checkpoint(&cx).unwrap();
            // Intake does not claim acknowledgment or write a probe.
            assert_eq!(
                client.read(&mut [0; 1]).unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
            let (mut peer, bytes) = shutdown.into_parts();
            assert_eq!(bytes, control::SHUTDOWN_REQUEST);
            super::super::write_response(
                &cx,
                &mut peer,
                control::SHUTDOWN_RESPONSE,
                Duration::from_secs(1),
            )
            .await
            .unwrap();
            let mut reply = vec![0; control::SHUTDOWN_RESPONSE.len()];
            client.read_exact(&mut reply).unwrap();
            assert_eq!(reply, control::SHUTDOWN_RESPONSE);
        });
    }

    #[test]
    fn only_exact_controls_preempt_and_other_ready_requests_are_retained() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            let _query = connect(&listener, b"shutdown\n");
            let _malformed = connect(
                &listener,
                b"{\"fsfs_complete_daemon\":\"shutdown\",\"version\":2}\n",
            );
            let _duplicate = connect(
                &listener,
                b"{\"fsfs_complete_daemon\":\"shutdown\",\"version\":1,\"version\":1}\n",
            );
            let result = intake.drive(&cx, &listener, async { 42 }).await.unwrap();
            assert!(matches!(result, DriveOutcome::Finished(42)));
            assert_eq!(intake.ready.len(), 3);
            let _shutdown = connect(&listener, control::SHUTDOWN_REQUEST);
            let polled = Cell::new(false);
            let result = intake
                .drive(&cx, &listener, async {
                    polled.set(true);
                    43
                })
                .await
                .unwrap();
            assert!(matches!(result, DriveOutcome::Shutdown(_)));
            assert!(!polled.get());
            assert_eq!(intake.ready.len(), 3);
        });
    }

    #[test]
    fn owner_cancellation_and_dropping_intake_release_owned_peers_and_work() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            let mut client = connect(&listener, b"");
            pump(&mut intake, &cx, &listener, Instant::now());
            let released = Arc::new(AtomicBool::new(false));
            let owned = Released(Arc::clone(&released));
            let polled = Cell::new(false);
            cx.set_cancel_requested(true);
            let result = intake
                .drive(&cx, &listener, async {
                    let _owned = owned;
                    polled.set(true);
                })
                .await;
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert!(!polled.get());
            assert!(released.load(Ordering::SeqCst));
            drop(intake);
            assert_eq!(client.read(&mut [0; 1]).unwrap(), 0);
        });
    }

    #[test]
    fn idle_expiry_and_unpolled_futures_do_not_accept_or_start_work() {
        run_test_with_cx(|cx| async move {
            let (_directory, listener) = listener();
            let mut intake = Intake::new(Duration::from_secs(60));
            assert!(
                intake.next(&cx, &listener, Some(Duration::ZERO)).await.unwrap().is_none()
            );
            let _client = connect(&listener, b"ready\n");
            let polled = Cell::new(false);
            drop(intake.drive(&cx, &listener, async { polled.set(true) }));
            drop(intake.next(&cx, &listener, None));
            assert!(!polled.get());
            assert_eq!(intake.len(), 0);
            pump(&mut intake, &cx, &listener, Instant::now());
            assert_eq!(intake.ready.len(), 1);
        });
    }

    #[test]
    fn shared_reader_yields_after_a_bounded_prefix_without_changing_framing() {
        struct Counted<'a> {
            input: std::io::Cursor<Vec<u8>>,
            consumed: &'a Cell<usize>,
        }
        impl Read for Counted<'_> {
            fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                let count = self.input.read(bytes)?;
                self.consumed.set(self.consumed.get() + count);
                Ok(count)
            }
        }
        run_test_with_cx(|cx| async move {
            let mut bytes = vec![b'x'; 64 * 1024];
            bytes.push(b'\n');
            let consumed = Cell::new(0);
            let mut input = Counted {
                input: std::io::Cursor::new(bytes.clone()),
                consumed: &consumed,
            };
            let mut reading = Box::pin(read_request(&cx, &mut input, Duration::from_secs(60)));
            let mut task = Context::from_waker(Waker::noop());
            assert!(reading.as_mut().poll(&mut task).is_pending());
            assert_eq!(consumed.get(), 64 * 1024);
            let Poll::Ready(Ok(actual)) = reading.as_mut().poll(&mut task) else {
                panic!("the second read turn must complete the original frame"); // ubs:ignore — cfg(test) assertion.
            };
            assert_eq!(actual, bytes);
            assert_eq!(consumed.get(), bytes.len());
        });
    }

    #[test]
    fn interrupted_reads_yield_and_observe_cancellation_before_retrying() {
        struct Interrupted<'a>(&'a Cell<usize>);
        impl Read for Interrupted<'_> {
            fn read(&mut self, _bytes: &mut [u8]) -> std::io::Result<usize> {
                self.0.set(self.0.get() + 1);
                Err(std::io::Error::from(std::io::ErrorKind::Interrupted))
            }
        }
        run_test_with_cx(|cx| async move {
            let calls = Cell::new(0);
            let mut input = Interrupted(&calls);
            let mut reading = Box::pin(read_request(&cx, &mut input, Duration::from_secs(60)));
            let mut task = Context::from_waker(Waker::noop());
            assert!(reading.as_mut().poll(&mut task).is_pending());
            assert_eq!(calls.get(), 32);
            cx.set_cancel_requested(true);
            let result = reading.as_mut().poll(&mut task);
            cx.set_cancel_requested(false);
            assert!(matches!(result, Poll::Ready(Err(SearchError::Cancelled { .. }))));
            assert_eq!(calls.get(), 32);
        });
    }
}
