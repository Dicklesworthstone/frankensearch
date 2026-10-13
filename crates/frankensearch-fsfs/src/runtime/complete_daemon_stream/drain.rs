//! Cooperative socket draining. Byte progress and interrupted system calls
//! both consume a bounded turn; a blocked peer does not request a busy retry.

use std::io::{self, ErrorKind, Write};

const MAX_DRAIN_BYTES: usize = 64 * 1024;
const MAX_DRAIN_WRITES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DrainState {
    Drained,
    Blocked,
    Yielded,
}

/// Send a bounded prefix, retaining the exact offset for the next owned poll.
/// `Yielded` requires a scheduler wake rather than waiting for the I/O timer:
/// a writable peer should not pay one timer interval per 64 KiB. In contrast,
/// `Blocked` must use the existing I/O tick to avoid spinning on backpressure.
pub(super) fn drain_pending(
    writer: &mut impl Write,
    pending: &[u8],
    offset: &mut usize,
    emitted: &mut bool,
) -> io::Result<DrainState> {
    let initial_offset = *offset;
    for _ in 0..MAX_DRAIN_WRITES {
        if *offset == pending.len() {
            return Ok(DrainState::Drained);
        }
        let budget = MAX_DRAIN_BYTES - (*offset - initial_offset);
        if budget == 0 {
            return Ok(DrainState::Yielded);
        }
        let remaining = &pending[*offset..];
        match writer.write(&remaining[..remaining.len().min(budget)]) {
            Ok(0) => return Err(io::Error::new(ErrorKind::WriteZero, "stream peer closed")),
            Ok(count) => {
                *emitted = true;
                *offset += count;
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                return Ok(DrainState::Blocked);
            }
            // Interrupted writes still consume a turn. Retrying forever here
            // would prevent the owner from checking cancellation or deadlines.
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(if *offset == pending.len() {
        DrainState::Drained
    } else {
        DrainState::Yielded
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sink {
        bytes: Vec<u8>,
        calls: usize,
        max_write: usize,
        error: Option<ErrorKind>,
    }

    impl Sink {
        fn new(max_write: usize) -> Self {
            Self {
                bytes: Vec::new(),
                calls: 0,
                max_write,
                error: None,
            }
        }
    }

    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if let Some(kind) = self.error {
                return Err(io::Error::new(kind, "scripted socket outcome"));
            }
            let count = bytes.len().min(self.max_write);
            self.bytes.extend_from_slice(&bytes[..count]);
            Ok(count)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn large_writable_response_yields_without_losing_or_replaying_bytes() {
        let bytes = (0_u8..=255)
            .cycle()
            .take(3 * MAX_DRAIN_BYTES + 17)
            .collect::<Vec<_>>();
        let mut sink = Sink::new(usize::MAX);
        let mut offset = 0;
        let mut emitted = false;
        for turn in 1..=3 {
            assert_eq!(
                drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
                DrainState::Yielded
            );
            assert_eq!(offset, turn * MAX_DRAIN_BYTES);
            assert_eq!(sink.calls, turn);
            assert!(emitted);
        }
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Drained
        );
        assert_eq!(offset, bytes.len());
        assert_eq!(sink.bytes, bytes);
        let calls = sink.calls;
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Drained
        );
        assert_eq!(sink.calls, calls);
    }

    #[test]
    fn short_writes_and_interruptions_cannot_monopolize_one_turn() {
        let bytes = vec![b'x'; MAX_DRAIN_WRITES + 1];
        let mut sink = Sink::new(1);
        let mut offset = 0;
        let mut emitted = false;
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Yielded
        );
        assert_eq!(sink.calls, MAX_DRAIN_WRITES);
        assert_eq!(offset, MAX_DRAIN_WRITES);
        assert!(emitted);
        sink.error = Some(ErrorKind::Interrupted);
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Yielded
        );
        assert_eq!(sink.calls, 2 * MAX_DRAIN_WRITES);
        assert_eq!(offset, MAX_DRAIN_WRITES);
        sink.error = None;
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Drained
        );
        assert_eq!(sink.bytes, bytes);
    }

    #[test]
    fn backpressure_waits_and_resumes_at_the_exact_partial_offset() {
        let bytes = vec![b'q'; MAX_DRAIN_WRITES + 2];
        let mut sink = Sink::new(1);
        let mut offset = 0;
        let mut emitted = false;
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Yielded
        );
        sink.error = Some(ErrorKind::WouldBlock);
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Blocked
        );
        assert_eq!(offset, MAX_DRAIN_WRITES);
        assert_eq!(sink.calls, MAX_DRAIN_WRITES + 1);
        sink.error = None;
        assert_eq!(
            drain_pending(&mut sink, &bytes, &mut offset, &mut emitted).unwrap(),
            DrainState::Drained
        );
        assert_eq!(sink.bytes, bytes);
    }

    #[test]
    fn interrupted_empty_and_failed_writes_do_not_claim_emission() {
        for kind in [
            ErrorKind::Interrupted,
            ErrorKind::WouldBlock,
            ErrorKind::BrokenPipe,
        ] {
            let mut sink = Sink::new(1);
            sink.error = Some(kind);
            let mut offset = 0;
            let mut emitted = false;
            let result = drain_pending(&mut sink, b"x", &mut offset, &mut emitted);
            match kind {
                ErrorKind::Interrupted => {
                    assert_eq!(result.unwrap(), DrainState::Yielded);
                    assert_eq!(sink.calls, MAX_DRAIN_WRITES);
                }
                ErrorKind::WouldBlock => assert_eq!(result.unwrap(), DrainState::Blocked),
                _ => assert_eq!(result.unwrap_err().kind(), kind),
            }
            assert_eq!(offset, 0);
            assert!(!emitted);
            assert!(sink.bytes.is_empty());
        }
        let mut sink = Sink::new(0);
        let mut offset = 0;
        let mut emitted = false;
        assert_eq!(
            drain_pending(&mut sink, b"x", &mut offset, &mut emitted)
                .unwrap_err()
                .kind(),
            ErrorKind::WriteZero
        );
        assert_eq!(offset, 0);
        assert!(!emitted);
        assert_eq!(
            drain_pending(&mut sink, b"", &mut offset, &mut emitted).unwrap(),
            DrainState::Drained
        );
        assert!(!emitted);
    }

    #[test]
    fn completed_producer_waits_for_every_byte_across_owned_drain_turns() {
        use super::super::{PhaseWriter, drive};
        use asupersync::test_utils::run_test_with_cx;
        use std::future::Future;
        use std::io::Read;
        use std::os::unix::net::UnixStream;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::task::{Context, Poll, Waker};

        run_test_with_cx(|cx| async move {
            let (mut client, server) = UnixStream::pair().unwrap();
            client.set_nonblocking(true).unwrap();
            let output = PhaseWriter::new(&server).unwrap();
            let mut writer = output.clone();
            let mut expected = vec![b'x'; 8 * MAX_DRAIN_BYTES + 17];
            expected.push(b'\n');
            writer.write_all(&expected).unwrap();
            writer.flush().unwrap();
            assert!(!output.lock().unwrap().pending.is_empty());
            let calls = AtomicUsize::new(0);
            let mut request = Box::pin(drive(&cx, &output, async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(42)
            }));
            let mut task = Context::from_waker(Waker::noop());
            let mut actual = Vec::new();
            let mut chunk = [0; 8192];
            let mut completed = None;
            // Manual owned polls avoid timing assumptions about socket buffer
            // sizes. Every pending turn is followed by real client reads.
            for _ in 0..1024 {
                loop {
                    match client.read(&mut chunk) {
                        Ok(0) => panic!("response socket unexpectedly closed"), // ubs:ignore — cfg(test) protocol negative control.
                        Ok(count) => actual.extend_from_slice(&chunk[..count]),
                        Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                        Err(error) if error.kind() == ErrorKind::Interrupted => {}
                        Err(error) => panic!("response read failed: {error}"), // ubs:ignore — cfg(test) unexpected transport failure.
                    }
                }
                if completed.is_some() {
                    break;
                }
                if let Poll::Ready(result) = request.as_mut().poll(&mut task) {
                    completed = Some(result);
                }
            }
            assert_eq!(completed.unwrap().unwrap(), 42);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(actual, expected);
            assert!(output.lock().unwrap().pending.is_empty());
            assert!(output.emitted());
        });
    }
}
