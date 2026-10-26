//! Cancellation boundaries for work supplied by a caller or provider.

use std::future::{Future, poll_fn};
use std::task::Poll;

use asupersync::Cx;
use asupersync::channel::oneshot;

use crate::{SearchError, SearchResult};

fn checkpoint(cx: &Cx, phase: &'static str) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(cancelled(phase));
    }
    Ok(())
}

fn cancelled(phase: &'static str) -> SearchError {
    SearchError::Cancelled {
        phase: phase.to_owned(),
        reason: "operation cancelled".to_owned(),
    }
}

/// Await work with an independent cancellation wake registration.
///
/// A provider future can remain pending without observing its [`Cx`]. Checking
/// cancellation only around `.await` cannot wake that task. This boundary keeps
/// a cancel-aware receive pending alongside the work, without spawning a task
/// or relying on a timer. Both futures are owned and dropped before returning,
/// including when the caller retains the completed outer future.
///
/// Cancellation wins over a simultaneous successful or failed result, including
/// cancellation requested by the work's destructor. Dropping this boundary also
/// drops the work and unregisters its cancellation wake. Synchronous callbacks
/// and already-running blocking kernels cannot be preempted by this boundary.
///
/// # Errors
///
/// Returns [`SearchError::Cancelled`] with `phase` when the context is cancelled
/// or its checkpoint fails. Otherwise preserves the work's result unchanged.
pub async fn with_cancellation<T>(
    cx: &Cx,
    phase: &'static str,
    work: impl Future<Output = SearchResult<T>>,
) -> SearchResult<T> {
    checkpoint(cx, phase)?;
    let (sender, mut receiver) = oneshot::channel::<()>();
    let outcome = {
        let mut cancellation = std::pin::pin!(receiver.recv(cx));
        let mut work = std::pin::pin!(work);
        poll_fn(|task| {
            if let Err(error) = checkpoint(cx, phase) {
                return Poll::Ready(Err(error));
            }
            // The sender remains alive and never sends. A ready receive means
            // cancellation, never successful work or a closed producer channel.
            if cancellation.as_mut().poll(task).is_ready() {
                return Poll::Ready(Err(cancelled(phase)));
            }
            let outcome = work.as_mut().poll(task);
            match checkpoint(cx, phase) {
                Ok(()) => outcome,
                Err(error) => Poll::Ready(Err(error)),
            }
        })
        .await
    };
    drop(receiver);
    drop(sender);
    checkpoint(cx, phase)?;
    outcome
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Wake, Waker};

    use super::*;

    const PHASE: &str = "test.provider";

    #[derive(Default)]
    struct CountingWake(AtomicUsize);

    impl Wake for CountingWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counting_waker() -> (Arc<CountingWake>, Waker) {
        let counter = Arc::new(CountingWake::default());
        let waker = Waker::from(Arc::clone(&counter));
        (counter, waker)
    }

    #[derive(Default)]
    struct WorkState {
        polls: AtomicUsize,
        drops: AtomicUsize,
        ready: AtomicBool,
    }

    /// This provider deliberately never clones or wakes its task's Waker.
    struct Work<'a> {
        cx: &'a Cx,
        state: &'a WorkState,
        outcome: Option<SearchResult<u32>>,
        cancel_on_poll: bool,
        cancel_on_drop: bool,
    }

    impl<'a> Work<'a> {
        fn new(cx: &'a Cx, state: &'a WorkState, outcome: SearchResult<u32>) -> Self {
            Self {
                cx,
                state,
                outcome: Some(outcome),
                cancel_on_poll: false,
                cancel_on_drop: false,
            }
        }
    }

    impl Future for Work<'_> {
        type Output = SearchResult<u32>;

        fn poll(mut self: Pin<&mut Self>, _task: &mut Context<'_>) -> Poll<Self::Output> {
            self.state.polls.fetch_add(1, Ordering::SeqCst);
            if self.cancel_on_poll {
                self.cx.cancel_fast(asupersync::CancelKind::User);
            }
            if self.state.ready.load(Ordering::SeqCst) {
                Poll::Ready(self.outcome.take().expect("provider completes once"))
            } else {
                Poll::Pending
            }
        }
    }

    impl Drop for Work<'_> {
        fn drop(&mut self) {
            self.state.drops.fetch_add(1, Ordering::SeqCst);
            if self.cancel_on_drop {
                self.cx.cancel_fast(asupersync::CancelKind::User);
            }
        }
    }

    fn provider_error() -> SearchError {
        SearchError::ModelNotFound {
            name: "provider-specific-model".to_owned(),
        }
    }

    fn assert_cancelled(result: Poll<SearchResult<u32>>) {
        assert!(
            matches!(result, Poll::Ready(Err(SearchError::Cancelled { phase, .. })) if phase == PHASE)
        );
    }

    #[test]
    fn pending_provider_is_woken_by_external_cancellation_and_dropped_before_ready() {
        let cx = Cx::for_testing();
        let state = WorkState::default();
        let (counter, waker) = counting_waker();
        let baseline = Arc::strong_count(&counter);
        let mut task = Context::from_waker(&waker);
        let mut future = Box::pin(with_cancellation(
            &cx,
            PHASE,
            Work::new(&cx, &state, Ok(42)),
        ));

        assert!(future.as_mut().poll(&mut task).is_pending());
        assert_eq!(state.polls.load(Ordering::SeqCst), 1);
        assert_eq!(counter.0.load(Ordering::SeqCst), 0);
        assert!(Arc::strong_count(&counter) > baseline);
        std::thread::scope(|scope| {
            scope
                .spawn(|| cx.cancel_fast(asupersync::CancelKind::User))
                .join()
                .expect("cancellation thread");
        });
        // Assert the wake before polling again: a manual repoll alone would
        // hide the defect in a boundary that only checks cancellation.
        assert!(counter.0.load(Ordering::SeqCst) > 0);
        assert_cancelled(future.as_mut().poll(&mut task));
        assert_eq!(state.polls.load(Ordering::SeqCst), 1);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
        assert_eq!(Arc::strong_count(&counter), baseline);
        drop(future);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn pending_provider_tracks_the_latest_waker_without_accumulating_registrations() {
        let cx = Cx::for_testing();
        let state = WorkState::default();
        let (first, first_waker) = counting_waker();
        let (latest, latest_waker) = counting_waker();
        let baseline = Arc::strong_count(&first);
        let mut future = Box::pin(with_cancellation(
            &cx,
            PHASE,
            Work::new(&cx, &state, Ok(42)),
        ));

        let mut task = Context::from_waker(&first_waker);
        assert!(future.as_mut().poll(&mut task).is_pending());
        let registered = Arc::strong_count(&first);
        assert!(registered > baseline);
        for _ in 0..3 {
            assert!(future.as_mut().poll(&mut task).is_pending());
            assert_eq!(Arc::strong_count(&first), registered);
        }
        let mut task = Context::from_waker(&latest_waker);
        assert!(future.as_mut().poll(&mut task).is_pending());
        assert_eq!(Arc::strong_count(&first), baseline);
        assert_eq!(Arc::strong_count(&latest), registered);
        cx.cancel_fast(asupersync::CancelKind::User);
        assert_eq!(first.0.load(Ordering::SeqCst), 0);
        assert!(latest.0.load(Ordering::SeqCst) > 0);
        assert_cancelled(future.as_mut().poll(&mut task));
        assert_eq!(Arc::strong_count(&latest), baseline);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn completion_preserves_success_and_typed_errors_and_releases_waiters_before_ready() {
        for outcome in [Ok(42), Err(provider_error())] {
            let succeeds = outcome.is_ok();
            let cx = Cx::for_testing();
            let state = WorkState::default();
            let (counter, waker) = counting_waker();
            let baseline = Arc::strong_count(&counter);
            let mut task = Context::from_waker(&waker);
            let mut future = Box::pin(with_cancellation(
                &cx,
                PHASE,
                Work::new(&cx, &state, outcome),
            ));

            assert!(future.as_mut().poll(&mut task).is_pending());
            state.ready.store(true, Ordering::SeqCst);
            let result = future.as_mut().poll(&mut task);
            if succeeds {
                assert!(matches!(result, Poll::Ready(Ok(42))));
            } else {
                assert!(matches!(
                    result,
                    Poll::Ready(Err(SearchError::ModelNotFound { name }))
                        if name == "provider-specific-model"
                ));
            }
            // Keep the completed outer future alive while checking ownership.
            assert_eq!(state.drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&counter), baseline);
            cx.cancel_fast(asupersync::CancelKind::User);
            assert_eq!(counter.0.load(Ordering::SeqCst), 0);
            drop(future);
            assert_eq!(state.drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn abandoning_work_drops_the_provider_and_releases_its_cancellation_waiter() {
        for polled in [false, true] {
            for cancel_on_drop in [false, true] {
                let cx = Cx::for_testing();
                let state = WorkState::default();
                let (counter, waker) = counting_waker();
                let baseline = Arc::strong_count(&counter);
                let mut work = Work::new(&cx, &state, Ok(42));
                work.cancel_on_drop = cancel_on_drop;
                let mut future = Box::pin(with_cancellation(&cx, PHASE, work));
                if polled {
                    assert!(
                        future
                            .as_mut()
                            .poll(&mut Context::from_waker(&waker))
                            .is_pending()
                    );
                    assert!(Arc::strong_count(&counter) > baseline);
                }
                drop(future);
                assert_eq!(state.drops.load(Ordering::SeqCst), 1);
                assert_eq!(Arc::strong_count(&counter), baseline);
                let wake_count = counter.0.load(Ordering::SeqCst);
                cx.cancel_fast(asupersync::CancelKind::User);
                assert_eq!(counter.0.load(Ordering::SeqCst), wake_count);
            }
        }
    }

    #[test]
    fn precancelled_context_never_polls_the_provider() {
        let cx = Cx::for_testing();
        cx.cancel_fast(asupersync::CancelKind::User);
        let state = WorkState::default();
        state.ready.store(true, Ordering::SeqCst);
        let (counter, waker) = counting_waker();
        let baseline = Arc::strong_count(&counter);
        let mut future = Box::pin(with_cancellation(
            &cx,
            PHASE,
            Work::new(&cx, &state, Ok(42)),
        ));

        assert_cancelled(future.as_mut().poll(&mut Context::from_waker(&waker)));
        assert_eq!(state.polls.load(Ordering::SeqCst), 0);
        assert_eq!(state.drops.load(Ordering::SeqCst), 1);
        assert_eq!(counter.0.load(Ordering::SeqCst), 0);
        assert_eq!(Arc::strong_count(&counter), baseline);
    }

    #[test]
    fn cancellation_during_provider_poll_wins_over_pending_success_and_error() {
        for completion in 0..3 {
            let cx = Cx::for_testing();
            let state = WorkState::default();
            state.ready.store(completion != 0, Ordering::SeqCst);
            let outcome = if completion == 2 {
                Err(provider_error())
            } else {
                Ok(42)
            };
            let mut work = Work::new(&cx, &state, outcome);
            work.cancel_on_poll = true;
            let (counter, waker) = counting_waker();
            let baseline = Arc::strong_count(&counter);
            let mut future = Box::pin(with_cancellation(&cx, PHASE, work));

            assert_cancelled(future.as_mut().poll(&mut Context::from_waker(&waker)));
            assert_eq!(state.polls.load(Ordering::SeqCst), 1);
            assert_eq!(state.drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&counter), baseline);
        }
    }

    #[test]
    fn cancellation_from_provider_destructor_wins_over_success_and_error() {
        for outcome in [Ok(42), Err(provider_error())] {
            let cx = Cx::for_testing();
            let state = WorkState::default();
            state.ready.store(true, Ordering::SeqCst);
            let mut work = Work::new(&cx, &state, outcome);
            work.cancel_on_drop = true;
            let (counter, waker) = counting_waker();
            let baseline = Arc::strong_count(&counter);
            let mut future = Box::pin(with_cancellation(&cx, PHASE, work));

            assert_cancelled(future.as_mut().poll(&mut Context::from_waker(&waker)));
            assert_eq!(state.polls.load(Ordering::SeqCst), 1);
            assert_eq!(state.drops.load(Ordering::SeqCst), 1);
            assert_eq!(Arc::strong_count(&counter), baseline);
        }
    }
}
