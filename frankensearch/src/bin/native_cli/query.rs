//! Query execution policy shared by buffered search and progressive serving.
//! Deadlines reuse the library's captured timer and drop semantics. Output and
//! activation are deliberately outside the timed future: they cannot be undone.

use std::future::Future;
use std::time::Duration;

use asupersync::time::TimerDriverHandle;
use frankensearch::SearchError;
use frankensearch::native_ann::builder::deadline::NativeSearchDeadline;

use super::*;

pub(super) const MAX_TIMEOUT_MS: u64 = 600_000;

/// A server budget is both the default and a ceiling on request overrides.
/// No configured budget preserves the previous unbounded-query policy.
#[derive(Default)]
pub(super) struct Policy {
    pub(super) maximum_ms: Option<u64>,
    pub(super) timer: Option<TimerDriverHandle>,
}

impl Policy {
    pub(super) fn new(cx: &Cx, maximum_ms: Option<u64>) -> Result<Self> {
        validate_timeout(maximum_ms)?;
        Ok(Self { maximum_ms, timer: cx.timer_driver() })
    }

    pub(super) fn maximum_ms(&self) -> Option<u64> {
        self.maximum_ms
    }

    pub(super) fn effective_ms(&self, requested_ms: Option<u64>) -> Result<Option<u64>> {
        validate_timeout(self.maximum_ms)?;
        validate_timeout(requested_ms)?;
        Ok(match (self.maximum_ms, requested_ms) {
            (Some(maximum), Some(requested)) => Some(maximum.min(requested)),
            (maximum, requested) => maximum.or(requested),
        })
    }

    pub(super) fn start(
        &self,
        cx: &Cx,
        requested_ms: Option<u64>,
    ) -> Result<Option<NativeSearchDeadline>> {
        self.effective_ms(requested_ms)?
            .map(|milliseconds| {
                let duration = Duration::from_millis(milliseconds);
                match &self.timer {
                    Some(timer) => NativeSearchDeadline::with_timer(cx, timer.clone(), duration),
                    None => NativeSearchDeadline::after(cx, duration),
                }
                .map_err(Into::into)
            })
            .transpose()
    }
}

pub(super) fn validate_timeout(milliseconds: Option<u64>) -> Result<()> {
    if milliseconds.is_some_and(|value| value == 0 || value > MAX_TIMEOUT_MS) {
        return Err(bad("query timeout must be between 1 and 600000 milliseconds"));
    }
    Ok(())
}

/// Preserve the native error rather than hiding cancellation inside a wrapper.
fn native_error(error: Box<dyn Error + Send + Sync>) -> SearchError {
    match error.downcast::<SearchError>() {
        Ok(error) => *error,
        Err(source) => SearchError::SubsystemError { subsystem: "native_cli.query", source },
    }
}

pub(super) async fn within<T>(
    cx: &Cx,
    deadline: Option<&NativeSearchDeadline>,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    match deadline {
        Some(deadline) => Ok(deadline.run(cx, async { future.await.map_err(native_error) }).await?),
        None => future.await,
    }
}

/// Stable timeout/cancellation facts accompany the human-readable error. There
/// is no query text, source document or model path in these additional fields.
pub(super) fn failure(error: &(dyn Error + 'static)) -> serde_json::Value {
    let mut fields = serde_json::json!({ "status": "failed", "error": error.to_string() });
    if let Some(error) = error.downcast_ref::<SearchError>() {
        match error {
            SearchError::SearchTimeout { elapsed_ms, budget_ms } => {
                fields["status"] = serde_json::json!("timed_out");
                fields["code"] = serde_json::json!("search_timeout");
                fields["elapsed_ms"] = serde_json::json!(elapsed_ms);
                fields["budget_ms"] = serde_json::json!(budget_ms);
            }
            SearchError::Cancelled { .. } => {
                fields["status"] = serde_json::json!("cancelled");
                fields["code"] = serde_json::json!("cancelled");
            }
            _ => {}
        }
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use asupersync::time::VirtualClock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    pub(crate) fn policy(cx: &Cx, milliseconds: u64) -> (Arc<VirtualClock>, Policy) {
        let clock = Arc::new(VirtualClock::new());
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let mut policy = Policy::new(cx, Some(milliseconds)).unwrap();
        policy.timer = Some(timer);
        (clock, policy)
    }

    #[test]
    fn timeout_policy_cannot_be_disabled_or_widened_by_a_request() {
        let cx = Cx::for_testing();
        let (_, policy) = policy(&cx, 100);
        assert_eq!(policy.effective_ms(None).unwrap(), Some(100));
        assert_eq!(policy.effective_ms(Some(1000)).unwrap(), Some(100));
        assert_eq!(policy.effective_ms(Some(10)).unwrap(), Some(10));
        assert!(policy.effective_ms(Some(0)).is_err());
        assert!(policy.effective_ms(Some(MAX_TIMEOUT_MS + 1)).is_err());
        let unbounded = Policy::new(&cx, None).unwrap();
        assert!(unbounded.start(&cx, None).unwrap().is_none());
        assert!(unbounded.start(&cx, Some(10)).is_err(), "no hidden timer fallback");
    }

    #[test]
    fn deadline_spans_multiple_calls_and_rejects_late_ready_work() {
        let cx = Cx::for_testing();
        let (clock, policy) = policy(&cx, 10);
        let deadline = policy.start(&cx, None).unwrap().unwrap();
        let mut first = Box::pin(within(&cx, Some(&deadline), async {
            clock.advance(7_000_000);
            Ok(7)
        }));
        assert!(matches!(first.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(7))));
        drop(first);
        let mut second = Box::pin(within(&cx, Some(&deadline), async {
            clock.advance(4_000_000);
            Ok(11)
        }));
        let Poll::Ready(Err(error)) = second.as_mut().poll(&mut Context::from_waker(Waker::noop())) else {
            panic!("late synchronous work must be refused"); // ubs:ignore — test assertion.
        };
        let facts = failure(error.as_ref());
        assert_eq!(facts["status"], "timed_out");
        assert_eq!(facts["budget_ms"], 10);
        assert_eq!(facts["elapsed_ms"], 11);
        assert!(!cx.is_cancel_requested());
    }

    #[test]
    fn expiry_and_drop_release_pending_work_but_cancellation_keeps_its_type() {
        struct DropCount<'a>(&'a AtomicUsize);
        impl Drop for DropCount<'_> {
            fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
        }
        let cx = Cx::for_testing();
        let drops = AtomicUsize::new(0);
        for cancel in [false, true] {
            let (clock, policy) = policy(&cx, 10);
            let deadline = policy.start(&cx, None).unwrap().unwrap();
            let mut work = Box::pin(within(&cx, Some(&deadline), async {
                let _drop = DropCount(&drops);
                std::future::pending::<Result<()>>().await
            }));
            assert!(work.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            clock.advance(10_000_000);
            cx.set_cancel_requested(cancel);
            let Poll::Ready(Err(error)) = work.as_mut().poll(&mut Context::from_waker(Waker::noop())) else {
                panic!("expired pending work must finish"); // ubs:ignore — test assertion.
            };
            let facts = failure(error.as_ref());
            assert_eq!(facts["status"], if cancel { "cancelled" } else { "timed_out" });
            assert!(policy.timer.as_ref().unwrap().is_empty());
            drop(work);
            cx.set_cancel_requested(false);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        let (_, policy) = policy(&cx, 10);
        let deadline = policy.start(&cx, None).unwrap().unwrap();
        let mut work = Box::pin(within(&cx, Some(&deadline), async {
            let _drop = DropCount(&drops);
            std::future::pending::<Result<()>>().await
        }));
        assert!(work.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        drop(work);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        assert!(policy.timer.as_ref().unwrap().is_empty());
    }
}
