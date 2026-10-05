//! Query execution policy shared by buffered search and progressive serving.
//! Deadlines reuse the library's captured timer and drop semantics. Output and
//! activation are deliberately outside the timed future: they cannot be undone.

use std::future::Future;
use std::time::Duration;

use asupersync::runtime::blocking_pool::BlockingPoolHandle;
use asupersync::time::TimerDriverHandle;
use frankensearch::SearchError;
use frankensearch::native_ann::builder::deadline::NativeSearchDeadline;
use frankensearch::{Reranker, native_ann::NativeSearchPhase};

use super::{
    Arc, Cx, Error, Mode, NativeBuiltHybridIndex, Path, Result, SCHEMA, bad, filter, search,
    validate_query,
};

pub const MAX_TIMEOUT_MS: u64 = 600_000;

/// A server budget is both the default and a ceiling on request overrides.
/// No configured budget preserves the previous unbounded-query policy.
#[derive(Default)]
pub struct Policy {
    pub(super) maximum_ms: Option<u64>,
    pub(super) timer: Option<TimerDriverHandle>,
    pub(super) rerank: Option<Rerank>,
}

impl Policy {
    pub(super) fn new(cx: &Cx, maximum_ms: Option<u64>) -> Result<Self> {
        validate_timeout(maximum_ms)?;
        Ok(Self {
            maximum_ms,
            timer: cx.timer_driver(),
            ..Self::default()
        })
    }

    pub(super) fn maximum_ms(&self) -> Option<u64> {
        self.maximum_ms
    }

    pub(super) fn reranker_for(&self, mode: Mode) -> Option<&Rerank> {
        if mode == Mode::Full {
            self.rerank.as_ref()
        } else {
            None
        }
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
                self.timer
                    .as_ref()
                    .map_or_else(
                        || NativeSearchDeadline::after(cx, duration),
                        |timer| NativeSearchDeadline::with_timer(cx, timer.clone(), duration),
                    )
                    .map_err(Into::into)
            })
            .transpose()
    }
}

/// One explicitly loaded model reused by every full-mode request. Sources come
/// only from the query's retained hybrid cohort, never from mutable file paths.
pub struct Rerank {
    pub(super) model: Arc<dyn Reranker>,
    pub(super) window: usize,
}

impl Rerank {
    #[cfg(any(feature = "native", feature = "rerank", test))]
    pub(super) fn new(model: Arc<dyn Reranker>, window: usize) -> Result<Self> {
        if window == 0 || window > 1_000 {
            return Err(bad("rerank window must be between 1 and 1000"));
        }
        Ok(Self { model, window })
    }

    pub(super) fn load(
        cx: &Cx,
        directory: &Path,
        window: usize,
        pool: Option<BlockingPoolHandle>,
    ) -> Result<Self> {
        cx.checkpoint().map_err(|_| SearchError::Cancelled {
            phase: "native_cli.rerank.load".to_owned(),
            reason: "reranker loading cancelled".to_owned(),
        })?;
        #[cfg(any(feature = "native", feature = "rerank"))]
        {
            let pool =
                pool.ok_or_else(|| bad("native reranking requires the caller's blocking pool"))?;
            let model = frankensearch::NativeReranker::load(directory)?.with_blocking_pool(pool);
            cx.checkpoint().map_err(|_| SearchError::Cancelled {
                phase: "native_cli.rerank.loaded".to_owned(),
                reason: "reranker loading cancelled".to_owned(),
            })?;
            Self::new(Arc::new(model), window)
        }
        #[cfg(not(any(feature = "native", feature = "rerank")))]
        {
            let _ = (directory, window, pool);
            Err(bad(
                "native reranking is not compiled in; build with --features hybrid,rerank",
            ))
        }
    }

    pub(super) fn annotate(&self, payload: &mut serde_json::Value) {
        payload["reranker"] = serde_json::json!(self.model.id());
        payload["rerank_window"] = serde_json::json!(self.window);
    }
}

/// Buffered full-mode reranking is strict: every requested stage must succeed.
/// The phase label comes from the phase actually delivered, so an empty pool is
/// not described as reranked when the native engine correctly skipped inference.
#[allow(clippy::too_many_arguments)]
pub async fn buffered(
    index: &NativeBuiltHybridIndex,
    cx: &Cx,
    text: &str,
    mode: Mode,
    limit: usize,
    filters: filter::Filters<'_>,
    policy: &Policy,
) -> Result<serde_json::Value> {
    let deadline = policy.start(cx, None)?;
    within(cx, deadline.as_ref(), async {
        let Some(rerank) = policy.reranker_for(mode) else {
            return search(index, cx, text, mode, limit, filters).await;
        };
        validate_query(text)?;
        let scoped = filter::Query::prepare(index, cx, filters)?;
        let mut stream = scoped.progressive_with_reranker(
            cx,
            text,
            limit,
            rerank.model.as_ref(),
            rerank.window,
        )?;
        let mut final_page = None;
        while let Some(phase) = stream.next_phase().await? {
            let (phase, results, evaluated) = match phase {
                NativeSearchPhase::Initial { results, .. } => ("initial", results, 0),
                NativeSearchPhase::Refined { results, .. } => ("refined", results, 0),
                NativeSearchPhase::Reranked {
                    results, evaluated, ..
                } => ("reranked", results, evaluated),
                NativeSearchPhase::RefinementFailed { error, .. }
                | NativeSearchPhase::RerankFailed { error, .. } => return Err(error.into()),
            };
            final_page = Some(serde_json::json!({
                "schema": SCHEMA, "event": "results", "ok": true,
                "generation": index.vectors().fast().index().owner_witness().generation,
                "phase": phase, "results": results, "evaluated": evaluated,
                "rerank_applied": phase == "reranked",
            }));
        }
        let mut payload = final_page.ok_or_else(|| bad("native search returned no phase"))?;
        scoped.annotate(&mut payload);
        rerank.annotate(&mut payload);
        Ok(payload)
    })
    .await
}

pub fn validate_timeout(milliseconds: Option<u64>) -> Result<()> {
    if milliseconds.is_some_and(|value| value == 0 || value > MAX_TIMEOUT_MS) {
        return Err(bad(
            "query timeout must be between 1 and 600000 milliseconds",
        ));
    }
    Ok(())
}

/// Preserve the native error rather than hiding cancellation inside a wrapper.
fn native_error(error: Box<dyn Error + Send + Sync>) -> SearchError {
    match error.downcast::<SearchError>() {
        Ok(error) => *error,
        Err(source) => SearchError::SubsystemError {
            subsystem: "native_cli.query",
            source,
        },
    }
}

pub async fn within<T>(
    cx: &Cx,
    deadline: Option<&NativeSearchDeadline>,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    match deadline {
        Some(deadline) => Ok(deadline
            .run(cx, async { future.await.map_err(native_error) })
            .await?),
        None => future.await,
    }
}

/// Stable timeout/cancellation facts accompany the human-readable error. There
/// is no query text, source document or model path in these additional fields.
pub fn failure(error: &(dyn Error + 'static)) -> serde_json::Value {
    let mut fields = serde_json::json!({ "status": "failed", "error": error.to_string() });
    if let Some(error) = error.downcast_ref::<SearchError>() {
        match error {
            SearchError::SearchTimeout {
                elapsed_ms,
                budget_ms,
            } => {
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

    pub fn policy(cx: &Cx, milliseconds: u64) -> (Arc<VirtualClock>, Policy) {
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
        assert!(
            unbounded.start(&cx, Some(10)).is_err(),
            "no hidden timer fallback"
        );
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
        assert!(matches!(
            first.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(7))
        ));
        drop(first);
        let mut second = Box::pin(within(&cx, Some(&deadline), async {
            clock.advance(4_000_000);
            Ok(11)
        }));
        let Poll::Ready(Err(error)) = second
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        else {
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
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
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
            assert!(
                work.as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            clock.advance(10_000_000);
            cx.set_cancel_requested(cancel);
            let Poll::Ready(Err(error)) =
                work.as_mut().poll(&mut Context::from_waker(Waker::noop()))
            else {
                panic!("expired pending work must finish"); // ubs:ignore — test assertion.
            };
            let facts = failure(error.as_ref());
            assert_eq!(
                facts["status"],
                if cancel { "cancelled" } else { "timed_out" }
            );
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
        assert!(
            work.as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(work);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        assert!(policy.timer.as_ref().unwrap().is_empty());
    }
}
