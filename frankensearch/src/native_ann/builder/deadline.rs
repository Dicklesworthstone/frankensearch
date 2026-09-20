//! End-to-end deadlines using a caller-owned timer, never a hidden runtime.
//!
//! One absolute deadline covers admission, waiting and all retrieval work. It
//! does not restart at a phase boundary. Timing uses the captured driver's clock
//! even when the future is polled without an ambient Cx. The caller must service
//! that driver (normally its runtime does so). No thread, task, fallback clock or
//! detached provider is created here. Synchronous graph/model/hydration work
//! cannot be preempted: late completion is refused after that poll returns.

use std::future::{Future, poll_fn};
use std::task::Poll;
use std::time::Duration;

use asupersync::time::{TimerDriverHandle, TimerHandle};
use asupersync::types::Time;

use super::live::{NativeHybridResults, NativeLiveHybridIndex};
use super::scope::NativeScopedHybridIndex;
use super::{NativeBuiltHybridIndex, checkpoint, invalid};
use crate::native_ann::NativeProgressiveSearch;
use crate::{Cx, ScoredResult, SearchError, SearchResult};

mod progressive;
pub use progressive::NativeDeadlineProgressiveSearch;

/// A single absolute query deadline in an explicitly retained timer domain.
///
/// Cloning shares the same expiry; it does not grant another budget. This bounds
/// waiting on cooperative providers and rejects late synchronous completion, not
/// the time a blocking model or graph call takes to return control. Cancellation
/// observed at a poll boundary or returned by a provider remains cancellation,
/// never a timeout fallback. Expiry does not cancel the caller's Cx, so unrelated
/// queries using that context remain valid.
#[derive(Debug, Clone)]
pub struct NativeSearchDeadline {
    timer: TimerDriverHandle,
    started: Time,
    expires: Time,
    budget_nanos: u64,
}

impl NativeSearchDeadline {
    /// Start a budget using the supplied context's timer capability.
    ///
    /// # Errors
    /// Refuses cancellation, a missing timer capability or arithmetic overflow.
    /// No ambient timer or wall-clock fallback is used. Zero expires immediately.
    pub fn after(cx: &Cx, budget: Duration) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.deadline.configure")?;
        let timer = cx.timer_driver().ok_or_else(|| {
            invalid(
                "deadline.timer",
                "missing",
                "a bounded query requires a caller-owned timer driver",
            )
        })?;
        Self::with_timer(cx, timer, budget)
    }

    /// Start a budget on an explicit caller-owned production or virtual timer.
    ///
    /// The caller must drive this timer while the query waits. This is useful
    /// when the future is polled outside its creating runtime's ambient context.
    ///
    /// # Errors
    /// Refuses cancellation or a duration/deadline that cannot fit nanosecond `Time`.
    pub fn with_timer(cx: &Cx, timer: TimerDriverHandle, budget: Duration) -> SearchResult<Self> {
        checkpoint(cx, "native_ann.deadline.configure")?;
        let budget_nanos = u64::try_from(budget.as_nanos()).map_err(|_| {
            invalid(
                "deadline.budget",
                "overflow",
                "budget must fit u64 nanoseconds",
            )
        })?;
        let started = timer.now();
        let expires = started
            .as_nanos()
            .checked_add(budget_nanos)
            .ok_or_else(|| {
                invalid(
                    "deadline.expiry",
                    "overflow",
                    "absolute deadline must fit Time",
                )
            })?;
        Ok(Self {
            timer,
            started,
            expires: Time::from_nanos(expires),
            budget_nanos,
        })
    }

    /// Absolute expiry in the captured driver's time domain.
    #[must_use]
    pub const fn expires_at(&self) -> Time {
        self.expires
    }

    /// Collect a configured native query under this original absolute deadline.
    ///
    /// Accepts the actual unstarted progressive query, including its fusion,
    /// candidate, beam, scope and retained-source reranker settings. No default
    /// query is reconstructed and no source/reader is replaced. Pass the same
    /// context used to prepare the query. A query from a live snapshot keeps
    /// that snapshot; hold the pin while interpreting returned physical rows.
    ///
    /// This returns a final page rather than streaming intermediate phases.
    /// The budget includes every requested phase, including scoped lexical
    /// widening and reranking. Configuration time after deadline creation also
    /// consumes the budget. Expiry never grants quality or reranking a new one.
    /// A dropped pending call releases the owned query and timer registration.
    /// Synchronous work is not preempted; late completion is refused on return.
    ///
    /// # Errors
    /// Returns timeout or any required phase failure, never an Initial-only
    /// success after failed refinement. Cancellation remains cancellation.
    /// Before expiry, the underlying collector also refuses a started query.
    pub async fn collect(
        &self,
        cx: &Cx,
        query: NativeProgressiveSearch<'_>,
    ) -> SearchResult<Vec<ScoredResult>> {
        self.run(cx, query.collect()).await
    }

    fn expired(&self) -> bool {
        self.timer.now() >= self.expires
    }

    fn timeout(&self) -> SearchError {
        SearchError::SearchTimeout {
            elapsed_ms: self
                .timer
                .now()
                .as_nanos()
                .saturating_sub(self.started.as_nanos())
                / 1_000_000,
            budget_ms: self.budget_nanos / 1_000_000,
        }
    }

    async fn run<T>(
        &self,
        cx: &Cx,
        future: impl Future<Output = SearchResult<T>>,
    ) -> SearchResult<T> {
        let mut future = std::pin::pin!(future);
        let mut registration = Registration {
            driver: self.timer.clone(),
            handle: None,
        };
        poll_fn(|task| {
            if let Err(error) = checkpoint(cx, "native_ann.deadline.before_poll") {
                return Poll::Ready(Err(error));
            }
            if self.expired() {
                return Poll::Ready(Err(self.timeout()));
            }
            let outcome = future.as_mut().poll(task);
            if let Err(error) = checkpoint(cx, "native_ann.deadline.after_poll") {
                return Poll::Ready(Err(error));
            }
            // Provider cancellation must not turn into a timeout, even if the
            // provider also consumed the remainder of the budget in this poll.
            if matches!(&outcome, Poll::Ready(Err(SearchError::Cancelled { .. }))) {
                return outcome;
            }
            // A synchronous provider can finish after expiry in a single poll.
            // Do not let a ready result bypass the end-to-end deadline.
            if self.expired() {
                return Poll::Ready(Err(self.timeout()));
            }
            if outcome.is_ready() {
                return outcome;
            }
            // Replace the registration, including its waker. A fired/clamped
            // timer may be stale, so updating that handle is not sufficient.
            // Early wheel-horizon wakes rearm until the REAL expiry is reached.
            registration.clear();
            registration.handle = Some(self.timer.register(self.expires, task.waker().clone()));
            // Close the clock-advance race between checking time and registering.
            if self.expired() {
                return Poll::Ready(Err(self.timeout()));
            }
            Poll::Pending
        })
        .await
        // registration and the owned pending provider are dropped on every exit.
    }
}

struct Registration {
    driver: TimerDriverHandle,
    handle: Option<TimerHandle>,
}

impl Registration {
    fn clear(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.driver.cancel(&handle);
        }
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.clear();
    }
}

impl NativeBuiltHybridIndex {
    /// Independently refined search under one end-to-end deadline.
    ///
    /// All required retrieval/hydration work must finish in time. This eager API
    /// returns an error, not an unannounced initial-only fallback, on expiry.
    ///
    /// # Errors
    /// Propagates refined-search errors, cancellation and [`SearchError::SearchTimeout`].
    pub async fn search_refined_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<Vec<ScoredResult>> {
        deadline.run(cx, self.search_refined(cx, text, k)).await
    }
}

impl NativeLiveHybridIndex {
    /// Bound snapshot acquisition and the entire refined query by the same expiry.
    ///
    /// The returned page carries its originating snapshot. No selection lock is
    /// held during inference/retrieval and a timed-out query cannot change the
    /// live generation or cancel another query. No retry selects a newer cohort.
    ///
    /// # Errors
    /// Propagates acquisition/search failures, cancellation and [`SearchError::SearchTimeout`].
    pub async fn search_refined_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<NativeHybridResults> {
        deadline.run(cx, self.search_refined(cx, text, k)).await
    }
}

impl NativeScopedHybridIndex<'_> {
    /// Collect scoped refinement with one deadline for all eligible retrieval.
    ///
    /// The original scope is retained through candidate widening, both vector
    /// tiers and hydration. No out-of-scope or newer-generation fallback occurs
    /// when time expires. For tuned or reranked queries, prepare this scope's
    /// progressive query and pass it to [`NativeSearchDeadline::collect`].
    ///
    /// # Errors
    /// Propagates scope/query admission, required-phase errors, timeout and
    /// cancellation. An expired deadline does not bypass constructor admission.
    pub async fn search_refined_before(
        &self,
        cx: &Cx,
        text: &str,
        k: usize,
        deadline: &NativeSearchDeadline,
    ) -> SearchResult<Vec<ScoredResult>> {
        deadline.collect(cx, self.progressive(cx, text, k)?).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Wake, Waker};

    use asupersync::time::VirtualClock;
    use frankensearch_core::generation::EmbeddingIdentityBundleV1;
    use frankensearch_core::traits::{IdentityBoundEmbedding, RerankDocument, RerankScore};
    use frankensearch_fusion::RrfConfig;

    use crate::{Embedder, ModelCategory, Reranker, SearchFuture};

    #[derive(Default)]
    struct Wakes(AtomicUsize);
    impl Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn clock(cx: &Cx, budget: Duration) -> (Arc<VirtualClock>, NativeSearchDeadline) {
        let clock = Arc::new(VirtualClock::new());
        let timer = TimerDriverHandle::with_virtual_clock(Arc::clone(&clock));
        let deadline = NativeSearchDeadline::with_timer(cx, timer, budget).unwrap();
        (clock, deadline)
    }

    fn assert_timeout<T>(outcome: Poll<SearchResult<T>>) {
        assert!(matches!(
            outcome,
            Poll::Ready(Err(SearchError::SearchTimeout { .. }))
        ));
    }

    #[test]
    fn expired_work_is_never_polled_and_success_does_not_register_a_timer() {
        let cx = Cx::for_testing();
        let (_, deadline) = clock(&cx, Duration::ZERO);
        let polls = AtomicUsize::new(0);
        let mut expired = Box::pin(deadline.run(&cx, async {
            polls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));
        assert_timeout(
            expired
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
        );
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert!(deadline.timer.is_empty());
        assert!(!cx.is_cancel_requested());
        let (_, deadline) = clock(&cx, Duration::from_millis(10));
        let mut ready = Box::pin(deadline.run(&cx, std::future::ready(Ok(7_u8))));
        assert!(matches!(
            ready.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(7))
        ));
        assert!(deadline.timer.is_empty());
    }

    #[test]
    fn timer_wakes_the_latest_waiter_and_expiry_removes_registration() {
        let cx = Cx::for_testing();
        let (clock, deadline) = clock(&cx, Duration::from_millis(10));
        let first = Arc::new(Wakes::default());
        let second = Arc::new(Wakes::default());
        let first_waker = Waker::from(Arc::clone(&first));
        let second_waker = Waker::from(Arc::clone(&second));
        let mut query = Box::pin(deadline.run(&cx, std::future::pending::<SearchResult<()>>()));
        assert!(
            query
                .as_mut()
                .poll(&mut Context::from_waker(&first_waker))
                .is_pending()
        );
        assert!(
            query
                .as_mut()
                .poll(&mut Context::from_waker(&second_waker))
                .is_pending()
        );
        assert_eq!(deadline.timer.pending_count(), 1);
        clock.advance(10_000_000);
        let _ = deadline.timer.process_timers();
        assert_eq!(first.0.load(Ordering::SeqCst), 0);
        assert!(second.0.load(Ordering::SeqCst) > 0);
        assert_timeout(query.as_mut().poll(&mut Context::from_waker(&second_waker)));
        // Completed-but-still-allocated futures must not retain a timer.
        assert!(deadline.timer.is_empty());
    }

    #[test]
    fn dropping_a_pending_query_releases_its_provider_and_timer() {
        struct Dropped<'a>(&'a AtomicUsize);
        impl Drop for Dropped<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let cx = Cx::for_testing();
        let (_, deadline) = clock(&cx, Duration::from_millis(10));
        let drops = AtomicUsize::new(0);
        let mut query = Box::pin(deadline.run(&cx, async {
            let _guard = Dropped(&drops);
            std::future::pending::<SearchResult<()>>().await
        }));
        assert!(
            query
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert_eq!(deadline.timer.pending_count(), 1);
        drop(query);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(deadline.timer.is_empty());
    }

    #[test]
    fn late_ready_work_is_refused_but_provider_cancellation_is_not_relabelled() {
        let cx = Cx::for_testing();
        for cancelled in [false, true] {
            let (clock, deadline) = clock(&cx, Duration::from_millis(10));
            let mut query = Box::pin(deadline.run(&cx, async {
                clock.advance(11_000_000);
                if cancelled {
                    Err(SearchError::Cancelled {
                        phase: "test.provider".to_owned(),
                        reason: "cancelled".to_owned(),
                    })
                } else {
                    Ok(())
                }
            }));
            let result = query.as_mut().poll(&mut Context::from_waker(Waker::noop()));
            if cancelled {
                assert!(matches!(
                    result,
                    Poll::Ready(Err(SearchError::Cancelled { .. }))
                ));
            } else {
                assert_timeout(result);
            }
            assert!(deadline.timer.is_empty());
        }
    }

    #[test]
    fn context_cancellation_wins_over_expiry_and_checked_deadlines_reject_overflow() {
        let cx = Cx::for_testing();
        let (_, deadline) = clock(&cx, Duration::ZERO);
        cx.set_cancel_requested(true);
        let mut query = Box::pin(deadline.run(&cx, std::future::ready(Ok(()))));
        assert!(matches!(
            query.as_mut().poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(SearchError::Cancelled { .. }))
        ));
        drop(query);
        cx.set_cancel_requested(false);
        assert!(
            NativeSearchDeadline::with_timer(&cx, deadline.timer.clone(), Duration::MAX).is_err()
        );
        let clock = Arc::new(VirtualClock::starting_at(Time::from_nanos(u64::MAX - 1)));
        let timer = TimerDriverHandle::with_virtual_clock(clock);
        assert!(NativeSearchDeadline::with_timer(&cx, timer, Duration::from_nanos(2)).is_err());
    }

    #[test]
    fn a_context_without_time_cannot_start_an_ambient_timer() {
        // Keep the API's capability type, but supply no runtime timer driver.
        let cx = Cx::for_testing();
        assert!(cx.timer_driver().is_none());
        assert!(
            matches!(NativeSearchDeadline::after(&cx, Duration::from_secs(1)),
            Err(SearchError::InvalidConfig { field, .. }) if field == "native_ann.deadline.timer")
        );
    }

    #[test]
    fn bounded_live_refinement_keeps_snapshot_provenance_and_expired_queries_do_no_work() {
        use super::super::NativeIndexBuilder;
        use super::super::tests::{Provider, Reply, documents, generation};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("fast", 2, Reply::Correct));
            let quality = Arc::new(Provider::new("quality", 3, Reply::Correct));
            let index =
                NativeIndexBuilder::new(directory.path().join("index"), generation(), fast.clone())
                    .unwrap()
                    .with_quality_embedder(quality.clone())
                    .unwrap()
                    .add_documents(documents())
                    .build_hybrid(&cx)
                    .await
                    .unwrap();
            let live = NativeLiveHybridIndex::new(&cx, index).unwrap();
            let (_, expired) = clock(&cx, Duration::ZERO);
            assert!(matches!(
                live.search_refined_before(&cx, "vertical", 1, &expired)
                    .await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
            let (_, deadline) = clock(&cx, Duration::from_secs(1));
            let page = live
                .search_refined_before(&cx, "vertical", 1, &deadline)
                .await
                .unwrap();
            assert_eq!(page.results[0].doc_id, "b");
            assert_eq!(page.snapshot.generation(), generation());
            assert_eq!(
                page.snapshot
                    .index()
                    .vectors()
                    .document("b")
                    .unwrap()
                    .content,
                "vertical"
            );
            assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
            assert!(!cx.is_cancel_requested());
        });
    }

    // Real native/Quill construction with a controlled query-only suspension.
    // The inherited provider is explicitly a hash-control test model.
    struct QueryGate {
        inner: super::super::tests::Provider,
        mode: AtomicUsize,
        calls: AtomicUsize,
        drops: AtomicUsize,
    }

    impl QueryGate {
        fn new(name: &str, dimension: u32) -> Self {
            Self {
                inner: super::super::tests::Provider::new(
                    name,
                    dimension,
                    super::super::tests::Reply::Correct,
                ),
                mode: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
            }
        }
    }

    struct GateDrop<'a>(&'a AtomicUsize);

    impl Drop for GateDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Embedder for QueryGate {
        fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _drop = GateDrop(&self.drops);
                match self.mode.load(Ordering::SeqCst) {
                    1 => std::future::pending().await,
                    2 => Err(invalid(
                        "deadline.test",
                        "failed",
                        "required provider failed",
                    )),
                    3 => Err(SearchError::Cancelled {
                        phase: "deadline.test".to_owned(),
                        reason: "provider cancelled".to_owned(),
                    }),
                    _ => self.inner.embed(cx, text).await,
                }
            })
        }

        fn embed_batch_bound<'a>(
            &'a self,
            cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            self.inner.embed_batch_bound(cx, texts)
        }

        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            self.inner.identity()
        }
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn dimension(&self) -> usize {
            self.inner.dimension()
        }
        fn is_semantic(&self) -> bool {
            false
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    async fn scoped_fixture(
        cx: &Cx,
    ) -> (
        tempfile::TempDir,
        Arc<QueryGate>,
        Arc<QueryGate>,
        NativeBuiltHybridIndex,
    ) {
        use super::super::NativeIndexBuilder;
        use super::super::tests::{documents, generation};

        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(QueryGate::new("fast", 2));
        let quality = Arc::new(QueryGate::new("quality", 3));
        let index =
            NativeIndexBuilder::new(directory.path().join("index"), generation(), fast.clone())
                .unwrap()
                .with_quality_embedder(quality.clone())
                .unwrap()
                .add_documents(documents())
                .build_hybrid(cx)
                .await
                .unwrap();
        (directory, fast, quality, index)
    }

    #[derive(Default)]
    struct ScopedReranker {
        ids: std::sync::Mutex<Vec<String>>,
    }

    impl Reranker for ScopedReranker {
        fn rerank<'a>(
            &'a self,
            _cx: &'a Cx,
            _query: &'a str,
            documents: &'a [RerankDocument],
        ) -> SearchFuture<'a, Vec<RerankScore>> {
            Box::pin(async move {
                *self.ids.lock().unwrap() =
                    documents.iter().map(|doc| doc.doc_id.clone()).collect();
                Ok(documents
                    .iter()
                    .enumerate()
                    .map(|(original_rank, document)| RerankScore {
                        doc_id: document.doc_id.clone(),
                        original_rank,
                        score: if document.doc_id == "b" { 1.0 } else { 0.0 },
                        raw_logit: None,
                    })
                    .collect())
            })
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn id(&self) -> &str {
            "deadline-scoped-reranker"
        }
        fn model_name(&self) -> &str {
            self.id()
        }
    }

    #[test]
    fn bounded_scoped_collection_preserves_tuning_and_excludes_reranker_documents() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = scoped_fixture(&cx).await;
            let scope = index
                .scope(&cx, |doc| Ok(doc.id == "a" || doc.id == "b"))
                .unwrap();
            let config = RrfConfig {
                k: 0.0,
                lexical_weight: 3.0,
                ..RrfConfig::default()
            };
            let query = || {
                scope
                    .progressive(&cx, "horizontal", 2)
                    .unwrap()
                    .with_fusion(config.clone(), 0.4)
                    .unwrap()
                    .with_candidate_multiplier(1)
                    .unwrap()
                    .with_beam_widths(Some(2), Some(2))
                    .unwrap()
            };
            let expected = query().collect().await.unwrap();
            let (_, deadline) = clock(&cx, Duration::from_secs(1));
            let actual = deadline.collect(&cx, query()).await.unwrap();
            assert_eq!(
                serde_json::to_value(&actual).unwrap(),
                serde_json::to_value(&expected).unwrap()
            );
            assert_eq!(actual[0].doc_id, "a");
            assert_eq!(actual[0].score.to_bits(), 4.0_f32.to_bits());
            assert_eq!(actual[1].doc_id, "b");
            assert_eq!(actual[1].score.to_bits(), 0.5_f32.to_bits());
            assert!(actual.iter().all(|hit| {
                hit.fast_score.is_none() && hit.quality_score.is_none() && hit.index.is_none()
            }));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 2);

            let reranker = ScopedReranker::default();
            let query = scope
                .progressive_with_reranker(&cx, "horizontal", 1, &reranker, 2)
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .with_fusion(config, 0.4)
                .unwrap();
            let reranked = deadline.collect(&cx, query).await.unwrap();
            assert_eq!(reranked[0].doc_id, "b");
            assert_eq!(reranked[0].rerank_score, Some(1.0));
            assert_eq!(*reranker.ids.lock().unwrap(), ["a", "b"]);
            assert!(deadline.timer.is_empty());

            let expected = scope.search_refined(&cx, "horizontal", 2).await.unwrap();
            let actual = scope
                .search_refined_before(&cx, "horizontal", 2, &deadline)
                .await
                .unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        });
    }

    #[test]
    fn configuring_a_scoped_query_does_not_grant_a_fresh_deadline() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = scoped_fixture(&cx).await;
            let (time, deadline) = clock(&cx, Duration::from_millis(10));
            time.advance(10_000_000);
            let scope = index.scope(&cx, |doc| Ok(doc.id == "b")).unwrap();
            let query = scope
                .progressive(&cx, "vertical", 1)
                .unwrap()
                .with_candidate_multiplier(1)
                .unwrap()
                .with_fusion(RrfConfig::default(), 0.0)
                .unwrap();
            assert!(matches!(
                deadline.collect(&cx, query).await,
                Err(SearchError::SearchTimeout {
                    elapsed_ms: 10,
                    budget_ms: 10
                })
            ));
            assert!(matches!(
                scope
                    .search_refined_before(&cx, "vertical", 1, &deadline)
                    .await,
                Err(SearchError::SearchTimeout { .. })
            ));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(deadline.timer.is_empty());
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn bounded_collection_does_not_hide_required_quality_failure_at_zero_weight() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = scoped_fixture(&cx).await;
            let scope = index.scope(&cx, |doc| Ok(doc.id == "b")).unwrap();
            let (_, deadline) = clock(&cx, Duration::from_secs(1));
            for mode in [2, 3] {
                quality.mode.store(mode, Ordering::SeqCst);
                let query = scope
                    .progressive(&cx, "vertical", 1)
                    .unwrap()
                    .with_fusion(RrfConfig::default(), 0.0)
                    .unwrap();
                let error = deadline.collect(&cx, query).await.unwrap_err();
                if mode == 2 {
                    assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                        if field == "native_ann.deadline.test"));
                } else {
                    assert!(matches!(error, SearchError::Cancelled { ref phase, .. }
                        if phase == "deadline.test"));
                }
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), 2);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 2);
            assert!(deadline.timer.is_empty());
        });
    }

    #[test]
    fn scoped_collection_timeout_drop_and_cancellation_release_required_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_directory, fast, quality, index) = scoped_fixture(&cx).await;
            let scope = index.scope(&cx, |doc| Ok(doc.id == "b")).unwrap();
            for ending in 0..3 {
                quality.mode.store(1, Ordering::SeqCst);
                let (time, deadline) = clock(&cx, Duration::from_millis(10));
                let query = scope.progressive(&cx, "vertical", 1).unwrap();
                let mut pending = Box::pin(deadline.collect(&cx, query));
                assert!(
                    pending
                        .as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
                assert_eq!(deadline.timer.pending_count(), 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), ending + 1);
                match ending {
                    0 => {
                        time.advance(10_000_000);
                        let _ = deadline.timer.process_timers();
                        assert!(matches!(
                            pending.await,
                            Err(SearchError::SearchTimeout { .. })
                        ));
                    }
                    1 => drop(pending),
                    _ => {
                        cx.set_cancel_requested(true);
                        time.advance(10_000_000);
                        assert!(matches!(pending.await, Err(SearchError::Cancelled { .. })));
                        cx.set_cancel_requested(false);
                    }
                }
                assert_eq!(quality.drops.load(Ordering::SeqCst), ending + 1);
                assert!(deadline.timer.is_empty());
            }
            quality.mode.store(0, Ordering::SeqCst);
            assert_eq!(
                scope.search(&cx, "vertical", 1).await.unwrap()[0].doc_id,
                "b"
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 4);
            assert!(!cx.is_cancel_requested());
        });
    }
}
