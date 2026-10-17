//! Pull-driven search over one retained database cohort.
//!
//! A pull executes exactly one independently identified tier. The next tier
//! does not start until the consumer asks for it; no producer, queue, task or
//! database transaction survives independently of the caller's future.

use std::future::{Future, poll_fn};
use std::task::Poll;
use std::time::{Duration, Instant};

use asupersync::Cx;
use asupersync::types::Time;
use frankensearch_core::{SearchError, SearchResult};

use crate::connection::map_storage_error;
use crate::{PublishedSearchCohort, PublishedSearchTextQuery, PublishedSearchTierHits};

const MAX_TIER_TIMEOUT_MS: u64 = 86_400_000;

/// Progress is explicit: interrupted work must not masquerade as completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishedSearchStreamState {
    /// The zero-based index of the next requested tier.
    Ready { next_tier: usize },
    /// Every requested tier was delivered, including any empty tier results.
    Complete,
    /// This tier returned an error. No later tier will be started.
    Failed { tier: usize },
    /// A pull was polled, then dropped or unwound before returning a result.
    /// The stream is terminal; it never silently retries that inference.
    Interrupted { tier: usize },
}

/// A lazy, source-coherent sequence of independently ranked tier results.
///
/// Results borrow the cohort, not the stream, and remain usable after this
/// stream is dropped. Already delivered tiers survive a later error. They are
/// NOT silently blended or relabelled as refined results: ranking/fusion is
/// still the consumer's explicit policy. Optional per-tier deadlines cover
/// admission, canonicalization, inference and exact retrieval in each pull.
///
/// There is no speculative work. Dropping an unpolled pull changes nothing;
/// dropping a pending pull drops its provider future and terminally interrupts
/// this stream, without cancelling the shared caller `Cx`. A provider that
/// internally starts independent work retains responsibility for its teardown.
/// This cannot preempt synchronous provider or exact-scoring work.
#[must_use]
pub struct PublishedSearchStream<'cohort, 'query> {
    cohort: &'cohort PublishedSearchCohort,
    text: &'query str,
    requests: Vec<PublishedSearchTextQuery<'query>>,
    timeout_ms: Vec<Option<u64>>,
    state: PublishedSearchStreamState,
}

impl std::fmt::Debug for PublishedSearchStream<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Do not expose query/source payloads or invoke provider callbacks.
        f.debug_struct("PublishedSearchStream")
            .field("tiers", &self.requests.len())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl PublishedSearchCohort {
    /// Prepare progressive text retrieval without starting inference.
    ///
    /// Names, duplicates and the complete request count are checked before any
    /// pull can run. The stream copies only bounded request descriptors; text,
    /// providers, canonicalizers and the immutable cohort remain caller-owned.
    /// Each pull uses the existing bound single-tier search path, including
    /// producer admission for empty queries/corpora and zero limits. Requests
    /// may select a subset or reverse the capture order. Quality-only retrieval
    /// searches its own entire membership, not the fast tier's candidate set.
    ///
    /// No database access or runtime/task creation occurs. Empty requests are
    /// immediately complete but still check cancellation at construction.
    pub fn search_text_stream<'cohort, 'query>(
        &'cohort self,
        cx: &Cx,
        text: &'query str,
        requests: &[PublishedSearchTextQuery<'query>],
    ) -> SearchResult<PublishedSearchStream<'cohort, 'query>> {
        checkpoint(cx)?;
        // The cohort already has at most sixteen unique tiers. A valid subset
        // cannot be larger; do not allocate based on an unbounded request list.
        if requests.len() > self.snapshots().len() {
            return Err(invalid("requests exceed the captured tier count"));
        }
        for (index, request) in requests.iter().enumerate() {
            checkpoint(cx)?;
            if self.get(request.embedder_id).is_none() {
                return Err(invalid("requested tier is not part of the captured cohort"));
            }
            if requests[..index]
                .iter()
                .any(|previous| previous.embedder_id == request.embedder_id)
            {
                return Err(invalid("duplicate tier requests are ambiguous"));
            }
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(requests.len())
            .map_err(map_storage_error)?;
        owned.extend_from_slice(requests);
        let mut timeout_ms = Vec::new();
        timeout_ms
            .try_reserve_exact(requests.len())
            .map_err(map_storage_error)?;
        timeout_ms.resize(requests.len(), None);
        checkpoint(cx)?;
        Ok(PublishedSearchStream {
            cohort: self,
            text,
            requests: owned,
            timeout_ms,
            state: if requests.is_empty() {
                PublishedSearchStreamState::Complete
            } else {
                PublishedSearchStreamState::Ready { next_tier: 0 }
            },
        })
    }
}

impl<'cohort> PublishedSearchStream<'cohort, '_> {
    /// Bound one selected tier before the stream's first pull.
    ///
    /// The budget starts when that tier is requested, not at construction or
    /// during consumer backpressure between tiers. Each tier may have a distinct
    /// budget of 1 ms through 24 hours; omitted budgets retain caller-owned
    /// deadlines. Configuration never calls providers or starts timers.
    ///
    /// Expiry returns `SearchTimeout` and terminally fails the stream. Pending
    /// work is dropped; already delivered results are unchanged. Synchronous
    /// callbacks/scoring cannot be preempted, but a late successful poll is
    /// refused. Cancellation dominates; observed identity/provider failures are
    /// retained rather than disguised as retryable timeouts. No shared `Cx` is
    /// cancelled.
    pub fn with_tier_timeout(
        mut self,
        cx: &Cx,
        embedder_id: &str,
        timeout_ms: u64,
    ) -> SearchResult<Self> {
        checkpoint(cx)?;
        if self.state != (PublishedSearchStreamState::Ready { next_tier: 0 }) {
            return Err(invalid("deadlines must be configured before the first pull"));
        }
        if !(1..=MAX_TIER_TIMEOUT_MS).contains(&timeout_ms) {
            return Err(invalid("tier timeout must be between 1 ms and 24 hours"));
        }
        let index = self
            .requests
            .iter()
            .position(|request| request.embedder_id == embedder_id)
            .ok_or_else(|| invalid("timeout names a tier not selected by this stream"))?;
        self.timeout_ms[index] = Some(timeout_ms);
        checkpoint(cx)?;
        Ok(self)
    }

    #[must_use]
    pub const fn state(&self) -> PublishedSearchStreamState {
        self.state
    }

    /// Deliver one complete tier before starting any later tier.
    ///
    /// `Some(Err(_))` is terminal, as is abandoning a polled pull. Subsequent
    /// pulls return `None`; inspect `state` to distinguish successful exhaustion
    /// from failure/interruption. Cancellation before/during a live pull wins
    /// over an ordinary result. Once terminal, the stream stays exhausted even
    /// when polled with a different context.
    pub async fn next(
        &mut self,
        cx: &Cx,
    ) -> Option<SearchResult<PublishedSearchTierHits<'cohort>>> {
        let PublishedSearchStreamState::Ready { next_tier } = self.state else {
            return None;
        };
        // Commit the terminal interruption marker BEFORE any callback/await.
        // Only returning a complete outcome below may advance this cursor.
        self.state = PublishedSearchStreamState::Interrupted { tier: next_tier };
        let cohort = self.cohort;
        let text = self.text;
        let request = self.requests[next_tier];
        let timeout_ms = self.timeout_ms[next_tier];
        let work = async move {
            checkpoint(cx)?;
            let result = cohort
                .search_text(cx, text, std::slice::from_ref(&request))
                .await;
            checkpoint(cx)?;
            let mut results = result?;
            if results.len() != 1 {
                return Err(invalid(
                    "a single tier pull returned an invalid result count",
                ));
            }
            results
                .pop()
                .ok_or_else(|| invalid("the requested tier result is missing"))
        };
        let result = match timeout_ms {
            Some(timeout_ms) => run_bounded(cx, timeout_ms, work).await,
            None => work.await,
        };
        self.state = if result.is_err() {
            PublishedSearchStreamState::Failed { tier: next_tier }
        } else if next_tier + 1 == self.requests.len() {
            PublishedSearchStreamState::Complete
        } else {
            PublishedSearchStreamState::Ready {
                next_tier: next_tier + 1,
            }
        };
        Some(result)
    }
}

#[allow(clippy::future_not_send)] // Caller-owned local futures are supported.
pub(crate) async fn run_bounded<T>(
    cx: &Cx,
    budget_ms: u64,
    work: impl Future<Output = SearchResult<T>>,
) -> SearchResult<T> {
    checkpoint(cx)?;
    let started = cx.now();
    let wall_started = Instant::now();
    run_bounded_with_elapsed(cx, started, budget_ms, work, || {
        // Virtual runtime time drives deterministic scheduling. Wall time also
        // catches a blocking callback that did not advance the runtime clock.
        Duration::from_nanos(cx.now().duration_since(started)).max(wall_started.elapsed())
    })
    .await
}

/// The same polling path accepts an elapsed clock for deterministic boundary
/// tests; production always uses the caller runtime and monotonic wall clock.
#[allow(clippy::future_not_send)] // The guard must not add a Send requirement.
async fn run_bounded_with_elapsed<T>(
    cx: &Cx,
    started: Time,
    budget_ms: u64,
    work: impl Future<Output = SearchResult<T>>,
    elapsed: impl Fn() -> Duration,
) -> SearchResult<T> {
    checkpoint(cx)?;
    let budget = Duration::from_millis(budget_ms);
    let expired = || {
        let duration = elapsed();
        (duration >= budget).then(|| timeout_error(budget_ms, duration))
    };
    let mut work = Box::pin(work);
    let guarded = poll_fn(|context| {
        if let Err(error) = checkpoint(cx) {
            return Poll::Ready(Err(error));
        }
        if let Some(error) = expired() {
            return Poll::Ready(Err(error));
        }
        let outcome = work.as_mut().poll(context);
        if let Err(error) = checkpoint(cx) {
            return Poll::Ready(Err(error));
        }
        // An actual failure, especially a producer-identity mismatch, must
        // not be changed into a weaker timeout by an elapsed clock.
        if matches!(&outcome, Poll::Ready(Err(_))) {
            return outcome;
        }
        if let Some(error) = expired() {
            return Poll::Ready(Err(error));
        }
        outcome
    });
    // The runtime timer wakes a parked provider even if it never wakes itself.
    // Poll checks above also reject synchronous completion after the deadline.
    let outcome = asupersync::time::timeout(started, budget, guarded).await;
    drop(work);
    checkpoint(cx)?;
    match outcome {
        Ok(result) => result,
        Err(_) => Err(timeout_error(budget_ms, elapsed().max(budget))),
    }
}

fn timeout_error(budget_ms: u64, elapsed: Duration) -> SearchError {
    SearchError::SearchTimeout {
        elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        budget_ms,
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "storage.publication_progressive".to_owned(),
        reason: error.to_string(),
    })
}

fn invalid(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "publication_progressive.request".to_owned(),
        value: "rejected".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arc_with_non_send_sync)]

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::task::{Context, Waker};

    use frankensearch_core::generation::{EmbeddingArtifactIdentityV1, EmbeddingSpaceKindV1};
    use frankensearch_core::traits::{ModelCategory, SearchFuture};
    use frankensearch_core::{
        Canonicalizer, Embedder, EmbeddingIdentityBundleV1, IdentityBoundEmbedding,
    };

    use super::*;
    use crate::{
        ClaimOutcome, ContentHasher, DocumentRecord, JobQueueConfig, PersistentJobQueue,
        PublishedSearchReadLimits, PublishedSearchTierRequest, Storage,
    };

    fn identity(name: &str) -> EmbeddingIdentityBundleV1 {
        let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, 2);
        identity.space.kind = EmbeddingSpaceKindV1::Semantic;
        identity.space.hash_control = None;
        identity.space.artifact_manifest_fingerprint = "a".repeat(64);
        identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
            role: "weights".to_owned(),
            sha256: "b".repeat(64),
            size: 1,
        }];
        identity.producer.space_fingerprint = identity.space.fingerprint();
        identity.validate().unwrap();
        identity
    }

    fn fixture(cx: &Cx) -> (Arc<Storage>, PublishedSearchCohort) {
        let storage = Arc::new(Storage::open_in_memory().unwrap());
        let queue = PersistentJobQueue::new(Arc::clone(&storage), JobQueueConfig::default());
        for (id, tier, values) in [
            ("fast-only", "fast", vec![1.0, 0.0]),
            ("quality-only", "quality", vec![1.0, 0.0]),
            ("shared", "fast", vec![0.0, 1.0]),
            ("shared", "quality", vec![0.0, 1.0]),
        ] {
            let body = format!("retained source for {id}");
            let hash = ContentHasher::hash(&body);
            storage
                .upsert_document(&DocumentRecord::new(id, &body, hash, body.chars().count(), 1, 1))
                .unwrap();
            storage.set_document_content(id, &body).unwrap();
            queue.enqueue(id, tier, &hash, 0).unwrap();
            let claim = queue.claim_batch("publisher", 1).unwrap().pop().unwrap();
            let response = IdentityBoundEmbedding {
                values,
                identity: identity(tier),
            };
            assert_eq!(
                storage
                    .publish_claimed_embedding(&claim, &hash, &response.identity, &response)
                    .unwrap(),
                ClaimOutcome::Applied(())
            );
        }
        let cohort = storage
            .load_published_search_cohort(
                cx,
                &[
                    PublishedSearchTierRequest {
                        embedder_id: "fast",
                        identity: &identity("fast"),
                    },
                    PublishedSearchTierRequest {
                        embedder_id: "quality",
                        identity: &identity("quality"),
                    },
                ],
                PublishedSearchReadLimits::default(),
            )
            .unwrap();
        (storage, cohort)
    }

    struct Canonical;
    impl Canonicalizer for Canonical {
        fn canonicalize(&self, _: &str) -> String {
            panic!("not document input")
        }
        fn canonicalize_query(&self, text: &str) -> String {
            text.trim().to_owned()
        }
    }

    #[derive(Clone, Copy)]
    enum Reply {
        Ready,
        Error,
        Foreign,
        Pending,
        Cancel,
    }

    struct Probe {
        name: &'static str,
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
        drops: Arc<AtomicUsize>,
        reply: Reply,
    }
    impl Probe {
        fn new(name: &'static str, reply: Reply) -> Self {
            Self {
                name,
                identity: identity(name),
                calls: AtomicUsize::new(0),
                drops: Arc::new(AtomicUsize::new(0)),
                reply,
            }
        }
        fn request(&self, limit: usize) -> PublishedSearchTextQuery<'_> {
            PublishedSearchTextQuery {
                embedder_id: self.name,
                embedder: self,
                canonicalizer: &Canonical,
                limit,
            }
        }
    }
    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    impl Embedder for Probe {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async { Err(invalid("raw inference must not run")) })
        }
        fn embed_bound<'a>(
            &'a self,
            cx: &'a Cx,
            _: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _dropped = Dropped(Arc::clone(&self.drops));
                match self.reply {
                    Reply::Pending => std::future::pending::<()>().await,
                    Reply::Error => return Err(invalid("provider failed")),
                    Reply::Cancel => {
                        cx.cancel_with(asupersync::CancelKind::User, Some("provider cancelled"));
                        return Err(invalid("provider failed after cancellation"));
                    }
                    Reply::Ready | Reply::Foreign => {}
                }
                Ok(IdentityBoundEmbedding {
                    values: vec![1.0, 0.0],
                    identity: if matches!(self.reply, Reply::Foreign) {
                        identity("foreign")
                    } else {
                        self.identity.clone()
                    },
                })
            })
        }
        fn dimension(&self) -> usize {
            2
        }
        fn id(&self) -> &'static str {
            self.name
        }
        fn model_name(&self) -> &'static str {
            "progressive-test-probe"
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::StaticEmbedder
        }
    }

    #[test]
    fn pulls_deliver_independent_tiers_lazily_from_retained_sources() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (storage, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Ready);
            let quality = Probe::new("quality", Reply::Ready);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[fast.request(1), quality.request(1)])
                .unwrap();
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            storage.delete_document("fast-only").unwrap();
            storage.delete_document("quality-only").unwrap();
            drop(storage);
            let initial = stream.next(&cx).await.unwrap().unwrap();
            assert_eq!(initial.hits()[0].hit().doc_id.as_str(), "fast-only");
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(
                stream.state(),
                PublishedSearchStreamState::Ready { next_tier: 1 }
            );
            let refined = stream.next(&cx).await.unwrap().unwrap();
            assert_eq!(refined.hits()[0].hit().doc_id.as_str(), "quality-only");
            assert_eq!(stream.state(), PublishedSearchStreamState::Complete);
            assert!(stream.next(&cx).await.is_none());
            drop(stream);
            assert_eq!(
                initial.hits()[0].document().canonical_text(),
                "retained source for fast-only"
            );
            assert_eq!(
                refined.hits()[0].document().canonical_text(),
                "retained source for quality-only"
            );
            assert!(std::ptr::eq(
                refined.snapshot(),
                cohort.get("quality").unwrap()
            ));
        });
    }

    #[test]
    fn reversed_subsets_and_zero_limits_keep_their_own_contracts() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Ready);
            let quality = Probe::new("quality", Reply::Ready);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[quality.request(1), fast.request(0)])
                .unwrap();
            let quality_hits = stream.next(&cx).await.unwrap().unwrap();
            assert_eq!(quality_hits.snapshot().embedder_id(), "quality");
            assert!(stream.next(&cx).await.unwrap().unwrap().hits().is_empty());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            let mut only = cohort
                .search_text_stream(&cx, "query", &[quality.request(1)])
                .unwrap();
            let only_hits = only.next(&cx).await.unwrap().unwrap();
            assert_eq!(only_hits.hits()[0].hit().doc_id.as_str(), "quality-only");
            assert!(only.next(&cx).await.is_none());
        });
    }

    #[test]
    fn failure_is_terminal_and_does_not_invalidate_an_already_delivered_tier() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            for reply in [Reply::Error, Reply::Foreign] {
                let fast = Probe::new("fast", Reply::Ready);
                let quality = Probe::new("quality", reply);
                let mut stream = cohort
                    .search_text_stream(&cx, "query", &[fast.request(1), quality.request(1)])
                    .unwrap();
                let initial = stream.next(&cx).await.unwrap().unwrap();
                assert!(stream.next(&cx).await.unwrap().is_err());
                assert_eq!(
                    stream.state(),
                    PublishedSearchStreamState::Failed { tier: 1 }
                );
                assert!(stream.next(&cx).await.is_none());
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                assert_eq!(initial.hits()[0].hit().doc_id.as_str(), "fast-only");
            }
            let fast = Probe::new("fast", Reply::Error);
            let quality = Probe::new("quality", Reply::Ready);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[fast.request(1), quality.request(1)])
                .unwrap();
            assert!(stream.next(&cx).await.unwrap().is_err());
            assert!(stream.next(&cx).await.is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn dropping_a_pending_pull_is_terminal_without_retry_or_context_cancellation() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Ready);
            let quality = Probe::new("quality", Reply::Pending);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[fast.request(1), quality.request(1)])
                .unwrap();
            drop(stream.next(&cx));
            assert_eq!(
                stream.state(),
                PublishedSearchStreamState::Ready { next_tier: 0 }
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            let initial = stream.next(&cx).await.unwrap().unwrap();
            {
                let mut pull = Box::pin(stream.next(&cx));
                let mut context = Context::from_waker(Waker::noop());
                assert!(matches!(pull.as_mut().poll(&mut context), Poll::Pending));
            }
            assert_eq!(quality.drops.load(Ordering::SeqCst), 1);
            assert_eq!(
                stream.state(),
                PublishedSearchStreamState::Interrupted { tier: 1 }
            );
            assert!(stream.next(&cx).await.is_none());
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            assert!(!cx.is_cancel_requested());
            assert_eq!(
                initial.hits()[0].document().canonical_text(),
                "retained source for fast-only"
            );
        });
    }

    #[test]
    fn invalid_names_and_duplicates_are_refused_before_any_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Ready);
            let unknown = Probe::new("unknown", Reply::Ready);
            for requests in [
                vec![fast.request(1), unknown.request(1)],
                vec![fast.request(1); 2],
                vec![fast.request(1); 17],
            ] {
                assert!(cohort.search_text_stream(&cx, "query", &requests).is_err());
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(unknown.calls.load(Ordering::SeqCst), 0);
            let mut empty = cohort.search_text_stream(&cx, "query", &[]).unwrap();
            assert_eq!(empty.state(), PublishedSearchStreamState::Complete);
            assert!(empty.next(&cx).await.is_none());
            assert!(!format!("{empty:?}").contains("query"));
        });
    }

    #[test]
    fn cancellation_precedes_noops_and_stops_later_tiers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Ready);
            let quality = Probe::new("quality", Reply::Ready);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[fast.request(1), quality.request(0)])
                .unwrap();
            let initial = stream.next(&cx).await.unwrap().unwrap();
            cx.cancel_with(asupersync::CancelKind::User, Some("consumer stopped"));
            assert!(matches!(
                stream.next(&cx).await,
                Some(Err(SearchError::Cancelled { .. }))
            ));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(stream.next(&cx).await.is_none());
            assert!(matches!(
                cohort.search_text_stream(&cx, "", &[]),
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(initial.hits()[0].hit().doc_id.as_str(), "fast-only");
        });
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Cancel);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[fast.request(1)])
                .unwrap();
            assert!(matches!(
                stream.next(&cx).await,
                Some(Err(SearchError::Cancelled { .. }))
            ));
            assert_eq!(
                stream.state(),
                PublishedSearchStreamState::Failed { tier: 0 }
            );
        });
    }

    #[test]
    fn empty_query_and_zero_limit_never_hide_a_wrong_provider() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let mut fast = Probe::new("fast", Reply::Ready);
            fast.identity = identity("foreign");
            for (text, limit) in [("", 1), ("query", 0)] {
                let mut stream = cohort
                    .search_text_stream(&cx, text, &[fast.request(limit)])
                    .unwrap();
                assert!(stream.next(&cx).await.unwrap().is_err());
                assert_eq!(
                    stream.state(),
                    PublishedSearchStreamState::Failed { tier: 0 }
                );
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn tier_deadlines_are_preflight_only_and_cannot_reconfigure_started_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            let fast = Probe::new("fast", Reply::Ready);
            let quality = Probe::new("quality", Reply::Ready);
            for (name, budget) in [
                ("quality", 0),
                ("quality", MAX_TIER_TIMEOUT_MS + 1),
                ("unknown", 100),
            ] {
                let stream = cohort
                    .search_text_stream(&cx, "query", &[fast.request(1), quality.request(1)])
                    .unwrap();
                assert!(stream.with_tier_timeout(&cx, name, budget).is_err());
            }
            let fast_only = cohort
                .search_text_stream(&cx, "query", &[fast.request(1)])
                .unwrap();
            assert!(fast_only.with_tier_timeout(&cx, "quality", 100).is_err());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            let mut stream = cohort
                .search_text_stream(&cx, "query", &[fast.request(1), quality.request(1)])
                .unwrap()
                .with_tier_timeout(&cx, "quality", MAX_TIER_TIMEOUT_MS)
                .unwrap();
            assert_eq!(stream.timeout_ms, [None, Some(MAX_TIER_TIMEOUT_MS)]);
            let initial = stream.next(&cx).await.unwrap().unwrap();
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert!(stream.with_tier_timeout(&cx, "quality", 10).is_err());
            assert_eq!(initial.hits()[0].hit().doc_id.as_str(), "fast-only");
        });
    }

    #[test]
    fn a_configured_tier_keeps_the_normal_bound_response_and_source_contract() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let (_, cohort) = fixture(&cx);
            for reply in [Reply::Ready, Reply::Foreign] {
                let quality = Probe::new("quality", reply);
                let mut stream = cohort
                    .search_text_stream(&cx, "query", &[quality.request(1)])
                    .unwrap()
                    .with_tier_timeout(&cx, "quality", MAX_TIER_TIMEOUT_MS)
                    .unwrap();
                let outcome = stream.next(&cx).await.unwrap();
                if matches!(reply, Reply::Ready) {
                    let hits = outcome.unwrap();
                    assert_eq!(hits.hits()[0].hit().doc_id.as_str(), "quality-only");
                    assert!(std::ptr::eq(hits.snapshot(), cohort.get("quality").unwrap()));
                    assert_eq!(stream.state(), PublishedSearchStreamState::Complete);
                } else {
                    assert!(outcome.is_err());
                    assert_eq!(
                        stream.state(),
                        PublishedSearchStreamState::Failed { tier: 0 }
                    );
                }
                assert!(stream.next(&cx).await.is_none());
                assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn a_ready_poll_crossing_the_budget_is_rejected_at_the_exact_boundary() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for finish_ms in [49, 50, 51] {
                let elapsed = AtomicU64::new(0);
                let result = run_bounded_with_elapsed(
                    &cx,
                    cx.now(),
                    50,
                    async {
                        elapsed.store(finish_ms, Ordering::SeqCst);
                        Ok(7)
                    },
                    || Duration::from_millis(elapsed.load(Ordering::SeqCst)),
                )
                .await;
                if finish_ms < 50 {
                    assert_eq!(result.unwrap(), 7);
                } else {
                    assert!(matches!(
                        result,
                        Err(SearchError::SearchTimeout { elapsed_ms, budget_ms: 50 })
                            if elapsed_ms == finish_ms
                    ));
                }
            }
        });
    }

    #[test]
    fn expiry_drops_pending_work_without_repolling_or_cancelling_the_caller() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let elapsed = AtomicU64::new(0);
            let polls = AtomicUsize::new(0);
            let drops = Arc::new(AtomicUsize::new(0));
            let work = async {
                let _dropped = Dropped(Arc::clone(&drops));
                poll_fn(|_| {
                    polls.fetch_add(1, Ordering::SeqCst);
                    Poll::<SearchResult<()>>::Pending
                })
                .await
            };
            let mut bounded = Box::pin(run_bounded_with_elapsed(
                &cx,
                cx.now(),
                50,
                work,
                || Duration::from_millis(elapsed.load(Ordering::SeqCst)),
            ));
            let mut context = Context::from_waker(Waker::noop());
            assert!(bounded.as_mut().poll(&mut context).is_pending());
            assert_eq!(polls.load(Ordering::SeqCst), 1);
            elapsed.store(50, Ordering::SeqCst);
            assert!(matches!(
                bounded.as_mut().poll(&mut context),
                Poll::Ready(Err(SearchError::SearchTimeout { budget_ms: 50, .. }))
            ));
            assert_eq!(polls.load(Ordering::SeqCst), 1);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn expired_work_is_not_started_and_observed_identity_failure_is_not_a_timeout() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let calls = AtomicUsize::new(0);
            let result = run_bounded_with_elapsed(
                &cx,
                cx.now(),
                50,
                async {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
                || Duration::from_millis(50),
            )
            .await;
            assert!(matches!(result, Err(SearchError::SearchTimeout { .. })));
            assert_eq!(calls.load(Ordering::SeqCst), 0);

            let elapsed = AtomicU64::new(0);
            let result = run_bounded_with_elapsed(
                &cx,
                cx.now(),
                50,
                async {
                    elapsed.store(60, Ordering::SeqCst);
                    Err::<(), _>(SearchError::UnverifiableRemoteSpace {
                        producer: "test".to_owned(),
                        reason: "changed producer".to_owned(),
                    })
                },
                || Duration::from_millis(elapsed.load(Ordering::SeqCst)),
            )
            .await;
            assert!(matches!(
                result,
                Err(SearchError::UnverifiableRemoteSpace { .. })
            ));
        });
    }

    #[test]
    fn cancellation_wins_over_a_late_ready_poll_or_provider_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let elapsed = AtomicU64::new(0);
            let result = run_bounded_with_elapsed(
                &cx,
                cx.now(),
                50,
                async {
                    elapsed.store(60, Ordering::SeqCst);
                    cx.cancel_with(asupersync::CancelKind::User, Some("budget cancellation"));
                    Err::<(), _>(invalid("provider failed"))
                },
                || Duration::from_millis(elapsed.load(Ordering::SeqCst)),
            )
            .await;
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
        });
    }
}
