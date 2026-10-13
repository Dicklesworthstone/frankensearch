//! Admission credits follow the owned text through queueing and dispatch.
//!
//! The budget lock never acquires the queue lock. Submission may hold the
//! queue lock while reserving credits, and request destruction may release
//! credits while draining that queue, without introducing a lock-order cycle.

use std::sync::{Arc, Mutex, MutexGuard};

use frankensearch_core::{
    Cx, Embedder, EmbeddingIdentityBundleV1, IdentityBoundEmbedding, SearchError, SearchResult,
};

use super::CoalescedBatch;

/// Bounds shared by queued requests and all outstanding dispatched batches.
///
/// These bound admitted input storage, not model scratch space, output vectors,
/// channel/allocator overhead, or memory a caller allocated before submission.
/// Text accounting uses `String::capacity`, not length: a short string retaining
/// a large allocation cannot evade the budget. Dispatch does not release credit;
/// delivery or dropping the request/batch does. Dropping a result receiver alone
/// does not release text that the queue or a worker still owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoalescerLimits {
    /// Maximum admitted requests, queued plus in flight. Default: 1,024.
    pub max_outstanding_requests: usize,
    /// Maximum sum of admitted text capacities. Default: 16 MiB.
    pub max_outstanding_text_bytes: usize,
}

impl Default for CoalescerLimits {
    fn default() -> Self {
        Self {
            max_outstanding_requests: 1_024,
            max_outstanding_text_bytes: 16 * 1024 * 1024,
        }
    }
}

impl CoalescerLimits {
    pub(super) fn validate(self) -> SearchResult<()> {
        if self.max_outstanding_requests == 0 || self.max_outstanding_text_bytes == 0 {
            return Err(SearchError::InvalidConfig {
                field: "batch_coalescer.limits".to_owned(),
                value: "zero".to_owned(),
                reason: "request and text-byte admission limits must both be positive".to_owned(),
            });
        }
        Ok(())
    }
}

/// A coherent instantaneous snapshot of queued and in-flight admission usage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CoalescerUsage {
    /// Requests whose text is still owned by the coalescer or a dispatched batch.
    pub outstanding_requests: usize,
    /// Sum of those requests' original text allocation capacities.
    pub outstanding_text_bytes: usize,
}

pub(super) struct AdmissionBudget {
    limits: CoalescerLimits,
    used: Mutex<CoalescerUsage>,
}

impl AdmissionBudget {
    pub(super) fn new(limits: CoalescerLimits) -> Self {
        Self {
            limits,
            used: Mutex::new(CoalescerUsage::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, CoalescerUsage> {
        self.used
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) const fn limits(&self) -> CoalescerLimits {
        self.limits
    }

    pub(super) fn usage(&self) -> CoalescerUsage {
        *self.lock()
    }

    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> SearchResult<AdmissionPermit> {
        if bytes > self.limits.max_outstanding_text_bytes {
            // This request can never fit, even after every worker drains. It
            // is an invalid request size, not retryable capacity contention.
            return Err(SearchError::InvalidConfig {
                field: "batch_coalescer.text_capacity".to_owned(),
                value: bytes.to_string(),
                reason: "one text allocation exceeds the entire text-byte budget".to_owned(),
            });
        }
        let mut used = self.lock();
        if used.outstanding_requests >= self.limits.max_outstanding_requests {
            return Err(SearchError::QueueFull {
                pending: used.outstanding_requests,
                capacity: self.limits.max_outstanding_requests,
            });
        }
        // Subtraction is safe under the invariant; unlike addition it cannot
        // overflow even when a caller explicitly selects usize::MAX limits.
        if bytes > self.limits.max_outstanding_text_bytes - used.outstanding_text_bytes {
            return Err(SearchError::QueueFull {
                pending: used.outstanding_text_bytes,
                capacity: self.limits.max_outstanding_text_bytes,
            });
        }
        used.outstanding_requests += 1;
        used.outstanding_text_bytes += bytes;
        drop(used);
        Ok(AdmissionPermit {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

/// Not cloneable: exactly one credit belongs to each admitted request.
pub(super) struct AdmissionPermit {
    budget: Arc<AdmissionBudget>,
    bytes: usize,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        let mut used = self.budget.lock();
        used.outstanding_requests -= 1;
        used.outstanding_text_bytes -= self.bytes;
    }
}

impl CoalescedBatch {
    /// Execute admitted inference and deliver only a complete validated batch.
    ///
    /// `expected` is the caller's frozen, trusted in-process producer contract,
    /// not a new identity inferred from model labels or the response. Compare
    /// it with the selected index before dispatch, and use one selected contract
    /// for the callers sharing a queue. This method does not bind a queue to an
    /// index or authenticate a remote provider by itself. The existing result
    /// channels carry raw values; their receivers must retain that same contract.
    ///
    /// Producers declaring `bound_batch_is_native()` receive one ordered bound
    /// batch. Other producers receive one `embed_bound` per input, preserving
    /// singleton overrides instead of bypassing them via the default raw batch.
    /// There is no retry, model substitution, detached task or new runtime.
    /// Before any delivery, every response must match the complete expected
    /// identity, exact dimensions, finite coordinates and in-process storage
    /// contract. An invalid final row cannot expose an earlier successful prefix.
    ///
    /// Cancellation wins over both inference success and ordinary failure. After
    /// the final checkpoint, delivery has no await or provider callback; it does
    /// not report cancellation after partially sending validated results. Each
    /// private channel has capacity one and receives at most one message.
    /// Disconnected receivers are skipped. Completion or dropping this future
    /// releases input credits; abandoning it disconnects undelivered receivers.
    /// Provider-owned blocking work is not preempted by dropping its future.
    ///
    /// # Errors
    /// Returns the detailed admission error or original inference error to the
    /// dispatch owner. Unavailable producer identity is a typed identity refusal.
    /// Receivers get the same cancellation class, or a bounded generic embedding
    /// failure for other errors, without formatting opaque provider exceptions.
    /// Even an empty batch checks expected/advertised identity and cancellation,
    /// but starts no inference. The low-level `deliver` API remains unchecked.
    pub async fn embed_and_deliver(
        self,
        cx: &Cx,
        embedder: &dyn Embedder,
        expected: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<()> {
        let outcome = self.admit_inference(cx, embedder, expected).await;
        let outcome = match inference_checkpoint(cx, "batch_coalescer.delivery") {
            Ok(()) => outcome,
            Err(cancelled) => Err(cancelled),
        };
        match outcome {
            Ok(responses) => {
                // Count and every row were admitted before this consuming zip.
                for (request, response) in self.requests.into_iter().zip(responses) {
                    let _ = request.result_tx.send(Ok(response.values));
                }
                Ok(())
            }
            Err(error) => {
                for request in self.requests {
                    let receiver_error = match &error {
                        SearchError::Cancelled { phase, reason } => SearchError::Cancelled {
                            phase: phase.clone(),
                            reason: reason.clone(),
                        },
                        _ => SearchError::EmbeddingFailed {
                            model: "batch_coalescer".to_owned(),
                            source: "identity-bound batch failed; dispatcher retains the detailed error"
                                .into(),
                        },
                    };
                    let _ = request.result_tx.send(Err(receiver_error));
                }
                Err(error)
            }
        }
    }

    async fn admit_inference(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        expected: &EmbeddingIdentityBundleV1,
    ) -> SearchResult<Vec<IdentityBoundEmbedding>> {
        inference_checkpoint(cx, "batch_coalescer.inference")?;
        expected.validate()?;
        check_producer(cx, embedder, expected)?;
        if self.is_empty() {
            return Ok(Vec::new());
        }
        let native = embedder.bound_batch_is_native();
        inference_checkpoint(cx, "batch_coalescer.batch_policy")?;
        let responses = if native {
            let texts = self.texts();
            let outcome = embedder.embed_batch_bound(cx, &texts).await;
            admit_outcome(cx, embedder, expected, outcome)?
        } else {
            let mut responses = Vec::with_capacity(self.len());
            for request in &self.requests {
                check_producer(cx, embedder, expected)?;
                let outcome = embedder.embed_bound(cx, &request.text).await;
                let response = admit_outcome(cx, embedder, expected, outcome)?;
                check_response(&response, expected)?;
                responses.push(response);
            }
            responses
        };
        if responses.len() != self.len() {
            return Err(SearchError::InvalidConfig {
                field: "batch_coalescer.response_count".to_owned(),
                value: responses.len().to_string(),
                reason: "bound inference must return exactly one output per queued input".to_owned(),
            });
        }
        for response in &responses {
            inference_checkpoint(cx, "batch_coalescer.response")?;
            check_response(response, expected)?;
        }
        Ok(responses)
    }
}

fn inference_checkpoint(cx: &Cx, phase: &str) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        return Err(SearchError::Cancelled {
            phase: phase.to_owned(),
            reason: cx.cancel_reason().map_or_else(
                || "coalesced embedding cancelled".to_owned(),
                |reason| reason.to_string(),
            ),
        });
    }
    Ok(())
}

fn identity_mismatch() -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "batch_coalescer".to_owned(),
        reason: "producer or response differs from the caller's frozen embedding contract"
            .to_owned(),
    }
}

fn check_producer(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    inference_checkpoint(cx, "batch_coalescer.before_identity")?;
    let actual = embedder.identity();
    inference_checkpoint(cx, "batch_coalescer.after_identity")?;
    if !actual.is_ok_and(|identity| identity == expected) {
        return Err(identity_mismatch());
    }
    let dimension = embedder.dimension();
    inference_checkpoint(cx, "batch_coalescer.after_dimension")?;
    if usize::try_from(expected.space.dimension).ok() != Some(dimension) {
        return Err(identity_mismatch());
    }
    Ok(())
}

fn admit_outcome<T>(
    cx: &Cx,
    embedder: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
    outcome: SearchResult<T>,
) -> SearchResult<T> {
    inference_checkpoint(cx, "batch_coalescer.after_inference")?;
    if matches!(&outcome, Err(SearchError::Cancelled { .. })) {
        return outcome;
    }
    // Losing/changing identity cannot become an ordinary retryable model error.
    check_producer(cx, embedder, expected)?;
    outcome
}

fn check_response(
    response: &IdentityBoundEmbedding,
    expected: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    // Refuse foreign fields before validators can reflect their diagnostic text.
    if &response.identity != expected {
        return Err(identity_mismatch());
    }
    response.validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_coalescer::{BatchCoalescer, CoalescerConfig, Priority};
    use std::sync::Barrier;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;
    use std::thread;

    fn bounded(requests: usize, bytes: usize, batch: usize) -> BatchCoalescer {
        BatchCoalescer::with_limits(
            CoalescerConfig {
                max_batch_size: batch,
                max_wait_ms: 0,
                min_batch_size: 1,
                use_priority_lanes: true,
            },
            CoalescerLimits {
                max_outstanding_requests: requests,
                max_outstanding_text_bytes: bytes,
            },
        )
        .unwrap()
    }

    #[test]
    fn default_admission_is_bounded_and_explicit_zero_limits_are_refused() {
        let queue = BatchCoalescer::new(CoalescerConfig::default());
        assert_eq!(queue.limits().max_outstanding_requests, 1_024);
        assert_eq!(queue.limits().max_outstanding_text_bytes, 16 * 1024 * 1024);
        assert_eq!(queue.usage(), CoalescerUsage::default());
        for (requests, bytes) in [(0, 1), (1, 0), (0, 0)] {
            assert!(matches!(
                BatchCoalescer::with_limits(
                    CoalescerConfig::default(),
                    CoalescerLimits {
                        max_outstanding_requests: requests,
                        max_outstanding_text_bytes: bytes,
                    },
                ),
                Err(SearchError::InvalidConfig { .. })
            ));
        }
        // A giant batch setting must not cause a giant eager queue allocation.
        let queue = bounded(2, 32, usize::MAX);
        assert_eq!(queue.usage(), CoalescerUsage::default());
        let queue = bounded(2, 32, 0);
        assert_eq!(queue.config().max_batch_size, 1);
    }

    #[test]
    fn dispatch_does_not_free_request_credits_and_completion_does() {
        let queue = bounded(2, 32, 1);
        let first = queue.submit("a".into(), Priority::Background);
        let second = queue.submit("b".into(), Priority::Interactive);
        let batch = queue.try_form_batch().unwrap();
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.usage().outstanding_requests, 2);
        let rejected = queue.submit("c".into(), Priority::Background);
        assert!(matches!(
            rejected.try_recv().unwrap(),
            Err(SearchError::QueueFull {
                pending: 2,
                capacity: 2
            })
        ));
        batch.deliver(Ok(vec![vec![11.0]]));
        assert_eq!(first.try_recv().unwrap().unwrap(), vec![11.0]);
        let third = queue.submit("c".into(), Priority::Background);
        assert_eq!(queue.usage().outstanding_requests, 2);
        queue
            .try_form_batch()
            .unwrap()
            .deliver(Ok(vec![vec![22.0]]));
        queue
            .try_form_batch()
            .unwrap()
            .deliver(Ok(vec![vec![33.0]]));
        assert_eq!(second.try_recv().unwrap().unwrap(), vec![22.0]);
        assert_eq!(third.try_recv().unwrap().unwrap(), vec![33.0]);
        assert_eq!(queue.usage(), CoalescerUsage::default());
        assert_eq!(
            queue.metrics().rejected_submissions.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn text_capacity_not_length_is_charged_through_dispatch() {
        let mut text = String::with_capacity(64);
        text.push('x');
        let bytes = text.capacity();
        let queue = bounded(8, bytes, 1);
        let first = queue.submit(text, Priority::Background);
        let batch = queue.try_form_batch().unwrap();
        assert_eq!(queue.usage().outstanding_text_bytes, bytes);
        assert!(
            matches!(queue.submit("y".into(), Priority::Background).try_recv().unwrap(),
            Err(SearchError::QueueFull { pending, capacity }) if pending == bytes && capacity == bytes)
        );
        let mut oversized = String::with_capacity(bytes + 1);
        oversized.push('z');
        assert!(
            matches!(queue.submit(oversized, Priority::Background).try_recv().unwrap(),
            Err(SearchError::InvalidConfig { ref field, .. }) if field == "batch_coalescer.text_capacity")
        );
        // Empty, zero-capacity strings still consume request credits, not bytes.
        let empty = queue.submit(String::new(), Priority::Interactive);
        assert_eq!(queue.usage().outstanding_requests, 2);
        drop(batch);
        assert!(matches!(first.try_recv(), Err(TryRecvError::Disconnected)));
        assert_eq!(queue.usage().outstanding_text_bytes, 0);
        queue.try_form_batch().unwrap().deliver(Ok(vec![vec![0.0]]));
        assert_eq!(empty.try_recv().unwrap().unwrap(), vec![0.0]);
        assert_eq!(queue.usage(), CoalescerUsage::default());
    }

    #[test]
    fn rejected_inputs_leave_no_request_or_budget_state() {
        let queue = bounded(1, 8, 1);
        let held = queue.submit(String::new(), Priority::Interactive);
        let before = queue.usage();
        for _ in 0..16 {
            assert!(
                queue
                    .submit(String::new(), Priority::Background)
                    .try_recv()
                    .unwrap()
                    .is_err()
            );
            assert_eq!(queue.pending_count(), 1);
            assert_eq!(queue.usage(), before);
        }
        drop(held);
        // A receiver drop does not free text still owned by the queued request.
        assert_eq!(queue.usage(), before);
        drop(queue.try_form_batch().unwrap());
        assert_eq!(queue.usage(), CoalescerUsage::default());
    }

    #[test]
    fn shutdown_refuses_late_submissions_but_preserves_admitted_work() {
        let queue = bounded(4, 32, 1);
        let queued = queue.submit("before".into(), Priority::Background);
        queue.shutdown();
        for _ in 0..2 {
            queue.shutdown();
            assert!(
                matches!(queue.submit("after".into(), Priority::Interactive).try_recv().unwrap(),
                Err(SearchError::Cancelled { ref phase, .. }) if phase == "batch_coalescer.submit")
            );
        }
        assert_eq!(queue.pending_count(), 1);
        queue.wait_for_batch().unwrap().deliver(Ok(vec![vec![1.0]]));
        assert_eq!(queued.try_recv().unwrap().unwrap(), vec![1.0]);
        assert!(queue.wait_for_batch().is_none());
        assert!(matches!(
            queue
                .submit("late".into(), Priority::Background)
                .try_recv()
                .unwrap(),
            Err(SearchError::Cancelled { .. })
        ));
        assert!(queue.try_form_batch().is_none());
        assert_eq!(queue.usage(), CoalescerUsage::default());
    }

    #[test]
    fn dropping_queue_and_batches_releases_exactly_their_credits() {
        let queue = bounded(3, 32, 1);
        let first = queue.submit("first".into(), Priority::Background);
        let second = queue.submit("second".into(), Priority::Background);
        let batch = queue.try_form_batch().unwrap();
        let budget = Arc::clone(&queue.admission);
        drop(queue);
        assert!(matches!(second.try_recv(), Err(TryRecvError::Disconnected)));
        assert_eq!(budget.usage().outstanding_requests, 1);
        assert!(matches!(first.try_recv(), Err(TryRecvError::Empty)));
        batch.deliver(Err(SearchError::Cancelled {
            phase: "fixture".into(),
            reason: "stopped".into(),
        }));
        assert!(first.try_recv().unwrap().is_err());
        assert_eq!(budget.usage(), CoalescerUsage::default());
        let weak = Arc::downgrade(&budget);
        drop(budget);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn simultaneous_submitters_cannot_overbook_or_cross_deliver() {
        let queue = Arc::new(bounded(8, 128, 3));
        let barrier = Arc::new(Barrier::new(33));
        let workers: Vec<_> = (0..32)
            .map(|id| {
                let queue = Arc::clone(&queue);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let text = id.to_string().into_boxed_str().into_string();
                    (id, queue.submit(text, Priority::Background))
                })
            })
            .collect();
        barrier.wait();
        let mut accepted = Vec::new();
        let mut rejected = 0;
        for worker in workers {
            let (id, receiver) = worker.join().unwrap();
            match receiver.try_recv() {
                Err(TryRecvError::Empty) => accepted.push((id, receiver)),
                Ok(Err(SearchError::QueueFull { .. })) => rejected += 1,
                _ => panic!("unexpected admission result"),
            }
        }
        assert_eq!(accepted.len(), 8);
        assert_eq!(rejected, 24);
        assert_eq!(queue.usage().outstanding_requests, 8);
        let mut batches = Vec::new();
        while let Some(batch) = queue.try_form_batch() {
            batches.push(batch);
        }
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.usage().outstanding_requests, 8);
        for batch in batches {
            let results = batch
                .texts()
                .iter()
                .map(|text| vec![text.parse::<f32>().unwrap()])
                .collect();
            batch.deliver(Ok(results));
        }
        for (id, receiver) in accepted {
            assert_eq!(
                receiver.try_recv().unwrap().unwrap(),
                vec![id.to_string().parse::<f32>().unwrap()]
            );
        }
        assert_eq!(queue.usage(), CoalescerUsage::default());
    }

    #[test]
    fn submit_shutdown_race_accounts_for_every_caller_without_sleeping() {
        let queue = Arc::new(bounded(32, 1_024, 3));
        let barrier = Arc::new(Barrier::new(17));
        // Every worker must be spawned before the barrier below; a lazy
        // iterator would spawn none and deadlock the 17-party barrier.
        #[allow(clippy::needless_collect)]
        let workers: Vec<_> = (0..16)
            .map(|id| {
                let queue = Arc::clone(&queue);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    queue.submit(id.to_string(), Priority::Background)
                })
            })
            .collect();
        barrier.wait();
        queue.shutdown();
        let receivers: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();
        while let Some(batch) = queue.wait_for_batch() {
            let results = vec![vec![1.0]; batch.len()];
            batch.deliver(Ok(results));
        }
        for receiver in receivers {
            assert!(matches!(
                receiver.try_recv().unwrap(),
                Ok(_) | Err(SearchError::Cancelled { .. })
            ));
        }
        assert_eq!(queue.usage(), CoalescerUsage::default());
        assert_eq!(queue.metrics().total_submitted.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn byte_accounting_at_address_space_limit_does_not_wrap() {
        let budget = Arc::new(AdmissionBudget::new(CoalescerLimits {
            max_outstanding_requests: 2,
            max_outstanding_text_bytes: usize::MAX,
        }));
        // Exercise integer admission without allocating an address space of text.
        let permit = budget.reserve(usize::MAX).unwrap();
        assert!(matches!(
            budget.reserve(1),
            Err(SearchError::QueueFull { .. })
        ));
        let empty = budget.reserve(0).unwrap();
        assert!(matches!(
            budget.reserve(0),
            Err(SearchError::QueueFull { .. })
        ));
        drop(permit);
        assert_eq!(budget.usage().outstanding_text_bytes, 0);
        drop(empty);
        assert_eq!(budget.usage(), CoalescerUsage::default());
    }
}

#[cfg(test)]
mod inference_tests {
    use super::*;
    use crate::batch_coalescer::{BatchCoalescer, CoalescerConfig, Priority};
    use frankensearch_core::{ModelCategory, SearchFuture};
    use std::fmt;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{Receiver, TryRecvError};
    use std::task::{Context, Poll, Waker};

    #[derive(Clone, Copy)]
    enum Reply {
        Valid,
        Short,
        Long,
        ForeignSpace,
        ForeignProducer,
        ForeignStorage,
        WrongDimension,
        NonFinite(f32),
        Failure,
        DriftSuccess,
        DriftFailure,
        CancelSuccess,
        CancelFailure,
        ProviderCancelled,
        Pending,
    }

    struct Probe {
        expected: EmbeddingIdentityBundleV1,
        foreign: EmbeddingIdentityBundleV1,
        drifted: AtomicBool,
        native: bool,
        reply: Reply,
        raw_calls: AtomicUsize,
        bound_calls: AtomicUsize,
        batch_calls: AtomicUsize,
        identity_calls: AtomicUsize,
        formatted: Arc<AtomicUsize>,
        dropped: AtomicUsize,
        cancel_metadata: Option<(Cx, usize)>,
    }

    impl Probe {
        fn new(native: bool, reply: Reply) -> Self {
            Self {
                expected: identity(),
                foreign: EmbeddingIdentityBundleV1::explicit_test_model("foreign-space", 2),
                drifted: AtomicBool::new(false),
                native,
                reply,
                raw_calls: AtomicUsize::new(0),
                bound_calls: AtomicUsize::new(0),
                batch_calls: AtomicUsize::new(0),
                identity_calls: AtomicUsize::new(0),
                formatted: Arc::new(AtomicUsize::new(0)),
                dropped: AtomicUsize::new(0),
                cancel_metadata: None,
            }
        }

        async fn produce(
            &self,
            cx: &Cx,
            texts: &[&str],
        ) -> SearchResult<Vec<IdentityBoundEmbedding>> {
            let _guard = DropGuard(&self.dropped);
            if matches!(self.reply, Reply::Pending) {
                return std::future::pending().await;
            }
            if matches!(self.reply, Reply::ProviderCancelled) {
                return Err(SearchError::Cancelled {
                    phase: "fixture.provider".to_owned(),
                    reason: "provider stopped".to_owned(),
                });
            }
            let mut values: Vec<_> = texts.iter().map(|text| IdentityBoundEmbedding {
                values: vec![text.parse().unwrap(), -0.0],
                identity: self.expected.clone(),
            }).collect();
            // Put corruption LAST: a streaming prefix delivery would already
            // expose two vectors before noticing it. Singleton execution also
            // fails only on the third input, after two real bound calls succeeded.
            if texts.contains(&"30") {
                match self.reply {
                    Reply::Short => { let _ = values.pop(); }
                    Reply::Long => values.push(values[0].clone()),
                    Reply::ForeignSpace => {
                        values.last_mut().unwrap().identity = self.foreign.clone();
                    }
                    Reply::ForeignProducer => {
                        values.last_mut().unwrap().identity.producer.backend = "foreign".to_owned();
                    }
                    Reply::ForeignStorage => {
                        values.last_mut().unwrap().identity.storage.format = "fsvi-v2".to_owned();
                    }
                    Reply::WrongDimension => { let _ = values.last_mut().unwrap().values.pop(); }
                    Reply::NonFinite(value) => values.last_mut().unwrap().values[1] = value,
                    _ => {}
                }
            }
            if matches!(self.reply, Reply::DriftSuccess | Reply::DriftFailure) {
                self.drifted.store(true, Ordering::Relaxed);
            }
            if matches!(self.reply, Reply::CancelSuccess | Reply::CancelFailure) {
                cx.set_cancel_requested(true);
            }
            if matches!(self.reply, Reply::Failure | Reply::DriftFailure | Reply::CancelFailure) {
                return Err(SearchError::EmbeddingFailed {
                    model: "fixture".to_owned(),
                    source: Box::new(Opaque(Arc::clone(&self.formatted))),
                });
            }
            Ok(values)
        }
    }

    struct DropGuard<'a>(&'a AtomicUsize);
    impl Drop for DropGuard<'_> {
        fn drop(&mut self) { self.0.fetch_add(1, Ordering::Relaxed); }
    }

    struct Opaque(Arc<AtomicUsize>);
    impl fmt::Display for Opaque {
        fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fetch_add(1, Ordering::Relaxed);
            Err(fmt::Error)
        }
    }
    impl fmt::Debug for Opaque {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self, formatter)
        }
    }
    impl std::error::Error for Opaque {}

    impl Embedder for Probe {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.raw_calls.fetch_add(1, Ordering::Relaxed);
            // Raw inference appears healthy, making bypassed bound validation
            // observable as a wrong successful result rather than a stub panic.
            Box::pin(async { Ok(vec![99.0, 0.0]) })
        }

        fn embed_bound<'a>(
            &'a self, cx: &'a Cx, text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            self.bound_calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move {
                let mut values = self.produce(cx, &[text]).await?;
                values.pop().ok_or_else(|| SearchError::EmbeddingFailed {
                    model: "fixture".to_owned(), source: "missing singleton".into(),
                })
            })
        }

        fn embed_batch_bound<'a>(
            &'a self, cx: &'a Cx, texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(self.produce(cx, texts))
        }

        fn bound_batch_is_native(&self) -> bool { self.native }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            let call = self.identity_calls.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some((cx, at)) = &self.cancel_metadata
                && call == *at
            {
                cx.set_cancel_requested(true);
            }
            Ok(if self.drifted.load(Ordering::Relaxed) { &self.foreign } else { &self.expected })
        }
        fn dimension(&self) -> usize { 2 }
        fn id(&self) -> &str { "coalescer-fixture" }
        fn model_name(&self) -> &str { self.id() }
        fn is_semantic(&self) -> bool { false }
        fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
    }

    fn identity() -> EmbeddingIdentityBundleV1 {
        EmbeddingIdentityBundleV1::explicit_test_model("coalescer-fixture", 2)
    }

    fn work() -> (BatchCoalescer, CoalescedBatch, Vec<Receiver<SearchResult<Vec<f32>>>>) {
        let queue = BatchCoalescer::with_limits(
            CoalescerConfig {
                max_batch_size: 3, max_wait_ms: 0, min_batch_size: 1, use_priority_lanes: true,
            },
            CoalescerLimits { max_outstanding_requests: 3, max_outstanding_text_bytes: 32 },
        ).unwrap();
        let receivers = ["10", "20", "30"].into_iter()
            .map(|text| queue.submit(text.to_owned(), Priority::Background)).collect();
        let batch = queue.try_form_batch().unwrap();
        (queue, batch, receivers)
    }

    fn failed_receivers(receivers: Vec<Receiver<SearchResult<Vec<f32>>>>, cancelled: bool) {
        for receiver in receivers {
            let error = receiver.try_recv().unwrap().expect_err("no prefix success is allowed");
            assert_eq!(matches!(error, SearchError::Cancelled { .. }), cancelled);
            if !cancelled {
                assert!(matches!(error, SearchError::EmbeddingFailed { .. }));
                assert!(error.to_string().len() < 256);
            }
            assert!(matches!(receiver.try_recv(), Err(TryRecvError::Disconnected)));
        }
    }

    #[test]
    fn bound_executor_delivers_original_bits_in_order_for_native_and_singleton_producers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for native in [false, true] {
                let probe = Probe::new(native, Reply::Valid);
                let (queue, batch, receivers) = work();
                batch.embed_and_deliver(&cx, &probe, &identity()).await.unwrap();
                for (receiver, expected) in receivers.into_iter().zip([10.0_f32, 20.0, 30.0]) {
                    let values = receiver.try_recv().unwrap().unwrap();
                    assert_eq!(values[0].to_bits(), expected.to_bits());
                    assert_eq!(values[1].to_bits(), (-0.0_f32).to_bits());
                }
                assert_eq!(probe.raw_calls.load(Ordering::Relaxed), 0);
                assert_eq!(probe.batch_calls.load(Ordering::Relaxed), usize::from(native));
                assert_eq!(probe.bound_calls.load(Ordering::Relaxed), if native { 0 } else { 3 });
                assert_eq!(queue.usage(), CoalescerUsage::default());
            }
        });
    }

    #[test]
    fn malformed_final_response_rejects_the_entire_batch_before_any_delivery() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for native in [false, true] {
                for reply in [
                    Reply::ForeignSpace, Reply::ForeignProducer, Reply::ForeignStorage,
                    Reply::WrongDimension, Reply::NonFinite(f32::NAN),
                    Reply::NonFinite(f32::INFINITY), Reply::NonFinite(f32::NEG_INFINITY),
                ] {
                    let probe = Probe::new(native, reply);
                    let (queue, batch, receivers) = work();
                    assert!(batch.embed_and_deliver(&cx, &probe, &identity()).await.is_err());
                    failed_receivers(receivers, false);
                    assert_eq!(queue.usage(), CoalescerUsage::default());
                    assert_eq!(probe.raw_calls.load(Ordering::Relaxed), 0);
                    if !native { assert_eq!(probe.bound_calls.load(Ordering::Relaxed), 3); }
                }
            }
            for reply in [Reply::Short, Reply::Long] {
                let probe = Probe::new(true, reply);
                let (queue, batch, receivers) = work();
                let error = batch.embed_and_deliver(&cx, &probe, &identity()).await.unwrap_err();
                assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                    if field == "batch_coalescer.response_count"));
                failed_receivers(receivers, false);
                assert_eq!(queue.usage(), CoalescerUsage::default());
            }
        });
    }

    #[test]
    fn advertised_drift_is_rejected_before_inference_and_after_success_or_failure() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for before in [false, true] {
                for reply in [Reply::DriftSuccess, Reply::DriftFailure] {
                    let probe = Probe::new(true, reply);
                    probe.drifted.store(before, Ordering::Relaxed);
                    let (queue, batch, receivers) = work();
                    let error = batch.embed_and_deliver(&cx, &probe, &identity()).await.unwrap_err();
                    assert!(matches!(error, SearchError::UnverifiableRemoteSpace { .. }));
                    assert_eq!(probe.batch_calls.load(Ordering::Relaxed), usize::from(!before));
                    assert_eq!(probe.formatted.load(Ordering::Relaxed), 0);
                    failed_receivers(receivers, false);
                    assert_eq!(queue.usage(), CoalescerUsage::default());
                }
            }
        });
    }

    #[test]
    fn cancellation_wins_at_entry_metadata_and_both_provider_outcomes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for reply in [Reply::CancelSuccess, Reply::CancelFailure, Reply::ProviderCancelled] {
                for native in [false, true] {
                    let probe = Probe::new(native, reply);
                    let (queue, batch, receivers) = work();
                    let error = batch.embed_and_deliver(&cx, &probe, &identity()).await.unwrap_err();
                    assert!(matches!(error, SearchError::Cancelled { .. }));
                    failed_receivers(receivers, true);
                    assert_eq!(probe.formatted.load(Ordering::Relaxed), 0);
                    assert_eq!(queue.usage(), CoalescerUsage::default());
                    cx.set_cancel_requested(false);
                }
            }
            for at in [0, 1, 2] {
                let mut probe = Probe::new(true, Reply::Valid);
                if at == 0 { cx.set_cancel_requested(true); }
                else { probe.cancel_metadata = Some((cx.clone(), at)); }
                let (queue, batch, receivers) = work();
                assert!(matches!(batch.embed_and_deliver(&cx, &probe, &identity()).await,
                    Err(SearchError::Cancelled { .. })));
                failed_receivers(receivers, true);
                assert_eq!(probe.batch_calls.load(Ordering::Relaxed), usize::from(at == 2));
                assert_eq!(queue.usage(), CoalescerUsage::default());
                cx.set_cancel_requested(false);
            }
        });
    }

    #[test]
    fn opaque_provider_error_is_returned_to_dispatcher_without_formatting_or_fanout_copy() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let probe = Probe::new(true, Reply::Failure);
            let (queue, batch, receivers) = work();
            let error = batch.embed_and_deliver(&cx, &probe, &identity()).await.unwrap_err();
            match error {
                SearchError::EmbeddingFailed { source, .. } => {
                    let opaque = source.downcast_ref::<Opaque>().expect("original source retained");
                    assert!(Arc::ptr_eq(&opaque.0, &probe.formatted));
                }
                _ => panic!("original provider error must reach dispatcher"),
            }
            failed_receivers(receivers, false);
            assert_eq!(probe.formatted.load(Ordering::Relaxed), 0);
            assert_eq!(probe.batch_calls.load(Ordering::Relaxed), 1);
            assert_eq!(queue.usage(), CoalescerUsage::default());
        });
    }

    #[test]
    fn abandoning_pending_inference_releases_credits_and_requires_explicit_resubmission() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let probe = Probe::new(true, Reply::Pending);
            let (queue, batch, receivers) = work();
            let expected = identity();
            let mut future = Box::pin(batch.embed_and_deliver(&cx, &probe, &expected));
            assert!(matches!(future.as_mut().poll(&mut Context::from_waker(Waker::noop())), Poll::Pending));
            assert_eq!(queue.usage().outstanding_requests, 3);
            for receiver in &receivers { assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty))); }
            drop(future);
            assert_eq!(probe.dropped.load(Ordering::Relaxed), 1);
            for receiver in receivers { assert!(matches!(receiver.try_recv(), Err(TryRecvError::Disconnected))); }
            assert_eq!(queue.usage(), CoalescerUsage::default());
            assert_eq!(queue.pending_count(), 0);
            assert!(!cx.is_cancel_requested());
            let receiver = queue.submit("10".to_owned(), Priority::Interactive);
            let ready = Probe::new(true, Reply::Valid);
            queue.try_form_batch().unwrap().embed_and_deliver(&cx, &ready, &expected).await.unwrap();
            assert_eq!(receiver.try_recv().unwrap().unwrap()[0], 10.0);
            assert_eq!(queue.usage(), CoalescerUsage::default());
        });
    }

    #[test]
    fn empty_batches_validate_identity_without_model_work_and_unpolled_drop_releases_input() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let probe = Probe::new(true, Reply::Valid);
            let expected = identity();
            CoalescedBatch { requests: Vec::new() }.embed_and_deliver(&cx, &probe, &expected).await.unwrap();
            assert_eq!(probe.batch_calls.load(Ordering::Relaxed), 0);
            assert_eq!(probe.identity_calls.load(Ordering::Relaxed), 1);
            probe.drifted.store(true, Ordering::Relaxed);
            assert!(CoalescedBatch { requests: Vec::new() }.embed_and_deliver(&cx, &probe, &expected).await.is_err());
            let (queue, batch, receivers) = work();
            drop(batch.embed_and_deliver(&cx, &probe, &expected));
            assert_eq!(queue.usage(), CoalescerUsage::default());
            for receiver in receivers { assert!(matches!(receiver.try_recv(), Err(TryRecvError::Disconnected))); }
            assert_eq!(probe.batch_calls.load(Ordering::Relaxed), 0);
        });
    }
}
