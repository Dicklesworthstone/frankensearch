//! Admission credits follow the owned text through queueing and dispatch.
//!
//! The budget lock never acquires the queue lock. Submission may hold the
//! queue lock while reserving credits, and request destruction may release
//! credits while draining that queue, without introducing a lock-order cycle.

use std::sync::{Arc, Mutex, MutexGuard};

use frankensearch_core::{SearchError, SearchResult};

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
