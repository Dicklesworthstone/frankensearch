//! Cancellation-safe ownership of a bounded embedding batch.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{EmbeddingJob, EmbeddingQueue, JobOutcome, QueueState};

#[derive(Debug)]
pub(super) struct ActiveLease {
    token: Arc<()>,
    unacknowledged: HashSet<String>,
}

impl QueueState {
    pub(super) fn outstanding_count(&self) -> usize {
        self.jobs.len()
            + self
                .lease
                .as_ref()
                .map_or(0, |lease| lease.unacknowledged.len())
    }

    pub(super) fn release_leased_job(&mut self, doc_id: &str) {
        if let Some(lease) = &mut self.lease {
            lease.unacknowledged.remove(doc_id);
        }
    }
}

/// An embedding batch whose unacknowledged work is returned on drop.
///
/// Unlike destructive draining, a lease reserves each job's queue slot until
/// [`EmbeddingQueue::record_embedded`] acknowledges publication or
/// [`EmbeddingQueue::requeue`] decides its retry outcome. Cancellation and
/// dropping a suspended future return the remaining jobs without incrementing
/// their retry counts. A newer pending submission for the same document wins.
///
/// Only one lease may be active per queue. Other leases and destructive drains
/// return empty while it is held. Producers may still submit into unreserved
/// capacity or replace an already-pending document. No queue lock is held while
/// the consumer embeds or publishes its batch.
///
/// This is in-process, at-least-once ownership, not a durable journal. The
/// consumer remains responsible for serializing writes and acknowledging only
/// after publication succeeds.
#[must_use = "dropping the batch returns its unacknowledged jobs to the queue"]
pub struct EmbeddingBatch<'a> {
    queue: &'a EmbeddingQueue,
    jobs: Vec<EmbeddingJob>,
    token: Option<Arc<()>>,
}

impl<'a> EmbeddingBatch<'a> {
    const fn empty(queue: &'a EmbeddingQueue) -> Self {
        Self {
            queue,
            jobs: Vec::new(),
            token: None,
        }
    }

    /// Original jobs in submission order, including any subsequently acknowledged.
    #[must_use]
    pub fn jobs(&self) -> &[EmbeddingJob] {
        &self.jobs
    }

    /// Number of jobs originally leased.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether this lease acquired any jobs.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Whether an original job still belongs to this lease.
    #[must_use]
    pub fn is_unacknowledged(&self, index: usize) -> bool {
        let Some(job) = self.jobs.get(index) else {
            return false;
        };
        let state = self.queue.lock_state();
        match (&state.lease, &self.token) {
            (Some(active), Some(token)) => {
                Arc::ptr_eq(&active.token, token) && active.unacknowledged.contains(&job.doc_id)
            }
            _ => false,
        }
    }

    /// Retry only jobs that have not already succeeded or received a retry outcome.
    ///
    /// Uses the queue's normal retry limit. Reserved capacity prevents unrelated
    /// producers from crowding these retries out. Returns the number receiving
    /// `Failed` (retry exhaustion or supersession by a newer pending submission).
    /// Dropping the lease without calling this method does not consume retries.
    pub fn retry_unacknowledged(&self) -> usize {
        let mut failed = 0;
        for (index, job) in self.jobs.iter().enumerate() {
            if self.is_unacknowledged(index)
                && self.queue.requeue(job.clone()) == JobOutcome::Failed
            {
                failed += 1;
            }
        }
        failed
    }
}

impl std::fmt::Debug for EmbeddingBatch<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddingBatch")
            .field("jobs", &self.jobs.len())
            .field("active", &self.token.is_some())
            .finish_non_exhaustive()
    }
}

impl Drop for EmbeddingBatch<'_> {
    fn drop(&mut self) {
        let Some(token) = self.token.take() else {
            return;
        };
        let mut state = self.queue.lock_state();
        if !state
            .lease
            .as_ref()
            .is_some_and(|active| Arc::ptr_eq(&active.token, &token))
        {
            return;
        }
        let Some(mut active) = state.lease.take() else {
            return;
        };
        // Reverse insertion restores the original prefix order. Inspect actual
        // queued jobs as well as maintaining the index: a stale pending_ids
        // entry must not cause an unacknowledged job to disappear.
        for job in self.jobs.drain(..).rev() {
            if !active.unacknowledged.remove(&job.doc_id)
                || state.jobs.iter().any(|pending| pending.doc_id == job.doc_id)
            {
                continue;
            }
            let sequence = state.sequence;
            state.sequence = state.sequence.wrapping_add(1);
            state.pending_ids.insert(job.doc_id.clone(), sequence);
            state.jobs.push_front(job);
        }
        debug_assert!(active.unacknowledged.is_empty());
        debug_assert!(state.jobs.len() <= self.queue.config.capacity);
    }
}

impl EmbeddingQueue {
    /// Lease up to the configured batch size without relinquishing queue capacity.
    pub fn lease_batch(&self) -> EmbeddingBatch<'_> {
        self.lease_batch_up_to(self.config.batch_size)
    }

    /// Lease at most `limit` pending jobs for cancellation-safe processing.
    ///
    /// Returns an empty batch for an empty queue, zero limit, or another active
    /// lease. A consumer must drop its current lease before taking another one.
    pub fn lease_batch_up_to(&self, limit: usize) -> EmbeddingBatch<'_> {
        let mut state = self.lock_state();
        if limit == 0 || state.jobs.is_empty() || state.lease.is_some() {
            return EmbeddingBatch::empty(self);
        }
        let count = state.jobs.len().min(limit);
        let mut jobs = Vec::with_capacity(count);
        let mut unacknowledged = HashSet::with_capacity(count);
        let token = Arc::new(());
        for _ in 0..count {
            if let Some(job) = state.jobs.pop_front() {
                state.pending_ids.remove(&job.doc_id);
                state.known_hashes.remove(&job.doc_id);
                unacknowledged.insert(job.doc_id.clone());
                jobs.push(job);
            }
        }
        state.lease = Some(ActiveLease {
            token: Arc::clone(&token),
            unacknowledged,
        });
        self.metrics.total_batches.fetch_add(1, Ordering::Relaxed);
        EmbeddingBatch {
            queue: self,
            jobs,
            token: Some(token),
        }
    }

    /// Jobs currently reserved by the active lease and not yet acknowledged.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.lock_state()
            .lease
            .as_ref()
            .map_or(0, |lease| lease.unacknowledged.len())
    }

    /// Queued plus unacknowledged leased jobs, bounded by configured capacity.
    #[must_use]
    pub fn outstanding_count(&self) -> usize {
        self.lock_state().outstanding_count()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use frankensearch_core::SearchError;
    use frankensearch_core::canonicalize::DefaultCanonicalizer;

    use super::*;
    use crate::queue::{EmbeddingQueueConfig, EmbeddingRequest};

    fn queue(capacity: usize) -> EmbeddingQueue {
        EmbeddingQueue::new(
            EmbeddingQueueConfig {
                capacity,
                batch_size: 2,
                max_retries: 1,
            },
            Box::new(DefaultCanonicalizer::default()),
        )
    }

    fn submit(queue: &EmbeddingQueue, id: &str, text: &str) {
        queue
            .submit(EmbeddingRequest {
                doc_id: id.to_owned(),
                text: text.to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .expect("submit job");
    }

    fn ids(jobs: &[EmbeddingJob]) -> Vec<&str> {
        jobs.iter().map(|job| job.doc_id.as_str()).collect()
    }

    #[test]
    fn lease_reserves_capacity_and_restores_order_without_retries() {
        let queue = queue(2);
        submit(&queue, "a", "first");
        submit(&queue, "b", "second");
        let batch = queue.lease_batch_up_to(1);
        assert_eq!(batch.len(), 1);
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.in_flight_count(), 1);
        assert_eq!(queue.outstanding_count(), 2);
        let error = queue
            .submit(EmbeddingRequest {
                doc_id: "c".to_owned(),
                text: "third".to_owned(),
                metadata: None,
                submitted_at: Instant::now(),
            })
            .expect_err("reserved slot is unavailable to producers");
        assert!(matches!(
            error,
            SearchError::QueueFull {
                pending: 2,
                capacity: 2
            }
        ));
        assert!(queue.drain_batch().is_empty());
        drop(batch);
        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch();
        assert_eq!(ids(&jobs), ["a", "b"]);
        assert!(jobs.iter().all(|job| job.retry_count == 0));
    }

    #[test]
    fn dropped_suspended_future_returns_its_batch() {
        use std::future::Future;
        use std::task::{Context, Wake, Waker};

        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: Arc<Self>) {}
        }

        let queue = queue(2);
        submit(&queue, "a", "first");
        submit(&queue, "b", "second");
        let mut future = Box::pin(async {
            let batch = queue.lease_batch();
            assert_eq!(batch.len(), 2);
            std::future::pending::<()>().await;
            drop(batch);
        });
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(queue.in_flight_count(), 2);
        assert_eq!(queue.pending_count(), 0);
        drop(future);
        assert_eq!(queue.in_flight_count(), 0);
        let jobs = queue.drain_batch();
        assert_eq!(ids(&jobs), ["a", "b"]);
        assert!(jobs.iter().all(|job| job.retry_count == 0));
    }

    #[test]
    fn newer_pending_submission_wins_over_cancelled_lease() {
        let queue = queue(3);
        submit(&queue, "a", "old first");
        submit(&queue, "b", "second");
        let batch = queue.lease_batch();
        submit(&queue, "a", "new first");
        assert_eq!(queue.outstanding_count(), 3);
        drop(batch);
        let jobs = queue.drain_batch();
        assert_eq!(ids(&jobs), ["b", "a"]);
        assert!(jobs[1].canonical_text.contains("new first"));
        assert!(jobs.iter().all(|job| job.retry_count == 0));
    }

    #[test]
    fn acknowledgement_releases_only_published_jobs() {
        let queue = queue(2);
        submit(&queue, "a", "first");
        submit(&queue, "b", "second");
        let batch = queue.lease_batch();
        queue.record_embedded(&batch.jobs()[0].doc_id, &batch.jobs()[0].content_hash);
        assert!(!batch.is_unacknowledged(0));
        assert!(batch.is_unacknowledged(1));
        assert_eq!(queue.in_flight_count(), 1);
        submit(&queue, "c", "third");
        drop(batch);
        let jobs = queue.drain_batch();
        assert_eq!(ids(&jobs), ["b", "c"]);
        assert_eq!(queue.metrics().total_succeeded.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_and_retried_jobs_are_not_restored_twice() {
        let queue = queue(1);
        submit(&queue, "a", "first");
        let batch = queue.lease_batch();
        assert_eq!(queue.requeue(batch.jobs()[0].clone()), JobOutcome::Retryable);
        assert_eq!(queue.in_flight_count(), 0);
        assert_eq!(queue.pending_count(), 1);
        drop(batch);
        let batch = queue.lease_batch();
        assert_eq!(batch.jobs()[0].retry_count, 1);
        assert_eq!(queue.requeue(batch.jobs()[0].clone()), JobOutcome::Failed);
        drop(batch);
        assert_eq!(queue.outstanding_count(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn an_empty_competing_lease_cannot_release_the_active_one() {
        let queue = queue(2);
        submit(&queue, "a", "first");
        let first = queue.lease_batch();
        submit(&queue, "b", "second");
        let competing = queue.lease_batch();
        assert!(competing.is_empty());
        drop(competing);
        assert_eq!(queue.in_flight_count(), 1);
        assert!(queue.drain_batch().is_empty());
        drop(first);
        assert_eq!(queue.pending_count(), 2);
    }

    #[test]
    fn retry_remaining_does_not_charge_already_retried_jobs_again() {
        let queue = queue(2);
        submit(&queue, "a", "first");
        submit(&queue, "b", "second");
        let batch = queue.lease_batch();
        assert_eq!(queue.requeue(batch.jobs()[0].clone()), JobOutcome::Retryable);
        assert_eq!(batch.retry_unacknowledged(), 0);
        assert_eq!(queue.metrics().total_retryable.load(Ordering::Relaxed), 2);
        drop(batch);
        let jobs = queue.drain_batch();
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|job| job.retry_count == 1));
    }

    #[test]
    fn cancelled_lease_does_not_lose_restoration_of_old_content() {
        let queue = queue(2);
        submit(&queue, "a", "original");
        let original = queue.drain_batch().pop().expect("original");
        queue.record_embedded(&original.doc_id, &original.content_hash);
        submit(&queue, "a", "replacement");
        let batch = queue.lease_batch();
        submit(&queue, "a", "original");
        drop(batch);
        let jobs = queue.drain_batch();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].content_hash, original.content_hash);
    }
}
