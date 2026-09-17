//! Cancellation-safe, prefix-acknowledged ownership of a planned flush.

/// Retain all unacknowledged actions even when an async caller returns an error
/// or its suspended future is dropped. Acknowledgements are absolute offsets
/// into the original queue, so mixed upsert/delete barriers cannot reorder it.
/// Compact the vector only once, on drop, rather than once per delete barrier.
pub(super) struct FlushProgress<'a, T> {
    pending: &'a mut Vec<T>,
    acknowledged: usize,
}

impl<'a, T> FlushProgress<'a, T> {
    pub(super) const fn new(pending: &'a mut Vec<T>) -> Self {
        Self {
            pending,
            acknowledged: 0,
        }
    }

    pub(super) fn actions(&self) -> &[T] {
        self.pending.as_slice()
    }

    /// Acknowledge the exclusive end of a successfully applied prefix.
    pub(super) fn acknowledge_through(&mut self, end: usize) {
        assert!(end >= self.acknowledged && end <= self.pending.len());
        self.acknowledged = end;
    }
}

impl<T> Drop for FlushProgress<'_, T> {
    fn drop(&mut self) {
        drop(self.pending.drain(..self.acknowledged));
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    use super::FlushProgress;

    #[test]
    fn no_acknowledgement_preserves_the_whole_queue() {
        let mut actions = vec!["upsert-a", "delete-a", "upsert-b"];
        drop(FlushProgress::new(&mut actions));
        assert_eq!(actions, ["upsert-a", "delete-a", "upsert-b"]);
    }

    #[test]
    fn completed_barriers_remove_only_the_acknowledged_prefix() {
        let mut actions = vec!["upsert-a", "delete-a", "upsert-b", "delete-b"];
        {
            let mut progress = FlushProgress::new(&mut actions);
            progress.acknowledge_through(1);
            progress.acknowledge_through(2);
            assert_eq!(progress.actions().len(), 4);
        }
        assert_eq!(actions, ["upsert-b", "delete-b"]);
    }

    #[test]
    fn failed_operation_preserves_itself_and_its_suffix() {
        fn fail_after_barrier(actions: &mut Vec<&str>) -> Result<(), &'static str> {
            let mut progress = FlushProgress::new(actions);
            progress.acknowledge_through(1);
            Err("injected delete failure")
        }
        let mut actions = vec!["published-upsert", "failed-delete", "later-upsert"];
        assert!(fail_after_barrier(&mut actions).is_err());
        assert_eq!(actions, ["failed-delete", "later-upsert"]);
    }

    #[test]
    fn dropped_suspended_future_preserves_unacknowledged_actions() {
        let mut actions = vec!["published-upsert", "pending-delete", "later-upsert"];
        let mut future = Box::pin(async {
            let mut progress = FlushProgress::new(&mut actions);
            progress.acknowledge_through(1);
            std::future::pending::<()>().await;
            progress.acknowledge_through(3);
        });
        let mut cx = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut cx), Poll::Pending));
        drop(future);
        assert_eq!(actions, ["pending-delete", "later-upsert"]);
    }

    #[test]
    fn successful_flush_empties_the_queue() {
        let mut actions = vec![1, 2, 3];
        {
            let mut progress = FlushProgress::new(&mut actions);
            progress.acknowledge_through(progress.actions().len());
        }
        assert!(actions.is_empty());
    }
}
