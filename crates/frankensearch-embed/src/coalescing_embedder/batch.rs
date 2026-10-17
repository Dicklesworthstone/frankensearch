//! Completion of one bounded group of the adapter's real bound-single calls.

use std::future::poll_fn;
use std::task::Poll;

use super::{Cx, IdentityBoundEmbedding, SearchFuture, SearchResult, checkpoint, invalid};

/// Keep successes in input order, but return the first observed failure without
/// waiting for pending siblings. All sibling futures are dropped before the
/// error leaves this boundary; no successful prefix escapes and no shared
/// caller context is cancelled. Queue-owned disconnected inputs retain the
/// coalescer's existing credit-until-dispatch/drop accounting.
pub(super) async fn collect(
    cx: &Cx,
    mut futures: Vec<SearchFuture<'_, IdentityBoundEmbedding>>,
) -> SearchResult<Vec<IdentityBoundEmbedding>> {
    checkpoint(cx)?;
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(futures.len())
        .map_err(|_| invalid("batch", "batch result allocation failed"))?;
    slots.resize_with(futures.len(), || None);
    let outcome = poll_fn(|context| {
        if let Err(error) = checkpoint(cx) {
            return Poll::Ready(Err(error));
        }
        let mut pending = false;
        for (future, slot) in futures.iter_mut().zip(&mut slots) {
            if slot.is_some() {
                continue;
            }
            let result = future.as_mut().poll(context);
            if let Err(error) = checkpoint(cx) {
                return Poll::Ready(Err(error));
            }
            match result {
                Poll::Ready(Ok(value)) => *slot = Some(value),
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Pending => pending = true,
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(Ok(()))
        }
    })
    .await;
    // In auto mode a sibling can own the dispatcher guard and in-flight batch.
    // Release those before observing the final cancellation state or returning.
    drop(futures);
    checkpoint(cx)?;
    outcome?;
    slots
        .into_iter()
        .map(|slot| slot.ok_or_else(|| invalid("batch", "completed batch slot is missing")))
        .collect()
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
