//! Completion of one bounded group of the adapter's real bound-single calls.

use std::future::poll_fn;
use std::task::Poll;

use super::{Cx, IdentityBoundEmbedding, SearchFuture, SearchResult, checkpoint, invalid};

/// Largest nonempty ordered prefix fitting both queue ceilings. Use byte
/// lengths (not character counts), and subtract from the remaining budget so
/// arithmetic cannot wrap. Other clients still compete for real queue credit:
/// this is grouping, not an unaccounted reservation or a retry policy.
pub(super) fn prefix_len(
    cx: &Cx,
    texts: &[&str],
    max_rows: usize,
    max_bytes: usize,
) -> SearchResult<usize> {
    checkpoint(cx)?;
    if texts.is_empty() {
        return Ok(0);
    }
    let mut remaining = max_bytes;
    let mut count = 0;
    for text in texts.iter().take(max_rows) {
        checkpoint(cx)?;
        if text.len() > remaining {
            break;
        }
        remaining -= text.len();
        count += 1;
    }
    if count == 0 {
        return Err(invalid(
            "batch",
            "input cannot fit a nonempty bounded group",
        ));
    }
    checkpoint(cx)?;
    Ok(count)
}

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

#[cfg(test)]
#[path = "batch_limits_tests.rs"]
mod limits_tests;
