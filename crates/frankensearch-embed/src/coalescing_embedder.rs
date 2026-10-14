//! Async `Embedder` access to the existing bounded batch coalescer.
//!
//! The caller retains and drives [`CoalescingDispatcher`]; construction never
//! spawns a task, thread or runtime. Polling a client future submits once, then
//! waits on an Asupersync oneshot, not a blocking `mpsc::Receiver::recv`.
//! The private FIFO below contains only reply channels, never another copy of
//! source text. Input storage and scheduling remain owned by `BatchCoalescer`.

use std::collections::VecDeque;
use std::fmt;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};

use asupersync::channel::oneshot;
use frankensearch_core::generation::QuantizationFormat;
use frankensearch_core::{
    Cx, Embedder, EmbeddingIdentityBundleV1, IdentityBoundEmbedding, ModelCategory, SearchError,
    SearchFuture, SearchResult,
};

use crate::batch_coalescer::{
    BatchCoalescer, CoalescedBatch, CoalescerConfig, CoalescerLimits, CoalescerUsage, Priority,
};

/// Cloneable async producer handle bound to one caller-selected model contract.
///
/// Pass this handle wherever an `Embedder` is accepted, while its dispatcher is
/// driven independently by the caller. Awaiting a submission without driving
/// the dispatcher cannot perform inference. Dropping one pending client future
/// disconnects only its reply; it never cancels the shared inference context.
/// Queued/in-flight text still consumes admission credit until dispatched or
/// the dispatcher is dropped. These limits do not bound output/model memory.
#[derive(Clone)]
pub struct CoalescingEmbedder {
    shared: Arc<Shared>,
    priority: Priority,
}

/// Sole dispatch owner for a [`CoalescingEmbedder`] and its clones.
///
/// `&mut self` serializes dispatch operations without holding a lock through
/// inference. A caller can schedule [`Self::dispatch_ready`] alongside client
/// tasks; no worker or timer is started implicitly. Dropping this owner closes
/// admission, disconnects pending clients and releases undispatched inputs.
/// Dropping a pending dispatch future abandons that batch, not the whole owner;
/// subsequent explicitly driven work remains possible. Provider-owned blocking
/// work is not preempted by dropping a future.
pub struct CoalescingDispatcher {
    shared: Arc<Shared>,
    provider: Arc<dyn Embedder>,
}

struct Description {
    id: String,
    name: String,
    dimension: usize,
    semantic: bool,
    category: ModelCategory,
    mrl: bool,
    ready: bool,
}

struct Shared {
    queue: BatchCoalescer,
    identity: EmbeddingIdentityBundleV1,
    description: Description,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    // All enqueue/dequeue operations on the private queue hold this lock, so
    // reply position always matches the coalescer's submission order.
    replies: VecDeque<Reply>,
    closed: bool,
}

struct Reply {
    raw: Receiver<SearchResult<Vec<f32>>>,
    bound: oneshot::Sender<SearchResult<IdentityBoundEmbedding>>,
}

struct Work {
    batch: CoalescedBatch,
    replies: Vec<Reply>,
}

impl fmt::Debug for CoalescingEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoalescingEmbedder")
            .field("identity", &self.shared.identity.fingerprint())
            .field("priority", &self.priority)
            .field("usage", &self.usage())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CoalescingDispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoalescingDispatcher")
            .field("identity", &self.shared.identity.fingerprint())
            .field("usage", &self.shared.queue.usage())
            .finish_non_exhaustive()
    }
}

impl CoalescingEmbedder {
    /// Bind an already loaded provider and create its caller-driven dispatcher.
    ///
    /// `expected` must be the trusted in-process contract selected by the
    /// caller, not an identity inferred from a response or a display name.
    /// Every inference is still admitted against it by `embed_and_deliver`.
    /// Metadata is captured once; later provider drift cannot retarget clients.
    /// The initial handle uses interactive priority. This neither opens an index
    /// nor proves that the selected model belongs to a particular index.
    ///
    /// # Errors
    /// Returns cancellation, invalid budgets/storage, or producer mismatch before
    /// admitting any input. No model loading, download or fallback occurs.
    pub fn new(
        cx: &Cx,
        provider: Arc<dyn Embedder>,
        expected: EmbeddingIdentityBundleV1,
        config: CoalescerConfig,
        limits: CoalescerLimits,
    ) -> SearchResult<(Self, CoalescingDispatcher)> {
        checkpoint(cx)?;
        expected.validate()?;
        if expected.storage.quantization != QuantizationFormat::F32
            || !expected.storage.format.starts_with("in-memory-")
            || !matches!(
                expected.storage.endianness.as_str(),
                "native-f32-values" | "native-test-only"
            )
        {
            return Err(invalid(
                "identity",
                "expected an in-memory native f32 contract",
            ));
        }
        check_provider(cx, provider.as_ref(), &expected)?;
        let description = Description {
            id: provider.id().to_owned(),
            name: provider.model_name().to_owned(),
            dimension: usize::try_from(expected.space.dimension)
                .map_err(|_| invalid("identity", "dimension does not fit the platform"))?,
            semantic: provider.is_semantic(),
            category: provider.category(),
            mrl: provider.supports_mrl(),
            ready: provider.is_ready(),
        };
        checkpoint(cx)?;
        check_provider(cx, provider.as_ref(), &expected)?;
        let shared = Arc::new(Shared {
            queue: BatchCoalescer::with_limits(config, limits)?,
            identity: expected,
            description,
            state: Mutex::new(State::default()),
        });
        Ok((
            Self {
                shared: Arc::clone(&shared),
                priority: Priority::Interactive,
            },
            CoalescingDispatcher { shared, provider },
        ))
    }

    /// Another handle to the same queue/producer with a different priority.
    #[must_use]
    pub fn with_priority(&self, priority: Priority) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            priority,
        }
    }

    /// Explicitly close admission for all clones; already queued work can drain.
    /// This is owner control, not the behavior of cancelling one client future.
    pub fn shutdown(&self) {
        self.shared.close();
    }

    /// Queued-plus-in-flight input accounting from the underlying coalescer.
    #[must_use]
    pub fn usage(&self) -> CoalescerUsage {
        self.shared.queue.usage()
    }

    fn submit(
        &self,
        cx: &Cx,
        text: &str,
    ) -> SearchResult<oneshot::Receiver<SearchResult<IdentityBoundEmbedding>>> {
        checkpoint(cx)?;
        // Reject an impossible input before copying the borrowed caller text.
        if text.len() > self.shared.queue.limits().max_outstanding_text_bytes {
            return Err(invalid("input", "text exceeds the entire input budget"));
        }
        let (bound, receiver) = oneshot::channel();
        {
            let mut state = self.shared.lock()?;
            if state.closed {
                return Err(stopped("coalescing admission is closed"));
            }
            // Reserve reply capacity BEFORE queue admission. Once submit succeeds,
            // no allocation is needed to establish its ordered reply mapping.
            state
                .replies
                .try_reserve(1)
                .map_err(|_| invalid("replies", "reply allocation failed"))?;
            let raw = self.shared.queue.submit(text.to_owned(), self.priority);
            match raw.try_recv() {
                Err(TryRecvError::Empty) => state.replies.push_back(Reply { raw, bound }),
                Ok(Err(error)) => return Err(error),
                _ => return Err(invalid("replies", "unexpected result before dispatch")),
            }
        }
        // No post-enqueue checkpoint: the receiving future now owns cancellation
        // of its reply. It cannot report admission as unperformed after enqueue.
        Ok(receiver)
    }
}

impl Embedder for CoalescingEmbedder {
    fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move { self.embed_bound(cx, text).await.map(|output| output.values) })
    }

    fn embed_bound<'a>(
        &'a self,
        cx: &'a Cx,
        text: &'a str,
    ) -> SearchFuture<'a, IdentityBoundEmbedding> {
        Box::pin(async move {
            let mut receiver = self.submit(cx, text)?;
            let outcome = receiver.recv(cx).await;
            // The channel deliberately prefers an available value over cancel.
            // At the embedding boundary, the caller's cancellation must win.
            checkpoint(cx)?;
            let output = outcome.map_err(|_| stopped("coalesced reply was abandoned"))??;
            if output.identity != self.shared.identity {
                return Err(identity_mismatch());
            }
            output.validate()?;
            checkpoint(cx)?;
            Ok(output)
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            checkpoint(cx)?;
            let mut outputs = Vec::with_capacity(texts.len());
            for text in texts {
                outputs.push(self.embed_bound(cx, text).await?);
            }
            checkpoint(cx)?;
            Ok(outputs)
        })
    }

    // A single caller's batch remains sequential; independent concurrent callers
    // are coalesced by the dispatcher. Never bypass our real bound singleton.
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.shared.identity)
    }
    fn dimension(&self) -> usize {
        self.shared.description.dimension
    }
    fn id(&self) -> &str {
        &self.shared.description.id
    }
    fn model_name(&self) -> &str {
        &self.shared.description.name
    }
    fn is_semantic(&self) -> bool {
        self.shared.description.semantic
    }
    fn category(&self) -> ModelCategory {
        self.shared.description.category
    }
    fn supports_mrl(&self) -> bool {
        self.shared.description.mrl
    }
    fn is_ready(&self) -> bool {
        self.shared.description.ready && self.shared.state.lock().is_ok_and(|state| !state.closed)
    }
}

impl Shared {
    fn lock(&self) -> SearchResult<MutexGuard<'_, State>> {
        // Poison can mean queue admission happened without its matching reply.
        // Do not repair that invariant by guessing an association.
        self.state
            .lock()
            .map_err(|_| invalid("state", "coalescing reply state is poisoned"))
    }

    fn take_work(&self) -> SearchResult<Option<Work>> {
        let mut state = self.lock()?;
        let Some(batch) = self.queue.try_form_batch() else {
            return Ok(None);
        };
        let count = batch.len();
        if count > state.replies.len() {
            state.closed = true;
            self.queue.shutdown();
            return Err(invalid(
                "replies",
                "coalesced request/reply inventory differs",
            ));
        }
        let replies = state.replies.drain(..count).collect();
        drop(state);
        Ok(Some(Work { batch, replies }))
    }

    fn close(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        drop(state);
        self.queue.shutdown();
    }

    fn abort_pending(&self) {
        let replies = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = true;
            self.queue.shutdown();
            std::mem::take(&mut state.replies)
        };
        // Channel close can wake arbitrary caller code: no adapter lock is held.
        while let Some(batch) = self.queue.try_form_batch() {
            drop(batch);
        }
        drop(replies);
    }
}

impl CoalescingDispatcher {
    /// Execute at most one batch whose existing scheduling conditions are met.
    ///
    /// Returns the number of discharged request slots (including disconnected
    /// callers), or zero when nothing is ready. All-disconnected batches skip
    /// inference. A partially disconnected batch retains its exact input order.
    /// No caller receives a result before whole-batch identity admission succeeds.
    ///
    /// # Errors
    /// Returns the original dispatcher error and sends a bounded typed failure
    /// to each live client. Failure does not retry, switch models or discard other
    /// queued batches. The owner decides whether to continue or drop the dispatcher.
    pub async fn dispatch_ready(&mut self, cx: &Cx) -> SearchResult<usize> {
        checkpoint(cx)?;
        let Some(work) = self.shared.take_work()? else {
            return Ok(0);
        };
        self.execute(cx, work).await
    }

    async fn execute(&self, cx: &Cx, work: Work) -> SearchResult<usize> {
        let Work { batch, replies } = work;
        let count = replies.len();
        if replies.iter().all(|reply| reply.bound.is_closed()) {
            drop(batch);
            return Ok(count);
        }
        // This is the ONLY writer of the private raw transport. It first proves
        // exact equality of every actual response with our immutable contract.
        // Reattaching that same contract below translates the validated transport;
        // it does not bind unchecked raw provider output to advertised metadata.
        let outcome = batch
            .embed_and_deliver(cx, self.provider.as_ref(), &self.shared.identity)
            .await;
        let outcome = checkpoint(cx).and(outcome);
        let outputs = outcome.and_then(|()| {
            replies
                .iter()
                .map(|reply| {
                    checkpoint(cx)?;
                    let Ok(Ok(values)) = reply.raw.try_recv() else {
                        return Err(invalid(
                            "replies",
                            "validated batch did not deliver its complete result",
                        ));
                    };
                    let output = IdentityBoundEmbedding {
                        values,
                        identity: self.shared.identity.clone(),
                    };
                    output.validate()?;
                    Ok(output)
                })
                .collect::<SearchResult<Vec<_>>>()
        });
        let outputs = checkpoint(cx).and(outputs);
        match outputs {
            Ok(outputs) => {
                // All admission is complete. send_blocking is the oneshot's
                // immediate sync bridge: it never waits for a receiver or polls
                // a runtime. No checkpoint/provider call follows this boundary.
                for (reply, output) in replies.into_iter().zip(outputs) {
                    let _ = reply.bound.send_blocking(Ok(output));
                }
                Ok(count)
            }
            Err(error) => {
                for reply in replies {
                    let _ = reply.bound.send_blocking(Err(public_failure(&error)));
                }
                Err(error)
            }
        }
    }
}

impl Drop for CoalescingDispatcher {
    fn drop(&mut self) {
        self.shared.abort_pending();
    }
}

fn invalid(field: &str, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: format!("coalescing_embedder.{field}"),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

fn stopped(reason: &str) -> SearchError {
    SearchError::Cancelled {
        phase: "coalescing_embedder".to_owned(),
        reason: reason.to_owned(),
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    if cx.checkpoint().is_err() || cx.is_cancel_requested() {
        Err(stopped("coalesced operation cancelled"))
    } else {
        Ok(())
    }
}

fn identity_mismatch() -> SearchError {
    SearchError::UnverifiableRemoteSpace {
        producer: "coalescing_embedder".to_owned(),
        reason: "provider differs from the caller-selected immutable contract".to_owned(),
    }
}

fn check_provider(
    cx: &Cx,
    provider: &dyn Embedder,
    expected: &EmbeddingIdentityBundleV1,
) -> SearchResult<()> {
    let identity = provider.identity();
    checkpoint(cx)?;
    if !identity.is_ok_and(|actual| actual == expected) {
        return Err(identity_mismatch());
    }
    let dimension = provider.dimension();
    checkpoint(cx)?;
    if usize::try_from(expected.space.dimension).ok() != Some(dimension) {
        return Err(identity_mismatch());
    }
    Ok(())
}

fn public_failure(error: &SearchError) -> SearchError {
    match error {
        SearchError::Cancelled { .. } => stopped("coalesced inference cancelled"),
        SearchError::UnverifiableRemoteSpace { .. } => identity_mismatch(),
        SearchError::InvalidConfig { .. } | SearchError::DimensionMismatch { .. } => {
            invalid("response", "coalesced response failed admission")
        }
        _ => SearchError::EmbeddingFailed {
            model: "coalescing_embedder".to_owned(),
            source: "coalesced inference failed; dispatcher retains the detailed error".into(),
        },
    }
}

#[cfg(test)]
mod tests;
