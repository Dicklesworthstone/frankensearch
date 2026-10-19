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

mod batch;

/// Cloneable async producer handle bound to one caller-selected model contract.
///
/// Pass this handle wherever an `Embedder` is accepted, while its dispatcher is
/// driven independently by the caller. In manual mode, awaiting a submission
/// without driving the dispatcher cannot perform inference. Dropping a manual
/// client disconnects only its reply; it never cancels the dispatch owner's Cx.
/// Queued/in-flight text still consumes admission credit until dispatched or
/// the dispatcher is dropped. These limits do not bound output/model memory.
/// Auto mode differs: its elected caller owns the whole in-flight dispatch;
/// cancelling or dropping that caller can fail other recipients in its batch.
/// Use a manual dispatcher with a service Cx for independent client cancellation.
/// [`CoalescingDispatcher::run`] drives that service without idle polling.
#[derive(Clone)]
pub struct CoalescingEmbedder {
    shared: Arc<Shared>,
    priority: Priority,
    auto_dispatcher: Option<Arc<asupersync::sync::Mutex<CoalescingDispatcher>>>,
}

struct YieldOnce(bool);

impl std::future::Future for YieldOnce {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
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
/// [`Self::run`] supplies continuous, work-conserving dispatch with an idle
/// submission/shutdown wake, using the caller's service context for inference.
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
    // At most one service can wait: its future borrows the sole dispatcher.
    // Take this sender under the FIFO lock, but signal/drop it only outside.
    service_wake: Option<oneshot::Sender<()>>,
}

struct Reply {
    raw: Receiver<SearchResult<Vec<f32>>>,
    bound: oneshot::Sender<SearchResult<IdentityBoundEmbedding>>,
}

struct Work {
    batch: CoalescedBatch,
    replies: Vec<Reply>,
}

/// Remove an idle registration when its owning service wait ends or is dropped.
/// Only the sole dispatcher installs registrations; a later run cannot begin
/// before the previous run releases its mutable borrow and this guard.
struct ServiceWait<'a>(&'a Shared);

impl Drop for ServiceWait<'_> {
    fn drop(&mut self) {
        let wake = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .service_wake
            .take();
        // The receiving future is dropped before this guard. In particular,
        // never destroy a channel/waker while holding the adapter mutex.
        drop(wake);
    }
}

impl fmt::Debug for CoalescingEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoalescingEmbedder")
            .field("identity", &self.shared.identity.fingerprint())
            .field("priority", &self.priority)
            .field("auto_dispatch", &self.auto_dispatcher.is_some())
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
                auto_dispatcher: None,
            },
            CoalescingDispatcher { shared, provider },
        ))
    }

    /// Bind an already loaded provider into a self-driving coalescing embedder.
    ///
    /// Concurrent callers of [`Embedder::embed_bound`] yield one executor turn
    /// after admission so same-turn arrivals accumulate in the queue, and then
    /// use work-conserving leader-follower dispatch (zero idle timer delay for
    /// solo queries while batching same-turn and in-flight arrivals).
    /// Explicit bound batches use the same dispatch path, with groups bounded
    /// by both request count and UTF-8 input bytes. An individually oversized
    /// input is refused before any prefix is submitted. Other callers still
    /// compete for the real queue credits; grouping does not bypass admission.
    /// Before admitting a new auto-mode request, fully disconnected queued
    /// batches may be discarded to release their credits. Live or in-flight work
    /// is never reclaimed this way; actual overload still returns `QueueFull`.
    ///
    /// # Errors
    /// Returns the same validation and identity admission errors as [`Self::new`].
    pub fn new_auto(
        cx: &Cx,
        provider: Arc<dyn Embedder>,
        expected: EmbeddingIdentityBundleV1,
        config: CoalescerConfig,
        limits: CoalescerLimits,
    ) -> SearchResult<Self> {
        let (_client, dispatcher) = Self::new(cx, provider, expected, config, limits)?;
        Ok(dispatcher.into_auto_embedder())
    }

    /// Another handle to the same queue/producer with a different priority.
    #[must_use]
    pub fn with_priority(&self, priority: Priority) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            priority,
            auto_dispatcher: self.auto_dispatcher.clone(),
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

    /// Lock-free batch coalescer telemetry counters.
    #[must_use]
    pub fn metrics(&self) -> &Arc<crate::batch_coalescer::CoalescerMetrics> {
        self.shared.queue.metrics()
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
        if self.auto_dispatcher.is_some() {
            // A cancelled burst can leave the queue full before any caller
            // reaches dispatch. Refusing every subsequent submission would
            // then prevent the only dispatch drivers from making progress.
            self.shared.discard_abandoned_front()?;
            checkpoint(cx)?;
        }
        let (bound, receiver) = oneshot::channel();
        let wake = {
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
            state.service_wake.take()
        };
        if let Some(wake) = wake {
            let _ = wake.send_blocking(());
        }
        // No post-enqueue checkpoint: the receiving future now owns cancellation
        // of its reply. It cannot report admission as unperformed after enqueue.
        Ok(receiver)
    }

    async fn embed_bound_auto(
        &self,
        cx: &Cx,
        text: &str,
        dispatcher_mutex: &Arc<asupersync::sync::Mutex<CoalescingDispatcher>>,
    ) -> SearchResult<IdentityBoundEmbedding> {
        let mut receiver = self.submit(cx, text)?;
        // Yield one poll turn so sibling futures polled in the same executor
        // turn also enqueue their requests before the leader forms a batch.
        YieldOnce(false).await;
        let outcome = loop {
            enum Step {
                Received(Result<SearchResult<IdentityBoundEmbedding>, oneshot::RecvError>),
                Locked(asupersync::sync::OwnedMutexGuard<CoalescingDispatcher>),
                LockFailed,
            }
            let step = {
                let mut recv_fut = std::pin::pin!(receiver.recv(cx));
                let mut lock_fut = std::pin::pin!(asupersync::sync::OwnedMutexGuard::lock(
                    Arc::clone(dispatcher_mutex),
                    cx
                ));
                std::future::poll_fn(|task_cx| {
                    if let std::task::Poll::Ready(outcome) = recv_fut.as_mut().poll(task_cx) {
                        return std::task::Poll::Ready(Step::Received(outcome));
                    }
                    match lock_fut.as_mut().poll(task_cx) {
                        std::task::Poll::Ready(Ok(guard)) => {
                            std::task::Poll::Ready(Step::Locked(guard))
                        }
                        std::task::Poll::Ready(Err(_)) => std::task::Poll::Ready(Step::LockFailed),
                        std::task::Poll::Pending => std::task::Poll::Pending,
                    }
                })
                .await
            };
            match step {
                Step::Received(outcome) => break outcome,
                Step::LockFailed => {
                    checkpoint(cx)?;
                    return Err(stopped("coalesced dispatcher lock failed"));
                }
                Step::Locked(mut guard) => {
                    let already_ready = {
                        let mut recv_fut = std::pin::pin!(receiver.recv(cx));
                        let noop = std::task::Waker::noop();
                        let mut probe_cx = std::task::Context::from_waker(noop);
                        match recv_fut.as_mut().poll(&mut probe_cx) {
                            std::task::Poll::Ready(outcome) => Some(outcome),
                            std::task::Poll::Pending => None,
                        }
                    };
                    if let Some(outcome) = already_ready {
                        drop(guard);
                        break outcome;
                    }
                    let dispatched = guard.dispatch_available(cx).await;
                    drop(guard);
                    let ready_after = {
                        let mut recv_fut = std::pin::pin!(receiver.recv(cx));
                        let noop = std::task::Waker::noop();
                        let mut probe_cx = std::task::Context::from_waker(noop);
                        match recv_fut.as_mut().poll(&mut probe_cx) {
                            std::task::Poll::Ready(outcome) => Some(outcome),
                            std::task::Poll::Pending => None,
                        }
                    };
                    if let Some(outcome) = ready_after {
                        break outcome;
                    }
                    checkpoint(cx)?;
                    if matches!(dispatched, Ok(0)) {
                        return Err(stopped("coalesced reply was abandoned"));
                    }
                }
            }
        };
        checkpoint(cx)?;
        let output = outcome.map_err(|_| stopped("coalesced reply was abandoned"))??;
        if output.identity != self.shared.identity {
            return Err(identity_mismatch());
        }
        output.validate()?;
        checkpoint(cx)?;
        Ok(output)
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
        if let Some(dispatcher_mutex) = &self.auto_dispatcher {
            let dispatcher_mutex = Arc::clone(dispatcher_mutex);
            return Box::pin(
                async move { self.embed_bound_auto(cx, text, &dispatcher_mutex).await },
            );
        }
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
            let max_text_bytes = self.shared.queue.limits().max_outstanding_text_bytes;
            // Refuse a known impossible input anywhere in the call before
            // submitting an earlier prefix or running any provider callback.
            for text in texts {
                checkpoint(cx)?;
                if text.len() > max_text_bytes {
                    return Err(invalid("input", "text exceeds the entire input budget"));
                }
            }
            let mut outputs = Vec::new();
            outputs
                .try_reserve_exact(texts.len())
                .map_err(|_| invalid("batch", "batch output allocation failed"))?;
            if self.auto_dispatcher.is_some() {
                let chunk_limit = self
                    .shared
                    .queue
                    .config()
                    .max_batch_size
                    .min(self.shared.queue.limits().max_outstanding_requests)
                    .max(1);
                let mut remaining = texts;
                while !remaining.is_empty() {
                    let count = batch::prefix_len(cx, remaining, chunk_limit, max_text_bytes)?;
                    let (chunk, rest) = remaining.split_at(count);
                    let mut futures = Vec::new();
                    futures
                        .try_reserve_exact(count)
                        .map_err(|_| invalid("batch", "batch future allocation failed"))?;
                    futures.extend(chunk.iter().map(|text| self.embed_bound(cx, text)));
                    // A known error ends the whole call; do not wait for an
                    // unrelated pending provider before returning that error.
                    outputs.extend(batch::collect(cx, futures).await?);
                    remaining = rest;
                }
            } else {
                for text in texts {
                    outputs.push(self.embed_bound(cx, text).await?);
                }
            }
            checkpoint(cx)?;
            Ok(outputs)
        })
    }

    // Auto batching preserves every bound-single override through the same
    // dispatcher. Advertise it so bound-aware indexers do not serialize all
    // requests before the coalescer has a chance to form a group. Manual mode
    // remains conservative and still requires its external dispatch owner.
    fn bound_batch_is_native(&self) -> bool {
        self.auto_dispatcher.is_some()
    }

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
        self.drain_work(&mut state, batch)
    }

    /// Bound reclamation by the batches present at entry, and reclaim only
    /// whole prefixes whose recipients have all disconnected. Keep the FIFO paired
    /// under its existing lock; never drain a live row or an in-flight batch.
    /// Channel destruction and credit release happen after both locks leave.
    fn discard_abandoned_front(&self) -> SearchResult<()> {
        let width = self.queue.config().max_batch_size;
        let batches = self.lock()?.replies.len().div_ceil(width);
        for _ in 0..batches {
            let abandoned = {
                let mut state = self.lock()?;
                if state.closed
                    || state.replies.is_empty()
                    || state
                        .replies
                        .iter()
                        .take(width)
                        .any(|reply| !reply.bound.is_closed())
                {
                    return Ok(());
                }
                let batch = self
                    .queue
                    .form_available_batch()
                    .ok_or_else(|| invalid("replies", "queued replies have no matching inputs"))?;
                self.drain_work(&mut state, batch)?
            };
            drop(abandoned);
        }
        Ok(())
    }

    fn take_available_work(&self) -> SearchResult<Option<Work>> {
        let mut state = self.lock()?;
        let Some(batch) = self.queue.form_available_batch() else {
            return Ok(None);
        };
        self.drain_work(&mut state, batch)
    }

    fn drain_work(
        &self,
        state: &mut MutexGuard<'_, State>,
        batch: CoalescedBatch,
    ) -> SearchResult<Option<Work>> {
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
        Ok(Some(Work { batch, replies }))
    }

    fn close(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        let wake = state.service_wake.take();
        drop(state);
        self.queue.shutdown();
        if let Some(wake) = wake {
            let _ = wake.send_blocking(());
        }
    }

    fn abort_pending(&self) {
        let (replies, wake) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = true;
            self.queue.shutdown();
            (
                std::mem::take(&mut state.replies),
                state.service_wake.take(),
            )
        };
        // Channel close can wake arbitrary caller code: no adapter lock is held.
        while let Some(batch) = self.queue.try_form_batch() {
            drop(batch);
        }
        drop(replies);
        drop(wake);
    }
}

impl CoalescingDispatcher {
    /// Drive a persistent embedding service on the caller's execution lane.
    ///
    /// Poll this future alongside manual clients returned by
    /// [`CoalescingEmbedder::new`] (or `EmbedderStack::into_coalescing`). Inference
    /// uses this service's `cx`; give it a cancellation scope distinct from
    /// recipient scopes so one recipient cannot interrupt its healthy peers.
    /// No task, runtime, thread or timer is created here. Idle service waits
    /// register with submission/shutdown under the FIFO lock, so a request
    /// arriving before or after registration cannot be missed.
    ///
    /// This is explicitly work-conserving: like [`Self::dispatch_available`],
    /// it does not wait for `max_wait_ms` or `min_batch_size`. One cooperative
    /// yield before each bounded batch lets same-turn arrivals coalesce and
    /// prevents ready providers from monopolizing the execution lane.
    ///
    /// [`CoalescingEmbedder::shutdown`] closes admission and wakes idle service;
    /// admitted work drains before success. Return the discharged request count
    /// (including abandoned slots), saturating at `usize::MAX`. Dropping all
    /// clients alone is not shutdown. Cancelling/dropping this service abandons
    /// its in-flight batch, but keeps the dispatcher and queued suffix available
    /// for explicit resumption. Dropping the dispatcher closes those too.
    ///
    /// # Errors
    /// The first dispatch, identity or cancellation error stops the service;
    /// it is not retried or formatted. Earlier delivered results remain valid,
    /// and undispatched requests remain queued for the owner's next decision.
    /// Synchronous provider code and independent provider work are not preempted.
    pub async fn run(&mut self, cx: &Cx) -> SearchResult<usize> {
        let mut discharged = 0_usize;
        while self.wait_for_work(cx).await? {
            YieldOnce(false).await;
            let count = self.dispatch_available(cx).await?;
            if count == 0 {
                return Err(invalid(
                    "service",
                    "pending replies made no dispatch progress",
                ));
            }
            discharged = discharged.saturating_add(count);
        }
        Ok(discharged)
    }

    /// Run until an owner-supplied shutdown signal completes, then drain.
    ///
    /// This supplies graceful control even when the coalescing clients have
    /// been erased to `dyn Embedder`, as in `EmbedderStack::into_coalescing`.
    /// The signal closes admission but does NOT cancel the service context or
    /// drop in-flight inference. Previously admitted work drains through
    /// [`Self::run`]; late submissions fail rather than extending shutdown.
    ///
    /// The signal is caller-owned, need not be `Send` or `'static`, and is
    /// dropped as soon as it completes. It is never polled after completion.
    /// When the service exits first, the unused signal is dropped too. No
    /// task, timer, runtime or new cancellation scope is created. Synchronous
    /// signal/provider callbacks cannot be preempted; a stuck provider still
    /// needs cancellation of the service context, not merely graceful shutdown.
    ///
    /// # Errors
    /// Returns [`Self::run`] errors unchanged. Cancellation takes precedence
    /// over signal completion, including cancellation requested by the signal.
    /// Without an observed shutdown signal, an error does not close admission;
    /// the owner can resume or drop the dispatcher under the existing contract.
    #[allow(clippy::future_not_send)] // Caller-owned local shutdown futures are supported.
    pub async fn run_until(
        &mut self,
        cx: &Cx,
        shutdown: impl std::future::Future<Output = ()>,
    ) -> SearchResult<usize> {
        checkpoint(cx)?;
        let shared = Arc::clone(&self.shared);
        let mut shutdown = Some(Box::pin(shutdown));
        let mut service = std::pin::pin!(self.run(cx));
        std::future::poll_fn(|task_cx| {
            if let Err(error) = checkpoint(cx) {
                return std::task::Poll::Ready(Err(error));
            }
            let requested = shutdown
                .as_mut()
                .is_some_and(|signal| signal.as_mut().poll(task_cx).is_ready());
            if requested {
                drop(shutdown.take());
            }
            // Both polling and dropping a signal can execute arbitrary code.
            if let Err(error) = checkpoint(cx) {
                return std::task::Poll::Ready(Err(error));
            }
            if requested {
                shared.close();
            }
            service.as_mut().poll(task_cx)
        })
        .await
    }

    async fn wait_for_work(&mut self, cx: &Cx) -> SearchResult<bool> {
        loop {
            checkpoint(cx)?;
            let receiver = {
                let mut state = self.shared.lock()?;
                // Shutdown drains admitted work, rather than losing its suffix.
                if !state.replies.is_empty() {
                    return Ok(true);
                }
                if state.closed {
                    return Ok(false);
                }
                if state.service_wake.is_some() {
                    return Err(invalid("service", "another idle registration is active"));
                }
                let (wake, receiver) = oneshot::channel();
                state.service_wake = Some(wake);
                receiver
            };
            let registration = ServiceWait(&self.shared);
            // This binding drops before the registration on cancellation/drop.
            let mut receiver = receiver;
            let outcome = receiver.recv(cx).await;
            drop(receiver);
            drop(registration);
            checkpoint(cx)?;
            outcome.map_err(|_| invalid("service", "idle notification was abandoned"))?;
            // A notification is only a reason to recheck the queue, not evidence
            // of a particular request count or permission to ignore shutdown.
        }
    }

    /// Convert this dispatcher into a self-driving [`CoalescingEmbedder`]
    /// whose concurrent callers automatically elect a leader to dispatch batches.
    #[must_use]
    pub fn into_auto_embedder(self) -> CoalescingEmbedder {
        let shared = Arc::clone(&self.shared);
        CoalescingEmbedder {
            shared,
            priority: Priority::Interactive,
            auto_dispatcher: Some(Arc::new(asupersync::sync::Mutex::new(self))),
        }
    }

    /// Execute at most one batch whose existing scheduling conditions are met.
    ///
    /// Returns the number of discharged request slots (including disconnected
    /// callers), or zero when nothing is ready. All-disconnected batches skip
    /// inference. A partially disconnected batch retains its exact input order.
    /// No caller receives a result before whole-batch identity admission succeeds.
    /// Cancellation of the dispatch Cx wakes and drops pending inference even
    /// when the provider never wakes itself. Synchronous provider code cannot
    /// be preempted, and provider-owned independent work retains its own lifetime.
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

    /// Execute at most one work-conserving batch from all currently queued
    /// requests without waiting for `max_wait_ms` to elapse.
    ///
    /// # Errors
    /// Returns the same dispatcher and admission errors as [`Self::dispatch_ready`].
    pub async fn dispatch_available(&mut self, cx: &Cx) -> SearchResult<usize> {
        checkpoint(cx)?;
        let Some(work) = self.shared.take_available_work()? else {
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
        // Register the dispatch owner's cancellation wake independently of
        // the provider. Before/after checkpoints alone cannot wake a provider
        // that returns Pending forever without consulting its Cx.
        let (cancel_sender, mut cancel_receiver) = oneshot::channel::<()>();
        let outcome = {
            let mut cancelled = std::pin::pin!(cancel_receiver.recv(cx));
            let mut inference = std::pin::pin!(batch.embed_and_deliver(
                cx,
                self.provider.as_ref(),
                &self.shared.identity,
            ));
            std::future::poll_fn(|task_cx| {
                if let Err(error) = checkpoint(cx) {
                    return std::task::Poll::Ready(Err(error));
                }
                // The sender stays alive and never sends: only cancellation
                // can complete this receive while inference is being polled.
                if cancelled.as_mut().poll(task_cx).is_ready() {
                    return std::task::Poll::Ready(Err(stopped("coalesced dispatch cancelled")));
                }
                let outcome = inference.as_mut().poll(task_cx);
                match checkpoint(cx) {
                    Ok(()) => outcome,
                    Err(error) => std::task::Poll::Ready(Err(error)),
                }
            })
            .await
        };
        // Both pinned futures have dropped before public failure delivery.
        // Close the receiver first, avoiding an unnecessary sender-drop wake.
        drop(cancel_receiver);
        drop(cancel_sender);
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

#[cfg(test)]
mod service_tests {
    use std::cell::Cell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    use asupersync::test_utils::run_test_with_cx;

    use super::*;

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        calls: AtomicUsize,
        drops: AtomicUsize,
        fail: AtomicBool,
        gate: Mutex<Option<oneshot::Receiver<()>>>,
    }

    struct InferenceGuard<'a>(&'a AtomicUsize);
    impl Drop for InferenceGuard<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
            panic!("service must preserve admitted bound batch inference")
        }

        fn embed_batch_bound<'a>(
            &'a self,
            cx: &'a Cx,
            texts: &'a [&'a str],
        ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let _guard = InferenceGuard(&self.drops);
                let gate = self.gate.lock().unwrap().take();
                if let Some(mut gate) = gate {
                    gate.recv(cx)
                        .await
                        .map_err(|_| stopped("fixture gate closed"))?;
                }
                if self.fail.load(Ordering::SeqCst) {
                    return Err(invalid("fixture", "service provider failed"));
                }
                Ok(texts
                    .iter()
                    .map(|text| IdentityBoundEmbedding {
                        identity: self.identity.clone(),
                        values: vec![text.parse::<f32>().unwrap(), -0.0],
                    })
                    .collect())
            })
        }

        fn bound_batch_is_native(&self) -> bool {
            true
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn dimension(&self) -> usize {
            2
        }
        fn id(&self) -> &str {
            "service-test"
        }
        fn model_name(&self) -> &str {
            "service-test"
        }
        fn is_semantic(&self) -> bool {
            false
        }
        fn category(&self) -> ModelCategory {
            ModelCategory::HashEmbedder
        }
    }

    fn provider() -> Arc<Provider> {
        Arc::new(Provider {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("service-test", 2),
            calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            fail: AtomicBool::new(false),
            gate: Mutex::new(None),
        })
    }

    fn fixture(cx: &Cx) -> (Arc<Provider>, CoalescingEmbedder, CoalescingDispatcher) {
        let provider = provider();
        let (client, dispatcher) = CoalescingEmbedder::new(
            cx,
            provider.clone(),
            provider.identity.clone(),
            CoalescerConfig {
                max_batch_size: 2,
                // A partial batch must run without waiting for this deadline.
                max_wait_ms: 86_400_000,
                min_batch_size: 2,
                use_priority_lanes: true,
            },
            CoalescerLimits {
                max_outstanding_requests: 8,
                max_outstanding_text_bytes: 128,
            },
        )
        .unwrap();
        (provider, client, dispatcher)
    }

    struct ServiceWake {
        shared: Arc<Shared>,
        calls: AtomicUsize,
    }
    impl Wake for ServiceWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            // A synchronous notification may immediately re-enter client code.
            // Fail instead of hanging if it is invoked under the FIFO mutex.
            assert!(self.shared.state.try_lock().is_ok());
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn wake_for(client: &CoalescingEmbedder) -> (Arc<ServiceWake>, Waker) {
        let wake = Arc::new(ServiceWake {
            shared: Arc::clone(&client.shared),
            calls: AtomicUsize::new(0),
        });
        let waker = Waker::from(Arc::clone(&wake));
        (wake, waker)
    }

    fn pending<F: Future + ?Sized>(future: Pin<&mut F>, waker: &Waker) {
        assert!(future.poll(&mut Context::from_waker(waker)).is_pending());
    }

    // Intentionally !Send: shutdown belongs to the caller, not a spawned task.
    struct StopSignal {
        ready: bool,
        cancel: Option<Cx>,
        cancel_on_drop: Option<Cx>,
        polls: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
    }
    impl Future for StopSignal {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            self.polls.set(self.polls.get() + 1);
            if let Some(cx) = &self.cancel {
                cx.cancel_with(
                    asupersync::CancelKind::User,
                    Some("cancel from stop signal"),
                );
            }
            if self.ready {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }
    }
    impl Drop for StopSignal {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
            if let Some(cx) = &self.cancel_on_drop {
                cx.cancel_with(asupersync::CancelKind::User, Some("stop signal dropped"));
            }
        }
    }

    #[test]
    fn service_sleeps_without_polling_and_wakes_for_submission_and_shutdown() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            let (wake, waker) = wake_for(&client);
            let mut service = Box::pin(dispatcher.run(&cx));
            pending(service.as_mut(), &waker);
            let before = wake.calls.load(Ordering::SeqCst);
            for _ in 0..4 {
                pending(service.as_mut(), &waker);
            }
            assert_eq!(wake.calls.load(Ordering::SeqCst), before);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            assert!(client.shared.lock().unwrap().service_wake.is_some());

            let mut first = client.embed_bound(&cx, "1");
            let mut second = client.embed_bound(&cx, "2");
            pending(first.as_mut(), Waker::noop());
            pending(second.as_mut(), Waker::noop());
            assert!(wake.calls.load(Ordering::SeqCst) > before);
            pending(service.as_mut(), &waker); // Coalesce one executor turn.
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            pending(service.as_mut(), &waker); // Complete batch, then park idle.
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            for (result, expected) in [(first.await, 1.0_f32), (second.await, 2.0)] {
                let output = result.unwrap();
                assert_eq!(output.identity, provider.identity);
                assert_eq!(output.values[0].to_bits(), expected.to_bits());
                assert_eq!(output.values[1].to_bits(), (-0.0_f32).to_bits());
            }
            assert_eq!(client.usage(), CoalescerUsage::default());
            let before = wake.calls.load(Ordering::SeqCst);
            client.shutdown();
            assert!(wake.calls.load(Ordering::SeqCst) > before);
            assert!(matches!(
                service.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Ready(Ok(2))
            ));
            drop(service);
            assert!(client.shared.lock().unwrap().service_wake.is_none());
            assert_eq!(dispatcher.run(&cx).await.unwrap(), 0);
        });
    }

    #[test]
    fn shutdown_drains_multiple_bounded_batches_and_skips_abandoned_inputs() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            let mut first = client.embed_bound(&cx, "must-not-infer");
            let mut second = client.embed_bound(&cx, "also-abandoned");
            pending(first.as_mut(), Waker::noop());
            pending(second.as_mut(), Waker::noop());
            drop(first);
            drop(second);
            let mut live: Vec<_> = ["3", "4", "5"]
                .into_iter()
                .map(|text| client.embed_bound(&cx, text))
                .collect();
            for request in &mut live {
                pending(request.as_mut(), Waker::noop());
            }
            client.shutdown();
            assert_eq!(dispatcher.run(&cx).await.unwrap(), 5);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            for (request, expected) in live.into_iter().zip([3.0_f32, 4.0, 5.0]) {
                assert_eq!(
                    request.await.unwrap().values[0].to_bits(),
                    expected.to_bits()
                );
            }
            assert_eq!(client.usage(), CoalescerUsage::default());
            assert!(!client.is_ready());
        });
    }

    #[test]
    fn cancelling_one_recipient_does_not_interrupt_service_owned_inference() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            let (release, gate) = oneshot::channel();
            *provider.gate.lock().unwrap() = Some(gate);
            let recipient = Cx::for_testing();
            let mut first = client.embed_bound(&recipient, "1");
            let mut second = client.embed_bound(&cx, "2");
            pending(first.as_mut(), Waker::noop());
            pending(second.as_mut(), Waker::noop());
            let (_, waker) = wake_for(&client);
            let mut service = Box::pin(dispatcher.run(&cx));
            pending(service.as_mut(), &waker); // Batch yield.
            pending(service.as_mut(), &waker); // Inference waits at gate.
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            recipient.cancel_with(asupersync::CancelKind::User, Some("recipient left"));
            assert!(matches!(first.await, Err(SearchError::Cancelled { .. })));
            pending(service.as_mut(), &waker);
            assert_eq!(provider.drops.load(Ordering::SeqCst), 0);
            assert!(!cx.is_cancel_requested());
            release.send_blocking(()).unwrap();
            pending(service.as_mut(), &waker); // Healthy peer delivered, service idle.
            let result = second.await.unwrap();
            assert_eq!(result.values[0].to_bits(), 2.0_f32.to_bits());
            assert_eq!(result.identity, provider.identity);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
            client.shutdown();
            assert_eq!(service.await.unwrap(), 2);
            assert_eq!(client.usage(), CoalescerUsage::default());
        });
    }

    #[test]
    fn cancelled_or_dropped_idle_service_clears_registration_and_can_restart() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            let owner = Cx::for_testing();
            let (wake, waker) = wake_for(&client);
            let mut service = Box::pin(dispatcher.run(&owner));
            pending(service.as_mut(), &waker);
            let before = wake.calls.load(Ordering::SeqCst);
            owner.cancel_with(asupersync::CancelKind::User, Some("service cancelled"));
            assert!(wake.calls.load(Ordering::SeqCst) > before);
            assert!(matches!(
                service.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Ready(Err(SearchError::Cancelled { .. }))
            ));
            drop(service);
            assert!(client.shared.lock().unwrap().service_wake.is_none());
            let mut abandoned = Box::pin(dispatcher.run(&cx));
            pending(abandoned.as_mut(), &waker);
            drop(abandoned);
            assert!(client.shared.lock().unwrap().service_wake.is_none());
            assert!(client.is_ready());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
            let mut request = client.embed_bound(&cx, "7");
            pending(request.as_mut(), Waker::noop());
            client.shutdown();
            assert_eq!(dispatcher.run(&cx).await.unwrap(), 1);
            assert_eq!(
                request.await.unwrap().values[0].to_bits(),
                7.0_f32.to_bits()
            );
            // An already cancelled owner cannot turn closed/empty into success.
            assert!(matches!(
                dispatcher.run(&owner).await,
                Err(SearchError::Cancelled { .. })
            ));
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn service_failure_keeps_undispatched_suffix_without_retrying_failed_batch() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            provider.fail.store(true, Ordering::SeqCst);
            let mut first = client.embed_bound(&cx, "1");
            let mut second = client.embed_bound(&cx, "2");
            let mut suffix = client.embed_bound(&cx, "3");
            for request in [&mut first, &mut second, &mut suffix] {
                pending(request.as_mut(), Waker::noop());
            }
            assert!(matches!(
                dispatcher.run(&cx).await,
                Err(SearchError::InvalidConfig { field, .. })
                    if field == "coalescing_embedder.fixture"
            ));
            for request in [first, second] {
                assert!(matches!(
                    request.await,
                    Err(SearchError::InvalidConfig { .. })
                ));
            }
            pending(suffix.as_mut(), Waker::noop());
            assert_eq!(client.usage().outstanding_requests, 1);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert!(client.is_ready());
            provider.fail.store(false, Ordering::SeqCst);
            client.shutdown();
            assert_eq!(dispatcher.run(&cx).await.unwrap(), 1);
            assert_eq!(suffix.await.unwrap().values[0].to_bits(), 3.0_f32.to_bits());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            assert_eq!(client.usage(), CoalescerUsage::default());
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn stack_service_shutdown_rejects_late_work_but_finishes_inflight_inference() {
        run_test_with_cx(|cx| async move {
            let provider = provider();
            let stack = crate::auto_detect::EmbedderStack::from_parts(provider.clone(), None);
            let (stack, mut dispatcher, quality) = stack
                .into_coalescing(
                    &cx,
                    CoalescerConfig {
                        max_batch_size: 2,
                        ..CoalescerConfig::default()
                    },
                    CoalescerLimits::default(),
                )
                .unwrap();
            assert!(quality.is_none());
            let wake = Arc::new(ServiceWake {
                shared: Arc::clone(&dispatcher.shared),
                calls: AtomicUsize::new(0),
            });
            let waker = Waker::from(Arc::clone(&wake));
            let (release, gate) = oneshot::channel();
            *provider.gate.lock().unwrap() = Some(gate);
            let (stop, mut stopping) = oneshot::channel();
            let mut service = Box::pin(dispatcher.run_until(&cx, async {
                stopping.recv(&cx).await.unwrap();
            }));
            pending(service.as_mut(), &waker);
            // Only the stack's public dyn Embedder is available to this client.
            let mut request = stack.fast().embed_bound(&cx, "4");
            pending(request.as_mut(), Waker::noop());
            pending(service.as_mut(), &waker);
            pending(service.as_mut(), &waker);
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            let before = wake.calls.load(Ordering::SeqCst);
            stop.send_blocking(()).unwrap();
            assert!(wake.calls.load(Ordering::SeqCst) > before);
            pending(service.as_mut(), &waker);
            assert!(!stack.fast().is_ready());
            assert!(matches!(
                stack.fast().embed_bound(&cx, "5").await,
                Err(SearchError::Cancelled { .. })
            ));
            assert_eq!(provider.drops.load(Ordering::SeqCst), 0);
            assert!(!cx.is_cancel_requested());
            release.send_blocking(()).unwrap();
            assert_eq!(service.await.unwrap(), 1);
            let output = request.await.unwrap();
            assert_eq!(output.identity, provider.identity);
            assert_eq!(output.values[0].to_bits(), 4.0_f32.to_bits());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
            assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
            assert_eq!(wake.shared.queue.usage(), CoalescerUsage::default());
        });
    }

    #[test]
    fn completed_local_shutdown_is_dropped_once_before_draining_finishes() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            let (release, gate) = oneshot::channel();
            *provider.gate.lock().unwrap() = Some(gate);
            let mut request = client.embed_bound(&cx, "1");
            pending(request.as_mut(), Waker::noop());
            let polls = Rc::new(Cell::new(0));
            let drops = Rc::new(Cell::new(0));
            let signal = StopSignal {
                ready: true,
                cancel: None,
                cancel_on_drop: None,
                polls: Rc::clone(&polls),
                drops: Rc::clone(&drops),
            };
            let mut service = Box::pin(dispatcher.run_until(&cx, signal));
            pending(service.as_mut(), Waker::noop());
            assert_eq!(polls.get(), 1);
            assert_eq!(drops.get(), 1);
            assert!(!client.is_ready());
            for _ in 0..3 {
                pending(service.as_mut(), Waker::noop());
            }
            assert_eq!(polls.get(), 1, "completed signals must not be repolled");
            assert_eq!(provider.drops.load(Ordering::SeqCst), 0);
            release.send_blocking(()).unwrap();
            assert_eq!(service.await.unwrap(), 1);
            assert_eq!(
                request.await.unwrap().values[0].to_bits(),
                1.0_f32.to_bits()
            );
            assert_eq!(drops.get(), 1);
            assert!(!cx.is_cancel_requested());
        });
    }

    #[test]
    fn service_errors_drop_shutdown_without_silently_closing_admission() {
        for mode in [
            "before",
            "signal",
            "drop",
            "pending-signal",
            "waiting",
            "provider",
        ] {
            run_test_with_cx(|cx| async move {
                let (provider, client, mut dispatcher) = fixture(&cx);
                let owner = Cx::for_testing();
                let mut request = None;
                if mode == "provider" {
                    provider.fail.store(true, Ordering::SeqCst);
                    let mut submitted = client.embed_bound(&cx, "1");
                    pending(submitted.as_mut(), Waker::noop());
                    request = Some(submitted);
                }
                if mode == "before" {
                    owner.cancel_with(asupersync::CancelKind::User, Some("before service"));
                }
                let polls = Rc::new(Cell::new(0));
                let drops = Rc::new(Cell::new(0));
                let signal = StopSignal {
                    ready: matches!(mode, "signal" | "drop"),
                    cancel: matches!(mode, "signal" | "pending-signal").then(|| owner.clone()),
                    cancel_on_drop: (mode == "drop").then(|| owner.clone()),
                    polls: Rc::clone(&polls),
                    drops: Rc::clone(&drops),
                };
                let (wake, waker) = wake_for(&client);
                let mut service = Box::pin(dispatcher.run_until(&owner, signal));
                if mode == "waiting" {
                    pending(service.as_mut(), &waker);
                    let before = wake.calls.load(Ordering::SeqCst);
                    owner.cancel_with(asupersync::CancelKind::User, Some("idle service"));
                    assert!(wake.calls.load(Ordering::SeqCst) > before);
                }
                let error = service.await.unwrap_err();
                if mode == "provider" {
                    assert!(matches!(
                        error,
                        SearchError::InvalidConfig { field, .. }
                            if field == "coalescing_embedder.fixture"
                    ));
                    assert!(matches!(
                        request.unwrap().await,
                        Err(SearchError::InvalidConfig { .. })
                    ));
                } else {
                    assert!(matches!(error, SearchError::Cancelled { .. }));
                    assert_eq!(polls.get(), usize::from(mode != "before"));
                }
                assert_eq!(drops.get(), 1);
                assert!(client.is_ready());
                assert!(client.shared.lock().unwrap().service_wake.is_none());
                assert_eq!(client.usage(), CoalescerUsage::default());
                assert!(!cx.is_cancel_requested());
            });
        }
    }

    #[test]
    fn dropping_active_service_releases_only_its_batch_and_preserves_queued_work() {
        run_test_with_cx(|cx| async move {
            let (provider, client, mut dispatcher) = fixture(&cx);
            let (release, gate) = oneshot::channel();
            *provider.gate.lock().unwrap() = Some(gate);
            let mut first = client.embed_bound(&cx, "1");
            let mut second = client.embed_bound(&cx, "2");
            let mut suffix = client.embed_bound(&cx, "3");
            for request in [&mut first, &mut second, &mut suffix] {
                pending(request.as_mut(), Waker::noop());
            }
            let mut service = Box::pin(dispatcher.run(&cx));
            pending(service.as_mut(), Waker::noop());
            pending(service.as_mut(), Waker::noop());
            drop(service);
            drop(release);
            for request in [first, second] {
                assert!(matches!(request.await, Err(SearchError::Cancelled { .. })));
            }
            assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
            assert_eq!(client.usage().outstanding_requests, 1);
            assert!(client.shared.lock().unwrap().service_wake.is_none());
            assert!(!cx.is_cancel_requested());
            client.shutdown();
            assert_eq!(dispatcher.run(&cx).await.unwrap(), 1);
            assert_eq!(suffix.await.unwrap().values[0].to_bits(), 3.0_f32.to_bits());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            assert_eq!(client.usage(), CoalescerUsage::default());
        });
    }
}
