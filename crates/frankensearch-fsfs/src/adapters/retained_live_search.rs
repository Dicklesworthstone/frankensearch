//! Progressive live queries over immutable complete generations.
//!
//! The ordinary retained reader owns retrieval, fusion, refinement and producer
//! admission. This adapter only schedules refreshes and publishes result changes.
//! One query's phases always use one reader, even if CURRENT changes in its sink.
//! No runtime, index writer, persistent cache or background task is created here.

use std::time::Instant;

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::live_search::{LiveSearchConfig, LiveSearchFrame, LiveSearchHit, LiveSearchTracker};
use super::live_search_session::LiveSearchRefreshConfig;
use crate::generation_store::{CompleteGenerationStore, PublishedGeneration};
use crate::output_schema::{SearchOutputPhase, SearchPayload};
use crate::runtime::{FsfsRuntime, RetainedSearchReader};

/// A progressive update with an explicit physical generation receipt.
///
/// `update.generation` is a *result revision*: generation ID, inventory digest,
/// and phase. This allows Initial and Refined to update the same visible window
/// without pretending that refinement published a new on-disk generation.
/// Apply each delta atomically. Reconnect with a new session/full snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetainedLiveSearchFrame {
    pub generation_id: String,
    pub manifest_sha256: String,
    pub phase: SearchOutputPhase,
    /// All ordinary [`SearchPayload`] fields except `hits`. This is not a standalone
    /// `SearchPayload`: the result window is delivered by `update` instead.
    /// Model identity, skip/failure reasons, rerank and explanation evidence are
    /// preserved without introducing a second ranking-policy representation.
    pub annotations: Value,
    /// None on `RefinementFailed`: the delivered Initial window remains valid.
    /// Hit items preserve the ordinary hit fields except rank/score, which are
    /// carried by [`LiveSearchFrame`] itself (and must not defeat score thresholds).
    pub update: Option<LiveSearchFrame<Value>>,
}

struct PendingGeneration {
    generation: PublishedGeneration,
    first_change: Instant,
    last_change: Instant,
}

/// Caller-driven progressive hybrid subscription, including configured fallback.
///
/// Query-resource admission waits until the debounce expires; continuous
/// publication is bounded by `max_wait`. Unchanged polls use a bounded probe.
/// Model state is shared with the supplied runtime. The original retained
/// reader remains usable until replacement admission succeeds.
///
/// Once query delivery starts, an error or a dropped future closes the session.
/// Some phases may already have reached the subscriber: automatically retrying
/// them would rewind or duplicate that subscriber's result history.
pub struct RetainedLiveSearchSession {
    runtime: FsfsRuntime,
    store: CompleteGenerationStore,
    query: String,
    limit: usize,
    refresh: LiveSearchRefreshConfig,
    tracker: LiveSearchTracker<Value>,
    reader: Option<RetainedSearchReader>,
    completed: Option<PublishedGeneration>,
    pending: Option<PendingGeneration>,
    last_poll: Option<Instant>,
    delivery_incomplete: bool,
}

impl std::fmt::Debug for RetainedLiveSearchSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedLiveSearchSession")
            .field("generation", &self.tracker.generation())
            .field("sequence", &self.tracker.sequence())
            .field("pending", &self.pending.is_some())
            .field("delivery_incomplete", &self.delivery_incomplete)
            .finish_non_exhaustive()
    }
}

impl RetainedLiveSearchSession {
    /// Bind a query without opening models, creating files or starting workers.
    ///
    /// # Errors
    /// Rejects invalid result-window, score-threshold and refresh settings.
    pub fn new(
        runtime: FsfsRuntime,
        store: CompleteGenerationStore,
        query: impl Into<String>,
        config: LiveSearchConfig,
        refresh: LiveSearchRefreshConfig,
    ) -> SearchResult<Self> {
        if refresh.max_wait.is_zero() || refresh.max_wait < refresh.debounce {
            return Err(invalid("max_wait must be positive and at least debounce"));
        }
        let query = query.into();
        let tracker = LiveSearchTracker::new(query.clone(), config).map_err(live_error)?;
        Ok(Self {
            runtime,
            store,
            query,
            limit: config.max_results,
            refresh,
            tracker,
            reader: None,
            completed: None,
            pending: None,
            last_poll: None,
            delivery_incomplete: false,
        })
    }

    /// Last delivered window. Failed sinks do not advance this baseline.
    #[must_use]
    pub const fn tracker(&self) -> &LiveSearchTracker<Value> {
        &self.tracker
    }

    /// Whether a failed or abandoned delivery requires a fresh subscription.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.delivery_incomplete
    }

    /// Admit a due generation and deliver Initial before awaiting refinement.
    ///
    /// The sink acknowledges one whole frame. A transport sink must finish its
    /// serialization, write and flush before returning Ok. Synchronous source
    /// I/O and sink work belong on the caller's blocking-capable command lane.
    /// Returns the number of delivered phases, or zero when no refresh is due.
    /// An initially unselected store waits; loss of an established selection
    /// fails closed, never masquerading as document deletion.
    ///
    /// # Errors
    /// Returns original admission/retrieval/cancellation/sink errors, malformed
    /// phase sequences, regressing clocks, or reuse of an incomplete delivery.
    pub async fn poll_with_sink(
        &mut self,
        cx: &Cx,
        now: Instant,
        sink: &mut (dyn FnMut(&RetainedLiveSearchFrame) -> SearchResult<()> + Send),
    ) -> SearchResult<usize> {
        checkpoint(cx)?;
        if self.delivery_incomplete {
            return Err(invalid(
                "delivery did not complete; reconnect with a new subscription",
            ));
        }
        if self.last_poll.is_some_and(|previous| now < previous) {
            return Err(invalid("poll timestamps must be monotone"));
        }
        self.last_poll = Some(now);
        let Some(observed) = self.observe(cx)? else {
            self.pending = None;
            return Ok(0);
        };
        if self.completed.as_ref() == Some(&observed) {
            self.pending = None;
            return Ok(0);
        }
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.generation != observed)
        {
            let first_change = self
                .pending
                .as_ref()
                .map_or(now, |pending| pending.first_change);
            self.pending = Some(PendingGeneration {
                generation: observed.clone(),
                first_change,
                last_change: now,
            });
        }
        let Some(pending) = &self.pending else {
            return Ok(0);
        };
        if self.completed.is_some()
            && !refresh_due(now, pending.first_change, pending.last_change, self.refresh)
        {
            return Ok(0);
        }
        if self
            .reader
            .as_ref()
            .is_none_or(|reader| reader.generation() != &observed)
        {
            // Open exactly once. A newer publication can win this admission;
            // its reader's real receipt, not `observed`, labels every phase.
            // There is no retry loop under sustained publication.
            let replacement = self
                .runtime
                .open_retained_search(cx, self.store.root())
                .await?;
            checkpoint(cx)?;
            self.reader = Some(replacement);
        }
        self.deliver(cx, sink).await
    }

    async fn deliver(
        &mut self,
        cx: &Cx,
        sink: &mut (dyn FnMut(&RetainedLiveSearchFrame) -> SearchResult<()> + Send),
    ) -> SearchResult<usize> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| invalid("reader admission returned no reader"))?;
        let generation = reader.generation().clone();
        if self.completed.as_ref() == Some(&generation) {
            self.pending = None;
            return Ok(0);
        }
        // Arm before constructing/polling the query future. Dropping at any
        // await cannot authorize a replay after externally visible Initial.
        self.delivery_incomplete = true;
        let count = {
            let mut delivery = PhaseDelivery::new(&mut self.tracker, &generation, &self.query);
            {
                let mut phase_sink = |payload: &SearchPayload| delivery.publish(cx, payload, sink);
                let _ = reader
                    .search_with_phase_sink(cx, &self.query, self.limit, &mut phase_sink)
                    .await?;
            }
            checkpoint(cx)?;
            if delivery.last_phase.is_none() {
                return Err(invalid("search completed without delivering Initial"));
            }
            delivery.count
        };
        self.completed = Some(generation);
        self.pending = None;
        self.delivery_incomplete = false;
        Ok(count)
    }

    fn observe(&self, cx: &Cx) -> SearchResult<Option<PublishedGeneration>> {
        let known = self
            .pending
            .as_ref()
            .map(|pending| &pending.generation)
            .or(self.completed.as_ref());
        if let Some(known) = known
            && self.store.is_selected(cx, known)?
        {
            return Ok(Some(known.clone()));
        }
        let observed = self.store.active(cx)?;
        checkpoint(cx)?;
        if observed.is_none() && self.tracker.sequence() != 0 {
            return Err(invalid(
                "the complete-generation selection disappeared; no stale fallback is permitted",
            ));
        }
        Ok(observed)
    }
}

fn refresh_due(
    now: Instant,
    first: Instant,
    last: Instant,
    config: LiveSearchRefreshConfig,
) -> bool {
    now.saturating_duration_since(last) >= config.debounce
        || now.saturating_duration_since(first) >= config.max_wait
}

struct PhaseDelivery<'a> {
    tracker: &'a mut LiveSearchTracker<Value>,
    generation: &'a PublishedGeneration,
    query: &'a str,
    last_phase: Option<SearchOutputPhase>,
    count: usize,
}

impl<'a> PhaseDelivery<'a> {
    const fn new(
        tracker: &'a mut LiveSearchTracker<Value>,
        generation: &'a PublishedGeneration,
        query: &'a str,
    ) -> Self {
        Self {
            tracker,
            generation,
            query,
            last_phase: None,
            count: 0,
        }
    }

    fn publish(
        &mut self,
        cx: &Cx,
        payload: &SearchPayload,
        sink: &mut (dyn FnMut(&RetainedLiveSearchFrame) -> SearchResult<()> + Send),
    ) -> SearchResult<()> {
        checkpoint(cx)?;
        if payload.query != self.query || payload.returned_hits != payload.hits.len() {
            return Err(invalid("phase query or returned-hit accounting changed"));
        }
        if !matches!(
            (self.last_phase, payload.phase),
            (None, SearchOutputPhase::Initial)
                | (
                    Some(SearchOutputPhase::Initial),
                    SearchOutputPhase::Refined | SearchOutputPhase::RefinementFailed
                )
        ) {
            return Err(invalid(
                "expected Initial followed by at most one refinement outcome",
            ));
        }
        let mut candidate = self.tracker.clone();
        let update = if payload.phase == SearchOutputPhase::RefinementFailed {
            None
        } else {
            let hits = payload
                .hits
                .iter()
                .enumerate()
                .map(|(index, hit)| {
                    if hit.rank != index + 1 {
                        return Err(invalid("phase ranks do not match backend result order"));
                    }
                    let mut item = serde_json::to_value(hit).map_err(json_error)?;
                    let fields = item
                        .as_object_mut()
                        .ok_or_else(|| invalid("hit payload is not an object"))?;
                    fields.remove("rank");
                    fields.remove("score");
                    Ok(LiveSearchHit {
                        doc_id: hit.path.clone(),
                        score: hit.score,
                        item,
                    })
                })
                .collect::<SearchResult<Vec<_>>>()?;
            let revision = format!(
                "{}@{}:{}",
                self.generation.id(),
                self.generation.manifest_sha256(),
                payload.phase,
            );
            candidate.apply(&revision, hits).map_err(live_error)?
        };
        let mut annotations = serde_json::to_value(payload).map_err(json_error)?;
        annotations
            .as_object_mut()
            .ok_or_else(|| invalid("search payload is not an object"))?
            .remove("hits");
        let frame = RetainedLiveSearchFrame {
            generation_id: self.generation.id().to_owned(),
            manifest_sha256: self.generation.manifest_sha256().to_owned(),
            phase: payload.phase,
            annotations,
            update,
        };
        sink(&frame)?;
        // No cancellation check after acknowledgment: these bytes are already
        // externally visible. The next query checkpoint can stop further work.
        *self.tracker = candidate;
        self.last_phase = Some(payload.phase);
        self.count += 1;
        Ok(())
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "fsfs.live_search.retained".to_owned(),
        reason: error.to_string(),
    })
}

fn invalid(reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: "live_search.retained".to_owned(),
        value: String::new(),
        reason: reason.to_owned(),
    }
}

fn live_error(source: super::live_search::LiveSearchError) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.live_search",
        source: Box::new(source),
    }
}

fn json_error(source: serde_json::Error) -> SearchError {
    SearchError::SubsystemError {
        subsystem: "fsfs.live_search.payload",
        source: Box::new(source),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::time::Duration;

    use asupersync::test_utils::run_test_with_cx;

    use super::*;
    use crate::FsfsConfig;
    use crate::adapters::live_search::{LiveSearchChange, LiveSearchEvent};
    use crate::generation_store::GenerationPublication;
    use crate::output_schema::SearchHitPayload;

    fn publish(cx: &Cx, store: &CompleteGenerationStore, value: &str) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        std::fs::write(build.path().join("fixture"), value).unwrap();
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("fixture publication did not reach durability");
        };
        generation
    }

    fn payload(phase: SearchOutputPhase, paths: &[(&str, f64)]) -> SearchPayload {
        SearchPayload::new(
            "alpha",
            phase,
            paths.len(),
            paths
                .iter()
                .enumerate()
                .map(|(index, (path, score))| SearchHitPayload {
                    rank: index + 1,
                    path: (*path).to_owned(),
                    line: Some(1),
                    score: *score,
                    snippet: Some("alpha excerpt".to_owned()),
                    lexical_rank: None,
                    semantic_rank: Some(index + 1),
                    hash_rank: None,
                    in_both_sources: false,
                })
                .collect(),
        )
    }

    #[test]
    fn progressive_phases_update_one_window_and_preserve_annotations() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let mut tracker = LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap();
            let mut delivery = PhaseDelivery::new(&mut tracker, &generation, "alpha");
            let mut frames = Vec::new();
            let mut sink = |frame: &RetainedLiveSearchFrame| {
                frames.push(frame.clone());
                Ok(())
            };
            let initial = payload(SearchOutputPhase::Initial, &[("a", 0.7), ("b", 0.3)])
                .with_vector_generation("fixture-space", false);
            delivery.publish(&cx, &initial, &mut sink).unwrap();
            let refined = payload(SearchOutputPhase::Refined, &[("b", 0.9), ("c", 0.6)])
                .with_skip_reason("fixture-policy");
            delivery.publish(&cx, &refined, &mut sink).unwrap();
            assert_eq!(delivery.count, 2);
            assert_eq!(
                frames[0].annotations["vector_generation_id"],
                "fixture-space"
            );
            assert_eq!(frames[1].annotations["skip_reason"], "fixture-policy");
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.annotations.get("hits").is_none())
            );
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.generation_id == generation.id())
            );
            let initial = frames[0].update.as_ref().unwrap();
            let refined = frames[1].update.as_ref().unwrap();
            assert!(matches!(&initial.event, LiveSearchEvent::Snapshot { .. }));
            assert_eq!(
                refined.previous_generation.as_deref(),
                Some(initial.generation.as_str())
            );
            let LiveSearchEvent::Delta { changes } = &refined.event else {
                panic!("refinement must be a delta");
            };
            assert!(changes.iter().any(|change| {
                matches!(change, LiveSearchChange::Removed { doc_id, .. } if doc_id == "a")
            }));
            assert_eq!(tracker.results()[0].hit.doc_id, "b");
            assert_eq!(tracker.results()[1].hit.doc_id, "c");
            assert!(tracker.results()[0].hit.item.get("rank").is_none());
            assert!(tracker.results()[0].hit.item.get("score").is_none());
        });
    }

    #[test]
    fn failed_refinement_keeps_initial_and_preserves_failure_details() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let mut tracker = LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap();
            let mut delivery = PhaseDelivery::new(&mut tracker, &generation, "alpha");
            delivery
                .publish(
                    &cx,
                    &payload(SearchOutputPhase::Initial, &[("a", 1.0)]),
                    &mut |_| Ok(()),
                )
                .unwrap();
            let mut failed = None;
            delivery
                .publish(
                    &cx,
                    &payload(SearchOutputPhase::RefinementFailed, &[])
                        .with_skip_reason("quality_timeout"),
                    &mut |frame| {
                        failed = Some(frame.clone());
                        Ok(())
                    },
                )
                .unwrap();
            let failed = failed.unwrap();
            assert!(failed.update.is_none());
            assert_eq!(failed.annotations["skip_reason"], "quality_timeout");
            assert_eq!(tracker.sequence(), 1);
            assert_eq!(tracker.results()[0].hit.doc_id, "a");
        });
    }

    #[test]
    fn sink_error_and_cancellation_do_not_acknowledge_a_phase() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let mut tracker = LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap();
            let mut delivery = PhaseDelivery::new(&mut tracker, &generation, "alpha");
            let initial = payload(SearchOutputPhase::Initial, &[("a", 1.0)]);
            let error = delivery
                .publish(&cx, &initial, &mut |_| {
                    Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed").into())
                })
                .unwrap_err();
            assert!(matches!(error, SearchError::Io(error)
                if error.kind() == std::io::ErrorKind::BrokenPipe));
            assert_eq!(delivery.tracker.sequence(), 0);
            assert_eq!(delivery.last_phase, None);
            cx.set_cancel_requested(true);
            assert!(matches!(
                delivery.publish(&cx, &initial, &mut |_| panic!(
                    "cancelled phase reached sink"
                )),
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(tracker.sequence(), 0);
        });
    }

    #[test]
    fn malformed_phase_order_and_rank_accounting_are_refused() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let generation = publish(&cx, &store, "one");
            let mut tracker = LiveSearchTracker::new("alpha", LiveSearchConfig::default()).unwrap();
            let mut delivery = PhaseDelivery::new(&mut tracker, &generation, "alpha");
            for phase in [
                SearchOutputPhase::Refined,
                SearchOutputPhase::RefinementFailed,
            ] {
                assert!(
                    delivery
                        .publish(&cx, &payload(phase, &[]), &mut |_| {
                            panic!("out-of-order phase delivered")
                        })
                        .is_err()
                );
            }
            let mut initial = payload(SearchOutputPhase::Initial, &[("a", 1.0)]);
            initial.hits[0].rank = 2;
            assert!(
                delivery
                    .publish(&cx, &initial, &mut |_| panic!("bad rank delivered"))
                    .is_err()
            );
            initial.hits[0].rank = 1;
            initial.returned_hits = 0;
            assert!(
                delivery
                    .publish(&cx, &initial, &mut |_| panic!("bad count delivered"))
                    .is_err()
            );
            assert_eq!(tracker.sequence(), 0);
        });
    }

    #[test]
    fn refresh_deadlines_coalesce_without_starvation() {
        let start = Instant::now();
        let config = LiveSearchRefreshConfig {
            debounce: Duration::from_millis(100),
            max_wait: Duration::from_millis(300),
        };
        assert!(!refresh_due(
            start + Duration::from_millis(99),
            start,
            start,
            config,
        ));
        assert!(refresh_due(
            start + Duration::from_millis(100),
            start,
            start,
            config,
        ));
        assert!(!refresh_due(
            start + Duration::from_millis(299),
            start,
            start + Duration::from_millis(290),
            config,
        ));
        assert!(refresh_due(
            start + Duration::from_millis(300),
            start,
            start + Duration::from_millis(290),
            config,
        ));
    }

    #[test]
    fn empty_store_waits_without_models_and_cancelled_poll_leaves_it_untouched() {
        run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, root.path()).unwrap();
            let mut session = RetainedLiveSearchSession::new(
                FsfsRuntime::new(FsfsConfig::default()),
                store.clone(),
                "alpha",
                LiveSearchConfig::default(),
                LiveSearchRefreshConfig::default(),
            )
            .unwrap();
            assert_eq!(
                session
                    .poll_with_sink(&cx, Instant::now(), &mut |_| {
                        panic!("empty store emitted results")
                    })
                    .await
                    .unwrap(),
                0
            );
            cx.set_cancel_requested(true);
            assert!(matches!(
                session
                    .poll_with_sink(&cx, Instant::now(), &mut |_| Ok(()))
                    .await,
                Err(SearchError::Cancelled { .. })
            ));
            cx.set_cancel_requested(false);
            assert_eq!(session.tracker().sequence(), 0);
            assert!(!session.is_closed());
            assert!(store.active(&cx).unwrap().is_none());
        });
    }
}
