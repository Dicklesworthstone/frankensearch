//! Committed-generation refresh scheduling for live search (bd-1074).
//!
//! The host drives `poll` from its watcher/scheduler. No sleeping, detached task,
//! secondary runtime, or unbounded queue is hidden here. A source admits a pinned
//! snapshot before querying; publication failures never replace the visible
//! baseline. Call these synchronous I/O operations on the host's blocking lane.

use std::fmt;
use std::io::Write;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use serde::Serialize;

use super::live_search::{
    LiveSearchConfig, LiveSearchError, LiveSearchFrame, LiveSearchHit, LiveSearchTracker,
};
use crate::generation_store::{CompleteGenerationStore, PublishedGeneration};

/// An immutable backend snapshot and its opaque committed-generation identity.
#[derive(Debug, Clone)]
pub struct CommittedLiveSearchSnapshot<S> {
    pub generation: String,
    pub snapshot: S,
}

/// Backend boundary for committed, coherent live-query refreshes.
pub trait LiveSearchSource {
    type Snapshot;
    type Item: Clone + PartialEq;

    /// Admit the currently selected committed generation, or report no selection.
    /// Raw file notifications are not committed snapshots.
    ///
    /// # Errors
    /// Surface admission, corruption, I/O, and cancellation failures, not an
    /// empty result set. An unavailable source must not look like mass deletion.
    fn snapshot(
        &mut self,
        cx: &Cx,
    ) -> SearchResult<Option<CommittedLiveSearchSnapshot<Self::Snapshot>>>;

    /// Query exactly the supplied pinned snapshot, preserving backend rank order.
    /// Do not resolve a changing CURRENT pointer again while opening its artifacts.
    ///
    /// # Errors
    /// Return the original query/cancellation error. A failed refresh is retriable
    /// without losing the last successfully published result window.
    fn search(
        &mut self,
        cx: &Cx,
        snapshot: &Self::Snapshot,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LiveSearchHit<Self::Item>>>;
}

/// Coalesce rapid publication changes without starving a continuously busy index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveSearchRefreshConfig {
    /// Quiet period after the most recently observed generation change.
    pub debounce: Duration,
    /// Maximum wait from the first pending change, even under sustained churn.
    pub max_wait: Duration,
}

impl Default for LiveSearchRefreshConfig {
    fn default() -> Self {
        Self {
            debounce: Duration::from_millis(100),
            max_wait: Duration::from_secs(1),
        }
    }
}

struct PendingRefresh<S> {
    observed: CommittedLiveSearchSnapshot<S>,
    first_change: Instant,
    last_change: Instant,
}

struct ReadyRefresh<T> {
    generation: String,
    hits: Vec<LiveSearchHit<T>>,
}

/// One long-lived query. At most one pending snapshot is retained.
///
/// The initial snapshot is immediate. Later refreshes coalesce to the newest
/// observed generation, and output backpressure naturally stops new queries.
/// Failed queries remain pending; invalid/cancelled/output-failed refreshes
/// never advance the subscriber's generation or result baseline.
pub struct LiveSearchSession<S: LiveSearchSource> {
    source: S,
    query: String,
    limit: usize,
    tracker: LiveSearchTracker<S::Item>,
    refresh: LiveSearchRefreshConfig,
    pending: Option<PendingRefresh<S::Snapshot>>,
    last_poll: Option<Instant>,
}

impl<S: LiveSearchSource> fmt::Debug for LiveSearchSession<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LiveSearchSession")
            .field("query", &self.query)
            .field("generation", &self.tracker.generation())
            .field("sequence", &self.tracker.sequence())
            .field("pending", &self.pending.is_some())
            .finish_non_exhaustive()
    }
}

impl<S: LiveSearchSource> LiveSearchSession<S> {
    /// Construct a subscription without opening a snapshot or starting workers.
    ///
    /// # Errors
    /// Rejects invalid tracker settings and a zero/shorter-than-debounce max wait.
    pub fn new(
        query: impl Into<String>,
        config: LiveSearchConfig,
        refresh: LiveSearchRefreshConfig,
        source: S,
    ) -> SearchResult<Self> {
        if refresh.max_wait.is_zero() || refresh.max_wait < refresh.debounce {
            return Err(SearchError::InvalidConfig {
                field: "live_search.max_wait".to_owned(),
                value: format!("{:?}", refresh.max_wait),
                reason: "must be positive and at least the debounce duration".to_owned(),
            });
        }
        let query = query.into();
        let tracker = LiveSearchTracker::new(query.clone(), config).map_err(live_error)?;
        Ok(Self {
            source,
            query,
            limit: config.max_results,
            tracker,
            refresh,
            pending: None,
            last_poll: None,
        })
    }

    /// Backend access for host-owned publication/admission integration.
    pub const fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// The last successfully delivered state; failed refreshes leave it intact.
    #[must_use]
    pub const fn tracker(&self) -> &LiveSearchTracker<S::Item> {
        &self.tracker
    }

    /// Drive one refresh for an in-process consumer, without sleeping.
    ///
    /// # Errors
    /// Returns source, cancellation, validation, or clock-regression errors.
    pub fn poll(
        &mut self,
        cx: &Cx,
        now: Instant,
    ) -> SearchResult<Option<LiveSearchFrame<S::Item>>> {
        let Some(ready) = self.refresh_ready(cx, now)? else {
            return Ok(None);
        };
        let frame = self
            .tracker
            .apply(&ready.generation, ready.hits)
            .map_err(live_error)?;
        self.pending = None;
        Ok(frame)
    }

    /// Drive one refresh and flush its atomic NDJSON frame before acknowledging it.
    ///
    /// # Errors
    /// Returns refresh or output errors. On output errors, stop this transport:
    /// even a failed write/flush can have exposed a partial frame to its reader.
    pub fn poll_ndjson<W: Write>(
        &mut self,
        cx: &Cx,
        now: Instant,
        writer: &mut W,
    ) -> SearchResult<Option<LiveSearchFrame<S::Item>>>
    where
        S::Item: Serialize,
    {
        let Some(ready) = self.refresh_ready(cx, now)? else {
            return Ok(None);
        };
        let frame = self
            .tracker
            .emit_ndjson(&ready.generation, ready.hits, writer)
            .map_err(live_error)?;
        self.pending = None;
        Ok(frame)
    }

    fn refresh_ready(
        &mut self,
        cx: &Cx,
        now: Instant,
    ) -> SearchResult<Option<ReadyRefresh<S::Item>>> {
        checkpoint(cx)?;
        if self.last_poll.is_some_and(|previous| now < previous) {
            return Err(SearchError::InvalidConfig {
                field: "live_search.clock".to_owned(),
                value: "regressed".to_owned(),
                reason: "poll timestamps must be monotone".to_owned(),
            });
        }
        self.last_poll = Some(now);
        let observation = self.source.snapshot(cx)?;
        checkpoint(cx)?;
        let Some(observed) = observation else {
            // No selected generation is not a committed zero-result generation.
            self.pending = None;
            return Ok(None);
        };
        if observed.generation.is_empty() {
            return Err(live_error(LiveSearchError::InvalidSnapshot(
                "source returned an empty generation identity",
            )));
        }
        if self.tracker.generation() == Some(observed.generation.as_str()) {
            // Also handles a pending B being restored to the already visible A.
            self.pending = None;
            return Ok(None);
        }
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.observed.generation != observed.generation)
        {
            let first_change = self
                .pending
                .as_ref()
                .map_or(now, |pending| pending.first_change);
            self.pending = Some(PendingRefresh {
                observed,
                first_change,
                last_change: now,
            });
        }
        let Some(pending) = self.pending.as_ref() else {
            return Ok(None);
        };
        let ready = self.tracker.generation().is_none()
            || now.saturating_duration_since(pending.last_change) >= self.refresh.debounce
            || now.saturating_duration_since(pending.first_change) >= self.refresh.max_wait;
        if !ready {
            return Ok(None);
        }
        let hits = self
            .source
            .search(cx, &pending.observed.snapshot, &self.query, self.limit)?;
        // A source can finish work just as cancellation arrives. Never publish
        // that result after observing cancellation at the operation boundary.
        checkpoint(cx)?;
        Ok(Some(ReadyRefresh {
            generation: pending.observed.generation.clone(),
            hits,
        }))
    }
}

/// Complete-generation store adapter with a host-provided lexical/hybrid query.
///
/// Admission verifies the whole bundle once per selection change. Unchanged
/// generations use the store's bounded selection probe instead of rehashing all
/// artifacts on every poll. The callback receives a pinned `PublishedGeneration`
/// and must open *all* query artifacts under its `path()`.
pub struct CompleteGenerationLiveSearch<F, T> {
    store: CompleteGenerationStore,
    admitted: Option<PublishedGeneration>,
    search: F,
    item: PhantomData<fn() -> T>,
}

impl<F, T> CompleteGenerationLiveSearch<F, T> {
    /// Bind an existing store and query callback; no I/O occurs until polling.
    #[must_use]
    pub const fn new(store: CompleteGenerationStore, search: F) -> Self {
        Self {
            store,
            admitted: None,
            search,
            item: PhantomData,
        }
    }
}

impl<F, T> fmt::Debug for CompleteGenerationLiveSearch<F, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompleteGenerationLiveSearch")
            .field("store", &self.store)
            .field("admitted", &self.admitted)
            .finish_non_exhaustive()
    }
}

impl<F, T> LiveSearchSource for CompleteGenerationLiveSearch<F, T>
where
    T: Clone + PartialEq,
    F: FnMut(&Cx, &PublishedGeneration, &str, usize) -> SearchResult<Vec<LiveSearchHit<T>>>,
{
    type Snapshot = PublishedGeneration;
    type Item = T;

    fn snapshot(
        &mut self,
        cx: &Cx,
    ) -> SearchResult<Option<CommittedLiveSearchSnapshot<PublishedGeneration>>> {
        if let Some(admitted) = &self.admitted
            && self.store.is_selected(cx, admitted)?
        {
            return Ok(Some(committed_snapshot(admitted.clone())));
        }
        let selected = self.store.active(cx)?;
        self.admitted.clone_from(&selected);
        Ok(selected.map(committed_snapshot))
    }

    fn search(
        &mut self,
        cx: &Cx,
        snapshot: &PublishedGeneration,
        query: &str,
        limit: usize,
    ) -> SearchResult<Vec<LiveSearchHit<T>>> {
        (self.search)(cx, snapshot, query, limit)
    }
}

fn committed_snapshot(
    generation: PublishedGeneration,
) -> CommittedLiveSearchSnapshot<PublishedGeneration> {
    CommittedLiveSearchSnapshot {
        // Bind inventory identity too, not just a directory/display name.
        generation: format!("{}@{}", generation.id(), generation.manifest_sha256()),
        snapshot: generation,
    }
}

fn checkpoint(cx: &Cx) -> SearchResult<()> {
    cx.checkpoint().map_err(|error| SearchError::Cancelled {
        phase: "fsfs.live_search.refresh".to_owned(),
        reason: error.to_string(),
    })
}

fn live_error(error: LiveSearchError) -> SearchError {
    match error {
        LiveSearchError::Output(error) => SearchError::Io(error),
        error => SearchError::SubsystemError {
            subsystem: "fsfs.live_search",
            source: Box::new(error),
        },
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::fs;
    use std::io;
    #[cfg(unix)]
    use std::path::Path;

    use asupersync::test_utils::run_test_with_cx;

    use super::*;
    #[cfg(unix)]
    use crate::generation_store::GenerationPublication;

    #[derive(Clone)]
    struct FakeSnapshot {
        generation: String,
        document: String,
    }

    #[derive(Default)]
    struct FakeSource {
        selected: Option<FakeSnapshot>,
        snapshots: usize,
        queries: Vec<(String, usize)>,
        fail_admission: bool,
        fail_search: bool,
        cancel_in_search: bool,
        publish_in_search: Option<FakeSnapshot>,
    }

    impl FakeSource {
        fn select(&mut self, generation: &str, document: &str) {
            self.selected = Some(FakeSnapshot {
                generation: generation.to_owned(),
                document: document.to_owned(),
            });
        }
    }

    impl LiveSearchSource for FakeSource {
        type Snapshot = FakeSnapshot;
        type Item = String;

        fn snapshot(
            &mut self,
            _cx: &Cx,
        ) -> SearchResult<Option<CommittedLiveSearchSnapshot<Self::Snapshot>>> {
            self.snapshots += 1;
            if self.fail_admission {
                return Err(io::Error::other("admission failed").into());
            }
            Ok(self
                .selected
                .clone()
                .map(|snapshot| CommittedLiveSearchSnapshot {
                    generation: snapshot.generation.clone(),
                    snapshot,
                }))
        }

        fn search(
            &mut self,
            cx: &Cx,
            snapshot: &Self::Snapshot,
            query: &str,
            limit: usize,
        ) -> SearchResult<Vec<LiveSearchHit<Self::Item>>> {
            assert_eq!(query, "query");
            self.queries.push((snapshot.generation.clone(), limit));
            if self.fail_search {
                return Err(io::Error::other("query failed").into());
            }
            if self.cancel_in_search {
                cx.set_cancel_requested(true);
            }
            if let Some(next) = self.publish_in_search.take() {
                self.selected = Some(next);
            }
            Ok(vec![LiveSearchHit {
                doc_id: snapshot.document.clone(),
                score: 1.0,
                item: snapshot.document.clone(),
            }])
        }
    }

    fn session(debounce_ms: u64, max_wait_ms: u64) -> LiveSearchSession<FakeSource> {
        let mut source = FakeSource::default();
        source.select("a", "old");
        LiveSearchSession::new(
            "query",
            LiveSearchConfig::default(),
            LiveSearchRefreshConfig {
                debounce: Duration::from_millis(debounce_ms),
                max_wait: Duration::from_millis(max_wait_ms),
            },
            source,
        )
        .unwrap()
    }

    fn tick(start: Instant, milliseconds: u64) -> Instant {
        start + Duration::from_millis(milliseconds)
    }

    #[test]
    fn initial_results_are_immediate_and_unchanged_generations_do_not_requery() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(20, 100);
            assert!(session.poll(&cx, start).unwrap().is_some());
            assert!(session.poll(&cx, tick(start, 10)).unwrap().is_none());
            assert!(session.poll(&cx, tick(start, 200)).unwrap().is_none());
            assert_eq!(session.source.queries, vec![("a".to_owned(), 20)]);
        });
    }

    #[test]
    fn rapid_changes_coalesce_to_the_latest_committed_snapshot() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(20, 100);
            session.poll(&cx, start).unwrap();
            session.source_mut().select("b", "intermediate");
            assert!(session.poll(&cx, tick(start, 10)).unwrap().is_none());
            session.source_mut().select("c", "latest");
            assert!(session.poll(&cx, tick(start, 15)).unwrap().is_none());
            assert!(session.poll(&cx, tick(start, 34)).unwrap().is_none());
            let frame = session.poll(&cx, tick(start, 35)).unwrap().unwrap();
            assert_eq!(frame.generation, "c");
            assert_eq!(session.source.queries.len(), 2);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "latest");
        });
    }

    #[test]
    fn sustained_publication_churn_cannot_starve_refresh() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(20, 50);
            session.poll(&cx, start).unwrap();
            for milliseconds in [10, 20, 30, 40, 50] {
                session
                    .source_mut()
                    .select(&milliseconds.to_string(), "changing");
                assert!(
                    session
                        .poll(&cx, tick(start, milliseconds))
                        .unwrap()
                        .is_none()
                );
            }
            session.source_mut().select("last", "newest");
            let frame = session.poll(&cx, tick(start, 60)).unwrap().unwrap();
            assert_eq!(frame.generation, "last");
            assert_eq!(session.source.queries.len(), 2);
        });
    }

    #[test]
    fn restoring_the_visible_generation_discards_an_unpublished_pending_change() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(20, 100);
            session.poll(&cx, start).unwrap();
            session.source_mut().select("b", "pending");
            session.poll(&cx, tick(start, 10)).unwrap();
            session.source_mut().select("a", "old");
            assert!(session.poll(&cx, tick(start, 50)).unwrap().is_none());
            assert!(session.pending.is_none());
            assert_eq!(session.source.queries.len(), 1);
        });
    }

    #[test]
    fn failed_query_is_retriable_without_losing_the_visible_baseline() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(0, 100);
            session.poll(&cx, start).unwrap();
            session.source_mut().select("b", "new");
            session.source_mut().fail_search = true;
            assert!(session.poll(&cx, tick(start, 1)).is_err());
            assert_eq!(session.tracker().generation(), Some("a"));
            assert_eq!(session.tracker().results()[0].hit.doc_id, "old");
            session.source_mut().fail_search = false;
            let frame = session.poll(&cx, tick(start, 2)).unwrap().unwrap();
            assert_eq!(frame.sequence, 2);
            assert_eq!(frame.generation, "b");
        });
    }

    #[test]
    fn absence_of_a_selection_is_not_a_mass_deletion() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(0, 100);
            session.source_mut().selected = None;
            assert!(session.poll(&cx, start).unwrap().is_none());
            assert!(session.source.queries.is_empty());
            session.source_mut().select("a", "old");
            session.poll(&cx, tick(start, 1)).unwrap();
            session.source_mut().selected = None;
            assert!(session.poll(&cx, tick(start, 2)).unwrap().is_none());
            assert_eq!(session.tracker().results()[0].hit.doc_id, "old");
        });
    }

    #[test]
    fn admission_failure_does_not_query_or_erase_the_previous_results() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(0, 100);
            session.poll(&cx, start).unwrap();
            session.source_mut().select("b", "new");
            session.source_mut().fail_admission = true;
            assert!(session.poll(&cx, tick(start, 1)).is_err());
            assert_eq!(session.source.queries.len(), 1);
            assert_eq!(session.tracker().generation(), Some("a"));
            assert_eq!(session.tracker().results()[0].hit.doc_id, "old");
            session.source_mut().fail_admission = false;
            assert_eq!(
                session
                    .poll(&cx, tick(start, 2))
                    .unwrap()
                    .unwrap()
                    .generation,
                "b"
            );
        });
    }

    struct BrokenTransport;

    impl Write for BrokenTransport {
        fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn failed_output_keeps_the_pending_generation_and_subscriber_baseline() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(0, 100);
            session.poll(&cx, start).unwrap();
            session.source_mut().select("b", "new");
            assert!(matches!(
                session.poll_ndjson(&cx, tick(start, 1), &mut BrokenTransport),
                Err(SearchError::Io(_))
            ));
            assert_eq!(session.tracker().generation(), Some("a"));
            assert_eq!(session.tracker().sequence(), 1);
            assert_eq!(session.tracker().results()[0].hit.doc_id, "old");
            assert_eq!(session.pending.as_ref().unwrap().observed.generation, "b");
        });
    }

    #[test]
    fn cancellation_before_poll_does_not_even_open_a_snapshot() {
        run_test_with_cx(|cx| async move {
            let mut session = session(0, 100);
            cx.set_cancel_requested(true);
            let result = session.poll(&cx, Instant::now());
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert_eq!(session.source.snapshots, 0);
            assert_eq!(session.tracker().sequence(), 0);
        });
    }

    #[test]
    fn cancellation_arriving_during_search_prevents_publication() {
        run_test_with_cx(|cx| async move {
            let mut session = session(0, 100);
            session.source_mut().cancel_in_search = true;
            let mut bytes = Vec::new();
            let result = session.poll_ndjson(&cx, Instant::now(), &mut bytes);
            cx.set_cancel_requested(false);
            assert!(matches!(result, Err(SearchError::Cancelled { .. })));
            assert!(bytes.is_empty());
            assert_eq!(session.tracker().sequence(), 0);
            assert!(session.pending.is_some());
        });
    }

    #[test]
    fn publication_during_search_cannot_mix_the_pinned_snapshot_and_identity() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(0, 100);
            session.source_mut().publish_in_search = Some(FakeSnapshot {
                generation: "b".to_owned(),
                document: "new".to_owned(),
            });
            let first = session.poll(&cx, start).unwrap().unwrap();
            assert_eq!(first.generation, "a");
            assert_eq!(session.tracker().results()[0].hit.doc_id, "old");
            let second = session.poll(&cx, tick(start, 1)).unwrap().unwrap();
            assert_eq!(second.generation, "b");
            assert_eq!(session.tracker().results()[0].hit.doc_id, "new");
        });
    }

    #[test]
    fn clock_regression_is_rejected_before_query_or_baseline_mutation() {
        run_test_with_cx(|cx| async move {
            let start = Instant::now();
            let mut session = session(0, 100);
            session.poll(&cx, tick(start, 10)).unwrap();
            session.source_mut().select("b", "new");
            assert!(matches!(
                session.poll(&cx, start),
                Err(SearchError::InvalidConfig { .. })
            ));
            assert_eq!(session.source.queries.len(), 1);
            assert_eq!(session.tracker().generation(), Some("a"));
        });
    }

    #[test]
    fn refresh_configuration_requires_a_bounded_non_starving_deadline() {
        for (debounce, max_wait) in [(0, 0), (10, 5)] {
            let result = LiveSearchSession::new(
                "query",
                LiveSearchConfig::default(),
                LiveSearchRefreshConfig {
                    debounce: Duration::from_millis(debounce),
                    max_wait: Duration::from_millis(max_wait),
                },
                FakeSource::default(),
            );
            assert!(matches!(result, Err(SearchError::InvalidConfig { .. })));
        }
    }

    #[cfg(unix)]
    fn write_bundle(path: &Path, value: &str) {
        fs::create_dir(path.join("lexical")).unwrap();
        for relative in [
            "lexical/MANIFEST",
            "vector.idx",
            "catalog.db",
            "content.txt",
        ] {
            fs::write(path.join(relative), value).unwrap();
        }
    }

    #[cfg(unix)]
    fn publish(store: &CompleteGenerationStore, cx: &Cx, value: &str) -> PublishedGeneration {
        let build = store.begin(cx).unwrap();
        write_bundle(build.path(), value);
        let GenerationPublication::Durable(generation) = build.publish(cx, |_, _| Ok(())).unwrap()
        else {
            panic!("test publication did not reach its durability boundary");
        };
        generation
    }

    #[test]
    #[cfg(unix)]
    fn complete_store_source_pins_all_callback_reads_to_one_inventory() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let store = CompleteGenerationStore::create(&cx, directory.path()).unwrap();
            let first = publish(&store, &cx, "old");
            let mut source = CompleteGenerationLiveSearch::new(
                store.clone(),
                |_: &Cx, generation: &PublishedGeneration, _: &str, _: usize| {
                    let item = fs::read_to_string(generation.path().join("content.txt"))?;
                    Ok(vec![LiveSearchHit {
                        doc_id: "document".to_owned(),
                        score: 1.0,
                        item,
                    }])
                },
            );
            let pinned = source.snapshot(&cx).unwrap().unwrap();
            let second = publish(&store, &cx, "new");
            let old = source.search(&cx, &pinned.snapshot, "query", 20).unwrap();
            assert_eq!(old[0].item, "old");
            assert!(pinned.generation.contains(first.manifest_sha256()));
            let current = source.snapshot(&cx).unwrap().unwrap();
            assert_ne!(pinned.generation, current.generation);
            assert_eq!(current.snapshot, second);
            let new = source.search(&cx, &current.snapshot, "query", 20).unwrap();
            assert_eq!(new[0].item, "new");
        });
    }
}
