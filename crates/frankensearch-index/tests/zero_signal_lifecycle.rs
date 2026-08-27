//! Restart/state-transition lifecycle for typed zero-signal classification
//! (bd-tqhc): every no-signal state an index passes through — freshly
//! created, populated, all-tombstoned, vacuumed, re-ingested — must carry
//! its typed [`ZeroSignalReason`] across process restarts, and the
//! two-tier availability logging must fire once per state transition, not
//! once per query.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use frankensearch_core::{TwoTierConfig, ZeroSignalReason};
use frankensearch_index::{TwoTierIndex, VectorIndex};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("frankensearch_zero_signal_test")
        .join(format!(
            "{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

const DIM: usize = 4;
const E1: [f32; DIM] = [1.0, 0.0, 0.0, 0.0];
const E2: [f32; DIM] = [0.0, 1.0, 0.0, 0.0];

fn classify(index: &VectorIndex, query: &[f32], k: usize) -> Option<ZeroSignalReason> {
    index
        .search_top_k_classified(query, k, None)
        .expect("classified search")
        .zero_signal
}

// ─── VectorIndex lifecycle across restarts ────────────────────────────────────

#[test]
fn vector_index_lifecycle_classifies_every_state_across_restarts() {
    let dir = temp_dir("lifecycle");
    let path = dir.join("index.fsvi");

    // Stage 1: created, never held a record.
    VectorIndex::create(&path, "test-embedder", DIM)
        .expect("create writer")
        .finish()
        .expect("finish empty index");
    let index = VectorIndex::open(&path).expect("open empty index");
    assert_eq!(
        classify(&index, &E1, 3),
        Some(ZeroSignalReason::NewlyCreatedEmpty)
    );
    // Request-scoped reasons outrank index state on the same empty index.
    assert_eq!(
        classify(&index, &E1, 0),
        Some(ZeroSignalReason::CallerRequestedZeroK)
    );
    assert_eq!(
        classify(&index, &[0.0; DIM], 3),
        Some(ZeroSignalReason::ZeroNormQuery)
    );
    drop(index);

    // Stage 2: ingest through the WAL; hits must clear the classification.
    let mut index = VectorIndex::open(&path).expect("reopen for ingest");
    index.append("doc-a", &E1).expect("append doc-a");
    index.append("doc-b", &E2).expect("append doc-b");
    let classified = index
        .search_top_k_classified(&E1, 3, None)
        .expect("search after ingest");
    assert_eq!(classified.hits.len(), 2);
    assert_eq!(classified.zero_signal, None);
    drop(index);

    // Stage 3: restart — WAL-resident records must survive and still clear.
    let mut index = VectorIndex::open(&path).expect("reopen after restart");
    let classified = index
        .search_top_k_classified(&E1, 3, None)
        .expect("search after restart");
    assert_eq!(classified.hits.len(), 2);
    assert_eq!(classified.zero_signal, None);

    // Stage 4: tombstone everything.
    //
    // Compact first, deliberately. Appended records live in the WAL, and
    // deleting a WAL-resident record REMOVES it rather than tombstoning it,
    // which lands the index in `NewlyCreatedEmpty` and would leave the
    // tombstone classification untested despite this test's name. Compaction
    // moves them into the main index so `soft_delete` produces real
    // tombstones.
    index.compact().expect("compact WAL into the main index");
    assert!(index.soft_delete("doc-a").expect("delete doc-a"));
    assert!(index.soft_delete("doc-b").expect("delete doc-b"));
    let reason = classify(&index, &E1, 3).expect("empty after delete-all");
    assert_eq!(
        reason,
        ZeroSignalReason::AllTombstoned,
        "every record tombstoned must classify as AllTombstoned, not as an empty index"
    );
    assert!(
        !reason.is_availability_failure(),
        "an intentionally emptied index is not an availability failure"
    );
    drop(index);

    // Stage 5: the tombstoned state survives restart.
    let mut index = VectorIndex::open(&path).expect("reopen tombstoned");
    let reason = classify(&index, &E1, 3).expect("still empty after restart");
    assert_eq!(
        reason,
        ZeroSignalReason::AllTombstoned,
        "the tombstoned classification must survive a restart unchanged"
    );

    // Stage 6: vacuum compacts tombstones away; census-wise the index is
    // indistinguishable from a fresh one.
    index.vacuum().expect("vacuum");
    assert_eq!(
        classify(&index, &E1, 3),
        Some(ZeroSignalReason::NewlyCreatedEmpty)
    );

    // Stage 7: re-ingest after vacuum; the lane recovers.
    index.append("doc-c", &E1).expect("append after vacuum");
    let classified = index
        .search_top_k_classified(&E1, 3, None)
        .expect("search after re-ingest");
    assert_eq!(classified.hits.len(), 1);
    assert_eq!(classified.zero_signal, None);
}

// ─── No-warn-storm bound on the two-tier transition machine ──────────────────

/// Captures tracing events whose message mentions the fast-tier semantic
/// lane, so tests can count state-transition logs emitted by
/// `TwoTierIndex::search_fast_classified`.
#[derive(Clone, Default)]
struct LaneEventCollector {
    messages: Arc<Mutex<Vec<String>>>,
}

impl LaneEventCollector {
    fn lane_events(&self) -> Vec<String> {
        self.messages.lock().expect("collector lock").clone()
    }
}

struct MessageExtractor {
    message: Option<String>,
}

impl tracing::field::Visit for MessageExtractor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
        }
    }
}

impl tracing::Subscriber for LaneEventCollector {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut extractor = MessageExtractor { message: None };
        event.record(&mut extractor);
        if let Some(message) = extractor.message
            && message.contains("fast-tier semantic lane")
        {
            self.messages.lock().expect("collector lock").push(message);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Build a two-tier directory whose fast index has every record tombstoned.
fn build_all_tombstoned_dir(name: &str) -> PathBuf {
    let dir = temp_dir(name);
    let fast_path = dir.join("vector.fast.idx");
    let mut writer = VectorIndex::create(&fast_path, "test-embedder", DIM).expect("create writer");
    writer.write_record("doc-a", &E1).expect("write doc-a");
    writer.write_record("doc-b", &E2).expect("write doc-b");
    writer.finish().expect("finish index");
    let mut index = VectorIndex::open(&fast_path).expect("open for tombstoning");
    assert!(index.soft_delete("doc-a").expect("delete doc-a"));
    assert!(index.soft_delete("doc-b").expect("delete doc-b"));
    dir
}

fn open_two_tier(dir: &Path) -> TwoTierIndex {
    TwoTierIndex::open(dir, TwoTierConfig::default()).expect("open two-tier index")
}

#[test]
fn repeated_empty_searches_log_one_transition_not_a_storm() {
    let dir = build_all_tombstoned_dir("storm");
    let index = open_two_tier(&dir);
    let collector = LaneEventCollector::default();

    tracing::subscriber::with_default(collector.clone(), || {
        // Five identical failing searches: the state transition None →
        // AllTombstoned must log exactly once.
        for _ in 0..5 {
            let classified = index
                .search_fast_classified(&E1, 3)
                .expect("classified search");
            assert_eq!(classified.hits, [] as [frankensearch_core::VectorHit; 0]);
            assert_eq!(
                classified.zero_signal,
                Some(ZeroSignalReason::AllTombstoned)
            );
        }

        // A request-scoped event (k = 0) between failing searches must not
        // touch the machine: no new event, no fabricated recovery.
        let classified = index.search_fast_classified(&E1, 0).expect("k = 0 search");
        assert_eq!(
            classified.zero_signal,
            Some(ZeroSignalReason::CallerRequestedZeroK)
        );
        let classified = index
            .search_fast_classified(&E1, 3)
            .expect("post-k0 search");
        assert_eq!(
            classified.zero_signal,
            Some(ZeroSignalReason::AllTombstoned)
        );
    });

    let events = collector.lane_events();
    assert_eq!(
        events.len(),
        1,
        "seven searches over one unchanged no-signal state must log one \
         transition, got {events:?}"
    );
    assert!(
        events[0].contains("no signal"),
        "AllTombstoned is a benign state, not an availability failure: {events:?}"
    );
}

#[test]
fn each_reopened_generation_logs_its_own_transition_once() {
    let dir = build_all_tombstoned_dir("regen");
    let collector = LaneEventCollector::default();

    tracing::subscriber::with_default(collector.clone(), || {
        // Two generations (fresh opens) of the same on-disk state: the
        // once-per-transition bound is per generation, so each logs once —
        // and repeats within a generation still stay silent.
        for _ in 0..2 {
            let index = open_two_tier(&dir);
            for _ in 0..3 {
                let classified = index
                    .search_fast_classified(&E1, 3)
                    .expect("classified search");
                assert_eq!(
                    classified.zero_signal,
                    Some(ZeroSignalReason::AllTombstoned)
                );
            }
        }
    });

    let events = collector.lane_events();
    assert_eq!(
        events.len(),
        2,
        "two generations x three searches must log exactly one transition \
         each, got {events:?}"
    );
}
