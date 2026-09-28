//! Generation-aware result updates for a long-lived query (bd-1074).
//!
//! A tracker belongs to one query and one subscriber. Feed it ranked results
//! from a *pinned, committed* index generation, not filesystem notifications.
//! Generation identifiers are opaque: restoring an older generation is a real
//! update, not a stale sequence number. Backend ranking and tie order survive
//! unchanged. State and work are bounded by the configured result window.
//!
//! NDJSON publication advances the baseline only after serialization, writing,
//! and flushing succeed. An output error still terminates that transport: a
//! writer may have accepted a prefix before failing. Reconnect with a fresh
//! tracker/full snapshot rather than appending to a potentially torn frame.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{self, Write};

use serde::{Deserialize, Serialize};

/// Wire schema for committed-generation snapshots and atomic result deltas.
pub const LIVE_SEARCH_SCHEMA_VERSION: &str = "fsfs.stream.live_search.v1";

/// Per-subscriber result-window and notification policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiveSearchConfig {
    /// Maximum number of hits accepted in a snapshot. Oversized input is rejected.
    pub max_results: usize,
    /// Suppress score-only changes no larger than this absolute difference.
    /// Differences are measured from the last *published* score, so gradual
    /// changes eventually cross the threshold. Rank/payload changes always emit.
    pub min_score_delta: f64,
}

impl Default for LiveSearchConfig {
    fn default() -> Self {
        Self {
            max_results: 20,
            min_score_delta: 0.0,
        }
    }
}

/// A backend hit. The input vector, not its scores, determines rank order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSearchHit<T> {
    /// Stable document identity, independent of its current rank or path label.
    pub doc_id: String,
    /// Finite backend score.
    pub score: f64,
    /// Caller-owned display/search payload, such as a path and snippet.
    pub item: T,
}

/// A hit with its one-based rank in the complete visible result window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedLiveSearchHit<T> {
    pub rank: u64,
    #[serde(flatten)]
    pub hit: LiveSearchHit<T>,
}

/// One change within an atomic delta. Apply the whole frame before rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "snake_case")]
pub enum LiveSearchChange<T> {
    Added {
        result: RankedLiveSearchHit<T>,
    },
    Removed {
        doc_id: String,
        previous_rank: u64,
    },
    Updated {
        previous_rank: u64,
        result: RankedLiveSearchHit<T>,
    },
}

/// Initial full state or subsequent changes to that state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum LiveSearchEvent<T> {
    Snapshot {
        results: Vec<RankedLiveSearchHit<T>>,
    },
    Delta {
        changes: Vec<LiveSearchChange<T>>,
    },
}

/// A complete, independently framed committed-generation notification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveSearchFrame<T> {
    pub schema_version: String,
    /// Immutable query text for this subscription.
    pub query: String,
    /// Monotone per-subscription sequence; the initial snapshot is sequence 1.
    pub sequence: u64,
    pub generation: String,
    /// The baseline to which a delta applies; absent on the initial snapshot.
    pub previous_generation: Option<String>,
    pub result_count: usize,
    #[serde(flatten)]
    pub event: LiveSearchEvent<T>,
}

/// Validation or transport failure. None advances a tracker's baseline.
#[derive(Debug)]
pub enum LiveSearchError {
    InvalidConfig(&'static str),
    InvalidSnapshot(&'static str),
    SequenceExhausted,
    Serialization(serde_json::Error),
    Output(io::Error),
}

impl fmt::Display for LiveSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid live-search config: {reason}")
            }
            Self::InvalidSnapshot(reason) => {
                write!(formatter, "invalid live-search snapshot: {reason}")
            }
            Self::SequenceExhausted => formatter.write_str("live-search sequence exhausted"),
            Self::Serialization(error) => write!(formatter, "live-search serialization: {error}"),
            Self::Output(error) => write!(formatter, "live-search output: {error}"),
        }
    }
}

impl std::error::Error for LiveSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(error) => Some(error),
            Self::Output(error) => Some(error),
            _ => None,
        }
    }
}

/// Bounded, deterministic state for one live query/subscriber.
#[derive(Debug, Clone)]
pub struct LiveSearchTracker<T> {
    query: String,
    config: LiveSearchConfig,
    sequence: u64,
    generation: Option<String>,
    results: Vec<RankedLiveSearchHit<T>>,
}

struct PreparedUpdate<T> {
    frame: LiveSearchFrame<T>,
    results: Vec<RankedLiveSearchHit<T>>,
}

impl<T: Clone + PartialEq> LiveSearchTracker<T> {
    /// Start a subscription. Its first accepted generation emits a full snapshot,
    /// including when the result set is empty.
    ///
    /// # Errors
    /// Rejects blank queries, zero result limits, and negative/non-finite thresholds.
    pub fn new(
        query: impl Into<String>,
        config: LiveSearchConfig,
    ) -> Result<Self, LiveSearchError> {
        let query = query.into();
        if query.trim().is_empty() {
            return Err(LiveSearchError::InvalidConfig("query must not be blank"));
        }
        if config.max_results == 0 {
            return Err(LiveSearchError::InvalidConfig(
                "max_results must be positive",
            ));
        }
        if !config.min_score_delta.is_finite() || config.min_score_delta < 0.0 {
            return Err(LiveSearchError::InvalidConfig(
                "min_score_delta must be finite and non-negative",
            ));
        }
        Ok(Self {
            query,
            config,
            sequence: 0,
            generation: None,
            results: Vec::new(),
        })
    }

    /// Last published generation, not the most recently observed filesystem state.
    #[must_use]
    pub fn generation(&self) -> Option<&str> {
        self.generation.as_deref()
    }

    /// Last published sequence (zero before the initial snapshot).
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Current client-visible result window, including suppressed-score baselines.
    #[must_use]
    pub fn results(&self) -> &[RankedLiveSearchHit<T>] {
        &self.results
    }

    /// Publish to an in-process subscriber, such as a TUI model.
    ///
    /// Repeating the last generation is a no-op: committed generations must be
    /// immutable. Every *different* generation produces a frame, even if its
    /// delta is empty, so consumers can track their exact baseline.
    ///
    /// # Errors
    /// Rejects malformed or oversized snapshots and sequence exhaustion without
    /// changing the previous state.
    pub fn apply(
        &mut self,
        generation: &str,
        hits: Vec<LiveSearchHit<T>>,
    ) -> Result<Option<LiveSearchFrame<T>>, LiveSearchError> {
        Ok(self
            .prepare(generation, hits)?
            .map(|prepared| self.commit(prepared)))
    }

    /// Serialize and flush one NDJSON frame before advancing the baseline.
    ///
    /// Synchronous output is intentional backpressure: no unbounded event queue
    /// or detached writer is created. Call from the host's blocking lane when
    /// the writer may block. Stop the transport on any output error.
    ///
    /// # Errors
    /// Returns snapshot, serialization, or I/O errors without advancing state.
    pub fn emit_ndjson<W: Write>(
        &mut self,
        generation: &str,
        hits: Vec<LiveSearchHit<T>>,
        writer: &mut W,
    ) -> Result<Option<LiveSearchFrame<T>>, LiveSearchError>
    where
        T: Serialize,
    {
        let Some(prepared) = self.prepare(generation, hits)? else {
            return Ok(None);
        };
        let mut bytes =
            serde_json::to_vec(&prepared.frame).map_err(LiveSearchError::Serialization)?;
        bytes.push(b'\n');
        writer.write_all(&bytes).map_err(LiveSearchError::Output)?;
        writer.flush().map_err(LiveSearchError::Output)?;
        Ok(Some(self.commit(prepared)))
    }

    fn prepare(
        &self,
        generation: &str,
        hits: Vec<LiveSearchHit<T>>,
    ) -> Result<Option<PreparedUpdate<T>>, LiveSearchError> {
        if generation.is_empty() {
            return Err(LiveSearchError::InvalidSnapshot(
                "generation must not be empty",
            ));
        }
        if self.generation.as_deref() == Some(generation) {
            return Ok(None);
        }
        if hits.len() > self.config.max_results {
            return Err(LiveSearchError::InvalidSnapshot(
                "result window exceeds max_results",
            ));
        }
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or(LiveSearchError::SequenceExhausted)?;
        let mut identities = BTreeSet::new();
        for hit in &hits {
            if hit.doc_id.is_empty() || !hit.score.is_finite() {
                return Err(LiveSearchError::InvalidSnapshot(
                    "hits require a non-empty identity and a finite score",
                ));
            }
            if !identities.insert(hit.doc_id.as_str()) {
                return Err(LiveSearchError::InvalidSnapshot(
                    "duplicate document identity",
                ));
            }
        }
        let previous: BTreeMap<_, _> = self
            .results
            .iter()
            .map(|result| (result.hit.doc_id.as_str(), result))
            .collect();
        let mut changes = Vec::new();
        // Removals follow old rank order; additions/updates follow new rank order.
        for result in &self.results {
            if !identities.contains(result.hit.doc_id.as_str()) {
                changes.push(LiveSearchChange::Removed {
                    doc_id: result.hit.doc_id.clone(),
                    previous_rank: result.rank,
                });
            }
        }
        let mut results = Vec::with_capacity(hits.len());
        for (index, hit) in hits.into_iter().enumerate() {
            let rank = u64::try_from(index + 1)
                .map_err(|_| LiveSearchError::InvalidSnapshot("rank exceeds protocol range"))?;
            let result = RankedLiveSearchHit { rank, hit };
            if let Some(old) = previous.get(result.hit.doc_id.as_str()) {
                let differs = old.rank != result.rank
                    || old.hit.item != result.hit.item
                    || (old.hit.score - result.hit.score).abs() > self.config.min_score_delta;
                if differs {
                    changes.push(LiveSearchChange::Updated {
                        previous_rank: old.rank,
                        result: result.clone(),
                    });
                    results.push(result);
                } else {
                    // Keep the *published* score, not the last observed score.
                    results.push((*old).clone());
                }
            } else {
                if self.generation.is_some() {
                    changes.push(LiveSearchChange::Added {
                        result: result.clone(),
                    });
                }
                results.push(result);
            }
        }
        let event = if self.generation.is_none() {
            LiveSearchEvent::Snapshot {
                results: results.clone(),
            }
        } else {
            LiveSearchEvent::Delta { changes }
        };
        Ok(Some(PreparedUpdate {
            frame: LiveSearchFrame {
                schema_version: LIVE_SEARCH_SCHEMA_VERSION.to_owned(),
                query: self.query.clone(),
                sequence,
                generation: generation.to_owned(),
                previous_generation: self.generation.clone(),
                result_count: results.len(),
                event,
            },
            results,
        }))
    }

    fn commit(&mut self, prepared: PreparedUpdate<T>) -> LiveSearchFrame<T> {
        self.sequence = prepared.frame.sequence;
        self.generation = Some(prepared.frame.generation.clone());
        self.results = prepared.results;
        prepared.frame
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f64) -> LiveSearchHit<String> {
        LiveSearchHit {
            doc_id: id.to_owned(),
            score,
            item: format!("payload:{id}"),
        }
    }

    fn tracker() -> LiveSearchTracker<String> {
        LiveSearchTracker::new("query", LiveSearchConfig::default()).unwrap()
    }

    fn changes(frame: LiveSearchFrame<String>) -> Vec<LiveSearchChange<String>> {
        match frame.event {
            LiveSearchEvent::Delta { changes } => changes,
            LiveSearchEvent::Snapshot { .. } => panic!("expected a delta"),
        }
    }

    #[test]
    fn initial_empty_generation_is_an_observable_snapshot() {
        let mut tracker = tracker();
        let frame = tracker.apply("g1", vec![]).unwrap().unwrap();
        assert_eq!(frame.sequence, 1);
        assert_eq!(frame.previous_generation, None);
        assert_eq!(frame.result_count, 0);
        assert!(matches!(frame.event, LiveSearchEvent::Snapshot { results } if results.is_empty()));
        assert!(tracker.apply("g1", vec![]).unwrap().is_none());
        assert_eq!(tracker.sequence(), 1);
    }

    #[test]
    fn backend_rank_and_tie_order_are_preserved() {
        let mut tracker = tracker();
        tracker
            .apply("g1", vec![hit("z", 1.0), hit("a", 1.0), hit("b", 9.0)])
            .unwrap();
        let ranks: Vec<_> = tracker
            .results()
            .iter()
            .map(|result| (result.hit.doc_id.as_str(), result.rank))
            .collect();
        assert_eq!(ranks, vec![("z", 1), ("a", 2), ("b", 3)]);
    }

    #[test]
    fn deltas_cover_removal_insertion_and_rank_change_in_stable_order() {
        let mut tracker = tracker();
        tracker
            .apply("g1", vec![hit("a", 3.0), hit("b", 2.0), hit("c", 1.0)])
            .unwrap();
        let frame = tracker
            .apply("g2", vec![hit("c", 1.0), hit("d", 0.5)])
            .unwrap()
            .unwrap();
        assert_eq!(frame.previous_generation.as_deref(), Some("g1"));
        assert_eq!(frame.result_count, 2);
        let delta = changes(frame);
        assert!(matches!(
            &delta[0],
            LiveSearchChange::Removed { doc_id, previous_rank: 1 } if doc_id == "a"
        ));
        assert!(matches!(
            &delta[1],
            LiveSearchChange::Removed { doc_id, previous_rank: 2 } if doc_id == "b"
        ));
        assert!(matches!(
            &delta[2],
            LiveSearchChange::Updated { previous_rank: 3, result }
                if result.rank == 1 && result.hit.doc_id == "c"
        ));
        assert!(matches!(
            &delta[3],
            LiveSearchChange::Added { result } if result.rank == 2 && result.hit.doc_id == "d"
        ));
    }

    #[test]
    fn score_threshold_uses_published_baseline_and_accumulates_drift() {
        let mut tracker = LiveSearchTracker::new(
            "query",
            LiveSearchConfig {
                max_results: 5,
                min_score_delta: 0.1,
            },
        )
        .unwrap();
        tracker.apply("g1", vec![hit("a", 1.0)]).unwrap();
        let suppressed = tracker.apply("g2", vec![hit("a", 1.06)]).unwrap().unwrap();
        assert!(changes(suppressed).is_empty());
        assert_eq!(tracker.results()[0].hit.score, 1.0);
        let emitted = tracker.apply("g3", vec![hit("a", 1.12)]).unwrap().unwrap();
        assert_eq!(changes(emitted).len(), 1);
        assert_eq!(tracker.results()[0].hit.score, 1.12);
    }

    #[test]
    fn payload_changes_are_not_hidden_by_score_threshold() {
        let mut tracker = tracker();
        tracker.apply("g1", vec![hit("a", 1.0)]).unwrap();
        let mut updated = hit("a", 1.0);
        updated.item = "new snippet".to_owned();
        let delta = changes(tracker.apply("g2", vec![updated]).unwrap().unwrap());
        assert!(matches!(
            &delta[0],
            LiveSearchChange::Updated { result, .. } if result.hit.item == "new snippet"
        ));
    }

    #[test]
    fn unchanged_results_still_advance_the_generation_boundary() {
        let mut tracker = tracker();
        tracker.apply("g1", vec![hit("a", 1.0)]).unwrap();
        let frame = tracker.apply("g2", vec![hit("a", 1.0)]).unwrap().unwrap();
        assert_eq!(frame.sequence, 2);
        assert!(changes(frame).is_empty());
        assert_eq!(tracker.generation(), Some("g2"));
    }

    #[test]
    fn a_restored_older_generation_is_not_rejected_as_stale() {
        let mut tracker = tracker();
        tracker.apply("z-new", vec![hit("a", 1.0)]).unwrap();
        tracker.apply("a-old", vec![hit("b", 1.0)]).unwrap();
        tracker.apply("z-new", vec![hit("a", 1.0)]).unwrap();
        assert_eq!(tracker.sequence(), 3);
        assert_eq!(tracker.results()[0].hit.doc_id, "a");
    }

    #[test]
    fn malformed_snapshots_leave_the_previous_state_intact() {
        let mut tracker = tracker();
        tracker.apply("g1", vec![hit("a", 1.0)]).unwrap();
        for bad in [
            vec![hit("a", 1.0), hit("a", 2.0)],
            vec![hit("", 1.0)],
            vec![hit("b", f64::NAN)],
            vec![hit("b", f64::INFINITY)],
            vec![hit("b", f64::NEG_INFINITY)],
        ] {
            assert!(tracker.apply("g2", bad).is_err());
            assert_eq!(tracker.sequence(), 1);
            assert_eq!(tracker.generation(), Some("g1"));
            assert_eq!(tracker.results()[0].hit.doc_id, "a");
        }
        assert!(tracker.apply("", vec![]).is_err());
    }

    #[test]
    fn oversized_windows_are_rejected_not_silently_truncated() {
        let mut tracker = LiveSearchTracker::new(
            "query",
            LiveSearchConfig {
                max_results: 1,
                ..LiveSearchConfig::default()
            },
        )
        .unwrap();
        assert!(
            tracker
                .apply("g1", vec![hit("a", 1.0), hit("b", 2.0)])
                .is_err()
        );
        assert_eq!(tracker.sequence(), 0);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        for threshold in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                LiveSearchTracker::<String>::new(
                    "query",
                    LiveSearchConfig {
                        min_score_delta: threshold,
                        ..LiveSearchConfig::default()
                    },
                )
                .is_err()
            );
        }
        assert!(LiveSearchTracker::<String>::new(" \n", LiveSearchConfig::default()).is_err());
        assert!(
            LiveSearchTracker::<String>::new(
                "query",
                LiveSearchConfig {
                    max_results: 0,
                    ..LiveSearchConfig::default()
                },
            )
            .is_err()
        );
    }

    struct FailingWriter {
        fail_flush: bool,
    }

    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.fail_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn failed_write_or_flush_never_advances_the_baseline() {
        for fail_flush in [false, true] {
            let mut tracker = tracker();
            tracker.apply("g1", vec![hit("a", 1.0)]).unwrap();
            let result =
                tracker.emit_ndjson("g2", vec![hit("b", 2.0)], &mut FailingWriter { fail_flush });
            assert!(matches!(result, Err(LiveSearchError::Output(_))));
            assert_eq!(tracker.sequence(), 1);
            assert_eq!(tracker.generation(), Some("g1"));
            assert_eq!(tracker.results()[0].hit.doc_id, "a");
        }
    }

    #[test]
    fn ndjson_frames_round_trip_and_escape_payload_newlines() {
        let mut tracker = tracker();
        let mut bytes = Vec::new();
        let mut item = hit("a", 1.0);
        item.item = "line one\nline two".to_owned();
        let frame = tracker
            .emit_ndjson("g1", vec![item], &mut bytes)
            .unwrap()
            .unwrap();
        assert_eq!(bytes.split(|&byte| byte == b'\n').count(), 2);
        assert_eq!(bytes.last(), Some(&b'\n'));
        let decoded: LiveSearchFrame<String> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, frame);
        let length = bytes.len();
        assert!(
            tracker
                .emit_ndjson("g1", vec![], &mut bytes)
                .unwrap()
                .is_none()
        );
        assert_eq!(bytes.len(), length);
    }

    #[test]
    fn sequence_overflow_cannot_wrap_or_mutate_state() {
        let mut tracker = tracker();
        tracker.sequence = u64::MAX;
        assert!(matches!(
            tracker.apply("g1", vec![]),
            Err(LiveSearchError::SequenceExhausted)
        ));
        assert!(tracker.generation().is_none());
    }

    #[test]
    fn deleting_every_match_publishes_all_removals() {
        let mut tracker = tracker();
        tracker
            .apply("g1", vec![hit("a", 2.0), hit("b", 1.0)])
            .unwrap();
        let frame = tracker.apply("g2", vec![]).unwrap().unwrap();
        assert_eq!(frame.result_count, 0);
        assert_eq!(changes(frame).len(), 2);
        assert!(tracker.results().is_empty());
    }

    #[derive(Debug, Clone, PartialEq)]
    struct CannotSerialize;

    impl Serialize for CannotSerialize {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(
                "intentional serialization failure",
            ))
        }
    }

    #[test]
    fn serialization_failure_writes_nothing_and_preserves_state() {
        let mut tracker = LiveSearchTracker::new("query", LiveSearchConfig::default()).unwrap();
        let mut bytes = Vec::new();
        let result = tracker.emit_ndjson(
            "g1",
            vec![LiveSearchHit {
                doc_id: "a".to_owned(),
                score: 1.0,
                item: CannotSerialize,
            }],
            &mut bytes,
        );
        assert!(matches!(result, Err(LiveSearchError::Serialization(_))));
        assert!(bytes.is_empty());
        assert_eq!(tracker.sequence(), 0);
        assert!(tracker.generation().is_none());
    }
}
