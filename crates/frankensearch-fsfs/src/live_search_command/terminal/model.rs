//! Transactional consumer of the shipping lexical and hybrid live protocols.
//! Rendering never computes rankings or invents a new generation identity.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};

use frankensearch_fsfs::adapters::live_search::{
    LIVE_SEARCH_SCHEMA_VERSION, LiveSearchChange, LiveSearchEvent, LiveSearchFrame,
    RankedLiveSearchHit,
};
use frankensearch_fsfs::adapters::retained_live_search::RetainedLiveSearchFrame;
use frankensearch_fsfs::output_schema::SearchOutputPhase;
use serde::Deserialize;
use serde_json::Value;

const HYBRID_SCHEMA: &str = "fsfs.stream.live_search.hybrid.v1";

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Model {
    pub query: String,
    pub hybrid: bool,
    pub max_results: usize,
    pub results: Vec<RankedLiveSearchHit<Value>>,
    pub selected: usize,
    pub sequence: u64,
    pub revision: Option<String>,
    pub physical: Option<(String, String, SearchOutputPhase)>,
    pub annotations: Value,
}

#[derive(Deserialize)]
struct HybridEnvelope {
    schema_version: String,
    #[serde(flatten)]
    frame: RetainedLiveSearchFrame,
}

pub(super) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl Model {
    pub fn new(query: String, hybrid: bool, max_results: usize) -> Self {
        Self {
            query,
            hybrid,
            max_results,
            results: Vec::new(),
            selected: 0,
            sequence: 0,
            revision: None,
            physical: None,
            annotations: Value::Null,
        }
    }

    /// Prepare a full replacement without changing the displayed baseline.
    /// The caller installs this value only after terminal presentation succeeds.
    pub fn prepare(&self, bytes: &[u8]) -> io::Result<Self> {
        if !bytes.ends_with(b"\n") || bytes.split(|byte| *byte == b'\n').count() != 2 {
            return Err(invalid("expected exactly one complete live-search record"));
        }
        let mut next = self.clone();
        if self.hybrid {
            let envelope: HybridEnvelope = serde_json::from_slice(bytes).map_err(invalid_json)?;
            if envelope.schema_version != HYBRID_SCHEMA {
                return Err(invalid("unsupported hybrid live-search schema"));
            }
            next.apply_hybrid(envelope.frame)?;
        } else {
            let frame = serde_json::from_slice(bytes).map_err(invalid_json)?;
            next.apply_update(frame)?;
        }
        Ok(next)
    }

    fn apply_hybrid(&mut self, frame: RetainedLiveSearchFrame) -> io::Result<()> {
        let same_generation = self.physical.as_ref().is_some_and(|(id, digest, _)| {
            id == &frame.generation_id && digest == &frame.manifest_sha256
        });
        let valid_phase = match frame.phase {
            SearchOutputPhase::Initial => !same_generation,
            SearchOutputPhase::Refined | SearchOutputPhase::RefinementFailed => {
                same_generation
                    && self
                        .physical
                        .as_ref()
                        .is_some_and(|(_, _, phase)| *phase == SearchOutputPhase::Initial)
            }
        };
        if !valid_phase || frame.generation_id.is_empty() || frame.manifest_sha256.is_empty() {
            return Err(invalid(
                "live-search phase does not follow its physical generation",
            ));
        }
        if frame.annotations.get("query").and_then(Value::as_str) != Some(self.query.as_str())
            || frame.annotations.get("phase")
                != Some(&serde_json::to_value(frame.phase).map_err(invalid_json)?)
        {
            return Err(invalid(
                "hybrid phase annotations contradict their query or phase",
            ));
        }
        match (frame.phase, frame.update) {
            (SearchOutputPhase::RefinementFailed, None) => {
                // Failure is an annotation of the retained Initial, not an empty
                // replacement or a new revision of its result window.
            }
            (SearchOutputPhase::Initial | SearchOutputPhase::Refined, Some(update)) => {
                let revision = format!(
                    "{}@{}:{}",
                    frame.generation_id, frame.manifest_sha256, frame.phase,
                );
                if update.generation != revision {
                    return Err(invalid(
                        "hybrid result revision does not bind its receipt and phase",
                    ));
                }
                self.apply_update(update)?;
            }
            _ => return Err(invalid("hybrid phase has an invalid result-update shape")),
        }
        self.physical = Some((frame.generation_id, frame.manifest_sha256, frame.phase));
        self.annotations = frame.annotations;
        Ok(())
    }

    fn apply_update(&mut self, frame: LiveSearchFrame<Value>) -> io::Result<()> {
        if frame.schema_version != LIVE_SEARCH_SCHEMA_VERSION || frame.query != self.query {
            return Err(invalid("live-search schema or query changed"));
        }
        if self.sequence.checked_add(1) != Some(frame.sequence)
            || frame.previous_generation != self.revision
            || frame.generation.is_empty()
            || self.revision.as_ref() == Some(&frame.generation)
        {
            return Err(invalid(
                "live-search revision chain is missing or out of order",
            ));
        }
        if frame.result_count > self.max_results {
            return Err(invalid(
                "live-search result count exceeds the requested window",
            ));
        }
        let selected_id = self
            .results
            .get(self.selected)
            .map(|result| result.hit.doc_id.clone());
        let by_id: BTreeMap<String, RankedLiveSearchHit<Value>> = match frame.event {
            LiveSearchEvent::Snapshot { results } if self.sequence == 0 => {
                if results.len() > self.max_results {
                    return Err(invalid("live-search snapshot exceeds the requested window"));
                }
                let mut by_id = BTreeMap::new();
                for result in results {
                    if by_id.insert(result.hit.doc_id.clone(), result).is_some() {
                        return Err(invalid("duplicate live-search snapshot identity"));
                    }
                }
                by_id
            }
            LiveSearchEvent::Delta { changes } if self.sequence != 0 => {
                if changes.len() > self.max_results.saturating_mul(2) {
                    return Err(invalid(
                        "live-search delta exceeds the bounded result window",
                    ));
                }
                let mut by_id: BTreeMap<_, _> = self
                    .results
                    .iter()
                    .cloned()
                    .map(|result| (result.hit.doc_id.clone(), result))
                    .collect();
                let mut touched = BTreeSet::new();
                for change in changes {
                    let id = match &change {
                        LiveSearchChange::Added { result }
                        | LiveSearchChange::Updated { result, .. } => &result.hit.doc_id,
                        LiveSearchChange::Removed { doc_id, .. } => doc_id,
                    };
                    if !touched.insert(id.clone()) {
                        return Err(invalid("one live-search delta changes an identity twice"));
                    }
                    match change {
                        LiveSearchChange::Added { result } => {
                            if by_id.insert(result.hit.doc_id.clone(), result).is_some() {
                                return Err(invalid(
                                    "live-search addition replaces an existing identity",
                                ));
                            }
                        }
                        LiveSearchChange::Removed {
                            doc_id,
                            previous_rank,
                        } => {
                            if by_id
                                .remove(&doc_id)
                                .is_none_or(|old| old.rank != previous_rank)
                            {
                                return Err(invalid(
                                    "live-search removal does not match its baseline",
                                ));
                            }
                        }
                        LiveSearchChange::Updated {
                            previous_rank,
                            result,
                        } => {
                            if by_id
                                .get(&result.hit.doc_id)
                                .is_none_or(|old| old.rank != previous_rank)
                            {
                                return Err(invalid(
                                    "live-search update does not match its baseline",
                                ));
                            }
                            by_id.insert(result.hit.doc_id.clone(), result);
                        }
                    }
                }
                by_id
            }
            _ => return Err(invalid("expected a first snapshot followed only by deltas")),
        };
        if by_id.len() != frame.result_count {
            return Err(invalid(
                "live-search result count does not match the complete delta",
            ));
        }
        // Apply the whole delta before checking ranks: a rank swap necessarily
        // has transient collisions while its individual changes are processed.
        let mut results = by_id.into_values().collect::<Vec<_>>();
        results.sort_by_key(|result| result.rank);
        for (offset, result) in results.iter().enumerate() {
            if result.hit.doc_id.is_empty()
                || !result.hit.score.is_finite()
                || u64::try_from(offset + 1).ok() != Some(result.rank)
            {
                return Err(invalid(
                    "live-search identities, scores or ranks are invalid",
                ));
            }
        }
        // A sequence of individually small deltas must not accumulate an
        // unbounded retained window. Count serialization without allocating it.
        let mut bound = WindowBytes(super::MAX_FRAME_BYTES);
        serde_json::to_writer(&mut bound, &results).map_err(invalid_json)?;
        self.selected = selected_id
            .as_ref()
            .and_then(|id| results.iter().position(|result| &result.hit.doc_id == id))
            .unwrap_or_else(|| self.selected.min(results.len().saturating_sub(1)));
        self.results = results;
        self.sequence = frame.sequence;
        self.revision = Some(frame.generation);
        Ok(())
    }

    pub fn move_by(&mut self, distance: isize) {
        self.selected = self
            .selected
            .saturating_add_signed(distance)
            .min(self.results.len().saturating_sub(1));
    }

    pub fn status(&self) -> String {
        if self.sequence == 0 {
            return "Waiting for a committed generation".to_owned();
        }
        let phase = match self.physical.as_ref().map(|(_, _, phase)| phase) {
            None => "Lexical",
            Some(SearchOutputPhase::Initial) => "Initial",
            Some(SearchOutputPhase::Refined) => "Refined",
            Some(SearchOutputPhase::RefinementFailed) => "Refinement failed; Initial retained",
        };
        let reason = self
            .annotations
            .get("skip_reason")
            .and_then(Value::as_str)
            .map_or_else(String::new, |reason| {
                format!(" | {}", display_text(reason, 100))
            });
        format!(
            "{phase} | {} results | revision {}{reason}",
            self.results.len(),
            self.sequence
        )
    }
}

struct WindowBytes(usize);

impl Write for WindowBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_sub(bytes.len())
            .ok_or_else(|| invalid("terminal result window exceeds its byte bound"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn invalid_json(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

/// Terminal data is text, never escape sequences, cursor motion or bidi controls.
/// Limit work before widget layout; the wire/result payload itself is untouched.
pub(super) fn display_text(value: &str, limit: usize) -> String {
    value.chars().take(limit).map(|character| {
        if character.is_control() || matches!(character,
            '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        {
            ' '
        } else {
            character
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use frankensearch_fsfs::adapters::live_search::{
        LiveSearchConfig, LiveSearchHit, LiveSearchTracker,
    };
    use serde_json::json;

    use super::*;

    fn hit(id: &str, score: f64) -> LiveSearchHit<Value> {
        LiveSearchHit {
            doc_id: id.to_owned(),
            score,
            item: json!({"path": id, "snippet": "alpha"}),
        }
    }

    fn tracker() -> LiveSearchTracker<Value> {
        LiveSearchTracker::new(
            "alpha",
            LiveSearchConfig {
                max_results: 3,
                min_score_delta: 0.0,
            },
        )
        .unwrap()
    }

    fn bytes(frame: &LiveSearchFrame<Value>) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(frame).unwrap();
        bytes.push(b'\n');
        bytes
    }

    fn fixture() -> (Model, LiveSearchTracker<Value>) {
        let mut tracker = tracker();
        let frame = tracker
            .apply("g1", vec![hit("a", 3.0), hit("b", 2.0), hit("c", 1.0)])
            .unwrap()
            .unwrap();
        let model = Model::new("alpha".to_owned(), false, 3)
            .prepare(&bytes(&frame))
            .unwrap();
        (model, tracker)
    }

    #[test]
    fn snapshots_and_atomic_rank_swaps_keep_the_selected_identity() {
        let (mut model, mut tracker) = fixture();
        model.selected = 1;
        let frame = tracker
            .apply("g2", vec![hit("b", 5.0), hit("a", 3.0), hit("c", 1.0)])
            .unwrap()
            .unwrap();
        let next = model.prepare(&bytes(&frame)).unwrap();
        assert_eq!(next.selected, 0);
        assert_eq!(next.results[next.selected].hit.doc_id, "b");
        assert_eq!(
            next.results
                .iter()
                .map(|result| result.rank)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(
            model.sequence, 1,
            "preparing must not acknowledge the old display"
        );
    }

    #[test]
    fn removing_selection_uses_a_remaining_rank_and_an_empty_window_is_not_waiting() {
        let (mut model, mut tracker) = fixture();
        model.selected = 2;
        let frame = tracker
            .apply("g2", vec![hit("a", 3.0), hit("b", 2.0)])
            .unwrap()
            .unwrap();
        let model = model.prepare(&bytes(&frame)).unwrap();
        assert_eq!(model.selected, 1);
        let frame = tracker.apply("g3", Vec::new()).unwrap().unwrap();
        let model = model.prepare(&bytes(&frame)).unwrap();
        assert!(model.results.is_empty());
        assert_eq!(model.selected, 0);
        assert!(!model.status().contains("Waiting"));
        assert!(
            Model::new("alpha".to_owned(), false, 3)
                .status()
                .contains("Waiting")
        );
    }

    #[test]
    fn opaque_predecessor_restoration_is_a_new_valid_revision() {
        let (model, mut tracker) = fixture();
        let frame = tracker.apply("g2", vec![hit("b", 2.0)]).unwrap().unwrap();
        let model = model.prepare(&bytes(&frame)).unwrap();
        let frame = tracker.apply("g1", vec![hit("a", 3.0)]).unwrap().unwrap();
        let model = model.prepare(&bytes(&frame)).unwrap();
        assert_eq!(model.sequence, 3);
        assert_eq!(model.revision.as_deref(), Some("g1"));
    }

    #[test]
    fn missing_duplicate_or_foreign_revisions_do_not_change_the_baseline() {
        let (model, mut tracker) = fixture();
        let frame = tracker.apply("g2", vec![hit("b", 2.0)]).unwrap().unwrap();
        for defect in 0..5 {
            let mut bad = frame.clone();
            match defect {
                0 => bad.sequence += 1,
                1 => bad.previous_generation = Some("foreign".to_owned()),
                2 => bad.query = "foreign query".to_owned(),
                3 => bad.schema_version = "unknown".to_owned(),
                _ => bad.generation = "g1".to_owned(),
            }
            assert!(model.prepare(&bytes(&bad)).is_err());
            assert_eq!(model.sequence, 1);
            assert_eq!(model.results[1].hit.doc_id, "b");
        }
        let next = model.prepare(&bytes(&frame)).unwrap();
        assert!(
            next.prepare(&bytes(&frame)).is_err(),
            "replay cannot double-apply a delta"
        );
    }

    #[test]
    fn malformed_delta_identities_ranks_and_counts_are_refused_atomically() {
        let (model, mut tracker) = fixture();
        let frame = tracker
            .apply("g2", vec![hit("b", 5.0), hit("a", 3.0)])
            .unwrap()
            .unwrap();
        for defect in 0..5 {
            let mut bad = frame.clone();
            let LiveSearchEvent::Delta { changes } = &mut bad.event else {
                panic!("delta");
            };
            match defect {
                0 => changes.push(changes[0].clone()),
                1 => bad.result_count = 3,
                2 => changes.push(LiveSearchChange::Removed {
                    doc_id: "missing".into(),
                    previous_rank: 1,
                }),
                3 => {
                    let update = changes
                        .iter_mut()
                        .find_map(|change| match change {
                            LiveSearchChange::Updated { result, .. } => Some(result),
                            _ => None,
                        })
                        .unwrap();
                    update.rank = 99;
                }
                _ => {
                    let previous = changes
                        .iter_mut()
                        .find_map(|change| match change {
                            LiveSearchChange::Updated { previous_rank, .. } => Some(previous_rank),
                            _ => None,
                        })
                        .unwrap();
                    *previous = 99;
                }
            }
            assert!(model.prepare(&bytes(&bad)).is_err(), "defect {defect}");
            assert_eq!(model.results.len(), 3);
        }
    }

    #[test]
    fn invalid_snapshot_and_multiple_records_are_not_admitted() {
        let mut tracker = tracker();
        let frame = tracker.apply("g1", vec![hit("a", 1.0)]).unwrap().unwrap();
        let original = Model::new("alpha".to_owned(), false, 3);
        let mut bad = frame.clone();
        if let LiveSearchEvent::Snapshot { results } = &mut bad.event {
            results.push(results[0].clone());
        }
        bad.result_count = 2;
        assert!(original.prepare(&bytes(&bad)).is_err());
        let mut combined = bytes(&frame);
        combined.extend_from_slice(&bytes(&frame));
        assert!(original.prepare(&combined).is_err());
        assert!(
            original
                .prepare(&serde_json::to_vec(&frame).unwrap())
                .is_err()
        );
    }

    #[test]
    fn score_thresholds_preserve_the_published_scores_in_empty_deltas() {
        let mut tracker = LiveSearchTracker::new(
            "alpha",
            LiveSearchConfig {
                max_results: 3,
                min_score_delta: 0.1,
            },
        )
        .unwrap();
        let first = tracker.apply("g1", vec![hit("a", 1.0)]).unwrap().unwrap();
        let model = Model::new("alpha".into(), false, 3)
            .prepare(&bytes(&first))
            .unwrap();
        let small = tracker.apply("g2", vec![hit("a", 1.05)]).unwrap().unwrap();
        let model = model.prepare(&bytes(&small)).unwrap();
        assert_eq!(model.results[0].hit.score.to_bits(), 1.0_f64.to_bits());
        let accumulated = tracker.apply("g3", vec![hit("a", 1.15)]).unwrap().unwrap();
        let model = model.prepare(&bytes(&accumulated)).unwrap();
        assert_eq!(model.results[0].hit.score.to_bits(), 1.15_f64.to_bits());
    }

    fn hybrid_bytes(
        tracker: &mut LiveSearchTracker<Value>,
        phase: SearchOutputPhase,
        hits: Vec<LiveSearchHit<Value>>,
    ) -> Vec<u8> {
        let update = if phase == SearchOutputPhase::RefinementFailed {
            None
        } else {
            tracker
                .apply(&format!("generation@digest:{phase}"), hits)
                .unwrap()
        };
        let frame = RetainedLiveSearchFrame {
            generation_id: "generation".into(),
            manifest_sha256: "digest".into(),
            phase,
            annotations: json!({"query": "alpha", "phase": phase, "skip_reason": "quality_timeout"}),
            update,
        };
        let mut value = serde_json::to_value(frame).unwrap();
        value["schema_version"] = json!(HYBRID_SCHEMA);
        let mut bytes = serde_json::to_vec(&value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    #[test]
    fn hybrid_refinement_updates_the_same_window_and_checks_its_receipt() {
        let mut tracker = tracker();
        let initial = hybrid_bytes(
            &mut tracker,
            SearchOutputPhase::Initial,
            vec![hit("a", 0.1), hit("b", 0.05)],
        );
        let mut model = Model::new("alpha".into(), true, 3)
            .prepare(&initial)
            .unwrap();
        model.selected = 1;
        let refined = hybrid_bytes(
            &mut tracker,
            SearchOutputPhase::Refined,
            vec![hit("b", 0.9), hit("a", 0.1)],
        );
        let next = model.prepare(&refined).unwrap();
        assert_eq!(next.selected, 0);
        assert_eq!(next.sequence, 2);
        assert!(next.status().starts_with("Refined"));
        let mut bad: Value = serde_json::from_slice(&refined).unwrap();
        bad["manifest_sha256"] = json!("another digest");
        let mut bad = serde_json::to_vec(&bad).unwrap();
        bad.push(b'\n');
        assert!(model.prepare(&bad).is_err());
    }

    #[test]
    fn refinement_failure_preserves_initial_without_fabricating_removals() {
        let mut tracker = tracker();
        let initial = hybrid_bytes(
            &mut tracker,
            SearchOutputPhase::Initial,
            vec![hit("a", 0.1)],
        );
        let model = Model::new("alpha".into(), true, 3)
            .prepare(&initial)
            .unwrap();
        let failed = hybrid_bytes(
            &mut tracker,
            SearchOutputPhase::RefinementFailed,
            Vec::new(),
        );
        let next = model.prepare(&failed).unwrap();
        assert_eq!(next.results, model.results);
        assert_eq!(next.sequence, model.sequence);
        assert_eq!(next.revision, model.revision);
        assert!(next.status().contains("Initial retained"));
        assert!(next.status().contains("quality_timeout"));
        assert!(next.prepare(&failed).is_err());
        assert!(
            Model::new("alpha".into(), true, 3)
                .prepare(&failed)
                .is_err()
        );
    }

    #[test]
    fn bounded_window_counter_counts_all_bytes_without_allocating_them() {
        let mut bound = WindowBytes(5);
        bound.write_all(b"abc").unwrap();
        assert!(bound.write_all(b"def").is_err());
        assert_eq!(bound.0, 2);
    }

    #[test]
    fn controls_and_bidi_are_not_terminal_instructions_and_navigation_is_bounded() {
        let text = display_text("é👩‍💻\u{1b}[2J\n\u{202e}suffix", 20);
        assert!(text.starts_with("é👩‍💻"));
        assert!(!text.chars().any(char::is_control));
        assert!(!text.contains('\u{202e}'));
        assert_eq!(display_text("世界hello", 2), "世界");
        let (mut model, _) = fixture();
        model.move_by(isize::MAX);
        assert_eq!(model.selected, 2);
        model.move_by(isize::MIN);
        assert_eq!(model.selected, 0);
    }
}
