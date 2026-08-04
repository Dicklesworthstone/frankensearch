//! Seeded maintenance schedules for the E6.3 index-maintenance metamorphic
//! laws (`bd-quill-e6-gauntlet-scale-rm3q.3`).
//!
//! Three E6.3 laws perturb how an index is MAINTAINED rather than what is put
//! into it — `e6.3-merge-schedule-v1`, `e6.3-reopen-recovery-v1`, and
//! `e6.3-tombstone-compaction-v1`. Each needs a deterministic, replayable
//! sequence of maintenance events derived from a seed, plus a shrinker that
//! reduces a failing sequence to a bounded fixture.
//!
//! This module is the harness-independent half of those laws: pure functions
//! over a seed and a corpus size, touching no engine, no config, and no
//! observation. The law executors bind these schedules to live observations
//! elsewhere; keeping the generators separable means a failing seed can be
//! replayed and shrunk without standing up an index at all.
//!
//! # Why non-degeneracy is tested here
//!
//! A metamorphic law is only as strong as its transform. A schedule generator
//! that returns "do nothing" would make its law pass on every input while
//! testing nothing — the law would be vacuously true, which is worse than
//! absent because it reads as coverage. Every generator here therefore has a
//! companion test proving the schedule actually perturbs the maintenance
//! sequence for every seed in its matrix, and the shrinkers refuse to shrink
//! away the last perturbing step.

use std::fmt;

/// Number of distinct seeds each law's campaign replays.
///
/// Fixed rather than time- or budget-derived so a CI run and a local run
/// explore exactly the same schedules.
pub const MAINTENANCE_SEED_MATRIX: [u64; 4] = [
    0x0e63_3a1e_7e5d_0001,
    0x0e63_3a1e_7e5d_0002,
    0x0e63_3a1e_7e5d_0003,
    0x0e63_3a1e_7e5d_0004,
];

/// One step in a seeded maintenance schedule.
///
/// The variants are deliberately coarse: they name maintenance events the
/// engine contract already exposes, not internal implementation steps, so a
/// schedule stays meaningful across engines and across storage revisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaintenanceStep {
    /// Ingest the next `count` documents from the corpus in order.
    Ingest {
        /// How many documents this step admits.
        count: usize,
    },
    /// Force a flush of whatever is currently buffered.
    Flush,
    /// Request a merge of the current segment set.
    Merge,
    /// Close and reopen the index, exercising recovery.
    Reopen,
    /// Delete the document at `corpus_index`, creating a tombstone.
    Tombstone {
        /// Position in the corpus of the document to delete.
        corpus_index: usize,
    },
    /// Request compaction, which may reclaim tombstoned space.
    Compact,
}

impl fmt::Display for MaintenanceStep {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ingest { count } => write!(formatter, "ingest({count})"),
            Self::Flush => formatter.write_str("flush"),
            Self::Merge => formatter.write_str("merge"),
            Self::Reopen => formatter.write_str("reopen"),
            Self::Tombstone { corpus_index } => write!(formatter, "tombstone({corpus_index})"),
            Self::Compact => formatter.write_str("compact"),
        }
    }
}

/// A complete, replayable maintenance schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaintenanceSchedule {
    seed: u64,
    corpus_len: usize,
    steps: Vec<MaintenanceStep>,
}

impl MaintenanceSchedule {
    /// The steps in execution order.
    #[must_use]
    pub fn steps(&self) -> &[MaintenanceStep] {
        &self.steps
    }

    /// The seed this schedule replays from.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Corpus length the schedule was generated against.
    #[must_use]
    pub const fn corpus_len(&self) -> usize {
        self.corpus_len
    }

    /// Total documents ingested across the schedule.
    ///
    /// Every generator here is required to ingest the whole corpus exactly
    /// once: a schedule that silently dropped documents would change WHAT is
    /// indexed, not merely HOW, and the resulting observation difference would
    /// be a corpus difference masquerading as a maintenance effect.
    #[must_use]
    pub fn ingested(&self) -> usize {
        self.steps
            .iter()
            .map(|step| match step {
                MaintenanceStep::Ingest { count } => *count,
                _ => 0,
            })
            .sum()
    }

    /// Count of steps that perturb maintenance rather than admit documents.
    ///
    /// This is the non-degeneracy measure: a schedule whose perturbing count is
    /// zero is just a plain ingest, and a law comparing it against a plain
    /// ingest would be vacuously true.
    #[must_use]
    pub fn perturbing_steps(&self) -> usize {
        self.steps
            .iter()
            .filter(|step| !matches!(step, MaintenanceStep::Ingest { .. }))
            .count()
    }

    /// A compact, redaction-safe rendering for replay artifacts.
    ///
    /// Only structural facts appear — step kinds, counts, and corpus indices —
    /// never document content, so a replay artifact from a failing campaign can
    /// be attached to a report without leaking corpus text.
    #[must_use]
    pub fn replay_signature(&self) -> String {
        let rendered: Vec<String> = self.steps.iter().map(ToString::to_string).collect();
        format!(
            "seed={:#018x} corpus_len={} steps=[{}]",
            self.seed,
            self.corpus_len,
            rendered.join(",")
        )
    }
}

/// Deterministic, non-cryptographic bit mixer.
///
/// `SplitMix64`. Chosen so a schedule depends only on its seed and corpus length,
/// never on wall-clock, thread count, or hash-map iteration order — a schedule
/// that cannot be replayed byte-identically cannot support a shrink/replay
/// contract.
const fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Split `corpus_len` into at least two ingest batches.
///
/// At least two so the schedule can interleave maintenance between them; a
/// single batch would leave nothing to perturb.
fn seeded_batches(state: &mut u64, corpus_len: usize) -> Vec<usize> {
    assert!(
        corpus_len >= 2,
        "a maintenance schedule needs at least two documents to interleave against"
    );
    let max_batches = corpus_len.min(4);
    let batches =
        2 + usize::try_from(mix(state) % u64::try_from(max_batches - 1).unwrap_or(1)).unwrap_or(0);
    let batches = batches.min(corpus_len);
    let base = corpus_len / batches;
    let mut sizes = vec![base; batches];
    for slot in sizes.iter_mut().take(corpus_len % batches) {
        *slot += 1;
    }
    sizes.retain(|size| *size > 0);
    sizes
}

/// Seeded schedule for `e6.3-merge-schedule-v1`.
///
/// Interleaves flushes and merges between ingest batches. The law compares this
/// against an unperturbed ingest: merging is a maintenance decision and must not
/// change the total lexical observation beyond declared tie order.
///
/// # Panics
///
/// Panics when `corpus_len` is below two, which cannot express the transform.
#[must_use]
pub fn merge_schedule(seed: u64, corpus_len: usize) -> MaintenanceSchedule {
    let mut state = seed;
    let batches = seeded_batches(&mut state, corpus_len);
    let mut steps = Vec::with_capacity(batches.len() * 3);
    for (index, count) in batches.iter().enumerate() {
        steps.push(MaintenanceStep::Ingest { count: *count });
        steps.push(MaintenanceStep::Flush);
        // Merge after at least the first batch, then on a seeded subset, so
        // schedules differ across seeds while always perturbing at least once.
        if index == 0 || mix(&mut state) % 2 == 0 {
            steps.push(MaintenanceStep::Merge);
        }
    }
    MaintenanceSchedule {
        seed,
        corpus_len,
        steps,
    }
}

/// Seeded schedule for `e6.3-reopen-recovery-v1`.
///
/// Interleaves close/reopen cycles between ingest batches. The law compares
/// this against an uninterrupted ingest: recovery must restore exactly the
/// durable state, so the observation must be unchanged.
///
/// # Panics
///
/// Panics when `corpus_len` is below two.
#[must_use]
pub fn reopen_recovery_schedule(seed: u64, corpus_len: usize) -> MaintenanceSchedule {
    let mut state = seed ^ 0x5eed_2222_5eed_2222;
    let batches = seeded_batches(&mut state, corpus_len);
    let mut steps = Vec::with_capacity(batches.len() * 3);
    for (index, count) in batches.iter().enumerate() {
        steps.push(MaintenanceStep::Ingest { count: *count });
        // A reopen without a preceding flush exercises recovery of buffered
        // state; with one, it exercises reopen of durable state. Both are in
        // contract, so the seed picks between them.
        if mix(&mut state) % 2 == 0 {
            steps.push(MaintenanceStep::Flush);
        }
        if index + 1 < batches.len() || index == 0 {
            steps.push(MaintenanceStep::Reopen);
        }
    }
    MaintenanceSchedule {
        seed,
        corpus_len,
        steps,
    }
}

/// Seeded schedule for `e6.3-tombstone-compaction-v1`.
///
/// Ingests the whole corpus, tombstones a seeded subset, and compacts. The law
/// compares this against a corpus that never contained the tombstoned
/// documents: compaction must not resurrect deleted documents, and must not
/// disturb the surviving ones beyond declared tie order.
///
/// The returned schedule always tombstones at least one document and always
/// leaves at least one alive — an all-or-nothing deletion would compare against
/// an empty or unchanged index and test far less than it appears to.
///
/// # Panics
///
/// Panics when `corpus_len` is below two.
#[must_use]
pub fn tombstone_compaction_schedule(seed: u64, corpus_len: usize) -> MaintenanceSchedule {
    let mut state = seed ^ 0x7031_3333_7031_3333;
    let batches = seeded_batches(&mut state, corpus_len);
    let mut steps = Vec::with_capacity(batches.len() + corpus_len);
    for count in &batches {
        steps.push(MaintenanceStep::Ingest { count: *count });
    }
    steps.push(MaintenanceStep::Flush);

    let mut tombstoned = Vec::new();
    for corpus_index in 0..corpus_len {
        if mix(&mut state) % 3 == 0 && tombstoned.len() + 1 < corpus_len {
            tombstoned.push(corpus_index);
        }
    }
    if tombstoned.is_empty() {
        tombstoned.push(usize::try_from(mix(&mut state)).unwrap_or(0) % corpus_len);
    }
    for corpus_index in tombstoned {
        steps.push(MaintenanceStep::Tombstone { corpus_index });
    }
    steps.push(MaintenanceStep::Compact);
    MaintenanceSchedule {
        seed,
        corpus_len,
        steps,
    }
}

/// Reduce a failing schedule toward a bounded fixture.
///
/// Greedy single-pass delta debugging: drop one perturbing step at a time,
/// keeping the reduction only while `still_fails` reports the schedule still
/// reproduces the failure. Ingest steps are never dropped, because removing
/// them changes the corpus rather than the maintenance sequence, and the last
/// perturbing step is never dropped, because a schedule with none no longer
/// exercises the law it came from.
///
/// The result is a schedule that still fails, is no longer than the input, and
/// retains at least one perturbing step.
pub fn shrink_schedule<F>(schedule: &MaintenanceSchedule, mut still_fails: F) -> MaintenanceSchedule
where
    F: FnMut(&MaintenanceSchedule) -> bool,
{
    let mut current = schedule.clone();
    let mut index = 0;
    while index < current.steps.len() {
        if matches!(current.steps[index], MaintenanceStep::Ingest { .. })
            || current.perturbing_steps() <= 1
        {
            index += 1;
            continue;
        }
        let mut candidate = current.clone();
        candidate.steps.remove(index);
        if still_fails(&candidate) {
            current = candidate;
        } else {
            index += 1;
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::{
        MAINTENANCE_SEED_MATRIX, MaintenanceSchedule, MaintenanceStep, merge_schedule,
        reopen_recovery_schedule, shrink_schedule, tombstone_compaction_schedule,
    };

    const CORPUS_LEN: usize = 7;

    /// A named seeded schedule generator, as the law campaigns consume them.
    type NamedGenerator = (&'static str, fn(u64, usize) -> MaintenanceSchedule);

    fn all_generators() -> Vec<NamedGenerator> {
        vec![
            ("merge-schedule", merge_schedule),
            ("reopen-recovery", reopen_recovery_schedule),
            ("tombstone-compaction", tombstone_compaction_schedule),
        ]
    }

    /// Replay contract: the same seed must reproduce the same schedule exactly.
    /// Without this a failing campaign seed cannot be replayed at all.
    #[test]
    fn every_generator_replays_byte_identically_from_its_seed() {
        for (name, generate) in all_generators() {
            for seed in MAINTENANCE_SEED_MATRIX {
                let first = generate(seed, CORPUS_LEN);
                let second = generate(seed, CORPUS_LEN);
                assert_eq!(
                    first, second,
                    "{name} seed {seed:#018x} must replay byte-identically"
                );
                assert_eq!(
                    first.replay_signature(),
                    second.replay_signature(),
                    "{name} seed {seed:#018x} replay signature must be stable"
                );
                // A schedule must carry the coordinates it was generated from,
                // or a failing campaign row cannot be turned back into a
                // reproduction: the signature alone is not enough if the
                // schedule disagrees with the seed it claims.
                assert_eq!(
                    first.seed(),
                    seed,
                    "{name} must report the seed it was generated from"
                );
                assert_eq!(
                    first.corpus_len(),
                    CORPUS_LEN,
                    "{name} seed {seed:#018x} must report its corpus length"
                );
            }
        }
    }

    /// NON-DEGENERACY. A generator that emitted no perturbing step would make
    /// its law vacuously true — it would compare a plain ingest against a plain
    /// ingest and pass forever while testing nothing. That is worse than no
    /// coverage, because it reads as coverage.
    #[test]
    fn every_generator_actually_perturbs_maintenance_for_every_seed() {
        for (name, generate) in all_generators() {
            for seed in MAINTENANCE_SEED_MATRIX {
                let schedule = generate(seed, CORPUS_LEN);
                assert!(
                    schedule.perturbing_steps() > 0,
                    "{name} seed {seed:#018x} produced a degenerate schedule with no \
                     maintenance perturbation, which would make its law vacuously true: {}",
                    schedule.replay_signature()
                );
            }
        }
    }

    /// Corpus conservation: a maintenance transform must change HOW documents
    /// are indexed, never WHICH. A schedule that dropped or duplicated ingests
    /// would produce an observation difference that looks like a maintenance
    /// defect but is really a corpus difference.
    #[test]
    fn every_generator_ingests_the_whole_corpus_exactly_once() {
        for (name, generate) in all_generators() {
            for seed in MAINTENANCE_SEED_MATRIX {
                let schedule = generate(seed, CORPUS_LEN);
                assert_eq!(
                    schedule.ingested(),
                    CORPUS_LEN,
                    "{name} seed {seed:#018x} must ingest the corpus exactly once: {}",
                    schedule.replay_signature()
                );
            }
        }
    }

    /// The seed matrix must actually explore distinct schedules; a matrix whose
    /// seeds all collapse to one schedule is a single test wearing four hats.
    #[test]
    fn the_seed_matrix_explores_more_than_one_schedule_per_generator() {
        for (name, generate) in all_generators() {
            let mut signatures: Vec<String> = MAINTENANCE_SEED_MATRIX
                .iter()
                .map(|seed| generate(*seed, CORPUS_LEN).replay_signature())
                .collect();
            signatures.sort_unstable();
            signatures.dedup();
            assert!(
                signatures.len() > 1,
                "{name} collapsed every seed in the matrix onto one schedule"
            );
        }
    }

    /// Tombstone schedules must leave a survivor. Deleting everything would
    /// compare an empty index against an empty index and prove nothing about
    /// compaction disturbing survivors.
    #[test]
    fn tombstone_schedules_delete_at_least_one_and_spare_at_least_one() {
        for seed in MAINTENANCE_SEED_MATRIX {
            let schedule = tombstone_compaction_schedule(seed, CORPUS_LEN);
            let tombstoned = schedule
                .steps()
                .iter()
                .filter(|step| matches!(step, MaintenanceStep::Tombstone { .. }))
                .count();
            assert!(
                tombstoned >= 1,
                "seed {seed:#018x} tombstoned nothing: {}",
                schedule.replay_signature()
            );
            assert!(
                tombstoned < CORPUS_LEN,
                "seed {seed:#018x} tombstoned the entire corpus, leaving no survivor to check: {}",
                schedule.replay_signature()
            );
        }
    }

    /// The shrinker must reduce, must preserve the failure, and must never
    /// shrink a schedule into one that no longer exercises its law.
    #[test]
    fn shrinking_reduces_while_preserving_the_failure_and_keeps_a_perturbation() {
        let schedule = merge_schedule(MAINTENANCE_SEED_MATRIX[0], CORPUS_LEN);
        let before = schedule.steps().len();

        // Failure predicate: "any Merge step remains". Every Merge but the last
        // is therefore droppable, so a correct shrinker reduces but stops.
        let shrunk = shrink_schedule(&schedule, |candidate| {
            candidate
                .steps()
                .iter()
                .any(|step| matches!(step, MaintenanceStep::Merge))
        });

        assert!(
            shrunk.steps().len() <= before,
            "shrinking must never grow a schedule"
        );
        assert!(
            shrunk
                .steps()
                .iter()
                .any(|step| matches!(step, MaintenanceStep::Merge)),
            "shrinking must preserve the failing condition"
        );
        assert!(
            shrunk.perturbing_steps() >= 1,
            "shrinking must not produce a schedule that no longer exercises its law"
        );
        assert_eq!(
            shrunk.ingested(),
            CORPUS_LEN,
            "shrinking must not drop ingest steps and change the corpus"
        );
    }

    /// A shrinker that reduces a schedule whose failure does NOT survive
    /// reduction would hand back a fixture that no longer reproduces anything.
    #[test]
    fn shrinking_keeps_the_original_when_no_reduction_preserves_the_failure() {
        let schedule = merge_schedule(MAINTENANCE_SEED_MATRIX[1], CORPUS_LEN);
        // Failure predicate that only the FULL schedule satisfies.
        let full_len = schedule.steps().len();
        let shrunk = shrink_schedule(&schedule, |candidate| candidate.steps().len() == full_len);
        assert_eq!(
            shrunk, schedule,
            "an irreducible failure must shrink to itself, not to a non-reproducing fixture"
        );
    }
}
