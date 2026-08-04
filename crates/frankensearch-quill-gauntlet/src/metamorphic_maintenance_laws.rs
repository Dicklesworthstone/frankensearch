//! Equivalence relations for the three E6.3 index-maintenance metamorphic laws
//! (`bd-quill-e6-gauntlet-scale-rm3q.3`).
//!
//! [`super::metamorphic_maintenance_schedules`] supplies the seeded, replayable
//! maintenance schedules; this module supplies the other harness-independent
//! half — what counts as EQUIVALENT once a schedule has been applied and two
//! observations compared.
//!
//! A law's equivalence relation is a pure decision over the comparator's
//! divergence set, so it is written and tested here without standing up an
//! index. The executors, wherever they end up living, feed live
//! [`crate::comparator::ComparisonReport`] output into relations that are
//! already proven.
//!
//! # The trap these relations exist to avoid
//!
//! The instruction for this bead is explicit: *do not label a transform
//! semantics-preserving merely because it is convenient*. Maintenance laws are
//! where that temptation bites hardest, because a merge or a compaction really
//! can reorder equal-scoring documents, and it is one short step from "ties may
//! reorder" to "any reordering is a tie". So each relation here admits a
//! NAMED, CLOSED set of divergence classes and rejects everything else,
//! including classes that do not exist yet. A future
//! [`DivergenceClass`][crate::comparator::DivergenceClass] variant is rejected
//! by default rather than silently tolerated.

use crate::comparator::{Divergence, DivergenceClass};

/// Outcome of applying one law's equivalence relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LawVerdict {
    /// Every observed divergence is permitted by the law.
    Equivalent,
    /// At least one divergence is not permitted; the law is violated.
    Violated {
        /// The offending classes, deduplicated, in taxonomy order.
        offending: Vec<DivergenceClass>,
    },
}

impl LawVerdict {
    /// Whether the law held.
    #[must_use]
    pub const fn is_equivalent(&self) -> bool {
        matches!(self, Self::Equivalent)
    }
}

/// Decide a verdict against a closed allow-list of divergence classes.
///
/// Anything not explicitly listed is a violation. This is deliberately an
/// allow-list rather than a deny-list: a deny-list silently admits every
/// divergence class added after it was written, which is how a metamorphic law
/// decays into a rubber stamp without anyone editing it.
fn verdict_against(allowed: &[DivergenceClass], divergences: &[Divergence]) -> LawVerdict {
    let offending: Vec<DivergenceClass> = divergences
        .iter()
        .map(|divergence| divergence.class)
        .filter(|class| !allowed.contains(class))
        .collect();
    if offending.is_empty() {
        return LawVerdict::Equivalent;
    }
    // Deduplicate without requiring Ord on the taxonomy: preserve first-seen
    // order, which is the order an operator reads them in the report.
    let mut seen: Vec<DivergenceClass> = Vec::with_capacity(offending.len());
    for class in offending {
        if !seen.contains(&class) {
            seen.push(class);
        }
    }
    LawVerdict::Violated { offending: seen }
}

/// Divergences permitted by `e6.3-merge-schedule-v1`.
///
/// Merging changes segment geometry, not content. Documents that tie on score
/// may legitimately be emitted in a different order afterwards, so
/// [`DivergenceClass::TieOrder`] is admitted — and nothing else is. In
/// particular a [`DivergenceClass::CountMismatch`] or
/// [`DivergenceClass::DocumentCountMismatch`] means merging changed what is
/// retrievable, which is a defect however convenient it would be to wave it
/// through as "just a merge artifact".
pub const MERGE_SCHEDULE_ALLOWED: &[DivergenceClass] = &[DivergenceClass::TieOrder];

/// Divergences permitted by `e6.3-reopen-recovery-v1`.
///
/// Reopen must restore exactly the durable state. Tie order may differ because
/// segment iteration order is not contractually pinned across a reopen; nothing
/// else may. Note this is the SAME allow-list as merge, not a copy made for
/// convenience: both laws permit exactly the reordering of equal-scoring
/// documents and no change to which documents exist.
pub const REOPEN_RECOVERY_ALLOWED: &[DivergenceClass] = &[DivergenceClass::TieOrder];

/// Divergences permitted by `e6.3-tombstone-compaction-v1`.
///
/// The comparison is against a corpus that never contained the tombstoned
/// documents, so a compacted index must be observationally identical to it
/// apart from tie order. A `CountMismatch` here would mean compaction either
/// resurrected a deleted document or lost a live one — the two failures this
/// law exists to catch — so it is NOT admitted, despite counts being the most
/// tempting thing to excuse when a compaction reclaims space.
///
/// # This relation is CONTINGENT on a precondition that is not yet met
///
/// Admitting only [`DivergenceClass::TieOrder`] is correct **only under the
/// score-insensitive projection this law's preconditions demand**. The registry
/// records `e6.3-tombstone-compaction-v1` as
/// `SkipWithReason(ScoreSensitiveCorpusStatistics)` with the precondition "a
/// score-insensitive projection approved by the runner", and that precondition
/// is currently unmet.
///
/// Under the CURRENT score-sensitive total lexical observation this relation
/// would be wrong, not merely strict: deleting documents changes corpus
/// statistics, so surviving documents legitimately shift in score and rank, and
/// those shifts would surface as [`DivergenceClass::ScoreEpsilon`] or
/// [`DivergenceClass::RankMismatch`] and be reported as violations of a law
/// they do not actually violate.
///
/// So whoever builds the executor must bind this relation to a score-insensitive
/// projection, exactly as the precondition says. Binding it to the score-
/// sensitive observation would manufacture false failures — the mirror image of
/// the vacuous-pass trap the rest of this module guards against, and just as
/// misleading. The relation is inert until then, because the law is skipped and
/// nothing calls it.
pub const TOMBSTONE_COMPACTION_ALLOWED: &[DivergenceClass] = &[DivergenceClass::TieOrder];

/// Equivalence relation for `e6.3-merge-schedule-v1`.
#[must_use]
pub fn merge_schedule_verdict(divergences: &[Divergence]) -> LawVerdict {
    verdict_against(MERGE_SCHEDULE_ALLOWED, divergences)
}

/// Equivalence relation for `e6.3-reopen-recovery-v1`.
#[must_use]
pub fn reopen_recovery_verdict(divergences: &[Divergence]) -> LawVerdict {
    verdict_against(REOPEN_RECOVERY_ALLOWED, divergences)
}

/// Equivalence relation for `e6.3-tombstone-compaction-v1`.
#[must_use]
pub fn tombstone_compaction_verdict(divergences: &[Divergence]) -> LawVerdict {
    verdict_against(TOMBSTONE_COMPACTION_ALLOWED, divergences)
}

#[cfg(test)]
mod tests {
    use super::{
        LawVerdict, MERGE_SCHEDULE_ALLOWED, REOPEN_RECOVERY_ALLOWED, TOMBSTONE_COMPACTION_ALLOWED,
        merge_schedule_verdict, reopen_recovery_verdict, tombstone_compaction_verdict,
    };
    use crate::comparator::{Divergence, DivergenceClass};

    /// Every class in the taxonomy, so the closed-set tests cannot silently
    /// stop covering a variant that gets added later.
    const EVERY_CLASS: [DivergenceClass; 14] = [
        DivergenceClass::TieOrder,
        DivergenceClass::ScoreEpsilon,
        DivergenceClass::RankMismatch,
        DivergenceClass::SnippetMismatch,
        DivergenceClass::SnippetWindow,
        DivergenceClass::CountMismatch,
        DivergenceClass::DocumentCountMismatch,
        DivergenceClass::GlobExpansionLimit,
        DivergenceClass::QueryCanonicalization,
        DivergenceClass::OracleBug,
        DivergenceClass::StatsSemantics,
        DivergenceClass::PostingRecordSemantics,
        DivergenceClass::UnicodeEdge,
        DivergenceClass::OversizedQueryToken,
    ];

    fn divergence(class: DivergenceClass) -> Divergence {
        Divergence {
            class,
            pointer: "/results/0".to_owned(),
            oracle: "oracle".to_owned(),
            subject: "subject".to_owned(),
        }
    }

    type NamedRelation = (
        &'static str,
        fn(&[Divergence]) -> LawVerdict,
        &'static [DivergenceClass],
    );

    fn all_relations() -> Vec<NamedRelation> {
        vec![
            (
                "merge-schedule",
                merge_schedule_verdict as fn(&[Divergence]) -> LawVerdict,
                MERGE_SCHEDULE_ALLOWED,
            ),
            (
                "reopen-recovery",
                reopen_recovery_verdict as fn(&[Divergence]) -> LawVerdict,
                REOPEN_RECOVERY_ALLOWED,
            ),
            (
                "tombstone-compaction",
                tombstone_compaction_verdict as fn(&[Divergence]) -> LawVerdict,
                TOMBSTONE_COMPACTION_ALLOWED,
            ),
        ]
    }

    /// POSITIVE FIXTURE: no divergence at all is trivially equivalent, and the
    /// allowed classes alone are equivalent.
    #[test]
    fn every_relation_admits_an_empty_and_an_allowed_divergence_set() {
        for (name, relation, allowed) in all_relations() {
            assert_eq!(
                relation(&[]),
                LawVerdict::Equivalent,
                "{name} must admit an empty divergence set"
            );
            let permitted: Vec<Divergence> = allowed.iter().copied().map(divergence).collect();
            assert_eq!(
                relation(&permitted),
                LawVerdict::Equivalent,
                "{name} must admit exactly its allowed classes"
            );
        }
    }

    /// PLANTED-INVALID NEGATIVE, and the load-bearing test in this module.
    ///
    /// Every class OUTSIDE a law's allow-list must be rejected. This is what
    /// stops a relation from decaying into "anything a maintenance operation
    /// happens to produce is fine", which would make the law vacuously true
    /// while still reporting as coverage.
    #[test]
    fn every_relation_rejects_every_class_outside_its_allow_list() {
        for (name, relation, allowed) in all_relations() {
            for class in EVERY_CLASS {
                if allowed.contains(&class) {
                    continue;
                }
                let verdict = relation(&[divergence(class)]);
                assert_eq!(
                    verdict,
                    LawVerdict::Violated {
                        offending: vec![class]
                    },
                    "{name} must reject {class:?}, which is not in its allow-list"
                );
                assert!(!verdict.is_equivalent(), "{name} must not admit {class:?}");
            }
        }
    }

    /// A forbidden class must still be caught when it arrives alongside an
    /// allowed one — the realistic shape, since a real merge produces tie-order
    /// noise and the defect hides in it.
    #[test]
    fn a_forbidden_class_is_caught_even_when_mixed_with_allowed_noise() {
        let mixed = vec![
            divergence(DivergenceClass::TieOrder),
            divergence(DivergenceClass::CountMismatch),
            divergence(DivergenceClass::TieOrder),
        ];
        assert_eq!(
            merge_schedule_verdict(&mixed),
            LawVerdict::Violated {
                offending: vec![DivergenceClass::CountMismatch]
            },
            "tie-order noise must not mask a count mismatch"
        );
    }

    /// Offending classes are deduplicated but every distinct one is reported,
    /// so an operator sees the whole failure rather than only its first symptom.
    #[test]
    fn offending_classes_are_deduplicated_and_complete() {
        let repeated = vec![
            divergence(DivergenceClass::CountMismatch),
            divergence(DivergenceClass::RankMismatch),
            divergence(DivergenceClass::CountMismatch),
        ];
        assert_eq!(
            reopen_recovery_verdict(&repeated),
            LawVerdict::Violated {
                offending: vec![
                    DivergenceClass::CountMismatch,
                    DivergenceClass::RankMismatch
                ]
            }
        );
    }

    /// The count-mismatch classes are the ones most likely to be excused as
    /// maintenance artifacts, so their rejection is asserted by name rather
    /// than only as part of the exhaustive sweep above. For tombstone
    /// compaction these are exactly the resurrect-a-deleted-document and
    /// lose-a-live-document failures the law exists to catch.
    #[test]
    fn count_mismatches_are_never_excused_as_maintenance_artifacts() {
        for class in [
            DivergenceClass::CountMismatch,
            DivergenceClass::DocumentCountMismatch,
        ] {
            for (name, relation, _) in all_relations() {
                assert!(
                    !relation(&[divergence(class)]).is_equivalent(),
                    "{name} must never admit {class:?}: maintenance may reorder ties, never \
                     change which documents exist"
                );
            }
        }
    }

    /// The allow-lists are closed sets, asserted explicitly so that widening one
    /// is a deliberate edit to a test rather than an invisible consequence of
    /// editing a constant.
    #[test]
    fn allow_lists_are_exactly_tie_order_for_all_three_laws() {
        for (name, _, allowed) in all_relations() {
            assert_eq!(
                allowed,
                &[DivergenceClass::TieOrder],
                "{name} allow-list changed; widening a metamorphic law's tolerated divergence \
                 set must be a deliberate, reviewed decision"
            );
        }
    }
}
