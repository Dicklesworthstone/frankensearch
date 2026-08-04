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
use crate::runner::{
    MetamorphicLawApplicability, MetamorphicLawApplicabilityEntry, MetamorphicLawScope,
    MetamorphicSkipReason,
};

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
/// # The score-sensitivity blocker was real, and COMPACTION discharges it
///
/// The registry records this law as
/// `SkipWithReason(ScoreSensitiveCorpusStatistics)`, precondition "a
/// score-insensitive projection approved by the runner". That skip was
/// well-founded: an earlier probe on this bead compared *delete* against
/// *never-added* and found a survivor's score bits moving (`402a35ec` ->
/// `40082c9c`), because a tombstoned document still occupies the corpus
/// statistics. Under that comparison this TieOrder-only relation would have
/// been wrong rather than strict, and would have manufactured false failures.
///
/// Measurement changed the answer. The transform this law actually names is
/// tombstone **and compaction**, and a real compaction folds the tombstones
/// away — statistics included. Measured across the whole seed matrix in
/// `compaction_restores_never_added_statistics`, a compacted index is
/// observationally identical to a corpus that never contained the deleted
/// documents, under the TOTAL lexical observation, with zero divergences.
/// `without_the_compaction_step_the_total_projection_diverges` is the control
/// that keeps that from being a property of the fixture: drop the `Compact`
/// step and the divergences come back.
///
/// So this relation is bound to the total projection, exactly like merge and
/// reopen, and admits only tie order. The score-insensitive membership
/// projection built for the fallback case remains in
/// [`super::metamorphic_maintenance_laws::maintenance_law_execution`] as
/// diagnostic evidence — it separates "membership broke" from "scores drifted"
/// when a failure does occur — but the law does not need it, and narrowing to
/// it would give up coverage this engine has earned.
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

/// Runner lifecycle capabilities the maintenance laws depend on.
///
/// Declared as data so a law's precondition is an EXECUTABLE GATE rather than a
/// sentence in a descriptor. The registry currently records each maintenance law
/// as `SkipWithReason`, and a prose precondition cannot be checked — which means
/// nothing detects the day the capability arrives, and nothing detects the day
/// it silently regresses either.
///
/// Every field defaults to `false`: a runner must positively declare a
/// capability to gain it. A default of `true` would make a runner that forgot to
/// declare its limits appear fully capable, which is the failure direction that
/// turns a skip into a false pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaintenanceRunnerCapabilities {
    /// The runner can request a merge at a chosen point in a schedule.
    pub deterministic_merge_scheduling: bool,
    /// The runner can close and reopen an index, exercising durable recovery.
    pub durable_reopen_lifecycle: bool,
    /// The runner can execute a real compaction, whose corpus statistics are
    /// then those of a corpus that never contained the deleted documents.
    ///
    /// The registry states this law's precondition as "a score-insensitive
    /// projection approved by the runner", because a bare delete leaves a
    /// tombstoned document in the statistics and moves survivors' scores. A
    /// real compaction removes that difference instead of projecting around it
    /// -- measured in `compaction_restores_never_added_statistics`, with
    /// `without_the_compaction_step_the_total_projection_diverges` as the
    /// control -- so what a runner must actually declare is that it COMPACTS,
    /// and the resulting comparison needs no narrowed projection at all.
    pub compaction_statistics_parity: bool,
}

impl MaintenanceRunnerCapabilities {
    /// The capability set of a runner that declares nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            deterministic_merge_scheduling: false,
            durable_reopen_lifecycle: false,
            compaction_statistics_parity: false,
        }
    }

    /// Applicability of `e6.3-merge-schedule-v1` under these capabilities.
    #[must_use]
    pub const fn merge_schedule(self) -> MetamorphicLawApplicability {
        if self.deterministic_merge_scheduling {
            MetamorphicLawApplicability::Applies
        } else {
            MetamorphicLawApplicability::SkipWithReason {
                reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable,
            }
        }
    }

    /// Applicability of `e6.3-reopen-recovery-v1` under these capabilities.
    #[must_use]
    pub const fn reopen_recovery(self) -> MetamorphicLawApplicability {
        if self.durable_reopen_lifecycle {
            MetamorphicLawApplicability::Applies
        } else {
            MetamorphicLawApplicability::SkipWithReason {
                reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable,
            }
        }
    }

    /// Applicability of `e6.3-tombstone-compaction-v1` under these capabilities.
    ///
    /// Gated on compaction rather than on a narrowed projection. A runner that
    /// deletes without compacting still faces the score-sensitivity the
    /// registry's skip reason names, so the skip reason is unchanged and
    /// correct for that runner; one that compacts has discharged it. See
    /// [`TOMBSTONE_COMPACTION_ALLOWED`].
    #[must_use]
    pub const fn tombstone_compaction(self) -> MetamorphicLawApplicability {
        if self.compaction_statistics_parity {
            MetamorphicLawApplicability::Applies
        } else {
            MetamorphicLawApplicability::SkipWithReason {
                reason: MetamorphicSkipReason::ScoreSensitiveCorpusStatistics,
            }
        }
    }

    /// The complete applicability matrix for the three maintenance laws.
    ///
    /// One entry per (law, scope) pair. All three are Quill-scoped only:
    /// merge scheduling, reopen lifecycle, and compaction are properties of the
    /// subject engine's own storage, so there is no cross-engine or Tantivy
    /// projection to compare them against. Declaring the absent scopes rather
    /// than omitting them would assert an applicability that was never analysed.
    #[must_use]
    pub fn applicability_matrix(self) -> Vec<MetamorphicLawApplicabilityEntry> {
        vec![
            MetamorphicLawApplicabilityEntry {
                law_id: "e6.3-merge-schedule-v1".to_owned(),
                scope: MetamorphicLawScope::Quill,
                applicability: self.merge_schedule(),
            },
            MetamorphicLawApplicabilityEntry {
                law_id: "e6.3-reopen-recovery-v1".to_owned(),
                scope: MetamorphicLawScope::Quill,
                applicability: self.reopen_recovery(),
            },
            MetamorphicLawApplicabilityEntry {
                law_id: "e6.3-tombstone-compaction-v1".to_owned(),
                scope: MetamorphicLawScope::Quill,
                applicability: self.tombstone_compaction(),
            },
        ]
    }
}

#[cfg(test)]
mod capability_tests {
    use super::MaintenanceRunnerCapabilities;
    use crate::runner::{
        MetamorphicLawApplicability, MetamorphicLawApplicabilityEntry, MetamorphicLawScope,
        MetamorphicSkipReason,
    };

    /// The gate must reproduce the registry's CURRENT verdict exactly. If it
    /// disagreed, one of the two would be lying about what the runner can do,
    /// and the descriptor is what campaign accounting reads.
    #[test]
    fn a_runner_declaring_nothing_reproduces_the_registry_skips() {
        let none = MaintenanceRunnerCapabilities::none();
        assert_eq!(
            none.merge_schedule(),
            MetamorphicLawApplicability::SkipWithReason {
                reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable
            }
        );
        assert_eq!(
            none.reopen_recovery(),
            MetamorphicLawApplicability::SkipWithReason {
                reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable
            }
        );
        assert_eq!(
            none.tombstone_compaction(),
            MetamorphicLawApplicability::SkipWithReason {
                reason: MetamorphicSkipReason::ScoreSensitiveCorpusStatistics
            }
        );
    }

    /// Default must equal [`MaintenanceRunnerCapabilities::none`]: a runner that
    /// forgot to declare its limits
    /// must not appear fully capable. That failure direction turns an honest
    /// skip into a false pass, which is the whole thing this bead guards.
    #[test]
    fn default_capabilities_grant_nothing() {
        assert_eq!(
            MaintenanceRunnerCapabilities::default(),
            MaintenanceRunnerCapabilities::none(),
            "a capability must be positively declared, never assumed"
        );
    }

    /// Each capability unlocks EXACTLY its own law. A gate that unlocked a
    /// neighbour would run a law against a runner that cannot execute its
    /// declared projection — the approximation trap in another guise.
    #[test]
    fn each_capability_unlocks_exactly_one_law() {
        let merge_only = MaintenanceRunnerCapabilities {
            deterministic_merge_scheduling: true,
            ..MaintenanceRunnerCapabilities::none()
        };
        assert_eq!(
            merge_only.merge_schedule(),
            MetamorphicLawApplicability::Applies
        );
        assert_ne!(
            merge_only.reopen_recovery(),
            MetamorphicLawApplicability::Applies,
            "merge scheduling must not unlock reopen recovery"
        );
        assert_ne!(
            merge_only.tombstone_compaction(),
            MetamorphicLawApplicability::Applies,
            "merge scheduling must not unlock tombstone compaction"
        );

        let reopen_only = MaintenanceRunnerCapabilities {
            durable_reopen_lifecycle: true,
            ..MaintenanceRunnerCapabilities::none()
        };
        assert_eq!(
            reopen_only.reopen_recovery(),
            MetamorphicLawApplicability::Applies
        );
        assert_ne!(
            reopen_only.merge_schedule(),
            MetamorphicLawApplicability::Applies
        );

        let projection_only = MaintenanceRunnerCapabilities {
            compaction_statistics_parity: true,
            ..MaintenanceRunnerCapabilities::none()
        };
        assert_eq!(
            projection_only.tombstone_compaction(),
            MetamorphicLawApplicability::Applies
        );
        assert_ne!(
            projection_only.merge_schedule(),
            MetamorphicLawApplicability::Applies
        );
    }

    /// The matrix declares one entry per law, Quill-scoped only, and carries the
    /// gate's verdict rather than a separately maintained copy of it.
    #[test]
    fn the_applicability_matrix_is_complete_and_quill_scoped() {
        let matrix = MaintenanceRunnerCapabilities::none().applicability_matrix();
        assert_eq!(matrix.len(), 3, "one entry per maintenance law");
        for entry in &matrix {
            assert_eq!(
                entry.scope,
                MetamorphicLawScope::Quill,
                "{} is a subject-storage property with no cross-engine projection",
                entry.law_id
            );
        }
        let ids: Vec<&str> = matrix.iter().map(|entry| entry.law_id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "e6.3-merge-schedule-v1",
                "e6.3-reopen-recovery-v1",
                "e6.3-tombstone-compaction-v1"
            ]
        );
    }

    /// The matrix must TRACK the gate, not restate it. If a capability flips,
    /// the matrix entry flips with it; otherwise the matrix becomes a stale
    /// second source of truth about what the runner can do.
    #[test]
    fn the_matrix_tracks_capability_changes() {
        let all = MaintenanceRunnerCapabilities {
            deterministic_merge_scheduling: true,
            durable_reopen_lifecycle: true,
            compaction_statistics_parity: true,
        };
        let matrix: Vec<MetamorphicLawApplicabilityEntry> = all.applicability_matrix();
        assert!(
            matrix
                .iter()
                .all(|entry| entry.applicability == MetamorphicLawApplicability::Applies),
            "a fully capable runner must make every maintenance law applicable"
        );
    }
}

/// Real merge/reopen execution for the maintenance laws (`bd-quill-e6-gauntlet-scale-rm3q.3`).
///
/// These laws were registered `SkipWithReason(LifecycleCapabilityUnavailable)`
/// because the E6.3 observation harness controls ingest batching only — there
/// was no way to make a real merge happen. The capability turns out to exist
/// already and simply was not reachable from the laws: `apply_tier_policy` runs
/// inside `QuillIndex::commit` (quill index.rs:6020) and drives
/// `plan_tier_merge` + `build_concat_merge`, so a low `tier_fanout` plus a
/// commit per ingest batch produces GENUINE concat-merges.
///
/// Nothing here approximates. A flush is not a merge, so the perturbation is
/// only accepted when the observed sealed-segment count proves segments were
/// actually combined — see `merge_actually_occurred`.
#[cfg(all(test, feature = "perf-harness"))]
pub mod maintenance_execution {
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::{CompactionPolicy, CompactionReport, QuillConfig, QuillIndex};

    use crate::engine::QuillSubject;
    use crate::metamorphic_maintenance_schedules::{MaintenanceSchedule, MaintenanceStep};

    /// The 6-document maintenance fixture.
    ///
    /// Shared by every route here so a law and its geometry witness never
    /// disagree about what was indexed. `alpha` matches five of the six
    /// documents with repeated terms, so a merge that dropped or duplicated a
    /// posting shows up as a rank or count change rather than as nothing.
    pub fn maintenance_corpus() -> Vec<IndexableDocument> {
        vec![
            IndexableDocument::new("doc-1", "alpha beta beta"),
            IndexableDocument::new("doc-2", "alpha gamma"),
            IndexableDocument::new("doc-3", "beta gamma gamma"),
            IndexableDocument::new("doc-4", "alpha beta gamma delta"),
            IndexableDocument::new("doc-5", "delta epsilon alpha"),
            IndexableDocument::new("doc-6", "alpha alpha beta"),
        ]
    }

    /// `tier_fanout: 2` makes the tier policy merge as soon as a second sealed
    /// segment appears, so committing per batch produces real concat-merges.
    pub fn merging_config() -> QuillConfig {
        QuillConfig {
            tier_fanout: 2,
            ..QuillConfig::default()
        }
    }

    /// Config for the durable reopen-recovery arms.
    ///
    /// `deterministic_ingest` so a replayed seed produces the same durable
    /// state, which is what makes a recovery comparison reproducible rather
    /// than merely repeated.
    pub fn recovery_config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        }
    }

    /// The compaction policy every maintenance schedule uses.
    ///
    /// Well below any density a tombstoned fixture segment can produce, so a
    /// scheduled `Compact` step always has eligible work. The default policy
    /// would legitimately no-op on a six-document fixture, and a law whose
    /// transform silently did nothing is the vacuous pass this module exists to
    /// refuse. The witness in [`MaintainedIndex::compactions_with_work`] is
    /// still what decides whether work happened -- the policy makes it likely,
    /// the report makes it PROVEN.
    pub const MAINTENANCE_COMPACTION_POLICY: CompactionPolicy = CompactionPolicy::new(0.01);

    /// Commit only when there is something to seal.
    ///
    /// Returns the resulting sealed-segment count, or `None` when the index was
    /// already clean. Committing a clean index would push a segment count that
    /// witnesses no work, which is exactly the kind of bookkeeping that makes a
    /// vacuous run look busy.
    async fn commit_if_dirty(cx: &asupersync::Cx, index: &QuillIndex) -> Option<usize> {
        if !index.has_uncommitted_changes() {
            return None;
        }
        let snapshot = index.commit(cx).await.expect("commit maintenance step");
        Some(snapshot.segments().len())
    }

    /// Execute one REAL concat-merge over the complete current manifest run.
    ///
    /// Returns `(segments_before, segments_after)`, or `None` when fewer than
    /// two segments exist so there is nothing to merge. The `None` case is
    /// deliberately not an error: the caller counts witnesses, and a schedule
    /// whose merges all degenerate to `None` must fail its non-degeneracy check
    /// rather than silently pass as "merged".
    async fn merge_full_manifest_run(
        cx: &asupersync::Cx,
        index: &QuillIndex,
    ) -> Option<(usize, usize)> {
        let source_segment_ids = index
            .snapshot()
            .segments()
            .iter()
            .map(|segment| segment.manifest().segment_id)
            .collect::<Vec<_>>();
        let segments_before = source_segment_ids.len();
        if segments_before < 2 {
            return None;
        }
        let output_segment_id = source_segment_ids
            .iter()
            .copied()
            .max()
            .and_then(|segment_id| segment_id.checked_add(1))
            .expect("merge fixture needs a collision-free successor segment id");
        let merged = index
            .concat_merge(cx, &source_segment_ids, output_segment_id, 0)
            .await
            .expect("execute concat merge over the full manifest run");
        let segments_after = merged.segments().len();
        assert!(
            segments_after < segments_before,
            "concat merge published without reducing segment count: \
             before={segments_before}, after={segments_after}"
        );
        Some((segments_before, segments_after))
    }

    /// A live index driven to a committed state, retained rather than probed.
    ///
    /// The subject is kept alive so a law can project it through the SAME
    /// [`crate::engine::GauntletEngine::observe`] path the differential harness
    /// uses. Returning a hand-rolled result summary instead would compare an
    /// invented side channel, and a law proven against a side channel says
    /// nothing about the shipping observation.
    pub struct MaintainedIndex {
        /// The committed subject, ready to observe.
        pub subject: QuillSubject,
        /// Sealed-segment count after each commit this run performed.
        pub sealed_after_each_commit: Vec<usize>,
        /// One `(before, after)` pair per executed concat-merge.
        pub merge_witnesses: Vec<(usize, usize)>,
        /// Live document count reported by each FRESH index instance opened by
        /// a `Reopen` step. Read from the reopened instance itself, so a
        /// non-zero entry proves the state came off disk rather than from the
        /// writer that was just dropped.
        pub reopen_witnesses: Vec<u64>,
        /// The engine's own report from each executed compaction pass.
        pub compaction_witnesses: Vec<CompactionReport>,
    }

    impl MaintainedIndex {
        /// Merges that provably reduced the sealed-segment count.
        ///
        /// This is the non-degeneracy measure for every merge-family law: a run
        /// reporting zero here executed no merge, so any agreement between its
        /// arms is agreement about nothing.
        pub fn real_merges(&self) -> usize {
            self.merge_witnesses
                .iter()
                .filter(|(before, after)| after < before)
                .count()
        }

        /// Reopens that provably recovered documents from durable state.
        ///
        /// A reopen that returned an empty index recovered nothing, so a law
        /// comparing its arms would be comparing an accident. Counting only
        /// non-empty recoveries is what stops "the reopen ran" from standing in
        /// for "the reopen restored the index".
        pub fn real_reopens(&self) -> usize {
            self.reopen_witnesses
                .iter()
                .filter(|doc_count| **doc_count > 0)
                .count()
        }

        /// Compaction passes that provably folded tombstoned rows away.
        ///
        /// `CompactionPolicy` no-ops below its density threshold, and a no-op
        /// pass leaves the law comparing an uncompacted index. Counting only
        /// passes that dropped documents is what makes "compaction ran" mean
        /// "compaction did something".
        pub fn compactions_with_work(&self) -> usize {
            self.compaction_witnesses
                .iter()
                .filter(|report| report.dropped_documents > 0)
                .count()
        }
    }

    /// Where a maintained index lives.
    ///
    /// This is not a convenience knob. A `Reopen` step is only executable
    /// against a directory: an in-memory index has no durable state to recover,
    /// so "reopening" one could only mean constructing a fresh empty index —
    /// the approximation this law family exists to avoid. The backing therefore
    /// decides which steps are executable at all, and an inexecutable step
    /// panics rather than degrading into something adjacent.
    #[derive(Clone, Copy)]
    pub enum MaintenanceBacking<'a> {
        /// Owned-buffer index. `Reopen` is NOT executable.
        InMemory,
        /// Durable index rooted at this directory. Every step is executable.
        Durable(&'a std::path::Path),
    }

    impl MaintenanceBacking<'_> {
        /// Open the index this backing describes.
        async fn open(self, cx: &asupersync::Cx, config: QuillConfig) -> QuillSubject {
            match self {
                Self::InMemory => QuillSubject::in_memory(config).expect("in-memory Quill subject"),
                Self::Durable(directory) => {
                    let index = QuillIndex::create(cx, directory, config.clone())
                        .await
                        .expect("create durable Quill index");
                    QuillSubject::from_open_index(index, config).expect("durable Quill subject")
                }
            }
        }
    }

    /// Execute a seeded [`MaintenanceSchedule`] against a live index.
    ///
    /// Every step is performed for real. `Flush` commits; `Merge` seals and
    /// calls [`QuillIndex::concat_merge`] over the whole manifest run;
    /// `Reopen` DROPS the writer and opens the directory again, so what comes
    /// back was reconstructed from durable state; `Tombstone` deletes through
    /// the shipping `LexicalWrite` path; `Compact` runs real compaction.
    ///
    /// A step this backing cannot perform PANICS by name. Silently ignoring one
    /// would leave the schedule's replay signature describing a run that never
    /// happened, which is the most expensive kind of green.
    pub async fn execute_schedule(
        cx: &asupersync::Cx,
        config: QuillConfig,
        backing: MaintenanceBacking<'_>,
        documents: &[IndexableDocument],
        schedule: &MaintenanceSchedule,
    ) -> MaintainedIndex {
        assert_eq!(
            schedule.corpus_len(),
            documents.len(),
            "schedule was generated for a different corpus length"
        );
        assert_eq!(
            schedule.ingested(),
            documents.len(),
            "a maintenance schedule must ingest the whole corpus exactly once"
        );
        let mut subject = backing.open(cx, config.clone()).await;
        subject
            .claim_fresh_campaign()
            .expect("claim maintenance campaign");
        let mut cursor = 0usize;
        let mut sealed_after_each_commit = Vec::new();
        let mut merge_witnesses = Vec::new();
        let mut reopen_witnesses = Vec::new();
        let mut compaction_witnesses = Vec::new();
        for step in schedule.steps() {
            match *step {
                MaintenanceStep::Ingest { count } => {
                    let end = cursor
                        .checked_add(count)
                        .expect("ingest cursor overflowed the corpus");
                    assert!(
                        end <= documents.len(),
                        "schedule ingests past the end of the corpus"
                    );
                    subject
                        .index_mut()
                        .expect("open maintenance campaign")
                        .index_documents(cx, &documents[cursor..end])
                        .await
                        .expect("index maintenance batch");
                    cursor = end;
                }
                MaintenanceStep::Flush => {
                    let index = subject.index_mut().expect("open maintenance campaign");
                    if let Some(sealed) = commit_if_dirty(cx, index).await {
                        sealed_after_each_commit.push(sealed);
                    }
                }
                MaintenanceStep::Merge => {
                    let index = subject.index_mut().expect("open maintenance campaign");
                    if let Some(sealed) = commit_if_dirty(cx, index).await {
                        sealed_after_each_commit.push(sealed);
                    }
                    if let Some(witness) = merge_full_manifest_run(cx, index).await {
                        merge_witnesses.push(witness);
                    }
                }
                MaintenanceStep::Reopen => {
                    let MaintenanceBacking::Durable(directory) = backing else {
                        panic!(
                            "a Reopen step requires a durable backing; an in-memory index has no \
                             durable state to recover, and constructing a fresh index instead \
                             would approximate the transform rather than execute it"
                        );
                    };
                    // CLOSE for real: the writer is dropped before the
                    // directory is opened again, so nothing in-process can
                    // carry state across the boundary.
                    let closed = subject.take_index().expect("close maintenance index");
                    drop(closed);
                    let reopened = QuillIndex::open(cx, directory, config.clone())
                        .await
                        .expect("reopen the maintenance index from durable state");
                    // Read the count from the FRESH instance: a non-zero value
                    // proves recovery happened rather than a new empty index.
                    reopen_witnesses.push(reopened.doc_count());
                    subject.restore_index(reopened);
                }
                MaintenanceStep::Tombstone { corpus_index } => {
                    let document = documents
                        .get(corpus_index)
                        .expect("tombstone step names a document outside the corpus");
                    let index = subject.index_mut().expect("open maintenance campaign");
                    // Deletion needs a committed target, exactly as the engine
                    // contracts; committing here is part of executing the
                    // tombstone, not a substitute for it.
                    if let Some(sealed) = commit_if_dirty(cx, index).await {
                        sealed_after_each_commit.push(sealed);
                    }
                    let deleted = index
                        .delete_document(cx, &document.id)
                        .await
                        .expect("delete the tombstoned document");
                    assert!(
                        deleted,
                        "tombstone step deleted nothing for {}; a no-op deletion would leave the \
                         law comparing two identical corpora",
                        document.id
                    );
                }
                MaintenanceStep::Compact => {
                    let index = subject.index_mut().expect("open maintenance campaign");
                    if let Some(sealed) = commit_if_dirty(cx, index).await {
                        sealed_after_each_commit.push(sealed);
                    }
                    let report = index
                        .compact(cx, MAINTENANCE_COMPACTION_POLICY)
                        .await
                        .expect("compact the maintained index");
                    compaction_witnesses.push(report);
                }
            }
        }
        assert_eq!(
            cursor,
            documents.len(),
            "schedule finished without ingesting the whole corpus"
        );
        let index = subject.index_mut().expect("open maintenance campaign");
        if let Some(sealed) = commit_if_dirty(cx, index).await {
            sealed_after_each_commit.push(sealed);
        }
        subject
            .mark_committed()
            .expect("publish maintenance campaign");
        MaintainedIndex {
            subject,
            sealed_after_each_commit,
            merge_witnesses,
            reopen_witnesses,
            compaction_witnesses,
        }
    }

    /// The unperturbed control arm: ingest everything once, commit once, and
    /// perform no maintenance at all. This is what a maintenance law compares
    /// against, and it uses the same backing as the perturbed arm so the only
    /// difference between them is the schedule.
    pub async fn ingest_baseline(
        cx: &asupersync::Cx,
        config: QuillConfig,
        backing: MaintenanceBacking<'_>,
        documents: &[IndexableDocument],
    ) -> MaintainedIndex {
        let mut subject = backing.open(cx, config).await;
        subject
            .claim_fresh_campaign()
            .expect("claim baseline campaign");
        let index = subject.index_mut().expect("open baseline campaign");
        index
            .index_documents(cx, documents)
            .await
            .expect("index baseline corpus");
        let mut sealed_after_each_commit = Vec::new();
        if let Some(sealed) = commit_if_dirty(cx, index).await {
            sealed_after_each_commit.push(sealed);
        }
        subject.mark_committed().expect("publish baseline campaign");
        MaintainedIndex {
            subject,
            sealed_after_each_commit,
            merge_witnesses: Vec::new(),
            reopen_witnesses: Vec::new(),
            compaction_witnesses: Vec::new(),
        }
    }

    /// Outcome of driving one ingest schedule to a committed index.
    pub struct MaintenanceOutcome {
        /// Document ids in ranked order for the probe query.
        pub ranked_ids: Vec<String>,
        /// Live document count of the committed snapshot.
        pub doc_count: u64,
        /// Segment count observed after each commit. A merge shows up as a
        /// count that does not grow monotonically with commits.
        pub sealed_after_each_commit: Vec<usize>,
        /// Direct concat-merge witness, when this route explicitly requested
        /// one. The pair is `(segments_before, segments_after)`.
        pub explicit_merge_segment_counts: Option<(usize, usize)>,
    }

    /// Ingest `documents` in `batch_size` chunks, committing after each chunk,
    /// then run `query`.
    ///
    /// Committing per chunk is what makes this a maintenance perturbation
    /// rather than a batching one: each commit seals a segment, and once the
    /// sealed count exceeds `tier_fanout` the tier policy performs a real
    /// concat-merge inside that commit.
    pub async fn ingest_and_probe(
        cx: &asupersync::Cx,
        config: QuillConfig,
        documents: &[IndexableDocument],
        batch_size: usize,
        query: &str,
    ) -> MaintenanceOutcome {
        assert!(batch_size > 0, "ingest batch size must be non-zero");
        let mut subject = QuillSubject::in_memory(config).expect("in-memory Quill subject");
        subject
            .claim_fresh_campaign()
            .expect("claim maintenance campaign");
        let mut sealed_after_each_commit = Vec::new();
        for batch in documents.chunks(batch_size) {
            let index = subject.index_mut().expect("open maintenance campaign");
            index
                .index_documents(cx, batch)
                .await
                .expect("index maintenance batch");
            // Commit per batch is THE SEAM: apply_tier_policy runs inside
            // commit (quill index.rs:6020) and performs a real concat-merge
            // once the sealed count exceeds tier_fanout. The returned snapshot
            // is the merge witness -- proof rather than assumption.
            let snapshot = index.commit(cx).await.expect("commit maintenance batch");
            sealed_after_each_commit.push(snapshot.segments().len());
        }
        let result = subject
            .index_mut()
            .expect("open maintenance campaign")
            .search_paginated(cx, query, 32, 0, true)
            .expect("probe the maintained index");
        let ranked_ids = result
            .hits
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>();
        let doc_count = result.doc_count;
        subject
            .mark_committed()
            .expect("publish maintenance campaign");
        MaintenanceOutcome {
            ranked_ids,
            doc_count,
            sealed_after_each_commit,
            explicit_merge_segment_counts: None,
        }
    }

    /// Ingest one sealed segment per batch, then execute an exact public
    /// [`QuillIndex::concat_merge`] over the complete current manifest run.
    ///
    /// This is deliberately separate from [`ingest_and_probe`]. The latter
    /// tests a policy-derived schedule and its non-degeneracy guard correctly
    /// caught that the chosen corpus did not make the policy merge. This route
    /// instead names the exact source segment IDs and verifies that their
    /// replacement reduced the sealed segment count, so it cannot confuse a
    /// sequence of flushes with a merge.
    pub async fn ingest_explicit_concat_merge_and_probe(
        cx: &asupersync::Cx,
        config: QuillConfig,
        documents: &[IndexableDocument],
        batch_size: usize,
        query: &str,
    ) -> MaintenanceOutcome {
        assert!(batch_size > 0, "ingest batch size must be non-zero");
        let mut subject = QuillSubject::in_memory(config).expect("in-memory Quill subject");
        subject
            .claim_fresh_campaign()
            .expect("claim explicit concat-merge campaign");
        let mut sealed_after_each_commit = Vec::new();
        for batch in documents.chunks(batch_size) {
            let index = subject.index_mut().expect("open explicit merge campaign");
            index
                .index_documents(cx, batch)
                .await
                .expect("index explicit merge batch");
            let snapshot = index.commit(cx).await.expect("commit explicit merge batch");
            sealed_after_each_commit.push(snapshot.segments().len());
        }

        let index = subject.index_mut().expect("open explicit merge campaign");
        let source_segment_ids = index
            .snapshot()
            .segments()
            .iter()
            .map(|segment| segment.manifest().segment_id)
            .collect::<Vec<_>>();
        let segments_before = source_segment_ids.len();
        assert!(
            segments_before >= 2,
            "an explicit concat merge needs at least two committed source segments"
        );
        let output_segment_id = source_segment_ids
            .iter()
            .copied()
            .max()
            .and_then(|segment_id| segment_id.checked_add(1))
            .expect("explicit merge fixture needs a collision-free successor segment id");
        let merged_snapshot = index
            .concat_merge(cx, &source_segment_ids, output_segment_id, 0)
            .await
            .expect("execute explicit concat merge over the full manifest run");
        let segments_after = merged_snapshot.segments().len();
        assert!(
            segments_after < segments_before,
            "explicit concat merge published without reducing segment count: before={segments_before}, after={segments_after}"
        );
        assert_eq!(
            segments_after, 1,
            "merging the full manifest run must leave exactly its replacement segment"
        );

        let result = index
            .search_paginated(cx, query, 32, 0, true)
            .expect("probe explicitly merged index");
        let ranked_ids = result
            .hits
            .iter()
            .map(|hit| hit.document_id.clone())
            .collect::<Vec<_>>();
        let doc_count = result.doc_count;
        subject
            .mark_committed()
            .expect("publish explicit concat-merge campaign");
        MaintenanceOutcome {
            ranked_ids,
            doc_count,
            sealed_after_each_commit,
            explicit_merge_segment_counts: Some((segments_before, segments_after)),
        }
    }
}

/// Live execution of the E6.3 maintenance laws.
///
/// This is the seam the three law families were blocked on: it drives a seeded
/// [`MaintenanceSchedule`] against a real index, projects BOTH arms through the
/// shipping [`GauntletEngine::observe`] path, compares them with the same
/// [`compare_observations`] the differential harness calls, and decides the
/// result with the equivalence relation proven at the top of this module.
///
/// Two properties are load-bearing and neither is optional:
///
/// * The projection is the total lexical observation, not a summary invented
///   here. A law proven against a side channel proves nothing about what ships.
/// * Every outcome carries the number of merges that PROVABLY happened, so a
///   caller can refuse to count agreement produced by a run that merged
///   nothing.
#[cfg(all(test, feature = "perf-harness"))]
pub mod maintenance_law_execution {
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::QuillConfig;

    use super::maintenance_execution::{MaintenanceBacking, execute_schedule, ingest_baseline};
    use super::{
        LawVerdict, merge_schedule_verdict, reopen_recovery_verdict, tombstone_compaction_verdict,
    };
    use crate::comparator::{
        ComparatorConfig, ComparisonReport, CountState, Divergence, DivergenceClass,
        EngineObservation, compare_observations,
    };
    use crate::engine::{DifferentialCase, GauntletEngine};
    use crate::metamorphic_maintenance_schedules::{MaintenanceSchedule, MaintenanceStep};

    /// One live execution of a maintenance law.
    pub struct MaintenanceLawOutcome {
        /// The equivalence relation's decision over the live divergence set.
        pub verdict: LawVerdict,
        /// The full comparator report both arms produced.
        pub report: ComparisonReport,
        /// Commits the perturbed arm actually performed. A schedule whose
        /// flushes all no-opped never sealed anything, and nothing downstream
        /// of a seal -- a merge above all -- could have happened.
        pub commits_executed: usize,
        /// Merges that provably reduced segment count in the perturbed arm.
        pub merges_executed: usize,
        /// Reopens that provably recovered a non-empty index from disk.
        pub reopens_executed: usize,
        /// Redaction-safe replay identity of the executed schedule.
        pub replay_signature: String,
    }

    /// The observation projection every maintenance law is compared through.
    ///
    /// Snippets are disabled because the scalar Quill adapter refuses them; the
    /// remaining projection — ranked hits with score bits, both tie groups,
    /// exact count, live document count, and recorded AST differences — is the
    /// complete one, deliberately NOT narrowed to the fields a merge is
    /// expected to preserve.
    fn law_case(law_id: &str, schedule: &MaintenanceSchedule, query: &str) -> DifferentialCase {
        let mut case = DifferentialCase::new(
            format!("{law_id}-seed-{:#018x}", schedule.seed()),
            query,
            16,
        );
        case.snippet_max_chars = None;
        case.tie_expansion_limit = 64;
        case.metadata.generator_id = Some(law_id.to_owned());
        case.metadata.generator_seed = Some(schedule.seed());
        case
    }

    /// Execute `e6.3-merge-schedule-v1` against real concat-merges.
    ///
    /// `maintained_documents` is a separate slice from `baseline_documents`
    /// solely so a planted-invalid fixture can mutate one arm. A positive run
    /// passes the same corpus twice; anything else is a negative control and
    /// must be labelled as one.
    pub async fn run_merge_schedule_law(
        cx: &asupersync::Cx,
        config: QuillConfig,
        baseline_documents: &[IndexableDocument],
        maintained_documents: &[IndexableDocument],
        schedule: &MaintenanceSchedule,
        query: &str,
    ) -> MaintenanceLawOutcome {
        run_total_projection_law(
            cx,
            config,
            MaintenanceBacking::InMemory,
            MaintenanceBacking::InMemory,
            baseline_documents,
            maintained_documents,
            schedule,
            query,
            "e6.3-merge-schedule-v1",
            merge_schedule_verdict,
        )
        .await
    }

    /// Execute `e6.3-reopen-recovery-v1` against real close/reopen cycles.
    ///
    /// Both arms are DURABLE and use the same config; the only difference is
    /// that the maintained arm closes and reopens its directory at the points
    /// the schedule names. An in-memory control would differ from the perturbed
    /// arm in storage backend as well as in maintenance, and any divergence
    /// would then be unattributable.
    pub async fn run_reopen_recovery_law(
        cx: &asupersync::Cx,
        config: QuillConfig,
        baseline_directory: &std::path::Path,
        maintained_directory: &std::path::Path,
        baseline_documents: &[IndexableDocument],
        maintained_documents: &[IndexableDocument],
        schedule: &MaintenanceSchedule,
        query: &str,
    ) -> MaintenanceLawOutcome {
        run_total_projection_law(
            cx,
            config,
            MaintenanceBacking::Durable(baseline_directory),
            MaintenanceBacking::Durable(maintained_directory),
            baseline_documents,
            maintained_documents,
            schedule,
            query,
            "e6.3-reopen-recovery-v1",
            reopen_recovery_verdict,
        )
        .await
    }

    /// Shared body for the laws whose projection is the TOTAL observation.
    ///
    /// Merge and reopen both claim exactness up to tie order, so both compare
    /// through the complete `EngineObservation`. Tombstone/compaction does not
    /// and must not use this path — see
    /// [`run_tombstone_compaction_law`].
    #[allow(clippy::too_many_arguments)] // every argument is an arm, a fixture, or a law identity
    async fn run_total_projection_law(
        cx: &asupersync::Cx,
        config: QuillConfig,
        baseline_backing: MaintenanceBacking<'_>,
        maintained_backing: MaintenanceBacking<'_>,
        baseline_documents: &[IndexableDocument],
        maintained_documents: &[IndexableDocument],
        schedule: &MaintenanceSchedule,
        query: &str,
        law_id: &str,
        verdict_of: fn(&[Divergence]) -> LawVerdict,
    ) -> MaintenanceLawOutcome {
        let maintained = execute_schedule(
            cx,
            config.clone(),
            maintained_backing,
            maintained_documents,
            schedule,
        )
        .await;
        let baseline = ingest_baseline(cx, config, baseline_backing, baseline_documents).await;
        let case = law_case(law_id, schedule, query);
        let maintained_observation = maintained
            .subject
            .observe(cx, &case)
            .await
            .expect("observe the maintained index");
        let baseline_observation = baseline
            .subject
            .observe(cx, &case)
            .await
            .expect("observe the unperturbed index");
        let report = compare_observations(
            maintained_observation,
            baseline_observation,
            ComparatorConfig::default(),
        )
        .expect("compare maintenance observations");
        let verdict = verdict_of(&report.divergences);
        MaintenanceLawOutcome {
            verdict,
            commits_executed: maintained.sealed_after_each_commit.len(),
            merges_executed: maintained.real_merges(),
            reopens_executed: maintained.real_reopens(),
            replay_signature: schedule.replay_signature(),
            report,
        }
    }

    /// The score-insensitive projection `e6.3-tombstone-compaction-v1` requires.
    ///
    /// # What it keeps, and why that is the whole law
    ///
    /// Membership of the returned set, the live document count, and the exact
    /// count state. Those are precisely the two failures this law exists to
    /// catch: a compaction that RESURRECTS a tombstoned document, and one that
    /// LOSES a survivor.
    ///
    /// # What it drops, named rather than implied
    ///
    /// Score bits and rank order. This is a real blind spot and it is
    /// deliberate: deleting documents changes corpus statistics, so a survivor
    /// legitimately moves in score and rank relative to a corpus that never
    /// contained the deleted documents. Comparing those under the total
    /// projection manufactures failures for a law that was never violated —
    /// the mirror image of a vacuous pass. The evidence for that claim is not
    /// an argument: `the_total_projection_would_report_a_false_tombstone_failure`
    /// measures it.
    ///
    /// # Why this is not a weakened gate
    ///
    /// A weakened gate admits the failure it was built to catch. This drops a
    /// dimension in which the two arms are NOT expected to agree, and keeps
    /// every dimension in which they are. Ordering regressions are still owned
    /// by the merge and reopen laws, which compare through the total
    /// projection with no such allowance.
    #[derive(Debug, PartialEq, Eq)]
    pub struct MembershipProjection {
        /// Returned document ids as a set: sorted and deduplicated.
        doc_ids: Vec<String>,
        /// Live document count of the snapshot.
        doc_count: u64,
        /// Exact-count evidence state.
        match_count: CountState,
    }

    impl MembershipProjection {
        /// Project one observation down to membership evidence.
        fn of(observation: &EngineObservation) -> Self {
            let mut doc_ids = observation
                .hits
                .iter()
                .map(|hit| hit.doc_id.clone())
                .collect::<Vec<_>>();
            doc_ids.sort_unstable();
            doc_ids.dedup();
            Self {
                doc_ids,
                doc_count: observation.doc_count,
                match_count: observation.match_count,
            }
        }

        /// Differences between two projections, expressed in the comparator's
        /// own taxonomy so the SAME equivalence relation decides them.
        ///
        /// Membership differences are reported as `RankMismatch` and count
        /// differences as `CountMismatch`/`DocumentCountMismatch` — the classes
        /// the real comparator would raise for the same facts, and all three
        /// are outside every maintenance law's allow-list.
        fn divergences_against(&self, other: &Self) -> Vec<Divergence> {
            let mut divergences = Vec::new();
            if self.doc_ids != other.doc_ids {
                divergences.push(Divergence {
                    class: DivergenceClass::RankMismatch,
                    pointer: "/hits/membership".to_owned(),
                    oracle: other.doc_ids.join(","),
                    subject: self.doc_ids.join(","),
                });
            }
            if self.doc_count != other.doc_count {
                divergences.push(Divergence {
                    class: DivergenceClass::DocumentCountMismatch,
                    pointer: "/doc_count".to_owned(),
                    oracle: other.doc_count.to_string(),
                    subject: self.doc_count.to_string(),
                });
            }
            if self.match_count != other.match_count {
                divergences.push(Divergence {
                    class: DivergenceClass::CountMismatch,
                    pointer: "/match_count".to_owned(),
                    oracle: format!("{:?}", other.match_count),
                    subject: format!("{:?}", self.match_count),
                });
            }
            divergences
        }
    }

    /// One live execution of the tombstone/compaction law.
    pub struct TombstoneLawOutcome {
        /// Verdict under the TOTAL lexical projection: the law.
        pub verdict: LawVerdict,
        /// Divergences the score-insensitive membership projection observed.
        /// Diagnostic only: it separates a membership failure (resurrection or
        /// loss) from score drift when the total projection reports something.
        pub membership_divergences: Vec<Divergence>,
        /// Divergences the total projection observed, which decide the verdict.
        pub total_divergences: Vec<Divergence>,
        /// Commits the perturbed arm actually performed.
        pub commits_executed: usize,
        /// Compaction passes that provably dropped tombstoned rows.
        pub compactions_with_work: usize,
        /// Redaction-safe replay identity of the executed schedule.
        pub replay_signature: String,
    }

    /// Execute `e6.3-tombstone-compaction-v1` against real deletes and a real
    /// compaction, comparing against a corpus that NEVER contained the deleted
    /// documents.
    ///
    /// `surviving_documents` is supplied by the caller rather than derived here
    /// so a negative control can plant a wrong survivor set; positive callers
    /// derive it from the schedule with [`survivors_of`].
    pub async fn run_tombstone_compaction_law(
        cx: &asupersync::Cx,
        config: QuillConfig,
        full_documents: &[IndexableDocument],
        surviving_documents: &[IndexableDocument],
        schedule: &MaintenanceSchedule,
        query: &str,
    ) -> TombstoneLawOutcome {
        let maintained = execute_schedule(
            cx,
            config.clone(),
            MaintenanceBacking::InMemory,
            full_documents,
            schedule,
        )
        .await;
        let baseline = ingest_baseline(
            cx,
            config,
            MaintenanceBacking::InMemory,
            surviving_documents,
        )
        .await;
        let case = law_case("e6.3-tombstone-compaction-v1", schedule, query);
        let maintained_observation = maintained
            .subject
            .observe(cx, &case)
            .await
            .expect("observe the compacted index");
        let baseline_observation = baseline
            .subject
            .observe(cx, &case)
            .await
            .expect("observe the never-added corpus");
        let membership_divergences = MembershipProjection::of(&maintained_observation)
            .divergences_against(&MembershipProjection::of(&baseline_observation));
        let total_divergences = compare_observations(
            maintained_observation,
            baseline_observation,
            ComparatorConfig::default(),
        )
        .expect("compare tombstone observations")
        .divergences;
        // THE LAW IS DECIDED ON THE TOTAL PROJECTION, the same instrument merge
        // and reopen use. The membership projection travels alongside as
        // evidence, not as the verdict: see TOMBSTONE_COMPACTION_ALLOWED for
        // why the narrower instrument turned out to be unnecessary, and
        // `without_the_compaction_step_the_total_projection_diverges` for the
        // control that proves compaction is what earned it.
        TombstoneLawOutcome {
            verdict: tombstone_compaction_verdict(&total_divergences),
            membership_divergences,
            total_divergences,
            commits_executed: maintained.sealed_after_each_commit.len(),
            compactions_with_work: maintained.compactions_with_work(),
            replay_signature: schedule.replay_signature(),
        }
    }

    /// The corpus a tombstone schedule leaves alive, in corpus order.
    ///
    /// This is the never-added control the law compares against: not "the same
    /// corpus with deletions applied", but a corpus that never contained the
    /// deleted documents at all.
    pub fn survivors_of(
        documents: &[IndexableDocument],
        schedule: &MaintenanceSchedule,
    ) -> Vec<IndexableDocument> {
        let tombstoned = schedule
            .steps()
            .iter()
            .filter_map(|step| match step {
                MaintenanceStep::Tombstone { corpus_index } => Some(*corpus_index),
                _ => None,
            })
            .collect::<Vec<_>>();
        documents
            .iter()
            .enumerate()
            .filter(|(index, _)| !tombstoned.contains(index))
            .map(|(_, document)| document.clone())
            .collect()
    }
}

#[cfg(all(test, feature = "perf-harness"))]
mod merge_execution_tests {
    use super::maintenance_execution::{
        ingest_and_probe, ingest_explicit_concat_merge_and_probe, maintenance_corpus,
    };
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::QuillConfig;

    fn corpus() -> Vec<IndexableDocument> {
        maintenance_corpus()
    }

    fn merging_config() -> QuillConfig {
        super::maintenance_execution::merging_config()
    }

    /// e6.3-merge-schedule-v1 against real merges.
    ///
    /// IGNORED, and deliberately not deleted or weakened. `tier_fanout: 2` plus
    /// a commit per batch does NOT by itself produce a merge in this harness:
    /// the run above observed six commits leaving six segments
    /// (`[1, 2, 3, 4, 5, 6]`). `plan_tier_merge` (keeper.rs:1459) requires more
    /// than a segment count — every segment in a `fanout`-wide window must
    /// classify to the SAME tier width via `classify_width(docid_hi - docid_lo)`,
    /// and the window must pass a `max_hole_ratio` check. A one-document-per-
    /// commit corpus does not satisfy that.
    ///
    /// The non-degeneracy guard below is what caught this. Without it the test
    /// would PASS — the two arms agree trivially when nothing is merged — and
    /// this law would have been reported as executed against real merges while
    /// exercising none. That is precisely the vacuous-pass the bead's
    /// "do not label a transform semantics-preserving merely because it is
    /// convenient" instruction exists to prevent, so the guard stays and the
    /// test stays ignored until the corpus provably drives a merge.
    ///
    /// To un-ignore: build a corpus whose committed segments land in one tier
    /// width with an acceptable hole ratio, confirm `sealed_after_each_commit`
    /// is non-monotonic, then remove `#[ignore]`. Do NOT remove the guard.
    #[ignore = "tier_fanout alone does not trigger a merge; see doc comment (bd-quill-e6-gauntlet-scale-rm3q.3)"]
    #[test]
    fn merge_schedule_law_holds_against_real_concat_merges() {
        let documents = corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline =
                ingest_and_probe(&cx, merging_config(), &documents, documents.len(), "alpha").await;
            let merged = ingest_and_probe(&cx, merging_config(), &documents, 1, "alpha").await;

            // NON-DEGENERACY, proven from the commit snapshots rather than
            // assumed: six commits that never merged would leave six segments.
            // A segment count below the commit count is a merge that happened.
            let commits = merged.sealed_after_each_commit.len();
            let final_segments = *merged
                .sealed_after_each_commit
                .last()
                .expect("at least one commit");
            assert!(
                final_segments < commits,
                "no real merge occurred: {commits} commits left {final_segments} segments \
                 ({:?}); this law would be vacuously true and a flush would have been \
                 indistinguishable from a merge",
                merged.sealed_after_each_commit
            );

            // THE LAW: merging changes segment geometry, not content.
            assert_eq!(
                merged.doc_count, baseline.doc_count,
                "merging must not change the live document count"
            );
            assert_eq!(
                merged.ranked_ids, baseline.ranked_ids,
                "merging must not change the ranked result"
            );
        });
    }

    /// The direct execution seam used by the E6.3 merge family. Unlike the
    /// policy-attempt test above, this names every current manifest segment as
    /// a source and observes the successor snapshot, which proves the
    /// transform happened before comparing query output.
    #[test]
    fn direct_concat_merge_changes_geometry_but_not_the_observable_corpus() {
        let documents = corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline =
                ingest_and_probe(&cx, merging_config(), &documents, documents.len(), "alpha").await;
            let merged = ingest_explicit_concat_merge_and_probe(
                &cx,
                merging_config(),
                &documents,
                1,
                "alpha",
            )
            .await;

            let (segments_before, segments_after) = merged
                .explicit_merge_segment_counts
                .expect("direct concat-merge route must return a geometry witness");
            assert!(
                segments_after < segments_before,
                "the direct merge witness must prove a real geometry transform"
            );
            assert_eq!(
                merged.doc_count, baseline.doc_count,
                "direct concat merge must not change the live document count"
            );
            assert_eq!(
                merged.ranked_ids, baseline.ranked_ids,
                "direct concat merge must not change the ranked result"
            );
        });
    }
}

/// `e6.3-merge-schedule-v1` executed end to end: seeded schedule, real merges,
/// shipping projection, proven equivalence relation.
///
/// These are the four acceptance components the law was missing — an executable
/// precondition, a live observable projection, a positive fixture measured
/// rather than asserted, and the applicability flip the precondition earns.
#[cfg(all(test, feature = "perf-harness"))]
mod merge_schedule_law_tests {
    use super::maintenance_execution::{maintenance_corpus, merging_config};
    use super::maintenance_law_execution::run_merge_schedule_law;
    use super::{LawVerdict, MaintenanceRunnerCapabilities};
    use crate::metamorphic_maintenance_schedules::{MAINTENANCE_SEED_MATRIX, merge_schedule};
    use crate::runner::{MetamorphicLawApplicability, MetamorphicSkipReason};
    use frankensearch_core::IndexableDocument;

    /// The probe query. `alpha` matches five of the six documents, so a merge
    /// that lost, duplicated, or reordered a posting cannot hide inside an
    /// empty or single-hit result.
    const PROBE: &str = "alpha";

    /// POSITIVE FIXTURE, measured live for every seed in the fixed matrix.
    ///
    /// Each seed's schedule is executed against a real index — real commits,
    /// real [`QuillIndex::concat_merge`] — and both arms are projected through
    /// the shipping observation path before the equivalence relation decides.
    ///
    /// The non-degeneracy assertion is not decoration. Merging is the whole
    /// transform: a seed whose schedule merged nothing would make this law
    /// vacuously true, and the run would report "law holds against real merges"
    /// having executed none.
    #[test]
    fn the_merge_schedule_law_holds_across_the_seed_matrix() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in MAINTENANCE_SEED_MATRIX {
                let schedule = merge_schedule(seed, documents.len());
                let outcome = run_merge_schedule_law(
                    &cx,
                    merging_config(),
                    &documents,
                    &documents,
                    &schedule,
                    PROBE,
                )
                .await;
                assert!(
                    outcome.commits_executed >= 2,
                    "a merge needs at least two seals to combine: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.merges_executed >= 1,
                    "seed executed no real merge, so the law would be vacuously true: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.verdict.is_equivalent(),
                    "merge-schedule law violated for {}: {:?} (divergences {:?})",
                    outcome.replay_signature,
                    outcome.verdict,
                    outcome.report.divergences
                );
                // The comparison must have had something to compare. An empty
                // result on both sides agrees perfectly and proves nothing.
                assert!(
                    !outcome.report.subject.hits.is_empty(),
                    "the probe returned no hits, so the observation is empty: {}",
                    outcome.replay_signature
                );
            }
        });
    }

    /// PLANTED-INVALID NEGATIVE against the SAME live path.
    ///
    /// One document's content differs in the merged arm. Nothing about the
    /// schedule, the projection, or the relation changes — only the corpus — so
    /// a violation here proves the executor can actually fail. Without this the
    /// positive test above would be indistinguishable from a comparison that
    /// silently compares an index against itself.
    /// The mutation has to be visible to the PROBE, not merely to the corpus.
    ///
    /// Replacing `delta` with `saffron` in doc-4 was the first attempt and it
    /// produced zero divergences — correctly, since the probe is `alpha`, both
    /// documents keep one `alpha` in four tokens, and the corpus statistics for
    /// `alpha` are untouched. A negative control that mutates something the
    /// observation cannot see does not fail, and would have been mistaken for a
    /// comparator that cannot fail at all. So this mutation removes `alpha`
    /// from doc-4 and thereby changes what the probe returns.
    #[test]
    fn a_planted_corpus_mutation_makes_the_merge_schedule_law_fail() {
        let baseline = maintenance_corpus();
        let mut mutated = maintenance_corpus();
        mutated[3] = IndexableDocument::new("doc-4", "beta gamma delta saffron");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let schedule = merge_schedule(MAINTENANCE_SEED_MATRIX[0], baseline.len());
            let outcome = run_merge_schedule_law(
                &cx,
                merging_config(),
                &baseline,
                &mutated,
                &schedule,
                PROBE,
            )
            .await;
            assert!(
                outcome.merges_executed >= 1,
                "the negative control must exercise the same real merge: {}",
                outcome.replay_signature
            );
            let LawVerdict::Violated { offending } = &outcome.verdict else {
                panic!(
                    "a mutated corpus must violate the merge-schedule law, got {:?} with \
                     divergences {:?}",
                    outcome.verdict, outcome.report.divergences
                );
            };
            assert!(
                !offending.is_empty(),
                "a violation must name the classes it rejected"
            );
        });
    }

    /// EXECUTABLE PRECONDITION, earned in the same invocation that proves it.
    ///
    /// The registry's precondition for this law is "runner-exposed deterministic
    /// merge scheduling". This test does not assert that sentence — it executes
    /// a schedule, requires a witnessed merge, and only then asserts that a
    /// runner declaring the capability flips the law to `Applies` while one
    /// declaring nothing still skips.
    ///
    /// The pairing is the point: a capability flag that can be set without a
    /// live merge behind it is exactly how a skip becomes a false pass.
    #[test]
    fn the_merge_capability_flip_is_earned_by_a_witnessed_merge() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let schedule = merge_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let outcome = run_merge_schedule_law(
                &cx,
                merging_config(),
                &documents,
                &documents,
                &schedule,
                PROBE,
            )
            .await;
            assert!(
                outcome.merges_executed >= 1,
                "no witnessed merge, so no capability may be declared: {}",
                outcome.replay_signature
            );

            let declared = MaintenanceRunnerCapabilities {
                deterministic_merge_scheduling: true,
                ..MaintenanceRunnerCapabilities::none()
            };
            assert_eq!(
                declared.merge_schedule(),
                MetamorphicLawApplicability::Applies,
                "a runner with a witnessed merge must apply the law"
            );
            assert_eq!(
                MaintenanceRunnerCapabilities::none().merge_schedule(),
                MetamorphicLawApplicability::SkipWithReason {
                    reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable,
                },
                "a runner without the capability must still skip, witness or not"
            );
            // The other two laws are NOT unlocked by a merge witness. Declaring
            // one capability must never leak into another law's gate.
            assert_eq!(
                declared.reopen_recovery(),
                MetamorphicLawApplicability::SkipWithReason {
                    reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable,
                },
            );
            assert_eq!(
                declared.tombstone_compaction(),
                MetamorphicLawApplicability::SkipWithReason {
                    reason: MetamorphicSkipReason::ScoreSensitiveCorpusStatistics,
                },
            );
        });
    }
}

/// `e6.3-reopen-recovery-v1` executed end to end against real close/reopen
/// cycles on durable indexes.
#[cfg(all(test, feature = "perf-harness"))]
mod reopen_recovery_law_tests {
    use super::maintenance_execution::{
        MaintenanceBacking, execute_schedule, maintenance_corpus, recovery_config,
    };
    use super::maintenance_law_execution::run_reopen_recovery_law;
    use super::{LawVerdict, MaintenanceRunnerCapabilities};
    use crate::metamorphic_maintenance_schedules::{
        MAINTENANCE_SEED_MATRIX, MaintenanceSchedule, MaintenanceStep, reopen_recovery_schedule,
    };
    use crate::runner::{MetamorphicLawApplicability, MetamorphicSkipReason};
    use frankensearch_core::IndexableDocument;

    const PROBE: &str = "alpha";

    /// POSITIVE FIXTURE, measured live for every seed in the fixed matrix.
    ///
    /// Both arms are durable; the maintained one closes its writer and opens
    /// the directory again wherever the schedule says. The recovery witness is
    /// read from the FRESH instance, so a run that never recovered anything
    /// cannot be counted as a run that did.
    #[test]
    fn the_reopen_recovery_law_holds_across_the_seed_matrix() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in MAINTENANCE_SEED_MATRIX {
                let baseline_root = tempfile::tempdir().expect("baseline index directory");
                let maintained_root = tempfile::tempdir().expect("maintained index directory");
                let schedule = reopen_recovery_schedule(seed, documents.len());
                let outcome = run_reopen_recovery_law(
                    &cx,
                    recovery_config(),
                    baseline_root.path(),
                    maintained_root.path(),
                    &documents,
                    &documents,
                    &schedule,
                    PROBE,
                )
                .await;
                assert!(
                    outcome.commits_executed >= 1,
                    "a recovery needs durable state, which needs a seal: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.reopens_executed >= 1,
                    "no reopen recovered a non-empty index, so the law would be vacuously \
                     true: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.verdict.is_equivalent(),
                    "reopen-recovery law violated for {}: {:?} (divergences {:?})",
                    outcome.replay_signature,
                    outcome.verdict,
                    outcome.report.divergences
                );
                assert!(
                    !outcome.report.subject.hits.is_empty(),
                    "the probe returned no hits, so the observation is empty: {}",
                    outcome.replay_signature
                );
            }
        });
    }

    /// PLANTED-INVALID NEGATIVE against the same live durable path.
    #[test]
    fn a_planted_corpus_mutation_makes_the_reopen_recovery_law_fail() {
        let baseline = maintenance_corpus();
        let mut mutated = maintenance_corpus();
        // Removes `alpha` from doc-4, so the probe's result set changes. A
        // mutation the probe cannot see would not fail, and would prove
        // nothing about the executor's ability to fail.
        mutated[3] = IndexableDocument::new("doc-4", "beta gamma delta saffron");
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline_root = tempfile::tempdir().expect("baseline index directory");
            let maintained_root = tempfile::tempdir().expect("maintained index directory");
            let schedule = reopen_recovery_schedule(MAINTENANCE_SEED_MATRIX[0], baseline.len());
            let outcome = run_reopen_recovery_law(
                &cx,
                recovery_config(),
                baseline_root.path(),
                maintained_root.path(),
                &baseline,
                &mutated,
                &schedule,
                PROBE,
            )
            .await;
            assert!(
                outcome.reopens_executed >= 1,
                "the negative control must exercise the same real recovery: {}",
                outcome.replay_signature
            );
            let LawVerdict::Violated { offending } = &outcome.verdict else {
                panic!(
                    "a mutated corpus must violate the reopen-recovery law, got {:?} with \
                     divergences {:?}",
                    outcome.verdict, outcome.report.divergences
                );
            };
            assert!(
                !offending.is_empty(),
                "a violation must name the classes it rejected"
            );
        });
    }

    /// THE MEASURED BOUNDARY the schedule generator's scope rests on.
    ///
    /// `reopen_recovery_schedule` flushes before every reopen, and its doc
    /// comment justifies that by asserting uncommitted ingest does not survive
    /// a close. This measures it rather than assuming it: the same schedule
    /// with the flush removed loses the buffered batch.
    ///
    /// If this ever stops holding — if uncommitted ingest becomes durable —
    /// this test fails and the generator's qualification must be revisited
    /// rather than silently keeping a restriction it no longer needs.
    #[test]
    fn an_uncommitted_reopen_loses_the_buffered_documents() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let root = tempfile::tempdir().expect("index directory");
            // Ingest everything, reopen WITHOUT flushing, then commit whatever
            // the reopened index still holds.
            let unflushed = MaintenanceSchedule::from_steps_for_test(
                MAINTENANCE_SEED_MATRIX[0],
                documents.len(),
                vec![
                    MaintenanceStep::Ingest {
                        count: documents.len(),
                    },
                    MaintenanceStep::Reopen,
                ],
            );
            let outcome = execute_schedule(
                &cx,
                recovery_config(),
                MaintenanceBacking::Durable(root.path()),
                &documents,
                &unflushed,
            )
            .await;
            assert_eq!(
                outcome.reopen_witnesses,
                vec![0],
                "uncommitted ingest survived a close/reopen; the reopen-recovery generator's \
                 flush-before-reopen qualification is based on this boundary and must be \
                 revisited if it moves"
            );
            assert_eq!(
                outcome.real_reopens(),
                0,
                "a reopen that recovered nothing must not count as a recovery"
            );
        });
    }

    /// EXECUTABLE PRECONDITION, earned by a witnessed recovery.
    #[test]
    fn the_reopen_capability_flip_is_earned_by_a_witnessed_recovery() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let baseline_root = tempfile::tempdir().expect("baseline index directory");
            let maintained_root = tempfile::tempdir().expect("maintained index directory");
            let schedule = reopen_recovery_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let outcome = run_reopen_recovery_law(
                &cx,
                recovery_config(),
                baseline_root.path(),
                maintained_root.path(),
                &documents,
                &documents,
                &schedule,
                PROBE,
            )
            .await;
            assert!(
                outcome.reopens_executed >= 1,
                "no witnessed recovery, so no capability may be declared: {}",
                outcome.replay_signature
            );

            let declared = MaintenanceRunnerCapabilities {
                durable_reopen_lifecycle: true,
                ..MaintenanceRunnerCapabilities::none()
            };
            assert_eq!(
                declared.reopen_recovery(),
                MetamorphicLawApplicability::Applies,
                "a runner with a witnessed recovery must apply the law"
            );
            assert_eq!(
                MaintenanceRunnerCapabilities::none().reopen_recovery(),
                MetamorphicLawApplicability::SkipWithReason {
                    reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable,
                },
                "a runner without the capability must still skip, witness or not"
            );
            assert_eq!(
                declared.merge_schedule(),
                MetamorphicLawApplicability::SkipWithReason {
                    reason: MetamorphicSkipReason::LifecycleCapabilityUnavailable,
                },
                "a reopen witness must not unlock the merge law"
            );
        });
    }
}

/// `e6.3-tombstone-compaction-v1` executed end to end against real deletes and
/// a real compaction, under the score-insensitive projection its precondition
/// demands.
#[cfg(all(test, feature = "perf-harness"))]
mod tombstone_compaction_law_tests {
    use super::maintenance_execution::{maintenance_corpus, merging_config};
    use super::maintenance_law_execution::{run_tombstone_compaction_law, survivors_of};
    use super::{LawVerdict, MaintenanceRunnerCapabilities};
    use crate::metamorphic_maintenance_schedules::{
        MAINTENANCE_SEED_MATRIX, MaintenanceSchedule, MaintenanceStep,
        tombstone_compaction_schedule,
    };
    use crate::runner::{MetamorphicLawApplicability, MetamorphicSkipReason};

    const PROBE: &str = "alpha";

    /// POSITIVE FIXTURE, measured live for every seed in the fixed matrix.
    ///
    /// The control is a corpus that NEVER contained the tombstoned documents,
    /// not the same corpus with deletions replayed onto it. Compaction must
    /// leave the two observationally identical in membership.
    #[test]
    fn the_tombstone_compaction_law_holds_across_the_seed_matrix() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in MAINTENANCE_SEED_MATRIX {
                let schedule = tombstone_compaction_schedule(seed, documents.len());
                let survivors = survivors_of(&documents, &schedule);
                assert!(
                    !survivors.is_empty() && survivors.len() < documents.len(),
                    "seed {seed:#018x} must delete some documents and spare some: {}",
                    schedule.replay_signature()
                );
                let outcome = run_tombstone_compaction_law(
                    &cx,
                    merging_config(),
                    &documents,
                    &survivors,
                    &schedule,
                    PROBE,
                )
                .await;
                assert!(
                    outcome.commits_executed >= 1,
                    "a tombstone needs a committed target: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.compactions_with_work >= 1,
                    "no compaction pass dropped a tombstoned row, so the law would be \
                     vacuously true: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.verdict.is_equivalent(),
                    "tombstone-compaction law violated for {}: {:?} (membership divergences \
                     {:?})",
                    outcome.replay_signature,
                    outcome.verdict,
                    outcome.membership_divergences
                );
            }
        });
    }

    /// THE MEASUREMENT THAT DISCHARGED THIS LAW'S SKIP REASON.
    ///
    /// The registry skips this law for `ScoreSensitiveCorpusStatistics`, and
    /// the note on [`super::TOMBSTONE_COMPACTION_ALLOWED`] predicted that the
    /// total projection would manufacture false failures here. Measurement says
    /// otherwise once COMPACTION is part of the transform: across the whole
    /// seed matrix, a compacted index is identical under the total lexical
    /// observation to a corpus that never contained the deleted documents.
    ///
    /// That is why this law is bound to the total projection rather than to the
    /// score-insensitive membership projection built for the fallback case. The
    /// membership projection is asserted empty too, so a future regression can
    /// be read straight off the pair: membership divergences mean a
    /// resurrection or a loss, total-only divergences mean statistics drifted.
    #[test]
    fn compaction_restores_never_added_statistics() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for seed in MAINTENANCE_SEED_MATRIX {
                let schedule = tombstone_compaction_schedule(seed, documents.len());
                let survivors = survivors_of(&documents, &schedule);
                let outcome = run_tombstone_compaction_law(
                    &cx,
                    merging_config(),
                    &documents,
                    &survivors,
                    &schedule,
                    PROBE,
                )
                .await;
                assert!(
                    outcome.compactions_with_work >= 1,
                    "no compaction pass dropped a row, so this measures nothing: {}",
                    outcome.replay_signature
                );
                assert!(
                    outcome.membership_divergences.is_empty(),
                    "compaction changed which documents are retrievable for {}: {:?}",
                    outcome.replay_signature,
                    outcome.membership_divergences
                );
                assert!(
                    outcome.total_divergences.is_empty(),
                    "compaction did not restore never-added statistics for {}: {:?}. If this \
                     becomes a standing difference, this law must move back to the membership \
                     projection and say so, not widen its allow-list",
                    outcome.replay_signature,
                    outcome.total_divergences
                );
            }
        });
    }

    /// THE CONTROL that keeps the parity above from being a property of the
    /// fixture rather than of compaction.
    ///
    /// Same corpus, same tombstones, same probe — with the `Compact` step
    /// removed. A tombstoned document still occupies the corpus statistics, so
    /// the survivors' scores differ from a never-added corpus and the total
    /// projection diverges. This reproduces the earlier probe on this bead that
    /// saw a survivor's score bits move, and it is what makes
    /// `compaction_restores_never_added_statistics` a statement about
    /// compaction.
    ///
    /// Membership must still agree here: deleting is not losing. If membership
    /// ever diverges in this control, the failure is a real deletion defect and
    /// not the statistics effect this test is about.
    #[test]
    fn without_the_compaction_step_the_total_projection_diverges() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let compacting =
                tombstone_compaction_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let survivors = survivors_of(&documents, &compacting);
            let uncompacted = MaintenanceSchedule::from_steps_for_test(
                compacting.seed(),
                compacting.corpus_len(),
                compacting
                    .steps()
                    .iter()
                    .filter(|step| !matches!(step, MaintenanceStep::Compact))
                    .copied()
                    .collect(),
            );
            let outcome = run_tombstone_compaction_law(
                &cx,
                merging_config(),
                &documents,
                &survivors,
                &uncompacted,
                PROBE,
            )
            .await;
            assert_eq!(
                outcome.compactions_with_work, 0,
                "this control must not compact: {}",
                outcome.replay_signature
            );
            assert!(
                outcome.membership_divergences.is_empty(),
                "deleting without compacting must not change which documents are \
                 retrievable: {:?}",
                outcome.membership_divergences
            );
            assert!(
                !outcome.total_divergences.is_empty(),
                "an uncompacted delete matched never-added statistics exactly for {}, so \
                 compaction is not what produces the parity and the claim that it is must be \
                 withdrawn",
                outcome.replay_signature
            );
        });
    }

    /// PLANTED-INVALID NEGATIVE #1: a RESURRECTED document.
    ///
    /// The control is given the full corpus, as if compaction had brought a
    /// tombstoned document back. This is one of the two failures the law exists
    /// to catch, and it must be caught under either projection — so the
    /// membership divergences are asserted non-empty too, proving the catch
    /// does not depend on the score-bearing half of the observation.
    #[test]
    fn a_resurrected_document_violates_the_tombstone_compaction_law() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let schedule =
                tombstone_compaction_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let outcome = run_tombstone_compaction_law(
                &cx,
                merging_config(),
                &documents,
                // Never-added control that never deleted anything: equivalent
                // to a compaction that resurrected every tombstone.
                &documents,
                &schedule,
                PROBE,
            )
            .await;
            let LawVerdict::Violated { offending } = &outcome.verdict else {
                panic!(
                    "a resurrected document must violate the law, got {:?} with membership \
                     divergences {:?}",
                    outcome.verdict, outcome.membership_divergences
                );
            };
            assert!(
                !offending.is_empty(),
                "a violation must name the classes it rejected"
            );
            assert!(
                !outcome.membership_divergences.is_empty(),
                "a resurrection must be caught by the score-insensitive projection too, or the \
                 catch depends on the score-bearing half of the observation"
            );
        });
    }

    /// PLANTED-INVALID NEGATIVE #2: a LOST survivor.
    ///
    /// The control drops a document the schedule never tombstoned, as if
    /// compaction had discarded a live row. This is the other failure the law
    /// exists to catch, and like the resurrection control it must be caught
    /// without leaning on score bits.
    #[test]
    fn a_lost_survivor_violates_the_tombstone_compaction_law() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let schedule =
                tombstone_compaction_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let mut survivors = survivors_of(&documents, &schedule);
            assert!(
                survivors.len() >= 2,
                "this control needs a survivor it can afford to drop: {}",
                schedule.replay_signature()
            );
            let dropped = survivors.remove(0);
            let outcome = run_tombstone_compaction_law(
                &cx,
                merging_config(),
                &documents,
                &survivors,
                &schedule,
                PROBE,
            )
            .await;
            let LawVerdict::Violated { offending } = &outcome.verdict else {
                panic!(
                    "losing survivor {} must violate the law, got {:?} with membership \
                     divergences {:?}",
                    dropped.id, outcome.verdict, outcome.membership_divergences
                );
            };
            assert!(
                !offending.is_empty(),
                "a violation must name the classes it rejected"
            );
            assert!(
                !outcome.membership_divergences.is_empty(),
                "losing {} must be caught by the score-insensitive projection too",
                dropped.id
            );
        });
    }

    /// EXECUTABLE PRECONDITION, earned by a witnessed compaction under the
    /// projection the precondition names.
    ///
    /// The capability is `compaction_statistics_parity`, and this test only
    /// declares it after a run in which a compaction pass provably dropped rows
    /// AND the law held under the total projection. Declaring it from the
    /// schedule alone would be a false pass with the paperwork filled in.
    #[test]
    fn the_tombstone_capability_flip_is_earned_by_a_witnessed_compaction() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let schedule =
                tombstone_compaction_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let survivors = survivors_of(&documents, &schedule);
            let outcome = run_tombstone_compaction_law(
                &cx,
                merging_config(),
                &documents,
                &survivors,
                &schedule,
                PROBE,
            )
            .await;
            assert!(
                outcome.compactions_with_work >= 1,
                "no witnessed compaction, so no capability may be declared: {}",
                outcome.replay_signature
            );
            assert!(outcome.verdict.is_equivalent());

            let declared = MaintenanceRunnerCapabilities {
                compaction_statistics_parity: true,
                ..MaintenanceRunnerCapabilities::none()
            };
            assert_eq!(
                declared.tombstone_compaction(),
                MetamorphicLawApplicability::Applies,
                "a runner with an approved score-insensitive projection must apply the law"
            );
            assert_eq!(
                MaintenanceRunnerCapabilities::none().tombstone_compaction(),
                MetamorphicLawApplicability::SkipWithReason {
                    reason: MetamorphicSkipReason::ScoreSensitiveCorpusStatistics,
                },
                "a runner without the projection must still skip"
            );
        });
    }
}

/// Live shrink/replay for all three maintenance families.
///
/// The shrinker was proven against synthetic predicates when it landed. That
/// proves the ALGORITHM, not that a failing campaign fixture reduces to
/// something that still reproduces against a real index — which is the only
/// property a replay artifact is actually used for. These tests drive the same
/// [`ShrinkDriver`] the synchronous helper drives, with the live law as the
/// predicate, and then REPLAY the reduced fixture to confirm it still fails.
#[cfg(all(test, feature = "perf-harness"))]
mod live_shrink_replay_tests {
    use super::maintenance_execution::{maintenance_corpus, merging_config, recovery_config};
    use super::maintenance_law_execution::{
        run_merge_schedule_law, run_reopen_recovery_law, run_tombstone_compaction_law, survivors_of,
    };
    use crate::metamorphic_maintenance_schedules::{
        MAINTENANCE_SEED_MATRIX, MaintenanceSchedule, ShrinkDriver, merge_schedule,
        reopen_recovery_schedule, tombstone_compaction_schedule,
    };
    use frankensearch_core::IndexableDocument;

    const PROBE: &str = "alpha";

    /// The corpus whose doc-4 no longer matches the probe, so every law's
    /// comparison against the unmutated control fails.
    fn mutated_corpus() -> Vec<IndexableDocument> {
        let mut documents = maintenance_corpus();
        documents[3] = IndexableDocument::new("doc-4", "beta gamma delta saffron");
        documents
    }

    /// Whether every `Reopen` in a candidate still follows a `Flush`.
    ///
    /// This mirrors the scope qualification on `reopen_recovery_schedule`: the
    /// law is about recovery of committed state, so a fixture that reopens over
    /// buffered ingest is not a smaller reproduction of the same failure.
    fn every_reopen_is_committed(candidate: &MaintenanceSchedule) -> bool {
        candidate
            .steps()
            .iter()
            .enumerate()
            .filter(|(_, step)| {
                matches!(
                    step,
                    crate::metamorphic_maintenance_schedules::MaintenanceStep::Reopen
                )
            })
            .all(|(position, _)| {
                position > 0
                    && matches!(
                        candidate.steps()[position - 1],
                        crate::metamorphic_maintenance_schedules::MaintenanceStep::Flush
                    )
            })
    }

    /// Assertions every shrunk fixture must satisfy, kept in one place so a
    /// family cannot quietly check fewer of them.
    fn assert_bounded_fixture(original: &MaintenanceSchedule, shrunk: &MaintenanceSchedule) {
        assert!(
            shrunk.steps().len() <= original.steps().len(),
            "shrinking must never grow a fixture"
        );
        assert!(
            shrunk.perturbing_steps() >= 1,
            "a shrunk fixture with no perturbation no longer exercises its law: {}",
            shrunk.replay_signature()
        );
        assert_eq!(
            shrunk.ingested(),
            original.ingested(),
            "shrinking must not drop ingest steps and change the corpus"
        );
        // The artifact this emits travels with a bug report, so it must carry
        // structure and no corpus text.
        let signature = shrunk.replay_signature();
        assert!(
            !signature.contains("saffron") && !signature.contains("alpha"),
            "a replay signature must not leak corpus text: {signature}"
        );
    }

    #[test]
    fn a_failing_merge_fixture_shrinks_and_still_reproduces_live() {
        let baseline = maintenance_corpus();
        let mutated = mutated_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let original = merge_schedule(MAINTENANCE_SEED_MATRIX[0], baseline.len());
            let mut driver = ShrinkDriver::new(&original);
            while let Some(candidate) = driver.next_candidate() {
                let outcome = run_merge_schedule_law(
                    &cx,
                    merging_config(),
                    &baseline,
                    &mutated,
                    &candidate,
                    PROBE,
                )
                .await;
                driver.accept(!outcome.verdict.is_equivalent());
            }
            let shrunk = driver.finish();
            assert_bounded_fixture(&original, &shrunk);

            // REPLAY: the reduced fixture must still fail, or the artifact
            // points at a reproduction that does not reproduce.
            let replayed =
                run_merge_schedule_law(&cx, merging_config(), &baseline, &mutated, &shrunk, PROBE)
                    .await;
            assert!(
                !replayed.verdict.is_equivalent(),
                "the shrunk merge fixture stopped reproducing: {}",
                replayed.replay_signature
            );
        });
    }

    #[test]
    fn a_failing_reopen_fixture_shrinks_and_still_reproduces_live() {
        let baseline = maintenance_corpus();
        let mutated = mutated_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let original = reopen_recovery_schedule(MAINTENANCE_SEED_MATRIX[0], baseline.len());
            let mut driver = ShrinkDriver::new(&original);
            while let Some(candidate) = driver.next_candidate() {
                // A reduction that moves a reopen to an uncommitted boundary is
                // OUT OF THIS LAW'S SCOPE, and it would still "fail" -- for a
                // durability reason the law does not name. Accepting it would
                // hand back a replay artifact that reproduces a different bug
                // than the one being shrunk, so the candidate is refused before
                // it is ever executed.
                if !every_reopen_is_committed(&candidate) {
                    driver.accept(false);
                    continue;
                }
                let baseline_root = tempfile::tempdir().expect("baseline index directory");
                let maintained_root = tempfile::tempdir().expect("maintained index directory");
                let outcome = run_reopen_recovery_law(
                    &cx,
                    recovery_config(),
                    baseline_root.path(),
                    maintained_root.path(),
                    &baseline,
                    &mutated,
                    &candidate,
                    PROBE,
                )
                .await;
                driver.accept(!outcome.verdict.is_equivalent());
            }
            let shrunk = driver.finish();
            assert_bounded_fixture(&original, &shrunk);
            assert!(
                every_reopen_is_committed(&shrunk),
                "the shrunk reopen fixture left this law's scope: {}",
                shrunk.replay_signature()
            );

            let baseline_root = tempfile::tempdir().expect("baseline index directory");
            let maintained_root = tempfile::tempdir().expect("maintained index directory");
            let replayed = run_reopen_recovery_law(
                &cx,
                recovery_config(),
                baseline_root.path(),
                maintained_root.path(),
                &baseline,
                &mutated,
                &shrunk,
                PROBE,
            )
            .await;
            assert!(
                !replayed.verdict.is_equivalent(),
                "the shrunk reopen fixture stopped reproducing: {}",
                replayed.replay_signature
            );
        });
    }

    /// The tombstone family shrinks against a RESURRECTION failure — the
    /// control is handed the full corpus as its never-added arm — because that
    /// is the failure mode a compaction bug would actually produce.
    #[test]
    fn a_failing_tombstone_fixture_shrinks_and_still_reproduces_live() {
        let documents = maintenance_corpus();
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let original =
                tombstone_compaction_schedule(MAINTENANCE_SEED_MATRIX[0], documents.len());
            let mut driver = ShrinkDriver::new(&original);
            while let Some(candidate) = driver.next_candidate() {
                let outcome = run_tombstone_compaction_law(
                    &cx,
                    merging_config(),
                    &documents,
                    &documents,
                    &candidate,
                    PROBE,
                )
                .await;
                driver.accept(!outcome.verdict.is_equivalent());
            }
            let shrunk = driver.finish();
            assert_bounded_fixture(&original, &shrunk);
            assert!(
                shrunk.steps().iter().any(|step| matches!(
                    step,
                    crate::metamorphic_maintenance_schedules::MaintenanceStep::Tombstone { .. }
                )),
                "a tombstone fixture that shrank away every deletion cannot reproduce a \
                 resurrection: {}",
                shrunk.replay_signature()
            );

            let replayed = run_tombstone_compaction_law(
                &cx,
                merging_config(),
                &documents,
                &documents,
                &shrunk,
                PROBE,
            )
            .await;
            assert!(
                !replayed.verdict.is_equivalent(),
                "the shrunk tombstone fixture stopped reproducing: {}",
                replayed.replay_signature
            );
            // The survivor set the fixture implies must still be a proper
            // subset, or the "reproduction" is just an empty deletion.
            let survivors = survivors_of(&documents, &shrunk);
            assert!(
                survivors.len() < documents.len(),
                "the shrunk fixture deletes nothing: {}",
                shrunk.replay_signature()
            );
        });
    }
}
