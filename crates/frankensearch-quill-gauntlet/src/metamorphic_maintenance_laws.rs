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
    /// The runner can project observations that do not vary with corpus
    /// statistics, so deleting documents cannot move a survivor's score.
    pub score_insensitive_projection: bool,
}

impl MaintenanceRunnerCapabilities {
    /// The capability set of a runner that declares nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            deterministic_merge_scheduling: false,
            durable_reopen_lifecycle: false,
            score_insensitive_projection: false,
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
    /// Gated on the score-insensitive projection rather than on a lifecycle
    /// operation: the runner can already delete and compact, but under a
    /// score-sensitive projection the comparison is meaningless, because
    /// removing documents legitimately moves every survivor's score. See
    /// [`TOMBSTONE_COMPACTION_ALLOWED`].
    #[must_use]
    pub const fn tombstone_compaction(self) -> MetamorphicLawApplicability {
        if self.score_insensitive_projection {
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
            score_insensitive_projection: true,
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
            score_insensitive_projection: true,
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
pub(crate) mod maintenance_execution {
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::QuillConfig;

    use crate::engine::QuillSubject;

    /// Outcome of driving one ingest schedule to a committed index.
    pub(crate) struct MaintenanceOutcome {
        /// Document ids in ranked order for the probe query.
        pub(crate) ranked_ids: Vec<String>,
        /// Live document count of the committed snapshot.
        pub(crate) doc_count: u64,
        /// Segment count observed after each commit. A merge shows up as a
        /// count that does not grow monotonically with commits.
        pub(crate) sealed_after_each_commit: Vec<usize>,
    }

    /// Ingest `documents` in `batch_size` chunks, committing after each chunk,
    /// then run `query`.
    ///
    /// Committing per chunk is what makes this a maintenance perturbation
    /// rather than a batching one: each commit seals a segment, and once the
    /// sealed count exceeds `tier_fanout` the tier policy performs a real
    /// concat-merge inside that commit.
    pub(crate) async fn ingest_and_probe(
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
        }
    }
}

#[cfg(all(test, feature = "perf-harness"))]
mod merge_execution_tests {
    use super::maintenance_execution::ingest_and_probe;
    use frankensearch_core::IndexableDocument;
    use frankensearch_quill::QuillConfig;

    fn corpus() -> Vec<IndexableDocument> {
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
    fn merging_config() -> QuillConfig {
        QuillConfig {
            tier_fanout: 2,
            ..QuillConfig::default()
        }
    }

    /// e6.3-merge-schedule-v1 EXECUTED AGAINST REAL MERGES.
    ///
    /// The unperturbed arm ingests the whole corpus in one batch and commits
    /// once. The perturbed arm commits per batch under `tier_fanout: 2`, which
    /// drives `apply_tier_policy` into genuine `concat_merge` calls. Merging is
    /// a maintenance decision, so the observation must be unchanged.
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
}
