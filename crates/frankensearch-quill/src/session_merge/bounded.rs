//! Opt-in maintenance for tails or same-tier runs that cannot fit a full fanout.

use std::ops::Range;
use std::sync::Arc;

use asupersync::Cx;

use super::{
    MAX_SESSION_MERGE_FANOUT, SessionMergeError, SessionMergePlan, SessionMergePolicy, checkpoint,
    validate_session_merge,
};
use crate::keeper::{ConcatMergeSizing, KeeperError, manifest_concat_fits};
use crate::{KeeperSnapshot, ManifestSegment, QuillIndex, QuillIndexError};

/// Select a bounded same-row-tier run, including a tail shorter than `fanout`.
///
/// Unlike [`super::plan_session_merge`], this opt-in planner admits between two
/// and `policy.fanout` consecutive entries. It chooses the smallest eligible
/// physical-row tier, then the earliest starting entry, then the longest prefix
/// that fits both input budgets and the conservative durable-layout bound.
/// The policy fanout remains the tier radix even when fewer sources are selected.
/// No intervening manifest entry is skipped and no singleton is rewritten.
///
/// This can make progress when exact-fanout maintenance cannot fit a large
/// run or leaves a short tail. Prefer exact-fanout maintenance during routine
/// ingestion: repeatedly calling this more aggressive policy after tiny batches
/// can increase write amplification. Neither the automatic tier policy nor the
/// existing exact-fanout API changes.
///
/// The complete manifest is validated before selection. Planning does not read
/// files or authorize publication. The manifest-only bound is conservative;
/// [`QuillIndex::merge_sessions_bounded_once`] uses the snapshot's section and
/// term metadata to enforce the actual schema and default reader budgets.
/// The schema-free estimate can decrease when a later source tightens its
/// field-count bound, so every longer prefix is checked before selecting one.
///
/// # Errors
///
/// Returns a typed error for invalid policy, malformed manifest ranges/counts,
/// cancellation, or allocation failure. `Ok(None)` means no eligible run fits.
pub fn plan_bounded_session_merge(
    cx: &Cx,
    segments: &[ManifestSegment],
    policy: SessionMergePolicy,
) -> Result<Option<SessionMergePlan>, SessionMergeError> {
    plan_bounded_session_merge_with_fit(cx, segments, policy, PrefixFitOrder::Arbitrary, |range| {
        manifest_concat_fits(&segments[range])
    })
}

#[derive(Clone, Copy)]
enum PrefixFitOrder {
    // Manifest-only field-count bounds take minima across the sources. Adding
    // a wide, compact source can make a formerly rejected hull fit again.
    Arbitrary,
    // Native sizing fixes the schema and uses increasing hulls/prefix sums.
    // Once a prefix fails those bounds, extending it cannot make it fit.
    Monotone,
}

fn plan_bounded_session_merge_with_fit(
    cx: &Cx,
    segments: &[ManifestSegment],
    policy: SessionMergePolicy,
    fit_order: PrefixFitOrder,
    mut fits: impl FnMut(Range<usize>) -> bool,
) -> Result<Option<SessionMergePlan>, SessionMergeError> {
    validate_session_merge(cx, segments, policy)?;
    let mut selected: Option<(u32, Range<usize>, u64, u64)> = None;
    for start in 0..segments.len().saturating_sub(1) {
        checkpoint(cx)?;
        let tier = policy.tier(segments[start].doc_count);
        if selected.as_ref().is_some_and(|(best, ..)| *best <= tier) {
            continue;
        }
        // Fixed, bounded scratch avoids a per-candidate allocation. It also
        // retains exact input totals if a codec budget requires a shorter run.
        let mut totals = [(0_u64, 0_u64); MAX_SESSION_MERGE_FANOUT + 1];
        let mut count = 0;
        for source in segments[start..].iter().take(policy.fanout) {
            checkpoint(cx)?;
            if policy.tier(source.doc_count) != tier {
                break;
            }
            let Some(docs) = totals[count].0.checked_add(u64::from(source.doc_count)) else {
                break;
            };
            let Some(bytes) = totals[count].1.checked_add(source.file_len) else {
                break;
            };
            if docs > policy.max_input_docs || bytes > policy.max_input_bytes {
                break;
            }
            count += 1;
            totals[count] = (docs, bytes);
        }
        if count < 2 {
            continue;
        }
        let Some(count) = longest_fitting_prefix(cx, start, count, fit_order, &mut fits)? else {
            continue;
        };
        let (docs, bytes) = totals[count];
        selected = Some((tier, start..start + count, docs, bytes));
    }
    // A fit predicate may have observed cancellation while rejecting its last
    // candidate. Never report that as a successful no-op or a usable plan.
    checkpoint(cx)?;
    let Some((tier, range, input_docs, input_bytes)) = selected else {
        return Ok(None);
    };
    let sources = &segments[range];
    let mut source_segment_ids = Vec::new();
    source_segment_ids
        .try_reserve_exact(sources.len())
        .map_err(|_| SessionMergeError::Allocation {
            count: sources.len(),
        })?;
    source_segment_ids.extend(sources.iter().map(|source| source.segment_id));
    checkpoint(cx)?;
    Ok(Some(SessionMergePlan {
        tier,
        source_segment_ids,
        docid_lo: sources[0].docid_lo,
        docid_hi: sources[sources.len() - 1].docid_hi,
        input_docs,
        input_bytes,
    }))
}

// Check the full candidate first. Only native, fixed-schema sizing is monotone
// and permits binary search (at most seven probes at fanout 64). Schema-free
// manifest estimates can alternate between fit/refusal as field bounds tighten;
// scan their prefixes longest-first (at most 63 probes). Neither path probes or
// selects a singleton, and every fit callback is followed by a cancel check.
fn longest_fitting_prefix(
    cx: &Cx,
    start: usize,
    count: usize,
    fit_order: PrefixFitOrder,
    fits: &mut impl FnMut(Range<usize>) -> bool,
) -> Result<Option<usize>, SessionMergeError> {
    debug_assert!((2..=MAX_SESSION_MERGE_FANOUT).contains(&count));
    checkpoint(cx)?;
    if fits(start..start + count) {
        checkpoint(cx)?;
        return Ok(Some(count));
    }
    if matches!(fit_order, PrefixFitOrder::Arbitrary) {
        for end in (2..count).rev() {
            checkpoint(cx)?;
            let accepted = fits(start..start + end);
            checkpoint(cx)?;
            if accepted {
                return Ok(Some(end));
            }
        }
        checkpoint(cx)?;
        return Ok(None);
    }
    let mut accepted = 1;
    let mut rejected = count;
    while rejected - accepted > 1 {
        checkpoint(cx)?;
        let middle = accepted + (rejected - accepted) / 2;
        if fits(start..start + middle) {
            accepted = middle;
        } else {
            rejected = middle;
        }
    }
    checkpoint(cx)?;
    Ok((accepted >= 2).then_some(accepted))
}

impl QuillIndex {
    /// Publish at most one bounded same-tier session merge, including short tails.
    ///
    /// This is the opt-in execution counterpart of [`plan_bounded_session_merge`].
    /// It can select fewer than `policy.fanout` sources when the full run exceeds
    /// input, format, or default term-reader budgets. No reader limits, durable
    /// formats, or automatic merge policies are relaxed.
    ///
    /// Call after an explicit commit, between ingest batches. The existing
    /// writer-locked [`Self::concat_merge`] revalidates source identities and
    /// consecutiveness, retires live leases before covering their gaps, and owns
    /// publication and recovery. For a selected run, uncommitted writes and
    /// output-ID collisions propagate from concat. Structural source errors
    /// are never treated as budget skips. Output identity and timestamp have
    /// the same caller-owned contract as concat.
    ///
    /// # Errors
    ///
    /// Returns planning, snapshot, or production concat errors. As with
    /// [`Self::merge_sessions_once`], reconcile publication failures before
    /// retrying; an error does not prove that publication did not occur.
    pub async fn merge_sessions_bounded_once(
        &self,
        cx: &Cx,
        policy: SessionMergePolicy,
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<Option<Arc<KeeperSnapshot>>, SessionMergeError> {
        checkpoint(cx)?;
        policy.validate()?;
        let snapshot = self.snapshot()?;
        // Keep global structural-source validation fail-closed. Only sizing
        // failures from check() are eligible for bounded candidate selection.
        let sizing = ConcatMergeSizing::new(snapshot.schema(), snapshot.segments())
            .map_err(KeeperError::from)
            .map_err(QuillIndexError::from)?;
        let plan = plan_bounded_session_merge_with_fit(
            cx,
            &snapshot.loaded_manifest().manifest.segments,
            policy,
            PrefixFitOrder::Monotone,
            |range| sizing.check(range).is_ok(),
        )?;
        drop(snapshot);
        let Some(plan) = plan else {
            return Ok(None);
        };
        checkpoint(cx)?;
        let published = self
            .concat_merge(
                cx,
                &plan.source_segment_ids,
                output_segment_id,
                created_unix_s,
            )
            .await?;
        // As in the exact-fanout path, never relabel durable success as a late
        // cancellation. The writer-locked publication path owns that boundary.
        Ok(Some(published))
    }
}

#[cfg(test)]
mod tests {
    use asupersync::test_utils::run_test_with_cx;

    use super::*;
    use crate::{TombstoneSet, plan_session_merge};

    fn segment(id: u64, docs: u32, bytes: u64) -> ManifestSegment {
        let lo = id * 65_536;
        ManifestSegment {
            segment_id: id,
            seal_seq: id,
            file_len: bytes,
            file_xxh3: 0,
            docid_lo: lo,
            docid_hi: lo + u64::from(docs).max(1),
            doc_count: docs,
            tombstones: TombstoneSet::new(),
        }
    }

    // Sizing-only manifests: large declared files/spans do not allocate their
    // bytes. A wide source tightens the shared schema-field upper bound.
    fn layout_segment(id: u64, lo: u64, span: u64, bytes: u64) -> ManifestSegment {
        let mut source = segment(id, 1, bytes);
        source.docid_lo = lo;
        source.docid_hi = lo + span;
        source
    }

    #[test]
    fn manifest_bound_recovery_keeps_the_earliest_maximal_run() {
        run_test_with_cx(|cx| async move {
            let gap = 1_u64 << 20;
            let small = 16 * 1024;
            let sources = [
                layout_segment(1, 0, 1, small),
                layout_segment(2, gap, 1, small),
                layout_segment(3, gap + 1, 1, small),
                layout_segment(4, gap + 2, 1, small),
                layout_segment(5, gap + 3, gap, 40 * 1024 * 1024),
                layout_segment(6, 2 * gap + 3, 1, small),
                layout_segment(7, 2 * gap + 4, 1, small),
                layout_segment(8, 1 << 27, 1, small),
            ];
            // This is the real production manifest bound, not a mocked fit
            // callback. Binary search probes 8, 4, 2 and misses prefixes 5..7.
            assert_eq!(
                (2..=sources.len())
                    .map(|end| manifest_concat_fits(&sources[..end]))
                    .collect::<Vec<_>>(),
                [false, false, false, true, true, true, false],
            );
            let plan = plan_bounded_session_merge(&cx, &sources, SessionMergePolicy::default())
                .unwrap()
                .expect("a recovered prefix at the earliest start must not be skipped");
            assert_eq!(plan.source_segment_ids, [1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(plan.tier, 0);
            assert_eq!(plan.input_docs, 7);
            assert_eq!(plan.input_bytes, 40 * 1024 * 1024 + 6 * small);
            assert_eq!((plan.docid_lo, plan.docid_hi), (0, 2 * gap + 5));
        });
    }

    #[test]
    fn manifest_bound_recovery_does_not_truncate_at_an_earlier_fit() {
        run_test_with_cx(|cx| async move {
            let gap = 1_u64 << 20;
            let small = 16 * 1024;
            let sources = [
                layout_segment(1, 0, 1, small),
                layout_segment(2, 1, 1, small),
                layout_segment(3, gap, 1, small),
                layout_segment(4, gap + 1, gap, 40 * 1024 * 1024),
                layout_segment(5, 1 << 27, 1, small),
            ];
            assert_eq!(
                (2..=sources.len())
                    .map(|end| manifest_concat_fits(&sources[..end]))
                    .collect::<Vec<_>>(),
                [true, false, true, false],
            );
            let plan = plan_bounded_session_merge(&cx, &sources, SessionMergePolicy::default())
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, [1, 2, 3, 4]);
            assert_eq!(plan.input_docs, 4);
            assert_eq!(plan.input_bytes, 40 * 1024 * 1024 + 3 * small);
            assert_eq!((plan.docid_lo, plan.docid_hi), (0, 2 * gap + 1));
        });
    }

    #[test]
    fn arbitrary_prefix_search_matches_every_small_acceptance_mask() {
        run_test_with_cx(|cx| async move {
            for count in 2..=12 {
                for mask in 0..(1_u64 << (count - 1)) {
                    let expected = (2..=count)
                        .filter(|&end| mask & (1_u64 << (end - 2)) != 0)
                        .max();
                    let mut probes = Vec::new();
                    let got = longest_fitting_prefix(
                        &cx,
                        17,
                        count,
                        PrefixFitOrder::Arbitrary,
                        &mut |range| {
                            assert_eq!(range.start, 17);
                            assert!((2..=count).contains(&range.len()));
                            probes.push(range.len());
                            mask & (1_u64 << (range.len() - 2)) != 0
                        },
                    )
                    .unwrap();
                    assert_eq!(got, expected, "count={count}, mask={mask:#x}");
                    assert_eq!(
                        probes.len(),
                        expected.map_or(count - 1, |end| count - end + 1)
                    );
                    assert!(probes.windows(2).all(|pair| pair[0] > pair[1]));
                }
            }
        });
    }

    #[test]
    fn arbitrary_prefix_search_covers_every_fanout_without_probing_singletons() {
        run_test_with_cx(|cx| async move {
            for count in 2..=MAX_SESSION_MERGE_FANOUT {
                for only_fit in 1..=count {
                    let mut probes = 0;
                    let got = longest_fitting_prefix(
                        &cx,
                        17,
                        count,
                        PrefixFitOrder::Arbitrary,
                        &mut |range| {
                            assert_eq!(range.start, 17);
                            assert!((2..=count).contains(&range.len()));
                            probes += 1;
                            range.len() == only_fit
                        },
                    )
                    .unwrap();
                    assert_eq!(got, (only_fit >= 2).then_some(only_fit));
                    assert!(probes < count);
                }
            }
        });
    }

    #[test]
    fn arbitrary_prefix_cancellation_stops_after_any_accepted_or_rejected_probe() {
        for count in [2, 8, MAX_SESSION_MERGE_FANOUT] {
            for cancel_on in 1..count {
                for accepted in [false, true] {
                    run_test_with_cx(move |cx| async move {
                        let mut probes = 0;
                        let result = longest_fitting_prefix(
                            &cx,
                            17,
                            count,
                            PrefixFitOrder::Arbitrary,
                            &mut |_| {
                                probes += 1;
                                if probes == cancel_on {
                                    cx.cancel_fast(asupersync::CancelKind::User);
                                    return accepted;
                                }
                                false
                            },
                        );
                        assert!(matches!(result, Err(SessionMergeError::Cancelled)));
                        assert_eq!(probes, cancel_on);
                    });
                }
            }
        }
    }

    #[test]
    fn bounded_tail_is_opt_in_and_keeps_the_policy_tier_radix() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy::default();
            let sources: Vec<_> = (0..3).map(|id| segment(id, 3, 1024)).collect();
            assert!(plan_session_merge(&cx, &sources, policy).unwrap().is_none());
            let plan = plan_bounded_session_merge(&cx, &sources, policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, [0, 1, 2]);
            assert_eq!(plan.tier, 0);
            assert_eq!((plan.input_docs, plan.input_bytes), (9, 3072));
            assert_eq!((plan.docid_lo, plan.docid_hi), (0, 2 * 65_536 + 3));
        });
    }

    #[test]
    fn full_fanout_matches_existing_planner_and_remains_a_hard_cap() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy::default();
            let sources: Vec<_> = (0..12).map(|id| segment(id, 3, 1024)).collect();
            let bounded = plan_bounded_session_merge(&cx, &sources, policy).unwrap();
            assert_eq!(bounded, plan_session_merge(&cx, &sources, policy).unwrap());
            assert_eq!(bounded.unwrap().source_segment_ids.len(), 8);
        });
    }

    #[test]
    fn input_limits_shrink_a_run_and_totals_describe_only_selected_sources() {
        run_test_with_cx(|cx| async move {
            let sources: Vec<_> = (0..8).map(|id| segment(id, 3, 1024)).collect();
            for (docs, bytes, count) in [(9, 8192, 3), (24, 2048, 2), (6, 2048, 2)] {
                let policy = SessionMergePolicy {
                    max_input_docs: docs,
                    max_input_bytes: bytes,
                    ..SessionMergePolicy::default()
                };
                assert!(plan_session_merge(&cx, &sources, policy).unwrap().is_none());
                let plan = plan_bounded_session_merge(&cx, &sources, policy)
                    .unwrap()
                    .unwrap();
                assert_eq!(plan.source_segment_ids.len(), count);
                assert_eq!(plan.input_docs, count as u64 * 3);
                assert_eq!(plan.input_bytes, count as u64 * 1024);
                assert_eq!(plan.docid_hi, sources[count - 1].docid_hi);
            }
        });
    }

    #[test]
    fn codec_budget_selects_the_longest_compatible_prefix() {
        run_test_with_cx(|cx| async move {
            let sources: Vec<_> = (0..8).map(|id| segment(id, 3, 1024)).collect();
            let plan = plan_bounded_session_merge_with_fit(
                &cx,
                &sources,
                SessionMergePolicy::default(),
                PrefixFitOrder::Monotone,
                |range| range.len() <= 3,
            )
            .unwrap()
            .unwrap();
            assert_eq!(plan.source_segment_ids, [0, 1, 2]);
            assert_eq!((plan.input_docs, plan.input_bytes), (9, 3072));
        });
    }

    #[test]
    fn real_manifest_layout_budget_still_applies_with_unlimited_inputs() {
        run_test_with_cx(|cx| async move {
            let bytes = u64::from(u32::MAX) / 3 + 1;
            // Contiguous sources: no holes, so the u32 file-length budget is
            // the only bound (two fit, three do not). `segment` spaces ids
            // 65,536 apart, and that many holes alone exceed the layout.
            let sources: Vec<_> = (0..8).map(|id| layout_segment(id, id, 1, bytes)).collect();
            let policy = SessionMergePolicy {
                max_input_docs: u64::MAX,
                max_input_bytes: u64::MAX,
                ..SessionMergePolicy::default()
            };
            assert!(plan_session_merge(&cx, &sources, policy).unwrap().is_none());
            let plan = plan_bounded_session_merge(&cx, &sources, policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, [0, 1]);
            assert_eq!(plan.input_bytes, bytes * 2);
        });
    }

    #[test]
    fn an_unmergeable_entry_is_not_a_bridge_between_small_sources() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                max_input_bytes: 2048,
                ..SessionMergePolicy::default()
            };
            let mut sources = vec![
                segment(0, 1, 1024),
                segment(1, 1, 2049),
                segment(2, 1, 1024),
            ];
            assert!(
                plan_bounded_session_merge(&cx, &sources, policy)
                    .unwrap()
                    .is_none()
            );
            sources.push(segment(3, 1, 1024));
            assert_eq!(
                plan_bounded_session_merge(&cx, &sources, policy)
                    .unwrap()
                    .unwrap()
                    .source_segment_ids,
                [2, 3],
            );
        });
    }

    #[test]
    fn smallest_tier_then_earliest_start_precedes_later_longer_runs() {
        run_test_with_cx(|cx| async move {
            let sources = [
                segment(0, 8, 1024),
                segment(1, 8, 1024),
                segment(2, 3, 1024),
                segment(3, 3, 1024),
                segment(4, 8, 1024),
                segment(5, 3, 1024),
                segment(6, 3, 1024),
                segment(7, 3, 1024),
            ];
            let plan = plan_bounded_session_merge(&cx, &sources, SessionMergePolicy::default())
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, [2, 3]);
            assert_eq!(plan.tier, 0);
        });
    }

    #[test]
    fn tombstones_and_retired_lease_gaps_do_not_change_physical_row_accounting() {
        run_test_with_cx(|cx| async move {
            let mut sources: Vec<_> = (0..3).map(|id| segment(id, 3, 1024)).collect();
            sources[0].tombstones.insert(0).unwrap();
            sources[0].tombstones.insert(1).unwrap();
            let policy = SessionMergePolicy {
                max_input_docs: 6,
                ..SessionMergePolicy::default()
            };
            let plan = plan_bounded_session_merge(&cx, &sources, policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, [0, 1]);
            assert_eq!(plan.input_docs, 6);
            assert_eq!(plan.docid_hi, 65_536 + 3);
        });
    }

    #[test]
    fn input_sum_overflow_cannot_wrap_into_a_small_budget() {
        run_test_with_cx(|cx| async move {
            let sources = [segment(0, 1, u64::MAX), segment(1, 1, 1), segment(2, 1, 1)];
            let policy = SessionMergePolicy {
                max_input_bytes: u64::MAX,
                ..SessionMergePolicy::default()
            };
            let plan = plan_bounded_session_merge_with_fit(
                &cx,
                &sources,
                policy,
                PrefixFitOrder::Monotone,
                |_| true,
            )
            .unwrap()
            .unwrap();
            assert_eq!(plan.source_segment_ids, [1, 2]);
            assert_eq!(plan.input_bytes, 2);
        });
    }

    #[test]
    fn malformed_later_source_is_not_hidden_by_an_earlier_candidate() {
        run_test_with_cx(|cx| async move {
            let mut sources: Vec<_> = (0..3).map(|id| segment(id, 1, 1024)).collect();
            sources[2].docid_lo = sources[1].docid_lo;
            assert!(matches!(
                plan_bounded_session_merge(&cx, &sources, SessionMergePolicy::default()),
                Err(SessionMergeError::InvalidSegment { .. })
            ));
        });
    }

    #[test]
    fn invalid_policy_and_singletons_cannot_become_bounded_merges() {
        run_test_with_cx(|cx| async move {
            for policy in [
                SessionMergePolicy {
                    fanout: 0,
                    ..SessionMergePolicy::default()
                },
                SessionMergePolicy {
                    fanout: 1,
                    ..SessionMergePolicy::default()
                },
                SessionMergePolicy {
                    fanout: 65,
                    ..SessionMergePolicy::default()
                },
                SessionMergePolicy {
                    max_input_docs: 0,
                    ..SessionMergePolicy::default()
                },
                SessionMergePolicy {
                    max_input_bytes: 0,
                    ..SessionMergePolicy::default()
                },
            ] {
                assert!(matches!(
                    plan_bounded_session_merge(&cx, &[], policy),
                    Err(SessionMergeError::InvalidPolicy { .. })
                ));
            }
            for sources in [vec![], vec![segment(0, 1, 1)], vec![segment(0, 0, 0)]] {
                assert!(
                    plan_bounded_session_merge(&cx, &sources, SessionMergePolicy::default())
                        .unwrap()
                        .is_none()
                );
            }
        });
    }

    #[test]
    fn every_prefix_boundary_is_maximal_and_uses_at_most_seven_fit_probes() {
        run_test_with_cx(|cx| async move {
            for count in 2..=MAX_SESSION_MERGE_FANOUT {
                for limit in 0..=count {
                    let mut probes = 0;
                    let got = longest_fitting_prefix(
                        &cx,
                        17,
                        count,
                        PrefixFitOrder::Monotone,
                        &mut |range| {
                            probes += 1;
                            assert_eq!(range.start, 17);
                            assert!((2..=count).contains(&range.len()));
                            range.len() <= limit
                        },
                    )
                    .unwrap();
                    assert_eq!(got, (limit >= 2).then_some(limit));
                    assert!(probes <= 7);
                }
            }
        });
    }

    #[test]
    fn cancellation_before_planning_is_not_a_noop() {
        run_test_with_cx(|cx| async move {
            cx.cancel_fast(asupersync::CancelKind::User);
            assert!(matches!(
                plan_bounded_session_merge(&cx, &[], SessionMergePolicy::default()),
                Err(SessionMergeError::Cancelled)
            ));
        });
    }

    #[test]
    fn cancellation_during_successful_fit_is_not_a_plan() {
        run_test_with_cx(|cx| async move {
            let sources = [segment(0, 1, 1), segment(1, 1, 1)];
            assert!(matches!(
                plan_bounded_session_merge_with_fit(
                    &cx,
                    &sources,
                    SessionMergePolicy::default(),
                    PrefixFitOrder::Monotone,
                    |_| {
                        cx.cancel_fast(asupersync::CancelKind::User);
                        true
                    },
                ),
                Err(SessionMergeError::Cancelled)
            ));
        });
    }

    #[test]
    fn cancellation_during_rejected_fit_is_not_nothing_to_merge() {
        run_test_with_cx(|cx| async move {
            let sources = [segment(0, 1, 1), segment(1, 1, 1)];
            assert!(matches!(
                plan_bounded_session_merge_with_fit(
                    &cx,
                    &sources,
                    SessionMergePolicy::default(),
                    PrefixFitOrder::Monotone,
                    |_| {
                        cx.cancel_fast(asupersync::CancelKind::User);
                        false
                    },
                ),
                Err(SessionMergeError::Cancelled)
            ));
        });
    }

    #[test]
    fn codec_refusal_never_bridges_an_unmergeable_source() {
        run_test_with_cx(|cx| async move {
            let sources: Vec<_> = (0..4).map(|id| segment(id, 1, 1024)).collect();
            let plan = plan_bounded_session_merge_with_fit(
                &cx,
                &sources,
                SessionMergePolicy::default(),
                PrefixFitOrder::Monotone,
                |range| !range.contains(&1),
            )
            .unwrap()
            .unwrap();
            assert_eq!(plan.source_segment_ids, [2, 3]);
        });
    }

    #[test]
    fn cancellation_during_prefix_search_stops_further_probes() {
        run_test_with_cx(|cx| async move {
            let sources: Vec<_> = (0..8).map(|id| segment(id, 1, 1024)).collect();
            let mut probes = 0;
            let result = plan_bounded_session_merge_with_fit(
                &cx,
                &sources,
                SessionMergePolicy::default(),
                PrefixFitOrder::Monotone,
                |_| {
                    probes += 1;
                    if probes == 2 {
                        cx.cancel_fast(asupersync::CancelKind::User);
                    }
                    false
                },
            );
            assert!(matches!(result, Err(SessionMergeError::Cancelled)));
            assert_eq!(probes, 2);
        });
    }
}

#[cfg(all(test, unix))]
mod publication_tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{IndexableDocument, LexicalWrite};

    use super::*;
    use crate::{
        Manifest, QuillConfig, QuillSearchIndex, QuillSearchResult, QuillSearchSnapshot,
        load_manifest_pair,
    };

    const QUERIES: [&str; 3] = ["shared", "\"amber rust\"", "amber OR cobalt"];
    type StoredRows = Vec<Vec<Option<Vec<u8>>>>;

    fn config() -> QuillConfig {
        QuillConfig {
            deterministic_ingest: true,
            ..QuillConfig::default()
        }
    }

    async fn fixture(cx: &Cx, directory: &Path) -> QuillIndex {
        let index = QuillIndex::create(cx, directory, config())
            .await
            .expect("create bounded-session fixture");
        for batch in 0..3 {
            // The newest source replaces a row in the oldest source. A newly
            // merged container must not mask the live replacement. Each source
            // is one ingest batch: a separate call that replaces a published
            // document first publishes the pending rows (the engine-neutral
            // LexicalWrite contract), which would split the source in two.
            let rows: Vec<_> = (0..2)
                .map(|row| {
                    let id = if batch == 2 && row == 1 {
                        "bounded-0-0".to_owned()
                    } else {
                        format!("bounded-{batch}-{row}")
                    };
                    let text = if row == 0 {
                        "shared amber rust"
                    } else {
                        "shared cobalt python"
                    };
                    IndexableDocument::new(id.as_str(), text)
                })
                .collect();
            LexicalWrite::index_documents(&index, cx, &rows)
                .await
                .expect("stage fixture rows");
            LexicalWrite::commit(&index, cx)
                .await
                .expect("seal a two-row source");
        }
        let snapshot = index.snapshot().expect("fixture authority");
        assert_eq!(snapshot.segments().len(), 3);
        assert_eq!(snapshot.doc_count(), 5);
        assert_eq!(
            snapshot
                .loaded_manifest()
                .manifest
                .segments
                .iter()
                .map(|segment| u64::from(segment.doc_count))
                .sum::<u64>(),
            6,
        );
        assert_eq!(index.collect_docids(cx, "shared").unwrap(), [1, 2, 3, 4, 5]);
        index
    }

    fn queries(index: &QuillIndex, cx: &Cx) -> Vec<QuillSearchResult> {
        QUERIES
            .iter()
            .map(|query| index.search_paginated(cx, query, 20, 0, true).unwrap())
            .collect()
    }

    fn stored(snapshot: &QuillSearchSnapshot) -> StoredRows {
        (0..6)
            .map(|docid| {
                (0..5)
                    .map(|field| snapshot.stored_field_value(field, docid).unwrap())
                    .collect()
            })
            .collect()
    }

    async fn assert_reopened(
        cx: &Cx,
        directory: &Path,
        expected_queries: &[QuillSearchResult],
        expected_stored: &[Vec<Option<Vec<u8>>>],
    ) {
        let reader = QuillSearchIndex::open(cx, directory, config())
            .await
            .expect("cold reader opens published authority");
        for (query, expected) in QUERIES.iter().zip(expected_queries) {
            assert_eq!(
                reader.search_paginated(cx, query, 20, 0, true).unwrap(),
                *expected,
                "cold query {query} preserves counts, scores, identities, and ordering",
            );
        }
        for (docid, fields) in expected_stored.iter().enumerate() {
            for (field, expected) in fields.iter().enumerate() {
                let actual = reader
                    .stored_field_value(
                        u16::try_from(field).unwrap(),
                        u32::try_from(docid).unwrap(),
                    )
                    .unwrap();
                assert_eq!(actual.as_deref(), expected.as_deref());
            }
        }
    }

    fn directory_bytes(directory: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap())
            .filter(|entry| entry.file_type().unwrap().is_file())
            .map(|entry| {
                let path = entry.path();
                let bytes = std::fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect()
    }

    #[test]
    fn real_bounded_tail_preserves_upserts_stored_fields_search_and_continued_ingest() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let index = fixture(&cx, directory.path()).await;
            let initial = load_manifest_pair(directory.path()).unwrap().manifest;
            let expected_queries = queries(&index, &cx);
            let held = index.search_snapshot().unwrap();
            let expected_stored = stored(&held);
            assert!(expected_stored[0].iter().all(Option::is_none));
            assert_eq!(expected_queries[0].total_count, Some(5));
            let policy = SessionMergePolicy {
                max_input_docs: 4,
                max_input_bytes: u64::MAX,
                ..SessionMergePolicy::default()
            };
            assert!(
                index
                    .merge_sessions_once(&cx, policy, 0xb000, 0)
                    .await
                    .unwrap()
                    .is_none()
            );
            let merged = index
                .merge_sessions_bounded_once(&cx, policy, 0xb001, 0)
                .await
                .unwrap()
                .expect("merge a compatible pair despite a fanout of eight");
            assert_eq!(merged.segments().len(), 2);
            assert_eq!(merged.doc_count(), 5);
            assert_eq!(
                merged.loaded_manifest().manifest.field_stats,
                initial.field_stats
            );
            assert_eq!(
                merged.loaded_manifest().manifest.segments[1],
                initial.segments[2]
            );
            assert_eq!(queries(&index, &cx), expected_queries);
            assert_eq!(stored(&held), expected_stored);
            assert_eq!(stored(&index.search_snapshot().unwrap()), expected_stored);
            assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;

            let before_noop = directory_bytes(directory.path());
            assert!(
                index
                    .merge_sessions_bounded_once(&cx, policy, 0xb002, 0)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(directory_bytes(directory.path()), before_noop);
            let merged = index
                .merge_sessions_bounded_once(
                    &cx,
                    SessionMergePolicy {
                        max_input_docs: 6,
                        ..policy
                    },
                    0xb003,
                    0,
                )
                .await
                .unwrap()
                .expect("a raised input budget admits the remaining two-source tail");
            assert_eq!(merged.segments().len(), 1);
            assert_eq!(merged.doc_count(), 5);
            assert_eq!(
                merged.loaded_manifest().manifest.field_stats,
                initial.field_stats
            );
            assert_eq!(queries(&index, &cx), expected_queries);
            assert_eq!(stored(&index.search_snapshot().unwrap()), expected_stored);
            assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;

            LexicalWrite::index_document(
                &index,
                &cx,
                &IndexableDocument::new("after-bounded", "shared additional document"),
            )
            .await
            .unwrap();
            LexicalWrite::commit(&index, &cx).await.unwrap();
            assert_eq!(index.doc_count().unwrap(), 6);
            assert_eq!(
                index.collect_docids(&cx, "shared").unwrap(),
                [1, 2, 3, 4, 5, 6]
            );
            assert_eq!(
                stored(&held),
                expected_stored,
                "held authority remains unchanged"
            );
            let reopened = QuillSearchIndex::open(&cx, directory.path(), config())
                .await
                .unwrap();
            assert_eq!(
                reopened
                    .search_paginated(&cx, "shared", 20, 0, true)
                    .unwrap(),
                index.search_paginated(&cx, "shared", 20, 0, true).unwrap(),
            );
        });
    }

    #[test]
    fn bounded_publication_failure_keeps_old_authority_and_survives_writer_reopen() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let index = fixture(&cx, directory.path()).await;
            // Start with no live ingest lease, as after a process restart.
            drop(index);
            let index = QuillIndex::open(&cx, directory.path(), config())
                .await
                .unwrap();
            let initial = load_manifest_pair(directory.path()).unwrap().manifest;
            let expected_queries = queries(&index, &cx);
            let held = index.search_snapshot().unwrap();
            let expected_stored = stored(&held);
            let policy = SessionMergePolicy {
                max_input_docs: 4,
                max_input_bytes: u64::MAX,
                ..SessionMergePolicy::default()
            };
            let blocked_temp = directory
                .path()
                .join(format!(".tmp-manifest-{}", initial.generation + 1));
            let blocker = b"preserve this conflicting proposal";
            std::fs::write(&blocked_temp, blocker).unwrap();
            let error = index
                .merge_sessions_bounded_once(&cx, policy, 0xb0ff, 0)
                .await
                .err()
                .expect("fail after output installation but before manifest promotion");
            assert!(
                error.to_string().contains("temp"),
                "unexpected error: {error}"
            );
            let output_path = directory.path().join("seg-000000000000b0ff.fslx");
            let output =
                std::fs::read(&output_path).expect("installed output exists but is unreachable");
            assert_eq!(std::fs::read(&blocked_temp).unwrap(), blocker);
            assert_eq!(
                load_manifest_pair(directory.path()).unwrap().manifest,
                initial
            );
            assert_eq!(stored(&held), expected_stored);
            assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;
            for source in &initial.segments {
                assert!(
                    directory
                        .path()
                        .join(format!("seg-{:016x}.fslx", source.segment_id))
                        .is_file()
                );
            }

            drop(index);
            std::fs::rename(
                &blocked_temp,
                directory.path().join("preserved-proposal.fixture"),
            )
            .unwrap();
            let recovered = QuillIndex::open(&cx, directory.path(), config())
                .await
                .unwrap();
            assert_eq!(
                recovered.snapshot().unwrap().loaded_manifest().manifest,
                initial
            );
            let merged = recovered
                .merge_sessions_bounded_once(&cx, policy, 0xb0ff, 0)
                .await
                .unwrap()
                .expect("recovered writer can retry the same bounded merge");
            assert_eq!(merged.segments().len(), 2);
            assert_eq!(merged.doc_count(), 5);
            assert_eq!(std::fs::read(&output_path).unwrap(), output);
            assert_eq!(queries(&recovered, &cx), expected_queries);
            assert_eq!(stored(&held), expected_stored);
            assert_eq!(
                stored(&recovered.search_snapshot().unwrap()),
                expected_stored
            );
            assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;
            assert_eq!(
                Manifest::from_bytes(
                    &std::fs::read(directory.path().join("MANIFEST.prev")).unwrap()
                )
                .unwrap(),
                initial,
            );
        });
    }

    #[test]
    fn bounded_noop_cancellation_and_output_collision_leave_authority_unchanged() {
        run_test_with_cx(|cx| async move {
            let directory = tempfile::tempdir().unwrap();
            let index = fixture(&cx, directory.path()).await;
            let initial = load_manifest_pair(directory.path()).unwrap().manifest;
            let expected_queries = queries(&index, &cx);
            let expected_stored = stored(&index.search_snapshot().unwrap());
            let before = directory_bytes(directory.path());
            // The first pair has three live rows, but four physical rows:
            // tombstones must not make it pass a three-row concat budget.
            let policy = SessionMergePolicy {
                max_input_docs: 3,
                ..SessionMergePolicy::default()
            };
            assert!(
                index
                    .merge_sessions_bounded_once(&cx, policy, 0xb010, 0)
                    .await
                    .unwrap()
                    .is_none()
            );
            assert_eq!(directory_bytes(directory.path()), before);

            cx.set_cancel_requested(true);
            let cancelled = index
                .merge_sessions_bounded_once(&cx, SessionMergePolicy::default(), 0xb011, 0)
                .await;
            cx.set_cancel_requested(false);
            assert!(matches!(cancelled, Err(SessionMergeError::Cancelled)));
            assert_eq!(directory_bytes(directory.path()), before);
            assert!(
                index
                    .merge_sessions_bounded_once(
                        &cx,
                        SessionMergePolicy::default(),
                        initial.segments[0].segment_id,
                        0,
                    )
                    .await
                    .is_err()
            );
            assert_eq!(directory_bytes(directory.path()), before);
            assert_eq!(
                load_manifest_pair(directory.path()).unwrap().manifest,
                initial
            );
            assert_eq!(queries(&index, &cx), expected_queries);
            assert_reopened(&cx, directory.path(), &expected_queries, &expected_stored).await;
        });
    }
}
