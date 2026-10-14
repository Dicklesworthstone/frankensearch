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
///
/// # Errors
///
/// Returns a typed error for invalid policy, malformed manifest ranges/counts,
/// cancellation, or allocation failure. `Ok(None)` means no eligible pair fits.
pub fn plan_bounded_session_merge(
    cx: &Cx,
    segments: &[ManifestSegment],
    policy: SessionMergePolicy,
) -> Result<Option<SessionMergePlan>, SessionMergeError> {
    plan_bounded_session_merge_with_fit(cx, segments, policy, |range| {
        manifest_concat_fits(&segments[range])
    })
}

fn plan_bounded_session_merge_with_fit(
    cx: &Cx,
    segments: &[ManifestSegment],
    policy: SessionMergePolicy,
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
        let Some(count) = longest_fitting_prefix(cx, start, count, &mut fits)? else {
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

// Both production fit predicates are monotone for a fixed starting entry:
// extending a run cannot reduce its projected layout or reader-budget bounds.
// Check the full candidate first; an oversized candidate needs at most six
// further probes at the maximum fanout of 64. Never probe or select a singleton.
fn longest_fitting_prefix(
    cx: &Cx,
    start: usize,
    count: usize,
    fits: &mut impl FnMut(Range<usize>) -> bool,
) -> Result<Option<usize>, SessionMergeError> {
    debug_assert!((2..=MAX_SESSION_MERGE_FANOUT).contains(&count));
    checkpoint(cx)?;
    if fits(start..start + count) {
        checkpoint(cx)?;
        return Ok(Some(count));
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
            let sources: Vec<_> = (0..8).map(|id| segment(id, 1, bytes)).collect();
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
            let sources = [
                segment(0, 1, u64::MAX),
                segment(1, 1, 1),
                segment(2, 1, 1),
            ];
            let policy = SessionMergePolicy {
                max_input_bytes: u64::MAX,
                ..SessionMergePolicy::default()
            };
            let plan = plan_bounded_session_merge_with_fit(&cx, &sources, policy, |_| true)
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
                    let got = longest_fitting_prefix(&cx, 17, count, &mut |range| {
                        probes += 1;
                        assert_eq!(range.start, 17);
                        assert!((2..=count).contains(&range.len()));
                        range.len() <= limit
                    })
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
