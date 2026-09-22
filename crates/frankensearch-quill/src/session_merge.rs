//! Bounded concat maintenance for indexes accumulated across short writer sessions.
//!
//! A retired lease is address space, not indexed data. This opt-in policy tiers
//! segments by physical row count instead of covering-interval width, and bounds
//! the actual input file bytes. It never changes the ordinary tier policy or the
//! Q1 requirement to merge an uninterrupted slice of the current manifest.

use std::sync::Arc;

use asupersync::Cx;
use thiserror::Error;

use crate::{KeeperSnapshot, ManifestSegment, QuillIndex, QuillIndexError};

/// Maximum number of source entries examined in one candidate window.
pub const MAX_SESSION_MERGE_FANOUT: usize = 64;

/// Resource limits for one explicit session-fragmentation maintenance step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionMergePolicy {
    /// Consecutive segments of the same row-count tier required for a merge.
    /// Must be in `2..=MAX_SESSION_MERGE_FANOUT`.
    pub fanout: usize,
    /// Maximum total physical rows in a selected run, including tombstoned rows.
    pub max_input_docs: u64,
    /// Maximum sum of source FSLX file lengths. This is an input-I/O budget,
    /// not a hard memory ceiling or an upper bound on output-file size.
    pub max_input_bytes: u64,
}

impl Default for SessionMergePolicy {
    fn default() -> Self {
        Self {
            fanout: 8,
            max_input_docs: 1_000_000,
            max_input_bytes: 256 * 1024 * 1024,
        }
    }
}

impl SessionMergePolicy {
    /// Reject invalid limits before taking a snapshot or doing maintenance.
    ///
    /// # Errors
    ///
    /// Returns [`SessionMergeError::InvalidPolicy`] for an invalid fanout or a
    /// zero input budget.
    pub fn validate(self) -> Result<(), SessionMergeError> {
        if !(2..=MAX_SESSION_MERGE_FANOUT).contains(&self.fanout) {
            return Err(SessionMergeError::InvalidPolicy {
                field: "fanout",
                reason: "must be between 2 and 64 inclusive",
            });
        }
        if self.max_input_docs == 0 {
            return Err(SessionMergeError::InvalidPolicy {
                field: "max_input_docs",
                reason: "must be positive",
            });
        }
        if self.max_input_bytes == 0 {
            return Err(SessionMergeError::InvalidPolicy {
                field: "max_input_bytes",
                reason: "must be positive",
            });
        }
        Ok(())
    }

    // Geometric row-count tiers avoid repeatedly rewriting an increasingly
    // large segment together with only a few newly appended tiny segments.
    // Zero-row segments occupy the smallest tier; no floating-point logarithm.
    fn tier(self, doc_count: u32) -> u32 {
        let mut rows = u64::from(doc_count).max(1);
        let radix = self.fanout as u64;
        let mut tier = 0;
        while rows >= radix {
            rows /= radix;
            tier += 1;
        }
        tier
    }
}

/// A proposed bound-consecutive merge, not an authority to publish a manifest.
///
/// Execution must still use [`QuillIndex::concat_merge`], which rechecks the
/// source run under its writer lock and performs the durable publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMergePlan {
    /// Physical row-count tier, using the policy fanout as its radix.
    pub tier: u32,
    /// Source identities in manifest order; no live interval was skipped.
    pub source_segment_ids: Vec<u64>,
    /// Inclusive lower bound of the merged covering interval.
    pub docid_lo: u64,
    /// Exclusive upper bound of the merged covering interval.
    pub docid_hi: u64,
    /// Total physical rows, not live rows after deletions.
    pub input_docs: u64,
    /// Sum of immutable input file lengths.
    pub input_bytes: u64,
}

/// Failure while planning or executing session-fragmentation maintenance.
#[derive(Debug, Error)]
pub enum SessionMergeError {
    /// Invalid caller-supplied policy.
    #[error("invalid session merge {field}: {reason}")]
    InvalidPolicy {
        /// Rejected policy field.
        field: &'static str,
        /// Required invariant.
        reason: &'static str,
    },
    /// The supplied manifest slice cannot describe a valid Q1 layout.
    #[error("invalid session merge segment {segment_id}: {reason}")]
    InvalidSegment {
        /// Segment at the invalid boundary.
        segment_id: u64,
        /// Violated layout invariant.
        reason: &'static str,
    },
    /// Cancellation was observed before publication was attempted.
    #[error("session merge planning was cancelled")]
    Cancelled,
    /// The selected source-ID vector could not be reserved.
    #[error("could not allocate {count} session merge source ids")]
    Allocation {
        /// Requested source count.
        count: usize,
    },
    /// The production snapshot, writer, codec, or publication path failed.
    #[error(transparent)]
    Index(#[from] QuillIndexError),
}

fn checkpoint(cx: &Cx) -> Result<(), SessionMergeError> {
    cx.checkpoint().map_err(|_| SessionMergeError::Cancelled)
}

/// Select a same-row-tier run without rejecting unused document-ID leases.
///
/// Smallest eligible tiers take priority, then the earliest run in manifest
/// order. Every selected run has exactly `fanout` adjacent entries and fits
/// both input budgets. Oversized runs are skipped, not partially selected.
/// Counts include tombstoned rows because concat copies their physical bytes.
///
/// The caller supplies the complete ordered manifest segment list. This
/// function validates its range/count shape; full codec/checksum admission
/// remains the snapshot reader's responsibility. Planning does not read files.
///
/// # Errors
///
/// Returns a typed error for invalid policy, malformed or overlapping ranges,
/// cancellation, or allocation failure. `Ok(None)` means no eligible run.
pub fn plan_session_merge(
    cx: &Cx,
    segments: &[ManifestSegment],
    policy: SessionMergePolicy,
) -> Result<Option<SessionMergePlan>, SessionMergeError> {
    checkpoint(cx)?;
    policy.validate()?;
    for (index, segment) in segments.iter().enumerate() {
        checkpoint(cx)?;
        let reason = if segment.docid_lo >= segment.docid_hi
            || segment.docid_hi > u64::from(u32::MAX) + 1
        {
            Some("range must be nonempty and fit the global u32 document space")
        } else if u64::from(segment.doc_count) > segment.docid_hi - segment.docid_lo {
            Some("physical row count exceeds the covering interval")
        } else if segment.tombstones.cardinality() > u64::from(segment.doc_count) {
            Some("tombstone count exceeds physical row count")
        } else if index > 0 && segments[index - 1].docid_hi > segment.docid_lo {
            Some("manifest ranges overlap or are out of order")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(SessionMergeError::InvalidSegment {
                segment_id: segment.segment_id,
                reason,
            });
        }
    }

    let mut selected: Option<(u32, usize, u64, u64)> = None;
    for (start, sources) in segments.windows(policy.fanout).enumerate() {
        checkpoint(cx)?;
        let tier = policy.tier(sources[0].doc_count);
        if selected.is_some_and(|(best, ..)| best <= tier)
            || sources
                .iter()
                .any(|source| policy.tier(source.doc_count) != tier)
        {
            continue;
        }
        let totals = sources
            .iter()
            .try_fold((0_u64, 0_u64), |(docs, bytes), source| {
                Some((
                    docs.checked_add(u64::from(source.doc_count))?,
                    bytes.checked_add(source.file_len)?,
                ))
            });
        let Some((docs, bytes)) = totals else {
            // Overflow cannot fit any u64 budget. A later run may still fit.
            continue;
        };
        if docs <= policy.max_input_docs && bytes <= policy.max_input_bytes {
            selected = Some((tier, start, docs, bytes));
        }
    }
    let Some((tier, start, input_docs, input_bytes)) = selected else {
        return Ok(None);
    };
    let sources = &segments[start..start + policy.fanout];
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

impl QuillIndex {
    /// Perform at most one gap-tolerant, resource-bounded concat merge.
    ///
    /// Call after an explicit commit, between ingest batches. Output identity
    /// and timestamp have the same caller-owned contract as `concat_merge`.
    /// No merge is attempted when fewer than `fanout` same-row-tier segments
    /// fit the budgets. The ordinary automatic tier policy is unchanged.
    ///
    /// Planning pins a readable snapshot. The existing writer-locked concat
    /// path then revalidates the chosen source run: concurrent ingest or
    /// maintenance is never overwritten. It also retires live leases before
    /// publishing a hull across their gaps, preserves document IDs, and keeps
    /// the usual durability, cancellation, and recovery checks.
    ///
    /// # Errors
    ///
    /// Returns planning errors or the production concat error, including
    /// uncommitted writes, stale source IDs, output-ID collision, cancellation,
    /// or publication failure. Do not blindly replay a publication failure;
    /// use the existing index reconciliation contract first.
    pub async fn merge_sessions_once(
        &self,
        cx: &Cx,
        policy: SessionMergePolicy,
        output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<Option<Arc<KeeperSnapshot>>, SessionMergeError> {
        checkpoint(cx)?;
        policy.validate()?;
        let snapshot = self.snapshot()?;
        let plan = plan_session_merge(cx, &snapshot.loaded_manifest().manifest.segments, policy)?;
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
        // No post-publication checkpoint: successful durable publication must
        // not be relabelled as a cancelled operation after it has committed.
        Ok(Some(published))
    }
}

#[cfg(test)]
mod tests {
    use asupersync::test_utils::run_test_with_cx;

    use super::*;
    use crate::{QuillConfig, TierMergePolicy, TombstoneSet, plan_tier_merge};

    fn segment(id: u64, lo: u64, docs: u32, bytes: u64) -> ManifestSegment {
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
    fn short_sessions_merge_without_relaxing_the_existing_tier_policy() {
        run_test_with_cx(|cx| async move {
            let segments: Vec<_> = (0..8)
                .map(|id| segment(id, id * 65_536, 3, 4096))
                .collect();
            let ordinary = TierMergePolicy::from_config(&QuillConfig::default());
            assert!(plan_tier_merge(&segments, ordinary).unwrap().is_none());
            let plan = plan_session_merge(&cx, &segments, SessionMergePolicy::default())
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, (0..8).collect::<Vec<_>>());
            assert_eq!(plan.input_docs, 24);
            assert_eq!(plan.input_bytes, 8 * 4096);
            assert_eq!((plan.docid_lo, plan.docid_hi), (0, 7 * 65_536 + 3));
        });
    }

    #[test]
    fn tier_uses_physical_rows_not_range_width_or_tombstone_density() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                fanout: 2,
                ..SessionMergePolicy::default()
            };
            let mut wide = segment(1, 0, 3, 1024);
            wide.docid_hi = 65_536;
            wide.tombstones.insert(0).unwrap();
            wide.tombstones.insert(1).unwrap();
            let narrow = segment(2, 131_072, 3, 1024);
            let plan = plan_session_merge(&cx, &[wide, narrow], policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.tier, 1);
            assert_eq!(plan.input_docs, 6);
        });
    }

    #[test]
    fn budgets_skip_whole_runs_and_preserve_bound_consecutiveness() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                fanout: 2,
                max_input_docs: 6,
                max_input_bytes: 2048,
            };
            let segments = [
                segment(1, 0, 3, 2049),
                segment(2, 65_536, 3, 1024),
                segment(3, 131_072, 3, 1024),
            ];
            let plan = plan_session_merge(&cx, &segments, policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, vec![2, 3]);
            assert_eq!(plan.input_bytes, policy.max_input_bytes);
            assert_eq!(plan.input_docs, policy.max_input_docs);
            let tighter = SessionMergePolicy {
                max_input_docs: 5,
                ..policy
            };
            assert!(
                plan_session_merge(&cx, &segments, tighter)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn never_skips_a_large_live_interval_between_small_segments() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                fanout: 2,
                ..SessionMergePolicy::default()
            };
            let segments = [
                segment(1, 0, 3, 1024),
                segment(2, 65_536, 1000, 1024),
                segment(3, 131_072, 3, 1024),
            ];
            assert!(
                plan_session_merge(&cx, &segments, policy)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn smallest_tier_wins_then_earliest_manifest_run() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                fanout: 2,
                ..SessionMergePolicy::default()
            };
            let segments = [
                segment(1, 0, 100, 1024),
                segment(2, 65_536, 100, 1024),
                segment(3, 131_072, 3, 1024),
                segment(4, 196_608, 3, 1024),
                segment(5, 262_144, 3, 1024),
            ];
            let plan = plan_session_merge(&cx, &segments, policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, vec![3, 4]);
        });
    }

    #[test]
    fn overflowing_byte_total_cannot_bypass_budget() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                fanout: 2,
                max_input_bytes: u64::MAX,
                ..SessionMergePolicy::default()
            };
            let segments = [
                segment(1, 0, 1, u64::MAX),
                segment(2, 65_536, 1, 1),
                segment(3, 131_072, 1, 1),
            ];
            let plan = plan_session_merge(&cx, &segments, policy)
                .unwrap()
                .unwrap();
            assert_eq!(plan.source_segment_ids, vec![2, 3]);
        });
    }

    #[test]
    fn malformed_later_ranges_fail_even_when_earlier_window_would_merge() {
        run_test_with_cx(|cx| async move {
            let policy = SessionMergePolicy {
                fanout: 2,
                ..SessionMergePolicy::default()
            };
            let mut segments = [
                segment(1, 0, 1, 1),
                segment(2, 65_536, 1, 1),
                segment(3, 65_536, 1, 1),
            ];
            assert!(matches!(
                plan_session_merge(&cx, &segments, policy),
                Err(SessionMergeError::InvalidSegment { .. })
            ));
            segments[2] = segment(3, 131_072, 1, 1);
            segments[2].doc_count = 2;
            assert!(matches!(
                plan_session_merge(&cx, &segments, policy),
                Err(SessionMergeError::InvalidSegment { .. })
            ));
            segments[2].doc_count = 1;
            segments[2].docid_hi = u64::MAX;
            assert!(matches!(
                plan_session_merge(&cx, &segments, policy),
                Err(SessionMergeError::InvalidSegment { .. })
            ));
        });
    }

    #[test]
    fn invalid_policies_are_rejected_even_for_an_empty_index() {
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
                    plan_session_merge(&cx, &[], policy),
                    Err(SessionMergeError::InvalidPolicy { .. })
                ));
            }
            assert!(
                plan_session_merge(&cx, &[], SessionMergePolicy::default())
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn cancellation_is_not_reported_as_nothing_to_merge() {
        run_test_with_cx(|cx| async move {
            cx.cancel_fast(asupersync::CancelKind::User);
            assert!(matches!(
                plan_session_merge(&cx, &[], SessionMergePolicy::default()),
                Err(SessionMergeError::Cancelled)
            ));
        });
    }
}
