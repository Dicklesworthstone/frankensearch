//! Multi-step maintenance with cumulative work limits and recoverable progress.

use asupersync::Cx;
use thiserror::Error;

use super::{
    SessionMergeError, SessionMergePlan, SessionMergePolicy, checkpoint,
    plan_session_merge_with_fit,
};
use crate::keeper::{ConcatMergeSizing, KeeperError};
use crate::{QuillIndex, QuillIndexError};

/// Cumulative limits for an explicit [`QuillIndex::merge_sessions`] invocation.
///
/// Inputs rewritten more than once are charged on every merge. Byte limits
/// count immutable input files, not output bytes or peak resident memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionMergeBudget {
    /// Maximum number of successful publications. Must be positive.
    pub max_merges: usize,
    /// Maximum sum of physical input rows, including tombstones. Must be positive.
    pub max_input_docs: u64,
    /// Maximum sum of input file lengths across all merges. Must be positive.
    pub max_input_bytes: u64,
}

impl Default for SessionMergeBudget {
    fn default() -> Self {
        Self {
            max_merges: 8,
            max_input_docs: 1_000_000,
            max_input_bytes: 256 * 1024 * 1024,
        }
    }
}

impl SessionMergeBudget {
    /// Validate limits and the caller-owned consecutive output-ID sequence.
    ///
    /// Collisions with existing segments are still rejected by the normal
    /// publication path; this check only prevents sequence arithmetic overflow.
    ///
    /// # Errors
    ///
    /// Returns [`SessionMergeError::InvalidPolicy`] for a zero limit or a
    /// sequence that would extend past `u64::MAX`.
    pub fn validate(self, first_output_segment_id: u64) -> Result<(), SessionMergeError> {
        for (field, zero) in [
            ("max_merges", self.max_merges == 0),
            ("total_input_docs", self.max_input_docs == 0),
            ("total_input_bytes", self.max_input_bytes == 0),
        ] {
            if zero {
                return Err(SessionMergeError::InvalidPolicy {
                    field,
                    reason: "must be positive",
                });
            }
        }
        let offset = u64::try_from(self.max_merges - 1).ok();
        if offset
            .and_then(|offset| first_output_segment_id.checked_add(offset))
            .is_none()
        {
            return Err(SessionMergeError::InvalidPolicy {
                field: "first_output_segment_id",
                reason: "the reserved output-ID sequence must fit u64",
            });
        }
        Ok(())
    }

    fn stop(self, progress: &SessionMergeProgress) -> Option<SessionMergeStop> {
        if progress.merges.len() >= self.max_merges {
            Some(SessionMergeStop::MergeLimit)
        } else if progress.input_docs >= self.max_input_docs
            || progress.input_bytes >= self.max_input_bytes
        {
            Some(SessionMergeStop::InputBudget)
        } else {
            None
        }
    }

    // Called only after stop() admitted another step. Keep the fanout (and
    // therefore the tier radix) unchanged while limiting remaining work.
    fn remaining_policy(
        self,
        policy: SessionMergePolicy,
        progress: &SessionMergeProgress,
    ) -> SessionMergePolicy {
        SessionMergePolicy {
            max_input_docs: policy
                .max_input_docs
                .min(self.max_input_docs - progress.input_docs),
            max_input_bytes: policy
                .max_input_bytes
                .min(self.max_input_bytes - progress.input_bytes),
            ..policy
        }
    }
}

/// One merge whose normal publication path returned durable success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMergeReceipt {
    /// Exact selected sources and charged input work.
    pub plan: SessionMergePlan,
    /// Caller-owned identity of the published output.
    pub output_segment_id: u64,
    /// Manifest generation returned by this publication.
    pub generation: u64,
}

/// Successful work retained even when a later step fails or is cancelled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionMergeProgress {
    /// Successful publications in execution order, not necessarily still-live outputs.
    pub merges: Vec<SessionMergeReceipt>,
    /// Physical input rows charged across successful publications.
    pub input_docs: u64,
    /// Input file bytes charged across successful publications.
    pub input_bytes: u64,
}

/// Reason a maintenance pass stopped without a publication error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMergeStop {
    /// The allowed number of publications completed; no further work was attempted.
    MergeLimit,
    /// At least one cumulative input budget was consumed exactly.
    InputBudget,
    /// No exact-fanout run fits the per-step policy and remaining cumulative budget.
    /// This is not a claim that a larger-budget or bounded-tail pass cannot progress.
    NoEligibleRunWithinBudget,
}

/// Successful completion of a budgeted pass; not an all-or-nothing transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMergeReport {
    /// Individually durable publications and their cumulative input work.
    pub progress: SessionMergeProgress,
    /// Why the pass stopped.
    pub stop: SessionMergeStop,
}

/// A concat call that returned an error, with its publication outcome unresolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMergeAttempt {
    /// Plan passed to the writer-locked concat path.
    pub plan: SessionMergePlan,
    /// Output identity passed to concat; useful when reconciling publication.
    pub output_segment_id: u64,
    /// Timestamp passed to concat.
    pub created_unix_s: i64,
}

/// A failed pass preserves successful receipts and does not automatically retry.
///
/// `attempted` is present only if concat returned an error. Such an error does
/// not prove that publication did not happen. Reconcile with the normal index
/// recovery contract before deciding whether to resume, and account for the
/// attempted input work separately if that publication did become authoritative.
#[derive(Debug, Error)]
#[error("session merge pass failed: {source}")]
pub struct SessionMergeFailure {
    /// Earlier publications that returned durable success; never rolled back here.
    pub progress: SessionMergeProgress,
    /// Concat attempt requiring reconciliation, if the failure reached that path.
    pub attempted: Option<Box<SessionMergeAttempt>>,
    /// Original planning, allocation, cancellation, or index error.
    #[source]
    pub source: Box<SessionMergeError>,
}

impl QuillIndex {
    /// Run a bounded amount of exact-fanout session maintenance after a commit.
    ///
    /// Every step replans from a fresh native snapshot, enforcing actual schema
    /// and default reader limits. The ordinary writer-locked concat path owns
    /// source revalidation, lease retirement, and durable publication. There is
    /// no background task, automatic retry, rollback, or durable-format change.
    /// Concurrent changes cannot be overwritten: a stale run fails through concat.
    ///
    /// `policy` limits each merge; `budget` limits the entire invocation. A
    /// previously produced segment selected again is charged again. The exact
    /// fanout is preserved to avoid the write amplification of repeatedly
    /// coalescing short tails. Use [`Self::merge_sessions_bounded_once`] when
    /// intentionally consolidating a tail shorter than that fanout.
    ///
    /// Output IDs are `first_output_segment_id + successful_merge_count`.
    /// The caller must provide unused identities, as for [`Self::concat_merge`].
    /// On failure, do not blindly replay the pass or reuse the attempted ID:
    /// inspect the returned progress and reconcile the possibly published attempt.
    /// Held search snapshots retain their existing visibility contract.
    ///
    /// # Errors
    ///
    /// Returns the original error together with all earlier successful receipts.
    /// Invalid policy/ID sequences fail before snapshot reads or publication.
    /// Cancellation between steps preserves progress. After the final permitted
    /// publication succeeds, a late cancellation is not substituted for success.
    pub async fn merge_sessions(
        &self,
        cx: &Cx,
        policy: SessionMergePolicy,
        budget: SessionMergeBudget,
        first_output_segment_id: u64,
        created_unix_s: i64,
    ) -> Result<SessionMergeReport, SessionMergeFailure> {
        let mut progress = SessionMergeProgress::default();
        let mut attempted = None;
        let result: Result<SessionMergeStop, SessionMergeError> = async {
            checkpoint(cx)?;
            policy.validate()?;
            budget.validate(first_output_segment_id)?;
            loop {
                // Check limits before another checkpoint: the final allowed
                // durable success cannot become a late cancellation failure.
                if let Some(stop) = budget.stop(&progress) {
                    return Ok(stop);
                }
                checkpoint(cx)?;
                let remaining = budget.remaining_policy(policy, &progress);
                let plan = plan_next(self, cx, remaining)?;
                let Some(plan) = plan else {
                    checkpoint(cx)?;
                    return Ok(SessionMergeStop::NoEligibleRunWithinBudget);
                };
                // Reserve before publishing so a recoverable allocation error
                // cannot lose the receipt for an already successful publication.
                progress.merges.try_reserve(1).map_err(|_| {
                    SessionMergeError::Allocation {
                        count: progress.merges.len() + 1,
                    }
                })?;
                // validate() checked the complete sequence before any I/O.
                let output_segment_id = first_output_segment_id + progress.merges.len() as u64;
                checkpoint(cx)?;
                match self
                    .concat_merge(cx, &plan.source_segment_ids, output_segment_id, created_unix_s)
                    .await
                {
                    Ok(published) => {
                        // The selected plan fits the remaining budgets, so
                        // these sums cannot overflow or exceed either cap.
                        progress.input_docs += plan.input_docs;
                        progress.input_bytes += plan.input_bytes;
                        progress.merges.push(SessionMergeReceipt {
                            plan,
                            output_segment_id,
                            generation: published.loaded_manifest().manifest.generation,
                        });
                    }
                    Err(error) => {
                        attempted = Some(Box::new(SessionMergeAttempt {
                            plan,
                            output_segment_id,
                            created_unix_s,
                        }));
                        return Err(error.into());
                    }
                }
            }
        }
        .await;
        match result {
            Ok(stop) => Ok(SessionMergeReport { progress, stop }),
            Err(source) => Err(SessionMergeFailure {
                progress,
                attempted,
                source: Box::new(source),
            }),
        }
    }
}

fn plan_next(
    index: &QuillIndex,
    cx: &Cx,
    policy: SessionMergePolicy,
) -> Result<Option<SessionMergePlan>, SessionMergeError> {
    let snapshot = index.snapshot()?;
    // Structural source errors remain errors, never budget-related no-ops.
    let sizing = ConcatMergeSizing::new(snapshot.schema(), snapshot.segments())
        .map_err(KeeperError::from)
        .map_err(QuillIndexError::from)?;
    let plan = plan_session_merge_with_fit(
        cx,
        &snapshot.loaded_manifest().manifest.segments,
        policy,
        |range| sizing.check(range).is_ok(),
    )?;
    // The exact-fanout planner can finish with a rejected final fit. Surface
    // cancellation there instead of reporting that maintenance was unnecessary.
    checkpoint(cx)?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_rejects_zero_limits_and_id_wrap_before_any_work() {
        let valid = SessionMergeBudget::default();
        for invalid in [
            SessionMergeBudget { max_merges: 0, ..valid },
            SessionMergeBudget { max_input_docs: 0, ..valid },
            SessionMergeBudget { max_input_bytes: 0, ..valid },
        ] {
            assert!(matches!(invalid.validate(0), Err(SessionMergeError::InvalidPolicy { .. })));
        }
        assert!(valid.validate(u64::MAX - 7).is_ok());
        assert!(valid.validate(u64::MAX - 6).is_err());
        assert!(SessionMergeBudget { max_merges: 1, ..valid }.validate(u64::MAX).is_ok());
    }

    #[test]
    fn remaining_budget_clamps_work_without_changing_tier_radix() {
        let policy = SessionMergePolicy { fanout: 8, max_input_docs: 40, max_input_bytes: 400 };
        let budget = SessionMergeBudget { max_merges: 3, max_input_docs: 100, max_input_bytes: 1000 };
        let mut progress = SessionMergeProgress::default();
        assert_eq!(budget.remaining_policy(policy, &progress), policy);
        progress.input_docs = 75;
        progress.input_bytes = 950;
        let remaining = budget.remaining_policy(policy, &progress);
        assert_eq!(remaining.fanout, 8);
        assert_eq!(remaining.max_input_docs, 25);
        assert_eq!(remaining.max_input_bytes, 50);
        assert_eq!(budget.stop(&progress), None);
        progress.input_docs = 100;
        assert_eq!(budget.stop(&progress), Some(SessionMergeStop::InputBudget));
        progress.input_docs = 99;
        progress.input_bytes = 1000;
        assert_eq!(budget.stop(&progress), Some(SessionMergeStop::InputBudget));
    }

    #[test]
    fn near_u64_budget_limits_neither_wrap_nor_relax_per_step_caps() {
        let budget = SessionMergeBudget {
            max_merges: 1,
            max_input_docs: u64::MAX,
            max_input_bytes: u64::MAX,
        };
        let progress = SessionMergeProgress {
            input_docs: u64::MAX - 7,
            input_bytes: u64::MAX - 11,
            ..SessionMergeProgress::default()
        };
        let remaining = budget.remaining_policy(SessionMergePolicy::default(), &progress);
        assert_eq!((remaining.max_input_docs, remaining.max_input_bytes), (7, 11));
    }

    #[test]
    fn publication_limit_terminates_even_zero_row_zero_byte_plans() {
        let budget = SessionMergeBudget { max_merges: 1, ..SessionMergeBudget::default() };
        let progress = SessionMergeProgress {
            merges: vec![SessionMergeReceipt {
                plan: SessionMergePlan {
                    tier: 0,
                    source_segment_ids: vec![1, 2],
                    docid_lo: 0,
                    docid_hi: 2,
                    input_docs: 0,
                    input_bytes: 0,
                },
                output_segment_id: u64::MAX,
                generation: 1,
            }],
            ..SessionMergeProgress::default()
        };
        assert_eq!(budget.stop(&progress), Some(SessionMergeStop::MergeLimit));
    }
}
