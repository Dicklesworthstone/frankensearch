//! Two-tier score blending utilities.
//!
//! Combines fast-tier and quality-tier semantic rankings into a single
//! blended ranking using:
//!
//! ```text
//! blended_score = alpha * quality_score + (1 - alpha) * fast_score
//! ```
//!
//! where `alpha` is `blend_factor` (default behavior target: `0.7`).
//!
//! Missing-source behavior is intentional:
//! - document only in fast set: `quality_score = 0.0`
//! - document only in quality set: `fast_score = 0.0`
//!
//! This naturally penalizes single-source hits when both tiers are available.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use frankensearch_core::{RankChanges, VectorHit};
use tracing::{debug, instrument};

use crate::normalize::min_max_normalize;

const DEFAULT_BLEND_FACTOR: f32 = 0.7;
const NON_FINITE_SCORE_FALLBACK: f32 = 0.0;

#[derive(Debug, Clone, Copy, Default)]
struct ScorePair {
    fast: f32,
    quality: f32,
    index: u32,
}

/// Blend fast-tier and quality-tier vector hits into a single ranking.
///
/// Both input score lists are min-max normalized independently before blending.
///
/// # Arguments
///
/// - `fast_results`: semantic hits from the fast embedder
/// - `quality_results`: semantic hits from the quality embedder
/// - `blend_factor`: `0.0` = fast-only, `1.0` = quality-only
///
/// Non-finite `blend_factor` values fall back to `0.7`.
#[must_use]
#[instrument(
    name = "frankensearch::blend",
    skip(fast_results, quality_results),
    fields(
        fast_count = fast_results.len(),
        quality_count = quality_results.len(),
        blend_factor,
    )
)]
pub fn blend_two_tier(
    fast_results: &[VectorHit],
    quality_results: &[VectorHit],
    blend_factor: f32,
) -> Vec<VectorHit> {
    let alpha = sanitize_blend_factor(blend_factor);

    let mut fast_scores: Vec<f32> = fast_results.iter().map(|hit| hit.score).collect();
    let mut quality_scores: Vec<f32> = quality_results.iter().map(|hit| hit.score).collect();
    min_max_normalize(&mut fast_scores);
    min_max_normalize(&mut quality_scores);

    let mut merged: HashMap<String, ScorePair> =
        HashMap::with_capacity(fast_results.len() + quality_results.len());

    for (hit, normalized) in fast_results.iter().zip(fast_scores.into_iter()) {
        let entry = merged
            .entry(hit.doc_id.clone())
            .or_insert_with(|| ScorePair {
                index: hit.index,
                ..ScorePair::default()
            });
        entry.fast = normalized;
    }

    for (hit, normalized) in quality_results.iter().zip(quality_scores.into_iter()) {
        let entry = merged
            .entry(hit.doc_id.clone())
            .or_insert_with(|| ScorePair {
                index: hit.index,
                ..ScorePair::default()
            });
        entry.quality = normalized;
    }

    let mut blended: Vec<VectorHit> = merged
        .into_iter()
        .map(|(doc_id, pair)| {
            let score = alpha.mul_add(pair.quality, (1.0 - alpha) * pair.fast);
            VectorHit {
                index: pair.index,
                score: sanitize_score(score),
                doc_id,
            }
        })
        .collect();

    blended.sort_by(|left, right| {
        sanitize_score(right.score)
            .total_cmp(&sanitize_score(left.score))
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });

    debug!(
        target: "frankensearch.blend",
        blended_count = blended.len(),
        effective_alpha = %alpha,
        "blending complete"
    );

    blended
}

/// Compute promoted/demoted/stable rank-change counts between two rankings.
///
/// - `promoted`: rank improved (smaller index), or new docs appearing in refined
/// - `demoted`: rank worsened (larger index), or docs dropped from refined
/// - `stable`: rank unchanged
#[must_use]
pub fn compute_rank_changes(initial: &[VectorHit], refined: &[VectorHit]) -> RankChanges {
    let initial_rank = build_rank_map(initial);
    let refined_rank = build_rank_map(refined);

    let mut promoted = 0;
    let mut demoted = 0;
    let mut stable = 0;

    for (doc_id, old_rank) in &initial_rank {
        match refined_rank.get(doc_id) {
            Some(new_rank) => match new_rank.cmp(old_rank) {
                Ordering::Less => promoted += 1,
                Ordering::Greater => demoted += 1,
                Ordering::Equal => stable += 1,
            },
            None => demoted += 1,
        }
    }

    for doc_id in refined_rank.keys() {
        if !initial_rank.contains_key(doc_id) {
            promoted += 1;
        }
    }

    RankChanges {
        promoted,
        demoted,
        stable,
    }
}

/// Compute Kendall's tau rank correlation between two rankings.
///
/// Uses merge-sort-based inversion counting for O(n log n) performance
/// instead of the naive O(n^2) pairwise comparison.
///
/// Returns `None` when fewer than two common documents exist.
#[must_use]
pub fn kendall_tau(initial: &[VectorHit], refined: &[VectorHit]) -> Option<f64> {
    let initial_rank = build_rank_map(initial);
    let refined_rank = build_rank_map(refined);

    // Collect common documents in initial-rank order (iteration order preserves it).
    let mut seen = HashSet::new();
    let mut common = Vec::new();
    for hit in initial {
        if refined_rank.contains_key(&hit.doc_id) && seen.insert(hit.doc_id.clone()) {
            common.push(hit.doc_id.clone());
        }
    }

    let n = common.len();
    if n < 2 {
        return None;
    }

    // Build array of refined ranks in initial-rank order.
    // An inversion in this array corresponds to a discordant pair.
    let mut refined_ranks: Vec<usize> = common
        .iter()
        .map(|doc_id| refined_rank[doc_id])
        .collect();

    let discordant = merge_sort_inversions(&mut refined_ranks);

    let n_u64 = u64::try_from(n).ok()?;
    let total_pairs = n_u64.checked_mul(n_u64 - 1)? / 2;
    if total_pairs == 0 {
        return None;
    }

    let concordant = total_pairs.saturating_sub(discordant);

    #[allow(clippy::cast_precision_loss)]
    let numerator = concordant as f64 - discordant as f64;
    #[allow(clippy::cast_precision_loss)]
    let denominator = total_pairs as f64;
    Some(numerator / denominator)
}

/// Count inversions in a slice using merge sort. O(n log n).
///
/// An inversion is a pair `(i, j)` where `i < j` but `arr[i] > arr[j]`.
/// The slice is sorted in place as a side effect.
fn merge_sort_inversions(arr: &mut [usize]) -> u64 {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let mut left = arr[..mid].to_vec();
    let mut right = arr[mid..].to_vec();

    let mut count = merge_sort_inversions(&mut left);
    count += merge_sort_inversions(&mut right);

    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            arr[k] = left[i];
            i += 1;
        } else {
            arr[k] = right[j];
            #[allow(clippy::cast_possible_truncation)]
            {
                count += (left.len() - i) as u64;
            }
            j += 1;
        }
        k += 1;
    }

    arr[k..k + left.len() - i].copy_from_slice(&left[i..]);
    k += left.len() - i;
    arr[k..].copy_from_slice(&right[j..]);

    count
}

const fn sanitize_blend_factor(blend_factor: f32) -> f32 {
    if blend_factor.is_finite() {
        blend_factor.clamp(0.0, 1.0)
    } else {
        DEFAULT_BLEND_FACTOR
    }
}

const fn sanitize_score(score: f32) -> f32 {
    if score.is_finite() {
        score
    } else {
        NON_FINITE_SCORE_FALLBACK
    }
}

fn build_rank_map(hits: &[VectorHit]) -> HashMap<String, usize> {
    let mut ranks = HashMap::with_capacity(hits.len());
    for (rank, hit) in hits.iter().enumerate() {
        ranks.entry(hit.doc_id.clone()).or_insert(rank);
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::{blend_two_tier, compute_rank_changes, kendall_tau};
    use frankensearch_core::VectorHit;

    const EPSILON: f32 = 1e-6;

    fn hit(doc_id: &str, score: f32, index: u32) -> VectorHit {
        VectorHit {
            index,
            score,
            doc_id: doc_id.to_owned(),
        }
    }

    fn score_for(doc_id: &str, hits: &[VectorHit]) -> f32 {
        hits.iter()
            .find(|hit| hit.doc_id == doc_id)
            .map(|hit| hit.score)
            .expect("missing doc")
    }

    #[test]
    fn blend_factor_point_seven_matches_weighted_formula() {
        let fast = vec![hit("a", 1.0, 0), hit("b", 0.0, 1), hit("c", 2.0, 2)];
        let quality = vec![hit("a", 2.0, 0), hit("b", 0.0, 1), hit("c", 1.0, 2)];

        let blended = blend_two_tier(&fast, &quality, 0.7);
        let a_score = score_for("a", &blended);

        // Fast normalized for "a" = 0.5, quality normalized for "a" = 1.0.
        // blended = 0.7*1.0 + 0.3*0.5 = 0.85
        assert!(
            (a_score - 0.85).abs() <= EPSILON,
            "expected 0.85, got {a_score}"
        );
    }

    #[test]
    fn alpha_one_uses_quality_only() {
        let fast = vec![hit("a", 10.0, 0), hit("b", 0.0, 1)];
        let quality = vec![hit("a", 5.0, 0), hit("b", 15.0, 1)];

        let blended = blend_two_tier(&fast, &quality, 1.0);
        assert!((score_for("a", &blended) - 0.0).abs() <= EPSILON);
        assert!((score_for("b", &blended) - 1.0).abs() <= EPSILON);
    }

    #[test]
    fn alpha_zero_uses_fast_only() {
        let fast = vec![hit("a", 10.0, 0), hit("b", 0.0, 1)];
        let quality = vec![hit("a", 5.0, 0), hit("b", 15.0, 1)];

        let blended = blend_two_tier(&fast, &quality, 0.0);
        assert!((score_for("a", &blended) - 1.0).abs() <= EPSILON);
        assert!((score_for("b", &blended) - 0.0).abs() <= EPSILON);
    }

    #[test]
    fn single_source_scores_are_penalized() {
        let fast = vec![hit("fast-only", 10.0, 0)];
        let quality = vec![hit("quality-only", 10.0, 1)];
        let blended = blend_two_tier(&fast, &quality, 0.7);

        let fast_only = score_for("fast-only", &blended);
        let quality_only = score_for("quality-only", &blended);

        // Degenerate single-entry normalization -> 0.5 for each source.
        assert!((fast_only - 0.15).abs() <= EPSILON);
        assert!((quality_only - 0.35).abs() <= EPSILON);
    }

    #[test]
    fn equal_scores_remain_equal() {
        let fast = vec![hit("same", 1.0, 0), hit("other", 1.0, 1)];
        let quality = vec![hit("same", 2.0, 0), hit("other", 2.0, 1)];
        let blended = blend_two_tier(&fast, &quality, 0.7);

        assert!((score_for("same", &blended) - 0.5).abs() <= EPSILON);
    }

    #[test]
    fn non_finite_scores_are_sanitized() {
        let fast = vec![hit("nan-doc", f32::NAN, 0), hit("ok-doc", 1.0, 1)];
        let blended = blend_two_tier(&fast, &[], 0.3);

        assert!(score_for("nan-doc", &blended).is_finite());
        assert!(score_for("ok-doc", &blended).is_finite());
    }

    #[test]
    fn ordering_prefers_higher_blended_score() {
        let fast = vec![hit("a", 10.0, 0), hit("b", 1.0, 1)];
        let quality = vec![hit("a", 1.0, 0), hit("b", 10.0, 1)];
        let blended = blend_two_tier(&fast, &quality, 0.7);

        assert_eq!(blended[0].doc_id, "b");
        assert_eq!(blended[1].doc_id, "a");
    }

    #[test]
    fn compute_rank_changes_tracks_promoted_demoted_stable() {
        let initial = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let refined = vec![hit("b", 1.0, 1), hit("a", 0.9, 0), hit("d", 0.7, 3)];

        let changes = compute_rank_changes(&initial, &refined);
        assert_eq!(changes.promoted, 2); // b up + d new
        assert_eq!(changes.demoted, 2); // a down + c dropped
        assert_eq!(changes.stable, 0);
    }

    #[test]
    fn kendall_tau_identical_rankings_is_one() {
        let initial = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let refined = vec![hit("a", 0.7, 0), hit("b", 0.6, 1), hit("c", 0.5, 2)];
        let tau = kendall_tau(&initial, &refined).expect("tau");
        assert!((tau - 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn kendall_tau_reversed_rankings_is_negative_one() {
        let initial = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let refined = vec![hit("c", 0.7, 2), hit("b", 0.6, 1), hit("a", 0.5, 0)];
        let tau = kendall_tau(&initial, &refined).expect("tau");
        assert!((tau + 1.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn kendall_tau_none_when_insufficient_overlap() {
        let initial = vec![hit("a", 1.0, 0)];
        let refined = vec![hit("b", 0.9, 1)];
        assert!(kendall_tau(&initial, &refined).is_none());
    }

    #[test]
    fn blend_both_empty_returns_empty() {
        let blended = blend_two_tier(&[], &[], 0.7);
        assert!(blended.is_empty());
    }

    #[test]
    fn blend_fast_only_returns_results() {
        let fast = vec![hit("a", 1.0, 0), hit("b", 0.5, 1)];
        let blended = blend_two_tier(&fast, &[], 0.7);
        assert_eq!(blended.len(), 2);
        assert!(blended.iter().all(|h| h.score.is_finite()));
    }

    #[test]
    fn blend_quality_only_returns_results() {
        let quality = vec![hit("a", 1.0, 0), hit("b", 0.5, 1)];
        let blended = blend_two_tier(&[], &quality, 0.7);
        assert_eq!(blended.len(), 2);
        assert!(blended.iter().all(|h| h.score.is_finite()));
    }

    #[test]
    fn blend_factor_half_weights_equally() {
        let fast = vec![hit("a", 10.0, 0), hit("b", 0.0, 1)];
        let quality = vec![hit("a", 0.0, 0), hit("b", 10.0, 1)];
        let blended = blend_two_tier(&fast, &quality, 0.5);
        let a_score = score_for("a", &blended);
        let b_score = score_for("b", &blended);
        assert!(
            (a_score - b_score).abs() <= EPSILON,
            "symmetric blend should produce equal scores: a={a_score}, b={b_score}"
        );
    }

    #[test]
    fn non_finite_blend_factor_falls_back_to_default() {
        let fast = vec![hit("a", 1.0, 0)];
        let quality = vec![hit("a", 1.0, 0)];
        let blended_nan = blend_two_tier(&fast, &quality, f32::NAN);
        let blended_default = blend_two_tier(&fast, &quality, 0.7);
        assert!(
            (blended_nan[0].score - blended_default[0].score).abs() <= EPSILON,
            "NaN blend_factor should fall back to 0.7"
        );
    }

    #[test]
    fn compute_rank_changes_identical_lists_are_all_stable() {
        let list = vec![hit("a", 1.0, 0), hit("b", 0.9, 1)];
        let changes = compute_rank_changes(&list, &list);
        assert_eq!(changes.stable, 2);
        assert_eq!(changes.promoted, 0);
        assert_eq!(changes.demoted, 0);
    }

    #[test]
    fn compute_rank_changes_empty_lists() {
        let changes = compute_rank_changes(&[], &[]);
        assert_eq!(changes.stable, 0);
        assert_eq!(changes.promoted, 0);
        assert_eq!(changes.demoted, 0);
    }

    #[test]
    #[ignore = "perf-only stress harness for optimization baseline/profile runs"]
    fn kendall_tau_stress_reverse_large() {
        let n: usize = 2_048;
        let initial: Vec<VectorHit> = (0..n)
            .map(|i| hit(&format!("doc-{i:05}"), 1.0, 0))
            .collect();
        let refined: Vec<VectorHit> = (0..n)
            .rev()
            .map(|i| hit(&format!("doc-{i:05}"), 1.0, 0))
            .collect();

        let tau = kendall_tau(&initial, &refined).expect("tau should exist for large overlap");
        assert!(
            (tau + 1.0).abs() <= f64::EPSILON,
            "reverse ordering should produce tau=-1.0, got {tau}"
        );
    }
}
