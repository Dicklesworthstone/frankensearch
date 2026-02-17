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

const DEFAULT_BLEND_FACTOR: f32 = 0.7;
const NON_FINITE_SCORE_FALLBACK: f32 = 0.0;

fn robust_normalize(scores: &mut [f32]) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut saw_finite = false;

    for &value in scores.iter() {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
            saw_finite = true;
        }
    }

    if !saw_finite {
        scores.fill(NON_FINITE_SCORE_FALLBACK);
        return;
    }

    let range = max - min;
    // If range is significant, use min-max to align distributions.
    // If range is tiny (e.g. single result or all identical), preserve absolute value
    // but clamp to [0, 1] (assuming cosine-ish input).
    if range > 0.01 {
        for score in scores.iter_mut() {
            if score.is_finite() {
                *score = ((*score - min) / range).clamp(0.0, 1.0);
            } else {
                *score = NON_FINITE_SCORE_FALLBACK;
            }
        }
    } else {
        // Fallback: clamp to [0, 1].
        // This avoids turning "0.99" into "0.5" just because it's the only result.
        for score in scores.iter_mut() {
            if score.is_finite() {
                *score = score.clamp(0.0, 1.0);
            } else {
                *score = NON_FINITE_SCORE_FALLBACK;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ScorePair {
    fast: Option<f32>,
    quality: Option<f32>,
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
    robust_normalize(&mut fast_scores);
    robust_normalize(&mut quality_scores);

    let mut merged: HashMap<&str, ScorePair> =
        HashMap::with_capacity(fast_results.len() + quality_results.len());

    for (hit, normalized) in fast_results.iter().zip(fast_scores.into_iter()) {
        let entry = merged
            .entry(hit.doc_id.as_str())
            .or_insert_with(|| ScorePair {
                index: hit.index,
                ..ScorePair::default()
            });
        // fast_results are sorted best-first. Keep the first (best) score.
        if entry.fast.is_none() {
            entry.fast = Some(normalized);
            // Keep the index associated with the best fast score
            entry.index = hit.index;
        }
    }

    for (hit, normalized) in quality_results.iter().zip(quality_scores.into_iter()) {
        let entry = merged
            .entry(hit.doc_id.as_str())
            .or_insert_with(|| ScorePair {
                index: hit.index,
                ..ScorePair::default()
            });
        // quality_results are sorted best-first. Keep the first (best) score.
        if entry.quality.is_none() {
            entry.quality = Some(normalized);
        }
    }

    let mut blended: Vec<VectorHit> = merged
        .into_iter()
        .map(|(doc_id, pair)| {
            let fast = pair.fast.unwrap_or(0.0);
            let quality = pair.quality.unwrap_or(0.0);
            let score = alpha.mul_add(quality, (1.0 - alpha) * fast);
            VectorHit {
                index: pair.index,
                score: sanitize_score(score),
                doc_id: doc_id.to_owned(),
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
    let initial_rank = build_borrowed_rank_map(initial);
    let refined_rank = build_borrowed_rank_map(refined);
    compute_rank_changes_with_maps(&initial_rank, &refined_rank)
}

/// Compute rank changes using precomputed rank maps.
///
/// Use this when calling both `compute_rank_changes` and `kendall_tau` on the
/// same inputs — precompute the maps once via [`build_borrowed_rank_map`] and
/// pass them to both functions to avoid redundant `HashMap` construction.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn compute_rank_changes_with_maps(
    initial_rank: &HashMap<&str, usize>,
    refined_rank: &HashMap<&str, usize>,
) -> RankChanges {
    let mut promoted = 0;
    let mut demoted = 0;
    let mut stable = 0;

    for (doc_id, old_rank) in initial_rank {
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

/// Compute Kendall's tau using a precomputed refined rank map.
///
/// Use this when calling both `compute_rank_changes` and `kendall_tau` on the
/// same inputs — precompute the refined rank map via [`build_borrowed_rank_map`]
/// and pass it to both functions.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn kendall_tau_with_refined_rank(
    initial: &[VectorHit],
    refined_rank: &HashMap<&str, usize>,
) -> Option<f64> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(initial.len());
    let mut refined_ranks = Vec::with_capacity(initial.len().min(refined_rank.len()));
    for hit in initial {
        let doc_id = hit.doc_id.as_str();
        if let Some(&rank) = refined_rank.get(doc_id)
            && seen.insert(doc_id)
        {
            refined_ranks.push(rank);
        }
    }

    let n = refined_ranks.len();
    if n < 2 {
        return None;
    }

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

/// Compute Kendall's tau rank correlation between two rankings.
///
/// Uses merge-sort-based inversion counting for O(n log n) performance
/// instead of the naive O(n^2) pairwise comparison.
///
/// Returns `None` when fewer than two common documents exist.
#[must_use]
pub fn kendall_tau(initial: &[VectorHit], refined: &[VectorHit]) -> Option<f64> {
    let refined_rank = build_borrowed_rank_map(refined);
    kendall_tau_with_refined_rank(initial, &refined_rank)
}

/// Count inversions in a slice using merge sort. O(n log n).
///
/// An inversion is a pair `(i, j)` where `i < j` but `arr[i] > arr[j]`.
/// The slice is sorted in place as a side effect.
fn merge_sort_inversions(arr: &mut [usize]) -> u64 {
    if arr.len() <= 1 {
        return 0;
    }
    let mut scratch = vec![0_usize; arr.len()];
    merge_sort_inversions_with_scratch(arr, &mut scratch)
}

fn merge_sort_inversions_with_scratch(arr: &mut [usize], scratch: &mut [usize]) -> u64 {
    let n = arr.len();
    if n <= 1 {
        return 0;
    }
    let mid = n / 2;
    let (left, right) = arr.split_at_mut(mid);
    let (scratch_left, scratch_right) = scratch.split_at_mut(mid);

    let mut count = merge_sort_inversions_with_scratch(left, scratch_left);
    count = count.saturating_add(merge_sort_inversions_with_scratch(right, scratch_right));

    let (mut i, mut j, mut out) = (0_usize, 0_usize, 0_usize);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            scratch[out] = left[i];
            i += 1;
        } else {
            scratch[out] = right[j];
            let remaining_left = left.len().saturating_sub(i);
            let remaining_left_u64 = u64::try_from(remaining_left).unwrap_or(u64::MAX);
            count = count.saturating_add(remaining_left_u64);
            j += 1;
        }
        out += 1;
    }

    if i < left.len() {
        let left_remaining = left.len() - i;
        scratch[out..out + left_remaining].copy_from_slice(&left[i..]);
        out += left_remaining;
    }
    if j < right.len() {
        scratch[out..n].copy_from_slice(&right[j..]);
    }

    arr.copy_from_slice(&scratch[..n]);
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

/// Build a rank map from `doc_id` to position for a list of vector hits.
///
/// First occurrence of each `doc_id` determines its rank.
#[must_use]
pub fn build_borrowed_rank_map(hits: &[VectorHit]) -> HashMap<&str, usize> {
    let mut ranks = HashMap::with_capacity(hits.len());
    for (rank, hit) in hits.iter().enumerate() {
        ranks.entry(hit.doc_id.as_str()).or_insert(rank);
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

        // Degenerate single-entry normalization clamps source scores to 1.0.
        assert!((fast_only - 0.3).abs() <= EPSILON);
        assert!((quality_only - 0.7).abs() <= EPSILON);
    }

    #[test]
    fn equal_scores_remain_equal() {
        let fast = vec![hit("same", 1.0, 0), hit("other", 1.0, 1)];
        let quality = vec![hit("same", 2.0, 0), hit("other", 2.0, 1)];
        let blended = blend_two_tier(&fast, &quality, 0.7);

        assert!((score_for("same", &blended) - 1.0).abs() <= EPSILON);
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
    fn kendall_tau_partial_overlap() {
        // initial: a(0), b(1), c(2), d(3)
        // refined: c(0), x(1), a(2), d(3)
        // common (initial order): a, c, d
        // refined ranks of common: a→2, c→0, d→3  →  [2, 0, 3]
        // inversions in [2, 0, 3]: (2,0) → 1 inversion
        // total_pairs = 3, concordant = 2, discordant = 1
        // tau = (2 - 1) / 3 = 1/3
        let initial = vec![
            hit("a", 1.0, 0),
            hit("b", 0.9, 1),
            hit("c", 0.8, 2),
            hit("d", 0.7, 3),
        ];
        let refined = vec![
            hit("c", 1.0, 2),
            hit("x", 0.9, 4),
            hit("a", 0.8, 0),
            hit("d", 0.7, 3),
        ];
        let tau = kendall_tau(&initial, &refined).expect("tau for partial overlap");
        assert!((tau - 1.0 / 3.0).abs() < 1e-10, "expected 1/3, got {tau}");
    }

    #[test]
    fn kendall_tau_two_elements() {
        let initial = vec![hit("a", 1.0, 0), hit("b", 0.5, 1)];
        let refined = vec![hit("b", 1.0, 1), hit("a", 0.5, 0)];
        let tau = kendall_tau(&initial, &refined).expect("tau for 2 elements");
        assert!(
            (tau + 1.0).abs() <= f64::EPSILON,
            "swapped pair should give tau=-1.0, got {tau}"
        );
    }

    #[test]
    fn kendall_tau_medium_sorted() {
        // Already sorted: 100 docs in same order → tau = 1.0
        let n = 100;
        let initial: Vec<VectorHit> = (0..n)
            .map(|i| hit(&format!("doc-{i:04}"), 1.0, 0))
            .collect();
        let refined: Vec<VectorHit> = (0..n)
            .map(|i| hit(&format!("doc-{i:04}"), 0.5, 0))
            .collect();
        let tau = kendall_tau(&initial, &refined).expect("tau for 100 identical order");
        assert!(
            (tau - 1.0).abs() <= f64::EPSILON,
            "same ordering should give tau=1.0, got {tau}"
        );
    }

    #[test]
    fn kendall_tau_medium_reversed() {
        // Fully reversed: 100 docs → tau = -1.0
        let n = 100;
        let initial: Vec<VectorHit> = (0..n)
            .map(|i| hit(&format!("doc-{i:04}"), 1.0, 0))
            .collect();
        let refined: Vec<VectorHit> = (0..n)
            .rev()
            .map(|i| hit(&format!("doc-{i:04}"), 0.5, 0))
            .collect();
        let tau = kendall_tau(&initial, &refined).expect("tau for 100 reversed");
        assert!(
            (tau + 1.0).abs() <= f64::EPSILON,
            "reverse ordering should give tau=-1.0, got {tau}"
        );
    }

    fn shuffle_deterministic(values: &mut [usize], seed: u64) {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for i in (1..values.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            let modulus = u64::try_from(i + 1).expect("modulus fits into u64");
            let j_u64 = state % modulus;
            let j = usize::try_from(j_u64).expect("index fits into usize");
            values.swap(i, j);
        }
    }

    fn naive_tau_from_refined_ranks(ranks: &[usize]) -> f64 {
        let n = ranks.len();
        let mut discordant = 0_u64;
        for i in 0..n {
            for j in (i + 1)..n {
                if ranks[i] > ranks[j] {
                    discordant += 1;
                }
            }
        }

        let n_u64 = u64::try_from(n).expect("length fits into u64");
        let total_pairs = n_u64.saturating_mul(n_u64.saturating_sub(1)) / 2;
        let concordant = total_pairs.saturating_sub(discordant);

        let concordant_f64 = f64::from(u32::try_from(concordant).expect("fits into u32"));
        let discordant_f64 = f64::from(u32::try_from(discordant).expect("fits into u32"));
        let total_pairs_f64 = f64::from(u32::try_from(total_pairs).expect("fits into u32"));
        (concordant_f64 - discordant_f64) / total_pairs_f64
    }

    #[test]
    fn kendall_tau_matches_naive_for_deterministic_permutations() {
        let sizes = [2_usize, 3, 5, 8, 16, 32];

        for &n in &sizes {
            let initial: Vec<VectorHit> = (0..n)
                .map(|i| hit(&format!("doc-{i:04}"), 1.0, 0))
                .collect();

            for seed in 0_u64..12 {
                let mut order: Vec<usize> = (0..n).collect();
                let n_u64 = u64::try_from(n).expect("size fits into u64");
                shuffle_deterministic(&mut order, seed.wrapping_add(n_u64));

                let refined: Vec<VectorHit> = order
                    .iter()
                    .map(|&idx| hit(&format!("doc-{idx:04}"), 0.5, 0))
                    .collect();

                let mut refined_rank_by_initial_index = vec![0_usize; n];
                for (rank, &idx) in order.iter().enumerate() {
                    refined_rank_by_initial_index[idx] = rank;
                }

                let expected = naive_tau_from_refined_ranks(&refined_rank_by_initial_index);
                let actual =
                    kendall_tau(&initial, &refined).expect("tau for deterministic permutation");

                assert!(
                    (actual - expected).abs() < 1e-12,
                    "n={n}, seed={seed}, expected={expected}, actual={actual}"
                );
            }
        }
    }

    #[test]
    fn merge_sort_inversions_counts_correctly() {
        use super::merge_sort_inversions;

        // [2, 0, 3] → 1 inversion: (2, 0)
        let mut arr = vec![2, 0, 3];
        assert_eq!(merge_sort_inversions(&mut arr), 1);
        assert_eq!(arr, [0, 2, 3]); // sorted as side effect

        // [3, 2, 1, 0] → 6 inversions (fully reversed, n=4)
        let mut arr = vec![3, 2, 1, 0];
        assert_eq!(merge_sort_inversions(&mut arr), 6);

        // [0, 1, 2, 3] → 0 inversions (already sorted)
        let mut arr = vec![0, 1, 2, 3];
        assert_eq!(merge_sort_inversions(&mut arr), 0);

        // empty and single
        assert_eq!(merge_sort_inversions(&mut []), 0);
        assert_eq!(merge_sort_inversions(&mut [42]), 0);
    }

    #[test]
    #[ignore = "perf-only stress harness for optimization baseline/profile runs"]
    fn kendall_tau_stress_reverse_large() {
        let n: usize = 4_096;
        let iterations: usize = 24;
        let initial: Vec<VectorHit> = (0..n)
            .map(|i| hit(&format!("doc-{i:05}"), 1.0, 0))
            .collect();
        let refined: Vec<VectorHit> = (0..n)
            .rev()
            .map(|i| hit(&format!("doc-{i:05}"), 1.0, 0))
            .collect();

        for _ in 0..iterations {
            let tau = kendall_tau(&initial, &refined).expect("tau should exist for large overlap");
            assert!(
                (tau + 1.0).abs() <= f64::EPSILON,
                "reverse ordering should produce tau=-1.0, got {tau}"
            );
        }
    }

    // ---- Tests for rank-map caching APIs (bd-1wgy) ----

    use super::{
        build_borrowed_rank_map, compute_rank_changes_with_maps, kendall_tau_with_refined_rank,
    };

    #[test]
    fn build_borrowed_rank_map_basic() {
        let hits = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let map = build_borrowed_rank_map(&hits);
        assert_eq!(map.len(), 3);
        assert_eq!(map["a"], 0);
        assert_eq!(map["b"], 1);
        assert_eq!(map["c"], 2);
    }

    #[test]
    fn build_borrowed_rank_map_empty() {
        let map = build_borrowed_rank_map(&[]);
        assert!(map.is_empty());
    }

    #[test]
    fn build_borrowed_rank_map_single() {
        let hits = vec![hit("only", 1.0, 0)];
        let map = build_borrowed_rank_map(&hits);
        assert_eq!(map.len(), 1);
        assert_eq!(map["only"], 0);
    }

    #[test]
    fn build_borrowed_rank_map_first_occurrence_wins() {
        // Duplicate doc_id: first occurrence should determine rank.
        let hits = vec![hit("dup", 1.0, 0), hit("other", 0.9, 1), hit("dup", 0.5, 2)];
        let map = build_borrowed_rank_map(&hits);
        assert_eq!(map.len(), 2);
        assert_eq!(map["dup"], 0, "first occurrence at rank 0 should win");
        assert_eq!(map["other"], 1);
    }

    #[test]
    fn compute_rank_changes_with_maps_matches_original() {
        let initial = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let refined = vec![hit("b", 1.0, 1), hit("a", 0.9, 0), hit("d", 0.7, 3)];

        let direct = compute_rank_changes(&initial, &refined);

        let initial_map = build_borrowed_rank_map(&initial);
        let refined_map = build_borrowed_rank_map(&refined);
        let via_maps = compute_rank_changes_with_maps(&initial_map, &refined_map);

        assert_eq!(direct.promoted, via_maps.promoted);
        assert_eq!(direct.demoted, via_maps.demoted);
        assert_eq!(direct.stable, via_maps.stable);
    }

    #[test]
    fn compute_rank_changes_with_maps_empty() {
        let empty = std::collections::HashMap::new();
        let changes = compute_rank_changes_with_maps(&empty, &empty);
        assert_eq!(changes.promoted, 0);
        assert_eq!(changes.demoted, 0);
        assert_eq!(changes.stable, 0);
    }

    #[test]
    fn compute_rank_changes_with_maps_all_new() {
        let initial_hits = vec![hit("a", 1.0, 0), hit("b", 0.9, 1)];
        let refined_hits = vec![hit("x", 1.0, 2), hit("y", 0.9, 3)];
        let initial_map = build_borrowed_rank_map(&initial_hits);
        let refined_map = build_borrowed_rank_map(&refined_hits);
        let changes = compute_rank_changes_with_maps(&initial_map, &refined_map);
        assert_eq!(changes.promoted, 2, "all new docs are promoted");
        assert_eq!(changes.demoted, 2, "all old docs are demoted");
        assert_eq!(changes.stable, 0);
    }

    #[test]
    fn compute_rank_changes_with_maps_identical() {
        let hits = vec![hit("a", 1.0, 0), hit("b", 0.9, 1)];
        let map = build_borrowed_rank_map(&hits);
        let changes = compute_rank_changes_with_maps(&map, &map);
        assert_eq!(changes.stable, 2);
        assert_eq!(changes.promoted, 0);
        assert_eq!(changes.demoted, 0);
    }

    #[test]
    fn kendall_tau_with_refined_rank_matches_original() {
        let initial = vec![
            hit("a", 1.0, 0),
            hit("b", 0.9, 1),
            hit("c", 0.8, 2),
            hit("d", 0.7, 3),
        ];
        let refined = vec![
            hit("c", 1.0, 2),
            hit("x", 0.9, 4),
            hit("a", 0.8, 0),
            hit("d", 0.7, 3),
        ];

        let direct = kendall_tau(&initial, &refined).expect("tau via original");

        let refined_map = build_borrowed_rank_map(&refined);
        let via_map = kendall_tau_with_refined_rank(&initial, &refined_map).expect("tau via map");

        assert!(
            (direct - via_map).abs() < 1e-12,
            "expected same tau: direct={direct}, via_map={via_map}"
        );
    }

    #[test]
    fn kendall_tau_with_refined_rank_identical() {
        let hits = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let map = build_borrowed_rank_map(&hits);
        let tau = kendall_tau_with_refined_rank(&hits, &map).expect("tau for identical");
        assert!(
            (tau - 1.0).abs() <= f64::EPSILON,
            "identical rankings should give tau=1.0, got {tau}"
        );
    }

    #[test]
    fn kendall_tau_with_refined_rank_reversed() {
        let initial = vec![hit("a", 1.0, 0), hit("b", 0.9, 1), hit("c", 0.8, 2)];
        let reversed = vec![hit("c", 1.0, 2), hit("b", 0.9, 1), hit("a", 0.8, 0)];
        let map = build_borrowed_rank_map(&reversed);
        let tau = kendall_tau_with_refined_rank(&initial, &map).expect("tau for reversed");
        assert!(
            (tau + 1.0).abs() <= f64::EPSILON,
            "reversed rankings should give tau=-1.0, got {tau}"
        );
    }

    #[test]
    fn kendall_tau_with_refined_rank_insufficient_overlap() {
        let initial = vec![hit("a", 1.0, 0)];
        let refined = vec![hit("b", 0.9, 1)];
        let map = build_borrowed_rank_map(&refined);
        assert!(kendall_tau_with_refined_rank(&initial, &map).is_none());
    }

    #[test]
    fn precompute_once_use_twice_pattern() {
        // Demonstrates the intended usage pattern: build maps once, use for both
        // rank changes and kendall_tau.
        let initial = vec![
            hit("a", 1.0, 0),
            hit("b", 0.9, 1),
            hit("c", 0.8, 2),
            hit("d", 0.7, 3),
        ];
        let refined = vec![
            hit("b", 1.0, 1),
            hit("c", 0.95, 2),
            hit("a", 0.9, 0),
            hit("e", 0.6, 4),
        ];

        let initial_map = build_borrowed_rank_map(&initial);
        let refined_map = build_borrowed_rank_map(&refined);

        let changes = compute_rank_changes_with_maps(&initial_map, &refined_map);
        let tau = kendall_tau_with_refined_rank(&initial, &refined_map).expect("tau");

        // Verify results match the non-map versions.
        let direct_changes = compute_rank_changes(&initial, &refined);
        let direct_tau = kendall_tau(&initial, &refined).expect("direct tau");

        assert_eq!(changes.promoted, direct_changes.promoted);
        assert_eq!(changes.demoted, direct_changes.demoted);
        assert_eq!(changes.stable, direct_changes.stable);
        assert!(
            (tau - direct_tau).abs() < 1e-12,
            "tau mismatch: precomputed={tau}, direct={direct_tau}"
        );
    }

    #[test]
    fn kendall_tau_with_refined_rank_deterministic_permutations() {
        // Verify the map-based API matches the original across many permutations.
        let sizes = [3_usize, 5, 8, 16];
        for &n in &sizes {
            let initial: Vec<VectorHit> = (0..n)
                .map(|i| hit(&format!("doc-{i:04}"), 1.0, 0))
                .collect();

            for seed in 0_u64..8 {
                let mut order: Vec<usize> = (0..n).collect();
                let n_u64 = u64::try_from(n).expect("size fits into u64");
                shuffle_deterministic(&mut order, seed.wrapping_add(n_u64));

                let refined: Vec<VectorHit> = order
                    .iter()
                    .map(|&idx| hit(&format!("doc-{idx:04}"), 0.5, 0))
                    .collect();

                let direct = kendall_tau(&initial, &refined);
                let map = build_borrowed_rank_map(&refined);
                let via_map = kendall_tau_with_refined_rank(&initial, &map);

                assert_eq!(
                    direct.is_some(),
                    via_map.is_some(),
                    "n={n}, seed={seed}: Some/None mismatch"
                );

                if let (Some(d), Some(m)) = (direct, via_map) {
                    assert!(
                        (d - m).abs() < 1e-12,
                        "n={n}, seed={seed}: direct={d}, via_map={m}"
                    );
                }
            }
        }
    }
}
