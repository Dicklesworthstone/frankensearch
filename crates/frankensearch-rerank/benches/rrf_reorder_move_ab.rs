//! Differential parity + paired A/B for `apply_rrf_combine`'s permutation write-back.
//!
//! The reorder gathers the permuted window into a `reordered` snapshot (one clone per
//! slot, unavoidable — the gather reads positions it would otherwise overwrite). The
//! legacy tail then did `window.clone_from_slice(&reordered)` — cloning every element a
//! SECOND time (2N clones total). The shipped tail MOVES the snapshot back (N clones).
//! Byte-identical final order. Replicates `pipeline.rs::apply_rrf_combine` (private) over
//! real `ScoredResult`s, matching the existing `combine_reorder_cost_ab` pattern.
//!
//! Compiles WITHOUT the heavy `native` feature. Within-process paired AB/BA.
//!
//! Run: `rch exec -- cargo bench -p frankensearch-rerank --profile release
//!   --bench rrf_reorder_move_ab`
#![allow(clippy::doc_markdown, clippy::cast_precision_loss)]

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_core::types::{ScoreSource, ScoredResult};

const K: f64 = 60.0;
const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;
const WINDOW_SIZES: &[usize] = &[32, 128, 512];

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
}

impl RatioDistribution {
    /// Largest `|median - 1|` an A/A control may show and still be admissible.
    /// Mirrors `frankensearch_core::bench_support::NULL_MEDIAN_TOLERANCE`.
    const NULL_MEDIAN_TOLERANCE: f64 = 0.02;

    /// A/A control admissibility: gate the null's accuracy, never its precision.
    ///
    /// Requiring `p5 <= 1.0 <= p95` vetoed a null for being *tight*: the
    /// narrower its spread, the likelier it excluded 1.0 — however small the
    /// bias and however large the effect (`bd-pjh09`). The spread still sets
    /// the effect floor below; only the median says whether the sampler itself
    /// is trustworthy.
    fn null_is_unbiased(self) -> bool {
        (self.median - 1.0).abs() <= Self::NULL_MEDIAN_TOLERANCE
    }
    fn verdict_against(self, null: Self) -> &'static str {
        if !null.null_is_unbiased() {
            "BIASED_NULL_UNDECIDABLE"
        } else if self.median < null.p5 {
            "CANDIDATE_FASTER"
        } else if self.median > null.p95 {
            "CANDIDATE_SLOWER"
        } else {
            "INSIDE_NULL_FLOOR"
        }
    }
}

#[derive(Clone, Copy)]
enum Arm {
    Legacy,
    Move,
}

fn candidate(i: usize, rerank_score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: format!("doc-{i:06}").into(),
        score: 1.0 - (i as f32) * 0.001,
        source: ScoreSource::Reranked,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: None,
        rerank_score: Some(rerank_score),
        explanation: None,
        metadata: None,
    }
}

fn window(n: usize) -> Vec<ScoredResult> {
    (0..n)
        .map(|i| {
            let rs = ((n - i) as f32) + ((i * 7 % 13) as f32) * 0.1;
            candidate(i, rs)
        })
        .collect()
}

fn finite(c: &ScoredResult) -> f32 {
    c.rerank_score
        .filter(|s| s.is_finite())
        .unwrap_or(f32::NEG_INFINITY)
}

fn cmp_rerank(a: &ScoredResult, b: &ScoredResult) -> Ordering {
    finite(b)
        .total_cmp(&finite(a))
        .then_with(|| a.doc_id.cmp(&b.doc_id))
}

#[derive(Clone, Copy)]
struct RrfOrder {
    position: usize,
    fused_key: f64,
}

/// Shared order computation (identical to `apply_rrf_combine`), then a tail closure.
fn reorder_with_tail(win: &mut [ScoredResult], tail: impl Fn(&mut [ScoredResult], Vec<usize>)) {
    let n = win.len();
    if n < 2 {
        return;
    }
    let mut order: Vec<RrfOrder> = (0..n)
        .map(|position| RrfOrder {
            position,
            fused_key: 0.0,
        })
        .collect();
    order.sort_by(|a, b| cmp_rerank(&win[a.position], &win[b.position]));
    for (rerank_rank, entry) in order.iter_mut().enumerate() {
        entry.fused_key = 1.0 / (K + entry.position as f64) + 1.0 / (K + rerank_rank as f64);
    }
    order.sort_by(|a, b| {
        b.fused_key
            .total_cmp(&a.fused_key)
            .then_with(|| win[a.position].doc_id.cmp(&win[b.position].doc_id))
    });
    let perm: Vec<usize> = order.into_iter().map(|e| e.position).collect();
    tail(win, perm);
}

/// Legacy tail: gather snapshot (N clones) then `clone_from_slice` (N more clones).
fn apply_legacy(win: &mut [ScoredResult]) {
    reorder_with_tail(win, |win, perm| {
        let reordered: Vec<ScoredResult> = perm.into_iter().map(|i| win[i].clone()).collect();
        win.clone_from_slice(&reordered);
    });
}

/// Shipped tail: gather snapshot (N clones) then MOVE back (0 further clones).
fn apply_move(win: &mut [ScoredResult]) {
    reorder_with_tail(win, |win, perm| {
        let reordered: Vec<ScoredResult> = perm.into_iter().map(|i| win[i].clone()).collect();
        for (slot, value) in win.iter_mut().zip(reordered) {
            *slot = value;
        }
    });
}

fn key(win: &[ScoredResult]) -> Vec<(String, u32)> {
    win.iter()
        .map(|c| (c.doc_id.to_string(), c.score.to_bits()))
        .collect()
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    sorted[((sorted.len() - 1) * pct + 50) / 100]
}

fn ratio_distribution(mut s: Vec<f64>) -> RatioDistribution {
    s.sort_unstable_by(f64::total_cmp);
    let idx = |p: usize| ((s.len() - 1) * p + 50) / 100;
    RatioDistribution {
        median: s[idx(50)],
        p5: s[idx(5)],
        p95: s[idx(95)],
    }
}

fn time_arm(base: &[ScoredResult], arm: Arm) -> Duration {
    let mut win = base.to_vec(); // untimed reset (in-place reorder mutates the window)
    let started = Instant::now();
    match arm {
        Arm::Legacy => apply_legacy(&mut win),
        Arm::Move => apply_move(&mut win),
    }
    let elapsed = started.elapsed();
    black_box(&win);
    elapsed
}

fn paired_ratio(base: &[ScoredResult], a: Arm, b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let aa = time_arm(base, a);
        let ab = time_arm(base, b);
        let bb = time_arm(base, b);
        let ba = time_arm(base, a);
        if record {
            Some(
                ((ab.as_secs_f64() / aa.as_secs_f64()) * (bb.as_secs_f64() / ba.as_secs_f64()))
                    .sqrt(),
            )
        } else {
            black_box((aa, ab, bb, ba));
            None
        }
    };
    for _ in 0..3 {
        let _ = run_pair(false);
    }
    ratio_distribution(
        (0..PAIRED_ROUND_PAIRS)
            .map(|_| run_pair(true).expect("pair"))
            .collect(),
    )
}

fn median_ns(base: &[ScoredResult], arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(base, arm));
    }
    let mut s: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(base, arm)).collect();
    s.sort_unstable();
    percentile(&s, 50).as_secs_f64() * 1e9
}

fn main() {
    eprintln!(
        "[profile-config] window_sizes={WINDOW_SIZES:?} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}"
    );
    let mut any_clears = false;
    for &n in WINDOW_SIZES {
        let base = window(n);
        let mut wl = base.clone();
        let mut wm = base.clone();
        apply_legacy(&mut wl);
        apply_move(&mut wm);
        assert_eq!(
            key(&wl),
            key(&wm),
            "move tail must equal clone_from_slice for n={n}"
        );
        eprintln!("[parity] window={n} output_identical=true");

        let legacy_ns = median_ns(&base, Arm::Legacy);
        let move_ns = median_ns(&base, Arm::Move);
        eprintln!(
            "[profile] window={n} legacy_median_ns={legacy_ns:.2} move_median_ns={move_ns:.2}"
        );

        let null = paired_ratio(&base, Arm::Legacy, Arm::Legacy);
        let lever = paired_ratio(&base, Arm::Legacy, Arm::Move);
        eprintln!(
            "[paired] window={n} comparison=null_legacy_legacy median={:.6} p5={:.6} p95={:.6}",
            null.median, null.p5, null.p95
        );
        eprintln!(
            "[paired] window={n} comparison=move_vs_legacy median={:.6} p5={:.6} p95={:.6}",
            lever.median, lever.p5, lever.p95
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        any_clears |= gate_pass;
        eprintln!(
            "[gate] window={n} verdict={} median_speedup={:.6}x gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median
        );
    }
    eprintln!(
        "[gate-summary] decision={} any_shape_clears_null_floor={any_clears}",
        if any_clears { "KEEP" } else { "HOLD" }
    );
}
