//! Differential parity + paired A/B for MRL rescored top-`limit` selection.
//!
//! `mrl.rs` rescores `rescore_top_k` (= 3·limit by default) candidates then keeps the
//! top `limit`. The legacy code full-sorted all of them (`sort_unstable_by`) and
//! truncated. The shipped code partitions to the top `limit` with `select_nth_unstable_by`
//! (O(n)) and sorts only those — but only for a large candidate set (`>= SELECT_NTH_MIN`);
//! small sets keep the full sort (select_nth's constant factors dominate there). Strict
//! total order (`score` + unique `index`) ⇒ byte-identical top-`limit`.
//!
//! Replicates the algorithm over a plain `{index, score}` struct (crate-agnostic).
//! Within-process paired AB/BA. `limit = n/3` matches the default `rescore_top_k = 3·limit`.
//!
//! Run: `rch exec -- cargo bench -p frankensearch-index --profile release --bench mrl_topk_select_ab`
#![allow(
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::trivially_copy_pass_by_ref
)]

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::{Duration, Instant};

const SELECT_NTH_MIN: usize = 256;
const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;
// (n rescored, limit); limit = n/3 mirrors the default rescore_top_k = 3*limit.
const SHAPES: &[(usize, usize)] = &[(90, 30), (300, 100), (768, 256), (3_000, 1_000)];

#[derive(Clone, Copy)]
struct Entry {
    index: u32,
    score: f32,
}

fn cmp(a: &Entry, b: &Entry) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.index.cmp(&b.index))
}

fn make_entries(n: usize) -> Vec<Entry> {
    (0..n)
        .map(|i| Entry {
            index: i as u32,
            // Many score ties (i % 41) broken by the unique index — realistic for
            // quantized rescore scores.
            score: (i % 41) as f32 + ((i * 7 % 11) as f32) * 0.1,
        })
        .collect()
}

fn apply_legacy(v: &mut Vec<Entry>, limit: usize) {
    v.sort_unstable_by(cmp);
    v.truncate(limit);
}

fn apply_select(v: &mut Vec<Entry>, limit: usize) {
    if limit < v.len() && v.len() >= SELECT_NTH_MIN {
        v.select_nth_unstable_by(limit, cmp);
        v.truncate(limit);
    }
    v.sort_unstable_by(cmp);
    v.truncate(limit);
}

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
    Select,
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

fn time_arm(base: &[Entry], limit: usize, arm: Arm) -> Duration {
    let mut v = base.to_vec(); // untimed reset
    let started = Instant::now();
    match arm {
        Arm::Legacy => apply_legacy(&mut v, limit),
        Arm::Select => apply_select(&mut v, limit),
    }
    let elapsed = started.elapsed();
    black_box(&v);
    elapsed
}

fn paired_ratio(base: &[Entry], limit: usize, a: Arm, b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let aa = time_arm(base, limit, a);
        let ab = time_arm(base, limit, b);
        let bb = time_arm(base, limit, b);
        let ba = time_arm(base, limit, a);
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

fn median_ns(base: &[Entry], limit: usize, arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(base, limit, arm));
    }
    let mut s: Vec<Duration> = (0..PROFILE_ROUNDS)
        .map(|_| time_arm(base, limit, arm))
        .collect();
    s.sort_unstable();
    percentile(&s, 50).as_secs_f64() * 1e9
}

fn key(v: &[Entry]) -> Vec<(u32, u32)> {
    v.iter().map(|e| (e.index, e.score.to_bits())).collect()
}

fn main() {
    eprintln!(
        "[profile-config] select_nth_min={SELECT_NTH_MIN} shapes={SHAPES:?} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}"
    );
    let mut any_clears = false;
    for &(n, limit) in SHAPES {
        let base = make_entries(n);
        let mut vl = base.clone();
        let mut vs = base.clone();
        apply_legacy(&mut vl, limit);
        apply_select(&mut vs, limit);
        assert_eq!(
            key(&vl),
            key(&vs),
            "select must equal legacy for n={n} limit={limit}"
        );
        let uses_select = limit < n && n >= SELECT_NTH_MIN;
        eprintln!("[parity] n={n} limit={limit} uses_select={uses_select} output_identical=true");

        let legacy_ns = median_ns(&base, limit, Arm::Legacy);
        let select_ns = median_ns(&base, limit, Arm::Select);
        eprintln!(
            "[profile] n={n} limit={limit} legacy_median_ns={legacy_ns:.1} select_median_ns={select_ns:.1}"
        );

        let null = paired_ratio(&base, limit, Arm::Legacy, Arm::Legacy);
        let lever = paired_ratio(&base, limit, Arm::Legacy, Arm::Select);
        eprintln!(
            "[paired] n={n} comparison=null_legacy_legacy median={:.6} p5={:.6} p95={:.6}",
            null.median, null.p5, null.p95
        );
        eprintln!(
            "[paired] n={n} comparison=select_vs_legacy median={:.6} p5={:.6} p95={:.6}",
            lever.median, lever.p5, lever.p95
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        any_clears |= gate_pass;
        eprintln!(
            "[gate] n={n} limit={limit} verdict={} median_speedup={:.6}x gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median
        );
    }
    eprintln!(
        "[gate-summary] decision={} any_shape_clears_null_floor={any_clears}",
        if any_clears { "KEEP" } else { "HOLD" }
    );
}
