//! Differential parity + paired A/B for the alerts/SLO fleet rollup row.
//!
//! `AlertsSloScreen::fleet_rollup_row` aggregates the per-project SLO rows into a
//! single "fleet" summary row. The legacy code made eight separate passes over
//! `project_rows` (three integer sums, two instance-weighted sums, one max, two
//! saturation-risk `any` scans). The shipped path fuses them into one fold that
//! visits each row once. Byte-identical (integer sums are order-independent; the
//! f64 weighted-burn folds in the same row order; `backlog_eta_s` is `u64` so a
//! 0-seeded max is exact; high>elevated>low priority is unchanged).
//!
//! Sibling of `fleet_project_values_ab` (the fleet.rs 5→1 fusion, commit
//! f4f98e0a). Within-process paired AB/BA ratio with an A/A null floor — both
//! arms on the same worker, immune to the RCH_WORKER soft-pin issue.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release \
//!     --features bench-internals --bench fleet_rollup_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_ops::screens::alerts_slo::{
    BenchSloRollup, bench_fleet_rollup_fused, bench_fleet_rollup_slow, bench_make_slo_rollup,
};

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// Project counts feeding the rollup (per-project rows, not instances).
const SHAPES: &[usize] = &[24, 128, 512, 2_048];

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
    round_pairs: usize,
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
    Fused,
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let idx = ((sorted.len() - 1) * pct + 50) / 100;
    sorted[idx]
}

fn ratio_distribution(mut samples: Vec<f64>) -> RatioDistribution {
    samples.sort_unstable_by(f64::total_cmp);
    let index = |pct: usize| ((samples.len() - 1) * pct + 50) / 100;
    RatioDistribution {
        median: samples[index(50)],
        p5: samples[index(5)],
        p95: samples[index(95)],
        round_pairs: samples.len(),
    }
}

fn run_arm(input: &BenchSloRollup, arm: Arm) -> usize {
    let out = match arm {
        Arm::Legacy => bench_fleet_rollup_slow(input),
        Arm::Fused => bench_fleet_rollup_fused(input),
    };
    out.map_or(0, |bits| bits.0)
}

fn time_arm(input: &BenchSloRollup, arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(input, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(input: &BenchSloRollup, arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let ab_a = time_arm(input, arm_a);
        let ab_b = time_arm(input, arm_b);
        let ba_b = time_arm(input, arm_b);
        let ba_a = time_arm(input, arm_a);
        if record {
            let ab = ab_b.as_secs_f64() / ab_a.as_secs_f64();
            let ba = ba_b.as_secs_f64() / ba_a.as_secs_f64();
            Some((ab * ba).sqrt())
        } else {
            black_box((ab_a, ab_b, ba_b, ba_a));
            None
        }
    };
    for _ in 0..3 {
        let _ = run_pair(false);
    }
    ratio_distribution(
        (0..PAIRED_ROUND_PAIRS)
            .map(|_| run_pair(true).expect("recorded round pair"))
            .collect(),
    )
}

fn median_us(input: &BenchSloRollup, arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(input, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(input, arm)).collect();
    samples.sort_unstable();
    percentile(&samples, 50).as_secs_f64() * 1e6
}

fn main() {
    eprintln!(
        "[profile-config] shapes={} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}",
        SHAPES.len()
    );

    let mut all_gates_pass = true;
    for &n_projects in SHAPES {
        let input = bench_make_slo_rollup(n_projects);

        let legacy = bench_fleet_rollup_slow(&input);
        let fused = bench_fleet_rollup_fused(&input);
        assert_eq!(
            legacy, fused,
            "fused rollup must be byte-identical to legacy for n_projects={n_projects}"
        );
        eprintln!("[parity] n_projects={n_projects} output_identical=true");

        let legacy_us = median_us(&input, Arm::Legacy);
        let fused_us = median_us(&input, Arm::Fused);
        eprintln!(
            "[profile] n_projects={n_projects} legacy_median_us={legacy_us:.4} fused_median_us={fused_us:.4}"
        );

        let null = paired_ratio(&input, Arm::Legacy, Arm::Legacy);
        let lever = paired_ratio(&input, Arm::Legacy, Arm::Fused);
        eprintln!(
            "[paired] n_projects={n_projects} comparison=null_legacy_legacy median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            null.median, null.p5, null.p95, null.round_pairs
        );
        eprintln!(
            "[paired] n_projects={n_projects} comparison=fused_vs_legacy median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            lever.median, lever.p5, lever.p95, lever.round_pairs
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        all_gates_pass &= gate_pass;
        eprintln!(
            "[gate] n_projects={n_projects} verdict={} median_speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median,
            null.null_is_unbiased(),
            lever.median < null.p5
        );
    }
    eprintln!(
        "[gate-summary] decision={} all_shapes_clear_null_floor={all_gates_pass}",
        if all_gates_pass { "KEEP" } else { "HOLD" }
    );
}
