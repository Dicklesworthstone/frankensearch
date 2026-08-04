//! Differential parity + paired A/B for the fleet monitor's per-project value build.
//!
//! `FleetOverviewScreen::selected_monitor_lines` builds five per-project value
//! vectors (docs/pending/cpu/memory/p95) for percentile ranking. The legacy code used
//! five separate filtered iterations over `fleet.instances`, calling `resources.get`
//! twice per matching instance (cpu + memory). The shipped path fuses them into one
//! pass that hits each map once. Byte-identical (percentile rank is order-independent;
//! cpu/memory stay aligned on the same `resources.get(..).is_some()` gate).
//!
//! Parity is proven over several fleet shapes before timing; timing is a within-process
//! paired AB/BA ratio with an A/A null floor (both arms on the same worker, immune to
//! the RCH_WORKER soft-pin issue).
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release \
//!     --features bench-internals --bench fleet_project_values_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_ops::screens::fleet::{
    BenchFleetValues, bench_make_fleet_values, bench_project_values_fused,
    bench_project_values_legacy,
};

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// (n_instances, n_projects); ~40% of instances land in the selected project.
const SHAPES: &[(usize, usize)] = &[(2_048, 24), (8_192, 48), (16_384, 64)];

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

fn run_arm(fleet: &BenchFleetValues, arm: Arm) -> usize {
    let out = match arm {
        Arm::Legacy => bench_project_values_legacy(fleet),
        Arm::Fused => bench_project_values_fused(fleet),
    };
    out.0.len() + out.2.len() + out.4.len()
}

fn time_arm(fleet: &BenchFleetValues, arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(fleet, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(fleet: &BenchFleetValues, arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let ab_a = time_arm(fleet, arm_a);
        let ab_b = time_arm(fleet, arm_b);
        let ba_b = time_arm(fleet, arm_b);
        let ba_a = time_arm(fleet, arm_a);
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

fn median_us(fleet: &BenchFleetValues, arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(fleet, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(fleet, arm)).collect();
    samples.sort_unstable();
    percentile(&samples, 50).as_secs_f64() * 1e6
}

fn main() {
    eprintln!(
        "[profile-config] shapes={} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}",
        SHAPES.len()
    );
    eprintln!(
        "[profile-config] binary_path={}",
        std::env::current_exe()
            .expect("resolve measured binary")
            .display()
    );

    let mut all_gates_pass = true;
    for &(n_instances, n_projects) in SHAPES {
        let fleet = bench_make_fleet_values(n_instances, n_projects);

        let legacy = bench_project_values_legacy(&fleet);
        let fused = bench_project_values_fused(&fleet);
        assert_eq!(
            legacy, fused,
            "fused builder must be byte-identical to legacy for shape {n_instances}/{n_projects}"
        );
        eprintln!(
            "[parity] shape={n_instances}/{n_projects} project_instances={} output_identical=true",
            fused.0.len()
        );

        let legacy_us = median_us(&fleet, Arm::Legacy);
        let fused_us = median_us(&fleet, Arm::Fused);
        eprintln!(
            "[profile] shape={n_instances}/{n_projects} legacy_median_us={legacy_us:.4} fused_median_us={fused_us:.4}"
        );

        let null = paired_ratio(&fleet, Arm::Legacy, Arm::Legacy);
        let lever = paired_ratio(&fleet, Arm::Legacy, Arm::Fused);
        eprintln!(
            "[paired] shape={n_instances}/{n_projects} comparison=null_legacy_legacy median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            null.median, null.p5, null.p95, null.round_pairs
        );
        eprintln!(
            "[paired] shape={n_instances}/{n_projects} comparison=fused_vs_legacy median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            lever.median, lever.p5, lever.p95, lever.round_pairs
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        all_gates_pass &= gate_pass;
        eprintln!(
            "[gate] shape={n_instances}/{n_projects} verdict={} median_speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
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
