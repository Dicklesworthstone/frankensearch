//! Differential parity + paired A/B for historical-analytics correlation grouping.
//!
//! `correlation_rows_for_rows` runs per render of the historical-analytics screen,
//! grouping the filtered evidence rows by `reason_code`. The legacy grouping keyed a
//! `BTreeMap<String, _>` by `reason_code.clone()` and accumulated a
//! `BTreeSet<String>` of projects — two `String` clones per row. The shipped path
//! keys by borrowed `&str` (rows outlive the map), cloning `reason_code` only once
//! per output group and counting projects without cloning. Byte-identical output
//! (the sort's terminal `reason_code` tiebreak is unique per group).
//!
//! Parity is proven over several corpus shapes before timing; timing is a
//! within-process paired AB/BA ratio with an A/A null floor, so both arms always run
//! on the same worker (immune to the RCH_WORKER soft-pin contention issue).
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release \
//!     --features bench-internals --bench correlation_group_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_ops::screens::historical_analytics::{
    BenchEvidenceRows, bench_correlation_borrowed, bench_correlation_owned,
    bench_make_evidence_rows,
};

const LAG_PRESSURE: u8 = 37;
const DROP_PRESSURE: u8 = 11;
const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// (n_rows, n_reasons, n_projects) shapes: many events per group is the realistic
/// evidence-log shape (clones drop from ~2N to ~G there); the last is a stress case.
const SHAPES: &[(usize, usize, usize)] = &[(2_048, 16, 24), (8_192, 24, 40), (16_384, 32, 64)];

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
    Owned,
    Borrowed,
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

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let idx = ((sorted.len() - 1) * pct + 50) / 100;
    sorted[idx]
}

fn run_arm(rows: &BenchEvidenceRows, arm: Arm) -> usize {
    let out = match arm {
        Arm::Owned => bench_correlation_owned(rows, LAG_PRESSURE, DROP_PRESSURE),
        Arm::Borrowed => bench_correlation_borrowed(rows, LAG_PRESSURE, DROP_PRESSURE),
    };
    out.len()
}

fn time_arm(rows: &BenchEvidenceRows, arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(rows, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(rows: &BenchEvidenceRows, arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let ab_a = time_arm(rows, arm_a);
        let ab_b = time_arm(rows, arm_b);
        let ba_b = time_arm(rows, arm_b);
        let ba_a = time_arm(rows, arm_a);
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

fn median_ms(rows: &BenchEvidenceRows, arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(rows, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(rows, arm)).collect();
    samples.sort_unstable();
    percentile(&samples, 50).as_secs_f64() * 1_000.0
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
    for &(n_rows, n_reasons, n_projects) in SHAPES {
        let rows = bench_make_evidence_rows(n_rows, n_reasons, n_projects);

        // Parity: borrowed-key output must equal legacy owned-key output exactly.
        let owned = bench_correlation_owned(&rows, LAG_PRESSURE, DROP_PRESSURE);
        let borrowed = bench_correlation_borrowed(&rows, LAG_PRESSURE, DROP_PRESSURE);
        assert_eq!(
            owned, borrowed,
            "borrowed-key grouping must be byte-identical to owned-key for shape {n_rows}/{n_reasons}/{n_projects}"
        );
        eprintln!(
            "[parity] shape={n_rows}/{n_reasons}/{n_projects} groups={} output_identical=true",
            borrowed.len()
        );

        let owned_ms = median_ms(&rows, Arm::Owned);
        let borrowed_ms = median_ms(&rows, Arm::Borrowed);
        eprintln!(
            "[profile] shape={n_rows}/{n_reasons}/{n_projects} owned_median_ms={owned_ms:.6} borrowed_median_ms={borrowed_ms:.6}"
        );

        let null = paired_ratio(&rows, Arm::Owned, Arm::Owned);
        let lever = paired_ratio(&rows, Arm::Owned, Arm::Borrowed);
        eprintln!(
            "[paired] shape={n_rows}/{n_reasons}/{n_projects} comparison=null_owned_owned median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            null.median, null.p5, null.p95, null.round_pairs
        );
        eprintln!(
            "[paired] shape={n_rows}/{n_reasons}/{n_projects} comparison=borrowed_vs_owned median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            lever.median, lever.p5, lever.p95, lever.round_pairs
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        all_gates_pass &= gate_pass;
        eprintln!(
            "[gate] shape={n_rows}/{n_reasons}/{n_projects} verdict={} median_speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
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
