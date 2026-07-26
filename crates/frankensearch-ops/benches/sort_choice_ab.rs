//! Retroactive measurement of the ops `sort_by` -> `sort_unstable_by` sweep.
//!
//! The per-instance / per-group ops sorts (fleet visible_instances, index_resources
//! monitor rows, live_stream row_data, alerts_slo project_slo_rows, fleet cards,
//! historical correlation, project_detail cards — 15dc56b2 / 64934f9f) were switched
//! from stable `sort_by` to `sort_unstable_by`. Each is a STRICT total order (a
//! unique final tiebreak: instance_id / project / reason_code), so the two are
//! byte-identical. The wins shipped on that argument + tests, never measured — and
//! the memory's `sort_unstable` lesson is n-dependent (loses on small tie-heavy).
//! For a strict total order there are no ties, so pdqsort should be >= Timsort at
//! every n; this quantifies it (or honestly corrects it to below-noise).
//!
//! To isolate the SORT from any row clone, both arms sort a `Vec<usize>` of indices
//! (cheap Copy clone) into a fixed `Vec<Row>`, using a multi-key comparator with a
//! unique `id` tiebreak that mirrors live_stream's `row_data`. Within-process paired
//! AB/BA against an A/A null floor.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release --bench sort_choice_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::{Duration, Instant};

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// Row counts (bounded by the fleet's `max_retained_events` = 4096 in production).
const SHAPES: &[usize] = &[64, 512, 2_048, 4_096];

struct Row {
    rank: u8,
    searches: u64,
    p95: u64,
    memory: u64,
    project: String,
    id: String,
}

/// Deterministic xorshift so the rows are in arbitrary (not near-sorted) order.
fn make_rows(n: usize) -> Vec<Row> {
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    (0..n)
        .map(|i| Row {
            rank: (next() % 3) as u8,
            searches: next() % 5_000,
            p95: next() % 20_000,
            memory: next() % (2 * 1024 * 1024 * 1024),
            project: format!("project-{}", next() % 8),
            id: format!("instance-{i:06}"),
        })
        .collect()
}

/// Mirrors live_stream `row_data`'s comparator: rank desc, searches desc, p95 desc,
/// memory desc, project asc, id asc. `id` is unique => strict total order.
fn cmp(a: &Row, b: &Row) -> Ordering {
    b.rank
        .cmp(&a.rank)
        .then_with(|| b.searches.cmp(&a.searches))
        .then_with(|| b.p95.cmp(&a.p95))
        .then_with(|| b.memory.cmp(&a.memory))
        .then_with(|| a.project.cmp(&b.project))
        .then_with(|| a.id.cmp(&b.id))
}

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
    round_pairs: usize,
}

impl RatioDistribution {
    fn null_contains_one(self) -> bool {
        self.p5 <= 1.0 && 1.0 <= self.p95
    }

    fn verdict_against(self, null: Self) -> &'static str {
        if !null.null_contains_one() {
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
    Stable,
    Unstable,
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

/// One timed sort of a fresh index permutation into `rows`. The `Vec<usize>` clone
/// is a cheap memcpy; the comparator (which reads the String-bearing rows) dominates.
fn time_arm(rows: &[Row], base: &[usize], arm: Arm) -> Duration {
    let mut idx = base.to_vec();
    let started = Instant::now();
    match arm {
        Arm::Stable => idx.sort_by(|&a, &b| cmp(&rows[a], &rows[b])),
        Arm::Unstable => idx.sort_unstable_by(|&a, &b| cmp(&rows[a], &rows[b])),
    }
    let elapsed = started.elapsed();
    black_box(idx.first().copied());
    elapsed
}

fn paired_ratio(rows: &[Row], base: &[usize], arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let ab_a = time_arm(rows, base, arm_a);
        let ab_b = time_arm(rows, base, arm_b);
        let ba_b = time_arm(rows, base, arm_b);
        let ba_a = time_arm(rows, base, arm_a);
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

fn median_us(rows: &[Row], base: &[usize], arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(rows, base, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS)
        .map(|_| time_arm(rows, base, arm))
        .collect();
    samples.sort_unstable();
    percentile(&samples, 50).as_secs_f64() * 1e6
}

fn main() {
    eprintln!(
        "[profile-config] shapes={} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}",
        SHAPES.len()
    );

    let mut all_gates_pass = true;
    for &n in SHAPES {
        let rows = make_rows(n);
        let base: Vec<usize> = (0..n).collect();

        // Parity: the two sorts of the strict-total-order comparator must agree.
        let mut a = base.clone();
        a.sort_by(|&x, &y| cmp(&rows[x], &rows[y]));
        let mut b = base.clone();
        b.sort_unstable_by(|&x, &y| cmp(&rows[x], &rows[y]));
        assert_eq!(
            a, b,
            "stable and unstable must agree on a strict total order (n={n})"
        );
        eprintln!("[parity] n={n} order_identical=true");

        let stable_us = median_us(&rows, &base, Arm::Stable);
        let unstable_us = median_us(&rows, &base, Arm::Unstable);
        eprintln!(
            "[profile] n={n} stable_median_us={stable_us:.4} unstable_median_us={unstable_us:.4}"
        );

        let null = paired_ratio(&rows, &base, Arm::Stable, Arm::Stable);
        let lever = paired_ratio(&rows, &base, Arm::Stable, Arm::Unstable);
        eprintln!(
            "[paired] n={n} comparison=null_stable_stable median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            null.median, null.p5, null.p95, null.round_pairs
        );
        eprintln!(
            "[paired] n={n} comparison=unstable_vs_stable median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            lever.median, lever.p5, lever.p95, lever.round_pairs
        );
        let gate_pass = null.null_contains_one() && lever.median < null.p5;
        all_gates_pass &= gate_pass;
        eprintln!(
            "[gate] n={n} verdict={} median_speedup={:.6}x null_contains_one={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median,
            null.null_contains_one(),
            lever.median < null.p5
        );
    }
    eprintln!(
        "[gate-summary] decision={} all_shapes_clear_null_floor={all_gates_pass}",
        if all_gates_pass {
            "unstable-faster"
        } else {
            "MIXED-or-floor"
        }
    );
}
