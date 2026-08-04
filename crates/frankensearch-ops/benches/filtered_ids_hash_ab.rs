//! Retroactive measurement of the ops `filtered_ids` membership-set hasher swap.
//!
//! `index_resources`' summary-line builder builds a membership set of the visible
//! rows' instance ids, then probes it once per `fleet.search_metrics` entry to sum
//! searches for the visible instances. That set was the default SipHash `HashSet`;
//! it was swapped to `ahash::AHashSet` (and `historical_analytics`' distinct-project
//! dedup set with it), because aHash beats SipHash on the short (~14-char) id keys
//! for both the `|rows|` inserts and the `|search_metrics|` probes — the same key
//! distribution the resolution-map swap measured 1.70-1.78x on (instance_map_hash_ab).
//!
//! This A/B builds the set from `n` instance ids and probes it with `n` keys under
//! each hasher (`std::collections::HashSet` vs `ahash::AHashSet`), the all-visible
//! case where build and probe are both `n`. Both hashers see identical membership,
//! so the accumulated hit count is asserted equal (parity). Within-process paired
//! AB/BA against an A/A null floor; both arms run in one process on one worker,
//! immune to the RCH_WORKER soft-pin issue.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release \
//!     --bench filtered_ids_hash_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// Instance / search-metrics counts (bounded by realistic fleet sizes).
const SHAPES: &[usize] = &[64, 512, 2_048, 4_096];

/// Deterministic ~14-char instance ids, mirroring the fleet's `instance-NNNNNN`
/// keys. A xorshift permutes the numeric suffix so the ids are not monotone.
fn make_ids(n: usize) -> Vec<String> {
    let mut r = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        r
    };
    (0..n)
        .map(|_| format!("instance-{:06}", next() % 1_000_000))
        .collect()
}

#[derive(Clone, Copy)]
enum Arm {
    Sip,
    AHash,
}

/// Build a SipHash membership set of `ids` and probe it with every id, summing hits
/// (mirrors `filtered_ids.contains(...)` in the search-metrics fold).
fn kernel_sip(ids: &[String]) -> u64 {
    let set: std::collections::HashSet<&str> = ids.iter().map(String::as_str).collect();
    let mut acc = 0_u64;
    for id in ids {
        if set.contains(id.as_str()) {
            acc += 1;
        }
    }
    acc
}

/// aHash twin of `kernel_sip`.
fn kernel_ahash(ids: &[String]) -> u64 {
    let set: ahash::AHashSet<&str> = ids.iter().map(String::as_str).collect();
    let mut acc = 0_u64;
    for id in ids {
        if set.contains(id.as_str()) {
            acc += 1;
        }
    }
    acc
}

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

fn time_arm(ids: &[String], arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = match arm {
        Arm::Sip => kernel_sip(ids),
        Arm::AHash => kernel_ahash(ids),
    };
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(ids: &[String], arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let ab_a = time_arm(ids, arm_a);
        let ab_b = time_arm(ids, arm_b);
        let ba_b = time_arm(ids, arm_b);
        let ba_a = time_arm(ids, arm_a);
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

fn median_us(ids: &[String], arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(ids, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(ids, arm)).collect();
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
        let ids = make_ids(n);

        // Parity: both hashers see identical membership => identical hit count.
        let sip = kernel_sip(&ids);
        let ah = kernel_ahash(&ids);
        assert_eq!(sip, ah, "sip and ahash must agree on membership (n={n})");
        eprintln!("[parity] n={n} hits={sip} count_identical=true");

        let sip_us = median_us(&ids, Arm::Sip);
        let ahash_us = median_us(&ids, Arm::AHash);
        eprintln!("[profile] n={n} sip_median_us={sip_us:.4} ahash_median_us={ahash_us:.4}");

        let null = paired_ratio(&ids, Arm::Sip, Arm::Sip);
        let lever = paired_ratio(&ids, Arm::Sip, Arm::AHash);
        eprintln!(
            "[paired] n={n} comparison=null_sip_sip median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            null.median, null.p5, null.p95, null.round_pairs
        );
        eprintln!(
            "[paired] n={n} comparison=ahash_vs_sip median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            lever.median, lever.p5, lever.p95, lever.round_pairs
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        all_gates_pass &= gate_pass;
        eprintln!(
            "[gate] n={n} verdict={} median_speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median,
            null.null_is_unbiased(),
            lever.median < null.p5
        );
    }
    eprintln!(
        "[gate-summary] decision={} all_shapes_clear_null_floor={all_gates_pass}",
        if all_gates_pass {
            "ahash-faster"
        } else {
            "MIXED-or-floor"
        }
    );
}
