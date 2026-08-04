//! Differential parity + paired A/B for the timeline project-filter resolution.
//!
//! `ActionTimelineScreen::filtered_events`, when a project filter is active,
//! resolved each event's instance id to its project via `project_for_instance` —
//! a linear `.find` over ALL fleet instances, PER event. That made the filter
//! O(events * instances). The shipped path builds an id -> project `HashMap` once
//! (O(instances)) and resolves each event in O(1), i.e. O(events + instances)
//! total. Byte-identical: `or_insert` keeps the first occurrence, matching
//! `.find`, and the shared `event_project_matches` predicate is unchanged.
//!
//! Corpus: `n` events with `instance-{i:06}` ids over `n` instances sharing the
//! id scheme, filtered by an existing project — so event i (position i) forces
//! the linear arm to scan ~i instances (quadratic in n), while the mapped arm is
//! linear. Within-process paired AB/BA with an A/A null floor.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release \
//!     --features bench-internals --bench project_resolve_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_ops::screens::timeline::{
    bench_count_project_linear, bench_count_project_mapped, bench_make_instances,
    bench_make_lifecycle_events,
};
use frankensearch_ops::state::{InstanceInfo, LifecycleEvent};

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// Event / instance counts feeding the project-filter resolution.
const SHAPES: &[usize] = &[64, 512, 2_048, 4_096];

/// An existing project (`bench_make_instances` spans `project-0..project-7`).
const FILTER_PROJECT: &str = "project-3";

struct Corpus {
    events: Vec<LifecycleEvent>,
    instances: Vec<InstanceInfo>,
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

fn run_arm(corpus: &Corpus, arm: Arm) -> usize {
    match arm {
        Arm::Legacy => {
            bench_count_project_linear(&corpus.events, &corpus.instances, FILTER_PROJECT)
        }
        Arm::Fused => bench_count_project_mapped(&corpus.events, &corpus.instances, FILTER_PROJECT),
    }
}

fn time_arm(corpus: &Corpus, arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(corpus, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(corpus: &Corpus, arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let ab_a = time_arm(corpus, arm_a);
        let ab_b = time_arm(corpus, arm_b);
        let ba_b = time_arm(corpus, arm_b);
        let ba_a = time_arm(corpus, arm_a);
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

fn median_us(corpus: &Corpus, arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(corpus, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(corpus, arm)).collect();
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
        let corpus = Corpus {
            events: bench_make_lifecycle_events(n),
            instances: bench_make_instances(n),
        };

        let legacy = run_arm(&corpus, Arm::Legacy);
        let fused = run_arm(&corpus, Arm::Fused);
        assert_eq!(
            legacy, fused,
            "mapped resolution must be byte-identical to linear for n={n}"
        );
        eprintln!("[parity] n={n} matches={legacy} output_identical=true");

        let legacy_us = median_us(&corpus, Arm::Legacy);
        let fused_us = median_us(&corpus, Arm::Fused);
        eprintln!("[profile] n={n} legacy_median_us={legacy_us:.4} fused_median_us={fused_us:.4}");

        let null = paired_ratio(&corpus, Arm::Legacy, Arm::Legacy);
        let lever = paired_ratio(&corpus, Arm::Legacy, Arm::Fused);
        eprintln!(
            "[paired] n={n} comparison=null_legacy_legacy median={:.6} p5={:.6} p95={:.6} round_pairs={}",
            null.median, null.p5, null.p95, null.round_pairs
        );
        eprintln!(
            "[paired] n={n} comparison=fused_vs_legacy median={:.6} p5={:.6} p95={:.6} round_pairs={}",
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
        if all_gates_pass { "KEEP" } else { "HOLD" }
    );
}
