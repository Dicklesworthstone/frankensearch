//! Differential parity + paired A/B for reranker token-id materialization.
//!
//! `NativeEmbedder::tokenize` converts the tokenizer's `&[u32]` ids to `Vec<i64>`
//! and truncates to `max_length`. The legacy path collected ALL ids then truncated;
//! the shipped path (`ids_to_truncated_i64`) truncates with `take(max_length)`
//! before the conversion+collect, so for a document that tokenizes to far more than
//! `max_length` tokens (common when reranking full document bodies) it materializes
//! only `max_length` ids instead of all of them. Byte-identical.
//!
//! Compiles WITHOUT the heavy `native` feature (pure function). Within-process
//! paired AB/BA with an A/A null floor.
//!
//! Run: `rch exec -- cargo bench -p frankensearch-rerank --profile release
//!   --bench token_id_truncate_ab`
#![allow(
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_rerank::ids_to_truncated_i64;

const MAX_LENGTH: usize = 512;
const PROFILE_ROUNDS: usize = 51;
const PAIRED_ROUND_PAIRS: usize = 51;

/// Representative tokenized document lengths (tokens). Reranking full document
/// bodies commonly exceeds the 512-token cap; short passages fit under it.
const TOKEN_LENGTHS: &[usize] = &[128, 512, 2_048, 8_192];

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
    Fast,
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

/// Legacy: collect all ids to `Vec<i64>`, then truncate. Mirrors the pre-change code.
fn ids_to_truncated_i64_legacy(ids: &[u32], max_length: usize) -> Vec<i64> {
    let mut out: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
    if out.len() > max_length {
        out.truncate(max_length);
    }
    out
}

fn make_ids(len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) % 30_000)
        .collect()
}

fn run_arm(ids: &[u32], arm: Arm) -> usize {
    let out = match arm {
        Arm::Legacy => ids_to_truncated_i64_legacy(ids, MAX_LENGTH),
        Arm::Fast => ids_to_truncated_i64(ids, MAX_LENGTH),
    };
    out.len()
}

fn time_arm(ids: &[u32], arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(ids, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(ids: &[u32], a: Arm, b: Arm) -> RatioDistribution {
    let run_pair = |record: bool| {
        let aa = time_arm(ids, a);
        let ab = time_arm(ids, b);
        let bb = time_arm(ids, b);
        let ba = time_arm(ids, a);
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

fn median_ns(ids: &[u32], arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(ids, arm));
    }
    let mut s: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(ids, arm)).collect();
    s.sort_unstable();
    percentile(&s, 50).as_secs_f64() * 1e9
}

fn main() {
    eprintln!(
        "[profile-config] max_length={MAX_LENGTH} token_lengths={TOKEN_LENGTHS:?} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}"
    );
    let mut any_clears = false;
    for &len in TOKEN_LENGTHS {
        let ids = make_ids(len);
        assert_eq!(
            ids_to_truncated_i64(&ids, MAX_LENGTH),
            ids_to_truncated_i64_legacy(&ids, MAX_LENGTH),
            "fast must equal legacy for len={len}"
        );
        let kept = len.min(MAX_LENGTH);
        eprintln!("[parity] tokens={len} kept={kept} output_identical=true");

        let legacy_ns = median_ns(&ids, Arm::Legacy);
        let fast_ns = median_ns(&ids, Arm::Fast);
        eprintln!(
            "[profile] tokens={len} legacy_median_ns={legacy_ns:.2} fast_median_ns={fast_ns:.2}"
        );

        let null = paired_ratio(&ids, Arm::Legacy, Arm::Legacy);
        let lever = paired_ratio(&ids, Arm::Legacy, Arm::Fast);
        eprintln!(
            "[paired] tokens={len} comparison=null_legacy_legacy median={:.6} p5={:.6} p95={:.6}",
            null.median, null.p5, null.p95
        );
        eprintln!(
            "[paired] tokens={len} comparison=fast_vs_legacy median={:.6} p5={:.6} p95={:.6}",
            lever.median, lever.p5, lever.p95
        );
        let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
        any_clears |= gate_pass;
        eprintln!(
            "[gate] tokens={len} verdict={} median_speedup={:.6}x gate_pass={gate_pass}",
            lever.verdict_against(null),
            1.0 / lever.median
        );
    }
    eprintln!(
        "[gate-summary] decision={} any_shape_clears_null_floor={any_clears}",
        if any_clears { "KEEP" } else { "HOLD" }
    );
}
