//! Differential parity + paired A/B for the RFC3339 → epoch-millis ingest parser.
//!
//! Telemetry envelopes are parsed once per ingested event via
//! `parse_rfc3339_timestamp_ms`. The shipped path now resolves the canonical UTC
//! form (`YYYY-MM-DDTHH:MM:SS[.fraction]Z`) with a strict hand-rolled fast path and
//! defers anything else to the general-purpose `time` crate reference parser.
//!
//! This bench first proves, over a canonical corpus plus an adversarial corpus,
//! that the shipped parser returns exactly what the reference parser returns
//! (`Ok`/`Err` and value), then times the reference parser against the shipped
//! fast path on the canonical corpus (the production distribution). All timing is
//! within a single process so both arms run on the same worker.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-ops --profile release \
//!     --features bench-internals --bench rfc3339_parse_ab
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_ops::storage::{
    bench_fast_parse_rfc3339_utc_ms, bench_parse_rfc3339_reference_ms,
    bench_parse_rfc3339_timestamp_ms,
};

const CANONICAL_CORPUS: usize = 4_096;
const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

#[derive(Clone, Copy)]
#[allow(clippy::struct_field_names)]
struct Distribution {
    median_ns: f64,
    p5_ns: f64,
    p95_ns: f64,
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
enum ParseArm {
    Reference,
    ShippedFast,
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    let index = ((sorted.len() - 1) * pct + 50) / 100;
    sorted[index]
}

fn distribution(mut samples: Vec<Duration>, iters_per_sample: usize) -> Distribution {
    samples.sort_unstable();
    let per = |d: Duration| d.as_secs_f64() * 1e9 / iters_per_sample as f64;
    Distribution {
        median_ns: per(percentile(&samples, 50)),
        p5_ns: per(percentile(&samples, 5)),
        p95_ns: per(percentile(&samples, 95)),
    }
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

// --- Corpus construction (deterministic, no RNG) ---

const fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Build a valid canonical UTC timestamp from an index, cycling field values and
/// fractional-digit counts (0..=9) so the fast path is exercised across shapes.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn canonical_timestamp(index: usize) -> String {
    let year = 1970 + (index * 7) % 8030; // 1970..=9999
    let month = 1 + (index % 12) as u32;
    let dim = days_in_month(year as i64, month);
    let day = 1 + (index as u32 % dim);
    let hour = (index * 3) % 24;
    let minute = (index * 13) % 60;
    let second = (index * 29) % 60;
    let frac_digits = index % 10;
    let mut ts = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}");
    if frac_digits > 0 {
        ts.push('.');
        // A varied but deterministic fractional value.
        let seed = (index * 2_654_435_761) % 1_000_000_000;
        let full = format!("{seed:09}");
        ts.push_str(&full[..frac_digits]);
    }
    ts.push('Z');
    ts
}

fn canonical_corpus() -> Vec<String> {
    (0..CANONICAL_CORPUS).map(canonical_timestamp).collect()
}

/// Non-canonical / invalid forms the fast path must decline (deferring to the
/// reference), plus a few valid-but-non-fast forms (offsets, lowercase, `t`).
fn adversarial_corpus() -> Vec<String> {
    vec![
        // Non-UTC offsets (valid RFC3339, must defer).
        "2026-07-11T20:37:46+05:30".to_owned(),
        "2026-07-11T20:37:46.5-08:00".to_owned(),
        "2026-07-11T20:37:46.123456789+00:00".to_owned(),
        // Lowercase separators (must defer).
        "2026-07-11t20:37:46Z".to_owned(),
        "2026-07-11T20:37:46z".to_owned(),
        // Invalid calendar / field values (reference errors; fast declines).
        "2026-02-30T00:00:00Z".to_owned(),
        "2025-02-29T00:00:00Z".to_owned(), // 2025 not a leap year
        "2026-13-01T00:00:00Z".to_owned(),
        "2026-00-10T00:00:00Z".to_owned(),
        "2026-07-00T00:00:00Z".to_owned(),
        "2026-07-11T24:00:00Z".to_owned(),
        "2026-07-11T20:60:00Z".to_owned(),
        "2026-07-11T20:37:60Z".to_owned(), // leap second, unsupported by `time`
        // Fraction edge cases.
        "2026-07-11T20:37:46.Z".to_owned(), // empty fraction
        "2026-07-11T20:37:46.1234567890Z".to_owned(), // 10 fractional digits
        // Structural junk.
        "2026-07-11 20:37:46Z".to_owned(), // space separator
        "2026/07/11T20:37:46Z".to_owned(),
        "not-a-timestamp".to_owned(),
        "2026-07-11T20:37:46".to_owned(),   // no zone
        "2026-07-11T20:37:46Z ".to_owned(), // trailing space
        String::new(),
        // Below the fast-path year floor (valid RFC3339, must defer).
        "1969-12-31T23:59:59Z".to_owned(),
        "0001-01-01T00:00:00Z".to_owned(),
        // A few plainly valid canonical forms to confirm the fast path also fires here.
        "2000-02-29T12:00:00Z".to_owned(), // leap day
        "1970-01-01T00:00:00Z".to_owned(),
        "9999-12-31T23:59:59.999999999Z".to_owned(),
    ]
}

fn prove_parity(canonical: &[String], adversarial: &[String]) {
    let mut fast_hits = 0usize;
    let mut fast_declines = 0usize;
    let mut checked = 0usize;
    for ts in canonical.iter().chain(adversarial.iter()) {
        let reference = bench_parse_rfc3339_reference_ms(ts);
        let shipped = bench_parse_rfc3339_timestamp_ms(ts);
        assert_eq!(
            shipped, reference,
            "shipped parser must equal reference for {ts:?}"
        );
        match bench_fast_parse_rfc3339_utc_ms(ts) {
            Some(value) => {
                assert_eq!(
                    Some(value),
                    reference,
                    "fast-path Some must equal reference Ok for {ts:?}"
                );
                fast_hits += 1;
            }
            None => fast_declines += 1,
        }
        checked += 1;
    }
    // The whole canonical corpus must take the fast path.
    assert!(
        fast_hits >= canonical.len(),
        "every canonical timestamp must hit the fast path (hits={fast_hits}, canonical={})",
        canonical.len()
    );
    eprintln!(
        "[parity] checked={checked} fast_hits={fast_hits} fast_declines={fast_declines} shipped_equals_reference=true fast_some_equals_reference=true"
    );
}

fn run_arm(corpus: &[String], arm: ParseArm) -> i64 {
    let mut acc = 0i64;
    match arm {
        ParseArm::Reference => {
            for ts in corpus {
                if let Some(ms) = bench_parse_rfc3339_reference_ms(ts) {
                    acc = acc.wrapping_add(ms);
                }
            }
        }
        ParseArm::ShippedFast => {
            for ts in corpus {
                if let Some(ms) = bench_parse_rfc3339_timestamp_ms(ts) {
                    acc = acc.wrapping_add(ms);
                }
            }
        }
    }
    acc
}

fn time_arm(corpus: &[String], arm: ParseArm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(corpus, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(corpus: &[String], arm_a: ParseArm, arm_b: ParseArm) -> RatioDistribution {
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

fn profile(corpus: &[String], arm: ParseArm, label: &str) {
    for _ in 0..3 {
        black_box(time_arm(corpus, arm));
    }
    let samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(corpus, arm)).collect();
    let dist = distribution(samples, corpus.len());
    eprintln!(
        "[profile] stage={label} per_call_median_ns={:.3} p5_ns={:.3} p95_ns={:.3} corpus={}",
        dist.median_ns,
        dist.p5_ns,
        dist.p95_ns,
        corpus.len()
    );
}

fn main() {
    let canonical = canonical_corpus();
    let adversarial = adversarial_corpus();
    eprintln!(
        "[profile-config] canonical_corpus={} adversarial_corpus={} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}",
        canonical.len(),
        adversarial.len()
    );
    eprintln!(
        "[profile-config] binary_path={}",
        std::env::current_exe()
            .expect("resolve measured binary")
            .display()
    );

    prove_parity(&canonical, &adversarial);

    profile(&canonical, ParseArm::Reference, "reference_time_crate");
    profile(&canonical, ParseArm::ShippedFast, "shipped_fast_path");

    let null = paired_ratio(&canonical, ParseArm::Reference, ParseArm::Reference);
    let lever = paired_ratio(&canonical, ParseArm::Reference, ParseArm::ShippedFast);
    eprintln!(
        "[paired] comparison=null_reference_reference median={:.6} p5={:.6} p95={:.6} round_pairs={}",
        null.median, null.p5, null.p95, null.round_pairs
    );
    eprintln!(
        "[paired] comparison=shipped_fast_vs_reference median={:.6} p5={:.6} p95={:.6} round_pairs={}",
        lever.median, lever.p5, lever.p95, lever.round_pairs
    );
    let gate_pass = null.null_is_unbiased() && lever.median < null.p5;
    eprintln!(
        "[gate] verdict={} median_speedup={:.6}x null_is_unbiased={} candidate_median_below_null_p5={} gate_pass={gate_pass}",
        lever.verdict_against(null),
        1.0 / lever.median,
        null.null_is_unbiased(),
        lever.median < null.p5
    );
    eprintln!(
        "[gate-summary] decision={}",
        if gate_pass { "KEEP" } else { "HOLD" }
    );
}
