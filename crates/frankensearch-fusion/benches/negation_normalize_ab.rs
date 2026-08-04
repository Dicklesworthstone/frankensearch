//! Differential parity + paired A/B for the negative-exclusion text normalizer.
//!
//! `normalize_for_negation_match` runs on each candidate document's text when a
//! query carries exclusion terms (`-term`), normalizing it before substring/phrase
//! matching. The legacy path always ran the `unicode_normalization` NFC composing
//! iterator plus Unicode lowercasing over the whole text. The shipped path takes an
//! `is_ascii()` fast path — for ASCII input NFC is the identity and Unicode
//! lowercasing equals ASCII lowercasing, so it is byte-identical while skipping the
//! Unicode machinery entirely; non-ASCII input still uses the full Unicode path.
//!
//! Parity is proven over an ASCII document corpus plus a non-ASCII corpus before
//! timing; timing is a within-process paired AB/BA ratio with an A/A null floor, so
//! both arms always run on the same worker (immune to the RCH_WORKER soft-pin issue).
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-fusion --profile release \
//!     --features bench-internals --bench negation_normalize_ab
//! ```
#![allow(clippy::doc_markdown, clippy::cast_precision_loss)]

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use frankensearch_fusion::bench_support::paired_median_ratio;
use frankensearch_fusion::searcher::{
    bench_normalize_for_negation_match_fast, bench_normalize_for_negation_match_reference,
};

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

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
    Reference,
    Fast,
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

/// Deterministic ASCII "document" of roughly `words` mixed-case tokens — the
/// representative input for the negation normalizer's hot path.
fn ascii_document(seed: usize, words: usize) -> String {
    const VOCAB: &[&str] = &[
        "Vector",
        "search",
        "INDEX",
        "shard",
        "Latency",
        "p95",
        "Throughput",
        "ranking",
        "Embedding",
        "cosine",
        "TOKEN",
        "Filter",
        "exclude",
        "danger",
        "Zone",
        "Segment",
        "Merge",
        "COMPACT",
        "recall",
        "Precision",
        "budget",
        "Query",
        "class",
        "Refine",
    ];
    let mut out = String::with_capacity(words * 8);
    for i in 0..words {
        if i > 0 {
            out.push(' ');
        }
        let word = VOCAB[(seed.wrapping_mul(31).wrapping_add(i * 7)) % VOCAB.len()];
        out.push_str(word);
        if i % 5 == 4 {
            write!(&mut out, "{}", (seed + i) % 1000)
                .expect("writing a number to String cannot fail");
        }
    }
    out
}

fn ascii_corpus() -> Vec<String> {
    let mut corpus = Vec::new();
    // Short query-term-like strings.
    for seed in 0..256 {
        corpus.push(ascii_document(seed, 1 + seed % 3));
    }
    // Medium and long document bodies (the hot input).
    for seed in 0..384 {
        corpus.push(ascii_document(seed, 40 + seed % 120));
    }
    corpus
}

fn non_ascii_corpus() -> Vec<String> {
    vec![
        "Café RÉSUMÉ naïve".to_owned(),
        "MÜNCHEN Straße GRÜßEN".to_owned(),
        // Combining marks (NFC should compose these).
        "e\u{0301}\u{0300} A\u{030A}".to_owned(),
        "CAFE\u{0301} danger".to_owned(),
        "ПРИВЕТ мир Danger".to_owned(),
        "日本語 ТЕСТ exclude".to_owned(),
        "ﬁle ﬂag Ⅳ Ⅸ".to_owned(),
        "İstanbul Ⓐ ①②③".to_owned(),
        String::new(),
        "plain ascii tail".to_owned(),
    ]
}

fn prove_parity(ascii: &[String], non_ascii: &[String]) {
    let mut ascii_checked = 0usize;
    let mut non_ascii_checked = 0usize;
    for value in ascii {
        assert_eq!(
            bench_normalize_for_negation_match_fast(value),
            bench_normalize_for_negation_match_reference(value),
            "fast path must equal reference for ASCII input {value:?}"
        );
        assert!(
            value.is_ascii(),
            "ascii corpus entry must be ASCII: {value:?}"
        );
        ascii_checked += 1;
    }
    for value in non_ascii {
        assert_eq!(
            bench_normalize_for_negation_match_fast(value),
            bench_normalize_for_negation_match_reference(value),
            "fast path must equal reference for non-ASCII input {value:?}"
        );
        non_ascii_checked += 1;
    }
    eprintln!(
        "[parity] ascii_checked={ascii_checked} non_ascii_checked={non_ascii_checked} fast_equals_reference=true"
    );
}

fn run_arm(corpus: &[String], arm: Arm) -> usize {
    let mut total = 0usize;
    for value in corpus {
        let normalized = match arm {
            Arm::Reference => bench_normalize_for_negation_match_reference(value),
            Arm::Fast => bench_normalize_for_negation_match_fast(value),
        };
        total = total.wrapping_add(normalized.len());
    }
    total
}

fn time_arm(corpus: &[String], arm: Arm) -> Duration {
    let started = Instant::now();
    let acc = run_arm(corpus, arm);
    let elapsed = started.elapsed();
    black_box(acc);
    elapsed
}

fn paired_ratio(corpus: &[String], arm_a: Arm, arm_b: Arm) -> RatioDistribution {
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

fn median_us_per_call(corpus: &[String], arm: Arm) -> f64 {
    for _ in 0..3 {
        black_box(time_arm(corpus, arm));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS).map(|_| time_arm(corpus, arm)).collect();
    samples.sort_unstable();
    percentile(&samples, 50).as_secs_f64() * 1e6 / corpus.len() as f64
}

fn main() {
    let ascii = ascii_corpus();
    let non_ascii = non_ascii_corpus();
    let total_bytes: usize = ascii.iter().map(String::len).sum();
    eprintln!(
        "[profile-config] ascii_corpus={} non_ascii_corpus={} ascii_bytes={total_bytes} profile_rounds={PROFILE_ROUNDS} paired_round_pairs={PAIRED_ROUND_PAIRS}",
        ascii.len(),
        non_ascii.len()
    );
    eprintln!(
        "[profile-config] binary_path={}",
        std::env::current_exe()
            .expect("resolve measured binary")
            .display()
    );

    prove_parity(&ascii, &non_ascii);

    let ref_us = median_us_per_call(&ascii, Arm::Reference);
    let fast_us = median_us_per_call(&ascii, Arm::Fast);
    eprintln!(
        "[profile] stage=reference_unicode_nfc per_call_median_us={ref_us:.4} corpus={}",
        ascii.len()
    );
    eprintln!(
        "[profile] stage=shipped_ascii_fast per_call_median_us={fast_us:.4} corpus={}",
        ascii.len()
    );

    let null = paired_ratio(&ascii, Arm::Reference, Arm::Reference);
    let lever = paired_ratio(&ascii, Arm::Reference, Arm::Fast);
    eprintln!(
        "[paired] comparison=null_reference_reference median={:.6} p5={:.6} p95={:.6} round_pairs={}",
        null.median, null.p5, null.p95, null.round_pairs
    );
    eprintln!(
        "[paired] comparison=fast_vs_reference median={:.6} p5={:.6} p95={:.6} round_pairs={}",
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

    // ── DECIDABILITY: shared alternating-round paired sampler + A/A null control ──
    //
    // Same measurement as the bespoke [paired]/[gate] block above, expressed through
    // the shared `paired_median_ratio` sampler (bd-zgq6) so this bench reports the
    // common [null]/[lever] rows. The closures mirror exactly what `time_arm` times:
    // one `run_arm` pass over the ASCII corpus per iteration.
    let reference = || {
        black_box(run_arm(black_box(&ascii), Arm::Reference));
    };
    let fast = || {
        black_box(run_arm(black_box(&ascii), Arm::Fast));
    };
    let null = paired_median_ratio(41, 8, reference, reference);
    let lever = paired_median_ratio(41, 8, reference, fast);
    eprintln!(
        "[null]  negation_normalize ascii: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] negation_normalize ascii: fast median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            "DECIDABLE"
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );
}
