//! Bench-only measurement harness. Not a shipping path (`feature = "bench-internals"`).
//!
//! Lives in `frankensearch-core` so **every** crate's benches can share one decidability harness —
//! including `frankensearch-index`, which cannot depend on `frankensearch-fusion` (that would be a
//! dependency cycle). This is what makes an int8-ADC-scan A/B (`bd-b5wl`) decidable on the same
//! footing as a fusion A/B. Std-only (`black_box` + `Instant`); zero new dependencies.
//!
//! # Why criterion alone cannot decide a small lever
//!
//! Registering ORIG and CAND as two criterion benchmarks — even with each one internally
//! interleaving a timed and an untimed half — does **not** cancel worker drift, because criterion
//! runs the two benchmarks *sequentially*, often minutes apart. The internal interleaving only
//! equalizes cache/branch state *within* an arm. Measured consequence (`neighbor_smooth`, worker
//! `hz1`, 120 samples): an A/A null control — the identical function registered as both arms —
//! reported a median ratio of **1.1265×** at pool 50 and **0.9268×** at pool 100, a range that does
//! not even contain 1.000. Any lever whose effect is smaller than that is undecidable on that
//! harness, and a WIN or REJECT resting on one is meaningless.
//!
//! # What this does instead
//!
//! [`paired_median_ratio`] runs both arms inside **one** measured routine, in **alternating rounds**:
//! round `r` times `(a, b)` when `r` is even and `(b, a)` when odd, so first-mover and cache-warm
//! bias cancel across rounds. It forms the ratio **per round**, so drift is shared by the two arms
//! within a few microseconds of each other rather than across minutes, then reports the **median**
//! ratio with a deterministic bootstrap 95% confidence interval. The raw p5/p95 spread remains
//! available as provenance; it is never the admission gate.
//!
//! Gate on the **median against the null's observed spread**, not on `cv_pct` — `cv < 5%` is
//! unattainable on this fleet. Admission of the null itself is a question of **accuracy** (is its
//! median at 1.0?) and never of precision — see [`PairedRatio::is_admissible_null`] for why a
//! "the CI must contain 1.0" clause gets that backwards. The floor is **per-function**: calibrate
//! it for the function you are
//! actually measuring by running `paired_median_ratio(rounds, inner, base, base)` (an A/A null)
//! before trusting `paired_median_ratio(rounds, inner, base, cand)`.

use std::hint::black_box;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// Deterministic bootstrap resamples used for the median confidence interval.
pub const BOOTSTRAP_RESAMPLES: usize = 2_000;
/// Required distance between a claim and its same-invocation A/A floor.
pub const NULL_FLOOR_MARGIN: f64 = 2.0;
/// Largest `|median - 1|` an A/A control may show and still be admissible.
///
/// An A/A null measures the same work on both arms, so its median belongs at
/// 1.0; a median further than this from 1.0 means the sampler itself is biased
/// (asymmetric round ordering, thermal drift, a mis-wired arm) and nothing
/// measured against it can be trusted. This bounds the null's *accuracy*. Its
/// *precision* is bounded separately and in the right direction by
/// [`PairedRatio::null_half_width`], which [`PairedRatio::decidable_against`]
/// requires a claim to clear by [`NULL_FLOOR_MARGIN`].
pub const NULL_MEDIAN_TOLERANCE: f64 = 0.02;

/// Identity of the benchmark executable that is actually running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchExecutableIdentity {
    /// Lowercase SHA-256 of the executing ELF/Mach-O/PE image.
    pub sha256: String,
    /// Executable byte length.
    pub bytes: usize,
    /// Path returned by [`std::env::current_exe`].
    pub path: PathBuf,
}

/// Hash the executing benchmark binary and print its identity as line one.
///
/// Call this before constructing Criterion or emitting any other benchmark
/// output. Hashing happens outside every measured region.
///
/// # Errors
///
/// Returns an I/O error if the current executable path cannot be resolved or
/// the executing binary cannot be read.
pub fn print_bench_elf_sha256() -> io::Result<BenchExecutableIdentity> {
    let path = std::env::current_exe()?;
    let executable = std::fs::read(&path)?;
    let identity = BenchExecutableIdentity {
        sha256: lower_hex(&Sha256::digest(&executable)),
        bytes: executable.len(),
        path,
    };
    println!(
        "bench_elf_sha256={} ({} bytes) {}",
        identity.sha256,
        identity.bytes,
        identity.path.display()
    );
    Ok(identity)
}

/// Median ratio `b/a` with a bootstrap median CI and raw spread.
#[derive(Debug, Clone, Copy)]
pub struct PairedRatio {
    /// Median of the per-round `b/a` ratios. For an A/A null this should sit at ~1.000.
    pub median: f64,
    /// Lower endpoint of the deterministic bootstrap 95% CI on [`Self::median`].
    pub median_ci95_low: f64,
    /// Upper endpoint of the deterministic bootstrap 95% CI on [`Self::median`].
    pub median_ci95_high: f64,
    /// 5th percentile of the per-round ratios.
    pub p5: f64,
    /// 95th percentile of the per-round ratios.
    pub p95: f64,
    /// Rounds actually measured.
    pub rounds: usize,
}

impl PairedRatio {
    /// Whether this A/A control is admissible for a claim.
    ///
    /// Gates on the null's **accuracy** — how far its median sits from 1.0 —
    /// and never on its precision. The CI is retained as telemetry and feeds
    /// [`Self::null_half_width`]; it is not itself an admission test.
    ///
    /// # Why not "the CI must contain 1.0"
    ///
    /// That clause, which this replaced (`bd-pjh09`), couples admission to the
    /// null's precision in the wrong direction: a *tighter* null has a narrower
    /// CI, so it is *more* likely to exclude 1.0 and veto its own row — however
    /// small the residual bias and however large the effect being measured. It
    /// punishes exactly the measurement quality (more rounds, pinned cores,
    /// `performance` governor) the harness is trying to buy, and it makes
    /// admission a property of the host's noise rather than of the code.
    /// Measured on one ELF across pinned cores, a reproducible planted effect
    /// kept its ratio while that clause flipped its verdict; see the
    /// `null_gate_admissibility_moves_while_the_effect_reproduces` probe.
    ///
    /// A null that is genuinely broken shows it in the median, not in the CI
    /// width, which is what [`NULL_MEDIAN_TOLERANCE`] bounds. This is strictly
    /// a *different* rule, not a looser one: it newly rejects a wide null whose
    /// median has drifted more than 2% off 1.0, which the straddle clause
    /// happily admitted.
    #[must_use]
    pub fn is_admissible_null(&self) -> bool {
        self.rounds >= 10
            && self.median.is_finite()
            && self.median_ci95_low.is_finite()
            && self.median_ci95_high.is_finite()
            && self.median_ci95_low <= self.median_ci95_high
            && (self.median - 1.0).abs() <= NULL_MEDIAN_TOLERANCE
    }

    /// A/A median-CI half-width around one.
    #[must_use]
    pub fn null_half_width(&self) -> f64 {
        (self.median_ci95_low - 1.0)
            .abs()
            .max((self.median_ci95_high - 1.0).abs())
    }

    /// Whether `self` clears the admissible A/A median-CI floor by 2×.
    ///
    /// Raw p5/p95 and CV values are provenance only; they never decide this
    /// result.
    #[must_use]
    pub fn decidable_against(&self, null: &Self) -> bool {
        null.is_admissible_null()
            && (self.median - 1.0).abs() >= NULL_FLOOR_MARGIN * null.null_half_width()
    }
}

/// Time `inner` back-to-back calls of `f`, returning the elapsed duration for the whole batch.
///
/// Batching amortizes the `Instant::now()` pair; the caller divides by `inner` for a per-call cost.
fn time_batch<F: FnMut()>(inner: u32, f: &mut F) -> Duration {
    let t = Instant::now();
    for _ in 0..inner {
        f();
    }
    t.elapsed()
}

/// Run `a` and `b` in alternating rounds within one routine and return the median `b/a` ratio.
///
/// Each round times `inner` calls of each arm. Even rounds run `a` then `b`; odd rounds run `b` then
/// `a`. Callers must `black_box` their inputs and results inside the closures — this function
/// `black_box`es the closures themselves but cannot see through them.
///
/// Panics if `rounds == 0` or `inner == 0`.
#[must_use]
pub fn paired_median_ratio<A: FnMut(), B: FnMut()>(
    rounds: usize,
    inner: u32,
    mut a: A,
    mut b: B,
) -> PairedRatio {
    assert!(rounds > 0 && inner > 0, "rounds and inner must be non-zero");

    // Warm both arms so the first measured round is not a cold-code outlier.
    for _ in 0..2 {
        black_box(time_batch(inner, &mut a));
        black_box(time_batch(inner, &mut b));
    }

    let mut ratios: Vec<f64> = Vec::with_capacity(rounds);
    for r in 0..rounds {
        let (ta, tb) = if r % 2 == 0 {
            let ta = time_batch(inner, &mut a);
            let tb = time_batch(inner, &mut b);
            (ta, tb)
        } else {
            let tb = time_batch(inner, &mut b);
            let ta = time_batch(inner, &mut a);
            (ta, tb)
        };
        let ta = ta.as_secs_f64();
        if ta > 0.0 {
            ratios.push(tb.as_secs_f64() / ta);
        }
    }

    assert!(
        !ratios.is_empty(),
        "no round produced a positive base timing"
    );
    ratios.sort_unstable_by(f64::total_cmp);
    let n = ratios.len();
    let (median_ci95_low, median_ci95_high) = bootstrap_median_ci95(&ratios);
    // `q ∈ [0,1]` and `n ≥ 1`, so the product is a finite non-negative index ≤ n-1.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pct = |q: f64| ratios[((n - 1) as f64 * q).round() as usize];
    PairedRatio {
        median: pct(0.5),
        median_ci95_low,
        median_ci95_high,
        p5: pct(0.05),
        p95: pct(0.95),
        rounds: n,
    }
}

fn bootstrap_median_ci95(samples: &[f64]) -> (f64, f64) {
    debug_assert!(!samples.is_empty());
    let sample_count = u64::try_from(samples.len()).expect("sample count fits u64");
    let mut seed = 0x6a09_e667_f3bc_c909_u64 ^ sample_count;
    for sample in samples {
        seed = splitmix64(seed ^ sample.to_bits());
    }

    let mut resample = Vec::with_capacity(samples.len());
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        resample.clear();
        for _ in 0..samples.len() {
            seed = splitmix64(seed);
            let index = usize::try_from(seed % sample_count).expect("sample modulus fits usize");
            resample.push(samples[index]);
        }
        resample.sort_unstable_by(f64::total_cmp);
        medians.push(percentile(&resample, 0.50));
    }
    medians.sort_unstable_by(f64::total_cmp);
    (percentile(&medians, 0.025), percentile(&medians, 0.975))
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let upper = sorted.len() - 1;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let index = (upper as f64 * quantile).round() as usize;
    sorted[index]
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An A/A null over identical closures must land at ~1.0 and bracket it.
    #[test]
    fn null_control_of_identical_work_is_near_one() {
        let work = || {
            let mut acc = 0u64;
            for i in 0..2_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let null = paired_median_ratio(41, 8, work, work);
        assert!(
            null.median > 0.75 && null.median < 1.33,
            "A/A null median {} strayed far from 1.0",
            null.median
        );
        assert!(null.median_ci95_low <= null.median);
        assert!(null.median <= null.median_ci95_high);
        assert!(null.p5 <= null.median && null.median <= null.p95);
        assert_eq!(null.rounds, 41);
    }

    /// A candidate doing ~4x the base work must be decidable against that null.
    #[test]
    fn a_large_effect_is_decidable_against_the_null() {
        let base = || {
            let mut acc = 0u64;
            for i in 0..2_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let cand = || {
            let mut acc = 0u64;
            for i in 0..8_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let null = paired_median_ratio(41, 8, base, base);
        let lever = paired_median_ratio(41, 8, base, cand);
        assert!(lever.median > 2.0, "expected ~4x, got {}", lever.median);
        if null.is_admissible_null() {
            assert!(
                lever.decidable_against(&null),
                "4x effect (median {}) should clear the null median CI [{}, {}]",
                lever.median,
                null.median_ci95_low,
                null.median_ci95_high
            );
        } else {
            assert!(
                !lever.decidable_against(&null),
                "an inadmissible A/A control must quarantine even a large effect"
            );
        }
    }

    /// A null is admitted on its median, never on how wide its CI is.
    ///
    /// The retired clause vetoed exactly this shape: a null so precise that its
    /// CI no longer reaches 1.0, despite a median 0.04% off identity.
    #[test]
    fn a_precise_null_is_admissible_even_though_its_ci_excludes_one() {
        let precise = PairedRatio {
            median: 1.0004,
            median_ci95_low: 1.0001,
            median_ci95_high: 1.0008,
            p5: 0.9990,
            p95: 1.0020,
            rounds: 641,
        };
        assert!(
            !(precise.median_ci95_low <= 1.0 && 1.0 <= precise.median_ci95_high),
            "fixture must exclude 1.0 or it does not exercise the retired clause"
        );
        assert!(precise.is_admissible_null());

        // Precision buys sensitivity: a 0.2% effect clears a 0.08% floor.
        let small_effect = PairedRatio {
            median: 0.998,
            ..precise
        };
        assert!(small_effect.decidable_against(&precise));
    }

    /// The corrected rule is a different rule, not a looser one.
    ///
    /// A null whose median has drifted 5% off identity is inadmissible even
    /// though its CI comfortably contains 1.0 — the shape the retired straddle
    /// clause admitted without complaint.
    #[test]
    fn a_biased_null_is_inadmissible_even_though_its_ci_contains_one() {
        let biased = PairedRatio {
            median: 1.05,
            median_ci95_low: 0.90,
            median_ci95_high: 1.20,
            p5: 0.85,
            p95: 1.25,
            rounds: 40,
        };
        assert!(
            biased.median_ci95_low <= 1.0 && 1.0 <= biased.median_ci95_high,
            "fixture must contain 1.0 or it does not exercise the retired clause"
        );
        assert!(!biased.is_admissible_null());

        // An inadmissible control quarantines its claim however large the effect.
        let large_effect = PairedRatio {
            median: 4.0,
            ..biased
        };
        assert!(!large_effect.decidable_against(&biased));

        // The tolerance is a boundary, not a suggestion.
        let at_edge = PairedRatio {
            median: 1.0 + NULL_MEDIAN_TOLERANCE,
            ..biased
        };
        let past_edge = PairedRatio {
            median: 1.0 + NULL_MEDIAN_TOLERANCE * 1.01,
            ..biased
        };
        assert!(at_edge.is_admissible_null());
        assert!(!past_edge.is_admissible_null());

        // The rounds floor and CI ordering sanity survive the rewrite.
        let too_few_rounds = PairedRatio {
            median: 1.0,
            rounds: 9,
            ..biased
        };
        let inverted_ci = PairedRatio {
            median: 1.0,
            median_ci95_low: 1.01,
            median_ci95_high: 0.99,
            ..biased
        };
        assert!(!too_few_rounds.is_admissible_null());
        assert!(!inverted_ci.is_admissible_null());
    }

    /// Same-ELF reproduction probe for the null-admissibility gate (`bd-pjh09`).
    ///
    /// Pinned to one core, it sweeps round counts on a planted ~4x effect. The
    /// effect is a property of the code and must reproduce across the sweep;
    /// the null's CI width is a property of the host's noise and shrinks as
    /// rounds grow. Each row prints both verdicts, so one binary shows whether
    /// the retired straddle clause flips while the effect holds still.
    ///
    /// Run it explicitly, pinned, one core per invocation:
    ///
    /// ```text
    /// taskset -c 7 cargo test -p frankensearch-core --features bench-internals \
    ///     --release null_gate_admissibility -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "timing probe: run pinned under taskset, see bd-pjh09"]
    fn null_gate_admissibility_moves_while_the_effect_reproduces() {
        let identity = print_bench_elf_sha256().expect("bench ELF identity");
        let cpu = observed_cpu();

        let base = || {
            let mut acc = 0u64;
            for i in 0..2_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };
        let cand = || {
            let mut acc = 0u64;
            for i in 0..8_000u64 {
                acc = acc.wrapping_add(black_box(i).wrapping_mul(2_654_435_761));
            }
            black_box(acc);
        };

        let mut effects = Vec::new();
        let mut new_verdicts = Vec::new();
        for &rounds in &[11usize, 41, 161, 641] {
            let null = paired_median_ratio(rounds, 8, base, base);
            let lever = paired_median_ratio(rounds, 8, base, cand);
            // The retired clause, evaluated inline so the same ELF reports both.
            let admissible_old =
                null.rounds >= 10 && null.median_ci95_low <= 1.0 && 1.0 <= null.median_ci95_high;
            let admissible_new = null.is_admissible_null();
            println!(
                "bd-pjh09 elf={} cpu={cpu} rounds={rounds} \
                 null_median={:.6} null_ci=[{:.6},{:.6}] null_half_width_pct={:.4} \
                 admissible_old={admissible_old} admissible_new={admissible_new} \
                 lever_median={:.6} decidable_new={}",
                identity.sha256,
                null.median,
                null.median_ci95_low,
                null.median_ci95_high,
                null.null_half_width() * 100.0,
                lever.median,
                lever.decidable_against(&null),
            );
            effects.push(lever.median);
            new_verdicts.push(admissible_new);
        }

        let low = effects.iter().copied().fold(f64::INFINITY, f64::min);
        let high = effects.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let spread_pct = (high / low - 1.0) * 100.0;
        println!("bd-pjh09 effect_spread_pct={spread_pct:.4} cpu={cpu}");
        assert!(
            spread_pct <= 2.5,
            "planted effect must reproduce across the precision sweep; spread {spread_pct:.4}% \
             over {effects:?} — the host is too loaded for this probe to say anything"
        );
        assert!(
            new_verdicts.iter().all(|&v| v == new_verdicts[0]),
            "corrected admissibility must not depend on how precisely the null was sampled: \
             {new_verdicts:?}"
        );
    }

    /// Kernel-reported CPU this thread last ran on, from `/proc/self/stat`.
    ///
    /// The `comm` field can contain spaces and parentheses, so fields are taken
    /// after the final `)`. Returns `-1` where the file is unavailable.
    fn observed_cpu() -> i64 {
        let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
            return -1;
        };
        let Some((_, after_comm)) = stat.rsplit_once(')') else {
            return -1;
        };
        // Field 3 (`state`) onward; `processor` is field 39, i.e. index 36 here.
        after_comm
            .split_whitespace()
            .nth(36)
            .and_then(|field| field.parse().ok())
            .unwrap_or(-1)
    }

    #[test]
    fn median_bootstrap_and_two_x_null_gate_are_deterministic() {
        let null_samples = [0.99, 1.01, 1.0, 0.995, 1.005, 1.0, 0.998, 1.002, 1.0, 1.0];
        let (low, high) = bootstrap_median_ci95(&null_samples);
        let repeated = bootstrap_median_ci95(&null_samples);
        assert_eq!(low.to_bits(), repeated.0.to_bits());
        assert_eq!(high.to_bits(), repeated.1.to_bits());

        let null = PairedRatio {
            median: 1.0,
            median_ci95_low: 0.99,
            median_ci95_high: 1.01,
            p5: 0.8,
            p95: 1.2,
            rounds: 10,
        };
        let below_floor = PairedRatio {
            median: 1.019,
            ..null
        };
        let clears_floor = PairedRatio {
            median: 1.021,
            ..null
        };
        assert!(!below_floor.decidable_against(&null));
        assert!(clears_floor.decidable_against(&null));
    }
}
