//! Same-binary scalar-versus-`wide::u32x8` differential and throughput probe.
//!
//! Correctness is asserted before timing. The paired A/B ratio alternates the
//! two implementations within each round and calibrates an A/A null floor;
//! Every production width is gated with a varied source offset. Criterion's
//! representative sequential arms supply absolute hot-kernel elements/second,
//! not the keep decision. This does not claim end-to-end streaming speed, and
//! it never infers cycles/element from a VM's nominal clock rate.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 RCH_WORKER=<pinned-worker> \
//!   rch exec -- env CARGO_TARGET_DIR=/tmp/frankensearch-quill-postings \
//!   cargo bench -p frankensearch-quill --features bench-internals \
//!   --bench postings_decode_ab
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use frankensearch_quill::quiver::differential::{
    BitpackError, FIXTURE_ID, pack_values, unpack_scalar_into, unpack_wide_into,
};

type Decoder = fn(&[u8], u8, &mut [u32]) -> Result<(), BitpackError>;

// The paired gate covers all 127-value doc widths 0..=32 and all canonical
// 128-value frequency widths 1..=32. Criterion retains representative rows.
const CRITERION_SHAPES: &[(usize, u8)] = &[
    (127, 0),
    (127, 1),
    (127, 3),
    (127, 8),
    (127, 9),
    (127, 16),
    (127, 17),
    (127, 25),
    (127, 31),
    (127, 32),
    (128, 1),
    (128, 3),
    (128, 8),
    (128, 9),
    (128, 16),
    (128, 17),
    (128, 25),
    (128, 31),
    (128, 32),
];
const GATE_OFFSETS: [usize; 4] = [0, 1, 15, 31];

#[derive(Clone, Copy)]
enum GateVerdict {
    Win,
    Regression,
    Hold,
    InvalidNull,
}

#[derive(Default)]
struct GateSummary {
    wins: usize,
    regressions: usize,
    holds: usize,
    invalid_nulls: usize,
}

impl GateSummary {
    fn record(&mut self, verdict: GateVerdict) {
        match verdict {
            GateVerdict::Win => self.wins += 1,
            GateVerdict::Regression => self.regressions += 1,
            GateVerdict::Hold => self.holds += 1,
            GateVerdict::InvalidNull => self.invalid_nulls += 1,
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn random_u32(state: &mut u64) -> u32 {
    let mut value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    (value.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
}

fn values(count: usize, width: u8) -> Vec<u32> {
    if width == 0 {
        return vec![0; count];
    }
    let mask = u32::MAX >> (u32::BITS - u32::from(width));
    let count_u64 = u64::try_from(count).unwrap_or(u64::MAX);
    let mut state = 0x0123_4567_89ab_cdef ^ u64::from(width) ^ count_u64;
    (0..count).map(|_| random_u32(&mut state) & mask).collect()
}

fn run_unpack(input: &[u8], width: u8, output: &mut [u32], decoder: Decoder) {
    black_box(decoder(input, width, output).is_ok());
    black_box(output);
}

fn fail_setup(stage: &str, count: usize, width: u8, error: &BitpackError) -> ! {
    eprintln!("{FIXTURE_ID} {stage} failed at c{count}/w{width}: {error}");
    std::process::exit(2);
}

fn fail_parity(stage: &str, count: usize, width: u8) -> ! {
    eprintln!("{FIXTURE_ID} {stage} mismatch at c{count}/w{width}");
    std::process::exit(2);
}

fn configured_inner() -> u32 {
    match std::env::var("QUILL_POSTINGS_AB_INNER") {
        Ok(raw) => match raw.parse::<u32>() {
            Ok(inner) if inner != 0 => inner,
            _ => {
                eprintln!(
                    "{FIXTURE_ID} invalid QUILL_POSTINGS_AB_INNER={raw:?}; expected a positive u32"
                );
                std::process::exit(2);
            }
        },
        Err(std::env::VarError::NotPresent) => 256,
        Err(error) => {
            eprintln!("{FIXTURE_ID} cannot read QUILL_POSTINGS_AB_INNER: {error}");
            std::process::exit(2);
        }
    }
}

/// Classify a lever against its A/A control.
///
/// Admissibility of the control is delegated to the shared helper, which gates
/// the null's accuracy. This used to re-derive it as `p5 <= 1.0 <= p95`, a
/// clause that vetoed a null for being *tight* rather than for being wrong
/// (`bd-pjh09`). The null's raw spread is still the effect floor below.
fn classify(null: &PairedRatio, lever: &PairedRatio) -> GateVerdict {
    if !null.is_admissible_null() {
        GateVerdict::InvalidNull
    } else if lever.median < 1.0 && lever.median < null.p5 {
        GateVerdict::Win
    } else if lever.median > 1.0 && lever.median > null.p95 {
        GateVerdict::Regression
    } else {
        GateVerdict::Hold
    }
}

fn run_gate(count: usize, width: u8, offset: usize, inner: u32) -> GateVerdict {
    let expected = values(count, width);
    let packed = match pack_values(&expected, width) {
        Ok(packed) => packed,
        Err(error) => fail_setup("pack", count, width, &error),
    };
    let mut storage = vec![0xa5_u8; offset];
    storage.extend_from_slice(&packed);
    storage.extend_from_slice(&[0x5a; 7]);
    let input = &storage[offset..offset + packed.len()];

    let mut scalar_output = vec![u32::MAX; count];
    let mut wide_output = vec![u32::MAX; count];
    if let Err(error) = unpack_scalar_into(input, width, &mut scalar_output) {
        fail_setup("scalar setup", count, width, &error);
    }
    if let Err(error) = unpack_wide_into(input, width, &mut wide_output) {
        fail_setup("wide setup", count, width, &error);
    }
    if scalar_output != expected {
        fail_parity("scalar", count, width);
    }
    if wide_output != expected {
        fail_parity("wide", count, width);
    }

    let null = {
        let mut control_output = vec![u32::MAX; count];
        let mut duplicate_output = vec![u32::MAX; count];
        let mut run_a = || {
            run_unpack(
                black_box(input),
                width,
                &mut control_output,
                unpack_scalar_into,
            );
        };
        let mut run_b = || {
            run_unpack(
                black_box(input),
                width,
                &mut duplicate_output,
                unpack_scalar_into,
            );
        };
        paired_median_ratio(41, inner, &mut run_a, &mut run_b)
    };
    let lever = {
        let mut run_scalar = || {
            run_unpack(
                black_box(input),
                width,
                &mut scalar_output,
                unpack_scalar_into,
            );
        };
        let mut run_wide = || {
            run_unpack(black_box(input), width, &mut wide_output, unpack_wide_into);
        };
        paired_median_ratio(41, inner, &mut run_scalar, &mut run_wide)
    };
    let verdict = classify(&null, &lever);
    eprintln!(
        "[null] {FIXTURE_ID}/c{count}/w{width}/o{offset}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] {FIXTURE_ID}/c{count}/w{width}/o{offset}: wide/scalar median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        match verdict {
            GateVerdict::Win => "DECIDABLE WIN",
            GateVerdict::Regression => "DECIDABLE REGRESSION",
            GateVerdict::Hold => "INSIDE DIRECTIONAL NULL FLOOR (HOLD)",
            GateVerdict::InvalidNull => "INVALID A/A NULL (does not bracket 1.0)",
        }
    );
    verdict
}

fn bench(c: &mut Criterion) {
    let inner = configured_inner();

    let mut summary = GateSummary::default();
    let mut width_zero = GateSummary::default();
    for offset in GATE_OFFSETS {
        width_zero.record(run_gate(127, 0, offset, inner));
    }
    for count in [127, 128] {
        for width in 1..=32 {
            for offset in GATE_OFFSETS {
                summary.record(run_gate(count, width, offset, inner));
            }
        }
    }
    eprintln!(
        "[aggregate] {FIXTURE_ID}/nonzero: wins={} regressions={} holds={} invalid_nulls={}",
        summary.wins, summary.regressions, summary.holds, summary.invalid_nulls
    );
    eprintln!(
        "[aggregate] {FIXTURE_ID}/width-zero-control: wins={} regressions={} holds={} invalid_nulls={}",
        width_zero.wins, width_zero.regressions, width_zero.holds, width_zero.invalid_nulls
    );
    let global_outcome = if summary.invalid_nulls != 0 || width_zero.invalid_nulls != 0 {
        "INVALID (at least one A/A null did not bracket 1.0)"
    } else if summary.regressions != 0 || width_zero.regressions != 0 {
        "GLOBAL KEEP BLOCKED (at least one decisive regression)"
    } else if summary.wins == 2 * 32 * GATE_OFFSETS.len() {
        "GLOBAL KEEP ELIGIBLE (all nonzero rows win; width zero has no regression)"
    } else {
        "GLOBAL HOLD (one or more nonzero rows remain inside the null floor)"
    };
    eprintln!(
        "[policy] {global_outcome}; four source offsets are required per width, width-zero HOLD is allowed, and every nonzero row must win for wide-for-all"
    );

    for &(count, width) in CRITERION_SHAPES {
        let expected = values(count, width);
        let packed = match pack_values(&expected, width) {
            Ok(packed) => packed,
            Err(error) => fail_setup("pack", count, width, &error),
        };
        let offset = usize::from(width) % 32;
        let mut storage = vec![0xa5_u8; offset];
        storage.extend_from_slice(&packed);
        storage.extend_from_slice(&[0x5a; 7]);
        let input = &storage[offset..offset + packed.len()];
        let mut scalar_output = vec![u32::MAX; count];
        let mut wide_output = vec![u32::MAX; count];
        if let Err(error) = unpack_scalar_into(input, width, &mut scalar_output) {
            fail_setup("scalar setup", count, width, &error);
        }
        if let Err(error) = unpack_wide_into(input, width, &mut wide_output) {
            fail_setup("wide setup", count, width, &error);
        }
        if scalar_output != expected || wide_output != expected {
            fail_parity("criterion setup", count, width);
        }

        let mut group = c.benchmark_group(format!("quiver_bitpack_c{count}_w{width}_o{offset}"));
        group.throughput(Throughput::Elements(
            u64::try_from(count).unwrap_or(u64::MAX),
        ));
        group.bench_with_input(
            BenchmarkId::new("scalar", count),
            &input,
            |bencher, input| {
                bencher.iter(|| {
                    run_unpack(
                        black_box(*input),
                        width,
                        &mut scalar_output,
                        unpack_scalar_into,
                    );
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("wide", count), &input, |bencher, input| {
            bencher.iter(|| {
                run_unpack(black_box(*input), width, &mut wide_output, unpack_wide_into);
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
