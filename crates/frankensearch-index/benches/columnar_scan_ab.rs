//! FUNDAMENTALLY-DIFFERENT PRIMITIVE: a tiled-columnar (SoA) exact-dot scan vs the shipped
//! row-major (AoS) 4-accumulator dot.
//!
//! Row-major: each candidate's `dim` floats are contiguous; the dot is `dim/8` f32x8 FMAs into
//! 4 accumulators PLUS one horizontal `reduce_add` per candidate.
//! Tiled-columnar: candidates are grouped by 8; each group stores its `dim×8` floats as `dim`
//! consecutive f32x8 blocks (`tile[d*8 + lane] = candidate(g*8+lane)[d]`). The scan does
//! `acc += splat(query[d]) * load8(tile[d])` over `dim`, so the 8 lanes ARE the 8 candidates'
//! dots — **no horizontal reduction**. Same FMA count per candidate (`dim/8`), one fewer reduce.
//! The tile is contiguous (`dim*8` floats per group) → sequential, cache-friendly (unlike a naive
//! transpose that strides by N across dims). This tests whether eliminating the per-candidate
//! reduce (a fixed cost that dominates more at small `dim`) beats the row-major dot.
//!
//! Both scan all N candidates and produce N dots; parity asserted approximately (the two
//! reassociations differ by ULP — ranking-preserving, the accepted search-time trade). Within-
//! process paired AB/BA against an A/A null floor.
//!
//! Run with:
//! ```bash
//! AGENT_NAME=cc_fse CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
//!   rch exec -- cargo bench -p frankensearch-index --profile release --bench columnar_scan_ab
//! ```
#![allow(clippy::doc_markdown)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use wide::f32x8;

const PROFILE_ROUNDS: usize = 41;
const PAIRED_ROUND_PAIRS: usize = 41;

/// (n_candidates, dim). n is a multiple of 8; dim a multiple of 32. dim=64/128 is the BOLD
/// range where the fixed per-candidate reduce is a meaningful fraction.
const SHAPES: &[(usize, usize)] = &[(4096, 64), (4096, 128), (16384, 64), (16384, 128)];

#[inline]
fn load8(s: &[f32]) -> f32x8 {
    f32x8::from([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
}

/// Deterministic xorshift → pseudo-random f32 in ~[-1, 1].
fn make_f32s(count: usize, seed: u64) -> Vec<f32> {
    let mut r = seed | 1;
    (0..count)
        .map(|_| {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            ((r >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Row-major AoS: `rowmajor[i*dim + d]`. Returns N dots via the shipped-style 4-acc dot. The
/// query f32x8 lanes are HOISTED once (not reloaded per candidate) so the A/B isolates the LAYOUT
/// / reduce difference, not query-load redundancy a hand-tuned kernel would avoid.
fn scan_rowmajor(query: &[f32], rowmajor: &[f32], n: usize, dim: usize, out: &mut [f32]) {
    let per = dim / 8; // dim assumed a multiple of 32 (so also of 8)
    let query_v: Vec<f32x8> = (0..per).map(|c| load8(&query[c * 8..])).collect();
    let quad = per / 4;
    for i in 0..n {
        let v = &rowmajor[i * dim..];
        let mut acc = [f32x8::splat(0.0); 4];
        for q in 0..quad {
            let c = q * 4;
            acc[0] += query_v[c] * load8(&v[c * 8..]);
            acc[1] += query_v[c + 1] * load8(&v[(c + 1) * 8..]);
            acc[2] += query_v[c + 2] * load8(&v[(c + 2) * 8..]);
            acc[3] += query_v[c + 3] * load8(&v[(c + 3) * 8..]);
        }
        out[i] = ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add();
    }
}

/// Tiled-columnar SoA: for group g, `tile[g*dim*8 + d*8 + lane]` = candidate (g*8+lane)'s dim d.
/// Returns N dots with NO per-candidate horizontal reduce (8 dots per group fall out of the lanes).
fn scan_columnar(query: &[f32], tiled: &[f32], n: usize, dim: usize, out: &mut [f32]) {
    let groups = n / 8;
    for g in 0..groups {
        let base = g * dim * 8;
        let mut acc = f32x8::splat(0.0);
        for d in 0..dim {
            acc += f32x8::splat(query[d]) * load8(&tiled[base + d * 8..]);
        }
        out[g * 8..g * 8 + 8].copy_from_slice(&acc.to_array());
    }
}

/// Reorganize a row-major slab into the tiled-columnar layout.
fn to_tiled(rowmajor: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut tiled = vec![0.0_f32; n * dim];
    for i in 0..n {
        let g = i / 8;
        let lane = i % 8;
        let base = g * dim * 8;
        for d in 0..dim {
            tiled[base + d * 8 + lane] = rowmajor[i * dim + d];
        }
    }
    tiled
}

#[derive(Clone, Copy)]
struct RatioDistribution {
    median: f64,
    p5: f64,
    p95: f64,
    #[allow(dead_code)] // populated for symmetry with the paired-ratio report
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
            "COLUMNAR_FASTER"
        } else if self.median > null.p95 {
            "COLUMNAR_SLOWER"
        } else {
            "INSIDE_NULL_FLOOR"
        }
    }
}

fn percentile(sorted: &[Duration], pct: usize) -> Duration {
    sorted[((sorted.len() - 1) * pct + 50) / 100]
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

#[derive(Clone, Copy)]
enum Arm {
    Row,
    Col,
}

struct Corpus {
    query: Vec<f32>,
    rowmajor: Vec<f32>,
    tiled: Vec<f32>,
    n: usize,
    dim: usize,
}

fn time_arm(c: &Corpus, arm: Arm, out: &mut [f32]) -> Duration {
    let started = Instant::now();
    match arm {
        Arm::Row => scan_rowmajor(&c.query, &c.rowmajor, c.n, c.dim, out),
        Arm::Col => scan_columnar(&c.query, &c.tiled, c.n, c.dim, out),
    }
    let elapsed = started.elapsed();
    black_box(out.first().copied());
    elapsed
}

fn paired_ratio(c: &Corpus, arm_a: Arm, arm_b: Arm) -> RatioDistribution {
    let mut out = vec![0.0_f32; c.n];
    let mut run = |record: bool| {
        let ab_a = time_arm(c, arm_a, &mut out);
        let ab_b = time_arm(c, arm_b, &mut out);
        let ba_b = time_arm(c, arm_b, &mut out);
        let ba_a = time_arm(c, arm_a, &mut out);
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
        let _ = run(false);
    }
    ratio_distribution(
        (0..PAIRED_ROUND_PAIRS)
            .map(|_| run(true).expect("pair"))
            .collect(),
    )
}

fn median_ns_per_cand(c: &Corpus, arm: Arm) -> f64 {
    let mut out = vec![0.0_f32; c.n];
    for _ in 0..3 {
        black_box(time_arm(c, arm, &mut out));
    }
    let mut samples: Vec<Duration> = (0..PROFILE_ROUNDS)
        .map(|_| time_arm(c, arm, &mut out))
        .collect();
    samples.sort_unstable();
    percentile(&samples, 50).as_secs_f64() * 1e9 / c.n as f64
}

fn main() {
    eprintln!(
        "[config] shapes={} profile_rounds={PROFILE_ROUNDS}",
        SHAPES.len()
    );
    let mut all_faster = true;
    for &(n, dim) in SHAPES {
        let query = make_f32s(dim, 0xA5A5_1234);
        let rowmajor = make_f32s(n * dim, 0x1234_ABCD);
        let tiled = to_tiled(&rowmajor, n, dim);
        let c = Corpus {
            query,
            rowmajor,
            tiled,
            n,
            dim,
        };

        // Parity: the two scans agree up to ULP reassociation.
        let mut a = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        scan_rowmajor(&c.query, &c.rowmajor, n, dim, &mut a);
        scan_columnar(&c.query, &c.tiled, n, dim, &mut b);
        // ABSOLUTE error: random vectors have near-zero dots (uncorrelated), so a relative metric
        // divides by ~0. The two arms differ only by f32 reassociation (ranking-preserving).
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-2,
            "row/columnar disagree beyond ULP: max_abs={max_abs} (n={n} dim={dim})"
        );
        eprintln!("[parity] n={n} dim={dim} max_abs_err={max_abs:.2e}");

        let row_ns = median_ns_per_cand(&c, Arm::Row);
        let col_ns = median_ns_per_cand(&c, Arm::Col);
        eprintln!(
            "[profile] n={n} dim={dim} rowmajor_ns/cand={row_ns:.3} columnar_ns/cand={col_ns:.3}"
        );

        let null = paired_ratio(&c, Arm::Row, Arm::Row);
        let lever = paired_ratio(&c, Arm::Row, Arm::Col);
        eprintln!(
            "[paired] n={n} dim={dim} null median={:.4} p5={:.4} p95={:.4}",
            null.median, null.p5, null.p95
        );
        eprintln!(
            "[paired] n={n} dim={dim} columnar_vs_row median={:.4} p5={:.4} p95={:.4}",
            lever.median, lever.p5, lever.p95
        );
        let faster = null.null_is_unbiased() && lever.median < null.p5;
        all_faster &= faster;
        eprintln!(
            "[gate] n={n} dim={dim} verdict={} columnar_speedup={:.4}x faster={faster}",
            lever.verdict_against(null),
            1.0 / lever.median
        );
    }
    eprintln!(
        "[gate-summary] decision={}",
        if all_faster {
            "columnar-faster-all-shapes"
        } else {
            "MIXED-or-floor"
        }
    );
}
