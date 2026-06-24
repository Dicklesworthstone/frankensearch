//! Dot-product hot-loop benchmarks for the brute-force vector top-k path.
//!
//! These mirror the inner loop of `search.rs`: for every query we score a batch
//! of stored vectors via the SIMD dot-product kernels. The dominant production
//! path is `dot_product_f16_bytes_f32` (f16 quantization decoded from mmap'd
//! bytes); the f32 variants are included for completeness.
//!
//! Each kernel is benched **head-to-head** against a `*_baseline` copy of the
//! original single-`f32x8`-accumulator implementation. Running old vs new in the
//! same process on the same CPU makes the `new/old` ratio immune to which rch
//! worker the run lands on (absolute ns vary by host; the ratio does not).
//!
//! Run with (set the target dir to your own agent lane):
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/<agent-lane> \
//!   rch exec -- cargo bench -p frankensearch-index --bench dot_product
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_index::{
    dot_product_f16_bytes_f32, dot_product_f16_f32, dot_product_f32_bytes_f32, dot_product_f32_f32,
};
use half::f16;
use wide::f32x8;

// ── Baselines: the original single-accumulator kernels (pre-change) ─────────
// Copied verbatim from the prior `simd.rs` so the head-to-head isolates the
// "4 independent accumulators" change (same multiply/add op semantics).

fn dot_f32_f32_baseline(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let a_arr = [ac[0], ac[1], ac[2], ac[3], ac[4], ac[5], ac[6], ac[7]];
        let b_arr = [bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], bc[6], bc[7]];
        sum += f32x8::from(a_arr) * f32x8::from(b_arr);
    }
    let mut result = sum.reduce_add();
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        result += x * y;
    }
    result
}

fn dot_f16_f32_baseline(stored: &[f16], query: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut stored_chunks = stored.chunks_exact(8);
    let mut query_chunks = query.chunks_exact(8);
    for (sc, qc) in stored_chunks.by_ref().zip(query_chunks.by_ref()) {
        let s = [
            sc[0].to_f32(),
            sc[1].to_f32(),
            sc[2].to_f32(),
            sc[3].to_f32(),
            sc[4].to_f32(),
            sc[5].to_f32(),
            sc[6].to_f32(),
            sc[7].to_f32(),
        ];
        let q = [qc[0], qc[1], qc[2], qc[3], qc[4], qc[5], qc[6], qc[7]];
        sum += f32x8::from(s) * f32x8::from(q);
    }
    let mut result = sum.reduce_add();
    for (s, q) in stored_chunks
        .remainder()
        .iter()
        .zip(query_chunks.remainder())
    {
        result += s.to_f32() * q;
    }
    result
}

fn dot_f16_bytes_baseline(stored_bytes: &[u8], query: &[f32]) -> f32 {
    let dim = query.len();
    let chunks = dim / 8;
    let mut sum = f32x8::splat(0.0);
    for chunk_index in 0..chunks {
        let b = &stored_bytes[chunk_index * 16..];
        let stored_chunk = f32x8::from([
            f16::from_le_bytes([b[0], b[1]]).to_f32(),
            f16::from_le_bytes([b[2], b[3]]).to_f32(),
            f16::from_le_bytes([b[4], b[5]]).to_f32(),
            f16::from_le_bytes([b[6], b[7]]).to_f32(),
            f16::from_le_bytes([b[8], b[9]]).to_f32(),
            f16::from_le_bytes([b[10], b[11]]).to_f32(),
            f16::from_le_bytes([b[12], b[13]]).to_f32(),
            f16::from_le_bytes([b[14], b[15]]).to_f32(),
        ]);
        let q = &query[chunk_index * 8..];
        let query_chunk = f32x8::from([q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]]);
        sum += stored_chunk * query_chunk;
    }
    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }
    result
}

fn dot_f32_bytes_baseline(stored_bytes: &[u8], query: &[f32]) -> f32 {
    let dim = query.len();
    let chunks = dim / 8;
    let mut sum = f32x8::splat(0.0);
    for chunk_index in 0..chunks {
        let b = &stored_bytes[chunk_index * 32..];
        let stored_chunk = f32x8::from([
            f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            f32::from_le_bytes([b[12], b[13], b[14], b[15]]),
            f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            f32::from_le_bytes([b[20], b[21], b[22], b[23]]),
            f32::from_le_bytes([b[24], b[25], b[26], b[27]]),
            f32::from_le_bytes([b[28], b[29], b[30], b[31]]),
        ]);
        let q = &query[chunk_index * 8..];
        let query_chunk = f32x8::from([q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]]);
        sum += stored_chunk * query_chunk;
    }
    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }
    result
}

// ── Corpus ──────────────────────────────────────────────────────────────────

/// Deterministic pseudo-random f32 in [-1, 1] (xorshift; no rng dep).
fn gen_f32(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    // Top 24 bits fit exactly in f32; map [0, 2^24) -> [0, 1) -> [-1, 1].
    let unit = (*state >> 40) as f32 / (1_u64 << 24) as f32;
    unit.mul_add(2.0, -1.0)
}

struct Corpus {
    query: Vec<f32>,
    stored_f32: Vec<Vec<f32>>,
    stored_f16: Vec<Vec<f16>>,
    stored_f16_bytes: Vec<Vec<u8>>,
    stored_f32_bytes: Vec<Vec<u8>>,
}

fn build_corpus(dim: usize, n: usize) -> Corpus {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ (dim as u64).wrapping_mul(0x1234_5678);
    let query: Vec<f32> = (0..dim).map(|_| gen_f32(&mut state)).collect();

    let mut stored_f32 = Vec::with_capacity(n);
    let mut stored_f16 = Vec::with_capacity(n);
    let mut stored_f16_bytes = Vec::with_capacity(n);
    let mut stored_f32_bytes = Vec::with_capacity(n);

    for _ in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| gen_f32(&mut state)).collect();
        let v16: Vec<f16> = v.iter().copied().map(f16::from_f32).collect();
        let mut b16 = Vec::with_capacity(dim * 2);
        for x in &v16 {
            b16.extend_from_slice(&x.to_le_bytes());
        }
        let mut b32 = Vec::with_capacity(dim * 4);
        for x in &v {
            b32.extend_from_slice(&x.to_le_bytes());
        }
        stored_f32.push(v);
        stored_f16.push(v16);
        stored_f16_bytes.push(b16);
        stored_f32_bytes.push(b32);
    }

    Corpus {
        query,
        stored_f32,
        stored_f16,
        stored_f16_bytes,
        stored_f32_bytes,
    }
}

fn bench_dot(c: &mut Criterion) {
    // Realistic embedding dims: 256 (potion fast tier) and 384 (MiniLM quality tier).
    for &dim in &[256_usize, 384] {
        let n = 10_000_usize;
        let corpus = build_corpus(dim, n);
        let q = &corpus.query;

        let mut group = c.benchmark_group(format!("dot/dim{dim}"));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(3));
        group.throughput(criterion::Throughput::Elements(n as u64));

        // f16_bytes — the dominant production path.
        group.bench_function(BenchmarkId::new("f16_bytes_new", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for bytes in &corpus.stored_f16_bytes {
                    acc += dot_product_f16_bytes_f32(black_box(bytes), black_box(q)).unwrap();
                }
                black_box(acc)
            });
        });
        group.bench_function(BenchmarkId::new("f16_bytes_old", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for bytes in &corpus.stored_f16_bytes {
                    acc += dot_f16_bytes_baseline(black_box(bytes), black_box(q));
                }
                black_box(acc)
            });
        });

        group.bench_function(BenchmarkId::new("f32_bytes_new", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for bytes in &corpus.stored_f32_bytes {
                    acc += dot_product_f32_bytes_f32(black_box(bytes), black_box(q)).unwrap();
                }
                black_box(acc)
            });
        });
        group.bench_function(BenchmarkId::new("f32_bytes_old", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for bytes in &corpus.stored_f32_bytes {
                    acc += dot_f32_bytes_baseline(black_box(bytes), black_box(q));
                }
                black_box(acc)
            });
        });

        group.bench_function(BenchmarkId::new("f16_slice_new", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for v in &corpus.stored_f16 {
                    acc += dot_product_f16_f32(black_box(v), black_box(q)).unwrap();
                }
                black_box(acc)
            });
        });
        group.bench_function(BenchmarkId::new("f16_slice_old", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for v in &corpus.stored_f16 {
                    acc += dot_f16_f32_baseline(black_box(v), black_box(q));
                }
                black_box(acc)
            });
        });

        group.bench_function(BenchmarkId::new("f32_slice_new", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for v in &corpus.stored_f32 {
                    acc += dot_product_f32_f32(black_box(v), black_box(q)).unwrap();
                }
                black_box(acc)
            });
        });
        group.bench_function(BenchmarkId::new("f32_slice_old", n), |b| {
            b.iter(|| {
                let mut acc = 0.0_f32;
                for v in &corpus.stored_f32 {
                    acc += dot_f32_f32_baseline(black_box(v), black_box(q));
                }
                black_box(acc)
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_dot);
criterion_main!(benches);
