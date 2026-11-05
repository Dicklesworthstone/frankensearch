//! CLS-attention SOFTWARE-PREFETCH A/B — hide the strided K/V load latency in the
//! shipped direct rank-1 kernel (`native.rs` / `cls_attention_ab` `direct_rank1`).
//!
//! In the direct CLS attention, for a fixed head the per-token K and V live
//! `STRIDE = 3H = 1152` floats apart (~4.6 KB) in the interleaved qkv buffer — so
//! scanning tokens touches a fresh, scattered cache line each step. A 4.6 KB
//! constant stride is at/beyond many HW stride-prefetchers' reach, so the QK dot
//! loop and the weighted-value-sum loop can be memory-latency-bound at large
//! `s_len`. This adds `_mm_prefetch` of the token `j+PF` K (QK loop) and V
//! (value loop). Prefetch is a hint — output is bit-identical to `direct` (parity
//! asserts max-delta 0), so this is exact and distribution-independent.
//!
//! Arms: `direct_rank1` (= shipped) vs `direct_prefetch` (new). Swept over `s_len`.
//!
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-rerank --features native \
//!     --profile release --bench cls_attention_prefetch
//! ```

use std::hint::black_box;
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wide::f32x8;

const H: usize = 384;
const NH: usize = 12;
const HD: usize = H / NH;
const STRIDE: usize = 3 * H;
const SCALE: f32 = 0.176_776_69; // 1 / sqrt(32)
const PF: usize = 4; // prefetch distance (tokens ahead)

#[derive(Default)]
struct Scratch {
    scores: Vec<f32>,
}
impl Scratch {
    fn ensure(&mut self, s_len: usize) {
        let sc = NH * s_len;
        if self.scores.len() < sc {
            self.scores.resize(sc, 0.0);
        }
    }
}

#[inline]
fn f32x8_from_slice(slice: &[f32]) -> f32x8 {
    let mut buf = [0.0f32; 8];
    buf.copy_from_slice(&slice[..8]);
    f32x8::new(buf)
}

#[inline]
fn dot_hd(q: &[f32], k: &[f32]) -> f32 {
    (f32x8_from_slice(&q[0..]) * f32x8_from_slice(&k[0..])
        + f32x8_from_slice(&q[8..]) * f32x8_from_slice(&k[8..])
        + f32x8_from_slice(&q[16..]) * f32x8_from_slice(&k[16..])
        + f32x8_from_slice(&q[24..]) * f32x8_from_slice(&k[24..]))
    .reduce_add()
}

fn softmax_row_fused(row: &mut [f32], scale: f32) {
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= row.len() {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum = sum_v.reduce_add();
    while i < row.len() {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row {
        *x *= inv;
    }
}

fn softmax_rows(data: &mut [f32], rows: usize, n: usize, scale: f32) {
    debug_assert_eq!(data.len(), rows * n);
    for row in data.chunks_exact_mut(n) {
        softmax_row_fused(row, scale);
    }
}

#[inline]
#[allow(unsafe_code)]
fn prefetch(qkv: &[f32], idx: usize) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: _mm_prefetch is a hint; idx is only dereferenced by hardware and any
    // address is architecturally valid to prefetch. We bound idx < len anyway.
    unsafe {
        if idx < qkv.len() {
            _mm_prefetch(qkv.as_ptr().add(idx).cast::<i8>(), _MM_HINT_T0);
        }
    }
}

fn weighted_value_sum_hd(qkv: &[f32], probs: &[f32], head: usize, out: &mut [f32], pf: bool) {
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    for (j, &prob) in probs.iter().enumerate() {
        if pf {
            prefetch(qkv, (j + PF) * STRIDE + 2 * H + head * HD);
        }
        let p = f32x8::splat(prob);
        let base = j * STRIDE + 2 * H + head * HD;
        acc0 += p * f32x8_from_slice(&qkv[base..base + 8]);
        acc1 += p * f32x8_from_slice(&qkv[base + 8..base + 16]);
        acc2 += p * f32x8_from_slice(&qkv[base + 16..base + 24]);
        acc3 += p * f32x8_from_slice(&qkv[base + 24..base + 32]);
    }
    out[0..8].copy_from_slice(&acc0.to_array());
    out[8..16].copy_from_slice(&acc1.to_array());
    out[16..24].copy_from_slice(&acc2.to_array());
    out[24..32].copy_from_slice(&acc3.to_array());
}

fn cls_attention_direct(
    scratch: &mut Scratch,
    qkv: &[f32],
    s_len: usize,
    out: &mut [f32],
    pf: bool,
) {
    scratch.ensure(s_len);
    let sc = NH * s_len;
    for head in 0..NH {
        let q = &qkv[head * HD..head * HD + HD];
        let row = &mut scratch.scores[head * s_len..(head + 1) * s_len];
        for (j, slot) in row.iter_mut().enumerate() {
            if pf {
                prefetch(qkv, (j + PF) * STRIDE + H + head * HD);
            }
            let k_base = j * STRIDE + H + head * HD;
            *slot = dot_hd(q, &qkv[k_base..k_base + HD]);
        }
    }
    let scores = &mut scratch.scores[..sc];
    softmax_rows(scores, NH, s_len, SCALE);
    for head in 0..NH {
        let row = &scores[head * s_len..(head + 1) * s_len];
        weighted_value_sum_hd(qkv, row, head, &mut out[head * HD..head * HD + HD], pf);
    }
}

fn qkv_fixture(s_len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(s_len * STRIDE);
    for i in 0..s_len * STRIDE {
        let v = ((i.wrapping_mul(31) % 257) as f32 - 128.0) * 0.0025;
        out.push(v);
    }
    out
}

fn assert_close(a: &[f32], b: &[f32]) {
    let max_delta = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta == 0.0,
        "prefetch variant diverged from direct by {max_delta} (must be bit-identical)"
    );
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("cls_attention_prefetch");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(800));

    for &s_len in &[64usize, 128, 256, 512] {
        let qkv = qkv_fixture(s_len);
        let mut sc0 = Scratch::default();
        let mut sc1 = Scratch::default();
        let mut base = vec![0.0f32; H];
        let mut pfd = vec![0.0f32; H];
        cls_attention_direct(&mut sc0, &qkv, s_len, &mut base, false);
        cls_attention_direct(&mut sc1, &qkv, s_len, &mut pfd, true);
        assert_close(&base, &pfd);

        group.bench_with_input(BenchmarkId::new("direct_rank1", s_len), &qkv, |b, qkv| {
            let mut scratch = Scratch::default();
            let mut out = vec![0.0f32; H];
            b.iter(|| {
                cls_attention_direct(
                    black_box(&mut scratch),
                    black_box(qkv),
                    black_box(s_len),
                    black_box(&mut out),
                    false,
                );
                black_box(&out);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("direct_prefetch", s_len),
            &qkv,
            |b, qkv| {
                let mut scratch = Scratch::default();
                let mut out = vec![0.0f32; H];
                b.iter(|| {
                    cls_attention_direct(
                        black_box(&mut scratch),
                        black_box(qkv),
                        black_box(s_len),
                        black_box(&mut out),
                        true,
                    );
                    black_box(&out);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
