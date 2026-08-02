//! ILP probe for the exact f16·f32 dot kernel: single accumulator (the shipped
//! `dot_product_f16_bytes_f32_avx2`) vs 4 independent accumulators.
//!
//! The shipped AVX2+F16C kernel accumulates into ONE `__m256` (`sum = add(sum,
//! mul(decode(f16), q))`), so each iteration's `vaddps` waits on the previous
//! one — the loop is **latency-bound on the ~4-cycle add chain** (~48 chunks for
//! dim=384). The hardware can retire `vcvtph2ps`/`vmulps` 3–4× faster than that
//! serial add allows, so independent accumulators should turn the loop
//! throughput-bound (decode-limited). This is the same multi-accumulator ILP
//! lever that paid on the MMR cosine reduction (4-acc, 1.6×).
//!
//! Both kernels are run head-to-head in one process; results are checked within a
//! tight relative epsilon (4-acc reorders the f32 partial sums — a quality-neutral
//! reassociation, the class the project already accepts for softmax/GELU/MMR).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench f16_dot_ilp
//! ```

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment,
    clippy::unreadable_literal
)]

#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_arch = "x86_64")]
use half::f16;

#[cfg(target_arch = "x86_64")]
const DIM: usize = 384;
#[cfg(target_arch = "x86_64")]
const N: usize = 4096;

#[cfg(target_arch = "x86_64")]
fn xorshift(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    #[allow(clippy::cast_precision_loss)]
    let u = (*s >> 40) as f32 / f64::from(1_u32 << 24) as f32;
    u.mul_add(2.0, -1.0)
}

#[cfg(target_arch = "x86_64")]
fn build_corpus(n: usize, dim: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut bytes = Vec::with_capacity(n * dim * 2);
    for _ in 0..n {
        for _ in 0..dim {
            bytes.extend_from_slice(&f16::from_f32(xorshift(&mut s)).to_le_bytes());
        }
    }
    bytes
}

/// Single-accumulator AVX2+F16C f16·f32 dot — a faithful copy of the shipped
/// `dot_product_f16_bytes_f32_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_f16_1acc(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let dim = query.len();
    let chunks = dim / 8;
    let mut arr = [0.0_f32; 8];
    unsafe {
        let mut sum = _mm256_setzero_ps();
        for ci in 0..chunks {
            let bits = _mm_loadu_si128(stored_bytes.as_ptr().add(ci * 16).cast::<__m128i>());
            let stored = _mm256_cvtph_ps(bits);
            let q = _mm256_loadu_ps(query.as_ptr().add(ci * 8));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(stored, q));
        }
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    let mut result = arr.iter().sum::<f32>();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        result = f16::from_le_bytes([b[0], b[1]])
            .to_f32()
            .mul_add(query[index], result);
    }
    result
}

/// 4-accumulator AVX2+F16C f16·f32 dot — four independent add chains, summed once
/// at the end. dim=384 ⇒ 48 chunks ⇒ 12 four-wide bodies, no chunk remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn dot_f16_4acc(stored_bytes: &[u8], query: &[f32]) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_add_ps, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    let dim = query.len();
    let chunks = dim / 8;
    let mut arr = [0.0_f32; 8];
    macro_rules! mul_chunk {
        ($ci:expr) => {{
            let bits = _mm_loadu_si128(stored_bytes.as_ptr().add($ci * 16).cast::<__m128i>());
            let stored = _mm256_cvtph_ps(bits);
            let q = _mm256_loadu_ps(query.as_ptr().add($ci * 8));
            _mm256_mul_ps(stored, q)
        }};
    }
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut ci = 0;
        while ci + 4 <= chunks {
            s0 = _mm256_add_ps(s0, mul_chunk!(ci));
            s1 = _mm256_add_ps(s1, mul_chunk!(ci + 1));
            s2 = _mm256_add_ps(s2, mul_chunk!(ci + 2));
            s3 = _mm256_add_ps(s3, mul_chunk!(ci + 3));
            ci += 4;
        }
        while ci < chunks {
            s0 = _mm256_add_ps(s0, mul_chunk!(ci));
            ci += 1;
        }
        let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        _mm256_storeu_ps(arr.as_mut_ptr(), sum);
    }
    let mut result = arr.iter().sum::<f32>();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        result = f16::from_le_bytes([b[0], b[1]])
            .to_f32()
            .mul_add(query[index], result);
    }
    result
}

#[cfg_attr(not(target_arch = "x86_64"), allow(clippy::needless_pass_by_ref_mut))]
fn bench(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")) {
        eprintln!("f16_dot_ilp: avx2+f16c not available; skipping");
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = c;
        eprintln!("f16_dot_ilp: non-x86_64; skipping");
    }
    #[cfg(target_arch = "x86_64")]
    bench_x86(c);
}

#[cfg(target_arch = "x86_64")]
fn bench_x86(c: &mut Criterion) {
    let stride = DIM * 2;
    let corpus = build_corpus(N, DIM, 0x51ED);
    let query: Vec<f32> = {
        let mut s = 0xC0FFEE_u64;
        (0..DIM).map(|_| xorshift(&mut s)).collect()
    };

    // Equivalence guard (within f32 reassociation epsilon).
    for i in 0..N {
        let v = &corpus[i * stride..i * stride + stride];
        let a = unsafe { dot_f16_1acc(v, &query) };
        let b = unsafe { dot_f16_4acc(v, &query) };
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1.0),
            "4-acc {b} diverged from 1-acc {a} at {i}"
        );
    }

    let mut group = c.benchmark_group("f16_dot_ilp");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    // Scan the whole corpus per iteration so we measure steady-state throughput.
    group.bench_with_input(BenchmarkId::new("acc1", N), &N, |bch, _| {
        bch.iter(|| {
            let mut acc = 0.0_f32;
            for i in 0..N {
                let v = &corpus[i * stride..i * stride + stride];
                acc += unsafe { dot_f16_1acc(black_box(v), black_box(&query)) };
            }
            black_box(acc)
        });
    });
    group.bench_with_input(BenchmarkId::new("acc4", N), &N, |bch, _| {
        bch.iter(|| {
            let mut acc = 0.0_f32;
            for i in 0..N {
                let v = &corpus[i * stride..i * stride + stride];
                acc += unsafe { dot_f16_4acc(black_box(v), black_box(&query)) };
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
