//! Quantized AVX2 dot probes: the shipped dynamic kernels versus the kept
//! fixed-shape 384-dim int8 (`bd-qdw5`) and prepared 4-bit (`bd-5ihn`)
//! specializations. Integer adds associate, so all arms are exact.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-index --bench i8_dot_ilp
//! ```

#![allow(unsafe_code)]
#![allow(
    clippy::cast_ptr_alignment,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::semicolon_if_nothing_returned,
    clippy::significant_drop_tightening
)]

#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(target_arch = "x86_64")]
use frankensearch_index::{
    PreparedQuery4bit, dot_4bit_prepared, dot_4bit_prepared_dynamic, prepare_4bit_query,
};

#[cfg(target_arch = "x86_64")]
const DIM: usize = 384;
#[cfg(target_arch = "x86_64")]
const PACKED_DIM: usize = DIM / 2;
#[cfg(target_arch = "x86_64")]
const N: usize = 4096;

#[cfg(target_arch = "x86_64")]
fn xorshift(s: &mut u64) -> i8 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    (((*s >> 40) % 255) as i8).wrapping_sub(127)
}

#[cfg(target_arch = "x86_64")]
fn build_corpus(n: usize, dim: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..n * dim).map(|_| xorshift(&mut s)).collect()
}

#[cfg(target_arch = "x86_64")]
fn build_fourbit_corpus(seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..N * PACKED_DIM)
        .map(|_| xorshift(&mut state).to_ne_bytes()[0])
        .collect()
}

/// 2-accumulator AVX2 i8·i8 dot — faithful copy of the shipped `dot_i8_i8_avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_2acc(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };
    let n = stored.len().min(query.len());
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let s = _mm256_loadu_si256(stored.as_ptr().add(i).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add(i).cast::<__m256i>());
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(s_lo, q_lo));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(s_hi, q_hi));
            i += 32;
        }
        let acc = _mm256_add_epi32(acc0, acc1);
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        let mut sum = _mm_cvtsi128_si32(sum32);
        while i < n {
            sum += i32::from(stored[i]) * i32::from(query[i]);
            i += 1;
        }
        sum
    }
}

/// Production-shape probe: the shipped two-accumulator reduction specialized
/// for MiniLM's fixed 384 dimensions. Expanding the 12 blocks removes the
/// dynamic `min`, loop test/increment, and scalar-tail machinery from every
/// corpus-vector dot while preserving the exact integer reduction tree.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_2acc_fixed384(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };
    debug_assert_eq!(stored.len(), DIM);
    debug_assert_eq!(query.len(), DIM);
    macro_rules! accumulate32 {
        ($offset:expr, $acc0:ident, $acc1:ident) => {{
            let s = _mm256_loadu_si256(stored.as_ptr().add($offset).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add($offset).cast::<__m256i>());
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            $acc0 = _mm256_add_epi32($acc0, _mm256_madd_epi16(s_lo, q_lo));
            $acc1 = _mm256_add_epi32($acc1, _mm256_madd_epi16(s_hi, q_hi));
        }};
    }
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        accumulate32!(0, acc0, acc1);
        accumulate32!(32, acc0, acc1);
        accumulate32!(64, acc0, acc1);
        accumulate32!(96, acc0, acc1);
        accumulate32!(128, acc0, acc1);
        accumulate32!(160, acc0, acc1);
        accumulate32!(192, acc0, acc1);
        accumulate32!(224, acc0, acc1);
        accumulate32!(256, acc0, acc1);
        accumulate32!(288, acc0, acc1);
        accumulate32!(320, acc0, acc1);
        accumulate32!(352, acc0, acc1);
        let acc = _mm256_add_epi32(acc0, acc1);
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        _mm_cvtsi128_si32(sum32)
    }
}

#[cfg(target_arch = "x86_64")]
fn time_fixed_batch<const FIXED: bool>(corpus: &[i8], query: &[i8], inner: usize) -> Duration {
    let started = Instant::now();
    let mut acc = 0_i32;
    for _ in 0..inner {
        for i in 0..N {
            let vector = &corpus[i * DIM..i * DIM + DIM];
            let dot = if FIXED {
                // SAFETY: the benchmark gates AVX2 and passes exact 384-element slices.
                unsafe { dot_i8_2acc_fixed384(black_box(vector), black_box(query)) }
            } else {
                // SAFETY: the benchmark gates AVX2.
                unsafe { dot_i8_2acc(black_box(vector), black_box(query)) }
            };
            acc = acc.wrapping_add(dot);
        }
    }
    black_box(acc);
    started.elapsed()
}

#[cfg(target_arch = "x86_64")]
fn paired_fixed_ratio<const CANDIDATE_FIXED: bool>(corpus: &[i8], query: &[i8]) -> Vec<f64> {
    const PAIRS: usize = 31;
    const INNER: usize = 32;
    let mut ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let (a1, b1, b2, a2) = if pair.is_multiple_of(2) {
            (
                time_fixed_batch::<false>(corpus, query, INNER),
                time_fixed_batch::<CANDIDATE_FIXED>(corpus, query, INNER),
                time_fixed_batch::<CANDIDATE_FIXED>(corpus, query, INNER),
                time_fixed_batch::<false>(corpus, query, INNER),
            )
        } else {
            let b1 = time_fixed_batch::<CANDIDATE_FIXED>(corpus, query, INNER);
            let a1 = time_fixed_batch::<false>(corpus, query, INNER);
            let a2 = time_fixed_batch::<false>(corpus, query, INNER);
            let b2 = time_fixed_batch::<CANDIDATE_FIXED>(corpus, query, INNER);
            (a1, b1, b2, a2)
        };
        ratios.push(
            ((b1.as_secs_f64() / a1.as_secs_f64()) * (b2.as_secs_f64() / a2.as_secs_f64())).sqrt(),
        );
    }
    ratios.sort_unstable_by(f64::total_cmp);
    ratios
}

#[cfg(target_arch = "x86_64")]
fn time_fourbit_batch<const FIXED: bool>(
    corpus: &[u8],
    query: &PreparedQuery4bit,
    inner: usize,
) -> Duration {
    let started = Instant::now();
    let mut acc = 0_i32;
    for _ in 0..inner {
        for i in 0..N {
            let vector = &corpus[i * PACKED_DIM..i * PACKED_DIM + PACKED_DIM];
            let dot = if FIXED {
                dot_4bit_prepared(black_box(vector), black_box(query))
            } else {
                dot_4bit_prepared_dynamic(black_box(vector), black_box(query))
            };
            acc = acc.wrapping_add(dot);
        }
    }
    black_box(acc);
    started.elapsed()
}

#[cfg(target_arch = "x86_64")]
fn paired_fourbit_ratio<const CANDIDATE_FIXED: bool>(
    corpus: &[u8],
    query: &PreparedQuery4bit,
) -> Vec<f64> {
    const PAIRS: usize = 31;
    const INNER: usize = 32;
    let mut ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let (a1, b1, b2, a2) = if pair.is_multiple_of(2) {
            (
                time_fourbit_batch::<false>(corpus, query, INNER),
                time_fourbit_batch::<CANDIDATE_FIXED>(corpus, query, INNER),
                time_fourbit_batch::<CANDIDATE_FIXED>(corpus, query, INNER),
                time_fourbit_batch::<false>(corpus, query, INNER),
            )
        } else {
            let b1 = time_fourbit_batch::<CANDIDATE_FIXED>(corpus, query, INNER);
            let a1 = time_fourbit_batch::<false>(corpus, query, INNER);
            let a2 = time_fourbit_batch::<false>(corpus, query, INNER);
            let b2 = time_fourbit_batch::<CANDIDATE_FIXED>(corpus, query, INNER);
            (a1, b1, b2, a2)
        };
        ratios.push(
            ((b1.as_secs_f64() / a1.as_secs_f64()) * (b2.as_secs_f64() / a2.as_secs_f64())).sqrt(),
        );
    }
    ratios.sort_unstable_by(f64::total_cmp);
    ratios
}

/// 4-accumulator AVX2 i8·i8 dot — unrolls two 32-byte chunks per iteration into
/// four independent `vpaddd` chains. Bit-identical (integer sum).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_4acc(stored: &[i8], query: &[i8]) -> i32 {
    use core::arch::x86_64::{
        __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_madd_epi16, _mm256_setzero_si256,
    };
    let n = stored.len().min(query.len());
    macro_rules! madd_at {
        ($off:expr) => {{
            let s = _mm256_loadu_si256(stored.as_ptr().add($off).cast::<__m256i>());
            let q = _mm256_loadu_si256(query.as_ptr().add($off).cast::<__m256i>());
            let s_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s));
            let q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
            let s_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(s));
            let q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(q));
            (_mm256_madd_epi16(s_lo, q_lo), _mm256_madd_epi16(s_hi, q_hi))
        }};
    }
    unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 64 <= n {
            let (a, b) = madd_at!(i);
            let (c, d) = madd_at!(i + 32);
            acc0 = _mm256_add_epi32(acc0, a);
            acc1 = _mm256_add_epi32(acc1, b);
            acc2 = _mm256_add_epi32(acc2, c);
            acc3 = _mm256_add_epi32(acc3, d);
            i += 64;
        }
        while i + 32 <= n {
            let (a, b) = madd_at!(i);
            acc0 = _mm256_add_epi32(acc0, a);
            acc1 = _mm256_add_epi32(acc1, b);
            i += 32;
        }
        let acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3));
        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256::<1>(acc),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0b01>(sum64));
        let mut sum = _mm_cvtsi128_si32(sum32);
        while i < n {
            sum += i32::from(stored[i]) * i32::from(query[i]);
            i += 1;
        }
        sum
    }
}

#[cfg_attr(not(target_arch = "x86_64"), allow(clippy::needless_pass_by_ref_mut))]
fn bench(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("i8_dot_ilp: avx2 not available; skipping");
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = c;
        eprintln!("i8_dot_ilp: non-x86_64; skipping");
    }
    #[cfg(target_arch = "x86_64")]
    {
        let corpus = build_corpus(N, DIM, 0x9E37);
        let query: Vec<i8> = {
            let mut s = 0xD1CE_u64;
            (0..DIM).map(|_| xorshift(&mut s)).collect()
        };
        for i in 0..N {
            let v = &corpus[i * DIM..i * DIM + DIM];
            let a = unsafe { dot_i8_2acc(v, &query) };
            let b = unsafe { dot_i8_4acc(v, &query) };
            let fixed = unsafe { dot_i8_2acc_fixed384(v, &query) };
            assert_eq!(a, b, "4-acc must equal 2-acc (integer) at {i}");
            assert_eq!(a, fixed, "fixed-384 must equal dynamic 2-acc at {i}");
        }

        // Paired AB/BA timing with an A/A control calibrates the sub-microsecond
        // scheduler/order floor in the same binary before Criterion's estimates.
        black_box(time_fixed_batch::<false>(&corpus, &query, 8));
        black_box(time_fixed_batch::<true>(&corpus, &query, 8));
        let null = paired_fixed_ratio::<false>(&corpus, &query);
        let lever = paired_fixed_ratio::<true>(&corpus, &query);
        let middle = null.len() / 2;
        let null_median = null[middle];
        let lever_median = lever[middle];
        eprintln!(
            "[fixed384-paired] null_median={null_median:.6} null_p5={:.6} null_p95={:.6} \
             lever_median={lever_median:.6} lever_p5={:.6} lever_p95={:.6} \
             calibrated_ratio={:.6} parity=true pairs={}",
            null[1],
            null[null.len() - 2],
            lever[1],
            lever[lever.len() - 2],
            lever_median / null_median,
            lever.len()
        );

        let mut fixed_group = c.benchmark_group("i8_dot_fixed384");
        fixed_group.sample_size(10);
        fixed_group.warm_up_time(Duration::from_millis(250));
        fixed_group.measurement_time(Duration::from_secs(1));
        fixed_group.bench_function("dynamic", |bch| {
            bch.iter(|| black_box(time_fixed_batch::<false>(&corpus, &query, 1)))
        });
        fixed_group.bench_function("fixed", |bch| {
            bch.iter(|| black_box(time_fixed_batch::<true>(&corpus, &query, 1)))
        });
        fixed_group.finish();

        let fourbit_corpus = build_fourbit_corpus(0x4b17_3840);
        let fourbit_query_bytes = {
            let mut state = 0x4b17_d1ce;
            (0..PACKED_DIM)
                .map(|_| xorshift(&mut state).to_ne_bytes()[0])
                .collect::<Vec<_>>()
        };
        let fourbit_query = prepare_4bit_query(&fourbit_query_bytes);
        for i in 0..N {
            let vector = &fourbit_corpus[i * PACKED_DIM..i * PACKED_DIM + PACKED_DIM];
            assert_eq!(
                dot_4bit_prepared_dynamic(vector, &fourbit_query),
                dot_4bit_prepared(vector, &fourbit_query),
                "fixed 384-dim 4-bit dot must equal dynamic at row {i}"
            );
        }
        black_box(time_fourbit_batch::<false>(
            &fourbit_corpus,
            &fourbit_query,
            8,
        ));
        black_box(time_fourbit_batch::<true>(
            &fourbit_corpus,
            &fourbit_query,
            8,
        ));
        let fourbit_null = paired_fourbit_ratio::<false>(&fourbit_corpus, &fourbit_query);
        let fourbit_lever = paired_fourbit_ratio::<true>(&fourbit_corpus, &fourbit_query);
        let fourbit_middle = fourbit_null.len() / 2;
        let fourbit_null_median = fourbit_null[fourbit_middle];
        let fourbit_lever_median = fourbit_lever[fourbit_middle];
        eprintln!(
            "[fourbit-fixed384-paired] null_median={fourbit_null_median:.6} \
             null_p5={:.6} null_p95={:.6} lever_median={fourbit_lever_median:.6} \
             lever_p5={:.6} lever_p95={:.6} calibrated_ratio={:.6} \
             parity=true pairs={}",
            fourbit_null[1],
            fourbit_null[fourbit_null.len() - 2],
            fourbit_lever[1],
            fourbit_lever[fourbit_lever.len() - 2],
            fourbit_lever_median / fourbit_null_median,
            fourbit_lever.len()
        );

        let mut fourbit_group = c.benchmark_group("fourbit_dot_fixed384");
        fourbit_group.sample_size(10);
        fourbit_group.warm_up_time(Duration::from_millis(250));
        fourbit_group.measurement_time(Duration::from_secs(1));
        fourbit_group.bench_function("dynamic", |bch| {
            bch.iter(|| {
                black_box(time_fourbit_batch::<false>(
                    &fourbit_corpus,
                    &fourbit_query,
                    1,
                ))
            })
        });
        fourbit_group.bench_function("fixed", |bch| {
            bch.iter(|| {
                black_box(time_fourbit_batch::<true>(
                    &fourbit_corpus,
                    &fourbit_query,
                    1,
                ))
            })
        });
        fourbit_group.finish();

        let mut group = c.benchmark_group("i8_dot_ilp");
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(3));
        group.bench_with_input(BenchmarkId::new("acc2", N), &N, |bch, _| {
            bch.iter(|| {
                let mut acc = 0_i32;
                for i in 0..N {
                    let v = &corpus[i * DIM..i * DIM + DIM];
                    acc = acc.wrapping_add(unsafe { dot_i8_2acc(black_box(v), black_box(&query)) });
                }
                black_box(acc)
            });
        });
        group.bench_with_input(BenchmarkId::new("acc4", N), &N, |bch, _| {
            bch.iter(|| {
                let mut acc = 0_i32;
                for i in 0..N {
                    let v = &corpus[i * DIM..i * DIM + DIM];
                    acc = acc.wrapping_add(unsafe { dot_i8_4acc(black_box(v), black_box(&query)) });
                }
                black_box(acc)
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
