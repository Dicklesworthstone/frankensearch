//! Small runtime-dispatched SIMD helpers for the embedders.
//!
//! The workspace builds without a global `+avx2`, so LLVM auto-vectorizes the hot
//! element-wise loops only to the SSE2 baseline (16-byte ops). These helpers
//! runtime-detect AVX2 and use 32-byte ops, which roughly doubles the per-cycle
//! load bandwidth on the memory-bound accumulate.

/// Element-wise `sum[d] += row[d]` — the model2vec mean-pool inner loop.
///
/// Runtime-dispatches to an AVX2 kernel (32-byte `vmovups`/`vaddps`) when
/// available; the portable scalar loop (which LLVM auto-vectorizes to SSE2) is the
/// fallback. **Bit-identical** to the scalar path: each `sum[d] += row[d]` is an
/// independent element-wise add (no cross-lane reduction), so SIMD only changes how
/// many dims are added per instruction, never the per-dim arithmetic.
#[inline]
pub fn accumulate_f32_into(sum: &mut [f32], row: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 verified present by the runtime check above.
            #[allow(unsafe_code)]
            unsafe {
                accumulate_f32_into_avx2(sum, row);
            }
            return;
        }
    }
    for (s, r) in sum.iter_mut().zip(row.iter()) {
        *s += *r;
    }
}

/// Minimum token count for software-prefetching `Model2Vec` embedding rows.
///
/// Short query sequences keep their gathered rows resident and regress when the
/// prefetch instructions are added. Long document sequences are the cache-cold
/// index-time regime where fetching a future full row can overlap the current
/// row's accumulation.
const MODEL2VEC_PREFETCH_MIN_TOKENS: usize = 512;

/// Distance, in token rows, between the row being accumulated and prefetched.
#[cfg(target_arch = "x86_64")]
const MODEL2VEC_PREFETCH_DISTANCE: usize = 4;

/// Number of `f32` values in one 64-byte cache line.
#[cfg(target_arch = "x86_64")]
const CACHE_LINE_F32: usize = 16;

/// Native width of the shipping Potion embedding rows.
const MODEL2VEC_NATIVE_DIMENSIONS: usize = 256;

/// Maximum number of valid rows whose ordered additions share one live sum row.
#[cfg(target_arch = "x86_64")]
const MODEL2VEC_FUSED_ROWS: usize = 4;

/// Mean-pool Model2Vec rows into `sum`, returning the number of in-vocabulary rows.
///
/// This is the production gather loop and the benchmark's candidate arm. On
/// x86-64 CPUs with AVX2, native-256 sequences shorter than
/// [`MODEL2VEC_PREFETCH_MIN_TOKENS`] keep each vector lane live while adding at
/// most four valid rows in token order. Portable builds, non-AVX2 x86-64 CPUs,
/// all other dimensions, and sequences at or above that threshold retain their
/// original accumulation paths exactly.
#[doc(hidden)]
#[inline]
pub fn accumulate_model2vec_rows(
    sum: &mut [f32],
    embeddings: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
) -> usize {
    debug_assert_eq!(
        embeddings.len(),
        vocab_size.saturating_mul(sum.len()),
        "embedding table shape must match vocab_size × dimensions"
    );

    if sum.len() == MODEL2VEC_NATIVE_DIMENSIONS && token_ids.len() < MODEL2VEC_PREFETCH_MIN_TOKENS {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime feature check above proves the required AVX2 ISA.
            #[allow(unsafe_code)]
            unsafe {
                #[cfg(test)]
                record_model2vec_accumulation_route(Model2VecAccumulationRoute::Native256ShortAvx2);
                return accumulate_model2vec_rows_native_256_short_avx2(
                    sum, embeddings, token_ids, vocab_size,
                );
            }
        }

        #[cfg(test)]
        record_model2vec_accumulation_route(Model2VecAccumulationRoute::Base);
        return accumulate_model2vec_rows_base(sum, embeddings, token_ids, vocab_size);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if token_ids.len() >= MODEL2VEC_PREFETCH_MIN_TOKENS {
            #[cfg(test)]
            record_model2vec_accumulation_route(Model2VecAccumulationRoute::Prefetched);
            return accumulate_model2vec_rows_prefetched(sum, embeddings, token_ids, vocab_size);
        }
    }

    #[cfg(test)]
    record_model2vec_accumulation_route(Model2VecAccumulationRoute::Base);
    accumulate_model2vec_rows_base(sum, embeddings, token_ids, vocab_size)
}

/// Accumulate short native-256 sequences with AVX2 vectors that remain live
/// across one to four gathered rows. Each lane adds its rows in original token
/// order; invalid token IDs do not consume a fused-row slot.
///
/// # Safety
///
/// The caller must prove AVX2 is available with a runtime feature check.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn accumulate_model2vec_rows_native_256_short_avx2(
    sum: &mut [f32],
    embeddings: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
) -> usize {
    debug_assert_eq!(sum.len(), MODEL2VEC_NATIVE_DIMENSIONS);

    let mut count = 0_usize;
    let mut position = 0_usize;
    while position < token_ids.len() {
        let mut rows: [&[f32]; MODEL2VEC_FUSED_ROWS] = [&[]; MODEL2VEC_FUSED_ROWS];
        let mut row_count = 0_usize;

        while position < token_ids.len() && row_count < MODEL2VEC_FUSED_ROWS {
            let index = token_ids[position] as usize;
            position += 1;
            if index < vocab_size {
                let start = index * MODEL2VEC_NATIVE_DIMENSIONS;
                rows[row_count] = &embeddings[start..start + MODEL2VEC_NATIVE_DIMENSIONS];
                row_count += 1;
                count += 1;
            }
        }

        if row_count != 0 {
            // SAFETY: `sum` and every selected row have exactly 256 f32 values.
            // The loop loads and stores eight values at offsets below 256, and
            // `row_count` is constrained to the 1..=4 range above.
            unsafe {
                accumulate_model2vec_rows_fused_avx2(sum, &rows, row_count);
            }
        }
    }
    count
}

/// Add one to four 256-wide rows to `sum` without spilling the running AVX2
/// vector between rows. The explicit match gives each case a fixed ordered add
/// sequence; there is no reduction across vector lanes.
///
/// # Safety
///
/// `sum` and the first `row_count` entries of `rows` must each have 256 values,
/// `row_count` must be in 1..=4, and the caller must provide AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn accumulate_model2vec_rows_fused_avx2(
    sum: &mut [f32],
    rows: &[&[f32]; MODEL2VEC_FUSED_ROWS],
    row_count: usize,
) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    for offset in (0..MODEL2VEC_NATIVE_DIMENSIONS).step_by(8) {
        // SAFETY: guaranteed by this function's contract and the eight-wide loop.
        unsafe {
            let sum_vector = _mm256_loadu_ps(sum.as_ptr().add(offset));
            let result = match row_count {
                1 => _mm256_add_ps(sum_vector, _mm256_loadu_ps(rows[0].as_ptr().add(offset))),
                2 => {
                    let with_first =
                        _mm256_add_ps(sum_vector, _mm256_loadu_ps(rows[0].as_ptr().add(offset)));
                    _mm256_add_ps(with_first, _mm256_loadu_ps(rows[1].as_ptr().add(offset)))
                }
                3 => {
                    let with_first =
                        _mm256_add_ps(sum_vector, _mm256_loadu_ps(rows[0].as_ptr().add(offset)));
                    let with_second =
                        _mm256_add_ps(with_first, _mm256_loadu_ps(rows[1].as_ptr().add(offset)));
                    _mm256_add_ps(with_second, _mm256_loadu_ps(rows[2].as_ptr().add(offset)))
                }
                4 => {
                    let with_first =
                        _mm256_add_ps(sum_vector, _mm256_loadu_ps(rows[0].as_ptr().add(offset)));
                    let with_second =
                        _mm256_add_ps(with_first, _mm256_loadu_ps(rows[1].as_ptr().add(offset)));
                    let with_third =
                        _mm256_add_ps(with_second, _mm256_loadu_ps(rows[2].as_ptr().add(offset)));
                    _mm256_add_ps(with_third, _mm256_loadu_ps(rows[3].as_ptr().add(offset)))
                }
                _ => unreachable!("the caller constrains row_count to 1..=4"),
            };
            _mm256_storeu_ps(sum.as_mut_ptr().add(offset), result);
        }
    }
}

#[inline]
fn accumulate_model2vec_rows_base(
    sum: &mut [f32],
    embeddings: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
) -> usize {
    let dimensions = sum.len();
    let mut count = 0_usize;
    for &token_id in token_ids {
        let idx = token_id as usize;
        if idx < vocab_size {
            let start = idx * dimensions;
            accumulate_f32_into(sum, &embeddings[start..start + dimensions]);
            count += 1;
        }
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn accumulate_model2vec_rows_prefetched(
    sum: &mut [f32],
    embeddings: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
) -> usize {
    let dimensions = sum.len();
    let mut count = 0_usize;
    for (position, &token_id) in token_ids.iter().enumerate() {
        if let Some(&future_id) = token_ids.get(position + MODEL2VEC_PREFETCH_DISTANCE) {
            let future_idx = future_id as usize;
            if future_idx < vocab_size {
                prefetch_f32_row(embeddings, future_idx * dimensions, dimensions);
            }
        }

        let idx = token_id as usize;
        if idx < vocab_size {
            let start = idx * dimensions;
            accumulate_f32_into(sum, &embeddings[start..start + dimensions]);
            count += 1;
        }
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_f32_row(embeddings: &[f32], start: usize, dimensions: usize) {
    use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

    debug_assert!(start.saturating_add(dimensions) <= embeddings.len());
    let mut offset = 0_usize;
    while offset < dimensions {
        // SAFETY: `start + offset` is within the embedding row by the loop bound
        // and caller's validated table shape. `_mm_prefetch` is only a cache hint.
        #[allow(unsafe_code)]
        unsafe {
            _mm_prefetch(
                embeddings.as_ptr().add(start + offset).cast::<i8>(),
                _MM_HINT_T0,
            );
        }
        offset += CACHE_LINE_F32;
    }
}

/// Hand-written AVX2 `sum[d] += row[d]` (8 f32 / instruction).
///
/// # Safety
/// Caller must ensure `avx2` is available (the dispatch in [`accumulate_f32_into`]
/// guarantees it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
fn accumulate_f32_into_avx2(sum: &mut [f32], row: &[f32]) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let n = sum.len().min(row.len());
    let chunks = n / 8;
    // SAFETY: avx2 by contract; every load/store is `c < chunks`-bounded (`c*8+8 ≤ n`).
    unsafe {
        for c in 0..chunks {
            let s = _mm256_loadu_ps(sum.as_ptr().add(c * 8));
            let r = _mm256_loadu_ps(row.as_ptr().add(c * 8));
            _mm256_storeu_ps(sum.as_mut_ptr().add(c * 8), _mm256_add_ps(s, r));
        }
    }
    for i in (chunks * 8)..n {
        sum[i] += row[i];
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Model2VecAccumulationRoute {
    #[cfg(target_arch = "x86_64")]
    Native256ShortAvx2,
    #[cfg(target_arch = "x86_64")]
    Prefetched,
    Base,
}

#[cfg(test)]
thread_local! {
    static LAST_MODEL2VEC_ACCUMULATION_ROUTE: std::cell::Cell<Option<Model2VecAccumulationRoute>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[inline]
fn record_model2vec_accumulation_route(route: Model2VecAccumulationRoute) {
    LAST_MODEL2VEC_ACCUMULATION_ROUTE.with(|last_route| last_route.set(Some(route)));
}

#[cfg(test)]
pub(crate) fn last_model2vec_accumulation_route_for_test() -> Model2VecAccumulationRoute {
    LAST_MODEL2VEC_ACCUMULATION_ROUTE.with(|last_route| {
        last_route
            .get()
            .expect("a Model2Vec accumulation route must have been observed")
    })
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_f32_into, accumulate_model2vec_rows, last_model2vec_accumulation_route_for_test,
        Model2VecAccumulationRoute,
    };

    fn former_model2vec_gather(
        sum: &mut [f32],
        embeddings: &[f32],
        ids: &[u32],
        vocab: usize,
    ) -> usize {
        let dimensions = sum.len();
        let mut count = 0_usize;
        for &token_id in ids {
            let index = token_id as usize;
            if index < vocab {
                let start = index * dimensions;
                accumulate_f32_into(sum, &embeddings[start..start + dimensions]);
                count += 1;
            }
        }
        count
    }

    fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], scenario: &str) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{scenario}"
        );
    }

    #[test]
    fn avx2_accumulate_matches_scalar() {
        // The AVX2 path must be byte-for-byte identical to the scalar fallback.
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            #[allow(clippy::cast_precision_loss)]
            ((state >> 40) as f32 / (1_u64 << 23) as f32 - 1.0)
        };
        for &dim in &[0_usize, 1, 7, 8, 9, 16, 31, 128, 256, 257, 384] {
            let row: Vec<f32> = (0..dim).map(|_| next()).collect();
            // Accumulate several rows so values build up beyond a single add.
            let mut simd = vec![0.0_f32; dim];
            let mut scalar = vec![0.0_f32; dim];
            for _ in 0..5 {
                accumulate_f32_into(&mut simd, &row);
                for (s, r) in scalar.iter_mut().zip(row.iter()) {
                    *s += *r;
                }
            }
            let sb: Vec<u32> = simd.iter().map(|x| x.to_bits()).collect();
            let cb: Vec<u32> = scalar.iter().map(|x| x.to_bits()).collect();
            assert_eq!(sb, cb, "dim={dim}");
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)] // VOCAB = 19 always fits in u32
    fn model2vec_prefetch_gate_matches_original_gather() {
        const VOCAB: usize = 19;
        for &dimensions in &[1_usize, 7, 8, 31, 128, 256, 257] {
            let embeddings: Vec<f32> = (0..VOCAB * dimensions)
                .map(|index| {
                    #[allow(clippy::cast_precision_loss)]
                    let value = index as f32;
                    value.mul_add(0.000_976_562_5, -0.5)
                })
                .collect();
            for &tokens in &[0_usize, 1, 127, 128, 255, 256, 511, 512, 513, 1024] {
                let ids: Vec<u32> = (0..tokens)
                    .map(|index| {
                        if index % 17 == 0 {
                            VOCAB as u32 + 3
                        } else {
                            (index % VOCAB) as u32
                        }
                    })
                    .collect();
                let mut expected = vec![0.0_f32; dimensions];
                let expected_count =
                    former_model2vec_gather(&mut expected, &embeddings, &ids, VOCAB);

                let mut actual = vec![0.0_f32; dimensions];
                let actual_count = accumulate_model2vec_rows(&mut actual, &embeddings, &ids, VOCAB);
                assert_eq!(
                    actual_count, expected_count,
                    "dim={dimensions}, tokens={tokens}"
                );
                assert_f32_bits_eq(
                    &actual,
                    &expected,
                    &format!("dim={dimensions}, tokens={tokens}"),
                );
            }
        }
    }

    #[test]
    fn model2vec_native_256_short_matches_the_actual_former_route_bit_for_bit() {
        const VOCAB: usize = 7;
        const DIMENSIONS: usize = 256;
        const OOV_TOKEN: u32 = VOCAB as u32 + 1;

        let mut embeddings: Vec<f32> = (0..VOCAB * DIMENSIONS)
            .map(|index| {
                #[allow(clippy::cast_precision_loss)]
                let value = index as f32;
                value.mul_add(0.000_976_562_5, -0.5)
            })
            .collect();
        embeddings[0] = -0.0;
        embeddings[DIMENSIONS] = 0.0;
        embeddings[DIMENSIONS * 2 + 1] = f32::INFINITY;
        embeddings[DIMENSIONS * 3 + 1] = f32::NEG_INFINITY;
        embeddings[DIMENSIONS * 4 + 2] = f32::from_bits(0x7fc0_0123);
        embeddings[DIMENSIONS * 5 + 3] = f32::INFINITY;
        embeddings[DIMENSIONS * 6 + 3] = f32::NEG_INFINITY;

        let special_initial_sum: Vec<f32> = (0..DIMENSIONS)
            .map(|dimension| match dimension % 8 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::INFINITY,
                3 => f32::NEG_INFINITY,
                4 => f32::from_bits(0x7fc0_0456),
                5 => 1.25,
                6 => -3.5,
                _ => f32::MIN_POSITIVE,
            })
            .collect();

        for (initial_name, initial_sum) in [
            ("zero-initial", vec![0.0_f32; DIMENSIONS]),
            ("arbitrary-initial", special_initial_sum),
        ] {
            for &tokens in &[1_usize, 2, 3, 4, 8, 16, 32, 64, 511, 512, 513] {
                let all_oov = vec![OOV_TOKEN; tokens];
                let interleaved: Vec<u32> = (0..tokens)
                    .map(|position| match position % 13 {
                        0 | 2 | 4 | 5 | 8 | 10 => OOV_TOKEN,
                        1 => 0,
                        3 => 1,
                        6 => 2,
                        7 => 3,
                        9 => 4,
                        11 => 5,
                        _ => 6,
                    })
                    .collect();

                for (pattern_name, ids) in [("all-oov", all_oov), ("interleaved", interleaved)] {
                    let mut expected = initial_sum.clone();
                    let expected_count =
                        former_model2vec_gather(&mut expected, &embeddings, &ids, VOCAB);
                    let mut actual = initial_sum.clone();
                    let actual_count =
                        accumulate_model2vec_rows(&mut actual, &embeddings, &ids, VOCAB);

                    let scenario = format!("{initial_name}, {pattern_name}, tokens={tokens}");
                    assert_eq!(actual_count, expected_count, "{scenario}");
                    assert_f32_bits_eq(&actual, &expected, &scenario);
                }
            }
        }
    }

    fn expected_model2vec_accumulation_route(
        dimensions: usize,
        token_count: usize,
    ) -> Model2VecAccumulationRoute {
        #[cfg(target_arch = "x86_64")]
        {
            if dimensions == 256 && token_count < MODEL2VEC_PREFETCH_MIN_TOKENS {
                if std::is_x86_feature_detected!("avx2") {
                    Model2VecAccumulationRoute::Native256ShortAvx2
                } else {
                    Model2VecAccumulationRoute::Base
                }
            } else if token_count >= MODEL2VEC_PREFETCH_MIN_TOKENS {
                Model2VecAccumulationRoute::Prefetched
            } else {
                Model2VecAccumulationRoute::Base
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (dimensions, token_count);
            Model2VecAccumulationRoute::Base
        }
    }

    #[test]
    fn model2vec_route_observer_rejects_boundary_misroutes() {
        const VOCAB: usize = 3;
        for &(dimensions, token_count) in &[
            (256_usize, 1_usize),
            (256, 2),
            (256, 3),
            (256, 4),
            (256, 511),
            (256, 512),
            (256, 513),
            (255, 511),
            (255, 512),
            (257, 511),
            (257, 512),
        ] {
            let embeddings = vec![0.25_f32; VOCAB * dimensions];
            let ids = vec![0_u32; token_count];
            let mut sum = vec![0.0_f32; dimensions];
            let count = accumulate_model2vec_rows(&mut sum, &embeddings, &ids, VOCAB);

            assert_eq!(
                count, token_count,
                "dimensions={dimensions}, tokens={token_count}"
            );
            assert_eq!(
                last_model2vec_accumulation_route_for_test(),
                expected_model2vec_accumulation_route(dimensions, token_count),
                "dimensions={dimensions}, tokens={token_count}"
            );
        }
    }
}
