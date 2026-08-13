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
///
/// # Panics
///
/// Panics when `sum` has zero dimensions, when `vocab_size * sum.len()`
/// overflows, or when `embeddings` is not exactly that many values. The exact
/// table-length requirement is the established contract: trailing rows are not
/// permitted. Every in-vocabulary token's row bounds are checked before any
/// architecture-specific path can form a pointer.
#[doc(hidden)]
#[inline]
pub fn accumulate_model2vec_rows(
    sum: &mut [f32],
    embeddings: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
) -> usize {
    validate_model2vec_accumulation_shape(sum, embeddings, token_ids, vocab_size);

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

/// Validate the safe public `Model2Vec` accumulation inputs before architecture-
/// specific code can form raw pointers.
#[inline]
fn validate_model2vec_accumulation_shape(
    sum: &[f32],
    embeddings: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
) {
    let dimensions = sum.len();
    assert_ne!(
        dimensions, 0,
        "Model2Vec accumulation dimensions must be non-zero"
    );

    let table_len = vocab_size
        .checked_mul(dimensions)
        .expect("Model2Vec embedding table size overflows usize");
    assert_eq!(
        embeddings.len(),
        table_len,
        "Model2Vec embedding table length must equal vocab_size * dimensions"
    );

    for &token_id in token_ids {
        let index = token_id as usize;
        if index < vocab_size {
            let (_, end) = model2vec_row_bounds(index, dimensions);
            assert!(
                end <= embeddings.len(),
                "Model2Vec token row must lie within the embedding table"
            );
        }
    }
}

/// Return the checked half-open bounds of one in-vocabulary embedding row.
#[inline]
fn model2vec_row_bounds(index: usize, dimensions: usize) -> (usize, usize) {
    let start = index
        .checked_mul(dimensions)
        .expect("Model2Vec embedding row offset overflows usize");
    let end = start
        .checked_add(dimensions)
        .expect("Model2Vec embedding row end overflows usize");
    (start, end)
}

/// Form one embedding row only after calculating its bounds with checked arithmetic.
#[inline]
fn model2vec_embedding_row(embeddings: &[f32], index: usize, dimensions: usize) -> &[f32] {
    let (start, end) = model2vec_row_bounds(index, dimensions);
    &embeddings[start..end]
}

/// Accumulate short native-256 sequences with AVX2 vectors that remain live
/// across one to four gathered rows. Each lane adds its rows in original token
/// order; invalid token IDs do not consume a fused-row slot.
///
/// # Safety
///
/// The caller must prove AVX2 is available with a runtime feature check. The
/// public boundary has also proved that `sum` is 256-wide, the table has an
/// exact checked shape, and every selected in-vocabulary row has a checked
/// 256-wide range in `embeddings`.
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
                rows[row_count] =
                    model2vec_embedding_row(embeddings, index, MODEL2VEC_NATIVE_DIMENSIONS);
                row_count += 1;
                count += 1;
            }
        }

        if row_count != 0 {
            // SAFETY: boundary validation proved `sum` and every selected row
            // are exactly 256 f32 values. The loop loads and stores eight values
            // at offsets below 256, and `row_count` is constrained to 1..=4.
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
            accumulate_f32_into(sum, model2vec_embedding_row(embeddings, idx, dimensions));
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
                prefetch_f32_row(model2vec_embedding_row(embeddings, future_idx, dimensions));
            }
        }

        let idx = token_id as usize;
        if idx < vocab_size {
            accumulate_f32_into(sum, model2vec_embedding_row(embeddings, idx, dimensions));
            count += 1;
        }
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn prefetch_f32_row(row: &[f32]) {
    use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

    for offset in (0..row.len()).step_by(CACHE_LINE_F32) {
        // SAFETY: `row` was formed from checked bounds before this call, so it
        // is an in-allocation slice. `offset < row.len()` by the range and the
        // pointer stays in that slice. `_mm_prefetch` is only a cache hint.
        #[allow(unsafe_code)]
        unsafe {
            _mm_prefetch(row.as_ptr().add(offset).cast::<i8>(), _MM_HINT_T0);
        }
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
        MODEL2VEC_PREFETCH_MIN_TOKENS, Model2VecAccumulationRoute, accumulate_f32_into,
        accumulate_model2vec_rows, last_model2vec_accumulation_route_for_test,
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
        // Fixture invariant: this vocabulary is a small literal, so every id in
        // it addresses as a u32 token. The conversion is checked rather than
        // truncating because a fixture grown past u32 would silently stop
        // producing an out-of-vocabulary id, and this test would keep passing
        // while no longer discriminating the OOV path at all.
        let oov_token =
            u32::try_from(VOCAB).expect("fixture vocabulary must fit a u32 token id") + 1;

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
                let all_oov = vec![oov_token; tokens];
                let interleaved: Vec<u32> = (0..tokens)
                    .map(|position| match position % 13 {
                        0 | 2 | 4 | 5 | 8 | 10 => oov_token,
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
            let ids: Vec<u32> = (0..token_count)
                .map(|position| if position % 5 == 0 { u32::MAX } else { 0 })
                .collect();
            let mut sum = vec![0.0_f32; dimensions];
            let count = accumulate_model2vec_rows(&mut sum, &embeddings, &ids, VOCAB);
            let expected_count = ids
                .iter()
                .filter(|&&token_id| (token_id as usize) < VOCAB)
                .count();

            assert_eq!(
                count, expected_count,
                "dimensions={dimensions}, tokens={token_count}"
            );
            assert_eq!(
                last_model2vec_accumulation_route_for_test(),
                expected_model2vec_accumulation_route(dimensions, token_count),
                "dimensions={dimensions}, tokens={token_count}"
            );
        }
    }

    #[test]
    #[should_panic(
        expected = "Model2Vec embedding table length must equal vocab_size * dimensions"
    )]
    fn model2vec_accumulator_rejects_short_table_before_512_token_prefetch() {
        const DIMENSIONS: usize = 256;
        let mut sum = [0.0_f32; DIMENSIONS];
        let embeddings = [0.0_f32; DIMENSIONS - 1];
        let token_ids = vec![0_u32; MODEL2VEC_PREFETCH_MIN_TOKENS];

        let _ = accumulate_model2vec_rows(&mut sum, &embeddings, &token_ids, 1);
    }

    #[test]
    #[should_panic(
        expected = "Model2Vec embedding table length must equal vocab_size * dimensions"
    )]
    fn model2vec_accumulator_rejects_mismatched_sum_and_extra_table_values() {
        let mut sum = [0.0_f32; 2];
        let embeddings = [0.0_f32; 3];

        let _ = accumulate_model2vec_rows(&mut sum, &embeddings, &[0], 1);
    }

    #[test]
    #[should_panic(expected = "Model2Vec accumulation dimensions must be non-zero")]
    fn model2vec_accumulator_rejects_zero_dimensions() {
        let mut sum: [f32; 0] = [];
        let embeddings: [f32; 0] = [];

        let _ = accumulate_model2vec_rows(&mut sum, &embeddings, &[0], 1);
    }

    #[test]
    #[should_panic(expected = "Model2Vec embedding table size overflows usize")]
    fn model2vec_accumulator_rejects_table_size_overflow_without_allocation() {
        let mut sum = [0.0_f32; 2];
        let embeddings: [f32; 0] = [];

        let _ = accumulate_model2vec_rows(&mut sum, &embeddings, &[], usize::MAX);
    }
}
