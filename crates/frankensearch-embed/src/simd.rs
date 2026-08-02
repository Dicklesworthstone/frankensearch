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
#[cfg(target_arch = "x86_64")]
const MODEL2VEC_PREFETCH_MIN_TOKENS: usize = 512;

/// Distance, in token rows, between the row being accumulated and prefetched.
#[cfg(target_arch = "x86_64")]
const MODEL2VEC_PREFETCH_DISTANCE: usize = 4;

/// Number of `f32` values in one 64-byte cache line.
#[cfg(target_arch = "x86_64")]
const CACHE_LINE_F32: usize = 16;

/// Mean-pool Model2Vec rows into `sum`, returning the number of in-vocabulary rows.
///
/// This is the production gather loop and the benchmark's candidate arm. On
/// x86-64, sequences of at least [`MODEL2VEC_PREFETCH_MIN_TOKENS`] prefetch every
/// cache line of the row four tokens ahead. Short sequences and non-x86 targets
/// retain the original no-prefetch loop exactly.
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

    #[cfg(target_arch = "x86_64")]
    if token_ids.len() >= MODEL2VEC_PREFETCH_MIN_TOKENS {
        return accumulate_model2vec_rows_prefetched(sum, embeddings, token_ids, vocab_size);
    }

    accumulate_model2vec_rows_base(sum, embeddings, token_ids, vocab_size)
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
    use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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
mod tests {
    use super::{accumulate_f32_into, accumulate_model2vec_rows};

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
                let mut expected_count = 0_usize;
                for &token_id in &ids {
                    let idx = token_id as usize;
                    if idx < VOCAB {
                        let start = idx * dimensions;
                        for (sum, value) in expected
                            .iter_mut()
                            .zip(&embeddings[start..start + dimensions])
                        {
                            *sum += *value;
                        }
                        expected_count += 1;
                    }
                }

                let mut actual = vec![0.0_f32; dimensions];
                let actual_count = accumulate_model2vec_rows(&mut actual, &embeddings, &ids, VOCAB);
                assert_eq!(
                    actual_count, expected_count,
                    "dim={dimensions}, tokens={tokens}"
                );
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "dim={dimensions}, tokens={tokens}"
                );
            }
        }
    }
}
