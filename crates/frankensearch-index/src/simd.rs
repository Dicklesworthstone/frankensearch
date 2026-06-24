//! Portable SIMD dot-product helpers for vector search.

use frankensearch_core::{SearchError, SearchResult};
use half::f16;
use wide::{f32x8, u32x8};

// ── SIMD load/decode helpers ────────────────────────────────────────────────
//
// The f32 byte-decoding dot-product kernel below uses four independent f32x8
// accumulators per 32 elements. A single accumulator serializes on SIMD add
// latency; four accumulators expose enough instruction-level parallelism to run
// the loop at throughput instead.

/// Load 8 consecutive `f32` lanes from a fixed 32-element array at a const offset.
#[inline(always)]
fn load8_f32<const OFF: usize>(a: &[f32; 32]) -> f32x8 {
    f32x8::from([
        a[OFF],
        a[OFF + 1],
        a[OFF + 2],
        a[OFF + 3],
        a[OFF + 4],
        a[OFF + 5],
        a[OFF + 6],
        a[OFF + 7],
    ])
}

/// Decode 8 little-endian `f32` values from a 32-byte block to `f32x8`.
#[inline(always)]
fn decode8_f32(b: &[u8; 32]) -> f32x8 {
    f32x8::from([
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
        f32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        f32::from_le_bytes([b[16], b[17], b[18], b[19]]),
        f32::from_le_bytes([b[20], b[21], b[22], b[23]]),
        f32::from_le_bytes([b[24], b[25], b[26], b[27]]),
        f32::from_le_bytes([b[28], b[29], b[30], b[31]]),
    ])
}

// ── Branchless SIMD f16 → f32 widen ─────────────────────────────────────────
//
// The f16 dot-product paths are decode-bound: per-element scalar `f16::to_f32()`
// dominates (see docs/NEGATIVE_EVIDENCE.md). This widens 8 f16 lanes at once via
// the Giesen "magic multiply" trick (a denormal-as-float multiply renormalizes
// both normals and subnormals in one shot), with a pure-integer inf/nan fixup.
//
// It is **bit-exact** to `f16::to_f32()` for every finite (incl. subnormal) and
// zero input, and maps inf→inf / nan→nan with the sign preserved (the nan
// payload's quiet bit may differ, which only affects the bits of a NaN score —
// never produced by real, finite, L2-normalized embeddings). Exhaustively
// verified over all 65 536 bit patterns by `simd_f16_widen_is_bit_exact`.
//
// Because the decoded f32 values are bit-identical for finite data and the
// accumulation order is unchanged, every dot-product score is bit-identical to
// the prior scalar-decode kernel on real corpora — no determinism/golden risk.

/// 2^112 — the Giesen magic factor that rebiases the f16 exponent.
const F16_WIDEN_MAGIC: f32 = f32::from_bits(0x7780_0000);

/// Widen 8 f16 bit-patterns (held in the low 16 bits of each `u32x8` lane) to f32.
#[inline(always)]
fn widen8_f16_lanes(h: u32x8) -> f32x8 {
    let sign = (h & u32x8::splat(0x0000_8000)) << 16_u32;
    let exp_mant = (h & u32x8::splat(0x0000_7fff)) << 13_u32;
    let scaled = bytemuck::cast::<u32x8, f32x8>(exp_mant) * f32x8::splat(F16_WIDEN_MAGIC);
    let scaled_bits = bytemuck::cast::<f32x8, u32x8>(scaled);

    // inf/nan: f16 exponent field == 0x7c00. Detect via carry out of bit 15
    // ((he + 0x0400) sets bit 15 iff he == 0x7c00), spread to a full lane mask,
    // and OR the f32 exponent up to all-ones (0xff << 23).
    let he = h & u32x8::splat(0x0000_7c00);
    let carry = (he + u32x8::splat(0x0000_0400)) & u32x8::splat(0x0000_8000);
    let infnan_mask = (carry >> 15_u32) * u32x8::splat(0xff << 23);

    bytemuck::cast::<u32x8, f32x8>((scaled_bits | infnan_mask) | sign)
}

/// Decode 8 little-endian f16 values from a 16-byte block to `f32x8` (SIMD widen).
#[inline(always)]
fn widen8_f16_bytes(b: &[u8; 16]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(u16::from_le_bytes([b[0], b[1]])),
        u32::from(u16::from_le_bytes([b[2], b[3]])),
        u32::from(u16::from_le_bytes([b[4], b[5]])),
        u32::from(u16::from_le_bytes([b[6], b[7]])),
        u32::from(u16::from_le_bytes([b[8], b[9]])),
        u32::from(u16::from_le_bytes([b[10], b[11]])),
        u32::from(u16::from_le_bytes([b[12], b[13]])),
        u32::from(u16::from_le_bytes([b[14], b[15]])),
    ];
    widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes))
}

/// Widen 8 consecutive `f16` values (from a fixed array) to `f32x8` (SIMD widen).
#[inline(always)]
fn widen8_f16_slice(s: &[f16; 8]) -> f32x8 {
    let lanes: [u32; 8] = [
        u32::from(s[0].to_bits()),
        u32::from(s[1].to_bits()),
        u32::from(s[2].to_bits()),
        u32::from(s[3].to_bits()),
        u32::from(s[4].to_bits()),
        u32::from(s[5].to_bits()),
        u32::from(s[6].to_bits()),
        u32::from(s[7].to_bits()),
    ];
    widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes))
}

/// Dot product between two f32 vectors.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f32_f32(a: &[f32], b: &[f32]) -> SearchResult<f32> {
    ensure_same_len(a.len(), b.len())?;
    Ok(dot_product_f32_f32_unchecked(a, b))
}

/// Dot product between an f16 stored vector and an f32 query vector.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn dot_product_f16_f32(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    ensure_same_len(stored.len(), query.len())?;

    let mut sum = f32x8::splat(0.0);
    let mut stored_chunks = stored.chunks_exact(8);
    let mut query_chunks = query.chunks_exact(8);

    for (stored_chunk, query_chunk) in stored_chunks.by_ref().zip(query_chunks.by_ref()) {
        let s: &[f16; 8] = stored_chunk.try_into().expect("chunks_exact(8) yields 8 elements");
        let q: &[f32; 8] = query_chunk.try_into().expect("chunks_exact(8) yields 8 elements");
        sum += widen8_f16_slice(s) * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for (s, q) in stored_chunks
        .remainder()
        .iter()
        .zip(query_chunks.remainder())
    {
        result += s.to_f32() * q;
    }
    Ok(result)
}

/// Cosine similarity helper for f16 stored vectors.
///
/// Assumes both vectors are already L2-normalized and therefore returns the
/// raw dot product value.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when slice lengths differ.
pub fn cosine_similarity_f16(stored: &[f16], query: &[f32]) -> SearchResult<f32> {
    dot_product_f16_f32(stored, query)
}

/// Dot product between f16 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f16s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 2`.
pub fn dot_product_f16_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 2 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 2,
        });
    }

    let chunks = dim / 8;
    let mut sum = f32x8::splat(0.0);

    for chunk_index in 0..chunks {
        let byte_offset = chunk_index * 16;
        let query_offset = chunk_index * 8;

        let block: &[u8; 16] = stored_bytes[byte_offset..byte_offset + 16]
            .try_into()
            .expect("16-byte f16 block");
        let stored_chunk = widen8_f16_bytes(block);

        let q: &[f32; 8] = query[query_offset..query_offset + 8]
            .try_into()
            .expect("8-element query block");

        sum += stored_chunk * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 2..];
        let val = f16::from_le_bytes([b[0], b[1]]).to_f32();
        result = val.mul_add(query[index], result);
    }

    Ok(result)
}

/// Dot product between f32 bytes and an f32 query vector.
///
/// Avoids intermediate allocation by decoding f32s on the fly.
///
/// # Errors
///
/// Returns `SearchError::DimensionMismatch` when `stored_bytes.len()` is not
/// exactly `query.len() * 4`.
pub fn dot_product_f32_bytes_f32(stored_bytes: &[u8], query: &[f32]) -> SearchResult<f32> {
    let dim = query.len();
    if stored_bytes.len() != dim * 4 {
        return Err(SearchError::DimensionMismatch {
            expected: dim,
            found: stored_bytes.len() / 4,
        });
    }

    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    // Main loop: 32 elements (128 stored bytes) per iteration, 4 accumulators.
    let groups = dim / 32;
    for g in 0..groups {
        let bo = g * 128;
        let qo = g * 32;
        let block: &[u8; 128] = stored_bytes[bo..bo + 128]
            .try_into()
            .expect("128-byte f32 block");
        let q: &[f32; 32] = query[qo..qo + 32]
            .try_into()
            .expect("32-element query block");
        acc0 +=
            decode8_f32(block[0..32].try_into().expect("32-byte sub-block")) * load8_f32::<0>(q);
        acc1 +=
            decode8_f32(block[32..64].try_into().expect("32-byte sub-block")) * load8_f32::<8>(q);
        acc2 +=
            decode8_f32(block[64..96].try_into().expect("32-byte sub-block")) * load8_f32::<16>(q);
        acc3 +=
            decode8_f32(block[96..128].try_into().expect("32-byte sub-block")) * load8_f32::<24>(q);
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3);

    // Tail of full 8-element chunks not covered by the 32-wide main loop.
    let chunks = dim / 8;
    for chunk_index in (groups * 4)..chunks {
        let bo = chunk_index * 32;
        let qo = chunk_index * 8;
        let block: &[u8; 32] = stored_bytes[bo..bo + 32].try_into().expect("32-byte block");
        let q: &[f32; 8] = query[qo..qo + 8].try_into().expect("8-element block");
        sum += decode8_f32(block) * f32x8::from(*q);
    }

    let mut result = sum.reduce_add();
    for index in (chunks * 8)..dim {
        let b = &stored_bytes[index * 4..];
        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        result = val.mul_add(query[index], result);
    }

    Ok(result)
}

fn dot_product_f32_f32_unchecked(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);

    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        let a_arr = [
            a_chunk[0], a_chunk[1], a_chunk[2], a_chunk[3], a_chunk[4], a_chunk[5], a_chunk[6],
            a_chunk[7],
        ];
        let b_arr = [
            b_chunk[0], b_chunk[1], b_chunk[2], b_chunk[3], b_chunk[4], b_chunk[5], b_chunk[6],
            b_chunk[7],
        ];
        sum += f32x8::from(a_arr) * f32x8::from(b_arr);
    }

    let mut result = sum.reduce_add();
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        result += x * y;
    }
    result
}

const fn ensure_same_len(expected: usize, found: usize) -> SearchResult<()> {
    if expected != found {
        return Err(SearchError::DimensionMismatch { expected, found });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive proof that the SIMD f16->f32 widen matches the scalar reference
    /// (`f16::to_f32`) for **every** one of the 65 536 f16 bit patterns. Finite,
    /// zero and subnormal inputs must be bit-identical; inf/nan must map to
    /// inf/nan with matching sign (the nan payload's quiet bit may legitimately
    /// differ between the two decode methods).
    #[test]
    fn simd_f16_widen_is_bit_exact() {
        for base in (0u32..=0xFFFF).step_by(8) {
            let lanes = [
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
            ];
            let widened = widen8_f16_lanes(bytemuck::cast::<[u32; 8], u32x8>(lanes));
            let out = bytemuck::cast::<f32x8, [f32; 8]>(widened);
            for (lane, &simd) in lanes.iter().zip(out.iter()) {
                let bits = u16::try_from(*lane).expect("<= 0xFFFF");
                let scalar = f16::from_bits(bits).to_f32();
                if scalar.is_nan() {
                    assert!(simd.is_nan(), "bits={bits:#06x}: scalar nan, simd={simd}");
                    assert_eq!(
                        scalar.is_sign_negative(),
                        simd.is_sign_negative(),
                        "bits={bits:#06x}: nan sign mismatch"
                    );
                } else {
                    assert_eq!(
                        scalar.to_bits(),
                        simd.to_bits(),
                        "bits={bits:#06x}: scalar={scalar} simd={simd}"
                    );
                }
            }
        }
    }

    fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn scalar_dot_f16(stored: &[f16], query: &[f32]) -> f32 {
        stored.iter().zip(query).map(|(x, y)| x.to_f32() * y).sum()
    }

    fn normalize(vec: &[f32]) -> Vec<f32> {
        let norm = vec.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            return vec.to_vec();
        }
        vec.iter().map(|value| value / norm).collect()
    }

    #[test]
    fn simd_matches_scalar_f32() {
        let a = vec![
            0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55,
            0.31,
        ];
        let b = vec![
            -0.8, 0.7, 0.6, -0.2, 0.3, 0.9, -0.4, 0.1, 0.12, 0.21, -0.14, 0.75, -0.22, 0.35, 0.66,
            -0.19,
        ];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn simd_matches_scalar_f16() {
        let query = vec![
            0.4, -0.1, 0.6, 0.2, -0.3, 0.8, 0.7, -0.5, 0.9, -0.6, 0.11, 0.25, 0.41, -0.72, 0.55,
            0.31,
        ];
        let stored = vec![
            f16::from_f32(-0.8),
            f16::from_f32(0.7),
            f16::from_f32(0.6),
            f16::from_f32(-0.2),
            f16::from_f32(0.3),
            f16::from_f32(0.9),
            f16::from_f32(-0.4),
            f16::from_f32(0.1),
            f16::from_f32(0.12),
            f16::from_f32(0.21),
            f16::from_f32(-0.14),
            f16::from_f32(0.75),
            f16::from_f32(-0.22),
            f16::from_f32(0.35),
            f16::from_f32(0.66),
            f16::from_f32(-0.19),
        ];
        let simd = dot_product_f16_f32(&stored, &query).expect("dot product");
        let scalar = scalar_dot_f16(&stored, &query);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn remainder_elements_are_handled() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let b = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!((simd - scalar).abs() < 1e-6, "simd={simd}, scalar={scalar}");
    }

    #[test]
    fn zero_vector_dot_product_is_zero() {
        let stored = vec![f16::from_f32(0.0); 16];
        let query = vec![1.0; 16];
        let result = dot_product_f16_f32(&stored, &query).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn nan_input_propagates_nan() {
        let mut a = vec![1.0; 16];
        a[3] = f32::NAN;
        let b = vec![1.0; 16];
        let result = dot_product_f32_f32(&a, &b).expect("dot product");
        assert!(result.is_nan());
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let a = vec![1.0; 8];
        let b = vec![1.0; 7];
        let err = dot_product_f32_f32(&a, &b).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 7
            }
        ));
    }

    #[test]
    fn f16_precision_error_is_bounded_for_unit_vectors() {
        let pattern = [
            0.11_f32, -0.07, 0.19, 0.02, -0.13, 0.23, 0.31, -0.17, 0.05, -0.29, 0.37, 0.41,
        ];
        let mut stored_full = Vec::with_capacity(384);
        let mut query = Vec::with_capacity(384);
        for index in 0..384 {
            let value = pattern[index % pattern.len()];
            let other = pattern[(index + 3) % pattern.len()];
            stored_full.push(value);
            query.push(other);
        }
        let stored_full = normalize(&stored_full);
        let query = normalize(&query);
        let stored_f16: Vec<f16> = stored_full.iter().copied().map(f16::from_f32).collect();

        let f32_dot = scalar_dot_f32(&stored_full, &query);
        let f16_dot = dot_product_f16_f32(&stored_f16, &query).expect("dot product");
        assert!(
            (f32_dot - f16_dot).abs() < 0.01,
            "f32_dot={f32_dot}, f16_dot={f16_dot}"
        );
    }

    // ─── bd-1l4g tests begin ───

    #[test]
    fn cosine_similarity_f16_matches_dot_product() {
        let stored: Vec<f16> = (0_u16..16)
            .map(|i| f16::from_f32(f32::from(i) * 0.1))
            .collect();
        let query: Vec<f32> = (0_u16..16).map(|i| f32::from(i) * 0.2).collect();

        let cosine = cosine_similarity_f16(&stored, &query).expect("cosine");
        let dot = dot_product_f16_f32(&stored, &query).expect("dot");
        assert!(
            (cosine - dot).abs() < f32::EPSILON,
            "cosine_similarity_f16 should delegate to dot_product_f16_f32"
        );
    }

    #[test]
    fn cosine_similarity_f16_dimension_mismatch() {
        let stored = vec![f16::from_f32(1.0); 8];
        let query = vec![1.0_f32; 9];
        let err = cosine_similarity_f16(&stored, &query).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 8,
                found: 9
            }
        ));
    }

    #[test]
    fn dot_product_f16_f32_dimension_mismatch() {
        let stored = vec![f16::from_f32(1.0); 4];
        let query = vec![1.0_f32; 5];
        let err = dot_product_f16_f32(&stored, &query).expect_err("must fail");
        assert!(matches!(
            err,
            SearchError::DimensionMismatch {
                expected: 4,
                found: 5
            }
        ));
    }

    #[test]
    fn empty_vectors_dot_product_f32() {
        let result = dot_product_f32_f32(&[], &[]).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn empty_vectors_dot_product_f16() {
        let result = dot_product_f16_f32(&[], &[]).expect("dot product");
        assert!(result.abs() < f32::EPSILON);
    }

    #[test]
    fn exactly_eight_elements_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!(
            (simd - scalar).abs() < 1e-6,
            "exactly 8 elements (one full SIMD chunk, no remainder)"
        );
    }

    #[test]
    fn single_element_dot_product() {
        let a = vec![3.0_f32];
        let b = vec![4.0_f32];
        let result = dot_product_f32_f32(&a, &b).expect("dot product");
        assert!((result - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn self_dot_product_is_norm_squared() {
        let v = vec![3.0_f32, 4.0];
        let result = dot_product_f32_f32(&v, &v).expect("dot product");
        assert!((result - 25.0).abs() < f32::EPSILON); // 3^2 + 4^2 = 25
    }

    #[test]
    fn f16_nan_propagates() {
        let stored = vec![
            f16::from_f32(1.0),
            f16::NAN,
            f16::from_f32(1.0),
            f16::from_f32(1.0),
        ];
        let query = vec![1.0_f32; 4];
        let result = dot_product_f16_f32(&stored, &query).expect("dot product");
        assert!(result.is_nan());
    }

    #[test]
    fn large_256d_matches_scalar_f32() {
        let a: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.01).sin()).collect();
        let b: Vec<f32> = (0_u16..256).map(|i| (f32::from(i) * 0.02).cos()).collect();
        let simd = dot_product_f32_f32(&a, &b).expect("dot product");
        let scalar = scalar_dot_f32(&a, &b);
        assert!(
            (simd - scalar).abs() < 1e-4,
            "256d: simd={simd}, scalar={scalar}"
        );
    }

    // ─── bd-1l4g tests end ───
}
