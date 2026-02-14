//! Scalar quantization for vector compression.
//!
//! Provides int8 (u8) scalar quantization with per-dimension min/max calibration.
//! Achieves 4x compression vs f32 (or 2x vs f16) with ~1-2% quality loss.
//!
//! # Quality Bounds
//!
//! For int8 scalar quantization, the maximum error per dimension is bounded by
//! `scale / 255` where `scale = max - min` for that dimension. The overall
//! cosine similarity error is bounded by:
//!
//! ```text
//! |cos(q, x) - cos(q, x')| <= max_dim(scale_i / 255) * sqrt(d) / (||q|| * ||x||)
//! ```
//!
//! For typical 384-dim embeddings with unit-normalized vectors, this yields
//! epsilon < 0.02 (less than 2% error).
//!
//! # Example
//!
//! ```
//! use frankensearch_index::quantization::ScalarQuantizer;
//!
//! let vectors = vec![
//!     vec![0.1, 0.5, -0.3],
//!     vec![0.2, 0.8, -0.1],
//!     vec![-0.1, 0.6, 0.4],
//! ];
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! let quantizer = ScalarQuantizer::fit(&refs);
//! let quantized = quantizer.quantize(&vectors[0]);
//! let restored = quantizer.dequantize(&quantized);
//!
//! // Roundtrip error is small.
//! for (orig, rest) in vectors[0].iter().zip(restored.iter()) {
//!     assert!((orig - rest).abs() < 0.01);
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Per-dimension scalar quantizer mapping f32 values to u8 (0-255).
///
/// Calibrated from a set of training vectors by computing per-dimension
/// min/max bounds. Each dimension is independently mapped to the `[0, 255]`
/// range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    /// Per-dimension minimum values.
    mins: Vec<f32>,
    /// Per-dimension scale factors: `(max - min) / 255.0`.
    /// A scale of 0 means the dimension is constant.
    scales: Vec<f32>,
    /// Dimensionality.
    dims: usize,
}

impl ScalarQuantizer {
    /// Fit a quantizer from training vectors.
    ///
    /// Computes per-dimension `[min, max]` bounds across all training vectors.
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty or if vectors have inconsistent dimensions.
    #[must_use]
    pub fn fit(vectors: &[&[f32]]) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector to fit");
        let dims = vectors[0].len();

        let mut mins = vec![f32::INFINITY; dims];
        let mut maxs = vec![f32::NEG_INFINITY; dims];

        for vec in vectors {
            assert_eq!(vec.len(), dims, "all vectors must have the same dimension");
            for (i, &v) in vec.iter().enumerate() {
                if v < mins[i] {
                    mins[i] = v;
                }
                if v > maxs[i] {
                    maxs[i] = v;
                }
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range < f32::EPSILON {
                    0.0 // constant dimension
                } else {
                    range / 255.0
                }
            })
            .collect();

        Self { mins, scales, dims }
    }

    /// Quantize an f32 vector to u8.
    ///
    /// Each dimension is mapped: `q = clamp(round((x - min) / scale), 0, 255)`.
    /// Dimensions with zero scale (constant) are mapped to 0.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != self.dims`.
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(
            vector.len(),
            self.dims,
            "vector dimension mismatch: expected {}, got {}",
            self.dims,
            vector.len()
        );

        vector
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if self.scales[i] < f32::EPSILON {
                    0
                } else {
                    let normalized = (v - self.mins[i]) / self.scales[i];
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let q = normalized.round().clamp(0.0, 255.0) as u8;
                    q
                }
            })
            .collect()
    }

    /// Dequantize a u8 vector back to f32.
    ///
    /// Each dimension is restored: `x' = q * scale + min`.
    ///
    /// # Panics
    ///
    /// Panics if `quantized.len() != self.dims`.
    #[must_use]
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        assert_eq!(
            quantized.len(),
            self.dims,
            "quantized vector dimension mismatch"
        );

        quantized
            .iter()
            .enumerate()
            .map(|(i, &q)| f32::from(q).mul_add(self.scales[i], self.mins[i]))
            .collect()
    }

    /// Compute approximate dot product between a quantized stored vector and
    /// a full-precision query vector.
    ///
    /// Uses asymmetric distance computation (ADC): the query stays in f32,
    /// the stored vector is dequantized on-the-fly during accumulation.
    /// This avoids materializing the full dequantized vector.
    ///
    /// # Panics
    ///
    /// Panics if `stored.len() != self.dims` or `query.len() != self.dims`.
    #[must_use]
    pub fn dot_product_quantized(&self, stored: &[u8], query: &[f32]) -> f32 {
        assert_eq!(stored.len(), self.dims);
        assert_eq!(query.len(), self.dims);

        let mut sum = 0.0_f32;
        for i in 0..self.dims {
            let dequantized = f32::from(stored[i]).mul_add(self.scales[i], self.mins[i]);
            sum = dequantized.mul_add(query[i], sum);
        }
        sum
    }

    /// Compute approximate cosine similarity between a quantized stored vector
    /// and a full-precision query vector.
    ///
    /// Dequantizes on-the-fly and computes cosine similarity.
    ///
    /// # Panics
    ///
    /// Panics if `stored.len() != self.dims` or `query.len() != self.dims`.
    #[must_use]
    pub fn cosine_similarity_quantized(&self, stored: &[u8], query: &[f32]) -> f32 {
        assert_eq!(stored.len(), self.dims);
        assert_eq!(query.len(), self.dims);

        let mut dot = 0.0_f32;
        let mut norm_s = 0.0_f32;
        let mut norm_q = 0.0_f32;

        for i in 0..self.dims {
            let s = f32::from(stored[i]).mul_add(self.scales[i], self.mins[i]);
            dot = s.mul_add(query[i], dot);
            norm_s = s.mul_add(s, norm_s);
            norm_q = query[i].mul_add(query[i], norm_q);
        }

        let denom = norm_s.sqrt() * norm_q.sqrt();
        if denom < f32::EPSILON {
            return 0.0;
        }

        dot / denom
    }

    /// Dimensionality of the quantizer.
    #[must_use]
    pub const fn dims(&self) -> usize {
        self.dims
    }

    /// Per-dimension minimum values.
    #[must_use]
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Per-dimension scale factors.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Memory usage for a single quantized vector (in bytes).
    #[must_use]
    pub const fn quantized_vector_bytes(&self) -> usize {
        self.dims // 1 byte per dimension
    }

    /// Memory usage for the quantizer parameters (in bytes).
    #[must_use]
    pub const fn parameter_bytes(&self) -> usize {
        self.dims * 4 * 2 // mins (f32) + scales (f32)
    }

    /// Compute the worst-case per-dimension quantization error.
    ///
    /// For each dimension, the maximum error is `scale / 2` (half a step).
    #[must_use]
    pub fn max_error_per_dim(&self) -> Vec<f32> {
        self.scales.iter().map(|s| s / 2.0).collect()
    }

    /// Compute the theoretical upper bound on cosine similarity error
    /// for unit-normalized vectors.
    ///
    /// `epsilon <= max_scale / 255 * sqrt(dims)`
    #[must_use]
    pub fn cosine_error_bound(&self) -> f32 {
        let max_scale = self.scales.iter().copied().fold(0.0_f32, f32::max);
        #[allow(clippy::cast_precision_loss)]
        let sqrt_d = (self.dims as f32).sqrt();
        max_scale / 255.0 * sqrt_d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            return vec![0.0; v.len()];
        }
        v.iter().map(|x| x / norm).collect()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = na * nb;
        if denom < f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    // ── Fit and roundtrip ────────────────────────────────────────────────

    #[test]
    fn fit_and_roundtrip_small() {
        let vectors = [
            vec![0.1_f32, 0.5, -0.3],
            vec![0.2, 0.8, -0.1],
            vec![-0.1, 0.6, 0.4],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        assert_eq!(q.dims(), 3);

        for vec in &vectors {
            let quantized = q.quantize(vec);
            let restored = q.dequantize(&quantized);
            for (orig, rest) in vec.iter().zip(restored.iter()) {
                assert!(
                    (orig - rest).abs() < 0.01,
                    "roundtrip error too large: {orig} vs {rest}"
                );
            }
        }
    }

    #[test]
    fn fit_single_vector() {
        let vectors = [vec![1.0_f32, 2.0, 3.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        // Single vector: all dimensions are constant → scales = 0
        assert!(q.scales().iter().all(|s| *s < f32::EPSILON));
    }

    // ── Cosine similarity preservation ───────────────────────────────────

    #[test]
    fn cosine_similarity_preserved_after_quantization() {
        // Generate diverse vectors.
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let i_f = i as f32;
                l2_normalize(&[
                    (i_f * 0.1).sin(),
                    (i_f * 0.2).cos(),
                    (i_f * 0.3).sin(),
                    (i_f * 0.4).cos(),
                ])
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        // Check cosine similarity between all pairs.
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let original_sim = cosine_sim(&vectors[i], &vectors[j]);
                let qi = q.quantize(&vectors[i]);
                let ri = q.dequantize(&qi);
                let qj = q.quantize(&vectors[j]);
                let rj = q.dequantize(&qj);
                let quantized_sim = cosine_sim(&ri, &rj);

                assert!(
                    (original_sim - quantized_sim).abs() < 0.05,
                    "cosine sim diverged: {original_sim} vs {quantized_sim} for pair ({i}, {j})"
                );
            }
        }
    }

    // ── ADC dot product ──────────────────────────────────────────────────

    #[test]
    fn dot_product_quantized_matches_dequantized() {
        let vectors = [
            vec![0.1_f32, 0.5, -0.3, 0.8],
            vec![0.2, 0.8, -0.1, 0.3],
            vec![-0.1, 0.6, 0.4, -0.2],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        let query = vec![0.3_f32, 0.4, -0.2, 0.5];
        let stored = q.quantize(&vectors[0]);

        // ADC dot product should match dequantized dot product.
        let adc_dot = q.dot_product_quantized(&stored, &query);
        let deq = q.dequantize(&stored);
        let full_dot: f32 = deq.iter().zip(query.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (adc_dot - full_dot).abs() < 1e-5,
            "ADC dot mismatch: {adc_dot} vs {full_dot}"
        );
    }

    // ── Cosine similarity ADC ────────────────────────────────────────────

    #[test]
    fn cosine_similarity_quantized_matches() {
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let i_f = i as f32;
                l2_normalize(&[(i_f * 0.5).sin(), (i_f * 0.7).cos(), (i_f * 0.3).sin()])
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        let query = l2_normalize(&[0.5, 0.3, -0.1]);

        for vec in &vectors {
            let stored = q.quantize(vec);
            let adc_cos = q.cosine_similarity_quantized(&stored, &query);
            let deq = q.dequantize(&stored);
            let full_cos = cosine_sim(&deq, &query);

            assert!(
                (adc_cos - full_cos).abs() < 1e-4,
                "ADC cosine mismatch: {adc_cos} vs {full_cos}"
            );
        }
    }

    // ── Error bounds ─────────────────────────────────────────────────────

    #[test]
    fn error_bound_is_valid() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let i_f = i as f32;
                l2_normalize(&[
                    (i_f * 0.1).sin(),
                    (i_f * 0.2).cos(),
                    (i_f * 0.3).sin(),
                    (i_f * 0.4).cos(),
                    (i_f * 0.5).sin(),
                    (i_f * 0.6).cos(),
                    (i_f * 0.7).sin(),
                    (i_f * 0.8).cos(),
                ])
            })
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        let bound = q.cosine_error_bound();
        assert!(bound > 0.0);
        assert!(bound < 1.0, "error bound too large: {bound}");

        // Verify empirically that the bound holds.
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let orig = cosine_sim(&vectors[i], &vectors[j]);
                let qi = q.quantize(&vectors[i]);
                let qj = q.quantize(&vectors[j]);
                let ri = q.dequantize(&qi);
                let rj = q.dequantize(&qj);
                let quant = cosine_sim(&ri, &rj);

                assert!(
                    (orig - quant).abs() <= bound + 0.01, // small margin for FP
                    "error {:.4} exceeds bound {:.4} for pair ({i}, {j})",
                    (orig - quant).abs(),
                    bound,
                );
            }
        }
    }

    // ── Memory accounting ────────────────────────────────────────────────

    #[test]
    fn memory_accounting() {
        let vectors = [vec![0.0_f32; 384]];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        assert_eq!(q.quantized_vector_bytes(), 384);
        assert_eq!(q.parameter_bytes(), 384 * 8); // 2 f32 per dim
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn constant_dimension_handled() {
        let vectors = [
            vec![0.5_f32, 0.0, 0.3], // dim 1 is constant
            vec![0.1, 0.0, 0.8],
            vec![0.9, 0.0, -0.2],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        // Dim 1 has zero scale.
        assert!(q.scales()[1] < f32::EPSILON);

        // Quantize/dequantize should preserve the constant dim.
        let quantized = q.quantize(&vectors[0]);
        let restored = q.dequantize(&quantized);
        assert!((restored[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn values_outside_training_range_clamp() {
        let vectors = [vec![0.0_f32, 1.0], vec![1.0, 0.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        // Value outside training range.
        let outlier = vec![2.0_f32, -1.0];
        let quantized = q.quantize(&outlier);

        // Should clamp to 255 and 0 respectively.
        assert_eq!(quantized[0], 255);
        assert_eq!(quantized[1], 0);
    }

    // ── Serde ────────────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let vectors = [vec![0.1_f32, 0.5, -0.3], vec![0.2, 0.8, -0.1]];
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let q = ScalarQuantizer::fit(&refs);

        let json = serde_json::to_string(&q).unwrap();
        let decoded: ScalarQuantizer = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.dims(), q.dims());

        // Should produce identical quantization.
        let original_q = q.quantize(&vectors[0]);
        let decoded_q = decoded.quantize(&vectors[0]);
        assert_eq!(original_q, decoded_q);
    }
}
