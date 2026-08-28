//! Pure-Rust cross-encoder reranker backed by frankentorch (no ONNX / no `ort`).
//!
//! Reimplements the `cross-encoder/ms-marco-MiniLM-L6-v2` `BertForSequenceClassification`
//! forward pass (6 layers, hidden 384, 12 heads, exact GELU, `LayerNorm` eps 1e-12,
//! `[CLS]` pooler + classifier, `sigmoid(logit)`) on frankentorch tensors, matching the
//! ONNX dynamic-quant scheme: an **f32 substrate** (embeddings, `LayerNorm`, softmax,
//! GELU, tanh) with **int8 Linear matmuls** (bd-1nl13.10/.15). Every Linear (attention
//! QKV/output, FFN, pooler, classifier) is statically int8-quantized per output channel
//! at load; its activation is dynamically int8-quantized per row at forward
//! (`tensor_linear_int8_dynamic`). Validated against the numpy/ONNX reference
//! (bd-1nl13.2/.3): the reference ranking is preserved.
//!
//! Embedding lookups go through `tensor_index_select` rather than `tensor_embedding`:
//! `index_select` preserves the weight dtype (f32 in/f32 out, frankentorch-40i), whereas
//! `tensor_embedding`'s custom gather still materialises f64. The two are semantically
//! identical here (no `padding_idx`). `LayerNorm` hits frankentorch's f32 fused no-grad
//! fast path.
//!
//! The only reranker backend (ort/ONNX was removed in bd-1nl13.6); feature-gated
//! behind `native`.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ft_api::{FrankenTorchSession, quantize_per_output_channel_i8};
use ft_autograd::TensorNodeId;
use ft_core::{DType, Device, ExecutionMode, TensorMeta};
use tokenizers::Tokenizer;
use wide::f32x8;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{RerankDocument, RerankScore, SyncRerank};

const H: usize = 384;
const RERANKER_LAYERS: usize = 6;
const NH: usize = 12;
const HD: usize = H / NH; // 32
const INTER: usize = 4 * H; // 1536 — FFN intermediate width
const EPS: f64 = 1.0e-12;
const EPS_F32: f32 = 1.0e-12;
const ATTN_SCALE_F32: f32 = 0.176_776_69;
const CLS_Q_CACHE_MIN_SEQ: usize = 256;
pub(crate) const DEFAULT_MAX_LENGTH: usize = 512;
/// Token budget per batched forward. Documents are reranked in chunks whose total
/// token count stays under this cap so the chunk's attention intermediates (which
/// co-exist on the tape until the per-chunk truncate) stay memory-bounded, while
/// still giving the int8 GEMM a large enough row count to amortize weight loads.
/// A single document longer than this is still processed alone (one-doc chunk).
const MAX_BATCH_TOKENS: usize = 2048;
/// Max per-document sequence length that the tape-free raw forward path handles.
/// The raw path's attention is itself a cache-blocking `gemm` bmm now (see
/// [`fused_attention`]), so it has no per-query disadvantage at long sequences —
/// it's strictly the tape path minus the per-op tape overhead, plus the CLS-only
/// final layer. We therefore route everything up to the default max length through
/// it; only configurations with a larger `max_length` fall back to the tape bmm
/// path. (Was 384, the crossover of the OLD per-query fused attention vs the tape
/// bmm — obsolete since the raw attention became a gemm.)
const FUSED_ATTN_MAX_SEQ: usize = DEFAULT_MAX_LENGTH;
const MODEL_NAME: &str = "ms-marco-minilm-l-6-v2";
pub(crate) const SAFETENSORS_PRIMARY: &str = "model_f32.safetensors";
pub(crate) const SAFETENSORS_FALLBACK: &str = "model.safetensors";
pub(crate) const TOKENIZER_JSON: &str = "tokenizer.json";

fn rerank_err(ctx: &str, e: impl std::fmt::Display) -> SearchError {
    SearchError::RerankFailed {
        model: MODEL_NAME.to_owned(),
        source: format!("{ctx}: {e}").into(),
    }
}

fn index_to_i64(index: usize, ctx: &str) -> SearchResult<i64> {
    i64::try_from(index).map_err(|_| rerank_err(ctx, format!("index {index} exceeds i64::MAX")))
}

/// In-place fused scale + numerically-stable softmax for one attention-score row.
/// Computes `softmax(scale · row)` over `row` (one head's query position), using
/// `wide`'s 8-wide polynomial `exp` (~1-2 ULP) instead of scalar libm `expf`.
/// `scale > 0`, so the argmax (hence the stabilising max-subtraction) is
/// unchanged by the scale: `exp(scale·(x − max)) == exp(scale·x)/exp(scale·max)`.
fn softmax_row_fused(row: &mut [f32], scale: f32) {
    let n = row.len();
    let mut max_raw = f32::NEG_INFINITY;
    for &x in row.iter() {
        max_raw = max_raw.max(x);
    }
    let max_v = f32x8::splat(max_raw);
    let scale_v = f32x8::splat(scale);
    let mut sum_v = f32x8::splat(0.0);
    let mut i = 0;
    // Issue 4 independent exps per iteration so the core overlaps their latency (the
    // exp is a long-latency polynomial that starves a one-group-in-flight loop). A
    // single accumulator added in the base loop's exact order keeps sum_v
    // BIT-IDENTICAL — only the exp scheduling changes. The attention softmax is the
    // dominant growing frame (~24% of the forward at seq 512); this pays a clean
    // ~3.9% there (bench softmax_exp_ilp), where it matters, and is within noise at
    // shorter rows (where softmax is a small fraction). A 4-accumulator variant that
    // breaks the sum chain wins LESS, confirming the bottleneck is exp latency (not
    // the reduction), so the bit-identical single-accumulator form is also the fastest.
    while i + 32 <= n {
        let e0 = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        let e1 = ((f32x8_from_slice(&row[i + 8..i + 16]) - max_v) * scale_v).exp();
        let e2 = ((f32x8_from_slice(&row[i + 16..i + 24]) - max_v) * scale_v).exp();
        let e3 = ((f32x8_from_slice(&row[i + 24..i + 32]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e0.to_array());
        row[i + 8..i + 16].copy_from_slice(&e1.to_array());
        row[i + 16..i + 24].copy_from_slice(&e2.to_array());
        row[i + 24..i + 32].copy_from_slice(&e3.to_array());
        sum_v += e0;
        sum_v += e1;
        sum_v += e2;
        sum_v += e3;
        i += 32;
    }
    while i + 8 <= n {
        let e = ((f32x8_from_slice(&row[i..i + 8]) - max_v) * scale_v).exp();
        row[i..i + 8].copy_from_slice(&e.to_array());
        sum_v += e;
        i += 8;
    }
    let mut sum: f32 = sum_v.to_array().iter().sum();
    while i < n {
        let e = ((row[i] - max_raw) * scale).exp();
        row[i] = e;
        sum += e;
        i += 1;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// Fused scale + softmax over the last dim of the attention scores `[rows, n]`
/// (row-major), returning the same buffer/layout. Replaces the separate
/// `mul_scalar` + `tensor_softmax` tape ops with one vectorized pass.
///
/// Profiling showed the attention softmax is the dominant *growing* f32 frame —
/// ~24% of the per-doc forward wall-clock at seq 512 (12·S² scalar `expf` calls,
/// which ONNX/MLAS fuses + vectorizes). Softmax normalisation makes the ~1e-6
/// relative exp error immaterial to the ranking (validated against the
/// numpy/ONNX reference: ranking + logit tolerance unchanged). Rows are
/// independent, so they parallelize across the forward's ambient rayon pool —
/// matching the multicore the stock softmax kernel had. This stays intra-forward
/// (the doc loop is sequential), so the nested-rayon + `Mutex` deadlock class
/// remains closed by construction.
fn fast_softmax_inplace(data: &mut [f32], rows: usize, n: usize, scale: f32) {
    debug_assert_eq!(data.len(), rows * n);
    if n == 0 {
        return;
    }
    // Parallelize only when there is enough work to amortise the fan-out and a
    // pool is actually available; otherwise stay serial (small seq / 1 thread).
    if rows >= 8 && rows * n >= 8192 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        data.par_chunks_exact_mut(n)
            .for_each(|row| softmax_row_fused(row, scale));
    } else {
        data.chunks_exact_mut(n)
            .for_each(|row| softmax_row_fused(row, scale));
    }
}

/// Exact-form GELU `0.5·x·(1 + erf(x/√2))` for one f32x8 lane group, with `erf`
/// from the Abramowitz–Stegun 7.1.26 rational×exp approximation (max abs error
/// 1.5e-7 — at the f32 precision floor, so indistinguishable from libm `erff` for
/// ranking). `erf` is odd, so it is evaluated on `|z|` and the sign is reapplied.
#[inline]
fn gelu_vec8(x: f32x8) -> f32x8 {
    const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let one = f32x8::splat(1.0);
    let z = x * f32x8::splat(C);
    let az = z.abs();
    let t = one / (one + f32x8::splat(0.327_591_1) * az);
    // erf poly = t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5))))  (A–S 7.1.26)
    let a1 = f32x8::splat(0.254_829_6);
    let a2 = f32x8::splat(-0.284_496_73);
    let a3 = f32x8::splat(1.421_413_7);
    let a4 = f32x8::splat(-1.453_152);
    let a5 = f32x8::splat(1.061_405_4);
    let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let erf_abs = one - poly * (-(z * z)).exp();
    let erf = erf_abs.copysign(z);
    f32x8::splat(0.5) * x * (one + erf)
}

/// Scalar GELU matching [`gelu_vec8`] (same A–S erf) for the < 8-element tail.
#[inline]
fn gelu_scalar(x: f32) -> f32 {
    const C: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let z = x * C;
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * az);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_73 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = (1.0 - poly * (-(z * z)).exp()).copysign(z);
    0.5 * x * (1.0 + erf)
}

/// Vectorized exact-GELU over a flat activation buffer, in place. Replaces the
/// `tensor_gelu` tape op (scalar libm `erff`) — profiling put GELU at ~10-14% of
/// the per-doc forward (a wide `[m, 1536]` elementwise transcendental). GELU is
/// elementwise so chunks are independent and parallelize across the forward's
/// rayon pool. A–S erf keeps the result within ~1e-7 of exact, so the ranking is
/// unchanged (validated against the numpy/ONNX reference).
fn fast_gelu_inplace(data: &mut [f32]) {
    let process = |chunk: &mut [f32]| {
        let n = chunk.len();
        let mut i = 0;
        // Process 4 independent f32x8 lane groups per iteration. GELU is a pure
        // elementwise map (no cross-lane reduction), so the four gelu_vec8 chains
        // are independent — issuing them back-to-back lets the core overlap their
        // latency (the erf Horner poly + exp, high-latency but pipelined) instead of
        // stalling one group at a time. Byte-identical to a one-group loop (same op
        // at the same position); measured ~4-5% faster on the in-cache FFN GELU
        // widths, fading toward memory-bound at multi-MB buffers (bench `gelu_ilp`).
        while i + 32 <= n {
            let g0 = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
            let g1 = gelu_vec8(f32x8_from_slice(&chunk[i + 8..i + 16]));
            let g2 = gelu_vec8(f32x8_from_slice(&chunk[i + 16..i + 24]));
            let g3 = gelu_vec8(f32x8_from_slice(&chunk[i + 24..i + 32]));
            chunk[i..i + 8].copy_from_slice(&g0.to_array());
            chunk[i + 8..i + 16].copy_from_slice(&g1.to_array());
            chunk[i + 16..i + 24].copy_from_slice(&g2.to_array());
            chunk[i + 24..i + 32].copy_from_slice(&g3.to_array());
            i += 32;
        }
        while i + 8 <= n {
            let g = gelu_vec8(f32x8_from_slice(&chunk[i..i + 8]));
            chunk[i..i + 8].copy_from_slice(&g.to_array());
            i += 8;
        }
        while i < n {
            chunk[i] = gelu_scalar(chunk[i]);
            i += 1;
        }
    };
    if data.len() >= 8192 && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        // Elementwise, so any chunking is correct; keep chunks a multiple of 8 so
        // only the final chunk has a scalar tail.
        data.par_chunks_mut(2048).for_each(process);
    } else {
        process(data);
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
    debug_assert_eq!(q.len(), HD);
    debug_assert_eq!(k.len(), HD);
    (f32x8_from_slice(&q[0..]) * f32x8_from_slice(&k[0..])
        + f32x8_from_slice(&q[8..]) * f32x8_from_slice(&k[8..])
        + f32x8_from_slice(&q[16..]) * f32x8_from_slice(&k[16..])
        + f32x8_from_slice(&q[24..]) * f32x8_from_slice(&k[24..]))
    .reduce_add()
}

#[inline]
fn q_lanes(q: &[f32]) -> [f32x8; 4] {
    debug_assert_eq!(q.len(), HD);
    [
        f32x8_from_slice(&q[0..]),
        f32x8_from_slice(&q[8..]),
        f32x8_from_slice(&q[16..]),
        f32x8_from_slice(&q[24..]),
    ]
}

#[inline]
fn dot_hd_q_lanes(q: [f32x8; 4], k: &[f32]) -> f32 {
    debug_assert_eq!(k.len(), HD);
    (q[0] * f32x8_from_slice(&k[0..])
        + q[1] * f32x8_from_slice(&k[8..])
        + q[2] * f32x8_from_slice(&k[16..])
        + q[3] * f32x8_from_slice(&k[24..]))
    .reduce_add()
}

fn weighted_value_sum_hd(qkv: &[f32], probs: &[f32], s_len: usize, head: usize, out: &mut [f32]) {
    debug_assert_eq!(probs.len(), s_len);
    debug_assert_eq!(out.len(), HD);
    const STRIDE: usize = 3 * H;
    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);
    for (j, &prob) in probs.iter().enumerate() {
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

/// Reusable per-document attention scratch. Holds the head-major repack operands and
/// the two `bmm` output buffers so a whole forward's worth of attention calls (every
/// document × every layer) reuse one set of allocations instead of allocating —
/// and zero-filling — fresh `Vec`s each call. The score buffer dominates (`NH·S²`),
/// so eliding its per-call realloc is the bulk of the win. Buffers only ever grow
/// (`ensure`), and every element each `bmm`/repack touches is overwritten, so reuse
/// is bit-identical to fresh allocation.
#[derive(Default)]
struct AttnScratch {
    q_hm: Vec<f32>,
    kt: Vec<f32>,
    v_hm: Vec<f32>,
    scores: Vec<f32>,
    ctx_hm: Vec<f32>,
}

impl AttnScratch {
    /// Grow (never shrink) every buffer to hold one document's full self-attention at
    /// sequence length `s_len`. The CLS-only path uses strict prefixes of these.
    fn ensure(&mut self, s_len: usize) {
        let hm = s_len * H; // NH * s_len * HD == s_len * H
        let sc = NH * s_len * s_len;
        if self.q_hm.len() < hm {
            self.q_hm.resize(hm, 0.0);
        }
        if self.kt.len() < hm {
            self.kt.resize(hm, 0.0);
        }
        if self.v_hm.len() < hm {
            self.v_hm.resize(hm, 0.0);
        }
        if self.scores.len() < sc {
            self.scores.resize(sc, 0.0);
        }
        if self.ctx_hm.len() < hm {
            self.ctx_hm.resize(hm, 0.0);
        }
    }
}

/// Multi-head self-attention for one document, writing the `[s_len, H]` context into
/// `out`. `qkv` is the fused QKV linear output `[s_len, 3H]` (Q at column 0, K at
/// column H, V at column 2H).
///
/// The per-query `f32x8` kernel was cheap to launch (no tape, no per-head gemm) but
/// runs the S² inner loop one dot at a time, which dominates at seq ≥ 256 (it was
/// ~9× over the FLOP floor — a register-blocked GEMM does the same MACs far faster).
/// So repack each head's Q/K/V into the head-major layout the batched `bmm` kernel
/// (`gemm::sgemm`, cache-blocking) wants, do `QKᵀ` and ·V as two batched GEMMs with a
/// vectorized softmax between, and transpose the context back. The repacks and both
/// GEMM outputs land in the caller's reused `scratch`, so the hot loop does not
/// allocate. The `bmm` kernel parallelizes across the NH head-batches internally and
/// stays intra-forward, so the nested-rayon + `Mutex` deadlock class remains closed.
/// Bit-exact to the bmm tape path (within f32 reassociation) — validated by parity +
/// `forward_batch_matches_per_doc`.
fn fused_attention(
    scratch: &mut AttnScratch,
    qkv: &[f32],
    s_len: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(qkv.len(), s_len * 3 * H);
    debug_assert_eq!(out.len(), s_len * H);
    if s_len == 0 {
        return;
    }
    const STRIDE: usize = 3 * H;
    scratch.ensure(s_len);
    let hm = s_len * H;
    let sc = NH * s_len * s_len;
    let AttnScratch {
        q_hm,
        kt,
        v_hm,
        scores,
        ctx_hm,
    } = scratch;
    let (q_hm, kt, v_hm) = (&mut q_hm[..hm], &mut kt[..hm], &mut v_hm[..hm]);
    let scores = &mut scores[..sc];
    let ctx_hm = &mut ctx_hm[..hm];
    // Repack into head-major operands: Q/V as [NH, S, HD], K transposed to [NH, HD, S]
    // so QKᵀ is a plain `bmm(Q, Kᵀ)`. Q/V are contiguous head-major copies.
    for j in 0..s_len {
        let base = j * STRIDE;
        for h in 0..NH {
            let hmj = (h * s_len + j) * HD;
            q_hm[hmj..hmj + HD].copy_from_slice(&qkv[base + h * HD..base + h * HD + HD]);
            v_hm[hmj..hmj + HD]
                .copy_from_slice(&qkv[base + 2 * H + h * HD..base + 2 * H + h * HD + HD]);
        }
    }
    // Kᵀ: each output row `kt[h, d, :]` is written sequentially (contiguous stores)
    // from the strided source column, instead of strided stores into kt.
    for h in 0..NH {
        for d in 0..HD {
            let col = H + h * HD + d;
            let row = &mut kt[h * HD * s_len + d * s_len..h * HD * s_len + d * s_len + s_len];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = qkv[j * STRIDE + col];
            }
        }
    }
    // scores[NH, S, S] = Q @ Kᵀ, then in-place fused-scale softmax over the last dim.
    let qm = TensorMeta::from_shape(vec![NH, s_len, HD], DType::F32, Device::Cpu);
    let km = TensorMeta::from_shape(vec![NH, HD, s_len], DType::F32, Device::Cpu);
    ft_api::bmm_tensor_contiguous_f32_into(q_hm, kt, &qm, &km, scores)
        .expect("attn QKᵀ bmm: shapes are internally consistent");
    fast_softmax_inplace(scores, NH * s_len, s_len, scale);
    // ctx_hm[NH, S, HD] = scores @ V.
    let sm = TensorMeta::from_shape(vec![NH, s_len, s_len], DType::F32, Device::Cpu);
    let vm = TensorMeta::from_shape(vec![NH, s_len, HD], DType::F32, Device::Cpu);
    ft_api::bmm_tensor_contiguous_f32_into(scores, v_hm, &sm, &vm, ctx_hm)
        .expect("attn ·V bmm: shapes are internally consistent");
    // Transpose the head-major context back to token-major [S, H] into `out`.
    for h in 0..NH {
        for i in 0..s_len {
            let src = (h * s_len + i) * HD;
            out[i * H + h * HD..i * H + h * HD + HD].copy_from_slice(&ctx_hm[src..src + HD]);
        }
    }
}

/// Self-attention for the `[CLS]` query ONLY (token 0), writing its `[H]` context into
/// `out`.
///
/// The reranker's logit is read solely from `[CLS]` of the final layer, so the last
/// encoder layer never needs any other token's attention output. `[CLS]` still attends
/// to every key, but the operation is rank-1 per head: one `[HD]` query row scores all
/// keys, then one probability row weights the values. Running that directly over the
/// fused QKV buffer avoids the old head-major K/V repacks plus two tiny `bmm` launches
/// whose `m = 1` shape is below the GEMM kernel's useful blocking regime. The softmax
/// kernel and output layout stay the same; only the CLS-only final layer uses this path.
fn fused_attention_cls(
    scratch: &mut AttnScratch,
    qkv: &[f32],
    s_len: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(qkv.len(), s_len * 3 * H);
    debug_assert_eq!(out.len(), H);
    if s_len == 0 {
        out.fill(0.0);
        return;
    }
    const STRIDE: usize = 3 * H;
    scratch.ensure(s_len);
    let sc = NH * s_len;
    for h in 0..NH {
        let row = &mut scratch.scores[h * s_len..(h + 1) * s_len];
        if s_len >= CLS_Q_CACHE_MIN_SEQ {
            let q = q_lanes(&qkv[h * HD..h * HD + HD]);
            for (j, slot) in row.iter_mut().enumerate() {
                let k_base = j * STRIDE + H + h * HD;
                *slot = dot_hd_q_lanes(q, &qkv[k_base..k_base + HD]);
            }
        } else {
            let q = &qkv[h * HD..h * HD + HD];
            for (j, slot) in row.iter_mut().enumerate() {
                let k_base = j * STRIDE + H + h * HD;
                *slot = dot_hd(q, &qkv[k_base..k_base + HD]);
            }
        }
    }
    let scores = &mut scratch.scores[..sc];
    fast_softmax_inplace(scores, NH, s_len, scale);
    for h in 0..NH {
        let row = &scores[h * s_len..(h + 1) * s_len];
        weighted_value_sum_hd(qkv, row, s_len, h, &mut out[h * HD..h * HD + HD]);
    }
}

/// A Linear layer's weights, statically quantized to int8 with per-output-channel
/// f32 scales, plus its f32 bias. The three buffers are `Arc`-shared so the parsed
/// weights are stored once and cloned cheaply into every pooled session.
#[derive(Clone)]
struct QLinear {
    /// Int8 weights. Row-major `[out, in]` when `packed` is false; NR=4
    /// panel-interleaved (from `pack_int8_weights_nr4`) when `packed` is true.
    w_i8: Arc<Vec<i8>>,
    /// Per-output-channel f32 dequantization scales (len `out`).
    w_scales: Arc<Vec<f32>>,
    /// f32 bias (len `out`).
    bias: Arc<Vec<f32>>,
    out: usize,
    in_: usize,
    /// Whether `w_i8` is NR=4-packed (pre-packed SDOT kernel) vs row-major.
    packed: bool,
}

/// Owns the frankentorch session and the loaded weight tensors. Mutated during the
/// forward pass, so it lives behind a `Mutex` in `NativeReranker`.
pub(crate) struct Model {
    s: FrankenTorchSession,
    /// f32 leaf nodes for the non-Linear parameters (`word/position/token_type`
    /// embeddings and every `LayerNorm` weight/bias) — these stay in f32.
    w: HashMap<String, TensorNodeId>,
    /// int8-quantized Linear weights (attention QKV/output, FFN, pooler,
    /// classifier), keyed by the layer prefix (the weight name minus `.weight`).
    qw: HashMap<String, QLinear>,
    /// Raw f32 values for the `LayerNorm` weight/bias parameters (same data as the
    /// `w` leaves, shared via `Arc`), so the tape-free fused-layer path can call the
    /// `add_layer_norm` kernel directly without round-tripping through the session.
    raw_params: HashMap<String, Arc<Vec<f32>>>,
    /// Contiguous encoder-layer count attested by the loaded weight set.
    encoder_layers: usize,
    /// Autograd tape node count captured right after the persistent weights are
    /// loaded. Each forward pass truncates the tape back to this boundary to free
    /// that pass's intermediate activations, so the session does not grow
    /// unbounded across many rerank calls (a single long-doc forward can allocate
    /// ~25 MB attention tensors per layer; without truncation they would
    /// accumulate for the life of the process).
    weights_boundary: usize,
}

impl Model {
    pub(crate) const fn encoder_layers(&self) -> usize {
        self.encoder_layers
    }

    fn g(&self, name: &str) -> SearchResult<TensorNodeId> {
        self.w
            .get(name)
            .copied()
            .ok_or_else(|| rerank_err("weights", format!("missing weight tensor {name}")))
    }

    /// `y = layer(x)` via the int8 kernel directly on raw f32 buffers (no tape
    /// node) — the tape-free counterpart of [`Self::linear`]. Output width is the
    /// `QLinear`'s `out`; input width is its `in_`.
    fn linear_raw(&self, x: &[f32], m: usize, prefix: &str) -> SearchResult<Vec<f32>> {
        let q = self
            .qw
            .get(prefix)
            .ok_or_else(|| rerank_err("linear_raw", format!("missing linear weights {prefix}")))?;
        debug_assert_eq!(x.len(), m * q.in_);
        let y = if q.packed {
            ft_api::linear_int8_dynamic_prepacked_f32(
                x,
                m,
                q.in_,
                &q.w_i8,
                &q.w_scales,
                q.out,
                Some(&q.bias),
            )
        } else {
            ft_api::linear_int8_dynamic_f32(x, m, q.in_, &q.w_i8, &q.w_scales, q.out, Some(&q.bias))
        };
        Ok(y)
    }

    /// `layer_norm(a + b)` on raw f32 buffers (no tape node) — the tape-free
    /// counterpart of [`Self::add_ln`].
    fn add_ln_raw(&self, a: &[f32], b: &[f32], m: usize, prefix: &str) -> SearchResult<Vec<f32>> {
        let w = self
            .raw_params
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| rerank_err("add_ln_raw", format!("missing {prefix}.weight")))?;
        let bias = self
            .raw_params
            .get(&format!("{prefix}.bias"))
            .ok_or_else(|| rerank_err("add_ln_raw", format!("missing {prefix}.bias")))?;
        Ok(ft_api::add_layer_norm_forward_f32(
            a,
            b,
            Some(w),
            Some(bias),
            m,
            H,
            EPS_F32,
        ))
    }

    /// One encoder layer entirely on raw f32 buffers — no tape nodes, no session
    /// round-trips. Calls the SAME optimized kernels the tape path uses (int8 GEMM,
    /// fused attention, add+LN, vectorized GELU), so the result is bit-identical,
    /// but the per-op tape-node creation / leaf allocation / truncation are gone.
    /// Self-attention is per document (each `[lenₙ, H]` slice) via
    /// [`fused_attention`], so this path is for chunks where every doc is short. The
    /// `scratch` is reused across documents and layers to avoid per-call allocation.
    fn encoder_layer_raw(
        &self,
        emb: &[f32],
        total: usize,
        offsets: &[usize],
        lens: &[usize],
        p: &str,
        scale: f32,
        scratch: &mut AttnScratch,
    ) -> SearchResult<Vec<f32>> {
        // Fused QKV projection (batched over all the chunk's tokens).
        // Fused QKV projection output shape: [total, 3H].
        let qkv = self.linear_raw(emb, total, &format!("{p}.attention.self.qkv"))?;
        // Per-document self-attention written straight into one re-concatenated
        // [total, H] context (no per-doc temporary).
        let mut ctx = vec![0.0f32; total * H];
        for (&off, &len) in offsets.iter().zip(lens) {
            let qkv_doc = &qkv[off * 3 * H..(off + len) * 3 * H];
            fused_attention(
                scratch,
                qkv_doc,
                len,
                scale,
                &mut ctx[off * H..(off + len) * H],
            );
        }
        let attn = self.linear_raw(&ctx, total, &format!("{p}.attention.output.dense"))?;
        let emb = self.add_ln_raw(
            emb,
            &attn,
            total,
            &format!("{p}.attention.output.LayerNorm"),
        )?;
        // FFN: [total, H] -> [total, INTER] -> GELU -> [total, H].
        let mut inter = self.linear_raw(&emb, total, &format!("{p}.intermediate.dense"))?;
        debug_assert_eq!(inter.len(), total * INTER);
        fast_gelu_inplace(&mut inter);
        let ffn = self.linear_raw(&inter, total, &format!("{p}.output.dense"))?;
        self.add_ln_raw(&emb, &ffn, total, &format!("{p}.output.LayerNorm"))
    }

    /// The FINAL encoder layer, computing ONLY each document's `[CLS]` row.
    ///
    /// The pooler reads the logit from `[CLS]` of the last layer alone, so the other
    /// tokens' layer outputs are dead. The QKV projection still runs over all tokens
    /// (CLS attends to every key, so K/V are needed in full), but attention is the
    /// CLS-query-only [`fused_attention_cls`], and the attn-out projection, residual
    /// add+LN, and the whole FFN then run on just `n_docs` rows (one `[CLS]` per doc)
    /// instead of `total`. Returns `[n_docs, H]` — exactly the pooler's input, so no
    /// post-hoc CLS gather is needed. Bit-exact to the `[CLS]` rows of
    /// [`encoder_layer_raw`] (same kernels, same reductions).
    fn encoder_layer_cls(
        &self,
        emb: &[f32],
        offsets: &[usize],
        lens: &[usize],
        total: usize,
        p: &str,
        scale: f32,
        scratch: &mut AttnScratch,
    ) -> SearchResult<Vec<f32>> {
        let n_docs = lens.len();
        // Fused QKV over all tokens (K/V needed in full for the CLS query's attention).
        // Fused QKV projection output shape: [total, 3H].
        let qkv = self.linear_raw(emb, total, &format!("{p}.attention.self.qkv"))?;
        // CLS-only self-attention per document written straight into a compact
        // [n_docs, H] context.
        let mut ctx = vec![0.0f32; n_docs * H];
        for (n, (&off, &len)) in offsets.iter().zip(lens).enumerate() {
            let qkv_doc = &qkv[off * 3 * H..(off + len) * 3 * H];
            fused_attention_cls(scratch, qkv_doc, len, scale, &mut ctx[n * H..(n + 1) * H]);
        }
        let attn = self.linear_raw(&ctx, n_docs, &format!("{p}.attention.output.dense"))?;
        // Residual is the input layer's CLS rows (the first token of each doc).
        let mut emb_cls = vec![0.0f32; n_docs * H];
        for (n, &off) in offsets.iter().enumerate() {
            emb_cls[n * H..(n + 1) * H].copy_from_slice(&emb[off * H..off * H + H]);
        }
        let emb_cls = self.add_ln_raw(
            &emb_cls,
            &attn,
            n_docs,
            &format!("{p}.attention.output.LayerNorm"),
        )?;
        // FFN on just the CLS rows: [n_docs, H] -> [n_docs, INTER] -> GELU -> [n_docs, H].
        let mut inter = self.linear_raw(&emb_cls, n_docs, &format!("{p}.intermediate.dense"))?;
        debug_assert_eq!(inter.len(), n_docs * INTER);
        fast_gelu_inplace(&mut inter);
        let ffn = self.linear_raw(&inter, n_docs, &format!("{p}.output.dense"))?;
        self.add_ln_raw(&emb_cls, &ffn, n_docs, &format!("{p}.output.LayerNorm"))
    }

    /// y = x @ Wᵀ + b via the int8 dynamic-quant kernel (weight stored row-major
    /// [out, in], `PyTorch` convention). The f32 activation `x` is dynamically
    /// quantized per-row; the weight is statically int8-quantized per-output-channel;
    /// the result is dequantized back to an f32 node.
    fn linear(&mut self, x: TensorNodeId, prefix: &str) -> SearchResult<TensorNodeId> {
        let q = self
            .qw
            .get(prefix)
            .ok_or_else(|| rerank_err("linear", format!("missing int8 linear weights {prefix}")))?;
        // Clone the Arcs + copy the dims so the `&self.qw` borrow ends before the
        // `&mut self.s` borrow below.
        let w_i8 = Arc::clone(&q.w_i8);
        let w_scales = Arc::clone(&q.w_scales);
        let bias = Arc::clone(&q.bias);
        let (out, in_, packed) = (q.out, q.in_, q.packed);
        if packed {
            self.s
                .tensor_linear_int8_dynamic_prepacked(x, &w_i8, &w_scales, out, in_, Some(&bias))
                .map_err(|e| rerank_err("linear.int8.packed", e))
        } else {
            self.s
                .tensor_linear_int8_dynamic(x, &w_i8, &w_scales, out, in_, Some(&bias))
                .map_err(|e| rerank_err("linear.int8", e))
        }
    }

    fn idx(&mut self, vals: &[i64]) -> SearchResult<TensorNodeId> {
        let f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        self.s
            .tensor_variable(f, vec![vals.len()], false)
            .map_err(|e| rerank_err("index_tensor", e))
    }

    /// Fused residual-add + `LayerNorm` `layer_norm(a + b)` (the "add & norm") in one
    /// op, so the residual sum is never materialized as its own tensor (2 per layer).
    fn add_ln(
        &mut self,
        a: TensorNodeId,
        b: TensorNodeId,
        prefix: &str,
    ) -> SearchResult<TensorNodeId> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let bias = self.g(&format!("{prefix}.bias"))?;
        self.s
            .tensor_add_layer_norm(a, b, H, Some(w), Some(bias), EPS)
            .map_err(|e| rerank_err("add_layer_norm", e))
    }

    /// Exact GELU via the vectorized [`fast_gelu`] (A–S erf), replacing the
    /// `tensor_gelu` tape op's scalar libm `erff`. Round-trips through f32 values +
    /// a fresh leaf, consistent with the int8 linear (which already returns a
    /// detached f32 leaf), so there is no live autograd graph to preserve.
    fn gelu(&mut self, inter: TensorNodeId) -> SearchResult<TensorNodeId> {
        // In-place: `inter` is a single-use intermediate (only the next FFN linear
        // reads it), so rewrite its storage rather than round-tripping through a
        // fresh leaf — kills the per-layer extract+reinsert of the wide `[m, 1536]`.
        let slice = self
            .s
            .tensor_values_f32_mut(inter)
            .map_err(|e| rerank_err("ffn.gelu_mut", e))?;
        fast_gelu_inplace(slice);
        Ok(inter)
    }

    /// [S, H] -> [NH, S, HD]
    fn heads(&mut self, x: TensorNodeId, s_len: usize) -> SearchResult<TensorNodeId> {
        let r = self
            .s
            .tensor_reshape(x, vec![s_len, NH, HD])
            .map_err(|e| rerank_err("heads.reshape", e))?;
        self.s
            .tensor_transpose(r, 0, 1)
            .map_err(|e| rerank_err("heads.transpose", e))
    }

    /// Multi-head self-attention for one sequence: given the per-token `q`/`k`/`v`
    /// projections (each `[s_len, H]`) returns the context `[s_len, H]`. Shared by
    /// the single-pair [`Self::forward`] and the batched [`Self::forward_batch`]
    /// (which calls it on each document's contiguous token slice), so the two paths
    /// compute identical attention — the batched path is parity-exact.
    ///
    /// Hybrid: for short/medium sequences the [`fused_attention`] kernel wins (it
    /// avoids the heads-transpose materialization and the per-head `gemm` launch
    /// overhead that dominate there); for long sequences the `bmm` path wins (the
    /// `gemm` crate's cache-blocking amortizes the K/V re-reads that the naive fused
    /// loop repeats per query). `FUSED_ATTN_MAX_SEQ` is the measured crossover.
    /// Short/medium-sequence attention: `qkv` is the fused QKV linear output
    /// `[s_len, 3H]`; runs the [`fused_attention`] kernel over it (reading borrowed,
    /// writing one fresh ctx leaf). Tape-path reference for the single-pair
    /// [`Self::forward`]; production goes through the tape-free `encoder_layer_raw`.
    #[cfg(test)]
    fn attn_fused(
        &mut self,
        qkv: TensorNodeId,
        s_len: usize,
        scale: f32,
    ) -> SearchResult<TensorNodeId> {
        let ctx_vals = {
            let qkv_v = self
                .s
                .tensor_values_f32_borrowed(qkv)
                .map_err(|e| rerank_err("attn.qkv_vals", e))?;
            let mut scratch = AttnScratch::default();
            let mut ctx = vec![0.0f32; s_len * H];
            fused_attention(&mut scratch, qkv_v, s_len, scale, &mut ctx);
            ctx
        };
        self.s
            .tensor_variable_f32(ctx_vals, vec![s_len, H], false)
            .map_err(|e| rerank_err("attn.ctx", e))
    }

    /// Long-sequence attention: separate `q`/`k`/`v` `[s_len, H]` through the batched
    /// f32 `bmm` with in-place fused-scale softmax (the `gemm` crate's cache-blocking
    /// wins past `FUSED_ATTN_MAX_SEQ`).
    fn attn_bmm(
        &mut self,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        s_len: usize,
        scale: f32,
    ) -> SearchResult<TensorNodeId> {
        let q = self.heads(q, s_len)?;
        let k = self.heads(k, s_len)?;
        let v = self.heads(v, s_len)?;
        let kt = self
            .s
            .tensor_transpose(k, 1, 2)
            .map_err(|e| rerank_err("attn.kt", e))?; // [NH, HD, S]
        let scores = self
            .s
            .tensor_bmm(q, kt)
            .map_err(|e| rerank_err("attn.qk", e))?;
        {
            let slice = self
                .s
                .tensor_values_f32_mut(scores)
                .map_err(|e| rerank_err("attn.softmax_mut", e))?;
            fast_softmax_inplace(slice, NH * s_len, s_len, scale);
        }
        let ctx = self
            .s
            .tensor_bmm(scores, v)
            .map_err(|e| rerank_err("attn.ctx", e))?;
        let ctx = self
            .s
            .tensor_transpose(ctx, 0, 1)
            .map_err(|e| rerank_err("attn.ctx_t", e))?;
        self.s
            .tensor_reshape(ctx, vec![s_len, H])
            .map_err(|e| rerank_err("attn.ctx_reshape", e))
    }

    /// Self-attention for one document's `[s_len, H]` activation `emb`: routes to the
    /// fused-QKV + fused-attention kernel for short/medium sequences (one int8 GEMM
    /// for Q/K/V, no transpose/per-head-launch overhead), or the separate-QKV + bmm
    /// path for long sequences. Returns the context `[s_len, H]`. Used by the
    /// single-pair [`Self::forward`] reference; production uses `encoder_layer_raw`.
    #[cfg(test)]
    fn attention(
        &mut self,
        emb: TensorNodeId,
        p: &str,
        s_len: usize,
        scale: f32,
    ) -> SearchResult<TensorNodeId> {
        if s_len <= FUSED_ATTN_MAX_SEQ {
            let qkv = self.linear(emb, &format!("{p}.attention.self.qkv"))?;
            self.attn_fused(qkv, s_len, scale)
        } else {
            let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
            let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
            let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
            self.attn_bmm(q, k, v, s_len, scale)
        }
    }

    /// Single-pair forward pass (batch = 1). Returns the raw logit. Retained as the
    /// per-document reference that [`Self::forward_batch`] is checked against
    /// (`forward_batch_matches_per_doc`); production reranking always goes through
    /// the batched path (which handles a one-document batch with negligible
    /// overhead), so this is `#[cfg(test)]`-only.
    ///
    /// Runs entirely in f32: weights are f32 leaves and every op preserves f32, so
    /// `embedding/matmul/softmax/layer_norm` all stay in the f32 kernels.
    #[cfg(test)]
    fn forward(&mut self, ids: &[i64], typ: &[i64]) -> SearchResult<f32> {
        let s_len = ids.len();
        // embeddings: word + position + token_type, then LayerNorm.
        // `index_select(weight, dim=0, indices)` is the embedding lookup; unlike
        // `tensor_embedding` it preserves the f32 weight dtype (frankentorch-40i).
        let id_t = self.idx(ids)?;
        let pos: Vec<i64> = (0..s_len)
            .map(|i| index_to_i64(i, "forward.position"))
            .collect::<SearchResult<_>>()?;
        let pos_t = self.idx(&pos)?;
        let typ_t = self.idx(typ)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let emb_wp = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        // Fuse the third-embedding add with the embedding LayerNorm.
        let mut emb = self.add_ln(emb_wp, e_typ, "bert.embeddings.LayerNorm")?;

        let scale = ATTN_SCALE_F32;
        for i in 0..self.encoder_layers {
            let p = format!("bert.encoder.layer.{i}");
            // self-attention (fused-QKV + fused kernel, or separate-QKV + bmm by len)
            let ctx = self.attention(emb, &p, s_len, scale)?;
            let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
            emb = self.add_ln(emb, attn, &format!("{p}.attention.output.LayerNorm"))?;
            // feed-forward
            let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
            let inter = self.gelu(inter)?;
            let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
            emb = self.add_ln(emb, ffn, &format!("{p}.output.LayerNorm"))?;
        }
        // pooler on [CLS] (row 0) + classifier
        let cls = self
            .s
            .tensor_narrow(emb, 0, 0, 1)
            .map_err(|e| rerank_err("pooler.narrow", e))?; // [1, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self
            .s
            .tensor_tanh(pooled)
            .map_err(|e| rerank_err("pooler.tanh", e))?;
        let logit_t = self.linear(pooled, "classifier")?; // [1, 1]
        let vals = self
            .s
            .tensor_values_f32(logit_t)
            .map_err(|e| rerank_err("classifier.values", e))?;
        let logit = vals
            .first()
            .copied()
            .ok_or_else(|| rerank_err("classifier", "empty logit output"))?;
        // Free this forward pass's intermediate tape nodes (everything created
        // after the weights), keeping the loaded parameters, so the session's
        // arena does not grow unbounded across rerank calls.
        self.s.truncate_autograd_graph(self.weights_boundary);
        Ok(logit)
    }

    /// Batched (multi-document) forward — the throughput lever. Returns one raw
    /// logit per document, in input order.
    ///
    /// Layout is **varlen**: the documents' tokens are concatenated end-to-end
    /// (NO padding, NO mask) into one `[Σlenₙ, H]` activation. Every per-token op
    /// (the int8 Linears, `LayerNorm`, GELU, residuals) runs once over the whole
    /// `Σlenₙ` rows, so each statically-quantized weight is loaded once and reused
    /// across all the documents' tokens instead of being re-streamed per document
    /// — that weight-reuse is what lifts the int8 GEMM toward peak and beats a
    /// per-document runtime on multi-doc rerank throughput. Self-attention is the
    /// only op that must stay within a document, so it runs per-doc on each
    /// document's contiguous token slice via the shared [`Self::attn_block`] — so
    /// every document gets byte-identical computation to [`Self::forward`] and the
    /// batched logits are parity-exact, not approximate.
    ///
    /// The caller chunks the batch so `Σlenₙ` stays bounded (attention
    /// intermediates for the whole chunk co-exist on the tape until the final
    /// truncate); see [`MAX_BATCH_TOKENS`].
    fn forward_batch(&mut self, batch: &[(Vec<i64>, Vec<i64>)]) -> SearchResult<Vec<f32>> {
        let n_docs = batch.len();
        let lens: Vec<usize> = batch.iter().map(|(ids, _)| ids.len()).collect();
        let total: usize = lens.iter().sum();
        if total == 0 {
            return Ok(vec![0.0; n_docs]);
        }
        let mut offsets = Vec::with_capacity(n_docs);
        {
            let mut o = 0usize;
            for &l in &lens {
                offsets.push(o);
                o += l;
            }
        }
        // Flat token / position / type ids over the concatenated documents. Each
        // document's positions restart at 0 (positions are intra-document).
        let mut ids_flat = Vec::with_capacity(total);
        let mut pos_flat = Vec::with_capacity(total);
        let mut typ_flat = Vec::with_capacity(total);
        for (ids, typ) in batch {
            for (i, (&id, &t)) in ids.iter().zip(typ.iter()).enumerate() {
                ids_flat.push(id);
                pos_flat.push(index_to_i64(i, "forward_batch.position")?);
                typ_flat.push(t);
            }
        }
        // Embeddings → [total, H].
        let id_t = self.idx(&ids_flat)?;
        let pos_t = self.idx(&pos_flat)?;
        let typ_t = self.idx(&typ_flat)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let emb_wp = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        // Fuse the third-embedding add with the embedding LayerNorm.
        let mut emb = self.add_ln(emb_wp, e_typ, "bert.embeddings.LayerNorm")?;

        let scale = ATTN_SCALE_F32;
        // True once the final layer has already collapsed the activation to one
        // `[CLS]` row per document (`[n_docs, H]`), so the pooler gathers identity
        // rows instead of the per-doc offsets.
        let cls_prepacked = if lens.iter().all(|&l| l <= FUSED_ATTN_MAX_SEQ) {
            // Tape-free fused-layer path (every doc short): extract the activation
            // once, run the encoder layers entirely on raw f32 buffers through the
            // SAME optimized kernels (fused QKV + fused attention + add&norm +
            // vectorized GELU) with no per-op tape-node creation / leaf allocation /
            // truncation, then reinsert once for the pooler. Bit-identical to the
            // tape path. The FINAL layer computes only each doc's `[CLS]` row (the
            // sole token the pooler reads), skipping ~S× of its attn-out + FFN work.
            // One attention scratch reused across every layer (and document) of this
            // chunk's forward, so the head-major repacks + the two bmm outputs are
            // allocated once instead of per attention call.
            let mut scratch = AttnScratch::default();
            let mut emb_vals = self
                .s
                .tensor_values_f32(emb)
                .map_err(|e| rerank_err("batch.emb_extract", e))?;
            for i in 0..self.encoder_layers - 1 {
                let p = format!("bert.encoder.layer.{i}");
                emb_vals = self.encoder_layer_raw(
                    &emb_vals,
                    total,
                    &offsets,
                    &lens,
                    &p,
                    scale,
                    &mut scratch,
                )?;
            }
            let p_last = format!("bert.encoder.layer.{}", self.encoder_layers - 1);
            let cls_vals = self.encoder_layer_cls(
                &emb_vals,
                &offsets,
                &lens,
                total,
                &p_last,
                scale,
                &mut scratch,
            )?;
            emb = self
                .s
                .tensor_variable_f32(cls_vals, vec![n_docs, H], false)
                .map_err(|e| rerank_err("batch.emb_reinsert", e))?;
            true
        } else {
            // Long-document path: separate-QKV + cache-blocking bmm attention through
            // the tape (a long doc in the chunk; rare).
            for i in 0..self.encoder_layers {
                let p = format!("bert.encoder.layer.{i}");
                let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
                let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
                let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
                let mut ctx_parts = Vec::with_capacity(n_docs);
                for n in 0..n_docs {
                    let (off, len) = (offsets[n], lens[n]);
                    let qn = self
                        .s
                        .tensor_narrow(q, 0, off, len)
                        .map_err(|e| rerank_err("batch.q_narrow", e))?;
                    let kn = self
                        .s
                        .tensor_narrow(k, 0, off, len)
                        .map_err(|e| rerank_err("batch.k_narrow", e))?;
                    let vn = self
                        .s
                        .tensor_narrow(v, 0, off, len)
                        .map_err(|e| rerank_err("batch.v_narrow", e))?;
                    ctx_parts.push(self.attn_bmm(qn, kn, vn, len, scale)?);
                }
                let ctx = self
                    .s
                    .tensor_cat(&ctx_parts, 0)
                    .map_err(|e| rerank_err("batch.ctx_cat", e))?; // [total, H]
                let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
                emb = self.add_ln(emb, attn, &format!("{p}.attention.output.LayerNorm"))?;
                let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
                let inter = self.gelu(inter)?;
                let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
                emb = self.add_ln(emb, ffn, &format!("{p}.output.LayerNorm"))?;
            }
            false
        };
        // Pooler on each document's [CLS] row → [N, H]. When the final layer already
        // emitted one CLS row per doc (`cls_prepacked`), those are rows 0..n_docs;
        // otherwise CLS is the first token of each doc, row `offsets[n]`.
        let cls_idx: Vec<i64> = if cls_prepacked {
            (0..n_docs)
                .map(|i| index_to_i64(i, "forward_batch.cls_idx"))
                .collect::<SearchResult<_>>()?
        } else {
            offsets
                .iter()
                .map(|&o| index_to_i64(o, "forward_batch.cls_offset"))
                .collect::<SearchResult<_>>()?
        };
        let cls_t = self.idx(&cls_idx)?;
        let cls = self
            .s
            .tensor_index_select(emb, 0, cls_t)
            .map_err(|e| rerank_err("batch.cls_gather", e))?; // [N, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self
            .s
            .tensor_tanh(pooled)
            .map_err(|e| rerank_err("pooler.tanh", e))?;
        let logit_t = self.linear(pooled, "classifier")?; // [N, 1]
        let vals = self
            .s
            .tensor_values_f32(logit_t)
            .map_err(|e| rerank_err("classifier.values", e))?;
        self.s.truncate_autograd_graph(self.weights_boundary);
        if vals.len() != n_docs {
            return Err(rerank_err(
                "classifier",
                format!("expected {n_docs} logits, got {}", vals.len()),
            ));
        }
        Ok(vals)
    }

    /// Sentence-embedding forward (the embedder head). Runs the SAME shared BERT
    /// encoder as the reranker over each input's tokens — identical embeddings build
    /// and `encoder_layer_raw` (same int8/SIMD kernels) — then replaces the reranker's
    /// `[CLS]` pooler + classifier with **mean-pooling over every token + L2-normalize**
    /// (the `sentence-transformers/all-MiniLM-L6-v2` head). Token-type ids are all 0
    /// (a single text, no query/doc split). Returns one `[H]` unit vector per input.
    ///
    /// Every input is ≤ `DEFAULT_MAX_LENGTH` (callers truncate at tokenization), so the
    /// whole batch goes through the tape-free raw path; there is no CLS-only shortcut
    /// because mean-pooling needs every token's final hidden state.
    pub(crate) fn embed_forward(&mut self, batch: &[Vec<i64>]) -> SearchResult<Vec<Vec<f32>>> {
        let n_docs = batch.len();
        let lens: Vec<usize> = batch.iter().map(Vec::len).collect();
        let total: usize = lens.iter().sum();
        if total == 0 {
            return Ok(vec![vec![0.0; H]; n_docs]);
        }
        let mut offsets = Vec::with_capacity(n_docs);
        {
            let mut o = 0usize;
            for &l in &lens {
                offsets.push(o);
                o += l;
            }
        }
        // Flat token / position / type ids over the concatenated inputs. Positions
        // restart at 0 per input; token-type is always 0 (single text).
        let mut ids_flat = Vec::with_capacity(total);
        let mut pos_flat = Vec::with_capacity(total);
        let mut typ_flat = Vec::with_capacity(total);
        for ids in batch {
            for (i, &id) in ids.iter().enumerate() {
                ids_flat.push(id);
                pos_flat.push(index_to_i64(i, "embed_forward.position")?);
                typ_flat.push(0i64);
            }
        }
        // Embeddings → [total, H]: word + position + token_type, then LayerNorm.
        let id_t = self.idx(&ids_flat)?;
        let pos_t = self.idx(&pos_flat)?;
        let typ_t = self.idx(&typ_flat)?;
        let we = self.g("bert.embeddings.word_embeddings.weight")?;
        let pe = self.g("bert.embeddings.position_embeddings.weight")?;
        let te = self.g("bert.embeddings.token_type_embeddings.weight")?;
        let e_word = self
            .s
            .tensor_index_select(we, 0, id_t)
            .map_err(|e| rerank_err("embed.word", e))?;
        let e_pos = self
            .s
            .tensor_index_select(pe, 0, pos_t)
            .map_err(|e| rerank_err("embed.pos", e))?;
        let e_typ = self
            .s
            .tensor_index_select(te, 0, typ_t)
            .map_err(|e| rerank_err("embed.type", e))?;
        let emb_wp = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        let emb = self.add_ln(emb_wp, e_typ, "bert.embeddings.LayerNorm")?;

        // Encoder: every attested layer on raw f32 buffers (mean-pooling needs every token's
        // final hidden state, so no CLS-only last layer). Same kernels as the reranker.
        let scale = ATTN_SCALE_F32;
        let mut scratch = AttnScratch::default();
        let mut emb_vals = self
            .s
            .tensor_values_f32(emb)
            .map_err(|e| rerank_err("embed.extract", e))?;
        for i in 0..self.encoder_layers {
            let p = format!("bert.encoder.layer.{i}");
            emb_vals =
                self.encoder_layer_raw(&emb_vals, total, &offsets, &lens, &p, scale, &mut scratch)?;
        }
        self.s.truncate_autograd_graph(self.weights_boundary);

        // Mean-pool each input's token rows → [H], then L2-normalize to a unit vector.
        let mut out = Vec::with_capacity(n_docs);
        for (&off, &len) in offsets.iter().zip(&lens) {
            let mut acc = vec![0.0f32; H];
            if len > 0 {
                let doc = &emb_vals[off * H..(off + len) * H];
                for t in 0..len {
                    let row = &doc[t * H..t * H + H];
                    for (a, &r) in acc.iter_mut().zip(row) {
                        *a += r;
                    }
                }
                let inv = 1.0 / len as f32;
                for a in &mut acc {
                    *a *= inv;
                }
            }
            let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                for a in &mut acc {
                    *a *= inv;
                }
            }
            out.push(acc);
        }
        Ok(out)
    }
}

/// Pure-Rust frankentorch cross-encoder reranker.
pub struct NativeReranker {
    /// A single frankentorch session behind a `Mutex`. Documents are reranked in
    /// a SEQUENTIAL loop, and each forward parallelizes internally across cores
    /// (the int8 Linear kernel + the f32 attention `bmm` / `softmax` /
    /// `layer_norm` ops use ambient rayon). Because there is no doc-level
    /// `par_iter`, nothing nests rayon while holding the lock, so the
    /// nested-rayon + `Mutex` deadlock is impossible by construction. Per-forward
    /// parallelism makes the common few-doc rerank fast (each forward uses all
    /// cores); a batched forward is the deferred next step for large-N throughput.
    inner: Mutex<Model>,
    tokenizer: Tokenizer,
    max_length: usize,
    name: String,
    id: String,
}

impl std::fmt::Debug for NativeReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeReranker")
            .field("name", &self.name)
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

impl NativeReranker {
    /// Load the reranker from a model directory containing a safetensors weight file
    /// (`model_f32.safetensors` preferred, else `model.safetensors`) and `tokenizer.json`.
    ///
    /// # Errors
    /// `SearchError::ModelNotFound` when required files are missing;
    /// `SearchError::ModelLoadFailed` when the tokenizer or weights fail to load.
    pub fn load(model_dir: impl AsRef<Path>) -> SearchResult<Self> {
        let dir = model_dir.as_ref();

        let tok_path = dir.join(TOKENIZER_JSON);
        if !tok_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{MODEL_NAME} (missing {TOKENIZER_JSON} in {})",
                    dir.display()
                ),
            });
        }
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("tokenizer load failed: {e}").into(),
            })?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| SearchError::ModelLoadFailed {
                path: tok_path.clone(),
                source: format!("failed to enable truncation: {e}").into(),
            })?;

        let weights_path = {
            let primary = dir.join(SAFETENSORS_PRIMARY);
            if primary.is_file() {
                primary
            } else {
                dir.join(SAFETENSORS_FALLBACK)
            }
        };
        if !weights_path.is_file() {
            return Err(SearchError::ModelNotFound {
                name: format!(
                    "{MODEL_NAME} (missing {SAFETENSORS_PRIMARY} or {SAFETENSORS_FALLBACK} in {})",
                    dir.display()
                ),
            });
        }

        // Parse + quantize the weights once and build a single session. Documents
        // are reranked sequentially; each forward parallelizes internally across
        // cores. No pool is needed (and one session keeps the f32 embedding table
        // resident only once, ~47 MB instead of per-slot copies).
        let shared = parse_weights(&weights_path)?;
        if shared.encoder_layers != RERANKER_LAYERS {
            return Err(SearchError::ModelLoadFailed {
                path: weights_path,
                source: format!(
                    "reranker weights contain {} encoder layers, expected {RERANKER_LAYERS}",
                    shared.encoder_layers
                )
                .into(),
            });
        }
        let linear_int8 = shared.qw.len();
        let f32_params = shared.f32_params.len();
        let model = build_model(shared)?;

        tracing::info!(
            model = MODEL_NAME,
            linear_int8,
            f32_params,
            max_length = DEFAULT_MAX_LENGTH,
            model_dir = %dir.display(),
            "native frankentorch reranker loaded (int8 linear, parallel forward)"
        );

        Ok(Self {
            inner: Mutex::new(model),
            tokenizer,
            max_length: DEFAULT_MAX_LENGTH,
            name: MODEL_NAME.to_owned(),
            id: MODEL_NAME.to_owned(),
        })
    }
}

/// A weight tensor is a Linear weight (to be int8-quantized) iff it is a `.weight`
/// that is neither a `LayerNorm` gain nor an embedding table.
fn is_linear_weight(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains("LayerNorm") && !name.contains("embeddings")
}

/// Parsed weight data: int8 Linear weights keyed by layer prefix, plus the f32
/// `embedding/LayerNorm` parameter values. [`build_model`] consumes this staging
/// representation so large embedding tables move into the session without a
/// second resident copy.
pub(crate) struct SharedWeights {
    qw: HashMap<String, QLinear>,
    f32_params: HashMap<String, (Arc<Vec<f32>>, Vec<usize>)>,
    encoder_layers: usize,
}

/// Parse a safetensors file: int8-quantize the Linear weights (per output channel)
/// and keep the embeddings + `LayerNorm` parameters as f32. Non-F32 tensors (e.g. the
/// I64 `position_ids` buffer) are skipped — those indices are regenerated at forward.
pub(crate) fn parse_weights(path: &Path) -> SearchResult<SharedWeights> {
    let bytes = fs::read(path).map_err(|e| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: format!("read safetensors: {e}").into(),
    })?;
    if bytes.len() < 8 {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors file too small".into(),
        });
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")))
        .map_err(|_| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header length exceeds usize::MAX".into(),
        })?;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header length out of range".into(),
        })?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| {
        SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors header parse: {e}").into(),
        }
    })?;
    let data = &bytes[header_end..];
    let obj = header
        .as_object()
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors header is not an object".into(),
        })?;

    // First pass: read every F32 tensor into raw (name -> (values, shape)).
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if dtype != "F32" {
            continue; // skip I64 position_ids and any non-float buffers
        }
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|u| {
                        usize::try_from(u).map_err(|_| SearchError::ModelLoadFailed {
                            path: path.to_path_buf(),
                            source: format!(
                                "safetensors tensor {name} shape dimension exceeds usize::MAX"
                            )
                            .into(),
                        })
                    })
                    .collect::<SearchResult<_>>()
            })
            .transpose()?
            .unwrap_or_default();
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} missing data_offsets").into(),
            })?;
        let start = usize::try_from(
            offsets
                .first()
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .map_err(|_| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors tensor {name} start offset exceeds usize::MAX").into(),
        })?;
        let end = usize::try_from(
            offsets
                .get(1)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        )
        .map_err(|_| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: format!("safetensors tensor {name} end offset exceeds usize::MAX").into(),
        })?;
        if start > end || end > data.len() {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} has out-of-range offsets").into(),
            });
        }
        let (chunks, _) = data[start..end].as_chunks::<4>();
        let vals: Vec<f32> = chunks
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        // Normalize HuggingFace BERT key conventions to the `bert.`-prefixed scheme
        // the shared encoder/`build_model` use. sentence-transformers all-MiniLM-L6-v2
        // ships bare `embeddings.*` / `encoder.*` keys; cross-encoder/ms-marco ships
        // `bert.`-prefixed ones — so this is a strict no-op there (those keys start with
        // `bert.embeddings`/`bert.encoder`, not bare `embeddings.`/`encoder.`). Backbone
        // keys only; `pooler`/`classifier` are left untouched.
        let key = if name.starts_with("embeddings.") || name.starts_with("encoder.") {
            format!("bert.{name}")
        } else {
            name.clone()
        };
        raw.insert(key, (vals, shape));
    }
    if raw.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no F32 tensors found in safetensors".into(),
        });
    }

    // Every retained tensor is now owned by `raw`; release the original 449 MiB
    // safetensors byte buffer before quantization creates another set of weights.
    drop(bytes);

    let encoder_layer_indices = raw
        .keys()
        .filter_map(|name| {
            name.strip_prefix("bert.encoder.layer.")?
                .split_once('.')?
                .0
                .parse::<usize>()
                .ok()
        })
        .collect::<BTreeSet<_>>();
    let encoder_layers = encoder_layer_indices
        .last()
        .copied()
        .and_then(|last| last.checked_add(1))
        .ok_or_else(|| SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors has no numbered BERT encoder layers".into(),
        })?;
    if encoder_layer_indices.len() != encoder_layers
        || !(0..encoder_layers).all(|index| encoder_layer_indices.contains(&index))
    {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "safetensors BERT encoder layers are not contiguous from zero".into(),
        });
    }

    // Second pass, part one: fuse each layer's Q/K/V projections while the raw
    // tensors are still present. Missing or malformed projections are a hard
    // topology failure: every attested BERT encoder layer must be executable.
    let mut qw: HashMap<String, QLinear> = HashMap::new();
    // Fuse each layer's Q/K/V projections into one `[3H, H]` linear
    // (key `…attention.self.qkv`). The forward then quantizes `emb` once and runs a
    // single int8 GEMM instead of three — cutting the per-call quant / rayon-launch
    // / dequant / tape-node overhead that is a real fraction of the short-sequence
    // forward (the SDOT math itself is at its M4 throughput ceiling). The stacked
    // weight re-quantizes per output channel, so each of the 3H rows keeps its own
    // scale and the fused output is byte-identical to the three separate linears.
    for i in 0..encoder_layers {
        let p = format!("bert.encoder.layer.{i}");
        let parts = ["query", "key", "value"];
        let mut stacked: Vec<f32> = Vec::with_capacity(3 * H * H);
        let mut bias: Vec<f32> = Vec::with_capacity(3 * H);
        for part in parts {
            let wn = format!("{p}.attention.self.{part}.weight");
            match raw.get(&wn) {
                Some((vals, shape)) if shape.len() == 2 && shape[0] == H && shape[1] == H => {
                    stacked.extend_from_slice(vals);
                    let b = raw
                        .get(&format!("{p}.attention.self.{part}.bias"))
                        .map(|(b, _)| b.clone())
                        .unwrap_or_else(|| vec![0.0f32; H]);
                    bias.extend_from_slice(&b);
                }
                _ => {
                    return Err(SearchError::ModelLoadFailed {
                        path: path.to_path_buf(),
                        source: format!(
                            "encoder layer {i} has missing or malformed {part} projection"
                        )
                        .into(),
                    });
                }
            }
        }
        let (out, in_) = (3 * H, H);
        let (w_i8, w_scales) = quantize_per_output_channel_i8(&stacked, out, in_);
        let packed = cfg!(target_arch = "aarch64") && out % 4 == 0 && in_ % 16 == 0;
        let w_i8 = if packed {
            ft_api::pack_int8_weights_nr4(&w_i8, out, in_)
        } else {
            w_i8
        };
        qw.insert(
            format!("{p}.attention.self.qkv"),
            QLinear {
                w_i8: Arc::new(w_i8),
                w_scales: Arc::new(w_scales),
                bias: Arc::new(bias),
                out,
                in_,
                packed,
            },
        );
    }

    // Second pass, part two: remove and quantize each Linear weight together
    // with its bias. Removing entries lets the large f32 vectors move into their
    // final owners instead of being cloned from the staging map.
    let linear_weight_names = raw
        .keys()
        .filter(|name| is_linear_weight(name))
        .cloned()
        .collect::<Vec<_>>();
    for name in linear_weight_names {
        let (vals, shape) = raw
            .remove(&name)
            .expect("linear weight name was collected from raw");
        let prefix = name.strip_suffix(".weight").expect("ends_with .weight");
        let out = *shape.first().unwrap_or(&0);
        let in_ = *shape.get(1).unwrap_or(&0);
        if out == 0 || in_ == 0 || vals.len() != out * in_ {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!(
                    "linear weight {name} bad shape {shape:?} for {} values",
                    vals.len()
                )
                .into(),
            });
        }
        let bias = raw
            .remove(&format!("{prefix}.bias"))
            .map_or_else(|| vec![0.0f32; out], |(bias, _)| bias);
        let (w_i8, w_scales) = quantize_per_output_channel_i8(&vals, out, in_);
        // Pre-pack the static weights into the NR=4 SDOT tile layout once at
        // load (the zero-per-forward weight-packing win) on aarch64, where the
        // packed micro-kernel runs and `out % 4 == 0 && in % 16 == 0` holds for
        // every linear except the 1-row classifier. Other targets / the
        // classifier keep row-major + the portable kernel.
        let packed = cfg!(target_arch = "aarch64") && out % 4 == 0 && in_ % 16 == 0;
        let w_i8 = if packed {
            ft_api::pack_int8_weights_nr4(&w_i8, out, in_)
        } else {
            w_i8
        };
        qw.insert(
            prefix.to_string(),
            QLinear {
                w_i8: Arc::new(w_i8),
                w_scales: Arc::new(w_scales),
                bias: Arc::new(bias),
                out,
                in_,
                packed,
            },
        );
    }
    if qw.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no Linear weights found to quantize".into(),
        });
    }

    let mut f32_params: HashMap<String, (Arc<Vec<f32>>, Vec<usize>)> = HashMap::new();
    for (name, (vals, shape)) in raw {
        if name.strip_suffix(".bias").is_some() && !name.contains("LayerNorm") {
            // Bias for a non-Linear auxiliary tensor; it is not a model leaf.
            continue;
        }
        f32_params.insert(name, (Arc::new(vals), shape));
    }

    Ok(SharedWeights {
        qw,
        f32_params,
        encoder_layers,
    })
}

/// Build a session by consuming parsed weights. Large embedding tables transfer
/// directly into the session; only the small `LayerNorm` values are duplicated
/// because the tape-free path also needs their raw buffers.
pub(crate) fn build_model(shared: SharedWeights) -> SearchResult<Model> {
    let SharedWeights {
        qw,
        f32_params,
        encoder_layers,
    } = shared;
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    session.no_grad_enter();
    let mut w = HashMap::with_capacity(f32_params.len());
    let mut raw_params = HashMap::with_capacity(f32_params.len());
    for (name, (vals, shape)) in f32_params {
        let raw_layer_norm = name.contains("LayerNorm").then(|| Arc::clone(&vals));
        let vals = Arc::try_unwrap(vals).unwrap_or_else(|vals| vals.as_ref().clone());
        let node = session
            .tensor_variable_f32(vals, shape, false)
            .map_err(|e| rerank_err("build_model", format!("create f32 tensor {name}: {e}")))?;
        if let Some(raw) = raw_layer_norm {
            raw_params.insert(name.clone(), raw);
        }
        w.insert(name, node);
    }
    // Tape boundary AFTER the persistent f32 leaves are created; each forward
    // truncates back to here to free intermediates while keeping parameters.
    let weights_boundary = session.autograd_graph_node_count();
    Ok(Model {
        s: session,
        w,
        qw,
        raw_params,
        encoder_layers,
        weights_boundary,
    })
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl SyncRerank for NativeReranker {
    fn rerank_sync(
        &self,
        query: &str,
        documents: &[RerankDocument],
    ) -> SearchResult<Vec<RerankScore>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        // Tokenize every (query, doc) pair, then rerank in BATCHED chunks: each
        // chunk is a single `forward_batch` over the concatenated tokens, so the
        // int8 Linear weights are reused across all the chunk's documents' tokens
        // (the throughput win). Chunks are bounded by `MAX_BATCH_TOKENS` to keep
        // the tape memory-bounded. There is NO doc-level `par_iter` — the session
        // `Mutex` is locked once and the forward parallelizes internally — so the
        // nested-rayon + `Mutex` deadlock still cannot occur. Output order (and
        // `original_rank`) follows the input and the logits are deterministic.
        let mut encoded: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(documents.len());
        for doc in documents {
            let encoding = self
                .tokenizer
                .encode((query, doc.text.as_str()), true)
                .map_err(|e| rerank_err("tokenize", e))?;
            // Truncate to `max_length` before the i64 conversion + collect (see
            // `crate::ids_to_truncated_i64`): for documents that tokenize past the cap
            // this materializes only `max_length` ids per side instead of the whole
            // sequence. `ids` and `typ` share the encoding length, so per-side
            // `take(max_length)` matches the old `if ids.len() > max { truncate both }`.
            let ids = crate::ids_to_truncated_i64(encoding.get_ids(), self.max_length);
            let typ = crate::ids_to_truncated_i64(encoding.get_type_ids(), self.max_length);
            encoded.push((ids, typ));
        }

        let mut model = self
            .inner
            .lock()
            .map_err(|e| rerank_err("lock", format!("reranker mutex poisoned: {e}")))?;
        let mut logits: Vec<f32> = Vec::with_capacity(documents.len());
        let mut chunk_start = 0usize;
        while chunk_start < encoded.len() {
            // Grow the chunk until adding the next doc would exceed the token
            // budget; always take at least one doc (a single over-budget doc runs
            // alone).
            let mut chunk_end = chunk_start + 1;
            let mut chunk_tokens = encoded[chunk_start].0.len();
            while chunk_end < encoded.len()
                && chunk_tokens + encoded[chunk_end].0.len() <= MAX_BATCH_TOKENS
            {
                chunk_tokens += encoded[chunk_end].0.len();
                chunk_end += 1;
            }
            logits.extend(model.forward_batch(&encoded[chunk_start..chunk_end])?);
            chunk_start = chunk_end;
        }
        drop(model);

        let out = documents
            .iter()
            .zip(logits)
            .enumerate()
            .map(|(rank, (doc, logit))| {
                let (score, raw_logit) = if logit.is_finite() {
                    (sigmoid(logit), Some(logit))
                } else {
                    (0.0, None)
                };
                RerankScore {
                    doc_id: doc.doc_id.clone(),
                    score,
                    original_rank: rank,
                    raw_logit,
                }
            })
            .collect();
        Ok(out)
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_length(&self) -> usize {
        self.max_length
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_DIR: &str = "/private/tmp/ee-reranker-port/model";

    // (query, document, reference logit) from the validated parity_cases.json
    // (numpy reference in f64, itself validated bit-for-ranking against the real ONNX
    // model). The forward runs int8 Linear matmuls on an f32 substrate, so logits track
    // the f64 reference only within an int8 quantization tolerance (PARITY_TOL); the
    // ranking is what must stay identical (as it did for the original int8 ONNX model).
    const PARITY_TOL: f64 = 0.6;
    const CASES: &[(&str, &str, f64)] = &[
        (
            "how to fix a failing release workflow",
            "the release pipeline builds cross platform binaries and uploads them to github",
            -9.808_567,
        ),
        (
            "how to fix a failing release workflow",
            "bananas are a good source of potassium and taste sweet",
            -11.332_987,
        ),
        (
            "what is the capital of france",
            "paris is the capital and most populous city of france",
            7.472_003,
        ),
        (
            "rust memory safety",
            "the borrow checker enforces ownership rules at compile time",
            -11.367_251,
        ),
    ];

    fn model_available() -> bool {
        Path::new(MODEL_DIR).join(TOKENIZER_JSON).is_file()
            && (Path::new(MODEL_DIR).join(SAFETENSORS_PRIMARY).is_file()
                || Path::new(MODEL_DIR).join(SAFETENSORS_FALLBACK).is_file())
    }

    fn doc(id: &str, text: &str) -> RerankDocument {
        RerankDocument {
            doc_id: id.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn parity_logits_and_ranking_match_reference() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP parity: model dir {MODEL_DIR} not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load native reranker");
        let mut logits = Vec::new();
        let mut max_diff = 0.0_f64;
        eprintln!("[native_reranker] idx |     ft_logit |    ref_logit |     diff");
        for (i, (query, document, ref_logit)) in CASES.iter().enumerate() {
            let scored = reranker
                .rerank_sync(query, &[doc("d", document)])
                .expect("rerank_sync");
            assert_eq!(scored.len(), 1, "one doc in, one score out");
            let logit = f64::from(scored[0].raw_logit.expect("raw logit present"));
            let diff = (logit - ref_logit).abs();
            max_diff = max_diff.max(diff);
            logits.push(logit);
            eprintln!("[native_reranker] {i:3} | {logit:12.6} | {ref_logit:12.6} | {diff:8.5}");
            assert!(
                diff < PARITY_TOL,
                "case {i} logit {logit} differs from reference {ref_logit} by {diff} (>{PARITY_TOL})"
            );
        }
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        eprintln!(
            "[native_reranker] ranking(desc)={order:?} expected=[2, 0, 1, 3] max_diff={max_diff:.6}"
        );
        assert_eq!(order, vec![2usize, 0, 1, 3], "ranking must match reference");
    }

    #[test]
    fn forward_batch_matches_per_doc() {
        // The batched (varlen) forward must produce the same logits as running each
        // document through the single-pair `forward`: it reuses the same attention
        // per document and every other op is row-wise, so batching is exact, not an
        // approximation. This is the contract that lets the throughput lever ship
        // without touching the validated ranking.
        if !model_available() {
            eprintln!("[native_reranker] SKIP batch-equiv: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load native reranker");
        let query = CASES[0].0;
        let mut batch: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
        for (_, document, _) in CASES {
            let enc = reranker
                .tokenizer
                .encode((query, *document), true)
                .expect("tokenize");
            let ids: Vec<i64> = enc.get_ids().iter().map(|&x| i64::from(x)).collect();
            let typ: Vec<i64> = enc.get_type_ids().iter().map(|&x| i64::from(x)).collect();
            batch.push((ids, typ));
        }
        let mut model = reranker.inner.lock().expect("lock");
        let per_doc: Vec<f32> = batch
            .iter()
            .map(|(ids, typ)| model.forward(ids, typ).expect("forward"))
            .collect();
        let batched = model.forward_batch(&batch).expect("forward_batch");
        drop(model);
        assert_eq!(batched.len(), per_doc.len());
        for (i, (b, p)) in batched.iter().zip(&per_doc).enumerate() {
            let diff = (f64::from(*b) - f64::from(*p)).abs();
            eprintln!("[native_reranker] doc {i}: batched={b:.6} per_doc={p:.6} diff={diff:.2e}");
            assert!(
                diff < 1e-3,
                "doc {i}: batched {b} vs per-doc {p} diff {diff} too large"
            );
        }
    }

    #[test]
    fn empty_documents_yield_empty_scores() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP empty-docs: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        let scored = reranker.rerank_sync("any query", &[]).expect("empty ok");
        assert!(scored.is_empty());
        eprintln!("[native_reranker] empty-docs -> empty scores OK");
    }

    #[test]
    fn whitespace_and_long_documents_do_not_panic() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP whitespace/long: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        // whitespace-only doc
        let ws = reranker
            .rerank_sync("q", &[doc("ws", "   ")])
            .expect("whitespace ok");
        assert_eq!(ws.len(), 1);
        // very long doc (forces truncation well beyond max_length)
        let long_text = "memory safety ".repeat(400);
        let lng = reranker
            .rerank_sync("rust", &[doc("long", &long_text)])
            .expect("long ok");
        assert_eq!(lng.len(), 1);
        assert!(lng[0].score.is_finite());
        eprintln!(
            "[native_reranker] whitespace score={:.6}, truncated-long score={:.6} OK",
            ws[0].score, lng[0].score
        );
    }

    #[test]
    fn ranking_is_deterministic_across_runs() {
        if !model_available() {
            eprintln!("[native_reranker] SKIP determinism: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        let docs: Vec<RerankDocument> = CASES
            .iter()
            .enumerate()
            .map(|(i, (_, d, _))| doc(&format!("d{i}"), d))
            .collect();
        let run1 = reranker
            .rerank_sync("what is the capital of france", &docs)
            .expect("run1");
        let run2 = reranker
            .rerank_sync("what is the capital of france", &docs)
            .expect("run2");
        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.doc_id, b.doc_id);
            assert_eq!(a.raw_logit, b.raw_logit, "logits must be deterministic");
        }
        eprintln!("[native_reranker] determinism across 2 runs OK");
    }

    #[test]
    fn many_documents_rerank_without_deadlock() {
        // Regression guard for the nested-rayon + Mutex deadlock. The fix runs
        // each forward on its slot's dedicated single-thread rayon pool, so no
        // frankentorch op (int8 linear, the f32 attention bmm, softmax,
        // layer_norm, ...) can fan rayon work back onto the doc-dispatch pool
        // while a worker holds a session Mutex. Reranking far more documents
        // than the pool size forces multiple rayon workers onto the same slot —
        // the exact collision that previously hung the multi-doc path. If a
        // future change re-introduces nesting (a forward spawning rayon work
        // onto the doc-dispatch pool while holding its session Mutex), this test
        // deadlocks (CI timeout) rather than silently shipping a hang.
        if !model_available() {
            eprintln!("[native_reranker] SKIP many-docs: model dir not present");
            return;
        }
        let reranker = NativeReranker::load(MODEL_DIR).expect("load");
        // 24 docs >> the 8-session pool cap, so several workers share a slot.
        let docs: Vec<RerankDocument> = (0..24)
            .map(|i| doc(&format!("d{i}"), CASES[i % CASES.len()].1))
            .collect();
        let scored = reranker
            .rerank_sync("what is the capital of france", &docs)
            .expect("many-doc rerank completes (no deadlock)");
        assert_eq!(scored.len(), docs.len(), "one score per doc");
        for (i, s) in scored.iter().enumerate() {
            assert_eq!(s.original_rank, i, "original_rank preserves input order");
            assert_eq!(s.doc_id, format!("d{i}"));
            assert!(s.score.is_finite());
        }
        // Same input twice -> identical scores (the parallel path is deterministic).
        let again = reranker
            .rerank_sync("what is the capital of france", &docs)
            .expect("rerun");
        for (a, b) in scored.iter().zip(again.iter()) {
            assert_eq!(
                a.raw_logit, b.raw_logit,
                "parallel rerank must be deterministic"
            );
        }
        eprintln!(
            "[native_reranker] {}-doc concurrent rerank OK (no deadlock, deterministic)",
            docs.len()
        );
    }

    #[test]
    fn load_missing_dir_errors() {
        let err = NativeReranker::load("/private/tmp/definitely-not-a-model-dir-xyz");
        assert!(err.is_err(), "loading a missing dir must error, not panic");
        eprintln!("[native_reranker] missing-dir load error OK: {err:?}");
    }
}
