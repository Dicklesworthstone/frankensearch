//! Pure-Rust cross-encoder reranker backed by frankentorch (no ONNX / no `ort`).
//!
//! Reimplements the `cross-encoder/ms-marco-MiniLM-L6-v2` `BertForSequenceClassification`
//! forward pass (6 layers, hidden 384, 12 heads, exact GELU, LayerNorm eps 1e-12,
//! `[CLS]` pooler + classifier, `sigmoid(logit)`) on frankentorch tensors, matching the
//! ONNX dynamic-quant scheme: an **f32 substrate** (embeddings, LayerNorm, softmax,
//! GELU, tanh) with **int8 Linear matmuls** (bd-1nl13.10/.15). Every Linear (attention
//! QKV/output, FFN, pooler, classifier) is statically int8-quantized per output channel
//! at load; its activation is dynamically int8-quantized per row at forward
//! (`tensor_linear_int8_dynamic`). Validated against the numpy/ONNX reference
//! (bd-1nl13.2/.3): the reference ranking is preserved.
//!
//! Embedding lookups go through `tensor_index_select` rather than `tensor_embedding`:
//! `index_select` preserves the weight dtype (f32 in/f32 out, frankentorch-40i), whereas
//! `tensor_embedding`'s custom gather still materialises f64. The two are semantically
//! identical here (no `padding_idx`). LayerNorm hits frankentorch's f32 fused no-grad
//! fast path.
//!
//! The only reranker backend (ort/ONNX was removed in bd-1nl13.6); feature-gated
//! behind `native`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ft_api::{FrankenTorchSession, quantize_per_output_channel_i8};
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;
use tokenizers::Tokenizer;
use wide::f32x8;

use frankensearch_core::error::{SearchError, SearchResult};
use frankensearch_core::traits::{RerankDocument, RerankScore, SyncRerank};

const H: usize = 384;
const L: usize = 6;
const NH: usize = 12;
const HD: usize = H / NH; // 32
const EPS: f64 = 1e-12;
const DEFAULT_MAX_LENGTH: usize = 512;
/// Token budget per batched forward. Documents are reranked in chunks whose total
/// token count stays under this cap so the chunk's attention intermediates (which
/// co-exist on the tape until the per-chunk truncate) stay memory-bounded, while
/// still giving the int8 GEMM a large enough row count to amortize weight loads.
/// A single document longer than this is still processed alone (one-doc chunk).
const MAX_BATCH_TOKENS: usize = 2048;
const MODEL_NAME: &str = "ms-marco-minilm-l-6-v2";
const SAFETENSORS_PRIMARY: &str = "model_f32.safetensors";
const SAFETENSORS_FALLBACK: &str = "model.safetensors";
const TOKENIZER_JSON: &str = "tokenizer.json";

fn rerank_err(ctx: &str, e: impl std::fmt::Display) -> SearchError {
    SearchError::RerankFailed {
        model: MODEL_NAME.to_owned(),
        source: format!("{ctx}: {e}").into(),
    }
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
    while i + 8 <= n {
        let mut buf = [0.0f32; 8];
        buf.copy_from_slice(&row[i..i + 8]);
        let e = ((f32x8::new(buf) - max_v) * scale_v).exp();
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
fn fast_softmax_last_dim(mut data: Vec<f32>, rows: usize, n: usize, scale: f32) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * n);
    if n == 0 {
        return data;
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
    data
}

/// A Linear layer's weights, statically quantized to int8 with per-output-channel
/// f32 scales, plus its f32 bias. The three buffers are `Arc`-shared so the parsed
/// weights are stored once and cloned cheaply into every pooled session.
#[derive(Clone)]
struct QLinear {
    /// Row-major `[out, in]` int8 weights.
    w_i8: Arc<Vec<i8>>,
    /// Per-output-channel f32 dequantization scales (len `out`).
    w_scales: Arc<Vec<f32>>,
    /// f32 bias (len `out`).
    bias: Arc<Vec<f32>>,
    out: usize,
    in_: usize,
}

/// Owns the frankentorch session and the loaded weight tensors. Mutated during the
/// forward pass, so it lives behind a `Mutex` in `NativeReranker`.
struct Model {
    s: FrankenTorchSession,
    /// f32 leaf nodes for the non-Linear parameters (word/position/token_type
    /// embeddings and every LayerNorm weight/bias) — these stay in f32.
    w: HashMap<String, TensorNodeId>,
    /// int8-quantized Linear weights (attention QKV/output, FFN, pooler,
    /// classifier), keyed by the layer prefix (the weight name minus `.weight`).
    qw: HashMap<String, QLinear>,
    /// Autograd tape node count captured right after the persistent weights are
    /// loaded. Each forward pass truncates the tape back to this boundary to free
    /// that pass's intermediate activations, so the session does not grow
    /// unbounded across many rerank calls (a single long-doc forward can allocate
    /// ~25 MB attention tensors per layer; without truncation they would
    /// accumulate for the life of the process).
    weights_boundary: usize,
}

impl Model {
    fn g(&self, name: &str) -> SearchResult<TensorNodeId> {
        self.w
            .get(name)
            .copied()
            .ok_or_else(|| rerank_err("weights", format!("missing weight tensor {name}")))
    }

    /// y = x @ Wᵀ + b via the int8 dynamic-quant kernel (weight stored row-major
    /// [out, in], PyTorch convention). The f32 activation `x` is dynamically
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
        let (out, in_) = (q.out, q.in_);
        self.s
            .tensor_linear_int8_dynamic(x, &w_i8, &w_scales, out, in_, Some(&bias))
            .map_err(|e| rerank_err("linear.int8", e))
    }

    fn ln(&mut self, x: TensorNodeId, prefix: &str) -> SearchResult<TensorNodeId> {
        let w = self.g(&format!("{prefix}.weight"))?;
        let b = self.g(&format!("{prefix}.bias"))?;
        self.s
            .tensor_layer_norm(x, vec![H], Some(w), Some(b), EPS)
            .map_err(|e| rerank_err("layer_norm", e))
    }

    fn idx(&mut self, vals: &[i64]) -> SearchResult<TensorNodeId> {
        let f: Vec<f64> = vals.iter().map(|&v| v as f64).collect();
        self.s
            .tensor_variable(f, vec![vals.len()], false)
            .map_err(|e| rerank_err("index_tensor", e))
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
    /// compute byte-identical attention — the batched path is therefore parity-exact
    /// with the per-doc path, not an approximation.
    fn attn_block(
        &mut self,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        s_len: usize,
        scale: f64,
    ) -> SearchResult<TensorNodeId> {
        let q = self.heads(q, s_len)?;
        let k = self.heads(k, s_len)?;
        let v = self.heads(v, s_len)?;
        let kt = self
            .s
            .tensor_transpose(k, 1, 2)
            .map_err(|e| rerank_err("attn.kt", e))?; // [NH, HD, S]
        let scores = self.s.tensor_bmm(q, kt).map_err(|e| rerank_err("attn.qk", e))?;
        // Fused scale + vectorized-exp softmax over the last dim, replacing the
        // separate `mul_scalar` + `tensor_softmax` tape ops. The int8 linear already
        // returns detached f32 leaves, so round-tripping the scores through f32
        // values + a fresh f32 leaf matches the forward's existing structure.
        let score_vals = self
            .s
            .tensor_values_f32(scores)
            .map_err(|e| rerank_err("attn.scores_vals", e))?;
        let prob_vals = fast_softmax_last_dim(score_vals, NH * s_len, s_len, scale as f32);
        let probs = self
            .s
            .tensor_variable_f32(prob_vals, vec![NH, s_len, s_len], false)
            .map_err(|e| rerank_err("attn.softmax_fast", e))?;
        let ctx = self.s.tensor_bmm(probs, v).map_err(|e| rerank_err("attn.ctx", e))?;
        let ctx = self
            .s
            .tensor_transpose(ctx, 0, 1)
            .map_err(|e| rerank_err("attn.ctx_t", e))?;
        self.s
            .tensor_reshape(ctx, vec![s_len, H])
            .map_err(|e| rerank_err("attn.ctx_reshape", e))
    }

    /// Single-pair forward pass (batch = 1). Returns the raw logit. Retained as the
    /// per-document reference that [`Self::forward_batch`] is checked against
    /// (`forward_batch_matches_per_doc`); production reranking always goes through
    /// the batched path (which handles a one-document batch with negligible
    /// overhead), so this is `#[cfg(test)]`-only.
    ///
    /// Runs entirely in f32: weights are f32 leaves and every op preserves f32, so
    /// embedding/matmul/softmax/layer_norm all stay in the f32 kernels.
    #[cfg(test)]
    fn forward(&mut self, ids: &[i64], typ: &[i64]) -> SearchResult<f32> {
        let s_len = ids.len();
        // embeddings: word + position + token_type, then LayerNorm.
        // `index_select(weight, dim=0, indices)` is the embedding lookup; unlike
        // `tensor_embedding` it preserves the f32 weight dtype (frankentorch-40i).
        let id_t = self.idx(ids)?;
        let pos: Vec<i64> = (0..s_len as i64).collect();
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
        let mut emb = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        emb = self.s.tensor_add(emb, e_typ).map_err(|e| rerank_err("embed.add2", e))?;
        emb = self.ln(emb, "bert.embeddings.LayerNorm")?;

        let scale = 1.0 / (HD as f64).sqrt();
        for i in 0..L {
            let p = format!("bert.encoder.layer.{i}");
            // self-attention
            let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
            let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
            let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
            let ctx = self.attn_block(q, k, v, s_len, scale)?;
            let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
            let sum1 = self.s.tensor_add(emb, attn).map_err(|e| rerank_err("attn.residual", e))?;
            emb = self.ln(sum1, &format!("{p}.attention.output.LayerNorm"))?;
            // feed-forward
            let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
            let inter = self.s.tensor_gelu(inter).map_err(|e| rerank_err("ffn.gelu", e))?;
            let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
            let sum2 = self.s.tensor_add(emb, ffn).map_err(|e| rerank_err("ffn.residual", e))?;
            emb = self.ln(sum2, &format!("{p}.output.LayerNorm"))?;
        }
        // pooler on [CLS] (row 0) + classifier
        let cls = self
            .s
            .tensor_narrow(emb, 0, 0, 1)
            .map_err(|e| rerank_err("pooler.narrow", e))?; // [1, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self.s.tensor_tanh(pooled).map_err(|e| rerank_err("pooler.tanh", e))?;
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
    /// (the int8 Linears, LayerNorm, GELU, residuals) runs once over the whole
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
                pos_flat.push(i as i64);
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
        let mut emb = self
            .s
            .tensor_add(e_word, e_pos)
            .map_err(|e| rerank_err("embed.add", e))?;
        emb = self.s.tensor_add(emb, e_typ).map_err(|e| rerank_err("embed.add2", e))?;
        emb = self.ln(emb, "bert.embeddings.LayerNorm")?;

        let scale = 1.0 / (HD as f64).sqrt();
        for i in 0..L {
            let p = format!("bert.encoder.layer.{i}");
            // Batched QKV projections over all documents' tokens at once.
            let q = self.linear(emb, &format!("{p}.attention.self.query"))?;
            let k = self.linear(emb, &format!("{p}.attention.self.key"))?;
            let v = self.linear(emb, &format!("{p}.attention.self.value"))?;
            // Self-attention per document (each on its own [lenₙ, H] slice), then
            // re-concatenate the contexts back into one [total, H] activation.
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
                ctx_parts.push(self.attn_block(qn, kn, vn, len, scale)?);
            }
            let ctx = self
                .s
                .tensor_cat(&ctx_parts, 0)
                .map_err(|e| rerank_err("batch.ctx_cat", e))?; // [total, H]
            let attn = self.linear(ctx, &format!("{p}.attention.output.dense"))?;
            let sum1 = self.s.tensor_add(emb, attn).map_err(|e| rerank_err("attn.residual", e))?;
            emb = self.ln(sum1, &format!("{p}.attention.output.LayerNorm"))?;
            // Batched feed-forward.
            let inter = self.linear(emb, &format!("{p}.intermediate.dense"))?;
            let inter = self.s.tensor_gelu(inter).map_err(|e| rerank_err("ffn.gelu", e))?;
            let ffn = self.linear(inter, &format!("{p}.output.dense"))?;
            let sum2 = self.s.tensor_add(emb, ffn).map_err(|e| rerank_err("ffn.residual", e))?;
            emb = self.ln(sum2, &format!("{p}.output.LayerNorm"))?;
        }
        // Pooler on each document's [CLS] row (the first token of each doc, i.e.
        // row `offsets[n]` in the concatenated activation) → [N, H].
        let cls_idx: Vec<i64> = offsets.iter().map(|&o| o as i64).collect();
        let cls_t = self.idx(&cls_idx)?;
        let cls = self
            .s
            .tensor_index_select(emb, 0, cls_t)
            .map_err(|e| rerank_err("batch.cls_gather", e))?; // [N, H]
        let pooled = self.linear(cls, "bert.pooler.dense")?;
        let pooled = self.s.tensor_tanh(pooled).map_err(|e| rerank_err("pooler.tanh", e))?;
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
                name: format!("{MODEL_NAME} (missing {TOKENIZER_JSON} in {})", dir.display()),
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
        let model = build_model(&shared)?;

        tracing::info!(
            model = MODEL_NAME,
            linear_int8 = shared.qw.len(),
            f32_params = shared.f32_params.len(),
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
/// that is neither a LayerNorm gain nor an embedding table.
fn is_linear_weight(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains("LayerNorm") && !name.contains("embeddings")
}

/// Parsed, immutable weight data: int8 Linear weights keyed by layer prefix, plus
/// the f32 embedding/LayerNorm parameter values. Parsed and quantized once, then
/// cloned (cheaply, via `Arc`) into each session by [`build_model`].
struct SharedWeights {
    qw: HashMap<String, QLinear>,
    f32_params: HashMap<String, (Arc<Vec<f32>>, Vec<usize>)>,
}

/// Parse a safetensors file: int8-quantize the Linear weights (per output channel)
/// and keep the embeddings + LayerNorm parameters as f32. Non-F32 tensors (e.g. the
/// I64 `position_ids` buffer) are skipped — those indices are regenerated at forward.
fn parse_weights(path: &Path) -> SearchResult<SharedWeights> {
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
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
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
    let obj = header.as_object().ok_or_else(|| SearchError::ModelLoadFailed {
        path: path.to_path_buf(),
        source: "safetensors header is not an object".into(),
    })?;

    // First pass: read every F32 tensor into raw (name -> (values, shape)).
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info.get("dtype").and_then(serde_json::Value::as_str).unwrap_or("");
        if dtype != "F32" {
            continue; // skip I64 position_ids and any non-float buffers
        }
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
            .unwrap_or_default();
        let offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} missing data_offsets").into(),
            })?;
        let start = offsets.first().and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        let end = offsets.get(1).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
        if start > end || end > data.len() {
            return Err(SearchError::ModelLoadFailed {
                path: path.to_path_buf(),
                source: format!("safetensors tensor {name} has out-of-range offsets").into(),
            });
        }
        let vals: Vec<f32> = data[start..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        raw.insert(name.clone(), (vals, shape));
    }
    if raw.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no F32 tensors found in safetensors".into(),
        });
    }

    // Second pass: classify. Linear `.weight`s are int8-quantized (folding in their
    // `.bias`, which is then skipped); everything else (embeddings + LayerNorm
    // weight/bias) stays f32.
    let mut qw: HashMap<String, QLinear> = HashMap::new();
    let mut f32_params: HashMap<String, (Arc<Vec<f32>>, Vec<usize>)> = HashMap::new();
    for (name, (vals, shape)) in &raw {
        if is_linear_weight(name) {
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
            let (w_i8, w_scales) = quantize_per_output_channel_i8(vals, out, in_);
            let bias = raw
                .get(&format!("{prefix}.bias"))
                .map(|(b, _)| b.clone())
                .unwrap_or_else(|| vec![0.0f32; out]);
            qw.insert(
                prefix.to_string(),
                QLinear {
                    w_i8: Arc::new(w_i8),
                    w_scales: Arc::new(w_scales),
                    bias: Arc::new(bias),
                    out,
                    in_,
                },
            );
        } else if name.ends_with(".bias") && !name.contains("LayerNorm") {
            // Linear bias — already folded into its QLinear above; do not keep as f32.
            continue;
        } else {
            // f32 parameter: embeddings and LayerNorm weight/bias.
            f32_params.insert(name.clone(), (Arc::new(vals.clone()), shape.clone()));
        }
    }
    if qw.is_empty() {
        return Err(SearchError::ModelLoadFailed {
            path: path.to_path_buf(),
            source: "no Linear weights found to quantize".into(),
        });
    }
    Ok(SharedWeights { qw, f32_params })
}

/// Build a fresh session from shared weights: create an f32 leaf for every
/// embedding/LayerNorm parameter and clone the (Arc-shared) int8 Linear weights.
fn build_model(shared: &SharedWeights) -> SearchResult<Model> {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    session.no_grad_enter();
    let mut w = HashMap::with_capacity(shared.f32_params.len());
    for (name, (vals, shape)) in &shared.f32_params {
        let node = session
            .tensor_variable_f32(vals.as_ref().clone(), shape.clone(), false)
            .map_err(|e| rerank_err("build_model", format!("create f32 tensor {name}: {e}")))?;
        w.insert(name.clone(), node);
    }
    // Tape boundary AFTER the persistent f32 leaves are created; each forward
    // truncates back to here to free intermediates while keeping parameters.
    let weights_boundary = session.autograd_graph_node_count();
    Ok(Model { s: session, w, qw: shared.qw.clone(), weights_boundary })
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
            let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
            let mut typ: Vec<i64> = encoding.get_type_ids().iter().map(|&t| i64::from(t)).collect();
            if ids.len() > self.max_length {
                ids.truncate(self.max_length);
                typ.truncate(self.max_length);
            }
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
            -9.808567,
        ),
        (
            "how to fix a failing release workflow",
            "bananas are a good source of potassium and taste sweet",
            -11.332987,
        ),
        (
            "what is the capital of france",
            "paris is the capital and most populous city of france",
            7.472003,
        ),
        (
            "rust memory safety",
            "the borrow checker enforces ownership rules at compile time",
            -11.367251,
        ),
    ];

    fn model_available() -> bool {
        Path::new(MODEL_DIR).join(TOKENIZER_JSON).is_file()
            && (Path::new(MODEL_DIR).join(SAFETENSORS_PRIMARY).is_file()
                || Path::new(MODEL_DIR).join(SAFETENSORS_FALLBACK).is_file())
    }

    fn doc(id: &str, text: &str) -> RerankDocument {
        RerankDocument { doc_id: id.to_owned(), text: text.to_owned() }
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
            let logit = scored[0].raw_logit.expect("raw logit present") as f64;
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
        eprintln!("[native_reranker] ranking(desc)={order:?} expected=[2, 0, 1, 3] max_diff={max_diff:.6}");
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
        assert_eq!(batched.len(), per_doc.len());
        for (i, (b, p)) in batched.iter().zip(&per_doc).enumerate() {
            let diff = (f64::from(*b) - f64::from(*p)).abs();
            eprintln!("[native_reranker] doc {i}: batched={b:.6} per_doc={p:.6} diff={diff:.2e}");
            assert!(diff < 1e-3, "doc {i}: batched {b} vs per-doc {p} diff {diff} too large");
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
        let ws = reranker.rerank_sync("q", &[doc("ws", "   ")]).expect("whitespace ok");
        assert_eq!(ws.len(), 1);
        // very long doc (forces truncation well beyond max_length)
        let long_text = "memory safety ".repeat(400);
        let lng = reranker.rerank_sync("rust", &[doc("long", &long_text)]).expect("long ok");
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
        let run1 = reranker.rerank_sync("what is the capital of france", &docs).expect("run1");
        let run2 = reranker.rerank_sync("what is the capital of france", &docs).expect("run2");
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
            assert_eq!(a.raw_logit, b.raw_logit, "parallel rerank must be deterministic");
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
