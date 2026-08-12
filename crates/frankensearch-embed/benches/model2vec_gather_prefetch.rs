//! `Model2Vec` embed GATHER software-prefetch A/B.
//!
//! `Model2VecEmbedder::embed_sync` mean-pools token vectors: for each token id it
//! indexes a random row `emb[id*DIM .. id*DIM+DIM]` of the `[vocab, DIM]` static
//! table and AVX2-accumulates it into `sum`. For a real model (potion-base-8M:
//! vocab ~30k, DIM 256 → a ~30 MB table) each token's row is a RANDOM location that
//! the hardware stride-prefetcher cannot predict → a first-touch cache miss per
//! token. The accumulate itself is cheap (DIM/8 AVX2 adds), so the loop is
//! MEMORY-LATENCY-bound on the indirect gather — the same shape as the CLS-attention
//! prefetch win, but on an indirect scatter instead of a fixed stride.
//!
//! The whole token-id sequence is known upfront, so `emb`'s row for token `i+PF` can
//! be `_mm_prefetch`'d while token `i`'s row accumulates — hiding the miss latency.
//! Prefetch is a hint → the accumulated `sum` is BIT-IDENTICAL to the base loop
//! (parity asserts max-delta 0), so this is exact and distribution-independent in
//! correctness.
//!
//! Historical Criterion arms retain `base`, `pf_head`, and `pf_row`. The shipping
//! gate additionally compares the exact former production loop with
//! `accumulate_model2vec_rows`: native-256 rows are fused in ordered groups of at
//! most four below 512 tokens, while the existing full-row prefetch remains at
//! 512+. Mean-scaling plus L2 normalization are included. Each timed arm traverses
//! a full 30 MB table copy so repeated sampling cannot turn the long-document
//! workload into an L2-resident microbenchmark.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo bench -j 4 \
//!   -p frankensearch-embed --features model2vec,bench-internals \
//!   --profile release --bench model2vec_gather_prefetch -- --noplot
//! ```

use std::hint::black_box;
use std::time::Duration;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_embed::simd::{accumulate_f32_into, accumulate_model2vec_rows};

const VOCAB: usize = 30_000; // potion-base-8M-class vocab
const DIM: usize = 256; // potion-base-8M dimension → 30k*256*4 ≈ 30 MB table
const PF: usize = 4; // prefetch distance (tokens ahead)
const LINE_F32: usize = 16; // 64-byte cache line = 16 f32

/// Prefetch the row starting at `row_start` (f32 offset). `full` = every cache line
/// of the DIM-wide row; else just the first line (HW streams the rest).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
fn prefetch_row(emb: &[f32], row_start: usize, full: bool) {
    // SAFETY: _mm_prefetch is a hint; any address is architecturally valid to
    // prefetch and we bound the offset by the slice length anyway.
    unsafe {
        if !full {
            if row_start < emb.len() {
                _mm_prefetch(emb.as_ptr().add(row_start).cast::<i8>(), _MM_HINT_T0);
            }
            return;
        }
        let mut off = row_start;
        let end = (row_start + DIM).min(emb.len());
        while off < end {
            _mm_prefetch(emb.as_ptr().add(off).cast::<i8>(), _MM_HINT_T0);
            off += LINE_F32;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn prefetch_row(_emb: &[f32], _row_start: usize, _full: bool) {}

/// ORIGINAL: gather each token's row and accumulate, no prefetch.
fn gather_base(emb: &[f32], ids: &[u32], sum: &mut [f32]) -> usize {
    sum.fill(0.0);
    let mut count = 0_usize;
    for &id in ids {
        let idx = id as usize;
        if idx < VOCAB {
            let start = idx * DIM;
            accumulate_f32_into(sum, &emb[start..start + DIM]);
            count += 1;
        }
    }
    count
}

/// CANDIDATE: prefetch token `i+PF`'s row while accumulating token `i`.
fn gather_prefetch(emb: &[f32], ids: &[u32], sum: &mut [f32], full: bool) {
    sum.fill(0.0);
    let n = ids.len();
    for i in 0..n {
        if i + PF < n {
            let pidx = ids[i + PF] as usize;
            if pidx < VOCAB {
                prefetch_row(emb, pidx * DIM, full);
            }
        }
        let idx = ids[i] as usize;
        if idx < VOCAB {
            let start = idx * DIM;
            accumulate_f32_into(sum, &emb[start..start + DIM]);
        }
    }
}

fn gather_gated(emb: &[f32], ids: &[u32], sum: &mut [f32]) -> usize {
    sum.fill(0.0);
    accumulate_model2vec_rows(sum, emb, ids, VOCAB)
}

fn finish_mean_pool(sum: &mut [f32], count: usize) {
    assert!(
        count > 0,
        "benchmark corpus must contain in-vocabulary rows"
    );
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / count as f32;
    for value in sum.iter_mut() {
        *value *= inv;
    }
    let norm_sq: f32 = sum.iter().map(|value| value * value).sum();
    if norm_sq.is_finite() && norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for value in sum.iter_mut() {
            *value *= inv_norm;
        }
    } else {
        sum.fill(0.0);
    }
}

#[derive(Clone, Copy)]
enum Arm {
    Original,
    ShippingCandidate,
}

fn run_corpus(emb: &[f32], ids: &[u32], tokens_per_doc: usize, sum: &mut [f32], arm: Arm) {
    for doc_ids in ids.chunks_exact(tokens_per_doc) {
        let count = match arm {
            Arm::Original => gather_base(emb, doc_ids, sum),
            Arm::ShippingCandidate => gather_gated(emb, doc_ids, sum),
        };
        finish_mean_pool(sum, count);
        black_box(&*sum);
    }
}

fn corpus_ids(tokens_per_doc: usize) -> Vec<u32> {
    let docs = VOCAB.div_ceil(tokens_per_doc);
    (0..docs * tokens_per_doc)
        .map(|position| u32::try_from((position * 7_919) % VOCAB).expect("VOCAB fits in u32"))
        .collect()
}

fn paired_shipping_gate(emb_original: &[f32], emb_candidate: &[f32]) {
    for &tokens_per_doc in &[1_usize, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512] {
        let ids = corpus_ids(tokens_per_doc);

        let mut parity_original = vec![0.0_f32; DIM];
        let mut parity_candidate = vec![0.0_f32; DIM];
        run_corpus(
            emb_original,
            &ids,
            tokens_per_doc,
            &mut parity_original,
            Arm::Original,
        );
        run_corpus(
            emb_original,
            &ids,
            tokens_per_doc,
            &mut parity_candidate,
            Arm::ShippingCandidate,
        );
        assert_eq!(
            parity_original
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            parity_candidate
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "gated production path changed pooled output at {tokens_per_doc} tokens"
        );

        let mut null_original = vec![0.0_f32; DIM];
        let mut null_clone = vec![0.0_f32; DIM];
        let null = paired_median_ratio(
            31,
            1,
            || {
                run_corpus(
                    black_box(emb_original),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut null_original),
                    Arm::Original,
                );
            },
            || {
                run_corpus(
                    black_box(emb_candidate),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut null_clone),
                    Arm::Original,
                );
            },
        );

        let mut lever_original = vec![0.0_f32; DIM];
        let mut lever_candidate = vec![0.0_f32; DIM];
        let lever = paired_median_ratio(
            31,
            1,
            || {
                run_corpus(
                    black_box(emb_original),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut lever_original),
                    Arm::Original,
                );
            },
            || {
                run_corpus(
                    black_box(emb_candidate),
                    black_box(&ids),
                    tokens_per_doc,
                    black_box(&mut lever_candidate),
                    Arm::ShippingCandidate,
                );
            },
        );
        let verdict = if tokens_per_doc < 512 {
            if lever.median > null.p95 {
                "SHORT_REGRESSION"
            } else {
                "SHORT_PRESERVED"
            }
        } else if lever.median < null.p5 {
            "LONG_WIN"
        } else {
            "INSIDE_NULL_FLOOR"
        };
        eprintln!(
            "[paired-gate] tokens={tokens_per_doc} null median={:.6} p5={:.6} p95={:.6} gated/original median={:.6} p5={:.6} p95={:.6} verdict={verdict}",
            null.median, null.p5, null.p95, lever.median, lever.p5, lever.p95,
        );
    }
}

fn emb_fixture() -> Vec<f32> {
    let mut out = vec![0.0f32; VOCAB * DIM];
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    for v in &mut out {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
    }
    out
}

fn ids_fixture(t: usize) -> Vec<u32> {
    // Uniform-random token ids = the cache-cold regime (broad-vocab doc embedding
    // over a 30 MB table). Realistic Zipfian text keeps hot tokens cached and would
    // see a smaller gather benefit; this is the gather-heavy case.
    let mut out = Vec::with_capacity(t);
    let mut s = 0x1234_5678_9abc_def0_u64;
    for _ in 0..t {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push(u32::try_from((s >> 33) % 30_000).expect("modulus fits in u32"));
    }
    out
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("model2vec_gather_prefetch");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_millis(1000));

    let emb = emb_fixture();
    let emb_candidate = emb.clone();
    paired_shipping_gate(&emb, &emb_candidate);

    for &t in &[16usize, 64, 256] {
        let ids = ids_fixture(t);

        // Parity: prefetch is a hint → sum is bit-identical to the base loop.
        let mut a = vec![0.0f32; DIM];
        let mut bh = vec![0.0f32; DIM];
        let mut br = vec![0.0f32; DIM];
        let _ = gather_base(&emb, &ids, &mut a);
        gather_prefetch(&emb, &ids, &mut bh, false);
        gather_prefetch(&emb, &ids, &mut br, true);
        let dh = a
            .iter()
            .zip(&bh)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let dr = a
            .iter()
            .zip(&br)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            dh == 0.0,
            "pf_head diverged from base by {dh} (t={t}) — must be bit-identical"
        );
        assert!(
            dr == 0.0,
            "pf_row diverged from base by {dr} (t={t}) — must be bit-identical"
        );

        group.bench_with_input(BenchmarkId::new("base", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                let _ = gather_base(black_box(&emb), black_box(ids), black_box(&mut sum));
                black_box(&sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("pf_head", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_prefetch(black_box(&emb), black_box(ids), black_box(&mut sum), false);
                black_box(&sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("pf_row", t), &ids, |bn, ids| {
            let mut sum = vec![0.0f32; DIM];
            bn.iter(|| {
                gather_prefetch(black_box(&emb), black_box(ids), black_box(&mut sum), true);
                black_box(&sum);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
