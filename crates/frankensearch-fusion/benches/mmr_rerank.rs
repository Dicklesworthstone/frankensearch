//! MMR re-rank inner-loop benchmark.
//!
//! `mmr_rerank` greedily selects `k` diverse documents. The committed loop did
//! two redundant things in its O(k·n) candidate sweep:
//!   1. it recomputed `max(sim(i, j) for j in selected)` from scratch every
//!      round — O(k²·n) cosine evaluations; and
//!   2. each `cosine_sim` re-derived both vectors' L2 norms on every pair, even
//!      though an embedding's norm is loop-invariant — 3 reductions per pair.
//!
//! The new path keeps a running max similarity (updated once per selection,
//! O(k·n)) and hoists the norms (1 dot reduction per pair). Both changes are
//! bit-identical for uniform-dimension embeddings, so the selection is unchanged.
//! This bench replicates old vs new self-contained (the selection internals are
//! private) over realistic candidate-pool / k / dim sizes.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench mmr_rerank
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const LAMBDA: f64 = 0.55;

// ── cosine: per-pair (old) vs precomputed-norm (new) ─────────────────────────
fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..len {
        let ai = f64::from(a[i]);
        let bi = f64::from(b[i]);
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }
    dot / denom
}

fn cosine_sim_pre(a: &[f32], b: &[f32], ra: f64, rb: f64) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    for i in 0..len {
        dot += f64::from(a[i]) * f64::from(b[i]);
    }
    let denom = ra * rb;
    if denom < f64::EPSILON {
        return 0.0;
    }
    dot / denom
}

fn norm_scores(scores: &[f64]) -> Vec<f64> {
    let (mn, mx) = scores.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(mn, mx), &s| (mn.min(s), mx.max(s)),
    );
    let range = mx - mn;
    scores
        .iter()
        .map(|&s| if range < f64::EPSILON { 1.0 } else { (s - mn) / range })
        .collect()
}

// Old: recompute max over the whole selected set each round, per-pair norms.
fn mmr_old(scores: &[f64], emb: &[&[f32]], k: usize) -> Vec<usize> {
    let n = scores.len();
    let ns = norm_scores(scores);
    let dw = 1.0 - LAMBDA;
    let mut selected = Vec::with_capacity(k);
    let mut remaining = vec![true; n];
    let first = ns
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bs), (i, &s)| if s > bs { (i, s) } else { (bi, bs) })
        .0;
    selected.push(first);
    remaining[first] = false;
    for _ in 1..k {
        let mut best_idx = usize::MAX;
        let mut best = f64::NEG_INFINITY;
        for i in 0..n {
            if !remaining[i] {
                continue;
            }
            let max_sim = selected
                .iter()
                .map(|&j| cosine_sim(emb[i], emb[j]))
                .fold(f64::NEG_INFINITY, f64::max);
            let mmr = LAMBDA.mul_add(ns[i], -(dw * max_sim));
            if mmr > best {
                best = mmr;
                best_idx = i;
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        selected.push(best_idx);
        remaining[best_idx] = false;
    }
    selected
}

// New: running max similarity + hoisted norms.
fn mmr_new(scores: &[f64], emb: &[&[f32]], k: usize) -> Vec<usize> {
    let n = scores.len();
    let ns = norm_scores(scores);
    let dw = 1.0 - LAMBDA;
    let roots: Vec<f64> = emb
        .iter()
        .map(|e| {
            let mut s = 0.0_f64;
            for &x in *e {
                let x = f64::from(x);
                s += x * x;
            }
            s.sqrt()
        })
        .collect();
    let sim = |i: usize, j: usize| cosine_sim_pre(emb[i], emb[j], roots[i], roots[j]);
    let mut selected = Vec::with_capacity(k);
    let mut remaining = vec![true; n];
    let first = ns
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bs), (i, &s)| if s > bs { (i, s) } else { (bi, bs) })
        .0;
    selected.push(first);
    remaining[first] = false;
    let mut max_sim = vec![f64::NEG_INFINITY; n];
    for i in 0..n {
        if remaining[i] {
            max_sim[i] = sim(i, first);
        }
    }
    for _ in 1..k {
        let mut best_idx = usize::MAX;
        let mut best = f64::NEG_INFINITY;
        for i in 0..n {
            if !remaining[i] {
                continue;
            }
            let mmr = LAMBDA.mul_add(ns[i], -(dw * max_sim[i]));
            if mmr > best {
                best = mmr;
                best_idx = i;
            }
        }
        if best_idx == usize::MAX {
            break;
        }
        selected.push(best_idx);
        remaining[best_idx] = false;
        for i in 0..n {
            if remaining[i] {
                let s = sim(i, best_idx);
                if s > max_sim[i] {
                    max_sim[i] = s;
                }
            }
        }
    }
    selected
}

fn make_inputs(n: usize, dim: usize) -> (Vec<f64>, Vec<Vec<f32>>) {
    let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ (dim as u64).wrapping_mul(n as u64 + 1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let emb: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect();
    let scores: Vec<f64> = (0..n).map(|i| (i as f64 * 7.0 % 13.0) / 13.0).collect();
    (scores, emb)
}

fn bench_mmr(c: &mut Criterion) {
    let dim = 384;
    // (candidate pool n, results k): realistic MMR diversity-rerank shapes.
    let cases = [(100usize, 20usize), (200, 50)];

    let mut g = c.benchmark_group("mmr_rerank");
    for (n, k) in cases {
        let (scores, emb) = make_inputs(n, dim);
        let refs: Vec<&[f32]> = emb.iter().map(Vec::as_slice).collect();
        // Correctness: identical selection.
        debug_assert_eq!(mmr_old(&scores, &refs, k), mmr_new(&scores, &refs, k));

        let id = format!("n{n}_k{k}_d{dim}");
        g.bench_with_input(BenchmarkId::new("old", &id), &(), |b, ()| {
            b.iter(|| black_box(mmr_old(black_box(&scores), black_box(&refs), k)));
        });
        g.bench_with_input(BenchmarkId::new("new", &id), &(), |b, ()| {
            b.iter(|| black_box(mmr_new(black_box(&scores), black_box(&refs), k)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_mmr);
criterion_main!(benches);
