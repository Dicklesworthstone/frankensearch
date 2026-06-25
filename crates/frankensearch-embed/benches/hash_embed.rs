//! Hash-embedder allocation-elision benchmark.
//!
//! `HashEmbedder::embed_sync` runs on every document at index time and every
//! query (the non-semantic `fnv1a-*` / `jl-*` fast tiers). The committed path did
//! two dimension-sized allocations per embed: `tokenize` collected a `Vec<&str>`,
//! and `l2_normalize` returned a freshly-allocated `Vec<f32>`. The new path
//! tokenizes lazily (iterator) and L2-normalizes the owned accumulator in place,
//! so the only allocation is the accumulator itself. This bench isolates that
//! head-to-head (the embed internals are private, so old/new are replicated here).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-embed --bench hash_embed
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
const MIN_TOKEN_LEN: usize = 2;
const DIM: usize = 384;
const JL_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── tokenize: old (materialized Vec) vs new (lazy iterator) ──────────────────
fn tokenize_vec(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= MIN_TOKEN_LEN)
        .collect()
}
fn tokenize_iter(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= MIN_TOKEN_LEN)
}

// ── L2 normalize: old (allocating collect) vs new (in place) ─────────────────
fn l2_norm_collect(vec: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        return vec![0.0; vec.len()];
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    vec.iter().map(|x| x * inv_norm).collect()
}
fn l2_norm_in_place(vec: &mut [f32]) {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq < f32::EPSILON {
        vec.iter_mut().for_each(|x| *x = 0.0);
        return;
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    vec.iter_mut().for_each(|x| *x *= inv_norm);
}

// ── FNV modular embed: old (2 allocs) vs new (1 alloc) ───────────────────────
fn fnv_old(text: &str) -> Vec<f32> {
    let tokens = tokenize_vec(text);
    let mut e = vec![0.0_f32; DIM];
    for token in &tokens {
        let hash = fnv1a_hash(token.as_bytes());
        let index = (hash as usize) % DIM;
        let sign = if (hash >> 63) == 1 { 1.0 } else { -1.0 };
        e[index] += sign;
    }
    l2_norm_collect(&e)
}
fn fnv_new(text: &str) -> Vec<f32> {
    let mut e = vec![0.0_f32; DIM];
    for token in tokenize_iter(text) {
        let hash = fnv1a_hash(token.as_bytes());
        let index = (hash as usize) % DIM;
        let sign = if (hash >> 63) == 1 { 1.0 } else { -1.0 };
        e[index] += sign;
    }
    l2_norm_in_place(&mut e);
    e
}

// ── JL projection embed: old (2 allocs) vs new (1 alloc) ─────────────────────
fn jl_old(text: &str) -> Vec<f32> {
    let tokens = tokenize_vec(text);
    let mut e = vec![0.0_f32; DIM];
    for token in &tokens {
        let hash = fnv1a_hash(token.as_bytes());
        let mut state = (JL_SEED ^ hash) | 1;
        for dim in &mut e {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = if (state & 1) == 0 { 1.0 } else { -1.0 };
            *dim += sign;
        }
    }
    l2_norm_collect(&e)
}
fn jl_new(text: &str) -> Vec<f32> {
    let mut e = vec![0.0_f32; DIM];
    for token in tokenize_iter(text) {
        let hash = fnv1a_hash(token.as_bytes());
        let mut state = (JL_SEED ^ hash) | 1;
        for dim in &mut e {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = if (state & 1) == 0 { 1.0 } else { -1.0 };
            *dim += sign;
        }
    }
    l2_norm_in_place(&mut e);
    e
}

fn bench_hash_embed(c: &mut Criterion) {
    // ~100-word document (typical chunk fed to the embedder).
    let doc = "the quick brown fox jumps over the lazy dog while the engineer \
               refactors a retry backoff loop and the parser tokenizes every \
               identifier in the source file before the index writer commits "
        .repeat(5);

    // Correctness sanity: old and new must be bit-identical.
    debug_assert_eq!(fnv_old(&doc), fnv_new(&doc));
    debug_assert_eq!(jl_old(&doc), jl_new(&doc));

    let mut fg = c.benchmark_group("hash_embed_fnv");
    fg.bench_with_input("old", doc.as_str(), |b, t| {
        b.iter(|| black_box(fnv_old(black_box(t))));
    });
    fg.bench_with_input("new", doc.as_str(), |b, t| {
        b.iter(|| black_box(fnv_new(black_box(t))));
    });
    fg.finish();

    let mut jg = c.benchmark_group("hash_embed_jl");
    jg.bench_with_input("old", doc.as_str(), |b, t| {
        b.iter(|| black_box(jl_old(black_box(t))));
    });
    jg.bench_with_input("new", doc.as_str(), |b, t| {
        b.iter(|| black_box(jl_new(black_box(t))));
    });
    jg.finish();
}

criterion_group!(benches, bench_hash_embed);
criterion_main!(benches);
