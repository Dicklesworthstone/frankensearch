//! RRF fusion hash-lookup benchmark.
//!
//! `rrf_fuse_with_graph` accumulates per-doc scores in an `AHashMap<&str, _>`.
//! The committed loops did a `get` (per-source dedup probe) **then** an `entry`
//! (insert/update) for every candidate — two hash lookups of the same key. The
//! new path consolidates to a single `entry` match. This bench isolates that
//! head-to-head over a realistic 1000-lexical + 1000-semantic candidate set with
//! ~50% doc overlap (the internal `FusedHitScratch` is private, so the old/new
//! accumulation is replicated here with an equivalently-shaped scratch struct).
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench rrf_fuse
//! ```

use std::collections::hash_map::Entry;
use std::hint::black_box;

use ahash::AHashMap;
use criterion::{Criterion, criterion_group, criterion_main};

const K: f64 = 60.0;

#[inline]
fn contribution(rank: usize) -> f64 {
    1.0 / (K + rank as f64 + 1.0)
}

/// Mirrors the shape/size of the real `FusedHitScratch`.
struct Scratch<'a> {
    doc_id: &'a str,
    rrf_score: f64,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    semantic_index: Option<usize>,
    lexical_score: Option<f32>,
    semantic_score: Option<f32>,
    in_both_sources: bool,
}

/// Old: `get` (dedup probe) + `entry` (update) — two hash lookups per candidate.
fn fuse_old(lexical: &[(String, f32)], semantic: &[(String, f32)]) -> f64 {
    let capacity = (lexical.len() + semantic.len()) * 3 / 4 + 1;
    let mut hits: AHashMap<&str, Scratch<'_>> = AHashMap::with_capacity(capacity);

    for (rank, (id, score)) in lexical.iter().enumerate() {
        if let Some(existing) = hits.get(id.as_str())
            && existing.lexical_rank.is_some()
        {
            continue;
        }
        let c = contribution(rank);
        hits.entry(id.as_str())
            .and_modify(|h| {
                h.rrf_score += c;
                h.lexical_rank = Some(rank);
                h.lexical_score = Some(*score);
                if h.semantic_rank.is_some() {
                    h.in_both_sources = true;
                }
            })
            .or_insert_with(|| Scratch {
                doc_id: id.as_str(),
                rrf_score: c,
                lexical_rank: Some(rank),
                semantic_rank: None,
                semantic_index: None,
                lexical_score: Some(*score),
                semantic_score: None,
                in_both_sources: false,
            });
    }

    for (rank, (id, score)) in semantic.iter().enumerate() {
        if let Some(existing) = hits.get(id.as_str())
            && existing.semantic_rank.is_some()
        {
            continue;
        }
        let c = contribution(rank);
        hits.entry(id.as_str())
            .and_modify(|h| {
                h.rrf_score += c;
                h.semantic_rank = Some(rank);
                h.semantic_score = Some(*score);
                h.semantic_index = Some(rank);
                if h.lexical_rank.is_some() {
                    h.in_both_sources = true;
                }
            })
            .or_insert_with(|| Scratch {
                doc_id: id.as_str(),
                rrf_score: c,
                lexical_rank: None,
                semantic_rank: Some(rank),
                semantic_index: Some(rank),
                lexical_score: None,
                semantic_score: Some(*score),
                in_both_sources: false,
            });
    }

    hits.values().map(|h| h.rrf_score).sum()
}

/// New: single `entry` match per candidate — one hash lookup.
fn fuse_new(lexical: &[(String, f32)], semantic: &[(String, f32)]) -> f64 {
    let capacity = (lexical.len() + semantic.len()) * 3 / 4 + 1;
    let mut hits: AHashMap<&str, Scratch<'_>> = AHashMap::with_capacity(capacity);

    for (rank, (id, score)) in lexical.iter().enumerate() {
        let c = contribution(rank);
        match hits.entry(id.as_str()) {
            Entry::Occupied(mut e) => {
                let h = e.get_mut();
                if h.lexical_rank.is_some() {
                    continue;
                }
                h.rrf_score += c;
                h.lexical_rank = Some(rank);
                h.lexical_score = Some(*score);
                if h.semantic_rank.is_some() {
                    h.in_both_sources = true;
                }
            }
            Entry::Vacant(e) => {
                e.insert(Scratch {
                    doc_id: id.as_str(),
                    rrf_score: c,
                    lexical_rank: Some(rank),
                    semantic_rank: None,
                    semantic_index: None,
                    lexical_score: Some(*score),
                    semantic_score: None,
                    in_both_sources: false,
                });
            }
        }
    }

    for (rank, (id, score)) in semantic.iter().enumerate() {
        let c = contribution(rank);
        match hits.entry(id.as_str()) {
            Entry::Occupied(mut e) => {
                let h = e.get_mut();
                if h.semantic_rank.is_some() {
                    continue;
                }
                h.rrf_score += c;
                h.semantic_rank = Some(rank);
                h.semantic_score = Some(*score);
                h.semantic_index = Some(rank);
                if h.lexical_rank.is_some() {
                    h.in_both_sources = true;
                }
            }
            Entry::Vacant(e) => {
                e.insert(Scratch {
                    doc_id: id.as_str(),
                    rrf_score: c,
                    lexical_rank: None,
                    semantic_rank: Some(rank),
                    semantic_index: Some(rank),
                    lexical_score: None,
                    semantic_score: Some(*score),
                    in_both_sources: false,
                });
            }
        }
    }

    hits.values().map(|h| h.rrf_score).sum()
}

fn bench_rrf_fuse(c: &mut Criterion) {
    // 1000 lexical + 1000 semantic with ~50% doc overlap (docs 500..1499 shared).
    let lexical: Vec<(String, f32)> = (0..1000)
        .map(|i| (format!("doc{i:06}"), 1.0 / (i as f32 + 1.0)))
        .collect();
    let semantic: Vec<(String, f32)> = (500..1500)
        .map(|i| (format!("doc{i:06}"), 1.0 / (i as f32 + 1.0)))
        .collect();

    // Sanity: both produce the same aggregate score sum.
    debug_assert!((fuse_old(&lexical, &semantic) - fuse_new(&lexical, &semantic)).abs() < 1e-9);

    let mut g = c.benchmark_group("rrf_fuse");
    g.bench_function("old", |b| {
        b.iter(|| black_box(fuse_old(black_box(&lexical), black_box(&semantic))));
    });
    g.bench_function("new", |b| {
        b.iter(|| black_box(fuse_new(black_box(&lexical), black_box(&semantic))));
    });
    g.finish();
}

criterion_group!(benches, bench_rrf_fuse);
criterion_main!(benches);
