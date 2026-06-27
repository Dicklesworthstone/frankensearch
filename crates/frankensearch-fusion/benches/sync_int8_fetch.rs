//! End-to-end A/B: `SyncTwoTierSearcher` hybrid search with the fast-tier fetch
//! routed to the int8 two-pass (default path) vs the exact f16 scan (explicit
//! `SearchParams`). The fast tier is a reranked candidate generator, so the int8
//! candidate set (recall=1.0) yields identical fused results — this measures the
//! hybrid-level speedup from the int8 fetch.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench sync_int8_fetch
//! ```

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::TwoTierConfig;
use frankensearch_fusion::SyncTwoTierSearcher;
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex, SearchParams};

const N: usize = 100_000;
const DIM: usize = 384;
const K: usize = 10;
const QUERIES: usize = 32;
const CLUSTERS: usize = 64;
const NOISE: f32 = 0.30;

fn raw_vector(seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut v = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        v.push((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
    }
    v
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn make_vector(centroids: &[Vec<f32>], c: usize, noise_seed: u64) -> Vec<f32> {
    let centroid = &centroids[c % centroids.len()];
    let noise = raw_vector(noise_seed);
    normalize(
        centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect(),
    )
}

fn bench_sync_int8_fetch(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    let fast_vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let quality_vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, 0xbeef_0000 + i as u64))
        .collect();
    let fast = InMemoryVectorIndex::from_vectors(ids.clone(), fast_vecs, DIM).expect("fast");
    let quality = InMemoryVectorIndex::from_vectors(ids, quality_vecs, DIM).expect("quality");
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));

    // Default path → int8 two-pass fast tier; explicit params → exact f16 scan.
    let int8 = SyncTwoTierSearcher::new(index.clone(), TwoTierConfig::default());
    let exact = SyncTwoTierSearcher::new(index.clone(), TwoTierConfig::default())
        .with_search_params(SearchParams::default());

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    let mut qi = 0usize;
    let mut g = c.benchmark_group("sync_int8_fetch");
    // Default path: the fast tier now uses the 4-bit two-pass (wired in
    // search_fast_hits); this arm measures the end-to-end 4-bit-fetch hybrid.
    g.bench_function("fast_fetch_4bit", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(int8.search_collect(black_box(q), K).expect("fast"))
        });
    });
    g.bench_function("exact_fetch", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(exact.search_collect(black_box(q), K).expect("exact"))
        });
    });
    g.finish();
}

criterion_group!(benches, bench_sync_int8_fetch);
criterion_main!(benches);
