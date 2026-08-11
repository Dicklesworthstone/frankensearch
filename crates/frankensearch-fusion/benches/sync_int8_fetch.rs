//! End-to-end A/B: `SyncTwoTierSearcher` hybrid search with the fast-tier fetch
//! routed explicitly to the int8 two-pass control vs the exact f16 product
//! default. The approximate route is benchmark-only; production sync retrieval
//! remains exact regardless of optional cache state.
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
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::types::{BoundQueryEmbedding, TieredQueryEmbeddings};
use frankensearch_fusion::SyncTwoTierSearcher;
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};

/// Bind a synthetic bench vector to an explicitly synthetic identity, for both
/// tiers (this bench builds both from the same generated space).
fn tiered_query(vector: &[f32]) -> TieredQueryEmbeddings {
    let dimension = u32::try_from(vector.len()).expect("bench dimension fits u32");
    let bind = || {
        BoundQueryEmbedding::new(
            vector.to_vec(),
            EmbeddingIdentityBundleV1::explicit_test_model("bench-fixture", dimension),
        )
        .expect("bench query binds")
    };
    TieredQueryEmbeddings::progressive(bind(), bind())
}

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

    // The approximate arm is an explicit control. Product defaults stay exact.
    let int8 = SyncTwoTierSearcher::new(index.clone(), TwoTierConfig::default())
        .with_approximate_int8_fast_fetch_for_bench(3);
    let exact = SyncTwoTierSearcher::new(index.clone(), TwoTierConfig::default());

    // Bound ONCE, outside every timed region: binding a query to its identity
    // is setup, not part of the search being measured.
    let queries: Vec<TieredQueryEmbeddings> = (0..QUERIES)
        .map(|q| {
            tiered_query(&make_vector(
                &centroids,
                q % CLUSTERS,
                0xdead_0000 + q as u64,
            ))
        })
        .collect();

    let mut qi = 0usize;
    let mut g = c.benchmark_group("sync_int8_fetch");
    g.bench_function("explicit_approximate_int8_fetch", |b| {
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
