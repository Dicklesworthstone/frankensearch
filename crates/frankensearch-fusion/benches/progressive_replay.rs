//! bd-tjkm baseline: progressive-search query-path latency under a realistic
//! (Zipf) query-replay trace.
//!
//! `SyncTwoTierSearcher::search_collect` runs the two-tier hybrid: a fast-tier
//! candidate scan (4-bit two-pass) → quality-tier rescore → blend. The expected-loss
//! candidate-budget controller (bd-tjkm) must calibrate against measured per-query
//! latency, so this is the **measurement-only baseline** the bead's AC#1 requires —
//! it adds no behavior change, it emits p50/p95/p99 of the end-to-end query path plus
//! the phase-1/phase-2 split, replayed over a Zipf access pattern (a few hot queries +
//! a long cold tail, the realistic shape). No adaptive policy is introduced here; this
//! is the trace the controller will be measured against.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench progressive_replay
//! ```

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

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

const N: usize = 20_000;
const DIM: usize = 384;
const K: usize = 10;
const QUERIES: usize = 64;
const REPLAY: usize = 4_000;
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

/// Zipf-ish access trace over `QUERIES` distinct queries: index = Q·u^skew (skew>1
/// concentrates on the hot few). Deterministic xorshift, no rng dep.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the finite generated index is bounded to the configured query count"
)]
fn zipf_query_order(seed: u64) -> Vec<usize> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(REPLAY);
    for _ in 0..REPLAY {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = ((s >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        out.push(((QUERIES as f64) * u.powf(2.0)) as usize % QUERIES);
    }
    out
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the percentile index is finite, nonnegative, and clamped to the slice"
)]
fn percentile(sorted: &[u128], q: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_progressive_replay(c: &mut Criterion) {
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
    let searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default());

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
    let order = zipf_query_order(0xa11ce);

    // ── Baseline trace: per-query end-to-end latency percentiles + phase split. ──
    let mut samples: Vec<u128> = Vec::with_capacity(REPLAY);
    let mut p1_ms_sum = 0.0_f64;
    let mut p2_ms_sum = 0.0_f64;
    for &qi in &order {
        let t = Instant::now();
        let (results, metrics) = searcher.search_collect(&queries[qi], K).expect("search");
        samples.push(t.elapsed().as_nanos());
        black_box(results);
        p1_ms_sum += metrics.phase1_total_ms;
        p2_ms_sum += metrics.phase2_total_ms;
    }
    samples.sort_unstable();
    eprintln!(
        "[progressive_replay] N={N} dim={DIM} k={K} replay={REPLAY} zipf: \
         p50={:.1}us p95={:.1}us p99={:.1}us | phase1_mean={:.3}ms phase2_mean={:.3}ms",
        percentile(&samples, 0.50) as f64 / 1000.0,
        percentile(&samples, 0.95) as f64 / 1000.0,
        percentile(&samples, 0.99) as f64 / 1000.0,
        p1_ms_sum / REPLAY as f64,
        p2_ms_sum / REPLAY as f64,
    );

    // ── Criterion timing of one replayed query (stable throughput number). ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("progressive_replay");
    g.bench_function("search_collect_zipf", |b| {
        b.iter(|| {
            let q = &queries[order[qi % order.len()]];
            qi += 1;
            black_box(searcher.search_collect(black_box(q), K).expect("search"))
        });
    });
    g.finish();
}

criterion_group!(benches, bench_progressive_replay);
criterion_main!(benches);
