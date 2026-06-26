//! HNSW (ANN) vs flat brute-force vector search — lever validation.
//!
//! The default vector search is a flat O(N) cosine scan (`VectorIndex::search_top_k`,
//! rayon-parallel). `HnswIndex` (behind the `ann` feature, **not** wired into the
//! default search path) is an approximate O(log N) graph index. This bench
//! quantifies the lever: HNSW `knn_search` latency vs flat `search_top_k`, plus the
//! **recall@10** of HNSW vs the exact flat top-10 (printed to stderr — HNSW is only a
//! win if recall is high). Validation only; nothing here is wired into product code.
//!
//! Run with (the `ann` feature is required):
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-index --features ann --bench hnsw_vs_flat
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "ann")]
fn bench_hnsw_vs_flat(c: &mut Criterion) {
    use std::hint::black_box;

    use frankensearch_index::{
        HNSW_DEFAULT_EF_SEARCH, HnswConfig, HnswIndex, Quantization, VectorIndex,
    };

    const N: usize = 10_000;
    const DIM: usize = 128;
    const K: usize = 10;
    const QUERIES: usize = 32;

    const CLUSTERS: usize = 64;
    const NOISE: f32 = 0.30;

    /// Deterministic xorshift64-based uniform-random raw vector (unnormalized).
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

    /// **Clustered** vector near centroid `c` plus small noise — mimics real
    /// embedding structure (semantically-similar docs cluster), which is what HNSW
    /// graph navigation exploits. Uniform-random vectors are a worst case for ANN
    /// recall; clustered data is the realistic case.
    fn make_vector(centroids: &[Vec<f32>], c: usize, noise_seed: u64) -> Vec<f32> {
        let centroid = &centroids[c % centroids.len()];
        let noise = raw_vector(noise_seed);
        let v: Vec<f32> = centroid
            .iter()
            .zip(&noise)
            .map(|(a, n)| a + NOISE * n)
            .collect();
        normalize(v)
    }

    fn recall_at_k(flat: &[String], ann: &[String]) -> f64 {
        let hits = ann.iter().filter(|id| flat.contains(id)).count();
        hits as f64 / flat.len().max(1) as f64
    }

    // Cluster centroids (the latent structure HNSW navigation exploits).
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|c| normalize(raw_vector(0xc000_0000 + c as u64)))
        .collect();

    // ── Build the FSVI index + HNSW once (setup, not measured). ──
    let path = std::env::temp_dir().join(format!("fs_hnsw_bench_{}.fsvi", std::process::id()));
    {
        let mut writer =
            VectorIndex::create_with_revision(&path, "hash", "bench", DIM, Quantization::F32)
                .expect("create writer");
        for i in 0..N {
            let v = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
            writer
                .write_record(&format!("doc-{i:06}"), &v)
                .expect("write record");
        }
        writer.finish().expect("finish");
    }
    let index = VectorIndex::open(&path).expect("open index");
    let hnsw =
        HnswIndex::build_from_vector_index(&index, HnswConfig::default()).expect("build hnsw");

    // Queries land near a cluster (so they have genuine near-neighbors), like real
    // queries against a semantically-clustered corpus.
    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // Exact flat top-K per query (recall reference).
    let flat_topk: Vec<Vec<String>> = queries
        .iter()
        .map(|q| {
            index
                .search_top_k(q, K, None)
                .expect("flat")
                .into_iter()
                .map(|h| h.doc_id)
                .collect()
        })
        .collect();

    // ── Recall@K sweep over ef_search (HNSW top-K vs exact flat top-K). ──
    const EF_SWEEP: [usize; 5] = [10, 20, 40, 100, 200];
    for ef in EF_SWEEP {
        let mut total = 0.0;
        for (qi, query) in queries.iter().enumerate() {
            let ann: Vec<String> = hnsw
                .knn_search(query, K, ef)
                .expect("ann")
                .into_iter()
                .map(|h| h.doc_id)
                .collect();
            total += recall_at_k(&flat_topk[qi], &ann);
        }
        eprintln!(
            "[hnsw_vs_flat] N={N} dim={DIM} k={K} ef_search={ef} recall@{K}={:.4}",
            total / QUERIES as f64
        );
    }

    // ── Latency: flat vs HNSW at a low and the default ef (cycle queries). ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("hnsw_vs_flat");
    g.bench_function("flat", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(index.search_top_k(black_box(q), K, None).expect("flat"))
        });
    });
    for ef in [10usize, 20, 40, HNSW_DEFAULT_EF_SEARCH] {
        g.bench_function(format!("hnsw_ef{ef}"), |b| {
            b.iter(|| {
                let q = &queries[qi % QUERIES];
                qi += 1;
                black_box(hnsw.knn_search(black_box(q), K, ef).expect("ann"))
            });
        });
    }
    g.finish();

    let _ = std::fs::remove_file(&path);
}

#[cfg(not(feature = "ann"))]
fn bench_hnsw_vs_flat(_c: &mut Criterion) {
    // HNSW lives behind the `ann` feature; build with `--features ann` to run it.
}

criterion_group!(benches, bench_hnsw_vs_flat);
criterion_main!(benches);
