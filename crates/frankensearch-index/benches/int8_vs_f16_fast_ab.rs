//! Fast-tier scan A/B: exact **f16** `search_top_k` vs the lossless **int8 two-pass** on the SAME
//! F16 index — the sibling-consistency lever for `TwoTierIndex::search_fast_with_params`.
//!
//! The async `TwoTierSearcher` fast tier (`searcher.rs` → `search_fast_with_params`) runs the exact
//! f16 scan (`VectorIndex::search_top_k`) as a **reranked candidate generator** (its hits feed RRF +
//! graph + phase-1 corrections), whereas the SYNC searcher already switched the same tier to the int8
//! two-pass (`sync_searcher::search_fast_hits`, "use int8 two-pass instead of the exact f16 scan").
//! int8 two-pass is candidate-lossless on realistic embeddings (candidate-recall@k = 1.0 at mult=3),
//! so the fused top-k is identical — this bench PROVES that (asserts the top-k `doc_id` set matches the
//! exact f16 scan for every query) and measures the median speedup.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR \
//!   rch exec -- cargo bench -p frankensearch-index --bench int8_vs_f16_fast_ab
//! ```

use std::hint::black_box;

use criterion::Criterion;
use frankensearch_core::bench_support::{paired_median_ratio, print_bench_elf_sha256};
use frankensearch_index::{Quantization, VectorIndex};

const N: usize = 100_000;
const DIM: usize = 128;
const K: usize = 10;
const QUERIES: usize = 32;
const CLUSTERS: usize = 256;
const NOISE: f32 = 0.15; // tight/realistic semantic clusters (see hnsw_vs_flat_100k)
const FAST_TIER_MULT: usize = 3; // matches sync_searcher::search_fast_hits

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

fn bench(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|c| normalize(raw_vector(0xc000_0000 + c as u64)))
        .collect();

    let path = std::env::temp_dir().join(format!("fs_int8_f16_ab_{}.fsvi", std::process::id()));
    {
        let mut writer =
            VectorIndex::create_with_revision(&path, "hash", "bench", DIM, Quantization::F16)
                .expect("create f16 writer");
        for i in 0..N {
            let v = make_vector(&centroids, i % CLUSTERS, i as u64 + 1);
            writer
                .write_record(&format!("doc-{i:06}"), &v)
                .expect("write record");
        }
        writer.finish().expect("finish");
    }
    let index = VectorIndex::open(&path).expect("open index");

    let queries: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_vector(&centroids, q % CLUSTERS, 0xdead_0000 + q as u64))
        .collect();

    // RECALL / ORDERING PROOF: the int8 two-pass top-k doc_id SET must equal the exact f16 top-k
    // for every query (candidate-lossless), so the downstream RRF-fused output is unchanged.
    let mut recall_sum = 0.0_f64;
    let mut exact_match = 0usize;
    for q in &queries {
        let f16_ids: Vec<String> = index
            .search_top_k(q, K, None)
            .expect("f16 exact")
            .iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        let i8_ids: Vec<String> = index
            .search_top_k_int8_two_pass(q, K, FAST_TIER_MULT)
            .expect("int8 two-pass")
            .iter()
            .map(|h| h.doc_id.to_string())
            .collect();
        let hits = i8_ids.iter().filter(|id| f16_ids.contains(id)).count();
        recall_sum += hits as f64 / f16_ids.len().max(1) as f64;
        if f16_ids == i8_ids {
            exact_match += 1;
        }
    }
    let recall = recall_sum / QUERIES as f64;
    eprintln!(
        "[recall] int8_two_pass vs f16-exact: set-recall@{K}={recall:.4} exact-order-match={exact_match}/{QUERIES}"
    );
    assert!(
        (recall - 1.0).abs() < 1e-9,
        "int8 two-pass must be candidate-lossless vs f16 exact (got {recall:.4})"
    );

    // Closure factories: each call returns a FRESH runner (own rotation counter, `index`/`queries`
    // by copied reference) so the same arm can be handed to the null, lever, and criterion phases
    // without being moved-from.
    let index_ref = &index;
    let queries_ref = &queries;
    let mk_f16 = || {
        let mut i = 0usize;
        move || {
            let q = &queries_ref[i % QUERIES];
            i += 1;
            black_box(index_ref.search_top_k(black_box(q), K, None).expect("f16"));
        }
    };
    let mk_i8 = || {
        let mut i = 0usize;
        move || {
            let q = &queries_ref[i % QUERIES];
            i += 1;
            black_box(
                index_ref
                    .search_top_k_int8_two_pass(black_box(q), K, FAST_TIER_MULT)
                    .expect("int8"),
            );
        }
    };

    // NULL (f16 vs f16) then lever (f16=ORIG vs int8). Ratio = int8/ORIG, <1.0 = int8 faster.
    let null = paired_median_ratio(31, 3, mk_f16(), mk_f16());
    let lever = paired_median_ratio(31, 3, mk_f16(), mk_i8());
    eprintln!(
        "[null]  fast_scan: median {:.4} median_ci95 [{:.4}, {:.4}] \
         p5 {:.4} p95 {:.4} admissible={}",
        null.median,
        null.median_ci95_low,
        null.median_ci95_high,
        null.p5,
        null.p95,
        null.is_admissible_null(),
    );
    eprintln!(
        "[lever] fast_scan int8/f16 median {:.4} median_ci95 [{:.4}, {:.4}] \
         p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.median_ci95_low,
        lever.median_ci95_high,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN (int8 two-pass faster, candidate-lossless)"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR (not decidable)"
        }
    );

    {
        let mut g = c.benchmark_group("int8_vs_f16_fast");
        g.sample_size(30);
        g.bench_function("f16_exact", |b| b.iter(mk_f16()));
        g.bench_function("int8_two_pass", |b| b.iter(mk_i8()));
        g.finish();
    }

    eprintln!(
        "[artifact] retained index={} (repository policy forbids benchmark teardown deletion)",
        path.display()
    );
}

fn main() {
    let _identity = print_bench_elf_sha256().expect("hash executing top-k benchmark");
    let mut criterion = Criterion::default().configure_from_args();
    bench(&mut criterion);
    criterion.final_summary();
}
