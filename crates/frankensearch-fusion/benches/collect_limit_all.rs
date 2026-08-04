//! `limit_all` end-to-end A/B for the discarded-phase clone elision.
//!
//! `SyncTwoTierSearcher::search_collect` returns only `(final_results, metrics)`
//! and throws away `outcome.phases` — yet `search_internal` used to build those
//! phases by cloning the full `Vec<ScoredResult>` (N owned `doc_id`s each) once per
//! phase (Initial + Refined). At `limit_all` (k = N) that is up to 2·N wasted
//! short-`String` allocations per query. `search_collect` now passes
//! `want_phases = false` and skips them; only `search_iter` (the streaming API)
//! still builds phases.
//!
//! This measures the exact end-to-end delta using only public APIs on the same
//! searcher over the same queries:
//!
//! - `collect` (new): `search_collect` — no phase clones.
//! - `iter` (old): `search_iter` — builds both phases (the pre-change
//!   `search_collect` behavior); drained so nothing is elided.
//!
//! Both run identical search work (fast scan → quality rescore → blend), so the
//! ratio is purely the phase-clone cost. `final_results` are unchanged.
//!
//! Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-fusion --bench collect_limit_all
//! ```

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::types::{BoundQueryEmbedding, TieredQueryEmbeddings};
use frankensearch_core::{
    ScoreSource, ScoredResult, SearchPhase, SearchResult, TwoTierConfig, VectorHit,
};
use frankensearch_fusion::rrf::{
    RrfConfig, RrfTiebreak, rrf_fuse, rrf_fuse_with_graph_merge_unique,
};
use frankensearch_fusion::{SyncLexicalSearch, SyncTwoTierSearcher, blend_two_tier_aligned};
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex};

/// Bind a synthetic bench vector to an explicitly synthetic identity, for both
/// tiers (this bench builds both from the same generated space).
fn tiered_query(vector: Vec<f32>) -> TieredQueryEmbeddings {
    let dimension = u32::try_from(vector.len()).expect("bench dimension fits u32");
    let bind = || {
        BoundQueryEmbedding::new(
            vector.clone(),
            EmbeddingIdentityBundleV1::explicit_test_model("bench-fixture", dimension),
        )
        .expect("bench query binds")
    };
    TieredQueryEmbeddings::progressive(bind(), bind())
}

const N: usize = 10_000;
const DIM: usize = 384;
const QUERIES: usize = 16;
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

fn lexical_result(doc_id: &str, score: f32) -> ScoredResult {
    ScoredResult {
        doc_id: doc_id.into(),
        score,
        source: ScoreSource::Lexical,
        index: None,
        fast_score: None,
        quality_score: None,
        lexical_score: Some(score),
        rerank_score: None,
        explanation: None,
        metadata: None,
    }
}

fn make_lexical_hits(ids: &[String]) -> Vec<ScoredResult> {
    ids.iter()
        .step_by(5)
        .enumerate()
        .map(|(rank, id)| lexical_result(id, (N - rank) as f32))
        .collect()
}

fn make_ranked_vector_hits(ids: &[String]) -> Vec<VectorHit> {
    ids.iter()
        .enumerate()
        .map(|(i, id)| VectorHit {
            index: u32::try_from(i).expect("bench N fits in u32"),
            score: 1.0 - (i as f32 / N as f32),
            doc_id: id.clone().into(),
        })
        .collect()
}

fn make_quality_scores() -> Vec<Option<f32>> {
    (0..N).map(|i| Some(1.0 - (i as f32 / N as f32))).collect()
}

struct StaticLexical {
    hits: Vec<ScoredResult>,
}

impl SyncLexicalSearch for StaticLexical {
    fn search_sync(&self, _query_vec: &[f32], limit: usize) -> SearchResult<Vec<ScoredResult>> {
        Ok(self.hits.iter().take(limit).cloned().collect())
    }
}

fn bench_collect_limit_all(c: &mut Criterion) {
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|i| normalize(raw_vector(0xc000_0000 + i as u64)))
        .collect();
    let ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    let lexical_hits = make_lexical_hits(&ids);
    let fast_vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, i as u64 + 1))
        .collect();
    let quality_vecs: Vec<Vec<f32>> = (0..N)
        .map(|i| make_vector(&centroids, i % CLUSTERS, 0xbeef_0000 + i as u64))
        .collect();
    let fast = InMemoryVectorIndex::from_vectors(ids.clone(), fast_vecs, DIM).expect("fast");
    let quality = InMemoryVectorIndex::from_vectors(ids, quality_vecs, DIM).expect("quality");
    let index = Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)));
    let searcher = SyncTwoTierSearcher::new(Arc::clone(&index), TwoTierConfig::default());
    let lexical_searcher = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
        .with_lexical(Arc::new(StaticLexical { hits: lexical_hits }));

    // Bound ONCE, outside every timed region: binding a query to its identity
    // is setup, not part of the search being measured.
    let queries: Vec<TieredQueryEmbeddings> = (0..QUERIES)
        .map(|q| {
            tiered_query(make_vector(
                &centroids,
                q % CLUSTERS,
                0xdead_0000 + q as u64,
            ))
        })
        .collect();

    // Sanity: both APIs agree on the final ranking (identical doc_id order).
    let collect_ids: Vec<String> = searcher
        .search_collect(&queries[0], N)
        .expect("collect")
        .0
        .iter()
        .map(|r| r.doc_id.to_string())
        .collect();
    let iter_final = searcher
        .search_iter(&queries[0], N)
        .last()
        .expect("iter has a final phase");
    let iter_results = match &iter_final {
        SearchPhase::Initial { results, .. }
        | SearchPhase::Refined { results, .. }
        | SearchPhase::Reranked { results, .. } => results,
        SearchPhase::RefinementFailed {
            initial_results, ..
        } => initial_results,
    };
    let iter_ids: Vec<String> = iter_results.iter().map(|r| r.doc_id.to_string()).collect();
    assert_eq!(
        collect_ids, iter_ids,
        "collect vs iter final ranking differs"
    );
    let (lexical_results, _) = lexical_searcher
        .search_collect(&queries[0], N)
        .expect("lexical collect");
    assert_eq!(lexical_results.len(), N, "lexical collect should return k");

    let mut qi = 0usize;
    let mut g = c.benchmark_group("collect_limit_all");
    // New: search_collect skips phase construction (no discarded clones).
    g.bench_function("collect", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(searcher.search_collect(black_box(q), N).expect("collect"))
        });
    });
    // Old-equivalent: search_iter builds both phases (the pre-change collect cost).
    g.bench_function("iter", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(searcher.search_iter(black_box(q), N).count())
        });
    });
    g.bench_function("lexical_collect", |b| {
        b.iter(|| {
            let q = &queries[qi % QUERIES];
            qi += 1;
            black_box(
                lexical_searcher
                    .search_collect(black_box(q), N)
                    .expect("lexical collect"),
            )
        });
    });
    g.finish();
}

fn bench_refined_rrf_callsite(c: &mut Criterion) {
    let ids: Vec<String> = (0..N).map(|i| format!("doc-{i:06}")).collect();
    let lexical_hits = make_lexical_hits(&ids);
    let fast_hits = make_ranked_vector_hits(&ids);
    let quality_scores = make_quality_scores();
    let blended = blend_two_tier_aligned(&fast_hits, &quality_scores, 0.7);
    let config = RrfConfig {
        k: TwoTierConfig::default().rrf_k,
        lexical_weight: 1.0,
        semantic_weight: 1.0,
        tiebreak: RrfTiebreak::LexicalThenId,
    };

    let legacy = rrf_fuse(&lexical_hits, &blended, N, 0, &config);
    let unique = rrf_fuse_with_graph_merge_unique(&lexical_hits, &blended, &[], 0.0, N, 0, &config);
    assert_eq!(
        legacy.len(),
        unique.len(),
        "legacy and unique fused lengths"
    );
    for (legacy_hit, unique_hit) in legacy.iter().zip(&unique) {
        assert_eq!(legacy_hit.doc_id, unique_hit.doc_id);
        assert_eq!(
            legacy_hit.rrf_score.to_bits(),
            unique_hit.rrf_score.to_bits()
        );
        assert_eq!(legacy_hit.lexical_rank, unique_hit.lexical_rank);
        assert_eq!(legacy_hit.semantic_rank, unique_hit.semantic_rank);
        assert_eq!(legacy_hit.semantic_index, unique_hit.semantic_index);
        assert_eq!(legacy_hit.lexical_score, unique_hit.lexical_score);
        assert_eq!(legacy_hit.semantic_score, unique_hit.semantic_score);
        assert_eq!(legacy_hit.in_both_sources, unique_hit.in_both_sources);
    }

    let mut g = c.benchmark_group("refined_rrf_callsite");
    g.bench_function("dedup", |b| {
        b.iter(|| {
            black_box(rrf_fuse(
                black_box(&lexical_hits),
                black_box(&blended),
                N,
                0,
                black_box(&config),
            ))
        });
    });
    g.bench_function("unique", |b| {
        b.iter(|| {
            black_box(rrf_fuse_with_graph_merge_unique(
                black_box(&lexical_hits),
                black_box(&blended),
                black_box(&[]),
                black_box(0.0),
                N,
                0,
                black_box(&config),
            ))
        });
    });
    g.finish();
}

criterion_group!(benches, bench_collect_limit_all, bench_refined_rrf_callsite);
criterion_main!(benches);
