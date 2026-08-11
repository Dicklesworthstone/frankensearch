//! Integration tests for synchronous two-tier search orchestration.

use std::sync::Arc;

use frankensearch_core::filter::SearchFilter;
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::types::{BoundQueryEmbedding, TieredQueryEmbeddings};
use frankensearch_core::{ScoreSource, SearchError, SearchPhase, TwoTierConfig};
use frankensearch_fusion::SyncTwoTierSearcher;
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex, SearchParams};

/// Bind a hand-built fixture vector to an explicitly synthetic identity.
///
/// These vectors are written by hand, so a synthetic identity is the truthful
/// one — `explicit_test_model` exists precisely so that fact travels with the
/// values instead of being fabricated into a model name.
fn bound(vector: Vec<f32>) -> BoundQueryEmbedding {
    let dimension = u32::try_from(vector.len()).expect("fixture dimension fits u32");
    BoundQueryEmbedding::new(
        vector,
        EmbeddingIdentityBundleV1::explicit_test_model("sync-integration-fixture", dimension),
    )
    .expect("synthetic fixture query binds")
}

/// Both tiers of these fixture indexes are built from vectors of the same
/// synthetic space, so one bundle describes both arms truthfully. A real
/// two-model pair would bind two different bundles here.
fn tiered(vector: Vec<f32>) -> TieredQueryEmbeddings {
    TieredQueryEmbeddings::progressive(bound(vector.clone()), bound(vector))
}

fn normalize(values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return values;
    }
    values.into_iter().map(|v| v / norm).collect()
}

fn rank_flip_index() -> Arc<InMemoryTwoTierIndex> {
    let doc_ids = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
    let fast = InMemoryVectorIndex::from_vectors(
        doc_ids.clone(),
        vec![
            normalize(vec![1.0, 0.0]),
            normalize(vec![0.95, 0.05]),
            normalize(vec![0.0, 1.0]),
        ],
        2,
    )
    .expect("fast index");
    let quality = InMemoryVectorIndex::from_vectors(
        doc_ids,
        vec![
            normalize(vec![0.0, 1.0]),
            normalize(vec![1.0, 0.0]),
            normalize(vec![0.2, 0.8]),
        ],
        2,
    )
    .expect("quality index");
    Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)))
}

#[allow(clippy::cast_precision_loss)]
fn deterministic_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut vector = Vec::with_capacity(dim);
    for _ in 0..dim {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        vector.push((state >> 40) as f32 / (1_u64 << 23) as f32 - 1.0);
    }
    vector
}

fn clustered_vector(centroids: &[Vec<f32>], cluster: usize, noise_seed: u64) -> Vec<f32> {
    const NOISE: f32 = 0.30;
    let centroid = &centroids[cluster % centroids.len()];
    let noise = deterministic_vector(centroid.len(), noise_seed);
    normalize(
        centroid
            .iter()
            .zip(noise)
            .map(|(base, perturb)| base + NOISE * perturb)
            .collect(),
    )
}

fn clustered_sync_index(
    doc_count: usize,
    dim: usize,
    clusters: usize,
) -> Arc<InMemoryTwoTierIndex> {
    let centroids = (0..clusters)
        .map(|idx| {
            normalize(deterministic_vector(
                dim,
                0xc000_0000 + u64::try_from(idx).expect("cluster index fits u64"),
            ))
        })
        .collect::<Vec<_>>();
    let ids = (0..doc_count)
        .map(|idx| format!("doc-{idx:06}"))
        .collect::<Vec<_>>();
    let fast_vectors = (0..doc_count)
        .map(|idx| {
            clustered_vector(
                &centroids,
                idx % clusters,
                u64::try_from(idx).expect("doc index fits u64") + 1,
            )
        })
        .collect::<Vec<_>>();
    let quality_vectors = (0..doc_count)
        .map(|idx| {
            clustered_vector(
                &centroids,
                idx % clusters,
                0xbeef_0000 + u64::try_from(idx).expect("doc index fits u64"),
            )
        })
        .collect::<Vec<_>>();
    let fast =
        InMemoryVectorIndex::from_vectors(ids.clone(), fast_vectors, dim).expect("fast index");
    let quality =
        InMemoryVectorIndex::from_vectors(ids, quality_vectors, dim).expect("quality index");
    Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)))
}

#[test]
fn search_collect_returns_progressive_metrics() {
    let searcher = SyncTwoTierSearcher::new(rank_flip_index(), TwoTierConfig::default());
    let (results, metrics) = searcher
        .search_collect(&tiered(normalize(vec![1.0, 0.0])), 3)
        .expect("search collect");

    assert_eq!(results.len(), 3);
    assert!(metrics.phase1_vectors_searched > 0);
    assert!(metrics.phase2_vectors_searched > 0);
    assert_eq!(results[0].source, ScoreSource::SemanticQuality);
}

#[test]
fn search_iter_yields_initial_then_refined() {
    let searcher = SyncTwoTierSearcher::new(rank_flip_index(), TwoTierConfig::default());
    let phases = searcher
        .search_iter(&tiered(normalize(vec![1.0, 0.0])), 3)
        .collect::<Vec<_>>();

    assert_eq!(phases.len(), 2);
    assert!(matches!(phases[0], SearchPhase::Initial { .. }));
    assert!(matches!(phases[1], SearchPhase::Refined { .. }));
}

#[test]
fn explicit_int8_fetch_control_matches_exact_on_clustered_fixture() {
    const DOCS: usize = 2_048;
    const DIM: usize = 128;
    const CLUSTERS: usize = 32;
    const K: usize = 10;

    let index = clustered_sync_index(DOCS, DIM, CLUSTERS);
    let centroids = (0..CLUSTERS)
        .map(|idx| {
            normalize(deterministic_vector(
                DIM,
                0xc000_0000 + u64::try_from(idx).expect("cluster index fits u64"),
            ))
        })
        .collect::<Vec<_>>();
    let approximate = SyncTwoTierSearcher::new(index.clone(), TwoTierConfig::default())
        .with_approximate_int8_fast_fetch_for_bench(3);
    let exact = SyncTwoTierSearcher::new(index, TwoTierConfig::default())
        .with_search_params(SearchParams::default());

    for query_idx in 0..12 {
        let query = tiered(clustered_vector(
            &centroids,
            query_idx % CLUSTERS,
            0xdead_0000 + u64::try_from(query_idx).expect("query index fits u64"),
        ));
        let (approx_results, _) = approximate
            .search_collect(&query, K)
            .expect("approximate search");
        let (exact_results, _) = exact.search_collect(&query, K).expect("exact search");
        let approx_ids = approx_results
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<Vec<_>>();
        let exact_ids = exact_results
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(approx_ids, exact_ids, "query_idx={query_idx}");
    }
}

#[test]
fn fast_only_mode_skips_phase_two() {
    let config = TwoTierConfig {
        fast_only: true,
        ..TwoTierConfig::default()
    };
    let searcher = SyncTwoTierSearcher::new(rank_flip_index(), config);
    let phases = searcher
        .search_iter(&tiered(normalize(vec![1.0, 0.0])), 3)
        .collect::<Vec<_>>();

    assert_eq!(phases.len(), 1);
    assert!(matches!(phases[0], SearchPhase::Initial { .. }));
    let (_, metrics) = searcher
        .search_collect(&tiered(normalize(vec![1.0, 0.0])), 3)
        .expect("search collect");
    assert_eq!(metrics.skip_reason.as_deref(), Some("fast_only"));
}

#[test]
fn search_filter_is_applied_to_results() {
    struct ExcludeA;
    impl SearchFilter for ExcludeA {
        fn matches(&self, doc_id: &str, _metadata: Option<&serde_json::Value>) -> bool {
            doc_id != "a"
        }
        fn name(&self) -> &'static str {
            "exclude-a"
        }
    }

    let searcher = SyncTwoTierSearcher::new(rank_flip_index(), TwoTierConfig::default());
    let (results, _) = searcher
        .search_collect_with_filter(&tiered(normalize(vec![1.0, 0.0])), 3, Some(&ExcludeA))
        .expect("filtered search");
    assert!(results.iter().all(|result| result.doc_id != "a"));
}

#[test]
fn quality_weight_controls_refined_ranking() {
    let query = tiered(normalize(vec![1.0, 0.0]));
    let fast_only_blend = SyncTwoTierSearcher::new(
        rank_flip_index(),
        TwoTierConfig {
            quality_weight: 0.0,
            candidate_multiplier: 3,
            ..TwoTierConfig::default()
        },
    );
    let quality_only_blend = SyncTwoTierSearcher::new(
        rank_flip_index(),
        TwoTierConfig {
            quality_weight: 1.0,
            candidate_multiplier: 3,
            ..TwoTierConfig::default()
        },
    );

    let (fast_results, _) = fast_only_blend
        .search_collect(&query, 3)
        .expect("fast blend");
    let (quality_results, _) = quality_only_blend
        .search_collect(&query, 3)
        .expect("quality blend");

    assert_eq!(fast_results[0].doc_id, "a");
    assert_eq!(quality_results[0].doc_id, "b");
}

#[test]
fn candidate_multiplier_changes_refinement_recall() {
    let query = tiered(normalize(vec![1.0, 0.0]));
    let strict_budget = SyncTwoTierSearcher::new(
        rank_flip_index(),
        TwoTierConfig {
            quality_weight: 1.0,
            candidate_multiplier: 1,
            ..TwoTierConfig::default()
        },
    );
    let relaxed_budget = SyncTwoTierSearcher::new(
        rank_flip_index(),
        TwoTierConfig {
            quality_weight: 1.0,
            candidate_multiplier: 3,
            ..TwoTierConfig::default()
        },
    )
    .with_search_params(SearchParams {
        parallel_enabled: true,
        parallel_threshold: 1,
        parallel_chunk_size: 2,
    });

    let (strict_results, _) = strict_budget
        .search_collect(&query, 1)
        .expect("strict budget search");
    let (relaxed_results, _) = relaxed_budget
        .search_collect(&query, 1)
        .expect("relaxed budget search");

    assert_eq!(strict_results[0].doc_id, "a");
    assert_eq!(relaxed_results[0].doc_id, "b");
}

#[test]
fn empty_index_is_graceful() {
    let empty_fast = InMemoryVectorIndex::from_vectors(Vec::new(), Vec::new(), 2).expect("empty");
    let empty_two_tier = Arc::new(InMemoryTwoTierIndex::new(empty_fast, None));
    let searcher = SyncTwoTierSearcher::new(empty_two_tier, TwoTierConfig::default());

    let (results, metrics) = searcher
        .search_collect(&tiered(normalize(vec![1.0, 0.0])), 5)
        .expect("empty search");
    assert!(results.is_empty());
    assert_eq!(metrics.phase1_vectors_searched, 0);
    assert_eq!(
        metrics.skip_reason.as_deref(),
        Some("quality_index_unavailable")
    );
}

#[test]
fn dimension_mismatch_is_reported_by_iterator() {
    let searcher = SyncTwoTierSearcher::new(rank_flip_index(), TwoTierConfig::default());
    // A width the index does not have. This used to be `&[]`, which the typed
    // surface cannot express; a 3-wide query in its own coherent 3-dimensional
    // space is the same test with a vector that actually exists. The fixture
    // index carries no space identity, so admission passes and the mismatch
    // surfaces where it always did — at the scan.
    let phases = searcher
        .search_iter(&tiered(vec![1.0, 0.0, 0.0]), 3)
        .collect::<Vec<_>>();

    assert_eq!(phases.len(), 1);
    assert!(matches!(
        &phases[0],
        SearchPhase::RefinementFailed {
            error: SearchError::DimensionMismatch { .. },
            ..
        }
    ));
}

#[test]
fn concurrent_search_collect_is_thread_safe() {
    let searcher = Arc::new(SyncTwoTierSearcher::new(
        rank_flip_index(),
        TwoTierConfig::default(),
    ));
    let query = tiered(normalize(vec![1.0, 0.0]));

    let mut handles = Vec::new();
    for _ in 0..8 {
        let searcher = Arc::clone(&searcher);
        let query = query.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..20 {
                let (results, metrics) = searcher.search_collect(&query, 3).expect("search");
                assert_eq!(results.len(), 3);
                assert!(metrics.phase1_total_ms >= 0.0);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread join");
    }
}
