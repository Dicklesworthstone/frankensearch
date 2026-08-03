//! Searcher-parity conformance suite — first slice (bd-k3089).
//!
//! `searcher.rs` (async) and `sync_searcher.rs` implement ONE two-tier
//! pipeline and have drifted with the correct behavior on opposite sides
//! (bd-180wl was fixed by copying the sync quality-index guard into the async
//! `should_run_quality`). This suite runs the SAME corpus, query vectors, and
//! configs through both searchers and asserts field-by-field agreement on the
//! ordered results, so the next divergence fails a test instead of shipping a
//! silently wrong ranking.
//!
//! Intentional divergences are typed allowlist entries in
//! [`KNOWN_DIVERGENCES`] — never silence. Wall-clock timing fields are out of
//! scope by design.

use std::sync::Arc;

use frankensearch_core::traits::SearchFuture;
use frankensearch_core::{
    Cx, Embedder, ModelCategory, ScoredResult, TwoTierConfig, TwoTierMetrics,
};
use frankensearch_fusion::{SyncTwoTierSearcher, TwoTierSearcher};
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex, TwoTierIndex};

/// Known, documented divergences between the two searchers (bd-k3089).
/// Each entry pins WHERE the divergence lives so accidental convergence or a
/// new divergence both surface as test failures worth reading.
const KNOWN_DIVERGENCES: &[&str] = &[
    // The sync searcher takes a pre-embedded query vector and has no
    // embedder, so fast/quality embedder ids are absent from its metrics.
    "fast_embedder_id/quality_embedder_id: async-only",
];

const DIM: usize = 4;

/// Corpus crafted so the quality tier disagrees with the fast tier (same
/// shape as the async promotion fixture): rank flips between phases are
/// exactly where the two implementations have historically drifted.
const DOCS: [(&str, [f32; DIM], [f32; DIM]); 4] = [
    ("doc-a", [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
    ("doc-b", [0.95, 0.05, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
    ("doc-c", [0.6, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
    ("doc-d", [0.5, 0.866, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
];

fn normalize(values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return values;
    }
    values.into_iter().map(|v| v / norm).collect()
}

/// Async-side embedder that returns one fixed vector for every query text,
/// so both searchers score the exact same query vector.
struct FixedVecEmbedder {
    id: &'static str,
    vector: Vec<f32>,
}

impl Embedder for FixedVecEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        let vector = self.vector.clone();
        Box::pin(async move { Ok(vector) })
    }

    fn dimension(&self) -> usize {
        self.vector.len()
    }

    fn id(&self) -> &str {
        self.id
    }

    fn model_name(&self) -> &str {
        self.id
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

fn sync_index() -> Arc<InMemoryTwoTierIndex> {
    let ids = DOCS
        .iter()
        .map(|(id, _, _)| (*id).to_owned())
        .collect::<Vec<_>>();
    let fast = InMemoryVectorIndex::from_vectors(
        ids.clone(),
        DOCS.iter().map(|(_, f, _)| normalize(f.to_vec())).collect(),
        DIM,
    )
    .expect("fast in-memory index");
    let quality = InMemoryVectorIndex::from_vectors(
        ids,
        DOCS.iter().map(|(_, _, q)| normalize(q.to_vec())).collect(),
        DIM,
    )
    .expect("quality in-memory index");
    Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality)))
}

fn async_index(tag: &str) -> Arc<TwoTierIndex> {
    let dir = std::env::temp_dir().join(format!(
        "fsx-parity-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
    builder.set_fast_embedder_id("parity-fast");
    builder.set_quality_embedder_id("parity-quality");
    for (id, fast, quality) in &DOCS {
        builder
            .add_fast_record((*id).to_owned(), &normalize(fast.to_vec()))
            .expect("add fast record");
        builder
            .add_quality_record((*id).to_owned(), &normalize(quality.to_vec()))
            .expect("add quality record");
    }
    Arc::new(builder.finish().expect("finish index"))
}

fn run_async(
    tag: &str,
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let index = async_index(tag);
    let fast: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-fast",
        vector: query_vec.to_vec(),
    });
    let quality: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-quality",
        vector: query_vec.to_vec(),
    });
    let config = config.clone();
    let mut out = None;
    asupersync::test_utils::run_test_with_cx(|cx| {
        let slot = &mut out;
        async move {
            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
            *slot = Some(
                searcher
                    .search_collect(&cx, "parity conformance query", k)
                    .await
                    .expect("async search_collect"),
            );
        }
    });
    out.expect("async search ran")
}

fn run_sync(
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let searcher = SyncTwoTierSearcher::new(sync_index(), config.clone());
    searcher
        .search_collect(query_vec, k)
        .expect("sync search_collect")
}

fn assert_result_parity(case: &str, sync_r: &[ScoredResult], async_r: &[ScoredResult]) {
    let sync_ids = sync_r.iter().map(|r| r.doc_id.as_str()).collect::<Vec<_>>();
    let async_ids = async_r
        .iter()
        .map(|r| r.doc_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        sync_ids, async_ids,
        "[{case}] ordered doc_id lists diverge (bd-k3089 class)"
    );
    for (s, a) in sync_r.iter().zip(async_r.iter()) {
        assert_eq!(
            s.source, a.source,
            "[{case}] {}: ScoreSource diverges",
            s.doc_id
        );
        assert_scores_close(case, &s.doc_id, "fast_score", s.fast_score, a.fast_score);
        assert_scores_close(
            case,
            &s.doc_id,
            "quality_score",
            s.quality_score,
            a.quality_score,
        );
    }
}

fn assert_scores_close(case: &str, doc: &str, field: &str, s: Option<f32>, a: Option<f32>) {
    match (s, a) {
        (None, None) => {}
        (Some(sv), Some(av)) => {
            assert!(
                (sv - av).abs() < 1e-5,
                "[{case}] {doc}: {field} diverges beyond fp tolerance: sync={sv} async={av}"
            );
        }
        _ => assert_eq!(
            s.is_some(),
            a.is_some(),
            "[{case}] {doc}: {field} present on one side only: sync={s:?} async={a:?}"
        ),
    }
}

fn assert_metric_parity(case: &str, sync_m: &TwoTierMetrics, async_m: &TwoTierMetrics) {
    assert_eq!(
        sync_m.skip_reason, async_m.skip_reason,
        "[{case}] skip_reason diverges"
    );
    assert_eq!(
        sync_m.phase1_vectors_searched, async_m.phase1_vectors_searched,
        "[{case}] phase1_vectors_searched diverges"
    );
    assert_eq!(
        sync_m.phase2_vectors_searched > 0,
        async_m.phase2_vectors_searched > 0,
        "[{case}] phase2 ran on one side only"
    );
}

fn conformance_case(case: &str, config: &TwoTierConfig, query: Vec<f32>, k: usize) {
    let query = normalize(query);
    let (sync_results, sync_metrics) = run_sync(config, &query, k);
    let (async_results, async_metrics) = run_async(case, config, &query, k);
    assert_result_parity(case, &sync_results, &async_results);
    assert_metric_parity(case, &sync_metrics, &async_metrics);
}

#[test]
fn default_config_agrees_on_rank_flip_corpus() {
    conformance_case(
        "default-k4",
        &TwoTierConfig::default(),
        vec![1.0, 0.0, 0.0, 0.0],
        4,
    );
}

#[test]
fn default_config_agrees_at_small_k() {
    conformance_case(
        "default-k2",
        &TwoTierConfig::default(),
        vec![1.0, 0.0, 0.0, 0.0],
        2,
    );
}

#[test]
fn fast_only_agrees_and_skips_phase_two_on_both_sides() {
    let config = TwoTierConfig {
        fast_only: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync(&config, &query, 4);
    let (async_results, async_metrics) = run_async("fast-only", &config, &query, 4);
    assert_result_parity("fast-only", &sync_results, &async_results);
    assert_eq!(sync_metrics.phase2_vectors_searched, 0, "sync ran phase 2");
    assert_eq!(
        async_metrics.phase2_vectors_searched, 0,
        "async ran phase 2"
    );
    assert_eq!(
        sync_metrics.skip_reason, async_metrics.skip_reason,
        "fast_only skip_reason diverges"
    );
}

#[test]
fn quality_dominant_blend_agrees() {
    conformance_case(
        "quality-weight-1",
        &TwoTierConfig {
            quality_weight: 1.0,
            candidate_multiplier: 3,
            ..TwoTierConfig::default()
        },
        vec![1.0, 0.0, 0.0, 0.0],
        4,
    );
}

#[test]
fn orthogonal_query_agrees() {
    conformance_case(
        "orthogonal-query",
        &TwoTierConfig::default(),
        vec![0.0, 1.0, 0.0, 0.0],
        4,
    );
}

#[test]
fn phase1_vectors_searched_reports_the_evaluated_corpus_at_small_k() {
    assert_eq!(KNOWN_DIVERGENCES.len(), 1);
    let config = TwoTierConfig::default();
    let k = 1;
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (_, sync_metrics) = run_sync(&config, &query, k);
    let (_, async_metrics) = run_async("small-k-vector-count", &config, &query, k);
    assert_eq!(
        sync_metrics.phase1_vectors_searched,
        DOCS.len(),
        "sync phase1_vectors_searched must count evaluated vectors, not the \
         returned candidate pool"
    );
    assert_eq!(
        sync_metrics.phase1_vectors_searched, async_metrics.phase1_vectors_searched,
        "small-k phase1_vectors_searched must agree across implementations"
    );
}
