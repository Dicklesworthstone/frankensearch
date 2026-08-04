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

use frankensearch_core::explanation::{ExplainedSource, HitExplanation};
use frankensearch_core::traits::{LexicalRead, SearchFuture};
use frankensearch_core::{
    Cx, Embedder, ModelCategory, ScoredResult, SearchPhase, TwoTierConfig, TwoTierMetrics,
};
use frankensearch_fusion::{SyncLexicalSearch, SyncTwoTierSearcher, TwoTierSearcher};
use frankensearch_index::{InMemoryTwoTierIndex, InMemoryVectorIndex, TwoTierIndex};

/// One seeded corpus document: its id plus its fast and quality vectors.
type SeededDoc = (String, Vec<f32>, Vec<f32>);

/// Phase snapshots captured from one searcher, labelled by phase name.
type PhaseSnapshots = Vec<(&'static str, Vec<ScoredResult>)>;

/// The sync and async snapshot pair a parity assertion compares.
type ParitySnapshots = (PhaseSnapshots, PhaseSnapshots);

/// Known, documented divergences between the two searchers (bd-k3089).
/// Each entry pins WHERE the divergence lives so accidental convergence or a
/// new divergence both surface as test failures worth reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownDivergence {
    /// The sync API receives an already embedded query vector.
    EmbedderIdentity,
    /// A vector has no source text from which the sync API could classify it.
    QueryClass,
    /// The sync API intentionally does not compute a timing-independent tau.
    KendallTau,
    /// The synchronous in-memory fixture and serialized FSVI fixture can use
    /// different physical row positions for the same document. Result order
    /// and `doc_id` are the portable rank identity; index is covered by the
    /// shared four-document fixture where both layouts are intentionally equal.
    IndexIdentity,
}

const KNOWN_DIVERGENCES: &[KnownDivergence] = &[
    KnownDivergence::EmbedderIdentity,
    KnownDivergence::QueryClass,
    KnownDivergence::KendallTau,
    KnownDivergence::IndexIdentity,
];

fn is_known_divergence(divergence: KnownDivergence) -> bool {
    KNOWN_DIVERGENCES.contains(&divergence)
}

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

fn sync_index(with_quality: bool) -> Arc<InMemoryTwoTierIndex> {
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
    let quality = with_quality.then(|| {
        InMemoryVectorIndex::from_vectors(
            ids,
            DOCS.iter().map(|(_, _, q)| normalize(q.to_vec())).collect(),
            DIM,
        )
        .expect("quality in-memory index")
    });
    Arc::new(InMemoryTwoTierIndex::new(fast, quality))
}

fn lexical_hits() -> Vec<ScoredResult> {
    [("doc-c", 3.0_f32), ("doc-a", 2.0_f32), ("doc-d", 1.0_f32)]
        .into_iter()
        .map(|(doc_id, score)| ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: frankensearch_core::ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: None,
            metadata: None,
        })
        .collect()
}

struct StaticLexical {
    hits: Vec<ScoredResult>,
}

impl SyncLexicalSearch for StaticLexical {
    fn search_sync(
        &self,
        _query_vec: &[f32],
        limit: usize,
    ) -> frankensearch_core::SearchResult<Vec<ScoredResult>> {
        Ok(self.hits.iter().take(limit).cloned().collect())
    }
}

impl LexicalRead for StaticLexical {
    fn search<'a>(
        &'a self,
        _cx: &'a Cx,
        _query: &'a str,
        limit: usize,
    ) -> SearchFuture<'a, Vec<ScoredResult>> {
        Box::pin(async move { Ok(self.hits.iter().take(limit).cloned().collect()) })
    }

    fn doc_count(&self) -> usize {
        self.hits.len()
    }
}

fn async_index(tag: &str, with_quality: bool) -> Arc<TwoTierIndex> {
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
    if with_quality {
        builder.set_quality_embedder_id("parity-quality");
    }
    for (id, fast, quality) in &DOCS {
        builder
            .add_fast_record((*id).to_owned(), &normalize(fast.to_vec()))
            .expect("add fast record");
        if with_quality {
            builder
                .add_quality_record((*id).to_owned(), &normalize(quality.to_vec()))
                .expect("add quality record");
        }
    }
    Arc::new(builder.finish().expect("finish index"))
}

fn run_async(
    tag: &str,
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    run_async_with_quality_index(tag, config, query_vec, k, true)
}

fn run_async_with_quality_index(
    tag: &str,
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
    with_quality: bool,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let index = async_index(tag, with_quality);
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
    run_sync_with_quality_index(config, query_vec, k, true)
}

fn run_sync_with_quality_index(
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
    with_quality: bool,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let searcher = SyncTwoTierSearcher::new(sync_index(with_quality), config.clone());
    searcher
        .search_collect(query_vec, k)
        .expect("sync search_collect")
}

const SEEDED_DIM: usize = 8;

fn seeded_unit_vector(state: &mut u64) -> Vec<f32> {
    let values = (0..SEEDED_DIM)
        .map(|_| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let high = u16::try_from(*state >> 48).expect("upper 16 bits fit u16");
            (f32::from(high) / f32::from(u16::MAX)) * 2.0 - 1.0
        })
        .collect();
    normalize(values)
}

fn seeded_corpus() -> (Vec<SeededDoc>, Vec<f32>) {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let docs = (0..32)
        .map(|index| {
            let fast = seeded_unit_vector(&mut state);
            let quality = seeded_unit_vector(&mut state);
            (format!("seeded-{index:02}"), fast, quality)
        })
        .collect();
    (docs, seeded_unit_vector(&mut state))
}

fn seeded_sync_search(
    docs: &[(String, Vec<f32>, Vec<f32>)],
    query: &[f32],
    config: &TwoTierConfig,
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let ids: Vec<String> = docs.iter().map(|(id, _, _)| id.clone()).collect();
    let fast = InMemoryVectorIndex::from_vectors(
        ids.clone(),
        docs.iter().map(|(_, fast, _)| fast.clone()).collect(),
        SEEDED_DIM,
    )
    .expect("seeded fast index");
    let quality = InMemoryVectorIndex::from_vectors(
        ids,
        docs.iter().map(|(_, _, quality)| quality.clone()).collect(),
        SEEDED_DIM,
    )
    .expect("seeded quality index");
    SyncTwoTierSearcher::new(
        Arc::new(InMemoryTwoTierIndex::new(fast, Some(quality))),
        config.clone(),
    )
    .search_collect(query, k)
    .expect("seeded sync search_collect")
}

fn seeded_async_search(
    docs: &[(String, Vec<f32>, Vec<f32>)],
    query: &[f32],
    config: &TwoTierConfig,
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let dir = std::env::temp_dir().join(format!(
        "fsx-parity-seeded-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let mut builder = TwoTierIndex::create(&dir, TwoTierConfig::default()).expect("create index");
    builder.set_fast_embedder_id("parity-fast");
    builder.set_quality_embedder_id("parity-quality");
    for (id, fast, quality) in docs {
        builder
            .add_fast_record(id.clone(), fast)
            .expect("add seeded fast record");
        builder
            .add_quality_record(id.clone(), quality)
            .expect("add seeded quality record");
    }
    let index = Arc::new(builder.finish().expect("finish seeded index"));
    let fast: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-fast",
        vector: query.to_vec(),
    });
    let quality: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-quality",
        vector: query.to_vec(),
    });
    let config = config.clone();
    let mut out = None;
    asupersync::test_utils::run_test_with_cx(|cx| {
        let slot = &mut out;
        async move {
            let searcher = TwoTierSearcher::new(index, fast, config).with_quality_embedder(quality);
            *slot = Some(
                searcher
                    .search_collect(&cx, "seeded corpus parity query", k)
                    .await
                    .expect("seeded async search_collect"),
            );
        }
    });
    out.expect("seeded async search ran")
}

fn run_async_with_lexical(
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    let index = async_index("lexical", true);
    let fast: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-fast",
        vector: query_vec.to_vec(),
    });
    let quality: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-quality",
        vector: query_vec.to_vec(),
    });
    let lexical = Arc::new(StaticLexical {
        hits: lexical_hits(),
    });
    let config = config.clone();
    let mut out = None;
    asupersync::test_utils::run_test_with_cx(|cx| {
        let slot = &mut out;
        async move {
            let searcher = TwoTierSearcher::new(index, fast, config)
                .with_quality_embedder(quality)
                .with_lexical(lexical);
            *slot = Some(
                searcher
                    .search_collect(
                        &cx,
                        "how does this parity conformance query retrieve results",
                        k,
                    )
                    .await
                    .expect("async lexical search_collect"),
            );
        }
    });
    out.expect("async lexical search ran")
}

fn run_sync_with_lexical(
    config: &TwoTierConfig,
    query_vec: &[f32],
    k: usize,
) -> (Vec<ScoredResult>, TwoTierMetrics) {
    SyncTwoTierSearcher::new(sync_index(true), config.clone())
        .with_lexical(Arc::new(StaticLexical {
            hits: lexical_hits(),
        }))
        .search_collect(query_vec, k)
        .expect("sync lexical search_collect")
}

fn phase_label(phase: &SearchPhase) -> &'static str {
    match phase {
        SearchPhase::Initial { .. } => "initial",
        SearchPhase::Refined { .. } => "refined",
        SearchPhase::Reranked { .. } => "reranked",
        SearchPhase::RefinementFailed { .. } => "refinement_failed",
    }
}

fn phase_results(phase: &SearchPhase) -> &[ScoredResult] {
    match phase {
        SearchPhase::Initial { results, .. }
        | SearchPhase::Refined { results, .. }
        | SearchPhase::Reranked { results, .. } => results,
        SearchPhase::RefinementFailed {
            initial_results, ..
        } => initial_results,
    }
}

fn phase_snapshots(
    query_vec: &[f32],
    with_quality: bool,
    config: TwoTierConfig,
) -> ParitySnapshots {
    let sync = SyncTwoTierSearcher::new(sync_index(with_quality), config.clone())
        .search_iter(query_vec, 4)
        .map(|phase| (phase_label(&phase), phase_results(&phase).to_vec()))
        .collect();
    let index = async_index("phase-snapshots", with_quality);
    let fast: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-fast",
        vector: query_vec.to_vec(),
    });
    let quality: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-quality",
        vector: query_vec.to_vec(),
    });
    let mut async_phases = Vec::new();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let slot = &mut async_phases;
        async move {
            TwoTierSearcher::new(index, fast, config)
                .with_quality_embedder(quality)
                .search(
                    &cx,
                    "parity conformance query",
                    4,
                    |_| None,
                    |phase| slot.push((phase_label(&phase), phase_results(&phase).to_vec())),
                )
                .await
                .expect("async phase search");
        }
    });
    (sync, async_phases)
}

fn lexical_phase_snapshots(query_vec: &[f32], config: TwoTierConfig) -> ParitySnapshots {
    let sync = SyncTwoTierSearcher::new(sync_index(true), config.clone())
        .with_lexical(Arc::new(StaticLexical {
            hits: lexical_hits(),
        }))
        .search_iter(query_vec, 3)
        .map(|phase| (phase_label(&phase), phase_results(&phase).to_vec()))
        .collect();
    let index = async_index("lexical-phase-snapshots", true);
    let fast: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-fast",
        vector: query_vec.to_vec(),
    });
    let quality: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-quality",
        vector: query_vec.to_vec(),
    });
    let lexical = Arc::new(StaticLexical {
        hits: lexical_hits(),
    });
    let mut async_phases = Vec::new();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let slot = &mut async_phases;
        async move {
            TwoTierSearcher::new(index, fast, config)
                .with_quality_embedder(quality)
                .with_lexical(lexical)
                .search(
                    &cx,
                    "how does this parity conformance query retrieve results",
                    3,
                    |_| None,
                    |phase| slot.push((phase_label(&phase), phase_results(&phase).to_vec())),
                )
                .await
                .expect("async lexical phase search");
        }
    });
    (sync, async_phases)
}

fn phase_labels(
    query_vec: &[f32],
    with_quality: bool,
    config: TwoTierConfig,
) -> (Vec<&'static str>, Vec<&'static str>) {
    let sync = SyncTwoTierSearcher::new(sync_index(with_quality), config.clone())
        .search_iter(query_vec, 4)
        .map(|phase| phase_label(&phase))
        .collect();
    let index = async_index("quality-index-phase", with_quality);
    let fast: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-fast",
        vector: query_vec.to_vec(),
    });
    let quality: Arc<dyn Embedder> = Arc::new(FixedVecEmbedder {
        id: "parity-quality",
        vector: query_vec.to_vec(),
    });
    let mut async_phases = Vec::new();
    asupersync::test_utils::run_test_with_cx(|cx| {
        let slot = &mut async_phases;
        async move {
            TwoTierSearcher::new(index, fast, config)
                .with_quality_embedder(quality)
                .search(
                    &cx,
                    "parity conformance query",
                    4,
                    |_| None,
                    |phase| slot.push(phase_label(&phase)),
                )
                .await
                .expect("async phase search");
        }
    });
    (sync, async_phases)
}

fn assert_result_parity(case: &str, sync_r: &[ScoredResult], async_r: &[ScoredResult]) {
    assert_result_parity_with_index_policy(case, sync_r, async_r, true);
}

fn assert_result_parity_allowing_index_identity(
    case: &str,
    sync_r: &[ScoredResult],
    async_r: &[ScoredResult],
) {
    assert!(is_known_divergence(KnownDivergence::IndexIdentity));
    assert_result_parity_with_index_policy(case, sync_r, async_r, false);
}

fn assert_result_parity_with_index_policy(
    case: &str,
    sync_r: &[ScoredResult],
    async_r: &[ScoredResult],
    compare_index: bool,
) {
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
        assert_scores_close(case, &s.doc_id, "score", Some(s.score), Some(a.score));
        assert_eq!(
            s.source, a.source,
            "[{case}] {}: ScoreSource diverges",
            s.doc_id
        );
        if compare_index {
            assert_eq!(
                s.index, a.index,
                "[{case}] {}: vector index diverges",
                s.doc_id
            );
        }
        assert_scores_close(case, &s.doc_id, "fast_score", s.fast_score, a.fast_score);
        assert_scores_close(
            case,
            &s.doc_id,
            "quality_score",
            s.quality_score,
            a.quality_score,
        );
        assert_scores_close(
            case,
            &s.doc_id,
            "lexical_score",
            s.lexical_score,
            a.lexical_score,
        );
        assert_scores_close(
            case,
            &s.doc_id,
            "rerank_score",
            s.rerank_score,
            a.rerank_score,
        );
        assert_eq!(
            s.explanation.is_some(),
            a.explanation.is_some(),
            "[{case}] {}: explanation presence diverges",
            s.doc_id
        );
        if let (Some(sync_explanation), Some(async_explanation)) = (&s.explanation, &a.explanation)
        {
            assert_explanation_parity(case, &s.doc_id, sync_explanation, async_explanation);
        }
        assert_eq!(
            s.metadata.as_deref(),
            a.metadata.as_deref(),
            "[{case}] {}: metadata diverges",
            s.doc_id
        );
    }
}

fn explained_source_kind(source: &ExplainedSource) -> &'static str {
    match source {
        ExplainedSource::LexicalBm25 { .. } => "lexical",
        ExplainedSource::SemanticFast { .. } => "semantic_fast",
        ExplainedSource::SemanticQuality { .. } => "semantic_quality",
        ExplainedSource::Rerank { .. } => "rerank",
    }
}

fn assert_explanation_parity(
    case: &str,
    doc: &str,
    sync: &HitExplanation,
    asynchronous: &HitExplanation,
) {
    assert_eq!(
        sync.phase, asynchronous.phase,
        "[{case}] {doc}: explanation phase diverges"
    );
    assert!(
        (sync.final_score - asynchronous.final_score).abs() < 1e-5,
        "[{case}] {doc}: explanation final score diverges"
    );
    assert_eq!(
        sync.components.len(),
        asynchronous.components.len(),
        "[{case}] {doc}: explanation component count diverges"
    );
    for (sync_component, async_component) in sync.components.iter().zip(&asynchronous.components) {
        assert_eq!(
            explained_source_kind(&sync_component.source),
            explained_source_kind(&async_component.source),
            "[{case}] {doc}: explanation component source diverges"
        );
        for (field, sync_value, async_value) in [
            (
                "raw score",
                sync_component.raw_score,
                async_component.raw_score,
            ),
            (
                "normalized score",
                sync_component.normalized_score,
                async_component.normalized_score,
            ),
            (
                "rrf contribution",
                sync_component.rrf_contribution,
                async_component.rrf_contribution,
            ),
            ("weight", sync_component.weight, async_component.weight),
        ] {
            assert!(
                (sync_value - async_value).abs() < 1e-5,
                "[{case}] {doc}: explanation {field} diverges: sync={sync_value}, async={async_value}"
            );
        }
    }
    assert_eq!(
        sync.rank_movement.is_some(),
        asynchronous.rank_movement.is_some(),
        "[{case}] {doc}: explanation rank-movement presence diverges"
    );
    if let (Some(sync_rank), Some(async_rank)) = (&sync.rank_movement, &asynchronous.rank_movement)
    {
        assert_eq!(
            (
                sync_rank.initial_rank,
                sync_rank.refined_rank,
                sync_rank.delta
            ),
            (
                async_rank.initial_rank,
                async_rank.refined_rank,
                async_rank.delta
            ),
            "[{case}] {doc}: explanation rank movement diverges"
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
    assert_eq!(
        sync_m.lexical_candidates, async_m.lexical_candidates,
        "[{case}] lexical candidate count diverges"
    );
    assert_eq!(
        sync_m.semantic_candidates, async_m.semantic_candidates,
        "[{case}] semantic candidate count diverges"
    );
    assert_eq!(
        sync_m.incomplete_embeddings, async_m.incomplete_embeddings,
        "[{case}] incomplete embedding count diverges"
    );
    assert_eq!(
        sync_m.zero_signal, async_m.zero_signal,
        "[{case}] zero-signal classification diverges"
    );
    assert_eq!(
        sync_m.rank_changes.promoted, async_m.rank_changes.promoted,
        "[{case}] promoted rank-change count diverges"
    );
    assert_eq!(
        sync_m.rank_changes.demoted, async_m.rank_changes.demoted,
        "[{case}] demoted rank-change count diverges"
    );
    assert_eq!(
        sync_m.rank_changes.stable, async_m.rank_changes.stable,
        "[{case}] stable rank-change count diverges"
    );

    // The sync API starts from a query vector, not query text, and therefore
    // cannot truthfully supply either an embedder identity or query class.
    assert!(is_known_divergence(KnownDivergence::EmbedderIdentity));
    assert_eq!(
        sync_m.fast_embedder_id, None,
        "[{case}] sync fast id changed"
    );
    assert_eq!(
        sync_m.quality_embedder_id, None,
        "[{case}] sync quality id changed"
    );
    assert!(
        async_m.fast_embedder_id.is_some(),
        "[{case}] async fast id missing"
    );
    assert_eq!(
        async_m.quality_embedder_id.is_some(),
        async_m.phase2_vectors_searched > 0,
        "[{case}] async quality id must reflect whether Phase 2 ran"
    );

    assert!(is_known_divergence(KnownDivergence::QueryClass));
    assert_eq!(
        sync_m.query_class, None,
        "[{case}] sync query class changed"
    );
    assert!(
        async_m.query_class.is_some(),
        "[{case}] async query class missing"
    );

    assert!(is_known_divergence(KnownDivergence::KendallTau));
    assert_eq!(
        sync_m.kendall_tau, None,
        "[{case}] sync kendall tau changed"
    );
    assert_eq!(
        async_m.kendall_tau.is_some(),
        async_m.phase2_vectors_searched > 0,
        "[{case}] async kendall tau must reflect whether Phase 2 ran"
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
fn zero_k_agrees_without_scanning_or_refining() {
    let config = TwoTierConfig::default();
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync(&config, &query, 0);
    let (async_results, async_metrics) = run_async("zero-k", &config, &query, 0);
    assert_result_parity("zero-k", &sync_results, &async_results);
    assert_eq!(
        sync_metrics.phase1_vectors_searched, 0,
        "sync scanned at k=0"
    );
    assert_eq!(
        async_metrics.phase1_vectors_searched, 0,
        "async scanned at k=0"
    );
    assert_eq!(
        sync_metrics.phase2_vectors_searched, 0,
        "sync refined at k=0"
    );
    assert_eq!(
        async_metrics.phase2_vectors_searched, 0,
        "async refined at k=0"
    );
}

#[test]
fn zero_norm_query_agrees_on_typed_zero_signal_and_skips_refinement() {
    let query = vec![0.0, 0.0, 0.0, 0.0];
    let (sync_results, sync_metrics) = run_sync(&TwoTierConfig::default(), &query, 4);
    let (async_results, async_metrics) =
        run_async("zero-norm", &TwoTierConfig::default(), &query, 4);
    assert_result_parity("zero-norm", &sync_results, &async_results);
    assert_eq!(sync_metrics.zero_signal, async_metrics.zero_signal);
    assert_eq!(sync_metrics.phase2_vectors_searched, 0);
    assert_eq!(async_metrics.phase2_vectors_searched, 0);
}

#[test]
fn seeded_corpus_agrees_on_non_lattice_rankings_and_metrics() {
    let (docs, query) = seeded_corpus();
    let config = TwoTierConfig {
        candidate_multiplier: 4,
        quality_weight: 0.65,
        explain: true,
        ..TwoTierConfig::default()
    };
    let (sync_results, sync_metrics) = seeded_sync_search(&docs, &query, &config, 8);
    let (async_results, async_metrics) = seeded_async_search(&docs, &query, &config, 8);
    assert_result_parity_allowing_index_identity("seeded-corpus", &sync_results, &async_results);
    assert_metric_parity("seeded-corpus", &sync_metrics, &async_metrics);
    assert!(
        sync_results
            .iter()
            .all(|result| result.explanation.is_some()),
        "sync explain mode must survive the seeded corpus"
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
fn fast_only_explanations_are_present_on_both_sides() {
    let config = TwoTierConfig {
        fast_only: true,
        explain: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync(&config, &query, 4);
    let (async_results, async_metrics) = run_async("fast-only-explain", &config, &query, 4);
    assert_result_parity("fast-only-explain", &sync_results, &async_results);
    assert_metric_parity("fast-only-explain", &sync_metrics, &async_metrics);
    assert!(
        sync_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
    assert!(
        async_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
}

#[test]
fn refined_explanations_are_present_on_both_sides() {
    let config = TwoTierConfig {
        explain: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync(&config, &query, 4);
    let (async_results, async_metrics) = run_async("refined-explain", &config, &query, 4);
    assert_result_parity("refined-explain", &sync_results, &async_results);
    assert_metric_parity("refined-explain", &sync_metrics, &async_metrics);
    assert!(
        sync_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
    assert!(
        async_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
}

#[test]
fn quality_index_unavailable_agrees_and_skips_phase_two_on_both_sides() {
    let config = TwoTierConfig::default();
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync_with_quality_index(&config, &query, 4, false);
    let (async_results, async_metrics) =
        run_async_with_quality_index("quality-index-unavailable", &config, &query, 4, false);
    assert_result_parity("quality-index-unavailable", &sync_results, &async_results);
    assert_eq!(sync_metrics.phase2_vectors_searched, 0, "sync ran phase 2");
    assert_eq!(
        async_metrics.phase2_vectors_searched, 0,
        "async ran phase 2"
    );
    assert_eq!(
        sync_metrics.skip_reason.as_deref(),
        Some("quality_index_unavailable"),
        "sync must report the typed unavailable-index skip"
    );
    assert_eq!(sync_metrics.skip_reason, async_metrics.skip_reason);
    let (sync_phases, async_phases) = phase_labels(&query, false, TwoTierConfig::default());
    assert_eq!(sync_phases, ["initial"]);
    assert_eq!(sync_phases, async_phases);
}

#[test]
fn quality_index_present_emits_matching_initial_then_refined_phases() {
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_phases, async_phases) = phase_labels(&query, true, TwoTierConfig::default());
    assert_eq!(sync_phases, ["initial", "refined"]);
    assert_eq!(sync_phases, async_phases);
}

#[test]
fn explain_mode_preserves_field_parity_in_each_progressive_phase() {
    let config = TwoTierConfig {
        explain: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_phases, async_phases) = phase_snapshots(&query, true, config);
    let sync_labels = sync_phases
        .iter()
        .map(|(label, _)| *label)
        .collect::<Vec<_>>();
    let async_labels = async_phases
        .iter()
        .map(|(label, _)| *label)
        .collect::<Vec<_>>();
    assert_eq!(sync_labels, ["initial", "refined"]);
    assert_eq!(sync_labels, async_labels);
    for ((label, sync_results), (_, async_results)) in sync_phases.iter().zip(async_phases.iter()) {
        assert_result_parity(
            &format!("phase-{label}-explain"),
            sync_results,
            async_results,
        );
        assert!(
            sync_results
                .iter()
                .all(|result| result.explanation.is_some())
        );
        assert!(
            async_results
                .iter()
                .all(|result| result.explanation.is_some())
        );
    }
}

#[test]
fn lexical_explain_mode_preserves_field_parity_in_each_progressive_phase() {
    let config = TwoTierConfig {
        candidate_multiplier: 3,
        explain: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_phases, async_phases) = lexical_phase_snapshots(&query, config);
    let sync_labels = sync_phases
        .iter()
        .map(|(label, _)| *label)
        .collect::<Vec<_>>();
    let async_labels = async_phases
        .iter()
        .map(|(label, _)| *label)
        .collect::<Vec<_>>();
    assert_eq!(sync_labels, ["initial", "refined"]);
    assert_eq!(sync_labels, async_labels);
    for ((label, sync_results), (_, async_results)) in sync_phases.iter().zip(async_phases.iter()) {
        assert_result_parity(
            &format!("lexical-phase-{label}-explain"),
            sync_results,
            async_results,
        );
        assert!(
            sync_results
                .iter()
                .all(|result| result.explanation.is_some())
        );
        assert!(
            async_results
                .iter()
                .all(|result| result.explanation.is_some())
        );
    }
}

#[test]
fn fast_only_emits_matching_initial_only_phase() {
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let config = TwoTierConfig {
        fast_only: true,
        ..TwoTierConfig::default()
    };
    let (sync_phases, async_phases) = phase_labels(&query, true, config);
    assert_eq!(sync_phases, ["initial"]);
    assert_eq!(sync_phases, async_phases);
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
fn lexical_rrf_agrees_on_ordered_results_and_metrics() {
    let config = TwoTierConfig {
        candidate_multiplier: 3,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync_with_lexical(&config, &query, 3);
    let (async_results, async_metrics) = run_async_with_lexical(&config, &query, 3);
    assert_result_parity("lexical-rrf", &sync_results, &async_results);
    assert_metric_parity("lexical-rrf", &sync_metrics, &async_metrics);
}

#[test]
fn lexical_refined_explanations_are_present_on_both_sides() {
    let config = TwoTierConfig {
        candidate_multiplier: 3,
        explain: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync_with_lexical(&config, &query, 3);
    let (async_results, async_metrics) = run_async_with_lexical(&config, &query, 3);
    assert_result_parity("lexical-refined-explain", &sync_results, &async_results);
    assert_metric_parity("lexical-refined-explain", &sync_metrics, &async_metrics);
    assert!(
        sync_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
    assert!(
        async_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
}

#[test]
fn lexical_fast_only_explanations_are_present_on_both_sides() {
    let config = TwoTierConfig {
        candidate_multiplier: 3,
        fast_only: true,
        explain: true,
        ..TwoTierConfig::default()
    };
    let query = normalize(vec![1.0, 0.0, 0.0, 0.0]);
    let (sync_results, sync_metrics) = run_sync_with_lexical(&config, &query, 3);
    let (async_results, async_metrics) = run_async_with_lexical(&config, &query, 3);
    assert_result_parity("lexical-fast-only-explain", &sync_results, &async_results);
    assert_metric_parity("lexical-fast-only-explain", &sync_metrics, &async_metrics);
    assert!(
        sync_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
    assert!(
        async_results
            .iter()
            .all(|result| result.explanation.is_some())
    );
}

#[test]
fn phase1_vectors_searched_reports_the_evaluated_corpus_at_small_k() {
    assert_eq!(KNOWN_DIVERGENCES.len(), 4);
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
