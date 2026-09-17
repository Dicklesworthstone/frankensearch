//! Independent native quality retrieval and two-tier candidate union.

use std::collections::BTreeMap;

use frankensearch_core::LexicalCandidateBatch;
use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_fusion::{RrfConfig, blend_two_tier, candidate_count, rrf_fuse_for_vector_lane};

use super::{NativeAnnIndex, checkpoint, hydrate_winners, invalid, join_sources, validate_lexical};
use crate::{Cx, Embedder, LexicalRead, ScoreSource, ScoredResult, SearchResult, VectorHit};

impl NativeAnnIndex {
    /// Use this native owner as the primary quality retrieval arm.
    ///
    /// No fast index or fast embedder is required. The quality graph retrieves
    /// candidates directly, rather than rescoring a fast-selected pool. The
    /// returned row indices belong to this owner, and raw semantic scores are
    /// reported in `quality_score`, not `fast_score`. Hash controls keep their
    /// nonsemantic source label and do not populate either semantic score.
    ///
    /// # Errors
    ///
    /// Propagates the identity, retrieval, hydration and cancellation errors of
    /// [`Self::search_hybrid_text`]. No fallback model or reader is substituted.
    pub async fn search_hybrid_quality_text(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        let mut results = self
            .search_hybrid_text(cx, embedder, lexical, text, k)
            .await?;
        for result in &mut results {
            checkpoint(cx, "native_ann.quality_result")?;
            result.quality_score = result.fast_score.take();
            if result.source == ScoreSource::SemanticFast {
                result.source = ScoreSource::SemanticQuality;
            }
        }
        Ok(results)
    }

    /// Independently retrieve both native tiers and fuse their candidate union.
    ///
    /// `self` and `fast_embedder` form the fast arm; `quality` supplies its own
    /// retained owner and embedder. Both configured identities, their complete
    /// artifact generation identities, and document-ID semantics are checked
    /// before any inference or lexical search, even for `k == 0`. Actual bound
    /// provider responses are checked again by each arm before graph traversal.
    ///
    /// Each source requests three times `k` candidates. Vector pools are
    /// independently normalized and blended with 0.7 quality / 0.3 fast, using
    /// the existing union-preserving blend policy; lexical ranks are then
    /// combined by default RRF. Quality-only candidates are not restricted to
    /// the fast pool. Missing coverage is permitted, not counted as complete.
    ///
    /// Raw per-tier scores are preserved. A row index belongs to the fast owner
    /// when `fast_score` is present, otherwise to the quality owner when
    /// `quality_score` is present. Two hash-control arms are supported, but their
    /// merged results omit ambiguous row indices and semantic score fields.
    /// Mixing a semantic arm with a hash-control arm is rejected rather than
    /// mislabelling a mixed vector rank as wholly semantic or wholly control.
    ///
    /// Provider futures are cooperatively polled without spawning tasks. Graph
    /// traversal remains synchronous on the caller's CPU/blocking lane. The
    /// lexical scoring snapshot stays pinned through winner hydration. Matching
    /// vector generations does NOT prove that a separately supplied lexical
    /// reader belongs to that generation; composite reader selection remains
    /// the caller's responsibility. This API neither publishes generations nor
    /// changes the default `TwoTierSearcher` backend.
    ///
    /// # Errors
    ///
    /// Propagates admission, inference, graph, lexical and hydration errors.
    /// A failed or cancelled arm never becomes a partial-success fallback.
    pub async fn search_hybrid_refined_text(
        &self,
        cx: &Cx,
        fast_embedder: &dyn Embedder,
        quality: (&Self, &dyn Embedder),
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
    ) -> SearchResult<Vec<ScoredResult>> {
        let is_hash = admit_pair(cx, self, fast_embedder, quality)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let budget = candidate_count(k, 0, 3);
        let (fast_hits, (quality_hits, batch)) = join_sources(
            cx,
            self.search_text(cx, fast_embedder, text, budget, None),
            join_sources(
                cx,
                quality.0.search_text(cx, quality.1, text, budget, None),
                checked_lexical(cx, lexical, text, budget),
            ),
        )
        .await?;
        refined_winners(cx, lexical, &batch, &fast_hits, &quality_hits, k, is_hash).await
    }
}

fn admit_pair(
    cx: &Cx,
    fast: &NativeAnnIndex,
    fast_embedder: &dyn Embedder,
    quality: (&NativeAnnIndex, &dyn Embedder),
) -> SearchResult<bool> {
    checkpoint(cx, "native_ann.tier_pair")?;
    let fast_identity = fast_embedder.identity()?;
    let quality_identity = quality.1.identity()?;
    fast.admit_identity(fast_identity)?;
    quality.0.admit_identity(quality_identity)?;
    if fast.owner_witness().generation != quality.0.owner_witness().generation {
        return Err(invalid(
            "tier_generation",
            "mismatch",
            "native fast and quality owners must bind the same complete artifact generation",
        ));
    }
    if fast_identity.input.doc_id_semantics != quality_identity.input.doc_id_semantics {
        return Err(invalid(
            "tier_doc_id_semantics",
            "mismatch",
            "candidate union requires one canonical document-ID contract across both tiers",
        ));
    }
    let fast_is_hash = matches!(fast_identity.space.kind, EmbeddingSpaceKindV1::HashControl);
    let quality_is_hash = matches!(
        quality_identity.space.kind,
        EmbeddingSpaceKindV1::HashControl
    );
    if fast_is_hash != quality_is_hash {
        return Err(invalid(
            "tier_kind",
            "mixed-control-and-semantic",
            "a fused vector rank cannot represent both hash-control and semantic provenance",
        ));
    }
    Ok(fast_is_hash)
}

async fn checked_lexical(
    cx: &Cx,
    lexical: &dyn LexicalRead,
    text: &str,
    budget: usize,
) -> SearchResult<LexicalCandidateBatch> {
    let response = lexical.search_candidates(cx, text, budget).await;
    checkpoint(cx, "native_ann.tiered_lexical")?;
    let batch = response?;
    validate_lexical(cx, &batch)?;
    Ok(batch)
}

async fn refined_winners(
    cx: &Cx,
    lexical: &dyn LexicalRead,
    batch: &LexicalCandidateBatch,
    fast_hits: &[VectorHit],
    quality_hits: &[VectorHit],
    k: usize,
    is_hash: bool,
) -> SearchResult<Vec<ScoredResult>> {
    checkpoint(cx, "native_ann.tiered_blend")?;
    let vectors = blend_two_tier(fast_hits, quality_hits, 0.7);
    let hits =
        rrf_fuse_for_vector_lane(batch.results(), &vectors, k, 0, &RrfConfig::default(), is_hash);
    let mut results = hydrate_winners(cx, lexical, hits, batch).await?;
    let fast: BTreeMap<_, _> = fast_hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit))
        .collect();
    let quality: BTreeMap<_, _> = quality_hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit))
        .collect();
    for result in &mut results {
        checkpoint(cx, "native_ann.tiered_result")?;
        let fast_hit = fast.get(result.doc_id.as_str());
        let quality_hit = quality.get(result.doc_id.as_str());
        // Never expose the normalized blend as a raw score from either model.
        result.fast_score = if is_hash {
            None
        } else {
            fast_hit.map(|hit| hit.score)
        };
        result.quality_score = if is_hash {
            None
        } else {
            quality_hit.map(|hit| hit.score)
        };
        result.index = if is_hash {
            None
        } else {
            fast_hit.or(quality_hit).map(|hit| hit.index)
        };
        if result.lexical_score.is_none() && !is_hash && quality_hit.is_some() {
            result.source = ScoreSource::SemanticQuality;
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use frankensearch_core::generation::{
        ArtifactGenerationIdentityV1, EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1,
        QuantizationFormat,
    };
    use frankensearch_core::traits::{IdentityBoundEmbedding, ModelCategory, SearchFuture};
    use frankensearch_index::native_hnsw::HnswParams;
    use frankensearch_index::{FsviV2IdentityBinding, ValidatedFsviBytes, VectorIndex};

    use crate::SearchError;

    struct Provider {
        identity: EmbeddingIdentityBundleV1,
        values: Vec<f32>,
        calls: AtomicUsize,
        substitute: bool,
    }

    impl Provider {
        fn new(name: &str, values: Vec<f32>, semantic: bool) -> Self {
            let dimension = u32::try_from(values.len()).unwrap();
            let mut identity = EmbeddingIdentityBundleV1::explicit_test_model(name, dimension);
            if semantic {
                // Synthetic semantic-shaped identity, not evidence of a real model.
                identity.space.kind = EmbeddingSpaceKindV1::Semantic;
                identity.space.hash_control = None;
                identity.space.artifact_manifest_fingerprint = "a".repeat(64);
                identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
                    role: "weights".to_owned(),
                    sha256: "b".repeat(64),
                    size: 1,
                }];
                identity.producer.space_fingerprint = identity.space.fingerprint();
            }
            identity.validate().unwrap();
            Self {
                identity,
                values,
                calls: AtomicUsize::new(0),
                substitute: false,
            }
        }
    }

    impl Embedder for Provider {
        fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move { Ok(self.values.clone()) })
        }
        fn embed_bound<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
        ) -> SearchFuture<'a, IdentityBoundEmbedding> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let mut identity = self.identity.clone();
                if self.substitute {
                    identity.producer.backend = "substituted-provider".to_owned();
                }
                Ok(IdentityBoundEmbedding {
                    values: self.values.clone(),
                    identity,
                })
            })
        }
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            Ok(&self.identity)
        }
        fn id(&self) -> &str {
            &self.identity.space.logical_model_id
        }
        fn model_name(&self) -> &str {
            self.id()
        }
        fn dimension(&self) -> usize {
            self.values.len()
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn is_semantic(&self) -> bool {
            matches!(self.identity.space.kind, EmbeddingSpaceKindV1::Semantic)
        }
        fn category(&self) -> ModelCategory {
            if self.is_semantic() {
                ModelCategory::TransformerEmbedder
            } else {
                ModelCategory::HashEmbedder
            }
        }
    }

    #[derive(Default)]
    struct Lexical {
        calls: AtomicUsize,
        include_late: bool,
    }

    impl LexicalRead for Lexical {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            _text: &'a str,
            limit: usize,
        ) -> SearchFuture<'a, Vec<ScoredResult>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if self.include_late && limit > 0 {
                    Ok(vec![ScoredResult {
                        doc_id: "late".into(),
                        score: 10.0,
                        source: ScoreSource::Lexical,
                        index: None,
                        fast_score: None,
                        quality_score: None,
                        lexical_score: Some(10.0),
                        rerank_score: None,
                        explanation: None,
                        metadata: Some(Arc::new(serde_json::json!({"snapshot": 1}))),
                    }])
                } else {
                    Ok(Vec::new())
                }
            })
        }
        fn doc_count(&self) -> SearchResult<usize> {
            Ok(usize::from(self.include_late))
        }
    }

    fn index(
        cx: &Cx,
        provider: &Provider,
        rows: &[(&str, &[f32])],
        generation: u64,
        nonce: u8,
        format: QuantizationFormat,
    ) -> NativeAnnIndex {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier.fsvi");
        let mut identity = provider.identity.clone();
        identity.storage.format = "fsvi-v2".to_owned();
        identity.storage.quantization = format;
        identity.storage.endianness = "little-endian".to_owned();
        let binding = FsviV2IdentityBinding::new(
            ArtifactGenerationIdentityV1::new(generation, [nonce; 16]).unwrap(),
            identity.freeze().unwrap(),
        )
        .unwrap();
        let mut writer = VectorIndex::create_v2(&path, binding.clone()).unwrap();
        for (id, vector) in rows {
            writer.write_record(id, vector).unwrap();
        }
        writer.finish().unwrap();
        let bytes: Arc<[u8]> = std::fs::read(path).unwrap().into();
        let owner = Arc::new(ValidatedFsviBytes::from_arc(bytes, &binding).unwrap());
        NativeAnnIndex::build(cx, owner, HnswParams::default(), 7).unwrap()
    }

    fn fast_rows() -> Vec<(&'static str, &'static [f32])> {
        vec![
            ("a", &[1.0, 0.0]),
            ("b", &[0.9, 0.1]),
            ("c", &[0.8, 0.2]),
            ("late", &[0.0, 1.0]),
        ]
    }

    #[test]
    fn independent_quality_retrieval_finds_a_winner_outside_the_fast_pool() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
                let fast = Provider::new("fast", vec![1.0, 0.0], true);
                let quality = Provider::new("quality", vec![0.0, 1.0, 0.0], true);
                let fast_index = index(&cx, &fast, &fast_rows(), 1, 7, format);
                let quality_index =
                    index(&cx, &quality, &[("late", &[0.0, 1.0, 0.0])], 1, 7, format);
                let pool = fast_index
                    .search_text(&cx, &fast, "query", 3, None)
                    .await
                    .unwrap();
                assert!(pool.iter().all(|hit| hit.doc_id != "late"));
                let quality_hit = quality_index
                    .search_text(&cx, &quality, "query", 1, None)
                    .await
                    .unwrap();
                let lexical = Lexical {
                    include_late: true,
                    ..Lexical::default()
                };
                let results = fast_index
                    .search_hybrid_refined_text(
                        &cx, &fast, (&quality_index, &quality), &lexical, "query", 1,
                    )
                    .await
                    .unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].doc_id, "late");
                assert_eq!(results[0].source, ScoreSource::Hybrid);
                assert!(results[0].fast_score.is_none());
                assert_eq!(results[0].quality_score, Some(quality_hit[0].score));
                assert_eq!(results[0].index, Some(quality_hit[0].index));
                assert_eq!(results[0].metadata.as_deref().unwrap()["snapshot"], 1);
                assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn quality_primary_needs_no_fast_owner_and_labels_quality_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new("quality", vec![0.0, 1.0], true);
            let owner = index(
                &cx, &provider, &[("late", &[0.0, 0.75])], 1, 7, QuantizationFormat::F32,
            );
            let results = owner
                .search_hybrid_quality_text(&cx, &provider, &Lexical::default(), "query", 1)
                .await
                .unwrap();
            assert_eq!(results[0].doc_id, "late");
            assert_eq!(results[0].source, ScoreSource::SemanticQuality);
            assert_eq!(results[0].quality_score, Some(0.75));
            assert!(results[0].fast_score.is_none());
        });
    }

    #[test]
    fn equal_dimensions_never_authorize_reusing_the_fast_query_in_quality_space() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], true);
            let quality = Provider::new("quality", vec![0.0, 1.0], true);
            let a = index(&cx, &fast, &[("fast", &[1.0, 0.0])], 1, 7, QuantizationFormat::F32);
            let b = index(
                &cx,
                &quality,
                &[("wrong-if-reused", &[1.0, 0.0]), ("late", &[0.0, 1.0])],
                1,
                7,
                QuantizationFormat::F32,
            );
            let results = a
                .search_hybrid_refined_text(
                    &cx, &fast, (&b, &quality), &Lexical::default(), "query", 3,
                )
                .await
                .unwrap();
            let late = results.iter().find(|hit| hit.doc_id == "late").unwrap();
            assert_eq!(late.quality_score, Some(1.0));
            let wrong = results
                .iter()
                .find(|hit| hit.doc_id == "wrong-if-reused")
                .unwrap();
            assert_eq!(wrong.quality_score, Some(0.0));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn generation_and_advertised_identity_mismatches_refuse_before_all_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], false);
            let quality = Provider::new("quality", vec![0.0, 1.0], false);
            let a = index(&cx, &fast, &[], 1, 7, QuantizationFormat::F32);
            let lexical = Lexical::default();
            for (generation, nonce) in [(2, 7), (1, 8)] {
                let b = index(&cx, &quality, &[], generation, nonce, QuantizationFormat::F32);
                for k in [0, 1] {
                    let error = a
                        .search_hybrid_refined_text(
                            &cx, &fast, (&b, &quality), &lexical, "query", k,
                        )
                        .await
                        .unwrap_err();
                    assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                        if field == "native_ann.tier_generation"));
                }
            }
            let b = index(&cx, &quality, &[], 1, 7, QuantizationFormat::F32);
            assert!(
                a.search_hybrid_refined_text(&cx, &fast, (&b, &fast), &lexical, "query", 0)
                    .await
                    .is_err()
            );
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn actual_quality_producer_substitution_is_not_accepted_or_retried() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], true);
            let mut quality = Provider::new("quality", vec![0.0, 1.0], true);
            let a = index(&cx, &fast, &fast_rows(), 1, 7, QuantizationFormat::F32);
            let b = index(&cx, &quality, &[("late", &[0.0, 1.0])], 1, 7, QuantizationFormat::F32);
            quality.substitute = true;
            let error = a
                .search_hybrid_refined_text(
                    &cx, &fast, (&b, &quality), &Lexical::default(), "query", 1,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                if field == "query_embedding.native_ann.producer_conformance"));
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn empty_fast_and_partial_quality_remain_functional_without_false_fast_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], true);
            let quality = Provider::new("quality", vec![0.0, 1.0], true);
            let a = index(&cx, &fast, &[], 1, 7, QuantizationFormat::F32);
            let b = index(&cx, &quality, &[("late", &[0.0, 0.75])], 1, 7, QuantizationFormat::F32);
            let results = a
                .search_hybrid_refined_text(
                    &cx, &fast, (&b, &quality), &Lexical::default(), "query", 1,
                )
                .await
                .unwrap();
            assert_eq!(results[0].source, ScoreSource::SemanticQuality);
            assert_eq!(results[0].quality_score, Some(0.75));
            assert!(results[0].fast_score.is_none());
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn hash_pairs_never_publish_semantic_scores_or_ambiguous_row_indices() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast-control", vec![1.0, 0.0], false);
            let quality = Provider::new("quality-control", vec![0.0, 1.0], false);
            let a = index(&cx, &fast, &fast_rows(), 1, 7, QuantizationFormat::F32);
            let b = index(&cx, &quality, &[("late", &[0.0, 1.0])], 1, 7, QuantizationFormat::F32);
            let results = a
                .search_hybrid_refined_text(
                    &cx, &fast, (&b, &quality), &Lexical::default(), "query", 4,
                )
                .await
                .unwrap();
            assert!(results.iter().all(|hit| {
                hit.source == ScoreSource::HashControl
                    && hit.fast_score.is_none()
                    && hit.quality_score.is_none()
                    && hit.index.is_none()
            }));
            let semantic = Provider::new("quality", vec![0.0, 1.0], true);
            let mixed = index(&cx, &semantic, &[], 1, 7, QuantizationFormat::F32);
            let error = a
                .search_hybrid_refined_text(
                    &cx, &fast, (&mixed, &semantic), &Lexical::default(), "query", 0,
                )
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                if field == "native_ann.tier_kind"));
            assert_eq!(semantic.calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn shared_candidates_retain_both_raw_scores_and_the_fast_owners_row() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], true);
            let quality = Provider::new("quality", vec![0.0, 1.0], true);
            let a = index(
                &cx, &fast, &[("shared", &[0.5, 0.5])],
                1, 7, QuantizationFormat::F32,
            );
            let b = index(
                &cx, &quality, &[("shared", &[0.0, 0.75])],
                1, 7, QuantizationFormat::F32,
            );
            let original = a.search_text(&cx, &fast, "query", 1, None).await.unwrap();
            let results = a
                .search_hybrid_refined_text(
                    &cx, &fast, (&b, &quality), &Lexical::default(), "query", 1,
                )
                .await
                .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].doc_id, "shared");
            assert_eq!(results[0].fast_score, Some(0.5));
            assert_eq!(results[0].quality_score, Some(0.75));
            assert_eq!(results[0].index, Some(original[0].index));
            assert_eq!(results[0].source, ScoreSource::SemanticQuality);
        });
    }

    #[test]
    fn document_identity_contracts_must_agree_before_union_or_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], true);
            let mut quality = Provider::new("quality", vec![0.0, 1.0], true);
            quality.identity.input.doc_id_semantics = "different-document-namespace".to_owned();
            quality.identity.space.input_contract_fingerprint =
                quality.identity.input.fingerprint();
            quality.identity.producer.space_fingerprint = quality.identity.space.fingerprint();
            quality.identity.validate().unwrap();
            let a = index(&cx, &fast, &[], 1, 7, QuantizationFormat::F32);
            let b = index(&cx, &quality, &[], 1, 7, QuantizationFormat::F32);
            let lexical = Lexical::default();
            let error = a
                .search_hybrid_refined_text(&cx, &fast, (&b, &quality), &lexical, "query", 1)
                .await
                .unwrap_err();
            assert!(matches!(error, SearchError::InvalidConfig { ref field, .. }
                if field == "native_ann.tier_doc_id_semantics"));
            assert_eq!(fast.calls.load(Ordering::SeqCst), 0);
            assert_eq!(quality.calls.load(Ordering::SeqCst), 0);
            assert_eq!(lexical.calls.load(Ordering::SeqCst), 0);
        });
    }
}
