//! Independent native quality retrieval and two-tier candidate union.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use asupersync::runtime::blocking_pool::BlockingPoolHandle;

use frankensearch_core::generation::EmbeddingSpaceKindV1;
use frankensearch_core::{BoundQueryEmbedding, DocId, LexicalCandidateBatch};
use frankensearch_fusion::{RrfConfig, blend_two_tier, candidate_count, rrf_fuse_for_vector_lane};

use super::{NativeAnnIndex, checkpoint, hydrate_winners, invalid, join_sources, validate_lexical};
use crate::native_ann::NativeWorker;
use crate::{Cx, Embedder, LexicalRead, ScoreSource, ScoredResult, SearchResult, VectorHit};

/// The actual query response stays with its retrieved pool. Supplementary
/// evidence must use these exact values, never a second inference of the text.
pub(super) struct RetainedCandidates<T> {
    // Keep the identity bundle out of line while this pool is held across phases.
    pub query: Option<Box<BoundQueryEmbedding>>,
    pub hits: Vec<T>,
}

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

    /// Quality-primary hybrid retrieval with vector traversal on the caller pool.
    ///
    /// The provider still runs through the ordinary identity-bound inference path.
    /// Only retained graph/exact traversal moves to the blocking pool.
    ///
    /// # Errors
    ///
    /// Has the same error surface as [`Self::search_hybrid_text_with_pool`].
    pub async fn search_hybrid_quality_text_with_pool(
        self: &Arc<Self>,
        cx: &Cx,
        embedder: &dyn Embedder,
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        let mut results = self
            .search_hybrid_text_with_pool(cx, embedder, lexical, text, k, pool)
            .await?;
        for result in &mut results {
            checkpoint(cx, "native_ann.quality_pool_result")?;
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
    /// Each retained candidate is also scored in the other owner when it has a
    /// live row there. Falling below a retrieval cutoff is not missing coverage;
    /// only genuinely absent or tombstoned rows keep the single-tier blend.
    /// Supplementation cannot add a document outside the retrieved union.
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
    /// Exclusion queries require a retained immutable cohort and are refused
    /// before any source work; see [`Self::search_hybrid_candidates`].
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
        super::query::reject_unbound_exclusions(cx, text)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let budget = candidate_count(k, 0, 3);
        let (mut fast, (mut quality_hits, batch)) = join_sources(
            cx,
            self.retrieve_candidates(cx, fast_embedder, text, budget, None, None, |_| true),
            join_sources(
                cx,
                quality
                    .0
                    .retrieve_candidates(cx, quality.1, text, budget, None, None, |_| true),
                checked_lexical(cx, lexical, text, budget),
            ),
        )
        .await?;
        let (fast_supplement, quality_supplement) = join_sources(
            cx,
            self.supplement_candidates(cx, &fast, &quality_hits.hits, None),
            quality
                .0
                .supplement_candidates(cx, &quality_hits, &fast.hits, None),
        )
        .await?;
        fast.hits.extend(fast_supplement);
        quality_hits.hits.extend(quality_supplement);
        refined_winners(
            cx,
            lexical,
            &batch,
            &fast.hits,
            &quality_hits.hits,
            k,
            RefinementFusion {
                quality_weight: 0.7,
                rrf: &RrfConfig::default(),
                is_hash,
            },
        )
        .await
    }
    /// Independently embed both tiers, then execute their retained traversals
    /// on the caller-owned blocking pool before ordinary blend/RRF/hydration.
    ///
    /// Both vector arms remain required. A failed, cancelled or abandoned worker
    /// never becomes a fast-only successful page.
    ///
    /// # Errors
    ///
    /// Propagates admission, inference, blocking-pool, lexical, and hydration errors.
    /// Exclusions have the same immutable-cohort requirement as
    /// [`Self::search_hybrid_refined_text`].
    pub async fn search_hybrid_refined_text_with_pool(
        self: &Arc<Self>,
        cx: &Cx,
        fast_embedder: &dyn Embedder,
        quality: (&Arc<Self>, &dyn Embedder),
        lexical: &dyn LexicalRead,
        text: &str,
        k: usize,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<ScoredResult>> {
        let is_hash = admit_pair(
            cx,
            self.as_ref(),
            fast_embedder,
            (quality.0.as_ref(), quality.1),
        )?;
        super::query::reject_unbound_exclusions(cx, text)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let budget = candidate_count(k, 0, 3);
        let (mut fast, (mut quality_hits, batch)) = join_sources(
            cx,
            self.retrieve_candidates(
                cx,
                fast_embedder,
                text,
                budget,
                None,
                Some((Arc::clone(self), pool.clone())),
                |_| true,
            ),
            join_sources(
                cx,
                quality.0.retrieve_candidates(
                    cx,
                    quality.1,
                    text,
                    budget,
                    None,
                    Some((Arc::clone(quality.0), pool.clone())),
                    |_| true,
                ),
                checked_lexical(cx, lexical, text, budget),
            ),
        )
        .await?;
        let (fast_supplement, quality_supplement) = join_sources(
            cx,
            self.supplement_candidates(
                cx,
                &fast,
                &quality_hits.hits,
                Some((Arc::clone(self), pool.clone())),
            ),
            quality.0.supplement_candidates(
                cx,
                &quality_hits,
                &fast.hits,
                Some((Arc::clone(quality.0), pool)),
            ),
        )
        .await?;
        fast.hits.extend(fast_supplement);
        quality_hits.hits.extend(quality_supplement);
        refined_winners(
            cx,
            lexical,
            &batch,
            &fast.hits,
            &quality_hits.hits,
            k,
            RefinementFusion {
                quality_weight: 0.7,
                rrf: &RrfConfig::default(),
                is_hash,
            },
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn retrieve_candidates<F>(
        &self,
        cx: &Cx,
        embedder: &dyn Embedder,
        text: &str,
        k: usize,
        ef: Option<usize>,
        worker: Option<(Arc<Self>, BlockingPoolHandle)>,
        accept: F,
    ) -> SearchResult<RetainedCandidates<VectorHit>>
    where
        F: Fn(&str) -> bool + Send + 'static,
    {
        checkpoint(cx, "native_ann.candidate_retrieval")?;
        self.admit_identity(embedder.identity()?)?;
        if let Some((_, pool)) = &worker {
            NativeWorker::QUERY.require_pool(cx, pool)?;
        }
        if k == 0 || self.live_count() == 0 {
            return Ok(RetainedCandidates {
                query: None,
                hits: Vec::new(),
            });
        }
        let query = Box::new(self.embed_query(cx, embedder, text).await?);
        let hits = match worker {
            Some((owner, pool)) => {
                owner
                    .search_filtered_with_pool(cx, (*query).clone(), k, ef, pool, accept)
                    .await?
            }
            None => self.search_filtered(cx, &query, k, ef, accept)?,
        };
        Ok(RetainedCandidates {
            query: Some(query),
            hits,
        })
    }

    pub(super) async fn supplement_candidates(
        &self,
        cx: &Cx,
        observed: &RetainedCandidates<VectorHit>,
        candidates: &[VectorHit],
        worker: Option<(Arc<Self>, BlockingPoolHandle)>,
    ) -> SearchResult<Vec<VectorHit>> {
        let Some(query) = observed.query.as_deref() else {
            checkpoint(cx, "native_ann.candidate_evidence.empty")?;
            return Ok(Vec::new());
        };
        let missing = missing_ids(
            observed.hits.iter().map(|hit| &hit.doc_id),
            candidates.iter().map(|hit| &hit.doc_id),
        );
        match worker {
            Some((owner, pool)) => {
                owner
                    .score_documents_with_pool(cx, query.clone(), missing, pool)
                    .await
            }
            None => self.score_documents(cx, query, &missing),
        }
    }

    /// Exact evidence for a bounded set of already selected documents. The
    /// bound query is admitted before any row lookup, including empty batches.
    /// Lookup checks the complete ID (not only its hash) and excludes tombstones.
    pub(super) fn score_documents(
        &self,
        cx: &Cx,
        query: &BoundQueryEmbedding,
        document_ids: &[DocId],
    ) -> SearchResult<Vec<VectorHit>> {
        checkpoint(cx, "native_ann.candidate_evidence.admission")?;
        self.admit_identity(query.identity())?;
        if query.vector().len() != self.owner.dimension() {
            return Err(crate::SearchError::DimensionMismatch {
                expected: self.owner.dimension(),
                found: query.vector().len(),
            });
        }
        if query.vector().iter().all(|value| *value == 0.0) {
            return Ok(Vec::new());
        }
        let mut hits = Vec::with_capacity(document_ids.len());
        for id in document_ids {
            checkpoint(cx, "native_ann.candidate_evidence.row")?;
            let Some(row) = self.owner.find_live_row_by_doc_id(id)? else {
                continue;
            };
            let physical = row.physical_index();
            let score = self.score_retained_row(physical, query.vector())?;
            if !score.is_finite() {
                return Err(invalid(
                    "score",
                    "non-finite",
                    "retained-owner candidate evidence must be finite",
                ));
            }
            hits.push(VectorHit {
                index: u32::try_from(physical).map_err(|_| {
                    invalid(
                        "physical_row",
                        "out-of-range",
                        "source row does not fit u32",
                    )
                })?,
                score,
                doc_id: id.clone(),
            });
        }
        checkpoint(cx, "native_ann.candidate_evidence.complete")?;
        Ok(hits)
    }

    pub(super) async fn score_documents_with_pool(
        self: &Arc<Self>,
        cx: &Cx,
        query: BoundQueryEmbedding,
        document_ids: Vec<DocId>,
        pool: BlockingPoolHandle,
    ) -> SearchResult<Vec<VectorHit>> {
        checkpoint(cx, "native_ann.candidate_evidence.worker")?;
        self.admit_identity(query.identity())?;
        NativeWorker::QUERY.require_pool(cx, &pool)?;
        if document_ids.is_empty() {
            return self.score_documents(cx, &query, &document_ids);
        }
        let index = Arc::clone(self);
        NativeWorker::QUERY
            .run(cx, pool, move |request| {
                index.score_documents(request, &query, &document_ids)
            })
            .await
    }
}

/// Complete only the retrieved union; an exact lookup is evidence, not another
/// retrieval. Keep the existing pool order and clone only IDs missing a score.
pub(super) fn missing_ids<'a>(
    observed: impl Iterator<Item = &'a DocId>,
    candidates: impl Iterator<Item = &'a DocId>,
) -> Vec<DocId> {
    let mut seen: BTreeSet<&str> = observed.map(DocId::as_str).collect();
    candidates
        .filter(|id| seen.insert(id.as_str()))
        .cloned()
        .collect()
}

pub(super) fn admit_pair(
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

pub(super) async fn checked_lexical(
    cx: &Cx,
    lexical: &dyn LexicalRead,
    text: &str,
    budget: usize,
) -> SearchResult<LexicalCandidateBatch> {
    let text = lexical.query_text(text);
    let response = frankensearch_core::with_cancellation(cx, "native_ann.tiered_lexical", async {
        lexical.search_candidates(cx, &text, budget).await
    })
    .await;
    checkpoint(cx, "native_ann.tiered_lexical")?;
    let batch = response?;
    validate_lexical(cx, &batch)?;
    Ok(batch)
}

/// Ranking policy and the independently admitted vector-lane classification.
/// Only native callers construct this; user tuning never chooses `is_hash`.
pub(super) struct RefinementFusion<'a> {
    pub quality_weight: f32,
    pub rrf: &'a RrfConfig,
    pub is_hash: bool,
}

pub(super) async fn refined_winners(
    cx: &Cx,
    lexical: &dyn LexicalRead,
    batch: &LexicalCandidateBatch,
    fast_hits: &[VectorHit],
    quality_hits: &[VectorHit],
    k: usize,
    fusion: RefinementFusion<'_>,
) -> SearchResult<Vec<ScoredResult>> {
    checkpoint(cx, "native_ann.tiered_blend")?;
    let RefinementFusion {
        quality_weight,
        rrf,
        is_hash,
    } = fusion;
    let vectors = blend_two_tier(fast_hits, quality_hits, quality_weight);
    let hits = rrf_fuse_for_vector_lane(batch.results(), &vectors, k, 0, rrf, is_hash);
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
        index_states(
            cx,
            provider,
            rows.iter().map(|&(id, vector)| (id, vector, true)),
            generation,
            nonce,
            format,
        )
    }

    fn index_states<'a>(
        cx: &Cx,
        provider: &Provider,
        rows: impl IntoIterator<Item = (&'a str, &'a [f32], bool)>,
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
        for (id, vector, live) in rows {
            if live {
                writer.write_record(id, vector).unwrap();
            } else {
                writer.write_tombstone_record(id, vector).unwrap();
            }
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
                        &cx,
                        &fast,
                        (&quality_index, &quality),
                        &lexical,
                        "query",
                        1,
                    )
                    .await
                    .unwrap();
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].doc_id, "late");
                assert_eq!(results[0].source, ScoreSource::Hybrid);
                assert_eq!(results[0].fast_score, Some(0.0));
                assert_eq!(results[0].quality_score, Some(quality_hit[0].score));
                assert_eq!(
                    results[0].index,
                    Some(
                        u32::try_from(
                            fast_index
                                .owner
                                .find_live_row_by_doc_id("late")
                                .unwrap()
                                .unwrap()
                                .physical_index()
                        )
                        .unwrap()
                    ),
                );
                assert_eq!(results[0].metadata.as_deref().unwrap()["snapshot"], 1);
                assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    fn opposite_fast_rows() -> Vec<(&'static str, &'static [f32])> {
        vec![
            ("a", &[1.0, 0.0]),
            ("b", &[0.875, 0.0]),
            ("c", &[0.75, 0.0]),
            ("d", &[0.625, 0.0]),
            ("e", &[0.5, 0.0]),
            ("f", &[0.375, 0.0]),
        ]
    }

    fn opposite_quality_rows() -> Vec<(&'static str, &'static [f32])> {
        vec![
            ("a", &[1.0, 0.0, 0.0]),
            ("b", &[0.0, 0.125, 0.0]),
            ("c", &[0.0, 0.25, 0.0]),
            ("d", &[0.0, 1.0, 0.0]),
            ("e", &[0.0, 0.875, 0.0]),
            ("f", &[0.0, 0.75, 0.0]),
        ]
    }

    #[test]
    fn refinement_scores_the_union_before_blending_on_exact_ann_and_worker_paths() {
        use asupersync::runtime::RuntimeBuilder;
        use asupersync::runtime::blocking_pool::BlockingPool;
        use std::time::Duration;

        let runtime = RuntimeBuilder::current_thread().build().unwrap();
        let pool = BlockingPool::new(0, 2);
        runtime.block_on(async {
            let cx = Cx::current().unwrap();
            for format in [QuantizationFormat::F16, QuantizationFormat::F32] {
                for ann in [false, true] {
                    for offload in [false, true] {
                        let fast = Provider::new("fast", vec![1.0, 0.0], true);
                        let quality = Provider::new("quality", vec![0.0, 1.0, 0.0], true);
                        let mut fast_index = index(&cx, &fast, &opposite_fast_rows(), 1, 7, format);
                        let mut quality_index =
                            index(&cx, &quality, &opposite_quality_rows(), 1, 7, format);
                        // A full-physical-row beam deliberately uses the exact
                        // fallback. Keep the ANN arm below that threshold so
                        // these variants actually traverse the dense graph.
                        fast_index.default_ef_search = 3;
                        quality_index.default_ef_search = 3;
                        assert!(fast_index.default_ef_search < fast_index.len());
                        assert!(quality_index.default_ef_search < quality_index.len());
                        let fast_index = Arc::new(if ann {
                            fast_index
                        } else {
                            NativeAnnIndex::exact(&cx, Arc::clone(&fast_index.owner)).unwrap()
                        });
                        let quality_index = Arc::new(if ann {
                            quality_index
                        } else {
                            NativeAnnIndex::exact(&cx, Arc::clone(&quality_index.owner)).unwrap()
                        });
                        let fast_query =
                            BoundQueryEmbedding::new(fast.values.clone(), fast.identity.clone())
                                .unwrap();
                        let quality_query = BoundQueryEmbedding::new(
                            quality.values.clone(),
                            quality.identity.clone(),
                        )
                        .unwrap();
                        let fast_top = fast_index.search(&cx, &fast_query, 3, None).unwrap();
                        let quality_top =
                            quality_index.search(&cx, &quality_query, 3, None).unwrap();
                        assert_eq!(
                            fast_top
                                .iter()
                                .map(|hit| hit.doc_id.as_str())
                                .collect::<Vec<_>>(),
                            ["a", "b", "c"]
                        );
                        assert_eq!(
                            quality_top
                                .iter()
                                .map(|hit| hit.doc_id.as_str())
                                .collect::<Vec<_>>(),
                            ["d", "e", "f"]
                        );
                        // The previous cutoff-as-absence behavior returns a,
                        // despite a real quality score of zero and d's one.
                        assert_eq!(blend_two_tier(&fast_top, &quality_top, 0.7)[0].doc_id, "a");
                        let lexical = Lexical::default();
                        let results = if offload {
                            fast_index
                                .search_hybrid_refined_text_with_pool(
                                    &cx,
                                    &fast,
                                    (&quality_index, &quality),
                                    &lexical,
                                    "query",
                                    1,
                                    pool.handle(),
                                )
                                .await
                                .unwrap()
                        } else {
                            fast_index
                                .search_hybrid_refined_text(
                                    &cx,
                                    &fast,
                                    (&quality_index, &quality),
                                    &lexical,
                                    "query",
                                    1,
                                )
                                .await
                                .unwrap()
                        };
                        assert_eq!(results.len(), 1);
                        assert_eq!(
                            results[0].doc_id, "d",
                            "format={format:?} ann={ann} offload={offload}"
                        );
                        assert_eq!(results[0].fast_score, Some(0.625));
                        assert_eq!(results[0].quality_score, Some(1.0));
                        assert_eq!(
                            results[0].index,
                            Some(
                                u32::try_from(
                                    fast_index
                                        .owner
                                        .find_live_row_by_doc_id("d")
                                        .unwrap()
                                        .unwrap()
                                        .physical_index()
                                )
                                .unwrap()
                            )
                        );
                        assert_eq!(fast.calls.load(Ordering::SeqCst), 1);
                        assert_eq!(quality.calls.load(Ordering::SeqCst), 1);
                        assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
                    }
                }
            }
        });
        assert!(pool.shutdown_and_wait(Duration::from_secs(10)));
    }

    #[test]
    fn refinement_keeps_absent_and_tombstoned_quality_evidence_missing() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for tombstone in [false, true] {
                let fast = Provider::new("fast", vec![1.0, 0.0], true);
                let quality = Provider::new("quality", vec![0.0, 1.0, 0.0], true);
                let fast_index = index(
                    &cx,
                    &fast,
                    &opposite_fast_rows(),
                    1,
                    7,
                    QuantizationFormat::F32,
                );
                let rows = opposite_quality_rows();
                let quality_index = index_states(
                    &cx,
                    &quality,
                    rows.iter()
                        .filter(|(id, _)| tombstone || *id != "a")
                        .map(|&(id, vector)| (id, vector, id != "a")),
                    1,
                    7,
                    QuantizationFormat::F32,
                );
                let results = fast_index
                    .search_hybrid_refined_text(
                        &cx,
                        &fast,
                        (&quality_index, &quality),
                        &Lexical::default(),
                        "query",
                        1,
                    )
                    .await
                    .unwrap();
                assert_eq!(results[0].doc_id, "a");
                assert_eq!(results[0].fast_score, Some(1.0));
                assert!(results[0].quality_score.is_none());
                assert_eq!(results[0].source, ScoreSource::SemanticFast);
            }
        });
    }

    #[test]
    fn candidate_evidence_requires_the_exact_tier_and_ignores_zero_signal() {
        let cx = Cx::for_testing();
        let fast = Provider::new("fast", vec![1.0, 0.0], true);
        let quality = Provider::new("quality", vec![0.0, 1.0], true);
        let owner = index(
            &cx,
            &fast,
            &opposite_fast_rows(),
            1,
            7,
            QuantizationFormat::F32,
        );
        let foreign = BoundQueryEmbedding::new(quality.values, quality.identity).unwrap();
        assert!(owner.score_documents(&cx, &foreign, &[]).is_err());
        assert!(owner.score_documents(&cx, &foreign, &["a".into()]).is_err());
        let zero = BoundQueryEmbedding::new(vec![0.0, -0.0], fast.identity).unwrap();
        assert!(
            owner
                .score_documents(&cx, &zero, &["a".into()])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn quality_primary_needs_no_fast_owner_and_labels_quality_scores() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new("quality", vec![0.0, 1.0], true);
            let owner = index(
                &cx,
                &provider,
                &[("late", &[0.0, 0.75])],
                1,
                7,
                QuantizationFormat::F32,
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
    fn zero_quality_query_preserves_lexical_winners_without_semantic_evidence() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let provider = Provider::new("quality", vec![0.0, -0.0, 0.0], true);
            let native = index(
                &cx,
                &provider,
                &[
                    ("late", &[0.0, 1.0, 0.0]),
                    ("vector-only", &[1.0, 0.0, 0.0]),
                ],
                1,
                7,
                QuantizationFormat::F32,
            );
            let exact = NativeAnnIndex::exact(&cx, Arc::clone(&native.owner)).unwrap();
            for owner in [exact, native] {
                for include_late in [false, true] {
                    let lexical = Lexical {
                        include_late,
                        ..Lexical::default()
                    };
                    let hits = owner
                        .search_hybrid_quality_text(&cx, &provider, &lexical, "query", 3)
                        .await
                        .unwrap();
                    assert_eq!(hits.len(), usize::from(include_late));
                    if include_late {
                        let hit = &hits[0];
                        assert_eq!(hit.doc_id, "late");
                        assert_eq!(hit.source, ScoreSource::Lexical);
                        assert_eq!(hit.lexical_score, Some(10.0));
                        assert!(hit.fast_score.is_none() && hit.quality_score.is_none());
                        assert!(hit.index.is_none());
                        assert_eq!(hit.metadata.as_deref().unwrap()["snapshot"], 1);
                    }
                    assert_eq!(lexical.calls.load(Ordering::SeqCst), 1);
                }
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
        });
    }

    #[test]
    fn equal_dimensions_never_authorize_reusing_the_fast_query_in_quality_space() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast = Provider::new("fast", vec![1.0, 0.0], true);
            let quality = Provider::new("quality", vec![0.0, 1.0], true);
            let a = index(
                &cx,
                &fast,
                &[("fast", &[1.0, 0.0])],
                1,
                7,
                QuantizationFormat::F32,
            );
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
                    &cx,
                    &fast,
                    (&b, &quality),
                    &Lexical::default(),
                    "query",
                    3,
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
                let b = index(
                    &cx,
                    &quality,
                    &[],
                    generation,
                    nonce,
                    QuantizationFormat::F32,
                );
                for k in [0, 1] {
                    let error = a
                        .search_hybrid_refined_text(
                            &cx,
                            &fast,
                            (&b, &quality),
                            &lexical,
                            "query",
                            k,
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
            let b = index(
                &cx,
                &quality,
                &[("late", &[0.0, 1.0])],
                1,
                7,
                QuantizationFormat::F32,
            );
            quality.substitute = true;
            let error = a
                .search_hybrid_refined_text(
                    &cx,
                    &fast,
                    (&b, &quality),
                    &Lexical::default(),
                    "query",
                    1,
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
            let b = index(
                &cx,
                &quality,
                &[("late", &[0.0, 0.75])],
                1,
                7,
                QuantizationFormat::F32,
            );
            let results = a
                .search_hybrid_refined_text(
                    &cx,
                    &fast,
                    (&b, &quality),
                    &Lexical::default(),
                    "query",
                    1,
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
            let b = index(
                &cx,
                &quality,
                &[("late", &[0.0, 1.0])],
                1,
                7,
                QuantizationFormat::F32,
            );
            let results = a
                .search_hybrid_refined_text(
                    &cx,
                    &fast,
                    (&b, &quality),
                    &Lexical::default(),
                    "query",
                    4,
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
                    &cx,
                    &fast,
                    (&mixed, &semantic),
                    &Lexical::default(),
                    "query",
                    0,
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
                &cx,
                &fast,
                &[("shared", &[0.5, 0.5])],
                1,
                7,
                QuantizationFormat::F32,
            );
            let b = index(
                &cx,
                &quality,
                &[("shared", &[0.0, 0.75])],
                1,
                7,
                QuantizationFormat::F32,
            );
            let original = a.search_text(&cx, &fast, "query", 1, None).await.unwrap();
            let results = a
                .search_hybrid_refined_text(
                    &cx,
                    &fast,
                    (&b, &quality),
                    &Lexical::default(),
                    "query",
                    1,
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
