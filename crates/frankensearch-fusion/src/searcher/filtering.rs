//! Request-owned metadata observations and eligible-candidate refill.
//!
//! The caller pins the lexical source after identity admission. A filter never
//! becomes mutable searcher state, and semantic or graph-only candidates use
//! selected-document lookup rather than pretending to be lexical matches.

use ahash::{AHashMap, AHashSet};
use frankensearch_core::DocId;
use frankensearch_core::filter::SearchFilter;

use super::{
    Arc, Cx, LexicalCandidateBatch, LexicalRead, NormalizedExclusions, ScoredResult, SearchError,
    SearchResult, VectorHit, cancellation_checkpoint, refill_lexical_candidates,
    refill_vector_after_exclusions, should_exclude_document,
};

/// Keep observations outside a child future when an enclosing search owns its
/// deadline. Dropping that child must not erase facts needed to validate a
/// previously published fallback phase.
pub(crate) struct RequestFilter<'a> {
    predicate: Option<&'a dyn SearchFilter>,
    metadata: AHashMap<DocId, Option<Arc<serde_json::Value>>>,
}

/// The lexical pool and scoring snapshot that must remain eligible while a
/// semantic arm learns metadata. Borrowing keeps ownership with the caller's
/// phase; reconciliation never replaces the retained lexical reader.
pub(super) struct LexicalRevalidation<'a> {
    pub(super) query: &'a str,
    pub(super) target: usize,
    pub(super) results: &'a mut Option<Vec<ScoredResult>>,
    pub(super) context: &'a mut Option<super::LexicalHydrationContext>,
}

impl<'a> RequestFilter<'a> {
    pub(crate) fn new(predicate: Option<&'a dyn SearchFilter>) -> Self {
        Self {
            predicate,
            metadata: AHashMap::new(),
        }
    }

    pub(super) const fn is_active(&self) -> bool {
        self.predicate.is_some()
    }

    fn observe(&mut self, results: &[ScoredResult]) {
        for result in results {
            let observed = self.metadata.entry(result.doc_id.clone()).or_default();
            if observed.is_none() {
                observed.clone_from(&result.metadata);
            }
        }
    }

    async fn hydrate(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        ids: &[&str],
    ) -> SearchResult<()> {
        let Some(lexical) = lexical else {
            return Ok(());
        };
        let mut seen = AHashSet::new();
        let unknown: Vec<_> = ids
            .iter()
            .copied()
            .filter(|id| self.metadata.get(*id).is_none_or(Option::is_none) && seen.insert(*id))
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        cancellation_checkpoint(cx, "filter_metadata_lookup")?;
        let values = frankensearch_core::with_cancellation(
            cx,
            "filter_metadata_lookup",
            lexical.metadata_for_documents(cx, &unknown),
        )
        .await?;
        cancellation_checkpoint(cx, "filter_metadata_to_candidates")?;
        if values.len() != unknown.len() {
            return Err(SearchError::InvalidConfig {
                field: "search.filter_metadata".to_owned(),
                value: "cardinality".to_owned(),
                reason: "selected-document metadata must preserve the requested cardinality"
                    .to_owned(),
            });
        }
        for (id, value) in unknown.into_iter().zip(values) {
            let observed = self.metadata.entry(id.into()).or_default();
            if observed.is_none() {
                *observed = value;
            }
        }
        Ok(())
    }

    fn includes(&self, cx: &Cx, id: &str) -> SearchResult<bool> {
        cancellation_checkpoint(cx, "filter_candidate")?;
        let included = self.predicate.is_none_or(|predicate| {
            predicate.matches(id, self.metadata.get(id).and_then(Option::as_deref))
        });
        cancellation_checkpoint(cx, "filter_candidate_to_search")?;
        Ok(included)
    }

    /// A failed refinement may have learned that an Initial candidate is no
    /// longer eligible. Check the entire retained pool before allowing either
    /// a fallback response or a final ranker to reuse it. This consults only
    /// observations already made by the request; failure recovery must not
    /// start another metadata lookup or change the original ranking.
    pub(crate) fn all_scored_eligible(
        &self,
        cx: &Cx,
        results: &[ScoredResult],
    ) -> SearchResult<bool> {
        if !self.is_active() {
            return Ok(true);
        }
        for result in results {
            if !self.includes(cx, &result.doc_id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn attach_metadata(&self, results: &mut [ScoredResult]) {
        if self.is_active() {
            for result in results {
                if let Some(Some(metadata)) = self.metadata.get(&result.doc_id) {
                    result.metadata = Some(Arc::clone(metadata));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn refill_lexical_with_exclusions(
        &mut self,
        cx: &Cx,
        lexical: &dyn LexicalRead,
        query: &str,
        target: usize,
        batch: LexicalCandidateBatch,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    ) -> SearchResult<LexicalCandidateBatch> {
        if !self.is_active() {
            return if let Some(exclusions) = exclusions {
                super::refill_lexical_after_exclusions(
                    cx, lexical, query, target, batch, exclusions, text_fn,
                )
                .await
            } else {
                Ok(batch)
            };
        }
        self.refill_lexical(cx, lexical, query, target, batch, |id| {
            exclusions.map_or(Ok(true), |exclusions| {
                should_exclude_document(id, exclusions, text_fn, "lexical")
                    .map(|excluded| !excluded)
            })
        })
        .await
    }

    /// A later retrieval arm can provide a document's first known metadata.
    /// Recheck the retained lexical pool against that observation and, only
    /// when membership changed, refill from the same pinned publication.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn revalidate_lexical(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        query: &str,
        target: usize,
        results: &mut Option<Vec<ScoredResult>>,
        context: &mut Option<super::LexicalHydrationContext>,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    ) -> SearchResult<()> {
        if !self.is_active() {
            return Ok(());
        }
        let (Some(lexical), Some(retained)) = (lexical, results.as_mut()) else {
            return Ok(());
        };
        let mut changed = false;
        for result in retained.iter() {
            if !self.includes(cx, &result.doc_id)? {
                changed = true;
                break;
            }
        }
        if !changed {
            self.attach_metadata(retained);
            return Ok(());
        }
        let batch = frankensearch_core::with_cancellation(
            cx,
            "filter_lexical_revalidation",
            lexical.search_candidates(cx, query, target),
        )
        .await?;
        let batch = self
            .refill_lexical_with_exclusions(cx, lexical, query, target, batch, exclusions, text_fn)
            .await?;
        let (replacements, pin) = batch.into_parts();
        *retained = replacements;
        *context = pin;
        Ok(())
    }

    /// Restore every examined candidate's actual metadata before making a
    /// membership decision. Retain rejected observations too, so another
    /// retrieval arm cannot reintroduce a document with a permissive None.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn refill_lexical(
        &mut self,
        cx: &Cx,
        lexical: &dyn LexicalRead,
        query: &str,
        target: usize,
        mut batch: LexicalCandidateBatch,
        include: impl Fn(&str) -> SearchResult<bool>,
    ) -> SearchResult<LexicalCandidateBatch> {
        if !self.is_active() {
            return refill_lexical_candidates(cx, lexical, query, target, batch, include).await;
        }
        // Wider searches replace the complete lexical prefix. Preserve known
        // observations from earlier arms, including a rejection that caused
        // revalidation, but replace this call's observations on every refill.
        // Unknown values must remain eligible for a later metadata upgrade.
        let retained_metadata: AHashSet<_> = self
            .metadata
            .iter()
            .filter_map(|(id, value)| value.as_ref().map(|_| id.clone()))
            .collect();
        let mut limit = target;
        loop {
            cancellation_checkpoint(cx, "filter_lexical_candidates")?;
            let (mut results, context) = batch.into_parts();
            let exhausted = results.len() < limit;
            if let Some(context) = context.as_ref() {
                frankensearch_core::with_cancellation(
                    cx,
                    "filter_lexical_hydration",
                    lexical.hydrate_candidates(cx, Some(context), &mut results),
                )
                .await?;
            }
            self.metadata.retain(|id, _| retained_metadata.contains(id));
            self.observe(&results);
            let ids: Vec<_> = results.iter().map(|hit| hit.doc_id.as_str()).collect();
            self.hydrate(cx, Some(lexical), &ids).await?;
            self.attach_metadata(&mut results);
            let mut seen = AHashSet::new();
            let mut retained = 0;
            for rank in 0..results.len() {
                if retained == target {
                    break;
                }
                let id = results[rank].doc_id.as_str();
                if !seen.insert(id.to_owned()) || !self.includes(cx, id)? {
                    continue;
                }
                let included = include(id);
                cancellation_checkpoint(cx, "filter_lexical_to_search")?;
                if included? {
                    results.swap(retained, rank);
                    retained += 1;
                }
            }
            results.truncate(retained);
            let next = limit.saturating_mul(2).max(limit.saturating_add(1));
            if retained == target || exhausted || next <= limit {
                return Ok(match context {
                    Some(context) => LexicalCandidateBatch::deferred(results, context),
                    None => LexicalCandidateBatch::eager(results),
                });
            }
            limit = next;
            batch = frankensearch_core::with_cancellation(
                cx,
                "filter_lexical_refill",
                lexical.search_candidates(cx, query, limit),
            )
            .await?;
        }
    }

    /// Keep one retrieval policy and one owner while widening to an eligible
    /// public-ID budget. A short physical prefix may contain duplicates and
    /// cannot establish exhaustion before the retained owner's row bound.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn refill_vectors(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        hits: Vec<VectorHit>,
        target: usize,
        physical_upper_bound: usize,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        source: &'static str,
        retrieve: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    ) -> SearchResult<(Vec<VectorHit>, usize)> {
        if !self.is_active() {
            return refill_vector_after_exclusions(
                cx,
                hits,
                target,
                physical_upper_bound,
                exclusions,
                text_fn,
                source,
                retrieve,
            );
        }
        self.refill_filtered_vectors(
            cx,
            lexical,
            hits,
            target,
            target,
            physical_upper_bound,
            exclusions,
            text_fn,
            source,
            retrieve,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn refill_filtered_vectors(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        mut hits: Vec<VectorHit>,
        target: usize,
        mut limit: usize,
        physical_upper_bound: usize,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        source: &'static str,
        mut retrieve: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    ) -> SearchResult<(Vec<VectorHit>, usize)> {
        loop {
            let fetched = hits.len();
            let ids: Vec<_> = hits.iter().map(|hit| hit.doc_id.as_str()).collect();
            self.hydrate(cx, lexical, &ids).await?;
            let mut seen = AHashSet::new();
            let mut retained = 0;
            for rank in 0..hits.len() {
                if retained == target {
                    break;
                }
                let id = hits[rank].doc_id.as_str();
                if !seen.insert(id.to_owned()) || !self.includes(cx, id)? {
                    continue;
                }
                let excluded = exclusions.map_or(Ok(false), |exclusions| {
                    should_exclude_document(id, exclusions, text_fn, source)
                });
                cancellation_checkpoint(cx, "filter_vector_to_search")?;
                if !excluded? {
                    hits.swap(retained, rank);
                    retained += 1;
                }
            }
            hits.truncate(retained);
            let next = limit
                .saturating_mul(2)
                .max(limit.saturating_add(1))
                .min(physical_upper_bound);
            if retained == target || fetched == 0 || next <= limit {
                return Ok((hits, limit));
            }
            cancellation_checkpoint(cx, "filter_vector_refill")?;
            limit = next;
            hits = retrieve(limit)?;
            cancellation_checkpoint(cx, "filter_vector_refill_complete")?;
        }
    }

    /// Refill both pools until newly discovered metadata no longer revokes a
    /// retained vector. A lexical refill can reject a formerly unknown vector,
    /// and fetching its replacement can in turn revoke a lexical candidate.
    ///
    /// A replaced vector pool contains raw retrieval scores; `true` tells the
    /// caller to apply calibration/corrections once to that replacement. The
    /// callback retains the admitted query, owner and retrieval policy. Reopen
    /// the same physical window before widening: its previously truncated tail
    /// may already contain enough eligible replacements, even at the row bound.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn reconcile_vectors(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        lexical_pool: LexicalRevalidation<'_>,
        hits: &mut Vec<VectorHit>,
        target: usize,
        retrieval_limit: &mut usize,
        physical_upper_bound: usize,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        source: &'static str,
        retrieve: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    ) -> SearchResult<bool> {
        self.reconcile_vectors_with_history(
            cx,
            lexical,
            lexical_pool,
            hits,
            target,
            retrieval_limit,
            physical_upper_bound,
            exclusions,
            text_fn,
            source,
            &mut AHashSet::new(),
            retrieve,
        )
        .await
    }

    /// Share revocation history across alternating semantic arms. A fresh
    /// history on every pass would let an unstable predicate cycle between
    /// fast and quality pools without reaching either arm's local guard.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn reconcile_vectors_with_history(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        lexical_pool: LexicalRevalidation<'_>,
        hits: &mut Vec<VectorHit>,
        target: usize,
        retrieval_limit: &mut usize,
        physical_upper_bound: usize,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
        source: &'static str,
        revoked: &mut AHashSet<DocId>,
        mut retrieve: impl FnMut(usize) -> SearchResult<Vec<VectorHit>>,
    ) -> SearchResult<bool> {
        if !self.is_active() {
            return Ok(false);
        }
        let mut replaced = false;
        loop {
            self.revalidate_lexical(
                cx,
                lexical,
                lexical_pool.query,
                lexical_pool.target,
                lexical_pool.results,
                lexical_pool.context,
                exclusions,
                text_fn,
            )
            .await?;
            let mut changed = false;
            for hit in hits.iter() {
                if !self.includes(cx, &hit.doc_id)? {
                    // Known observations persist between arms. Rejecting the
                    // same refilled ID again therefore means the predicate is
                    // unstable; fail rather than retry forever or lose recall.
                    if !revoked.insert(hit.doc_id.clone()) {
                        return Err(SearchError::InvalidConfig {
                            field: "search.filter".to_owned(),
                            value: "unstable_predicate".to_owned(),
                            reason: "filter eligibility changed repeatedly during candidate refill"
                                .to_owned(),
                        });
                    }
                    changed = true;
                }
            }
            if !changed {
                return Ok(replaced);
            }
            cancellation_checkpoint(cx, "filter_vector_revalidation")?;
            let candidates = retrieve(*retrieval_limit)?;
            cancellation_checkpoint(cx, "filter_vector_revalidation_complete")?;
            let (replacement, limit) = self
                .refill_filtered_vectors(
                    cx,
                    lexical,
                    candidates,
                    target,
                    *retrieval_limit,
                    physical_upper_bound,
                    exclusions,
                    text_fn,
                    source,
                    &mut retrieve,
                )
                .await?;
            *hits = replacement;
            *retrieval_limit = limit;
            replaced = true;
        }
    }

    /// Recheck already retrieved rows after another arm supplied metadata that
    /// was previously unknown. This never invokes a model or changes scores.
    pub(super) fn retain_vectors(&self, cx: &Cx, hits: &mut Vec<VectorHit>) -> SearchResult<()> {
        if !self.is_active() {
            return Ok(());
        }
        let mut retained = 0;
        for rank in 0..hits.len() {
            if self.includes(cx, &hits[rank].doc_id)? {
                hits.swap(retained, rank);
                retained += 1;
            }
        }
        hits.truncate(retained);
        Ok(())
    }

    #[cfg(feature = "graph")]
    pub(super) fn retain_scored(
        &self,
        cx: &Cx,
        results: &mut Vec<ScoredResult>,
    ) -> SearchResult<bool> {
        if !self.is_active() {
            return Ok(false);
        }
        let original_len = results.len();
        let mut retained = 0;
        for rank in 0..results.len() {
            if self.includes(cx, &results[rank].doc_id)? {
                results.swap(retained, rank);
                retained += 1;
            }
        }
        results.truncate(retained);
        self.attach_metadata(results);
        Ok(retained != original_len)
    }

    #[cfg(feature = "graph")]
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn graph_candidates(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        graph: &frankensearch_core::DocumentGraph,
        query: &str,
        seeds: &[VectorHit],
        target: usize,
        exclusions: Option<&NormalizedExclusions>,
        text_fn: &(dyn Fn(&str) -> Option<String> + Send + Sync),
    ) -> SearchResult<Option<Vec<ScoredResult>>> {
        cancellation_checkpoint(cx, "filter_graph_retrieval")?;
        if target == 0 {
            return Ok(None);
        }
        // PageRank already scores and sorts every reachable node before it
        // truncates. Retain one ranking for this seed set while widening the
        // metadata/exclusion prefix; recomputing it for every refill repeats
        // the full graph walk and can change tie scores between prefixes.
        let Some(mut ranked) =
            super::GraphRanker::new().rank_phase1(cx, query, graph, seeds, graph.node_count())?
        else {
            return Ok(None);
        };
        let available = ranked.len();
        let mut limit = target.min(available);
        loop {
            cancellation_checkpoint(cx, "filter_graph_retrieval")?;
            // The final prefix is also the exhaustion boundary, so it can
            // consume the ranking without cloning the complete graph pool.
            let prefix = if limit == available {
                std::mem::take(&mut ranked)
            } else {
                ranked[..limit].to_vec()
            };
            let mut results = self.filter_scored(cx, lexical, prefix).await?;
            if let Some(exclusions) = exclusions {
                results = super::filter_scored_results_by_negations(
                    cx, results, target, exclusions, text_fn, "graph",
                )?;
            }
            let next = limit
                .saturating_mul(2)
                .max(limit.saturating_add(1))
                .min(available);
            if (!self.is_active() && exclusions.is_none())
                || results.len() >= target
                || next <= limit
            {
                results.truncate(target);
                cancellation_checkpoint(cx, "filter_graph_to_fusion")?;
                return Ok(Some(results));
            }
            limit = next;
        }
    }

    #[cfg(feature = "graph")]
    pub(super) async fn filter_scored(
        &mut self,
        cx: &Cx,
        lexical: Option<&dyn LexicalRead>,
        mut results: Vec<ScoredResult>,
    ) -> SearchResult<Vec<ScoredResult>> {
        if !self.is_active() {
            return Ok(results);
        }
        self.observe(&results);
        let ids: Vec<_> = results.iter().map(|hit| hit.doc_id.as_str()).collect();
        self.hydrate(cx, lexical, &ids).await?;
        self.retain_scored(cx, &mut results)?;
        Ok(results)
    }
}
