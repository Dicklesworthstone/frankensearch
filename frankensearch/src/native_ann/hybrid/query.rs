//! Exclusion membership captured from a complete immutable lexical cohort.
//!
//! Only the built/reopened source wrappers and retained generation readers can
//! opt into this preparation. An arbitrary `LexicalRead` may refresh between
//! the exclusion census and candidate collection and cannot supply this proof.

use std::collections::BTreeSet;
use std::sync::Arc;

#[cfg(feature = "quill")]
use frankensearch_core::ParsedQuery;
#[cfg(feature = "quill")]
use frankensearch_core::parsed_query::{normalize_phrase_negation_keywords, quote_query_phrase};

use super::{checkpoint, invalid, validate_lexical};
use crate::{Cx, LexicalCandidateBatch, LexicalRead, SearchResult};

pub(super) struct ImmutableTextQuery {
    pub(super) semantic: String,
    pub(super) lexical: String,
    exclusion_query: String,
    pub(super) excluded: Arc<BTreeSet<String>>,
    ceiling: usize,
}

impl ImmutableTextQuery {
    #[cfg(feature = "quill")]
    pub(super) fn new(lexical: &dyn LexicalRead, text: &str) -> Option<Self> {
        let parsed = ParsedQuery::parse(text);
        if !parsed.has_negations() {
            return None;
        }
        // Match the lexical engine's own analyzer, fields, and phrase rules.
        // Decode and requote phrases so escapes cannot become query syntax;
        // prepare bare terms exactly as the ordinary user-query boundary does.
        let exclusion_query = parsed
            .negative_terms
            .iter()
            .map(|term| lexical.query_text(term).into_owned())
            .chain(
                parsed
                    .negative_phrases
                    .iter()
                    .map(|phrase| quote_query_phrase(phrase)),
            )
            .collect::<Vec<_>>()
            .join(" OR ");
        Some(Self {
            semantic: parsed.positive,
            lexical: normalize_phrase_negation_keywords(text).into_owned(),
            exclusion_query,
            excluded: Arc::default(),
            ceiling: 0,
        })
    }

    pub(super) async fn prepare(&mut self, cx: &Cx, lexical: &dyn LexicalRead) -> SearchResult<()> {
        checkpoint(cx, "native_ann.exclusions.start")?;
        let ceiling = lexical.doc_count()?;
        self.ceiling = ceiling;
        if ceiling == 0 {
            return Ok(());
        }
        // This is a complete membership census, not a top-k sample. Existing
        // source scopes widen their immutable reader before returning this
        // many eligible candidates, so explicit filters remain conjunctive.
        let batch = frankensearch_core::with_cancellation(
            cx,
            "native_ann.exclusions.membership",
            lexical.search_candidates(cx, &self.exclusion_query, ceiling),
        )
        .await?;
        checkpoint(cx, "native_ann.exclusions.response")?;
        validate_lexical(cx, &batch)?;
        if batch.results().len() > ceiling {
            return Err(invalid(
                "query.exclusions",
                "overfull",
                "exclusion membership exceeds the retained lexical document count",
            ));
        }
        let mut excluded = BTreeSet::new();
        for result in batch.results() {
            checkpoint(cx, "native_ann.exclusions.document")?;
            if !excluded.insert(result.doc_id.to_string()) {
                return Err(invalid(
                    "query.exclusions",
                    "duplicate-id",
                    "exclusion membership must contain unique document IDs",
                ));
            }
        }
        checkpoint(cx, "native_ann.exclusions.complete")?;
        self.excluded = Arc::new(excluded);
        Ok(())
    }

    pub(super) async fn candidates(
        &self,
        cx: &Cx,
        lexical: &dyn LexicalRead,
        budget: usize,
    ) -> SearchResult<LexicalCandidateBatch> {
        let target = budget.min(self.ceiling);
        if target == 0 {
            checkpoint(cx, "native_ann.exclusions.lexical_empty")?;
            return Ok(LexicalCandidateBatch::eager(Vec::new()));
        }
        let mut width = target;
        loop {
            let batch =
                super::refinement::checked_lexical(cx, lexical, &self.lexical, width).await?;
            let count = batch.results().len();
            if count > width {
                return Err(invalid(
                    "query.exclusions",
                    "overfull-candidates",
                    "lexical candidates exceed their requested window",
                ));
            }
            let (results, context) = batch.into_parts();
            let mut eligible = Vec::new();
            for result in results {
                checkpoint(cx, "native_ann.exclusions.lexical_candidate")?;
                if !self.excluded.contains(result.doc_id.as_str()) && eligible.len() < target {
                    eligible.push(result);
                }
            }
            // Core exclusions are global constraints. A Boolean lexical query
            // may still rank an excluded document in a different OR branch;
            // keep that query's order, then widen until its eligible pool is
            // full. Every replacement batch belongs to this immutable reader.
            if eligible.len() == target || count < width || width == self.ceiling {
                return Ok(match context {
                    Some(context) => LexicalCandidateBatch::deferred(eligible, context),
                    None => LexicalCandidateBatch::eager(eligible),
                });
            }
            width = width.saturating_mul(2).min(self.ceiling);
        }
    }
}
