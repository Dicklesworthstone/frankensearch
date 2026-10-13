//! Explicit full-mode CLI policy, applied to the original retained query.
//! Parsing is bounded and precedes model loading. No request can change this
//! startup policy, a source scope, model identity or the session's time budget.

use frankensearch_fusion::RrfConfig;
use serde::{Deserialize, Serialize};

use super::{Cx, Mode, Policy, Result, bad, cohort};

const MAX_POLICY_BYTES: usize = 8 * 1024;
const MAX_MULTIPLIER: usize = 100;
const MAX_BEAM: usize = 1_000_000;

/// Settings for full-mode queries only. Omitted fields retain native defaults.
/// Private fields ensure production callers use validated parsing. Serde is
/// used only for command input/output, never generation identity or disk state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FullQuery {
    candidate_multiplier: usize,
    fast_ef: Option<usize>,
    quality_ef: Option<usize>,
    rrf_k: f64,
    lexical_weight: f64,
    semantic_weight: f64,
    quality_weight: f32,
}

impl Default for FullQuery {
    fn default() -> Self {
        let rrf = RrfConfig::default();
        Self {
            candidate_multiplier: 3,
            fast_ef: None,
            quality_ef: None,
            rrf_k: rrf.k,
            lexical_weight: rrf.lexical_weight,
            semantic_weight: rrf.semantic_weight,
            quality_weight: 0.7,
        }
    }
}

impl FullQuery {
    pub(crate) fn parse(text: &str) -> Result<Self> {
        if text.len() > MAX_POLICY_BYTES {
            return Err(bad("full-query policy exceeds 8 KiB"));
        }
        let policy: Self = serde_json::from_str(text)
            .map_err(|_| bad("invalid full-query policy JSON or unknown/duplicate field"))?;
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<()> {
        if !(1..=MAX_MULTIPLIER).contains(&self.candidate_multiplier) {
            return Err(bad("candidate_multiplier must be between 1 and 100"));
        }
        if [self.fast_ef, self.quality_ef]
            .into_iter()
            .flatten()
            .any(|beam| !(1..=MAX_BEAM).contains(&beam))
        {
            return Err(bad("graph beams must be null or between 1 and 1000000"));
        }
        if !self.quality_weight.is_finite() || !(0.0..=1.0).contains(&self.quality_weight) {
            return Err(bad("quality_weight must be finite and between zero and one"));
        }
        if !self.rrf_k.is_finite()
            || self.rrf_k < 0.0
            || !self.lexical_weight.is_finite()
            || self.lexical_weight <= 0.0
            || !self.semantic_weight.is_finite()
            || self.semantic_weight <= 0.0
        {
            return Err(bad(
                "RRF K must be finite and nonnegative; source weights finite and positive",
            ));
        }
        let first_rank = 1.0 / (self.rrf_k + 1.0);
        let maximum = first_rank * self.lexical_weight + first_rank * self.semantic_weight;
        if !maximum.is_finite() || maximum > f64::from(f32::MAX) {
            return Err(bad("full-query maximum fused score exceeds f32 precision"));
        }
        Ok(())
    }

    fn apply<'a>(&self, query: cohort::Progressive<'a>) -> Result<cohort::Progressive<'a>> {
        // Recheck at use as well: test/deserialized construction must not bypass
        // CLI bounds. Native setters also enforce their own state/cancel gates.
        self.validate()?;
        let rrf = RrfConfig {
            k: self.rrf_k,
            lexical_weight: self.lexical_weight,
            semantic_weight: self.semantic_weight,
            ..RrfConfig::default()
        };
        Ok(match query {
            cohort::Progressive::Single(query) => cohort::Progressive::Single(Box::new(
                (*query)
                    .with_fusion(rrf, self.quality_weight)?
                    .with_candidate_multiplier(self.candidate_multiplier)?
                    .with_beam_widths(self.fast_ef, self.quality_ef)?,
            )),
            cohort::Progressive::Sharded(query) => cohort::Progressive::Sharded(Box::new(
                (*query)
                    .with_fusion(rrf, self.quality_weight)?
                    .with_candidate_multiplier(self.candidate_multiplier)?
                    .with_beam_widths(self.fast_ef, self.quality_ef)?,
            )),
        })
    }
}

impl Policy {
    /// Prepare the same full-mode query for buffered and streaming execution.
    /// Attach retained-source reranking first so candidate tuning accounts for
    /// its larger window. The complete scope and all original owners survive.
    pub(crate) fn progressive<'a>(
        &'a self,
        scoped: &'a cohort::Prepared<'_>,
        cx: &'a Cx,
        text: &'a str,
        limit: usize,
    ) -> Result<cohort::Progressive<'a>> {
        let query = match &self.rerank {
            Some(rerank) => scoped.progressive_with_reranker(
                cx,
                text,
                limit,
                rerank.model.as_ref(),
                rerank.window,
            )?,
            None => scoped.progressive(cx, text, limit)?,
        };
        match &self.full_query {
            Some(policy) => policy.apply(query),
            None => Ok(query),
        }
    }

    pub(crate) fn annotate_full_query(&self, mode: Mode, payload: &mut serde_json::Value) {
        if mode == Mode::Full
            && let Some(policy) = &self.full_query
        {
            // These are requested settings, not measured work/recall or model
            // provenance. Native retrieval retains its beam floors and caps.
            payload["full_query"] = serde_json::json!(policy);
        }
    }
}
