//! Frozen per-query policy for global sharded retrieval, never per-shard fusion.

use frankensearch_fusion::RrfConfig;

use super::{NativeShardedProgressiveSearch, State, checkpoint, invalid};
use crate::SearchResult;

pub(super) struct QueryPolicy {
    pub(super) candidate_multiplier: usize,
    pub(super) rrf: RrfConfig,
    pub(super) quality_weight: f32,
    pub(super) fast_ef: Option<usize>,
    pub(super) quality_ef: Option<usize>,
}

impl Default for QueryPolicy {
    fn default() -> Self {
        Self {
            candidate_multiplier: 3,
            rrf: RrfConfig::default(),
            quality_weight: 0.7,
            fast_ef: None,
            quality_ef: None,
        }
    }
}

pub(super) struct ShardFusion<'a> {
    pub(super) k: usize,
    pub(super) is_hash: bool,
    pub(super) quality_weight: Option<f32>,
    pub(super) rrf: &'a RrfConfig,
}

impl QueryPolicy {
    pub(super) fn budget(&self, window: usize) -> SearchResult<usize> {
        window.checked_mul(self.candidate_multiplier).ok_or_else(|| {
            invalid(
                "shards.candidate_budget",
                "overflow",
                "the configured candidate budget must fit usize",
            )
        })
    }

    pub(super) fn fusion(&self, k: usize, is_hash: bool, refined: bool) -> ShardFusion<'_> {
        ShardFusion {
            k,
            is_hash,
            quality_weight: refined.then_some(self.quality_weight),
            rrf: &self.rrf,
        }
    }
}

impl NativeShardedProgressiveSearch<'_> {
    /// Set the global fusion policy before requesting any phase.
    ///
    /// Uses the same RRF and union-preserving blend as single-owner native
    /// retrieval. Defaults remain ordinary RRF and 0.7 quality weight. Fusion
    /// happens ONCE after each tier's shard candidates have been merged, not
    /// independently per shard. This query owns its policy through hydration
    /// and reranking; no model, source snapshot or row coordinate is replaced.
    ///
    /// The blend weight affects shared candidates; one-sided candidates retain
    /// their normalized score even at weights 0 and 1. Every configured tier
    /// remains required regardless of its weight. Hash-control lanes retain
    /// nonsemantic provenance and never acquire fabricated semantic scores.
    ///
    /// # Errors
    /// Refuses cancellation, started queries, invalid/nonfinite weights or K,
    /// and a maximum fused score that cannot fit the public f32 result.
    pub fn with_fusion(mut self, rrf: RrfConfig, quality_weight: f32) -> SearchResult<Self> {
        self.configure_policy()?;
        if !quality_weight.is_finite() || !(0.0..=1.0).contains(&quality_weight) {
            return Err(invalid(
                "shards.quality_weight",
                "out-of-range",
                "quality blend weight must be finite and between zero and one",
            ));
        }
        if !rrf.k.is_finite()
            || rrf.k < 0.0
            || !rrf.lexical_weight.is_finite()
            || rrf.lexical_weight <= 0.0
            || !rrf.semantic_weight.is_finite()
            || rrf.semantic_weight <= 0.0
        {
            return Err(invalid(
                "shards.rrf",
                "invalid",
                "RRF K must be finite and nonnegative, and source weights finite and positive",
            ));
        }
        let first_rank = 1.0 / (rrf.k + 1.0);
        let maximum_score = first_rank * rrf.lexical_weight + first_rank * rrf.semantic_weight;
        if !maximum_score.is_finite() || maximum_score > f64::from(f32::MAX) {
            return Err(invalid(
                "shards.rrf",
                "score-overflow",
                "the maximum fused score must fit the public f32 result",
            ));
        }
        self.policy.rrf = rrf;
        self.policy.quality_weight = quality_weight;
        Ok(self)
    }

    /// Request `multiplier * max(k, rerank_window)` GLOBAL candidates per source.
    ///
    /// The default is three. Each vector tier merges its complete partition
    /// inventory to this bound before fusion; adding shards does not multiply
    /// the final pool size. The one global lexical population uses the same
    /// requested limit. Changing the rerank window before or after this setting
    /// yields the same budget. Pool size may change normalization and ranking.
    ///
    /// Scoped queries keep their original eligible-membership/widening policy.
    /// This limits requested hits, not graph visits, source size or peak memory.
    /// It never permits omitting a failing shard or required provider.
    ///
    /// # Errors
    /// Refuses zero, arithmetic overflow, cancellation or a started query.
    pub fn with_candidate_multiplier(mut self, multiplier: usize) -> SearchResult<Self> {
        self.configure_policy()?;
        if multiplier == 0 {
            return Err(invalid(
                "shards.candidate_multiplier",
                "zero",
                "candidate multiplier must be positive",
            ));
        }
        let budget = self.result_window().checked_mul(multiplier).ok_or_else(|| {
            invalid(
                "shards.candidate_budget",
                "overflow",
                "the configured candidate budget must fit usize",
            )
        })?;
        self.policy.candidate_multiplier = multiplier;
        self.budget = budget;
        Ok(self)
    }

    /// Override native graph beams independently for fast and quality tiers.
    ///
    /// The chosen beam is passed to every shard of its tier, including scoped
    /// retrieval. None keeps each owner's admitted default. Existing retrieval
    /// raises a beam to the requested-hit floor, caps it at physical row count
    /// and widens on filtered underfill. Exact shards ignore the beam as before.
    /// No graph is rebuilt and no approximate-recall guarantee is implied.
    ///
    /// # Errors
    /// Refuses zero beams, cancellation or configuration after any phase starts.
    pub fn with_beam_widths(
        mut self,
        fast: Option<usize>,
        quality: Option<usize>,
    ) -> SearchResult<Self> {
        self.configure_policy()?;
        if fast == Some(0) || quality == Some(0) {
            return Err(invalid(
                "shards.beam",
                "zero",
                "an explicit graph beam must be positive",
            ));
        }
        self.policy.fast_ef = fast;
        self.policy.quality_ef = quality;
        Ok(self)
    }

    fn configure_policy(&self) -> SearchResult<()> {
        checkpoint(self.cx, "native_ann.shards.configuration")?;
        if !matches!(self.state, State::Initial) {
            return Err(invalid(
                "shards.configuration",
                "started",
                "query policy must be fixed before requesting its first phase",
            ));
        }
        Ok(())
    }
}

// Execution placement is independent of ranking and required-tier policy.
mod execution;
