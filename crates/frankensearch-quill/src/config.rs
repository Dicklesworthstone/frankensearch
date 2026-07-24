//! Explicit Quill configuration.
//!
//! Quill never reads environment variables. Applications may populate this
//! structure from their own configuration layers, but the engine sees only
//! explicit values.

use frankensearch_core::{SearchError, SearchResult};

/// Default per-shard Scribe arena budget (64 MiB).
pub const DEFAULT_SCRIBE_SHARD_BUDGET_BYTES: usize = 64 * 1024 * 1024;
/// Default searchable delta budget per shard (8 MiB).
pub const DEFAULT_DELTA_BUDGET_BYTES: usize = 8 * 1024 * 1024;
/// Default number of same-tier segments that triggers a merge.
pub const DEFAULT_TIER_FANOUT: usize = 8;
/// Default inclusive upper bound for the small segment tier: one Q1 lease.
pub const DEFAULT_TIER_SMALL_MAX_DOCID_WIDTH: u64 = 1 << 16;
/// Default inclusive upper bound for the medium segment tier: eight Q1 leases.
pub const DEFAULT_TIER_MEDIUM_MAX_DOCID_WIDTH: u64 = 8 << 16;
/// Default number of sealed mini-segments between bulk-build publications.
pub const DEFAULT_BULK_PUBLISH_SEGMENT_CADENCE: usize = 64;
/// Default live-segment tombstone density that triggers compaction.
pub const DEFAULT_COMPACTION_TOMBSTONE_DENSITY: f64 = 0.20;
/// Default maximum hole ratio accepted by the concat-merge policy.
pub const DEFAULT_MERGE_MAX_HOLE_RATIO: f64 = 0.50;
/// Default maximum number of terms produced by one glob expansion.
pub const DEFAULT_GLOB_EXPANSION_LIMIT: usize = 16_384;
/// Default deterministic query-work budget.
///
/// One unit is charged per segment transition, dictionary block, posting
/// block, or phrase candidate whose positions are verified. Ten million keeps
/// ordinary and fixture-corpus queries on the zero-contention fast path while
/// bounding adversarial glob and phrase tails.
pub const DEFAULT_QUERY_FUEL_BUDGET: u64 = 10_000_000;
/// Default cap on independent ingest shards.
pub const DEFAULT_MAX_INGEST_SHARDS: usize = 32;
/// Default cross-process visibility bound: one second between the first
/// unpublished change and its durable publication (visibility contract, bead
/// bd-quill-duel-visibility-contract-9rk3).
pub const DEFAULT_MAX_VISIBILITY_LAG_MS: u64 = 1_000;

/// All engine-level Quill tuning knobs.
///
/// Defaults are deterministic constants, not environment-dependent values.
#[derive(Debug, Clone, PartialEq)]
pub struct QuillConfig {
    /// Maximum bytes in one Scribe shard's accumulation arenas before flush.
    pub scribe_shard_budget_bytes: usize,
    /// Maximum bytes in one shard's searchable delta before seal.
    pub delta_budget_bytes: usize,
    /// Number of segments at one tier that triggers promotion/merge.
    pub tier_fanout: usize,
    /// Inclusive maximum docid-range width classified as a small segment.
    pub tier_small_max_docid_width: u64,
    /// Inclusive maximum docid-range width classified as a medium segment.
    pub tier_medium_max_docid_width: u64,
    /// Suppress ordinary tier merges until [`crate::QuillIndex::finish_bulk_load`].
    pub bulk_load_mode: bool,
    /// Sealed mini-segments between crash-resumable bulk MANIFEST publishes.
    pub bulk_publish_segment_cadence: usize,
    /// Per-segment tombstone density that triggers compaction.
    pub compaction_tombstone_density: f64,
    /// Maximum fraction of holes tolerated by concat-merge policy.
    pub merge_max_hole_ratio: f64,
    /// Maximum terms a glob may expand into before returning a typed error.
    pub glob_expansion_limit: usize,
    /// Deterministic coarse work-unit ceiling for one query.
    pub query_fuel_budget: u64,
    /// Upper bound on independent ingest shards.
    pub max_ingest_shards: usize,
    /// Force one deterministic shard for replay and conformance runs.
    pub deterministic_ingest: bool,
    /// Cross-process visibility bound: once unpublished changes are this old,
    /// the writer must run a seal-and-publish barrier instead of waiting for
    /// the ordinary cadence (visibility contract, `max_visibility_lag_ms`).
    pub max_visibility_lag_ms: u64,
    /// Permit a durability-enabled writer to quarantine an unrepairable
    /// segment and publish a degraded successor that omits it.
    ///
    /// This is deliberately off for library embedders. Applications that turn
    /// it on must couple the resulting degraded snapshot to a freshness audit
    /// and backfill path.
    pub quarantine_on_unrepairable: bool,
}

impl Default for QuillConfig {
    fn default() -> Self {
        Self {
            scribe_shard_budget_bytes: DEFAULT_SCRIBE_SHARD_BUDGET_BYTES,
            delta_budget_bytes: DEFAULT_DELTA_BUDGET_BYTES,
            tier_fanout: DEFAULT_TIER_FANOUT,
            tier_small_max_docid_width: DEFAULT_TIER_SMALL_MAX_DOCID_WIDTH,
            tier_medium_max_docid_width: DEFAULT_TIER_MEDIUM_MAX_DOCID_WIDTH,
            bulk_load_mode: false,
            bulk_publish_segment_cadence: DEFAULT_BULK_PUBLISH_SEGMENT_CADENCE,
            compaction_tombstone_density: DEFAULT_COMPACTION_TOMBSTONE_DENSITY,
            merge_max_hole_ratio: DEFAULT_MERGE_MAX_HOLE_RATIO,
            glob_expansion_limit: DEFAULT_GLOB_EXPANSION_LIMIT,
            query_fuel_budget: DEFAULT_QUERY_FUEL_BUDGET,
            max_ingest_shards: DEFAULT_MAX_INGEST_SHARDS,
            deterministic_ingest: false,
            max_visibility_lag_ms: DEFAULT_MAX_VISIBILITY_LAG_MS,
            quarantine_on_unrepairable: false,
        }
    }
}

impl QuillConfig {
    /// Validate every knob before the engine allocates resources.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for zero budgets/counts or for
    /// non-finite/out-of-range density values.
    pub fn validate(&self) -> SearchResult<()> {
        require_positive("scribe_shard_budget_bytes", self.scribe_shard_budget_bytes)?;
        require_positive("delta_budget_bytes", self.delta_budget_bytes)?;
        if self.tier_fanout < 2 {
            return Err(invalid_config(
                "tier_fanout",
                &self.tier_fanout,
                "must be at least 2",
            ));
        }
        if self.tier_small_max_docid_width == 0 {
            return Err(invalid_config(
                "tier_small_max_docid_width",
                &self.tier_small_max_docid_width,
                "must be greater than zero",
            ));
        }
        if self.tier_medium_max_docid_width <= self.tier_small_max_docid_width {
            return Err(invalid_config(
                "tier_medium_max_docid_width",
                &self.tier_medium_max_docid_width,
                "must be greater than tier_small_max_docid_width",
            ));
        }
        require_positive(
            "bulk_publish_segment_cadence",
            self.bulk_publish_segment_cadence,
        )?;
        require_fraction_open_closed(
            "compaction_tombstone_density",
            self.compaction_tombstone_density,
        )?;
        require_fraction_closed("merge_max_hole_ratio", self.merge_max_hole_ratio)?;
        require_positive("glob_expansion_limit", self.glob_expansion_limit)?;
        if self.query_fuel_budget == 0 {
            return Err(invalid_config(
                "query_fuel_budget",
                &self.query_fuel_budget,
                "must be greater than zero",
            ));
        }
        require_positive("max_ingest_shards", self.max_ingest_shards)?;
        if self.max_visibility_lag_ms == 0 {
            return Err(invalid_config(
                "max_visibility_lag_ms",
                &self.max_visibility_lag_ms,
                "must be positive so the cross-process freshness bound is a guarantee, not a suggestion",
            ));
        }
        Ok(())
    }

    /// Resolve an externally detected parallelism count to Quill's shard count.
    ///
    /// The caller chooses how parallelism is detected. Deterministic mode always
    /// returns one; otherwise the count is clamped to `1..=max_ingest_shards`.
    #[must_use]
    pub fn resolved_ingest_shards(&self, detected_parallelism: usize) -> usize {
        if self.deterministic_ingest {
            1
        } else {
            detected_parallelism
                .max(1)
                .min(self.max_ingest_shards.max(1))
        }
    }
}

fn invalid_config(field: &str, value: &impl ToString, reason: &str) -> SearchError {
    SearchError::InvalidConfig {
        field: field.to_owned(),
        value: value.to_string(),
        reason: reason.to_owned(),
    }
}

fn require_positive(field: &str, value: usize) -> SearchResult<()> {
    if value == 0 {
        Err(invalid_config(field, &value, "must be greater than zero"))
    } else {
        Ok(())
    }
}

fn require_fraction_open_closed(field: &str, value: f64) -> SearchResult<()> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(invalid_config(
            field,
            &value,
            "must be finite and in (0, 1]",
        ))
    }
}

fn require_fraction_closed(field: &str, value: f64) -> SearchResult<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid_config(
            field,
            &value,
            "must be finite and in [0, 1]",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_pinned() {
        assert_eq!(
            QuillConfig::default(),
            QuillConfig {
                scribe_shard_budget_bytes: 64 * 1024 * 1024,
                delta_budget_bytes: 8 * 1024 * 1024,
                tier_fanout: 8,
                tier_small_max_docid_width: 65_536,
                tier_medium_max_docid_width: 524_288,
                bulk_load_mode: false,
                bulk_publish_segment_cadence: 64,
                compaction_tombstone_density: 0.20,
                merge_max_hole_ratio: 0.50,
                glob_expansion_limit: 16_384,
                query_fuel_budget: 10_000_000,
                max_ingest_shards: 32,
                deterministic_ingest: false,
                max_visibility_lag_ms: 1_000,
                quarantine_on_unrepairable: false,
            }
        );
        assert!(QuillConfig::default().validate().is_ok());
    }

    #[test]
    fn shard_resolution_clamps_and_honors_determinism() {
        let mut config = QuillConfig::default();
        assert_eq!(config.resolved_ingest_shards(0), 1);
        assert_eq!(config.resolved_ingest_shards(12), 12);
        assert_eq!(config.resolved_ingest_shards(128), 32);
        config.deterministic_ingest = true;
        assert_eq!(config.resolved_ingest_shards(128), 1);
    }

    #[test]
    fn validation_rejects_boundaries_and_non_finite_values() {
        let mut config = QuillConfig {
            delta_budget_bytes: 0,
            ..QuillConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "delta_budget_bytes"
        ));

        config = QuillConfig::default();
        config.compaction_tombstone_density = f64::NAN;
        assert!(config.validate().is_err());
        config.compaction_tombstone_density = 0.0;
        assert!(config.validate().is_err());
        config.compaction_tombstone_density = 1.0;
        assert!(config.validate().is_ok());

        config.merge_max_hole_ratio = 0.0;
        assert!(config.validate().is_ok());
        config.merge_max_hole_ratio = 1.01;
        assert!(config.validate().is_err());

        config = QuillConfig::default();
        config.query_fuel_budget = 0;
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "query_fuel_budget"
        ));

        config = QuillConfig::default();
        config.max_visibility_lag_ms = 0;
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "max_visibility_lag_ms"
        ));
        config.max_visibility_lag_ms = 1;
        assert!(config.validate().is_ok());

        config.tier_small_max_docid_width = 0;
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "tier_small_max_docid_width"
        ));
        config = QuillConfig::default();
        config.tier_medium_max_docid_width = config.tier_small_max_docid_width;
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "tier_medium_max_docid_width"
        ));
        config = QuillConfig::default();
        config.bulk_publish_segment_cadence = 0;
        assert!(matches!(
            config.validate(),
            Err(SearchError::InvalidConfig { field, .. }) if field == "bulk_publish_segment_cadence"
        ));
    }
}
