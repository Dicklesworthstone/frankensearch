//! Time-travel query API for Native Mode distributed search.
//!
//! Allows querying historical search generations by commit sequence number.
//! Each retained generation covers a contiguous commit range; the
//! [`GenerationHistory`] resolves an `as_of_commit_seq` to the appropriate
//! generation snapshot, ensuring stable per-request reads.
//!
//! Integrates with [`GenerationController`](crate::activation::GenerationController):
//! when a new generation is activated, the previous one is retained in history
//! (subject to the configured [`RetentionPolicy`]).

use std::sync::Arc;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::activation::ActiveGeneration;

// ---------------------------------------------------------------------------
// Retention policy
// ---------------------------------------------------------------------------

/// Controls how many and how long historical generations are retained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Maximum number of generations to keep in history.
    /// Oldest generations are pruned first when this limit is exceeded.
    pub max_retained: usize,
    /// Maximum age (millis) for retained generations.
    /// Generations older than this are pruned regardless of count.
    /// Set to `0` to disable age-based pruning.
    pub max_age_ms: u64,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_retained: 10,
            max_age_ms: 3_600_000, // 1 hour
        }
    }
}

// ---------------------------------------------------------------------------
// Retained generation
// ---------------------------------------------------------------------------

/// A generation kept in history for time-travel queries.
#[derive(Debug, Clone)]
pub struct RetainedGeneration {
    /// The generation snapshot.
    pub generation: Arc<ActiveGeneration>,
    /// Unix timestamp (millis) when this generation was retained.
    pub retained_at: u64,
}

// ---------------------------------------------------------------------------
// Query result
// ---------------------------------------------------------------------------

/// Result of a time-travel generation resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeTravelResult {
    /// Found a generation covering the requested commit sequence.
    Found {
        /// The generation identifier.
        generation_id: String,
        /// Commit range covered.
        commit_low: u64,
        /// Commit range high bound.
        commit_high: u64,
    },
    /// No retained generation covers the requested commit sequence.
    NotFound {
        /// The requested commit sequence.
        requested_seq: u64,
        /// Closest available commit range (if any).
        closest_range: Option<(u64, u64)>,
    },
    /// The requested commit sequence is covered by the currently active generation.
    ActiveGeneration {
        /// The active generation identifier.
        generation_id: String,
    },
}

// ---------------------------------------------------------------------------
// Generation history
// ---------------------------------------------------------------------------

/// Manages retained generations for time-travel queries.
///
/// Thread-safe: reads and writes are synchronized via `std::sync::RwLock`
/// (not asupersync) for synchronous access on the query hot path.
pub struct GenerationHistory {
    /// Retained generations ordered by commit range high bound (ascending).
    retained: RwLock<Vec<RetainedGeneration>>,
    /// Retention policy governing pruning.
    policy: RetentionPolicy,
}

impl GenerationHistory {
    /// Create a new history with the given retention policy.
    #[must_use]
    pub const fn new(policy: RetentionPolicy) -> Self {
        Self {
            retained: RwLock::new(Vec::new()),
            policy,
        }
    }

    /// Add a generation to the retention history.
    ///
    /// Inserts in commit-range order (by `commit_range.high`) and prunes
    /// excess generations per the retention policy.
    pub fn retain(&self, generation: Arc<ActiveGeneration>, now_millis: u64) {
        let mut retained = self
            .retained
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let entry = RetainedGeneration {
            generation,
            retained_at: now_millis,
        };

        // Insert in sorted order by commit_range.high.
        let high = entry.generation.manifest.commit_range.high;
        let pos = retained
            .binary_search_by_key(&high, |r| r.generation.manifest.commit_range.high)
            .unwrap_or_else(|p| p);
        retained.insert(pos, entry);

        // Prune by policy.
        self.prune_locked(&mut retained, now_millis);
        drop(retained);
    }

    /// Resolve the generation that covers `as_of_commit_seq`.
    ///
    /// Returns the retained generation whose commit range includes the
    /// requested sequence number (`low <= as_of_commit_seq <= high`).
    #[must_use]
    pub fn resolve(&self, as_of_commit_seq: u64) -> Option<Arc<ActiveGeneration>> {
        let retained = self
            .retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        retained
            .iter()
            .find(|r| {
                let range = &r.generation.manifest.commit_range;
                range.low <= as_of_commit_seq && as_of_commit_seq <= range.high
            })
            .map(|r| Arc::clone(&r.generation))
    }

    /// Resolve with detailed result information.
    #[must_use]
    pub fn resolve_detailed(&self, as_of_commit_seq: u64) -> TimeTravelResult {
        let retained = self
            .retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Look for exact match.
        for r in retained.iter() {
            let range = &r.generation.manifest.commit_range;
            if range.low <= as_of_commit_seq && as_of_commit_seq <= range.high {
                return TimeTravelResult::Found {
                    generation_id: r.generation.manifest.generation_id.clone(),
                    commit_low: range.low,
                    commit_high: range.high,
                };
            }
        }

        // Find closest range for the error message.
        let closest_range = retained
            .iter()
            .min_by_key(|r| {
                let range = &r.generation.manifest.commit_range;
                let below = as_of_commit_seq.saturating_sub(range.high);
                let above = range.low.saturating_sub(as_of_commit_seq);
                below.max(above)
            })
            .map(|r| {
                let range = &r.generation.manifest.commit_range;
                (range.low, range.high)
            });
        drop(retained);

        TimeTravelResult::NotFound {
            requested_seq: as_of_commit_seq,
            closest_range,
        }
    }

    /// Explicitly prune expired generations per the retention policy.
    ///
    /// Returns the number of generations pruned.
    pub fn prune(&self, now_millis: u64) -> usize {
        let mut retained = self
            .retained
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = retained.len();
        self.prune_locked(&mut retained, now_millis);
        before - retained.len()
    }

    /// Number of retained generations.
    #[must_use]
    pub fn retained_count(&self) -> usize {
        self.retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Snapshot of all retained generation identifiers and their commit ranges.
    #[must_use]
    pub fn retained_summary(&self) -> Vec<(String, u64, u64)> {
        self.retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|r| {
                let m = &r.generation.manifest;
                (
                    m.generation_id.clone(),
                    m.commit_range.low,
                    m.commit_range.high,
                )
            })
            .collect()
    }

    /// Clear all retained generations.
    pub fn clear(&self) {
        let mut retained = self
            .retained
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        retained.clear();
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Prune retained generations per policy (must hold write lock).
    fn prune_locked(&self, retained: &mut Vec<RetainedGeneration>, now_millis: u64) {
        // Age-based pruning.
        if self.policy.max_age_ms > 0 {
            retained.retain(|r| {
                let age = now_millis.saturating_sub(r.retained_at);
                age <= self.policy.max_age_ms
            });
        }

        // Count-based pruning (remove oldest first).
        while retained.len() > self.policy.max_retained {
            retained.remove(0);
        }
    }
}

impl Default for GenerationHistory {
    fn default() -> Self {
        Self::new(RetentionPolicy::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::*;
    use std::collections::BTreeMap;

    fn make_generation(gen_id: &str, low: u64, high: u64) -> Arc<ActiveGeneration> {
        let mut embedders = BTreeMap::new();
        embedders.insert(
            "fast".into(),
            EmbedderRevision {
                model_name: "potion-128M".into(),
                weights_hash: "abcdef1234567890".into(),
                dimension: 256,
                quantization: QuantizationFormat::F16,
            },
        );
        let mut manifest = GenerationManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generation_id: gen_id.into(),
            manifest_hash: String::new(),
            commit_range: CommitRange { low, high },
            build_started_at: 1_700_000_000_000,
            build_completed_at: 1_700_000_060_000,
            embedders,
            vector_artifacts: vec![],
            lexical_artifacts: vec![],
            repair_descriptors: vec![],
            activation_invariants: vec![],
            total_documents: high - low + 1,
            metadata: BTreeMap::new(),
        };
        manifest.manifest_hash = compute_manifest_hash(&manifest).expect("hash");
        Arc::new(ActiveGeneration {
            manifest,
            activation_seq: low,
            activated_at: 1_700_000_000_000 + low * 1000,
            vector_paths: BTreeMap::new(),
            lexical_paths: BTreeMap::new(),
        })
    }

    #[test]
    fn empty_history() {
        let history = GenerationHistory::default();
        assert_eq!(history.retained_count(), 0);
        assert!(history.resolve(50).is_none());
    }

    #[test]
    fn retain_and_resolve() {
        let history = GenerationHistory::default();
        let generation = make_generation("gen-1", 1, 100);
        history.retain(generation, 1000);

        assert_eq!(history.retained_count(), 1);

        // Within range.
        let resolved = history.resolve(50).unwrap();
        assert_eq!(resolved.manifest.generation_id, "gen-1");

        // At boundaries.
        assert!(history.resolve(1).is_some());
        assert!(history.resolve(100).is_some());

        // Outside range.
        assert!(history.resolve(0).is_none());
        assert!(history.resolve(101).is_none());
    }

    #[test]
    fn multiple_generations_resolve_correctly() {
        let history = GenerationHistory::default();

        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);
        history.retain(make_generation("gen-3", 201, 300), 3000);

        assert_eq!(history.retained_count(), 3);

        let r1 = history.resolve(50).unwrap();
        assert_eq!(r1.manifest.generation_id, "gen-1");

        let r2 = history.resolve(150).unwrap();
        assert_eq!(r2.manifest.generation_id, "gen-2");

        let r3 = history.resolve(250).unwrap();
        assert_eq!(r3.manifest.generation_id, "gen-3");

        // Gap — no generation covers seq 0.
        assert!(history.resolve(0).is_none());
    }

    #[test]
    fn resolve_detailed_found() {
        let history = GenerationHistory::default();
        history.retain(make_generation("gen-1", 1, 100), 1000);

        let result = history.resolve_detailed(50);
        assert!(matches!(
            result,
            TimeTravelResult::Found {
                generation_id,
                commit_low: 1,
                commit_high: 100,
            } if generation_id == "gen-1"
        ));
    }

    #[test]
    fn resolve_detailed_not_found_with_closest() {
        let history = GenerationHistory::default();
        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 201, 300), 2000);

        // Seq 150 is between the two generations.
        let result = history.resolve_detailed(150);
        assert!(matches!(result, TimeTravelResult::NotFound { .. }));
        if let TimeTravelResult::NotFound {
            requested_seq,
            closest_range,
        } = result
        {
            assert_eq!(requested_seq, 150);
            assert!(closest_range.is_some());
        }
    }

    #[test]
    fn resolve_detailed_not_found_empty_history() {
        let history = GenerationHistory::default();
        let result = history.resolve_detailed(50);
        assert!(matches!(
            result,
            TimeTravelResult::NotFound {
                requested_seq: 50,
                closest_range: None,
            }
        ));
    }

    #[test]
    fn count_based_pruning() {
        let policy = RetentionPolicy {
            max_retained: 2,
            max_age_ms: 0, // Disable age-based pruning.
        };
        let history = GenerationHistory::new(policy);

        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);
        history.retain(make_generation("gen-3", 201, 300), 3000);

        // Should keep only the 2 most recent.
        assert_eq!(history.retained_count(), 2);
        assert!(history.resolve(50).is_none()); // gen-1 pruned
        assert!(history.resolve(150).is_some()); // gen-2 kept
        assert!(history.resolve(250).is_some()); // gen-3 kept
    }

    #[test]
    fn age_based_pruning() {
        let policy = RetentionPolicy {
            max_retained: 100,
            max_age_ms: 5000,
        };
        let history = GenerationHistory::new(policy);

        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 4000);

        // At t=7000: gen-1 is 6000ms old (>5000), gen-2 is 3000ms old (<5000).
        history.retain(make_generation("gen-3", 201, 300), 7000);

        // gen-1 should be pruned by age.
        assert!(history.resolve(50).is_none());
        assert!(history.resolve(150).is_some());
        assert!(history.resolve(250).is_some());
    }

    #[test]
    fn explicit_prune() {
        let policy = RetentionPolicy {
            max_retained: 100,
            max_age_ms: 5000,
        };
        let history = GenerationHistory::new(policy);

        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);

        assert_eq!(history.retained_count(), 2);

        // Prune at t=8000: both are >5000ms old.
        let pruned = history.prune(8000);
        assert_eq!(pruned, 2);
        assert_eq!(history.retained_count(), 0);
    }

    #[test]
    fn clear_removes_all() {
        let history = GenerationHistory::default();
        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);

        assert_eq!(history.retained_count(), 2);

        history.clear();
        assert_eq!(history.retained_count(), 0);
    }

    #[test]
    fn retained_summary() {
        let history = GenerationHistory::default();
        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);

        let summary = history.retained_summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0], ("gen-1".into(), 1, 100));
        assert_eq!(summary[1], ("gen-2".into(), 101, 200));
    }

    #[test]
    fn sorted_insertion_order() {
        let history = GenerationHistory::default();

        // Insert out of order.
        history.retain(make_generation("gen-3", 201, 300), 3000);
        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);

        let summary = history.retained_summary();
        assert_eq!(summary[0].0, "gen-1");
        assert_eq!(summary[1].0, "gen-2");
        assert_eq!(summary[2].0, "gen-3");
    }

    #[test]
    fn retention_policy_default_is_reasonable() {
        let policy = RetentionPolicy::default();
        assert!(policy.max_retained > 0);
        assert!(policy.max_age_ms > 0);
    }

    #[test]
    fn retention_policy_serde_roundtrip() {
        let policy = RetentionPolicy {
            max_retained: 5,
            max_age_ms: 30_000,
        };
        let json = serde_json::to_string(&policy).expect("serialize");
        let back: RetentionPolicy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(policy, back);
    }

    #[test]
    fn time_travel_result_serde_roundtrip() {
        let results = vec![
            TimeTravelResult::Found {
                generation_id: "gen-1".into(),
                commit_low: 1,
                commit_high: 100,
            },
            TimeTravelResult::NotFound {
                requested_seq: 150,
                closest_range: Some((1, 100)),
            },
            TimeTravelResult::NotFound {
                requested_seq: 50,
                closest_range: None,
            },
            TimeTravelResult::ActiveGeneration {
                generation_id: "gen-active".into(),
            },
        ];
        for result in &results {
            let json = serde_json::to_string(result).expect("serialize");
            let back: TimeTravelResult = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(result, &back);
        }
    }

    #[test]
    fn overlapping_ranges_resolve_first_match() {
        let history = GenerationHistory::default();

        // Two generations with overlapping ranges.
        history.retain(make_generation("gen-1", 1, 150), 1000);
        history.retain(make_generation("gen-2", 100, 200), 2000);

        // Seq 120 is covered by both — should return first match (gen-1).
        let resolved = history.resolve(120).unwrap();
        assert_eq!(resolved.manifest.generation_id, "gen-1");
    }

    #[test]
    fn prune_with_no_age_limit() {
        let policy = RetentionPolicy {
            max_retained: 100,
            max_age_ms: 0, // Disabled.
        };
        let history = GenerationHistory::new(policy);

        history.retain(make_generation("gen-1", 1, 100), 1000);

        // Even at far future time, gen-1 should still be retained.
        let pruned = history.prune(999_999_999);
        assert_eq!(pruned, 0);
        assert_eq!(history.retained_count(), 1);
    }
}
