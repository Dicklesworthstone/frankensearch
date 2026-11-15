//! Time-travel query API for Native Mode distributed search.
//!
//! Allows querying historical search generations by commit sequence number.
//! Each retained generation covers a contiguous commit range; the
//! [`GenerationHistory`] resolves an `as_of_commit_seq` only when exactly one
//! retained snapshot covers it. Overlapping rebuilds and rollback activations
//! require an explicit manifest or activation selection, never insertion order.
//!
//! Callers can retain snapshots from
//! [`GenerationController`](crate::activation::GenerationController), subject to
//! the configured [`RetentionPolicy`]. This in-process registry neither changes
//! durable authority nor authenticates or deletes artifact files. Retaining an
//! `Arc` alone is not a cross-process retention root.

use std::sync::Arc;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::activation::ActiveGeneration;

/// Identity and coverage of one retained activation.
///
/// The manifest hash distinguishes rebuilds with the same generation name or
/// source range. The activation sequence additionally distinguishes rollback
/// activations of the same manifest. This is an in-process history selector,
/// not proof that the referenced artifact files remain present or unmodified.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TimeTravelGeneration {
    /// The generation identifier recorded in the manifest.
    pub generation_id: String,
    /// The canonical manifest hash recorded by activation admission.
    pub manifest_hash: String,
    /// The sequence of the retained activation.
    pub activation_seq: u64,
    /// Inclusive lower source commit bound.
    pub commit_low: u64,
    /// Inclusive upper source commit bound.
    pub commit_high: u64,
}

impl TimeTravelGeneration {
    fn from_generation(generation: &ActiveGeneration) -> Self {
        Self {
            generation_id: generation.manifest.generation_id.clone(),
            manifest_hash: generation.manifest.manifest_hash.to_ascii_lowercase(),
            activation_seq: generation.activation_seq,
            commit_low: generation.manifest.commit_range.low,
            commit_high: generation.manifest.commit_range.high,
        }
    }

    fn matches(&self, generation: &ActiveGeneration) -> bool {
        self.generation_id == generation.manifest.generation_id
            && self
                .manifest_hash
                .eq_ignore_ascii_case(&generation.manifest.manifest_hash)
            && self.activation_seq == generation.activation_seq
            && self.commit_low == generation.manifest.commit_range.low
            && self.commit_high == generation.manifest.commit_range.high
    }
}

/// A historical query must choose between multiple retained snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum TimeTravelError {
    /// More than one retained snapshot matches the request and its selector.
    #[error(
        "commit sequence {requested_seq} matches multiple retained generations; select an exact retained generation"
    )]
    Ambiguous {
        /// The requested commit sequence.
        requested_seq: u64,
        /// Matching activations in deterministic identity order.
        candidates: Vec<TimeTravelGeneration>,
    },
}

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
        /// The canonical manifest hash, usable for explicit selection.
        manifest_hash: String,
        /// The retained activation sequence.
        activation_seq: u64,
        /// Commit range covered.
        commit_low: u64,
        /// Commit range high bound.
        commit_high: u64,
    },
    /// More than one retained snapshot covers the requested commit sequence.
    Ambiguous {
        /// The requested commit sequence.
        requested_seq: u64,
        /// Matching activations in deterministic identity order.
        candidates: Vec<TimeTravelGeneration>,
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
    /// excess generations per the retention policy. Re-retaining the same
    /// `Arc` is idempotent and does not renew its original retention age.
    pub fn retain(&self, generation: Arc<ActiveGeneration>, now_millis: u64) {
        let mut retained = self
            .retained
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // Re-retaining the same owner does not create a second possible
        // answer, consume another retention slot, or renew its age lease.
        if retained
            .iter()
            .any(|entry| Arc::ptr_eq(&entry.generation, &generation))
        {
            self.prune_locked(&mut retained, now_millis);
            return;
        }

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
    /// Returns the unique retained generation whose inclusive commit range
    /// contains the requested sequence, or `Ok(None)` when none is retained.
    /// The returned `Arc` stays valid when this history is pruned or cleared.
    ///
    /// # Errors
    /// Returns [`TimeTravelError::Ambiguous`] for overlapping snapshots. Use
    /// [`Self::resolve_manifest`] or [`Self::resolve_generation`] to select one.
    pub fn resolve(
        &self,
        as_of_commit_seq: u64,
    ) -> Result<Option<Arc<ActiveGeneration>>, TimeTravelError> {
        let retained = self
            .retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        Self::resolve_matching(&retained, as_of_commit_seq, |_| true)
    }

    /// Resolve an explicit manifest within the requested commit range.
    ///
    /// This matches the hash recorded in the admitted manifest; it does not
    /// reauthenticate files on disk. Comparison accepts either hexadecimal case.
    /// An absent hash or a
    /// manifest that does not cover the sequence returns `Ok(None)`; selection
    /// never falls back to another retained manifest.
    ///
    /// # Errors
    /// Returns ambiguity when multiple activations of this manifest cover the
    /// sequence. Select their exact identity with [`Self::resolve_generation`].
    pub fn resolve_manifest(
        &self,
        as_of_commit_seq: u64,
        manifest_hash: &str,
    ) -> Result<Option<Arc<ActiveGeneration>>, TimeTravelError> {
        let retained = self
            .retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::resolve_matching(&retained, as_of_commit_seq, |generation| {
            generation
                .manifest
                .manifest_hash
                .eq_ignore_ascii_case(manifest_hash)
        })
    }

    /// Resolve the exact retained identity reported in an ambiguity result.
    ///
    /// All identity fields and the requested inclusive commit range must match.
    /// Missing or pruned selections return `Ok(None)`, without following a later
    /// activation or another snapshot that reused the generation name.
    ///
    /// # Errors
    /// Returns ambiguity if distinct retained owners advertise the same exact
    /// identity. The caller must repair the history rather than choose by order.
    pub fn resolve_generation(
        &self,
        as_of_commit_seq: u64,
        selected: &TimeTravelGeneration,
    ) -> Result<Option<Arc<ActiveGeneration>>, TimeTravelError> {
        let retained = self
            .retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::resolve_matching(&retained, as_of_commit_seq, |generation| {
            selected.matches(generation)
        })
    }

    /// Resolve with detailed result information.
    #[must_use]
    pub fn resolve_detailed(&self, as_of_commit_seq: u64) -> TimeTravelResult {
        let retained = self
            .retained
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        match Self::resolve_matching(&retained, as_of_commit_seq, |_| true) {
            Ok(Some(generation)) => {
                let range = &generation.manifest.commit_range;
                return TimeTravelResult::Found {
                    generation_id: generation.manifest.generation_id.clone(),
                    manifest_hash: generation.manifest.manifest_hash.to_ascii_lowercase(),
                    activation_seq: generation.activation_seq,
                    commit_low: range.low,
                    commit_high: range.high,
                };
            }
            Err(TimeTravelError::Ambiguous {
                requested_seq,
                candidates,
            }) => {
                return TimeTravelResult::Ambiguous {
                    requested_seq,
                    candidates,
                };
            }
            Ok(None) => {}
        }

        // Find closest range for the error message.
        let closest_range = retained
            .iter()
            .min_by_key(|r| {
                let range = &r.generation.manifest.commit_range;
                let below = as_of_commit_seq.saturating_sub(range.high);
                let above = range.low.saturating_sub(as_of_commit_seq);
                (below.max(above), range.low, range.high)
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

    fn resolve_matching(
        retained: &[RetainedGeneration],
        as_of_commit_seq: u64,
        selected: impl Fn(&ActiveGeneration) -> bool,
    ) -> Result<Option<Arc<ActiveGeneration>>, TimeTravelError> {
        let mut matches = retained.iter().filter(|entry| {
            let generation = &entry.generation;
            let range = &generation.manifest.commit_range;
            range.low <= as_of_commit_seq && as_of_commit_seq <= range.high && selected(generation)
        });
        let Some(first) = matches.next() else {
            return Ok(None);
        };
        let Some(second) = matches.next() else {
            return Ok(Some(Arc::clone(&first.generation)));
        };
        let mut candidates = vec![
            TimeTravelGeneration::from_generation(&first.generation),
            TimeTravelGeneration::from_generation(&second.generation),
        ];
        candidates
            .extend(matches.map(|entry| TimeTravelGeneration::from_generation(&entry.generation)));
        candidates.sort_unstable();
        Err(TimeTravelError::Ambiguous {
            requested_seq: as_of_commit_seq,
            candidates,
        })
    }

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
        if retained.len() > self.policy.max_retained {
            let excess = retained.len() - self.policy.max_retained;
            retained.drain(..excess);
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
            EmbedderRevision::explicit_test_model("potion-128M", 256),
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
        assert!(history.resolve(50).unwrap().is_none());
    }

    #[test]
    fn retain_and_resolve() {
        let history = GenerationHistory::default();
        let generation = make_generation("gen-1", 1, 100);
        history.retain(generation, 1000);

        assert_eq!(history.retained_count(), 1);

        // Within range.
        let resolved = history.resolve(50).unwrap().unwrap();
        assert_eq!(resolved.manifest.generation_id, "gen-1");

        // At boundaries.
        assert!(history.resolve(1).unwrap().is_some());
        assert!(history.resolve(100).unwrap().is_some());

        // Outside range.
        assert!(history.resolve(0).unwrap().is_none());
        assert!(history.resolve(101).unwrap().is_none());
    }

    #[test]
    fn multiple_generations_resolve_correctly() {
        let history = GenerationHistory::default();

        history.retain(make_generation("gen-1", 1, 100), 1000);
        history.retain(make_generation("gen-2", 101, 200), 2000);
        history.retain(make_generation("gen-3", 201, 300), 3000);

        assert_eq!(history.retained_count(), 3);

        let r1 = history.resolve(50).unwrap().unwrap();
        assert_eq!(r1.manifest.generation_id, "gen-1");

        let r2 = history.resolve(150).unwrap().unwrap();
        assert_eq!(r2.manifest.generation_id, "gen-2");

        let r3 = history.resolve(250).unwrap().unwrap();
        assert_eq!(r3.manifest.generation_id, "gen-3");

        // Gap — no generation covers seq 0.
        assert!(history.resolve(0).unwrap().is_none());
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
                ..
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
        assert!(matches!(&result, TimeTravelResult::NotFound { .. }));
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
        assert!(history.resolve(50).unwrap().is_none()); // gen-1 pruned
        assert!(history.resolve(150).unwrap().is_some()); // gen-2 kept
        assert!(history.resolve(250).unwrap().is_some()); // gen-3 kept
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
        assert!(history.resolve(50).unwrap().is_none());
        assert!(history.resolve(150).unwrap().is_some());
        assert!(history.resolve(250).unwrap().is_some());
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
                manifest_hash: "a".repeat(64),
                activation_seq: 1,
                commit_low: 1,
                commit_high: 100,
            },
            TimeTravelResult::Ambiguous {
                requested_seq: 100,
                candidates: vec![
                    TimeTravelGeneration::from_generation(&make_generation("gen-1", 1, 100)),
                    TimeTravelGeneration::from_generation(&make_generation("gen-2", 100, 200)),
                ],
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
    fn overlapping_ranges_require_explicit_selection() {
        let history = GenerationHistory::default();
        let first = make_generation("gen-1", 1, 150);
        let second = make_generation("gen-2", 100, 200);
        history.retain(Arc::clone(&first), 1000);
        history.retain(Arc::clone(&second), 2000);

        let expected = vec![
            TimeTravelGeneration::from_generation(&first),
            TimeTravelGeneration::from_generation(&second),
        ];
        for seq in [100, 120, 150] {
            assert_eq!(
                history.resolve(seq).unwrap_err(),
                TimeTravelError::Ambiguous {
                    requested_seq: seq,
                    candidates: expected.clone(),
                }
            );
            assert_eq!(
                history.resolve_detailed(seq),
                TimeTravelResult::Ambiguous {
                    requested_seq: seq,
                    candidates: expected.clone(),
                }
            );
            for generation in [&first, &second] {
                let selected = history
                    .resolve_manifest(seq, &generation.manifest.manifest_hash)
                    .unwrap()
                    .unwrap();
                assert!(Arc::ptr_eq(&selected, generation));
            }
        }
        assert!(Arc::ptr_eq(&history.resolve(99).unwrap().unwrap(), &first));
        assert!(Arc::ptr_eq(
            &history.resolve(151).unwrap().unwrap(),
            &second
        ));
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

    // ─── bd-2qr7 tests begin ───

    #[test]
    fn retention_policy_default_exact_values() {
        let policy = RetentionPolicy::default();
        assert_eq!(policy.max_retained, 10);
        assert_eq!(policy.max_age_ms, 3_600_000);
    }

    #[test]
    fn retained_generation_debug_format() {
        let generation = make_generation("gen-dbg", 1, 50);
        let retained = RetainedGeneration {
            generation,
            retained_at: 42_000,
        };
        let dbg = format!("{retained:?}");
        assert!(dbg.contains("RetainedGeneration"));
        assert!(dbg.contains("42000"));
    }

    #[test]
    fn resolve_detailed_at_boundaries() {
        let history = GenerationHistory::default();
        history.retain(make_generation("gen-1", 10, 20), 1000);

        // At low boundary.
        let result = history.resolve_detailed(10);
        assert!(matches!(
            result,
            TimeTravelResult::Found {
                commit_low: 10,
                commit_high: 20,
                ..
            }
        ));

        // At high boundary.
        let result = history.resolve_detailed(20);
        assert!(matches!(
            result,
            TimeTravelResult::Found {
                commit_low: 10,
                commit_high: 20,
                ..
            }
        ));

        // One below low → not found.
        let result = history.resolve_detailed(9);
        assert!(matches!(&result, TimeTravelResult::NotFound { .. }));

        // One above high → not found.
        let result = history.resolve_detailed(21);
        assert!(matches!(result, TimeTravelResult::NotFound { .. }));
    }

    #[test]
    fn prune_returns_zero_when_empty() {
        let history = GenerationHistory::default();
        let pruned = history.prune(999_999);
        assert_eq!(pruned, 0);
        assert_eq!(history.retained_count(), 0);
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let history = GenerationHistory::default();
        history.clear();
        assert_eq!(history.retained_count(), 0);
    }

    #[test]
    fn retained_summary_on_empty() {
        let history = GenerationHistory::default();
        let summary = history.retained_summary();
        assert!(summary.is_empty());
    }

    #[test]
    fn count_pruning_keeps_newest_drops_oldest() {
        let policy = RetentionPolicy {
            max_retained: 3,
            max_age_ms: 0,
        };
        let history = GenerationHistory::new(policy);

        // Insert 5 generations.
        for i in 1..=5 {
            let low = (i - 1) * 100 + 1;
            let high = i * 100;
            history.retain(make_generation(&format!("gen-{i}"), low, high), i * 1000);
        }

        // Should keep 3 newest by drain(..excess) which removes lowest-indexed
        // (sorted by commit_range.high ascending, so oldest by commit range).
        assert_eq!(history.retained_count(), 3);
        let summary = history.retained_summary();
        assert_eq!(summary[0].0, "gen-3");
        assert_eq!(summary[1].0, "gen-4");
        assert_eq!(summary[2].0, "gen-5");
    }

    #[test]
    fn time_travel_result_debug_format() {
        let found = TimeTravelResult::Found {
            generation_id: "g1".into(),
            manifest_hash: "a".repeat(64),
            activation_seq: 1,
            commit_low: 1,
            commit_high: 10,
        };
        assert!(format!("{found:?}").contains("Found"));

        let not_found = TimeTravelResult::NotFound {
            requested_seq: 99,
            closest_range: None,
        };
        assert!(format!("{not_found:?}").contains("NotFound"));
        assert!(format!("{not_found:?}").contains("99"));

        let active = TimeTravelResult::ActiveGeneration {
            generation_id: "active-1".into(),
        };
        assert!(format!("{active:?}").contains("ActiveGeneration"));
    }

    #[test]
    fn equal_high_bounds_are_ambiguous_in_either_insertion_order() {
        let first = make_generation("gen-a", 1, 100);
        let second = make_generation("gen-b", 50, 100);
        let mut outcomes = Vec::new();
        for generations in [[&first, &second], [&second, &first]] {
            let history = GenerationHistory::default();
            for generation in generations {
                history.retain(Arc::clone(generation), 1000);
            }
            assert_eq!(history.retained_count(), 2);
            outcomes.push(history.resolve(75).unwrap_err());
            assert!(matches!(
                history.resolve_detailed(100),
                TimeTravelResult::Ambiguous { .. }
            ));
            assert!(Arc::ptr_eq(&history.resolve(49).unwrap().unwrap(), &first));
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }

    #[test]
    fn manifest_selection_distinguishes_rebuilt_generation_names_without_fallback() {
        let history = GenerationHistory::default();
        let first = make_generation("reused-name", 1, 100);
        let mut rebuilt = first.as_ref().clone();
        rebuilt
            .manifest
            .metadata
            .insert("build".into(), "replacement".into());
        rebuilt.manifest.manifest_hash = compute_manifest_hash(&rebuilt.manifest).unwrap();
        let rebuilt = Arc::new(rebuilt);
        history.retain(Arc::clone(&first), 1000);
        history.retain(Arc::clone(&rebuilt), 1000);

        assert!(matches!(
            history.resolve(50),
            Err(TimeTravelError::Ambiguous { .. })
        ));
        for generation in [&first, &rebuilt] {
            let selected = history
                .resolve_manifest(50, &generation.manifest.manifest_hash.to_ascii_uppercase())
                .unwrap()
                .unwrap();
            assert!(Arc::ptr_eq(&selected, generation));
            assert!(
                history
                    .resolve_manifest(101, &generation.manifest.manifest_hash)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            history
                .resolve_manifest(50, &"0".repeat(64))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn repeated_manifest_activation_requires_the_exact_activation() {
        let history = GenerationHistory::default();
        let original = make_generation("gen-1", 1, 100);
        let mut rollback = original.as_ref().clone();
        rollback.activation_seq = original.activation_seq + 2;
        rollback.activated_at += 2000;
        let rollback = Arc::new(rollback);
        history.retain(Arc::clone(&original), 1000);
        history.retain(Arc::clone(&rollback), 3000);

        let error = history
            .resolve_manifest(50, &original.manifest.manifest_hash)
            .unwrap_err();
        let TimeTravelError::Ambiguous { candidates, .. } = &error;
        assert_eq!(candidates.len(), 2);
        for (candidate, expected) in candidates.iter().zip([&original, &rollback]) {
            let resolved = history.resolve_generation(50, candidate).unwrap().unwrap();
            assert!(Arc::ptr_eq(&resolved, expected));
            let mut stale = candidate.clone();
            stale.activation_seq += 100;
            assert!(history.resolve_generation(50, &stale).unwrap().is_none());
            assert!(history.resolve_generation(0, candidate).unwrap().is_none());
        }
        let json = serde_json::to_string(&error).unwrap();
        assert_eq!(
            serde_json::from_str::<TimeTravelError>(&json).unwrap(),
            error
        );
    }

    #[test]
    fn distinct_owners_with_duplicate_identity_are_never_chosen_by_order() {
        let history = GenerationHistory::default();
        let original = make_generation("gen-1", 1, 100);
        let mut conflicting = original.as_ref().clone();
        conflicting
            .vector_paths
            .insert("fast".into(), "/different-owner".into());
        history.retain(Arc::clone(&original), 1000);
        history.retain(Arc::new(conflicting), 1000);
        let selected = TimeTravelGeneration::from_generation(&original);
        assert!(matches!(
            history.resolve_generation(50, &selected),
            Err(TimeTravelError::Ambiguous { .. })
        ));
    }

    #[test]
    fn retaining_the_same_owner_does_not_duplicate_or_renew_its_lease() {
        let history = GenerationHistory::new(RetentionPolicy {
            max_retained: 2,
            max_age_ms: 1000,
        });
        let first = make_generation("gen-1", 1, 100);
        history.retain(Arc::clone(&first), 1000);
        history.retain(Arc::clone(&first), 1500);
        assert_eq!(history.retained_count(), 1);
        let pinned = history.resolve(50).unwrap().unwrap();
        assert_eq!(history.prune(2001), 1);
        assert!(history.resolve(50).unwrap().is_none());
        assert_eq!(pinned.manifest.generation_id, "gen-1");
        assert!(Arc::ptr_eq(&pinned, &first));
    }

    #[test]
    fn concurrent_retention_reports_every_matching_owner_and_pinned_read_survives_clear() {
        let history = GenerationHistory::default();
        let original = make_generation("gen-original", 1, 100);
        history.retain(Arc::clone(&original), 1000);
        let pinned = history.resolve(50).unwrap().unwrap();
        std::thread::scope(|scope| {
            for id in ["gen-a", "gen-b", "gen-c"] {
                let history = &history;
                scope.spawn(move || history.retain(make_generation(id, 1, 100), 1000));
            }
        });
        let TimeTravelError::Ambiguous { candidates, .. } = history.resolve(50).unwrap_err();
        assert_eq!(candidates.len(), 4);
        assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
        history.clear();
        assert!(
            history
                .resolve_generation(50, &TimeTravelGeneration::from_generation(&pinned))
                .unwrap()
                .is_none()
        );
        assert!(Arc::ptr_eq(&pinned, &original));
    }

    #[test]
    fn resolve_detailed_closest_range_accuracy() {
        let history = GenerationHistory::default();

        // Ranges: [1,100], [500,600]
        history.retain(make_generation("gen-near", 1, 100), 1000);
        history.retain(make_generation("gen-far", 500, 600), 2000);

        // Seq 120 is 20 away from gen-near (high=100) and 380 from gen-far (low=500).
        let result = history.resolve_detailed(120);
        assert!(matches!(result, TimeTravelResult::NotFound { .. }));
        if let TimeTravelResult::NotFound { closest_range, .. } = result {
            assert_eq!(closest_range, Some((1, 100)));
        }

        // Seq 400 is 300 away from gen-near and 100 from gen-far.
        let result = history.resolve_detailed(400);
        assert!(matches!(result, TimeTravelResult::NotFound { .. }));
        if let TimeTravelResult::NotFound { closest_range, .. } = result {
            assert_eq!(closest_range, Some((500, 600)));
        }
    }

    // ─── bd-2qr7 tests end ───
}
