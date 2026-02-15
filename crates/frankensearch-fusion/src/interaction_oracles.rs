//! Cross-feature invariant oracles for interaction lane testing.
//!
//! Each oracle is a composable assertion function that checks a specific
//! invariant class against search results. Lanes declare which oracles
//! apply, and test runners execute them in order.
//!
//! Oracle categories:
//! - **Ordering**: Result ordering stability, deterministic tie-breaking
//! - **Phase**: Phase transition correctness (`Initial` -> `Refined` | `RefinementFailed`)
//! - **Explanation**: Per-hit explanation completeness and consistency
//! - **Calibration**: Score distribution properties after calibration
//! - **Feedback**: Boost map effect on ranking
//! - **Fallback**: Graceful degradation when features are unavailable

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::interaction_lanes::{ExpectedPhase, FeatureToggles, InteractionLane};

// ── Oracle verdict ───────────────────────────────────────────────────

/// Result of running an oracle against a search result set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleVerdict {
    /// Which oracle produced this verdict.
    pub oracle_id: String,
    /// Lane that was being tested.
    pub lane_id: String,
    /// Pass or fail with details.
    pub outcome: OracleOutcome,
    /// Human-readable context for debugging failures.
    pub context: String,
}

/// Pass/fail/skip outcome for an oracle check.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OracleOutcome {
    /// All assertions passed.
    Pass,
    /// One or more assertions failed.
    Fail,
    /// Oracle was not applicable to this lane (e.g., explanation oracle on
    /// a lane with `explain: false`).
    Skip,
}

impl OracleVerdict {
    /// Create a passing verdict.
    #[must_use]
    pub fn pass(oracle_id: impl Into<String>, lane_id: impl Into<String>) -> Self {
        Self {
            oracle_id: oracle_id.into(),
            lane_id: lane_id.into(),
            outcome: OracleOutcome::Pass,
            context: String::new(),
        }
    }

    /// Create a failing verdict with context.
    #[must_use]
    pub fn fail(oracle_id: impl Into<String>, lane_id: impl Into<String>, context: String) -> Self {
        Self {
            oracle_id: oracle_id.into(),
            lane_id: lane_id.into(),
            outcome: OracleOutcome::Fail,
            context,
        }
    }

    /// Create a skipped verdict.
    #[must_use]
    pub fn skip(oracle_id: impl Into<String>, lane_id: impl Into<String>, reason: &str) -> Self {
        Self {
            oracle_id: oracle_id.into(),
            lane_id: lane_id.into(),
            outcome: OracleOutcome::Skip,
            context: reason.to_string(),
        }
    }

    /// True if the oracle passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.outcome == OracleOutcome::Pass
    }

    /// True if the oracle was skipped.
    #[must_use]
    pub fn skipped(&self) -> bool {
        self.outcome == OracleOutcome::Skip
    }
}

impl fmt::Display for OracleVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self.outcome {
            OracleOutcome::Pass => "PASS",
            OracleOutcome::Fail => "FAIL",
            OracleOutcome::Skip => "SKIP",
        };
        write!(
            f,
            "[{status}] {oracle}/{lane}",
            oracle = self.oracle_id,
            lane = self.lane_id,
        )?;
        if !self.context.is_empty() {
            write!(f, " — {}", self.context)?;
        }
        Ok(())
    }
}

// ── Invariant kinds ──────────────────────────────────────────────────

/// Categories of invariants that oracles check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InvariantCategory {
    /// Result ordering and deterministic tie-breaking.
    Ordering,
    /// Phase transition correctness.
    Phase,
    /// Explanation completeness and accuracy.
    Explanation,
    /// Score calibration distribution properties.
    Calibration,
    /// Feedback boost application.
    Feedback,
    /// Graceful degradation / circuit breaker fallback.
    Fallback,
    /// Conformal prediction coverage guarantees.
    Conformal,
    /// Negative/exclusion query semantics.
    Exclusion,
    /// MMR diversity properties.
    Diversity,
    /// PRF expansion properties.
    Expansion,
    /// Adaptive fusion posterior updates.
    Adaptive,
}

impl fmt::Display for InvariantCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ordering => write!(f, "ordering"),
            Self::Phase => write!(f, "phase"),
            Self::Explanation => write!(f, "explanation"),
            Self::Calibration => write!(f, "calibration"),
            Self::Feedback => write!(f, "feedback"),
            Self::Fallback => write!(f, "fallback"),
            Self::Conformal => write!(f, "conformal"),
            Self::Exclusion => write!(f, "exclusion"),
            Self::Diversity => write!(f, "diversity"),
            Self::Expansion => write!(f, "expansion"),
            Self::Adaptive => write!(f, "adaptive"),
        }
    }
}

// ── Oracle descriptors ───────────────────────────────────────────────

/// Describes an oracle assertion that can be applied to a lane's results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OracleDescriptor {
    /// Stable oracle identifier.
    pub id: &'static str,
    /// Human-readable description of what this oracle checks.
    pub description: &'static str,
    /// Invariant category.
    pub category: InvariantCategory,
    /// Which feature toggles must be active for this oracle to apply.
    /// `None` means the oracle applies unconditionally.
    pub requires: OracleRequirements,
}

/// Prerequisites for an oracle to be applicable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct OracleRequirements {
    /// Features that must be enabled (if any).
    pub features: &'static [RequiredFeature],
    /// Expected phase behavior the lane must declare.
    pub expected_phase: Option<ExpectedPhase>,
}

/// A feature requirement for an oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequiredFeature {
    Explain,
    Mmr,
    NegationQueries,
    Prf,
    AdaptiveFusion,
    NonIdentityCalibration,
    Conformal,
    CircuitBreaker,
    Feedback,
}

impl RequiredFeature {
    /// Check if this requirement is satisfied by the given toggles.
    #[must_use]
    pub fn satisfied_by(self, toggles: &FeatureToggles) -> bool {
        match self {
            Self::Explain => toggles.explain,
            Self::Mmr => toggles.mmr,
            Self::NegationQueries => toggles.negation_queries,
            Self::Prf => toggles.prf,
            Self::AdaptiveFusion => toggles.adaptive_fusion,
            Self::NonIdentityCalibration => {
                toggles.calibration != crate::interaction_lanes::CalibratorChoice::Identity
            }
            Self::Conformal => toggles.conformal,
            Self::CircuitBreaker => toggles.circuit_breaker,
            Self::Feedback => toggles.feedback,
        }
    }
}

// ── Oracle catalog ───────────────────────────────────────────────────

// -- Ordering oracles --

/// Two identical queries with the same seed must produce identical result
/// ordering (including `doc_id` sequence).
pub const ORACLE_DETERMINISTIC_ORDERING: OracleDescriptor = OracleDescriptor {
    id: "deterministic_ordering",
    description: "Identical query+seed produces identical result ordering",
    category: InvariantCategory::Ordering,
    requires: OracleRequirements {
        features: &[],
        expected_phase: None,
    },
};

/// No duplicate `doc_ids` in the result set.
pub const ORACLE_NO_DUPLICATES: OracleDescriptor = OracleDescriptor {
    id: "no_duplicates",
    description: "Result set contains no duplicate doc_ids",
    category: InvariantCategory::Ordering,
    requires: OracleRequirements {
        features: &[],
        expected_phase: None,
    },
};

/// Scores are monotonically non-increasing in the result set.
pub const ORACLE_MONOTONIC_SCORES: OracleDescriptor = OracleDescriptor {
    id: "monotonic_scores",
    description: "Scores are monotonically non-increasing (descending order)",
    category: InvariantCategory::Ordering,
    requires: OracleRequirements {
        features: &[],
        expected_phase: None,
    },
};

// -- Phase oracles --

/// Phase 1 (Initial) always yields results (possibly empty for Empty queries).
pub const ORACLE_PHASE1_ALWAYS_YIELDS: OracleDescriptor = OracleDescriptor {
    id: "phase1_always_yields",
    description: "Phase 1 (Initial) always completes and yields a result set",
    category: InvariantCategory::Phase,
    requires: OracleRequirements {
        features: &[],
        expected_phase: None,
    },
};

/// When `expected_phase` is `InitialThenRefined`, Phase 2 must yield `Refined`.
pub const ORACLE_PHASE2_REFINED: OracleDescriptor = OracleDescriptor {
    id: "phase2_refined",
    description: "Phase 2 yields Refined when circuit breaker is closed and quality tier is healthy",
    category: InvariantCategory::Phase,
    requires: OracleRequirements {
        features: &[],
        expected_phase: Some(ExpectedPhase::InitialThenRefined),
    },
};

/// When `expected_phase` is `InitialThenMaybeRefined`, Phase 2 yields either
/// `Refined` or `RefinementFailed` (never panic/hang).
pub const ORACLE_PHASE2_GRACEFUL: OracleDescriptor = OracleDescriptor {
    id: "phase2_graceful",
    description: "Phase 2 yields either Refined or RefinementFailed (no panic/hang)",
    category: InvariantCategory::Phase,
    requires: OracleRequirements {
        features: &[],
        expected_phase: Some(ExpectedPhase::InitialThenMaybeRefined),
    },
};

/// Phase 1 results are a superset of Phase 2 results for the same `doc_ids`
/// (refinement may reorder or remove, but never add new `doc_ids` not in the
/// candidate pool).
pub const ORACLE_REFINEMENT_SUBSET: OracleDescriptor = OracleDescriptor {
    id: "refinement_subset",
    description: "Phase 2 doc_ids are a subset of Phase 1 candidate pool",
    category: InvariantCategory::Phase,
    requires: OracleRequirements {
        features: &[],
        expected_phase: None,
    },
};

// -- Explanation oracles --

/// When `explain: true`, every result has a non-empty `HitExplanation`.
pub const ORACLE_EXPLAIN_PRESENT: OracleDescriptor = OracleDescriptor {
    id: "explain_present",
    description: "Every result has a HitExplanation when explain=true",
    category: InvariantCategory::Explanation,
    requires: OracleRequirements {
        features: &[RequiredFeature::Explain],
        expected_phase: None,
    },
};

/// Explanation `final_score` matches the result's `score` field.
pub const ORACLE_EXPLAIN_SCORE_CONSISTENT: OracleDescriptor = OracleDescriptor {
    id: "explain_score_consistent",
    description: "Explanation final_score matches result score (within epsilon)",
    category: InvariantCategory::Explanation,
    requires: OracleRequirements {
        features: &[RequiredFeature::Explain],
        expected_phase: None,
    },
};

/// When MMR reranks results, explanation `rank_movement` is populated with
/// correct `initial_rank` and `refined_rank` values.
pub const ORACLE_EXPLAIN_MMR_RANK_MOVEMENT: OracleDescriptor = OracleDescriptor {
    id: "explain_mmr_rank_movement",
    description: "MMR reranking produces correct rank_movement in explanations",
    category: InvariantCategory::Explanation,
    requires: OracleRequirements {
        features: &[RequiredFeature::Explain, RequiredFeature::Mmr],
        expected_phase: Some(ExpectedPhase::InitialThenRefined),
    },
};

/// Explanation `ExplanationPhase` matches the search phase that produced it.
pub const ORACLE_EXPLAIN_PHASE_MATCHES: OracleDescriptor = OracleDescriptor {
    id: "explain_phase_matches",
    description: "Explanation phase field matches the search phase (Initial or Refined)",
    category: InvariantCategory::Explanation,
    requires: OracleRequirements {
        features: &[RequiredFeature::Explain],
        expected_phase: None,
    },
};

// -- Calibration oracles --

/// Calibrated scores are in [0.0, 1.0].
pub const ORACLE_CALIBRATED_RANGE: OracleDescriptor = OracleDescriptor {
    id: "calibrated_range",
    description: "All calibrated scores are in [0.0, 1.0]",
    category: InvariantCategory::Calibration,
    requires: OracleRequirements {
        features: &[RequiredFeature::NonIdentityCalibration],
        expected_phase: None,
    },
};

/// Calibration preserves relative ordering (monotonicity): if `raw_a` > `raw_b`
/// then `calibrated_a` >= `calibrated_b`.
pub const ORACLE_CALIBRATION_MONOTONIC: OracleDescriptor = OracleDescriptor {
    id: "calibration_monotonic",
    description: "Calibration preserves relative score ordering (monotonic)",
    category: InvariantCategory::Calibration,
    requires: OracleRequirements {
        features: &[RequiredFeature::NonIdentityCalibration],
        expected_phase: None,
    },
};

/// Explanation `normalized_score` reflects post-calibration values when
/// a calibrator is active.
pub const ORACLE_EXPLAIN_CALIBRATED_SCORES: OracleDescriptor = OracleDescriptor {
    id: "explain_calibrated_scores",
    description: "Explanation normalized_score reflects post-calibration values",
    category: InvariantCategory::Calibration,
    requires: OracleRequirements {
        features: &[
            RequiredFeature::Explain,
            RequiredFeature::NonIdentityCalibration,
        ],
        expected_phase: None,
    },
};

// -- Feedback oracles --

/// Documents with positive feedback history have scores >= their no-feedback
/// baseline scores.
pub const ORACLE_FEEDBACK_BOOST_POSITIVE: OracleDescriptor = OracleDescriptor {
    id: "feedback_boost_positive",
    description: "Positive feedback boosts raise scores above no-feedback baseline",
    category: InvariantCategory::Feedback,
    requires: OracleRequirements {
        features: &[RequiredFeature::Feedback],
        expected_phase: None,
    },
};

/// Feedback boost values are within [`min_boost`, `max_boost`] after decay.
pub const ORACLE_FEEDBACK_BOOST_CLAMPED: OracleDescriptor = OracleDescriptor {
    id: "feedback_boost_clamped",
    description: "Effective boost values are within configured [min_boost, max_boost]",
    category: InvariantCategory::Feedback,
    requires: OracleRequirements {
        features: &[RequiredFeature::Feedback],
        expected_phase: None,
    },
};

// -- Fallback oracles --

/// When circuit breaker is open, Phase 2 is skipped and `RefinementFailed`
/// is emitted with appropriate error context.
pub const ORACLE_BREAKER_SKIPS_PHASE2: OracleDescriptor = OracleDescriptor {
    id: "breaker_skips_phase2",
    description: "Open circuit breaker skips Phase 2 with RefinementFailed",
    category: InvariantCategory::Fallback,
    requires: OracleRequirements {
        features: &[RequiredFeature::CircuitBreaker],
        expected_phase: Some(ExpectedPhase::InitialThenMaybeRefined),
    },
};

/// When circuit breaker skips Phase 2, Phase 1 results are still returned
/// (not empty, unless query is Empty class).
pub const ORACLE_BREAKER_PHASE1_PRESERVED: OracleDescriptor = OracleDescriptor {
    id: "breaker_phase1_preserved",
    description: "Circuit breaker degradation preserves Phase 1 results",
    category: InvariantCategory::Fallback,
    requires: OracleRequirements {
        features: &[RequiredFeature::CircuitBreaker],
        expected_phase: Some(ExpectedPhase::InitialThenMaybeRefined),
    },
};

/// When circuit breaker is open and adaptive fusion is active, posteriors
/// are NOT updated (no evidence from skipped phases).
pub const ORACLE_BREAKER_NO_POSTERIOR_UPDATE: OracleDescriptor = OracleDescriptor {
    id: "breaker_no_posterior_update",
    description: "Skipped Phase 2 does not update adaptive fusion posteriors",
    category: InvariantCategory::Fallback,
    requires: OracleRequirements {
        features: &[
            RequiredFeature::CircuitBreaker,
            RequiredFeature::AdaptiveFusion,
        ],
        expected_phase: Some(ExpectedPhase::InitialThenMaybeRefined),
    },
};

// -- Exclusion oracles --

/// Documents matching negated terms/phrases are excluded from results.
pub const ORACLE_EXCLUSION_APPLIED: OracleDescriptor = OracleDescriptor {
    id: "exclusion_applied",
    description: "Documents matching negated terms/phrases are excluded from results",
    category: InvariantCategory::Exclusion,
    requires: OracleRequirements {
        features: &[RequiredFeature::NegationQueries],
        expected_phase: None,
    },
};

/// Explanation components for excluded sources are absent.
pub const ORACLE_EXPLAIN_EXCLUSION_ABSENT: OracleDescriptor = OracleDescriptor {
    id: "explain_exclusion_absent",
    description: "Explanation components omit excluded terms/sources",
    category: InvariantCategory::Exclusion,
    requires: OracleRequirements {
        features: &[RequiredFeature::NegationQueries, RequiredFeature::Explain],
        expected_phase: None,
    },
};

// -- Diversity oracles --

/// MMR reranked results have lower maximum pairwise similarity than
/// non-MMR results (diversity increased).
pub const ORACLE_MMR_DIVERSITY_INCREASED: OracleDescriptor = OracleDescriptor {
    id: "mmr_diversity_increased",
    description: "MMR reduces maximum pairwise similarity vs non-MMR baseline",
    category: InvariantCategory::Diversity,
    requires: OracleRequirements {
        features: &[RequiredFeature::Mmr],
        expected_phase: Some(ExpectedPhase::InitialThenRefined),
    },
};

/// MMR result set has no duplicates and respects the limit parameter.
pub const ORACLE_MMR_LIMIT_RESPECTED: OracleDescriptor = OracleDescriptor {
    id: "mmr_limit_respected",
    description: "MMR result count does not exceed requested limit",
    category: InvariantCategory::Diversity,
    requires: OracleRequirements {
        features: &[RequiredFeature::Mmr],
        expected_phase: None,
    },
};

// -- Expansion oracles --

/// PRF expanded embedding is L2-normalized.
pub const ORACLE_PRF_NORMALIZED: OracleDescriptor = OracleDescriptor {
    id: "prf_normalized",
    description: "PRF-expanded embedding is L2-normalized (norm within [0.99, 1.01])",
    category: InvariantCategory::Expansion,
    requires: OracleRequirements {
        features: &[RequiredFeature::Prf],
        expected_phase: Some(ExpectedPhase::InitialThenRefined),
    },
};

/// PRF expansion only activates for `NaturalLanguage` queries (not `Identifier`
/// or `ShortKeyword`).
pub const ORACLE_PRF_QUERY_CLASS_GUARD: OracleDescriptor = OracleDescriptor {
    id: "prf_query_class_guard",
    description: "PRF only activates for NaturalLanguage queries",
    category: InvariantCategory::Expansion,
    requires: OracleRequirements {
        features: &[RequiredFeature::Prf],
        expected_phase: None,
    },
};

/// When PRF is active with negation, feedback docs matching excluded terms
/// are not included in the centroid computation.
pub const ORACLE_PRF_RESPECTS_NEGATION: OracleDescriptor = OracleDescriptor {
    id: "prf_respects_negation",
    description: "PRF centroid excludes docs matching negated terms",
    category: InvariantCategory::Expansion,
    requires: OracleRequirements {
        features: &[RequiredFeature::Prf, RequiredFeature::NegationQueries],
        expected_phase: Some(ExpectedPhase::InitialThenRefined),
    },
};

// -- Adaptive oracles --

/// After sufficient queries, adaptive blend posterior mean converges toward
/// the blend weight that produced better results.
pub const ORACLE_ADAPTIVE_BLEND_CONVERGES: OracleDescriptor = OracleDescriptor {
    id: "adaptive_blend_converges",
    description: "Adaptive blend posterior converges toward better blend weight",
    category: InvariantCategory::Adaptive,
    requires: OracleRequirements {
        features: &[RequiredFeature::AdaptiveFusion],
        expected_phase: None,
    },
};

/// Adaptive K posterior stays within configured [`k_min`, `k_max`] bounds.
pub const ORACLE_ADAPTIVE_K_BOUNDED: OracleDescriptor = OracleDescriptor {
    id: "adaptive_k_bounded",
    description: "Adaptive K posterior mean stays within [k_min, k_max]",
    category: InvariantCategory::Adaptive,
    requires: OracleRequirements {
        features: &[RequiredFeature::AdaptiveFusion],
        expected_phase: None,
    },
};

// -- Conformal oracles --

/// Conformal prediction `required_k(alpha)` returns a value >= 1 for
/// any alpha in (0, 1).
pub const ORACLE_CONFORMAL_K_POSITIVE: OracleDescriptor = OracleDescriptor {
    id: "conformal_k_positive",
    description: "Conformal required_k returns >= 1 for valid alpha",
    category: InvariantCategory::Conformal,
    requires: OracleRequirements {
        features: &[RequiredFeature::Conformal],
        expected_phase: None,
    },
};

/// Conformal coverage holds: fraction of queries where relevant doc appears
/// within `required_k(alpha)` results is >= (1 - alpha).
pub const ORACLE_CONFORMAL_COVERAGE: OracleDescriptor = OracleDescriptor {
    id: "conformal_coverage",
    description: "Empirical coverage >= (1-alpha) for calibration queries",
    category: InvariantCategory::Conformal,
    requires: OracleRequirements {
        features: &[RequiredFeature::Conformal],
        expected_phase: None,
    },
};

// ── Oracle catalog accessor ──────────────────────────────────────────

/// Returns all defined oracle descriptors.
#[must_use]
pub fn all_oracles() -> Vec<OracleDescriptor> {
    vec![
        // Ordering (universal)
        ORACLE_DETERMINISTIC_ORDERING,
        ORACLE_NO_DUPLICATES,
        ORACLE_MONOTONIC_SCORES,
        // Phase (universal / conditional)
        ORACLE_PHASE1_ALWAYS_YIELDS,
        ORACLE_PHASE2_REFINED,
        ORACLE_PHASE2_GRACEFUL,
        ORACLE_REFINEMENT_SUBSET,
        // Explanation
        ORACLE_EXPLAIN_PRESENT,
        ORACLE_EXPLAIN_SCORE_CONSISTENT,
        ORACLE_EXPLAIN_MMR_RANK_MOVEMENT,
        ORACLE_EXPLAIN_PHASE_MATCHES,
        // Calibration
        ORACLE_CALIBRATED_RANGE,
        ORACLE_CALIBRATION_MONOTONIC,
        ORACLE_EXPLAIN_CALIBRATED_SCORES,
        // Feedback
        ORACLE_FEEDBACK_BOOST_POSITIVE,
        ORACLE_FEEDBACK_BOOST_CLAMPED,
        // Fallback
        ORACLE_BREAKER_SKIPS_PHASE2,
        ORACLE_BREAKER_PHASE1_PRESERVED,
        ORACLE_BREAKER_NO_POSTERIOR_UPDATE,
        // Exclusion
        ORACLE_EXCLUSION_APPLIED,
        ORACLE_EXPLAIN_EXCLUSION_ABSENT,
        // Diversity
        ORACLE_MMR_DIVERSITY_INCREASED,
        ORACLE_MMR_LIMIT_RESPECTED,
        // Expansion
        ORACLE_PRF_NORMALIZED,
        ORACLE_PRF_QUERY_CLASS_GUARD,
        ORACLE_PRF_RESPECTS_NEGATION,
        // Adaptive
        ORACLE_ADAPTIVE_BLEND_CONVERGES,
        ORACLE_ADAPTIVE_K_BOUNDED,
        // Conformal
        ORACLE_CONFORMAL_K_POSITIVE,
        ORACLE_CONFORMAL_COVERAGE,
    ]
}

/// Returns oracles applicable to a given lane based on its feature toggles
/// and expected phase behavior.
#[must_use]
pub fn oracles_for_lane(lane: &InteractionLane) -> Vec<OracleDescriptor> {
    all_oracles()
        .into_iter()
        .filter(|oracle| oracle_applicable(oracle, lane))
        .collect()
}

/// Check if an oracle is applicable to a lane.
#[must_use]
pub fn oracle_applicable(oracle: &OracleDescriptor, lane: &InteractionLane) -> bool {
    // Check feature requirements
    for req in oracle.requires.features {
        if !req.satisfied_by(&lane.toggles) {
            return false;
        }
    }

    // Check phase requirement
    if let Some(required_phase) = oracle.requires.expected_phase
        && lane.expected_phase != required_phase
    {
        return false;
    }

    true
}

// ── Runner templates ────────────────────────────────────────────────

/// Coarse-grained invariant groups expected for each interaction lane.
///
/// These groups are used by higher-level runners to assert lane completeness
/// before executing oracle-level checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InvariantGroup {
    /// Deterministic ordering and no-duplicate guarantees.
    Ordering,
    /// Progressive phase transition expectations.
    PhaseTransitions,
    /// Machine-readable reason-code contracts.
    ReasonCodes,
    /// Degraded-mode and graceful-fallback expectations.
    FallbackSemantics,
    /// Explanation payload integrity.
    ExplanationIntegrity,
    /// Calibration correctness and stability.
    CalibrationConsistency,
    /// Feedback-loop correctness.
    FeedbackConsistency,
    /// Negative-query exclusion semantics.
    ExclusionSemantics,
    /// MMR diversity guarantees.
    DiversityGuarantees,
    /// PRF expansion semantics.
    ExpansionSemantics,
    /// Adaptive posterior stability guarantees.
    AdaptiveStability,
    /// Conformal coverage guarantees.
    ConformalCoverage,
}

impl fmt::Display for InvariantGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ordering => write!(f, "ordering"),
            Self::PhaseTransitions => write!(f, "phase_transitions"),
            Self::ReasonCodes => write!(f, "reason_codes"),
            Self::FallbackSemantics => write!(f, "fallback_semantics"),
            Self::ExplanationIntegrity => write!(f, "explanation_integrity"),
            Self::CalibrationConsistency => write!(f, "calibration_consistency"),
            Self::FeedbackConsistency => write!(f, "feedback_consistency"),
            Self::ExclusionSemantics => write!(f, "exclusion_semantics"),
            Self::DiversityGuarantees => write!(f, "diversity_guarantees"),
            Self::ExpansionSemantics => write!(f, "expansion_semantics"),
            Self::AdaptiveStability => write!(f, "adaptive_stability"),
            Self::ConformalCoverage => write!(f, "conformal_coverage"),
        }
    }
}

/// Deterministic execution template consumed by integration/e2e test runners.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneOracleTemplate {
    /// Lane identifier.
    pub lane_id: String,
    /// Deterministic lane seed.
    pub seed: u64,
    /// Expected progressive-phase behavior.
    pub expected_phase: ExpectedPhase,
    /// Required invariant groups for this lane.
    pub invariant_groups: Vec<InvariantGroup>,
    /// Ordered list of oracle IDs to run.
    pub oracle_ids: Vec<String>,
    /// Expected reason-code keys.
    pub reason_codes: Vec<String>,
    /// Expected metric keys.
    pub metric_keys: Vec<String>,
    /// Expected log event keys.
    pub log_events: Vec<String>,
}

fn lane_groups(lane: &InteractionLane) -> Vec<InvariantGroup> {
    let mut groups = vec![
        InvariantGroup::Ordering,
        InvariantGroup::PhaseTransitions,
        InvariantGroup::ReasonCodes,
        InvariantGroup::FallbackSemantics,
    ];

    if lane.toggles.explain {
        groups.push(InvariantGroup::ExplanationIntegrity);
    }
    if lane.toggles.mmr {
        groups.push(InvariantGroup::DiversityGuarantees);
    }
    if lane.toggles.negation_queries {
        groups.push(InvariantGroup::ExclusionSemantics);
    }
    if lane.toggles.prf {
        groups.push(InvariantGroup::ExpansionSemantics);
    }
    if lane.toggles.adaptive_fusion {
        groups.push(InvariantGroup::AdaptiveStability);
    }
    if lane.toggles.calibration != crate::interaction_lanes::CalibratorChoice::Identity {
        groups.push(InvariantGroup::CalibrationConsistency);
    }
    if lane.toggles.conformal {
        groups.push(InvariantGroup::ConformalCoverage);
    }
    if lane.toggles.feedback {
        groups.push(InvariantGroup::FeedbackConsistency);
    }

    groups.sort_unstable();
    groups.dedup();
    groups
}

fn lane_reason_codes(lane: &InteractionLane) -> Vec<&'static str> {
    let mut reason_codes = vec![
        "phase.initial.emitted",
        "fallback.circuit_breaker.evaluated",
    ];

    match lane.expected_phase {
        ExpectedPhase::InitialOnly => reason_codes.push("phase.initial_only"),
        ExpectedPhase::InitialThenRefined => reason_codes.push("phase.refined.emitted"),
        ExpectedPhase::InitialThenMaybeRefined => {
            reason_codes.push("phase.refined_or_failed.emitted");
            reason_codes.push("fallback.phase2.skipped_or_failed");
        }
    }

    if lane.toggles.explain {
        reason_codes.push("explain.payload.complete");
    }
    if lane.toggles.mmr {
        reason_codes.push("diversity.mmr.applied");
    }
    if lane.toggles.negation_queries {
        reason_codes.push("exclusion.negation.applied");
    }
    if lane.toggles.prf {
        reason_codes.push("expansion.prf.applied");
    }
    if lane.toggles.adaptive_fusion {
        reason_codes.push("adaptive.posterior.updated_or_frozen");
    }
    if lane.toggles.calibration != crate::interaction_lanes::CalibratorChoice::Identity {
        reason_codes.push("calibration.transform.applied");
    }
    if lane.toggles.conformal {
        reason_codes.push("conformal.coverage.checked");
    }
    if lane.toggles.feedback {
        reason_codes.push("feedback.boost.applied");
    }

    reason_codes
}

fn lane_metric_keys(lane: &InteractionLane) -> Vec<&'static str> {
    let mut metric_keys = vec![
        "search.phase1.latency_ms",
        "search.initial.result_count",
        "search.kendall_tau",
        "search.fallback.circuit_breaker_state",
    ];

    if lane.expected_phase != ExpectedPhase::InitialOnly {
        metric_keys.push("search.phase2.latency_ms");
        metric_keys.push("search.refined.result_count");
    }
    if lane.expected_phase == ExpectedPhase::InitialThenMaybeRefined {
        metric_keys.push("search.phase2.skipped_count");
    }

    if lane.toggles.explain {
        metric_keys.push("explain.coverage_ratio");
    }
    if lane.toggles.mmr {
        metric_keys.push("mmr.max_pairwise_similarity");
    }
    if lane.toggles.negation_queries {
        metric_keys.push("exclusion.filtered_docs_count");
    }
    if lane.toggles.prf {
        metric_keys.push("prf.centroid_norm");
    }
    if lane.toggles.adaptive_fusion {
        metric_keys.push("adaptive.posterior_delta");
    }
    if lane.toggles.calibration != crate::interaction_lanes::CalibratorChoice::Identity {
        metric_keys.push("calibration.ece");
    }
    if lane.toggles.conformal {
        metric_keys.push("conformal.coverage_rate");
    }
    if lane.toggles.feedback {
        metric_keys.push("feedback.boost_applied_count");
    }

    metric_keys
}

fn lane_log_events(lane: &InteractionLane) -> Vec<&'static str> {
    let mut log_events = vec!["phase.initial.yielded", "fallback.circuit_breaker.state"];

    match lane.expected_phase {
        ExpectedPhase::InitialOnly => log_events.push("phase.initial_only.yielded"),
        ExpectedPhase::InitialThenRefined => log_events.push("phase.refined.yielded"),
        ExpectedPhase::InitialThenMaybeRefined => {
            log_events.push("phase.refined_or_failed.yielded");
            log_events.push("phase.refinement_failed.yielded");
        }
    }

    if lane.toggles.explain {
        log_events.push("explain.hit.generated");
    }
    if lane.toggles.mmr {
        log_events.push("mmr.rerank.applied");
    }
    if lane.toggles.negation_queries {
        log_events.push("exclusion.query.parsed");
    }
    if lane.toggles.prf {
        log_events.push("prf.expansion.applied");
    }
    if lane.toggles.adaptive_fusion {
        log_events.push("adaptive.posterior.updated");
    }
    if lane.toggles.calibration != crate::interaction_lanes::CalibratorChoice::Identity {
        log_events.push("calibration.transform.applied");
    }
    if lane.toggles.conformal {
        log_events.push("conformal.coverage.checked");
    }
    if lane.toggles.feedback {
        log_events.push("feedback.signal.applied");
    }

    log_events
}

fn sorted_unique_strings(mut values: Vec<&'static str>) -> Vec<String> {
    values.sort_unstable();
    values.dedup();
    values.into_iter().map(ToString::to_string).collect()
}

/// Build a deterministic execution template for a single lane.
#[must_use]
pub fn oracle_template_for_lane(lane: &InteractionLane) -> LaneOracleTemplate {
    let mut oracle_ids: Vec<String> = oracles_for_lane(lane)
        .iter()
        .map(|oracle| oracle.id.to_string())
        .collect();
    oracle_ids.sort_unstable();
    oracle_ids.dedup();

    LaneOracleTemplate {
        lane_id: lane.id.to_string(),
        seed: lane.seed,
        expected_phase: lane.expected_phase,
        invariant_groups: lane_groups(lane),
        oracle_ids,
        reason_codes: sorted_unique_strings(lane_reason_codes(lane)),
        metric_keys: sorted_unique_strings(lane_metric_keys(lane)),
        log_events: sorted_unique_strings(lane_log_events(lane)),
    }
}

/// Build deterministic execution templates for the full lane catalog.
#[must_use]
pub fn lane_oracle_templates() -> Vec<LaneOracleTemplate> {
    crate::interaction_lanes::lane_catalog()
        .iter()
        .map(oracle_template_for_lane)
        .collect()
}

// ── Lane-oracle mapping ──────────────────────────────────────────────

/// Summary of which oracles apply to a lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneOracleMapping {
    /// Lane identifier.
    pub lane_id: String,
    /// Oracle IDs that apply to this lane.
    pub oracle_ids: Vec<String>,
    /// Invariant categories covered.
    pub categories: Vec<InvariantCategory>,
}

/// Compute the oracle mapping for all lanes.
#[must_use]
pub fn compute_lane_oracle_mappings() -> Vec<LaneOracleMapping> {
    crate::interaction_lanes::lane_catalog()
        .iter()
        .map(|lane| {
            let applicable = oracles_for_lane(lane);
            let oracle_ids: Vec<String> = applicable.iter().map(|o| o.id.to_string()).collect();
            let mut categories: Vec<InvariantCategory> =
                applicable.iter().map(|o| o.category).collect();
            categories.sort_by_key(|c| format!("{c}"));
            categories.dedup();
            LaneOracleMapping {
                lane_id: lane.id.to_string(),
                oracle_ids,
                categories,
            }
        })
        .collect()
}

// ── Test report ──────────────────────────────────────────────────────

/// Aggregated results from running all oracles for a lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneTestReport {
    /// Lane identifier.
    pub lane_id: String,
    /// Individual oracle verdicts.
    pub verdicts: Vec<OracleVerdict>,
}

impl LaneTestReport {
    /// Create a new empty report for a lane.
    #[must_use]
    pub fn new(lane_id: impl Into<String>) -> Self {
        Self {
            lane_id: lane_id.into(),
            verdicts: Vec::new(),
        }
    }

    /// Add a verdict.
    pub fn add(&mut self, verdict: OracleVerdict) {
        self.verdicts.push(verdict);
    }

    /// True if all non-skipped oracles passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.verdicts.iter().all(|v| v.passed() || v.skipped())
    }

    /// Count of failed oracles.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.verdicts
            .iter()
            .filter(|v| v.outcome == OracleOutcome::Fail)
            .count()
    }

    /// Count of passed oracles.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.verdicts.iter().filter(|v| v.passed()).count()
    }

    /// Count of skipped oracles.
    #[must_use]
    pub fn skip_count(&self) -> usize {
        self.verdicts.iter().filter(|v| v.skipped()).count()
    }
}

impl fmt::Display for LaneTestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Lane [{lane}]: {pass} passed, {fail} failed, {skip} skipped",
            lane = self.lane_id,
            pass = self.pass_count(),
            fail = self.failure_count(),
            skip = self.skip_count(),
        )?;
        for v in &self.verdicts {
            if v.outcome == OracleOutcome::Fail {
                writeln!(f, "  {v}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_lanes::{CalibratorChoice, FeatureToggles, lane_by_id, lane_catalog};

    #[test]
    fn all_oracle_ids_are_unique() {
        let oracles = all_oracles();
        let mut ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();
        let original_len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "duplicate oracle IDs found");
    }

    #[test]
    fn oracle_count() {
        assert_eq!(all_oracles().len(), 30);
    }

    #[test]
    fn baseline_lane_gets_universal_oracles_only() {
        let baseline = lane_by_id("baseline").unwrap();
        let oracles = oracles_for_lane(&baseline);
        let ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();

        // Universal oracles (no feature requirements)
        assert!(ids.contains(&"deterministic_ordering"));
        assert!(ids.contains(&"no_duplicates"));
        assert!(ids.contains(&"monotonic_scores"));
        assert!(ids.contains(&"phase1_always_yields"));
        assert!(ids.contains(&"phase2_refined"));
        assert!(ids.contains(&"refinement_subset"));

        // Feature-specific oracles should NOT apply
        assert!(!ids.contains(&"explain_present"));
        assert!(!ids.contains(&"mmr_diversity_increased"));
        assert!(!ids.contains(&"prf_normalized"));
        assert!(!ids.contains(&"feedback_boost_positive"));
        assert!(!ids.contains(&"conformal_coverage"));
    }

    #[test]
    fn kitchen_sink_gets_most_oracles() {
        let ks = lane_by_id("kitchen_sink").unwrap();
        let oracles = oracles_for_lane(&ks);
        // Kitchen sink has all features on + InitialThenRefined phase,
        // so most oracles should apply (except those requiring MaybeRefined)
        assert!(
            oracles.len() >= 20,
            "kitchen_sink should have >= 20 oracles, got {}",
            oracles.len()
        );
    }

    #[test]
    fn explain_mmr_lane_includes_rank_movement_oracle() {
        let lane = lane_by_id("explain_mmr").unwrap();
        let oracles = oracles_for_lane(&lane);
        let ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();
        assert!(
            ids.contains(&"explain_mmr_rank_movement"),
            "explain_mmr lane should include rank movement oracle"
        );
        assert!(ids.contains(&"explain_present"));
        assert!(ids.contains(&"mmr_diversity_increased"));
    }

    #[test]
    fn breaker_lane_includes_fallback_oracles() {
        let lane = lane_by_id("breaker_adaptive_feedback").unwrap();
        let oracles = oracles_for_lane(&lane);
        let ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();
        assert!(ids.contains(&"breaker_skips_phase2"));
        assert!(ids.contains(&"breaker_phase1_preserved"));
        assert!(ids.contains(&"breaker_no_posterior_update"));
    }

    #[test]
    fn prf_negation_lane_includes_expansion_and_exclusion_oracles() {
        let lane = lane_by_id("prf_negation").unwrap();
        let oracles = oracles_for_lane(&lane);
        let ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();
        assert!(ids.contains(&"prf_normalized"));
        assert!(ids.contains(&"prf_query_class_guard"));
        assert!(ids.contains(&"prf_respects_negation"));
        assert!(ids.contains(&"exclusion_applied"));
    }

    #[test]
    fn calibration_lane_includes_calibration_oracles() {
        let lane = lane_by_id("calibration_conformal").unwrap();
        let oracles = oracles_for_lane(&lane);
        let ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();
        assert!(ids.contains(&"calibrated_range"));
        assert!(ids.contains(&"calibration_monotonic"));
        assert!(ids.contains(&"conformal_k_positive"));
        assert!(ids.contains(&"conformal_coverage"));
    }

    #[test]
    fn oracle_not_applicable_when_feature_missing() {
        let baseline = lane_by_id("baseline").unwrap();
        assert!(!oracle_applicable(&ORACLE_EXPLAIN_PRESENT, &baseline));
        assert!(!oracle_applicable(
            &ORACLE_MMR_DIVERSITY_INCREASED,
            &baseline
        ));
        assert!(!oracle_applicable(&ORACLE_PRF_NORMALIZED, &baseline));
    }

    #[test]
    fn oracle_not_applicable_when_phase_mismatches() {
        let breaker_lane = lane_by_id("breaker_adaptive_feedback").unwrap();
        // This lane has InitialThenMaybeRefined, so phase2_refined should NOT apply
        assert!(!oracle_applicable(&ORACLE_PHASE2_REFINED, &breaker_lane));
        // But phase2_graceful SHOULD apply
        assert!(oracle_applicable(&ORACLE_PHASE2_GRACEFUL, &breaker_lane));
        // refinement_subset is valid whenever a refined phase is emitted.
        assert!(oracle_applicable(&ORACLE_REFINEMENT_SUBSET, &breaker_lane));
    }

    #[test]
    fn every_lane_has_at_least_universal_oracles() {
        for lane in &lane_catalog() {
            let oracles = oracles_for_lane(lane);
            assert!(
                oracles.len() >= 4,
                "lane {} has only {} oracles (expected at least 4 universal)",
                lane.id,
                oracles.len()
            );
        }
    }

    #[test]
    fn compute_mappings_covers_all_lanes() {
        let mappings = compute_lane_oracle_mappings();
        let catalog = lane_catalog();
        assert_eq!(mappings.len(), catalog.len());
        for (mapping, lane) in mappings.iter().zip(catalog.iter()) {
            assert_eq!(mapping.lane_id, lane.id);
            assert!(!mapping.oracle_ids.is_empty());
        }
    }

    #[test]
    fn verdict_display_format() {
        let v = OracleVerdict::fail(
            "monotonic_scores",
            "explain_mmr",
            "scores[2]=0.9 > scores[1]=0.8".into(),
        );
        let s = format!("{v}");
        assert!(s.contains("FAIL"));
        assert!(s.contains("monotonic_scores"));
        assert!(s.contains("explain_mmr"));
        assert!(s.contains("scores[2]"));
    }

    #[test]
    fn verdict_pass_and_skip_helpers() {
        let pass = OracleVerdict::pass("test", "lane");
        assert!(pass.passed());
        assert!(!pass.skipped());

        let skip = OracleVerdict::skip("test", "lane", "not applicable");
        assert!(!skip.passed());
        assert!(skip.skipped());
    }

    #[test]
    fn report_tracks_counts() {
        let mut report = LaneTestReport::new("test_lane");
        report.add(OracleVerdict::pass("o1", "test_lane"));
        report.add(OracleVerdict::pass("o2", "test_lane"));
        report.add(OracleVerdict::fail("o3", "test_lane", "bad".into()));
        report.add(OracleVerdict::skip("o4", "test_lane", "n/a"));

        assert_eq!(report.pass_count(), 2);
        assert_eq!(report.failure_count(), 1);
        assert_eq!(report.skip_count(), 1);
        assert!(!report.all_passed());
    }

    #[test]
    fn report_all_passed_ignores_skips() {
        let mut report = LaneTestReport::new("test_lane");
        report.add(OracleVerdict::pass("o1", "test_lane"));
        report.add(OracleVerdict::skip("o2", "test_lane", "n/a"));
        assert!(report.all_passed());
    }

    #[test]
    fn report_display_shows_failures() {
        let mut report = LaneTestReport::new("test_lane");
        report.add(OracleVerdict::pass("o1", "test_lane"));
        report.add(OracleVerdict::fail(
            "o2",
            "test_lane",
            "score mismatch".into(),
        ));
        let display = format!("{report}");
        assert!(display.contains("1 passed"));
        assert!(display.contains("1 failed"));
        assert!(display.contains("score mismatch"));
    }

    #[test]
    fn required_feature_satisfied_by_toggles() {
        let toggles = FeatureToggles {
            explain: true,
            mmr: false,
            calibration: CalibratorChoice::Platt,
            ..FeatureToggles::default()
        };
        assert!(RequiredFeature::Explain.satisfied_by(&toggles));
        assert!(!RequiredFeature::Mmr.satisfied_by(&toggles));
        assert!(RequiredFeature::NonIdentityCalibration.satisfied_by(&toggles));
        assert!(RequiredFeature::CircuitBreaker.satisfied_by(&toggles)); // default is true
        assert!(!RequiredFeature::Feedback.satisfied_by(&toggles));
    }

    #[test]
    fn template_catalog_covers_all_lanes() {
        let templates = lane_oracle_templates();
        let lanes = lane_catalog();
        assert_eq!(templates.len(), lanes.len());
        for (template, lane) in templates.iter().zip(lanes.iter()) {
            assert_eq!(template.lane_id, lane.id);
            assert_eq!(template.seed, lane.seed);
            assert_eq!(template.expected_phase, lane.expected_phase);
        }
    }

    #[test]
    fn templates_include_core_groups_and_expectations() {
        for lane in lane_catalog() {
            let template = oracle_template_for_lane(&lane);
            assert!(
                template
                    .invariant_groups
                    .contains(&InvariantGroup::Ordering)
            );
            assert!(
                template
                    .invariant_groups
                    .contains(&InvariantGroup::PhaseTransitions)
            );
            assert!(
                template
                    .invariant_groups
                    .contains(&InvariantGroup::ReasonCodes)
            );
            assert!(
                template
                    .invariant_groups
                    .contains(&InvariantGroup::FallbackSemantics)
            );
            assert!(!template.reason_codes.is_empty());
            assert!(!template.metric_keys.is_empty());
            assert!(!template.log_events.is_empty());
        }
    }

    #[test]
    fn templates_are_sorted_and_deterministic() {
        let lane = lane_by_id("kitchen_sink").unwrap();
        let template_a = oracle_template_for_lane(&lane);
        let template_b = oracle_template_for_lane(&lane);
        assert_eq!(template_a, template_b);

        for values in [
            &template_a.oracle_ids,
            &template_a.reason_codes,
            &template_a.metric_keys,
            &template_a.log_events,
        ] {
            for pair in values.windows(2) {
                assert!(pair[0] <= pair[1], "values are not sorted");
            }
        }
    }

    #[test]
    fn breaker_template_asserts_degraded_signals() {
        let lane = lane_by_id("breaker_adaptive_feedback").unwrap();
        let template = oracle_template_for_lane(&lane);
        assert!(
            template
                .reason_codes
                .contains(&"fallback.phase2.skipped_or_failed".to_string())
        );
        assert!(
            template
                .metric_keys
                .contains(&"search.phase2.skipped_count".to_string())
        );
        assert!(
            template
                .log_events
                .contains(&"phase.refined_or_failed.yielded".to_string())
        );
    }

    #[test]
    fn feature_lanes_include_feature_specific_groups() {
        let explain_mmr = lane_by_id("explain_mmr").unwrap();
        let explain_mmr_template = oracle_template_for_lane(&explain_mmr);
        assert!(
            explain_mmr_template
                .invariant_groups
                .contains(&InvariantGroup::ExplanationIntegrity)
        );
        assert!(
            explain_mmr_template
                .invariant_groups
                .contains(&InvariantGroup::DiversityGuarantees)
        );

        let prf_negation = lane_by_id("prf_negation").unwrap();
        let prf_negation_template = oracle_template_for_lane(&prf_negation);
        assert!(
            prf_negation_template
                .invariant_groups
                .contains(&InvariantGroup::ExpansionSemantics)
        );
        assert!(
            prf_negation_template
                .invariant_groups
                .contains(&InvariantGroup::ExclusionSemantics)
        );

        let calibration_conformal = lane_by_id("calibration_conformal").unwrap();
        let calibration_conformal_template = oracle_template_for_lane(&calibration_conformal);
        assert!(
            calibration_conformal_template
                .invariant_groups
                .contains(&InvariantGroup::CalibrationConsistency)
        );
        assert!(
            calibration_conformal_template
                .invariant_groups
                .contains(&InvariantGroup::ConformalCoverage)
        );
    }

    #[test]
    fn invariant_category_display() {
        assert_eq!(InvariantCategory::Ordering.to_string(), "ordering");
        assert_eq!(InvariantCategory::Fallback.to_string(), "fallback");
        assert_eq!(InvariantCategory::Conformal.to_string(), "conformal");
    }

    #[test]
    fn serde_roundtrip_oracle_verdict() {
        let v = OracleVerdict::fail("test_oracle", "test_lane", "some context".into());
        let json = serde_json::to_string(&v).unwrap();
        let back: OracleVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn serde_roundtrip_oracle_descriptor() {
        for oracle in &all_oracles() {
            let json = serde_json::to_string(oracle).expect("serialize oracle");
            let value: serde_json::Value =
                serde_json::from_str(&json).expect("parse serialized oracle json");
            assert_eq!(
                value.get("id").and_then(serde_json::Value::as_str),
                Some(oracle.id),
            );
        }
    }

    #[test]
    fn serde_roundtrip_lane_oracle_mapping() {
        let mappings = compute_lane_oracle_mappings();
        for mapping in &mappings {
            let json = serde_json::to_string(mapping).unwrap();
            let back: LaneOracleMapping = serde_json::from_str(&json).unwrap();
            assert_eq!(mapping, &back);
        }
    }

    // ─── bd-2q5f tests begin ───

    #[test]
    fn invariant_group_display_all_variants() {
        let expected = [
            (InvariantGroup::Ordering, "ordering"),
            (InvariantGroup::PhaseTransitions, "phase_transitions"),
            (InvariantGroup::ReasonCodes, "reason_codes"),
            (InvariantGroup::FallbackSemantics, "fallback_semantics"),
            (
                InvariantGroup::ExplanationIntegrity,
                "explanation_integrity",
            ),
            (
                InvariantGroup::CalibrationConsistency,
                "calibration_consistency",
            ),
            (InvariantGroup::FeedbackConsistency, "feedback_consistency"),
            (InvariantGroup::ExclusionSemantics, "exclusion_semantics"),
            (InvariantGroup::DiversityGuarantees, "diversity_guarantees"),
            (InvariantGroup::ExpansionSemantics, "expansion_semantics"),
            (InvariantGroup::AdaptiveStability, "adaptive_stability"),
            (InvariantGroup::ConformalCoverage, "conformal_coverage"),
        ];
        for (group, label) in expected {
            assert_eq!(group.to_string(), label, "Display mismatch for {group:?}");
        }
    }

    #[test]
    fn invariant_category_display_all_variants() {
        let expected = [
            (InvariantCategory::Ordering, "ordering"),
            (InvariantCategory::Phase, "phase"),
            (InvariantCategory::Explanation, "explanation"),
            (InvariantCategory::Calibration, "calibration"),
            (InvariantCategory::Feedback, "feedback"),
            (InvariantCategory::Fallback, "fallback"),
            (InvariantCategory::Conformal, "conformal"),
            (InvariantCategory::Exclusion, "exclusion"),
            (InvariantCategory::Diversity, "diversity"),
            (InvariantCategory::Expansion, "expansion"),
            (InvariantCategory::Adaptive, "adaptive"),
        ];
        for (cat, label) in expected {
            assert_eq!(cat.to_string(), label, "Display mismatch for {cat:?}");
        }
    }

    #[test]
    fn oracle_outcome_serde_roundtrip() {
        for outcome in [
            OracleOutcome::Pass,
            OracleOutcome::Fail,
            OracleOutcome::Skip,
        ] {
            let json = serde_json::to_string(&outcome).unwrap();
            let back: OracleOutcome = serde_json::from_str(&json).unwrap();
            assert_eq!(outcome, back);
        }
    }

    #[test]
    fn invariant_group_serde_roundtrip() {
        let groups = [
            InvariantGroup::Ordering,
            InvariantGroup::PhaseTransitions,
            InvariantGroup::ReasonCodes,
            InvariantGroup::FallbackSemantics,
            InvariantGroup::ExplanationIntegrity,
            InvariantGroup::CalibrationConsistency,
            InvariantGroup::FeedbackConsistency,
            InvariantGroup::ExclusionSemantics,
            InvariantGroup::DiversityGuarantees,
            InvariantGroup::ExpansionSemantics,
            InvariantGroup::AdaptiveStability,
            InvariantGroup::ConformalCoverage,
        ];
        for group in groups {
            let json = serde_json::to_string(&group).unwrap();
            let back: InvariantGroup = serde_json::from_str(&json).unwrap();
            assert_eq!(group, back);
        }
    }

    #[test]
    fn lane_oracle_template_serde_roundtrip() {
        let lane = lane_by_id("kitchen_sink").unwrap();
        let template = oracle_template_for_lane(&lane);
        let json = serde_json::to_string(&template).unwrap();
        let back: LaneOracleTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(template, back);
    }

    #[test]
    fn required_feature_all_variants_satisfied_by() {
        // All features ON
        let all_on = FeatureToggles {
            explain: true,
            mmr: true,
            negation_queries: true,
            prf: true,
            adaptive_fusion: true,
            conformal: true,
            circuit_breaker: true,
            feedback: true,
            calibration: CalibratorChoice::Platt,
        };
        for feat in [
            RequiredFeature::Explain,
            RequiredFeature::Mmr,
            RequiredFeature::NegationQueries,
            RequiredFeature::Prf,
            RequiredFeature::AdaptiveFusion,
            RequiredFeature::NonIdentityCalibration,
            RequiredFeature::Conformal,
            RequiredFeature::CircuitBreaker,
            RequiredFeature::Feedback,
        ] {
            assert!(
                feat.satisfied_by(&all_on),
                "{feat:?} should be satisfied when all features ON"
            );
        }

        // All features OFF (identity calibration counts as "off")
        let all_off = FeatureToggles {
            explain: false,
            mmr: false,
            negation_queries: false,
            prf: false,
            adaptive_fusion: false,
            conformal: false,
            circuit_breaker: false,
            feedback: false,
            calibration: CalibratorChoice::Identity,
        };
        for feat in [
            RequiredFeature::Explain,
            RequiredFeature::Mmr,
            RequiredFeature::NegationQueries,
            RequiredFeature::Prf,
            RequiredFeature::AdaptiveFusion,
            RequiredFeature::NonIdentityCalibration,
            RequiredFeature::Conformal,
            RequiredFeature::CircuitBreaker,
            RequiredFeature::Feedback,
        ] {
            assert!(
                !feat.satisfied_by(&all_off),
                "{feat:?} should NOT be satisfied when all features OFF"
            );
        }
    }

    #[test]
    fn empty_report_all_passed() {
        let report = LaneTestReport::new("empty_lane");
        assert!(report.all_passed(), "empty report is vacuously passing");
        assert_eq!(report.pass_count(), 0);
        assert_eq!(report.failure_count(), 0);
        assert_eq!(report.skip_count(), 0);
    }

    #[test]
    fn verdict_display_pass_no_context() {
        let v = OracleVerdict::pass("deterministic_ordering", "baseline");
        let s = format!("{v}");
        assert!(s.contains("PASS"));
        assert!(s.contains("deterministic_ordering"));
        assert!(s.contains("baseline"));
        // Pass has empty context, so no dash separator
        assert!(!s.contains('—'));
    }

    #[test]
    fn verdict_display_skip_with_reason() {
        let v = OracleVerdict::skip("explain_present", "baseline", "explain disabled");
        let s = format!("{v}");
        assert!(s.contains("SKIP"));
        assert!(s.contains("explain_present"));
        assert!(s.contains("explain disabled"));
    }

    #[test]
    fn oracle_requirements_default_is_empty() {
        let reqs = OracleRequirements::default();
        assert!(reqs.features.is_empty());
        assert_eq!(reqs.expected_phase, None);
    }

    #[test]
    fn oracle_applicable_multi_feature_requires_all() {
        use crate::interaction_lanes::{CorpusSlice, QuerySlice, RiskLevel};

        // ORACLE_BREAKER_NO_POSTERIOR_UPDATE requires CircuitBreaker + AdaptiveFusion
        let only_breaker = FeatureToggles {
            circuit_breaker: true,
            adaptive_fusion: false,
            ..FeatureToggles::default()
        };
        let lane_breaker_only = InteractionLane {
            id: "test_breaker_only",
            description: "test",
            bead_refs: &[],
            toggles: only_breaker,
            corpus_slices: &[CorpusSlice::Full],
            query_slice: QuerySlice::default(),
            expected_phase: ExpectedPhase::InitialThenMaybeRefined,
            risk: RiskLevel::Low,
            seed: 42,
        };
        // Missing AdaptiveFusion → not applicable
        assert!(!oracle_applicable(
            &ORACLE_BREAKER_NO_POSTERIOR_UPDATE,
            &lane_breaker_only
        ));

        let both_features = FeatureToggles {
            circuit_breaker: true,
            adaptive_fusion: true,
            ..FeatureToggles::default()
        };
        let lane_both = InteractionLane {
            id: "test_both",
            description: "test",
            bead_refs: &[],
            toggles: both_features,
            corpus_slices: &[CorpusSlice::Full],
            query_slice: QuerySlice::default(),
            expected_phase: ExpectedPhase::InitialThenMaybeRefined,
            risk: RiskLevel::Low,
            seed: 42,
        };
        // Both features + correct phase → applicable
        assert!(oracle_applicable(
            &ORACLE_BREAKER_NO_POSTERIOR_UPDATE,
            &lane_both
        ));
    }

    #[test]
    fn report_display_no_failures_shows_zero() {
        let mut report = LaneTestReport::new("clean_lane");
        report.add(OracleVerdict::pass("o1", "clean_lane"));
        report.add(OracleVerdict::pass("o2", "clean_lane"));
        let display = format!("{report}");
        assert!(display.contains("2 passed"));
        assert!(display.contains("0 failed"));
        assert!(display.contains("0 skipped"));
    }

    // ─── bd-2q5f tests end ───

    // ─── bd-1pst tests begin ───

    #[test]
    fn invariant_group_ord_is_declaration_order() {
        // Ord is derived, so variants follow declaration order
        assert!(InvariantGroup::Ordering < InvariantGroup::PhaseTransitions);
        assert!(InvariantGroup::PhaseTransitions < InvariantGroup::ReasonCodes);
        assert!(InvariantGroup::ReasonCodes < InvariantGroup::FallbackSemantics);
        assert!(InvariantGroup::FallbackSemantics < InvariantGroup::ExplanationIntegrity);
        assert!(InvariantGroup::ExplanationIntegrity < InvariantGroup::CalibrationConsistency);
        assert!(InvariantGroup::CalibrationConsistency < InvariantGroup::FeedbackConsistency);
        assert!(InvariantGroup::FeedbackConsistency < InvariantGroup::ExclusionSemantics);
        assert!(InvariantGroup::ExclusionSemantics < InvariantGroup::DiversityGuarantees);
        assert!(InvariantGroup::DiversityGuarantees < InvariantGroup::ExpansionSemantics);
        assert!(InvariantGroup::ExpansionSemantics < InvariantGroup::AdaptiveStability);
        assert!(InvariantGroup::AdaptiveStability < InvariantGroup::ConformalCoverage);
    }

    #[test]
    fn oracle_outcome_hash_usable_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(OracleOutcome::Pass);
        set.insert(OracleOutcome::Fail);
        set.insert(OracleOutcome::Skip);
        set.insert(OracleOutcome::Pass); // duplicate
        assert_eq!(set.len(), 3);
        assert!(set.contains(&OracleOutcome::Pass));
        assert!(set.contains(&OracleOutcome::Fail));
        assert!(set.contains(&OracleOutcome::Skip));
    }

    #[test]
    fn invariant_category_hash_usable_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for cat in [
            InvariantCategory::Ordering,
            InvariantCategory::Phase,
            InvariantCategory::Explanation,
            InvariantCategory::Calibration,
            InvariantCategory::Feedback,
            InvariantCategory::Fallback,
            InvariantCategory::Conformal,
            InvariantCategory::Exclusion,
            InvariantCategory::Diversity,
            InvariantCategory::Expansion,
            InvariantCategory::Adaptive,
        ] {
            set.insert(cat);
        }
        assert_eq!(
            set.len(),
            11,
            "all 11 InvariantCategory variants are distinct"
        );
    }

    #[test]
    fn report_verdicts_preserve_insertion_order() {
        let mut report = LaneTestReport::new("order_test");
        report.add(OracleVerdict::pass("oracle_a", "order_test"));
        report.add(OracleVerdict::fail("oracle_b", "order_test", "err".into()));
        report.add(OracleVerdict::skip("oracle_c", "order_test", "n/a"));
        assert_eq!(report.verdicts.len(), 3);
        assert_eq!(report.verdicts[0].oracle_id, "oracle_a");
        assert_eq!(report.verdicts[1].oracle_id, "oracle_b");
        assert_eq!(report.verdicts[2].oracle_id, "oracle_c");
    }

    #[test]
    fn all_invariant_categories_have_at_least_one_oracle() {
        use std::collections::HashSet;
        let categories_in_catalog: HashSet<InvariantCategory> =
            all_oracles().iter().map(|o| o.category).collect();
        for cat in [
            InvariantCategory::Ordering,
            InvariantCategory::Phase,
            InvariantCategory::Explanation,
            InvariantCategory::Calibration,
            InvariantCategory::Feedback,
            InvariantCategory::Fallback,
            InvariantCategory::Conformal,
            InvariantCategory::Exclusion,
            InvariantCategory::Diversity,
            InvariantCategory::Expansion,
            InvariantCategory::Adaptive,
        ] {
            assert!(
                categories_in_catalog.contains(&cat),
                "no oracle found for category {cat:?}"
            );
        }
    }

    #[test]
    fn initial_only_lane_excludes_phase2_oracles() {
        use crate::interaction_lanes::{CorpusSlice, QuerySlice, RiskLevel};

        let lane = InteractionLane {
            id: "test_initial_only",
            description: "initial-only phase test",
            bead_refs: &[],
            toggles: FeatureToggles::default(),
            corpus_slices: &[CorpusSlice::Full],
            query_slice: QuerySlice::default(),
            expected_phase: ExpectedPhase::InitialOnly,
            risk: RiskLevel::Low,
            seed: 99,
        };
        let oracles = oracles_for_lane(&lane);
        let ids: Vec<&str> = oracles.iter().map(|o| o.id).collect();
        // InitialOnly should NOT get phase2_refined (requires InitialThenRefined)
        assert!(!ids.contains(&"phase2_refined"));
        // InitialOnly should NOT get phase2_graceful (requires InitialThenMaybeRefined)
        assert!(!ids.contains(&"phase2_graceful"));
        // Universal oracles still apply
        assert!(ids.contains(&"deterministic_ordering"));
        assert!(ids.contains(&"phase1_always_yields"));
    }

    #[test]
    fn oracle_explain_exclusion_absent_requires_both_features() {
        use crate::interaction_lanes::{CorpusSlice, QuerySlice, RiskLevel};

        // Only explain — exclusion oracle should NOT apply
        let explain_only = InteractionLane {
            id: "test_explain_only",
            description: "test",
            bead_refs: &[],
            toggles: FeatureToggles {
                explain: true,
                negation_queries: false,
                ..FeatureToggles::default()
            },
            corpus_slices: &[CorpusSlice::Full],
            query_slice: QuerySlice::default(),
            expected_phase: ExpectedPhase::InitialThenRefined,
            risk: RiskLevel::Low,
            seed: 50,
        };
        assert!(!oracle_applicable(
            &ORACLE_EXPLAIN_EXCLUSION_ABSENT,
            &explain_only
        ));

        // Only negation — exclusion oracle should NOT apply
        let negation_only = InteractionLane {
            id: "test_negation_only",
            description: "test",
            bead_refs: &[],
            toggles: FeatureToggles {
                explain: false,
                negation_queries: true,
                ..FeatureToggles::default()
            },
            corpus_slices: &[CorpusSlice::Full],
            query_slice: QuerySlice::default(),
            expected_phase: ExpectedPhase::InitialThenRefined,
            risk: RiskLevel::Low,
            seed: 51,
        };
        assert!(!oracle_applicable(
            &ORACLE_EXPLAIN_EXCLUSION_ABSENT,
            &negation_only
        ));

        // Both features — oracle SHOULD apply
        let both = InteractionLane {
            id: "test_both",
            description: "test",
            bead_refs: &[],
            toggles: FeatureToggles {
                explain: true,
                negation_queries: true,
                ..FeatureToggles::default()
            },
            corpus_slices: &[CorpusSlice::Full],
            query_slice: QuerySlice::default(),
            expected_phase: ExpectedPhase::InitialThenRefined,
            risk: RiskLevel::Low,
            seed: 52,
        };
        assert!(oracle_applicable(&ORACLE_EXPLAIN_EXCLUSION_ABSENT, &both));
    }

    #[test]
    fn required_feature_serde_roundtrip() {
        for feat in [
            RequiredFeature::Explain,
            RequiredFeature::Mmr,
            RequiredFeature::NegationQueries,
            RequiredFeature::Prf,
            RequiredFeature::AdaptiveFusion,
            RequiredFeature::NonIdentityCalibration,
            RequiredFeature::Conformal,
            RequiredFeature::CircuitBreaker,
            RequiredFeature::Feedback,
        ] {
            let json = serde_json::to_string(&feat).unwrap();
            let back: RequiredFeature = serde_json::from_str(&json).unwrap();
            assert_eq!(
                feat, back,
                "RequiredFeature serde roundtrip failed for {feat:?}"
            );
        }
    }

    #[test]
    fn lane_mapping_categories_are_unique() {
        for mapping in compute_lane_oracle_mappings() {
            let original_len = mapping.categories.len();
            let mut deduped = mapping.categories.clone();
            deduped.dedup();
            assert_eq!(
                deduped.len(),
                original_len,
                "lane {} has duplicate categories in mapping",
                mapping.lane_id
            );
        }
    }

    #[test]
    fn oracle_descriptor_debug_format() {
        let debug = format!("{ORACLE_DETERMINISTIC_ORDERING:?}");
        assert!(debug.contains("deterministic_ordering"));
        assert!(debug.contains("Ordering"));
        let debug_prf = format!("{ORACLE_PRF_NORMALIZED:?}");
        assert!(debug_prf.contains("prf_normalized"));
        assert!(debug_prf.contains("Expansion"));
    }

    // ─── bd-1pst tests end ───
}
