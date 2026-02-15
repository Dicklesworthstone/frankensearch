//! Query latency optimization contracts for the retrieval/fusion/explanation path.
//!
//! Provides:
//! - Phase-wise latency decomposition with budgets and actuals
//! - Prioritized optimization levers (opportunity matrix for query path)
//! - Correctness-preserving verification protocol for behavioral equivalence

use serde::{Deserialize, Serialize};

use crate::profiling::{OpportunityCandidate, OpportunityMatrix};

// --- Schema Version ----------------------------------------------------------

/// Schema version for query latency optimization contracts.
pub const QUERY_LATENCY_OPT_SCHEMA_VERSION: &str = "fsfs-query-latency-opt-v1";

// --- Phase-wise Latency Decomposition ----------------------------------------

/// Canonical query-path phases in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryPhase {
    /// Query text normalization and canonicalization (NFC, markdown strip).
    Canonicalize,
    /// Intent classification and budget assignment.
    Classify,
    /// Fast-tier embedding (potion-128M, ~0.57ms).
    FastEmbed,
    /// BM25 lexical retrieval via Tantivy.
    LexicalRetrieve,
    /// Fast-tier vector search (FSVI scan).
    FastVectorSearch,
    /// RRF rank fusion across lexical + semantic.
    Fuse,
    /// Quality-tier embedding (MiniLM-L6-v2, ~128ms).
    QualityEmbed,
    /// Quality-tier vector lookup.
    QualityVectorSearch,
    /// Two-tier score blending.
    Blend,
    /// Cross-encoder reranking (optional).
    Rerank,
    /// Explanation/evidence payload generation.
    Explain,
    /// Output serialization (JSON/TOON).
    Serialize,
}

impl QueryPhase {
    /// All phases in canonical execution order.
    pub const ALL: &[Self] = &[
        Self::Canonicalize,
        Self::Classify,
        Self::FastEmbed,
        Self::LexicalRetrieve,
        Self::FastVectorSearch,
        Self::Fuse,
        Self::QualityEmbed,
        Self::QualityVectorSearch,
        Self::Blend,
        Self::Rerank,
        Self::Explain,
        Self::Serialize,
    ];

    /// Whether this phase is on the initial (fast) path.
    #[must_use]
    pub const fn is_initial_path(self) -> bool {
        matches!(
            self,
            Self::Canonicalize
                | Self::Classify
                | Self::FastEmbed
                | Self::LexicalRetrieve
                | Self::FastVectorSearch
                | Self::Fuse
        )
    }

    /// Whether this phase is on the refinement (quality) path.
    #[must_use]
    pub const fn is_refinement_path(self) -> bool {
        matches!(
            self,
            Self::QualityEmbed | Self::QualityVectorSearch | Self::Blend | Self::Rerank
        )
    }

    /// Default budget in microseconds for this phase.
    ///
    /// Based on empirical measurements and SLO targets:
    /// - Initial path total: ~15ms
    /// - Refinement path total: ~150ms (dominated by quality embedding)
    #[must_use]
    pub const fn default_budget_us(self) -> u64 {
        match self {
            Self::Canonicalize => 200,                               // 0.2ms
            Self::Classify => 100,                                   // 0.1ms
            Self::FastEmbed => 800, // 0.8ms (potion ~0.57ms + overhead)
            Self::LexicalRetrieve | Self::FastVectorSearch => 5_000, // 5ms each
            Self::Fuse | Self::Blend | Self::Serialize => 500, // 0.5ms each
            Self::QualityEmbed => 130_000, // 130ms (MiniLM-L6-v2)
            Self::QualityVectorSearch => 2_000, // 2ms
            Self::Rerank => 25_000, // 25ms (cross-encoder)
            Self::Explain => 1_000, // 1ms
        }
    }
}

/// One phase's timing observation within a query execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseObservation {
    pub phase: QueryPhase,
    /// Wall-clock time in microseconds.
    pub actual_us: u64,
    /// Budget for this phase in microseconds.
    pub budget_us: u64,
    /// Whether this phase was skipped (e.g., rerank disabled).
    pub skipped: bool,
}

impl PhaseObservation {
    /// Whether actual exceeded budget.
    #[must_use]
    pub const fn over_budget(&self) -> bool {
        !self.skipped && self.actual_us > self.budget_us
    }

    /// Overshoot in microseconds (0 if under budget or skipped).
    #[must_use]
    pub const fn overshoot_us(&self) -> u64 {
        if self.skipped || self.actual_us <= self.budget_us {
            0
        } else {
            self.actual_us - self.budget_us
        }
    }

    /// Budget utilization as a fraction (0.0 to unbounded).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.skipped || self.budget_us == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.actual_us as f64 / self.budget_us as f64;
            ratio
        }
    }
}

/// Complete latency decomposition for one query execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyDecomposition {
    pub schema_version: String,
    /// Observations for each phase (ordered by execution sequence).
    pub phases: Vec<PhaseObservation>,
    /// Total wall-clock time in microseconds.
    pub total_us: u64,
    /// Sum of all phase budgets in microseconds.
    pub total_budget_us: u64,
    /// Number of result documents returned.
    pub result_count: usize,
    /// Index size at query time (number of documents).
    pub index_size: usize,
}

impl LatencyDecomposition {
    /// Build a decomposition from phase observations.
    #[must_use]
    pub fn new(phases: Vec<PhaseObservation>, result_count: usize, index_size: usize) -> Self {
        let total_us = phases
            .iter()
            .filter(|p| !p.skipped)
            .map(|p| p.actual_us)
            .sum();
        let total_budget_us = phases
            .iter()
            .filter(|p| !p.skipped)
            .map(|p| p.budget_us)
            .sum();
        Self {
            schema_version: QUERY_LATENCY_OPT_SCHEMA_VERSION.to_owned(),
            phases,
            total_us,
            total_budget_us,
            result_count,
            index_size,
        }
    }

    /// Phases that exceeded their budget, sorted by overshoot descending.
    #[must_use]
    pub fn over_budget_phases(&self) -> Vec<&PhaseObservation> {
        let mut over: Vec<&PhaseObservation> =
            self.phases.iter().filter(|p| p.over_budget()).collect();
        over.sort_by_key(|o| std::cmp::Reverse(o.overshoot_us()));
        over
    }

    /// Total initial-path time in microseconds.
    #[must_use]
    pub fn initial_path_us(&self) -> u64 {
        self.phases
            .iter()
            .filter(|p| p.phase.is_initial_path() && !p.skipped)
            .map(|p| p.actual_us)
            .sum()
    }

    /// Total refinement-path time in microseconds.
    #[must_use]
    pub fn refinement_path_us(&self) -> u64 {
        self.phases
            .iter()
            .filter(|p| p.phase.is_refinement_path() && !p.skipped)
            .map(|p| p.actual_us)
            .sum()
    }

    /// Whether the overall query met its total budget.
    #[must_use]
    pub const fn met_budget(&self) -> bool {
        self.total_us <= self.total_budget_us
    }

    /// Reason code summarizing the decomposition verdict.
    #[must_use]
    pub fn verdict_reason_code(&self) -> &'static str {
        if self.met_budget() {
            "query.latency.on_budget"
        } else if self.over_budget_phases().len() == 1 {
            "query.latency.single_phase_over_budget"
        } else {
            "query.latency.multiple_phases_over_budget"
        }
    }
}

// --- Optimization Lever Catalog -----------------------------------------------

/// Optimization lever targeting a specific query-path phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryOptimizationLever {
    /// Stable lever identifier.
    pub id: String,
    /// Target phase.
    pub phase: QueryPhase,
    /// Human-readable description of the optimization.
    pub description: String,
    /// Mechanism category.
    pub mechanism: OptimizationMechanism,
    /// Whether correctness proof is trivial (pure refactoring) vs requires
    /// behavioral equivalence testing.
    pub correctness_proof: CorrectnessProofKind,
}

/// Category of optimization mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationMechanism {
    /// Reduce heap allocations (fewer Vec/String/HashMap creates).
    AllocationReduction,
    /// Reuse buffers across iterations/queries.
    BufferReuse,
    /// Improve cache locality (data layout, access patterns).
    CacheLocality,
    /// Use cheaper algorithm with same output.
    AlgorithmReplacement,
    /// Parallelize sequential work.
    Parallelism,
    /// Reduce data movement (fewer clones, borrows instead).
    DataMovement,
    /// Pre-compute or cache intermediate results.
    Precomputation,
}

/// How to prove an optimization preserves correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectnessProofKind {
    /// Output is bit-identical before and after.
    BitIdentical,
    /// Output ordering and scores match within epsilon.
    NumericallyEquivalent,
    /// Ranking order preserved but scores may differ slightly.
    RankPreserving,
}

/// Build the canonical query-path optimization opportunity matrix.
///
/// Candidates are populated from empirical profiling of the retrieval/fusion path.
/// ICE scores use the standard (impact * confidence * 1000) / effort formula.
#[must_use]
pub fn query_path_opportunity_matrix() -> OpportunityMatrix {
    OpportunityMatrix::new(vec![
        // --- Fuse phase (RRF) ---
        OpportunityCandidate {
            id: "fuse.string_clone_reduction".into(),
            summary: "Reduce String cloning in RRF HashMap entry/merge loop".into(),
            impact: 80,
            confidence: 95,
            effort: 15,
        },
        OpportunityCandidate {
            id: "fuse.hashmap_capacity".into(),
            summary: "Adjust HashMap pre-allocation for typical overlap ratios".into(),
            impact: 35,
            confidence: 75,
            effort: 5,
        },
        // --- Blend phase ---
        OpportunityCandidate {
            id: "blend.string_clone_reduction".into(),
            summary: "Reduce String cloning in blend HashMap entry loop".into(),
            impact: 75,
            confidence: 95,
            effort: 15,
        },
        OpportunityCandidate {
            id: "blend.rank_map_cache".into(),
            summary: "Cache rank map from Phase 1 to avoid rebuild in Phase 2".into(),
            impact: 40,
            confidence: 85,
            effort: 20,
        },
        // --- Vector search ---
        OpportunityCandidate {
            id: "vector_search.scratch_buffer_reuse".into(),
            summary: "Reuse scratch Vec across sequential scan iterations".into(),
            impact: 70,
            confidence: 90,
            effort: 10,
        },
        OpportunityCandidate {
            id: "vector_search.parallel_threshold_tuning".into(),
            summary: "Lower PARALLEL_THRESHOLD from 10k based on benchmark data".into(),
            impact: 50,
            confidence: 70,
            effort: 30,
        },
        // --- Blend phase (Kendall tau) ---
        OpportunityCandidate {
            id: "blend.kendall_tau_approximation".into(),
            summary: "Replace O(n^2) Kendall tau with O(n log n) merge-sort variant".into(),
            impact: 45,
            confidence: 80,
            effort: 40,
        },
        // --- Explain/Serialize ---
        OpportunityCandidate {
            id: "serialize.preallocate_json_buffer".into(),
            summary: "Pre-size serde_json buffer from typical payload size".into(),
            impact: 20,
            confidence: 70,
            effort: 10,
        },
    ])
}

/// Build the full lever catalog with mechanism and correctness metadata.
#[must_use]
pub fn query_path_lever_catalog() -> Vec<QueryOptimizationLever> {
    vec![
        QueryOptimizationLever {
            id: "fuse.string_clone_reduction".into(),
            phase: QueryPhase::Fuse,
            description: "Replace doc_id.clone() in RRF entry/merge with borrowed key \
                           or single allocation on first insert."
                .into(),
            mechanism: OptimizationMechanism::DataMovement,
            correctness_proof: CorrectnessProofKind::BitIdentical,
        },
        QueryOptimizationLever {
            id: "fuse.hashmap_capacity".into(),
            phase: QueryPhase::Fuse,
            description: "Multiply capacity by 0.75 for typical 50% overlap, \
                           reducing rehash probability."
                .into(),
            mechanism: OptimizationMechanism::AllocationReduction,
            correctness_proof: CorrectnessProofKind::BitIdentical,
        },
        QueryOptimizationLever {
            id: "blend.string_clone_reduction".into(),
            phase: QueryPhase::Blend,
            description: "Reduce String cloning in blend HashMap by borrowing doc_id \
                           from input slices."
                .into(),
            mechanism: OptimizationMechanism::DataMovement,
            correctness_proof: CorrectnessProofKind::BitIdentical,
        },
        QueryOptimizationLever {
            id: "blend.rank_map_cache".into(),
            phase: QueryPhase::Blend,
            description: "Pass Phase 1 rank map into Phase 2 instead of rebuilding.".into(),
            mechanism: OptimizationMechanism::Precomputation,
            correctness_proof: CorrectnessProofKind::BitIdentical,
        },
        QueryOptimizationLever {
            id: "vector_search.scratch_buffer_reuse".into(),
            phase: QueryPhase::FastVectorSearch,
            description: "Allocate scratch Vec once outside the scan loop, \
                           reuse across iterations with clear()."
                .into(),
            mechanism: OptimizationMechanism::BufferReuse,
            correctness_proof: CorrectnessProofKind::NumericallyEquivalent,
        },
        QueryOptimizationLever {
            id: "vector_search.parallel_threshold_tuning".into(),
            phase: QueryPhase::FastVectorSearch,
            description: "Benchmark and potentially lower PARALLEL_THRESHOLD \
                           from 10_000 to 4_000-6_000."
                .into(),
            mechanism: OptimizationMechanism::Parallelism,
            correctness_proof: CorrectnessProofKind::RankPreserving,
        },
        QueryOptimizationLever {
            id: "blend.kendall_tau_approximation".into(),
            phase: QueryPhase::Blend,
            description: "Replace O(n^2) pair-comparison Kendall tau with \
                           O(n log n) merge-sort-based algorithm."
                .into(),
            mechanism: OptimizationMechanism::AlgorithmReplacement,
            correctness_proof: CorrectnessProofKind::NumericallyEquivalent,
        },
        QueryOptimizationLever {
            id: "serialize.preallocate_json_buffer".into(),
            phase: QueryPhase::Serialize,
            description: "Use serde_json::to_writer with pre-sized Vec buffer \
                           (est. 2KB per result * count)."
                .into(),
            mechanism: OptimizationMechanism::AllocationReduction,
            correctness_proof: CorrectnessProofKind::BitIdentical,
        },
    ]
}

// --- Correctness Verification Protocol ----------------------------------------

/// A correctness assertion for one optimization step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectnessAssertion {
    /// Lever this assertion verifies.
    pub lever_id: String,
    /// Kind of proof required.
    pub proof_kind: CorrectnessProofKind,
    /// Test query corpus identifiers used for verification.
    pub test_corpus_ids: Vec<String>,
    /// Description of what is being asserted.
    pub assertion: String,
    /// Whether the assertion passed.
    pub passed: bool,
    /// Machine-readable reason code.
    pub reason_code: String,
}

/// Verification result for a batch of correctness assertions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationResult {
    pub schema_version: String,
    /// Lever being verified.
    pub lever_id: String,
    /// All assertions checked.
    pub assertions: Vec<CorrectnessAssertion>,
    /// Overall pass/fail.
    pub passed: bool,
    /// Machine-readable reason code.
    pub reason_code: String,
}

impl VerificationResult {
    /// Build result from assertions.
    #[must_use]
    pub fn from_assertions(lever_id: &str, assertions: Vec<CorrectnessAssertion>) -> Self {
        let passed = assertions.iter().all(|a| a.passed);
        let reason_code = if passed {
            "opt.verify.passed"
        } else {
            "opt.verify.failed"
        };
        Self {
            schema_version: QUERY_LATENCY_OPT_SCHEMA_VERSION.to_owned(),
            lever_id: lever_id.to_owned(),
            assertions,
            passed,
            reason_code: reason_code.to_owned(),
        }
    }

    /// Count of failed assertions.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.assertions.iter().filter(|a| !a.passed).count()
    }
}

/// Protocol for verifying behavioral equivalence before and after an optimization.
///
/// Each lever produces a [`VerificationResult`] proving:
/// - `BitIdentical`: output bytes are identical for all test queries
/// - `NumericallyEquivalent`: scores differ by less than epsilon
/// - `RankPreserving`: result ordering is identical
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationProtocol {
    pub schema_version: String,
    /// Required test corpus IDs.
    pub required_corpus_ids: Vec<String>,
    /// Epsilon for numerical equivalence checks.
    pub score_epsilon_str: String,
    /// Lever IDs that must pass before merging.
    pub required_lever_ids: Vec<String>,
}

impl VerificationProtocol {
    /// Default protocol matching the golden test fixture corpus.
    #[must_use]
    pub fn default_protocol() -> Self {
        Self {
            schema_version: QUERY_LATENCY_OPT_SCHEMA_VERSION.to_owned(),
            required_corpus_ids: vec![
                "golden_100".into(),
                "adversarial_unicode".into(),
                "empty_query".into(),
                "identifier_query".into(),
                "natural_language_query".into(),
            ],
            score_epsilon_str: "1e-9".to_owned(),
            required_lever_ids: query_path_lever_catalog()
                .iter()
                .map(|l| l.id.clone())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Phase metadata tests ---

    #[test]
    fn all_phases_have_positive_budgets() {
        for phase in QueryPhase::ALL {
            assert!(
                phase.default_budget_us() > 0,
                "phase {phase:?} has zero budget"
            );
        }
    }

    #[test]
    fn initial_and_refinement_paths_are_disjoint() {
        for phase in QueryPhase::ALL {
            assert!(
                !(phase.is_initial_path() && phase.is_refinement_path()),
                "phase {phase:?} is on both paths"
            );
        }
    }

    #[test]
    fn initial_path_budget_under_target() {
        let initial_budget_us: u64 = QueryPhase::ALL
            .iter()
            .filter(|p| p.is_initial_path())
            .map(|p| p.default_budget_us())
            .sum();
        // Initial path target: 15ms = 15_000us
        assert!(
            initial_budget_us <= 15_000,
            "initial path budget {initial_budget_us}us exceeds 15ms target"
        );
    }

    #[test]
    fn phase_count_is_twelve() {
        assert_eq!(QueryPhase::ALL.len(), 12);
    }

    // --- Decomposition tests ---

    fn sample_decomposition() -> LatencyDecomposition {
        LatencyDecomposition::new(
            vec![
                PhaseObservation {
                    phase: QueryPhase::Canonicalize,
                    actual_us: 150,
                    budget_us: 200,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::Classify,
                    actual_us: 80,
                    budget_us: 100,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::FastEmbed,
                    actual_us: 600,
                    budget_us: 800,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::LexicalRetrieve,
                    actual_us: 4_500,
                    budget_us: 5_000,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::FastVectorSearch,
                    actual_us: 7_200,
                    budget_us: 5_000,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::Fuse,
                    actual_us: 400,
                    budget_us: 500,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::QualityEmbed,
                    actual_us: 0,
                    budget_us: 130_000,
                    skipped: true,
                },
                PhaseObservation {
                    phase: QueryPhase::Rerank,
                    actual_us: 0,
                    budget_us: 25_000,
                    skipped: true,
                },
            ],
            10,
            5_000,
        )
    }

    #[test]
    fn decomposition_total_excludes_skipped() {
        let d = sample_decomposition();
        // 150 + 80 + 600 + 4500 + 7200 + 400 = 12930
        assert_eq!(d.total_us, 12_930);
    }

    #[test]
    fn over_budget_phases_detected() {
        let d = sample_decomposition();
        let over = d.over_budget_phases();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].phase, QueryPhase::FastVectorSearch);
        assert_eq!(over[0].overshoot_us(), 2_200);
    }

    #[test]
    fn initial_path_time_sums_correctly() {
        let d = sample_decomposition();
        // Canonicalize + Classify + FastEmbed + LexicalRetrieve + FastVectorSearch + Fuse
        assert_eq!(d.initial_path_us(), 12_930);
    }

    #[test]
    fn refinement_path_time_excludes_skipped() {
        let d = sample_decomposition();
        assert_eq!(d.refinement_path_us(), 0);
    }

    #[test]
    fn budget_verdict_reason_codes() {
        let on_budget = LatencyDecomposition::new(
            vec![PhaseObservation {
                phase: QueryPhase::Fuse,
                actual_us: 300,
                budget_us: 500,
                skipped: false,
            }],
            5,
            100,
        );
        assert_eq!(on_budget.verdict_reason_code(), "query.latency.on_budget");
        assert!(on_budget.met_budget());

        let single_over = LatencyDecomposition::new(
            vec![
                PhaseObservation {
                    phase: QueryPhase::Fuse,
                    actual_us: 300,
                    budget_us: 500,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::FastVectorSearch,
                    actual_us: 8_000,
                    budget_us: 5_000,
                    skipped: false,
                },
            ],
            5,
            100,
        );
        assert_eq!(
            single_over.verdict_reason_code(),
            "query.latency.single_phase_over_budget"
        );
    }

    #[test]
    fn phase_observation_utilization() {
        let obs = PhaseObservation {
            phase: QueryPhase::Fuse,
            actual_us: 250,
            budget_us: 500,
            skipped: false,
        };
        assert!((obs.utilization() - 0.5).abs() < 1e-9);
        assert!(!obs.over_budget());

        let skipped = PhaseObservation {
            phase: QueryPhase::Rerank,
            actual_us: 0,
            budget_us: 25_000,
            skipped: true,
        };
        assert!((skipped.utilization() - 0.0).abs() < 1e-9);
        assert!(!skipped.over_budget());
    }

    // --- Opportunity matrix tests ---

    #[test]
    fn query_path_matrix_has_all_levers() {
        let matrix = query_path_opportunity_matrix();
        assert!(
            matrix.candidates.len() >= 8,
            "expected at least 8 candidates, got {}",
            matrix.candidates.len()
        );
    }

    #[test]
    fn opportunity_matrix_ranking_is_deterministic() {
        let matrix = query_path_opportunity_matrix();
        let ranked1 = matrix.ranked();
        let ranked2 = matrix.ranked();
        assert_eq!(ranked1.len(), ranked2.len());
        for (a, b) in ranked1.iter().zip(ranked2.iter()) {
            assert_eq!(a.candidate.id, b.candidate.id);
            assert_eq!(a.rank, b.rank);
        }
    }

    #[test]
    fn top_lever_has_highest_ice_score() {
        let matrix = query_path_opportunity_matrix();
        let ranked = matrix.ranked();
        assert!(!ranked.is_empty());
        for window in ranked.windows(2) {
            assert!(
                window[0].ice_score_per_mille >= window[1].ice_score_per_mille,
                "ranking not descending: {} (ICE={}) before {} (ICE={})",
                window[0].candidate.id,
                window[0].ice_score_per_mille,
                window[1].candidate.id,
                window[1].ice_score_per_mille,
            );
        }
    }

    #[test]
    fn all_candidates_have_valid_ice_scores() {
        let matrix = query_path_opportunity_matrix();
        for candidate in &matrix.candidates {
            assert!(candidate.impact <= 100);
            assert!(candidate.confidence <= 100);
            assert!(candidate.effort > 0 && candidate.effort <= 100);
            assert!(candidate.ice_score_per_mille() > 0);
        }
    }

    // --- Lever catalog tests ---

    #[test]
    fn lever_catalog_covers_matrix_candidates() {
        let matrix = query_path_opportunity_matrix();
        let catalog = query_path_lever_catalog();
        let catalog_ids: Vec<&str> = catalog.iter().map(|l| l.id.as_str()).collect();
        for candidate in &matrix.candidates {
            assert!(
                catalog_ids.contains(&candidate.id.as_str()),
                "candidate '{}' has no matching lever in catalog",
                candidate.id
            );
        }
    }

    #[test]
    fn lever_catalog_ids_are_unique() {
        let catalog = query_path_lever_catalog();
        let mut ids: Vec<&str> = catalog.iter().map(|l| l.id.as_str()).collect();
        let original_len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "duplicate lever IDs found");
    }

    #[test]
    fn every_lever_targets_a_query_phase() {
        let catalog = query_path_lever_catalog();
        for lever in &catalog {
            assert!(
                QueryPhase::ALL.contains(&lever.phase),
                "lever '{}' targets unknown phase {:?}",
                lever.id,
                lever.phase
            );
        }
    }

    // --- Verification protocol tests ---

    #[test]
    fn default_protocol_covers_all_levers() {
        let protocol = VerificationProtocol::default_protocol();
        let catalog = query_path_lever_catalog();
        for lever in &catalog {
            assert!(
                protocol.required_lever_ids.contains(&lever.id),
                "lever '{}' not covered by default protocol",
                lever.id
            );
        }
    }

    #[test]
    fn verification_result_passes_when_all_assertions_pass() {
        let result = VerificationResult::from_assertions(
            "fuse.string_clone_reduction",
            vec![
                CorrectnessAssertion {
                    lever_id: "fuse.string_clone_reduction".into(),
                    proof_kind: CorrectnessProofKind::BitIdentical,
                    test_corpus_ids: vec!["golden_100".into()],
                    assertion: "Output bytes identical for golden corpus".into(),
                    passed: true,
                    reason_code: "opt.assert.bit_identical.passed".into(),
                },
                CorrectnessAssertion {
                    lever_id: "fuse.string_clone_reduction".into(),
                    proof_kind: CorrectnessProofKind::BitIdentical,
                    test_corpus_ids: vec!["adversarial_unicode".into()],
                    assertion: "Output bytes identical for adversarial corpus".into(),
                    passed: true,
                    reason_code: "opt.assert.bit_identical.passed".into(),
                },
            ],
        );
        assert!(result.passed);
        assert_eq!(result.failure_count(), 0);
        assert_eq!(result.reason_code, "opt.verify.passed");
    }

    #[test]
    fn verification_result_fails_when_any_assertion_fails() {
        let result = VerificationResult::from_assertions(
            "vector_search.parallel_threshold_tuning",
            vec![
                CorrectnessAssertion {
                    lever_id: "vector_search.parallel_threshold_tuning".into(),
                    proof_kind: CorrectnessProofKind::RankPreserving,
                    test_corpus_ids: vec!["golden_100".into()],
                    assertion: "Ranking order preserved".into(),
                    passed: true,
                    reason_code: "opt.assert.rank_preserving.passed".into(),
                },
                CorrectnessAssertion {
                    lever_id: "vector_search.parallel_threshold_tuning".into(),
                    proof_kind: CorrectnessProofKind::RankPreserving,
                    test_corpus_ids: vec!["adversarial_unicode".into()],
                    assertion: "Ranking order preserved".into(),
                    passed: false,
                    reason_code: "opt.assert.rank_preserving.rank_swap_at_position_3".into(),
                },
            ],
        );
        assert!(!result.passed);
        assert_eq!(result.failure_count(), 1);
        assert_eq!(result.reason_code, "opt.verify.failed");
    }

    #[test]
    fn protocol_has_required_corpus_ids() {
        let protocol = VerificationProtocol::default_protocol();
        assert!(protocol.required_corpus_ids.len() >= 3);
        assert!(
            protocol
                .required_corpus_ids
                .contains(&"golden_100".to_string())
        );
    }

    // --- Serde roundtrip tests ---

    #[test]
    fn decomposition_serde_roundtrip() {
        let d = sample_decomposition();
        let json = serde_json::to_string(&d).unwrap();
        let decoded: LatencyDecomposition = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_us, d.total_us);
        assert_eq!(decoded.phases.len(), d.phases.len());
        assert_eq!(decoded.result_count, d.result_count);
    }

    #[test]
    fn lever_catalog_serde_roundtrip() {
        let catalog = query_path_lever_catalog();
        let json = serde_json::to_string(&catalog).unwrap();
        let decoded: Vec<QueryOptimizationLever> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.len(), catalog.len());
        for (original, decoded) in catalog.iter().zip(decoded.iter()) {
            assert_eq!(original.id, decoded.id);
            assert_eq!(original.phase, decoded.phase);
        }
    }

    #[test]
    fn phase_serde_roundtrip() {
        for phase in QueryPhase::ALL {
            let json = serde_json::to_string(phase).unwrap();
            let decoded: QueryPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(*phase, decoded);
        }
    }

    // --- PhaseObservation edge cases ---

    #[test]
    fn observation_zero_budget_utilization_is_zero() {
        let obs = PhaseObservation {
            phase: QueryPhase::Fuse,
            actual_us: 500,
            budget_us: 0,
            skipped: false,
        };
        assert!((obs.utilization() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn observation_skipped_not_over_budget_despite_high_actual() {
        let obs = PhaseObservation {
            phase: QueryPhase::QualityEmbed,
            actual_us: 999_999,
            budget_us: 100,
            skipped: true,
        };
        assert!(!obs.over_budget());
        assert_eq!(obs.overshoot_us(), 0);
        assert!((obs.utilization() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn observation_exact_budget_is_not_over() {
        let obs = PhaseObservation {
            phase: QueryPhase::Fuse,
            actual_us: 500,
            budget_us: 500,
            skipped: false,
        };
        assert!(!obs.over_budget());
        assert_eq!(obs.overshoot_us(), 0);
        assert!((obs.utilization() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn observation_one_over_budget() {
        let obs = PhaseObservation {
            phase: QueryPhase::Fuse,
            actual_us: 501,
            budget_us: 500,
            skipped: false,
        };
        assert!(obs.over_budget());
        assert_eq!(obs.overshoot_us(), 1);
    }

    #[test]
    fn observation_zero_actual_zero_budget_skipped() {
        let obs = PhaseObservation {
            phase: QueryPhase::Rerank,
            actual_us: 0,
            budget_us: 0,
            skipped: true,
        };
        assert!(!obs.over_budget());
        assert_eq!(obs.overshoot_us(), 0);
        assert!((obs.utilization() - 0.0).abs() < 1e-9);
    }

    // --- LatencyDecomposition edge cases ---

    #[test]
    fn empty_decomposition() {
        let d = LatencyDecomposition::new(vec![], 0, 0);
        assert_eq!(d.total_us, 0);
        assert_eq!(d.total_budget_us, 0);
        assert!(d.over_budget_phases().is_empty());
        assert_eq!(d.initial_path_us(), 0);
        assert_eq!(d.refinement_path_us(), 0);
        assert!(d.met_budget());
        assert_eq!(d.verdict_reason_code(), "query.latency.on_budget");
    }

    #[test]
    fn all_skipped_decomposition() {
        let d = LatencyDecomposition::new(
            vec![
                PhaseObservation {
                    phase: QueryPhase::Canonicalize,
                    actual_us: 0,
                    budget_us: 200,
                    skipped: true,
                },
                PhaseObservation {
                    phase: QueryPhase::QualityEmbed,
                    actual_us: 0,
                    budget_us: 130_000,
                    skipped: true,
                },
            ],
            0,
            100,
        );
        assert_eq!(d.total_us, 0);
        assert_eq!(d.total_budget_us, 0);
        assert!(d.met_budget());
    }

    #[test]
    fn multiple_over_budget_verdict() {
        let d = LatencyDecomposition::new(
            vec![
                PhaseObservation {
                    phase: QueryPhase::FastVectorSearch,
                    actual_us: 8_000,
                    budget_us: 5_000,
                    skipped: false,
                },
                PhaseObservation {
                    phase: QueryPhase::LexicalRetrieve,
                    actual_us: 7_000,
                    budget_us: 5_000,
                    skipped: false,
                },
            ],
            5,
            100,
        );
        assert_eq!(
            d.verdict_reason_code(),
            "query.latency.multiple_phases_over_budget"
        );
        assert!(!d.met_budget());
        let over = d.over_budget_phases();
        assert_eq!(over.len(), 2);
        // Sorted by overshoot descending: FastVectorSearch (3000) > LexicalRetrieve (2000).
        assert_eq!(over[0].phase, QueryPhase::FastVectorSearch);
        assert_eq!(over[1].phase, QueryPhase::LexicalRetrieve);
    }

    #[test]
    fn decomposition_schema_version_matches_constant() {
        let d = LatencyDecomposition::new(vec![], 0, 0);
        assert_eq!(d.schema_version, QUERY_LATENCY_OPT_SCHEMA_VERSION);
    }

    // --- Phase path classification edge cases ---

    #[test]
    fn explain_and_serialize_are_neither_initial_nor_refinement() {
        assert!(!QueryPhase::Explain.is_initial_path());
        assert!(!QueryPhase::Explain.is_refinement_path());
        assert!(!QueryPhase::Serialize.is_initial_path());
        assert!(!QueryPhase::Serialize.is_refinement_path());
    }

    #[test]
    fn every_phase_is_covered_by_at_least_one_category() {
        // Phases are initial, refinement, or "other" (Explain/Serialize).
        let other_phases = [QueryPhase::Explain, QueryPhase::Serialize];
        for phase in QueryPhase::ALL {
            assert!(
                phase.is_initial_path()
                    || phase.is_refinement_path()
                    || other_phases.contains(phase),
                "phase {phase:?} is uncategorized"
            );
        }
    }

    #[test]
    fn refinement_path_budget_sum() {
        let refinement_budget_us: u64 = QueryPhase::ALL
            .iter()
            .filter(|p| p.is_refinement_path())
            .map(|p| p.default_budget_us())
            .sum();
        // QualityEmbed(130000) + QualityVectorSearch(2000) + Blend(500) + Rerank(25000) = 157500
        assert_eq!(refinement_budget_us, 157_500);
    }

    #[test]
    fn total_budget_sum_matches_all_phases() {
        let total: u64 = QueryPhase::ALL.iter().map(|p| p.default_budget_us()).sum();
        // Sum all budgets from the match arms.
        let expected =
            200 + 100 + 800 + 5_000 + 5_000 + 500 + 130_000 + 2_000 + 500 + 25_000 + 1_000 + 500;
        assert_eq!(total, expected);
    }

    // --- VerificationResult edge cases ---

    #[test]
    fn verification_result_empty_assertions_passes() {
        let result = VerificationResult::from_assertions("test.lever", vec![]);
        assert!(result.passed);
        assert_eq!(result.failure_count(), 0);
        assert_eq!(result.reason_code, "opt.verify.passed");
    }

    #[test]
    fn verification_result_all_failures() {
        let result = VerificationResult::from_assertions(
            "test.lever",
            vec![
                CorrectnessAssertion {
                    lever_id: "test.lever".into(),
                    proof_kind: CorrectnessProofKind::BitIdentical,
                    test_corpus_ids: vec!["c1".into()],
                    assertion: "a1".into(),
                    passed: false,
                    reason_code: "opt.assert.failed".into(),
                },
                CorrectnessAssertion {
                    lever_id: "test.lever".into(),
                    proof_kind: CorrectnessProofKind::RankPreserving,
                    test_corpus_ids: vec!["c2".into()],
                    assertion: "a2".into(),
                    passed: false,
                    reason_code: "opt.assert.failed".into(),
                },
            ],
        );
        assert!(!result.passed);
        assert_eq!(result.failure_count(), 2);
    }

    #[test]
    fn verification_result_schema_version() {
        let result = VerificationResult::from_assertions("x", vec![]);
        assert_eq!(result.schema_version, QUERY_LATENCY_OPT_SCHEMA_VERSION);
    }

    // --- VerificationProtocol edge cases ---

    #[test]
    fn default_protocol_schema_version_matches_constant() {
        let protocol = VerificationProtocol::default_protocol();
        assert_eq!(protocol.schema_version, QUERY_LATENCY_OPT_SCHEMA_VERSION);
    }

    #[test]
    fn default_protocol_lever_ids_match_catalog_exactly() {
        let protocol = VerificationProtocol::default_protocol();
        let catalog = query_path_lever_catalog();
        assert_eq!(
            protocol.required_lever_ids.len(),
            catalog.len(),
            "protocol lever count should match catalog"
        );
    }

    #[test]
    fn default_protocol_epsilon_is_parseable() {
        let protocol = VerificationProtocol::default_protocol();
        let eps: f64 = protocol
            .score_epsilon_str
            .parse()
            .expect("epsilon should be parseable as f64");
        assert!(eps > 0.0 && eps < 1.0);
    }

    // --- Lever catalog edge cases ---

    #[test]
    fn lever_catalog_descriptions_nonempty() {
        for lever in query_path_lever_catalog() {
            assert!(
                !lever.description.is_empty(),
                "lever '{}' has empty description",
                lever.id
            );
        }
    }

    #[test]
    fn lever_ids_match_matrix_candidate_ids() {
        let matrix = query_path_opportunity_matrix();
        let catalog = query_path_lever_catalog();
        let matrix_ids: std::collections::BTreeSet<&str> =
            matrix.candidates.iter().map(|c| c.id.as_str()).collect();
        let catalog_ids: std::collections::BTreeSet<&str> =
            catalog.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(
            matrix_ids, catalog_ids,
            "matrix and catalog IDs should match exactly"
        );
    }

    // --- Serde edge cases ---

    #[test]
    fn correctness_proof_kind_serde_roundtrip() {
        let kinds = [
            CorrectnessProofKind::BitIdentical,
            CorrectnessProofKind::NumericallyEquivalent,
            CorrectnessProofKind::RankPreserving,
        ];
        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let decoded: CorrectnessProofKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, decoded);
        }
    }

    #[test]
    fn optimization_mechanism_serde_roundtrip() {
        let mechanisms = [
            OptimizationMechanism::AllocationReduction,
            OptimizationMechanism::BufferReuse,
            OptimizationMechanism::CacheLocality,
            OptimizationMechanism::AlgorithmReplacement,
            OptimizationMechanism::Parallelism,
            OptimizationMechanism::DataMovement,
            OptimizationMechanism::Precomputation,
        ];
        for mechanism in &mechanisms {
            let json = serde_json::to_string(mechanism).unwrap();
            let decoded: OptimizationMechanism = serde_json::from_str(&json).unwrap();
            assert_eq!(*mechanism, decoded);
        }
    }

    #[test]
    fn verification_protocol_serde_roundtrip() {
        let protocol = VerificationProtocol::default_protocol();
        let json = serde_json::to_string(&protocol).unwrap();
        let decoded: VerificationProtocol = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.schema_version, protocol.schema_version);
        assert_eq!(decoded.required_corpus_ids, protocol.required_corpus_ids);
        assert_eq!(decoded.required_lever_ids, protocol.required_lever_ids);
    }
}
