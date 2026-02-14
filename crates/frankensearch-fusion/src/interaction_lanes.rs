//! Interaction lane catalog for cross-feature composition testing.
//!
//! Each **lane** represents a specific combination of ranking/control features
//! that must be tested together to catch regressions invisible to isolated
//! per-feature tests. The catalog defines:
//!
//! - **Lane ID**: Stable identifier (e.g., `LANE_EXPLAIN_MMR`)
//! - **Feature toggles**: Which features are active for the lane
//! - **Fixture slices**: Which corpus subsets and query classes to exercise
//! - **Expected phase behavior**: Whether Phase 2 (refinement) should fire
//! - **Seed strategy**: Deterministic PRNG seed for reproducibility
//!
//! Downstream test beads (bd-3un.52.2 through 52.6) consume this catalog to
//! generate unit, integration, and e2e test matrices.

use std::fmt;

use serde::{Deserialize, Serialize};

use frankensearch_core::QueryClass;

// ── Feature toggles ──────────────────────────────────────────────────

/// Describes which ranking/control features are active for a given lane.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureToggles {
    /// Per-hit explanation generation (bd-11n).
    pub explain: bool,
    /// MMR diversified reranking (bd-z3j).
    pub mmr: bool,
    /// Negative/exclusion query parsing (bd-2n6). Always active (parsing is
    /// unconditional), but this flag controls whether test queries *contain*
    /// negation syntax.
    pub negation_queries: bool,
    /// PRF query expansion (bd-3st).
    pub prf: bool,
    /// Adaptive fusion parameters (bd-21g).
    pub adaptive_fusion: bool,
    /// Score calibration — which calibrator variant to use (bd-22k).
    pub calibration: CalibratorChoice,
    /// Conformal prediction wrappers (bd-2yj).
    pub conformal: bool,
    /// Quality-tier circuit breaker (bd-1do).
    pub circuit_breaker: bool,
    /// Implicit relevance feedback loop (bd-2tv).
    pub feedback: bool,
}

/// Which calibrator to activate for a lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibratorChoice {
    /// No calibration (identity passthrough).
    Identity,
    /// Temperature scaling on raw scores.
    Temperature,
    /// Platt (logistic) scaling.
    Platt,
    /// Isotonic regression mapping.
    Isotonic,
}

impl Default for FeatureToggles {
    fn default() -> Self {
        Self {
            explain: false,
            mmr: false,
            negation_queries: false,
            prf: false,
            adaptive_fusion: false,
            calibration: CalibratorChoice::Identity,
            conformal: false,
            circuit_breaker: true, // on by default in production
            feedback: false,
        }
    }
}

impl FeatureToggles {
    /// Count of non-default features active.
    #[must_use]
    pub fn active_count(&self) -> usize {
        let defaults = Self::default();
        let mut count = 0;
        if self.explain != defaults.explain {
            count += 1;
        }
        if self.mmr != defaults.mmr {
            count += 1;
        }
        if self.negation_queries != defaults.negation_queries {
            count += 1;
        }
        if self.prf != defaults.prf {
            count += 1;
        }
        if self.adaptive_fusion != defaults.adaptive_fusion {
            count += 1;
        }
        if self.calibration != defaults.calibration {
            count += 1;
        }
        if self.conformal != defaults.conformal {
            count += 1;
        }
        if self.circuit_breaker != defaults.circuit_breaker {
            count += 1;
        }
        if self.feedback != defaults.feedback {
            count += 1;
        }
        count
    }
}

// ── Fixture slices ───────────────────────────────────────────────────

/// Which subset of the test corpus (from `tests/fixtures/corpus.json`) to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorpusSlice {
    /// All 120 documents (5 clusters + supplemental).
    Full,
    /// 20 Rust programming documents (`test-rust-*`).
    Rust,
    /// 20 ML/NLP documents (`test-ml-*`).
    Ml,
    /// 20 sysadmin/devops documents (`test-sysadmin-*`).
    Sysadmin,
    /// 20 cooking/recipe documents (`test-cooking-*`).
    Cooking,
    /// 20 cross-domain mixed documents (`test-mixed-*`).
    Mixed,
    /// Supplemental code/config/adversarial/log documents.
    Supplemental,
}

impl CorpusSlice {
    /// Primary doc ID prefix for filtering the fixture corpus.
    ///
    /// For multi-prefix slices (like `Supplemental`), this returns the
    /// primary prefix. Prefer [`Self::id_prefixes`] for precise filtering.
    #[must_use]
    pub const fn id_prefix(self) -> &'static str {
        match self {
            Self::Full => "",
            Self::Rust => "test-rust-",
            Self::Ml => "test-ml-",
            Self::Sysadmin => "test-sysadmin-",
            Self::Cooking => "test-cooking-",
            Self::Mixed => "test-mixed-",
            Self::Supplemental => "test-adversarial-",
        }
    }

    /// All doc ID prefixes included in this slice.
    #[must_use]
    pub const fn id_prefixes(self) -> &'static [&'static str] {
        match self {
            Self::Full => &[],
            Self::Rust => &["test-rust-"],
            Self::Ml => &["test-ml-"],
            Self::Sysadmin => &["test-sysadmin-"],
            Self::Cooking => &["test-cooking-"],
            Self::Mixed => &["test-mixed-"],
            Self::Supplemental => &[
                "test-code-",
                "test-config-",
                "test-adversarial-",
                "test-log-",
            ],
        }
    }

    /// Whether a fixture document ID belongs to this slice.
    #[must_use]
    pub fn matches_doc_id(self, doc_id: &str) -> bool {
        if self == Self::Full {
            return true;
        }
        self.id_prefixes()
            .iter()
            .any(|prefix| doc_id.starts_with(prefix))
    }

    /// Approximate document count in this slice.
    #[must_use]
    pub const fn approx_count(self) -> usize {
        match self {
            Self::Full => 120,
            _ => 20,
        }
    }
}

/// Which query classes to exercise in a lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuerySlice {
    /// Query classes to include. Empty means all non-Empty classes.
    pub classes: Vec<QueryClass>,
    /// Whether to include negation-syntax variants of queries.
    pub include_negated: bool,
    /// Minimum number of queries per class to exercise.
    pub min_per_class: usize,
}

impl Default for QuerySlice {
    fn default() -> Self {
        Self {
            classes: vec![
                QueryClass::Identifier,
                QueryClass::ShortKeyword,
                QueryClass::NaturalLanguage,
            ],
            include_negated: false,
            min_per_class: 2,
        }
    }
}

// ── Expected phase behavior ──────────────────────────────────────────

/// What phase behavior a lane expects from the progressive search iterator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpectedPhase {
    /// Phase 1 only (`fast_only` mode or circuit breaker tripped).
    InitialOnly,
    /// Phase 1 then Phase 2 refinement (normal two-tier flow).
    InitialThenRefined,
    /// Phase 2 may fail gracefully (circuit breaker half-open probe).
    InitialThenMaybeRefined,
}

// ── Risk level ───────────────────────────────────────────────────────

/// How likely this interaction is to produce regressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk — features are mostly orthogonal.
    Low,
    /// Medium risk — features share scoring paths or state.
    Medium,
    /// High risk — features directly conflict or interfere.
    High,
}

// ── Interaction lane ─────────────────────────────────────────────────

/// A single interaction lane in the regression matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InteractionLane {
    /// Stable lane identifier (e.g., `"explain_mmr"`).
    pub id: &'static str,
    /// Human-readable description of what this lane validates.
    pub description: &'static str,
    /// Bead IDs of the features being composed (for traceability).
    pub bead_refs: &'static [&'static str],
    /// Which features are active.
    pub toggles: FeatureToggles,
    /// Which corpus slices to exercise.
    pub corpus_slices: &'static [CorpusSlice],
    /// Query slice configuration.
    pub query_slice: QuerySlice,
    /// Expected phase behavior.
    pub expected_phase: ExpectedPhase,
    /// How risky is this composition.
    pub risk: RiskLevel,
    /// Deterministic seed for reproducible ordering.
    pub seed: u64,
}

impl fmt::Display for InteractionLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lane[{}] risk={:?} features={} | {}",
            self.id,
            self.risk,
            self.toggles.active_count(),
            self.description,
        )
    }
}

// ── Canonical lane catalog ───────────────────────────────────────────

/// Lane 1: Explanations + MMR diversity reranking.
///
/// Validates that per-hit explanations correctly reflect MMR-reranked ordering,
/// including `rank_movement` tracking after MMR shuffles the result order.
pub const LANE_EXPLAIN_MMR: InteractionLane = InteractionLane {
    id: "explain_mmr",
    description: "Explanations reflect MMR-reranked ordering and rank movements",
    bead_refs: &["bd-11n", "bd-z3j"],
    toggles: FeatureToggles {
        explain: true,
        mmr: true,
        negation_queries: false,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![], // compile-time: see lane_catalog() for runtime
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::High,
    seed: 0x_CAFE_0001,
};

/// Lane 2: Explanations + exclusion query parsing.
///
/// Validates that explanations for queries with `-term` or `NOT "phrase"`
/// correctly show that excluded terms were not matched, and that explanation
/// components omit excluded sources.
pub const LANE_EXPLAIN_NEGATION: InteractionLane = InteractionLane {
    id: "explain_negation",
    description: "Explanations correctly handle excluded terms/phrases in queries",
    bead_refs: &["bd-11n", "bd-2n6"],
    toggles: FeatureToggles {
        explain: true,
        mmr: false,
        negation_queries: true,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Rust, CorpusSlice::Mixed],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: true,
        min_per_class: 3,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::Medium,
    seed: 0x_CAFE_0002,
};

/// Lane 3: PRF expansion + exclusion query parsing.
///
/// Validates that PRF centroid computation does not incorporate embeddings of
/// documents matching excluded terms, and that expanded queries preserve
/// negation semantics.
pub const LANE_PRF_NEGATION: InteractionLane = InteractionLane {
    id: "prf_negation",
    description: "PRF expansion respects negation — excludes negative hits from centroid",
    bead_refs: &["bd-3st", "bd-2n6"],
    toggles: FeatureToggles {
        explain: false,
        mmr: false,
        negation_queries: true,
        prf: true,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: true,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::High,
    seed: 0x_CAFE_0003,
};

/// Lane 4: Adaptive fusion + score calibration + conformal wrappers.
///
/// Validates the full Bayesian pipeline: calibrated scores feed into adaptive
/// blend/K posteriors, and conformal prediction coverage guarantees hold under
/// the calibrated distribution.
pub const LANE_ADAPTIVE_CALIBRATION_CONFORMAL: InteractionLane = InteractionLane {
    id: "adaptive_calibration_conformal",
    description: "Calibrated scores feed adaptive posteriors; conformal coverage holds",
    bead_refs: &["bd-21g", "bd-22k", "bd-2yj"],
    toggles: FeatureToggles {
        explain: false,
        mmr: false,
        negation_queries: false,
        prf: false,
        adaptive_fusion: true,
        calibration: CalibratorChoice::Platt,
        conformal: true,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 3,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::High,
    seed: 0x_CAFE_0004,
};

/// Lane 5: Circuit breaker + adaptive fusion + implicit feedback.
///
/// Validates that when the circuit breaker trips (quality tier too slow/poor),
/// adaptive fusion gracefully degrades (posteriors don't update on skipped
/// phases), and feedback boosts are still applied to Phase 1 results.
pub const LANE_BREAKER_ADAPTIVE_FEEDBACK: InteractionLane = InteractionLane {
    id: "breaker_adaptive_feedback",
    description: "Circuit breaker trip degrades adaptive fusion; feedback still applies",
    bead_refs: &["bd-1do", "bd-21g", "bd-2tv"],
    toggles: FeatureToggles {
        explain: false,
        mmr: false,
        negation_queries: false,
        prf: false,
        adaptive_fusion: true,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: true,
    },
    corpus_slices: &[CorpusSlice::Ml, CorpusSlice::Mixed],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenMaybeRefined,
    risk: RiskLevel::High,
    seed: 0x_CAFE_0005,
};

/// Lane 6: MMR + feedback — diversity vs boost conflict.
///
/// Validates that MMR diversification and feedback boosts produce a stable
/// order: feedback boosts raise relevance scores, but MMR may re-order them
/// for diversity. The net ranking should be deterministic for a given seed.
pub const LANE_MMR_FEEDBACK: InteractionLane = InteractionLane {
    id: "mmr_feedback",
    description: "MMR diversity and feedback boosts produce deterministic stable order",
    bead_refs: &["bd-z3j", "bd-2tv"],
    toggles: FeatureToggles {
        explain: false,
        mmr: true,
        negation_queries: false,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: true,
    },
    corpus_slices: &[CorpusSlice::Cooking, CorpusSlice::Mixed],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::Medium,
    seed: 0x_CAFE_0006,
};

/// Lane 7: PRF + adaptive fusion — expanded query meets learned parameters.
///
/// Validates that PRF-expanded embeddings work correctly with adaptive blend/K
/// parameters, especially that the posterior update uses the expanded query's
/// retrieval quality signal (not the original query's).
pub const LANE_PRF_ADAPTIVE: InteractionLane = InteractionLane {
    id: "prf_adaptive",
    description: "PRF-expanded query uses adaptive blend/K; posterior updates on expanded results",
    bead_refs: &["bd-3st", "bd-21g"],
    toggles: FeatureToggles {
        explain: false,
        mmr: false,
        negation_queries: false,
        prf: true,
        adaptive_fusion: true,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::Medium,
    seed: 0x_CAFE_0007,
};

/// Lane 8: Calibration + conformal — distribution assumptions.
///
/// Validates that conformal prediction coverage holds under each calibrator
/// variant (Temperature, Platt, Isotonic), not just Identity. Different
/// calibrators produce different score distributions, which affects the
/// nonconformity score quantiles.
pub const LANE_CALIBRATION_CONFORMAL: InteractionLane = InteractionLane {
    id: "calibration_conformal",
    description: "Conformal coverage holds under Temperature, Platt, and Isotonic calibrators",
    bead_refs: &["bd-22k", "bd-2yj"],
    toggles: FeatureToggles {
        explain: false,
        mmr: false,
        negation_queries: false,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Temperature,
        conformal: true,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::Medium,
    seed: 0x_CAFE_0008,
};

/// Lane 9: Explain + calibration — explanation scores reflect calibrated values.
///
/// Validates that `ScoreComponent::normalized_score` values in explanations
/// reflect the post-calibration scores, not raw scores, when a calibrator is
/// active.
pub const LANE_EXPLAIN_CALIBRATION: InteractionLane = InteractionLane {
    id: "explain_calibration",
    description: "Explanation normalized_score values reflect post-calibration scores",
    bead_refs: &["bd-11n", "bd-22k"],
    toggles: FeatureToggles {
        explain: true,
        mmr: false,
        negation_queries: false,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Platt,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Ml, CorpusSlice::Rust],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::Medium,
    seed: 0x_CAFE_0009,
};

/// Lane 10: Circuit breaker tripped + explain — explanation in degraded mode.
///
/// Validates that when the circuit breaker skips Phase 2, explanations still
/// have complete Phase 1 data and `ExplanationPhase::Initial`, and that
/// `RefinementFailed` phase carries appropriate error context.
pub const LANE_BREAKER_EXPLAIN: InteractionLane = InteractionLane {
    id: "breaker_explain",
    description: "Explanations in degraded mode (circuit breaker tripped) are complete for Phase 1",
    bead_refs: &["bd-1do", "bd-11n"],
    toggles: FeatureToggles {
        explain: true,
        mmr: false,
        negation_queries: false,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 2,
    },
    expected_phase: ExpectedPhase::InitialThenMaybeRefined,
    risk: RiskLevel::High,
    seed: 0x_CAFE_000A,
};

/// Lane 11: Full kitchen sink — all features active simultaneously.
///
/// Stress test: every feature toggle active at once. This catches emergent
/// interactions that only appear when the full pipeline is engaged. Expected
/// to be the slowest lane.
pub const LANE_KITCHEN_SINK: InteractionLane = InteractionLane {
    id: "kitchen_sink",
    description: "All features active simultaneously — catches emergent interactions",
    bead_refs: &[
        "bd-11n", "bd-z3j", "bd-2n6", "bd-3st", "bd-21g", "bd-22k", "bd-2yj", "bd-1do", "bd-2tv",
    ],
    toggles: FeatureToggles {
        explain: true,
        mmr: true,
        negation_queries: true,
        prf: true,
        adaptive_fusion: true,
        calibration: CalibratorChoice::Platt,
        conformal: true,
        circuit_breaker: true,
        feedback: true,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: true,
        min_per_class: 3,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::High,
    seed: 0x_CAFE_00FF,
};

/// Lane 12: Baseline — no optional features.
///
/// Control lane: only default features (circuit breaker on, everything else
/// off). Used to establish baseline metrics against which composition lanes
/// are compared.
pub const LANE_BASELINE: InteractionLane = InteractionLane {
    id: "baseline",
    description: "No optional features — control lane for baseline metrics comparison",
    bead_refs: &[],
    toggles: FeatureToggles {
        explain: false,
        mmr: false,
        negation_queries: false,
        prf: false,
        adaptive_fusion: false,
        calibration: CalibratorChoice::Identity,
        conformal: false,
        circuit_breaker: true,
        feedback: false,
    },
    corpus_slices: &[CorpusSlice::Full],
    query_slice: QuerySlice {
        classes: vec![],
        include_negated: false,
        min_per_class: 3,
    },
    expected_phase: ExpectedPhase::InitialThenRefined,
    risk: RiskLevel::Low,
    seed: 0x_CAFE_0000,
};

// ── Catalog accessor ─────────────────────────────────────────────────

/// Returns the full interaction lane catalog.
///
/// Lanes are ordered by risk level (High first), then by ID. This ordering
/// prioritizes the riskiest compositions in CI where time budgets are tight.
#[must_use]
pub fn lane_catalog() -> Vec<InteractionLane> {
    let mut catalog = vec![
        // ── From bd-3un.52 scope (5 primary interaction families) ──
        lane_with_default_query_slice(LANE_EXPLAIN_MMR),
        lane_with_default_query_slice(LANE_EXPLAIN_NEGATION),
        lane_with_default_query_slice(LANE_PRF_NEGATION),
        lane_with_default_query_slice(LANE_ADAPTIVE_CALIBRATION_CONFORMAL),
        lane_with_default_query_slice(LANE_BREAKER_ADAPTIVE_FEEDBACK),
        // ── Additional pairwise interactions ──
        lane_with_default_query_slice(LANE_MMR_FEEDBACK),
        lane_with_default_query_slice(LANE_PRF_ADAPTIVE),
        lane_with_default_query_slice(LANE_CALIBRATION_CONFORMAL),
        lane_with_default_query_slice(LANE_EXPLAIN_CALIBRATION),
        lane_with_default_query_slice(LANE_BREAKER_EXPLAIN),
        // ── Boundary lanes ──
        lane_with_default_query_slice(LANE_KITCHEN_SINK),
        lane_with_default_query_slice(LANE_BASELINE),
    ];
    catalog.sort_by(|a, b| b.risk.cmp(&a.risk).then_with(|| a.id.cmp(b.id)));
    catalog
}

/// Fills in the query slice with default classes when the const left it empty.
///
/// Const `Vec::new()` is not possible in const context, so we use an empty
/// vec sentinel in the const definitions and fill defaults at runtime.
fn lane_with_default_query_slice(mut lane: InteractionLane) -> InteractionLane {
    if lane.query_slice.classes.is_empty() {
        lane.query_slice.classes = vec![
            QueryClass::Identifier,
            QueryClass::ShortKeyword,
            QueryClass::NaturalLanguage,
        ];
    }
    lane
}

/// Returns lane IDs only, for quick enumeration.
#[must_use]
pub fn lane_ids() -> Vec<&'static str> {
    lane_catalog().iter().map(|l| l.id).collect()
}

/// Look up a lane by ID. Returns `None` if not found.
#[must_use]
pub fn lane_by_id(id: &str) -> Option<InteractionLane> {
    lane_catalog().into_iter().find(|l| l.id == id)
}

/// Returns only lanes at or above the given risk level.
#[must_use]
pub fn lanes_at_risk(min_risk: RiskLevel) -> Vec<InteractionLane> {
    lane_catalog()
        .into_iter()
        .filter(|l| l.risk >= min_risk)
        .collect()
}

/// Fixture query set for interaction testing.
///
/// Each entry maps a query class to representative queries from
/// `tests/fixtures/queries.json`. Downstream tests select from these
/// based on the lane's `QuerySlice`.
#[must_use]
pub fn fixture_queries() -> Vec<FixtureQuery> {
    vec![
        // ── Identifier ──
        FixtureQuery {
            text: "SearchIndex::new",
            class: QueryClass::Identifier,
            negated_variant: Some("SearchIndex::new -deprecated"),
        },
        FixtureQuery {
            text: "ONNX Runtime inference",
            class: QueryClass::Identifier,
            negated_variant: Some("ONNX Runtime inference -python"),
        },
        FixtureQuery {
            text: "BM25 scoring algorithm",
            class: QueryClass::Identifier,
            negated_variant: Some("BM25 scoring algorithm -lucene"),
        },
        FixtureQuery {
            text: "sentence transformers MiniLM",
            class: QueryClass::Identifier,
            negated_variant: Some(r#"sentence transformers MiniLM NOT "large model""#),
        },
        FixtureQuery {
            text: "HNSW approximate nearest neighbor",
            class: QueryClass::Identifier,
            negated_variant: None,
        },
        // ── ShortKeyword ──
        FixtureQuery {
            text: "rust ownership borrowing",
            class: QueryClass::ShortKeyword,
            negated_variant: Some("rust ownership borrowing -unsafe"),
        },
        FixtureQuery {
            text: "cosine similarity vector search",
            class: QueryClass::ShortKeyword,
            negated_variant: Some("cosine similarity vector search -euclidean"),
        },
        FixtureQuery {
            text: "f16 quantization memory savings",
            class: QueryClass::ShortKeyword,
            negated_variant: None,
        },
        FixtureQuery {
            text: "docker compose health check",
            class: QueryClass::ShortKeyword,
            negated_variant: Some("docker compose health check -swarm"),
        },
        FixtureQuery {
            text: "async await future executor",
            class: QueryClass::ShortKeyword,
            negated_variant: Some("async await future executor -tokio"),
        },
        // ── NaturalLanguage ──
        FixtureQuery {
            text: "how do transformers work for text embeddings",
            class: QueryClass::NaturalLanguage,
            negated_variant: Some("how do transformers work for text embeddings -GPT"),
        },
        FixtureQuery {
            text: "how to deploy containers with kubernetes",
            class: QueryClass::NaturalLanguage,
            negated_variant: Some(r#"how to deploy containers with kubernetes NOT "docker swarm""#),
        },
        FixtureQuery {
            text: "error handling Result type in Rust",
            class: QueryClass::NaturalLanguage,
            negated_variant: Some("error handling Result type in Rust -unwrap -panic"),
        },
        FixtureQuery {
            text: "reciprocal rank fusion hybrid search",
            class: QueryClass::NaturalLanguage,
            negated_variant: None,
        },
        FixtureQuery {
            text: "chocolate chip cookies recipe",
            class: QueryClass::NaturalLanguage,
            negated_variant: Some("chocolate chip cookies recipe -gluten"),
        },
    ]
}

/// A fixture query with optional negated variant for testing exclusion
/// interactions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureQuery {
    /// The query text (positive form).
    pub text: &'static str,
    /// Expected query class.
    pub class: QueryClass,
    /// Negated variant for lanes that test exclusion syntax. `None` if no
    /// meaningful negation exists for this query.
    pub negated_variant: Option<&'static str>,
}

impl FixtureQuery {
    /// Returns the query to use based on whether negation is active.
    #[must_use]
    pub fn query_for_lane(&self, include_negated: bool) -> &str {
        if include_negated {
            self.negated_variant.unwrap_or(self.text)
        } else {
            self.text
        }
    }
}

/// Returns queries filtered for a specific lane's requirements.
#[must_use]
pub fn queries_for_lane(lane: &InteractionLane) -> Vec<FixtureQuery> {
    let all = fixture_queries();
    all.into_iter()
        .filter(|q| lane.query_slice.classes.contains(&q.class))
        .collect()
}

// ── Seed strategy ────────────────────────────────────────────────────

/// Derives a per-query seed from the lane seed and query index.
///
/// This ensures each (lane, query) pair has a unique but reproducible seed
/// for any randomized operations (Thompson sampling, tie-breaking, etc.).
#[must_use]
pub const fn derive_query_seed(lane_seed: u64, query_index: usize) -> u64 {
    // Simple but effective: XOR with shifted index to avoid collisions.
    lane_seed ^ ((query_index as u64).wrapping_mul(0x_9E37_79B9_7F4A_7C15))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_expected_lane_count() {
        let catalog = lane_catalog();
        assert_eq!(catalog.len(), 12);
    }

    #[test]
    fn all_lane_ids_are_unique() {
        let ids = lane_ids();
        let mut dedup = ids.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(ids.len(), dedup.len(), "duplicate lane IDs found");
    }

    #[test]
    fn all_lane_seeds_are_unique() {
        let catalog = lane_catalog();
        let mut seeds: Vec<u64> = catalog.iter().map(|l| l.seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(
            seeds.len(),
            catalog.len(),
            "duplicate seeds found — each lane needs a unique seed"
        );
    }

    #[test]
    fn catalog_sorted_by_risk_descending() {
        let catalog = lane_catalog();
        for pair in catalog.windows(2) {
            assert!(
                pair[0].risk >= pair[1].risk,
                "lane {} (risk {:?}) should come before {} (risk {:?})",
                pair[0].id,
                pair[0].risk,
                pair[1].id,
                pair[1].risk,
            );
        }
    }

    #[test]
    fn baseline_lane_has_default_toggles() {
        let baseline = lane_by_id("baseline").expect("baseline lane not found");
        assert_eq!(baseline.toggles.active_count(), 0);
        assert_eq!(baseline.risk, RiskLevel::Low);
        assert_eq!(baseline.expected_phase, ExpectedPhase::InitialThenRefined);
    }

    #[test]
    fn kitchen_sink_has_all_features_active() {
        let ks = lane_by_id("kitchen_sink").expect("kitchen_sink lane not found");
        assert!(ks.toggles.explain);
        assert!(ks.toggles.mmr);
        assert!(ks.toggles.negation_queries);
        assert!(ks.toggles.prf);
        assert!(ks.toggles.adaptive_fusion);
        assert!(ks.toggles.conformal);
        assert!(ks.toggles.circuit_breaker);
        assert!(ks.toggles.feedback);
        assert_ne!(ks.toggles.calibration, CalibratorChoice::Identity);
    }

    #[test]
    fn lane_by_id_returns_none_for_unknown() {
        assert!(lane_by_id("nonexistent_lane").is_none());
    }

    #[test]
    fn high_risk_lanes_include_expected_compositions() {
        let high = lanes_at_risk(RiskLevel::High);
        let ids: Vec<&str> = high.iter().map(|l| l.id).collect();
        assert!(
            ids.contains(&"explain_mmr"),
            "explain_mmr missing from high-risk"
        );
        assert!(
            ids.contains(&"prf_negation"),
            "prf_negation missing from high-risk"
        );
        assert!(
            ids.contains(&"adaptive_calibration_conformal"),
            "adaptive_calibration_conformal missing from high-risk"
        );
        assert!(
            ids.contains(&"breaker_adaptive_feedback"),
            "breaker_adaptive_feedback missing from high-risk"
        );
        assert!(
            ids.contains(&"breaker_explain"),
            "breaker_explain missing from high-risk"
        );
        assert!(
            ids.contains(&"kitchen_sink"),
            "kitchen_sink missing from high-risk"
        );
    }

    #[test]
    fn fixture_queries_cover_all_non_empty_classes() {
        let queries = fixture_queries();
        let classes: Vec<QueryClass> = queries.iter().map(|q| q.class).collect();
        assert!(classes.contains(&QueryClass::Identifier));
        assert!(classes.contains(&QueryClass::ShortKeyword));
        assert!(classes.contains(&QueryClass::NaturalLanguage));
        // Empty is intentionally excluded — no meaningful interaction tests for empty queries
        assert!(!classes.contains(&QueryClass::Empty));
    }

    #[test]
    fn fixture_queries_have_negated_variants() {
        let queries = fixture_queries();
        let with_negation = queries
            .iter()
            .filter(|q| q.negated_variant.is_some())
            .count();
        assert!(
            with_negation >= 8,
            "expected at least 8 queries with negated variants, got {with_negation}"
        );
    }

    #[test]
    fn queries_for_lane_filters_correctly() {
        let baseline = lane_by_id("baseline").unwrap();
        let filtered = queries_for_lane(&baseline);
        // Baseline includes all 3 classes, should get all queries
        assert!(filtered.len() >= 10);

        for q in &filtered {
            assert!(baseline.query_slice.classes.contains(&q.class));
        }
    }

    #[test]
    fn query_for_lane_selects_negated_when_active() {
        let q = FixtureQuery {
            text: "rust error handling",
            class: QueryClass::ShortKeyword,
            negated_variant: Some("rust error handling -panic"),
        };
        assert_eq!(q.query_for_lane(false), "rust error handling");
        assert_eq!(q.query_for_lane(true), "rust error handling -panic");
    }

    #[test]
    fn query_for_lane_falls_back_when_no_negation() {
        let q = FixtureQuery {
            text: "search index",
            class: QueryClass::ShortKeyword,
            negated_variant: None,
        };
        assert_eq!(q.query_for_lane(true), "search index");
        assert_eq!(q.query_for_lane(false), "search index");
    }

    #[test]
    fn derive_query_seed_is_deterministic() {
        let s1 = derive_query_seed(0x_CAFE_0001, 0);
        let s2 = derive_query_seed(0x_CAFE_0001, 0);
        assert_eq!(s1, s2);
    }

    #[test]
    fn derive_query_seed_differs_per_index() {
        let s0 = derive_query_seed(0x_CAFE_0001, 0);
        let s1 = derive_query_seed(0x_CAFE_0001, 1);
        let s2 = derive_query_seed(0x_CAFE_0001, 2);
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
        assert_ne!(s0, s2);
    }

    #[test]
    fn derive_query_seed_differs_per_lane() {
        let a = derive_query_seed(0x_CAFE_0001, 5);
        let b = derive_query_seed(0x_CAFE_0002, 5);
        assert_ne!(a, b);
    }

    #[test]
    fn corpus_slice_id_prefix_non_empty_for_clusters() {
        for slice in [
            CorpusSlice::Rust,
            CorpusSlice::Ml,
            CorpusSlice::Sysadmin,
            CorpusSlice::Cooking,
            CorpusSlice::Mixed,
        ] {
            assert!(!slice.id_prefix().is_empty(), "{slice:?} has empty prefix");
        }
        assert!(CorpusSlice::Full.id_prefix().is_empty());
    }

    #[test]
    fn corpus_slice_counts_sum_correctly() {
        let cluster_count = [
            CorpusSlice::Rust,
            CorpusSlice::Ml,
            CorpusSlice::Sysadmin,
            CorpusSlice::Cooking,
            CorpusSlice::Mixed,
        ]
        .iter()
        .map(|s| s.approx_count())
        .sum::<usize>();
        assert_eq!(cluster_count, 100, "5 clusters x 20 = 100");
        assert_eq!(CorpusSlice::Supplemental.approx_count(), 20);
        assert_eq!(
            cluster_count + CorpusSlice::Supplemental.approx_count(),
            CorpusSlice::Full.approx_count()
        );
    }

    #[test]
    fn supplemental_slice_matches_all_prefixes() {
        let slice = CorpusSlice::Supplemental;
        assert!(slice.matches_doc_id("test-code-001"));
        assert!(slice.matches_doc_id("test-config-001"));
        assert!(slice.matches_doc_id("test-adversarial-001"));
        assert!(slice.matches_doc_id("test-log-001"));
        assert!(!slice.matches_doc_id("test-rust-001"));
    }

    #[test]
    fn feature_toggles_active_count_zero_for_default() {
        assert_eq!(FeatureToggles::default().active_count(), 0);
    }

    #[test]
    fn feature_toggles_active_count_matches_toggles() {
        let toggles = FeatureToggles {
            explain: true,
            mmr: true,
            ..FeatureToggles::default()
        };
        assert_eq!(toggles.active_count(), 2);
    }

    #[test]
    fn all_lanes_have_bead_refs() {
        let catalog = lane_catalog();
        for lane in &catalog {
            if lane.id != "baseline" {
                assert!(
                    !lane.bead_refs.is_empty(),
                    "lane {} has no bead refs",
                    lane.id
                );
            }
        }
    }

    #[test]
    fn all_lanes_have_corpus_slices() {
        for lane in &lane_catalog() {
            assert!(
                !lane.corpus_slices.is_empty(),
                "lane {} has no corpus slices",
                lane.id
            );
        }
    }

    #[test]
    fn negation_lanes_have_negation_queries_toggle() {
        for lane in &lane_catalog() {
            if lane.query_slice.include_negated {
                assert!(
                    lane.toggles.negation_queries,
                    "lane {} has include_negated but negation_queries is false",
                    lane.id
                );
            }
        }
    }

    #[test]
    fn display_format_includes_lane_id() {
        let lane = lane_by_id("explain_mmr").unwrap();
        let display = format!("{lane}");
        assert!(display.contains("explain_mmr"));
        assert!(display.contains("High"));
    }

    #[test]
    fn serde_roundtrip_lane() {
        for lane in &lane_catalog() {
            let json = serde_json::to_string(lane).expect("serialize lane");
            let value: serde_json::Value =
                serde_json::from_str(&json).expect("parse serialized lane json");
            assert_eq!(
                value.get("id").and_then(serde_json::Value::as_str),
                Some(lane.id)
            );
            assert_eq!(
                value.get("seed").and_then(serde_json::Value::as_u64),
                Some(lane.seed)
            );
        }
    }

    #[test]
    fn serde_roundtrip_feature_toggles() {
        let toggles = FeatureToggles {
            explain: true,
            mmr: true,
            negation_queries: true,
            prf: true,
            adaptive_fusion: true,
            calibration: CalibratorChoice::Isotonic,
            conformal: true,
            circuit_breaker: false,
            feedback: true,
        };
        let json = serde_json::to_string(&toggles).unwrap();
        let back: FeatureToggles = serde_json::from_str(&json).unwrap();
        assert_eq!(toggles, back);
    }

    #[test]
    fn calibrator_choice_all_variants_serialize() {
        for choice in [
            CalibratorChoice::Identity,
            CalibratorChoice::Temperature,
            CalibratorChoice::Platt,
            CalibratorChoice::Isotonic,
        ] {
            let json = serde_json::to_string(&choice).unwrap();
            let back: CalibratorChoice = serde_json::from_str(&json).unwrap();
            assert_eq!(choice, back);
        }
    }
}
