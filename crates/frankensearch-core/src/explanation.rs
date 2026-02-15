//! Per-hit search result explanations.
//!
//! When `TwoTierConfig { explain: true }`, each search result carries a
//! [`HitExplanation`] that decomposes its final score into individual
//! [`ScoreComponent`]s from each scoring source (BM25, fast semantic,
//! quality semantic, reranker).
//!
//! Zero allocation overhead when `explain = false`.
//!
//! # Example
//!
//! ```
//! use frankensearch_core::explanation::*;
//!
//! let explanation = HitExplanation {
//!     final_score: 0.032,
//!     components: vec![
//!         ScoreComponent {
//!             source: ExplainedSource::LexicalBm25 {
//!                 matched_terms: vec!["rust".into(), "async".into()],
//!                 tf: 2.0,
//!                 idf: 3.5,
//!             },
//!             raw_score: 12.5,
//!             normalized_score: 0.85,
//!             rrf_contribution: 0.016,
//!             weight: 0.3,
//!         },
//!     ],
//!     phase: ExplanationPhase::Refined,
//!     rank_movement: Some(RankMovement {
//!         initial_rank: 5,
//!         refined_rank: 2,
//!         delta: -3,
//!         reason: "promoted by quality embedder".into(),
//!     }),
//! };
//!
//! assert_eq!(explanation.components.len(), 1);
//! assert!(explanation.rank_movement.is_some());
//! ```

use serde::{Deserialize, Serialize};

/// Which search phase produced this explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplanationPhase {
    /// Fast-tier results (Phase 1).
    Initial,
    /// Quality-refined results (Phase 2).
    Refined,
}

impl std::fmt::Display for ExplanationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initial => write!(f, "Initial"),
            Self::Refined => write!(f, "Refined"),
        }
    }
}

/// Detailed score source with decomposition data.
///
/// Unlike [`crate::ScoreSource`] (which is a simple tag), this carries
/// the raw scoring signals for debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplainedSource {
    /// Lexical BM25 score with term-level detail.
    LexicalBm25 {
        /// Terms from the query that matched this document.
        matched_terms: Vec<String>,
        /// Aggregate term frequency.
        tf: f64,
        /// Aggregate inverse document frequency.
        idf: f64,
    },

    /// Fast-tier semantic score.
    SemanticFast {
        /// Embedder model identifier.
        embedder: String,
        /// Cosine similarity between query and document embeddings.
        cosine_sim: f64,
    },

    /// Quality-tier semantic score.
    SemanticQuality {
        /// Embedder model identifier.
        embedder: String,
        /// Cosine similarity between query and document embeddings.
        cosine_sim: f64,
    },

    /// Cross-encoder reranker score.
    Rerank {
        /// Reranker model identifier.
        model: String,
        /// Raw logit output from the model.
        logit: f64,
        /// Sigmoid-activated score (probability-like, 0.0-1.0).
        sigmoid: f64,
    },
}

impl std::fmt::Display for ExplainedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LexicalBm25 {
                matched_terms,
                tf,
                idf,
            } => {
                write!(
                    f,
                    "BM25(terms=[{}], tf={tf:.2}, idf={idf:.2})",
                    matched_terms.join(", ")
                )
            }
            Self::SemanticFast {
                embedder,
                cosine_sim,
            } => {
                write!(f, "FastSemantic({embedder}, cos={cosine_sim:.4})")
            }
            Self::SemanticQuality {
                embedder,
                cosine_sim,
            } => {
                write!(f, "QualitySemantic({embedder}, cos={cosine_sim:.4})")
            }
            Self::Rerank {
                model,
                logit,
                sigmoid,
            } => {
                write!(f, "Rerank({model}, logit={logit:.4}, sig={sigmoid:.4})")
            }
        }
    }
}

/// A single scoring component's contribution to the final score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreComponent {
    /// Which scoring source produced this component.
    pub source: ExplainedSource,
    /// Raw score from the source (BM25 score, cosine sim, logit, etc.).
    pub raw_score: f64,
    /// Score after normalization (min-max or z-score).
    pub normalized_score: f64,
    /// This source's RRF contribution: `1 / (K + rank + 1)`.
    pub rrf_contribution: f64,
    /// Weight applied during blending (e.g., 0.7 for quality, 0.3 for fast).
    pub weight: f64,
}

impl std::fmt::Display for ScoreComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: raw={:.4} norm={:.4} rrf={:.6} w={:.2}",
            self.source, self.raw_score, self.normalized_score, self.rrf_contribution, self.weight,
        )
    }
}

/// How a document's rank changed between initial and refined phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankMovement {
    /// Rank in Phase 1 (0-indexed).
    pub initial_rank: usize,
    /// Rank in Phase 2 (0-indexed).
    pub refined_rank: usize,
    /// Signed rank delta (`refined_rank as i32 - initial_rank as i32`).
    /// Negative means promoted (moved to a better rank).
    pub delta: i32,
    /// Human-readable reason for the movement.
    pub reason: String,
}

impl std::fmt::Display for RankMovement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let direction = match self.delta.cmp(&0) {
            std::cmp::Ordering::Less => "promoted",
            std::cmp::Ordering::Greater => "demoted",
            std::cmp::Ordering::Equal => "stable",
        };
        write!(
            f,
            "{direction} #{} -> #{} (delta={}) {}",
            self.initial_rank, self.refined_rank, self.delta, self.reason,
        )
    }
}

/// Per-hit explanation decomposing why a result was ranked where it is.
///
/// Attached to [`crate::ScoredResult`] when `TwoTierConfig { explain: true }`.
/// When `explain = false`, no `HitExplanation` is allocated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitExplanation {
    /// The final composite score for this hit.
    pub final_score: f64,
    /// Individual scoring components that contributed to the final score.
    pub components: Vec<ScoreComponent>,
    /// Which search phase produced this explanation.
    pub phase: ExplanationPhase,
    /// How this document's rank changed between phases, if applicable.
    /// `None` for Phase 1 results (no prior phase to compare).
    pub rank_movement: Option<RankMovement>,
}

impl HitExplanation {
    /// Sum of all RRF contributions across components.
    #[must_use]
    pub fn total_rrf_contribution(&self) -> f64 {
        self.components.iter().map(|c| c.rrf_contribution).sum()
    }

    /// Number of scoring sources that contributed to this hit.
    #[must_use]
    pub const fn source_count(&self) -> usize {
        self.components.len()
    }

    /// Whether this hit was promoted (moved to a better rank) during refinement.
    #[must_use]
    pub fn was_promoted(&self) -> bool {
        self.rank_movement.as_ref().is_some_and(|m| m.delta < 0)
    }

    /// Whether this hit was demoted (moved to a worse rank) during refinement.
    #[must_use]
    pub fn was_demoted(&self) -> bool {
        self.rank_movement.as_ref().is_some_and(|m| m.delta > 0)
    }
}

impl std::fmt::Display for HitExplanation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Score: {:.6} ({})", self.final_score, self.phase)?;
        for (i, component) in self.components.iter().enumerate() {
            writeln!(f, "  [{i}] {component}")?;
        }
        if let Some(ref movement) = self.rank_movement {
            writeln!(f, "  Rank: {movement}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────

    #[test]
    fn explanation_with_all_sources() {
        let explanation = HitExplanation {
            final_score: 0.032,
            components: vec![
                ScoreComponent {
                    source: ExplainedSource::LexicalBm25 {
                        matched_terms: vec!["rust".into(), "async".into()],
                        tf: 2.0,
                        idf: 3.5,
                    },
                    raw_score: 12.5,
                    normalized_score: 0.85,
                    rrf_contribution: 0.016,
                    weight: 0.3,
                },
                ScoreComponent {
                    source: ExplainedSource::SemanticFast {
                        embedder: "potion-128M".into(),
                        cosine_sim: 0.72,
                    },
                    raw_score: 0.72,
                    normalized_score: 0.80,
                    rrf_contribution: 0.012,
                    weight: 0.3,
                },
                ScoreComponent {
                    source: ExplainedSource::SemanticQuality {
                        embedder: "MiniLM-L6-v2".into(),
                        cosine_sim: 0.88,
                    },
                    raw_score: 0.88,
                    normalized_score: 0.92,
                    rrf_contribution: 0.015,
                    weight: 0.7,
                },
                ScoreComponent {
                    source: ExplainedSource::Rerank {
                        model: "flashrank".into(),
                        logit: 1.5,
                        sigmoid: 0.82,
                    },
                    raw_score: 1.5,
                    normalized_score: 0.82,
                    rrf_contribution: 0.0,
                    weight: 1.0,
                },
            ],
            phase: ExplanationPhase::Refined,
            rank_movement: Some(RankMovement {
                initial_rank: 5,
                refined_rank: 2,
                delta: -3,
                reason: "promoted by quality embedder".into(),
            }),
        };

        assert_eq!(explanation.source_count(), 4);
        assert!(explanation.was_promoted());
        assert!(!explanation.was_demoted());
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    #[test]
    fn total_rrf_contribution() {
        let explanation = HitExplanation {
            final_score: 0.028,
            components: vec![
                ScoreComponent {
                    source: ExplainedSource::LexicalBm25 {
                        matched_terms: vec!["test".into()],
                        tf: 1.0,
                        idf: 2.0,
                    },
                    raw_score: 5.0,
                    normalized_score: 0.5,
                    rrf_contribution: 0.016,
                    weight: 0.3,
                },
                ScoreComponent {
                    source: ExplainedSource::SemanticFast {
                        embedder: "potion-128M".into(),
                        cosine_sim: 0.65,
                    },
                    raw_score: 0.65,
                    normalized_score: 0.7,
                    rrf_contribution: 0.012,
                    weight: 0.3,
                },
            ],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };

        assert!((explanation.total_rrf_contribution() - 0.028).abs() < 1e-10);
    }

    #[test]
    fn no_rank_movement_for_initial() {
        let explanation = HitExplanation {
            final_score: 0.02,
            components: vec![],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };

        assert!(!explanation.was_promoted());
        assert!(!explanation.was_demoted());
    }

    #[test]
    fn rank_movement_demoted() {
        let explanation = HitExplanation {
            final_score: 0.015,
            components: vec![],
            phase: ExplanationPhase::Refined,
            rank_movement: Some(RankMovement {
                initial_rank: 2,
                refined_rank: 7,
                delta: 5,
                reason: "demoted after rerank".into(),
            }),
        };

        assert!(explanation.was_demoted());
        assert!(!explanation.was_promoted());
    }

    #[test]
    fn rank_movement_stable() {
        let explanation = HitExplanation {
            final_score: 0.02,
            components: vec![],
            phase: ExplanationPhase::Refined,
            rank_movement: Some(RankMovement {
                initial_rank: 3,
                refined_rank: 3,
                delta: 0,
                reason: "unchanged".into(),
            }),
        };

        assert!(!explanation.was_promoted());
        assert!(!explanation.was_demoted());
    }

    // ── Display ──────────────────────────────────────────────────────────

    #[test]
    fn display_explanation() {
        let explanation = HitExplanation {
            final_score: 0.032,
            components: vec![ScoreComponent {
                source: ExplainedSource::LexicalBm25 {
                    matched_terms: vec!["rust".into()],
                    tf: 1.0,
                    idf: 3.0,
                },
                raw_score: 8.0,
                normalized_score: 0.7,
                rrf_contribution: 0.016,
                weight: 0.3,
            }],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };

        let display = explanation.to_string();
        assert!(display.contains("0.032000"));
        assert!(display.contains("Initial"));
        assert!(display.contains("BM25"));
        assert!(display.contains("rust"));
    }

    #[test]
    fn display_explained_source_variants() {
        let bm25 = ExplainedSource::LexicalBm25 {
            matched_terms: vec!["a".into(), "b".into()],
            tf: 2.0,
            idf: 3.0,
        };
        assert!(bm25.to_string().contains("BM25"));
        assert!(bm25.to_string().contains("a, b"));

        let fast = ExplainedSource::SemanticFast {
            embedder: "model".into(),
            cosine_sim: 0.75,
        };
        assert!(fast.to_string().contains("FastSemantic"));
        assert!(fast.to_string().contains("0.7500"));

        let quality = ExplainedSource::SemanticQuality {
            embedder: "model".into(),
            cosine_sim: 0.88,
        };
        assert!(quality.to_string().contains("QualitySemantic"));

        let rerank = ExplainedSource::Rerank {
            model: "flashrank".into(),
            logit: 1.5,
            sigmoid: 0.82,
        };
        assert!(rerank.to_string().contains("Rerank"));
        assert!(rerank.to_string().contains("sig="));
    }

    #[test]
    fn display_rank_movement() {
        let promoted = RankMovement {
            initial_rank: 8,
            refined_rank: 2,
            delta: -6,
            reason: "boosted by quality embedder".into(),
        };
        let s = promoted.to_string();
        assert!(s.contains("promoted"));
        assert!(s.contains("#8 -> #2"));

        let demoted = RankMovement {
            initial_rank: 1,
            refined_rank: 5,
            delta: 4,
            reason: "penalized after rerank".into(),
        };
        assert!(demoted.to_string().contains("demoted"));
    }

    #[test]
    fn display_explanation_phase() {
        assert_eq!(ExplanationPhase::Initial.to_string(), "Initial");
        assert_eq!(ExplanationPhase::Refined.to_string(), "Refined");
    }

    // ── Serialization ────────────────────────────────────────────────────

    #[test]
    fn serde_roundtrip() {
        let explanation = HitExplanation {
            final_score: 0.025,
            components: vec![
                ScoreComponent {
                    source: ExplainedSource::LexicalBm25 {
                        matched_terms: vec!["search".into()],
                        tf: 1.0,
                        idf: 2.5,
                    },
                    raw_score: 6.0,
                    normalized_score: 0.6,
                    rrf_contribution: 0.014,
                    weight: 0.3,
                },
                ScoreComponent {
                    source: ExplainedSource::Rerank {
                        model: "flashrank".into(),
                        logit: 0.8,
                        sigmoid: 0.69,
                    },
                    raw_score: 0.8,
                    normalized_score: 0.69,
                    rrf_contribution: 0.0,
                    weight: 1.0,
                },
            ],
            phase: ExplanationPhase::Refined,
            rank_movement: Some(RankMovement {
                initial_rank: 4,
                refined_rank: 1,
                delta: -3,
                reason: "promoted".into(),
            }),
        };

        let json = serde_json::to_string(&explanation).unwrap();
        let decoded: HitExplanation = serde_json::from_str(&json).unwrap();

        assert!((decoded.final_score - 0.025).abs() < f64::EPSILON);
        assert_eq!(decoded.components.len(), 2);
        assert_eq!(decoded.phase, ExplanationPhase::Refined);
        assert!(decoded.rank_movement.is_some());
        let rm = decoded.rank_movement.unwrap();
        assert_eq!(rm.initial_rank, 4);
        assert_eq!(rm.refined_rank, 1);
        assert_eq!(rm.delta, -3);
    }

    #[test]
    fn serde_explained_source_variants() {
        // Ensure each variant round-trips correctly.
        let sources = vec![
            ExplainedSource::LexicalBm25 {
                matched_terms: vec!["test".into()],
                tf: 1.0,
                idf: 2.0,
            },
            ExplainedSource::SemanticFast {
                embedder: "potion".into(),
                cosine_sim: 0.7,
            },
            ExplainedSource::SemanticQuality {
                embedder: "minilm".into(),
                cosine_sim: 0.85,
            },
            ExplainedSource::Rerank {
                model: "flashrank".into(),
                logit: 1.2,
                sigmoid: 0.77,
            },
        ];

        for source in &sources {
            let json = serde_json::to_string(source).unwrap();
            let decoded: ExplainedSource = serde_json::from_str(&json).unwrap();
            // Verify the variant tag survived.
            let json2 = serde_json::to_string(&decoded).unwrap();
            assert_eq!(json, json2);
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn empty_components() {
        let explanation = HitExplanation {
            final_score: 0.0,
            components: vec![],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };

        assert_eq!(explanation.source_count(), 0);
        assert!(explanation.total_rrf_contribution().abs() < f64::EPSILON);
    }

    #[test]
    fn empty_matched_terms() {
        let source = ExplainedSource::LexicalBm25 {
            matched_terms: vec![],
            tf: 0.0,
            idf: 0.0,
        };
        let display = source.to_string();
        assert!(display.contains("BM25(terms=[]"));
    }

    // ─── bd-7fn8 tests begin ───

    #[test]
    fn explanation_phase_serde_roundtrip() {
        for phase in [ExplanationPhase::Initial, ExplanationPhase::Refined] {
            let json = serde_json::to_string(&phase).unwrap();
            let decoded: ExplanationPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, phase);
        }
    }

    #[test]
    fn explanation_phase_clone_copy_eq() {
        let a = ExplanationPhase::Initial;
        let b = a; // Copy
        let c = a; // Copy (ExplanationPhase is Copy)
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(ExplanationPhase::Initial, ExplanationPhase::Refined);
    }

    #[test]
    fn explanation_phase_debug() {
        let dbg = format!("{:?}", ExplanationPhase::Initial);
        assert_eq!(dbg, "Initial");
        let dbg2 = format!("{:?}", ExplanationPhase::Refined);
        assert_eq!(dbg2, "Refined");
    }

    #[test]
    fn explained_source_clone() {
        let original = ExplainedSource::LexicalBm25 {
            matched_terms: vec!["hello".into(), "world".into()],
            tf: 2.0,
            idf: 3.5,
        };
        let cloned = original.clone();
        assert_eq!(original.to_string(), cloned.to_string());

        let fast = ExplainedSource::SemanticFast {
            embedder: "potion".into(),
            cosine_sim: 0.75,
        };
        let fast_cloned = fast.clone();
        assert_eq!(fast.to_string(), fast_cloned.to_string());
    }

    #[test]
    fn explained_source_debug() {
        let source = ExplainedSource::Rerank {
            model: "test-model".into(),
            logit: 1.0,
            sigmoid: 0.73,
        };
        let dbg = format!("{source:?}");
        assert!(dbg.contains("Rerank"));
        assert!(dbg.contains("test-model"));
    }

    #[test]
    fn score_component_display() {
        let component = ScoreComponent {
            source: ExplainedSource::SemanticFast {
                embedder: "potion-128M".into(),
                cosine_sim: 0.65,
            },
            raw_score: 0.65,
            normalized_score: 0.72,
            rrf_contribution: 0.012_345,
            weight: 0.3,
        };
        let display = component.to_string();
        assert!(display.contains("FastSemantic"));
        assert!(display.contains("raw=0.6500"));
        assert!(display.contains("norm=0.7200"));
        assert!(display.contains("rrf=0.012345"));
        assert!(display.contains("w=0.30"));
    }

    #[test]
    fn score_component_clone_debug() {
        let component = ScoreComponent {
            source: ExplainedSource::LexicalBm25 {
                matched_terms: vec!["test".into()],
                tf: 1.0,
                idf: 2.0,
            },
            raw_score: 5.0,
            normalized_score: 0.5,
            rrf_contribution: 0.016,
            weight: 0.3,
        };
        let cloned = component.clone();
        assert!((cloned.raw_score - 5.0).abs() < f64::EPSILON);
        assert!((cloned.weight - 0.3).abs() < f64::EPSILON);

        let dbg = format!("{component:?}");
        assert!(dbg.contains("ScoreComponent"));
        assert!(dbg.contains("raw_score"));
    }

    #[test]
    fn score_component_serde_roundtrip() {
        let component = ScoreComponent {
            source: ExplainedSource::SemanticQuality {
                embedder: "minilm".into(),
                cosine_sim: 0.88,
            },
            raw_score: 0.88,
            normalized_score: 0.92,
            rrf_contribution: 0.015,
            weight: 0.7,
        };
        let json = serde_json::to_string(&component).unwrap();
        let decoded: ScoreComponent = serde_json::from_str(&json).unwrap();
        assert!((decoded.raw_score - 0.88).abs() < f64::EPSILON);
        assert!((decoded.weight - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn rank_movement_stable_display() {
        let movement = RankMovement {
            initial_rank: 3,
            refined_rank: 3,
            delta: 0,
            reason: "no change".into(),
        };
        let display = movement.to_string();
        assert!(display.contains("stable"));
        assert!(display.contains("#3 -> #3"));
        assert!(display.contains("delta=0"));
    }

    #[test]
    fn rank_movement_clone_debug() {
        let movement = RankMovement {
            initial_rank: 1,
            refined_rank: 4,
            delta: 3,
            reason: "penalized".into(),
        };
        let cloned = movement.clone();
        assert_eq!(cloned.initial_rank, 1);
        assert_eq!(cloned.refined_rank, 4);
        assert_eq!(cloned.delta, 3);
        assert_eq!(cloned.reason, "penalized");

        let dbg = format!("{movement:?}");
        assert!(dbg.contains("RankMovement"));
        assert!(dbg.contains("penalized"));
    }

    #[test]
    fn rank_movement_serde_roundtrip() {
        let movement = RankMovement {
            initial_rank: 10,
            refined_rank: 2,
            delta: -8,
            reason: "major promotion".into(),
        };
        let json = serde_json::to_string(&movement).unwrap();
        let decoded: RankMovement = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.initial_rank, 10);
        assert_eq!(decoded.refined_rank, 2);
        assert_eq!(decoded.delta, -8);
        assert_eq!(decoded.reason, "major promotion");
    }

    #[test]
    fn hit_explanation_display_with_movement() {
        let explanation = HitExplanation {
            final_score: 0.05,
            components: vec![ScoreComponent {
                source: ExplainedSource::SemanticFast {
                    embedder: "potion".into(),
                    cosine_sim: 0.9,
                },
                raw_score: 0.9,
                normalized_score: 0.95,
                rrf_contribution: 0.016,
                weight: 0.5,
            }],
            phase: ExplanationPhase::Refined,
            rank_movement: Some(RankMovement {
                initial_rank: 7,
                refined_rank: 1,
                delta: -6,
                reason: "boosted".into(),
            }),
        };
        let display = explanation.to_string();
        assert!(display.contains("Refined"));
        assert!(display.contains("Rank:"));
        assert!(display.contains("promoted"));
        assert!(display.contains("#7 -> #1"));
    }

    #[test]
    fn hit_explanation_clone() {
        let explanation = HitExplanation {
            final_score: 0.03,
            components: vec![ScoreComponent {
                source: ExplainedSource::LexicalBm25 {
                    matched_terms: vec!["clone".into()],
                    tf: 1.0,
                    idf: 2.0,
                },
                raw_score: 4.0,
                normalized_score: 0.4,
                rrf_contribution: 0.01,
                weight: 0.3,
            }],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };
        assert!((explanation.final_score - 0.03).abs() < f64::EPSILON);
        assert_eq!(explanation.source_count(), 1);
        assert_eq!(explanation.phase, ExplanationPhase::Initial);
        assert!(explanation.rank_movement.is_none());
    }

    #[test]
    fn hit_explanation_debug() {
        let explanation = HitExplanation {
            final_score: 0.01,
            components: vec![],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };
        let dbg = format!("{explanation:?}");
        assert!(dbg.contains("HitExplanation"));
        assert!(dbg.contains("final_score"));
    }

    #[test]
    fn negative_cosine_sim() {
        let source = ExplainedSource::SemanticFast {
            embedder: "model".into(),
            cosine_sim: -0.35,
        };
        let display = source.to_string();
        assert!(display.contains("-0.3500"));

        let component = ScoreComponent {
            source,
            raw_score: -0.35,
            normalized_score: 0.0,
            rrf_contribution: 0.001,
            weight: 0.3,
        };
        let json = serde_json::to_string(&component).unwrap();
        let decoded: ScoreComponent = serde_json::from_str(&json).unwrap();
        assert!((decoded.raw_score - (-0.35)).abs() < f64::EPSILON);
    }

    #[test]
    fn many_components() {
        let components: Vec<ScoreComponent> = (0..100)
            .map(|i| ScoreComponent {
                source: ExplainedSource::LexicalBm25 {
                    matched_terms: vec![format!("term{i}")],
                    tf: f64::from(i),
                    idf: 1.0,
                },
                raw_score: f64::from(i),
                normalized_score: f64::from(i) / 100.0,
                rrf_contribution: 1.0 / (60.0 + f64::from(i) + 1.0),
                weight: 0.3,
            })
            .collect();

        let explanation = HitExplanation {
            final_score: 0.5,
            components,
            phase: ExplanationPhase::Refined,
            rank_movement: None,
        };
        assert_eq!(explanation.source_count(), 100);
        assert!(explanation.total_rrf_contribution() > 0.0);

        // Display should render all 100 components
        let display = explanation.to_string();
        assert!(display.contains("[99]"));
    }

    #[test]
    fn serde_explanation_no_movement() {
        let explanation = HitExplanation {
            final_score: 0.0,
            components: vec![],
            phase: ExplanationPhase::Initial,
            rank_movement: None,
        };
        let json = serde_json::to_string(&explanation).unwrap();
        let decoded: HitExplanation = serde_json::from_str(&json).unwrap();
        assert!(decoded.rank_movement.is_none());
        assert_eq!(decoded.phase, ExplanationPhase::Initial);
        assert!((decoded.final_score).abs() < f64::EPSILON);
        assert!(decoded.components.is_empty());
    }

    // ─── bd-7fn8 tests end ───
}
