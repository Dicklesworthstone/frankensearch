use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::SearchError;
use crate::query_class::QueryClass;

// ---------------------------------------------------------------------------
// Document types
// ---------------------------------------------------------------------------

/// A document to be indexed for search.
///
/// This is the input type consumed by both vector indexing and lexical indexing.
/// It intentionally does NOT carry computed data (embeddings, BM25 scores) --
/// those are produced during indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexableDocument {
    /// Unique document identifier (caller-defined).
    pub id: String,
    /// Main searchable text content.
    pub content: String,
    /// Optional title (receives BM25 boost in lexical search).
    pub title: Option<String>,
    /// Extensible key-value metadata. Stored alongside results and available
    /// in `ScoredResult.metadata` after search.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl IndexableDocument {
    /// Creates a new document with the required fields.
    #[must_use]
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the optional title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Adds a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Search result types
// ---------------------------------------------------------------------------

/// A raw hit from vector similarity search.
///
/// Produced by the vector index before fusion. Scores are raw cosine similarity
/// values (not normalized), typically in the range \[-1.0, 1.0\].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    /// Positional index into the vector store (used for fast lookup).
    pub index: u32,
    /// Raw cosine similarity score.
    pub score: f32,
    /// Document identifier resolved from the index.
    pub doc_id: String,
}

impl VectorHit {
    /// Ordering by score descending with NaN-safe semantics.
    /// NaN sorts below all real values (treated as worst possible score).
    #[must_use]
    pub fn cmp_by_score(&self, other: &Self) -> std::cmp::Ordering {
        // Map NaN to NEG_INFINITY so it sorts last in descending order.
        let a = if self.score.is_nan() {
            f32::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other.score.is_nan() {
            f32::NEG_INFINITY
        } else {
            other.score
        };
        // Descending: higher scores first.
        b.total_cmp(&a)
    }
}

/// A hit from hybrid fusion (lexical + semantic combined via RRF).
///
/// RRF scores are computed in f64 for precision during accumulation of many
/// small `1/(K+rank+1)` values, then carried as f64 throughout fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedHit {
    /// Document identifier.
    pub doc_id: String,
    /// RRF-fused score (f64 for precision during fusion).
    pub rrf_score: f64,
    /// Rank in the lexical (BM25) source, if present.
    pub lexical_rank: Option<usize>,
    /// Rank in the semantic (vector) source, if present.
    pub semantic_rank: Option<usize>,
    /// Raw BM25 score from lexical search, if applicable.
    pub lexical_score: Option<f32>,
    /// Raw cosine similarity from semantic search, if applicable.
    pub semantic_score: Option<f32>,
    /// True if this document appeared in both lexical and semantic results.
    pub in_both_sources: bool,
}

impl FusedHit {
    /// Four-level deterministic tie-breaking for RRF results:
    /// 1. Higher RRF score first
    /// 2. Documents in both sources preferred
    /// 3. Higher lexical score preferred
    /// 4. Lexicographic doc\_id (deterministic fallback)
    #[must_use]
    pub fn cmp_for_ranking(&self, other: &Self) -> std::cmp::Ordering {
        // 1. RRF score descending
        other
            .rrf_score
            .total_cmp(&self.rrf_score)
            // 2. in_both_sources preferred (true > false)
            .then(other.in_both_sources.cmp(&self.in_both_sources))
            // 3. Lexical score descending (treat None as -inf)
            .then_with(|| {
                let a = self.lexical_score.unwrap_or(f32::NEG_INFINITY);
                let b = other.lexical_score.unwrap_or(f32::NEG_INFINITY);
                b.total_cmp(&a)
            })
            // 4. doc_id ascending (deterministic)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Which search backend produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreSource {
    /// Lexical (BM25) search only.
    Lexical,
    /// Fast-tier semantic search only.
    SemanticFast,
    /// Quality-tier semantic search only.
    SemanticQuality,
    /// Hybrid fusion (lexical + semantic via RRF).
    Hybrid,
    /// Result was reranked by cross-encoder.
    Reranked,
}

/// The final scored search result delivered to consumers.
///
/// Intentionally does NOT carry document text. Text is expensive and most
/// consumers only need `doc_id` + scores. When text is needed (e.g., for
/// reranking or display), look it up from your document store via `doc_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredResult {
    /// Unique document identifier.
    pub doc_id: String,
    /// Primary relevance score (RRF or blended, truncated to f32).
    pub score: f32,
    /// Which search backend produced this result.
    pub source: ScoreSource,
    /// Score from fast-tier semantic search, if applicable.
    pub fast_score: Option<f32>,
    /// Score from quality-tier semantic search, if applicable.
    pub quality_score: Option<f32>,
    /// BM25 score from lexical search, if applicable.
    pub lexical_score: Option<f32>,
    /// Cross-encoder score from reranking, if applicable.
    pub rerank_score: Option<f32>,
    /// Arbitrary document metadata (from index stored fields).
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Search mode and phases
// ---------------------------------------------------------------------------

/// Search mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchMode {
    /// BM25 keyword matching only.
    Lexical,
    /// Embedding similarity only.
    Semantic,
    /// RRF fusion of lexical + semantic.
    Hybrid,
    /// Progressive: fast semantic -> quality refinement + lexical fusion.
    TwoTier,
}

/// Diagnostic metrics for a search phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Which embedder was used for this phase.
    pub embedder_id: String,
    /// Number of vectors searched.
    pub vectors_searched: usize,
    /// Number of lexical candidates retrieved.
    pub lexical_candidates: usize,
    /// Number of results after fusion.
    pub fused_count: usize,
}

/// Structured telemetry for a completed search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Which search mode ran.
    pub mode: SearchMode,
    /// Query class selected by the classifier, when available.
    pub query_class: Option<QueryClass>,
    /// End-to-end latency for the search request.
    pub total_latency_ms: f64,
    /// Latency for phase 1 (`Initial`) when available.
    pub phase1_latency_ms: Option<f64>,
    /// Latency for phase 2 (`Refined`) when available.
    pub phase2_latency_ms: Option<f64>,
    /// Number of results returned to the caller.
    pub result_count: usize,
    /// Number of lexical candidates retrieved.
    pub lexical_candidates: usize,
    /// Number of semantic candidates retrieved.
    pub semantic_candidates: usize,
    /// Whether quality refinement was applied.
    pub refined: bool,
}

/// Structured telemetry for an embedding operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetrics {
    /// Stable embedder identifier (for example, `potion-multilingual-128M`).
    pub embedder_id: String,
    /// Number of texts embedded in this operation.
    pub batch_size: usize,
    /// Embedding operation latency.
    pub duration_ms: f64,
    /// Embedding vector dimension.
    pub dimension: usize,
    /// Whether this embedder is semantically meaningful.
    pub is_semantic: bool,
}

/// Structured telemetry for index update operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetrics {
    /// Document count after the update.
    pub doc_count: usize,
    /// Total on-disk index size after the update.
    pub index_size_bytes: u64,
    /// Number of documents added or modified by this update.
    pub updated_docs: usize,
    /// Whether the index was marked stale during this update.
    pub staleness_detected: bool,
}

/// Tracks how rankings changed between initial and refined phases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankChanges {
    /// Documents that moved up in ranking after refinement.
    pub promoted: usize,
    /// Documents that moved down in ranking after refinement.
    pub demoted: usize,
    /// Documents whose rank did not change.
    pub stable: usize,
}

impl RankChanges {
    /// Total number of documents tracked.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.promoted + self.demoted + self.stable
    }
}

/// Progressive search phases for two-tier display.
///
/// The iterator contract:
/// 1. Always yields `Initial` first (~15ms).
/// 2. Then yields either `Refined` or `RefinementFailed` (never both).
/// 3. Iterator is fused after yielding 2 phases (`next()` returns `None`).
///
/// Consumers can stop after `Initial` if latency-sensitive.
///
/// # Example
///
/// ```rust,ignore
/// for phase in searcher.search("distributed consensus", 10) {
///     match phase {
///         SearchPhase::Initial { results, .. } => display_immediately(&results),
///         SearchPhase::Refined { results, .. } => update_display(&results),
///         SearchPhase::RefinementFailed { initial_results, error, .. } => {
///             // Keep showing initial results, log the error
///             log_warning(&error);
///         }
///     }
/// }
/// ```
#[derive(Debug)]
pub enum SearchPhase {
    /// Fast-tier results ready for immediate display.
    ///
    /// Contains RRF-fused results from fast embedding + BM25.
    /// Scores are RRF values (~0.01-0.03 range), sorted descending
    /// with deterministic tie-breaking.
    Initial {
        /// Fast-tier search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase.
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
    },

    /// Quality-refined results ready to replace initial display.
    ///
    /// Contains blended scores (0.7 quality + 0.3 fast by default).
    /// Results may have different ordering than `Initial`.
    /// `rerank_score` is `Some(_)` if the reranker was applied.
    Refined {
        /// Quality-refined search results.
        results: Vec<ScoredResult>,
        /// Time elapsed for this phase.
        latency: Duration,
        /// Diagnostic metrics for this phase.
        metrics: PhaseMetrics,
        /// How rankings changed compared to Initial.
        rank_changes: RankChanges,
    },

    /// Quality refinement failed; initial results remain valid.
    ///
    /// This is NOT an error state -- it is graceful degradation.
    /// The consumer should display `initial_results` and log the error.
    RefinementFailed {
        /// The original `Initial` results, carried forward unchanged.
        initial_results: Vec<ScoredResult>,
        /// Why refinement failed (timeout, model error, etc.).
        error: SearchError,
        /// How long we waited before failing.
        latency: Duration,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexable_document_builder() {
        let doc = IndexableDocument::new("doc-1", "Hello world")
            .with_title("Greeting")
            .with_metadata("source", "test");

        assert_eq!(doc.id, "doc-1");
        assert_eq!(doc.content, "Hello world");
        assert_eq!(doc.title.as_deref(), Some("Greeting"));
        assert_eq!(doc.metadata.get("source").map(String::as_str), Some("test"));
    }

    #[test]
    fn indexable_document_minimal() {
        let doc = IndexableDocument::new("id", "text");
        assert!(doc.title.is_none());
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn vector_hit_nan_safe_ordering() {
        let hit_a = VectorHit {
            index: 0,
            score: 0.9,
            doc_id: "a".into(),
        };
        let hit_nan = VectorHit {
            index: 1,
            score: f32::NAN,
            doc_id: "b".into(),
        };
        // NaN should sort below real values (hit_a should come first).
        assert_eq!(
            hit_a.cmp_by_score(&hit_nan),
            std::cmp::Ordering::Less // a comes first (better score)
        );
    }

    #[test]
    fn fused_hit_tie_breaking() {
        let hit_both = FusedHit {
            doc_id: "a".into(),
            rrf_score: 0.02,
            lexical_rank: Some(1),
            semantic_rank: Some(3),
            lexical_score: Some(5.0),
            semantic_score: Some(0.8),
            in_both_sources: true,
        };
        let hit_semantic_only = FusedHit {
            doc_id: "b".into(),
            rrf_score: 0.02, // Same RRF score
            lexical_rank: None,
            semantic_rank: Some(2),
            lexical_score: None,
            semantic_score: Some(0.9),
            in_both_sources: false,
        };
        // Same RRF -> in_both_sources wins
        assert_eq!(
            hit_both.cmp_for_ranking(&hit_semantic_only),
            std::cmp::Ordering::Less // hit_both ranks first
        );
    }

    #[test]
    fn fused_hit_rrf_score_dominates() {
        let high = FusedHit {
            doc_id: "a".into(),
            rrf_score: 0.03,
            lexical_rank: None,
            semantic_rank: Some(1),
            lexical_score: None,
            semantic_score: Some(0.9),
            in_both_sources: false,
        };
        let low = FusedHit {
            doc_id: "b".into(),
            rrf_score: 0.01,
            lexical_rank: Some(1),
            semantic_rank: Some(1),
            lexical_score: Some(10.0),
            semantic_score: Some(0.99),
            in_both_sources: true,
        };
        // Higher RRF always wins regardless of other fields.
        assert_eq!(
            high.cmp_for_ranking(&low),
            std::cmp::Ordering::Less // high ranks first
        );
    }

    #[test]
    fn fused_hit_deterministic_doc_id_tiebreak() {
        let a = FusedHit {
            doc_id: "alpha".into(),
            rrf_score: 0.02,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        let b = FusedHit {
            doc_id: "beta".into(),
            rrf_score: 0.02,
            lexical_rank: None,
            semantic_rank: None,
            lexical_score: None,
            semantic_score: None,
            in_both_sources: false,
        };
        // All else equal -> lexicographic doc_id ascending.
        assert_eq!(a.cmp_for_ranking(&b), std::cmp::Ordering::Less);
    }

    #[test]
    fn scored_result_serde_roundtrip() {
        let result = ScoredResult {
            doc_id: "doc-42".into(),
            score: 0.85,
            source: ScoreSource::Hybrid,
            fast_score: Some(0.7),
            quality_score: Some(0.9),
            lexical_score: Some(12.5),
            rerank_score: None,
            metadata: Some(serde_json::json!({"tags": ["rust", "search"]})),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let roundtripped: ScoredResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtripped.doc_id, "doc-42");
        assert!((roundtripped.score - 0.85).abs() < f32::EPSILON);
        assert_eq!(roundtripped.source, ScoreSource::Hybrid);
        assert!(roundtripped.metadata.is_some());
    }

    #[test]
    fn rank_changes_total() {
        let changes = RankChanges {
            promoted: 3,
            demoted: 2,
            stable: 5,
        };
        assert_eq!(changes.total(), 10);
    }

    #[test]
    fn search_phase_initial_construction() {
        let phase = SearchPhase::Initial {
            results: vec![],
            latency: Duration::from_millis(12),
            metrics: PhaseMetrics {
                embedder_id: "potion-128M".into(),
                vectors_searched: 1000,
                lexical_candidates: 50,
                fused_count: 10,
            },
        };
        if let SearchPhase::Initial { latency, .. } = phase {
            assert_eq!(latency, Duration::from_millis(12));
        }
    }

    #[test]
    fn search_phase_refinement_failed_carries_results() {
        let initial = vec![ScoredResult {
            doc_id: "doc-1".into(),
            score: 0.5,
            source: ScoreSource::Hybrid,
            fast_score: None,
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            metadata: None,
        }];
        let phase = SearchPhase::RefinementFailed {
            initial_results: initial,
            error: SearchError::SearchTimeout {
                elapsed_ms: 500,
                budget_ms: 300,
            },
            latency: Duration::from_millis(500),
        };
        if let SearchPhase::RefinementFailed {
            initial_results, ..
        } = phase
        {
            assert_eq!(initial_results.len(), 1);
            assert_eq!(initial_results[0].doc_id, "doc-1");
        }
    }

    #[test]
    fn indexable_document_serde_roundtrip() {
        let doc = IndexableDocument::new("id-1", "Hello world")
            .with_title("Test")
            .with_metadata("lang", "en");
        let json = serde_json::to_string(&doc).expect("serialize");
        let rt: IndexableDocument = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(rt.id, "id-1");
        assert_eq!(rt.title.as_deref(), Some("Test"));
        assert_eq!(rt.metadata.get("lang").map(String::as_str), Some("en"));
    }

    #[test]
    fn search_metrics_serde_roundtrip() {
        let metrics = SearchMetrics {
            mode: SearchMode::Hybrid,
            query_class: Some(QueryClass::NaturalLanguage),
            total_latency_ms: 12.5,
            phase1_latency_ms: Some(5.4),
            phase2_latency_ms: Some(7.1),
            result_count: 10,
            lexical_candidates: 60,
            semantic_candidates: 45,
            refined: true,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let decoded: SearchMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.mode, SearchMode::Hybrid);
        assert_eq!(decoded.query_class, Some(QueryClass::NaturalLanguage));
        assert!((decoded.total_latency_ms - 12.5).abs() < f64::EPSILON);
        assert_eq!(decoded.result_count, 10);
        assert!(decoded.refined);
    }

    #[test]
    fn embedding_metrics_serde_roundtrip() {
        let metrics = EmbeddingMetrics {
            embedder_id: "potion-multilingual-128M".into(),
            batch_size: 32,
            duration_ms: 1.9,
            dimension: 256,
            is_semantic: true,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let decoded: EmbeddingMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.embedder_id, "potion-multilingual-128M");
        assert_eq!(decoded.batch_size, 32);
        assert_eq!(decoded.dimension, 256);
        assert!(decoded.is_semantic);
    }

    #[test]
    fn index_metrics_serde_roundtrip() {
        let metrics = IndexMetrics {
            doc_count: 1_000,
            index_size_bytes: 12_345_678,
            updated_docs: 11,
            staleness_detected: false,
        };

        let json = serde_json::to_string(&metrics).expect("serialize");
        let decoded: IndexMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.doc_count, 1_000);
        assert_eq!(decoded.index_size_bytes, 12_345_678);
        assert_eq!(decoded.updated_docs, 11);
        assert!(!decoded.staleness_detected);
    }

    // ─── bd-ta55 tests begin ───

    #[test]
    fn indexable_document_multiple_metadata() {
        let doc = IndexableDocument::new("d1", "content")
            .with_metadata("a", "1")
            .with_metadata("b", "2")
            .with_metadata("c", "3");
        assert_eq!(doc.metadata.len(), 3);
        assert_eq!(doc.metadata["a"], "1");
        assert_eq!(doc.metadata["b"], "2");
        assert_eq!(doc.metadata["c"], "3");
    }

    #[test]
    fn indexable_document_metadata_overwrite() {
        let doc = IndexableDocument::new("d1", "content")
            .with_metadata("key", "old")
            .with_metadata("key", "new");
        assert_eq!(doc.metadata.len(), 1);
        assert_eq!(doc.metadata["key"], "new");
    }

    #[test]
    fn indexable_document_clone_debug() {
        let doc = IndexableDocument::new("d1", "text").with_title("T");
        let cloned = doc.clone();
        assert_eq!(cloned.id, "d1");
        assert_eq!(cloned.title.as_deref(), Some("T"));
        let dbg = format!("{doc:?}");
        assert!(dbg.contains("IndexableDocument"));
        assert!(dbg.contains("d1"));
    }

    #[test]
    fn vector_hit_partial_eq() {
        let a = VectorHit {
            index: 0,
            score: 0.5,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 0,
            score: 0.5,
            doc_id: "a".into(),
        };
        assert_eq!(a, b);

        let c = VectorHit {
            index: 1,
            score: 0.5,
            doc_id: "a".into(),
        };
        assert_ne!(a, c);
    }

    #[test]
    fn vector_hit_serde_roundtrip() {
        let hit = VectorHit {
            index: 42,
            score: 0.95,
            doc_id: "doc-x".into(),
        };
        let json = serde_json::to_string(&hit).unwrap();
        let decoded: VectorHit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, hit);
    }

    #[test]
    fn vector_hit_both_nan() {
        let a = VectorHit {
            index: 0,
            score: f32::NAN,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 1,
            score: f32::NAN,
            doc_id: "b".into(),
        };
        // Both NaN -> treated as equal (both map to NEG_INFINITY)
        assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn vector_hit_equal_scores() {
        let a = VectorHit {
            index: 0,
            score: 0.75,
            doc_id: "a".into(),
        };
        let b = VectorHit {
            index: 1,
            score: 0.75,
            doc_id: "b".into(),
        };
        assert_eq!(a.cmp_by_score(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn vector_hit_descending_order() {
        let high = VectorHit {
            index: 0,
            score: 0.9,
            doc_id: "h".into(),
        };
        let low = VectorHit {
            index: 1,
            score: 0.1,
            doc_id: "l".into(),
        };
        // Descending: high comes first (Less means "before")
        assert_eq!(high.cmp_by_score(&low), std::cmp::Ordering::Less);
        assert_eq!(low.cmp_by_score(&high), std::cmp::Ordering::Greater);
    }

    #[test]
    fn fused_hit_serde_roundtrip() {
        let hit = FusedHit {
            doc_id: "fused-1".into(),
            rrf_score: 0.025,
            lexical_rank: Some(3),
            semantic_rank: Some(7),
            lexical_score: Some(8.5),
            semantic_score: Some(0.72),
            in_both_sources: true,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let decoded: FusedHit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.doc_id, "fused-1");
        assert!((decoded.rrf_score - 0.025).abs() < f64::EPSILON);
        assert!(decoded.in_both_sources);
        assert_eq!(decoded.lexical_rank, Some(3));
    }

    #[test]
    fn fused_hit_lexical_score_tiebreak() {
        // Same RRF, same in_both_sources -> lexical_score descending
        let high_lex = FusedHit {
            doc_id: "z".into(), // worse doc_id
            rrf_score: 0.02,
            lexical_rank: Some(1),
            semantic_rank: None,
            lexical_score: Some(15.0),
            semantic_score: None,
            in_both_sources: false,
        };
        let low_lex = FusedHit {
            doc_id: "a".into(), // better doc_id
            rrf_score: 0.02,
            lexical_rank: Some(5),
            semantic_rank: None,
            lexical_score: Some(3.0),
            semantic_score: None,
            in_both_sources: false,
        };
        // Higher lexical_score wins (level 3)
        assert_eq!(high_lex.cmp_for_ranking(&low_lex), std::cmp::Ordering::Less);
    }

    #[test]
    fn fused_hit_clone_debug() {
        let hit = FusedHit {
            doc_id: "test".into(),
            rrf_score: 0.01,
            lexical_rank: None,
            semantic_rank: Some(5),
            lexical_score: None,
            semantic_score: Some(0.6),
            in_both_sources: false,
        };
        let cloned = hit.clone();
        assert_eq!(cloned.doc_id, "test");
        let dbg = format!("{hit:?}");
        assert!(dbg.contains("FusedHit"));
    }

    #[test]
    fn score_source_all_variants_serde() {
        let variants = [
            ScoreSource::Lexical,
            ScoreSource::SemanticFast,
            ScoreSource::SemanticQuality,
            ScoreSource::Hybrid,
            ScoreSource::Reranked,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ScoreSource = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn score_source_clone_copy_eq() {
        let a = ScoreSource::Hybrid;
        let b = a; // Copy
        let c = a; // Copy (ScoreSource is Copy)
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_ne!(ScoreSource::Lexical, ScoreSource::Hybrid);
    }

    #[test]
    fn scored_result_all_none_optionals() {
        let result = ScoredResult {
            doc_id: "min".into(),
            score: 0.1,
            source: ScoreSource::Lexical,
            fast_score: None,
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            metadata: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: ScoredResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.fast_score.is_none());
        assert!(decoded.quality_score.is_none());
        assert!(decoded.rerank_score.is_none());
        assert!(decoded.metadata.is_none());
    }

    #[test]
    fn scored_result_all_some_optionals() {
        let result = ScoredResult {
            doc_id: "max".into(),
            score: 0.99,
            source: ScoreSource::Reranked,
            fast_score: Some(0.7),
            quality_score: Some(0.9),
            lexical_score: Some(15.0),
            rerank_score: Some(0.95),
            metadata: Some(serde_json::json!({"key": "value"})),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: ScoredResult = serde_json::from_str(&json).unwrap();
        assert!(decoded.fast_score.is_some());
        assert!(decoded.quality_score.is_some());
        assert!(decoded.lexical_score.is_some());
        assert!(decoded.rerank_score.is_some());
        assert!(decoded.metadata.is_some());
    }

    #[test]
    fn scored_result_clone_debug() {
        let result = ScoredResult {
            doc_id: "d".into(),
            score: 0.5,
            source: ScoreSource::SemanticFast,
            fast_score: Some(0.5),
            quality_score: None,
            lexical_score: None,
            rerank_score: None,
            metadata: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.doc_id, "d");
        let dbg = format!("{result:?}");
        assert!(dbg.contains("ScoredResult"));
        assert!(dbg.contains("SemanticFast"));
    }

    #[test]
    fn search_mode_all_variants_serde() {
        let variants = [
            SearchMode::Lexical,
            SearchMode::Semantic,
            SearchMode::Hybrid,
            SearchMode::TwoTier,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: SearchMode = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn phase_metrics_serde_clone_debug() {
        let metrics = PhaseMetrics {
            embedder_id: "test-embed".into(),
            vectors_searched: 500,
            lexical_candidates: 30,
            fused_count: 10,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        let decoded: PhaseMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.embedder_id, "test-embed");
        assert_eq!(decoded.vectors_searched, 500);

        let cloned = metrics.clone();
        assert_eq!(cloned.fused_count, 10);

        let dbg = format!("{metrics:?}");
        assert!(dbg.contains("PhaseMetrics"));
    }

    #[test]
    fn rank_changes_default_and_zero_total() {
        let changes = RankChanges::default();
        assert_eq!(changes.promoted, 0);
        assert_eq!(changes.demoted, 0);
        assert_eq!(changes.stable, 0);
        assert_eq!(changes.total(), 0);
    }

    #[test]
    fn rank_changes_serde_roundtrip() {
        let changes = RankChanges {
            promoted: 5,
            demoted: 3,
            stable: 12,
        };
        let json = serde_json::to_string(&changes).unwrap();
        let decoded: RankChanges = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.promoted, 5);
        assert_eq!(decoded.demoted, 3);
        assert_eq!(decoded.stable, 12);
        assert_eq!(decoded.total(), 20);
    }

    #[test]
    fn search_phase_refined_construction() {
        let phase = SearchPhase::Refined {
            results: vec![ScoredResult {
                doc_id: "r1".into(),
                score: 0.9,
                source: ScoreSource::Hybrid,
                fast_score: Some(0.7),
                quality_score: Some(0.9),
                lexical_score: None,
                rerank_score: None,
                metadata: None,
            }],
            latency: Duration::from_millis(120),
            metrics: PhaseMetrics {
                embedder_id: "minilm".into(),
                vectors_searched: 2000,
                lexical_candidates: 100,
                fused_count: 20,
            },
            rank_changes: RankChanges {
                promoted: 4,
                demoted: 2,
                stable: 14,
            },
        };
        if let SearchPhase::Refined {
            results,
            latency,
            rank_changes,
            ..
        } = phase
        {
            assert_eq!(results.len(), 1);
            assert_eq!(latency, Duration::from_millis(120));
            assert_eq!(rank_changes.total(), 20);
        } else {
            panic!("expected Refined variant");
        }
    }

    #[test]
    fn search_phase_debug() {
        let phase = SearchPhase::Initial {
            results: vec![],
            latency: Duration::from_millis(5),
            metrics: PhaseMetrics {
                embedder_id: "e".into(),
                vectors_searched: 0,
                lexical_candidates: 0,
                fused_count: 0,
            },
        };
        let dbg = format!("{phase:?}");
        assert!(dbg.contains("Initial"));
    }

    // ─── bd-ta55 tests end ───
}
