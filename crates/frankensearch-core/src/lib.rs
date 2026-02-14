//! Core traits, types, and error types for the frankensearch hybrid search library.
//!
//! This crate defines the shared interfaces (`Embedder`, `Reranker`, `LexicalSearch`),
//! result types (`ScoredResult`, `VectorHit`, `FusedHit`), error types (`SearchError`),
//! text canonicalization, and query classification used across all frankensearch crates.
//!
//! It has minimal external dependencies and is intended to be depended on by every
//! other crate in the workspace.

pub mod cache;
pub mod canonicalize;
pub mod collectors;
pub mod config;
pub mod decision_plane;
pub mod error;
pub mod explanation;
pub mod filter;
pub mod fingerprint;
pub mod parsed_query;
pub mod query_class;
pub mod tracing_config;
pub mod traits;
pub mod types;

pub use cache::{CachePolicy, NoCache, S3FifoCache, S3FifoConfig};
pub use canonicalize::{Canonicalizer, DefaultCanonicalizer};
pub use collectors::{
    CollectorConfig, CollectorSnapshot, DEFAULT_COLLECTION_INTERVAL_MS,
    DEFAULT_SEARCH_STREAM_CAPACITY, EmbedderTier, EmbeddingCollectorSample, EmbeddingStage,
    EmbeddingStatus, IndexCollectorSample, IndexInventory, IndexOperation, IndexStatus,
    LifecycleSeverity, LifecycleState, LiveSearchFrame, LiveSearchStreamEmitter,
    MIN_COLLECTION_INTERVAL_MS, PressureProfile, QuantizationMode, ResourceCollectorSample,
    RuntimeMetricsCollector, SearchCollectorSample, SearchEventPhase, SearchStreamConfig,
    SearchStreamHealth, SearchStreamMode, SearchStreamPublishOutcome, TELEMETRY_SCHEMA_VERSION,
    TelemetryCorrelation, TelemetryEmbedderInfo, TelemetryEmbeddingJob, TelemetryEnvelope,
    TelemetryEvent, TelemetryInstance, TelemetryQueryClass, TelemetryResourceSample,
    TelemetrySearchMetrics, TelemetrySearchQuery, TelemetrySearchResults,
};
pub use config::{TwoTierConfig, TwoTierMetrics};
pub use decision_plane::{
    CalibrationFallbackReason, CalibrationStatus, CalibrationThresholds, DecisionContext,
    DecisionOutcome, EvidenceEventType, EvidenceRecord, ExhaustionPolicy, LossVector, LossWeights,
    PipelineAction, PipelineState, ReasonCode, ResourceBudget, ResourceUsage, Severity,
};
pub use error::{SearchError, SearchResult};
pub use explanation::{
    ExplainedSource, ExplanationPhase, HitExplanation, RankMovement, ScoreComponent,
};
pub use filter::{
    BitsetFilter, DateRangeFilter, DocTypeFilter, FilterChain, FilterMode, PredicateFilter,
    SearchFilter, fnv1a_hash,
};
pub use fingerprint::{
    DEFAULT_SEMANTIC_CHANGE_THRESHOLD, DocumentFingerprint, SIGNIFICANT_CHAR_COUNT_CHANGE_THRESHOLD,
};
pub use parsed_query::ParsedQuery;
pub use query_class::QueryClass;
pub use traits::{
    Embedder, LexicalSearch, MetricsExporter, ModelCategory, ModelInfo, ModelTier,
    NoOpMetricsExporter, RerankDocument, RerankScore, Reranker, SearchFuture,
    SharedMetricsExporter, cosine_similarity, l2_normalize, truncate_embedding,
};
pub use types::{
    EmbeddingMetrics, FusedHit, IndexMetrics, IndexableDocument, PhaseMetrics, RankChanges,
    ScoreSource, ScoredResult, SearchMetrics, SearchMode, SearchPhase, VectorHit,
};
