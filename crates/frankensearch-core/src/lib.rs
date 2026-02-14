//! Core traits, types, and error types for the frankensearch hybrid search library.
//!
//! This crate defines the shared interfaces (`Embedder`, `Reranker`, `LexicalSearch`),
//! result types (`ScoredResult`, `VectorHit`, `FusedHit`), error types (`SearchError`),
//! text canonicalization, and query classification used across all frankensearch crates.
//!
//! It has minimal external dependencies and is intended to be depended on by every
//! other crate in the workspace.

pub mod activation;
pub mod cache;
pub mod canonicalize;
pub mod collectors;
pub mod commit_replay;
pub mod config;
pub mod contract_sanity;
pub mod decision_plane;
pub mod distributed_observability;
pub mod e2e_artifact;
pub mod error;
pub mod explanation;
pub mod filter;
pub mod fingerprint;
pub mod generation;
pub mod host_adapter;
pub mod metrics_eval;
pub mod observability_lint;
pub mod parsed_query;
pub mod query_class;
pub mod repair;
pub mod time_travel;
pub mod tracing_config;
pub mod traits;
pub mod types;

pub use activation::{
    ActiveGeneration, ArtifactVerification, ArtifactVerifier, GenerationController, InvariantCheck,
    check_invariants,
};
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
pub use commit_replay::{
    CommitEntry, CommitOutcome, CommitReplayEngine, DocumentOp, ReplayConsumer, ReplayPolicy,
    ReplayWatermark, SkipReason,
};
pub use config::{TwoTierConfig, TwoTierMetrics};
pub use contract_sanity::{
    AdapterContractResult, CompatibilityStatus, ContractSanityChecker, ContractSanityReport,
    ContractViolationDiagnostic, MAX_SCHEMA_VERSION_LAG, ViolationSeverity, classify_version,
    classify_version_against, replay_command_for_reason,
};
pub use decision_plane::{
    CalibrationFallbackReason, CalibrationStatus, CalibrationThresholds, DecisionContext,
    DecisionOutcome, EvidenceEventType, EvidenceRecord, ExhaustionPolicy, LossVector, LossWeights,
    PipelineAction, PipelineState, ReasonCode, ResourceBudget, ResourceUsage, Severity,
};
pub use distributed_observability::{
    DistributedEvent, DistributedMetrics, emit_event, service_state_label,
};
pub use e2e_artifact::{
    ArtifactEntry, ClockMode, Correlation, DeterminismTier, DiffEntry, E2E_SCHEMA_VERSION,
    E2eEnvelope, E2eEventType, E2eOutcome, E2eSeverity, EventBody, ExitStatus, LaneReport,
    ManifestBody, ModelVersion, OracleReportBody, OracleVerdictRecord, Platform, ReplayBody,
    ReplayEventType, ReportTotals, SnapshotDiffBody, Suite,
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
pub use generation::{
    ActivationInvariant, CommitRange, EmbedderRevision, EmbedderTierTag, GenerationManifest,
    InvariantKind, LexicalArtifact, MANIFEST_SCHEMA_VERSION, QuantizationFormat, RepairDescriptor,
    ValidationFinding, ValidationResult, VectorArtifact, compute_manifest_hash, require_valid,
    validate_manifest,
};
pub use host_adapter::{
    AdapterIdentity, AdapterLifecycleEvent, ConformanceConfig, ConformanceHarness,
    ConformanceReport, ConformanceViolation, DEFAULT_REDACTION_FORBIDDEN_PATTERNS, HostAdapter,
};
pub use observability_lint::{
    LintFinding, LintReport, LintRuleId, LintSeverity, lint_component_coverage, lint_record,
    lint_stream,
};
pub use parsed_query::ParsedQuery;
pub use repair::{
    CorruptionEvent, CorruptionPolicy, DegradedReason, DetectionMethod, RepairAttempt,
    RepairOrchestrator, RepairOutcome, RepairProvider, ServiceState,
};
pub use time_travel::{GenerationHistory, RetainedGeneration, RetentionPolicy, TimeTravelResult};

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
