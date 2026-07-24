//! Core traits, types, and error types for the frankensearch hybrid search library.
//!
//! This crate defines the shared interfaces (`Embedder`, `Reranker`, `LexicalSearch`),
//! result types (`ScoredResult`, `VectorHit`, `FusedHit`), error types (`SearchError`),
//! text canonicalization, and query classification used across all frankensearch crates.
//!
//! It has minimal external dependencies and is intended to be depended on by every
//! other crate in the workspace.

pub mod activation;
#[cfg(feature = "bench-internals")]
pub mod bench_support;
pub mod cache;
pub mod canonicalize;
pub mod collectors;
pub mod commit_replay;
pub mod config;
pub mod contract_sanity;
pub mod daemon;
pub mod decision_plane;
pub mod distributed_observability;
pub mod e2e_artifact;
pub mod error;
pub mod explanation;
pub mod filter;
pub mod fingerprint;
pub mod generation;
pub mod graph;
pub mod host_adapter;
pub mod metrics_eval;
pub mod observability_lint;
pub mod parsed_query;
pub mod query_class;
pub mod repair;
pub mod shadow;
pub mod simd;
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
pub use config::{FusionStrategy, TwoTierConfig, TwoTierMetrics};
pub use contract_sanity::{
    AdapterContractResult, CompatibilityStatus, ContractSanityChecker, ContractSanityReport,
    ContractViolationDiagnostic, MAX_SCHEMA_VERSION_LAG, ViolationSeverity, classify_version,
    classify_version_against, replay_command_for_reason,
};
pub use daemon::{DaemonClient, DaemonError, DaemonRetryConfig, apply_jitter, next_request_id};
pub use decision_plane::{
    CalibrationFallbackReason, CalibrationStatus, CalibrationThresholds, DecisionContext,
    DecisionOutcome, EvidenceEventType, EvidenceRecord, ExhaustionPolicy, LossVector, LossWeights,
    PipelineAction, PipelineState, ReasonCode, ResourceBudget, ResourceUsage, Severity,
};
pub use distributed_observability::{
    DistributedEvent, DistributedMetrics, emit_event, service_state_label,
};
pub use e2e_artifact::{
    ArtifactEmissionInput, ArtifactEntry, ClockMode, Correlation, DeterminismTier, DiffEntry,
    E2E_ARTIFACT_ARTIFACTS_INDEX_JSON, E2E_ARTIFACT_ENV_JSON, E2E_ARTIFACT_MANIFEST_JSON,
    E2E_ARTIFACT_REPLAY_COMMAND_TXT, E2E_ARTIFACT_REPRO_LOCK, E2E_ARTIFACT_STRUCTURED_EVENTS_JSONL,
    E2E_ARTIFACT_TERMINAL_TRANSCRIPT_TXT, E2E_SCHEMA_EVENT, E2E_SCHEMA_MANIFEST,
    E2E_SCHEMA_ORACLE_REPORT, E2E_SCHEMA_REPLAY, E2E_SCHEMA_SNAPSHOT_DIFF, E2E_SCHEMA_VERSION,
    E2eArtifactEmitterError, E2eArtifactValidationError, E2eEnvelope, E2eEventType, E2eOutcome,
    E2eSeverity, EventBody, ExitStatus, LaneReport, ManifestBody, ModelVersion, OracleReportBody,
    OracleVerdictRecord, Platform, ReplayBody, ReplayEventType, ReportTotals, SnapshotDiffBody,
    Suite, build_artifact_entries, build_core_manifest_artifacts, normalize_artifact_file_name,
    normalize_replay_command, render_artifacts_index, sha256_checksum, validate_envelope,
    validate_event_body, validate_event_envelope, validate_manifest_body,
    validate_manifest_envelope,
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
pub use graph::{DocumentGraph, EdgeType, GraphDocId, GraphEdge};
pub use host_adapter::{
    AdapterIdentity, AdapterLifecycleEvent, AdapterSink, CanonicalHostProject, ConformanceConfig,
    ConformanceHarness, ConformanceReport, ConformanceViolation,
    DEFAULT_REDACTION_FORBIDDEN_PATTERNS, ForwardingHostAdapter, HostAdapter, NoopAdapterSink,
};
pub use metrics_eval::{
    BootstrapCi, BootstrapComparison, QualityComparison, QualityMetric, QualityMetricComparison,
    QualityMetricSamples, RunStabilityVerdict, bootstrap_ci, bootstrap_compare,
    coefficient_of_variation, detect_outliers_iqr, map_at_k, mrr, ndcg_at_k, quality_comparison,
    recall_at_k, trim_outliers, verify_run_stability,
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
pub use shadow::{
    AlwaysAdmitShadowLoad, AtomicShadowLoadProbe, SHADOW_ARTIFACT_DIRECTORY,
    SHADOW_DEGRADATION_SCHEMA_VERSION, SHADOW_DEGRADATIONS_FILE, SHADOW_DIVERGENCE_SCHEMA_VERSION,
    SHADOW_DIVERGENCES_FILE, SHADOW_OBSERVATION_SCHEMA_VERSION, SHADOW_OBSERVATIONS_FILE,
    SHADOW_SAMPLE_DENOMINATOR, ShadowArtifactSummary, ShadowDegradationKind,
    ShadowDegradationRecord, ShadowDivergenceClass, ShadowDivergenceRecord, ShadowDocument,
    ShadowLexical, ShadowLexicalConfig, ShadowLexicalObserver, ShadowLoadProbe,
    ShadowObservationRecord, ShadowQuery, ShadowRankedHit, ShadowShedReason, ShadowStatus,
    append_shadow_degradation, compute_shadow_corpus_hash,
};
pub use simd::scale_f32_in_place;
pub use time_travel::{GenerationHistory, RetainedGeneration, RetentionPolicy, TimeTravelResult};

pub use asupersync::Cx;
pub use query_class::QueryClass;
pub use traits::{
    Embedder, LexicalSearch, MetricsExporter, ModelCategory, ModelInfo, ModelTier,
    NoOpMetricsExporter, RerankDocument, RerankScore, Reranker, SearchFuture,
    SharedMetricsExporter, SyncEmbed, SyncEmbedderAdapter, SyncRerank, SyncRerankerAdapter,
    cosine_similarity, l2_normalize, truncate_embedding,
};
pub use types::{
    DocId, EmbeddingMetrics, FusedHit, IndexMetrics, IndexableDocument, PhaseMetrics, RankChanges,
    ScoreSource, ScoredResult, SearchMetrics, SearchMode, SearchPhase, VectorHit,
};
