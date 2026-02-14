//! fsfs scaffold crate.
//!
//! This crate establishes the standalone fsfs binary surface with explicit
//! separation between reusable runtime/config logic and UX adapters.

#![forbid(unsafe_code)]

pub mod adapters;
pub mod agent_ergonomics;
pub mod catalog;
pub mod cli_e2e;
pub mod concurrency;
pub mod config;
pub mod evidence;
pub mod explanation_payload;
pub mod interaction_primitives;
pub mod lexical_pipeline;
pub mod lifecycle;
pub mod mount_info;
pub mod orchestration;
pub mod output_schema;
pub mod pressure;
pub mod pressure_sensing;
pub mod profiling;
pub mod query_execution;
pub mod query_planning;
pub mod ranking_priors;
pub mod redaction;
pub mod repro;
pub mod runtime;
pub mod shutdown;
pub mod stream_protocol;
pub mod watcher;

pub use adapters::cli::{
    CliCommand, CliInput, CommandSource, ConfigAction, OutputFormat, detect_auto_mode, exit_code,
    parse_cli_args,
};
pub use adapters::format_emitter::{
    emit_envelope, emit_envelope_string, emit_stream_frame, emit_stream_frame_string,
    meta_for_format, verify_json_toon_parity,
};
pub use adapters::tui::{
    ContextRetentionPolicy, ContextRetentionRule, FsfsScreen, FsfsTuiShellModel,
    TuiAdapterSettings, TuiKeyBindingScope, TuiKeyBindingSpec, TuiKeymapModel,
    TuiModelValidationError, TuiNavigationModel, TuiPaletteActionSpec, TuiPaletteCategory,
    TuiPaletteModel,
};
pub use agent_ergonomics::{
    CompactEnvelope, CompactError, CompactHit, CompactLevel, CompactSearchResponse,
    QUERY_TEMPLATE_VERSION, QueryTemplate, RESULT_ID_PREFIX, ResultIdEntry, ResultIdRegistry,
    TemplateParam, TemplateStep, builtin_templates, compactify, parse_result_id, result_id,
};
pub use catalog::{
    CATALOG_SCHEMA_VERSION, CHANGELOG_REPLAY_BATCH_SQL, CLEANUP_TOMBSTONES_SQL, CatalogChangeKind,
    CatalogIngestionClass, CatalogPipelineStatus, DIRTY_CATALOG_LOOKUP_SQL, INDEX_CATALOG_CLEANUP,
    INDEX_CATALOG_CONTENT_HASH, INDEX_CATALOG_DIRTY_LOOKUP, INDEX_CATALOG_REVISIONS,
    INDEX_CHANGELOG_FILE_REVISION, INDEX_CHANGELOG_PENDING_APPLY, INDEX_CHANGELOG_REPLAY,
    ReplayDecision, bootstrap_catalog_schema, classify_replay_sequence,
    current_catalog_schema_version,
};
pub use cli_e2e::{
    CLI_E2E_REASON_SCENARIO_DEGRADE, CLI_E2E_REASON_SCENARIO_PASS, CLI_E2E_REASON_SCENARIO_START,
    CLI_E2E_SCHEMA_VERSION, CliE2eArtifactBundle, CliE2eRunConfig, CliE2eScenario,
    CliE2eScenarioKind, build_default_cli_e2e_bundles, default_cli_e2e_scenarios,
    replay_command_for_scenario,
};
pub use concurrency::{
    AccessMode, ContentionMetrics, ContentionPolicy, ContentionSnapshot, LockLevel, LockOrderGuard,
    LockSentinel, PipelineStageAccess, ResourceId, ResourceToken, pipeline_access_matrix,
    read_sentinel, remove_sentinel, try_acquire_sentinel, write_sentinel,
};
pub use config::{
    CliOverrides, ConfigLoadResult, ConfigLoadedEvent, ConfigSource, ConfigWarning, Density,
    DiscoveryConfig, FsfsConfig, IndexingConfig, MountPolicyEntry, PRESSURE_PROFILE_VERSION,
    PROFILE_PRECEDENCE_CHAIN, PathExpansion, PressureConfig, PressureProfile,
    PressureProfileEffectiveSettings, PressureProfileField, PressureProfileOverrideDecision,
    PressureProfileOverridePolicy, PressureProfileResolution, PressureProfileResolutionDiagnostics,
    PrivacyConfig, ProfileOverrideSource, ProfileSchedulerMode, SearchConfig, StorageConfig,
    TextSelectionMode, TuiConfig, TuiTheme, default_config_file_path, emit_config_loaded,
    load_from_sources, load_from_str,
};
pub use evidence::{
    ALL_FSFS_REASON_CODES, FsfsEventFamily, FsfsEvidenceEvent, FsfsReasonCode, ScopeDecision,
    ScopeDecisionKind, TraceLink, ValidationResult, ValidationViolation, is_valid_fsfs_reason_code,
    validate_event,
};
pub use explanation_payload::{
    EXPLANATION_PAYLOAD_SCHEMA_VERSION, FsfsExplanationPayload, FusionContext,
    PolicyDecisionExplanation, PolicyDomain, RankMovementSnapshot, RankingExplanation,
    ScoreComponentBreakdown, ScoreComponentSource, TuiExplanationPanel,
};
pub use interaction_primitives::{
    CyclicFilter, INTERACTION_PRIMITIVES_SCHEMA_VERSION, InteractionBudget, InteractionCycleTiming,
    InteractionSnapshot, LayoutConstraint, LayoutDirection, LatencyPhase, PanelDescriptor,
    PanelFocusState, PanelRole, PhaseTiming, RenderTier, ScreenAction, ScreenLayout,
    VirtualizedListState, canonical_layout, fnv1a_64,
};
pub use lexical_pipeline::{
    InMemoryLexicalBackend, InMemoryLexicalEntry, LexicalAction, LexicalBatchStats, LexicalChunk,
    LexicalChunkPolicy, LexicalIndexBackend, LexicalMutation, LexicalMutationKind,
    LexicalPerformanceTargets, LexicalPipeline, LexicalToken, TARGET_INCREMENTAL_P95_LATENCY_MS,
    TARGET_INCREMENTAL_UPDATES_PER_SECOND, TARGET_INITIAL_DOCS_PER_SECOND, tokenize_lexical,
};
pub use lifecycle::{
    DaemonPhase, DaemonStatus, DiskBudgetAction, DiskBudgetPolicy, DiskBudgetSnapshot,
    DiskBudgetStage, HealthStatus, LifecycleTracker, LimitViolation, PidFile, PidFileContents,
    ResourceLimits, ResourceUsage, SubsystemHealth, SubsystemId, WatchdogConfig,
};
pub use mount_info::{
    ChangeDetectionStrategy, ErrorClass, FsCategory, MountEntry, MountOverride, MountPolicy,
    MountTable, ProbeResult, classify_fstype, classify_io_error, parse_proc_mounts, probe_mount,
    read_system_mounts,
};
pub use orchestration::{
    BackpressureMode, LaneBudget, OrchestrationPhase, OrchestrationState, QueuePolicy,
    QueuePushResult, ResumeToken, SchedulerMode, SchedulerPolicy, StartupBootstrapPlan, WorkItem,
    WorkKind,
};
pub use output_schema::{
    ALL_OUTPUT_ERROR_CODES, ALL_OUTPUT_WARNING_CODES, CompatibilityMode, ENVELOPE_FIELDS,
    FieldDescriptor, FieldPresence, OUTPUT_SCHEMA_MIN_SUPPORTED, OUTPUT_SCHEMA_VERSION,
    OutputEnvelope, OutputError, OutputErrorCode, OutputMeta, OutputWarning, OutputWarningCode,
    decode_envelope_toon, encode_envelope_toon, error_code_for, exit_code_for,
    is_version_compatible, output_error_from, validate_envelope,
};
pub use pressure::{
    HostPressureCollector, PressureController, PressureControllerConfig, PressureSignal,
    PressureSnapshot, PressureState, PressureTransition, ProcIoCounters,
};
pub use pressure_sensing::{
    ControlState, HostSampler, PressureSample, PressureSensor,
    PressureThresholds as SensingThresholds, SmoothedReadings, ThresholdPair,
};
pub use profiling::{
    ITERATION_REASON_ACCEPTED, ITERATION_REASON_MULTI_CHANGE, ITERATION_REASON_NO_CHANGE,
    IterationValidation, LeverSnapshot, OPPORTUNITY_MATRIX_SCHEMA_VERSION,
    OneLeverIterationProtocol, OpportunityCandidate, OpportunityMatrix,
    PROFILING_WORKFLOW_SCHEMA_VERSION, ProfileArtifact, ProfileKind, ProfileStep, ProfileWorkflow,
    RankedOpportunity,
};
pub use query_execution::{
    CancellationAction, CancellationDirective, CancellationPoint, DegradationOverride,
    DegradationStatus, DegradationTransition, DegradedRetrievalMode, FusedCandidate, FusionPolicy,
    LexicalCandidate, QueryExecutionOrchestrator, QueryExecutionPlan, RetrievalStage,
    SemanticCandidate, StagePlan,
};
pub use query_planning::{
    DEFAULT_LOW_CONFIDENCE_THRESHOLD_PER_MILLE, QueryBudgetProfile, QueryFallbackPath,
    QueryIntentClass, QueryIntentDecision, QueryPlanner, QueryPlannerConfig, RetrievalBudget,
};
pub use ranking_priors::{
    ALL_PRIOR_FAMILIES, DEFAULT_MAX_PRIOR_BOOST, DEFAULT_PATH_PROXIMITY_RADIUS,
    DEFAULT_RECENCY_HALF_LIFE_DAYS, DocumentPriorMetadata, MIN_PRIOR_MULTIPLIER,
    PRIOR_CONFIG_SCHEMA_VERSION, PRIOR_TIE_BREAK_CONTRACT, PriorApplicationResult, PriorApplier,
    PriorEvidence, PriorFamily, PriorFamilyConfig, QueryPriorContext, RankingPriorConfig,
    path_proximity_multiplier, project_affinity_multiplier, recency_multiplier,
    shared_prefix_depth,
};
pub use redaction::{
    ArtifactRetention, ArtifactType, DataClass, HARD_DENY_PATH_PATTERNS, MaskSeed, OutputSurface,
    REDACTION_POLICY_VERSION, RedactionPolicy, RedactionResult, RedactionTransform, TransformRule,
    classify_path, default_artifact_retention, default_rule_matrix, deterministic_hash,
    deterministic_mask, deterministic_truncate, is_hard_deny_path,
};
pub use repro::{
    ArtifactEntry, CaptureReason, EnvEntry, EnvSnapshot, FrameSeqRange, IndexChecksum,
    IndexChecksums, ModelManifest, ModelSnapshot, PACK_FILES, REPRO_SCHEMA_VERSION, ReplayMeta,
    ReplayMode, ReproInstance, ReproManifest, RetentionPolicy, RetentionTier, files_for_tier,
    should_capture_env, should_redact_env,
};
pub use runtime::{FsfsRuntime, InterfaceMode};
pub use shutdown::{FORCE_EXIT_WINDOW, ShutdownCoordinator, ShutdownReason, ShutdownState};
pub use stream_protocol::{
    STREAM_PROTOCOL_VERSION, STREAM_SCHEMA_VERSION, StreamEvent, StreamEventKind,
    StreamExplainEvent, StreamFailureCategory, StreamFrame, StreamProgressEvent, StreamResultEvent,
    StreamRetryDirective, StreamStartedEvent, StreamTerminalEvent, StreamTerminalStatus,
    StreamWarningEvent, TOON_STREAM_RECORD_SEPARATOR, TOON_STREAM_RECORD_SEPARATOR_BYTE,
    decode_stream_frame_ndjson, decode_stream_frame_toon, encode_stream_frame_ndjson,
    encode_stream_frame_toon, failure_category_for_error, is_retryable_error, retry_backoff_ms,
    terminal_event_completed, terminal_event_from_error, validate_stream_frame,
};
pub use watcher::{
    DEFAULT_BATCH_SIZE, DEFAULT_DEBOUNCE_MS, FileSnapshot, FsWatcher, NoopWatchIngestPipeline,
    WatchBatchOutcome, WatchEvent, WatchEventKind, WatchIngestOp, WatchIngestPipeline,
    WatcherExecutionPolicy, WatcherStats,
};
