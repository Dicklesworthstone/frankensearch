#![forbid(unsafe_code)]
//! Dev-only conformance harness for Quill and the pinned Tantivy oracle.
//!
//! This crate is a workspace member but intentionally excluded from
//! `default-members` and cannot be published. Shipping crates must never depend
//! on it. The current G0 milestone provides engine identity guards, pure
//! comparators, immutable content-addressed artifacts, pending Q1 fixtures, and
//! an executable same-engine Quiver codec differential, and deterministic,
//! content-addressed E6 corpus/query generators, and the live scalar Quill
//! subject used by the G1a default-syntax campaign.

mod artifact;
mod comparator;
mod engine;
mod generator;
mod local_perf_runner;
mod machine_class_registry;
mod perf;
mod perf_assembly;
mod perf_evidence;
mod perf_ratchet;
mod qg6_prepared;
mod runner;
mod version_contract;

use std::path::PathBuf;

use thiserror::Error;

pub use artifact::{
    ArtifactLexicalContractEvidence, ArtifactObject, ArtifactStore, CANONICALIZATION_VERSION,
    CampaignArtifactContext, OBJECT_SCHEMA_VERSION, PreparedArtifact, RUN_MANIFEST_SCHEMA_VERSION,
    RunManifest,
};
pub use comparator::{
    AstDifference, AstLoweringKind, ComparatorConfig, ComparisonReport, ComparisonStatus,
    CountState, Divergence, DivergenceClass, EngineObservation,
    LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION, LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION,
    LEXICAL_OBSERVATION_SCHEMA_VERSION, LexicalBackendIdentity, LexicalBoundary,
    LexicalComparisonReport, LexicalComparisonStatus, LexicalContractBundle,
    LexicalContractComparison, LexicalContractCoverage, LexicalCountExposure, LexicalCountState,
    LexicalDeferredSide, LexicalEmptyShape, LexicalEngineRole, LexicalEquivalenceLaw,
    LexicalErrorClass, LexicalErrorObservation, LexicalExposureContract, LexicalFieldExposure,
    LexicalFieldMismatch, LexicalHighlightSpan, LexicalHitObservation, LexicalHitSupplement,
    LexicalHydrationExecution, LexicalHydrationNotRunReason, LexicalHydrationProbe,
    LexicalHydrationResult, LexicalHydrationSelection, LexicalHydrationTransition,
    LexicalMismatchClass, LexicalNonLexicalControlKind, LexicalNormalizedQuery, LexicalObservation,
    LexicalObservationContext, LexicalObservationOutcome, LexicalObservationSupplement,
    LexicalObserved, LexicalProbeCoverage, LexicalQueryClass, LexicalScoreSource,
    LexicalSideCoverage, LexicalWaivedDifference, LexicalWaiverTarget, LexicalWinnerOrigin,
    LexicalWinnerProjection, MAX_LEXICAL_DOC_ID_BYTES, MAX_LEXICAL_ERROR_SOURCE_DEPTH,
    MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT, MAX_LEXICAL_OBSERVATION_HITS, MAX_LEXICAL_QUERY_BYTES,
    MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES, NativeTieKey, QUILL_CANCELLATION_RECEIPT_SCHEMA_VERSION,
    QuillCancellationCheckpoint, QuillCancellationEvidenceOrigin, QuillCancellationObservation,
    QuillCancellationReceipt, QuillCancellationReceiptBody, RankClass, RankedHit, SCORE_EPSILON,
    ScoreEpsilonReason, SensitiveValueObservation, compare_lexical_contracts,
    compare_lexical_observations, compare_observations, observe_lexical_outcome,
    observe_live_quill_cancellation_receipt,
};
pub use engine::{
    CASS_TANTIVY_ORACLE_CONFIG_HASH, ComparisonMode, DifferentialCase, DifferentialCaseMetadata,
    DifferentialHarness, EngineDescriptor, EngineFamily, EnginePairIdentity, GauntletEngine,
    GauntletFuture, HarnessRun, MAX_SNIPPET_CHARS, QuillSubject,
};
#[cfg(feature = "tantivy-oracle")]
pub use engine::{CassQuillSubject, CassTantivyOracle, TantivyOracle};
pub use generator::{
    CORE_RELEVANCE_DOCUMENT_COUNT, CassDocumentFields, CorpusManifest, CorpusSourceManifest,
    FULL_SHARED_DOCUMENT_COUNT, GENERATOR_ID, GENERATOR_SCHEMA_VERSION, GeneratedDocument,
    GeneratedQueryCase, GeneratedQueryFilters, GeneratedQueryKind, GeneratedQuerySuite,
    GeneratedSourceFilter, GlobPatternClass, HarvestedContractQuery, MAX_CORPUS_DOCUMENT_COUNT,
    MAX_DOCUMENT_BYTES, MAX_DOCUMENT_ID_BYTES, MAX_QUERY_CASES, MAX_QUERY_ID_BYTES,
    MAX_QUERY_SUITE_TEXT_BYTES, MAX_QUERY_TEXT_BYTES, Pathology, QUERY_MANIFEST_SCHEMA_VERSION,
    QueryGeneratorSpec, QueryManifest, QuerySuiteSource, QuerySyntax, RangeClass, RepositoryEntry,
    RepositoryFileDigest, RepositorySkipReason, RepositorySnapshot, SharedCorpusView,
    SharedEdgeCase, SharedFixtureSuite, SharedRelevanceQuery, SkippedRepositoryEntry,
    SourceFileDigest, StructuredFilterClass, SyntheticCorpus, SyntheticCorpusIter,
    SyntheticCorpusSpec, UnicodeLane, XLARGE_DOCUMENT_COUNT, ZipfExponent,
};
pub use local_perf_runner::{
    LOCAL_PERF_ATTEMPT_RECEIPT_SCHEMA_VERSION, LocalPerfAttemptOutcome, LocalPerfAttemptReceipt,
    LocalPerfInternalLifecycleGaps, LocalPerfInternalLifecycleUnavailable,
    LocalPerfProcessLifecycle, LocalPerfRetryPredicate, LocalPerfRunConfig, LocalPerfRunError,
    LocalPerfRunOutput, LocalPerfRunSelection, LocalPerfUnsupportedControl,
    local_perf_producer_contract_json, run_local_perf_command, run_selected_local_perf_command,
};
pub use machine_class_registry::{
    DefaultFlipDisposition, ExecutionCapacitySemantics, ExecutionProfileId, HardwareClassId,
    LOCAL_PERF_PRODUCER_CONTRACT_VERSION, MACHINE_CLASS_REGISTRY_GIT_BLOB,
    MACHINE_CLASS_REGISTRY_SCHEMA_VERSION, MACHINE_CLASS_REGISTRY_SHA256,
    MACHINE_CLASS_REGISTRY_SPEC_COMMIT, MachineClassAdmissionContext,
    MachineClassCanonicalizationBinding, MachineClassDecision, MachineClassDerivedHashes,
    MachineClassError, MachineClassEvidenceBinding, MachineClassLookup, MachineClassReason,
    MachineClassRegistry, MachineExecutionProfile, MachineProfileAvailability,
    MachineProfileGatePolicy, MachineProfileKey, RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    RUNNER_RECEIPT_SCHEMA_VERSION, RunnerArtifactManifest, RunnerArtifactManifestBinding,
    VerifiedRunnerIdentity,
};
pub use perf::{
    DistributionSummary, LEGACY_PERF_ARTIFACT_SCHEMA_VERSION_V3, PAIRED_ESTIMATOR_SCHEMA_VERSION,
    PERF_APPLICABILITY_PLAN_SCHEMA_VERSION, PERF_ARTIFACT_SCHEMA_VERSION, PERF_MAX_CV_PCT,
    PERF_MIN_RUNS, PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES, PERF_WRITER_HEAP_BYTES, PairedClaimState,
    PairedEffectEstimate, PairedEstimatorConfig, PairedEstimatorError, PairedEstimatorReason,
    PairedEvidenceStatus, PairedExperimentResult, PerfApplicabilityPlan,
    PerfApplicabilityPlanBinding, PerfApplicabilityPlanError, PerfCellApplicability,
    PerfCellApplicabilityEntry, PerfCellApplicabilityReason, PerfCellResult, PerfCellSpec,
    PerfCorpus, PerfExecutionProvenance, PerfGate, PerfGateArtifact, PerfInputIdentity,
    PerfMatrixSpec, PerfMetricSemantics, PerfOperationScope, PerfProducerOs, PerfQueryClass,
    PerfRawSample, PerfSampleArm, PerfSampleOrder, PerfSamplePhase, PerfSampleProvenance,
    PerfTopology, PositionMode, QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS, Qg6SampleBinding,
    estimate_paired_experiment, machine_fingerprint, parse_macos_time_max_rss_bytes,
    peak_rss_bytes, perf_manifest_contract_sha256, perf_writer_heap_bytes,
    seeded_balanced_pair_order, validate_matrix,
};
pub use perf_assembly::{
    PERF_ASSEMBLY_ENGINE_LIFECYCLE_NO_CLAIM_CODE, PERF_ASSEMBLY_MAX_ARTIFACT_BYTES,
    PERF_ASSEMBLY_MAX_RECEIPT_BYTES, PERF_ASSEMBLY_MAX_RETRY_PREDICATE_BYTES,
    PERF_ASSEMBLY_MAX_SHARDS, PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_CODE,
    PERF_ASSEMBLY_PARTIAL_SHARD_NO_CLAIM_DETAIL, PERF_ASSEMBLY_PROCESS_TREE_NO_CLAIM_CODE,
    PERF_EVIDENCE_ASSEMBLY_MATRIX_SCHEMA_VERSION, PERF_EVIDENCE_ASSEMBLY_SCHEMA_VERSION,
    PERF_EVIDENCE_SEMANTIC_CELL_SET_SCHEMA_VERSION, PerfAssemblyMachineIdentity,
    PerfAssemblyProcessReceipt, PerfEvidenceAssemblyArtifact, PerfEvidenceAssemblyCompatibility,
    PerfEvidenceAssemblyCompleteness, PerfEvidenceAssemblyCounts, PerfEvidenceAssemblyError,
    PerfEvidenceAssemblyFailedAttempt, PerfEvidenceAssemblyMatrixCell,
    PerfEvidenceAssemblyMatrixManifest, PerfEvidenceAssemblyNoClaimCell,
    PerfEvidenceAssemblyNoClaimSource, PerfEvidenceAssemblyReadiness, PerfEvidenceAssemblySource,
    PerfEvidenceCellSource, PerfEvidenceSemanticCellSetSeal, VerifiedLocalPerfAttemptBundle,
};
pub use perf_evidence::{
    AbsoluteRelativeReconciliation, BuildIdentity, ColdCacheEvidence, CorpusIdentity,
    EVIDENCE_MAX_REASON_MESSAGE_BYTES, EVIDENCE_MAX_REASONS, EngineConcurrencyObservation,
    EvidenceArtifactError, EvidenceArtifactPaths, EvidenceCell, EvidenceCellBody, EvidenceCellSpec,
    EvidenceDecisionStatus, EvidenceEstimand, EvidencePolicy, EvidenceProvenance, EvidenceReason,
    EvidenceRole, EvidenceSeverity, HIERARCHICAL_LATENCY_SCHEMA_VERSION, HierarchicalGroupSummary,
    HierarchicalLatencyEstimate, MachineIdentity, PERF_EVIDENCE_SCHEMA_VERSION, PeakRssEvidence,
    PerfConcurrencyEngine, PerfConcurrencyObserver, PerfConcurrencyWitness, PerfEvidenceArtifact,
    command_sha256_from_argv, estimate_hierarchical_latency, human_table_from_json,
    load_legacy_gate_artifact_v3, required_estimand,
};
pub use perf_ratchet::{
    PERF_MAX_REGRESSION_PCT, PERF_MAX_REPRODUCTION_DELTA_PCT, PERF_RATCHET_SCHEMA_VERSION,
    PERF_REGRESSION_ROBUST_Z, PerfCellComparison, PerfEvidenceFile, PerfGateDecision,
    PerfRatchetEvaluation, PerfRatchetMode, PerfRatchetReason, PerfRatchetRequest,
    evaluate_perf_ratchet, is_explicit_bootstrap, is_explicit_bootstrap_for,
};
pub use qg6_prepared::{
    Qg6ArmLifecycle, Qg6ArmRole, Qg6Comparison, Qg6ExperimentIdentity, Qg6FourArmResultReceipts,
    Qg6HarnessError, Qg6LifecycleReceipt, Qg6Measurement, Qg6PairBlock, Qg6Phase,
    Qg6PreparedExperiment, Qg6QueryGroupReceipt, Qg6QueryIdentityReceipt, Qg6QuerySpec,
    Qg6RankedHitReceipt, Qg6ResultReceipt, Qg6SampleOrder, Qg6SearchHit, Qg6SearchResult,
    Qg6SelectionClaim, Qg6SelectionScope, Qg6SemanticContract, Qg6SetupRecorder, Qg6TimedSample,
    Qg6ValidatedExperiment, qg6_result_sequence_sha256, seeded_interleaved_four_arm_schedule,
};
pub use runner::{
    CAMPAIGN_REPORT_SCHEMA_VERSION, CASS_ANALYZER_CONTRACT_PREIMAGE, CASS_SCHEMA_CONTRACT_PREIMAGE,
    CampaignCaseResult, CampaignConfig, CampaignContractMode, CampaignDisposition, CampaignFuture,
    CampaignLexicalCaseSummary, CampaignLexicalCoverageSummary, CampaignProvenance, CampaignReport,
    CampaignSelection, DEFAULT_ANALYZER_CONTRACT_HASH, DEFAULT_ANALYZER_CONTRACT_PREIMAGE,
    DEFAULT_SCHEMA_CONTRACT_HASH, DEFAULT_SCHEMA_CONTRACT_PREIMAGE, DEFAULT_SHRINK_FUEL,
    DIVERGENCE_REGISTER_LEDGER_SCHEMA_VERSION, DIVERGENCE_REGISTER_REDACTION_POLICY_VERSION,
    DifferentialCampaignEngine, DifferentialCampaignRunner, DivergenceCensus,
    DivergenceClassCensus, DivergenceDisposition, DivergenceDispositionEvent,
    DivergenceFixtureEvidence, DivergenceObservationEvent, DivergencePredictionEvent,
    DivergenceRegisterDecision, DivergenceRegisterEntry, DivergenceRegisterEvent,
    DivergenceRegisterEventHeader, DivergenceRegisterLedger, DivergenceRegistry,
    DivergenceRevisionSet, EngineIndexReceipt, GeneratedCorpusReplay, LexicalMismatchGroup,
    LexicalSideCoverageCounts, MismatchGroup, PredictedDivergenceState, ProbeCoverageCounts,
    QueryClassSummary, RedactedDivergenceDiagnostic, SCALAR_G1A_SCHEMA_CONTRACT_PREIMAGE,
    SemanticContract, ShadowDivergenceRecord, ShrinkDriver, ShrinkEngineFactory, ShrinkError,
    ShrinkRequest, ShrunkReproduction, SuspectedLayer, TriageConfidence, TriageVerdict,
    persist_shrunk_reproduction,
};
pub use version_contract::{
    InternalDifferentialFixture, OracleVersionContract, Q1Fixture, Q1FixtureCatalog,
    oracle_version_contract, q1_fixture_catalog, run_q1_live_fixtures,
};

/// Typed failure surface for harness setup, execution, comparison, and storage.
#[derive(Debug, Error)]
pub enum GauntletError {
    #[error("engine identity collision in {comparison_mode:?}: {subject} vs {oracle}")]
    EngineIdentityCollision {
        comparison_mode: ComparisonMode,
        subject: String,
        oracle: String,
    },
    #[error("invalid comparator configuration: {reason}")]
    InvalidComparatorConfig { reason: String },
    #[error("invalid engine observation: {reason}")]
    InvalidObservation { reason: String },
    #[error("invalid differential case: {reason}")]
    InvalidCase { reason: String },
    #[error("invalid deterministic generator input: {reason}")]
    InvalidGenerator { reason: String },
    #[error("invalid differential campaign: {reason}")]
    InvalidCampaign { reason: String },
    #[error("content-addressed replay mismatch: {reason}")]
    ManifestMismatch { reason: String },
    #[error("subject is unavailable: {reason}")]
    SubjectUnavailable { reason: String },
    #[error("invalid committed contract: {reason}")]
    InvalidContract { reason: String },
    #[error("invalid run ID {run_id:?}")]
    InvalidRunId { run_id: String },
    #[error("invalid prepared artifact: {reason}")]
    InvalidPreparedArtifact { reason: String },
    #[error("unsafe gauntlet store path: {path}")]
    UnsafeStorePath { path: PathBuf },
    #[error("content-address collision at {path}")]
    ArtifactCollision { path: PathBuf },
    #[error("run manifest already points at different content: {path}")]
    RunManifestConflict { path: PathBuf },
    #[error(transparent)]
    Search(#[from] frankensearch_core::SearchError),
    #[error(transparent)]
    Quill(#[from] frankensearch_quill::QuillIndexError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
