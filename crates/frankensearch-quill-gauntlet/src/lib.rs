#![forbid(unsafe_code)]
#![recursion_limit = "256"]
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
mod campaign_contract;
mod comparator;
pub mod cutoff_certificate;
mod engine;
mod generator;
mod local_perf_runner;
mod machine_class_registry;
// E6.3 metamorphic-law infrastructure is test-only, exactly like the law
// executors it feeds in `engine.rs`, which all live inside that file's
// `#[cfg(test)] mod tests`. Declaring it `#[cfg(test)]` states that honestly
// instead of exporting unstable internals as crate API or silencing dead-code
// with an allow — under `--lib` there is genuinely no production consumer yet,
// because the executors that will call these schedules are still blocked.
#[cfg(test)]
mod metamorphic_maintenance_laws;
#[cfg(test)]
mod metamorphic_maintenance_schedules;
/// The native enriched Quill/Tantivy witness (bd-8nqz.4.1).
///
/// Layered OVER Core Lexical V3 (`campaign_contract`) and separately
/// versioned; that contract is deliberately untouched by this module.
pub mod native_enriched_witness;
mod perf;
mod perf_assembly;
mod perf_evidence;
mod perf_ratchet;
mod privacy;
mod qg2_contract;
mod qg6_prepared;
pub mod replacement_authorization;
mod runner;
/// ArtifactStore v4 F2: supervisor-issued execution/completion authentication.
///
/// External nonce authority, Ed25519 receipts, and typed completion absences
/// per the frozen F0 trust contract.
pub mod supervisor;
mod version_contract;

use std::path::PathBuf;

use thiserror::Error;

pub use artifact::{
    ArtifactExecutionRole, ArtifactLexicalContractEvidence, ArtifactObject,
    ArtifactOracleDependency, ArtifactStore, ArtifactStoreV4BuildInput,
    ArtifactStoreV4BuildInputKind, ArtifactStoreV4BuildSnapshot,
    ArtifactStoreV4DependencyBuildScriptRecord, ArtifactStoreV4SourceBuildBinding,
    ArtifactStoreV4SourceBuildSnapshots, ArtifactStoreV4SourceEntry,
    ArtifactStoreV4SourceEntryKind, ArtifactStoreV4SourceInclusionReason,
    ArtifactStoreV4SourceSnapshot, ArtifactTrustCeiling, CANONICALIZATION_VERSION,
    CampaignArtifactContext, GauntletProducerBuildIdentity, GauntletProducerSourceVerification,
    IntegrityCheckedCampaign, OBJECT_HASH_SCHEME_V9_SHA256, OBJECT_SCHEMA_VERSION,
    PreparedArtifact, RUN_MANIFEST_SCHEMA_VERSION, RunManifest, SerializedSchemaDisposition,
    classify_artifact_object_schema, classify_campaign_report_schema,
    collect_dependency_build_script_records, dependency_build_script_build_input,
    dependency_build_script_records_sha256, pinned_campaign_report_v8,
};
pub use campaign_contract::{
    BuiltInEvidenceBindingV1, CampaignCellEvidenceV1, CampaignCellKeyV1, CampaignContractModeV1,
    CampaignContractValueError, CampaignCorpusV1, CampaignEvidenceRole, CampaignExecutionProfileV1,
    CampaignHardwareClassV1, CampaignMachineProfileV1, CampaignNightV1, CampaignProfileV1,
    CampaignReplayV1, CampaignSeedBundleV1, CampaignSeedScheduleV1, CampaignSeedSlotV1,
    CampaignSha256V1, CampaignTopologyV1, REPLACEMENT_COMPLETENESS_EXPECTED_CELL_COUNT,
    REPLACEMENT_COMPLETENESS_POLICY_SCHEMA_VERSION, ReplacementCompletenessError,
    ReplacementCompletenessPolicyV1, UnexpectedCampaignCellReasonV1, frozen_replacement_cell_keys,
    frozen_replacement_completeness_policy, frozen_replacement_seed_bundle,
    frozen_replacement_seed_schedule, replacement_completeness_policy_sha256,
    validate_replacement_completeness,
};
#[cfg(feature = "tantivy-oracle")]
pub use comparator::observe_tantivy_cass_profile;
pub use comparator::{
    AstDifference, AstLoweringKind, CASS_LEXICAL_PROFILE_OBSERVATION_SCHEMA_VERSION,
    CassLexicalProfileComparison, CassLexicalProfileContext, CassLexicalProfileObservation,
    CassLexicalProfileOutcome, CassLexicalProfileProvenance, CassLexicalProfileSuccess,
    CassProfileAuthority, CassProfileDiagnosticKind, CassProfileDiagnosticObservation,
    CassProfileField, CassProfileFilters, CassProfileMismatch, CassProfileMismatchClass,
    CassProfileNotExposedReason, CassProfileRankedHit, CassProfileRequest, CassProfileSourceFilter,
    CassProfileTieMember, CassProfileTokenKind, CassProfileTokenObservation, ComparatorConfig,
    ComparisonReport, ComparisonStatus, CountState, Divergence, DivergenceClass, EngineObservation,
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
    LexicalWinnerProjection, MAX_CASS_PROFILE_DIAGNOSTICS, MAX_CASS_PROFILE_DOCUMENT_ID_BYTES,
    MAX_CASS_PROFILE_FETCH_HITS, MAX_CASS_PROFILE_FILTER_BYTES,
    MAX_CASS_PROFILE_FILTER_VALUE_BYTES, MAX_CASS_PROFILE_FILTER_VALUES, MAX_CASS_PROFILE_TOKENS,
    MAX_LEXICAL_DOC_ID_BYTES, MAX_LEXICAL_ERROR_SOURCE_DEPTH, MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT,
    MAX_LEXICAL_OBSERVATION_HITS, MAX_LEXICAL_QUERY_BYTES, MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES,
    NativeTieKey, OracleBugControlObservation, QUILL_CANCELLATION_RECEIPT_SCHEMA_VERSION,
    QuillCancellationCheckpoint, QuillCancellationEvidenceOrigin, QuillCancellationObservation,
    QuillCancellationReceipt, QuillCancellationReceiptBody, RankClass, RankedHit, SCORE_EPSILON,
    ScoreEpsilonReason, SensitiveValueObservation, compare_cass_lexical_profiles,
    compare_lexical_contracts, compare_lexical_observations, compare_observations,
    compare_observations_stored_v7, compare_observations_stored_v8, compare_observations_stored_v9,
    compare_observations_with_control, observe_lexical_outcome,
    observe_live_quill_cancellation_receipt, observe_quill_cass_profile,
};
pub use engine::{
    CASS_TANTIVY_ORACLE_CONFIG_HASH, ComparisonMode, DifferentialCase, DifferentialCaseMetadata,
    DifferentialHarness, EngineDescriptor, EngineFamily, EnginePairIdentity, GauntletEngine,
    GauntletFuture, HarnessRun, MAX_SNIPPET_CHARS, QuillSubject,
};
#[cfg(feature = "tantivy-oracle")]
pub use engine::{CassQuillSubject, CassTantivyOracle, TantivyOracle};
#[cfg(feature = "fuzz-harness")]
pub use engine::{
    TYPED_QUERY_FUZZ_GENERATOR_ID, TYPED_QUERY_FUZZ_MAX_INPUT_BYTES,
    TYPED_QUERY_FUZZ_REPLAY_SCHEMA_VERSION, TYPED_QUERY_FUZZ_SHRINK_FUEL,
    TypedQueryFailureFingerprint, TypedQueryFuzzReplay, TypedQueryFuzzReplayArtifact,
    TypedQueryFuzzWorkload, TypedQueryLenientAsymmetry, TypedQueryOracleBehavior, TypedQueryTree,
    load_typed_query_fuzz_replay, materialize_typed_query_fuzz_workload,
    persist_typed_query_fuzz_replay, scalar_g1a_fuzz_pair, typed_query_failure_divergence_class,
    typed_query_fuzz_corpus_spec, typed_query_fuzz_seed, typed_query_fuzz_vocabulary,
};
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
    Qg1StartupControlFrameV1, Qg1StartupHandshakeV1, Qg6StartupAuthoritySetV1,
    Qg6StartupHandshakeV1, local_perf_producer_contract_json,
    publish_qg6_startup_authorities_and_wait_for_ack, run_local_perf_command,
    run_selected_local_perf_command,
};
pub use machine_class_registry::{
    DefaultFlipDisposition, ExecutionCapacitySemantics, ExecutionProfileId, HardwareClassId,
    LOCAL_PERF_PRODUCER_CONTRACT_VERSION, MACHINE_CLASS_REGISTRY_GIT_BLOB,
    MACHINE_CLASS_REGISTRY_SCHEMA_VERSION, MACHINE_CLASS_REGISTRY_SHA256,
    MACHINE_CLASS_REGISTRY_SPEC_COMMIT, MachineClassAdmissionContext,
    MachineClassCanonicalizationBinding, MachineClassDecision, MachineClassDerivedHashes,
    MachineClassError, MachineClassEvidenceBinding, MachineClassLookup, MachineClassReason,
    MachineClassRegistry, MachineExecutionProfile, MachineProfileAvailability,
    MachineProfileGatePolicy, MachineProfileKey, QG5_DURABILITY_WITNESS_FILE_NAME,
    QG5_DURABILITY_WITNESS_SCHEMA_VERSION, Qg5CellDurabilityWitness,
    Qg5DeletePublicationObservation, Qg5DurabilityEngine, Qg5DurabilityObservation,
    Qg5DurabilityWitnessSet, Qg5ReopenValidationObservation, Qg5SampleDurabilityWitness,
    Qg5StreamRole, Qg5TimedMaintenanceObservation, RUNNER_ARTIFACT_MANIFEST_SCHEMA_VERSION,
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
    PerfTopology, PositionMode, QG1_QUILL_ENGINE_ID, QG1_STREAM_ROLE_EFFECT,
    QG1_STREAM_ROLE_QUILL_NULL, QG1_STREAM_ROLE_TANTIVY_NULL, QG1_STREAM_ROLE_TANTIVY_PILOT_EFFECT,
    QG1_STREAM_ROLE_TANTIVY_PILOT_NULL, QG1_TANTIVY_ENGINE_ID,
    QG1_TANTIVY_INCUMBENT_SCREEN_SCHEMA_VERSION, QG1_TANTIVY_INCUMBENT_TANTIVY_VERSION,
    QG6_QUERY_GROUP_IDS, QG6_QUERY_GROUPS, Qg1AuthorityRegisterEntryV1, Qg1AuthorityRoleV1,
    Qg1BatchCoverage, Qg1ExpectedAuthority, Qg1LifecycleProducer, Qg1LifecycleWitness,
    Qg1PinnedAuthorityTargetV1, Qg1SampleBinding, Qg1TantivyBoundStream,
    Qg1TantivyDecisionStreamKind, Qg1TantivyIncumbentCandidate, Qg1TantivyIncumbentDecision,
    Qg1TantivyIncumbentError, Qg1TantivyIncumbentPilot, Qg1TantivyIncumbentScreen,
    Qg1TantivyIncumbentScreenPlan, Qg1TantivySemanticContract, Qg1TantivyWriterMode,
    Qg1TargetPinV1, Qg1VerifiedAuthorityRegistrationV1, Qg6EffectLeafDistributions,
    Qg6SampleBinding, estimate_paired_experiment, estimate_paired_experiment_against_qg1_authority,
    machine_fingerprint, parse_macos_time_max_rss_bytes, peak_rss_bytes,
    perf_manifest_contract_sha256, perf_writer_heap_bytes, preregister_qg1_tantivy_incumbents,
    project_qg6_effect_leaf_distributions, seeded_balanced_pair_order, validate_matrix,
};
pub use perf::{PERF_RUN_PLAN_DOC_PATH, render_perf_run_plan_markdown};
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
    HierarchicalLatencyEstimate, MachineIdentity, PERF_EVIDENCE_MAX_ARTIFACT_BYTES,
    PERF_EVIDENCE_SCHEMA_VERSION, PeakRssEvidence, PerfConcurrencyEngine, PerfConcurrencyObserver,
    PerfConcurrencyWitness, PerfEvidenceArtifact, QG6_JOINT_TAIL_MAX_BOOTSTRAP_REPLICATES,
    QG6_JOINT_TAIL_MIN_BOOTSTRAP_REPLICATES, QG6_JOINT_TAIL_SCHEMA_VERSION,
    QG6_P50_TOST_WINDOW_RATIO, QG6_P99_UCB_LIMIT_RATIO, QG6_PER_CELL_ALPHA,
    QG10_DEPENDENCY_FACTS_SCHEMA_VERSION, QG10_MAX_CARGO_TREE_STDOUT_BYTES,
    QG10_MAX_PACKAGE_ID_BYTES, QG10_MAX_PACKAGE_IDS, QG10_MEASURED_FEATURES,
    QG10_POSITIVE_CONTROL_FEATURES, QG10_POSITIVE_CONTROL_REFUSAL, Qg1IncumbentScreenEvidence,
    Qg6FormalProtocolEvidence, Qg6JointTailContrast, Qg6JointTailEstimate, Qg10DependencyFacts,
    Qg10DependencyGraph, command_sha256_from_argv, estimate_hierarchical_latency,
    estimate_qg6_joint_tail, human_table_from_json, load_legacy_gate_artifact_v3,
    required_estimand,
};
pub use perf_ratchet::{
    PERF_HISTORY_POINTER_SCHEMA_VERSION, PERF_MAX_REGRESSION_PCT, PERF_MAX_REPRODUCTION_DELTA_PCT,
    PERF_RATCHET_SCHEMA_VERSION, PERF_REGRESSION_ROBUST_Z, PerfCellComparison,
    PerfEvidenceAdmission, PerfEvidenceFile, PerfGateDecision, PerfRatchetEvaluation,
    PerfRatchetMode, PerfRatchetQg1AuthoritySets, PerfRatchetQg6AuthoritySets, PerfRatchetReason,
    PerfRatchetRequest, PerfReleaseEligibility, PerfTargetDecision, evaluate_perf_ratchet,
    evaluate_perf_ratchet_against_authorities, evaluate_perf_ratchet_against_qg1_authorities,
    is_explicit_bootstrap, is_explicit_bootstrap_for,
};
pub use privacy::{
    ARTIFACT_PRIVACY_POLICY_SCHEMA_VERSION, ArtifactClassification, ArtifactContentKind,
    ArtifactEnvelopeMetadata, ArtifactExportDestination, ArtifactPrivacyContext,
    ArtifactPrivacyError, ArtifactPrivacyPolicy, ArtifactRetentionStatus, OpenedArtifactBytes,
    PRIVATE_ARTIFACT_MAX_RETENTION_SECONDS, PRIVATE_ARTIFACT_MIN_RETENTION_SECONDS,
    PrivateArtifactKey, RedactedArtifactValue,
};
pub use qg2_contract::{
    QG2_CANONICAL_CONTRACT, QG2_CONTRACT_REPORT_SCHEMA_VERSION, QG2_LOGICAL_SURFACE_COUNT,
    QG2_MANIFEST_BLOCK_POST_REGION, QG2_MANIFEST_BLOCK_PRE_REGION, QG2_NO_CLAIM,
    QG2_PHYSICAL_LOCATOR_COUNT, QG2_PREFLIGHT_REPORT_SCHEMA_VERSION, QG2_SENTINEL_COUNT,
    Qg2CommitBoundary, Qg2ComparatorContract, Qg2ContractDivergence, Qg2ContractReport,
    Qg2ContractStatus, Qg2DurabilityScope, Qg2ExcludedOperation, Qg2PreflightReport,
    Qg2PreflightState, Qg2PreservedValueReceipt, Qg2SelectorReceipt, Qg2SelectorState,
    Qg2SentinelRebind, Qg2SentinelSummary, Qg2SourceNonregression, Qg2StaleHistoryDisposition,
    Qg2StaleHistoryReceipt, Qg2StorageTopology, Qg2SurfaceReceipt, Qg2TimingEnd, Qg2TimingStart,
    Qg2TopologySummary, validate_qg2_contract, validate_qg2_preflight,
};
pub use qg6_prepared::{
    QG6_TIMED_SEARCHES_PER_SAMPLE, Qg6ArmLifecycle, Qg6ArmRole, Qg6Comparison,
    Qg6ExperimentIdentity, Qg6HarnessError, Qg6LifecycleReceipt, Qg6Measurement, Qg6PairBlock,
    Qg6Phase, Qg6PreparedExperiment, Qg6QueryGroupReceipt, Qg6QueryIdentityReceipt, Qg6QuerySpec,
    Qg6RankedHitReceipt, Qg6ResidualArmRole, Qg6ResidualCacheDisposition,
    Qg6ResidualJointContrastVector, Qg6ResidualLeafObservation, Qg6ResidualScheduleAdmission,
    Qg6ResidualStratum, Qg6ResidualValidationError, Qg6ResidualWilliamsLeaf, Qg6ResultReceipt,
    Qg6SampleOrder, Qg6ScheduleAuthority, Qg6SearchHit, Qg6SearchResult, Qg6SelectionClaim,
    Qg6SelectionScope, Qg6SemanticContract, Qg6SetupRecorder, Qg6SixArmResultReceipts,
    Qg6TimedSample, Qg6ValidatedExperiment, admit_qg6_residual_schedule_meta_block,
    qg6_residual_source_elf_consistency_sha256, qg6_residual_standardized_workload_sha256,
    qg6_residual_williams_schedule, qg6_result_sequence_sha256, query_manifest_sha256,
    seeded_interleaved_six_arm_schedule, validate_qg6_residual_williams_schedule,
};
pub use runner::{
    CAMPAIGN_REPORT_SCHEMA_VERSION, CAMPAIGN_REPORT_V9_SCHEMA_VERSION,
    CASS_ANALYZER_CONTRACT_PREIMAGE, CASS_SCHEMA_CONTRACT_PREIMAGE, CampaignCaseReason,
    CampaignCaseResult, CampaignConfig, CampaignContractMode, CampaignDisposition, CampaignFuture,
    CampaignLexicalCaseSummary, CampaignLexicalCoverageSummary, CampaignProvenance, CampaignReport,
    CampaignSelection, DEFAULT_ANALYZER_CONTRACT_HASH, DEFAULT_ANALYZER_CONTRACT_PREIMAGE,
    DEFAULT_SCHEMA_CONTRACT_HASH, DEFAULT_SCHEMA_CONTRACT_PREIMAGE, DEFAULT_SHRINK_FUEL,
    DIVERGENCE_PREDICTION_POLICY_PREIMAGE, DIVERGENCE_PREDICTION_POLICY_VERSION,
    DIVERGENCE_REGISTER_LEDGER_SCHEMA_VERSION, DIVERGENCE_REGISTER_REDACTION_POLICY_VERSION,
    DifferentialCampaignEngine, DifferentialCampaignRunner, DivergenceArtifactObjectHash,
    DivergenceArtifactObjectHashScheme, DivergenceDisposition, DivergenceDispositionEvent,
    DivergenceFixtureContentWitness, DivergenceFixtureEvidence, DivergenceObservationEvent,
    DivergencePredictionEvent, DivergenceRegisterDecision, DivergenceRegisterEntry,
    DivergenceRegisterEvent, DivergenceRegisterEventHeader, DivergenceRegisterLedger,
    DivergenceRegistry, DivergenceRevisionSet, EngineIndexReceipt, GeneratedCorpusReplay,
    LexicalMismatchGroup, LexicalSideCoverageCounts, METAMORPHIC_LAW_REGISTRY_SCHEMA_VERSION,
    MetamorphicLawApplicability, MetamorphicLawApplicabilityEntry, MetamorphicLawDescriptor,
    MetamorphicLawOutcome, MetamorphicLawRegistry, MetamorphicLawResult, MetamorphicLawScope,
    MetamorphicLawSummary, MetamorphicSkipReason, MismatchGroup, PredictedDivergenceState,
    ProbeCoverageCounts, QueryClassSummary, RedactedDivergenceDiagnostic,
    SCALAR_G1A_SCHEMA_CONTRACT_PREIMAGE, SemanticContract, ShadowDivergenceRecord, ShrinkDriver,
    ShrinkEngineFactory, ShrinkError, ShrinkRequest, ShrunkReproduction, SuspectedLayer,
    TriageConfidence, TriageVerdict, divergence_prediction_policy_sha256,
    load_pinned_campaign_report_v8, load_read_only_campaign_report_v7, persist_shrunk_reproduction,
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
