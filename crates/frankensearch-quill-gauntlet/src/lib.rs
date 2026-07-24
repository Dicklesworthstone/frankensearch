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
mod perf;
mod perf_ratchet;
mod runner;
mod version_contract;

use std::path::PathBuf;

use thiserror::Error;

pub use artifact::{
    ArtifactObject, ArtifactStore, CANONICALIZATION_VERSION, CampaignArtifactContext,
    OBJECT_SCHEMA_VERSION, PreparedArtifact, RunManifest,
};
pub use comparator::{
    AstDifference, AstLoweringKind, ComparatorConfig, ComparisonReport, ComparisonStatus,
    CountState, Divergence, DivergenceClass, EngineObservation, NativeTieKey, RankClass, RankedHit,
    SCORE_EPSILON, ScoreEpsilonReason, compare_observations,
};
#[cfg(feature = "tantivy-oracle")]
pub use engine::TantivyOracle;
pub use engine::{
    ComparisonMode, DifferentialCase, DifferentialCaseMetadata, DifferentialHarness,
    EngineDescriptor, EngineFamily, EnginePairIdentity, GauntletEngine, GauntletFuture, HarnessRun,
    MAX_SNIPPET_CHARS, QuillSubject,
};
pub use generator::{
    CORE_RELEVANCE_DOCUMENT_COUNT, CassDocumentFields, CorpusManifest, CorpusSourceManifest,
    FULL_SHARED_DOCUMENT_COUNT, GENERATOR_ID, GENERATOR_SCHEMA_VERSION, GeneratedDocument,
    GeneratedQueryCase, GeneratedQueryFilters, GeneratedQueryKind, GeneratedQuerySuite,
    GlobPatternClass, HarvestedContractQuery, MAX_CORPUS_DOCUMENT_COUNT, MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_ID_BYTES, MAX_QUERY_CASES, MAX_QUERY_ID_BYTES, MAX_QUERY_SUITE_TEXT_BYTES,
    MAX_QUERY_TEXT_BYTES, Pathology, QUERY_MANIFEST_SCHEMA_VERSION, QueryGeneratorSpec,
    QueryManifest, QuerySuiteSource, QuerySyntax, RangeClass, RepositoryEntry,
    RepositoryFileDigest, RepositorySkipReason, RepositorySnapshot, SharedCorpusView,
    SharedEdgeCase, SharedFixtureSuite, SharedRelevanceQuery, SkippedRepositoryEntry,
    SourceFileDigest, SyntheticCorpus, SyntheticCorpusIter, SyntheticCorpusSpec, UnicodeLane,
    XLARGE_DOCUMENT_COUNT, ZipfExponent,
};
pub use perf::{
    DistributionSummary, PERF_ARTIFACT_SCHEMA_VERSION, PERF_MAX_CV_PCT, PERF_MIN_RUNS,
    PERF_MIN_WRITER_HEAP_PER_THREAD_BYTES, PERF_WRITER_HEAP_BYTES, PerfCellResult, PerfCellSpec,
    PerfCorpus, PerfGate, PerfGateArtifact, PerfMatrixSpec, PerfQueryClass, PerfTopology,
    PositionMode, machine_fingerprint, parse_macos_time_max_rss_bytes, peak_rss_bytes,
    perf_writer_heap_bytes, validate_matrix,
};
pub use perf_ratchet::{
    PERF_MAX_REGRESSION_PCT, PERF_MAX_REPRODUCTION_DELTA_PCT, PERF_RATCHET_SCHEMA_VERSION,
    PERF_REGRESSION_ROBUST_Z, PerfCellComparison, PerfEvidenceFile, PerfGateDecision,
    PerfRatchetEvaluation, PerfRatchetMode, PerfRatchetReason, PerfRatchetRequest,
    evaluate_perf_ratchet,
};
pub use runner::{
    CAMPAIGN_REPORT_SCHEMA_VERSION, CampaignCaseResult, CampaignConfig, CampaignDisposition,
    CampaignFuture, CampaignReport, CampaignSelection, DEFAULT_ANALYZER_CONTRACT_HASH,
    DEFAULT_ANALYZER_CONTRACT_PREIMAGE, DEFAULT_SCHEMA_CONTRACT_HASH,
    DEFAULT_SCHEMA_CONTRACT_PREIMAGE, DEFAULT_SHRINK_FUEL, DifferentialCampaignEngine,
    DifferentialCampaignRunner, DivergenceRegisterDecision, DivergenceRegisterEntry,
    DivergenceRegistry, EngineIndexReceipt, GeneratedCorpusReplay, MismatchGroup,
    QueryClassSummary, SCALAR_G1A_SCHEMA_CONTRACT_PREIMAGE, SemanticContract,
    ShadowDivergenceRecord, ShrinkDriver, ShrinkEngineFactory, ShrinkError, ShrinkRequest,
    ShrunkReproduction, SuspectedLayer, TriageConfidence, TriageVerdict,
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
