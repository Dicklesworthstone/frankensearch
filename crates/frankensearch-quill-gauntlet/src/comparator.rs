use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as _;
use std::sync::Arc;

use asupersync::Cx;
use frankensearch_core::{
    ExplanationPhase, HitExplanation, LexicalSearch, QueryClass, ScoreSource, ScoredResult,
    SearchError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::GauntletError;

pub const SCORE_EPSILON: f32 = 0.0001;
/// Stable schema identifier for the complete public lexical result envelope.
pub const LEXICAL_OBSERVATION_SCHEMA_VERSION: &str = "lexical-observation-v3";
/// Stable schema identifier for the total public lexical contract bundle.
pub const LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION: &str = "lexical-contract-bundle-v3";
/// Stable schema identifier for a replayable total-contract comparison.
pub const LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION: &str = "lexical-contract-comparison-v3";
/// Maximum number of hits admitted into one lexical observation artifact.
pub const MAX_LEXICAL_OBSERVATION_HITS: usize = 100_000;
/// Maximum UTF-8 byte length of a consumer-visible document identifier.
pub const MAX_LEXICAL_DOC_ID_BYTES: usize = 1_024;
/// Maximum canonical byte length represented by one redacted payload digest.
pub const MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES: usize = 16 * 1_024 * 1_024;
/// Maximum number of ordered highlight spans represented for one hit.
pub const MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT: usize = 4_096;
/// Maximum raw or explicitly observed normalized query size.
pub const MAX_LEXICAL_QUERY_BYTES: usize = 1024 * 1024;
/// Maximum public source-chain depth represented for one typed error.
pub const MAX_LEXICAL_ERROR_SOURCE_DEPTH: usize = 16;

/// Closed artifact-stable projection of [`QueryClass`].
///
/// Keeping the evidence enum local makes an upstream `QueryClass` addition a
/// compile error until the lexical observation schema is reviewed and bumped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalQueryClass {
    Empty,
    Identifier,
    ShortKeyword,
    NaturalLanguage,
}

impl From<QueryClass> for LexicalQueryClass {
    fn from(value: QueryClass) -> Self {
        match value {
            QueryClass::Empty => Self::Empty,
            QueryClass::Identifier => Self::Identifier,
            QueryClass::ShortKeyword => Self::ShortKeyword,
            QueryClass::NaturalLanguage => Self::NaturalLanguage,
        }
    }
}

/// Closed artifact-stable projection of [`ScoreSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalScoreSource {
    Lexical,
    SemanticFast,
    SemanticQuality,
    Hybrid,
    Reranked,
}

impl From<ScoreSource> for LexicalScoreSource {
    fn from(value: ScoreSource) -> Self {
        match value {
            ScoreSource::Lexical => Self::Lexical,
            ScoreSource::SemanticFast => Self::SemanticFast,
            ScoreSource::SemanticQuality => Self::SemanticQuality,
            ScoreSource::Hybrid => Self::Hybrid,
            ScoreSource::Reranked => Self::Reranked,
        }
    }
}

/// Whether a supplemental public value is part of this boundary's contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalFieldExposure {
    NotExposed,
    Exposed,
}

/// Exact count request/exposure state for this public boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalCountExposure {
    NotExposed,
    NotRequested,
    ExactRequested,
}

/// Request-derived contract for values not carried directly by `ScoredResult`.
///
/// An exposed field may never silently fall back to `NotExposed`; the adapter
/// must provide one explicit state for every returned hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalExposureContract {
    pub metadata: LexicalFieldExposure,
    pub explanation: LexicalFieldExposure,
    pub total_count: LexicalCountExposure,
    pub snippet: LexicalFieldExposure,
    pub highlight_spans: LexicalFieldExposure,
}

impl LexicalExposureContract {
    /// Ordinary `LexicalSearch` returns no count/snippet/highlight supplement.
    pub const CORE_LEXICAL_SEARCH: Self = Self {
        metadata: LexicalFieldExposure::Exposed,
        explanation: LexicalFieldExposure::Exposed,
        total_count: LexicalCountExposure::NotExposed,
        snippet: LexicalFieldExposure::NotExposed,
        highlight_spans: LexicalFieldExposure::NotExposed,
    };
}

/// Honest evidence for an optional named query transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalNormalizedQuery {
    /// The observed public boundary exposes no normalized query bytes.
    NotExposed,
    /// A named transform emitted these exact normalized bytes.
    Value {
        transform_id: String,
        sha256: String,
        byte_len: u64,
    },
}

impl LexicalNormalizedQuery {
    fn validate(&self) -> bool {
        match self {
            Self::NotExposed => true,
            Self::Value {
                transform_id,
                sha256,
                byte_len,
            } => {
                !transform_id.trim().is_empty()
                    && transform_id.len() <= 256
                    && *byte_len <= u64::try_from(MAX_LEXICAL_QUERY_BYTES).unwrap_or(u64::MAX)
                    && is_lower_sha256(sha256)
            }
        }
    }
}

/// Public lexical boundary observed by one result-contract witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalBoundary {
    /// Ordinary [`frankensearch_core::LexicalSearch::search`] output.
    FullSearch,
    /// Fusion candidates before optional metadata hydration.
    FusionCandidates,
    /// Input containing every candidate projected as a lexical-only final winner.
    FusionHydrationAllLexicalInput,
    /// Lexical-only final-winner state after hydration.
    FusionHydrationAllLexicalPostState,
    /// Input containing a deterministic non-prefix subset of hybrid final winners.
    FusionHydrationHybridSubsetInput,
    /// Hybrid final-winner subset after hydration.
    FusionHydrationHybridSubsetPostState,
    /// Isolated semantic-only control before hydration.
    FusionHydrationSemanticOnlyInput,
    /// Isolated semantic-only control after hydration.
    FusionHydrationSemanticOnlyPostState,
    /// Mixed lexical and non-lexical final winners before hydration.
    FusionHydrationMixedInput,
    /// Mixed lexical and non-lexical final winners after hydration.
    FusionHydrationMixedPostState,
}

/// Backend identity retained as provenance but not treated as an equivalence field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalBackendIdentity {
    /// Stable engine name, such as `quill` or `tantivy`.
    pub engine: String,
    /// Exact engine source or package revision.
    pub revision: String,
    /// Stable public index/source identity for this invocation.
    pub index_identity: String,
}

impl LexicalBackendIdentity {
    fn validate(&self) -> Result<(), GauntletError> {
        if [
            self.engine.as_str(),
            self.revision.as_str(),
            self.index_identity.as_str(),
        ]
        .into_iter()
        .any(|value| value.trim().is_empty() || value.len() > 256)
        {
            return Err(GauntletError::InvalidObservation {
                reason: "lexical backend identity fields must be bounded and non-empty".to_owned(),
            });
        }
        Ok(())
    }
}

/// Query, corpus, seed, and backend provenance for one lexical observation.
///
/// Query text is never persisted. Raw and normalized inputs are represented by
/// exact byte lengths and SHA-256 digests, while the classification that alters
/// retrieval policy remains explicit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalObservationContext {
    /// Observation schema identifier.
    pub schema_version: String,
    /// Exact public boundary that emitted the result.
    pub boundary: LexicalBoundary,
    /// Engine and revision that emitted this observation.
    pub backend: LexicalBackendIdentity,
    /// SHA-256 of the exact indexed corpus.
    pub corpus_sha256: String,
    /// SHA-256 of the analyzer, parser, and query-input contract.
    pub query_contract_sha256: String,
    /// SHA-256 of the raw query bytes.
    pub query_sha256: String,
    /// Raw query byte length.
    pub query_bytes: u64,
    /// Named normalized-query evidence, only when the boundary exposes it.
    pub normalized_query: LexicalNormalizedQuery,
    /// Query class used by adaptive retrieval.
    pub query_class: LexicalQueryClass,
    /// Deterministic case/generator seed.
    pub seed: u64,
    /// Requested top-k limit.
    pub limit: u64,
    /// Request-derived exposure requirements for supplemental fields.
    pub exposure: LexicalExposureContract,
}

impl LexicalObservationContext {
    /// Build a CI-safe context without retaining plaintext query material.
    ///
    /// # Errors
    ///
    /// Rejects malformed corpus/query-contract hashes and unbounded backend
    /// identity fields.
    pub fn new(
        boundary: LexicalBoundary,
        backend: LexicalBackendIdentity,
        corpus_sha256: String,
        query_contract_sha256: String,
        raw_query: &str,
        seed: u64,
        limit: usize,
        exposure: LexicalExposureContract,
    ) -> Result<Self, GauntletError> {
        let context = Self {
            schema_version: LEXICAL_OBSERVATION_SCHEMA_VERSION.to_owned(),
            boundary,
            backend,
            corpus_sha256,
            query_contract_sha256,
            query_sha256: sha256_hex(raw_query.as_bytes()),
            query_bytes: u64::try_from(raw_query.len()).map_err(|_| {
                GauntletError::InvalidObservation {
                    reason: "raw lexical query length does not fit u64".to_owned(),
                }
            })?,
            normalized_query: LexicalNormalizedQuery::NotExposed,
            query_class: QueryClass::classify(raw_query).into(),
            seed,
            limit: u64::try_from(limit).map_err(|_| GauntletError::InvalidObservation {
                reason: "lexical result limit does not fit u64".to_owned(),
            })?,
            exposure,
        };
        context.validate()?;
        Ok(context)
    }

    /// Bind normalized bytes emitted by a specific named public transform.
    ///
    /// This is deliberately separate from [`Self::new`]: ordinary
    /// `LexicalSearch` does not expose its internal parser/analyzer transform
    /// and therefore must retain `NotExposed`.
    #[cfg(test)]
    pub(crate) fn with_normalized_query(
        mut self,
        transform_id: impl Into<String>,
        normalized_query: &str,
    ) -> Result<Self, GauntletError> {
        self.normalized_query = LexicalNormalizedQuery::Value {
            transform_id: transform_id.into(),
            sha256: sha256_hex(normalized_query.as_bytes()),
            byte_len: u64::try_from(normalized_query.len()).map_err(|_| {
                GauntletError::InvalidObservation {
                    reason: "normalized lexical query length does not fit u64".to_owned(),
                }
            })?,
        };
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<(), GauntletError> {
        self.backend.validate()?;
        if self.schema_version != LEXICAL_OBSERVATION_SCHEMA_VERSION
            || !is_lower_sha256(&self.corpus_sha256)
            || !is_lower_sha256(&self.query_contract_sha256)
            || !is_lower_sha256(&self.query_sha256)
            || self.query_bytes > u64::try_from(MAX_LEXICAL_QUERY_BYTES).unwrap_or(u64::MAX)
            || !self.normalized_query.validate()
        {
            return Err(GauntletError::InvalidObservation {
                reason: "lexical context schema or SHA-256 provenance is invalid".to_owned(),
            });
        }
        Ok(())
    }
}

/// Explicit observation state for public values that a boundary may not expose.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(
    tag = "state",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum LexicalObserved<T> {
    /// The selected public boundary does not expose this value.
    #[default]
    NotExposed,
    /// The boundary exposes the field and reported absence.
    Absent,
    /// The boundary exposed this exact value.
    Value(T),
}

/// CI-safe representation of metadata, snippets, explanations, and errors.
///
/// Payload bytes never enter the artifact. Presence, emptiness, byte length,
/// and a stable digest remain independently observable.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum SensitiveValueObservation {
    /// The selected public boundary does not expose this field.
    #[default]
    NotExposed,
    /// The field was exposed and absent.
    Absent,
    /// The field was present but logically empty.
    PresentEmpty {
        /// SHA-256 of the canonical payload bytes.
        sha256: String,
        /// Canonical payload byte length.
        byte_len: u64,
    },
    /// The field was present and non-empty.
    Present {
        /// SHA-256 of the canonical payload bytes.
        sha256: String,
        /// Canonical payload byte length.
        byte_len: u64,
    },
}

impl SensitiveValueObservation {
    /// Hash a snippet or other UTF-8 payload without retaining plaintext.
    #[must_use]
    pub fn from_text(value: &str) -> Self {
        Self::from_bytes(value.as_bytes(), value.is_empty())
    }

    fn from_serializable<T: Serialize>(
        value: &T,
        logically_empty: bool,
    ) -> Result<Self, GauntletError> {
        let bytes = canonical_json_bytes(value)?;
        if bytes.len() > MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES {
            return Err(GauntletError::InvalidObservation {
                reason: "lexical sensitive payload exceeds the observation byte limit".to_owned(),
            });
        }
        Ok(Self::from_bytes(&bytes, logically_empty))
    }

    fn from_bytes(bytes: &[u8], logically_empty: bool) -> Self {
        let sha256 = sha256_hex(bytes);
        let byte_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if logically_empty {
            Self::PresentEmpty { sha256, byte_len }
        } else {
            Self::Present { sha256, byte_len }
        }
    }

    fn validate(&self) -> bool {
        match self {
            Self::NotExposed | Self::Absent => true,
            Self::PresentEmpty { sha256, byte_len } | Self::Present { sha256, byte_len } => {
                *byte_len <= u64::try_from(MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES).unwrap_or(u64::MAX)
                    && is_lower_sha256(sha256)
            }
        }
    }
}

/// Half-open highlight span in UTF-8 byte offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalHighlightSpan {
    /// Inclusive byte offset.
    pub start: u64,
    /// Exclusive byte offset.
    pub end: u64,
}

/// Optional fields supplied by a richer backend boundary for one hit.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalHitSupplement {
    /// Snippet presence and digest.
    pub snippet: SensitiveValueObservation,
    /// Ordered highlight spans, or an explicit unexposed/absent marker.
    pub highlight_spans: LexicalObserved<Vec<LexicalHighlightSpan>>,
}

/// Exact count semantics for a lexical result page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalCountState {
    /// The selected boundary has no total-count contract.
    #[default]
    NotExposed,
    /// The caller explicitly did not request a count.
    NotRequested,
    /// Exact total number of matches.
    Value(u64),
}

/// Rich values supplied beside the common `ScoredResult` envelope.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalObservationSupplement {
    /// Exact count semantics for this page.
    pub total_count: LexicalCountState,
    /// Optional per-hit snippet/highlight observations keyed by document ID.
    pub hits: BTreeMap<String, LexicalHitSupplement>,
}

/// Exhaustive public observation of one lexical hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalHitObservation {
    /// Zero-based native result rank.
    pub rank: u64,
    /// Stable external document identifier.
    pub doc_id: String,
    /// Public normalized/final score bits.
    pub normalized_score_bits: u32,
    /// Raw lexical score bits, when the public result exposes them.
    pub raw_lexical_score_bits: Option<u32>,
    /// Public source identity.
    pub source: LexicalScoreSource,
    /// Public index/source ordinal, when present.
    pub index: Option<u32>,
    /// Fast semantic component bits.
    pub fast_score_bits: Option<u32>,
    /// Quality semantic component bits.
    pub quality_score_bits: Option<u32>,
    /// Reranker component bits.
    pub rerank_score_bits: Option<u32>,
    /// Metadata presence/emptiness/content digest.
    pub metadata: SensitiveValueObservation,
    /// Explanation presence/emptiness/content digest.
    pub explanation: SensitiveValueObservation,
    /// Snippet presence/emptiness/content digest.
    pub snippet: SensitiveValueObservation,
    /// Ordered highlight spans.
    pub highlight_spans: LexicalObserved<Vec<LexicalHighlightSpan>>,
}

/// Explicit shape of a successful result vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalEmptyShape {
    /// Successful search returned zero hits.
    Empty,
    /// Successful search returned at least one hit.
    NonEmpty,
}

/// Stable error class independent of backend error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalErrorClass {
    /// Embedding/model availability or inference.
    Embedding,
    /// Index availability, corruption, version, or dimensions.
    Index,
    /// Query parsing.
    Query,
    /// Time budget exhaustion.
    Timeout,
    /// Federated quorum failure.
    Federated,
    /// Reranking availability or inference.
    Rerank,
    /// File or device I/O.
    Io,
    /// Invalid configuration.
    Configuration,
    /// Artifact hash verification.
    Integrity,
    /// Structured cancellation.
    Cancellation,
    /// Bounded queue capacity.
    Capacity,
    /// Optional backend subsystem.
    Subsystem,
    /// Requested feature was not compiled.
    FeatureDisabled,
}

/// Typed, redacted error observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalErrorObservation {
    /// Stable broad error class.
    pub class: LexicalErrorClass,
    /// Stable variant-level code.
    pub code: String,
    /// Canonical digest of exhaustively matched, stable variant fields.
    pub contract_payload: SensitiveValueObservation,
    /// Hash and length of backend `Display` text for triage only.
    pub diagnostic: SensitiveValueObservation,
    /// Ordered public [`std::error::Error::source`] display chain.
    pub source_chain: Vec<SensitiveValueObservation>,
}

/// Successful or failed public lexical outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalObservationOutcome {
    /// Successful ordered result page.
    Success {
        /// Results in native consumer-visible order.
        hits: Vec<LexicalHitObservation>,
        /// Returned vector length, retained independently for schema checks.
        returned_count: u64,
        /// Exact empty/non-empty result shape.
        empty_shape: LexicalEmptyShape,
        /// Total-count request/value semantics.
        total_count: LexicalCountState,
    },
    /// Typed failure without unrestricted diagnostic text.
    Error(LexicalErrorObservation),
}

/// Complete backend-neutral observation at the public lexical result boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalObservation {
    /// Query/corpus/backend provenance.
    pub context: LexicalObservationContext,
    /// Successful result envelope or typed error.
    pub outcome: LexicalObservationOutcome,
}

/// Engine role bound by the outer differential harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalEngineRole {
    Subject,
    Oracle,
}

/// Candidate selection passed to one hydration invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalHydrationSelection {
    /// Every candidate projected as a lexical-only fused winner.
    AllLexicalWinners,
    /// Deterministic non-prefix subset projected as hybrid fused winners.
    StrictHybridWinnerSubset { candidate_ranks: Vec<u64> },
    /// One isolated semantic-only control with nonempty sentinel payloads.
    SemanticOnlyControl { control_id: u32 },
    /// Ordered production-shaped lexical and non-lexical final winners.
    MixedFinalWinners { origins: Vec<LexicalWinnerOrigin> },
}

/// Production result projection used for a lexical-origin fused winner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalWinnerProjection {
    LexicalOnly,
    HybridFast,
}

/// Kind of deterministic non-lexical control passed through hydration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalNonLexicalControlKind {
    SemanticFast,
    GraphOnlyHybrid,
}

/// Ordered origin of one final fused winner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalWinnerOrigin {
    Lexical {
        candidate_rank: u64,
        projection: LexicalWinnerProjection,
    },
    NonLexicalControl {
        control_id: u32,
        kind: LexicalNonLexicalControlKind,
    },
}

/// Why a hydration probe could not be invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalHydrationNotRunReason {
    CandidateSearchFailed,
    InsufficientCandidates { available: u64, required: u64 },
    InsufficientResultCapacity { limit: u64, required: u64 },
    NoMixedWinnerFixture,
}

/// Public return value from `hydrate_fusion_metadata`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalHydrationResult {
    Success,
    Error(LexicalErrorObservation),
}

/// Method-bound hydration execution evidence.
///
/// `post_state` is retained even on error so partial mutation cannot disappear
/// behind the typed failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalHydrationExecution {
    NotRun {
        reason: LexicalHydrationNotRunReason,
    },
    Attempted {
        input: Box<LexicalObservation>,
        post_state: Box<LexicalObservation>,
        result: LexicalHydrationResult,
    },
}

/// One explicit hydration transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalHydrationTransition {
    pub selection: LexicalHydrationSelection,
    pub execution: LexicalHydrationExecution,
}

/// Complete public lexical result contract for one backend snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalContractBundle {
    /// Bundle schema identifier.
    pub(crate) schema_version: String,
    /// Differential role bound by the live runner.
    pub(crate) engine_role: LexicalEngineRole,
    /// SHA-256 identity of the immutable committed snapshot queried here.
    pub(crate) snapshot_sha256: String,
    /// Capability read from `LexicalSearch::fusion_metadata_is_deferred`.
    pub(crate) fusion_metadata_deferred: bool,
    /// Ordinary full search result.
    pub(crate) full_search: LexicalObservation,
    /// Fusion candidates before hydration.
    pub(crate) fusion_candidates: LexicalObservation,
    /// Hydration of every candidate projected as a lexical-only final winner.
    pub(crate) all_lexical_winners_hydration: LexicalHydrationTransition,
    /// Hydration of a deterministic non-prefix hybrid-winner subset.
    pub(crate) strict_hybrid_winners_hydration: LexicalHydrationTransition,
    /// Isolated no-op probe for a semantic-only winner.
    pub(crate) semantic_only_hydration: LexicalHydrationTransition,
    /// Mixed lexical-only, hybrid, and semantic final winners.
    pub(crate) mixed_winners_hydration: LexicalHydrationTransition,
}

impl LexicalContractBundle {
    #[must_use]
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    #[must_use]
    pub fn engine_role(&self) -> LexicalEngineRole {
        self.engine_role
    }

    #[must_use]
    pub fn snapshot_sha256(&self) -> &str {
        &self.snapshot_sha256
    }

    #[must_use]
    pub fn fusion_metadata_is_deferred(&self) -> bool {
        self.fusion_metadata_deferred
    }

    #[must_use]
    pub fn full_search(&self) -> &LexicalObservation {
        &self.full_search
    }

    #[must_use]
    pub fn fusion_candidates(&self) -> &LexicalObservation {
        &self.fusion_candidates
    }

    #[must_use]
    pub fn all_lexical_winners_hydration(&self) -> &LexicalHydrationTransition {
        &self.all_lexical_winners_hydration
    }

    #[must_use]
    pub fn strict_hybrid_winners_hydration(&self) -> &LexicalHydrationTransition {
        &self.strict_hybrid_winners_hydration
    }

    #[must_use]
    pub fn semantic_only_hydration(&self) -> &LexicalHydrationTransition {
        &self.semantic_only_hydration
    }

    #[must_use]
    pub fn mixed_winners_hydration(&self) -> &LexicalHydrationTransition {
        &self.mixed_winners_hydration
    }
}

/// Registered equivalence laws applied by the lexical comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalEquivalenceLaw {
    /// Result arrays retain native order and rank exactly.
    NativeOrderExact,
    /// Floating-point scores compare by exact IEEE-754 bits.
    ScoreBitsExact,
    /// `NotExposed`, `Absent`, empty, and populated remain distinct.
    PresenceExact,
    /// Sensitive payload contents compare by canonical SHA-256 and length.
    SensitivePayloadDigest,
    /// Count request/value and empty-success shapes compare exactly.
    CountAndEmptyShapeExact,
    /// Error class, stable code, and canonical variant fields compare exactly.
    ///
    /// Backend `Display` text and its source-chain text remain redacted triage
    /// evidence, but are deliberately outside equivalence because neither is
    /// a versioned public data contract.
    TypedErrorExact,
    /// Hydration may not change order, scores, identity, or non-metadata fields.
    HydrationNonMetadataStable,
    /// Successfully hydrated lexical winners equal ordinary full-search results.
    HydratedWinnerMetadataExact,
    /// Winners with no lexical contribution remain bit-identical.
    SemanticOnlyHydrationNoOp,
    /// A successful exercised candidate transition may defer metadata.
    DeferredMetadataHydration,
}

const LEXICAL_EQUIVALENCE_LAWS: [LexicalEquivalenceLaw; 9] = [
    LexicalEquivalenceLaw::NativeOrderExact,
    LexicalEquivalenceLaw::ScoreBitsExact,
    LexicalEquivalenceLaw::PresenceExact,
    LexicalEquivalenceLaw::SensitivePayloadDigest,
    LexicalEquivalenceLaw::CountAndEmptyShapeExact,
    LexicalEquivalenceLaw::TypedErrorExact,
    LexicalEquivalenceLaw::HydrationNonMetadataStable,
    LexicalEquivalenceLaw::HydratedWinnerMetadataExact,
    LexicalEquivalenceLaw::SemanticOnlyHydrationNoOp,
];

/// Stable mismatch taxonomy for result-contract diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalMismatchClass {
    /// Corpus/query/seed/limit provenance differs.
    Context,
    /// Success/error or result arity differs.
    Outcome,
    /// Rank, document identity, or array order differs.
    Ordering,
    /// Raw, normalized, or component score bits differ.
    Score,
    /// Public score source or index identity differs.
    SourceIdentity,
    /// Snippet presence or digest differs.
    Snippet,
    /// Ordered highlight spans differ.
    Highlight,
    /// Metadata presence or digest differs.
    Metadata,
    /// Explanation presence or digest differs.
    Explanation,
    /// Count or empty-result shape differs.
    Count,
    /// Typed error class, code, or redacted digest differs.
    Error,
}

/// One bounded, field-addressable lexical contract mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalFieldMismatch {
    /// Stable mismatch class.
    pub class: LexicalMismatchClass,
    /// JSON pointer into the subject observation.
    pub path: String,
    /// Bounded safe oracle diagnostic.
    pub oracle: String,
    /// Bounded safe subject diagnostic.
    pub subject: String,
}

/// Overall lexical comparison status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalComparisonStatus {
    /// Every registered equivalence law passed.
    Equivalent,
    /// At least one consumer-visible field differed.
    Mismatch,
}

/// Pure, replayable result-contract comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalComparisonReport {
    /// Overall comparison status.
    pub status: LexicalComparisonStatus,
    /// Fixed equivalence laws applied to this comparison.
    pub applied_laws: Vec<LexicalEquivalenceLaw>,
    /// Ordered, field-level mismatches.
    pub mismatches: Vec<LexicalFieldMismatch>,
    /// First differing field for compact logs and triage.
    pub first_mismatch: Option<LexicalFieldMismatch>,
    /// Subject evidence.
    pub subject: LexicalObservation,
    /// Oracle evidence.
    pub oracle: LexicalObservation,
}

/// One reviewed, path-specific difference permitted by a scoped law.
///
/// Waivers remain in immutable evidence; they are never deleted from the
/// comparison merely to manufacture an `Equivalent` status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalDeferredSide {
    Subject,
    Oracle,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalHydrationProbe {
    AllLexicalWinners,
    StrictHybridWinnerSubset,
    SemanticOnlyControl,
    MixedFinalWinners,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalWaiverTarget {
    FusionCandidateMetadata {
        rank: u64,
    },
    HydrationInputMetadata {
        probe: LexicalHydrationProbe,
        position: u64,
        candidate_rank: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalWaivedDifference {
    pub law: LexicalEquivalenceLaw,
    pub deferred_side: LexicalDeferredSide,
    pub target: LexicalWaiverTarget,
    pub mismatch: LexicalFieldMismatch,
}

/// Coverage state remains separate from equivalence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum LexicalProbeCoverage {
    ExercisedSuccess,
    /// A successful hydration changed at least one lexical origin from absent
    /// metadata to the exact full-search value.
    ExercisedRestoration,
    ExercisedError,
    ExercisedEmpty,
    NotRun {
        reason: LexicalHydrationNotRunReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalSideCoverage {
    pub full_search: LexicalProbeCoverage,
    pub fusion_candidates: LexicalProbeCoverage,
    pub all_lexical_winners_hydration: LexicalProbeCoverage,
    pub strict_hybrid_winners_hydration: LexicalProbeCoverage,
    pub semantic_only_hydration: LexicalProbeCoverage,
    pub mixed_winners_hydration: LexicalProbeCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalContractCoverage {
    pub subject: LexicalSideCoverage,
    pub oracle: LexicalSideCoverage,
}

/// Replayable comparison of every required public lexical boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalContractComparison {
    /// Comparison schema identifier.
    pub schema_version: String,
    /// Aggregate status across all searches and hydration transitions.
    pub status: LexicalComparisonStatus,
    /// Registered laws evaluated by the comparator.
    pub applied_laws: Vec<LexicalEquivalenceLaw>,
    /// Exercise/error/empty/not-run state for every required probe.
    pub coverage: LexicalContractCoverage,
    /// Explicit differences permitted only by an exercised scoped law.
    pub waived_differences: Vec<LexicalWaivedDifference>,
    /// Every unwaived field-level mismatch, in deterministic path order.
    pub mismatches: Vec<LexicalFieldMismatch>,
    /// First unwaived mismatch for compact diagnostics.
    pub first_mismatch: Option<LexicalFieldMismatch>,
    /// Subject input evidence retained for replay.
    pub subject: LexicalContractBundle,
    /// Oracle input evidence retained for replay.
    pub oracle: LexicalContractBundle,
}

/// Validated inputs from the outer campaign used to observe one live engine.
///
/// Fields are intentionally private: callers cannot relabel a boundary or
/// claim a capability. [`observe_live_lexical_contract`] constructs every
/// boundary and reads `fusion_metadata_is_deferred` from the engine itself.
#[derive(Debug, Clone)]
pub struct LexicalContractBuildContext<'a> {
    engine_role: LexicalEngineRole,
    backend: LexicalBackendIdentity,
    snapshot_sha256: String,
    corpus_sha256: String,
    query_contract_sha256: String,
    query: &'a str,
    seed: u64,
    limit: usize,
}

impl<'a> LexicalContractBuildContext<'a> {
    pub fn new(
        engine_role: LexicalEngineRole,
        backend: LexicalBackendIdentity,
        snapshot_sha256: String,
        corpus_sha256: String,
        query_contract_sha256: String,
        query: &'a str,
        seed: u64,
        limit: usize,
    ) -> Result<Self, GauntletError> {
        if !is_lower_sha256(&snapshot_sha256) {
            return Err(GauntletError::InvalidObservation {
                reason: "live lexical snapshot identity must be a lowercase SHA-256".to_owned(),
            });
        }
        LexicalObservationContext::new(
            LexicalBoundary::FullSearch,
            backend.clone(),
            corpus_sha256.clone(),
            query_contract_sha256.clone(),
            query,
            seed,
            limit,
            LexicalExposureContract::CORE_LEXICAL_SEARCH,
        )?;
        Ok(Self {
            engine_role,
            backend,
            snapshot_sha256,
            corpus_sha256,
            query_contract_sha256,
            query,
            seed,
            limit,
        })
    }

    fn observation_context(
        &self,
        boundary: LexicalBoundary,
    ) -> Result<LexicalObservationContext, GauntletError> {
        LexicalObservationContext::new(
            boundary,
            self.backend.clone(),
            self.corpus_sha256.clone(),
            self.query_contract_sha256.clone(),
            self.query,
            self.seed,
            self.limit,
            LexicalExposureContract::CORE_LEXICAL_SEARCH,
        )
    }
}

/// Convert the public lexical result/error envelope into a complete observation.
///
/// `ScoredResult` is deliberately destructured without `..`: adding a public
/// result field breaks this adapter at compile time until the new field is
/// deliberately observed or documented.
///
/// # Errors
///
/// Rejects invalid provenance, non-finite scores, unknown supplement document
/// IDs, malformed highlight spans, and supplements attached to an error.
pub fn observe_lexical_outcome(
    context: LexicalObservationContext,
    outcome: Result<Vec<ScoredResult>, SearchError>,
    supplement: &LexicalObservationSupplement,
) -> Result<LexicalObservation, GauntletError> {
    context.validate()?;
    if context.exposure.metadata != LexicalFieldExposure::Exposed
        || context.exposure.explanation != LexicalFieldExposure::Exposed
    {
        return Err(GauntletError::InvalidObservation {
            reason: "the ScoredResult lexical adapter must expose metadata and explanation states"
                .to_owned(),
        });
    }
    let outcome = match outcome {
        Ok(results) => {
            let mut observed_ids = BTreeSet::new();
            let mut hits = Vec::with_capacity(results.len());
            for (rank, result) in results.into_iter().enumerate() {
                let rank = u64::try_from(rank).map_err(|_| GauntletError::InvalidObservation {
                    reason: "lexical result rank does not fit u64".to_owned(),
                })?;
                let ScoredResult {
                    doc_id,
                    score,
                    source,
                    index,
                    fast_score,
                    quality_score,
                    lexical_score,
                    rerank_score,
                    explanation,
                    metadata,
                } = result;
                let doc_id = doc_id.to_string();
                observed_ids.insert(doc_id.clone());
                let hit_supplement = supplement.hits.get(&doc_id).cloned();
                validate_hit_supplement_exposure(
                    context.exposure,
                    &doc_id,
                    hit_supplement.as_ref(),
                )?;
                let (snippet, highlight_spans) = match hit_supplement {
                    Some(hit_supplement) => {
                        (hit_supplement.snippet, hit_supplement.highlight_spans)
                    }
                    None => (
                        SensitiveValueObservation::NotExposed,
                        LexicalObserved::NotExposed,
                    ),
                };
                let metadata = match metadata {
                    None => SensitiveValueObservation::Absent,
                    Some(value) => {
                        let logically_empty =
                            value.as_object().is_some_and(serde_json::Map::is_empty);
                        SensitiveValueObservation::from_serializable(
                            value.as_ref(),
                            logically_empty,
                        )?
                    }
                };
                let explanation = match explanation {
                    None => SensitiveValueObservation::Absent,
                    Some(value) => {
                        SensitiveValueObservation::from_serializable(value.as_ref(), false)?
                    }
                };
                hits.push(LexicalHitObservation {
                    rank,
                    doc_id,
                    normalized_score_bits: score.to_bits(),
                    raw_lexical_score_bits: lexical_score.map(f32::to_bits),
                    source: source.into(),
                    index,
                    fast_score_bits: fast_score.map(f32::to_bits),
                    quality_score_bits: quality_score.map(f32::to_bits),
                    rerank_score_bits: rerank_score.map(f32::to_bits),
                    metadata,
                    explanation,
                    snippet,
                    highlight_spans,
                });
            }
            if let Some(doc_id) = supplement
                .hits
                .keys()
                .find(|doc_id| !observed_ids.contains(doc_id.as_str()))
            {
                return Err(GauntletError::InvalidObservation {
                    reason: format!(
                        "lexical supplement names document {} outside the result page",
                        safe_text_diagnostic(doc_id)
                    ),
                });
            }
            match (context.exposure.total_count, supplement.total_count) {
                (LexicalCountExposure::NotExposed, LexicalCountState::NotExposed)
                | (LexicalCountExposure::NotRequested, LexicalCountState::NotRequested)
                | (LexicalCountExposure::ExactRequested, LexicalCountState::Value(_)) => {}
                _ => {
                    return Err(GauntletError::InvalidObservation {
                        reason: "lexical total-count supplement contradicts the exposure contract"
                            .to_owned(),
                    });
                }
            }
            let returned_count =
                u64::try_from(hits.len()).map_err(|_| GauntletError::InvalidObservation {
                    reason: "lexical returned count does not fit u64".to_owned(),
                })?;
            let empty_shape = if hits.is_empty() {
                LexicalEmptyShape::Empty
            } else {
                LexicalEmptyShape::NonEmpty
            };
            LexicalObservationOutcome::Success {
                hits,
                returned_count,
                empty_shape,
                total_count: supplement.total_count,
            }
        }
        Err(error) => {
            if supplement != &LexicalObservationSupplement::default() {
                return Err(GauntletError::InvalidObservation {
                    reason: "a failed lexical result cannot carry successful-hit supplements"
                        .to_owned(),
                });
            }
            LexicalObservationOutcome::Error(observe_lexical_search_error(&error)?)
        }
    };
    let observation = LexicalObservation { context, outcome };
    validate_lexical_observation(&observation)?;
    Ok(observation)
}

fn observe_borrowed_lexical_outcome(
    context: LexicalObservationContext,
    outcome: &Result<Vec<ScoredResult>, SearchError>,
) -> Result<LexicalObservation, GauntletError> {
    match outcome {
        Ok(results) => observe_lexical_outcome(
            context,
            Ok(results.clone()),
            &LexicalObservationSupplement::default(),
        ),
        Err(error) => {
            let observation = LexicalObservation {
                context,
                outcome: LexicalObservationOutcome::Error(observe_lexical_search_error(error)?),
            };
            validate_lexical_observation(&observation)?;
            Ok(observation)
        }
    }
}

fn observe_successful_lexical_state(
    context: LexicalObservationContext,
    results: Vec<ScoredResult>,
) -> Result<LexicalObservation, GauntletError> {
    observe_lexical_outcome(
        context,
        Ok(results),
        &LexicalObservationSupplement::default(),
    )
}

fn observe_hydration_result(
    result: &Result<(), SearchError>,
) -> Result<LexicalHydrationResult, GauntletError> {
    match result {
        Ok(()) => Ok(LexicalHydrationResult::Success),
        Err(error) => Ok(LexicalHydrationResult::Error(observe_lexical_search_error(
            error,
        )?)),
    }
}

fn not_run_hydration(
    selection: LexicalHydrationSelection,
    reason: LexicalHydrationNotRunReason,
) -> LexicalHydrationTransition {
    LexicalHydrationTransition {
        selection,
        execution: LexicalHydrationExecution::NotRun { reason },
    }
}

fn synthetic_final_score(winner_position: usize) -> Result<f32, GauntletError> {
    let position =
        u32::try_from(winner_position).map_err(|_| GauntletError::InvalidObservation {
            reason: "synthetic winner position does not fit the persisted score schedule"
                .to_owned(),
        })?;
    let bits = 0.02_f32.to_bits().checked_sub(position).ok_or_else(|| {
        GauntletError::InvalidObservation {
            reason: "synthetic winner position exceeds the descending score schedule".to_owned(),
        }
    })?;
    let score = f32::from_bits(bits);
    if score.is_finite() && score > 0.0 {
        Ok(score)
    } else {
        Err(GauntletError::InvalidObservation {
            reason: "synthetic winner score schedule produced an invalid score".to_owned(),
        })
    }
}

fn synthetic_explanation(score: f32) -> HitExplanation {
    HitExplanation {
        final_score: f64::from(score),
        components: Vec::new(),
        phase: ExplanationPhase::Initial,
        rank_movement: None,
    }
}

fn non_lexical_control_kind_tag(kind: LexicalNonLexicalControlKind) -> u8 {
    match kind {
        LexicalNonLexicalControlKind::SemanticFast => 1,
        LexicalNonLexicalControlKind::GraphOnlyHybrid => 2,
    }
}

fn synthetic_control_doc_id(
    corpus_sha256: &str,
    query_sha256: &str,
    control_id: u32,
    kind: LexicalNonLexicalControlKind,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"frankensearch/quill/non-lexical-hydration-probe/v2\0");
    hasher.update(corpus_sha256.as_bytes());
    hasher.update(query_sha256.as_bytes());
    hasher.update(control_id.to_le_bytes());
    hasher.update([non_lexical_control_kind_tag(kind)]);
    format!("gauntlet-non-lexical-{}", lower_hex(&hasher.finalize()))
}

fn synthetic_control_metadata(
    control_id: u32,
    kind: LexicalNonLexicalControlKind,
) -> serde_json::Value {
    serde_json::json!({
        "gauntlet_non_lexical_control": control_id,
        "kind": match kind {
            LexicalNonLexicalControlKind::SemanticFast => "semantic_fast",
            LexicalNonLexicalControlKind::GraphOnlyHybrid => "graph_only_hybrid",
        },
    })
}

fn synthetic_lexical_winner(
    candidate: &ScoredResult,
    candidate_rank: usize,
    winner_position: usize,
    projection: LexicalWinnerProjection,
) -> Result<ScoredResult, GauntletError> {
    let lexical_score =
        candidate
            .lexical_score
            .ok_or_else(|| GauntletError::InvalidObservation {
                reason: "lexical fusion candidate is missing its lexical score".to_owned(),
            })?;
    let score = synthetic_final_score(winner_position)?;
    let semantic_index =
        u32::try_from(candidate_rank).map_err(|_| GauntletError::InvalidObservation {
            reason: "lexical candidate rank does not fit the synthetic semantic index".to_owned(),
        })?;
    let (source, index, fast_score) = match projection {
        LexicalWinnerProjection::LexicalOnly => (ScoreSource::Lexical, None, None),
        LexicalWinnerProjection::HybridFast => {
            (ScoreSource::Hybrid, Some(semantic_index), Some(0.125))
        }
    };
    Ok(ScoredResult {
        doc_id: candidate.doc_id.clone(),
        score,
        source,
        index,
        fast_score,
        quality_score: None,
        lexical_score: Some(lexical_score),
        rerank_score: None,
        explanation: Some(Box::new(synthetic_explanation(score))),
        metadata: candidate.metadata.clone(),
    })
}

fn synthetic_non_lexical_control(
    build: &LexicalContractBuildContext<'_>,
    control_id: u32,
    kind: LexicalNonLexicalControlKind,
    winner_position: usize,
) -> Result<ScoredResult, GauntletError> {
    let doc_id = synthetic_control_doc_id(
        &build.corpus_sha256,
        &sha256_hex(build.query.as_bytes()),
        control_id,
        kind,
    );
    let score = synthetic_final_score(winner_position)?;
    let (source, index, fast_score) = match kind {
        LexicalNonLexicalControlKind::SemanticFast => (
            ScoreSource::SemanticFast,
            Some(u32::MAX - control_id),
            Some(0.25),
        ),
        LexicalNonLexicalControlKind::GraphOnlyHybrid => (ScoreSource::Hybrid, None, None),
    };
    Ok(ScoredResult {
        doc_id: doc_id.into(),
        score,
        source,
        index,
        fast_score,
        quality_score: None,
        lexical_score: None,
        rerank_score: None,
        explanation: Some(Box::new(synthetic_explanation(score))),
        metadata: Some(Arc::new(synthetic_control_metadata(control_id, kind))),
    })
}

async fn observe_hydration_attempt(
    cx: &Cx,
    engine: &dyn LexicalSearch,
    build: &LexicalContractBuildContext<'_>,
    selection: LexicalHydrationSelection,
    input_boundary: LexicalBoundary,
    post_boundary: LexicalBoundary,
    mut results: Vec<ScoredResult>,
) -> Result<LexicalHydrationTransition, GauntletError> {
    let input = observe_successful_lexical_state(
        build.observation_context(input_boundary)?,
        results.clone(),
    )?;
    let hydration_result = engine.hydrate_fusion_metadata(cx, &mut results).await;
    let post_state =
        observe_successful_lexical_state(build.observation_context(post_boundary)?, results)?;
    Ok(LexicalHydrationTransition {
        selection,
        execution: LexicalHydrationExecution::Attempted {
            input: Box::new(input),
            post_state: Box::new(post_state),
            result: observe_hydration_result(&hydration_result)?,
        },
    })
}

/// Observe every public lexical search and hydration transition from one live
/// engine invocation.
///
/// Search failures are retained as typed evidence and do not short-circuit the
/// other search method. Hydration errors retain both the pre-call input and the
/// post-call state so partial mutation remains observable.
pub async fn observe_live_lexical_contract(
    cx: &Cx,
    engine: &dyn LexicalSearch,
    build: LexicalContractBuildContext<'_>,
) -> Result<LexicalContractBundle, GauntletError> {
    let full_result = engine.search(cx, build.query, build.limit).await;
    let candidate_result = engine
        .search_fusion_candidates(cx, build.query, build.limit)
        .await;
    let full_search = observe_borrowed_lexical_outcome(
        build.observation_context(LexicalBoundary::FullSearch)?,
        &full_result,
    )?;
    let fusion_candidates = observe_borrowed_lexical_outcome(
        build.observation_context(LexicalBoundary::FusionCandidates)?,
        &candidate_result,
    )?;
    let fusion_metadata_deferred = engine.fusion_metadata_is_deferred();

    let limit = u64::try_from(build.limit).map_err(|_| GauntletError::InvalidObservation {
        reason: "lexical request limit does not fit persisted hydration evidence".to_owned(),
    })?;
    let semantic_only_hydration = if build.limit == 0 {
        not_run_hydration(
            LexicalHydrationSelection::SemanticOnlyControl { control_id: 0 },
            LexicalHydrationNotRunReason::InsufficientResultCapacity { limit, required: 1 },
        )
    } else {
        observe_hydration_attempt(
            cx,
            engine,
            &build,
            LexicalHydrationSelection::SemanticOnlyControl { control_id: 0 },
            LexicalBoundary::FusionHydrationSemanticOnlyInput,
            LexicalBoundary::FusionHydrationSemanticOnlyPostState,
            vec![synthetic_non_lexical_control(
                &build,
                0,
                LexicalNonLexicalControlKind::SemanticFast,
                0,
            )?],
        )
        .await?
    };

    let (all_lexical_winners_hydration, strict_hybrid_winners_hydration, mixed_winners_hydration) =
        match candidate_result {
            Err(_) => (
                not_run_hydration(
                    LexicalHydrationSelection::AllLexicalWinners,
                    LexicalHydrationNotRunReason::CandidateSearchFailed,
                ),
                not_run_hydration(
                    LexicalHydrationSelection::StrictHybridWinnerSubset {
                        candidate_ranks: Vec::new(),
                    },
                    LexicalHydrationNotRunReason::CandidateSearchFailed,
                ),
                not_run_hydration(
                    LexicalHydrationSelection::MixedFinalWinners {
                        origins: Vec::new(),
                    },
                    LexicalHydrationNotRunReason::CandidateSearchFailed,
                ),
            ),
            Ok(candidates) => {
                let all_winners = candidates
                    .iter()
                    .enumerate()
                    .map(|(rank, candidate)| {
                        synthetic_lexical_winner(
                            candidate,
                            rank,
                            rank,
                            LexicalWinnerProjection::LexicalOnly,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let all_lexical_winners_hydration = observe_hydration_attempt(
                    cx,
                    engine,
                    &build,
                    LexicalHydrationSelection::AllLexicalWinners,
                    LexicalBoundary::FusionHydrationAllLexicalInput,
                    LexicalBoundary::FusionHydrationAllLexicalPostState,
                    all_winners,
                )
                .await?;

                let strict_hybrid_winners_hydration = if candidates.len() >= 2 {
                    let candidate_rank = candidates.len() - 1;
                    observe_hydration_attempt(
                        cx,
                        engine,
                        &build,
                        LexicalHydrationSelection::StrictHybridWinnerSubset {
                            candidate_ranks: vec![u64::try_from(candidate_rank).map_err(|_| {
                                GauntletError::InvalidObservation {
                                    reason: "strict candidate rank does not fit persisted evidence"
                                        .to_owned(),
                                }
                            })?],
                        },
                        LexicalBoundary::FusionHydrationHybridSubsetInput,
                        LexicalBoundary::FusionHydrationHybridSubsetPostState,
                        vec![synthetic_lexical_winner(
                            &candidates[candidate_rank],
                            candidate_rank,
                            0,
                            LexicalWinnerProjection::HybridFast,
                        )?],
                    )
                    .await?
                } else {
                    not_run_hydration(
                        LexicalHydrationSelection::StrictHybridWinnerSubset {
                            candidate_ranks: Vec::new(),
                        },
                        LexicalHydrationNotRunReason::InsufficientCandidates {
                            available: u64::try_from(candidates.len()).map_err(|_| {
                                GauntletError::InvalidObservation {
                                    reason:
                                        "candidate count does not fit persisted hydration evidence"
                                            .to_owned(),
                                }
                            })?,
                            required: 2,
                        },
                    )
                };

                let mixed_winners_hydration = if build.limit < 2 {
                    not_run_hydration(
                        LexicalHydrationSelection::MixedFinalWinners {
                            origins: Vec::new(),
                        },
                        LexicalHydrationNotRunReason::InsufficientResultCapacity {
                            limit,
                            required: 2,
                        },
                    )
                } else if candidates.is_empty() {
                    not_run_hydration(
                        LexicalHydrationSelection::MixedFinalWinners {
                            origins: Vec::new(),
                        },
                        LexicalHydrationNotRunReason::NoMixedWinnerFixture,
                    )
                } else {
                    let last_candidate_rank = candidates.len() - 1;
                    let mut origins = vec![
                        LexicalWinnerOrigin::NonLexicalControl {
                            control_id: 1,
                            kind: LexicalNonLexicalControlKind::SemanticFast,
                        },
                        LexicalWinnerOrigin::Lexical {
                            candidate_rank: u64::try_from(last_candidate_rank).map_err(|_| {
                                GauntletError::InvalidObservation {
                                    reason: "mixed candidate rank does not fit persisted evidence"
                                        .to_owned(),
                                }
                            })?,
                            projection: LexicalWinnerProjection::HybridFast,
                        },
                    ];
                    if build.limit >= 3 && candidates.len() >= 2 {
                        origins.push(LexicalWinnerOrigin::Lexical {
                            candidate_rank: 0,
                            projection: LexicalWinnerProjection::LexicalOnly,
                        });
                    }
                    if build.limit >= 4 {
                        origins.push(LexicalWinnerOrigin::NonLexicalControl {
                            control_id: 2,
                            kind: LexicalNonLexicalControlKind::GraphOnlyHybrid,
                        });
                    }
                    let mut results = Vec::with_capacity(origins.len());
                    for (winner_position, origin) in origins.iter().copied().enumerate() {
                        results.push(match origin {
                            LexicalWinnerOrigin::Lexical {
                                candidate_rank,
                                projection,
                            } => {
                                let candidate_rank =
                                    usize::try_from(candidate_rank).map_err(|_| {
                                        GauntletError::InvalidObservation {
                                            reason:
                                                "mixed candidate rank does not fit the live process"
                                                    .to_owned(),
                                        }
                                    })?;
                                synthetic_lexical_winner(
                                    &candidates[candidate_rank],
                                    candidate_rank,
                                    winner_position,
                                    projection,
                                )?
                            }
                            LexicalWinnerOrigin::NonLexicalControl { control_id, kind } => {
                                synthetic_non_lexical_control(
                                    &build,
                                    control_id,
                                    kind,
                                    winner_position,
                                )?
                            }
                        });
                    }
                    observe_hydration_attempt(
                        cx,
                        engine,
                        &build,
                        LexicalHydrationSelection::MixedFinalWinners { origins },
                        LexicalBoundary::FusionHydrationMixedInput,
                        LexicalBoundary::FusionHydrationMixedPostState,
                        results,
                    )
                    .await?
                };

                (
                    all_lexical_winners_hydration,
                    strict_hybrid_winners_hydration,
                    mixed_winners_hydration,
                )
            }
        };

    let bundle = LexicalContractBundle {
        schema_version: LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION.to_owned(),
        engine_role: build.engine_role,
        snapshot_sha256: build.snapshot_sha256,
        fusion_metadata_deferred,
        full_search,
        fusion_candidates,
        all_lexical_winners_hydration,
        strict_hybrid_winners_hydration,
        semantic_only_hydration,
        mixed_winners_hydration,
    };
    validate_lexical_contract_bundle(&bundle)?;
    Ok(bundle)
}

fn validate_hit_supplement_exposure(
    exposure: LexicalExposureContract,
    doc_id: &str,
    supplement: Option<&LexicalHitSupplement>,
) -> Result<(), GauntletError> {
    let snippets_exposed = exposure.snippet == LexicalFieldExposure::Exposed;
    let highlights_exposed = exposure.highlight_spans == LexicalFieldExposure::Exposed;
    if (snippets_exposed || highlights_exposed) && supplement.is_none() {
        return Err(GauntletError::InvalidObservation {
            reason: format!(
                "lexical result {} is missing an enabled hit supplement",
                safe_text_diagnostic(doc_id)
            ),
        });
    }
    let Some(supplement) = supplement else {
        return Ok(());
    };
    let snippet_state_is_exposed = supplement.snippet != SensitiveValueObservation::NotExposed;
    let highlight_state_is_exposed = supplement.highlight_spans != LexicalObserved::NotExposed;
    if snippets_exposed != snippet_state_is_exposed
        || highlights_exposed != highlight_state_is_exposed
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!(
                "lexical result {} has a hit supplement inconsistent with its exposure contract",
                safe_text_diagnostic(doc_id)
            ),
        });
    }
    Ok(())
}

/// Compare complete lexical result envelopes under the fixed registered laws.
///
/// Backend names and revisions are provenance: Quill and Tantivy are expected
/// to differ there. Corpus/query identity, ordered public results, score bits,
/// presence states, sensitive digests, count shape, and typed errors are exact.
///
/// # Errors
///
/// Rejects malformed observations before comparison.
pub fn compare_lexical_observations(
    subject: LexicalObservation,
    oracle: LexicalObservation,
) -> Result<LexicalComparisonReport, GauntletError> {
    compare_lexical_observations_inner(subject, oracle, true)
}

fn compare_lexical_observations_inner(
    subject: LexicalObservation,
    oracle: LexicalObservation,
    emit_mismatch_log: bool,
) -> Result<LexicalComparisonReport, GauntletError> {
    validate_lexical_observation(&subject)?;
    validate_lexical_observation(&oracle)?;
    let mut mismatches = Vec::new();

    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/boundary",
        &oracle.context.boundary,
        &subject.context.boundary,
    );
    compare_safe_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/corpus_sha256",
        &oracle.context.corpus_sha256,
        &subject.context.corpus_sha256,
    );
    compare_safe_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/query_contract_sha256",
        &oracle.context.query_contract_sha256,
        &subject.context.query_contract_sha256,
    );
    compare_safe_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/query_sha256",
        &oracle.context.query_sha256,
        &subject.context.query_sha256,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/query_bytes",
        &oracle.context.query_bytes,
        &subject.context.query_bytes,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/normalized_query",
        &oracle.context.normalized_query,
        &subject.context.normalized_query,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/query_class",
        &oracle.context.query_class,
        &subject.context.query_class,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/seed",
        &oracle.context.seed,
        &subject.context.seed,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/limit",
        &oracle.context.limit,
        &subject.context.limit,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/exposure",
        &oracle.context.exposure,
        &subject.context.exposure,
    );
    compare_lexical_outcomes(&subject.outcome, &oracle.outcome, &mut mismatches);

    let status = if mismatches.is_empty() {
        LexicalComparisonStatus::Equivalent
    } else {
        LexicalComparisonStatus::Mismatch
    };
    let first_mismatch = mismatches.first().cloned();
    if emit_mismatch_log && let Some(first) = &first_mismatch {
        info!(
            subject_engine = %subject.context.backend.engine,
            subject_revision = %subject.context.backend.revision,
            oracle_engine = %oracle.context.backend.engine,
            oracle_revision = %oracle.context.backend.revision,
            corpus_sha256 = %subject.context.corpus_sha256,
            query_sha256 = %subject.context.query_sha256,
            seed = subject.context.seed,
            field_path = %first.path,
            mismatch_class = ?first.class,
            "lexical result contract mismatch"
        );
    }
    Ok(LexicalComparisonReport {
        status,
        applied_laws: LEXICAL_EQUIVALENCE_LAWS.to_vec(),
        mismatches,
        first_mismatch,
        subject,
        oracle,
    })
}

/// Compare all required public lexical searches and hydration transitions.
///
/// # Errors
///
/// Rejects malformed, incomplete, or internally inconsistent bundles.
pub fn compare_lexical_contracts(
    subject: LexicalContractBundle,
    oracle: LexicalContractBundle,
) -> Result<LexicalContractComparison, GauntletError> {
    compare_lexical_contracts_inner(subject, oracle, true)
}

fn compare_lexical_contracts_inner(
    subject: LexicalContractBundle,
    oracle: LexicalContractBundle,
    emit_mismatch_log: bool,
) -> Result<LexicalContractComparison, GauntletError> {
    validate_lexical_contract_bundle(&subject)?;
    validate_lexical_contract_bundle(&oracle)?;
    if subject.engine_role != LexicalEngineRole::Subject
        || oracle.engine_role != LexicalEngineRole::Oracle
    {
        return Err(GauntletError::InvalidObservation {
            reason: "lexical contract bundles are not bound to subject/oracle roles".to_owned(),
        });
    }
    if subject.snapshot_sha256 != oracle.snapshot_sha256 {
        return Err(GauntletError::InvalidObservation {
            reason: "lexical contract bundles refer to different committed snapshots".to_owned(),
        });
    }

    let full_search = compare_lexical_observations_inner(
        subject.full_search.clone(),
        oracle.full_search.clone(),
        false,
    )?;
    let fusion_candidates = compare_lexical_observations_inner(
        subject.fusion_candidates.clone(),
        oracle.fusion_candidates.clone(),
        false,
    )?;
    let mut mismatches = Vec::new();
    append_prefixed_mismatches(&mut mismatches, "/full_search", full_search.mismatches);

    let mut waived_differences = Vec::new();
    for mut mismatch in fusion_candidates.mismatches {
        mismatch.path = format!("/fusion_candidates{}", mismatch.path);
        if let Some(waiver) = scoped_deferred_metadata_waiver(&mismatch, &subject, &oracle) {
            waived_differences.push(waiver);
        } else {
            mismatches.push(mismatch);
        }
    }

    let mut hydration_mismatches = Vec::new();
    compare_hydration_transitions(
        "/all_lexical_winners_hydration",
        &subject.all_lexical_winners_hydration,
        &oracle.all_lexical_winners_hydration,
        &mut hydration_mismatches,
    )?;
    compare_hydration_transitions(
        "/strict_hybrid_winners_hydration",
        &subject.strict_hybrid_winners_hydration,
        &oracle.strict_hybrid_winners_hydration,
        &mut hydration_mismatches,
    )?;
    compare_hydration_transitions(
        "/semantic_only_hydration",
        &subject.semantic_only_hydration,
        &oracle.semantic_only_hydration,
        &mut hydration_mismatches,
    )?;
    compare_hydration_transitions(
        "/mixed_winners_hydration",
        &subject.mixed_winners_hydration,
        &oracle.mixed_winners_hydration,
        &mut hydration_mismatches,
    )?;
    for mismatch in hydration_mismatches {
        if let Some(waiver) = scoped_deferred_metadata_waiver(&mismatch, &subject, &oracle) {
            waived_differences.push(waiver);
        } else {
            mismatches.push(mismatch);
        }
    }

    let first_mismatch = mismatches.first().cloned();
    let status = if first_mismatch.is_none() {
        LexicalComparisonStatus::Equivalent
    } else {
        LexicalComparisonStatus::Mismatch
    };
    let mut applied_laws = LEXICAL_EQUIVALENCE_LAWS.to_vec();
    if deferred_metadata_waiver_is_admissible(&subject, &oracle) {
        applied_laws.push(LexicalEquivalenceLaw::DeferredMetadataHydration);
    }
    let coverage = LexicalContractCoverage {
        subject: lexical_side_coverage(&subject),
        oracle: lexical_side_coverage(&oracle),
    };

    let comparison = LexicalContractComparison {
        schema_version: LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION.to_owned(),
        status,
        applied_laws,
        coverage,
        waived_differences,
        mismatches,
        first_mismatch,
        subject,
        oracle,
    };
    if emit_mismatch_log {
        log_lexical_contract_mismatch(&comparison);
    }
    Ok(comparison)
}

fn validate_lexical_contract_bundle(bundle: &LexicalContractBundle) -> Result<(), GauntletError> {
    if bundle.schema_version != LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION
        || !is_lower_sha256(&bundle.snapshot_sha256)
    {
        return Err(GauntletError::InvalidObservation {
            reason: "lexical contract bundle schema or snapshot identity is invalid".to_owned(),
        });
    }
    for (expected, observation) in [
        (LexicalBoundary::FullSearch, &bundle.full_search),
        (LexicalBoundary::FusionCandidates, &bundle.fusion_candidates),
    ] {
        validate_lexical_observation(observation)?;
        if observation.context.boundary != expected {
            return Err(GauntletError::InvalidObservation {
                reason: "lexical contract bundle contains a mislabeled boundary".to_owned(),
            });
        }
    }
    if !same_lexical_request_context(
        &bundle.full_search.context,
        &bundle.fusion_candidates.context,
    ) {
        return Err(GauntletError::InvalidObservation {
            reason: "lexical contract bundle lanes do not share one request context".to_owned(),
        });
    }
    if bundle.full_search.context.exposure.metadata != LexicalFieldExposure::Exposed
        || bundle.full_search.context.exposure.explanation != LexicalFieldExposure::Exposed
    {
        return Err(GauntletError::InvalidObservation {
            reason:
                "the core lexical contract must observe metadata and explanation presence states"
                    .to_owned(),
        });
    }
    if bundle.fusion_metadata_deferred
        && matches!(
            (
                &bundle.full_search.outcome,
                &bundle.fusion_candidates.outcome
            ),
            (
                LexicalObservationOutcome::Success { .. },
                LexicalObservationOutcome::Success { .. }
            )
        )
    {
        if !candidate_metadata_is_absent_when_successful(&bundle.fusion_candidates.outcome) {
            return Err(GauntletError::InvalidObservation {
                reason: "deferred lexical fusion candidates expose metadata before hydration"
                    .to_owned(),
            });
        }
        if let Some(path) = lexical_outcome_difference_except_metadata(
            &bundle.full_search.outcome,
            &bundle.fusion_candidates.outcome,
        ) {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "deferred lexical fusion candidates differ from full search at {path}"
                ),
            });
        }
    }
    if !bundle.fusion_metadata_deferred
        && matches!(
            (
                &bundle.full_search.outcome,
                &bundle.fusion_candidates.outcome
            ),
            (
                LexicalObservationOutcome::Success { .. },
                LexicalObservationOutcome::Success { .. }
            )
        )
    {
        let mut candidates_as_full = bundle.fusion_candidates.clone();
        candidates_as_full.context.boundary = LexicalBoundary::FullSearch;
        let report = compare_lexical_observations_inner(
            bundle.full_search.clone(),
            candidates_as_full,
            false,
        )?;
        if report.status != LexicalComparisonStatus::Equivalent {
            return Err(GauntletError::InvalidObservation {
                reason: "nondeferred fusion candidates differ from ordinary full-search results"
                    .to_owned(),
            });
        }
    }

    validate_hydration_transition(
        bundle,
        &bundle.all_lexical_winners_hydration,
        HydrationProbeKind::AllLexical,
    )?;
    validate_hydration_transition(
        bundle,
        &bundle.strict_hybrid_winners_hydration,
        HydrationProbeKind::StrictHybridSubset,
    )?;
    validate_hydration_transition(
        bundle,
        &bundle.semantic_only_hydration,
        HydrationProbeKind::SemanticOnly,
    )?;
    validate_hydration_transition(
        bundle,
        &bundle.mixed_winners_hydration,
        HydrationProbeKind::Mixed,
    )?;
    Ok(())
}

fn log_lexical_contract_mismatch(comparison: &LexicalContractComparison) {
    if let Some(first) = &comparison.first_mismatch {
        info!(
            subject_engine = %comparison.subject.full_search.context.backend.engine,
            subject_revision = %comparison.subject.full_search.context.backend.revision,
            oracle_engine = %comparison.oracle.full_search.context.backend.engine,
            oracle_revision = %comparison.oracle.full_search.context.backend.revision,
            corpus_sha256 = %comparison.subject.full_search.context.corpus_sha256,
            query_sha256 = %comparison.subject.full_search.context.query_sha256,
            seed = comparison.subject.full_search.context.seed,
            field_path = %first.path,
            mismatch_class = ?first.class,
            "lexical contract bundle mismatch"
        );
    }
}

fn same_lexical_request_context(
    left: &LexicalObservationContext,
    right: &LexicalObservationContext,
) -> bool {
    left.schema_version == right.schema_version
        && left.backend == right.backend
        && left.corpus_sha256 == right.corpus_sha256
        && left.query_contract_sha256 == right.query_contract_sha256
        && left.query_sha256 == right.query_sha256
        && left.query_bytes == right.query_bytes
        && left.normalized_query == right.normalized_query
        && left.query_class == right.query_class
        && left.seed == right.seed
        && left.limit == right.limit
        && left.exposure == right.exposure
}

fn candidate_metadata_is_absent_when_successful(outcome: &LexicalObservationOutcome) -> bool {
    match outcome {
        LexicalObservationOutcome::Success { hits, .. } => hits
            .iter()
            .all(|hit| hit.metadata == SensitiveValueObservation::Absent),
        LexicalObservationOutcome::Error(_) => true,
    }
}

fn lexical_outcome_difference_except_metadata(
    full: &LexicalObservationOutcome,
    candidates: &LexicalObservationOutcome,
) -> Option<String> {
    match (full, candidates) {
        (
            LexicalObservationOutcome::Success {
                hits: full_hits,
                returned_count: full_returned,
                empty_shape: full_empty,
                total_count: full_total,
            },
            LexicalObservationOutcome::Success {
                hits: candidate_hits,
                returned_count: candidate_returned,
                empty_shape: candidate_empty,
                total_count: candidate_total,
            },
        ) => {
            if full_returned != candidate_returned {
                return Some("/outcome/returned_count".to_owned());
            }
            if full_empty != candidate_empty {
                return Some("/outcome/empty_shape".to_owned());
            }
            if full_total != candidate_total {
                return Some("/outcome/total_count".to_owned());
            }
            if full_hits.len() != candidate_hits.len() {
                return Some("/outcome/hits/length".to_owned());
            }
            full_hits.iter().zip(candidate_hits).enumerate().find_map(
                |(index, (full_hit, candidate_hit))| {
                    hit_difference_except_metadata(full_hit, candidate_hit)
                        .map(|field| format!("/outcome/hits/{index}/{field}"))
                },
            )
        }
        (LexicalObservationOutcome::Error(full), LexicalObservationOutcome::Error(candidate)) => {
            (full != candidate).then(|| "/outcome/error".to_owned())
        }
        (LexicalObservationOutcome::Success { .. }, LexicalObservationOutcome::Error(_))
        | (LexicalObservationOutcome::Error(_), LexicalObservationOutcome::Success { .. }) => {
            Some("/outcome/kind".to_owned())
        }
    }
}

fn hit_difference_except_metadata(
    full: &LexicalHitObservation,
    candidate: &LexicalHitObservation,
) -> Option<&'static str> {
    if full.rank != candidate.rank {
        Some("rank")
    } else if full.doc_id != candidate.doc_id {
        Some("doc_id")
    } else if full.normalized_score_bits != candidate.normalized_score_bits {
        Some("normalized_score_bits")
    } else if full.raw_lexical_score_bits != candidate.raw_lexical_score_bits {
        Some("raw_lexical_score_bits")
    } else if full.source != candidate.source {
        Some("source")
    } else if full.index != candidate.index {
        Some("index")
    } else if full.fast_score_bits != candidate.fast_score_bits {
        Some("fast_score_bits")
    } else if full.quality_score_bits != candidate.quality_score_bits {
        Some("quality_score_bits")
    } else if full.rerank_score_bits != candidate.rerank_score_bits {
        Some("rerank_score_bits")
    } else if full.explanation != candidate.explanation {
        Some("explanation")
    } else if full.snippet != candidate.snippet {
        Some("snippet")
    } else if full.highlight_spans != candidate.highlight_spans {
        Some("highlight_spans")
    } else if !hits_equal_except_metadata(full, candidate) {
        Some("unclassified_nonmetadata_field")
    } else {
        None
    }
}

#[derive(Clone, Copy)]
enum HydrationProbeKind {
    AllLexical,
    StrictHybridSubset,
    SemanticOnly,
    Mixed,
}

fn validate_hydration_transition(
    bundle: &LexicalContractBundle,
    transition: &LexicalHydrationTransition,
    probe: HydrationProbeKind,
) -> Result<(), GauntletError> {
    validate_hydration_selection(&transition.selection, probe)?;
    let candidate_hits = lexical_success_hits(&bundle.fusion_candidates.outcome);
    match &transition.execution {
        LexicalHydrationExecution::NotRun { reason } => {
            if !valid_hydration_not_run(
                reason,
                &transition.selection,
                probe,
                candidate_hits,
                bundle.full_search.context.limit,
            ) {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical hydration not-run reason contradicts candidate evidence"
                        .to_owned(),
                });
            }
        }
        LexicalHydrationExecution::Attempted {
            input,
            post_state,
            result,
        } => {
            if candidate_hits.is_none() && !matches!(probe, HydrationProbeKind::SemanticOnly) {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical hydration was attempted without successful candidates"
                        .to_owned(),
                });
            }
            let candidate_hits = candidate_hits.unwrap_or(&[]);
            validate_lexical_observation(input)?;
            validate_lexical_observation(post_state)?;
            let (expected_input_boundary, expected_post_boundary) = match probe {
                HydrationProbeKind::AllLexical => (
                    LexicalBoundary::FusionHydrationAllLexicalInput,
                    LexicalBoundary::FusionHydrationAllLexicalPostState,
                ),
                HydrationProbeKind::StrictHybridSubset => (
                    LexicalBoundary::FusionHydrationHybridSubsetInput,
                    LexicalBoundary::FusionHydrationHybridSubsetPostState,
                ),
                HydrationProbeKind::SemanticOnly => (
                    LexicalBoundary::FusionHydrationSemanticOnlyInput,
                    LexicalBoundary::FusionHydrationSemanticOnlyPostState,
                ),
                HydrationProbeKind::Mixed => (
                    LexicalBoundary::FusionHydrationMixedInput,
                    LexicalBoundary::FusionHydrationMixedPostState,
                ),
            };
            if input.context.boundary != expected_input_boundary
                || post_state.context.boundary != expected_post_boundary
                || !same_lexical_request_context(&bundle.full_search.context, &input.context)
                || !same_lexical_request_context(&bundle.full_search.context, &post_state.context)
            {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical hydration evidence has a wrong boundary or request context"
                        .to_owned(),
                });
            }
            validate_hydration_result(result)?;
            let input_hits = require_success_hits(input, "hydration input")?;
            let post_hits = require_success_hits(post_state, "hydration post-state")?;
            if input_hits.len() != post_hits.len() {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical hydration changed candidate arity".to_owned(),
                });
            }
            validate_hydration_input_selection(
                &transition.selection,
                candidate_hits,
                input_hits,
                &input.context,
            )?;
            let metadata_may_change = bundle.fusion_metadata_deferred;
            if !hydration_post_state_is_stable(
                &input.outcome,
                &post_state.outcome,
                metadata_may_change,
            ) {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical hydration changed a forbidden public result field".to_owned(),
                });
            }
            if matches!(result, LexicalHydrationResult::Success)
                && let Some(full_hits) = lexical_success_hits(&bundle.full_search.outcome)
                && !hydrated_lexical_metadata_matches_full_results(input_hits, post_hits, full_hits)
            {
                return Err(GauntletError::InvalidObservation {
                    reason:
                        "successfully hydrated lexical-winner metadata differs from ordinary full search"
                            .to_owned(),
                });
            }
            if !non_lexical_metadata_is_unchanged(input_hits, post_hits) {
                return Err(GauntletError::InvalidObservation {
                    reason: "hydration changed a non-lexical winner's metadata".to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn validate_hydration_selection(
    selection: &LexicalHydrationSelection,
    probe: HydrationProbeKind,
) -> Result<(), GauntletError> {
    let valid = matches!(
        (probe, selection),
        (
            HydrationProbeKind::AllLexical,
            LexicalHydrationSelection::AllLexicalWinners
        ) | (
            HydrationProbeKind::StrictHybridSubset,
            LexicalHydrationSelection::StrictHybridWinnerSubset { .. }
        ) | (
            HydrationProbeKind::SemanticOnly,
            LexicalHydrationSelection::SemanticOnlyControl { control_id: 0 }
        ) | (
            HydrationProbeKind::Mixed,
            LexicalHydrationSelection::MixedFinalWinners { .. }
        )
    );
    if valid {
        Ok(())
    } else {
        Err(GauntletError::InvalidObservation {
            reason: "lexical hydration selection is in the wrong bundle slot".to_owned(),
        })
    }
}

fn valid_hydration_not_run(
    reason: &LexicalHydrationNotRunReason,
    selection: &LexicalHydrationSelection,
    probe: HydrationProbeKind,
    candidate_hits: Option<&[LexicalHitObservation]>,
    request_limit: u64,
) -> bool {
    match reason {
        LexicalHydrationNotRunReason::CandidateSearchFailed => {
            candidate_hits.is_none()
                && match (probe, selection) {
                    (
                        HydrationProbeKind::AllLexical,
                        LexicalHydrationSelection::AllLexicalWinners,
                    ) => true,
                    (
                        HydrationProbeKind::StrictHybridSubset,
                        LexicalHydrationSelection::StrictHybridWinnerSubset { candidate_ranks },
                    ) => candidate_ranks.is_empty(),
                    (
                        HydrationProbeKind::Mixed,
                        LexicalHydrationSelection::MixedFinalWinners { origins },
                    ) => origins.is_empty(),
                    (_, _) => false,
                }
        }
        LexicalHydrationNotRunReason::InsufficientCandidates {
            available,
            required,
        } => {
            matches!(probe, HydrationProbeKind::StrictHybridSubset)
                && matches!(
                    selection,
                    LexicalHydrationSelection::StrictHybridWinnerSubset {
                        candidate_ranks
                    } if candidate_ranks.is_empty()
                )
                && candidate_hits.is_some_and(|hits| {
                    u64::try_from(hits.len()).ok() == Some(*available)
                        && *available < *required
                        && *required == 2
                })
        }
        LexicalHydrationNotRunReason::InsufficientResultCapacity { limit, required } => {
            *limit == request_limit
                && *limit < *required
                && match (probe, selection) {
                    (
                        HydrationProbeKind::SemanticOnly,
                        LexicalHydrationSelection::SemanticOnlyControl { control_id: 0 },
                    ) => *required == 1,
                    (
                        HydrationProbeKind::Mixed,
                        LexicalHydrationSelection::MixedFinalWinners { origins },
                    ) => *required == 2 && origins.is_empty(),
                    (_, _) => false,
                }
        }
        LexicalHydrationNotRunReason::NoMixedWinnerFixture => {
            matches!(probe, HydrationProbeKind::Mixed)
                && request_limit >= 2
                && matches!(
                    selection,
                    LexicalHydrationSelection::MixedFinalWinners { origins }
                        if origins.is_empty()
                )
                && candidate_hits.is_some_and(<[LexicalHitObservation]>::is_empty)
        }
    }
}

fn validate_hydration_input_selection(
    selection: &LexicalHydrationSelection,
    candidates: &[LexicalHitObservation],
    input: &[LexicalHitObservation],
    context: &LexicalObservationContext,
) -> Result<(), GauntletError> {
    let valid = match selection {
        LexicalHydrationSelection::AllLexicalWinners => {
            input.len() == candidates.len()
                && candidates.iter().enumerate().zip(input).all(
                    |((candidate_rank, candidate), observed)| {
                        expected_lexical_winner_hit(
                            candidate,
                            candidate_rank,
                            candidate_rank,
                            LexicalWinnerProjection::LexicalOnly,
                        )
                        .is_some_and(|expected| expected == *observed)
                    },
                )
        }
        LexicalHydrationSelection::StrictHybridWinnerSubset { candidate_ranks } => {
            let unique_ranks = candidate_ranks.iter().copied().collect::<BTreeSet<_>>();
            !candidate_ranks.is_empty()
                && candidate_ranks.len() == input.len()
                && candidate_ranks.len() < candidates.len()
                && unique_ranks.len() == candidate_ranks.len()
                && candidate_ranks
                    .iter()
                    .enumerate()
                    .any(|(position, rank)| usize::try_from(*rank).ok() != Some(position))
                && candidate_ranks.iter().zip(input).enumerate().all(
                    |(output_rank, (candidate_rank, observed))| {
                        usize::try_from(*candidate_rank)
                            .ok()
                            .and_then(|candidate_rank| candidates.get(candidate_rank))
                            .and_then(|candidate| {
                                expected_lexical_winner_hit(
                                    candidate,
                                    usize::try_from(*candidate_rank).ok()?,
                                    output_rank,
                                    LexicalWinnerProjection::HybridFast,
                                )
                            })
                            .is_some_and(|expected| expected == *observed)
                    },
                )
        }
        LexicalHydrationSelection::SemanticOnlyControl { control_id } => {
            *control_id == 0
                && input.len() == 1
                && expected_non_lexical_control_hit(
                    context,
                    *control_id,
                    LexicalNonLexicalControlKind::SemanticFast,
                    0,
                )
                .is_some_and(|expected| expected == input[0])
        }
        LexicalHydrationSelection::MixedFinalWinners { origins } => {
            mixed_hydration_input_is_valid(origins, candidates, input, context)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(GauntletError::InvalidObservation {
            reason: "lexical hydration input does not match its recorded selection".to_owned(),
        })
    }
}

fn synthetic_explanation_observation(score: f32) -> Option<SensitiveValueObservation> {
    SensitiveValueObservation::from_serializable(&synthetic_explanation(score), false).ok()
}

fn expected_lexical_winner_hit(
    candidate: &LexicalHitObservation,
    candidate_rank: usize,
    winner_rank: usize,
    projection: LexicalWinnerProjection,
) -> Option<LexicalHitObservation> {
    let score = synthetic_final_score(winner_rank).ok()?;
    let semantic_index = u32::try_from(candidate_rank).ok()?;
    let (source, index, fast_score_bits) = match projection {
        LexicalWinnerProjection::LexicalOnly => (LexicalScoreSource::Lexical, None, None),
        LexicalWinnerProjection::HybridFast => (
            LexicalScoreSource::Hybrid,
            Some(semantic_index),
            Some(0.125_f32.to_bits()),
        ),
    };
    Some(LexicalHitObservation {
        rank: u64::try_from(winner_rank).ok()?,
        doc_id: candidate.doc_id.clone(),
        normalized_score_bits: score.to_bits(),
        raw_lexical_score_bits: candidate.raw_lexical_score_bits,
        source,
        index,
        fast_score_bits,
        quality_score_bits: None,
        rerank_score_bits: None,
        metadata: candidate.metadata.clone(),
        explanation: synthetic_explanation_observation(score)?,
        snippet: SensitiveValueObservation::NotExposed,
        highlight_spans: LexicalObserved::NotExposed,
    })
}

fn expected_non_lexical_control_hit(
    context: &LexicalObservationContext,
    control_id: u32,
    kind: LexicalNonLexicalControlKind,
    winner_rank: usize,
) -> Option<LexicalHitObservation> {
    let score = synthetic_final_score(winner_rank).ok()?;
    let (source, index, fast_score_bits) = match kind {
        LexicalNonLexicalControlKind::SemanticFast => (
            LexicalScoreSource::SemanticFast,
            Some(u32::MAX - control_id),
            Some(0.25_f32.to_bits()),
        ),
        LexicalNonLexicalControlKind::GraphOnlyHybrid => (LexicalScoreSource::Hybrid, None, None),
    };
    let metadata = synthetic_control_metadata(control_id, kind);
    Some(LexicalHitObservation {
        rank: u64::try_from(winner_rank).ok()?,
        doc_id: synthetic_control_doc_id(
            &context.corpus_sha256,
            &context.query_sha256,
            control_id,
            kind,
        ),
        normalized_score_bits: score.to_bits(),
        raw_lexical_score_bits: None,
        source,
        index,
        fast_score_bits,
        quality_score_bits: None,
        rerank_score_bits: None,
        metadata: SensitiveValueObservation::from_serializable(&metadata, false).ok()?,
        explanation: synthetic_explanation_observation(score)?,
        snippet: SensitiveValueObservation::NotExposed,
        highlight_spans: LexicalObserved::NotExposed,
    })
}

fn mixed_hydration_input_is_valid(
    origins: &[LexicalWinnerOrigin],
    candidates: &[LexicalHitObservation],
    input: &[LexicalHitObservation],
    context: &LexicalObservationContext,
) -> bool {
    if origins.len() != input.len() || origins.len() < 2 {
        return false;
    }
    let mut candidate_ranks = BTreeSet::new();
    let mut control_ids = BTreeSet::new();
    let mut saw_lexical = false;
    let mut saw_non_lexical = false;
    for (winner_position, (origin, observed)) in origins.iter().zip(input).enumerate() {
        let expected = match *origin {
            LexicalWinnerOrigin::Lexical {
                candidate_rank,
                projection,
            } => {
                let Some(candidate_rank) = usize::try_from(candidate_rank).ok() else {
                    return false;
                };
                let Some(candidate) = candidates.get(candidate_rank) else {
                    return false;
                };
                if !candidate_ranks.insert(candidate_rank) {
                    return false;
                }
                saw_lexical = true;
                expected_lexical_winner_hit(candidate, candidate_rank, winner_position, projection)
            }
            LexicalWinnerOrigin::NonLexicalControl { control_id, kind } => {
                if !control_ids.insert(control_id) {
                    return false;
                }
                saw_non_lexical = true;
                expected_non_lexical_control_hit(context, control_id, kind, winner_position)
            }
        };
        if expected.is_none_or(|expected| expected != *observed) {
            return false;
        }
    }
    saw_lexical && saw_non_lexical
}

fn validate_hydration_result(result: &LexicalHydrationResult) -> Result<(), GauntletError> {
    match result {
        LexicalHydrationResult::Success => Ok(()),
        LexicalHydrationResult::Error(error) if valid_lexical_error_observation(error) => Ok(()),
        LexicalHydrationResult::Error(_) => Err(GauntletError::InvalidObservation {
            reason: "lexical hydration error evidence is malformed".to_owned(),
        }),
    }
}

fn lexical_success_hits(outcome: &LexicalObservationOutcome) -> Option<&[LexicalHitObservation]> {
    match outcome {
        LexicalObservationOutcome::Success { hits, .. } => Some(hits),
        LexicalObservationOutcome::Error(_) => None,
    }
}

fn require_success_hits<'a>(
    observation: &'a LexicalObservation,
    label: &str,
) -> Result<&'a [LexicalHitObservation], GauntletError> {
    lexical_success_hits(&observation.outcome).ok_or_else(|| GauntletError::InvalidObservation {
        reason: format!("{label} must retain the mutated result state separately from its error"),
    })
}

fn hydration_post_state_is_stable(
    input: &LexicalObservationOutcome,
    post_state: &LexicalObservationOutcome,
    metadata_may_change: bool,
) -> bool {
    if !metadata_may_change {
        return input == post_state;
    }
    let (Some(input_hits), Some(post_hits)) = (
        lexical_success_hits(input),
        lexical_success_hits(post_state),
    ) else {
        return false;
    };
    input_hits.len() == post_hits.len()
        && input_hits
            .iter()
            .zip(post_hits)
            .all(|(before, after)| hits_equal_except_metadata(before, after))
        && success_shape_without_hits(input) == success_shape_without_hits(post_state)
}

fn success_shape_without_hits(
    outcome: &LexicalObservationOutcome,
) -> Option<(u64, LexicalEmptyShape, LexicalCountState)> {
    match outcome {
        LexicalObservationOutcome::Success {
            returned_count,
            empty_shape,
            total_count,
            ..
        } => Some((*returned_count, *empty_shape, *total_count)),
        LexicalObservationOutcome::Error(_) => None,
    }
}

fn hits_equal_except_metadata(left: &LexicalHitObservation, right: &LexicalHitObservation) -> bool {
    let mut left = left.clone();
    let mut right = right.clone();
    left.metadata = SensitiveValueObservation::NotExposed;
    right.metadata = SensitiveValueObservation::NotExposed;
    left == right
}

fn hydrated_lexical_metadata_matches_full_results(
    input: &[LexicalHitObservation],
    hydrated: &[LexicalHitObservation],
    full: &[LexicalHitObservation],
) -> bool {
    input.len() == hydrated.len()
        && input
            .iter()
            .zip(hydrated)
            .filter(|(before, _)| before.raw_lexical_score_bits.is_some())
            .all(|(_, hydrated_hit)| {
                full.iter()
                    .find(|full_hit| full_hit.doc_id == hydrated_hit.doc_id)
                    .is_some_and(|full_hit| full_hit.metadata == hydrated_hit.metadata)
            })
}

fn non_lexical_metadata_is_unchanged(
    input: &[LexicalHitObservation],
    hydrated: &[LexicalHitObservation],
) -> bool {
    input.len() == hydrated.len()
        && input
            .iter()
            .zip(hydrated)
            .filter(|(before, _)| before.raw_lexical_score_bits.is_none())
            .all(|(before, after)| before.metadata == after.metadata)
}

fn append_prefixed_mismatches(
    destination: &mut Vec<LexicalFieldMismatch>,
    prefix: &str,
    mismatches: Vec<LexicalFieldMismatch>,
) {
    destination.extend(mismatches.into_iter().map(|mut mismatch| {
        mismatch.path = format!("{prefix}{}", mismatch.path);
        mismatch
    }));
}

fn compare_hydration_transitions(
    path: &str,
    subject: &LexicalHydrationTransition,
    oracle: &LexicalHydrationTransition,
    mismatches: &mut Vec<LexicalFieldMismatch>,
) -> Result<(), GauntletError> {
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Context,
        &format!("{path}/selection"),
        &oracle.selection,
        &subject.selection,
    );
    match (&subject.execution, &oracle.execution) {
        (
            LexicalHydrationExecution::NotRun {
                reason: subject_reason,
            },
            LexicalHydrationExecution::NotRun {
                reason: oracle_reason,
            },
        ) => compare_debug_field(
            mismatches,
            LexicalMismatchClass::Outcome,
            &format!("{path}/execution/reason"),
            oracle_reason,
            subject_reason,
        ),
        (
            LexicalHydrationExecution::Attempted {
                input: subject_input,
                post_state: subject_post,
                result: subject_result,
            },
            LexicalHydrationExecution::Attempted {
                input: oracle_input,
                post_state: oracle_post,
                result: oracle_result,
            },
        ) => {
            let input_report = compare_lexical_observations_inner(
                subject_input.as_ref().clone(),
                oracle_input.as_ref().clone(),
                false,
            )?;
            append_prefixed_mismatches(
                mismatches,
                &format!("{path}/input"),
                input_report.mismatches,
            );
            let post_report = compare_lexical_observations_inner(
                subject_post.as_ref().clone(),
                oracle_post.as_ref().clone(),
                false,
            )?;
            append_prefixed_mismatches(
                mismatches,
                &format!("{path}/post_state"),
                post_report.mismatches,
            );
            compare_hydration_results(
                &format!("{path}/result"),
                subject_result,
                oracle_result,
                mismatches,
            );
        }
        (subject_execution, oracle_execution) => push_mismatch(
            mismatches,
            LexicalMismatchClass::Outcome,
            format!("{path}/execution/state"),
            hydration_execution_kind(oracle_execution),
            hydration_execution_kind(subject_execution),
        ),
    }
    Ok(())
}

fn compare_hydration_results(
    path: &str,
    subject: &LexicalHydrationResult,
    oracle: &LexicalHydrationResult,
    mismatches: &mut Vec<LexicalFieldMismatch>,
) {
    match (subject, oracle) {
        (LexicalHydrationResult::Success, LexicalHydrationResult::Success) => {}
        (LexicalHydrationResult::Error(subject), LexicalHydrationResult::Error(oracle)) => {
            let before = mismatches.len();
            compare_lexical_errors(subject, oracle, mismatches);
            for mismatch in &mut mismatches[before..] {
                let suffix = mismatch
                    .path
                    .strip_prefix("/outcome/error")
                    .unwrap_or(&mismatch.path);
                mismatch.path = format!("{path}{suffix}");
            }
        }
        (subject, oracle) => push_mismatch(
            mismatches,
            LexicalMismatchClass::Outcome,
            format!("{path}/kind"),
            hydration_result_kind(oracle),
            hydration_result_kind(subject),
        ),
    }
}

fn hydration_execution_kind(execution: &LexicalHydrationExecution) -> &'static str {
    match execution {
        LexicalHydrationExecution::NotRun { .. } => "not_run",
        LexicalHydrationExecution::Attempted { .. } => "attempted",
    }
}

fn hydration_result_kind(result: &LexicalHydrationResult) -> &'static str {
    match result {
        LexicalHydrationResult::Success => "success",
        LexicalHydrationResult::Error(_) => "error",
    }
}

fn scoped_deferred_metadata_waiver(
    mismatch: &LexicalFieldMismatch,
    subject: &LexicalContractBundle,
    oracle: &LexicalContractBundle,
) -> Option<LexicalWaivedDifference> {
    if mismatch.class != LexicalMismatchClass::Metadata || deferred_side(subject, oracle).is_none()
    {
        return None;
    }
    let (target, probe) = if let Some(rank) =
        metadata_path_rank(&mismatch.path, "/fusion_candidates/outcome/hits/")
    {
        (
            LexicalWaiverTarget::FusionCandidateMetadata { rank },
            LexicalHydrationProbe::AllLexicalWinners,
        )
    } else if let Some(position) = metadata_path_rank(
        &mismatch.path,
        "/all_lexical_winners_hydration/input/outcome/hits/",
    ) {
        if subject.all_lexical_winners_hydration.selection
            != oracle.all_lexical_winners_hydration.selection
            || subject.all_lexical_winners_hydration.selection
                != LexicalHydrationSelection::AllLexicalWinners
        {
            return None;
        }
        (
            LexicalWaiverTarget::HydrationInputMetadata {
                probe: LexicalHydrationProbe::AllLexicalWinners,
                position,
                candidate_rank: position,
            },
            LexicalHydrationProbe::AllLexicalWinners,
        )
    } else if let Some(position) = metadata_path_rank(
        &mismatch.path,
        "/strict_hybrid_winners_hydration/input/outcome/hits/",
    ) {
        (subject.strict_hybrid_winners_hydration.selection
            == oracle.strict_hybrid_winners_hydration.selection)
            .then_some(())?;
        let LexicalHydrationSelection::StrictHybridWinnerSubset { candidate_ranks } =
            &subject.strict_hybrid_winners_hydration.selection
        else {
            return None;
        };
        let candidate_rank = usize::try_from(position)
            .ok()
            .and_then(|position| candidate_ranks.get(position))
            .copied()?;
        (
            LexicalWaiverTarget::HydrationInputMetadata {
                probe: LexicalHydrationProbe::StrictHybridWinnerSubset,
                position,
                candidate_rank,
            },
            LexicalHydrationProbe::StrictHybridWinnerSubset,
        )
    } else {
        let position = metadata_path_rank(
            &mismatch.path,
            "/mixed_winners_hydration/input/outcome/hits/",
        )?;
        (subject.mixed_winners_hydration.selection == oracle.mixed_winners_hydration.selection)
            .then_some(())?;
        let LexicalHydrationSelection::MixedFinalWinners { origins } =
            &subject.mixed_winners_hydration.selection
        else {
            return None;
        };
        let origin = usize::try_from(position)
            .ok()
            .and_then(|position| origins.get(position))?;
        let LexicalWinnerOrigin::Lexical { candidate_rank, .. } = origin else {
            return None;
        };
        (
            LexicalWaiverTarget::HydrationInputMetadata {
                probe: LexicalHydrationProbe::MixedFinalWinners,
                position,
                candidate_rank: *candidate_rank,
            },
            LexicalHydrationProbe::MixedFinalWinners,
        )
    };
    if !deferred_metadata_waiver_is_admissible_for_probe(subject, oracle, probe) {
        return None;
    }
    Some(LexicalWaivedDifference {
        law: LexicalEquivalenceLaw::DeferredMetadataHydration,
        deferred_side: deferred_side(subject, oracle)?,
        target,
        mismatch: mismatch.clone(),
    })
}

fn deferred_metadata_waiver_is_admissible_for_probe(
    subject: &LexicalContractBundle,
    oracle: &LexicalContractBundle,
    probe: LexicalHydrationProbe,
) -> bool {
    (subject.fusion_metadata_deferred || oracle.fusion_metadata_deferred)
        && (!subject.fusion_metadata_deferred
            || metadata_deferral_probe_is_exercised(subject, probe))
        && (!oracle.fusion_metadata_deferred || metadata_deferral_probe_is_exercised(oracle, probe))
}

fn metadata_deferral_probe_is_exercised(
    bundle: &LexicalContractBundle,
    probe: LexicalHydrationProbe,
) -> bool {
    let transition = match probe {
        LexicalHydrationProbe::AllLexicalWinners => &bundle.all_lexical_winners_hydration,
        LexicalHydrationProbe::StrictHybridWinnerSubset => &bundle.strict_hybrid_winners_hydration,
        LexicalHydrationProbe::SemanticOnlyControl => &bundle.semantic_only_hydration,
        LexicalHydrationProbe::MixedFinalWinners => &bundle.mixed_winners_hydration,
    };
    match &transition.execution {
        LexicalHydrationExecution::Attempted {
            input,
            result: LexicalHydrationResult::Success,
            ..
        } => lexical_success_hits(&input.outcome)
            .is_some_and(|hits| hits.iter().any(|hit| hit.raw_lexical_score_bits.is_some())),
        LexicalHydrationExecution::NotRun { .. }
        | LexicalHydrationExecution::Attempted {
            result: LexicalHydrationResult::Error(_),
            ..
        } => false,
    }
}

fn metadata_path_rank(path: &str, prefix: &str) -> Option<u64> {
    let (rank, field) = path.strip_prefix(prefix)?.split_once('/')?;
    if field != "metadata" || rank.is_empty() {
        return None;
    }
    rank.parse().ok()
}

fn deferred_side(
    subject: &LexicalContractBundle,
    oracle: &LexicalContractBundle,
) -> Option<LexicalDeferredSide> {
    match (
        subject.fusion_metadata_deferred,
        oracle.fusion_metadata_deferred,
    ) {
        (true, false) => Some(LexicalDeferredSide::Subject),
        (false, true) => Some(LexicalDeferredSide::Oracle),
        (true, true) => Some(LexicalDeferredSide::Both),
        (false, false) => None,
    }
}

fn deferred_metadata_waiver_is_admissible(
    subject: &LexicalContractBundle,
    oracle: &LexicalContractBundle,
) -> bool {
    (subject.fusion_metadata_deferred || oracle.fusion_metadata_deferred)
        && (!subject.fusion_metadata_deferred || metadata_deferral_is_exercised(subject))
        && (!oracle.fusion_metadata_deferred || metadata_deferral_is_exercised(oracle))
}

fn metadata_deferral_is_exercised(bundle: &LexicalContractBundle) -> bool {
    let Some(candidate_hits) = lexical_success_hits(&bundle.fusion_candidates.outcome) else {
        return false;
    };
    if candidate_hits.is_empty() || lexical_success_hits(&bundle.full_search.outcome).is_none() {
        return false;
    }
    matches!(
        &bundle.all_lexical_winners_hydration.execution,
        LexicalHydrationExecution::Attempted {
            result: LexicalHydrationResult::Success,
            ..
        }
    )
}

fn lexical_side_coverage(bundle: &LexicalContractBundle) -> LexicalSideCoverage {
    LexicalSideCoverage {
        full_search: search_probe_coverage(&bundle.full_search.outcome),
        fusion_candidates: search_probe_coverage(&bundle.fusion_candidates.outcome),
        all_lexical_winners_hydration: hydration_probe_coverage(
            &bundle.all_lexical_winners_hydration.execution,
        ),
        strict_hybrid_winners_hydration: hydration_probe_coverage(
            &bundle.strict_hybrid_winners_hydration.execution,
        ),
        semantic_only_hydration: hydration_probe_coverage(
            &bundle.semantic_only_hydration.execution,
        ),
        mixed_winners_hydration: hydration_probe_coverage(
            &bundle.mixed_winners_hydration.execution,
        ),
    }
}

fn search_probe_coverage(outcome: &LexicalObservationOutcome) -> LexicalProbeCoverage {
    match outcome {
        LexicalObservationOutcome::Success { hits, .. } if hits.is_empty() => {
            LexicalProbeCoverage::ExercisedEmpty
        }
        LexicalObservationOutcome::Success { .. } => LexicalProbeCoverage::ExercisedSuccess,
        LexicalObservationOutcome::Error(_) => LexicalProbeCoverage::ExercisedError,
    }
}

fn hydration_probe_coverage(execution: &LexicalHydrationExecution) -> LexicalProbeCoverage {
    match execution {
        LexicalHydrationExecution::NotRun { reason } => {
            LexicalProbeCoverage::NotRun { reason: *reason }
        }
        LexicalHydrationExecution::Attempted {
            input,
            result: LexicalHydrationResult::Success,
            ..
        } if lexical_success_hits(&input.outcome)
            .is_some_and(<[LexicalHitObservation]>::is_empty) =>
        {
            LexicalProbeCoverage::ExercisedEmpty
        }
        LexicalHydrationExecution::Attempted {
            input,
            post_state,
            result: LexicalHydrationResult::Success,
        } if lexical_metadata_restoration_happened(input, post_state) => {
            LexicalProbeCoverage::ExercisedRestoration
        }
        LexicalHydrationExecution::Attempted {
            result: LexicalHydrationResult::Success,
            ..
        } => LexicalProbeCoverage::ExercisedSuccess,
        LexicalHydrationExecution::Attempted {
            result: LexicalHydrationResult::Error(_),
            ..
        } => LexicalProbeCoverage::ExercisedError,
    }
}

fn lexical_metadata_restoration_happened(
    input: &LexicalObservation,
    post_state: &LexicalObservation,
) -> bool {
    let (Some(input_hits), Some(post_hits)) = (
        lexical_success_hits(&input.outcome),
        lexical_success_hits(&post_state.outcome),
    ) else {
        return false;
    };
    input_hits.iter().zip(post_hits).any(|(before, after)| {
        before.raw_lexical_score_bits.is_some()
            && before.metadata == SensitiveValueObservation::Absent
            && matches!(
                after.metadata,
                SensitiveValueObservation::PresentEmpty { .. }
                    | SensitiveValueObservation::Present { .. }
            )
    })
}

impl LexicalContractComparison {
    /// Recompute every derived field from the retained input bundles.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidContract`] when the stored schema or any
    /// replay-derived field differs from a fresh comparison of the retained
    /// subject and oracle bundles.
    pub fn validate_replay(&self) -> Result<(), GauntletError> {
        if self.schema_version != LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION {
            return Err(GauntletError::InvalidContract {
                reason: "unknown lexical contract comparison schema".to_owned(),
            });
        }
        let recomputed =
            compare_lexical_contracts_inner(self.subject.clone(), self.oracle.clone(), false)?;
        if recomputed != *self {
            return Err(GauntletError::InvalidContract {
                reason: "lexical contract comparison does not match replayed bundles".to_owned(),
            });
        }
        Ok(())
    }
}

fn compare_lexical_outcomes(
    subject: &LexicalObservationOutcome,
    oracle: &LexicalObservationOutcome,
    mismatches: &mut Vec<LexicalFieldMismatch>,
) {
    match (subject, oracle) {
        (
            LexicalObservationOutcome::Success {
                hits: subject_hits,
                returned_count: subject_returned,
                empty_shape: subject_empty,
                total_count: subject_total,
            },
            LexicalObservationOutcome::Success {
                hits: oracle_hits,
                returned_count: oracle_returned,
                empty_shape: oracle_empty,
                total_count: oracle_total,
            },
        ) => {
            compare_debug_field(
                mismatches,
                LexicalMismatchClass::Outcome,
                "/outcome/returned_count",
                oracle_returned,
                subject_returned,
            );
            compare_debug_field(
                mismatches,
                LexicalMismatchClass::Count,
                "/outcome/empty_shape",
                oracle_empty,
                subject_empty,
            );
            compare_debug_field(
                mismatches,
                LexicalMismatchClass::Count,
                "/outcome/total_count",
                oracle_total,
                subject_total,
            );
            compare_debug_field(
                mismatches,
                LexicalMismatchClass::Outcome,
                "/outcome/hits/length",
                &oracle_hits.len(),
                &subject_hits.len(),
            );
            for (index, (subject_hit, oracle_hit)) in
                subject_hits.iter().zip(oracle_hits).enumerate()
            {
                compare_lexical_hits(index, subject_hit, oracle_hit, mismatches);
            }
        }
        (
            LexicalObservationOutcome::Error(subject_error),
            LexicalObservationOutcome::Error(oracle_error),
        ) => compare_lexical_errors(subject_error, oracle_error, mismatches),
        (subject_outcome, oracle_outcome) => push_mismatch(
            mismatches,
            LexicalMismatchClass::Outcome,
            "/outcome/kind",
            outcome_kind(oracle_outcome),
            outcome_kind(subject_outcome),
        ),
    }
}

fn compare_lexical_hits(
    index: usize,
    subject: &LexicalHitObservation,
    oracle: &LexicalHitObservation,
    mismatches: &mut Vec<LexicalFieldMismatch>,
) {
    let path = |field: &str| format!("/outcome/hits/{index}/{field}");
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Ordering,
        &path("rank"),
        &oracle.rank,
        &subject.rank,
    );
    if !oracle.doc_id.eq(&subject.doc_id) {
        push_mismatch(
            mismatches,
            LexicalMismatchClass::Ordering,
            path("doc_id"),
            &safe_text_diagnostic(&oracle.doc_id),
            &safe_text_diagnostic(&subject.doc_id),
        );
    }
    compare_score_bits(
        mismatches,
        &path("normalized_score_bits"),
        oracle.normalized_score_bits,
        subject.normalized_score_bits,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Score,
        &path("raw_lexical_score_bits"),
        &oracle.raw_lexical_score_bits.map(hex_bits),
        &subject.raw_lexical_score_bits.map(hex_bits),
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::SourceIdentity,
        &path("source"),
        &oracle.source,
        &subject.source,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::SourceIdentity,
        &path("index"),
        &oracle.index,
        &subject.index,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Score,
        &path("fast_score_bits"),
        &oracle.fast_score_bits.map(hex_bits),
        &subject.fast_score_bits.map(hex_bits),
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Score,
        &path("quality_score_bits"),
        &oracle.quality_score_bits.map(hex_bits),
        &subject.quality_score_bits.map(hex_bits),
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Score,
        &path("rerank_score_bits"),
        &oracle.rerank_score_bits.map(hex_bits),
        &subject.rerank_score_bits.map(hex_bits),
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Metadata,
        &path("metadata"),
        &oracle.metadata,
        &subject.metadata,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Explanation,
        &path("explanation"),
        &oracle.explanation,
        &subject.explanation,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Snippet,
        &path("snippet"),
        &oracle.snippet,
        &subject.snippet,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Highlight,
        &path("highlight_spans"),
        &oracle.highlight_spans,
        &subject.highlight_spans,
    );
}

fn compare_lexical_errors(
    subject: &LexicalErrorObservation,
    oracle: &LexicalErrorObservation,
    mismatches: &mut Vec<LexicalFieldMismatch>,
) {
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Error,
        "/outcome/error/class",
        &oracle.class,
        &subject.class,
    );
    compare_safe_field(
        mismatches,
        LexicalMismatchClass::Error,
        "/outcome/error/code",
        &oracle.code,
        &subject.code,
    );
    compare_debug_field(
        mismatches,
        LexicalMismatchClass::Error,
        "/outcome/error/contract_payload",
        &oracle.contract_payload,
        &subject.contract_payload,
    );
}

fn compare_score_bits(
    mismatches: &mut Vec<LexicalFieldMismatch>,
    path: &str,
    oracle: u32,
    subject: u32,
) {
    if !oracle.eq(&subject) {
        push_mismatch(
            mismatches,
            LexicalMismatchClass::Score,
            path.to_owned(),
            &hex_bits(oracle),
            &hex_bits(subject),
        );
    }
}

fn compare_safe_field<T: AsRef<str> + PartialEq>(
    mismatches: &mut Vec<LexicalFieldMismatch>,
    class: LexicalMismatchClass,
    path: &str,
    oracle: &T,
    subject: &T,
) {
    if !oracle.eq(subject) {
        push_mismatch(
            mismatches,
            class,
            path.to_owned(),
            oracle.as_ref(),
            subject.as_ref(),
        );
    }
}

fn compare_debug_field<T: std::fmt::Debug + PartialEq>(
    mismatches: &mut Vec<LexicalFieldMismatch>,
    class: LexicalMismatchClass,
    path: &str,
    oracle: &T,
    subject: &T,
) {
    if !oracle.eq(subject) {
        push_mismatch(
            mismatches,
            class,
            path.to_owned(),
            &format!("{oracle:?}"),
            &format!("{subject:?}"),
        );
    }
}

fn push_mismatch(
    mismatches: &mut Vec<LexicalFieldMismatch>,
    class: LexicalMismatchClass,
    path: impl Into<String>,
    oracle: &str,
    subject: &str,
) {
    mismatches.push(LexicalFieldMismatch {
        class,
        path: path.into(),
        oracle: bounded_diagnostic(oracle),
        subject: bounded_diagnostic(subject),
    });
}

fn validate_lexical_observation(observation: &LexicalObservation) -> Result<(), GauntletError> {
    observation.context.validate()?;
    match &observation.outcome {
        LexicalObservationOutcome::Success {
            hits,
            returned_count,
            empty_shape,
            total_count,
        } => {
            let hit_count =
                u64::try_from(hits.len()).map_err(|_| GauntletError::InvalidObservation {
                    reason: "lexical hit count does not fit u64".to_owned(),
                })?;
            if *returned_count != hit_count
                || hits.len() > MAX_LEXICAL_OBSERVATION_HITS
                || hit_count > observation.context.limit
                || (*empty_shape == LexicalEmptyShape::Empty) != hits.is_empty()
                || matches!(total_count, LexicalCountState::Value(total) if *total < u64::try_from(hits.len()).unwrap_or(u64::MAX))
                || !count_state_matches_exposure(
                    *total_count,
                    observation.context.exposure.total_count,
                )
                || !scores_are_non_increasing(hits)
            {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical returned-count, empty-shape, or total-count evidence is inconsistent"
                        .to_owned(),
                });
            }
            let mut doc_ids = BTreeSet::new();
            for (expected_rank, hit) in hits.iter().enumerate() {
                let expected_rank = u64::try_from(expected_rank).map_err(|_| {
                    GauntletError::InvalidObservation {
                        reason: "lexical expected rank does not fit u64".to_owned(),
                    }
                })?;
                if hit.rank != expected_rank
                    || hit.doc_id.is_empty()
                    || hit.doc_id.len() > MAX_LEXICAL_DOC_ID_BYTES
                    || !doc_ids.insert(hit.doc_id.as_str())
                    || !float_bits_are_finite(hit.normalized_score_bits)
                    || !optional_float_bits_are_finite(hit.raw_lexical_score_bits)
                    || !optional_float_bits_are_finite(hit.fast_score_bits)
                    || !optional_float_bits_are_finite(hit.quality_score_bits)
                    || !optional_float_bits_are_finite(hit.rerank_score_bits)
                    || !hit_matches_boundary_score_contract(observation.context.boundary, hit)
                    || !hit.metadata.validate()
                    || !hit.explanation.validate()
                    || !hit.snippet.validate()
                    || !valid_highlight_state(&hit.highlight_spans)
                    || !snippet_and_highlights_are_coherent(&hit.snippet, &hit.highlight_spans)
                    || !sensitive_state_matches_exposure(
                        &hit.metadata,
                        observation.context.exposure.metadata,
                    )
                    || !sensitive_state_matches_exposure(
                        &hit.explanation,
                        observation.context.exposure.explanation,
                    )
                    || !sensitive_state_matches_exposure(
                        &hit.snippet,
                        observation.context.exposure.snippet,
                    )
                    || !highlight_state_matches_exposure(
                        &hit.highlight_spans,
                        observation.context.exposure.highlight_spans,
                    )
                {
                    return Err(GauntletError::InvalidObservation {
                        reason: format!(
                            "lexical hit {} violates rank, identity, score, payload, or highlight invariants",
                            safe_text_diagnostic(&hit.doc_id)
                        ),
                    });
                }
            }
        }
        LexicalObservationOutcome::Error(error) => {
            if !valid_lexical_error_observation(error) {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical error code or redacted payload is invalid".to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn hit_matches_boundary_score_contract(
    boundary: LexicalBoundary,
    hit: &LexicalHitObservation,
) -> bool {
    match boundary {
        LexicalBoundary::FullSearch | LexicalBoundary::FusionCandidates => {
            hit.source == LexicalScoreSource::Lexical
                && hit.index.is_none()
                && hit.raw_lexical_score_bits == Some(hit.normalized_score_bits)
                && hit.fast_score_bits.is_none()
                && hit.quality_score_bits.is_none()
                && hit.rerank_score_bits.is_none()
        }
        LexicalBoundary::FusionHydrationAllLexicalInput
        | LexicalBoundary::FusionHydrationAllLexicalPostState => {
            hit.source == LexicalScoreSource::Lexical
                && hit.index.is_none()
                && hit.raw_lexical_score_bits.is_some()
                && hit.fast_score_bits.is_none()
                && hit.quality_score_bits.is_none()
                && hit.rerank_score_bits.is_none()
        }
        LexicalBoundary::FusionHydrationHybridSubsetInput
        | LexicalBoundary::FusionHydrationHybridSubsetPostState => {
            hit.source == LexicalScoreSource::Hybrid
                && hit.index.is_some()
                && hit.raw_lexical_score_bits.is_some()
                && hit.fast_score_bits.is_some()
                && hit.quality_score_bits.is_none()
                && hit.rerank_score_bits.is_none()
        }
        LexicalBoundary::FusionHydrationSemanticOnlyInput
        | LexicalBoundary::FusionHydrationSemanticOnlyPostState => {
            hit.source == LexicalScoreSource::SemanticFast
                && hit.index.is_some()
                && hit.raw_lexical_score_bits.is_none()
                && hit.fast_score_bits.is_some()
                && hit.quality_score_bits.is_none()
                && hit.rerank_score_bits.is_none()
        }
        LexicalBoundary::FusionHydrationMixedInput
        | LexicalBoundary::FusionHydrationMixedPostState => match hit.source {
            LexicalScoreSource::Lexical => {
                hit.index.is_none()
                    && hit.raw_lexical_score_bits.is_some()
                    && hit.fast_score_bits.is_none()
                    && hit.quality_score_bits.is_none()
                    && hit.rerank_score_bits.is_none()
            }
            LexicalScoreSource::Hybrid => {
                ((hit.index.is_some()
                    && hit.raw_lexical_score_bits.is_some()
                    && hit.fast_score_bits.is_some())
                    || (hit.index.is_none()
                        && hit.raw_lexical_score_bits.is_none()
                        && hit.fast_score_bits.is_none()))
                    && hit.quality_score_bits.is_none()
                    && hit.rerank_score_bits.is_none()
            }
            LexicalScoreSource::SemanticFast => {
                hit.index.is_some()
                    && hit.raw_lexical_score_bits.is_none()
                    && hit.fast_score_bits.is_some()
                    && hit.quality_score_bits.is_none()
                    && hit.rerank_score_bits.is_none()
            }
            LexicalScoreSource::SemanticQuality | LexicalScoreSource::Reranked => false,
        },
    }
}

fn scores_are_non_increasing(hits: &[LexicalHitObservation]) -> bool {
    hits.windows(2).all(|pair| {
        f32::from_bits(pair[0].normalized_score_bits)
            >= f32::from_bits(pair[1].normalized_score_bits)
    })
}

fn count_state_matches_exposure(state: LexicalCountState, exposure: LexicalCountExposure) -> bool {
    matches!(
        (exposure, state),
        (
            LexicalCountExposure::NotExposed,
            LexicalCountState::NotExposed
        ) | (
            LexicalCountExposure::NotRequested,
            LexicalCountState::NotRequested
        ) | (
            LexicalCountExposure::ExactRequested,
            LexicalCountState::Value(_)
        )
    )
}

fn sensitive_state_matches_exposure(
    state: &SensitiveValueObservation,
    exposure: LexicalFieldExposure,
) -> bool {
    match exposure {
        LexicalFieldExposure::NotExposed => {
            matches!(state, SensitiveValueObservation::NotExposed)
        }
        LexicalFieldExposure::Exposed => !matches!(state, SensitiveValueObservation::NotExposed),
    }
}

fn highlight_state_matches_exposure(
    state: &LexicalObserved<Vec<LexicalHighlightSpan>>,
    exposure: LexicalFieldExposure,
) -> bool {
    match exposure {
        LexicalFieldExposure::NotExposed => matches!(state, LexicalObserved::NotExposed),
        LexicalFieldExposure::Exposed => !matches!(state, LexicalObserved::NotExposed),
    }
}

fn valid_lexical_error_observation(error: &LexicalErrorObservation) -> bool {
    !error.code.is_empty()
        && error.code.len() <= 128
        && error.contract_payload.validate()
        && error.diagnostic.validate()
        && error.source_chain.len() <= MAX_LEXICAL_ERROR_SOURCE_DEPTH
        && error
            .source_chain
            .iter()
            .all(SensitiveValueObservation::validate)
}

fn valid_highlight_state(spans: &LexicalObserved<Vec<LexicalHighlightSpan>>) -> bool {
    let LexicalObserved::Value(spans) = spans else {
        return true;
    };
    if spans.len() > MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT {
        return false;
    }
    let mut previous_end = 0;
    for span in spans {
        if span.start >= span.end || span.start < previous_end {
            return false;
        }
        previous_end = span.end;
    }
    true
}

fn snippet_and_highlights_are_coherent(
    snippet: &SensitiveValueObservation,
    highlights: &LexicalObserved<Vec<LexicalHighlightSpan>>,
) -> bool {
    let LexicalObserved::Value(highlights) = highlights else {
        return true;
    };
    let snippet_len = match snippet {
        SensitiveValueObservation::PresentEmpty { byte_len, .. }
        | SensitiveValueObservation::Present { byte_len, .. } => *byte_len,
        SensitiveValueObservation::NotExposed | SensitiveValueObservation::Absent => {
            return highlights.is_empty();
        }
    };
    highlights.last().is_none_or(|span| span.end <= snippet_len)
}

pub fn observe_lexical_search_error(
    error: &SearchError,
) -> Result<LexicalErrorObservation, GauntletError> {
    let redact = |value: &str| SensitiveValueObservation::from_text(value);
    let redact_path = |path: &std::path::Path| redact(&path.to_string_lossy());
    let fixed_width = |value: usize, field: &str| {
        u64::try_from(value).map_err(|_| GauntletError::InvalidObservation {
            reason: format!("lexical error field {field} does not fit persisted u64 evidence"),
        })
    };
    let (class, code, contract_fields) = match error {
        SearchError::EmbedderUnavailable { model, reason } => (
            LexicalErrorClass::Embedding,
            "embedder_unavailable",
            serde_json::json!({"model": redact(model), "reason": redact(reason)}),
        ),
        SearchError::EmbeddingFailed { model, source: _ } => (
            LexicalErrorClass::Embedding,
            "embedding_failed",
            serde_json::json!({"model": redact(model)}),
        ),
        SearchError::ModelNotFound { name } => (
            LexicalErrorClass::Embedding,
            "model_not_found",
            serde_json::json!({"name": redact(name)}),
        ),
        SearchError::ModelLoadFailed { path, source: _ } => (
            LexicalErrorClass::Embedding,
            "model_load_failed",
            serde_json::json!({"path": redact_path(path)}),
        ),
        SearchError::IndexCorrupted { path, detail } => (
            LexicalErrorClass::Index,
            "index_corrupted",
            serde_json::json!({"path": redact_path(path), "detail": redact(detail)}),
        ),
        SearchError::IndexVersionMismatch { expected, found } => (
            LexicalErrorClass::Index,
            "index_version_mismatch",
            serde_json::json!({"expected": expected, "found": found}),
        ),
        SearchError::DimensionMismatch { expected, found } => (
            LexicalErrorClass::Index,
            "dimension_mismatch",
            serde_json::json!({
                "expected": fixed_width(*expected, "dimension.expected")?,
                "found": fixed_width(*found, "dimension.found")?,
            }),
        ),
        SearchError::IndexNotFound { path } => (
            LexicalErrorClass::Index,
            "index_not_found",
            serde_json::json!({"path": redact_path(path)}),
        ),
        SearchError::IndexCandidatesNotFound { paths } => (
            LexicalErrorClass::Index,
            "index_candidates_not_found",
            serde_json::json!({
                "paths": paths.iter().map(|path| redact_path(path)).collect::<Vec<_>>()
            }),
        ),
        SearchError::QueryParseError { query, detail } => (
            LexicalErrorClass::Query,
            "query_parse_error",
            serde_json::json!({"query": redact(query), "detail": redact(detail)}),
        ),
        SearchError::SearchTimeout {
            elapsed_ms,
            budget_ms,
        } => (
            LexicalErrorClass::Timeout,
            "search_timeout",
            serde_json::json!({"elapsed_ms": elapsed_ms, "budget_ms": budget_ms}),
        ),
        SearchError::FederatedInsufficientResponses { required, received } => (
            LexicalErrorClass::Federated,
            "federated_insufficient_responses",
            serde_json::json!({
                "required": fixed_width(*required, "federated.required")?,
                "received": fixed_width(*received, "federated.received")?,
            }),
        ),
        SearchError::RerankerUnavailable { model } => (
            LexicalErrorClass::Rerank,
            "reranker_unavailable",
            serde_json::json!({"model": redact(model)}),
        ),
        SearchError::RerankFailed { model, source: _ } => (
            LexicalErrorClass::Rerank,
            "rerank_failed",
            serde_json::json!({"model": redact(model)}),
        ),
        SearchError::Io(source) => (
            LexicalErrorClass::Io,
            "io",
            serde_json::json!({
                "kind": format!("{:?}", source.kind()),
                "raw_os_error": source.raw_os_error(),
            }),
        ),
        SearchError::InvalidConfig {
            field,
            value,
            reason,
        } => (
            LexicalErrorClass::Configuration,
            "invalid_config",
            serde_json::json!({
                "field": redact(field),
                "value": redact(value),
                "reason": redact(reason),
            }),
        ),
        SearchError::HashMismatch {
            path,
            expected,
            actual,
        } => (
            LexicalErrorClass::Integrity,
            "hash_mismatch",
            serde_json::json!({
                "path": redact_path(path),
                "expected": redact(expected),
                "actual": redact(actual),
            }),
        ),
        SearchError::UnverifiableRemoteSpace { producer, reason } => (
            LexicalErrorClass::Integrity,
            "unverifiable_remote_space",
            serde_json::json!({"producer": redact(producer), "reason": redact(reason)}),
        ),
        SearchError::Cancelled { phase, reason } => (
            LexicalErrorClass::Cancellation,
            "cancelled",
            serde_json::json!({"phase": redact(phase), "reason": redact(reason)}),
        ),
        SearchError::QueueFull { pending, capacity } => (
            LexicalErrorClass::Capacity,
            "queue_full",
            serde_json::json!({
                "pending": fixed_width(*pending, "queue.pending")?,
                "capacity": fixed_width(*capacity, "queue.capacity")?,
            }),
        ),
        SearchError::SubsystemError {
            subsystem,
            source: _,
        } => (
            LexicalErrorClass::Subsystem,
            "subsystem_error",
            serde_json::json!({"subsystem": redact(subsystem)}),
        ),
        SearchError::DurabilityDisabled => (
            LexicalErrorClass::FeatureDisabled,
            "durability_disabled",
            serde_json::json!({}),
        ),
    };
    let mut source_chain = Vec::new();
    let mut source = error.source();
    while let Some(current) = source {
        if source_chain.len() == MAX_LEXICAL_ERROR_SOURCE_DEPTH {
            return Err(GauntletError::InvalidObservation {
                reason: "lexical error source chain exceeds the persisted depth limit".to_owned(),
            });
        }
        source_chain.push(SensitiveValueObservation::from_text(&current.to_string()));
        source = current.source();
    }
    Ok(LexicalErrorObservation {
        class,
        code: code.to_owned(),
        contract_payload: SensitiveValueObservation::from_serializable(
            &contract_fields,
            contract_fields
                .as_object()
                .is_some_and(serde_json::Map::is_empty),
        )?,
        diagnostic: SensitiveValueObservation::from_text(&error.to_string()),
        source_chain,
    })
}

fn outcome_kind(outcome: &LexicalObservationOutcome) -> &'static str {
    match outcome {
        LexicalObservationOutcome::Success { .. } => "success",
        LexicalObservationOutcome::Error(_) => "error",
    }
}

fn optional_float_bits_are_finite(bits: Option<u32>) -> bool {
    bits.is_none_or(float_bits_are_finite)
}

fn float_bits_are_finite(bits: u32) -> bool {
    f32::from_bits(bits).is_finite()
}

fn hex_bits(bits: u32) -> String {
    format!("0x{bits:08x}")
}

fn safe_text_diagnostic(value: &str) -> String {
    let digest = sha256_hex(value.as_bytes());
    format!("sha256:{} bytes={}", &digest[..16], value.len())
}

fn bounded_diagnostic(value: &str) -> String {
    const MAX_DIAGNOSTIC_CHARS: usize = 192;
    if value.chars().count() <= MAX_DIAGNOSTIC_CHARS {
        return value.to_owned();
    }
    let mut bounded = value.chars().take(MAX_DIAGNOSTIC_CHARS).collect::<String>();
    bounded.push('…');
    bounded
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let value = canonicalize_json_value(serde_json::to_value(value)?);
    serde_json::to_vec(&value)
}

fn canonicalize_json_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json_value).collect())
        }
        serde_json::Value::Object(values) => {
            let sorted = values.into_iter().collect::<BTreeMap<_, _>>();
            let mut canonical = serde_json::Map::new();
            for (key, value) in sorted {
                canonical.insert(key, canonicalize_json_value(value));
            }
            serde_json::Value::Object(canonical)
        }
        scalar => scalar,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    lower_hex(&digest)
}

fn lower_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Native secondary ordering evidence retained for every ranked hit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NativeTieKey {
    /// Quill's globally assigned document ID.
    QuillDocId { doc_id: u32 },
    /// Tantivy's full segment-qualified document address.
    TantivyDocAddress { segment_ord: u32, doc_id: u32 },
}

/// One engine-native ranked hit. Scores are stored as raw bits in artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedHit {
    pub doc_id: String,
    pub score_bits: u32,
    pub native_tie_key: NativeTieKey,
}

impl RankedHit {
    /// Recover the score without changing its bit pattern.
    #[must_use]
    pub fn score(&self) -> f32 {
        f32::from_bits(self.score_bits)
    }
}

/// Complete observable output for one differential query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineObservation {
    /// Native top-k order. The comparator never sorts this vector by external ID.
    pub hits: Vec<RankedHit>,
    /// Full oracle cutoff tie group when top-k cuts an exact-score group.
    pub cutoff_tie_group: Vec<RankedHit>,
    /// Whether `cutoff_tie_group` is proven complete rather than fetch-limited.
    pub cutoff_tie_complete: bool,
    /// Full oracle tie group at the page's leading (offset) boundary, when
    /// the first returned rank cuts an exact-score group. Same evidence
    /// contract as `cutoff_tie_group`, for the offset edge: without it, a
    /// page starting inside a tie cannot prove its membership is order-only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub offset_tie_group: Vec<RankedHit>,
    /// Whether `offset_tie_group` is proven complete rather than
    /// fetch-limited. Defaults to the conservative `false` (no proof) so
    /// legacy artifacts without leading evidence keep their behavior.
    #[serde(default, skip_serializing_if = "is_false")]
    pub offset_tie_complete: bool,
    /// Snippets keyed by external document ID.
    pub snippets: BTreeMap<String, String>,
    /// Exact match count, or an explicit marker that the case did not request it.
    pub match_count: CountState,
    /// Exact live-document count.
    pub doc_count: u64,
    /// AST/diagnostic lowering differences the engine recorded while
    /// executing this query. Result-level equivalence is still proven by the
    /// rank/count comparison; these records make intentional lowerings
    /// (register classes) visible instead of silent. Empty records are
    /// omitted from canonical bytes so artifact hashes are unchanged for
    /// queries with no recorded lowering difference.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ast_differences: Vec<AstDifference>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde skip_serializing_if protocol
const fn is_false(value: &bool) -> bool {
    !*value
}

/// Stable taxonomy for AST/diagnostic lowering differences. New kinds must
/// land with a divergence-register class in the same commit; the comparator
/// fails closed on kinds it cannot classify.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AstLoweringKind {
    /// An oversized (>65,530-byte) query token lowered to `MatchNone` under
    /// Quill's symmetric admission rule (register DIV-004).
    OversizedQueryToken,
    /// A match- or score-affecting parser/canonicalization shape differs from
    /// the reviewed logical query contract.
    QueryCanonicalization,
    /// Quill deliberately refuses to reproduce behavior proven incorrect in
    /// the pinned oracle.
    OracleBug,
    /// A bounded wildcard expansion selected a different admissible subset.
    GlobExpansionLimit,
    /// Snapshot statistics differ at a reviewed lifecycle boundary.
    StatsSemantics,
    /// Analyzer behavior differs on one reviewed Unicode edge case.
    UnicodeEdge,
}

/// One recorded AST/diagnostic lowering difference between subject and oracle
/// for the same logical query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AstDifference {
    /// Stable lowering class.
    pub kind: AstLoweringKind,
    /// Human-reviewable oracle AST/diagnostic summary.
    pub oracle: String,
    /// Human-reviewable subject AST/diagnostic summary.
    pub subject: String,
}

/// Query-count observation. Missing evidence is never treated as zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CountState {
    NotRequested,
    Value(u64),
}

/// Reviewed reason that permits `ScoreEpsilon` instead of `RankExact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreEpsilonReason {
    OracleSegmentGeometry,
    PlatformLibm,
}

/// Comparator configuration encoded without JSON floating-point ambiguity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparatorConfig {
    pub score_epsilon_bits: u32,
    pub score_epsilon_reason: Option<ScoreEpsilonReason>,
}

impl Default for ComparatorConfig {
    fn default() -> Self {
        Self {
            score_epsilon_bits: 0.0001_f32.to_bits(),
            score_epsilon_reason: None,
        }
    }
}

impl ComparatorConfig {
    /// Construct a comparator configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GauntletError::InvalidComparatorConfig`] unless
    /// `score_epsilon` is the contract-pinned [`SCORE_EPSILON`].
    pub fn new(score_epsilon: f32) -> Result<Self, GauntletError> {
        if score_epsilon.to_bits() != SCORE_EPSILON.to_bits() {
            return Err(GauntletError::InvalidComparatorConfig {
                reason: format!("score epsilon must be the contract-pinned {SCORE_EPSILON}"),
            });
        }
        Ok(Self {
            score_epsilon_bits: score_epsilon.to_bits(),
            score_epsilon_reason: None,
        })
    }

    /// Permit `ScoreEpsilon` with a closed, artifact-visible reason.
    #[must_use]
    pub const fn with_score_epsilon_reason(mut self, reason: ScoreEpsilonReason) -> Self {
        self.score_epsilon_reason = Some(reason);
        self
    }

    #[must_use]
    pub fn score_epsilon(self) -> f32 {
        f32::from_bits(self.score_epsilon_bits)
    }

    pub(crate) fn validate_contract(self) -> Result<(), GauntletError> {
        if self.score_epsilon_bits == SCORE_EPSILON.to_bits() {
            Ok(())
        } else {
            Err(GauntletError::InvalidComparatorConfig {
                reason: format!("score epsilon must be the contract-pinned {SCORE_EPSILON}"),
            })
        }
    }
}

/// Rank-level outcome before snippet and count checks are folded in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankClass {
    RankExact,
    TieOrder,
    ScoreEpsilon,
    RankMismatch,
}

/// Overall comparison posture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonStatus {
    Exact,
    Classified,
    Failed,
}

/// Stable taxonomy used by differential artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DivergenceClass {
    TieOrder,
    ScoreEpsilon,
    RankMismatch,
    SnippetMismatch,
    SnippetWindow,
    CountMismatch,
    DocumentCountMismatch,
    GlobExpansionLimit,
    QueryCanonicalization,
    OracleBug,
    StatsSemantics,
    UnicodeEdge,
    OversizedQueryToken,
}

impl DivergenceClass {
    const fn is_failure(self) -> bool {
        match self {
            Self::TieOrder
            | Self::ScoreEpsilon
            | Self::SnippetWindow
            | Self::GlobExpansionLimit
            | Self::QueryCanonicalization
            | Self::OracleBug
            | Self::StatsSemantics
            | Self::UnicodeEdge
            | Self::OversizedQueryToken => false,
            Self::RankMismatch
            | Self::SnippetMismatch
            | Self::CountMismatch
            | Self::DocumentCountMismatch => true,
        }
    }
}

/// First-class, pointer-addressable comparison difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Divergence {
    pub class: DivergenceClass,
    /// RFC 6901 JSON pointer into the containing `ArtifactObject`.
    pub pointer: String,
    pub oracle: String,
    pub subject: String,
}

/// Pure comparator output, including the native-order evidence it evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub status: ComparisonStatus,
    pub rank_class: RankClass,
    pub score_epsilon_reason: Option<ScoreEpsilonReason>,
    pub divergences: Vec<Divergence>,
    pub first_divergence: Option<String>,
    pub subject: EngineObservation,
    pub oracle: EngineObservation,
}

/// Compare two observations while preserving each engine's native order.
///
/// Exact-score tie reorders are classified only after proving that the change
/// remains inside one oracle score group. A cutoff substitution additionally
/// requires a complete expanded oracle tie group. Insufficient evidence fails
/// closed as [`RankClass::RankMismatch`].
///
/// # Errors
///
/// Returns an error for duplicate document IDs, non-finite scores, invalid
/// score ordering, or invalid comparator configuration.
pub fn compare_observations(
    subject: EngineObservation,
    oracle: EngineObservation,
    config: ComparatorConfig,
) -> Result<ComparisonReport, GauntletError> {
    config.validate_contract()?;
    let epsilon = config.score_epsilon();

    validate_observation("subject", &subject)?;
    validate_observation("oracle", &oracle)?;

    let (rank_class, rank_divergence) = classify_rank(
        &subject,
        &oracle,
        epsilon,
        config.score_epsilon_reason.is_some(),
    );
    let mut divergences = Vec::new();
    if let Some(divergence) = rank_divergence {
        divergences.push(divergence);
    }
    classify_ast_differences(&subject, &oracle, &mut divergences);
    compare_snippets(&subject, &oracle, &mut divergences);
    if subject.match_count != oracle.match_count {
        divergences.push(Divergence {
            class: DivergenceClass::CountMismatch,
            pointer: "/comparison/subject/match_count".to_owned(),
            oracle: describe_count(oracle.match_count),
            subject: describe_count(subject.match_count),
        });
    }
    if subject.doc_count != oracle.doc_count {
        divergences.push(Divergence {
            class: DivergenceClass::DocumentCountMismatch,
            pointer: "/comparison/subject/doc_count".to_owned(),
            oracle: oracle.doc_count.to_string(),
            subject: subject.doc_count.to_string(),
        });
    }

    let status = if divergences.is_empty() {
        ComparisonStatus::Exact
    } else if divergences.iter().any(|item| item.class.is_failure()) {
        ComparisonStatus::Failed
    } else {
        ComparisonStatus::Classified
    };
    let first_divergence = divergences.first().map(|item| item.pointer.clone());
    let score_epsilon_reason = if rank_class == RankClass::ScoreEpsilon {
        config.score_epsilon_reason
    } else {
        None
    };

    Ok(ComparisonReport {
        status,
        rank_class,
        score_epsilon_reason,
        divergences,
        first_divergence,
        subject,
        oracle,
    })
}

fn validate_observation(label: &str, observation: &EngineObservation) -> Result<(), GauntletError> {
    validate_hit_slice(label, "hits", &observation.hits, true)?;
    validate_hit_slice(
        label,
        "cutoff_tie_group",
        &observation.cutoff_tie_group,
        false,
    )?;
    validate_hit_slice(
        label,
        "offset_tie_group",
        &observation.offset_tie_group,
        false,
    )?;
    validate_cross_evidence_identity(label, &observation.hits, &observation.cutoff_tie_group)?;
    validate_cross_evidence_identity(label, &observation.hits, &observation.offset_tie_group)?;
    validate_cross_evidence_identity(
        label,
        &observation.offset_tie_group,
        &observation.cutoff_tie_group,
    )?;
    let hit_key = observation.hits.first().map(|hit| &hit.native_tie_key);
    let cutoff_key = observation
        .cutoff_tie_group
        .first()
        .map(|hit| &hit.native_tie_key);
    let offset_key = observation
        .offset_tie_group
        .first()
        .map(|hit| &hit.native_tie_key);
    if hit_key
        .zip(cutoff_key)
        .is_some_and(|(left, right)| !same_tie_key_family(left, right))
        || hit_key
            .zip(offset_key)
            .is_some_and(|(left, right)| !same_tie_key_family(left, right))
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!("{label} mixes native tie-key families across evidence"),
        });
    }
    let hit_ids = observation
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    if observation
        .snippets
        .keys()
        .any(|doc_id| !hit_ids.contains(doc_id.as_str()))
    {
        return Err(GauntletError::InvalidObservation {
            reason: format!("{label}.snippets contains a document outside top-k hits"),
        });
    }
    Ok(())
}

fn validate_hit_slice(
    label: &str,
    field: &str,
    hits: &[RankedHit],
    require_score_order: bool,
) -> Result<(), GauntletError> {
    let mut ids = BTreeSet::new();
    let mut native_keys = BTreeSet::new();
    let mut previous_score: Option<f32> = None;
    let mut previous_tie_key: Option<&NativeTieKey> = None;
    let mut tie_key_family: Option<&NativeTieKey> = None;
    for hit in hits {
        if hit.doc_id.is_empty() {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} contains an empty document ID"),
            });
        }
        if !ids.insert(hit.doc_id.as_str()) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} repeats document ID {}", hit.doc_id),
            });
        }
        if !native_keys.insert(&hit.native_tie_key) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} repeats a native tie key"),
            });
        }
        let score = hit.score();
        if !score.is_finite() {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} has a non-finite score for {}", hit.doc_id),
            });
        }
        if tie_key_family.is_some_and(|key| !same_tie_key_family(key, &hit.native_tie_key)) {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} mixes native tie-key families"),
            });
        }
        if require_score_order
            && let Some(previous) = previous_score
            && previous.total_cmp(&score).is_lt()
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!("{label}.{field} is not ordered by descending score"),
            });
        }
        if previous_score.is_some_and(|previous| {
            previous.total_cmp(&score).is_eq()
                && previous_tie_key.is_some_and(|key| key >= &hit.native_tie_key)
        }) {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label}.{field} is not ordered by ascending native key inside an exact-score tie"
                ),
            });
        }
        tie_key_family.get_or_insert(&hit.native_tie_key);
        previous_score = Some(score);
        previous_tie_key = Some(&hit.native_tie_key);
    }
    Ok(())
}

fn validate_cross_evidence_identity(
    label: &str,
    hits: &[RankedHit],
    cutoff_tie_group: &[RankedHit],
) -> Result<(), GauntletError> {
    let hits_by_doc = hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), (hit.score_bits, &hit.native_tie_key)))
        .collect::<BTreeMap<_, _>>();
    let hits_by_native_key = hits
        .iter()
        .map(|hit| (hit.native_tie_key.clone(), hit.doc_id.as_str()))
        .collect::<BTreeMap<_, _>>();

    for cutoff_hit in cutoff_tie_group {
        if let Some((score_bits, native_tie_key)) = hits_by_doc.get(cutoff_hit.doc_id.as_str())
            && (!scores_exact(*score_bits, cutoff_hit.score_bits)
                || *native_tie_key != &cutoff_hit.native_tie_key)
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label} gives document {} inconsistent top-k and cutoff evidence",
                    cutoff_hit.doc_id
                ),
            });
        }
        if let Some(hit_doc_id) = hits_by_native_key.get(&cutoff_hit.native_tie_key)
            && *hit_doc_id != cutoff_hit.doc_id
        {
            return Err(GauntletError::InvalidObservation {
                reason: format!(
                    "{label} assigns one native tie key to multiple document IDs across evidence"
                ),
            });
        }
    }
    Ok(())
}

fn classify_rank(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    epsilon: f32,
    score_epsilon_allowed: bool,
) -> (RankClass, Option<Divergence>) {
    if sequence_is_exact(&subject.hits, &oracle.hits) {
        return (RankClass::RankExact, None);
    }

    let pointer = first_rank_pointer(&subject.hits, &oracle.hits);
    if is_exact_tie_reorder(&subject.hits, &oracle.hits)
        || is_proven_cutoff_tie_substitution(subject, oracle)
    {
        return (
            RankClass::TieOrder,
            Some(rank_divergence(
                DivergenceClass::TieOrder,
                pointer,
                subject,
                oracle,
            )),
        );
    }

    if score_epsilon_allowed && is_score_epsilon_equivalent(&subject.hits, &oracle.hits, epsilon) {
        return (
            RankClass::ScoreEpsilon,
            Some(rank_divergence(
                DivergenceClass::ScoreEpsilon,
                pointer,
                subject,
                oracle,
            )),
        );
    }

    (
        RankClass::RankMismatch,
        Some(rank_divergence(
            DivergenceClass::RankMismatch,
            pointer,
            subject,
            oracle,
        )),
    )
}

fn same_tie_key_family(left: &NativeTieKey, right: &NativeTieKey) -> bool {
    matches!(
        (left, right),
        (
            NativeTieKey::QuillDocId { .. },
            NativeTieKey::QuillDocId { .. }
        ) | (
            NativeTieKey::TantivyDocAddress { .. },
            NativeTieKey::TantivyDocAddress { .. }
        )
    )
}

fn sequence_is_exact(subject: &[RankedHit], oracle: &[RankedHit]) -> bool {
    subject.len() == oracle.len()
        && subject.iter().zip(oracle).all(|(subject_hit, oracle_hit)| {
            subject_hit.doc_id == oracle_hit.doc_id
                && scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        })
}

fn is_exact_tie_reorder(subject: &[RankedHit], oracle: &[RankedHit]) -> bool {
    if subject.len() != oracle.len() || subject.is_empty() {
        return false;
    }
    let Some(oracle_map) = score_map(oracle) else {
        return false;
    };
    let Some(subject_map) = score_map(subject) else {
        return false;
    };
    if oracle_map != subject_map {
        return false;
    }

    let groups = exact_group_map(oracle);
    groups_are_nondecreasing(subject, &groups)
}

fn is_proven_cutoff_tie_substitution(
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> bool {
    is_proven_boundary_tie_substitution(subject, oracle)
}

/// One validated single-score boundary group: the boundary score plus the
/// complete document membership. `None` when the evidence is absent,
/// incomplete, or internally inconsistent (mixed scores or repeated IDs).
fn complete_boundary_group(
    group: &[RankedHit],
    complete: bool,
    boundary_score_bits: u32,
) -> Option<(u32, BTreeSet<&str>)> {
    if !complete || group.is_empty() {
        return None;
    }
    let score_bits = group.first()?.score_bits;
    if !scores_exact(score_bits, boundary_score_bits) {
        return None;
    }
    let mut docs = BTreeSet::new();
    for hit in group {
        if !scores_exact(hit.score_bits, score_bits) || !docs.insert(hit.doc_id.as_str()) {
            return None;
        }
    }
    Some((score_bits, docs))
}

/// Whether both sides of one differing rank are members of the same complete
/// boundary group at their shared exact score.
fn pair_explained_by_boundary_group(
    subject: &RankedHit,
    oracle: &RankedHit,
    group: Option<&(u32, BTreeSet<&str>)>,
) -> bool {
    group.is_some_and(|(score_bits, docs)| {
        scores_exact(subject.score_bits, *score_bits)
            && scores_exact(oracle.score_bits, *score_bits)
            && docs.contains(subject.doc_id.as_str())
            && docs.contains(oracle.doc_id.as_str())
    })
}

/// Generalized tie-substitution proof covering both page boundaries.
///
/// Every position where the subject and oracle pages differ — by document
/// identity or score — must be explained by a complete oracle boundary
/// group: the differing documents on both sides must share one exact score
/// and belong to the complete leading (offset) or trailing (cutoff) tie
/// group. Positions outside the differing span are exact by construction.
/// With only trailing evidence present this reduces to the original cutoff
/// substitution proof; absent or incomplete evidence fails closed.
fn is_proven_boundary_tie_substitution(
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> bool {
    if subject.hits.len() != oracle.hits.len() || subject.hits.is_empty() {
        return false;
    }

    let leading = oracle.hits.first().and_then(|boundary| {
        complete_boundary_group(
            &oracle.offset_tie_group,
            oracle.offset_tie_complete,
            boundary.score_bits,
        )
    });
    let trailing = oracle.hits.last().and_then(|boundary| {
        complete_boundary_group(
            &oracle.cutoff_tie_group,
            oracle.cutoff_tie_complete,
            boundary.score_bits,
        )
    });

    let mut saw_difference = false;
    for (subject_hit, oracle_hit) in subject.hits.iter().zip(&oracle.hits) {
        if subject_hit.doc_id == oracle_hit.doc_id
            && scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        {
            continue;
        }
        saw_difference = true;
        if !scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
            || !(pair_explained_by_boundary_group(subject_hit, oracle_hit, leading.as_ref())
                || pair_explained_by_boundary_group(subject_hit, oracle_hit, trailing.as_ref()))
        {
            return false;
        }
    }
    saw_difference
}

fn is_score_epsilon_equivalent(subject: &[RankedHit], oracle: &[RankedHit], epsilon: f32) -> bool {
    if subject.len() != oracle.len() || subject.is_empty() {
        return false;
    }
    let Some(oracle_scores) = score_map(oracle) else {
        return false;
    };
    let subject_ids = subject
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    if subject_ids.len() != subject.len()
        || subject_ids != oracle_scores.keys().copied().collect::<BTreeSet<_>>()
    {
        return false;
    }
    if subject.iter().any(|hit| {
        oracle_scores
            .get(hit.doc_id.as_str())
            .is_none_or(|oracle_bits| {
                !within_relative_epsilon(hit.score(), f32::from_bits(*oracle_bits), epsilon)
            })
    }) {
        return false;
    }

    let groups = epsilon_group_map(oracle, epsilon);
    groups_are_nondecreasing(subject, &groups)
}

fn score_map(hits: &[RankedHit]) -> Option<BTreeMap<&str, u32>> {
    let map = hits
        .iter()
        .map(|hit| (hit.doc_id.as_str(), hit.score_bits))
        .collect::<BTreeMap<_, _>>();
    (map.len() == hits.len()).then_some(map)
}

fn exact_group_map(hits: &[RankedHit]) -> BTreeMap<&str, usize> {
    group_map(hits, scores_exact)
}

fn epsilon_group_map(hits: &[RankedHit], epsilon: f32) -> BTreeMap<&str, usize> {
    group_map(hits, |left, right| {
        within_relative_epsilon(f32::from_bits(left), f32::from_bits(right), epsilon)
    })
}

fn group_map(hits: &[RankedHit], adjacent: impl Fn(u32, u32) -> bool) -> BTreeMap<&str, usize> {
    let mut groups = BTreeMap::new();
    let mut group = 0usize;
    let mut previous = None;
    for hit in hits {
        if previous.is_some_and(|score| !adjacent(score, hit.score_bits)) {
            group = group.saturating_add(1);
        }
        groups.insert(hit.doc_id.as_str(), group);
        previous = Some(hit.score_bits);
    }
    groups
}

fn groups_are_nondecreasing(hits: &[RankedHit], groups: &BTreeMap<&str, usize>) -> bool {
    let mut previous = None;
    for hit in hits {
        let Some(group) = groups.get(hit.doc_id.as_str()).copied() else {
            return false;
        };
        if previous.is_some_and(|prior| group < prior) {
            return false;
        }
        previous = Some(group);
    }
    true
}

fn scores_exact(left: u32, right: u32) -> bool {
    f32::from_bits(left)
        .total_cmp(&f32::from_bits(right))
        .is_eq()
}

fn within_relative_epsilon(left: f32, right: f32, epsilon: f32) -> bool {
    let left = f64::from(left);
    let right = f64::from(right);
    let denominator = left.abs().max(right.abs()).max(1e-12);
    (left - right).abs() / denominator <= f64::from(epsilon)
}

fn first_rank_pointer(subject: &[RankedHit], oracle: &[RankedHit]) -> String {
    let index = subject
        .iter()
        .zip(oracle)
        .position(|(subject_hit, oracle_hit)| {
            subject_hit.doc_id != oracle_hit.doc_id
                || !scores_exact(subject_hit.score_bits, oracle_hit.score_bits)
        });
    index.map_or_else(
        || "/comparison/subject/hits".to_owned(),
        |index| format!("/comparison/subject/hits/{index}"),
    )
}

fn rank_divergence(
    class: DivergenceClass,
    pointer: String,
    subject: &EngineObservation,
    oracle: &EngineObservation,
) -> Divergence {
    let index = pointer
        .rsplit('/')
        .next()
        .and_then(|value| value.parse::<usize>().ok());
    Divergence {
        class,
        pointer,
        oracle: index
            .and_then(|index| oracle.hits.get(index))
            .map_or_else(|| oracle.hits.len().to_string(), describe_hit),
        subject: index
            .and_then(|index| subject.hits.get(index))
            .map_or_else(|| subject.hits.len().to_string(), describe_hit),
    }
}

fn describe_hit(hit: &RankedHit) -> String {
    format!("{}@{:08x}", hit.doc_id, hit.score_bits)
}

fn describe_count(count: CountState) -> String {
    match count {
        CountState::NotRequested => "not_requested".to_owned(),
        CountState::Value(value) => value.to_string(),
    }
}

/// Fold recorded AST/diagnostic lowering differences into the divergence
/// list. Every recorded kind maps to a reviewed register class; kinds the
/// comparator does not know cannot reach here because the enum is closed,
/// and adding a kind without its register class fails review.
fn classify_ast_differences(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    divergences: &mut Vec<Divergence>,
) {
    for (index, difference) in subject.ast_differences.iter().enumerate() {
        let class = match difference.kind {
            AstLoweringKind::OversizedQueryToken => DivergenceClass::OversizedQueryToken,
            AstLoweringKind::QueryCanonicalization => DivergenceClass::QueryCanonicalization,
            AstLoweringKind::OracleBug => DivergenceClass::OracleBug,
            AstLoweringKind::GlobExpansionLimit => DivergenceClass::GlobExpansionLimit,
            AstLoweringKind::StatsSemantics => DivergenceClass::StatsSemantics,
            AstLoweringKind::UnicodeEdge => DivergenceClass::UnicodeEdge,
        };
        divergences.push(Divergence {
            class,
            pointer: format!("/comparison/subject/ast_differences/{index}"),
            oracle: difference.oracle.clone(),
            subject: difference.subject.clone(),
        });
    }
    for (index, difference) in oracle.ast_differences.iter().enumerate() {
        divergences.push(Divergence {
            class: DivergenceClass::RankMismatch,
            pointer: format!("/comparison/oracle/ast_differences/{index}"),
            oracle: difference.oracle.clone(),
            subject: difference.subject.clone(),
        });
    }
}

fn compare_snippets(
    subject: &EngineObservation,
    oracle: &EngineObservation,
    divergences: &mut Vec<Divergence>,
) {
    let subject_ids = subject
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    let oracle_ids = oracle
        .hits
        .iter()
        .map(|hit| hit.doc_id.as_str())
        .collect::<BTreeSet<_>>();
    let ids = subject_ids.intersection(&oracle_ids);
    for doc_id in ids {
        let subject_snippet = subject.snippets.get(*doc_id);
        let oracle_snippet = oracle.snippets.get(*doc_id);
        if subject_snippet != oracle_snippet {
            divergences.push(Divergence {
                class: DivergenceClass::SnippetMismatch,
                pointer: format!(
                    "/comparison/{}/snippets/{}",
                    if subject_snippet.is_some() {
                        "subject"
                    } else {
                        "oracle"
                    },
                    escape_json_pointer_token(doc_id)
                ),
                oracle: oracle_snippet
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_owned()),
                subject: subject_snippet
                    .cloned()
                    .unwrap_or_else(|| "<missing>".to_owned()),
            });
            break;
        }
    }
}

fn escape_json_pointer_token(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn quill_hit(doc_id: &str, score: f32, native_doc_id: u32) -> RankedHit {
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key: NativeTieKey::QuillDocId {
                doc_id: native_doc_id,
            },
        }
    }

    fn tantivy_hit(doc_id: &str, score: f32, doc_id_in_segment: u32) -> RankedHit {
        RankedHit {
            doc_id: doc_id.to_owned(),
            score_bits: score.to_bits(),
            native_tie_key: NativeTieKey::TantivyDocAddress {
                segment_ord: 3,
                doc_id: doc_id_in_segment,
            },
        }
    }

    fn observation(hits: Vec<RankedHit>) -> EngineObservation {
        EngineObservation {
            match_count: CountState::Value(u64::try_from(hits.len()).unwrap_or(u64::MAX)),
            doc_count: 9,
            hits,
            cutoff_tie_group: Vec::new(),
            cutoff_tie_complete: true,
            offset_tie_group: Vec::new(),
            offset_tie_complete: false,
            snippets: BTreeMap::new(),
            ast_differences: Vec::new(),
        }
    }

    fn lexical_context(engine: &str) -> LexicalObservationContext {
        LexicalObservationContext::new(
            LexicalBoundary::FullSearch,
            LexicalBackendIdentity {
                engine: engine.to_owned(),
                revision: format!("{engine}-test-revision"),
                index_identity: "in-memory".to_owned(),
            },
            "a".repeat(64),
            "b".repeat(64),
            " Rust  search ",
            42,
            10,
            LexicalExposureContract::CORE_LEXICAL_SEARCH,
        )
        .expect("valid lexical observation context")
    }

    fn rich_lexical_context(engine: &str) -> LexicalObservationContext {
        LexicalObservationContext::new(
            LexicalBoundary::FullSearch,
            LexicalBackendIdentity {
                engine: engine.to_owned(),
                revision: format!("{engine}-test-revision"),
                index_identity: "in-memory".to_owned(),
            },
            "a".repeat(64),
            "b".repeat(64),
            " Rust  search ",
            42,
            10,
            LexicalExposureContract {
                metadata: LexicalFieldExposure::Exposed,
                explanation: LexicalFieldExposure::Exposed,
                total_count: LexicalCountExposure::ExactRequested,
                snippet: LexicalFieldExposure::Exposed,
                highlight_spans: LexicalFieldExposure::Exposed,
            },
        )
        .expect("valid rich lexical observation context")
    }

    fn counted_lexical_context(engine: &str) -> LexicalObservationContext {
        let mut context = lexical_context(engine);
        context.exposure.total_count = LexicalCountExposure::ExactRequested;
        context
    }

    fn lexical_result(doc_id: &str, score: f32) -> ScoredResult {
        use frankensearch_core::{ExplanationPhase, HitExplanation};

        ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(score),
            rerank_score: None,
            explanation: Some(Box::new(HitExplanation {
                final_score: f64::from(score),
                components: Vec::new(),
                phase: ExplanationPhase::Refined,
                rank_movement: None,
            })),
            metadata: Some(Arc::new(serde_json::json!({
                "language": "rust",
                "nested": {"stable": true}
            }))),
        }
    }

    fn lexical_supplement(doc_id: &str) -> LexicalObservationSupplement {
        LexicalObservationSupplement {
            total_count: LexicalCountState::Value(1),
            hits: BTreeMap::from([(
                doc_id.to_owned(),
                LexicalHitSupplement {
                    snippet: SensitiveValueObservation::from_text("safe snippet"),
                    highlight_spans: LexicalObserved::Value(vec![LexicalHighlightSpan {
                        start: 0,
                        end: 4,
                    }]),
                },
            )]),
        }
    }

    fn complete_lexical_observation(engine: &str) -> LexicalObservation {
        observe_lexical_outcome(
            rich_lexical_context(engine),
            Ok(vec![lexical_result("doc-1", 3.5)]),
            &lexical_supplement("doc-1"),
        )
        .expect("complete lexical observation")
    }

    fn lexical_page(
        engine: &str,
        boundary: LexicalBoundary,
        results: Vec<ScoredResult>,
    ) -> LexicalObservation {
        let mut context = lexical_context(engine);
        context.boundary = boundary;
        observe_lexical_outcome(
            context,
            Ok(results),
            &LexicalObservationSupplement::default(),
        )
        .expect("valid lexical page fixture")
    }

    fn hydration_transition(
        selection: LexicalHydrationSelection,
        input: LexicalObservation,
        post_state: LexicalObservation,
    ) -> LexicalHydrationTransition {
        LexicalHydrationTransition {
            selection,
            execution: LexicalHydrationExecution::Attempted {
                input: Box::new(input),
                post_state: Box::new(post_state),
                result: LexicalHydrationResult::Success,
            },
        }
    }

    fn lexical_contract_bundle(
        engine: &str,
        fusion_metadata_deferred: bool,
    ) -> LexicalContractBundle {
        let role = if engine == "quill" {
            LexicalEngineRole::Subject
        } else {
            LexicalEngineRole::Oracle
        };
        let full_search = lexical_page(
            engine,
            LexicalBoundary::FullSearch,
            vec![lexical_result("doc-1", 3.5), lexical_result("doc-2", 2.5)],
        );
        let mut fusion_candidates = lexical_page(
            engine,
            LexicalBoundary::FusionCandidates,
            vec![lexical_result("doc-1", 3.5), lexical_result("doc-2", 2.5)],
        );
        if fusion_metadata_deferred {
            let (hits, _, _, _) = lexical_success_mut(&mut fusion_candidates)
                .expect("candidate fixture must be successful");
            for hit in hits {
                hit.metadata = SensitiveValueObservation::Absent;
            }
        }

        let candidate_hits = lexical_success_hits(&fusion_candidates.outcome)
            .expect("candidate fixture must be successful");
        let full_hits =
            lexical_success_hits(&full_search.outcome).expect("full fixture must be successful");
        let hydrate = |hits: Vec<LexicalHitObservation>| {
            hits.into_iter()
                .map(|mut hit| {
                    if hit.raw_lexical_score_bits.is_some() {
                        hit.metadata = full_hits
                            .iter()
                            .find(|full| full.doc_id == hit.doc_id)
                            .expect("lexical fixture origin exists in full search")
                            .metadata
                            .clone();
                    }
                    hit
                })
                .collect::<Vec<_>>()
        };
        let page = |boundary, hits| lexical_observation_from_hits(engine, boundary, hits);

        let all_input_hits = candidate_hits
            .iter()
            .enumerate()
            .map(|(rank, candidate)| {
                expected_lexical_winner_hit(
                    candidate,
                    rank,
                    rank,
                    LexicalWinnerProjection::LexicalOnly,
                )
                .expect("all-lexical fixture projection")
            })
            .collect::<Vec<_>>();
        let all_input = page(
            LexicalBoundary::FusionHydrationAllLexicalInput,
            all_input_hits.clone(),
        );
        let all_post_state = page(
            LexicalBoundary::FusionHydrationAllLexicalPostState,
            hydrate(all_input_hits),
        );

        let strict_candidate_rank = 1;
        let strict_input_hits = vec![
            expected_lexical_winner_hit(
                &candidate_hits[strict_candidate_rank],
                strict_candidate_rank,
                0,
                LexicalWinnerProjection::HybridFast,
            )
            .expect("strict-hybrid fixture projection"),
        ];
        let strict_input = page(
            LexicalBoundary::FusionHydrationHybridSubsetInput,
            strict_input_hits.clone(),
        );
        let strict_post_state = page(
            LexicalBoundary::FusionHydrationHybridSubsetPostState,
            hydrate(strict_input_hits),
        );

        let semantic_context = {
            let mut context = lexical_context(engine);
            context.boundary = LexicalBoundary::FusionHydrationSemanticOnlyInput;
            context
        };
        let semantic_input_hits = vec![
            expected_non_lexical_control_hit(
                &semantic_context,
                0,
                LexicalNonLexicalControlKind::SemanticFast,
                0,
            )
            .expect("semantic-only fixture projection"),
        ];
        let semantic_input = page(
            LexicalBoundary::FusionHydrationSemanticOnlyInput,
            semantic_input_hits.clone(),
        );
        let semantic_post_state = page(
            LexicalBoundary::FusionHydrationSemanticOnlyPostState,
            semantic_input_hits,
        );

        let mixed_origins = vec![
            LexicalWinnerOrigin::NonLexicalControl {
                control_id: 1,
                kind: LexicalNonLexicalControlKind::SemanticFast,
            },
            LexicalWinnerOrigin::Lexical {
                candidate_rank: 1,
                projection: LexicalWinnerProjection::HybridFast,
            },
            LexicalWinnerOrigin::Lexical {
                candidate_rank: 0,
                projection: LexicalWinnerProjection::LexicalOnly,
            },
            LexicalWinnerOrigin::NonLexicalControl {
                control_id: 2,
                kind: LexicalNonLexicalControlKind::GraphOnlyHybrid,
            },
        ];
        let mixed_context = {
            let mut context = lexical_context(engine);
            context.boundary = LexicalBoundary::FusionHydrationMixedInput;
            context
        };
        let mixed_input_hits = mixed_origins
            .iter()
            .copied()
            .enumerate()
            .map(|(position, origin)| match origin {
                LexicalWinnerOrigin::Lexical {
                    candidate_rank,
                    projection,
                } => expected_lexical_winner_hit(
                    &candidate_hits
                        [usize::try_from(candidate_rank).expect("fixture candidate rank")],
                    usize::try_from(candidate_rank).expect("fixture candidate rank"),
                    position,
                    projection,
                )
                .expect("mixed lexical fixture projection"),
                LexicalWinnerOrigin::NonLexicalControl { control_id, kind } => {
                    expected_non_lexical_control_hit(&mixed_context, control_id, kind, position)
                        .expect("mixed non-lexical fixture projection")
                }
            })
            .collect::<Vec<_>>();
        let mixed_input = page(
            LexicalBoundary::FusionHydrationMixedInput,
            mixed_input_hits.clone(),
        );
        let mixed_post_state = page(
            LexicalBoundary::FusionHydrationMixedPostState,
            hydrate(mixed_input_hits),
        );

        LexicalContractBundle {
            schema_version: LEXICAL_CONTRACT_BUNDLE_SCHEMA_VERSION.to_owned(),
            engine_role: role,
            snapshot_sha256: "c".repeat(64),
            fusion_metadata_deferred,
            full_search,
            fusion_candidates,
            all_lexical_winners_hydration: hydration_transition(
                LexicalHydrationSelection::AllLexicalWinners,
                all_input,
                all_post_state,
            ),
            strict_hybrid_winners_hydration: hydration_transition(
                LexicalHydrationSelection::StrictHybridWinnerSubset {
                    candidate_ranks: vec![1],
                },
                strict_input,
                strict_post_state,
            ),
            semantic_only_hydration: hydration_transition(
                LexicalHydrationSelection::SemanticOnlyControl { control_id: 0 },
                semantic_input,
                semantic_post_state,
            ),
            mixed_winners_hydration: hydration_transition(
                LexicalHydrationSelection::MixedFinalWinners {
                    origins: mixed_origins,
                },
                mixed_input,
                mixed_post_state,
            ),
        }
    }

    fn lexical_observation_from_hits(
        engine: &str,
        boundary: LexicalBoundary,
        hits: Vec<LexicalHitObservation>,
    ) -> LexicalObservation {
        let mut context = lexical_context(engine);
        context.boundary = boundary;
        let returned_count = u64::try_from(hits.len()).expect("fixture hit count");
        let empty_shape = if hits.is_empty() {
            LexicalEmptyShape::Empty
        } else {
            LexicalEmptyShape::NonEmpty
        };
        let observation = LexicalObservation {
            context,
            outcome: LexicalObservationOutcome::Success {
                hits,
                returned_count,
                empty_shape,
                total_count: LexicalCountState::NotExposed,
            },
        };
        validate_lexical_observation(&observation).expect("valid projected lexical fixture");
        observation
    }

    fn lexical_success_mut(
        observation: &mut LexicalObservation,
    ) -> Option<(
        &mut Vec<LexicalHitObservation>,
        &mut u64,
        &mut LexicalEmptyShape,
        &mut LexicalCountState,
    )> {
        match &mut observation.outcome {
            LexicalObservationOutcome::Success {
                hits,
                returned_count,
                empty_shape,
                total_count,
            } => Some((hits, returned_count, empty_shape, total_count)),
            LexicalObservationOutcome::Error(_) => None,
        }
    }

    fn lexical_hit_mut(observation: &mut LexicalObservation) -> &mut LexicalHitObservation {
        lexical_hit_at_mut(observation, 0)
    }

    fn lexical_hit_at_mut(
        observation: &mut LexicalObservation,
        position: usize,
    ) -> &mut LexicalHitObservation {
        lexical_success_mut(observation)
            .expect("test fixture must be a successful lexical observation")
            .0
            .get_mut(position)
            .expect("test fixture must contain the requested lexical hit")
    }

    fn attempted_observations_mut(
        transition: &mut LexicalHydrationTransition,
    ) -> (&mut LexicalObservation, &mut LexicalObservation) {
        let LexicalHydrationExecution::Attempted {
            input, post_state, ..
        } = &mut transition.execution
        else {
            panic!("test fixture must contain an attempted hydration transition");
        };
        (input, post_state)
    }

    fn lexical_error_mut(
        observation: &mut LexicalObservation,
    ) -> Option<&mut LexicalErrorObservation> {
        match &mut observation.outcome {
            LexicalObservationOutcome::Success { .. } => None,
            LexicalObservationOutcome::Error(error) => Some(error),
        }
    }

    fn lexical_error(observation: &LexicalObservation) -> Option<&LexicalErrorObservation> {
        match &observation.outcome {
            LexicalObservationOutcome::Success { .. } => None,
            LexicalObservationOutcome::Error(error) => Some(error),
        }
    }

    fn assert_single_lexical_mismatch(
        mutate: impl FnOnce(&mut LexicalObservation),
        expected_class: LexicalMismatchClass,
        expected_path: &str,
    ) {
        let oracle = complete_lexical_observation("tantivy");
        let mut subject = complete_lexical_observation("quill");
        mutate(&mut subject);
        let report = compare_lexical_observations(subject, oracle)
            .expect("mutated observation remains structurally valid");
        assert_eq!(report.status, LexicalComparisonStatus::Mismatch);
        assert_eq!(
            report
                .mismatches
                .iter()
                .map(|mismatch| (mismatch.class, mismatch.path.as_str()))
                .collect::<Vec<_>>(),
            vec![(expected_class, expected_path)]
        );
        assert_eq!(report.first_mismatch, report.mismatches.first().cloned());
    }

    fn assert_invalid_lexical_mutation(mutate: impl FnOnce(&mut LexicalObservation)) {
        let oracle = complete_lexical_observation("tantivy");
        let mut subject = complete_lexical_observation("quill");
        mutate(&mut subject);
        assert!(matches!(
            compare_lexical_observations(subject, oracle),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SpySearchOutcome {
        Success,
        Cancelled,
        QueryError,
    }

    struct LexicalContractSpy {
        deferred: bool,
        full_outcome: SpySearchOutcome,
        candidate_outcome: SpySearchOutcome,
        fail_hydration_call: Option<usize>,
        hydration_calls: Mutex<usize>,
        calls: Mutex<Vec<String>>,
    }

    impl LexicalContractSpy {
        fn new(deferred: bool) -> Self {
            Self {
                deferred,
                full_outcome: SpySearchOutcome::Success,
                candidate_outcome: SpySearchOutcome::Success,
                fail_hydration_call: None,
                hydration_calls: Mutex::new(0),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn record(&self, call: impl Into<String>) {
            self.calls
                .lock()
                .expect("lexical spy call log")
                .push(call.into());
        }

        fn results(include_metadata: bool) -> Vec<ScoredResult> {
            [("doc-1", 3.5), ("doc-2", 2.5)]
                .into_iter()
                .map(|(doc_id, score)| {
                    let mut result = lexical_result(doc_id, score);
                    if !include_metadata {
                        result.metadata = None;
                    }
                    result
                })
                .collect()
        }
    }

    fn assert_spy_search_outcome(observed: &LexicalObservationOutcome, expected: SpySearchOutcome) {
        match expected {
            SpySearchOutcome::Success => {
                assert!(matches!(
                    observed,
                    LexicalObservationOutcome::Success { .. }
                ));
            }
            SpySearchOutcome::Cancelled => {
                assert!(matches!(
                    observed,
                    LexicalObservationOutcome::Error(LexicalErrorObservation {
                        class: LexicalErrorClass::Cancellation,
                        code,
                        ..
                    }) if code == "cancelled"
                ));
            }
            SpySearchOutcome::QueryError => {
                assert!(matches!(
                    observed,
                    LexicalObservationOutcome::Error(LexicalErrorObservation {
                        class: LexicalErrorClass::Query,
                        code,
                        ..
                    }) if code == "query_parse_error"
                ));
            }
        }
    }

    impl LexicalSearch for LexicalContractSpy {
        fn search<'a>(
            &'a self,
            _cx: &'a Cx,
            query: &'a str,
            limit: usize,
        ) -> frankensearch_core::SearchFuture<'a, Vec<ScoredResult>> {
            self.record(format!("search:{query}:{limit}"));
            let mut results = Self::results(true);
            results.truncate(limit);
            let result = match self.full_outcome {
                SpySearchOutcome::Success => Ok(results),
                SpySearchOutcome::Cancelled => Err(SearchError::Cancelled {
                    phase: "spy full search".to_owned(),
                    reason: "parent scope cancelled".to_owned(),
                }),
                SpySearchOutcome::QueryError => Err(SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: "spy full failure".to_owned(),
                }),
            };
            Box::pin(async move { result })
        }

        fn search_fusion_candidates<'a>(
            &'a self,
            _cx: &'a Cx,
            query: &'a str,
            limit: usize,
        ) -> frankensearch_core::SearchFuture<'a, Vec<ScoredResult>> {
            self.record(format!("candidates:{query}:{limit}"));
            let result = match self.candidate_outcome {
                SpySearchOutcome::Success => {
                    let mut results = Self::results(!self.deferred);
                    results.truncate(limit);
                    Ok(results)
                }
                SpySearchOutcome::Cancelled => Err(SearchError::Cancelled {
                    phase: "spy candidate search".to_owned(),
                    reason: "parent scope cancelled".to_owned(),
                }),
                SpySearchOutcome::QueryError => Err(SearchError::QueryParseError {
                    query: query.to_owned(),
                    detail: "spy candidate failure".to_owned(),
                }),
            };
            Box::pin(async move { result })
        }

        fn fusion_metadata_is_deferred(&self) -> bool {
            self.record("capability");
            self.deferred
        }

        fn hydrate_fusion_metadata<'a>(
            &'a self,
            _cx: &'a Cx,
            results: &'a mut [ScoredResult],
        ) -> frankensearch_core::SearchFuture<'a, ()> {
            Box::pin(async move {
                let hydration_call = {
                    let mut calls = self
                        .hydration_calls
                        .lock()
                        .expect("lexical spy hydration call counter");
                    *calls += 1;
                    *calls
                };
                self.record(format!("hydrate:{}", results.len()));
                if self.fail_hydration_call == Some(hydration_call) {
                    if let Some(first) = results.first_mut() {
                        first.metadata =
                            lexical_result(first.doc_id.as_str(), first.score).metadata;
                    }
                    return Err(SearchError::QueryParseError {
                        query: "redacted by evidence adapter".to_owned(),
                        detail: "spy hydration failure".to_owned(),
                    });
                }
                if self.deferred {
                    for result in results {
                        if result.lexical_score.is_some() {
                            result.metadata =
                                lexical_result(result.doc_id.as_str(), result.score).metadata;
                        }
                    }
                }
                Ok(())
            })
        }

        fn index_document<'a>(
            &'a self,
            _cx: &'a Cx,
            _doc: &'a frankensearch_core::IndexableDocument,
        ) -> frankensearch_core::SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn commit<'a>(&'a self, _cx: &'a Cx) -> frankensearch_core::SearchFuture<'a, ()> {
            Box::pin(async { Ok(()) })
        }

        fn doc_count(&self) -> usize {
            2
        }
    }

    fn live_build_context(
        engine: &str,
        role: LexicalEngineRole,
    ) -> LexicalContractBuildContext<'static> {
        live_build_context_with_limit(engine, role, 10)
    }

    fn live_build_context_with_limit(
        engine: &str,
        role: LexicalEngineRole,
        limit: usize,
    ) -> LexicalContractBuildContext<'static> {
        LexicalContractBuildContext::new(
            role,
            LexicalBackendIdentity {
                engine: engine.to_owned(),
                revision: format!("{engine}-test-revision"),
                index_identity: "immutable-test-snapshot".to_owned(),
            },
            "c".repeat(64),
            "a".repeat(64),
            "b".repeat(64),
            "rust search",
            0x51_7e_a2,
            limit,
        )
        .expect("valid live lexical build context")
    }

    #[test]
    fn live_lexical_builder_exercises_each_public_method_and_scopes_deferral() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let subject_engine = LexicalContractSpy::new(true);
            let oracle_engine = LexicalContractSpy::new(false);
            let subject = observe_live_lexical_contract(
                &cx,
                &subject_engine,
                live_build_context("quill", LexicalEngineRole::Subject),
            )
            .await
            .expect("observe deferred subject");
            let oracle = observe_live_lexical_contract(
                &cx,
                &oracle_engine,
                live_build_context("tantivy", LexicalEngineRole::Oracle),
            )
            .await
            .expect("observe eager oracle");

            assert_eq!(
                subject_engine
                    .calls
                    .lock()
                    .expect("subject call log")
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec![
                    "search:rust search:10",
                    "candidates:rust search:10",
                    "capability",
                    "hydrate:1",
                    "hydrate:2",
                    "hydrate:1",
                    "hydrate:4",
                ]
            );
            assert_eq!(
                oracle_engine
                    .calls
                    .lock()
                    .expect("oracle call log")
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec![
                    "search:rust search:10",
                    "candidates:rust search:10",
                    "capability",
                    "hydrate:1",
                    "hydrate:2",
                    "hydrate:1",
                    "hydrate:4",
                ]
            );

            let comparison =
                compare_lexical_contracts(subject, oracle).expect("compare live bundles");
            assert_eq!(comparison.status, LexicalComparisonStatus::Equivalent);
            assert!(!comparison.waived_differences.is_empty());
            assert!(comparison.waived_differences.iter().all(|waiver| {
                waiver.deferred_side == LexicalDeferredSide::Subject
                    && waiver.law == LexicalEquivalenceLaw::DeferredMetadataHydration
            }));
            comparison
                .validate_replay()
                .expect("replay live comparison");
        });
    }

    #[test]
    fn live_lexical_builder_retains_candidate_failure_without_erasing_full_search() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut engine = LexicalContractSpy::new(true);
            engine.candidate_outcome = SpySearchOutcome::QueryError;
            let bundle = observe_live_lexical_contract(
                &cx,
                &engine,
                live_build_context("quill", LexicalEngineRole::Subject),
            )
            .await
            .expect("candidate failure is valid typed evidence");

            assert!(matches!(
                bundle.full_search.outcome,
                LexicalObservationOutcome::Success { .. }
            ));
            assert!(matches!(
                bundle.fusion_candidates.outcome,
                LexicalObservationOutcome::Error(LexicalErrorObservation {
                    code,
                    ..
                }) if code == "query_parse_error"
            ));
            for transition in [
                &bundle.all_lexical_winners_hydration,
                &bundle.strict_hybrid_winners_hydration,
                &bundle.mixed_winners_hydration,
            ] {
                assert!(matches!(
                    &transition.execution,
                    LexicalHydrationExecution::NotRun {
                        reason: LexicalHydrationNotRunReason::CandidateSearchFailed
                    }
                ));
            }
            assert_eq!(
                engine
                    .calls
                    .lock()
                    .expect("candidate-error call log")
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec![
                    "search:rust search:10",
                    "candidates:rust search:10",
                    "capability",
                    "hydrate:1",
                ]
            );
            assert!(matches!(
                bundle.semantic_only_hydration.execution,
                LexicalHydrationExecution::Attempted {
                    result: LexicalHydrationResult::Success,
                    ..
                }
            ));
        });
    }

    #[test]
    fn live_lexical_builder_retains_full_cancellation_without_erasing_candidates() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut engine = LexicalContractSpy::new(false);
            engine.full_outcome = SpySearchOutcome::Cancelled;
            let bundle = observe_live_lexical_contract(
                &cx,
                &engine,
                live_build_context("quill", LexicalEngineRole::Subject),
            )
            .await
            .expect("full-search cancellation is valid typed evidence");

            assert!(matches!(
                bundle.full_search.outcome,
                LexicalObservationOutcome::Error(LexicalErrorObservation {
                    class: LexicalErrorClass::Cancellation,
                    code,
                    ..
                }) if code == "cancelled"
            ));
            assert!(matches!(
                bundle.fusion_candidates.outcome,
                LexicalObservationOutcome::Success { .. }
            ));
            for transition in [
                &bundle.all_lexical_winners_hydration,
                &bundle.strict_hybrid_winners_hydration,
                &bundle.semantic_only_hydration,
                &bundle.mixed_winners_hydration,
            ] {
                assert!(matches!(
                    &transition.execution,
                    LexicalHydrationExecution::Attempted { .. }
                ));
            }
        });
    }

    #[test]
    fn live_lexical_builder_persists_independent_lane_error_and_cancellation_matrix() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let outcomes = [
                SpySearchOutcome::Success,
                SpySearchOutcome::Cancelled,
                SpySearchOutcome::QueryError,
            ];
            for full_outcome in outcomes {
                for candidate_outcome in outcomes {
                    let mut subject_engine = LexicalContractSpy::new(true);
                    subject_engine.full_outcome = full_outcome;
                    subject_engine.candidate_outcome = candidate_outcome;
                    let mut oracle_engine = LexicalContractSpy::new(true);
                    oracle_engine.full_outcome = full_outcome;
                    oracle_engine.candidate_outcome = candidate_outcome;

                    let subject = observe_live_lexical_contract(
                        &cx,
                        &subject_engine,
                        live_build_context("quill", LexicalEngineRole::Subject),
                    )
                    .await
                    .expect("subject lane outcome pair is admissible typed evidence");
                    let oracle = observe_live_lexical_contract(
                        &cx,
                        &oracle_engine,
                        live_build_context("tantivy", LexicalEngineRole::Oracle),
                    )
                    .await
                    .expect("oracle lane outcome pair is admissible typed evidence");
                    for bundle in [&subject, &oracle] {
                        assert_spy_search_outcome(&bundle.full_search.outcome, full_outcome);
                        assert_spy_search_outcome(
                            &bundle.fusion_candidates.outcome,
                            candidate_outcome,
                        );
                        let persisted = serde_json::to_vec(bundle).expect("persist lexical bundle");
                        let replayed: LexicalContractBundle =
                            serde_json::from_slice(&persisted).expect("replay lexical bundle");
                        assert_eq!(&replayed, bundle);
                    }

                    let comparison = compare_lexical_contracts(subject, oracle)
                        .expect("equivalent independent lane outcomes compare");
                    assert_eq!(comparison.status, LexicalComparisonStatus::Equivalent);
                    comparison
                        .validate_replay()
                        .expect("independent lane outcome comparison replays");
                    let persisted =
                        serde_json::to_vec(&comparison).expect("persist lane outcome comparison");
                    let replayed: LexicalContractComparison =
                        serde_json::from_slice(&persisted).expect("replay lane outcome comparison");
                    assert_eq!(replayed, comparison);
                    replayed
                        .validate_replay()
                        .expect("persisted lane outcome comparison remains valid");
                }
            }
        });
    }

    #[test]
    fn live_lexical_builder_retains_partial_post_state_on_hydration_error() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let mut engine = LexicalContractSpy::new(true);
            engine.fail_hydration_call = Some(2);
            let bundle = observe_live_lexical_contract(
                &cx,
                &engine,
                live_build_context("quill", LexicalEngineRole::Subject),
            )
            .await
            .expect("partial hydration mutation is retained as valid evidence");

            let LexicalHydrationExecution::Attempted {
                input,
                post_state,
                result,
            } = &bundle.all_lexical_winners_hydration.execution
            else {
                panic!("all-lexical-winner hydration must have been attempted");
            };
            assert!(matches!(
                result,
                LexicalHydrationResult::Error(LexicalErrorObservation {
                    code,
                    ..
                }) if code == "query_parse_error"
            ));
            assert_eq!(
                lexical_success_hits(&input.outcome).expect("successful input")[0].metadata,
                SensitiveValueObservation::Absent
            );
            assert_ne!(
                lexical_success_hits(&post_state.outcome).expect("retained post-state")[0].metadata,
                SensitiveValueObservation::Absent,
                "the partial metadata mutation must not disappear behind the error"
            );
            assert_eq!(
                hydration_probe_coverage(&bundle.all_lexical_winners_hydration.execution),
                LexicalProbeCoverage::ExercisedError
            );
        });
    }

    #[test]
    fn live_lexical_builder_records_canonical_capacity_boundaries() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let zero = observe_live_lexical_contract(
                &cx,
                &LexicalContractSpy::new(false),
                live_build_context_with_limit("quill", LexicalEngineRole::Subject, 0),
            )
            .await
            .expect("zero-capacity contract evidence");
            assert!(matches!(
                zero.semantic_only_hydration.execution,
                LexicalHydrationExecution::NotRun {
                    reason: LexicalHydrationNotRunReason::InsufficientResultCapacity {
                        limit: 0,
                        required: 1
                    }
                }
            ));
            assert!(matches!(
                zero.strict_hybrid_winners_hydration.execution,
                LexicalHydrationExecution::NotRun {
                    reason: LexicalHydrationNotRunReason::InsufficientCandidates {
                        available: 0,
                        required: 2
                    }
                }
            ));
            assert!(matches!(
                zero.mixed_winners_hydration.execution,
                LexicalHydrationExecution::NotRun {
                    reason: LexicalHydrationNotRunReason::InsufficientResultCapacity {
                        limit: 0,
                        required: 2
                    }
                }
            ));

            let one = observe_live_lexical_contract(
                &cx,
                &LexicalContractSpy::new(false),
                live_build_context_with_limit("quill", LexicalEngineRole::Subject, 1),
            )
            .await
            .expect("one-result-capacity contract evidence");
            assert!(matches!(
                one.semantic_only_hydration.execution,
                LexicalHydrationExecution::Attempted {
                    result: LexicalHydrationResult::Success,
                    ..
                }
            ));
            assert!(matches!(
                one.strict_hybrid_winners_hydration.execution,
                LexicalHydrationExecution::NotRun {
                    reason: LexicalHydrationNotRunReason::InsufficientCandidates {
                        available: 1,
                        required: 2
                    }
                }
            ));
            let mut forged_required_count = one.clone();
            let LexicalHydrationExecution::NotRun {
                reason: LexicalHydrationNotRunReason::InsufficientCandidates { required, .. },
            } = &mut forged_required_count
                .strict_hybrid_winners_hydration
                .execution
            else {
                panic!("one-result strict subset must be a canonical not-run probe");
            };
            *required = 3;
            assert!(
                validate_lexical_contract_bundle(&forged_required_count).is_err(),
                "replay must reject a not-run threshold the live builder can never emit"
            );
            assert!(matches!(
                one.mixed_winners_hydration.execution,
                LexicalHydrationExecution::NotRun {
                    reason: LexicalHydrationNotRunReason::InsufficientResultCapacity {
                        limit: 1,
                        required: 2
                    }
                }
            ));

            let two = observe_live_lexical_contract(
                &cx,
                &LexicalContractSpy::new(false),
                live_build_context_with_limit("quill", LexicalEngineRole::Subject, 2),
            )
            .await
            .expect("two-result-capacity contract evidence");
            for execution in [
                &two.all_lexical_winners_hydration.execution,
                &two.strict_hybrid_winners_hydration.execution,
                &two.semantic_only_hydration.execution,
                &two.mixed_winners_hydration.execution,
            ] {
                assert!(matches!(
                    execution,
                    LexicalHydrationExecution::Attempted {
                        result: LexicalHydrationResult::Success,
                        ..
                    }
                ));
            }
        });
    }

    #[test]
    fn exact_score_tie_order_is_classified_without_rewriting_native_order() {
        let subject = observation(vec![quill_hit("b", 4.0, 1), quill_hit("a", 4.0, 2)]);
        let oracle = observation(vec![tantivy_hit("a", 4.0, 8), tantivy_hit("b", 4.0, 9)]);

        let report =
            compare_observations(subject.clone(), oracle.clone(), ComparatorConfig::default())
                .expect("tie comparison");

        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.subject.hits, subject.hits);
        assert_eq!(report.oracle.hits, oracle.hits);
        assert_eq!(
            report.first_divergence.as_deref(),
            Some("/comparison/subject/hits/0")
        );
    }

    #[test]
    fn top_k_tie_substitution_requires_complete_expansion() {
        let subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("c", 4.0, 3)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);
        oracle.match_count = CountState::Value(3);
        oracle.cutoff_tie_group = vec![tantivy_hit("b", 4.0, 2), tantivy_hit("c", 4.0, 3)];

        let classified = compare_observations(
            EngineObservation {
                match_count: CountState::Value(3),
                ..subject.clone()
            },
            oracle.clone(),
            ComparatorConfig::default(),
        )
        .expect("complete tie evidence");
        assert_eq!(classified.rank_class, RankClass::TieOrder);

        oracle.cutoff_tie_complete = false;
        let failed = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("incomplete tie evidence fails closed");
        assert_eq!(failed.rank_class, RankClass::RankMismatch);
        assert_eq!(failed.status, ComparisonStatus::Failed);
    }

    #[test]
    fn epsilon_reorders_only_inside_oracle_connected_components() {
        let subject = observation(vec![
            quill_hit("b", 10.0004, 2),
            quill_hit("a", 9.9996, 1),
            quill_hit("c", 8.0, 3),
        ]);
        let oracle = observation(vec![
            tantivy_hit("a", 10.0, 1),
            tantivy_hit("b", 9.9999, 2),
            tantivy_hit("c", 8.0, 3),
        ]);
        let report = compare_observations(
            subject,
            oracle,
            ComparatorConfig::default()
                .with_score_epsilon_reason(ScoreEpsilonReason::OracleSegmentGeometry),
        )
        .expect("epsilon comparison");
        assert_eq!(report.rank_class, RankClass::ScoreEpsilon);
        assert_eq!(
            report.score_epsilon_reason,
            Some(ScoreEpsilonReason::OracleSegmentGeometry)
        );
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn epsilon_above_contract_or_without_reason_is_rejected() {
        assert!(ComparatorConfig::new(0.001).is_err());
        let subject = observation(vec![quill_hit("a", 1.000_05, 1)]);
        let oracle = observation(vec![tantivy_hit("a", 1.0, 1)]);
        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("unreviewed epsilon fails as a rank mismatch");
        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
        assert_eq!(report.score_epsilon_reason, None);
    }

    #[test]
    fn rank_exact_rejects_out_of_order_native_tie_keys() {
        let subject = observation(vec![quill_hit("a", 1.0, 2), quill_hit("b", 1.0, 1)]);
        let oracle = observation(vec![tantivy_hit("a", 1.0, 1), tantivy_hit("b", 1.0, 2)]);
        assert!(compare_observations(subject, oracle, ComparatorConfig::default()).is_err());
    }

    #[test]
    fn duplicate_native_identity_across_scores_is_rejected() {
        let subject = observation(vec![quill_hit("a", 2.0, 7), quill_hit("b", 1.0, 7)]);
        let oracle = observation(vec![tantivy_hit("a", 2.0, 1), tantivy_hit("b", 1.0, 2)]);

        assert!(matches!(
            compare_observations(subject, oracle, ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn top_k_and_cutoff_identity_must_be_consistent() {
        let oracle = observation(vec![tantivy_hit("a", 2.0, 1)]);

        let mut changed_score = observation(vec![quill_hit("a", 2.0, 7)]);
        changed_score.cutoff_tie_group = vec![quill_hit("a", 1.0, 7)];
        assert!(matches!(
            compare_observations(changed_score, oracle.clone(), ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut changed_doc = observation(vec![quill_hit("a", 2.0, 7)]);
        changed_doc.cutoff_tie_group = vec![quill_hit("b", 2.0, 7)];
        assert!(matches!(
            compare_observations(changed_doc, oracle, ComparatorConfig::default()),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn cutoff_tie_substitution_does_not_compare_unaligned_snippets() {
        let mut subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("c", 4.0, 3)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);
        subject.match_count = CountState::Value(3);
        oracle.match_count = CountState::Value(3);
        oracle.cutoff_tie_group = vec![tantivy_hit("b", 4.0, 2), tantivy_hit("c", 4.0, 3)];
        subject.snippets.insert("c".to_owned(), "c body".to_owned());
        oracle.snippets.insert("b".to_owned(), "b body".to_owned());

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("complete cutoff tie evidence");
        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn snippet_and_counts_have_stable_json_pointers() {
        let mut subject = observation(vec![quill_hit("a/b~c", 1.0, 1)]);
        let mut oracle = observation(vec![tantivy_hit("a/b~c", 1.0, 1)]);
        subject
            .snippets
            .insert("a/b~c".to_owned(), "left".to_owned());
        oracle
            .snippets
            .insert("a/b~c".to_owned(), "right".to_owned());
        subject.match_count = CountState::Value(2);
        subject.doc_count = 8;

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("surface comparison");
        let pointers = report
            .divergences
            .iter()
            .map(|item| item.pointer.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            pointers,
            vec![
                "/comparison/subject/snippets/a~1b~0c",
                "/comparison/subject/match_count",
                "/comparison/subject/doc_count"
            ]
        );
        let projection = serde_json::json!({ "comparison": &report });
        assert!(
            report
                .divergences
                .iter()
                .all(|divergence| projection.pointer(&divergence.pointer).is_some())
        );
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn oversized_query_token_ast_difference_is_classified_not_failed() {
        // Result-equivalent lowering: identical hits, one recorded AST
        // difference for the oversized-token admission (register DIV-004).
        let mut subject = observation(vec![quill_hit("a", 5.0, 1), quill_hit("b", 4.0, 2)]);
        subject.ast_differences.push(AstDifference {
            kind: AstLoweringKind::OversizedQueryToken,
            oracle: "BooleanQuery(TermQuery(content:oversized))".to_owned(),
            subject: "Empty (oversized > 65,530-byte token admitted as MatchNone)".to_owned(),
        });
        let oracle = observation(vec![tantivy_hit("a", 5.0, 1), tantivy_hit("b", 4.0, 2)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("oversized-token comparison");

        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.rank_class, RankClass::RankExact);
        assert_eq!(report.divergences.len(), 1);
        assert_eq!(
            report.divergences[0].class,
            DivergenceClass::OversizedQueryToken
        );
        assert_eq!(
            report.divergences[0].pointer,
            "/comparison/subject/ast_differences/0"
        );
        // The classified pointer resolves in the serialized artifact.
        let projection = serde_json::json!({ "comparison": &report });
        assert!(projection.pointer(&report.divergences[0].pointer).is_some());
    }

    #[test]
    fn reviewed_ast_difference_taxonomy_is_closed_and_classified() {
        let reviewed = [
            (
                AstLoweringKind::QueryCanonicalization,
                DivergenceClass::QueryCanonicalization,
            ),
            (AstLoweringKind::OracleBug, DivergenceClass::OracleBug),
            (
                AstLoweringKind::GlobExpansionLimit,
                DivergenceClass::GlobExpansionLimit,
            ),
            (
                AstLoweringKind::StatsSemantics,
                DivergenceClass::StatsSemantics,
            ),
            (AstLoweringKind::UnicodeEdge, DivergenceClass::UnicodeEdge),
        ];
        for (kind, expected_class) in reviewed {
            let mut subject = observation(vec![quill_hit("a", 5.0, 1)]);
            subject.ast_differences.push(AstDifference {
                kind,
                oracle: "reviewed oracle shape".to_owned(),
                subject: "reviewed subject shape".to_owned(),
            });
            let oracle = observation(vec![tantivy_hit("a", 5.0, 1)]);
            let report = compare_observations(subject, oracle, ComparatorConfig::default())
                .expect("reviewed AST comparison");
            assert_eq!(report.status, ComparisonStatus::Classified);
            assert_eq!(report.rank_class, RankClass::RankExact);
            assert_eq!(report.divergences.len(), 1);
            assert_eq!(report.divergences[0].class, expected_class);
        }
    }

    #[test]
    fn ast_difference_does_not_mask_result_level_failures() {
        // An oversized-token record may accompany only result-equivalent
        // runs; a real rank divergence still fails closed.
        let mut subject = observation(vec![quill_hit("a", 5.0, 1)]);
        subject.ast_differences.push(AstDifference {
            kind: AstLoweringKind::OversizedQueryToken,
            oracle: "TermQuery".to_owned(),
            subject: "Empty".to_owned(),
        });
        let oracle = observation(vec![tantivy_hit("b", 5.0, 1)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("rank divergence with ast record");

        assert_eq!(report.status, ComparisonStatus::Failed);
        assert!(
            report
                .divergences
                .iter()
                .any(|divergence| divergence.class == DivergenceClass::RankMismatch)
        );
    }

    #[test]
    fn oracle_side_ast_difference_fails_closed() {
        let subject = observation(vec![quill_hit("a", 5.0, 1)]);
        let mut oracle = observation(vec![tantivy_hit("a", 5.0, 1)]);
        oracle.ast_differences.push(AstDifference {
            kind: AstLoweringKind::OversizedQueryToken,
            oracle: "unexpected oracle diagnostic".to_owned(),
            subject: "unexpected oracle lowering".to_owned(),
        });

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("oracle-side AST evidence is represented as a failure");

        assert_eq!(report.status, ComparisonStatus::Failed);
        assert_eq!(report.divergences.len(), 1);
        assert_eq!(report.divergences[0].class, DivergenceClass::RankMismatch);
        assert_eq!(
            report.divergences[0].pointer,
            "/comparison/oracle/ast_differences/0"
        );
    }

    #[test]
    fn observation_without_ast_differences_still_deserializes() {
        // Artifacts written before the ast_differences channel existed must
        // keep parsing (serde default).
        let legacy = serde_json::json!({
            "hits": [],
            "cutoff_tie_group": [],
            "cutoff_tie_complete": true,
            "snippets": {},
            "match_count": "not_requested",
            "doc_count": 0,
        });
        let observation: EngineObservation =
            serde_json::from_value(legacy).expect("legacy observation parses");
        assert!(observation.ast_differences.is_empty());
        assert!(observation.offset_tie_group.is_empty());
        assert!(!observation.offset_tie_complete);
    }

    #[test]
    fn leading_offset_tie_substitution_is_classified_tie_order() {
        // Page [C9, D8] at offset 2 inside oracle order A10,B9,C9,D8: the
        // subject's native order walked B9 into the page instead of C9. The
        // complete leading group {B9, C9} proves the membership difference
        // is order-only.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("offset tie comparison");

        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
        assert_eq!(report.divergences[0].class, DivergenceClass::TieOrder);
    }

    #[test]
    fn leading_offset_tie_substitution_fails_closed_without_complete_group() {
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = false;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("incomplete offset evidence fails closed");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn leading_and_trailing_substitutions_combine_across_boundaries() {
        // Offset 2 cuts {B9, C9}; the page tail cuts {D8, E8}. Both
        // substitutions are explainable when both complete groups exist.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        oracle.cutoff_tie_group = vec![tantivy_hit("d", 8.0, 3), tantivy_hit("e", 8.0, 4)];
        oracle.cutoff_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("e", 8.0, 4)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("two-boundary tie comparison");

        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn two_boundary_substitutions_allow_an_exact_middle_hit() {
        let mut oracle = observation(vec![
            tantivy_hit("c", 9.0, 2),
            tantivy_hit("middle", 8.0, 3),
            tantivy_hit("d", 7.0, 4),
        ]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        oracle.cutoff_tie_group = vec![tantivy_hit("d", 7.0, 4), tantivy_hit("e", 7.0, 5)];
        oracle.cutoff_tie_complete = true;
        let subject = observation(vec![
            quill_hit("b", 9.0, 1),
            quill_hit("middle", 8.0, 3),
            quill_hit("e", 7.0, 5),
        ]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("two boundary substitutions with exact middle");

        assert_eq!(report.rank_class, RankClass::TieOrder);
        assert_eq!(report.status, ComparisonStatus::Classified);
    }

    #[test]
    fn cross_boundary_score_substitution_fails_closed() {
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        oracle.cutoff_tie_group = vec![tantivy_hit("d", 8.0, 3), tantivy_hit("e", 8.0, 4)];
        oracle.cutoff_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 9.0, 1), quill_hit("c", 9.0, 2)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("cross-boundary substitution is a real mismatch");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn unrelated_boundary_membership_fails_closed() {
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("x", 9.0, 8), tantivy_hit("y", 9.0, 9)];
        oracle.offset_tie_complete = true;
        let subject = observation(vec![quill_hit("x", 9.0, 8), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("unrelated boundary group is not proof");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn unexplained_score_difference_at_leading_edge_is_rank_mismatch() {
        // The substitute has a different score than every boundary group:
        // genuinely divergent, not a tie artifact.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("b", 9.0, 1), tantivy_hit("c", 9.0, 2)];
        oracle.offset_tie_complete = true;
        let subject = observation(vec![quill_hit("b", 8.0, 1), quill_hit("d", 8.0, 3)]);

        let report = compare_observations(subject, oracle, ComparatorConfig::default())
            .expect("score-mismatched substitute fails closed");

        assert_eq!(report.rank_class, RankClass::RankMismatch);
        assert_eq!(report.status, ComparisonStatus::Failed);
    }

    #[test]
    fn offset_group_inconsistent_with_page_hits_is_rejected() {
        // The same document appears in the page and in the offset group with
        // a different score: the evidence is internally inconsistent.
        let mut oracle = observation(vec![tantivy_hit("c", 9.0, 2), tantivy_hit("d", 8.0, 3)]);
        oracle.offset_tie_group = vec![tantivy_hit("c", 7.0, 2)];
        oracle.offset_tie_complete = true;

        let error = compare_observations(oracle.clone(), oracle, ComparatorConfig::default())
            .expect_err("inconsistent offset evidence is rejected");
        assert!(matches!(error, GauntletError::InvalidObservation { .. }));
    }

    #[test]
    fn lexical_observation_backend_identity_is_provenance_not_equivalence() {
        let subject = complete_lexical_observation("quill");
        let oracle = complete_lexical_observation("tantivy");

        let report =
            compare_lexical_observations(subject, oracle).expect("complete lexical comparison");

        assert_eq!(report.status, LexicalComparisonStatus::Equivalent);
        assert!(report.mismatches.is_empty());
        assert_eq!(report.applied_laws, LEXICAL_EQUIVALENCE_LAWS.to_vec());
        assert_eq!(report.subject.context.backend.engine, "quill");
        assert_eq!(report.oracle.context.backend.engine, "tantivy");
    }

    #[test]
    fn lexical_contract_accepts_only_registered_deferred_metadata() {
        let subject = lexical_contract_bundle("quill", true);
        let oracle = lexical_contract_bundle("tantivy", false);

        let report =
            compare_lexical_contracts(subject, oracle).expect("valid total-contract comparison");

        assert_eq!(
            report.schema_version,
            LEXICAL_CONTRACT_COMPARISON_SCHEMA_VERSION
        );
        assert_eq!(report.status, LexicalComparisonStatus::Equivalent);
        assert!(report.mismatches.is_empty());
        assert!(report.subject.fusion_metadata_deferred);
        assert!(!report.oracle.fusion_metadata_deferred);
        assert!(
            report
                .applied_laws
                .contains(&LexicalEquivalenceLaw::DeferredMetadataHydration)
        );
        assert!(!report.waived_differences.is_empty());
        assert_eq!(
            report.coverage.subject.all_lexical_winners_hydration,
            LexicalProbeCoverage::ExercisedRestoration
        );
        assert_eq!(
            report.coverage.subject.strict_hybrid_winners_hydration,
            LexicalProbeCoverage::ExercisedRestoration
        );
        assert_eq!(
            report.coverage.subject.semantic_only_hydration,
            LexicalProbeCoverage::ExercisedSuccess
        );
        assert_eq!(
            report.coverage.subject.mixed_winners_hydration,
            LexicalProbeCoverage::ExercisedRestoration
        );
        report.validate_replay().expect("comparison must replay");
    }

    #[test]
    fn lexical_contract_deferral_law_covers_all_eager_and_deferred_pairings() {
        for (subject_deferred, oracle_deferred, expected_side, expect_waivers) in [
            (false, false, None, false),
            (true, false, Some(LexicalDeferredSide::Subject), true),
            (false, true, Some(LexicalDeferredSide::Oracle), true),
            (true, true, Some(LexicalDeferredSide::Both), false),
        ] {
            let report = compare_lexical_contracts(
                lexical_contract_bundle("quill", subject_deferred),
                lexical_contract_bundle("tantivy", oracle_deferred),
            )
            .expect("valid eager/deferred pairing");
            assert_eq!(report.status, LexicalComparisonStatus::Equivalent);
            assert_eq!(
                report
                    .applied_laws
                    .contains(&LexicalEquivalenceLaw::DeferredMetadataHydration),
                expected_side.is_some()
            );
            assert_eq!(!report.waived_differences.is_empty(), expect_waivers);
            assert!(
                report
                    .waived_differences
                    .iter()
                    .all(|waiver| Some(waiver.deferred_side) == expected_side)
            );
            if subject_deferred {
                assert_eq!(
                    report.coverage.subject.all_lexical_winners_hydration,
                    LexicalProbeCoverage::ExercisedRestoration
                );
            }
            if oracle_deferred {
                assert_eq!(
                    report.coverage.oracle.all_lexical_winners_hydration,
                    LexicalProbeCoverage::ExercisedRestoration
                );
            }
            report.validate_replay().expect("pairing must replay");
        }
    }

    #[test]
    fn lexical_contract_accepts_the_recorded_non_prefix_hybrid_subset() {
        let subject = lexical_contract_bundle("quill", true);
        let oracle = lexical_contract_bundle("tantivy", false);

        let report = compare_lexical_contracts(subject, oracle)
            .expect("rank-one strict subset is valid total-contract evidence");

        assert_eq!(report.status, LexicalComparisonStatus::Equivalent);
        assert!(report.waived_differences.iter().any(|waiver| {
            matches!(
                &waiver.target,
                LexicalWaiverTarget::HydrationInputMetadata {
                    probe: LexicalHydrationProbe::StrictHybridWinnerSubset,
                    position: 0,
                    candidate_rank: 1,
                }
            )
        }));
        report.validate_replay().expect("rank-one subset replay");
    }

    #[test]
    fn lexical_contract_rejects_missing_or_mislabeled_lanes() {
        let oracle = lexical_contract_bundle("tantivy", false);

        let encoded =
            serde_json::to_value(lexical_contract_bundle("quill", false)).expect("bundle JSON");
        for lane in [
            "full_search",
            "fusion_candidates",
            "all_lexical_winners_hydration",
            "strict_hybrid_winners_hydration",
            "semantic_only_hydration",
            "mixed_winners_hydration",
        ] {
            let mut missing_lane = encoded.clone();
            assert!(
                missing_lane
                    .as_object_mut()
                    .expect("bundle object")
                    .remove(lane)
                    .is_some(),
                "fixture must contain required lane {lane}"
            );
            assert!(
                serde_json::from_value::<LexicalContractBundle>(missing_lane).is_err(),
                "a persisted bundle missing required lane {lane} must fail before verification"
            );
        }

        let mut mislabeled = lexical_contract_bundle("quill", false);
        mislabeled.fusion_candidates.context.boundary = LexicalBoundary::FullSearch;
        assert!(matches!(
            compare_lexical_contracts(mislabeled, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut different_request = lexical_contract_bundle("quill", false);
        let LexicalHydrationExecution::Attempted { input, .. } =
            &mut different_request.all_lexical_winners_hydration.execution
        else {
            panic!("fixture must attempt all-lexical-winner hydration");
        };
        input.context.query_contract_sha256 = "f".repeat(64);
        assert!(matches!(
            compare_lexical_contracts(different_request, oracle),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_contract_rejects_false_deferral_and_incomplete_hydration() {
        let oracle = lexical_contract_bundle("tantivy", false);

        let mut false_deferral = lexical_contract_bundle("quill", true);
        let hydrated_metadata = lexical_hit_mut(&mut false_deferral.full_search)
            .metadata
            .clone();
        lexical_hit_mut(&mut false_deferral.fusion_candidates).metadata = hydrated_metadata;
        assert!(matches!(
            compare_lexical_contracts(false_deferral, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut incomplete_hydration = lexical_contract_bundle("quill", true);
        let LexicalHydrationExecution::Attempted { post_state, .. } =
            &mut incomplete_hydration.all_lexical_winners_hydration.execution
        else {
            panic!("fixture must attempt all-lexical-winner hydration");
        };
        lexical_hit_mut(post_state).metadata = SensitiveValueObservation::Absent;
        assert!(matches!(
            compare_lexical_contracts(incomplete_hydration, oracle),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut candidate_order_drift = lexical_contract_bundle("quill", true);
        lexical_hit_at_mut(&mut candidate_order_drift.fusion_candidates, 0).doc_id =
            "different-doc".to_owned();
        let error = compare_lexical_contracts(
            candidate_order_drift,
            lexical_contract_bundle("tantivy", false),
        )
        .expect_err("candidate order drift must make a deferred bundle internally invalid");
        assert!(
            matches!(
                error,
                GauntletError::InvalidObservation { ref reason }
                    if reason.ends_with("/outcome/hits/0/doc_id")
            ),
            "internal coherence diagnostics must identify the first bounded field path: {error}"
        );
    }

    #[test]
    fn lexical_contract_never_hides_non_metadata_divergence() {
        let mut subject = lexical_contract_bundle("quill", true);
        let oracle = lexical_contract_bundle("tantivy", false);
        for observation in [&mut subject.full_search, &mut subject.fusion_candidates] {
            lexical_hit_at_mut(observation, 1).doc_id = "different-doc".to_owned();
        }
        for transition in [
            &mut subject.all_lexical_winners_hydration,
            &mut subject.strict_hybrid_winners_hydration,
            &mut subject.mixed_winners_hydration,
        ] {
            let position = match transition.selection {
                LexicalHydrationSelection::StrictHybridWinnerSubset { .. } => 0,
                LexicalHydrationSelection::AllLexicalWinners
                | LexicalHydrationSelection::SemanticOnlyControl { .. }
                | LexicalHydrationSelection::MixedFinalWinners { .. } => 1,
            };
            let (input, post_state) = attempted_observations_mut(transition);
            lexical_hit_at_mut(input, position).doc_id = "different-doc".to_owned();
            lexical_hit_at_mut(post_state, position).doc_id = "different-doc".to_owned();
        }

        let report =
            compare_lexical_contracts(subject, oracle).expect("internally coherent bundle");

        assert_eq!(report.status, LexicalComparisonStatus::Mismatch);
        assert!(report.mismatches.iter().any(|mismatch| {
            mismatch.class == LexicalMismatchClass::Ordering
                && mismatch.path == "/full_search/outcome/hits/1/doc_id"
        }));
        assert!(report.mismatches.iter().any(|mismatch| {
            mismatch.class == LexicalMismatchClass::Ordering
                && mismatch.path
                    == "/all_lexical_winners_hydration/post_state/outcome/hits/1/doc_id"
        }));
        assert!(
            report
                .waived_differences
                .iter()
                .all(|waiver| { waiver.mismatch.class == LexicalMismatchClass::Metadata })
        );
    }

    #[test]
    fn lexical_contract_schema_round_trips_and_rejects_unknown_fields() {
        let bundle = lexical_contract_bundle("quill", true);
        let encoded = serde_json::to_value(&bundle).expect("serialize lexical contract bundle");
        let decoded: LexicalContractBundle =
            serde_json::from_value(encoded.clone()).expect("deserialize lexical contract bundle");
        assert_eq!(decoded, bundle);

        let mut unknown_top_level = encoded.clone();
        unknown_top_level
            .as_object_mut()
            .expect("bundle object")
            .insert("unexpected".to_owned(), serde_json::json!(true));
        assert!(serde_json::from_value::<LexicalContractBundle>(unknown_top_level).is_err());

        let mut unknown_nested_context = encoded;
        unknown_nested_context["full_search"]["context"]["unexpected"] =
            serde_json::json!("must fail closed");
        assert!(serde_json::from_value::<LexicalContractBundle>(unknown_nested_context).is_err());

        let mut wrong_schema = bundle;
        wrong_schema.schema_version = "lexical-contract-bundle-v999".to_owned();
        assert!(matches!(
            compare_lexical_contracts(wrong_schema, lexical_contract_bundle("tantivy", false)),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_contract_replay_rejects_every_derived_field_tamper() {
        let comparison = compare_lexical_contracts(
            lexical_contract_bundle("quill", true),
            lexical_contract_bundle("tantivy", false),
        )
        .expect("valid replay fixture");
        comparison.validate_replay().expect("untampered replay");

        let mut wrong_status = comparison.clone();
        wrong_status.status = LexicalComparisonStatus::Mismatch;
        assert!(matches!(
            wrong_status.validate_replay(),
            Err(GauntletError::InvalidContract { .. })
        ));

        let mut missing_waiver = comparison.clone();
        missing_waiver.waived_differences.pop();
        assert!(matches!(
            missing_waiver.validate_replay(),
            Err(GauntletError::InvalidContract { .. })
        ));

        let mut wrong_coverage = comparison.clone();
        wrong_coverage
            .coverage
            .subject
            .strict_hybrid_winners_hydration = LexicalProbeCoverage::ExercisedEmpty;
        assert!(matches!(
            wrong_coverage.validate_replay(),
            Err(GauntletError::InvalidContract { .. })
        ));

        let mut unknown_field =
            serde_json::to_value(&comparison).expect("serialize lexical comparison");
        unknown_field
            .as_object_mut()
            .expect("comparison object")
            .insert("unexpected".to_owned(), serde_json::json!(true));
        assert!(serde_json::from_value::<LexicalContractComparison>(unknown_field).is_err());
    }

    #[test]
    fn lexical_contract_exposure_and_transition_mutations_fail_closed() {
        let oracle = lexical_contract_bundle("tantivy", false);

        let mut missing_exposed_metadata = lexical_contract_bundle("quill", false);
        lexical_hit_mut(&mut missing_exposed_metadata.full_search).metadata =
            SensitiveValueObservation::NotExposed;
        assert!(matches!(
            compare_lexical_contracts(missing_exposed_metadata, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut changed_non_lexical = lexical_contract_bundle("quill", false);
        let LexicalHydrationExecution::Attempted { post_state, .. } =
            &mut changed_non_lexical.semantic_only_hydration.execution
        else {
            panic!("fixture must attempt semantic-only hydration");
        };
        lexical_hit_mut(post_state).normalized_score_bits = 3.25_f32.to_bits();
        assert!(matches!(
            compare_lexical_contracts(changed_non_lexical, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut out_of_range_subset = lexical_contract_bundle("quill", false);
        out_of_range_subset
            .strict_hybrid_winners_hydration
            .selection = LexicalHydrationSelection::StrictHybridWinnerSubset {
            candidate_ranks: vec![99],
        };
        assert!(matches!(
            compare_lexical_contracts(out_of_range_subset, oracle),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_contract_rejects_score_order_cardinality_and_origin_mutations() {
        let oracle = lexical_contract_bundle("tantivy", false);

        let mut unsorted = lexical_contract_bundle("quill", false);
        let hit = lexical_hit_at_mut(&mut unsorted.full_search, 1);
        hit.normalized_score_bits = 9.0_f32.to_bits();
        hit.raw_lexical_score_bits = Some(9.0_f32.to_bits());
        assert!(matches!(
            compare_lexical_contracts(unsorted, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut extra_rank = lexical_contract_bundle("quill", false);
        extra_rank.strict_hybrid_winners_hydration.selection =
            LexicalHydrationSelection::StrictHybridWinnerSubset {
                candidate_ranks: vec![1, 0],
            };
        assert!(matches!(
            compare_lexical_contracts(extra_rank, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut missing_input = lexical_contract_bundle("quill", false);
        let observations: [&mut LexicalObservation; 2] =
            attempted_observations_mut(&mut missing_input.strict_hybrid_winners_hydration).into();
        for observation in observations {
            let (hits, returned_count, empty_shape, _) =
                lexical_success_mut(observation).expect("strict fixture is successful");
            hits.clear();
            *returned_count = 0;
            *empty_shape = LexicalEmptyShape::Empty;
        }
        assert!(matches!(
            compare_lexical_contracts(missing_input, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut wrong_strict_origin = lexical_contract_bundle("quill", false);
        wrong_strict_origin
            .strict_hybrid_winners_hydration
            .selection = LexicalHydrationSelection::StrictHybridWinnerSubset {
            candidate_ranks: vec![0],
        };
        assert!(matches!(
            compare_lexical_contracts(wrong_strict_origin, oracle.clone()),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut reordered_mixed_origins = lexical_contract_bundle("quill", false);
        let LexicalHydrationSelection::MixedFinalWinners { origins } =
            &mut reordered_mixed_origins.mixed_winners_hydration.selection
        else {
            panic!("mixed fixture must record exact origins");
        };
        origins.swap(0, 1);
        assert!(matches!(
            compare_lexical_contracts(reordered_mixed_origins, oracle),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_contract_rejects_nonlexical_hydration_mutation() {
        let oracle = lexical_contract_bundle("tantivy", false);

        for mutate in [
            |hit: &mut LexicalHitObservation| {
                hit.metadata = SensitiveValueObservation::Absent;
            },
            |hit: &mut LexicalHitObservation| {
                hit.explanation = SensitiveValueObservation::Absent;
            },
        ] {
            let mut semantic = lexical_contract_bundle("quill", false);
            let (_, post_state) = attempted_observations_mut(&mut semantic.semantic_only_hydration);
            mutate(lexical_hit_mut(post_state));
            assert!(matches!(
                compare_lexical_contracts(semantic, oracle.clone()),
                Err(GauntletError::InvalidObservation { .. })
            ));
        }

        let mut mixed = lexical_contract_bundle("quill", false);
        let (_, post_state) = attempted_observations_mut(&mut mixed.mixed_winners_hydration);
        lexical_hit_at_mut(post_state, 0).metadata = SensitiveValueObservation::Absent;
        assert!(matches!(
            compare_lexical_contracts(mixed, oracle),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_observation_context_field_mutations_are_detected() {
        assert_single_lexical_mismatch(
            |observation| observation.context.boundary = LexicalBoundary::FusionCandidates,
            LexicalMismatchClass::Context,
            "/context/boundary",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.corpus_sha256 = "b".repeat(64),
            LexicalMismatchClass::Context,
            "/context/corpus_sha256",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.query_contract_sha256 = "c".repeat(64),
            LexicalMismatchClass::Context,
            "/context/query_contract_sha256",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.query_sha256 = "b".repeat(64),
            LexicalMismatchClass::Context,
            "/context/query_sha256",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.query_bytes += 1,
            LexicalMismatchClass::Context,
            "/context/query_bytes",
        );
        assert_single_lexical_mismatch(
            |observation| {
                observation.context.normalized_query = LexicalNormalizedQuery::Value {
                    transform_id: "test-normalizer-v1".to_owned(),
                    sha256: "c".repeat(64),
                    byte_len: 11,
                };
            },
            LexicalMismatchClass::Context,
            "/context/normalized_query",
        );
        assert_single_lexical_mismatch(
            |observation| {
                observation.context.query_class = LexicalQueryClass::NaturalLanguage;
            },
            LexicalMismatchClass::Context,
            "/context/query_class",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.seed += 1,
            LexicalMismatchClass::Context,
            "/context/seed",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.limit += 1,
            LexicalMismatchClass::Context,
            "/context/limit",
        );
    }

    #[test]
    fn lexical_observation_hit_field_mutations_are_detected_or_rejected_by_boundary() {
        assert_single_lexical_mismatch(
            |observation| lexical_hit_mut(observation).doc_id = "doc-2".to_owned(),
            LexicalMismatchClass::Ordering,
            "/outcome/hits/0/doc_id",
        );
        // FullSearch requires raw lexical score bits to equal the normalized
        // score bits. Mutating either half alone is malformed evidence.
        assert_invalid_lexical_mutation(|observation| {
            let hit = lexical_hit_mut(observation);
            hit.normalized_score_bits = f32::from_bits(hit.normalized_score_bits + 1).to_bits();
        });
        assert_invalid_lexical_mutation(|observation| {
            lexical_hit_mut(observation).raw_lexical_score_bits = Some(4.0_f32.to_bits());
        });
        // These component shapes are forbidden at the ordinary FullSearch
        // boundary and are covered separately by hydration-boundary fixtures.
        assert_invalid_lexical_mutation(|observation| {
            lexical_hit_mut(observation).source = LexicalScoreSource::Hybrid;
        });
        assert_invalid_lexical_mutation(|observation| lexical_hit_mut(observation).index = Some(8));
        assert_invalid_lexical_mutation(|observation| {
            lexical_hit_mut(observation).fast_score_bits = Some(0.26_f32.to_bits());
        });
        assert_invalid_lexical_mutation(|observation| {
            lexical_hit_mut(observation).quality_score_bits = Some(0.51_f32.to_bits());
        });
        assert_invalid_lexical_mutation(|observation| {
            lexical_hit_mut(observation).rerank_score_bits = Some(0.76_f32.to_bits());
        });
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).metadata = SensitiveValueObservation::Absent;
            },
            LexicalMismatchClass::Metadata,
            "/outcome/hits/0/metadata",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).explanation = SensitiveValueObservation::Absent;
            },
            LexicalMismatchClass::Explanation,
            "/outcome/hits/0/explanation",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).snippet =
                    SensitiveValueObservation::from_text("changed snippet");
            },
            LexicalMismatchClass::Snippet,
            "/outcome/hits/0/snippet",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).highlight_spans =
                    LexicalObserved::Value(vec![LexicalHighlightSpan { start: 1, end: 4 }]);
            },
            LexicalMismatchClass::Highlight,
            "/outcome/hits/0/highlight_spans",
        );
    }

    #[test]
    fn lexical_observation_rejects_highlights_without_matching_snippet_bytes() {
        let mut past_end = complete_lexical_observation("quill");
        lexical_hit_mut(&mut past_end).highlight_spans =
            LexicalObserved::Value(vec![LexicalHighlightSpan {
                start: 0,
                end: 10_000,
            }]);
        assert!(matches!(
            compare_lexical_observations(past_end, complete_lexical_observation("tantivy")),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let mut absent_snippet = complete_lexical_observation("quill");
        lexical_hit_mut(&mut absent_snippet).snippet = SensitiveValueObservation::Absent;
        assert!(matches!(
            compare_lexical_observations(absent_snippet, complete_lexical_observation("tantivy")),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_observation_count_and_shape_contracts_fail_closed() {
        let oracle = complete_lexical_observation("tantivy");
        let mut subject = complete_lexical_observation("quill");
        let (_, _, _, total_count) =
            lexical_success_mut(&mut subject).expect("test fixture must be a success");
        *total_count = LexicalCountState::Value(2);
        let report = compare_lexical_observations(subject, oracle).expect("valid count comparison");
        assert_eq!(
            report
                .mismatches
                .iter()
                .map(|mismatch| mismatch.path.as_str())
                .collect::<Vec<_>>(),
            vec!["/outcome/total_count"]
        );

        for mutate in [
            |observation: &mut LexicalObservation| {
                let (_, returned_count, _, _) =
                    lexical_success_mut(observation).expect("test fixture must be a success");
                *returned_count = 2;
            },
            |observation: &mut LexicalObservation| {
                let (_, _, empty_shape, _) =
                    lexical_success_mut(observation).expect("test fixture must be a success");
                *empty_shape = LexicalEmptyShape::Empty;
            },
            |observation: &mut LexicalObservation| {
                let (_, _, _, total_count) =
                    lexical_success_mut(observation).expect("test fixture must be a success");
                *total_count = LexicalCountState::Value(0);
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).rank = 1;
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).normalized_score_bits = f32::NAN.to_bits();
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).highlight_spans =
                    LexicalObserved::Value(vec![LexicalHighlightSpan { start: 4, end: 4 }]);
            },
            |observation: &mut LexicalObservation| {
                observation.context.limit = 0;
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).doc_id = "d".repeat(MAX_LEXICAL_DOC_ID_BYTES + 1);
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).metadata = SensitiveValueObservation::Present {
                    sha256: "d".repeat(64),
                    byte_len: u64::try_from(MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES)
                        .unwrap_or(u64::MAX)
                        .saturating_add(1),
                };
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).highlight_spans = LexicalObserved::Value(
                    (0..=MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT)
                        .map(|index| LexicalHighlightSpan {
                            start: u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(2),
                            end: u64::try_from(index)
                                .unwrap_or(u64::MAX)
                                .saturating_mul(2)
                                .saturating_add(1),
                        })
                        .collect(),
                );
            },
        ] {
            let mut malformed = complete_lexical_observation("quill");
            mutate(&mut malformed);
            let error =
                compare_lexical_observations(malformed, complete_lexical_observation("tantivy"))
                    .expect_err("malformed public observation must fail closed");
            assert!(matches!(error, GauntletError::InvalidObservation { .. }));
        }

        let mut invalid_schema = complete_lexical_observation("quill");
        invalid_schema.context.schema_version = "unknown".to_owned();
        assert!(matches!(
            compare_lexical_observations(invalid_schema, complete_lexical_observation("tantivy")),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_observation_rejects_unknown_supplements_and_duplicate_hits() {
        let unknown = LexicalObservationSupplement {
            total_count: LexicalCountState::NotExposed,
            hits: BTreeMap::from([("not-returned".to_owned(), LexicalHitSupplement::default())]),
        };
        assert!(matches!(
            observe_lexical_outcome(
                lexical_context("quill"),
                Ok(vec![lexical_result("doc-1", 3.5)]),
                &unknown,
            ),
            Err(GauntletError::InvalidObservation { .. })
        ));
        assert!(matches!(
            observe_lexical_outcome(
                lexical_context("quill"),
                Ok(vec![
                    lexical_result("duplicate", 3.5),
                    lexical_result("duplicate", 3.0),
                ]),
                &LexicalObservationSupplement::default(),
            ),
            Err(GauntletError::InvalidObservation { .. })
        ));
        assert!(matches!(
            observe_lexical_outcome(
                lexical_context("quill"),
                Err(SearchError::QueryParseError {
                    query: "secret".to_owned(),
                    detail: "secret detail".to_owned(),
                }),
                &lexical_supplement("doc-1"),
            ),
            Err(GauntletError::InvalidObservation { .. })
        ));
    }

    #[test]
    fn lexical_observation_canonicalizes_metadata_key_order() {
        let metadata = |reverse: bool| {
            let mut object = serde_json::Map::new();
            if reverse {
                object.insert("zeta".to_owned(), serde_json::json!(2));
                object.insert("alpha".to_owned(), serde_json::json!(1));
            } else {
                object.insert("alpha".to_owned(), serde_json::json!(1));
                object.insert("zeta".to_owned(), serde_json::json!(2));
            }
            Arc::new(serde_json::Value::Object(object))
        };
        let result = |engine: &str, reverse| {
            let mut hit = lexical_result("doc-1", 3.5);
            hit.metadata = Some(metadata(reverse));
            observe_lexical_outcome(
                lexical_context(engine),
                Ok(vec![hit]),
                &LexicalObservationSupplement::default(),
            )
            .expect("metadata observation")
        };

        let report = compare_lexical_observations(result("quill", true), result("tantivy", false))
            .expect("canonical metadata comparison");

        assert_eq!(report.status, LexicalComparisonStatus::Equivalent);
    }

    #[test]
    fn lexical_observation_detects_tie_reordering_without_score_rounding() {
        let observe = |engine: &str, ids: [&str; 2]| {
            observe_lexical_outcome(
                counted_lexical_context(engine),
                Ok(ids
                    .into_iter()
                    .map(|doc_id| lexical_result(doc_id, 3.5))
                    .collect()),
                &LexicalObservationSupplement {
                    total_count: LexicalCountState::Value(2),
                    hits: BTreeMap::new(),
                },
            )
            .expect("ordered tie observation")
        };
        let reordered = compare_lexical_observations(
            observe("quill", ["doc-b", "doc-a"]),
            observe("tantivy", ["doc-a", "doc-b"]),
        )
        .expect("tie ordering comparison");
        assert_eq!(reordered.status, LexicalComparisonStatus::Mismatch);
        assert_eq!(
            reordered
                .mismatches
                .iter()
                .map(|mismatch| mismatch.path.as_str())
                .collect::<Vec<_>>(),
            vec!["/outcome/hits/0/doc_id", "/outcome/hits/1/doc_id"]
        );

        let oracle = complete_lexical_observation("tantivy");
        let mut subject = complete_lexical_observation("quill");
        let changed_bits = lexical_hit_mut(&mut subject)
            .normalized_score_bits
            .checked_add(1)
            .expect("finite test score has a successor bit pattern");
        let hit = lexical_hit_mut(&mut subject);
        hit.normalized_score_bits = changed_bits;
        hit.raw_lexical_score_bits = Some(changed_bits);
        let score_report = compare_lexical_observations(subject, oracle)
            .expect("paired one-bit score mutation remains a valid lexical hit");
        assert_eq!(
            score_report
                .mismatches
                .iter()
                .map(|mismatch| mismatch.path.as_str())
                .collect::<Vec<_>>(),
            vec![
                "/outcome/hits/0/normalized_score_bits",
                "/outcome/hits/0/raw_lexical_score_bits"
            ]
        );
    }

    #[test]
    fn lexical_observation_preserves_empty_and_typed_error_shapes() {
        let empty = observe_lexical_outcome(
            counted_lexical_context("quill"),
            Ok(Vec::new()),
            &LexicalObservationSupplement {
                total_count: LexicalCountState::Value(0),
                hits: BTreeMap::new(),
            },
        )
        .expect("empty success observation");
        assert!(matches!(
            empty.outcome,
            LexicalObservationOutcome::Success {
                ref hits,
                returned_count: 0,
                empty_shape: LexicalEmptyShape::Empty,
                total_count: LexicalCountState::Value(0),
            } if hits.is_empty()
        ));

        let error_observation = |engine: &str| {
            observe_lexical_outcome(
                counted_lexical_context(engine),
                Err(SearchError::QueryParseError {
                    query: "raw secret query".to_owned(),
                    detail: "secret parser detail".to_owned(),
                }),
                &LexicalObservationSupplement::default(),
            )
            .expect("typed error observation")
        };
        let equivalent =
            compare_lexical_observations(error_observation("quill"), error_observation("tantivy"))
                .expect("typed error comparison");
        assert_eq!(equivalent.status, LexicalComparisonStatus::Equivalent);

        let unverifiable = observe_lexical_outcome(
            lexical_context("quill"),
            Err(SearchError::UnverifiableRemoteSpace {
                producer: "remote".to_owned(),
                reason: "sensitive explanation".to_owned(),
            }),
            &LexicalObservationSupplement::default(),
        )
        .expect("observe unverifiable remote space");
        assert!(matches!(
            unverifiable.outcome,
            LexicalObservationOutcome::Error(LexicalErrorObservation {
                class: LexicalErrorClass::Integrity,
                ref code,
                ..
            }) if code == "unverifiable_remote_space"
        ));

        let mutate_error = |mutate: fn(&mut LexicalErrorObservation), path: &str| {
            let oracle = error_observation("tantivy");
            let mut subject = error_observation("quill");
            let error = lexical_error_mut(&mut subject).expect("test fixture must be an error");
            mutate(error);
            let report =
                compare_lexical_observations(subject, oracle).expect("valid error comparison");
            assert_eq!(
                report
                    .mismatches
                    .iter()
                    .map(|mismatch| mismatch.path.as_str())
                    .collect::<Vec<_>>(),
                vec![path]
            );
        };
        mutate_error(
            |error| error.class = LexicalErrorClass::Timeout,
            "/outcome/error/class",
        );
        mutate_error(
            |error| error.code = "query_parse_changed".to_owned(),
            "/outcome/error/code",
        );
        mutate_error(
            |error| {
                error.contract_payload = SensitiveValueObservation::from_text("changed");
            },
            "/outcome/error/contract_payload",
        );
        let oracle = error_observation("tantivy");
        let mut subject = error_observation("quill");
        lexical_error_mut(&mut subject)
            .expect("test fixture must be an error")
            .diagnostic = SensitiveValueObservation::from_text("backend-specific display text");
        let diagnostic_report = compare_lexical_observations(subject, oracle)
            .expect("diagnostic-only difference must remain structurally valid");
        assert_eq!(
            diagnostic_report.status,
            LexicalComparisonStatus::Equivalent,
            "unstable Display text must not define error equivalence"
        );

        let outcome_mismatch = compare_lexical_observations(empty, error_observation("tantivy"))
            .expect("success/error comparison");
        assert_eq!(
            outcome_mismatch
                .first_mismatch
                .as_ref()
                .map(|mismatch| mismatch.path.as_str()),
            Some("/outcome/kind")
        );
    }

    #[test]
    fn lexical_error_observation_is_fixed_width_and_source_complete() {
        let queue = observe_lexical_search_error(&SearchError::QueueFull {
            pending: 7,
            capacity: 11,
        })
        .expect("queue error observation");
        assert_eq!(
            queue.contract_payload,
            SensitiveValueObservation::from_serializable(
                &serde_json::json!({
                    "pending": 7_u64,
                    "capacity": 11_u64,
                }),
                false,
            )
            .expect("canonical fixed-width queue payload")
        );

        let dimension = observe_lexical_search_error(&SearchError::DimensionMismatch {
            expected: 384,
            found: 128,
        })
        .expect("dimension error observation");
        assert_eq!(
            dimension.contract_payload,
            SensitiveValueObservation::from_serializable(
                &serde_json::json!({
                    "expected": 384_u64,
                    "found": 128_u64,
                }),
                false,
            )
            .expect("canonical fixed-width dimension payload")
        );

        let nested = |depth: usize| {
            let mut error = SearchError::DurabilityDisabled;
            for _ in 0..depth {
                error = SearchError::SubsystemError {
                    subsystem: "nested-test",
                    source: Box::new(error),
                };
            }
            error
        };
        let bounded = observe_lexical_search_error(&nested(MAX_LEXICAL_ERROR_SOURCE_DEPTH))
            .expect("maximum-depth source chain");
        assert_eq!(bounded.source_chain.len(), MAX_LEXICAL_ERROR_SOURCE_DEPTH);
        assert!(matches!(
            observe_lexical_search_error(&nested(MAX_LEXICAL_ERROR_SOURCE_DEPTH.saturating_add(1))),
            Err(GauntletError::InvalidObservation { .. })
        ));

        let error_observation = |engine: &str| {
            observe_lexical_outcome(
                lexical_context(engine),
                Err(SearchError::SubsystemError {
                    subsystem: "storage",
                    source: Box::new(std::io::Error::other("locked")),
                }),
                &LexicalObservationSupplement::default(),
            )
            .expect("nested source observation")
        };
        let oracle = error_observation("tantivy");
        let mut subject = error_observation("quill");
        lexical_error_mut(&mut subject)
            .expect("subject error")
            .source_chain[0] = SensitiveValueObservation::from_text("different source");
        let report = compare_lexical_observations(subject, oracle)
            .expect("mutated but structurally valid source chain");
        assert_eq!(
            report.status,
            LexicalComparisonStatus::Equivalent,
            "unversioned backend source Display text is diagnostic, not equivalence"
        );
        assert_ne!(
            lexical_error(&report.subject)
                .expect("subject error")
                .source_chain,
            lexical_error(&report.oracle)
                .expect("oracle error")
                .source_chain,
            "the diagnostic source-chain difference must remain in retained evidence"
        );
    }

    #[test]
    fn lexical_observation_artifacts_redact_sensitive_values() {
        const QUERY_CANARY: &str = "sensitive-query-canary-9f3a";
        const NORMALIZED_CANARY: &str = "sensitive-normalized-canary-2e7b";
        const METADATA_CANARY: &str = "sensitive-metadata-canary-36c1";
        const SNIPPET_CANARY: &str = "sensitive-snippet-canary-73d4";
        const EXPLANATION_CANARY: &str = "sensitive-explanation-canary-8bc2";

        let context = |engine: &str| {
            LexicalObservationContext::new(
                LexicalBoundary::FullSearch,
                LexicalBackendIdentity {
                    engine: engine.to_owned(),
                    revision: "redaction-test".to_owned(),
                    index_identity: "in-memory".to_owned(),
                },
                "d".repeat(64),
                "e".repeat(64),
                QUERY_CANARY,
                91,
                1,
                LexicalExposureContract {
                    metadata: LexicalFieldExposure::Exposed,
                    explanation: LexicalFieldExposure::Exposed,
                    total_count: LexicalCountExposure::NotRequested,
                    snippet: LexicalFieldExposure::Exposed,
                    highlight_spans: LexicalFieldExposure::Exposed,
                },
            )
            .expect("redaction context")
            .with_normalized_query("test-normalizer-v1", NORMALIZED_CANARY)
            .expect("normalized query witness")
        };
        let observation = |engine: &str, snippet: &str| {
            let mut hit = lexical_result("public-doc-id", 1.0);
            hit.metadata = Some(Arc::new(serde_json::json!({
                "private": METADATA_CANARY
            })));
            hit.explanation = Some(Box::new(HitExplanation {
                final_score: 1.0,
                components: Vec::new(),
                phase: ExplanationPhase::Refined,
                rank_movement: Some(frankensearch_core::RankMovement {
                    initial_rank: 1,
                    refined_rank: 0,
                    delta: -1,
                    reason: EXPLANATION_CANARY.to_owned(),
                }),
            }));
            observe_lexical_outcome(
                context(engine),
                Ok(vec![hit]),
                &LexicalObservationSupplement {
                    total_count: LexicalCountState::NotRequested,
                    hits: BTreeMap::from([(
                        "public-doc-id".to_owned(),
                        LexicalHitSupplement {
                            snippet: SensitiveValueObservation::from_text(snippet),
                            highlight_spans: LexicalObserved::Absent,
                        },
                    )]),
                },
            )
            .expect("redacted observation")
        };
        let subject = observation("quill", SNIPPET_CANARY);
        let oracle = observation("tantivy", "different-private-snippet");
        let report =
            compare_lexical_observations(subject, oracle).expect("redacted mismatch comparison");
        let artifact = serde_json::to_string(&report).expect("serialize redacted report");

        for canary in [
            QUERY_CANARY,
            NORMALIZED_CANARY,
            METADATA_CANARY,
            SNIPPET_CANARY,
            EXPLANATION_CANARY,
            "different-private-snippet",
        ] {
            assert!(
                !artifact.contains(canary),
                "sensitive canary escaped into artifact"
            );
        }
        assert!(artifact.contains(&sha256_hex(QUERY_CANARY.as_bytes())));
        assert!(artifact.contains(&sha256_hex(SNIPPET_CANARY.as_bytes())));
        assert!(report.mismatches.iter().all(|mismatch| {
            mismatch.oracle.chars().count() <= 193 && mismatch.subject.chars().count() <= 193
        }));

        let error = observe_lexical_outcome(
            context("quill"),
            Err(SearchError::QueryParseError {
                query: QUERY_CANARY.to_owned(),
                detail: METADATA_CANARY.to_owned(),
            }),
            &LexicalObservationSupplement::default(),
        )
        .expect("redacted error observation");
        let error_artifact = serde_json::to_string(&error).expect("serialize redacted error");
        assert!(!error_artifact.contains(QUERY_CANARY));
        assert!(!error_artifact.contains(METADATA_CANARY));
    }

    #[test]
    fn lexical_observation_shared_comparator_distinguishes_absent_from_empty_metadata() {
        let context = |engine: &str| {
            LexicalObservationContext::new(
                LexicalBoundary::FullSearch,
                LexicalBackendIdentity {
                    engine: engine.to_owned(),
                    revision: "test-revision".to_owned(),
                    index_identity: "in-memory".to_owned(),
                },
                "a".repeat(64),
                "b".repeat(64),
                "rust",
                42,
                10,
                LexicalExposureContract::CORE_LEXICAL_SEARCH,
            )
            .expect("valid lexical observation context")
        };
        let scored = |metadata| ScoredResult {
            doc_id: "doc-1".into(),
            score: 3.5,
            source: ScoreSource::Lexical,
            index: None,
            fast_score: None,
            quality_score: None,
            lexical_score: Some(3.5),
            rerank_score: None,
            explanation: None,
            metadata,
        };
        let subject = observe_lexical_outcome(
            context("quill"),
            Ok(vec![scored(None)]),
            &LexicalObservationSupplement::default(),
        )
        .expect("subject observation");
        let oracle = observe_lexical_outcome(
            context("tantivy"),
            Ok(vec![scored(Some(Arc::new(serde_json::json!({}))))]),
            &LexicalObservationSupplement::default(),
        )
        .expect("oracle observation");

        let report =
            compare_lexical_observations(subject, oracle).expect("shared lexical comparison");

        assert_eq!(report.status, LexicalComparisonStatus::Mismatch);
        assert_eq!(
            report
                .first_mismatch
                .as_ref()
                .map(|item| item.path.as_str()),
            Some("/outcome/hits/0/metadata")
        );
    }

    #[cfg(feature = "tantivy-oracle")]
    #[test]
    fn lexical_observation_real_quill_tantivy_public_boundary_is_exact() {
        use frankensearch_core::{IndexableDocument, LexicalSearch};
        use frankensearch_lexical::TantivyIndex;
        use frankensearch_quill::{QuillConfig, QuillIndex};

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let documents = vec![
                IndexableDocument::new("doc-none", "rust rust systems programming"),
                IndexableDocument::new("doc-metadata", "rust search")
                    .with_metadata("language", "rust")
                    .with_metadata("stable", "true"),
                IndexableDocument::new("doc-other", "unrelated database material"),
            ];
            let corpus_sha256 =
                sha256_hex(&canonical_json_bytes(&documents).expect("canonical corpus"));
            let quill = QuillIndex::in_memory(QuillConfig {
                deterministic_ingest: true,
                ..QuillConfig::default()
            })
            .expect("create Quill subject");
            let tantivy =
                TantivyIndex::in_memory_single_threaded_oracle().expect("create Tantivy oracle");

            LexicalSearch::index_documents(&quill, &cx, &documents)
                .await
                .expect("index Quill corpus");
            LexicalSearch::index_documents(&tantivy, &cx, &documents)
                .await
                .expect("index Tantivy corpus");
            LexicalSearch::commit(&quill, &cx)
                .await
                .expect("commit Quill corpus");
            LexicalSearch::commit(&tantivy, &cx)
                .await
                .expect("commit Tantivy corpus");

            let query = "rust";
            let quill_results = LexicalSearch::search(&quill, &cx, query, 10)
                .await
                .expect("search Quill");
            let tantivy_results = LexicalSearch::search(&tantivy, &cx, query, 10)
                .await
                .expect("search Tantivy");
            for results in [&quill_results, &tantivy_results] {
                assert_eq!(results.len(), 2);
                assert!(
                    results
                        .iter()
                        .find(|result| result.doc_id.as_str() == "doc-none")
                        .is_some_and(|result| result.metadata.is_none()),
                    "empty metadata must remain absent at the public boundary"
                );
                assert!(
                    results
                        .iter()
                        .find(|result| result.doc_id.as_str() == "doc-metadata")
                        .is_some_and(|result| result.metadata.is_some()),
                    "populated metadata must remain observable"
                );
            }

            let context = |engine: &str, revision: &str| {
                LexicalObservationContext::new(
                    LexicalBoundary::FullSearch,
                    LexicalBackendIdentity {
                        engine: engine.to_owned(),
                        revision: revision.to_owned(),
                        index_identity: "in-memory".to_owned(),
                    },
                    corpus_sha256.clone(),
                    "b".repeat(64),
                    query,
                    0x51_7e_a2,
                    10,
                    LexicalExposureContract::CORE_LEXICAL_SEARCH,
                )
                .expect("real backend observation context")
            };
            let subject = observe_lexical_outcome(
                context("quill", env!("CARGO_PKG_VERSION")),
                Ok(quill_results),
                &LexicalObservationSupplement::default(),
            )
            .expect("observe real Quill result envelope");
            let oracle = observe_lexical_outcome(
                context("tantivy", env!("CARGO_PKG_VERSION")),
                Ok(tantivy_results),
                &LexicalObservationSupplement::default(),
            )
            .expect("observe real Tantivy result envelope");

            let report = compare_lexical_observations(subject, oracle)
                .expect("compare real public lexical envelopes");
            assert_eq!(
                report.status,
                LexicalComparisonStatus::Equivalent,
                "real backend mismatch: {:?}",
                report.first_mismatch
            );
        });
    }
}
