use std::collections::{BTreeMap, BTreeSet};

use frankensearch_core::{QueryClass, ScoreSource, ScoredResult, SearchError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::GauntletError;

pub const SCORE_EPSILON: f32 = 0.0001;
/// Stable schema identifier for the complete public lexical result envelope.
pub const LEXICAL_OBSERVATION_SCHEMA_VERSION: &str = "lexical-observation-v1";
/// Maximum number of hits admitted into one lexical observation artifact.
pub const MAX_LEXICAL_OBSERVATION_HITS: usize = 100_000;
/// Maximum UTF-8 byte length of a consumer-visible document identifier.
pub const MAX_LEXICAL_DOC_ID_BYTES: usize = 1_024;
/// Maximum canonical byte length represented by one redacted payload digest.
pub const MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES: usize = 16 * 1_024 * 1_024;
/// Maximum number of ordered highlight spans represented for one hit.
pub const MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT: usize = 4_096;

/// Backend identity retained as provenance but not treated as an equivalence field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct LexicalObservationContext {
    /// Observation schema identifier.
    pub schema_version: String,
    /// Engine and revision that emitted this observation.
    pub backend: LexicalBackendIdentity,
    /// SHA-256 of the exact indexed corpus.
    pub corpus_sha256: String,
    /// SHA-256 of the raw query bytes.
    pub query_sha256: String,
    /// Raw query byte length.
    pub query_bytes: usize,
    /// SHA-256 of the normalized query bytes.
    pub normalized_query_sha256: String,
    /// Normalized query byte length.
    pub normalized_query_bytes: usize,
    /// Query class used by adaptive retrieval.
    pub query_class: QueryClass,
    /// Deterministic case/generator seed.
    pub seed: u64,
    /// Requested top-k limit.
    pub limit: usize,
}

impl LexicalObservationContext {
    /// Build a CI-safe context without retaining plaintext query material.
    ///
    /// # Errors
    ///
    /// Rejects malformed corpus hashes and unbounded backend identity fields.
    pub fn new(
        backend: LexicalBackendIdentity,
        corpus_sha256: String,
        raw_query: &str,
        normalized_query: &str,
        query_class: QueryClass,
        seed: u64,
        limit: usize,
    ) -> Result<Self, GauntletError> {
        let context = Self {
            schema_version: LEXICAL_OBSERVATION_SCHEMA_VERSION.to_owned(),
            backend,
            corpus_sha256,
            query_sha256: sha256_hex(raw_query.as_bytes()),
            query_bytes: raw_query.len(),
            normalized_query_sha256: sha256_hex(normalized_query.as_bytes()),
            normalized_query_bytes: normalized_query.len(),
            query_class,
            seed,
            limit,
        };
        context.validate()?;
        Ok(context)
    }

    fn validate(&self) -> Result<(), GauntletError> {
        self.backend.validate()?;
        if self.schema_version != LEXICAL_OBSERVATION_SCHEMA_VERSION
            || !is_lower_sha256(&self.corpus_sha256)
            || !is_lower_sha256(&self.query_sha256)
            || !is_lower_sha256(&self.normalized_query_sha256)
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
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
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
#[serde(tag = "state", rename_all = "snake_case")]
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
        byte_len: usize,
    },
    /// The field was present and non-empty.
    Present {
        /// SHA-256 of the canonical payload bytes.
        sha256: String,
        /// Canonical payload byte length.
        byte_len: usize,
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
        let byte_len = bytes.len();
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
                *byte_len <= MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES && is_lower_sha256(sha256)
            }
        }
    }
}

/// Half-open highlight span in UTF-8 byte offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalHighlightSpan {
    /// Inclusive byte offset.
    pub start: usize,
    /// Exclusive byte offset.
    pub end: usize,
}

/// Optional fields supplied by a richer backend boundary for one hit.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
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
pub struct LexicalObservationSupplement {
    /// Exact count semantics for this page.
    pub total_count: LexicalCountState,
    /// Optional per-hit snippet/highlight observations keyed by document ID.
    pub hits: BTreeMap<String, LexicalHitSupplement>,
}

/// Exhaustive public observation of one lexical hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalHitObservation {
    /// Zero-based native result rank.
    pub rank: usize,
    /// Stable external document identifier.
    pub doc_id: String,
    /// Public normalized/final score bits.
    pub normalized_score_bits: u32,
    /// Raw lexical score bits, when the public result exposes them.
    pub raw_lexical_score_bits: Option<u32>,
    /// Public source identity.
    pub source: ScoreSource,
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
pub struct LexicalErrorObservation {
    /// Stable broad error class.
    pub class: LexicalErrorClass,
    /// Stable variant-level code.
    pub code: String,
    /// Hash and length of the complete display diagnostic.
    pub detail: SensitiveValueObservation,
}

/// Successful or failed public lexical outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LexicalObservationOutcome {
    /// Successful ordered result page.
    Success {
        /// Results in native consumer-visible order.
        hits: Vec<LexicalHitObservation>,
        /// Returned vector length, retained independently for schema checks.
        returned_count: usize,
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
pub struct LexicalObservation {
    /// Query/corpus/backend provenance.
    pub context: LexicalObservationContext,
    /// Successful result envelope or typed error.
    pub outcome: LexicalObservationOutcome,
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
    /// Error class and stable variant code compare exactly.
    TypedErrorExact,
}

const LEXICAL_EQUIVALENCE_LAWS: [LexicalEquivalenceLaw; 6] = [
    LexicalEquivalenceLaw::NativeOrderExact,
    LexicalEquivalenceLaw::ScoreBitsExact,
    LexicalEquivalenceLaw::PresenceExact,
    LexicalEquivalenceLaw::SensitivePayloadDigest,
    LexicalEquivalenceLaw::CountAndEmptyShapeExact,
    LexicalEquivalenceLaw::TypedErrorExact,
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
    let outcome = match outcome {
        Ok(results) => {
            let mut observed_ids = BTreeSet::new();
            let mut hits = Vec::with_capacity(results.len());
            for (rank, result) in results.into_iter().enumerate() {
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
                let hit_supplement = supplement.hits.get(&doc_id).cloned().unwrap_or_default();
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
                    source,
                    index,
                    fast_score_bits: fast_score.map(f32::to_bits),
                    quality_score_bits: quality_score.map(f32::to_bits),
                    rerank_score_bits: rerank_score.map(f32::to_bits),
                    metadata,
                    explanation,
                    snippet: hit_supplement.snippet,
                    highlight_spans: hit_supplement.highlight_spans,
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
            let returned_count = hits.len();
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
            LexicalObservationOutcome::Error(observe_search_error(&error))
        }
    };
    let observation = LexicalObservation { context, outcome };
    validate_lexical_observation(&observation)?;
    Ok(observation)
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
    validate_lexical_observation(&subject)?;
    validate_lexical_observation(&oracle)?;
    let mut mismatches = Vec::new();

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
    compare_safe_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/normalized_query_sha256",
        &oracle.context.normalized_query_sha256,
        &subject.context.normalized_query_sha256,
    );
    compare_debug_field(
        &mut mismatches,
        LexicalMismatchClass::Context,
        "/context/normalized_query_bytes",
        &oracle.context.normalized_query_bytes,
        &subject.context.normalized_query_bytes,
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
    compare_lexical_outcomes(&subject.outcome, &oracle.outcome, &mut mismatches);

    let status = if mismatches.is_empty() {
        LexicalComparisonStatus::Equivalent
    } else {
        LexicalComparisonStatus::Mismatch
    };
    let first_mismatch = mismatches.first().cloned();
    if let Some(first) = &first_mismatch {
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
        "/outcome/error/detail",
        &oracle.detail,
        &subject.detail,
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
            if *returned_count != hits.len()
                || hits.len() > MAX_LEXICAL_OBSERVATION_HITS
                || hits.len() > observation.context.limit
                || (*empty_shape == LexicalEmptyShape::Empty) != hits.is_empty()
                || matches!(total_count, LexicalCountState::Value(total) if *total < u64::try_from(hits.len()).unwrap_or(u64::MAX))
            {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical returned-count, empty-shape, or total-count evidence is inconsistent"
                        .to_owned(),
                });
            }
            let mut doc_ids = BTreeSet::new();
            for (expected_rank, hit) in hits.iter().enumerate() {
                if hit.rank != expected_rank
                    || hit.doc_id.is_empty()
                    || hit.doc_id.len() > MAX_LEXICAL_DOC_ID_BYTES
                    || !doc_ids.insert(hit.doc_id.as_str())
                    || !float_bits_are_finite(hit.normalized_score_bits)
                    || !optional_float_bits_are_finite(hit.raw_lexical_score_bits)
                    || !optional_float_bits_are_finite(hit.fast_score_bits)
                    || !optional_float_bits_are_finite(hit.quality_score_bits)
                    || !optional_float_bits_are_finite(hit.rerank_score_bits)
                    || !hit.metadata.validate()
                    || !hit.explanation.validate()
                    || !hit.snippet.validate()
                    || !valid_highlight_state(&hit.highlight_spans)
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
            if error.code.is_empty() || error.code.len() > 128 || !error.detail.validate() {
                return Err(GauntletError::InvalidObservation {
                    reason: "lexical error code or redacted detail is invalid".to_owned(),
                });
            }
        }
    }
    Ok(())
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

fn observe_search_error(error: &SearchError) -> LexicalErrorObservation {
    let detail = SensitiveValueObservation::from_text(&error.to_string());
    let (class, code) = match error {
        SearchError::EmbedderUnavailable {
            model: _,
            reason: _,
        } => (
            LexicalErrorClass::Embedding,
            "embedder_unavailable".to_owned(),
        ),
        SearchError::EmbeddingFailed {
            model: _,
            source: _,
        } => (LexicalErrorClass::Embedding, "embedding_failed".to_owned()),
        SearchError::ModelNotFound { name: _ } => {
            (LexicalErrorClass::Embedding, "model_not_found".to_owned())
        }
        SearchError::ModelLoadFailed { path: _, source: _ } => {
            (LexicalErrorClass::Embedding, "model_load_failed".to_owned())
        }
        SearchError::UnverifiableRemoteSpace { producer: _, reason: _ } => (
            LexicalErrorClass::Integrity,
            "unverifiable_remote_space".to_owned(),
        ),
        SearchError::IndexCorrupted { path: _, detail: _ } => {
            (LexicalErrorClass::Index, "index_corrupted".to_owned())
        }
        SearchError::IndexVersionMismatch {
            expected: _,
            found: _,
        } => (
            LexicalErrorClass::Index,
            "index_version_mismatch".to_owned(),
        ),
        SearchError::DimensionMismatch {
            expected: _,
            found: _,
        } => (LexicalErrorClass::Index, "dimension_mismatch".to_owned()),
        SearchError::IndexNotFound { path: _ } => {
            (LexicalErrorClass::Index, "index_not_found".to_owned())
        }
        SearchError::IndexCandidatesNotFound { paths: _ } => (
            LexicalErrorClass::Index,
            "index_candidates_not_found".to_owned(),
        ),
        SearchError::QueryParseError {
            query: _,
            detail: _,
        } => (LexicalErrorClass::Query, "query_parse_error".to_owned()),
        SearchError::SearchTimeout {
            elapsed_ms: _,
            budget_ms: _,
        } => (LexicalErrorClass::Timeout, "search_timeout".to_owned()),
        SearchError::FederatedInsufficientResponses {
            required: _,
            received: _,
        } => (
            LexicalErrorClass::Federated,
            "federated_insufficient_responses".to_owned(),
        ),
        SearchError::RerankerUnavailable { model: _ } => {
            (LexicalErrorClass::Rerank, "reranker_unavailable".to_owned())
        }
        SearchError::RerankFailed {
            model: _,
            source: _,
        } => (LexicalErrorClass::Rerank, "rerank_failed".to_owned()),
        SearchError::Io(_) => (LexicalErrorClass::Io, "io".to_owned()),
        SearchError::InvalidConfig {
            field: _,
            value: _,
            reason: _,
        } => (
            LexicalErrorClass::Configuration,
            "invalid_config".to_owned(),
        ),
        SearchError::HashMismatch {
            path: _,
            expected: _,
            actual: _,
        } => (LexicalErrorClass::Integrity, "hash_mismatch".to_owned()),
        SearchError::Cancelled {
            phase: _,
            reason: _,
        } => (LexicalErrorClass::Cancellation, "cancelled".to_owned()),
        SearchError::QueueFull {
            pending: _,
            capacity: _,
        } => (LexicalErrorClass::Capacity, "queue_full".to_owned()),
        SearchError::SubsystemError {
            subsystem,
            source: _,
        } => (
            LexicalErrorClass::Subsystem,
            format!("subsystem.{subsystem}"),
        ),
        SearchError::DurabilityDisabled => (
            LexicalErrorClass::FeatureDisabled,
            "durability_disabled".to_owned(),
        ),
    };
    LexicalErrorObservation {
        class,
        code,
        detail,
    }
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
    let mut output = String::with_capacity(64);
    for byte in digest {
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
    use std::sync::Arc;

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
            LexicalBackendIdentity {
                engine: engine.to_owned(),
                revision: format!("{engine}-test-revision"),
                index_identity: "in-memory".to_owned(),
            },
            "a".repeat(64),
            " Rust  search ",
            "rust search",
            QueryClass::ShortKeyword,
            42,
            10,
        )
        .expect("valid lexical observation context")
    }

    fn lexical_result(doc_id: &str, score: f32) -> ScoredResult {
        use frankensearch_core::{ExplanationPhase, HitExplanation};

        ScoredResult {
            doc_id: doc_id.into(),
            score,
            source: ScoreSource::Lexical,
            index: Some(7),
            fast_score: Some(0.25),
            quality_score: Some(0.5),
            lexical_score: Some(score),
            rerank_score: Some(0.75),
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
            lexical_context(engine),
            Ok(vec![lexical_result("doc-1", 3.5)]),
            &lexical_supplement("doc-1"),
        )
        .expect("complete lexical observation")
    }

    fn lexical_success_mut(
        observation: &mut LexicalObservation,
    ) -> Option<(
        &mut Vec<LexicalHitObservation>,
        &mut usize,
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
        lexical_success_mut(observation)
            .expect("test fixture must be a successful lexical observation")
            .0
            .first_mut()
            .expect("test fixture must contain one lexical hit")
    }

    fn lexical_error_mut(
        observation: &mut LexicalObservation,
    ) -> Option<&mut LexicalErrorObservation> {
        match &mut observation.outcome {
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
        assert_eq!(
            report.applied_laws,
            vec![
                LexicalEquivalenceLaw::NativeOrderExact,
                LexicalEquivalenceLaw::ScoreBitsExact,
                LexicalEquivalenceLaw::PresenceExact,
                LexicalEquivalenceLaw::SensitivePayloadDigest,
                LexicalEquivalenceLaw::CountAndEmptyShapeExact,
                LexicalEquivalenceLaw::TypedErrorExact,
            ]
        );
        assert_eq!(report.subject.context.backend.engine, "quill");
        assert_eq!(report.oracle.context.backend.engine, "tantivy");
    }

    #[test]
    fn lexical_observation_context_field_mutations_are_detected() {
        assert_single_lexical_mismatch(
            |observation| observation.context.corpus_sha256 = "b".repeat(64),
            LexicalMismatchClass::Context,
            "/context/corpus_sha256",
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
            |observation| observation.context.normalized_query_sha256 = "c".repeat(64),
            LexicalMismatchClass::Context,
            "/context/normalized_query_sha256",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.normalized_query_bytes += 1,
            LexicalMismatchClass::Context,
            "/context/normalized_query_bytes",
        );
        assert_single_lexical_mismatch(
            |observation| observation.context.query_class = QueryClass::NaturalLanguage,
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
    fn lexical_observation_hit_field_mutations_are_detected() {
        assert_single_lexical_mismatch(
            |observation| lexical_hit_mut(observation).doc_id = "doc-2".to_owned(),
            LexicalMismatchClass::Ordering,
            "/outcome/hits/0/doc_id",
        );
        assert_single_lexical_mismatch(
            |observation| {
                let hit = lexical_hit_mut(observation);
                hit.normalized_score_bits = f32::from_bits(hit.normalized_score_bits + 1).to_bits();
            },
            LexicalMismatchClass::Score,
            "/outcome/hits/0/normalized_score_bits",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).raw_lexical_score_bits = Some(4.0_f32.to_bits());
            },
            LexicalMismatchClass::Score,
            "/outcome/hits/0/raw_lexical_score_bits",
        );
        assert_single_lexical_mismatch(
            |observation| lexical_hit_mut(observation).source = ScoreSource::Hybrid,
            LexicalMismatchClass::SourceIdentity,
            "/outcome/hits/0/source",
        );
        assert_single_lexical_mismatch(
            |observation| lexical_hit_mut(observation).index = Some(8),
            LexicalMismatchClass::SourceIdentity,
            "/outcome/hits/0/index",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).fast_score_bits = Some(0.26_f32.to_bits());
            },
            LexicalMismatchClass::Score,
            "/outcome/hits/0/fast_score_bits",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).quality_score_bits = Some(0.51_f32.to_bits());
            },
            LexicalMismatchClass::Score,
            "/outcome/hits/0/quality_score_bits",
        );
        assert_single_lexical_mismatch(
            |observation| {
                lexical_hit_mut(observation).rerank_score_bits = Some(0.76_f32.to_bits());
            },
            LexicalMismatchClass::Score,
            "/outcome/hits/0/rerank_score_bits",
        );
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
                lexical_hit_mut(observation).snippet = SensitiveValueObservation::Absent;
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
                    byte_len: MAX_LEXICAL_SENSITIVE_PAYLOAD_BYTES + 1,
                };
            },
            |observation: &mut LexicalObservation| {
                lexical_hit_mut(observation).highlight_spans = LexicalObserved::Value(
                    (0..=MAX_LEXICAL_HIGHLIGHT_SPANS_PER_HIT)
                        .map(|index| LexicalHighlightSpan {
                            start: index * 2,
                            end: index * 2 + 1,
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
                lexical_context(engine),
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

        assert_single_lexical_mismatch(
            |observation| lexical_hit_mut(observation).normalized_score_bits += 1,
            LexicalMismatchClass::Score,
            "/outcome/hits/0/normalized_score_bits",
        );
    }

    #[test]
    fn lexical_observation_preserves_empty_and_typed_error_shapes() {
        let empty = observe_lexical_outcome(
            lexical_context("quill"),
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
                lexical_context(engine),
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
            |error| error.detail = SensitiveValueObservation::from_text("changed"),
            "/outcome/error/detail",
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
    fn lexical_observation_artifacts_redact_sensitive_values() {
        const QUERY_CANARY: &str = "sensitive-query-canary-9f3a";
        const NORMALIZED_CANARY: &str = "sensitive-normalized-canary-2e7b";
        const METADATA_CANARY: &str = "sensitive-metadata-canary-36c1";
        const SNIPPET_CANARY: &str = "sensitive-snippet-canary-73d4";

        let context = |engine: &str| {
            LexicalObservationContext::new(
                LexicalBackendIdentity {
                    engine: engine.to_owned(),
                    revision: "redaction-test".to_owned(),
                    index_identity: "in-memory".to_owned(),
                },
                "d".repeat(64),
                QUERY_CANARY,
                NORMALIZED_CANARY,
                QueryClass::Identifier,
                91,
                1,
            )
            .expect("redaction context")
        };
        let observation = |engine: &str, snippet: &str| {
            let mut hit = lexical_result("public-doc-id", 1.0);
            hit.metadata = Some(Arc::new(serde_json::json!({
                "private": METADATA_CANARY
            })));
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
                LexicalBackendIdentity {
                    engine: engine.to_owned(),
                    revision: "test-revision".to_owned(),
                    index_identity: "in-memory".to_owned(),
                },
                "a".repeat(64),
                "rust",
                "rust",
                QueryClass::ShortKeyword,
                42,
                10,
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
                    LexicalBackendIdentity {
                        engine: engine.to_owned(),
                        revision: revision.to_owned(),
                        index_identity: "in-memory".to_owned(),
                    },
                    corpus_sha256.clone(),
                    query,
                    query,
                    QueryClass::classify(query),
                    0x51_7e_a2,
                    10,
                )
                .expect("real backend observation context")
            };
            let quill_count = u64::try_from(quill_results.len()).expect("small Quill result count");
            let tantivy_count =
                u64::try_from(tantivy_results.len()).expect("small Tantivy result count");
            let subject = observe_lexical_outcome(
                context("quill", env!("CARGO_PKG_VERSION")),
                Ok(quill_results),
                &LexicalObservationSupplement {
                    total_count: LexicalCountState::Value(quill_count),
                    hits: BTreeMap::new(),
                },
            )
            .expect("observe real Quill result envelope");
            let oracle = observe_lexical_outcome(
                context("tantivy", env!("CARGO_PKG_VERSION")),
                Ok(tantivy_results),
                &LexicalObservationSupplement {
                    total_count: LexicalCountState::Value(tantivy_count),
                    hits: BTreeMap::new(),
                },
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
